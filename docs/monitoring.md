# Monitoring runbook: fraud scoring service

For the on-call engineer: what is watched, what each alert means, and what to do.
Thresholds quoted here come from `params.yaml` → `monitoring` and from the Grafana dashboard. If you change one, update this page.

## 1. What watches what

```
                       real time (seconds)                      batch (hourly)                 late (≥ 30 days)
 client ──> fraud API ──> /metrics ──> Prometheus ──> Grafana
               │                                                                        fraud labels
               └──> logs/predictions.jsonl ──────> monitor job ──> logs/monitoring/ <──── (chargebacks,
                    (every prediction, full detail)  (python -m             monitoring.json    disputes)
                                                     fraud_detection.       heartbeat.json
                                                     monitoring run)
```

| Layer | Answers | Where |
|---|---|---|
| **Service metrics** | Is the API up, fast and error-free? | Grafana, row "Service" |
| **Model metrics, live** | Are the decisions sensible *right now*: block rate, score percentiles, unseen values? | Grafana, row "Model" |
| **Monitor job** | *What* changed? Per-feature drift, score drift, rates per segment | `logs/monitoring/monitoring.json` |
| **Performance on labels** | Is the model still *right*? Overall and per segment, once labels mature | same report, `performance` section |
| **Heartbeat** | Is the monitor itself still running? | `logs/monitoring/heartbeat.json` |

Prometheus uses only labels with a few fixed values: decision, risk band, merchant type, endpoint, feature.
Anything per transaction, merchant or city lives in the prediction log, and the monitor job reads it.

## 2. Alerts: meaning and response

Severity: **alert** means act now; **warn** means look today and watch the trend.
Every monitor-job alert carries its own `message` and `action`; the entries below expand on them.

### Service

| Signal | Threshold | Likely causes | Check | Do |
|---|---|---|---|---|
| **API down** (`up{job="fraud-api"} == 0`) | 1 min | Container crashed, model failed to load (the service refuses to start without one), host out of memory | `docker compose ps`, `docker compose logs api` | Restart. If the model fails to load, roll back to the previous model files. Until it's back, callers must fall back to their rules. |
| **5xx errors** | warn ≥ 0.1%, alert ≥ 1% (5 min) | Bug, a corrupt model file, the log disk full | API logs. Is `/health` OK? | Roll back the last release. Free disk space if the log write fails. |
| **4xx spike** (requests rejected by the input contract, HTTP 422) | warn ≥ 1%, alert ≥ 5% | A caller changed its payload: a field renamed, amounts sent in cents or as text, a new merchant type | The 422 response says which field. Compare with the caller's last release. | Contact the caller's team. Do **not** loosen the contract to make errors disappear: the model would score bad data silently. |
| **Latency** p99 on `/predict` | warn ≥ 300 ms, alert ≥ 700 ms (the real-time budget) | Traffic above one worker's capacity (~50 req/s), a slow log disk, CPU contention | "Model time vs request time" panel: is the model slow, or the overhead? | Add replicas. Scoring time is usually small (~25 ms); queueing is the usual cause. |

### Model, live (Grafana)

| Signal | Threshold | Likely causes | Check | Do |
|---|---|---|---|---|
| **Block rate** | warn ≥ 10%, alert ≥ 15% (5 min); January reference ≈ 6% | A fraud attack (real), upstream data corruption (false), a wrong policy file deployed | "Block rate by merchant type": one segment or all? Then the score percentiles, unseen categories and the model version in use | One segment and plausible: inform the fraud team. All segments at once with a new model version: roll back. Data issue: see input drift. |
| **Review rate** | warn ≥ 5%, alert ≥ 10%; January ≈ 2.5% | As above; also the analysts' workload | Same as above | Warn the analyst team about queue size. |
| **Unseen categories** per transaction | warn ≥ 2%, alert ≥ 10% | A new city or merchant launched; spelling or encoding changes upstream (for example "SÃ£o Paulo" instead of "São Paulo") | The monitor job's `top_moves` for `cidade` and `nome_estabelecimento` | Encoding bug: fix it upstream. Genuinely new values: they're scored as unknown, so include them in the next training. |
| **Score percentiles** move | No fixed threshold; compare with the reference | Same as input drift or block rate | p90 and p99 move first | Follow the input drift entry. |
| **Shadow errors** (`fraud_shadow_errors_total` rising) | Any | The challenger's bundle is incompatible, or it crashes on some inputs | API logs ("shadow scoring failed") | Customers are unaffected: the champion answers. Do not promote this challenger; fix and re-register it. |
| **Shadow disagreement** (`fraud_shadow_comparisons_total`) | Agreement well below what the approval report suggested | The challenger behaves differently on live traffic than on the holdout | `python -m fraud_detection.monitoring shadow`: which decisions move? | Pause the promotion and bring the agreement matrix to the approver. |
| **Wrong model after a release** | Version on the dashboard ≠ the registry `champion` | Forgot to restart after promote or rollback; wrong bundle exported | `python -m fraud_detection.promote status` vs `/model-info` | `docker compose restart api`. If the new champion misbehaves: `promote rollback --reason "<why>"`, then restart. |

### Monitor job (`logs/monitoring/monitoring.json`)

| Alert | Threshold | Meaning | Do |
|---|---|---|---|
| `input_drift` | PSI warn ≥ 0.10, alert ≥ 0.25 | A feature's distribution differs from training. The message names the bin that moved most. | First rule out a **pipeline** change upstream (units, defaults, joins). If the population really changed (new market, campaign), plan a retraining challenger. |
| `missing_values` | +1 percentage point vs reference | A field started arriving empty | Check the producer; the model treats missing as unknown. |
| `score_drift` | PSI warn ≥ 0.10, alert ≥ 0.25 | The model's outputs moved | Look for an input drift explaining it. If there is none, check that the deployed model and policy versions are the expected ones. |
| `performance` | PR-AUC or recall at block down ≥ 0.10 (warn at 0.05) vs the baseline | The model is less accurate on matured labels | Train a **challenger** on recent data and compare it with the champion (retraining pipeline). Re-check the thresholds; the policy is retrained with the model. |
| `segment_performance` | Recall down ≥ 0.10, or share of all fraud moved ≥ 0.15 | Fraud moved into, or out of, a segment. Related findings are grouped under the largest one: fix the root cause, not the 14 echoes. | Review recent fraud cases in that segment with the fraud team. Consider a **temporary rule** for that segment while a challenger is trained with recent data up-weighted. |
| `new_segment_value` | Any | A segment value never seen in the baseline | Confirm it is expected. |
| **Heartbeat stale** | No success for > 2 h | The monitor stopped: the scheduler is broken, the log is unreadable, or the job crashed. **No drift alert can fire while it's down.** | `python -m fraud_detection.monitoring heartbeat --max-age-hours 2`, then the scheduler logs. Run the job by hand to see the error. |

**Label delay.** Performance checks use only transactions older than `label_maturity_days` (30), because fraud is reported late. Newer data would look falsely clean.
The drift drill showed what that costs: December 2025's shift would have been detected on **1 January**, 31 days after it began.
Earlier, weaker labels (analyst review outcomes, customer disputes) shorten that delay.

## 3. How to

```bash
# Start / stop the local stack: API :8000, Prometheus :9090, Grafana :3000 (admin/admin), MLflow :5000
docker compose up -d --build
docker compose down

# Run the monitor on the API's prediction log (writes logs/monitoring/monitoring.json + heartbeat.json)
python -m fraud_detection.monitoring run
python -m fraud_detection.monitoring run --fail-on-alert      # exit code 2 on an alert, for schedulers/CI

# Join late labels (CSV with transaction_id and flag_fraude) to measure performance
python -m fraud_detection.monitoring run --labels labels.csv

# Replay any file of transactions as of a date (labels used if the file has flag_fraude)
python -m fraud_detection.monitoring run --window some_month.csv --as-of 2026-02-01 \
    --report logs/monitoring/replay.json --no-heartbeat

# Is the monitor alive?  exit 0 = yes, 1 = stale or never ran
python -m fraud_detection.monitoring heartbeat --max-age-hours 2

# Rebuild the reference for the released model (normally done by `dvc repro`, stage `reference`)
python -m fraud_detection.monitoring reference

# Rehearse the December 2025 incident
python scripts/drift_drill.py

# Releases (MLflow aliases; restart the API after promote / rollback)
python -m fraud_detection.promote status                          # champion, challenger, previous
python -m fraud_detection.monitoring shadow                       # challenger vs champion on live traffic
python -m fraud_detection.promote rollback --reason "block rate spike after release"
docker compose restart api
```

**Scheduling.** The live monitor is a recurring job, not a pipeline stage. Run it hourly:

```cron
# crontab: monitor every hour, heartbeat check every 15 minutes (alert on non-zero exit)
0 * * * *    cd /app && python -m fraud_detection.monitoring run --fail-on-alert
*/15 * * * * cd /app && python -m fraud_detection.monitoring heartbeat --max-age-hours 2
```

On Kubernetes the same commands become two `CronJob`s, and their failures feed the alerting system.
The pipeline's `monitor` stage (`dvc repro`) checks only the January batch against the reference, which makes it reproducible.

## 4. Known limits

- **Concept drift is invisible without labels.** PSI and score drift detect changes in the *inputs*. A change in *which* inputs are fraud, like December 2025, shows up only in label-based checks, after the label delay.
- **Prometheus counters reset** when the API restarts. Rate queries (`rate()`, `increase()`) handle this; raw counter values don't.
- **One worker per container.** Counters live in process memory. Several Uvicorn workers in one container would need `prometheus_client` multiprocess mode. Scale with replicas instead.
- **Dashboard thresholds are colours, not pages.** Prometheus alert rules and Alertmanager routing are not set up yet. The thresholds above are the intended rules.
- **MLflow UI in Docker** shows runs and the registry, but not artifact files: they were logged with Windows paths. A shared artifact store such as S3 or MinIO would fix this.
- **The reference** is built from the final model's own training data, so its score distribution is in-sample. The performance baseline uses the evaluation model on the test month, which keeps it out-of-time and honest.
