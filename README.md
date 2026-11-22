# Fraud Detection: from pilot notebook to production-style MLOps

[![CI](https://github.com/devienne/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/devienne/fraud-detection-mlops/actions/workflows/ci.yml)

A machine learning system that classifies card transactions as fraudulent or legitimate. It routes each transaction to **approve**, **manual review** or **block**.

The project started as an exploratory notebook, the **pilot**. It is being extended step by step into a production-style service with:
- versioned data and models;
- a reproducible pipeline;
- an API, tests and CI;
- monitoring;
- safe retraining.

See the [roadmap](#roadmap) for the milestones.

## The problem

A bank's fraud-prevention team reviews suspicious transactions mostly by hand. As volume grows, that process does not scale, and it creates two risks:

- **Missed fraud**, which means financial loss.
- **Wrongly blocked customers**, which hurts the customer experience.

The goal is to learn from the labelled 2025 history and classify every transaction of January 2026.

## Pilot results

The model was evaluated on December 2025, a month held out from training.

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| **LightGBM** (selected) | **0.62** | **0.94** |
| CatBoost | 0.61 | 0.92 |
| Logistic regression | 0.46 | 0.85 |
| Rule baseline: fraud rate per (merchant type, hour) | 0.31 | 0.86 |

- **Cost-based decisions.** The block threshold of 0.28 minimises missed-fraud amount plus an assumed R$ 10 per false block.
  It catches about 63% of fraud with about 59% precision, and 91% of transactions are approved automatically.
- **Key finding.** All fraud occurs in a few (merchant type, hour) windows. Inside them, risk rises with a new device, mobile, São Paulo or Rio, and R$ 20–45 amounts.
  The internal score `score_interno` carries no signal.
- **Drift.** In December, e-commerce fraud at night jumped from 15% to 60%. Retraining with recent data weighted 3× lifted late-December PR-AUC from 0.63 to 0.71.

The full analysis is in [notebooks/01_pilot_analysis.ipynb](notebooks/01_pilot_analysis.ipynb).

## Data provenance

| | |
|---|---|
| Source | Synthetic, Portuguese-language card-transaction dataset |
| Files | `data/raw/transacoes_historico_2025.csv`: 523,525 labelled transactions from 2025 (`flag_fraude`) |
| | `data/raw/transacoes_jan2026.csv`: 44,451 unlabelled transactions from January 2026 |
| Nature | The data appears **synthetic**. Fraud is exactly zero outside fixed hour windows, and there is exactly one transaction per minute. |
| Versioning | Tracked with [DVC](https://dvc.org). Git stores only the `.dvc` pointer files with each file's hash. The CSVs live in DVC storage, so they are not in this repository. |

## Repository layout

```
├── params.yaml               # every tunable setting (splits, model, policy, MLflow)
├── dvc.yaml / dvc.lock       # pipeline definition / exact hashes of the last run
├── src/fraud_detection/
│   ├── config.py             # loads and validates params.yaml
│   ├── data.py               # data contract (pandera) and loaders
│   ├── features.py           # make_features(): shared by training, batch scoring and serving
│   ├── train.py              # evaluation model (early stopping) + final model
│   ├── evaluate.py           # thresholds on validation, metrics on test, model metadata
│   ├── policy.py             # probability -> risk band / block decision
│   ├── tracking.py           # MLflow run logging and model registry
│   ├── predict.py            # batch scoring (FraudScorer, reused by the API)
│   └── monitoring.py         # drift (PSI), segments, performance on matured labels, alerts, heartbeat
├── api/
│   ├── schemas.py            # request/response contract (Pydantic)
│   ├── metrics.py            # Prometheus model metrics (scores, decisions, unseen values)
│   └── main.py               # FastAPI service: /health, /model-info, /predict, /predict/batch, /metrics
├── Dockerfile                # API image (slim, non-root, health check)
├── docker-compose.yml        # local stack: API + Prometheus + Grafana + MLflow UI
├── monitoring/               # Prometheus config, Grafana datasource + dashboard (as code)
├── docs/monitoring.md        # runbook: what each alert means and what to do
├── requirements-api.txt      # pinned runtime dependencies of the API image
├── scripts/
│   ├── load_test.py          # latency percentiles and throughput
│   └── drift_drill.py        # replays Nov/Dec 2025: would the monitor have caught the drift?
├── data/raw/                 # raw CSVs (DVC-tracked; pointers in git)
├── data/predictions/         # batch predictions (DVC output)
├── models/                   # model files, policy, metadata (DVC outputs)
├── reports/                  # data validation, metrics, prediction summary (in git)
├── notebooks/
│   └── 01_pilot_analysis.ipynb
└── requirements.txt          # pinned dependencies
```

## Pipeline

```
validate ──> train ──> evaluate ──> predict ──> reference ──> monitor
data         eval model  thresholds   January    training      January vs
contract     + final     + metrics    classified distributions reference
             model       + MLflow                + baseline    (drift check)
```

| Stage | Uses | Produces |
|---|---|---|
| `validate` | raw CSVs | `reports/data_validation.json` (fails on any contract violation) |
| `train` | Jan–Sep (early stopping on Oct–Nov); then all 2025 | `models/eval_model.txt`, `models/final_model.txt`, `models/training_info.json` |
| `evaluate` | Oct–Nov to choose thresholds; December to report | `models/policy.json`, `models/metadata.json`, `reports/metrics.json`, an MLflow run and registered model version |
| `predict` | final model + policy | `data/predictions/transacoes_jan2026_classified.csv`, `reports/prediction_summary.json` |
| `reference` | final model on all 2025 data; evaluation model on December as the honest performance baseline | `models/reference.json` |
| `monitor` | January batch vs reference | `reports/monitoring_jan2026.json` |

## How to run

Requirements: Python 3.10 and git.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .                  # install the fraud_detection package

dvc pull                          # restore data (and cached models) from DVC storage
dvc repro                         # run the pipeline; unchanged stages are skipped
dvc metrics show                  # headline metrics
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs at http://127.0.0.1:5000
```

The DVC storage is currently a local folder configured per machine in `.dvc/config.local`, so `dvc pull` only works on the author's machine.
A public DVC remote is on the roadmap.

## Serving: the fraud API

The API scores one transaction, or a batch of up to 1,000, with the final model. It returns:
- the fraud probability;
- the risk band;
- the decision (block, review or approve);
- the model version.

Every prediction is logged to `logs/predictions.jsonl` for monitoring.

```bash
# locally (after `dvc repro` or `dvc pull`)
uvicorn api.main:app --port 8000          # interactive docs: http://127.0.0.1:8000/docs

# in Docker
docker build -t fraud-api:latest .
docker run -p 8000:8000 fraud-api:latest
```

Example request and response:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "transaction_id": "tx-000123", "datahora_transacao": "2026-01-15T22:14:00",
  "nome_estabelecimento": "Online Shop 1", "tipo_estabelecimento": "E-commerce",
  "valor_transacao": 32.90, "cidade": "São Paulo", "device_type": "mobile", "is_new_device": true}'
```
```json
{"transaction_id": "tx-000123", "fraud_probability": 0.796063, "risk_band": "High",
 "decision": "block", "model_version": "20261121-623d6d0"}
```

**The input contract** rejects anything the model was not trained on, returning HTTP 422 with the reason. Examples:
- amounts that are not positive numbers in reais;
- `is_new_device` values other than `true` or `false`;
- unknown merchant or device types.

Timestamps with a time zone, such as UTC, are converted to Brazil time, which is how the model was trained. New merchants and cities are accepted and scored as unknown.

**Latency**, measured with `python scripts/load_test.py`, Docker, 1 worker, 1,000 real January transactions:

| Mode | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|
| Single requests, 1 client | 19 ms | 27 ms | 48 ms | 48 tx/s |
| Single requests, 8 concurrent clients | 153 ms | 234 ms | 307 ms | 51 tx/s |
| Batches of 100, 1 client | 26 ms per batch | 30 ms | 30 ms | 3,700 tx/s |

A single check is well inside a real-time budget such as 700 ms.

Under concurrent load, requests queue behind one worker, and most of the cost is per-request overhead, not the model. The next levers would be:
- more workers or replicas;
- building single-row features without pandas.

## Testing & CI

```bash
pip install -r requirements-dev.txt
pytest                 # all tests (model tests run only where the DVC data/model exist)
pytest -m "not model"  # code tests only, as on CI
pre-commit install     # run the checks automatically on every `git commit`
```

| Test file | What it protects |
|---|---|
| `tests/test_features.py` | **Train/serve parity.** Training and the API build identical features. Category codes never depend on the batch. Unseen values become missing. Booleans are strict. UTC is converted. |
| `tests/test_data_contract.py` | **The data contract** catches each kind of bad data, plus implausible fraud rates. **Consistency:** the training contract (pandera) and the API contract (Pydantic) accept and reject exactly the same values. |
| `tests/test_api.py` | **Service behaviour,** using a tiny fixture model in production format: health, model info, consistent decisions, 422 on invalid input, batch limits, API scores equal batch scores, every prediction logged, and refusal to start without a model. |
| `tests/test_monitoring.py`, `tests/test_monitoring_job.py` | **Monitoring.** PSI maths, fixed reference bins, drift flagged or not flagged as expected. Concept drift (same inputs, fraud moved) is invisible to PSI but caught by labels. Alerts are grouped and actionable. Label delay is respected. Plus the prediction log format, the heartbeat and the command line. |
| `tests/test_model_quality.py` | **Model quality gate** (`model` marker). It checks the test month against `params.yaml` → `quality_gate`: PR-AUC ≥ 0.60, ≥ 1.5× the rule baseline, recall ≥ 50%, precision ≥ 40%. A model with no skill must be rejected. |

**pre-commit hooks** run before every commit:
- ruff, the linter;
- a block on files over 1 MB, such as data or models;
- YAML, JSON and TOML validity;
- whitespace checks.

**GitHub Actions** (`.github/workflows/ci.yml`) runs on every push:
1. **Lint and test:** pre-commit plus pytest. Model tests are skipped, because DVC data isn't available on CI.
2. **Docker:** build the image with a tiny test model, start it, wait until healthy, then smoke-test it. The test confirms a non-root user, a valid prediction, and HTTP 422 on invalid input.

Model lineage: `models/metadata.json` records the git commit the model was trained from. It's flagged `-dirty` only when *source* files were uncommitted, not the pipeline's own outputs.

## Monitoring

Monitoring works at three speeds. The runbook, [docs/monitoring.md](docs/monitoring.md), explains every alert and what to do about it.

| Speed | Tool | Watches |
|---|---|---|
| Seconds | `/metrics` → Prometheus → Grafana | Requests, errors, p50/p95/p99 latency; block and review rate (overall and per merchant type); score percentiles; unseen categories; live model version |
| Hourly | `python -m fraud_detection.monitoring run`, reading the prediction log | PSI per feature, score drift, missing values, decision rates per segment, alerts with actions, heartbeat |
| Once labels mature (30 days) | The same job, with labels joined | PR-AUC and recall at the block threshold, overall and per segment, against the model's evaluation baseline |

```bash
docker compose up -d --build     # API :8000, Prometheus :9090, Grafana :3000 (admin/admin), MLflow :5000
python scripts/load_test.py --url http://127.0.0.1:8000   # traffic for the dashboard
python -m fraud_detection.monitoring run                  # drift report -> logs/monitoring/monitoring.json
```

Grafana's "Fraud API: service and model" dashboard and its Prometheus connection are loaded from files in `monitoring/` (dashboards as code).

**Drift drill** (`python scripts/drift_drill.py` writes `reports/drift_drill.json`). It asks: if the model had been live in late 2025, would the monitor have caught December's fraud shift? It uses the evaluation model, trained on Jan–Sep only, with its October performance as the baseline, and replays November and December:

| Month | Max feature PSI | Score PSI | PR-AUC | Recall at block | Monitor |
|---|---|---|---|---|---|
| October (baseline) | | | 0.672 | 78% | |
| November | 0.0002 | 0.0002 | 0.659 | 79% | quiet (one noise-level warning) |
| December | 0.0004 | 0.0001 | 0.618 | 63% | **alert** |

December's alert, as the on-call engineer receives it:

> **ALERT:** Fraud pattern changed in tipo_estabelecimento=E-commerce: recall at block 69% → 35%; share of all fraud 12% → 39%. 14 related segment findings (likely the same cause).
> → Review recent fraud cases in this segment with the fraud team; consider a temporary rule for it and a retraining challenger with recent data up-weighted.

**What the drill shows:**
- **The drift was *concept* drift.** The same kinds of transactions kept arriving; only which of them were fraud changed. Input and score drift checks (PSI) stayed quiet, as they should. Only label-based checks per segment caught it.
- **Time to detect equals the label delay.** Replayed day by day with a 30-day label delay, the first alert fires on **1 January 2026**, 31 days after the shift began. Faster labels would detect it sooner: analyst review outcomes, customer disputes.
- **Grouped alerts.** One cause raised 16 segment findings, delivered as one alert with the rest attached.

## Roadmap

| Version | Milestone |
|---|---|
| `v0.1-pilot` | Pilot notebook, data versioned with DVC |
| `v0.2-pipeline` | Reproducible training pipeline (`src/`, `params.yaml`, `dvc repro`, MLflow) |
| `v0.3-api` | FastAPI service in Docker |
| `v0.4-ci` | Tests, quality gate and GitHub Actions |
| `v0.5-monitoring` | Prometheus, Grafana and drift monitoring |
| `v1.0` | Champion/challenger retraining, shadow mode, documentation |
