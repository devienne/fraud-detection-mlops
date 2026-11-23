# Model card: card-transaction fraud classifier

| | |
|---|---|
| Model | `fraud-lightgbm`, LightGBM binary classifier plus a decision policy (approve / review / block) |
| Released version | The registry alias `champion`; `/model-info` on the API shows the version being served |
| Owner | Repository author |
| Status | Portfolio project on synthetic data: **not for real payment decisions** |

## Intended use

- **What it does.** It scores a card transaction at authorisation time with a fraud probability, and turns that into an action:
  - **block** or step-up authentication: score ≥ 0.28;
  - **manual review**: score ≥ 0.153;
  - **approve**: otherwise.
- **Who uses it.** The payment authorisation flow, through the API, and the fraud operations team, through the review queue and the monitoring.
- **Out of scope.**
  - Decisions about customers beyond a single transaction, such as closing accounts or credit limits.
  - Other products or countries.
  - Use without the monitoring and retraining process described in [architecture.md](architecture.md).

## Data

| | |
|---|---|
| Source | A synthetic, Portuguese-language card-transaction dataset, versioned with DVC |
| Training data | 523,525 labelled transactions, 1 Jan to 31 Dec 2025, one per minute |
| Label | `flag_fraude`. Fraud rate 3.9% overall: 3.7–4.0% per month from January to November, **5.4% in December**. |
| Scoring data | 44,451 unlabelled transactions from January 2026 |
| Signs it is synthetic | Fraud is exactly zero outside fixed hour windows; one transaction per minute. Real fraud is messier, so results here are optimistic about how separable fraud is. |

**Features (7).**
- hour of day;
- merchant type, merchant name, city, device type (categorical);
- new device (yes/no);
- amount in R$.

**Deliberately excluded:**
- `score_interno`: no signal.
- Customer name and postal code: identifiers, a privacy risk, and an invitation to overfit.
- Temperature: irrelevant.
- The derived shift and amount buckets: redundant with hour and amount.

There are **no protected attributes** (sex, race, age) in the data. City and device can still act as proxies for income or region. See *Fairness*.

## Training

- **Time-based split.** Train on Jan–Sep, validate on Oct–Nov (early stopping, threshold choice), test on December. Nothing is randomly shuffled across time.
- **Evaluation model.** It stops after 234 trees, when validation PR-AUC stops improving.
- **Final model.**
  - Trained on all of 2025 with 280 trees (234 × 1.2).
  - **December weighted 3×**, because fraud patterns move.
  - Its quality is estimated by the evaluation model, because the final model has no held-out data.
- **Settings** (`params.yaml`): learning rate 0.03, 15 leaves, at least 500 rows per leaf, 80% row and column subsampling, L2 regularisation 1.0.
- **Reproducibility.** `dvc repro` rebuilds the model byte for byte. Each model records its git commit and data hash.

## Decision policy

- **Block threshold: the lowest business cost on validation.** Cost is the missed fraud amount plus an **assumed R$ 10 per blocked honest customer**. The R$ 10 is a business assumption, not a measured cost, and it drives the threshold.
- **Review threshold:** block and review together must catch 95% of validation fraud.
- **Retrained models get their own thresholds,** chosen by the same rules plus a customer-impact budget: at most the champion's honest-customer block rate + 0.5 pts.

## Performance

Test month December 2025, evaluation model, thresholds from Oct–Nov:

| Metric | Model | Rule baseline (fraud rate per merchant type × hour) |
|---|---|---|
| PR-AUC | **0.618** | 0.310 |
| ROC-AUC | 0.941 | 0.865 |
| Fraud caught by blocking (recall) | 63% | |
| Blocks that are fraud (precision) | 59% | |
| Transactions blocked / sent to review | 5.7% / 2.7% | |
| Fraud amount prevented | R$ 45,870 of R$ 73,434 | |

**Segment performance on the test month** (block threshold 0.28):

| Segment | Fraud rate | Fraud caught | Honest customers blocked |
|---|---|---|---|
| E-commerce | 10.2% | **35%** | 1.0% |
| Food & Beverage | 6.4% | 85% | **4.9%** |
| Fuel | 8.2% | 85% | **5.1%** |
| Pharmacy / Retail / Supermarket | 2.0–2.7% | 67–74% | 1.5–1.6% |
| Mobile / desktop / tablet | 6.8% / 4.0% / 2.6% | 68% / 52% / 41% | 3.4% / 1.6% / 0.7% |

**Weakest point: e-commerce at night.**
- In December, the share of fraud that happened in e-commerce jumped from 12% to 39% (concept drift), and the model catches only about a third of it.
- The December retrained challenger still blocks only 35% of e-commerce fraud, and **0% at 20:00–21:00**.
- A targeted rule or new features (velocity, device history) are needed there. A model refresh alone is not enough.

## Fairness

The model doesn't see protected attributes, but **honest customers are not affected equally**:

| City | Honest customers blocked (Dec 2025) |
|---|---|
| Rio de Janeiro | 3.9% |
| São Paulo | 3.6% |
| Belo Horizonte / Curitiba / Porto Alegre | 2.4–2.6% |
| Recife | 1.5% |
| Salvador | 1.1% |

Customers in Rio de Janeiro and São Paulo are blocked about **3×** as often as in Salvador. Mobile users are blocked **5×** as often as tablet users (3.4% vs 0.7%).
- **Partly justified.** Fraud is more common in those cities (6.9–7.1% vs 4.1%) and on mobile.
- **Still an unequal burden.** The cost falls on honest customers in some places more than others, and city and device can act as proxies for region or income.

**Required practice:**
- Monitor the honest-customer block rate **per segment**: the comparison and the approval report already show recall and blocks per city and device.
- Treat a widening gap as a release blocker for review.
- Offer step-up authentication instead of a hard block where possible.

## Limitations and risks

- **Synthetic data.** Real fraud is less separable; expect lower PR-AUC in production.
- **Concept drift is detected late.** Input checks (PSI) can't see it, and labels arrive after about 30 days. In the drift drill, December's shift would have been caught on 1 January.
- **Uncalibrated scores after retraining.** The December challenger ranked new fraud well but scored it far too low: score bins 0.01–0.10 held 18–38% fraud. Thresholds handle this, but raw scores shouldn't be read as probabilities. Calibration is in the backlog.
- **Assumed costs.** The R$ 10 false-block cost and the +0.5 pt customer budget are business assumptions. Changing them changes who gets blocked.
- **New categories.** Merchants or cities unseen in training are scored as "unknown" until retraining. They are monitored.
- **Adversarial adaptation.** Fraudsters change behaviour in response to blocking, which is exactly what December looked like.

## Monitoring and maintenance

- **Live:** traffic, errors, latency, block and review rates per merchant type, score percentiles and unseen values. Prometheus and Grafana, with thresholds in the [runbook](monitoring.md).
- **Hourly:** drift per feature (PSI), score drift, segment rates and a heartbeat.
- **Once labels mature:** PR-AUC and recall overall and per segment, against this model's evaluation baseline.
- **Retraining:**
  1. Train a challenger with its own thresholds.
  2. Compare on an unseen holdout with promotion rules: PR-AUC gain and its bootstrap interval, honest customers blocked, business cost, no segment regression.
  3. Produce an approval report for a human.
  4. Run in shadow on live traffic.
  5. Promote by registry alias, with one-command rollback.
