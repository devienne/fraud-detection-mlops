# Fraud Detection: from pilot notebook to production-style MLOps

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
│   └── predict.py            # batch scoring (FraudScorer, reused by the API)
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
validate ──> train ──> evaluate ──> predict
data         eval model  thresholds   January
contract     + final     + metrics    classified
             model       + MLflow
```

| Stage | Uses | Produces |
|---|---|---|
| `validate` | raw CSVs | `reports/data_validation.json` (fails on any contract violation) |
| `train` | Jan–Sep (early stopping on Oct–Nov); then all 2025 | `models/eval_model.txt`, `models/final_model.txt`, `models/training_info.json` |
| `evaluate` | Oct–Nov to choose thresholds; December to report | `models/policy.json`, `models/metadata.json`, `reports/metrics.json`, an MLflow run and registered model version |
| `predict` | final model + policy | `data/predictions/transacoes_jan2026_classified.csv`, `reports/prediction_summary.json` |

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

## Roadmap

| Version | Milestone |
|---|---|
| `v0.1-pilot` | Pilot notebook, data versioned with DVC |
| `v0.2-pipeline` | Reproducible training pipeline (`src/`, `params.yaml`, `dvc repro`, MLflow) |
| `v0.3-api` | FastAPI service in Docker |
| `v0.4-ci` | Tests, quality gate and GitHub Actions |
| `v0.5-monitoring` | Prometheus, Grafana and drift monitoring |
| `v1.0` | Champion/challenger retraining, shadow mode, documentation |
