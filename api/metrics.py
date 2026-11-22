"""Model metrics exposed at /metrics for Prometheus.

Service metrics (requests, status codes, HTTP latency) are added by prometheus-fastapi-instrumentator
in main.py. This module adds what a healthy-looking service can still get wrong: the model's
behaviour.

Rule for labels: only small, fixed sets of values (decision, band, merchant type, endpoint, feature).
Every distinct label combination becomes a separate time series, so per-transaction or per-merchant
detail stays in the prediction log, never in a label.

Note: prometheus_client keeps counters in process memory. We run one Uvicorn worker per container
and scale with replicas, each scraped separately; several workers in one container would need the
library's multiprocess mode.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from prometheus_client import Counter, Gauge, Histogram

# Fine resolution at the low end, where most transactions and both decision thresholds sit.
SCORE_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

MODEL_INFO = Gauge(
    "fraud_model_info", "Model version currently served (value is always 1).", ["model_version"])

SCORE = Histogram(
    "fraud_score", "Distribution of predicted fraud probabilities.", buckets=SCORE_BUCKETS)

DECISIONS = Counter(
    "fraud_decisions", "Decisions taken, by risk band and merchant type.",
    ["decision", "risk_band", "merchant_type"])

UNKNOWN_CATEGORY = Counter(
    "fraud_unknown_category",
    "Categorical values never seen in training (scored as missing): an early train/serve skew signal.",
    ["feature"])

SCORING_SECONDS = Histogram(
    "fraud_scoring_seconds", "Time spent building features and scoring (model only, no HTTP).",
    ["endpoint"], buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))


def set_model_info(version: str) -> None:
    MODEL_INFO.clear()                      # exactly one version reported per process
    MODEL_INFO.labels(model_version=version).set(1)


def record_predictions(raw: pd.DataFrame, probabilities: Sequence[float], bands: Sequence[str],
                       decisions: Sequence[str], category_levels: dict[str, list[str]],
                       endpoint: str, seconds: float) -> None:
    """Update every model metric for one scored request (single or batch)."""
    SCORING_SECONDS.labels(endpoint=endpoint).observe(seconds)
    for p in probabilities:
        SCORE.observe(p)
    for decision, band, merchant_type in zip(decisions, bands, raw["tipo_estabelecimento"], strict=True):
        DECISIONS.labels(decision=decision, risk_band=band, merchant_type=merchant_type).inc()
    for feature, levels in category_levels.items():
        unknown = int((~raw[feature].isin(levels)).sum())
        if unknown:
            UNKNOWN_CATEGORY.labels(feature=feature).inc(unknown)
