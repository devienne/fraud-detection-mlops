"""Fraud scoring API.

Run locally:  uvicorn api.main:app --port 8000      (interactive docs at /docs)

Design choices
- The model, its metadata (features, category levels) and the decision policy are loaded ONCE at
  startup; if they are missing or broken the service refuses to start instead of serving bad answers.
- Scoring reuses fraud_detection.predict.FraudScorer, the same code as batch scoring, so online and
  batch scores for a transaction are identical.
- Artifact locations come from environment variables (config, not code): a new policy or model is
  a file swap + restart, not a code change.
- Every prediction is appended to a JSON Lines log (inputs, score, band, decision, model version,
  latency). It feeds monitoring and lets predictions be joined to fraud labels later.
- GET /metrics exposes Prometheus metrics: service metrics (requests, status codes, latency) and
  model metrics (score distribution, decisions, unseen categories, scoring time); see api/metrics.py.
- Shadow mode (optional, FRAUD_SHADOW_MODEL_DIR): a challenger scores the same transactions AFTER the
  response is sent (background task), its results go to a separate log, and it never changes an answer.
  If it fails to load or to score, the champion keeps serving; failures are counted.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from api import metrics
from api.schemas import BatchRequest, BatchResponse, FraudScore, Transaction
from fraud_detection import __version__
from fraud_detection.config import MODELS_DIR, PROJECT_ROOT
from fraud_detection.policy import ACTIONS
from fraud_detection.predict import FraudScorer

MODEL_DIR = Path(os.getenv("FRAUD_MODEL_DIR", MODELS_DIR))
POLICY_PATH = Path(os.getenv("FRAUD_POLICY_PATH", MODEL_DIR / "policy.json"))
PREDICTION_LOG = Path(os.getenv("FRAUD_PREDICTION_LOG", PROJECT_ROOT / "logs" / "predictions.jsonl"))
SHADOW_MODEL_DIR = Path(p) if (p := os.getenv("FRAUD_SHADOW_MODEL_DIR")) else None   # unset: shadow mode off
SHADOW_LOG = Path(os.getenv("FRAUD_SHADOW_LOG", PREDICTION_LOG.parent / "shadow.jsonl"))
# Share of requests also scored by the shadow model. In this single process it competes for the same
# CPU as the champion (measured: shadowing every request halves throughput), so sample it; in
# production the shadow would be a separate consumer of the request stream.
SHADOW_SAMPLE_RATE = float(os.getenv("FRAUD_SHADOW_SAMPLE_RATE", "1.0"))

log = logging.getLogger("fraud_api")


def load_shadow(folder: Path | None) -> FraudScorer | None:
    """The shadow model, or None. Never raises: a broken challenger must not stop the champion."""
    if folder is None or not (folder / "final_model.txt").exists():
        return None
    try:
        return FraudScorer(folder / "final_model.txt", folder / "metadata.json", folder / "policy.json")
    except Exception:                                        # noqa: BLE001 - any failure just disables shadow mode
        log.exception("shadow model in %s failed to load: shadow mode disabled", folder)
        return None


class PredictionLogger:
    """Thread-safe append-only JSON Lines log of every prediction."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, records: list[dict]) -> None:
        lines = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(lines)


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast: a broken or missing artifact stops the service from starting.
    state["scorer"] = FraudScorer(
        model_path=MODEL_DIR / "final_model.txt",
        metadata_path=MODEL_DIR / "metadata.json",
        policy_path=POLICY_PATH,
    )
    state["logger"] = PredictionLogger(PREDICTION_LOG)
    metrics.set_model_info(state["scorer"].version)
    state["shadow"] = load_shadow(SHADOW_MODEL_DIR)
    state["shadow_logger"] = PredictionLogger(SHADOW_LOG) if state["shadow"] else None
    metrics.set_shadow_info(state["shadow"].version if state["shadow"] else None)
    yield
    state.clear()


app = FastAPI(
    title="Fraud Detection API",
    version=__version__,
    description="Scores card transactions: fraud probability, risk band and decision (approve / review / block).",
    lifespan=lifespan,
)

# Service metrics for every endpoint (count, status class, latency histogram), served at /metrics.
# Per-endpoint latency buckets are fine-grained around our ~20 ms typical latency and the 700 ms
# real-time budget (the library default, 0.1/0.5/1 s, cannot resolve p50/p95/p99 here).
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.5)
Instrumentator(excluded_handlers=["/metrics"]).instrument(
    app, latency_lowr_buckets=LATENCY_BUCKETS).expose(app, include_in_schema=False)


def _scorer() -> FraudScorer:
    scorer = state.get("scorer")
    if scorer is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return scorer


def _score(transactions: list[Transaction], endpoint: str) -> list[FraudScore]:
    scorer = _scorer()
    start = time.perf_counter()
    raw = pd.DataFrame([t.model_dump() for t in transactions])
    scored = scorer.score(raw)
    seconds = time.perf_counter() - start
    latency_ms = seconds * 1000

    results = [
        FraudScore(
            transaction_id=t.transaction_id,
            fraud_probability=float(s.fraud_probability),
            risk_band=s.risk_band,
            decision=ACTIONS[s.risk_band],
            model_version=s.model_version,
        )
        for t, s in zip(transactions, scored.itertuples(index=False), strict=True)
    ]

    logged_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    state["logger"].write([
        {"logged_at": logged_at, "endpoint": endpoint, "latency_ms": round(latency_ms, 3),
         "inputs": t.model_dump(mode="json"), **r.model_dump()}
        for t, r in zip(transactions, results, strict=True)
    ])
    metrics.record_predictions(
        raw,
        probabilities=[r.fraud_probability for r in results],
        bands=[r.risk_band for r in results],
        decisions=[r.decision for r in results],
        category_levels=scorer.metadata["features"]["category_levels"],
        endpoint=endpoint,
        seconds=seconds,
    )
    return results


def _shadow(transactions: list[Transaction], results: list[FraudScore]) -> None:
    """Score with the shadow model and log it next to the champion's decision. Runs after the response."""
    shadow = state.get("shadow")
    if shadow is None:
        return
    try:
        start = time.perf_counter()
        scored = shadow.score(pd.DataFrame([t.model_dump() for t in transactions]))
        seconds = time.perf_counter() - start
        decisions = [ACTIONS[b] for b in scored["risk_band"]]
        logged_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        state["shadow_logger"].write([
            {"logged_at": logged_at, "transaction_id": r.transaction_id,
             "champion_version": r.model_version, "champion_probability": r.fraud_probability,
             "champion_decision": r.decision, "shadow_version": shadow.version,
             "shadow_probability": float(p), "shadow_decision": d}
            for r, p, d in zip(results, scored["fraud_probability"], decisions, strict=True)
        ])
        metrics.record_shadow([r.decision for r in results], decisions, seconds)
    except Exception:                                        # noqa: BLE001 - shadow failures never reach clients
        metrics.SHADOW_ERRORS.inc()
        log.exception("shadow scoring failed")


def _score_and_shadow(transactions: list[Transaction], endpoint: str, tasks: BackgroundTasks) -> list[FraudScore]:
    results = _score(transactions, endpoint)
    if state.get("shadow") is not None and random.random() < SHADOW_SAMPLE_RATE:
        tasks.add_task(_shadow, transactions, results)
    return results


@app.get("/health")
def health() -> dict:
    """Liveness + readiness: the process is up AND the model is loaded."""
    scorer = _scorer()
    return {"status": "ok", "model_version": scorer.version}


@app.get("/model-info")
def model_info() -> dict:
    """Model version, lineage, features, decision policy and test metrics."""
    scorer = _scorer()
    meta = scorer.metadata
    return {
        "model_version": scorer.version,
        "algorithm": meta["algorithm"],
        "lineage": meta["lineage"],
        "features": meta["features"]["columns"],
        "policy": scorer.policy.model_dump(),
        "evaluation": meta["evaluation"],
        "mlflow": meta.get("mlflow"),
        "api_version": __version__,
        "shadow": None if (s := state.get("shadow")) is None else
        {"model_version": s.version, "policy": s.policy.model_dump(), "log": str(SHADOW_LOG),
         "sample_rate": SHADOW_SAMPLE_RATE},
    }


@app.post("/predict", response_model=FraudScore)
def predict(transaction: Transaction, background_tasks: BackgroundTasks) -> FraudScore:
    """Score one transaction (the online authorisation case)."""
    return _score_and_shadow([transaction], "/predict", background_tasks)[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest, background_tasks: BackgroundTasks) -> BatchResponse:
    """Score up to 1,000 transactions in one call."""
    results = _score_and_shadow(request.transactions, "/predict/batch", background_tasks)
    return BatchResponse(model_version=_scorer().version, results=results)
