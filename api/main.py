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
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import BatchRequest, BatchResponse, FraudScore, Transaction
from fraud_detection import __version__
from fraud_detection.config import MODELS_DIR, PROJECT_ROOT
from fraud_detection.policy import ACTIONS
from fraud_detection.predict import FraudScorer

MODEL_DIR = Path(os.getenv("FRAUD_MODEL_DIR", MODELS_DIR))
POLICY_PATH = Path(os.getenv("FRAUD_POLICY_PATH", MODEL_DIR / "policy.json"))
PREDICTION_LOG = Path(os.getenv("FRAUD_PREDICTION_LOG", PROJECT_ROOT / "logs" / "predictions.jsonl"))


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
    yield
    state.clear()


app = FastAPI(
    title="Fraud Detection API",
    version=__version__,
    description="Scores card transactions: fraud probability, risk band and decision (approve / review / block).",
    lifespan=lifespan,
)


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
    latency_ms = (time.perf_counter() - start) * 1000

    results = [
        FraudScore(
            transaction_id=t.transaction_id,
            fraud_probability=float(s.fraud_probability),
            risk_band=s.risk_band,
            decision=ACTIONS[s.risk_band],
            model_version=s.model_version,
        )
        for t, s in zip(transactions, scored.itertuples(index=False))
    ]

    logged_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    state["logger"].write([
        {"logged_at": logged_at, "endpoint": endpoint, "latency_ms": round(latency_ms, 3),
         "inputs": t.model_dump(mode="json"), **r.model_dump()}
        for t, r in zip(transactions, results)
    ])
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
    }


@app.post("/predict", response_model=FraudScore)
def predict(transaction: Transaction) -> FraudScore:
    """Score one transaction (the online authorisation case)."""
    return _score([transaction], "/predict")[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Score up to 1,000 transactions in one call."""
    results = _score(request.transactions, "/predict/batch")
    return BatchResponse(model_version=_scorer().version, results=results)
