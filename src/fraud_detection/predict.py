"""Batch scoring: classify every transaction of the scoring file (January 2026).

Uses exactly the pieces the online API will use (saved final model, shared feature function with the
training category vocabulary, saved decision policy), so batch and online scores always agree.

Run as a pipeline stage:  python -m fraud_detection.predict
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud_detection.config import (
    FINAL_MODEL_PATH, METADATA_PATH, POLICY_PATH, PREDICTIONS_DIR, REPORTS_DIR, Params, load_params,
)
from fraud_detection.data import load_scoring
from fraud_detection.features import make_features
from fraud_detection.policy import BANDS, Policy

OUTPUT_PATH = PREDICTIONS_DIR / "transacoes_jan2026_classified.csv"


class FraudScorer:
    """Loads the model artifacts once and scores raw transactions. Shared by batch and API."""

    def __init__(self, model_path: Path = FINAL_MODEL_PATH, metadata_path: Path = METADATA_PATH,
                 policy_path: Path = POLICY_PATH):
        self.booster = lgb.Booster(model_file=str(model_path))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.policy = Policy.load(policy_path)
        self.version = self.metadata["model_version"]

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return probability, binary decision, risk band and model version for each raw transaction."""
        feats = self.metadata["features"]
        X = make_features(raw, feats["category_levels"], feats["columns"], feats["timestamp_col"])
        probs = self.booster.predict(X)
        return pd.DataFrame({
            "fraud_probability": np.round(probs, 6),
            "fraud_prediction": self.policy.decide(probs),
            "risk_band": self.policy.band(probs),
            "model_version": self.version,
        }, index=raw.index)


def predict(params: Params) -> dict:
    raw = load_scoring(params)                       # validated against the data contract
    scores = FraudScorer().score(raw)

    # Keep the original columns exactly as delivered (timestamps as in the source file).
    original = pd.read_csv(params.path(params.data.scoring_path))
    out = pd.concat([original, scores.reset_index(drop=True)], axis=1)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    bands = out["risk_band"].value_counts()
    summary = {
        "model_version": scores["model_version"].iloc[0],
        "rows": int(len(out)),
        "mean_fraud_probability": round(float(out["fraud_probability"].mean()), 6),
        "predicted_fraud": int(out["fraud_prediction"].sum()),
        "predicted_fraud_rate": round(float(out["fraud_prediction"].mean()), 6),
        "bands": {b: int(bands.get(b, 0)) for b in BANDS},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    s = predict(load_params())
    print(f"Scored {s['rows']:,} transactions with model {s['model_version']}")
    print(f"Expected fraud rate (mean probability): {s['mean_fraud_probability']:.2%}")
    print(f"Classified as fraud: {s['predicted_fraud']:,} ({s['predicted_fraud_rate']:.2%})")
    print("Bands:", s["bands"])
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
