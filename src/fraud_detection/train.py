"""Train the two LightGBM models.

1. Evaluation model: learns from the training months only. After each tree it is scored on the
   validation months, and training stops when that score has not improved for
   `early_stopping_rounds` trees (early stopping). Used only for honest evaluation and for
   choosing the decision thresholds (see evaluate.py).
2. Final model: retrained from scratch on ALL labelled months with the evaluation model's tree
   count x `final_trees_multiplier`, recent months up-weighted. This is the model that scores
   new transactions.

Run as a pipeline stage:  python -m fraud_detection.train
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from fraud_detection.config import (
    EVAL_MODEL_PATH,
    FINAL_MODEL_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    TRAINING_INFO_PATH,
    Params,
    load_params,
)
from fraud_detection.data import load_history
from fraud_detection.features import fit_category_levels, make_features


def split_masks(df: pd.DataFrame, params: Params) -> dict[str, np.ndarray]:
    """Boolean row masks for the time-based train / validation / test split."""
    month = df[params.data.timestamp_col].dt.month
    return {
        "train": month.isin(params.split.train_months).to_numpy(),
        "validation": month.isin(params.split.validation_months).to_numpy(),
        "test": month.isin(params.split.test_months).to_numpy(),
    }


def recency_weights(df: pd.DataFrame, params: Params) -> np.ndarray:
    month = df[params.data.timestamp_col].dt.month
    return np.where(month.isin(params.model.recency_months), params.model.recency_weight, 1.0)


# Files the pipeline itself writes: their changes do not make the *code* uncommitted.
PIPELINE_OUTPUTS = ["dvc.lock", "reports", "models", "data/predictions"]


def git_commit() -> str:
    """Current commit, with '-dirty' when source files have uncommitted changes (lineage of the model)."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        exclude = [f":(exclude){p}" for p in PIPELINE_OUTPUTS]
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", ".", *exclude],
                                        cwd=PROJECT_ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dvc_data_hash(params: Params) -> str | None:
    """md5 recorded by DVC for the training file: ties the model to the exact data version."""
    pointer = params.path(params.data.history_path).with_suffix(".csv.dvc")
    if not pointer.exists():
        return None
    return yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"]


def train(params: Params) -> dict:
    target, cols = params.data.target, params.features.columns
    history = load_history(params)
    y = history[target].to_numpy()
    masks = split_masks(history, params)
    tr, va = masks["train"], masks["validation"]

    # --- 1. Evaluation model: train months only, early stopping watched on validation months
    eval_levels = fit_category_levels(history[tr], params.features.categorical)
    X_tr = make_features(history[tr], eval_levels, cols, params.data.timestamp_col)
    X_va = make_features(history[va], eval_levels, cols, params.data.timestamp_col)
    eval_model = lgb.LGBMClassifier(n_estimators=params.model.max_estimators, **params.model.lgbm_params())
    eval_model.fit(
        X_tr, y[tr],
        eval_set=[(X_va, y[va])], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(params.model.early_stopping_rounds, verbose=False)],
    )
    n_trees = int(eval_model.best_iteration_)

    # --- 2. Final model: all labelled data, more trees, recent months up-weighted
    final_levels = fit_category_levels(history, params.features.categorical)
    X_all = make_features(history, final_levels, cols, params.data.timestamp_col)
    n_trees_final = int(n_trees * params.model.final_trees_multiplier)
    final_model = lgb.LGBMClassifier(n_estimators=n_trees_final, **params.model.lgbm_params())
    final_model.fit(X_all, y, sample_weight=recency_weights(history, params))

    # --- Save models (LightGBM text format: portable, diffable) and training facts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    eval_model.booster_.save_model(str(EVAL_MODEL_PATH))     # keeps only the best iteration's trees
    final_model.booster_.save_model(str(FINAL_MODEL_PATH))

    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data_md5": dvc_data_hash(params),
        "feature_columns": cols,
        "eval_model": {"n_trees": n_trees, "category_levels": eval_levels},
        "final_model": {"n_trees": n_trees_final, "category_levels": final_levels,
                        "recency_months": params.model.recency_months,
                        "recency_weight": params.model.recency_weight},
        "splits": {name: {"rows": int(m.sum()), "fraud_rate": round(float(y[m].mean()), 6)}
                   for name, m in masks.items()},
    }
    TRAINING_INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def main() -> None:
    info = train(load_params())
    print(f"Evaluation model: {info['eval_model']['n_trees']} trees (early stopping)")
    print(f"Final model     : {info['final_model']['n_trees']} trees, trained on all labelled data")
    for name, s in info["splits"].items():
        print(f"  {name:10s} {s['rows']:>8,} rows, fraud rate {s['fraud_rate']:.2%}")
    print(f"Saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
