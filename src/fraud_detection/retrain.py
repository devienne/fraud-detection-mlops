"""Train a CHALLENGER model from the labels available at an "as of" date.

Models decay (December 2025: fraud moved into e-commerce). Retraining is routine, but a retrained
model is only a candidate: it must beat the champion on data neither has seen before it is released.

Time windows, counted back from the as-of date (all in params.yaml -> retrain):

    labels mature after `label_maturity_days`  ->  labelled data ends at  as_of - maturity
    ... training (train_window_days, last recent_days up-weighted) | validation | holdout |

- holdout    : the most recent matured days. NEVER used here; reserved for champion vs challenger.
- validation : early stopping (the same procedure as the champion, via train.fit_early_stopped), then
               the challenger's OWN decision thresholds (evaluate.fit_policy): a new model has a new
               score scale, so the champion's thresholds would waste its improvement.
- training   : a sliding window, recent days weighted more: fraud patterns move.

The challenger is saved in its own folder (models/challenger/ by default). The champion's files
are never touched: promotion is a separate, human-approved step.

Run:  python -m fraud_detection.retrain --as-of 2026-01-31
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud_detection.config import (
    CHALLENGER_DIR,
    EVAL_MODEL_PATH,
    FINAL_MODEL_PATH,
    METADATA_PATH,
    POLICY_PATH,
    TRAINING_INFO_PATH,
    Params,
    load_params,
)
from fraud_detection.data import load_history
from fraud_detection.evaluate import false_block_rate, fit_policy
from fraud_detection.features import fit_category_levels, make_features
from fraud_detection.policy import Policy
from fraud_detection.train import dvc_data_hash, fit_early_stopped, git_commit

MODEL_FILE = "model.txt"
INFO_FILE = "training_info.json"
POLICY_FILE = "policy.json"
METADATA_FILE = "metadata.json"


def retrain_windows(timestamps: pd.Series, as_of: pd.Timestamp, params: Params) -> dict:
    """Boolean masks and date ranges of the training, validation and holdout windows."""
    r = params.retrain
    labelled_until = pd.Timestamp(as_of) - pd.Timedelta(days=params.monitoring.label_maturity_days)
    holdout_start = labelled_until - pd.Timedelta(days=r.holdout_days)
    validation_start = holdout_start - pd.Timedelta(days=r.validation_days)
    train_start = validation_start - pd.Timedelta(days=r.train_window_days)
    recent_start = validation_start - pd.Timedelta(days=r.recent_days)

    ts = pd.to_datetime(timestamps)
    masks = {
        "train": ((ts >= train_start) & (ts < validation_start)).to_numpy(),
        "validation": ((ts >= validation_start) & (ts < holdout_start)).to_numpy(),
        "holdout": ((ts >= holdout_start) & (ts < labelled_until)).to_numpy(),
    }
    ranges = {"train": (train_start, validation_start), "validation": (validation_start, holdout_start),
              "holdout": (holdout_start, labelled_until), "recent_weighted": (recent_start, validation_start)}
    return {"masks": masks, "ranges": {k: [str(a), str(b)] for k, (a, b) in ranges.items()},
            "labelled_until": labelled_until, "recent_start": recent_start}


def champion_artifacts(kind: str, history: pd.DataFrame, params: Params) -> dict:
    """The model a challenger competes with: file, category levels, policy, and the last data it learned from.

    'released' is the production model. 'eval' is the evaluation model (train Jan-Sep, early stopping
    Oct-Nov): the model that was effectively in production before the December 2025 drift, used to
    replay that retraining decision honestly.
    """
    ts, month = history[params.data.timestamp_col], history[params.data.timestamp_col].dt.month
    if kind == "eval":
        levels = json.loads(TRAINING_INFO_PATH.read_text(encoding="utf-8"))["eval_model"]["category_levels"]
        seen = month.isin(params.split.train_months + params.split.validation_months)
        return {"model_path": EVAL_MODEL_PATH, "category_levels": levels, "policy": Policy.load(POLICY_PATH),
                "seen_until": ts[seen].max(),
                "description": "evaluation model (train Jan-Sep, early stopping Oct-Nov)"}
    if kind == "released":
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return {"model_path": FINAL_MODEL_PATH, "category_levels": meta["features"]["category_levels"],
                "policy": Policy.load(POLICY_PATH), "seen_until": ts.max(),
                "description": f"released model {meta['model_version']}"}
    raise ValueError(f"unknown champion {kind!r}: use 'released' or 'eval'")


def score_with(champion: dict, rows: pd.DataFrame, params: Params) -> np.ndarray:
    booster = lgb.Booster(model_file=str(champion["model_path"]))
    return booster.predict(make_features(rows, champion["category_levels"], params.features.columns,
                                         params.data.timestamp_col))


def challenger_weights(timestamps: pd.Series, recent_start: pd.Timestamp, params: Params) -> np.ndarray:
    """Recent training rows weigh `recent_weight`, older ones 1."""
    recent = (pd.to_datetime(timestamps) >= recent_start).to_numpy()
    return np.where(recent, params.retrain.recent_weight, 1.0)


def train_challenger(history: pd.DataFrame, params: Params, as_of: pd.Timestamp, out_dir: Path,
                     champion_false_block_rate: float | None = None) -> dict:
    """Fit the challenger on the training window (early stopping on validation) and save it to `out_dir`.

    `champion_false_block_rate`: the share of legitimate transactions the champion blocks on the same
    validation window. The challenger's block threshold is then the lowest-cost one that blocks at most
    that share + `max_false_block_rate_increase` (customer-impact budget). Without it: lowest cost only.
    """
    ts_col, target, cols = params.data.timestamp_col, params.data.target, params.features.columns
    w = retrain_windows(history[ts_col], as_of, params)
    tr, va = w["masks"]["train"], w["masks"]["validation"]
    y = history[target].to_numpy()
    for name, mask in w["masks"].items():
        if mask.sum() == 0 or y[mask].sum() == 0:
            raise ValueError(f"{name} window {w['ranges'][name]} has no rows or no fraud: "
                             "check the as-of date and params.retrain")

    levels = fit_category_levels(history[tr], params.features.categorical)   # what the challenger can know
    X_tr = make_features(history[tr], levels, cols, ts_col)
    X_va = make_features(history[va], levels, cols, ts_col)
    weights = challenger_weights(history.loc[tr, ts_col], w["recent_start"], params)
    model = fit_early_stopped(X_tr, y[tr], X_va, y[va], params, sample_weight=weights)
    va_scores = model.predict_proba(X_va)[:, 1]
    va_amounts = history.loc[va, "valor_transacao"].to_numpy()
    budget = None
    if champion_false_block_rate is not None:
        budget = round(champion_false_block_rate + params.retrain.max_false_block_rate_increase, 6)
    unconstrained = fit_policy(y[va], va_scores, va_amounts, params, chosen_on="unconstrained")
    policy = fit_policy(y[va], va_scores, va_amounts, params, max_false_block_rate=budget,
                        chosen_on="challenger validation {} to {}".format(*(d[:10] for d in w["ranges"]["validation"]))
                                  + (f", false-block budget {budget:.4f}" if budget is not None else ""))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_dir / MODEL_FILE))
    policy.save(out_dir / POLICY_FILE)
    info = {
        "role": "challenger",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": str(pd.Timestamp(as_of)),
        "labelled_until": str(w["labelled_until"]),
        "git_commit": git_commit(),
        "data_md5": dvc_data_hash(params),
        "n_trees": int(model.best_iteration_),
        "best_validation_pr_auc": round(float(model.best_score_["valid_0"]["average_precision"]), 4),
        "policy": {
            "block_threshold": policy.block_threshold, "review_threshold": policy.review_threshold,
            "validation_false_block_rate": round(false_block_rate(y[va], va_scores, policy.block_threshold), 4),
            "false_block_budget": budget,
            "champion_validation_false_block_rate": champion_false_block_rate,
            # what pure cost minimisation would have chosen: shows how much the budget binds
            "unconstrained_block_threshold": unconstrained.block_threshold,
            "unconstrained_false_block_rate": round(
                false_block_rate(y[va], va_scores, unconstrained.block_threshold), 4),
        },
        "windows": {name: {"range": w["ranges"][name], "rows": int(m.sum()),
                           "frauds": int(y[m].sum())} for name, m in w["masks"].items()},
        "recent_weighted": {"range": w["ranges"]["recent_weighted"],
                            "rows": int((weights > 1).sum()), "weight": params.retrain.recent_weight},
        "features": {"columns": cols, "category_levels": levels, "timestamp_col": ts_col},
        "params": {"model": params.model.model_dump(), "retrain": params.retrain.model_dump()},
    }
    (out_dir / INFO_FILE).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    write_serving_metadata(info, out_dir)
    return info


def write_serving_metadata(info: dict, out_dir: Path) -> dict:
    """metadata.json in the same format as the champion's, so FraudScorer and the API can serve the challenger."""
    sha = info["git_commit"].split("-")[0][:7]
    metadata = {
        "model_version": f"{pd.Timestamp(info['as_of']):%Y%m%d}-challenger-{sha}",
        "model_file": "final_model.txt",          # its name inside a deploy bundle (promote.py)
        "algorithm": "LightGBM binary classifier",
        "role": "challenger",
        "lineage": {"trained_at": info["trained_at"], "git_commit": info["git_commit"], "data_md5": info["data_md5"],
                    "as_of": info["as_of"], "labelled_until": info["labelled_until"]},
        "features": info["features"],
        "training": {"n_trees": info["n_trees"], "windows": info["windows"],
                     "recent_weighted": info["recent_weighted"]},
        "policy": {k: info["policy"][k] for k in ("block_threshold", "review_threshold")},
        "evaluation": {"best_validation_pr_auc": info["best_validation_pr_auc"],
                       "holdout": "see reports/champion_vs_challenger.json"},
    }
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fraud_detection.retrain", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", type=pd.Timestamp, required=True,
                        help="date the retrain happens: only labels matured by then are used")
    parser.add_argument("--out", type=Path, default=CHALLENGER_DIR)
    parser.add_argument("--champion", choices=["released", "eval", "none"], default="released",
                        help="model whose false-block rate sets the customer-impact budget ('none': no budget)")
    args = parser.parse_args(argv)

    params = load_params()
    history = load_history(params)
    champion_rate = None
    if args.champion != "none":
        champion = champion_artifacts(args.champion, history, params)
        va = retrain_windows(history[params.data.timestamp_col], args.as_of, params)["masks"]["validation"]
        champion_rate = round(false_block_rate(history.loc[va, params.data.target].to_numpy(),
                                               score_with(champion, history[va], params),
                                               champion["policy"].block_threshold), 6)
    info = train_challenger(history, params, args.as_of, args.out, champion_false_block_rate=champion_rate)
    pol = info["policy"]
    print(f"Challenger as of {info['as_of']} (labels until {info['labelled_until']}): "
          f"{info['n_trees']} trees, best validation PR-AUC {info['best_validation_pr_auc']}")
    if pol["false_block_budget"] is not None:
        print(f"  customer-impact budget: champion blocks {pol['champion_validation_false_block_rate']:.2%} "
              f"of honest customers on validation -> at most {pol['false_block_budget']:.2%}")
    print(f"  own thresholds: block {pol['block_threshold']} (blocks {pol['validation_false_block_rate']:.2%} "
          f"of honest customers), review {pol['review_threshold']:.4f}")
    print(f"  without the budget it would be block {pol['unconstrained_block_threshold']} "
          f"({pol['unconstrained_false_block_rate']:.2%} of honest customers)")
    for name, win in info["windows"].items():
        print(f"  {name:10s} {win['range'][0][:10]} -> {win['range'][1][:10]}  "
              f"{win['rows']:>8,} rows, {win['frauds']:>6,} frauds")
    rw = info["recent_weighted"]
    print(f"  weighted x{rw['weight']:g}: {rw['range'][0][:10]} -> {rw['range'][1][:10]} ({rw['rows']:,} rows)")
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
