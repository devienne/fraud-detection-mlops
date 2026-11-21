"""Evaluate the evaluation model and select the decision policy.

Validation months -> choose the thresholds (decisions).
Test month        -> measure and report (never used for any decision).

Run as a pipeline stage:  python -m fraud_detection.evaluate
"""

from __future__ import annotations

import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # headless: the pipeline runs without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from fraud_detection.config import (
    EVAL_MODEL_PATH,
    FINAL_MODEL_PATH,
    METADATA_PATH,
    POLICY_PATH,
    REPORTS_DIR,
    TRAINING_INFO_PATH,
    Params,
    load_params,
)
from fraud_detection.data import load_history
from fraud_detection.features import make_features
from fraud_detection.policy import BANDS, Policy
from fraud_detection.tracking import log_run
from fraud_detection.train import split_masks


def choose_block_threshold(y, probs, amounts, params: Params) -> float:
    """Threshold minimising (missed fraud amount + cost per false block) on the given data."""
    g = params.policy.threshold_grid
    grid = np.arange(g.start, g.stop, g.step).round(6)

    def total_cost(t: float) -> float:
        flagged = probs >= t
        missed_fraud = amounts[(y == 1) & ~flagged].sum()
        false_blocks = ((y == 0) & flagged).sum()
        return missed_fraud + params.policy.false_block_cost * false_blocks

    return float(min(grid, key=total_cost))   # first (lowest) threshold among equal costs


def choose_review_threshold(y, probs, recall_target: float) -> float:
    """Highest threshold that still flags `recall_target` of fraud into High + Medium."""
    _, recall, thresholds = precision_recall_curve(y, probs)
    return float(thresholds[np.where(recall[:-1] >= recall_target)[0][-1]])


def business_metrics(y, probs, amounts, policy: Policy) -> dict:
    blocked = policy.decide(probs)
    caught = (y == 1) & blocked
    false_blocks = int(((y == 0) & blocked).sum())
    prevented = float(amounts[caught].sum())
    return {
        "block_rate": float(blocked.mean()),
        "recall": float(caught.sum() / y.sum()),
        "precision": float(caught.sum() / blocked.sum()),
        "fraud_amount": float(amounts[y == 1].sum()),
        "fraud_amount_prevented": prevented,
        "false_blocks": false_blocks,
        "net_benefit": prevented - policy.false_block_cost * false_blocks,
    }


def band_table(y, probs, policy: Policy) -> dict:
    bands = policy.band(probs)
    out = {}
    for b in BANDS:
        m = bands == b
        out[b] = {"transactions": int(m.sum()), "frauds": int(y[m].sum()),
                  "share_of_transactions": float(m.mean()), "share_of_frauds": float(y[m].sum() / y.sum()),
                  "fraud_rate": float(y[m].mean()) if m.any() else 0.0}
    return out


def rule_baseline(history: pd.DataFrame, masks: dict, params: Params) -> np.ndarray:
    """Analyst-style baseline: training fraud rate of each (merchant type, hour) cell."""
    hour = history[params.data.timestamp_col].dt.hour
    key = pd.MultiIndex.from_arrays([history["tipo_estabelecimento"], hour])
    train = history[masks["train"]]
    cell_rate = train.groupby(["tipo_estabelecimento", hour[masks["train"]]])[params.data.target].mean()
    return key.map(cell_rate).fillna(0).to_numpy()


def plot_pr_curve(y, model_probs, baseline_probs, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, p, color in [("LightGBM", model_probs, "#2a78d6"), ("Rule baseline", baseline_probs, "#898781")]:
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, color=color, lw=2, label=f"{name} (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color="#c3c2b7", ls="--", lw=1)
    ax.set(xlabel="Recall: share of fraud caught", ylabel="Precision: share of blocks that are fraud",
           title="Test month: precision-recall", ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(params: Params) -> dict:
    info = json.loads(TRAINING_INFO_PATH.read_text(encoding="utf-8"))
    history = load_history(params)
    y = history[params.data.target].to_numpy()
    amounts = history["valor_transacao"].to_numpy()
    masks = split_masks(history, params)
    va, te = masks["validation"], masks["test"]

    model = lgb.Booster(model_file=str(EVAL_MODEL_PATH))
    X = make_features(history, info["eval_model"]["category_levels"], info["feature_columns"],
                      params.data.timestamp_col)
    probs = model.predict(X)

    # --- Decisions: thresholds chosen on the VALIDATION months only
    policy = Policy(
        block_threshold=choose_block_threshold(y[va], probs[va], amounts[va], params),
        review_threshold=choose_review_threshold(y[va], probs[va], params.policy.review_recall_target),
        false_block_cost=params.policy.false_block_cost,
        review_recall_target=params.policy.review_recall_target,
        chosen_on=f"validation months {params.split.validation_months}",
    )
    policy.save(POLICY_PATH)

    # --- Reporting: everything measured on the TEST month only
    baseline = rule_baseline(history, masks, params)
    metrics = {
        "validation": {"pr_auc": average_precision_score(y[va], probs[va]),
                       "roc_auc": roc_auc_score(y[va], probs[va])},
        "test": {
            "pr_auc": average_precision_score(y[te], probs[te]),
            "roc_auc": roc_auc_score(y[te], probs[te]),
            **business_metrics(y[te], probs[te], amounts[te], policy),
            "bands": band_table(y[te], probs[te], policy),
        },
        "baseline_rule": {"test_pr_auc": average_precision_score(y[te], baseline[te]),
                          "test_roc_auc": roc_auc_score(y[te], baseline[te])},
        "policy": {"block_threshold": policy.block_threshold, "review_threshold": policy.review_threshold},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_pr_curve(y[te], probs[te], baseline[te], REPORTS_DIR / "pr_curve.png")
    metadata = write_metadata(info, policy, metrics, params)

    # Experiment tracking + registry; the registry reference is added to the metadata file.
    final_sample = make_features(history.head(5), info["final_model"]["category_levels"],
                                 info["feature_columns"], params.data.timestamp_col)
    metadata["mlflow"] = log_run(params, metadata, metrics, final_sample)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def model_version(info: dict) -> str:
    """Human-readable, sortable version: training date + short git commit, e.g. 20261120-4e87d2b."""
    date = info["trained_at"][:10].replace("-", "")
    return f"{date}-{info['git_commit'][:7]}"


def write_metadata(info: dict, policy: Policy, metrics: dict, params: Params) -> dict:
    """Everything needed to trace a prediction back to its model, code, data, features and policy."""
    test = metrics["test"]
    metadata = {
        "model_version": model_version(info),
        "model_file": FINAL_MODEL_PATH.name,
        "algorithm": "LightGBM binary classifier",
        "lineage": {
            "trained_at": info["trained_at"],
            "git_commit": info["git_commit"],
            "data_md5": info["data_md5"],
            "training_data": params.data.history_path.as_posix(),
            "trained_on_months": sorted({*params.split.train_months, *params.split.validation_months,
                                         *params.split.test_months}),
        },
        "features": {
            "columns": info["feature_columns"],
            "category_levels": info["final_model"]["category_levels"],
            "timestamp_col": params.data.timestamp_col,
        },
        "training": {
            "n_trees": info["final_model"]["n_trees"],
            "n_trees_early_stopping": info["eval_model"]["n_trees"],
            "recency_months": info["final_model"]["recency_months"],
            "recency_weight": info["final_model"]["recency_weight"],
            "lgbm_params": params.model.lgbm_params(),
        },
        "policy": policy.model_dump(),
        "evaluation": {
            "evaluated_model": "same settings trained on train months only (final model has no held-out data)",
            "test_months": params.split.test_months,
            "test_pr_auc": round(test["pr_auc"], 4),
            "test_roc_auc": round(test["roc_auc"], 4),
            "test_recall_at_block": round(test["recall"], 4),
            "test_precision_at_block": round(test["precision"], 4),
            "test_net_benefit_brl": round(test["net_benefit"], 2),
            "baseline_rule_test_pr_auc": round(metrics["baseline_rule"]["test_pr_auc"], 4),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    m = evaluate(load_params())
    t = m["test"]
    print(f"Thresholds (chosen on validation): block {m['policy']['block_threshold']:.2f}, "
          f"review {m['policy']['review_threshold']:.3f}")
    print(f"Validation PR-AUC {m['validation']['pr_auc']:.4f}  (optimistic: used for decisions)")
    print(f"Test PR-AUC {t['pr_auc']:.4f} | ROC-AUC {t['roc_auc']:.4f} | "
          f"rule baseline PR-AUC {m['baseline_rule']['test_pr_auc']:.4f}")
    print(f"Test at block threshold: recall {t['recall']:.1%}, precision {t['precision']:.1%}, "
          f"net benefit R$ {t['net_benefit']:,.2f}")
    for b, s in t["bands"].items():
        print(f"  {b:6s} {s['transactions']:>6,} transactions, {s['frauds']:>5,} frauds "
              f"({s['share_of_frauds']:.1%} of fraud, rate {s['fraud_rate']:.1%})")


if __name__ == "__main__":
    main()
