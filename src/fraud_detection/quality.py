"""Model quality gate: the minimum standard a model must meet to be released.

Kept free of heavy dependencies so it can be used by tests, the pipeline and (later) the
retraining / promotion step.
"""

from __future__ import annotations

from fraud_detection.config import QualityGateConfig


def check_quality(metrics: dict, gate: QualityGateConfig) -> list[str]:
    """Return the list of failed checks (empty list = the model passes the gate).

    `metrics` has the structure of reports/metrics.json.
    """
    test, policy = metrics["test"], metrics["policy"]
    baseline = metrics["baseline_rule"]["test_pr_auc"]
    failures = []
    if test["pr_auc"] < gate.min_test_pr_auc:
        failures.append(f"test PR-AUC {test['pr_auc']:.3f} < minimum {gate.min_test_pr_auc}")
    if test["pr_auc"] < gate.min_lift_over_rule * baseline:
        failures.append(f"test PR-AUC {test['pr_auc']:.3f} is not >= {gate.min_lift_over_rule}x "
                        f"the rule baseline ({baseline:.3f})")
    if test["recall"] < gate.min_recall_at_block:
        failures.append(f"recall at block threshold {test['recall']:.3f} < minimum {gate.min_recall_at_block}")
    if test["precision"] < gate.min_precision_at_block:
        failures.append(f"precision at block threshold {test['precision']:.3f} < minimum {gate.min_precision_at_block}")
    if policy["block_threshold"] <= policy["review_threshold"]:
        failures.append("block threshold must be above the review threshold")
    return failures
