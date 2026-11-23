"""Approval report tests: the approver sees the decision, every rule, the judgment calls, and a sign-off."""

from __future__ import annotations

import numpy as np

from fraud_detection.compare import compare_models
from fraud_detection.report import approver_notes, render
from tests.test_compare import POLICY
from tests.test_compare import holdout as holdout  # noqa: F401  (fixture re-used)

INFO = {
    "as_of": "2026-01-31 00:00:00", "labelled_until": "2026-01-01 00:00:00", "git_commit": "abc1234",
    "data_md5": "d41d8cd9", "n_trees": 120, "best_validation_pr_auc": 0.66,
    "windows": {"train": {"range": ["2025-01-01", "2025-12-04"], "rows": 1000, "frauds": 50},
                "validation": {"range": ["2025-12-04", "2025-12-18"], "rows": 200, "frauds": 10},
                "holdout": {"range": ["2025-12-18", "2026-01-01"], "rows": 200, "frauds": 12}},
    "recent_weighted": {"range": ["2025-11-04", "2025-12-04"], "rows": 100, "weight": 3.0},
    "policy": {"block_threshold": 0.25, "review_threshold": 0.01, "validation_false_block_rate": 0.0287,
               "false_block_budget": 0.0302, "champion_validation_false_block_rate": 0.0252,
               "unconstrained_block_threshold": 0.02, "unconstrained_false_block_rate": 0.0665},
}


def _comparison(params, raw, y, champion, challenger, challenger_policy=POLICY) -> dict:
    result = compare_models(raw, y, champion, challenger, POLICY, challenger_policy, params)
    return {"generated_at": "2026-01-31T09:00:00+00:00", "champion": "champion model",
            "challenger": "challenger model", "holdout": {"from": "2025-12-18 00:00:00", "to": "2025-12-31 23:59:00"},
            **result}


def test_promote_report_shows_decision_rules_and_sign_off(params, holdout):  # noqa: F811
    raw, y, champion, better = holdout
    md = render(_comparison(params, raw, y, champion, better), INFO)
    assert "**Automatic recommendation: PROMOTE the challenger.** All promotion rules pass." in md
    assert md.count("✅ pass") == 5 and "❌ FAIL" not in md
    for must_have in ("Honest customers blocked", "Business cost", "Manual review queue", "- [ ] **Approve**",
                      "- [ ] **Reject**", "Approver:", "abc1234", "d41d8cd9"):
        assert must_have in md, must_have
    assert "None: no segment lost recall." in md                       # zero changes are not shown as losses
    assert "R$ -" not in md                                            # money changes read "−R$ 844"


def test_keep_report_names_the_failing_rule_and_segment(params, holdout):  # noqa: F811
    raw, y, champion, better = holdout
    fuel_fraud = (raw["tipo_estabelecimento"] == "Fuel").to_numpy() & (y == 1)
    md = render(_comparison(params, raw, y, champion, np.where(fuel_fraud, champion - 0.3, better)), INFO)
    assert "KEEP the champion" in md and "❌ FAIL" in md
    assert "tipo_estabelecimento=Fuel" in md
    losses = md.split("**Largest losses**")[1].split("**Largest gains**")[0]
    assert "tipo_estabelecimento = Fuel" in losses


def test_segments_still_weak_after_retraining_are_flagged(params, holdout):  # noqa: F811
    raw, y, champion, better = holdout
    fuel = (raw["tipo_estabelecimento"] == "Fuel").to_numpy()
    blind_on_fuel = np.where(fuel, 0.0, better)                       # never blocks fuel-station fraud
    notes = " ".join(approver_notes(_comparison(params, raw, y, np.where(fuel, 0.0, champion), blind_on_fuel), INFO))
    assert "Still weak after retraining" in notes and "tipo_estabelecimento = Fuel (0.0%" in notes


def test_bigger_review_queue_and_binding_budget_are_raised_for_the_approver(params, holdout):  # noqa: F811
    raw, y, champion, better = holdout
    loose_review = POLICY.model_copy(update={"review_threshold": 0.05})     # sends many more cases to analysts
    comparison = _comparison(params, raw, y, champion, better, loose_review)
    notes = " ".join(approver_notes(comparison, INFO))
    assert "Analyst workload" in notes and "Can the team absorb this queue?" in notes
    assert "budget is binding" in notes and "not calibrated" in notes


def test_report_is_clean_markdown(params, holdout):  # noqa: F811
    raw, y, champion, better = holdout
    md = render(_comparison(params, raw, y, champion, better), INFO)
    assert md.endswith("\n") and not any(line != line.rstrip() for line in md.splitlines())
    table_rows = [line for line in md.splitlines() if line.startswith("|")]
    assert all(line.endswith("|") for line in table_rows)


def test_report_without_a_budget_skips_budget_notes(params, holdout):  # noqa: F811
    raw, y, champion, better = holdout
    info = {**INFO, "policy": {**INFO["policy"], "false_block_budget": None}}
    notes = " ".join(approver_notes(_comparison(params, raw, y, champion, better), info))
    assert "budget" not in notes.lower()
