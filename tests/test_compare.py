"""Champion vs challenger tests: the promotion rule promotes real gains and blocks hidden harm."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.compare import bootstrap_pr_auc_gain, compare_models, model_metrics
from tests.model_fixture import synthetic_history
from tests.test_monitoring import POLICY

RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def holdout(params):
    raw = synthetic_history(12_000, seed=5)
    y = raw.pop("label").to_numpy()
    legit = RNG.uniform(0.0, 0.55, len(y))
    champion = np.where(y == 1, RNG.uniform(0.30, 0.80, len(y)), legit)       # catches about 60% of fraud
    better = np.where(y == 1, RNG.uniform(0.45, 1.00, len(y)), legit)         # same legit scores, more fraud caught
    return raw, y, champion, better


def test_customer_impact_and_cost_count_the_right_transactions():
    y = np.array([1, 1, 0, 0, 0])
    m = model_metrics(y, np.array([0.9, 0.1, 0.9, 0.1, 0.1]), POLICY, amounts=np.array([100.0, 40.0, 5, 5, 5]))
    assert m["block_rate"] == 0.4 and m["false_block_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["missed_fraud_amount"] == 40.0 and m["false_blocks"] == 1
    assert m["total_cost"] == 40.0 + POLICY.false_block_cost * 1


def test_identical_models_have_a_gain_interval_around_zero(holdout):
    _, y, champion, _ = holdout
    ci = bootstrap_pr_auc_gain(y, champion, champion, samples=100)["ci95"]
    assert ci[0] <= 0 <= ci[1]


def test_a_real_improvement_is_promoted(params, holdout):
    raw, y, champion, better = holdout
    result = compare_models(raw, y, champion, better, POLICY, POLICY, params)
    assert result["overall"]["recall_at_block"]["change"] > 0.1
    assert result["overall"]["block_rate"]["change"] > 0           # blocks more...
    assert result["overall"]["false_block_rate"]["change"] == 0    # ...but no more honest customers
    rec = result["recommendation"]
    assert rec["decision"] == "promote" and rec["reasons"] == [] and rec["segment_regressions"] == []
    assert len(rec["checks"]) == 5 and all(c["passed"] for c in rec["checks"])


def test_a_hidden_segment_regression_blocks_promotion(params, holdout):
    raw, y, champion, better = holdout
    fuel_fraud = (raw["tipo_estabelecimento"] == "Fuel").to_numpy() & (y == 1)
    worse_on_fuel = np.where(fuel_fraud, champion - 0.3, better)        # better overall, worse for fuel stations
    result = compare_models(raw, y, champion, worse_on_fuel, POLICY, POLICY, params)
    assert result["overall"]["pr_auc"]["change"] > 0                    # the global number still looks good
    rec = result["recommendation"]
    assert rec["decision"] == "keep_champion"
    assert "tipo_estabelecimento=Fuel" in rec["segment_regressions"]


def test_noise_is_not_promoted(params, holdout):
    raw, y, champion, _ = holdout
    noisy = np.clip(champion + RNG.normal(0, 0.01, len(y)), 0, 1)
    rec = compare_models(raw, y, champion, noisy, POLICY, POLICY, params)["recommendation"]
    assert rec["decision"] == "keep_champion"
    assert any("not significant" in r or "below the required" in r for r in rec["reasons"])


def test_a_better_model_needs_its_own_thresholds(params, holdout):
    """Same better ranking on a halved score scale: the champion's 0.5 cut-off wastes it."""
    raw, y, champion, better = holdout
    rescaled = better * 0.5                                            # identical ranking, different scale
    own_policy = POLICY.model_copy(update={"block_threshold": 0.25, "review_threshold": 0.1})

    with_old = compare_models(raw, y, champion, rescaled, POLICY, POLICY, params)
    assert with_old["overall"]["pr_auc"]["change"] > 0                 # ranking is better...
    assert with_old["overall"]["recall_at_block"]["challenger"] == 0   # ...but nothing reaches the old cut-off
    assert with_old["recommendation"]["decision"] == "keep_champion"

    with_own = compare_models(raw, y, champion, rescaled, POLICY, own_policy, params)
    assert with_own["recommendation"]["decision"] == "promote"
    assert with_own["thresholds"]["challenger"]["block"] == 0.25


def test_blocking_more_honest_customers_blocks_promotion(params, holdout):
    raw, y, champion, better = holdout
    trigger_happy = np.where(y == 0, np.clip(better + 0.05, 0, 1), better)   # pushes legit scores over 0.5
    rec = compare_models(raw, y, champion, trigger_happy, POLICY, POLICY, params)["recommendation"]
    assert rec["decision"] == "keep_champion"
    assert any("honest customers" in r for r in rec["reasons"])
