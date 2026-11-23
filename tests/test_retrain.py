"""Challenger training tests: time windows without leakage, recency weights, and safe output."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import FINAL_MODEL_PATH
from fraud_detection.evaluate import choose_block_threshold, false_block_rate
from fraud_detection.policy import Policy
from fraud_detection.retrain import (
    INFO_FILE,
    MODEL_FILE,
    POLICY_FILE,
    challenger_weights,
    retrain_windows,
    train_challenger,
)
from tests.model_fixture import synthetic_history

AS_OF = pd.Timestamp("2025-04-30")   # synthetic data covers 2025-01-01 .. 2025-03-31; labels mature 30 days


@pytest.fixture(scope="module")
def history(params):
    hist = synthetic_history(6_000, seed=3).rename(columns={"label": params.data.target})
    return hist.sort_values(params.data.timestamp_col, ignore_index=True)


def test_windows_are_ordered_disjoint_and_end_before_matured_cutoff(params, history):
    ts = history[params.data.timestamp_col]
    w = retrain_windows(ts, AS_OF, params)
    m = w["masks"]
    assert not (m["train"] & m["validation"]).any() and not (m["validation"] & m["holdout"]).any()
    assert ts[m["train"]].max() < ts[m["validation"]].min() <= ts[m["validation"]].max() < ts[m["holdout"]].min()
    # nothing newer than the label-maturity cutoff is used anywhere: those labels are not final yet
    assert ts[m["train"] | m["validation"] | m["holdout"]].max() < AS_OF - pd.Timedelta(days=30)


def test_holdout_is_the_most_recent_matured_days(params, history):
    ts = history[params.data.timestamp_col]
    w = retrain_windows(ts, AS_OF, params)
    span = ts[w["masks"]["holdout"]]
    assert (span.max() - span.min()) <= pd.Timedelta(days=params.retrain.holdout_days)
    assert w["ranges"]["holdout"][1] == str(AS_OF - pd.Timedelta(days=params.monitoring.label_maturity_days))


def test_only_recent_training_rows_are_up_weighted(params, history):
    ts = history[params.data.timestamp_col]
    w = retrain_windows(ts, AS_OF, params)
    train_ts = ts[w["masks"]["train"]]
    weights = challenger_weights(train_ts, w["recent_start"], params)
    recent = (train_ts >= w["recent_start"]).to_numpy()
    assert set(np.unique(weights[recent])) == {params.retrain.recent_weight}
    assert set(np.unique(weights[~recent])) == {1.0}


def test_challenger_is_trained_and_saved_without_touching_the_champion(params, history, tmp_path):
    champion_before = FINAL_MODEL_PATH.stat().st_mtime if FINAL_MODEL_PATH.exists() else None
    info = train_challenger(history, params, AS_OF, tmp_path / "challenger")

    assert (tmp_path / "challenger" / MODEL_FILE).exists()
    saved = json.loads((tmp_path / "challenger" / INFO_FILE).read_text(encoding="utf-8"))
    assert saved["role"] == "challenger" and saved["n_trees"] >= 1
    assert saved["windows"]["holdout"]["rows"] > 0
    assert saved["features"]["columns"] == params.features.columns
    assert info["windows"] == saved["windows"]
    # its own decision policy, chosen on its validation window, saved next to it
    policy = Policy.load(tmp_path / "challenger" / POLICY_FILE)
    assert 0 < policy.review_threshold <= policy.block_threshold < 1
    assert "challenger validation" in policy.chosen_on
    assert saved["policy"]["block_threshold"] == policy.block_threshold
    champion_after = FINAL_MODEL_PATH.stat().st_mtime if FINAL_MODEL_PATH.exists() else None
    assert champion_after == champion_before


def test_category_vocabulary_comes_from_training_window_only(params, history, tmp_path):
    hist = history.copy()
    w = retrain_windows(hist[params.data.timestamp_col], AS_OF, params)
    hist.loc[w["masks"]["holdout"], "cidade"] = "Cidade do Futuro"   # appears only in the holdout
    info = train_challenger(hist, params, AS_OF, tmp_path / "c2")
    assert "Cidade do Futuro" not in info["features"]["category_levels"]["cidade"]


# --------------------------------------------------------------------------- customer-impact budget
@pytest.fixture(scope="module")
def cheap_blocks():
    """Fraud worth R$ 100 each, so pure cost minimisation blocks generously."""
    rng = np.random.default_rng(1)
    y = (rng.random(20_000) < 0.05).astype(int)
    probs = np.clip(np.where(y == 1, rng.beta(4, 2, len(y)), rng.beta(1, 6, len(y))), 0, 1)
    return y, probs, np.full(len(y), 100.0)


def test_budget_caps_the_share_of_honest_customers_blocked(params, cheap_blocks):
    y, probs, amounts = cheap_blocks
    free = choose_block_threshold(y, probs, amounts, params)
    capped = choose_block_threshold(y, probs, amounts, params, max_false_block_rate=0.02)
    assert false_block_rate(y, probs, free) > 0.02                     # unconstrained would exceed the budget
    assert false_block_rate(y, probs, capped) <= 0.02 and capped > free
    # the cheapest threshold within the budget: the next lower grid step would break it
    assert false_block_rate(y, probs, round(capped - params.policy.threshold_grid.step, 6)) > 0.02


def test_a_loose_budget_changes_nothing(params, cheap_blocks):
    y, probs, amounts = cheap_blocks
    assert choose_block_threshold(y, probs, amounts, params, max_false_block_rate=1.0) == \
        choose_block_threshold(y, probs, amounts, params)


def test_an_impossible_budget_falls_back_to_the_strictest_threshold(params, cheap_blocks):
    y, probs, amounts = cheap_blocks
    probs = probs.copy()
    probs[np.flatnonzero(y == 0)[:50]] = 0.99           # some honest customers score above every threshold
    g = params.policy.threshold_grid
    strictest = np.arange(g.start, g.stop, g.step).round(6)[-1]
    assert false_block_rate(y, probs, strictest) > 0     # so a 0% budget cannot be met
    assert choose_block_threshold(y, probs, amounts, params, max_false_block_rate=0.0) == strictest


def test_challenger_policy_respects_the_champion_based_budget(params, history, tmp_path):
    info = train_challenger(history, params, AS_OF, tmp_path / "c4", champion_false_block_rate=0.0)
    pol = info["policy"]
    assert pol["false_block_budget"] == params.retrain.max_false_block_rate_increase
    assert pol["validation_false_block_rate"] <= pol["false_block_budget"]
    assert pol["unconstrained_block_threshold"] <= pol["block_threshold"]
    assert "false-block budget" in Policy.load(tmp_path / "c4" / POLICY_FILE).chosen_on


def test_impossible_as_of_date_fails_clearly(params, history, tmp_path):
    with pytest.raises(ValueError, match="no rows or no fraud"):
        train_challenger(history, params, pd.Timestamp("2025-01-20"), tmp_path / "c3")
