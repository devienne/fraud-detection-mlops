"""Monitoring tests: PSI maths, fixed reference bins, and that drift is (and is not) flagged."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fraud_detection.config import MonitoringConfig, Params, load_params
from fraud_detection.monitoring import (
    MISSING,
    OTHER,
    build_reference,
    feature_drift,
    feature_spec,
    mature_mask,
    monitoring_frame,
    performance,
    performance_drift,
    performance_profile,
    profile,
    psi,
    score_drift,
    segment_rates,
)
from fraud_detection.policy import Policy

CFG = MonitoringConfig(numeric_bins=10, top_categories=20, psi_warn=0.10, psi_alert=0.25,
                       score_percentiles=[0.5, 0.9, 0.95, 0.99],
                       segments=["tipo_estabelecimento", "cidade", "hour"], label_maturity_days=30,
                       min_segment_frauds=30, max_pr_auc_drop=0.10, max_recall_drop=0.10,
                       max_fraud_share_shift=0.15)
POLICY = Policy(block_threshold=0.5, review_threshold=0.2, false_block_cost=10.0, review_recall_target=0.95,
                chosen_on="test")
RNG = np.random.default_rng(0)


def _window(n: int, amount_mean: float = 50.0, ecommerce_share: float = 0.2, new_city_share: float = 0.0,
            seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(["E-commerce", "Retail", "Fuel"], size=n,
                       p=[ecommerce_share, (1 - ecommerce_share) / 2, (1 - ecommerce_share) / 2])
    cities = rng.choice(["São Paulo", "Recife", "Curitiba"], size=n)
    cities = np.where(rng.random(n) < new_city_share, "Cidade Nova", cities)
    return pd.DataFrame({
        "hour": rng.integers(0, 24, n),
        "tipo_estabelecimento": types,
        "cidade": cities,
        "valor_transacao": rng.normal(amount_mean, 10, n),
    })


CATEGORICAL = ["tipo_estabelecimento", "cidade"]


@pytest.fixture(scope="module")
def reference():
    return build_reference(_window(20_000, seed=1), RNG.beta(1, 20, 20_000), CATEGORICAL, CFG)


# --------------------------------------------------------------------------- PSI maths
def test_psi_of_identical_distributions_is_zero():
    shares = {"a": 0.2, "b": 0.3, "c": 0.5}
    assert psi(shares, shares) == 0.0


def test_psi_matches_hand_calculation():
    # (0.9-0.5)*ln(0.9/0.5) + (0.1-0.5)*ln(0.1/0.5)
    expected = 0.4 * math.log(1.8) + 0.4 * math.log(5)
    assert psi({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1}) == pytest.approx(expected)


def test_psi_stays_finite_when_a_bin_empties():
    assert math.isfinite(psi({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0}))


# --------------------------------------------------------------------------- reference bins
def test_continuous_feature_gets_roughly_equal_quantile_bins(reference):
    spec = reference["features"]["valor_transacao"]
    assert spec["kind"] == "numeric" and len(spec["edges"]) == CFG.numeric_bins - 1
    shares = [v for k, v in spec["proportions"].items() if k != MISSING]
    assert all(v == pytest.approx(0.1, abs=0.01) for v in shares)


def test_discrete_numeric_column_is_binned_by_value(reference):
    assert reference["features"]["hour"]["kind"] == "categorical"     # 24 values, not quantiles


def test_spiky_column_gets_fewer_bins_without_crashing():
    values = pd.Series(np.r_[np.zeros(9_000), RNG.normal(100, 5, 1_000)])
    spec = feature_spec(values, categorical=False, cfg=CFG)
    assert spec["kind"] == "numeric" and len(spec["edges"]) < CFG.numeric_bins - 1


def test_rare_categories_are_pooled_as_other():
    values = pd.Series([f"m{i}" for i in range(100)] * 10)
    spec = feature_spec(values, categorical=True, cfg=CFG)
    assert len(spec["categories"]) == CFG.top_categories
    assert profile(values, spec)["proportions"][OTHER] == pytest.approx(0.8)


def test_missing_values_have_their_own_bin_and_rate():
    values = pd.Series([1.0, 2.0, None, 4.0] * 50)
    spec = feature_spec(pd.Series(np.arange(100.0)), categorical=False, cfg=CFG)
    p = profile(values, spec)
    assert p["missing_rate"] == 0.25 and p["proportions"][MISSING] == 0.25


def test_reference_is_json_serialisable(reference):
    assert json.loads(json.dumps(reference))["rows"] == 20_000


# --------------------------------------------------------------------------- drift detection
def test_same_distribution_is_not_flagged(reference):
    drift = feature_drift(reference, _window(5_000, seed=2), CFG)
    assert all(d["status"] == "ok" for d in drift.values()), drift


def test_shifted_amount_is_flagged(reference):
    drift = feature_drift(reference, _window(5_000, amount_mean=65.0, seed=3), CFG)
    assert drift["valor_transacao"]["status"] == "alert"
    assert next(iter(drift)) == "valor_transacao"                     # worst feature listed first


def test_segment_mix_change_is_flagged_and_located(reference):
    drift = feature_drift(reference, _window(5_000, ecommerce_share=0.6, seed=4), CFG)
    d = drift["tipo_estabelecimento"]
    assert d["status"] == "alert"
    assert "E-commerce" in d["top_moves"]                              # says WHERE it moved


def test_new_city_shows_up_as_other(reference):
    drift = feature_drift(reference, _window(5_000, new_city_share=0.3, seed=5), CFG)
    d = drift["cidade"]
    assert d["status"] == "alert"
    assert d["top_moves"][OTHER]["current"] == pytest.approx(0.3, abs=0.03)


def test_score_drift_reports_percentiles_and_flags_a_shift(reference):
    same = score_drift(reference, RNG.beta(1, 20, 5_000), CFG)
    shifted = score_drift(reference, RNG.beta(2, 10, 5_000), CFG)     # model now much more suspicious
    assert same["status"] == "ok" and shifted["status"] == "alert"
    assert set(shifted["percentiles"]["current"]) == {"p50", "p90", "p95", "p99"}
    assert shifted["percentiles"]["current"]["p99"] > shifted["percentiles"]["reference"]["p99"]


# --------------------------------------------------------------------------- inputs & config
def test_monitoring_frame_keeps_unseen_values_and_derives_hour():
    raw = pd.DataFrame({"datahora_transacao": ["2026-01-15T22:14:00", "2026-01-15T03:00:00"],
                        "cidade": ["Atlantis", "Recife"], "is_new_device": [True, False]})
    frame = monitoring_frame(raw, ["hour", "cidade", "is_new_device"], "datahora_transacao")
    assert frame["hour"].tolist() == [22, 3]
    assert frame["cidade"].tolist() == ["Atlantis", "Recife"]          # not turned into missing
    assert frame["is_new_device"].tolist() == [1, 0]


def test_config_rejects_warn_threshold_above_alert():
    with pytest.raises(ValidationError, match="psi_warn"):
        MonitoringConfig(**{**CFG.model_dump(), "psi_warn": 0.3})


def test_config_rejects_a_segment_that_is_not_a_feature():
    raw = load_params().model_dump()
    raw["monitoring"]["segments"] = ["nome_cliente"]
    with pytest.raises(ValidationError, match="nome_cliente"):
        Params.model_validate(raw)


# --------------------------------------------------------------------------- segments & labels
def _model_scores(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """A stand-in model that learned: Retail at night is risky."""
    rng = np.random.default_rng(seed)
    risky = (frame["tipo_estabelecimento"] == "Retail") & (frame["hour"] >= 20)
    return np.clip(np.where(risky, 0.8, 0.02) + rng.normal(0, 0.05, len(frame)), 0, 1)


def _labels(frame: pd.DataFrame, fraud_type: str, seed: int) -> np.ndarray:
    """Fraud happens at night in `fraud_type` merchants, half of the time."""
    rng = np.random.default_rng(seed)
    pattern = (frame["tipo_estabelecimento"] == fraud_type) & (frame["hour"] >= 20)
    return (pattern & (rng.random(len(frame)) < 0.5)).astype(int).to_numpy()


@pytest.fixture(scope="module")
def baseline_window():
    frame = _window(20_000, seed=10)
    return frame, _model_scores(frame, 11), _labels(frame, "Retail", 12)


def test_mature_mask_keeps_only_transactions_with_final_labels():
    ts = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-16", "2026-01-17", "2026-02-14"]))
    assert mature_mask(ts, pd.Timestamp("2026-02-15"), 30).tolist() == [True, True, False, False]


def test_performance_matches_hand_calculation():
    perf = performance([1, 1, 0, 0, 1], [0.9, 0.3, 0.6, 0.1, 0.8], POLICY)
    # blocked (>= 0.5): rows 0, 2, 4 -> catches 2 of 3 frauds; 2 of 3 blocks are fraud
    assert perf["frauds"] == 3 and perf["recall_at_block"] == pytest.approx(0.6667, abs=1e-4)
    assert perf["precision_at_block"] == pytest.approx(0.6667, abs=1e-4) and perf["block_rate"] == 0.6


def test_performance_without_fraud_does_not_crash():
    perf = performance([0, 0, 0], [0.1, 0.6, 0.2], POLICY)
    assert perf["pr_auc"] is None and perf["recall_at_block"] is None and perf["precision_at_block"] == 0.0


def test_segment_rates_locate_where_blocks_happen(baseline_window):
    frame, scores, _ = baseline_window
    rates = segment_rates(frame, scores, POLICY, CFG)
    retail, fuel = rates["tipo_estabelecimento"]["Retail"], rates["tipo_estabelecimento"]["Fuel"]
    assert retail["block_rate"] > 0.1 and fuel["block_rate"] == 0.0
    for seg in rates["tipo_estabelecimento"].values():
        assert seg["block_rate"] + seg["review_rate"] + seg["approve_rate"] == pytest.approx(1.0, abs=1e-3)


def test_stable_fraud_pattern_raises_no_alert(baseline_window):
    frame, scores, labels = baseline_window
    base = performance_profile(frame, labels, scores, POLICY, CFG)
    new = _window(10_000, seed=20)
    current = performance_profile(new, _labels(new, "Retail", 21), _model_scores(new, 22), POLICY, CFG)
    drift = performance_drift(base, current, CFG)
    assert all(m["status"] == "ok" for m in drift["overall"].values()), drift["overall"]
    assert not [s for s in drift["segments"] if s["status"] == "alert"]


def test_concept_drift_is_caught_by_labels_but_invisible_to_psi(baseline_window):
    """December 2025 in miniature: same inputs, same scores, but fraud moved to another segment."""
    frame, scores, labels = baseline_window
    reference = build_reference(frame, scores, CATEGORICAL, CFG)
    base = performance_profile(frame, labels, scores, POLICY, CFG)

    new = _window(10_000, seed=30)                                  # same input distribution
    new_scores = _model_scores(new, 31)                             # the model has not changed
    new_labels = _labels(new, "E-commerce", 32)                     # ...but fraud has moved

    # Unsupervised checks see nothing: inputs and scores look exactly like the reference
    assert all(d["status"] == "ok" for d in feature_drift(reference, new, CFG).values())
    assert score_drift(reference, new_scores, CFG)["status"] == "ok"

    # Label-based checks catch it, and say where
    drift = performance_drift(base, performance_profile(new, new_labels, new_scores, POLICY, CFG), CFG)
    assert drift["overall"]["recall_at_block"]["status"] == "alert"
    alerts = {(s["value"], s["check"]) for s in drift["segments"] if s["status"] == "alert"}
    assert ("E-commerce", "fraud_share") in alerts and ("Retail", "fraud_share") in alerts


def test_new_segment_value_is_reported(baseline_window):
    frame, scores, labels = baseline_window
    base = performance_profile(frame, labels, scores, POLICY, CFG)
    new = _window(5_000, new_city_share=0.2, seed=40)
    drift = performance_drift(base, performance_profile(new, _labels(new, "Retail", 41),
                                                        _model_scores(new, 42), POLICY, CFG), CFG)
    assert any(s["value"] == "Cidade Nova" and s["check"] == "new_segment_value" for s in drift["segments"])
