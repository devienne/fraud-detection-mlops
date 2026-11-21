"""Model quality gate: tests the MODEL, not the code.

Needs the real data and the trained model (DVC outputs), so every test here is marked `model` and
is skipped automatically where they are not available (e.g. on CI machines without DVC access).
Run on a machine with the data:  pytest -m model
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fraud_detection.config import FINAL_MODEL_PATH, PROJECT_ROOT, REPORTS_DIR, load_params
from fraud_detection.policy import BANDS
from fraud_detection.quality import check_quality

PARAMS = load_params()
REQUIRED = [FINAL_MODEL_PATH, REPORTS_DIR / "metrics.json",
            PARAMS.path(PARAMS.data.history_path), PARAMS.path(PARAMS.data.scoring_path)]
MISSING = [str(p.relative_to(PROJECT_ROOT)) for p in REQUIRED if not p.exists()]

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(bool(MISSING), reason=f"real data/model not available: {MISSING}"),
]


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))


def test_current_model_passes_the_quality_gate(metrics):
    failures = check_quality(metrics, PARAMS.quality_gate)
    assert not failures, "Model failed the quality gate:\n- " + "\n- ".join(failures)


def test_gate_rejects_a_model_with_no_skill(metrics):
    """Shuffle the real December scores (same distribution, no skill): the gate must reject it."""
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score

    from fraud_detection.config import EVAL_MODEL_PATH, POLICY_PATH, TRAINING_INFO_PATH
    from fraud_detection.data import load_history
    from fraud_detection.evaluate import business_metrics
    from fraud_detection.features import make_features
    from fraud_detection.policy import Policy
    from fraud_detection.train import split_masks

    info = json.loads(TRAINING_INFO_PATH.read_text(encoding="utf-8"))
    history = load_history(PARAMS)
    test = history[split_masks(history, PARAMS)["test"]]
    X = make_features(test, info["eval_model"]["category_levels"], info["feature_columns"])
    probs = np.random.default_rng(0).permutation(lgb.Booster(model_file=str(EVAL_MODEL_PATH)).predict(X))
    y, amounts = test[PARAMS.data.target].to_numpy(), test["valor_transacao"].to_numpy()

    bad = {**metrics, "test": {"pr_auc": average_precision_score(y, probs),
                               **business_metrics(y, probs, amounts, Policy.load(POLICY_PATH))}}
    failures = check_quality(bad, PARAMS.quality_gate)
    assert failures, "the gate accepted a model with no skill"
    assert any("PR-AUC" in f for f in failures)


def test_released_model_loads_and_scores_real_transactions():
    from fraud_detection.data import load_scoring
    from fraud_detection.predict import FraudScorer

    sample = load_scoring(PARAMS).sample(200, random_state=0)
    scores = FraudScorer().score(sample)
    assert scores["fraud_probability"].between(0, 1).all()
    assert set(scores["risk_band"]) <= set(BANDS)
    assert (scores["fraud_prediction"] == (scores["risk_band"] == "High")).all()
