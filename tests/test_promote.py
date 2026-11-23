"""Registry release tests on a TEMPORARY MLflow store: init, register, promote, rollback, refusals.

The real registry (mlflow.db) is never touched.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
import pytest

from fraud_detection.predict import FraudScorer
from fraud_detection.promote import (
    CHALLENGER,
    CHAMPION,
    init_champion,
    promote,
    register_challenger,
    rollback,
    status,
)
from fraud_detection.retrain import train_challenger
from fraud_detection.tracking import _experiment_id
from tests.model_fixture import TEST_MODEL_VERSION, build_test_model, synthetic_history


@pytest.fixture(scope="module")
def registry(params, tmp_path_factory):
    """Isolated tracking store + a registered champion + a trained challenger with a comparison."""
    root = tmp_path_factory.mktemp("registry")
    original_uri = mlflow.get_tracking_uri()
    p = params.model_copy(update={"mlflow": params.mlflow.model_copy(update={
        "tracking_uri": f"sqlite:///{(root / 'mlflow.db').as_posix()}", "experiment": "test-fraud"})})

    champ_dir = build_test_model(root / "champion_model")              # production-format champion files
    with mlflow.start_run(experiment_id=_experiment_id(p)) as run:
        info = mlflow.lightgbm.log_model(lgb.Booster(model_file=str(champ_dir / "final_model.txt")), name="model",
                                         registered_model_name=p.mlflow.registered_model)
    meta = json.loads((champ_dir / "metadata.json").read_text(encoding="utf-8"))
    meta["mlflow"] = {"run_id": run.info.run_id, "registered_version": info.registered_model_version}
    (champ_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    hist = synthetic_history(6_000, seed=3).rename(columns={"label": p.data.target})
    train_challenger(hist, p, pd.Timestamp("2025-04-30"), root / "challenger")
    comparison = root / "comparison.json"
    comparison.write_text(json.dumps({"recommendation": {"decision": "promote"},
                                      "overall": {"pr_auc": {"champion": 0.5, "challenger": 0.6, "change": 0.1}}}),
                          encoding="utf-8")
    yield {"params": p, "root": root, "champ_dir": champ_dir, "deploy": root / "deploy", "comparison": comparison}
    mlflow.set_tracking_uri(original_uri)


def _served_version(folder) -> str:
    return json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["model_version"]


def test_release_lifecycle(registry):
    p, deploy = registry["params"], registry["deploy"]

    # 1. init: the released model becomes champion and is exported where the API reads it
    exported = init_champion(p, metadata_path=registry["champ_dir"] / "metadata.json", deploy_dir=deploy)
    assert _served_version(deploy / CHAMPION) == TEST_MODEL_VERSION == exported["model_version"]
    champion_version = exported["version"]

    # 2. register: the challenger gets its alias and is exported for shadow mode, ready to be served
    reg = register_challenger(p, registry["root"] / "challenger", registry["comparison"], deploy_dir=deploy)
    assert reg["recommendation"] == "promote" and reg["version"] != champion_version
    shadow = FraudScorer(deploy / CHALLENGER / "final_model.txt", deploy / CHALLENGER / "metadata.json",
                         deploy / CHALLENGER / "policy.json")
    assert "challenger" in shadow.version
    scored = shadow.score(synthetic_history(5, seed=9))
    assert len(scored) == 5 and scored["fraud_probability"].between(0, 1).all()

    # 3. promote needs a named approver
    with pytest.raises(SystemExit, match="approver"):
        promote(p, approved_by=" ", deploy_dir=deploy)

    # 4. promote: pointer moves, the old champion is kept as previous, the challenger stops shadowing
    result = promote(p, approved_by="Fraud Ops Lead", deploy_dir=deploy)
    assert (result["champion"], result["previous"]) == (reg["version"], champion_version)
    assert _served_version(deploy / CHAMPION) == shadow.version
    assert not (deploy / CHALLENGER / "final_model.txt").exists()
    s = status(p)
    assert s["champion"]["approved_by"] == "Fraud Ops Lead" and s["challenger"] is None
    assert s["previous"]["version"] == champion_version

    # 5. rollback: one command restores the old champion, with the reason recorded
    back = rollback(p, reason="block rate spike", deploy_dir=deploy)
    assert back["champion"] == champion_version
    assert _served_version(deploy / CHAMPION) == TEST_MODEL_VERSION
    assert status(p)["previous"]["rollback_reason"] == "block rate spike"


def test_a_challenger_not_recommended_needs_an_override(registry):
    p, deploy = registry["params"], registry["deploy"]
    init_champion(p, metadata_path=registry["champ_dir"] / "metadata.json", deploy_dir=deploy)
    keep = registry["root"] / "keep.json"
    keep.write_text(json.dumps({"recommendation": {"decision": "keep_champion"},
                                "overall": {"pr_auc": {"champion": 0.6, "challenger": 0.5, "change": -0.1}}}),
                    encoding="utf-8")
    register_challenger(p, registry["root"] / "challenger", keep, deploy_dir=deploy)
    with pytest.raises(SystemExit, match="keep_champion"):
        promote(p, approved_by="Fraud Ops Lead", deploy_dir=deploy)
    result = promote(p, approved_by="Fraud Ops Lead", override=True, deploy_dir=deploy)
    assert status(p)["champion"]["override"] == "True" and result["champion"]
