"""API behaviour tests.

They test the SERVICE (endpoints, contract, decisions, logging, startup), not the model's quality.
CI has no access to the real model, so a tiny LightGBM model is trained here on synthetic
transactions and saved in the exact production format (model + metadata.json + policy.json).
"""

from __future__ import annotations

import json

import lightgbm as lgb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from fraud_detection.policy import ACTIONS
from fraud_detection.predict import FraudScorer
from tests.model_fixture import COLUMNS, build_test_model

VALID = {"transaction_id": "tx-1", "datahora_transacao": "2026-01-15T22:14:00",
         "nome_estabelecimento": "Online Shop 1", "tipo_estabelecimento": "E-commerce",
         "valor_transacao": 32.9, "cidade": "São Paulo", "device_type": "mobile", "is_new_device": True}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A tiny model saved in the production artifact format (see tests/model_fixture.py)."""
    return build_test_model(tmp_path_factory.mktemp("model"))


@pytest.fixture
def client(model_dir, tmp_path, monkeypatch):
    """The real app, pointed at the test model; predictions logged to a temp file."""
    monkeypatch.setattr(api_main, "MODEL_DIR", model_dir)
    monkeypatch.setattr(api_main, "POLICY_PATH", model_dir / "policy.json")
    monkeypatch.setattr(api_main, "PREDICTION_LOG", tmp_path / "predictions.jsonl")
    with TestClient(api_main.app) as c:          # `with` runs the startup (model loading)
        yield c


# --------------------------------------------------------------------------- health & info
def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "test-0001"}


def test_model_info_exposes_version_features_and_policy(client):
    info = client.get("/model-info").json()
    assert info["model_version"] == "test-0001"
    assert info["features"] == COLUMNS
    assert info["policy"]["block_threshold"] == 0.5 and info["policy"]["review_threshold"] == 0.2


# --------------------------------------------------------------------------- predict
def test_predict_returns_a_consistent_decision(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"transaction_id", "fraud_probability", "risk_band", "decision", "model_version"}
    assert body["transaction_id"] == "tx-1" and body["model_version"] == "test-0001"
    p = body["fraud_probability"]
    assert 0.0 <= p <= 1.0
    expected_band = "High" if p >= 0.5 else "Medium" if p >= 0.2 else "Low"
    assert body["risk_band"] == expected_band
    assert body["decision"] == ACTIONS[expected_band]


@pytest.mark.parametrize("change", [
    {"is_new_device": "yes"},                     # wrong type
    {"tipo_estabelecimento": "Casino"},           # unknown business category
    {"valor_transacao": 0},                       # not a positive amount
    {"valor_transacao": "30,50"},                 # amount as text
])
def test_invalid_request_gets_422_and_service_keeps_running(client, change):
    r = client.post("/predict", json={**VALID, **change})
    assert r.status_code == 422
    assert r.json()["detail"], "the response must explain what is wrong"
    assert client.get("/health").status_code == 200


def test_missing_field_gets_422(client):
    body = {k: v for k, v in VALID.items() if k != "device_type"}
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert "device_type" in json.dumps(r.json()["detail"])


# --------------------------------------------------------------------------- batch
def test_batch_returns_one_result_per_transaction_in_order(client):
    txs = [{**VALID, "transaction_id": f"tx-{i}", "datahora_transacao": f"2026-01-15T{h:02d}:00:00"}
           for i, h in enumerate([3, 12, 22, 23])]
    r = client.post("/predict/batch", json={"transactions": txs})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["transaction_id"] for x in results] == ["tx-0", "tx-1", "tx-2", "tx-3"]


@pytest.mark.parametrize("n", [0, 1001])
def test_batch_size_limits(client, n):
    r = client.post("/predict/batch", json={"transactions": [VALID] * n})
    assert r.status_code == 422


def test_api_scores_equal_batch_scores(client, model_dir):
    txs = [{**VALID, "transaction_id": f"tx-{i}", "valor_transacao": float(v), "cidade": c}
           for i, (v, c) in enumerate([(5.0, "Recife"), (32.9, "São Paulo"), (250.0, "Manaus")])]
    results = client.post("/predict/batch", json={"transactions": txs}).json()["results"]
    api_probs = [x["fraud_probability"] for x in results]
    raw = pd.DataFrame(txs)
    raw["datahora_transacao"] = pd.to_datetime(raw["datahora_transacao"])
    scorer = FraudScorer(model_dir / "final_model.txt", model_dir / "metadata.json", model_dir / "policy.json")
    batch = scorer.score(raw)
    assert api_probs == batch["fraud_probability"].tolist()


# --------------------------------------------------------------------------- logging & startup
def test_every_prediction_is_logged(client, tmp_path):
    client.post("/predict", json=VALID)
    client.post("/predict/batch", json={"transactions": [VALID, {**VALID, "transaction_id": "tx-2"}]})
    records = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    for rec in records:
        assert {"logged_at", "endpoint", "latency_ms", "inputs", "transaction_id",
                "fraud_probability", "risk_band", "decision", "model_version"} <= set(rec)
    assert records[0]["inputs"]["cidade"] == "São Paulo"          # text kept intact (no encoding damage)


def test_service_refuses_to_start_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "MODEL_DIR", tmp_path)            # empty folder
    monkeypatch.setattr(api_main, "POLICY_PATH", tmp_path / "policy.json")
    monkeypatch.setattr(api_main, "PREDICTION_LOG", tmp_path / "predictions.jsonl")
    # LightGBM cannot open the (missing) model file -> startup fails, no app is served.
    with pytest.raises(lgb.basic.LightGBMError, match="final_model.txt"), TestClient(api_main.app):
        pass
