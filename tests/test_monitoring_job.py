"""Monitoring job tests: actionable alerts, the end-to-end run, label joining, heartbeat and CLI.

Uses the synthetic transactions and tiny model of tests/model_fixture.py, so it runs on CI.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import fraud_detection.predict as predict_module
from fraud_detection.monitoring import (
    attach_labels,
    build_alerts,
    build_reference,
    check_heartbeat,
    feature_drift,
    load_window,
    main,
    monitoring_frame,
    performance_drift,
    performance_profile,
    run_monitoring,
    score_drift,
    write_heartbeat,
)
from fraud_detection.predict import FraudScorer
from tests.model_fixture import build_test_model, synthetic_history
from tests.test_monitoring import CATEGORICAL, CFG, POLICY, _labels, _model_scores, _window


# --------------------------------------------------------------------------- alerts
def _report(window: pd.DataFrame, labels: np.ndarray, scores: np.ndarray, base_window) -> dict:
    frame, base_scores, base_labels = base_window
    reference = build_reference(frame, base_scores, CATEGORICAL, CFG)
    base = performance_profile(frame, base_labels, base_scores, POLICY, CFG)
    current = performance_profile(window, labels, scores, POLICY, CFG)
    return {"input_drift": feature_drift(reference, window, CFG), "score_drift": score_drift(reference, scores, CFG),
            "performance": {"drift": performance_drift(base, current, CFG)}}


@pytest.fixture(scope="module")
def base_window():
    frame = _window(20_000, seed=10)
    return frame, _model_scores(frame, 11), _labels(frame, "Retail", 12)


def test_concept_drift_gives_one_grouped_actionable_alert(base_window):
    new = _window(10_000, seed=30)
    alerts = build_alerts(_report(new, _labels(new, "E-commerce", 32), _model_scores(new, 31), base_window))

    assert {a["check"] for a in alerts} == {"performance", "segment_performance"}   # no input/score noise
    segment_alerts = [a for a in alerts if a["check"] == "segment_performance"]
    assert len(segment_alerts) == 1                                  # grouped, not one page per segment
    head = segment_alerts[0]
    assert head["severity"] == "alert" and head["subject"].startswith("tipo_estabelecimento=")
    assert head["related"], "the other findings are attached, not dropped"
    for a in alerts:
        assert a["message"] and a["action"], "every alert says what moved and what to do"
    assert alerts[0]["severity"] == "alert"


def test_stable_window_gives_no_alerts(base_window):
    new = _window(10_000, seed=20)
    assert build_alerts(_report(new, _labels(new, "Retail", 21), _model_scores(new, 22), base_window)) == []


def test_missing_values_raise_an_alert(base_window):
    new = _window(10_000, seed=50)
    new.loc[new.sample(frac=0.05, random_state=0).index, "valor_transacao"] = np.nan
    alerts = build_alerts(_report(new, _labels(new, "Retail", 51), _model_scores(new, 52), base_window))
    assert any(a["check"] == "missing_values" and a["subject"] == "valor_transacao" for a in alerts)


# --------------------------------------------------------------------------- end-to-end run
@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return build_test_model(tmp_path_factory.mktemp("model"))


@pytest.fixture(scope="module")
def scorer(model_dir):
    return FraudScorer(model_dir / "final_model.txt", model_dir / "metadata.json", model_dir / "policy.json")


@pytest.fixture(scope="module")
def reference(params, scorer):
    hist = synthetic_history(3_000, seed=0)
    frame = monitoring_frame(hist, params.features.columns, params.data.timestamp_col)
    scores = scorer.score(hist)["fraud_probability"]
    ref = build_reference(frame, scores, params.features.categorical, params.monitoring, "synthetic")
    ref["performance"] = performance_profile(frame, hist["label"].to_numpy(), scores, scorer.policy,
                                             params.monitoring)
    ref["model_version"] = scorer.version
    return ref


@pytest.fixture
def window(params):
    return synthetic_history(2_000, seed=1).rename(columns={"label": params.data.target})


def test_run_evaluates_matured_labels(params, scorer, reference, window):
    scores = scorer.score(window)["fraud_probability"]
    report = run_monitoring(window, scores, reference, scorer.policy, params, pd.Timestamp("2026-01-01"))
    assert report["performance"]["status"] == "evaluated"
    assert report["performance"]["matured_rows"] == len(window)
    assert set(report["input_drift"]) == set(params.features.columns)
    assert report["status"] in {"ok", "warn", "alert"}
    json.dumps(report, default=str)                                   # the report can be written


def test_recent_labels_are_not_trusted_yet(params, scorer, reference, window):
    scores = scorer.score(window)["fraud_probability"]
    ts = window[params.data.timestamp_col]                            # the window spans 90 days
    first_month = run_monitoring(window, scores, reference, scorer.policy, params, ts.min() + pd.Timedelta(days=29))
    assert first_month["performance"]["status"] == "no_mature_labels"   # nothing is 30 days old yet
    at_end = run_monitoring(window, scores, reference, scorer.policy, params, ts.max())
    expected = int((ts <= ts.max() - pd.Timedelta(days=30)).sum())     # only the older part is evaluated
    assert at_end["performance"]["matured_rows"] == expected < len(window)


def test_run_without_labels_still_checks_drift(params, scorer, reference, window):
    unlabelled = window.drop(columns=params.data.target)
    report = run_monitoring(unlabelled, scorer.score(unlabelled)["fraud_probability"], reference,
                            scorer.policy, params, pd.Timestamp("2026-01-01"))
    assert report["performance"] == {"status": "no_labels"}
    assert report["input_drift"] and report["score_drift"]


# --------------------------------------------------------------------------- prediction log & labels
def test_prediction_log_is_read_and_labels_joined(tmp_path, params):
    log = tmp_path / "predictions.jsonl"
    rows = [{"logged_at": "2026-01-15T22:14:01Z", "endpoint": "/predict", "latency_ms": 20.0,
             "inputs": {"transaction_id": f"tx-{i}", "datahora_transacao": "2026-01-15T22:14:00",
                        "nome_estabelecimento": "Online Shop 1", "tipo_estabelecimento": "E-commerce",
                        "valor_transacao": 32.9, "cidade": "São Paulo", "device_type": "mobile",
                        "is_new_device": True},
             "transaction_id": f"tx-{i}", "fraud_probability": 0.7, "risk_band": "High", "decision": "block",
             "model_version": "test-0001"} for i in range(3)]
    log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"transaction_id": ["tx-0", "tx-2"], params.data.target: [1, 0]}).to_csv(labels, index=False)

    window = attach_labels(load_window(log, params.data.timestamp_col), labels, params.data.target)
    assert window["cidade"].tolist() == ["São Paulo"] * 3                         # inputs flattened, text intact
    assert window[params.data.timestamp_col].dt.hour.tolist() == [22, 22, 22]
    assert window[params.data.target].tolist()[0] == 1 and pd.isna(window[params.data.target].iloc[1])


# --------------------------------------------------------------------------- heartbeat
def test_heartbeat_detects_a_stalled_monitor(tmp_path):
    path = tmp_path / "heartbeat.json"
    assert check_heartbeat(path, 2)[0] is False                                     # never ran
    write_heartbeat(path, {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "ok",
                           "alerts": [], "window": {"rows": 10}})
    assert check_heartbeat(path, 2)[0] is True
    later = datetime.now(timezone.utc) + timedelta(hours=3)
    ok, message = check_heartbeat(path, 2, now=later)
    assert ok is False and "scheduler" in message


# --------------------------------------------------------------------------- command line
def test_cli_run_writes_report_and_heartbeat(tmp_path, monkeypatch, params, model_dir, reference, window):
    monkeypatch.setattr(predict_module, "FraudScorer", lambda: FraudScorer(
        model_dir / "final_model.txt", model_dir / "metadata.json", model_dir / "policy.json"))
    ref_path, win_path = tmp_path / "reference.json", tmp_path / "window.csv"
    ref_path.write_text(json.dumps(reference), encoding="utf-8")
    window.to_csv(win_path, index=False)
    report, heartbeat = tmp_path / "monitoring.json", tmp_path / "heartbeat.json"

    code = main(["run", "--window", str(win_path), "--reference", str(ref_path), "--as-of", "2026-01-01",
                 "--report", str(report), "--heartbeat", str(heartbeat)])
    assert code == 0
    assert json.loads(report.read_text(encoding="utf-8"))["performance"]["status"] == "evaluated"
    assert main(["heartbeat", "--path", str(heartbeat), "--max-age-hours", "1"]) == 0
