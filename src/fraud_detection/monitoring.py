"""Batch monitoring: compare a window of recent transactions with the training reference.

Prometheus (api/metrics.py) says "something looks odd right now" with a few low-cardinality numbers.
This job says WHAT changed, with full detail, from the prediction log or a batch file:

- input drift   : PSI per model feature, against bins fixed on the training reference
- output drift  : score percentiles and score PSI (tails move before means do)
- segments      : decision rates and scores per merchant type, device, city, hour
- performance   : once labels are final (label delay), PR-AUC, recall/precision at the block threshold,
                  overall and per segment, against the baseline the model achieved when it was evaluated.
                  This is the only check that sees CONCEPT drift (same inputs, different fraud).
- alerts        : one actionable message per problem (related segment findings grouped under the
                  largest one), plus a heartbeat file so a stalled monitor is itself detectable

Commands:
  python -m fraud_detection.monitoring reference          # build models/reference.json for the released model
  python -m fraud_detection.monitoring run [--window logs/predictions.jsonl] [--labels labels.csv]
                                           [--as-of 2026-02-01] [--fail-on-alert]
  python -m fraud_detection.monitoring heartbeat --max-age-hours 2   # exit 1 if the monitor stopped running
  python -m fraud_detection.monitoring shadow                        # challenger vs champion on live traffic

PSI (Population Stability Index): split a feature into bins, compare the share of rows in each bin,
    PSI = sum over bins of (current% - reference%) * ln(current% / reference%)
0 = identical; the usual reading is < 0.10 stable, 0.10-0.25 watch, > 0.25 act (params.yaml).

The bins are decided ONCE, on the reference, and saved with it. Re-binning each window would make
windows incomparable and could hide drift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_detection.config import (
    EVAL_MODEL_PATH,
    HEARTBEAT_PATH,
    MONITORING_REPORT_PATH,
    PREDICTION_LOG_PATH,
    REFERENCE_PATH,
    SHADOW_LOG_PATH,
    TRAINING_INFO_PATH,
    MonitoringConfig,
    Params,
    load_params,
)
from fraud_detection.policy import ACTIONS, Policy

MISSING = "__missing__"
OTHER = "__other__"
MAX_DISCRETE_VALUES = 30          # a non-categorical column with this few values (hour, 0/1) is binned by value
PSI_EPS = 1e-4                    # floor for empty bins, so ln() stays finite


# --------------------------------------------------------------------------- inputs
def monitoring_frame(raw: pd.DataFrame, columns: Sequence[str], timestamp_col: str) -> pd.DataFrame:
    """The model's input columns, as human-readable values.

    Deliberately NOT make_features(): that maps unseen categories to missing, which would hide exactly
    the kind of change monitoring must see (a new city, a renamed merchant).
    """
    frame = pd.DataFrame(index=raw.index)
    for col in columns:
        if col == "hour":
            frame[col] = pd.to_datetime(raw[timestamp_col]).dt.hour
        elif col == "is_new_device":
            frame[col] = raw[col].map({True: 1, False: 0})     # also matches 1/0 (True == 1 in Python)
        else:
            frame[col] = raw[col]
    return frame


# --------------------------------------------------------------------------- binning
def _numeric_edges(values: pd.Series, n_bins: int) -> list[float]:
    """Inner quantile edges of the reference (duplicates removed: spiky data gives fewer bins)."""
    qs = np.quantile(values.dropna().to_numpy(dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1])
    return [float(e) for e in np.unique(qs)]


def _assign_bins(values: pd.Series, spec: dict) -> pd.Series:
    """Map each value to the name of its reference bin."""
    missing = values.isna()
    if spec["kind"] == "numeric":
        idx = np.searchsorted(np.asarray(spec["edges"]), values.to_numpy(dtype=float), side="right")
        labels = pd.Series([f"bin_{i:02d}" for i in idx], index=values.index)
    else:
        as_text = values.astype(str)
        labels = as_text.where(as_text.isin(spec["categories"]), OTHER)
    return labels.mask(missing, MISSING)


def _proportions(labels: pd.Series, bins: Sequence[str]) -> dict[str, float]:
    counts = labels.value_counts()
    total = max(len(labels), 1)
    return {b: float(counts.get(b, 0)) / total for b in bins}


def _bin_names(spec: dict) -> list[str]:
    if spec["kind"] == "numeric":
        names = [f"bin_{i:02d}" for i in range(len(spec["edges"]) + 1)]
    else:
        names = [*spec["categories"], OTHER]
    return [*names, MISSING]


def feature_spec(values: pd.Series, categorical: bool, cfg: MonitoringConfig) -> dict:
    """Decide the bins for one feature from the reference data."""
    if not categorical and values.nunique(dropna=True) > MAX_DISCRETE_VALUES:
        return {"kind": "numeric", "edges": _numeric_edges(values, cfg.numeric_bins)}
    top = values.dropna().astype(str).value_counts().head(cfg.top_categories)
    return {"kind": "categorical", "categories": sorted(top.index.tolist())}


def profile(values: pd.Series, spec: dict) -> dict:
    """Bin shares and missing rate of one column, using fixed reference bins."""
    labels = _assign_bins(values, spec)
    return {"proportions": _proportions(labels, _bin_names(spec)),
            "missing_rate": float(values.isna().mean()) if len(values) else 0.0}


# --------------------------------------------------------------------------- PSI
def psi(reference: dict[str, float], current: dict[str, float], eps: float = PSI_EPS) -> float:
    """Population Stability Index between two bin-share dicts over the same bins."""
    total = 0.0
    for b in reference:
        r = max(reference[b], eps)
        c = max(current.get(b, 0.0), eps)
        total += (c - r) * math.log(c / r)
    return total


def psi_status(value: float, cfg: MonitoringConfig) -> str:
    if value >= cfg.psi_alert:
        return "alert"
    return "warn" if value >= cfg.psi_warn else "ok"


def score_percentiles(scores: Sequence[float], cfg: MonitoringConfig) -> dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return {}
    return {f"p{round(p * 100)}": round(float(np.quantile(arr, p)), 6) for p in cfg.score_percentiles}


# --------------------------------------------------------------------------- reference & drift
def build_reference(frame: pd.DataFrame, scores: Sequence[float], categorical: Sequence[str],
                    cfg: MonitoringConfig, description: str = "") -> dict:
    """Snapshot of the training-time distributions: bins, shares, missing rates, score percentiles.

    Saved next to the model, so every model version carries the reference it must be compared with.
    """
    features = {}
    for col in frame.columns:
        spec = feature_spec(frame[col], col in categorical, cfg)
        features[col] = {**spec, **profile(frame[col], spec)}
    score_series = pd.Series(np.asarray(scores, dtype=float))
    score_spec = {"kind": "numeric", "edges": _numeric_edges(score_series, cfg.numeric_bins)}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "description": description,
        "rows": int(len(frame)),
        "features": features,
        "score": {**score_spec, **profile(score_series, score_spec),
                  "percentiles": score_percentiles(scores, cfg)},
    }


def feature_drift(reference: dict, frame: pd.DataFrame, cfg: MonitoringConfig) -> dict[str, dict]:
    """PSI, status and missing rate per feature of the current window, worst first."""
    out = {}
    for col, ref in reference["features"].items():
        cur = profile(frame[col], ref)
        value = psi(ref["proportions"], cur["proportions"])
        moved = sorted(ref["proportions"], key=lambda b: abs(cur["proportions"][b] - ref["proportions"][b]),
                       reverse=True)[:3]
        out[col] = {
            "psi": round(value, 4),
            "status": psi_status(value, cfg),
            "missing_rate": round(cur["missing_rate"], 6),
            "reference_missing_rate": round(ref["missing_rate"], 6),
            # the bins that moved most: tells the on-call engineer WHERE the change is
            "top_moves": {b: {"reference": round(ref["proportions"][b], 4),
                              "current": round(cur["proportions"][b], 4)} for b in moved},
        }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["psi"], reverse=True))


def score_drift(reference: dict, scores: Sequence[float], cfg: MonitoringConfig) -> dict:
    """Score PSI plus reference vs current percentiles."""
    ref = reference["score"]
    cur = profile(pd.Series(np.asarray(scores, dtype=float)), ref)
    value = psi(ref["proportions"], cur["proportions"])
    return {"psi": round(value, 4), "status": psi_status(value, cfg),
            "percentiles": {"reference": ref["percentiles"], "current": score_percentiles(scores, cfg)}}


# --------------------------------------------------------------------------- segments (no labels needed)
def segment_rates(frame: pd.DataFrame, scores: Sequence[float], policy: Policy,
                  cfg: MonitoringConfig) -> dict[str, dict]:
    """Decision shares and score percentiles for every value of every segment column.

    A global block rate can look normal while one segment is broken (e.g. every e-commerce payment
    blocked, hidden by many approved supermarket payments).
    """
    scores = np.asarray(scores, dtype=float)
    decisions = pd.Series([ACTIONS[b] for b in policy.band(scores)], index=frame.index)
    out: dict[str, dict] = {}
    for col in cfg.segments:
        out[col] = {}
        for value, idx in frame.groupby(col, dropna=False).groups.items():
            shares = decisions.loc[idx].value_counts(normalize=True)
            seg_scores = scores[frame.index.get_indexer(idx)]
            out[col][str(value)] = {
                "rows": int(len(idx)),
                **{f"{a}_rate": round(float(shares.get(a, 0.0)), 4) for a in ("block", "review", "approve")},
                "score_percentiles": score_percentiles(seg_scores, cfg),
            }
    return out


# --------------------------------------------------------------------------- performance (labels needed)
def mature_mask(timestamps: pd.Series, as_of: pd.Timestamp, maturity_days: int) -> np.ndarray:
    """True for transactions old enough for their fraud label to be final.

    Fraud is reported days or weeks later (chargebacks, customer complaints). Recent transactions
    look clean only because nobody has reported them yet; evaluating them would overstate precision
    and understate the fraud rate.
    """
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=maturity_days)
    return (pd.to_datetime(timestamps) <= cutoff).to_numpy()


def performance(labels: Sequence[int], scores: Sequence[float], policy: Policy) -> dict:
    """Overall quality on labelled transactions: same definitions as evaluate.py."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    blocked = policy.decide(p)
    frauds = int(y.sum())
    return {
        "rows": int(len(y)),
        "frauds": frauds,
        "fraud_rate": round(float(y.mean()), 6) if len(y) else None,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if 0 < frauds < len(y) else None,
        "recall_at_block": round(float((blocked & (y == 1)).sum() / frauds), 4) if frauds else None,
        "precision_at_block": round(float((blocked & (y == 1)).sum() / blocked.sum()), 4) if blocked.any() else None,
        "block_rate": round(float(blocked.mean()), 4) if len(y) else None,
    }


def segment_performance(frame: pd.DataFrame, labels: Sequence[int], scores: Sequence[float], policy: Policy,
                        cfg: MonitoringConfig) -> dict[str, dict]:
    """Per segment value: frauds, share of ALL fraud found there, and recall at the block threshold."""
    y = np.asarray(labels, dtype=int)
    blocked = policy.decide(np.asarray(scores, dtype=float))
    total = max(int(y.sum()), 1)
    out: dict[str, dict] = {}
    for col in cfg.segments:
        out[col] = {}
        for value, idx in frame.groupby(col, dropna=False).groups.items():
            pos = frame.index.get_indexer(idx)
            frauds = int(y[pos].sum())
            out[col][str(value)] = {
                "rows": int(len(pos)),
                "frauds": frauds,
                "fraud_share": round(frauds / total, 4),
                "recall_at_block": round(float((blocked[pos] & (y[pos] == 1)).sum() / frauds), 4) if frauds else None,
            }
    return out


def performance_profile(frame: pd.DataFrame, labels: Sequence[int], scores: Sequence[float], policy: Policy,
                        cfg: MonitoringConfig) -> dict:
    """Overall + per-segment performance. Saved in the reference as the baseline, and computed for each window."""
    return {"overall": performance(labels, scores, policy),
            "segments": segment_performance(frame, labels, scores, policy, cfg)}


def _drop_status(drop: float, limit: float) -> str:
    if drop >= limit:
        return "alert"
    return "warn" if drop >= limit / 2 else "ok"


def performance_drift(baseline: dict, current: dict, cfg: MonitoringConfig) -> dict:
    """Compare a labelled window with the baseline: overall metrics, segment recall and fraud mix."""
    b, c = baseline["overall"], current["overall"]
    overall = {}
    for metric, limit in (("pr_auc", cfg.max_pr_auc_drop), ("recall_at_block", cfg.max_recall_drop)):
        if b[metric] is None or c[metric] is None:
            overall[metric] = {"baseline": b[metric], "current": c[metric], "status": "insufficient_data"}
            continue
        drop = b[metric] - c[metric]
        overall[metric] = {"baseline": b[metric], "current": c[metric], "drop": round(drop, 4),
                           "status": _drop_status(drop, limit)}

    segments = []
    for col, values in current["segments"].items():
        for value, cur in values.items():
            base = baseline["segments"].get(col, {}).get(value)
            if base is None:
                segments.append({"segment": col, "value": value, "check": "new_segment_value",
                                 "status": "warn", "current": cur})
                continue
            shift = cur["fraud_share"] - base["fraud_share"]
            if c["frauds"] >= cfg.min_segment_frauds:
                segments.append({"segment": col, "value": value, "check": "fraud_share",
                                 "baseline": base["fraud_share"], "current": cur["fraud_share"],
                                 "change": round(shift, 4),
                                 "status": _drop_status(abs(shift), cfg.max_fraud_share_shift)})
            if min(base["frauds"], cur["frauds"]) >= cfg.min_segment_frauds:
                drop = base["recall_at_block"] - cur["recall_at_block"]
                segments.append({"segment": col, "value": value, "check": "recall_at_block",
                                 "baseline": base["recall_at_block"], "current": cur["recall_at_block"],
                                 "drop": round(drop, 4), "status": _drop_status(drop, cfg.max_recall_drop)})
    rank = {"alert": 0, "warn": 1, "ok": 2}
    segments.sort(key=lambda s: (rank.get(s["status"], 3), -abs(s.get("drop", s.get("change", 0)) or 0)))
    return {"overall": overall, "segments": segments,
            "frauds": {"baseline": b["frauds"], "current": c["frauds"]}}


# --------------------------------------------------------------------------- alerts
SEVERITY_RANK = {"alert": 0, "warn": 1}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def build_alerts(report: dict) -> list[dict]:
    """Turn check results into a short list of actionable alerts, most severe first.

    Each alert says what moved, by how much, and what to do. Segment findings usually share one
    cause (fraud moving into e-commerce also lowers recall in every city), so they are grouped under
    the largest one instead of paging someone sixteen times.
    """
    alerts: list[dict] = []

    for feature, d in report["input_drift"].items():
        if d["status"] in SEVERITY_RANK:
            bin_name, move = next(iter(d["top_moves"].items()))
            alerts.append({
                "severity": d["status"], "check": "input_drift", "subject": feature,
                "message": f"Input '{feature}' changed (PSI {d['psi']:.2f}); largest move: "
                           f"'{bin_name}' {_pct(move['reference'])} -> {_pct(move['current'])}.",
                "action": "Check the upstream source of this field for a pipeline or schema change. "
                          "If the population really changed, plan a retraining challenger.",
            })
        if d["missing_rate"] - d["reference_missing_rate"] >= 0.01:
            alerts.append({
                "severity": "alert", "check": "missing_values", "subject": feature,
                "message": f"Missing values in '{feature}' rose from {_pct(d['reference_missing_rate'])} "
                           f"to {_pct(d['missing_rate'])}.",
                "action": "Check the producer of this field: the model scores missing values as unknown.",
            })

    sd = report["score_drift"]
    if sd["status"] in SEVERITY_RANK:
        ref_p, cur_p = sd["percentiles"]["reference"], sd["percentiles"]["current"]
        alerts.append({
            "severity": sd["status"], "check": "score_drift", "subject": "fraud_probability",
            "message": f"Score distribution shifted (PSI {sd['psi']:.2f}); p99 {ref_p.get('p99')} -> "
                       f"{cur_p.get('p99')}, p90 {ref_p.get('p90')} -> {cur_p.get('p90')}.",
            "action": "Compare with input drift and the block rate on the dashboard; "
                      "check whether the decision thresholds still fit.",
        })

    perf = report.get("performance") or {}
    drift = perf.get("drift")
    if drift:
        for metric, d in drift["overall"].items():
            if d["status"] in SEVERITY_RANK:
                alerts.append({
                    "severity": d["status"], "check": "performance", "subject": metric,
                    "message": f"{metric} on matured labels fell from {d['baseline']:.3f} to {d['current']:.3f}.",
                    "action": "Train a challenger on recent data and compare it with the champion "
                              "(retraining pipeline); re-check the decision thresholds.",
                })

        findings = [s for s in drift["segments"] if s["status"] in SEVERITY_RANK]
        changes = [s for s in findings if s["check"] != "new_segment_value"]
        if changes:
            head = changes[0]                      # already sorted: most severe, largest move first
            same = [s for s in changes if (s["segment"], s["value"]) == (head["segment"], head["value"])]
            parts = [f"{s['check']} {_pct(s['baseline'])} -> {_pct(s['current'])}" for s in same]
            related = [s for s in changes if s not in same]
            alerts.append({
                "severity": head["status"], "check": "segment_performance",
                "subject": f"{head['segment']}={head['value']}",
                "message": f"Fraud pattern changed in {head['segment']}={head['value']}: " + "; ".join(parts)
                           + (f". {len(related)} related segment findings (likely the same cause)." if related
                              else "."),
                "action": "Review recent fraud cases in this segment with the fraud team; consider a temporary "
                          "rule for it and a retraining challenger with recent data up-weighted.",
                "related": [{k: s[k] for k in ("segment", "value", "check", "baseline", "current")}
                            for s in related],
            })
        new_values = [f"{s['segment']}={s['value']}" for s in findings if s["check"] == "new_segment_value"]
        if new_values:
            alerts.append({
                "severity": "warn", "check": "new_segment_value", "subject": ", ".join(new_values),
                "message": f"Values never seen in the baseline: {', '.join(new_values)}.",
                "action": "Confirm they are expected (new merchant, city or market); "
                          "the model scores them as unknown until the next training.",
            })

    alerts.sort(key=lambda a: SEVERITY_RANK[a["severity"]])
    return alerts


# --------------------------------------------------------------------------- the monitoring run
def load_window(path: Path, timestamp_col: str) -> pd.DataFrame:
    """Transactions to monitor: the API's prediction log (.jsonl) or a raw file (.csv) for replays."""
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if not records:
            raise ValueError(f"no predictions logged in {path}")
        df = pd.json_normalize(records)
        df.columns = [c.removeprefix("inputs.") for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]          # transaction_id is logged in inputs and response
    else:
        df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    return df


def attach_labels(window: pd.DataFrame, labels_path: Path, target: str) -> pd.DataFrame:
    """Join late-arriving fraud labels (transaction_id, target) onto logged predictions."""
    labels = pd.read_csv(labels_path, usecols=["transaction_id", target])
    return window.merge(labels, on="transaction_id", how="left", validate="many_to_one")


def run_monitoring(window: pd.DataFrame, scores: Sequence[float], reference: dict, policy: Policy,
                   params: Params, as_of: pd.Timestamp) -> dict:
    """Every check on one window, plus alerts. Labels are used only where present and matured."""
    cfg, ts_col, target = params.monitoring, params.data.timestamp_col, params.data.target
    scores = np.asarray(scores, dtype=float)
    frame = monitoring_frame(window, params.features.columns, ts_col)
    versions = window["model_version"].value_counts().to_dict() if "model_version" in window else {}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": {"model_version": reference.get("model_version"), "description": reference.get("description")},
        "window": {"rows": int(len(window)), "from": str(window[ts_col].min()), "to": str(window[ts_col].max()),
                   "model_versions": {str(k): int(v) for k, v in versions.items()}},
        "input_drift": feature_drift(reference, frame, cfg),
        "score_drift": score_drift(reference, scores, cfg),
        "segments": segment_rates(frame, scores, policy, cfg),
    }

    if target not in window or window[target].isna().all():
        report["performance"] = {"status": "no_labels"}
    else:
        mature = mature_mask(window[ts_col], as_of, cfg.label_maturity_days) & window[target].notna().to_numpy()
        if mature.sum() == 0:
            report["performance"] = {"status": "no_mature_labels",
                                     "label_maturity_days": cfg.label_maturity_days}
        else:
            current = performance_profile(frame[mature], window.loc[mature, target].astype(int).to_numpy(),
                                          scores[mature], policy, cfg)
            report["performance"] = {"status": "evaluated", "as_of": str(as_of),
                                     "matured_rows": int(mature.sum()), "current": current["overall"],
                                     "drift": performance_drift(reference["performance"], current, cfg)}

    report["alerts"] = build_alerts(report)
    report["status"] = report["alerts"][0]["severity"] if report["alerts"] else "ok"
    return report


# --------------------------------------------------------------------------- heartbeat
def write_heartbeat(path: Path, report: dict) -> None:
    """Written only when a run completes: its age tells whether the monitor itself is alive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "last_success_at": report["generated_at"], "status": report["status"],
        "alerts": len(report["alerts"]), "window_rows": report["window"]["rows"],
    }, indent=2), encoding="utf-8")


def check_heartbeat(path: Path, max_age_hours: float, now: datetime | None = None) -> tuple[bool, str]:
    if not path.exists():
        return False, f"no heartbeat at {path}: the monitor has never completed a run"
    last = datetime.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["last_success_at"])
    age = ((now or datetime.now(timezone.utc)) - last).total_seconds() / 3600
    if age > max_age_hours:
        return False, f"monitor last succeeded {age:.1f} h ago (limit {max_age_hours} h): check the scheduler"
    return True, f"monitor alive: last success {age:.1f} h ago"


# --------------------------------------------------------------------------- shadow mode
def shadow_summary(path: Path) -> dict:
    """How the shadow model would have decided on live traffic, compared with the champion.

    Read before promotion: a challenger that looked good on the holdout but behaves very differently
    on live traffic (e.g. blocks twice as much) is a reason to stop and look.
    """
    with open(path, encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    if df.empty:
        raise ValueError(f"no shadow records in {path}")
    agreement = pd.crosstab(df["champion_decision"], df["shadow_decision"])
    rate = {side: df[f"{side}_decision"].value_counts(normalize=True).round(4).to_dict()
            for side in ("champion", "shadow")}
    return {
        "rows": int(len(df)),
        "from": str(df["logged_at"].min()), "to": str(df["logged_at"].max()),
        "champion_versions": df["champion_version"].value_counts().to_dict(),
        "shadow_versions": df["shadow_version"].value_counts().to_dict(),
        "decision_rates": rate,
        "agreement_rate": round(float((df["champion_decision"] == df["shadow_decision"]).mean()), 4),
        "agreement_matrix": {c: {s: int(agreement.loc[c, s]) for s in agreement.columns} for c in agreement.index},
        "newly_blocked": int(((df["shadow_decision"] == "block") & (df["champion_decision"] != "block")).sum()),
        "no_longer_blocked": int(((df["champion_decision"] == "block") & (df["shadow_decision"] != "block")).sum()),
    }


# --------------------------------------------------------------------------- reference for the released model
def build_release_reference(params: Params) -> dict:
    """Reference for the released model, saved next to it.

    - distributions: the final model's training data (all labelled months) and its scores on it
    - performance baseline: the EVALUATION model on the test months, i.e. honest out-of-time quality
      (the final model has seen every labelled month, so its in-sample quality would be optimistic)
    """
    import lightgbm as lgb

    from fraud_detection.data import load_history
    from fraud_detection.features import make_features
    from fraud_detection.predict import FraudScorer

    cfg, ts_col = params.monitoring, params.data.timestamp_col
    history = load_history(params)
    scorer = FraudScorer()
    frame = monitoring_frame(history, params.features.columns, ts_col)
    reference = build_reference(frame, scorer.score(history)["fraud_probability"], params.features.categorical,
                                cfg, description="final model on all labelled months")
    reference["model_version"] = scorer.version

    test = history[ts_col].dt.month.isin(params.split.test_months).to_numpy()
    levels = json.loads(TRAINING_INFO_PATH.read_text(encoding="utf-8"))["eval_model"]["category_levels"]
    eval_scores = lgb.Booster(model_file=str(EVAL_MODEL_PATH)).predict(
        make_features(history[test], levels, params.features.columns, ts_col))
    reference["performance"] = performance_profile(frame[test], history.loc[test, params.data.target].to_numpy(),
                                                   eval_scores, scorer.policy, cfg)
    reference["performance_source"] = f"evaluation model on test months {params.split.test_months}"
    return reference


# --------------------------------------------------------------------------- command line
def _print_summary(report: dict) -> None:
    w = report["window"]
    print(f"Monitoring {w['rows']:,} transactions ({w['from']} -> {w['to']}): status {report['status'].upper()}")
    perf = report["performance"]
    if perf["status"] == "evaluated":
        c = perf["current"]
        print(f"  labels: {perf['matured_rows']:,} matured, PR-AUC {c['pr_auc']}, recall@block {c['recall_at_block']}")
    else:
        print(f"  labels: {perf['status']}")
    for a in report["alerts"]:
        print(f"  [{a['severity'].upper()}] {a['message']}\n          -> {a['action']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fraud_detection.monitoring", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference", help="build the reference for the released model")
    ref.add_argument("--out", type=Path, default=REFERENCE_PATH)
    run = sub.add_parser("run", help="monitor a window of transactions")
    run.add_argument("--window", type=Path, default=PREDICTION_LOG_PATH, help="prediction log (.jsonl) or .csv")
    run.add_argument("--labels", type=Path, help="CSV with transaction_id and the fraud label")
    run.add_argument("--as-of", type=pd.Timestamp, default=None, help="evaluation date (default: now)")
    run.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    run.add_argument("--report", type=Path, default=MONITORING_REPORT_PATH)
    run.add_argument("--heartbeat", type=Path, default=HEARTBEAT_PATH)
    run.add_argument("--no-heartbeat", action="store_true",
                     help="one-off or pipeline runs: do not touch the scheduled monitor's heartbeat")
    run.add_argument("--fail-on-alert", action="store_true", help="exit with code 2 when an alert is raised")
    hb = sub.add_parser("heartbeat", help="exit 1 if the monitor has not succeeded recently")
    hb.add_argument("--path", type=Path, default=HEARTBEAT_PATH)
    hb.add_argument("--max-age-hours", type=float, default=2.0)
    sh = sub.add_parser("shadow", help="summarise the shadow model's decisions vs the champion's")
    sh.add_argument("--log", type=Path, default=SHADOW_LOG_PATH)
    args = parser.parse_args(argv)

    if args.command == "shadow":
        s = shadow_summary(args.log)
        print(json.dumps(s, indent=2, ensure_ascii=False))
        return 0

    if args.command == "heartbeat":
        ok, message = check_heartbeat(args.path, args.max_age_hours)
        print(message)
        return 0 if ok else 1

    params = load_params()
    if args.command == "reference":
        reference = build_release_reference(params)
        args.out.write_text(json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Reference for model {reference['model_version']} saved to {args.out}")
        return 0

    from fraud_detection.predict import FraudScorer

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    window = load_window(args.window, params.data.timestamp_col)
    if args.labels:
        window = attach_labels(window, args.labels, params.data.target)
    scorer = FraudScorer()
    # Logged predictions are monitored as served; raw files are scored with the released model.
    scores = window["fraud_probability"] if "fraud_probability" in window else scorer.score(window)["fraud_probability"]
    as_of = args.as_of if args.as_of is not None else pd.Timestamp.now().floor("s")
    report = run_monitoring(window, scores, reference, scorer.policy, params, as_of)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not args.no_heartbeat:
        write_heartbeat(args.heartbeat, report)
    _print_summary(report)
    return 2 if args.fail_on_alert and report["status"] == "alert" else 0


if __name__ == "__main__":
    sys.exit(main())
