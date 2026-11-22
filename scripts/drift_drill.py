"""Drift drill: would the monitor have caught the December 2025 fraud shift if the model had been live?

Simulates deployment honestly with the EVALUATION model, which never saw October-December:
  - deployed model   : trained on Jan-Sep (models/eval_model.txt), released thresholds (models/policy.json)
  - reference        : Jan-Sep inputs and that model's scores on them
  - performance base : the model's quality on October (labels already matured at deployment)
  - replay           : November, then December, as if each month's labels had matured

A second replay walks through December day by day with a 30-day label delay, to measure WHEN the
first label-based alert would have fired: the cost of relying on labels.

Usage:  python scripts/drift_drill.py        (needs the DVC data and models: run `dvc repro` first)
Writes: reports/drift_drill.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402

from fraud_detection.config import (  # noqa: E402
    EVAL_MODEL_PATH,
    POLICY_PATH,
    REPORTS_DIR,
    TRAINING_INFO_PATH,
    load_params,
)
from fraud_detection.data import load_history  # noqa: E402
from fraud_detection.features import make_features  # noqa: E402
from fraud_detection.monitoring import (  # noqa: E402
    build_reference,
    monitoring_frame,
    performance_profile,
    run_monitoring,
)
from fraud_detection.policy import Policy  # noqa: E402

OUT_PATH = REPORTS_DIR / "drift_drill.json"
REFERENCE_MONTHS, BASELINE_MONTH, REPLAY_MONTHS = range(1, 10), 10, {11: "November", 12: "December"}


def month_summary(report: dict) -> dict:
    """The few numbers that tell the story of one replayed month."""
    worst_feature, worst = next(iter(report["input_drift"].items()))
    perf = report["performance"]
    return {
        "rows": report["window"]["rows"],
        "status": report["status"],
        "max_feature_psi": {"feature": worst_feature, "psi": worst["psi"], "status": worst["status"]},
        "score_psi": report["score_drift"]["psi"],
        "score_percentiles": report["score_drift"]["percentiles"],
        "performance": {k: perf["current"][k] for k in ("frauds", "pr_auc", "recall_at_block", "precision_at_block")}
        if perf["status"] == "evaluated" else perf["status"],
        "alerts": [{k: a[k] for k in ("severity", "check", "subject", "message")}
                   | {"related_findings": len(a.get("related", []))} for a in report["alerts"]],
    }


def main() -> None:
    params = load_params()
    ts, target = params.data.timestamp_col, params.data.target
    history = load_history(params)
    month = history[ts].dt.month
    levels = json.loads(TRAINING_INFO_PATH.read_text(encoding="utf-8"))["eval_model"]["category_levels"]
    booster = lgb.Booster(model_file=str(EVAL_MODEL_PATH))
    policy = Policy.load(POLICY_PATH)

    def scores_of(df: pd.DataFrame):
        return booster.predict(make_features(df, levels, params.features.columns, ts))

    def frame_of(df: pd.DataFrame) -> pd.DataFrame:
        return monitoring_frame(df, params.features.columns, ts)

    # --- the deployment snapshot: reference distributions + October performance baseline
    ref_rows = history[month.isin(REFERENCE_MONTHS)]
    reference = build_reference(frame_of(ref_rows), scores_of(ref_rows), params.features.categorical,
                                params.monitoring, description="drill: evaluation model, Jan-Sep 2025")
    october = history[month == BASELINE_MONTH]
    reference["performance"] = performance_profile(frame_of(october), october[target].to_numpy(),
                                                   scores_of(october), policy, params.monitoring)
    reference["model_version"] = "eval-model-jan-sep"

    # --- replay 1: each month with all its labels matured
    months = {}
    for m, name in REPLAY_MONTHS.items():
        window = history[month == m]
        as_of = window[ts].max() + pd.Timedelta(days=params.monitoring.label_maturity_days + 1)
        months[name] = month_summary(run_monitoring(window, scores_of(window), reference, policy, params, as_of))

    # --- replay 2: December day by day, labels maturing after 30 days -> when is the first alert?
    december = history[month == 12]
    dec_scores = pd.Series(scores_of(december), index=december.index)
    timeline, first_alert = [], None
    for as_of in pd.date_range("2025-12-02", "2026-02-01", freq="D"):
        seen = december[december[ts] < as_of]
        report = run_monitoring(seen, dec_scores.loc[seen.index], reference, policy, params, as_of)
        perf = report["performance"]
        timeline.append({"as_of": str(as_of.date()), "status": report["status"],
                         "matured_rows": perf.get("matured_rows", 0)})
        if first_alert is None and report["status"] == "alert":
            first_alert = {"as_of": str(as_of.date()), "matured_rows": perf.get("matured_rows", 0),
                           "alerts": [a["message"] for a in report["alerts"]]}

    result = {
        "setup": {"model": "evaluation model trained on Jan-Sep 2025", "reference_months": list(REFERENCE_MONTHS),
                  "baseline_month": BASELINE_MONTH, "policy": policy.model_dump(),
                  "label_maturity_days": params.monitoring.label_maturity_days},
        "baseline_october": reference["performance"]["overall"],
        "months": months,
        "december_live_replay": {"first_alert": first_alert,
                                 "days_after_shift_started": (pd.Timestamp(first_alert["as_of"])
                                                              - pd.Timestamp("2025-12-01")).days
                                 if first_alert else None,
                                 "timeline": timeline},
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # --- console story
    b = result["baseline_october"]
    print(f"Baseline (October): PR-AUC {b['pr_auc']}, recall@block {b['recall_at_block']}")
    for name, s in months.items():
        perf = s["performance"]
        print(f"\n{name}: status {s['status'].upper()} | max feature PSI {s['max_feature_psi']['psi']} "
              f"({s['max_feature_psi']['feature']}) | score PSI {s['score_psi']} | "
              f"PR-AUC {perf['pr_auc']} | recall@block {perf['recall_at_block']}")
        for a in s["alerts"]:
            print(f"  [{a['severity'].upper()}] {a['message']}")
    if first_alert:
        print(f"\nLive replay: first alert on {first_alert['as_of']} "
              f"({result['december_live_replay']['days_after_shift_started']} days after December began, "
              f"{first_alert['matured_rows']:,} matured transactions)")
        for msg in first_alert["alerts"]:
            print(f"  {msg}")
    else:
        print("\nLive replay: no alert fired before 2026-02-01")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
