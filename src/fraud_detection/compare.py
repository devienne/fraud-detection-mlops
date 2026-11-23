"""Champion vs challenger on the holdout: overall, per segment, and a promotion recommendation.

The holdout is the most recent matured window (retrain.retrain_windows), which the challenger never
trained on. The champion must not have seen it either, or the comparison would flatter it: the
command refuses a champion whose training data reaches into the holdout.

Each model is judged at its OWN operating point: the champion with its policy, the challenger with
the thresholds re-fitted on its validation window (retrain.py). That is how each would really run.
PR-AUC (ranking quality) does not depend on thresholds; every other metric does.

Promotion rule (params.yaml -> retrain), ALL must hold:
  1. overall PR-AUC gain >= min_pr_auc_gain, with a 95% paired-bootstrap interval above 0
  2. no segment with enough fraud loses more than max_segment_recall_drop of recall at the block threshold
  3. the share of LEGITIMATE transactions blocked rises by at most max_false_block_rate_increase
     (blocking more is fine when it is fraud; blocking more honest customers is the cost)
  4. the business cost (missed fraud amount + false_block_cost per legitimate block) does not rise:
     the quantity the thresholds are chosen to minimise

Run:  python -m fraud_detection.compare --champion eval      (after `python -m fraud_detection.retrain`)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_detection.config import CHALLENGER_DIR, POLICY_PATH, REPORTS_DIR, Params, load_params
from fraud_detection.features import make_features
from fraud_detection.monitoring import monitoring_frame, performance
from fraud_detection.policy import Policy
from fraud_detection.retrain import (
    INFO_FILE,
    MODEL_FILE,
    POLICY_FILE,
    champion_artifacts,
    retrain_windows,
    score_with,
)

REPORT_PATH = REPORTS_DIR / "champion_vs_challenger.json"
METRICS = ("pr_auc", "recall_at_block", "precision_at_block", "block_rate", "false_block_rate",
           "missed_fraud_amount", "false_blocks", "total_cost",
           "review_queue", "review_rate", "review_fraud_rate", "recall_high_medium")


def model_metrics(y: np.ndarray, scores: np.ndarray, policy: Policy, amounts: np.ndarray) -> dict:
    """monitoring.performance (same definitions everywhere) + customer impact, business cost, analyst workload."""
    out = performance(y, scores, policy)
    legit, blocked = y == 0, policy.decide(scores)
    out["false_block_rate"] = round(float(blocked[legit].mean()), 4) if legit.any() else None
    out["missed_fraud_amount"] = round(float(amounts[(y == 1) & ~blocked].sum()), 2)
    out["false_blocks"] = int((legit & blocked).sum())
    out["total_cost"] = round(out["missed_fraud_amount"] + policy.false_block_cost * out["false_blocks"], 2)
    bands = policy.band(scores)
    review, flagged = bands == "Medium", bands != "Low"
    out["review_queue"] = int(review.sum())                                      # analyst workload
    out["review_rate"] = round(float(review.mean()), 4) if len(y) else None
    out["review_fraud_rate"] = round(float(y[review].mean()), 4) if review.any() else None
    out["recall_high_medium"] = round(float(y[flagged].sum() / y.sum()), 4) if y.sum() else None
    return out


def _side_by_side(a: dict, b: dict) -> dict:
    return {m: {"champion": a[m], "challenger": b[m],
                "change": None if a[m] is None or b[m] is None else round(b[m] - a[m], 4)} for m in METRICS}


# --------------------------------------------------------------------------- comparison
def bootstrap_pr_auc_gain(y: np.ndarray, champion: np.ndarray, challenger: np.ndarray, samples: int,
                          seed: int = 0) -> dict:
    """Paired bootstrap of PR-AUC(challenger) - PR-AUC(champion): is the gain more than noise?"""
    rng = np.random.default_rng(seed)
    n, gains = len(y), []
    for _ in range(samples):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        gains.append(average_precision_score(y[idx], challenger[idx]) - average_precision_score(y[idx], champion[idx]))
    low, high = np.percentile(gains, [2.5, 97.5])
    return {"ci95": [round(float(low), 4), round(float(high), 4)], "samples": len(gains)}


def segment_comparison(frame: pd.DataFrame, y: np.ndarray, amounts: np.ndarray, champion: np.ndarray,
                       challenger: np.ndarray, policies: tuple[Policy, Policy], segments: Sequence[str],
                       min_frauds: int) -> list[dict]:
    """Every metric of both models (each with its own policy) in every segment value."""
    rows = []
    for col in segments:
        for value, idx in frame.groupby(col, dropna=False).groups.items():
            pos = frame.index.get_indexer(idx)
            champ = model_metrics(y[pos], champion[pos], policies[0], amounts[pos])
            chall = model_metrics(y[pos], challenger[pos], policies[1], amounts[pos])
            rows.append({"segment": col, "value": str(value), "rows": champ["rows"], "frauds": champ["frauds"],
                         "enough_fraud": champ["frauds"] >= min_frauds, **_side_by_side(champ, chall)})
    return rows


def recommend(overall: dict, gain_ci: dict, segments: list[dict], params: Params) -> dict:
    """Apply the promotion rule: every check with its value and limit, the failures, and segment regressions."""
    r = params.retrain
    gain = overall["pr_auc"]["change"]
    false_blocks = overall["false_block_rate"]["change"]
    cost = overall["total_cost"]
    regressions = [s for s in segments if s["enough_fraud"] and s["recall_at_block"]["change"] is not None
                   and s["recall_at_block"]["change"] < -r.max_segment_recall_drop]
    checks = [
        {"rule": "PR-AUC gain", "value": f"{gain:+.4f}", "limit": f">= {r.min_pr_auc_gain:+.4f}",
         "passed": gain >= r.min_pr_auc_gain,
         "failure": f"PR-AUC gain {gain:+.4f} is below the required {r.min_pr_auc_gain:+.4f}"},
        {"rule": "PR-AUC gain is not noise (95% bootstrap interval)", "value": str(gain_ci["ci95"]),
         "limit": "lower bound > 0", "passed": gain_ci["ci95"][0] > 0,
         "failure": f"PR-AUC gain is not significant: 95% interval {gain_ci['ci95']} includes 0"},
        {"rule": "Honest customers blocked (false-block rate change)", "value": f"{false_blocks:+.2%}",
         "limit": f"<= {r.max_false_block_rate_increase:+.2%}",
         "passed": false_blocks <= r.max_false_block_rate_increase,
         "failure": f"legitimate transactions blocked rise {false_blocks:+.2%} "
                    f"(limit {r.max_false_block_rate_increase:+.2%}): more honest customers blocked"},
        {"rule": "Business cost (missed fraud + false blocks)",
         "value": f"R$ {cost['champion']:,.0f} -> R$ {cost['challenger']:,.0f}", "limit": "must not rise",
         "passed": cost["change"] <= 0,
         "failure": f"business cost rises R$ {cost['champion']:,.0f} -> R$ {cost['challenger']:,.0f}"},
        {"rule": f"No segment loses more than {r.max_segment_recall_drop:.0%} recall",
         "value": ", ".join(f"{s['segment']}={s['value']}" for s in regressions) or "none",
         "limit": f"segments with >= {params.monitoring.min_segment_frauds} frauds", "passed": not regressions,
         "failure": "; ".join(f"segment {s['segment']}={s['value']} loses recall {s['recall_at_block']['champion']:.0%}"
                              f" -> {s['recall_at_block']['challenger']:.0%}" for s in regressions)},
    ]
    reasons = [c["failure"] for c in checks if not c["passed"]]
    return {"decision": "promote" if not reasons else "keep_champion", "reasons": reasons,
            "segment_regressions": [f"{s['segment']}={s['value']}" for s in regressions],
            "checks": [{k: c[k] for k in ("rule", "value", "limit", "passed")} for c in checks]}


def compare_models(raw: pd.DataFrame, y: np.ndarray, champion: np.ndarray, challenger: np.ndarray,
                   champion_policy: Policy, challenger_policy: Policy, params: Params) -> dict:
    """Full comparison of two score vectors on the same labelled rows, each model with its own policy."""
    frame = monitoring_frame(raw, params.features.columns, params.data.timestamp_col)
    amounts = raw["valor_transacao"].to_numpy(dtype=float)
    champion, challenger = np.asarray(champion, dtype=float), np.asarray(challenger, dtype=float)
    policies = (champion_policy, challenger_policy)
    overall = _side_by_side(model_metrics(y, champion, champion_policy, amounts),
                            model_metrics(y, challenger, challenger_policy, amounts))
    gain_ci = bootstrap_pr_auc_gain(y, champion, challenger, params.retrain.bootstrap_samples)
    segments = segment_comparison(frame, y, amounts, champion, challenger, policies, params.monitoring.segments,
                                  params.monitoring.min_segment_frauds)
    thresholds = {name: {"block": p.block_threshold, "review": round(p.review_threshold, 4), "chosen_on": p.chosen_on}
                  for name, p in zip(("champion", "challenger"), policies, strict=True)}
    return {"rows": int(len(y)), "frauds": int(y.sum()), "min_segment_frauds": params.monitoring.min_segment_frauds,
            "thresholds": thresholds, "overall": overall,
            "pr_auc_gain_bootstrap": gain_ci, "segments": segments,
            "recommendation": recommend(overall, gain_ci, segments, params)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fraud_detection.compare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--champion", choices=["released", "eval"], default="released",
                        help="'eval' simulates the model that was in production before December 2025")
    parser.add_argument("--challenger", type=Path, default=CHALLENGER_DIR)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--same-policy", action="store_true",
                        help="judge the challenger with the champion's thresholds (to isolate the model's effect)")
    args = parser.parse_args(argv)

    from fraud_detection.data import load_history

    params = load_params()
    ts_col, cols = params.data.timestamp_col, params.features.columns
    history = load_history(params)
    info = json.loads((args.challenger / INFO_FILE).read_text(encoding="utf-8"))
    holdout = retrain_windows(history[ts_col], pd.Timestamp(info["as_of"]), params)["masks"]["holdout"]
    rows = history[holdout]

    champ = champion_artifacts(args.champion, history, params)
    champ_desc = champ["description"]
    if champ["seen_until"] >= rows[ts_col].min():
        print(f"Refusing: the champion ({champ_desc}) learned from data up to {champ['seen_until']}, which overlaps "
              f"the holdout starting {rows[ts_col].min()}. Its holdout score would be flattered.")
        return 1

    champion = score_with(champ, rows, params)
    challenger = lgb.Booster(model_file=str(args.challenger / MODEL_FILE)).predict(
        make_features(rows, info["features"]["category_levels"], cols, ts_col))
    challenger_policy_path = args.challenger / POLICY_FILE
    if args.same_policy or not challenger_policy_path.exists():
        challenger_policy_path = POLICY_PATH          # judge the challenger with the champion's thresholds
    result = compare_models(rows, rows[params.data.target].to_numpy(), champion, challenger,
                            champ["policy"], Policy.load(challenger_policy_path), params)
    result = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "holdout": {"from": str(rows[ts_col].min()), "to": str(rows[ts_col].max())},
              "champion": champ_desc, "challenger": f"challenger as of {info['as_of']} ({info['n_trees']} trees)",
              **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Holdout {result['holdout']['from'][:10]} -> {result['holdout']['to'][:10]}: "
          f"{result['rows']:,} rows, {result['frauds']:,} frauds")
    print(f"  champion  : {champ_desc}\n  challenger: {result['challenger']}")
    for name, t in result["thresholds"].items():
        print(f"  {name:10s} thresholds: block {t['block']}, review {t['review']}  ({t['chosen_on']})")
    for m, v in result["overall"].items():
        print(f"  {m:20s} {v['champion']!s:>11} -> {v['challenger']!s:>11}  ({v['change']:+,.4f})")
    print(f"  PR-AUC gain 95% interval: {result['pr_auc_gain_bootstrap']['ci95']}")
    moved = sorted((s for s in result["segments"] if s["enough_fraud"]),
                   key=lambda s: s["recall_at_block"]["change"] or 0)
    print("  recall at block, segments with the largest changes:")
    for s in [*moved[:3], *moved[-3:]]:
        r = s["recall_at_block"]
        print(f"    {s['segment']}={s['value']:<16} {r['champion']:.0%} -> {r['challenger']:.0%} "
              f"({s['frauds']} frauds)")
    rec = result["recommendation"]
    print(f"RECOMMENDATION: {rec['decision'].upper()}")
    for reason in rec["reasons"]:
        print(f"  - {reason}")
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
