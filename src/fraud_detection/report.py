"""Approval report: one Markdown page for the person who decides whether the challenger is promoted.

Everything up to the decision is automated (retrain -> compare); a human approves. The report is
written for an approver who may not be an engineer (e.g. the fraud operations lead): decision first,
plain-language numbers, then the evidence. It is also the audit record of what was approved and why.

Run:  python -m fraud_detection.report        (after retrain and compare)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from fraud_detection.compare import REPORT_PATH as COMPARISON_PATH
from fraud_detection.config import CHALLENGER_DIR, REPORTS_DIR
from fraud_detection.retrain import INFO_FILE

WEAK_RECALL = 0.40     # a segment where the challenger still blocks less than this share of fraud is flagged

# (metric, label, format, which direction is better)
HEADLINE = [
    ("pr_auc", "Ranking quality (PR-AUC)", "ratio", "up"),
    ("recall_at_block", "Fraud blocked (recall at block threshold)", "pct", "up"),
    ("missed_fraud_amount", "Fraud amount missed", "money", "down"),
    ("false_blocks", "Honest customers blocked", "int", "down"),
    ("false_block_rate", "Honest customers blocked (share of honest transactions)", "pct", "down"),
    ("total_cost", "Business cost (missed fraud + R$ per false block)", "money", "down"),
    ("review_queue", "Manual review queue (transactions)", "int", "neutral"),
    ("review_fraud_rate", "Fraud rate inside the review queue", "pct", "up"),
    ("recall_high_medium", "Fraud flagged for block or review", "pct", "up"),
]


def _fmt(value, kind: str) -> str:
    if value is None:
        return "n/a"
    return {"ratio": f"{value:.3f}", "pct": f"{value:.1%}", "money": f"R$ {value:,.0f}",
            "int": f"{value:,}"}[kind]


def _change(v: dict, kind: str, better: str) -> str:
    if v["change"] is None:
        return ""
    sign = "+" if v["change"] > 0 else "−" if v["change"] < 0 else ""
    size = abs(v["change"])
    text = {"ratio": f"{sign}{size:.3f}", "pct": f"{sign}{size * 100:.1f} pts",
            "money": f"{sign}R$ {size:,.0f}", "int": f"{sign}{size:,.0f}"}[kind]
    if better == "neutral" or v["change"] == 0:
        return text
    good = (v["change"] > 0) == (better == "up")
    return f"{text} {'✅' if good else '⚠️'}"


def approver_notes(comparison: dict, info: dict) -> list[str]:
    """Judgment calls the automatic rules do not make: workload, how much the budget bound, limits."""
    o, pol, notes = comparison["overall"], info["policy"], []
    days = max((pd.Timestamp(comparison["holdout"]["to"]) - pd.Timestamp(comparison["holdout"]["from"])).days + 1, 1)
    q = o["review_queue"]
    if q["champion"] and q["challenger"] > 1.25 * q["champion"]:
        notes.append(
            f"**Analyst workload.** The manual review queue grows from {q['champion']:,} to {q['challenger']:,} "
            f"transactions over the {days}-day holdout (about {q['champion'] / days:,.0f} → "
            f"{q['challenger'] / days:,.0f} per day). The share of fraud per reviewed case goes from "
            f"{_fmt(o['review_fraud_rate']['champion'], 'pct')} to "
            f"{_fmt(o['review_fraud_rate']['challenger'], 'pct')}, "
            f"and block-or-review now catches {_fmt(o['recall_high_medium']['challenger'], 'pct')} of fraud "
            f"(was {_fmt(o['recall_high_medium']['champion'], 'pct')}). **Can the team absorb this queue?**")
    weak = [s for s in comparison["segments"] if s["enough_fraud"] and s["recall_at_block"]["challenger"] is not None
            and s["recall_at_block"]["challenger"] < WEAK_RECALL]
    if weak:
        weak.sort(key=lambda s: -s["frauds"])
        listed = ", ".join(f"{s['segment']} = {s['value']} ({_fmt(s['recall_at_block']['challenger'], 'pct')} of "
                           f"{s['frauds']} frauds)" for s in weak[:6])
        notes.append(
            f"**Still weak after retraining.** The challenger blocks less than {WEAK_RECALL:.0%} of fraud in: "
            f"{listed}. "
            "Promotion does not fix these; consider a targeted rule or features for them.")
    if pol.get("false_block_budget") is not None and pol["unconstrained_block_threshold"] != pol["block_threshold"]:
        notes.append(
            f"**Customer-impact budget is binding.** Cost alone would block at {pol['unconstrained_block_threshold']} "
            f"({_fmt(pol['unconstrained_false_block_rate'], 'pct')} of honest customers on validation). The budget "
            f"(champion's {_fmt(pol['champion_validation_false_block_rate'], 'pct')} + "
            f"{(pol['false_block_budget'] - pol['champion_validation_false_block_rate']) * 100:.1f} pts) "
            f"keeps it at {pol['block_threshold']}. Loosening the budget would catch more fraud and block more "
            "honest customers: a business decision about the assumed cost of a false block.")
        if pol["unconstrained_block_threshold"] < 0.05:
            notes.append(
                "**Scores are not calibrated probabilities.** A cost-optimal threshold this low means the challenger "
                "gives new fraud patterns very low scores while still ranking them well. Decisions use thresholds, so "
                "this is safe, but dashboards showing raw scores will read low. Calibration is in the backlog.")
    notes.append(
        f"**Evidence size.** The holdout is {days} days ({comparison['rows']:,} transactions, "
        f"{comparison['frauds']:,} frauds), with labels at least 30 days old. Segment results with fewer than 30 "
        "frauds are shown for information only.")
    return notes


def render(comparison: dict, info: dict) -> str:
    rec, o, t = comparison["recommendation"], comparison["overall"], comparison["thresholds"]
    decision = "PROMOTE the challenger" if rec["decision"] == "promote" else "KEEP the champion"
    ci = comparison["pr_auc_gain_bootstrap"]["ci95"]
    lines = [
        f"# Model promotion review: challenger as of {info['as_of'][:10]}",
        "",
        f"**Automatic recommendation: {decision}.** "
        + ("All promotion rules pass." if not rec["reasons"] else f"{len(rec['reasons'])} rule(s) fail (see below)."),
        "",
        "| | |",
        "|---|---|",
        f"| Champion | {comparison['champion']} |",
        f"| Challenger | {comparison['challenger']} |",
        f"| Holdout (neither model trained on it) | {comparison['holdout']['from'][:10]} to "
        f"{comparison['holdout']['to'][:10]}: {comparison['rows']:,} transactions, {comparison['frauds']:,} frauds |",
        f"| Report generated | {comparison['generated_at']} |",
        "",
        "## Headline: what changes for the business",
        "",
        "Each model is measured at its own decision thresholds, as it would actually run.",
        "",
        "| | Champion | Challenger | Change |",
        "|---|---|---|---|",
    ]
    for metric, label, kind, better in HEADLINE:
        v = o[metric]
        lines.append(f"| {label} | {_fmt(v['champion'], kind)} | {_fmt(v['challenger'], kind)} "
                     f"| {_change(v, kind, better)} |")
    lines += ["", f"The ranking gain's 95% bootstrap interval is {ci[0]:+.3f} to {ci[1]:+.3f}.", ""]

    lines += ["## Promotion rules", "", "| Rule | Result | Limit | |", "|---|---|---|---|"]
    lines += [f"| {c['rule']} | {c['value']} | {c['limit']} | {'✅ pass' if c['passed'] else '❌ FAIL'} |"
              for c in rec["checks"]]

    lines += ["", "## For your judgment", "", "The rules cannot decide these points:", ""]
    lines += [f"- {n}" for n in approver_notes(comparison, info)]

    pol = info["policy"]
    lines += ["", "## Decision thresholds", "", "| | Champion | Challenger |", "|---|---|---|",
              f"| Block if score ≥ | {t['champion']['block']} | {t['challenger']['block']} |",
              f"| Review if score ≥ | {t['champion']['review']} | {t['challenger']['review']} |",
              f"| Chosen on | {t['champion']['chosen_on']} | {t['challenger']['chosen_on']} |"]
    if pol.get("false_block_budget") is not None:
        lines += ["", f"Challenger block threshold: the lowest-cost threshold that blocks at most "
                      f"{_fmt(pol['false_block_budget'], 'pct')} of honest customers on its validation window "
                      f"(it blocks {_fmt(pol['validation_false_block_rate'], 'pct')}). Without that budget it would "
                      f"be {pol['unconstrained_block_threshold']}."]

    reliable = [s for s in comparison["segments"] if s["enough_fraud"] and s["recall_at_block"]["change"] is not None]
    losses = sorted((s for s in reliable if s["recall_at_block"]["change"] < 0),
                    key=lambda s: s["recall_at_block"]["change"])[:5]
    gains = sorted((s for s in reliable if s["recall_at_block"]["change"] > 0),
                   key=lambda s: -s["recall_at_block"]["change"])[:5]
    lines += ["", f"## Segments: fraud blocked (recall), segments with at least "
                  f"{comparison['min_segment_frauds']} frauds", ""]
    for title, group in (("Largest losses", losses), ("Largest gains", gains)):
        lines += [f"**{title}**", ""]
        if not group:
            lines += [f"None: no segment {'lost' if title.endswith('losses') else 'gained'} recall.", ""]
            continue
        lines += ["| Segment | Frauds | Champion | Challenger | Change |", "|---|---|---|---|---|"]
        for s in group:
            r = s["recall_at_block"]
            lines.append(f"| {s['segment']} = {s['value']} | {s['frauds']} | {_fmt(r['champion'], 'pct')} "
                         f"| {_fmt(r['challenger'], 'pct')} | {_change(r, 'pct', 'up')} |")
        lines.append("")
    lines.pop()                                                           # no blank line before the next heading

    lines += ["", "## How the challenger was trained", "", "| Window | Dates | Transactions | Frauds |",
              "|---|---|---|---|"]
    for name, w in info["windows"].items():
        lines.append(f"| {name} | {w['range'][0][:10]} to {w['range'][1][:10]} | {w['rows']:,} | {w['frauds']:,} |")
    rw = info["recent_weighted"]
    lines += ["", f"- Recent days {rw['range'][0][:10]} to {rw['range'][1][:10]} weighted ×{rw['weight']:g} "
                  f"({rw['rows']:,} transactions): fraud patterns move.",
              f"- {info['n_trees']} trees (early stopping on validation; best validation PR-AUC "
              f"{info['best_validation_pr_auc']}).",
              f"- Labels used only up to {info['labelled_until'][:10]} (as of {info['as_of'][:10]}, "
              "labels need 30 days to mature).",
              f"- Code commit `{info['git_commit']}`, training data hash `{info['data_md5']}`."]

    lines += ["", "## Decision", "",
              "- [ ] **Approve**: promote the challenger (it first runs in shadow mode, then replaces the champion)",
              "- [ ] **Reject**: keep the champion",
              "", "Approver: ____________________    Date: ____________", "", "Notes:", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fraud_detection.report", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", type=Path, default=COMPARISON_PATH)
    parser.add_argument("--challenger", type=Path, default=CHALLENGER_DIR)
    parser.add_argument("--out", type=Path, help="default: reports/approval_<as-of date>.md")
    args = parser.parse_args(argv)

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    info = json.loads((args.challenger / INFO_FILE).read_text(encoding="utf-8"))
    out = args.out or REPORTS_DIR / f"approval_{info['as_of'][:10]}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(comparison, info), encoding="utf-8")
    print(f"Recommendation: {comparison['recommendation']['decision'].upper()} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
