"""
Run the Layer 1 detector against a labelled eval set (from generate_eval_set.py /
relabel_eval_set.py) and report precision/recall/F1.

Reading the output:
- "Skipped" rows have label None (truncated, empty, or broken gold): unusable,
  not a detector failure.
- The per-reason table is the key output. Layer 1 can only see STRUCTURAL
  failures (unparseable SQL, invented tables/columns, runtime errors). It is
  expected to miss `result_mismatch` rows: those executed fine and returned the
  wrong answer, which only a semantic layer can catch.
- Against `generated_fails_to_execute` rows the execution check agrees by construction
  (that label IS "fails to execute"). The informative columns are `schema` (the static
  check, independent of execution) and `schema+runs` (flagged by schema linking although
  the query executes, i.e. errors the execution check alone would miss).
- Intervals are 95% Wilson score intervals; with few positives they are wide.
  Quote counts (e.g. "3/38") next to any rate.

Usage:
    python3 src/run_layer1_on_eval_set.py --eval-set eval_set_v3.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import sqrt
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from layer1_detector import check_query


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion k/n (None if n == 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(name: str, k: int, n: int) -> str:
    if n == 0:
        return f"{name}: n/a (0/0)"
    lo, hi = wilson(k, n)
    return f"{name}: {k / n:.3f}  ({k}/{n}, 95% CI {lo:.3f}-{hi:.3f})"


def one_line(sql: str, width: int = 150) -> str:
    s = " ".join(sql.split())
    return s if len(s) <= width else s[: width - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, default=Path("eval_set.jsonl"))
    ap.add_argument(
        "--max-misses",
        type=int,
        default=10,
        help="How many missed hallucinations (false negatives) to print. False positives "
        "are always printed in full. Use a large number to see everything.",
    )
    args = ap.parse_args()

    recs: list[dict] = []
    skipped: Counter = Counter()
    with open(args.eval_set) as f:
        for line in f:
            rec = json.loads(line)
            if rec["is_hallucinated"] is None:
                skipped[rec.get("label_reason", "unlabelled")] += 1
                continue
            rec["_verdict"] = check_query(rec["generated_sql"], rec["db_path"])
            recs.append(rec)

    skip_detail = ", ".join(f"{r}={n}" for r, n in skipped.most_common()) or "none"
    print(f"Scored {len(recs)} examples; skipped {sum(skipped.values())} as unusable ({skip_detail})\n")
    if not recs:
        print("No usable examples to score.")
        return

    y_true = [r["is_hallucinated"] for r in recs]
    y_pred = [r["_verdict"].flagged_hallucination for r in recs]

    false_pos = [r for r in recs if r["_verdict"].flagged_hallucination and not r["is_hallucinated"]]
    false_neg = [r for r in recs if not r["_verdict"].flagged_hallucination and r["is_hallucinated"]]

    if false_pos:
        print(f"--- False positives: Layer 1 flagged a query that matched gold ({len(false_pos)}) ---")
        for r in false_pos:
            print(f"Q: {r['question']}\n   SQL: {one_line(r['generated_sql'])}\n   reasons={r['_verdict'].reasons}")
        print()
    if false_neg:
        shown = false_neg[: args.max_misses]
        print(f"--- Missed hallucinations (showing {len(shown)} of {len(false_neg)}) ---")
        for r in shown:
            print(f"[{r['label_reason']}] Q: {r['question']}\n   SQL: {one_line(r['generated_sql'])}")
        print()

    print("--- Layer 1 by ground-truth label reason, and which check fired ---")
    print(f"{'label_reason':<30}{'rows':>6}{'flagged':>9}{'parse':>7}{'schema':>8}{'exec':>6}{'schema+runs':>13}")
    by_reason: dict[str, list[dict]] = {}
    for r in recs:
        by_reason.setdefault(r["label_reason"], []).append(r)
    for reason, group in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        v = [r["_verdict"] for r in group]
        n_flagged = sum(x.flagged_hallucination for x in v)
        n_parse = sum(not x.parses for x in v)
        n_schema = sum(x.parses and not x.schema_link_ok for x in v)
        n_exec = sum(x.parses and not x.executes for x in v)
        n_silent = sum(x.parses and not x.schema_link_ok and x.executes for x in v)
        print(f"{reason:<30}{len(group):>6}{n_flagged:>9}{n_parse:>7}{n_schema:>8}{n_exec:>6}{n_silent:>13}")
    print("(schema+runs = flagged by schema linking although the query executes)")
    print()
    degraded = [r for r in recs if r.get("hidden")]
    if degraded:
        print(f"--- Rows with a schema item hidden from the model's prompt: {len(degraded)} of {len(recs)} ---")
        for reason, n in Counter(r["label_reason"] for r in degraded).most_common():
            n_flagged = sum(r["_verdict"].flagged_hallucination for r in degraded if r["label_reason"] == reason)
            print(f"  {reason:<28}{n:>5}   flagged {n_flagged}")
        print()

    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    n_pos = sum(y_true)
    print(fmt_rate("Precision", tp, tp + fp))
    print(fmt_rate("Recall   ", tp, n_pos))
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("Confusion matrix (rows=true, cols=pred), order [Hallucinated, Correct]:")
    print(confusion_matrix(y_true, y_pred, labels=[True, False]))


if __name__ == "__main__":
    main()
