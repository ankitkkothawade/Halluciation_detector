"""
Derive ground-truth labels for an eval set produced by generate_eval_set.py.
Purely local (re-executes gold + generated SQL; no Groq calls), and also
imported by generate_eval_set.py so there is exactly one labelling rule.

Label rules, checked in this order:
  model stopped at its token limit  -> None   (truncated_generation)
  gold fails to execute             -> None   (gold_fails)
  generated SQL empty/whitespace    -> None   (empty_generation)
  generated fails, gold executes    -> True   (generated_fails_to_execute)
  both execute, result sets differ  -> True   (result_mismatch)
  both execute, result sets equal   -> False  (match)

Why truncation is excluded: a query cut off by max_completion_tokens is an
infrastructure failure, not a model hallucination. Counting it as
"hallucinated" inflates the positive class with rows that any parser flags,
which flatters (or distorts) the detector being scored. Rows from files
written before `finish_reason` was recorded have no such field and fall
through to the remaining rules; regenerate them to get truncation excluded.

Caveat for the write-up: "result_mismatch" means "disagrees with Spider's gold
result set" (compared as sets of row-tuples, so duplicates/order are ignored
but column order matters). That is a proxy for hallucination, not proof of one.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from execution_check import run_query


def label_row(rec: dict) -> tuple[bool | None, str]:
    if rec.get("finish_reason") == "length":
        return None, "truncated_generation"
    gold_rows, gold_err = run_query(rec["db_path"], rec["gold_sql"])
    if gold_err:
        return None, "gold_fails"
    if not rec["generated_sql"].strip():
        return None, "empty_generation"
    gen_rows, gen_err = run_query(rec["db_path"], rec["generated_sql"])
    if gen_err:
        return True, "generated_fails_to_execute"
    if set(gen_rows) != set(gold_rows):
        return True, "result_mismatch"
    return False, "match"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    reasons: Counter = Counter()
    dbs: set[str] = set()
    with open(args.in_path) as fin, open(args.out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            label, reason = label_row(rec)
            rec["is_hallucinated"] = label
            rec["label_reason"] = reason
            reasons[reason] += 1
            dbs.add(rec["db_id"])
            fout.write(json.dumps(rec) + "\n")

    print(f"Relabelled {sum(reasons.values())} rows across {len(dbs)} databases -> {args.out}")
    for reason, n in reasons.most_common():
        print(f"  {reason:<28} {n}")


if __name__ == "__main__":
    main()
