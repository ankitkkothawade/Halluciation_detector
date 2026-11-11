"""
Print gold SQL next to generated SQL (plus the first few result rows of each) for
the rows that need a human decision. Local only: no Groq calls.

    python3 src/show_pairs.py --eval-set eval_set_v3.jsonl > mismatch_pairs.txt

Sections:
  1. every row labelled hallucinated (result differs from gold)
  2. CORRECT rows that check A flags (degenerate result)      -> false positives of A
  3. CORRECT rows that check B flags (literal not in the DB)  -> false positives of B

Each row shows its line number in the eval-set file (0-based), so a decision about
it can be recorded later by that number. Gold SQL is shown here for human review
only; it is never an input to a detector.

Section 1 deliberately leaves out the literal-in-DB notes, so that judging whether
those rows are real errors stays blind to what the detectors flagged. Sections 2
and 3 are for studying the detectors' false positives; they must not be used to
change any label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnose_mismatches import classify, find_literal, is_degenerate, string_literals
from execution_check import run_query


def flat(s: str, width: int = 600) -> str:
    s = " ".join(s.split())
    return s if len(s) <= width else s[: width - 3] + "..."


def show_rows(rows: list[tuple] | None, err: str | None) -> str:
    if rows is None:
        return f"ERROR {flat(err or '', 150)}"
    head = repr(rows[:3])
    if len(head) > 170:
        head = head[:167] + "..."
    return f"{len(rows)} row(s) {head}"


def literal_note(db_path: str, sql: str) -> tuple[bool, str]:
    bad = []
    for lit in string_literals(sql):
        kind, where = find_literal(db_path, lit)
        if kind != "exact":
            bad.append(f"'{lit}' -> {kind}" + (f" ({where})" if where else ""))
    return bool(bad), "; ".join(bad)


def print_row(
    idx: int, tag: str, r: dict, gen_rows, gen_err, gold_rows, gold_err, note: str, show_note: bool = True
) -> None:
    print(f"#{idx} [{tag}] db={r['db_id']}")
    print(f"  Q:    {flat(r['question'])}")
    print(f"  GOLD: {flat(r['gold_sql'])}")
    print(f"  GEN:  {flat(r['generated_sql'])}")
    print(f"  gold -> {show_rows(gold_rows, gold_err)}")
    print(f"  gen  -> {show_rows(gen_rows, gen_err)}")
    if note and show_note:
        print(f"  literals: {note}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.eval_set)]

    halluc, a_fp, b_fp = [], [], []
    for idx, r in enumerate(rows):
        if r["is_hallucinated"] is None:
            continue
        gen_rows, gen_err = run_query(r["db_path"], r["generated_sql"])
        gold_rows, gold_err = run_query(r["db_path"], r["gold_sql"])
        has_bad_literal, note = literal_note(r["db_path"], r["generated_sql"])
        item = (idx, r, gen_rows, gen_err, gold_rows, gold_err, note)

        if r["is_hallucinated"]:
            if gen_err:
                cause = "generated_fails_to_execute"
            elif gold_rows is None:
                cause = "gold_fails"
            else:
                cause = classify(gen_rows, gold_rows)
            halluc.append((cause, item))
        else:
            if gen_rows is not None and is_degenerate(gen_rows):
                a_fp.append(item)
            if has_bad_literal:
                b_fp.append(item)

    print(f"=== 1. Rows labelled hallucinated ({len(halluc)}) ===\n")
    for cause, item in sorted(halluc, key=lambda x: (x[0], x[1][0])):
        print_row(item[0], cause, *item[1:], show_note=False)

    print(f"=== 2. CORRECT rows flagged by check A, degenerate result ({len(a_fp)}) ===\n")
    for item in a_fp:
        print_row(item[0], "A false positive", *item[1:])

    print(f"=== 3. CORRECT rows flagged by check B, literal not in DB ({len(b_fp)}) ===\n")
    for item in b_fp:
        print_row(item[0], "B false positive", *item[1:])


if __name__ == "__main__":
    main()
