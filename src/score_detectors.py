"""
Score the detectors (Layer 1 and the two cheap no-gold checks) under three label
policies. Local only: no Groq calls.

    python3 src/score_detectors.py --eval-set eval_set_v3.jsonl --adjudication adjudication_v3.json

Detectors (none of them reads the gold SQL):
  Layer 1       parse / schema linking / execution
  A degenerate  generated result has no rows, or one row of only 0/NULL
  B literal     a string literal is absent from the DB, or present only in other letter case
  A or B        either of the two checks

Label policies (gold is used only to build labels, never as detector input):
  strict          the labels in the eval set: positive = result differs from gold
  adjudicated     positive = real_error; negative = result matched gold, or a difference
                  judged an artifact (gen is right); ambiguous / gold_questionable rows
                  are left out
  informative     adjudicated, minus matches where gold's own answer is empty or all
                  0/NULL (matching an empty answer says little about correctness)

Rows whose generated result equals gold's up to column order or float rounding are
set to `artifact` automatically when the adjudication file has no entry for them,
so the file only needs the rows that need a judgement.

Caveats: the adjudication file is a human-style judgement from reading the SQL, not
a measurement; edit it where you disagree. The `informative` rule was chosen after
seeing the false positives of check A, so report all three policies side by side and
make claims only on a held-out set. With few positives the intervals are wide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnose_mismatches import classify, find_literal, is_degenerate, string_literals
from execution_check import run_query
from layer1_detector import check_query
from run_layer1_on_eval_set import wilson

CATS = {"real_error", "ambiguous", "artifact", "gold_questionable"}
DETECTORS = ["Layer 1", "A degenerate", "B literal", "A or B"]


def rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a (0/0)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.2f} [{lo:.2f}-{hi:.2f}]"


def b_flag(db_path: str, sql: str) -> bool:
    return any(find_literal(db_path, lit)[0] != "exact" for lit in string_literals(sql))


def build_rows(all_rows: list[dict]) -> list[dict]:
    out = []
    for idx, r in enumerate(all_rows):
        if r["is_hallucinated"] is None:
            continue
        gen_rows, _ = run_query(r["db_path"], r["generated_sql"])
        gold_rows, _ = run_query(r["db_path"], r["gold_sql"])
        flags = {
            "Layer 1": check_query(r["generated_sql"], r["db_path"]).flagged_hallucination,
            "A degenerate": gen_rows is not None and is_degenerate(gen_rows),
            "B literal": b_flag(r["db_path"], r["generated_sql"]),
        }
        flags["A or B"] = flags["A degenerate"] or flags["B literal"]
        # Same values in another column order, or equal after rounding floats: not an error.
        shape_only = None
        if r["is_hallucinated"] and gen_rows is not None and gold_rows is not None:
            kind = classify(gen_rows, gold_rows)
            shape_only = kind if kind in {"column_order", "float_rounding"} else None
        out.append(
            {
                "idx": idx,
                "rec": r,
                "flags": flags,
                "gold_empty": gold_rows is not None and is_degenerate(gold_rows),
                "shape_only": shape_only,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--adjudication", type=Path, required=True)
    args = ap.parse_args()

    all_rows = [json.loads(line) for line in open(args.eval_set)]
    adj = {int(k): v for k, v in json.load(open(args.adjudication)).items()}

    bad_cat = {k: v["cat"] for k, v in adj.items() if v["cat"] not in CATS}
    if bad_cat:
        raise SystemExit(f"Unknown categories in adjudication file: {bad_cat}")

    rows = build_rows(all_rows)
    n_auto = 0
    for row in rows:
        if row["rec"]["is_hallucinated"] and row["idx"] not in adj and row["shape_only"]:
            adj[row["idx"]] = {"cat": "artifact", "why": f"auto: {row['shape_only']}"}
            n_auto += 1
    positives = {row["idx"] for row in rows if row["rec"]["is_hallucinated"]}
    if positives != set(adj):
        raise SystemExit(
            f"Adjudication file does not match the eval set. Missing: {sorted(positives - set(adj))}; "
            f"extra: {sorted(set(adj) - positives)}"
        )
    if n_auto:
        print(f"[auto] {n_auto} row(s) not in the adjudication file were set to 'artifact' (column order / float rounding only)")

    def strict(row):
        return row["rec"]["is_hallucinated"]

    def adjudicated(row):
        if not row["rec"]["is_hallucinated"]:
            return False
        return {"real_error": True, "artifact": False}.get(adj[row["idx"]]["cat"])  # else None = left out

    def informative(row):
        y = adjudicated(row)
        return None if (y is False and row["gold_empty"]) else y

    cat_counts: dict[str, int] = {}
    for v in adj.values():
        cat_counts[v["cat"]] = cat_counts.get(v["cat"], 0) + 1
    print(f"{len(rows)} labelled rows; the {len(adj)} labelled hallucinated were adjudicated as {cat_counts}\n")

    for name, policy in [("strict", strict), ("adjudicated", adjudicated), ("informative", informative)]:
        kept = [(row, policy(row)) for row in rows if policy(row) is not None]
        n_pos = sum(1 for _, y in kept if y)
        print(f"--- policy: {name}  (kept {len(kept)}: {n_pos} positive, {len(kept) - n_pos} negative; left out {len(rows) - len(kept)}) ---")
        for det in DETECTORS:
            tp = sum(1 for row, y in kept if y and row["flags"][det])
            fp = sum(1 for row, y in kept if not y and row["flags"][det])
            print(f"  {det:<13} flagged {tp + fp:>3}   precision {rate(tp, tp + fp):<24} recall {rate(tp, n_pos)}")
        print()

    print("--- adjudicated real errors, and which detector flagged each ---")
    for row in rows:
        if adj.get(row["idx"], {}).get("cat") == "real_error":
            short = {"Layer 1": "L1", "A degenerate": "A", "B literal": "B"}
            marks = " ".join(f"{short[d]}:{'Y' if row['flags'][d] else '-'}" for d in DETECTORS[:3])
            print(f"  #{row['idx']:<4}{marks:<26}{adj[row['idx']]['why']}")


if __name__ == "__main__":
    main()
