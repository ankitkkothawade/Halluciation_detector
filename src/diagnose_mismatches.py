"""
Diagnose WHY queries labelled hallucinated disagree with Spider's gold, and test
two cheap checks that need NO gold answer. Local only: no Groq calls.

    python3 src/diagnose_mismatches.py --eval-set eval_set_v3.jsonl

Part 1 -- cause of each mismatch (compares generated vs gold result sets):
  empty_or_zero_result   generated returned no rows, or one row of only 0/NULL
  different_column_count e.g. 'first || last' where gold returns two columns
  float_rounding         equal after rounding floats to 6 places
  column_order           same values, columns in a different order
  row_subset/row_superset  generated filter too strict / too loose
  different_rows         anything else

Part 2 -- two detector ideas, scored on ALL labelled rows (so false positives show):
  A. degenerate result : generated returns no rows, or one row of only 0/NULL
  B. literal not in DB : a string literal in the SQL appears in no column of the
                         database (exactly), or only with different letter case
Neither idea uses the gold SQL. Gold is used only to score them, as everywhere else.

Caveats: B searches every column for the literal, not just the one it is compared
with, so it under-detects. Literals containing '%' are checked as LIKE patterns
(some column must match); bare wildcards and whitespace-only literals are skipped.
Both ideas are measured here, not yet built into Layer 1.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import sqlglot
from sqlglot import exp

from execution_check import connect_readonly, run_query
from layer1_detector import get_schema_from_sqlite


def is_degenerate(rows: list[tuple]) -> bool:
    """No rows, or a single row whose values are all 0 / NULL (e.g. COUNT(*) = 0)."""
    if not rows:
        return True
    return len(rows) == 1 and all(v is None or v == 0 for v in rows[0])


def _rounded(row: tuple) -> tuple:
    return tuple(round(v, 6) if isinstance(v, float) else v for v in row)


def classify(gen_rows: list[tuple], gold_rows: list[tuple]) -> str:
    gen, gold = set(gen_rows), set(gold_rows)
    if is_degenerate(gen_rows) and not is_degenerate(gold_rows):
        return "empty_or_zero_result"
    if not gen_rows or not gold_rows:
        return "different_rows"
    if len(gen_rows[0]) != len(gold_rows[0]):
        return "different_column_count"
    if {_rounded(r) for r in gen} == {_rounded(r) for r in gold}:
        return "float_rounding"
    if {tuple(sorted(map(repr, r))) for r in gen} == {tuple(sorted(map(repr, r))) for r in gold}:
        return "column_order"
    if gen < gold:
        return "row_subset"
    if gen > gold:
        return "row_superset"
    return "different_rows"


def string_literals(sql: str) -> list[str]:
    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return []
    lits = [l.this for l in tree.find_all(exp.Literal) if l.is_string]
    # whitespace-only literals and bare wildcards ('%', '%%') say nothing about the data
    return [s for s in dict.fromkeys(lits) if s.strip() and s.replace("%", "")]


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


_cache: dict[tuple[str, str], tuple[str, str | None]] = {}


def find_literal(db_path: str, literal: str) -> tuple[str, str | None]:
    """('exact', None) | ('case_only', 'table.column') | ('absent', None).

    A literal containing '%' is treated as a LIKE pattern: 'exact' then means some column
    matches the pattern (LIKE already ignores ASCII letter case, so there is no 'case_only').
    """
    key = (db_path, literal)
    if key in _cache:
        return _cache[key]
    schema = get_schema_from_sqlite(db_path)
    conn = connect_readonly(db_path)
    result: tuple[str, str | None] = ("absent", None)
    try:
        if "%" in literal:
            passes = (("LIKE", "", "exact"),)
        else:
            passes = (("=", "", "exact"), ("=", " COLLATE NOCASE", "case_only"))
        for op, collate, label in passes:
            for table, cols in schema.items():
                for col in sorted(cols):
                    try:
                        hit = conn.execute(
                            f"SELECT 1 FROM {_q(table)} WHERE {_q(col)} {op} ?{collate} LIMIT 1", (literal,)
                        ).fetchone()
                    except sqlite3.Error:
                        continue
                    if hit:
                        result = (label, None if label == "exact" else f"{table}.{col}")
                        break
                if result[0] != "absent":
                    break
            if result[0] != "absent":
                break
    finally:
        conn.close()
    _cache[key] = result
    return result


def one_line(s: str, width: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= width else s[: width - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--quiet", action="store_true", help="Skip the per-row listing.")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.eval_set)]
    rows = [r for r in rows if r["is_hallucinated"] is not None]

    causes: Counter = Counter()
    heur_a: Counter = Counter()  # (flagged, hallucinated) -> n
    heur_b: Counter = Counter()
    heur_b_kind: Counter = Counter()
    listing = []

    for r in rows:
        gen_rows, gen_err = run_query(r["db_path"], r["generated_sql"])
        gold_rows, _ = run_query(r["db_path"], r["gold_sql"])
        hallucinated = r["is_hallucinated"]

        found = {lit: find_literal(r["db_path"], lit) for lit in string_literals(r["generated_sql"])}
        bad = {lit: f for lit, f in found.items() if f[0] != "exact"}

        heur_a[(gen_rows is not None and is_degenerate(gen_rows), hallucinated)] += 1
        heur_b[(bool(bad), hallucinated)] += 1
        if bad and hallucinated:
            heur_b_kind.update(f[0] for f in bad.values())

        if hallucinated:
            cause = "generated_fails_to_execute" if gen_err else classify(gen_rows, gold_rows)
            causes[cause] += 1
            note = "; ".join(
                f"'{lit}' -> {kind}" + (f" ({where})" if where else "") for lit, (kind, where) in bad.items()
            )
            listing.append((cause, r["question"], r["generated_sql"], note))

    n_h = sum(v for (f, h), v in heur_a.items() if h)
    n_c = sum(v for (f, h), v in heur_a.items() if not h)
    print(f"{len(rows)} labelled rows: {n_h} hallucinated, {n_c} correct\n")

    print("--- Part 1: why the hallucinated rows differ from gold ---")
    for cause, n in causes.most_common():
        print(f"  {cause:<28} {n}")
    print()

    print("--- Part 2: two no-gold checks, scored against the labels ---")
    print(f"{'check':<28}{'flags hallucinated':>20}{'flags correct (FP)':>20}")
    print(f"{'A. degenerate result':<28}{heur_a[(True, True)]:>14} / {n_h:<5}{heur_a[(True, False)]:>14} / {n_c:<5}")
    print(f"{'B. literal not in DB':<28}{heur_b[(True, True)]:>14} / {n_h:<5}{heur_b[(True, False)]:>14} / {n_c:<5}")
    if heur_b_kind:
        print(f"   B on hallucinated rows, by kind: {dict(heur_b_kind)}")
    print()

    if not args.quiet:
        print("--- Hallucinated rows ---")
        for cause, q, sql, note in sorted(listing, key=lambda x: x[0]):
            print(f"[{cause}] Q: {one_line(q, 100)}\n   SQL: {one_line(sql, 130)}")
            if note:
                print(f"   literals: {note}")


if __name__ == "__main__":
    main()
