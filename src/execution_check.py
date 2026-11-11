"""
Layer 1, check 3: execution check.

The most direct possible signal: does the generated SQL actually run
against the real database? This catches everything schema-linking might
miss (e.g. type errors, malformed joins that parse fine but fail at
runtime) — but note it CANNOT catch a query that executes successfully
and returns a plausible-looking but wrong result (that needs the gold
query for comparison at evaluation time, or the metamorphic-testing layer
at inference time).

`run_query` is the ONE place in the project that executes SQL. Both this
check and the evaluation labeller (relabel_eval_set.py) call it, so "does it
execute?" cannot mean two different things. Two details matter:

- It reads EVERY row (fetchall), not just the first few. A runtime error on a
  late row (e.g. a text value that is not valid UTF-8) only surfaces when that
  row is fetched, so reading a prefix silently under-reports failures.
- It decodes text tolerantly (invalid bytes become U+FFFD). Some Spider
  databases (e.g. wta_1) contain mis-encoded player names; with strict decoding
  the gold query itself "fails", and so does any correct generated query that
  touches those rows. That is dirty data, not a hallucination.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    ok: bool
    row_count: int | None = None
    error: str | None = None
    rows: list[tuple] | None = None


def connect_readonly(db_path: str) -> sqlite3.Connection:
    """Read-only connection, so a hallucinated query can't mutate the validation DB."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    return conn


def run_query(db_path: str, sql: str) -> tuple[list[tuple] | None, str | None]:
    """Execute `sql`; return (all_rows, None) on success or (None, error) on failure."""
    conn = None
    try:
        conn = connect_readonly(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall(), None
    except sqlite3.Error as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        if conn is not None:
            conn.close()


def check_execution(sql: str, db_path: str, row_limit: int = 50) -> ExecutionResult:
    """Execute `sql` against the SQLite database at `db_path`.

    Every row is read (see module docstring); only the first `row_limit` are
    kept in the result.
    """
    rows, error = run_query(db_path, sql)
    if error is not None:
        return ExecutionResult(ok=False, error=error)
    return ExecutionResult(ok=True, row_count=len(rows[:row_limit]), rows=rows[:row_limit])
