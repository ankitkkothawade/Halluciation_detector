"""
Layer 1 detector: combine parse + schema-linking + execution checks into a
single hallucination verdict for a generated SQL query.

This layer needs NO labelled training data and NO gold SQL to run — it only
needs the real database schema (or the DB file itself, for execution). Gold
SQL is used later, only in evaluate.py, to SCORE how well this detector's
verdicts line up with actual correctness — never as an input to the
detector itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schema_linking import check_schema_linking
from execution_check import check_execution


@dataclass
class Layer1Verdict:
    flagged_hallucination: bool
    parses: bool
    schema_link_ok: bool
    executes: bool
    reasons: list[str] = field(default_factory=list)


def get_schema_from_sqlite(db_path: str) -> dict[str, set[str]]:
    """Introspect a real schema straight from a SQLite file (no manual typing)."""
    import sqlite3

    schema: dict[str, set[str]] = {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        for (table_name,) in cur.fetchall():
            cur.execute(f'PRAGMA table_info("{table_name}")')
            cols = {row[1] for row in cur.fetchall()}
            schema[table_name] = cols
    finally:
        conn.close()
    return schema


def check_query(sql: str, db_path: str, dialect: str = "sqlite") -> Layer1Verdict:
    schema = get_schema_from_sqlite(db_path)

    link_result = check_schema_linking(sql, schema, dialect=dialect)
    reasons: list[str] = []

    if not link_result.parses:
        reasons.append(f"parse_error: {link_result.parse_error}")
        # Can't meaningfully execute something that doesn't parse.
        return Layer1Verdict(
            flagged_hallucination=True,
            parses=False,
            schema_link_ok=False,
            executes=False,
            reasons=reasons,
        )

    if link_result.unknown_tables:
        reasons.append(f"unknown_tables: {link_result.unknown_tables}")
    if link_result.unknown_columns:
        reasons.append(f"unknown_columns: {link_result.unknown_columns}")

    exec_result = check_execution(sql, db_path)
    if not exec_result.ok:
        reasons.append(f"execution_error: {exec_result.error}")

    flagged = (not link_result.ok) or (not exec_result.ok)

    return Layer1Verdict(
        flagged_hallucination=flagged,
        parses=True,
        schema_link_ok=link_result.ok,
        executes=exec_result.ok,
        reasons=reasons,
    )
