"""
Degraded-schema prompts: a stress condition for testing Layer 1 on identifier errors
that a model makes by itself (not ones we inject).

Why: with the full schema in the prompt, no model we tried wrote a query that failed
to execute (0 of about 550 rows), so Layer 1 never had anything to catch. Here the
model is shown a schema with ONE table (or column) removed, picked from the ones the
GOLD query uses, so the model has to guess a name. That is how a schema retriever
that misses a table looks in a real system.

What is and is not used:
- Gold SQL is used only to choose what to hide. It never reaches a detector.
- Labels still come from running the generated SQL against the FULL database and
  comparing with gold's result (relabel_eval_set.py), and Layer 1 still checks against
  the full schema. Only the prompt is degraded.
- The prompt schema is written from the database's own metadata (PRAGMA calls), not
  copied from its CREATE statements, so every row of a degraded run has the same
  format. Primary keys and foreign keys are kept unless they touch something hidden;
  a hidden table's name therefore appears nowhere in the prompt.

Limitations: one hidden item per question; gold-referenced columns are resolved with
a simple alias map (ambiguous unqualified columns are skipped, CTEs and subquery
aliases are not modelled), so a few gold queries yield no candidate and nothing is
hidden for them (recorded as an empty `hidden` list).
"""

from __future__ import annotations

import random
import sqlite3

import sqlglot
from sqlglot import exp


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_schema(db_path) -> dict[str, dict]:
    """{table: {"columns": [(name, type, pk_position)], "fks": [(from_cols, ref_table, to_cols)]}}."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        ]
        schema: dict[str, dict] = {}
        for t in tables:
            columns = [(r[1], r[2] or "", r[5]) for r in conn.execute(f"PRAGMA table_info({_q(t)})")]
            groups: dict[int, list] = {}
            for r in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                groups.setdefault(r[0], []).append(r)  # (id, seq, table, from, to, ...)
            fks = []
            for rows in groups.values():
                rows.sort(key=lambda r: r[1])
                fks.append(([r[3] for r in rows], rows[0][2], [r[4] for r in rows]))
            schema[t] = {"columns": columns, "fks": fks}
        return schema
    finally:
        conn.close()


def build_ddl(db_path, hidden=()) -> str:
    """CREATE TABLE text for the prompt, without the hidden items.

    hidden: list of {"table": t} (hide the whole table) or {"table": t, "column": c}.
    """
    hidden_tables = {h["table"].lower() for h in hidden if "column" not in h}
    hidden_cols = {(h["table"].lower(), h["column"].lower()) for h in hidden if "column" in h}

    def col_hidden(table: str, col: str) -> bool:
        return table.lower() in hidden_tables or (table.lower(), col.lower()) in hidden_cols

    schema = read_schema(db_path)
    statements = []
    for table, info in schema.items():
        if table.lower() in hidden_tables:
            continue
        parts = [f"{_q(n)} {typ}".rstrip() for n, typ, _pk in info["columns"] if not col_hidden(table, n)]
        pk = sorted((c for c in info["columns"] if c[2]), key=lambda c: c[2])
        if pk and not any(col_hidden(table, c[0]) for c in pk):
            parts.append("PRIMARY KEY (" + ", ".join(_q(c[0]) for c in pk) + ")")
        for from_cols, ref_table, to_cols in info["fks"]:
            if any(col_hidden(table, c) for c in from_cols):
                continue
            if ref_table.lower() in hidden_tables:
                continue
            if any(c is not None and col_hidden(ref_table, c) for c in to_cols):
                continue
            ref = _q(ref_table)
            if all(c is not None for c in to_cols):
                ref += "(" + ", ".join(_q(c) for c in to_cols) + ")"
            parts.append("FOREIGN KEY (" + ", ".join(_q(c) for c in from_cols) + ") REFERENCES " + ref)
        statements.append(f"CREATE TABLE {_q(table)} (\n  " + ",\n  ".join(parts) + "\n);")
    return "\n".join(statements)


def gold_references(gold_sql: str, schema: dict[str, dict]) -> tuple[list[str], list[tuple[str, str]]]:
    """Real tables and (table, column) pairs that the gold query uses, in order of appearance."""
    try:
        tree = sqlglot.parse_one(gold_sql, dialect="sqlite")
    except Exception:
        return [], []
    real = {t.lower(): t for t in schema}
    cols_by_table = {t: {c[0].lower(): c[0] for c in info["columns"]} for t, info in schema.items()}
    alias_to_table: dict[str, str] = {}
    tables: list[str] = []
    for node in tree.find_all(exp.Table):
        name = real.get(node.name.lower())
        if name is None:
            continue
        alias_to_table[(node.alias or node.name).lower()] = name
        if name not in tables:
            tables.append(name)
    columns: list[tuple[str, str]] = []
    for col in tree.find_all(exp.Column):
        name = col.name.lower()
        if not name or name == "*":
            continue
        if col.table:
            t = alias_to_table.get(col.table.lower())
            candidates = [t] if t else []
        else:
            candidates = [t for t in tables if name in cols_by_table[t]]
        if len(candidates) != 1 or name not in cols_by_table[candidates[0]]:
            continue
        pair = (candidates[0], cols_by_table[candidates[0]][name])
        if pair not in columns:
            columns.append(pair)
    return tables, columns


def choose_hidden(mode: str, gold_sql: str, db_path, key: str) -> list[dict]:
    """What to hide for one question. Deterministic given `key` (e.g. "42:450")."""
    if mode == "none":
        return []
    if mode not in {"table", "column"}:
        raise ValueError(f"unknown degrade mode: {mode!r}")
    schema = read_schema(db_path)
    tables, columns = gold_references(gold_sql, schema)
    rng = random.Random(key)
    if mode == "table":
        if len(schema) < 2 or not tables:
            return []
        return [{"table": rng.choice(tables)}]
    candidates = [(t, c) for t, c in columns if len(schema[t]["columns"]) > 1]
    if not candidates:
        return []
    table, column = rng.choice(candidates)
    return [{"table": table, "column": column}]
