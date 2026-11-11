"""
Layer 1, check 2: schema-linking hallucination detection.

Catches the most common *structural* SQL hallucination: an LLM inventing a
table or column name that does not exist in the real database schema
(e.g. writing `e.salary_amount` when the real column is `e.salary`).

This is a *static* check (no DB connection needed) — it works purely off a
parsed AST and a schema description.

What it understands (v2):
- Real tables, with aliases.
- CTEs (`WITH x AS (...)`) and aliased derived tables (`JOIN (SELECT ...) pc`)
  are modelled as *virtual tables* whose columns are the subquery's output
  columns. A `SELECT *` inside one makes its columns unknowable, so any column
  is accepted from it (deliberately lenient, to avoid false positives).
- Select-list aliases (`COUNT(*) AS n ... ORDER BY n`) are valid names.
- SQLite's implicit `rowid` / `oid` / `_rowid_`.
- `SELECT *` and `COUNT(*)` are not column references.

Design decision worth knowing: in SQLite, a double-quoted identifier that
matches no column is silently treated as a *string literal*, so
`SELECT "Nmae" FROM singer` executes and returns the text 'Nmae' for every row.
Execution checks cannot see this; this check does (it flags the unknown column
regardless of quoting). The flip side: `WHERE Country = "France"` (double quotes
used as a string) is also flagged, since strict SQL engines would reject it.

Known limitations (stated, not hidden):
- Aliases are resolved in one flat namespace for the whole query, not per
  scope. If an outer and inner query reuse the same alias for different tables,
  the last one wins, which can cause a false positive or a missed error.
- Unqualified columns are checked against the union of the sources visible to
  their own SELECT (plus outer queries, for correlated subqueries), not
  resolved to one specific table, so an ambiguous-but-wrong unqualified column
  can be under-detected.
- If a query references any unknown table, unqualified columns are not checked
  (they can't be resolved), so the report lists only the unknown table(s) and
  any qualified columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

_WILDCARD = "*"
_IMPLICIT_COLUMNS = {"rowid", "oid", "_rowid_"}


@dataclass
class SchemaLinkResult:
    ok: bool
    parses: bool
    unknown_tables: list[str] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)  # "alias.column" or "column"
    parse_error: str | None = None


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _output_columns(body: exp.Expression) -> set[str]:
    """Output column names of a SELECT / UNION body (lower-cased)."""
    return {name.lower() for name in body.named_selects}


def check_schema_linking(
    sql: str,
    schema: dict[str, set[str]],
    dialect: str = "sqlite",
) -> SchemaLinkResult:
    """
    Args:
        sql: the SQL query to check.
        schema: {table_name: {column_name, ...}} — the REAL schema.
        dialect: sqlglot dialect string (e.g. "sqlite", "postgres").

    Returns:
        SchemaLinkResult with ok=True only if the query parses AND every
        table/column reference resolves against `schema` (plus any CTEs /
        derived tables defined inside the query itself).
    """
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except Exception as e:  # sqlglot raises sqlglot.errors.ParseError
        return SchemaLinkResult(ok=False, parses=False, parse_error=str(e))

    schema_lower = {t.lower(): {c.lower() for c in cols} for t, cols in schema.items()}

    # 1. Virtual tables defined inside the query: CTEs and aliased derived tables.
    virtual: dict[str, set[str]] = {}
    for cte in tree.find_all(exp.CTE):
        if not cte.alias:
            continue
        declared = cte.args.get("alias")
        declared_cols = {c.name.lower() for c in declared.columns} if declared else set()
        virtual[cte.alias.lower()] = declared_cols or _output_columns(cte.this)
    derived_aliases: dict[str, set[str]] = {}
    for sq in tree.find_all(exp.Subquery):
        if sq.alias:
            derived_aliases[sq.alias.lower()] = _output_columns(sq.this)

    # 2. Map every alias visible in the query to the set of columns it exposes.
    alias_to_cols: dict[str, set[str]] = dict(derived_aliases)
    unknown_tables: list[str] = []

    for t in tree.find_all(exp.Table):
        table_name = t.name.lower()
        alias = (t.alias or t.name).lower()
        if table_name in virtual:
            alias_to_cols[alias] = virtual[table_name]
        elif table_name in schema_lower:
            alias_to_cols[alias] = schema_lower[table_name]
        else:
            unknown_tables.append(t.name)

    # 3. Per-SELECT scope: which columns are visible to unqualified references.
    #    A CTE body or a FROM-position derived table sees only its OWN sources
    #    (this stops an invented column that becomes a CTE output name from
    #    vouching for itself); a correlated subquery also sees its outer query.
    local_cols: dict[int, set[str]] = {}

    def _add_source(node: exp.Expression, cols: set[str]) -> None:
        select = node.find_ancestor(exp.Select)
        if select is not None:
            local_cols.setdefault(id(select), set()).update(cols)

    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if name in virtual:
            _add_source(t, virtual[name])
        elif name in schema_lower:
            _add_source(t, schema_lower[name])
    for sq in tree.find_all(exp.Subquery):
        if sq.alias and isinstance(sq.parent, (exp.From, exp.Join)):
            _add_source(sq, derived_aliases[sq.alias.lower()])

    def visible_columns(col: exp.Column) -> tuple[set[str], bool]:
        """(columns visible to this reference, whether any Select scope was found)."""
        visible: set[str] = set()
        found_select = False
        node = col.parent
        while node is not None:
            if isinstance(node, exp.Select):
                found_select = True
                visible |= local_cols.get(id(node), set())
            elif isinstance(node, exp.CTE):
                break
            elif isinstance(node, exp.Subquery) and isinstance(node.parent, (exp.From, exp.Join)):
                break
            node = node.parent
        return visible, found_select

    select_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    global_cols: set[str] = set().union(*alias_to_cols.values()) if alias_to_cols else set()

    # 4. Check every column reference.
    unknown_columns: list[str] = []
    for c in tree.find_all(exp.Column):
        col_name = c.name.lower()
        if not col_name or col_name == _WILDCARD:
            continue

        qualifier = c.table.lower() if c.table else None

        if qualifier:
            cols = alias_to_cols.get(qualifier)
            if cols is None:
                unknown_columns.append(f"{c.table}.{c.name}")
            elif _WILDCARD not in cols and col_name not in cols and col_name not in _IMPLICIT_COLUMNS:
                unknown_columns.append(f"{c.table}.{c.name}")
        else:
            if unknown_tables:
                continue  # can't resolve reliably; the unknown table is already flagged
            visible, found_select = visible_columns(c)
            if not found_select:
                visible = global_cols  # e.g. ORDER BY on a UNION: fall back to all sources
            if _WILDCARD in visible:
                continue  # a SELECT * source makes its columns unknowable; stay lenient
            if col_name in visible or col_name in select_aliases or col_name in _IMPLICIT_COLUMNS:
                continue
            unknown_columns.append(c.name)

    unknown_tables = _dedupe(unknown_tables)
    unknown_columns = _dedupe(unknown_columns)
    ok = not unknown_tables and not unknown_columns
    return SchemaLinkResult(
        ok=ok,
        parses=True,
        unknown_tables=unknown_tables,
        unknown_columns=unknown_columns,
    )
