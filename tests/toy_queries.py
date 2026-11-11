"""
A small, hand-labelled set of (sql, is_hallucinated) pairs against the toy
company schema (departments(id, name), employees(id, name, department_id,
salary)).

`is_hallucinated=True` means: this query has a real, checkable defect
(invented table/column, syntax error, or a runtime execution error) that
Layer 1 should catch. These labels are hand-assigned by construction, not
inferred — they exist purely to sanity-check the detector's logic before
it ever sees real model output or the real Spider/BIRD data.
"""

TOY_CASES: list[tuple[str, bool, str]] = [
    # (sql, is_hallucinated, why)
    (
        "SELECT name, salary FROM employees WHERE department_id = 1",
        False,
        "correct, simple",
    ),
    (
        "SELECT name, salary_amount FROM employees",
        True,
        "hallucinated column: real column is 'salary', not 'salary_amount'",
    ),
    (
        "SELECT * FROM bonuses",
        True,
        "hallucinated table: 'bonuses' does not exist in the schema",
    ),
    (
        "SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.id",
        False,
        "correct join",
    ),
    (
        "SELECT e.name, d.department_name FROM employees e "
        "JOIN departments d ON e.department_id = d.id",
        True,
        "hallucinated column: departments has 'name', not 'department_name'",
    ),
    (
        "SELEKT name FROM employees",
        True,
        "syntax error (misspelled keyword) -> should fail to parse",
    ),
    (
        "SELECT NON_EXISTENT_FUNC(salary) FROM employees",
        True,
        "parses and schema-links fine, but fails at EXECUTION (no such function)",
    ),
    (
        "SELECT department_id, AVG(salary) FROM employees GROUP BY department_id",
        False,
        "correct aggregate query",
    ),
    (
        "SELECT name FROM employees",
        False,
        "correct, unqualified column, single table",
    ),
    (
        "SELECT e.name, b.amount FROM employees e "
        "JOIN bonuses b ON e.id = b.employee_id",
        True,
        "hallucinated table 'bonuses' inside a join",
    ),
    (
        "SELECT * FROM employees",
        False,
        "correct, SELECT * should not be treated as a column hallucination",
    ),
    (
        "SELECT e.name, e.bonus_pct FROM employees e",
        True,
        "hallucinated column 'bonus_pct' does not exist on employees",
    ),
]
