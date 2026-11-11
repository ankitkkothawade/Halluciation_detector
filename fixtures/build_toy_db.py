"""
Builds a tiny toy SQLite database (2 tables, a handful of rows) purely so
the Layer 1 detector can be smoke-tested end-to-end before you plug in the
real Spider/BIRD databases. This is NOT a benchmark — it's a unit-test
fixture with a hand-picked, hand-labelled set of queries (see
tests/toy_queries.py) so we know the ground truth for certain.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "toy_company.db"


def build():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER,
            salary INTEGER,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );
        """
    )

    cur.executemany(
        "INSERT INTO departments (id, name) VALUES (?, ?)",
        [(1, "Engineering"), (2, "Sales"), (3, "Marketing")],
    )

    cur.executemany(
        "INSERT INTO employees (id, name, department_id, salary) VALUES (?, ?, ?, ?)",
        [
            (1, "Alice", 1, 72000),
            (2, "Bob", 1, 68000),
            (3, "Carol", 2, 55000),
            (4, "Dave", 3, 51000),
            (5, "Eve", 2, 59000),
        ],
    )

    conn.commit()
    conn.close()
    print(f"Built toy DB at {DB_PATH}")


if __name__ == "__main__":
    build()
