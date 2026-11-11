"""
Evaluation harness: scores the Layer 1 detector's verdicts against
ground-truth labels.

IMPORTANT — separation of concerns:
- The detector (layer1_detector.check_query) never sees a ground-truth
  label or the gold SQL. It only sees the generated SQL + the real schema
  (+ the DB, for execution).
- Ground truth is used ONLY here, to measure precision/recall/F1 of the
  detector's verdicts. This is what makes it a fair "does this
  ground-truth-free method work" validation, not a leak-y evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from layer1_detector import check_query


@dataclass
class CaseResult:
    sql: str
    true_label: bool
    predicted_label: bool
    reasons: list[str]
    why: str


def run_evaluation(cases: list[tuple[str, bool, str]], db_path: str) -> list[CaseResult]:
    results: list[CaseResult] = []
    for sql, true_label, why in cases:
        verdict = check_query(sql, db_path)
        results.append(
            CaseResult(
                sql=sql,
                true_label=true_label,
                predicted_label=verdict.flagged_hallucination,
                reasons=verdict.reasons,
                why=why,
            )
        )
    return results


def report(results: list[CaseResult]) -> None:
    y_true = [r.true_label for r in results]
    y_pred = [r.predicted_label for r in results]

    print(f"{'SQL':<70} {'true':<6} {'pred':<6} {'match':<6}")
    print("-" * 100)
    for r in results:
        match = "OK" if r.true_label == r.predicted_label else "MISS"
        sql_short = (r.sql[:67] + "...") if len(r.sql) > 70 else r.sql
        print(f"{sql_short:<70} {str(r.true_label):<6} {str(r.predicted_label):<6} {match:<6}")
        if match == "MISS":
            print(f"    -> why labelled this way: {r.why}")
            print(f"    -> detector reasons: {r.reasons}")

    print()
    p = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])

    print(f"Precision: {p:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    print("Confusion matrix (rows=true, cols=pred), order [Hallucinated, Correct]:")
    print(cm)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from tests.toy_queries import TOY_CASES  # type: ignore

    db_path = str(project_root / "fixtures" / "toy_company.db")
    results = run_evaluation(TOY_CASES, db_path)
    report(results)