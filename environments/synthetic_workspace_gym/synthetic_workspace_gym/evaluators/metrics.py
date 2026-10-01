from __future__ import annotations

import json
from collections import Counter
from typing import Any


def row_overlap_metrics(expected: Any, actual: Any) -> dict[str, float]:
    if not isinstance(expected, list) or not isinstance(actual, list):
        return {
            "row_precision": 0.0,
            "row_recall": 0.0,
            "row_f1": 0.0,
            "exact_match": 1.0 if actual == expected else 0.0,
        }
    expected_rows = Counter(normalize_row(row) for row in expected)
    actual_rows = Counter(normalize_row(row) for row in actual)
    matches = sum((expected_rows & actual_rows).values())
    precision = matches / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = matches / len(expected) if expected else (1.0 if not actual else 0.0)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return {
        "row_precision": round(precision, 6),
        "row_recall": round(recall, 6),
        "row_f1": round(f1, 6),
        "exact_match": 1.0 if actual == expected else 0.0,
    }


def weighted_match_score(*, output_exists: float, valid_structure: float, metrics: dict[str, float]) -> float:
    if metrics["exact_match"] == 1.0 and output_exists == 1.0 and valid_structure == 1.0:
        return 1.0
    score = (
        0.2 * output_exists
        + 0.2 * valid_structure
        + 0.3 * metrics["row_precision"]
        + 0.2 * metrics["row_recall"]
        + 0.1 * metrics["exact_match"]
    )
    return round(min(1.0, max(0.0, score)), 6)


def baseline_normalized_score(score: float, baseline: float) -> float:
    """Measure semantic progress above the artifact shipped to the agent."""

    score = min(1.0, max(0.0, float(score)))
    baseline = min(1.0, max(0.0, float(baseline)))
    if score >= 1.0:
        return 1.0
    if score <= baseline or baseline >= 1.0:
        return 0.0
    return round((score - baseline) / (1.0 - baseline), 6)


def normalize_row(row: Any) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def row_diff_diagnostics(expected: Any, actual: Any, *, limit: int = 3) -> dict[str, Any]:
    """Return compact, artifact-safe diagnostics for list-of-row mismatches."""

    if not isinstance(expected, list) or not isinstance(actual, list):
        return {"expected_type": type(expected).__name__, "actual_type": type(actual).__name__}

    diagnostics: dict[str, Any] = {
        "expected_schema_keys": sorted(_row_keys(expected)),
        "actual_schema_keys": sorted(_row_keys(actual)),
    }
    first_difference = _first_differing_row(expected, actual)
    if first_difference is not None:
        index, expected_row, actual_row = first_difference
        diagnostics["first_differing_row"] = {
            "index": index,
            "expected": expected_row,
            "actual": actual_row,
        }

    expected_rows = Counter(normalize_row(row) for row in expected)
    actual_rows = Counter(normalize_row(row) for row in actual)
    diagnostics["missing_rows_preview"] = _counter_preview(expected_rows - actual_rows, limit=limit)
    diagnostics["unexpected_rows_preview"] = _counter_preview(actual_rows - expected_rows, limit=limit)
    return diagnostics


def json_field_diff_diagnostics(expected: Any, actual: Any, *, limit: int = 5) -> dict[str, Any]:
    """Return path-level diagnostics for nested JSON contract mismatches."""

    expected_fields = {str(row["path"]): row["value"] for row in flatten_json(expected)}
    actual_fields = {str(row["path"]): row["value"] for row in flatten_json(actual)}
    missing_paths = sorted(set(expected_fields) - set(actual_fields))
    unexpected_paths = sorted(set(actual_fields) - set(expected_fields))
    mismatches = []
    for path in sorted(set(expected_fields) & set(actual_fields)):
        if expected_fields[path] != actual_fields[path]:
            mismatches.append(
                {
                    "path": path,
                    "expected": expected_fields[path],
                    "actual": actual_fields[path],
                }
            )
        if len(mismatches) >= limit:
            break
    return {
        "missing_paths": missing_paths[:limit],
        "unexpected_paths": unexpected_paths[:limit],
        "value_mismatches": mismatches,
    }


def flatten_json(value: Any, *, prefix: str = "$") -> list[dict[str, object]]:
    if isinstance(value, dict):
        if not value:
            return [{"path": prefix, "value": "<empty_object>"}]
        flattened: list[dict[str, object]] = []
        for key in sorted(value):
            flattened.extend(flatten_json(value[key], prefix=f"{prefix}.{key}"))
        return flattened
    if isinstance(value, list):
        if not value:
            return [{"path": prefix, "value": "<empty_list>"}]
        flattened = []
        for index, item in enumerate(value):
            flattened.extend(flatten_json(item, prefix=f"{prefix}[{index}]"))
        return flattened
    return [{"path": prefix, "value": value}]


def _row_keys(rows: list[Any]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            keys.update(str(key) for key in row)
    return keys


def _first_differing_row(expected: list[Any], actual: list[Any]) -> tuple[int, Any, Any] | None:
    for index in range(max(len(expected), len(actual))):
        expected_row = expected[index] if index < len(expected) else None
        actual_row = actual[index] if index < len(actual) else None
        if expected_row != actual_row:
            return index, expected_row, actual_row
    return None


def _counter_preview(rows: Counter[str], *, limit: int) -> list[Any]:
    preview: list[Any] = []
    for row, count in rows.most_common(limit):
        try:
            parsed = json.loads(row)
        except json.JSONDecodeError:
            parsed = row
        if count > 1:
            preview.append({"row": parsed, "count": count})
        else:
            preview.append(parsed)
    return preview
