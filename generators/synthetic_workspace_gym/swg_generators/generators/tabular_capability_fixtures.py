from __future__ import annotations

import csv
import io
from collections.abc import Iterable


def _csv(rows: Iterable[dict[str, object]], fields: list[str]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build_focused_capability_assets() -> dict[str, object]:
    """Build small fixtures where one semantic operation determines correctness."""

    cases = {
        "active_coercion": {
            "aliases": [],
            "statuses": [
                {
                    "account_id": "A",
                    "effective_at": "2026-01-01T00:00:00Z",
                    "active": "YES",
                }
            ],
            "events": [_event("E1", "A", "2026-02-01T00:00:00Z", "2.00")],
            "expected": [_row("A", 1, 2.0)],
        },
        "fractional_aggregation": {
            "aliases": [],
            "statuses": [
                {
                    "account_id": "A",
                    "effective_at": "2026-01-01T00:00:00Z",
                    "active": "true",
                }
            ],
            "events": [
                _event("E1", "A", "2026-02-01T00:00:00Z", "1.25"),
                _event("E2", "A", "2026-02-02T00:00:00Z", "2.50"),
            ],
            "expected": [_row("A", 2, 3.75)],
        },
        "canonical_identity": {
            "aliases": [{"alias": " legacy-a ", "canonical_id": "A"}],
            "statuses": [
                {
                    "account_id": "A",
                    "effective_at": "2026-01-01T00:00:00Z",
                    "active": "true",
                }
            ],
            "events": [_event("E1", "LEGACY-A", "2026-02-01T00:00:00Z", "2.00")],
            "expected": [_row("A", 1, 2.0)],
        },
        "deduplication": {
            "aliases": [{"alias": "legacy-a", "canonical_id": "A"}],
            "statuses": [
                {
                    "account_id": "A",
                    "effective_at": "2026-01-01T00:00:00Z",
                    "active": "true",
                },
                {
                    "account_id": "legacy-a",
                    "effective_at": "2026-01-01T00:00:00Z",
                    "active": "true",
                },
            ],
            "events": [
                _event(
                    "E1",
                    "legacy-a",
                    "2026-02-01T00:00:00Z",
                    "1.00",
                    updated_at="2026-02-01T00:01:00Z",
                ),
                _event(
                    "E1",
                    "A",
                    "2026-02-01T00:00:00Z",
                    "2.00",
                    updated_at="2026-02-01T00:02:00Z",
                ),
            ],
            "expected": [_row("A", 1, 2.0)],
        },
        "timestamp_normalization": {
            "aliases": [],
            "statuses": [
                {
                    "account_id": "A",
                    "effective_at": "2026-01-01T00:00:00Z",
                    "active": "true",
                }
            ],
            "events": [
                _event(
                    "E1",
                    "A",
                    "2026-02-01T00:00:00Z",
                    "1.00",
                    updated_at="2026-02-01T10:30:00+00:00",
                ),
                _event(
                    "E1",
                    "A",
                    "2026-02-01T00:00:00Z",
                    "2.00",
                    updated_at="2026-02-01T06:00:00-05:00",
                ),
            ],
            "expected": [_row("A", 1, 2.0)],
        },
        "temporal_status_join": {
            "aliases": [],
            "statuses": [
                {
                    "account_id": "A",
                    "effective_at": "2026-01-01T00:00:00Z",
                    "active": "true",
                },
                {
                    "account_id": "A",
                    "effective_at": "2026-03-01T00:00:00Z",
                    "active": "no",
                },
            ],
            "events": [_event("E1", "A", "2026-02-01T00:00:00Z", "2.00")],
            "expected": [_row("A", 1, 2.0)],
        },
    }
    text_assets: dict[str, str] = {}
    json_assets: dict[str, object] = {}
    entries: list[dict[str, str]] = []
    event_fields = [
        "event_id",
        "account_ref",
        "status",
        "occurred_at",
        "updated_at",
        "amount",
    ]
    for capability, case in cases.items():
        root = f"capability_fixtures/{capability}"
        text_assets[f"{root}/events.csv"] = _csv(case["events"], event_fields)
        text_assets[f"{root}/status_history.csv"] = _csv(case["statuses"], ["account_id", "effective_at", "active"])
        json_assets[f"{root}/account_aliases.json"] = case["aliases"]
        expected_path = f"capability_expected/{capability}.json"
        json_assets[expected_path] = case["expected"]
        entries.append(
            {
                "capability": capability,
                "input_dir": root,
                "expected_path": expected_path,
            }
        )
    return {
        "hidden_text_assets": text_assets,
        "hidden_json_assets": json_assets,
        "focused_fixtures": entries,
    }


def _event(
    event_id: str,
    account_ref: str,
    occurred_at: str,
    amount: str,
    *,
    updated_at: str | None = None,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "account_ref": account_ref,
        "status": "posted",
        "occurred_at": occurred_at,
        "updated_at": updated_at or occurred_at,
        "amount": amount,
    }


def _row(account_id: str, event_count: int, total_amount: float) -> dict[str, object]:
    return {
        "account_id": account_id,
        "event_count": event_count,
        "total_amount": total_amount,
    }
