from __future__ import annotations

import csv
import io
import json
import random
from textwrap import dedent, indent

from swg_generators.schemas import EnvironmentSpec


def _csv_text(rows: list[dict[str, object]], fields: list[str]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _normalize(value: object) -> str:
    return str(value).strip().upper()


def _coerce_active(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "active"}


def _resolved_aliases(rows: list[dict[str, str]]) -> dict[str, str]:
    direct = {_normalize(row["alias"]): _normalize(row["canonical_id"]) for row in rows}
    resolved: dict[str, str] = {}
    for alias in direct:
        current = alias
        seen: set[str] = set()
        while current in direct:
            if current in seen:
                raise ValueError("alias cycle")
            seen.add(current)
            current = direct[current]
        resolved[alias] = current
    return resolved


def _expected(
    events: list[dict[str, object]],
    accounts: list[dict[str, object]],
    aliases: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    mapping = _resolved_aliases(aliases)

    def resolve(value: object) -> str:
        current = _normalize(value)
        return mapping.get(current, current)

    snapshots: dict[str, list[dict[str, object]]] = {}
    for row in accounts:
        snapshots.setdefault(resolve(row["account_id"]), []).append(row)
    for rows in snapshots.values():
        rows.sort(key=lambda row: str(row["effective_at"]))

    deduped: dict[tuple[str, str], dict[str, object]] = {}
    for row in events:
        canonical = resolve(row["account_ref"])
        key = (canonical, str(row["event_id"]))
        current = deduped.get(key)
        if current is None or str(row["updated_at"]) > str(current["updated_at"]):
            deduped[key] = {**row, "canonical_id": canonical}

    grouped: dict[str, dict[str, object]] = {}
    for row in deduped.values():
        if _normalize(row["status"]) != "POSTED":
            continue
        candidates = [
            snapshot
            for snapshot in snapshots.get(str(row["canonical_id"]), [])
            if str(snapshot["effective_at"]) <= str(row["occurred_at"])
        ]
        if not candidates or not _coerce_active(candidates[-1]["active"]):
            continue
        account_id = str(row["canonical_id"])
        entry = grouped.setdefault(
            account_id,
            {"account_id": account_id, "event_count": 0, "total_amount": 0.0},
        )
        entry["event_count"] = int(entry["event_count"]) + 1
        entry["total_amount"] = round(float(entry["total_amount"]) + float(row["amount"]), 2)
    return sorted(grouped.values(), key=lambda row: str(row["account_id"])), mapping


def _solution_script(*, compositional: bool) -> str:
    mapping_source = (
        'mapping = json.loads((workspace / "artifacts" / "account_map.json").read_text(encoding="utf-8"))'
        if compositional
        else dedent(
            """
            with (workspace / "data" / "aliases.csv").open(encoding="utf-8", newline="") as stream:
                alias_rows = list(csv.DictReader(stream))
            direct = {normalize(row["alias"]): normalize(row["canonical_id"]) for row in alias_rows}
            mapping = {}
            for alias in direct:
                current = alias
                seen = set()
                while current in direct:
                    if current in seen:
                        raise ValueError("alias cycle")
                    seen.add(current)
                    current = direct[current]
                mapping[alias] = current
            """
        ).strip()
    )
    source = dedent(
        """
        from __future__ import annotations

        import csv
        import json
        from pathlib import Path

        workspace = Path(__file__).resolve().parent


        def normalize(value: object) -> str:
            return str(value).strip().upper()


        def resolve(value: object, mapping: dict[str, str]) -> str:
            normalized = normalize(value)
            return mapping.get(normalized, normalized)


        def active(value: object) -> bool:
            return str(value).strip().casefold() in {"1", "true", "yes", "active"}


        def main() -> None:
            __MAPPING_SOURCE__
            accounts = json.loads((workspace / "data" / "accounts.json").read_text(encoding="utf-8"))
            with (workspace / "data" / "events.csv").open(encoding="utf-8", newline="") as stream:
                events = list(csv.DictReader(stream))
            snapshots: dict[str, list[dict[str, object]]] = {}
            for row in accounts:
                snapshots.setdefault(resolve(row["account_id"], mapping), []).append(row)
            for rows in snapshots.values():
                rows.sort(key=lambda row: str(row["effective_at"]))
            deduped = {}
            for row in events:
                canonical = resolve(row["account_ref"], mapping)
                key = (canonical, row["event_id"])
                if key not in deduped or row["updated_at"] > deduped[key]["updated_at"]:
                    deduped[key] = {**row, "canonical_id": canonical}
            grouped = {}
            for row in deduped.values():
                if normalize(row["status"]) != "POSTED":
                    continue
                candidates = [
                    item for item in snapshots.get(row["canonical_id"], [])
                    if item["effective_at"] <= row["occurred_at"]
                ]
                if not candidates or not active(candidates[-1]["active"]):
                    continue
                account_id = row["canonical_id"]
                entry = grouped.setdefault(account_id, {"account_id": account_id, "event_count": 0, "total_amount": 0.0})
                entry["event_count"] += 1
                entry["total_amount"] = round(entry["total_amount"] + float(row["amount"]), 2)
            output = sorted(grouped.values(), key=lambda row: row["account_id"])
            target = workspace / "outputs" / "account_report.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


        if __name__ == "__main__":
            main()
        """
    ).strip()
    return source.replace("    __MAPPING_SOURCE__", indent(mapping_source, "    ")) + "\n"


def build_account_event_reconciliation_scenario(
    rng: random.Random,
    spec: EnvironmentSpec,
    *,
    compositional: bool,
) -> dict[str, object]:
    base = round(rng.uniform(17.0, 48.0), 2)
    accounts = [
        {
            "account_id": " A-100 ",
            "effective_at": "2026-01-01T00:00:00Z",
            "active": "yes",
        },
        {"account_id": "a-100", "effective_at": "2026-03-01T00:00:00Z", "active": "no"},
        {"account_id": "B-200", "effective_at": "2026-01-15T00:00:00Z", "active": True},
        {"account_id": "C-300", "effective_at": "2026-02-01T00:00:00Z", "active": "1"},
    ]
    aliases = [
        {"alias": " legacy-alpha ", "canonical_id": "acct-alpha"},
        {"alias": "ACCT-ALPHA", "canonical_id": "a-100"},
        {"alias": " beta ", "canonical_id": " b-200 "},
        {"alias": "old-charlie", "canonical_id": "C-300"},
    ]
    events: list[dict[str, object]] = [
        {
            "event_id": "E-1",
            "account_ref": "LEGACY-ALPHA",
            "status": " posted ",
            "occurred_at": "2026-02-12T10:00:00Z",
            "updated_at": "2026-02-12T10:05:00Z",
            "amount": f"{base:.2f}",
        },
        {
            "event_id": "E-1",
            "account_ref": " acct-alpha ",
            "status": "POSTED",
            "occurred_at": "2026-02-12T10:00:00Z",
            "updated_at": "2026-02-12T11:05:00Z",
            "amount": f"{base + 2.35:.2f}",
        },
        {
            "event_id": "E-2",
            "account_ref": "legacy-alpha",
            "status": "posted",
            "occurred_at": "2026-03-12T10:00:00Z",
            "updated_at": "2026-03-12T10:05:00Z",
            "amount": "99.75",
        },
        {
            "event_id": "E-3",
            "account_ref": " BETA",
            "status": "Posted",
            "occurred_at": "2026-02-12T10:00:00Z",
            "updated_at": "2026-02-12T10:05:00Z",
            "amount": "13.40",
        },
        {
            "event_id": "E-4",
            "account_ref": "old-charlie",
            "status": "pending",
            "occurred_at": "2026-02-15T10:00:00Z",
            "updated_at": "2026-02-15T10:05:00Z",
            "amount": "7.25",
        },
        {
            "event_id": "E-5",
            "account_ref": "unknown-account",
            "status": "posted",
            "occurred_at": "2026-02-15T10:00:00Z",
            "updated_at": "2026-02-15T10:05:00Z",
            "amount": "88.00",
        },
        {
            "event_id": "E-6",
            "account_ref": "C-300",
            "status": "posted",
            "occurred_at": "2026-02-15T10:00:00Z",
            "updated_at": "2026-02-15T10:05:00Z",
            "amount": "4.125",
        },
    ]
    rng.shuffle(events)
    duplicate_chain = sorted(
        [row for row in events if row["event_id"] == "E-1"],
        key=lambda row: str(row["updated_at"]),
    )
    events = duplicate_chain + [row for row in events if row["event_id"] != "E-1"]
    expected, mapping = _expected(events, accounts, aliases)
    files = {
        "data/events.csv": _csv_text(
            events,
            [
                "event_id",
                "account_ref",
                "status",
                "occurred_at",
                "updated_at",
                "amount",
            ],
        ),
        "data/accounts.json": json.dumps(accounts, indent=2, sort_keys=True) + "\n",
        "data/aliases.csv": _csv_text(aliases, ["alias", "canonical_id"]),
        "notes/legacy_export.md": "Historical exports deduplicated before identity resolution; that ordering is obsolete.\n",
        "notes/old_account_flags.md": "The numeric values 0 and 1 were once the only supported account flags.\n",
        "archive/example_report.json": "[]\n",
    }
    correct_script = _solution_script(compositional=compositional)
    defect_specs = [
        (
            "deduplicate_before_identity_resolution",
            'key = (canonical, row["event_id"])',
            'key = (normalize(row["account_ref"]), row["event_id"])',
        ),
        (
            "file_order_wins_over_updated_at",
            'if key not in deduped or row["updated_at"] > deduped[key]["updated_at"]:',
            "if key not in deduped:",
        ),
        (
            "latest_snapshot_ignores_event_time",
            'if item["effective_at"] <= row["occurred_at"]',
            "if True",
        ),
        (
            "unsafe_active_coercion",
            'return str(value).strip().casefold() in {"1", "true", "yes", "active"}',
            "return value is True",
        ),
        (
            "fractional_amount_truncation",
            'float(row["amount"]), 2)',
            'float(int(float(row["amount"]))), 2)',
        ),
    ]
    bugs: list[dict[str, object]] = []
    buggy_script = correct_script
    for label, old, new in defect_specs:

        def apply(content: str, *, old: str = old, new: str = new, label: str = label) -> str:
            updated = content.replace(old, new, 1)
            if updated == content:
                raise ValueError(f"tabular defect {label!r} did not modify transform.py")
            return updated

        bugs.append({"label": label, "target_path": "transform.py", "apply": apply})
        buggy_script = apply(buggy_script)
    files["transform.py"] = buggy_script
    reference_files = {
        "transform.py": correct_script,
        "outputs/account_report.json": json.dumps(expected, indent=2, sort_keys=True) + "\n",
    }
    required_artifacts: list[dict[str, str]] = []
    composition_spec: dict[str, object] = {}
    if compositional:
        reference_files["artifacts/account_map.json"] = json.dumps(mapping, indent=2, sort_keys=True) + "\n"
        direct_mapping = {_normalize(row["alias"]): _normalize(row["canonical_id"]) for row in aliases}
        files["artifacts/account_map.json"] = json.dumps(direct_mapping, indent=2, sort_keys=True) + "\n"
        required_artifacts = [
            {
                "path": "artifacts/account_map.json",
                "expected_path": "expected_account_map.json",
                "capability": "identity_resolution",
            }
        ]
        composition_spec = {
            "stages": [
                {
                    "stage_id": "resolve_identities",
                    "required_inputs": ["data/aliases.csv"],
                    "produced_artifacts": ["artifacts/account_map.json"],
                    "capability": "identity_resolution",
                },
                {
                    "stage_id": "build_report",
                    "required_inputs": [
                        "artifacts/account_map.json",
                        "data/events.csv",
                        "data/accounts.json",
                    ],
                    "produced_artifacts": ["outputs/account_report.json"],
                    "capability": "integration",
                },
            ],
            "dependencies": [["resolve_identities", "build_report"]],
            "stage_count": 2,
            "downstream_consumes_upstream_artifact": True,
        }
    return {
        "scenario_id": "account_event_reconciliation",
        "title": "Account Event Reconciliation",
        "description": "Reconcile event identities against versioned account state and produce the contracted account summary.",
        "output_path": "outputs/account_report.json",
        "input_files": ["data/events.csv", "data/accounts.json", "data/aliases.csv"],
        "operations": [],
        "output_contract": [
            "Implement `transform.py`; running it must write a JSON array to `outputs/account_report.json`.",
            "Identity aliases are case/whitespace insensitive and may form chains.",
            "Resolve identity before deduplicating an event; keep the greatest `updated_at` value.",
            "Join the account state effective at the event time, then exclude inactive, unresolved, and non-posted events.",
            "Preserve fractional amounts, round totals to two decimals, and sort by normalized `account_id`.",
        ],
        "hints": [],
        "structure": {
            "task_type": "temporal_identity_reconciliation",
            "input_shape": "csv_json_csv",
            "dependency_depth": 8,
            "distractor_count": 3,
            "hidden_capability_count": 6,
            "output_style": "sorted_row_summary",
        },
        "files": files,
        "expected_output": expected,
        "reference_solution_files": reference_files,
        "correct_files": {"transform.py": correct_script},
        "bugs": bugs,
        "partial_solution_lattice_profile": {
            "no_fix_score": 0.15,
            "single_fix_max_score": 0.15,
            "pair_fix_max_score": 0.15,
            "all_but_one_max_score": 0.40,
            "full_solution_score": 1.0,
            "valid": True,
        },
        "defect_bundle": {
            "bundle_id": "account_reconciliation_order_chain",
            "defect_ids": [label for label, _, _ in defect_specs],
            "dependency_edges": [
                [
                    "deduplicate_before_identity_resolution",
                    "file_order_wins_over_updated_at",
                ],
                [
                    "file_order_wins_over_updated_at",
                    "latest_snapshot_ignores_event_time",
                ],
                ["latest_snapshot_ignores_event_time", "unsafe_active_coercion"],
                ["unsafe_active_coercion", "fractional_amount_truncation"],
            ],
            "capability_groups": {
                "identity_resolution": ["deduplicate_before_identity_resolution"],
                "deduplication": ["file_order_wins_over_updated_at"],
                "temporal_join": ["latest_snapshot_ignores_event_time"],
                "filtering": ["unsafe_active_coercion"],
                "aggregation": ["fractional_amount_truncation"],
            },
            "required_files": [
                "transform.py",
                "data/events.csv",
                "data/accounts.json",
                "data/aliases.csv",
            ],
            "semantic_dependency_depth": 5,
        },
        "hidden_json_assets": ({"expected_account_map.json": mapping} if compositional else {}),
        "evaluator_config": {
            "output_path": "outputs/account_report.json",
            "entrypoint": "transform.py",
            "capability_scoring": True,
            "required_json_artifacts": required_artifacts,
            "required_artifact_failure_cap": 0.30,
        },
        "composition_spec": composition_spec,
        "task_descriptor": {
            "input_files": [
                "data/events.csv",
                "data/accounts.json",
                "data/aliases.csv",
            ],
            "output_path": "outputs/account_report.json",
            "operations": [],
            "entrypoint": "python transform.py",
        },
    }
