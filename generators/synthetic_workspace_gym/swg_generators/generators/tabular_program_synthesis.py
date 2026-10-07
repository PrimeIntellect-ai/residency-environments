from __future__ import annotations

import csv
import io
import json
import random
from textwrap import dedent

from swg_generators.generators.d5_profiles import select_d5_profile
from swg_generators.generators.tabular_capability_fixtures import (
    build_focused_capability_assets,
)
from swg_generators.schemas import EnvironmentSpec


def _csv_text(rows: list[dict[str, object]], fields: list[str]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _fixture(rng: random.Random, *, hidden: bool) -> dict[str, object]:
    number = rng.randint(410, 899) + (1000 if hidden else 0)
    account_a = f"A-{number}"
    account_b = f"B-{number + 1}"
    account_c = f"C-{number + 2}"
    base = round(rng.uniform(18.0, 44.0), 2)
    aliases = [
        {"alias": " legacy-alpha ", "canonical_id": "acct-alpha"},
        {"alias": "ACCT-ALPHA", "canonical_id": account_a.lower()},
        {"alias": " beta ", "canonical_id": f" {account_b.lower()} "},
        {"alias": "old-charlie", "canonical_id": account_c},
    ]
    statuses: list[dict[str, object]] = [
        {
            "account_id": f" {account_a.lower()} ",
            "effective_at": "2026-01-01T00:00:00Z",
            "active": "yes",
        },
        {
            "account_id": account_a,
            "effective_at": "2026-03-01T00:00:00Z",
            "active": "no",
        },
        {
            "account_id": account_b,
            "effective_at": "2026-01-15T00:00:00+00:00",
            "active": "TRUE",
        },
        {
            "account_id": account_c.lower(),
            "effective_at": "2026-02-01T00:00:00Z",
            "active": "1",
        },
    ]
    events: list[dict[str, object]] = [
        {
            "event_id": "E-1",
            "account_ref": "LEGACY-ALPHA",
            "status": " posted ",
            "occurred_at": "2026-02-12T10:00:00Z",
            "updated_at": "2026-02-12T10:30:00+00:00",
            "amount": f"{base:.2f}",
        },
        {
            "event_id": "E-1",
            "account_ref": " acct-alpha ",
            "status": "POSTED",
            "occurred_at": "2026-02-12T10:00:00+00:00",
            "updated_at": "2026-02-12T06:00:00-05:00",
            "amount": f"{base + 2.35:.2f}",
        },
        {
            "event_id": "E-2",
            "account_ref": "legacy-alpha",
            "status": "posted",
            "occurred_at": "2026-02-28T20:00:00-05:00",
            "updated_at": "2026-03-01T01:05:00Z",
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
            "account_ref": account_c,
            "status": "posted",
            "occurred_at": "2026-02-15T10:00:00Z",
            "updated_at": "2026-02-15T10:05:00Z",
            "amount": "4.125",
        },
    ]
    rng.shuffle(events)
    expected = [
        {
            "account_id": account_a,
            "event_count": 1,
            "total_amount": round(base + 2.35, 2),
        },
        {"account_id": account_b, "event_count": 1, "total_amount": 13.4},
        {"account_id": account_c, "event_count": 1, "total_amount": 4.12},
    ]
    return {
        "events": events,
        "aliases": aliases,
        "statuses": statuses,
        "expected": expected,
        "edge_account_ids": [account_a, account_c],
    }


def _solution_script() -> str:
    return (
        dedent(
            """
            from __future__ import annotations

            import argparse
            import csv
            import json
            from datetime import datetime, timezone
            from pathlib import Path


            def normalize(value: object) -> str:
                return str(value).strip().upper()


            def timestamp(value: object) -> datetime:
                text = str(value).strip().replace("Z", "+00:00")
                parsed = datetime.fromisoformat(text)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)


            def active(value: object) -> bool:
                return str(value).strip().casefold() in {"1", "true", "yes", "active"}


            def alias_map(rows: list[dict[str, object]]) -> dict[str, str]:
                direct = {normalize(row["alias"]): normalize(row["canonical_id"]) for row in rows}
                resolved = {}
                for alias in direct:
                    current = alias
                    seen = set()
                    while current in direct:
                        if current in seen:
                            raise ValueError("alias cycle")
                        seen.add(current)
                        current = direct[current]
                    resolved[alias] = current
                return resolved


            def resolve(value: object, mapping: dict[str, str]) -> str:
                current = normalize(value)
                return mapping.get(current, current)


            def build_report(input_dir: Path) -> list[dict[str, object]]:
                aliases = json.loads((input_dir / "account_aliases.json").read_text(encoding="utf-8"))
                mapping = alias_map(aliases)
                with (input_dir / "events.csv").open(encoding="utf-8", newline="") as stream:
                    events = list(csv.DictReader(stream))
                with (input_dir / "status_history.csv").open(encoding="utf-8", newline="") as stream:
                    statuses = list(csv.DictReader(stream))

                snapshots: dict[str, list[dict[str, object]]] = {}
                for row in statuses:
                    canonical = resolve(row["account_id"], mapping)
                    snapshots.setdefault(canonical, []).append(row)
                for rows in snapshots.values():
                    rows.sort(key=lambda row: timestamp(row["effective_at"]))

                deduped: dict[tuple[str, str], dict[str, object]] = {}
                for row in events:
                    canonical = resolve(row["account_ref"], mapping)
                    key = (canonical, str(row["event_id"]))
                    if key not in deduped or timestamp(row["updated_at"]) > timestamp(deduped[key]["updated_at"]):
                        deduped[key] = {**row, "canonical_id": canonical}

                grouped: dict[str, dict[str, object]] = {}
                for row in deduped.values():
                    if normalize(row["status"]) != "POSTED":
                        continue
                    candidates = [
                        item
                        for item in snapshots.get(str(row["canonical_id"]), [])
                        if timestamp(item["effective_at"]) <= timestamp(row["occurred_at"])
                    ]
                    if not candidates or not active(candidates[-1]["active"]):
                        continue
                    account_id = str(row["canonical_id"])
                    entry = grouped.setdefault(
                        account_id,
                        {"account_id": account_id, "event_count": 0, "total_amount": 0.0},
                    )
                    entry["event_count"] = int(entry["event_count"]) + 1
                    entry["total_amount"] = round(
                        float(entry["total_amount"]) + float(row["amount"]), 2
                    )
                return sorted(grouped.values(), key=lambda row: str(row["account_id"]))


            def main() -> None:
                parser = argparse.ArgumentParser()
                parser.add_argument("--input-dir", required=True)
                parser.add_argument("--output", required=True)
                args = parser.parse_args()
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(build_report(Path(args.input_dir)), indent=2, sort_keys=True) + "\\n",
                    encoding="utf-8",
                )


            if __name__ == "__main__":
                main()
            """
        ).strip()
        + "\n"
    )


def _buggy_script(correct: str, *, profile_id: str) -> tuple[str, list[dict[str, object]]]:
    defects = [
        (
            "alias_resolution_bypassed",
            "return mapping.get(current, current)",
            "return current",
        ),
        (
            "deduplicate_before_alias_resolution",
            'key = (canonical, str(row["event_id"]))',
            'key = (normalize(row["account_ref"]), str(row["event_id"]))',
        ),
        (
            "lexical_updated_at_comparison",
            'timestamp(row["updated_at"]) > timestamp(deduped[key]["updated_at"])',
            'str(row["updated_at"]) > str(deduped[key]["updated_at"])',
        ),
        (
            "status_join_ignores_event_time",
            'if timestamp(item["effective_at"]) <= timestamp(row["occurred_at"])',
            "if True",
        ),
        (
            "unsafe_active_coercion",
            'return str(value).strip().casefold() in {"1", "true", "yes", "active"}',
            'return str(value).strip().casefold() == "true"',
        ),
        (
            "fractional_amount_truncation",
            'float(entry["total_amount"]) + float(row["amount"]), 2',
            'float(entry["total_amount"]) + int(float(row["amount"])), 2',
        ),
    ]
    profile_defects = {
        "d5_a": {
            "alias_resolution_bypassed",
            "unsafe_active_coercion",
            "fractional_amount_truncation",
        },
        "d5_b": {
            "alias_resolution_bypassed",
            "deduplicate_before_alias_resolution",
            "lexical_updated_at_comparison",
            "status_join_ignores_event_time",
        },
        "d5_c": {label for label, _, _ in defects},
    }
    selected = profile_defects[profile_id]
    profile_order = {
        "d5_a": [
            "unsafe_active_coercion",
            "fractional_amount_truncation",
            "alias_resolution_bypassed",
        ],
        "d5_b": [
            "alias_resolution_bypassed",
            "deduplicate_before_alias_resolution",
            "lexical_updated_at_comparison",
            "status_join_ignores_event_time",
        ],
        "d5_c": [
            "unsafe_active_coercion",
            "fractional_amount_truncation",
            "alias_resolution_bypassed",
            "deduplicate_before_alias_resolution",
            "lexical_updated_at_comparison",
            "status_join_ignores_event_time",
        ],
    }[profile_id]
    by_label = {item[0]: item for item in defects if item[0] in selected}
    defects = [by_label[label] for label in profile_order]
    bugs: list[dict[str, object]] = []
    buggy = correct
    for label, old, new in defects:

        def apply(content: str, *, old: str = old, new: str = new, label: str = label) -> str:
            updated = content.replace(old, new, 1)
            if updated == content:
                raise ValueError(f"tabular program defect {label!r} did not apply")
            return updated

        bugs.append({"label": label, "target_path": "process_report.py", "apply": apply})
        buggy = apply(buggy)
    return buggy, bugs


def build_account_event_program_scenario(rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
    visible = _fixture(rng, hidden=False)
    hidden = _fixture(random.Random(f"{spec.seed}:hidden-program-fixture"), hidden=True)
    profile = select_d5_profile(spec.difficulty, spec.seed)
    if profile is None:
        raise ValueError("account-event program synthesis requires difficulty 5")
    correct_script = _solution_script()
    buggy_script, bugs = _buggy_script(correct_script, profile_id=profile.profile_id)
    focused_assets = build_focused_capability_assets()
    visible_expected = list(visible["expected"])
    hidden_expected = list(hidden["expected"])
    files = {
        "data/events.csv": _csv_text(
            list(visible["events"]),
            [
                "event_id",
                "account_ref",
                "status",
                "occurred_at",
                "updated_at",
                "amount",
            ],
        ),
        "data/account_aliases.json": json.dumps(visible["aliases"], indent=2, sort_keys=True) + "\n",
        "data/status_history.csv": _csv_text(list(visible["statuses"]), ["account_id", "effective_at", "active"]),
        "process_report.py": buggy_script,
        "notes/legacy_ordering.md": "A retired implementation deduplicated raw aliases before canonicalization.\n",
        "archive/example_report.json": "[]\n",
        "notes/manual_run.txt": "Old one-off reports are not authoritative transformation logic.\n",
    }
    hidden_text_assets = {
        "hidden_fixture/events.csv": _csv_text(
            list(hidden["events"]),
            [
                "event_id",
                "account_ref",
                "status",
                "occurred_at",
                "updated_at",
                "amount",
            ],
        ),
        "hidden_fixture/status_history.csv": _csv_text(
            list(hidden["statuses"]), ["account_id", "effective_at", "active"]
        ),
    }
    return {
        "scenario_id": "account_event_program_synthesis",
        "unmodified_reward_limit": {
            "d5_a": 0.40,
            "d5_b": 0.55,
            "d5_c": 0.40,
        }[profile.profile_id],
        "title": "Account Event Program Synthesis",
        "description": "Implement a reusable order-sensitive account-event transformation and generate the visible report.",
        "output_path": "artifacts/report.json",
        "input_files": [
            "data/events.csv",
            "data/account_aliases.json",
            "data/status_history.csv",
        ],
        "operations": [
            "normalize aliases and resolve canonical account identity",
            "deduplicate by canonical identity using the latest normalized update timestamp",
            "select the account status effective at each event time",
            "remove inactive, unresolved, and non-posted events after the temporal join",
            "aggregate fractional amounts, round totals to two decimals, and sort by account_id",
        ],
        "output_contract": [
            "Create `process_report.py` and generate `artifacts/report.json`.",
            "The supported command is `python process_report.py --input-dir data --output artifacts/report.json`.",
            "The script must use only its input directory and output arguments; it will be run on unseen fixtures.",
            "Output must be a JSON list with `account_id`, `event_count`, and `total_amount` fields.",
            "Operation order is part of the contract: resolve aliases before deduplication and perform the temporal status join before filtering.",
        ],
        "hints": [],
        "structure": {
            "task_type": "executable_program_synthesis",
            "input_shape": "csv_json_csv",
            "dependency_depth": profile.semantic_dependency_depth,
            "distractor_count": 3,
            "hidden_capability_count": 12,
            "d5_profile": profile.profile_id,
            "output_style": "script_and_sorted_report",
        },
        "files": files,
        "expected_output": visible_expected,
        "reference_solution_files": {
            "process_report.py": correct_script,
            "artifacts/report.json": json.dumps(visible_expected, indent=2, sort_keys=True) + "\n",
        },
        "correct_files": {"process_report.py": correct_script},
        "bugs": bugs,
        "hidden_json_assets": {
            "hidden_fixture/account_aliases.json": hidden["aliases"],
            "hidden_expected_output.json": hidden_expected,
            **dict(focused_assets["hidden_json_assets"]),
        },
        "hidden_text_assets": {
            **hidden_text_assets,
            **dict(focused_assets["hidden_text_assets"]),
        },
        "evaluator_entrypoint": "swg_generators.evaluators.tabular_capability_program:TabularCapabilityProgramEvaluator",
        "evaluator_config": {
            "mode": "program_synthesis",
            "script_path": "process_report.py",
            "visible_input_dir": "data",
            "output_path": "artifacts/report.json",
            "hidden_fixture_dir": "hidden_fixture",
            "hidden_expected_path": "hidden_expected_output.json",
            "edge_account_ids": hidden["edge_account_ids"],
            "focused_fixtures": list(focused_assets["focused_fixtures"]),
            "d5_profile": profile.profile_id,
        },
        "partial_solution_lattice_profile": {
            "no_fix_score": 0.10,
            "single_fix_max_score": 0.30,
            "pair_fix_max_score": 0.55,
            "all_but_one_max_score": 0.80,
            "full_solution_score": 1.0,
            "valid": True,
        },
        "defect_bundle": {
            "bundle_id": "account_program_order_chain",
            "defect_ids": [str(bug["label"]) for bug in bugs],
            "dependency_edges": [
                [str(bugs[index]["label"]), str(bugs[index + 1]["label"])] for index in range(len(bugs) - 1)
            ],
            "capability_groups": {str(bug["label"]): [str(bug["label"])] for bug in bugs},
            "required_files": [
                "process_report.py",
                "artifacts/report.json",
                "data/events.csv",
                "data/account_aliases.json",
                "data/status_history.csv",
            ],
            "semantic_dependency_depth": profile.semantic_dependency_depth,
        },
        "composition_spec": {},
        "task_descriptor": {
            "input_files": [
                "data/events.csv",
                "data/account_aliases.json",
                "data/status_history.csv",
            ],
            "output_path": "artifacts/report.json",
            "required_files": ["process_report.py", "artifacts/report.json"],
            "operations": [],
            "entrypoint": "python process_report.py --input-dir data --output artifacts/report.json",
        },
    }
