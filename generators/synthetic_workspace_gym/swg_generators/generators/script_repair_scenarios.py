from __future__ import annotations

import json
import random
from datetime import datetime
from textwrap import dedent

from swg_generators.generators.common import normalize_composition_mode


def _d5_composition_mode(spec) -> str:
    override = normalize_composition_mode(spec.generation_params.get("composition_mode"))
    return override or ("compositional" if spec.seed % 2 else "hard_atomic")


def build_csv_schema_drift_scenario(generator) -> dict[str, object]:
    orders_csv = (
        "account_id,region,status,amount\n"
        "A-100,North,complete,120.00\n"
        "A-101,West,complete,95.50\n"
        "A-102,East,cancelled,20.00\n"
        "A-103,North,complete,90.00\n"
        "A-104,East,complete,85.00\n"
    )
    expected_report = [
        {"region": "east", "row_count": 1, "total_amount": 85.0},
        {"region": "north", "row_count": 2, "total_amount": 210.0},
        {"region": "west", "row_count": 1, "total_amount": 95.5},
    ]
    normalization = dedent(
        """\
        from __future__ import annotations


        def normalize_text(value: object) -> str:
            return str(value).strip().casefold()


        def normalize_amount(value: object) -> float:
            return round(float(value), 2)
        """
    )
    parser = dedent(
        """\
        from __future__ import annotations

        import csv
        from pathlib import Path

        from repair_target.normalization import normalize_amount, normalize_text


        def load_orders(path: Path) -> list[dict[str, object]]:
            with Path(path).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = []
                for row in reader:
                    if not row.get("account_id"):
                        continue
                    rows.append(
                        {
                            "account_id": row["account_id"],
                            "region": normalize_text(row["region"]),
                            "status": normalize_text(row["status"]),
                            "amount": normalize_amount(row["amount"]),
                        }
                    )
            return rows
        """
    )
    report = dedent(
        """\
        from __future__ import annotations


        def build_region_report(rows: list[dict[str, object]]) -> list[dict[str, object]]:
            summary: dict[str, dict[str, object]] = {}
            for row in rows:
                if row["status"] == "cancelled":
                    continue
                region = str(row["region"])
                if region not in summary:
                    summary[region] = {"region": region, "row_count": 0, "total_amount": 0.0}
                summary[region]["row_count"] = int(summary[region]["row_count"]) + 1
                summary[region]["total_amount"] = round(
                    float(summary[region]["total_amount"]) + float(row["amount"]),
                    2,
                )
            return sorted(summary.values(), key=lambda item: str(item["region"]))
        """
    )

    def legacy_test_runner() -> str:
        return generator.hidden_runner(
            asset_setup='expected = json.loads((hidden_root / "expected_region_report.json").read_text(encoding="utf-8"))',
            import_block="""
            from repair_target.parser import load_orders
            from repair_target.report import build_region_report
            """,
            test_methods="""
            def test_rows_are_loaded(self) -> None:
                rows = load_orders(workspace / "data" / "orders.csv")
                self.assertEqual(len(rows), 5)
                self.assertTrue(all(str(row["account_id"]).startswith("A-") for row in rows))

            def test_report_contract(self) -> None:
                rows = load_orders(workspace / "data" / "orders.csv")
                self.assertEqual(build_region_report(rows), expected)

            def test_sorted_regions(self) -> None:
                rows = load_orders(workspace / "data" / "orders.csv")
                actual = build_region_report(rows)
                self.assertEqual([row["region"] for row in actual], ["east", "north", "west"])
            """,
        )

    def materialize(spec) -> dict[str, object]:
        rng = random.Random(spec.seed)
        delta = (spec.seed % 9) / 10
        rows = [
            {
                "account_id": f"A-{spec.seed}-0",
                "region": " North ",
                "status": "complete",
                "amount": 120.25 + delta,
            },
            {
                "account_id": f"A-{spec.seed}-1",
                "region": "WEST",
                "status": "pending",
                "amount": 15.5,
            },
            {
                "account_id": f"A-{spec.seed}-2",
                "region": "east",
                "status": "cancelled",
                "amount": 99.0,
            },
            {
                "account_id": f"A-{spec.seed}-3",
                "region": "north",
                "status": "COMPLETE",
                "amount": 80.1,
            },
            {
                "account_id": f"A-{spec.seed}-4",
                "region": " East ",
                "status": "complete",
                "amount": 42.35 + delta,
            },
        ]
        rng.shuffle(rows)
        if spec.difficulty == 5:
            csv_lines = ["account_id,customer_id,region,status,amount"]
            csv_lines.extend(
                f"{row['account_id']},LEGACY-{row['account_id']},{row['region']},{row['status']},{row['amount']}"
                for row in rows
            )
        else:
            csv_lines = ["account_id,region,status,amount"]
            csv_lines.extend(f"{row['account_id']},{row['region']},{row['status']},{row['amount']}" for row in rows)
        parsed = [
            {
                "account_id": row["account_id"],
                "region": str(row["region"]).strip().casefold(),
                "status": str(row["status"]).strip().casefold(),
                "amount": round(float(row["amount"]), 2),
            }
            for row in rows
        ]
        summary: dict[str, dict[str, object]] = {}
        for row in parsed:
            if row["status"] == "cancelled":
                continue
            region = str(row["region"])
            if region not in summary:
                summary[region] = {
                    "region": region,
                    "row_count": 0,
                    "total_amount": 0.0,
                }
            summary[region]["row_count"] = int(summary[region]["row_count"]) + 1
            summary[region]["total_amount"] = round(float(summary[region]["total_amount"]) + float(row["amount"]), 2)
        expected = sorted(summary.values(), key=lambda item: str(item["region"]))
        files = {
            "src/repair_target/__init__.py": "",
            "src/repair_target/normalization.py": normalization,
            "src/repair_target/parser.py": parser,
            "src/repair_target/report.py": report,
            "run_example.py": generator.json_runner(
                import_block="""
                from repair_target.parser import load_orders
                from repair_target.report import build_region_report
                """,
                expression='build_region_report(load_orders(workspace / "data" / "orders.csv"))',
            ),
            "data/orders.csv": "\n".join(csv_lines) + "\n",
        }
        distractor_count = 0
        composition_mode = None
        source_families: list[str] = []
        if spec.difficulty == 5:
            composition_mode = _d5_composition_mode(spec)
            source_families = ["tabular", "script_repair"] if composition_mode == "compositional" else ["script_repair"]
            files.update(
                {
                    "src/repair_target/legacy_parser.py": '"""Deprecated customer_id parser; not used by the current entrypoint."""\n',
                    "docs/api-v2.md": "# Orders API v2 (authoritative)\n\nThis is the current contract. Use account_id, normalize region and status before filtering, retain every non-cancelled row, preserve fractional amounts, and sort the final report by canonical region.\n",
                    "docs/api-v1.md": "# Orders API v1\n\nLegacy deployments used customer_id and treated only complete rows as reportable.\n",
                    "changelog/2026-04-migration.md": "# April 2026 migration\n\nAPI v2 became authoritative on 2026-04-15. Compatibility input may be read, but v2 semantics and output win.\n",
                    "notes/schema_v1_archived.md": "# Handoff note\n\nKeep customer_id and complete-only filtering for older consumers. This note predates the April migration.\n",
                    "data/orders_v1_sample.csv": "customer_id,region,status,amount\nOLD-1,legacy,complete,1.00\n",
                }
            )
            if composition_mode == "hard_atomic":
                for path in (
                    "docs/api-v2.md",
                    "docs/api-v1.md",
                    "changelog/2026-04-migration.md",
                ):
                    files.pop(path)
            else:
                files["analysis/observed_schema.json"] = (
                    json.dumps(
                        {
                            "columns": ["account_id", "region", "status", "amount"],
                            "observations": [
                                "region and status contain inconsistent casing and whitespace",
                                "amounts include fractional values",
                            ],
                        },
                        indent=2,
                    )
                    + "\n"
                )
            distractor_count = 3
        realized = {
            "files": files,
            "hidden_json_assets": {"expected_region_report.json": expected},
            "structure": {
                "repair_surface": "normalization_parser_and_reporting",
                "bug_scope": "three_file_dependency_chain",
                "failure_mode": "semantic_and_formatting",
                "smoke_test_quality": "partially_informative",
                "dependency_depth": 3,
                "hidden_capability_count": 6 if spec.difficulty == 5 else 3,
                "distractor_count": distractor_count,
                "contract_source_count": (4 if composition_mode == "compositional" else 1),
                "composition_mode": composition_mode,
                "source_families": source_families,
                "composition_depth": len(source_families),
                "public_check_coverage": ["execution", "visible_fixture"],
            },
        }
        if spec.difficulty < 5:
            realized["test_runner"] = legacy_test_runner()
        return realized

    return {
        "scenario_id": "csv_schema_drift",
        "d5_compositional_families": ["tabular", "script_repair"],
        "title": "CSV Schema Drift Repair",
        "materialize": materialize,
        "debug_note": "The smoke test may still run after a partial fix, but the hidden tests check that rows are not silently dropped and the final ordering contract is preserved.\n",
        "hints": [
            "The CSV headers and the parser must agree on the account identifier column.",
            "Cancelled rows should be excluded from the final report, not dropped before parsing.",
            "The final report is sorted by region, not by aggregate size.",
        ],
        "repair_contract": [
            "Load every row keyed by the visible `account_id` column; do not look for `customer_id`.",
            "Normalize region and status with surrounding whitespace removed and Unicode-aware case folding.",
            "Exclude rows whose normalized status is `cancelled` in the report step.",
            "Return report rows sorted lexicographically by lowercase `region`.",
            "Preserve fractional amounts and round reported totals to two decimal places.",
        ],
        "structure": {
            "repair_surface": "parser_interface",
            "bug_scope": "cross_file",
            "failure_mode": "semantic_and_formatting",
            "smoke_test_quality": "partially_informative",
        },
        "files": {
            "src/repair_target/__init__.py": "",
            "src/repair_target/normalization.py": normalization,
            "src/repair_target/parser.py": parser,
            "src/repair_target/report.py": report,
            "run_example.py": generator.json_runner(
                import_block="""
                from repair_target.parser import load_orders
                from repair_target.report import build_region_report
                """,
                expression='build_region_report(load_orders(workspace / "data" / "orders.csv"))',
            ),
            "data/orders.csv": orders_csv,
        },
        "hidden_json_assets": {
            "expected_region_report.json": expected_report,
        },
        "core_bugs": [
            {
                "label": "schema_mapping_bug",
                "target_path": "src/repair_target/parser.py",
                "apply": generator.replace_once(
                    '"account_id": row["account_id"],',
                    '"account_id": row["customer_id"],',
                    label="schema_mapping_bug",
                    target_path="src/repair_target/parser.py",
                ),
            },
            {
                "label": "missing_rows_bug",
                "target_path": "src/repair_target/parser.py",
                "apply": generator.replace_once(
                    'if not row.get("account_id"):',
                    'if not row.get("customer_id"):',
                    label="missing_rows_bug",
                    target_path="src/repair_target/parser.py",
                ),
            },
            {
                "label": "wrong_sort_key",
                "target_path": "src/repair_target/report.py",
                "apply": generator.replace_once(
                    'return sorted(summary.values(), key=lambda item: str(item["region"]))',
                    'return sorted(summary.values(), key=lambda item: str(item["row_count"]))',
                    label="wrong_sort_key",
                    target_path="src/repair_target/report.py",
                ),
            },
        ],
        "advanced_bug_budget": 2,
        "advanced_bugs": [
            {
                "label": "legacy_identifier_precedence_bug",
                "target_path": "src/repair_target/parser.py",
                "apply": generator.replace_once(
                    '"account_id": row["account_id"],',
                    '"account_id": row.get("customer_id") or row["account_id"],',
                    label="legacy_identifier_precedence_bug",
                    target_path="src/repair_target/parser.py",
                ),
            },
            {
                "label": "text_normalization_bug",
                "target_path": "src/repair_target/normalization.py",
                "apply": generator.replace_once(
                    "return str(value).strip().casefold()",
                    "return str(value).casefold()",
                    label="text_normalization_bug",
                    target_path="src/repair_target/normalization.py",
                ),
            },
            {
                "label": "amount_precision_bug",
                "target_path": "src/repair_target/normalization.py",
                "apply": generator.replace_once(
                    "return round(float(value), 2)",
                    "return float(int(float(value)))",
                    label="amount_precision_bug",
                    target_path="src/repair_target/normalization.py",
                ),
            },
            {
                "label": "cancelled_only_filter_bug",
                "target_path": "src/repair_target/report.py",
                "apply": generator.replace_once(
                    'if row["status"] == "cancelled":',
                    'if row["status"] != "complete":',
                    label="cancelled_only_filter_bug",
                    target_path="src/repair_target/report.py",
                ),
            },
        ],
        "d5_bug_bundles": [
            {
                "bundle_id": "schema_migration_chain",
                "bugs": [
                    "legacy_identifier_precedence_bug",
                    "wrong_sort_key",
                    "text_normalization_bug",
                    "amount_precision_bug",
                    "cancelled_only_filter_bug",
                ],
                "dependency_edges": [
                    ["text_normalization_bug", "cancelled_only_filter_bug"],
                    ["cancelled_only_filter_bug", "amount_precision_bug"],
                    ["amount_precision_bug", "wrong_sort_key"],
                    ["legacy_identifier_precedence_bug", "text_normalization_bug"],
                ],
                "capabilities": [
                    "contract_resolution",
                    "transformation",
                    "cross_file_consistency",
                    "integration",
                    "edge_cases",
                ],
                "semantic_dependency_depth": 5,
            }
        ],
        "capability_groups": {
            "contract_resolution": ["test_rows_are_loaded"],
            "transformation": ["test_text_normalization", "test_amount_precision"],
            "cross_file_consistency": ["test_pending_rows_are_retained"],
            "integration": ["test_report_contract"],
            "edge_cases": ["test_sorted_regions"],
        },
        "partial_solution_lattice_profile": {
            "validation_seed": 91,
            "no_fix_score": 0.0,
            "single_fix_max_score": 0.2,
            "pair_fix_max_score": 0.4,
            "all_but_one_max_score": 0.55,
            "full_solution_score": 1.0,
            "valid": True,
        },
        "partial_solution_lattice": {
            "no_fixes_max": 0.15,
            "single_fix_max": 0.40,
            "pair_fix_max": 0.65,
            "all_but_one_max": 0.85,
            "complete_score": 1.0,
        },
        "capability_score_caps": {
            "integration": 0.55,
            "contract_resolution": 0.45,
        },
        "test_runner": generator.hidden_runner(
            asset_setup='expected = json.loads((hidden_root / "expected_region_report.json").read_text(encoding="utf-8"))',
            import_block="""
            from repair_target.parser import load_orders
            from repair_target.report import build_region_report
            from repair_target.normalization import normalize_amount, normalize_text
            """,
            test_methods="""
            def test_rows_are_loaded(self) -> None:
                rows = load_orders(workspace / "data" / "orders.csv")
                self.assertEqual(len(rows), 5)
                self.assertTrue(all(str(row["account_id"]).startswith("A-") for row in rows))

            def test_report_contract(self) -> None:
                rows = load_orders(workspace / "data" / "orders.csv")
                self.assertEqual(build_region_report(rows), expected)

            def test_sorted_regions(self) -> None:
                rows = load_orders(workspace / "data" / "orders.csv")
                actual = build_region_report(rows)
                self.assertEqual([row["region"] for row in actual], ["east", "north", "west"])
            def test_text_normalization(self) -> None:
                self.assertEqual(normalize_text("  NORTH  "), "north")

            def test_amount_precision(self) -> None:
                self.assertEqual(normalize_amount("1.255"), 1.25)

            def test_pending_rows_are_retained(self) -> None:
                rows = [
                    {"account_id": "a", "region": "west", "status": "pending", "amount": 2.5},
                    {"account_id": "b", "region": "west", "status": "cancelled", "amount": 9.0},
                ]
                self.assertEqual(
                    build_region_report(rows),
                    [{"region": "west", "row_count": 1, "total_amount": 2.5}],
                )
            """,
        ),
    }


def build_timestamp_normalization_scenario(generator) -> dict[str, object]:
    events = [
        {"amount": 2.0, "event_id": "evt-1", "timestamp": "2024-02-01 09:30"},
        {"amount": 1.5, "event_id": "evt-2", "timestamp": "01/31/2024 23:00"},
        {"amount": 3.0, "event_id": "evt-3", "timestamp": "2024/02/01 08:15"},
        {"amount": 4.0, "event_id": "evt-4", "timestamp": "02-Feb-2024 10:00"},
    ]
    expected_report = {
        "daily_totals": [
            {"day": "2024-01-31", "total_amount": 1.5},
            {"day": "2024-02-01", "total_amount": 5.0},
            {"day": "2024-02-02", "total_amount": 4.0},
        ],
        "ordered_ids": ["evt-2", "evt-3", "evt-1", "evt-4"],
    }
    amounts = dedent(
        """\
        from __future__ import annotations


        def normalize_amount(value: object) -> float:
            return round(float(value), 2)
        """
    )
    time_utils = dedent(
        """\
        from __future__ import annotations

        from datetime import datetime

        FORMATS = ("%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M", "%d-%b-%Y %H:%M")


        def parse_timestamp(value: str) -> datetime:
            value = value.strip()
            for fmt in FORMATS:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unsupported timestamp format: {value}")


        def sort_events(rows: list[dict[str, object]]) -> list[dict[str, object]]:
            return sorted(rows, key=lambda row: (parse_timestamp(str(row["timestamp"])), str(row["event_id"])))
        """
    )
    report = dedent(
        """\
        from __future__ import annotations

        from repair_target.time_utils import parse_timestamp, sort_events

        from repair_target.amounts import normalize_amount

        def build_report(rows: list[dict[str, object]]) -> dict[str, object]:
            ordered = sort_events(rows)
            daily_totals: dict[str, float] = {}
            ordered_ids: list[str] = []
            for row in ordered:
                stamp = parse_timestamp(str(row["timestamp"]))
                ordered_ids.append(str(row["event_id"]))
                day = stamp.strftime("%Y-%m-%d")
                daily_totals[day] = round(daily_totals.get(day, 0.0) + normalize_amount(row["amount"]), 2)
            return {
                "ordered_ids": ordered_ids,
                "daily_totals": [
                    {"day": day, "total_amount": daily_totals[day]}
                    for day in sorted(daily_totals)
                ],
            }
        """
    )

    def legacy_test_runner() -> str:
        return generator.hidden_runner(
            asset_setup='expected = json.loads((hidden_root / "expected_timeline_report.json").read_text(encoding="utf-8"))',
            import_block="""
            from repair_target.report import build_report
            from repair_target.time_utils import parse_timestamp, sort_events
            """,
            test_methods="""
            def test_timestamp_formats(self) -> None:
                self.assertEqual(parse_timestamp("2024/02/01 08:15").strftime("%Y-%m-%d %H:%M"), "2024-02-01 08:15")

            def test_chronological_order(self) -> None:
                rows = json.loads((workspace / "data" / "events.json").read_text(encoding="utf-8"))
                ordered = sort_events(rows)
                self.assertEqual([row["event_id"] for row in ordered], expected["ordered_ids"])

            def test_report(self) -> None:
                rows = json.loads((workspace / "data" / "events.json").read_text(encoding="utf-8"))
                self.assertEqual(build_report(rows), expected)
            """,
        )

    def materialize(spec) -> dict[str, object]:
        rng = random.Random(spec.seed)
        delta = (spec.seed % 5) / 100
        seeded_events = [
            {
                "amount": 1.55 + delta,
                "event_id": f"evt-{spec.seed}-d",
                "timestamp": "01/31/2024 23:00",
            },
            {
                "amount": 2.25,
                "event_id": f"evt-{spec.seed}-b",
                "timestamp": "2024/02/01 08:15",
            },
            {
                "amount": -0.35,
                "event_id": f"evt-{spec.seed}-a",
                "timestamp": "2024/02/01 08:15",
            },
            {
                "amount": 3.1,
                "event_id": f"evt-{spec.seed}-c",
                "timestamp": " 2024-02-01 09:30 ",
            },
            {
                "amount": 4.05 + delta,
                "event_id": f"evt-{spec.seed}-e",
                "timestamp": "02-Feb-2024 10:00",
            },
        ]
        rng.shuffle(seeded_events)
        formats = (
            "%Y-%m-%d %H:%M",
            "%m/%d/%Y %H:%M",
            "%Y/%m/%d %H:%M",
            "%d-%b-%Y %H:%M",
        )

        def parse_visible(value: object) -> datetime:
            cleaned = str(value).strip()
            for fmt in formats:
                try:
                    return datetime.strptime(cleaned, fmt)
                except ValueError:
                    continue
            raise ValueError(cleaned)

        ordered = sorted(
            seeded_events,
            key=lambda row: (parse_visible(row["timestamp"]), str(row["event_id"])),
        )
        totals: dict[str, float] = {}
        for row in ordered:
            day = parse_visible(row["timestamp"]).strftime("%Y-%m-%d")
            totals[day] = round(totals.get(day, 0.0) + round(float(row["amount"]), 2), 2)
        expected = {
            "ordered_ids": [str(row["event_id"]) for row in ordered],
            "daily_totals": [{"day": day, "total_amount": totals[day]} for day in sorted(totals)],
        }
        files = {
            "src/repair_target/__init__.py": "",
            "src/repair_target/amounts.py": amounts,
            "src/repair_target/time_utils.py": time_utils,
            "src/repair_target/report.py": report,
            "run_example.py": generator.json_runner(
                import_block="from repair_target.report import build_report",
                expression='build_report(json.loads((workspace / "data" / "events.json").read_text(encoding="utf-8")))',
            ),
            "data/events.json": json.dumps(seeded_events, indent=2, sort_keys=True) + "\n",
        }
        distractor_count = 0
        composition_mode = None
        source_families: list[str] = []
        if spec.difficulty == 5:
            composition_mode = _d5_composition_mode(spec)
            source_families = (
                ["pipeline", "script_repair"] if composition_mode == "compositional" else ["script_repair"]
            )
            files.update(
                {
                    "src/repair_target/legacy_dates.py": '"""Deprecated lexical date sorter; retained for historical reference only."""\n',
                    "docs/timestamps-v2.md": "# Timestamp contract v2 (authoritative)\n\nTrim documented timestamps, parse before filtering or ordering, use event_id for equal-time ties, preserve signed fractional amounts, and emit ISO day keys.\n",
                    "docs/timestamps-v1.md": "# Timestamp contract v1\n\nLegacy jobs sorted raw timestamp strings and emitted locale day keys.\n",
                    "changelog/2026-04-migration.md": "# Timestamp migration\n\nv2 became authoritative in April 2026; v1 examples remain for input compatibility only.\n",
                    "notes/timestamp_v1_archived.md": "# Operations handoff\n\nRaw ordering is faster and should remain the default. This note predates the v2 migration.\n",
                    "data/events_v1_sample.json": json.dumps(
                        [{"event_id": "legacy", "timestamp": "2020-01-01", "amount": 1}]
                    )
                    + "\n",
                }
            )
            if composition_mode == "hard_atomic":
                for path in (
                    "docs/timestamps-v2.md",
                    "docs/timestamps-v1.md",
                    "changelog/2026-04-migration.md",
                ):
                    files.pop(path)
            else:
                files["logs/production_incident.log"] = (
                    "2026-04-19T03:12:07Z daily-report ordering differs across workers\n"
                    "2026-04-19T03:12:08Z signed fractional totals disagree with ledger\n"
                )
            distractor_count = 3
        realized = {
            "files": files,
            "hidden_json_assets": {"expected_timeline_report.json": expected},
            "structure": {
                "repair_surface": "timestamp_normalization_sorting_and_amounts",
                "bug_scope": "three_file_dependency_chain",
                "failure_mode": "semantic",
                "smoke_test_quality": "misleading",
                "dependency_depth": 3,
                "hidden_capability_count": 7 if spec.difficulty == 5 else 3,
                "distractor_count": distractor_count,
                "contract_source_count": (4 if composition_mode == "compositional" else 1),
                "composition_mode": composition_mode,
                "source_families": source_families,
                "composition_depth": len(source_families),
                "public_check_coverage": ["execution", "visible_fixture"],
            },
        }
        if spec.difficulty < 5:
            realized["test_runner"] = legacy_test_runner()
        return realized

    return {
        "scenario_id": "timestamp_normalization",
        "d5_compositional_families": ["pipeline", "script_repair"],
        "title": "Timestamp Normalization Repair",
        "materialize": materialize,
        "debug_note": "The visible smoke test can look plausible even when timestamps are sorted lexicographically instead of chronologically.\n",
        "hints": [
            "This workspace mixes several timestamp formats.",
            "Chronological ordering should use parsed datetimes, not raw strings.",
            "The final daily summary uses ISO day strings.",
        ],
        "repair_contract": [
            "Parse every documented timestamp format after trimming surrounding whitespace.",
            "Sort chronologically and break equal-timestamp ties lexicographically by event_id.",
            "Preserve signed fractional amounts and round daily totals to two decimal places.",
            "Emit ISO YYYY-MM-DD daily keys and stable ordered_ids.",
        ],
        "structure": {
            "repair_surface": "normalization_and_sorting",
            "bug_scope": "cross_file",
            "failure_mode": "semantic",
            "smoke_test_quality": "misleading",
        },
        "files": {
            "src/repair_target/__init__.py": "",
            "src/repair_target/amounts.py": amounts,
            "src/repair_target/time_utils.py": time_utils,
            "src/repair_target/report.py": report,
            "run_example.py": generator.json_runner(
                import_block="from repair_target.report import build_report",
                expression='build_report(json.loads((workspace / "data" / "events.json").read_text(encoding="utf-8")))',
            ),
            "data/events.json": json.dumps(events, indent=2, sort_keys=True) + "\n",
        },
        "hidden_json_assets": {
            "expected_timeline_report.json": expected_report,
        },
        "core_bugs": [
            {
                "label": "missing_format",
                "target_path": "src/repair_target/time_utils.py",
                "apply": generator.replace_once(
                    '"%Y/%m/%d %H:%M"',
                    '"%Y-%d-%m %H:%M"',
                    label="missing_format",
                    target_path="src/repair_target/time_utils.py",
                ),
            },
            {
                "label": "wrong_sort_key",
                "target_path": "src/repair_target/time_utils.py",
                "apply": generator.replace_once(
                    'return sorted(rows, key=lambda row: (parse_timestamp(str(row["timestamp"])), str(row["event_id"])))',
                    'return sorted(rows, key=lambda row: str(row["timestamp"]))',
                    label="wrong_sort_key",
                    target_path="src/repair_target/time_utils.py",
                ),
            },
            {
                "label": "wrong_output_schema",
                "target_path": "src/repair_target/report.py",
                "apply": generator.replace_once(
                    'day = stamp.strftime("%Y-%m-%d")',
                    'day = stamp.strftime("%m/%d/%Y")',
                    label="wrong_output_schema",
                    target_path="src/repair_target/report.py",
                ),
            },
        ],
        "advanced_bug_budget": 2,
        "advanced_bugs": [
            {
                "label": "timestamp_whitespace_bug",
                "target_path": "src/repair_target/time_utils.py",
                "apply": generator.replace_once(
                    "value = value.strip()",
                    "value = value",
                    label="timestamp_whitespace_bug",
                    target_path="src/repair_target/time_utils.py",
                ),
            },
            {
                "label": "invalid_timestamp_handling_bug",
                "target_path": "src/repair_target/time_utils.py",
                "apply": generator.replace_once(
                    'raise ValueError(f"Unsupported timestamp format: {value}")',
                    "return datetime.min",
                    label="invalid_timestamp_handling_bug",
                    target_path="src/repair_target/time_utils.py",
                ),
            },
            {
                "label": "amount_precision_bug",
                "target_path": "src/repair_target/amounts.py",
                "apply": generator.replace_once(
                    "return round(float(value), 2)",
                    "return float(int(float(value)))",
                    label="amount_precision_bug",
                    target_path="src/repair_target/amounts.py",
                ),
            },
            {
                "label": "signed_amount_bug",
                "target_path": "src/repair_target/amounts.py",
                "apply": generator.replace_once(
                    "return round(float(value), 2)",
                    "return round(abs(float(value)), 2)",
                    label="signed_amount_bug",
                    target_path="src/repair_target/amounts.py",
                ),
            },
        ],
        "d5_bug_bundles": [
            {
                "bundle_id": "temporal_contract_chain",
                "bugs": [
                    "wrong_sort_key",
                    "wrong_output_schema",
                    "timestamp_whitespace_bug",
                    "invalid_timestamp_handling_bug",
                    "signed_amount_bug",
                ],
                "dependency_edges": [
                    ["timestamp_whitespace_bug", "wrong_sort_key"],
                    ["timestamp_whitespace_bug", "invalid_timestamp_handling_bug"],
                    ["wrong_sort_key", "signed_amount_bug"],
                    ["signed_amount_bug", "wrong_output_schema"],
                ],
                "capabilities": [
                    "contract_resolution",
                    "transformation",
                    "cross_file_consistency",
                    "integration",
                    "edge_cases",
                ],
                "semantic_dependency_depth": 5,
            }
        ],
        "capability_groups": {
            "contract_resolution": ["test_invalid_timestamp_raises"],
            "transformation": [
                "test_timestamp_whitespace",
                "test_signed_fractional_amount",
            ],
            "cross_file_consistency": ["test_chronological_order"],
            "integration": ["test_report"],
            "edge_cases": ["test_equal_timestamp_tie_break"],
        },
        "partial_solution_lattice_profile": {
            "validation_seed": 91,
            "no_fix_score": 0.0,
            "single_fix_max_score": 0.2,
            "pair_fix_max_score": 0.45,
            "all_but_one_max_score": 0.55,
            "full_solution_score": 1.0,
            "valid": True,
        },
        "partial_solution_lattice": {
            "no_fixes_max": 0.15,
            "single_fix_max": 0.40,
            "pair_fix_max": 0.65,
            "all_but_one_max": 0.85,
            "complete_score": 1.0,
        },
        "capability_score_caps": {
            "integration": 0.55,
            "contract_resolution": 0.45,
        },
        "test_runner": generator.hidden_runner(
            asset_setup='expected = json.loads((hidden_root / "expected_timeline_report.json").read_text(encoding="utf-8"))',
            import_block="""
            from repair_target.report import build_report
            from repair_target.time_utils import parse_timestamp, sort_events
            from repair_target.amounts import normalize_amount
            """,
            test_methods="""
            def test_timestamp_formats(self) -> None:
                self.assertEqual(parse_timestamp("2024/02/01 08:15").strftime("%Y-%m-%d %H:%M"), "2024-02-01 08:15")

            def test_chronological_order(self) -> None:
                rows = json.loads((workspace / "data" / "events.json").read_text(encoding="utf-8"))
                ordered = sort_events(rows)
                self.assertEqual([row["event_id"] for row in ordered], expected["ordered_ids"])

            def test_report(self) -> None:
                rows = json.loads((workspace / "data" / "events.json").read_text(encoding="utf-8"))
                self.assertEqual(build_report(rows), expected)
            def test_timestamp_whitespace(self) -> None:
                self.assertEqual(
                    parse_timestamp(" 2024-02-01 09:30 ").strftime("%Y-%m-%d %H:%M"),
                    "2024-02-01 09:30",
                )

            def test_equal_timestamp_tie_break(self) -> None:
                rows = [
                    {"event_id": "b", "timestamp": "2024-02-01 08:15", "amount": 1},
                    {"event_id": "a", "timestamp": "2024-02-01 08:15", "amount": 1},
                ]
                self.assertEqual([row["event_id"] for row in sort_events(rows)], ["a", "b"])

            def test_signed_fractional_amount(self) -> None:
                self.assertEqual(normalize_amount("-1.255"), -1.25)

            def test_invalid_timestamp_raises(self) -> None:
                with self.assertRaises(ValueError):
                    parse_timestamp("not-a-timestamp")
            """,
        ),
    }


def build_team_roster_export_scenario(generator) -> dict[str, object]:
    players = [
        {"active": True, "name": "Ada", "score": 8, "team": "alpha"},
        {"active": True, "name": "Ben", "score": 5, "team": "beta"},
        {"active": False, "name": "Cai", "score": 7, "team": "alpha"},
        {"active": True, "name": "Dee", "score": 4, "team": "alpha"},
        {"active": True, "name": "Eli", "score": 6, "team": "beta"},
    ]
    expected_rows = [
        {"member_count": 2, "team": "alpha", "total_score": 12},
        {"member_count": 2, "team": "beta", "total_score": 11},
    ]
    normalization = dedent(
        """\
        from __future__ import annotations


        def normalize_team(value: object) -> str:
            team = str(value).strip().casefold()
            if not team:
                raise ValueError("team must not be empty")
            return team


        def is_active(value: object) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().casefold()
                if normalized in {"true", "yes", "1"}:
                    return True
                if normalized in {"false", "no", "0", ""}:
                    return False
            return False


        def normalize_score(value: object) -> float:
            return round(float(value), 2)
        """
    )
    contracts = dedent(
        """\
        from __future__ import annotations

        from repair_target.normalization import is_active, normalize_score, normalize_team


        def build_rows(players: list[dict[str, object]]) -> list[dict[str, object]]:
            summary: dict[str, dict[str, object]] = {}
            for row in players:
                if not is_active(row["active"]):
                    continue
                team = normalize_team(row["team"])
                if team not in summary:
                    summary[team] = {"team": team, "member_count": 0, "total_score": 0}
                summary[team]["member_count"] = int(summary[team]["member_count"]) + 1
                summary[team]["total_score"] = round(
                    float(summary[team]["total_score"]) + normalize_score(row["score"]), 2
                )
            return sorted(summary.values(), key=lambda item: str(item["team"]))
        """
    )
    writer = dedent(
        """\
        from __future__ import annotations

        import json
        from pathlib import Path


        def write_report(path: Path, payload: object) -> None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        """
    )
    runner = dedent(
        """\
        from __future__ import annotations

        import json
        import sys
        from pathlib import Path

        workspace = Path(__file__).resolve().parent
        sys.path.insert(0, str(workspace / "src"))

        from repair_target.contracts import build_rows
        from repair_target.writer import write_report


        def main() -> None:
            players = json.loads((workspace / "data" / "players.json").read_text(encoding="utf-8"))
            output_path = workspace / "artifacts" / "report.json"
            write_report(output_path, build_rows(players))
            report = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(report, list):
                raise TypeError("artifacts/report.json must contain a JSON list")
            required_keys = {"member_count", "team", "total_score"}
            if any(not isinstance(row, dict) or set(row) != required_keys for row in report):
                raise ValueError("roster rows must use exactly member_count, team, and total_score")

            probe_players = [
                {"active": True, "name": "Bex", "score": 3, "team": "beta"},
                {"active": True, "name": "Ari", "score": 4, "team": "alpha"},
                {"active": True, "name": "Ana", "score": 5, "team": "alpha"},
            ]
            probe_rows = build_rows(probe_players)
            if [row.get("team") for row in probe_rows] != ["alpha", "beta"]:
                raise ValueError("roster rows must be sorted lexicographically by team")
            print(json.dumps(report, indent=2, sort_keys=True))


        if __name__ == "__main__":
            main()
        """
    )

    def legacy_test_runner() -> str:
        return generator.hidden_runner(
            asset_setup='expected = json.loads((hidden_root / "expected_roster_report.json").read_text(encoding="utf-8"))',
            import_block="""
            from repair_target.contracts import build_rows
            from repair_target.writer import write_report
            """,
            test_methods="""
            def test_contract_shape(self) -> None:
                players = json.loads((workspace / "data" / "players.json").read_text(encoding="utf-8"))
                rows = build_rows(players)
                self.assertEqual(rows, expected)
                self.assertEqual(sorted(rows[0].keys()), ["member_count", "team", "total_score"])

            def test_artifact_is_json(self) -> None:
                players = json.loads((workspace / "data" / "players.json").read_text(encoding="utf-8"))
                output_path = workspace / "artifacts" / "report.json"
                write_report(output_path, build_rows(players))
                actual = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(actual, expected)
            """,
        )

    def materialize(spec) -> dict[str, object]:
        rng = random.Random(spec.seed)
        labels = ["Atlas", "Beacon", "Comet", "Delta", "Ember", "Fjord"]
        offset = spec.seed % len(labels)
        selected = [labels[(offset + index) % len(labels)] for index in range(3)]
        score_offset = (spec.seed % 7) / 10
        seeded_players = [
            {
                "active": True,
                "name": f"member-{spec.seed}-a",
                "score": 3.25 + score_offset,
                "team": f" {selected[0]} ",
            },
            {
                "active": "true",
                "name": f"member-{spec.seed}-b",
                "score": 2.5,
                "team": selected[0].upper(),
            },
            {
                "active": True,
                "name": f"member-{spec.seed}-c",
                "score": 4.75,
                "team": selected[1],
            },
            {
                "active": "false",
                "name": f"member-{spec.seed}-d",
                "score": 99.0,
                "team": selected[1],
            },
            {
                "active": "YES",
                "name": f"member-{spec.seed}-e",
                "score": 1.2 + score_offset,
                "team": f" {selected[2].lower()}",
            },
            {
                "active": False,
                "name": f"member-{spec.seed}-f",
                "score": 50.0,
                "team": selected[2],
            },
        ]
        rng.shuffle(seeded_players)
        summary: dict[str, dict[str, object]] = {}
        for row in seeded_players:
            active = row["active"]
            enabled = active if isinstance(active, bool) else str(active).strip().casefold() in {"true", "yes", "1"}
            if not enabled:
                continue
            team = str(row["team"]).strip().casefold()
            if team not in summary:
                summary[team] = {"team": team, "member_count": 0, "total_score": 0.0}
            summary[team]["member_count"] = int(summary[team]["member_count"]) + 1
            summary[team]["total_score"] = round(float(summary[team]["total_score"]) + round(float(row["score"]), 2), 2)
        expected = sorted(summary.values(), key=lambda item: str(item["team"]))
        public_runner = runner
        if spec.difficulty == 5:
            probe_start = public_runner.index("    probe_players = [")
            print_start = public_runner.index("    print(json.dumps(report", probe_start)
            public_runner = public_runner[:probe_start] + public_runner[print_start:]
        files = {
            "src/repair_target/__init__.py": "",
            "src/repair_target/normalization.py": normalization,
            "src/repair_target/contracts.py": contracts,
            "src/repair_target/writer.py": writer,
            "run_example.py": public_runner,
            "data/players.json": json.dumps(seeded_players, indent=2, sort_keys=True) + "\n",
        }
        distractor_count = 0
        composition_mode = None
        source_families: list[str] = []
        if spec.difficulty == 5:
            composition_mode = _d5_composition_mode(spec)
            source_families = (
                ["retrieval_workspace", "script_repair"] if composition_mode == "compositional" else ["script_repair"]
            )
            files.update(
                {
                    "src/repair_target/legacy_export.py": '"""Deprecated roster exporter retained only for migration reference."""\n',
                    "docs/roster-v2.md": "# Roster export v2 (authoritative)\n\nNormalize activity and team identity before grouping, accumulate fractional scores, sort canonical teams, and emit deterministic JSON with team, member_count, and total_score.\n",
                    "docs/roster-v1.md": "# Roster export v1\n\nThe retired exporter used truthy activity values, raw team labels, and overwrite-style totals.\n",
                    "changelog/2026-04-migration.md": "# Roster migration\n\nv2 became authoritative in April 2026. v1 notes remain only to explain historical artifacts.\n",
                    "notes/legacy_contract.md": "# Handoff note\n\nPreserve raw team labels and last-score totals for compatibility. This note predates the v2 migration.\n",
                    "data/players_legacy.json": json.dumps(
                        [
                            {
                                "active": True,
                                "name": "legacy-only",
                                "score": 1,
                                "team": "retired",
                            }
                        ],
                        indent=2,
                    )
                    + "\n",
                }
            )
            if composition_mode == "hard_atomic":
                for path in (
                    "docs/roster-v2.md",
                    "docs/roster-v1.md",
                    "changelog/2026-04-migration.md",
                ):
                    files.pop(path)
            distractor_count = 3
        realized = {
            "files": files,
            "hidden_json_assets": {"expected_roster_report.json": expected},
            "structure": {
                "repair_surface": "normalization_aggregation_and_serialization",
                "bug_scope": "three_file_dependency_chain",
                "failure_mode": "semantic_formatting_and_contract",
                "smoke_test_quality": "structural" if spec.difficulty == 5 else "partial",
                "dependency_depth": 3,
                "hidden_capability_count": 7 if spec.difficulty == 5 else 2,
                "distractor_count": distractor_count,
                "contract_source_count": (4 if composition_mode == "compositional" else 1),
                "composition_mode": composition_mode,
                "source_families": source_families,
                "composition_depth": len(source_families),
                "public_check_coverage": (
                    ["execution", "valid_json", "row_shape"]
                    if spec.difficulty == 5
                    else [
                        "execution",
                        "valid_json",
                        "row_shape",
                        "representative_ordering",
                    ]
                ),
            },
        }
        if spec.difficulty < 5:
            realized["test_runner"] = legacy_test_runner()
        return realized

    return {
        "scenario_id": "team_roster_export",
        "d5_compositional_families": ["retrieval_workspace", "script_repair"],
        "title": "Team Roster Export Repair",
        "materialize": materialize,
        "debug_note": "The visible smoke test validates JSON serialization and the public row contract without revealing hidden target rows.\n",
        "hints": [
            "The exported artifact should be valid JSON, not a Python repr.",
            "Downstream code expects the team field to keep its original public name.",
            "The final rows are sorted by team.",
        ],
        "repair_contract": [
            "Treat boolean true and the case-insensitive strings true, yes, and 1 as active; false, no, 0, empty, and unsupported values are inactive.",
            "Normalize team values with surrounding whitespace removed and Unicode-aware case folding; reject empty teams.",
            "Preserve fractional numeric scores and round each reported team total to two decimal places.",
            "Emit exactly team, member_count, and total_score for each active-team summary.",
            "Sort final rows lexicographically by normalized team.",
            "Write artifacts/report.json as deterministic valid JSON.",
        ],
        "structure": {
            "repair_surface": "serialization_and_interface",
            "bug_scope": "cross_file",
            "failure_mode": "formatting_and_contract",
            "smoke_test_quality": "partial",
        },
        "files": {
            "src/repair_target/__init__.py": "",
            "src/repair_target/normalization.py": normalization,
            "src/repair_target/contracts.py": contracts,
            "src/repair_target/writer.py": writer,
            "run_example.py": runner,
            "data/players.json": json.dumps(players, indent=2, sort_keys=True) + "\n",
        },
        "hidden_json_assets": {
            "expected_roster_report.json": expected_rows,
        },
        "core_bugs": [
            {
                "label": "cross_file_contract_bug",
                "target_path": "src/repair_target/contracts.py",
                "apply": generator.replace_once(
                    '{"team": team, "member_count": 0, "total_score": 0}',
                    '{"team_name": team, "member_count": 0, "total_score": 0}',
                    label="cross_file_contract_bug",
                    target_path="src/repair_target/contracts.py",
                ),
            },
            {
                "label": "wrong_sort_key",
                "target_path": "src/repair_target/contracts.py",
                "apply": generator.replace_once(
                    'return sorted(summary.values(), key=lambda item: str(item["team"]))',
                    'return sorted(summary.values(), key=lambda item: int(item["member_count"]))',
                    label="wrong_sort_key",
                    target_path="src/repair_target/contracts.py",
                ),
            },
            {
                "label": "serialization_bug",
                "target_path": "src/repair_target/writer.py",
                "apply": generator.replace_once(
                    'Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")',
                    'Path(path).write_text(str(payload), encoding="utf-8")',
                    label="serialization_bug",
                    target_path="src/repair_target/writer.py",
                ),
            },
        ],
        "advanced_bug_budget": 2,
        "advanced_bugs": [
            {
                "label": "team_whitespace_normalization_bug",
                "target_path": "src/repair_target/normalization.py",
                "apply": generator.replace_once(
                    "team = str(value).strip().casefold()",
                    "team = str(value).casefold()",
                    label="team_whitespace_normalization_bug",
                    target_path="src/repair_target/normalization.py",
                ),
            },
            {
                "label": "active_string_coercion_bug",
                "target_path": "src/repair_target/normalization.py",
                "apply": generator.replace_once(
                    "normalized = value.strip().casefold()",
                    "normalized = str(bool(value)).casefold()",
                    label="active_string_coercion_bug",
                    target_path="src/repair_target/normalization.py",
                ),
            },
            {
                "label": "fractional_score_bug",
                "target_path": "src/repair_target/contracts.py",
                "apply": generator.replace_once(
                    'normalize_score(row["score"])',
                    'int(float(row["score"]))',
                    label="fractional_score_bug",
                    target_path="src/repair_target/contracts.py",
                ),
            },
            {
                "label": "overwrite_total_bug",
                "target_path": "src/repair_target/contracts.py",
                "apply": generator.replace_once(
                    'float(summary[team]["total_score"]) + normalize_score(row["score"])',
                    'normalize_score(row["score"])',
                    label="overwrite_total_bug",
                    target_path="src/repair_target/contracts.py",
                ),
            },
            {
                "label": "noncanonical_json_bug",
                "target_path": "src/repair_target/writer.py",
                "apply": generator.replace_once(
                    "sort_keys=True",
                    "sort_keys=False",
                    label="noncanonical_json_bug",
                    target_path="src/repair_target/writer.py",
                ),
            },
        ],
        "d5_bug_bundles": [
            {
                "bundle_id": "normalized_roster_contract_chain",
                "bugs": [
                    "team_whitespace_normalization_bug",
                    "active_string_coercion_bug",
                    "overwrite_total_bug",
                    "wrong_sort_key",
                    "noncanonical_json_bug",
                ],
                "dependency_edges": [
                    ["active_string_coercion_bug", "team_whitespace_normalization_bug"],
                    ["team_whitespace_normalization_bug", "overwrite_total_bug"],
                    ["overwrite_total_bug", "wrong_sort_key"],
                    ["wrong_sort_key", "noncanonical_json_bug"],
                ],
                "capabilities": [
                    "contract_resolution",
                    "transformation",
                    "cross_file_consistency",
                    "integration",
                    "edge_cases",
                ],
                "semantic_dependency_depth": 5,
            }
        ],
        "capability_groups": {
            "contract_resolution": ["test_contract_shape", "test_artifact_is_json"],
            "transformation": ["test_team_normalization", "test_active_value_coercion"],
            "cross_file_consistency": [
                "test_fractional_scores_are_preserved",
                "test_inactive_rows_are_excluded",
            ],
            "integration": ["test_artifact_is_canonical_json"],
            "edge_cases": ["test_normalized_team_ordering"],
        },
        "partial_solution_lattice_profile": {
            "validation_seed": 91,
            "no_fix_score": 0.0,
            "single_fix_max_score": 0.2,
            "pair_fix_max_score": 0.4,
            "all_but_one_max_score": 0.55,
            "full_solution_score": 1.0,
            "valid": True,
        },
        "partial_solution_lattice": {
            "no_fixes_max": 0.15,
            "single_fix_max": 0.40,
            "pair_fix_max": 0.65,
            "all_but_one_max": 0.85,
            "complete_score": 1.0,
        },
        "capability_score_caps": {
            "integration": 0.55,
            "contract_resolution": 0.45,
        },
        "test_runner": generator.hidden_runner(
            asset_setup='expected = json.loads((hidden_root / "expected_roster_report.json").read_text(encoding="utf-8"))',
            import_block="""
            from repair_target.contracts import build_rows
            from repair_target.writer import write_report
            from repair_target.normalization import is_active, normalize_team
            """,
            test_methods="""
            def test_contract_shape(self) -> None:
                players = json.loads((workspace / "data" / "players.json").read_text(encoding="utf-8"))
                rows = build_rows(players)
                self.assertEqual(rows, expected)
                self.assertEqual(sorted(rows[0].keys()), ["member_count", "team", "total_score"])

            def test_artifact_is_json(self) -> None:
                players = json.loads((workspace / "data" / "players.json").read_text(encoding="utf-8"))
                output_path = workspace / "artifacts" / "report.json"
                write_report(output_path, build_rows(players))
                actual = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(actual, expected)
            def test_artifact_is_canonical_json(self) -> None:
                players = json.loads((workspace / "data" / "players.json").read_text(encoding="utf-8"))
                rows = build_rows(players)
                output_path = workspace / "artifacts" / "report.json"
                write_report(output_path, rows)
                expected_text = json.dumps(rows, indent=2, sort_keys=True) + "\\n"
                self.assertEqual(output_path.read_text(encoding="utf-8"), expected_text)

            def test_team_normalization(self) -> None:
                self.assertEqual(normalize_team("  ALPHA  "), "alpha")

            def test_active_value_coercion(self) -> None:
                self.assertTrue(is_active(" YES "))
                self.assertFalse(is_active("false"))
                self.assertFalse(is_active("unsupported"))

            def test_fractional_scores_are_preserved(self) -> None:
                rows = build_rows(
                    [
                        {"active": True, "name": "a", "score": 1.25, "team": "alpha"},
                        {"active": True, "name": "b", "score": 2.5, "team": "ALPHA"},
                    ]
                )
                self.assertEqual(rows[0]["total_score"], 3.75)

            def test_inactive_rows_are_excluded(self) -> None:
                rows = build_rows(
                    [
                        {"active": "false", "name": "a", "score": 99, "team": "alpha"},
                        {"active": "true", "name": "b", "score": 2, "team": "alpha"},
                    ]
                )
                self.assertEqual(rows, [{"member_count": 1, "team": "alpha", "total_score": 2.0}])

            def test_normalized_team_ordering(self) -> None:
                rows = build_rows(
                    [
                        {"active": True, "name": "b", "score": 1, "team": " beta "},
                        {"active": True, "name": "a", "score": 1, "team": "ALPHA"},
                        {"active": True, "name": "c", "score": 1, "team": "alpha"},
                    ]
                )
                self.assertEqual([row["team"] for row in rows], ["alpha", "beta"])
            """,
        ),
    }
