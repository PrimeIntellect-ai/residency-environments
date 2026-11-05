from __future__ import annotations

import json
from textwrap import dedent


def build_sales_csv_pipeline_scenario(generator) -> dict[str, object]:
    sales_csv = (
        "region,segment,amount,active\n"
        " North ,enterprise,120.0,true\n"
        "south,smb,75.5,false\n"
        "east,enterprise,90.0,true\n"
        "north,midmarket,45.0,true\n"
        "west,smb,60.0,true\n"
    )
    expected_output = [
        {"region": "east", "row_count": 1, "total_amount": 90.0},
        {"region": "north", "row_count": 2, "total_amount": 165.0},
        {"region": "west", "row_count": 1, "total_amount": 60.0},
    ]
    config = {
        "include_inactive": False,
        "input_path": "data/sales.csv",
        "output_path": "artifacts/normalized_sales.json",
    }
    csv_loader = dedent(
        """\
        from __future__ import annotations

        import csv
        from pathlib import Path


        def load_rows(path: Path) -> list[dict[str, str]]:
            with Path(path).open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        """
    )
    steps = dedent(
        """\
        from __future__ import annotations


        def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
            normalized = []
            for row in rows:
                normalized.append(
                    {
                        "region": str(row["region"]).strip().lower(),
                        "active": str(row["active"]).strip().lower() == "true",
                        "amount": float(row["amount"]),
                    }
                )
            return normalized


        def build_summary(rows: list[dict[str, object]], *, include_inactive: bool) -> list[dict[str, object]]:
            summary: dict[str, dict[str, object]] = {}
            for row in rows:
                if not include_inactive and not row["active"]:
                    continue
                region = str(row["region"])
                if region not in summary:
                    summary[region] = {"region": region, "row_count": 0, "total_amount": 0.0}
                summary[region]["row_count"] = int(summary[region]["row_count"]) + 1
                summary[region]["total_amount"] = round(
                    float(summary[region]["total_amount"]) + float(row["amount"]),
                    1,
                )
            return sorted(summary.values(), key=lambda item: str(item["region"]))
        """
    )
    return {
        "scenario_id": "sales_csv_pipeline",
        "title": "Sales CSV Normalization Pipeline",
        "required_output_path": "artifacts/normalized_sales.json",
        "debug_note": "This smoke test is useful, but hidden evaluation also checks that inactive rows stay excluded and the artifact remains valid JSON.\n",
        "hints": [
            "Whitespace and casing in the CSV need normalization before aggregation.",
            "Inactive rows should be filtered only when the config asks for it.",
            "The final artifact path is part of the contract.",
        ],
        "structure": {
            "repair_surface": "config_and_transform",
            "bug_scope": "cross_file",
            "failure_mode": "semantic_and_formatting",
            "smoke_test_quality": "informative",
        },
        "files": {
            "src/pipeline_app/__init__.py": "",
            "src/pipeline_app/csv_loader.py": csv_loader,
            "src/pipeline_app/io_utils.py": generator.io_utils_module(),
            "src/pipeline_app/steps.py": steps,
            "run_pipeline.py": dedent(
                """\
                from __future__ import annotations

                import json
                import sys
                from pathlib import Path

                workspace = Path(__file__).resolve().parent
                sys.path.insert(0, str(workspace / "src"))

                from pipeline_app.csv_loader import load_rows
                from pipeline_app.io_utils import write_json
                from pipeline_app.steps import build_summary, normalize_rows


                def main() -> None:
                    config = json.loads((workspace / "config" / "pipeline_config.json").read_text(encoding="utf-8"))
                    rows = load_rows(workspace / config["input_path"])
                    normalized = normalize_rows(rows)
                    summary = build_summary(normalized, include_inactive=bool(config["include_inactive"]))
                    write_json(workspace / config["output_path"], summary)


                if __name__ == "__main__":
                    main()
                """
            ),
            "config/pipeline_config.json": json.dumps(config, indent=2, sort_keys=True) + "\n",
            "data/sales.csv": sales_csv,
        },
        "expected_output": expected_output,
        "bugs": [
            {
                "label": "stale_input_path",
                "target_path": "config/pipeline_config.json",
                "apply": generator.replace_once(
                    "data/sales.csv",
                    "data/sale.csv",
                    label="stale_input_path",
                    target_path="config/pipeline_config.json",
                ),
            },
            {
                "label": "wrong_output_path",
                "target_path": "config/pipeline_config.json",
                "apply": generator.replace_once(
                    "artifacts/normalized_sales.json",
                    "artifacts/sales.json",
                    label="wrong_output_path",
                    target_path="config/pipeline_config.json",
                ),
            },
            {
                "label": "missing_normalization_step",
                "target_path": "run_pipeline.py",
                "apply": generator.replace_once(
                    "normalized = normalize_rows(rows)",
                    "normalized = rows",
                    label="missing_normalization_step",
                    target_path="run_pipeline.py",
                ),
            },
            {
                "label": "wrong_filter_policy",
                "target_path": "src/pipeline_app/steps.py",
                "apply": generator.replace_once(
                    'if not include_inactive and not row["active"]:',
                    'if not include_inactive and row["active"]:',
                    label="wrong_filter_policy",
                    target_path="src/pipeline_app/steps.py",
                ),
            },
            {
                "label": "invalid_serialization",
                "target_path": "src/pipeline_app/io_utils.py",
                "apply": generator.replace_once(
                    'Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")',
                    'Path(path).write_text(str(payload), encoding="utf-8")',
                    label="invalid_serialization",
                    target_path="src/pipeline_app/io_utils.py",
                ),
            },
        ],
    }


def build_artifact_stitch_pipeline_scenario(generator) -> dict[str, object]:
    config = {
        "filenames": ["north.json", "south.json"],
        "fragment_dir": "data/fragments",
        "output_path": "artifacts/merged_report.json",
    }
    north_rows = [{"count": 2, "report": "hardware"}, {"count": 1, "report": "services"}]
    south_rows = [{"count": 3, "report": "hardware"}, {"count": 4, "report": "support"}]
    expected_output = [
        {"count": 5, "report": "hardware"},
        {"count": 1, "report": "services"},
        {"count": 4, "report": "support"},
    ]
    loader = dedent(
        """\
        from __future__ import annotations

        import json
        from pathlib import Path


        def load_fragments(base_dir: Path, filenames: list[str]) -> list[dict[str, object]]:
            rows: list[dict[str, object]] = []
            for name in filenames:
                rows.extend(json.loads((Path(base_dir) / name).read_text(encoding="utf-8")))
            return rows
        """
    )
    merge = dedent(
        """\
        from __future__ import annotations


        def stitch_fragments(rows: list[dict[str, object]]) -> list[dict[str, object]]:
            summary: dict[str, dict[str, object]] = {}
            for row in rows:
                report = str(row["report"])
                if report not in summary:
                    summary[report] = {"report": report, "count": 0}
                summary[report]["count"] = int(summary[report]["count"]) + int(row["count"])
            return sorted(summary.values(), key=lambda item: str(item["report"]))
        """
    )
    return {
        "scenario_id": "artifact_stitch_pipeline",
        "title": "Artifact Stitch Pipeline",
        "required_output_path": "artifacts/merged_report.json",
        "debug_note": "The smoke test may produce an artifact even when only one fragment was stitched or the final path drifted from the contract.\n",
        "hints": [
            "The fragment directory and filenames come from config, not hardcoded assumptions.",
            "The merge step needs to aggregate existing counts rather than counting rows.",
            "The final artifact is sorted by report name.",
        ],
        "structure": {
            "repair_surface": "artifact_merge",
            "bug_scope": "cross_file",
            "failure_mode": "semantic",
            "smoke_test_quality": "partially_informative",
        },
        "files": {
            "src/pipeline_app/__init__.py": "",
            "src/pipeline_app/io_utils.py": generator.io_utils_module(),
            "src/pipeline_app/loader.py": loader,
            "src/pipeline_app/merge.py": merge,
            "run_pipeline.py": dedent(
                """\
                from __future__ import annotations

                import json
                import sys
                from pathlib import Path

                workspace = Path(__file__).resolve().parent
                sys.path.insert(0, str(workspace / "src"))

                from pipeline_app.io_utils import write_json
                from pipeline_app.loader import load_fragments
                from pipeline_app.merge import stitch_fragments


                def main() -> None:
                    config = json.loads((workspace / "config" / "pipeline_config.json").read_text(encoding="utf-8"))
                    rows = load_fragments(workspace / config["fragment_dir"], list(config["filenames"]))
                    stitched = stitch_fragments(rows)
                    write_json(workspace / config["output_path"], stitched)


                if __name__ == "__main__":
                    main()
                """
            ),
            "config/pipeline_config.json": json.dumps(config, indent=2, sort_keys=True) + "\n",
            "data/fragments/north.json": json.dumps(north_rows, indent=2, sort_keys=True) + "\n",
            "data/fragments/south.json": json.dumps(south_rows, indent=2, sort_keys=True) + "\n",
        },
        "expected_output": expected_output,
        "bugs": [
            {
                "label": "stale_fragment_dir",
                "target_path": "config/pipeline_config.json",
                "apply": generator.replace_once(
                    "data/fragments",
                    "data/fragment",
                    label="stale_fragment_dir",
                    target_path="config/pipeline_config.json",
                ),
            },
            {
                "label": "wrong_output_path",
                "target_path": "config/pipeline_config.json",
                "apply": generator.replace_once(
                    "artifacts/merged_report.json",
                    "artifacts/report.json",
                    label="wrong_output_path",
                    target_path="config/pipeline_config.json",
                ),
            },
            {
                "label": "omitted_merge_stage",
                "target_path": "run_pipeline.py",
                "apply": generator.replace_once(
                    "stitched = stitch_fragments(rows)",
                    "stitched = rows",
                    label="omitted_merge_stage",
                    target_path="run_pipeline.py",
                ),
            },
            {
                "label": "aggregation_bug",
                "target_path": "src/pipeline_app/merge.py",
                "apply": generator.replace_once(
                    'summary[report]["count"] = int(summary[report]["count"]) + int(row["count"])',
                    'summary[report]["count"] = int(summary[report]["count"]) + 1',
                    label="aggregation_bug",
                    target_path="src/pipeline_app/merge.py",
                ),
            },
        ],
    }


def build_quality_gate_pipeline_scenario(generator) -> dict[str, object]:
    config = {
        "blocked_sources": ["legacy"],
        "input_path": "data/events.json",
        "minimum_quality": "pass",
        "output_path": "artifacts/quality_report.json",
    }
    events = [
        {"hours": 2.5, "quality": "pass", "source": "primary", "team": "Alpha"},
        {"hours": 3.0, "quality": "fail", "source": "primary", "team": "Alpha"},
        {"hours": 4.0, "quality": "pass", "source": "legacy", "team": "Beta"},
        {"hours": 1.5, "quality": "pass", "source": "primary", "team": "Beta"},
        {"hours": 5.0, "quality": "pass", "source": "primary", "team": "Alpha"},
    ]
    expected_output = [
        {"row_count": 2, "team": "alpha", "total_hours": 7.5},
        {"row_count": 1, "team": "beta", "total_hours": 1.5},
    ]
    quality = dedent(
        """\
        from __future__ import annotations


        def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
            normalized = []
            for row in rows:
                normalized.append(
                    {
                        "team": str(row["team"]).lower(),
                        "quality": str(row["quality"]).lower(),
                        "source": str(row["source"]).lower(),
                        "hours": float(row["hours"]),
                    }
                )
            return normalized


        def select_rows(
            rows: list[dict[str, object]],
            *,
            minimum_quality: str,
            blocked_sources: list[str],
        ) -> list[dict[str, object]]:
            selected = []
            for row in rows:
                if row["quality"] != minimum_quality:
                    continue
                if row["source"] in blocked_sources:
                    continue
                selected.append(row)
            return selected


        def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
            summary: dict[str, dict[str, object]] = {}
            for row in rows:
                team = str(row["team"])
                if team not in summary:
                    summary[team] = {"team": team, "row_count": 0, "total_hours": 0.0}
                summary[team]["row_count"] = int(summary[team]["row_count"]) + 1
                summary[team]["total_hours"] = round(
                    float(summary[team]["total_hours"]) + float(row["hours"]),
                    1,
                )
            return sorted(summary.values(), key=lambda item: str(item["team"]))
        """
    )
    return {
        "scenario_id": "quality_gate_pipeline",
        "title": "Data Quality Gate Pipeline",
        "required_output_path": "artifacts/quality_report.json",
        "debug_note": "The smoke test is only partially reliable here: a report may be emitted even when blocked sources or quality filters are wrong.\n",
        "hints": [
            "Rows must be normalized before quality and source filters run.",
            "Blocked sources should be excluded even if they otherwise pass validation.",
            "The report aggregates total hours per team.",
        ],
        "structure": {
            "repair_surface": "multi_stage_transform",
            "bug_scope": "cross_file",
            "failure_mode": "semantic",
            "smoke_test_quality": "partially_informative",
        },
        "files": {
            "src/pipeline_app/__init__.py": "",
            "src/pipeline_app/io_utils.py": generator.io_utils_module(include_loader=True),
            "src/pipeline_app/quality.py": quality,
            "run_pipeline.py": dedent(
                """\
                from __future__ import annotations

                import json
                import sys
                from pathlib import Path

                workspace = Path(__file__).resolve().parent
                sys.path.insert(0, str(workspace / "src"))

                from pipeline_app.io_utils import load_rows, write_json
                from pipeline_app.quality import build_summary, normalize_rows, select_rows


                def main() -> None:
                    config = json.loads((workspace / "config" / "pipeline_config.json").read_text(encoding="utf-8"))
                    rows = load_rows(workspace / config["input_path"])
                    normalized = normalize_rows(rows)
                    filtered = select_rows(
                        normalized,
                        minimum_quality=str(config["minimum_quality"]),
                        blocked_sources=list(config["blocked_sources"]),
                    )
                    report = build_summary(filtered)
                    write_json(workspace / config["output_path"], report)


                if __name__ == "__main__":
                    main()
                """
            ),
            "config/pipeline_config.json": json.dumps(config, indent=2, sort_keys=True) + "\n",
            "data/events.json": json.dumps(events, indent=2, sort_keys=True) + "\n",
        },
        "expected_output": expected_output,
        "bugs": [
            {
                "label": "wrong_output_path",
                "target_path": "config/pipeline_config.json",
                "apply": generator.replace_once(
                    "artifacts/quality_report.json",
                    "artifacts/report.json",
                    label="wrong_output_path",
                    target_path="config/pipeline_config.json",
                ),
            },
            {
                "label": "missing_normalization_stage",
                "target_path": "run_pipeline.py",
                "apply": generator.replace_once(
                    "normalized = normalize_rows(rows)",
                    "normalized = rows",
                    label="missing_normalization_stage",
                    target_path="run_pipeline.py",
                ),
            },
            {
                "label": "wrong_filter_policy",
                "target_path": "src/pipeline_app/quality.py",
                "apply": generator.replace_once(
                    'if row["quality"] != minimum_quality:',
                    'if row["quality"] == minimum_quality:',
                    label="wrong_filter_policy",
                    target_path="src/pipeline_app/quality.py",
                ),
            },
            {
                "label": "helper_drift",
                "target_path": "src/pipeline_app/quality.py",
                "apply": generator.replace_once(
                    'float(summary[team]["total_hours"]) + float(row["hours"])',
                    'float(summary[team]["total_hours"]) + 1',
                    label="helper_drift",
                    target_path="src/pipeline_app/quality.py",
                ),
            },
        ],
    }
