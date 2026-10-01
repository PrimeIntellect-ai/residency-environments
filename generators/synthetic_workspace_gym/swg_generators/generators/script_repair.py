from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

from swg_generators.evaluators.script_repair import ScriptRepairEvaluator
from swg_generators.generators.base import BaseGenerator, GeneratedPayload
from swg_generators.generators.common import (
    build_d5_composition_profile,
    build_difficulty_realization,
    select_visible_hints,
)
from swg_generators.generators.d5_quality import normalize_lattice_profile
from swg_generators.generators.difficulty_primitives import (
    coerce_defect_bundle,
)
from swg_generators.generators.script_repair_scenarios import (
    build_csv_schema_drift_scenario,
    build_team_roster_export_scenario,
    build_timestamp_normalization_scenario,
)
from swg_generators.schemas import EnvironmentFamily, EnvironmentSpec
from swg_generators.utils.io import write_json, write_text


class ScriptRepairGenerator(BaseGenerator):
    family = EnvironmentFamily.SCRIPT_REPAIR

    def replace_once(self, old: str, new: str, *, label: str, target_path: str):
        def apply(content: str) -> str:
            updated = content.replace(old, new, 1)
            if updated == content:
                raise ValueError(f"Bug application '{label}' did not modify {target_path!r}; canonical source drifted.")
            return updated

        return apply

    def build_environment(
        self,
        spec: EnvironmentSpec,
        *,
        root: Path,
        visible_root: Path,
        hidden_root: Path,
    ) -> GeneratedPayload:
        scenarios = self.scenario_pool(spec)
        scenario = dict(self.select_scenario(spec, scenarios))
        materialize = scenario.get("materialize")
        if callable(materialize):
            scenario.update(dict(materialize(spec)))
        partner = None
        compositional_families = list(scenario.get("d5_compositional_families", []))
        if compositional_families:
            partner = EnvironmentFamily(str(compositional_families[0]))
        composition = build_d5_composition_profile(
            self.family,
            spec.difficulty,
            spec.seed,
            partner=partner,
            mode=spec.generation_params.get("composition_mode"),
        )
        selected_bugs, bug_selection = self.select_bugs(scenario, spec)
        bug_selection.update(composition)

        correct_files = dict(scenario["files"])
        composition_artifacts: dict[str, str] = {}
        composition_spec: dict[str, object] = {}
        if composition.get("composition_mode") == "compositional":
            contract_lines = "\n".join(f"- {line}" for line in scenario.get("repair_contract", []))
            correct_files.setdefault(
                "docs/api_contract.md",
                "# Public API contract\n\n"
                f"{contract_lines}\n\n"
                "The contract describes behavior, not implementation locations.\n",
            )
            correct_files.setdefault(
                "changelog/schema_v4.md",
                "# Schema v4 rollout\n\n"
                "Schema v4 is the active production contract. Canonicalization must "
                "happen before domain filtering, aggregation, and serialization.\n\n"
                "Materialize the resolved decision at `artifacts/resolved_contract.json` "
                "with `schema_version` set to `v4` and `scenario_id` set to the active task id.\n",
            )
            correct_files.setdefault(
                "logs/failing_request.json",
                json.dumps(
                    {
                        "status": "plausible_output_failed_semantic_validation",
                        "incident": str(scenario.get("debug_note", "")).strip(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
            correct_files.setdefault(
                "notes/old_handoff.md",
                "# Old handoff (historical)\n\n"
                "This predates schema v4 and may describe obsolete field names. "
                "Use the active contract and changelog when it conflicts.\n",
            )
            contract_payload = {
                "schema_version": "v4",
                "scenario_id": str(scenario["scenario_id"]),
            }
            composition_artifacts["artifacts/resolved_contract.json"] = (
                json.dumps(contract_payload, indent=2, sort_keys=True) + "\n"
            )
            guard = """from pathlib import Path as _ContractPath
import json as _contract_json

_contract_file = _ContractPath(__file__).resolve().parent / "artifacts" / "resolved_contract.json"
_contract = _contract_json.loads(_contract_file.read_text(encoding="utf-8"))
if not isinstance(_contract, dict):
    raise RuntimeError("resolved contract must be a JSON object")

"""
            future = "from __future__ import annotations\n\n"
            runner = correct_files["run_example.py"]
            if future not in runner:
                raise ValueError("run_example.py is missing the expected future import")
            correct_files["run_example.py"] = runner.replace(future, future + guard, 1)
            composition_spec = {
                "stages": [
                    {
                        "stage_id": "resolve_contract",
                        "required_inputs": [
                            "docs/api_contract.md",
                            "changelog/schema_v4.md",
                            "logs/failing_request.json",
                        ],
                        "produced_artifacts": ["artifacts/resolved_contract.json"],
                        "capability": "contract_resolution",
                    },
                    {
                        "stage_id": "repair_implementation",
                        "required_inputs": ["artifacts/resolved_contract.json"],
                        "produced_artifacts": sorted(str(bug["target_path"]) for bug in selected_bugs),
                        "capability": "integration",
                    },
                ],
                "dependencies": [["resolve_contract", "repair_implementation"]],
                "stage_count": 2,
                "downstream_consumes_upstream_artifact": True,
            }
        buggy_files = dict(correct_files)
        if composition_artifacts:
            buggy_files["artifacts/resolved_contract.json"] = (
                json.dumps(
                    {
                        "schema_version": "v3",
                        "scenario_id": str(scenario["scenario_id"]),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        applied_bug_labels: list[str] = []
        touched_files: set[str] = set()
        for bug in selected_bugs:
            target_path = bug["target_path"]
            buggy_files[target_path] = bug["apply"](buggy_files[target_path])
            touched_files.add(target_path)
            applied_bug_labels.append(bug["label"])

        for relative_path, content in buggy_files.items():
            write_text(visible_root / relative_path, content)

        if spec.difficulty >= 4:
            write_text(visible_root / "notes" / "incident_log.md", str(scenario["debug_note"]))

        disclosed_targets = sorted(touched_files)
        if spec.difficulty == 5:
            disclosed_targets = sorted(
                path
                for path in correct_files
                if path.startswith("src/") and path.endswith(".py") and not path.endswith("/__init__.py")
            )
            disclosed_targets.extend(sorted(composition_artifacts))

        task_descriptor = {
            "family": "script_repair",
            "scenario_id": scenario["scenario_id"],
            "entrypoint": "python run_example.py",
            "target_files": disclosed_targets,
            "hints": self.visible_hints(list(scenario["hints"]), spec.difficulty),
            "repair_contract": ([] if spec.difficulty == 5 else list(scenario.get("repair_contract", []))),
        }
        if spec.difficulty == 5:
            task_descriptor.update(
                {
                    "composition_mode": bug_selection.get("composition_mode"),
                    "source_families": bug_selection.get("source_families", []),
                    "composition_depth": bug_selection.get("composition_depth", 1),
                }
            )
            if bug_selection.get("composition_mode") == "compositional":
                task_descriptor["composition_spec"] = composition_spec
                task_descriptor["composition_evidence_paths"] = sorted(
                    path for path in correct_files if path.startswith(("analysis/", "changelog/", "docs/", "logs/"))
                )
        write_text(visible_root / "README.md", self.build_readme(scenario, task_descriptor))
        write_json(visible_root / "task.json", task_descriptor)

        hidden_runner = scenario["test_runner"]
        if callable(hidden_runner):
            hidden_runner = hidden_runner(task_descriptor)
        write_text(hidden_root / "run_hidden_tests.py", hidden_runner)
        for relative_path, payload in scenario.get("hidden_json_assets", {}).items():
            write_json(hidden_root / relative_path, payload)
        if composition_artifacts:
            write_json(
                hidden_root / "expected_resolved_contract.json",
                {
                    "schema_version": "v4",
                    "scenario_id": str(scenario["scenario_id"]),
                },
            )
        reference_solution = {
            "files": {
                **{path: correct_files[path] for path in sorted(touched_files)},
                **composition_artifacts,
            },
            "scenario_id": scenario["scenario_id"],
            "bug_labels": applied_bug_labels,
        }
        write_json(hidden_root / "solution_files.json", reference_solution)
        evaluator_config = {
            "runner": "run_hidden_tests.py",
            "scenario_id": scenario["scenario_id"],
            "capability_groups": (scenario.get("capability_groups", {}) if spec.difficulty == 5 else {}),
            "capability_score_caps": (scenario.get("capability_score_caps", {}) if spec.difficulty == 5 else {}),
            "public_entrypoint": ("run_example.py" if composition_artifacts else None),
            "required_json_artifacts": (
                [
                    {
                        "path": "artifacts/resolved_contract.json",
                        "expected_path": "expected_resolved_contract.json",
                        "capability": "contract_materialization",
                    }
                ]
                if composition_artifacts
                else []
            ),
            "required_artifact_failure_cap": 0.30,
        }
        write_json(hidden_root / "evaluator_config.json", evaluator_config)
        with tempfile.TemporaryDirectory(prefix="swg-script-baseline-") as temporary_dir:
            initial_workspace = Path(temporary_dir) / "workspace"
            shutil.copytree(visible_root, initial_workspace)
            initial_result = ScriptRepairEvaluator().evaluate(
                initial_workspace,
                SimpleNamespace(time_limit_seconds=spec.time_limit_seconds),
                hidden_root,
            )
        evaluator_config.update(
            {
                "initial_score": initial_result.score,
                "score_normalization": "initial_workspace_progress_v1",
            }
        )
        write_json(hidden_root / "evaluator_config.json", evaluator_config)

        metadata = {
            "task_descriptor": task_descriptor,
            "complexity_profile": spec.complexity_profile.to_dict() if spec.complexity_profile else {},
            "bug_labels": applied_bug_labels,
            "scenario_id": scenario["scenario_id"],
            "scenario_profile": scenario["structure"],
            "scenario_selection": {
                "requested_scenario_id": spec.scenario_id,
                "selection_mode": "explicit" if spec.scenario_id else "seed_modulo",
            },
            "difficulty_realization": build_difficulty_realization(
                spec.difficulty,
                hint_count=len(task_descriptor["hints"]),
                candidate_file_count=len(task_descriptor["target_files"]),
                applied_bug_count=len(applied_bug_labels),
                touched_file_count=(len(touched_files) + len(composition_artifacts)),
                core_bug_count=bug_selection["core_bug_count"],
                advanced_bug_count=bug_selection["advanced_bug_count"],
                nested_bug_selection=bug_selection["nested_bug_selection"],
                bug_bundle_id=bug_selection.get("bug_bundle_id"),
                dependency_edges=bug_selection.get("dependency_edges", []),
                capabilities=bug_selection.get("capabilities", []),
                composition_mode=bug_selection.get("composition_mode"),
                source_families=bug_selection.get("source_families", []),
                composition_depth=int(bug_selection.get("composition_depth", 1)),
                semantic_dependency_depth=int(bug_selection.get("semantic_dependency_depth", 0)),
                lattice_thresholds=dict(scenario.get("partial_solution_lattice", {})),
                lattice_validation=dict(scenario.get("partial_solution_lattice_profile", {})),
                dependency_depth=int(dict(scenario.get("structure", {})).get("dependency_depth", 2)),
                hidden_capability_count=int(dict(scenario.get("structure", {})).get("hidden_capability_count", 0)),
                distractor_count=int(dict(scenario.get("structure", {})).get("distractor_count", 0)),
                composition_spec=composition_spec,
                oracle_profile=normalize_lattice_profile(dict(scenario.get("partial_solution_lattice_profile", {}))),
                public_check_coverage=list(dict(scenario.get("structure", {})).get("public_check_coverage", [])),
                unmodified_reward_limit=0.15,
            ),
        }
        return GeneratedPayload(
            instruction="Repair the provided Python workspace so that the hidden tests pass.",
            metadata=metadata,
            reference_solution=reference_solution,
            evaluator_entrypoint="swg_generators.evaluators.script_repair:ScriptRepairEvaluator",
        )

    def select_bugs(
        self, scenario: dict[str, object], spec: EnvironmentSpec
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Select nested quality tiers while preserving legacy scenario definitions."""
        if "core_bugs" not in scenario:
            candidates = [bug for bug in scenario["bugs"] if bug["label"] != "syntax_error"]
            bug_budget = min(len(candidates), {1: 1, 2: 1, 3: 2, 4: 3, 5: 4}[spec.difficulty])
            selected = candidates[:bug_budget]
            return selected, {
                "core_bug_count": len(selected),
                "advanced_bug_count": 0,
                "nested_bug_selection": False,
            }

        core = [dict(bug) for bug in scenario.get("core_bugs", [])]
        advanced = [dict(bug) for bug in scenario.get("advanced_bugs", [])]
        if spec.difficulty == 5 and scenario.get("d5_bug_bundles"):
            bug_map = {str(bug["label"]): bug for bug in [*core, *advanced]}
            bundles = list(scenario["d5_bug_bundles"])
            bundle = coerce_defect_bundle(
                random.Random(f"{scenario['scenario_id']}:{spec.seed}:semantic-bundle").choice(bundles)
            )
            labels = list(bundle.defect_ids)
            missing = [label for label in labels if label not in bug_map]
            if missing:
                raise ValueError(f"D5 bundle {bundle.bundle_id!r} references unknown bugs: {missing}")
            selected = [bug_map[label] for label in labels]
            core_labels = {str(bug["label"]) for bug in core}
            composition_mode = "compositional" if spec.seed % 2 else "hard_atomic"
            source_families = (
                list(scenario.get("d5_compositional_families", []))
                if composition_mode == "compositional"
                else ["script_repair"]
            )
            return selected, {
                "core_bug_count": sum(str(bug["label"]) in core_labels for bug in selected),
                "advanced_bug_count": sum(str(bug["label"]) not in core_labels for bug in selected),
                "nested_bug_selection": False,
                "bug_bundle_id": bundle.bundle_id,
                "composition_mode": composition_mode,
                "source_families": source_families,
                "composition_depth": len(source_families),
                "dependency_edges": [list(edge) for edge in bundle.dependency_edges],
                "capabilities": list(bundle.capability_groups),
                "capability_groups": {name: list(defect_ids) for name, defect_ids in bundle.capability_groups.items()},
                "required_files": list(bundle.required_files),
                "semantic_dependency_depth": int(bundle.to_dict()["semantic_dependency_depth"]),
            }
        core_budget = min(len(core), {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}[spec.difficulty])
        selected_core = core[:core_budget]
        selected_advanced: list[dict[str, object]] = []
        if spec.difficulty == 5:
            advanced = [dict(bug) for bug in scenario.get("advanced_bugs", [])]
            advanced_budget = min(int(scenario.get("advanced_bug_budget", 2)), len(advanced))
            grouped: dict[str, list[dict[str, object]]] = {}
            for bug in advanced:
                grouped.setdefault(str(bug["target_path"]), []).append(bug)
            group_names = sorted(grouped)
            group_rng = random.Random(f"{scenario['scenario_id']}:{spec.seed}:advanced-groups")
            group_rng.shuffle(group_names)
            for target_path in group_names:
                choices = grouped[target_path]
                choice_rng = random.Random(f"{scenario['scenario_id']}:{spec.seed}:advanced:{target_path}")
                selected_advanced.append(choice_rng.choice(choices))
                if len(selected_advanced) >= advanced_budget:
                    break
            if len(selected_advanced) < advanced_budget:
                remaining = [bug for bug in advanced if bug not in selected_advanced]
                fill_rng = random.Random(f"{scenario['scenario_id']}:{spec.seed}:advanced-fill")
                fill_rng.shuffle(remaining)
                selected_advanced.extend(remaining[: advanced_budget - len(selected_advanced)])
        return selected_core + selected_advanced, {
            "core_bug_count": len(selected_core),
            "advanced_bug_count": len(selected_advanced),
            "nested_bug_selection": True,
        }

    def visible_hints(self, hints: list[str], difficulty: int) -> list[str]:
        return select_visible_hints(hints, difficulty)

    def build_readme(self, scenario: dict[str, object], task_descriptor: dict[str, object]) -> str:
        hints = "\n".join(f"- {hint}" for hint in task_descriptor["hints"])
        targets = "\n".join(f"- `{item}`" for item in task_descriptor["target_files"])
        contract = "\n".join(f"- {item}" for item in task_descriptor.get("repair_contract", []))
        contract_section = f"## Expected behavior\n{contract}\n\n" if contract else ""
        return (
            f"# {scenario['title']}\n\n"
            "One or more Python files in this workspace are buggy. Repair the code so the hidden tests pass.\n\n"
            "## What to preserve\n"
            "- Keep the public function names stable.\n"
            "- Prefer targeted fixes over rewrites.\n"
            "- Use the visible smoke test command to sanity-check your changes.\n\n"
            "## Smoke test\n"
            f"- `{task_descriptor['entrypoint']}`\n\n"
            "## Likely target files\n"
            f"{targets}\n\n"
            f"{contract_section}"
            "## Hints\n"
            f"{hints}\n"
        )

    def json_runner(self, *, import_block: str, expression: str) -> str:
        lines = [
            "from __future__ import annotations",
            "",
            "import json",
            "import sys",
            "from pathlib import Path",
            "",
            "workspace = Path(__file__).resolve().parent",
            'sys.path.insert(0, str(workspace / "src"))',
            "",
        ]
        lines.extend(dedent(import_block).strip().splitlines())
        lines.extend(
            [
                "",
                "",
                "def main() -> None:",
                f"    print(json.dumps({expression}, indent=2, sort_keys=True))",
                "",
                "",
                'if __name__ == "__main__":',
                "    main()",
            ]
        )
        return "\n".join(lines) + "\n"

    def hidden_runner(self, *, asset_setup: str, import_block: str, test_methods: str) -> str:
        lines = [
            "from __future__ import annotations",
            "",
            "import json",
            "import sys",
            "import unittest",
            "from pathlib import Path",
            "",
            "",
            "def build_suite(workspace: Path) -> unittest.TestSuite:",
            "    hidden_root = Path(__file__).resolve().parent",
        ]
        lines.extend(f"    {line}" if line else "" for line in dedent(asset_setup).strip().splitlines())
        lines.append('    sys.path.insert(0, str(workspace / "src"))')
        lines.extend(f"    {line}" if line else "" for line in dedent(import_block).strip().splitlines())
        lines.extend(
            [
                "",
                "    class HiddenTests(unittest.TestCase):",
            ]
        )
        lines.extend(f"        {line}" if line else "" for line in dedent(test_methods).strip().splitlines())
        lines.extend(
            [
                "",
                "    return unittest.defaultTestLoader.loadTestsFromTestCase(HiddenTests)",
                "",
                "",
                "def main() -> None:",
                "    workspace = Path(sys.argv[1]).resolve()",
                "    suite = build_suite(workspace)",
                "    result = unittest.TextTestRunner(verbosity=2).run(suite)",
                "    payload = {",
                '        "success": result.wasSuccessful(),',
                '        "score": 1.0 if result.wasSuccessful() else 0.0,',
                '        "subscores": {',
                '            "tests_passed": result.testsRun - len(result.failures) - len(result.errors),',
                '            "tests_total": result.testsRun,',
                "        },",
                '        "failure_labels": ["hidden_tests_failed"] if not result.wasSuccessful() else [],',
                '        "diagnostics": {',
                '            "tests_run": result.testsRun,',
                '            "failures": [case[0].id() for case in result.failures],',
                '            "errors": [case[0].id() for case in result.errors],',
                "        },",
                "    }",
                "    print(json.dumps(payload, sort_keys=True))",
                "    sys.exit(0 if result.wasSuccessful() else 1)",
                "",
                "",
                'if __name__ == "__main__":',
                "    main()",
            ]
        )
        return "\n".join(lines) + "\n"

    def scenario_pool(self, spec: EnvironmentSpec | None = None) -> list[dict[str, object]]:
        scenarios = [
            self.inventory_report_scenario(),
            self.path_batch_scenario(),
            build_csv_schema_drift_scenario(self),
            build_timestamp_normalization_scenario(self),
            build_team_roster_export_scenario(self),
        ]
        if spec is not None and spec.difficulty == 5 and spec.scenario_id is None:
            return [scenario for scenario in scenarios if scenario.get("d5_bug_bundles")]
        return scenarios

    def inventory_report_scenario(self) -> dict[str, object]:
        items = [
            {"name": "alpha", "status": "active", "count": 3},
            {"name": "beta", "status": "archived", "count": 2},
            {"name": "gamma", "status": "active", "count": 4},
            {"name": "delta", "status": "active", "count": 6},
            {"name": "epsilon", "status": "archived", "count": 3},
        ]
        expected_report = {
            "summary": {"active": 13, "archived": 5},
            "rolling_average": [3.5, 5.0],
        }
        analytics = """from __future__ import annotations


def rolling_average(values: list[int], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    if len(values) < window:
        return []
    averages = []
    for index in range(len(values) - window + 1):
        chunk = values[index : index + window]
        averages.append(round(sum(chunk) / window, 2))
    return averages


def summarize_items(rows: list[dict[str, object]]) -> dict[str, int]:
    summary = {"active": 0, "archived": 0}
    for row in rows:
        status = str(row["status"]).lower()
        if status not in summary:
            continue
        summary[status] += int(row["count"])
    return summary
"""
        report = """from __future__ import annotations

import json
from pathlib import Path

from repair_target.analytics import rolling_average, summarize_items


def build_report(data_path: Path) -> dict[str, object]:
    rows = json.loads(Path(data_path).read_text(encoding="utf-8"))
    active_counts = [int(row["count"]) for row in rows if str(row["status"]).lower() == "active"]
    return {
        "summary": summarize_items(rows),
        "rolling_average": rolling_average(active_counts, 2),
    }
"""
        runner = """from __future__ import annotations

import json
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace / "src"))

from repair_target.report import build_report


def main() -> None:
    data_path = workspace / "data" / "items.json"
    print(json.dumps(build_report(data_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"""
        test_runner = """from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


def build_suite(workspace: Path) -> unittest.TestSuite:
    hidden_root = Path(__file__).resolve().parent
    expected = json.loads((hidden_root / "expected_inventory_report.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(workspace / "src"))
    from repair_target.analytics import rolling_average, summarize_items
    from repair_target.report import build_report

    class HiddenTests(unittest.TestCase):
        def test_rolling_average(self) -> None:
            self.assertEqual(rolling_average([3, 4, 6], 2), [3.5, 5.0])

        def test_summary(self) -> None:
            rows = json.loads((workspace / "data" / "items.json").read_text(encoding="utf-8"))
            self.assertEqual(summarize_items(rows), expected["summary"])

        def test_report(self) -> None:
            actual = build_report(workspace / "data" / "items.json")
            self.assertEqual(actual, expected)

    return unittest.defaultTestLoader.loadTestsFromTestCase(HiddenTests)


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    suite = build_suite(workspace)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "success": result.wasSuccessful(),
        "score": 1.0 if result.wasSuccessful() else 0.0,
        "subscores": {
            "tests_passed": result.testsRun - len(result.failures) - len(result.errors),
            "tests_total": result.testsRun,
        },
        "failure_labels": ["hidden_tests_failed"] if not result.wasSuccessful() else [],
        "diagnostics": {
            "tests_run": result.testsRun,
            "failures": [case[0].id() for case in result.failures],
            "errors": [case[0].id() for case in result.errors],
        },
    }
    print(json.dumps(payload, sort_keys=True))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
"""
        return {
            "scenario_id": "inventory_report",
            "title": "Inventory Report Repair",
            "debug_note": "The smoke test is reliable here, but some hidden assertions care about both summary counts and report ordering.\n",
            "hints": [
                "The rolling average should include every complete 2-item window.",
                "Archived rows still contribute to the summary counts.",
                "The smoke test should print a stable JSON report.",
            ],
            "structure": {
                "repair_surface": "aggregation_and_reporting",
                "bug_scope": "cross_file",
                "failure_mode": "semantic",
                "smoke_test_quality": "informative",
            },
            "files": {
                "src/repair_target/__init__.py": "",
                "src/repair_target/analytics.py": analytics,
                "src/repair_target/report.py": report,
                "run_example.py": runner,
                "data/items.json": json.dumps(items, indent=2, sort_keys=True) + "\n",
            },
            "hidden_json_assets": {
                "expected_inventory_report.json": expected_report,
            },
            "bugs": [
                {
                    "label": "off_by_one",
                    "target_path": "src/repair_target/analytics.py",
                    "apply": self.replace_once(
                        "range(len(values) - window + 1)",
                        "range(len(values) - window)",
                        label="off_by_one",
                        target_path="src/repair_target/analytics.py",
                    ),
                },
                {
                    "label": "wrong_condition",
                    "target_path": "src/repair_target/analytics.py",
                    "apply": self.replace_once(
                        "if status not in summary:",
                        'if status == "archived":',
                        label="wrong_condition",
                        target_path="src/repair_target/analytics.py",
                    ),
                },
                {
                    "label": "wrong_return_value",
                    "target_path": "src/repair_target/report.py",
                    "apply": self.replace_once(
                        '"rolling_average": rolling_average(active_counts, 2),',
                        '"rolling_average": rolling_average(active_counts, 3),',
                        label="wrong_return_value",
                        target_path="src/repair_target/report.py",
                    ),
                },
            ],
            "test_runner": lambda task: test_runner,
        }

    def path_batch_scenario(self) -> dict[str, object]:
        measurements = "day,value\nmon,5\ntue,8\nwed,3\nthu,9\n"
        expected_summary = {"count": 4, "maximum": 9, "minimum": 3, "total": 25}
        io_helpers = """from __future__ import annotations

from pathlib import Path


def load_measurements(data_dir: Path) -> list[int]:
    path = data_dir / "measurements.csv"
    rows = Path(path).read_text(encoding="utf-8").strip().splitlines()[1:]
    return [int(row.split(",")[1]) for row in rows]
"""
        batch = """from __future__ import annotations

from pathlib import Path

from repair_target.io_helpers import load_measurements


def compute_batch_summary(base_dir: Path) -> dict[str, int]:
    data_dir = Path(base_dir) / "data"
    values = load_measurements(data_dir)
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "total": sum(values),
    }
"""
        runner = """from __future__ import annotations

import json
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace / "src"))

from repair_target.batch import compute_batch_summary


def main() -> None:
    print(json.dumps(compute_batch_summary(workspace), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"""
        test_runner = """from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


def build_suite(workspace: Path) -> unittest.TestSuite:
    hidden_root = Path(__file__).resolve().parent
    expected = json.loads((hidden_root / "expected_batch_summary.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(workspace / "src"))
    from repair_target.batch import compute_batch_summary
    from repair_target.io_helpers import load_measurements

    class HiddenTests(unittest.TestCase):
        def test_loader(self) -> None:
            values = load_measurements(workspace / "data")
            self.assertEqual(values, [5, 8, 3, 9])

        def test_summary(self) -> None:
            self.assertEqual(compute_batch_summary(workspace), expected)

    return unittest.defaultTestLoader.loadTestsFromTestCase(HiddenTests)


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    suite = build_suite(workspace)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    payload = {
        "success": result.wasSuccessful(),
        "score": 1.0 if result.wasSuccessful() else 0.0,
        "subscores": {
            "tests_passed": result.testsRun - len(result.failures) - len(result.errors),
            "tests_total": result.testsRun,
        },
        "failure_labels": ["hidden_tests_failed"] if not result.wasSuccessful() else [],
        "diagnostics": {
            "tests_run": result.testsRun,
            "failures": [case[0].id() for case in result.failures],
            "errors": [case[0].id() for case in result.errors],
        },
    }
    print(json.dumps(payload, sort_keys=True))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
"""
        return {
            "scenario_id": "path_batch",
            "title": "Batch Summary Repair",
            "debug_note": "This smoke test is execution-sensitive: import issues and small path errors surface immediately, but hidden tests still check the final aggregation.\n",
            "hints": [
                "The measurement loader should read from the visible `data` directory.",
                "The batch summary should aggregate every row in the CSV.",
                "If the module fails to import, inspect recent edits around function signatures and imports.",
            ],
            "structure": {
                "repair_surface": "file_path_and_batching",
                "bug_scope": "local",
                "failure_mode": "execution_and_semantic",
                "smoke_test_quality": "informative",
            },
            "files": {
                "src/repair_target/__init__.py": "",
                "src/repair_target/io_helpers.py": io_helpers,
                "src/repair_target/batch.py": batch,
                "run_example.py": runner,
                "data/measurements.csv": measurements,
            },
            "hidden_json_assets": {
                "expected_batch_summary.json": expected_summary,
            },
            "bugs": [
                {
                    "label": "missing_import",
                    "target_path": "src/repair_target/io_helpers.py",
                    "apply": self.replace_once(
                        "from pathlib import Path\n\n",
                        "",
                        label="missing_import",
                        target_path="src/repair_target/io_helpers.py",
                    ),
                },
                {
                    "label": "file_path_issue",
                    "target_path": "src/repair_target/io_helpers.py",
                    "apply": self.replace_once(
                        'data_dir / "measurements.csv"',
                        'data_dir.parent / "measurements.csv"',
                        label="file_path_issue",
                        target_path="src/repair_target/io_helpers.py",
                    ),
                },
                {
                    "label": "aggregation_bug",
                    "target_path": "src/repair_target/batch.py",
                    "apply": self.replace_once(
                        '"total": sum(values),',
                        '"total": len(values),',
                        label="aggregation_bug",
                        target_path="src/repair_target/batch.py",
                    ),
                },
                {
                    "label": "syntax_error",
                    "target_path": "src/repair_target/batch.py",
                    "apply": self.replace_once(
                        "def compute_batch_summary(base_dir: Path) -> dict[str, int]:",
                        "def compute_batch_summary(base_dir: Path) -> dict[str, int]",
                        label="syntax_error",
                        target_path="src/repair_target/batch.py",
                    ),
                },
            ],
            "test_runner": lambda task: test_runner,
        }
