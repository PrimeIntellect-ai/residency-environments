from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

from swg_generators.evaluators.metrics import row_overlap_metrics, weighted_match_score
from swg_generators.generators.base import BaseGenerator, GeneratedPayload
from swg_generators.generators.common import (
    build_d5_composition_profile,
    build_difficulty_realization,
    select_visible_hints,
)
from swg_generators.generators.d5_profiles import d5_profile_metadata_for_family
from swg_generators.generators.d5_quality import normalize_lattice_profile
from swg_generators.generators.pipeline_profile_scenarios import (
    build_profiled_team_hours_scenario,
)
from swg_generators.generators.pipeline_scenarios import (
    build_artifact_stitch_pipeline_scenario,
    build_quality_gate_pipeline_scenario,
    build_sales_csv_pipeline_scenario,
)
from swg_generators.schemas import EnvironmentFamily, EnvironmentSpec
from swg_generators.utils.io import write_json, write_text


class PipelineCompletionGenerator(BaseGenerator):
    family = EnvironmentFamily.PIPELINE

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
        rng = random.Random(spec.seed)
        scenarios = self.scenario_pool(rng, spec)
        scenario = self.select_scenario(spec, scenarios)
        composition = build_d5_composition_profile(
            self.family,
            spec.difficulty,
            spec.seed,
            mode=spec.generation_params.get("composition_mode"),
        )
        expected_output = scenario["expected_output"]
        correct_files = dict(scenario["files"])
        composition_spec: dict[str, object] = {}
        if composition.get("composition_mode") == "compositional" and scenario["scenario_id"] == "team_hours_pipeline":
            if scenario.get("profile_managed_composition"):
                composition_spec = dict(scenario.get("composition_spec", {}))
            else:
                correct_files = self.make_team_hours_compositional(correct_files)
                composition_spec = {
                    "stages": [
                        {
                            "stage_id": "normalize",
                            "required_inputs": [
                                "data/jobs.json",
                                "config/pipeline_config.json",
                            ],
                            "produced_artifacts": ["artifacts/normalized_jobs.json"],
                            "capability": "normalization",
                        },
                        {
                            "stage_id": "aggregate_and_serialize",
                            "required_inputs": ["artifacts/normalized_jobs.json"],
                            "produced_artifacts": ["artifacts/summary.json"],
                            "capability": "integration",
                        },
                    ],
                    "dependencies": [["normalize", "aggregate_and_serialize"]],
                    "stage_count": 2,
                    "downstream_consumes_upstream_artifact": True,
                }

        bug_budget = min(len(scenario["bugs"]), {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}[spec.difficulty])
        bug_candidates = list(scenario["bugs"])
        selected_bugs = bug_candidates if spec.difficulty == 5 else bug_candidates[:bug_budget]
        if spec.difficulty < 5 and len(bug_candidates) > bug_budget:
            selected_bugs = rng.sample(bug_candidates, k=bug_budget)

        buggy_files = dict(correct_files)
        touched_files: set[str] = set()
        bug_labels: list[str] = []
        for bug in selected_bugs:
            target_path = bug["target_path"]
            buggy_files[target_path] = bug["apply"](buggy_files[target_path])
            touched_files.add(target_path)
            bug_labels.append(bug["label"])

        if composition.get("composition_mode") == "compositional":
            buggy_files["docs/current_pipeline_contract.md"] = (
                "# Current pipeline contract (authoritative)\n\n"
                f"Run `python run_pipeline.py` and write the final valid JSON artifact to "
                f"`{scenario['required_output_path']}`. Preserve the scenario's filtering, "
                "normalization, aggregation, and deterministic ordering semantics.\n"
            )
            buggy_files["logs/production_incident.log"] = (
                "2026-04-19T03:12:07Z pipeline completed but artifact validation failed\n"
                f"2026-04-19T03:12:08Z {str(scenario['debug_note']).strip()}\n"
            )

        for relative_path, content in buggy_files.items():
            write_text(visible_root / relative_path, content)

        if spec.difficulty >= 4:
            write_text(
                visible_root / "config" / "README.txt",
                "Only `config/pipeline_config.json` is authoritative. Other config snippets are legacy leftovers.\n",
            )
            write_text(visible_root / "notes" / "handoff.md", str(scenario["debug_note"]))

        disclosed_targets = sorted(touched_files)
        if spec.difficulty == 5:
            disclosed_targets = sorted(
                path
                for path in correct_files
                if path == "run_pipeline.py"
                or path.startswith("config/")
                or (path.startswith("src/") and path.endswith(".py") and not path.endswith("/__init__.py"))
            )

        output_contract = (
            {
                "top_level": "list",
                "row_schema": {
                    "team": "string",
                    "job_count": "integer",
                    "total_hours": "number",
                },
                "semantics": {
                    "job_count": "number of non-cancelled jobs for the normalized team",
                    "total_hours": "sum of hours for non-cancelled jobs, rounded to one decimal",
                    "profile_requirements": list(scenario.get("profile_contract", [])),
                },
                "ordering": "rows sorted by team ascending",
                "schema_version": "v2",
            }
            if spec.difficulty == 5 and scenario["scenario_id"] == "team_hours_pipeline"
            else None
        )
        task_descriptor = {
            "family": "pipeline",
            "scenario_id": scenario["scenario_id"],
            "entrypoint": "python run_pipeline.py",
            "required_output_path": scenario["required_output_path"],
            "target_files": disclosed_targets,
            "hints": self.visible_hints(list(scenario["hints"]), spec.difficulty),
            **composition,
        }
        if output_contract is not None:
            task_descriptor["output_contract"] = output_contract
        if composition_spec:
            task_descriptor["composition_spec"] = composition_spec
        if composition.get("composition_mode") == "compositional":
            task_descriptor["composition_evidence_paths"] = [
                "logs/production_incident.log",
                "docs/current_pipeline_contract.md",
            ]
        write_text(visible_root / "README.md", self.build_readme(scenario, task_descriptor))
        write_json(visible_root / "task.json", task_descriptor)

        reference_solution = {
            "files": {path: correct_files[path] for path in sorted(touched_files)},
            "bug_labels": bug_labels,
            "scenario_id": scenario["scenario_id"],
        }
        write_json(hidden_root / "expected_output.json", expected_output)
        for relative_path, payload in dict(scenario.get("hidden_json_assets", {})).items():
            write_json(hidden_root / relative_path, payload)
        for relative_path, content in dict(scenario.get("hidden_text_assets", {})).items():
            write_text(hidden_root / relative_path, str(content))
        if composition_spec:
            write_json(
                hidden_root / "expected_normalized_jobs.json",
                scenario["normalized_output"],
            )
        evaluator_config = {
            "entrypoint": "run_pipeline.py",
            "required_output_path": scenario["required_output_path"],
            "capability_scoring": spec.difficulty == 5,
            **dict(scenario.get("evaluator_config", {})),
        }
        if spec.difficulty < 5:
            evaluator_config.update(
                {
                    "initial_score": self.initial_output_score(
                        visible_root,
                        required_output_path=str(scenario["required_output_path"]),
                        expected_output=expected_output,
                        timeout_seconds=spec.time_limit_seconds,
                    ),
                    "score_normalization": "initial_workspace_progress_v1",
                }
            )
        if composition_spec and not scenario.get("profile_managed_composition"):
            evaluator_config["required_json_artifacts"] = [
                {
                    "path": "artifacts/normalized_jobs.json",
                    "expected_path": "expected_normalized_jobs.json",
                    "capability": "intermediate_normalization",
                }
            ]
        write_json(hidden_root / "evaluator_config.json", evaluator_config)
        write_json(hidden_root / "solution_files.json", reference_solution)

        metadata = {
            "task_descriptor": task_descriptor,
            "complexity_profile": spec.complexity_profile.to_dict() if spec.complexity_profile else {},
            "bug_labels": bug_labels,
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
                applied_bug_count=len(bug_labels),
                unmodified_reward_limit=(0.85 if spec.difficulty == 5 else 0.15),
                touched_file_count=len(touched_files),
                bug_bundle_id=dict(scenario.get("defect_bundle", {})).get("bundle_id"),
                dependency_edges=list(dict(scenario.get("defect_bundle", {})).get("dependency_edges", [])),
                capabilities=(
                    [
                        "execution",
                        "top_level_shape",
                        "row_schema",
                        "normalization",
                        "deduplication",
                        "filtering",
                        "aggregation",
                        "ordering",
                        "determinism",
                    ]
                    if spec.difficulty == 5 and scenario["scenario_id"] == "team_hours_pipeline"
                    else []
                ),
                capability_count=(
                    9 if spec.difficulty == 5 and scenario["scenario_id"] == "team_hours_pipeline" else 0
                ),
                semantic_dependency_depth=int(dict(scenario.get("structure", {})).get("dependency_depth", 0)),
                composition_spec=composition_spec,
                oracle_profile=normalize_lattice_profile(dict(scenario.get("partial_solution_lattice_profile", {}))),
                profile=d5_profile_metadata_for_family("pipeline", spec.difficulty, spec.seed).get("profile"),
                **composition,
            ),
        }
        return GeneratedPayload(
            instruction="Repair the mini-project so running the pipeline produces the required final artifact.",
            metadata=metadata,
            reference_solution=reference_solution,
            evaluator_entrypoint=str(
                scenario.get(
                    "evaluator_entrypoint",
                    "swg_generators.evaluators.pipeline:PipelineEvaluator",
                )
            ),
        )

    def initial_output_score(
        self,
        visible_root: Path,
        *,
        required_output_path: str,
        expected_output: object,
        timeout_seconds: int,
    ) -> float:
        with tempfile.TemporaryDirectory(prefix="swg-pipeline-baseline-") as temporary_dir:
            workspace = Path(temporary_dir) / "workspace"
            shutil.copytree(visible_root, workspace)
            completed = subprocess.run(
                [sys.executable, str(workspace / "run_pipeline.py")],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            output_path = workspace / required_output_path
            if completed.returncode != 0:
                return 0.0
            if not output_path.is_file():
                return 0.05
            try:
                actual = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return 0.10
            return weighted_match_score(
                output_exists=1.0,
                valid_structure=1.0,
                metrics=row_overlap_metrics(expected_output, actual),
            )

    def make_team_hours_compositional(self, correct_files: dict[str, str]) -> dict[str, str]:
        files = dict(correct_files)
        old = """    normalized = normalize_rows(rows)
    summary = build_summary(normalized, exclude_states=config["exclude_states"])
"""
        new = """    normalized = normalize_rows(rows)
    normalized_path = workspace / "artifacts" / "normalized_jobs.json"
    write_json(normalized_path, normalized)
    normalized_from_artifact = load_rows(normalized_path)
    summary = build_summary(normalized_from_artifact, exclude_states=config["exclude_states"])
"""
        if old not in files["run_pipeline.py"]:
            raise ValueError("team-hours runner drifted from the composition template")
        files["run_pipeline.py"] = files["run_pipeline.py"].replace(old, new, 1)
        return files

    def visible_hints(self, hints: list[str], difficulty: int) -> list[str]:
        return select_visible_hints(hints, difficulty)

    def scenario_pool(self, rng: random.Random, spec: EnvironmentSpec) -> list[dict[str, object]]:
        scenarios = [
            self.team_hours_pipeline_scenario(rng, spec),
            build_sales_csv_pipeline_scenario(self),
            build_artifact_stitch_pipeline_scenario(self),
            build_quality_gate_pipeline_scenario(self),
        ]
        if spec.difficulty == 5 and spec.scenario_id is None:
            return scenarios[:1]
        return scenarios

    def team_hours_pipeline_scenario(self, rng: random.Random, spec: EnvironmentSpec) -> dict[str, object]:
        if spec.difficulty == 5:
            return build_profiled_team_hours_scenario(rng, spec)
        jobs = self.build_jobs(rng, count=6 + spec.difficulty)
        return {
            "scenario_id": "team_hours_pipeline",
            "title": "Team Hours Pipeline",
            "required_output_path": "artifacts/summary.json",
            "debug_note": "The final artifact must land in `artifacts/summary.json` even if the current config disagrees.\n",
            "hints": [
                "The config file and the code need to agree on both input and output paths.",
                "Cancelled jobs should not count toward the final summary.",
                "The final artifact must be JSON and sorted by team.",
            ],
            "structure": {
                "repair_surface": "config_and_aggregation",
                "bug_scope": "cross_file",
                "failure_mode": "semantic_and_formatting",
                "smoke_test_quality": "informative",
            },
            "files": self.correct_files(jobs),
            "expected_output": self.build_expected_output(jobs),
            "normalized_output": [
                {
                    "team": str(row["team"]).lower(),
                    "state": str(row["state"]).lower(),
                    "hours": float(row["hours"]),
                }
                for row in jobs
            ],
            "partial_solution_lattice_profile": (
                {
                    "no_fix_score": 0.15,
                    "single_fix_max_score": 0.15,
                    "pair_fix_max_score": 0.50,
                    "all_but_one_max_score": 0.833333,
                    "full_solution_score": 1.0,
                    "valid": True,
                }
                if spec.difficulty == 5
                else {}
            ),
            "bugs": (self.semantic_bug_candidates() if spec.difficulty == 5 else self.legacy_bug_candidates()),
        }

    def io_utils_module(self, *, include_loader: bool = False, atomic_write: bool = False) -> str:
        loader = ""
        if include_loader:
            loader = dedent(
                """\

                def load_rows(path: Path) -> list[dict[str, object]]:
                    return json.loads(Path(path).read_text(encoding="utf-8"))
                """
            )
        writer = (
            "def write_json(path: Path, payload: object) -> None:\n"
            "    target = Path(path)\n"
            "    target.parent.mkdir(parents=True, exist_ok=True)\n"
            "    temporary = target.with_suffix(target.suffix + '.tmp')\n"
            '    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")\n'
            "    temporary.replace(target)\n"
            if atomic_write
            else (
                "def write_json(path: Path, payload: object) -> None:\n"
                "    Path(path).parent.mkdir(parents=True, exist_ok=True)\n"
                '    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")\n'
            )
        )
        return f"from __future__ import annotations\n\nimport json\nfrom pathlib import Path\n{loader}\n{writer}"

    def build_jobs(self, rng: random.Random, *, count: int) -> list[dict[str, object]]:
        teams = ["platform", "research", "ops"]
        states = ["ready", "complete", "cancelled"]
        jobs = []
        for index in range(count):
            jobs.append(
                {
                    "job_id": f"J{index + 1:03d}",
                    "team": rng.choice(teams).upper() if index % 2 == 0 else rng.choice(teams).title(),
                    "state": rng.choices(states, weights=(0.2, 0.6, 0.2), k=1)[0],
                    "hours": round(rng.uniform(1.5, 8.0), 1),
                }
            )
        return jobs

    def build_expected_output(self, jobs: list[dict[str, object]]) -> list[dict[str, object]]:
        summary: dict[str, dict[str, object]] = {}
        for row in jobs:
            team = str(row["team"]).lower()
            state = str(row["state"]).lower()
            if state == "cancelled":
                continue
            if team not in summary:
                summary[team] = {"team": team, "job_count": 0, "total_hours": 0.0}
            summary[team]["job_count"] = int(summary[team]["job_count"]) + 1
            summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + float(row["hours"]), 1)
        return sorted(summary.values(), key=lambda item: str(item["team"]))

    def correct_files(self, jobs: list[dict[str, object]]) -> dict[str, str]:
        config = {
            "schema_version": "v2",
            "input_path": "data/jobs.json",
            "output_path": "artifacts/summary.json",
            "exclude_states": ["cancelled"],
        }
        io_utils = self.io_utils_module(include_loader=True, atomic_write=True)
        steps = """from __future__ import annotations


def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "team": str(row["team"]).lower(),
                "state": str(row["state"]).lower(),
                "hours": float(row["hours"]),
            }
        )
    return normalized


def build_summary(rows: list[dict[str, object]], *, exclude_states: list[str]) -> list[dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["state"] in exclude_states:
            continue
        team = str(row["team"])
        if team not in summary:
            summary[team] = {"team": team, "job_count": 0, "total_hours": 0.0}
        summary[team]["job_count"] = int(summary[team]["job_count"]) + 1
        summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + float(row["hours"]), 1)
    return sorted(summary.values(), key=lambda item: str(item["team"]))
"""
        runner = """from __future__ import annotations

import json
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace / "src"))

from pipeline_app.io_utils import load_rows, write_json
from pipeline_app.steps import build_summary, normalize_rows


def main() -> None:
    config = json.loads((workspace / "config" / "pipeline_config.json").read_text(encoding="utf-8"))
    rows = load_rows(workspace / config["input_path"])
    normalized = normalize_rows(rows)
    summary = build_summary(normalized, exclude_states=config["exclude_states"])
    if config.get("schema_version") != "v2":
        summary = [
            {"team": row["team"], "jobs": row["job_count"], "hours": row["total_hours"]}
            for row in summary
        ]
    write_json(workspace / config["output_path"], summary)


if __name__ == "__main__":
    main()
"""
        return {
            "src/pipeline_app/__init__.py": "",
            "src/pipeline_app/io_utils.py": io_utils,
            "src/pipeline_app/steps.py": steps,
            "run_pipeline.py": runner,
            "config/pipeline_config.json": json.dumps(config, indent=2, sort_keys=True) + "\n",
            "data/jobs.json": json.dumps(jobs, indent=2, sort_keys=True) + "\n",
        }

    def legacy_bug_candidates(self) -> list[dict[str, object]]:
        return [
            {
                "label": "wrong_input_path",
                "target_path": "config/pipeline_config.json",
                "apply": self.replace_once(
                    "data/jobs.json",
                    "data/job.json",
                    label="wrong_input_path",
                    target_path="config/pipeline_config.json",
                ),
            },
            {
                "label": "wrong_output_path",
                "target_path": "config/pipeline_config.json",
                "apply": self.replace_once(
                    "artifacts/summary.json",
                    "artifacts/result.json",
                    label="wrong_output_path",
                    target_path="config/pipeline_config.json",
                ),
            },
            {
                "label": "missing_normalization_step",
                "target_path": "run_pipeline.py",
                "apply": self.replace_once(
                    "normalized = normalize_rows(rows)",
                    "normalized = rows",
                    label="missing_normalization_step",
                    target_path="run_pipeline.py",
                ),
            },
            {
                "label": "aggregation_bug",
                "target_path": "src/pipeline_app/steps.py",
                "apply": self.replace_once(
                    'summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + float(row["hours"]), 1)',
                    'summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + 1, 1)',
                    label="aggregation_bug",
                    target_path="src/pipeline_app/steps.py",
                ),
            },
            {
                "label": "output_format_bug",
                "target_path": "src/pipeline_app/io_utils.py",
                "apply": self.replace_once(
                    'temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")',
                    'temporary.write_text(str(payload), encoding="utf-8")',
                    label="output_format_bug",
                    target_path="src/pipeline_app/io_utils.py",
                ),
            },
        ]

    def semantic_bug_candidates(self) -> list[dict[str, object]]:
        return [
            {
                "label": "stale_schema_version",
                "target_path": "config/pipeline_config.json",
                "apply": self.replace_once(
                    '"schema_version": "v2"',
                    '"schema_version": "v1"',
                    label="stale_schema_version",
                    target_path="config/pipeline_config.json",
                ),
            },
            {
                "label": "missing_normalization_step",
                "target_path": "run_pipeline.py",
                "apply": self.replace_once(
                    "normalized = normalize_rows(rows)",
                    "normalized = rows",
                    label="missing_normalization_step",
                    target_path="run_pipeline.py",
                ),
            },
            {
                "label": "filter_before_canonicalization",
                "target_path": "src/pipeline_app/steps.py",
                "apply": self.replace_once(
                    'if row["state"] in exclude_states:',
                    'if str(row["state"]) == "CANCELLED":',
                    label="filter_before_canonicalization",
                    target_path="src/pipeline_app/steps.py",
                ),
            },
            {
                "label": "aggregation_uses_record_count",
                "target_path": "src/pipeline_app/steps.py",
                "apply": self.replace_once(
                    'summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + float(row["hours"]), 1)',
                    'summary[team]["total_hours"] = round(float(summary[team]["total_hours"]) + 1, 1)',
                    label="aggregation_uses_record_count",
                    target_path="src/pipeline_app/steps.py",
                ),
            },
            {
                "label": "serializer_wraps_legacy_contract",
                "target_path": "src/pipeline_app/io_utils.py",
                "apply": self.replace_once(
                    "json.dumps(payload, indent=2, sort_keys=True)",
                    'json.dumps({"rows": payload}, indent=2, sort_keys=True)',
                    label="serializer_wraps_legacy_contract",
                    target_path="src/pipeline_app/io_utils.py",
                ),
            },
        ]

    def build_readme(self, scenario: dict[str, object], task_descriptor: dict[str, object]) -> str:
        hints = "\n".join(f"- {hint}" for hint in task_descriptor["hints"])
        targets = "\n".join(f"- `{item}`" for item in task_descriptor["target_files"])
        if task_descriptor.get("composition_mode") == "compositional":
            output_contract = (
                "- Diagnose `logs/production_incident.log`, then resolve the current "
                "contract in `docs/current_pipeline_contract.md`.\n"
            )
        else:
            output_contract = (
                f"- The final artifact must be written to `{task_descriptor['required_output_path']}`.\n"
                "- The artifact must be valid JSON.\n"
                "- Rows must be sorted by `team`.\n"
                "- Cancelled jobs must be excluded.\n"
            )
        visible_schema_contract = ""
        if task_descriptor.get("output_contract"):
            visible_schema_contract = (
                "## Exact v2 output contract\n"
                "```json\n"
                f"{json.dumps(task_descriptor['output_contract'], indent=2, sort_keys=True)}\n"
                "```\n\n"
                "The existing `schema_version` value may be stale. The final artifact "
                "must follow the v2 output contract defined above.\n\n"
            )
        return (
            f"# {scenario['title']}\n\n"
            "This mini-project is almost complete, but one or more files are inconsistent or incomplete.\n"
            "Repair the pipeline so the final artifact is produced correctly.\n\n"
            "## Smoke test\n"
            f"- `{task_descriptor['entrypoint']}`\n\n"
            "## Output contract\n"
            f"{output_contract}\n"
            f"{visible_schema_contract}"
            "## Likely target files\n"
            f"{targets}\n\n"
            "## Hints\n"
            f"{hints}\n"
        )
