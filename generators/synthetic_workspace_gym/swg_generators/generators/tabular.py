from __future__ import annotations

import json
import random
from pathlib import Path

from swg_generators.generators.base import BaseGenerator, GeneratedPayload
from swg_generators.generators.common import (
    build_d5_composition_profile,
    build_difficulty_realization,
    select_visible_hints,
)
from swg_generators.generators.d5_profiles import d5_profile_metadata
from swg_generators.generators.d5_quality import normalize_lattice_profile
from swg_generators.generators.tabular_program_synthesis import (
    build_account_event_program_scenario,
)
from swg_generators.generators.tabular_quality import (
    build_account_event_reconciliation_scenario,
)
from swg_generators.generators.tabular_scenarios import (
    build_channel_status_pivot_scenario,
    build_monthly_segment_report_scenario,
    build_supplier_restock_summary_scenario,
    build_weekly_refund_rollup_scenario,
)
from swg_generators.schemas import EnvironmentFamily, EnvironmentSpec
from swg_generators.utils.io import write_json, write_text


class TabularTransformationGenerator(BaseGenerator):
    family = EnvironmentFamily.TABULAR

    def build_environment(
        self,
        spec: EnvironmentSpec,
        *,
        root: Path,
        visible_root: Path,
        hidden_root: Path,
    ) -> GeneratedPayload:
        composition = build_d5_composition_profile(
            self.family,
            spec.difficulty,
            spec.seed,
            mode=spec.generation_params.get("composition_mode"),
        )
        scenario = self.select_scenario(spec, self.scenario_pool(spec, composition))
        task_descriptor = {
            "family": "tabular",
            "scenario_id": scenario["scenario_id"],
            **dict(scenario["task_descriptor"]),
            "hints": self.visible_hints(list(scenario["hints"]), spec.difficulty),
            **composition,
        }
        if scenario.get("composition_spec"):
            task_descriptor["composition_spec"] = dict(scenario["composition_spec"])
        files = dict(scenario["files"])
        if composition.get("composition_mode") == "compositional":
            task_descriptor["composition_evidence_paths"] = list(task_descriptor["input_files"])

        for relative_path, content in files.items():
            write_text(visible_root / relative_path, str(content))
        write_text(visible_root / "README.md", self.build_readme(scenario, task_descriptor))
        write_json(visible_root / "task.json", task_descriptor)

        expected_output = scenario["expected_output"]
        output_path = str(task_descriptor["output_path"])
        write_json(hidden_root / "expected_output.json", expected_output)
        for relative_path, payload in dict(scenario.get("hidden_json_assets", {})).items():
            write_json(hidden_root / relative_path, payload)
        for relative_path, content in dict(scenario.get("hidden_text_assets", {})).items():
            write_text(hidden_root / relative_path, str(content))
        evaluator_config = {
            "output_path": output_path,
            "comparison_mode": "exact_json",
            **dict(scenario.get("evaluator_config", {})),
        }
        write_json(hidden_root / "evaluator_config.json", evaluator_config)
        evaluator_entrypoint = str(
            scenario.get(
                "evaluator_entrypoint",
                "swg_generators.evaluators.tabular:TabularEvaluator",
            )
        )
        evaluator_config.update(
            {
                "initial_score": 0.0,
                "score_normalization": "initial_workspace_progress_v1",
            }
        )
        write_json(hidden_root / "evaluator_config.json", evaluator_config)
        reference_solution = {
            "files": dict(
                scenario.get(
                    "reference_solution_files",
                    {output_path: json.dumps(expected_output, indent=2, sort_keys=True) + "\n"},
                )
            ),
            "seed": spec.seed,
            "scenario_id": scenario["scenario_id"],
            "task_descriptor": task_descriptor,
        }
        write_json(hidden_root / "reference_solution.json", reference_solution)

        metadata = {
            "task_descriptor": task_descriptor,
            "complexity_profile": spec.complexity_profile.to_dict() if spec.complexity_profile else {},
            "scenario_id": scenario["scenario_id"],
            "scenario_profile": scenario["structure"],
            "scenario_selection": {
                "requested_scenario_id": spec.scenario_id,
                "selection_mode": "explicit" if spec.scenario_id else "seed_modulo",
            },
            "visible_artifact_layout": {
                "workspace_root": "visible",
                "output_path": output_path,
            },
            "difficulty_realization": build_difficulty_realization(
                spec.difficulty,
                hint_count=len(task_descriptor["hints"]),
                candidate_file_count=len(task_descriptor["input_files"]),
                operation_count=len(task_descriptor["operations"]),
                input_file_count=len(task_descriptor["input_files"]),
                touched_file_count=len(reference_solution["files"]),
                capability_count=int(dict(scenario.get("structure", {})).get("hidden_capability_count", 0)),
                semantic_dependency_depth=int(dict(scenario.get("structure", {})).get("dependency_depth", 0)),
                distractor_count=int(dict(scenario.get("structure", {})).get("distractor_count", 0)),
                composition_spec=dict(scenario.get("composition_spec", {})),
                oracle_profile=normalize_lattice_profile(dict(scenario.get("partial_solution_lattice_profile", {}))),
                bug_bundle_id=dict(scenario.get("defect_bundle", {})).get("bundle_id"),
                dependency_edges=list(dict(scenario.get("defect_bundle", {})).get("dependency_edges", [])),
                capabilities=list(dict(dict(scenario.get("defect_bundle", {})).get("capability_groups", {}))),
                applied_bug_count=len(scenario.get("bugs", [])),
                unmodified_reward_limit=float(scenario.get("unmodified_reward_limit", 0.15)),
                profile=d5_profile_metadata(spec.difficulty, spec.seed).get("profile"),
                **composition,
            ),
        }
        return GeneratedPayload(
            instruction=self.build_instruction(task_descriptor),
            metadata=metadata,
            reference_solution=reference_solution,
            evaluator_entrypoint=evaluator_entrypoint,
        )

    def scenario_pool(
        self, spec: EnvironmentSpec, composition: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        scenarios = [
            build_monthly_segment_report_scenario(random.Random(f"{spec.seed}:monthly_segment_report"), spec),
            build_channel_status_pivot_scenario(random.Random(f"{spec.seed}:channel_status_pivot"), spec),
            build_weekly_refund_rollup_scenario(random.Random(f"{spec.seed}:weekly_refund_rollup"), spec),
            build_supplier_restock_summary_scenario(random.Random(f"{spec.seed}:supplier_restock_summary"), spec),
        ]
        if spec.difficulty == 5:
            compositional = dict(composition or {}).get("composition_mode") == "compositional"
            if spec.scenario_id == "account_event_reconciliation" or (spec.scenario_id is None and compositional):
                return [
                    build_account_event_reconciliation_scenario(
                        random.Random(f"{spec.seed}:account_event_reconciliation"),
                        spec,
                        compositional=compositional,
                    )
                ]
            if spec.scenario_id == "account_event_program_synthesis" or (
                spec.scenario_id is None and not compositional
            ):
                return [
                    build_account_event_program_scenario(
                        random.Random(f"{spec.seed}:account_event_program_synthesis"),
                        spec,
                    )
                ]
        return scenarios

    def visible_hints(self, hints: list[str], difficulty: int) -> list[str]:
        return select_visible_hints(hints, difficulty)

    def build_instruction(self, task_descriptor: dict[str, object]) -> str:
        return (
            "Complete the tabular transformation described in README.md. "
            f"Write the final artifact to {task_descriptor['output_path']}."
        )

    def build_readme(self, scenario: dict[str, object], task_descriptor: dict[str, object]) -> str:
        inputs = "\n".join(f"- `{item}`" for item in task_descriptor["input_files"])
        operations = "\n".join(
            f"{index}. `{operation}`" for index, operation in enumerate(task_descriptor["operations"], start=1)
        )
        operations_section = "## Required operations\n" + operations + "\n\n" if operations else ""
        if task_descriptor.get("composition_mode") == "compositional":
            contract_lines = (
                "- Retrieve the authoritative output rules from "
                "`evidence/current_output_contract.md` before transforming the inputs."
            )
        else:
            contract_lines = "\n".join(f"- {line}" for line in scenario["output_contract"])
        hints = "\n".join(f"- {hint}" for hint in task_descriptor["hints"])
        return (
            f"# {scenario['title']}\n\n"
            f"{scenario['description']}\n\n"
            "## Inputs\n"
            f"{inputs}\n\n"
            f"{operations_section}"
            "## Output contract\n"
            f"{contract_lines}\n\n"
            "## Hints\n"
            f"{hints}\n"
        )
