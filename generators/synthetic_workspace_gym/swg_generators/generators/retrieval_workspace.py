from __future__ import annotations

import json
import random
from pathlib import Path

from swg_generators.evaluators.metrics import flatten_json, row_overlap_metrics, weighted_match_score
from swg_generators.generators.base import BaseGenerator, GeneratedPayload
from swg_generators.generators.common import (
    build_d5_composition_profile,
    build_difficulty_realization,
    select_visible_hints,
)
from swg_generators.generators.d5_profiles import (
    d5_profile_metadata_for_family,
    select_weighted_d5_profile,
)
from swg_generators.generators.d5_quality import normalize_lattice_profile
from swg_generators.generators.grounding import validate_exact_json_grounding
from swg_generators.generators.retrieval_profile_scenarios import (
    RETRIEVAL_PROFILE_SCENARIOS,
    build_profiled_retrieval_scenario,
)
from swg_generators.generators.retrieval_workspace_scenarios import (
    build_client_adapter_sync_scenario,
    build_incident_report_bundle_scenario,
    build_migration_plan_bundle_scenario,
    build_service_config_reconciliation_scenario,
)
from swg_generators.schemas import EnvironmentFamily, EnvironmentSpec
from swg_generators.utils.io import write_json, write_text


class RetrievalWorkspaceGenerator(BaseGenerator):
    family = EnvironmentFamily.RETRIEVAL_WORKSPACE

    def build_environment(
        self,
        spec: EnvironmentSpec,
        *,
        root: Path,
        visible_root: Path,
        hidden_root: Path,
    ) -> GeneratedPayload:
        scenario = dict(self.select_scenario(spec, self.scenario_pool(spec)))
        composition = build_d5_composition_profile(
            self.family,
            spec.difficulty,
            spec.seed,
            mode=spec.generation_params.get("composition_mode"),
        )
        if composition.get("composition_mode") == "compositional" and scenario["scenario_id"] == "client_adapter_sync":
            scenario = self.make_client_adapter_compositional(scenario)
        task_descriptor = {
            "family": "retrieval_workspace",
            "scenario_id": scenario["scenario_id"],
            "task_type": scenario["task_type"],
            "target_path": scenario["target_path"],
            "output_style": scenario["output_style"],
            "document_roots": scenario["document_roots"],
            "hints": self.visible_hints(list(scenario["hints"]), spec.difficulty),
            **composition,
        }
        if composition.get("composition_mode") == "compositional":
            task_descriptor["composition_evidence_paths"] = sorted(
                path
                for path in scenario["files"]
                if path.startswith(
                    (
                        "changelog/",
                        "docs/",
                        "logs/",
                        "notes/",
                        "policies/",
                        "release/",
                        "specs/",
                    )
                )
            )
            task_descriptor["composition_spec"] = dict(scenario.get("composition_spec", {}))
        if scenario.get("schema_version"):
            task_descriptor["schema_version"] = scenario["schema_version"]
        if scenario.get("schema_spec_path"):
            task_descriptor["schema_spec_path"] = scenario["schema_spec_path"]
        if scenario.get("entrypoint"):
            task_descriptor["entrypoint"] = scenario["entrypoint"]

        files = dict(scenario["files"])
        evaluator_config = dict(scenario["evaluator_config"])
        if evaluator_config.get("mode") == "exact_json":
            expected_output = scenario["expected_output"]
            target_path = str(scenario["target_path"])
            grounding_evidence = validate_exact_json_grounding(
                expected_output,
                files,
                target_path=target_path,
                exempt_paths=set(evaluator_config.get("grounding_exempt_paths", [])),
            )
            initial_output = json.loads(str(files[target_path]))
            initial_metrics = row_overlap_metrics(flatten_json(expected_output), flatten_json(initial_output))
            evaluator_config.update(
                {
                    "grounding_evidence": grounding_evidence,
                    "initial_score": weighted_match_score(
                        output_exists=1.0,
                        valid_structure=1.0,
                        metrics=initial_metrics,
                    ),
                    "score_normalization": "initial_workspace_progress_v1",
                }
            )

        for relative_path, content in files.items():
            write_text(visible_root / relative_path, str(content))
        write_text(visible_root / "README.md", self.build_readme(scenario, task_descriptor))
        write_json(visible_root / "task.json", task_descriptor)

        if scenario.get("expected_output") is not None:
            write_json(hidden_root / "expected_output.json", scenario["expected_output"])
        for relative_path, payload in dict(scenario.get("hidden_json_assets", {})).items():
            write_json(hidden_root / relative_path, payload)
        for relative_path, content in dict(scenario.get("hidden_text_assets", {})).items():
            write_text(hidden_root / relative_path, str(content))

        reference_solution = {
            "files": dict(scenario["reference_solution_files"]),
            "seed": spec.seed,
            "scenario_id": scenario["scenario_id"],
            "task_descriptor": task_descriptor,
        }
        write_json(hidden_root / "reference_solution.json", reference_solution)
        write_json(hidden_root / "evaluator_config.json", evaluator_config)

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
                "target_path": scenario["target_path"],
                "document_roots": scenario["document_roots"],
            },
            "difficulty_realization": build_difficulty_realization(
                spec.difficulty,
                hint_count=len(task_descriptor["hints"]),
                candidate_file_count=int(scenario["structure"]["document_count"]),
                document_count=int(scenario["structure"]["document_count"]),
                distractor_count=int(scenario["structure"]["distractor_count"]),
                retrieval_hops=int(scenario["structure"]["retrieval_hops"]),
                staleness_pattern=str(scenario["structure"]["staleness_pattern"]),
                touched_file_count=len(reference_solution["files"]),
                capability_count=int(
                    dict(scenario.get("structure", {})).get("capability_count", 10 if spec.difficulty == 5 else 0)
                ),
                semantic_dependency_depth=int(
                    dict(scenario.get("structure", {})).get(
                        "semantic_dependency_depth",
                        d5_profile_metadata_for_family("retrieval_workspace", spec.difficulty, spec.seed).get(
                            "semantic_dependency_depth", 0
                        ),
                    )
                ),
                composition_spec=dict(scenario.get("composition_spec", {})),
                oracle_profile=normalize_lattice_profile(dict(scenario.get("partial_solution_lattice_profile", {}))),
                bug_bundle_id=dict(scenario.get("defect_bundle", {})).get("bundle_id"),
                dependency_edges=list(dict(scenario.get("defect_bundle", {})).get("dependency_edges", [])),
                capabilities=list(dict(dict(scenario.get("defect_bundle", {})).get("capability_groups", {}))),
                applied_bug_count=len(scenario.get("bugs", [])),
                unmodified_reward_limit=float(scenario.get("unmodified_reward_limit", 0.15)),
                composition_evidence_count=(len(task_descriptor.get("composition_evidence_paths", []))),
                profile=dict(scenario.get("structure", {})).get("d5_profile"),
                **composition,
            ),
        }
        return GeneratedPayload(
            instruction=self.build_instruction(task_descriptor),
            metadata=metadata,
            reference_solution=reference_solution,
            evaluator_entrypoint="swg_generators.evaluators.retrieval_workspace:RetrievalWorkspaceEvaluator",
        )

    def scenario_pool(self, spec: EnvironmentSpec) -> list[dict[str, object]]:
        if spec.difficulty == 5:
            profiled_ids = set(RETRIEVAL_PROFILE_SCENARIOS.values())
            if spec.scenario_id is None or spec.scenario_id in profiled_ids:
                selected_id = str(spec.scenario_id) if spec.scenario_id else None
                return [
                    build_profiled_retrieval_scenario(
                        random.Random(f"{spec.seed}:profiled_retrieval"),
                        spec,
                        scenario_id=selected_id,
                    )
                ]
        return [
            build_service_config_reconciliation_scenario(
                random.Random(f"{spec.seed}:service_config_reconciliation"),
                spec,
            ),
            build_migration_plan_bundle_scenario(
                random.Random(f"{spec.seed}:migration_plan_bundle"),
                spec,
            ),
            build_incident_report_bundle_scenario(
                random.Random(f"{spec.seed}:incident_report_bundle"),
                spec,
            ),
            build_client_adapter_sync_scenario(
                random.Random(f"{spec.seed}:client_adapter_sync"),
                spec,
            ),
        ]

    def resolve_scenario_id(
        self,
        *,
        difficulty: int,
        seed: int,
        requested_scenario: str | None = None,
    ) -> str:
        if requested_scenario is not None:
            return str(requested_scenario)
        if difficulty == 5:
            profile = select_weighted_d5_profile(difficulty, seed)
            if profile is None:
                raise ValueError("D5 retrieval profile selection failed")
            return RETRIEVAL_PROFILE_SCENARIOS[profile.profile_id]
        return super().resolve_scenario_id(difficulty=difficulty, seed=seed)

    def make_client_adapter_compositional(self, scenario: dict[str, object]) -> dict[str, object]:
        """Require an evidence-derived config that the repaired adapter consumes."""
        scenario = {**scenario}
        files = dict(scenario["files"])
        reference_files = dict(scenario["reference_solution_files"])
        hidden_json_assets = dict(scenario.get("hidden_json_assets", {}))
        contract = {
            "collection_field": "records",
            "quantity_field": "quantity",
            "cursor_field": "next_cursor",
            "warehouse_default": "unknown",
        }
        correct_adapter = """from __future__ import annotations

import json
from pathlib import Path


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "config" / "adapter_contract.json"


def build_summary(response: dict[str, object]) -> dict[str, object]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for candidate in response.get(contract["collection_field"], []):
        if not isinstance(candidate, dict):
            continue
        try:
            quantity = int(str(candidate[contract["quantity_field"]]).strip())
        except (KeyError, TypeError, ValueError):
            continue
        records.append({**candidate, "quantity": quantity})
    return {
        "request_id": str(response["request_id"]),
        "next_cursor": response.get(contract["cursor_field"]),
        "record_count": len(records),
        "total_quantity": sum(int(record["quantity"]) for record in records),
        "warehouses": sorted(
            {
                str(record.get("warehouse", contract["warehouse_default"]))
                for record in records
            }
        ),
    }
"""
        reference_files["src/client_adapter.py"] = correct_adapter
        reference_files["config/adapter_contract.json"] = json.dumps(contract, indent=2, sort_keys=True) + "\n"
        hidden_json_assets["expected_adapter_contract.json"] = contract
        output_contract = [
            *list(scenario["output_contract"]),
            "Materialize `config/adapter_contract.json` from the active evidence chain.",
            "The repaired adapter must read that artifact at runtime; do not duplicate the mapping in code.",
        ]
        evaluator_config = {
            **dict(scenario["evaluator_config"]),
            "required_json_artifacts": [
                {
                    "path": "config/adapter_contract.json",
                    "expected_path": "expected_adapter_contract.json",
                    "capability": "evidence_materialization",
                }
            ],
            "required_json_artifact_weights": {
                "evidence_materialization": 0.10,
            },
        }
        scenario.update(
            {
                "files": files,
                "reference_solution_files": reference_files,
                "hidden_json_assets": hidden_json_assets,
                "output_contract": output_contract,
                "evaluator_config": evaluator_config,
                "composition_spec": {
                    "stages": [
                        {
                            "stage_id": "resolve_contract",
                            "required_inputs": [
                                "docs/api_reference.md",
                                "notes/api_changelog.md",
                                "changelog/evidence_index.md",
                            ],
                            "produced_artifacts": ["config/adapter_contract.json"],
                            "capability": "evidence_materialization",
                        },
                        {
                            "stage_id": "repair_adapter",
                            "required_inputs": ["config/adapter_contract.json"],
                            "produced_artifacts": ["src/client_adapter.py"],
                            "capability": "integration",
                        },
                    ],
                    "dependencies": [["resolve_contract", "repair_adapter"]],
                    "stage_count": 2,
                    "downstream_consumes_upstream_artifact": True,
                },
            }
        )
        return scenario

    def visible_hints(self, hints: list[str], difficulty: int) -> list[str]:
        return select_visible_hints(hints, difficulty)

    def build_instruction(self, task_descriptor: dict[str, object]) -> str:
        target_path = str(task_descriptor["target_path"])
        return (
            "Inspect the local document set, retrieve the relevant evidence, and update the target artifact. "
            f"Ground the final change in the visible workspace documents and write the result to {target_path}."
        )

    def build_readme(self, scenario: dict[str, object], task_descriptor: dict[str, object]) -> str:
        document_roots = "\n".join(f"- `{root}/`" for root in task_descriptor["document_roots"])
        output_contract = "\n".join(f"- {line}" for line in scenario["output_contract"])
        composition_step = ""
        if task_descriptor.get("composition_mode") == "compositional":
            composition_step = (
                "- Resolve the active evidence chain, materialize the required intermediate "
                "artifact, and ensure the downstream implementation consumes it.\n"
            )
        hints = "\n".join(f"- {hint}" for hint in task_descriptor["hints"])
        smoke_test = ""
        if task_descriptor.get("entrypoint"):
            smoke_test = f"## Optional smoke test\n- `{task_descriptor['entrypoint']}`\n\n"
        return (
            f"# {scenario['title']}\n\n"
            f"{scenario['description']}\n\n"
            "## Objective\n"
            f"- Update or create `{task_descriptor['target_path']}` using evidence from the local document set.\n"
            "- Ignore irrelevant or stale documents when they conflict with newer authoritative sources.\n"
            f"{composition_step}\n"
            "## Document roots\n"
            f"{document_roots}\n\n"
            f"{smoke_test}"
            "## Output contract\n"
            f"{output_contract}\n\n"
            "## Hints\n"
            f"{hints}\n"
        )
