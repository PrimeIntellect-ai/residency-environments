from __future__ import annotations

import json
import random
from pathlib import Path

from swg_generators.generators.base import BaseGenerator, GeneratedPayload
from swg_generators.generators.common import (
    build_difficulty_realization,
    select_visible_hints,
)
from swg_generators.generators.composite_workspace_templates import (
    AGGREGATE_SOURCE,
    CONTRACT_SOURCE,
    HIDDEN_RUNNER,
    NORMALIZE_SOURCE,
    PUBLIC_CHECK,
    RUNNER_SOURCE,
)
from swg_generators.schemas import EnvironmentFamily, EnvironmentSpec
from swg_generators.utils.io import write_json, write_text

SCENARIO_ID = "retrieval_guided_pipeline_repair"
CAPABILITIES = [
    "execution",
    "contract_materialization",
    "alias_resolution",
    "active_filtering",
    "fractional_aggregation",
    "duplicate_resolution",
    "malformed_record_handling",
    "output_schema",
    "deterministic_ordering",
    "intermediate_artifact_consumption",
    "alternate_contract_generalization",
]


class CompositeWorkspaceGenerator(BaseGenerator):
    family = EnvironmentFamily.COMPOSITE_WORKSPACE

    def scenario_pool(self, spec):
        return [{"scenario_id": SCENARIO_ID}]

    def build_environment(
        self,
        spec: EnvironmentSpec,
        *,
        root: Path,
        visible_root: Path,
        hidden_root: Path,
    ) -> GeneratedPayload:
        if spec.scenario_id not in (None, SCENARIO_ID):
            raise ValueError(f"Unsupported composite scenario: {spec.scenario_id}")
        split = str(spec.generation_params.get("split", "development"))
        rng = random.Random(f"{split}:{spec.seed}:{SCENARIO_ID}")
        fixture_id = f"{split}-{spec.seed}-{rng.randrange(10_000_000):07d}"
        correct = self.correct_files(fixture_id)
        selected = self.defects()[: {1: 1, 2: 2, 3: 3, 4: 5, 5: 8}[spec.difficulty]]
        broken, touched = dict(correct), set()
        for defect in selected:
            path, before, after = defect["path"], defect["before"], defect["after"]
            if before not in broken[path]:
                raise ValueError(f"Composite defect drifted: {defect['id']}")
            broken[path] = broken[path].replace(before, after, 1)
            touched.add(path)
        for path, content in broken.items():
            write_text(visible_root / path, content)
        self.write_evidence(visible_root, split, fixture_id)
        write_text(visible_root / "README.md", self.readme())
        hints = select_visible_hints(
            [
                "Follow the active release pointer.",
                "Preserve the normalized artifact boundary.",
                "Run public_check.py.",
            ],
            spec.difficulty,
        )
        descriptor = {
            "family": self.family.value,
            "scenario_id": SCENARIO_ID,
            "archetype": SCENARIO_ID,
            "source_families": ["retrieval_workspace", "pipeline"],
            "composition_mode": "compositional",
            "composition_depth": 2,
            "document_roots": ["release", "policies", "docs"],
            "entrypoint": "run_pipeline.py",
            "target_files": sorted(correct),
            "hints": hints,
        }
        write_json(visible_root / "task.json", descriptor)
        write_text(visible_root / "public_check.py", PUBLIC_CHECK)
        write_text(hidden_root / "run_hidden_tests.py", HIDDEN_RUNNER)
        write_json(
            hidden_root / "evaluator_config.json",
            {
                "mode": "hidden_tests",
                "runner": "run_hidden_tests.py",
                "target_path": "run_pipeline.py",
                "initial_score": {
                    1: 0.727273,
                    2: 0.636364,
                    3: 0.545455,
                    4: 0.454545,
                    5: 0.000751,
                }[spec.difficulty],
                "score_normalization": "initial_workspace_progress_v1",
            },
        )
        reference = {path: correct[path] for path in sorted(touched)}
        realization = build_difficulty_realization(
            spec.difficulty,
            hint_count=len(hints),
            candidate_file_count=len(broken) + 7,
            distractor_count=3,
            touched_file_count=len(touched),
            semantic_dependency_depth=3,
            capability_count=len(CAPABILITIES),
            capabilities=CAPABILITIES,
            applied_bug_count=len(selected),
            applied_bug_ids=[item["id"] for item in selected],
            bug_bundle_id=f"composite-d{spec.difficulty}-v1",
            dependency_edges=[
                ["release", "contract"],
                ["contract", "normalized"],
                ["normalized", "summary"],
            ],
            unmodified_reward_limit=0.35,
            profile=SCENARIO_ID,
            oracle_profile={"reference_solution_reward": 1.0, "unmodified_reward": 0.0},
            composition_mode="compositional",
            source_families=["retrieval_workspace", "pipeline"],
            composition_depth=2,
            composition_spec={
                "stage_count": 3,
                "downstream_consumes_upstream_artifact": True,
            },
        )
        metadata = {
            "scenario_id": SCENARIO_ID,
            "archetype": SCENARIO_ID,
            "contributing_families": ["retrieval_workspace", "pipeline"],
            "source_families": ["retrieval_workspace", "pipeline"],
            "split_fixture_id": fixture_id,
            "evidence_template_id": f"evidence-{split}-v1",
            "repair_template_id": f"repair-{split}-v1",
            "task_descriptor": descriptor,
            "complexity_profile": spec.complexity_profile.to_dict() if spec.complexity_profile else {},
            "composition_spec": {
                "stages": [
                    {
                        "stage_id": "resolve_authority",
                        "produced_artifacts": ["config/resolved_contract.json"],
                    },
                    {
                        "stage_id": "normalize",
                        "required_inputs": ["config/resolved_contract.json"],
                        "produced_artifacts": ["artifacts/normalized_jobs.json"],
                    },
                    {
                        "stage_id": "aggregate",
                        "required_inputs": ["artifacts/normalized_jobs.json"],
                        "produced_artifacts": ["artifacts/summary.json"],
                    },
                ],
                "dependencies": [
                    ["resolve_authority", "normalize"],
                    ["normalize", "aggregate"],
                ],
                "stage_count": 3,
                "downstream_consumes_upstream_artifact": True,
            },
            "difficulty_realization": realization,
        }
        return GeneratedPayload(
            instruction="Repair the retrieval-guided pipeline. Follow visible evidence precedence, materialize the resolved contract, preserve the normalized artifact boundary, and run `python public_check.py`. Do not hard-code the sample.",
            metadata=metadata,
            reference_solution={"files": reference, "scenario_id": SCENARIO_ID},
            evaluator_entrypoint="swg_generators.evaluators.composite_workspace:CompositeWorkspaceEvaluator",
        )

    def correct_files(self, fixture_id):
        records = [
            {
                "event_id": f"{fixture_id}-1",
                "updated_at": "2026-06-01T10:00:00Z",
                "team_name": " platform ",
                "duration_hours": "1.25",
                "enabled": "YES",
            },
            {
                "event_id": f"{fixture_id}-2",
                "updated_at": "2026-06-02T10:00:00Z",
                "team_name": "CORE",
                "duration_hours": 2.5,
                "enabled": 1,
            },
            {
                "event_id": f"{fixture_id}-2",
                "updated_at": "2026-06-03T10:00:00Z",
                "team_name": "core",
                "duration_hours": 3.75,
                "enabled": "true",
            },
            {
                "event_id": f"{fixture_id}-3",
                "updated_at": "2026-06-03T11:00:00Z",
                "team_name": "Research",
                "duration_hours": 9,
                "enabled": "no",
            },
            {
                "event_id": f"{fixture_id}-bad",
                "updated_at": "not-a-date",
                "team_name": "Core",
                "duration_hours": "oops",
                "enabled": "yes",
            },
            "malformed",
        ]
        return {
            "src/__init__.py": "",
            "src/contract.py": CONTRACT_SOURCE,
            "src/normalize.py": NORMALIZE_SOURCE,
            "src/aggregate.py": AGGREGATE_SOURCE,
            "run_pipeline.py": RUNNER_SOURCE,
            "data/jobs.json": json.dumps({"jobs": records}, indent=2, sort_keys=True) + "\n",
        }

    def write_evidence(self, root, split, fixture_id):
        contract = {
            "collection_field": "jobs",
            "identity_field": "event_id",
            "updated_at_field": "updated_at",
            "team_field": "team_name",
            "hours_field": "duration_hours",
            "active_field": "enabled",
            "active_values": ["yes", "true", "1"],
            "team_aliases": {"platform": "Core", "core": "Core"},
        }
        write_json(
            root / "release/active_bundle.json",
            {
                "release": fixture_id,
                "policy_path": f"policies/{split}/pipeline_contract.json",
            },
        )
        write_json(
            root / f"policies/{split}/pipeline_contract.json",
            {"status": "active", "pipeline_contract": contract},
        )
        write_json(
            root / "docs/legacy_contract.json",
            {
                "status": "retired",
                "pipeline_contract": {
                    **contract,
                    "collection_field": "records",
                    "hours_field": "hours",
                    "active_values": ["active"],
                },
            },
        )
        write_text(
            root / "docs/precedence.md",
            "The policy selected by `release/active_bundle.json` is authoritative. Retired and draft documents are not.\n",
        )
        write_text(
            root / "notes/old_handoff.md",
            "Legacy note: read docs/legacy_contract.json directly. Superseded by the release pointer.\n",
        )
        write_text(
            root / "archive/prototype.txt",
            "Historical CSV prototype; not a current requirement.\n",
        )
        write_text(root / "config/example.ini", "collection=work_items\nstatus=active\n")

    def defects(self):
        defects = [
            {
                "id": "legacy_authority",
                "path": "src/contract.py",
                "before": 'active = json.loads((root / "release/active_bundle.json").read_text(encoding="utf-8"))\n    policy_path = root / str(active["policy_path"])',
                "after": 'active = {"policy_path": "docs/legacy_contract.json"}\n    policy_path = root / str(active["policy_path"])',
            },
            {
                "id": "wrong_collection",
                "path": "src/normalize.py",
                "before": 'payload.get(str(contract["collection_field"]), [])',
                "after": 'payload.get("records", [])',
            },
            {
                "id": "alias_bypass",
                "path": "src/normalize.py",
                "before": "aliases.get(team_raw.casefold(), team_raw.title())",
                "after": "team_raw.title()",
            },
            {
                "id": "unsafe_active",
                "path": "src/normalize.py",
                "before": "active not in active_values",
                "after": 'not raw.get(contract["active_field"])',
            },
            {
                "id": "integer_hours",
                "path": "src/normalize.py",
                "before": 'hours = float(str(raw[contract["hours_field"]]).strip())',
                "after": 'hours = int(float(str(raw[contract["hours_field"]]).strip()))',
            },
            {
                "id": "oldest_duplicate",
                "path": "src/normalize.py",
                "before": 'timestamp(updated_at) > timestamp(prior["updated_at"])',
                "after": 'timestamp(updated_at) < timestamp(prior["updated_at"])',
            },
            {
                "id": "reverse_order",
                "path": "src/aggregate.py",
                "before": "sorted(totals.items(), key=lambda item: item[0].casefold())",
                "after": "sorted(totals.items(), key=lambda item: item[0].casefold(), reverse=True)",
            },
            {
                "id": "bypass_intermediate",
                "path": "run_pipeline.py",
                "before": "summary = aggregate(persisted)",
                "after": "summary = aggregate(normalized)",
            },
        ]
        order = {
            name: index
            for index, name in enumerate(
                [
                    "alias_bypass",
                    "unsafe_active",
                    "integer_hours",
                    "oldest_duplicate",
                    "reverse_order",
                    "legacy_authority",
                    "wrong_collection",
                    "bypass_intermediate",
                ]
            )
        }
        return sorted(defects, key=lambda defect: order[defect["id"]])

    def readme(self):
        return "# Retrieval-guided pipeline repair\n\nRun `python run_pipeline.py`. The active release pointer selects the authoritative policy. Materialize it as `config/resolved_contract.json`. Use its mappings, aliases, and active values. Normalize case/whitespace, preserve fractional hours, skip malformed records, and retain the newest valid record per identity by timestamp. Write `artifacts/normalized_jobs.json`; aggregation must reload it and write deterministically sorted `artifacts/summary.json` rows with `team`, `job_count`, and `total_hours`. Generalize to alternate policies.\n"
