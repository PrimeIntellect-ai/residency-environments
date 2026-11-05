from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from swg_generators.analysis.oracle_calibration import (
    evaluate_ordered_oracle_states,
)
from swg_generators.evaluators.registry import get_evaluator
from swg_generators.generators.registry import get_generator
from swg_generators.generators.retrieval_profile_scenarios import (
    build_profiled_retrieval_scenario,
)
from swg_generators.generators.tabular_program_synthesis import (
    build_account_event_program_scenario,
)
from swg_generators.utils.io import write_json

PROFILE_AWARE_SEEDS = (100, 103, 106, 108, 109)
FAMILIES = (
    "pipeline",
    "retrieval_workspace",
    "script_repair",
    "tabular",
    "composite_workspace",
)
QUALITY_LIMITS = {
    "untouched": 0.15,
    "one_capability": 0.40,
    "two_capabilities": 0.65,
    "all_but_one": 0.85,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/d5-calibration-report.json"),
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(".tmp-tests/d5-calibration"),
    )
    args = parser.parse_args()
    if args.scratch_root.exists():
        raise FileExistsError(f"Scratch root already exists; choose a fresh path: {args.scratch_root}")
    args.scratch_root.mkdir(parents=True)

    reports: list[dict[str, object]] = []
    for family in FAMILIES:
        generator = get_generator(family)
        for seed in PROFILE_AWARE_SEEDS:
            scenario_id = _scenario_id(generator, family, seed)
            spec = generator.sample_spec(
                difficulty=5,
                seed=seed,
                scenario_id=scenario_id,
                generation_params={"composition_mode": "hard_atomic"},
            )
            instance_root = args.scratch_root / family / f"seed-{seed}"
            bundle = generator.generate_instance(spec, instance_root / "generated")
            correct_files, selected_bugs = _calibration_inputs(generator, family, spec, bundle)
            realization = dict(bundle.manifest.metadata["difficulty_realization"])
            profile = str(realization.get("profile", scenario_id))
            report = evaluate_ordered_oracle_states(
                task_id=(f"swg.calibration.{family}.{scenario_id}.d5.{profile}.s{seed}"),
                correct_files=correct_files,
                selected_bugs=selected_bugs,
                visible_template=bundle.visible_root,
                scratch_root=instance_root / "states",
                evaluator=get_evaluator(
                    family,
                    evaluator_entrypoint=bundle.manifest.evaluator_entrypoint,
                ),
                manifest=bundle.manifest,
                hidden_root=bundle.hidden_root,
                capability_count=int(realization.get("capability_count", 0)),
                semantic_dependency_depth=int(realization.get("semantic_dependency_depth", 0)),
            )
            report.update(
                {
                    "family": family,
                    "seed": seed,
                    "profile": profile,
                }
            )
            quality_violations = _quality_violations(dict(report["states"]), len(selected_bugs))
            report["quality_gate_passed"] = not quality_violations
            report["quality_violations"] = quality_violations
            reports.append(report)

    payload = {
        "schema_version": "d5-calibration-v2",
        "seeds": list(PROFILE_AWARE_SEEDS),
        "task_count": len(reports),
        "monotonic_task_count": sum(bool(report["monotonic"]) for report in reports),
        "monotonicity_violation_count": sum(len(report["violations"]) for report in reports),
        "quality_violation_count": sum(len(report["quality_violations"]) for report in reports),
        "violation_count": sum(len(report["violations"]) + len(report["quality_violations"]) for report in reports),
        "tasks": reports,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["violation_count"] == 0 else 2


def _scenario_id(generator, family: str, seed: int) -> str:
    if family == "retrieval_workspace":
        return str(generator.resolve_scenario_id(difficulty=5, seed=seed))
    return {
        "pipeline": "team_hours_pipeline",
        "script_repair": "inventory_report",
        "tabular": "account_event_program_synthesis",
        "composite_workspace": "retrieval_guided_pipeline_repair",
    }[family]


def _calibration_inputs(generator, family: str, spec, bundle):
    if family == "pipeline":
        scenario = generator.team_hours_pipeline_scenario(random.Random(spec.seed), spec)
        return dict(scenario["files"]), list(scenario["bugs"])
    if family == "retrieval_workspace":
        scenario = build_profiled_retrieval_scenario(random.Random(f"{spec.seed}:profiled_retrieval"), spec)
        return dict(scenario["correct_files"]), list(scenario["bugs"])
    if family == "tabular":
        scenario = build_account_event_program_scenario(
            random.Random(f"{spec.seed}:account_event_program_synthesis"), spec
        )
        return dict(bundle.manifest.reference_solution["files"]), list(scenario["bugs"])
    if family == "script_repair":
        scenario = dict(generator.select_scenario(spec, generator.scenario_pool(spec)))
        materialize = scenario.get("materialize")
        if callable(materialize):
            scenario.update(dict(materialize(spec)))
        selected_bugs, _ = generator.select_bugs(scenario, spec)
        return dict(bundle.manifest.reference_solution["files"]), list(selected_bugs)

    defects = list(generator.defects())[:8]
    selected_bugs = [
        {
            "label": defect["id"],
            "target_path": defect["path"],
            "apply": _replacement(defect["before"], defect["after"]),
        }
        for defect in defects
    ]
    return dict(bundle.manifest.reference_solution["files"]), selected_bugs


def _replacement(before: str, after: str):
    def apply(content: str) -> str:
        updated = content.replace(before, after, 1)
        if updated == content:
            raise ValueError("composite calibration defect no longer matches source")
        return updated

    return apply


def _quality_violations(states: dict[str, float], defect_count: int) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for state, maximum in QUALITY_LIMITS.items():
        # With two defects, the two-capability state is the complete solution;
        # it is not a partial state and must score 1.0.
        if state == "two_capabilities" and defect_count <= 2:
            continue
        actual = float(states[state])
        if actual > maximum + 1e-9:
            violations.append(
                {
                    "state": state,
                    "expected_max": maximum,
                    "actual": actual,
                    "reason": "partial solution exceeds shared D5 ceiling",
                }
            )
    if float(states["full"]) != 1.0:
        violations.append(
            {
                "state": "full",
                "expected": 1.0,
                "actual": float(states["full"]),
                "reason": "reference solution must score exactly 1.0",
            }
        )
    return violations


if __name__ == "__main__":
    raise SystemExit(main())
