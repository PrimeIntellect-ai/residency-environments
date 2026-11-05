from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from swg_generators.evaluators.registry import get_evaluator
from swg_generators.generators.registry import get_generator
from swg_generators.generators.retrieval_profile_scenarios import (
    build_profiled_retrieval_scenario,
)
from swg_generators.utils.io import write_json

PROFILE_SEEDS = (100, 102, 103, 107, 108)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/retrieval-d5-oracle-report.json"),
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(".tmp-tests/retrieval-d5-oracle"),
    )
    args = parser.parse_args()
    if args.scratch_root.exists():
        raise FileExistsError(f"choose a fresh scratch root: {args.scratch_root}")
    args.scratch_root.mkdir(parents=True)

    generator = get_generator("retrieval_workspace")
    reports: list[dict[str, object]] = []
    for seed in PROFILE_SEEDS:
        spec = generator.sample_spec(
            difficulty=5,
            seed=seed,
            generation_params={"composition_mode": "hard_atomic"},
        )
        scenario = build_profiled_retrieval_scenario(random.Random(f"{seed}:profiled_retrieval"), spec)
        root = args.scratch_root / f"seed-{seed}"
        bundle = generator.generate_instance(spec, root / "generated")
        evaluator = get_evaluator(
            "retrieval_workspace",
            evaluator_entrypoint=bundle.manifest.evaluator_entrypoint,
        )
        labels = [str(bug["label"]) for bug in scenario["bugs"]]
        if scenario["structure"]["d5_profile"] == "d5_a":
            states = [
                ("untouched", set()),
                ("one_fix", set(labels[:1])),
                ("two_fixes", set(labels[:2])),
                ("all_except_edge", set(labels[:-1])),
                ("full", set(labels)),
            ]
        else:
            schema_fix = {"schema_mapping"} if "schema_mapping" in labels else set()
            states = [
                ("untouched", set()),
                ("authority_schema", {"authority_config", *schema_fix}),
                (
                    "parser_only",
                    {"quantity_parsing", "missing_value_policy", *schema_fix},
                ),
                (
                    "parser_config",
                    {
                        "authority_config",
                        "quantity_parsing",
                        "missing_value_policy",
                        *schema_fix,
                    },
                ),
                ("all_except_edge", set(labels) - {"output_contract"}),
                ("full", set(labels)),
            ]
        rewards = {
            name: _evaluate(
                root / f"state-{index}",
                fixed,
                scenario,
                bundle.visible_root,
                evaluator,
                bundle.manifest,
                bundle.hidden_root,
            )
            for index, (name, fixed) in enumerate(states)
        }
        violations = _violations(str(scenario["structure"]["d5_profile"]), rewards)
        reports.append(
            {
                "seed": seed,
                "profile": scenario["structure"]["d5_profile"],
                "scenario": scenario["scenario_id"],
                "states": rewards,
                "monotonic": list(rewards.values()) == sorted(rewards.values()),
                "violations": violations,
            }
        )

    payload = {
        "schema_version": "retrieval-d5-structural-calibration-v3",
        "profiles": reports,
        "violation_count": sum(len(report["violations"]) for report in reports),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["violation_count"] == 0 else 2


def _evaluate(
    workspace: Path,
    fixed: set[str],
    scenario: dict[str, object],
    visible_root: Path,
    evaluator,
    manifest,
    hidden_root: Path,
) -> float:
    shutil.copytree(visible_root, workspace)
    files = dict(scenario["correct_files"])
    for bug in scenario["bugs"]:
        if str(bug["label"]) not in fixed:
            target = str(bug["target_path"])
            files[target] = bug["apply"](files[target])
    for relative_path, content in files.items():
        target = workspace / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
    return round(float(evaluator.evaluate(workspace, manifest, hidden_root).score), 6)


def _violations(profile: str, rewards: dict[str, float]) -> list[str]:
    violations: list[str] = []
    if list(rewards.values()) != sorted(rewards.values()):
        violations.append("oracle states are not monotonic")
    if rewards["untouched"] > 0.15:
        violations.append("untouched reward exceeds 0.15")
    if rewards["full"] != 1.0:
        violations.append("full reference solution does not score 1.0")
    if profile != "d5_a":
        bounds = {
            "authority_schema": (0.15, 0.30),
            "parser_only": (0.25, 0.45),
            "parser_config": (0.40, 0.65),
            "all_except_edge": (0.65, 0.85),
        }
        for state, (lower, upper) in bounds.items():
            if not lower <= rewards[state] <= upper:
                violations.append(f"{state} reward {rewards[state]:.2f} outside {lower:.2f}-{upper:.2f}")
    return violations


if __name__ == "__main__":
    raise SystemExit(main())
