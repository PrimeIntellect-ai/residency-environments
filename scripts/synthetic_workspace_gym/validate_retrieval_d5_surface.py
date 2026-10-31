from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from swg_generators.evaluators.registry import get_evaluator
from swg_generators.generators.d5_profiles import select_weighted_d5_profile
from swg_generators.generators.registry import get_generator
from swg_generators.generators.retrieval_profile_scenarios import (
    build_profiled_retrieval_scenario,
)
from swg_generators.utils.io import write_json

PROFILE_SCENARIOS = {
    "d5_a": "client_adapter_sync",
    "d5_b": "client_adapter_policy_sync",
    "d5_c": "versioned_client_migration",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-per-profile", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_SCENARIOS),
        default=tuple(PROFILE_SCENARIOS),
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(".tmp-tests/retrieval-d5-surface-validation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/retrieval-d5-surface-validation.json"),
    )
    args = parser.parse_args()
    if args.scratch_root.exists():
        raise FileExistsError(f"choose a fresh scratch root: {args.scratch_root}")
    args.scratch_root.mkdir(parents=True)

    generator = get_generator("retrieval_workspace")
    reports: list[dict[str, object]] = []
    all_violations: list[str] = []

    for profile in args.profiles:
        selected = profile_seeds(profile, args.start_index + args.seeds_per_profile)
        seeds = selected[args.start_index :]
        fingerprints: set[str] = set()
        bug_failures: Counter[str] = Counter()
        public_rejections = 0
        public_reference_passes = 0
        reference_passes = 0
        profile_violations: list[str] = []

        for index, seed in enumerate(seeds):
            spec = generator.sample_spec(
                difficulty=5,
                seed=seed,
                generation_params={"composition_mode": "hard_atomic"},
            )
            scenario = build_profiled_retrieval_scenario(random.Random(f"{seed}:profiled_retrieval"), spec)
            prefix = f"{profile}.seed-{seed}"
            profile_violations.extend(validate_evidence_surface(prefix, scenario))

            seed_root = args.scratch_root / profile / f"seed-{index:03d}-{seed}"
            bundle = generator.generate_instance(spec, seed_root / "generated")
            provenance = dict(bundle.manifest.metadata["release_provenance"])
            fingerprints.add(str(provenance["generation_fingerprint"]))

            evaluator = get_evaluator(
                "retrieval_workspace",
                evaluator_entrypoint=bundle.manifest.evaluator_entrypoint,
            )
            solved = seed_root / "reference"
            shutil.copytree(bundle.visible_root, solved)
            apply_files(solved, dict(scenario["correct_files"]))
            reference = evaluator.evaluate(solved, bundle.manifest, bundle.hidden_root)
            if reference.score == 1.0 and reference.success:
                reference_passes += 1
            else:
                profile_violations.append(f"{prefix}: reference reward {reference.score:.3f} is not 1.0")

            broken_public = run_public(bundle.visible_root)
            if broken_public.returncode != 0:
                public_rejections += 1
            else:
                profile_violations.append(f"{prefix}: public check accepted the untouched broken workspace")
            solved_public = run_public(solved)
            if solved_public.returncode == 0:
                public_reference_passes += 1
            else:
                profile_violations.append(
                    f"{prefix}: public check rejected the reference solution: {solved_public.stderr[-400:]}"
                )

            for bug_index, bug in enumerate(scenario["bugs"]):
                label = str(bug["label"])
                workspace = seed_root / f"single-bug-{bug_index:02d}-{label}"
                shutil.copytree(bundle.visible_root, workspace)
                files = dict(scenario["correct_files"])
                target_path = str(bug["target_path"])
                files[target_path] = bug["apply"](files[target_path])
                apply_files(workspace, files)
                result = evaluator.evaluate(workspace, bundle.manifest, bundle.hidden_root)
                if result.score < 1.0 and result.diagnostics.get("failed_capabilities"):
                    bug_failures[label] += 1
                else:
                    profile_violations.append(f"{prefix}: isolated bug {label} did not lose capability credit")

        expected_bug_checks = args.seeds_per_profile
        for label, count in sorted(bug_failures.items()):
            if count != expected_bug_checks:
                profile_violations.append(
                    f"{profile}: isolated bug {label} failed {count}/{expected_bug_checks} validation checks"
                )

        report = {
            "profile": profile,
            "scenario": PROFILE_SCENARIOS[profile],
            "seed_count": len(seeds),
            "seeds": seeds,
            "distinct_generation_fingerprints": len(fingerprints),
            "reference_passes": reference_passes,
            "public_broken_rejections": public_rejections,
            "public_reference_passes": public_reference_passes,
            "isolated_bug_failure_counts": dict(sorted(bug_failures.items())),
            "violations": profile_violations,
        }
        reports.append(report)
        all_violations.extend(profile_violations)

    payload = {
        "schema_version": "retrieval-d5-surface-validation-v2",
        "profiles": reports,
        "total_generated": sum(int(report["seed_count"]) for report in reports),
        "violation_count": len(all_violations),
        "violations": all_violations,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not all_violations else 2


def profile_seeds(profile: str, count: int) -> list[int]:
    seeds: list[int] = []
    seed = 0
    while len(seeds) < count:
        selected = select_weighted_d5_profile(5, seed)
        if selected is not None and selected.profile_id == profile:
            seeds.append(seed)
        seed += 1
    return seeds


def validate_evidence_surface(prefix: str, scenario: dict[str, object]) -> list[str]:
    violations: list[str] = []
    files = dict(scenario["files"])
    roots = set(str(item) for item in scenario["document_roots"])
    if scenario["scenario_id"] == "client_adapter_sync":
        if not {"changelog", "docs", "notes"}.issubset(roots):
            violations.append(f"{prefix}: active A evidence roots are not advertised")
        combined = "\n".join(
            str(files[path])
            for path in (
                "docs/api_reference.md",
                "docs/pagination_contract.md",
                "notes/warehouse_policy.md",
                "notes/record_quality.md",
            )
        )
        for fact in (
            "records",
            "quantity",
            "next_cursor",
            "unknown",
            "non-numeric",
        ):
            if fact not in combined:
                violations.append(f"{prefix}: A authority surface omits {fact}")
        return violations
    if not {"policies", "release"}.issubset(roots):
        violations.append(f"{prefix}: policies/release missing from document roots")

    stages = dict(scenario["composition_spec"])["stages"]
    for stage in stages:
        for required in dict(stage)["required_inputs"]:
            path = str(required)
            if path not in files and path not in dict(scenario["correct_files"]):
                violations.append(f"{prefix}: required input {path} is absent")
                continue
            root = path.split("/", 1)[0]
            if path in files and root not in roots and root not in {"config", "src"}:
                violations.append(f"{prefix}: required evidence {path} is outside advertised roots")

    manifest = json.loads(str(files["release/current_manifest.json"]))
    expected_manifest = {
        "active_api": "v3",
        "cutover": "2026-03-01",
        "policy_bundle": "warehouse-policy-2026-03",
        "region": "eu-west",
    }
    if manifest != expected_manifest:
        violations.append(f"{prefix}: release manifest is incomplete or ambiguous")

    precedence = str(files["docs/authority_precedence.md"]).lower()
    if "manifest" not in precedence or "archived" not in precedence:
        violations.append(f"{prefix}: precedence document is incomplete")
    policy_path = (
        "policies/warehouse-policy-2026-03.md"
        if scenario["scenario_id"] == "client_adapter_policy_sync"
        else "policies/regional_override.md"
    )
    policy = str(files[policy_path])
    for fact in ("unknown", "eu-west", "ams-old", "eu-central", "dub-legacy"):
        if fact not in policy:
            violations.append(f"{prefix}: policy omits authoritative fact {fact}")
    legacy = str(files["notes/legacy_rollout.md"]).lower()
    if "archived" not in legacy or "superseded" not in legacy:
        violations.append(f"{prefix}: stale evidence is not explicitly superseded")
    return violations


def apply_files(workspace: Path, files: dict[str, object]) -> None:
    for relative_path, content in files.items():
        target = workspace / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")


def run_public(workspace: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "run_example.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
