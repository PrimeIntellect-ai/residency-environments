from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs" / "synthetic_workspace_gym"
RESIDENCY_MANIFESTS = (
    "sft-easy-v4",
    "sft-validation-v4",
    "rl-hard-v4",
    "rl-eval-v4",
    "eval-d1-d4-paired-panel-48-v4",
    "eval-d5-family-calibration-40-v4",
)
VERSION = "0.2.0.dev1"
CREATED_AT = "2026-07-27T00:00:00+05:30"
REBALANCED_CREATED_AT = "2026-08-25T00:00:00+05:30"
TRAINING_READY_CREATED_AT = "2026-09-11T00:00:00+05:30"
PORTABLE_RELEASE_CREATED_AT = "2026-09-21T00:00:00+05:30"
ORIGINAL = {
    "tabular": [
        "monthly_segment_report",
        "channel_status_pivot",
        "weekly_refund_rollup",
    ],
    "script_repair": [
        "inventory_report",
        "path_batch",
        "csv_schema_drift",
        "timestamp_normalization",
    ],
    "pipeline": [
        "team_hours_pipeline",
        "sales_csv_pipeline",
        "artifact_stitch_pipeline",
    ],
    "retrieval_workspace": [
        "service_config_reconciliation",
        "migration_plan_bundle",
        "incident_report_bundle",
    ],
}
HELDOUT = {
    "tabular": ["supplier_restock_summary"],
    "script_repair": ["team_roster_export"],
    "pipeline": ["quality_gate_pipeline"],
    "retrieval_workspace": ["client_adapter_sync"],
}
COMPOSITE = "retrieval_guided_pipeline_repair"

# Qwen3.5-4B calibration showed that a globally balanced curriculum hides large
# family effects.  V2 therefore freezes quotas per family and difficulty.  Each
# family still contributes 128 SFT and 98 atomic RL assignments.
SFT_V2_QUOTAS = {
    "pipeline": {1: 24, 2: 40, 3: 64},
    "retrieval_workspace": {1: 64, 2: 40, 3: 24},
    "script_repair": {1: 16, 2: 32, 3: 80},
    "tabular": {1: 32, 2: 40, 3: 56},
}
RL_V2_ATOMIC_QUOTAS = {
    "pipeline": {4: 78, 5: 20},
    "retrieval_workspace": {4: 28, 5: 70},
    "script_repair": {4: 69, 5: 29},
    "tabular": {4: 39, 5: 59},
}
RL_V2_COMPOSITE_QUOTAS = {4: 40, 5: 80}
RETRIEVAL_D5_SCENARIOS = {
    "d5_a": "client_adapter_sync",
    "d5_b": "client_adapter_policy_sync",
    "d5_c": "versioned_client_migration",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def row(split, family, scenario, difficulty, seed, experiment):
    task_id = f"swg.{split}.{family}.{scenario}.d{difficulty}.s{seed}"
    core = {
        "split": split,
        "family": family,
        "scenario": scenario,
        "difficulty": difficulty,
        "seed": seed,
        "task_id": task_id,
    }
    contributors = ["retrieval_workspace", "pipeline"] if family == "composite_workspace" else [family]
    return {
        **core,
        "env_id": None,
        "metadata": {
            "experiment": experiment,
            "contributing_families": contributors,
            "assignment_fingerprint": digest(core),
        },
    }


def pool(families, split, difficulties, seeds, experiment):
    return [
        row(split, family, scenario, difficulty, seed, experiment)
        for family, scenarios in families.items()
        for scenario in scenarios
        for difficulty in difficulties
        for seed in seeds
    ]


def balanced(items, count, shuffle_seed):
    groups = {}
    for item in items:
        groups.setdefault((item["family"], item["scenario"], item["difficulty"]), []).append(copy.deepcopy(item))
    rng = random.Random(shuffle_seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        progress = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            raise ValueError("not enough unique tasks")
    return selected


def balanced_by_family(items, count, shuffle_seed):
    families = sorted({item["family"] for item in items})
    random.Random(shuffle_seed).shuffle(families)
    base, remainder = divmod(count, len(families))
    selected = []
    for index, family in enumerate(families):
        family_items = [item for item in items if item["family"] == family]
        selected.extend(balanced(family_items, base + int(index < remainder), shuffle_seed + index))
    random.Random(shuffle_seed).shuffle(selected)
    return selected


def quota_sample(items, quotas, shuffle_seed):
    """Select deterministic counts for every family/difficulty cell."""

    selected = []
    offset = 0
    for family, difficulty_quotas in quotas.items():
        for difficulty, count in difficulty_quotas.items():
            cell = [item for item in items if item["family"] == family and item["difficulty"] == difficulty]
            selected.extend(balanced(cell, count, shuffle_seed + offset))
            offset += 1
    random.Random(shuffle_seed).shuffle(selected)
    return selected


def retrieval_d5_profile(seed):
    """Mirror the generator's deterministic 20/50/30 retrieval profile routing."""

    bucket = int(seed) % 10
    return "d5_a" if bucket < 2 else "d5_b" if bucket < 7 else "d5_c"


def retrieval_d5_rows(split, seeds, experiment, *, include_profiles=None):
    allowed = set(include_profiles or RETRIEVAL_D5_SCENARIOS)
    return [
        row(
            split,
            "retrieval_workspace",
            RETRIEVAL_D5_SCENARIOS[profile],
            5,
            seed,
            experiment,
        )
        for seed in seeds
        if (profile := retrieval_d5_profile(seed)) in allowed
    ]


def freeze(
    name,
    assignments,
    metadata,
    *,
    manifest_version="v1",
    created_at=CREATED_AT,
):
    body = {
        "name": name,
        "version": manifest_version,
        "created_at": created_at,
        "split_specs": {},
        "assignments": assignments,
    }
    body["metadata"] = {
        "environment_version": VERSION,
        "frozen": True,
        "assignment_count": len(assignments),
        "manifest_fingerprint": digest(body),
        **metadata,
    }
    return body


def build():
    result = {}
    for family, scenarios in ORIGINAL.items():
        name = f"train-specialist-{family}"
        result[name] = freeze(
            name,
            balanced(pool({family: scenarios}, "train", [2, 3, 4], range(80), name), 512, 42),
            {"curriculum": "specialist", "shuffle_seed": 42, "composite_fraction": 0.0},
        )
    originals = pool(ORIGINAL, "train", [2, 3, 4], range(80), "all-family")
    composites = pool(
        {"composite_workspace": [COMPOSITE]},
        "train",
        [2, 3, 4],
        range(80),
        "composition",
    )
    for seed in (42, 43):
        name = f"train-all-family-seed-{seed}"
        chosen = balanced_by_family(originals, 512, seed)
        for item in chosen:
            item["metadata"]["experiment"] = name
        result[name] = freeze(
            name,
            chosen,
            {
                "curriculum": "all_family",
                "shuffle_seed": seed,
                "composite_fraction": 0.0,
            },
        )
        name = f"train-composition-20pct-seed-{seed}"
        chosen = balanced_by_family(originals, 410, seed) + balanced(composites, 102, seed)
        random.Random(seed).shuffle(chosen)
        for item in chosen:
            item["metadata"]["experiment"] = name
        result[name] = freeze(
            name,
            chosen,
            {
                "curriculum": "composition_augmented",
                "shuffle_seed": seed,
                "original_count": 410,
                "composite_count": 102,
                "composite_fraction": 102 / 512,
            },
        )
    name = "eval-id-d3-d5"
    result[name] = freeze(
        name,
        pool(ORIGINAL, "test", [3, 4, 5], range(90, 100), name),
        {"panel": "in_distribution"},
    )
    name = "eval-scenario-heldout"
    result[name] = freeze(
        name,
        pool(HELDOUT, "heldout", [3, 4, 5], range(100, 120), name),
        {"panel": "scenario_heldout"},
    )
    name = "eval-d5-panel-24"
    panel = []
    for family, scenarios in ORIGINAL.items():
        for index in range(6):
            panel.append(
                row(
                    "test",
                    family,
                    scenarios[index % len(scenarios)],
                    5,
                    190 + index,
                    name,
                )
            )
    result[name] = freeze(name, panel, {"panel": "frozen_d5", "rollouts_per_task": 5})

    difficulty_band_panels = {
        "eval-d1-d2-panel-24": [
            row("heldout", family, scenarios[0], difficulty, seed, "eval-d1-d2-panel-24")
            for family, scenarios in HELDOUT.items()
            for difficulty in (1, 2)
            for seed in range(120, 123)
        ],
        "eval-d3-d4-panel-24": [
            row(
                "heldout",
                family,
                scenarios[0],
                difficulty,
                seed,
                "eval-d3-d4-panel-24",
            )
            for family, scenarios in HELDOUT.items()
            for difficulty, seeds in ((3, range(123, 126)), (4, range(220, 223)))
            for seed in seeds
        ],
        "eval-d5-heldout-panel-24": [
            row(
                "heldout",
                family,
                scenarios[0],
                5,
                seed,
                "eval-d5-heldout-panel-24",
            )
            for family, scenarios in HELDOUT.items()
            for seed in range(223, 229)
        ],
    }
    for name, panel in difficulty_band_panels.items():
        result[name] = freeze(
            name,
            panel,
            {
                "panel": "difficulty_band_heldout",
                "scenario_partition": "heldout",
                "rollouts_per_task": 5,
            },
        )

    name = "eval-composite-heldout-24"
    panel = [
        row("heldout", "composite_workspace", COMPOSITE, difficulty, seed, name)
        for difficulty in (3, 4, 5)
        for seed in range(200, 208)
    ]
    result[name] = freeze(name, panel, {"panel": "composite_heldout", "document_fixture_split": "heldout"})

    # Training-purpose manifests deliberately use non-overlapping seed bands and
    # explicit split labels.  They are independent of the historical Qwen RL
    # matrix above, which mixed D2-D4 tasks under generic ``train`` manifests.
    name = "sft-easy-v1"
    result[name] = freeze(
        name,
        balanced_by_family(
            pool(ORIGINAL, "sft_train", [1, 2, 3], range(0, 80), name),
            512,
            20260822,
        ),
        {
            "training_purpose": "supervised_midtraining",
            "curriculum": "easy_atomic",
            "difficulty_range": [1, 3],
            "scenario_partition": "development",
            "seed_range": [0, 79],
            "shuffle_seed": 20260822,
        },
    )

    name = "sft-validation-v1"
    result[name] = freeze(
        name,
        pool(HELDOUT, "sft_validation", [1, 2, 3], range(120, 128), name),
        {
            "training_purpose": "supervised_validation",
            "curriculum": "easy_atomic_heldout",
            "difficulty_range": [1, 3],
            "scenario_partition": "heldout",
            "seed_range": [120, 127],
            "disjoint_from": ["sft-easy-v1"],
        },
    )

    name = "rl-hard-v1"
    hard_atomic = pool(ORIGINAL, "rl_train", [4, 5], range(128, 190), name)
    hard_composite = pool(
        {"composite_workspace": [COMPOSITE]},
        "rl_train",
        [4, 5],
        range(128, 190),
        name,
    )
    hard_assignments = balanced_by_family(hard_atomic, 392, 20260823) + balanced(hard_composite, 120, 20260823)
    random.Random(20260823).shuffle(hard_assignments)
    result[name] = freeze(
        name,
        hard_assignments,
        {
            "training_purpose": "reinforcement_learning",
            "curriculum": "hard_atomic_and_composite",
            "difficulty_range": [4, 5],
            "scenario_partition": "development",
            "seed_range": [128, 189],
            "atomic_count": 392,
            "composite_count": 120,
            "composite_fraction": 120 / 512,
            "shuffle_seed": 20260823,
        },
    )

    name = "rl-eval-v1"
    rl_eval_atomic = pool(HELDOUT, "rl_eval", [4, 5], range(220, 230), name)
    rl_eval_composite = pool(
        {"composite_workspace": [COMPOSITE]},
        "rl_eval",
        [4, 5],
        range(230, 250),
        name,
    )
    result[name] = freeze(
        name,
        rl_eval_atomic + rl_eval_composite,
        {
            "training_purpose": "reinforcement_learning_evaluation",
            "curriculum": "hard_scenario_seed_and_template_heldout",
            "difficulty_range": [4, 5],
            "scenario_partition": "heldout",
            "atomic_seed_range": [220, 229],
            "composite_seed_range": [230, 249],
            "atomic_count": len(rl_eval_atomic),
            "composite_count": len(rl_eval_composite),
            "disjoint_from": ["rl-hard-v1"],
            "document_fixture_split": "heldout",
        },
    )

    # V2 keeps the immutable V1 curricula intact and encodes calibration as
    # explicit family/difficulty quotas.  The atomic RL mix targets a useful
    # early-RL success band: pipeline and script-repair receive more D4 work,
    # while retrieval receives mostly hard D5 profiles.  D5 composites are
    # doubled relative to D4 so retrieval-guided compositions remain demanding.
    name = "sft-easy-v2"
    sft_v2_pool = pool(ORIGINAL, "sft_train", [1, 2, 3], range(0, 80), name)
    result[name] = freeze(
        name,
        quota_sample(sft_v2_pool, SFT_V2_QUOTAS, 20260825),
        {
            "training_purpose": "supervised_midtraining",
            "curriculum": "family_calibrated_easy_atomic",
            "difficulty_range": [1, 3],
            "scenario_partition": "development",
            "seed_range": [0, 79],
            "family_difficulty_quotas": SFT_V2_QUOTAS,
            "calibration_model": "Qwen/Qwen3.5-4B",
            "shuffle_seed": 20260825,
            "supersedes": "sft-easy-v1",
        },
        manifest_version="v2",
        created_at=REBALANCED_CREATED_AT,
    )

    name = "sft-validation-v2"
    result[name] = freeze(
        name,
        pool(HELDOUT, "sft_validation", [1, 2, 3], range(120, 128), name),
        {
            "training_purpose": "supervised_validation",
            "curriculum": "family_stratified_easy_atomic_heldout",
            "difficulty_range": [1, 3],
            "scenario_partition": "heldout",
            "seed_range": [120, 127],
            "reporting_strata": ["family", "difficulty"],
            "disjoint_from": ["sft-easy-v2"],
            "supersedes": "sft-validation-v1",
        },
        manifest_version="v2",
        created_at=REBALANCED_CREATED_AT,
    )

    name = "rl-hard-v2"
    rl_seed_range = range(128, 220)
    standard_rl_pool = pool(
        ORIGINAL,
        "rl_train",
        [4, 5],
        rl_seed_range,
        name,
    )
    # Profiled retrieval D5 rows replace the legacy explicit scenarios.  Drop
    # d5_a entirely and take all 27 available d5_c seeds plus 43 d5_b seeds.
    standard_rl_pool = [
        item for item in standard_rl_pool if not (item["family"] == "retrieval_workspace" and item["difficulty"] == 5)
    ]
    retrieval_b = retrieval_d5_rows("rl_train", rl_seed_range, name, include_profiles={"d5_b"})
    retrieval_c = retrieval_d5_rows("rl_train", rl_seed_range, name, include_profiles={"d5_c"})
    atomic_v2 = quota_sample(
        standard_rl_pool,
        {family: quotas for family, quotas in RL_V2_ATOMIC_QUOTAS.items() if family != "retrieval_workspace"}
        | {"retrieval_workspace": {4: 28}},
        20260826,
    )
    atomic_v2.extend(balanced(retrieval_b, 43, 20260836))
    atomic_v2.extend(balanced(retrieval_c, 27, 20260837))
    composite_v2_pool = pool(
        {"composite_workspace": [COMPOSITE]},
        "rl_train",
        [4, 5],
        rl_seed_range,
        name,
    )
    composite_v2 = quota_sample(
        composite_v2_pool,
        {"composite_workspace": RL_V2_COMPOSITE_QUOTAS},
        20260827,
    )
    hard_v2 = atomic_v2 + composite_v2
    random.Random(20260828).shuffle(hard_v2)
    result[name] = freeze(
        name,
        hard_v2,
        {
            "training_purpose": "reinforcement_learning",
            "curriculum": "family_calibrated_hard_atomic_and_composite",
            "difficulty_range": [4, 5],
            "scenario_partition": "development",
            "seed_range": [128, 219],
            "atomic_count": len(atomic_v2),
            "composite_count": len(composite_v2),
            "composite_fraction": len(composite_v2) / len(hard_v2),
            "family_difficulty_quotas": RL_V2_ATOMIC_QUOTAS,
            "composite_difficulty_quotas": RL_V2_COMPOSITE_QUOTAS,
            "retrieval_d5_profile_quotas": {"d5_a": 0, "d5_b": 43, "d5_c": 27},
            "calibration_model": "Qwen/Qwen3.5-4B",
            "shuffle_seed": 20260828,
            "supersedes": "rl-hard-v1",
        },
        manifest_version="v2",
        created_at=REBALANCED_CREATED_AT,
    )

    name = "rl-eval-v2"
    atomic_eval_v2 = []
    for family, scenarios in HELDOUT.items():
        atomic_eval_v2.extend(row("rl_eval", family, scenarios[0], 4, seed, name) for seed in range(220, 230))
        if family == "retrieval_workspace":
            atomic_eval_v2.extend(retrieval_d5_rows("rl_eval", range(220, 230), name))
        else:
            atomic_eval_v2.extend(row("rl_eval", family, scenarios[0], 5, seed, name) for seed in range(220, 230))
    composite_eval_v2 = pool(
        {"composite_workspace": [COMPOSITE]},
        "rl_eval",
        [4, 5],
        range(230, 250),
        name,
    )
    result[name] = freeze(
        name,
        atomic_eval_v2 + composite_eval_v2,
        {
            "training_purpose": "reinforcement_learning_evaluation",
            "curriculum": "family_stratified_hard_heldout",
            "difficulty_range": [4, 5],
            "scenario_partition": "heldout",
            "atomic_seed_range": [220, 229],
            "composite_seed_range": [230, 249],
            "atomic_count": len(atomic_eval_v2),
            "composite_count": len(composite_eval_v2),
            "reporting_strata": ["family", "difficulty", "profile"],
            "disjoint_from": ["rl-hard-v2"],
            "supersedes": "rl-eval-v1",
        },
        manifest_version="v2",
        created_at=REBALANCED_CREATED_AT,
    )

    name = "eval-d1-d4-paired-panel-48-v2"
    paired_panel = [
        row("heldout", family, scenarios[0], difficulty, seed, name)
        for family, scenarios in HELDOUT.items()
        for seed in range(250, 253)
        for difficulty in (1, 2, 3, 4)
    ]
    result[name] = freeze(
        name,
        paired_panel,
        {
            "panel": "paired_family_difficulty_calibration",
            "scenario_partition": "heldout",
            "paired_seed_range": [250, 252],
            "reporting_strata": ["family", "difficulty"],
            "rollouts_per_task": 5,
        },
        manifest_version="v2",
        created_at=REBALANCED_CREATED_AT,
    )

    name = "eval-d5-family-calibration-40-v2"
    d5_panel = []
    for family, scenarios in HELDOUT.items():
        if family == "retrieval_workspace":
            d5_panel.extend(retrieval_d5_rows("heldout", range(260, 270), name))
        else:
            d5_panel.extend(row("heldout", family, scenarios[0], 5, seed, name) for seed in range(260, 270))
    result[name] = freeze(
        name,
        d5_panel,
        {
            "panel": "family_profile_d5_calibration",
            "scenario_partition": "heldout_and_profiled",
            "seed_range": [260, 269],
            "reporting_strata": ["family", "profile"],
            "rollouts_per_task": 5,
        },
        manifest_version="v2",
        created_at=REBALANCED_CREATED_AT,
    )
    v3_sources = {
        "sft-easy-v2": "sft-easy-v3",
        "sft-validation-v2": "sft-validation-v3",
        "rl-hard-v2": "rl-hard-v3",
        "rl-eval-v2": "rl-eval-v3",
        "eval-d1-d4-paired-panel-48-v2": "eval-d1-d4-paired-panel-48-v3",
        "eval-d5-family-calibration-40-v2": "eval-d5-family-calibration-40-v3",
    }
    for source_name, name in v3_sources.items():
        source = result[source_name]
        assignments = copy.deepcopy(source["assignments"])
        for assignment in assignments:
            assignment["metadata"]["experiment"] = name
        metadata = {
            key: copy.deepcopy(value)
            for key, value in source["metadata"].items()
            if key not in {"assignment_count", "environment_version", "frozen", "manifest_fingerprint"}
        }
        metadata.update(
            {
                "supersedes": source_name,
                "training_readiness_correction": "grounded-exact-json-and-baseline-normalized-rewards-v1",
            }
        )
        metadata["disjoint_from"] = [v3_sources.get(item, item) for item in metadata.get("disjoint_from", [])]
        result[name] = freeze(
            name,
            assignments,
            metadata,
            manifest_version="v3",
            created_at=TRAINING_READY_CREATED_AT,
        )
    v4_sources = {name: name.replace("-v3", "-v4") for name in v3_sources.values()}
    for source_name, name in v4_sources.items():
        source = result[source_name]
        assignments = copy.deepcopy(source["assignments"])
        for assignment in assignments:
            assignment["metadata"]["experiment"] = name
        metadata = {
            key: copy.deepcopy(value)
            for key, value in source["metadata"].items()
            if key not in {"assignment_count", "environment_version", "frozen", "manifest_fingerprint"}
        }
        metadata.update(
            {
                "supersedes": source_name,
                "training_readiness_correction": "canonical-lf-and-evaluator-identical-baselines-v1",
            }
        )
        metadata["disjoint_from"] = [v4_sources.get(item, item) for item in metadata.get("disjoint_from", [])]
        result[name] = freeze(
            name,
            assignments,
            metadata,
            manifest_version="v4",
            created_at=PORTABLE_RELEASE_CREATED_AT,
        )
    return result


def training_config(manifest_name: str, steps: int = 200) -> str:
    return (
        'model = "Qwen/Qwen3.5-4B"\nloss = "rl"\n'
        f"max_steps = {steps}\nbatch_size = 128\nrollouts_per_example = 8\n\n"
        "[sampling]\nmax_tokens = 1024\ntemperature = 0.7\n\n"
        "[checkpoints]\ninterval = 25\nkeep_cloud = 8\n\n"
        "[adapters]\ninterval = 25\nkeep_last = 8\n\n"
        '[[env]]\nid = "yadnyesh/synthetic-workspace-gym@0.2.0.dev1"\n\n'
        f'[env.args]\nfrozen_manifest = "{manifest_name}"\nsplit = "train"\n'
        "max_examples = 512\nmax_turns = 25\nmax_tool_steps = 64\n"
        'reward_mode = "score"\nsample_strategy = "first"\nshuffle = false\n'
    )


def evaluation_config(manifest_name: str, count: int, rollouts: int, split: str) -> str:
    return (
        'model = "Qwen/Qwen3.5-4B"\n'
        f"num_examples = {count}\nrollouts_per_example = {rollouts}\n"
        "max_tokens = 4096\ntemperature = 0.7\nmax_concurrent = 5\n"
        "max_retries = 0\ntimeout_minutes = 180\n\n"
        '[[eval]]\nid = "yadnyesh/synthetic-workspace-gym"\n'
        f'eval_name = "base-{manifest_name}"\n'
        f'env_args = {{ manifest = "{manifest_name}" }}\n'
    )


def native_evaluation_config(manifest_name: str, count: int, rollouts: int) -> str:
    """Render the self-hosted Verifiers V1 + Prime runtime configuration."""

    return (
        'model = "Qwen/Qwen3.5-4B"\n'
        f"num_tasks = {count}\nnum_rollouts = {rollouts}\n"
        "shuffle = false\nmax_concurrent = 5\nrich = false\nserver = false\n"
        "push = true\n"
        f'output_dir = "outputs/qwen35-4b-{manifest_name}-5r-bash"\n\n'
        "[env]\nmax_concurrent_agents = 1\n\n"
        "[env.taskset]\n"
        'id = "synthetic-workspace-gym"\n'
        f'manifest = "{manifest_name}"\n'
        'image = "python:3.12-slim"\n\n'
        "[env.interception]\n"
        'type = "elastic"\nmultiplex = 32\n\n'
        "[env.agent.harness]\n"
        'id = "bash"\ntool_timeout = 600.0\nedit = true\nsearch = false\n\n'
        "[env.agent.runtime]\n"
        'type = "prime"\nimage = "python:3.11-slim"\nworkdir = "/app"\n'
        "vm = true\ncpu = 1.0\nmemory = 2.0\ndisk = 5.0\n"
        "idle_timeout = 3600.0\n\n"
        "[env.agent.timeout]\nrollout = 360.0\n\n"
        "[client]\n"
        'base_url = "https://api.pinference.ai/api/v1"\n'
        'api_key_var = "PRIME_API_KEY"\ntype = "eval"\n\n'
        "[sampling]\ntemperature = 0.7\nmax_tokens = 4096\n"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = build()
    for name in RESIDENCY_MANIFESTS:
        payload = manifests[name]
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        (OUT / f"{name}.json").write_bytes(serialized.encode("utf-8"))
    print(
        json.dumps(
            {
                "manifests": len(RESIDENCY_MANIFESTS),
                "tasks": sum(len(manifests[name]["assignments"]) for name in RESIDENCY_MANIFESTS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
