from __future__ import annotations

import hashlib
import json
from typing import Any

from swg_generators.schemas import ComplexityProfile, EnvironmentFamily


def normalize_difficulty(value: int | str) -> int:
    if isinstance(value, int):
        difficulty = value
    else:
        mapping = {"easy": 1, "medium": 3, "hard": 5}
        key = value.strip().lower()
        if key.isdigit():
            difficulty = int(key)
        elif key in mapping:
            difficulty = mapping[key]
        else:
            raise ValueError(f"Unsupported difficulty value: {value}")
    if not 1 <= difficulty <= 5:
        raise ValueError("difficulty must be between 1 and 5")
    return difficulty


def make_env_id(
    family: EnvironmentFamily,
    difficulty: int,
    seed: int,
    task_params: dict[str, Any],
    *,
    scenario_id: str | None = None,
    generation_params: dict[str, Any] | None = None,
) -> str:
    fingerprint = hashlib.sha1(
        json.dumps(
            {
                "family": family.value,
                "difficulty": difficulty,
                "seed": seed,
                "scenario_id": scenario_id,
                "task_params": task_params,
                "generation_params": generation_params or {},
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:8]
    return f"{family.value}-d{difficulty}-s{seed}-{fingerprint}"


def build_complexity_profile(family: EnvironmentFamily, difficulty: int) -> ComplexityProfile:
    base = {
        1: dict(
            file_count=3,
            distractor_count=0,
            dependency_depth=1,
            reasoning_hops=1,
            transformation_count=1,
            bug_subtlety=1,
            execution_required=False,
            output_constraint_strength=2,
        ),
        2: dict(
            file_count=4,
            distractor_count=1,
            dependency_depth=1,
            reasoning_hops=2,
            transformation_count=2,
            bug_subtlety=1,
            execution_required=True,
            output_constraint_strength=2,
        ),
        3: dict(
            file_count=5,
            distractor_count=1,
            dependency_depth=2,
            reasoning_hops=3,
            transformation_count=3,
            bug_subtlety=2,
            execution_required=True,
            output_constraint_strength=3,
        ),
        4: dict(
            file_count=6,
            distractor_count=2,
            dependency_depth=2,
            reasoning_hops=4,
            transformation_count=4,
            bug_subtlety=3,
            execution_required=True,
            output_constraint_strength=4,
        ),
        5: dict(
            file_count=8,
            distractor_count=3,
            dependency_depth=3,
            reasoning_hops=5,
            transformation_count=5,
            bug_subtlety=4,
            execution_required=True,
            output_constraint_strength=5,
        ),
    }[difficulty]
    if family == EnvironmentFamily.TABULAR:
        base["bug_subtlety"] = 0
    elif family == EnvironmentFamily.SCRIPT_REPAIR:
        base["transformation_count"] = max(1, difficulty - 1)
    elif family == EnvironmentFamily.PIPELINE:
        base["dependency_depth"] += 1
        base["execution_required"] = True
    elif family == EnvironmentFamily.RETRIEVAL_WORKSPACE:
        base["file_count"] += 1
        base["reasoning_hops"] = max(base["reasoning_hops"], difficulty)
        base["bug_subtlety"] = 0 if difficulty <= 2 else 1
        base["execution_required"] = difficulty >= 4
    elif family == EnvironmentFamily.COMPOSITE_WORKSPACE:
        base["file_count"] += 4
        base["distractor_count"] += 1
        base["dependency_depth"] += 2
        base["reasoning_hops"] += 2
        base["transformation_count"] += 2
        base["execution_required"] = True
    return ComplexityProfile(**base)


def select_visible_hints(hints: list[str], difficulty: int) -> list[str]:
    """Reduce guidance monotonically while preserving levels 1-4."""
    if difficulty <= 2:
        return hints
    if difficulty == 3:
        return hints[:2]
    if difficulty == 4:
        return hints[:1]
    return []


_D5_COMPOSITION_PARTNERS = {
    EnvironmentFamily.TABULAR: EnvironmentFamily.RETRIEVAL_WORKSPACE,
    EnvironmentFamily.SCRIPT_REPAIR: EnvironmentFamily.RETRIEVAL_WORKSPACE,
    EnvironmentFamily.PIPELINE: EnvironmentFamily.RETRIEVAL_WORKSPACE,
    EnvironmentFamily.RETRIEVAL_WORKSPACE: EnvironmentFamily.TABULAR,
    EnvironmentFamily.COMPOSITE_WORKSPACE: EnvironmentFamily.RETRIEVAL_WORKSPACE,
}


def build_d5_composition_profile(
    family: EnvironmentFamily,
    difficulty: int,
    seed: int,
    *,
    partner: EnvironmentFamily | None = None,
    mode: str | None = None,
) -> dict[str, object]:
    """Return a deterministic D5 assignment with an optional matched-mode override."""
    family = EnvironmentFamily(family)
    if difficulty != 5:
        return {}
    if family == EnvironmentFamily.COMPOSITE_WORKSPACE:
        return {
            "composition_mode": "compositional",
            "source_families": [
                EnvironmentFamily.RETRIEVAL_WORKSPACE.value,
                EnvironmentFamily.PIPELINE.value,
            ],
            "composition_depth": 2,
        }
    selected_mode = normalize_composition_mode(mode)
    if selected_mode is None:
        selected_mode = "compositional" if seed % 2 else "hard_atomic"
    if selected_mode == "hard_atomic":
        return {
            "composition_mode": "hard_atomic",
            "source_families": [family.value],
            "composition_depth": 1,
        }
    paired_family = partner or _D5_COMPOSITION_PARTNERS[family]
    return {
        "composition_mode": "compositional",
        "source_families": [EnvironmentFamily(paired_family).value, family.value],
        "composition_depth": 2,
    }


def normalize_composition_mode(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().casefold().replace("-", "_")
    aliases = {
        "atomic": "hard_atomic",
        "hard_atomic": "hard_atomic",
        "compositional": "compositional",
        "composition": "compositional",
    }
    if normalized not in aliases:
        raise ValueError("composition_mode must be atomic, hard_atomic, or compositional")
    return aliases[normalized]


def build_difficulty_realization(
    difficulty: int,
    *,
    hint_count: int,
    candidate_file_count: int,
    **metrics: object,
) -> dict[str, object]:
    """Describe the concrete generated challenge for audits and analysis."""
    return {
        "level": difficulty,
        "guidance": "none" if hint_count == 0 else "reduced" if difficulty >= 3 else "full",
        "hint_count": hint_count,
        "candidate_file_count": candidate_file_count,
        "discovery_required": difficulty == 5,
        **metrics,
    }
