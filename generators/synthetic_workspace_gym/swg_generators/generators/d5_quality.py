from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from swg_generators.generators.difficulty_primitives import CompositionSpec


@dataclass(frozen=True)
class AtomicQualityThresholds:
    unmodified_max: float = 0.15
    single_fix_max: float = 0.40
    pair_fix_max: float = 0.65
    all_but_one_max: float = 0.85
    complete_score: float = 1.0
    required_capability_count: int = 4
    touched_file_count: int = 2
    semantic_dependency_depth: int = 3


@dataclass(frozen=True)
class CompositionQualityThresholds:
    stage_a_only_max: float = 0.40
    stage_b_only_max: float = 0.30
    stage_a_b_partial_max: float = 0.65
    complete_score: float = 1.0
    coupling_margin_min: float = 0.50
    stage_count: int = 2


def normalize_lattice_profile(profile: Mapping[str, object]) -> dict[str, object]:
    """Translate persisted exhaustive-lattice metrics into the shared oracle schema."""
    if not profile:
        return {}
    return {
        "unmodified_reward": float(profile["no_fix_score"]),
        "single_fix_max_reward": float(profile["single_fix_max_score"]),
        "pair_fix_max_reward": float(profile["pair_fix_max_score"]),
        "all_but_one_reward": float(profile["all_but_one_max_score"]),
        "reference_solution_reward": float(profile["full_solution_score"]),
    }


def summarize_atomic_oracle(
    scores: Mapping[frozenset[str], float],
    defect_ids: Sequence[str],
) -> dict[str, object]:
    ordered = tuple(dict.fromkeys(str(item) for item in defect_ids))
    expected = frozenset(ordered)
    missing = [sorted(subset) for subset in _required_atomic_subsets(ordered) if subset not in scores]
    if missing:
        raise ValueError(f"missing oracle scores for subsets: {missing}")
    by_size: dict[int, list[float]] = {}
    for subset, score in scores.items():
        by_size.setdefault(len(subset), []).append(float(score))
    return {
        "unmodified_reward": round(float(scores[frozenset()]), 6),
        "single_fix_max_reward": round(max(by_size.get(1, [0.0])), 6),
        "pair_fix_max_reward": round(max(by_size.get(2, [0.0])), 6),
        "all_but_one_reward": round(
            max(by_size.get(max(len(ordered) - 1, 0), [0.0])),
            6,
        ),
        "reference_solution_reward": round(float(scores[expected]), 6),
        "defect_count": len(ordered),
    }


def validate_atomic_oracle(
    profile: Mapping[str, object],
    *,
    thresholds: AtomicQualityThresholds | None = None,
) -> dict[str, object]:
    limits = thresholds or AtomicQualityThresholds()
    violations: list[str] = []
    _maximum(
        profile,
        "unmodified_reward",
        limits.unmodified_max,
        violations,
    )
    _maximum(
        profile,
        "single_fix_max_reward",
        limits.single_fix_max,
        violations,
    )
    _maximum(
        profile,
        "pair_fix_max_reward",
        limits.pair_fix_max,
        violations,
    )
    _maximum(
        profile,
        "all_but_one_reward",
        limits.all_but_one_max,
        violations,
    )
    _exact(
        profile,
        "reference_solution_reward",
        limits.complete_score,
        violations,
    )
    return _finish(profile, violations)


def summarize_composition_oracle(
    *,
    unmodified_reward: float,
    stage_a_only_reward: float,
    stage_b_only_reward: float,
    stage_a_b_partial_reward: float,
    reference_solution_reward: float,
) -> dict[str, object]:
    return {
        "unmodified_reward": round(float(unmodified_reward), 6),
        "stage_a_only_reward": round(float(stage_a_only_reward), 6),
        "stage_b_only_reward": round(float(stage_b_only_reward), 6),
        "stage_a_b_partial_reward": round(float(stage_a_b_partial_reward), 6),
        "reference_solution_reward": round(float(reference_solution_reward), 6),
        "coupling_margin": round(
            float(reference_solution_reward) - max(float(stage_a_only_reward), float(stage_b_only_reward)),
            6,
        ),
    }


def validate_composition_oracle(
    profile: Mapping[str, object],
    composition_spec: CompositionSpec,
    *,
    thresholds: CompositionQualityThresholds | None = None,
) -> dict[str, object]:
    limits = thresholds or CompositionQualityThresholds()
    violations: list[str] = []
    _maximum(
        profile,
        "stage_a_only_reward",
        limits.stage_a_only_max,
        violations,
    )
    _maximum(
        profile,
        "stage_b_only_reward",
        limits.stage_b_only_max,
        violations,
    )
    _maximum(
        profile,
        "stage_a_b_partial_reward",
        limits.stage_a_b_partial_max,
        violations,
    )
    _exact(
        profile,
        "reference_solution_reward",
        limits.complete_score,
        violations,
    )
    if float(profile.get("coupling_margin", -1.0)) < limits.coupling_margin_min:
        violations.append("coupling_margin is below the required minimum")
    if len(composition_spec.stages) < limits.stage_count:
        violations.append("composition has too few stages")
    if not composition_spec.downstream_consumes_upstream_artifact:
        violations.append("downstream stage does not consume an upstream artifact")
    return _finish(
        {
            **dict(profile),
            **composition_spec.to_dict(),
        },
        violations,
    )


def validate_d5_realization(
    realization: Mapping[str, object],
    *,
    atomic_oracle: Mapping[str, object],
    composition_oracle: Mapping[str, object] | None = None,
    composition_spec: CompositionSpec | None = None,
    thresholds: AtomicQualityThresholds | None = None,
) -> dict[str, object]:
    limits = thresholds or AtomicQualityThresholds()
    violations: list[str] = []
    atomic = validate_atomic_oracle(atomic_oracle, thresholds=limits)
    violations.extend(str(item) for item in atomic["violations"])
    if int(realization.get("capability_count", 0)) < limits.required_capability_count:
        violations.append("required_capability_count is below the minimum")
    if int(realization.get("touched_file_count", 0)) < limits.touched_file_count:
        violations.append("touched_file_count is below the minimum")
    if int(realization.get("semantic_dependency_depth", 0)) < limits.semantic_dependency_depth:
        violations.append("semantic_dependency_depth is below the minimum")
    composition: dict[str, object] | None = None
    if composition_spec is not None:
        if composition_oracle is None:
            violations.append("compositional task is missing an oracle profile")
        else:
            composition = validate_composition_oracle(
                composition_oracle,
                composition_spec,
            )
            violations.extend(str(item) for item in composition["violations"])
    return _finish(
        {
            "atomic_oracle": atomic,
            "composition_oracle": composition,
            "structural_metrics": dict(realization),
        },
        violations,
    )


def _required_atomic_subsets(
    defect_ids: Sequence[str],
) -> set[frozenset[str]]:
    ordered = tuple(defect_ids)
    subsets = {frozenset(), frozenset(ordered)}
    subsets.update(frozenset((item,)) for item in ordered)
    subsets.update(
        frozenset((ordered[left], ordered[right]))
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
    )
    subsets.update(frozenset(item for item in ordered if item != omitted) for omitted in ordered)
    return subsets


def _maximum(
    profile: Mapping[str, object],
    key: str,
    maximum: float,
    violations: list[str],
) -> None:
    if float(profile.get(key, float("inf"))) > maximum:
        violations.append(f"{key} exceeds {maximum:.2f}")


def _exact(
    profile: Mapping[str, object],
    key: str,
    expected: float,
    violations: list[str],
) -> None:
    if abs(float(profile.get(key, float("nan"))) - expected) > 1e-9:
        violations.append(f"{key} must equal {expected:.2f}")


def _finish(
    profile: Mapping[str, object],
    violations: Sequence[str],
) -> dict[str, object]:
    result = {
        **dict(profile),
        "valid": not violations,
        "violations": list(dict.fromkeys(str(item) for item in violations)),
    }
    return result
