from __future__ import annotations

from dataclasses import dataclass

D5_PROFILE_WEIGHTS = {
    "d5_a": 0.30,
    "d5_b": 0.50,
    "d5_c": 0.20,
}

# Pipeline is only slightly hardened by shifting 10% of generated D5 tasks
# from the recoverable profile to the deepest profile. Tabular intentionally
# keeps D5_PROFILE_WEIGHTS unchanged.
PIPELINE_D5_PROFILE_WEIGHTS = {
    "d5_a": 0.20,
    "d5_b": 0.50,
    "d5_c": 0.30,
}

D5_RETRIEVAL_PROFILE_WEIGHTS = {
    "d5_a": 0.20,
    "d5_b": 0.50,
    "d5_c": 0.30,
}


@dataclass(frozen=True)
class D5Profile:
    profile_id: str
    capability_count: int
    semantic_dependency_depth: int

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile_id,
            "capability_count": self.capability_count,
            "semantic_dependency_depth": self.semantic_dependency_depth,
        }


_PROFILES = {
    "d5_a": D5Profile("d5_a", capability_count=2, semantic_dependency_depth=2),
    "d5_b": D5Profile("d5_b", capability_count=4, semantic_dependency_depth=3),
    "d5_c": D5Profile("d5_c", capability_count=6, semantic_dependency_depth=5),
}


def select_d5_profile(difficulty: int, seed: int) -> D5Profile | None:
    """Select the exact 30/50/20 profile mix over each ten-seed block."""

    if difficulty != 5:
        return None
    bucket = int(seed) % 10
    profile_id = "d5_a" if bucket < 3 else "d5_b" if bucket < 8 else "d5_c"
    return _PROFILES[profile_id]


def d5_profile_metadata(difficulty: int, seed: int) -> dict[str, object]:
    profile = select_d5_profile(difficulty, seed)
    return profile.to_dict() if profile is not None else {}


def select_weighted_d5_profile(difficulty: int, seed: int) -> D5Profile | None:
    """Select the exact 20/50/30 mix used by pipeline and retrieval."""

    if difficulty != 5:
        return None
    bucket = int(seed) % 10
    profile_id = "d5_a" if bucket < 2 else "d5_b" if bucket < 7 else "d5_c"
    return _PROFILES[profile_id]


def d5_profile_metadata_for_family(family: str, difficulty: int, seed: int) -> dict[str, object]:
    profile = (
        select_weighted_d5_profile(difficulty, seed)
        if family in {"pipeline", "retrieval_workspace"}
        else select_d5_profile(difficulty, seed)
    )
    return profile.to_dict() if profile is not None else {}
