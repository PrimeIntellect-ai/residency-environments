from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityScore:
    """One independently observable evaluator capability."""

    name: str
    value: float
    weight: float
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Capability names must not be empty.")
        if self.weight < 0:
            raise ValueError("Capability weights must be non-negative.")

    @property
    def clamped_value(self) -> float:
        return max(0.0, min(1.0, float(self.value)))


def weighted_capability_score(capabilities: list[CapabilityScore]) -> float:
    """Return a normalized weighted score without semantic floors or caps."""

    total_weight = sum(item.weight for item in capabilities)
    if total_weight <= 0:
        raise ValueError("Capability weights must sum to a positive value.")
    score = sum(item.clamped_value * item.weight for item in capabilities)
    return round(max(0.0, min(1.0, score / total_weight)), 6)


def capability_subscores(
    capabilities: list[CapabilityScore],
) -> dict[str, float]:
    return {f"capability_{item.name}": item.clamped_value for item in capabilities}


def capability_diagnostics(
    capabilities: list[CapabilityScore],
) -> dict[str, str]:
    return {item.name: item.diagnostic for item in capabilities if item.diagnostic is not None}
