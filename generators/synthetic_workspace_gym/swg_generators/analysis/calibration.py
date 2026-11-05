from __future__ import annotations

from collections.abc import Mapping, Sequence


def build_monotonicity_report(
    *,
    task_id: str,
    ordered_states: Sequence[tuple[str, float]],
    capability_count: int | None = None,
    semantic_dependency_depth: int | None = None,
) -> dict[str, object]:
    """Describe an ordered oracle staircase and any reward regressions."""

    if not ordered_states:
        raise ValueError("At least one calibration state is required.")
    states = {name: round(float(score), 6) for name, score in ordered_states}
    violations: list[dict[str, object]] = []
    for (before_name, before), (after_name, after) in zip(ordered_states, ordered_states[1:]):
        if float(after) + 1e-9 < float(before):
            violations.append(
                {
                    "before": before_name,
                    "before_reward": round(float(before), 6),
                    "after": after_name,
                    "after_reward": round(float(after), 6),
                }
            )
    full_name, full_reward = ordered_states[-1]
    if full_name != "full" or abs(float(full_reward) - 1.0) > 1e-9:
        violations.append(
            {
                "state": full_name,
                "expected": 1.0,
                "actual": round(float(full_reward), 6),
                "reason": "final state must be named full and score exactly 1.0",
            }
        )
    return {
        "task_id": task_id,
        "states": states,
        "monotonic": not violations,
        "violations": violations,
        "capability_count": capability_count,
        "semantic_dependency_depth": semantic_dependency_depth,
    }


def capability_pass_rates(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for name, value in dict(row.get("subscores", {})).items():
            name = str(name)
            if not name.startswith("capability_"):
                continue
            capability = name.removeprefix("capability_")
            totals[capability] = totals.get(capability, 0.0) + float(value)
            counts[capability] = counts.get(capability, 0) + 1
    return {name: round(totals[name] / counts[name], 6) for name in sorted(totals) if counts[name]}
