from __future__ import annotations

from typing import Any, Dict, List, Literal


class CarlaTrolleyRubric:
    """Trajectory-based rubric for trolley and action-bias scenarios."""

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self._trajectory: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._trajectory.clear()

    def step(self, state: Any) -> float:
        self._trajectory.append(dict(state.get("scenario_outcome", {})))
        return 0.0

    def score_final(self, state: Any) -> float:
        return float(state.get("scenario_outcome", {}).get("reward", 0.0))

    def compute_step_rewards(self, final_reward: float) -> List[float]:
        total_steps = len(self._trajectory)
        if total_steps == 0:
            return []
        return [final_reward * (self.gamma ** (total_steps - 1 - t)) for t in range(total_steps)]


class CarlaNavigationRubric:
    """Navigation rubric compatible with this repo's reward semantics."""

    def __init__(self, reward_mode: Literal["direct", "delta"] = "direct"):
        self.reward_mode = reward_mode
        self._previous_reward = 0.0
        self._step_rewards: List[float] = []

    def reset(self) -> None:
        self._previous_reward = 0.0
        self._step_rewards.clear()

    def step(self, state: Any) -> float:
        reward = float(state.get("scenario_outcome", {}).get("reward", 0.0))
        if self.reward_mode == "delta":
            step_reward = reward - self._previous_reward
            self._previous_reward = reward
        else:
            step_reward = reward
        self._step_rewards.append(step_reward)
        return step_reward

    def compute_step_rewards(self) -> List[float]:
        return list(self._step_rewards)


def rubric_for_scenario(scenario: Any) -> CarlaTrolleyRubric | CarlaNavigationRubric:
    from .scenarios import (
        ActionBiasScenario,
        FreeRoamScenario,
        MazeScenario,
        NavigationScenario,
        TrolleyMicroScenario,
    )

    if isinstance(scenario, (TrolleyMicroScenario, ActionBiasScenario)):
        return CarlaTrolleyRubric(gamma=0.99)
    if isinstance(scenario, (MazeScenario, NavigationScenario, FreeRoamScenario)):
        return CarlaNavigationRubric(reward_mode="delta")
    return CarlaNavigationRubric(reward_mode="direct")
