from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar


@dataclass
class ScenarioConfig:
    name: str
    description: str
    max_steps: int = 50
    weather: str = "ClearNoon"
    # CARLA docker images can ship a reduced blueprint set; mkz is usually present.
    vehicle_blueprint: str = "vehicle.lincoln.mkz"
    initial_speed_kmh: float = 0.0
    # If True, CarlaEnv will append a user observation message after each turn.
    auto_observe: bool = True
    # Default ticks to advance when the model does nothing (trolley inaction).
    idle_ticks: int = 10
    # Vision flags.
    enable_vision: bool = False
    vision_only: bool = False
    # Front RGB camera settings.
    camera_width: int = 640
    camera_height: int = 360
    camera_fov: int = 90
    jpeg_quality: int = 75
    # Optional episode recording (requires rendering, but not vision tool exposure).
    record_video: bool = False
    video_output_dir: str = "_out"
    video_fps: float = 20.0


C = TypeVar("C", bound=ScenarioConfig)


class BaseScenario(ABC, Generic[C]):
    def __init__(self, config: C):
        self.config: C = config

    def supports_goal_info(self) -> bool:
        return False

    def goal_info_enabled(self) -> bool:
        return self.supports_goal_info()

    def motion_tools_enabled(self) -> bool:
        return True

    def observe_tool_enabled(self) -> bool:
        return not bool(getattr(self.config, "vision_only", False))

    @abstractmethod
    def build_system_prompt(self, state: Any) -> str:
        pass

    @abstractmethod
    def reset(self, state: Any) -> None:
        """Reset per-episode scenario state before spawning actors."""
        pass

    @abstractmethod
    def setup(self, state: Any) -> None:
        """Spawn/initialize scenario actors. Called after ego + sensors exist."""
        pass

    @abstractmethod
    def is_done(self, state: Any) -> bool:
        pass

    @abstractmethod
    def compute_outcome(self, state: Any) -> Dict[str, Any]:
        """
        Compute a serializable outcome dict for scoring.

        Must not call CARLA APIs after cleanup; CarlaEnv will call this during env_response
        while CARLA actors are still alive.
        """
        pass

    def ticks_after_tool(self, tool_name: str, tool_args: dict, state: Any) -> int:
        """
        Scenario-specific time advancement policy.

        Note: some tools may tick the CARLA world internally (e.g. navigation agent
        driving). Those tools must set `state["_tool_did_tick"] = True` so CarlaEnv
        does not apply the default post-tool tick after the tool returns. Scenarios
        may still choose to return additional "settle" ticks even when this flag is set.
        """
        # By default: advance 1 tick after normal tools; 0 after tools that already advanced time.
        return 0 if state.get("_tool_did_tick") else 1
