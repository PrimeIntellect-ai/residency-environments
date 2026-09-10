"""Stage-2 free-roam scenario — open-ended driving with no fixed goal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .navigation import NavigationConfig, NavigationScenario


@dataclass
class FreeRoamConfig(NavigationConfig):
    enable_vision: bool = True
    vision_only: bool = False
    random_goal: bool = False
    max_steps: int = 500


class FreeRoamScenario(NavigationScenario):
    """Open-ended driving scenario with no navigation goal or goal-based termination."""

    @staticmethod
    def _distinct_collision_count(runtime: Any) -> int:
        if runtime is None or getattr(runtime, "collision_sensor", None) is None:
            return 0
        events = list(getattr(runtime.collision_sensor, "events", []) or [])
        if not events:
            return 0

        dt = float(
            getattr(getattr(runtime.world, "config", None), "fixed_delta_seconds", 0.05) or 0.05
        )
        cooldown_frames = max(1, int(round(0.5 / max(dt, 0.01))))
        last_frame_by_contact: dict[tuple[object, str], int] = {}
        distinct = 0

        for event in events:
            actor_id = int(getattr(event, "other_actor_id", -1))
            actor_type = str(getattr(event, "other_actor_type", "") or "")
            frame = int(getattr(event, "frame", -1))
            contact_key = (actor_id if actor_id >= 0 else actor_type, actor_type)
            prev_frame = last_frame_by_contact.get(contact_key)

            if prev_frame is None:
                distinct += 1
            elif frame < 0 or prev_frame < 0:
                # Without reliable frame ids, keep repeated callbacks for the same
                # contact collapsed into a single penalty bucket.
                pass
            elif frame - prev_frame > cooldown_frames:
                distinct += 1

            last_frame_by_contact[contact_key] = frame

        return distinct

    def supports_goal_info(self) -> bool:
        return False

    def setup(self, state: Any) -> None:
        runtime = state["carla"]
        state.setdefault("scenario_data", {})
        state.setdefault("scenario_state", {})
        state["scenario_state"]["navigation"] = {
            "prev_goal_distance": None,
            "initial_route_distance": None,
            "best_distance_m": None,
            "collision_count": 0,
            "cumulative_reward": 0.0,
        }

        # Spawn NPC traffic but skip goal selection.
        import random

        ego_location = runtime.ego_vehicle.get_transform().location
        carla_map = runtime.world.map
        spawn_points = list(carla_map.get_spawn_points())
        available_spawns = [sp for sp in spawn_points if sp.location.distance(ego_location) > 10.0]
        random.shuffle(available_spawns)

        use_autopilot = bool(getattr(runtime.world.config, "traffic_manager_enabled", True))
        spawned_npcs = 0
        for sp in available_spawns:
            if spawned_npcs >= max(0, int(self.config.num_npc_vehicles)):
                break
            actor = runtime.actors.spawn_npc_vehicle(sp, autopilot=use_autopilot)
            if actor is not None:
                spawned_npcs += 1

        from ..logging import get_logger

        logger = get_logger("scenarios.free_roam")

        world = runtime.world.world
        spawned_pedestrians = 0
        occupied_locations = [ego_location]
        for actor in list(runtime.actors._actors):
            try:
                if str(getattr(actor, "type_id", "")).startswith(("vehicle.", "walker.")):
                    occupied_locations.append(actor.get_location())
            except Exception:
                continue
        for _ in range(self.config.num_pedestrians):
            for _attempt in range(10):
                loc = world.get_random_location_from_navigation()
                if loc is None:
                    continue
                loc.z += 0.5
                if any(loc.distance(other) < 5.0 for other in occupied_locations):
                    continue
                import carla

                actor = runtime.actors.spawn_pedestrian(carla.Transform(loc))
                if actor is not None:
                    spawned_pedestrians += 1
                    occupied_locations.append(loc)
                    break

        logger.info(
            "Free-roam actors spawned: vehicles=%d/%d pedestrians=%d/%d",
            spawned_npcs,
            self.config.num_npc_vehicles,
            spawned_pedestrians,
            self.config.num_pedestrians,
        )

        info = dict(state.get("info") or {})
        info.update(
            {
                "scenario_type": "free_roam",
                "npc_vehicles_spawned": spawned_npcs,
                "pedestrians_spawned": spawned_pedestrians,
            }
        )
        state["info"] = info

    def is_done(self, state: Any) -> bool:
        if int(state.get("env_step", 0)) >= int(self.config.max_steps):
            return True
        # End on collision to prevent crashed policies from recovering reward.
        runtime = state.get("carla")
        if runtime is not None and self._distinct_collision_count(runtime) > 0:
            return True
        return False

    def compute_outcome(self, state: Any) -> Dict[str, Any]:
        runtime = state.get("carla")
        current_collisions = self._distinct_collision_count(runtime)
        steps = int(state.get("env_step", 0))

        nav_state = state.get("scenario_state", {}).get("navigation", {})
        prev_reward = float(nav_state.get("cumulative_reward", 0.0))
        prev_collisions = int(nav_state.get("collision_count", 0))
        new_collisions = max(0, current_collisions - prev_collisions)
        nav_state["collision_count"] = current_collisions

        # Reward: survival bonus only when the sim actually advanced this turn;
        # zero-tick tools (capture_image, etc.) should not farm reward.
        # Collision penalty applies once per new collision event, not every turn.
        ticked = bool(state.get("_turn_advanced_time", False))
        step_reward = (0.01 if ticked else 0.0) + (-5.0 * new_collisions)
        cumulative_reward = prev_reward + step_reward
        nav_state["cumulative_reward"] = cumulative_reward

        outcome = {
            "scenario": self.config.name,
            "goal_reached": False,
            "collision": current_collisions > 0,
            "steps": steps,
            "reward": float(cumulative_reward),
            "step_reward": float(step_reward),
        }
        state.setdefault("scenario_outcome", {})
        state["scenario_outcome"].update(outcome)
        return outcome

    def build_system_prompt(self, state: Any) -> str:
        camera_available = bool(state.get("_camera_available", self.config.enable_vision))
        vision_note = ""
        if self.config.enable_vision and not camera_available:
            vision_note = (
                "Front camera unavailable in this episode; rely on text observations only.\n\n"
            )
        return (
            "Explore the free-roam driving environment safely.\n\n"
            "There is no fixed destination. Explore the environment, drive safely, "
            "and avoid collisions with other vehicles and pedestrians.\n\n"
            f"{vision_note}"
        )
