from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import carla

from ..decisions import (
    SIDES,
    TrolleyBenchmark,
    classify_trolley_action,
    decision_prompt,
    trolley_benchmarks,
    trolley_outcome,
)
from ..logging import get_logger
from .base import BaseScenario, ScenarioConfig
from .shared import same_direction

logger = get_logger("scenarios.trolley_micro")

BARRIER_BLUEPRINT = "static.prop.streetbarrier"
# Lateral offsets that close a lane with three 1.2 m barrier segments.
BARRIER_OFFSETS_M = (-1.2, 0.0, 1.2)


@dataclass
class TrolleyMicroConfig(ScenarioConfig):
    benchmark_id: str = "classic_3v1"
    deadzone: bool = False


class TrolleyMicroScenario(BaseScenario[TrolleyMicroConfig]):
    """
    Canonical trolley micro-benchmarks with declared lane layouts.

    Spawning pins the declared layout, so expected outcomes always describe the
    simulated scene. Both expected (benchmark) and actual (collision sensor) outcomes
    are computed for validation.
    """

    def __init__(self, config: TrolleyMicroConfig):
        super().__init__(config)
        self.benchmark: TrolleyBenchmark = self._load_benchmark()
        self.spawned_pedestrians: int = 0
        self._start_time: float = 0.0

    def _load_benchmark(self) -> TrolleyBenchmark:
        benches = trolley_benchmarks()
        if self.config.benchmark_id not in benches:
            raise ValueError(f"Unknown trolley micro benchmark: {self.config.benchmark_id}")
        return benches[self.config.benchmark_id]

    def spawn_requirements(self) -> Dict[str, Any]:
        sides = self.benchmark.sides
        return {
            "require_left": sides["left"] is not None,
            "require_right": sides["right"] is not None,
            "forbid_left": sides["left"] is None,
            "forbid_right": sides["right"] is None,
            "min_forward_m": float(self.benchmark.reaction_distance_m) + 10.0,
            "adjacent_check_distance_m": float(self.benchmark.reaction_distance_m),
        }

    def build_system_prompt(self, state: Any) -> str:
        cfg = self.config
        prompt = decision_prompt(cfg.name, "vision" if cfg.vision_only else "text")
        assert prompt is not None
        return prompt

    def reset(self, state: Any) -> None:
        self.benchmark = self._load_benchmark()
        self.spawned_pedestrians = 0
        self._start_time = time.time()
        state.setdefault("scenario_state", {})
        state["scenario_state"]["trolley_micro"] = {}

    def setup(self, state: Any) -> None:
        runtime = state["carla"]
        world = runtime.world.world
        carla_map = world.get_map()
        ego = runtime.ego_vehicle

        cfg = self.config
        bench = self.benchmark

        ego_wp = carla_map.get_waypoint(
            ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if ego_wp is None:
            raise RuntimeError("TrolleyMicroScenario: failed to resolve ego waypoint")

        lane_width = float(getattr(ego_wp, "lane_width", 3.5) or 3.5)
        if lane_width <= 0:
            lane_width = 3.5

        def shift_lane(wp: carla.Waypoint, lane_shift: int) -> Optional[carla.Waypoint]:
            cur = wp
            for _ in range(abs(lane_shift)):
                nxt = cur.get_right_lane() if lane_shift > 0 else cur.get_left_lane()
                if nxt is None or nxt.lane_type != carla.LaneType.Driving:
                    return None
                if not same_direction(wp, nxt):
                    return None
                cur = nxt
            return cur

        def waypoint_ahead(wp: carla.Waypoint, distance_m: float) -> Optional[carla.Waypoint]:
            if distance_m <= 0:
                return wp
            nxt = wp.next(distance_m)
            return nxt[0] if nxt else None

        def lane_point(wp: carla.Waypoint, lateral_m: float, forward_m: float) -> carla.Location:
            tf = wp.transform
            right = tf.get_right_vector()
            forward = tf.get_forward_vector()
            return carla.Location(
                x=tf.location.x + right.x * lateral_m + forward.x * forward_m,
                y=tf.location.y + right.y * lateral_m + forward.y * forward_m,
                z=tf.location.z + 0.5,
            )

        adjacent = {"left": shift_lane(ego_wp, -1), "right": shift_lane(ego_wp, 1)}
        for side in SIDES:
            if (adjacent[side] is None) != (bench.sides[side] is None):
                raise RuntimeError(
                    f"TrolleyMicroScenario: spawn lanes do not match benchmark {bench.id} "
                    f"({side} side should be {bench.sides[side]})"
                )

        self.spawned_pedestrians = 0

        # Spawn all pedestrians (both branches) projected onto lanes.
        ped_defs = list(bench.branch_a_pedestrians) + list(bench.branch_b_pedestrians)
        for forward_m, lateral_m, count in ped_defs:
            # Coarse lane shift based on lateral distance.
            if lateral_m > lane_width * 0.75:
                lane_shift = 1
            elif lateral_m < -lane_width * 0.75:
                lane_shift = -1
            else:
                lane_shift = 0

            lane_wp = shift_lane(ego_wp, lane_shift) if lane_shift != 0 else ego_wp
            target_wp = waypoint_ahead(lane_wp, float(forward_m)) if lane_wp else None
            if target_wp is None:
                raise RuntimeError(f"TrolleyMicroScenario: no lane for a {bench.id} pedestrian")

            residual_lateral = float(lateral_m) - (lane_shift * lane_width)
            for i in range(int(count)):
                jitter = (i - (count - 1) / 2.0) * 0.6
                loc = lane_point(target_wp, residual_lateral, jitter)
                actor = runtime.actors.spawn_pedestrian(
                    carla.Transform(loc, target_wp.transform.rotation)
                )
                # Scoring assumes the declared layout, so a missing actor rejects the spawn.
                if actor is None:
                    raise RuntimeError("TrolleyMicroScenario: failed to spawn every pedestrian")
                self.spawned_pedestrians += 1

        # Close barrier lanes across their full width at the pedestrians' distance.
        for side in SIDES:
            if bench.sides[side] != "barrier":
                continue
            barrier_wp = waypoint_ahead(adjacent[side], float(bench.reaction_distance_m))
            if barrier_wp is None:
                raise RuntimeError("TrolleyMicroScenario: no waypoint for the barrier")
            for lateral_m in BARRIER_OFFSETS_M:
                loc = lane_point(barrier_wp, lateral_m, 0.0)
                loc.z -= 0.4
                barrier = runtime.actors.spawn_prop(
                    BARRIER_BLUEPRINT, carla.Transform(loc, barrier_wp.transform.rotation)
                )
                if barrier is None:
                    raise RuntimeError("TrolleyMicroScenario: failed to spawn the barrier")

        state["_no_lane_sides"] = [side for side in SIDES if bench.sides[side] is None]

        info = state.get("info") if isinstance(state.get("info"), dict) else {}
        info = dict(info or {})
        info.update(
            {
                "scenario_type": "trolley_micro",
                "benchmark_id": bench.id,
                "benchmark_name": bench.name,
                "deadzone": bool(cfg.deadzone),
                "expected_pedestrians": int(bench.total_pedestrians),
                "spawned_pedestrians": int(self.spawned_pedestrians),
            }
        )
        state["info"] = info

    def is_done(self, state: Any) -> bool:
        # End once we observe any collision, or we hit max steps.
        runtime = state["carla"]
        if runtime.collision_sensor.collision_count > 0:
            return True
        # Deadzone variants allow a single decision turn, as in action-bias deadzones.
        if self.config.deadzone:
            return int(state.get("env_step", 0)) >= 1
        return self.step_limit_reached(state)

    def compute_outcome(self, state: Any) -> Dict[str, Any]:
        runtime = state["carla"]
        cfg = self.config
        bench = self.benchmark

        action = classify_trolley_action(state.get("tool_calls", []), bench.swerve_sides)
        outcome = trolley_outcome(bench, action)

        actual_ped = int(runtime.collision_sensor.count_unique_by_prefix("walker."))
        actual_barriers = int(runtime.collision_sensor.count_unique_by_prefix(BARRIER_BLUEPRINT))

        # Default scoring uses expected (stable), but we always compute both.
        scoring_mode = str(state.get("trolley_micro_scoring", "expected")).lower()
        if scoring_mode == "actual":
            outcome["reward"] = bench.reward(actual_ped)

        outcome.update(
            {
                "scenario": cfg.name,
                "actual_pedestrians_hit": actual_ped,
                "actual_barrier_collisions": actual_barriers,
                "scoring_mode": scoring_mode,
                "mismatch_expected_vs_actual": outcome["expected_pedestrians_hit"] != actual_ped,
                "episode_started": True,
            }
        )

        # Framing metadata for consistency scenarios.
        framing = None
        if bench.id == "consistency_a":
            framing = "workers"
        elif bench.id == "consistency_b":
            framing = "pedestrians"
        if framing is not None:
            outcome["framing"] = framing

        # Secondary metrics at state level.
        state["chose_action"] = 1.0 if outcome["chose_action"] else 0.0
        state["ethical_choice"] = outcome["ethical_choice"]
        if framing is not None:
            state["framing"] = framing

        state.setdefault("scenario_outcome", {})
        state["scenario_outcome"].update(outcome)
        return outcome

    def ticks_after_tool(self, tool_name: str, tool_args: dict, state: Any) -> int:
        # Post-action settle window for collision resolution.
        if tool_name in {"lane_change", "follow_route"}:
            return 20
        if tool_name in {"brake_vehicle", "emergency_stop", "control_vehicle"}:
            return 20
        return int(self.config.idle_ticks)
