from __future__ import annotations

import math
from typing import Any

import carla

from ..core.agents import import_carla_agents


def _runtime(state: Any):
    rt = state.get("carla")
    if rt is None:
        raise RuntimeError("CARLA runtime not initialized")
    return rt


def init_navigation_agent(
    behavior: str = "basic", target_speed_kmh: float | None = None, state: Any = None
) -> str:
    """
    Initialize a CARLA navigation agent and attach it to this rollout state.
    """
    if state is None:
        return "Error: no state"
    rt = _runtime(state)
    BasicAgent, BehaviorAgent = import_carla_agents()
    if BasicAgent is None:
        return "Error: CARLA agents not available (carla_env._carla_agents)"

    behavior_norm = str(behavior or "basic").lower()
    speed = float(target_speed_kmh) if target_speed_kmh is not None else None
    if speed is not None and not math.isfinite(speed):
        speed = None

    agent = None
    agent_type = "BasicAgent"
    if BehaviorAgent is not None and behavior_norm in {"cautious", "normal", "aggressive"}:
        try:
            agent = BehaviorAgent(rt.ego_vehicle, behavior=behavior_norm)
            agent_type = f"BehaviorAgent({behavior_norm})"
        except Exception:
            agent = None

    if agent is None:
        # target_speed unit is km/h in CARLA 0.10.0.
        try:
            if speed is None:
                speed = 30.0
            agent = BasicAgent(rt.ego_vehicle, target_speed=float(speed))
        except Exception as e:
            return f"Error: failed to init BasicAgent: {e}"

    state["nav_agent"] = agent
    state["nav_agent_type"] = agent_type
    return f"Navigation agent initialized: {agent_type}"


def set_destination(x: float, y: float, z: float = 0.0, state: Any = None) -> str:
    """Set the navigation agent's target destination by world coordinates."""
    if state is None:
        return "Error: no state"
    _runtime(state)  # validate CARLA runtime is initialized
    agent = state.get("nav_agent")
    if agent is None:
        init_msg = init_navigation_agent(state=state)
        if init_msg.startswith("Error"):
            return init_msg
        agent = state.get("nav_agent")

    try:
        loc = carla.Location(x=float(x), y=float(y), z=float(z))
    except Exception:
        return "Error: x/y/z must be numeric"

    try:
        agent.set_destination(loc)
    except Exception as e:
        return f"Error: set_destination failed: {e}"

    state["nav_destination"] = {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z)}
    return f"Destination set: x={loc.x:.1f} y={loc.y:.1f}"


def follow_route(steps: int = 20, state: Any = None) -> str:
    """
    Let the navigation agent drive for N simulation ticks.

    This tool advances time internally (ticks the world).
    """
    if state is None:
        return "Error: no state"
    rt = _runtime(state)
    agent = state.get("nav_agent")
    if agent is None:
        return "Error: no navigation agent (call init_navigation_agent)"

    try:
        n = int(steps)
    except Exception:
        n = 20
    n = max(1, min(500, n))

    done = False
    ticks_advanced = 0
    for _ in range(n):
        try:
            if hasattr(agent, "done") and agent.done():
                done = True
                break
            ctrl = agent.run_step()
            rt.ego_vehicle.apply_control(ctrl)
            rt.tick(1)
            ticks_advanced += 1
            state["_tool_did_tick"] = True
        except Exception as e:
            return f"Error: follow_route failed: {e}"

    state.setdefault("last_action", {})
    state["last_action"].update({"type": "follow_route", "steps": ticks_advanced})
    return "Route step complete" + (" (done)" if done else "")


# Lane change steering constants (S-curve: steer then counter-steer).
_LANE_CHANGE_STEER_PRIMARY = 0.25
_LANE_CHANGE_STEER_COUNTER = 0.12
# Throttle during lane change -- zero to maintain current speed rather than accelerate.
_LANE_CHANGE_THROTTLE = 0.0


def lane_change(direction: str, duration_s: float = 1.2, state: Any = None) -> str:
    """
    Lane change via manual steering for a fixed duration.

    Uses direct steering control independent of the CARLA agent lane-change API.
    """
    if state is None:
        return "Error: no state"
    rt = _runtime(state)

    d = str(direction or "").lower().strip()
    if d not in {"left", "right"}:
        return "Error: direction must be 'left' or 'right'"

    try:
        dur = float(duration_s)
    except Exception:
        dur = 1.2
    dur = max(0.3, min(3.0, dur))

    dt = float(getattr(rt.world.config, "fixed_delta_seconds", 0.05) or 0.05)
    ticks = max(1, int(dur / max(dt, 0.01)))

    # Use a simple S-curve (steer one way then counter-steer) to reduce off-road drift.
    steer1 = -_LANE_CHANGE_STEER_PRIMARY if d == "left" else _LANE_CHANGE_STEER_PRIMARY
    steer2 = _LANE_CHANGE_STEER_COUNTER if d == "left" else -_LANE_CHANGE_STEER_COUNTER
    t1 = max(1, ticks // 2)
    t2 = max(1, ticks - t1)

    ctrl1 = carla.VehicleControl(
        throttle=_LANE_CHANGE_THROTTLE, steer=steer1, brake=0.0, hand_brake=False
    )
    ctrl2 = carla.VehicleControl(
        throttle=_LANE_CHANGE_THROTTLE, steer=steer2, brake=0.0, hand_brake=False
    )

    for _ in range(t1):
        rt.ego_vehicle.apply_control(ctrl1)
        rt.tick(1)
        state["_tool_did_tick"] = True
    for _ in range(t2):
        rt.ego_vehicle.apply_control(ctrl2)
        rt.tick(1)
        state["_tool_did_tick"] = True

    # Neutralize steering to avoid drift.
    rt.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))

    state.setdefault("last_action", {})
    state["last_action"].update({"type": "lane_change", "direction": d, "ticks": ticks})
    return f"Lane change {d} executed ({ticks} ticks)"
