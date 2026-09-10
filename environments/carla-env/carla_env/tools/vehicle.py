from __future__ import annotations

from typing import Any

import carla


def _runtime(state: Any):
    rt = state.get("carla")
    if rt is None:
        raise RuntimeError("CARLA runtime not initialized")
    return rt


def control_vehicle(throttle: float, steer: float, state: Any = None) -> str:
    """
    Manual vehicle control (single control update; env ticks separately).
    """
    if state is None:
        return "Error: no state"
    rt = _runtime(state)
    try:
        throttle_f = float(throttle)
        steer_f = float(steer)
    except Exception:
        return "Error: throttle/steer must be numbers"

    throttle_f = max(0.0, min(1.0, throttle_f))
    steer_f = max(-1.0, min(1.0, steer_f))

    ctrl = carla.VehicleControl(throttle=throttle_f, steer=steer_f, brake=0.0, hand_brake=False)
    rt.ego_vehicle.apply_control(ctrl)

    state.setdefault("last_action", {})
    state["last_action"].update(
        {"type": "control_vehicle", "throttle": throttle_f, "steer": steer_f}
    )
    return f"Applied control: throttle={throttle_f:.2f} steer={steer_f:.2f}"


def brake_vehicle(intensity: float = 1.0, state: Any = None) -> str:
    """
    Apply brakes (single update; env ticks separately).
    """
    if state is None:
        return "Error: no state"
    rt = _runtime(state)
    try:
        b = float(intensity)
    except Exception:
        b = 1.0
    b = max(0.0, min(1.0, b))

    ctrl = carla.VehicleControl(throttle=0.0, steer=0.0, brake=b, hand_brake=False)
    rt.ego_vehicle.apply_control(ctrl)
    state.setdefault("last_action", {})
    state["last_action"].update({"type": "brake_vehicle", "intensity": b})
    return f"Applied brake: intensity={b:.2f}"


def emergency_stop(state: Any = None) -> str:
    """
    Convenience: full braking.
    """
    return brake_vehicle(intensity=1.0, state=state)
