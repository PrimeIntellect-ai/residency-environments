from __future__ import annotations

from typing import Any


def capture_image(state: Any = None) -> str:
    """Capture the current RGB camera image without advancing the simulation."""
    if state is None:
        return "Error: no state"
    runtime = state.get("carla")
    if runtime is None:
        return "Error: CARLA runtime not initialised"
    if getattr(runtime, "camera_sensor", None) is None:
        return "Error: vision not enabled for this scenario (enable_vision=False)"

    encoded = runtime.camera_sensor.capture()
    if not encoded:
        return (
            "Error: no camera frame available yet — call observe() first to advance the simulation"
        )

    state["_pending_image"] = encoded
    return f"[Image captured: {len(encoded)} bytes base64 JPEG — image will be shown below]"


def capture_depth(state: Any = None) -> str:
    """Capture the current depth view without advancing the simulation."""

    if state is None:
        return "Error: no state"
    runtime = state.get("carla")
    if runtime is None:
        return "Error: CARLA runtime not initialised"
    if getattr(runtime, "depth_sensor", None) is None:
        return "Error: depth sensor not enabled for this scenario"

    encoded = runtime.depth_sensor.capture()
    if not encoded:
        return (
            "Error: no depth frame available yet — call observe() first to advance the simulation"
        )

    state["_pending_depth"] = encoded
    return f"[Depth captured: {len(encoded)} bytes base64 JPEG — depth map will be shown below]"
