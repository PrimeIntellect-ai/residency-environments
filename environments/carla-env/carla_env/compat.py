"""CARLA 0.10.0 compatibility helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any


class CarlaVersion(Enum):
    V0_10_0 = "0.10.0"


def detect_version(server_version: str) -> CarlaVersion:
    version = str(server_version).strip()
    if version.startswith("0.10.0"):
        return CarlaVersion.V0_10_0
    raise RuntimeError(f"CARLA 0.10.0 is required, got server version {server_version!r}")


def safe_stop_sensor(sensor: Any) -> None:
    if sensor is None:
        return
    try:
        if getattr(sensor, "is_listening", False):
            sensor.stop()
    except (RuntimeError, AttributeError):
        pass


def available_maps() -> list[str]:
    return ["Town10HD"]
