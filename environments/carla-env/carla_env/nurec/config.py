"""NuRec neural rendering configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_NUREC_CAMERA_LOGICAL_ID = "camera_front_wide_120fov"


@dataclass
class NuRecConfig:
    """
    Configuration for NVIDIA NuRec neural rendering integration.

    NuRec replaces CARLA's game-engine rendering with photorealistic
    imagery via 3D Gaussian Splatting. It requires CARLA 0.9.16.
    """

    enabled: bool = False
    mode: str = "replay"
    docker_image: str = "carlasimulator/nvidia-nurec-grpc:0.2.0"
    grpc_port: int = 46435
    scene_path: str = ""
    camera_logical_id: str = DEFAULT_NUREC_CAMERA_LOGICAL_ID
    resolution_ratio: float = 0.25
    framerate: float = 20.0
    gpu_device: str = "0"
    auto_start_container: bool = True
    reuse_container: bool = True
    startup_timeout_s: float = 120.0
    sdk_path: str = ""

    @classmethod
    def from_obj(cls, obj: Any) -> "NuRecConfig":
        if obj is None:
            return cls()
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            import dataclasses

            known = {field.name for field in dataclasses.fields(cls)}
            return cls(**{k: v for k, v in obj.items() if k in known})
        raise TypeError(f"Cannot create NuRecConfig from {type(obj).__name__}")
