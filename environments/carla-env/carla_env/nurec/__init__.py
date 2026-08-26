"""NuRec neural rendering integration for CARLA."""

from .config import DEFAULT_NUREC_CAMERA_LOGICAL_ID, NuRecConfig, normalize_nurec_mode
from .manager import NuRecManager

__all__ = [
    "DEFAULT_NUREC_CAMERA_LOGICAL_ID",
    "NuRecConfig",
    "NuRecManager",
    "normalize_nurec_mode",
]
