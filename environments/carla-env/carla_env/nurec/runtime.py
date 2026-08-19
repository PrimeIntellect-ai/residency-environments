"""Pure helpers for NuRec runtime timing."""

from __future__ import annotations

DEFAULT_NUREC_FRAMERATE = 20.0


def sanitize_nurec_framerate(
    fps: float | None,
    *,
    default: float = DEFAULT_NUREC_FRAMERATE,
) -> float:
    """Return a safe positive NuRec framerate."""
    try:
        resolved = float(default if fps is None else fps)
    except (TypeError, ValueError):
        resolved = float(default)
    return max(1.0, resolved)


def nurec_fixed_delta_seconds(
    fps: float | None,
    *,
    default: float = DEFAULT_NUREC_FRAMERATE,
) -> float:
    """CARLA fixed delta required by the NuRec SDK for synchronized replay."""
    return 1.0 / (sanitize_nurec_framerate(fps, default=default) * 2.0)
