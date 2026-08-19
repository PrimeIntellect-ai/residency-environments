"""CARLA version compatibility layer.

Centralises version-specific behaviour so the rest of the codebase can remain
mostly version-agnostic.
"""

from __future__ import annotations

import importlib.util
from enum import Enum
from pathlib import Path
from typing import Any


class CarlaVersion(Enum):
    """Supported CARLA server versions."""

    V0_9_16 = "0.9.16"
    V0_10_0 = "0.10.0"
    AUTO = "auto"


VERSION_PRESETS: dict[CarlaVersion, dict[str, str]] = {
    CarlaVersion.V0_9_16: {
        "docker_image": "carlasim/carla:0.9.16",
        "start_command": "./CarlaUE4.sh -RenderOffScreen -nosound",
    },
    CarlaVersion.V0_10_0: {
        "docker_image": "carlasim/carla:0.10.0",
        "start_command": "./CarlaUnreal.sh -nullrhi -nosound",
    },
}


_V09_ALIASES = frozenset({"0.9", "0.9.16"})
_V010_ALIASES = frozenset({"0.10", "0.10.0"})


def parse_version(value: str) -> CarlaVersion:
    """Parse a user-supplied version string into a ``CarlaVersion``.

    Only documented aliases are accepted: ``"0.9"``/``"0.9.16"`` and
    ``"0.10"``/``"0.10.0"``.  Unsupported patch versions like ``"0.9.15"``
    are rejected.
    """

    v = str(value).strip().lower()
    if v in ("", "auto"):
        return CarlaVersion.AUTO
    if v in _V09_ALIASES:
        return CarlaVersion.V0_9_16
    if v in _V010_ALIASES:
        return CarlaVersion.V0_10_0
    raise ValueError(
        f"Unsupported CARLA version: {value!r} "
        f"(expected one of {sorted(_V09_ALIASES | _V010_ALIASES)} or 'auto')"
    )


def detect_version(server_version: str) -> CarlaVersion:
    """Detect ``CarlaVersion`` from ``client.get_server_version()`` output.

    Server version strings are dotted triples (e.g. ``"0.9.16"``).  We
    accept any ``0.9.16.x`` as V0_9_16 and any ``0.10.0.x`` as V0_10_0.
    """

    sv = str(server_version).strip()
    if sv.startswith("0.9.16"):
        return CarlaVersion.V0_9_16
    if sv.startswith("0.10.0"):
        return CarlaVersion.V0_10_0
    raise RuntimeError(
        f"Unsupported CARLA server version: {server_version!r}. "
        "Only 0.9.16 and 0.10.0 are supported."
    )


def detect_client_version() -> CarlaVersion:
    """Detect the CARLA version from the installed Python client package.

    Detection strategy (in order):

    1. Inspect the importable ``carla`` module location and map it back to the
       owning installed distribution.
    2. Check module-level version attributes on the importable ``carla`` module.
    3. Fall back to a single unambiguous installed client distribution.

    Returns ``V0_10_0`` when the ``carla`` module is not importable.
    """
    from importlib.metadata import (
        PackageNotFoundError,
        distribution,
        packages_distributions,
        version,
    )

    def _module_paths() -> list[Path]:
        spec = importlib.util.find_spec("carla")
        if spec is None:
            return []

        paths: list[Path] = []
        if spec.origin and spec.origin not in {"built-in", "frozen"}:
            paths.append(Path(spec.origin).resolve())
        if spec.submodule_search_locations:
            paths.extend(Path(loc).resolve() for loc in spec.submodule_search_locations)
        return paths

    def _distribution_version(dist_name: str) -> CarlaVersion | None:
        known = {
            "carla": CarlaVersion.V0_9_16,
            "carla-ue5-api": CarlaVersion.V0_10_0,
        }
        normalized_name = dist_name.lower()
        try:
            dist_version = version(dist_name)
        except Exception:
            return None
        if normalized_name in known:
            try:
                return parse_version(dist_version)
            except ValueError as exc:
                raise RuntimeError(
                    f"Installed CARLA client distribution {dist_name!r} has unsupported "
                    f"version {dist_version!r}. Expected {known[normalized_name].value}."
                ) from exc
        try:
            return parse_version(dist_version)
        except ValueError:
            return None

    def _distribution_matches_module(dist_name: str, module_paths: list[Path]) -> bool:
        try:
            dist = distribution(dist_name)
        except PackageNotFoundError:
            return False

        files = dist.files or []
        for file in files:
            candidate = Path(dist.locate_file(file)).resolve()
            for module_path in module_paths:
                if candidate == module_path:
                    return True
                if candidate.is_relative_to(module_path):
                    return True
                if module_path.is_relative_to(candidate):
                    return True
        return False

    module_paths = _module_paths()
    if module_paths:
        try:
            candidate_dists = packages_distributions().get("carla", [])
        except Exception:
            candidate_dists = []

        matched_versions = {
            resolved
            for dist_name in candidate_dists
            if (resolved := _distribution_version(dist_name)) is not None
            and _distribution_matches_module(dist_name, module_paths)
        }
        if len(matched_versions) == 1:
            return next(iter(matched_versions))
        if len(matched_versions) > 1:
            raise RuntimeError(
                "Multiple CARLA client distributions match the active `carla` module. "
                "Specify carla_version explicitly or remove the unused client package."
            )

    # 2. Try module-level version attributes.
    try:
        import carla as _carla

        for attr in ("__version__", "VERSION", "version"):
            version_str = getattr(_carla, attr, None)
            if version_str:
                try:
                    return parse_version(str(version_str))
                except ValueError:
                    continue
    except ImportError:
        pass

    # 3. Fall back to a single unambiguous installed client distribution.
    installed_versions = {
        resolved
        for dist_name in ("carla", "carla-ue5-api")
        if (resolved := _distribution_version(dist_name)) is not None
    }
    if len(installed_versions) == 1:
        return next(iter(installed_versions))
    if len(installed_versions) > 1:
        raise RuntimeError(
            "Multiple CARLA client distributions are installed but the active client could not be "
            "determined. Specify carla_version explicitly or remove the unused client package."
        )

    raise RuntimeError(
        "Cannot determine the installed CARLA client version. "
        "Specify carla_version explicitly (e.g. '0.10.0' or '0.9.16')."
    )


def validate_nurec_version(version: str) -> None:
    """Raise when NuRec is requested with an incompatible CARLA version."""

    parsed = parse_version(version)
    if parsed != CarlaVersion.V0_9_16:
        raise ValueError(
            "NuRec requires CARLA 0.9.16. Set carla_version='0.9.16' or "
            "enable NuRec without an explicit 0.10.0 override."
        )


def sensor_is_listening(sensor: Any) -> bool:
    """Check whether a CARLA sensor is currently listening."""

    value = getattr(sensor, "is_listening", None)
    if value is None:
        return False
    if callable(value):
        return bool(value())
    return bool(value)


def safe_stop_sensor(sensor: Any) -> None:
    """Stop a sensor safely across CARLA versions."""

    if sensor is None:
        return
    try:
        if sensor_is_listening(sensor):
            sensor.stop()
    except Exception:
        pass


_V09_MAPS = [
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town11",
    "Town12",
    "Town13",
    "Town15",
]
_V010_MAPS = ["Town10HD"]


def available_maps(version: CarlaVersion) -> list[str]:
    """Return map names available for the given CARLA version."""

    if version == CarlaVersion.V0_9_16:
        return list(_V09_MAPS)
    return list(_V010_MAPS)
