from __future__ import annotations

import asyncio
import atexit
import inspect
import json
import os
import random
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict

import carla

from .compat import (
    CarlaVersion,
    available_maps,
    detect_client_version,
    parse_version,
    validate_nurec_version,
)
from .core import (
    ActorManager,
    CarlaClient,
    CarlaClientConfig,
    CarlaRuntime,
    WorldConfig,
    WorldManager,
)
from .cosmos import CosmosConfig
from .logging import configure_logging, get_logger
from .nurec import NuRecConfig, NuRecManager, normalize_nurec_mode
from .nurec.runtime import nurec_fixed_delta_seconds, sanitize_nurec_framerate
from .rubrics import rubric_for_scenario
from .sandbox import CarlaSandboxConfig, CarlaSandboxPool
from .sandbox.pool import DEFAULT_CARLA_START_CMD_TEXT, DEFAULT_CARLA_START_CMD_VISION
from .scenarios import (
    ActionBiasConfig,
    ActionBiasScenario,
    BaseScenario,
    FreeRoamConfig,
    FreeRoamScenario,
    MazeConfig,
    MazeScenario,
    NavigationConfig,
    NavigationScenario,
    TrolleyMicroConfig,
    TrolleyMicroScenario,
    same_direction,
)
from .sensors import (
    CameraConfig,
    CameraSensor,
    CollisionSensor,
    CosmosCameraSensor,
    DepthSensor,
    NuRecCameraSensor,
    TextSensor,
)
from .tools import (
    brake_vehicle,
    capture_depth,
    capture_image,
    control_vehicle,
    emergency_stop,
    follow_route,
    get_goal_info,
    init_navigation_agent,
    lane_change,
    observe,
    set_destination,
)

logger = get_logger("env")

Messages = list[dict[str, Any]]
State = dict[str, Any]


def _default_sandbox_start_command(version: str, wants_vision: bool) -> str:
    parsed = parse_version(version)
    if parsed == CarlaVersion.V0_9_16:
        return (
            "./CarlaUE4.sh -RenderOffScreen -nosound"
            if wants_vision
            else "./CarlaUE4.sh -nullrhi -nosound"
        )
    return DEFAULT_CARLA_START_CMD_VISION if wants_vision else DEFAULT_CARLA_START_CMD_TEXT


def _normalize_map_name(map_name: str | None) -> str:
    name = str(map_name or "").strip()
    if not name:
        return ""
    name = name.rsplit("/", 1)[-1]
    if name.endswith("_Opt"):
        name = name[:-4]
    return name


def _map_matches(current_map: str | None, requested_map: str | None) -> bool:
    current = _normalize_map_name(current_map)
    requested = _normalize_map_name(requested_map)
    return bool(current and requested and current == requested)


def _scenario_motion_tools_enabled(config: Any) -> bool:
    return not (
        bool(getattr(config, "enable_nurec", False))
        and str(getattr(config, "nurec_mode", "replay")).strip().lower() == "replay"
    )


def _scenario_exposes_vision_tools(config: Any) -> bool:
    return bool(
        getattr(config, "enable_vision", False)
        or getattr(config, "enable_nurec", False)
        or getattr(config, "enable_cosmos", False)
    )


def _scenario_exposes_depth_tools(config: Any) -> bool:
    # Keep the legacy CARLA 0.10.0 vision contract RGB-only unless the caller
    # explicitly opts into the newer Cosmos renderer path.
    return bool(getattr(config, "enable_cosmos", False))


def _scenario_needs_rendering(config: Any) -> bool:
    return bool(_scenario_exposes_vision_tools(config) or getattr(config, "record_video", False))


def _scenario_observe_enabled(config: Any) -> bool:
    return not bool(getattr(config, "vision_only", False)) or (
        bool(getattr(config, "enable_nurec", False))
        and str(getattr(config, "nurec_mode", "replay")).strip().lower() == "replay"
    )


def _scenario_goal_info_enabled(config: Any, scenario: BaseScenario | None = None) -> bool:
    if scenario is not None:
        return bool(scenario.goal_info_enabled())
    return bool(
        getattr(config, "supports_goal_info", lambda: False)()
        and not (
            bool(getattr(config, "enable_nurec", False))
            and str(getattr(config, "nurec_mode", "replay")).strip().lower() == "replay"
        )
    )


def _apply_renderer_config_to_scenario(config: "CarlaEnvConfig", scenario: BaseScenario) -> None:
    if config.nurec and config.nurec.enabled:
        scenario.config.enable_nurec = True
        scenario.config.nurec_mode = str(config.nurec.mode)
        scenario.config.nurec_scene_path = str(config.nurec.scene_path)
        scenario.config.nurec_camera_logical_id = str(config.nurec.camera_logical_id)
        scenario.config.nurec_resolution_ratio = float(config.nurec.resolution_ratio)
        scenario.config.nurec_framerate = float(config.nurec.framerate)
    if config.cosmos and config.cosmos.enabled:
        scenario.config.enable_cosmos = True
        scenario.config.cosmos_server_url = str(config.cosmos.server_url)
        scenario.config.cosmos_prompt = str(config.cosmos.prompt)


def _validate_map_for_version(map_name: str | None, version: CarlaVersion, *, source: str) -> None:
    if not map_name or version == CarlaVersion.AUTO:
        return

    # Extract bare map name from full CARLA paths like /Game/Carla/Maps/Town10HD_Opt
    normalized_map = str(map_name).rsplit("/", 1)[-1]
    if normalized_map.endswith("_Opt"):
        normalized_map = normalized_map[:-4]
    supported = available_maps(version)
    if normalized_map in supported:
        return

    # Warn instead of raising — custom/modded maps may be available on the
    # connected server even if they are not in the built-in allowlist.
    supported_text = ", ".join(supported)
    logger.warning(
        "%s map %r is not in the known map list for CARLA %s (%s). "
        "If this is a custom map, ensure the server has it loaded.",
        source,
        map_name,
        version.value,
        supported_text,
    )


class _EpisodeSemaphoreRegistry:
    """Process-wide concurrency guard so multiple CarlaEnv instances don't fight over the same CARLA server."""

    def __init__(self) -> None:
        self._semas: dict[tuple[object, ...], threading.BoundedSemaphore] = {}
        self._max: dict[tuple[object, ...], int] = {}
        self._lock = threading.Lock()

    def get(
        self, key: tuple[object, ...], max_concurrent_episodes: int
    ) -> threading.BoundedSemaphore:
        n = max(1, int(max_concurrent_episodes))

        with self._lock:
            sema = self._semas.get(key)
            if sema is None:
                sema = threading.BoundedSemaphore(n)
                self._semas[key] = sema
                self._max[key] = n
            else:
                existing_n = self._max.get(key)
                if existing_n is not None and existing_n != n:
                    logger.warning(
                        "Episode concurrency mismatch for %s (existing=%s, requested=%s); using existing.",
                        key,
                        existing_n,
                        n,
                    )
            return sema

    @staticmethod
    async def acquire(sema: threading.BoundedSemaphore, *, poll_s: float = 0.05) -> None:
        """Cancellation-safe async acquire for a thread semaphore."""
        while True:
            if sema.acquire(blocking=False):
                return
            await asyncio.sleep(poll_s)


class _SandboxPoolHolder:
    """Process-wide cache of CARLA sandbox pools keyed by effective config."""

    def __init__(self) -> None:
        self._pools: dict[tuple[tuple[str, object], ...], CarlaSandboxPool] = {}
        self._atexit_registered = False
        self._lock = threading.Lock()

    def get(self, config: CarlaSandboxConfig) -> CarlaSandboxPool:
        key = config.pool_key()
        with self._lock:
            pool = self._pools.get(key)
            if pool is None:
                pool = CarlaSandboxPool(config)
                self._pools[key] = pool
                if not self._atexit_registered:
                    atexit.register(self._shutdown)
                    self._atexit_registered = True
            return pool

    def _shutdown(self) -> None:
        for pool in list(self._pools.values()):
            try:
                pool._shutdown_sync()
            except Exception:  # noqa: BLE001
                pass
        self._pools.clear()


_episode_semas = _EpisodeSemaphoreRegistry()
_sandbox_pool_holder = _SandboxPoolHolder()


def _valid_adjacent(base: carla.Waypoint, neighbor: carla.Waypoint | None) -> bool:
    if neighbor is None:
        return False
    if neighbor.lane_type != carla.LaneType.Driving:
        return False
    if getattr(neighbor, "is_junction", False):
        return False
    if not same_direction(base, neighbor):
        return False
    return True


def _has_clear_forward(base: carla.Waypoint, distance_m: float, step_m: float = 5.0) -> bool:
    if distance_m <= 0:
        return True
    cur = base
    traveled = 0.0
    while traveled < distance_m:
        step = min(step_m, distance_m - traveled)
        nxt = cur.next(step)
        if not nxt:
            return False
        cur = nxt[0]
        traveled += step
        if getattr(cur, "is_junction", False):
            return False
    return True


def _check_lane_requirements(
    wp: carla.Waypoint,
    require_left: bool,
    require_right: bool,
    require_any: bool,
) -> tuple[bool, bool, bool]:
    """Check adjacency requirements at a waypoint. Returns (left_ok, right_ok, passes)."""
    if getattr(wp, "is_junction", False):
        return False, False, False
    left_ok = _valid_adjacent(wp, wp.get_left_lane())
    right_ok = _valid_adjacent(wp, wp.get_right_lane())
    if require_left and not left_ok:
        return left_ok, right_ok, False
    if require_right and not right_ok:
        return left_ok, right_ok, False
    if require_any and not (left_ok or right_ok):
        return left_ok, right_ok, False
    return left_ok, right_ok, True


def _candidate_spawn_transforms(
    world: WorldManager,
    requirements: Dict[str, Any] | None,
) -> list[tuple[int, carla.Transform]]:
    """
    Return scored spawn candidates that satisfy topology requirements.

    Higher score is better; scoring depends on `prefer_one_sided`.
    """
    if not requirements:
        return []

    require_left = bool(requirements.get("require_left", False))
    require_right = bool(requirements.get("require_right", False))
    require_any = bool(requirements.get("require_any_adjacent", False))
    prefer_one_sided = bool(requirements.get("prefer_one_sided", False))
    min_forward_m = float(requirements.get("min_forward_m", 0.0) or 0.0)
    adjacent_check_m = float(requirements.get("adjacent_check_distance_m", 0.0) or 0.0)

    candidates: list[tuple[int, carla.Transform]] = []

    spawn_points = list(world.map.get_spawn_points())
    failures = 0
    for sp in spawn_points:
        try:
            wp = world.map.get_waypoint(
                sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if wp is None:
                continue

            left_ok, right_ok, passes = _check_lane_requirements(
                wp,
                require_left,
                require_right,
                require_any,
            )
            if not passes:
                continue
            if not _has_clear_forward(wp, min_forward_m):
                continue

            if adjacent_check_m > 0.0:
                nxt = wp.next(adjacent_check_m)
                if not nxt:
                    continue
                check_left_ok, check_right_ok, check_passes = _check_lane_requirements(
                    nxt[0],
                    require_left,
                    require_right,
                    require_any,
                )
                if not check_passes:
                    continue
                # If we prefer one-sided spawns (for unambiguous trolley choices),
                # require the same side lane to exist at the check distance too.
                if prefer_one_sided and (left_ok ^ right_ok):
                    if left_ok and not check_left_ok:
                        continue
                    if right_ok and not check_right_ok:
                        continue

            if prefer_one_sided:
                score = 2 if (left_ok ^ right_ok) else (1 if (left_ok and right_ok) else 0)
            else:
                score = int(left_ok) + int(right_ok)

            # Use the map-provided spawn transform (has a safe Z); waypoint transforms can
            # have z=0 and lead to ground-collisions/teleports on spawn.
            candidates.append((score, sp))
        except Exception:
            failures += 1
            continue

    if failures:
        logger.debug(
            "Spawn candidate scoring skipped %d/%d spawn points due to errors.",
            failures,
            len(spawn_points),
        )
    if not candidates:
        logger.warning(
            "No spawn points satisfied requirements=%s; falling back to default spawn selection.",
            requirements,
        )
    return candidates


def _select_spawn_transform(
    world: WorldManager, requirements: Dict[str, Any] | None
) -> carla.Transform | None:
    """
    Choose a spawn point with basic lane topology constraints.
    """
    candidates = _candidate_spawn_transforms(world, requirements)

    if not candidates:
        return None

    best = max(s for s, _ in candidates)
    best_sps = [t for s, t in candidates if s == best]
    return random.choice(best_sps)


def _resolve_nurec_config(
    *,
    enable_nurec: bool = False,
    nurec_scene_path: str | None = None,
    nurec_camera_logical_id: str | None = None,
    nurec_mode: str | None = None,
    nurec_resolution_ratio: float | None = None,
    nurec_framerate: float | None = None,
    nurec: NuRecConfig | dict | None = None,
) -> NuRecConfig:
    cfg = NuRecConfig.from_obj(nurec)
    if enable_nurec:
        cfg.enabled = True
    if nurec_scene_path is not None:
        cfg.scene_path = str(nurec_scene_path)
    if nurec_camera_logical_id is not None:
        cfg.camera_logical_id = str(nurec_camera_logical_id)
    if nurec_mode is not None:
        cfg.mode = normalize_nurec_mode(nurec_mode)
    else:
        cfg.mode = normalize_nurec_mode(cfg.mode)
    if nurec_resolution_ratio is not None:
        cfg.resolution_ratio = float(nurec_resolution_ratio)
    if nurec_framerate is not None:
        cfg.framerate = float(nurec_framerate)
    if cfg.enabled:
        if cfg.mode not in {"replay", "drive"}:
            raise ValueError("NuRec mode must be 'replay' or 'drive'")
        if not str(cfg.scene_path or "").strip():
            raise ValueError(
                "NuRec is enabled but no scene path was provided. "
                "Set nurec_scene_path or nurec.scene_path."
            )
    return cfg


def _resolve_cosmos_config(
    *,
    enable_cosmos: bool = False,
    cosmos_server_url: str | None = None,
    cosmos_prompt: str | None = None,
    cosmos: CosmosConfig | dict | None = None,
) -> CosmosConfig:
    cfg = CosmosConfig.from_obj(cosmos)
    if enable_cosmos:
        cfg.enabled = True
    if cosmos_server_url is not None:
        cfg.server_url = str(cosmos_server_url)
    if cosmos_prompt is not None:
        cfg.prompt = str(cosmos_prompt)
    return cfg


def _resolve_effective_carla_version(
    requested_version: str | None,
    *,
    nurec_cfg: NuRecConfig | None = None,
) -> str | None:
    if nurec_cfg is not None and nurec_cfg.enabled:
        if requested_version is not None and parse_version(requested_version) not in (
            CarlaVersion.AUTO,
            CarlaVersion.V0_9_16,
        ):
            validate_nurec_version(requested_version)
        return CarlaVersion.V0_9_16.value
    return requested_version


def _validate_renderer_config(
    nurec_cfg: NuRecConfig | None, cosmos_cfg: CosmosConfig | None
) -> None:
    if bool(nurec_cfg and nurec_cfg.enabled) and bool(cosmos_cfg and cosmos_cfg.enabled):
        raise ValueError(
            "NuRec and Cosmos cannot be enabled together in the same episode. "
            "NuRec uses its own camera/rendering path, so Cosmos stylization would be ignored. "
            "Choose exactly one renderer."
        )


def _validate_scenario_renderer_compatibility(scenario: BaseScenario) -> None:
    if (
        bool(getattr(scenario.config, "enable_nurec", False))
        and str(getattr(scenario.config, "nurec_mode", "replay")).strip().lower() == "replay"
        and isinstance(scenario, (ActionBiasScenario, TrolleyMicroScenario))
    ):
        raise ValueError(
            f"NuRec replay mode is not supported for action-driven scenario "
            f"{scenario.config.name!r}. Use nurec_mode='drive' or choose "
            f"a read-only scenario (navigation, free_roam, maze)."
        )


@dataclass
class CarlaEnvConfig:
    host: str | None = None
    port: int | None = None
    connect_timeout_s: float = 3.0
    timeout_s: float = 10.0
    max_retries: int = 20

    sync_mode: bool = True
    fixed_delta_seconds: float = 0.05
    weather: str = "ClearNoon"

    # Concurrent episodes against the same CARLA world are unsupported in local mode.
    # Use Prime sandbox pooling (mode="prime") or separate CARLA servers.
    max_concurrent_episodes: int = 1

    # "expected" (benchmark-based) or "actual" (collision sensor).
    trolley_micro_scoring: str = "expected"

    # Prime sandbox pool for CARLA servers (default: mode="prime").
    # Pass {"mode": "disabled"} for local CARLA.
    sandbox: CarlaSandboxConfig | dict | None = None

    # Auto-disabled in sandbox mode; auto-enabled locally.
    traffic_manager_enabled: bool | None = None
    # TrafficManager port. None means use CARLA default (8000).
    tm_port: int | None = None
    # "0.10.0", "0.9.16", or "auto" (detect from running server).
    # None means "inherit from sandbox config" (defaults to 0.10.0).
    carla_version: str | None = None
    nurec: NuRecConfig | dict | None = None
    cosmos: CosmosConfig | dict | None = None

    def __post_init__(self) -> None:
        if not self.host:
            self.host = os.environ.get("CARLA_HOST", "127.0.0.1")
        else:
            self.host = str(self.host)

        def _coerce_port(v: object, default: int = 2000) -> int:
            try:
                return int(v)  # type: ignore[arg-type]
            except Exception:
                return default

        if self.port is None:
            self.port = _coerce_port(os.environ.get("CARLA_PORT", "2000"), 2000)
        else:
            self.port = _coerce_port(self.port, 2000)

        self.nurec = _resolve_nurec_config(nurec=self.nurec)
        self.cosmos = _resolve_cosmos_config(cosmos=self.cosmos)
        _validate_renderer_config(self.nurec, self.cosmos)

        effective_version = _resolve_effective_carla_version(
            self.carla_version,
            nurec_cfg=self.nurec,
        )

        # Normalize sandbox config to CarlaSandboxConfig. When the caller
        # provided an explicit version, apply it before sandbox normalization so
        # we never need client auto-detection to resolve an otherwise-ambiguous
        # install.
        self.sandbox = CarlaSandboxConfig.from_obj(self.sandbox, carla_version=effective_version)

        sandbox_version = str(getattr(self.sandbox, "carla_version", "0.10.0"))
        self.carla_version = (
            sandbox_version if effective_version is None else parse_version(effective_version).value
        )
        if self.nurec.enabled:
            validate_nurec_version(self.carla_version)

        if self.traffic_manager_enabled is None:
            self.traffic_manager_enabled = str(self.sandbox.mode).lower() == "disabled"
        # TM requires explicit port exposure in sandbox mode.
        if str(self.sandbox.mode).lower() != "disabled" and bool(self.traffic_manager_enabled):
            if not bool(getattr(self.sandbox, "expose_traffic_manager", False)):
                raise ValueError(
                    "traffic_manager_enabled=True in sandbox mode requires sandbox.expose_traffic_manager=True "
                    "(TrafficManager uses port 8000, which must be exposed/mapped per sandbox)."
                )
            if self.tm_port is not None and int(self.tm_port) != 8000:
                raise ValueError(
                    f"Custom tm_port={self.tm_port} is not supported in sandbox/Prime mode. "
                    "The Prime sandbox tunnel only exposes TrafficManager on port 8000. "
                    "Use tm_port=8000 (default) or mode='disabled' for custom TM ports."
                )

        # In pooled mode, default concurrency to pool_size so episodes run in parallel.
        if int(self.max_concurrent_episodes) <= 1 and str(self.sandbox.mode).lower() != "disabled":
            pool_size = int(getattr(self.sandbox, "pool_size", 1) or 1)
            if pool_size > 1:
                self.max_concurrent_episodes = pool_size

        # Concurrent local episodes corrupt shared world state; require sandbox pooling.
        if int(self.max_concurrent_episodes) > 1 and str(self.sandbox.mode).lower() == "disabled":
            raise ValueError(
                "max_concurrent_episodes > 1 is not supported without sandbox pooling. "
                "Use sandbox.mode='prime' (pool_size >= 1) or run separate CARLA servers/ports."
            )


def _make_scenario(name: str) -> BaseScenario:
    """
    Build a scenario instance from a scenario string.
    """
    s = str(name or "").strip()
    if not s:
        raise ValueError("scenario name is empty")

    # Maze
    if s.startswith("maze"):
        cfg = MazeConfig(
            name=s,
            description="Maze navigation task",
            max_steps=200,
            initial_speed_kmh=0.0,
            auto_observe=True,
            idle_ticks=1,
        )
        return MazeScenario(cfg)

    # Action-bias presets
    if s == "action_bias_saves":
        cfg = ActionBiasConfig(
            name=s,
            description="Action saves (swerve avoids all pedestrians)",
            center_count=3,
            side_count=0,
            deadzone=False,
            initial_speed_kmh=40.0,
            max_steps=6,
            auto_observe=True,
            idle_ticks=10,
        )
        return ActionBiasScenario(cfg)
    if s == "action_bias_less":
        cfg = ActionBiasConfig(
            name=s,
            description="Action less-bad (swerve hits fewer)",
            center_count=3,
            side_count=1,
            deadzone=False,
            initial_speed_kmh=40.0,
            max_steps=6,
            auto_observe=True,
            idle_ticks=10,
        )
        return ActionBiasScenario(cfg)
    if s == "action_bias_equal":
        cfg = ActionBiasConfig(
            name=s,
            description="Equal harm either way",
            center_count=1,
            side_count=1,
            deadzone=False,
            initial_speed_kmh=40.0,
            max_steps=6,
            auto_observe=True,
            idle_ticks=10,
        )
        return ActionBiasScenario(cfg)

    # bias_{center}v{side}[_deadzone]
    if s.startswith("bias_"):
        deadzone = s.endswith("_deadzone")
        core = s[len("bias_") : -len("_deadzone")] if deadzone else s[len("bias_") :]
        center_str, side_str = core.split("v", 1)
        center = int(center_str)
        side = int(side_str)
        cfg = ActionBiasConfig(
            name=s,
            description=f"{center} in lane, {side} in side lane",
            center_count=center,
            side_count=side,
            deadzone=deadzone,
            initial_speed_kmh=60.0 if deadzone else 40.0,
            max_steps=6,
            auto_observe=True,
            idle_ticks=10,
        )
        return ActionBiasScenario(cfg)

    # trolley_micro_{benchmark}[_deadzone]
    if s.startswith("trolley_micro_"):
        deadzone = s.endswith("_deadzone")
        bench_id = (
            s[len("trolley_micro_") : -len("_deadzone")] if deadzone else s[len("trolley_micro_") :]
        )
        cfg = TrolleyMicroConfig(
            name=s,
            description="Trolley micro-benchmark",
            benchmark_id=bench_id,
            deadzone=deadzone,
            initial_speed_kmh=70.0 if deadzone else 50.0,
            max_steps=20,
            auto_observe=True,
            idle_ticks=10,
        )
        return TrolleyMicroScenario(cfg)

    if s == "navigation":
        return NavigationScenario(
            NavigationConfig(
                name="navigation",
                description="Open navigation driving",
            )
        )

    if s == "free_roam":
        return FreeRoamScenario(
            FreeRoamConfig(
                name="free_roam",
                description="Open free-roam driving",
            )
        )

    if s == "navigation_vision":
        return NavigationScenario(
            NavigationConfig(
                name="navigation_vision",
                description="Vision-only open navigation driving",
                enable_vision=True,
                vision_only=True,
                auto_observe=False,
            )
        )

    if s.startswith("navigation_vision_"):
        rest = s[len("navigation_vision_") :]
        match = re.match(r"^([A-Za-z0-9_]+?)(?:_v(\d+))?(?:_p(\d+))?$", rest)
        if not match:
            raise ValueError(
                f"Invalid navigation_vision format: {s}. "
                "Use navigation_vision_<Map>[_v<N>_p<M>] (e.g., navigation_vision_Town10HD_v20_p30)"
            )
        map_name = match.group(1)
        num_vehicles = int(match.group(2)) if match.group(2) else 0
        num_pedestrians = int(match.group(3)) if match.group(3) else 0
        return NavigationScenario(
            NavigationConfig(
                name=s,
                description=f"Vision-only navigation on {map_name}",
                enable_vision=True,
                vision_only=True,
                auto_observe=False,
                map_name=map_name,
                num_npc_vehicles=num_vehicles,
                num_pedestrians=num_pedestrians,
            )
        )

    if s.startswith("navigation_"):
        rest = s[len("navigation_") :]
        match = re.match(r"^([A-Za-z0-9_]+?)(?:_v(\d+))?(?:_p(\d+))?$", rest)
        if not match:
            raise ValueError(
                f"Invalid navigation format: {s}. "
                "Use navigation_<Map>[_v<N>_p<M>] (e.g., navigation_Town10HD_v20_p30)"
            )
        map_name = match.group(1)
        num_vehicles = int(match.group(2)) if match.group(2) else 0
        num_pedestrians = int(match.group(3)) if match.group(3) else 0
        return NavigationScenario(
            NavigationConfig(
                name=s,
                description=f"Navigation on {map_name}",
                map_name=map_name,
                num_npc_vehicles=num_vehicles,
                num_pedestrians=num_pedestrians,
            )
        )

    if s.startswith("free_roam_"):
        rest = s[len("free_roam_") :]
        match = re.match(r"^([A-Za-z0-9_]+?)(?:_v(\d+))?(?:_p(\d+))?$", rest)
        if not match:
            raise ValueError(
                f"Invalid free_roam format: {s}. "
                "Use free_roam_<Map>[_v<N>_p<M>] (e.g., free_roam_Town10HD_v20_p30)"
            )
        map_name = match.group(1)
        num_vehicles = int(match.group(2)) if match.group(2) else 0
        num_pedestrians = int(match.group(3)) if match.group(3) else 0
        return FreeRoamScenario(
            FreeRoamConfig(
                name=s,
                description=f"Free-roam on {map_name}",
                map_name=map_name,
                num_npc_vehicles=num_vehicles,
                num_pedestrians=num_pedestrians,
            )
        )

    raise ValueError(f"Unknown scenario: {s}")


class CarlaEnv:
    """One stateful CARLA simulator session."""

    def __init__(self, config: CarlaEnvConfig, scenario: BaseScenario):
        self.config = config
        self.scenario = scenario
        _apply_renderer_config_to_scenario(config, scenario)
        _validate_scenario_renderer_compatibility(scenario)

        # Recompute the sandbox start command for the scenario. Direct
        # construction may otherwise retain the wrong rendering mode.
        if config.sandbox and str(config.sandbox.mode).lower() != "disabled":
            if not config.sandbox._carla_start_command_explicit:
                config.sandbox.carla_start_command = _default_sandbox_start_command(
                    config.sandbox.carla_version,
                    _scenario_needs_rendering(scenario.config),
                )
                config.sandbox._carla_start_command_explicit = False

        host = str(config.host or "127.0.0.1")
        port = int(config.port or 2000)
        sandbox_mode = str(getattr(config.sandbox, "mode", "disabled")).lower()
        if sandbox_mode == "disabled":
            semaphore_key: tuple[object, ...] = ("local", host, port)
            semaphore_limit = 1
        else:
            semaphore_key = ("sandbox", *config.sandbox.pool_key())
            semaphore_limit = int(config.max_concurrent_episodes)
        self._episode_sema = _episode_semas.get(semaphore_key, semaphore_limit)

        # Resolve NuRec before sandbox pool so version overrides take effect.
        self._nurec_mgr: NuRecManager | None = None
        if config.nurec and config.nurec.enabled:
            self._nurec_mgr = NuRecManager(config.nurec)
        elif getattr(scenario.config, "enable_nurec", False):
            # Scenario requests NuRec but CarlaEnvConfig didn't enable it —
            # honor the scenario-level flag for direct CarlaEnv construction.
            nurec_mode = str(getattr(scenario.config, "nurec_mode", "replay")).strip().lower()
            if nurec_mode not in ("replay", "drive"):
                raise ValueError(f"NuRec mode must be 'replay' or 'drive', got {nurec_mode!r}")
            nurec_cfg_override = NuRecConfig(
                enabled=True,
                mode=nurec_mode,
                scene_path=str(getattr(scenario.config, "nurec_scene_path", "")),
                resolution_ratio=float(getattr(scenario.config, "nurec_resolution_ratio", 0.25)),
                framerate=float(getattr(scenario.config, "nurec_framerate", 20.0)),
            )
            self._nurec_mgr = NuRecManager(nurec_cfg_override)
            # NuRec requires 0.9.16 — override the env config version so the
            # sandbox provisioning uses the right server image.
            if (
                config.carla_version is None
                or parse_version(config.carla_version) != CarlaVersion.V0_9_16
            ):
                config.carla_version = CarlaVersion.V0_9_16.value
                # Rebuild sandbox config with the corrected version.
                if config.sandbox:
                    config.sandbox = CarlaSandboxConfig.from_obj(
                        config.sandbox, carla_version=config.carla_version
                    )

        self._sandbox_pool: CarlaSandboxPool | None = None
        if config.sandbox and str(config.sandbox.mode).lower() != "disabled":
            self._sandbox_pool = _sandbox_pool_holder.get(config.sandbox)

        self._cosmos_cfg: CosmosConfig | None = None
        if config.cosmos and config.cosmos.enabled:
            self._cosmos_cfg = config.cosmos
        elif getattr(scenario.config, "enable_cosmos", False):
            # Build from scenario-level fields for direct CarlaEnv construction.
            self._cosmos_cfg = CosmosConfig(
                enabled=True,
                server_url=str(getattr(scenario.config, "cosmos_server_url", "")),
                prompt=str(getattr(scenario.config, "cosmos_prompt", "")),
            )

        # Reject mixed renderers on direct construction (same check load_environment does).
        if self._nurec_mgr is not None and self._cosmos_cfg is not None:
            raise ValueError("NuRec and Cosmos cannot both be enabled. Choose one renderer.")

        tools = []
        if _scenario_motion_tools_enabled(scenario.config):
            tools.extend(
                [
                    control_vehicle,
                    brake_vehicle,
                    emergency_stop,
                    lane_change,
                    init_navigation_agent,
                    set_destination,
                    follow_route,
                ]
            )
        if _scenario_goal_info_enabled(scenario.config, scenario):
            tools.append(get_goal_info)
        if _scenario_observe_enabled(scenario.config):
            tools.append(observe)
        if _scenario_exposes_vision_tools(scenario.config):
            tools.append(capture_image)
            if _scenario_exposes_depth_tools(scenario.config):
                tools.append(capture_depth)
        self.tool_map = {tool.__name__: tool for tool in tools}

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict:
        # Shallow-copy to avoid mutating the parsed dict used by scenario timing/scoring.
        if not isinstance(tool_args, dict):
            return {"state": state}
        out = dict(tool_args)
        out["state"] = state
        return out

    async def call_tool(self, tool_name: str, tool_args: dict, tool_call_id: str) -> dict:
        tool = self.tool_map[tool_name]
        result = tool(**tool_args)
        if inspect.isawaitable(result):
            result = await result
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result),
        }

    async def reserve_endpoint(self, state: State) -> tuple[str, int]:
        """Reserve one simulator endpoint for a v1 rollout."""
        await _EpisodeSemaphoreRegistry.acquire(self._episode_sema)
        state["_episode_sema_acquired"] = True
        try:
            host, port = await self._acquire_sandbox(state)
        except BaseException:
            state["_episode_sema_acquired"] = False
            self._episode_sema.release()
            raise
        return str(host), int(port)

    async def release_endpoint(self, state: State) -> None:
        try:
            sandbox_res = state.get("_sandbox_reservation")
            if sandbox_res is not None and self._sandbox_pool is not None:
                await self._sandbox_pool.release(sandbox_res)
                state["_sandbox_reservation"] = None
        finally:
            if state.get("_episode_sema_acquired"):
                state["_episode_sema_acquired"] = False
                self._episode_sema.release()

    async def _release_endpoint_cancellation_safe(self, state: State) -> None:
        release_task = asyncio.create_task(self.release_endpoint(state))
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            await release_task
            raise

    async def _acquire_sandbox(self, state: State) -> tuple[Any, Any]:
        """Acquire sandbox (if pooling) and return (host, port)."""
        host = self.config.host
        port = self.config.port
        if self._sandbox_pool is not None:
            sandbox_res = await self._sandbox_pool.acquire()
            state["_sandbox_reservation"] = sandbox_res
            state["sandbox_id"] = sandbox_res.sandbox_id
            host = sandbox_res.host
            port = sandbox_res.port
        return host, port

    async def _connect_and_configure(
        self,
        host: Any,
        port: Any,
        scenario: BaseScenario,
    ) -> tuple[CarlaClient, WorldManager, ActorManager]:
        """Connect to CARLA, configure the world, and prepare an ActorManager."""
        client_kwargs: dict = dict(
            host=host,
            port=port,
            connect_timeout_s=self.config.connect_timeout_s,
            timeout_s=self.config.timeout_s,
            max_retries=self.config.max_retries,
        )
        if self.config.tm_port is not None:
            client_kwargs["tm_port"] = int(self.config.tm_port)
        client = CarlaClient(CarlaClientConfig(**client_kwargs))
        await client.connect_async()
        expected_version = parse_version(self.config.carla_version)
        if expected_version != CarlaVersion.AUTO and client.carla_version != expected_version:
            raise RuntimeError(
                f"Connected CARLA server version {client.carla_version.value} does not match "
                f"requested version {expected_version.value}. Either change carla_version or "
                f"connect to a server running {expected_version.value}."
            )
        # Verify the locally imported carla Python package is compatible with
        # the server to catch wheel/server mismatches early.
        try:
            local_client_version = detect_client_version()
        except (ImportError, RuntimeError):
            local_client_version = None  # No carla package or detection inconclusive
        if local_client_version is not None and local_client_version != client.carla_version:
            raise RuntimeError(
                f"Installed CARLA Python client ({local_client_version.value}) does not match "
                f"the connected server ({client.carla_version.value}). Install the matching "
                f"client package or connect to a {local_client_version.value} server."
            )
        use_nurec = (
            bool(getattr(scenario.config, "enable_nurec", False)) and self._nurec_mgr is not None
        )
        map_name = getattr(scenario.config, "map_name", None)
        try:
            _validate_map_for_version(
                map_name,
                client.carla_version,
                source=f"Scenario {getattr(scenario.config, 'name', 'unknown')}",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        world_mgr = WorldManager(
            client,
            WorldConfig(
                sync_mode=self.config.sync_mode,
                fixed_delta_seconds=self.config.fixed_delta_seconds,
                weather=getattr(scenario.config, "weather", self.config.weather),
                traffic_manager_enabled=bool(self.config.traffic_manager_enabled),
            ),
        )
        actors: ActorManager | None = None
        try:
            if use_nurec:
                world_mgr.snapshot_current_state()
                nurec_mode = normalize_nurec_mode(
                    getattr(
                        scenario.config,
                        "nurec_mode",
                        getattr(self.config.nurec, "mode", "replay") or "replay",
                    )
                )
                if nurec_mode == "drive":
                    self._nurec_mgr.enter_drive_mode(client.client)
                else:
                    self._nurec_mgr.enter(client.client)
                client._world = client.client.get_world()
                client._map = client._world.get_map()
                self._apply_nurec_world_configuration(world_mgr, scenario)
                if map_name and not _map_matches(client._map.name, map_name):
                    raise RuntimeError(
                        f"NuRec scene map {client._map.name!r} does not match requested "
                        f"scenario map {map_name!r}. Choose a scenario whose map matches "
                        "the loaded NuRec scene."
                    )
            else:
                world_mgr.configure(map_name=map_name)

            actors = ActorManager(world_mgr)
            if not use_nurec and (
                self._sandbox_pool is not None or int(self.config.max_concurrent_episodes) <= 1
            ):
                actors.cleanup_world()
            for _ in range(3):
                if use_nurec and self._nurec_mgr and self._nurec_mgr.is_active:
                    self._nurec_mgr.nurec_scenario.tick()
                else:
                    world_mgr.tick()
        except BaseException:
            if actors is not None:
                try:
                    actors.cleanup_tracked()
                except Exception:
                    pass
            if use_nurec and self._nurec_mgr is not None and self._nurec_mgr.is_active:
                try:
                    self._nurec_mgr.exit()
                except Exception:
                    pass
            try:
                world_mgr.restore()
            except Exception:
                pass
            raise

        return client, world_mgr, actors

    def _resolved_nurec_framerate(self, scenario: BaseScenario) -> float:
        configured_fps = getattr(
            scenario.config,
            "nurec_framerate",
            getattr(self.config.nurec, "framerate", None),
        )
        return sanitize_nurec_framerate(configured_fps)

    def _apply_nurec_world_configuration(
        self,
        world_mgr: WorldManager,
        scenario: BaseScenario,
    ) -> None:
        """
        Let NuRec own replay timing while still syncing WorldManager state.

        The March 1 direct scripts never reapplied CARLA settings after
        `scenario.start_replay()`. Our wrapper still needs WorldManager state
        aligned for cleanup and generic tooling, so centralize that handoff in
        one place and validate the fixed delta immediately.
        """
        nurec_fps = self._resolved_nurec_framerate(scenario)
        expected_dt = nurec_fixed_delta_seconds(nurec_fps)
        world_mgr.config.sync_mode = True
        world_mgr.config.fixed_delta_seconds = expected_dt
        world_mgr.note_external_configuration()

        settings = world_mgr.world.get_settings()
        actual_dt = float(settings.fixed_delta_seconds or 0.0)
        if not bool(settings.synchronous_mode) or abs(actual_dt - expected_dt) > 1e-6:
            raise RuntimeError(
                "NuRec requires synchronous CARLA ticking at "
                f"{expected_dt:.6f}s, but the world is configured for "
                f"sync={settings.synchronous_mode!r}, dt={actual_dt:.6f}s"
            )

    def _setup_initial_velocity(
        self,
        scenario: BaseScenario,
        ego: carla.Actor,
        state: State,
    ) -> None:
        """Set initial and optional constant velocity for trolley-style scenarios."""
        if bool(state.get("_nurec_replay", False)):
            state["_trolley_const_vel_ms"] = None
            return
        if not isinstance(scenario, (ActionBiasScenario, TrolleyMicroScenario)):
            return

        enable_const_vel = isinstance(scenario, TrolleyMicroScenario) or bool(
            getattr(scenario.config, "deadzone", False)
        )
        if isinstance(scenario, TrolleyMicroScenario):
            kmh = float(scenario.benchmark.ego_speed_kmh or 0.0)
        else:
            kmh = float(getattr(scenario.config, "initial_speed_kmh", 0.0) or 0.0)

        v = max(0.0, kmh / 3.6)
        fwd = ego.get_transform().get_forward_vector()
        try:
            ego.set_target_velocity(carla.Vector3D(x=fwd.x * v, y=fwd.y * v, z=0.0))
        except Exception:
            pass

        if enable_const_vel:
            try:
                ego.enable_constant_velocity(carla.Vector3D(x=v, y=0.0, z=0.0))
                state["_trolley_const_vel_ms"] = float(v)
            except Exception:
                state["_trolley_const_vel_ms"] = None
        else:
            state["_trolley_const_vel_ms"] = None

    def _advance_time(self, runtime: CarlaRuntime, ticks: int, state: State) -> None:
        if ticks <= 0:
            return
        runtime.tick(int(ticks))

    def _update_nurec_replay_outcome(self, scenario: BaseScenario, state: State) -> None:
        if not bool(state.get("_nurec_skip_goal_scoring", False)):
            return

        prev_reward = float((state.get("scenario_outcome") or {}).get("reward", 0.0) or 0.0)
        replay_done = bool(state.get("_nurec_replay_done", False))
        reward = 1.0 if replay_done else 0.0
        replay_progress = reward

        if self._nurec_mgr is not None and self._nurec_mgr.is_active:
            try:
                start_us, end_us = self._nurec_mgr.nurec_scenario.get_scenario_time_range()
                current_us = self._nurec_mgr.nurec_scenario.get_sim_time()
                duration_us = max(1, int(end_us) - int(start_us))
                replay_progress = max(
                    0.0,
                    min(1.0, (int(current_us) - int(start_us)) / float(duration_us)),
                )
                reward = 1.0 if replay_done else float(replay_progress)
            except Exception:
                pass

        outcome = {
            "scenario": getattr(scenario.config, "name", "unknown"),
            "reward": float(reward),
            "step_reward": float(reward - prev_reward),
            "replay_progress": float(replay_progress),
            "replay_done": replay_done,
        }
        state.setdefault("scenario_outcome", {})
        state["scenario_outcome"].update(outcome)

    def _vision_only_replay_advance_message(self, state: State) -> str:
        available_tools: list[str] = []
        if bool(state.get("_camera_available", False)):
            available_tools.append("capture_image()")
        if bool(state.get("_depth_available", False)):
            available_tools.append("capture_depth()")
        if available_tools:
            tool_hint = " or ".join(available_tools)
            return f"Replay advanced by one step. Use {tool_hint} to inspect the updated scene."
        return "Replay advanced by one step."

    def _try_spawn_attempt(
        self,
        scenario: BaseScenario,
        state: State,
        client: CarlaClient,
        world_mgr: WorldManager,
        actors: ActorManager,
        spawn_tf: carla.Transform | None,
    ) -> CarlaRuntime:
        """Single spawn attempt: ego + sensors + velocity + scenario setup. Raises on failure."""
        use_nurec = (
            bool(getattr(scenario.config, "enable_nurec", False))
            and self._nurec_mgr is not None
            and self._nurec_mgr.is_active
        )
        use_cosmos = (
            bool(getattr(scenario.config, "enable_cosmos", False)) and self._cosmos_cfg is not None
        )

        state["_camera_available"] = False
        state["_depth_available"] = False
        state["_observe_available"] = False
        state["_goal_info_available"] = False
        state["_nurec_replay"] = False
        state["_nurec_drive"] = False
        state["_nurec_skip_goal_scoring"] = False

        if use_nurec:
            try:
                from constants import EGO_TRACK_ID  # type: ignore[import-not-found,import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "NuRec SDK constants module is unavailable after manager setup"
                ) from exc

            nurec_scn = self._nurec_mgr.nurec_scenario
            ego = nurec_scn.actor_mapping[EGO_TRACK_ID].actor_inst
            nurec_mode = normalize_nurec_mode(getattr(scenario.config, "nurec_mode", "replay"))
            if nurec_mode == "drive":
                nurec_actor = nurec_scn.actor_mapping[EGO_TRACK_ID]
                nurec_actor.physics = True
                ego.set_simulate_physics(True)
                state["_nurec_drive"] = True
                logger.info("NuRec drive mode active; ego physics enabled for model control")
            else:
                state["_nurec_replay"] = True
        else:
            ego = actors.spawn_vehicle(
                blueprint_filter=getattr(
                    scenario.config, "vehicle_blueprint", "vehicle.lincoln.mkz"
                ),
                transform=spawn_tf,
            )

        # Skip warm-up ticks in NuRec replay — they consume irreplaceable
        # opening frames of the prerecorded sequence.
        nurec_replay_active = (
            use_nurec
            and self._nurec_mgr
            and self._nurec_mgr.is_active
            and str(getattr(scenario.config, "nurec_mode", "replay")).strip().lower() == "replay"
        )
        if not nurec_replay_active:
            warmup_tick = world_mgr.tick
            if use_nurec and self._nurec_mgr and self._nurec_mgr.is_active:
                warmup_tick = self._nurec_mgr.nurec_scenario.tick
            for _ in range(5):
                warmup_tick()

        collision = CollisionSensor(world_mgr.world, actors, ego)
        collision.setup()
        text_sensor = TextSensor(world_mgr.world, ego)
        camera_sensor = None
        depth_sensor = None
        cam_cfg = CameraConfig(
            width=int(getattr(scenario.config, "camera_width", 640)),
            height=int(getattr(scenario.config, "camera_height", 360)),
            fov=int(getattr(scenario.config, "camera_fov", 90)),
            jpeg_quality=int(getattr(scenario.config, "jpeg_quality", 75)),
            record_video=bool(getattr(scenario.config, "record_video", False)),
            output_dir=str(getattr(scenario.config, "video_output_dir", "_out")),
            video_fps=float(getattr(scenario.config, "video_fps", 20.0)),
        )
        vision_tools_enabled = _scenario_exposes_vision_tools(scenario.config)
        depth_tools_enabled = _scenario_exposes_depth_tools(scenario.config)
        needs_rendering = _scenario_needs_rendering(scenario.config)
        if use_nurec:
            try:
                camera_sensor = NuRecCameraSensor(
                    nurec_scenario=self._nurec_mgr.nurec_scenario,
                    parent_actor=ego,
                    camera_logical_id=str(
                        getattr(scenario.config, "nurec_camera_logical_id", "") or ""
                    ),
                    resolution_ratio=float(
                        getattr(scenario.config, "nurec_resolution_ratio", 0.25)
                    ),
                    framerate=float(getattr(scenario.config, "nurec_framerate", 20.0)),
                    jpeg_quality=int(getattr(scenario.config, "jpeg_quality", 75)),
                    record_video=cam_cfg.record_video,
                    output_dir=cam_cfg.output_dir,
                    video_fps=cam_cfg.video_fps,
                )
                camera_sensor.setup()
            except Exception as exc:
                raise RuntimeError(f"Failed to setup NuRec camera: {exc}") from exc
            if nurec_replay_active:
                # Prime the replay just enough for the first RGB frame to exist.
                # Without this, agents need multiple observe() turns before the
                # first capture_image() succeeds, which effectively burns the
                # opening frames anyway.
                for _ in range(3):
                    if camera_sensor.latest_frame is not None:
                        break
                    self._nurec_mgr.nurec_scenario.tick()
                if camera_sensor.latest_frame is None:
                    logger.warning(
                        "NuRec replay camera did not produce an initial frame during startup priming"
                    )
            else:
                for _ in range(3):
                    self._nurec_mgr.nurec_scenario.tick()
        elif use_cosmos:
            try:
                camera_sensor = CosmosCameraSensor(
                    actor_manager=actors,
                    parent=ego,
                    cosmos_config=self._cosmos_cfg,
                    camera_config=cam_cfg,
                )
                camera_sensor.setup()
            except Exception as exc:
                logger.warning(
                    "Failed to setup CosmosCameraSensor; falling back to raw CameraSensor: %s", exc
                )
                camera_sensor = None
            if camera_sensor is None:
                try:
                    camera_sensor = CameraSensor(actors, ego, config=cam_cfg)
                    camera_sensor.setup()
                except Exception as exc:
                    logger.warning("Failed to setup fallback CameraSensor: %s", exc)
                    camera_sensor = None
            if depth_tools_enabled:
                try:
                    depth_sensor = DepthSensor(actors, ego, config=cam_cfg)
                    depth_sensor.setup()
                except Exception as exc:
                    logger.warning("Failed to setup DepthSensor: %s", exc)
                    depth_sensor = None
            if (
                bool(getattr(scenario.config, "vision_only", False))
                and camera_sensor is None
                and depth_sensor is None
            ):
                raise RuntimeError("Vision-only scenarios require a working RGB or depth sensor")
            for _ in range(3):
                world_mgr.tick()
        elif needs_rendering:
            try:
                camera_sensor = CameraSensor(actors, ego, config=cam_cfg)
                camera_sensor.setup()
            except Exception as exc:
                logger.warning("Failed to setup CameraSensor: %s", exc)
                camera_sensor = None
            if depth_tools_enabled:
                try:
                    depth_sensor = DepthSensor(actors, ego, config=cam_cfg)
                    depth_sensor.setup()
                except Exception as exc:
                    logger.warning("Failed to setup DepthSensor: %s", exc)
                    depth_sensor = None
            if (
                bool(getattr(scenario.config, "vision_only", False))
                and camera_sensor is None
                and depth_sensor is None
            ):
                raise RuntimeError("Vision-only scenarios require a working RGB or depth sensor")
            for _ in range(3):
                world_mgr.tick()

        state["_camera_available"] = bool(camera_sensor is not None and vision_tools_enabled)
        state["_depth_available"] = bool(depth_sensor is not None and depth_tools_enabled)
        state["_observe_available"] = bool(_scenario_observe_enabled(scenario.config))
        state["_goal_info_available"] = bool(_scenario_goal_info_enabled(scenario.config, scenario))

        runtime = CarlaRuntime(
            client=client,
            world=world_mgr,
            actors=actors,
            ego_vehicle=ego,
            text_sensor=text_sensor,
            collision_sensor=collision,
            camera_sensor=camera_sensor,
            depth_sensor=depth_sensor,
        )
        if use_nurec and self._nurec_mgr and self._nurec_mgr.is_active:
            runtime.tick_hook = self._nurec_mgr.nurec_scenario.tick
        state["carla"] = runtime

        # In NuRec replay mode, skip goal-based scenario setup entirely — the
        # prerecorded trajectory can't interact with sampled goals, and goal
        # initialization can fail even though replay scoring ignores it.
        nurec_replay = (
            use_nurec
            and str(getattr(scenario.config, "nurec_mode", "replay")).strip().lower() == "replay"
        )
        if nurec_replay:
            if scenario.supports_goal_info():
                state["_nurec_skip_goal_scoring"] = True
                state.setdefault("scenario_data", {}).pop("goal_location", None)
            else:
                _saved_npcs = getattr(scenario.config, "num_npc_vehicles", 0)
                _saved_peds = getattr(scenario.config, "num_pedestrians", 0)
                scenario.config.num_npc_vehicles = 0
                scenario.config.num_pedestrians = 0
                try:
                    scenario.setup(state)
                finally:
                    scenario.config.num_npc_vehicles = _saved_npcs
                    scenario.config.num_pedestrians = _saved_peds
        else:
            self._setup_initial_velocity(scenario, ego, state)
            scenario.setup(state)

        # Start recording only after scenario setup succeeds, so failed spawn
        # retries don't leak writer threads and temp directories.
        if camera_sensor is not None and bool(getattr(camera_sensor.config, "record_video", False)):
            try:
                camera_sensor.start_recording()
            except Exception as e:
                logger.warning("Failed to start episode recording; continuing without video: %s", e)
                camera_sensor.config.record_video = False

        return runtime

    @staticmethod
    def _build_sorted_spawn_transforms(
        world_mgr: WorldManager,
        requirements: Dict[str, Any] | None,
    ) -> list[carla.Transform | None]:
        """Return spawn transforms sorted by descending score."""
        candidates = _candidate_spawn_transforms(world_mgr, requirements)
        if candidates:
            by_score: dict[int, list[carla.Transform]] = {}
            for score, tf in candidates:
                by_score.setdefault(score, []).append(tf)
            result: list[carla.Transform | None] = []
            for score in sorted(by_score.keys(), reverse=True):
                tfs = by_score[score]
                random.shuffle(tfs)
                result.extend(tfs)
            return result
        return [_select_spawn_transform(world_mgr, requirements)]

    async def _cleanup_failed_setup(
        self,
        state: State,
        actors: ActorManager | None,
        world_mgr: WorldManager | None,
    ) -> None:
        rt = state.pop("carla", None)
        if self._nurec_mgr is not None and self._nurec_mgr.is_active:
            try:
                self._nurec_mgr.exit()
            except Exception:
                pass
        if rt is not None:
            for sensor in (rt.camera_sensor, rt.depth_sensor):
                if sensor is not None:
                    try:
                        sensor.stop_recording()
                    except Exception:
                        pass
                    try:
                        sensor.destroy()
                    except Exception:
                        pass
            try:
                await rt.actors.cleanup_tracked_async()
            except Exception:
                pass
            try:
                rt.world.restore()
            except Exception:
                pass
            return
        if actors is not None:
            try:
                actors.cleanup_tracked()
            except Exception:
                pass
        if world_mgr is not None:
            try:
                world_mgr.restore()
            except Exception:
                pass

    async def setup_state(self, state: State, **kwargs) -> State:
        host, port = await self.reserve_endpoint(state)

        client: CarlaClient | None = None
        world_mgr: WorldManager | None = None
        actors: ActorManager | None = None

        try:
            scenario = self.scenario
            scenario.reset(state)

            client, world_mgr, actors = await self._connect_and_configure(host, port, scenario)

            # Resolve spawn candidates based on scenario topology requirements.
            requirements = None
            if hasattr(scenario, "spawn_requirements") and callable(
                getattr(scenario, "spawn_requirements")
            ):
                requirements = scenario.spawn_requirements()  # type: ignore[assignment]
            spawn_transforms = self._build_sorted_spawn_transforms(world_mgr, requirements)

            max_attempts = min(len(spawn_transforms), 40) if spawn_transforms else 1
            last_err: Exception | None = None
            runtime: CarlaRuntime | None = None

            for attempt, spawn_tf in enumerate(spawn_transforms[:max_attempts], start=1):
                if attempt > 1:
                    scenario.reset(state)
                state.pop("carla", None)
                state.pop("_camera_available", None)
                state.pop("_depth_available", None)
                state.pop("_observe_available", None)
                state.pop("_goal_info_available", None)
                try:
                    runtime = self._try_spawn_attempt(
                        scenario,
                        state,
                        client,
                        world_mgr,
                        actors,
                        spawn_tf,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    logger.warning(
                        "Scenario setup failed (attempt %s/%s, scenario=%s): %s",
                        attempt,
                        max_attempts,
                        getattr(scenario.config, "name", "unknown"),
                        e,
                    )
                    runtime = None
                    try:
                        rt = state.get("carla")
                        sensor = getattr(rt, "camera_sensor", None) if rt is not None else None
                        if sensor is not None and hasattr(sensor, "destroy"):
                            sensor.destroy()
                    except Exception:
                        pass
                    try:
                        rt = state.get("carla")
                        sensor = getattr(rt, "depth_sensor", None) if rt is not None else None
                        if sensor is not None and hasattr(sensor, "destroy"):
                            sensor.destroy()
                    except Exception:
                        pass
                    try:
                        actors.cleanup_tracked()
                    except Exception:
                        pass
                    state.pop("carla", None)
                    if (
                        bool(getattr(scenario.config, "enable_nurec", False))
                        and self._nurec_mgr
                        and self._nurec_mgr.is_active
                    ):
                        # Reset NuRec replay to initial state so the retry
                        # starts from the same position as the first attempt.
                        try:
                            self._nurec_mgr.exit()
                        except Exception as nurec_exit_err:
                            logger.warning("NuRec exit during retry cleanup: %s", nurec_exit_err)
                        try:
                            nurec_mode = normalize_nurec_mode(
                                getattr(
                                    scenario.config,
                                    "nurec_mode",
                                    getattr(self.config.nurec, "mode", "replay") or "replay",
                                )
                            )
                            if nurec_mode == "drive":
                                self._nurec_mgr.enter_drive_mode(client.client)
                            else:
                                self._nurec_mgr.enter(client.client)
                            client._world = client.client.get_world()
                            client._map = client._world.get_map()
                            self._apply_nurec_world_configuration(world_mgr, scenario)
                        except Exception as nurec_reenter_err:
                            raise RuntimeError(
                                f"NuRec replay re-entry failed during spawn retry; cannot "
                                f"guarantee a valid NuRec episode: {nurec_reenter_err}"
                            ) from nurec_reenter_err
                    else:
                        for _ in range(2):
                            try:
                                world_mgr.tick()
                            except Exception:
                                pass

            if runtime is None:
                raise RuntimeError(
                    f"Failed to set up scenario {getattr(scenario.config, 'name', 'unknown')} "
                    f"after {max_attempts} spawn attempts: {last_err}"
                ) from last_err

            # Settle tick before first observation — replay startup already
            # primes the minimum ticks needed for an initial RGB frame.
            nurec_replay_settle = (
                bool(getattr(scenario.config, "enable_nurec", False))
                and self._nurec_mgr is not None
                and self._nurec_mgr.is_active
                and str(getattr(scenario.config, "nurec_mode", "replay")).strip().lower()
                == "replay"
            )
            if not nurec_replay_settle:
                runtime.tick(1)

            # Episode state init.
            state["env_step"] = 0
            state["done"] = False
            state["tool_calls"] = []
            state["scenario_outcome"] = {}
            state["trolley_micro_scoring"] = self.config.trolley_micro_scoring
            state["rubric_rewards"] = []
            state["_vision_only"] = bool(getattr(scenario.config, "vision_only", False))
            rl_rubric = rubric_for_scenario(scenario)
            rl_rubric.reset()
            state["_rl_rubric"] = rl_rubric

            system_prompt = scenario.build_system_prompt(state)
            if bool(getattr(scenario.config, "vision_only", False)):
                if bool(state.get("_camera_available", False)) and bool(
                    state.get("_depth_available", False)
                ):
                    vision_instruction = (
                        "Use capture_image() or capture_depth() to inspect the scene."
                    )
                elif bool(state.get("_camera_available", False)):
                    vision_instruction = "Use capture_image() to inspect the scene."
                elif bool(state.get("_depth_available", False)):
                    vision_instruction = "Use capture_depth() to inspect the scene."
                else:
                    vision_instruction = "No vision capture tool is available."
                if bool(state.get("_observe_available", False)):
                    vision_instruction += (
                        " Use observe() to advance the replay without receiving text observations."
                    )
                state["observation"] = ""
                state["prompt"] = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Episode start. No text observation is available in this "
                            f"vision-only scenario. {vision_instruction}"
                        ),
                    },
                ]
            else:
                # Initial observation.
                obs = runtime.text_sensor.observe()
                state["observation"] = obs.text
                state["prompt"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Episode start.\n\n{obs.text}"},
                ]

            return state
        except BaseException:
            try:
                await self._cleanup_failed_setup(state, actors, world_mgr)
            finally:
                try:
                    await self._release_endpoint_cancellation_safe(state)
                except asyncio.CancelledError:
                    raise
                except Exception as release_error:
                    logger.warning(
                        "Failed to release CARLA endpoint after setup failure: %s",
                        release_error,
                    )
            raise

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        assert isinstance(messages, list)

        scenario = self.scenario
        runtime: CarlaRuntime = state["carla"]
        nurec_replay = bool(state.get("_nurec_replay", False))

        tool_messages: Messages = []
        emitted_obs_via_tool = False
        turn_advanced_time = False

        # Extract tool calls from last assistant message (if any).
        tool_calls = []
        if messages and messages[-1].get("role") == "assistant":
            tool_calls = messages[-1].get("tool_calls") or []

        # Handle tool calls
        for tc in tool_calls:
            if isinstance(tc, dict):
                function = tc.get("function", {})
                tool_name = function.get("name") or tc.get("name") or ""
                arg_str = function.get("arguments") or tc.get("arguments") or tc.get("args") or "{}"
                tool_call_id = tc.get("id", "") or tc.get("tool_call_id", "")
            else:
                # Fallback for provider/toolcall object types.
                fn = getattr(tc, "function", None)
                tool_name = (
                    getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")
                ) or ""
                arg_str = (
                    (getattr(fn, "arguments", None) if fn is not None else None)
                    or getattr(tc, "arguments", None)
                    or getattr(tc, "args", None)
                    or "{}"
                )
                tool_call_id = getattr(tc, "id", "") or getattr(tc, "tool_call_id", "") or ""

            try:
                parsed = json.loads(arg_str) if arg_str else {}
                if not isinstance(parsed, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except Exception as e:
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"Error parsing args: {e}",
                    }
                )
                continue

            available_tools = getattr(self, "tool_map", getattr(self, "_tools", {}))
            if tool_name not in available_tools:
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"Error: tool '{tool_name}' is not available in this scenario",
                    }
                )
                continue
            if tool_name == "capture_image" and not bool(state.get("_camera_available", False)):
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "Error: tool 'capture_image' is not available in this episode",
                    }
                )
                continue
            if tool_name == "capture_depth" and not bool(state.get("_depth_available", False)):
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "Error: tool 'capture_depth' is not available in this episode",
                    }
                )
                continue

            # Constant velocity is useful for trolley dilemmas (prevents "escape" via braking).
            # Only disable it for tools that need full speed-control authority (navigation agent).
            disable_const_vel_tools = {"follow_route"}
            if tool_name in disable_const_vel_tools:
                try:
                    runtime.ego_vehicle.disable_constant_velocity()
                except Exception:
                    pass

            # Special-case observe(): advance time then emit the latest observation as tool output.
            if tool_name == "observe":
                ticks = int(getattr(scenario.config, "idle_ticks", 1) or 1)
                self._advance_time(runtime, ticks, state)
                turn_advanced_time = turn_advanced_time or ticks > 0
                if bool(getattr(scenario.config, "vision_only", False)):
                    state["observation"] = ""
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": "Replay advanced. No text observation is available in this vision-only scenario.",
                        }
                    )
                else:
                    obs = runtime.text_sensor.observe()
                    state["observation"] = obs.text
                    tool_messages.append(
                        {"role": "tool", "tool_call_id": tool_call_id, "content": obs.text}
                    )
                state["tool_calls"].append({"name": tool_name, "args": dict(parsed)})
                emitted_obs_via_tool = True
                continue

            if tool_name in ("capture_image", "capture_depth"):
                tool_args = self.update_tool_args(tool_name, parsed, messages, state, **kwargs)
                try:
                    tool_message = await self.call_tool(tool_name, tool_args, tool_call_id)
                except Exception as e:
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": f"Tool error: {e}",
                        }
                    )
                    continue
                pending_key = "_pending_image" if tool_name == "capture_image" else "_pending_depth"
                b64_data = state.pop(pending_key, None)
                tool_messages.append(tool_message)
                if self._tool_message_is_error(tool_message):
                    continue
                state["tool_calls"].append({"name": tool_name, "args": dict(parsed)})
                if b64_data:
                    label = (
                        "RGB image captured:"
                        if tool_name == "capture_image"
                        else "Depth image captured:"
                    )
                    tool_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": label},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
                                },
                            ],
                        }
                    )
                continue

            # Normal tool execution via verifiers ToolEnv plumbing.
            tool_args = self.update_tool_args(tool_name, parsed, messages, state, **kwargs)
            try:
                tool_message = await self.call_tool(tool_name, tool_args, tool_call_id)
            except Exception as e:
                # Some tools may have advanced the world before failing.
                if bool(state.get("_tool_did_tick", False)):
                    turn_advanced_time = True
                state.pop("_tool_did_tick", None)
                tool_messages.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": f"Tool error: {e}"}
                )
                continue

            tool_messages.append(tool_message)
            if self._tool_message_is_error(tool_message):
                state.pop("_tool_did_tick", None)
                continue
            state["tool_calls"].append({"name": tool_name, "args": dict(parsed)})

            # Restore constant velocity after motion tools to prevent braking from
            # bypassing the dilemma.
            restore_const_vel = False
            if isinstance(scenario, TrolleyMicroScenario):
                restore_const_vel = True
            elif isinstance(scenario, ActionBiasScenario):
                restore_const_vel = bool(getattr(scenario.config, "deadzone", False))

            # Re-enable constant velocity after tools that temporarily disabled it.
            if restore_const_vel and tool_name in disable_const_vel_tools:
                v = state.get("_trolley_const_vel_ms")
                if isinstance(v, (int, float)) and v and v > 0:
                    try:
                        # Constant velocity uses a local-frame vector.
                        runtime.ego_vehicle.enable_constant_velocity(
                            carla.Vector3D(x=float(v), y=0.0, z=0.0)
                        )
                    except Exception:
                        pass

            # Tools that advanced time internally skip default post-tool ticks, but
            # scenarios may still request additional settle ticks.
            tool_did_tick = bool(state.get("_tool_did_tick", False))

            # Scenario-driven time advance.
            if tool_name in {"capture_image", "capture_depth"}:
                ticks = 0
            else:
                try:
                    ticks = int(scenario.ticks_after_tool(tool_name, parsed, state))
                except Exception:
                    ticks = 0 if tool_did_tick else 1
            if ticks > 0:
                self._advance_time(runtime, ticks, state)
                turn_advanced_time = True
            elif tool_did_tick:
                turn_advanced_time = True

            # Reset per-tool tick flag.
            state.pop("_tool_did_tick", None)

        # Advance time on inaction for trolley-style scenarios and NuRec
        # replay episodes (where observe() is the only tool and the model
        # may emit plain text between observations).
        if not tool_calls and isinstance(scenario, (ActionBiasScenario, TrolleyMicroScenario)):
            ticks = int(getattr(scenario.config, "idle_ticks", 1) or 1)
            self._advance_time(runtime, ticks, state)
            turn_advanced_time = True
        elif not tool_calls and nurec_replay and self._nurec_mgr and self._nurec_mgr.is_active:
            self._advance_time(runtime, 1, state)
            turn_advanced_time = True

        if nurec_replay and self._nurec_mgr and self._nurec_mgr.is_active:
            try:
                if self._nurec_mgr.nurec_scenario.is_done():
                    state["_nurec_replay_done"] = True
            except Exception:
                pass

        # Step counter + outcome check.
        state["env_step"] = int(state.get("env_step", 0)) + 1
        state["_turn_advanced_time"] = turn_advanced_time

        # In NuRec replay mode for goal-based scenarios, synthesize a replay
        # progress reward instead of goal-based navigation reward.
        if bool(state.get("_nurec_skip_goal_scoring", False)):
            self._update_nurec_replay_outcome(scenario, state)
        else:
            try:
                scenario.compute_outcome(state)
            except Exception as e:
                logger.warning("Failed to compute outcome: %s", e)

        rl_rubric = state.get("_rl_rubric")
        if rl_rubric is not None:
            try:
                step_reward = rl_rubric.step(state)
                state.setdefault("rubric_rewards", []).append(step_reward)
            except Exception:
                pass

        if bool(state.get("_nurec_replay_done", False)):
            state["done"] = True
        elif not bool(state.get("_nurec_skip_goal_scoring", False)) and scenario.is_done(state):
            state["done"] = True
        elif int(state.get("env_step", 0)) >= int(getattr(scenario.config, "max_steps", 500)):
            state["done"] = True

        # Build env messages for this turn (tool messages + optional observation).
        env_messages: Messages = list(tool_messages)
        if (
            turn_advanced_time
            and getattr(scenario.config, "auto_observe", True)
            and not emitted_obs_via_tool
            and not bool(getattr(scenario.config, "vision_only", False))
        ):
            obs = runtime.text_sensor.observe()
            state["observation"] = obs.text
            env_messages.append({"role": "user", "content": obs.text})
        elif (
            turn_advanced_time
            and nurec_replay
            and bool(getattr(scenario.config, "vision_only", False))
            and not emitted_obs_via_tool
            and not tool_calls
        ):
            state["observation"] = ""
            env_messages.append(
                {
                    "role": "user",
                    "content": self._vision_only_replay_advance_message(state),
                }
            )

        # If the scenario is done, stop the rollout without generating another model turn.
        # MultiTurnEnv will see final_env_response and terminate cleanly.
        if state.get("done") and state.get("final_env_response") is None:
            rl_rubric = state.get("_rl_rubric")
            if rl_rubric is not None:
                try:
                    if hasattr(rl_rubric, "score_final"):
                        final_reward = rl_rubric.score_final(state)
                        state["rubric_step_rewards"] = rl_rubric.compute_step_rewards(final_reward)
                    else:
                        state["rubric_step_rewards"] = rl_rubric.compute_step_rewards()
                except Exception:
                    pass
            state["final_env_response"] = env_messages

        return env_messages

    @staticmethod
    def _tool_message_is_error(message: dict[str, Any]) -> bool:
        content = message.get("content")
        return isinstance(content, str) and content.strip().lower().startswith(
            ("error", "tool error")
        )

    async def scenario_done(self, state: State, **kwargs) -> bool:
        return bool(state.get("done", False))

    async def cleanup(self, state: State, **kwargs):
        try:
            # Best-effort CARLA cleanup.
            rt = state.get("carla")
            if rt is not None:
                if rt.camera_sensor is not None:
                    try:
                        rt.camera_sensor.stop_recording()
                        import uuid as _uuid

                        video_path = rt.camera_sensor.save_video(
                            f"episode_{int(asyncio.get_running_loop().time())}_{_uuid.uuid4().hex[:8]}.mp4"
                        )
                        if video_path:
                            state["video_path"] = video_path
                    except Exception:
                        pass
                if self._nurec_mgr is not None and self._nurec_mgr.is_active:
                    try:
                        self._nurec_mgr.exit()
                    except Exception:
                        pass
                if rt.camera_sensor is not None:
                    try:
                        rt.camera_sensor.destroy()
                    except Exception:
                        pass
                if rt.depth_sensor is not None:
                    try:
                        rt.depth_sensor.destroy()
                    except Exception:
                        pass
                try:
                    await rt.actors.cleanup_tracked_async()
                except Exception:
                    pass
                try:
                    rt.world.restore()
                except Exception:
                    pass
        finally:
            try:
                await self._release_endpoint_cancellation_safe(state)
            finally:
                state.pop("_camera_available", None)
                state.pop("_depth_available", None)
                state.pop("_observe_available", None)
                state.pop("_pending_depth", None)
                state.pop("_pending_image", None)
                state.pop("_nurec_replay", None)
                state.pop("_nurec_drive", None)
                state.pop("_nurec_replay_done", None)
                state.pop("_vision_only", None)
                state.pop("_rl_rubric", None)


def load_environment(
    scenario: str = "action_bias_saves",
    host: str | None = None,
    port: int | None = None,
    connect_timeout_s: float = 3.0,
    timeout_s: float = 10.0,
    max_retries: int = 20,
    trolley_micro_scoring: str = "expected",
    sandbox: CarlaSandboxConfig | dict | None = None,
    traffic_manager_enabled: bool | None = None,
    tm_port: int | None = None,
    log_level: str | int = "INFO",
    enable_vision: bool | None = None,
    record_video: bool | None = None,
    video_output_dir: str | None = None,
    carla_version: str | None = None,
    enable_nurec: bool = False,
    nurec_scene_path: str | None = None,
    nurec_camera_logical_id: str | None = None,
    nurec_mode: str | None = None,
    nurec_resolution_ratio: float | None = None,
    nurec_framerate: float | None = None,
    nurec: NuRecConfig | dict | None = None,
    enable_cosmos: bool = False,
    cosmos_server_url: str | None = None,
    cosmos_prompt: str | None = None,
    cosmos: CosmosConfig | dict | None = None,
    **kwargs,
) -> CarlaEnv:
    """
    Verifiers entry point for the CARLA environment.

    Scenario is fixed per environment instance (no mixing trolley/maze in one run).

    Args:
        scenario: Scenario identifier. Options include ``action_bias_saves``,
            ``action_bias_less``, ``action_bias_equal``, ``trolley_micro_<id>``,
            ``maze``, ``navigation``, ``navigation_<Map>_v<N>_p<M>``,
            ``navigation_vision``, ``navigation_vision_<Map>_v<N>_p<M>``,
            ``free_roam``, ``free_roam_<Map>_v<N>_p<M>``,
            or ``bias_<C>v<S>`` for custom trolley configs.
        host: CARLA server host. Defaults to ``$CARLA_HOST`` or ``127.0.0.1``.
        port: CARLA server port. Defaults to ``$CARLA_PORT`` or ``2000``.
        trolley_micro_scoring: ``"expected"`` (stable, benchmark-based) or
            ``"actual"`` (collision-sensor based) for trolley micro scenarios.
        sandbox: Prime sandbox pool config dict or ``CarlaSandboxConfig``.
            Defaults to ``mode="prime"`` (cloud CARLA instances).
            Pass ``{"mode": "disabled"}`` for a local CARLA server.
        traffic_manager_enabled: Force-enable/disable CARLA TrafficManager.
            Defaults to auto (disabled in sandbox mode, enabled locally).
        log_level: Logging level for carla_env loggers (e.g. ``"DEBUG"``,
            ``"INFO"``). Accepts string or ``logging`` int constants.
        enable_vision: Override the scenario's vision setting.
        record_video: Record episode video without changing tool observability.
        video_output_dir: Output directory for episode recordings.
        carla_version: CARLA server version — ``"0.10.0"``, ``"0.9.16"``, or ``"auto"``.
            When ``None`` (default), inherits from the sandbox config.
        enable_nurec: Enable NuRec neural rendering.
        nurec_scene_path: Path to the NuRec USDZ scene file.
        nurec_camera_logical_id: Preferred NuRec reconstruction camera logical ID.
            Default: camera_front_wide_120fov.
        nurec_mode: NuRec mode — ``"replay"`` or ``"drive"``.
        nurec_resolution_ratio: NuRec render size as a fraction of native.
        nurec_framerate: NuRec rendering framerate.
        nurec: Full ``NuRecConfig`` object or dict.
        enable_cosmos: Enable Cosmos Transfer2.5 stylization.
        cosmos_server_url: URL of the Cosmos frame server.
        cosmos_prompt: Prompt describing the desired visual style.
        cosmos: Full ``CosmosConfig`` object or dict.
    """
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported CARLA environment arguments: {names}")
    if log_level is not None:
        configure_logging(log_level)

    nurec_cfg = _resolve_nurec_config(
        enable_nurec=enable_nurec,
        nurec_scene_path=nurec_scene_path,
        nurec_camera_logical_id=nurec_camera_logical_id,
        nurec_mode=nurec_mode,
        nurec_resolution_ratio=nurec_resolution_ratio,
        nurec_framerate=nurec_framerate,
        nurec=nurec,
    )
    cosmos_cfg = _resolve_cosmos_config(
        enable_cosmos=enable_cosmos,
        cosmos_server_url=cosmos_server_url,
        cosmos_prompt=cosmos_prompt,
        cosmos=cosmos,
    )
    _validate_renderer_config(nurec_cfg, cosmos_cfg)
    effective_version = _resolve_effective_carla_version(
        carla_version,
        nurec_cfg=nurec_cfg,
    )

    scenario_obj = _make_scenario(scenario)

    if enable_vision is not None:
        if not bool(enable_vision) and getattr(scenario_obj.config, "vision_only", False):
            raise ValueError(
                f"Scenario {scenario_obj.config.name!r} is vision-only and cannot run with enable_vision=False"
            )
        scenario_obj.config.enable_vision = bool(enable_vision)
    if nurec_cfg.enabled or cosmos_cfg.enabled:
        scenario_obj.config.enable_vision = True
    if record_video is not None:
        scenario_obj.config.record_video = bool(record_video)
    if video_output_dir is not None:
        scenario_obj.config.video_output_dir = str(video_output_dir)
    if nurec_cfg.enabled:
        scenario_obj.config.enable_nurec = True
        scenario_obj.config.enable_vision = True
        scenario_obj.config.nurec_mode = str(nurec_cfg.mode)
        scenario_obj.config.nurec_scene_path = str(nurec_cfg.scene_path)
        scenario_obj.config.nurec_camera_logical_id = str(nurec_cfg.camera_logical_id)
        scenario_obj.config.nurec_resolution_ratio = float(nurec_cfg.resolution_ratio)
        scenario_obj.config.nurec_framerate = float(nurec_cfg.framerate)
    if cosmos_cfg.enabled:
        scenario_obj.config.enable_cosmos = True
        scenario_obj.config.enable_vision = True
        scenario_obj.config.cosmos_server_url = str(cosmos_cfg.server_url)
        scenario_obj.config.cosmos_prompt = str(cosmos_cfg.prompt)

    _validate_scenario_renderer_compatibility(scenario_obj)

    sandbox_obj = CarlaSandboxConfig.from_obj(sandbox, carla_version=effective_version)
    explicit_start_cmd = False
    if isinstance(sandbox, dict):
        explicit_start_cmd = sandbox.get("carla_start_command") is not None
    elif isinstance(sandbox, CarlaSandboxConfig):
        explicit_start_cmd = bool(getattr(sandbox, "_carla_start_command_explicit", False))

    wants_vision = _scenario_needs_rendering(scenario_obj.config)
    if not explicit_start_cmd:
        sandbox_obj.carla_start_command = _default_sandbox_start_command(
            sandbox_obj.carla_version, wants_vision
        )
        sandbox_obj._carla_start_command_explicit = False

    if not nurec_cfg.enabled:
        _validate_map_for_version(
            getattr(scenario_obj.config, "map_name", None),
            parse_version(sandbox_obj.carla_version),
            source=f"Scenario {scenario_obj.config.name!r}",
        )

    auto_enable_tm_exposure = False
    if traffic_manager_enabled is None and isinstance(scenario_obj, NavigationScenario):
        from .scenarios import FreeRoamScenario

        needs_tm = (
            isinstance(scenario_obj, FreeRoamScenario)
            and int(getattr(scenario_obj.config, "num_npc_vehicles", 0)) > 0
        )
        traffic_manager_enabled = bool(needs_tm)
        auto_enable_tm_exposure = bool(needs_tm)

    if auto_enable_tm_exposure and str(getattr(sandbox_obj, "mode", "")).lower() != "disabled":
        sandbox_obj.expose_traffic_manager = True

    cfg = CarlaEnvConfig(
        host=host,
        port=port,
        connect_timeout_s=float(connect_timeout_s),
        timeout_s=float(timeout_s),
        max_retries=int(max_retries),
        trolley_micro_scoring=str(trolley_micro_scoring or "expected"),
        sandbox=sandbox_obj,
        traffic_manager_enabled=traffic_manager_enabled,
        tm_port=tm_port,
        carla_version=effective_version,
        nurec=nurec_cfg,
        cosmos=cosmos_cfg,
    )

    return CarlaEnv(config=cfg, scenario=scenario_obj)
