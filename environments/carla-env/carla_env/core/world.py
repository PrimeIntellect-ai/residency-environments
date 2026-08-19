from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import carla

from ..logging import get_logger
from .client import CarlaClient

logger = get_logger("core.world")


def _normalize_map_name(map_name: str) -> str:
    name = str(map_name or "").strip()
    if not name:
        return ""
    name = name.rsplit("/", 1)[-1]
    if name.endswith("_Opt"):
        name = name[:-4]
    return name


def _map_matches(current_map: str, requested_map: str) -> bool:
    current = _normalize_map_name(current_map)
    requested = _normalize_map_name(requested_map)
    if not current or not requested:
        return False
    return current == requested


def _map_load_candidates(requested_map: str) -> list[str]:
    requested = str(requested_map or "").strip()
    if not requested:
        return []
    candidates = [requested]
    if not requested.endswith("_Opt"):
        candidates.append(f"{requested}_Opt")
    return candidates


@dataclass
class WorldConfig:
    sync_mode: bool = True
    fixed_delta_seconds: float = 0.05
    weather: str = "ClearNoon"
    traffic_manager_enabled: bool = True


class WorldManager:
    """
    Applies/Restores CARLA world settings and provides a safe tick() wrapper.
    """

    MAX_CONSECUTIVE_TICK_FAILURES = 10

    def __init__(self, client: CarlaClient, config: WorldConfig):
        self.client = client
        self.config = config
        self._original_settings: Optional[carla.WorldSettings] = None
        self._original_map_name: Optional[str] = None
        self._map_changed = False
        self._configured = False
        self._tm: Optional[carla.TrafficManager] = None
        self._consecutive_tick_failures: int = 0

    @property
    def world(self) -> carla.World:
        return self.client.world

    @property
    def map(self) -> carla.Map:
        return self.client.map

    @property
    def blueprint_library(self) -> carla.BlueprintLibrary:
        return self.world.get_blueprint_library()

    def configure(self, map_name: str | None = None) -> None:
        if self._configured:
            return

        self.snapshot_current_state()

        if map_name and not _map_matches(self._original_map_name, map_name):
            logger.info("Loading map %s (current: %s)", map_name, self._original_map_name)
            last_err: Exception | None = None
            for candidate in _map_load_candidates(map_name):
                try:
                    self.client.client.load_world(candidate)
                    self.client._world = self.client.client.get_world()
                    self.client._map = self.client._world.get_map()
                    self._map_changed = True
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
            else:
                raise RuntimeError(f"Failed to load map {map_name!r}: {last_err}") from last_err

        self._apply_world_settings()
        self._configure_traffic_manager()
        self._set_weather(self.config.weather)
        self._configured = True
        logger.info(
            "World configured (sync=%s, dt=%s)",
            self.config.sync_mode,
            self.config.fixed_delta_seconds,
        )

    def snapshot_current_state(self) -> None:
        if self._original_settings is None:
            self._original_settings = self.world.get_settings()
        if self._original_map_name is None:
            self._original_map_name = self.client.map.name
        self._map_changed = False

    def note_external_configuration(self) -> None:
        self.snapshot_current_state()
        self._map_changed = bool(
            self._original_map_name is not None
            and not _map_matches(self.client.map.name, self._original_map_name)
        )
        self._apply_world_settings()
        self._configure_traffic_manager()
        self._set_weather(self.config.weather)
        self._configured = True

    def _apply_world_settings(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = bool(self.config.sync_mode)
        settings.fixed_delta_seconds = float(self.config.fixed_delta_seconds)
        self.world.apply_settings(settings)

    def _configure_traffic_manager(self) -> None:
        self._tm = None
        if self.config.sync_mode and self.config.traffic_manager_enabled:
            # TrafficManager sync is required when using agents in sync mode.
            try:
                self._tm = self.client.get_traffic_manager()
                self._tm.set_synchronous_mode(True)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to initialize TrafficManager: %s", e)
                self._tm = None
        elif self.config.sync_mode:
            logger.info("TrafficManager disabled for this run (traffic_manager_enabled=False).")

    def _set_weather(self, weather_name: str) -> None:
        try:
            weather = getattr(
                carla.WeatherParameters, weather_name, carla.WeatherParameters.ClearNoon
            )
            self.world.set_weather(weather)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to set weather %r: %s", weather_name, e)

    def tick(self) -> int:
        if not self.config.sync_mode:
            try:
                snapshot = self.world.wait_for_tick(seconds=10.0)
                self._consecutive_tick_failures = 0
                return snapshot.frame
            except Exception as e:  # noqa: BLE001
                self._consecutive_tick_failures += 1
                logger.warning(
                    "World.wait_for_tick() failed (%d consecutive): %s",
                    self._consecutive_tick_failures,
                    e,
                )
                if self._consecutive_tick_failures >= self.MAX_CONSECUTIVE_TICK_FAILURES:
                    raise RuntimeError(
                        f"World.wait_for_tick() failed {self._consecutive_tick_failures} "
                        f"consecutive times, aborting episode"
                    ) from e
                return 0
        try:
            frame = self.world.tick()
            self._consecutive_tick_failures = 0
            return frame
        except Exception as e:  # noqa: BLE001
            self._consecutive_tick_failures += 1
            logger.warning(
                "World.tick() failed (%d consecutive): %s",
                self._consecutive_tick_failures,
                e,
            )
            if self._consecutive_tick_failures >= self.MAX_CONSECUTIVE_TICK_FAILURES:
                raise RuntimeError(
                    f"World.tick() failed {self._consecutive_tick_failures} "
                    f"consecutive times, aborting episode"
                ) from e
            return 0

    def restore(self) -> None:
        if self._tm is not None and self.config.sync_mode:
            try:
                self._tm.set_synchronous_mode(False)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to disable TrafficManager sync: %s", e)

        if self._map_changed and self._original_map_name is not None:
            try:
                self.client.client.load_world(self._original_map_name)
                self.client._world = self.client.client.get_world()
                self.client._map = self.client._world.get_map()
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to restore world map: %s", e)

        if self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to restore world settings: %s", e)

        self._original_map_name = None
        self._map_changed = False
        self._configured = False
