from __future__ import annotations

import asyncio
import socket
import time
from dataclasses import dataclass
from typing import Optional

import carla

from ..compat import CarlaVersion, detect_version
from ..logging import get_logger

logger = get_logger("core.client")


@dataclass
class CarlaClientConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    # Fail fast on dead/unroutable hosts. This is a TCP connect-style timeout.
    connect_timeout_s: float = 3.0
    # CARLA RPC timeout (used for world/map calls once the server is reachable).
    timeout_s: float = 10.0
    max_retries: int = 20
    retry_delay_s: float = 1.0
    tm_port: int = 8000  # Traffic Manager port (CARLA default)

    def get_tm_port(self) -> int:
        """Return TM port."""
        return self.tm_port


class CarlaClient:
    """
    Small wrapper around carla.Client with retry + cached world/map.

    CARLA's PythonAPI calls are blocking; callers should use connect_async()
    in async contexts.
    """

    def __init__(self, config: CarlaClientConfig):
        self.config = config
        self._client: Optional[carla.Client] = None
        self._world: Optional[carla.World] = None
        self._map: Optional[carla.Map] = None
        self._carla_version: Optional[CarlaVersion] = None

    @property
    def client(self) -> carla.Client:
        if self._client is None:
            raise RuntimeError("CARLA client not connected")
        return self._client

    @property
    def world(self) -> carla.World:
        if self._world is None:
            raise RuntimeError("CARLA world not available (not connected?)")
        return self._world

    @property
    def map(self) -> carla.Map:
        if self._map is None:
            raise RuntimeError("CARLA map not available (not connected?)")
        return self._map

    @property
    def carla_version(self) -> CarlaVersion:
        if self._carla_version is None:
            raise RuntimeError("CARLA version not available (not connected?)")
        return self._carla_version

    def connect(self) -> None:
        cfg = self.config
        last_err: Exception | None = None
        for attempt in range(1, cfg.max_retries + 1):
            try:
                logger.info(
                    "Connecting to CARLA at %s:%s (attempt %s/%s)",
                    cfg.host,
                    cfg.port,
                    attempt,
                    cfg.max_retries,
                )

                # TCP reachability check before the full RPC handshake.
                if float(cfg.connect_timeout_s) > 0:
                    with socket.create_connection(
                        (cfg.host, int(cfg.port)), timeout=float(cfg.connect_timeout_s)
                    ):
                        pass

                client = carla.Client(cfg.host, cfg.port)
                # Use a short timeout for the initial handshake/version call.
                client.set_timeout(
                    float(cfg.connect_timeout_s)
                    if float(cfg.connect_timeout_s) > 0
                    else float(cfg.timeout_s)
                )
                server_version = client.get_server_version()

                # Use the regular RPC timeout for heavier calls (world/map).
                client.set_timeout(float(cfg.timeout_s))
                world = client.get_world()
                carla_map = world.get_map()

                self._client = client
                self._world = world
                self._map = carla_map
                self._carla_version = detect_version(server_version)
                logger.info(
                    "Connected to CARLA (server=%s, detected=%s, map=%s)",
                    server_version,
                    self._carla_version.value,
                    carla_map.name,
                )
                return
            except Exception as e:  # noqa: BLE001 - external API
                last_err = e
                self._client = None
                self._world = None
                self._map = None
                self._carla_version = None
                if attempt < cfg.max_retries:
                    time.sleep(cfg.retry_delay_s)

        raise RuntimeError(f"Failed to connect to CARLA at {cfg.host}:{cfg.port}: {last_err}")

    async def connect_async(self) -> None:
        await asyncio.to_thread(self.connect)

    def get_traffic_manager(self, port: int | None = None) -> carla.TrafficManager:
        tm_port = port if port is not None else self.config.get_tm_port()
        return self.client.get_trafficmanager(tm_port)
