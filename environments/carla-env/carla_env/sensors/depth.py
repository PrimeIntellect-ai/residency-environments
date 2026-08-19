from __future__ import annotations

import base64
import io
import threading
from typing import Any, Optional

import numpy as np

from ..compat import safe_stop_sensor
from ..logging import get_logger
from .camera import CameraConfig

logger = get_logger("sensors.depth")


class DepthSensor:
    """CARLA depth camera co-located with the RGB camera."""

    def __init__(
        self,
        actor_manager: Any,
        parent: Any,
        config: CameraConfig | None = None,
    ):
        self._actor_manager = actor_manager
        self._parent = parent
        self._config = config or CameraConfig()
        self._sensor: Optional[Any] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def setup(self) -> None:
        import carla

        cfg = self._config
        transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        sensor = self._actor_manager.spawn_sensor(
            "sensor.camera.depth",
            transform,
            attach_to=self._parent,
            attributes={
                "image_size_x": str(cfg.width),
                "image_size_y": str(cfg.height),
                "fov": str(cfg.fov),
            },
        )
        self._sensor = sensor

        def _on_depth(image: Any) -> None:
            raw = np.frombuffer(image.raw_data, dtype=np.uint8)
            raw = raw.reshape((image.height, image.width, 4))
            r = raw[:, :, 2].astype(np.float32)
            g = raw[:, :, 1].astype(np.float32)
            b = raw[:, :, 0].astype(np.float32)
            depth = (r + g * 256.0 + b * 65536.0) / (256.0**3 - 1.0) * 1000.0
            with self._lock:
                self._latest_depth = depth

        sensor.listen(_on_depth)
        logger.info("DepthSensor attached (%dx%d)", cfg.width, cfg.height)

    @property
    def latest_depth(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_depth is None:
                return None
            return self._latest_depth.copy()

    def capture(self) -> str:
        with self._lock:
            depth = None if self._latest_depth is None else self._latest_depth.copy()
        if depth is None:
            return ""

        vis = np.clip(depth, 0.0, 100.0) / 100.0 * 255.0
        vis = vis.astype(np.uint8)

        from PIL import Image

        img = Image.fromarray(vis, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def destroy(self) -> None:
        safe_stop_sensor(self._sensor)
        self._sensor = None
