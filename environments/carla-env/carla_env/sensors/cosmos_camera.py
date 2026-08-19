from __future__ import annotations

import base64
import io
import shutil
import tempfile
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Optional

import numpy as np

from ..cosmos import CosmosClient, CosmosConfig
from ..logging import get_logger
from .camera import CameraConfig, CameraSensor

logger = get_logger("sensors.cosmos_camera")


class CosmosCameraSensor:
    """Camera wrapper that stylizes RGB frames through a Cosmos server."""

    _HEALTH_RETRY_INTERVAL_S = 2.0

    def __init__(
        self,
        actor_manager: Any,
        parent: Any,
        cosmos_config: CosmosConfig,
        camera_config: CameraConfig | None = None,
    ):
        self._config = cosmos_config
        self._client = CosmosClient(cosmos_config)
        self._inner = CameraSensor(actor_manager, parent, camera_config)
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stylization_enabled = True
        self._recording = False
        self._temp_dir: Optional[Path] = None
        self._frame_count = 0
        self._write_queue: Queue[tuple[int, np.ndarray]] = Queue(maxsize=100)
        self._writer_thread: Optional[threading.Thread] = None
        self._record_thread: Optional[threading.Thread] = None
        self._stop_writer = threading.Event()
        self._last_recorded_id = -1
        self._retry_health_after_setup_failure = False
        self._next_health_retry_ts = 0.0

    @property
    def config(self) -> CameraConfig:
        return self._inner.config

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return self._inner.latest_frame

    def setup(self) -> None:
        self._inner.setup()
        self._stylization_enabled = self._client.health()
        if not self._stylization_enabled:
            self._retry_health_after_setup_failure = True
            self._next_health_retry_ts = time.monotonic() + self._HEALTH_RETRY_INTERVAL_S
            logger.warning(
                "Cosmos server at %s is not healthy; using raw CARLA RGB frames until it recovers",
                self._config.server_url,
            )
        else:
            self._retry_health_after_setup_failure = False
            logger.info("CosmosCameraSensor connected to %s", self._config.server_url)

    def capture(self) -> str:
        raw_b64 = self._inner.capture()
        if not raw_b64:
            return ""
        self._maybe_reenable_stylization()
        if not self._stylization_enabled:
            with self._lock:
                self._latest_frame = self._inner.latest_frame
            return raw_b64
        try:
            stylized_b64, stylized_rgb = self._stylize_base64(raw_b64)
            with self._lock:
                self._latest_frame = stylized_rgb.copy()
            return stylized_b64
        except Exception as exc:
            self._disable_stylization(exc)
            with self._lock:
                self._latest_frame = self._inner.latest_frame
            return raw_b64

    def start_recording(self) -> None:
        if self._recording:
            return
        try:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="cosmos_recording_"))
            self._frame_count = 0
            self._last_recorded_id = -1
            self._stop_writer.clear()
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self._writer_thread.start()
            self._record_thread.start()
            self._recording = True
        except Exception:
            self._recording = False
            self._stop_writer.set()
            self._writer_thread = None
            self._record_thread = None
            self._frame_count = 0
            while not self._write_queue.empty():
                try:
                    self._write_queue.get_nowait()
                except Empty:
                    break
            self._cleanup_temp()
            raise

    def stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._stop_writer.set()
        if self._record_thread is not None:
            self._record_thread.join(timeout=45.0)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10.0)
        while not self._write_queue.empty():
            try:
                frame_num, frame = self._write_queue.get_nowait()
                self._write_frame(frame_num, frame)
            except Empty:
                break

    def save_video(self, filename: str) -> Optional[str]:
        from PIL import Image

        from .camera import CameraSensor

        if not self._temp_dir or not self._temp_dir.exists():
            return None

        frames = sorted(self._temp_dir.glob("frame_*.jpg"))
        if not frames:
            self._cleanup_temp()
            return None

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        frame_w = int(self._inner.config.width)
        frame_h = int(self._inner.config.height)
        try:
            with Image.open(frames[0]) as first_frame:
                frame_w, frame_h = first_frame.size
        except Exception as exc:
            logger.warning("Failed to inspect Cosmos frame size; using raw camera size: %s", exc)

        helper = CameraSensor.__new__(CameraSensor)
        helper._config = CameraConfig(
            width=frame_w,
            height=frame_h,
            fov=self._inner.config.fov,
            jpeg_quality=self._inner.config.jpeg_quality,
            record_video=self._inner.config.record_video,
            output_dir=self._inner.config.output_dir,
            video_fps=self._inner.config.video_fps,
        )
        helper._temp_dir = self._temp_dir

        path = helper._compile_video_cv2(frames, output_path)
        if path is None:
            path = helper._compile_video_imageio(frames, output_path)
        if path is None:
            path = helper._compile_video_ffmpeg(output_path)

        if path:
            self._cleanup_temp()
            return path
        return self.save_frames(output_path.stem)

    def save_frames(self, prefix: str) -> Optional[str]:
        if not self._temp_dir or not self._temp_dir.exists():
            return None
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / f"{prefix}_frames"
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(self._temp_dir), str(dest))
            self._temp_dir = None
            return str(dest)
        except Exception as exc:
            logger.warning("Failed to save Cosmos frames: %s", exc)
            return None

    def destroy(self) -> None:
        self.stop_recording()
        self._cleanup_temp()
        self._inner.destroy()
        with self._lock:
            self._latest_frame = None

    def _stylize_base64(self, frame_base64: str) -> tuple[str, np.ndarray]:
        from PIL import Image

        stylized_b64 = self._client.stylize(frame_base64)
        image_bytes = base64.b64decode(stylized_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return stylized_b64, np.array(image)

    def _stylize_rgb_frame(self, rgb_frame: np.ndarray) -> np.ndarray:
        from PIL import Image

        image = Image.fromarray(rgb_frame)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=self._inner.config.jpeg_quality)
        buf.seek(0)
        _, stylized_rgb = self._stylize_base64(base64.b64encode(buf.read()).decode("utf-8"))
        return stylized_rgb

    def _enqueue_frame(self, _source_frame_id: int, frame: np.ndarray) -> None:
        try:
            frame_num = self._frame_count
            self._write_queue.put_nowait((frame_num, frame.copy()))
            self._frame_count += 1
        except Full:
            pass

    def _record_loop(self) -> None:
        while not self._stop_writer.is_set():
            frame_id = self._inner.latest_frame_id
            frame = self._inner.latest_frame
            if frame is None or frame_id < 0 or frame_id == self._last_recorded_id:
                time.sleep(0.05)
                continue
            stylized = frame
            self._maybe_reenable_stylization()
            if self._stylization_enabled:
                try:
                    stylized = self._stylize_rgb_frame(frame)
                except Exception as exc:
                    self._disable_stylization(exc)
                    stylized = frame
            with self._lock:
                self._latest_frame = stylized.copy()
            self._last_recorded_id = frame_id
            self._enqueue_frame(frame_id, stylized)

    def _writer_loop(self) -> None:
        while not self._stop_writer.is_set():
            try:
                frame_num, frame = self._write_queue.get(timeout=0.5)
                self._write_frame(frame_num, frame)
            except Empty:
                continue
            except Exception:
                continue

    def _write_frame(self, frame_num: int, rgb: np.ndarray) -> None:
        if self._temp_dir is None:
            return
        from PIL import Image

        path = self._temp_dir / f"frame_{frame_num:06d}.jpg"
        Image.fromarray(rgb).save(path, format="JPEG", quality=95)

    def _cleanup_temp(self) -> None:
        if self._temp_dir is not None and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as exc:
                logger.warning("Failed to cleanup Cosmos temp dir: %s", exc)
        self._temp_dir = None

    def _disable_stylization(self, exc: Exception | None = None) -> None:
        if self._stylization_enabled and exc is not None:
            logger.warning(
                "Cosmos stylization failed; using raw CARLA frame for the rest of the episode: %s",
                exc,
            )
        self._stylization_enabled = False
        self._retry_health_after_setup_failure = False

    def _maybe_reenable_stylization(self) -> None:
        if self._stylization_enabled or not self._retry_health_after_setup_failure:
            return
        now = time.monotonic()
        if now < self._next_health_retry_ts:
            return
        self._next_health_retry_ts = now + self._HEALTH_RETRY_INTERVAL_S
        try:
            healthy = self._client.health()
        except Exception:
            healthy = False
        if healthy:
            self._stylization_enabled = True
            self._retry_health_after_setup_failure = False
            logger.info(
                "Cosmos server at %s recovered; resuming stylized frames", self._config.server_url
            )
