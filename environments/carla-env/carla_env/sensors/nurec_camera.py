from __future__ import annotations

import base64
import io
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..logging import get_logger
from ..nurec.config import DEFAULT_NUREC_CAMERA_LOGICAL_ID
from ..nurec.recording import RecordingFrameQueue, RecordingStats
from .camera import CameraConfig

logger = get_logger("sensors.nurec_camera")

NUREC_NATIVE_W = 3848
NUREC_NATIVE_H = 2168
NUREC_FRONT_CAMERA_PATTERNS = (
    "camera_front_wide_120fov",
    "front_wide_120fov",
    "camera_front_wide",
    "front_wide",
    "camera_front",
    "front",
)
NUREC_DEPRIORITIZED_CAMERA_HINTS = ("rear", "cross", "left", "right", "side")


class NuRecCameraSensor:
    """Camera-like wrapper for frames produced by the NuRec renderer."""

    def __init__(
        self,
        nurec_scenario: Any,
        parent_actor: Any,
        *,
        camera_logical_id: str = DEFAULT_NUREC_CAMERA_LOGICAL_ID,
        resolution_ratio: float = 0.25,
        framerate: float = 20.0,
        jpeg_quality: int = 75,
        record_video: bool = False,
        output_dir: str = "_out",
        video_fps: float = 20.0,
    ):
        self._nurec_scenario = nurec_scenario
        self._parent_actor = parent_actor
        self._preferred_camera_logical_id = str(camera_logical_id or "").strip()
        self._resolution_ratio = float(resolution_ratio)
        self._framerate = float(framerate)
        self._jpeg_quality = int(jpeg_quality)
        self._width = int(NUREC_NATIVE_W * self._resolution_ratio)
        self._height = int(NUREC_NATIVE_H * self._resolution_ratio)
        self._config = CameraConfig(
            width=self._width,
            height=self._height,
            fov=120,
            jpeg_quality=self._jpeg_quality,
            record_video=bool(record_video),
            output_dir=str(output_dir),
            video_fps=float(video_fps),
        )

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = -1
        self._lock = threading.Lock()

        self._recording = False
        self._temp_dir: Optional[Path] = None
        self._record_queue: RecordingFrameQueue[np.ndarray] = RecordingFrameQueue(maxsize=100)
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_writer = threading.Event()
        self._sensor_handle: Any = None

    @property
    def config(self) -> CameraConfig:
        return self._config

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._record_queue.frame_count

    @property
    def latest_frame_id(self) -> int:
        with self._lock:
            return self._latest_frame_id

    @property
    def recording_stats(self) -> RecordingStats:
        return self._record_queue.stats

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def setup(self) -> None:
        def _on_frame(rgb_frame: np.ndarray) -> None:
            with self._lock:
                self._latest_frame = rgb_frame.copy()
                self._latest_frame_id += 1
                frame_id = self._latest_frame_id
            if self._recording:
                self._record_frame(frame_id, rgb_frame)

        logical_id, available = self._default_camera_logical_id()
        self._nurec_scenario.add_camera(
            camera_spec=logical_id,
            callback=_on_frame,
            framerate=int(self._framerate),
            resolution_ratio=self._resolution_ratio,
        )
        try:
            cameras = getattr(self._nurec_scenario, "cameras", None)
            if isinstance(cameras, list) and cameras:
                self._sensor_handle = cameras[-1]
        except Exception:
            self._sensor_handle = None
        logger.info(
            "NuRecCameraSensor attached (%dx%d, ratio=%.3f, logical_id=%s, available=%s)",
            self._width,
            self._height,
            self._resolution_ratio,
            logical_id,
            available,
        )

    def capture(self) -> str:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            frame_id = self._latest_frame_id
        if frame is None:
            return ""

        if self._recording:
            self._record_frame(frame_id, frame)

        from PIL import Image

        image = Image.fromarray(frame)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=self._jpeg_quality)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def start_recording(self) -> None:
        if self._recording:
            return
        try:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="nurec_recording_"))
            self._record_queue.reset()
            self._stop_writer.clear()
            self._recording = True
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()
        except Exception:
            self._recording = False
            self._stop_writer.set()
            self._writer_thread = None
            self._record_queue.reset()
            self._cleanup_temp()
            raise

    def stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._stop_writer.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        self._writer_thread = None
        for frame_num, frame in self._record_queue.drain():
            self._write_frame(frame_num, frame)

    def save_video(self, filename: str) -> Optional[str]:
        from .camera import CameraSensor

        if not self._temp_dir or not self._temp_dir.exists():
            return None

        frames = sorted(self._temp_dir.glob("frame_*.jpg"))
        if not frames:
            self._cleanup_temp()
            return None

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        helper = CameraSensor.__new__(CameraSensor)
        helper._config = self._config
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
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / f"{prefix}_frames"
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(self._temp_dir), str(dest))
            self._temp_dir = None
            return str(dest)
        except Exception as exc:
            logger.warning("Failed to save NuRec frames: %s", exc)
            return None

    def destroy(self) -> None:
        self.stop_recording()
        self._cleanup_temp()
        try:
            cameras = getattr(self._nurec_scenario, "cameras", None)
            if isinstance(cameras, list) and self._sensor_handle is not None:
                self._nurec_scenario.cameras = [
                    camera for camera in cameras if camera is not self._sensor_handle
                ]
        except Exception:
            pass
        self._sensor_handle = None
        with self._lock:
            self._latest_frame = None

    def _record_frame(self, frame_id: int, frame: np.ndarray) -> None:
        action, frame_num, should_warn = self._record_queue.submit(
            source_frame_id=frame_id,
            frame=frame.copy(),
        )
        if action == "queued" or action == "duplicate":
            return
        if should_warn:
            logger.warning(
                "NuRec recording queue saturated; falling back to inline frame writes to preserve continuity"
            )
        self._write_frame(frame_num, frame)

    def _writer_loop(self) -> None:
        while not self._stop_writer.is_set():
            try:
                frame_num, frame = self._record_queue.get(timeout=0.5)
                self._write_frame(frame_num, frame)
            except Exception as exc:
                from queue import Empty

                if isinstance(exc, Empty):
                    continue
                logger.debug("NuRec async frame write failed", exc_info=True)
                continue

    def _write_frame(self, frame_num: int, rgb: np.ndarray) -> None:
        if self._temp_dir is None:
            return
        from PIL import Image

        path = self._temp_dir / f"frame_{frame_num:06d}.jpg"
        Image.fromarray(rgb).save(path, format="JPEG", quality=self._jpeg_quality)

    def _cleanup_temp(self) -> None:
        if self._temp_dir is not None:
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
            self._temp_dir = None

    def _default_camera_logical_id(self) -> tuple[str, list[str]]:
        logical_id = DEFAULT_NUREC_CAMERA_LOGICAL_ID
        try:
            available = list(self._nurec_scenario.get_available_cameras())
        except Exception:
            available = []

        if not available:
            try:
                calibrations = getattr(self._nurec_scenario.scenario, "camera_calibrations", {})
                available = [camera.logical_sensor_name for camera in calibrations.values()]
            except Exception:
                available = []

        try:
            renderer = getattr(self._nurec_scenario, "renderer", None)
            if renderer is not None:
                renderer_available = list(getattr(renderer, "available_cameras", {}).keys())
                if renderer_available:
                    available = renderer_available
        except Exception:
            pass

        selected = self._select_camera_logical_id(available)
        if selected is None:
            selected = logical_id
        return selected, available

    def _select_camera_logical_id(self, available: list[str]) -> str | None:
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in available:
            normalized = str(candidate or "").strip()
            if not normalized or normalized in seen:
                continue
            deduped.append(normalized)
            seen.add(normalized)

        if not deduped:
            return None

        preferred = self._preferred_camera_logical_id
        if preferred:
            for candidate in deduped:
                if candidate == preferred:
                    return candidate
            logger.warning(
                "Requested NuRec camera %r was not available; falling back to automatic selection from %s",
                preferred,
                deduped,
            )

        lowered = [(candidate, candidate.lower()) for candidate in deduped]

        for pattern in NUREC_FRONT_CAMERA_PATTERNS:
            pattern = pattern.lower()
            for candidate, lowered_candidate in lowered:
                if pattern in lowered_candidate:
                    return candidate

        non_deprioritized = [
            candidate
            for candidate, lowered_candidate in lowered
            if not any(hint in lowered_candidate for hint in NUREC_DEPRIORITIZED_CAMERA_HINTS)
        ]
        if non_deprioritized:
            for candidate in non_deprioritized:
                lowered_candidate = candidate.lower()
                if "wide" in lowered_candidate or "120" in lowered_candidate:
                    return candidate
            return non_deprioritized[0]

        for candidate, lowered_candidate in lowered:
            if "wide" in lowered_candidate or "120" in lowered_candidate:
                return candidate
        return deduped[0]
