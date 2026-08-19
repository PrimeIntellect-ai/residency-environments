from __future__ import annotations

import base64
import io
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Optional

import numpy as np

from ..compat import safe_stop_sensor
from ..logging import get_logger

logger = get_logger("sensors.camera")


@dataclass
class CameraConfig:
    """Shared config for RGB and depth cameras."""

    width: int = 640
    height: int = 360
    fov: int = 90
    jpeg_quality: int = 75
    record_video: bool = False
    output_dir: str = "_out"
    video_fps: float = 20.0


class CameraSensor:
    """Front-facing RGB camera with JPEG capture and optional episode recording."""

    def __init__(
        self,
        actor_manager: Any,
        parent: Any,
        config: CameraConfig | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
        fov: int | None = None,
        jpeg_quality: int | None = None,
    ):
        self._actor_manager = actor_manager
        self._parent = parent
        if config is None:
            config = CameraConfig(
                width=int(width or 640),
                height=int(height or 360),
                fov=int(fov or 90),
                jpeg_quality=int(jpeg_quality or 75),
            )
        self._config = config
        self._sensor: Optional[Any] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id: int = -1
        self._lock = threading.Lock()

        self._recording = False
        self._temp_dir: Optional[Path] = None
        self._frame_count = 0
        self._write_queue: Queue[tuple[int, np.ndarray]] = Queue(maxsize=100)
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_writer = threading.Event()

    @property
    def config(self) -> CameraConfig:
        return self._config

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame[:, :, ::-1].copy()

    @property
    def latest_frame_id(self) -> int:
        with self._lock:
            return self._latest_frame_id

    def setup(self) -> None:
        import carla

        cfg = self._config
        transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        sensor = self._actor_manager.spawn_sensor(
            "sensor.camera.rgb",
            transform,
            attach_to=self._parent,
            attributes={
                "image_size_x": str(cfg.width),
                "image_size_y": str(cfg.height),
                "fov": str(cfg.fov),
            },
        )
        self._sensor = sensor

        def _on_image(image: Any) -> None:
            try:
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                bgr = array[:, :, :3]
            except Exception:
                return

            frame_id = int(getattr(image, "frame", -1))
            with self._lock:
                self._latest_frame = bgr.copy()
                self._latest_frame_id = frame_id

            if self._recording:
                try:
                    self._write_queue.put_nowait((self._frame_count, bgr.copy()))
                    self._frame_count += 1
                except Full:
                    pass

        sensor.listen(_on_image)
        logger.info("CameraSensor attached (%dx%d, fov=%d)", cfg.width, cfg.height, cfg.fov)

    def capture(self) -> str:
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return ""
        from PIL import Image

        rgb = frame[:, :, ::-1]
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._config.jpeg_quality)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def start_recording(self) -> None:
        if self._recording:
            return

        try:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="carla_recording_"))
            self._frame_count = 0
            self._stop_writer.clear()
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()
            self._recording = True
            logger.info("Recording started (temp: %s)", self._temp_dir)
        except Exception:
            self._recording = False
            self._stop_writer.set()
            self._writer_thread = None
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
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)

        while not self._write_queue.empty():
            try:
                frame_num, frame = self._write_queue.get_nowait()
                self._write_frame(frame_num, frame)
            except Empty:
                break

        logger.info("Recording stopped (%d frames)", self._frame_count)

    def save_video(self, filename: str) -> Optional[str]:
        if not self._temp_dir or not self._temp_dir.exists():
            return None

        frames = sorted(self._temp_dir.glob("frame_*.jpg"))
        if not frames:
            self._cleanup_temp()
            return None

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        path = self._compile_video_cv2(frames, output_path)
        if path is None:
            path = self._compile_video_imageio(frames, output_path)
        if path is None:
            path = self._compile_video_ffmpeg(output_path)

        if path:
            self._cleanup_temp()
            logger.info("Video saved: %s (%d frames)", path, len(frames))
            return path

        # No encoder available — preserve raw frames instead of deleting them.
        frames_dir = self.save_frames(output_path.stem)
        if frames_dir:
            logger.warning(
                "Video compilation failed; raw frames preserved at %s",
                frames_dir,
            )
            return frames_dir

        logger.warning("Video compilation failed and frame preservation also failed")
        return None

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
            logger.info("Frames saved to %s", dest)
            return str(dest)
        except Exception as e:
            logger.warning("Failed to save frames: %s", e)
            return None

    def destroy(self) -> None:
        self.stop_recording()
        self._cleanup_temp()
        safe_stop_sensor(self._sensor)
        self._sensor = None

    def _writer_loop(self) -> None:
        while not self._stop_writer.is_set():
            try:
                frame_num, frame = self._write_queue.get(timeout=0.5)
                self._write_frame(frame_num, frame)
            except Empty:
                continue
            except Exception:
                continue

    def _write_frame(self, frame_num: int, bgr: np.ndarray) -> None:
        if self._temp_dir is None:
            return
        path = self._temp_dir / f"frame_{frame_num:06d}.jpg"
        try:
            from PIL import Image

            rgb = bgr[:, :, ::-1]
            Image.fromarray(rgb).save(str(path), format="JPEG", quality=95)
        except Exception as e:
            logger.debug("Failed to write frame %d: %s", frame_num, e)

    def _compiled_video_is_usable(self, output_path: Path) -> bool:
        if not output_path.exists():
            return False
        if output_path.stat().st_size < 1024:
            return False

        try:
            import cv2
        except ImportError:
            return True

        capture = cv2.VideoCapture(str(output_path))
        try:
            if not capture.isOpened():
                return False
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            return frame_count > 0 and width > 0 and height > 0
        finally:
            capture.release()

    def _accept_compiled_video(self, output_path: Path, *, source: str) -> Optional[str]:
        if self._compiled_video_is_usable(output_path):
            return str(output_path)
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug("Failed to remove unusable %s video %s: %s", source, output_path, e)
        logger.warning(
            "%s video compilation produced an unusable file; trying fallback backend", source
        )
        return None

    def _compile_video_cv2(self, frames: list[Path], output_path: Path) -> Optional[str]:
        try:
            import cv2
        except ImportError:
            return None

        writer = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self._config.video_fps,
                (self._config.width, self._config.height),
            )
            if not writer.isOpened():
                return None
            written = 0
            for frame_path in frames:
                image = cv2.imread(str(frame_path))
                if image is not None:
                    writer.write(image)
                    written += 1
            writer.release()
            writer = None
            if written == 0:
                return None
            return self._accept_compiled_video(output_path, source="cv2")
        except Exception as e:
            logger.warning("cv2 video compilation failed: %s", e)
            return None
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

    def _compile_video_imageio(self, frames: list[Path], output_path: Path) -> Optional[str]:
        try:
            import imageio.v2 as imageio
        except ImportError:
            return None

        writer = None
        try:
            writer = imageio.get_writer(
                str(output_path),
                fps=max(1.0, float(self._config.video_fps)),
                codec="libx264",
                macro_block_size=1,
            )
            written = 0
            for frame_path in frames:
                writer.append_data(imageio.imread(frame_path))
                written += 1
            writer.close()
            writer = None
            if written == 0:
                return None
            return self._accept_compiled_video(output_path, source="imageio")
        except Exception as e:
            logger.warning("imageio video compilation failed: %s", e)
            return None
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass

    def _compile_video_ffmpeg(self, output_path: Path) -> Optional[str]:
        if self._temp_dir is None:
            return None

        import subprocess

        pattern = str(self._temp_dir / "frame_%06d.jpg")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(self._config.video_fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                return self._accept_compiled_video(output_path, source="ffmpeg")
            logger.warning(
                "ffmpeg failed (rc=%d): %s",
                result.returncode,
                result.stderr.decode(errors="replace")[:200],
            )
            return None
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning("ffmpeg video compilation failed: %s", e)
            return None

    def _cleanup_temp(self) -> None:
        if self._temp_dir is not None and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning("Failed to cleanup temp dir: %s", e)
        self._temp_dir = None
