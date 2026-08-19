"""Pure helpers for NuRec frame recording."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Generic, Iterator, TypeVar

FrameT = TypeVar("FrameT")


@dataclass(frozen=True)
class RecordingStats:
    frame_count: int
    inline_write_fallbacks: int


class RecordingFrameQueue(Generic[FrameT]):
    """
    Assign stable frame numbers and avoid silent loss on queue saturation.

    The March 1 direct scripts wrote frames inline in the camera callback, so
    every delivered frame reached disk. This helper preserves that guarantee:
    when the async queue is saturated, callers can synchronously write the
    returned frame number instead of dropping it.
    """

    def __init__(self, *, maxsize: int = 100):
        self._queue: Queue[tuple[int, FrameT]] = Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._next_frame_num = 0
        self._last_source_frame_id = -1
        self._inline_write_fallbacks = 0
        self._warned_inline_fallback = False

    def reset(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        with self._lock:
            self._next_frame_num = 0
            self._last_source_frame_id = -1
            self._inline_write_fallbacks = 0
            self._warned_inline_fallback = False

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._next_frame_num

    @property
    def stats(self) -> RecordingStats:
        with self._lock:
            return RecordingStats(
                frame_count=self._next_frame_num,
                inline_write_fallbacks=self._inline_write_fallbacks,
            )

    def submit(self, *, source_frame_id: int, frame: FrameT) -> tuple[str, int, bool]:
        """
        Queue a frame for async writing.

        Returns `(action, frame_num, should_warn)` where action is one of:
        - `queued`: frame was queued successfully
        - `inline`: queue is saturated; caller should write inline
        - `duplicate`: source frame id was already recorded
        """
        with self._lock:
            if source_frame_id == self._last_source_frame_id:
                return "duplicate", -1, False
            self._last_source_frame_id = source_frame_id
            frame_num = self._next_frame_num
            self._next_frame_num += 1

        try:
            self._queue.put_nowait((frame_num, frame))
            return "queued", frame_num, False
        except Full:
            with self._lock:
                self._inline_write_fallbacks += 1
                should_warn = not self._warned_inline_fallback
                self._warned_inline_fallback = True
            return "inline", frame_num, should_warn

    def get(self, *, timeout: float) -> tuple[int, FrameT]:
        return self._queue.get(timeout=timeout)

    def drain(self) -> Iterator[tuple[int, FrameT]]:
        while True:
            try:
                yield self._queue.get_nowait()
            except Empty:
                return
