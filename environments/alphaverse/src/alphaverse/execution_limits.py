"""Wall-clock budgets for trusted server work, independent of market time."""

from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

_rpc_deadline: ContextVar[float | None] = ContextVar("alphaverse_rpc_deadline", default=None)
_slice_deadline: ContextVar[float | None] = ContextVar("alphaverse_slice_deadline", default=None)


class ToolDeadlineExceeded(BaseException):
    """Escape synchronous work without being mistaken for a strategy fault."""


class SimulationSliceExpired(Exception):
    """Yield at an event boundary without discarding pending simulation work."""


@contextmanager
def rpc_budget(seconds: float):
    token = _rpc_deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _rpc_deadline.reset(token)


def remaining_seconds(default: float) -> float:
    deadline = _rpc_deadline.get()
    return default if deadline is None else max(0.001, min(default, deadline - time.monotonic()))


@contextmanager
def synchronous_deadline(seconds: float):
    """Interrupt blocking Python work in the dedicated POSIX Toolset process."""

    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise RuntimeError("bounded Alphaverse tools require a POSIX main-thread Toolset server")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError("Alphaverse requires exclusive use of the Toolset deadline timer")

    def expired(signum, frame):
        raise ToolDeadlineExceeded("server-side tool deadline exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining_seconds(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def simulation_slice(seconds: float):
    token = _slice_deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _slice_deadline.reset(token)


def check_simulation_slice() -> None:
    deadline = _slice_deadline.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise SimulationSliceExpired
