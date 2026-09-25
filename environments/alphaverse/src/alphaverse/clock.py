"""Deterministic virtual time and scheduled-event ordering.

The clock is intentionally independent of wall time.  Callers schedule arbitrary
payloads at integer nanosecond timestamps and explicitly decide when virtual time
advances.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar, cast

from alphaverse.models import MarketTime, Sequence

PayloadT = TypeVar("PayloadT")


class EventProcessingLimitExceeded(RuntimeError):
    """Raised when one bounded clock advance encounters pathological work."""

    def __init__(
        self,
        *,
        processed_events: int,
        market_time: MarketTime,
        target_time: MarketTime,
    ) -> None:
        self.processed_events = processed_events
        self.market_time = market_time
        self.target_time = target_time
        super().__init__(
            f"scheduled-event limit exceeded after {processed_events} events "
            f"at market time {market_time} while advancing to {target_time}"
        )


def _require_time(value: object, *, name: str) -> MarketTime:
    """Validate a nanosecond timestamp without accepting ``bool`` as an integer."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer number of nanoseconds")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class ScheduledEvent(Generic[PayloadT]):
    """A payload's deterministic position in the virtual event queue."""

    market_time: MarketTime
    priority: int
    sequence: Sequence
    payload: PayloadT

    def __post_init__(self) -> None:
        _require_time(self.market_time, name="market_time")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError("priority must be an integer")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool):
            raise TypeError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")

    @property
    def time(self) -> MarketTime:
        """Short alias for ``market_time``."""

        return self.market_time


@dataclass(frozen=True, slots=True)
class CancellationToken:
    """A capability that can cancel one still-pending scheduled event."""

    sequence: Sequence
    _owner: VirtualClock[Any] = field(repr=False, compare=False)

    def cancel(self) -> bool:
        """Cancel the associated event, returning whether it was still pending."""

        return self._owner.cancel(self)


class VirtualClock(Generic[PayloadT]):
    """A deterministic nanosecond clock backed by a stable priority queue.

    Events are ordered by ``(market_time, priority, insertion sequence)``.  Lower
    priority numbers run first.  Insertion sequence is allocated monotonically,
    making equal-time, equal-priority scheduling stable.
    """

    def __init__(self, start_time: MarketTime = 0) -> None:
        self._now = _require_time(start_time, name="start_time")
        self._next_sequence: Sequence = 0
        self._heap: list[tuple[MarketTime, int, Sequence, ScheduledEvent[PayloadT]]] = []
        self._pending: dict[Sequence, ScheduledEvent[PayloadT]] = {}

    @property
    def now(self) -> MarketTime:
        """Current virtual time in nanoseconds."""

        return self._now

    def __len__(self) -> int:
        """Return the number of non-cancelled pending events."""

        return len(self._pending)

    def schedule(
        self,
        market_time: MarketTime,
        payload: PayloadT,
        *,
        priority: int = 0,
    ) -> CancellationToken:
        """Schedule ``payload`` at an absolute virtual timestamp.

        Scheduling at the current time is allowed.  Scheduling before it is not.
        """

        event_time = _require_time(market_time, name="market_time")
        if event_time < self._now:
            raise ValueError("cannot schedule an event in the past")
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")

        sequence = self._next_sequence
        self._next_sequence += 1
        event = ScheduledEvent(event_time, priority, sequence, payload)
        heapq.heappush(self._heap, (event_time, priority, sequence, event))
        self._pending[sequence] = event
        return CancellationToken(sequence, cast("VirtualClock[Any]", self))

    def cancel(self, token: CancellationToken) -> bool:
        """Cancel a pending event.

        A token from another clock, or one whose event has already run or been
        cancelled, has no effect and returns ``False``.
        """

        if not isinstance(token, CancellationToken) or token._owner is not self:
            return False
        return self._pending.pop(token.sequence, None) is not None

    def peek(self) -> ScheduledEvent[PayloadT] | None:
        """Return the next pending event without removing or advancing to it."""

        self._discard_cancelled_head()
        if not self._heap:
            return None
        return self._heap[0][3]

    def pop(self) -> ScheduledEvent[PayloadT]:
        """Remove the next event and advance virtual time to its timestamp."""

        self._discard_cancelled_head()
        if not self._heap:
            raise IndexError("pop from an empty event queue")
        _, _, sequence, event = heapq.heappop(self._heap)
        del self._pending[sequence]
        self._now = event.market_time
        return event

    def advance_to(self, market_time: MarketTime) -> None:
        """Advance without processing events, but never skip pending work.

        The clock may advance to the exact timestamp of its next event.  Moving
        beyond that event requires processing it first (usually via ``run_until``).
        """

        target = self._validate_target(market_time)
        next_event = self.peek()
        if next_event is not None and next_event.market_time < target:
            raise ValueError("cannot advance past a pending event")
        self._now = target

    def run_until(
        self,
        market_time: MarketTime,
        handler: Callable[[ScheduledEvent[PayloadT]], object] | None = None,
        *,
        max_events: int | None = None,
    ) -> tuple[ScheduledEvent[PayloadT], ...]:
        """Process every event due by ``market_time`` and advance to that time.

        The queue is checked again after each handler call, so events scheduled by
        a handler for the current time or another time before the target are also
        processed during the same run. ``max_events`` bounds one call's work
        without discarding the next pending event.
        """

        target = self._validate_target(market_time)
        if max_events is not None:
            if isinstance(max_events, bool) or not isinstance(max_events, int):
                raise TypeError("max_events must be an int")
            if max_events <= 0:
                raise ValueError("max_events must be positive")
        processed: list[ScheduledEvent[PayloadT]] = []
        while (event := self.peek()) is not None and event.market_time <= target:
            if max_events is not None and len(processed) >= max_events:
                raise EventProcessingLimitExceeded(
                    processed_events=len(processed),
                    market_time=self.now,
                    target_time=target,
                )
            event = self.pop()
            processed.append(event)
            if handler is not None:
                handler(event)
        self._now = target
        return tuple(processed)

    def run_until_count(
        self,
        market_time: MarketTime,
        handler: Callable[[ScheduledEvent[PayloadT]], object] | None = None,
        *,
        max_events: int | None = None,
    ) -> int:
        """Process due events like :meth:`run_until`, returning only their count.

        Episode execution does not inspect the processed records. Avoiding a
        result list and repeated ``peek``/``pop`` calls keeps that hot path lean
        while the ordinary method remains available to callers that need it.
        """

        target = self._validate_target(market_time)
        if max_events is not None:
            if isinstance(max_events, bool) or not isinstance(max_events, int):
                raise TypeError("max_events must be an int")
            if max_events <= 0:
                raise ValueError("max_events must be positive")
        processed = 0
        while True:
            self._discard_cancelled_head()
            if not self._heap or self._heap[0][0] > target:
                break
            if max_events is not None and processed >= max_events:
                raise EventProcessingLimitExceeded(
                    processed_events=processed,
                    market_time=self.now,
                    target_time=target,
                )
            _, _, sequence, event = heapq.heappop(self._heap)
            del self._pending[sequence]
            self._now = event.market_time
            processed += 1
            if handler is not None:
                handler(event)
        self._now = target
        return processed

    def _validate_target(self, market_time: MarketTime) -> MarketTime:
        target = _require_time(market_time, name="market_time")
        if target < self._now:
            raise ValueError("cannot move virtual time backwards")
        return target

    def _discard_cancelled_head(self) -> None:
        while self._heap and self._heap[0][2] not in self._pending:
            heapq.heappop(self._heap)


__all__ = [
    "CancellationToken",
    "EventProcessingLimitExceeded",
    "ScheduledEvent",
    "VirtualClock",
]
