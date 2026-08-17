from __future__ import annotations

import pytest
from alphaverse.clock import (
    EventProcessingLimitExceeded,
    ScheduledEvent,
    VirtualClock,
)


def test_clock_requires_non_negative_integer_nanoseconds() -> None:
    with pytest.raises(ValueError, match="start_time must be non-negative"):
        VirtualClock(-1)
    with pytest.raises(TypeError, match="integer number of nanoseconds"):
        VirtualClock(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer number of nanoseconds"):
        VirtualClock(True)


def test_scheduled_event_validates_its_ordering_fields() -> None:
    with pytest.raises(ValueError, match="market_time must be non-negative"):
        ScheduledEvent(-1, 0, 0, "payload")
    with pytest.raises(TypeError, match="priority must be an integer"):
        ScheduledEvent(0, True, 0, "payload")
    with pytest.raises(ValueError, match="sequence must be non-negative"):
        ScheduledEvent(0, 0, -1, "payload")


def test_scheduling_orders_by_time_priority_then_insertion_sequence() -> None:
    clock = VirtualClock[str]()
    clock.schedule(20, "later", priority=-10)
    clock.schedule(10, "normal-first")
    clock.schedule(10, "high-priority", priority=-1)
    clock.schedule(10, "normal-second")

    events = clock.run_until(20)

    assert [event.payload for event in events] == [
        "high-priority",
        "normal-first",
        "normal-second",
        "later",
    ]
    assert [event.sequence for event in events] == [2, 1, 3, 0]
    assert clock.now == 20
    assert len(clock) == 0


def test_peek_does_not_remove_event_or_advance_time() -> None:
    clock = VirtualClock[str](start_time=5)
    clock.schedule(8, "message")

    first = clock.peek()

    assert first is clock.peek()
    assert first is not None
    assert first.time == 8
    assert first.market_time == 8
    assert first.payload == "message"
    assert clock.now == 5
    assert len(clock) == 1


def test_pop_removes_event_and_advances_to_its_time() -> None:
    clock = VirtualClock[str]()
    clock.schedule(9, "message")

    event = clock.pop()

    assert event.payload == "message"
    assert clock.now == 9
    assert clock.peek() is None
    with pytest.raises(IndexError, match="empty event queue"):
        clock.pop()


def test_schedule_allows_now_but_rejects_past_and_invalid_priority() -> None:
    clock = VirtualClock[str](start_time=10)
    clock.schedule(10, "now")

    with pytest.raises(ValueError, match="in the past"):
        clock.schedule(9, "past")
    with pytest.raises(ValueError, match="market_time must be non-negative"):
        clock.schedule(-1, "negative")
    with pytest.raises(TypeError, match="priority must be an integer"):
        clock.schedule(11, "bad priority", priority=True)


def test_advance_to_is_monotonic_and_does_not_skip_pending_event() -> None:
    clock = VirtualClock[str](start_time=2)
    clock.schedule(5, "due")

    clock.advance_to(5)
    assert clock.now == 5

    with pytest.raises(ValueError, match="pending event"):
        clock.advance_to(6)
    with pytest.raises(ValueError, match="backwards"):
        clock.advance_to(4)

    assert clock.pop().payload == "due"
    clock.advance_to(10)
    assert clock.now == 10


def test_run_until_processes_due_events_and_leaves_future_events() -> None:
    clock = VirtualClock[str]()
    clock.schedule(3, "first")
    clock.schedule(5, "boundary")
    clock.schedule(6, "future")

    processed = clock.run_until(5)

    assert [event.payload for event in processed] == ["first", "boundary"]
    assert clock.now == 5
    assert clock.peek() is not None
    assert clock.peek().payload == "future"  # type: ignore[union-attr]


def test_run_until_handler_can_schedule_more_due_work() -> None:
    clock = VirtualClock[str]()
    clock.schedule(2, "root")
    handled: list[str] = []

    def handle(event: ScheduledEvent[str]) -> None:
        payload = event.payload
        handled.append(payload)
        if payload == "root":
            clock.schedule(2, "same-time")
            clock.schedule(4, "before-target")

    processed = clock.run_until(5, handle)

    assert handled == ["root", "same-time", "before-target"]
    assert [event.payload for event in processed] == handled
    assert clock.now == 5


def test_run_until_event_limit_preserves_pending_work() -> None:
    clock = VirtualClock[str]()
    for index in range(4):
        clock.schedule(2, f"event-{index}")

    with pytest.raises(EventProcessingLimitExceeded) as raised:
        clock.run_until(5, max_events=3)

    assert raised.value.processed_events == 3
    assert raised.value.market_time == 2
    assert raised.value.target_time == 5
    assert clock.now == 2
    assert len(clock) == 1
    assert clock.run_until(5, max_events=3)[0].payload == "event-3"
    assert clock.now == 5


def test_run_until_count_matches_order_and_handles_new_and_cancelled_work() -> None:
    clock = VirtualClock[str]()
    cancelled = clock.schedule(2, "cancelled")
    clock.schedule(1, "root")
    cancelled.cancel()
    handled: list[str] = []

    def handle(event) -> None:
        handled.append(event.payload)
        if event.payload == "root":
            clock.schedule(1, "same-time")
            clock.schedule(3, "later")

    assert clock.run_until_count(3, handle) == 3
    assert handled == ["root", "same-time", "later"]
    assert clock.now == 3


def test_run_until_count_limit_preserves_pending_work() -> None:
    clock = VirtualClock[str]()
    for index in range(4):
        clock.schedule(2, f"event-{index}")

    with pytest.raises(EventProcessingLimitExceeded) as raised:
        clock.run_until_count(5, max_events=3)

    assert raised.value.processed_events == 3
    assert clock.now == 2
    assert len(clock) == 1


def test_cancellation_is_idempotent_and_skipped_by_queue() -> None:
    clock = VirtualClock[str]()
    cancelled = clock.schedule(1, "cancelled")
    kept = clock.schedule(2, "kept")

    assert cancelled.cancel() is True
    assert cancelled.cancel() is False
    assert len(clock) == 1
    assert clock.peek() is not None
    assert clock.peek().payload == "kept"  # type: ignore[union-attr]
    assert clock.run_until(2)[0].payload == "kept"
    assert kept.cancel() is False


def test_token_cannot_cancel_event_on_another_clock() -> None:
    first = VirtualClock[str]()
    second = VirtualClock[str]()
    token = first.schedule(1, "first")
    second.schedule(1, "second")

    assert second.cancel(token) is False
    assert second.pop().payload == "second"


def test_cancelled_event_does_not_block_manual_advancement() -> None:
    clock = VirtualClock[str]()
    token = clock.schedule(3, "cancelled")
    token.cancel()

    clock.advance_to(10)

    assert clock.now == 10
    assert clock.peek() is None


def test_run_until_rejects_backwards_or_non_integer_target() -> None:
    clock = VirtualClock[str](5)

    with pytest.raises(ValueError, match="backwards"):
        clock.run_until(4)
    with pytest.raises(TypeError, match="integer number of nanoseconds"):
        clock.run_until(5.1)  # type: ignore[arg-type]
