from __future__ import annotations

import pytest
from alphaverse.event_log import EventLog
from alphaverse.models import EventKind


def _sample_log() -> EventLog:
    log = EventLog()
    log.append(
        market_time=10,
        match_event_id="M1",
        kind=EventKind.ORDER_ACCEPTED,
        product_id="ALPHA",
        data={"quantity": 3, "order_id": "O1"},
    )
    log.append(
        market_time=10,
        match_event_id="M1",
        kind=EventKind.MBO_CHANGE,
        product_id="ALPHA",
        data={"order_id": "O1", "action": "add"},
    )
    return log


def test_assigns_contiguous_sequences() -> None:
    log = _sample_log()
    assert [event.sequence for event in log] == [1, 2]
    assert log.last_sequence == 2


def test_returns_only_the_suffix_from_a_recorded_log_length() -> None:
    log = _sample_log()

    assert [event.sequence for event in log.from_index(0)] == [1, 2]
    assert [event.sequence for event in log.from_index(1)] == [2]
    assert log.from_index(2) == ()
    assert log.from_index(200) == ()


@pytest.mark.parametrize(("index", "error"), [(-1, ValueError), (True, TypeError)])
def test_suffix_lookup_validates_index(index: int, error: type[Exception]) -> None:
    with pytest.raises(error):
        _sample_log().from_index(index)


def test_jsonl_round_trip_is_byte_stable() -> None:
    original = _sample_log().to_jsonl()
    restored = EventLog.from_jsonl(original)
    assert restored.to_jsonl() == original


def test_same_events_serialize_identically_despite_mapping_order() -> None:
    first = EventLog()
    second = EventLog()
    first.append(
        market_time=1,
        match_event_id="M1",
        kind=EventKind.SESSION,
        product_id="ALPHA",
        data={"b": 2, "a": 1},
    )
    second.append(
        market_time=1,
        match_event_id="M1",
        kind=EventKind.SESSION,
        product_id="ALPHA",
        data={"a": 1, "b": 2},
    )
    assert first.to_jsonl() == second.to_jsonl()


def test_rejects_non_contiguous_replayed_sequence() -> None:
    payload = b'{"data":{},"kind":"session","market_time":0,"match_event_id":"M1","product_id":"ALPHA","sequence":2}\n'
    with pytest.raises(ValueError, match="expected 1"):
        EventLog.from_jsonl(payload)


def test_rejects_non_json_event_data() -> None:
    log = EventLog()
    with pytest.raises(TypeError):
        log.append(
            market_time=0,
            match_event_id="M1",
            kind=EventKind.SESSION,
            product_id="ALPHA",
            data={"bad": {1, 2}},  # type: ignore[dict-item]
        )


def test_disk_backed_history_replays_old_events_after_the_memory_tail() -> None:
    log = EventLog(retained_events=2)
    for sequence in range(1, 2_051):
        log.append(
            market_time=sequence,
            match_event_id=f"M{sequence}",
            kind=EventKind.SESSION,
            product_id="ALPHA",
            data={"sequence": sequence},
        )

    assert [event.sequence for event in log] == list(range(1, 2_051))
    assert [event.sequence for event in log.from_index(3)] == list(range(4, 2_051))
    assert EventLog.from_jsonl(log.to_jsonl()).to_jsonl() == log.to_jsonl()


def test_close_removes_disk_backing_file() -> None:
    log = _sample_log()
    path = log._path

    log.close()

    assert not path.exists()
