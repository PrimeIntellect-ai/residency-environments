from __future__ import annotations

from alphaverse.capture import market_capture_spec, select_market_capture_events
from alphaverse.player import PlayerFeedEvent
from alphaverse.strategy import InputEnvelope, InputKind


def _event(cursor: int, event_kind: str, kind: InputKind) -> PlayerFeedEvent:
    return PlayerFeedEvent(
        cursor=cursor,
        envelope=InputEnvelope(
            session_id="session",
            strategy_instance_id="strategy",
            event_id=f"E{cursor}",
            kind=kind,
            exchange_time=cursor,
            available_at=cursor,
            source_event_seq=cursor,
            payload={"event_kind": event_kind},
        ),
    )


def test_capture_specs_define_raw_feed_variants_and_cursor_semantics() -> None:
    mbo = market_capture_spec("mbo")
    levels = market_capture_spec("levels")

    assert mbo["event_kinds"] == ["mbo_change", "trade"]
    assert levels["event_kinds"] == ["levels", "trade"]
    assert mbo["format"] == "ndjson"
    assert mbo["cursor"]["gaps_expected"] is True
    assert len(mbo["record_schema"]["oneOf"]) == 2


def test_capture_selection_excludes_private_and_other_feed_packets() -> None:
    events = (
        _event(1, "order_accepted", InputKind.EXECUTION),
        _event(2, "mbo_change", InputKind.MARKET),
        _event(3, "trade", InputKind.MARKET),
        _event(4, "levels", InputKind.LEVELS),
        _event(5, "fill", InputKind.EXECUTION),
    )

    assert [item.cursor for item in select_market_capture_events(events, "mbo")] == [
        2,
        3,
    ]
    assert [item.cursor for item in select_market_capture_events(events, "levels")] == [3, 4]
