from __future__ import annotations

import pytest
from alphaverse.episode import Episode
from alphaverse.models import CancelOrder, NewOrder, Side
from alphaverse.player import PlayerSession, WaitResult
from alphaverse.profiles import MarginConfig, ParticipantSpec, TechnologyProfile
from alphaverse.strategy import Strategy
from alphaverse.strategy.protocol import InputEnvelope, InputKind


def _envelope(sequence: int) -> InputEnvelope:
    return InputEnvelope(
        session_id="S1",
        strategy_instance_id="I1",
        event_id=f"event-{sequence}",
        kind=InputKind.MARKET,
        exchange_time=sequence,
        available_at=sequence,
        source_event_seq=sequence,
        payload={"event_kind": "trade", "sequence": sequence},
    )


def _session(*, entry_latency: int = 0) -> PlayerSession:
    return PlayerSession(
        Episode(),
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="direct-api:v1",
            account_starting_cash=10_000,
            technology=TechnologyProfile(order_entry_latency=entry_latency),
        ),
    )


def test_player_action_is_queued_and_processed_by_virtual_wait() -> None:
    session = _session(entry_latency=7)
    receipt = session.submit_limit(
        client_order_id="bid-1",
        side=Side.BUY,
        price=99,
        quantity=2,
    )

    assert receipt.arrival_at == 7
    session.wait(until=6)
    assert session.open_orders() == ()
    session.wait(duration=1)
    assert session.open_orders()[0]["client_order_id"] == "bid-1"


def test_player_feed_obeys_entitlement_and_delivery_latency() -> None:
    session = PlayerSession(
        Episode(),
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="direct-api:v1",
            account_starting_cash=10_000,
            technology=TechnologyProfile(
                market_data_latency=3,
                level_feed_latency=4,
                mbo_entitled=False,
            ),
        ),
    )
    session.submit_limit(client_order_id="bid", side=Side.BUY, price=99, quantity=1)
    session.wait(until=2)
    assert session.events() == ()
    session.wait(until=3)
    assert [item.envelope.payload["event_kind"] for item in session.events()] == ["order_accepted"]
    session.wait(until=7)
    assert [item.envelope.payload["event_kind"] for item in session.events()] == [
        "order_accepted",
        "levels",
    ]


def test_wait_requires_exactly_one_clock_target() -> None:
    session = _session()
    with pytest.raises(ValueError, match="exactly one"):
        session.wait()
    with pytest.raises(ValueError, match="exactly one"):
        session.wait(duration=1, until=1)


def test_staged_source_is_validated_without_replacing_live_strategy() -> None:
    session = _session()
    first = """
from alphaverse.strategy import Strategy
class StrategyImpl(Strategy):
    pass
"""
    second = first.replace("pass", "def on_start(self, ctx, event):\n        return None")
    live_version = session.deploy_source(first)

    staged_version = session.stage_source(second)

    assert session.strategy_status()["strategy_version_id"] == live_version
    assert session.strategy_status()["staged_strategy_version_id"] == staged_version
    assert len(session.deployment_records()) == 1

    assert session.activate_staged_source() == staged_version
    assert session.strategy_status()["strategy_version_id"] == staged_version
    assert session.strategy_status()["staged_strategy_version_id"] is None
    assert len(session.deployment_records()) == 2


def test_staged_source_rejects_constructor_failure_and_preserves_incumbent() -> None:
    session = _session()
    valid = """
from alphaverse.strategy import Strategy
class StrategyImpl(Strategy):
    pass
"""
    invalid = """
from alphaverse.strategy import Strategy
class StrategyImpl(Strategy):
    def __init__(self):
        raise ValueError("invalid parameter")
"""
    live_version = session.deploy_source(valid)

    with pytest.raises(RuntimeError, match="strategy process failed to initialize"):
        session.stage_source(invalid)

    status = session.strategy_status()
    assert status["strategy_version_id"] == live_version
    assert status["staged_strategy_version_id"] is None
    assert len(session.deployment_records()) == 1


def test_capture_window_replays_spilled_player_history() -> None:
    session = _session()
    inbox = session._strategy.inbox
    # Use a tiny tail in the test while exercising the same cursor store used
    # by the public capture APIs.
    inbox.close()
    from alphaverse.player import _PlayerInbox

    session._strategy.inbox = _PlayerInbox(retained_events=2)
    for sequence in range(1, 2_051):
        session._strategy.inbox.append(_envelope(sequence))

    events, next_cursor = session.capture_window(after_cursor=2)

    assert next_cursor == 2_050
    assert [item.cursor for item in events] == list(range(3, 2_051))
    assert [item.envelope.event_id for item in events] == [f"event-{sequence}" for sequence in range(3, 2_051)]


def test_capture_page_freezes_and_bounds_spilled_player_history() -> None:
    session = _session()
    inbox = session._strategy.inbox
    inbox.close()
    from alphaverse.player import _PlayerInbox

    session._strategy.inbox = _PlayerInbox(retained_events=2)
    for sequence in range(1, 2_051):
        session._strategy.inbox.append(_envelope(sequence))

    first, next_cursor, snapshot_end = session.capture_page(
        after_cursor=2,
        limit=1_000,
    )
    session._strategy.inbox.append(_envelope(2_051))
    second, final_cursor, repeated_end = session.capture_page(
        after_cursor=next_cursor,
        through_cursor=snapshot_end,
        limit=1_000,
    )
    third, final_cursor, repeated_end = session.capture_page(
        after_cursor=final_cursor,
        through_cursor=snapshot_end,
        limit=1_000,
    )

    assert [item.cursor for item in first] == list(range(3, 1_003))
    assert next_cursor == 1_002
    assert snapshot_end == repeated_end == final_cursor == 2_050
    assert second[-1].envelope.event_id == "event-2002"
    assert third[-1].envelope.event_id == "event-2050"


def test_strategy_alerts_are_durable_cursor_addressable_and_interrupt_waits() -> None:
    class Alerting(Strategy):
        def on_start(self, ctx, event):
            return [ctx.set_timer("alert", fire_at=2)]

        def on_timer(self, ctx, event):
            return [ctx.emit_alert("signal", "flow threshold crossed", {"score": 8})]

        def on_alert(self, ctx, event):
            raise AssertionError("player alerts must not be redelivered to automation")

    session = _session()
    session.deploy_strategy(Alerting())

    result = session.wait(duration=10_000_000_000, interrupt_on_alert=True)

    assert result == WaitResult(
        market_time=5_000_000_000,
        target_market_time=10_000_000_000,
        interrupted_by_alert=True,
        alert_cursor=1,
    )
    alerts = session.alerts()
    assert [(item.cursor, item.envelope.kind, item.envelope.payload["code"]) for item in alerts] == [
        (1, InputKind.ALERT, "signal")
    ]
    assert session.events()[0] == alerts[0]
    assert session.alert_status()["has_unacknowledged_alert"] is True

    assert session.acknowledge_alerts(alerts[0].cursor) == alerts[0].cursor
    completed = session.wait(until=10_000_000_000, interrupt_on_alert=True)
    assert completed == WaitResult(10_000_000_000, 10_000_000_000, False)

    session.stop_strategy()
    assert session.alerts()[0] == alerts[0]


def test_strategy_can_turn_a_margin_call_into_a_player_alert() -> None:
    class RiskAlerting(Strategy):
        def on_risk(self, ctx, event):
            return [ctx.emit_alert("margin", "account needs review", event.payload)]

    episode = Episode()
    session = PlayerSession(
        episode,
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="direct-api:v1",
            account_starting_cash=260,
            margin=MarginConfig(100, 90, grace_period=5),
        ),
    )
    session.deploy_strategy(RiskAlerting())
    exchange = episode.exchange
    for participant_id in ("high-bid", "seller", "low-bid", "low-ask"):
        exchange.register_account(participant_id, starting_cash=1_000)

    def submit(command: NewOrder):
        result = exchange.submit_order(command)
        episode._publish(result.events)
        return result

    high_bid = submit(NewOrder("high-bid", "bid", Side.BUY, 99, 5))
    submit(NewOrder("seller", "offer", Side.SELL, 100, 2))
    submit(NewOrder("player", "entry", Side.BUY, 100, 2))
    assert high_bid.order_id is not None
    cancelled = exchange.cancel_order(CancelOrder("high-bid", high_bid.order_id))
    episode._publish(cancelled.events)
    submit(NewOrder("low-bid", "bid", Side.BUY, 50, 3))
    submit(NewOrder("low-ask", "ask", Side.SELL, 60, 3))

    result = session.wait(duration=0, interrupt_on_alert=True)

    assert isinstance(result, WaitResult)
    assert result.interrupted_by_alert
    alert = session.alerts()[0].envelope
    assert alert.payload["code"] == "margin"
    assert alert.payload["data"]["state"] == "margin_call"


def test_event_storm_faults_automation_without_stranding_episode() -> None:
    class SameTimeTimerLoop(Strategy):
        def on_start(self, ctx, event):
            return [ctx.set_timer("loop", fire_at=ctx.now)]

        def on_timer(self, ctx, event):
            return [ctx.set_timer("loop", fire_at=ctx.now)]

    session = PlayerSession(
        Episode(),
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="direct-api:v1",
            account_starting_cash=10_000,
        ),
        simulation_step_ns=10,
        max_scheduled_events_per_step=100,
    )
    session.deploy_strategy(SameTimeTimerLoop())

    assert session.wait(duration=1) == 1
    status = session.strategy_status()
    assert status["active"] is False
    assert "scheduled-event limit exceeded" in str(status["fault"])
