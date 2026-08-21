from __future__ import annotations

import json

import pytest
from alphaverse.episode import Episode
from alphaverse.models import CancelOrder, NewOrder, Side
from alphaverse.observability import build_observability_snapshot, strategy_family
from alphaverse.player import PlayerSession
from alphaverse.profiles import ParticipantSpec
from alphaverse.strategy import InputEnvelope, Strategy, StrategyContext


def _spec(participant_id: str, *, starting_cash: int = 10_000) -> ParticipantSpec:
    return ParticipantSpec(
        participant_id=participant_id,
        strategy_version_id=f"test:{participant_id}",
        account_starting_cash=starting_cash,
    )


class _Quotes(Strategy):
    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return (
            ctx.submit_limit("bid", Side.BUY, 99, 4),
            ctx.submit_limit("ask", Side.SELL, 102, 5),
        )


class _Buyer(Strategy):
    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.submit_limit("cross", Side.BUY, 102, 2)]


class _BidOnly(Strategy):
    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.submit_limit("bid", Side.BUY, 99, 1)]


class _AskOnly(Strategy):
    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.submit_limit("ask", Side.SELL, 101, 1)]


def _traded_episode() -> Episode:
    episode = Episode(session_id="observe-test")
    episode.add_strategy(_spec("maker"), _Quotes())
    episode.add_strategy(_spec("buyer"), _Buyer())
    episode.run_until(0)
    return episode


def test_snapshot_is_json_ready_and_derived_from_canonical_state() -> None:
    episode = _traded_episode()

    snapshot = build_observability_snapshot(episode, depth=1)

    assert json.loads(json.dumps(snapshot)) == snapshot
    assert snapshot["session_id"] == "observe-test"
    assert snapshot["market_time"] == 0
    assert snapshot["event_sequence"] == episode.exchange.event_log.last_sequence
    assert snapshot["product"] == {
        "product_id": "ALPHA",
        "tick_value": 1,
        "contract_multiplier": 1,
        "cash_subunits_per_tick": 100,
        "transaction_fee_per_contract_subunits": 5,
        "transaction_fee_per_contract": 0.05,
    }
    assert snapshot["book"] == {
        "depth": 1,
        "bids": [{"price": 99, "quantity": 4, "order_count": 1}],
        "asks": [{"price": 102, "quantity": 3, "order_count": 1}],
        "best_bid": 99,
        "best_ask": 102,
        "mid_price": 100.5,
        "spread": 3,
        "mark_status": "midpoint",
    }
    assert snapshot["recent_trades"] == [
        {
            "sequence": 8,
            "market_time": 0,
            "match_event_id": "M3",
            "trade_id": "T1",
            "price": 102,
            "quantity": 2,
            "aggressor_side": "buy",
            "maker_order_id": "O2",
            "taker_order_id": "O3",
        }
    ]
    assert len(snapshot["price_history"]) == 1
    assert snapshot["price_history"][0] == {
        "market_time": 0,
        "source_event_sequence": 14,
        "best_bid": 99,
        "best_ask": 102,
        "mid_price": 100.5,
        "spread": 3,
    }
    assert set(snapshot["price_history_windows"]) == {"2m", "10m", "30m", "1h"}
    assert snapshot["price_history_windows"]["2m"] == snapshot["price_history"]
    assert snapshot["recent_events"][-1]["sequence"] == snapshot["event_sequence"]

    participants = {item["participant_id"]: item for item in snapshot["participants"]}
    assert participants["buyer"]["cash"] == 9_795.9
    assert participants["buyer"]["fees_paid"] == 0.1
    assert participants["buyer"]["fees_paid_subunits"] == 10
    assert participants["buyer"]["position"] == 2
    assert participants["buyer"]["marked_equity"] == 9_996.9
    assert participants["buyer"]["pnl"] == -3.1
    assert participants["buyer"]["order_count"] == 1
    assert participants["buyer"]["fill_count"] == 1
    assert participants["buyer"]["gross_filled_quantity"] == 2
    assert participants["buyer"]["strategy_version_id"] == "test:buyer"
    assert participants["buyer"]["strategy_family"] == "strategy"
    assert strategy_family("reservation-demand-03") == "reservation_demand"
    assert participants["buyer"]["live_order_count"] == 0
    assert participants["maker"]["live_order_count"] == 2
    assert participants["maker"]["pnl"] == 2.9

    assert snapshot["totals"]["participant_count"] == 2
    assert snapshot["totals"]["active_participant_count"] == 2
    assert snapshot["totals"]["trade_count"] == 1
    assert snapshot["totals"]["traded_quantity"] == 2
    assert snapshot["totals"]["fill_event_count"] == 2
    assert snapshot["totals"]["net_position"] == 0
    assert snapshot["totals"]["total_fees_paid"] == 0.2
    assert snapshot["totals"]["total_fees_paid_subunits"] == 20
    assert snapshot["totals"]["total_pnl"] == -0.2


def test_player_session_identifies_focal_participant_without_using_delayed_feed() -> None:
    episode = Episode()
    session = PlayerSession(episode, _spec("focal"))

    snapshot = build_observability_snapshot(session)

    assert snapshot["focal_participant_id"] == "focal"
    assert snapshot["participants"][0]["participant_id"] == "focal"


@pytest.mark.parametrize(
    ("strategy", "expected_status", "expected_bid", "expected_ask"),
    [
        (None, "empty", None, None),
        (_BidOnly(), "bid_only", 99, None),
        (_AskOnly(), "ask_only", None, 101),
    ],
)
def test_one_sided_and_empty_books_do_not_invent_a_midpoint_mark(
    strategy: Strategy | None,
    expected_status: str,
    expected_bid: int | None,
    expected_ask: int | None,
) -> None:
    episode = Episode()
    episode.add_strategy(_spec("participant"), strategy or Strategy())
    episode.run_until(0)

    snapshot = build_observability_snapshot(episode)

    assert snapshot["book"]["mark_status"] == expected_status
    assert snapshot["book"]["best_bid"] == expected_bid
    assert snapshot["book"]["best_ask"] == expected_ask
    assert snapshot["book"]["mid_price"] is None
    assert snapshot["book"]["spread"] is None
    assert snapshot["participants"][0]["marked_equity"] is None
    assert snapshot["participants"][0]["pnl"] is None
    assert snapshot["totals"]["total_marked_equity"] is None
    assert snapshot["totals"]["total_pnl"] is None


def test_projection_does_not_mutate_episode_and_limits_recent_trades() -> None:
    episode = _traded_episode()
    before_time = episode.now
    before_events = episode.exchange.event_log.to_jsonl()
    before_orders = episode.exchange.book.snapshot()

    snapshot = build_observability_snapshot(
        episode,
        recent_trade_limit=0,
        recent_event_limit=0,
        price_history_limit=0,
    )

    assert snapshot["recent_trades"] == []
    assert snapshot["recent_events"] == []
    assert snapshot["price_history"] == []
    assert snapshot["price_history_windows"] == {}
    assert episode.now == before_time
    assert episode.exchange.event_log.to_jsonl() == before_events
    assert episode.exchange.book.snapshot() == before_orders


def test_projection_streams_the_event_log_without_materializing_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = _traded_episode()
    log_type = type(episode.exchange.event_log)

    def fail_if_materialized(_log: object) -> tuple[object, ...]:
        raise AssertionError("terminal observability must not materialize all events")

    monkeypatch.setattr(log_type, "events", property(fail_if_materialized))

    snapshot = build_observability_snapshot(episode)

    assert snapshot["totals"]["event_count"] == len(episode.exchange.event_log)
    assert snapshot["totals"]["trade_count"] == 1
    assert snapshot["recent_trades"][0]["trade_id"] == "T1"


def test_price_history_uses_last_two_sided_observation_in_each_bucket() -> None:
    episode = Episode()
    episode.add_strategy(_spec("buyer"), Strategy())
    episode.add_strategy(_spec("seller"), Strategy())
    episode.run_until(0)
    episode.exchange.submit_order(NewOrder("buyer", "bid", Side.BUY, 99, 1), market_time=0)
    ask = episode.exchange.submit_order(NewOrder("seller", "ask", Side.SELL, 101, 1), market_time=0)
    assert ask.order_id is not None
    episode.exchange.cancel_order(CancelOrder("seller", ask.order_id), market_time=500_000_000)

    history = build_observability_snapshot(episode)["price_history"]

    assert len(history) == 1
    assert history[0]["market_time"] == 0
    assert history[0]["mid_price"] == 100


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"depth": -1}, ValueError),
        ({"depth": True}, TypeError),
        ({"recent_trade_limit": -1}, ValueError),
        ({"recent_event_limit": -1}, ValueError),
        ({"price_history_limit": -1}, ValueError),
        ({"price_history_bucket_ns": 0}, ValueError),
        ({"price_history_bucket_ns": True}, TypeError),
    ],
)
def test_projection_validates_limits(kwargs: dict[str, int], error: type[Exception]) -> None:
    with pytest.raises(error):
        build_observability_snapshot(Episode(), **kwargs)
