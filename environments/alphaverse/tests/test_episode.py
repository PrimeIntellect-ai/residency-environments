from __future__ import annotations

from alphaverse.episode import Episode
from alphaverse.models import CancelOrder, NewOrder, Side
from alphaverse.profiles import (
    MarginConfig,
    ParticipantSpec,
    RiskLimits,
    TechnologyProfile,
)
from alphaverse.strategy import InputEnvelope, Strategy, StrategyContext


def _spec(
    participant_id: str,
    *,
    technology: TechnologyProfile | None = None,
    risk: RiskLimits | None = None,
    margin: MarginConfig | None = None,
    starting_cash: int = 10_000,
) -> ParticipantSpec:
    return ParticipantSpec(
        participant_id=participant_id,
        strategy_version_id=f"test:{participant_id}",
        account_starting_cash=starting_cash,
        technology=technology or TechnologyProfile(),
        risk=risk or RiskLimits(),
        margin=margin,
    )


class TimerBuyer(Strategy):
    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.set_timer("enter", fire_at=100)]

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.submit_limit("buy", Side.BUY, 99, 2)]


def test_timer_and_decision_entry_latency_control_order_arrival() -> None:
    episode = Episode()
    episode.add_strategy(
        _spec(
            "buyer",
            technology=TechnologyProfile(decision_latency=7, order_entry_latency=3),
        ),
        TimerBuyer(),
    )

    episode.run_until(109)
    assert episode.exchange.book.live_orders == ()
    episode.run_until(110)

    order = episode.exchange.book.live_orders[0]
    assert order.participant_id == "buyer"
    assert order.price == 99
    accepted = next(event for event in episode.exchange.event_log if event.data.get("participant_id") == "buyer")
    assert accepted.market_time == 110


class RestingSeller(Strategy):
    def __init__(self) -> None:
        self.fills: list[dict] = []

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.submit_limit("ask", Side.SELL, 101, 3)]

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        if event.payload.get("event_kind") == "fill":
            self.fills.append(dict(event.payload))


class CrossingBuyer(Strategy):
    def __init__(self) -> None:
        self.fills: list[dict] = []

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.set_timer("cross", fire_at=1)]

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.submit_limit("buy", Side.BUY, 101, 2)]

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        if event.payload.get("event_kind") == "fill":
            self.fills.append(dict(event.payload))


def test_participants_trade_via_common_path_and_receive_private_fills() -> None:
    seller = RestingSeller()
    buyer = CrossingBuyer()
    episode = Episode()
    episode.add_strategy(_spec("seller"), seller)
    episode.add_strategy(_spec("buyer"), buyer)

    episode.run_until(1)

    assert episode.exchange.clearing.snapshot("seller").position == -2
    assert episode.exchange.clearing.snapshot("buyer").position == 2
    assert seller.fills[0]["liquidity_role"] == "maker"
    assert buyer.fills[0]["liquidity_role"] == "taker"


class LevelObserver(Strategy):
    def __init__(self) -> None:
        self.deliveries: list[tuple[int, int]] = []

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope):
        self.deliveries.append((event.exchange_time, ctx.now))


def test_level_feed_has_market_data_plus_derived_feed_latency() -> None:
    observer = LevelObserver()
    episode = Episode()
    episode.add_strategy(
        _spec(
            "observer",
            technology=TechnologyProfile(
                market_data_latency=5,
                level_feed_latency=11,
            ),
        ),
        observer,
    )
    episode.add_strategy(_spec("seller"), RestingSeller())

    episode.run_until(15)
    assert observer.deliveries == []
    episode.run_until(16)
    assert observer.deliveries == [(0, 16)]


class OversizedTrader(Strategy):
    def __init__(self) -> None:
        self.risk_reasons: list[str] = []

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.submit_limit("too-big", Side.BUY, 100, 3)]

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        if event.payload.get("event_kind") == "order_rejected":
            self.risk_reasons.append(str(event.payload["reason"]))


def test_external_risk_limit_rejects_strategy_action() -> None:
    strategy = OversizedTrader()
    episode = Episode()
    episode.add_strategy(
        _spec("trader", risk=RiskLimits(max_order_quantity=2)),
        strategy,
    )

    episode.run_until(0)

    assert episode.exchange.book.live_orders == ()
    assert strategy.risk_reasons == ["max_order_quantity"]


class MarginRejectedTrader(Strategy):
    def __init__(self) -> None:
        self.execution_reasons: list[str] = []

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.submit_limit("entry", Side.BUY, 100, 1)]

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        if event.payload.get("event_kind") == "order_rejected":
            self.execution_reasons.append(str(event.payload["reason"]))


def test_margin_rejection_is_an_ordinary_execution_event() -> None:
    strategy = MarginRejectedTrader()
    episode = Episode()
    episode.add_strategy(
        _spec(
            "trader",
            margin=MarginConfig(100, 80),
            starting_cash=50,
        ),
        strategy,
    )
    episode.run_until(0)

    assert strategy.execution_reasons == ["insufficient_initial_margin"]


class MarginRiskRecorder(Strategy):
    def __init__(self) -> None:
        self.states: list[str] = []

    def on_risk(self, ctx: StrategyContext, event: InputEnvelope):
        self.states.append(str(event.payload["state"]))


def test_episode_routes_margin_state_transitions_to_on_risk_and_snapshot() -> None:
    observer = MarginRiskRecorder()
    episode = Episode()
    episode.add_strategy(
        _spec(
            "focal",
            margin=MarginConfig(100, 90, grace_period=5),
            starting_cash=260,
        ),
        observer,
    )
    for participant_id in ("high-bid", "seller", "low-bid", "low-ask"):
        episode.add_strategy(_spec(participant_id), Strategy())
    episode.run_until(0)

    def submit(command: NewOrder):
        result = episode.exchange.submit_order(command)
        episode._publish(result.events)
        return result

    high_bid = submit(NewOrder("high-bid", "bid", Side.BUY, 99, 5))
    submit(NewOrder("seller", "offer", Side.SELL, 100, 2))
    submit(NewOrder("focal", "entry", Side.BUY, 100, 2))
    assert high_bid.order_id is not None
    cancelled = episode.exchange.cancel_order(CancelOrder("high-bid", high_bid.order_id))
    episode._publish(cancelled.events)
    submit(NewOrder("low-bid", "bid", Side.BUY, 50, 3))
    submit(NewOrder("low-ask", "ask", Side.SELL, 60, 3))
    episode.run_until(0)

    snapshot = episode.risk_snapshot("focal")
    assert observer.states == ["margin_call"]
    assert snapshot is not None
    assert snapshot.reduce_only
    assert snapshot.liquidation_deadline == 5

    episode.run_until(4)
    assert episode.exchange.clearing.snapshot("focal").position == 2
    episode.run_until(5)
    assert episode.exchange.clearing.snapshot("focal").position == 1
    assert episode.risk_snapshot("focal").state.value == "normal"  # type: ignore[union-attr]


class StopsImmediately(Strategy):
    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.stop("complete")]


def test_publisher_does_not_schedule_feeds_for_inactive_actors() -> None:
    episode = Episode()
    episode.add_strategy(_spec("stopped"), StopsImmediately())
    episode.add_strategy(_spec("active"), Strategy())
    episode.run_until(0)

    result = episode.exchange.submit_order(
        NewOrder("active", "bid", Side.BUY, 99, 1),
        market_time=0,
    )
    episode._publish(result.events)

    # The active actor receives its private acknowledgement plus MBO and level
    # messages. The stopped actor receives neither public feed message.
    assert len(episode.exchange.clock) == 3
