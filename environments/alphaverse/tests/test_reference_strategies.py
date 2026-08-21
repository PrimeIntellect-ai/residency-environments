from __future__ import annotations

from alphaverse.models import Side
from alphaverse.reference_strategies import (
    AdaptiveMarketMaker,
    EventDrivenAdaptiveMarketMaker,
    InventoryAwareMarketMaker,
    LatentValueTrader,
    PersistentNoiseTrader,
    ReservationDemandTrader,
    RollingNoiseExecutor,
    ScheduledNoiseExecutor,
)
from alphaverse.strategy import (
    CancelOrderAction,
    EmitLog,
    InputEnvelope,
    InputKind,
    RequestStop,
    SetTimer,
    StrategyContext,
    StrategyRunner,
    SubmitLimitOrder,
)
from alphaverse.world import ExecutionStyle, LatentValueProcess, ParentOrder


class CohortProcess:
    def __init__(self, shifts: dict[int, int] | None = None) -> None:
        self.shifts = shifts or {}

    def target_shift(self, cohort_id: str, at_time: int) -> int:
        assert cohort_id == "slow"
        eligible = [time for time in self.shifts if time <= at_time]
        return self.shifts[max(eligible)] if eligible else 0


def reservation_trader(
    process: CohortProcess,
    **overrides,
) -> ReservationDemandTrader:
    parameters = {
        "demand_process": process,
        "cohort_id": "slow",
        "initial_anchor": 100,
        "update_interval": 10,
        "anchor_half_life": 1_000,
        "price_elasticity": 1,
        "base_target_position": 0,
        "target_loading": 1,
        "clip_quantity": 4,
    }
    parameters.update(overrides)
    return ReservationDemandTrader(**parameters)


def event(
    kind: InputKind,
    *,
    at: int,
    payload: dict[str, object] | None = None,
    number: int = 1,
) -> InputEnvelope:
    return InputEnvelope(
        session_id="S1",
        strategy_instance_id="I1",
        event_id=f"E{number}",
        kind=kind,
        exchange_time=at,
        available_at=at,
        source_event_seq=number,
        payload=payload or {},  # type: ignore[arg-type]
    )


def runner_for(strategy) -> StrategyRunner:
    return StrategyRunner(
        strategy,
        StrategyContext(participant_id="participant", strategy_instance_id="I1"),
    )


def levels(bid: int = 99, ask: int = 101) -> dict[str, object]:
    return {
        "bids": [{"price": bid, "quantity": 20, "order_count": 2}],
        "asks": [{"price": ask, "quantity": 30, "order_count": 3}],
    }


def test_market_maker_quotes_deterministically_and_reschedules() -> None:
    maker = InventoryAwareMarketMaker(
        refresh_interval=10,
        quote_quantity=5,
        base_half_spread=2,
        volatility_multiplier=1,
        inventory_skew_per_unit=1,
    )
    runner = runner_for(maker)

    start = runner.handle(event(InputKind.START, at=0))
    observed = runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    quoted = runner.handle(event(InputKind.TIMER, at=10, payload={"timer_id": "quote-refresh"}))

    assert start.actions == (SetTimer("quote-refresh", 10),)
    assert observed.actions == ()
    assert quoted.actions == (
        SubmitLimitOrder("mm-bid-1", Side.BUY, 98, 5),
        SubmitLimitOrder("mm-ask-1", Side.SELL, 102, 5),
        SetTimer("quote-refresh", 20),
    )


def test_market_maker_tracks_acknowledgements_and_skews_for_long_inventory() -> None:
    maker = InventoryAwareMarketMaker(
        refresh_interval=10,
        quote_quantity=5,
        base_half_spread=2,
        volatility_multiplier=0,
        inventory_skew_per_unit=1,
    )
    runner = runner_for(maker)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    runner.handle(event(InputKind.TIMER, at=10, payload={"timer_id": "quote-refresh"}))
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=11,
            payload={
                "event_kind": "order_accepted",
                "client_order_id": "mm-bid-1",
                "order_id": "O10",
            },
        )
    )
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=11,
            payload={
                "event_kind": "order_accepted",
                "client_order_id": "mm-ask-1",
                "order_id": "O11",
                "position": 2,
            },
        )
    )
    runner.handle(event(InputKind.LEVELS, at=12, payload=levels()))

    replacement = runner.handle(event(InputKind.TIMER, at=20, payload={"timer_id": "quote-refresh"}))

    assert maker.position == 2
    assert replacement.actions == (
        CancelOrderAction("O10"),
        CancelOrderAction("O11"),
        SubmitLimitOrder("mm-bid-2", Side.BUY, 96, 5),
        SubmitLimitOrder("mm-ask-2", Side.SELL, 100, 5),
        SetTimer("quote-refresh", 30),
    )


def test_market_maker_widens_with_recent_midpoint_volatility() -> None:
    maker = InventoryAwareMarketMaker(
        refresh_interval=5,
        quote_quantity=2,
        base_half_spread=1,
        volatility_multiplier=1,
        volatility_lookback=2,
    )
    runner = runner_for(maker)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels(99, 101)))
    runner.handle(event(InputKind.LEVELS, at=2, payload=levels(102, 104)))

    batch = runner.handle(event(InputKind.TIMER, at=5, payload={"timer_id": "quote-refresh"}))

    # Mid moved three ticks: half-spread is base 1 + widening 3 around mid 103.
    assert batch.actions[:2] == (
        SubmitLimitOrder("mm-bid-1", Side.BUY, 99, 2),
        SubmitLimitOrder("mm-ask-1", Side.SELL, 107, 2),
    )


def adaptive_maker(**overrides) -> AdaptiveMarketMaker:
    parameters = {
        "refresh_interval": 10,
        "initial_refresh_delay": 5,
        "quote_quantity": 10,
        "base_half_spread": 2,
        "volatility_multiplier": 0,
        "inventory_skew_per_unit": 0.1,
        "inventory_soft_limit": 20,
        "minimum_quote_fraction": 0.25,
        "fill_pressure_per_unit": 0.1,
        "fill_pressure_half_life": 10,
        "maximum_fill_skew": 4,
        "markout_horizon": 5,
        "markout_learning_rate": 1,
        "toxicity_widening_multiplier": 1,
        "toxicity_half_life": 20,
        "maximum_toxicity_widening": 6,
        "directional_toxicity": True,
    }
    parameters.update(overrides)
    return AdaptiveMarketMaker(**parameters)


def event_maker(**overrides) -> EventDrivenAdaptiveMarketMaker:
    parameters = {
        "refresh_interval": 10,
        "initial_refresh_delay": 5,
        "quote_quantity": 10,
        "base_half_spread": 2,
        "volatility_multiplier": 0,
        "inventory_skew_per_unit": 0,
        "inventory_soft_limit": 20,
        "minimum_quote_fraction": 0.25,
        "fill_pressure_per_unit": 0,
        "fill_pressure_half_life": 10,
        "maximum_fill_skew": 4,
        "markout_horizon": 5,
        "markout_learning_rate": 1,
        "toxicity_widening_multiplier": 1,
        "toxicity_half_life": 20,
        "maximum_toxicity_widening": 6,
        "directional_toxicity": True,
        "improvement_reprice_ticks": 2,
        "retreat_reprice_ticks": 1,
        "external_book_reference": True,
        "replenish_partial_fills": False,
    }
    parameters.update(overrides)
    return EventDrivenAdaptiveMarketMaker(**parameters)


def acknowledge_event_quotes(
    runner: StrategyRunner,
    quoted,
    *,
    at: int,
) -> None:
    for index, action in enumerate(quoted.actions, start=1):
        if not isinstance(action, SubmitLimitOrder):
            continue
        runner.handle(
            event(
                InputKind.EXECUTION,
                at=at,
                number=100 + index,
                payload={
                    "event_kind": "order_accepted",
                    "client_order_id": action.client_order_id,
                    "order_id": f"O{index}",
                    "side": action.side.value,
                    "price": action.price,
                    "quantity": action.quantity,
                },
            )
        )


def test_event_maker_quotes_on_first_coherent_level_update() -> None:
    maker = event_maker()
    runner = runner_for(maker)

    assert runner.handle(event(InputKind.START, at=0)).actions == (SetTimer("adaptive-quote-refresh", 5),)
    quoted = runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))

    assert quoted.actions == (
        SubmitLimitOrder("event-mm-bid-1", Side.BUY, 98, 10),
        SubmitLimitOrder("event-mm-ask-2", Side.SELL, 102, 10),
    )


def test_event_maker_excludes_own_quotes_from_reference_book() -> None:
    maker = event_maker()
    runner = runner_for(maker)
    runner.handle(event(InputKind.START, at=0))
    quoted = runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    acknowledge_event_quotes(runner, quoted, at=1)

    own_and_external = {
        "bids": [
            {"price": 98, "quantity": 10, "order_count": 1},
            {"price": 97, "quantity": 20, "order_count": 2},
        ],
        "asks": [
            {"price": 102, "quantity": 10, "order_count": 1},
            {"price": 103, "quantity": 20, "order_count": 2},
        ],
    }
    reassessed = runner.handle(event(InputKind.LEVELS, at=2, payload=own_and_external))

    assert reassessed.actions == ()
    assert maker._latest_midpoint == 100


def test_event_maker_retreats_immediately_but_improves_with_hysteresis() -> None:
    maker = event_maker()
    runner = runner_for(maker)
    runner.handle(event(InputKind.START, at=0))
    quoted = runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    acknowledge_event_quotes(runner, quoted, at=1)

    moved_up_one_tick = runner.handle(
        event(
            InputKind.LEVELS,
            at=2,
            payload=levels(bid=100, ask=102),
        )
    )

    # The old ask is now too aggressive and retreats by one tick.  The bid
    # would improve by only one tick, so it keeps its queue position.
    assert moved_up_one_tick.actions == (
        CancelOrderAction("O2"),
        SubmitLimitOrder("event-mm-ask-3", Side.SELL, 103, 10),
    )


def test_event_maker_does_not_top_up_partial_fill() -> None:
    maker = event_maker()
    runner = runner_for(maker)
    runner.handle(event(InputKind.START, at=0))
    quoted = runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    acknowledge_event_quotes(runner, quoted, at=1)
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=2,
            payload={
                "event_kind": "fill",
                "order_id": "O2",
                "side": "sell",
                "price": 102,
                "quantity": 4,
                "position": -4,
            },
        )
    )

    reassessed = runner.handle(event(InputKind.LEVELS, at=2, payload=levels()))

    assert reassessed.actions == ()


def test_adaptive_maker_requotes_immediately_after_fill() -> None:
    maker = adaptive_maker()
    runner = runner_for(maker)

    assert runner.handle(event(InputKind.START, at=0)).actions == (SetTimer("adaptive-quote-refresh", 5),)
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    initial = runner.handle(
        event(
            InputKind.TIMER,
            at=5,
            payload={"timer_id": "adaptive-quote-refresh"},
        )
    )
    assert initial.actions[:2] == (
        SubmitLimitOrder("adaptive-mm-bid-1", Side.BUY, 98, 10),
        SubmitLimitOrder("adaptive-mm-ask-1", Side.SELL, 102, 10),
    )
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=6,
            payload={
                "event_kind": "order_accepted",
                "client_order_id": "adaptive-mm-bid-1",
                "order_id": "O1",
            },
        )
    )
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=6,
            payload={
                "event_kind": "order_accepted",
                "client_order_id": "adaptive-mm-ask-1",
                "order_id": "O2",
            },
        )
    )

    filled = runner.handle(
        event(
            InputKind.EXECUTION,
            at=7,
            payload={
                "event_kind": "fill",
                "order_id": "O2",
                "side": "sell",
                "price": 102,
                "quantity": 4,
            },
        )
    )
    assert filled.actions == (SetTimer("adaptive-fill-requote", 7),)
    assert maker.position == -4
    assert maker.fill_pressure == 4

    replacement = runner.handle(
        event(
            InputKind.TIMER,
            at=7,
            payload={"timer_id": "adaptive-fill-requote"},
        )
    )
    assert replacement.actions[:4] == (
        CancelOrderAction("O1"),
        CancelOrderAction("O2"),
        SubmitLimitOrder("adaptive-mm-bid-2", Side.BUY, 98, 10),
        SubmitLimitOrder("adaptive-mm-ask-2", Side.SELL, 103, 9),
    )
    assert replacement.actions[-1] == SubmitLimitOrder("adaptive-mm-ask-2", Side.SELL, 103, 9)

    periodic = runner.handle(
        event(
            InputKind.TIMER,
            at=15,
            payload={"timer_id": "adaptive-quote-refresh"},
        )
    )
    assert periodic.actions[-1] == SetTimer("adaptive-quote-refresh", 25)


def test_adaptive_maker_coalesces_fill_requotes_without_forking_refresh_chain() -> None:
    maker = adaptive_maker()
    runner = runner_for(maker)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    runner.handle(
        event(
            InputKind.TIMER,
            at=5,
            payload={"timer_id": "adaptive-quote-refresh"},
        )
    )

    first = runner.handle(
        event(
            InputKind.EXECUTION,
            at=7,
            payload={
                "event_kind": "fill",
                "order_id": "O1",
                "side": "buy",
                "price": 98,
                "quantity": 2,
            },
        )
    )
    second = runner.handle(
        event(
            InputKind.EXECUTION,
            at=7,
            payload={
                "event_kind": "fill",
                "order_id": "O1",
                "side": "buy",
                "price": 98,
                "quantity": 2,
            },
        )
    )

    assert first.actions == (SetTimer("adaptive-fill-requote", 7),)
    assert second.actions == ()
    immediate = runner.handle(
        event(
            InputKind.TIMER,
            at=7,
            payload={"timer_id": "adaptive-fill-requote"},
        )
    )
    assert all(not isinstance(action, SetTimer) for action in immediate.actions)


def test_adaptive_maker_learns_adverse_fill_markout() -> None:
    maker = adaptive_maker(
        inventory_skew_per_unit=0,
        fill_pressure_per_unit=0,
    )
    runner = runner_for(maker)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=2,
            payload={
                "event_kind": "fill",
                "order_id": "O1",
                "side": "buy",
                "price": 99,
                "quantity": 10,
            },
        )
    )
    runner.handle(event(InputKind.LEVELS, at=6, payload=levels(94, 96)))
    assert maker.adverse_markout == 0
    runner.handle(event(InputKind.LEVELS, at=7, payload=levels(94, 96)))
    assert maker.adverse_markout == 4
    assert maker.buy_adverse_markout == 4
    assert maker.sell_adverse_markout == 0

    widened = runner.handle(
        event(
            InputKind.TIMER,
            at=7,
            payload={"timer_id": "adaptive-quote-refresh"},
        )
    )
    assert widened.actions[:2] == (
        SubmitLimitOrder("adaptive-mm-bid-1", Side.BUY, 89, 7),
        SubmitLimitOrder("adaptive-mm-ask-1", Side.SELL, 97, 10),
    )


def test_adaptive_maker_can_apply_symmetric_toxicity_widening() -> None:
    maker = adaptive_maker(
        inventory_skew_per_unit=0,
        fill_pressure_per_unit=0,
        directional_toxicity=False,
    )
    runner = runner_for(maker)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=2,
            payload={
                "event_kind": "fill",
                "order_id": "O1",
                "side": "buy",
                "price": 99,
                "quantity": 10,
            },
        )
    )
    runner.handle(event(InputKind.LEVELS, at=7, payload=levels(94, 96)))

    widened = runner.handle(
        event(
            InputKind.TIMER,
            at=7,
            payload={"timer_id": "adaptive-quote-refresh"},
        )
    )
    assert widened.actions[:2] == (
        SubmitLimitOrder("adaptive-mm-bid-1", Side.BUY, 89, 7),
        SubmitLimitOrder("adaptive-mm-ask-1", Side.SELL, 101, 10),
    )


def parent(
    *,
    side: Side = Side.BUY,
    style: ExecutionStyle = ExecutionStyle.SCHEDULED,
) -> ParentOrder:
    return ParentOrder("noise-1", side, 10, 10, 50, style)


def test_noise_executor_slices_passively_and_reschedules_to_deadline() -> None:
    strategy = ScheduledNoiseExecutor(parent(), slice_interval=10, max_slice_quantity=4)
    runner = runner_for(strategy)

    assert runner.handle(event(InputKind.START, at=0)).actions == (SetTimer("parent-slice", 10),)
    runner.handle(event(InputKind.LEVELS, at=5, payload=levels()))

    first = runner.handle(event(InputKind.TIMER, at=10, payload={"timer_id": "parent-slice"}))
    second = runner.handle(event(InputKind.TIMER, at=20, payload={"timer_id": "parent-slice"}))
    third = runner.handle(event(InputKind.TIMER, at=30, payload={"timer_id": "parent-slice"}))

    assert first.actions == (
        SubmitLimitOrder("noise-slice-1", Side.BUY, 99, 4),
        SetTimer("parent-slice", 20),
    )
    assert second.actions[0] == SubmitLimitOrder("noise-slice-2", Side.BUY, 99, 4)
    assert third.actions == (
        SubmitLimitOrder("noise-slice-3", Side.BUY, 99, 2),
        SetTimer("parent-slice", 40),
    )
    assert strategy.submitted_quantity == 10


def test_noise_executor_uses_marketable_side_for_aggressive_style() -> None:
    strategy = ScheduledNoiseExecutor(
        parent(side=Side.SELL, style=ExecutionStyle.BURST),
        slice_interval=10,
        max_slice_quantity=3,
    )
    runner = runner_for(strategy)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=5, payload=levels()))

    batch = runner.handle(event(InputKind.TIMER, at=10, payload={"timer_id": "parent-slice"}))

    assert batch.actions[0] == SubmitLimitOrder("noise-slice-1", Side.SELL, 99, 3)


def test_noise_executor_tracks_own_fills_and_stops_when_complete() -> None:
    strategy = ScheduledNoiseExecutor(parent(), slice_interval=10, max_slice_quantity=10)
    runner = runner_for(strategy)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=5, payload=levels()))
    runner.handle(event(InputKind.TIMER, at=10, payload={"timer_id": "parent-slice"}))
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=10,
            payload={
                "event_kind": "order_accepted",
                "client_order_id": "noise-slice-1",
                "order_id": "O1",
            },
        )
    )

    partial = runner.handle(
        event(
            InputKind.EXECUTION,
            at=11,
            payload={
                "event_kind": "fill",
                "order_id": "O1",
                "quantity": 4,
            },
        )
    )
    complete = runner.handle(
        event(
            InputKind.EXECUTION,
            at=12,
            payload={
                "event_kind": "fill",
                "order_id": "O1",
                "quantity": 6,
            },
        )
    )

    assert partial.actions == ()
    assert strategy.filled_quantity == 10
    assert complete.actions == (RequestStop("parent_order_complete"),)


def test_noise_executor_stops_at_end_and_patient_crosses_near_deadline() -> None:
    strategy = ScheduledNoiseExecutor(
        parent(style=ExecutionStyle.PATIENT),
        slice_interval=10,
        max_slice_quantity=3,
    )
    runner = runner_for(strategy)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=5, payload=levels()))

    near_end = runner.handle(event(InputKind.TIMER, at=40, payload={"timer_id": "parent-slice"}))
    expired = runner.handle(event(InputKind.TIMER, at=50, payload={"timer_id": "parent-slice"}))

    assert near_end.actions[0] == SubmitLimitOrder("noise-slice-1", Side.BUY, 101, 3)
    assert near_end.actions[-1] == SetTimer("parent-slice", 50)
    assert expired.actions == (RequestStop("parent_order_expired"),)


def test_rolling_noise_executor_reuses_one_actor_for_successive_mandates() -> None:
    first = ParentOrder("noise-001", Side.BUY, 4, 10, 20, ExecutionStyle.BURST)
    second = ParentOrder("noise-001", Side.SELL, 4, 30, 40, ExecutionStyle.BURST)
    strategy = RollingNoiseExecutor((first, second), slice_interval=5, max_slice_quantity=4)
    runner = runner_for(strategy)

    assert runner.handle(event(InputKind.START, at=0)).actions == (SetTimer("rolling-parent-slice", 10),)
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    first_slice = runner.handle(event(InputKind.TIMER, at=10, payload={"timer_id": "rolling-parent-slice"}))
    transition = runner.handle(event(InputKind.TIMER, at=20, payload={"timer_id": "rolling-parent-slice"}))
    second_slice = runner.handle(event(InputKind.TIMER, at=30, payload={"timer_id": "rolling-parent-slice"}))

    assert first_slice.actions[0] == SubmitLimitOrder("rolling-noise-1-1", Side.BUY, 101, 4)
    assert transition.actions[-1] == SetTimer("rolling-parent-slice", 30)
    assert second_slice.actions[0] == SubmitLimitOrder("rolling-noise-2-2", Side.SELL, 99, 4)
    assert not any(
        isinstance(action, RequestStop) for batch in (first_slice, transition, second_slice) for action in batch.actions
    )


def test_rolling_noise_executor_clears_filled_slices_and_cancels_open_ones() -> None:
    first = ParentOrder("noise-001", Side.BUY, 4, 10, 20, ExecutionStyle.BURST)
    second = ParentOrder("noise-001", Side.SELL, 4, 30, 40, ExecutionStyle.BURST)
    strategy = RollingNoiseExecutor((first, second), slice_interval=5, max_slice_quantity=4)
    runner = runner_for(strategy)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    runner.handle(event(InputKind.TIMER, at=10, payload={"timer_id": "rolling-parent-slice"}))
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=11,
            payload={
                "event_kind": "order_accepted",
                "client_order_id": "rolling-noise-1-1",
                "order_id": "O1",
            },
        )
    )
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=12,
            payload={"event_kind": "fill", "order_id": "O1", "quantity": 4},
        )
    )
    completed_first = runner.handle(event(InputKind.TIMER, at=15, payload={"timer_id": "rolling-parent-slice"}))
    assert completed_first.actions == (SetTimer("rolling-parent-slice", 30),)

    runner.handle(event(InputKind.TIMER, at=30, payload={"timer_id": "rolling-parent-slice"}))
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=31,
            payload={
                "event_kind": "order_accepted",
                "client_order_id": "rolling-noise-2-2",
                "order_id": "O2",
            },
        )
    )
    retired_second = runner.handle(event(InputKind.TIMER, at=40, payload={"timer_id": "rolling-parent-slice"}))

    assert retired_second.actions == (CancelOrderAction("O2"),)
    assert (
        runner.handle(
            event(
                InputKind.EXECUTION,
                at=41,
                payload={"event_kind": "cancel_accepted", "order_id": "O2"},
            )
        ).actions
        == ()
    )


def test_persistent_noise_trader_reschedules_and_submits_marketably() -> None:
    strategy = PersistentNoiseTrader(
        min_interval=10,
        max_interval=10,
        min_quantity=3,
        max_quantity=3,
        aggressive_probability=1,
        side_persistence=1,
        soft_inventory_limit=20,
    )
    runner = runner_for(strategy)

    assert runner.handle(event(InputKind.START, at=0)).actions == (SetTimer("noise-arrival", 10),)
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    first = runner.handle(event(InputKind.TIMER, at=10, payload={"timer_id": "noise-arrival"}))
    second = runner.handle(event(InputKind.TIMER, at=20, payload={"timer_id": "noise-arrival"}))

    assert isinstance(first.actions[0], SubmitLimitOrder)
    assert first.actions[0].quantity == 3
    assert first.actions[0].price in (99, 101)
    assert first.actions[-1] == SetTimer("noise-arrival", 20)
    assert second.actions[0].side is first.actions[0].side
    assert second.actions[-1] == SetTimer("noise-arrival", 30)


def test_persistent_noise_trader_mean_reverts_at_inventory_limit() -> None:
    strategy = PersistentNoiseTrader(
        min_interval=5,
        max_interval=5,
        min_quantity=2,
        max_quantity=2,
        aggressive_probability=1,
        side_persistence=1,
        soft_inventory_limit=10,
    )
    runner = runner_for(strategy)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=2,
            payload={"event_kind": "account", "position": 10},
        )
    )

    batch = runner.handle(event(InputKind.TIMER, at=5, payload={"timer_id": "noise-arrival"}))
    assert batch.actions[0] == SubmitLimitOrder(
        "recurring-noise-1",
        Side.SELL,
        99,
        2,
    )


def test_latent_value_trader_crosses_book_toward_private_value() -> None:
    strategy = LatentValueTrader(
        initial_value=100,
        update_interval=10,
        order_quantity=7,
        normal_step=0,
        shock_probability=1,
        shock_size=5,
    )
    runner = StrategyRunner(
        strategy,
        StrategyContext(
            participant_id="informed",
            strategy_instance_id="I1",
            seed=0,
        ),
    )
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))

    batch = runner.handle(event(InputKind.TIMER, at=10, payload={"timer_id": "latent-value-update"}))

    assert batch.actions[0] == SubmitLimitOrder(
        "latent-value-1",
        Side.BUY,
        105,
        7,
    )
    assert isinstance(batch.actions[-2], EmitLog)
    assert batch.actions[-2].message == "informed_decision"
    assert batch.actions[-2].fields["participant_id"] == "informed"
    assert batch.actions[-2].fields["side"] == "buy"
    assert batch.actions[-2].fields["execution_model"] == ("single_marketable_limit")
    assert batch.actions[-1] == SetTimer("latent-value-update", 20)


def test_latent_value_traders_observe_subsets_of_one_shared_path() -> None:
    process = LatentValueProcess(
        seed=4,
        normal_step=0,
        shock_probability=1,
        shock_size=5,
    )
    fully_informed = LatentValueTrader(
        initial_value=100,
        update_interval=10,
        order_quantity=7,
        normal_step=0,
        shock_probability=1,
        shock_size=5,
        signal_process=process,
        signal_observation_probability=1,
    )
    uninformed = LatentValueTrader(
        initial_value=100,
        update_interval=10,
        order_quantity=7,
        normal_step=0,
        shock_probability=1,
        shock_size=5,
        signal_process=process,
        signal_observation_probability=0,
    )
    informed_runner = runner_for(fully_informed)
    uninformed_runner = runner_for(uninformed)
    informed_runner.handle(event(InputKind.START, at=0))
    uninformed_runner.handle(event(InputKind.START, at=0))

    for market_time in (10, 20, 30):
        timer = event(
            InputKind.TIMER,
            at=market_time,
            payload={"timer_id": "latent-value-update"},
        )
        informed_runner.handle(timer)
        uninformed_runner.handle(timer)

    assert fully_informed.latent_value == uninformed.latent_value
    assert fully_informed.fair_value == fully_informed.latent_value
    assert uninformed.fair_value == 100


def test_proportional_signal_noise_can_create_strong_wrong_side_alpha() -> None:
    strategy = LatentValueTrader(
        initial_value=100,
        update_interval=10,
        order_quantity=7,
        normal_step=0,
        shock_probability=1,
        shock_size=8,
        signal_process=LatentValueProcess(
            seed=11,
            normal_step=0,
            shock_probability=1,
            shock_size=8,
        ),
        signal_observation_probability=1,
        signal_proportional_noise_multiplier=2,
    )
    runner = runner_for(strategy)
    runner.handle(event(InputKind.START, at=0))

    strong_wrong_updates = []
    for update_index in range(1, 51):
        runner.handle(
            event(
                InputKind.TIMER,
                at=10 * update_index,
                payload={"timer_id": "latent-value-update"},
                number=update_index,
            )
        )
        if strategy.last_private_innovation * strategy.last_latent_innovation < 0 and abs(
            strategy.last_private_innovation
        ) >= abs(strategy.last_latent_innovation):
            strong_wrong_updates.append(update_index)

    assert strong_wrong_updates


def test_reservation_demand_opposes_an_isolated_price_displacement() -> None:
    trader = reservation_trader(CohortProcess())
    runner = runner_for(trader)
    assert runner.handle(event(InputKind.START, at=0)).actions == (SetTimer("reservation-demand-update", 10),)
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels(109, 111)))

    decision = runner.handle(
        event(
            InputKind.TIMER,
            at=10,
            payload={"timer_id": "reservation-demand-update"},
        )
    )

    assert decision.actions[0] == SubmitLimitOrder("reservation-demand-1", Side.SELL, 101, 4)
    assert isinstance(decision.actions[-2], EmitLog)
    assert decision.actions[-2].message == "reservation_demand_decision"
    assert decision.actions[-2].fields["desired_position"] == -10
    assert decision.actions[-1] == SetTimer("reservation-demand-update", 20)


def test_reservation_demand_cohort_shift_changes_direction_and_persists() -> None:
    process = CohortProcess({10: 6})
    trader = reservation_trader(process)
    runner = runner_for(trader)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))

    first = runner.handle(
        event(
            InputKind.TIMER,
            at=10,
            payload={"timer_id": "reservation-demand-update"},
        )
    )
    second = runner.handle(
        event(
            InputKind.TIMER,
            at=20,
            payload={"timer_id": "reservation-demand-update"},
        )
    )

    assert first.actions[0] == SubmitLimitOrder("reservation-demand-1", Side.BUY, 106, 4)
    assert second.actions[0] == SubmitLimitOrder("reservation-demand-2", Side.BUY, 106, 4)
    assert second.actions[-2].fields["target_shift"] == 6
    assert second.actions[-2].fields["desired_position"] == 6


def test_reservation_demand_anchor_adapts_at_its_configured_half_life() -> None:
    trader = reservation_trader(
        CohortProcess(),
        anchor_half_life=10,
    )
    runner = runner_for(trader)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels(109, 111)))

    runner.handle(
        event(
            InputKind.TIMER,
            at=10,
            payload={"timer_id": "reservation-demand-update"},
        )
    )
    assert trader.anchor == 105
    runner.handle(
        event(
            InputKind.TIMER,
            at=20,
            payload={"timer_id": "reservation-demand-update"},
        )
    )
    assert trader.anchor == 107.5


def test_reservation_demand_tracks_order_lifecycle_and_replaces_child() -> None:
    trader = reservation_trader(CohortProcess({10: 5}))
    runner = runner_for(trader)
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    first = runner.handle(
        event(
            InputKind.TIMER,
            at=10,
            payload={"timer_id": "reservation-demand-update"},
        )
    )
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=11,
            payload={
                "event_kind": "order_accepted",
                "client_order_id": "reservation-demand-1",
                "order_id": "O1",
            },
        )
    )

    replacement = runner.handle(
        event(
            InputKind.TIMER,
            at=20,
            payload={"timer_id": "reservation-demand-update"},
        )
    )
    assert first.actions[0] == SubmitLimitOrder("reservation-demand-1", Side.BUY, 105, 4)
    assert replacement.actions[:2] == (
        CancelOrderAction("O1"),
        SubmitLimitOrder("reservation-demand-2", Side.BUY, 105, 4),
    )
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=21,
            payload={"event_kind": "cancel_accepted", "order_id": "O1"},
        )
    )
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=21,
            payload={
                "event_kind": "order_accepted",
                "client_order_id": "reservation-demand-2",
                "order_id": "O2",
            },
        )
    )
    runner.handle(
        event(
            InputKind.EXECUTION,
            at=22,
            payload={
                "event_kind": "fill",
                "order_id": "O2",
                "side": "buy",
                "quantity": 4,
            },
        )
    )

    assert trader.position == 4
    assert trader._owned == {}
