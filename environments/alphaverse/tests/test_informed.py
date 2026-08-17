from __future__ import annotations

from alphaverse.informed import FutureFlowInformedTrader
from alphaverse.models import Side
from alphaverse.strategy import (
    EmitLog,
    InputEnvelope,
    InputKind,
    SetTimer,
    StrategyContext,
    StrategyRunner,
    SubmitLimitOrder,
)
from alphaverse.world import (
    ExecutionStyle,
    ParentOrder,
    RegimeConfig,
    WorldGenerator,
)


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


def levels(bid: int = 99, ask: int = 101) -> dict[str, object]:
    return {
        "bids": [{"price": bid, "quantity": 20, "order_count": 2}],
        "asks": [{"price": ask, "quantity": 30, "order_count": 3}],
    }


def informed(style: ExecutionStyle) -> StrategyRunner:
    parent = ParentOrder(
        "noise-001",
        Side.BUY,
        90,
        10,
        100,
        ExecutionStyle.SCHEDULED,
    )
    world = WorldGenerator(
        1,
        RegimeConfig(
            participant_count=1,
            parent_order_count=0,
            start_time=0,
            end_time=200,
            min_duration=10,
            max_duration=20,
        ),
    )
    strategy = FutureFlowInformedTrader(
        world=world,
        parent_orders=(parent,),
        update_interval=10,
        signal_horizon=30,
        signal_loading=1,
        signal_observation_probability=1,
        signal_noise_multiplier=0,
        signal_noise_floor=0,
        minimum_signal=1,
        minimum_parent_quantity=4,
        maximum_parent_quantity=12,
        quantity_per_signal_unit=0.2,
        metaorder_duration=12,
        slice_interval=3,
        execution_style=style,
    )
    return StrategyRunner(
        strategy,
        StrategyContext(
            participant_id="informed-01",
            strategy_instance_id="I1",
            seed=7,
        ),
    )


def start_metaorder(runner: StrategyRunner):
    assert runner.handle(event(InputKind.START, at=0)).actions == (SetTimer("future-flow-signal", 10),)
    runner.handle(event(InputKind.LEVELS, at=5, payload=levels()))
    return runner.handle(
        event(
            InputKind.TIMER,
            at=10,
            payload={"timer_id": "future-flow-signal"},
        )
    )


def test_burst_submits_full_marketable_metaorder_and_logs_signal() -> None:
    batch = start_metaorder(informed(ExecutionStyle.BURST))

    submissions = [action for action in batch if isinstance(action, SubmitLimitOrder)]
    messages = [action.message for action in batch if isinstance(action, EmitLog)]
    assert submissions == [
        SubmitLimitOrder("informed-1-1", Side.BUY, 103, 6),
    ]
    assert {
        "informed_signal",
        "informed_metaorder_started",
        "informed_decision",
        "informed_child_submitted",
    } <= set(messages)
    child_log = next(
        action for action in batch if isinstance(action, EmitLog) and action.message == "informed_child_submitted"
    )
    assert child_log.fields["client_order_id"] == "informed-1-1"


def test_execution_styles_produce_distinct_initial_behavior() -> None:
    scheduled = start_metaorder(informed(ExecutionStyle.SCHEDULED))
    patient = start_metaorder(informed(ExecutionStyle.PATIENT))
    participation_runner = informed(ExecutionStyle.PARTICIPATION)
    participation = start_metaorder(participation_runner)
    momentum_runner = informed(ExecutionStyle.MOMENTUM)
    momentum = start_metaorder(momentum_runner)

    assert [action for action in scheduled if isinstance(action, SubmitLimitOrder)] == [
        SubmitLimitOrder("informed-1-1", Side.BUY, 103, 2)
    ]
    assert [action for action in patient if isinstance(action, SubmitLimitOrder)] == [
        SubmitLimitOrder("informed-1-1", Side.BUY, 99, 3)
    ]
    assert not any(isinstance(action, SubmitLimitOrder) for action in participation)
    assert not any(isinstance(action, SubmitLimitOrder) for action in momentum)

    participated = participation_runner.handle(
        event(
            InputKind.MARKET,
            at=11,
            payload={"event_kind": "trade", "quantity": 20},
        )
    )
    accelerated = momentum_runner.handle(event(InputKind.LEVELS, at=11, payload=levels(101, 103)))
    assert [action for action in participated if isinstance(action, SubmitLimitOrder)] == [
        SubmitLimitOrder("informed-1-1", Side.BUY, 103, 2)
    ]
    assert [action for action in accelerated if isinstance(action, SubmitLimitOrder)] == [
        SubmitLimitOrder("informed-1-1", Side.BUY, 105, 2)
    ]


def test_metaorder_deadline_emits_completion_log() -> None:
    runner = informed(ExecutionStyle.SCHEDULED)
    start_metaorder(runner)

    completed = runner.handle(
        event(
            InputKind.TIMER,
            at=22,
            payload={"timer_id": "informed-execution"},
        )
    )
    completion_logs = [
        action
        for action in completed
        if isinstance(action, EmitLog) and action.message == "informed_metaorder_completed"
    ]
    assert len(completion_logs) == 1
    assert completion_logs[0].fields["status"] == "deadline"
    assert completion_logs[0].fields["execution_style"] == "scheduled"


def test_informed_signal_reads_the_worlds_rolling_schedule_at_late_time() -> None:
    world = WorldGenerator(
        21,
        RegimeConfig(
            participant_count=1,
            parent_order_count=0,
            start_time=0,
            end_time=200,
            min_quantity=20,
            max_quantity=20,
            min_duration=20,
            max_duration=20,
            rolling_mandates=True,
            min_mandate_gap=10,
            max_mandate_gap=10,
            initial_mandate_spread=0,
        ),
    )
    strategy = FutureFlowInformedTrader(
        world=world,
        parent_orders=None,
        update_interval=100,
        signal_horizon=10,
        signal_loading=1,
        signal_observation_probability=1,
        signal_noise_multiplier=0,
        signal_noise_floor=0,
        minimum_signal=1,
        minimum_parent_quantity=1,
        maximum_parent_quantity=5,
        quantity_per_signal_unit=0.1,
        metaorder_duration=10,
        slice_interval=5,
        execution_style=ExecutionStyle.BURST,
    )
    runner = StrategyRunner(
        strategy,
        StrategyContext(participant_id="informed-rolling", strategy_instance_id="I1", seed=9),
    )
    runner.handle(event(InputKind.START, at=0))
    runner.handle(event(InputKind.LEVELS, at=1, payload=levels()))
    batch = runner.handle(
        event(
            InputKind.TIMER,
            at=100,
            payload={"timer_id": "future-flow-signal"},
        )
    )

    signal = next(action for action in batch if isinstance(action, EmitLog) and action.message == "informed_signal")
    assert signal.fields["true_future_flow"] != 0
