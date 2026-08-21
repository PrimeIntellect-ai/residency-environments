from __future__ import annotations

import pytest
from alphaverse.models import Side
from alphaverse.strategy import (
    InputEnvelope,
    InputKind,
    Strategy,
    StrategyContext,
    StrategyRunner,
    SubmitLimitOrder,
)


def _event(kind: InputKind, *, instance: str = "I1") -> InputEnvelope:
    return InputEnvelope("S1", instance, "E1", kind, 10, 12, 3, {})


class Quoter(Strategy):
    def __init__(self) -> None:
        self.callbacks: list[InputKind] = []

    def on_start(self, ctx, event):
        self.callbacks.append(event.kind)
        return [ctx.set_timer("refresh", fire_at=ctx.now + 10)]

    def on_levels(self, ctx, event):
        self.callbacks.append(event.kind)
        return [ctx.submit_limit("bid", Side.BUY, -1, 2)]


def test_runner_dispatches_and_constructs_actions() -> None:
    strategy = Quoter()
    context = StrategyContext(participant_id="p1", strategy_instance_id="I1", seed=4)
    runner = StrategyRunner(strategy, context)

    start = runner.handle(_event(InputKind.START))
    levels = runner.handle(_event(InputKind.LEVELS))

    assert strategy.callbacks == [InputKind.START, InputKind.LEVELS]
    assert start.actions[0].fire_at == 22
    assert levels.actions == (SubmitLimitOrder("bid", Side.BUY, -1, 2),)
    assert context.now == 12


def test_runner_enforces_instance_and_action_limit() -> None:
    class Noisy(Strategy):
        def on_market(self, ctx, event):
            return [ctx.stop(str(index)) for index in range(2)]

    runner = StrategyRunner(
        Noisy(),
        StrategyContext(participant_id="p1", strategy_instance_id="I1"),
        max_actions_per_callback=1,
    )
    with pytest.raises(ValueError, match="another strategy"):
        runner.handle(_event(InputKind.MARKET, instance="I2"))
    with pytest.raises(ValueError, match="exceeded"):
        runner.handle(_event(InputKind.MARKET))


def test_context_rng_is_deterministic() -> None:
    first = StrategyContext(participant_id="p", strategy_instance_id="I", seed=7)
    second = StrategyContext(participant_id="p", strategy_instance_id="I", seed=7)
    assert [first.random.random() for _ in range(3)] == [second.random.random() for _ in range(3)]


def test_context_constructs_alert_action() -> None:
    context = StrategyContext(participant_id="p", strategy_instance_id="I")

    alert = context.emit_alert("spread", "spread widened", {"ticks": 4})

    assert alert.code == "spread"
    assert alert.message == "spread widened"
    assert alert.data == {"ticks": 4}
