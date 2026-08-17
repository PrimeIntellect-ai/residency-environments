"""Python authoring SDK and in-process reference strategy runner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from random import Random

from alphaverse.models import Side
from alphaverse.strategy.protocol import (
    Action,
    ActionBatch,
    CancelOrderAction,
    CancelTimer,
    EmitAlert,
    EmitLog,
    InputEnvelope,
    InputKind,
    LogLevel,
    RequestStop,
    SetTimer,
    SubmitLimitOrder,
)

StrategyResult = ActionBatch | Iterable[Action] | None


class StrategyContext:
    """Capabilities exposed to Python strategies during callbacks."""

    def __init__(
        self,
        *,
        participant_id: str,
        strategy_instance_id: str,
        product_id: str = "ALPHA",
        parameters: Mapping[str, object] | None = None,
        seed: int = 0,
    ) -> None:
        if not participant_id or not strategy_instance_id or not product_id:
            raise ValueError("context identifiers must not be empty")
        self.participant_id = participant_id
        self.strategy_instance_id = strategy_instance_id
        self.product_id = product_id
        self.parameters = dict(parameters or {})
        self.random = Random(seed)
        self.now = 0
        self.exchange_time = 0
        self.source_event_seq = 0

    def _update(self, envelope: InputEnvelope) -> None:
        self.now = envelope.available_at
        self.exchange_time = envelope.exchange_time
        self.source_event_seq = envelope.source_event_seq

    def submit_limit(
        self,
        client_order_id: str,
        side: Side,
        price: int,
        quantity: int,
    ) -> SubmitLimitOrder:
        return SubmitLimitOrder(
            client_order_id,
            side,
            price,
            quantity,
            self.product_id,
        )

    def cancel(self, order_id: str) -> CancelOrderAction:
        return CancelOrderAction(order_id, self.product_id)

    def set_timer(self, timer_id: str, *, fire_at: int) -> SetTimer:
        if fire_at < self.now:
            raise ValueError("timer cannot fire in the past")
        return SetTimer(timer_id, fire_at)

    def cancel_timer(self, timer_id: str) -> CancelTimer:
        return CancelTimer(timer_id)

    def log(
        self,
        message: str,
        *,
        level: LogLevel = LogLevel.INFO,
        fields: Mapping[str, object] | None = None,
    ) -> EmitLog:
        return EmitLog(level, message, fields or {})

    def emit_alert(
        self,
        code: str,
        message: str,
        data: Mapping[str, object] | None = None,
    ) -> EmitAlert:
        """Request a durable notification for the focal player."""

        return EmitAlert(code, message, data or {})

    def stop(self, reason: str = "") -> RequestStop:
        return RequestStop(reason)


class Strategy:
    """Stateful callback interface layered over the canonical actor protocol."""

    def on_start(self, ctx: StrategyContext, event: InputEnvelope) -> StrategyResult:
        return None

    def on_market(self, ctx: StrategyContext, event: InputEnvelope) -> StrategyResult:
        return None

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope) -> StrategyResult:
        return None

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope) -> StrategyResult:
        return None

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope) -> StrategyResult:
        return None

    def on_risk(self, ctx: StrategyContext, event: InputEnvelope) -> StrategyResult:
        return None

    def on_stop(self, ctx: StrategyContext, event: InputEnvelope) -> StrategyResult:
        return None

    def on_signal(self, ctx: StrategyContext, event: InputEnvelope) -> StrategyResult:
        return None

    def on_alert(self, ctx: StrategyContext, event: InputEnvelope) -> StrategyResult:
        return None


_CALLBACKS = {
    InputKind.START: "on_start",
    InputKind.MARKET: "on_market",
    InputKind.LEVELS: "on_levels",
    InputKind.EXECUTION: "on_execution",
    InputKind.TIMER: "on_timer",
    InputKind.RISK: "on_risk",
    InputKind.STOP: "on_stop",
    InputKind.SIGNAL: "on_signal",
    InputKind.ALERT: "on_alert",
}


class StrategyRunner:
    """Reference in-process runner used for tests and trusted participants."""

    def __init__(
        self,
        strategy: Strategy,
        context: StrategyContext,
        *,
        max_actions_per_callback: int = 100,
    ) -> None:
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy must be a Strategy")
        if max_actions_per_callback <= 0:
            raise ValueError("max_actions_per_callback must be positive")
        self.strategy = strategy
        self.context = context
        self.max_actions_per_callback = max_actions_per_callback

    def handle(self, envelope: InputEnvelope) -> ActionBatch:
        if envelope.strategy_instance_id != self.context.strategy_instance_id:
            raise ValueError("event belongs to another strategy instance")
        self.context._update(envelope)
        callback = getattr(self.strategy, _CALLBACKS[envelope.kind])
        result = callback(self.context, envelope)
        if result is None:
            batch = ActionBatch()
        elif isinstance(result, ActionBatch):
            batch = result
        else:
            batch = ActionBatch(result)
        if len(batch) > self.max_actions_per_callback:
            raise ValueError("strategy exceeded max_actions_per_callback")
        return batch


__all__ = ["Strategy", "StrategyContext", "StrategyRunner"]
