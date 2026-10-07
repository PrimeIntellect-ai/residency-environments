"""Deterministic episode orchestration for strategy-backed participants.

The episode is the causal boundary between strategy actors and the exchange.  It
delivers delayed observations, schedules actions after configured latency, and
routes every order through the ordinary exchange command path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

from alphaverse.clock import CancellationToken, ScheduledEvent
from alphaverse.exchange import Exchange, MarginMetrics, TerminationResult
from alphaverse.models import CancelOrder, Event, EventKind, NewOrder
from alphaverse.profiles import InformationGrant, ParticipantSpec
from alphaverse.strategy.protocol import (
    Action,
    CancelOrderAction,
    CancelTimer,
    EmitLog,
    InputEnvelope,
    InputKind,
    RequestStop,
    SetTimer,
    SubmitLimitOrder,
    WireValue,
)
from alphaverse.strategy.sdk import Strategy, StrategyContext, StrategyRunner


@dataclass(frozen=True, slots=True)
class StrategyLog:
    strategy_instance_id: str
    market_time: int
    entry: EmitLog


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    strategy_instance_id: str
    accepted_for_delivery: bool
    arrival_at: int


@dataclass(frozen=True, slots=True)
class _Delivery:
    strategy_instance_id: str
    envelope: InputEnvelope


@dataclass(frozen=True, slots=True)
class _ActionArrival:
    strategy_instance_id: str
    action: SubmitLimitOrder | CancelOrderAction
    strategy_generation: int | None


@dataclass(frozen=True, slots=True)
class _TimerFire:
    strategy_instance_id: str
    timer_id: str
    generation: int


@dataclass(frozen=True, slots=True)
class _MarginDeadline:
    participant_id: str
    deadline: int


_ScheduledPayload: TypeAlias = _Delivery | _ActionArrival | _TimerFire | _MarginDeadline


@dataclass(slots=True)
class _Actor:
    spec: ParticipantSpec
    runner: StrategyRunner
    active: bool = True
    timers: dict[str, tuple[int, CancellationToken]] = field(default_factory=dict)
    strategy_generation: int = 1


class Episode:
    """Run trusted strategy adapters against one deterministic exchange."""

    _CALLBACK_PRIORITY = 10
    _ACTION_PRIORITY = 20
    _DELIVERY_PRIORITY = 30

    def __init__(self, exchange: Exchange | None = None, *, session_id: str = "S1") -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        self.exchange = exchange or Exchange()
        self.session_id = session_id
        self._actors: dict[str, _Actor] = {}
        self._participant_instances: dict[str, str] = {}
        self._next_instance_number = 1
        self._next_synthetic_event_number = 1
        self._timer_generations: dict[tuple[str, str], int] = {}
        self._logs: list[StrategyLog] = []

    @property
    def now(self) -> int:
        return self.exchange.clock.now

    @property
    def logs(self) -> tuple[StrategyLog, ...]:
        return tuple(self._logs)

    @property
    def participant_specs(self) -> tuple[ParticipantSpec, ...]:
        """Return immutable participant configurations in stable ID order."""

        return tuple(actor.spec for actor in sorted(self._actors.values(), key=lambda item: item.spec.participant_id))

    def add_strategy(self, spec: ParticipantSpec, strategy: Strategy) -> str:
        """Register one account and schedule its start callback at current time."""

        if spec.participant_id in self._participant_instances:
            raise ValueError(f"participant already has a strategy: {spec.participant_id}")
        instance_id = f"I{self._next_instance_number}"
        self._next_instance_number += 1
        context = StrategyContext(
            participant_id=spec.participant_id,
            strategy_instance_id=instance_id,
            product_id=self.exchange.product.product_id,
            parameters=spec.parameters,
            seed=spec.seed,
        )
        actor = _Actor(
            spec=spec,
            runner=StrategyRunner(
                strategy,
                context,
                max_actions_per_callback=spec.risk.max_actions_per_callback,
            ),
        )
        self.exchange.register_account(
            spec.participant_id,
            starting_cash=spec.account_starting_cash,
            margin=spec.margin,
        )
        self._actors[instance_id] = actor
        self._participant_instances[spec.participant_id] = instance_id
        account = self.exchange.clearing.snapshot(spec.participant_id)
        self._schedule_envelope(
            instance_id,
            kind=InputKind.START,
            exchange_time=self.now,
            available_at=self.now,
            source_event_seq=self.exchange.event_log.last_sequence,
            payload={
                "participant_id": spec.participant_id,
                "product_id": self.exchange.product.product_id,
                "cash": account.cash,
                "position": account.position,
            },
            priority=self._CALLBACK_PRIORITY,
        )
        return instance_id

    def run_until(
        self,
        market_time: int,
        *,
        max_events: int | None = None,
    ) -> int:
        """Process all scheduled work through an absolute virtual timestamp."""

        processed = self.exchange.clock.run_until_count(
            market_time,
            self._handle_scheduled,
            max_events=max_events,
        )
        margin_events = self.exchange.process_margin(market_time=self.now)
        self._publish(margin_events)
        if margin_events:
            processed += self.exchange.clock.run_until_count(
                market_time,
                self._handle_scheduled,
                max_events=max_events,
            )
        return processed

    def risk_snapshot(self, participant_id: str) -> MarginMetrics | None:
        """Return persistent margin/mark state for a focal participant."""

        return self.exchange.margin_metrics(participant_id)

    def enqueue_player_action(
        self,
        participant_id: str,
        action: SubmitLimitOrder | CancelOrderAction,
    ) -> ActionReceipt:
        """Queue a player-originated command with order-entry latency.

        Model thinking time happens outside this method, so the strategy decision
        latency is deliberately not charged a second time.
        """

        if not isinstance(action, (SubmitLimitOrder, CancelOrderAction)):
            raise TypeError("player action must be an order submission or cancellation")
        instance_id = self._instance_for_participant(participant_id)
        actor = self._actors[instance_id]
        if not actor.active:
            raise RuntimeError("strategy instance is not active")
        arrival_at = self.now + actor.spec.technology.order_entry_latency
        self.exchange.clock.schedule(
            arrival_at,
            _ActionArrival(instance_id, action, None),
            priority=self._ACTION_PRIORITY,
        )
        return ActionReceipt(instance_id, True, arrival_at)

    def replace_strategy(self, instance_id: str, strategy: Strategy) -> None:
        """Replace one actor at a virtual-time boundary while retaining its account."""

        try:
            actor = self._actors[instance_id]
        except KeyError:
            raise KeyError(f"unknown strategy instance: {instance_id}") from None
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy must be a Strategy")

        for _, token in actor.timers.values():
            token.cancel()
        actor.timers.clear()
        actor.strategy_generation += 1

        owned_orders = tuple(
            order.order_id for order in self.exchange.book.orders_for_participant(actor.spec.participant_id)
        )
        for order_id in owned_orders:
            result = self.exchange.cancel_order(
                CancelOrder(
                    actor.spec.participant_id,
                    order_id,
                    self.exchange.product.product_id,
                ),
                market_time=self.now,
            )
            self._publish(result.events)

        context = StrategyContext(
            participant_id=actor.spec.participant_id,
            strategy_instance_id=instance_id,
            product_id=self.exchange.product.product_id,
            parameters=actor.spec.parameters,
            seed=actor.spec.seed,
        )
        actor.runner = StrategyRunner(
            strategy,
            context,
            max_actions_per_callback=actor.spec.risk.max_actions_per_callback,
        )
        actor.active = True
        account = self.exchange.clearing.snapshot(actor.spec.participant_id)
        self._schedule_envelope(
            instance_id,
            kind=InputKind.START,
            exchange_time=self.now,
            available_at=self.now,
            source_event_seq=self.exchange.event_log.last_sequence,
            payload={
                "participant_id": actor.spec.participant_id,
                "product_id": self.exchange.product.product_id,
                "cash": account.cash,
                "position": account.position,
                "replacement": True,
            },
            priority=self._CALLBACK_PRIORITY,
        )

    def emit_signal(
        self,
        participant_id: str,
        payload: dict[str, object],
        *,
        grant: InformationGrant,
        available_at: int | None = None,
    ) -> None:
        """Deliver privileged world information only to an explicitly entitled actor."""

        if grant not in (
            InformationGrant.FUTURE_FLOW_SIGNAL,
            InformationGrant.LATENT_VALUE_SIGNAL,
        ):
            raise ValueError("grant must identify a private signal capability")
        instance_id = self._instance_for_participant(participant_id)
        actor = self._actors[instance_id]
        if grant not in actor.spec.information_grants:
            raise PermissionError(f"participant is not entitled to {grant.value}")
        delivery_time = self.now if available_at is None else available_at
        self._schedule_envelope(
            instance_id,
            kind=InputKind.SIGNAL,
            exchange_time=self.now,
            available_at=delivery_time,
            source_event_seq=self.exchange.event_log.last_sequence,
            payload={"signal_type": grant.value, **payload},
        )

    def terminate_participant(self, participant_id: str) -> TerminationResult:
        """Request account liquidation and publish the resulting exchange events."""

        result = self.exchange.request_termination(participant_id, market_time=self.now)
        self._publish(result.events)
        return result

    def _handle_scheduled(self, scheduled: ScheduledEvent[object]) -> None:
        payload = scheduled.payload
        if isinstance(payload, _Delivery):
            self._deliver(payload)
        elif isinstance(payload, _ActionArrival):
            self._arrive_action(payload)
        elif isinstance(payload, _TimerFire):
            self._fire_timer(payload)
        elif isinstance(payload, _MarginDeadline):
            self._process_margin_deadline(payload)
        else:
            raise TypeError(f"unknown episode payload: {type(payload).__name__}")

    def _deliver(self, delivery: _Delivery) -> None:
        actor = self._actors[delivery.strategy_instance_id]
        if not actor.active and delivery.envelope.kind is not InputKind.STOP:
            return
        batch = actor.runner.handle(delivery.envelope)
        for action in batch:
            self._handle_action(delivery.strategy_instance_id, action)
        if delivery.envelope.kind is InputKind.STOP:
            actor.active = False

    def _handle_action(self, instance_id: str, action: Action) -> None:
        actor = self._actors[instance_id]
        if isinstance(action, (SubmitLimitOrder, CancelOrderAction)):
            arrival_time = self.now + actor.spec.technology.decision_latency + actor.spec.technology.order_entry_latency
            self.exchange.clock.schedule(
                arrival_time,
                _ActionArrival(instance_id, action, actor.strategy_generation),
                priority=self._ACTION_PRIORITY,
            )
        elif isinstance(action, SetTimer):
            self._set_timer(instance_id, action)
        elif isinstance(action, CancelTimer):
            timer = actor.timers.pop(action.timer_id, None)
            if timer is not None:
                timer[1].cancel()
        elif isinstance(action, EmitLog):
            self._logs.append(StrategyLog(instance_id, self.now, action))
        elif isinstance(action, RequestStop):
            self._request_strategy_stop(instance_id, action.reason)
        else:
            raise TypeError(f"unknown strategy action: {type(action).__name__}")

    def _arrive_action(self, arrival: _ActionArrival) -> None:
        actor = self._actors[arrival.strategy_instance_id]
        if not actor.active:
            return
        if arrival.strategy_generation is not None and arrival.strategy_generation != actor.strategy_generation:
            return
        participant_id = actor.spec.participant_id
        if isinstance(arrival.action, SubmitLimitOrder):
            command = NewOrder(
                participant_id=participant_id,
                client_order_id=arrival.action.client_order_id,
                side=arrival.action.side,
                price=arrival.action.price,
                quantity=arrival.action.quantity,
                product_id=arrival.action.product_id,
            )
            reason = self._risk_rejection_reason(actor, arrival.action)
            result = (
                self.exchange.reject_order(
                    command,
                    reason,
                    market_time=self.now,
                )
                if reason is not None
                else self.exchange.submit_order(command, market_time=self.now)
            )
        else:
            result = self.exchange.cancel_order(
                CancelOrder(
                    participant_id=participant_id,
                    order_id=arrival.action.order_id,
                    product_id=arrival.action.product_id,
                ),
                market_time=self.now,
            )
        self._publish(result.events)

    def _risk_rejection_reason(
        self,
        actor: _Actor,
        action: SubmitLimitOrder | CancelOrderAction,
    ) -> str | None:
        if not isinstance(action, SubmitLimitOrder):
            return None
        risk = actor.spec.risk
        if action.quantity > risk.max_order_quantity:
            return "max_order_quantity"
        live_orders = self.exchange.book.orders_for_participant(actor.spec.participant_id)
        if len(live_orders) >= risk.max_live_orders:
            return "max_live_orders"
        position = self.exchange.clearing.snapshot(actor.spec.participant_id).position
        same_side_live_quantity = sum(order.remaining_quantity for order in live_orders if order.side is action.side)
        worst_case_position = position + action.side.signed(same_side_live_quantity + action.quantity)
        if abs(worst_case_position) > risk.max_abs_position:
            return "max_abs_position"
        return None

    def _publish(self, events: tuple[Event, ...]) -> None:
        for event in events:
            payload = InputEnvelope.freeze_payload(
                {
                    "event_kind": event.kind.value,
                    "match_event_id": event.match_event_id,
                    "product_id": event.product_id,
                    **dict(event.data),
                }
            )
            if event.kind in (EventKind.TRADE, EventKind.MBO_CHANGE):
                for instance_id, actor in self._actors.items():
                    if not actor.active:
                        continue
                    if event.kind is EventKind.MBO_CHANGE and not actor.spec.technology.mbo_entitled:
                        continue
                    latency = actor.spec.technology.market_data_latency
                    self._schedule_exchange_event(
                        instance_id,
                        event,
                        InputKind.MARKET,
                        event.market_time + latency,
                        payload,
                    )
            elif event.kind is EventKind.LEVELS:
                for instance_id, actor in self._actors.items():
                    if not actor.active:
                        continue
                    latency = actor.spec.technology.market_data_latency + actor.spec.technology.level_feed_latency
                    self._schedule_exchange_event(
                        instance_id,
                        event,
                        InputKind.LEVELS,
                        event.market_time + latency,
                        payload,
                    )
            elif event.kind is EventKind.RISK:
                participant_id = event.data.get("participant_id")
                if not isinstance(participant_id, str):
                    continue
                instance_id = self._participant_instances.get(participant_id)
                if instance_id is None or not self._actors[instance_id].active:
                    continue
                latency = self._actors[instance_id].spec.technology.market_data_latency
                self._schedule_exchange_event(
                    instance_id,
                    event,
                    InputKind.RISK,
                    event.market_time + latency,
                    payload,
                )
                deadline = event.data.get("liquidation_deadline")
                if event.data.get("state") == "margin_call" and isinstance(deadline, int):
                    self.exchange.clock.schedule(
                        deadline,
                        _MarginDeadline(participant_id, deadline),
                        priority=self._ACTION_PRIORITY,
                    )
            elif event.kind in (
                EventKind.ORDER_ACCEPTED,
                EventKind.ORDER_REJECTED,
                EventKind.CANCEL_ACCEPTED,
                EventKind.CANCEL_REJECTED,
                EventKind.FILL,
                EventKind.ACCOUNT,
                EventKind.SESSION,
            ):
                participant_id = event.data.get("participant_id")
                if not isinstance(participant_id, str):
                    continue
                instance_id = self._participant_instances.get(participant_id)
                if instance_id is None:
                    continue
                actor = self._actors[instance_id]
                if not actor.active:
                    continue
                latency = actor.spec.technology.market_data_latency
                self._schedule_exchange_event(
                    instance_id,
                    event,
                    InputKind.EXECUTION,
                    event.market_time + latency,
                    payload,
                )

    def _schedule_exchange_event(
        self,
        instance_id: str,
        event: Event,
        kind: InputKind,
        available_at: int,
        payload: Mapping[str, WireValue],
    ) -> None:
        self._schedule_envelope(
            instance_id,
            kind=kind,
            exchange_time=event.market_time,
            available_at=available_at,
            source_event_seq=event.sequence,
            payload=payload,
            event_id=f"E{event.sequence}",
        )

    def _set_timer(self, instance_id: str, action: SetTimer) -> None:
        actor = self._actors[instance_id]
        previous = actor.timers.pop(action.timer_id, None)
        if previous is not None:
            previous[1].cancel()
        key = (instance_id, action.timer_id)
        generation = self._timer_generations.get(key, 0) + 1
        self._timer_generations[key] = generation
        token = self.exchange.clock.schedule(
            action.fire_at,
            _TimerFire(instance_id, action.timer_id, generation),
            priority=self._CALLBACK_PRIORITY,
        )
        actor.timers[action.timer_id] = (generation, token)

    def _fire_timer(self, timer: _TimerFire) -> None:
        actor = self._actors[timer.strategy_instance_id]
        current = actor.timers.get(timer.timer_id)
        if current is None or current[0] != timer.generation:
            return
        del actor.timers[timer.timer_id]
        self._schedule_envelope(
            timer.strategy_instance_id,
            kind=InputKind.TIMER,
            exchange_time=self.now,
            available_at=self.now,
            source_event_seq=self.exchange.event_log.last_sequence,
            payload={"timer_id": timer.timer_id},
            priority=self._CALLBACK_PRIORITY,
        )

    def _process_margin_deadline(self, deadline: _MarginDeadline) -> None:
        snapshot = self.exchange.margin_metrics(deadline.participant_id)
        if snapshot is None or snapshot.liquidation_deadline != deadline.deadline:
            return
        self._publish(self.exchange.process_margin(market_time=self.now))

    def _request_strategy_stop(self, instance_id: str, reason: str) -> None:
        actor = self._actors[instance_id]
        for timer_id, (_, token) in tuple(actor.timers.items()):
            token.cancel()
            del actor.timers[timer_id]
        owned_orders = tuple(
            order.order_id for order in self.exchange.book.orders_for_participant(actor.spec.participant_id)
        )
        for order_id in owned_orders:
            result = self.exchange.cancel_order(
                CancelOrder(actor.spec.participant_id, order_id, self.exchange.product.product_id),
                market_time=self.now,
            )
            self._publish(result.events)
        self._schedule_envelope(
            instance_id,
            kind=InputKind.STOP,
            exchange_time=self.now,
            available_at=self.now,
            source_event_seq=self.exchange.event_log.last_sequence,
            payload={"reason": reason},
            priority=self._CALLBACK_PRIORITY,
        )

    def _schedule_envelope(
        self,
        instance_id: str,
        *,
        kind: InputKind,
        exchange_time: int,
        available_at: int,
        source_event_seq: int,
        payload: Mapping[str, WireValue],
        event_id: str | None = None,
        priority: int | None = None,
    ) -> None:
        if event_id is None:
            event_id = f"X{self._next_synthetic_event_number}"
            self._next_synthetic_event_number += 1
        envelope = InputEnvelope._from_internal_payload(
            session_id=self.session_id,
            strategy_instance_id=instance_id,
            event_id=event_id,
            kind=kind,
            exchange_time=exchange_time,
            available_at=available_at,
            source_event_seq=source_event_seq,
            payload=payload,
        )
        self.exchange.clock.schedule(
            available_at,
            _Delivery(instance_id, envelope),
            priority=self._DELIVERY_PRIORITY if priority is None else priority,
        )

    def _instance_for_participant(self, participant_id: str) -> str:
        try:
            return self._participant_instances[participant_id]
        except KeyError:
            raise KeyError(f"unknown strategy participant: {participant_id}") from None


__all__ = ["ActionReceipt", "Episode", "StrategyLog"]
