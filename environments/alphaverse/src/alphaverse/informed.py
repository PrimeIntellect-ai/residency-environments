"""Composable future-flow informed participant used by populated scenarios."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from alphaverse.models import Side
from alphaverse.strategy import Action, InputEnvelope, Strategy, StrategyContext
from alphaverse.world import ExecutionStyle, ParentOrder, WorldGenerator


def _best_price(payload: Mapping[str, object], key: str) -> int | None:
    levels = payload.get(key)
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)) or not levels:
        return None
    first = levels[0]
    if not isinstance(first, Mapping):
        return None
    price = first.get("price")
    return price if isinstance(price, int) and not isinstance(price, bool) else None


@dataclass(slots=True)
class _Metaorder:
    metaorder_id: str
    side: Side
    total_quantity: int
    unassigned_quantity: int
    start_time: int
    end_time: int
    initial_midpoint: float
    true_signal: int
    private_signal: int
    target_position: int
    submitted_quantity: int = 0
    filled_quantity: int = 0
    child_count: int = 0


class FutureFlowInformedTrader(Strategy):
    """Trade a noisy projection of scheduled future alpha-less demand.

    Signal quality and execution style are constructor parameters so scenarios
    can assign them independently. The private signal produces a target-position
    change and a finite metaorder. Child orders still use the same ordinary
    strategy and exchange interfaces as every other participant.
    """

    def __init__(
        self,
        *,
        world: WorldGenerator,
        parent_orders: tuple[ParentOrder, ...] | None,
        update_interval: int,
        signal_horizon: int,
        signal_loading: float,
        signal_observation_probability: float,
        signal_noise_multiplier: float,
        signal_noise_floor: int,
        minimum_signal: int,
        minimum_parent_quantity: int,
        maximum_parent_quantity: int,
        quantity_per_signal_unit: float,
        metaorder_duration: int,
        slice_interval: int,
        execution_style: ExecutionStyle,
        aggressive_limit_ticks: int = 2,
        participation_rate: float = 0.25,
        momentum_trigger_ticks: float = 0.5,
        signal_timer_id: str = "future-flow-signal",
        execution_timer_id: str = "informed-execution",
    ) -> None:
        if not isinstance(world, WorldGenerator):
            raise TypeError("world must be a WorldGenerator")
        if parent_orders is not None and not all(isinstance(parent, ParentOrder) for parent in parent_orders):
            raise TypeError("parent_orders must contain ParentOrder values")
        for name, value in (
            ("update_interval", update_interval),
            ("signal_horizon", signal_horizon),
            ("minimum_signal", minimum_signal),
            ("minimum_parent_quantity", minimum_parent_quantity),
            ("maximum_parent_quantity", maximum_parent_quantity),
            ("metaorder_duration", metaorder_duration),
            ("slice_interval", slice_interval),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if maximum_parent_quantity < minimum_parent_quantity:
            raise ValueError("maximum_parent_quantity must be at least its minimum")
        for name, value in (
            ("signal_loading", signal_loading),
            (
                "signal_observation_probability",
                signal_observation_probability,
            ),
            ("signal_noise_multiplier", signal_noise_multiplier),
            ("quantity_per_signal_unit", quantity_per_signal_unit),
            ("participation_rate", participation_rate),
            ("momentum_trigger_ticks", momentum_trigger_ticks),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if signal_noise_floor < 0:
            raise ValueError("signal_noise_floor must be non-negative")
        if not 0 <= signal_observation_probability <= 1:
            raise ValueError("signal_observation_probability must be between zero and one")
        if not 0 < participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")
        if not isinstance(execution_style, ExecutionStyle):
            raise TypeError("execution_style must be an ExecutionStyle")
        if aggressive_limit_ticks < 0:
            raise ValueError("aggressive_limit_ticks must be non-negative")
        if not signal_timer_id or not execution_timer_id:
            raise ValueError("timer identifiers must not be empty")

        self.world = world
        # Keeping an explicit collection is useful for small isolated strategy
        # tests. Populated scenarios leave it unset, so every signal reads the
        # world's rolling mandate schedule at the current market time.
        self.parent_orders = parent_orders
        self.update_interval = update_interval
        self.signal_horizon = signal_horizon
        self.signal_loading = signal_loading
        self.signal_observation_probability = signal_observation_probability
        self.signal_noise_multiplier = signal_noise_multiplier
        self.signal_noise_floor = signal_noise_floor
        self.minimum_signal = minimum_signal
        self.minimum_parent_quantity = minimum_parent_quantity
        self.maximum_parent_quantity = maximum_parent_quantity
        self.quantity_per_signal_unit = quantity_per_signal_unit
        self.metaorder_duration = metaorder_duration
        self.slice_interval = slice_interval
        self.execution_style = execution_style
        self.aggressive_limit_ticks = aggressive_limit_ticks
        self.participation_rate = participation_rate
        self.momentum_trigger_ticks = momentum_trigger_ticks
        self.signal_timer_id = signal_timer_id
        self.execution_timer_id = execution_timer_id

        self.position = 0
        self._best_bid: int | None = None
        self._best_ask: int | None = None
        self._signal_index = 0
        self._metaorder_index = 0
        self._active: _Metaorder | None = None
        self._pending: dict[str, tuple[str, int]] = {}
        self._owned: dict[str, tuple[str, int]] = {}
        self._recent_public_volume = 0
        self._last_child_at: int | None = None

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [
            ctx.set_timer(
                self.signal_timer_id,
                fire_at=ctx.now + self.update_interval,
            )
        ]

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope):
        self._best_bid = _best_price(event.payload, "bids")
        self._best_ask = _best_price(event.payload, "asks")
        active = self._active
        if active is None or self.execution_style is not ExecutionStyle.MOMENTUM or not self._can_submit_now(ctx.now):
            return None
        midpoint = self._midpoint()
        if midpoint is None:
            return None
        signed_move = active.side.value * (midpoint - active.initial_midpoint)
        if signed_move < self.momentum_trigger_ticks:
            return None
        return self._submit_child(
            ctx,
            min(
                active.unassigned_quantity,
                self._default_slice_quantity(active),
            ),
            aggressive=True,
            trigger="momentum",
        )

    def on_market(self, ctx: StrategyContext, event: InputEnvelope):
        if event.payload.get("event_kind") != "trade":
            return None
        quantity = event.payload.get("quantity")
        if isinstance(quantity, int) and not isinstance(quantity, bool):
            self._recent_public_volume += quantity
        active = self._active
        if (
            active is None
            or self.execution_style is not ExecutionStyle.PARTICIPATION
            or not self._can_submit_now(ctx.now)
            or self._recent_public_volume <= 0
        ):
            return None
        participation_quantity = max(
            1,
            math.ceil(self._recent_public_volume * self.participation_rate),
        )
        self._recent_public_volume = 0
        return self._submit_child(
            ctx,
            min(
                active.unassigned_quantity,
                self._default_slice_quantity(active),
                participation_quantity,
            ),
            aggressive=True,
            trigger="public_volume",
        )

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        payload = event.payload
        position = payload.get("position")
        if isinstance(position, int) and not isinstance(position, bool):
            self.position = position

        event_kind = payload.get("event_kind")
        client_order_id = payload.get("client_order_id")
        if event_kind == "order_accepted":
            order_id = payload.get("order_id")
            if isinstance(client_order_id, str) and isinstance(order_id, str) and client_order_id in self._pending:
                self._owned[order_id] = self._pending.pop(client_order_id)
            return None

        if event_kind == "order_rejected" and isinstance(client_order_id, str):
            pending = self._pending.pop(client_order_id, None)
            if pending is not None:
                metaorder_id, quantity = pending
                if self._active_id() == metaorder_id:
                    assert self._active is not None
                    self._active.unassigned_quantity += quantity
            return None

        order_id = payload.get("order_id")
        if not isinstance(order_id, str):
            return None
        if event_kind == "cancel_accepted":
            self._owned.pop(order_id, None)
            return None
        if event_kind != "fill":
            return None

        owned = self._owned.get(order_id)
        quantity = payload.get("quantity")
        if owned is None or not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
            return None
        metaorder_id, remaining = owned
        remaining -= quantity
        if remaining > 0:
            self._owned[order_id] = (metaorder_id, remaining)
        else:
            self._owned.pop(order_id, None)

        active = self._active
        if active is None or active.metaorder_id != metaorder_id:
            return None
        active.filled_quantity += quantity
        if active.filled_quantity >= active.total_quantity:
            return self._finish_metaorder(ctx, "filled")
        if (
            self.execution_style is ExecutionStyle.PATIENT
            and active.unassigned_quantity > 0
            and not self._has_outstanding(active.metaorder_id)
        ):
            return self._submit_child(
                ctx,
                min(
                    active.unassigned_quantity,
                    self._default_slice_quantity(active),
                ),
                aggressive=False,
                trigger="replenishment",
            )
        return None

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        timer_id = event.payload.get("timer_id")
        if timer_id == self.signal_timer_id:
            actions = self._observe_signal(ctx)
            actions.append(
                ctx.set_timer(
                    self.signal_timer_id,
                    fire_at=ctx.now + self.update_interval,
                )
            )
            return actions
        if timer_id != self.execution_timer_id:
            return None

        active = self._active
        if active is None:
            return None
        if ctx.now >= active.end_time:
            return self._finish_metaorder(ctx, "deadline")

        actions: list[Action] = []
        if self.execution_style is ExecutionStyle.SCHEDULED:
            actions.extend(
                self._submit_child(
                    ctx,
                    min(
                        active.unassigned_quantity,
                        self._default_slice_quantity(active),
                    ),
                    aggressive=True,
                    trigger="schedule",
                )
            )
        elif self.execution_style is ExecutionStyle.PARTICIPATION:
            if self._recent_public_volume > 0:
                quantity = max(
                    1,
                    math.ceil(self._recent_public_volume * self.participation_rate),
                )
                self._recent_public_volume = 0
                actions.extend(
                    self._submit_child(
                        ctx,
                        min(
                            active.unassigned_quantity,
                            self._default_slice_quantity(active),
                            quantity,
                        ),
                        aggressive=True,
                        trigger="participation_timer",
                    )
                )
        elif (
            self.execution_style is ExecutionStyle.MOMENTUM
            and ctx.now >= active.start_time + self.metaorder_duration // 2
        ):
            actions.extend(
                self._submit_child(
                    ctx,
                    min(
                        active.unassigned_quantity,
                        self._default_slice_quantity(active),
                    ),
                    aggressive=True,
                    trigger="urgency_fallback",
                )
            )

        actions.append(
            ctx.set_timer(
                self.execution_timer_id,
                fire_at=min(
                    ctx.now + self.slice_interval,
                    active.end_time,
                ),
            )
        )
        return actions

    def _observe_signal(self, ctx: StrategyContext) -> list[Action]:
        self._signal_index += 1
        true_signal = self.world.future_signed_remaining_flow(
            ctx.now,
            self.signal_horizon,
            self.parent_orders,
        )
        observed = self.signal_observation_probability == 1 or ctx.random.random() < self.signal_observation_probability
        if observed:
            noise_bound = math.ceil(abs(true_signal) * self.signal_noise_multiplier) + self.signal_noise_floor
            noise = ctx.random.randint(-noise_bound, noise_bound) if noise_bound else 0
            private_signal = round(self.signal_loading * true_signal) + noise
        else:
            private_signal = 0
        actions: list[Action] = [
            ctx.log(
                "informed_signal",
                fields={
                    "diagnostic_kind": "informed_signal",
                    "participant_id": ctx.participant_id,
                    "signal_index": self._signal_index,
                    "true_future_flow": true_signal,
                    "private_future_flow": private_signal,
                    "signal_horizon_ns": self.signal_horizon,
                    "signal_loading": self.signal_loading,
                    "signal_observation_probability": (self.signal_observation_probability),
                    "signal_observed": observed,
                    "signal_noise_multiplier": self.signal_noise_multiplier,
                    "execution_style": self.execution_style.value,
                    "active_metaorder_id": self._active_id(),
                },
            )
        ]
        if self._active is not None or abs(private_signal) < self.minimum_signal or self._midpoint() is None:
            return actions

        side = Side.BUY if private_signal > 0 else Side.SELL
        total_quantity = min(
            self.maximum_parent_quantity,
            max(
                self.minimum_parent_quantity,
                math.ceil(abs(private_signal) * self.quantity_per_signal_unit),
            ),
        )
        self._metaorder_index += 1
        metaorder_id = f"{ctx.participant_id}-meta-{self._metaorder_index}"
        assert self._midpoint() is not None
        self._active = _Metaorder(
            metaorder_id=metaorder_id,
            side=side,
            total_quantity=total_quantity,
            unassigned_quantity=total_quantity,
            start_time=ctx.now,
            end_time=ctx.now + self.metaorder_duration,
            initial_midpoint=self._midpoint(),
            true_signal=true_signal,
            private_signal=private_signal,
            target_position=self.position + side.signed(total_quantity),
        )
        actions.extend(
            (
                ctx.log(
                    "informed_metaorder_started",
                    fields={
                        "diagnostic_kind": "informed_metaorder_started",
                        "participant_id": ctx.participant_id,
                        "metaorder_id": metaorder_id,
                        "side": side.name.lower(),
                        "total_quantity": total_quantity,
                        "target_position": self._active.target_position,
                        "start_time_ns": ctx.now,
                        "end_time_ns": self._active.end_time,
                        "true_future_flow": true_signal,
                        "private_future_flow": private_signal,
                        "execution_style": self.execution_style.value,
                    },
                ),
                ctx.log(
                    "informed_decision",
                    fields={
                        "diagnostic_kind": "informed_decision",
                        "participant_id": ctx.participant_id,
                        "decision_index": self._signal_index,
                        "latent_innovation": true_signal,
                        "private_innovation": private_signal,
                        "best_bid": self._best_bid,
                        "best_ask": self._best_ask,
                        "side": side.name.lower(),
                        "planned_quantity": total_quantity,
                        "execution_model": self.execution_style.value,
                        "signal_source": "future_flow",
                    },
                ),
            )
        )
        if self.execution_style is ExecutionStyle.BURST:
            actions.extend(
                self._submit_child(
                    ctx,
                    total_quantity,
                    aggressive=True,
                    trigger="burst",
                )
            )
        elif self.execution_style is ExecutionStyle.PATIENT:
            actions.extend(
                self._submit_child(
                    ctx,
                    min(
                        total_quantity,
                        self._default_slice_quantity(self._active),
                    ),
                    aggressive=False,
                    trigger="passive_start",
                )
            )
        elif self.execution_style is ExecutionStyle.SCHEDULED:
            actions.extend(
                self._submit_child(
                    ctx,
                    min(
                        total_quantity,
                        self._default_slice_quantity(self._active),
                    ),
                    aggressive=True,
                    trigger="schedule_start",
                )
            )
        actions.append(
            ctx.set_timer(
                self.execution_timer_id,
                fire_at=min(
                    ctx.now + self.slice_interval,
                    self._active.end_time,
                ),
            )
        )
        return actions

    def _submit_child(
        self,
        ctx: StrategyContext,
        quantity: int,
        *,
        aggressive: bool,
        trigger: str,
    ) -> list[Action]:
        active = self._active
        if active is None or quantity <= 0 or active.unassigned_quantity <= 0:
            return []
        quantity = min(quantity, active.unassigned_quantity)
        price = self._child_price(active.side, aggressive)
        if price is None:
            return []
        active.child_count += 1
        client_order_id = f"informed-{self._metaorder_index}-{active.child_count}"
        active.unassigned_quantity -= quantity
        active.submitted_quantity += quantity
        self._pending[client_order_id] = (active.metaorder_id, quantity)
        self._last_child_at = ctx.now
        return [
            ctx.submit_limit(client_order_id, active.side, price, quantity),
            ctx.log(
                "informed_child_submitted",
                fields={
                    "diagnostic_kind": "informed_child_submitted",
                    "participant_id": ctx.participant_id,
                    "metaorder_id": active.metaorder_id,
                    "child_index": active.child_count,
                    "client_order_id": client_order_id,
                    "side": active.side.name.lower(),
                    "price": price,
                    "quantity": quantity,
                    "aggressive": aggressive,
                    "trigger": trigger,
                    "execution_style": self.execution_style.value,
                },
            ),
        ]

    def _finish_metaorder(
        self,
        ctx: StrategyContext,
        status: str,
    ) -> list[Action]:
        active = self._active
        if active is None:
            return []
        actions: list[Action] = [
            ctx.cancel(order_id)
            for order_id, (metaorder_id, _) in self._owned.items()
            if metaorder_id == active.metaorder_id
        ]
        actions.append(
            ctx.log(
                "informed_metaorder_completed",
                fields={
                    "diagnostic_kind": "informed_metaorder_completed",
                    "participant_id": ctx.participant_id,
                    "metaorder_id": active.metaorder_id,
                    "status": status,
                    "side": active.side.name.lower(),
                    "total_quantity": active.total_quantity,
                    "submitted_quantity": active.submitted_quantity,
                    "filled_quantity": active.filled_quantity,
                    "child_count": active.child_count,
                    "start_time_ns": active.start_time,
                    "end_time_ns": ctx.now,
                    "execution_style": self.execution_style.value,
                },
            )
        )
        self._active = None
        self._recent_public_volume = 0
        return actions

    def _default_slice_quantity(self, active: _Metaorder) -> int:
        divisor = {
            ExecutionStyle.SCHEDULED: 4,
            ExecutionStyle.PARTICIPATION: 4,
            ExecutionStyle.PATIENT: 2,
            ExecutionStyle.MOMENTUM: 3,
            ExecutionStyle.BURST: 1,
        }[self.execution_style]
        return max(1, math.ceil(active.total_quantity / divisor))

    def _child_price(self, side: Side, aggressive: bool) -> int | None:
        if side is Side.BUY:
            if aggressive:
                return None if self._best_ask is None else self._best_ask + self.aggressive_limit_ticks
            return self._best_bid
        if aggressive:
            return None if self._best_bid is None else self._best_bid - self.aggressive_limit_ticks
        return self._best_ask

    def _midpoint(self) -> float | None:
        if self._best_bid is None or self._best_ask is None:
            return None
        return (self._best_bid + self._best_ask) / 2

    def _can_submit_now(self, now: int) -> bool:
        active = self._active
        return (
            active is not None
            and active.unassigned_quantity > 0
            and (self._last_child_at is None or now - self._last_child_at >= self.slice_interval)
        )

    def _has_outstanding(self, metaorder_id: str) -> bool:
        return any(
            child_metaorder_id == metaorder_id
            for child_metaorder_id, _ in (
                *self._pending.values(),
                *self._owned.values(),
            )
        )

    def _active_id(self) -> str | None:
        return None if self._active is None else self._active.metaorder_id


__all__ = ["FutureFlowInformedTrader"]
