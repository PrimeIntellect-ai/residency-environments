"""Small deterministic reference policies built on the shared strategy SDK.

Payload assumptions are intentionally explicit and narrower than a production
exchange schema:

* ``LEVELS`` payloads contain ``bids`` and ``asks`` arrays.  Each entry is a
  mapping with ``price``, ``quantity``, and ``order_count`` fields; best first.
* Order acknowledgements are ``{"event_kind": "order_accepted",
  "client_order_id": ..., "order_id": ...}``.
* Execution payloads may contain an authoritative integer ``position``.  A fill
  uses ``{"event_kind": "fill", "order_id": ..., "side": "buy"|"sell",
  "quantity": n}``.
* Rejections use the exchange event names ``order_rejected`` and
  ``cancel_rejected``.
* Timer payloads contain the strategy's ``timer_id``.

These policies are calibration tools and baseline participants, not claims of
profitable or production-grade trading behavior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from alphaverse.models import Side
from alphaverse.strategy import (
    Action,
    InputEnvelope,
    Strategy,
    StrategyContext,
)
from alphaverse.world import ExecutionStyle, LatentValueProcess, ParentOrder


class _CohortDemandProcess(Protocol):
    """Trusted in-process source of a cohort's current target shift."""

    def target_shift(self, cohort_id: str, at_time: int) -> int:
        """Return the signed target-position shift at ``at_time``."""


def _best_price(payload: Mapping[str, object], key: str) -> int | None:
    levels = payload.get(key)
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)) or not levels:
        return None
    first = levels[0]
    if not isinstance(first, Mapping):
        return None
    price = first.get("price")
    if isinstance(price, bool) or not isinstance(price, int):
        return None
    return price


def _timer_matches(event: InputEnvelope, timer_id: str) -> bool:
    return event.payload.get("timer_id") == timer_id


class InventoryAwareMarketMaker(Strategy):
    """One-level symmetric quoter with volatility and inventory adjustments."""

    def __init__(
        self,
        *,
        refresh_interval: int,
        quote_quantity: int,
        base_half_spread: int,
        volatility_multiplier: float = 1.0,
        volatility_lookback: int = 8,
        inventory_skew_per_unit: float = 0.0,
        timer_id: str = "quote-refresh",
    ) -> None:
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        if quote_quantity <= 0:
            raise ValueError("quote_quantity must be positive")
        if base_half_spread <= 0:
            raise ValueError("base_half_spread must be positive")
        if not math.isfinite(volatility_multiplier) or volatility_multiplier < 0:
            raise ValueError("volatility_multiplier must be finite and non-negative")
        if volatility_lookback <= 0:
            raise ValueError("volatility_lookback must be positive")
        if not math.isfinite(inventory_skew_per_unit):
            raise ValueError("inventory_skew_per_unit must be finite")
        if not timer_id:
            raise ValueError("timer_id must not be empty")

        self.refresh_interval = refresh_interval
        self.quote_quantity = quote_quantity
        self.base_half_spread = base_half_spread
        self.volatility_multiplier = volatility_multiplier
        self.volatility_lookback = volatility_lookback
        self.inventory_skew_per_unit = inventory_skew_per_unit
        self.timer_id = timer_id

        self.position = 0
        self._midpoints: list[float] = []
        self._active_quote_order_ids: dict[str, str] = {}
        self._pending_quote_clients: set[str] = set()
        self._quote_sequence = 0

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [ctx.set_timer(self.timer_id, fire_at=ctx.now + self.refresh_interval)]

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope):
        bid = _best_price(event.payload, "bids")
        ask = _best_price(event.payload, "asks")
        if bid is not None and ask is not None:
            midpoint = (bid + ask) / 2
            self._midpoints.append(midpoint)
            # One extra midpoint is required to produce N absolute changes.
            del self._midpoints[: -(self.volatility_lookback + 1)]
        return None

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        payload = event.payload
        position = payload.get("position")
        if isinstance(position, int) and not isinstance(position, bool):
            self.position = position
        elif payload.get("event_kind") == "fill":
            quantity = payload.get("quantity")
            side = payload.get("side")
            if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity > 0:
                if side == "buy":
                    self.position += quantity
                elif side == "sell":
                    self.position -= quantity

        if payload.get("event_kind") == "order_accepted":
            client_order_id = payload.get("client_order_id")
            order_id = payload.get("order_id")
            if (
                isinstance(client_order_id, str)
                and isinstance(order_id, str)
                and client_order_id in self._pending_quote_clients
            ):
                self._pending_quote_clients.remove(client_order_id)
                self._active_quote_order_ids[client_order_id] = order_id
        elif payload.get("event_kind") == "cancel_accepted":
            cancelled_order_id = payload.get("order_id")
            if isinstance(cancelled_order_id, str):
                self._active_quote_order_ids = {
                    client_id: order_id
                    for client_id, order_id in self._active_quote_order_ids.items()
                    if order_id != cancelled_order_id
                }
        return None

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        if not _timer_matches(event, self.timer_id):
            return None

        actions: list[Action] = []
        for order_id in self._active_quote_order_ids.values():
            actions.append(ctx.cancel(order_id))
        self._active_quote_order_ids.clear()

        if self._midpoints:
            self._quote_sequence += 1
            center = self._midpoints[-1] - self.position * self.inventory_skew_per_unit
            half_spread = self.base_half_spread + self._volatility_widening()
            bid_price = math.floor(center - half_spread)
            ask_price = math.ceil(center + half_spread)
            bid_client_id = f"mm-bid-{self._quote_sequence}"
            ask_client_id = f"mm-ask-{self._quote_sequence}"
            self._pending_quote_clients.update((bid_client_id, ask_client_id))
            actions.extend(
                (
                    ctx.submit_limit(bid_client_id, Side.BUY, bid_price, self.quote_quantity),
                    ctx.submit_limit(ask_client_id, Side.SELL, ask_price, self.quote_quantity),
                )
            )

        actions.append(ctx.set_timer(self.timer_id, fire_at=ctx.now + self.refresh_interval))
        return actions

    def _volatility_widening(self) -> int:
        changes = [abs(current - previous) for previous, current in zip(self._midpoints, self._midpoints[1:])]
        if not changes:
            return 0
        mean_absolute_change = sum(changes) / len(changes)
        return math.ceil(self.volatility_multiplier * mean_absolute_change)


class AdaptiveMarketMaker(Strategy):
    """Maker that learns short-horizon toxicity from its own executions.

    The policy deliberately consumes only ordinary maker information: public
    levels and private acknowledgements, fills, and account updates. A fill
    causes an immediate cancel/requote. Recent signed fill flow shifts the quote
    center, while adverse post-fill midpoint markouts widen the spread. Both
    signals decay in market time.
    """

    def __init__(
        self,
        *,
        refresh_interval: int,
        initial_refresh_delay: int,
        quote_quantity: int,
        base_half_spread: int,
        volatility_multiplier: float = 1.0,
        volatility_lookback: int = 8,
        inventory_skew_per_unit: float = 0.0,
        inventory_soft_limit: int,
        minimum_quote_fraction: float = 0.25,
        fill_pressure_per_unit: float = 0.02,
        fill_pressure_half_life: int,
        maximum_fill_skew: float = 4.0,
        markout_horizon: int,
        markout_learning_rate: float = 0.25,
        toxicity_widening_multiplier: float = 1.0,
        toxicity_half_life: int,
        maximum_toxicity_widening: int = 6,
        directional_toxicity: bool = False,
        timer_id: str = "adaptive-quote-refresh",
        fill_requote_timer_id: str = "adaptive-fill-requote",
        schedule_fill_requote: bool = True,
    ) -> None:
        if refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")
        if initial_refresh_delay < 0:
            raise ValueError("initial_refresh_delay must be non-negative")
        if quote_quantity <= 0:
            raise ValueError("quote_quantity must be positive")
        if base_half_spread <= 0:
            raise ValueError("base_half_spread must be positive")
        if not math.isfinite(volatility_multiplier) or volatility_multiplier < 0:
            raise ValueError("volatility_multiplier must be finite and non-negative")
        if volatility_lookback <= 0:
            raise ValueError("volatility_lookback must be positive")
        if not math.isfinite(inventory_skew_per_unit):
            raise ValueError("inventory_skew_per_unit must be finite")
        if inventory_soft_limit <= 0:
            raise ValueError("inventory_soft_limit must be positive")
        if not math.isfinite(minimum_quote_fraction) or not 0 < minimum_quote_fraction <= 1:
            raise ValueError("minimum_quote_fraction must be in (0, 1]")
        if not math.isfinite(fill_pressure_per_unit) or fill_pressure_per_unit < 0:
            raise ValueError("fill_pressure_per_unit must be finite and non-negative")
        if fill_pressure_half_life <= 0:
            raise ValueError("fill_pressure_half_life must be positive")
        if not math.isfinite(maximum_fill_skew) or maximum_fill_skew < 0:
            raise ValueError("maximum_fill_skew must be finite and non-negative")
        if markout_horizon <= 0:
            raise ValueError("markout_horizon must be positive")
        if not math.isfinite(markout_learning_rate) or not 0 < markout_learning_rate <= 1:
            raise ValueError("markout_learning_rate must be in (0, 1]")
        if not math.isfinite(toxicity_widening_multiplier) or toxicity_widening_multiplier < 0:
            raise ValueError("toxicity_widening_multiplier must be finite and non-negative")
        if toxicity_half_life <= 0:
            raise ValueError("toxicity_half_life must be positive")
        if maximum_toxicity_widening < 0:
            raise ValueError("maximum_toxicity_widening must be non-negative")
        if not isinstance(directional_toxicity, bool):
            raise TypeError("directional_toxicity must be a bool")
        if not timer_id:
            raise ValueError("timer_id must not be empty")
        if not fill_requote_timer_id:
            raise ValueError("fill_requote_timer_id must not be empty")
        if fill_requote_timer_id == timer_id:
            raise ValueError("fill_requote_timer_id must differ from timer_id")
        if not isinstance(schedule_fill_requote, bool):
            raise TypeError("schedule_fill_requote must be a bool")

        self.refresh_interval = refresh_interval
        self.initial_refresh_delay = initial_refresh_delay
        self.quote_quantity = quote_quantity
        self.base_half_spread = base_half_spread
        self.volatility_multiplier = volatility_multiplier
        self.volatility_lookback = volatility_lookback
        self.inventory_skew_per_unit = inventory_skew_per_unit
        self.inventory_soft_limit = inventory_soft_limit
        self.minimum_quote_fraction = minimum_quote_fraction
        self.fill_pressure_per_unit = fill_pressure_per_unit
        self.fill_pressure_half_life = fill_pressure_half_life
        self.maximum_fill_skew = maximum_fill_skew
        self.markout_horizon = markout_horizon
        self.markout_learning_rate = markout_learning_rate
        self.toxicity_widening_multiplier = toxicity_widening_multiplier
        self.toxicity_half_life = toxicity_half_life
        self.maximum_toxicity_widening = maximum_toxicity_widening
        self.directional_toxicity = directional_toxicity
        self.timer_id = timer_id
        self.fill_requote_timer_id = fill_requote_timer_id
        self.schedule_fill_requote = schedule_fill_requote

        self.position = 0
        self.fill_pressure = 0.0
        self.adverse_markout = 0.0
        self.buy_adverse_markout = 0.0
        self.sell_adverse_markout = 0.0
        self._last_decay_at: int | None = None
        self._latest_midpoint: float | None = None
        self._midpoints: list[float] = []
        self._pending_markouts: list[tuple[int, Side, int, int]] = []
        self._active_quote_order_ids: dict[str, str] = {}
        self._active_quote_remaining: dict[str, int] = {}
        self._pending_quote_quantities: dict[str, int] = {}
        self._quote_sequence = 0
        self._next_regular_refresh_at: int | None = None
        self._fill_requote_pending_at: int | None = None

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        self._next_regular_refresh_at = ctx.now + self.initial_refresh_delay
        return [
            ctx.set_timer(
                self.timer_id,
                fire_at=self._next_regular_refresh_at,
            )
        ]

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope):
        self._decay_signals(ctx.now)
        bid = _best_price(event.payload, "bids")
        ask = _best_price(event.payload, "asks")
        if bid is None or ask is None:
            return None

        midpoint = (bid + ask) / 2
        self._latest_midpoint = midpoint
        self._midpoints.append(midpoint)
        del self._midpoints[: -(self.volatility_lookback + 1)]
        self._settle_markouts(ctx.now, midpoint)
        return None

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        self._decay_signals(ctx.now)
        payload = event.payload
        event_kind = payload.get("event_kind")

        position = payload.get("position")
        if isinstance(position, int) and not isinstance(position, bool):
            self.position = position

        if event_kind == "order_accepted":
            client_order_id = payload.get("client_order_id")
            order_id = payload.get("order_id")
            if (
                isinstance(client_order_id, str)
                and isinstance(order_id, str)
                and client_order_id in self._pending_quote_quantities
            ):
                quantity = self._pending_quote_quantities.pop(client_order_id)
                self._active_quote_order_ids[client_order_id] = order_id
                self._active_quote_remaining[order_id] = quantity
            return None

        if event_kind in ("cancel_accepted", "cancel_rejected"):
            order_id = payload.get("order_id")
            if isinstance(order_id, str):
                self._forget_order(order_id)
            return None

        if event_kind == "order_rejected":
            client_order_id = payload.get("client_order_id")
            if isinstance(client_order_id, str):
                self._pending_quote_quantities.pop(client_order_id, None)
            return None

        if event_kind != "fill":
            return None

        side_value = payload.get("side")
        price = payload.get("price")
        quantity = payload.get("quantity")
        order_id = payload.get("order_id")
        if (
            side_value not in ("buy", "sell")
            or isinstance(price, bool)
            or not isinstance(price, int)
            or isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or quantity <= 0
        ):
            return None

        side = Side.BUY if side_value == "buy" else Side.SELL
        if position is None:
            self.position += side.signed(quantity)
        if isinstance(order_id, str):
            remaining = self._active_quote_remaining.get(order_id)
            if remaining is not None:
                remaining -= quantity
                if remaining <= 0:
                    self._forget_order(order_id)
                else:
                    self._active_quote_remaining[order_id] = remaining

        # A maker sale indicates aggressive buying pressure; a maker buy
        # indicates aggressive selling pressure.
        direction = 1 if side is Side.SELL else -1
        self.fill_pressure += direction * quantity
        self._pending_markouts.append((ctx.now + self.markout_horizon, side, price, quantity))
        if not self.schedule_fill_requote:
            return None
        if self._next_regular_refresh_at is not None and ctx.now >= self._next_regular_refresh_at:
            return None
        if self._fill_requote_pending_at is not None:
            return None
        self._fill_requote_pending_at = ctx.now
        return [ctx.set_timer(self.fill_requote_timer_id, fire_at=ctx.now)]

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        is_regular_refresh = _timer_matches(event, self.timer_id)
        is_fill_requote = _timer_matches(event, self.fill_requote_timer_id)
        if not is_regular_refresh and not is_fill_requote:
            return None
        if is_fill_requote:
            self._fill_requote_pending_at = None
            if self._next_regular_refresh_at is not None and ctx.now >= self._next_regular_refresh_at:
                return None

        self._decay_signals(ctx.now)
        actions: list[Action] = [ctx.cancel(order_id) for order_id in self._active_quote_order_ids.values()]
        self._active_quote_order_ids.clear()
        self._active_quote_remaining.clear()

        quote_plan = self._quote_plan()
        if quote_plan is not None:
            self._quote_sequence += 1
            bid_price, bid_quantity, ask_price, ask_quantity = quote_plan
            bid_client_id = f"adaptive-mm-bid-{self._quote_sequence}"
            ask_client_id = f"adaptive-mm-ask-{self._quote_sequence}"
            self._pending_quote_quantities.update(
                {
                    bid_client_id: bid_quantity,
                    ask_client_id: ask_quantity,
                }
            )
            actions.extend(
                (
                    ctx.submit_limit(
                        bid_client_id,
                        Side.BUY,
                        bid_price,
                        bid_quantity,
                    ),
                    ctx.submit_limit(
                        ask_client_id,
                        Side.SELL,
                        ask_price,
                        ask_quantity,
                    ),
                )
            )

        if is_regular_refresh:
            self._next_regular_refresh_at = ctx.now + self.refresh_interval
            actions.append(
                ctx.set_timer(
                    self.timer_id,
                    fire_at=self._next_regular_refresh_at,
                )
            )
        return actions

    def _quote_plan(self) -> tuple[int, int, int, int] | None:
        if self._latest_midpoint is None:
            return None
        center = self._latest_midpoint + self._fill_skew() - self.position * self.inventory_skew_per_unit
        common_half_spread = self.base_half_spread + self._volatility_widening()
        bid_price = math.floor(center - common_half_spread - self._toxicity_widening(Side.BUY))
        ask_price = math.ceil(center + common_half_spread + self._toxicity_widening(Side.SELL))
        bid_quantity, ask_quantity = self._quote_quantities()
        return bid_price, bid_quantity, ask_price, ask_quantity

    def _decay_signals(self, now: int) -> None:
        if self._last_decay_at is None:
            self._last_decay_at = now
            return
        elapsed = now - self._last_decay_at
        if elapsed <= 0:
            return
        self.fill_pressure *= 2 ** (-elapsed / self.fill_pressure_half_life)
        self.adverse_markout *= 2 ** (-elapsed / self.toxicity_half_life)
        self.buy_adverse_markout *= 2 ** (-elapsed / self.toxicity_half_life)
        self.sell_adverse_markout *= 2 ** (-elapsed / self.toxicity_half_life)
        self._last_decay_at = now

    def _settle_markouts(self, now: int, midpoint: float) -> None:
        pending: list[tuple[int, Side, int, int]] = []
        for due_at, side, fill_price, quantity in self._pending_markouts:
            if due_at > now:
                pending.append((due_at, side, fill_price, quantity))
                continue
            adverse = fill_price - midpoint if side is Side.BUY else midpoint - fill_price
            weight = min(
                1.0,
                self.markout_learning_rate * quantity / self.quote_quantity,
            )
            self.adverse_markout = (1 - weight) * self.adverse_markout + weight * adverse
            if side is Side.BUY:
                self.buy_adverse_markout = (1 - weight) * self.buy_adverse_markout + weight * adverse
            else:
                self.sell_adverse_markout = (1 - weight) * self.sell_adverse_markout + weight * adverse
        self._pending_markouts = pending

    def _fill_skew(self) -> float:
        raw = self.fill_pressure * self.fill_pressure_per_unit
        return max(-self.maximum_fill_skew, min(self.maximum_fill_skew, raw))

    def _toxicity_widening(self, side: Side) -> int:
        markout = self.adverse_markout
        if self.directional_toxicity:
            markout = self.buy_adverse_markout if side is Side.BUY else self.sell_adverse_markout
        raw = math.ceil(max(0.0, markout) * self.toxicity_widening_multiplier)
        return min(self.maximum_toxicity_widening, raw)

    def _volatility_widening(self) -> int:
        changes = [abs(current - previous) for previous, current in zip(self._midpoints, self._midpoints[1:])]
        if not changes:
            return 0
        return math.ceil(self.volatility_multiplier * sum(changes) / len(changes))

    def _quote_quantities(self) -> tuple[int, int]:
        inventory_fraction = min(1.0, abs(self.position) / self.inventory_soft_limit)
        worsening_fraction = max(
            self.minimum_quote_fraction,
            1 - (1 - self.minimum_quote_fraction) * inventory_fraction,
        )
        reduced_quantity = max(1, math.ceil(self.quote_quantity * worsening_fraction))
        if self.position > 0:
            return reduced_quantity, self.quote_quantity
        if self.position < 0:
            return self.quote_quantity, reduced_quantity
        return self.quote_quantity, self.quote_quantity

    def _forget_order(self, order_id: str) -> None:
        self._active_quote_remaining.pop(order_id, None)
        self._active_quote_order_ids = {
            client_id: active_order_id
            for client_id, active_order_id in self._active_quote_order_ids.items()
            if active_order_id != order_id
        }


class EventDrivenAdaptiveMarketMaker(AdaptiveMarketMaker):
    """Adaptive maker that reassesses quotes after each coherent level update.

    Reassessment and replacement are deliberately separate.  Each level event
    updates fair-value and risk state, but a live quote keeps its FIFO position
    unless it has become too aggressive, sufficiently uncompetitive, or too
    large for the current inventory target.  The reference book can exclude
    the strategy's own displayed quantity so quote updates do not chase level
    changes caused by the strategy itself.

    A periodic timer remains as a watchdog for signals that decay while the
    public book is quiet; it does not force quote replacement.
    """

    def __init__(
        self,
        *,
        improvement_reprice_ticks: int = 1,
        retreat_reprice_ticks: int = 1,
        external_book_reference: bool = True,
        replenish_partial_fills: bool = False,
        **kwargs,
    ) -> None:
        # Event-driven makers reconcile on the coherent level snapshot emitted
        # after a fill, so the base strategy must not enqueue a second callback.
        kwargs.setdefault("schedule_fill_requote", False)
        super().__init__(**kwargs)
        if improvement_reprice_ticks <= 0:
            raise ValueError("improvement_reprice_ticks must be positive")
        if retreat_reprice_ticks <= 0:
            raise ValueError("retreat_reprice_ticks must be positive")
        if not isinstance(external_book_reference, bool):
            raise TypeError("external_book_reference must be a bool")
        if not isinstance(replenish_partial_fills, bool):
            raise TypeError("replenish_partial_fills must be a bool")

        self.improvement_reprice_ticks = improvement_reprice_ticks
        self.retreat_reprice_ticks = retreat_reprice_ticks
        self.external_book_reference = external_book_reference
        self.replenish_partial_fills = replenish_partial_fills
        self._event_pending_quotes: dict[str, tuple[Side, int, int, int]] = {}
        self._event_active_quotes: dict[Side, tuple[str, int, int, int]] = {}

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope):
        # An action batch can emit cancel, bid, and ask level snapshots at the
        # same timestamp.  Wait for our acknowledgement chain to finish rather
        # than interpreting those intermediate self-generated snapshots as new
        # market information.
        if self._event_pending_quotes:
            return None

        self._decay_signals(ctx.now)
        bid = self._reference_best_price(event.payload, "bids", Side.BUY)
        ask = self._reference_best_price(event.payload, "asks", Side.SELL)
        if bid is None or ask is None:
            return None

        midpoint = (bid + ask) / 2
        self._latest_midpoint = midpoint
        self._midpoints.append(midpoint)
        del self._midpoints[: -(self.volatility_lookback + 1)]
        self._settle_markouts(ctx.now, midpoint)
        return self._reconcile_quotes(ctx)

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        # Retain the adaptive signal/accounting behavior but coalesce fills and
        # acknowledgements into the following event-end level snapshot.
        super().on_execution(ctx, event)
        payload = event.payload
        event_kind = payload.get("event_kind")

        if event_kind == "order_accepted":
            client_order_id = payload.get("client_order_id")
            order_id = payload.get("order_id")
            if isinstance(client_order_id, str) and isinstance(order_id, str):
                pending = self._event_pending_quotes.pop(client_order_id, None)
                if pending is not None:
                    side, price, quantity, submitted_at = pending
                    self._event_active_quotes[side] = (
                        order_id,
                        price,
                        quantity,
                        submitted_at,
                    )
            return None

        if event_kind == "order_rejected":
            client_order_id = payload.get("client_order_id")
            if isinstance(client_order_id, str):
                self._event_pending_quotes.pop(client_order_id, None)
            return None

        order_id = payload.get("order_id")
        if not isinstance(order_id, str):
            return None

        if event_kind in ("cancel_accepted", "cancel_rejected"):
            self._forget_event_order(order_id)
            return None

        if event_kind != "fill":
            return None
        quantity = payload.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            return None
        for side, quote in tuple(self._event_active_quotes.items()):
            active_order_id, price, remaining, submitted_at = quote
            if active_order_id != order_id:
                continue
            remaining -= quantity
            if remaining <= 0:
                self._event_active_quotes.pop(side, None)
            else:
                self._event_active_quotes[side] = (
                    active_order_id,
                    price,
                    remaining,
                    submitted_at,
                )
            break
        return None

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        if not _timer_matches(event, self.timer_id):
            return None
        self._decay_signals(ctx.now)
        actions = self._reconcile_quotes(ctx)
        actions.append(
            ctx.set_timer(
                self.timer_id,
                fire_at=ctx.now + self.refresh_interval,
            )
        )
        return actions

    def _reconcile_quotes(self, ctx: StrategyContext) -> list[Action]:
        quote_plan = self._quote_plan()
        if quote_plan is None:
            return []
        bid_price, bid_quantity, ask_price, ask_quantity = quote_plan
        desired = (
            (Side.BUY, bid_price, bid_quantity),
            (Side.SELL, ask_price, ask_quantity),
        )
        pending_sides = {pending[0] for pending in self._event_pending_quotes.values()}
        actions: list[Action] = []
        for side, price, quantity in desired:
            if side in pending_sides:
                continue
            active = self._event_active_quotes.get(side)
            if active is None:
                actions.append(self._submit_event_quote(ctx, side, price, quantity))
                continue
            order_id, active_price, remaining, _ = active
            if not self._should_replace(
                side,
                active_price=active_price,
                active_quantity=remaining,
                desired_price=price,
                desired_quantity=quantity,
            ):
                continue
            actions.append(ctx.cancel(order_id))
            self._event_active_quotes.pop(side, None)
            self._forget_order(order_id)
            actions.append(self._submit_event_quote(ctx, side, price, quantity))
        return actions

    def _should_replace(
        self,
        side: Side,
        *,
        active_price: int,
        active_quantity: int,
        desired_price: int,
        desired_quantity: int,
    ) -> bool:
        if active_quantity > desired_quantity:
            return True
        if self.replenish_partial_fills and active_quantity < desired_quantity:
            return True
        if active_price == desired_price:
            return False

        too_aggressive = active_price > desired_price if side is Side.BUY else active_price < desired_price
        distance = abs(active_price - desired_price)
        threshold = self.retreat_reprice_ticks if too_aggressive else self.improvement_reprice_ticks
        return distance >= threshold

    def _submit_event_quote(
        self,
        ctx: StrategyContext,
        side: Side,
        price: int,
        quantity: int,
    ) -> Action:
        self._quote_sequence += 1
        label = "bid" if side is Side.BUY else "ask"
        client_order_id = f"event-mm-{label}-{self._quote_sequence}"
        self._pending_quote_quantities[client_order_id] = quantity
        self._event_pending_quotes[client_order_id] = (
            side,
            price,
            quantity,
            ctx.now,
        )
        return ctx.submit_limit(client_order_id, side, price, quantity)

    def _reference_best_price(
        self,
        payload: Mapping[str, object],
        key: str,
        side: Side,
    ) -> int | None:
        if not self.external_book_reference:
            return _best_price(payload, key)
        levels = payload.get(key)
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
            return None
        own_by_price: dict[int, int] = {}
        active = self._event_active_quotes.get(side)
        if active is not None:
            _, price, remaining, _ = active
            own_by_price[price] = own_by_price.get(price, 0) + remaining
        for level in levels:
            if not isinstance(level, Mapping):
                continue
            price = level.get("price")
            quantity = level.get("quantity")
            if (
                isinstance(price, bool)
                or not isinstance(price, int)
                or isinstance(quantity, bool)
                or not isinstance(quantity, int)
            ):
                continue
            if quantity > own_by_price.get(price, 0):
                return price
        return None

    def _forget_event_order(self, order_id: str) -> None:
        for side, quote in tuple(self._event_active_quotes.items()):
            if quote[0] == order_id:
                self._event_active_quotes.pop(side, None)


class ScheduledNoiseExecutor(Strategy):
    """Slice one hidden parent mandate using observed top-of-book levels."""

    _ALWAYS_AGGRESSIVE = frozenset((ExecutionStyle.MOMENTUM, ExecutionStyle.BURST))

    def __init__(
        self,
        parent_order: ParentOrder,
        *,
        slice_interval: int,
        max_slice_quantity: int,
        timer_id: str = "parent-slice",
    ) -> None:
        if not isinstance(parent_order, ParentOrder):
            raise TypeError("parent_order must be a ParentOrder")
        if slice_interval <= 0:
            raise ValueError("slice_interval must be positive")
        if max_slice_quantity <= 0:
            raise ValueError("max_slice_quantity must be positive")
        if not timer_id:
            raise ValueError("timer_id must not be empty")
        self.parent_order = parent_order
        self.slice_interval = slice_interval
        self.max_slice_quantity = max_slice_quantity
        self.timer_id = timer_id

        self.filled_quantity = 0
        self.submitted_quantity = 0
        self._best_bid: int | None = None
        self._best_ask: int | None = None
        self._slice_sequence = 0
        self._submitted_client_ids: set[str] = set()
        self._owned_order_ids: set[str] = set()
        self._stop_requested = False

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        if ctx.now >= self.parent_order.end_time:
            self._stop_requested = True
            return [ctx.stop("parent_order_expired")]
        first_fire = max(ctx.now, self.parent_order.start_time)
        return [ctx.set_timer(self.timer_id, fire_at=first_fire)]

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope):
        self._best_bid = _best_price(event.payload, "bids")
        self._best_ask = _best_price(event.payload, "asks")
        return None

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        payload = event.payload
        event_kind = payload.get("event_kind")
        client_order_id = payload.get("client_order_id")
        if event_kind == "order_accepted":
            order_id = payload.get("order_id")
            if (
                isinstance(client_order_id, str)
                and client_order_id in self._submitted_client_ids
                and isinstance(order_id, str)
            ):
                self._owned_order_ids.add(order_id)
            return None

        order_id = payload.get("order_id")
        if event_kind == "fill" and order_id not in self._owned_order_ids:
            return None
        if event_kind == "order_rejected" and (
            not isinstance(client_order_id, str) or client_order_id not in self._submitted_client_ids
        ):
            return None

        quantity = payload.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            return None
        if event_kind == "fill":
            self.filled_quantity = min(self.parent_order.total_quantity, self.filled_quantity + quantity)
            if self.filled_quantity >= self.parent_order.total_quantity:
                self._stop_requested = True
                return [ctx.stop("parent_order_complete")]
        elif event_kind == "order_rejected":
            self.submitted_quantity = max(0, self.submitted_quantity - quantity)
        return None

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        if not _timer_matches(event, self.timer_id) or self._stop_requested:
            return None
        if self.filled_quantity >= self.parent_order.total_quantity:
            self._stop_requested = True
            return [ctx.stop("parent_order_complete")]
        if ctx.now >= self.parent_order.end_time:
            self._stop_requested = True
            return [ctx.stop("parent_order_expired")]

        actions: list[Action] = []
        unsent_quantity = self.parent_order.total_quantity - self.submitted_quantity
        price = self._slice_price(ctx.now)
        if unsent_quantity > 0 and price is not None:
            quantity = min(self.max_slice_quantity, unsent_quantity)
            self._slice_sequence += 1
            client_order_id = f"noise-slice-{self._slice_sequence}"
            self._submitted_client_ids.add(client_order_id)
            self.submitted_quantity += quantity
            actions.append(
                ctx.submit_limit(
                    client_order_id,
                    self.parent_order.side,
                    price,
                    quantity,
                )
            )

        next_fire = min(ctx.now + self.slice_interval, self.parent_order.end_time)
        actions.append(ctx.set_timer(self.timer_id, fire_at=next_fire))
        return actions

    def _slice_price(self, now: int) -> int | None:
        aggressive = self.parent_order.execution_style in self._ALWAYS_AGGRESSIVE
        if self.parent_order.execution_style is ExecutionStyle.PATIENT:
            aggressive = now + self.slice_interval >= self.parent_order.end_time

        if self.parent_order.side is Side.BUY:
            return self._best_ask if aggressive else self._best_bid
        return self._best_bid if aggressive else self._best_ask


class RollingNoiseExecutor(Strategy):
    """Execute repeated hidden mandates from one persistent institution account.

    A scenario gives this strategy every mandate assigned to its participant.
    Mandates are non-overlapping, so the executor can retire one execution
    state at a time while retaining its account, inventory, and strategy actor
    over the entire market horizon.
    """

    _ALWAYS_AGGRESSIVE = ScheduledNoiseExecutor._ALWAYS_AGGRESSIVE

    def __init__(
        self,
        parent_orders: tuple[ParentOrder, ...],
        *,
        slice_interval: int,
        max_slice_quantity: int,
        timer_id: str = "rolling-parent-slice",
    ) -> None:
        if not parent_orders:
            raise ValueError("parent_orders must not be empty")
        if not all(isinstance(parent, ParentOrder) for parent in parent_orders):
            raise TypeError("parent_orders must contain ParentOrder values")
        participant_ids = {parent.participant_id for parent in parent_orders}
        if len(participant_ids) != 1:
            raise ValueError("parent_orders must belong to one participant")
        if tuple(sorted(parent_orders, key=lambda parent: parent.start_time)) != parent_orders:
            raise ValueError("parent_orders must be sorted by start_time")
        if any(previous.end_time > current.start_time for previous, current in zip(parent_orders, parent_orders[1:])):
            raise ValueError("parent_orders must not overlap")
        if slice_interval <= 0:
            raise ValueError("slice_interval must be positive")
        if max_slice_quantity <= 0:
            raise ValueError("max_slice_quantity must be positive")
        if not timer_id:
            raise ValueError("timer_id must not be empty")

        self.parent_orders = parent_orders
        self.slice_interval = slice_interval
        self.max_slice_quantity = max_slice_quantity
        self.timer_id = timer_id
        self.participant_id = next(iter(participant_ids))

        self._best_bid: int | None = None
        self._best_ask: int | None = None
        self._parent_index = 0
        self._slice_sequence = 0
        self._submitted_by_parent = [0] * len(parent_orders)
        self._filled_by_parent = [0] * len(parent_orders)
        self._pending: dict[str, tuple[int, int]] = {}
        self._owned: dict[str, tuple[int, int]] = {}

    @property
    def current_parent_order(self) -> ParentOrder | None:
        if self._parent_index >= len(self.parent_orders):
            return None
        return self.parent_orders[self._parent_index]

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [
            ctx.set_timer(
                self.timer_id,
                fire_at=max(ctx.now, self.parent_orders[0].start_time),
            )
        ]

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope):
        self._best_bid = _best_price(event.payload, "bids")
        self._best_ask = _best_price(event.payload, "asks")
        return None

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        payload = event.payload
        event_kind = payload.get("event_kind")
        client_order_id = payload.get("client_order_id")
        if event_kind == "order_accepted":
            order_id = payload.get("order_id")
            pending = self._pending.pop(client_order_id, None) if isinstance(client_order_id, str) else None
            if isinstance(order_id, str) and pending is not None:
                parent_index, quantity = pending
                self._owned[order_id] = (parent_index, quantity)
                if parent_index < self._parent_index:
                    # An action may be accepted after its mandate has expired
                    # because exchange latency crosses the boundary.
                    return [ctx.cancel(order_id)]
            return None
        if event_kind == "cancel_accepted":
            order_id = payload.get("order_id")
            if isinstance(order_id, str):
                self._owned.pop(order_id, None)
            return None
        if event_kind == "cancel_rejected":
            order_id = payload.get("order_id")
            if isinstance(order_id, str):
                # A rejected cancellation means the order is no longer a live
                # slice to carry into later mandate retirement attempts.
                self._owned.pop(order_id, None)
            return None
        if event_kind == "order_rejected":
            pending = self._pending.pop(client_order_id, None) if isinstance(client_order_id, str) else None
            if pending is not None:
                parent_index, quantity = pending
                self._submitted_by_parent[parent_index] = max(0, self._submitted_by_parent[parent_index] - quantity)
            return None
        if event_kind != "fill":
            return None

        order_id = payload.get("order_id")
        quantity = payload.get("quantity")
        owned = self._owned.get(order_id) if isinstance(order_id, str) else None
        if owned is None or isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            return None
        parent_index, remaining = owned
        parent = self.parent_orders[parent_index]
        self._filled_by_parent[parent_index] = min(
            parent.total_quantity,
            self._filled_by_parent[parent_index] + quantity,
        )
        assert isinstance(order_id, str)
        if quantity >= remaining:
            self._owned.pop(order_id, None)
        else:
            self._owned[order_id] = (parent_index, remaining - quantity)
        return None

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        if not _timer_matches(event, self.timer_id):
            return None
        parent = self.current_parent_order
        if parent is None:
            return None
        if ctx.now < parent.start_time:
            return [ctx.set_timer(self.timer_id, fire_at=parent.start_time)]

        actions: list[Action] = []
        if ctx.now >= parent.end_time or self._filled_by_parent[self._parent_index] >= parent.total_quantity:
            actions.extend(self._retire_current_parent(ctx))
            self._parent_index += 1
            next_parent = self.current_parent_order
            if next_parent is not None:
                actions.append(
                    ctx.set_timer(
                        self.timer_id,
                        fire_at=max(ctx.now, next_parent.start_time),
                    )
                )
            return actions

        unsent_quantity = parent.total_quantity - self._submitted_by_parent[self._parent_index]
        price = self._slice_price(parent, ctx.now)
        if unsent_quantity > 0 and price is not None:
            quantity = min(self.max_slice_quantity, unsent_quantity)
            self._slice_sequence += 1
            client_order_id = f"rolling-noise-{self._parent_index + 1}-{self._slice_sequence}"
            self._pending[client_order_id] = (self._parent_index, quantity)
            self._submitted_by_parent[self._parent_index] += quantity
            actions.append(ctx.submit_limit(client_order_id, parent.side, price, quantity))
        actions.append(
            ctx.set_timer(
                self.timer_id,
                fire_at=min(ctx.now + self.slice_interval, parent.end_time),
            )
        )
        return actions

    def _retire_current_parent(self, ctx: StrategyContext) -> list[Action]:
        """Cancel residual passive slices before the institution changes mandate."""

        return [
            ctx.cancel(order_id)
            for order_id, (parent_index, _) in self._owned.items()
            if parent_index == self._parent_index
        ]

    def _slice_price(self, parent: ParentOrder, now: int) -> int | None:
        aggressive = parent.execution_style in self._ALWAYS_AGGRESSIVE
        if parent.execution_style is ExecutionStyle.PATIENT:
            aggressive = now + self.slice_interval >= parent.end_time
        if parent.side is Side.BUY:
            return self._best_ask if aggressive else self._best_bid
        return self._best_bid if aggressive else self._best_ask


class PersistentNoiseTrader(Strategy):
    """Recurring heterogeneous flow with clustered signs and inventory control."""

    def __init__(
        self,
        *,
        min_interval: int,
        max_interval: int,
        min_quantity: int,
        max_quantity: int,
        aggressive_probability: float,
        side_persistence: float,
        soft_inventory_limit: int,
        timer_id: str = "noise-arrival",
    ) -> None:
        if min_interval <= 0:
            raise ValueError("min_interval must be positive")
        if max_interval < min_interval:
            raise ValueError("max_interval must be at least min_interval")
        if min_quantity <= 0:
            raise ValueError("min_quantity must be positive")
        if max_quantity < min_quantity:
            raise ValueError("max_quantity must be at least min_quantity")
        for name, value in (
            ("aggressive_probability", aggressive_probability),
            ("side_persistence", side_persistence),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if soft_inventory_limit <= 0:
            raise ValueError("soft_inventory_limit must be positive")
        if not timer_id:
            raise ValueError("timer_id must not be empty")

        self.min_interval = min_interval
        self.max_interval = max_interval
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.aggressive_probability = aggressive_probability
        self.side_persistence = side_persistence
        self.soft_inventory_limit = soft_inventory_limit
        self.timer_id = timer_id

        self.position = 0
        self._best_bid: int | None = None
        self._best_ask: int | None = None
        self._last_side: Side | None = None
        self._sequence = 0
        self._pending: dict[str, tuple[Side, int]] = {}
        self._owned: dict[str, tuple[Side, int]] = {}

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [self._next_timer(ctx)]

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope):
        self._best_bid = _best_price(event.payload, "bids")
        self._best_ask = _best_price(event.payload, "asks")
        return None

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        payload = event.payload
        position = payload.get("position")
        if isinstance(position, int) and not isinstance(position, bool):
            self.position = position

        event_kind = payload.get("event_kind")
        if event_kind == "order_accepted":
            client_order_id = payload.get("client_order_id")
            order_id = payload.get("order_id")
            if isinstance(client_order_id, str) and isinstance(order_id, str) and client_order_id in self._pending:
                self._owned[order_id] = self._pending.pop(client_order_id)
        elif event_kind == "cancel_accepted":
            order_id = payload.get("order_id")
            if isinstance(order_id, str):
                self._owned.pop(order_id, None)
        elif event_kind == "order_rejected":
            client_order_id = payload.get("client_order_id")
            if isinstance(client_order_id, str):
                self._pending.pop(client_order_id, None)
        elif event_kind == "fill":
            order_id = payload.get("order_id")
            quantity = payload.get("quantity")
            owned = self._owned.get(order_id) if isinstance(order_id, str) else None
            if owned is not None and isinstance(quantity, int) and not isinstance(quantity, bool) and quantity > 0:
                side, remaining = owned
                remaining -= quantity
                self.position += side.signed(quantity)
                if remaining > 0:
                    self._owned[order_id] = (side, remaining)
                else:
                    self._owned.pop(order_id, None)
        return None

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        if not _timer_matches(event, self.timer_id):
            return None

        actions: list[Action] = [ctx.cancel(order_id) for order_id in self._owned]
        self._owned.clear()

        side = self._choose_side(ctx)
        aggressive = ctx.random.random() < self.aggressive_probability
        if side is Side.BUY:
            price = self._best_ask if aggressive else self._best_bid
        else:
            price = self._best_bid if aggressive else self._best_ask
        if price is not None:
            quantity = ctx.random.randint(self.min_quantity, self.max_quantity)
            self._sequence += 1
            client_order_id = f"recurring-noise-{self._sequence}"
            self._pending[client_order_id] = (side, quantity)
            actions.append(ctx.submit_limit(client_order_id, side, price, quantity))
        actions.append(self._next_timer(ctx))
        return actions

    def _choose_side(self, ctx: StrategyContext) -> Side:
        if self.position >= self.soft_inventory_limit:
            side = Side.SELL
        elif self.position <= -self.soft_inventory_limit:
            side = Side.BUY
        elif self._last_side is not None and ctx.random.random() < self.side_persistence:
            side = self._last_side
        else:
            side = Side.BUY if ctx.random.getrandbits(1) else Side.SELL
        self._last_side = side
        return side

    def _next_timer(self, ctx: StrategyContext):
        delay = ctx.random.randint(self.min_interval, self.max_interval)
        return ctx.set_timer(self.timer_id, fire_at=ctx.now + delay)


class LatentValueTrader(Strategy):
    """Slow informed trader with a complete or partial private-value signal."""

    def __init__(
        self,
        *,
        initial_value: int,
        update_interval: int,
        order_quantity: int,
        normal_step: int,
        shock_probability: float,
        shock_size: int,
        signal_process: LatentValueProcess | None = None,
        signal_observation_probability: float = 1.0,
        signal_noise_step: int = 0,
        signal_proportional_noise_multiplier: float = 0.0,
        timer_id: str = "latent-value-update",
    ) -> None:
        if isinstance(initial_value, bool) or not isinstance(initial_value, int):
            raise TypeError("initial_value must be an int")
        if update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if order_quantity <= 0:
            raise ValueError("order_quantity must be positive")
        if normal_step < 0:
            raise ValueError("normal_step must be non-negative")
        if not math.isfinite(shock_probability) or not 0 <= shock_probability <= 1:
            raise ValueError("shock_probability must be between zero and one")
        if shock_size < normal_step:
            raise ValueError("shock_size must be at least normal_step")
        if signal_process is not None and not isinstance(signal_process, LatentValueProcess):
            raise TypeError("signal_process must be a LatentValueProcess")
        if not math.isfinite(signal_observation_probability) or not 0 <= signal_observation_probability <= 1:
            raise ValueError("signal_observation_probability must be between zero and one")
        if isinstance(signal_noise_step, bool) or not isinstance(signal_noise_step, int):
            raise TypeError("signal_noise_step must be an int")
        if signal_noise_step < 0:
            raise ValueError("signal_noise_step must be non-negative")
        if not math.isfinite(signal_proportional_noise_multiplier) or signal_proportional_noise_multiplier < 0:
            raise ValueError("signal_proportional_noise_multiplier must be finite and non-negative")
        if not timer_id:
            raise ValueError("timer_id must not be empty")
        self.fair_value = initial_value
        self.latent_value = initial_value
        self.update_interval = update_interval
        self.order_quantity = order_quantity
        self.normal_step = normal_step
        self.shock_probability = shock_probability
        self.shock_size = shock_size
        self.signal_process = signal_process
        self.signal_observation_probability = signal_observation_probability
        self.signal_noise_step = signal_noise_step
        self.signal_proportional_noise_multiplier = signal_proportional_noise_multiplier
        self.timer_id = timer_id
        self._best_bid: int | None = None
        self._best_ask: int | None = None
        self._sequence = 0
        self._update_index = 0
        self.last_latent_innovation = 0
        self.last_private_innovation = 0

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        return [
            ctx.set_timer(
                self.timer_id,
                fire_at=ctx.now + self.update_interval,
            )
        ]

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope):
        self._best_bid = _best_price(event.payload, "bids")
        self._best_ask = _best_price(event.payload, "asks")
        return None

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        if not _timer_matches(event, self.timer_id):
            return None

        self._update_private_value(ctx)

        actions: list[Action] = []
        side: Side | None = None
        if self._best_ask is not None and self.fair_value > self._best_ask:
            side = Side.BUY
        elif self._best_bid is not None and self.fair_value < self._best_bid:
            side = Side.SELL
        if side is not None:
            self._sequence += 1
            actions.append(
                ctx.submit_limit(
                    f"latent-value-{self._sequence}",
                    side,
                    self.fair_value,
                    self.order_quantity,
                )
            )
        actions.append(
            ctx.log(
                "informed_decision",
                fields={
                    "diagnostic_kind": "informed_decision",
                    "participant_id": ctx.participant_id,
                    "decision_index": self._update_index,
                    "latent_innovation": self.last_latent_innovation,
                    "private_innovation": self.last_private_innovation,
                    "latent_value": self.latent_value,
                    "fair_value": self.fair_value,
                    "best_bid": self._best_bid,
                    "best_ask": self._best_ask,
                    "side": None if side is None else side.name.lower(),
                    "planned_quantity": 0 if side is None else self.order_quantity,
                    "execution_model": "single_marketable_limit",
                    "signal_source": "independent_latent_value_process",
                },
            )
        )
        actions.append(
            ctx.set_timer(
                self.timer_id,
                fire_at=ctx.now + self.update_interval,
            )
        )
        return actions

    def _update_private_value(self, ctx: StrategyContext) -> None:
        self._update_index += 1
        if self.signal_process is None:
            if ctx.random.random() < self.shock_probability:
                direction = 1 if ctx.random.getrandbits(1) else -1
                innovation = direction * self.shock_size
            elif self.normal_step:
                innovation = ctx.random.randint(
                    -self.normal_step,
                    self.normal_step,
                )
            else:
                innovation = 0
        else:
            innovation = self.signal_process.innovation(self._update_index)

        self.latent_value += innovation
        self.last_latent_innovation = innovation
        observed = self.signal_observation_probability == 1 or ctx.random.random() < self.signal_observation_probability
        private_innovation = innovation if observed else 0
        if self.signal_proportional_noise_multiplier and innovation:
            proportional_bound = math.ceil(abs(innovation) * self.signal_proportional_noise_multiplier)
            private_innovation += ctx.random.randint(
                -proportional_bound,
                proportional_bound,
            )
        if self.signal_noise_step:
            private_innovation += ctx.random.randint(
                -self.signal_noise_step,
                self.signal_noise_step,
            )
        self.last_private_innovation = private_innovation
        self.fair_value += private_innovation


class ReservationDemandTrader(Strategy):
    """Slow, price-elastic demand around a public-market anchor.

    The strategy has no special exchange path: it learns the current midpoint
    from ordinary level updates, receives fills through its own execution
    stream, and submits one ordinary limit order at a time.  Its trusted demand
    process changes a target inventory, rather than prescribing a price.
    """

    def __init__(
        self,
        *,
        demand_process: _CohortDemandProcess,
        cohort_id: str,
        initial_anchor: float,
        update_interval: int,
        anchor_half_life: int,
        price_elasticity: float,
        base_target_position: int,
        target_loading: float,
        clip_quantity: int,
        position_tolerance: int = 0,
        timer_id: str = "reservation-demand-update",
    ) -> None:
        target_shift = getattr(demand_process, "target_shift", None)
        if not callable(target_shift):
            raise TypeError("demand_process must provide target_shift")
        if not isinstance(cohort_id, str) or not cohort_id:
            raise ValueError("cohort_id must not be empty")
        if (
            isinstance(initial_anchor, bool)
            or not isinstance(initial_anchor, (int, float))
            or not math.isfinite(initial_anchor)
        ):
            raise ValueError("initial_anchor must be finite")
        for name, value in (
            ("update_interval", update_interval),
            ("anchor_half_life", anchor_half_life),
            ("clip_quantity", clip_quantity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(position_tolerance, bool) or not isinstance(position_tolerance, int) or position_tolerance < 0:
            raise ValueError("position_tolerance must be a non-negative integer")
        if isinstance(base_target_position, bool) or not isinstance(base_target_position, int):
            raise TypeError("base_target_position must be an integer")
        for name, value, inclusive in (
            ("price_elasticity", price_elasticity, False),
            ("target_loading", target_loading, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (not inclusive and value == 0)
            ):
                comparator = "non-negative" if inclusive else "positive"
                raise ValueError(f"{name} must be finite and {comparator}")
        if not isinstance(timer_id, str) or not timer_id:
            raise ValueError("timer_id must not be empty")

        self.demand_process = demand_process
        self.cohort_id = cohort_id
        self.anchor = float(initial_anchor)
        self.update_interval = update_interval
        self.anchor_half_life = anchor_half_life
        self.price_elasticity = float(price_elasticity)
        self.base_target_position = base_target_position
        self.target_loading = float(target_loading)
        self.clip_quantity = clip_quantity
        self.position_tolerance = position_tolerance
        self.timer_id = timer_id

        self.position = 0
        self._best_bid: int | None = None
        self._best_ask: int | None = None
        self._latest_midpoint: float | None = None
        self._last_anchor_update_at: int | None = None
        self._order_sequence = 0
        self._pending: dict[str, tuple[Side, int]] = {}
        self._owned: dict[str, tuple[Side, int]] = {}
        self._cancel_pending: set[str] = set()

    def on_start(self, ctx: StrategyContext, event: InputEnvelope):
        self._last_anchor_update_at = ctx.now
        return [ctx.set_timer(self.timer_id, fire_at=ctx.now + self.update_interval)]

    def on_levels(self, ctx: StrategyContext, event: InputEnvelope):
        bid = _best_price(event.payload, "bids")
        ask = _best_price(event.payload, "asks")
        if bid is None or ask is None:
            return None
        self._best_bid = bid
        self._best_ask = ask
        self._latest_midpoint = (bid + ask) / 2
        return None

    def on_execution(self, ctx: StrategyContext, event: InputEnvelope):
        payload = event.payload
        position = payload.get("position")
        has_position = isinstance(position, int) and not isinstance(position, bool)
        if has_position:
            self.position = position

        kind = payload.get("event_kind")
        client_order_id = payload.get("client_order_id")
        if kind == "order_accepted":
            order_id = payload.get("order_id")
            if isinstance(client_order_id, str) and isinstance(order_id, str) and client_order_id in self._pending:
                self._owned[order_id] = self._pending.pop(client_order_id)
            return None
        if kind == "order_rejected":
            if isinstance(client_order_id, str):
                self._pending.pop(client_order_id, None)
            return None

        order_id = payload.get("order_id")
        if kind == "cancel_accepted" and isinstance(order_id, str):
            self._forget_order(order_id)
            return None
        if kind == "cancel_rejected" and isinstance(order_id, str):
            self._cancel_pending.discard(order_id)
            return None
        if kind != "fill" or not isinstance(order_id, str):
            return None

        owned = self._owned.get(order_id)
        quantity = payload.get("quantity")
        if owned is None or isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            return None
        side, remaining = owned
        if not has_position:
            self.position += side.signed(quantity)
        remaining -= quantity
        if remaining <= 0:
            self._forget_order(order_id)
        else:
            self._owned[order_id] = (side, remaining)
        return None

    def on_timer(self, ctx: StrategyContext, event: InputEnvelope):
        if not _timer_matches(event, self.timer_id):
            return None

        actions: list[Action] = []
        for order_id in tuple(self._owned):
            if order_id not in self._cancel_pending:
                actions.append(ctx.cancel(order_id))
                self._cancel_pending.add(order_id)

        self._update_anchor(ctx.now)
        shift = self.demand_process.target_shift(self.cohort_id, ctx.now)
        if isinstance(shift, bool) or not isinstance(shift, (int, float)):
            raise TypeError("demand_process.target_shift must return a number")
        if not math.isfinite(shift):
            raise ValueError("demand_process.target_shift must return a finite value")
        target_position = self.base_target_position + round(self.target_loading * shift)
        desired_position: int | None = None
        indifference_price: float | None = None
        side: Side | None = None
        quantity = 0
        limit_price: int | None = None
        if self._latest_midpoint is not None:
            desired_position = round(target_position + self.price_elasticity * (self.anchor - self._latest_midpoint))
            difference = desired_position - self.position
            if abs(difference) > self.position_tolerance:
                side = Side.BUY if difference > 0 else Side.SELL
                quantity = min(
                    self.clip_quantity,
                    abs(difference) - self.position_tolerance,
                )
                indifference_price = self.anchor + (target_position - self.position) / self.price_elasticity
                limit_price = math.floor(indifference_price) if side is Side.BUY else math.ceil(indifference_price)
                self._order_sequence += 1
                client_order_id = f"reservation-demand-{self._order_sequence}"
                self._pending[client_order_id] = (side, quantity)
                actions.append(ctx.submit_limit(client_order_id, side, limit_price, quantity))

        actions.append(
            ctx.log(
                "reservation_demand_decision",
                fields={
                    "diagnostic_kind": "reservation_demand_decision",
                    "participant_id": ctx.participant_id,
                    "cohort_id": self.cohort_id,
                    "anchor": self.anchor,
                    "midpoint": self._latest_midpoint,
                    "target_shift": shift,
                    "target_position": target_position,
                    "desired_position": desired_position,
                    "position": self.position,
                    "position_tolerance": self.position_tolerance,
                    "side": None if side is None else side.name.lower(),
                    "quantity": quantity,
                    "indifference_price": indifference_price,
                    "limit_price": limit_price,
                },
            )
        )
        actions.append(ctx.set_timer(self.timer_id, fire_at=ctx.now + self.update_interval))
        return actions

    def _update_anchor(self, now: int) -> None:
        if self._latest_midpoint is None:
            self._last_anchor_update_at = now
            return
        previous = self._last_anchor_update_at
        if previous is None:
            self._last_anchor_update_at = now
            return
        elapsed = max(0, now - previous)
        weight = 1 - math.exp(-math.log(2) * elapsed / self.anchor_half_life)
        self.anchor += weight * (self._latest_midpoint - self.anchor)
        self._last_anchor_update_at = now

    def _forget_order(self, order_id: str) -> None:
        self._owned.pop(order_id, None)
        self._cancel_pending.discard(order_id)


__all__ = [
    "AdaptiveMarketMaker",
    "EventDrivenAdaptiveMarketMaker",
    "InventoryAwareMarketMaker",
    "LatentValueTrader",
    "PersistentNoiseTrader",
    "ReservationDemandTrader",
    "RollingNoiseExecutor",
    "ScheduledNoiseExecutor",
]
