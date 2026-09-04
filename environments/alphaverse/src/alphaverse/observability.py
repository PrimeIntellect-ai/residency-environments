"""Read-only development projections over canonical episode state.

This module deliberately exposes information that must never be available to a
focal trading agent.  It is intended for trusted development tooling: an admin
API can serialize the returned projection directly without maintaining a
second, potentially divergent model of the exchange.
"""

from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction
from typing import TypeAlias

from alphaverse.episode import Episode
from alphaverse.models import Event, EventKind, JsonValue
from alphaverse.player import PlayerSession

_Source: TypeAlias = Episode | PlayerSession
_SECOND_NS = 1_000_000_000


def build_observability_snapshot(
    source: _Source,
    *,
    depth: int = 10,
    recent_trade_limit: int = 50,
    recent_event_limit: int = 100,
    price_history_limit: int = 120,
    price_history_bucket_ns: int = _SECOND_NS,
) -> dict[str, JsonValue]:
    """Return a detached, JSON-ready view of one episode's privileged state.

    The projection reads immutable snapshots from the order book and clearing
    ledger, then derives activity counts from the canonical append-only event
    log.  It never advances the clock or calls a strategy.

    Midpoint marking is intentionally unavailable when either book side is
    empty.  ``book.mark_status`` distinguishes a bid-only, ask-only, or empty
    book from an ordinary two-sided midpoint, and participant marked values are
    ``None`` in those cases rather than silently using a stale or one-sided
    price.
    """

    _require_non_negative_int("depth", depth)
    _require_non_negative_int("recent_trade_limit", recent_trade_limit)
    _require_non_negative_int("recent_event_limit", recent_event_limit)
    _require_non_negative_int("price_history_limit", price_history_limit)
    _require_positive_int("price_history_bucket_ns", price_history_bucket_ns)

    if isinstance(source, PlayerSession):
        episode = source.episode
        focal_participant_id: str | None = source.participant_id
    elif isinstance(source, Episode):
        episode = source
        focal_participant_id = None
    else:
        raise TypeError("source must be an Episode or PlayerSession")

    exchange = episode.exchange
    live_orders = exchange.book.live_orders
    levels = exchange.book.top_k(depth)
    best_bid = exchange.book.best_bid
    best_ask = exchange.book.best_ask
    mark_status, mark = _midpoint_status(best_bid, best_ask)

    specs = {spec.participant_id: spec for spec in episode.participant_specs}
    participant_ids = sorted(specs)

    order_counts: dict[str, int] = defaultdict(int)
    fill_counts: dict[str, int] = defaultdict(int)
    rejection_counts: dict[str, int] = defaultdict(int)
    gross_filled_quantities: dict[str, int] = defaultdict(int)
    recent_trades: deque[Event] = deque(maxlen=recent_trade_limit or None)
    recent_events: deque[Event] = deque(maxlen=recent_event_limit or None)
    accepted_order_count = 0
    rejected_order_count = 0
    rejected_cancel_count = 0
    trade_count = 0
    traded_quantity = 0

    history_specs = (
        {
            "2m": (price_history_limit, price_history_bucket_ns),
            "10m": (120, 5 * _SECOND_NS),
            "30m": (120, 15 * _SECOND_NS),
            "1h": (120, 30 * _SECOND_NS),
        }
        if price_history_limit
        else {}
    )
    price_points: dict[str, deque[tuple[int, dict[str, JsonValue]]]] = {
        name: deque(maxlen=limit) for name, (limit, _) in history_specs.items()
    }

    # EventLog iteration decodes one disk-backed row at a time.  Keep every
    # projection below bounded so a terminal snapshot does not reconstruct a
    # multi-hour episode's entire canonical history in RAM.
    for event in exchange.event_log:
        if recent_event_limit:
            recent_events.append(event)
        participant_id = event.data.get("participant_id")
        if event.kind is EventKind.ORDER_ACCEPTED:
            accepted_order_count += 1
            if isinstance(participant_id, str):
                order_counts[participant_id] += 1
        elif event.kind is EventKind.FILL:
            if isinstance(participant_id, str):
                fill_counts[participant_id] += 1
                quantity = event.data.get("quantity")
                if isinstance(quantity, int) and not isinstance(quantity, bool):
                    gross_filled_quantities[participant_id] += quantity
        elif event.kind is EventKind.ORDER_REJECTED:
            rejected_order_count += 1
            if isinstance(participant_id, str):
                rejection_counts[participant_id] += 1
        elif event.kind is EventKind.CANCEL_REJECTED:
            rejected_cancel_count += 1
            if isinstance(participant_id, str):
                rejection_counts[participant_id] += 1
        elif event.kind is EventKind.TRADE:
            trade_count += 1
            quantity = event.data.get("quantity")
            if isinstance(quantity, int) and not isinstance(quantity, bool):
                traded_quantity += quantity
            if recent_trade_limit:
                recent_trades.append(event)
        elif event.kind is EventKind.LEVELS and history_specs:
            best_bid_at_event = _level_price(event.data.get("bids"))
            best_ask_at_event = _level_price(event.data.get("asks"))
            if best_bid_at_event is not None and best_ask_at_event is not None:
                for name, (_, bucket_ns) in history_specs.items():
                    bucket = event.market_time // bucket_ns
                    point: dict[str, JsonValue] = {
                        "market_time": event.market_time,
                        "source_event_sequence": event.sequence,
                        "best_bid": best_bid_at_event,
                        "best_ask": best_ask_at_event,
                        "mid_price": _json_number(Fraction(best_bid_at_event + best_ask_at_event, 2)),
                        "spread": best_ask_at_event - best_bid_at_event,
                    }
                    history = price_points[name]
                    if history and history[-1][0] == bucket:
                        history[-1] = (bucket, point)
                    else:
                        history.append((bucket, point))

    orders_by_participant: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
    for order in live_orders:
        orders_by_participant[order.participant_id].append(
            {
                "order_id": order.order_id,
                "client_order_id": order.client_order_id,
                "side": order.side.name.lower(),
                "price": order.price,
                "remaining_quantity": order.remaining_quantity,
                "priority": order.priority,
            }
        )

    participants: list[dict[str, JsonValue]] = []
    total_cash = Fraction(0)
    total_fees_paid = Fraction(0)
    net_position = 0
    gross_abs_position = 0
    total_starting_cash = 0
    state_counts: dict[str, int] = defaultdict(int)
    total_marked_equity: Fraction | None = Fraction(0) if mark is not None else None

    for participant_id in participant_ids:
        spec = specs[participant_id]
        account = exchange.clearing.snapshot(participant_id)
        state = exchange.state(participant_id).value
        state_counts[state] += 1
        total_starting_cash += account.starting_cash
        total_cash += account.exact_cash
        total_fees_paid += account.exact_fees_paid
        net_position += account.position
        gross_abs_position += abs(account.position)

        if mark is None:
            marked_equity = None
            pnl = None
        else:
            marked_equity = account.exact_cash + (account.position * mark * exchange.product.contract_multiplier)
            pnl = marked_equity - account.starting_cash
            assert total_marked_equity is not None
            total_marked_equity += marked_equity

        owned_orders = orders_by_participant[participant_id]
        participants.append(
            {
                "participant_id": participant_id,
                "strategy_version_id": spec.strategy_version_id,
                "strategy_family": strategy_family(participant_id),
                "session_state": state,
                "starting_cash": account.starting_cash,
                "cash": account.cash,
                "cash_subunits": account.cash_subunits,
                "fees_paid": account.fees_paid,
                "fees_paid_subunits": account.fees_paid_subunits,
                "position": account.position,
                "realized_cash_change": _json_number(account.exact_cash - account.starting_cash),
                "marked_equity": _json_number(marked_equity),
                "pnl": _json_number(pnl),
                "live_order_count": len(owned_orders),
                "live_orders": owned_orders,
                "order_count": order_counts[participant_id],
                "fill_count": fill_counts[participant_id],
                "rejection_count": rejection_counts[participant_id],
                "gross_filled_quantity": gross_filled_quantities[participant_id],
                "technology": {
                    "market_data_latency_ns": spec.technology.market_data_latency,
                    "level_feed_latency_ns": spec.technology.level_feed_latency,
                    "decision_latency_ns": spec.technology.decision_latency,
                    "order_entry_latency_ns": spec.technology.order_entry_latency,
                },
            }
        )

    total_pnl = None if total_marked_equity is None else total_marked_equity - total_starting_cash
    price_histories = {name: [point for _, point in history] for name, history in price_points.items()}

    return {
        "session_id": episode.session_id,
        "product_id": exchange.product.product_id,
        "product": {
            "product_id": exchange.product.product_id,
            "tick_value": exchange.product.tick_value,
            "contract_multiplier": exchange.product.contract_multiplier,
            "cash_subunits_per_tick": exchange.product.cash_subunits_per_tick,
            "transaction_fee_per_contract_subunits": (exchange.product.transaction_fee_per_contract_subunits),
            "transaction_fee_per_contract": _json_number(
                Fraction(
                    exchange.product.transaction_fee_per_contract_subunits,
                    exchange.product.cash_subunits_per_tick,
                )
            ),
        },
        "focal_participant_id": focal_participant_id,
        "market_time": episode.now,
        "event_sequence": exchange.event_log.last_sequence,
        "book": {
            "depth": depth,
            "bids": [_level_data(level) for level in levels.bids],
            "asks": [_level_data(level) for level in levels.asks],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": _json_number(mark),
            "spread": (None if best_bid is None or best_ask is None else best_ask - best_bid),
            "mark_status": mark_status,
        },
        "recent_trades": [_trade_data(event) for event in recent_trades] if recent_trade_limit else [],
        "price_history": price_histories.get("2m", []),
        "price_history_windows": price_histories,
        "recent_events": [_event_data(event) for event in recent_events] if recent_event_limit else [],
        "participants": participants,
        "totals": {
            "participant_count": len(participant_ids),
            "active_participant_count": state_counts["active"],
            "liquidating_participant_count": state_counts["liquidating"],
            "terminated_participant_count": state_counts["terminated"],
            "event_count": len(exchange.event_log),
            "live_order_count": len(live_orders),
            "live_order_quantity": sum(order.remaining_quantity for order in live_orders),
            "accepted_order_count": accepted_order_count,
            "rejected_order_count": rejected_order_count,
            "rejected_cancel_count": rejected_cancel_count,
            "trade_count": trade_count,
            "traded_quantity": traded_quantity,
            "fill_event_count": sum(fill_counts.values()),
            "total_starting_cash": total_starting_cash,
            "total_cash": _json_number(total_cash),
            "total_fees_paid": _json_number(total_fees_paid),
            "total_fees_paid_subunits": sum(
                exchange.clearing.snapshot(participant_id).fees_paid_subunits for participant_id in participant_ids
            ),
            "net_position": net_position,
            "gross_abs_position": gross_abs_position,
            "total_marked_equity": _json_number(total_marked_equity),
            "total_pnl": _json_number(total_pnl),
        },
    }


def _midpoint_status(
    best_bid: int | None,
    best_ask: int | None,
) -> tuple[str, Fraction | None]:
    if best_bid is not None and best_ask is not None:
        return "midpoint", Fraction(best_bid + best_ask, 2)
    if best_bid is not None:
        return "bid_only", None
    if best_ask is not None:
        return "ask_only", None
    return "empty", None


def _json_number(value: Fraction | None) -> int | float | None:
    if value is None:
        return None
    if value.denominator == 1:
        return value.numerator
    return float(value)


def _level_data(level: object) -> dict[str, JsonValue]:
    return {
        "price": getattr(level, "price"),
        "quantity": getattr(level, "total_quantity"),
        "order_count": getattr(level, "order_count"),
    }


def _trade_data(event: Event) -> dict[str, JsonValue]:
    return {
        "sequence": event.sequence,
        "market_time": event.market_time,
        "match_event_id": event.match_event_id,
        "trade_id": event.data.get("trade_id"),
        "price": event.data.get("price"),
        "quantity": event.data.get("quantity"),
        "aggressor_side": event.data.get("aggressor_side"),
        "maker_order_id": event.data.get("maker_order_id"),
        "taker_order_id": event.data.get("taker_order_id"),
    }


def _level_price(levels: JsonValue) -> int | None:
    if not isinstance(levels, list) or not levels:
        return None
    first = levels[0]
    if not isinstance(first, dict):
        return None
    price = first.get("price")
    if isinstance(price, bool) or not isinstance(price, int):
        return None
    return price


def _event_data(event: Event) -> dict[str, JsonValue]:
    return {
        "sequence": event.sequence,
        "market_time": event.market_time,
        "match_event_id": event.match_event_id,
        "kind": event.kind.value,
        "product_id": event.product_id,
        "participant_id": event.data.get("participant_id"),
        "data": dict(event.data),
    }


def strategy_family(participant_id: str) -> str:
    """Return the trusted research/admin cohort for a participant identifier."""

    if participant_id == "player":
        return "focal_agent"
    if participant_id == "opening-liquidity":
        return "opening_liquidity"
    if participant_id.startswith("adaptive-maker-"):
        return "adaptive_market_maker"
    if participant_id.startswith("maker-"):
        return "market_maker"
    if participant_id.startswith("noise-"):
        return "noise_executor"
    if participant_id.startswith("recurring-noise-"):
        return "noise_trader"
    if participant_id.startswith("reservation-demand-"):
        return "reservation_demand"
    if participant_id.startswith("informed-"):
        return "informed_trader"
    if participant_id.startswith("latent-value-"):
        return "informed_trader"
    return "strategy"


def _require_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_int(name: str, value: object) -> None:
    _require_non_negative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


__all__ = ["build_observability_snapshot", "strategy_family"]
