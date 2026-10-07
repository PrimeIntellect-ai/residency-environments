"""Evaluator-only longitudinal market and account diagnostics.

The player never receives these samples.  They are reconstructed from canonical
exchange events during finalization so benchmark plots do not add work to the
matching-engine hot path or inflate the agent's context.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import groupby
from typing import Any

from alphaverse.models import Event, EventKind, JsonValue
from alphaverse.player import PlayerSession
from alphaverse.scenario import SECOND

DEFAULT_EVALUATION_SERIES_INTERVAL_NS = 60 * SECOND


def build_evaluation_series(
    session: PlayerSession,
    *,
    interval_ns: int = DEFAULT_EVALUATION_SERIES_INTERVAL_NS,
) -> list[dict[str, JsonValue]]:
    """Reconstruct regular snapshots using all canonical events at or before ``t``."""

    if isinstance(interval_ns, bool) or not isinstance(interval_ns, int):
        raise TypeError("interval_ns must be an int")
    if interval_ns <= 0:
        raise ValueError("interval_ns must be positive")

    exchange = session.episode.exchange
    account = exchange.clearing.snapshot(session.participant_id)
    focal_id = session.participant_id
    cash_subunits = account.starting_cash * account.cash_subunits_per_tick
    fees_subunits = 0
    position = 0
    best_bid: int | None = None
    best_ask: int | None = None
    event_sequence = 0
    market_trade_count = 0
    market_quantity = 0
    focal_fill_count = 0
    focal_quantity = 0
    focal_order_count = 0
    focal_rejection_count = 0
    margin_call_count = 0
    prior_market_quantity = 0
    prior_focal_quantity = 0
    prior_market_trade_count = 0
    prior_focal_fill_count = 0
    deployments = session.deployment_records()
    points: list[dict[str, JsonValue]] = []

    def midpoint() -> Fraction | None:
        if best_bid is not None and best_ask is not None:
            return Fraction(best_bid + best_ask, 2)
        return None

    def emit(market_time_ns: int) -> None:
        nonlocal prior_market_quantity, prior_focal_quantity
        nonlocal prior_market_trade_count, prior_focal_fill_count
        mark = midpoint()
        marked_equity_subunits = Fraction(cash_subunits)
        if mark is not None:
            marked_equity_subunits += (
                position * mark * exchange.product.contract_multiplier * account.cash_subunits_per_tick
            )
        completed_deployments = sum(record.deployed_at <= market_time_ns for record in deployments)
        active_strategy = any(
            record.deployed_at <= market_time_ns and (record.ended_at is None or market_time_ns < record.ended_at)
            for record in deployments
        )
        points.append(
            {
                "market_time_ns": market_time_ns,
                "market_time_seconds": market_time_ns / SECOND,
                "event_sequence": event_sequence,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": _json_number(mark),
                "spread": (None if best_bid is None or best_ask is None else best_ask - best_bid),
                "market_trade_count": market_trade_count,
                "market_trade_count_interval": (market_trade_count - prior_market_trade_count),
                "market_traded_quantity": market_quantity,
                "market_traded_quantity_interval": (market_quantity - prior_market_quantity),
                "focal_fill_count": focal_fill_count,
                "focal_fill_count_interval": focal_fill_count - prior_focal_fill_count,
                "focal_traded_quantity": focal_quantity,
                "focal_traded_quantity_interval": focal_quantity - prior_focal_quantity,
                "focal_order_count": focal_order_count,
                "focal_rejection_count": focal_rejection_count,
                "position": position,
                "cash": _json_number(Fraction(cash_subunits, account.cash_subunits_per_tick)),
                "fees_paid": _json_number(Fraction(fees_subunits, account.cash_subunits_per_tick)),
                "marked_pnl": _json_number(
                    (marked_equity_subunits - account.starting_cash * account.cash_subunits_per_tick)
                    / account.cash_subunits_per_tick
                ),
                "deployment_count": completed_deployments,
                "strategy_active": active_strategy,
                "margin_call_count": margin_call_count,
            }
        )
        prior_market_quantity = market_quantity
        prior_focal_quantity = focal_quantity
        prior_market_trade_count = market_trade_count
        prior_focal_fill_count = focal_fill_count

    def apply(event: Event) -> None:
        nonlocal cash_subunits, fees_subunits, position
        nonlocal best_bid, best_ask, event_sequence
        nonlocal market_trade_count, market_quantity
        nonlocal focal_fill_count, focal_quantity, focal_order_count
        nonlocal focal_rejection_count, margin_call_count
        event_sequence = event.sequence
        if event.kind is EventKind.LEVELS:
            bids = event.data.get("bids")
            asks = event.data.get("asks")
            best_bid = _first_price(bids)
            best_ask = _first_price(asks)
        elif event.kind is EventKind.TRADE:
            quantity = event.data.get("quantity")
            if isinstance(quantity, int) and not isinstance(quantity, bool):
                market_trade_count += 1
                market_quantity += quantity
        elif event.data.get("participant_id") == focal_id:
            if event.kind is EventKind.ACCOUNT:
                raw_cash = event.data.get("cash_subunits")
                raw_fees = event.data.get("fees_paid_subunits")
                raw_position = event.data.get("position")
                if isinstance(raw_cash, int) and not isinstance(raw_cash, bool):
                    cash_subunits = raw_cash
                if isinstance(raw_fees, int) and not isinstance(raw_fees, bool):
                    fees_subunits = raw_fees
                if isinstance(raw_position, int) and not isinstance(raw_position, bool):
                    position = raw_position
            elif event.kind is EventKind.FILL:
                quantity = event.data.get("quantity")
                if isinstance(quantity, int) and not isinstance(quantity, bool):
                    focal_fill_count += 1
                    focal_quantity += quantity
            elif event.kind is EventKind.ORDER_ACCEPTED:
                if not event.data.get("liquidation", False):
                    focal_order_count += 1
            elif event.kind in (EventKind.ORDER_REJECTED, EventKind.CANCEL_REJECTED):
                focal_rejection_count += 1
            elif event.kind is EventKind.RISK and event.data.get("state") == "margin_call":
                margin_call_count += 1
            elif event.kind is EventKind.SESSION:
                raw_cash = event.data.get("terminal_cash_subunits")
                raw_fees = event.data.get("fees_paid_subunits")
                raw_position = event.data.get("position")
                if isinstance(raw_cash, int) and not isinstance(raw_cash, bool):
                    cash_subunits = raw_cash
                if isinstance(raw_fees, int) and not isinstance(raw_fees, bool):
                    fees_subunits = raw_fees
                if isinstance(raw_position, int) and not isinstance(raw_position, bool):
                    position = raw_position

    next_sample = 0
    for event_time, same_time_events in groupby(
        exchange.event_log,
        key=lambda event: event.market_time,
    ):
        while next_sample < event_time and next_sample <= session.now:
            emit(next_sample)
            next_sample += interval_ns
        for event in same_time_events:
            apply(event)
        if next_sample == event_time and next_sample <= session.now:
            emit(next_sample)
            next_sample += interval_ns
    while next_sample <= session.now:
        emit(next_sample)
        next_sample += interval_ns
    if not points or points[-1]["market_time_ns"] != session.now:
        emit(session.now)
    return points


def _first_price(raw_levels: Any) -> int | None:
    if not isinstance(raw_levels, list) or not raw_levels:
        return None
    first = raw_levels[0]
    if not isinstance(first, dict):
        return None
    price = first.get("price")
    return price if isinstance(price, int) and not isinstance(price, bool) else None


def _json_number(value: Fraction | None) -> int | float | None:
    if value is None:
        return None
    if value.denominator == 1:
        return value.numerator
    return float(value)
