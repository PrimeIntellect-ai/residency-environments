from __future__ import annotations

from fractions import Fraction

from alphaverse.exchange import Exchange, MarginState, SessionState
from alphaverse.models import CancelOrder, EventKind, NewOrder, Side
from alphaverse.profiles import MarginConfig


def _exchange_with_accounts(*participants: str) -> Exchange:
    exchange = Exchange()
    for participant in participants:
        exchange.register_account(participant, starting_cash=1_000)
    return exchange


def test_integrates_matching_clearing_and_feed_event_boundary() -> None:
    exchange = _exchange_with_accounts("maker", "taker")
    exchange.submit_order(NewOrder("maker", "ask", Side.SELL, 101, 5))
    result = exchange.submit_order(NewOrder("taker", "buy", Side.BUY, 105, 3))

    assert result.accepted
    assert exchange.clearing.snapshot("maker").position == -3
    assert exchange.clearing.snapshot("taker").position == 3
    assert exchange.clearing.snapshot("maker").cash == 1_302.85
    assert exchange.clearing.snapshot("taker").cash == 696.85

    event_kinds = [event.kind for event in result.events]
    assert event_kinds == [
        EventKind.ORDER_ACCEPTED,
        EventKind.TRADE,
        EventKind.FILL,
        EventKind.FILL,
        EventKind.MBO_CHANGE,
        EventKind.ACCOUNT,
        EventKind.ACCOUNT,
        EventKind.LEVELS,
    ]
    private_fills = [event for event in result.events if event.kind is EventKind.FILL]
    assert [event.data["fee"] for event in private_fills] == [0.15, 0.15]
    assert [event.data["fee_subunits"] for event in private_fills] == [15, 15]
    account_events = [event for event in result.events if event.kind is EventKind.ACCOUNT]
    assert [event.data["fees_paid_subunits"] for event in account_events] == [
        15,
        15,
    ]
    levels = result.events[-1]
    assert levels.data["event_end"] is True
    assert levels.data["through_event_seq"] == result.events[-2].sequence
    assert levels.data["asks"] == [{"price": 101, "quantity": 2, "order_count": 1}]


def test_midpoint_account_metrics_are_exact() -> None:
    exchange = _exchange_with_accounts("bidder", "asker")
    exchange.submit_order(NewOrder("bidder", "bid", Side.BUY, 99, 2))
    exchange.submit_order(NewOrder("asker", "ask", Side.SELL, 100, 2))
    metrics = exchange.account_metrics("bidder")
    assert metrics.mark == Fraction(199, 2)
    assert metrics.marked_equity == Fraction(1_000)
    assert metrics.pnl == 0


def test_identical_commands_produce_byte_identical_event_logs() -> None:
    def run() -> bytes:
        exchange = _exchange_with_accounts("a", "b")
        exchange.submit_order(NewOrder("a", "a1", Side.BUY, -2, 4), market_time=10)
        exchange.submit_order(NewOrder("b", "b1", Side.SELL, -3, 2), market_time=12)
        return exchange.event_log.to_jsonl()

    assert run() == run()


def test_new_order_that_would_cross_own_resting_order_is_rejected() -> None:
    exchange = _exchange_with_accounts("owner", "other")
    resting = exchange.submit_order(NewOrder("owner", "own-ask", Side.SELL, 101, 5))

    rejected = exchange.submit_order(NewOrder("owner", "crossing-bid", Side.BUY, 101, 3))
    external = exchange.submit_order(NewOrder("other", "external-bid", Side.BUY, 101, 3))

    assert resting.order_id is not None
    assert not rejected.accepted
    assert rejected.reason == "self_match_prevention"
    assert [event.kind for event in rejected.events] == [EventKind.ORDER_REJECTED]
    assert rejected.events[0].data["reason"] == "self_match_prevention"
    assert exchange.book.get_order(resting.order_id) is not None
    assert external.accepted
    assert exchange.clearing.snapshot("owner").position == -3
    assert exchange.clearing.snapshot("other").position == 3
    assert exchange.clearing.snapshot("owner").fees_paid == 0.15
    assert exchange.clearing.snapshot("other").fees_paid == 0.15


def test_termination_cancels_orders_and_realizes_book_slippage() -> None:
    exchange = _exchange_with_accounts("trader", "seller", "bidder1", "bidder2")

    # Trader acquires three at 100.
    exchange.submit_order(NewOrder("seller", "offer", Side.SELL, 100, 3))
    exchange.submit_order(NewOrder("trader", "entry", Side.BUY, 100, 3))

    # Trader also has an unrelated resting order that termination must cancel.
    own_resting = exchange.submit_order(NewOrder("trader", "resting", Side.BUY, 90, 1))
    assert own_resting.order_id is not None

    # Liquidation walks two bid levels: 2 @ 99 and 1 @ 98.
    exchange.submit_order(NewOrder("bidder1", "bid1", Side.BUY, 99, 2))
    exchange.submit_order(NewOrder("bidder2", "bid2", Side.BUY, 98, 2))
    result = exchange.request_termination("trader")

    assert result.state is SessionState.TERMINATED
    assert result.remaining_position == 0
    account = exchange.clearing.snapshot("trader")
    assert account.position == 0
    assert account.cash == 995.7
    assert account.fees_paid == 0.3
    assert exchange.book.get_order(own_resting.order_id) is None
    assert [event.data["price"] for event in result.events if event.kind is EventKind.TRADE] == [99, 98]


def test_incomplete_liquidation_resumes_when_liquidity_arrives() -> None:
    exchange = _exchange_with_accounts("trader", "seller", "late_bidder")
    exchange.submit_order(NewOrder("seller", "offer", Side.SELL, 100, 2))
    exchange.submit_order(NewOrder("trader", "entry", Side.BUY, 100, 2))

    result = exchange.request_termination("trader")
    assert result.state is SessionState.LIQUIDATING
    assert result.remaining_position == 2

    exchange.submit_order(NewOrder("late_bidder", "bid", Side.BUY, 97, 2))
    assert exchange.state("trader") is SessionState.TERMINATED
    assert exchange.clearing.snapshot("trader").position == 0
    assert exchange.clearing.snapshot("trader").cash == 993.8


def test_margin_pretrade_rejection_includes_working_exposure_diagnostics() -> None:
    exchange = Exchange()
    exchange.register_account("focal", starting_cash=250, margin=MarginConfig(100, 80))
    exchange.register_account("bid", starting_cash=1_000)
    exchange.register_account("seller", starting_cash=1_000)
    exchange.submit_order(NewOrder("bid", "bid", Side.BUY, 99, 5))
    exchange.submit_order(NewOrder("seller", "offer", Side.SELL, 101, 5))

    assert exchange.submit_order(NewOrder("focal", "b1", Side.BUY, 99, 1)).accepted
    rejected = exchange.submit_order(NewOrder("focal", "b2", Side.BUY, 99, 2))

    assert not rejected.accepted
    event = rejected.events[-1]
    assert event.kind is EventKind.ORDER_REJECTED
    assert event.data["reason"] == "insufficient_initial_margin"
    assert event.data["same_side_working_quantity"] == 1
    assert event.data["projected_position"] == 3
    assert event.data["projected_initial_requirement"] == 300


def test_margin_mark_excludes_the_focals_own_resting_quote() -> None:
    exchange = Exchange()
    exchange.register_account("focal", starting_cash=250, margin=MarginConfig(100, 80))
    exchange.register_account("bid", starting_cash=1_000)
    exchange.register_account("ask", starting_cash=1_000)
    exchange.submit_order(NewOrder("bid", "bid", Side.BUY, 99, 2))
    exchange.submit_order(NewOrder("ask", "ask", Side.SELL, 101, 2))
    exchange.submit_order(NewOrder("focal", "own-bid", Side.BUY, 100, 1))

    margin = exchange.margin_metrics("focal")
    assert margin is not None
    assert margin.mark == Fraction(100)
    assert margin.mark_source == "external_two_sided_bbo"


def test_margin_call_uses_external_persistent_mark_and_partially_liquidates() -> None:
    exchange = Exchange()
    exchange.register_account("focal", starting_cash=260, margin=MarginConfig(100, 90, grace_period=10))
    exchange.register_account("high-bid", starting_cash=1_000)
    exchange.register_account("seller", starting_cash=1_000)
    exchange.register_account("low-bid", starting_cash=1_000)
    exchange.register_account("low-ask", starting_cash=1_000)
    high_bid = exchange.submit_order(NewOrder("high-bid", "bid", Side.BUY, 99, 5))
    exchange.submit_order(NewOrder("seller", "offer", Side.SELL, 100, 2))
    exchange.submit_order(NewOrder("focal", "entry", Side.BUY, 100, 2))
    exchange.submit_order(NewOrder("focal", "own-offer", Side.SELL, 200, 1))
    assert high_bid.order_id is not None
    exchange.cancel_order(CancelOrder("high-bid", high_bid.order_id))
    exchange.submit_order(NewOrder("low-bid", "bid", Side.BUY, 50, 3))
    trigger = exchange.submit_order(NewOrder("low-ask", "ask", Side.SELL, 60, 3))

    margin = exchange.margin_metrics("focal")
    assert margin is not None
    assert margin.state is MarginState.MARGIN_CALL
    assert margin.mark == Fraction(55)
    assert margin.mark_source == "external_two_sided_bbo"
    assert margin.liquidation_deadline == 10
    assert exchange.state("focal") is SessionState.ACTIVE
    assert not exchange.book.orders_for_participant("focal")
    assert any(event.kind is EventKind.RISK for event in trigger.events)

    liquidation = exchange.process_margin(market_time=10)
    assert exchange.clearing.snapshot("focal").position == 1
    assert exchange.margin_metrics("focal").state is MarginState.NORMAL  # type: ignore[union-attr]
    assert any(event.kind is EventKind.ORDER_ACCEPTED and event.data["liquidation"] for event in liquidation)


def test_margin_call_reduce_only_orders_cannot_reverse_the_position() -> None:
    exchange = Exchange()
    exchange.register_account("focal", starting_cash=260, margin=MarginConfig(100, 90, grace_period=10))
    for participant_id in ("high-bid", "seller", "low-bid", "low-ask"):
        exchange.register_account(participant_id, starting_cash=1_000)
    high_bid = exchange.submit_order(NewOrder("high-bid", "bid", Side.BUY, 99, 5))
    exchange.submit_order(NewOrder("seller", "offer", Side.SELL, 100, 2))
    exchange.submit_order(NewOrder("focal", "entry", Side.BUY, 100, 2))
    assert high_bid.order_id is not None
    exchange.cancel_order(CancelOrder("high-bid", high_bid.order_id))
    exchange.submit_order(NewOrder("low-bid", "bid", Side.BUY, 50, 3))
    exchange.submit_order(NewOrder("low-ask", "ask", Side.SELL, 60, 3))

    first = exchange.submit_order(NewOrder("focal", "reduce-1", Side.SELL, 70, 1))
    second = exchange.submit_order(NewOrder("focal", "reduce-2", Side.SELL, 70, 2))

    assert first.accepted
    assert not second.accepted
    assert second.reason == "margin_reduce_only"
