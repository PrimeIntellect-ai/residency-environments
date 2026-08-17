from __future__ import annotations

import pytest
from alphaverse.models import BookChangeKind, CancelOrder, NewOrder, Side
from alphaverse.order_book import OrderBook, PriceLevel


def order(
    participant: str,
    client_id: str,
    side: Side,
    price: int,
    quantity: int,
    *,
    product_id: str = "ALPHA",
) -> NewOrder:
    return NewOrder(participant, client_id, side, price, quantity, product_id)


def test_non_marketable_order_rests_and_negative_prices_work() -> None:
    book = OrderBook()

    result = book.submit(order("buyer", "b1", Side.BUY, -10, 7))

    assert result.order_id == "O1"
    assert result.fills == ()
    assert result.remaining_quantity == 7
    assert [(change.kind, change.order_id) for change in result.book_changes] == [(BookChangeKind.ADD, "O1")]
    assert book.best_bid == -10
    assert book.best_ask is None
    assert book.get_order("O1") is not None
    assert book.get_order("O1").remaining_quantity == 7  # type: ignore[union-attr]


def test_price_priority_and_resting_price_execution() -> None:
    book = OrderBook()
    book.submit(order("seller-low", "s1", Side.SELL, 99, 2))
    book.submit(order("seller-high", "s2", Side.SELL, 101, 2))

    result = book.submit(order("buyer", "b1", Side.BUY, 105, 3))

    assert [(fill.price, fill.quantity, fill.maker_order_id) for fill in result.fills] == [
        (99, 2, "O1"),
        (101, 1, "O2"),
    ]
    assert result.remaining_quantity == 0
    assert book.best_ask == 101
    assert book.get_order("O2").remaining_quantity == 1  # type: ignore[union-attr]


def test_fifo_within_price_and_partial_fill_preserves_priority() -> None:
    book = OrderBook()
    first = book.submit(order("seller-1", "s1", Side.SELL, 100, 5))
    second = book.submit(order("seller-2", "s2", Side.SELL, 100, 5))

    partial = book.submit(order("buyer-1", "b1", Side.BUY, 100, 3))
    sweep = book.submit(order("buyer-2", "b2", Side.BUY, 100, 4))

    assert [(fill.maker_order_id, fill.quantity) for fill in partial.fills] == [(first.order_id, 3)]
    assert [(fill.maker_order_id, fill.quantity) for fill in sweep.fills] == [
        (first.order_id, 2),
        (second.order_id, 2),
    ]
    remaining = book.snapshot().asks
    assert [(item.order_id, item.remaining_quantity) for item in remaining] == [("O2", 3)]
    assert remaining[0].priority == 2


def test_marketable_residual_rests_at_limit_price() -> None:
    book = OrderBook()
    book.submit(order("seller", "s1", Side.SELL, 10, 2))

    result = book.submit(order("buyer", "b1", Side.BUY, 12, 5))

    assert [(fill.price, fill.quantity) for fill in result.fills] == [(10, 2)]
    assert result.remaining_quantity == 3
    assert book.best_bid == 12
    assert [change.kind for change in result.book_changes] == [
        BookChangeKind.DELETE,
        BookChangeKind.ADD,
    ]
    assert result.book_changes[-1].order_id == result.order_id
    assert result.book_changes[-1].remaining_quantity == 3


def test_multi_level_sweep_emits_causal_changes() -> None:
    book = OrderBook()
    book.submit(order("s1", "s1", Side.SELL, 100, 2))
    book.submit(order("s2", "s2", Side.SELL, 101, 3))
    book.submit(order("s3", "s3", Side.SELL, 103, 5))

    result = book.submit(order("buyer", "b1", Side.BUY, 102, 8))

    assert [(fill.trade_id, fill.price, fill.quantity) for fill in result.fills] == [
        ("T1", 100, 2),
        ("T2", 101, 3),
    ]
    assert [change.kind for change in result.book_changes] == [
        BookChangeKind.DELETE,
        BookChangeKind.DELETE,
        BookChangeKind.ADD,
    ]
    assert [change.order_id for change in result.book_changes] == ["O1", "O2", "O4"]
    assert result.remaining_quantity == 3
    assert book.best_bid == 102
    assert book.best_ask == 103


def test_top_k_aggregates_and_orders_both_sides() -> None:
    book = OrderBook()
    book.submit(order("b1", "b1", Side.BUY, 99, 2))
    book.submit(order("b2", "b2", Side.BUY, 100, 3))
    book.submit(order("b3", "b3", Side.BUY, 100, 4))
    book.submit(order("s1", "s1", Side.SELL, 104, 5))
    book.submit(order("s2", "s2", Side.SELL, 103, 6))

    levels = book.top_k(1)

    assert levels.bids == (PriceLevel(100, 7, 2),)
    assert levels.asks == (PriceLevel(103, 6, 1),)
    assert book.top_k(0).bids == ()
    with pytest.raises(ValueError, match="k must be non-negative"):
        book.top_k(-1)


def test_cancel_requires_owner_and_removes_only_target_order() -> None:
    book = OrderBook()
    first = book.submit(order("owner", "b1", Side.BUY, 100, 2))
    second = book.submit(order("owner", "b2", Side.BUY, 100, 3))

    rejected = book.cancel(CancelOrder("intruder", first.order_id))
    accepted = book.cancel(CancelOrder("owner", first.order_id))

    assert not rejected.cancelled
    assert rejected.reason == "not_owner"
    assert book.get_order(first.order_id) is None
    assert accepted.cancelled
    assert accepted.reason is None
    assert accepted.book_changes[0].kind is BookChangeKind.DELETE
    assert accepted.book_changes[0].remaining_quantity == 0
    assert [item.order_id for item in book.snapshot().bids] == [second.order_id]


def test_unknown_cancel_is_rejected_without_change() -> None:
    book = OrderBook()

    result = book.cancel(CancelOrder("p1", "O404"))

    assert not result.cancelled
    assert result.reason == "unknown_order"
    assert result.book_changes == ()


def test_live_and_participant_orders_preserve_surviving_arrival_order() -> None:
    book = OrderBook()
    first = book.submit(order("first", "b1", Side.BUY, 98, 1))
    removed = book.submit(order("other", "b2", Side.BUY, 99, 1))
    third = book.submit(order("first", "b3", Side.BUY, 100, 1))
    book.cancel(CancelOrder("other", removed.order_id))

    assert [item.order_id for item in book.live_orders] == [
        first.order_id,
        third.order_id,
    ]
    assert [item.order_id for item in book.orders_for_participant("first")] == [first.order_id, third.order_id]
    assert book.orders_for_participant("missing") == ()


def test_self_match_probe_detects_only_crossing_owned_contra_orders() -> None:
    book = OrderBook()
    book.submit(order("owner", "ask", Side.SELL, 101, 2))
    book.submit(order("other", "better-ask", Side.SELL, 100, 2))

    assert not book.would_self_match(order("owner", "bid", Side.BUY, 100, 1))
    assert book.would_self_match(order("owner", "bid", Side.BUY, 101, 1))
    assert not book.would_self_match(order("outsider", "bid", Side.BUY, 100, 1))
    assert not book.would_self_match(order("owner", "ask-2", Side.SELL, 99, 1))


def test_wrong_product_is_rejected() -> None:
    book = OrderBook("OTHER")

    with pytest.raises(ValueError, match="does not match book"):
        book.submit(order("p1", "b1", Side.BUY, 10, 1))
    with pytest.raises(ValueError, match="does not match book"):
        book.cancel(CancelOrder("p1", "O1"))


def test_fills_report_buyer_and_seller_for_sell_aggressor() -> None:
    book = OrderBook()
    book.submit(order("resting-buyer", "b1", Side.BUY, 100, 2))

    result = book.submit(order("aggressive-seller", "s1", Side.SELL, 99, 2))

    assert result.fills[0].buyer_participant_id == "resting-buyer"
    assert result.fills[0].seller_participant_id == "aggressive-seller"
