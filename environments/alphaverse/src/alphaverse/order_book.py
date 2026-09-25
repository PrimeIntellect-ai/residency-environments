"""Deterministic price-time FIFO limit order book for one product.

The book deliberately owns only matching state.  Arrival scheduling, risk checks,
event sequencing, and dissemination belong to higher exchange layers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from alphaverse.models import (
    BookChange,
    BookChangeKind,
    CancelOrder,
    Fill,
    MatchResult,
    NewOrder,
    Price,
    Quantity,
    Sequence,
    Side,
)


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    """Immutable view of a live resting order."""

    order_id: str
    client_order_id: str
    participant_id: str
    product_id: str
    side: Side
    price: Price
    remaining_quantity: Quantity
    priority: Sequence


@dataclass(frozen=True, slots=True)
class PriceLevel:
    """Aggregated quantity and order count at one price."""

    price: Price
    total_quantity: Quantity
    order_count: int


@dataclass(frozen=True, slots=True)
class BookLevels:
    """Top-of-book aggregates, best price first on each side."""

    bids: tuple[PriceLevel, ...]
    asks: tuple[PriceLevel, ...]

    def __iter__(self):
        """Allow convenient ``bids, asks = book.top_k()`` unpacking."""

        yield self.bids
        yield self.asks


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Immutable full-depth market-by-order snapshot."""

    bids: tuple[OrderSnapshot, ...]
    asks: tuple[OrderSnapshot, ...]


@dataclass(frozen=True, slots=True)
class CancelResult:
    """Outcome of a cancellation request.

    Rejections are ordinary outcomes rather than exceptions because the exchange
    API will need to turn them into private cancel-rejection messages.
    """

    order_id: str
    cancelled: bool
    book_changes: tuple[BookChange, ...] = ()
    reason: str | None = None


@dataclass(slots=True)
class _RestingOrder:
    order_id: str
    client_order_id: str
    participant_id: str
    product_id: str
    side: Side
    price: Price
    remaining_quantity: Quantity
    priority: Sequence

    def snapshot(self) -> OrderSnapshot:
        return OrderSnapshot(
            order_id=self.order_id,
            client_order_id=self.client_order_id,
            participant_id=self.participant_id,
            product_id=self.product_id,
            side=self.side,
            price=self.price,
            remaining_quantity=self.remaining_quantity,
            priority=self.priority,
        )


class OrderBook:
    """A one-product, GTC limit order book using strict price-time priority."""

    def __init__(self, product_id: str = "ALPHA") -> None:
        if not product_id:
            raise ValueError("product_id must not be empty")
        self.product_id = product_id
        self._bids: dict[Price, deque[_RestingOrder]] = {}
        self._asks: dict[Price, deque[_RestingOrder]] = {}
        self._orders: dict[str, _RestingOrder] = {}
        self._participant_orders: dict[str, dict[str, None]] = {}
        self._next_order_number = 1
        self._next_trade_number = 1

    @property
    def best_bid(self) -> Price | None:
        """Best live bid price, or ``None`` when the bid book is empty."""

        return max(self._bids, default=None)

    @property
    def best_ask(self) -> Price | None:
        """Best live ask price, or ``None`` when the ask book is empty."""

        return min(self._asks, default=None)

    @property
    def live_orders(self) -> tuple[OrderSnapshot, ...]:
        """All live orders in original arrival order."""

        # Dict insertion order is the book's monotonic arrival order. Deletions
        # preserve the relative ordering of every surviving order.
        return tuple(order.snapshot() for order in self._orders.values())

    def orders_for_participant(self, participant_id: str) -> tuple[OrderSnapshot, ...]:
        """Return one participant's live orders in arrival order."""

        return tuple(self._orders[order_id].snapshot() for order_id in self._participant_orders.get(participant_id, ()))

    def get_order(self, order_id: str) -> OrderSnapshot | None:
        """Return an immutable view of a live order, if present."""

        order = self._orders.get(order_id)
        return None if order is None else order.snapshot()

    # A descriptive alias for callers that prefer to make the copy explicit.
    order_snapshot = get_order

    def would_self_match(self, command: NewOrder) -> bool:
        """Whether a new limit would cross one of its owner's contra orders."""

        self._check_product(command.product_id)
        opposite_side = command.side.opposite
        return any(
            (order := self._orders[order_id]).side is opposite_side and self._crosses(command, order.price)
            for order_id in self._participant_orders.get(command.participant_id, ())
        )

    def snapshot(self) -> BookSnapshot:
        """Return full depth in market priority order."""

        return BookSnapshot(
            bids=self._side_snapshot(Side.BUY),
            asks=self._side_snapshot(Side.SELL),
        )

    def top_k(self, k: int = 10) -> BookLevels:
        """Return at most ``k`` aggregated price levels on each side."""

        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            return BookLevels((), ())
        return BookLevels(
            bids=self._aggregate_levels(Side.BUY, k),
            asks=self._aggregate_levels(Side.SELL, k),
        )

    def submit(self, command: NewOrder) -> MatchResult:
        """Accept and match a GTC limit order.

        A marketable order consumes the opposite book at each resting order's
        price.  Any residual quantity then rests at its limit price.
        """

        self._check_product(command.product_id)
        incoming = _RestingOrder(
            order_id=f"O{self._next_order_number}",
            client_order_id=command.client_order_id,
            participant_id=command.participant_id,
            product_id=command.product_id,
            side=command.side,
            price=command.price,
            remaining_quantity=command.quantity,
            priority=self._next_order_number,
        )
        self._next_order_number += 1

        fills: list[Fill] = []
        changes: list[BookChange] = []
        opposite_book = self._asks if command.side is Side.BUY else self._bids

        while incoming.remaining_quantity > 0:
            opposite_price = self.best_ask if command.side is Side.BUY else self.best_bid
            if opposite_price is None or not self._crosses(incoming, opposite_price):
                break

            level = opposite_book[opposite_price]
            maker = level[0]
            fill_quantity = min(incoming.remaining_quantity, maker.remaining_quantity)
            incoming.remaining_quantity -= fill_quantity
            maker.remaining_quantity -= fill_quantity

            fills.append(
                Fill(
                    trade_id=f"T{self._next_trade_number}",
                    product_id=self.product_id,
                    price=maker.price,
                    quantity=fill_quantity,
                    aggressor_side=incoming.side,
                    maker_order_id=maker.order_id,
                    taker_order_id=incoming.order_id,
                    maker_participant_id=maker.participant_id,
                    taker_participant_id=incoming.participant_id,
                )
            )
            self._next_trade_number += 1

            if maker.remaining_quantity == 0:
                level.popleft()
                del self._orders[maker.order_id]
                self._untrack_order(maker)
                changes.append(self._book_change(BookChangeKind.DELETE, maker))
                if not level:
                    del opposite_book[opposite_price]
            else:
                changes.append(self._book_change(BookChangeKind.REDUCE, maker))

        if incoming.remaining_quantity > 0:
            own_book = self._bids if incoming.side is Side.BUY else self._asks
            own_book.setdefault(incoming.price, deque()).append(incoming)
            self._orders[incoming.order_id] = incoming
            self._track_order(incoming)
            changes.append(self._book_change(BookChangeKind.ADD, incoming))

        return MatchResult(
            order_id=incoming.order_id,
            fills=tuple(fills),
            book_changes=tuple(changes),
            remaining_quantity=incoming.remaining_quantity,
        )

    def cancel(self, command: CancelOrder) -> CancelResult:
        """Cancel an owned live order without disturbing other FIFO priority."""

        self._check_product(command.product_id)
        order = self._orders.get(command.order_id)
        if order is None:
            return CancelResult(
                order_id=command.order_id,
                cancelled=False,
                reason="unknown_order",
            )
        if order.participant_id != command.participant_id:
            return CancelResult(
                order_id=command.order_id,
                cancelled=False,
                reason="not_owner",
            )

        side_book = self._bids if order.side is Side.BUY else self._asks
        level = side_book[order.price]
        level.remove(order)
        if not level:
            del side_book[order.price]
        del self._orders[order.order_id]
        self._untrack_order(order)

        order.remaining_quantity = 0
        change = self._book_change(BookChangeKind.DELETE, order)
        return CancelResult(
            order_id=order.order_id,
            cancelled=True,
            book_changes=(change,),
        )

    def _check_product(self, product_id: str) -> None:
        if product_id != self.product_id:
            raise ValueError(f"order product {product_id!r} does not match book {self.product_id!r}")

    @staticmethod
    def _crosses(incoming: _RestingOrder | NewOrder, opposite_price: Price) -> bool:
        if incoming.side is Side.BUY:
            return incoming.price >= opposite_price
        return incoming.price <= opposite_price

    @staticmethod
    def _book_change(kind: BookChangeKind, order: _RestingOrder) -> BookChange:
        return BookChange(
            kind=kind,
            order_id=order.order_id,
            participant_id=order.participant_id,
            side=order.side,
            price=order.price,
            remaining_quantity=order.remaining_quantity,
            priority=order.priority,
        )

    def _track_order(self, order: _RestingOrder) -> None:
        self._participant_orders.setdefault(order.participant_id, {})[order.order_id] = None

    def _untrack_order(self, order: _RestingOrder) -> None:
        orders = self._participant_orders[order.participant_id]
        del orders[order.order_id]
        if not orders:
            del self._participant_orders[order.participant_id]

    def _side_snapshot(self, side: Side) -> tuple[OrderSnapshot, ...]:
        side_book = self._bids if side is Side.BUY else self._asks
        prices = sorted(side_book, reverse=side is Side.BUY)
        return tuple(order.snapshot() for price in prices for order in side_book[price])

    def _aggregate_levels(self, side: Side, k: int) -> tuple[PriceLevel, ...]:
        side_book = self._bids if side is Side.BUY else self._asks
        prices = sorted(side_book, reverse=side is Side.BUY)[:k]
        return tuple(
            PriceLevel(
                price=price,
                total_quantity=sum(order.remaining_quantity for order in side_book[price]),
                order_count=len(side_book[price]),
            )
            for price in prices
        )


# Both names are useful vocabulary; keep them identical rather than introducing
# an unnecessary wrapper type.
LimitOrderBook = OrderBook


__all__ = [
    "BookLevels",
    "BookSnapshot",
    "CancelResult",
    "LimitOrderBook",
    "OrderBook",
    "OrderSnapshot",
    "PriceLevel",
]
