"""Canonical dependency-free domain types shared by the exchange components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Mapping, TypeAlias

MarketTime: TypeAlias = int
Price: TypeAlias = int
Quantity: TypeAlias = int
Sequence: TypeAlias = int


class Side(IntEnum):
    """Order side with a useful signed-quantity representation."""

    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> Side:
        return Side(-self.value)

    def signed(self, quantity: Quantity) -> Quantity:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        return self.value * quantity


@dataclass(frozen=True, slots=True)
class Product:
    product_id: str = "ALPHA"
    tick_value: int = 1
    contract_multiplier: int = 1
    cash_subunits_per_tick: int = 100
    transaction_fee_per_contract_subunits: int = 5

    def __post_init__(self) -> None:
        if not self.product_id:
            raise ValueError("product_id must not be empty")
        if self.tick_value <= 0:
            raise ValueError("tick_value must be positive")
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be positive")
        if self.cash_subunits_per_tick <= 0:
            raise ValueError("cash_subunits_per_tick must be positive")
        if self.transaction_fee_per_contract_subunits < 0:
            raise ValueError("transaction_fee_per_contract_subunits must be non-negative")


@dataclass(frozen=True, slots=True)
class NewOrder:
    participant_id: str
    client_order_id: str
    side: Side
    price: Price
    quantity: Quantity
    product_id: str = "ALPHA"

    def __post_init__(self) -> None:
        if not self.participant_id:
            raise ValueError("participant_id must not be empty")
        if not self.client_order_id:
            raise ValueError("client_order_id must not be empty")
        if not isinstance(self.side, Side):
            raise TypeError("side must be a Side")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if not self.product_id:
            raise ValueError("product_id must not be empty")


@dataclass(frozen=True, slots=True)
class CancelOrder:
    participant_id: str
    order_id: str
    product_id: str = "ALPHA"

    def __post_init__(self) -> None:
        if not self.participant_id:
            raise ValueError("participant_id must not be empty")
        if not self.order_id:
            raise ValueError("order_id must not be empty")


@dataclass(frozen=True, slots=True)
class Fill:
    trade_id: str
    product_id: str
    price: Price
    quantity: Quantity
    aggressor_side: Side
    maker_order_id: str
    taker_order_id: str
    maker_participant_id: str
    taker_participant_id: str

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

    @property
    def buyer_participant_id(self) -> str:
        if self.aggressor_side is Side.BUY:
            return self.taker_participant_id
        return self.maker_participant_id

    @property
    def seller_participant_id(self) -> str:
        if self.aggressor_side is Side.SELL:
            return self.taker_participant_id
        return self.maker_participant_id


class BookChangeKind(str, Enum):
    ADD = "add"
    REDUCE = "reduce"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class BookChange:
    kind: BookChangeKind
    order_id: str
    participant_id: str
    side: Side
    price: Price
    remaining_quantity: Quantity
    priority: Sequence

    def __post_init__(self) -> None:
        if self.remaining_quantity < 0:
            raise ValueError("remaining_quantity must be non-negative")


@dataclass(frozen=True, slots=True)
class MatchResult:
    order_id: str
    fills: tuple[Fill, ...]
    book_changes: tuple[BookChange, ...]
    remaining_quantity: Quantity

    def __post_init__(self) -> None:
        if self.remaining_quantity < 0:
            raise ValueError("remaining_quantity must be non-negative")


class EventKind(str, Enum):
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    CANCEL_ACCEPTED = "cancel_accepted"
    CANCEL_REJECTED = "cancel_rejected"
    TRADE = "trade"
    FILL = "fill"
    MBO_CHANGE = "mbo_change"
    LEVELS = "levels"
    ACCOUNT = "account"
    RISK = "risk"
    SESSION = "session"


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class Event:
    sequence: Sequence
    market_time: MarketTime
    match_event_id: str
    kind: EventKind
    product_id: str
    data: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.market_time < 0:
            raise ValueError("market_time must be non-negative")
        if not isinstance(self.data, MappingProxyType):
            object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
