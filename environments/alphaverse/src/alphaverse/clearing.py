"""Fixed-point clearing and exact midpoint marking for one product."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from alphaverse.models import Fill, Price


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """An immutable view of an account's clearing state."""

    participant_id: str
    starting_cash: int
    cash: int | float
    cash_subunits: int
    fees_paid: int | float
    fees_paid_subunits: int
    cash_subunits_per_tick: int
    position: int

    @property
    def exact_cash(self) -> Fraction:
        return Fraction(self.cash_subunits, self.cash_subunits_per_tick)

    @property
    def exact_fees_paid(self) -> Fraction:
        return Fraction(self.fees_paid_subunits, self.cash_subunits_per_tick)


@dataclass(slots=True)
class _Account:
    starting_cash: int
    cash_subunits: int
    fees_paid_subunits: int = 0
    position: int = 0


def midpoint(best_bid: Price, best_ask: Price) -> Fraction:
    """Return the exact midpoint, including for half-ticks and negative prices."""

    if best_bid > best_ask:
        raise ValueError("best_bid must not exceed best_ask")
    return Fraction(best_bid + best_ask, 2)


class ClearingLedger:
    """Clear fills between registered accounts for a single linear product.

    Cash is stored as integer subunits and positions are integral. Human-facing
    values are derived from that fixed-point state. Marked values use
    :class:`Fraction` so neither fees nor half-tick midpoints are rounded.
    """

    def __init__(
        self,
        *,
        product_id: str = "ALPHA",
        contract_multiplier: int = 1,
        cash_subunits_per_tick: int = 100,
        transaction_fee_per_contract_subunits: int = 5,
    ) -> None:
        if not product_id:
            raise ValueError("product_id must not be empty")
        if contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be positive")
        if cash_subunits_per_tick <= 0:
            raise ValueError("cash_subunits_per_tick must be positive")
        if transaction_fee_per_contract_subunits < 0:
            raise ValueError("transaction_fee_per_contract_subunits must be non-negative")
        self.product_id = product_id
        self.contract_multiplier = contract_multiplier
        self.cash_subunits_per_tick = cash_subunits_per_tick
        self.transaction_fee_per_contract_subunits = transaction_fee_per_contract_subunits
        self._accounts: dict[str, _Account] = {}

    def register_account(self, participant_id: str, *, starting_cash: int) -> None:
        """Register a participant exactly once."""

        if not participant_id:
            raise ValueError("participant_id must not be empty")
        if participant_id in self._accounts:
            raise ValueError(f"account already registered: {participant_id}")
        self._accounts[participant_id] = _Account(
            starting_cash=starting_cash,
            cash_subunits=starting_cash * self.cash_subunits_per_tick,
        )

    def apply_fill(self, fill: Fill) -> None:
        """Apply both sides of a fill atomically to the clearing ledger."""

        if fill.product_id != self.product_id:
            raise ValueError(f"fill product {fill.product_id!r} does not match ledger product {self.product_id!r}")

        buyer = self._get_account(fill.buyer_participant_id)
        seller = self._get_account(fill.seller_participant_id)
        cash_value_subunits = fill.price * fill.quantity * self.contract_multiplier * self.cash_subunits_per_tick
        fee_subunits = self.fee_subunits(fill.quantity)

        buyer.cash_subunits -= cash_value_subunits + fee_subunits
        buyer.fees_paid_subunits += fee_subunits
        buyer.position += fill.quantity
        seller.cash_subunits += cash_value_subunits - fee_subunits
        seller.fees_paid_subunits += fee_subunits
        seller.position -= fill.quantity

    def fee_subunits(self, quantity: int) -> int:
        """Return the exact fee charged to one side for a filled quantity."""

        if quantity <= 0:
            raise ValueError("quantity must be positive")
        return quantity * self.transaction_fee_per_contract_subunits

    def snapshot(self, participant_id: str) -> AccountSnapshot:
        """Return a detached, immutable account snapshot."""

        account = self._get_account(participant_id)
        exact_cash = Fraction(
            account.cash_subunits,
            self.cash_subunits_per_tick,
        )
        exact_fees = Fraction(
            account.fees_paid_subunits,
            self.cash_subunits_per_tick,
        )
        return AccountSnapshot(
            participant_id=participant_id,
            starting_cash=account.starting_cash,
            cash=_number(exact_cash),
            cash_subunits=account.cash_subunits,
            fees_paid=_number(exact_fees),
            fees_paid_subunits=account.fees_paid_subunits,
            cash_subunits_per_tick=self.cash_subunits_per_tick,
            position=account.position,
        )

    @property
    def accounts(self) -> tuple[AccountSnapshot, ...]:
        """Return every account in stable participant-id order."""

        return tuple(self.snapshot(participant_id) for participant_id in sorted(self._accounts))

    def marked_equity(
        self,
        participant_id: str,
        *,
        best_bid: Price,
        best_ask: Price,
    ) -> Fraction:
        """Return cash plus the position marked at the exact midpoint."""

        account = self.snapshot(participant_id)
        mark = midpoint(best_bid, best_ask)
        return account.exact_cash + (account.position * mark * self.contract_multiplier)

    def marked_equity_at_mark(
        self,
        participant_id: str,
        *,
        mark: Fraction,
    ) -> Fraction:
        """Return equity using a previously established valid market mark."""

        account = self.snapshot(participant_id)
        return account.exact_cash + (account.position * mark * self.contract_multiplier)

    def pnl(
        self,
        participant_id: str,
        *,
        best_bid: Price,
        best_ask: Price,
    ) -> Fraction:
        """Return midpoint-marked profit or loss relative to starting cash."""

        account = self._get_account(participant_id)
        return (
            self.marked_equity(
                participant_id,
                best_bid=best_bid,
                best_ask=best_ask,
            )
            - account.starting_cash
        )

    def _get_account(self, participant_id: str) -> _Account:
        try:
            return self._accounts[participant_id]
        except KeyError:
            raise KeyError(f"unknown account: {participant_id}") from None


def _number(value: Fraction) -> int | float:
    if value.denominator == 1:
        return value.numerator
    return float(value)
