from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest
from alphaverse.clearing import ClearingLedger, midpoint
from alphaverse.models import Fill, Side


def make_fill(
    *,
    trade_id: str = "T1",
    price: int = 100,
    quantity: int = 2,
    aggressor_side: Side = Side.BUY,
    maker: str = "seller",
    taker: str = "buyer",
) -> Fill:
    return Fill(
        trade_id=trade_id,
        product_id="ALPHA",
        price=price,
        quantity=quantity,
        aggressor_side=aggressor_side,
        maker_order_id=f"{trade_id}-maker",
        taker_order_id=f"{trade_id}-taker",
        maker_participant_id=maker,
        taker_participant_id=taker,
    )


def registered_ledger(*, multiplier: int = 1) -> ClearingLedger:
    ledger = ClearingLedger(contract_multiplier=multiplier)
    ledger.register_account("buyer", starting_cash=10_000)
    ledger.register_account("seller", starting_cash=10_000)
    return ledger


def test_fill_uses_contract_multiplier_and_charges_both_sides() -> None:
    ledger = registered_ledger(multiplier=10)

    ledger.apply_fill(make_fill(price=101, quantity=3))

    buyer = ledger.snapshot("buyer")
    seller = ledger.snapshot("seller")
    assert buyer.cash == 10_000 - 3_030 - 0.15
    assert buyer.fees_paid == 0.15
    assert buyer.fees_paid_subunits == 15
    assert buyer.position == 3
    assert seller.cash == 10_000 + 3_030 - 0.15
    assert seller.fees_paid == 0.15
    assert seller.position == -3
    assert buyer.cash + seller.cash == 20_000 - 0.3
    assert buyer.position + seller.position == 0


def test_multiple_fills_clear_both_aggressor_directions() -> None:
    ledger = registered_ledger()
    ledger.apply_fill(make_fill(trade_id="T1", price=100, quantity=4))
    ledger.apply_fill(
        make_fill(
            trade_id="T2",
            price=110,
            quantity=1,
            aggressor_side=Side.SELL,
            maker="buyer",
            taker="seller",
        )
    )

    buyer = ledger.snapshot("buyer")
    seller = ledger.snapshot("seller")
    assert buyer.cash == 9_489.75
    assert buyer.fees_paid == 0.25
    assert buyer.position == 5
    assert seller.cash == 10_509.75
    assert seller.fees_paid == 0.25
    assert seller.position == -5
    assert buyer.cash + seller.cash == 19_999.5
    assert buyer.position + seller.position == 0


def test_midpoint_marking_and_pnl_are_exact() -> None:
    ledger = registered_ledger(multiplier=2)
    ledger.apply_fill(make_fill(price=100, quantity=3))

    assert midpoint(100, 101) == Fraction(201, 2)
    assert ledger.marked_equity("buyer", best_bid=100, best_ask=101) == Fraction(200_057, 20)
    assert ledger.pnl("buyer", best_bid=100, best_ask=101) == Fraction(57, 20)
    assert ledger.pnl("seller", best_bid=100, best_ask=101) == Fraction(-63, 20)


def test_negative_fill_prices_and_negative_midpoint_are_supported() -> None:
    ledger = registered_ledger()
    ledger.apply_fill(make_fill(price=-10, quantity=2))

    assert ledger.snapshot("buyer").cash == 10_019.9
    assert ledger.snapshot("seller").cash == 9_979.9
    assert midpoint(-9, -8) == Fraction(-17, 2)
    assert ledger.pnl("buyer", best_bid=-9, best_ask=-8) == Fraction(29, 10)
    assert ledger.pnl("seller", best_bid=-9, best_ask=-8) == Fraction(-31, 10)


def test_snapshot_is_immutable_and_detached() -> None:
    ledger = registered_ledger()
    before = ledger.snapshot("buyer")

    with pytest.raises(FrozenInstanceError):
        before.cash = 0  # type: ignore[misc]

    ledger.apply_fill(make_fill())
    assert before.cash == 10_000
    assert ledger.snapshot("buyer").cash == 9_799.9


def test_duplicate_registration_and_unknown_accounts_are_rejected_atomically() -> None:
    ledger = registered_ledger()
    with pytest.raises(ValueError, match="already registered"):
        ledger.register_account("buyer", starting_cash=5)
    with pytest.raises(KeyError, match="unknown account"):
        ledger.snapshot("missing")

    fill = make_fill(maker="missing")
    with pytest.raises(KeyError, match="unknown account: missing"):
        ledger.apply_fill(fill)
    assert ledger.snapshot("buyer").cash == 10_000
    assert ledger.snapshot("buyer").position == 0


def test_crossed_market_is_not_a_valid_midpoint() -> None:
    with pytest.raises(ValueError, match="best_bid must not exceed best_ask"):
        midpoint(101, 100)


def test_fee_configuration_is_exact_and_validated() -> None:
    ledger = ClearingLedger(
        cash_subunits_per_tick=1_000,
        transaction_fee_per_contract_subunits=7,
    )
    ledger.register_account("buyer", starting_cash=1_000)
    ledger.register_account("seller", starting_cash=1_000)
    ledger.apply_fill(make_fill(quantity=3))

    buyer = ledger.snapshot("buyer")
    assert buyer.cash_subunits == 699_979
    assert buyer.fees_paid == 0.021
    assert buyer.exact_fees_paid == Fraction(21, 1_000)

    with pytest.raises(ValueError, match="cash_subunits_per_tick"):
        ClearingLedger(cash_subunits_per_tick=0)
    with pytest.raises(ValueError, match="transaction_fee"):
        ClearingLedger(transaction_fee_per_contract_subunits=-1)
