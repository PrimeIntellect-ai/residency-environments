from __future__ import annotations

import pytest
from alphaverse.models import Fill, NewOrder, Product, Side


def test_side_signed_quantity_and_opposite() -> None:
    assert Side.BUY.signed(3) == 3
    assert Side.SELL.signed(3) == -3
    assert Side.BUY.opposite is Side.SELL


def test_order_accepts_negative_price() -> None:
    order = NewOrder("p1", "client-1", Side.BUY, price=-20, quantity=4)
    assert order.price == -20


def test_order_rejects_non_positive_quantity() -> None:
    with pytest.raises(ValueError, match="quantity must be positive"):
        NewOrder("p1", "client-1", Side.BUY, price=10, quantity=0)


def test_fill_derives_buyer_and_seller() -> None:
    fill = Fill(
        trade_id="T1",
        product_id="ALPHA",
        price=100,
        quantity=2,
        aggressor_side=Side.SELL,
        maker_order_id="O1",
        taker_order_id="O2",
        maker_participant_id="buyer",
        taker_participant_id="seller",
    )
    assert fill.buyer_participant_id == "buyer"
    assert fill.seller_participant_id == "seller"


def test_product_validates_fixed_point_fee_terms() -> None:
    product = Product()
    assert product.cash_subunits_per_tick == 100
    assert product.transaction_fee_per_contract_subunits == 5

    with pytest.raises(ValueError, match="cash_subunits_per_tick"):
        Product(cash_subunits_per_tick=0)
    with pytest.raises(ValueError, match="transaction_fee"):
        Product(transaction_fee_per_contract_subunits=-1)
