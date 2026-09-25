"""Dependency-free types available to trading programs."""

from __future__ import annotations

from enum import IntEnum


class Side(IntEnum):
    """Order side with a useful signed-quantity representation."""

    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> Side:
        return Side(-self.value)

    def signed(self, quantity: int) -> int:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        return self.value * quantity
