from __future__ import annotations

from dataclasses import dataclass

SUITS: dict[str, tuple[str, str]] = {
    "H": ("hearts", "red"),
    "D": ("diamonds", "red"),
    "C": ("clubs", "black"),
    "S": ("spades", "black"),
}
RANK_LABELS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
RANK_VALUES = {label: index + 1 for index, label in enumerate(RANK_LABELS)}


@dataclass(frozen=True)
class Card:
    symbol: str
    suit: str
    suit_symbol: str
    color: str
    rank: int
    rank_label: str

    @property
    def is_ace(self) -> bool:
        return self.rank == 1

    @property
    def is_face(self) -> bool:
        return self.rank in {11, 12, 13}

    def __str__(self) -> str:
        return self.symbol

    # String-protocol compatibility so lenient rule strings like `"H" in card`
    # or `card[0]` keep working inside the sandboxed rule evaluator.
    def __contains__(self, value: object) -> bool:
        return str(value) in self.symbol

    def __getitem__(self, index: int | slice) -> str:
        return self.symbol[index]

    def __iter__(self):
        return iter(self.symbol)

    def __len__(self) -> int:
        return len(self.symbol)

    def count(self, value: str) -> int:
        return self.symbol.count(value)

    def index(self, value: str) -> int:
        return self.symbol.index(value)

    def startswith(self, prefix: str) -> bool:
        return self.symbol.startswith(prefix)

    def endswith(self, suffix: str) -> bool:
        return self.symbol.endswith(suffix)


def deck() -> list[str]:
    """One standard 52-card deck, as symbols like "AH" or "10S"."""
    return [f"{rank}{suit}" for suit in SUITS for rank in RANK_LABELS]


def double_deck() -> list[str]:
    """Two standard decks shuffled together (104 cards, so duplicates exist)."""
    return deck() + deck()


def parse_card(symbol: str) -> Card:
    normalized = str(symbol or "").strip().upper()
    if len(normalized) < 2:
        raise ValueError(f"Unknown card: {symbol}")
    suit_symbol = normalized[-1]
    rank_label = normalized[:-1]
    if suit_symbol not in SUITS:
        raise ValueError(f"Unknown card suit: {symbol}")
    if rank_label not in RANK_VALUES:
        raise ValueError(f"Unknown card rank: {symbol}")
    suit, color = SUITS[suit_symbol]
    return Card(
        symbol=f"{rank_label}{suit_symbol}",
        suit=suit,
        suit_symbol=suit_symbol,
        color=color,
        rank=RANK_VALUES[rank_label],
        rank_label=rank_label,
    )
