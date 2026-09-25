"""Raw public market-data capture formats and filtering."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal, TypeAlias

from alphaverse.player import PlayerFeedEvent

MarketCaptureFeed: TypeAlias = Literal["mbo", "levels"]

_COMMON_PROPERTIES = {
    "cursor": {"type": "integer", "minimum": 1},
    "event_id": {"type": "string", "minLength": 1},
    "kind": {"type": "string"},
    "exchange_time": {"type": "integer", "minimum": 0},
    "available_at": {"type": "integer", "minimum": 0},
    "source_event_seq": {"type": "integer", "minimum": 0},
}
_COMMON_REQUIRED = [
    "cursor",
    "event_id",
    "kind",
    "exchange_time",
    "available_at",
    "source_event_seq",
    "payload",
]


def _payload_schema(
    event_kind: str,
    properties: dict[str, object],
    required: list[str],
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event_kind": {"const": event_kind},
            "match_event_id": {"type": "string", "minLength": 1},
            "product_id": {"type": "string", "minLength": 1},
            **properties,
        },
        "required": [
            "event_kind",
            "match_event_id",
            "product_id",
            *required,
        ],
    }


_PAYLOAD_SCHEMAS = {
    "mbo_change": _payload_schema(
        "mbo_change",
        {
            "action": {"enum": ["add", "reduce", "delete"]},
            "order_id": {"type": "string", "minLength": 1},
            "side": {"enum": ["buy", "sell"]},
            "price": {"type": "integer"},
            "remaining_quantity": {"type": "integer", "minimum": 0},
            "priority": {"type": "integer", "minimum": 0},
            "event_end": {"const": False},
        },
        [
            "action",
            "order_id",
            "side",
            "price",
            "remaining_quantity",
            "priority",
            "event_end",
        ],
    ),
    "trade": _payload_schema(
        "trade",
        {
            "trade_id": {"type": "string", "minLength": 1},
            "price": {"type": "integer"},
            "quantity": {"type": "integer", "minimum": 1},
            "aggressor_side": {"enum": ["buy", "sell"]},
            "maker_order_id": {"type": "string", "minLength": 1},
            "taker_order_id": {"type": "string", "minLength": 1},
        },
        [
            "trade_id",
            "price",
            "quantity",
            "aggressor_side",
            "maker_order_id",
            "taker_order_id",
        ],
    ),
    "levels": _payload_schema(
        "levels",
        {
            "depth": {"type": "integer", "minimum": 0},
            "through_event_seq": {"type": "integer", "minimum": 0},
            "bids": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "price": {"type": "integer"},
                        "quantity": {"type": "integer", "minimum": 1},
                        "order_count": {"type": "integer", "minimum": 1},
                    },
                    "required": ["price", "quantity", "order_count"],
                },
            },
            "asks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "price": {"type": "integer"},
                        "quantity": {"type": "integer", "minimum": 1},
                        "order_count": {"type": "integer", "minimum": 1},
                    },
                    "required": ["price", "quantity", "order_count"],
                },
            },
            "event_end": {"const": True},
        },
        [
            "depth",
            "through_event_seq",
            "bids",
            "asks",
            "event_end",
        ],
    ),
}

_FEED_EVENT_KINDS: dict[MarketCaptureFeed, tuple[str, ...]] = {
    "mbo": ("mbo_change", "trade"),
    "levels": ("levels", "trade"),
}


def require_market_capture_feed(feed: str) -> MarketCaptureFeed:
    """Validate and narrow a public capture feed name."""

    if feed not in _FEED_EVENT_KINDS:
        raise ValueError("feed must be 'mbo' or 'levels'")
    return feed


def market_capture_spec(feed: str) -> dict[str, object]:
    """Return the authoritative NDJSON record specification for one feed."""

    selected = require_market_capture_feed(feed)
    event_kinds = _FEED_EVENT_KINDS[selected]
    variants = []
    for event_kind in event_kinds:
        variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    **_COMMON_PROPERTIES,
                    "kind": {"const": "levels" if event_kind == "levels" else "market"},
                    "payload": _PAYLOAD_SCHEMAS[event_kind],
                },
                "required": _COMMON_REQUIRED,
            }
        )
    return deepcopy(
        {
            "feed": selected,
            "format": "ndjson",
            "media_type": "application/x-ndjson",
            "encoding": "utf-8",
            "event_kinds": list(event_kinds),
            "cursor": {
                "scope": "all delivered player events",
                "semantics": "exclusive after_cursor",
                "gaps_expected": True,
                "next_cursor_header": "X-Alphaverse-Next-After-Cursor",
            },
            "record_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": f"Alphaverse {selected.upper()} capture record",
                "oneOf": variants,
            },
        }
    )


def select_market_capture_events(
    events: tuple[PlayerFeedEvent, ...],
    feed: str,
) -> tuple[PlayerFeedEvent, ...]:
    """Select raw public packets for a capture without transforming them."""

    selected = require_market_capture_feed(feed)
    event_kinds = _FEED_EVENT_KINDS[selected]
    return tuple(
        item
        for item in events
        if item.envelope.payload.get("event_kind") in event_kinds and item.envelope.kind.value in {"market", "levels"}
    )


__all__ = [
    "MarketCaptureFeed",
    "market_capture_spec",
    "require_market_capture_feed",
    "select_market_capture_events",
]
