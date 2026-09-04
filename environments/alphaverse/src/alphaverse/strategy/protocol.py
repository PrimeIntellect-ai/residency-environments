"""Language-neutral strategy input and action wire records."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias

from alphaverse.models import Side

WireScalar: TypeAlias = str | int | float | bool | None
WireValue: TypeAlias = WireScalar | tuple["WireValue", ...] | Mapping[str, "WireValue"]


class _FrozenWireMapping(Mapping[str, WireValue]):
    """Validated immutable wire mapping that can be safely shared by envelopes."""

    __slots__ = ("_view",)

    def __init__(self, values: dict[str, WireValue]) -> None:
        self._view = MappingProxyType(values)

    def __getitem__(self, key: str) -> WireValue:
        return self._view[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._view)

    def __len__(self) -> int:
        return len(self._view)

    def __repr__(self) -> str:
        return repr(self._view)

    def __eq__(self, other: object) -> bool:
        return self._view == other


class InputKind(str, Enum):
    START = "start"
    MARKET = "market"
    LEVELS = "levels"
    EXECUTION = "execution"
    TIMER = "timer"
    RISK = "risk"
    STOP = "stop"
    SIGNAL = "signal"
    ALERT = "alert"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class InputEnvelope:
    session_id: str
    strategy_instance_id: str
    event_id: str
    kind: InputKind
    exchange_time: int
    available_at: int
    source_event_seq: int
    payload: Mapping[str, WireValue] = field(default_factory=dict)

    @staticmethod
    def freeze_payload(payload: Mapping[str, WireValue]) -> Mapping[str, WireValue]:
        """Validate and freeze a payload once for reuse across recipient envelopes."""

        return _freeze_mapping(payload, "payload")

    @classmethod
    def _from_internal_payload(
        cls,
        *,
        session_id: str,
        strategy_instance_id: str,
        event_id: str,
        kind: InputKind,
        exchange_time: int,
        available_at: int,
        source_event_seq: int,
        payload: Mapping[str, WireValue],
    ) -> InputEnvelope:
        """Construct an internal envelope, reusing a validated payload if possible."""

        frozen_payload = payload if type(payload) is _FrozenWireMapping else _freeze_mapping(payload, "payload")
        envelope = object.__new__(cls)
        object.__setattr__(envelope, "session_id", session_id)
        object.__setattr__(envelope, "strategy_instance_id", strategy_instance_id)
        object.__setattr__(envelope, "event_id", event_id)
        object.__setattr__(envelope, "kind", kind)
        object.__setattr__(envelope, "exchange_time", exchange_time)
        object.__setattr__(envelope, "available_at", available_at)
        object.__setattr__(envelope, "source_event_seq", source_event_seq)
        object.__setattr__(envelope, "payload", frozen_payload)
        return envelope

    def __post_init__(self) -> None:
        for name in ("session_id", "strategy_instance_id", "event_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if not isinstance(self.kind, InputKind):
            raise TypeError("kind must be an InputKind")
        _require_non_negative_int("exchange_time", self.exchange_time)
        _require_non_negative_int("available_at", self.available_at)
        _require_non_negative_int("source_event_seq", self.source_event_seq)
        if self.available_at < self.exchange_time:
            raise ValueError("available_at must not precede exchange_time")
        object.__setattr__(self, "payload", _freeze_mapping(self.payload, "payload"))

    def to_json(self) -> str:
        return _canonical_json(
            {
                "session_id": self.session_id,
                "strategy_instance_id": self.strategy_instance_id,
                "event_id": self.event_id,
                "kind": self.kind.value,
                "exchange_time": self.exchange_time,
                "available_at": self.available_at,
                "source_event_seq": self.source_event_seq,
                "payload": _thaw(self.payload),
            }
        )

    @classmethod
    def from_json(cls, payload: str | bytes) -> InputEnvelope:
        raw = json.loads(payload)
        _require_exact_keys(
            raw,
            {
                "session_id",
                "strategy_instance_id",
                "event_id",
                "kind",
                "exchange_time",
                "available_at",
                "source_event_seq",
                "payload",
            },
        )
        return cls(
            session_id=raw["session_id"],
            strategy_instance_id=raw["strategy_instance_id"],
            event_id=raw["event_id"],
            kind=InputKind(raw["kind"]),
            exchange_time=raw["exchange_time"],
            available_at=raw["available_at"],
            source_event_seq=raw["source_event_seq"],
            payload=raw["payload"],
        )


@dataclass(frozen=True, slots=True)
class SubmitLimitOrder:
    client_order_id: str
    side: Side
    price: int
    quantity: int
    product_id: str = "ALPHA"

    def __post_init__(self) -> None:
        if not self.client_order_id:
            raise ValueError("client_order_id must not be empty")
        if not isinstance(self.side, Side):
            raise TypeError("side must be a Side")
        _require_int("price", self.price)
        _require_positive_int("quantity", self.quantity)
        if not self.product_id:
            raise ValueError("product_id must not be empty")


@dataclass(frozen=True, slots=True)
class CancelOrderAction:
    order_id: str
    product_id: str = "ALPHA"

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id must not be empty")
        if not self.product_id:
            raise ValueError("product_id must not be empty")


@dataclass(frozen=True, slots=True)
class SetTimer:
    timer_id: str
    fire_at: int

    def __post_init__(self) -> None:
        if not self.timer_id:
            raise ValueError("timer_id must not be empty")
        _require_non_negative_int("fire_at", self.fire_at)


@dataclass(frozen=True, slots=True)
class CancelTimer:
    timer_id: str

    def __post_init__(self) -> None:
        if not self.timer_id:
            raise ValueError("timer_id must not be empty")


@dataclass(frozen=True, slots=True)
class EmitLog:
    level: LogLevel
    message: str
    fields: Mapping[str, WireValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.level, LogLevel):
            raise TypeError("level must be a LogLevel")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        object.__setattr__(self, "fields", _freeze_mapping(self.fields, "fields"))


@dataclass(frozen=True, slots=True)
class EmitAlert:
    """A concise strategy-to-player notification with JSON-compatible context."""

    code: str
    message: str
    data: Mapping[str, WireValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, str):
            raise TypeError("code must be a string")
        if not self.code or len(self.code) > 64:
            raise ValueError("code must be between 1 and 64 characters")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if not self.message or len(self.message) > 512:
            raise ValueError("message must be between 1 and 512 characters")
        frozen = _freeze_mapping(self.data, "data")
        if len(frozen) > 16:
            raise ValueError("data may contain at most 16 fields")
        if len(_canonical_json(_thaw(frozen)).encode("utf-8")) > 4_096:
            raise ValueError("data must serialize to at most 4096 bytes")
        object.__setattr__(self, "data", frozen)


@dataclass(frozen=True, slots=True)
class RequestStop:
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")


Action: TypeAlias = SubmitLimitOrder | CancelOrderAction | SetTimer | CancelTimer | EmitLog | EmitAlert | RequestStop


_ACTION_TYPES: dict[str, type] = {
    "submit_limit_order": SubmitLimitOrder,
    "cancel_order": CancelOrderAction,
    "set_timer": SetTimer,
    "cancel_timer": CancelTimer,
    "emit_log": EmitLog,
    "emit_alert": EmitAlert,
    "request_stop": RequestStop,
}
_TYPE_NAMES = {action_type: name for name, action_type in _ACTION_TYPES.items()}


@dataclass(frozen=True, slots=True)
class ActionBatch:
    actions: tuple[Action, ...] = ()

    def __init__(self, actions: Iterable[Action] = ()) -> None:
        frozen = tuple(actions)
        if not all(type(action) in _TYPE_NAMES for action in frozen):
            raise TypeError("actions contains an unsupported action type")
        object.__setattr__(self, "actions", frozen)

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def to_json(self) -> str:
        records: list[dict] = []
        for action in self.actions:
            if isinstance(action, SubmitLimitOrder):
                record = {
                    "type": "submit_limit_order",
                    "client_order_id": action.client_order_id,
                    "side": action.side.name.lower(),
                    "price": action.price,
                    "quantity": action.quantity,
                    "product_id": action.product_id,
                }
            elif isinstance(action, CancelOrderAction):
                record = {
                    "type": "cancel_order",
                    "order_id": action.order_id,
                    "product_id": action.product_id,
                }
            elif isinstance(action, SetTimer):
                record = {
                    "type": "set_timer",
                    "timer_id": action.timer_id,
                    "fire_at": action.fire_at,
                }
            elif isinstance(action, CancelTimer):
                record = {"type": "cancel_timer", "timer_id": action.timer_id}
            elif isinstance(action, EmitLog):
                record = {
                    "type": "emit_log",
                    "level": action.level.value,
                    "message": action.message,
                    "fields": _thaw(action.fields),
                }
            elif isinstance(action, EmitAlert):
                record = {
                    "type": "emit_alert",
                    "code": action.code,
                    "message": action.message,
                    "data": _thaw(action.data),
                }
            else:
                record = {"type": "request_stop", "reason": action.reason}
            records.append(record)
        return _canonical_json({"actions": records})

    @classmethod
    def from_json(cls, payload: str | bytes) -> ActionBatch:
        raw = json.loads(payload)
        _require_exact_keys(raw, {"actions"})
        if not isinstance(raw["actions"], list):
            raise TypeError("actions must be a list")
        actions: list[Action] = []
        for record in raw["actions"]:
            if not isinstance(record, dict) or "type" not in record:
                raise ValueError("each action requires a type")
            action_type = record.pop("type")
            if action_type not in _ACTION_TYPES:
                raise ValueError(f"unknown action type: {action_type}")
            if action_type == "submit_limit_order":
                record["side"] = Side[record["side"].upper()]
            elif action_type == "emit_log":
                record["level"] = LogLevel(record["level"])
            try:
                actions.append(_ACTION_TYPES[action_type](**record))
            except TypeError as exc:
                raise ValueError(f"invalid {action_type} action") from exc
        return cls(actions)


def _freeze_mapping(value: object, name: str) -> Mapping[str, WireValue]:
    if type(value) is _FrozenWireMapping:
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen: dict[str, WireValue] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        frozen[key] = _freeze_value(child, f"{name}.{key}")
    return _FrozenWireMapping(frozen)


def _freeze_value(value: object, name: str) -> WireValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value, name)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item, name) for item in value)
    raise TypeError(f"{name} is not JSON-compatible")


def _thaw(value: WireValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _require_exact_keys(value: object, expected: set[str]) -> None:
    if not isinstance(value, dict):
        raise TypeError("wire payload must be an object")
    if set(value) != expected:
        raise ValueError("wire payload fields do not match the schema")


def _require_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")


def _require_non_negative_int(name: str, value: object) -> None:
    _require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_int(name: str, value: object) -> None:
    _require_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "Action",
    "ActionBatch",
    "CancelOrderAction",
    "CancelTimer",
    "EmitAlert",
    "EmitLog",
    "InputEnvelope",
    "InputKind",
    "LogLevel",
    "RequestStop",
    "SetTimer",
    "SubmitLimitOrder",
]
