"""Immutable configuration for simulated and focal market participants."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias

from alphaverse.models import MarketTime

ParameterScalar: TypeAlias = str | int | float | bool | None
ParameterValue: TypeAlias = ParameterScalar | tuple["ParameterValue", ...] | Mapping[str, "ParameterValue"]


class InformationGrant(str, Enum):
    """A named information capability available to a strategy instance."""

    PUBLIC_MARKET = "public_market"
    OWN_EXECUTIONS = "own_executions"
    FUTURE_FLOW_SIGNAL = "future_flow_signal"
    LATENT_VALUE_SIGNAL = "latent_value_signal"


@dataclass(frozen=True, slots=True)
class TechnologyProfile:
    """Delivery latency, entitlement, and runner resource configuration."""

    market_data_latency: MarketTime = 0
    level_feed_latency: MarketTime = 0
    decision_latency: MarketTime = 0
    order_entry_latency: MarketTime = 0
    mbo_entitled: bool = True
    storage_limit_bytes: int = 1_000_000
    callback_timeout_ns: int = 1_000_000_000
    callback_memory_limit_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "market_data_latency",
            "level_feed_latency",
            "decision_latency",
            "order_entry_latency",
        ):
            _require_non_negative_int(name, getattr(self, name))
        if not isinstance(self.mbo_entitled, bool):
            raise TypeError("mbo_entitled must be a bool")
        _require_non_negative_int("storage_limit_bytes", self.storage_limit_bytes)
        _require_positive_int("callback_timeout_ns", self.callback_timeout_ns)
        _require_positive_int("callback_memory_limit_bytes", self.callback_memory_limit_bytes)


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Exchange-enforced limits that participant code cannot override."""

    max_abs_position: int = 10_000
    max_order_quantity: int = 1_000
    max_live_orders: int = 1_000
    max_actions_per_callback: int = 100
    max_drawdown: int | None = None
    min_equity: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_abs_position",
            "max_order_quantity",
            "max_live_orders",
            "max_actions_per_callback",
        ):
            _require_positive_int(name, getattr(self, name))
        if self.max_drawdown is not None:
            _require_non_negative_int("max_drawdown", self.max_drawdown)
        if self.min_equity is not None:
            _require_int("min_equity", self.min_equity)


@dataclass(frozen=True, slots=True)
class MarginConfig:
    """Fixed per-contract margin requirements for one participant account.

    ``None`` on :class:`ParticipantSpec` disables margin entirely, preserving
    the unconstrained behavior of background participants.
    """

    initial_margin_per_contract: int
    maintenance_margin_per_contract: int
    grace_period: MarketTime = 0

    def __post_init__(self) -> None:
        _require_positive_int("initial_margin_per_contract", self.initial_margin_per_contract)
        _require_positive_int("maintenance_margin_per_contract", self.maintenance_margin_per_contract)
        if self.maintenance_margin_per_contract > self.initial_margin_per_contract:
            raise ValueError("maintenance_margin_per_contract must not exceed initial_margin_per_contract")
        _require_non_negative_int("grace_period", self.grace_period)


def _default_information_grants() -> frozenset[InformationGrant]:
    return frozenset((InformationGrant.PUBLIC_MARKET, InformationGrant.OWN_EXECUTIONS))


@dataclass(frozen=True, slots=True)
class ParticipantSpec:
    """Complete immutable configuration for one strategy-backed participant."""

    participant_id: str
    strategy_version_id: str
    account_starting_cash: int
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)
    information_grants: frozenset[InformationGrant] = field(default_factory=_default_information_grants)
    technology: TechnologyProfile = field(default_factory=TechnologyProfile)
    risk: RiskLimits = field(default_factory=RiskLimits)
    margin: MarginConfig | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.participant_id, str):
            raise TypeError("participant_id must be a string")
        if not self.participant_id:
            raise ValueError("participant_id must not be empty")
        if not isinstance(self.strategy_version_id, str):
            raise TypeError("strategy_version_id must be a string")
        if not self.strategy_version_id:
            raise ValueError("strategy_version_id must not be empty")
        _require_positive_int("account_starting_cash", self.account_starting_cash)
        _require_non_negative_int("seed", self.seed)
        if not isinstance(self.technology, TechnologyProfile):
            raise TypeError("technology must be a TechnologyProfile")
        if not isinstance(self.risk, RiskLimits):
            raise TypeError("risk must be RiskLimits")
        if self.margin is not None and not isinstance(self.margin, MarginConfig):
            raise TypeError("margin must be MarginConfig or None")

        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        frozen_parameters: dict[str, ParameterValue] = {}
        for key, value in self.parameters.items():
            if not isinstance(key, str):
                raise TypeError("parameter names must be strings")
            if not key:
                raise ValueError("parameter names must not be empty")
            frozen_parameters[key] = _freeze_parameter(value, path=key)
        object.__setattr__(self, "parameters", MappingProxyType(frozen_parameters))

        try:
            frozen_grants = frozenset(self.information_grants)
        except TypeError:
            raise TypeError("information_grants must be an iterable") from None
        if not all(isinstance(grant, InformationGrant) for grant in frozen_grants):
            raise TypeError("information_grants must contain InformationGrant values")
        if InformationGrant.PUBLIC_MARKET not in frozen_grants:
            raise ValueError("participants must be granted public market information")
        if InformationGrant.OWN_EXECUTIONS not in frozen_grants:
            raise ValueError("participants must be granted their own executions")
        object.__setattr__(self, "information_grants", frozen_grants)


def _freeze_parameter(value: object, *, path: str) -> ParameterValue:
    """Copy a JSON-like parameter tree into immutable containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"parameter {path!r} must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, ParameterValue] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"parameter {path!r} has a non-string mapping key")
            if not key:
                raise ValueError(f"parameter {path!r} has an empty mapping key")
            frozen[key] = _freeze_parameter(child, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_parameter(child, path=f"{path}[{index}]") for index, child in enumerate(value))
    raise TypeError(f"parameter {path!r} has unsupported type {type(value).__name__}")


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
