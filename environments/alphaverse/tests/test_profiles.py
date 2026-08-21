from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from alphaverse.profiles import (
    InformationGrant,
    MarginConfig,
    ParticipantSpec,
    RiskLimits,
    TechnologyProfile,
)


def test_information_grants_have_stable_wire_values() -> None:
    assert InformationGrant.PUBLIC_MARKET.value == "public_market"
    assert InformationGrant.OWN_EXECUTIONS.value == "own_executions"
    assert InformationGrant.FUTURE_FLOW_SIGNAL.value == "future_flow_signal"
    assert InformationGrant.LATENT_VALUE_SIGNAL.value == "latent_value_signal"


def test_technology_profile_is_immutable_and_accepts_zero_latency() -> None:
    profile = TechnologyProfile(
        market_data_latency=0,
        level_feed_latency=5,
        decision_latency=10,
        order_entry_latency=2,
        mbo_entitled=False,
        storage_limit_bytes=0,
        callback_timeout_ns=500,
        callback_memory_limit_bytes=1_024,
    )
    assert profile.level_feed_latency == 5
    assert profile.storage_limit_bytes == 0
    with pytest.raises(FrozenInstanceError):
        profile.decision_latency = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    [
        "market_data_latency",
        "level_feed_latency",
        "decision_latency",
        "order_entry_latency",
        "storage_limit_bytes",
    ],
)
def test_technology_profile_rejects_negative_counts(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be non-negative"):
        TechnologyProfile(**{field: -1})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["callback_timeout_ns", "callback_memory_limit_bytes"])
def test_technology_profile_requires_positive_callback_resources(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        TechnologyProfile(**{field: 0})  # type: ignore[arg-type]


def test_technology_profile_rejects_bool_as_latency_and_non_bool_entitlement() -> None:
    with pytest.raises(TypeError, match="market_data_latency must be an int"):
        TechnologyProfile(market_data_latency=True)
    with pytest.raises(TypeError, match="mbo_entitled must be a bool"):
        TechnologyProfile(mbo_entitled=1)  # type: ignore[arg-type]


def test_risk_limits_support_optional_financial_thresholds() -> None:
    limits = RiskLimits(
        max_abs_position=20,
        max_order_quantity=5,
        max_live_orders=12,
        max_actions_per_callback=4,
        max_drawdown=2_000,
        min_equity=-500,
    )
    assert limits.max_drawdown == 2_000
    assert limits.min_equity == -500
    with pytest.raises(FrozenInstanceError):
        limits.max_abs_position = 30  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    [
        "max_abs_position",
        "max_order_quantity",
        "max_live_orders",
        "max_actions_per_callback",
    ],
)
def test_risk_limits_require_positive_operational_limits(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        RiskLimits(**{field: 0})  # type: ignore[arg-type]


def test_risk_limits_validate_optional_thresholds() -> None:
    with pytest.raises(ValueError, match="max_drawdown must be non-negative"):
        RiskLimits(max_drawdown=-1)
    with pytest.raises(TypeError, match="min_equity must be an int"):
        RiskLimits(min_equity=1.5)  # type: ignore[arg-type]


def test_margin_configuration_is_opt_in_and_validates_requirements() -> None:
    margin = MarginConfig(100, 80, grace_period=5)
    assert margin.initial_margin_per_contract == 100
    assert ParticipantSpec("m", "margin", 1_000, margin=margin).margin == margin
    with pytest.raises(ValueError, match="must not exceed"):
        MarginConfig(80, 100)


def test_participant_defaults_are_public_and_immutable() -> None:
    spec = ParticipantSpec("p1", "strategy:v1", 100_000)

    assert spec.information_grants == frozenset({InformationGrant.PUBLIC_MARKET, InformationGrant.OWN_EXECUTIONS})
    assert spec.parameters == {}
    assert isinstance(spec.technology, TechnologyProfile)
    assert isinstance(spec.risk, RiskLimits)
    with pytest.raises(FrozenInstanceError):
        spec.seed = 4  # type: ignore[misc]


def test_participant_copies_and_deep_freezes_parameters_and_grants() -> None:
    source_parameters = {
        "levels": [1, 2],
        "quote": {"width": 3.5, "enabled": True},
    }
    source_grants = {
        InformationGrant.PUBLIC_MARKET,
        InformationGrant.OWN_EXECUTIONS,
        InformationGrant.FUTURE_FLOW_SIGNAL,
    }
    spec = ParticipantSpec(
        "informed",
        "mft:v3",
        50_000,
        parameters=source_parameters,
        information_grants=source_grants,  # type: ignore[arg-type]
        seed=42,
    )

    source_parameters["levels"].append(3)  # type: ignore[union-attr]
    source_grants.remove(InformationGrant.FUTURE_FLOW_SIGNAL)
    assert spec.parameters["levels"] == (1, 2)
    assert spec.parameters["quote"] == {"width": 3.5, "enabled": True}
    assert InformationGrant.FUTURE_FLOW_SIGNAL in spec.information_grants
    with pytest.raises(TypeError):
        spec.parameters["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.parameters["quote"]["width"] = 4  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"participant_id": ""}, ValueError, "participant_id must not be empty"),
        (
            {"strategy_version_id": ""},
            ValueError,
            "strategy_version_id must not be empty",
        ),
        (
            {"account_starting_cash": 0},
            ValueError,
            "account_starting_cash must be positive",
        ),
        ({"seed": -1}, ValueError, "seed must be non-negative"),
    ],
)
def test_participant_validates_identity_capital_and_seed(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    values: dict[str, object] = {
        "participant_id": "p1",
        "strategy_version_id": "s:v1",
        "account_starting_cash": 10,
    }
    values.update(kwargs)
    with pytest.raises(error, match=message):
        ParticipantSpec(**values)  # type: ignore[arg-type]


def test_participant_rejects_invalid_parameter_trees() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ParticipantSpec("p1", "s:v1", 10, parameters={"x": float("nan")})
    with pytest.raises(TypeError, match="unsupported type set"):
        ParticipantSpec("p1", "s:v1", 10, parameters={"x": {1, 2}})
    with pytest.raises(TypeError, match="non-string mapping key"):
        ParticipantSpec("p1", "s:v1", 10, parameters={"x": {1: 2}})  # type: ignore[dict-item]


def test_participant_requires_public_and_private_execution_baselines() -> None:
    with pytest.raises(ValueError, match="public market information"):
        ParticipantSpec(
            "p1",
            "s:v1",
            10,
            information_grants=frozenset({InformationGrant.OWN_EXECUTIONS}),
        )
    with pytest.raises(ValueError, match="their own executions"):
        ParticipantSpec(
            "p1",
            "s:v1",
            10,
            information_grants=frozenset({InformationGrant.PUBLIC_MARKET}),
        )
    with pytest.raises(TypeError, match="InformationGrant values"):
        ParticipantSpec(
            "p1",
            "s:v1",
            10,
            information_grants=frozenset({"public_market", "own_executions"}),  # type: ignore[arg-type]
        )
