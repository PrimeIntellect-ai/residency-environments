from __future__ import annotations

import pytest
from alphaverse.prop_trader import (
    PROP_KNOB_DEFAULTS,
    competitive_prop_source,
    prop_baseline_source,
    prop_role_readme,
    prop_system_prompt,
)


@pytest.mark.parametrize("profile", ("passive", "competitive"))
def test_prop_seed_source_compiles_and_constructs(profile: str) -> None:
    source = prop_baseline_source(profile)
    namespace: dict[str, object] = {}

    exec(compile(source, f"<{profile}-prop-seed>", "exec"), namespace)
    strategy_type = namespace["StrategyImpl"]
    strategy = strategy_type()  # type: ignore[operator]

    assert strategy is not None


def test_arbitrary_framing_explicitly_deanchors_the_starter() -> None:
    prompt = prop_system_prompt("arbitrary")
    role = prop_role_readme("passive", "arbitrary")

    for text in (prompt, role):
        normalized = " ".join(text.split())
        assert "arbitrary scaffolding" in normalized
        assert "not a recommendation" in normalized
        assert "Incremental changes are not preferred" in normalized

    assert "arbitrary scaffolding" not in prop_system_prompt("incumbent")


def test_neutral_framing_removes_privilege_without_preferring_replacement() -> None:
    prompt = prop_system_prompt("neutral")
    role = prop_role_readme("competitive", "neutral")

    for text in (prompt, role):
        normalized = " ".join(text.split())
        assert "no privileged status" in normalized
        assert "not guaranteed to be optimal" in normalized
        assert "equally valid choices" in normalized
        assert "Incremental changes are not preferred" not in normalized


def test_knob_scope_discloses_bounded_control_without_source_deployment() -> None:
    prompt = prop_system_prompt("neutral", "knobs")
    role = prop_role_readme("competitive", "neutral", "knobs")

    for text in (prompt, role):
        normalized = " ".join(text.split())
        assert "deploy_strategy" in normalized
        assert "JSON" in normalized
        assert "Raw Python strategy-source deployment is unavailable" in normalized
    assert "modify strategy.py" not in prompt


def test_unknown_prop_ablation_profiles_are_rejected() -> None:
    with pytest.raises(ValueError, match="seed profile"):
        prop_baseline_source("unknown")
    with pytest.raises(ValueError, match="framing profile"):
        prop_system_prompt("unknown")
    with pytest.raises(ValueError, match="control scope"):
        prop_system_prompt("neutral", "unknown")


def test_competitive_knob_defaults_reproduce_the_seed_source() -> None:
    assert competitive_prop_source(**PROP_KNOB_DEFAULTS) == prop_baseline_source("competitive")


def test_competitive_knobs_render_and_construct_a_bounded_variant() -> None:
    source = competitive_prop_source(
        quote_quantity=6,
        correction_quantity=14,
        quote_offset_ticks=2,
        inventory_soft_limit=10,
        max_abs_position=40,
        inventory_retreat_ticks=3,
        refresh_interval_ns=750_000_000,
    )
    namespace: dict[str, object] = {}
    exec(compile(source, "<knob-prop>", "exec"), namespace)
    strategy = namespace["StrategyImpl"]()  # type: ignore[operator]

    assert strategy.quote_quantity == 6
    assert strategy.correction_quantity == 14
    assert strategy.quote_offset_ticks == 2
    assert strategy.inventory_soft_limit == 10
    assert strategy.max_abs_position == 40
    assert strategy.inventory_retreat_ticks == 3
    assert strategy.refresh_interval_ns == 750_000_000


@pytest.mark.parametrize(
    "overrides",
    (
        {"quote_quantity": 0},
        {"quote_offset_ticks": -1},
        {"inventory_soft_limit": 16, "max_abs_position": 16},
        {"refresh_interval_ns": 10_000},
    ),
)
def test_competitive_knobs_reject_out_of_scope_values(overrides) -> None:
    with pytest.raises((TypeError, ValueError)):
        competitive_prop_source(**overrides)
