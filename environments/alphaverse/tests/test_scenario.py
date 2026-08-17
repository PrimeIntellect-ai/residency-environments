from __future__ import annotations

from dataclasses import replace

import pytest
from alphaverse.models import EventKind
from alphaverse.profiles import InformationGrant
from alphaverse.scenario import (
    ENDOGENOUS_MIXED_OVERRIDES,
    ENDOGENOUS_MIXED_SCENARIO_VERSION,
    MVP_SCENARIO_VERSION,
    LatentDemandProfile,
    ScenarioConfig,
    create_populated_scenario,
    scenario_config_for_version,
)


def config(*, seed: int = 17) -> ScenarioConfig:
    return ScenarioConfig(
        seed=seed,
        session_id="scenario-test",
        reference_price=1_000,
        opening_half_spread=3,
        opening_quantity=200,
        starting_cash=1_000_000,
        maker_count=3,
        maker_refresh_interval=5,
        maker_quote_quantity=10,
        maker_base_half_spread=2,
        maker_latency_step=1,
        noise_order_count=8,
        warmup_duration=20,
        active_duration=200,
        noise_min_quantity=8,
        noise_max_quantity=24,
        noise_min_duration=40,
        noise_max_duration=100,
        noise_slice_interval=10,
        noise_max_slice_quantity=4,
    )


def test_named_scenario_profiles_are_stable_and_allow_episode_overrides() -> None:
    legacy = scenario_config_for_version(MVP_SCENARIO_VERSION, seed=9)
    mixed = scenario_config_for_version(
        ENDOGENOUS_MIXED_SCENARIO_VERSION,
        seed=9,
        active_duration=900 * 1_000_000_000,
    )

    assert legacy.seed == 9
    assert legacy.reservation_trader_count == 0
    assert mixed.seed == 9
    assert mixed.active_duration == 900 * 1_000_000_000
    for name, expected in ENDOGENOUS_MIXED_OVERRIDES.items():
        assert getattr(mixed, name) == expected

    with pytest.raises(ValueError, match="unknown scenario_version"):
        scenario_config_for_version("does-not-exist")


def test_scenario_exposes_configured_latent_demand_profile() -> None:
    scenario = create_populated_scenario(
        replace(
            config(seed=23),
            latent_demand_profile=LatentDemandProfile.STRONG_LONG,
        ),
        run_warmup=False,
    )

    assert scenario.config.latent_demand_profile is LatentDemandProfile.STRONG_LONG
    assert scenario.latent_parent_gross_quantity > 0
    assert scenario.latent_parent_signed_quantity > 0
    assert scenario.latent_parent_imbalance > 0.25

    coerced = replace(config(), latent_demand_profile="moderate-short")
    assert coerced.latent_demand_profile is LatentDemandProfile.MODERATE_SHORT
    with pytest.raises(ValueError, match="unknown latent_demand_profile"):
        replace(config(), latent_demand_profile="missing")


def test_warmed_scenario_has_heterogeneous_population_and_two_sided_book() -> None:
    scenario = create_populated_scenario(config())

    assert scenario.now == 20
    assert scenario.maker_ids == ("maker-01", "maker-02", "maker-03")
    assert len(scenario.noise_ids) == 8
    assert len(set(scenario.noise_ids)) == 8
    assert len(scenario.persistent_noise_ids) == scenario.config.persistent_noise_count
    assert len(scenario.informed_ids) == scenario.config.informed_trader_count
    assert len(scenario.latent_value_ids) == scenario.config.latent_value_trader_count
    assert all(not spec.technology.mbo_entitled for spec in scenario.episode.participant_specs)
    assert scenario.episode.exchange.book.best_bid is not None
    assert scenario.episode.exchange.book.best_ask is not None
    assert scenario.episode.exchange.book.best_bid < scenario.episode.exchange.book.best_ask


def test_same_seed_produces_identical_canonical_event_log() -> None:
    first = create_populated_scenario(config())
    second = create_populated_scenario(config())

    first.wait(200)
    second.wait(200)

    assert first.parent_orders == second.parent_orders
    assert first.episode.exchange.event_log.to_jsonl() == second.episode.exchange.event_log.to_jsonl()


def test_larger_maker_populations_repeat_bounded_size_variants() -> None:
    scenario = create_populated_scenario(
        ScenarioConfig(maker_count=8),
        run_warmup=False,
    )
    specs = {spec.participant_id: spec for spec in scenario.episode.participant_specs}

    assert [specs[f"maker-{index:02d}"].risk.max_order_quantity for index in range(1, 9)] == [
        80,
        100,
        120,
        140,
        80,
        100,
        120,
        140,
    ]


def test_wait_advances_market_without_player_activity() -> None:
    scenario = create_populated_scenario(config())
    events_before = len(scenario.episode.exchange.event_log)
    sequence_before = scenario.episode.exchange.event_log.last_sequence

    processed = scenario.wait(30)

    later_events = tuple(scenario.episode.exchange.event_log)[events_before:]
    assert processed > 0
    assert scenario.now == 50
    assert scenario.episode.exchange.event_log.last_sequence > sequence_before
    assert later_events
    assert any(event.market_time > 20 for event in later_events)
    assert any(event.kind in (EventKind.ORDER_ACCEPTED, EventKind.CANCEL_ACCEPTED) for event in later_events)


def test_temporary_opening_liquidity_withdraws_after_bootstrap() -> None:
    scenario = create_populated_scenario(
        replace(config(), opening_liquidity_duration=5),
        run_warmup=False,
    )
    scenario.wait(1)

    assert scenario.episode.exchange.book.orders_for_participant("opening-liquidity")

    scenario.wait(5)

    assert not scenario.episode.exchange.book.orders_for_participant("opening-liquidity")


def test_different_seed_changes_hidden_world_and_later_market_path() -> None:
    first = create_populated_scenario(config(seed=1))
    second = create_populated_scenario(config(seed=2))

    first.wait(200)
    second.wait(200)

    assert first.parent_orders != second.parent_orders
    assert first.episode.exchange.event_log.to_jsonl() != second.episode.exchange.event_log.to_jsonl()


def test_noise_institution_accounts_persist_across_rolling_mandates() -> None:
    scenario = create_populated_scenario(
        replace(
            config(seed=8),
            active_duration=1_000,
            noise_min_duration=20,
            noise_max_duration=30,
            noise_min_mandate_gap=10,
            noise_max_mandate_gap=20,
        )
    )
    mandates_by_institution = {
        participant_id: tuple(parent for parent in scenario.parent_orders if parent.participant_id == participant_id)
        for participant_id in scenario.noise_ids
    }

    assert set(mandates_by_institution) == set(scenario.noise_ids)
    assert any(len(mandates) > 1 for mandates in mandates_by_institution.values())
    assert any(parent.start_time > 300 for parent in scenario.parent_orders)
    specs = {
        spec.participant_id for spec in scenario.episode.participant_specs if spec.participant_id in scenario.noise_ids
    }
    assert specs == set(scenario.noise_ids)


def test_persistent_participant_groups_can_be_disabled() -> None:
    scenario = create_populated_scenario(
        replace(
            config(seed=3),
            persistent_noise_count=0,
            informed_trader_count=0,
            latent_value_trader_count=0,
        )
    )

    assert scenario.persistent_noise_ids == ()
    assert scenario.informed_ids == ()
    assert scenario.latent_value_ids == ()


def test_future_flow_informed_cohort_separates_quality_from_execution() -> None:
    scenario = create_populated_scenario(
        replace(
            config(seed=4),
            informed_trader_count=10,
            latent_value_trader_count=0,
        )
    )

    assert scenario.informed_ids == tuple(f"informed-{index:02d}" for index in range(1, 11))
    specs = {spec.participant_id: spec for spec in scenario.episode.participant_specs}
    assert all(
        InformationGrant.FUTURE_FLOW_SIGNAL in specs[participant_id].information_grants
        for participant_id in scenario.informed_ids
    )
    assert {specs[participant_id].risk.max_order_quantity for participant_id in scenario.informed_ids} == {
        scenario.config.informed_maximum_parent_quantity
    }


def test_adaptive_market_makers_can_join_legacy_cohort() -> None:
    scenario = create_populated_scenario(
        replace(
            config(seed=5),
            adaptive_maker_count=2,
            adaptive_quote_quantity_fraction=0.25,
        )
    )

    assert scenario.maker_ids == ("maker-01", "maker-02", "maker-03")
    assert scenario.adaptive_maker_ids == (
        "adaptive-maker-01",
        "adaptive-maker-02",
    )
    assert {spec.participant_id for spec in scenario.episode.participant_specs}.issuperset(scenario.adaptive_maker_ids)
    max_order_quantity = {
        spec.participant_id: spec.risk.max_order_quantity for spec in scenario.episode.participant_specs
    }
    assert max_order_quantity["maker-01"] == 10
    assert max_order_quantity["adaptive-maker-01"] == 3


def test_adaptive_market_makers_can_replace_legacy_cohort() -> None:
    scenario = create_populated_scenario(
        replace(
            config(seed=6),
            maker_count=0,
            adaptive_maker_count=3,
        )
    )

    assert scenario.maker_ids == ()
    assert len(scenario.adaptive_maker_ids) == 3
    assert scenario.episode.exchange.book.best_bid is not None
    assert scenario.episode.exchange.book.best_ask is not None


def test_informed_cohort_can_split_capacity_across_partial_observers() -> None:
    scenario = create_populated_scenario(
        replace(
            config(seed=7),
            latent_value_trader_count=4,
            latent_value_order_quantity=25,
            latent_value_signal_observation_probability=0.5,
            latent_value_signal_noise_step=1,
        )
    )

    assert scenario.latent_value_ids == (
        "latent-value-01",
        "latent-value-02",
        "latent-value-03",
        "latent-value-04",
    )
    max_order_quantity = {
        spec.participant_id: spec.risk.max_order_quantity for spec in scenario.episode.participant_specs
    }
    assert {max_order_quantity[participant_id] for participant_id in scenario.latent_value_ids} == {25}
    specs = {spec.participant_id: spec for spec in scenario.episode.participant_specs}
    assert all(
        InformationGrant.LATENT_VALUE_SIGNAL in specs[participant_id].information_grants
        for participant_id in scenario.latent_value_ids
    )
    assert InformationGrant.LATENT_VALUE_SIGNAL not in specs["maker-01"].information_grants


def test_endogenous_reservation_demand_is_opt_in_and_uses_hidden_target_events() -> None:
    scenario = create_populated_scenario(
        replace(
            config(seed=9),
            reservation_trader_count=8,
            reservation_cohort_count=4,
            reservation_update_interval=5,
            reservation_anchor_half_life=100,
            reservation_anchor_half_life_multipliers=(0.5, 2.0),
            reservation_anchor_dispersion_ticks=4,
            reservation_price_elasticity=1.0,
            reservation_price_elasticity_multipliers=(0.5, 2.0),
            reservation_base_target_spread=4,
            reservation_clip_quantity=2,
            cohort_demand_event_count=1,
            cohort_demand_start_delay=5,
            cohort_demand_end_buffer=120,
            cohort_demand_min_target_delta=8,
            cohort_demand_max_target_delta=8,
            cohort_demand_min_ramp_duration=10,
            cohort_demand_max_ramp_duration=10,
        ),
        run_warmup=False,
    )

    assert len(scenario.reservation_demand_ids) == 8
    assert scenario.reservation_aggregate_price_elasticity == 10.0
    assert len(scenario.cohort_demand_process.events) == 1
    event = scenario.cohort_demand_process.events[0]
    assert event.cohort_id in {
        "cohort-001",
        "cohort-002",
        "cohort-003",
        "cohort-004",
    }
    assert scenario.cohort_demand_process.target_shift(
        event.cohort_id,
        event.end_time,
    ) == event.side.signed(event.total_target_delta)
    specs = {spec.participant_id: spec for spec in scenario.episode.participant_specs}
    ordinary_grants = frozenset(
        (
            InformationGrant.PUBLIC_MARKET,
            InformationGrant.OWN_EXECUTIONS,
        )
    )
    assert all(
        specs[participant_id].information_grants == ordinary_grants
        for participant_id in scenario.reservation_demand_ids
    )
