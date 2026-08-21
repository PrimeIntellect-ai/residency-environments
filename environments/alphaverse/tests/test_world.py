from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
from alphaverse.models import Side
from alphaverse.world import (
    SECOND,
    CohortDemandConfig,
    CohortDemandEvent,
    CohortDemandProcess,
    DemandCause,
    ExecutionStyle,
    LatentDemandProfile,
    LatentValueProcess,
    ParentOrder,
    RegimeConfig,
    WorldGenerator,
)


def small_regime(**overrides: object) -> RegimeConfig:
    values: dict[str, object] = {
        "participant_count": 4,
        "parent_order_count": 12,
        "start_time": 100,
        "end_time": 1_100,
        "min_quantity": 10,
        "max_quantity": 50,
        "min_duration": 100,
        "max_duration": 400,
    }
    values.update(overrides)
    return RegimeConfig(**values)  # type: ignore[arg-type]


def mandate(
    participant: str,
    side: Side,
    quantity: int,
    start: int,
    end: int,
) -> ParentOrder:
    return ParentOrder(
        participant_id=participant,
        side=side,
        total_quantity=quantity,
        start_time=start,
        end_time=end,
        execution_style=ExecutionStyle.SCHEDULED,
    )


def test_same_seed_and_regime_produce_identical_worlds() -> None:
    regime = small_regime()

    first = WorldGenerator(73, regime)
    second = WorldGenerator(73, regime)

    assert first.parent_orders == second.parent_orders
    assert first.generate_parent_orders() == first.generate_parent_orders()


def test_latent_value_process_is_shared_and_counter_addressable() -> None:
    first = LatentValueProcess(
        seed=17,
        normal_step=2,
        shock_probability=0.1,
        shock_size=8,
    )
    second = LatentValueProcess(
        seed=17,
        normal_step=2,
        shock_probability=0.1,
        shock_size=8,
    )

    path = [first.innovation(index) for index in range(1, 21)]
    assert path == [second.innovation(index) for index in range(1, 21)]
    assert first.innovation(7) == first.innovation(7)
    assert all(-8 <= innovation <= 8 for innovation in path)


def test_different_seeds_produce_different_worlds() -> None:
    regime = small_regime()

    assert WorldGenerator(73, regime).parent_orders != WorldGenerator(74, regime).parent_orders


@pytest.mark.parametrize(
    ("short_profile", "long_profile"),
    [
        (LatentDemandProfile.MODERATE_SHORT, LatentDemandProfile.MODERATE_LONG),
        (LatentDemandProfile.STRONG_SHORT, LatentDemandProfile.STRONG_LONG),
    ],
)
def test_signed_demand_profiles_are_exact_side_mirrors(
    short_profile: LatentDemandProfile,
    long_profile: LatentDemandProfile,
) -> None:
    short = WorldGenerator(73, small_regime(parent_order_count=200, demand_skew=short_profile.skew)).parent_orders
    long = WorldGenerator(73, small_regime(parent_order_count=200, demand_skew=long_profile.skew)).parent_orders

    def unsigned(records):
        return [
            (
                record.participant_id,
                record.total_quantity,
                record.start_time,
                record.end_time,
                record.execution_style,
            )
            for record in records
        ]

    assert unsigned(short) == unsigned(long)
    assert all(left.side is not right.side for left, right in zip(short, long))
    assert sum(record.side.signed(record.total_quantity) for record in short) == -sum(
        record.side.signed(record.total_quantity) for record in long
    )


def test_stronger_profiles_increase_expected_latent_imbalance() -> None:
    gross_and_signed = []
    for profile in (
        LatentDemandProfile.BALANCED,
        LatentDemandProfile.MODERATE_LONG,
        LatentDemandProfile.STRONG_LONG,
    ):
        records = WorldGenerator(
            91,
            small_regime(parent_order_count=1_000, demand_skew=profile.skew),
        ).parent_orders
        gross = sum(record.total_quantity for record in records)
        signed = sum(record.side.signed(record.total_quantity) for record in records)
        gross_and_signed.append(signed / gross)

    balanced, moderate, strong = gross_and_signed
    assert abs(balanced) < 0.10
    assert 0.15 < moderate < strong
    assert strong > 0.45


def test_generated_orders_respect_regime_bounds_and_are_time_sorted() -> None:
    regime = small_regime()
    records = WorldGenerator(9, regime).parent_orders

    assert len(records) == regime.parent_order_count
    assert [record.start_time for record in records] == sorted(record.start_time for record in records)
    for record in records:
        assert regime.start_time <= record.start_time < record.end_time <= regime.end_time
        assert regime.min_quantity <= record.total_quantity <= regime.max_quantity
        assert record.execution_style in regime.execution_styles
        assert record.participant_id.startswith("noise-")


def test_rolling_mandates_are_reproducible_and_continue_past_three_minutes() -> None:
    regime = RegimeConfig(
        participant_count=3,
        parent_order_count=0,
        start_time=0,
        end_time=6 * 60 * 60 * SECOND,
        min_quantity=10,
        max_quantity=20,
        min_duration=10 * SECOND,
        max_duration=20 * SECOND,
        rolling_mandates=True,
        min_mandate_gap=5 * SECOND,
        max_mandate_gap=10 * SECOND,
        initial_mandate_spread=0,
    )
    first = WorldGenerator(19, regime)
    second = WorldGenerator(19, regime)

    assert first.parent_orders == second.parent_orders
    by_participant = {
        participant_id: tuple(parent for parent in first.parent_orders if parent.participant_id == participant_id)
        for participant_id in {parent.participant_id for parent in first.parent_orders}
    }
    assert all(len(mandates) > 10 for mandates in by_participant.values())
    assert all(
        earlier.end_time <= later.start_time
        for mandates in by_participant.values()
        for earlier, later in zip(mandates, mandates[1:])
    )
    assert any(parent.start_time > 3 * 60 * SECOND for parent in first.parent_orders)
    assert any(
        first.future_flow(at_time, 30 * SECOND) != 0
        for at_time in range(3 * 60 * SECOND, 6 * 60 * 60 * SECOND, 5 * SECOND)
    )
    for boundary in (60 * 60 * SECOND, 5 * 60 * 60 * SECOND):
        assert (
            sum(boundary <= parent.start_time < boundary + 60 * SECOND for parent in first.parent_orders)
            >= regime.participant_count
        )


def test_future_flow_filters_to_horizon_and_prorates_active_mandates() -> None:
    generator = WorldGenerator(1, small_regime(parent_order_count=0))
    records = (
        mandate("before", Side.BUY, 100, 0, 10),
        mandate("active", Side.BUY, 100, 10, 20),
        mandate("future", Side.BUY, 100, 30, 40),
    )

    # The interval [12, 17] contains half of only the active mandate.
    assert generator.future_signed_remaining_flow(12, 5, records) == 50
    assert generator.future_signed_remaining_flow(20, 5, records) == 0


def test_future_flow_aggregates_buy_positive_and_sell_negative() -> None:
    generator = WorldGenerator(1, small_regime(parent_order_count=0))
    records = (
        mandate("buyer", Side.BUY, 80, 10, 20),
        mandate("seller", Side.SELL, 30, 10, 20),
        mandate("later-seller", Side.SELL, 100, 20, 30),
    )

    assert generator.future_flow(10, 10, records) == 50
    assert generator.future_flow(10, 20, records) == -50
    assert generator.future_flow(10, 0, records) == 0


def test_records_configuration_and_generated_collection_are_immutable() -> None:
    mutable_styles = [ExecutionStyle.SCHEDULED]
    regime = small_regime(execution_styles=mutable_styles)
    generator = WorldGenerator(2, regime)
    original = generator.parent_orders

    mutable_styles.append(ExecutionStyle.BURST)

    assert regime.execution_styles == (ExecutionStyle.SCHEDULED,)
    assert generator.parent_orders is original
    assert isinstance(generator.parent_orders, tuple)
    with pytest.raises(FrozenInstanceError):
        generator.parent_orders[0].total_quantity = 999  # type: ignore[misc]
    assert generator.parent_orders == original


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("participant_count", 0, "participant_count must be positive"),
        ("parent_order_count", -1, "parent_order_count must be non-negative"),
        ("start_time", -1, "start_time must be non-negative"),
        ("min_quantity", 0, "min_quantity must be positive"),
        ("min_duration", 0, "min_duration must be positive"),
    ],
)
def test_regime_rejects_invalid_bounds(field: str, value: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        small_regime(**{field: value})


def test_parent_order_and_flow_validate_time_and_quantity() -> None:
    with pytest.raises(ValueError, match="total_quantity must be positive"):
        mandate("bad", Side.BUY, 0, 1, 2)
    with pytest.raises(ValueError, match="end_time must be greater"):
        mandate("bad", Side.BUY, 1, 2, 2)

    generator = WorldGenerator(1, small_regime(parent_order_count=0))
    with pytest.raises(ValueError, match="at_time must be non-negative"):
        generator.future_flow(-1, 2)
    with pytest.raises(ValueError, match="horizon must be non-negative"):
        generator.future_flow(1, -2)


def cohort_config(**overrides: object) -> CohortDemandConfig:
    values: dict[str, object] = {
        "cohort_count": 3,
        "event_count": 4,
        "start_time": 100,
        "end_time": 900,
        "min_target_delta": 10,
        "max_target_delta": 30,
        "min_ramp_duration": 40,
        "max_ramp_duration": 160,
    }
    values.update(overrides)
    return CohortDemandConfig(**values)  # type: ignore[arg-type]


def cohort_event(
    event_id: str,
    cohort_id: str,
    side: Side,
    quantity: int,
    start: int,
    end: int,
) -> CohortDemandEvent:
    return CohortDemandEvent(
        event_id=event_id,
        cohort_id=cohort_id,
        side=side,
        total_target_delta=quantity,
        start_time=start,
        end_time=end,
    )


def test_demand_cause_is_diagnostic_source_label() -> None:
    assert DemandCause.IDIOSYNCRATIC.value == "idiosyncratic"
    assert DemandCause.COHORT.value == "cohort"


def test_cohort_demand_events_are_deterministic_and_bucket_spaced() -> None:
    config = cohort_config()
    first = CohortDemandProcess(73, config)
    second = CohortDemandProcess(73, config)
    changed_seed = CohortDemandProcess(74, config)

    assert first.events == second.events
    assert first.generate_events() is first.events
    assert first.events != changed_seed.events
    assert [event.event_id for event in first.events] == [
        "cohort-demand-001",
        "cohort-demand-002",
        "cohort-demand-003",
        "cohort-demand-004",
    ]
    window = config.end_time - config.start_time
    for index, event in enumerate(first.events):
        bucket_start = config.start_time + index * window // config.event_count
        bucket_end = config.start_time + (index + 1) * window // config.event_count
        assert bucket_start <= event.start_time < event.end_time <= bucket_end
        assert event.cohort_id.startswith("cohort-")
        assert config.min_target_delta <= event.total_target_delta <= config.max_target_delta


def test_cohort_event_ramps_linearly_then_persists() -> None:
    buy = cohort_event("event-1", "cohort-001", Side.BUY, 100, 10, 20)
    sell = cohort_event("event-2", "cohort-001", Side.SELL, 100, 10, 20)

    assert [buy.cumulative_target_change(at) for at in (0, 10, 12, 15, 19, 20, 100)] == [
        0,
        0,
        20,
        50,
        90,
        100,
        100,
    ]
    assert sell.cumulative_target_change(15) == -50
    assert buy.target_change_between(12, 17) == 50
    assert buy.target_change_between(20, 100) == 0


def test_cohort_event_allows_zero_integer_progress_inside_ramp() -> None:
    event = cohort_event("event", "cohort-001", Side.BUY, 2, 10, 20)

    assert event.cumulative_target_change(11) == 0


def test_cohort_process_aggregates_target_levels_and_future_changes() -> None:
    process = CohortDemandProcess(1, cohort_config(event_count=0))
    first = cohort_event("buy-a", "cohort-001", Side.BUY, 100, 10, 20)
    second = cohort_event("sell-a", "cohort-001", Side.SELL, 40, 15, 25)
    third = cohort_event("buy-b", "cohort-002", Side.BUY, 80, 10, 30)
    process._events = (first, second, third)

    assert process.target_shift("cohort-001", 12) == 20
    assert process.target_shift("cohort-001", 20) == 80
    assert process.target_shift("cohort-001", 30) == 60
    assert process.target_shift("cohort-002", 40) == 80
    assert process.future_signed_target_change(12, 8) == 92
    assert process.future_signed_target_change(12, 8, ("cohort-001",)) == 60
    assert process.future_signed_target_change(25, 100) == 20


def test_cohort_demand_types_have_no_price_or_value_fields() -> None:
    forbidden = {"price", "value", "fair_value", "reference_price"}
    for type_ in (CohortDemandConfig, CohortDemandEvent):
        assert forbidden.isdisjoint(field.name for field in fields(type_))
    assert forbidden.isdisjoint(vars(CohortDemandProcess(1, cohort_config(event_count=0))))


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (
            lambda: cohort_event("", "cohort-001", Side.BUY, 1, 1, 2),
            "event_id must not be empty",
        ),
        (
            lambda: cohort_event("event", "", Side.BUY, 1, 1, 2),
            "cohort_id must not be empty",
        ),
        (
            lambda: cohort_event("event", "cohort", Side.BUY, 0, 1, 2),
            "total_target_delta must be positive",
        ),
        (
            lambda: cohort_event("event", "cohort", Side.BUY, 1, -1, 2),
            "start_time must be non-negative",
        ),
        (
            lambda: cohort_event("event", "cohort", Side.BUY, 1, 2, 2),
            "end_time must be greater",
        ),
        (
            lambda: cohort_config(cohort_count=0),
            "cohort_count must be positive",
        ),
        (
            lambda: cohort_config(event_count=-1),
            "event_count must be non-negative",
        ),
        (
            lambda: cohort_config(max_target_delta=9),
            "max_target_delta must be at least",
        ),
        (
            lambda: cohort_config(max_ramp_duration=39),
            "max_ramp_duration must be at least",
        ),
        (
            lambda: cohort_config(event_count=40),
            "min_ramp_duration must fit within every event time bucket",
        ),
    ],
)
def test_cohort_demand_configuration_rejects_invalid_values(constructor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()


def test_cohort_demand_process_and_queries_validate_inputs() -> None:
    with pytest.raises(ValueError, match="seed must be non-negative"):
        CohortDemandProcess(-1)
    process = CohortDemandProcess(1, cohort_config(event_count=0))
    with pytest.raises(ValueError, match="cohort_id must not be empty"):
        process.target_shift("", 0)
    with pytest.raises(ValueError, match="at_time must be non-negative"):
        process.target_shift("cohort-001", -1)
    with pytest.raises(ValueError, match="horizon must be non-negative"):
        process.future_signed_target_change(0, -1)
    with pytest.raises(ValueError, match="cohort_ids must contain non-empty strings"):
        process.future_signed_target_change(0, 1, ("",))
