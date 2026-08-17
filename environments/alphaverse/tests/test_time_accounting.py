from __future__ import annotations

from alphaverse.episode import Episode
from alphaverse.player import PlayerSession
from alphaverse.profiles import ParticipantSpec
from alphaverse.time_accounting import (
    EpisodeTimeController,
    TimeMode,
    TokenTimeConfig,
    TokenUsage,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000

    def __call__(self) -> int:
        return self.now


def session() -> PlayerSession:
    return PlayerSession(
        Episode(),
        ParticipantSpec(
            participant_id="player",
            strategy_version_id="control:v1",
            account_starting_cash=10_000,
        ),
    )


def test_wall_time_is_quantized_and_charged_before_operations() -> None:
    clock = FakeClock()
    controller = EpisodeTimeController(
        session(),
        mode=TimeMode.WALL,
        wall_time_scale=2.0,
        wall_quantum_ns=10,
        monotonic_ns=clock,
    )
    # Evaluator/setup time before the first player operation is not charged.
    clock.now += 100
    assert controller.before_operation() == 0
    clock.now += 17

    assert controller.before_operation() == 30
    assert controller.advances[0].reason == "wall_time"


def test_voluntary_wait_resets_wall_anchor() -> None:
    clock = FakeClock()
    controller = EpisodeTimeController(
        session(),
        mode=TimeMode.WALL,
        wall_quantum_ns=1,
        monotonic_ns=clock,
    )
    controller.voluntary_wait(1_000)
    clock.now += 5

    assert controller.before_operation() == 1_005


def test_token_turn_charge_is_trusted_weighted_and_idempotent() -> None:
    controller = EpisodeTimeController(
        session(),
        mode=TimeMode.TOKENS,
        token_config=TokenTimeConfig(
            ns_per_turn=7,
            ns_per_input_token=2,
            ns_per_cached_input_token=1,
            ns_per_output_token=3,
        ),
    )
    usage = TokenUsage(input_tokens=10, cached_input_tokens=4, output_tokens=5)

    assert controller.commit_turn("turn-1", usage) == 46
    assert controller.commit_turn("turn-1", usage) == 0
    assert controller.session.now == 46


def test_time_cap_clamps_all_advance_sources() -> None:
    controller = EpisodeTimeController(
        session(),
        mode=TimeMode.TOKENS,
        token_config=TokenTimeConfig(ns_per_turn=100),
        max_market_time=40,
    )

    assert controller.commit_turn("turn-1", TokenUsage()) == 40
    assert controller.voluntary_wait(1_000) == 40


def test_session_boundary_and_pause_freeze_every_time_source() -> None:
    clock = FakeClock()
    controlled = session()
    controller = EpisodeTimeController(
        controlled,
        mode=TimeMode.WALL,
        wall_quantum_ns=1,
        monotonic_ns=clock,
    )
    controller.set_advance_limit(50)
    controller.voluntary_wait(1_000)

    assert controlled.now == 50
    controller.pause()
    clock.now += 10_000
    assert controller.before_operation() == 50
    assert controller.voluntary_wait(500) == 50

    controller.set_advance_limit(100)
    controller.resume()
    assert controller.before_operation() == 50
    clock.now += 10
    assert controller.before_operation() == 60
