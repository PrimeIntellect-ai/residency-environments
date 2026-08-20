from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from alphaverse.exchange import SessionState
from alphaverse.hosted import (
    EpisodeFinalized,
    EpisodeNotFound,
    EpisodeRegistry,
    InvalidCapability,
)
from alphaverse.models import NewOrder, Side
from alphaverse.profiles import ParticipantSpec
from alphaverse.time_accounting import TimeMode, TokenTimeConfig, TokenUsage


def _spec(participant_id: str = "player") -> ParticipantSpec:
    return ParticipantSpec(
        participant_id=participant_id,
        strategy_version_id="direct-api:v1",
        account_starting_cash=10_000,
    )


def _registry(**kwargs) -> EpisodeRegistry:
    ids = iter(("episode-b", "episode-a", "episode-c"))
    tokens = iter(("secret-b", "secret-a", "secret-c"))
    return EpisodeRegistry(
        episode_id_factory=lambda: next(ids),
        token_factory=lambda: next(tokens),
        **kwargs,
    )


def test_create_authenticates_an_isolated_player_session() -> None:
    registry = _registry()
    credentials = registry.create(_spec(), max_market_time=100)

    assert credentials.episode_id == "episode-b"
    assert credentials.capability_token == "secret-b"
    assert credentials.capture_token
    assert credentials.capture_token != credentials.capability_token
    assert credentials.max_market_time == 100
    assert registry.observe_session("episode-b", "secret-b", lambda session: session.participant_id) == "player"
    assert (
        registry.with_capture_session(
            "episode-b",
            credentials.capture_token,
            lambda session: session.participant_id,
        )
        == "player"
    )

    with pytest.raises(InvalidCapability, match="invalid episode capability"):
        registry.observe_session("episode-b", "wrong", lambda session: session.participant_id)
    with pytest.raises(InvalidCapability, match="invalid capture capability"):
        registry.with_capture_session(
            "episode-b",
            credentials.capability_token,
            lambda session: session.participant_id,
        )
    with pytest.raises(EpisodeNotFound, match="unknown episode"):
        registry.observe_session("missing", "secret-b", lambda session: session.participant_id)


def test_capture_capability_must_be_distinct_from_episode_capability() -> None:
    registry = EpisodeRegistry(
        token_factory=lambda: "same-secret",
        capture_token_factory=lambda: "same-secret",
    )

    with pytest.raises(ValueError, match="must be distinct"):
        registry.create(_spec())


def test_registry_serializes_operations_and_rejects_mutation_after_finalize() -> None:
    registry = _registry()
    credentials = registry.create(_spec())
    result = registry.with_session(
        credentials.episode_id,
        credentials.capability_token,
        lambda session: session.account(),
    )
    assert result["cash"] == 10_000

    registry.terminate(credentials.episode_id, credentials.capability_token)
    with pytest.raises(EpisodeFinalized, match="finalized"):
        registry.with_session(
            credentials.episode_id,
            credentials.capability_token,
            lambda session: session.wait(duration=1),
        )


def test_slow_episode_does_not_block_an_unrelated_episode() -> None:
    registry = _registry()
    slow_credentials = registry.create(_spec("slow-player"))
    fast_credentials = registry.create(_spec("fast-player"))
    entered = Event()
    release = Event()

    def slow_operation(session):
        entered.set()
        assert release.wait(timeout=2)
        return session.account()

    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(
            registry.with_session,
            slow_credentials.episode_id,
            slow_credentials.capability_token,
            slow_operation,
        )
        assert entered.wait(timeout=1)
        fast = pool.submit(
            registry.with_session,
            fast_credentials.episode_id,
            fast_credentials.capability_token,
            lambda session: session.account(),
        )
        try:
            assert fast.result(timeout=1)["participant_id"] == "fast-player"
        finally:
            release.set()
        assert slow.result(timeout=1)["participant_id"] == "slow-player"


def test_virtual_time_cap_clamps_wait_and_forces_finalization() -> None:
    registry = _registry()
    credentials = registry.create(_spec(), max_market_time=25)

    assert registry.run_for(credentials.episode_id, credentials.capability_token, 10) == 10
    assert registry.terminal_metrics(credentials.episode_id, credentials.capability_token) is None
    assert registry.run_until(credentials.episode_id, credentials.capability_token, 1_000) == 25

    metrics = registry.terminal_metrics(credentials.episode_id, credentials.capability_token)
    assert metrics is not None
    assert metrics.market_time == 25
    assert metrics.termination_state is SessionState.TERMINATED
    assert registry.run_for(credentials.episode_id, credentials.capability_token, 20) == 25


def test_terminal_metrics_are_reconstructed_from_canonical_events() -> None:
    registry = _registry()
    credentials = registry.create(_spec())

    def trade(session):
        exchange = session.episode.exchange
        exchange.register_account("maker", starting_cash=10_000)
        exchange.submit_order(
            NewOrder("maker", "maker-bid", Side.BUY, 99, 2),
            market_time=0,
        )
        exchange.submit_order(
            NewOrder("maker", "maker-ask", Side.SELL, 101, 2),
            market_time=0,
        )
        session.submit_limit(
            client_order_id="take-ask",
            side=Side.BUY,
            price=101,
            quantity=2,
        )
        session.cancel("does-not-exist")
        session.wait(duration=0)
        return exchange

    exchange = registry.with_session(
        credentials.episode_id,
        credentials.capability_token,
        trade,
    )

    metrics = registry.terminate(credentials.episode_id, credentials.capability_token)
    assert metrics.starting_cash == 10_000
    assert metrics.cash == 9_995.8
    assert metrics.position == 0
    assert metrics.pnl == -4.2
    assert metrics.fees_paid == 0.2
    assert metrics.fees_paid_subunits == 20
    assert metrics.order_count == 1  # Forced liquidation orders are excluded.
    assert metrics.fill_count == 2  # Entry plus terminal liquidation.
    assert metrics.rejection_count == 1
    assert metrics.order_rejection_count == 0
    assert metrics.cancel_rejection_count == 1
    assert metrics.margin_rejection_count == 0
    assert metrics.gross_filled_quantity == 4
    assert metrics.margin_call_count == 0
    assert metrics.margin_liquidation_count == 0
    assert metrics.margin_liquidated_quantity == 0
    assert metrics.max_abs_position == 2
    assert metrics.max_drawdown == 4.2
    assert metrics.deployment_count == 0
    assert metrics.unique_strategy_version_count == 0
    assert metrics.strategy_stop_count == 0
    assert metrics.strategy_fault_count == 0
    assert metrics.termination_state is SessionState.TERMINATED
    assert metrics.terminal_event_sequence == exchange.event_log.last_sequence
    series = registry.terminal_time_series(credentials.episode_id, credentials.capability_token)
    assert series is not None
    assert series[-1]["market_time_ns"] == metrics.market_time
    assert series[-1]["marked_pnl"] == -4.2
    assert series[-1]["focal_traded_quantity"] == 4
    assert series[-1]["position"] == 0


def test_terminal_time_series_has_regular_virtual_time_samples() -> None:
    registry = _registry()
    credentials = registry.create(_spec())
    registry.run_for(
        credentials.episode_id,
        credentials.capability_token,
        120_000_000_000,
    )

    registry.terminate(credentials.episode_id, credentials.capability_token)
    series = registry.terminal_time_series(credentials.episode_id, credentials.capability_token)

    assert series is not None
    assert [point["market_time_ns"] for point in series] == [
        0,
        60_000_000_000,
        120_000_000_000,
    ]
    assert all(point["marked_pnl"] == 0 for point in series)


def test_terminal_time_series_tracks_strategy_deployment_state() -> None:
    registry = _registry()
    credentials = registry.create(_spec())
    source = """
from alphaverse.strategy import Strategy

class StrategyImpl(Strategy):
    pass
"""
    registry.with_session(
        credentials.episode_id,
        credentials.capability_token,
        lambda session: session.deploy_source(source),
    )
    registry.run_for(
        credentials.episode_id,
        credentials.capability_token,
        60_000_000_000,
    )

    registry.terminate(credentials.episode_id, credentials.capability_token)
    series = registry.terminal_time_series(credentials.episode_id, credentials.capability_token)

    assert series is not None
    assert series[0]["deployment_count"] == 1
    assert series[0]["strategy_active"] is True
    assert series[-1]["strategy_active"] is False


def test_terminate_is_idempotent_under_concurrent_retries() -> None:
    registry = _registry()
    credentials = registry.create(_spec())

    def terminate():
        return registry.terminate(
            credentials.episode_id,
            credentials.capability_token,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(lambda _: terminate(), range(32)))

    assert all(result is results[0] for result in results)
    lifecycle_states = registry.observe_session(
        credentials.episode_id,
        credentials.capability_token,
        lambda session: [
            event.data.get("state")
            for event in session.episode.exchange.event_log
            if event.data.get("participant_id") == "player" and event.kind.value == "session"
        ],
    )
    assert lifecycle_states == ["liquidation_started", "terminated"]


def test_registry_charges_trusted_token_turns_and_reports_timing() -> None:
    registry = _registry()
    credentials = registry.create(
        _spec(),
        time_mode=TimeMode.TOKENS,
        token_time=TokenTimeConfig(
            ns_per_turn=10,
            ns_per_input_token=2,
            ns_per_output_token=3,
        ),
    )
    usage = TokenUsage(input_tokens=5, output_tokens=4)

    assert registry.commit_turn(credentials.episode_id, credentials.capability_token, "turn-1", usage) == 32
    assert registry.commit_turn(credentials.episode_id, credentials.capability_token, "turn-1", usage) == 0
    registry.run_for(credentials.episode_id, credentials.capability_token, 18)
    metrics = registry.terminate(credentials.episode_id, credentials.capability_token)

    assert metrics.market_time == 50
    assert metrics.time_mode is TimeMode.TOKENS
    assert metrics.charged_agent_time_ns == 32
    assert metrics.voluntary_wait_ns == 18
    assert metrics.model_turn_count == 1


def test_read_only_observation_of_finalized_wall_episode_does_not_advance_time() -> None:
    registry = _registry()
    credentials = registry.create(
        _spec(),
        time_mode=TimeMode.WALL,
        wall_time_scale=1_000_000_000,
        wall_quantum_ns=1,
    )
    metrics = registry.terminate(credentials.episode_id, credentials.capability_token)

    observed_time = registry.with_session(
        credentials.episode_id,
        credentials.capability_token,
        lambda session: session.now,
        allow_finalized=True,
    )

    assert observed_time == metrics.market_time


def test_observer_projection_does_not_charge_active_wall_time() -> None:
    registry = _registry()
    credentials = registry.create(
        _spec(),
        time_mode=TimeMode.WALL,
        wall_time_scale=1_000_000_000,
        wall_quantum_ns=1,
    )

    first = registry.observe_session(
        credentials.episode_id,
        credentials.capability_token,
        lambda session: session.now,
    )
    time.sleep(0.001)
    second = registry.observe_session(
        credentials.episode_id,
        credentials.capability_token,
        lambda session: session.now,
    )

    assert first == second == 0
