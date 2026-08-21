from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from alphaverse.episode_state import EpisodeFinalized, EpisodeState
from alphaverse.exchange import SessionState
from alphaverse.models import NewOrder, Side
from alphaverse.profiles import ParticipantSpec
from alphaverse.time_accounting import TimeMode


def _spec(participant_id: str = "player") -> ParticipantSpec:
    return ParticipantSpec(
        participant_id=participant_id,
        strategy_version_id="direct-api:v1",
        account_starting_cash=10_000,
    )


def _state(**kwargs) -> EpisodeState:
    return EpisodeState("episode-b", _spec(), **kwargs)


def test_state_owns_one_episode_with_participant_scoped_views() -> None:
    state = _state(max_market_time=100)
    state.add_participant(_spec("prop"))

    assert state.observe_session("player", lambda session: session.participant_id) == "player"
    assert state.with_capture_session("prop", lambda session: session.participant_id) == "prop"

    with pytest.raises(ValueError, match="unknown participant"):
        state.observe_session("missing", lambda session: session.participant_id)


def test_state_rejects_unwired_token_time() -> None:
    with pytest.raises(ValueError, match="trusted turn-usage bridge"):
        _state(time_mode=TimeMode.TOKENS)


def test_state_serializes_operations_and_rejects_mutation_after_finalize() -> None:
    state = _state()
    result = state.with_session("player", lambda session: session.account())
    assert result["cash"] == 10_000

    state.terminate()
    with pytest.raises(EpisodeFinalized, match="finalized"):
        state.with_session("player", lambda session: session.wait(duration=1))


def test_virtual_time_cap_clamps_wait_and_forces_finalization() -> None:
    state = _state(max_market_time=25)

    assert state.run_for(10) == 10
    assert state.terminal_metrics() is None
    assert state.run_until(1_000) == 25

    metrics = state.terminal_metrics()
    assert metrics is not None
    assert metrics.market_time == 25
    assert metrics.termination_state is SessionState.TERMINATED
    assert state.run_for(20) == 25


def test_terminal_metrics_are_reconstructed_from_canonical_events() -> None:
    state = _state()

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

    exchange = state.with_session("player", trade)
    metrics = state.terminate()

    assert metrics.starting_cash == 10_000
    assert metrics.cash == 9_995.8
    assert metrics.position == 0
    assert metrics.pnl == -4.2
    assert metrics.fees_paid == 0.2
    assert metrics.fees_paid_subunits == 20
    assert metrics.order_count == 1
    assert metrics.fill_count == 2
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
    series = state.terminal_time_series()
    assert series is not None
    assert series[-1]["market_time_ns"] == metrics.market_time
    assert series[-1]["marked_pnl"] == -4.2
    assert series[-1]["focal_traded_quantity"] == 4
    assert series[-1]["position"] == 0


def test_terminal_time_series_has_regular_virtual_time_samples() -> None:
    state = _state()
    state.run_for(120_000_000_000)
    state.terminate()
    series = state.terminal_time_series()

    assert series is not None
    assert [point["market_time_ns"] for point in series] == [
        0,
        60_000_000_000,
        120_000_000_000,
    ]
    assert all(point["marked_pnl"] == 0 for point in series)


def test_terminal_time_series_tracks_strategy_deployment_state() -> None:
    state = _state()
    source = """
from alphaverse.strategy import Strategy

class StrategyImpl(Strategy):
    pass
"""
    state.with_session(
        "player",
        lambda session: session.deploy_source(source),
    )
    state.run_for(60_000_000_000)
    state.terminate()
    series = state.terminal_time_series()

    assert series is not None
    assert series[0]["deployment_count"] == 1
    assert series[0]["strategy_active"] is True
    assert series[-1]["strategy_active"] is False


def test_terminate_is_idempotent_under_concurrent_retries() -> None:
    state = _state()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(lambda _: state.terminate(), range(32)))

    assert all(result is results[0] for result in results)
    lifecycle_states = state.observe_session(
        "player",
        lambda session: [
            event.data.get("state")
            for event in session.episode.exchange.event_log
            if event.data.get("participant_id") == "player" and event.kind.value == "session"
        ],
    )
    assert lifecycle_states == ["liquidation_started", "terminated"]


def test_read_only_observation_of_finalized_wall_episode_does_not_advance_time() -> None:
    state = _state(
        time_mode=TimeMode.WALL,
        wall_time_scale=1_000_000_000,
        wall_quantum_ns=1,
    )
    metrics = state.terminate()

    observed_time = state.with_session(
        "player",
        lambda session: session.now,
        allow_finalized=True,
    )

    assert observed_time == metrics.market_time


def test_observer_projection_does_not_charge_active_wall_time() -> None:
    state = _state(
        time_mode=TimeMode.WALL,
        wall_time_scale=1_000_000_000,
        wall_quantum_ns=1,
    )

    first = state.observe_session("player", lambda session: session.now)
    time.sleep(0.001)
    second = state.observe_session("player", lambda session: session.now)

    assert first == second == 0
