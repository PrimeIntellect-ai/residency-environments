"""Coherent state and finalization for one task-scoped Alphaverse episode."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import TypeVar

from alphaverse.episode import Episode
from alphaverse.evaluation_series import build_evaluation_series
from alphaverse.exchange import SessionState
from alphaverse.models import EventKind
from alphaverse.player import PlayerSession, WaitResult
from alphaverse.profiles import ParticipantSpec
from alphaverse.time_accounting import (
    EpisodeTimeController,
    TimeMode,
)


class EpisodeFinalized(RuntimeError):
    """A mutating operation was requested after terminal finalization."""


class MarketSessionState(str, Enum):
    """Whether virtual market time may currently advance."""

    OPEN = "open"
    INTERMISSION = "intermission"
    FINALIZED = "finalized"


@dataclass(frozen=True, slots=True)
class MarketSessionInfo:
    """Public session-boundary state."""

    state: MarketSessionState
    session_index: int
    session_start_ns: int
    session_end_ns: int | None
    market_time_ns: int


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Canonical terminal metrics for the focal participant.

    Execution statistics are reconstructed from the exchange event log.  Cash
    and position use the last account/session event, with the registered
    starting account as the zero-trade baseline.  ``pnl`` is realized cash PnL;
    a nonzero terminal position is therefore visible rather than silently
    marked using noncanonical state.
    """

    episode_id: str
    participant_id: str
    market_time: int
    terminal_event_sequence: int
    starting_cash: int
    cash: int | float
    cash_subunits: int
    cash_subunits_per_tick: int
    position: int
    pnl: int | float
    fees_paid: int | float
    fees_paid_subunits: int
    order_count: int
    fill_count: int
    rejection_count: int
    order_rejection_count: int
    cancel_rejection_count: int
    margin_rejection_count: int
    gross_filled_quantity: int
    margin_call_count: int
    margin_liquidation_count: int
    margin_liquidated_quantity: int
    max_abs_position: int
    max_drawdown: int | float
    deployment_count: int
    unique_strategy_version_count: int
    strategy_stop_count: int
    strategy_fault_count: int
    termination_state: SessionState
    time_mode: TimeMode
    voluntary_wait_ns: int
    charged_agent_time_ns: int
    model_turn_count: int


@dataclass(slots=True)
class _EpisodeRecord:
    episode_id: str
    max_market_time: int | None
    focal_participant_id: str
    session: PlayerSession
    time_controller: EpisodeTimeController
    sessions: dict[str, PlayerSession] = field(default_factory=dict)
    session_duration_ns: int | None = None
    session_index: int = 1
    session_start_ns: int = 0
    session_end_ns: int | None = None
    intermission: bool = False
    terminal_metrics: EpisodeMetrics | None = None
    terminal_metrics_by_participant: dict[str, EpisodeMetrics] = field(default_factory=dict)
    terminal_time_series: tuple[dict[str, object], ...] | None = None
    terminal_time_series_by_participant: dict[str, tuple[dict[str, object], ...]] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


_T = TypeVar("_T")


class EpisodeState:
    """Own the sessions, clock, and terminal data for one Toolset episode."""

    def __init__(
        self,
        episode_id: str,
        focal_spec: ParticipantSpec,
        *,
        episode: Episode | None = None,
        max_market_time: int | None = None,
        time_mode: TimeMode = TimeMode.MANUAL,
        wall_time_scale: float = 1.0,
        wall_quantum_ns: int = 1_000_000,
        session_duration_ns: int | None = None,
    ) -> None:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        if time_mode is TimeMode.TOKENS:
            raise ValueError("token time requires a trusted turn-usage bridge")

        if max_market_time is not None:
            if isinstance(max_market_time, bool) or not isinstance(max_market_time, int):
                raise TypeError("max_market_time must be an int")
            if max_market_time < 0:
                raise ValueError("max_market_time must be non-negative")
        if session_duration_ns is not None:
            if isinstance(session_duration_ns, bool) or not isinstance(session_duration_ns, int):
                raise TypeError("session_duration_ns must be an int")
            if session_duration_ns <= 0:
                raise ValueError("session_duration_ns must be positive")
        owned_episode = episode or Episode(session_id=episode_id)
        if max_market_time is not None and max_market_time < owned_episode.now:
            raise ValueError("max_market_time precedes the episode start")
        session = PlayerSession(owned_episode, focal_spec)
        time_controller = EpisodeTimeController(
            session,
            mode=time_mode,
            wall_time_scale=wall_time_scale,
            wall_quantum_ns=wall_quantum_ns,
            max_market_time=max_market_time,
        )
        session_end_ns = (
            None
            if session_duration_ns is None
            else min(
                session.now + session_duration_ns,
                max_market_time if max_market_time is not None else session.now + session_duration_ns,
            )
        )
        time_controller.set_advance_limit(session_end_ns)
        self._record = _EpisodeRecord(
            episode_id=episode_id,
            max_market_time=max_market_time,
            focal_participant_id=focal_spec.participant_id,
            session=session,
            time_controller=time_controller,
            sessions={focal_spec.participant_id: session},
            session_duration_ns=session_duration_ns,
            session_start_ns=session.now,
            session_end_ns=session_end_ns,
        )
        if max_market_time == session.now:
            self._finalize(self._record)

    def add_participant(
        self,
        spec: ParticipantSpec,
        *,
        baseline_source: str | None = None,
    ) -> None:
        """Add another isolated account to the shared episode."""

        with self._access_record() as record:
            if record.terminal_metrics is not None:
                raise EpisodeFinalized(f"episode is finalized: {record.episode_id}")
            if spec.participant_id in record.sessions:
                raise ValueError(f"participant already exists: {spec.participant_id}")
            session = PlayerSession(record.session.episode, spec)
            if baseline_source is not None:
                # Seed source is evaluator-owned. Later role deployments use the
                # ordinary untrusted path and cannot import private participants.
                session.deploy_trusted_source(baseline_source)
            record.sessions[spec.participant_id] = session

    def with_session(
        self,
        participant_id: str,
        operation: Callable[[PlayerSession], _T],
        *,
        allow_finalized: bool = False,
    ) -> _T:
        """Run a charged participant operation under the episode lock."""

        with self._access_record() as record:
            if record.terminal_metrics is None:
                record.time_controller.before_operation()
                self._enforce_market_cap(record)
                self._sync_intermission(record)
            session = self._session(record, participant_id)
            if record.terminal_metrics is not None:
                if not allow_finalized:
                    raise EpisodeFinalized(f"episode is finalized: {record.episode_id}")
                return operation(session)
            result = operation(session)
            self._enforce_market_cap(record)
            self._sync_intermission(record)
            return result

    def observe_session(
        self,
        participant_id: str,
        operation: Callable[[PlayerSession], _T],
    ) -> _T:
        """Run a read-only projection without charging or advancing market time.

        The episode lock still provides a coherent observation point. Unlike
        :meth:`with_session`, this path never invokes the configured wall-time
        bridge, so observer refresh cadence cannot drive an episode.
        """

        with self._access_record() as record:
            return operation(self._session(record, participant_id))

    def with_capture_session(
        self,
        participant_id: str,
        operation: Callable[[PlayerSession], _T],
    ) -> _T:
        """Run a read-only capture after charging this episode's time bridge."""

        with self._access_record() as record:
            if record.terminal_metrics is None:
                record.time_controller.before_operation()
                self._enforce_market_cap(record)
                self._sync_intermission(record)
            return operation(self._session(record, participant_id))

    def run_until(
        self,
        market_time: int,
        *,
        interrupt_on_alert: bool = False,
    ) -> int | WaitResult:
        """Advance virtual time, stopping exactly at the configured episode cap."""

        if isinstance(market_time, bool) or not isinstance(market_time, int):
            raise TypeError("market_time must be an int")
        with self._access_record() as record:
            record.time_controller.before_operation()
            self._enforce_market_cap(record)
            self._sync_intermission(record)
            if record.terminal_metrics is not None:
                return record.session.now
            if record.intermission:
                return WaitResult(
                    market_time=record.session.now,
                    target_market_time=market_time,
                    interrupted_by_alert=False,
                )
            target = market_time
            cap = record.max_market_time
            if cap is not None:
                target = min(target, cap)
            result = record.time_controller.voluntary_wait(
                max(0, target - record.session.now),
                interrupt_on_alert=interrupt_on_alert,
            )
            self._enforce_market_cap(record)
            self._sync_intermission(record)
            return result

    def run_for(
        self,
        duration: int,
        *,
        interrupt_on_alert: bool = False,
    ) -> int | WaitResult:
        """Advance virtual time by a duration, respecting the configured cap."""

        if isinstance(duration, bool) or not isinstance(duration, int):
            raise TypeError("duration must be an int")
        if duration < 0:
            raise ValueError("duration must be non-negative")
        with self._access_record() as record:
            record.time_controller.before_operation()
            self._enforce_market_cap(record)
            self._sync_intermission(record)
            if record.terminal_metrics is not None:
                return record.session.now
            if record.intermission:
                return WaitResult(
                    market_time=record.session.now,
                    target_market_time=record.session.now + duration,
                    interrupted_by_alert=False,
                )
            result = record.time_controller.voluntary_wait(
                duration,
                interrupt_on_alert=interrupt_on_alert,
            )
            self._enforce_market_cap(record)
            self._sync_intermission(record)
            return result

    def terminate(
        self,
        participant_id: str | None = None,
    ) -> EpisodeMetrics:
        """Finalize once and return the same persisted metrics on every retry."""

        with self._access_record() as record:
            record.time_controller.before_operation()
            metrics = self._finalize(record)
            session = self._session(record, participant_id or record.focal_participant_id)
            return record.terminal_metrics_by_participant.get(
                session.participant_id,
                metrics,
            )

    def deploy_source(
        self,
        participant_id: str,
        source: str,
        *,
        entrypoint: str,
    ) -> tuple[str, bool, dict[str, object]]:
        """Deploy while open, or stage the validated artifact during a close."""

        with self._access_record() as record:
            if record.terminal_metrics is not None:
                raise EpisodeFinalized(f"episode is finalized: {record.episode_id}")
            record.time_controller.before_operation()
            self._enforce_market_cap(record)
            self._sync_intermission(record)
            session = self._session(record, participant_id)
            if record.intermission:
                version_id = session.stage_source(source, entrypoint=entrypoint)
                staged = True
            else:
                version_id = session.deploy_source(source, entrypoint=entrypoint)
                staged = False
            return version_id, staged, session.strategy_status()

    def market_session_info(self) -> MarketSessionInfo:
        """Return the current open/intermission boundary state without charging time."""

        with self._access_record() as record:
            self._sync_intermission(record)
            return self._market_session_info(record)

    def require_market_open(self) -> None:
        """Reject direct trading mutations while the whole market is frozen."""

        with self._access_record() as record:
            self._sync_intermission(record)
            if record.intermission:
                raise ValueError("market is closed for a research intermission")
            if record.terminal_metrics is not None:
                raise EpisodeFinalized(f"episode is finalized: {record.episode_id}")

    def resume_market_session(self) -> MarketSessionInfo:
        """Activate staged strategies and open the next virtual-time chunk."""

        with self._access_record() as record:
            if record.terminal_metrics is not None:
                return self._market_session_info(record)
            if record.session_duration_ns is None:
                raise RuntimeError("episode has no market-session schedule")
            if not record.intermission:
                raise RuntimeError("market is not in an intermission")
            for participant_id in sorted(record.sessions):
                session = record.sessions[participant_id]
                try:
                    session.activate_staged_source()
                except Exception as exc:
                    # Candidate validation normally catches constructor/import
                    # errors during the research turn. A later nondeterministic
                    # activation failure must still be participant-local: keep
                    # the incumbent and reopen the rest of the market.
                    session.discard_staged_source(f"staged activation failed: {type(exc).__name__}: {exc}")
            record.session_index += 1
            record.session_start_ns = record.session.now
            boundary = record.session.now + record.session_duration_ns
            cap = record.max_market_time
            record.session_end_ns = min(boundary, cap) if cap is not None else boundary
            record.intermission = False
            record.time_controller.set_advance_limit(record.session_end_ns)
            record.time_controller.resume()
            return self._market_session_info(record)

    def terminal_metrics(
        self,
        participant_id: str | None = None,
    ) -> EpisodeMetrics | None:
        """Return the persisted terminal snapshot, if finalization has occurred."""

        with self._access_record() as record:
            self._enforce_market_cap(record)
            if record.terminal_metrics is None:
                return None
            selected = participant_id or record.focal_participant_id
            return record.terminal_metrics_by_participant.get(selected)

    def terminal_time_series(
        self,
        participant_id: str | None = None,
    ) -> tuple[dict[str, object], ...] | None:
        """Return evaluator diagnostics produced during terminal finalization."""

        with self._access_record() as record:
            self._enforce_market_cap(record)
            if record.terminal_metrics is not None and not record.terminal_time_series_by_participant:
                self._finalize(record)
            if record.terminal_metrics is None:
                return None
            selected = participant_id or record.focal_participant_id
            return record.terminal_time_series_by_participant.get(selected)

    @contextmanager
    def _access_record(self):
        """Hold the episode lock for one coherent operation."""

        record = self._record
        with record.lock:
            yield record

    def _enforce_market_cap(self, record: _EpisodeRecord) -> None:
        cap = record.max_market_time
        if cap is not None and record.session.now >= cap and record.terminal_metrics is None:
            self._finalize(record)

    def _sync_intermission(self, record: _EpisodeRecord) -> None:
        if (
            record.terminal_metrics is None
            and record.session_end_ns is not None
            and record.session.now >= record.session_end_ns
            and (record.max_market_time is None or record.session.now < record.max_market_time)
        ):
            record.intermission = True
            record.time_controller.pause()

    @staticmethod
    def _market_session_info(record: _EpisodeRecord) -> MarketSessionInfo:
        state = (
            MarketSessionState.FINALIZED
            if record.terminal_metrics is not None
            else MarketSessionState.INTERMISSION
            if record.intermission
            else MarketSessionState.OPEN
        )
        return MarketSessionInfo(
            state=state,
            session_index=record.session_index,
            session_start_ns=record.session_start_ns,
            session_end_ns=record.session_end_ns,
            market_time_ns=record.session.now,
        )

    @staticmethod
    def _session(record: _EpisodeRecord, participant_id: str) -> PlayerSession:
        try:
            return record.sessions[participant_id]
        except KeyError as exc:
            raise ValueError(f"unknown participant: {participant_id}") from exc

    def _finalize(self, record: _EpisodeRecord) -> EpisodeMetrics:
        if record.terminal_metrics is None:
            for participant_id in sorted(record.sessions):
                record.sessions[participant_id].terminate()
            record.intermission = False
            record.time_controller.pause()
            record.terminal_metrics_by_participant = {
                participant_id: self._derive_metrics(
                    record,
                    session=session,
                )
                for participant_id, session in record.sessions.items()
            }
            record.terminal_metrics = record.terminal_metrics_by_participant[record.focal_participant_id]
        if record.terminal_time_series is None:
            record.terminal_time_series_by_participant = {
                participant_id: tuple(build_evaluation_series(session))
                for participant_id, session in record.sessions.items()
            }
            record.terminal_time_series = record.terminal_time_series_by_participant[record.focal_participant_id]
        return record.terminal_metrics

    @staticmethod
    def _derive_metrics(
        record: _EpisodeRecord,
        *,
        session: PlayerSession | None = None,
    ) -> EpisodeMetrics:
        session = session or record.session
        participant_id = session.participant_id
        exchange = session.episode.exchange
        account = exchange.clearing.snapshot(participant_id)
        cash_subunits = account.starting_cash * account.cash_subunits_per_tick
        fees_paid_subunits = 0
        position = 0
        order_count = 0
        fill_count = 0
        rejection_count = 0
        order_rejection_count = 0
        cancel_rejection_count = 0
        margin_rejection_count = 0
        gross_filled_quantity = 0
        margin_call_count = 0
        margin_liquidation_count = 0
        margin_liquidated_quantity = 0
        max_abs_position = 0
        termination_state = SessionState.ACTIVE
        mark_price: Fraction | None = None
        peak_equity_subunits = Fraction(cash_subunits)
        max_drawdown_subunits = Fraction(0)

        def update_drawdown() -> None:
            nonlocal peak_equity_subunits, max_drawdown_subunits
            equity_subunits = Fraction(cash_subunits)
            if mark_price is not None:
                equity_subunits += (
                    position * mark_price * exchange.product.contract_multiplier * account.cash_subunits_per_tick
                )
            peak_equity_subunits = max(peak_equity_subunits, equity_subunits)
            max_drawdown_subunits = max(
                max_drawdown_subunits,
                peak_equity_subunits - equity_subunits,
            )

        for event in session.episode.exchange.event_log:
            if event.kind is EventKind.LEVELS:
                raw_bids = event.data.get("bids")
                raw_asks = event.data.get("asks")
                best_bid = (
                    raw_bids[0].get("price")
                    if isinstance(raw_bids, list) and raw_bids and isinstance(raw_bids[0], dict)
                    else None
                )
                best_ask = (
                    raw_asks[0].get("price")
                    if isinstance(raw_asks, list) and raw_asks and isinstance(raw_asks[0], dict)
                    else None
                )
                if isinstance(best_bid, int) and isinstance(best_ask, int):
                    mark_price = Fraction(best_bid + best_ask, 2)
                elif isinstance(best_bid, int):
                    mark_price = Fraction(best_bid)
                elif isinstance(best_ask, int):
                    mark_price = Fraction(best_ask)
                update_drawdown()
                continue
            if event.data.get("participant_id") != participant_id:
                continue
            if event.kind is EventKind.ORDER_ACCEPTED:
                if event.data.get("liquidation_reason") == "margin":
                    margin_liquidation_count += 1
                    quantity = event.data.get("quantity")
                    if isinstance(quantity, int):
                        margin_liquidated_quantity += quantity
                elif not event.data.get("liquidation", False):
                    order_count += 1
            elif event.kind is EventKind.ORDER_REJECTED:
                rejection_count += 1
                order_rejection_count += 1
                if event.data.get("reason") in {
                    "insufficient_initial_margin",
                    "margin_reduce_only",
                }:
                    margin_rejection_count += 1
            elif event.kind is EventKind.CANCEL_REJECTED:
                rejection_count += 1
                cancel_rejection_count += 1
            elif event.kind is EventKind.FILL:
                quantity = event.data.get("quantity")
                if isinstance(quantity, int):
                    fill_count += 1
                    gross_filled_quantity += quantity
            elif event.kind is EventKind.ACCOUNT:
                event_cash_subunits = event.data.get("cash_subunits")
                event_position = event.data.get("position")
                event_fees_paid_subunits = event.data.get("fees_paid_subunits")
                if isinstance(event_position, int):
                    position = event_position
                    max_abs_position = max(max_abs_position, abs(position))
                if isinstance(event_cash_subunits, int):
                    cash_subunits = event_cash_subunits
                if isinstance(event_fees_paid_subunits, int):
                    fees_paid_subunits = event_fees_paid_subunits
                update_drawdown()
            elif event.kind is EventKind.SESSION:
                state = event.data.get("state")
                if state in ("liquidation_started", "liquidation_waiting"):
                    termination_state = SessionState.LIQUIDATING
                elif state == "terminated":
                    termination_state = SessionState.TERMINATED
                terminal_cash_subunits = event.data.get("terminal_cash_subunits")
                event_position = event.data.get("position")
                terminal_fees_paid_subunits = event.data.get("fees_paid_subunits")
                if isinstance(terminal_cash_subunits, int):
                    cash_subunits = terminal_cash_subunits
                if isinstance(terminal_fees_paid_subunits, int):
                    fees_paid_subunits = terminal_fees_paid_subunits
                if isinstance(event_position, int):
                    position = event_position
                    max_abs_position = max(max_abs_position, abs(position))
                update_drawdown()
            elif event.kind is EventKind.RISK and event.data.get("state") == "margin_call":
                margin_call_count += 1

        exact_cash = Fraction(cash_subunits, account.cash_subunits_per_tick)
        exact_fees_paid = Fraction(
            fees_paid_subunits,
            account.cash_subunits_per_tick,
        )
        cash = _number(exact_cash)
        exact_max_drawdown = max_drawdown_subunits / account.cash_subunits_per_tick
        strategy_diagnostics = session.strategy_diagnostics()
        advances = record.time_controller.advances
        is_focal_player = participant_id == record.focal_participant_id
        return EpisodeMetrics(
            episode_id=record.episode_id,
            participant_id=participant_id,
            market_time=session.now,
            terminal_event_sequence=exchange.event_log.last_sequence,
            starting_cash=account.starting_cash,
            cash=cash,
            cash_subunits=cash_subunits,
            cash_subunits_per_tick=account.cash_subunits_per_tick,
            position=position,
            pnl=_number(exact_cash - account.starting_cash),
            fees_paid=_number(exact_fees_paid),
            fees_paid_subunits=fees_paid_subunits,
            order_count=order_count,
            fill_count=fill_count,
            rejection_count=rejection_count,
            order_rejection_count=order_rejection_count,
            cancel_rejection_count=cancel_rejection_count,
            margin_rejection_count=margin_rejection_count,
            gross_filled_quantity=gross_filled_quantity,
            margin_call_count=margin_call_count,
            margin_liquidation_count=margin_liquidation_count,
            margin_liquidated_quantity=margin_liquidated_quantity,
            max_abs_position=max_abs_position,
            max_drawdown=_number(exact_max_drawdown),
            deployment_count=strategy_diagnostics["deployment_count"],
            unique_strategy_version_count=strategy_diagnostics["unique_strategy_version_count"],
            strategy_stop_count=strategy_diagnostics["strategy_stop_count"],
            strategy_fault_count=strategy_diagnostics["strategy_fault_count"],
            termination_state=termination_state,
            time_mode=record.time_controller.mode,
            voluntary_wait_ns=(
                sum(advance.duration for advance in advances if advance.reason == "voluntary_wait")
                if is_focal_player
                else 0
            ),
            charged_agent_time_ns=(
                sum(advance.duration for advance in advances if advance.reason in ("wall_time", "model_turn"))
                if is_focal_player
                else 0
            ),
            model_turn_count=(record.time_controller.committed_turn_count if is_focal_player else 0),
        )


def _number(value: Fraction) -> int | float:
    if value.denominator == 1:
        return value.numerator
    return float(value)


__all__ = [
    "EpisodeFinalized",
    "EpisodeMetrics",
    "EpisodeState",
]
