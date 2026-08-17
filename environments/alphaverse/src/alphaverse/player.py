"""Player-scoped application service over an Alphaverse episode."""

from __future__ import annotations

import os
import tempfile
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path

from alphaverse.clock import EventProcessingLimitExceeded
from alphaverse.episode import ActionReceipt, Episode
from alphaverse.exchange import TerminationResult
from alphaverse.models import Side
from alphaverse.profiles import ParticipantSpec
from alphaverse.strategy import (
    ActionBatch,
    CancelOrderAction,
    EmitAlert,
    InputEnvelope,
    InputKind,
    RequestStop,
    Strategy,
    SubmitLimitOrder,
)


class _PlayerInbox:
    """Append-only, disk-backed delivered-event history for one player.

    Capture cursors are one-based delivery positions, so the file deliberately
    stores every envelope (including private packets) in arrival order.  Only
    a bounded tail remains resident; old capture windows are reconstructed
    from canonical ``InputEnvelope`` JSON.
    """

    _CHECKPOINT_STRIDE = 1_024
    _DEFAULT_RETAINED_EVENTS = 2_048

    def __init__(self, *, retained_events: int = _DEFAULT_RETAINED_EVENTS) -> None:
        if isinstance(retained_events, bool) or not isinstance(retained_events, int):
            raise TypeError("retained_events must be an int")
        if retained_events < 0:
            raise ValueError("retained_events must be non-negative")
        descriptor, name = tempfile.mkstemp(prefix="alphaverse-player-feed-", suffix=".jsonl")
        self._path = Path(name)
        self._stream = os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            buffering=64 * 1024,
        )
        self._count = 0
        self._retained_events = retained_events
        self._recent: deque[InputEnvelope] = deque(maxlen=retained_events or None)
        self._recent_start = 0
        self._checkpoints: list[int] = [0]
        self._closed = False

    def __len__(self) -> int:
        return self._count

    def append(self, envelope: InputEnvelope) -> None:
        self._require_open()
        if self._count and self._count % self._CHECKPOINT_STRIDE == 0:
            self._stream.flush()
            self._checkpoints.append(self._stream.tell())
        self._stream.write(envelope.to_json())
        self._stream.write("\n")
        self._count += 1
        if self._retained_events:
            self._recent.append(envelope)
            self._recent_start = self._count - len(self._recent)

    def window(self, start: int, stop: int) -> tuple[InputEnvelope, ...]:
        """Return the half-open cursor range, reconstructing old packets."""

        self._require_open()
        if start >= stop:
            return ()
        return tuple(envelope for _, envelope in islice(self.iter_from(start), stop - start))

    def iter_from(self, start: int) -> Iterator[tuple[int, InputEnvelope]]:
        """Yield zero-based inbox positions from ``start`` without buffering history."""

        self._require_open()
        if start >= self._count:
            return iter(())
        if self._retained_events and start >= self._recent_start:
            return iter(
                enumerate(
                    tuple(self._recent)[start - self._recent_start :],
                    start=start,
                )
            )
        self._stream.flush()
        checkpoint_index = start // self._CHECKPOINT_STRIDE
        checkpoint_event = checkpoint_index * self._CHECKPOINT_STRIDE
        offset = self._checkpoints[checkpoint_index]

        def read() -> Iterator[tuple[int, InputEnvelope]]:
            with self._path.open(encoding="utf-8") as stream:
                stream.seek(offset)
                for event_index, line in enumerate(stream, start=checkpoint_event):
                    if event_index >= start:
                        yield event_index, InputEnvelope.from_json(line)

        return read()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.close()
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("player inbox is closed")


class _PlayerStrategy(Strategy):
    """No-op actor giving direct API actions the same account and event path."""

    def __init__(self) -> None:
        self.inbox = _PlayerInbox()
        self.last_levels: InputEnvelope | None = None
        self.last_alert_cursor = 0
        self._next_alert_number = 1

    def _capture(self, event: InputEnvelope) -> None:
        self.inbox.append(event)

    def capture_alert(self, source: InputEnvelope, alert: EmitAlert) -> int:
        """Persist a strategy alert without scheduling it back to the strategy."""

        event_number = self._next_alert_number
        self._next_alert_number += 1
        self.inbox.append(
            InputEnvelope(
                session_id=source.session_id,
                strategy_instance_id=source.strategy_instance_id,
                event_id=f"A{event_number}",
                kind=InputKind.ALERT,
                exchange_time=source.exchange_time,
                available_at=source.available_at,
                source_event_seq=source.source_event_seq,
                payload={
                    "event_kind": "strategy_alert",
                    "code": alert.code,
                    "message": alert.message,
                    "data": dict(alert.data),
                },
            )
        )
        self.last_alert_cursor = len(self.inbox)
        return self.last_alert_cursor

    def on_market(self, ctx, event: InputEnvelope):
        self._capture(event)

    def on_levels(self, ctx, event: InputEnvelope):
        self.last_levels = event
        self._capture(event)

    def on_execution(self, ctx, event: InputEnvelope):
        self._capture(event)

    def on_risk(self, ctx, event: InputEnvelope):
        self._capture(event)

    def on_stop(self, ctx, event: InputEnvelope):
        self._capture(event)

    def on_signal(self, ctx, event: InputEnvelope):
        self._capture(event)


class _CompositePlayerStrategy(Strategy):
    """Keep the player inbox alive while forwarding events to an automation."""

    def __init__(self, observer: _PlayerStrategy, automation: Strategy) -> None:
        self.observer = observer
        self.automation: Strategy | None = automation
        self.fault: str | None = None
        self.stop_reason: str | None = None

    def _forward(self, method: str, ctx, event: InputEnvelope):
        getattr(self.observer, method)(ctx, event)
        if self.automation is None:
            return None
        try:
            result = getattr(self.automation, method)(ctx, event)
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            self.close()
            return None
        if result is None:
            return None
        batch = result if isinstance(result, ActionBatch) else ActionBatch(result)
        retained = []
        for action in batch:
            if isinstance(action, RequestStop):
                self.stop_reason = action.reason
                close = getattr(self.automation, "close", None)
                if close is not None:
                    close()
                self.automation = None
            elif isinstance(action, EmitAlert):
                self.observer.capture_alert(event, action)
            else:
                retained.append(action)
        return retained

    def on_start(self, ctx, event):
        return self._forward("on_start", ctx, event)

    def on_market(self, ctx, event):
        return self._forward("on_market", ctx, event)

    def on_levels(self, ctx, event):
        return self._forward("on_levels", ctx, event)

    def on_execution(self, ctx, event):
        return self._forward("on_execution", ctx, event)

    def on_timer(self, ctx, event):
        return self._forward("on_timer", ctx, event)

    def on_risk(self, ctx, event):
        return self._forward("on_risk", ctx, event)

    def on_stop(self, ctx, event):
        return self._forward("on_stop", ctx, event)

    def on_signal(self, ctx, event):
        return self._forward("on_signal", ctx, event)

    def close(self) -> None:
        if self.automation is not None:
            close = getattr(self.automation, "close", None)
            if close is not None:
                close()
            self.automation = None


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    market_time: int
    available_at: int
    source_event_seq: int
    bids: tuple[dict[str, int], ...]
    asks: tuple[dict[str, int], ...]


@dataclass(frozen=True, slots=True)
class PlayerFeedEvent:
    cursor: int
    envelope: InputEnvelope


@dataclass(frozen=True, slots=True)
class WaitResult:
    """Result of an alert-interruptible wait operation."""

    market_time: int
    target_market_time: int
    interrupted_by_alert: bool
    alert_cursor: int | None = None


@dataclass(slots=True)
class StrategyDeploymentRecord:
    """Server-side audit record for one successfully launched source artifact."""

    ordinal: int
    version_id: str
    entrypoint: str
    source: str
    deployed_at: int
    ended_at: int | None = None
    outcome: str = "active"
    fault: str | None = None
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StagedStrategyDeployment:
    """Validated source waiting for the next market-session reopen."""

    version_id: str
    entrypoint: str
    source: str
    staged_at: int


class PlayerSession:
    """Capabilities visible to one focal player, with private data filtering."""

    def __init__(
        self,
        episode: Episode,
        focal_spec: ParticipantSpec,
        *,
        simulation_step_ns: int = 5_000_000_000,
        max_scheduled_events_per_step: int = 250_000,
    ) -> None:
        if simulation_step_ns <= 0:
            raise ValueError("simulation_step_ns must be positive")
        if max_scheduled_events_per_step <= 0:
            raise ValueError("max_scheduled_events_per_step must be positive")
        self.episode = episode
        self.participant_id = focal_spec.participant_id
        self.spec = focal_spec
        self._simulation_step_ns = simulation_step_ns
        self._max_scheduled_events_per_step = max_scheduled_events_per_step
        self._strategy = _PlayerStrategy()
        self._automation: _CompositePlayerStrategy | None = None
        self._strategy_version_id: str | None = None
        self._last_strategy_fault: str | None = None
        self._deployments: list[StrategyDeploymentRecord] = []
        self._staged_deployment: StagedStrategyDeployment | None = None
        self._strategy_stop_count = 0
        self._acknowledged_alert_cursor = 0
        self.strategy_instance_id = episode.add_strategy(focal_spec, self._strategy)
        self._seed_recovery_snapshot()
        episode.run_until(episode.now)

    def _seed_recovery_snapshot(self) -> None:
        """Seed the queryable level snapshot without inventing a feed event."""

        levels = self.episode.exchange.book.top_k(self.episode.exchange.level_depth)
        available_at = self.now + (self.spec.technology.market_data_latency + self.spec.technology.level_feed_latency)

        def wire(selected):
            return tuple(
                {
                    "price": level.price,
                    "quantity": level.total_quantity,
                    "order_count": level.order_count,
                }
                for level in selected
            )

        sequence = self.episode.exchange.event_log.last_sequence
        self._strategy.last_levels = InputEnvelope(
            session_id=self.episode.session_id,
            strategy_instance_id=self.strategy_instance_id,
            event_id=f"snapshot-{sequence}",
            kind=InputKind.LEVELS,
            exchange_time=self.now,
            available_at=available_at,
            source_event_seq=sequence,
            payload={
                "event_kind": "levels",
                "product_id": self.episode.exchange.product.product_id,
                "depth": self.episode.exchange.level_depth,
                "through_event_seq": sequence,
                "bids": wire(levels.bids),
                "asks": wire(levels.asks),
                "event_end": True,
                "recovery_snapshot": True,
            },
        )

    @property
    def now(self) -> int:
        return self.episode.now

    def submit_limit(
        self,
        *,
        client_order_id: str,
        side: Side,
        price: int,
        quantity: int,
    ) -> ActionReceipt:
        return self.episode.enqueue_player_action(
            self.participant_id,
            SubmitLimitOrder(
                client_order_id=client_order_id,
                side=side,
                price=price,
                quantity=quantity,
                product_id=self.episode.exchange.product.product_id,
            ),
        )

    def cancel(self, order_id: str) -> ActionReceipt:
        return self.episode.enqueue_player_action(
            self.participant_id,
            CancelOrderAction(order_id, self.episode.exchange.product.product_id),
        )

    def wait(
        self,
        *,
        duration: int | None = None,
        until: int | None = None,
        interrupt_on_alert: bool = False,
    ) -> int | WaitResult:
        """Advance virtual time, optionally stopping after a new strategy alert.

        The default return value and time progression remain unchanged.  With
        ``interrupt_on_alert=True``, each bounded simulation step checks for
        an alert not yet acknowledged through :meth:`acknowledge_alerts`.
        """

        if (duration is None) == (until is None):
            raise ValueError("provide exactly one of duration or until")
        if not isinstance(interrupt_on_alert, bool):
            raise TypeError("interrupt_on_alert must be a bool")
        if duration is not None:
            if isinstance(duration, bool) or not isinstance(duration, int):
                raise TypeError("duration must be an int")
            if duration < 0:
                raise ValueError("duration must be non-negative")
            target = self.now + duration
        else:
            if isinstance(until, bool) or not isinstance(until, int):
                raise TypeError("until must be an int")
            target = until
        result = self._run_bounded(target, interrupt_on_alert=interrupt_on_alert)
        return result if interrupt_on_alert else self.now

    def _run_bounded(
        self,
        target: int,
        *,
        interrupt_on_alert: bool = False,
    ) -> WaitResult:
        """Advance in bounded slices and remove automation that creates a storm."""

        if interrupt_on_alert and self._has_unacknowledged_alert():
            return WaitResult(
                market_time=self.now,
                target_market_time=target,
                interrupted_by_alert=True,
                alert_cursor=self._strategy.last_alert_cursor,
            )
        first = True
        recovery_batches = 0
        while first or self.now < target:
            first = False
            step_target = min(target, self.now + self._simulation_step_ns)
            try:
                self.episode.run_until(
                    step_target,
                    max_events=self._max_scheduled_events_per_step,
                )
                recovery_batches = 0
                if interrupt_on_alert and self._has_unacknowledged_alert():
                    return WaitResult(
                        market_time=self.now,
                        target_market_time=target,
                        interrupted_by_alert=True,
                        alert_cursor=self._strategy.last_alert_cursor,
                    )
            except EventProcessingLimitExceeded as exc:
                if self._automation is not None:
                    self._fault_automation(str(exc))
                    recovery_batches = 0
                    continue
                recovery_batches += 1
                if recovery_batches >= 10:
                    raise RuntimeError(
                        "episode remained above its scheduled-event budget after focal automation was removed"
                    ) from exc
        return WaitResult(
            market_time=self.now,
            target_market_time=target,
            interrupted_by_alert=False,
        )

    def account(self) -> dict[str, object]:
        account = self.episode.exchange.clearing.snapshot(self.participant_id)
        result: dict[str, object] = {
            "participant_id": self.participant_id,
            "starting_cash": account.starting_cash,
            "cash": account.cash,
            "cash_subunits": account.cash_subunits,
            "fees_paid": account.fees_paid,
            "fees_paid_subunits": account.fees_paid_subunits,
            "cash_subunits_per_tick": account.cash_subunits_per_tick,
            "position": account.position,
            "session_state": self.episode.exchange.state(self.participant_id).value,
        }
        margin = self.episode.risk_snapshot(self.participant_id)
        result["margin"] = (
            None
            if margin is None
            else {
                "state": margin.state.value,
                "mark": _number(margin.mark),
                "mark_source": margin.mark_source,
                "mark_time": margin.mark_time,
                "marked_equity": _number(margin.marked_equity),
                "initial_margin_required": margin.initial_requirement,
                "maintenance_margin_required": margin.maintenance_requirement,
                "available_initial_margin": _number(margin.available_margin),
                "reduce_only": margin.reduce_only,
                "liquidation_deadline": margin.liquidation_deadline,
            }
        )
        return result

    def product(self) -> dict[str, object]:
        product = self.episode.exchange.product
        risk = self.spec.risk
        technology = self.spec.technology
        return {
            "product_id": product.product_id,
            "tick_value": product.tick_value,
            "contract_multiplier": product.contract_multiplier,
            "cash_subunits_per_tick": product.cash_subunits_per_tick,
            "transaction_fee_per_contract_subunits": (product.transaction_fee_per_contract_subunits),
            "transaction_fee_per_contract": (
                product.transaction_fee_per_contract_subunits / product.cash_subunits_per_tick
            ),
            "risk_limits": {
                "max_abs_position": risk.max_abs_position,
                "max_order_quantity": risk.max_order_quantity,
                "max_live_orders": risk.max_live_orders,
                "max_actions_per_callback": risk.max_actions_per_callback,
            },
            "margin": (
                None
                if self.spec.margin is None
                else {
                    "initial_margin_per_contract": (self.spec.margin.initial_margin_per_contract),
                    "maintenance_margin_per_contract": (self.spec.margin.maintenance_margin_per_contract),
                    "liquidation_grace_period_ns": self.spec.margin.grace_period,
                    "risk_mark": "latest external two-sided midpoint",
                }
            ),
            "technology": {
                "market_data_latency_ns": technology.market_data_latency,
                "level_feed_latency_ns": technology.level_feed_latency,
                "decision_latency_ns": technology.decision_latency,
                "order_entry_latency_ns": technology.order_entry_latency,
                "mbo_entitled": technology.mbo_entitled,
                "strategy_callback_timeout_ns": technology.callback_timeout_ns,
                "strategy_callback_memory_limit_bytes": (technology.callback_memory_limit_bytes),
            },
            "simulation_limits": {
                "step_ns": self._simulation_step_ns,
                "max_scheduled_events_per_step": (self._max_scheduled_events_per_step),
            },
        }

    def open_orders(self) -> tuple[dict[str, int | str], ...]:
        return tuple(
            {
                "order_id": order.order_id,
                "client_order_id": order.client_order_id,
                "side": order.side.name.lower(),
                "price": order.price,
                "remaining_quantity": order.remaining_quantity,
                "priority": order.priority,
            }
            for order in self.episode.exchange.book.orders_for_participant(self.participant_id)
        )

    def market_snapshot(self, *, depth: int = 10) -> MarketSnapshot:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        delivered = self._strategy.last_levels
        if delivered is None or delivered.available_at > self.now:
            return MarketSnapshot(self.now, self.now, 0, (), ())

        def select(side: str) -> tuple[dict[str, int], ...]:
            raw = delivered.payload.get(side, ())
            if not isinstance(raw, tuple):
                return ()
            selected: list[dict[str, int]] = []
            for level in raw[:depth]:
                if not isinstance(level, dict) and not hasattr(level, "get"):
                    continue
                price = level.get("price")
                quantity = level.get("quantity")
                order_count = level.get("order_count")
                if all(isinstance(value, int) for value in (price, quantity, order_count)):
                    selected.append(
                        {
                            "price": price,
                            "quantity": quantity,
                            "order_count": order_count,
                        }
                    )
            return tuple(selected)

        return MarketSnapshot(
            market_time=delivered.exchange_time,
            available_at=delivered.available_at,
            source_event_seq=delivered.source_event_seq,
            bids=select("bids"),
            asks=select("asks"),
        )

    def events(self, *, after_cursor: int = 0, limit: int = 1_000) -> tuple[PlayerFeedEvent, ...]:
        if after_cursor < 0:
            raise ValueError("after_cursor must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        start = min(after_cursor, len(self._strategy.inbox))
        envelopes = self._strategy.inbox.window(start, start + limit)
        return tuple(
            PlayerFeedEvent(cursor=index + 1, envelope=envelope)
            for index, envelope in enumerate(envelopes, start=start)
        )

    def capture_window(
        self,
        *,
        after_cursor: int = 0,
    ) -> tuple[tuple[PlayerFeedEvent, ...], int]:
        """Freeze every delivered event after an exclusive capture cursor."""

        if isinstance(after_cursor, bool) or not isinstance(after_cursor, int):
            raise TypeError("after_cursor must be an int")
        if after_cursor < 0:
            raise ValueError("after_cursor must be non-negative")
        next_after_cursor = len(self._strategy.inbox)
        if after_cursor > next_after_cursor:
            raise ValueError("after_cursor is beyond the latest delivered event cursor")
        return (
            tuple(
                PlayerFeedEvent(cursor=index + 1, envelope=envelope)
                for index, envelope in enumerate(
                    self._strategy.inbox.window(after_cursor, next_after_cursor),
                    start=after_cursor,
                )
            ),
            next_after_cursor,
        )

    def capture_page(
        self,
        *,
        after_cursor: int = 0,
        through_cursor: int | None = None,
        limit: int = 10_000,
    ) -> tuple[tuple[PlayerFeedEvent, ...], int, int]:
        """Read one bounded page from a frozen delivered-event window.

        ``snapshot_end_cursor`` is fixed by the first page and supplied as
        ``through_cursor`` on later pages. This lets file-oriented clients copy
        a coherent capture without materializing the full inbox in either the
        Toolset or agent process.
        """

        if isinstance(after_cursor, bool) or not isinstance(after_cursor, int):
            raise TypeError("after_cursor must be an int")
        if after_cursor < 0:
            raise ValueError("after_cursor must be non-negative")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an int")
        if not 1 <= limit <= 20_000:
            raise ValueError("limit must be between 1 and 20000")
        latest_cursor = len(self._strategy.inbox)
        snapshot_end_cursor = latest_cursor if through_cursor is None else through_cursor
        if isinstance(snapshot_end_cursor, bool) or not isinstance(snapshot_end_cursor, int):
            raise TypeError("through_cursor must be an int or null")
        if not 0 <= after_cursor <= snapshot_end_cursor <= latest_cursor:
            raise ValueError("capture cursors must satisfy 0 <= after <= through <= latest")
        next_after_cursor = min(snapshot_end_cursor, after_cursor + limit)
        envelopes = self._strategy.inbox.window(after_cursor, next_after_cursor)
        return (
            tuple(
                PlayerFeedEvent(cursor=index + 1, envelope=envelope)
                for index, envelope in enumerate(
                    envelopes,
                    start=after_cursor,
                )
            ),
            next_after_cursor,
            snapshot_end_cursor,
        )

    def alerts(
        self,
        *,
        after_cursor: int = 0,
        limit: int = 1_000,
    ) -> tuple[PlayerFeedEvent, ...]:
        """Return durable strategy alerts after an exclusive inbox cursor."""

        if isinstance(after_cursor, bool) or not isinstance(after_cursor, int):
            raise TypeError("after_cursor must be an int")
        if after_cursor < 0:
            raise ValueError("after_cursor must be non-negative")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an int")
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected: list[PlayerFeedEvent] = []
        for index, envelope in self._strategy.inbox.iter_from(after_cursor):
            if envelope.kind is InputKind.ALERT and envelope.payload.get("event_kind") == "strategy_alert":
                selected.append(PlayerFeedEvent(cursor=index + 1, envelope=envelope))
                if len(selected) == limit:
                    break
        return tuple(selected)

    def alert_status(self) -> dict[str, int | bool]:
        """Expose the alert cursors needed by a hosted API integration."""

        latest = self._strategy.last_alert_cursor
        return {
            "latest_alert_cursor": latest,
            "acknowledged_alert_cursor": self._acknowledged_alert_cursor,
            "has_unacknowledged_alert": latest > self._acknowledged_alert_cursor,
        }

    def acknowledge_alerts(self, through_cursor: int) -> int:
        """Acknowledge every alert at or before an inclusive inbox cursor."""

        if isinstance(through_cursor, bool) or not isinstance(through_cursor, int):
            raise TypeError("through_cursor must be an int")
        if through_cursor < 0:
            raise ValueError("through_cursor must be non-negative")
        latest_inbox_cursor = len(self._strategy.inbox)
        if through_cursor > latest_inbox_cursor:
            raise ValueError("through_cursor is beyond the latest delivered event")
        self._acknowledged_alert_cursor = max(
            self._acknowledged_alert_cursor,
            min(through_cursor, self._strategy.last_alert_cursor),
        )
        return self._acknowledged_alert_cursor

    def _has_unacknowledged_alert(self) -> bool:
        return self._strategy.last_alert_cursor > self._acknowledged_alert_cursor

    def close_history(self, *, close_exchange_history: bool = True) -> None:
        """Release retained capture data after the host has deleted an episode."""

        self._strategy.inbox.close()
        if close_exchange_history:
            self.episode.exchange.event_log.close()

    def terminate(self) -> TerminationResult:
        self._staged_deployment = None
        self._finish_current_deployment("terminated")
        if self._automation is not None:
            self._automation.close()
            self._automation = None
        return self.episode.terminate_participant(self.participant_id)

    def _install_strategy(self, strategy: Strategy) -> _CompositePlayerStrategy:
        self._finish_current_deployment("replaced")
        if self._automation is not None:
            self._automation.close()
        self._last_strategy_fault = None
        composite = _CompositePlayerStrategy(self._strategy, strategy)
        self._automation = composite
        self._strategy_version_id = None
        self.episode.replace_strategy(
            self.strategy_instance_id,
            composite,
        )
        return composite

    def deploy_strategy(self, strategy: Strategy) -> None:
        """Install a trusted in-process strategy without a source audit artifact."""

        self._install_strategy(strategy)

    def deploy_source(
        self,
        source: str,
        *,
        entrypoint: str = "strategy:StrategyImpl",
    ) -> str:
        from alphaverse.strategy.subprocess import StrategyArtifact

        artifact = StrategyArtifact.build(source, entrypoint=entrypoint)
        return self._deploy_artifact(artifact)

    def deploy_trusted_source(
        self,
        source: str,
        *,
        entrypoint: str = "strategy:StrategyImpl",
    ) -> str:
        """Install evaluator-owned source that may use private strategy classes."""

        from alphaverse.strategy.subprocess import StrategyArtifact

        artifact = StrategyArtifact.build_trusted(source, entrypoint=entrypoint)
        return self._deploy_artifact(artifact)

    def _deploy_artifact(self, artifact) -> str:
        strategy = self._subprocess_strategy(artifact)
        self._install_strategy(strategy)
        self._strategy_version_id = artifact.version_id
        self._deployments.append(
            StrategyDeploymentRecord(
                ordinal=len(self._deployments) + 1,
                version_id=artifact.version_id,
                entrypoint=artifact.entrypoint,
                source=artifact.source,
                deployed_at=self.now,
            )
        )
        return artifact.version_id

    def stage_source(
        self,
        source: str,
        *,
        entrypoint: str = "strategy:StrategyImpl",
    ) -> str:
        """Validate a source artifact without changing the live strategy."""

        from alphaverse.strategy.subprocess import StrategyArtifact

        artifact = StrategyArtifact.build(source, entrypoint=entrypoint)
        # Building an artifact only validates syntax and the entrypoint name.
        # Start the isolated worker as well so constructor/import failures are
        # reported during the deployment turn, not at the atomic market reopen.
        candidate = self._subprocess_strategy(artifact)
        candidate.close()
        self._staged_deployment = StagedStrategyDeployment(
            version_id=artifact.version_id,
            entrypoint=artifact.entrypoint,
            source=artifact.source,
            staged_at=self.now,
        )
        return artifact.version_id

    def _subprocess_strategy(self, artifact):
        from alphaverse.strategy.subprocess import SubprocessStrategy

        return SubprocessStrategy(
            artifact,
            participant_id=self.participant_id,
            strategy_instance_id=self.strategy_instance_id,
            product_id=self.episode.exchange.product.product_id,
            parameters=dict(self.spec.parameters),
            seed=self.spec.seed,
            max_actions_per_callback=self.spec.risk.max_actions_per_callback,
            callback_timeout_ns=self.spec.technology.callback_timeout_ns,
            memory_limit_bytes=self.spec.technology.callback_memory_limit_bytes,
        )

    def discard_staged_source(self, fault: str | None = None) -> None:
        """Leave the incumbent live after a staged activation failure."""

        self._staged_deployment = None
        if fault:
            self._last_strategy_fault = fault

    def activate_staged_source(self) -> str | None:
        """Promote the last validated artifact at a market reopen boundary."""

        staged = self._staged_deployment
        if staged is None:
            return None
        version_id = self.deploy_source(
            staged.source,
            entrypoint=staged.entrypoint,
        )
        if version_id != staged.version_id:  # pragma: no cover - content invariant
            raise RuntimeError("staged strategy version changed during activation")
        self._staged_deployment = None
        return version_id

    def stop_strategy(self) -> None:
        if self._automation is not None:
            self._strategy_stop_count += 1
        self._finish_current_deployment("stopped")
        if self._automation is not None:
            self._automation.close()
            self._automation = None
        self.episode.replace_strategy(self.strategy_instance_id, self._strategy)
        self._strategy_version_id = None

    def _fault_automation(self, fault: str) -> None:
        """Remove a focal automation that exceeds deterministic simulation work."""

        self._last_strategy_fault = fault
        if self._deployments:
            record = self._deployments[-1]
            if record.ended_at is None:
                record.fault = fault
                record.outcome = "faulted"
                record.ended_at = self.now
        if self._automation is not None:
            self._automation.close()
            self._automation = None
        self.episode.replace_strategy(self.strategy_instance_id, self._strategy)
        self._strategy_version_id = None

    def _finish_current_deployment(self, outcome: str) -> None:
        if not self._deployments:
            return
        record = self._deployments[-1]
        if record.ended_at is not None:
            return
        automation = self._automation
        if automation is not None:
            record.fault = automation.fault
            record.stop_reason = automation.stop_reason
        if record.fault is not None:
            record.outcome = "faulted"
        elif record.stop_reason is not None:
            record.outcome = "self_stopped"
        else:
            record.outcome = outcome
        record.ended_at = self.now

    def _sync_current_deployment(self) -> None:
        if not self._deployments:
            return
        record = self._deployments[-1]
        if record.ended_at is not None:
            return
        automation = self._automation
        if automation is None:
            return
        record.fault = automation.fault
        record.stop_reason = automation.stop_reason
        if record.fault is not None:
            record.outcome = "faulted"
            record.ended_at = self.now
        elif record.stop_reason is not None:
            record.outcome = "self_stopped"
            record.ended_at = self.now

    def deployment_records(self) -> tuple[StrategyDeploymentRecord, ...]:
        """Return successful source deployments for trusted evaluation inspection."""

        self._sync_current_deployment()
        return tuple(self._deployments)

    def strategy_diagnostics(self) -> dict[str, int]:
        """Return aggregate strategy lifecycle counters for terminal metrics."""

        records = self.deployment_records()
        return {
            "deployment_count": len(records),
            "unique_strategy_version_count": len({record.version_id for record in records}),
            "strategy_stop_count": self._strategy_stop_count,
            "strategy_fault_count": sum(record.fault is not None for record in records),
        }

    def strategy_status(self) -> dict[str, object]:
        self._sync_current_deployment()
        active = self._automation is not None and self._automation.automation is not None
        return {
            "strategy_instance_id": self.strategy_instance_id,
            "active": active,
            "strategy_version_id": self._strategy_version_id,
            "fault": (self._last_strategy_fault if self._automation is None else self._automation.fault),
            "stop_reason": (None if self._automation is None else self._automation.stop_reason),
            "staged_strategy_version_id": (
                None if self._staged_deployment is None else self._staged_deployment.version_id
            ),
            "staged_at": (None if self._staged_deployment is None else self._staged_deployment.staged_at),
        }


def _number(value: Fraction | None) -> int | float | None:
    if value is None:
        return None
    if value.denominator == 1:
        return value.numerator
    return float(value)


__all__ = [
    "MarketSnapshot",
    "PlayerFeedEvent",
    "PlayerSession",
    "StagedStrategyDeployment",
    "StrategyDeploymentRecord",
    "WaitResult",
]
