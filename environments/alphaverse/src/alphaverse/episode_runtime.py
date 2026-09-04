"""Transport-independent ownership and operations for one evaluation episode."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from alphaverse.artifacts import EpisodeArtifactWriter, archive_bundle, load_manifest
from alphaverse.capture import select_market_capture_events
from alphaverse.episode_state import EpisodeMetrics, EpisodeState
from alphaverse.models import Side
from alphaverse.opponent_roster import (
    legacy_prop_roster_id,
    opponent_roster,
)
from alphaverse.player import PlayerSession, WaitResult
from alphaverse.profiles import MarginConfig, ParticipantSpec, TechnologyProfile
from alphaverse.prop_trader import prop_baseline_source
from alphaverse.scenario import SECOND, create_populated_scenario, scenario_config_for_version
from alphaverse.time_accounting import TimeMode
from alphaverse.world import LatentDemandProfile


@dataclass(frozen=True, slots=True)
class EpisodeRuntimeConfig:
    episode_id: str
    scenario_seed: int
    scenario_version: str
    latent_demand_profile: LatentDemandProfile = LatentDemandProfile.BALANCED
    starting_cash: int = 1_000_000
    max_market_time_ns: int | None = None
    time_mode: Literal["manual", "wall"] = "wall"
    wall_time_scale: float = 1.0
    wall_quantum_ns: int = 1_000_000
    initial_margin_per_contract: int = 5_000
    maintenance_margin_per_contract: int = 4_000
    margin_liquidation_grace_ns: int = 30 * SECOND
    adaptive_prop: bool = False
    prop_seed_profile: Literal["passive", "competitive"] = "passive"
    prop_control_scope: Literal["full_source", "knobs"] = "full_source"
    opponent_roster_id: str | None = None
    session_duration_ns: int | None = None
    artifact_root: str | Path = "/logs/artifacts"
    inline_artifact_max_bytes: int = 24 * 1024 * 1024
    artifact_transport: Literal["auto", "inline", "stream"] = "auto"
    artifact_export_chunk_bytes: int = 4 * 1024 * 1024


def event_record(cursor: int, event: Any) -> dict[str, object]:
    """Serialize one player-delivered envelope without leaking Python types."""

    wire_payload = json.loads(event.to_json())["payload"]
    return {
        "cursor": cursor,
        "event_id": event.event_id,
        "kind": event.kind.value,
        "exchange_time": event.exchange_time,
        "available_at": event.available_at,
        "source_event_seq": event.source_event_seq,
        "payload": wire_payload,
    }


def wait_response(session: PlayerSession, result: int | WaitResult) -> dict[str, object]:
    """Serialize a wait and acknowledge only alerts returned by that wait."""

    if isinstance(result, int):
        return {
            "market_time": result,
            "requested_until": result,
            "woke_on_alert": False,
            "alerts": [],
        }
    status = session.alert_status()
    after_cursor = int(status["acknowledged_alert_cursor"])
    alerts = session.alerts(after_cursor=after_cursor, limit=20)
    records = [event_record(item.cursor, item.envelope) for item in alerts]
    if records:
        session.acknowledge_alerts(int(records[-1]["cursor"]))
    return {
        "market_time": result.market_time,
        "requested_until": result.target_market_time,
        "woke_on_alert": result.interrupted_by_alert,
        "alerts": records,
        "next_alert_cursor": (int(records[-1]["cursor"]) if records else after_cursor),
    }


class EpisodeRuntime:
    """The authoritative in-process market behind an evaluator Toolset."""

    def __init__(self, config: EpisodeRuntimeConfig) -> None:
        self.config = config
        self.latent_demand_profile = LatentDemandProfile(config.latent_demand_profile)
        # Scenario construction can schedule latent demand processes beyond a short
        # evaluation cap. Keep the latent world valid; episode state independently
        # enforces the player's smaller market-time horizon.
        default_active_duration = 5 * 60 * SECOND
        active_duration = default_active_duration
        if config.max_market_time_ns is not None:
            active_duration = max(
                default_active_duration,
                config.max_market_time_ns - 2 * SECOND,
            )
        self.scenario = create_populated_scenario(
            scenario_config_for_version(
                config.scenario_version,
                seed=config.scenario_seed,
                session_id=f"eval-{config.scenario_seed}",
                active_duration=active_duration,
                latent_demand_profile=self.latent_demand_profile,
            )
        )
        episode = self.scenario.episode
        margin = MarginConfig(
            initial_margin_per_contract=config.initial_margin_per_contract,
            maintenance_margin_per_contract=config.maintenance_margin_per_contract,
            grace_period=config.margin_liquidation_grace_ns,
        )
        spec = ParticipantSpec(
            participant_id="player",
            strategy_version_id="toolset-player:v1",
            account_starting_cash=config.starting_cash,
            technology=TechnologyProfile(),
            margin=margin,
            seed=config.scenario_seed,
        )
        self.state = EpisodeState(
            config.episode_id,
            spec,
            episode=episode,
            max_market_time=config.max_market_time_ns,
            time_mode=TimeMode(config.time_mode),
            wall_time_scale=config.wall_time_scale,
            wall_quantum_ns=config.wall_quantum_ns,
            session_duration_ns=config.session_duration_ns,
        )

        roster_id = config.opponent_roster_id or legacy_prop_roster_id(
            adaptive_prop=config.adaptive_prop,
            seed_profile=config.prop_seed_profile,
            control_scope=config.prop_control_scope,
        )
        self.opponent_roster = opponent_roster(roster_id)
        for slot in self.opponent_roster.slots:
            self.state.add_participant(
                ParticipantSpec(
                    participant_id=slot.participant_id,
                    strategy_version_id=(f"opponent-roster:{roster_id}:{slot.participant_id}"),
                    account_starting_cash=config.starting_cash,
                    technology=TechnologyProfile(),
                    margin=margin,
                    seed=config.scenario_seed + slot.seed_offset,
                ),
                baseline_source=prop_baseline_source(slot.seed_profile),
            )
        self._artifact_directory: Path | None = None
        self._artifact_manifest: dict[str, object] | None = None
        self._replay_projection: dict[str, object] | None = None
        self._artifact_transport: dict[str, object] | None = None

    @property
    def episode_id(self) -> str:
        return self.config.episode_id

    def _observe(self, operation, *, participant_id: str = "player"):
        return self.state.observe_session(participant_id, operation)

    def _use(self, operation, *, participant_id: str = "player"):
        return self.state.with_session(participant_id, operation)

    def market_snapshot(self, depth: int = 10, *, participant_id: str = "player") -> dict[str, object]:
        snapshot = self._use(
            lambda session: session.market_snapshot(depth=depth),
            participant_id=participant_id,
        )
        return asdict(snapshot)

    def events(
        self,
        after_cursor: int,
        limit: int,
        *,
        participant_id: str = "player",
    ) -> dict[str, object]:
        items = self._use(
            lambda session: session.events(after_cursor=after_cursor, limit=limit),
            participant_id=participant_id,
        )
        records = [event_record(item.cursor, item.envelope) for item in items]
        return {
            "events": records,
            "next_after_cursor": records[-1]["cursor"] if records else after_cursor,
        }

    def account(self, *, participant_id: str = "player") -> dict[str, object]:
        return self._use(lambda session: session.account(), participant_id=participant_id)

    def product(self, *, participant_id: str = "player") -> dict[str, object]:
        return self._use(lambda session: session.product(), participant_id=participant_id)

    def open_orders(self, *, participant_id: str = "player") -> dict[str, object]:
        return {
            "orders": self._use(
                lambda session: session.open_orders(),
                participant_id=participant_id,
            )
        }

    def submit_limit_order(
        self,
        *,
        client_order_id: str,
        side: str,
        price: int,
        quantity: int,
    ) -> dict[str, object]:
        self.state.require_market_open()
        receipt = self._use(
            lambda session: session.submit_limit(
                client_order_id=client_order_id,
                side=Side.BUY if side == "buy" else Side.SELL,
                price=price,
                quantity=quantity,
            )
        )
        return asdict(receipt)

    def cancel_order(self, order_id: str) -> dict[str, object]:
        self.state.require_market_open()
        return asdict(self._use(lambda session: session.cancel(order_id)))

    def wait(
        self,
        *,
        duration_ns: int | None,
        until_ns: int | None,
        wake_on_alert: bool,
    ) -> dict[str, object]:
        if (duration_ns is None) == (until_ns is None):
            raise ValueError("provide exactly one of duration_ns or until_ns")
        if duration_ns is not None:
            result = self.state.run_for(
                duration_ns,
                interrupt_on_alert=wake_on_alert,
            )
        else:
            assert until_ns is not None
            result = self.state.run_until(
                until_ns,
                interrupt_on_alert=wake_on_alert,
            )
        response = self._observe(lambda session: wait_response(session, result))
        market_session = asdict(self.state.market_session_info())
        market_session["state"] = market_session["state"].value
        response["market_session"] = market_session
        response["market_time_ns"] = market_session["market_time_ns"]
        return response

    def deploy_strategy(
        self,
        source: str,
        entrypoint: str,
        *,
        participant_id: str = "player",
    ) -> dict[str, object]:
        version_id, staged, status = self.state.deploy_source(
            participant_id,
            source,
            entrypoint=entrypoint,
        )
        return {
            **status,
            "strategy_version_id": version_id,
            "staged": staged,
        }

    def strategy_status(self, *, participant_id: str = "player") -> dict[str, object]:
        return self._use(
            lambda session: session.strategy_status(),
            participant_id=participant_id,
        )

    def session_status(self, *, participant_id: str = "player") -> dict[str, object]:
        status = asdict(self.state.market_session_info())
        status["state"] = status["state"].value
        return status

    def stop_strategy(self) -> dict[str, object]:
        self.state.require_market_open()

        def stop(session: PlayerSession) -> dict[str, object]:
            session.stop_strategy()
            return session.strategy_status()

        return self._use(stop)

    def capture(
        self,
        feed: str,
        after_cursor: int,
        *,
        through_cursor: int | None = None,
        limit: int = 10_000,
        participant_id: str = "player",
    ) -> dict[str, object]:
        def select(session: PlayerSession):
            events, next_cursor, snapshot_end_cursor = session.capture_page(
                after_cursor=after_cursor,
                through_cursor=through_cursor,
                limit=limit,
            )
            chosen = select_market_capture_events(events, feed)
            return (
                chosen,
                next_cursor,
                snapshot_end_cursor,
                len(events),
                session.now,
            )

        (
            events,
            next_cursor,
            snapshot_end_cursor,
            raw_event_count,
            market_time,
        ) = self.state.with_capture_session(
            participant_id,
            select,
        )
        return {
            "feed": feed,
            "after_cursor": after_cursor,
            "next_after_cursor": next_cursor,
            "snapshot_end_cursor": snapshot_end_cursor,
            "complete": next_cursor == snapshot_end_cursor,
            "raw_event_count": raw_event_count,
            "market_time": market_time,
            "records": [event_record(item.cursor, item.envelope) for item in events],
        }

    def resume_market_session(self) -> dict[str, object]:
        status = asdict(self.state.resume_market_session())
        status["state"] = status["state"].value
        return status

    def participant_terminal_summary(self, participant_id: str) -> dict[str, object]:
        """Build evaluator-only terminal metrics for a non-owning role trace."""

        metrics = self.state.terminal_metrics(participant_id)
        series = self.state.terminal_time_series(participant_id)
        if metrics is None or series is None:
            raise RuntimeError("participant terminal metrics are unavailable")
        session = self._observe(lambda value: value, participant_id=participant_id)
        result = asdict(metrics)
        result["termination_state"] = metrics.termination_state.value
        result["time_mode"] = metrics.time_mode.value
        result["replay"] = {
            "evaluation_series": list(series),
            "deployments": [asdict(record) for record in session.deployment_records()],
        }
        result["artifact_bundle"] = {
            "status": "shared",
            "owner_participant_id": "player",
        }
        return result

    def terminal_metrics(self) -> EpisodeMetrics | None:
        return self.state.terminal_metrics()

    def sync_terminal(self) -> dict[str, object] | None:
        """Persist and return terminal state when a horizon finalized the episode."""

        metrics = self.terminal_metrics()
        if metrics is None:
            return None
        self._write_artifacts(metrics)
        result = asdict(metrics)
        result["termination_state"] = metrics.termination_state.value
        result["time_mode"] = metrics.time_mode.value
        result["artifact_directory"] = str(self._artifact_directory)
        result["artifact_manifest"] = self._artifact_manifest
        result["replay"] = self._replay_projection
        result["artifact_bundle"] = self._artifact_bundle()
        return result

    def terminate(self) -> dict[str, object]:
        metrics = self.state.terminate()
        self._write_artifacts(metrics)
        result = asdict(metrics)
        result["termination_state"] = metrics.termination_state.value
        result["time_mode"] = metrics.time_mode.value
        # Keep the tool response compact. The complete artifact manifest and replay
        # projection are synchronized through Toolset state, outside model context.
        result["artifact_written"] = self._artifact_directory is not None
        return result

    def _write_artifacts(self, metrics: EpisodeMetrics) -> None:
        if self._artifact_directory is not None:
            return
        series = self.state.terminal_time_series()
        if series is None:
            raise RuntimeError("terminal evaluation series is unavailable")
        session = self._observe(lambda value: value)
        series_rows = tuple(series)
        self._artifact_directory = EpisodeArtifactWriter(self.config.artifact_root).write(
            session=session,
            metrics=metrics,
            evaluation_series=series_rows,
            scenario_seed=self.config.scenario_seed,
            scenario_version=self.config.scenario_version,
            metadata={
                "runtime": "verifiers-toolset",
                "latent_demand_profile": self.latent_demand_profile.value,
                "latent_parent_order_count": len(self.scenario.parent_orders),
                "latent_parent_gross_quantity": (self.scenario.latent_parent_gross_quantity),
                "latent_parent_signed_quantity": (self.scenario.latent_parent_signed_quantity),
                "latent_parent_imbalance": self.scenario.latent_parent_imbalance,
                "opponent_roster_id": self.opponent_roster.roster_id,
            },
        )
        self._artifact_manifest = load_manifest(self._artifact_directory)
        self._replay_projection = {
            "evaluation_series": list(series_rows),
            "deployments": [asdict(record) for record in session.deployment_records()],
        }

    def _artifact_bundle(self) -> dict[str, object]:
        """Return a bounded inline transport for a task-scoped Toolset bundle."""

        if self._artifact_transport is not None:
            return self._artifact_transport
        if self._artifact_directory is None:
            raise RuntimeError("episode artifacts have not been written")
        if self.config.artifact_transport == "stream":
            self._artifact_transport = self._stream_descriptor()
            return self._artifact_transport
        payload = archive_bundle(self._artifact_directory)
        digest = hashlib.sha256(payload).hexdigest()
        descriptor: dict[str, object] = {
            "encoding": "base64+tar+gzip",
            "bytes": len(payload),
            "sha256": digest,
        }
        if len(payload) <= self.config.inline_artifact_max_bytes:
            descriptor["status"] = "inline"
            descriptor["data"] = base64.b64encode(payload).decode("ascii")
        elif self.config.artifact_transport == "auto":
            descriptor = self._stream_descriptor()
        else:
            descriptor["status"] = "too_large"
            descriptor["limit_bytes"] = self.config.inline_artifact_max_bytes
        self._artifact_transport = descriptor
        return descriptor

    def _stream_descriptor(self) -> dict[str, object]:
        if self._artifact_manifest is None:
            raise RuntimeError("episode artifact manifest is unavailable")
        files = self._artifact_manifest.get("files")
        if not isinstance(files, list):
            raise RuntimeError("episode artifact manifest has no file list")
        total_bytes = sum(int(entry.get("bytes", 0)) for entry in files if isinstance(entry, dict))
        return {
            "status": "stream",
            "encoding": "base64-file-chunks",
            "bytes": total_bytes,
            "chunk_bytes": self.config.artifact_export_chunk_bytes,
        }

    def export_artifact_file(
        self,
        *,
        path: str,
        offset: int,
        max_bytes: int,
    ) -> dict[str, object]:
        """Read one verified terminal artifact file through a bounded chunk."""

        if self._artifact_directory is None or self._artifact_manifest is None:
            raise RuntimeError("episode artifacts are not terminal and immutable")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
            or max_bytes > self.config.artifact_export_chunk_bytes
        ):
            raise ValueError("max_bytes must be between 1 and the configured export chunk size")
        files = self._artifact_manifest.get("files")
        if not isinstance(files, list):
            raise RuntimeError("episode artifact manifest has no file list")
        entries = {
            str(entry["path"]): entry
            for entry in files
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }
        entry = entries.get(path)
        if entry is None:
            raise ValueError("path is not declared by the terminal artifact manifest")
        expected_size = int(entry["bytes"])
        if offset > expected_size:
            raise ValueError("offset exceeds artifact file size")
        source = self._artifact_directory / path
        with source.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(max_bytes)
        next_offset = offset + len(payload)
        return {
            "path": path,
            "offset": offset,
            "next_offset": next_offset,
            "bytes": len(payload),
            "file_bytes": expected_size,
            "file_sha256": entry["sha256"],
            "chunk_sha256": hashlib.sha256(payload).hexdigest(),
            "eof": next_offset == expected_size,
            "data": base64.b64encode(payload).decode("ascii"),
        }


__all__ = [
    "EpisodeRuntime",
    "EpisodeRuntimeConfig",
    "event_record",
    "wait_response",
]
