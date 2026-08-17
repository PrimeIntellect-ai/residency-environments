"""Canonical, replay-oriented artifacts for one Alphaverse episode.

The bundle is transport independent: a Verifiers Toolset, a deterministic
research run, or a temporary migration adapter can all write the same files.
Human-facing dashboards consume these files and never become an authority for
market state.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from alphaverse.hosted import EpisodeMetrics
from alphaverse.observability import build_observability_snapshot
from alphaverse.player import PlayerSession

ARTIFACT_SCHEMA = "alphaverse.episode-artifact"
ARTIFACT_SCHEMA_VERSION = 1


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_jsonl_gzip(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as stream:
        for row in rows:
            stream.write(_json_bytes(dict(row)))
    temporary.replace(path)


def _file_entry(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


class EpisodeArtifactWriter:
    """Write an immutable episode bundle suitable for replay and analysis."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(
        self,
        *,
        session: PlayerSession,
        metrics: EpisodeMetrics,
        evaluation_series: Iterable[Mapping[str, object]],
        scenario_seed: int,
        scenario_version: str,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Persist the authoritative log plus compact derived projections."""

        directory = self.root / metrics.episode_id
        directory.mkdir(parents=True, exist_ok=True)

        summary_path = directory / "summary.json"
        series_path = directory / "evaluation-series.ndjson.gz"
        events_path = directory / "canonical-events.ndjson.gz"
        deployments_path = directory / "deployments.json"
        snapshot_path = directory / "final-snapshot.json"

        summary = asdict(metrics)
        summary["termination_state"] = metrics.termination_state.value
        summary["time_mode"] = metrics.time_mode.value
        _atomic_write(summary_path, _json_bytes(summary, pretty=True))
        _write_jsonl_gzip(series_path, evaluation_series)

        temporary_events = events_path.with_name(f".{events_path.name}.{os.getpid()}.tmp")
        with gzip.open(temporary_events, "wb", compresslevel=6) as stream:
            session.episode.exchange.event_log.copy_jsonl(stream)
        temporary_events.replace(events_path)

        deployments = [asdict(record) for record in session.deployment_records()]
        _atomic_write(
            deployments_path,
            _json_bytes({"deployments": deployments}, pretty=True),
        )
        _atomic_write(
            snapshot_path,
            _json_bytes(
                build_observability_snapshot(
                    session,
                    recent_event_limit=200,
                    recent_trade_limit=200,
                    price_history_limit=600,
                ),
                pretty=True,
            ),
        )

        payload_files = (
            summary_path,
            series_path,
            events_path,
            deployments_path,
            snapshot_path,
        )
        manifest = {
            "schema": ARTIFACT_SCHEMA,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "episode_id": metrics.episode_id,
            "participant_id": metrics.participant_id,
            "scenario_seed": scenario_seed,
            "scenario_version": scenario_version,
            "terminal_market_time_ns": metrics.market_time,
            "terminal_event_sequence": metrics.terminal_event_sequence,
            "metadata": dict(metadata or {}),
            "files": [_file_entry(path) for path in payload_files],
        }
        _atomic_write(
            directory / "manifest.json",
            _json_bytes(manifest, pretty=True),
        )
        return directory


def load_manifest(directory: str | Path) -> dict[str, object]:
    """Load and minimally validate a bundle manifest."""

    path = Path(directory) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact manifest must be an object")
    if payload.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("unknown artifact schema")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema version")
    return payload


def archive_bundle(directory: str | Path) -> bytes:
    """Pack a completed bundle for transport across a task-scoped runtime boundary."""

    root = Path(directory)
    load_manifest(root)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if path.is_file():
                archive.add(path, arcname=path.name, recursive=False)
    return buffer.getvalue()


def extract_bundle(payload: bytes, destination: str | Path) -> Path:
    """Restore a transported bundle, rejecting links and path traversal."""

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            candidate = Path(member.name)
            if candidate.is_absolute() or ".." in candidate.parts or not member.isfile():
                raise ValueError(f"unsafe episode artifact member: {member.name!r}")
        for member in members:
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable episode artifact member: {member.name!r}")
            (root / member.name).write_bytes(source.read())
    load_manifest(root)
    return root


__all__ = [
    "ARTIFACT_SCHEMA",
    "ARTIFACT_SCHEMA_VERSION",
    "EpisodeArtifactWriter",
    "archive_bundle",
    "extract_bundle",
    "load_manifest",
]
