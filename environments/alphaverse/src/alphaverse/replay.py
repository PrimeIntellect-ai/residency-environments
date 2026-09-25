"""Read-only projections over immutable Alphaverse episode artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterator

from alphaverse.artifacts import load_manifest


def _read_json(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return decoded


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise ValueError(f"expected JSON objects in {path}")
            rows.append(decoded)
    return rows


class EpisodeReplay:
    """Verified, transport-free reader used by reports and playback UIs."""

    def __init__(self, directory: str | Path, *, verify: bool = True) -> None:
        self.directory = Path(directory)
        self.manifest = load_manifest(self.directory)
        if verify:
            self.verify()
        self.summary = _read_json(self.directory / "summary.json")
        self.deployments = _read_json(self.directory / "deployments.json")
        self.final_snapshot = _read_json(self.directory / "final-snapshot.json")
        self.series = _read_gzip_jsonl(self.directory / "evaluation-series.ndjson.gz")
        self._series_times = [int(row["market_time_ns"]) for row in self.series]

    def verify(self) -> None:
        """Check every declared payload against the immutable manifest."""

        files = self.manifest.get("files")
        if not isinstance(files, list):
            raise ValueError("episode artifact manifest has no file list")
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("invalid episode artifact file entry")
            path = self.directory / entry["path"]
            payload = path.read_bytes()
            if len(payload) != entry.get("bytes"):
                raise ValueError(f"episode artifact size mismatch: {path.name}")
            if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
                raise ValueError(f"episode artifact checksum mismatch: {path.name}")

    def frame_at(self, market_time_ns: int) -> dict[str, Any]:
        """Return the latest compact observation at or before a virtual time."""

        if not self.series:
            raise ValueError("episode artifact has no evaluation series")
        index = bisect_right(self._series_times, market_time_ns) - 1
        return self.series[max(0, index)]

    def events(self) -> Iterator[dict[str, Any]]:
        """Stream canonical events without loading the full log into memory."""

        path = self.directory / "canonical-events.ndjson.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                decoded = json.loads(line)
                if not isinstance(decoded, dict):
                    raise ValueError(f"expected JSON objects in {path}")
                yield decoded

    def receipt(self) -> dict[str, Any]:
        """Small dashboard/report descriptor with no live exchange dependency."""

        return {
            "schema": self.manifest["schema"],
            "episode_id": self.manifest["episode_id"],
            "scenario_seed": self.manifest["scenario_seed"],
            "scenario_version": self.manifest["scenario_version"],
            "terminal_market_time_ns": self.manifest["terminal_market_time_ns"],
            "terminal_event_sequence": self.manifest["terminal_event_sequence"],
            "series_records": len(self.series),
            "deployment_records": len(self.deployments.get("deployments", [])),
            "terminal_pnl": self.summary.get("pnl"),
        }


__all__ = ["EpisodeReplay"]
