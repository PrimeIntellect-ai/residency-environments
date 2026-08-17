from __future__ import annotations

import base64
import gzip
import hashlib
import json

import pytest
from alphaverse.artifacts import ARTIFACT_SCHEMA, extract_bundle, load_manifest
from alphaverse.episode_runtime import EpisodeRuntime, EpisodeRuntimeConfig


def test_episode_artifact_bundle_is_replayable_and_self_describing(tmp_path) -> None:
    runtime = EpisodeRuntime(
        EpisodeRuntimeConfig(
            episode_id="ep-artifact",
            capability_token="control",
            capture_token="capture",
            scenario_seed=41,
            scenario_version="mvp-v1",
            max_market_time_ns=3_000_000_000,
            time_mode="manual",
            artifact_root=tmp_path,
        )
    )
    runtime.wait(
        duration_ns=1_000_000_000,
        until_ns=None,
        wake_on_alert=False,
    )
    result = runtime.terminate()
    directory = tmp_path / "ep-artifact"

    manifest = load_manifest(directory)
    assert manifest["schema"] == ARTIFACT_SCHEMA
    assert manifest["scenario_seed"] == 41
    assert manifest["terminal_event_sequence"] > 0
    assert {entry["path"] for entry in manifest["files"]} == {
        "canonical-events.ndjson.gz",
        "deployments.json",
        "evaluation-series.ndjson.gz",
        "final-snapshot.json",
        "summary.json",
    }

    with gzip.open(directory / "canonical-events.ndjson.gz", "rt") as stream:
        events = [json.loads(line) for line in stream]
    with gzip.open(directory / "evaluation-series.ndjson.gz", "rt") as stream:
        series = [json.loads(line) for line in stream]

    assert events[-1]["sequence"] == manifest["terminal_event_sequence"]
    assert series[-1]["market_time_ns"] == result["market_time"]
    assert "replay" not in result
    trusted = runtime.sync_terminal()
    assert trusted is not None
    assert trusted["artifact_manifest"]["schema"] == ARTIFACT_SCHEMA
    assert trusted["replay"]["evaluation_series"] == series
    bundle = trusted["artifact_bundle"]
    assert bundle["status"] == "inline"
    payload = base64.b64decode(bundle["data"], validate=True)
    assert len(payload) == bundle["bytes"]
    assert hashlib.sha256(payload).hexdigest() == bundle["sha256"]
    restored = extract_bundle(payload, tmp_path / "restored")
    assert load_manifest(restored) == manifest


def test_terminal_artifacts_stream_as_verified_bounded_file_chunks(tmp_path) -> None:
    runtime = EpisodeRuntime(
        EpisodeRuntimeConfig(
            episode_id="ep-stream",
            capability_token="control",
            capture_token="capture",
            scenario_seed=43,
            scenario_version="mvp-v1",
            max_market_time_ns=3_000_000_000,
            time_mode="manual",
            artifact_root=tmp_path,
            artifact_transport="stream",
            artifact_export_chunk_bytes=1_024,
        )
    )
    runtime.wait(
        duration_ns=1_000_000_000,
        until_ns=None,
        wake_on_alert=False,
    )
    runtime.terminate()
    trusted = runtime.sync_terminal()

    assert trusted is not None
    assert trusted["artifact_bundle"] == {
        "status": "stream",
        "encoding": "base64-file-chunks",
        "bytes": sum(entry["bytes"] for entry in trusted["artifact_manifest"]["files"]),
        "chunk_bytes": 1_024,
    }
    for entry in trusted["artifact_manifest"]["files"]:
        payload = bytearray()
        offset = 0
        while offset < entry["bytes"]:
            chunk = runtime.export_artifact_file(
                path=entry["path"],
                offset=offset,
                max_bytes=1_024,
            )
            decoded = base64.b64decode(chunk["data"], validate=True)
            assert chunk["chunk_sha256"] == hashlib.sha256(decoded).hexdigest()
            payload.extend(decoded)
            offset = chunk["next_offset"]
        assert bytes(payload) == (tmp_path / "ep-stream" / entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]

    with pytest.raises(ValueError, match="not declared"):
        runtime.export_artifact_file(path="../secret", offset=0, max_bytes=1)
