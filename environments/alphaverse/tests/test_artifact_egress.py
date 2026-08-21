from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import alphaverse.artifact_egress as egress
import pytest
from alphaverse.episode_runtime import EpisodeRuntime, EpisodeRuntimeConfig
from alphaverse.replay import EpisodeReplay


def test_framework_transport_uses_a_non_mcp_route(monkeypatch) -> None:
    observed: list[tuple[str, dict[str, object]]] = []

    def fake_request(url, payload):
        observed.append((url, payload))
        return {"state": "open"}

    monkeypatch.setattr(egress, "_request_json", fake_request)

    result = asyncio.run(
        egress.call_framework(
            "http://toolset/mcp?vf_state_route=trace",
            "coordinator-secret",
            {"operation": "resume"},
        )
    )

    assert result == {"state": "open"}
    assert observed == [
        (
            "http://toolset/alphaverse-framework?vf_state_route=trace",
            {
                "capability": "coordinator-secret",
                "request": '{"operation":"resume"}',
            },
        )
    ]


def test_harness_streams_and_verifies_terminal_artifact_files(
    tmp_path,
    monkeypatch,
) -> None:
    runtime = EpisodeRuntime(
        EpisodeRuntimeConfig(
            episode_id="ep-harness-egress",
            scenario_seed=53,
            scenario_version="mvp-v1",
            max_market_time_ns=3_000_000_000,
            time_mode="manual",
            artifact_root=tmp_path / "toolset",
            artifact_transport="stream",
            artifact_export_chunk_bytes=1_024,
        )
    )
    runtime.terminate()
    trusted = runtime.sync_terminal()
    assert trusted is not None

    async def fake_call(url, capability, request):
        assert url == "http://toolset/mcp"
        assert capability == "export-secret"
        assert request["operation"] == "artifact_chunk"
        return runtime.export_artifact_file(
            path=request["path"],
            offset=request["offset"],
            max_bytes=request["max_bytes"],
        )

    monkeypatch.setattr(egress, "call_framework", fake_call)
    trace = SimpleNamespace(
        id="trace-egress",
        state=SimpleNamespace(
            terminal_summary=trusted,
            artifact_export_token="export-secret",
            coordination_token="coordinator-secret",
            artifact_egress_directory=None,
        ),
    )

    destination = asyncio.run(
        egress.export_terminal_artifacts(
            trace,
            {"alphaverse": "http://toolset/mcp"},
            root=tmp_path / "host",
        )
    )

    assert destination == tmp_path / "host" / "trace-egress" / "episode"
    assert trace.state.artifact_egress_directory == str(destination)
    assert trace.state.artifact_export_token is None
    assert trace.state.coordination_token == "coordinator-secret"
    assert trace.state.artifact_egress_complete is True
    replay = EpisodeReplay(destination)
    assert replay.manifest["episode_id"] == "ep-harness-egress"
    canonical = destination / "canonical-events.ndjson.gz"
    expected = tmp_path / "toolset" / "ep-harness-egress" / canonical.name
    assert canonical.read_bytes() == expected.read_bytes()
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == next(
        entry["sha256"] for entry in replay.manifest["files"] if entry["path"] == canonical.name
    )


def test_failed_stream_removes_incoming_directory(tmp_path, monkeypatch) -> None:
    runtime = EpisodeRuntime(
        EpisodeRuntimeConfig(
            episode_id="ep-failed-egress",
            scenario_seed=59,
            scenario_version="mvp-v1",
            max_market_time_ns=3_000_000_000,
            time_mode="manual",
            artifact_root=tmp_path / "toolset",
            artifact_transport="stream",
            artifact_export_chunk_bytes=1_024,
        )
    )
    runtime.terminate()
    trusted = runtime.sync_terminal()
    assert trusted is not None

    async def corrupt_call(url, capability, request):
        chunk = runtime.export_artifact_file(
            path=request["path"],
            offset=request["offset"],
            max_bytes=request["max_bytes"],
        )
        chunk["chunk_sha256"] = "0" * 64
        return chunk

    monkeypatch.setattr(egress, "call_framework", corrupt_call)
    trace = SimpleNamespace(
        id="trace-failed-egress",
        state=SimpleNamespace(
            terminal_summary=trusted,
            artifact_export_token="export-secret",
            artifact_egress_directory=None,
        ),
    )
    host = tmp_path / "host"

    with pytest.raises(RuntimeError, match="chunk checksum mismatch"):
        asyncio.run(
            egress.export_terminal_artifacts(
                trace,
                {"alphaverse": "http://toolset/mcp"},
                root=host,
            )
        )

    trace_root = host / "trace-failed-egress"
    assert not (trace_root / ".episode.incoming").exists()
    assert not (trace_root / "episode").exists()
    assert trace.state.artifact_export_token == "export-secret"
    assert not getattr(trace.state, "artifact_egress_complete", False)
