"""Trusted post-agent artifact streaming from a task-scoped Toolset."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from alphaverse.replay import EpisodeReplay

FRAMEWORK_ROUTE = "/alphaverse-framework"


def _host_url(url: str) -> str:
    """Translate Docker Desktop's agent-facing host alias back to host loopback."""

    parts = urlsplit(url)
    if parts.hostname != "host.docker.internal":
        return url
    port = f":{parts.port}" if parts.port is not None else ""
    return urlunsplit(parts._replace(netloc=f"127.0.0.1{port}"))


def _request_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        _host_url(url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        decoded = json.loads(response.read())
    if not isinstance(decoded, dict):
        raise RuntimeError("artifact MCP response is not a JSON object")
    error = decoded.get("error")
    if isinstance(error, dict):
        raise RuntimeError(f"artifact MCP error: {error.get('message', error)}")
    return decoded


async def call_mcp_tool(url: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call one MCP tool directly from trusted evaluator-side orchestration."""

    response = await asyncio.to_thread(
        _request_json,
        url,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("artifact MCP response has no result")
    content = result.get("content")
    if result.get("isError") or not isinstance(content, list) or not content:
        raise RuntimeError(f"artifact MCP tool failed: {content!r}")
    first = content[0]
    text = first.get("text") if isinstance(first, dict) else None
    decoded = json.loads(text) if isinstance(text, str) else None
    if not isinstance(decoded, dict):
        raise RuntimeError("artifact MCP tool returned a non-object payload")
    return decoded


async def call_framework(
    mcp_url: str,
    capability: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Call the evaluator-only route without advertising an MCP tool."""

    parts = urlsplit(_host_url(mcp_url))
    url = urlunsplit(parts._replace(path=FRAMEWORK_ROUTE))
    return await asyncio.to_thread(
        _request_json,
        url,
        {
            "capability": capability,
            "request": json.dumps(request, separators=(",", ":")),
        },
    )


async def export_terminal_artifacts(
    trace: Any,
    mcp_urls: dict[str, str],
    *,
    root: str | Path | None = None,
) -> Path | None:
    """Stream an immutable episode bundle after the agent has left the market."""

    if getattr(trace.state, "artifact_egress_complete", False):
        existing = getattr(trace.state, "artifact_egress_directory", None)
        return Path(existing) if isinstance(existing, str) and existing else None
    summary = getattr(trace.state, "terminal_summary", None)
    if not isinstance(summary, dict):
        return None
    bundle = summary.get("artifact_bundle")
    if not isinstance(bundle, dict) or bundle.get("status") != "stream":
        trace.state.artifact_egress_complete = True
        return None
    manifest = summary.get("artifact_manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("streamed artifact summary has no manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("streamed artifact manifest has no files")
    export_token = getattr(trace.state, "artifact_export_token", None)
    if not isinstance(export_token, str) or not export_token:
        raise RuntimeError("artifact export capability is unavailable")
    url = mcp_urls.get("alphaverse")
    if not isinstance(url, str) or not url:
        raise RuntimeError("Alphaverse Toolset URL is unavailable for artifact export")
    chunk_bytes = bundle.get("chunk_bytes")
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool):
        raise RuntimeError("streamed artifact descriptor has no chunk size")

    base = Path(root) if root is not None else Path.cwd() / ".alphaverse" / "toolset-artifacts"
    destination = base / str(trace.id) / "episode"
    incoming = destination.with_name(".episode.incoming")
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)

    try:
        for raw_entry in files:
            if not isinstance(raw_entry, dict):
                raise RuntimeError("invalid streamed artifact file entry")
            name = raw_entry.get("path")
            expected_bytes = raw_entry.get("bytes")
            expected_sha256 = raw_entry.get("sha256")
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or name in {".", ".."}
                or not isinstance(expected_bytes, int)
                or isinstance(expected_bytes, bool)
                or expected_bytes < 0
                or not isinstance(expected_sha256, str)
            ):
                raise RuntimeError("unsafe streamed artifact file entry")
            target = incoming / name
            temporary = target.with_suffix(target.suffix + ".part")
            digest = hashlib.sha256()
            offset = 0
            with temporary.open("wb") as stream:
                while offset < expected_bytes:
                    chunk = await call_framework(
                        url,
                        export_token,
                        {
                            "operation": "artifact_chunk",
                            "path": name,
                            "offset": offset,
                            "max_bytes": chunk_bytes,
                        },
                    )
                    encoded = chunk.get("data")
                    if not isinstance(encoded, str):
                        raise RuntimeError(f"artifact chunk for {name} has no data")
                    payload = base64.b64decode(encoded, validate=True)
                    if chunk.get("offset") != offset or chunk.get("bytes") != len(payload):
                        raise RuntimeError(f"artifact chunk offset mismatch for {name}")
                    if chunk.get("chunk_sha256") != hashlib.sha256(payload).hexdigest():
                        raise RuntimeError(f"artifact chunk checksum mismatch for {name}")
                    stream.write(payload)
                    digest.update(payload)
                    next_offset = chunk.get("next_offset")
                    if not isinstance(next_offset, int) or next_offset <= offset:
                        raise RuntimeError(f"artifact chunk made no progress for {name}")
                    offset = next_offset
            if offset != expected_bytes or digest.hexdigest() != expected_sha256:
                raise RuntimeError(f"streamed artifact file verification failed: {name}")
            temporary.replace(target)

        (incoming / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        EpisodeReplay(incoming, verify=True)
        if destination.exists():
            shutil.rmtree(destination)
        incoming.replace(destination)
    except Exception:
        shutil.rmtree(incoming, ignore_errors=True)
        raise
    trace.state.artifact_egress_directory = str(destination)
    trace.state.artifact_export_token = None
    trace.state.artifact_egress_complete = True
    return destination


__all__ = [
    "FRAMEWORK_ROUTE",
    "call_framework",
    "call_mcp_tool",
    "export_terminal_artifacts",
]
