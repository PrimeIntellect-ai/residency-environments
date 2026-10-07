#!/usr/bin/env python3
"""Download raw Alphaverse public market data without placing it in model context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SESSION_FILE = Path(__file__).resolve().parent / ".alphaverse-session.json"


def _session() -> dict[str, str]:
    try:
        decoded = json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("the Alphaverse session configuration is unavailable") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("the Alphaverse session configuration is invalid")
    mcp_url = decoded.get("mcp_url")
    if isinstance(mcp_url, str) and mcp_url:
        return {"mcp_url": mcp_url}
    raise RuntimeError("the Alphaverse Toolset transport is unavailable")


def _mcp_request(payload: dict[str, object]) -> dict[str, object]:
    mcp_url = _session().get("mcp_url")
    if not mcp_url:
        raise RuntimeError("the Alphaverse Toolset URL is unavailable")
    request = urllib.request.Request(
        mcp_url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
    if not body:
        return {}
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise RuntimeError("the Alphaverse Toolset returned invalid JSON-RPC")
    return decoded


def _mcp_call(name: str, arguments: dict[str, object]) -> dict[str, object]:
    _mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "alphaverse-market-capture", "version": "1"},
            },
        }
    )
    _mcp_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
    listed = _mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
    )
    tools = listed.get("result", {}).get("tools", []) if isinstance(listed, dict) else []
    names = [tool.get("name") for tool in tools if isinstance(tool, dict)]
    resolved = name if name in names else f"alphaverse_{name}"
    if resolved not in names:
        raise RuntimeError(f"Alphaverse MCP tool {name!r} is unavailable; advertised={names!r}")
    response = _mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": resolved, "arguments": arguments},
        }
    )
    if not isinstance(response, dict):
        raise RuntimeError("the Alphaverse Toolset returned no response")
    if "error" in response:
        raise RuntimeError(f"Alphaverse Toolset error: {response['error']}")
    result = response.get("result")
    if not isinstance(result, dict) or result.get("isError") is True:
        raise RuntimeError("the Alphaverse Toolset capture call failed")
    content = result.get("content")
    if not isinstance(content, list) or not content:
        raise RuntimeError("the Alphaverse Toolset returned no capture content")
    text = content[0].get("text") if isinstance(content[0], dict) else None
    decoded = json.loads(text) if isinstance(text, str) else None
    if not isinstance(decoded, dict):
        raise RuntimeError("the Alphaverse Toolset returned invalid capture content")
    return decoded


def _write_spec(feed: str, output: Path | None) -> None:
    decoded = _mcp_call("market_capture_spec", {"feed": feed})
    rendered = json.dumps(decoded, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"feed": feed, "schema_file": str(output.resolve())}))


def _capture(
    *,
    feed: str,
    output: Path,
    after_cursor: int,
    append: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.part")
    digest = hashlib.sha256()
    byte_count = 0
    toolset_result: dict[str, object] | None = None
    try:
        cursor = after_cursor
        snapshot_end_cursor: int | None = None
        record_count = 0
        first_page = True
        with temporary.open("wb") as destination:
            while first_page or cursor < snapshot_end_cursor:
                first_page = False
                arguments: dict[str, object] = {
                    "feed": feed,
                    "after_cursor": cursor,
                    "limit": 10_000,
                }
                if snapshot_end_cursor is not None:
                    arguments["through_cursor"] = snapshot_end_cursor
                page = _mcp_call("capture_market_data", arguments)
                page_end = page.get("snapshot_end_cursor")
                next_cursor = page.get("next_after_cursor")
                records = page.get("records")
                if not isinstance(page_end, int) or not isinstance(next_cursor, int):
                    raise RuntimeError("the Alphaverse Toolset omitted capture cursors")
                if snapshot_end_cursor is None:
                    snapshot_end_cursor = page_end
                elif page_end != snapshot_end_cursor:
                    raise RuntimeError("the Alphaverse capture window changed between pages")
                if not cursor < next_cursor <= snapshot_end_cursor and not (
                    cursor == snapshot_end_cursor == next_cursor
                ):
                    raise RuntimeError("the Alphaverse Toolset returned a non-progressing page")
                if not isinstance(records, list):
                    raise RuntimeError("the Alphaverse Toolset omitted capture records")
                for record in records:
                    chunk = (
                        json.dumps(
                            record,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n"
                    )
                    destination.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                record_count += len(records)
                cursor = next_cursor
                toolset_result = page
        assert snapshot_end_cursor is not None
        toolset_result = {
            **(toolset_result or {}),
            "next_after_cursor": snapshot_end_cursor,
            "record_count": record_count,
        }
        if append:
            with output.open("ab") as destination, temporary.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            temporary.unlink()
        else:
            temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(
        json.dumps(
            {
                "feed": feed,
                "output_file": str(output.resolve()),
                "after_cursor": after_cursor,
                "next_after_cursor": int(toolset_result["next_after_cursor"]),
                "market_time": int(toolset_result["market_time"]),
                "record_count": int(toolset_result["record_count"]),
                "bytes_downloaded": byte_count,
                "download_sha256": digest.hexdigest(),
                "appended": append,
            },
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture raw delivered Alphaverse public market data")
    commands = parser.add_subparsers(dest="command", required=True)

    spec = commands.add_parser(
        "spec",
        help="print or save the authoritative NDJSON record specification",
    )
    spec.add_argument("--feed", choices=("mbo", "levels"), required=True)
    spec.add_argument("--output", type=Path)

    capture = commands.add_parser(
        "capture",
        help="download currently delivered packets after an exclusive cursor",
    )
    capture.add_argument("--feed", choices=("mbo", "levels"), required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--after-cursor", type=int, default=0)
    capture.add_argument(
        "--append",
        action="store_true",
        help="append selected records instead of atomically replacing the file",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "spec":
            _write_spec(args.feed, args.output)
        else:
            if args.after_cursor < 0:
                raise ValueError("--after-cursor must be non-negative")
            _capture(
                feed=args.feed,
                output=args.output,
                after_cursor=args.after_cursor,
                append=args.append,
            )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:2_000]
        print(
            f"market capture failed with HTTP {exc.code}: {detail}",
            file=sys.stderr,
        )
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"market capture failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
