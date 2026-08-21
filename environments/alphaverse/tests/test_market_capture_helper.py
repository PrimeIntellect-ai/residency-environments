from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _helper():
    path = Path(__file__).parents[1] / "src" / "alphaverse" / "agent_workspace" / "market_capture.py"
    spec = importlib.util.spec_from_file_location("alphaverse_market_capture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_accepts_only_the_toolset_transport(tmp_path) -> None:
    helper = _helper()
    session = tmp_path / ".alphaverse-session.json"
    helper._SESSION_FILE = session

    session.write_text(json.dumps({"mcp_url": "http://tools/mcp"}), encoding="utf-8")
    assert helper._session() == {"mcp_url": "http://tools/mcp"}

    session.write_text(
        json.dumps({"base_url": "http://legacy", "capture_token": "secret"}),
        encoding="utf-8",
    )
    try:
        helper._session()
    except RuntimeError as exc:
        assert "Toolset transport" in str(exc)
    else:
        raise AssertionError("legacy HTTP capture transport was accepted")


def test_capture_paginates_one_stable_snapshot(tmp_path, monkeypatch, capsys) -> None:
    helper = _helper()
    calls: list[dict[str, object]] = []

    def fake_call(name: str, arguments: dict[str, object]):
        assert name == "capture_market_data"
        calls.append(dict(arguments))
        if len(calls) == 1:
            return {
                "market_time": 10,
                "snapshot_end_cursor": 3,
                "next_after_cursor": 2,
                "records": [{"sequence": 1}, {"sequence": 2}],
            }
        return {
            "market_time": 10,
            "snapshot_end_cursor": 3,
            "next_after_cursor": 3,
            "records": [{"sequence": 3}],
        }

    monkeypatch.setattr(helper, "_mcp_call", fake_call)
    output = tmp_path / "capture.ndjson"
    helper._capture(feed="mbo", output=output, after_cursor=0, append=False)

    assert calls == [
        {"feed": "mbo", "after_cursor": 0, "limit": 10_000},
        {
            "feed": "mbo",
            "after_cursor": 2,
            "limit": 10_000,
            "through_cursor": 3,
        },
    ]
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"sequence": 1},
        {"sequence": 2},
        {"sequence": 3},
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary["next_after_cursor"] == 3
    assert summary["record_count"] == 3
