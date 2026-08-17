from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from alphaverse.acp_harness import ArtifactExportSession
from alphaverse.claude_code_harness import AlphaverseClaudeCodeHarness
from alphaverse.codex_harness import AlphaverseCodexHarness
from alphaverse.eval_harness import AlphaverseHarness, install_role_workspace
from alphaverse.verifiers_v1 import AlphaverseData
from verifiers.v1.acp import ACPConfig
from verifiers.v1.harnesses.bash import BashHarness, BashHarnessConfig
from verifiers.v1.harnesses.claude_code import (
    ClaudeCodeHarness,
    ClaudeCodeHarnessConfig,
)
from verifiers.v1.harnesses.codex import CodexHarness, CodexHarnessConfig
from verifiers.v1.task import TaskData
from verifiers.v1.utils.loaders import harness_class


class _FakeRuntime:
    def __init__(self) -> None:
        self.prepared = False
        self.files: dict[str, bytes] = {}
        self.program_calls: list[tuple[list[str], dict[str, str]]] = []

    async def prepare_uv_script(self, source: str, env: dict[str, str]) -> None:
        self.prepared = bool(source)

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def run_program(self, argv, env):
        self.program_calls.append((list(argv), dict(env)))
        return "launched"


def test_bash_plugin_resolves_and_installs_only_the_public_player_kit() -> None:
    assert harness_class("alphaverse_eval_harness") is AlphaverseHarness
    runtime = _FakeRuntime()
    harness = AlphaverseHarness(BashHarnessConfig(id="alphaverse_eval_harness"))

    asyncio.run(harness.setup(runtime))  # type: ignore[arg-type]

    assert runtime.prepared
    assert set(runtime.files) == {"README.md", "API.md", "market_capture.py"}
    assert b"cannot inspect the exchange implementation" in runtime.files["README.md"]
    assert b"deploy_strategy" in runtime.files["API.md"]
    assert b'"method": "tools/call"' in runtime.files["market_capture.py"]


def test_prop_workspace_uses_configured_seed_and_framing() -> None:
    runtime = _FakeRuntime()
    data = AlphaverseData(
        idx=0,
        prompt=None,
        scenario_seed=7,
        participant_id="prop",
        prop_seed_profile="competitive",
        prop_framing="arbitrary",
    )

    asyncio.run(install_role_workspace(data, runtime))  # type: ignore[arg-type]

    assert b"inside-market starter" in runtime.files["ROLE.md"]
    assert b"arbitrary scaffolding" in runtime.files["ROLE.md"]
    assert b"self.best_bid" in runtime.files["strategy.py"]
    assert b"base_half_spread=3" not in runtime.files["strategy.py"]


def test_codex_plugin_resolves_and_installs_only_the_public_player_kit(
    monkeypatch,
) -> None:
    async def fake_setup(self, runtime):
        return None

    monkeypatch.setattr(CodexHarness, "setup", fake_setup)
    assert harness_class("alphaverse_codex_harness") is AlphaverseCodexHarness
    runtime = _FakeRuntime()
    harness = AlphaverseCodexHarness(CodexHarnessConfig(id="alphaverse_codex_harness"))

    asyncio.run(harness.setup(runtime))  # type: ignore[arg-type]

    assert set(runtime.files) == {"README.md", "API.md", "market_capture.py"}
    assert b"cannot inspect the exchange implementation" in runtime.files["README.md"]


def test_codex_prepare_acp_installs_capture_and_disables_web_search(
    monkeypatch,
) -> None:
    async def fake_prepare(*args, **kwargs):
        return ACPConfig(
            env={"CODEX_CONFIG": json.dumps({"model": "openai/gpt-5.6-sol"})},
            command=["codex-acp"],
            prompt="play",
        )

    monkeypatch.setattr(CodexHarness, "prepare_acp", fake_prepare)
    runtime = _FakeRuntime()
    harness = AlphaverseCodexHarness(CodexHarnessConfig(id="alphaverse_codex_harness"))
    trace = SimpleNamespace(state=SimpleNamespace(toolset_url=None))

    config = asyncio.run(
        harness.prepare_acp(
            SimpleNamespace(model="openai/gpt-5.6-sol"),  # type: ignore[arg-type]
            trace,  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
            "http://interception/v1",
            "interception-secret",
            {"alphaverse": "http://tools/mcp?vf_state_route=trace"},
            TaskData(prompt="play"),
        )
    )

    assert json.loads(runtime.files[".alphaverse-session.json"]) == {"mcp_url": "http://tools/mcp?vf_state_route=trace"}
    assert json.loads(config.env["CODEX_CONFIG"])["web_search"] == "disabled"


def test_claude_code_plugin_resolves_and_installs_public_player_kit(
    monkeypatch,
) -> None:
    async def fake_setup(self, runtime):
        return None

    monkeypatch.setattr(ClaudeCodeHarness, "setup", fake_setup)
    assert harness_class("alphaverse_claude_code_harness") is AlphaverseClaudeCodeHarness
    runtime = _FakeRuntime()
    harness = AlphaverseClaudeCodeHarness(ClaudeCodeHarnessConfig(id="alphaverse_claude_code_harness"))

    asyncio.run(harness.setup(runtime))  # type: ignore[arg-type]

    assert set(runtime.files) == {"README.md", "API.md", "market_capture.py"}
    assert b"cannot inspect the exchange implementation" in runtime.files["README.md"]


def test_launch_installs_episode_scoped_capture_configuration(monkeypatch) -> None:
    async def fake_launch(*args, **kwargs):
        return "launched"

    monkeypatch.setattr(BashHarness, "launch", fake_launch)
    runtime = _FakeRuntime()
    harness = AlphaverseHarness(BashHarnessConfig(id="alphaverse_eval_harness"))
    trace = SimpleNamespace(state=SimpleNamespace(toolset_url=None))

    result = asyncio.run(
        harness.launch(
            None,  # type: ignore[arg-type]
            trace,  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
            "http://interception",
            "interception-secret",
            {"alphaverse": "http://tools/mcp?vf_state_route=trace"},
            TaskData(prompt="play"),
        )
    )

    assert result == "launched"
    assert json.loads(runtime.files[".alphaverse-session.json"]) == {"mcp_url": "http://tools/mcp?vf_state_route=trace"}
    assert trace.state.toolset_url == "http://tools/mcp?vf_state_route=trace"


def test_claude_prepare_acp_installs_capture_and_blocks_web_tools(
    monkeypatch,
) -> None:
    async def fake_prepare(*args, **kwargs):
        return ACPConfig(
            env={},
            command=["claude-agent-acp"],
            prompt="play",
            session_meta={"claudeCode": {"options": {"disallowedTools": ["Existing"]}}},
        )

    monkeypatch.setattr(ClaudeCodeHarness, "prepare_acp", fake_prepare)
    runtime = _FakeRuntime()
    harness = AlphaverseClaudeCodeHarness(ClaudeCodeHarnessConfig(id="alphaverse_claude_code_harness"))
    trace = SimpleNamespace(state=SimpleNamespace(toolset_url=None))

    config = asyncio.run(
        harness.prepare_acp(
            SimpleNamespace(model="anthropic/claude-opus-5"),  # type: ignore[arg-type]
            trace,  # type: ignore[arg-type]
            runtime,  # type: ignore[arg-type]
            "http://interception/v1",
            "interception-secret",
            {"alphaverse": "http://tools/mcp"},
            TaskData(prompt="play"),
        )
    )

    assert json.loads(runtime.files[".alphaverse-session.json"]) == {"mcp_url": "http://tools/mcp"}
    assert config.session_meta is not None
    disabled = config.session_meta["claudeCode"]["options"]["disallowedTools"]
    assert disabled == ["Existing", "WebFetch", "WebSearch"]


def test_artifact_export_session_exports_after_each_native_turn(monkeypatch) -> None:
    exported: list[tuple[object, dict[str, str]]] = []

    async def fake_export(trace, mcp_urls):
        exported.append((trace, mcp_urls))

    monkeypatch.setattr("alphaverse.acp_harness.export_terminal_artifacts", fake_export)

    class FakeSession:
        def __init__(self) -> None:
            self.trace = SimpleNamespace(id="trace-native")
            self.mcp_urls = {"alphaverse": "http://tools/mcp"}
            self.turns = 0
            self.closed = False

        async def turn(self, messages=None):
            self.turns += 1

        async def close(self):
            self.closed = True

    inner = FakeSession()
    session = ArtifactExportSession(inner)  # type: ignore[arg-type]
    asyncio.run(session.turn())
    asyncio.run(session.close())

    assert inner.turns == 1
    assert inner.closed
    assert exported == [(inner.trace, inner.mcp_urls)]
