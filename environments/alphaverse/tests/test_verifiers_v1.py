from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

vf = pytest.importorskip("verifiers.v1")
import alphaverse.verifiers_v1 as adapter
from alphaverse.adaptive_env import AlphaverseAdaptiveEnv
from verifiers.v1.runtimes import DockerConfig, PrimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config


def test_tool_annotations_register_with_fastmcp() -> None:
    from mcp.server.fastmcp import FastMCP

    toolset = adapter.AlphaverseToolset(adapter.AlphaverseToolsetConfig())
    server = FastMCP("alphaverse-schema-smoke")
    toolset.register(server)

    assert set(server._tool_manager._tools) == toolset._PUBLIC_TOOLS
    assert "framework_channel" not in server._tool_manager._tools
    assert any(route.path == "/alphaverse-framework" for route in server._custom_starlette_routes)

    prop = adapter.AlphaversePropToolset(adapter.AlphaverseToolsetConfig())
    prop_server = FastMCP("alphaverse-prop-schema-smoke")
    prop.register(prop_server)
    assert set(prop_server._tool_manager._tools) == prop._ALLOWED_TOOLS
    assert "wait" not in prop_server._tool_manager._tools
    assert "terminate_session" not in prop_server._tool_manager._tools

    knob_prop = adapter.AlphaverseKnobPropToolset(adapter.AlphaverseToolsetConfig(prop_control_scope="knobs"))
    knob_server = FastMCP("alphaverse-knob-prop-schema-smoke")
    knob_prop.register(knob_server)
    assert "deploy_strategy" in knob_server._tool_manager._tools
    assert "deploy_prop_knobs" not in knob_server._tool_manager._tools


def test_framework_route_is_callable_without_mcp_discovery(monkeypatch) -> None:
    from mcp.server.fastmcp import FastMCP
    from starlette.testclient import TestClient

    toolset = adapter.AlphaverseToolset(adapter.AlphaverseToolsetConfig())

    async def fake_framework_request(capability: str, request: str):
        return {"capability": capability, "request": request}

    monkeypatch.setattr(toolset, "_framework_request", fake_framework_request)
    monkeypatch.setattr(toolset, "_with_state", lambda fn: fn)
    server = FastMCP("alphaverse-private-route-smoke")
    toolset.register(server)

    with TestClient(server.streamable_http_app()) as client:
        response = client.post(
            "/alphaverse-framework",
            json={"capability": "secret", "request": '{"operation":"resume"}'},
        )

    assert response.status_code == 200
    assert response.json() == {
        "capability": "secret",
        "request": '{"operation":"resume"}',
    }
    assert "framework_channel" not in server._tool_manager._tools


def test_conversational_event_reads_are_bounded() -> None:
    toolset = adapter.AlphaverseToolset(adapter.AlphaverseToolsetConfig())

    with pytest.raises(ValueError, match="between 1 and 200"):
        asyncio.run(toolset.events(limit=201))
    with pytest.raises(ValueError, match="between 1 and 200"):
        asyncio.run(toolset.events(limit=0))

    assert asyncio.run(toolset.market_capture_spec("mbo"))["event_kinds"] == [
        "mbo_change",
        "trade",
    ]


def _task(**config_overrides):
    config = adapter.AlphaverseTaskConfig(
        toolset=adapter.AlphaverseToolsetConfig(),
        **config_overrides,
    )
    data = adapter.AlphaverseData(
        idx=0,
        prompt="trade",
        scenario_seed=23,
        starting_cash=1_000_000,
        max_market_time_ns=500,
    )
    return adapter.AlphaverseTask(data, config)


def _trace(task):
    return vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        state=adapter.AlphaverseState(),
        ok=True,
    )


def test_taskset_loads_seeded_rows() -> None:
    taskset = adapter.AlphaverseTaskset(
        adapter.AlphaverseTasksetConfig(
            num_tasks=3,
            seed=17,
            max_market_time_ns=600_000_000_000,
            model_turn_cap=80,
            latent_demand_profile="strong-short",
        )
    )

    tasks = taskset.load()
    assert [task.data.scenario_seed for task in tasks] == [17, 18, 19]
    assert all(task.data.scenario_version == "mvp-v1" for task in tasks)
    assert all(task.data.latent_demand_profile.value == "strong-short" for task in tasks)
    assert all(task.data.network_allow == [] for task in tasks)
    assert all(task.data.network_block == ["*"] for task in tasks)
    assert "600 seconds" in tasks[0].data.prompt
    assert "Maximize terminal realized PnL over that horizon" in tasks[0].data.prompt
    assert "including research" in tasks[0].data.prompt
    assert "at most 80 model turns" in tasks[0].data.prompt


def test_alphaverse_data_rejects_widened_network_access() -> None:
    with pytest.raises(ValueError, match="framework-only network access"):
        adapter.AlphaverseData(
            prompt="trade",
            scenario_seed=23,
            network_allow=["*"],
            network_block=[],
        )


def test_task_config_rejects_unwired_token_time() -> None:
    with pytest.raises(ValueError, match="time_mode"):
        adapter.AlphaverseTaskConfig(time_mode="tokens")  # type: ignore[arg-type]


@pytest.mark.parametrize("runtime_config", [DockerConfig(), PrimeConfig()])
def test_alphaverse_task_resolves_to_framework_only_network(runtime_config) -> None:
    runtime = resolve_runtime_config(runtime_config, _task())

    assert runtime.allow == []
    assert runtime.block == ["*"]


def test_uncapped_turns_are_disclosed_with_a_fixed_market_horizon() -> None:
    task = adapter.AlphaverseTaskset(
        adapter.AlphaverseTasksetConfig(
            num_tasks=1,
            max_market_time_ns=21_600_000_000_000,
            model_turn_cap=None,
        )
    ).load()[0]

    assert "21600 seconds" in task.data.prompt
    assert "There is no gameplay model-turn cap" in task.data.prompt
    assert "remaining inventory is liquidated" in task.data.prompt


def test_default_prompt_states_objective_without_prescribing_strategy_workflow() -> None:
    prompt = adapter.AlphaverseTasksetConfig().prompt
    normalized = " ".join(prompt.split())

    assert "maximize terminal" in normalized
    assert "README.md and API.md" in normalized
    assert "whether or not a strategy is deployed" in normalized
    assert "make at least one" not in normalized
    assert "reserve your final turn" not in normalized
    assert "strategy.py" not in normalized


def test_adaptive_player_prompt_discloses_configured_session_length() -> None:
    task = _task(adaptive_prop=True, session_duration_ns=150_000_000_000)

    adaptive_task = AlphaverseAdaptiveEnv._player_task(task)
    prompt = " ".join(str(adaptive_task.data.prompt).split())

    assert "150000000000 ns (150 seconds)" in prompt
    assert "one-hour" not in prompt
    assert "may change their behavior, enter, or exit" in prompt
    assert "update schedules" in prompt


@pytest.mark.parametrize(
    ("prop_mode", "adaptive_prop"),
    [
        ("none", False),
        ("static", True),
        ("knobs", True),
        ("adaptive", True),
    ],
)
def test_adaptive_env_binds_prop_creation_to_experimental_arm(
    prop_mode: str,
    adaptive_prop: bool,
) -> None:
    task = _task(adaptive_prop=not adaptive_prop, session_duration_ns=150)

    episode_task = AlphaverseAdaptiveEnv._episode_task(task, prop_mode)

    assert episode_task.config.adaptive_prop is adaptive_prop
    assert task.config.adaptive_prop is (not adaptive_prop)
    assert episode_task.config.prop_control_scope == ("knobs" if prop_mode == "knobs" else "full_source")


def test_adaptive_env_binds_prop_seed_profile_without_mutating_source_task() -> None:
    task = _task(session_duration_ns=150)

    episode_task = AlphaverseAdaptiveEnv._episode_task(
        task,
        "adaptive",
        "competitive",
    )

    assert episode_task.config.prop_seed_profile == "competitive"
    assert episode_task.data.prop_seed_profile == "competitive"
    assert task.config.prop_seed_profile == "passive"
    assert task.data.prop_seed_profile == "passive"


def test_prop_task_varies_seed_and_framing_independently() -> None:
    task = _task(session_duration_ns=150)

    prop_task = AlphaverseAdaptiveEnv._prop_task(
        task,
        "ep-test",
        prop_seed_profile="competitive",
        prop_framing="arbitrary",
    )

    assert prop_task.data.prop_seed_profile == "competitive"
    assert prop_task.data.prop_framing == "arbitrary"
    assert prop_task.data.network_allow == []
    assert prop_task.data.network_block == ["*"]
    assert "arbitrary scaffolding" in str(prop_task.data.system_prompt)


def test_knob_prop_task_uses_bounded_tool_schema_and_prompt() -> None:
    task = _task(session_duration_ns=150)

    prop_task = AlphaverseAdaptiveEnv._prop_task(
        task,
        "ep-test",
        prop_seed_profile="competitive",
        prop_framing="neutral",
        prop_control_scope="knobs",
    )

    assert isinstance(prop_task, adapter.AlphaverseKnobPropTask)
    assert prop_task.data.prop_control_scope == "knobs"
    assert "deploy_strategy" in str(prop_task.data.system_prompt)
    assert "JSON knob document" in str(prop_task.data.system_prompt)


def test_task_scoped_setup_and_finalize_keep_episode_inside_toolset(monkeypatch) -> None:
    task = _task(adaptive_prop=True)
    trace = _trace(task)
    installed: list[object] = []

    async def fake_install(data, runtime):
        installed.append(data)

    monkeypatch.setattr(adapter, "install_task_workspace", fake_install)

    asyncio.run(task.setup(trace, None))  # type: ignore[arg-type]

    assert installed == [task.data]
    assert trace.state.episode_id.startswith("ep_")
    assert trace.state.artifact_export_token
    assert trace.state.prop_access_token
    assert trace.state.coordination_token

    trace.state.terminal_summary = {
        "termination_state": "terminated",
        "position": 0,
        "pnl": 5,
        "artifact_bundle": {"status": "inline"},
    }
    asyncio.run(task.finalize(trace, None))  # type: ignore[arg-type]

    assert trace.info["alphaverse"]["pnl"] == 5
    assert trace.state.artifact_export_token is None
    assert trace.state.coordination_token is None


def test_embedded_toolset_owns_episode_and_terminal_replay(tmp_path) -> None:
    config = adapter.AlphaverseToolsetConfig(
        artifact_root=str(tmp_path),
        time_mode="manual",
    )
    toolset = adapter.AlphaverseToolset(config)
    data = adapter.AlphaverseData(
        idx=0,
        prompt="trade",
        scenario_seed=31,
        starting_cash=1_000_000,
        max_market_time_ns=3_000_000_000,
    )
    toolset._inert_state = adapter.AlphaverseState(
        episode_id="ep-embedded",
    )

    async def run_episode():
        await toolset.setup_task(data)
        product = await toolset.product_terms()
        waited = await toolset.wait(
            until_ns=3_000_000_000,
            wake_on_alert=False,
        )
        return product, waited

    product, waited = asyncio.run(run_episode())

    assert product["product_id"] == "ALPHA"
    assert waited["market_time_ns"] == 3_000_000_000
    assert toolset.state.terminated is True
    summary = toolset.state.terminal_summary
    assert isinstance(summary, dict)
    assert summary["termination_state"] == "terminated"
    assert summary["replay"]["evaluation_series"]
    artifact_directory = tmp_path / "ep-embedded"
    assert (artifact_directory / "manifest.json").is_file()
    assert (artifact_directory / "canonical-events.ndjson.gz").is_file()


def test_embedded_artifact_export_requires_hidden_terminal_capability(tmp_path) -> None:
    config = adapter.AlphaverseToolsetConfig(
        artifact_root=str(tmp_path),
        artifact_transport="stream",
        artifact_export_chunk_bytes=64 * 1024,
        time_mode="manual",
    )
    toolset = adapter.AlphaverseToolset(config)
    data = adapter.AlphaverseData(
        idx=0,
        prompt="trade",
        scenario_seed=37,
        starting_cash=1_000_000,
        max_market_time_ns=3_000_000_000,
    )
    toolset._inert_state = adapter.AlphaverseState(
        episode_id="ep-export",
        artifact_export_token="export-secret",
    )

    async def run_episode():
        await toolset.setup_task(data)
        await toolset.wait(until_ns=3_000_000_000, wake_on_alert=False)
        request = json.dumps(
            {
                "operation": "artifact_chunk",
                "path": "summary.json",
                "max_bytes": 64 * 1024,
            }
        )
        with pytest.raises(PermissionError, match="invalid framework capability"):
            await toolset._framework_request(
                "wrong",
                request,
            )
        return await toolset._framework_request(
            "export-secret",
            request,
        )

    chunk = asyncio.run(run_episode())
    assert chunk["path"] == "summary.json"
    assert chunk["eof"] is True
    assert base64.b64decode(chunk["data"], validate=True)


def test_embedded_prop_role_is_scoped_to_own_account_and_safe_tools(tmp_path, monkeypatch) -> None:
    config = adapter.AlphaverseToolsetConfig(
        artifact_root=str(tmp_path),
        artifact_transport="stream",
        adaptive_prop=True,
        session_duration_ns=1_000_000_000,
        time_mode="manual",
    )
    toolset = adapter.AlphaverseToolset(config)
    data = adapter.AlphaverseData(
        idx=0,
        prompt="trade",
        scenario_seed=41,
        starting_cash=1_000_000,
        max_market_time_ns=3_000_000_000,
    )
    toolset._inert_state = adapter.AlphaverseState(
        episode_id="ep-adaptive-embedded",
        artifact_export_token="evaluator-secret",
        coordination_token="coordinator-secret",
        prop_access_token="prop-role-secret",
    )
    role_query = {
        "alphaverse_role": "prop",
        "alphaverse_role_token": "prop-role-secret",
    }
    monkeypatch.setattr(adapter, "_mcp_query_value", role_query.get)

    async def exercise_role():
        await toolset.setup_task(data)
        status = await toolset.strategy_status()
        account = await toolset.account()
        with pytest.raises(PermissionError, match="cannot call POST orders"):
            await toolset.submit_limit_order("forbidden", "buy", 10_000, 1)
        role_query.clear()
        await toolset.terminate_session()
        summary = await toolset._framework_request(
            "coordinator-secret",
            json.dumps(
                {
                    "operation": "participant_result",
                    "participant_id": "prop",
                }
            ),
        )
        return status, account, summary

    status, account, summary = asyncio.run(exercise_role())
    assert status["active"] is True
    assert str(status["strategy_version_id"]).startswith("sha256:")
    assert account["participant_id"] == "prop"
    assert summary["participant_id"] == "prop"
    assert summary["artifact_bundle"]["status"] == "shared"
    assert summary["position"] == 0


def test_embedded_knob_prop_cannot_deploy_raw_source(tmp_path, monkeypatch) -> None:
    toolset = adapter.AlphaverseToolset(
        adapter.AlphaverseToolsetConfig(
            artifact_root=str(tmp_path),
            adaptive_prop=True,
            prop_seed_profile="competitive",
            prop_control_scope="knobs",
            session_duration_ns=1_000_000_000,
            time_mode="manual",
        )
    )
    data = adapter.AlphaverseData(
        idx=0,
        prompt="trade",
        scenario_seed=43,
        starting_cash=1_000_000,
        max_market_time_ns=3_000_000_000,
        prop_control_scope="knobs",
    )
    toolset._inert_state = adapter.AlphaverseState(
        episode_id="ep-knob-prop",
        prop_access_token="prop-role-secret",
    )
    role_query = {
        "alphaverse_role": "prop",
        "alphaverse_role_token": "prop-role-secret",
    }
    monkeypatch.setattr(adapter, "_mcp_query_value", role_query.get)

    async def exercise_role():
        await toolset.setup_task(data)
        with pytest.raises(PermissionError, match="raw Python strategy source"):
            await toolset.deploy_strategy("class StrategyImpl: pass")
        deployed = await toolset.deploy_strategy(
            json.dumps(
                {
                    "quote_quantity": 6,
                    "correction_quantity": 14,
                    "quote_offset_ticks": 1,
                    "inventory_soft_limit": 12,
                    "max_abs_position": 48,
                    "inventory_retreat_ticks": 2,
                    "refresh_interval_ns": 1_000_000_000,
                }
            )
        )
        return deployed

    deployed = asyncio.run(exercise_role())
    assert deployed["strategy_knobs"]["quote_quantity"] == 6
    assert deployed["strategy_knobs"]["max_abs_position"] == 48
    assert str(deployed["strategy_version_id"]).startswith("sha256:")


def test_embedded_tool_error_still_synchronizes_terminal_state(
    monkeypatch,
) -> None:
    toolset = adapter.AlphaverseToolset(adapter.AlphaverseToolsetConfig())
    terminal = {
        "termination_state": "terminated",
        "position": 0,
        "pnl": 12.5,
        "artifact_bundle": {"status": "stream"},
    }
    toolset._inert_state = adapter.AlphaverseState(
        episode_id="ep-terminal-error",
    )
    toolset._runtime = SimpleNamespace(sync_terminal=lambda: terminal)

    def finalized_call(*args, **kwargs):
        raise RuntimeError("episode is finalized")

    monkeypatch.setattr(toolset, "_embedded_call", finalized_call)

    with pytest.raises(RuntimeError, match="episode is finalized"):
        asyncio.run(toolset._call("GET", "account"))

    assert toolset.state.terminal_summary == terminal
    assert toolset.state.terminated is True


def test_embedded_stop_waits_for_harness_artifact_egress() -> None:
    config = adapter.AlphaverseTaskConfig(toolset=adapter.AlphaverseToolsetConfig())
    data = adapter.AlphaverseData(
        idx=0,
        prompt="trade",
        scenario_seed=41,
        starting_cash=1_000_000,
        max_market_time_ns=500,
    )
    task = adapter.AlphaverseTask(data, config)
    trace = _trace(task)
    trace.state.terminated = True
    trace.state.market_time_ns = 500
    trace.state.terminal_summary = {
        "termination_state": "terminated",
        "artifact_bundle": {"status": "stream"},
    }

    assert asyncio.run(task.session_terminated(trace)) is False

    trace.state.artifact_egress_complete = True
    assert asyncio.run(task.session_terminated(trace)) is True


def test_reward_does_not_penalize_pre_scoring_ok_state() -> None:
    task = _task()
    trace = _trace(task)
    trace.ok = False  # Verifiers sets this to true only after scoring succeeds.
    trace.info["alphaverse"] = {
        "pnl": 2_000,
        "position": 0,
        "termination_state": "terminated",
    }

    assert asyncio.run(task.realized_pnl(trace)) == pytest.approx(0.2)


def test_reward_rejects_noncanonical_terminal_schema() -> None:
    task = _task()
    trace = _trace(task)
    trace.info["alphaverse"] = {
        "terminal_pnl": 2_000,
        "position": 0,
        "termination_state": "terminated",
    }

    with pytest.raises(KeyError, match="pnl"):
        asyncio.run(task.realized_pnl(trace))

    trace.info["alphaverse"] = {
        "pnl": "2000",
        "position": 0,
        "termination_state": "terminated",
    }
    with pytest.raises(TypeError, match="must be numeric"):
        asyncio.run(task.realized_pnl(trace))


def test_incomplete_rollout_uses_only_the_liquidation_penalty() -> None:
    task = _task()
    trace = _trace(task)
    trace.info["alphaverse"] = {
        "pnl": 0,
        "position": 0,
        "termination_state": "incomplete",
        "artifact_error": "episode did not terminate",
    }

    assert asyncio.run(task.realized_pnl(trace)) == -task.config.incomplete_liquidation_penalty
