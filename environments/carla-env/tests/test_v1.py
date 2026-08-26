from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from carla_env.cosmos import CosmosConfig
from carla_env.nurec import NuRecConfig
from carla_env.toolset import CarlaToolset, _enabled_tools
from carla_env.v1 import CarlaState, CarlaTask, CarlaTaskData, CarlaTaskset, CarlaTasksetConfig


def test_taskset_builds_native_v1_task() -> None:
    taskset = CarlaTaskset(CarlaTasksetConfig(id="carla-env", scenario="maze"))
    task = next(iter(taskset))

    assert isinstance(task, vf.Task)
    assert task.data.scenario == "maze"
    assert isinstance(task.toolsets(task.config)[0], CarlaToolset)


@pytest.mark.parametrize(
    ("scenario", "env_args", "included", "excluded"),
    [
        ("maze", {}, {"observe", "get_goal_info", "follow_route"}, {"capture_image"}),
        (
            "navigation_vision",
            {},
            {"capture_image", "get_goal_info", "control_vehicle"},
            {"observe", "capture_depth"},
        ),
        (
            "navigation_vision",
            {"enable_nurec": True, "nurec_mode": "replay"},
            {"observe", "capture_image"},
            {"control_vehicle", "get_goal_info"},
        ),
        (
            "navigation_vision",
            {"enable_nurec": True, "nurec_mode": "DRIVE"},
            {"control_vehicle", "capture_image", "get_goal_info"},
            {"observe", "capture_depth"},
        ),
        (
            "navigation_vision",
            {"nurec": NuRecConfig(enabled=True, mode="replay")},
            {"observe", "capture_image"},
            {"control_vehicle", "get_goal_info"},
        ),
        (
            "navigation_vision",
            {"enable_cosmos": True},
            {"capture_image", "capture_depth"},
            {"observe"},
        ),
        (
            "navigation_vision",
            {"cosmos": CosmosConfig(enabled=True)},
            {"capture_image", "capture_depth"},
            {"observe"},
        ),
    ],
)
def test_scenario_tool_selection(
    scenario: str,
    env_args: dict,
    included: set[str],
    excluded: set[str],
) -> None:
    enabled = _enabled_tools(CarlaTaskData(scenario=scenario, env_args=env_args))

    assert included <= enabled
    assert enabled.isdisjoint(excluded)


@pytest.mark.asyncio
async def test_toolset_registers_only_enabled_tools() -> None:
    toolset = CarlaToolset(vf.ToolsetConfig())
    await toolset.setup_task(CarlaTaskData(scenario="maze", env_args={}))
    server = FastMCP("test")
    toolset.register(server)

    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert "observe" in names
    assert "get_goal_info" in names
    assert "capture_image" not in names
    observe_tool = next(
        tool for tool in server._tool_manager.list_tools() if tool.name == "observe"
    )
    assert observe_tool.output_schema is None
    await toolset._exit_stack.aclose()


@pytest.mark.asyncio
async def test_call_tool_result_survives_fastmcp_conversion(monkeypatch) -> None:
    toolset = CarlaToolset(vf.ToolsetConfig())
    await toolset.setup_task(CarlaTaskData(scenario="maze", env_args={}))

    async def fake_call(name: str, arguments: dict) -> CallToolResult:
        assert name == "observe"
        assert arguments == {}
        return CallToolResult(content=[TextContent(type="text", text="observation")])

    monkeypatch.setattr(toolset, "_call", fake_call)
    server = FastMCP("test")
    toolset.register(server)
    result = await server._tool_manager.call_tool("observe", {}, convert_result=True)

    assert isinstance(result, CallToolResult)
    assert result.content[0].text == "observation"
    await toolset._exit_stack.aclose()


@pytest.mark.asyncio
async def test_toolset_reuses_task_reserved_endpoint(monkeypatch) -> None:
    from carla_env import env as env_module

    setup_kwargs: dict = {}

    class FakeSession:
        async def setup_state(self, state: dict, **kwargs) -> None:
            setup_kwargs.update(kwargs)
            state["prompt"] = []

        async def cleanup(self, state: dict) -> None:
            return None

    monkeypatch.setattr(env_module, "load_environment", lambda **kwargs: FakeSession())
    toolset = CarlaToolset(vf.ToolsetConfig())
    await toolset.setup_task(CarlaTaskData(scenario="maze", env_args={}))
    toolset.state.endpoint_host = "127.0.0.2"
    toolset.state.endpoint_port = 2000
    toolset.state.carla_version = "0.10.0"

    await toolset._ensure_session()

    assert setup_kwargs == {"external_endpoint_reserved": True}
    await toolset._exit_stack.aclose()


@pytest.mark.asyncio
async def test_native_reward_and_metrics_read_typed_state() -> None:
    taskset = CarlaTaskset(CarlaTasksetConfig(id="carla-env", scenario="maze"))
    task = next(iter(taskset))
    state = CarlaState(
        reward=0.75,
        scenario_outcome={"goal_reached": True, "goal_distance": 4.0},
        tool_calls=[{"name": "observe", "args": {}}, {"name": "observe", "args": {}}],
    )
    trace = SimpleNamespace(state=state)

    assert await task.carla_reward(trace) == 0.75
    assert await task.carla_metrics(trace) == {
        "total_tool_calls": 2.0,
        "observe_calls": 2.0,
        "goal_reached": 1.0,
        "goal_distance": 4.0,
    }

    native_trace = vf.Trace(
        task=vf.TraceTask(type="CarlaTask", data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        state=state,
    )
    await task.score(native_trace)
    assert native_trace.metrics == {
        "total_tool_calls": 2.0,
        "observe_calls": 2.0,
        "goal_reached": 1.0,
        "goal_distance": 4.0,
    }
    assert native_trace.rewards["carla_reward"].value == 0.75


@pytest.mark.asyncio
async def test_task_lifecycle_reserves_and_releases_exact_endpoint(monkeypatch) -> None:
    from carla_env import env as env_module

    calls: list[tuple[str, object]] = []

    class FakeSession:
        config = SimpleNamespace(carla_version="0.9.16", traffic_manager_enabled=False)

        async def reserve_endpoint(self, lease: dict) -> tuple[str, int]:
            calls.append(("reserve", lease))
            lease["_sandbox_reservation"] = SimpleNamespace(
                sandbox_id="sandbox-1",
                host="127.0.0.2",
                port=2000,
                lease_id="lease-1",
            )
            return "127.0.0.2", 2000

        async def release_endpoint(self, lease: dict) -> None:
            calls.append(("release", lease["_sandbox_reservation"]))

    monkeypatch.setattr(env_module, "load_environment", lambda **_: FakeSession())
    task = CarlaTask(CarlaTaskData(scenario="maze", env_args={}))
    trace = SimpleNamespace(state=CarlaState())

    await task.setup(trace, SimpleNamespace())
    assert trace.state.endpoint_reserved
    assert trace.state.sandbox_id == "sandbox-1"
    assert trace.state.sandbox_lease_id == "lease-1"

    await task.finalize(trace, SimpleNamespace())
    assert not trace.state.endpoint_reserved
    assert [name for name, _ in calls] == ["reserve", "release"]
    released = calls[-1][1]
    assert released.sandbox_id == "sandbox-1"
    assert released.host == "127.0.0.2"
    assert released.port == 2000
    assert released.lease_id == "lease-1"


def test_package_has_no_legacy_verifiers_imports() -> None:
    package = Path(__file__).parents[1] / "carla_env"
    sources = "\n".join(path.read_text() for path in package.rglob("*.py"))

    assert "verifiers.legacy" not in sources
