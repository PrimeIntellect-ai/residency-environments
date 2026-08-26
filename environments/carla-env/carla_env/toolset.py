"""Per-rollout CARLA controls exposed through the Verifiers v1 MCP boundary."""

from __future__ import annotations

import json
from typing import Any

import verifiers.v1 as vf
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent
from verifiers.v1.utils.decorators import discover_decorated

from .nurec import normalize_nurec_mode
from .v1 import CarlaState, CarlaTaskData


def _config_value(config: object, name: str, default: object = None) -> object:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _enabled_tools(data: CarlaTaskData) -> set[str]:
    scenario = data.scenario
    args = data.env_args
    nurec = args.get("nurec")
    cosmos = args.get("cosmos")
    nurec_enabled = bool(args.get("enable_nurec") or _config_value(nurec, "enabled", False))
    nurec_mode = normalize_nurec_mode(
        args.get("nurec_mode") or _config_value(nurec, "mode", "replay")
    )
    cosmos_enabled = bool(args.get("enable_cosmos") or _config_value(cosmos, "enabled", False))
    vision_enabled = bool(
        args.get("enable_vision")
        or scenario.startswith(("navigation_vision", "free_roam"))
        or nurec_enabled
        or cosmos_enabled
    )

    tools: set[str] = set()
    if not (nurec_enabled and nurec_mode == "replay"):
        tools.update(
            {
                "control_vehicle",
                "brake_vehicle",
                "emergency_stop",
                "lane_change",
                "init_navigation_agent",
                "set_destination",
                "follow_route",
            }
        )
    if scenario.startswith(("maze", "navigation")) and not (
        nurec_enabled and nurec_mode == "replay"
    ):
        tools.add("get_goal_info")
    if not scenario.startswith("navigation_vision") or (nurec_enabled and nurec_mode == "replay"):
        tools.add("observe")
    if vision_enabled:
        tools.add("capture_image")
    if cosmos_enabled:
        tools.add("capture_depth")
    return tools


class CarlaToolset(vf.Toolset[vf.ToolsetConfig, CarlaState]):
    """Own one simulator session and expose its controls to one rollout."""

    TOOL_PREFIX = None

    def __init__(self, config: vf.ToolsetConfig) -> None:
        super().__init__(config)
        self._task_data: CarlaTaskData | None = None
        self._session = None
        self._session_state: dict[str, Any] | None = None
        self._initial_context = ""

    async def setup_task(self, task: CarlaTaskData) -> None:
        self._task_data = task
        self._exit_stack.push_async_callback(self._close_session)

    def register(self, mcp: FastMCP) -> None:
        if self._task_data is None:
            raise RuntimeError("CARLA tool server did not receive task data")
        enabled = _enabled_tools(self._task_data)
        for fn in discover_decorated(self, "tool"):
            if fn.__name__ not in enabled:
                continue
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
                structured_output=False,
            )

    async def _ensure_session(self) -> None:
        if self._session is not None:
            return
        if self._task_data is None:
            raise RuntimeError("CARLA tool server did not receive task data")
        if not self.state.endpoint_host or self.state.endpoint_port is None:
            raise RuntimeError("CARLA task did not reserve a simulator endpoint")

        from .env import load_environment

        args = dict(self._task_data.env_args)
        args.update(
            {
                "host": self.state.endpoint_host,
                "port": self.state.endpoint_port,
                "sandbox": {"mode": "disabled"},
                "carla_version": self.state.carla_version,
                "traffic_manager_enabled": self.state.traffic_manager_enabled,
            }
        )
        session = load_environment(scenario=self._task_data.scenario, **args)
        session_state: dict[str, Any] = {}
        await session.setup_state(session_state, external_endpoint_reserved=True)
        self._session = session
        self._session_state = session_state
        prompt = session_state.get("prompt") or []
        self._initial_context = "\n\n".join(
            str(message.get("content") or "")
            for message in prompt
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        )

    async def _close_session(self) -> None:
        if self._session is None or self._session_state is None:
            return
        session, state = self._session, self._session_state
        try:
            await session.cleanup(state)
        finally:
            self._sync_state()
            self._session = None
            self._session_state = None

    def _sync_state(self) -> None:
        if self._session_state is None:
            return
        source = self._session_state
        outcome = source.get("scenario_outcome") or {}
        self.state.done = bool(source.get("done", False))
        self.state.env_step = int(source.get("env_step", 0))
        self.state.observation = str(source.get("observation") or "")
        self.state.reward = float(outcome.get("reward", 0.0) or 0.0)
        self.state.scenario_outcome = dict(outcome)
        self.state.tool_calls = [dict(call) for call in source.get("tool_calls") or []]
        video_path = source.get("video_path")
        self.state.video_path = str(video_path) if video_path else None

    async def _call(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        await self._ensure_session()
        assert self._session is not None
        assert self._session_state is not None
        call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "carla-v1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }
        messages = await self._session.env_response([call], self._session_state)
        self._sync_state()

        blocks: list[TextContent | ImageContent] = []
        if self._initial_context:
            blocks.append(TextContent(type="text", text=self._initial_context))
            self._initial_context = ""
        is_error = False
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                blocks.append(TextContent(type="text", text=content))
                is_error = is_error or content.lower().startswith(("error", "tool error"))
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "text":
                    blocks.append(TextContent(type="text", text=str(part.get("text") or "")))
                elif part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url") or "")
                    prefix = "data:image/jpeg;base64,"
                    if url.startswith(prefix):
                        blocks.append(
                            ImageContent(
                                type="image", data=url[len(prefix) :], mimeType="image/jpeg"
                            )
                        )
        if not blocks:
            blocks.append(TextContent(type="text", text="CARLA action completed."))
        return CallToolResult(content=blocks, isError=is_error)

    @vf.tool
    async def observe(self) -> CallToolResult:
        """Advance the simulator and return the latest observation."""
        return await self._call("observe", {})

    @vf.tool
    async def control_vehicle(self, throttle: float, steer: float) -> CallToolResult:
        """Apply one manual throttle and steering control update."""
        return await self._call("control_vehicle", {"throttle": throttle, "steer": steer})

    @vf.tool
    async def brake_vehicle(self, intensity: float = 1.0) -> CallToolResult:
        """Apply a braking control update at the requested intensity."""
        return await self._call("brake_vehicle", {"intensity": intensity})

    @vf.tool
    async def emergency_stop(self) -> CallToolResult:
        """Apply full braking immediately."""
        return await self._call("emergency_stop", {})

    @vf.tool
    async def lane_change(self, direction: str, duration_s: float = 1.2) -> CallToolResult:
        """Execute a fixed-duration lane change in the requested direction."""
        return await self._call("lane_change", {"direction": direction, "duration_s": duration_s})

    @vf.tool
    async def init_navigation_agent(
        self, behavior: str = "basic", target_speed_kmh: float | None = None
    ) -> CallToolResult:
        """Initialize a CARLA navigation controller for the ego vehicle."""
        return await self._call(
            "init_navigation_agent",
            {"behavior": behavior, "target_speed_kmh": target_speed_kmh},
        )

    @vf.tool
    async def set_destination(self, x: float, y: float, z: float = 0.0) -> CallToolResult:
        """Set the navigation controller destination in world coordinates."""
        return await self._call("set_destination", {"x": x, "y": y, "z": z})

    @vf.tool
    async def follow_route(self, steps: int = 20) -> CallToolResult:
        """Run the navigation controller for a bounded number of simulator ticks."""
        return await self._call("follow_route", {"steps": steps})

    @vf.tool
    async def get_goal_info(self) -> CallToolResult:
        """Return the current distance and coarse direction to the active goal."""
        return await self._call("get_goal_info", {})

    @vf.tool
    async def capture_image(self) -> CallToolResult:
        """Capture the current front RGB camera frame without advancing time."""
        return await self._call("capture_image", {})

    @vf.tool
    async def capture_depth(self) -> CallToolResult:
        """Capture the current front depth frame without advancing time."""
        return await self._call("capture_depth", {})


if __name__ == "__main__":
    CarlaToolset.run()
