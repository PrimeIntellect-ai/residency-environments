"""Verifiers v1 adapter for evaluator-owned Alphaverse episodes.

The exchange runs inside a non-colocated, task-scoped Toolset runtime.
"""

import json
import secrets
from typing import Any, Literal
from urllib.parse import quote

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphaverse.capture import market_capture_spec as build_market_capture_spec
from alphaverse.episode_runtime import EpisodeRuntime, EpisodeRuntimeConfig
from alphaverse.prop_trader import PROP_PARTICIPANT_ID, competitive_prop_source
from alphaverse.world import LatentDemandProfile


def _mcp_query_value(name: str) -> str | None:
    """Read one role-routing query value from the active MCP request."""

    from mcp.server.lowlevel.server import request_ctx

    try:
        request = request_ctx.get().request
    except LookupError:
        return None
    if request is None:
        return None
    values = request.query_params.getlist(name)
    if len(values) > 1:
        raise ValueError(f"duplicate {name!r} role coordinate")
    return values[0] if values else None


_DEFAULT_PROMPT = """You are playing Alphaverse, an automated trading game.

Read README.md and API.md in your workspace. Your goal is to maximize terminal
realized profit and loss (PnL). You decide how to research, trade, use automated
strategies, and manage the session.

Market time advances while you think and during explicit waits, whether or not
a strategy is deployed. Ending the session cancels live orders and aggressively
liquidates remaining inventory; fees and liquidation slippage count toward PnL.
Call terminate_session when you choose to finish.
"""


def _prompt_with_limits(
    prompt: str,
    cap_ns: int | None,
    model_turn_cap: int | None,
) -> str:
    """Add configured limits as neutral constraints, not a workflow."""

    constraints: list[str] = []
    if cap_ns is not None:
        seconds = cap_ns / 1_000_000_000
        constraints.append(
            f"You have exactly {cap_ns} ns ({seconds:g} seconds) of market time "
            "in this episode. Maximize terminal realized PnL over that horizon. "
            "The horizon applies to total market time, including research and "
            "periods when no automated strategy is deployed. At the horizon, "
            "live orders are cancelled and remaining inventory is liquidated "
            "through ordinary book liquidity. You may terminate earlier."
        )
    if model_turn_cap is not None:
        constraints.append(f"The rollout permits at most {model_turn_cap} model turns.")
    else:
        constraints.append("There is no gameplay model-turn cap.")
    if not constraints:
        return prompt
    return f"{prompt.rstrip()}\n\n" + "\n".join(constraints) + "\n"


class AlphaverseData(vf.TaskData):
    """Immutable scenario row serialized with a rollout trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    network_allow: list[str] = Field(default_factory=list)
    network_block: list[str] = Field(default_factory=lambda: ["*"])
    scenario_seed: int
    scenario_version: str = "mvp-v1"
    latent_demand_profile: LatentDemandProfile = LatentDemandProfile.BALANCED
    participant_id: str = "player"
    starting_cash: int = Field(default=1_000_000, gt=0)
    max_market_time_ns: int | None = Field(default=None, gt=0)
    join_episode_id: str | None = None
    prop_seed_profile: Literal["passive", "competitive"] = "passive"
    prop_framing: Literal["incumbent", "neutral", "arbitrary"] = "incumbent"
    prop_control_scope: Literal["full_source", "knobs"] = "full_source"
    opponent_roster_id: str | None = None

    @model_validator(mode="after")
    def require_framework_only_network(self) -> "AlphaverseData":
        """Prevent an eval row from widening agent egress by accident."""

        if self.network_allow or self.network_block != ["*"]:
            raise ValueError("Alphaverse agent tasks require framework-only network access")
        return self


class AlphaverseState(vf.State):
    """Compact per-rollout state synchronized around MCP calls."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str | None = None
    capability_token: str | None = None
    capture_token: str | None = None
    terminated: bool = False
    event_cursor: int = Field(default=0, ge=0)
    market_time_ns: int = Field(default=0, ge=0)
    terminal_summary: dict[str, Any] | None = None
    artifact_export_token: str | None = None
    coordination_token: str | None = None
    artifact_egress_directory: str | None = None
    artifact_egress_complete: bool = False
    participant_id: str = "player"
    prop_access_token: str | None = None
    toolset_url: str | None = None


class AlphaverseToolsetConfig(vf.ToolsetConfig):
    """Placement and simulation settings for the market Toolset."""

    model_config = ConfigDict(extra="forbid")

    colocated: Literal[False] = False
    artifact_root: str = "/tmp/alphaverse-artifacts"
    inline_artifact_max_bytes: int = Field(default=24 * 1024 * 1024, gt=0)
    artifact_transport: Literal["auto", "inline", "stream"] = "auto"
    artifact_export_chunk_bytes: int = Field(
        default=4 * 1024 * 1024,
        ge=64 * 1024,
        le=8 * 1024 * 1024,
    )
    time_mode: Literal["manual", "wall", "tokens"] = "wall"
    wall_time_scale: float = Field(default=1.0, ge=0)
    wall_quantum_ns: int = Field(default=1_000_000, gt=0)
    ns_per_turn: int = Field(default=0, ge=0)
    ns_per_input_token: int = Field(default=0, ge=0)
    ns_per_cached_input_token: int = Field(default=0, ge=0)
    ns_per_output_token: int = Field(default=0, ge=0)
    initial_margin_per_contract: int = Field(default=5_000, gt=0)
    maintenance_margin_per_contract: int = Field(default=4_000, gt=0)
    margin_liquidation_grace_ns: int = Field(default=30_000_000_000, ge=0)
    adaptive_prop: bool = False
    prop_seed_profile: Literal["passive", "competitive"] = "passive"
    prop_control_scope: Literal["full_source", "knobs"] = "full_source"
    opponent_roster_id: str | None = None
    session_duration_ns: int | None = Field(default=None, gt=0)


class PropStrategyKnobs(BaseModel):
    """Bounded controls for the competitive prop strategy family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quote_quantity: int = Field(default=8, ge=1, le=32)
    correction_quantity: int = Field(default=12, ge=1, le=32)
    quote_offset_ticks: int = Field(default=0, ge=0, le=4)
    inventory_soft_limit: int = Field(default=16, ge=1, le=48)
    max_abs_position: int = Field(default=64, ge=16, le=120)
    inventory_retreat_ticks: int = Field(default=1, ge=0, le=4)
    refresh_interval_ns: int = Field(
        default=2_000_000_000,
        ge=250_000_000,
        le=10_000_000_000,
    )

    @model_validator(mode="after")
    def validate_inventory_bounds(self) -> "PropStrategyKnobs":
        if self.inventory_soft_limit >= self.max_abs_position:
            raise ValueError("inventory_soft_limit must be below max_abs_position")
        if self.quote_quantity >= self.max_abs_position:
            raise ValueError("quote_quantity must be below max_abs_position")
        return self


class AlphaverseTaskConfig(vf.TaskConfig):
    """Runtime and scoring settings shared by every row in a taskset."""

    model_config = ConfigDict(extra="forbid")

    toolset: AlphaverseToolsetConfig = AlphaverseToolsetConfig()
    time_mode: Literal["manual", "wall", "tokens"] = "wall"
    wall_time_scale: float = Field(default=1.0, ge=0)
    wall_quantum_ns: int = Field(default=1_000_000, gt=0)
    ns_per_turn: int = Field(default=0, ge=0)
    ns_per_input_token: int = Field(default=0, ge=0)
    ns_per_cached_input_token: int = Field(default=0, ge=0)
    ns_per_output_token: int = Field(default=0, ge=0)
    initial_margin_per_contract: int = Field(default=5_000, gt=0)
    maintenance_margin_per_contract: int = Field(default=4_000, gt=0)
    margin_liquidation_grace_ns: int = Field(default=30_000_000_000, ge=0)
    reward_scale: float = Field(default=10_000.0, gt=0)
    reward_clip: float = Field(default=10.0, gt=0)
    incomplete_liquidation_penalty: float = Field(default=1.0, ge=0)
    rollout_error_penalty: float = Field(default=0.25, ge=0)
    adaptive_prop: bool = False
    prop_seed_profile: Literal["passive", "competitive"] = "passive"
    prop_control_scope: Literal["full_source", "knobs"] = "full_source"
    opponent_roster_id: str | None = None
    session_duration_ns: int | None = Field(default=None, gt=0)


class AlphaverseTasksetConfig(vf.TasksetConfig):
    """Dataset-like scenario generation settings for local evals and training."""

    model_config = ConfigDict(extra="forbid")

    id: str = "alphaverse"
    num_tasks: int = Field(default=5, gt=0)
    seed: int = 0
    scenario_version: str = "mvp-v1"
    latent_demand_profile: LatentDemandProfile = LatentDemandProfile.BALANCED
    participant_id: str = "player"
    starting_cash: int = Field(default=1_000_000, gt=0)
    max_market_time_ns: int | None = Field(
        default=1_800_000_000_000,
        gt=0,
    )
    model_turn_cap: int | None = Field(default=None, gt=0)
    prompt: str = _DEFAULT_PROMPT
    task: AlphaverseTaskConfig = AlphaverseTaskConfig()


class AlphaverseToolset(vf.Toolset[AlphaverseToolsetConfig, AlphaverseState]):
    """Task-scoped MCP surface for one evaluator-owned episode."""

    TOOL_PREFIX = "alphaverse"

    def __init__(self, config: AlphaverseToolsetConfig) -> None:
        super().__init__(config)
        self._task_data: AlphaverseData | None = None
        self._runtime: EpisodeRuntime | None = None
        self._role_event_cursors: dict[str, int] = {}

    async def setup_task(self, task: AlphaverseData) -> None:
        self._task_data = task

    def _embedded(self) -> EpisodeRuntime:
        if self._runtime is not None:
            return self._runtime
        data = self._task_data
        if data is None:
            raise RuntimeError("Alphaverse task data is unavailable")
        episode_id, capability = self._coordinates()
        capture = self.state.capture_token
        if not isinstance(capture, str) or not capture:
            raise RuntimeError("Alphaverse capture capability is unavailable")
        self._runtime = EpisodeRuntime(
            EpisodeRuntimeConfig(
                episode_id=episode_id,
                capability_token=capability,
                capture_token=capture,
                scenario_seed=data.scenario_seed,
                scenario_version=data.scenario_version,
                latent_demand_profile=data.latent_demand_profile,
                starting_cash=data.starting_cash,
                max_market_time_ns=data.max_market_time_ns,
                time_mode=self.config.time_mode,
                wall_time_scale=self.config.wall_time_scale,
                wall_quantum_ns=self.config.wall_quantum_ns,
                ns_per_turn=self.config.ns_per_turn,
                ns_per_input_token=self.config.ns_per_input_token,
                ns_per_cached_input_token=self.config.ns_per_cached_input_token,
                ns_per_output_token=self.config.ns_per_output_token,
                initial_margin_per_contract=self.config.initial_margin_per_contract,
                maintenance_margin_per_contract=(self.config.maintenance_margin_per_contract),
                margin_liquidation_grace_ns=(self.config.margin_liquidation_grace_ns),
                adaptive_prop=self.config.adaptive_prop,
                prop_seed_profile=self.config.prop_seed_profile,
                prop_control_scope=self.config.prop_control_scope,
                opponent_roster_id=self.config.opponent_roster_id,
                session_duration_ns=self.config.session_duration_ns,
                artifact_root=self.config.artifact_root,
                inline_artifact_max_bytes=self.config.inline_artifact_max_bytes,
                artifact_transport=self.config.artifact_transport,
                artifact_export_chunk_bytes=(self.config.artifact_export_chunk_bytes),
            )
        )
        return self._runtime

    def _coordinates(self) -> tuple[str, str]:
        episode_id = self.state.episode_id
        token = self.state.capability_token
        if not episode_id or not token:
            raise RuntimeError("Alphaverse episode has not been initialized")
        return episode_id, token

    def _embedded_participant_id(self) -> str:
        role = _mcp_query_value("alphaverse_role")
        if role is None:
            return "player"
        if role != PROP_PARTICIPANT_ID or not self.config.adaptive_prop:
            raise PermissionError("invalid embedded participant role")
        supplied = _mcp_query_value("alphaverse_role_token")
        expected = self.state.prop_access_token
        if not supplied or not expected or not secrets.compare_digest(supplied, expected):
            raise PermissionError("invalid embedded participant role capability")
        return PROP_PARTICIPANT_ID

    async def _call(
        self,
        method: str,
        suffix: str,
        *,
        body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        participant_id = self._embedded_participant_id()
        try:
            response = self._embedded_call(
                method,
                suffix,
                body=body or {},
                query=query or {},
                participant_id=participant_id,
            )
        finally:
            terminal = self._embedded().sync_terminal()
            if terminal is not None:
                # A domain operation may discover that the horizon finalized
                # the episode and then raise. Synchronize terminal state even
                # on that error path so the harness can export before teardown.
                self.state.terminal_summary = terminal
                self.state.terminated = True
        self._update_state(response, participant_id=participant_id)
        return response

    def _embedded_call(
        self,
        method: str,
        suffix: str,
        *,
        body: dict[str, Any],
        query: dict[str, Any],
        participant_id: str,
    ) -> dict[str, Any]:
        runtime = self._embedded()
        if participant_id == PROP_PARTICIPANT_ID:
            allowed = {
                ("GET", "market/snapshot"),
                ("GET", "events"),
                ("GET", "account"),
                ("GET", "product"),
                ("GET", "orders"),
                ("GET", "strategy/status"),
                ("GET", "session/status"),
            }
            allowed.add(("POST", "strategy/deploy"))
            if (method, suffix) not in allowed:
                raise PermissionError(f"participant {participant_id} cannot call {method} {suffix}")
        if method == "GET" and suffix == "market/snapshot":
            return runtime.market_snapshot(int(query.get("depth", 10)), participant_id=participant_id)
        if method == "GET" and suffix == "events":
            return runtime.events(
                int(query.get("after_cursor", 0)),
                int(query.get("limit", 100)),
                participant_id=participant_id,
            )
        if method == "GET" and suffix == "account":
            return runtime.account(participant_id=participant_id)
        if method == "GET" and suffix == "product":
            return runtime.product(participant_id=participant_id)
        if method == "GET" and suffix == "orders":
            return runtime.open_orders(participant_id=participant_id)
        if method == "POST" and suffix == "orders":
            return runtime.submit_limit_order(
                client_order_id=str(body["client_order_id"]),
                side=str(body["side"]),
                price=int(body["price"]),
                quantity=int(body["quantity"]),
            )
        if method == "DELETE" and suffix.startswith("orders/"):
            return runtime.cancel_order(suffix.removeprefix("orders/"))
        if method == "POST" and suffix == "market/wait":
            return runtime.wait(
                duration_ns=body.get("duration_ns"),
                until_ns=body.get("until_ns"),
                wake_on_alert=bool(body.get("wake_on_alert", True)),
            )
        if method == "POST" and suffix == "strategy/deploy":
            if participant_id == PROP_PARTICIPANT_ID and self.config.prop_control_scope == "knobs":
                raw_source = body.get("source")
                if not isinstance(raw_source, str) or not raw_source:
                    raise ValueError("knobs control requires a JSON source document")
                try:
                    raw_knobs = json.loads(raw_source)
                except json.JSONDecodeError as exc:
                    raise PermissionError("raw Python strategy source is unavailable in knobs control") from exc
                if not isinstance(raw_knobs, dict):
                    raise ValueError("knob document must be a JSON object")
                knobs = PropStrategyKnobs.model_validate(raw_knobs).model_dump()
                response = runtime.deploy_strategy(
                    competitive_prop_source(**knobs),
                    "strategy:StrategyImpl",
                    participant_id=participant_id,
                )
                response["strategy_knobs"] = knobs
                return response
            source = body.get("source")
            if not isinstance(source, str) or not source:
                raise ValueError("source deployment requires non-empty source")
            return runtime.deploy_strategy(
                source,
                str(body.get("entrypoint", "strategy:StrategyImpl")),
                participant_id=participant_id,
            )
        if method == "GET" and suffix == "strategy/status":
            return runtime.strategy_status(participant_id=participant_id)
        if method == "GET" and suffix == "session/status":
            return runtime.session_status(participant_id=participant_id)
        if method == "POST" and suffix == "strategy/stop":
            return runtime.stop_strategy()
        if method == "POST" and suffix == "session/terminate":
            return runtime.terminate()
        raise RuntimeError(f"unsupported embedded operation: {method} {suffix}")

    def _update_state(self, response: dict[str, Any], *, participant_id: str = "player") -> None:
        cursor = response.get("next_after_cursor", response.get("event_cursor"))
        if isinstance(cursor, int):
            if participant_id == "player":
                self.state.event_cursor = max(self.state.event_cursor, cursor)
            else:
                self._role_event_cursors[participant_id] = max(self._role_event_cursors.get(participant_id, 0), cursor)
        market_time = response.get("market_time_ns", response.get("market_time"))
        if isinstance(market_time, int):
            self.state.market_time_ns = max(self.state.market_time_ns, market_time)
        status = response.get("state", response.get("status"))
        terminal_summary = response.get("terminal_summary")
        if isinstance(terminal_summary, dict):
            self.state.terminal_summary = terminal_summary
            self.state.terminated = True
        if response.get("terminated") is True or status in {
            "terminated",
            "complete",
            "completed",
        }:
            self.state.terminated = True

    @vf.tool
    async def market_snapshot(self, depth: int = 10) -> dict[str, Any]:
        """Return the most recently delivered top-of-book levels, up to ``depth``."""

        return await self._call("GET", "market/snapshot", query={"depth": depth})

    @vf.tool
    async def events(self, after_cursor: int | None = None, limit: int = 100) -> dict[str, Any]:
        """Read 1-200 delivered events after a cursor; paginate with the returned cursor."""

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an int")
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        participant_id = self._embedded_participant_id()
        default_cursor = (
            self.state.event_cursor if participant_id == "player" else self._role_event_cursors.get(participant_id, 0)
        )
        cursor = default_cursor if after_cursor is None else after_cursor
        return await self._call("GET", "events", query={"after_cursor": cursor, "limit": limit})

    @vf.tool
    async def account(self) -> dict[str, Any]:
        """Return authoritative cash, position, fees, and current margin state."""

        return await self._call("GET", "account")

    @vf.tool
    async def product_terms(self) -> dict[str, Any]:
        """Return product economics, fees, risk limits, margin, and technology terms."""

        return await self._call("GET", "product")

    @vf.tool
    async def market_capture_spec(
        self,
        feed: Literal["mbo", "levels"],
    ) -> dict[str, Any]:
        """Return the exact NDJSON schema for a raw public market-data capture."""

        return build_market_capture_spec(feed)

    @vf.tool
    async def capture_market_data(
        self,
        feed: Literal["mbo", "levels"],
        after_cursor: int = 0,
        through_cursor: int | None = None,
        limit: int = 10_000,
    ) -> dict[str, Any]:
        """Return one bounded page of raw delivered capture records."""

        participant_id = self._embedded_participant_id()
        response = self._embedded().capture(
            feed,
            after_cursor,
            through_cursor=through_cursor,
            limit=limit,
            participant_id=participant_id,
        )
        self._update_state(response, participant_id=participant_id)
        return response

    @vf.tool
    async def framework_channel(
        self,
        capability: str,
        request: str,
    ) -> dict[str, Any]:
        """Framework-reserved maintenance channel requiring a hidden capability."""

        try:
            payload = json.loads(request)
        except json.JSONDecodeError as exc:
            raise PermissionError("invalid framework request") from exc
        if not isinstance(payload, dict):
            raise PermissionError("invalid framework request")
        operation = payload.get("operation")
        export_token = self.state.artifact_export_token
        if export_token and secrets.compare_digest(capability, export_token):
            if operation != "artifact_chunk" or not self.state.terminated:
                raise PermissionError("invalid framework request")
            return self._embedded().export_artifact_file(
                path=str(payload.get("path", "")),
                offset=int(payload.get("offset", 0)),
                max_bytes=int(payload.get("max_bytes", 4 * 1024 * 1024)),
            )
        control_token = self.state.coordination_token
        if not control_token or not secrets.compare_digest(capability, control_token):
            raise PermissionError("invalid framework capability")
        runtime = self._embedded()
        if operation == "resume":
            return runtime.resume_market_session()
        if operation != "participant_result" or payload.get("participant_id") != PROP_PARTICIPANT_ID:
            raise PermissionError("invalid framework request")
        summary = runtime.participant_terminal_summary(PROP_PARTICIPANT_ID)
        summary["shared_episode_owner"] = "player"
        return summary

    @vf.tool
    async def open_orders(self) -> dict[str, Any]:
        """Return the player's currently live exchange orders."""

        return await self._call("GET", "orders")

    @vf.tool
    async def submit_limit_order(
        self,
        client_order_id: str,
        side: Literal["buy", "sell"],
        price: int,
        quantity: int,
    ) -> dict[str, Any]:
        """Queue a limit order; its exchange acceptance arrives asynchronously."""

        return await self._call(
            "POST",
            "orders",
            body={
                "client_order_id": client_order_id,
                "side": side,
                "price": price,
                "quantity": quantity,
            },
        )

    @vf.tool
    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Queue cancellation of one live order by exchange order ID."""

        return await self._call("DELETE", f"orders/{quote(order_id, safe='')}")

    @vf.tool
    async def wait(
        self,
        duration_ns: int | None = None,
        until_ns: int | None = None,
        wake_on_alert: bool = True,
    ) -> dict[str, Any]:
        """Advance market time, returning early for a strategy-authored alert by default."""

        return await self._call(
            "POST",
            "market/wait",
            body={
                "duration_ns": duration_ns,
                "until_ns": until_ns,
                "wake_on_alert": wake_on_alert,
            },
        )

    @vf.tool
    async def deploy_strategy(
        self,
        source: str,
        entrypoint: str = "strategy:StrategyImpl",
    ) -> dict[str, Any]:
        """Validate and deploy a content-addressed strategy artifact."""

        return await self._call(
            "POST",
            "strategy/deploy",
            body={"source": source, "entrypoint": entrypoint},
        )

    @vf.tool
    async def strategy_status(self) -> dict[str, Any]:
        """Inspect the current deployed strategy, health, version, and faults."""

        return await self._call("GET", "strategy/status")

    @vf.tool
    async def session_status(self) -> dict[str, Any]:
        """Return the current open/intermission state and virtual-time boundary."""

        return await self._call("GET", "session/status")

    @vf.tool
    async def stop_strategy(self) -> dict[str, Any]:
        """Stop the deployed strategy and cancel its orders without flattening."""

        return await self._call("POST", "strategy/stop", body={})

    @vf.tool
    async def terminate_session(self) -> dict[str, Any]:
        """End play, cancel orders, and aggressively liquidate the remaining position."""

        response = await self._call("POST", "session/terminate", body={})
        self.state.terminated = True
        summary = response.get("terminal_summary")
        if self.state.terminal_summary is None and isinstance(summary, dict):
            self.state.terminal_summary = summary
        return response


class AlphaversePropToolset(AlphaverseToolset):
    """Research/deployment-only view used by the scheduled prop controller."""

    _ALLOWED_TOOLS = {
        "market_snapshot",
        "events",
        "account",
        "product_terms",
        "market_capture_spec",
        "capture_market_data",
        "open_orders",
        "deploy_strategy",
        "strategy_status",
        "session_status",
    }

    def register(self, mcp) -> None:
        from verifiers.v1.utils.decorators import discover_decorated

        for fn in discover_decorated(self, "tool"):
            if fn.__name__ not in self._ALLOWED_TOOLS:
                continue
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )


class AlphaverseKnobPropToolset(AlphaversePropToolset):
    """Research plus bounded strategy parameters for the knobs-only arm."""

    _ALLOWED_TOOLS = AlphaversePropToolset._ALLOWED_TOOLS


class AlphaverseTask(vf.Task[AlphaverseData, AlphaverseState, AlphaverseTaskConfig]):
    """Episode lifecycle, termination condition, and authoritative scoring."""

    @classmethod
    def toolsets(cls, config: AlphaverseTaskConfig) -> list[vf.Toolset]:
        """Launch one evaluator-owned market Toolset for each rollout."""

        toolset = config.toolset.model_copy(
            update={
                "time_mode": config.time_mode,
                "wall_time_scale": config.wall_time_scale,
                "wall_quantum_ns": config.wall_quantum_ns,
                "ns_per_turn": config.ns_per_turn,
                "ns_per_input_token": config.ns_per_input_token,
                "ns_per_cached_input_token": config.ns_per_cached_input_token,
                "ns_per_output_token": config.ns_per_output_token,
                "initial_margin_per_contract": config.initial_margin_per_contract,
                "maintenance_margin_per_contract": (config.maintenance_margin_per_contract),
                "margin_liquidation_grace_ns": (config.margin_liquidation_grace_ns),
                "adaptive_prop": config.adaptive_prop,
                "prop_seed_profile": config.prop_seed_profile,
                "prop_control_scope": config.prop_control_scope,
                "opponent_roster_id": config.opponent_roster_id,
                "session_duration_ns": config.session_duration_ns,
            }
        )
        return [AlphaverseToolset(toolset)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if self.data.join_episode_id is not None:
            raise RuntimeError("the episode owner cannot join an existing task-scoped Toolset")
        trace.state.episode_id = f"ep_{secrets.token_urlsafe(12)}"
        trace.state.capability_token = secrets.token_urlsafe(32)
        trace.state.capture_token = secrets.token_urlsafe(32)
        trace.state.artifact_export_token = secrets.token_urlsafe(32)
        trace.state.participant_id = self.data.participant_id
        if self.config.adaptive_prop:
            trace.state.prop_access_token = secrets.token_urlsafe(32)
            trace.state.coordination_token = secrets.token_urlsafe(32)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        summary = trace.state.terminal_summary
        if isinstance(summary, dict):
            trace.state.terminated = True
            trace.info["alphaverse"] = summary
            egress_directory = trace.state.artifact_egress_directory
            bundle = summary.get("artifact_bundle")
            if (
                isinstance(bundle, dict)
                and bundle.get("status") == "stream"
                and (not egress_directory or not trace.state.artifact_egress_complete)
            ):
                raise RuntimeError("streamed terminal artifacts were not exported before Toolset teardown")
            artifact_directory = summary.get("artifact_directory")
            if isinstance(egress_directory, str):
                trace.info["alphaverse_artifacts"] = {
                    "artifact_directory": egress_directory,
                    "transport": "harness-stream",
                }
            elif isinstance(artifact_directory, str):
                trace.info["alphaverse_artifacts"] = {
                    "artifact_directory": artifact_directory,
                    "transport": "toolset-runtime",
                }
            trace.state.artifact_export_token = None
            trace.state.coordination_token = None
        else:
            trace.info["alphaverse"] = {
                "termination_state": "incomplete",
                "position": 0,
                "pnl": 0,
                "artifact_error": ("episode ended before terminate_session or the market horizon"),
            }

    @vf.stop
    async def session_terminated(self, trace: vf.Trace) -> bool:
        # A terminal market is not enough to tear down a task-scoped Toolset.
        # Its terminal tool response must first reach the harness, and the
        # harness must finish any artifact stream while the server is alive.
        return bool(isinstance(trace.state.terminal_summary, dict) and trace.state.artifact_egress_complete)

    @staticmethod
    def _number(summary: dict[str, Any], *keys: str) -> float:
        for key in keys:
            value = summary.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    pass
        return 0.0

    @classmethod
    def _liquidation_failed(cls, summary: dict[str, Any]) -> bool:
        remaining = cls._number(summary, "remaining_position", "position")
        state = summary.get("termination_state")
        return bool(
            summary.get("liquidation_failed", False)
            or remaining != 0
            or state
            in {
                "active",
                "incomplete",
                "liquidating",
                "liquidation_waiting",
            }
        )

    @vf.reward(weight=1.0)
    async def realized_pnl(self, trace: vf.Trace) -> float:
        """Normalized terminal realized PnL with explicit failure penalties."""

        summary = trace.info.get("alphaverse", {})
        if not isinstance(summary, dict):
            return -self.config.incomplete_liquidation_penalty
        pnl = self._number(summary, "realized_pnl", "terminal_pnl", "pnl")
        if not any(key in summary for key in ("realized_pnl", "terminal_pnl", "pnl")):
            terminal_cash = self._number(summary, "terminal_cash", "cash")
            pnl = terminal_cash - self.data.starting_cash
        score = pnl / self.config.reward_scale
        if self._liquidation_failed(summary):
            score -= self.config.incomplete_liquidation_penalty
        # Scoring runs before Verifiers marks a successfully completed trace
        # ``ok=True``.  Recorded errors/its stop condition, rather than the
        # transient ``ok`` flag, identify an actual rollout failure here.
        if trace.errors or trace.stop_condition == "error":
            score -= self.config.rollout_error_penalty
        return max(-self.config.reward_clip, min(self.config.reward_clip, score))

    @vf.metric
    async def terminal_metrics(self, trace: vf.Trace) -> dict[str, float]:
        """Record raw episode diagnostics separately from the optimized reward."""

        summary = trace.info.get("alphaverse", {})
        if not isinstance(summary, dict):
            summary = {}
        # Verifiers v0.2.2 records inference usage on ModelCall objects.  Older
        # releases attached it to sampled message nodes, so keep that fallback
        # to make saved/pre-upgrade traces and local adapters harmless.
        usages = (
            [call.usage for call in trace.calls if call.usage is not None]
            + [node_usage for node in trace.nodes if (node_usage := getattr(node, "usage", None)) is not None]
            + list(trace.extra_usage)
        )
        prompt_tokens = sum(usage.prompt_tokens for usage in usages)
        completion_tokens = sum(usage.completion_tokens for usage in usages)
        cached_input_tokens = sum(usage.cached_input_tokens or 0 for usage in usages)
        reasoning_tokens = sum(usage.reasoning_tokens or 0 for usage in usages)
        reported_costs = [usage.cost for usage in usages if usage.cost is not None]
        observed_model_turns = trace.num_turns
        terminal_cash = self._number(summary, "terminal_cash", "cash")
        pnl = self._number(summary, "realized_pnl", "terminal_pnl", "pnl")
        if not any(key in summary for key in ("realized_pnl", "terminal_pnl", "pnl")):
            pnl = terminal_cash - self.data.starting_cash
        metrics = {
            "terminal_cash": terminal_cash,
            "terminal_pnl": pnl,
            "remaining_position": self._number(summary, "remaining_position", "position"),
            "max_abs_position": self._number(summary, "max_abs_position"),
            "max_drawdown": self._number(summary, "max_drawdown", "drawdown"),
            "gross_traded_quantity": self._number(
                summary,
                "gross_traded_quantity",
                "gross_filled_quantity",
                "gross_quantity",
            ),
            "fill_count": self._number(summary, "fill_count", "fills"),
            "fees_paid": self._number(summary, "fees_paid"),
            "order_count": self._number(summary, "order_count", "orders"),
            "rejection_count": self._number(summary, "rejection_count", "rejections"),
            "order_rejection_count": self._number(summary, "order_rejection_count"),
            "cancel_rejection_count": self._number(summary, "cancel_rejection_count"),
            "margin_rejection_count": self._number(summary, "margin_rejection_count"),
            "margin_call_count": self._number(summary, "margin_call_count"),
            "margin_liquidation_count": self._number(summary, "margin_liquidation_count"),
            "margin_liquidated_quantity": self._number(summary, "margin_liquidated_quantity"),
            "strategy_fault_count": self._number(summary, "strategy_fault_count", "strategy_faults"),
            "deployment_count": self._number(summary, "deployment_count", "deployments"),
            "unique_strategy_version_count": self._number(summary, "unique_strategy_version_count"),
            "strategy_stop_count": self._number(summary, "strategy_stop_count", "strategy_stops"),
            "market_time_ns": self._number(summary, "market_time_ns", "market_time"),
            "voluntary_wait_ns": self._number(summary, "voluntary_wait_ns"),
            "charged_agent_time_ns": self._number(summary, "charged_agent_time_ns"),
            "model_turn_count": max(
                self._number(summary, "model_turn_count"),
                float(observed_model_turns),
            ),
            "prompt_tokens": float(prompt_tokens),
            "completion_tokens": float(completion_tokens),
            "cached_input_tokens": float(cached_input_tokens),
            "reasoning_tokens": float(reasoning_tokens),
            "liquidation_failed": float(self._liquidation_failed(summary)),
        }
        if reported_costs:
            metrics["inference_cost"] = float(sum(reported_costs))
        return metrics


class AlphaversePropTask(AlphaverseTask):
    """Shared-episode role with a research/deployment-only MCP surface."""

    @classmethod
    def toolsets(cls, config: AlphaverseTaskConfig) -> list[vf.Toolset]:
        return [AlphaversePropToolset(config.toolset)]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if self.config.toolset.url:
            if not self.data.join_episode_id:
                raise RuntimeError("prop role requires a joined episode id")
            trace.state.episode_id = self.data.join_episode_id
            trace.state.participant_id = PROP_PARTICIPANT_ID
            return
        await super().setup(trace, runtime)


class AlphaverseKnobPropTask(AlphaversePropTask):
    """Prop role whose MCP schema excludes arbitrary source deployment."""

    @classmethod
    def toolsets(cls, config: AlphaverseTaskConfig) -> list[vf.Toolset]:
        return [AlphaverseKnobPropToolset(config.toolset)]


class AlphaverseTaskset(vf.Taskset[AlphaverseTask, AlphaverseTasksetConfig]):
    """Construct deterministic seeded Alphaverse scenario tasks."""

    def load(self) -> list[AlphaverseTask]:
        config = self.config
        return [
            AlphaverseTask(
                AlphaverseData(
                    idx=index,
                    name=f"alphaverse-{config.seed + index}",
                    description="Agentic automated-trading episode",
                    prompt=_prompt_with_limits(
                        config.prompt,
                        config.max_market_time_ns,
                        config.model_turn_cap,
                    ),
                    scenario_seed=config.seed + index,
                    scenario_version=config.scenario_version,
                    latent_demand_profile=config.latent_demand_profile,
                    participant_id=config.participant_id,
                    starting_cash=config.starting_cash,
                    max_market_time_ns=config.max_market_time_ns,
                ),
                config.task,
            )
            for index in range(config.num_tasks)
        ]


__all__ = ["AlphaverseTaskset"]


if __name__ == "__main__":
    AlphaverseToolset.run()
