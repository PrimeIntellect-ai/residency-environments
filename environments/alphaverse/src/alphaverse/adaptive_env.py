"""Two-role Verifiers environment with scheduled prop research windows."""

from __future__ import annotations

import asyncio
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import verifiers.v1 as vf
from pydantic import Field

from alphaverse.artifact_egress import call_framework, call_mcp_tool
from alphaverse.opponent_roster import (
    OpponentRoster,
    legacy_prop_roster_id,
    opponent_roster,
)
from alphaverse.prop_trader import PROP_PARTICIPANT_ID, prop_system_prompt
from alphaverse.verifiers_v1 import (
    AlphaverseData,
    AlphaverseKnobPropTask,
    AlphaversePropTask,
    AlphaverseTask,
)

PLAYER_SESSION_RULES = """

This episode is divided into scheduled virtual trading sessions of {duration_ns}
ns ({duration_seconds:g} seconds) each, separated by research intermissions.
During an open session, market time behaves normally and you may research or deploy
at any time. At the session boundary the whole market freezes: no matching,
participant callbacks, margin processing, or latent events occur. When a wait
reports `market_session.state = "intermission"`, finish the current response
promptly. You will then receive a dedicated intermission turn in which you can
analyze data and stage a deployment for the next session. Market time does not
advance during that turn. Do not try to reopen the exchange yourself.

Other market participants may change their behavior, enter, or exit during the
episode. Their identities, update schedules, and decision processes are not
disclosed.
"""


class AlphaverseAdaptiveEnvConfig(vf.EnvConfig):
    player: vf.AgentConfig = vf.AgentConfig()
    prop: vf.AgentConfig = vf.AgentConfig()
    prop_mode: Literal["none", "static", "knobs", "adaptive"] = "adaptive"
    prop_seed_profile: Literal["passive", "competitive"] = "passive"
    prop_framing: Literal["incumbent", "neutral", "arbitrary"] = "incumbent"
    opponent_roster_id: str | None = None
    max_open_segments_per_session: int = Field(default=4, ge=1)
    """Player segments before the host advances an otherwise-idle session."""


class AlphaverseAdaptiveEnv(vf.Env[AlphaverseAdaptiveEnvConfig]):
    """Run matched scheduled-session arms with an optional prop controller."""

    async def setup(self, agents: vf.Agents) -> None:
        agents.player.trainable = True
        agents.prop.trainable = False

    @staticmethod
    def _player_task(task: AlphaverseTask) -> AlphaverseTask:
        duration_ns = task.config.session_duration_ns
        if duration_ns is None:
            raise ValueError("adaptive player task requires session_duration_ns")
        rules = PLAYER_SESSION_RULES.format(
            duration_ns=duration_ns,
            duration_seconds=duration_ns / 1_000_000_000,
        )
        prompt = str(task.data.prompt or "").rstrip() + rules
        data = task.data.model_copy(update={"prompt": prompt})
        return AlphaverseTask(data, task.config)

    @staticmethod
    def _episode_task(
        task: AlphaverseTask,
        prop_mode: Literal["none", "static", "knobs", "adaptive"],
        prop_seed_profile: Literal["passive", "competitive"] = "passive",
        opponent_roster_id: str | None = None,
    ) -> AlphaverseTask:
        """Bind participant creation to the experimental arm."""

        roster_id = opponent_roster_id or legacy_prop_roster_id(
            adaptive_prop=prop_mode != "none",
            seed_profile=prop_seed_profile,
            control_scope=("knobs" if prop_mode == "knobs" else "full_source"),
            static=prop_mode == "static",
        )
        roster = opponent_roster(roster_id)
        first_slot = roster.slots[0] if roster.slots else None

        config = task.config.model_copy(
            update={
                "adaptive_prop": bool(roster.slots),
                "prop_seed_profile": (first_slot.seed_profile if first_slot else prop_seed_profile),
                "prop_control_scope": (
                    first_slot.control_scope if first_slot and first_slot.control_scope != "static" else "full_source"
                ),
                "opponent_roster_id": roster_id,
            }
        )
        data = task.data.model_copy(
            update={
                "prop_seed_profile": (first_slot.seed_profile if first_slot else prop_seed_profile),
                "prop_control_scope": (
                    first_slot.control_scope if first_slot and first_slot.control_scope != "static" else "full_source"
                ),
                "opponent_roster_id": roster_id,
            }
        )
        return AlphaverseTask(data, config)

    @staticmethod
    def _prop_task(
        task: AlphaverseTask,
        episode_id: str,
        *,
        toolset_url: str | None = None,
        prop_seed_profile: Literal["passive", "competitive"] = "passive",
        prop_framing: Literal["incumbent", "neutral", "arbitrary"] = "incumbent",
        prop_control_scope: Literal["full_source", "knobs"] = "full_source",
        participant_id: str = PROP_PARTICIPANT_ID,
        opponent_roster_id: str | None = None,
    ) -> AlphaversePropTask:
        data = AlphaverseData(
            idx=task.data.idx,
            name=f"{task.data.name}-{participant_id}",
            description="Scheduled adaptive proprietary-trader controller",
            prompt=None,
            system_prompt=prop_system_prompt(prop_framing, prop_control_scope),
            scenario_seed=task.data.scenario_seed,
            scenario_version=task.data.scenario_version,
            latent_demand_profile=task.data.latent_demand_profile,
            participant_id=participant_id,
            starting_cash=task.data.starting_cash,
            max_market_time_ns=task.data.max_market_time_ns,
            join_episode_id=episode_id,
            prop_seed_profile=prop_seed_profile,
            prop_framing=prop_framing,
            prop_control_scope=prop_control_scope,
            opponent_roster_id=opponent_roster_id,
        )
        config = task.config
        if toolset_url is not None:
            config = config.model_copy(update={"toolset": config.toolset.model_copy(update={"url": toolset_url})})
        task_type = AlphaverseKnobPropTask if prop_control_scope == "knobs" else AlphaversePropTask
        return task_type(data, config)

    @staticmethod
    def _episode_id(interaction) -> str:
        episode_id = getattr(interaction.trace.state, "episode_id", None)
        if not isinstance(episode_id, str) or not episode_id:
            raise RuntimeError("adaptive episode did not initialize")
        return episode_id

    @staticmethod
    def _embedded_toolset_url(interaction) -> str:
        url = getattr(interaction.trace.state, "toolset_url", None)
        if not isinstance(url, str) or not url:
            raise RuntimeError("embedded adaptive Toolset URL is unavailable")
        return url

    @classmethod
    def _embedded_prop_url(cls, interaction, participant_id: str = PROP_PARTICIPANT_ID) -> str:
        url = cls._embedded_toolset_url(interaction)
        token = getattr(interaction.trace.state, "prop_access_token", None)
        if not isinstance(token, str) or not token:
            raise RuntimeError("embedded prop role capability is unavailable")
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.update(
            {
                "alphaverse_role": participant_id,
                "alphaverse_role_token": token,
            }
        )
        return urlunsplit(parts._replace(query=urlencode(query)))

    @classmethod
    async def _embedded_call(
        cls,
        interaction,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return await call_mcp_tool(cls._embedded_toolset_url(interaction), name, arguments)

    @staticmethod
    def _coordinator_token(interaction) -> str:
        token = getattr(interaction.trace.state, "coordination_token", None)
        if not isinstance(token, str) or not token:
            raise RuntimeError("embedded evaluator control capability is unavailable")
        return token

    @classmethod
    async def _framework_call(
        cls,
        interaction,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return await call_framework(
            cls._embedded_toolset_url(interaction),
            cls._coordinator_token(interaction),
            request,
        )

    async def _status(
        self,
        player,
    ) -> dict[str, Any]:
        return await self._embedded_call(player, "session_status", {})

    async def _advance_to_close(
        self,
        status: dict[str, Any],
        player,
    ) -> dict[str, Any]:
        end = status.get("session_end_ns")
        now = status.get("market_time_ns")
        if not isinstance(end, int) or not isinstance(now, int):
            raise RuntimeError("market session status omitted its boundary")
        if end > now:
            body = {
                "until_ns": end,
                "duration_ns": None,
                "wake_on_alert": False,
            }
            await self._embedded_call(player, "wait", body)
        return await self._status(player)

    async def _run_interactions(
        self,
        task: AlphaverseTask,
        player,
        *,
        prop=None,
        initial_player_segment=None,
        prop_participant_id: str = PROP_PARTICIPANT_ID,
    ) -> None:
        session_records: list[dict[str, Any]] = []
        forced_advances = 0
        prop_active = prop is not None
        player_segment = initial_player_segment or await player.turn()
        open_segments = 1
        while not player_segment.terminated:
            status = await self._status(player)
            state = status.get("state")
            if state == "finalized":
                break
            if state == "open":
                if open_segments >= self.config.max_open_segments_per_session:
                    status = await self._advance_to_close(
                        status,
                        player,
                    )
                    forced_advances += 1
                else:
                    player_segment = await player.turn(
                        "The current trading session is still open. Continue "
                        "playing. Use the market wait tool when you want your "
                        "deployed strategy to run, and yield after the market "
                        "reports an intermission."
                    )
                    open_segments += 1
                    continue
            if status.get("state") != "intermission":
                continue

            session_index = int(status.get("session_index", 0))
            market_time = int(status.get("market_time_ns", 0))
            player_turn = player.turn(
                f"Research intermission after trading session "
                f"{session_index}. Market time is frozen at "
                f"{market_time} ns. Review available data and optionally "
                "stage a strategy for the next session. Finish your response "
                "when ready to reopen."
            )
            if prop_active:
                assert prop is not None
                player_review, prop_review = await asyncio.gather(
                    player_turn,
                    prop.turn(
                        f"Deployment window {session_index} is open at market "
                        f"time {market_time} ns. Review the completed session, "
                        "continue any prior research, and optionally stage a "
                        "valid deployment for the next session. Finish your "
                        "response when ready."
                    ),
                )
                prop_active = not prop_review.terminated
                prop_reply: str | None = prop_review.last_reply
                prop_terminated: bool | None = prop_review.terminated
            else:
                player_review = await player_turn
                prop_reply = None
                prop_terminated = None
            session_records.append(
                {
                    "prop_mode": self.config.prop_mode,
                    "session_index": session_index,
                    "market_time_ns": market_time,
                    "player_reply": player_review.last_reply,
                    "prop_reply": prop_reply,
                    "player_terminated": player_review.terminated,
                    "prop_terminated": prop_terminated,
                }
            )
            if player_review.terminated:
                break
            status = await self._status(player)
            if status.get("state") == "finalized":
                break
            await self._framework_call(
                player,
                {
                    "operation": "resume",
                },
            )
            player_segment = await player.turn(
                f"Trading session {session_index + 1} is now open. Staged "
                "strategies have activated. Continue playing until the next "
                "intermission or until you terminate the episode."
            )
            open_segments = 1

        if prop is not None and isinstance(player.trace.state.terminal_summary, dict):
            prop_summary = await self._framework_call(
                player,
                {
                    "operation": "participant_result",
                    "participant_id": prop_participant_id,
                },
            )
            prop_summary["shared_episode_trace_id"] = player.trace.id
            prop.trace.state.terminal_summary = prop_summary
            prop.trace.state.terminated = True

        player.trace.info["prop_mode"] = self.config.prop_mode
        player.trace.info["prop_seed_profile"] = self.config.prop_seed_profile
        player.trace.info["prop_framing"] = self.config.prop_framing
        player.trace.info["prop_control_scope"] = "knobs" if self.config.prop_mode == "knobs" else "full_source"
        player.trace.info["adaptive_sessions"] = list(session_records)
        player.trace.info["opponent_roster_id"] = task.config.opponent_roster_id
        player.trace.info["forced_session_advances"] = forced_advances
        if prop is not None:
            prop.trace.info["prop_mode"] = self.config.prop_mode
            prop.trace.info["prop_seed_profile"] = self.config.prop_seed_profile
            prop.trace.info["prop_framing"] = self.config.prop_framing
            prop.trace.info["prop_control_scope"] = "knobs" if self.config.prop_mode == "knobs" else "full_source"
            prop.trace.info["adaptive_sessions"] = list(session_records)
            prop.trace.info["opponent_roster_id"] = task.config.opponent_roster_id
            prop.trace.info["forced_session_advances"] = forced_advances
            prop.trace.info["adaptive_controller_active_at_end"] = prop_active

    async def run(self, task: AlphaverseTask, agents: vf.Agents) -> None:
        if task.config.session_duration_ns is None:
            raise ValueError("scheduled env requires session_duration_ns")
        roster_id = self.config.opponent_roster_id or legacy_prop_roster_id(
            adaptive_prop=self.config.prop_mode != "none",
            seed_profile=self.config.prop_seed_profile,
            control_scope=("knobs" if self.config.prop_mode == "knobs" else "full_source"),
            static=self.config.prop_mode == "static",
        )
        roster: OpponentRoster = opponent_roster(roster_id)
        controlled_slots = roster.controlled_slots
        if len(controlled_slots) > 1:
            raise ValueError(
                "the current Verifiers coordinator supports one adaptive "
                "opponent; the roster runtime already supports multiple slots"
            )
        controlled_slot = controlled_slots[0] if controlled_slots else None
        if controlled_slot and controlled_slot.participant_id != PROP_PARTICIPANT_ID:
            raise ValueError("the current Verifiers role capability is bound to participant 'prop'")
        episode_task = self._episode_task(
            task,
            self.config.prop_mode,
            self.config.prop_seed_profile,
            roster_id,
        )
        player_task = self._player_task(episode_task)

        async with agents.player.interaction(player_task) as player:
            episode_id = self._episode_id(player)
            if controlled_slot is not None:
                initial_player_segment = await player.turn()
                prop_toolset_url = self._embedded_prop_url(player, controlled_slot.participant_id)
                prop_task = self._prop_task(
                    episode_task,
                    episode_id,
                    toolset_url=prop_toolset_url,
                    prop_seed_profile=controlled_slot.seed_profile,
                    prop_framing=controlled_slot.framing,
                    prop_control_scope=controlled_slot.control_scope,
                    participant_id=controlled_slot.participant_id,
                    opponent_roster_id=roster_id,
                )
                async with agents.prop.interaction(prop_task) as prop:
                    await self._run_interactions(
                        episode_task,
                        player,
                        prop=prop,
                        initial_player_segment=initial_player_segment,
                        prop_participant_id=controlled_slot.participant_id,
                    )
            else:
                await self._run_interactions(episode_task, player)


__all__ = ["AlphaverseAdaptiveEnv"]
