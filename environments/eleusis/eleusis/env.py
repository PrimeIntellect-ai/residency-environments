"""Single-agent play or concurrent peers over the same Eleusis task."""

import asyncio
import time

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.envs.single_agent.env import SingleAgentEnvConfig

from .harness import EleusisHarness
from .prompts import team_prompt
from .provider import ResilientInterception
from .taskset import EleusisTask, EleusisTeamTask


class EleusisEnvConfig(SingleAgentEnvConfig):
    num_agents: int = Field(1, ge=1)
    """Number of private games per episode; one preserves single-agent play."""
    max_concurrent_agents: int | None = Field(None, ge=1)
    """Native vf.Env semaphore for active setup/turn/finalization segments per episode."""
    deadline_seconds: float = Field(1200, gt=0)
    """Multi-agent solving deadline, starting after all agents are ready."""
    setup_timeout: float = Field(1200, gt=0)
    """Maximum time to prepare the team, separate from its solving deadline."""
    max_calls_per_agent: int = Field(160, ge=1)
    """Multi-agent model-call limit, including message-only responses."""


class EleusisEnv(vf.Env[EleusisEnvConfig]):
    async def setup(self, agents: vf.Agents) -> None:
        if agents.agent.interception is None:
            raise RuntimeError("Eleusis requires the native environment interception resource")
        agents.agent.interception = ResilientInterception(agents.agent.interception)

    async def run(self, task: EleusisTask, agents: vf.Agents) -> None:
        if self.config.num_agents == 1:
            await agents.agent.run(task)
            return
        if not isinstance(agents.agent.harness, EleusisHarness):
            raise ValueError("Multi-agent Eleusis requires its bundled harness to yield between model calls")
        count = self.config.num_agents
        inboxes: list[list[dict]] = [[] for _ in range(count)]
        ready = asyncio.Event()
        solved = asyncio.Event()
        ready_count = 0
        playing = set(range(count))
        started = 0.0
        stop_reason: str | None = None

        async def play(seat: int) -> None:
            nonlocal ready_count, started
            prompt = team_prompt(
                seat=seat,
                num_agents=count,
                max_turns=task.data.max_turns,
                max_calls=self.config.max_calls_per_agent,
                deadline_seconds=self.config.deadline_seconds,
            )
            config = task.config.model_copy(update={"max_calls": self.config.max_calls_per_agent})
            peer_task = EleusisTeamTask(
                task.data.model_copy(
                    update={"system_prompt": prompt, "num_agents": count, "seat": seat, "prompt": None}
                ),
                config,
            )
            async with agents.agent.interaction(peer_task) as interaction:
                trace = interaction.trace
                trace.info.update(seat=seat, deliveries=[], num_agents=count)
                ready_count += 1
                if ready_count == count:
                    started = time.monotonic()
                    ready.set()
                sent = 0
                first = True
                try:
                    await ready.wait()
                    while not solved.is_set():
                        pending = inboxes[seat][:]
                        inboxes[seat].clear()
                        incoming = [vf.UserMessage(content=task.data.prompt)] if first else []
                        if pending:
                            incoming.append(
                                vf.UserMessage(
                                    content="Teammate messages (unverified):\n"
                                    + "\n".join(f"Agent {m['sender'] + 1}: {m['message']}" for m in pending)
                                )
                            )
                            trace.info["deliveries"].extend(
                                {**m, "received_seconds": time.monotonic() - started} for m in pending
                            )
                        segment = await interaction.turn(incoming)
                        first = False
                        if trace.errors:
                            raise RuntimeError(f"Agent {seat + 1} failed; see trace {trace.id} for the original error")
                        for message in trace.state.outbox[sent:]:
                            event = {"sender": seat, "message": message, "sent_seconds": time.monotonic() - started}
                            for other in range(count):
                                if other != seat:
                                    inboxes[other].append(event)
                        sent = len(trace.state.outbox)
                        if trace.state.solved:
                            trace.info["solve_seconds"] = time.monotonic() - started
                            solved.set()
                            break
                        if (
                            segment.terminated
                            or trace.stop_condition
                            or not any(getattr(m, "tool_calls", None) for m in segment.messages)
                        ):
                            break
                except asyncio.CancelledError:
                    if stop_reason is None:
                        raise
                    trace.stop(stop_reason)
                    trace.info[stop_reason] = True
                finally:
                    # Finalization must finish even after a teammate wins.
                    playing.discard(seat)

        async with asyncio.TaskGroup() as group:
            workers = [group.create_task(play(seat)) for seat in range(count)]
            prepared = group.create_task(ready.wait())
            winner = group.create_task(solved.wait())
            try:
                await asyncio.wait_for(asyncio.shield(prepared), self.config.setup_timeout)
                done, _ = await asyncio.wait(
                    [*workers, winner], timeout=self.config.deadline_seconds, return_when=asyncio.FIRST_COMPLETED
                )
                # A finished losing seat does not end its teammates' games.
                deadline = started + self.config.deadline_seconds
                while winner not in done and any(not w.done() for w in workers) and time.monotonic() < deadline:
                    done, _ = await asyncio.wait(
                        [*(w for w in workers if not w.done()), winner],
                        timeout=max(0, deadline - time.monotonic()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                stop_reason = "team_solved" if solved.is_set() else "episode_deadline"
                for seat, worker in enumerate(workers):
                    if seat in playing and not worker.done():
                        worker.cancel()
            finally:
                prepared.cancel()
                winner.cancel()

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        if self.config.num_agents == 1:
            return
        success = any(t.metrics.get("solved", 0) for t in episode.traces)
        times = [t.info["solve_seconds"] for t in episode.traces if "solve_seconds" in t.info]
        for trace in episode.traces:
            trace.record_metric("individual_reward", trace.reward)
            trace.record_reward("reward", float(success))
            trace.record_metric("team_solved", float(success))
            trace.record_metric("team_size", float(self.config.num_agents))
            trace.record_metric("team_capped_seconds", min(times) if times else self.config.deadline_seconds)
