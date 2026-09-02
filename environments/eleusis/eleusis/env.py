"""Eleusis v1 environment (verifiers 0.3.x).

0.3.x moved from a bare taskset+harness pair to an `Env` that bundles them.
The single-agent case is exactly Eleusis: one `play(rule, card)` policy seat
over the seed taskset, using the bundled EleusisHarness (the standard
null-harness program) and the task's `play` toolset. The interaction protocol
is unchanged from the frozen calibration: automatic tool selection, tool
results flow directly into the next call, no injected messages.
"""

from __future__ import annotations

import verifiers.v1 as vf
from verifiers.v1.envs.single_agent.env import SingleAgentEnvConfig


class EleusisEnvConfig(SingleAgentEnvConfig):
    """One agent seat over the Eleusis taskset (narrowed by id "eleusis")."""


class EleusisEnv(vf.Env[EleusisEnvConfig]):
    """Benchmark-shaped single-player Eleusis env for verifiers 0.3.x."""

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        await agents.agent.run(task)
