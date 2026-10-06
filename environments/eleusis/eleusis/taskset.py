"""Single-player Eleusis tasks.

One task = one hidden rule x one seeded deal. The model plays through the
`play(rule, card)` tool: every turn it tests one card and submits its current
best rule hypothesis, which is checked automatically by behavioral equivalence.
Reward is the normalized no-stakes score:
(max_turns + 1 - turns_used) / max_turns when solved, else 0.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Iterator

import verifiers.v1 as vf

from .engine import EleusisState, deal
from .prompts import SYSTEM_PROMPT
from .rules import DEFAULT_RULE_DATASET, Rule, load_rules
from .server import EleusisToolset


def _initial_prompt(starter: str, hand: list[str], max_turns: int) -> str:
    return (
        "New game.\n"
        f"Turn 0/{max_turns}.\n"
        f"Board: {starter}\n"
        f"Hand: {', '.join(hand)}\n"
        "Call play(rule, card) to take your first turn."
    )


def _round_seed(base: int, rule_code: str, round_index: int) -> int:
    digest = int(hashlib.md5(rule_code.encode()).hexdigest(), 16)
    return (base + digest + round_index) & 0xFFFFFFFF


def _interleave_families(rules: list[Rule]) -> list[Rule]:
    """Round-robin rules by family while preserving order within each family."""
    family_order: list[str] = []
    families: dict[str, deque[Rule]] = {}
    for rule in rules:
        family = rule.family or ""
        if family not in families:
            family_order.append(family)
            families[family] = deque()
        families[family].append(rule)

    interleaved: list[Rule] = []
    while any(families.values()):
        for family in family_order:
            if families[family]:
                interleaved.append(families[family].popleft())
    return interleaved


class EleusisData(vf.TaskData):
    rule_id: str
    rule_code: str
    round_index: int
    max_turns: int
    starter: str
    hand: list[str]
    draw_pile: list[str]
    num_agents: int = 1
    seat: int = 0


class EleusisTaskConfig(vf.TaskConfig):
    max_calls: int = 160


class EleusisTask(vf.Task[EleusisData, EleusisState, EleusisTaskConfig]):
    @classmethod
    def toolsets(cls, config) -> list[vf.Toolset]:
        from .server import EleusisToolsetConfig

        return [EleusisToolset(EleusisToolsetConfig())]

    async def finalize(self, trace, runtime=None) -> None:
        # Persist a structured per-turn game log for post-eval analysis (the raw
        # transcript is also there, but this avoids parsing free text). `state` is
        # excluded from serialized traces, so copy it into persisted `info`.
        trace.info["turn_log"] = list(getattr(trace.state, "turn_log", []))
        if self.data.num_agents > 1:
            trace.info["messages_sent"] = list(trace.state.outbox)

    @vf.stop
    async def game_over(self, trace: vf.Trace) -> bool:
        # The game gets its valid-play horizon plus a fixed allowance of 15
        # calls for invalid actions.
        if self.data.num_agents > 1:
            return (
                trace.state.game_over or trace.state.invalid_actions >= 15 or trace.num_turns >= self.config.max_calls
            )
        return trace.state.game_over or trace.num_turns >= self.data.max_turns + 15

    @vf.reward
    async def reward(self, trace: vf.Trace) -> float:
        """Normalized no-stakes score: (max_turns + 1 - turns_used) / max_turns."""
        state = trace.state
        if not state.solved:
            return 0.0
        return (self.data.max_turns + 1 - state.turn) / self.data.max_turns

    @vf.metric
    async def turns_saved(self, trace: vf.Trace) -> float:
        state = trace.state
        if not state.solved:
            return 0.0
        return float(self.data.max_turns + 1 - state.turn)

    @vf.metric
    async def solved(self, trace: vf.Trace) -> float:
        return float(trace.state.solved)

    @vf.metric
    async def turns_used(self, trace: vf.Trace) -> float:
        return float(trace.state.turn)

    @vf.metric
    async def invalid_actions(self, trace: vf.Trace) -> float:
        return float(trace.state.invalid_actions)

    @vf.metric
    async def early_abandonment(self, trace: vf.Trace) -> float:
        """Unsolved rollout that ended before the game exhausted its play budget.

        Provider and context failures are filtered by the analyzer; this metric
        captures the behavior needed to diagnose models that give up despite a
        remaining long-horizon budget.
        """
        state = trace.state
        return float(not state.solved and not state.game_over and state.turn < self.data.max_turns)

    @vf.metric
    async def exhausted_horizon(self, trace: vf.Trace) -> float:
        return float(not trace.state.solved and trace.state.turn >= self.data.max_turns)


class EleusisTeamTask(EleusisTask):
    @classmethod
    def toolsets(cls, config) -> list[vf.Toolset]:
        from .server import EleusisToolsetConfig
        from .team_server import EleusisTeamToolset

        return [EleusisTeamToolset(EleusisToolsetConfig())]


class EleusisConfig(vf.TasksetConfig):
    task: EleusisTaskConfig = EleusisTaskConfig()
    dataset: str = DEFAULT_RULE_DATASET
    """Hugging Face repository ID or local Dataset/DatasetDict path."""
    dataset_config: str | None = None
    """Optional Hugging Face dataset configuration name."""
    revision: str | None = None
    """Optional immutable Hub revision, tag, branch, or commit hash."""
    split: str = "test"
    """Rule split to evaluate; defaults to `test`."""
    rounds_per_rule: int = 1
    """Seeded deals per rule."""
    max_turns: int = 100
    """Valid game turns available in each round."""
    hand_size: int = 12
    seed: int = 20260812


class EleusisTaskset(vf.Taskset[EleusisTask, EleusisConfig]):
    def load(self) -> Iterator[EleusisTask]:
        rules = load_rules(
            self.config.dataset,
            self.config.revision,
            self.config.dataset_config,
        )
        if self.config.split not in rules:
            raise ValueError(
                f"Split {self.config.split!r} not in {self.config.dataset!r} (available: {sorted(rules)})."
            )
        idx = 0
        for round_index in range(self.config.rounds_per_rule):
            for rule in _interleave_families(rules[self.config.split]):
                seed = _round_seed(self.config.seed, rule.code, round_index)
                starter, hand, draw_pile = deal(
                    rule.code,
                    seed,
                    self.config.hand_size,
                    min_playable_turns=self.config.max_turns,
                )
                yield EleusisTask(
                    EleusisData(
                        idx=idx,
                        name=f"{rule.rule_id}/r{round_index}",
                        network_allow=[],
                        prompt=_initial_prompt(starter, hand, self.config.max_turns),
                        system_prompt=SYSTEM_PROMPT.replace("{max_turns}", str(self.config.max_turns)),
                        rule_id=rule.rule_id,
                        rule_code=rule.code,
                        round_index=round_index,
                        max_turns=self.config.max_turns,
                        starter=starter,
                        hand=hand,
                        draw_pile=draw_pile,
                    ),
                    self.config.task,
                )
                idx += 1
