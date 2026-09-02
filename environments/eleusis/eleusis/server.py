"""The per-rollout tool server: one bare `play` tool over the game engine."""

from __future__ import annotations

import asyncio

import verifiers.v1 as vf

from .engine import EleusisState, init_state, play_turn
from .rules import IsolatedRuleChecker, compile_python_rule


class EleusisToolsetConfig(vf.ToolsetConfig):
    pass


class EleusisToolset(vf.Toolset[EleusisToolsetConfig, EleusisState]):
    TOOL_PREFIX = None  # advertise the tool as bare `play`

    async def setup_task(self, task) -> None:
        self._task = task
        self._target = compile_python_rule(task.rule_code)
        self._rule_checker = IsolatedRuleChecker(task.rule_code)
        self._exit_stack.callback(self._rule_checker.close)

    @vf.tool
    async def play(self, rule: str, card: str) -> str:
        """Play one card from your hand and submit your current best rule hypothesis.

        Args:
            rule: Your current best hypothesis for the hidden rule, as a Python
                boolean expression over `card` and `mainline`. Checked every
                turn; if it matches the hidden rule the game ends successfully.
            card: One card from your hand, e.g. "AH" or "10S". The card is
                tested against the hidden rule, leaves your hand, and is
                replaced by a draw from the deck.
        """
        if not self.state.initialized:
            init_state(
                self.state,
                starter=self._task.starter,
                hand=self._task.hand,
                draw_pile=self._task.draw_pile,
            )
        hypothesis_correct = await asyncio.to_thread(
            self._rule_checker.matches,
            rule,
            self.state.observations,
        )
        return play_turn(
            self.state,
            target=self._target,
            rule_matches_target=lambda _: hypothesis_correct,
            rule=rule,
            card=card,
            max_turns=self._task.max_turns,
        )


if __name__ == "__main__":
    EleusisToolset.run()
