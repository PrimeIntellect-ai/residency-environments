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
from .rules import DEFAULT_RULE_DATASET, Rule, load_rules
from .server import EleusisToolset

SYSTEM_PROMPT = """# PATTERN DISCOVERY CARD GAME (ELEUSIS)

A hidden, deterministic rule decides whether each card you play is accepted
onto the mainline or rejected to a sideline. Discover the rule.

## THE CARDS

The game uses two standard 52-card decks shuffled together (104 cards, so
duplicate cards exist). If needed to reach the configured turn limit, another
shuffled two-deck shoe supplies additional draws.
- Ranks: A=1, 2-10, J=11, Q=12, K=13.
- Suits: H hearts and D diamonds are red; C clubs and S spades are black.
- A card symbol is rank followed by suit, e.g. AH, 10S, QD.

## THE BOARD

- Mainline: the sequence of accepted cards, oldest first. God starts the game
  by placing one accepted starter card on the mainline; it satisfies the
  hidden rule and is public evidence. It is not from your hand.
- Sidelines: rejected cards, shown in brackets immediately after the mainline
  card they were played after. Example board: `AH 5C [2S] [9C] 7H` means 2S
  and 9C were rejected while 5C was the last accepted card.
- Hand: your private cards. After every play, accepted or rejected, the played
  card leaves your hand and you draw one replacement while the deck lasts.

## THE HIDDEN RULE

- Deterministic and objective: for any candidate card and any mainline state
  it answers accept or reject, unambiguously.
- It uses only visible information: the candidate card and the accepted cards
  already on the mainline. It never depends on rejected cards, deck order, or
  the contents of your hand.
- It is simple enough to state in one sentence. It may be static (depends only
  on the candidate card) or relational (depends on previous accepted cards or
  on the candidate's position in the accepted sequence).
- Relational rules may use the last few accepted cards, fixed-size groups, a
  repeating position pattern, or a simple summary of the accepted mainline.
- A rule may also choose between simple relations using a property of the
  previous card or a history summary, or compose two such conditions. These are
  still deterministic rules over only the candidate and accepted mainline.

Examples of possible rules: "the card must differ in color from the previous
card", "only hearts and spades", "the rank must be within 2 of the previous
card's rank", "cards come in suit pairs".

## YOUR ACTION

Each turn, make exactly one call to the `play` tool with both arguments:
- `card`: one card from your hand. God tests it against the hidden rule.
  Accepted cards join the mainline; rejected cards go to the sideline. Either
  way the card leaves your hand and you draw a replacement.
- `rule`: your current best hypothesis for the hidden rule, written as a
  Python boolean expression (see below). It is checked automatically every
  turn at no cost: if it matches the hidden rule the game ends and you score;
  otherwise the result says only that it was incorrect and play continues.
  Incorrect hypotheses are never penalized.

The tool result is deliberately compact. A valid play reports only the turn,
whether the card was accepted or rejected, any replacement card drawn, and
whether the rule hypothesis was correct. It does not repeat the board or hand.
Maintain the live state from the initial deal and prior tool results:
- On acceptance, append the played card to the mainline.
- On rejection, attach it to the sideline after the current last mainline card.
- In either case, remove one copy of the played card from your hand and add the
  drawn card.
- An invalid card consumes no turn and changes no state.

## RULE EXPRESSIONS

Write `rule` as a boolean expression over `card` and optionally `mainline`
(the list of accepted Card objects, oldest first):
- card.color is "red" or "black"
- card.suit is "hearts", "diamonds", "clubs", or "spades"
- card.suit_symbol is "H", "D", "C", or "S"
- card.rank is numeric: A=1, ..., J=11, Q=12, K=13
- card.rank_label is "A", "2", ..., "10", "J", "Q", or "K"
- card.is_face is True for J, Q, K

Examples:
- card.color == "red"
- card.rank % 2 == 0
- card.suit in {"hearts", "spades"}
- card.rank >= 8 and card.color == "black"
- not mainline or card.color != mainline[-1].color
- not mainline or abs(card.rank - mainline[-1].rank) <= 2
- not mainline or (card.suit == mainline[-1].suit if len(mainline) % 2 == 1 else card.suit != mainline[-1].suit)

Multi-line function bodies using `return` are also accepted. Your hypothesis
is judged by behavioral equivalence — whether it produces the same accepts and
rejects as the hidden rule across board situations — not by string match, so
any equivalent formulation counts. A malformed or unsafe expression is simply
an incorrect hypothesis; the card play still counts.

## SCORING

You have {max_turns} turns. If your submitted hypothesis first matches the
hidden rule on turn T, you score {max_turns} + 1 - T points out of
{max_turns}. If you never match it, you score 0. Incorrect hypotheses are
never penalized, so submit your best rule every single turn: the earlier it
becomes correct, the higher your score.

"""


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


class EleusisTask(vf.Task[EleusisData, EleusisState]):
    @classmethod
    def toolsets(cls, config) -> list[vf.Toolset]:
        # verifiers 0.3.x: `toolsets(cls, config)` classmethod; the toolset is
        # constructed with its config (local subprocess by default).
        return [EleusisToolset(vf.ToolsetConfig())]

    async def finalize(self, trace, runtime=None) -> None:
        # Persist a structured per-turn game log for post-eval analysis (the raw
        # transcript is also there, but this avoids parsing free text). `state` is
        # excluded from serialized traces, so copy it into persisted `info`.
        trace.info["turn_log"] = list(getattr(trace.state, "turn_log", []))

    @vf.stop
    async def game_over(self, trace: vf.Trace) -> bool:
        # The game gets its valid-play horizon plus a fixed allowance of 15
        # calls for invalid actions.
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


class EleusisConfig(vf.TasksetConfig):
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
