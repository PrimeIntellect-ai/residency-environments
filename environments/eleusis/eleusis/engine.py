"""The Eleusis game engine: dealing, state, and one turn of play.

Pure logic, shared by the tool server (which mutates the rollout state) and the
tests. Dealing is fully deterministic for a given seed: shuffle two standard
decks with the round's seed, deal a 12-card hand, then draw from the deck until a
card satisfies the hidden rule on an empty mainline — that card is God's
starter; non-qualifying draws are discarded silently.  Long-horizon games
append independently shuffled two-deck reserve shoes when the primary shoe
cannot supply the configured number of plays.
"""

from __future__ import annotations

import ast
import random
from typing import Callable

import verifiers.v1 as vf

from .cards import double_deck, parse_card
from .rules import CompiledRule, compile_python_rule


def _rule_ast_complexity(source: str) -> int | None:
    """Rough AST node count of a submitted rule string (post-eval complexity analysis);
    None when the rule is not valid Python."""
    try:
        return len(list(ast.walk(ast.parse(str(source or "").strip(), mode="exec"))))
    except Exception:
        return None


class EleusisState(vf.State):
    initialized: bool = False
    mainline: list[str] = []
    sidelines: list[list[str]] = []
    hand: list[str] = []
    draw_pile: list[str] = []
    turn: int = 0
    solved: bool = False
    game_over: bool = False
    invalid_actions: int = 0
    observations: list[tuple[str, list[str], bool]] = []
    turn_log: list[dict] = []


def deal(
    rule_code: str,
    seed: int,
    hand_size: int = 12,
    min_playable_turns: int | None = None,
) -> tuple[str, list[str], list[str]]:
    """Return a deterministic ``(starter, hand, draw_pile)``.

    The primary deal is deterministic and stable across versions.
    If ``min_playable_turns`` exceeds the remaining primary supply, complete,
    independently seeded two-deck reserve shoes are appended.  Appending rather
    than reshuffling preserves the starter, hand, and entire primary-shoe prefix.
    """
    if min_playable_turns is not None and min_playable_turns < 1:
        raise ValueError("min_playable_turns must be positive when provided.")
    target = compile_python_rule(rule_code)
    cards = double_deck()
    random.Random(seed).shuffle(cards)
    hand, pile = cards[:hand_size], cards[hand_size:]
    while pile:
        candidate = pile.pop(0)
        if target(parse_card(candidate), []):
            reserve_index = 0
            while min_playable_turns is not None and len(hand) + len(pile) < min_playable_turns:
                reserve_index += 1
                reserve = double_deck()
                # A fixed integer mix keeps reserve shoes independent and stable
                # across Python processes without relying on randomized hashes.
                reserve_seed = (seed ^ 0xE1E515 ^ (reserve_index * 0x9E3779B9)) & 0xFFFFFFFF
                random.Random(reserve_seed).shuffle(reserve)
                pile.extend(reserve)
            return candidate, hand, pile
    raise ValueError("Rule accepts no starter card in a double deck.")


def init_state(state: EleusisState, *, starter: str, hand: list[str], draw_pile: list[str]) -> None:
    state.initialized = True
    state.mainline = [starter]
    state.sidelines = [[]]
    state.hand = list(hand)
    state.draw_pile = list(draw_pile)


def board_str(state: EleusisState) -> str:
    parts: list[str] = []
    for card, rejects in zip(state.mainline, state.sidelines):
        parts.append(card)
        parts.extend(f"[{reject}]" for reject in rejects)
    return " ".join(parts)


def play_turn(
    state: EleusisState,
    *,
    target: CompiledRule,
    rule_matches_target: Callable[[str], bool],
    rule: str,
    card: str,
    max_turns: int,
) -> str:
    """Resolve one play and return the minimal state delta."""
    if state.game_over:
        return "Game over."

    try:
        symbol = parse_card(card).symbol
    except ValueError:
        symbol = None
    if symbol is None or symbol not in state.hand:
        state.invalid_actions += 1
        state.turn_log.append(
            {
                "turn": state.turn,
                "card": str(card) if card is not None else None,
                "accepted": None,
                "drew": None,
                "rule": rule if rule else None,
                "correct": False,
                "rule_nodes": _rule_ast_complexity(rule) if rule else None,
                "hand": None,
                "mainline": None,
                "invalid": True,
            }
        )
        return "Invalid card; choose a card in your hand. No turn used."

    state.turn += 1
    mainline_cards = [parse_card(item) for item in state.mainline]
    accepted = bool(target(parse_card(symbol), mainline_cards))
    state.observations.append((symbol, list(state.mainline), accepted))
    state.hand.remove(symbol)
    if accepted:
        state.mainline.append(symbol)
        state.sidelines.append([])
    else:
        state.sidelines[-1].append(symbol)
    drawn = ""
    if state.draw_pile:
        drawn = state.draw_pile.pop(0)
        state.hand.append(drawn)

    hypothesis_correct = rule_matches_target(rule)
    state.turn_log.append(
        {
            "turn": state.turn,
            "card": symbol,
            "accepted": accepted,
            "drew": drawn or None,
            "rule": rule,
            "correct": hypothesis_correct,
            "rule_nodes": _rule_ast_complexity(rule),
            "hand": list(state.hand),
            "mainline": list(state.mainline),
            "invalid": False,
        }
    )
    result = f"{state.turn}/{max_turns}: {symbol} {'accepted' if accepted else 'rejected'}"
    if drawn:
        result += f"; drew {drawn}"

    if hypothesis_correct:
        state.solved = True
        state.game_over = True
        return f"{result}; rule correct. Game over."

    result += "; rule incorrect."
    if state.turn >= max_turns:
        state.game_over = True
        return f"{result} Turn limit reached; game over."

    return result
