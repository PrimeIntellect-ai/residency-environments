"""Compose solo and cooperative prompts from explicit mode contracts."""

GAME_RULES = """# PATTERN DISCOVERY CARD GAME (ELEUSIS)

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

"""

SOLO_ACTION = """Each turn, make exactly one call to the `play` tool with both arguments:"""

TEAM_ACTION = """For a card play, call the `play` tool with both arguments:"""

PLAY_CONTRACT = """
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

"""

SOLO_SCORING = """## SCORING

You have {max_turns} turns. If your submitted hypothesis first matches the
hidden rule on turn T, you score {max_turns} + 1 - T points out of
{max_turns}. If you never match it, you score 0. Incorrect hypotheses are
never penalized, so submit your best rule every single turn: the earlier it
becomes correct, the higher your score.

"""

SYSTEM_PROMPT = GAME_RULES + SOLO_ACTION + PLAY_CONTRACT + SOLO_SCORING


def team_prompt(*, seat: int, num_agents: int, max_turns: int, max_calls: int, deadline_seconds: float) -> str:
    return (
        GAME_RULES + TEAM_ACTION + PLAY_CONTRACT + f"\n## TEAM\n\nAgent identifier: {seat + 1} of {num_agents}. "
        "All teammates face the same hidden rule in separate private games, initially "
        "with identical boards and hands. The team finishes when anyone solves it. "
        "Every teammate receives reward 1 if anyone solves, otherwise 0. "
        f"You have {max_turns} valid plays and at most {max_calls} "
        f"model calls. The team has {deadline_seconds:g} seconds after all agents are ready.\n"
        "You may call send_message(message) to broadcast text to all teammates. "
        "Queued messages arrive before their next model call. Messages do not play "
        "cards or submit hypotheses. You may send messages without playing a card. "
        "Make at most one play call per response.\n"
    )
