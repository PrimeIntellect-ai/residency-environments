#!/usr/bin/env python
"""Generate and validate the Eleusis frontier-calibration candidate bank.

The generator produces reviewed, parameterized Python-rule templates rather
than arbitrary programs.  Its first-stage output is intentionally larger than
the eventual benchmark: model calibration selects the final test rules.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from eleusis.cards import deck, parse_card
from eleusis.rules import (
    DEFAULT_RULE_DATASET,
    compile_python_rule,
    representative_mainlines,
    rule_extension_signature,
)

VERSION = "v0.1-frontier-candidates-20260809"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "calibrated_rules" / "candidates"
CORE_COLUMNS = ("rule_id", "label", "family", "code")
META_DEFAULTS: dict[str, Any] = {
    "description": "",
    "template_id": "",
    "template_parameters": "{}",
    "component_families": "[]",
    "split_type": "",
    "semantic_features": "{}",
    "history_depth": 0,
    "period_or_chunk_length": 0,
    "branch_count": 0,
    "attribute_cardinality": 0,
    "ast_nodes": 0,
    "acceptance_rate": 0.0,
    "acceptance_entropy": 0.0,
    "signature_hash": "",
    "designed_level": 0,
    "frontier_solve_rate": -1.0,
    "mean_reward": -1.0,
    "difficulty_band": "uncalibrated",
    "calibration_model_panel": "[]",
    "dataset_version": VERSION,
}


def _slug(value: Any) -> str:
    text = str(value).lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _node_count(code: str) -> int:
    body = "\n".join(f"    {line}" for line in code.splitlines())
    return sum(1 for _ in ast.walk(ast.parse(f"def rule(card, mainline):\n{body}")))


def _acceptance_stats(code: str) -> tuple[float, float]:
    rule = compile_python_rule(code)
    outcomes = [
        bool(rule(parse_card(symbol), mainline))
        for mainline in representative_mainlines()
        if mainline
        for symbol in deck()
    ]
    rate = statistics.mean(outcomes)
    entropy = 0.0 if rate in (0.0, 1.0) else -(rate * math.log2(rate) + (1 - rate) * math.log2(1 - rate))
    return rate, entropy


def _full_row(row: dict[str, Any]) -> dict[str, Any]:
    return {**META_DEFAULTS, **row, "dataset_version": VERSION}


class RuleBank:
    def __init__(self, occupied: dict[tuple[bool, ...], str]) -> None:
        self.occupied = dict(occupied)
        self.rows: list[dict[str, Any]] = []
        self.rejections: Counter[str] = Counter()

    def add(
        self,
        *,
        family: str,
        template_id: str,
        label: str,
        code: str,
        params: dict[str, Any],
        level: int,
        history_depth: int = 0,
        period: int = 0,
        branches: int = 0,
        cardinality: int = 0,
        components: list[str] | None = None,
        min_acceptance: float = 0.07,
        max_acceptance: float = 0.93,
    ) -> None:
        try:
            compiled = compile_python_rule(code)
            if not any(compiled(parse_card(symbol), []) for symbol in deck()):
                self.rejections["no_starter"] += 1
                return
            signature = rule_extension_signature(code)
            if signature is None:
                self.rejections["invalid_signature"] += 1
                return
            if signature in self.occupied:
                self.rejections["duplicate"] += 1
                return
            rate, entropy = _acceptance_stats(code)
            if not min_acceptance <= rate <= max_acceptance:
                self.rejections["extreme_acceptance"] += 1
                return
        except Exception:
            self.rejections["compile_or_runtime"] += 1
            return

        key = hashlib.sha256((template_id + "\0" + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:10]
        rule_id = f"cal_{_slug(family)}_{_slug(template_id)}_{key}"
        row = _full_row(
            {
                "rule_id": rule_id,
                "label": label,
                "family": family,
                "code": code,
                "description": label,
                "template_id": template_id,
                "template_parameters": json.dumps(params, sort_keys=True),
                "component_families": json.dumps(components or [family]),
                "split_type": "candidate",
                "semantic_features": json.dumps(
                    {
                        "history_depth": history_depth,
                        "period_or_chunk_length": period,
                        "branch_count": branches,
                        "attribute_cardinality": cardinality,
                        "designed_level": level,
                    },
                    sort_keys=True,
                ),
                "history_depth": history_depth,
                "period_or_chunk_length": period,
                "branch_count": branches,
                "attribute_cardinality": cardinality,
                "ast_nodes": _node_count(code),
                "acceptance_rate": round(rate, 6),
                "acceptance_entropy": round(entropy, 6),
                "signature_hash": hashlib.sha256(bytes(signature)).hexdigest(),
                "designed_level": level,
            }
        )
        self.occupied[signature] = rule_id
        self.rows.append(row)


def _source_rows(source: str) -> tuple[list[dict[str, Any]], dict[tuple[bool, ...], str]]:
    """Ingest every rule of an existing bank: emitted as source rows and used
    to occupy behavioral signatures so generated candidates cannot duplicate
    anything already published."""
    dataset = load_dataset(source)
    rows = []
    occupied: dict[tuple[bool, ...], str] = {}
    for split in dataset:
        for raw in dataset[split]:
            code = str(raw["code"])
            signature = rule_extension_signature(code)
            if signature is None:
                raise ValueError(f"Invalid source rule: {raw['rule_id']}")
            occupied.setdefault(signature, str(raw["rule_id"]))
            rate, entropy = _acceptance_stats(code)
            rows.append(
                _full_row(
                    {
                        **{column: str(raw[column]) for column in CORE_COLUMNS},
                        "description": str(raw["label"]),
                        "template_id": "source",
                        "component_families": json.dumps([str(raw["family"])]),
                        "split_type": f"source_{split}",
                        "ast_nodes": _node_count(code),
                        "acceptance_rate": round(rate, 6),
                        "acceptance_entropy": round(entropy, 6),
                        "signature_hash": hashlib.sha256(bytes(signature)).hexdigest(),
                    }
                )
            )
    return rows, occupied


RELATIONS = {
    "same_suit": ("card.suit == prev.suit", "same suit"),
    "different_suit": ("card.suit != prev.suit", "different suit"),
    "same_color": ("card.color == prev.color", "same color"),
    "different_color": ("card.color != prev.color", "different color"),
    "rank_up": ("card.rank > prev.rank", "higher rank"),
    "rank_down": ("card.rank < prev.rank", "lower rank"),
    "same_parity": ("card.rank % 2 == prev.rank % 2", "same parity"),
    "different_parity": ("card.rank % 2 != prev.rank % 2", "different parity"),
}


def add_static(bank: RuleBank) -> None:
    for low in range(1, 11):
        for high in range(low + 1, 14):
            bank.add(
                family="static_predicate",
                template_id="rank_interval",
                label=f"ranks {low} through {high}",
                code=f"return {low} <= card.rank <= {high}",
                params={"low": low, "high": high},
                level=1 if high - low >= 5 else 2,
                cardinality=13,
            )
    for modulus in (3, 4, 5):
        for residue in range(modulus):
            bank.add(
                family="static_predicate",
                template_id="rank_modulus",
                label=f"ranks congruent to {residue} modulo {modulus}",
                code=f"return card.rank % {modulus} == {residue}",
                params={"modulus": modulus, "residue": residue},
                level=2,
                cardinality=modulus,
            )
    for color in ("red", "black"):
        for threshold in (4, 6, 8, 10, 11):
            for op, words in (("<=", "at most"), (">=", "at least")):
                bank.add(
                    family="static_predicate",
                    template_id="color_rank_threshold",
                    label=f"{color} cards with rank {words} {threshold}",
                    code=f'return card.color == "{color}" and card.rank {op} {threshold}',
                    params={"color": color, "threshold": threshold, "operator": op},
                    level=2,
                    branches=1,
                    cardinality=2,
                )
    for suit in ("hearts", "diamonds", "clubs", "spades"):
        for parity, word in ((0, "even"), (1, "odd")):
            bank.add(
                family="static_predicate",
                template_id="suit_parity",
                label=f"{suit} with {word} rank",
                code=f'return card.suit == "{suit}" and card.rank % 2 == {parity}',
                params={"suit": suit, "parity": parity},
                level=2,
                cardinality=4,
            )


def add_first_order(bank: RuleBank) -> None:
    for distance in range(1, 7):
        for kind, expression, words in (
            ("exact", f"== {distance}", "exactly"),
            ("at_most", f"<= {distance}", "at most"),
            ("at_least", f">= {distance}", "at least"),
        ):
            bank.add(
                family="first_order_transition",
                template_id=f"rank_distance_{kind}",
                label=f"rank differs by {words} {distance}",
                code="if not mainline:\n    return True\n" + f"return abs(card.rank - mainline[-1].rank) {expression}",
                params={"distance": distance},
                level=2 if kind != "exact" else 3,
                history_depth=1,
                cardinality=13,
            )
    for threshold in range(3, 12):
        for same in (True, False):
            op = "==" if same else "!="
            word = "same" if same else "different"
            bank.add(
                family="first_order_transition",
                template_id="rank_group_transition",
                label=f"{word} side of rank threshold {threshold} as previous",
                code="if not mainline:\n    return True\n"
                + f"return (card.rank >= {threshold}) {op} (mainline[-1].rank >= {threshold})",
                params={"threshold": threshold, "same": same},
                level=2,
                history_depth=1,
                cardinality=2,
            )
    suits = ("hearts", "diamonds", "clubs", "spades")
    for order in itertools.permutations(suits[1:]):
        cycle = (suits[0],) + order
        bank.add(
            family="first_order_transition",
            template_id="suit_cycle",
            label="suits cycle " + ", ".join(cycle),
            code="if not mainline:\n    return True\n"
            + f"order = {list(cycle)!r}\nidx = order.index(mainline[-1].suit)\n"
            + "return card.suit == order[(idx + 1) % 4]",
            params={"order": cycle},
            level=2,
            history_depth=1,
            period=4,
            cardinality=4,
        )
    for color_same in (True, False):
        for parity_same in (True, False):
            color_op = "==" if color_same else "!="
            parity_op = "==" if parity_same else "!="
            bank.add(
                family="first_order_transition",
                template_id="color_and_parity",
                label=("same" if color_same else "different")
                + " color and "
                + ("same" if parity_same else "different")
                + " parity",
                code="if not mainline:\n    return True\nprev = mainline[-1]\n"
                + f"return card.color {color_op} prev.color and card.rank % 2 {parity_op} prev.rank % 2",
                params={"color_same": color_same, "parity_same": parity_same},
                level=3,
                history_depth=1,
                cardinality=2,
            )


def _add_conditional(
    bank: RuleBank,
    *,
    template: str,
    gate_code: str,
    gate_label: str,
    params: dict[str, Any],
    level: int,
) -> None:
    pairs = list(itertools.permutations(RELATIONS, 2))
    for first, second in pairs:
        first_expr, first_label = RELATIONS[first]
        second_expr, second_label = RELATIONS[second]
        code = (
            "if not mainline:\n    return True\n"
            "prev = mainline[-1]\n"
            f"if {gate_code}:\n    return {first_expr}\n"
            f"return {second_expr}"
        )
        bank.add(
            family="conditional_transition",
            template_id=template,
            label=f"if previous is {gate_label}, require {first_label}; otherwise {second_label}",
            code=code,
            params={**params, "then": first, "else": second},
            level=level,
            history_depth=1,
            branches=2,
            cardinality=4 if "suit" in gate_code else 2,
        )


def add_conditional(bank: RuleBank) -> None:
    _add_conditional(
        bank,
        template="previous_color_gate",
        gate_code='prev.color == "red"',
        gate_label="red",
        params={"gate": "red"},
        level=3,
    )
    _add_conditional(
        bank,
        template="previous_face_gate",
        gate_code="prev.is_face",
        gate_label="a face card",
        params={"gate": "face"},
        level=3,
    )
    _add_conditional(
        bank,
        template="previous_parity_gate",
        gate_code="prev.rank % 2 == 0",
        gate_label="even",
        params={"gate": "even"},
        level=3,
    )
    selected_pairs = list(itertools.permutations(RELATIONS, 2))[::5]
    for threshold in (5, 8, 10, 11):
        for first, second in selected_pairs:
            first_expr, first_label = RELATIONS[first]
            second_expr, second_label = RELATIONS[second]
            bank.add(
                family="conditional_transition",
                template_id="previous_threshold_gate",
                label=f"after rank {threshold} or above require {first_label}; otherwise {second_label}",
                code="if not mainline:\n    return True\nprev = mainline[-1]\n"
                + f"if prev.rank >= {threshold}:\n    return {first_expr}\n"
                + f"return {second_expr}",
                params={"threshold": threshold, "then": first, "else": second},
                level=4,
                history_depth=1,
                branches=2,
                cardinality=2,
            )


def add_periodic(bank: RuleBank) -> None:
    rng = random.Random(20260809)
    for period in range(2, 7):
        binary_patterns = [pattern for pattern in itertools.product((0, 1), repeat=period) if 0 < sum(pattern) < period]
        rng.shuffle(binary_patterns)
        for pattern in binary_patterns[: min(12, len(binary_patterns))]:
            bank.add(
                family="periodic_cycle",
                template_id="position_color_pattern",
                label=f"accepted-position color pattern {''.join('R' if x else 'B' for x in pattern)}",
                code="if not mainline:\n    return True\n"
                + f'pattern = {list(pattern)!r}\nreturn (card.color == "red") == bool(pattern[len(mainline) % {period}])',
                params={"period": period, "pattern": pattern},
                level=2 + period // 2,
                history_depth=0,
                period=period,
                cardinality=2,
            )
    suit_orders = list(itertools.permutations(("hearts", "diamonds", "clubs", "spades")))
    rng.shuffle(suit_orders)
    for period in (2, 3, 4):
        for order in suit_orders[:18]:
            pattern = order[:period]
            bank.add(
                family="periodic_cycle",
                template_id="position_suit_pattern",
                label=f"position cycle over {', '.join(pattern)}",
                code="if not mainline:\n    return True\n"
                + f"pattern = {list(pattern)!r}\nreturn card.suit == pattern[len(mainline) % {period}]",
                params={"period": period, "pattern": pattern},
                level=3 + period // 2,
                period=period,
                cardinality=4,
            )


ATTRIBUTES = {
    "color": ("card.color", "mainline[-1].color", "color", 2),
    "suit": ("card.suit", "mainline[-1].suit", "suit", 4),
    "parity": ("card.rank % 2", "mainline[-1].rank % 2", "rank parity", 2),
    "face": ("card.is_face", "mainline[-1].is_face", "face status", 2),
    "rank": ("card.rank", "mainline[-1].rank", "rank", 13),
}


def add_chunks(bank: RuleBank) -> None:
    for length in range(2, 6):
        for attribute, (candidate, previous, label, cardinality) in ATTRIBUTES.items():
            for same_within in (True, False):
                within_op = "==" if same_within else "!="
                boundary_op = "!=" if same_within else "=="
                within_word = "same" if same_within else "different"
                boundary_word = "changes" if same_within else "matches"
                bank.add(
                    family="chunk_run",
                    template_id="fixed_chunk",
                    label=f"{label} is {within_word} within chunks of {length} and {boundary_word} at boundaries",
                    code="if not mainline:\n    return True\n"
                    + f"if len(mainline) % {length} == 0:\n    return {candidate} {boundary_op} {previous}\n"
                    + f"return {candidate} {within_op} {previous}",
                    params={"length": length, "attribute": attribute, "same_within": same_within},
                    level=2 + length // 2 + (1 if cardinality == 13 else 0),
                    history_depth=1,
                    period=length,
                    cardinality=cardinality,
                )
    for length in (2, 3, 4):
        for direction in ("up", "down"):
            op = ">" if direction == "up" else "<"
            bank.add(
                family="chunk_run",
                template_id="suit_chunks_rank_boundary",
                label=f"suits repeat in chunks of {length} and rank moves {direction} at boundaries",
                code="if not mainline:\n    return True\nprev = mainline[-1]\n"
                + f"if len(mainline) % {length} == 0:\n    return card.suit != prev.suit and card.rank {op} prev.rank\n"
                + "return card.suit == prev.suit",
                params={"length": length, "direction": direction},
                level=4,
                history_depth=1,
                period=length,
                branches=2,
                cardinality=4,
                components=["chunk_run", "first_order_transition"],
            )


def add_higher_order(bank: RuleBank) -> None:
    for window in range(2, 6):
        for target in ("majority", "minority"):
            comparison = ">=" if target == "majority" else "<"
            label = "majority" if target == "majority" else "minority"
            threshold = (window + 1) // 2
            bank.add(
                family="higher_order_history",
                template_id="window_color_vote",
                label=f"candidate follows the {label} color in the last {window} accepted cards",
                code=f"if len(mainline) < {window}:\n    return True\n"
                + f'reds = sum(1 for item in mainline[-{window}:] if item.color == "red")\n'
                + f'return (card.color == "red") == (reds {comparison} {threshold})',
                params={"window": window, "target": target},
                level=2 + window // 2,
                history_depth=window,
                cardinality=2,
            )
        for direction, fn, op, words in (
            ("above_max", "max", ">", "above the maximum"),
            ("below_min", "min", "<", "below the minimum"),
            ("at_least_max", "max", ">=", "at least the maximum"),
            ("at_most_min", "min", "<=", "at most the minimum"),
        ):
            bank.add(
                family="higher_order_history",
                template_id="window_rank_extreme",
                label=f"rank is {words} of the last {window}",
                code=f"if len(mainline) < {window}:\n    return True\n"
                + f"recent = [item.rank for item in mainline[-{window}:]]\n"
                + f"return card.rank {op} {fn}(recent)",
                params={"window": window, "direction": direction},
                level=3 + window // 2,
                history_depth=window,
                cardinality=13,
            )
    for attribute, (candidate, _previous, label, cardinality) in ATTRIBUTES.items():
        if attribute == "rank":
            continue
        item = {
            "color": "item.color",
            "suit": "item.suit",
            "parity": "item.rank % 2",
            "face": "item.is_face",
        }[attribute]
        for window in (2, 3, 4):
            for repeat in (True, False):
                quantifier = "any" if repeat else "all"
                op = "==" if repeat else "!="
                verb = "repeats" if repeat else "avoids"
                bank.add(
                    family="higher_order_history",
                    template_id="window_seen_attribute",
                    label=f"candidate {verb} a {label} from the last {window}",
                    code=f"if len(mainline) < {window}:\n    return True\n"
                    + f"return {quantifier}({candidate} {op} {item} for item in mainline[-{window}:])",
                    params={"window": window, "attribute": attribute, "repeat": repeat},
                    level=2 + window // 2,
                    history_depth=window,
                    cardinality=cardinality,
                )
    for motif in ("aba", "abb", "abc"):
        for attribute, (candidate, _previous, label, cardinality) in ATTRIBUTES.items():
            if attribute == "rank":
                continue
            indexes = {"aba": -2, "abb": -1, "abc": -3}
            reference = {
                "color": f"mainline[{indexes[motif]}].color",
                "suit": f"mainline[{indexes[motif]}].suit",
                "parity": f"mainline[{indexes[motif]}].rank % 2",
                "face": f"mainline[{indexes[motif]}].is_face",
            }[attribute]
            op = "!=" if motif == "abc" else "=="
            bank.add(
                family="higher_order_history",
                template_id="three_step_motif",
                label=f"continue {motif.upper()} motif over {label}",
                code="if len(mainline) < 2:\n    return True\n" + f"return {candidate} {op} {reference}",
                params={"motif": motif, "attribute": attribute},
                level=3,
                history_depth=2,
                period=3,
                cardinality=cardinality,
            )


def add_global(bank: RuleBank) -> None:
    specs = {
        "color": ("card.color", "item.color", "color", 2),
        "suit": ("card.suit", "item.suit", "suit", 4),
        "parity": ("card.rank % 2", "item.rank % 2", "rank parity", 2),
        "face": ("card.is_face", "item.is_face", "face status", 2),
        "rank": ("card.rank", "item.rank", "rank", 13),
    }
    for attribute, (candidate, item, label, cardinality) in specs.items():
        for repeat in (True, False):
            quantifier = "any" if repeat else "all"
            op = "==" if repeat else "!="
            verb = "has appeared" if repeat else "has not appeared"
            bank.add(
                family="global_history",
                template_id="global_seen_attribute",
                label=f"candidate {label} {verb} on the mainline",
                code="if not mainline:\n    return True\n"
                + f"return {quantifier}({candidate} {op} {item} for item in mainline)",
                params={"attribute": attribute, "repeat": repeat},
                level=4 if cardinality <= 4 else 5,
                history_depth=8,
                cardinality=cardinality,
            )
    for direction, fn, op, words in (
        ("new_high", "max", ">=", "at least the highest rank seen"),
        ("new_low", "min", "<=", "at most the lowest rank seen"),
    ):
        bank.add(
            family="global_history",
            template_id="global_rank_extreme",
            label=words,
            code="if not mainline:\n    return True\n" + f"return card.rank {op} {fn}(item.rank for item in mainline)",
            params={"direction": direction},
            level=5,
            history_depth=8,
            cardinality=13,
        )
    for modulus in (2, 3, 4):
        for residue in range(modulus):
            bank.add(
                family="global_history",
                template_id="accepted_count_phase",
                label=f"red when accepted count is {residue} modulo {modulus}, black otherwise",
                code="if not mainline:\n    return True\n"
                + f'return (card.color == "red") == (len(mainline) % {modulus} == {residue})',
                params={"modulus": modulus, "residue": residue},
                level=3 + modulus // 2,
                history_depth=8,
                period=modulus,
                cardinality=2,
            )


def add_hybrids(bank: RuleBank) -> None:
    selected_relations = ("same_suit", "different_suit", "rank_up", "rank_down", "different_parity")
    for relation in selected_relations:
        expression, relation_label = RELATIONS[relation]
        for color in ("red", "black"):
            bank.add(
                family="compositional_hybrid",
                template_id="static_and_transition",
                label=f"{color} cards that have {relation_label} relative to previous",
                code="if not mainline:\n    return True\nprev = mainline[-1]\n"
                + f'return card.color == "{color}" and {expression}',
                params={"relation": relation, "color": color},
                level=4,
                history_depth=1,
                branches=1,
                cardinality=2,
                components=["static_predicate", "first_order_transition"],
            )
    for period in (2, 3, 4):
        for even_relation, odd_relation in itertools.permutations(selected_relations, 2):
            first_expr, first_label = RELATIONS[even_relation]
            second_expr, second_label = RELATIONS[odd_relation]
            bank.add(
                family="compositional_hybrid",
                template_id="periodic_transition",
                label=f"every {period}th position requires {first_label}; others require {second_label}",
                code="if not mainline:\n    return True\nprev = mainline[-1]\n"
                + f"if len(mainline) % {period} == 0:\n    return {first_expr}\n"
                + f"return {second_expr}",
                params={"period": period, "boundary": even_relation, "other": odd_relation},
                level=4 + period // 3,
                history_depth=1,
                period=period,
                branches=2,
                cardinality=4,
                components=["periodic_cycle", "first_order_transition"],
            )
    for window in (2, 3, 4):
        for gate in ("red", "black"):
            for relation in ("same_suit", "different_suit", "rank_up", "rank_down"):
                expression, relation_label = RELATIONS[relation]
                bank.add(
                    family="compositional_hybrid",
                    template_id="window_gate_transition",
                    label=f"when {gate} is majority in last {window}, require {relation_label}; otherwise opposite color",
                    code=f"if len(mainline) < {window}:\n    return True\n"
                    + f'count = sum(1 for item in mainline[-{window}:] if item.color == "{gate}")\n'
                    + "prev = mainline[-1]\n"
                    + f"if count > {window // 2}:\n    return {expression}\n"
                    + "return card.color != prev.color",
                    params={"window": window, "gate": gate, "relation": relation},
                    level=5,
                    history_depth=window,
                    branches=2,
                    cardinality=2,
                    components=["higher_order_history", "conditional_transition"],
                )


def _stratified_split(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Template-aware 45/15/40 split into generated train/validation/candidate."""
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_template[row["template_id"]].append(row)
    train, validation, candidate = [], [], []
    for template, group in sorted(by_template.items()):
        group.sort(key=lambda row: row["rule_id"])
        # Within a reviewed template we hold out parameter values.  Explicit
        # composition/extrapolation labels are assigned by the final selector.
        for index, row in enumerate(group):
            bucket = int(hashlib.sha256(row["rule_id"].encode()).hexdigest(), 16) % 20
            if bucket < 9:
                destination, split_type = train, "train_parameterization"
            elif bucket < 12:
                destination, split_type = validation, "validation_parameterization"
            else:
                destination, split_type = candidate, "candidate_test"
            destination.append({**row, "split_type": split_type})
    return train, validation, candidate


def build(source: str) -> tuple[DatasetDict, dict[str, Any]]:
    source_rows, occupied = _source_rows(source)
    bank = RuleBank(occupied)
    add_static(bank)
    add_first_order(bank)
    add_conditional(bank)
    add_periodic(bank)
    add_chunks(bank)
    add_higher_order(bank)
    add_global(bank)
    add_hybrids(bank)
    generated_train, validation, candidate = _stratified_split(bank.rows)
    train = source_rows + generated_train
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(validation),
            "candidate": Dataset.from_list(candidate),
        }
    )
    summary = {
        "version": VERSION,
        "source": source,
        "counts": {split: len(rows) for split, rows in dataset.items()},
        "families": {split: dict(Counter(rows["family"])) for split, rows in dataset.items()},
        "generated_total": len(bank.rows),
        "rejections": dict(bank.rejections),
    }
    return dataset, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_RULE_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset, summary = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output))
    summary_path = args.output.parent / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved dataset to {args.output}")
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
