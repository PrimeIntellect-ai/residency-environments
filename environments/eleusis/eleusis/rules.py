"""Rule compilation, behavioral-equivalence guess checking, and dataset loading.

A rule is a small Python predicate over (card, mainline). Rules are compiled in
a restricted-AST sandbox. A guess is judged correct by behavioral equivalence:
it must agree with the hidden rule on every card of the deck across a battery
of representative mainline contexts — the exact-match convention this
environment has used since its training era, kept deterministic (no auxiliary
LLM, no sampling) so results are reproducible.
"""

from __future__ import annotations

import ast
import math
import multiprocessing
import random
import re
import resource
import sys
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Sequence

from .cards import Card, deck, parse_card

CompiledRule = Callable[[Card, Sequence[Card]], bool]
ObservedVerdict = tuple[str, list[str], bool]

DEFAULT_RULE_DATASET = "nph4rd/eleusis-calibrated"
DEFAULT_RULE_DATASET_REVISION = "f4d1aeef1617df8ac30454dd2884a67d3e7d0d93"
RULE_CHECK_CPU_SECONDS = 2
RULE_CHECK_MEMORY_BYTES = 1_073_741_824
RULE_CHECK_WALL_SECONDS = 5.0


@dataclass(frozen=True)
class Rule:
    rule_id: str
    code: str
    family: str | None = None


@lru_cache(maxsize=8)
def load_rules(
    repo_id: str = DEFAULT_RULE_DATASET,
    revision: str | None = None,
    config_name: str | None = None,
) -> dict[str, list[Rule]]:
    """Load a local or Hub rule dataset, keyed by split.

    A rule source only needs ``rule_id`` and ``code`` columns. Hub datasets may
    expose any split names and an optional dataset configuration. A single
    locally saved ``Dataset`` is treated as a ``test`` split, while a locally
    saved ``DatasetDict`` preserves its split names.
    """
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    local_path = Path(repo_id).expanduser()
    resolved_revision = revision
    if repo_id == DEFAULT_RULE_DATASET and revision is None:
        resolved_revision = DEFAULT_RULE_DATASET_REVISION
    dataset = (
        load_from_disk(str(local_path))
        if local_path.exists()
        else load_dataset(repo_id, name=config_name, revision=resolved_revision)
    )
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"test": dataset})
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Rule source {repo_id!r} returned {type(dataset).__name__}; expected a Dataset or DatasetDict."
        )
    splits: dict[str, list[Rule]] = {}
    for split_name, rows in dataset.items():
        required = {"rule_id", "code"}
        missing = required - set(rows.column_names)
        if missing:
            raise ValueError(
                f"Rule dataset {repo_id!r} split {split_name!r} is missing required columns: {sorted(missing)}."
            )
        seen: set[str] = set()
        loaded: list[Rule] = []
        for row in rows:
            rule_id = str(row["rule_id"]).strip()
            code = str(row["code"]).strip()
            if not rule_id or not code:
                raise ValueError(f"Rule dataset {repo_id!r} split {split_name!r} contains an empty rule_id or code.")
            if rule_id in seen:
                raise ValueError(f"Duplicate rule_id {rule_id!r} in {repo_id!r} split {split_name!r}.")
            compile_python_rule(code)
            seen.add(rule_id)
            family = str(row["family"]).strip() if "family" in rows.column_names else None
            loaded.append(Rule(rule_id=rule_id, code=code, family=family or None))
        splits[str(split_name)] = loaded
    return splits


@lru_cache(maxsize=512)
def rule_signature(source: str, include_empty: bool = True) -> bytes | None:
    """Return a rule's verdicts over the fixed behavioral probe battery."""
    if not include_empty:
        signature = rule_signature(source, include_empty=True)
        return None if signature is None else signature[len(deck()) :]
    try:
        compiled = compile_python_rule(str(source or "").strip())
    except Exception:
        return None
    bits = bytearray()
    try:
        for mainline in representative_mainlines():
            for symbol in deck():
                bits.append(bool(compiled(parse_card(symbol), mainline)))
    except Exception:
        return None
    return bytes(bits)


@lru_cache(maxsize=512)
def rule_extension_signature(source: str) -> bytes | None:
    """Behavioral signature of a rule string over the probe battery.

    Two rule strings with the same signature are extensionally equivalent for
    this environment (same accept/reject on every non-empty probe mainline x
    card). Returns None when the rule does not compile or errors. Used by the
    dataset builder to prove the train split leaks no test rule.
    """
    return rule_signature(source, include_empty=False)


def _set_rule_check_limits() -> None:
    if sys.platform == "darwin":
        # Darwin accounts shared-library mappings against RLIMIT_AS, making a
        # practical address-space cap lower than the process's baseline usage.
        return
    _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    soft_limit = RULE_CHECK_MEMORY_BYTES
    if hard_limit != resource.RLIM_INFINITY:
        soft_limit = min(soft_limit, hard_limit)
    resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))


def _renew_rule_check_cpu_limit() -> None:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    soft_limit = math.ceil(usage.ru_utime + usage.ru_stime + RULE_CHECK_CPU_SECONDS)
    _, hard_limit = resource.getrlimit(resource.RLIMIT_CPU)
    if hard_limit != resource.RLIM_INFINITY:
        soft_limit = min(soft_limit, hard_limit)
    resource.setrlimit(resource.RLIMIT_CPU, (soft_limit, hard_limit))


def _matches_observations(guess: str, observations: Sequence[ObservedVerdict]) -> bool:
    try:
        compiled = compile_python_rule(guess)
        for symbol, mainline_symbols, expected in observations:
            mainline = [parse_card(item) for item in mainline_symbols]
            if bool(compiled(parse_card(symbol), mainline)) != expected:
                return False
    except Exception:
        return False
    return True


def _rule_check_worker(connection: Connection, target_code: str) -> None:
    _set_rule_check_limits()
    target_signature = rule_signature(target_code)
    target_reachable_signature = rule_signature(target_code, include_empty=False)
    connection.send(target_signature is not None and target_reachable_signature is not None)
    while True:
        try:
            payload = connection.recv()
        except EOFError:
            return
        if payload is None:
            return
        guess, observations = payload
        _renew_rule_check_cpu_limit()
        matches = _matches_observations(guess, observations)
        guessed_signature = rule_signature(guess) if matches else None
        matches = matches and guessed_signature is not None and guessed_signature == target_signature
        if not matches:
            reachable_signature = rule_signature(guess, include_empty=False) if guessed_signature is not None else None
            matches = reachable_signature is not None and reachable_signature == target_reachable_signature
        connection.send(matches)


class IsolatedRuleChecker:
    """Check untrusted model hypotheses in a resource-limited child process."""

    def __init__(self, target_code: str) -> None:
        self._target_code = target_code
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._start()

    def matches(self, guess: str, observations: Sequence[ObservedVerdict] = ()) -> bool:
        if self._process is None or not self._process.is_alive():
            self._start()
        assert self._connection is not None
        connection = self._connection
        try:
            connection.send((guess, observations))
            if not connection.poll(RULE_CHECK_WALL_SECONDS):
                self.close()
                return False
            return bool(connection.recv())
        except (BrokenPipeError, EOFError, OSError):
            self.close()
            return False

    def close(self) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            if process is not None and process.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
            connection.close()
        if process is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

    def _start(self) -> None:
        self.close()
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        process = context.Process(
            target=_rule_check_worker,
            args=(child_connection, self._target_code),
            daemon=True,
        )
        process.start()
        child_connection.close()
        if not parent_connection.poll(RULE_CHECK_WALL_SECONDS):
            process.terminate()
            process.join(timeout=1)
            parent_connection.close()
            raise RuntimeError("Rule checker failed to initialize before its deadline.")
        try:
            initialized = parent_connection.recv()
        except EOFError as error:
            process.join(timeout=1)
            parent_connection.close()
            raise RuntimeError("Rule checker exited during initialization.") from error
        if not initialized:
            process.terminate()
            process.join(timeout=1)
            parent_connection.close()
            raise ValueError("Target rule failed behavioral signature generation.")
        self._connection = parent_connection
        self._process = process


_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_ALLOWED_CALL_METHODS = {"count", "endswith", "index", "startswith"}


@lru_cache(maxsize=256)
def compile_python_rule(source: str) -> CompiledRule:
    function_source = _as_function_source(source)
    tree = ast.parse(function_source, mode="exec")
    _validate_rule_ast(tree)
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    exec(compile(tree, "<eleusis_rule>", "exec"), namespace, namespace)
    fn = namespace.get("evaluate_rule")
    if not callable(fn):
        raise ValueError("Rule code must define evaluate_rule(card, mainline).")

    def evaluate(card: Card, mainline: Sequence[Card]) -> bool:
        return bool(fn(card, list(mainline)))

    return evaluate


def _as_function_source(source: str) -> str:
    code = textwrap.dedent(str(source or "")).strip()
    if not code:
        raise ValueError("Empty rule code.")
    if len(code) > 2_000:
        raise ValueError("Rule code is too long.")
    if re.match(r"^def\s+evaluate_rule\s*\(", code):
        return code
    if re.match(r"^def\s+is_playable\s*\(", code):
        call = "is_playable(card, mainline)" if _defines_two_arg_is_playable(code) else "is_playable(card)"
        return "\n\n".join(
            [
                code,
                f"def evaluate_rule(card, mainline):\n    return {call}",
            ]
        )
    if "\n" in code or re.search(r"\breturn\b", code):
        return "def evaluate_rule(card, mainline):\n" + textwrap.indent(code, "    ")
    return "def evaluate_rule(card, mainline):\n" + textwrap.indent(f"return bool({code})", "    ")


def _defines_two_arg_is_playable(code: str) -> bool:
    tree = ast.parse(code, mode="exec")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "is_playable":
            return len(node.args.args) >= 2
    return False


def _validate_rule_ast(tree: ast.AST) -> None:
    allowed_nodes = (
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Expr,
        ast.Assign,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Attribute,
        ast.Constant,
        ast.Compare,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.If,
        ast.IfExp,
        ast.List,
        ast.ListComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.Subscript,
        ast.Slice,
        ast.Call,
        ast.keyword,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    assigned_names = _assigned_names(tree)
    function_names = _function_names(tree)
    allowed_names = {
        "card",
        "mainline",
        "evaluate_rule",
        "is_playable",
        *assigned_names,
        *function_names,
        *_SAFE_BUILTINS,
    }
    callable_names = {*_SAFE_BUILTINS, *function_names}

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Disallowed Python syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attributes are not allowed.")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unknown name in rule code: {node.id}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in callable_names:
                    raise ValueError(f"Function is not allowed: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in _ALLOWED_CALL_METHODS:
                    raise ValueError(f"Method is not allowed: {node.func.attr}")
            else:
                raise ValueError("Unsupported call target.")


def _assigned_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, ast.comprehension):
            names.update(_target_names(node.target))
    return names


def _function_names(tree: ast.AST) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_target_names(item))
        return names
    return set()


@lru_cache(maxsize=1)
def representative_mainlines() -> tuple[tuple[Card, ...], ...]:
    """Deterministic board contexts used for rule-equivalence checks."""
    cards = {symbol: parse_card(symbol) for symbol in deck()}
    mainlines = [
        [],
        [cards["AH"]],
        [cards["2H"]],
        [cards["AC"]],
        [cards["2S"]],
        [cards["5H"]],
        [cards["5C"]],
        [cards["8C"]],
        [cards["QD"]],
        [cards["KS"]],
        [cards["AH"], cards["5C"]],
        [cards["2H"], cards["8C"]],
        [cards["AC"], cards["5H"]],
        [cards["8C"], cards["2D"]],
        [cards["5H"], cards["9C"], cards["KS"]],
        [cards["2H"], cards["8C"], cards["3D"]],
    ]
    # The core battery above is intentionally retained verbatim: published
    # behavioral signatures were computed against it.  The added
    # histories exercise position, chunk, recent-window, and aggregate rules.
    # They are deterministic so dataset signatures and eval results remain
    # reproducible, and short enough to keep per-turn equivalence checks cheap.
    patterns = [
        ["AH", "2C", "3D", "4S", "5H", "6C", "7D", "8S"],
        ["KH", "QC", "JD", "10S", "9H", "8C", "7D", "6S"],
        ["AH", "3H", "5C", "7C", "9D", "JD", "QS", "KS"],
        ["2H", "2C", "4D", "4S", "6H", "6C", "8D", "8S"],
        ["AH", "AD", "AC", "AS", "2H", "2D", "2C", "2S"],
        ["3H", "7C", "3D", "7S", "3C", "7H", "3S", "7D"],
        ["JH", "2C", "QD", "3S", "KH", "4C", "JD", "5S"],
        ["5H", "5C", "5D", "5S", "9H", "9C", "9D", "9S"],
        ["AH", "KC", "2D", "QS", "3H", "JC", "4D", "10S"],
        ["4H", "8C", "QD", "3S", "7H", "JC", "2D", "6S"],
        ["2H", "3C", "5D", "7S", "JH", "KC", "AD", "4S"],
        ["10H", "9C", "8D", "7S", "6H", "5C", "4D", "3S"],
    ]
    for pattern in patterns:
        converted = [cards[symbol] for symbol in pattern]
        for length in range(2, len(converted) + 1):
            mainlines.append(converted[:length])

    # The 100-turn dataset includes rules whose smallest distinguishing context
    # can occur after an eight-card window/period.  Keep every core probe above
    # and add longer contexts so such rules are not falsely deduplicated or
    # accepted as equivalent to a shorter-period hypothesis.
    # Run/repeat contexts: consecutive same-color, same-parity, same-suit and
    # repeated-rank adjacencies. Without these, rules whose acceptance depends
    # on runs or recent repeats are indistinguishable from vacuous hypotheses
    # (a bare "True" once passed equivalence for a streak-cap rule).
    run_patterns = [
        ["AH", "AD", "AC", "AS"],
        ["2H", "3H", "4H", "5H", "6H"],
        ["JC", "QC", "KC"],
        ["AS", "KS", "QS", "JS", "10S"],
        ["7H", "7D", "7C", "7S"],
        ["9H", "9C", "9D"],
        ["2H", "4H", "6H", "8H"],
        ["3C", "5C", "7C"],
        ["AH", "AD", "2C", "3C", "KH", "KD"],
        ["5S", "9S", "KS", "4S", "8D", "8H"],
        ["7D", "9D", "JH"],
    ]
    for pattern in run_patterns:
        converted = [cards[symbol] for symbol in pattern]
        for length in range(1, len(converted) + 1):
            mainlines.append(converted[:length])

    long_patterns = [
        ["AH", "2C", "3D", "4S", "5H", "6C", "7D", "8S", "9H", "10C", "JD", "QS", "KH", "AC", "2D", "3S"],
        ["KH", "QC", "JD", "10S", "9H", "8C", "7D", "6S", "5H", "4C", "3D", "2S", "AH", "KC", "QD", "JS"],
        ["AH", "AD", "AC", "AS", "2H", "2D", "2C", "2S", "3H", "3D", "3C", "3S", "4H", "4D", "4C", "4S"],
        ["2H", "3C", "5D", "7S", "JH", "KC", "AD", "4S", "6H", "8C", "10D", "QS", "3H", "7C", "9D", "KS"],
        ["JH", "2C", "QD", "3S", "KH", "4C", "JD", "5S", "QH", "6C", "KD", "7S", "AH", "8C", "2D", "9S"],
        ["3H", "7C", "3D", "7S", "3C", "7H", "3S", "7D", "5H", "9C", "5D", "9S", "5C", "9H", "5S", "9D"],
        ["10H", "9C", "8D", "7S", "6H", "5C", "4D", "3S", "2H", "AC", "KD", "QS", "JH", "10C", "9D", "8S"],
        ["AH", "KC", "2D", "QS", "3H", "JC", "4D", "10S", "5H", "9C", "6D", "8S", "7H", "AC", "KD", "2S"],
    ]
    for pattern in long_patterns:
        converted = [cards[symbol] for symbol in pattern]
        for length in range(9, len(converted) + 1):
            mainlines.append(converted[:length])

    seen = {tuple(card.symbol for card in mainline) for mainline in mainlines}

    def add_context(symbols: Sequence[str]) -> None:
        key = tuple(symbols)
        if key not in seen:
            seen.add(key)
            mainlines.append([cards[symbol] for symbol in key])

    # Cover every candidate position in a 100-turn game under several distinct
    # periodic histories. This prevents a finite position lookup from agreeing
    # only on the short prefix exercised by the original battery.
    horizon_patterns = [
        ["AH", "2C", "3D", "4S", "5H", "6C", "7D", "8S"],
        ["AH", "2D", "3C", "4S", "5H", "6D", "7C", "8S"],
        ["AC", "2C", "3H", "4H", "5D", "6D", "7S", "8S"],
        ["KH", "QD", "JC", "10S", "9H", "8D", "7C", "6S"],
    ]
    for pattern in horizon_patterns:
        extended = (pattern * 13)[:100]
        for length in range(17, 101):
            add_context(extended[:length])

    # Separate recent runs from global color counts. Padding each base history
    # with balanced pairs preserves its global discrepancy while changing its
    # absolute position, and the mirrored histories exercise both colors.
    balance_bases = [
        ["AC", "2C", "3C", "4C", "AH", "2D", "3H", "4D"],
        ["AH", "2D", "3H", "4D", "AC", "5H", "2C", "6D"],
    ]
    color_mirror = {"H": "C", "D": "S", "C": "H", "S": "D"}
    for padding in range(47):
        prefix = [symbol for _ in range(padding) for symbol in ("KH", "KC")]
        for base in balance_bases:
            history = prefix + base
            add_context(history)
            add_context([symbol[:-1] + color_mirror[symbol[-1]] for symbol in history])

    # Reproducible varied histories reduce accidental aliases outside the
    # hand-designed axes without making checker outcomes nondeterministic.
    rng = random.Random(0xE1E515)
    symbols = deck()
    red = [symbol for symbol in symbols if cards[symbol].color == "red"]
    black = [symbol for symbol in symbols if cards[symbol].color == "black"]
    for index in range(96):
        length = 1 + (index * 37) % 100
        if index % 3 == 0:
            pool = symbols
        else:
            favored, other = (red, black) if index % 2 else (black, red)
            pool = favored * 4 + other
        add_context([rng.choice(pool) for _ in range(length)])

    return tuple(tuple(mainline) for mainline in mainlines)
