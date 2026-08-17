"""Conservative source policy for evaluator-hosted player strategies.

This is defense in depth around the child-process boundary.  It deliberately
keeps the authoring surface small enough for the documented SDK while denying
ordinary filesystem, network, process, reflection, and hidden-package access.
It is not a substitute for a kernel-enforced hostile-code sandbox.
"""

from __future__ import annotations

import ast

SAFE_STDLIB_IMPORTS = {
    "collections": frozenset({"Counter", "OrderedDict", "defaultdict", "deque", "namedtuple"}),
    "decimal": frozenset({"Decimal", "ROUND_DOWN", "ROUND_HALF_EVEN", "ROUND_UP"}),
    "enum": frozenset({"Enum", "Flag", "IntEnum", "IntFlag", "auto"}),
    "fractions": frozenset({"Fraction"}),
    "functools": frozenset({"cache", "lru_cache", "partial", "reduce"}),
    "heapq": frozenset({"heapify", "heappop", "heappush", "heappushpop", "heapreplace", "merge"}),
    "itertools": frozenset({"accumulate", "chain", "combinations", "count", "islice", "pairwise", "product"}),
    "math": frozenset({"ceil", "comb", "exp", "floor", "isfinite", "log", "log10", "sqrt"}),
    "statistics": frozenset({"fmean", "mean", "median", "median_high", "median_low", "pstdev", "pvariance"}),
    "typing": frozenset({"Any", "ClassVar", "Iterable", "Literal", "Mapping", "Sequence"}),
}

PUBLIC_IMPORTS = {
    "alphaverse": frozenset({"Side"}),
    "alphaverse.strategy": frozenset({"InputEnvelope", "InputKind", "LogLevel", "Strategy", "StrategyContext"}),
}

BLOCKED_CALLS = frozenset(
    {
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
        "__import__",
    }
)


class StrategySourcePolicyError(ValueError):
    """Uploaded source uses an API outside the public strategy surface."""


class _PolicyVisitor(ast.NodeVisitor):
    def _reject(self, node: ast.AST, message: str) -> None:
        raise StrategySourcePolicyError(
            f"strategy source policy violation on line {getattr(node, 'lineno', '?')}: {message}"
        )

    def visit_Import(self, node: ast.Import) -> None:
        self._reject(node, "module imports are not allowed; import named safe symbols")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self._reject(node, "relative imports are not allowed")
        module = node.module or ""
        names = {alias.name for alias in node.names}
        if module == "__future__":
            if names != {"annotations"}:
                self._reject(node, "only future annotations are allowed")
            return
        if "*" in names:
            self._reject(node, "wildcard imports are not allowed")
        allowed = SAFE_STDLIB_IMPORTS.get(module) or PUBLIC_IMPORTS.get(module)
        if allowed is None or not names <= allowed:
            self._reject(
                node,
                f"from {module!r} import {', '.join(sorted(names))} is not allowed",
            )

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            self._reject(node, f"call to {node.func.id!r} is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self._reject(node, "dunder attribute access is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self._reject(node, "dunder names are not allowed")


def validate_strategy_source(source: str) -> ast.Module:
    """Parse and validate one uploaded strategy module."""

    try:
        tree = ast.parse(source, filename="strategy.py")
    except SyntaxError as exc:
        raise StrategySourcePolicyError(f"invalid strategy syntax: {exc.msg}") from exc
    _PolicyVisitor().visit(tree)
    return tree


__all__ = [
    "PUBLIC_IMPORTS",
    "SAFE_STDLIB_IMPORTS",
    "StrategySourcePolicyError",
    "validate_strategy_source",
]
