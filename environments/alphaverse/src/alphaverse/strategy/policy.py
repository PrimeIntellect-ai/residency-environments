"""Source syntax checks; Verifiers runtimes enforce the security boundary.

Uploaded code is never imported by the evaluator. It runs only in a separate,
zero-egress Docker container or Prime micro-VM containing the public SDK and
that firm's own source. Python reflection and local files are intentionally
allowed there; they confer no access to the exchange's memory or filesystem.
"""

from __future__ import annotations

import ast


class StrategySourcePolicyError(ValueError):
    """Uploaded source is not valid Python."""


def validate_strategy_source(source: str) -> ast.Module:
    """Parse source without executing it; this is not a security sandbox."""
    try:
        return ast.parse(source, filename="strategy.py")
    except SyntaxError as exc:
        raise StrategySourcePolicyError(f"invalid strategy syntax: {exc.msg}") from exc


__all__ = ["StrategySourcePolicyError", "validate_strategy_source"]
