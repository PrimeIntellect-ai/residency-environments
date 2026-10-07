"""Content-addressed strategy source; building an artifact never executes it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from alphaverse.strategy.policy import validate_strategy_source


@dataclass(frozen=True, slots=True)
class StrategyArtifact:
    version_id: str
    entrypoint: str
    source: str

    @classmethod
    def build(cls, source: str, *, entrypoint: str = "strategy:StrategyImpl"):
        return cls._build(source, entrypoint=entrypoint, trusted=False)

    @classmethod
    def build_trusted(cls, source: str, *, entrypoint: str = "strategy:StrategyImpl"):
        return cls._build(source, entrypoint=entrypoint, trusted=True)

    @classmethod
    def _build(cls, source: str, *, entrypoint: str, trusted: bool):
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if not source.strip():
            raise ValueError("source must not be empty")
        if len(source.encode()) > 1_000_000:
            raise ValueError("source exceeds the 1 MB MVP artifact limit")
        if not trusted:
            validate_strategy_source(source)
        module, separator, class_name = entrypoint.partition(":")
        if not separator or module != "strategy" or not class_name.isidentifier():
            raise ValueError("entrypoint must have the form strategy:ClassName")
        digest = hashlib.sha256((entrypoint + "\0" + source).encode()).hexdigest()
        return cls(f"sha256:{digest}", entrypoint, source)
