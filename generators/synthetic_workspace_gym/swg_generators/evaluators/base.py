from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from swg_generators.schemas import EnvironmentManifest, EvaluatorResult


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, workspace_path: Path, manifest: EnvironmentManifest, hidden_root: Path) -> EvaluatorResult:
        raise NotImplementedError
