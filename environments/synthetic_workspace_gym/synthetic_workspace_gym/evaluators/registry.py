from __future__ import annotations

from importlib import import_module

from synthetic_workspace_gym.evaluators.base import BaseEvaluator
from synthetic_workspace_gym.schemas import EnvironmentFamily

from .composite_workspace import CompositeWorkspaceEvaluator
from .pipeline import PipelineEvaluator
from .retrieval_workspace import RetrievalWorkspaceEvaluator
from .script_repair import ScriptRepairEvaluator
from .tabular import TabularEvaluator

EVALUATORS = {
    EnvironmentFamily.TABULAR: TabularEvaluator(),
    EnvironmentFamily.SCRIPT_REPAIR: ScriptRepairEvaluator(),
    EnvironmentFamily.PIPELINE: PipelineEvaluator(),
    EnvironmentFamily.RETRIEVAL_WORKSPACE: RetrievalWorkspaceEvaluator(),
    EnvironmentFamily.COMPOSITE_WORKSPACE: CompositeWorkspaceEvaluator(),
}


def get_evaluator(family: EnvironmentFamily | str, evaluator_entrypoint: str | None = None):
    if evaluator_entrypoint:
        return load_evaluator_from_entrypoint(evaluator_entrypoint)
    return D5CalibratedEvaluator(EVALUATORS[EnvironmentFamily(family)])


def list_evaluators() -> list[str]:
    return [family.value for family in EVALUATORS]


def load_evaluator_from_entrypoint(entrypoint: str) -> BaseEvaluator:
    module_name, separator, attr_name = entrypoint.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(f"Invalid evaluator entrypoint: {entrypoint}")
    module = import_module(module_name)
    loaded = getattr(module, attr_name)
    if isinstance(loaded, BaseEvaluator):
        return D5CalibratedEvaluator(loaded)
    instance = loaded()
    if not isinstance(instance, BaseEvaluator):
        raise TypeError(f"Evaluator entrypoint did not resolve to a BaseEvaluator: {entrypoint}")
    return D5CalibratedEvaluator(instance)


class D5CalibratedEvaluator(BaseEvaluator):
    """Compress incomplete D5 rewards while preserving evaluator diagnostics."""

    def __init__(self, evaluator: BaseEvaluator) -> None:
        self.evaluator = evaluator

    def evaluate(self, workspace_path, manifest, hidden_root):
        result = self.evaluator.evaluate(workspace_path, manifest, hidden_root)
        realization = dict(manifest.metadata.get("difficulty_realization", {}))
        if int(realization.get("level", 0)) != 5 or result.success:
            return result
        raw_score = float(result.score)
        result.score = round(raw_score**3, 6)
        result.diagnostics = {
            **result.diagnostics,
            "d5_raw_partial_score": raw_score,
            "d5_partial_score_exponent": 3,
        }
        return result
