from __future__ import annotations

from swg_generators.schemas import EnvironmentFamily

from .composite_workspace import CompositeWorkspaceGenerator
from .pipeline_completion import PipelineCompletionGenerator
from .retrieval_workspace import RetrievalWorkspaceGenerator
from .script_repair import ScriptRepairGenerator
from .tabular import TabularTransformationGenerator

GENERATORS = {
    EnvironmentFamily.TABULAR: TabularTransformationGenerator(),
    EnvironmentFamily.SCRIPT_REPAIR: ScriptRepairGenerator(),
    EnvironmentFamily.PIPELINE: PipelineCompletionGenerator(),
    EnvironmentFamily.RETRIEVAL_WORKSPACE: RetrievalWorkspaceGenerator(),
    EnvironmentFamily.COMPOSITE_WORKSPACE: CompositeWorkspaceGenerator(),
}


def get_generator(family: EnvironmentFamily | str):
    return GENERATORS[EnvironmentFamily(family)]


def list_generators() -> list[str]:
    return [family.value for family in GENERATORS]
