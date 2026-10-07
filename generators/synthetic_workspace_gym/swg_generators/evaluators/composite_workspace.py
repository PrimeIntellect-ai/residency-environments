from __future__ import annotations

from .retrieval_workspace import RetrievalWorkspaceEvaluator


class CompositeWorkspaceEvaluator(RetrievalWorkspaceEvaluator):
    """Capability-scored evaluator for retrieval-guided pipeline tasks."""
