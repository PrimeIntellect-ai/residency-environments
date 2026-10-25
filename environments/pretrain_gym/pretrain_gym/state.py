import verifiers.v1 as vf

from .workspace import SandboxKind


class PretrainState(vf.State):
    workspace: str = ""
    """The agent's sandbox (Docker container or Prime VM) id, set before the agent starts; tool servers read it."""
    workspace_kind: SandboxKind = "docker"
    rollout: str = ""
    """The trace id, naming this rollout's host directory; tool servers read it."""
    result: dict = {}
    """Final scoring: the loss, metrics, or the corpus error; set at finalize."""
