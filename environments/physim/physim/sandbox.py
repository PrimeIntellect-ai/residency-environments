"""Resource policy and error types for isolated predictor execution."""

from dataclasses import asdict, dataclass

from .artifact_store import SandboxError as SandboxError


@dataclass(frozen=True)
class ExecutionLimits:
    """Host-owned resource policy, shared by validation and grading."""

    cpu_seconds: int = 20
    wall_seconds: int = 30
    cpus: int = 1
    memory_gib: int = 1
    artifact_mib: int = 64
    file_mib: int = 20

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("predictor execution limits must be positive integers")


class SandboxInfrastructureError(SandboxError):
    """Host/container transport failure; details are not predictor feedback."""


class ExecutionLimitError(SandboxError):
    """An execution resource boundary interrupted the submitted program."""
