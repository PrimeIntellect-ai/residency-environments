"""Exception types for environment loading, path handling, and scoring.

``PoolIntegrityError`` also inherits from ``RuntimeError`` for compatibility with
callers that handle task-loading failures as runtime errors.
"""


class PMPPError(Exception):
    """Base exception for environment-specific failures."""


class PoolIntegrityError(PMPPError, RuntimeError):
    """Raised when the on-disk task pool fails integrity validation."""


class PathDisciplineError(PMPPError):
    """Raised when a sandbox path violates workspace boundaries."""


class SanityGenerationError(PMPPError):
    """Raised when a held-out sanity artifact cannot be generated safely."""


class ScoreInfraError(PMPPError):
    """Raised when scoring infrastructure cannot be provisioned or run."""


class GraderExecError(PMPPError):
    """Raised when grader invocation fails before marker evaluation."""
