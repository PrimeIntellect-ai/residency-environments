"""Failures of trading infrastructure, distinct from strategy execution faults."""


class StrategyInfrastructureError(RuntimeError):
    """The runtime provider or transport failed; the rollout cannot be scored."""
