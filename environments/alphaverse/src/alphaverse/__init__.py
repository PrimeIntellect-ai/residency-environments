"""Public strategy symbol plus lazy Verifiers plugin exports."""

from alphaverse.public_types import Side

__all__ = ["AlphaverseEnv", "AlphaverseHarness", "AlphaverseTaskset", "Side"]


def __getattr__(name: str):
    if name == "AlphaverseEnv":
        from alphaverse.env import AlphaverseEnv

        return AlphaverseEnv
    if name == "AlphaverseHarness":
        from alphaverse.eval_harness import AlphaverseHarness

        return AlphaverseHarness
    if name == "AlphaverseTaskset":
        from alphaverse.verifiers_v1 import AlphaverseTaskset

        return AlphaverseTaskset
    raise AttributeError(name)
