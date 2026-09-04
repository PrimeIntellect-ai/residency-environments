"""Public strategy symbol plus lazy Verifiers plugin exports."""

from alphaverse.models import Side

__all__ = ["AlphaverseHarness", "AlphaverseTaskset", "Side"]


def __getattr__(name: str):
    if name == "AlphaverseHarness":
        from alphaverse.eval_harness import AlphaverseHarness

        return AlphaverseHarness
    if name == "AlphaverseTaskset":
        from alphaverse.verifiers_v1 import AlphaverseTaskset

        return AlphaverseTaskset
    raise AttributeError(name)
