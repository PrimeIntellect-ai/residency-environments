"""Eleusis harness.

This deliberately uses the standard Verifiers null-harness program unchanged.
The model interaction protocol is therefore the plain one: tool selection is
automatic and tool results flow directly into the next model call without
injected messages or recovery turns.
"""

from verifiers.v1.harnesses.null.harness import PROGRAM_SOURCE, NullHarness

ELEUSIS_PROGRAM_SOURCE = PROGRAM_SOURCE


class EleusisHarness(NullHarness):
    """Standard chat/tool harness, bundled so Eleusis is runnable by itself."""
