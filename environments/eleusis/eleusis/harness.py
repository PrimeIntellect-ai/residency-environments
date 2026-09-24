"""Eleusis harness.

This deliberately uses the standard Verifiers null-harness program unchanged.
The model interaction protocol is therefore the plain one: tool selection is
automatic and tool results flow directly into the next model call without
injected messages or recovery turns.
"""

from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE

ELEUSIS_PROGRAM_SOURCE = CHAT_PROGRAM_SOURCE


class EleusisHarness(NullHarness):
    """Standard chat/tool harness, bundled so Eleusis is runnable by itself."""
