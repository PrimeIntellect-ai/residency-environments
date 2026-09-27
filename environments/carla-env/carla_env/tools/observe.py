from __future__ import annotations

from typing import Any


def observe(state: Any = None) -> str:
    """
    Get the latest environment observation.

    CarlaEnv intercepts this tool to return the post-tick world state.
    The function body is a fallback.
    """
    if state is None:
        return "Error: no state"
    obs = state.get("observation")
    return str(obs) if obs is not None else "(no observation)"
