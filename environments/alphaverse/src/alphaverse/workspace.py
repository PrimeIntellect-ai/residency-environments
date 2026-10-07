"""Task-owned files installed into an Alphaverse agent runtime."""

from __future__ import annotations

from pathlib import Path

from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData

from alphaverse.prop_trader import (
    PROP_PARTICIPANT_ID,
    prop_baseline_source,
    prop_role_readme,
)

_WORKSPACE = Path(__file__).resolve().parent / "agent_workspace"
_WORKSPACE_FILES = ("README.md", "API.md", "market_capture.py")


async def install_task_workspace(data: TaskData, runtime: Runtime) -> None:
    """Install public task files and any evaluator-owned role seed."""

    for name in _WORKSPACE_FILES:
        await runtime.write(name, (_WORKSPACE / name).read_bytes())
    if getattr(data, "participant_id", None) != PROP_PARTICIPANT_ID:
        return
    seed_profile = str(getattr(data, "prop_seed_profile", "passive"))
    framing = str(getattr(data, "prop_framing", "incumbent"))
    control_scope = str(getattr(data, "prop_control_scope", "full_source"))
    await runtime.write(
        "ROLE.md",
        prop_role_readme(seed_profile, framing, control_scope).encode("utf-8"),
    )
    await runtime.write(
        "strategy.py",
        prop_baseline_source(seed_profile).encode("utf-8"),
    )


__all__ = ["install_task_workspace"]
