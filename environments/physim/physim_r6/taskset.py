"""Compatibility module for saved configs using physim_r6."""

from physim.taskset import (  # noqa: F401
    AGENT_IMAGE,
    DEFAULT_OUTPUT,
    DEFAULT_PROMPT,
    PROTOCOL,
    E,
    LaboratoryTools,
    R6Config,
    R6Data,
    R6State,
    R6Task,
    R6TaskConfig,
    R6Taskset,
    R6ToolsConfig,
    load_checkpoint,
    public_prompt,
    recovery_prompt,
    required_bundle,
)

if __name__ == "__main__":
    LaboratoryTools.run()
