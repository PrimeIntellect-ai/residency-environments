from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any

JSONValue = Any


class SerializableDataclass:
    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for item in fields(self):
            payload[item.name] = serialize_value(getattr(self, item.name))
        return payload


def serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, SerializableDataclass):
        return value.to_dict()
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    return value


class EnvironmentFamily(str, Enum):
    TABULAR = "tabular"
    SCRIPT_REPAIR = "script_repair"
    PIPELINE = "pipeline"
    RETRIEVAL_WORKSPACE = "retrieval_workspace"
    COMPOSITE_WORKSPACE = "composite_workspace"


class ActionType(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    APPEND_FILE = "append_file"
    LIST_DIRECTORY = "list_directory"
    RUN_SHELL = "run_shell"
    RUN_PYTHON = "run_python"
    SUBMIT = "submit"


@dataclass(slots=True)
class ToolPermissions(SerializableDataclass):
    read_file: bool = True
    write_file: bool = True
    append_file: bool = True
    list_directory: bool = True
    run_shell: bool = True
    run_python: bool = True
    submit: bool = True
    shell_timeout_seconds: int = 10
    python_timeout_seconds: int = 10

    def enabled_tools(self) -> list[str]:
        enabled: list[str] = []
        if self.read_file:
            enabled.append(ActionType.READ_FILE.value)
        if self.write_file:
            enabled.append(ActionType.WRITE_FILE.value)
        if self.append_file:
            enabled.append(ActionType.APPEND_FILE.value)
        if self.list_directory:
            enabled.append(ActionType.LIST_DIRECTORY.value)
        if self.run_shell:
            enabled.append(ActionType.RUN_SHELL.value)
        if self.run_python:
            enabled.append(ActionType.RUN_PYTHON.value)
        if self.submit:
            enabled.append(ActionType.SUBMIT.value)
        return enabled

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolPermissions":
        return cls(**payload)


@dataclass(slots=True)
class ObservabilitySettings(SerializableDataclass):
    show_instruction: bool = True
    include_workspace_digest: bool = True
    redact_hidden_paths: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservabilitySettings":
        return cls(**payload)


@dataclass(slots=True)
class ComplexityProfile(SerializableDataclass):
    file_count: int
    distractor_count: int
    dependency_depth: int
    reasoning_hops: int
    transformation_count: int
    bug_subtlety: int
    execution_required: bool
    output_constraint_strength: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ComplexityProfile":
        return cls(**payload)


@dataclass(slots=True)
class EnvironmentSpec(SerializableDataclass):
    env_family: EnvironmentFamily
    difficulty: int
    seed: int
    scenario_id: str | None = None
    max_steps: int = 12
    time_limit_seconds: int = 60
    tool_permissions: ToolPermissions = field(default_factory=ToolPermissions)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    task_params: dict[str, JSONValue] = field(default_factory=dict)
    evaluator_params: dict[str, JSONValue] = field(default_factory=dict)
    generation_params: dict[str, JSONValue] = field(default_factory=dict)
    complexity_profile: ComplexityProfile | None = None

    def __post_init__(self) -> None:
        self.env_family = EnvironmentFamily(self.env_family)
        if self.scenario_id == "":
            self.scenario_id = None
        if not 1 <= self.difficulty <= 5:
            raise ValueError("difficulty must be between 1 and 5")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnvironmentSpec":
        return cls(
            env_family=EnvironmentFamily(payload["env_family"]),
            difficulty=payload["difficulty"],
            seed=payload["seed"],
            scenario_id=payload.get("scenario_id"),
            max_steps=payload.get("max_steps", 12),
            time_limit_seconds=payload.get("time_limit_seconds", 60),
            tool_permissions=ToolPermissions.from_dict(payload.get("tool_permissions", {})),
            observability=ObservabilitySettings.from_dict(payload.get("observability", {})),
            task_params=payload.get("task_params", {}),
            evaluator_params=payload.get("evaluator_params", {}),
            generation_params=payload.get("generation_params", {}),
            complexity_profile=(
                ComplexityProfile.from_dict(payload["complexity_profile"])
                if payload.get("complexity_profile")
                else None
            ),
        )


@dataclass(slots=True)
class EnvironmentManifest(SerializableDataclass):
    env_id: str
    family: EnvironmentFamily
    difficulty: int
    seed: int
    instruction: str
    workspace_root: str
    visible_files: list[str]
    hidden_root: str
    hidden_files: list[str]
    tool_permissions: ToolPermissions
    max_steps: int
    time_limit_seconds: int
    metadata: dict[str, JSONValue]
    evaluator_entrypoint: str
    reference_solution: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.family = EnvironmentFamily(self.family)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnvironmentManifest":
        return cls(
            env_id=payload["env_id"],
            family=EnvironmentFamily(payload["family"]),
            difficulty=payload["difficulty"],
            seed=payload["seed"],
            instruction=payload["instruction"],
            workspace_root=payload["workspace_root"],
            visible_files=list(payload.get("visible_files", [])),
            hidden_root=payload.get("hidden_root", "hidden"),
            hidden_files=list(payload.get("hidden_files", [])),
            tool_permissions=ToolPermissions.from_dict(payload.get("tool_permissions", {})),
            max_steps=payload["max_steps"],
            time_limit_seconds=payload.get("time_limit_seconds", 60),
            metadata=payload.get("metadata", {}),
            evaluator_entrypoint=payload["evaluator_entrypoint"],
            reference_solution=payload.get("reference_solution", {}),
        )


@dataclass(slots=True)
class EvaluatorResult(SerializableDataclass):
    success: bool
    score: float
    subscores: dict[str, float] = field(default_factory=dict)
    failure_labels: list[str] = field(default_factory=list)
    diagnostics: dict[str, JSONValue] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluatorResult":
        return cls(
            success=payload["success"],
            score=payload["score"],
            subscores=payload.get("subscores", {}),
            failure_labels=payload.get("failure_labels", []),
            diagnostics=payload.get("diagnostics", {}),
            runtime_seconds=payload.get("runtime_seconds", 0.0),
        )


@dataclass(slots=True)
class Action(SerializableDataclass):
    action_type: ActionType
    arguments: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action_type = ActionType(self.action_type)


@dataclass(slots=True)
class ToolObservation(SerializableDataclass):
    success: bool
    message: str
    content: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    listing: list[str] = field(default_factory=list)
    error: str | None = None
    touched_files: list[str] = field(default_factory=list)
    workspace_digest: str | None = None


@dataclass(slots=True)
class TrajectoryEvent(SerializableDataclass):
    step_index: int
    timestamp: str
    action_type: ActionType
    action_arguments: dict[str, JSONValue]
    observation_summary: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    files_touched: list[str] = field(default_factory=list)
    workspace_digest: str | None = None
    success: bool = True

    def __post_init__(self) -> None:
        self.action_type = ActionType(self.action_type)


@dataclass(slots=True)
class ToolState(SerializableDataclass):
    step_index: int
    remaining_steps: int
    available_tools: list[str]
    recent_files: list[str] = field(default_factory=list)
    last_exit_code: int | None = None
    submitted: bool = False


@dataclass(slots=True)
class EpisodeSummary(SerializableDataclass):
    episode_id: str
    env_id: str
    agent_name: str
    step_count: int
    submitted: bool
    duration_seconds: float
    files_touched: list[str]
    evaluation: EvaluatorResult
    artifact_root: str


def utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
