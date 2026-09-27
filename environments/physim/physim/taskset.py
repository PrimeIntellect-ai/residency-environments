"""R6 task hooks and MCP tools; execution is stock Verifiers v1.

The agent uses a provided harness in DockerRuntime, driven by SingleAgentEnv/eval.
The trusted MCP tool process keeps the simulator and grader outside that box.
Only observations and submitted workspace files cross the task boundary.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from functools import wraps
from importlib.resources import files
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Literal

import numpy as np
import verifiers.v1 as vf
from pydantic import BaseModel, Field, model_validator

from physim.blobround6_explore import ExperimentService, RequestError
from physim.bundles import Bundle

from . import evaluation as E
from .artifact_store import MARKER, read_artifact_files
from .rewards import DEFAULT_PRECISION, REWARD_MAPPING, precision_label, precision_reward
from .sandbox import ExecutionLimits

PROMPT_CONDITION = "interface-only-v3-log-reward"
PROTOCOL = f"r6-verifiers-v1-bash-1-{PROMPT_CONDITION}"
AGENT_IMAGE = "physim-agent:0.12.2"
DEFAULT_OUTPUT = Path("outputs/r6/artifacts")
DEFAULT_PROMPT = "Investigate the laboratory and submit your executable predictor."


class R6State(vf.State):
    # This state channel is host-only; these fields are never tool arguments.
    container_id: str = ""
    prompt_condition: str = PROMPT_CONDITION
    reward_precision: float = DEFAULT_PRECISION
    output: str = ""
    submitted: bool = False
    artifact: str | None = None
    checks: list[dict] = Field(default_factory=list)
    experiments: list[dict] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    origin: dict = Field(default_factory=dict)
    infrastructure_error: str | None = None
    exploration_closed: bool = False
    checkpoint: dict = Field(default_factory=dict)
    rejected_requests: list[dict] = Field(default_factory=list)


class HubBundleConfig(BaseModel):
    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    cache: Path | None = None
    offline: bool = False


class R6ToolsConfig(vf.ToolsetConfig):
    bundle: Path | None = None
    bundle_source: HubBundleConfig | None = None
    max_experiments: int | None = Field(1000, ge=1)
    max_total_tu: float | None = Field(50000, gt=0, le=1_000_000)
    max_validation_attempts: int | None = Field(128, ge=1)
    max_submission_attempts: int | None = Field(128, ge=1)
    predictor_limits: ExecutionLimits = ExecutionLimits()
    reward_precision: float = Field(DEFAULT_PRECISION, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def unambiguous_bundle(self):
        if self.bundle is not None and self.bundle_source is not None:
            raise ValueError("Select either a local bundle or bundle_source, not both")
        return self


class R6TaskConfig(vf.TaskConfig):
    tools: R6ToolsConfig = R6ToolsConfig()
    output_root: Path = DEFAULT_OUTPUT
    agent_image: str = AGENT_IMAGE
    coding_interface: Literal["shell", "ipython"] = "shell"
    setup_timeout: float = Field(180, gt=0, le=900)
    checkpoint_artifact: Path | None = None


class R6Config(vf.TasksetConfig):
    task: R6TaskConfig = R6TaskConfig()
    prompt: str = DEFAULT_PROMPT


def required_bundle(config: R6ToolsConfig, *, profile="evaluation") -> Bundle:
    path = config.bundle
    if config.bundle_source is not None:
        from physim.hub import fetch_bundle

        path = fetch_bundle(**config.bundle_source.model_dump(), profile=profile)
    if path is None:
        raise ValueError(
            "No world selected: Physim has no default world and does not automatically select "
            "eval-ready registry entries. Set env.taskset.task.tools.bundle to a verified local "
            "bundle directory, pass --env.taskset.task.tools.bundle /path/to/bundle to eval, "
            "or set bundle_source with an explicit HF repo, full commit revision, and bundle path."
        )
    bundle = Bundle(path, profile=profile)
    bundle.check_runtime()
    if config.bundle is not None:
        config.bundle = bundle.root
    return bundle


def public_roster(config: R6ToolsConfig):
    if config.bundle is not None:
        return Bundle(config.bundle, profile="simulation").roster
    if config.bundle_source is not None:
        return required_bundle(config, profile="simulation").roster
    return E.E.DEFAULT_ROSTER


def public_prompt(config: R6ToolsConfig, coding_interface: str = "shell") -> str:
    """Render one ordered contract, shared by the prompt and workspace manual."""

    def budget(value):
        return "unlimited" if value is None else f"{value:g}"

    roster = public_roster(config)
    name = "agent_spec_v1.txt" if roster.protocol == E.E.R6.LEGACY_PROTOCOL else "agent_spec.txt"
    text = files("physim").joinpath("data", name).read_text()
    coding_tools = "Use the bash and edit tools to run commands and work with files."
    if coding_interface == "ipython":
        coding_tools = (
            "Use the harness's persistent IPython session to analyze data and write files.\n"
            "Use the MCP skill wrappers advertised by the harness for laboratory calls;\n"
            "follow their actual import and calling instructions."
        )
    values = dict(
        n_ports=roster.n_ports,
        last_port=roster.n_ports - 1,
        example_port=min(2, roster.n_ports - 1),
        max_experiments=budget(config.max_experiments),
        max_total_tu=budget(config.max_total_tu),
        max_validation_attempts=budget(config.max_validation_attempts),
        max_submission_attempts=budget(config.max_submission_attempts),
        predictor_cpus=config.predictor_limits.cpus,
        predictor_memory_gib=config.predictor_limits.memory_gib,
        predictor_cpu_seconds=config.predictor_limits.cpu_seconds,
        predictor_wall_seconds=config.predictor_limits.wall_seconds,
        coding_tools=coding_tools,
        reward_precision=precision_label(config.reward_precision),
        reward_threshold=f"10^(-{precision_label(config.reward_precision)})",
    )
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def _container(state: R6State) -> str:
    if not re.fullmatch(r"[0-9a-f]{12,64}", state.container_id):
        raise vf.ToolsetError("missing trusted Docker runtime identity")
    return state.container_id


def _snapshot(state: R6State, target: Path, limits=None) -> dict:
    # Reuse the existing bounded, regular-file-only artifact transport. This
    # does not create a runtime or run an agent; Verifiers owns that runtime.
    return E.Sandbox.export_workspace(
        SimpleNamespace(name=_container(state), limits=limits or ExecutionLimits()), target, excludes=("./.vf-*",)
    )


def _put_observation(state: R6State, path: Path) -> None:
    # No host path or code from the model is accepted. The trusted filename is
    # fixed by the experiment counter. O_NOFOLLOW prevents a workspace symlink
    # from redirecting this write within the agent container.
    script = """import os,sys
p='/observations/'+sys.argv[1]
fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
with os.fdopen(fd,'wb') as f: f.write(sys.stdin.buffer.read())
"""
    E.docker(
        ["exec", "-i", "--user", "0", _container(state), "python", "-c", script, path.name], data=path.read_bytes()
    )


def _checkpoint_file(root: Path, name: str, digest: str) -> tuple[str, bytes]:
    """Read only a hashed regular file in a trusted prior artifact manifest."""
    rel = PurePosixPath(name)
    if rel.is_absolute() or not rel.parts or any(p in (".", "..") or p.startswith(".vf-") for p in rel.parts):
        raise vf.TaskError("invalid checkpoint file path")
    path = root.joinpath(*rel.parts)
    if any(root.joinpath(*rel.parts[:i]).is_symlink() for i in range(1, len(rel.parts) + 1)) or not path.is_file():
        raise vf.TaskError("checkpoint file is not a regular file")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise vf.TaskError("checkpoint file hash mismatch")
    return str(rel), data


def load_checkpoint(artifact: Path, reward_precision: float = DEFAULT_PRECISION) -> tuple[dict, list, list]:
    """Recover public workspace/data only; never copy host state or private origin."""
    artifact = artifact.resolve()
    state_path = artifact.parent / "laboratory_state.json"
    prior = json.loads(state_path.read_text())
    if prior.get("prompt_condition") != PROMPT_CONDITION:
        raise vf.TaskError("checkpoint belongs to a different or unrecorded prompt condition")
    if prior.get("reward_precision") != reward_precision:
        raise vf.TaskError("checkpoint belongs to a different or unrecorded reward precision")
    check = next((c for c in prior["checks"] if c["path"] == artifact.name), None)
    if not check or not check.get("validation", {}).get("ok"):
        raise vf.TaskError("checkpoint must be a previously validated artifact")
    manifest = check["snapshot"]["files"]
    if (artifact / MARKER).exists():
        try:
            files = read_artifact_files(artifact, manifest)
        except E.SandboxError as exc:
            raise vf.TaskError(str(exc)) from exc
    else:
        files = [_checkpoint_file(artifact, f["path"], f["sha256"]) for f in manifest]
    if "predictor.py" not in {name for name, _ in files}:
        raise vf.TaskError("checkpoint has no predictor.py")
    experiments = prior["experiments"]
    observations = []
    for index, event in enumerate(experiments, 1):
        if event["file"] != f"experiment_{index:03d}.npz":
            raise vf.TaskError("invalid checkpoint observation sequence")
        observations.append(_checkpoint_file(artifact.parent / "observations", event["file"], event["sha256"]))
    if len(experiments) != prior["usage"]["experiments"]:
        raise vf.TaskError("checkpoint experiment accounting mismatch")
    metadata = dict(
        source_artifact=str(artifact),
        source_state_sha256=E.file_digest(state_path),
        files=manifest,
        observations=[dict(file=e["file"], sha256=e["sha256"]) for e in experiments],
        inherited_experiments=len(experiments),
        inherited_charged_tu=prior["usage"]["charged_tu"],
        exploration_closed=True,
    )
    return metadata, files, observations


def recovery_prompt() -> str:
    return (
        "\n\nThis is a submission recovery after a framework interruption. Your last "
        "validated predictor and its supporting files are restored in /workspace, "
        "with your collected data in /observations. Exploration is closed: no new "
        "experiments are allowed. Inspect the restored predictor if needed, then "
        "call laboratory_submit to validate and freeze it. No accuracy feedback "
        "has been provided. This recovery inherits the earlier experiment/time usage."
    )


def _check(state: R6State, config: R6ToolsConfig, *, final: bool) -> dict:
    if state.submitted:
        return dict(ok=True, accepted=True, finalized=True)
    kind = "submit" if final else "validate"
    attempt = 1 + sum(row["kind"] == kind for row in state.checks)
    cap = config.max_submission_attempts if final else config.max_validation_attempts
    if cap is not None and attempt > cap:
        state.rejected_requests.append(dict(kind=kind, error="attempt limit reached", attempt=attempt))
        return dict(ok=False, error=f"{kind} attempt limit reached", finalized=False)
    target = Path(state.output) / f"{kind}_{attempt:02d}"
    event = dict(kind=kind, attempt=attempt, path=target.name)
    state.checks.append(event)
    try:
        event["snapshot"] = _snapshot(state, target, config.predictor_limits)
    except E.SandboxError as exc:
        report = dict(ok=False, gate=E.SUBMISSION_GATE_VERSION, failure=dict(stage="artifact", error=str(exc)[-3000:]))
    else:
        # A validator-container startup failure is infrastructure failure, not
        # evidence that the submitted predictor violates the contract.
        report = E.validate_predictor(
            target,
            Path(state.output) / "observations",
            roster=public_roster(config),
            execution_limits=config.predictor_limits,
        )
    event["validation"] = report
    if final and report["ok"]:
        state.submitted = True
        state.artifact = str(target)
    E.dump(Path(state.output) / "laboratory_state.json", state.model_dump(exclude={"artifacts"}))
    return dict(report, finalized=state.submitted, accepted=final and report["ok"])


class LaboratoryTools(vf.Toolset[R6ToolsConfig, R6State]):
    TOOL_PREFIX = "laboratory"

    def __init__(self, config):
        super().__init__(config)
        self.service = None
        self.lock = asyncio.Lock()
        self.state_transaction = asyncio.Lock()

    def _with_state(self, fn):
        @wraps(fn)
        async def public_call(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                # Record failure inside VF's state transaction. Returning a
                # neutral response lets VF push this state; the task stop hook
                # then raises a framework error, so this can never earn a score.
                self.state.infrastructure_error = self.state.infrastructure_error or f"{type(exc).__name__}: {exc}"
                if self.state.output:
                    E.dump(
                        Path(self.state.output) / "laboratory_state.json", self.state.model_dump(exclude={"artifacts"})
                    )
                return json.dumps(dict(error="Laboratory service failed; infrastructure intervention is required"))

        wrapped = super()._with_state(public_call)

        @wraps(wrapped)
        async def serialized(*args, **kwargs):
            # VF's state channel replaces the full state. Serialize the entire
            # pull/call/push transaction, including under parallel-tool harnesses.
            async with self.state_transaction:
                try:
                    return await wrapped(*args, **kwargs)
                except Exception:
                    # State-channel failures happen outside the public call.
                    # Their URLs and host paths are not agent-facing feedback.
                    raise vf.ToolsetError("Laboratory connection failed") from None

        return serialized

    def _service(self):
        if self.service is None:
            bundle = required_bundle(self.config, profile="simulation")
            self.state.origin = bundle.references()
            self.service = ExperimentService(
                bundle.make_oracle(),
                limits=bundle.limits,
                roster=bundle.roster,
                max_experiments=self.config.max_experiments,
                max_total_tu=self.config.max_total_tu,
            )
        return self.service

    @vf.tool
    async def experiment(self, actions: list[dict], queries: list[dict]) -> str:
        """Run from the prepared start and save NPZ measurements. Use the action
        keys and scheduling rules in AGENT_SPEC.md. queries:
        [{sensor:'device0'|'device1'|'global',t:[strictly increasing times]}].
        All timestamps are absolute, 0..50 tu. Return path, shapes and usage.
        """
        async with self.lock:
            if self.state.submitted:
                return json.dumps(dict(error="predictor already submitted"))
            if self.state.exploration_closed:
                return json.dumps(dict(error="exploration closed for checkpoint recovery", usage=self.state.usage))
            try:
                service = self._service()
                data = await asyncio.to_thread(service.experiment, actions, queries)
            except RequestError as exc:
                self.state.usage = service.usage()
                self.state.rejected_requests.append(dict(kind="experiment", error=str(exc)))
                E.dump(Path(self.state.output) / "laboratory_state.json", self.state.model_dump(exclude={"artifacts"}))
                return json.dumps(dict(error=str(exc), usage=self.state.usage))
            except Exception as exc:
                # Internal failures are infrastructure errors, never feedback
                # about the hidden implementation. Keep details on the host.
                self.state.infrastructure_error = f"{type(exc).__name__}: {exc}"
                E.dump(Path(self.state.output) / "laboratory_state.json", self.state.model_dump(exclude={"artifacts"}))
                raise vf.ToolsetError("Laboratory execution failed; infrastructure intervention is required") from None
            self.state.usage = service.usage()
            path = Path(self.state.output) / "observations" / f"experiment_{self.state.usage['experiments']:03d}.npz"
            request = dict(actions=actions, queries=queries)
            np.savez_compressed(
                path, request=json.dumps(request), **{f"query{i}": a for i, a in enumerate(data["samples"])}
            )
            await asyncio.to_thread(_put_observation, self.state, path)
            arrays = {f"query{i}": list(a.shape) for i, a in enumerate(data["samples"])}
            self.state.experiments.append(
                dict(request=request, file=path.name, sha256=E.file_digest(path), arrays=arrays)
            )
            E.dump(Path(self.state.output) / "laboratory_state.json", self.state.model_dump(exclude={"artifacts"}))
            return json.dumps(dict(path="/observations/" + path.name, arrays=arrays, usage=self.state.usage))

    @vf.tool
    async def usage(self) -> str:
        """Return experiment/time budgets and validation/submission attempt counts."""
        return json.dumps(
            dict(
                **self.state.usage,
                validation_attempts=sum(c["kind"] == "validate" for c in self.state.checks),
                submission_attempts=sum(c["kind"] == "submit" for c in self.state.checks),
            )
        )

    @vf.tool
    async def validate(self) -> str:
        """Run example requests to check execution, output format and repeatability.
        No accuracy feedback or experiment cost. Does not submit; repair and retry."""
        async with self.lock:
            return json.dumps(await asyncio.to_thread(_check, self.state, self.config, final=False))

    @vf.tool
    async def submit(self) -> str:
        """Validate and freeze predictor.py plus supporting files; success ends
        exploration. Errors leave the workspace open for repairs."""
        async with self.lock:
            return json.dumps(await asyncio.to_thread(_check, self.state, self.config, final=True))


class R6Data(vf.TaskData):
    protocol: str = PROTOCOL


class R6Task(vf.Task[R6Data, R6State, R6TaskConfig]):
    @classmethod
    def toolsets(cls, config):
        if config.tools.colocated or config.tools.runtime.type != "subprocess" or config.tools.url:
            raise ValueError("R6 laboratory tools must run in their trusted host process")
        return [LaboratoryTools(config.tools)]

    async def setup(self, trace, runtime):
        bundle = required_bundle(self.config.tools)
        if runtime.config.type != "docker" or not runtime.config.network_restricted:
            raise vf.TaskError("R6 requires the Verifiers Docker runtime with framework-only network access")
        state = trace.state
        state.container_id = runtime.info.id
        state.reward_precision = self.config.tools.reward_precision
        output = self.config.output_root.resolve() / trace.id
        output.mkdir(parents=True, exist_ok=False)
        (output / "observations").mkdir()
        state.output = str(output)
        state.usage = dict(
            experiments=0,
            max_experiments=self.config.tools.max_experiments,
            charged_tu=0,
            max_total_tu=self.config.tools.max_total_tu,
        )
        result = await runtime.run(["mkdir", "-p", "/workspace", "/observations"], {})
        if result.exit_code:
            raise vf.TaskError("could not initialize laboratory workspace")
        await runtime.write(
            "/workspace/AGENT_SPEC.md", public_prompt(self.config.tools, self.config.coding_interface).encode()
        )
        if self.config.checkpoint_artifact is not None:
            metadata, files, observations = await asyncio.to_thread(
                load_checkpoint, self.config.checkpoint_artifact, self.config.tools.reward_precision
            )
            state.checkpoint = metadata
            state.exploration_closed = True
            state.usage.update(
                experiments=metadata["inherited_experiments"],
                charged_tu=metadata["inherited_charged_tu"],
                exploration_closed=True,
            )
            for name, data in files:
                result = await runtime.run(["mkdir", "-p", str(PurePosixPath("/workspace", name).parent)], {})
                if result.exit_code:
                    raise vf.TaskError("could not restore checkpoint directory")
                await runtime.write("/workspace/" + name, data)
            for name, data in observations:
                path = output / "observations" / name
                path.write_bytes(data)
                await asyncio.to_thread(_put_observation, state, path)
            E.dump(output / "laboratory_state.json", state.model_dump(exclude={"artifacts"}))
        trace.info["r6"] = dict(
            protocol=self.data.protocol,
            artifact_directory=str(output),
            references=bundle.references(),
            verifiers_version=__import__("verifiers").__version__,
            gate=E.SUBMISSION_GATE_VERSION,
            checkpoint=state.checkpoint,
        )

    @vf.stop
    async def infrastructure_failed(self, trace: vf.Trace) -> bool:
        if trace.state.infrastructure_error:
            raise vf.TaskError("Laboratory service failed; inspect the host diagnostic record")
        return False

    @vf.stop
    async def submitted(self, trace: vf.Trace) -> bool:
        # This installed ACP adapter reports an externally stopped RLM prompt
        # as a harness error. Let RLM return its final answer naturally; submit
        # has already frozen the artifact and disabled further experiments.
        return trace.state.submitted and self.config.coding_interface == "shell"

    async def finalize(self, trace, runtime):
        state = trace.state
        if state.infrastructure_error:
            raise vf.TaskError("Laboratory service failed; inspect the host diagnostic record")
        if not state.output:
            return
        # Stock VF ends naturally on final text or a configured limit. The task
        # collects the last executable artifact, without another model call.
        if not state.submitted:
            report = await asyncio.to_thread(_check, state, self.config.tools, final=True)
            trace.info["r6"]["final_collection"] = report
        E.dump(Path(state.output) / "laboratory_state.json", state.model_dump(exclude={"artifacts"}))
        audit = dict(
            stop_condition=trace.stop_condition,
            truncated=trace.is_truncated,
            submitted=state.submitted,
            final_collection=trace.info.get("r6", {}).get("final_collection"),
            model_turns=trace.num_turns,
            input_tokens=trace.num_input_tokens,
            output_tokens=trace.num_output_tokens,
            length_finished_calls=[i for i, call in enumerate(trace.calls) if call.finish_reason == "length"],
            agent_limits={
                name: getattr(trace.agent.config, name, None)
                for name in ("max_turns", "max_input_tokens", "max_output_tokens", "max_total_tokens")
            },
            timeouts=trace.agent.config.timeout.model_dump(),
            laboratory_limits=self.config.tools.model_dump(mode="json", exclude={"bundle", "bundle_source"}),
            usage=state.usage,
            rejected_requests=state.rejected_requests,
            validation_attempts=sum(c["kind"] == "validate" for c in state.checks),
            submission_attempts=sum(c["kind"] == "submit" for c in state.checks),
        )
        trace.info.setdefault("r6", {})["limit_audit"] = audit
        E.dump(Path(state.output) / "limit_audit.json", audit)

    @vf.reward(weight=1)
    async def prediction_reward(self, trace) -> float:
        state = trace.state
        info = trace.info.setdefault("r6", {})
        info["reward_precision"] = self.config.tools.reward_precision
        info["reward_mapping"] = REWARD_MAPPING
        if not state.submitted:
            attempted = any(c.get("snapshot") for c in state.checks)
            info["score_kind"] = "infinity" if attempted else "nan"
            info["score_reason"] = "invalid_predictor" if attempted else "no_predictor"
            info["primary_joint_energy"] = None
            return 0.0
        context = dict(
            trace_id=trace.id,
            protocol=self.data.protocol,
            models=sorted({call.model for call in trace.calls if call.model}),
            harness=trace.agent.config.harness.id,
            coding_interface=self.config.coding_interface,
            reward_precision=self.config.tools.reward_precision,
            reward_mapping=REWARD_MAPPING,
            agent_limits={
                name: getattr(trace.agent.config, name, None)
                for name in ("max_turns", "max_input_tokens", "max_output_tokens")
            },
            laboratory_budget={
                name: getattr(self.config.tools, name)
                for name in ("max_experiments", "max_total_tu", "max_validation_attempts", "max_submission_attempts")
            },
            usage=state.usage,
            checkpoint=state.checkpoint,
            spend=info.get("spend_final"),
            laboratory_state_sha256=E.file_digest(Path(state.output) / "laboratory_state.json"),
        )
        grade = await asyncio.to_thread(
            E.grade,
            Path(state.artifact),
            Path(state.output) / "observations",
            Path(state.output),
            bundle=required_bundle(self.config.tools),
            members=64,
            run_context=context,
            execution_limits=self.config.tools.predictor_limits,
        )
        info["grade"] = grade
        score = grade["primary_joint_energy"]
        info["score_kind"] = "finite" if score is not None else "infinity"
        info["primary_joint_energy"] = score
        # Preserve the lower-is-better scientific score. VF rewards are larger
        # is better; this monotone transform is explicit and recorded.
        return precision_reward(score, self.config.tools.reward_precision)


class R6Taskset(vf.Taskset[R6Task, R6Config]):
    DEFAULT_HARNESS = "bash"

    def load(self):
        required_bundle(self.config.task.tools)
        protocol = (
            PROTOCOL
            if self.config.task.coding_interface == "shell"
            else f"r6-verifiers-v1-ipython-1-{PROMPT_CONDITION}"
        )
        n_ports = public_roster(self.config.task.tools).n_ports
        protocol += "-k-" + precision_label(self.config.task.tools.reward_precision)
        if n_ports != 12:
            protocol += f"-ports-{n_ports}"
        if self.config.prompt != DEFAULT_PROMPT:
            protocol += "-prompt-" + hashlib.sha256(self.config.prompt.encode()).hexdigest()[:12]
        system_prompt = public_prompt(self.config.task.tools, self.config.task.coding_interface)
        if self.config.task.checkpoint_artifact is not None:
            protocol += (
                "-checkpoint-"
                + hashlib.sha256(str(self.config.task.checkpoint_artifact.resolve()).encode()).hexdigest()[:12]
            )
            system_prompt += recovery_prompt()
        yield R6Task(
            R6Data(
                idx=0,
                name="r6-prepared-laboratory",
                protocol=protocol,
                prompt=self.config.prompt,
                system_prompt=system_prompt,
                image=self.config.task.agent_image,
                workdir="/workspace",
                network_allow=[],
                resources=vf.TaskResources(cpu=4, memory=8),
                timeout=vf.TaskTimeout(setup=self.config.task.setup_timeout, agent=86400, finalize=180, scoring=900),
            ),
            self.config.task,
        )


if __name__ == "__main__":
    LaboratoryTools.run()
