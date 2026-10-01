from __future__ import annotations

import asyncio
import base64
import gzip
import json
import posixpath
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import verifiers.v1 as vf
from huggingface_hub import hf_hub_download
from pydantic import Field, field_validator
from verifiers.v1.runtimes import provision_runtime

Family = Literal[
    "tabular",
    "script_repair",
    "pipeline",
    "retrieval_workspace",
    "composite_workspace",
]

DEFAULT_MANIFEST = "sft-easy-v4"
WORKDIR = "/workspace"
EXECUTION_ROOT = "/opt/swg-exec"
GRADER_EXECUTION_ROOT = "/opt/swg-grade/execution"
SYSTEM_PROMPT = "Work in /workspace and complete the task using the provided terminal tools."

WORKSPACE_INVENTORY_SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import stat
    import sys

    root = sys.argv[1]
    max_files = int(sys.argv[2])
    max_directories = int(sys.argv[3])
    max_file_bytes = int(sys.argv[4])
    max_total_bytes = int(sys.argv[5])
    directories = []
    files = []
    total_bytes = 0

    if not os.path.isdir(root):
        raise SystemExit("workspace root is missing")
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for name in dirnames:
            path = os.path.join(current, name)
            mode = os.lstat(path).st_mode
            relative = os.path.relpath(path, root)
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise SystemExit(f"unsupported workspace directory entry: {relative}")
            directories.append(relative)
            if len(directories) > max_directories:
                raise SystemExit("workspace directory-count limit exceeded")
        for name in filenames:
            path = os.path.join(current, name)
            info = os.lstat(path)
            relative = os.path.relpath(path, root)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SystemExit(f"unsupported workspace file entry: {relative}")
            if info.st_size > max_file_bytes:
                raise SystemExit(f"workspace per-file byte limit exceeded: {relative}")
            files.append(relative)
            total_bytes += info.st_size
            if len(files) > max_files:
                raise SystemExit("workspace file-count limit exceeded")
            if total_bytes > max_total_bytes:
                raise SystemExit("workspace total-byte limit exceeded")
    print(json.dumps({"directories": directories, "files": files}, sort_keys=True))
    """
).strip()

WORKSPACE_CLEAR_SCRIPT = textwrap.dedent(
    """
    import os
    import shutil
    import sys

    root = sys.argv[1]
    for entry in os.scandir(root):
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.unlink(entry.path)
    """
).strip()


class SyntheticWorkspaceData(vf.TaskData):
    family: Family
    scenario: str
    difficulty: int
    seed: int
    split: str
    generation_fingerprint: str
    task_blob: str
    visible_files: list[str]
    initial_digests: dict[str, str]
    grader_image: str
    dataset_repo: str
    dataset_revision: str


class SyntheticWorkspaceState(vf.State):
    evaluator_result: dict[str, Any] = Field(default_factory=dict)
    changed_file_count: int = 0
    final_file_count: int = 0


class SyntheticWorkspaceTaskConfig(vf.TaskConfig):
    max_workspace_files: int = Field(default=1024, gt=0)
    max_workspace_directories: int = Field(default=1024, gt=0)
    max_file_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    max_workspace_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    max_result_bytes: int = Field(default=2 * 1024 * 1024, gt=0)
    candidate_execution_timeout_seconds: float = Field(default=120.0, gt=0)
    candidate_kill_grace_seconds: float = Field(default=5.0, gt=0)


@dataclass(frozen=True)
class WorkspaceSnapshot:
    directories: tuple[str, ...]
    files: dict[str, bytes]


class SyntheticWorkspaceTasksetConfig(vf.TasksetConfig):
    manifest: str = DEFAULT_MANIFEST
    families: list[Family] = Field(default_factory=list)
    difficulties: list[int] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    task: SyntheticWorkspaceTaskConfig = SyntheticWorkspaceTaskConfig()

    @field_validator("manifest")
    @classmethod
    def validate_manifest_name(cls, value: str) -> str:
        if not value or any(token in value for token in ("/", "\\", "..")):
            raise ValueError("manifest must be a configured SWG curriculum name")
        return value

    @field_validator("difficulties")
    @classmethod
    def validate_difficulties(cls, values: list[int]) -> list[int]:
        if any(value < 1 or value > 5 for value in values):
            raise ValueError("difficulties must be between 1 and 5")
        return values


class SyntheticWorkspaceTask(
    vf.Task[
        SyntheticWorkspaceData,
        SyntheticWorkspaceState,
        SyntheticWorkspaceTaskConfig,
    ]
):
    NEEDS_CONTAINER = True

    async def setup(self, runtime: vf.Runtime) -> None:
        await _restore_baked_workspace(runtime, self.data.name or "swg-task")

    async def finalize(
        self,
        trace: vf.Trace[SyntheticWorkspaceData, SyntheticWorkspaceState],
        runtime: vf.Runtime,
    ) -> None:
        task_blob = await asyncio.to_thread(
            _load_task_blob,
            self.data.dataset_repo,
            self.data.dataset_revision,
            self.data.task_blob,
        )
        if task_blob.get("task_id") != self.data.name:
            raise ValueError("downloaded SWG task blob does not match the selected task")
        manifest = dict(task_blob["manifest"])
        hidden_files = _decode_tree(task_blob["hidden_files"])
        execution_files = _decode_tree(task_blob.get("execution_files", {}))
        execution_plan = dict(task_blob["execution_plan"])
        snapshot = await _snapshot_workspace(
            runtime,
            max_files=self.config.max_workspace_files,
            max_directories=self.config.max_workspace_directories,
            max_file_bytes=self.config.max_file_bytes,
            max_total_bytes=self.config.max_workspace_bytes,
        )

        runner_config = _isolated_config(runtime, image=self.data.image or "")
        async with provision_runtime(runner_config, name=f"swg-runner-{trace.id}") as runner:
            await runner.prepare_setup()
            await _restore_baked_workspace(runner, self.data.name or "swg-task")
            await _replace_workspace(runner, snapshot)
            for relative_path, content in execution_files.items():
                await runner.write(_safe_path(EXECUTION_ROOT, relative_path), content)
            await runner.prepare_execution([])
            execution, execution_outputs, executed_snapshot = await _execute_plan(
                runner,
                execution_plan,
                execution_files,
                max_files=self.config.max_workspace_files,
                max_directories=self.config.max_workspace_directories,
                max_file_bytes=self.config.max_file_bytes,
                max_total_bytes=self.config.max_workspace_bytes,
                timeout_seconds=self.config.candidate_execution_timeout_seconds,
                kill_grace_seconds=self.config.candidate_kill_grace_seconds,
            )

        grader_config = _isolated_config(runtime, image=self.data.grader_image)
        async with provision_runtime(grader_config, name=f"swg-grader-{trace.id}") as grader:
            await grader.prepare_setup()
            mkdir = await grader.run(
                [
                    "/bin/mkdir",
                    "-p",
                    WORKDIR,
                    "/opt/swg-grade/hidden",
                    "/opt/swg-grade/lib",
                    GRADER_EXECUTION_ROOT,
                ],
                {},
            )
            if mkdir.exit_code != 0:
                raise vf.SandboxError(f"grader setup failed: {(mkdir.stderr or mkdir.stdout).strip()[-2000:]}")
            await _replace_workspace(grader, executed_snapshot)
            for relative_path, content in hidden_files.items():
                await grader.write(_safe_path("/opt/swg-grade/hidden", relative_path), content)
            for relative_path, content in execution_outputs.items():
                await grader.write(_safe_path(GRADER_EXECUTION_ROOT, relative_path), content)
            for relative_path, content in _grader_support_files().items():
                await grader.write(_safe_path("/opt/swg-grade/lib", relative_path), content)
            await grader.write(
                "/opt/swg-grade/trusted_grader.py", Path(__file__).with_name("trusted_grader.py").read_bytes()
            )
            await grader.write("/opt/swg-grade/manifest.json", _json_bytes(manifest))
            await grader.write("/opt/swg-grade/initial-digests.json", _json_bytes(self.data.initial_digests))
            await grader.write("/opt/swg-grade/execution.json", _json_bytes(execution))
            await grader.prepare_execution([])
            result = await grader.run(
                [
                    "/usr/local/bin/python",
                    "/opt/swg-grade/trusted_grader.py",
                    WORKDIR,
                    "/opt/swg-grade/hidden",
                    "/opt/swg-grade/manifest.json",
                    "/opt/swg-grade/initial-digests.json",
                    "/opt/swg-grade/execution.json",
                    GRADER_EXECUTION_ROOT,
                    "/opt/swg-grade/result.json",
                ],
                {},
            )
            if result.exit_code != 0:
                detail = (result.stderr or result.stdout).strip()[-2000:]
                raise RuntimeError(f"trusted SWG grader failed: {detail}")
            evaluator_result = json.loads(
                (await grader.read("/opt/swg-grade/result.json", max_bytes=self.config.max_result_bytes)).decode()
            )

        trace.state.changed_file_count = int(evaluator_result.pop("changed_file_count", 0))
        trace.state.final_file_count = int(evaluator_result.pop("final_file_count", 0))
        trace.state.evaluator_result = evaluator_result
        trace.info["synthetic_workspace_gym"] = {
            "family": self.data.family,
            "scenario": self.data.scenario,
            "difficulty": self.data.difficulty,
            "seed": self.data.seed,
            "split": self.data.split,
            "generation_fingerprint": self.data.generation_fingerprint,
            "success": bool(evaluator_result.get("success", False)),
            "failure_labels": list(evaluator_result.get("failure_labels", [])),
            "secure_execution_protocol": execution_plan["kind"],
        }

    @vf.reward(weight=1.0)
    async def workspace_score(self, trace: vf.Trace[SyntheticWorkspaceData, SyntheticWorkspaceState]) -> float:
        return float(trace.state.evaluator_result.get("score", 0.0))

    @vf.metric
    async def workspace_metrics(
        self,
        trace: vf.Trace[SyntheticWorkspaceData, SyntheticWorkspaceState],
    ) -> dict[str, float]:
        result = trace.state.evaluator_result
        metrics = {
            "success": float(bool(result.get("success", False))),
            "changed_file_count": float(trace.state.changed_file_count),
            "final_file_count": float(trace.state.final_file_count),
        }
        for name, value in dict(result.get("subscores", {})).items():
            metrics[f"subscore/{name}"] = float(value)
        return metrics

    async def apply_gold_solution(self, runtime: vf.Runtime) -> None:
        task_blob = await asyncio.to_thread(
            _load_task_blob,
            self.data.dataset_repo,
            self.data.dataset_revision,
            self.data.task_blob,
        )
        solution_files = dict(task_blob["manifest"].get("reference_solution", {}).get("files", {}))
        for relative_path, content in solution_files.items():
            await runtime.write(_safe_path(WORKDIR, relative_path), str(content).encode())


class SyntheticWorkspaceTaskset(vf.Taskset[SyntheticWorkspaceTask, SyntheticWorkspaceTasksetConfig]):
    def load(self) -> Iterable[SyntheticWorkspaceTask]:
        artifacts = _load_artifacts()
        dataset_repo = str(artifacts["dataset_repo"])
        dataset_revision = str(artifacts["dataset_revision"])
        images = dict(artifacts["images"])
        assignments = _load_assignments(dataset_repo, dataset_revision, self.config.manifest)
        image = _runtime_image(images.get(self.config.manifest), f"agent image for {self.config.manifest}")
        grader_image = _runtime_image(images.get("grader"), "grader image")

        selected_families = set(self.config.families)
        selected_difficulties = set(self.config.difficulties)
        selected_tasks = set(self.config.tasks)
        matched = 0
        for assignment in assignments:
            family = str(assignment["family"])
            difficulty = int(assignment["difficulty"])
            task_id = str(assignment["task_id"])
            if selected_families and family not in selected_families:
                continue
            if selected_difficulties and difficulty not in selected_difficulties:
                continue
            if selected_tasks and task_id not in selected_tasks:
                continue
            data = SyntheticWorkspaceData(
                idx=matched,
                name=task_id,
                description=f"SWG {family} workspace task",
                prompt=str(assignment["prompt"]),
                system_prompt=SYSTEM_PROMPT,
                image=image,
                workdir=WORKDIR,
                network_allow=[],
                network_block=["*"],
                timeout=vf.TaskTimeout(setup=120, agent=900, finalize=900, scoring=60),
                resources=vf.TaskResources(cpu=2, memory=4, disk=4),
                family=family,
                scenario=str(assignment["scenario"]),
                difficulty=difficulty,
                seed=int(assignment["seed"]),
                split=str(assignment["split"]),
                generation_fingerprint=str(assignment["generation_fingerprint"]),
                task_blob=str(assignment["task_blob"]),
                visible_files=[str(path) for path in assignment["visible_files"]],
                initial_digests={str(path): str(digest) for path, digest in assignment["initial_digests"].items()},
                grader_image=grader_image,
                dataset_repo=dataset_repo,
                dataset_revision=dataset_revision,
            )
            matched += 1
            yield SyntheticWorkspaceTask(data, self.config.task)
        if matched == 0:
            raise ValueError("no SWG tasks matched the requested manifest filters")


async def _restore_baked_workspace(runtime: vf.Runtime, task_id: str) -> None:
    source = _safe_path("/opt/swg/workspaces", task_id)
    result = await runtime.run(["/bin/test", "-d", source], {})
    if result.exit_code != 0:
        raise RuntimeError(f"visible workspace is missing from the split image: {task_id}")
    result = await runtime.run(["/bin/mkdir", "-p", WORKDIR], {})
    if result.exit_code == 0:
        result = await runtime.run(["/bin/cp", "-a", f"{source}/.", WORKDIR], {})
    if result.exit_code == 0:
        result = await runtime.run(["/bin/rm", "-rf", "/opt/swg/workspaces", "/opt/swg/task_ids.json"], {})
    if result.exit_code != 0:
        raise RuntimeError(f"failed to restore the baked SWG workspace: {result.stderr}")


async def _snapshot_workspace(
    runtime: vf.Runtime,
    *,
    max_files: int,
    max_directories: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> WorkspaceSnapshot:
    inventory = await runtime.run(
        [
            "/usr/local/bin/python",
            "-c",
            WORKSPACE_INVENTORY_SCRIPT,
            WORKDIR,
            str(max_files),
            str(max_directories),
            str(max_file_bytes),
            str(max_total_bytes),
        ],
        {},
    )
    if inventory.exit_code != 0:
        detail = (inventory.stderr or inventory.stdout).strip()[-2000:]
        raise vf.SandboxError(f"failed to inventory bounded workspace: {detail}")
    try:
        payload = json.loads(inventory.stdout)
        directories = tuple(str(path) for path in payload["directories"])
        paths = tuple(str(path) for path in payload["files"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise vf.SandboxError("workspace inventory returned an invalid payload") from exc
    if len(paths) > max_files or len(directories) > max_directories:
        raise vf.SandboxError("workspace inventory exceeded configured entry limits")
    for relative_path in (*directories, *paths):
        _validate_workspace_path(relative_path)

    semaphore = asyncio.Semaphore(32)

    async def read(relative_path: str) -> tuple[str, bytes]:
        async with semaphore:
            content = await runtime.read(
                _safe_path(WORKDIR, relative_path),
                max_bytes=max_file_bytes + 1,
            )
        if len(content) > max_file_bytes:
            raise vf.SandboxError(f"workspace file exceeds the {max_file_bytes}-byte transfer limit")
        return relative_path, content

    values = await asyncio.gather(*(read(path) for path in paths))
    snapshot = WorkspaceSnapshot(directories=directories, files=dict(values))
    total_bytes = sum(len(content) for content in snapshot.files.values())
    if total_bytes > max_total_bytes:
        raise vf.SandboxError(f"workspace exceeds the {max_total_bytes}-byte aggregate transfer limit")
    return snapshot


async def _replace_workspace(runtime: vf.Runtime, snapshot: WorkspaceSnapshot) -> None:
    result = await runtime.run(["/usr/local/bin/python", "-c", WORKSPACE_CLEAR_SCRIPT, WORKDIR], {})
    if result.exit_code == 0:
        directories = [WORKDIR, *(_safe_path(WORKDIR, path) for path in snapshot.directories)]
        result = await runtime.run(["/bin/mkdir", "-p", *directories], {})
    if result.exit_code != 0:
        raise vf.SandboxError(f"failed to replace workspace tree: {(result.stderr or result.stdout).strip()[-2000:]}")
    for relative_path, content in snapshot.files.items():
        await runtime.write(_safe_path(WORKDIR, relative_path), content)


async def _execute_plan(
    runtime: vf.Runtime,
    plan: dict[str, Any],
    execution_files: dict[str, bytes],
    *,
    max_files: int,
    max_directories: int,
    max_file_bytes: int,
    max_total_bytes: int,
    timeout_seconds: float,
    kill_grace_seconds: float,
) -> tuple[dict[str, Any], dict[str, bytes], WorkspaceSnapshot]:
    kind = str(plan["kind"])
    if kind == "observations":
        output = f"{EXECUTION_ROOT}/observations.json"
        await _remove_file(runtime, output)
        result, timed_out = await _run_candidate(
            runtime,
            [
                "/usr/local/bin/python",
                _safe_path(EXECUTION_ROOT, str(plan["runner"])),
                WORKDIR,
                output,
            ],
            {"PYTHONDONTWRITEBYTECODE": "1"},
            timeout_seconds=timeout_seconds,
            kill_grace_seconds=kill_grace_seconds,
        )
        observations = b""
        if result.exit_code == 0:
            try:
                observations = await _read_bounded(runtime, output, max_file_bytes)
            except (FileNotFoundError, vf.SandboxError):
                pass
        snapshot = await _snapshot_workspace(
            runtime,
            max_files=max_files,
            max_directories=max_directories,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
        )
        execution = {
            "kind": kind,
            "visible": _program_result(result, timed_out=timed_out),
            "observations_present": bool(observations),
        }
        outputs = {"observations.json": observations} if observations else {}
        return execution, outputs, snapshot

    if kind == "profiled-command":
        visible, visible_output = await _run_entrypoint(
            runtime,
            str(plan["entrypoint"]),
            str(plan["output"]),
            max_file_bytes,
            timeout_seconds=timeout_seconds,
            kill_grace_seconds=kill_grace_seconds,
        )
        snapshot = await _snapshot_workspace(
            runtime,
            max_files=max_files,
            max_directories=max_directories,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
        )
        fixture_prefix = f"{plan['fixture_dir']}/"
        for relative_path, content in execution_files.items():
            if relative_path.startswith(fixture_prefix):
                await runtime.write(_safe_path(WORKDIR, relative_path.removeprefix(fixture_prefix)), content)
        if visible["timed_out"]:
            hidden = _timeout_program_result("hidden run skipped after visible execution timeout")
            hidden_output = b""
        else:
            hidden, hidden_output = await _run_entrypoint(
                runtime,
                str(plan["entrypoint"]),
                str(plan["output"]),
                max_file_bytes,
                timeout_seconds=timeout_seconds,
                kill_grace_seconds=kill_grace_seconds,
            )
        outputs = {}
        if visible_output:
            outputs["visible-output.json"] = visible_output
        if hidden_output:
            outputs["hidden-output.json"] = hidden_output
        return {"kind": kind, "visible": visible, "hidden": hidden}, outputs, snapshot

    if kind == "command" and plan.get("entrypoint"):
        visible, _ = await _run_entrypoint(
            runtime,
            str(plan["entrypoint"]),
            str(plan["output"]),
            max_file_bytes,
            timeout_seconds=timeout_seconds,
            kill_grace_seconds=kill_grace_seconds,
        )
        execution = {"kind": kind, "visible": visible}
    else:
        execution = {"kind": kind, "visible": {"returncode": 0, "deterministic": True}}
    snapshot = await _snapshot_workspace(
        runtime,
        max_files=max_files,
        max_directories=max_directories,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    return execution, {}, snapshot


async def _run_entrypoint(
    runtime: vf.Runtime,
    entrypoint: str,
    output_path: str,
    max_bytes: int,
    *,
    timeout_seconds: float,
    kill_grace_seconds: float,
) -> tuple[dict[str, Any], bytes]:
    command = ["/usr/local/bin/python", _safe_path(WORKDIR, entrypoint)]
    output_file = _safe_path(WORKDIR, output_path)
    await _remove_file(runtime, output_file)
    environment = {"PYTHONDONTWRITEBYTECODE": "1"}
    first, first_timed_out = await _run_candidate(
        runtime,
        command,
        environment,
        timeout_seconds=timeout_seconds,
        kill_grace_seconds=kill_grace_seconds,
    )
    output = b""
    if first.exit_code == 0:
        try:
            output = await _read_bounded(runtime, output_file, max_bytes)
        except (FileNotFoundError, vf.SandboxError):
            pass
    await _remove_file(runtime, output_file)
    second = first
    second_timed_out = False
    if first.exit_code == 0:
        second, second_timed_out = await _run_candidate(
            runtime,
            command,
            environment,
            timeout_seconds=timeout_seconds,
            kill_grace_seconds=kill_grace_seconds,
        )
    second_output = b""
    if second.exit_code == 0:
        try:
            second_output = await _read_bounded(runtime, output_file, max_bytes)
        except (FileNotFoundError, vf.SandboxError):
            pass
    return (
        {
            **_program_result(first, timed_out=first_timed_out or second_timed_out),
            "rerun_returncode": second.exit_code,
            "deterministic": bool(
                output and first.exit_code == 0 and second.exit_code == 0 and output == second_output
            ),
            "output_present": bool(output),
        },
        output,
    )


async def _run_candidate(
    runtime: vf.Runtime,
    argv: list[str],
    environment: dict[str, str],
    *,
    timeout_seconds: float,
    kill_grace_seconds: float,
) -> tuple[vf.ProgramResult, bool]:
    wrapped = [
        "/usr/bin/timeout",
        "--signal=TERM",
        f"--kill-after={kill_grace_seconds:.3f}s",
        f"{timeout_seconds:.3f}s",
        *argv,
    ]
    rpc_timeout = timeout_seconds + kill_grace_seconds + 30.0
    try:
        result = await asyncio.wait_for(runtime.run(wrapped, environment), timeout=rpc_timeout)
    except TimeoutError:
        return vf.ProgramResult(124, "", "candidate execution exceeded the runtime RPC timeout"), True
    return result, result.exit_code in {124, 137}


async def _remove_file(runtime: vf.Runtime, path: str) -> None:
    result = await runtime.run(["/bin/rm", "-f", path], {})
    if result.exit_code != 0:
        raise vf.SandboxError(f"failed to clear {path!r}: {(result.stderr or result.stdout).strip()[-2000:]}")


def _program_result(result: vf.ProgramResult, *, timed_out: bool = False) -> dict[str, Any]:
    return {
        "returncode": result.exit_code,
        "timed_out": timed_out,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def _timeout_program_result(message: str) -> dict[str, Any]:
    return {"returncode": 124, "timed_out": True, "stdout": "", "stderr": message}


async def _read_bounded(runtime: vf.Runtime, path: str, max_bytes: int) -> bytes:
    content = await runtime.read(path)
    if len(content) > max_bytes:
        raise vf.SandboxError(f"{path!r} exceeds the {max_bytes}-byte transfer limit")
    return content


def _isolated_config(runtime: vf.Runtime, *, image: str) -> vf.RuntimeConfig:
    config = runtime.config
    values = config.model_dump()
    if "image" not in values:
        raise RuntimeError("SWG secure grading requires an image-backed runtime")
    values.update({"image": image, "workdir": WORKDIR})
    if "allow" in values:
        values.update({"allow": [], "block": ["*"]})
    return type(config).model_validate(values)


def _safe_path(root: str, relative_path: str) -> str:
    pure = PurePosixPath(relative_path.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe SWG path: {relative_path!r}")
    return posixpath.join(root, pure.as_posix())


def _validate_workspace_path(relative_path: str) -> None:
    if "\\" in relative_path:
        raise vf.SandboxError(f"workspace path contains an unsupported backslash: {relative_path!r}")
    try:
        relative_path.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise vf.SandboxError("workspace path is not valid UTF-8") from exc
    _safe_path(WORKDIR, relative_path)


def _decode_tree(encoded: dict[str, str]) -> dict[str, bytes]:
    return {path: base64.b64decode(content, validate=True) for path, content in encoded.items()}


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _load_artifacts() -> dict[str, Any]:
    payload = json.loads(files("synthetic_workspace_gym").joinpath("artifacts.json").read_text(encoding="utf-8"))
    if payload.get("dataset_revision") == "PENDING":
        raise RuntimeError("SWG immutable dataset and image artifacts have not been published")
    return payload


def _load_assignments(repo_id: str, revision: str, manifest: str) -> list[dict[str, Any]]:
    path = hf_hub_download(repo_id, f"manifests/{manifest}.jsonl", repo_type="dataset", revision=revision)
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_task_blob(repo_id: str, revision: str, blob_path: str) -> dict[str, Any]:
    path = hf_hub_download(repo_id, blob_path, repo_type="dataset", revision=revision)
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def _require_digest_pin(image: str, label: str) -> None:
    marker = "@sha256:"
    if marker not in image or len(image.rpartition(marker)[2]) != 64:
        raise ValueError(f"{label} must be pinned by digest")


def _runtime_image(artifact: object, label: str) -> str:
    if not isinstance(artifact, dict):
        raise ValueError(f"{label} must declare runtime and container references")
    runtime = str(artifact.get("runtime", ""))
    container = str(artifact.get("container", ""))
    _require_digest_pin(container, f"{label} container")
    if not runtime.startswith("prime/") or ":" not in runtime.rpartition("/")[2]:
        raise ValueError(f"{label} runtime must be a versioned public Prime image reference")
    return runtime


def _grader_support_files() -> dict[str, bytes]:
    package_root = Path(__file__).parent
    payload = {"synthetic_workspace_gym/__init__.py": b""}
    for directory_name in ("evaluators", "schemas", "utils"):
        for path in sorted((package_root / directory_name).rglob("*.py")):
            relative = path.relative_to(package_root).as_posix()
            payload[f"synthetic_workspace_gym/{relative}"] = path.read_bytes()
    return payload


__all__ = [
    "SyntheticWorkspaceData",
    "SyntheticWorkspaceState",
    "SyntheticWorkspaceTask",
    "SyntheticWorkspaceTaskConfig",
    "SyntheticWorkspaceTaskset",
    "SyntheticWorkspaceTasksetConfig",
]
