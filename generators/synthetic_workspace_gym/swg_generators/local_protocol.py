from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from swg_generators.secure_protocol import build_protocol
from swg_generators.utils.paths import list_relative_files


def evaluate_with_shipped_protocol(evaluator, workspace_path: Path, manifest, hidden_root: Path):
    """Exercise the same execution/grading boundary used by the shipped taskset."""

    manifest_payload = manifest.to_dict()
    visible_files = {path: (workspace_path / path).read_bytes() for path in list_relative_files(workspace_path)}
    hidden_files = {
        path: base64.b64encode((hidden_root / path).read_bytes()).decode("ascii")
        for path in list_relative_files(hidden_root)
    }
    plan, execution_files, grader_files = build_protocol(manifest_payload, visible_files, hidden_files)

    with tempfile.TemporaryDirectory(prefix="swg-shipped-eval-") as temporary_dir:
        root = Path(temporary_dir)
        candidate = root / "workspace"
        shutil.copytree(workspace_path, candidate)
        execution_root = root / "execution"
        grader_root = root / "hidden"
        outputs_root = grader_root / "precomputed_outputs"
        execution_root.mkdir()
        grader_root.mkdir()
        outputs_root.mkdir()
        _write_encoded_tree(execution_root, execution_files)
        _write_encoded_tree(grader_root, grader_files)
        execution, outputs, graded_workspace = _execute_plan(
            candidate,
            execution_root,
            plan,
            timeout=int(manifest.time_limit_seconds),
        )
        for relative_path, content in outputs.items():
            destination = outputs_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        (grader_root / "precomputed_execution.json").write_text(
            json.dumps(execution, sort_keys=True),
            encoding="utf-8",
        )
        return evaluator.evaluate(graded_workspace, manifest, grader_root)


def _execute_plan(
    workspace: Path,
    execution_root: Path,
    plan: dict[str, Any],
    *,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, bytes], Path]:
    kind = str(plan["kind"])
    if kind == "observations":
        output = execution_root / "observations.json"
        completed = _run(
            [sys.executable, str(execution_root / str(plan["runner"])), str(workspace), str(output)],
            cwd=execution_root,
            timeout=timeout,
        )
        observations = output.read_bytes() if completed.returncode == 0 and output.is_file() else b""
        execution = {
            "kind": kind,
            "visible": _program_result(completed),
            "observations_present": bool(observations),
        }
        return execution, ({"observations.json": observations} if observations else {}), workspace

    if kind == "profiled-command":
        visible, visible_output = _run_entrypoint(
            workspace,
            str(plan["entrypoint"]),
            str(plan["output"]),
            timeout=timeout,
        )
        graded_workspace = workspace.parent / "graded-workspace"
        shutil.copytree(workspace, graded_workspace)
        fixture_root = execution_root / str(plan["fixture_dir"])
        for source in fixture_root.rglob("*"):
            if source.is_file():
                destination = workspace / source.relative_to(fixture_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        hidden, hidden_output = _run_entrypoint(
            workspace,
            str(plan["entrypoint"]),
            str(plan["output"]),
            timeout=timeout,
        )
        outputs = {}
        if visible_output:
            outputs["visible-output.json"] = visible_output
        if hidden_output:
            outputs["hidden-output.json"] = hidden_output
        return {"kind": kind, "visible": visible, "hidden": hidden}, outputs, graded_workspace

    if kind == "command" and plan.get("entrypoint"):
        visible, _ = _run_entrypoint(
            workspace,
            str(plan["entrypoint"]),
            str(plan["output"]),
            timeout=timeout,
        )
        return {"kind": kind, "visible": visible}, {}, workspace
    return {"kind": kind, "visible": {"returncode": 0, "deterministic": True}}, {}, workspace


def _run_entrypoint(workspace: Path, entrypoint: str, output_path: str, *, timeout: int):
    output = workspace / output_path
    output.unlink(missing_ok=True)
    command = [sys.executable, str(workspace / entrypoint)]
    first = _run(command, cwd=workspace, timeout=timeout)
    first_output = output.read_bytes() if first.returncode == 0 and output.is_file() else b""
    output.unlink(missing_ok=True)
    second = first
    if first.returncode == 0:
        second = _run(command, cwd=workspace, timeout=timeout)
    second_output = output.read_bytes() if second.returncode == 0 and output.is_file() else b""
    return (
        {
            **_program_result(first),
            "rerun_returncode": second.returncode,
            "deterministic": bool(
                first_output and first.returncode == 0 and second.returncode == 0 and first_output == second_output
            ),
            "output_present": bool(first_output),
        },
        first_output,
    )


def _run(argv: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            argv,
            124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "candidate execution timed out",
        )


def _program_result(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timed_out": completed.returncode == 124,
    }


def _write_encoded_tree(root: Path, files: dict[str, str]) -> None:
    for relative_path, encoded in files.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(base64.b64decode(encoded))
