from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _decode_tree(encoded: dict[str, str]) -> dict[str, bytes]:
    return {path: base64.b64decode(content, validate=True) for path, content in encoded.items()}


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)


def _command_result(workspace: Path, entrypoint: str, output_path: str) -> tuple[dict[str, object], bytes]:
    command = [sys.executable, str(workspace / entrypoint)]
    output = workspace / output_path
    output.unlink(missing_ok=True)
    first = _run(command, workspace)
    first_bytes = output.read_bytes() if output.is_file() else b""
    output.unlink(missing_ok=True)
    second = _run(command, workspace) if first.returncode == 0 else first
    second_bytes = output.read_bytes() if output.is_file() else b""
    return (
        {
            "returncode": first.returncode,
            "rerun_returncode": second.returncode,
            "stdout": first.stdout,
            "stderr": first.stderr,
            "deterministic": bool(
                first_bytes and first.returncode == 0 and second.returncode == 0 and first_bytes == second_bytes
            ),
            "output_present": bool(first_bytes),
        },
        first_bytes,
    )


def _execute(workspace: Path, row: dict[str, object]) -> tuple[dict[str, object], dict[str, bytes]]:
    plan = dict(row["execution_plan"])
    execution_files = _decode_tree(dict(row.get("execution_files", {})))
    kind = str(plan["kind"])
    if kind == "observations":
        probe = workspace.parent / "probe.py"
        probe.write_bytes(execution_files[str(plan["runner"])])
        output = workspace.parent / "observations.json"
        output.unlink(missing_ok=True)
        result = _run([sys.executable, str(probe), str(workspace), str(output)], workspace.parent)
        observations = output.read_bytes() if result.returncode == 0 and output.is_file() else b""
        execution = {
            "kind": kind,
            "visible": {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
            "observations_present": bool(observations),
        }
        return execution, ({"observations.json": observations} if observations else {})
    if kind == "profiled-command":
        visible, visible_output = _command_result(workspace, str(plan["entrypoint"]), str(plan["output"]))
        prefix = f"{plan['fixture_dir']}/"
        for relative_path, content in execution_files.items():
            if relative_path.startswith(prefix):
                destination = workspace / relative_path.removeprefix(prefix)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        hidden, hidden_output = _command_result(workspace, str(plan["entrypoint"]), str(plan["output"]))
        outputs = {}
        if visible_output:
            outputs["visible-output.json"] = visible_output
        if hidden_output:
            outputs["hidden-output.json"] = hidden_output
        return {"kind": kind, "visible": visible, "hidden": hidden}, outputs
    if kind == "command":
        visible, _ = _command_result(workspace, str(plan["entrypoint"]), str(plan["output"]))
        return {"kind": kind, "visible": visible}, {}
    return {"kind": kind, "visible": {"returncode": 0, "deterministic": True}}, {}


def _workspace_context(export_root: Path, task_id: str) -> Path:
    matches = list((export_root / "images").glob(f"*/workspaces/{task_id}"))
    if not matches:
        raise FileNotFoundError(f"no visible image context for {task_id}")
    return matches[0]


def _validate_one(export_root: Path, environment_root: Path, row: dict[str, object]) -> None:
    task_id = str(row["task_id"])
    with tempfile.TemporaryDirectory(prefix="swg-secure-oracle-") as temporary_dir:
        root = Path(temporary_dir)
        workspace = root / "workspace"
        shutil.copytree(_workspace_context(export_root, task_id), workspace)
        initial = _digests(workspace)
        manifest = dict(row["manifest"])
        for relative_path, content in dict(manifest.get("reference_solution", {}).get("files", {})).items():
            path = workspace / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
        for planted_name in (".swg-observations.json", ".swg-visible-output.json", ".swg-hidden-output.json"):
            (workspace / planted_name).write_text('{"forged": true}', encoding="utf-8")
        execution, execution_outputs = _execute(workspace, row)

        grade_root = root / "grade"
        hidden = grade_root / "hidden"
        library = grade_root / "lib" / "synthetic_workspace_gym"
        _write_tree(hidden, _decode_tree(dict(row["hidden_files"])))
        library.mkdir(parents=True)
        (library / "__init__.py").write_text("", encoding="utf-8")
        for directory in ("evaluators", "schemas", "utils"):
            shutil.copytree(environment_root / "synthetic_workspace_gym" / directory, library / directory)
        grader = grade_root / "trusted_grader.py"
        shutil.copy2(environment_root / "synthetic_workspace_gym" / "trusted_grader.py", grader)
        manifest_path = grade_root / "manifest.json"
        initial_path = grade_root / "initial.json"
        execution_path = grade_root / "execution.json"
        execution_outputs_root = grade_root / "execution"
        result_path = grade_root / "result.json"
        execution_outputs_root.mkdir(parents=True)
        _write_tree(execution_outputs_root, execution_outputs)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        initial_path.write_text(json.dumps(initial), encoding="utf-8")
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
        grader_command = [
            sys.executable,
            str(grader),
            str(workspace),
            str(hidden),
            str(manifest_path),
            str(initial_path),
            str(execution_path),
            str(execution_outputs_root),
            str(result_path),
        ]
        completed = _run(grader_command, grade_root)
        if completed.returncode != 0:
            raise RuntimeError(f"{task_id}: trusted grader failed: {completed.stderr}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result.get("success") or float(result.get("score", 0.0)) != 1.0:
            raise AssertionError(f"{task_id}: reference solution did not score 1.0: {result}")

        kind = str(row["execution_plan"]["kind"])
        if kind in {"observations", "profiled-command"}:
            config = json.loads((hidden / "evaluator_config.json").read_text(encoding="utf-8"))
            if kind == "observations":
                planted = hidden / str(config["expected_observations_path"])
                (workspace / ".swg-observations.json").write_bytes(planted.read_bytes())
                forged_execution = {
                    "kind": kind,
                    "visible": {"returncode": 1, "stdout": "", "stderr": "probe failed"},
                    "observations_present": False,
                }
            else:
                (workspace / ".swg-visible-output.json").write_bytes((hidden / "expected_output.json").read_bytes())
                hidden_expected = hidden / str(config["hidden_expected_path"])
                (workspace / ".swg-hidden-output.json").write_bytes(hidden_expected.read_bytes())
                failed_run = {
                    "returncode": 0,
                    "rerun_returncode": 0,
                    "deterministic": False,
                    "output_present": False,
                    "stdout": "",
                    "stderr": "",
                }
                forged_execution = {"kind": kind, "visible": failed_run, "hidden": failed_run}
            execution_path.write_text(json.dumps(forged_execution), encoding="utf-8")
            shutil.rmtree(execution_outputs_root)
            execution_outputs_root.mkdir()
            completed = _run(grader_command, grade_root)
            if completed.returncode != 0:
                raise RuntimeError(f"{task_id}: forged-output rejection grade failed: {completed.stderr}")
            forged_result = json.loads(result_path.read_text(encoding="utf-8"))
            if forged_result.get("success"):
                raise AssertionError(f"{task_id}: agent-planted execution output produced success")
            if kind == "observations" and float(forged_result["subscores"].get("tests_passed", 0.0)) != 0.0:
                raise AssertionError(f"{task_id}: planted observations were scored: {forged_result}")
            if kind == "profiled-command" and float(forged_result.get("score", 0.0)) != 0.0:
                raise AssertionError(f"{task_id}: planted profile outputs were scored: {forged_result}")

        shutil.rmtree(workspace)
        shutil.copytree(_workspace_context(export_root, task_id), workspace)
        untouched_initial = _digests(workspace)
        untouched_execution, untouched_outputs = _execute(workspace, row)
        initial_path.write_text(json.dumps(untouched_initial), encoding="utf-8")
        execution_path.write_text(json.dumps(untouched_execution), encoding="utf-8")
        shutil.rmtree(execution_outputs_root)
        execution_outputs_root.mkdir()
        _write_tree(execution_outputs_root, untouched_outputs)
        completed = _run(grader_command, grade_root)
        if completed.returncode != 0:
            raise RuntimeError(f"{task_id}: untouched trusted grade failed: {completed.stderr}")
        untouched_result = json.loads(result_path.read_text(encoding="utf-8"))
        realization = dict(manifest.get("metadata", {}).get("difficulty_realization", {}))
        reward_limit = float(realization["unmodified_reward_limit"])
        if float(untouched_result.get("score", 0.0)) > reward_limit:
            raise AssertionError(
                f"{task_id}: untouched score {untouched_result['score']} exceeds {reward_limit}: {untouched_result}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate each secure SWG protocol against a reference solution.")
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--environment-root", type=Path, required=True)
    args = parser.parse_args()
    representatives: dict[str, dict[str, object]] = {}
    for path in sorted((args.export_root / "dataset" / "tasks").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            row = json.load(source)
        representatives.setdefault(str(row["execution_plan"]["kind"]), row)
    expected = {"artifacts", "command", "observations", "profiled-command"}
    if set(representatives) != expected:
        raise AssertionError(f"missing protocol representatives: {sorted(expected - set(representatives))}")
    for kind, row in sorted(representatives.items()):
        _validate_one(args.export_root, args.environment_root, row)
        print(f"{kind}: {row['task_id']} passed")


if __name__ == "__main__":
    main()
