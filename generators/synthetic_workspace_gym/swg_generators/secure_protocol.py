from __future__ import annotations

import ast
import base64
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


class _ObservationTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.test_name = "setup"
        self.assertion_index = 0

    def _label(self) -> str:
        self.assertion_index += 1
        return f"{self.test_name}:{self.assertion_index}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        previous = self.test_name
        if node.name.startswith("test_"):
            self.test_name = node.name
        updated = self.generic_visit(node)
        self.test_name = previous
        return updated

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        call = node.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            return self.generic_visit(node)
        method = call.func.attr
        if method == "assertEqual" and call.args:
            value = call.args[0]
        elif method in {"assertTrue", "assertFalse"} and call.args:
            value = ast.Call(func=ast.Name(id="bool", ctx=ast.Load()), args=[call.args[0]], keywords=[])
        else:
            return self.generic_visit(node)
        return ast.copy_location(
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="_swg_record", ctx=ast.Load()),
                    args=[ast.Constant(self._label()), value],
                    keywords=[],
                )
            ),
            node,
        )

    def visit_With(self, node: ast.With) -> ast.AST:
        if len(node.items) != 1:
            return self.generic_visit(node)
        context = node.items[0].context_expr
        if not (
            isinstance(context, ast.Call)
            and isinstance(context.func, ast.Attribute)
            and context.func.attr == "assertRaises"
        ):
            return self.generic_visit(node)
        exception_name = "_swg_exception"
        label = self._label()

        def record(value: ast.expr) -> ast.Expr:
            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="_swg_record", ctx=ast.Load()),
                    args=[ast.Constant(label), value],
                    keywords=[],
                )
            )

        raised_value = ast.Dict(
            keys=[ast.Constant("raised")],
            values=[
                ast.Attribute(
                    value=ast.Call(
                        func=ast.Name(id="type", ctx=ast.Load()),
                        args=[ast.Name(id=exception_name, ctx=ast.Load())],
                        keywords=[],
                    ),
                    attr="__name__",
                    ctx=ast.Load(),
                )
            ],
        )
        missing_value = ast.Dict(keys=[ast.Constant("raised")], values=[ast.Constant(None)])
        return ast.copy_location(
            ast.Try(
                body=[self.visit(item) for item in node.body],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id="BaseException", ctx=ast.Load()),
                        name=exception_name,
                        body=[record(raised_value)],
                    )
                ],
                orelse=[record(missing_value)],
                finalbody=[],
            ),
            node,
        )


_OBSERVATION_SUPPORT = ast.parse(
    """
import os
import sys

_SWG_OBSERVATIONS = []

def _swg_normalize(value):
    if isinstance(value, dict):
        return {str(key): _swg_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_swg_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_swg_normalize(item) for item in value), key=repr)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {"type": type(value).__name__, "repr": repr(value)}

def _swg_record(name, value):
    _SWG_OBSERVATIONS.append({"name": name, "value": _swg_normalize(value)})

def _swg_emit(value):
    payload = json.dumps(_swg_normalize(value), sort_keys=True).encode("utf-8")
    frame = bytes.fromhex(os.environ["SWG_OBSERVATION_FRAME"])
    sys.stdout.buffer.write(frame + len(payload).to_bytes(8, "big") + payload)
    sys.stdout.buffer.flush()
"""
).body


def observation_probe(source: str) -> str:
    child_source = _instrumented_observation_probe(source)
    return (
        textwrap.dedent(
            f"""
            import os
            import secrets
            import subprocess
            import sys
            from pathlib import Path

            child_source = {child_source!r}
            frame = secrets.token_bytes(32)
            environment = {{**dict(os.environ), "SWG_OBSERVATION_FRAME": frame.hex()}}
            child_path = Path(__file__).with_name(f".swg-probe-{{secrets.token_hex(16)}}.py")
            child_path.write_text(child_source, encoding="utf-8")
            try:
                child = subprocess.run(
                    [sys.executable, str(child_path), sys.argv[1]],
                    cwd=str(Path(__file__).resolve().parent),
                    capture_output=True,
                    env=environment,
                )
            finally:
                child_path.unlink(missing_ok=True)
            offset = child.stdout.find(frame)
            if child.returncode != 0 or offset < 0:
                sys.stderr.buffer.write(child.stderr or child.stdout or b"observation probe failed")
                raise SystemExit(child.returncode or 1)
            framed = child.stdout[offset + len(frame) :]
            if len(framed) < 8:
                raise SystemExit("observation probe returned a truncated header")
            size = int.from_bytes(framed[:8], "big")
            payload = framed[8 : 8 + size]
            if len(payload) != size:
                raise SystemExit("observation probe returned a truncated payload")
            Path(sys.argv[2]).write_bytes(payload)
            """
        ).strip()
        + "\n"
    )


def _instrumented_observation_probe(source: str) -> str:
    tree = ast.parse(source)
    tree = _ObservationTransformer().visit(tree)
    assert isinstance(tree, ast.Module)
    insertion = 0
    while insertion < len(tree.body) and isinstance(tree.body[insertion], (ast.Import, ast.ImportFrom)):
        insertion += 1
    tree.body[insertion:insertion] = _OBSERVATION_SUPPORT
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main" and "build_suite" in functions:
            node.body = ast.parse(
                """
workspace = Path(sys.argv[1]).resolve()
suite = build_suite(workspace)
unittest.TextTestRunner(verbosity=0).run(suite)
_swg_emit(_SWG_OBSERVATIONS)
"""
            ).body
        elif isinstance(node, ast.FunctionDef) and node.name == "main":
            _rewrite_capability_runner(node)
    if "execute" in functions and "main" not in functions:
        _rewrite_composite_runner(tree)
    _PruneUnusedExpectedAssignments().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


class _PruneUnusedExpectedAssignments(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        node.body = self._prune(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._prune(node.body)
        return node

    def _prune(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        loaded = {
            item.id
            for statement in statements
            for item in ast.walk(statement)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        }
        retained: list[ast.stmt] = []
        for statement in statements:
            if isinstance(statement, ast.Assign):
                names = {target.id for target in statement.targets if isinstance(target, ast.Name)}
                if names and all("expected" in name.lower() and name not in loaded for name in names):
                    continue
            retained.append(statement)
        return retained


def _project_expression(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "check" and node.args:
        callback = node.args[0]
        if isinstance(callback, ast.Lambda):
            return _project_expression(callback.body)
    if isinstance(node, ast.Compare):
        return _project_expression(node.left)
    if isinstance(node, ast.BoolOp):
        return ast.List(elts=[_project_expression(value) for value in node.values], ctx=ast.Load())
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _project_expression(node.operand)
    return node


def _write_observations_statement(expression: str) -> list[ast.stmt]:
    return ast.parse(f"_swg_emit({expression})").body


def _rewrite_capability_runner(node: ast.FunctionDef) -> None:
    rewritten: list[ast.stmt] = []
    found = False
    for statement in node.body:
        if (
            isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "capabilities" for target in statement.targets)
            and isinstance(statement.value, ast.Dict)
        ):
            statement.value.values = [_project_expression(value) for value in statement.value.values]
            rewritten.append(statement)
            found = True
            continue
        if found:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "capabilities"
                for target in statement.targets
            ):
                assert isinstance(statement.targets[0], ast.Subscript)
                statement.value = _project_expression(statement.value)
                rewritten.append(statement)
                continue
            if isinstance(statement, ast.Assign):
                assigned_names = {target.id for target in statement.targets if isinstance(target, ast.Name)}
                if assigned_names & {"passed", "success", "subscores", "payload"}:
                    continue
                rewritten.append(statement)
                continue
            if isinstance(statement, (ast.Expr, ast.Return)):
                continue
            rewritten.append(statement)
            continue
        rewritten.append(statement)
    if not found:
        raise ValueError("runner main() has no capabilities dictionary")
    rewritten.extend(_write_observations_statement("capabilities"))
    node.body = rewritten


def _rewrite_composite_runner(tree: ast.Module) -> None:
    rewritten: list[ast.stmt] = []
    found = False
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "caps" for target in statement.targets
        ):
            rewritten.append(
                ast.Assign(
                    targets=[ast.Name(id="caps", ctx=ast.Store())],
                    value=ast.Dict(
                        keys=[
                            ast.Constant(name)
                            for name in (
                                "code",
                                "resolved",
                                "normalized",
                                "summary",
                                "acode",
                                "aresolved",
                                "asummary",
                                "source",
                            )
                        ],
                        values=[
                            ast.Name(id="code", ctx=ast.Load()),
                            ast.Name(id="resolved", ctx=ast.Load()),
                            ast.Name(id="normalized", ctx=ast.Load()),
                            ast.Name(id="summary", ctx=ast.Load()),
                            ast.Name(id="acode", ctx=ast.Load()),
                            ast.Name(id="aresolved", ctx=ast.Load()),
                            ast.Name(id="asummary", ctx=ast.Load()),
                            ast.Call(
                                func=ast.Attribute(
                                    value=ast.BinOp(
                                        left=ast.Name(id="source", ctx=ast.Load()),
                                        op=ast.Div(),
                                        right=ast.Constant("run_pipeline.py"),
                                    ),
                                    attr="read_text",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[ast.keyword(arg="encoding", value=ast.Constant("utf-8"))],
                            ),
                        ],
                    ),
                )
            )
            found = True
            continue
        if found:
            continue
        rewritten.append(statement)
    if not found:
        raise ValueError("composite runner has no caps dictionary")
    rewritten.extend(_write_observations_statement("caps"))
    tree.body = rewritten


def _decode_files(encoded: dict[str, str]) -> dict[str, bytes]:
    return {path: base64.b64decode(content) for path, content in encoded.items()}


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _reference_workspace(visible_files: dict[str, bytes], manifest: dict[str, Any], root: Path) -> Path:
    workspace = root / "workspace"
    _write_files(workspace, visible_files)
    for relative_path, content in dict(manifest.get("reference_solution", {}).get("files", {})).items():
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(content).encode("utf-8"))
    return workspace


def build_protocol(
    manifest: dict[str, Any],
    visible_files: dict[str, bytes],
    hidden_files: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Return execution plan, execution-only files, and grader-only hidden files."""
    hidden = _decode_files(hidden_files)
    config = json.loads(hidden["evaluator_config.json"])
    entrypoint = str(manifest["evaluator_entrypoint"])
    execution_files: dict[str, str] = {}

    runner_name = config.get("runner")
    if runner_name:
        runner_source = hidden[str(runner_name)].decode()
        probe_source = observation_probe(runner_source)
        with tempfile.TemporaryDirectory(prefix="swg-reference-probe-") as temporary_dir:
            root = Path(temporary_dir)
            workspace = _reference_workspace(visible_files, manifest, root)
            probe_root = root / "probe"
            probe_root.mkdir()
            probe_path = probe_root / "probe.py"
            probe_path.write_bytes(probe_source.encode("utf-8"))
            output_path = root / "observations.json"
            completed = subprocess.run(
                [sys.executable, str(probe_path), str(workspace), str(output_path)],
                cwd=probe_root,
                capture_output=True,
                text=True,
                timeout=int(manifest["time_limit_seconds"]),
            )
            if completed.returncode != 0 or not output_path.is_file():
                raise RuntimeError(f"reference observation probe failed: {completed.stderr}")
            expected = json.loads(output_path.read_text(encoding="utf-8"))
        execution_files["probe.py"] = base64.b64encode(probe_source.encode()).decode("ascii")
        grader_files = dict(hidden_files)
        grader_files.pop(str(runner_name), None)
        grader_files["expected_observations.json"] = base64.b64encode(_json(expected)).decode("ascii")
        config["secure_protocol"] = "observations-v1"
        config["expected_observations_path"] = "expected_observations.json"
        grader_files["evaluator_config.json"] = base64.b64encode(_json(config)).decode("ascii")
        return (
            {"kind": "observations", "runner": "probe.py", "output": "observations.json"},
            execution_files,
            grader_files,
        )

    if entrypoint.endswith("pipeline_profile:ProfiledPipelineEvaluator"):
        fixture_dir = str(config["hidden_fixture_dir"])
        for path, content in hidden_files.items():
            if path.startswith(f"{fixture_dir}/"):
                execution_files[path] = content
        return (
            {
                "kind": "profiled-command",
                "entrypoint": str(config["entrypoint"]),
                "output": str(config["required_output_path"]),
                "fixture_dir": fixture_dir,
            },
            execution_files,
            hidden_files,
        )

    command_entrypoint = config.get("entrypoint")
    output = config.get("required_output_path") or config.get("output_path")
    return (
        {
            "kind": "command" if command_entrypoint else "artifacts",
            "entrypoint": str(command_entrypoint) if command_entrypoint else None,
            "output": str(output) if output else None,
            "rerun": bool(command_entrypoint),
        },
        execution_files,
        hidden_files,
    )


def _json(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
