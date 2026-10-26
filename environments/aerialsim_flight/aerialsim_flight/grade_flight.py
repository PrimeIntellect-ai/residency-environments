"""Run a submitted flight program with imports limited to the drone API.

Usage: python3 grade_flight.py /path/to/flight.py
Exit 0 when the program ran. Exit 2 when the gate rejected it.

The gate raises the bar inside the process. The real boundary is the
grader box itself: fresh, separate, no network, thrown away after.
"""

import ast
import builtins
import os
import sys

SIM_HOST = "127.0.0.1"
SIM_PORT = int(os.environ.get("AERIALSIM_SIM_PORT", "41451"))

ALLOWED_IMPORTS = {"aerialsim_flight", "aerialsim_flight.drone", "math", "time"}
FORBIDDEN_NAMES = {
    "eval",
    "exec",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "compile",
    "open",
    "breakpoint",
    "memoryview",
    "input",
    "__import__",
    "__builtins__",
    "__loader__",
    "__spec__",
}


def gate(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    return f"import of {alias.name} is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module is None:
                return "relative imports are not allowed"
            if node.module not in ALLOWED_IMPORTS:
                return f"import from {node.module} is not allowed"
            for alias in node.names:
                if alias.name == "*":
                    return "import * is not allowed"
                if f"{node.module}.{alias.name}" in sys.modules:
                    return f"importing the module {alias.name} is not allowed"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in ("modules", "sys"):
                return f"attribute {node.attr} is not allowed"
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                return f"name {node.id} is not allowed"


def guarded_import(name, *args, **kwargs):
    if name not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name} is not allowed")
    return __import__(name, *args, **kwargs)


def lock_connect_to_local() -> None:
    """Make connect() always talk to the simulator in this box."""
    try:
        from aerialsim_flight import drone
    except ImportError:
        return

    original = drone.connect

    def connect(host=SIM_HOST, port=SIM_PORT):
        if (host, port) != (SIM_HOST, SIM_PORT):
            raise ValueError(f"connect is fixed to {SIM_HOST}:{SIM_PORT}")
        return original(SIM_HOST, SIM_PORT)

    drone.connect = connect


def main() -> int:
    source = open(sys.argv[1]).read()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print(f"GATE_REJECTED syntax error: {exc}")
        return 2
    reason = gate(tree)
    if reason:
        print(f"GATE_REJECTED {reason}")
        return 2
    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in FORBIDDEN_NAMES}
    safe_builtins["__import__"] = guarded_import
    lock_connect_to_local()
    namespace = {"__name__": "__main__", "__builtins__": safe_builtins}
    code = compile(tree, sys.argv[1], "exec")
    exec(code, namespace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
