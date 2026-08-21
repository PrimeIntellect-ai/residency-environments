"""JSON-lines worker process for one user-authored Python strategy."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
import traceback
import types
from pathlib import Path

from alphaverse.strategy.policy import PUBLIC_IMPORTS, SAFE_STDLIB_IMPORTS
from alphaverse.strategy.protocol import InputEnvelope
from alphaverse.strategy.sdk import Strategy, StrategyContext, StrategyRunner


def _deny_private_capabilities(event: str, args: tuple[object, ...]) -> None:
    """Deny common escape paths after trusted worker initialization."""

    del args
    if (
        event == "open"
        or event == "subprocess.Popen"
        or event == "os.system"
        or event.startswith("os.spawn")
        or event.startswith("socket.")
        or event.startswith("ctypes.")
    ):
        raise PermissionError(f"strategy capability denied: {event}")


def _prime_strategy_imports() -> None:
    """Load permitted modules before the audit hook denies filesystem access."""

    for module_name in (*SAFE_STDLIB_IMPORTS, *PUBLIC_IMPORTS):
        importlib.import_module(module_name)
    # The evaluator's trusted passive-prop seed subclasses this implementation.
    # Player uploads still cannot import it because the AST policy exposes only
    # PUBLIC_IMPORTS.
    importlib.import_module("alphaverse.reference_strategies")


def _load_strategy(source: str, entrypoint: str) -> Strategy:
    module_name, separator, class_name = entrypoint.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("entrypoint must have the form module:ClassName")
    if module_name != "strategy":
        raise ValueError("entrypoint module must match the uploaded source filename")
    code = compile(source, "strategy.py", "exec")
    _prime_strategy_imports()
    sys.addaudithook(_deny_private_capabilities)
    module = types.ModuleType(module_name)
    module.__file__ = "strategy.py"
    with contextlib.redirect_stdout(sys.stderr):
        exec(code, module.__dict__)
    strategy_type = getattr(module, class_name)
    strategy = strategy_type()
    if not isinstance(strategy, Strategy):
        raise TypeError("entrypoint must construct an alphaverse Strategy")
    return strategy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=Path)
    parser.add_argument("entrypoint")
    parser.add_argument("participant_id")
    parser.add_argument("strategy_instance_id")
    parser.add_argument("product_id")
    parser.add_argument("parameters_json")
    parser.add_argument("seed", type=int)
    parser.add_argument("max_actions", type=int)
    args = parser.parse_args()

    source = args.source_path.read_text(encoding="utf-8")
    strategy = _load_strategy(source, args.entrypoint)
    runner = StrategyRunner(
        strategy,
        StrategyContext(
            participant_id=args.participant_id,
            strategy_instance_id=args.strategy_instance_id,
            product_id=args.product_id,
            parameters=json.loads(args.parameters_json),
            seed=args.seed,
        ),
        max_actions_per_callback=args.max_actions,
    )
    sys.stdout.write('{"ready":true}\n')
    sys.stdout.flush()

    for line in sys.stdin:
        try:
            envelope = InputEnvelope.from_json(line)
            with contextlib.redirect_stdout(sys.stderr):
                batch = runner.handle(envelope)
            response = batch.to_json()
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = json.dumps(
                {"error": type(exc).__name__, "message": str(exc)},
                separators=(",", ":"),
                sort_keys=True,
            )
        sys.stdout.write(response + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
