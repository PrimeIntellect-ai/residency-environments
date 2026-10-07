"""JSON-lines worker installed only in an isolated trading runtime."""

from __future__ import annotations

import argparse
import contextlib
import json
import resource
import sys
import traceback
import types
from pathlib import Path

from alphaverse.strategy.protocol import InputEnvelope
from alphaverse.strategy.sdk import Strategy, StrategyContext, StrategyRunner


def _load_strategy(source: str, entrypoint: str) -> Strategy:
    module_name, separator, class_name = entrypoint.partition(":")
    if not separator or module_name != "strategy" or not class_name.isidentifier():
        raise ValueError("entrypoint must have the form strategy:ClassName")
    module = types.ModuleType(module_name)
    module.__file__ = "strategy.py"
    sys.modules[module_name] = module
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(source, "strategy.py", "exec"), module.__dict__)
        strategy = getattr(module, class_name)()
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
    parser.add_argument("memory_limit_bytes", type=int)
    args = parser.parse_args()

    resource.setrlimit(resource.RLIMIT_AS, (args.memory_limit_bytes, args.memory_limit_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    strategy = _load_strategy(args.source_path.read_text(encoding="utf-8"), args.entrypoint)
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
    print('{"ready":true}', flush=True)
    for line in sys.stdin:
        frame = json.loads(line)
        if frame.get("operation") == "ping":
            print(json.dumps({"request_id": frame["request_id"], "ready": True}), flush=True)
            continue
        try:
            envelope = InputEnvelope.from_json(json.dumps(frame["event"]))
            with contextlib.redirect_stdout(sys.stderr):
                batch = runner.handle(envelope)
            result = {"request_id": frame["request_id"], "batch": json.loads(batch.to_json())}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result = {"request_id": frame["request_id"], "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
