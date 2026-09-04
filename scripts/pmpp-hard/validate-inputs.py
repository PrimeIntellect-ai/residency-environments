"""Run release references against fresh correctness inputs on local NVIDIA Docker.

Run from the repository root with ``uv run scripts/pmpp-hard/validate-inputs.py``.
This exercises the actual CUDA/Triton graders without model calls or performance
measurements. Reference submissions bypass the agent's source-policy checks.
"""

import argparse
import asyncio
import json
import pathlib
import re
import uuid

from pmpp_hard import grader_inputs, pool, scoring
from pmpp_hard.config import PMPPHardConfig
from pmpp_hard.paths import DataTree
from pmpp_hard.provision import grader_files
from verifiers.v1.runtimes import make_runtime


async def validate(args) -> None:
    tree = DataTree()
    config = PMPPHardConfig(preflight=False, include_ids=args.ids or None)
    tasks, _ = pool.build_tasks(config, tree)
    fixed_sanity = "v2-pl-09-penalty_filter_sample"
    tasks = [t for t in tasks if t.task_id in grader_inputs.RANDOMIZED_TASKS or t.task_id == fixed_sanity]
    if not tasks:
        raise ValueError("no randomized tasks selected")
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    for kind, image in (("cuda", args.cuda_image), ("triton", args.triton_image)):
        selected = [t for t in tasks if t.runtime_kind == kind]
        if not selected:
            continue
        runtime = make_runtime(
            scoring.score_runtime_config(config, image), name=f"pmpp-input-check-{uuid.uuid4().hex[:10]}"
        )
        try:
            await runtime.start()
            if kind == "triton":
                await scoring.ensure_python_env(runtime, "validation sandbox")
            for task in selected:
                for rollout in range(args.rollouts):
                    parameter = grader_inputs.draw_parameter(task.task_id)
                    base = f"/checks/{task.task_id}/{rollout}"
                    bundle = tree.bundle(task.task_id)
                    source_dir = tree.sanity_dir(task.task_id) if task.task_id == fixed_sanity else bundle
                    reference = tree.find_reference(task.task_id)
                    if reference is None:
                        raise ValueError(f"{task.task_id}: missing release reference")
                    await runtime.write(f"{base}/Makefile", (bundle / "Makefile").read_bytes())
                    for source in grader_files(source_dir):
                        content = source.read_bytes()
                        if parameter is not None and source.name.startswith("test_"):
                            content = grader_inputs.render(task.task_id, content, parameter)
                        await runtime.write(f"{base}/{source.name}", content)
                    await runtime.write(f"{base}/{task.student_file}", reference.read_bytes())
                    result = await runtime.run(
                        [
                            "bash",
                            "-lc",
                            f"cd {base} && timeout -k 5 180 make {task.test_target} PY=python3 NVCCFLAGS='-O3 -std=c++17 -arch=sm_120' && timeout -k 5 180 ./{task.test_target}",
                        ],
                        {},
                    )
                    output = result.stdout + result.stderr
                    summaries = re.findall(r"passed\s+(\d+)\s*/\s*(\d+)", output)
                    passed = (
                        result.exit_code == 0
                        and bool(summaries)
                        and summaries[-1][0] == summaries[-1][1]
                        and int(summaries[-1][1]) > 0
                    )
                    record = {
                        "task": task.task_id,
                        "rollout": rollout,
                        "mode": "randomized" if parameter is not None else "fixed_sanity",
                        "parameter": f"{parameter:016x}" if parameter is not None else "2ca222e1095e1da9",
                        "passed": passed,
                        "exit_code": result.exit_code,
                        "summary": summaries[-1] if summaries else None,
                    }
                    results.append(record)
                    (args.output / f"{task.task_id}-{rollout}.log").write_text(output)
                    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                    print(json.dumps(record), flush=True)
        finally:
            await runtime.stop()
    failures = [r for r in results if not r["passed"]]
    print(f"reference correctness: {len(results) - len(failures)}/{len(results)} passed; logs: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--cuda-image", default="pmpp-cuda-agent:cu128")
    parser.add_argument("--triton-image", default="pmpp-triton:cu128")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    if arguments.rollouts < 1:
        parser.error("--rollouts must be positive")
    asyncio.run(validate(arguments))
