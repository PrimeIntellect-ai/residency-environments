"""Run release references against fresh correctness inputs on local NVIDIA Docker.

Run from the repository root with ``uv run scripts/pmpp-hard/validate-inputs.py``.
This exercises the actual CUDA/CUTLASS/Triton graders without model calls or performance
measurements. Reference submissions bypass the agent's source-policy checks.
"""

import argparse
import asyncio
import hashlib
import json
import pathlib
import re
import uuid

from pmpp_hard import grader_inputs, pool, scoring
from pmpp_hard.config import PMPPHardConfig
from pmpp_hard.paths import DataTree
from pmpp_hard.provision import grader_files
from verifiers.v1.runtimes import make_runtime


def probe_outputs(source: bytes) -> bytes:
    """Fingerprint only guarded output downloads, not input copies or timing."""
    match = re.search(rb"struct GuardedDeviceBuffer \{.*?^\};", source, re.M | re.S)
    if match is None or match[0].count(b"return host;") != 1:
        raise ValueError("unsupported guarded-output probe layout")
    guarded = match[0].replace(
        b"return host;",
        b"pmpp_validation_digest.add(host.data(), host.size() * sizeof(T)); return host;",
    )
    source = source[: match.start()] + guarded + source[match.end() :]
    anchor = b"#define CUDA_CHECK"
    if source.count(anchor) != 1:
        raise ValueError("unsupported CUDA_CHECK probe anchor")
    helper = b"""struct PMPPValidationDigest {
    uint64_t hash = 1469598103934665603ULL;
    void add(const void* data, size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < size; ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
    }
    ~PMPPValidationDigest() {
        std::printf("pmpp_validation_output=%016llx\\n", (unsigned long long)hash);
    }
};
static PMPPValidationDigest pmpp_validation_digest;
"""
    return source.replace(anchor, helper + anchor)


async def validate(args) -> None:
    tree = DataTree()
    config = PMPPHardConfig(preflight=False, include_ids=args.ids or None)
    tasks, stats = pool.build_tasks(config, tree)
    if stats.unmatched_include_ids:
        raise ValueError(f"unknown task IDs: {stats.unmatched_include_ids}")
    fixed_sanity = "v2-pl-09-penalty_filter_sample"
    unsupported = [
        t.task_id for t in tasks if t.task_id not in grader_inputs.RANDOMIZED_TASKS and t.task_id != fixed_sanity
    ]
    if args.ids and unsupported:
        raise ValueError(f"tasks have no input validation mode: {unsupported}")
    tasks = [t for t in tasks if t.task_id in grader_inputs.RANDOMIZED_TASKS or t.task_id == fixed_sanity]
    if not tasks:
        raise ValueError("no randomized tasks selected")
    if args.parameter and any(t.task_id == fixed_sanity for t in tasks):
        raise ValueError("explicit parameters cannot select fixed task 09")
    if args.probe_outputs and any(t.runtime_kind == "triton" for t in tasks):
        raise ValueError("guarded-output probes require CUDA/CUTLASS tasks")
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output}")
    images = {"cuda": args.cuda_image, "cutlass": args.cutlass_image, "triton": args.triton_image}
    unsupported_runtimes = {t.runtime_kind for t in tasks} - images.keys()
    if unsupported_runtimes:
        raise ValueError(f"unsupported validation runtimes: {sorted(unsupported_runtimes)}")
    for kind, image in images.items():
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
                parameters = args.parameter or [
                    grader_inputs.draw_parameter(task.task_id) for _ in range(args.rollouts)
                ]
                for rollout, parameter in enumerate(parameters):
                    base = f"/checks/{task.task_id}/{rollout}"
                    bundle = tree.bundle(task.task_id)
                    source_dir = tree.sanity_dir(task.task_id) if task.task_id == fixed_sanity else bundle
                    reference = tree.find_reference(task.task_id)
                    if reference is None:
                        raise ValueError(f"{task.task_id}: missing release reference")
                    await runtime.write(f"{base}/Makefile", (bundle / "Makefile").read_bytes())
                    source_hashes = {}
                    for source in grader_files(source_dir):
                        content = source.read_bytes()
                        if parameter is not None and source.name.startswith("test_"):
                            content = grader_inputs.render(task.task_id, content, parameter)
                        source_hashes[source.name] = hashlib.sha256(content).hexdigest()
                        if args.probe_outputs and source.name.startswith("test_"):
                            content = probe_outputs(content)
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
                    digests = re.findall(r"pmpp_validation_output=([0-9a-f]{16})", output)
                    passed = (
                        result.exit_code == 0
                        and bool(summaries)
                        and summaries[-1][0] == summaries[-1][1]
                        and int(summaries[-1][1]) > 0
                        and (not args.probe_outputs or len(digests) == 1)
                    )
                    record = {
                        "task": task.task_id,
                        "runtime_kind": kind,
                        "image": image,
                        "rollout": rollout,
                        "mode": "randomized" if parameter is not None else "fixed_sanity",
                        "parameter": f"{parameter:016x}" if parameter is not None else "2ca222e1095e1da9",
                        "passed": passed,
                        "exit_code": result.exit_code,
                        "summary": summaries[-1] if summaries else None,
                        "policy": grader_inputs.POLICY,
                        "source_sha256": source_hashes,
                        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                        "parameter_override": bool(args.parameter),
                        "output_probe": digests[0] if len(digests) == 1 else None,
                    }
                    results.append(record)
                    (args.output / f"{task.task_id}-{rollout}.log").write_text(output)
                    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                    print(json.dumps(record), flush=True)
        finally:
            await runtime.stop()
    expected = {
        (t.task_id, rollout)
        for t in tasks
        for rollout in range(len(args.parameter) if args.parameter else args.rollouts)
    }
    observed = {(r["task"], r["rollout"]) for r in results}
    if observed != expected or len(results) != len(expected):
        raise RuntimeError(f"incomplete validation coverage: {len(results)}/{len(expected)} runs")
    failures = [r for r in results if not r["passed"]]
    print(f"reference correctness: {len(results) - len(failures)}/{len(results)} passed; logs: {args.output}")
    if failures:
        raise SystemExit(1)
    if args.probe_outputs:
        for task in tasks:
            by_parameter = {}
            for row in results:
                if row["task"] == task.task_id:
                    by_parameter.setdefault(row["parameter"], set()).add(row["output_probe"])
            if any(len(digests) != 1 for digests in by_parameter.values()):
                raise RuntimeError(f"{task.task_id}: same-parameter output replay differs")
            if len({next(iter(digests)) for digests in by_parameter.values()}) != len(by_parameter):
                raise RuntimeError(f"{task.task_id}: distinct parameters did not change reference outputs")
        print(f"same/different-parameter output probes: {len(tasks)}/{len(tasks)} passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument(
        "--parameter",
        type=lambda value: int(value, 0),
        action="append",
        help="Repeatable diagnostic-only odd uint64; overrides --rollouts",
    )
    parser.add_argument(
        "--probe-outputs",
        action="store_true",
        help="Instrument guarded output downloads; requires repeated and distinct --parameter values",
    )
    parser.add_argument("--cuda-image", default="pmpp-cuda-agent:cu128")
    parser.add_argument("--cutlass-image", default="pmpp-cutlass:12.8.1")
    parser.add_argument("--triton-image", default="pmpp-triton:cu128")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    if arguments.rollouts < 1:
        parser.error("--rollouts must be positive")
    if arguments.parameter and any(not 0 < p < 1 << 64 or p % 2 == 0 for p in arguments.parameter):
        parser.error("--parameter must be an odd uint64")
    if arguments.probe_outputs and (
        not arguments.parameter
        or len(set(arguments.parameter)) < 2
        or len(set(arguments.parameter)) == len(arguments.parameter)
    ):
        parser.error("--probe-outputs requires at least two distinct parameters and one repeated parameter")
    asyncio.run(validate(arguments))
