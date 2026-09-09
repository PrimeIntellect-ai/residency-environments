"""Exercise the real paired gate with no-op CUDA/CUTLASS and stale-output Triton controls.

These deliberately bad submissions bypass correctness/source gates to isolate
benchmark output verification. They are regression controls, not a proof against
arbitrary offline lookup or grader tampering.
"""

import argparse
import asyncio
import hashlib
import json
import pathlib
import re
import subprocess
import types
import uuid

from pmpp_hard import pool, scoring
from pmpp_hard.config import PMPPHardConfig
from pmpp_hard.paths import DataTree, Workspace
from pmpp_hard.provision import provision_grader
from verifiers.v1.runtimes import make_runtime

STALE_WRAPPER = """
def _stale_wrapper(fn):
    cache = {}
    def clone(v):
        if isinstance(v, (tuple, list)):
            return tuple(clone(x) for x in v)
        return v.clone()
    def key(v):
        if isinstance(v, torch.Tensor):
            return (tuple(v.shape), str(v.dtype), tuple(v.stride()))
        if isinstance(v, (tuple, list)):
            return tuple(key(x) for x in v)
        return v
    def cached(*args, **kwargs):
        k = (tuple(key(v) for v in args), tuple((n, key(v)) for n, v in sorted(kwargs.items())))
        if k not in cache:
            cache[k] = clone(fn(*args, **kwargs))
        return cache[k]
    return cached
"""


async def validate(args):
    tree = DataTree()
    config = PMPPHardConfig(preflight=False, include_ids=args.ids or None)
    tasks, stats = pool.build_tasks(config, tree)
    if stats.unmatched_include_ids:
        raise ValueError(f"unknown task IDs: {stats.unmatched_include_ids}")
    if args.ids and any(not t.perf_gated for t in tasks):
        raise ValueError("selection must contain only performance-gated tasks")
    defaults = {"v2-pl-01-route_compact_reduce", "v2-pl-13-fused_gemm_nvfp4_epilogue"}
    tasks = [t for t in tasks if t.perf_gated and (args.ids or t.runtime_kind == "triton" or t.task_id in defaults)]
    if not tasks:
        raise ValueError("no performance tasks selected")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output}")
    images = {"cuda": args.cuda_image, "cutlass": args.cutlass_image, "triton": args.triton_image}
    results = []
    for task in tasks:
        bundle = tree.bundle(task.task_id)
        if task.runtime_kind == "triton":
            reference = tree.find_reference(task.task_id)
            if reference is None:
                raise ValueError(f"{task.task_id}: no release reference")
            (bench,) = bundle.glob("bench_*.py")
            (fn,) = set(re.findall(r"\bsolution\.(\w+)", bench.read_text()))
            source = reference.read_bytes() + STALE_WRAPPER.encode() + f"\n{fn} = _stale_wrapper({fn})\n".encode()
            control = "cached_first_output_per_shape_not_input_values"
        else:
            source = (bundle / task.student_file).read_bytes()
            # Get past the benchmark's nonzero-workspace ABI guard so this
            # control reaches output verification, while doing no computation.
            if source.count(b"{ return 0; }") != 1:
                raise ValueError(f"{task.task_id}: unexpected no-op workspace anchor")
            source = source.replace(b"{ return 0; }", b"{ return 256; }")
            control = "no_op_with_nonzero_workspace"
        trace = types.SimpleNamespace(info={})
        record = {
            "task": task.task_id,
            "control": control,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "image": images[task.runtime_kind],
            "image_id": subprocess.check_output(
                ["docker", "image", "inspect", "--format", "{{.Id}}", images[task.runtime_kind]], text=True
            ).strip(),
        }
        runtime = make_runtime(
            scoring.score_runtime_config(config, images[task.runtime_kind]),
            name=f"pmpp-negative-{uuid.uuid4().hex[:10]}",
        )
        try:
            await runtime.start()
            await runtime.write(f"/app/{task.student_file}", source)
            await provision_grader(config, task, Workspace(runtime), bundle, tree)
            accepted = await scoring.perf_paired(config, runtime, task, trace, tree)
            record["accepted"] = accepted
            # A control that merely fails to compile is not a successful digest test.
            record["rejected_by_digest"] = not accepted and trace.info["perf_ratio"].get("out_fnv_mismatch") is True
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            await runtime.stop()
        record["info"] = trace.info
        results.append(record)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "info"}), flush=True)
    if len(results) != len(tasks) or not all(r.get("rejected_by_digest") for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--cuda-image", default="pmpp-cuda-agent:cu128")
    parser.add_argument("--cutlass-image", default="pmpp-cutlass:12.8.1")
    parser.add_argument("--triton-image", default="pmpp-triton:cu128")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    asyncio.run(validate(parser.parse_args()))
