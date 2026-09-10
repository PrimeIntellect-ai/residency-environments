"""Validate all active performance input generators on local GPU Docker, without model calls.

References run on A/A/B seeds; A defaults to UINT64_MAX to exercise Torch offset
overflow. Triton input fingerprints instrument random tensor creation, outside any
calibration claim. CUDA/CUTLASS output digests provide a weaker variation check.
An optional paired pass uses the actual scorer with five pairs and a fresh seed.
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

from pmpp_hard import benchmarks, pool, scoring
from pmpp_hard.config import PMPPHardConfig
from pmpp_hard.paths import DataTree, Workspace
from pmpp_hard.provision import grader_files, provision_grader
from verifiers.v1.runtimes import make_runtime

# Run inside the Triton image. Sample generated input bytes, not just the seed or
# the benchmark digest (some tolerance graders hash only oracle agreement flags).
INPUT_PROBE = """import functools, hashlib, runpy, sys
import torch
h = hashlib.sha256()
oracle_checks = oracle_failures = 0
original_allclose = torch.allclose
def checked_allclose(*args, **kwargs):
    global oracle_checks, oracle_failures
    result = original_allclose(*args, **kwargs)
    oracle_checks += 1
    oracle_failures += not result
    return result
torch.allclose = checked_allclose
def wrap(fn):
    @functools.wraps(fn)
    def observed(*args, **kwargs):
        out = fn(*args, **kwargs)
        h.update(str((out.shape, out.dtype)).encode())
        h.update(out.reshape(-1)[:128].contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return out
    return observed
for name in ("rand", "randn", "randint"):
    setattr(torch, name, wrap(getattr(torch, name)))
path = sys.argv[1]
sys.argv = [path]
module = runpy.run_path(path)
rc = module["main"]()
print("input_sha256=" + h.hexdigest())
print(f"oracle_checks={oracle_checks} oracle_failures={oracle_failures}")
raise SystemExit(rc)
"""


def capture(argv):
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    return result.stdout.strip()


async def validate(args):
    tree = DataTree()
    config = PMPPHardConfig(preflight=False, include_ids=args.ids or None)
    tasks, stats = pool.build_tasks(config, tree)
    if stats.unmatched_include_ids or (args.ids and any(not t.perf_gated for t in tasks)):
        raise ValueError("selection must contain only known performance-gated tasks")
    tasks = [t for t in tasks if t.perf_gated]
    if not tasks:
        raise ValueError("no performance tasks selected")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output}")
    seed_b = benchmarks.draw_seed()
    while seed_b == args.seed:
        seed_b = benchmarks.draw_seed()
    images = {"cuda": args.cuda_image, "cutlass": args.cutlass_image, "triton": args.triton_image}
    manifest = {
        "policy": benchmarks.POLICY,
        "seed_a": f"{args.seed:016x}",
        "seed_b": f"{seed_b:016x}",
        "git_head": capture(["git", "rev-parse", "HEAD"]),
        "worktree_diff_sha256": hashlib.sha256(capture(["git", "diff"]).encode()).hexdigest(),
        "package_sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tree.root.parent.glob("*.py")},
        "gpu": capture(["nvidia-smi"]),
        "images": {k: capture(["docker", "image", "inspect", "--format", "{{.Id}}", v]) for k, v in images.items()},
        "expected_tasks": [t.task_id for t in tasks],
        "paired": args.paired,
        "timing_note": "diagnostic only; instrumented input probes and host display activity are not idle-GPU calibration",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for task in tasks:
        image = images[task.runtime_kind]
        runtime = make_runtime(
            scoring.score_runtime_config(config, image), name=f"pmpp-bench-check-{uuid.uuid4().hex[:10]}"
        )
        record = {"task": task.task_id, "checks": {}, "runs": []}
        try:
            await runtime.start()
            bundle = tree.bundle(task.task_id)
            (bench,) = [p for p in bundle.glob("bench_*") if p.suffix in (".cu", ".py")]
            source = bench.read_bytes()
            if task.runtime_kind == "triton":
                if 'os.environ.get("PMPP_BENCH_SEED"' not in source.decode():
                    raise ValueError(f"{task.task_id}: missing seed consumer")
                await scoring.ensure_python_env(runtime, "benchmark validation")
            elif (
                b"pmpp::bench_seed(" not in source
                or b'getenv("PMPP_BENCH_SEED")' not in (bundle / "pmpp_bench_digest.cuh").read_bytes()
            ):
                raise ValueError(f"{task.task_id}: missing seed consumer")
            reference = tree.find_reference(task.task_id)
            if reference is None:
                raise ValueError(f"{task.task_id}: missing reference")
            await runtime.write("/app/Makefile", (bundle / "Makefile").read_bytes())
            for path in grader_files(bundle):
                data = benchmarks.prepare_source(task.task_id, source) if path == bench else path.read_bytes()
                await runtime.write(f"/app/{path.name}", data)
            await runtime.write(f"/app/{task.student_file}", reference.read_bytes())
            built = await runtime.run(
                [
                    "bash",
                    "-lc",
                    f"cd /app && timeout -k 5 900 make {task.bench_target} PY=python3 NVCCFLAGS='-O3 -std=c++17 -arch={config.arch}'",
                ],
                {},
            )
            (args.output / f"{task.task_id}-build.log").write_text(built.stdout + built.stderr)
            if built.exit_code != 0:
                raise RuntimeError(f"reference build exited {built.exit_code}")
            executable = f"./{task.bench_target}"
            if task.runtime_kind == "triton":
                await runtime.write("/app/input_probe.py", INPUT_PROBE.encode())
                executable = f"python3 input_probe.py {bench.name}"
            for label, seed in (("a", args.seed), ("repeat_a", args.seed), ("b", seed_b)):
                result = await runtime.run(
                    ["bash", "-lc", f"cd /app && PMPP_BENCH_SEED={seed} timeout -k 5 420 {executable}"], {}
                )
                output = result.stdout + result.stderr
                measured = benchmarks.measurement(result.exit_code, output)
                fingerprints = re.findall(r"input_sha256=([0-9a-f]{64})", output)
                measured.update(label=label, seed=f"{seed:016x}")
                if fingerprints:
                    measured["input_sha256"] = fingerprints[-1]
                if task.runtime_kind == "triton":
                    checks = re.findall(r"oracle_checks=(\d+) oracle_failures=(\d+)", output)
                    measured["oracle_checks"] = int(checks[-1][0]) if checks else 0
                    measured["oracle_failures"] = int(checks[-1][1]) if checks else None
                record["runs"].append(measured)
                (args.output / f"{task.task_id}-{label}.log").write_text(output)
            a, repeat, b = record["runs"]
            valid = all("error" not in r for r in record["runs"])
            record["checks"]["measurements_valid"] = valid
            if valid:
                key = "input_sha256" if task.runtime_kind == "triton" else "out_fnv"
                record["checks"].update(
                    same_seed_stable=a.get(key) is not None and a[key] == repeat.get(key),
                    different_seed_changes=a.get(key) is not None and a[key] != b.get(key),
                )
                record["variation_evidence"] = key
                if task.runtime_kind == "triton":
                    record["checks"]["reference_matches_oracle"] = all(
                        r["oracle_checks"] > 0 and r["oracle_failures"] == 0 for r in record["runs"]
                    )
            if args.paired:
                await provision_grader(config, task, Workspace(runtime), bundle, tree)
                trace = types.SimpleNamespace(info={})
                record["paired_pass"] = await scoring.perf_paired(config, runtime, task, trace, tree)
                record["paired_trace"] = trace.info
                record["checks"]["complete_matching_pairs"] = (
                    trace.info["perf_ratio"]["out_fnv_verified"]
                    and trace.info["perf_ratio"]["pairs"] == config.perf_ratio_runs
                )
        except Exception as error:  # Continue the matrix, preserving failed task evidence.
            record["error"] = f"{type(error).__name__}: {error}"
        finally:
            await runtime.stop()
        record["passed"] = bool(record["checks"]) and all(record["checks"].values()) and "error" not in record
        results.append(record)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k not in {"runs", "paired_trace"}}), flush=True)
    complete = len(results) == len(tasks) and {r["task"] for r in results} == {t.task_id for t in tasks}
    summary = {"complete": complete, "passed": sum(r["passed"] for r in results), "total": len(tasks)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not complete or not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=(1 << 64) - 1)
    parser.add_argument("--paired", action="store_true", help="also run the real five-pair scorer on each reference")
    parser.add_argument("--cuda-image", default="pmpp-cuda-agent:cu128")
    parser.add_argument("--cutlass-image", default="pmpp-cutlass:12.8.1")
    parser.add_argument("--triton-image", default="pmpp-triton:cu128")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.seed < 1 << 64:
        parser.error("--seed must be an unsigned 64-bit integer")
    asyncio.run(validate(args))
