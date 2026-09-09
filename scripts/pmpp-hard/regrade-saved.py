"""Diagnose saved submissions with fresh local grading, without rewriting old scores.

Only the fully captured submitted source is read from each selected historical
trace. New inputs, policies, metrics and logs belong to a separate diagnostic
artifact; this is not a new model evaluation or a leaderboard result.
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import pathlib
import subprocess
import types
from unittest.mock import patch

from pmpp_hard import pool, scoring
from pmpp_hard.config import PMPPHardConfig
from pmpp_hard.paths import DataTree
from pmpp_hard.preflight import PreflightContext, run_preflight


class Submission:
    def __init__(self, filename, kernel):
        self.path = f"/app/{filename}"
        self.kernel = kernel

    async def read(self, path):
        if path != self.path:
            raise ValueError(f"unexpected submission path: {path}")
        return self.kernel


async def regrade(args):
    tree = DataTree()
    config = PMPPHardConfig(
        include_ids=args.ids,
        score_image=args.cuda_image,
        score_images={"cuda": args.cuda_image, "triton": args.triton_image, "cutlass": args.cutlass_image},
        residency_time_mode=args.residency_time_mode,
    )
    tasks, stats = pool.build_tasks(config, tree)
    if stats.unmatched_include_ids:
        raise ValueError(f"unknown task IDs: {stats.unmatched_include_ids}")
    report = run_preflight(
        PreflightContext(tree, config, tasks, stats, pool.pool_content_hash(tree, tasks, stats.rows))
    )
    report.raise_if_failed()
    selected = {t.task_id: t for t in tasks}
    saved = []
    for path in args.traces:
        with path.open() as source:
            for line_no, line in enumerate(source, 1):
                row = json.loads(line)
                task_id = row["task"]["data"]["task_id"]
                if task_id not in selected:
                    continue
                kernel = row["info"].get("submitted_kernel")
                if not isinstance(kernel, str) or not kernel.strip():
                    raise ValueError(f"{path}:{line_no}: no complete captured submission")
                saved.append((path.resolve(), line_no, row["id"], selected[task_id], kernel.encode()))
    missing = selected.keys() - {entry[3].task_id for entry in saved}
    if missing:
        raise ValueError(f"no saved submission for: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output}")
    provenance = {
        "kind": "fresh_saved_submission_diagnostic_not_model_eval",
        "benchmark_seed_override": f"{args.benchmark_seed:016x}" if args.benchmark_seed is not None else None,
        "config": config.model_dump(mode="json"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": subprocess.check_output(["nvidia-smi"], text=True),
        "levers": report.lever.as_json(),
        "package_sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tree.root.parent.glob("*.py")},
        "images": {
            kind: subprocess.check_output(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True
            ).strip()
            for kind, image in config.score_images.items()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    results = []
    for path, line_no, trace_id, task, kernel in saved:
        for rollout in range(args.rollouts):
            trace = types.SimpleNamespace(info={})
            record = {
                "source_trace_file": str(path),
                "source_line": line_no,
                "source_trace_id": trace_id,
                "kernel_sha256": hashlib.sha256(kernel).hexdigest(),
                "task": task.task_id,
                "rollout": rollout,
            }
            try:
                seed_context = contextlib.nullcontext()
                if args.benchmark_seed is not None:
                    seed_context = patch("pmpp_hard.benchmarks.draw_seed", return_value=args.benchmark_seed)
                    trace.info["diagnostic_benchmark_seed_override"] = f"{args.benchmark_seed:016x}"
                with seed_context:
                    record["metrics"] = await scoring.score_clean(
                        config, task, trace, Submission(task.student_file, kernel), tree
                    )
            except Exception as error:  # Keep other selected diagnostics runnable, never turn errors into zeros.
                record["error"] = f"{type(error).__name__}: {error}"
            record["info"] = trace.info
            results.append(record)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k != "info"}), flush=True)
    if len(results) != len(saved) * args.rollouts:
        raise RuntimeError("incomplete diagnostic coverage")
    if any("error" in r for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument(
        "--benchmark-seed",
        type=lambda s: int(s, 0),
        help="fixed uint64 seed for explicitly labeled timing diagnostics only",
    )
    parser.add_argument("--residency-time-mode", choices=("diagnostic", "enforce"), default="diagnostic")
    parser.add_argument("--cuda-image", default="pmpp-cuda-agent:cu128")
    parser.add_argument("--cutlass-image", default="pmpp-cutlass:12.8.1")
    parser.add_argument("--triton-image", default="pmpp-triton:cu128")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.rollouts < 1:
        parser.error("--rollouts must be positive")
    if args.benchmark_seed is not None and not 0 <= args.benchmark_seed < 1 << 64:
        parser.error("--benchmark-seed must be an unsigned 64-bit integer")
    asyncio.run(regrade(args))
