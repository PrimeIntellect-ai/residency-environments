"""Score kernels in isolated sandboxes with paired performance comparisons.

Correctness is verified before performance. Performance-gated tasks compare
interleaved student and reference runs in the same runtime. Infrastructure
errors are classified separately from solution failures, and optional source,
residency, and KernelGuard checks provide additional validation.
"""

import asyncio
import contextlib
import os
import re
import shutil
import statistics
import subprocess
import uuid

from pmpp_hard import grader_inputs, kernelguard_gate, markers, residency
from pmpp_hard.config import DEFAULT_SCORE_IMAGES, PMPPHardConfig, PMPPHardTaskData
from pmpp_hard.errors import ScoreInfraError
from pmpp_hard.paths import DataTree, Workspace
from pmpp_hard.provision import perf_enabled, provision_grader

# Serialize GPU grading so concurrent rollouts do not interfere with timing.
_GPU_LOCK = asyncio.Lock()
_MODAL_IMAGES_READY: set[str] = set()
_MODAL_IMAGE_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}
_MODAL_APP_NAME = "verifiers-v1"

_ZEROS = {"build_pass": 0.0, "run_pass": 0.0, "performance_pass": 0.0}
# A successful process status does not establish that every test case ran.
# Require the test suite's completion summary before accepting correctness.
_PASSED_RE = re.compile(r"passed\s+(\d+)\s*/\s*(\d+)")


async def confirm_correctness(runtime, task: PMPPHardTaskData, trace) -> bool:
    """Return whether the sandbox log confirms that all test cases passed."""
    try:
        res = await runtime.run(["bash", "-lc", "cat /tmp/test.log 2>/dev/null"], {})
        log = res.stdout + res.stderr
    except Exception as e:  # noqa: BLE001 - an unreadable log cannot confirm correctness
        log = ""
        _pmpp_info(trace)["correctness_confirm_error"] = f"{type(e).__name__}"
    m = _PASSED_RE.findall(log)
    if m:
        p, t = int(m[-1][0]), int(m[-1][1])
        if t > 0 and p == t:
            _pmpp_info(trace)["correctness_confirmed"] = f"passed {p} / {t}"
            return True
        _pmpp_info(trace)["correctness_forged_exit"] = f"test reported passed {p} / {t}"
        return False
    _pmpp_info(trace)["correctness_forged_exit"] = (
        "no 'passed N / N' completion line in test log"
    )
    return False


# The optional circuit breaker counts consecutive scoring-infrastructure
# failures. Solution failures leave the count unchanged.
_consecutive_infra_failures = 0


def _pmpp_info(trace) -> dict:
    """Return the PMPP-specific trace namespace."""
    return trace.info.setdefault("pmpp", {})


def parse_markers(
    config: PMPPHardConfig, out: str, task: PMPPHardTaskData, trace
) -> dict:
    """Parse grader markers into metrics for each grading stage."""
    build = markers.M_BUILD_PASS in out
    run = markers.M_RUN_PASS in out
    if perf_enabled(config, task):
        perf_pass = markers.M_PERF_PASS in out
    else:
        perf_pass = run
    trace.info["verify_markers"] = out.strip()[-600:]
    return {
        "build_pass": float(build),
        "run_pass": float(run),
        "performance_pass": float(perf_pass),
    }


def runtime_kind_of(task: PMPPHardTaskData) -> str:
    """Return the runtime kind required by the task manifest."""
    return task.runtime_kind


def resolve_score_image(config: PMPPHardConfig, kind: str) -> str:
    """Resolve the scoring image for a runtime kind."""
    imgs = config.score_images or {}
    if kind not in imgs:
        return config.score_image
    authored = set(getattr(config, "model_fields_set", ()))
    if (
        imgs == DEFAULT_SCORE_IMAGES
        and "score_image" in authored
        and "score_images" not in authored
    ):
        return config.score_image
    return imgs[kind]


_DOCKER_HUB_OFFICIAL = frozenset(
    {
        "alpine",
        "busybox",
        "debian",
        "golang",
        "node",
        "openjdk",
        "python",
        "ruby",
        "rust",
        "ubuntu",
    }
)


def score_image_is_pullable(image: str) -> bool:
    """Return whether Docker may resolve the image from a registry.

    Names with a registry or namespace path are pullable. Unqualified names are
    ambiguous, so only common Docker Official Images are accepted as shorthand;
    project and operator tags remain local-only unless explicitly qualified.
    """
    repository = image.split("@", 1)[0]
    if ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    return "/" in repository or repository in _DOCKER_HUB_OFFICIAL


def modal_image_is_registry_addressable(image: str) -> bool:
    """Return whether Modal can resolve an image through a remote registry."""
    repository = image.split("@", 1)[0]
    host = repository.split("/", 1)[0].split(":", 1)[0].lower()
    return host not in {"localhost", "127.0.0.1", "::1"} and score_image_is_pullable(
        image
    )


def modal_prerequisite_error() -> str | None:
    """Return a user-facing Modal setup error without exposing credentials."""
    try:
        import modal
    except ModuleNotFoundError:
        return (
            "Modal scoring requires the optional dependency; install pmpp-hard[modal]"
        )
    try:
        modal_config = modal.config.Config()
        token_id = modal_config.get("token_id")
        token_secret = modal_config.get("token_secret")
    except Exception as error:  # noqa: BLE001 - provider config errors need context
        return f"Modal configuration could not be read ({type(error).__name__})"
    if not token_id or not token_secret:
        return (
            "Modal credentials are missing; configure a Modal profile or set "
            "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET"
        )
    return None


def modal_gpu_arch_error(arch: str, gpu: str | None) -> str | None:
    """Reject Modal GPU selections that cannot execute the configured CUDA arch."""
    gpu_type = (gpu or "").split(":", 1)[0].upper()
    if arch == "sm_120" and gpu_type != "RTX-PRO-6000":
        return (
            f"arch={arch} requires Modal gpu=RTX-PRO-6000, got "
            f"{gpu or 'no GPU request'}"
        )
    return None


async def ensure_modal_registry_image(image: str) -> None:
    """Resolve and build-cache a public registry image in the Verifiers Modal app."""
    if error := modal_prerequisite_error():
        raise ScoreInfraError(error)
    if not modal_image_is_registry_addressable(image):
        raise ScoreInfraError(
            f"Modal image {image} is local-only; publish it to a public "
            "registry and configure the runtime with that reference"
        )
    if image in _MODAL_IMAGES_READY:
        return
    loop_key = (id(asyncio.get_running_loop()), image)
    lock = _MODAL_IMAGE_LOCKS.setdefault(loop_key, asyncio.Lock())
    async with lock:
        if image in _MODAL_IMAGES_READY:
            return
        try:
            import modal

            app = await modal.App.lookup.aio(_MODAL_APP_NAME, create_if_missing=True)
            modal_image = modal.Image.from_registry(image).entrypoint([])
            await modal_image.build.aio(app)
        except Exception as error:  # noqa: BLE001 - normalize provider failures
            raise ScoreInfraError(
                f"Modal could not resolve image {image}: "
                f"{type(error).__name__}: {str(error)[:180]}"
            ) from error
        _MODAL_IMAGES_READY.add(image)


def score_runtime_config(config: PMPPHardConfig, image: str | None = None):
    """Build the selected isolated scoring-sandbox config."""
    image = image or config.score_image
    if config.score_runtime_type == "modal":
        if error := modal_gpu_arch_error(config.arch, config.score_modal_gpu):
            raise ValueError(error)
        from verifiers.v1.runtimes.modal import ModalConfig

        return ModalConfig(
            image=image,
            gpu=config.score_modal_gpu,
            cpu=config.score_cpu,
            memory=config.score_memory,
            region=config.score_modal_region,
            network_access=config.score_modal_network_access,
            creates_per_sec=config.score_modal_creates_per_sec,
        )
    from verifiers.v1.runtimes.docker import DockerConfig

    return DockerConfig(
        image=image,
        gpu=config.score_gpu,
        cpu=config.score_cpu,
        memory=config.score_memory,
        allow=[],
        block=["*"],
    )


async def ensure_score_image_ready(
    config: PMPPHardConfig, task: PMPPHardTaskData
) -> None:
    """Check scoring prerequisites before model generation without pulling images."""
    image = resolve_score_image(config, runtime_kind_of(task))
    if config.score_runtime_type == "modal":
        if error := modal_prerequisite_error():
            raise ScoreInfraError(error)
        if error := modal_gpu_arch_error(config.arch, config.score_modal_gpu):
            raise ScoreInfraError(error)
        if not modal_image_is_registry_addressable(image):
            raise ScoreInfraError(
                f"Modal image {image} is local-only; publish it to a public "
                "registry and configure the runtime with that reference"
            )
        return
    docker = shutil.which("docker")
    if docker is None:
        raise ScoreInfraError(
            "Docker is required for PMPP release scoring but is not on PATH"
        )
    daemon = await asyncio.to_thread(
        subprocess.run, [docker, "info"], capture_output=True, timeout=10
    )
    if daemon.returncode != 0:
        raise ScoreInfraError(
            "Docker daemon is unavailable; refusing to start a paid rollout"
        )
    result = await asyncio.to_thread(
        subprocess.run,
        [docker, "image", "inspect", image],
        capture_output=True,
        timeout=10,
    )
    if result.returncode == 0:
        return
    if not score_image_is_pullable(image):
        raise ScoreInfraError(
            f"required local scoring image {image} is missing; run scripts/pmpp-hard/build-images.sh"
        )
    manifest = await asyncio.to_thread(
        subprocess.run,
        [docker, "manifest", "inspect", image],
        capture_output=True,
        timeout=60,
    )
    if manifest.returncode != 0:
        raise ScoreInfraError(
            f"required scoring image {image} is neither local nor available from its registry"
        )


async def ensure_score_image_available(
    config: PMPPHardConfig, task: PMPPHardTaskData
) -> None:
    """Prepare the scoring image inside the scoring phase."""
    image = resolve_score_image(config, runtime_kind_of(task))
    if config.score_runtime_type == "modal":
        await ensure_modal_registry_image(image)
        return
    docker = shutil.which("docker")
    if docker is None:
        raise ScoreInfraError(
            "Docker is required for PMPP release scoring but is not on PATH"
        )
    daemon = await asyncio.to_thread(
        subprocess.run, [docker, "info"], capture_output=True, timeout=10
    )
    if daemon.returncode != 0:
        raise ScoreInfraError(
            "Docker daemon is unavailable; refusing to start a paid rollout"
        )
    result = await asyncio.to_thread(
        subprocess.run,
        [docker, "image", "inspect", image],
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        if not score_image_is_pullable(image):
            raise ScoreInfraError(
                f"required local scoring image {image} is missing; run scripts/pmpp-hard/build-images.sh"
            )
        pull = await asyncio.to_thread(
            subprocess.run, [docker, "pull", image], capture_output=True, timeout=900
        )
        if pull.returncode != 0:
            raise ScoreInfraError(
                f"required scoring image {image} is not local and Docker could not pull it"
            )
        result = await asyncio.to_thread(
            subprocess.run,
            [docker, "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise ScoreInfraError(
                f"Docker pulled {image} but it is still not inspectable"
            )


async def ensure_python_env(runtime, where: str) -> None:
    """Ensure the scoring runtime can import the required Python packages."""
    probe = "python3 -c 'import torch, triton' 2>&1"
    res = await runtime.run(["bash", "-lc", probe], {})
    if res.exit_code == 0:
        return
    tail = (res.stdout + res.stderr).strip()[-300:]
    raise ScoreInfraError(
        f"{where}: image cannot import torch/triton; "
        f"rebuild the Triton image with scripts/pmpp-hard/build-images.sh: {tail}"
    )


async def run_grader(
    config: PMPPHardConfig, runtime, task: PMPPHardTaskData, env: dict | None = None
) -> str:
    perf = "1" if perf_enabled(config, task) else "0"
    res = await runtime.run(
        ["bash", "-lc", f"PMPP_PERF={perf} bash /app/verify_student"], env or {}
    )
    return res.stdout + res.stderr


async def perf_paired(
    config: PMPPHardConfig,
    runtime,
    task: PMPPHardTaskData,
    trace,
    tree: DataTree,
    env: dict | None = None,
) -> bool:
    """Compare interleaved student and reference timings in one runtime.

    Student and reference builds remain separate so reference failures are
    reported as infrastructure errors. Per-run samples are retained for later
    performance-gate calibration.
    """
    ref = tree.find_reference(task.task_id)
    if ref is None:
        raise ScoreInfraError(
            f"release reference missing for perf-gated task {task.task_id}"
        )
    await runtime.write(f"/app/.grader/{ref.name}", ref.read_bytes())
    sf, runs = task.student_file, config.perf_ratio_runs
    # Use randomized executable names and a fresh seed for each comparison.
    # Output checksums bind accepted timings to matching computed results.
    seed = int.from_bytes(os.urandom(4), "big")
    tag = uuid.uuid4().hex[:10]
    switch_student = switch_reference = ""
    student_env = reference_env = ""
    if task.student_file.endswith(".py"):
        # Python benchmark launchers import solution.py at process start. Switch the
        # implementation before each process so student and reference measurements
        # remain distinct; copying the launcher alone would make both import whichever
        # source was installed last.
        switch_student = f"cp -f /app/{sf} ./{sf}; "
        switch_reference = f"cp -f ./{ref.name} ./{sf}; "
        student_env = "PYTHONPYCACHEPREFIX=/tmp/pmpp-pycache-student "
        reference_env = "PYTHONPYCACHEPREFIX=/tmp/pmpp-pycache-reference "
    script = (
        f'cd /app/.grader && cp -f /app/{sf} ./{sf} && NV="$(cat .nvccflags)" && '
        f'timeout -k 5 900 make {task.bench_target} PY=python3 NVCCFLAGS="$NV" >/tmp/pb.log 2>&1 '
        f"|| {{ echo PB_STU_BUILD_FAIL; tail -5 /tmp/pb.log; exit 0; }}; "
        f'timeout -k 5 900 make bench_reference PY=python3 NVCCFLAGS="$NV" >>/tmp/pb.log 2>&1 '
        f"|| {{ echo PB_REF_BUILD_FAIL; tail -5 /tmp/pb.log; exit 0; }}; "
        f"cp -f ./{task.bench_target} ./s_{tag} && cp -f ./bench_reference ./r_{tag}; "
        f"for i in $(seq 1 {runs}); do "
        f"{switch_student}"
        f"so=$({student_env}PMPP_BENCH_SEED={seed} timeout -k 5 420 ./s_{tag} 2>&1); "
        f'echo "STU $(echo "$so" | grep -oE \'avg_ms=[0-9.]+\' | head -1)"; '
        f'echo "STUF $(echo "$so" | grep -oE \'out_fnv=[0-9a-fA-Fx]+\' | head -1)"; '
        f"{switch_reference}"
        f"ro=$({reference_env}PMPP_BENCH_SEED={seed} timeout -k 5 420 ./r_{tag} 2>&1); "
        f'echo "REF $(echo "$ro" | grep -oE \'avg_ms=[0-9.]+\' | head -1)"; '
        f'echo "REFF $(echo "$ro" | grep -oE \'out_fnv=[0-9a-fA-Fx]+\' | head -1)"; done'
    )
    res = await runtime.run(["bash", "-lc", script], env or {})
    out = res.stdout + res.stderr
    if "PB_REF_BUILD_FAIL" in out:
        raise ScoreInfraError(
            f"bench_reference failed to BUILD in the scoring sandbox — "
            f"env fault, not a student perf fail: {out.strip()[-300:]}"
        )
    stu = [float(x) for x in re.findall(r"STU avg_ms=([0-9.]+)", out)]
    rfs = [float(x) for x in re.findall(r"REF avg_ms=([0-9.]+)", out)]
    stu_fnv = re.findall(r"STUF out_fnv=([0-9a-fA-Fx]+)", out)
    ref_fnv = re.findall(r"REFF out_fnv=([0-9a-fA-Fx]+)", out)
    n = min(len(stu), len(rfs))
    if n == 0:
        if stu and not rfs:
            raise ScoreInfraError(
                f"bench_reference produced no timing in the scoring sandbox "
                f"— env fault, not a student perf fail: {out.strip()[-300:]}"
            )
        trace.info["perf_ratio"] = f"paired bench failed; {out.strip()[-300:]}"
        return False
    # Require matching checksums from the student and reference timing runs.
    # A missing reference checksum indicates an incomplete benchmark setup.
    ref_has_fnv = len(ref_fnv) >= n
    out_fnv_verified = None
    if ref_has_fnv:
        out_fnv_verified = len(stu_fnv) >= n and all(
            s.lower() == r.lower()
            for s, r in zip(stu_fnv[:n], ref_fnv[:n], strict=True)
        )
    else:
        raise ScoreInfraError(
            f"bench_reference emits no out_fnv — release69 requires every "
            f"perf bench to be output-verified: {task.task_id}"
        )
    if out_fnv_verified is False:
        trace.info["perf_ratio"] = {
            "out_fnv_mismatch": True,
            "student_fnv": stu_fnv[:n],
            "reference_fnv": ref_fnv[:n],
            "note": "timed kernel did not reproduce the graded outputs — no-op/shape-detect hack",
        }
        return False
    ratios = [s / r for s, r in zip(stu[:n], rfs[:n], strict=True)]
    med = statistics.median(sorted(ratios))
    trace.info["perf_ratio"] = {
        "median_ratio": round(med, 4),
        "target": task.perf_ratio_target or config.perf_ratio_target,
        "student_ms": round(statistics.median(stu[:n]), 4),
        "reference_ms": round(statistics.median(rfs[:n]), 4),
        "pairs": n,
        # Retain the samples behind the median for performance-gate calibration.
        "ratios": [round(x, 4) for x in ratios],
        "student_ms_runs": [round(x, 4) for x in stu[:n]],
        "reference_ms_runs": [round(x, 4) for x in rfs[:n]],
        "out_fnv_verified": out_fnv_verified,  # True when checksums were present and matched.
    }
    return med <= (task.perf_ratio_target or config.perf_ratio_target)


async def _capture_grader_logs(runtime, trace) -> None:
    """Store grader log tails in private trace metadata for diagnostics."""
    try:
        res = await runtime.run(
            [
                "bash",
                "-lc",
                "for f in /tmp/build.log /tmp/test.log /tmp/bbuild.log /tmp/bench.log /tmp/pb.log; do "
                '[ -f $f ] && { echo "== $f"; tail -c 1500 $f; }; done',
            ],
            {},
        )
        tails = (res.stdout + res.stderr).strip()
        if tails:
            _pmpp_info(trace)["grader_logs"] = tails[-8000:]
    except Exception:  # noqa: BLE001 - diagnostics must not fail scoring
        pass


async def score_clean(
    config: PMPPHardConfig, task: PMPPHardTaskData, trace, agent_runtime, tree: DataTree
) -> dict:
    """Grade a captured kernel in a newly provisioned scoring sandbox."""
    global _consecutive_infra_failures
    if config.score_breaker > 0 and _consecutive_infra_failures >= config.score_breaker:
        raise ScoreInfraError(
            f"scoring circuit breaker tripped: {_consecutive_infra_failures} consecutive "
            f"infra failures — scoring backend is down, refusing to emit more silent zeros"
        )
    try:
        kernel = await agent_runtime.read(f"/app/{task.student_file}")
    except Exception as e:  # noqa: BLE001
        trace.info["score_error"] = f"could not read kernel: {type(e).__name__}"
        _pmpp_info(trace)["score_error_kind"] = "read_kernel"
        return dict(_ZEROS)
    # Apply source-policy checks before provisioning a scoring sandbox. Policy
    # failures remain distinct from correctness failures in the trace.
    if config.source_policy:
        violation = residency.static_policy_violation(task, kernel)
        if (
            violation is None
            and config.triton_premise
            and task.student_file.endswith(".py")
        ):
            violation = residency.triton_static_violation(kernel)
        if violation is not None:
            _pmpp_info(trace)["policy_fail"] = violation
            trace.info["verify_markers"] = f"{markers.M_POLICY_FAIL} ({violation})"
            return dict(_ZEROS)
    await ensure_score_image_available(config, task)
    image = resolve_score_image(config, runtime_kind_of(task))
    _pmpp_info(trace)["score_image"] = image
    _pmpp_info(trace)["score_runtime_type"] = config.score_runtime_type
    cfg = score_runtime_config(config, image)
    lock = _GPU_LOCK if config.gpu_lock else contextlib.nullcontext()
    async with lock:
        return await grade_in_fresh_runtime(config, cfg, kernel, task, trace, tree)


async def grade_in_fresh_runtime(
    config: PMPPHardConfig,
    cfg,
    kernel: bytes,
    task: PMPPHardTaskData,
    trace,
    tree: DataTree,
) -> dict:
    global _consecutive_infra_failures
    from verifiers.v1.runtimes import make_runtime

    score_rt = None
    last_exc = None
    # Include a rollout tag to avoid collisions between concurrent task sandboxes.
    rollout_tag = uuid.uuid4().hex[:8]
    for attempt in range(
        4
    ):  # Scoring-sandbox provisioning may fail transiently under concurrent load.
        score_rt = make_runtime(
            cfg, name=f"pmpp-score-{task.task_id}-{rollout_tag}-{attempt}"
        )
        try:
            await score_rt.start()
            last_exc = None
            break
        except Exception as e:  # noqa: BLE001
            last_exc = e
            await score_rt.stop()
            await asyncio.sleep(5 * (attempt + 1))
    if last_exc is not None:
        # Raise provisioning failures so the evaluation framework records an
        # infrastructure error instead of a solution failure.
        trace.info["score_error"] = (
            f"provisioning failed after retries: "
            f"{type(last_exc).__name__}: {str(last_exc)[:160]}"
        )
        _pmpp_info(trace)["score_error_kind"] = "provisioning"
        _consecutive_infra_failures += 1
        raise ScoreInfraError(
            f"scoring sandbox provisioning failed after 4 attempts for {task.task_id}: "
            f"{type(last_exc).__name__}: {str(last_exc)[:160]}"
        ) from last_exc
    try:
        correctness_parameter = grader_inputs.draw_parameter(task.task_id)
        _pmpp_info(trace)["correctness_inputs"] = {
            "policy": grader_inputs.POLICY,
            "mode": "randomized" if correctness_parameter is not None else "fixed",
            "parameter": (
                f"{correctness_parameter:016x}"
                if correctness_parameter is not None
                else None
            ),
        }
        await score_rt.write(f"/app/{task.student_file}", kernel)
        ws = Workspace(score_rt)
        await provision_grader(
            config,
            task,
            ws,
            tree.bundle(task.task_id),
            tree,
            correctness_parameter=correctness_parameter,
        )  # Provision the configured grader.
        if task.student_file.endswith(".py"):
            await ensure_python_env(
                score_rt, "scoring sandbox"
            )  # Validate Python dependencies before grading.
        trace.info["scored_in"] = "clean_sandbox"
        paired = task.perf_gated
        if paired:
            triton_probe = config.triton_premise and task.student_file.endswith(".py")
            run_env = await residency.install(score_rt) if triton_probe else {}
            # Verify correctness before measuring the paired performance ratio.
            out = await score_rt.run(
                ["bash", "-lc", "PMPP_PERF=0 bash /app/verify_student"], run_env
            )
            res = parse_markers(config, out.stdout + out.stderr, task, trace)
            if res["run_pass"] and not await confirm_correctness(score_rt, task, trace):
                res["run_pass"] = (
                    0.0  # The test suite did not confirm that every case completed successfully.
                )
            if triton_probe and res["run_pass"]:
                if not await residency.enforce(
                    config, task, score_rt, trace, tree, kernel
                ):
                    res["run_pass"] = 0.0
            if res["build_pass"] and res["run_pass"]:
                res["performance_pass"] = float(
                    await perf_paired(config, score_rt, task, trace, tree)
                )
            else:
                res["performance_pass"] = 0.0
            # Analyze the captured kernel and paired timings with KernelGuard.
            pr = trace.info.get("perf_ratio")
            kg_v = kernelguard_gate.check(
                config,
                task,
                kernel,
                trace,
                student_ms=pr.get("student_ms") if isinstance(pr, dict) else None,
                reference_ms=pr.get("reference_ms") if isinstance(pr, dict) else None,
                out_fnv_verified=pr.get("out_fnv_verified")
                if isinstance(pr, dict)
                else None,
            )
            res = kernelguard_gate.enforce(kg_v, res, trace)
            await _capture_grader_logs(score_rt, trace)
            _consecutive_infra_failures = (
                0  # A completed grade confirms backend health and resets the breaker.
            )
            return res
        # For tasks without a performance gate, optionally verify that the
        # correctness run executes work on the GPU.
        resident = residency.applies(config, task)
        run_env = {}
        if resident:
            run_env.update(await residency.install(score_rt))
        out = await run_grader(config, score_rt, task, run_env)
        res = parse_markers(config, out, task, trace)
        if res["run_pass"] and not await confirm_correctness(score_rt, task, trace):
            # Both run and performance metrics require confirmed test completion.
            res["run_pass"] = 0.0
            res["performance_pass"] = 0.0
        if resident and res["run_pass"]:
            if not await residency.enforce(config, task, score_rt, trace, tree, kernel):
                res["run_pass"] = 0.0  # GPU-residency requirement not met.
                res["performance_pass"] = 0.0
        # Without paired timings, KernelGuard evaluates the source channel only.
        kg_v = kernelguard_gate.check(config, task, kernel, trace)
        res = kernelguard_gate.enforce(kg_v, res, trace)
        await _capture_grader_logs(score_rt, trace)
        _consecutive_infra_failures = 0
        return res
    except ScoreInfraError as e:
        # Preserve infrastructure failures as structured errors rather than
        # converting them into solution metrics.
        trace.info["score_error"] = f"ScoreInfraError: {str(e)[:300]}"
        _pmpp_info(trace)["score_error_kind"] = "env"
        _consecutive_infra_failures += 1
        await _capture_grader_logs(score_rt, trace)
        raise
    except Exception as e:  # noqa: BLE001
        trace.info["score_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        _pmpp_info(trace)["score_error_kind"] = "unexpected"
        _consecutive_infra_failures += 1
        await _capture_grader_logs(score_rt, trace)
        raise ScoreInfraError(
            f"unexpected scoring failure for {task.task_id}: "
            f"{type(e).__name__}: {str(e)[:200]}"
        ) from e
    finally:
        await score_rt.stop()
