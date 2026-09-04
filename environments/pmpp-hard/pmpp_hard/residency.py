"""Enforce GPU-residency and submitted-source policies during scoring.

Correctness-only tasks may expose oracle headers required by their tests. Static
checks prevent direct dependencies on grader implementations and control inputs.
A CUPTI activity shim independently records executed GPU kernels in the scoring
sandbox, with an optional GPU-time floor derived from the reference solution.

The source policy covers oracle headers and symbols, grader translation units,
the Python reference module, and access to residency or benchmark controls.

Python tasks can additionally require a declared and launched Triton JIT kernel
whose name appears in the runtime activity report. These checks establish useful
execution invariants but do not attribute every computation to a particular
kernel. Missing or ambiguous probe data is classified as an infrastructure error.
"""

import dataclasses
import json
import re

from pmpp_hard import markers
from pmpp_hard.config import PMPPHardConfig, PMPPHardTaskData
from pmpp_hard.errors import ScoreInfraError
from pmpp_hard.paths import DataTree
from pmpp_hard.provision import perf_enabled

# Paths used only inside the scoring sandbox.
RES_DIR = "/app/.residency"
SHIM_SRC = f"{RES_DIR}/pmpp_residency_shim.c"
SHIM_SO = f"{RES_DIR}/pmpp_residency_shim.so"
STUDENT_OUT = f"{RES_DIR}/student.jsonl"
REF_OUT = f"{RES_DIR}/reference.jsonl"

# CUPTI activity shim compiled against the scoring image's installed headers.
SHIM_C = r"""/* PMPP GPU-residency probe: CUPTI Activity API shim, loaded via LD_PRELOAD into the
 * authoritative correctness run. Counts real device kernel executions + their GPU time
 * regardless of launch path (runtime API, driver API, static cudart, Triton, graphs).
 * Each process appends ONE JSON line to $PMPP_RESIDENCY_OUT at exit; the scorer sums.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cupti.h>

static unsigned long long g_kernels = 0;
static unsigned long long g_gpu_ns = 0;
static unsigned long long g_memcpy = 0;
static char g_names[6144] = "|";
static size_t g_names_len = 1;
static int g_cupti_rc = -1; /* -1 = never attempted, 0 = ok, >0 = CUptiResult */
static pid_t g_pid = 0;     /* constructor's pid: forked children (shell subshells)
                             * must NOT flush (CUPTI deadlocks on inherited state
                             * post-fork) and must not double-write the counters. */

static void add_name(const char *n) {
    if (!n || !*n) return;
    size_t l = strlen(n);
    if (l > 512) l = 512;
    char probe[516];
    probe[0] = '|';
    memcpy(probe + 1, n, l);
    probe[l + 1] = '|';
    probe[l + 2] = 0;
    if (strstr(g_names, probe)) return; /* already recorded */
    if (g_names_len + l + 2 >= sizeof(g_names)) return;
    memcpy(g_names + g_names_len, n, l);
    g_names_len += l;
    g_names[g_names_len++] = '|';
    g_names[g_names_len] = 0;
}

static void CUPTIAPI buf_requested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *size = 1 << 20;
    *buffer = (uint8_t *)malloc(*size);
    *maxNumRecords = 0;
}

static void CUPTIAPI buf_completed(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                                   size_t size, size_t validSize) {
    (void)ctx; (void)streamId; (void)size;
    CUpti_Activity *rec = NULL;
    while (cuptiActivityGetNextRecord(buffer, validSize, &rec) == CUPTI_SUCCESS) {
        switch (rec->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)rec;
            g_kernels++;
            if (k->end > k->start) g_gpu_ns += (unsigned long long)(k->end - k->start);
            add_name(k->name);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY:
            g_memcpy++;
            break;
        default:
            break;
        }
    }
    free(buffer);
}

__attribute__((constructor)) static void pmpp_res_init(void) {
    if (!getenv("PMPP_RESIDENCY_OUT")) return;
    g_pid = getpid();
    CUptiResult rc = cuptiActivityRegisterCallbacks(buf_requested, buf_completed);
    if (rc == CUPTI_SUCCESS) {
        rc = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        if (rc != CUPTI_SUCCESS) rc = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    }
    if (rc == CUPTI_SUCCESS) {
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY); /* forensics only; best-effort */
    }
    g_cupti_rc = (int)rc;
}

__attribute__((destructor)) static void pmpp_res_fini(void) {
    const char *out = getenv("PMPP_RESIDENCY_OUT");
    if (!out || g_cupti_rc == -1) return;
    if (getpid() != g_pid) return; /* forked child: no flush (deadlock), no double-count */
    if (g_cupti_rc == 0) cuptiActivityFlushAll(1);
    char line[8192];
    int n = snprintf(line, sizeof(line),
                     "{\"pid\":%d,\"cupti_rc\":%d,\"kernels\":%llu,\"gpu_ns\":%llu,"
                     "\"memcpy\":%llu,\"names\":\"%s\"}\n",
                     (int)getpid(), g_cupti_rc, g_kernels, g_gpu_ns, g_memcpy, g_names);
    if (n <= 0) return;
    if ((size_t)n >= sizeof(line)) n = sizeof(line) - 1;
    int fd = open(out, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) return;
    ssize_t w = write(fd, line, (size_t)n);
    (void)w;
    close(fd);
}
"""

BUILD_CMD = (
    f"gcc -O2 -fPIC -shared -I/usr/local/cuda/include {SHIM_SRC} -o {SHIM_SO} "
    f"-L/usr/local/cuda/lib64 -lcupti -Wl,-rpath,/usr/local/cuda/lib64 "
    f"&& test -x {SHIM_SO} && echo PMPP_SHIM_OK"
)


def applies(config: PMPPHardConfig, task: PMPPHardTaskData) -> bool:
    """Return whether residency checking applies outside a performance gate."""
    return config.residency_check and not perf_enabled(config, task)


# Static submitted-source policy.

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.M)
_ORACLE_HDR_RE = re.compile(r".*_oracle\.(h|hpp|cuh|inc)$")
_ORACLE_SYM_RE = re.compile(r"\b\w*_oracle\w*\b")
_REF_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+reference\b|from\s+reference\s+import)", re.M
)
_C_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def static_policy_violation(task: PMPPHardTaskData, kernel: bytes) -> str | None:
    """Return a violation for direct dependencies on grader internals.

    Environment checks inspect the original source. C and C++ include and symbol
    checks ignore comments so explanatory prose does not trigger the policy. The
    policy is intentionally limited to direct references.
    """
    src = kernel.decode("utf-8", "replace")
    if "PMPP_RESIDENCY" in src:
        return "references the residency probe environment (PMPP_RESIDENCY*)"
    # Submitted solutions do not need the benchmark seed; it is reserved for the grader.
    if "PMPP_BENCH_SEED" in src:
        return (
            "reads the per-rollout bench seed (PMPP_BENCH_SEED) — input-prediction hack"
        )
    if task.student_file.endswith(".py"):
        if _REF_IMPORT_RE.search(src):
            return "imports the grader oracle module (reference.py)"
        return None
    code = _C_COMMENT_RE.sub(" ", src)
    for inc in _INCLUDE_RE.findall(code):
        base = inc.rsplit("/", 1)[-1]
        if _ORACLE_HDR_RE.fullmatch(base):
            return f"includes the oracle header {inc!r}"
        if base.startswith(("test_", "bench_")):
            return f"includes a grader TU {inc!r}"
    if _ORACLE_SYM_RE.search(code):
        return "references the oracle symbol namespace (*_oracle*)"
    return None


# Static Triton requirements.

_JIT_LINE_RE = re.compile(r"^\s*@\s*(?:triton\.)?jit\b")
_GRID_LAUNCH_RE = re.compile(r"\w\s*\[[^\]\n]+\]\s*\(")
_DEF_RE = re.compile(r"\s*def\s+(\w+)\s*\(")


def jit_kernel_names(src: str) -> set[str]:
    """Return function names defined under a Triton JIT decorator."""
    names: set[str] = set()
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if not _JIT_LINE_RE.match(line):
            continue
        for j in range(i + 1, min(i + 8, len(lines))):
            m = _DEF_RE.match(lines[j])
            if m:
                names.add(m.group(1))
                break
            if not re.match(r"\s*@", lines[j]):
                break
    return names


def triton_static_violation(kernel: bytes) -> str | None:
    """Return a violation when Triton source lacks a JIT kernel or grid launch."""
    src = kernel.decode("utf-8", "replace")
    if not jit_kernel_names(src):
        return "no @triton.jit kernel defined"
    if not _GRID_LAUNCH_RE.search(src):
        return "no kernel[grid](...) launch"
    return None


# Runtime residency probe.


@dataclasses.dataclass
class ProbeReport:
    """Aggregated activity reported by the processes in one probed run."""

    lines: int = 0
    kernels: int = 0
    gpu_ns: int = 0
    memcpy: int = 0
    names: set = dataclasses.field(default_factory=set)
    cupti_errors: int = 0
    parse_errors: int = 0


def parse_report(text: str) -> ProbeReport:
    rep = ProbeReport()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            rep.lines += 1
            rep.kernels += int(d.get("kernels", 0))
            rep.gpu_ns += int(d.get("gpu_ns", 0))
            rep.memcpy += int(d.get("memcpy", 0))
            if int(d.get("cupti_rc", 0)) != 0:
                rep.cupti_errors += 1
            rep.names.update(n for n in str(d.get("names", "")).split("|") if n)
        except (ValueError, TypeError):
            rep.parse_errors += 1
    return rep


async def install(runtime) -> dict:
    """Build the probe in the scoring sandbox and return its runtime environment.

    Build failures are reported as infrastructure errors.
    """
    await runtime.write(SHIM_SRC, SHIM_C.encode())
    res = await runtime.run(["bash", "-lc", BUILD_CMD], {})
    out = res.stdout + res.stderr
    if "PMPP_SHIM_OK" not in out:
        raise ScoreInfraError(
            f"residency shim failed to build in the scoring sandbox — env fault "
            f"(image lacks gcc/cupti?), not a student fail: {out.strip()[-300:]}"
        )
    return {"LD_PRELOAD": SHIM_SO, "PMPP_RESIDENCY_OUT": STUDENT_OUT}


async def _reference_baseline(
    config: PMPPHardConfig, task: PMPPHardTaskData, runtime, tree: DataTree, info: dict
) -> ProbeReport | None:
    """Measure the reference on the same test for an optional GPU-time baseline.

    If a comparable reference is unavailable, record the reason and omit the
    optional floor while retaining the launch-count requirement.
    """
    if not task.student_file.endswith(".cu"):
        info["gpu_floor"] = "skipped: no CUDA reference baseline for .py tasks"
        return None
    ref = tree.find_reference(task.task_id, cuda_only=True)
    if ref is None:
        info["gpu_floor"] = "skipped: no data/reference CUDA solution"
        return None
    await runtime.write("/app/.grader/reference_solution.cu", ref.read_bytes())
    script = (
        f'cd /app/.grader && NV="$(cat .nvccflags)" && '
        f"timeout -k 5 900 make {task.test_target.replace('student', 'reference')} "
        f'PY=python3 NVCCFLAGS="$NV" >/tmp/resrefb.log 2>&1 && '
        f"rm -f {REF_OUT} && "
        f"LD_PRELOAD={SHIM_SO} PMPP_RESIDENCY_OUT={REF_OUT} "
        f"timeout -k 5 420 ./{task.test_target.replace('student', 'reference')} "
        f">/tmp/resref.log 2>&1; cat {REF_OUT} 2>/dev/null"
    )
    res = await runtime.run(["bash", "-lc", script], {})
    rep = parse_report(res.stdout + res.stderr)
    if rep.lines == 0 or rep.kernels == 0:
        info["gpu_floor"] = "skipped: reference baseline unavailable (build/run failed)"
        return None
    return rep


def _match_jit_names(jit_names: set[str], record_names: set[str]) -> bool:
    """Return whether an activity record matches a JIT-defined function name.

    The boundary check avoids matching common length-prefixed C++ symbol segments.
    """
    for j in jit_names:
        pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(j)}")
        if any(pat.search(n) for n in record_names):
            return True
    return False


async def enforce(
    config: PMPPHardConfig,
    task: PMPPHardTaskData,
    runtime,
    trace,
    tree: DataTree,
    kernel: bytes,
) -> bool:
    """Evaluate residency after a correct run.

    Return whether the residency requirements pass. Missing or ambiguous telemetry
    raises ``ScoreInfraError`` so infrastructure failures are classified separately.
    """
    res = await runtime.run(["bash", "-lc", f"cat {STUDENT_OUT} 2>/dev/null"], {})
    rep = parse_report(res.stdout + res.stderr)
    info = trace.info.setdefault("pmpp", {}).setdefault("residency", {})
    info.update(
        {
            "kernels": rep.kernels,
            "gpu_ns": rep.gpu_ns,
            "memcpy": rep.memcpy,
            "report_lines": rep.lines,
            "cupti_errors": rep.cupti_errors,
            "kernel_names": sorted(rep.names)[:16],
        }
    )
    if rep.lines == 0:
        raise ScoreInfraError(
            f"residency probe produced no report for {task.task_id} — LD_PRELOAD/CUPTI "
            f"env fault (shim never ran), not a student fail"
        )
    fail = None
    if rep.kernels < config.residency_min_launches:
        if rep.cupti_errors:
            raise ScoreInfraError(
                f"residency probe ambiguous for {task.task_id}: CUPTI init failed in "
                f"{rep.cupti_errors} process(es) and zero kernels recorded — env fault"
            )
        fail = (
            f"no GPU kernel execution recorded "
            f"(kernels={rep.kernels} < min {config.residency_min_launches})"
        )
    if fail is None and config.triton_premise and task.student_file.endswith(".py"):
        jn = jit_kernel_names(kernel.decode("utf-8", "replace"))
        if not jn or not _match_jit_names(jn, rep.names):
            fail = "no @triton.jit-defined kernel executed on the GPU (torch-only solution)"
        info["jit_names"] = sorted(jn)
    if fail is None and config.residency_gpu_frac > 0:
        ref = await _reference_baseline(config, task, runtime, tree, info)
        if ref is not None:
            floor = config.residency_gpu_frac * ref.gpu_ns
            info["reference_gpu_ns"] = ref.gpu_ns
            info["gpu_floor_ns"] = int(floor)
            if rep.gpu_ns < floor:
                fail = (
                    f"GPU-time floor: student {rep.gpu_ns}ns < "
                    f"{config.residency_gpu_frac} x reference {ref.gpu_ns}ns "
                    f"(decoy-kernel host compute?)"
                )
    marker = (
        markers.M_RESIDENCY_PASS
        if fail is None
        else f"{markers.M_RESIDENCY_FAIL} ({fail})"
    )
    info["marker"] = marker
    if fail is not None:
        info["fail_reason"] = fail
    trace.info["verify_markers"] = (
        trace.info.get("verify_markers", "") + "\n" + marker
    ).strip()[-800:]
    return fail is None
