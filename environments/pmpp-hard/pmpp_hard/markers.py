"""Stable strings exposed to agents and result-processing tools.

Agent-visible and programmatically matched values are tracked in ``GOLDEN_SHA256``.
The pool preflight calls ``check_golden`` to detect compatibility changes.
"""

import hashlib

# Sandbox verification wrapper. When a reference binary is available, the
# performance check reports a paired student/reference ratio against the gate.
# Otherwise, it falls back to the visible absolute benchmark.
VERIFY_SCRIPT = r"""#!/bin/bash
set -u
G=/app/.grader
STUDENT="{student_file}"
cp -f "/app/$STUDENT" "$G/$STUDENT" 2>/dev/null || {{ echo "PMPP_VERIFY_BUILD: FAIL"; exit 0; }}
cd "$G" || {{ echo "PMPP_VERIFY_BUILD: FAIL"; exit 0; }}
NVCCFLAGS="$(cat /app/.grader/.nvccflags 2>/dev/null || echo '-O3 -std=c++17')"
mk() {{ make "$@" PY="${{PMPP_PY:-python3}}" NVCCFLAGS="$NVCCFLAGS"; }}
if mk {test_target} >/tmp/build.log 2>&1; then echo "PMPP_VERIFY_BUILD: PASS"; else echo "PMPP_VERIFY_BUILD: FAIL"; echo "PMPP_VERIFY_LOG: hidden"; exit 0; fi
if timeout -k 5 420 ./{test_target} >/tmp/test.log 2>&1; then echo "PMPP_VERIFY_CORRECTNESS: PASS"; else echo "PMPP_VERIFY_CORRECTNESS: FAIL"; echo "PMPP_VERIFY_LOG: hidden"; exit 0; fi
PERF="${{PMPP_PERF:-$(cat /app/.grader/.perf 2>/dev/null || echo 0)}}"
if [ "$PERF" = "1" ]; then
  if mk {bench_target} >/tmp/bbuild.log 2>&1; then
    if [ -x ./bench_reference ]; then
      bash /app/.grader/paired_selfcheck.sh {bench_target} "$(cat /app/.grader/.perf_target 2>/dev/null || echo 1.25)"
    elif timeout -k 5 420 ./{bench_target} >/tmp/bench.log 2>&1; then echo "PMPP_VERIFY_PERFORMANCE: UNAVAILABLE (private_reference)"; else echo "PMPP_VERIFY_PERFORMANCE: FAIL (bench_run)"; fi
  else echo "PMPP_VERIFY_PERFORMANCE: FAIL (bench_build)"; echo "PMPP_VERIFY_LOG: hidden"; fi
else
  echo "PMPP_VERIFY_PERFORMANCE: SKIP"
fi
exit 0
"""

# Measures student and reference binaries in an interleaved loop and reports the
# median ratio against the gate. It uses shell and awk for compatibility with the
# CUDA runtime image. Arguments are the benchmark path and target ratio.
PAIRED_SELFCHECK = r"""#!/bin/bash
BENCH="$1"; TGT="${2:-1.25}"; TMP="$(mktemp)"
for i in 1 2 3 4 5; do
  s=$(timeout -k 5 420 ./"$BENCH" 2>/dev/null | grep -oE 'avg_ms=[0-9.]+' | head -1 | cut -d= -f2)
  r=$(timeout -k 5 420 ./bench_reference 2>/dev/null | grep -oE 'avg_ms=[0-9.]+' | head -1 | cut -d= -f2)
  [ -n "$s" ] && [ -n "$r" ] && awk -v a="$s" -v b="$r" 'BEGIN{ if (b>0) printf "%.6f\n", a/b }' >> "$TMP"
done
if [ ! -s "$TMP" ]; then echo "PMPP_VERIFY_PERFORMANCE: FAIL (no_timing)"; rm -f "$TMP"; exit 0; fi
RATIO=$(sort -n "$TMP" | awk '{ v[NR]=$1 } END { print v[int((NR+1)/2)] }')
rm -f "$TMP"
PASS=$(awk -v x="$RATIO" -v t="$TGT" 'BEGIN{ print (x<=t)?1:0 }')
if [ "$PASS" = "1" ]; then echo "PMPP_VERIFY_PERFORMANCE: PASS (paired_ratio=$RATIO target=$TGT)"
else echo "PMPP_VERIFY_PERFORMANCE: FAIL (paired_ratio=$RATIO target=$TGT)"; fi
"""

# Guides agents on performance-gated tasks to use the verification script for
# both correctness and performance feedback.
FEEDBACK_INSTRUCTION = (
    "\n\n## Feedback loop (required for this performance-gated task)\n"
    "A `./verify_student` script is in your workspace. Use it as your MAIN feedback loop: after every "
    "edit, run `./verify_student` and read the markers. Your solution is NOT done until it reports BOTH "
    "`PMPP_VERIFY_CORRECTNESS: PASS` and `PMPP_VERIFY_PERFORMANCE: PASS`.\n"
    "The performance line shows your live ratio vs the target gate, e.g. "
    "`PMPP_VERIFY_PERFORMANCE: FAIL (paired_ratio=1.54 target=1.25)` — this is the SAME paired "
    "student/reference gate you are scored on. Keep optimizing the kernel until `paired_ratio` is at or "
    "below the target, with a little margin (the benchmark is noisy). A passing correctness check, a "
    "`PERFORMANCE: SKIP`, or not running `verify_student` at all does NOT mean you are done."
    " If the wrapper instead reports `PMPP_VERIFY_PERFORMANCE: UNAVAILABLE (private_reference)`, "
    "the source-stripped local reference could not be provided. Use `./.grader/bench_student` to "
    "track `avg_ms`; the clean scorer still applies the private paired gate, so this degraded local "
    "state is not a performance pass and cannot be resolved by further edits alone."
)

TRITON_FEEDBACK_INSTRUCTION = (
    "\n\n## Feedback loop (required for this performance-gated task)\n"
    "A `./verify_student` script is in your workspace. Use it after every edit for build and "
    "correctness feedback. The private Python reference is intentionally absent from the agent "
    "sandbox, so local verification reports `PMPP_VERIFY_PERFORMANCE: UNAVAILABLE "
    "(private_reference)` after a successful benchmark instead of claiming a paired pass. Run "
    "`./.grader/bench_student` to compare your own `avg_ms` while optimizing. Final scoring runs "
    "your submission and the private reference separately in a clean sandbox and applies the "
    "configured paired-ratio gate. A local correctness pass does NOT imply a final performance pass."
)

# Directs agents to save their implementation in the evaluated file and leave a
# best-effort solution when verification is incomplete. ``student_file`` is
# substituted when the prompt is built.
ALWAYS_SUBMIT_INSTRUCTION = (
    "\n\n## Submit a solution — never decline (required)\n"
    "Write your implementation DIRECTLY into the `{student_file}` stub in your workspace (edit that "
    "file on disk and keep it current as you iterate). That on-disk file is what gets compiled and "
    "graded — code you only put in a chat message is NOT graded. Always leave your most complete, "
    "best-effort implementation in `{student_file}`, even if you are unsure it is fully correct or "
    "run short on time to verify it: a partial or imperfect kernel is still compiled and scored, but "
    "declining the task, apologizing, or leaving the stub unmodified scores ZERO. You have ample turns "
    "and context — do NOT stop early and do NOT reply that you cannot complete or verify it 'in time'. "
    "Always produce and save your best full attempt."
)

SINGLE_SHOT_INSTRUCTION = (
    "\n\n## Single-shot (no tools, no iteration)\n"
    "You CANNOT compile, run, or test anything — there is no shell and no grader in your workspace. "
    "Reason carefully from the contract above and emit your COMPLETE, final CUDA kernel exactly once. "
    "Your entire answer's last fenced code block must be a single ```cuda block containing the full "
    'translation unit (all includes and the `extern "C" __global__ void solution_kernel(...)` '
    "definition). It will be extracted verbatim and graded — there is no second attempt and no feedback. "
    "Do not output a partial kernel, pseudocode, or placeholders."
)

GRADER_HIDDEN_SINGLE_SHOT = (
    b"Single-shot: no grader, no tools. Emit your "
    b"complete kernel as one ```cuda block in your final message.\n"
)
GRADER_HIDDEN_OPAQUE = (
    b"No local grader for this task; write the kernel from the contract in README.md.\n"
)

# Operational tooling relies on this stable prefix.
PREFLIGHT_FAIL_PREFIX = "PMPP preflight FAILED — refusing to load a corrupt pool"

# The scorer and agent tools match these marker substrings. The performance pass
# prefix covers both paired-ratio and absolute-fallback results.
M_BUILD_PASS = "PMPP_VERIFY_BUILD: PASS"
M_BUILD_FAIL = "PMPP_VERIFY_BUILD: FAIL"
M_RUN_PASS = "PMPP_VERIFY_CORRECTNESS: PASS"
M_RUN_FAIL = "PMPP_VERIFY_CORRECTNESS: FAIL"
M_PERF_PASS = "PMPP_VERIFY_PERFORMANCE: PASS"
M_PERF_FAIL = "PMPP_VERIFY_PERFORMANCE: FAIL"
M_PERF_SKIP = "PMPP_VERIFY_PERFORMANCE: SKIP"
M_LOG_HIDDEN = "PMPP_VERIFY_LOG: hidden"

# Scorer-side residency and policy markers are written to ``trace.info`` rather
# than the agent sandbox. They are compatibility-tracked like other parsed values.
M_RESIDENCY_PASS = "PMPP_VERIFY_RESIDENCY: PASS"
M_RESIDENCY_FAIL = "PMPP_VERIFY_RESIDENCY: FAIL"
M_POLICY_FAIL = "PMPP_VERIFY_POLICY: FAIL"

# SHA-256 values for compatibility-tracked strings. Updating a tracked string
# requires updating its corresponding digest.
GOLDEN_SHA256: dict[str, str] = {
    "VERIFY_SCRIPT": "3d89ef2f15c0de3aecd013ce22740cac7f22e8340dde131baba4bf891c41d55c",
    "PAIRED_SELFCHECK": "0110b3686c597458730488d7179b3d0b82f71d4a334fe9dd0ccaf92e20163f97",
    "FEEDBACK_INSTRUCTION": "025911efaa5ee85a26a3343b354862a8159d8aebdc05c27a883a666feb7569c6",
    "TRITON_FEEDBACK_INSTRUCTION": "b84a1cafc6a141088508efd6f6d3cacadf1517abd3954f4491f0025cc5ac96af",
    "SINGLE_SHOT_INSTRUCTION": "64077c0ecb01f93645f41f17492bb5624192b51f13313db2ff4d63ee320cd1a8",
    "GRADER_HIDDEN_SINGLE_SHOT": "e6ac30407e9337184a622754c8685e923a74e3f50ef6f51a542d8ab12f49f0eb",
    "GRADER_HIDDEN_OPAQUE": "6d7edff894c14bf136be7e5e3654b59fa0f772483b5e5d10cc518a631246d414",
    # Residency and policy markers are tracked from their introduction.
    "M_RESIDENCY_PASS": "b4538972a0fd637f1503a20b67921bb106c75df90a8b50e5e4957014894e7cf8",
    "M_RESIDENCY_FAIL": "fcf343739391c1542bc61566d70fc1c2d83851a5054ead35e286a876c397957d",
    "M_POLICY_FAIL": "d600699f2a1255db51435342c3374a50ef404772dbe8de6717bdf54ddba4af7e",
}
# Hash of a representative rendered verification script.
GOLDEN_RENDER_SAMPLE = (
    "da6ee1691940e8c45b35f7a791f47ba84c036dbd742db6f6781d7c96f9695429"
)


def _sha(x: str | bytes) -> str:
    return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()


def check_golden() -> list[str]:
    """Return tracked values whose content does not match its compatibility hash."""
    vals = {
        "VERIFY_SCRIPT": VERIFY_SCRIPT,
        "PAIRED_SELFCHECK": PAIRED_SELFCHECK,
        "FEEDBACK_INSTRUCTION": FEEDBACK_INSTRUCTION,
        "TRITON_FEEDBACK_INSTRUCTION": TRITON_FEEDBACK_INSTRUCTION,
        "SINGLE_SHOT_INSTRUCTION": SINGLE_SHOT_INSTRUCTION,
        "GRADER_HIDDEN_SINGLE_SHOT": GRADER_HIDDEN_SINGLE_SHOT,
        "GRADER_HIDDEN_OPAQUE": GRADER_HIDDEN_OPAQUE,
        "M_RESIDENCY_PASS": M_RESIDENCY_PASS,
        "M_RESIDENCY_FAIL": M_RESIDENCY_FAIL,
        "M_POLICY_FAIL": M_POLICY_FAIL,
    }
    bad = [name for name, val in vals.items() if _sha(val) != GOLDEN_SHA256[name]]
    sample = VERIFY_SCRIPT.format(
        student_file="solution.cu",
        test_target="test_student",
        bench_target="bench_student",
    )
    if _sha(sample) != GOLDEN_RENDER_SAMPLE:
        bad.append("VERIFY_SCRIPT_RENDER")
    return bad
