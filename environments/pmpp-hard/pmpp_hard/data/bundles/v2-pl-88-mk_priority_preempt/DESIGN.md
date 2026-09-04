# v2-pl-88-mk_priority_preempt — DESIGN

Vein: mk_* megakernel systems — multi-queue priorities / quantum
preemption. Validated 2026-07-04 on RTX 5080 (sm_120, CUDA 13.0), WSL2.

## What the task is

A stateful multi-queue priority scheduler with quantum preemption.
Persistent state: a gjid-indexed job table (class, remaining slices,
arrival round, and the job's FULL stored int8 weight row — jobs outlive
the run that ingested them and execute against later runs' X data). One
`solution_run` = ingest njobs new jobs + R scheduling rounds. Per round:
pending = live jobs whose arrival has come; the active class is the
smallest non-empty one (strict priority — arrivals of a lower class
preempt the machine between rounds); every pending job of that class
executes min(quantum, rem) slices (quantum preemption / yielding). Each
slice is a salted int8 dot-batch over 256 X rows x K (dp4a-heavy), where
BOTH the salt AND the X block selection depend on the global trace index
g — any scheduling error corrupts all arithmetic. Outputs: byte-exact
trace log, per-execution hierarchical FNV digests, commutative wraparound
y accumulation, final queue dumps in (class, gjid) order, and a
three-level FNV checksum over the post-run job table incl. stored
weights. All exact; zero tolerance.

## Coupled cruxes

1. **Exact preemptive schedule reconstruction.** Active-class selection,
   arrival gating, quantum pacing, per-round execution lists (gjid
   order), the global trace index chain, remaining counters, and empty
   rounds must all be bit-right — and they feed the salts and data-block
   choices of every slice, so the trace, digests, y and the state
   checksum all cascade. Carryover semantics (arrival resets to 0 at run
   end, gids chain, weight rows persist) couple runs.
2. **Launch-bound R-round pipeline.** A correct implementation is ~70
   small kernels per run; the contract explicitly legalizes CUDA graphs
   and persistent cooperative kernels because per-launch overhead is a
   first-order cost. The reference merges each round into two kernels
   (a single-block scheduler that also folds the previous round's
   execution digests, computes n_s, updates rem and writes trace words
   via a warp-shuffle two-level scan; and a grid-strided work kernel) and
   captures the whole run into a cached CUDA graph.
3. **Heavy schedule-coupled arithmetic done right.** Up to ~24K slices x
   256 rows x 256 MAC (~1.5 G MAC) per run: the reference stages the
   job's weight row in shared, XORs the salt word-wise, and uses
   warp-per-row lane-strided dp4a with a shuffle reduction so X traffic
   is fully coalesced (thread-per-row wastes ~8x of every sector and was
   measured 2.4x slower end-to-end).

## Validation evidence (real outputs, RTX 5080)

- `make test_reference && ./test_reference` -> `passed 60 / 60`
  (60 stateful scenarios x 1-3 runs: 5 distributions x 7 shape combos,
  2 max-shape, 15 seed-shifted, longtail starvation, quantum=1 thrash,
  all-arrivals-last (empty rounds), single-populated-class, mid-scenario
  reset, full-drain-to-empty-queues, carryover across changing M/X,
  minimum shape; 3 scenarios fully replayed for determinism; guards +
  input immutability every run).
- `make test_naive && ./test_naive` -> `passed 60 / 60` (independent
  clean impl agrees with the independent sequential CPU oracle).
- Stub `solution.cu` compiles; `./test_student` -> `passed 0 / 60`.
- `./bench_reference 30` twice: `avg_ms=0.816479`, `avg_ms=0.817340`.
- `./bench_naive 10`: `avg_ms=4.706897` => **ratio 5.8x** (>= 5x bar).
- Per-case ref vs naive (ms): mk_max_drain 1.032/7.180 (7.0x),
  mk_longtail 0.896/5.281 (5.9x), mk_spread 0.878/3.297 (3.8x),
  mk_burst 0.646/4.327 (6.7x), mk_thrash_q1 0.630/3.449 (5.5x).

naive_ref.cu = identical round structure but per-kernel launches (no
graph), scalar per-byte row dots (no dp4a, no word-XOR, no staging, no
coalescing), serial per-block digest folds. The scheduler bookkeeping
kernels are shared-shape with the reference on purpose: the gate measures
the arithmetic + launch-graph headroom, not the (un-gateable) serial
coordination floor. Bench iterations are reset+run pairs so the
jobs-per-reset cap holds for any iteration count.

## Gate rationale

Arithmetic dominates the gate: the three arithmetic-heavy cases are
5.9-7.0x, and the two scheduler-stress cases (arrival spread, quantum=1)
still clear 3.8x/5.5x because their slice work is real. Serial floors in
the reference are small and bounded (single-block scheduler ~5 us/round;
digest chains <= 64 folds). Host replay is impossible: no sync copies
allowed and a max run is ~1.5 G MAC + a 32-round stateful simulation
against a ~1 ms budget.

## Fairness notes

- Every rule (orders, salts, block selection, digest levels, trace/dump
  word layouts, carryover, reset, caps) is normative in
  `mk_priority_preempt_common.h`; the header is the complete spec. The
  launch-bound nature and the legality of CUDA graphs / cooperative
  kernels are stated in the contract rules.
- Shape family: M in [16384,32768] mult. of 256 (per run), K/Q
  reset-scoped, R in [8,32], quantum in {1,2,4,8}, njobs in [64,1024],
  five named distributions; harness guarantees valid descriptors and the
  MKP_MAX_JOBS_TOTAL cap (documented).
- Unspecified regions (trace/exec_digest beyond trace_len, queue_dump
  beyond the live count) are never compared; guard bytes police bounds;
  inputs immutable.
- No overflow: |res| <= 256*127*127 fits int32; y is defined wraparound
  u64 and commutative; rem <= 64 fits the 16-bit trace/dump fields;
  gjid < 4096 per reset epoch by grader guarantee.
