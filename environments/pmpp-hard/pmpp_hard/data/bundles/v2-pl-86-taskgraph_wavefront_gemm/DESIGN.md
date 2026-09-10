# v2-pl-86-taskgraph_wavefront_gemm — DESIGN

Vein: in-kernel schedulers / megakernel systems (stateful, perf-gated).
Validated 2026-07-04 on RTX 5080 (sm_120, CUDA 13.0), WSL2.

## What the task is

A stateful GPU-resident mini-system. Persistent state: an R x C grid of
32x32 uint32 accumulator tiles plus a round counter. One `solution_run` =
one wavefront round: every tile (i,j) is updated exactly once by a task
whose inputs include the ALREADY-UPDATED left and top neighbor tiles of the
SAME round (border row/col of the neighbors feed a salt and linear terms).
The per-task math is a salted int8 32x32xK GEMM: U-tile bytes are XOR-mixed
with the 4 bytes of a salt derived from the updated neighbor borders + the
round counter + (i,j); the products accumulate into the persistent tile
with wraparound uint32 adds plus P1/P2-weighted neighbor border terms.
Outputs per round: round counter, global border row/col, a two-level
FNV-1a-64 checksum of the FULL state, and (dump rounds only) the entire
ACC grid. Everything exact; zero tolerance.

## Coupled cruxes

1. **A real dependency schedule that cannot be shortcut.** The salt makes
   every GEMM operand depend on the *updated* neighbor tiles of the same
   round, so tile (i,j) cannot start its heavy arithmetic before both
   neighbors finish — the wavefront DAG is load-bearing, not decorative.
   There is no algebraic decoupling: the XOR of salt bytes into int8
   operands is nonlinear in the accumulator values.
2. **GPU-resident scheduling at speed.** A one-launch-per-diagonal
   implementation eats 63 kernel-launch latencies per round on the max
   shape and idles the GPU at the narrow wavefront ends; the fast path is
   a persistent kernel with device-side dependency counters, an MPMC
   published-slot queue, and correct release/acquire discipline
   (`__threadfence` + L2-scoped `.cg` accessors) for cross-block
   producer/consumer visibility inside one grid.
3. **Heavy per-task arithmetic.** Each task is a 32x32xK int8 GEMM
   (K up to 256; max round = 1024 tiles x 262144 MAC = 268M MAC) — dp4a
   over packed u32 words with the salt XOR folded word-wise, shared-memory
   operand staging, register micro-tiles. Scalar per-byte arithmetic alone
   is >10x slower.
4. **Statefulness + exact checksum.** Rounds chain (round counter enters
   the salt; ACC accumulates), resets must restore a bit-exact zero state,
   and the full-state checksum uses the exact two-level FNV recipe — the
   root fold is done with the affine-FNV chunk composition
   `FNV(chunk,h) = A*(h & ~0xFF) + T[h & 0xFF]`, never a serial byte fold
   of the data.

## Reference architecture

- `twg_ref_sched_init_kernel`: per-round dep counters (i>0)+(j>0), queue
  slots cleared, task (0,0) pre-published, head=0/tail=1.
- `twg_ref_persistent_kernel` (128 threads/block, min(R*C, resident)
  blocks): thread 0 claims a slot via `atomicAdd(head)`, spins with
  `__nanosleep` until the slot is published; block loads the two updated
  neighbor border vectors via `__ldcg`, computes the salt, stages U/V
  tiles as packed u32 in padded shared memory, runs a 4x2 register
  micro-tile dp4a loop (salt XOR applied word-wise to U), updates the
  persistent tile via `__ldcg/__stcg`, then `__threadfence()` and
  decrements the right/down dependency counters, publishing newly-ready
  tasks with `atomicExch` into `atomicAdd(tail)` slots. Queue progress is
  deadlock-free: claimed slots are the lowest indices and every completed
  prefix publishes the next ready task.
- Outputs: round bump kernel; border copy kernel; per-tile FNV digest
  kernel (one thread per tile, R*C-way parallel); affine root table kernel
  (256 threads per 256-digest chunk building T[v] and A=P^n in parallel);
  single-thread chunk combine (<= 4 affine steps at max shape).
- `solution_reset` = cudaMemsetAsync of ACC + round counter (contract:
  grader resets before changing R or C, so run-local W=C*32 indexing of
  the persistent buffer is stable between resets).

## Validation evidence (real outputs, RTX 5080)

- `make test_reference && ./test_reference` -> `passed 60 / 60`
  (60 stateful scenarios x 2-6 rounds each: 5 distributions x 7 shape
  combos, 2 max-shape 32x32xK256, 15 seed-shifted, chain DAGs R=1/C=1,
  single tile 6 rounds, K flipping per round, P1=P2=0, zero-U with max P,
  mid-scenario reset, dump-every-round; 3 scenarios fully replayed for
  determinism; every round checks guards, input immutability, acc_dump
  untouched on dump==0, and all outputs vs the stateful CPU oracle).
- `make test_naive && ./test_naive` -> `passed 60 / 60` (independent clean
  impl agrees with the independent CPU oracle).
- Stub `solution.cu` compiles; `./test_student` -> `passed 0 / 60`.
- `./bench_reference 30` twice: `avg_ms=0.261684`, `avg_ms=0.261854`.
- `./bench_naive 15`: `avg_ms=3.969551` => **ratio 15.2x** (>= 5x bar).
- Per-case ref vs naive (ms): wave_max_uniform 0.333/6.368 (19.1x),
  wave_max_sparse 0.282/3.394 (12.0x), wave_wide_uniform 0.244/3.931
  (16.1x), wave_tall_small 0.238/4.006 (16.8x), wave_mid_saturate
  0.211/2.149 (10.2x).

naive_ref.cu = one clean kernel launch per anti-diagonal (dependency order
from launch boundaries), scalar per-byte GEMM (no dp4a, no staging), serial
single-thread root fold. It passes the full suite, so the 15.2x gap is pure
optimization headroom, not correctness slack.

## Gate rationale

The gate couples BOTH cruxes: the naive loses ~6x to scalar-vs-dp4a
arithmetic and the rest to 63 launch latencies + wavefront idling that a
persistent scheduler overlaps. Neither a fast GEMM behind per-diagonal
launches nor a persistent scheduler with scalar arithmetic passes a 5x
bar against 0.26 ms. Serial floors in the reference are negligible (one
single-thread combine over <= 4 affine chunks). Host replay is impossible:
`solution_run` may not perform sync host copies, and one max round costs
268M scalar MAC on the CPU against a 0.33 ms GPU budget.

## Fairness notes

- No hidden thresholds: every rule (salt formula, wraparound semantics,
  digest byte order, root fold order, reset semantics, round counter) is
  normative in `taskgraph_wavefront_gemm_common.h`; the header is the
  complete spec.
- Shape family: R, C in [1,32] (any integer), K in {64,128,256} and
  changeable per round, P1/P2 in [0,15], dump flag per round; 5 named
  distributions; tests include degenerate chains (R=1, C=1), odd sizes
  (17x13), all-zero U, and mid-scenario resets.
- acc_dump is checked as untouched on dump==0 rounds; guard bytes around
  every output catch out-of-bounds writes; inputs must not be modified.
- int32 GEMM sums cannot overflow: |g| <= 256*127*127 < 2^22; all uint32
  accumulation is defined as wraparound.
