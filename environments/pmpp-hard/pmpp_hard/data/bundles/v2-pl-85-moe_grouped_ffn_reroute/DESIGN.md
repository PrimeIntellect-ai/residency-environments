# v2-pl-85-moe_grouped_ffn_reroute — DESIGN

Vein: MoE routing / scatter-combine (stateless heavy pipeline, perf-gated).
Validated 2026-07-02 on RTX 5080 (sm_120, CUDA 13.0), WSL2.

## What the task is

One `solution_run` = a full exact-integer MoE layer:
router int8 GEMM (logits[N,E]) -> group-limited candidate ranking
(top-g_sel groups by max-logit, then (logit desc, id asc) candidate order)
-> phase-1 capacity dispatch ranked (gate desc, token asc, slot asc)
-> deterministic overflow RE-ROUTING to per-slot backup expert (cand[K+k])
-> phase-2 dispatch into residual capacity (phase-1 never displaced)
-> expert-major packing (phase-1 slice then phase-2 slice)
-> per-route int8 two-layer FFN (H hidden, ReLU >> qshift clamp 127)
-> int64 wraparound combine y[N,D] -> two-level FNV-1a-64 y_checksum.
Every output exact; zero tolerance.

## Coupled cruxes

1. **Exact routing semantics under ties.** Group-limited top-K with three
   distinct total orders (group rank, candidate rank, capacity rank), all
   tie-broken by ids. TIES / ZERO_X distributions generate massive logit
   collisions; any deviation in any of the three orders shifts packing,
   statuses, FFN inputs, y, and the checksum.
2. **Two-phase overflow re-routing.** Phase-2 candidates exist only when
   `K + slot < g_sel*(E/G)`; phase-2 rank uses the *backup* expert's logit;
   residual capacity couples phase-2 to phase-1 outcomes. Getting the
   "phase-1 never displaced" rule or the DROP_NO_BACKUP / DROP_OVERFLOW
   distinction wrong is invisible on uniform data and fatal on HOT_EXPERT.
3. **Perf gate is real arithmetic.** Router GEMM (up to 32768x128x128
   int8 MAC) + grouped per-expert FFN (up to 32768 kept routes x 2*D*H =
   2.1G MAC) demand dp4a + shared-memory tiling + gather-friendly grouped
   GEMM. A scalar-but-parallel clean implementation is 5-14x slower.
4. **Deterministic dispatch at speed.** Capacity ranking needs a stable
   order over up to 131072 routes with 22-bit gates — the reference uses a
   46-bit key LSD radix sort (chunked histograms + warp-synchronous stable
   scatter via `__match_any_sync`); phase-2 record order is made
   deterministic by sorting on the full key (atomic append order never
   observable).

## Reference architecture

- `mgf_ref_logits_kernel<D4>`: 128 tokens/block, wr + x staged in shared,
  x row in registers, dp4a inner loop, int4 vectorized row stores.
- `mgf_ref_route_kernel`: warp per token; group maxes via shfl butterflies,
  replicated group selection, 2K warp-argmax candidate rounds on packed
  `(biased_logit << 7) | (127 - e)` keys.
- LSD radix sort, 6x8-bit passes over `e(7) | inv_gate(22) | token(15) |
  slot(2)`: hist kernel (shared atomics) -> single-block chunk scan ->
  stable scatter (4 warps/block, per-warp bases, `__match_any_sync` batch
  ranking). Phase 2 re-sorts with a device-resident count (m2).
- `mgf_ref_ffn_kernel<D4,H4>`: one block per 32-route tile of one expert
  (tile map via prefix over `ceil(counts[e]/32)`, binary search); x rows
  gathered to shared then registers; w1/w2 staged tile-wise (64 rows);
  requantized hidden bytes in padded shared rows (bank-conflict-free);
  epilogue writes packed_y and wrapping u64 atomics into y (commutative =>
  deterministic).
- Checksum: per-16-row digests in parallel threads; root folded with the
  exact affine-FNV chunk composition `FNV(chunk,h) = A*(h & ~0xFF) +
  T[h & 0xFF]` (tables built in parallel; no serial byte fold of data).

## Validation evidence (real outputs, RTX 5080)

- `make test_reference && ./test_reference` -> `passed 55 / 55`
  (55 cases: 5 distributions x 7 shape combos + 2 max-shape + 15
  seed-shifted + 3 adversarial edges; 3 cases replayed for determinism).
- `make test_naive && ./test_naive` -> `passed 55 / 55` (independent
  clean impl agrees with independent CPU oracle).
- Stub `solution.cu` compiles; `./test_student` -> `passed 0 / 55`.
- `./bench_reference 20` twice: `avg_ms=0.861984`, `avg_ms=0.868243`.
- `./bench_naive 10`: `avg_ms=6.402311`  => **ratio 7.4x** (>= 5x gate bar).
- Per-case ref vs naive (ms, 10/5 iters): uniform_max 1.145/15.815 (13.8x),
  hot_reroute_tinycap 1.026/6.592 (6.4x), ties_wide 0.816/5.108 (6.3x),
  saturate_mid 0.731/3.966 (5.4x), uniform_smallE_routing 0.545/2.761
  (5.1x), zero_x_tie_cascade 0.767/4.645 (6.1x).

naive_ref.cu = clean parallel scalar GEMMs (no dp4a/tiling), global
bitonic sort, scalar FFN, serial root fold. It passes the full suite, so
the 7.4x gap is pure optimization headroom, not correctness slack.

## Gate rationale

Arithmetic (2.7G int8 MAC on the max case) dominates; coordination alone
cannot pass. Serial floors are negligible in the reference (largest serial
step: 8 affine root combines + two E<=128 prefix loops). Host replay of one
max case costs ~2.6G scalar MAC on CPU (~2-4 s single-core, >2000x the
1.1 ms GPU budget) and `solution_run` may not perform sync host copies, so
host-side replay is hopeless inside the timed path.

## Fairness notes

- No hidden thresholds: every rank/tie-break/fold order is normative in
  `moe_grouped_ffn_reroute_common.h`.
- Shape family (8 axes) + 5 named distributions; tests include odd N=5000,
  no-backup configs (g_sel*S == K), all-zero inputs, tiny-cap mass reroute.
- All unspecified regions (packed arrays beyond offsets[E]) are excluded
  from checks; guard bytes verify no out-of-bounds writes.
- int32 sums provably cannot overflow inside the legal family
  (|logit| <= 2,064,512; layer-2 |acc| <= 256*127*127).
