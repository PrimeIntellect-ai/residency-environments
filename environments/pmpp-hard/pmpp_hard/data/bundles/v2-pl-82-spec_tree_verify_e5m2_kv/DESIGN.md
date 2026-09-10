# v2-pl-82-spec_tree_verify_e5m2_kv — DESIGN

Vein: attention / quantized-KV / decode pipelines (stateful, perf-gated).
Validated 2026-07-04 on RTX 5080 (sm_120, CUDA 13.0), WSL2.

## What the task is

A stateful speculative-decoding TREE VERIFICATION engine (SpecInfer/Medusa
lineage) over a paged FP8-E5M2 committed KV cache with grouped-query
attention. One solution_run = one speculation round per active sequence: a
draft forest (up to 64 nodes, parent[i] < i, multiple roots) is quantized
into transient SCRATCH pages drawn from the same physical pool as the
committed cache; every node gets a tree-causal attention output over
(committed prefix + its ancestor path including itself); nodes are verified
by comparing the byte-exact FNV signature of their quantized bytes against
supplied target signatures with a parent-AND cascade; the deepest fully
accepted chain (ties -> smallest node index) plus one unconditional bonus
token is committed (bit-identical byte copies, never requantized); all
scratch pages are released. E5M2 (S EEEEE MM, bias 15, max finite 57344) is
a deliberately different byte format from the sibling tasks' E4M3/int8.

Sources: SpecInfer token trees, Medusa/EAGLE multi-candidate verification,
vLLM speculative slot allocation; Pro-authoring vein notes in
metadata/pmpp-model-evals/v2-pro-authoring/ (spec-decode + paged-KV specs).

## Coupled cruxes

1. **Quantization IS control flow.** Acceptance compares the FNV-1a-64 fold
   (non-canonical basis) of each node's quantized scratch bytes against
   target_sig. One rounding slip in the normative E5M2 encoder (RNE
   half-to-even on a 2-bit mantissa, saturation above 57344 — reachable by
   the amax element itself since scales put amax in [32768, 65536) —
   subnormals below 2^-14, signed zeros) flips acceptance, which changes
   accepted_len, seq_len, committed bytes, kv_hash, the page tables, and
   every later step. The adversarial scenario feeds exact ties and
   saturating anchors (K crafts are pow2-downscaled so bytes/sigs are
   identical but scores stay well-conditioned).
2. **Tree-causal attention with a shared pool.** Every node attends the
   committed paged prefix PLUS its own ancestor path read from scratch
   pages; nodes off the path are excluded even at earlier indices; the set
   always includes the node itself (never empty, even at L=0). Binary-tree
   scenarios place sig-valid nodes under rejected parents (must not be
   accepted) and equal-depth accepted branches (smallest index wins).
3. **Two-phase page lifecycle in ONE pool.** Scratch pages are allocated
   row-ascending (lowest-free-id) at step start, committed pages are
   allocated WHILE scratch is still held, and scratch is released only after
   all commits — so committed page ids depend on the transient scratch
   footprint. churn_tight_pool sizes the pool exactly at the guaranteed
   bound. free_pages / total_allocs / total_frees / page_state_checksum are
   exact.
4. **Incremental hashing + bit-identical commits.** kv_hash folds every
   committed token's bytes with seq_len LAST (running-hash-friendly); commit
   must copy the exact scratch bytes (a requantization that is numerically
   harmless would still change nothing — but any byte-level divergence shows
   up immediately), and the naive from-scratch refold of ~500 tokens x 1 KB
   per sequence per step is a serial disaster.

## Reference architecture

- Quantize kernels: block per (row, node) / per row for the bonus; one warp
  per (KV head, K/V section); shuffle amax, exponent-derived power-of-two
  scale, bit-twiddled E5M2 encode (2-bit RNE with carry into the exponent).
- Sig/verify kernel: block per row; one thread per node folds its ~1 KB
  signature and writes its root-to-self path; thread 0 runs the <=64-step
  acceptance cascade and emits the commit source list.
- Attention kernel: grid ((row, node), kv-head, split); 4 warps round-robin
  the query heads of the kv group; per-warp online (m, l, acc[D]) with
  per-byte E5M2 dequant and shuffle score reductions; split 0 appends the
  ancestor-path walk; per-split partial records merged by LSE rescaling in
  a separate kernel.
- Commit: single-thread plan kernel (lowest-free-id bitmap allocation with
  scratch held, then scratch release) + block-parallel byte-copy kernel per
  committed token + one-thread-per-row incremental hash extension.
- Final kernel: per-sequence kv_hash finish (4 length bytes), page-state
  checksum, popcount free-page count.

naive_ref.cu (authoring artifact, not shipped): linear byte-array page
scans, one scalar thread per node for quantization, single-thread per-row
signature+cascade, one scalar thread per (row, node, head) two-pass
attention re-reading the paged cache and re-walking the path, and a full
kv_hash refold every step.

## Validation evidence (real outputs, RTX 5080)

- `make test_reference && ./test_reference` -> `passed 354 / 354`
  (8 shape-family scenarios x {base, exact_replay, reversed_active_order},
  118 steps per pass; scenarios: chain_min_B2,
  quant_e5m2_ties_saturation, forest_star_depth_ties (equal-depth ties +
  sig-ok-under-rejected-parent), deep_chain64_midbreak (64-node chains,
  D=128), churn_tight_pool (pool exactly at bound), gqa_hq32_hkv8,
  reject_all_bonus_only (incl. empty steps), bench_like_small).
- `make test_naive && ./test_naive` -> `passed 354 / 354` (independent
  clean GPU impl vs independent double CPU oracle — 3-way).
- Stub `solution.cu` compiles; `./test_student` -> `passed 0 / 354`.
- `./bench_reference` twice: `avg_ms=193.799796`, `avg_ms=191.540417`
  (avg per full 110-step sequence: 4x8 draft forests per row, planned
  acceptance avg 3.5+bonus, B=24 Hq=16 Hkv=4 D=128 P=16 N=32, L to ~495).
- `./bench_naive 2`: `avg_ms=1892.455811` => **ratio 9.8x** (>= 5x bar).

## Gate rationale

Per bench step, 12288 (node, head) tree queries each attend ~250 committed
tokens plus their path at D=128 (~800M fp32 MAC/step, ~88 GMAC per timed
sequence) over ~1 GB of quantized byte traffic with inline E5M2 decode.
Host replay is minutes of single-core work against a ~192 ms GPU budget and
solution_run may not synchronize or copy to host. Reference serial floors:
the bitmap allocator (<=~60 allocations/step), the per-row 64-node cascade,
and the per-row incremental hash (<= ~4.5 KB/step) — all microseconds. The
9.8x naive gap is optimization headroom on identical semantics (naive
passes the full suite); scalar-per-query attention plus full hash refolds
cannot approach the gate.

## Fairness notes

- Every byte rule is normative in spec_tree_verify_e5m2_kv_common.h: the
  E5M2 layout and encoder (ties, saturation, signed zeros, subnormals), the
  TOKEN FOLD order shared by signatures and kv_hash, the non-canonical FNV
  basis, the phase order (scratch alloc -> quantize -> attention -> verify
  -> commit-with-scratch-held -> release), lowest-free-id allocation, and
  the acceptance cascade with its tie-break.
- Attention is tolerance-checked (ATOL=RTOL=2.5e-3) and any numerically
  stable order is allowed; every control-flow-relevant output is integer or
  hash (exact) and is decided by bytes, not by float comparisons — no
  hidden thresholds anywhere.
- Shape family: B 1..24 scenarios, Hq to 32, Hkv to 8, D {64,128},
  P {8,16,32}, N {16,32,64}, chains/forests/binary trees, full-accept,
  all-reject (bonus-only), empty steps, node_count=1, pad-slot garbage.
- Input guarantees (finite ranges, normal amax, pool-sufficiency formula,
  parent[i] < i) are stated in the header.
- Determinism testable and tested: bit-exact replay; row-permutation
  invariance for everything except page_state_checksum (allocation
  interleaving is row-ordered by contract, as documented).
