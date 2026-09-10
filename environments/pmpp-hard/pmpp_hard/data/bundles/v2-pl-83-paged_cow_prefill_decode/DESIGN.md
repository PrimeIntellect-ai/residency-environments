# v2-pl-83-paged_cow_prefill_decode — DESIGN

Vein: attention / quantized-KV / decode pipelines (stateful, perf-gated).
Validated 2026-07-04 on RTX 5080 (sm_120, CUDA 13.0), WSL2.

## What the task is

A stateful continuous-batching attention engine (vLLM lineage) over a
REFCOUNTED shared paged fp16 KV cache with copy-on-write forks. One
solution_run = one scheduler step over a mixed batch: chunked-prefill rows
(up to 64 tokens, one causal output per appended token), decode rows (same
path, 1 token), FORK rows (a fresh sequence id shares all full pages of a
live source by refcount; only the partial tail page is eagerly copied), and
RELEASE rows (references dropped; pages free when refcount hits zero and
are reusable in the SAME step). KV storage is IEEE binary16 with normative
round-to-nearest-even conversion (== __float2half_rn), so the byte-exact
kv_hash and page-state checksum pin the storage without a group-quant
recipe — the lifecycle is the hard part here, deliberately complementary
to the sibling tasks (80: E4M3 sink/window eviction; 81: MLA int8 latent;
82: E5M2 tree verify).

Sources: vLLM PagedAttention block sharing / copy-on-write and continuous
batching with chunked prefill (Sarathi-style mixed batches); Pro-authoring
vein notes in metadata/pmpp-model-evals/v2-pro-authoring/.

## Coupled cruxes

1. **Refcounted sharing + eager COW.** Fork semantics are exact and
   observable: full pages shared (refcount grows), partial tail copied into
   the lowest free page, source keeps its original tail; releases decrement
   and free at zero with same-step reuse. The page-state checksum folds the
   tables, and free_pages/alloc/free/fork/release counters are exact — any
   sharing shortcut (copying everything, or sharing the tail) changes page
   ids or counters immediately. Scenarios fork at len%P==0 (no copy),
   len%P==P-1 (copy then fill), and chain forks across steps.
2. **Running-hash inheritance.** kv_hash folds the LOGICAL token stream
   with seq_len LAST; a fork's stream is byte-identical to its source's
   prefix, so the running hash must be inherited verbatim, extended by
   appends, and reset on release. A from-scratch refold (naive) is
   O(L * 2 KB) serial per sequence per step; recomputing per fork is the
   same trap.
3. **Mixed-regime attention in one step.** The same launch must cover
   64-token causal prefill chunks (GEMM-shaped work per row) and single
   decode tokens over sequences of ~470 tokens read through shared page
   tables. Per-token causal prefixes ([0, p] including itself), exact-zero
   pad slots under garbage inputs, and RELEASE rows emitting all-zero
   output blocks are all checked.
4. **Bit-exact fp16 storage.** Conversion is normative binary16 RNE
   including subnormal halves and signed zeros; the adversarial scenario
   feeds exact ties at normal steps (1+2^-11, 2049), subnormal ties
   (2^-25 -> 0, 1.5*2^-24 -> 2 units), the 2^-14 boundary, and -0.0f. One
   rounding slip changes kv_hash forever.

## Reference architecture

- Admin kernel (single thread): releases -> forks -> append planning in
  contract order over a refcount array + free bitmap (ffs lowest-id);
  records COW copy jobs, per-token destination slots, and pre-append base
  lengths; inherits/resets running hashes.
- COW copy kernel: block per fork row, parallel fp16 copies of the
  occupied tail slots.
- Append kernel: block per (row, token), threads convert Hkv*D values with
  __float2half_rn.
- Hash-extend kernel: one thread per appending row folds only this step's
  stored halves into the persistent running hash.
- Attention kernel: grid ((row, token), kv-head, split); 4 warps
  round-robin the query heads of the kv group; per-warp online (m, l,
  acc[D]) with fp16 loads through the (possibly shared) page table and
  shuffle score reductions; split partials merged by LSE rescaling.
- Final kernel: kv_hash finish (4 length bytes), page-state checksum,
  refcount-zero free-page count, counters.

naive_ref.cu (authoring artifact, not shipped): single-thread admin with
linear refcount scans and a single-thread serial tail copy, one scalar
thread per token for conversion, one scalar thread per (row, token, head)
two-pass attention re-reading the paged cache, and a full kv_hash refold
of every sequence every step.

## Validation evidence (real outputs, RTX 5080)

- `make test_reference && ./test_reference` -> `passed 342 / 342`
  (8 shape-family scenarios x {base, exact_replay, reversed_active_order},
  114 steps per pass; scenarios: basic_mixed_min, f16_rne_ties_subnormals,
  fork_chain_cow (chained forks, fork-only rows), release_recycle_ids
  (id + page reuse), cow_page_boundaries (rem 0 and P-1), gqa_hq32_chunk64
  (D=128, C=64), decode_heavy_empty_steps, bench_like_small).
- `make test_naive && ./test_naive` -> `passed 342 / 342` (independent
  clean GPU impl vs independent double CPU oracle — 3-way; the adversarial
  pass also pins the host binary16 converter against __float2half_rn).
- Stub `solution.cu` compiles; `./test_student` -> `passed 0 / 342`.
- `./bench_reference` twice: `avg_ms=170.681293`, `avg_ms=171.071039`
  (avg per full 96-step sequence: 6 prefill steps then 90 mixed steps with
  12 decode rows to L=474, 2 rotating 6x64 prefill/release rows, 2
  fork/COW/decode rows; B=16 Hq=16 Hkv=4 D=128 P=16 C=64).
- `./bench_naive 2`: `avg_ms=1679.281128` => **ratio 9.8x** (>= 5x bar).

(Authoring note: the first validation run caught an oracle bug — the token
fold interleaving per KV head — via the Hkv>=2 scenarios; fixed in the
oracle, after which all three implementations agreed. The shipped contract
text was always the interleaved order.)

## Gate rationale

Per mixed step, ~140 appended tokens x 16 heads attend prefixes up to ~470
tokens at D=128 (~130-630M fp32 MAC/step, ~20 GMAC per timed sequence) over
~2 GB of fp16 cache traffic through shared page tables. Host replay is
minutes single-core against a ~171 ms budget and solution_run may not
synchronize or copy to host. Reference serial floors: the admin kernel
(~150 plan iterations + a handful of ffs allocations per step) and the
per-row incremental hash (<= 128 KB/step on prefill rows, parallel across
rows). The 9.8x naive gap is optimization headroom on identical semantics
(naive passes the full suite).

## Fairness notes

- Every rule is normative in paged_cow_prefill_decode_common.h: binary16
  RNE conversion, the TOKEN FOLD and non-canonical FNV basis, the phase
  order (releases -> forks -> appends -> attention), lowest-free-id
  allocation, eager-COW fork semantics, same-step page reuse after
  release, and the shared-page immutability invariant.
- Attention is tolerance-checked (ATOL=RTOL=2.5e-3), any stable order
  allowed; all lifecycle outputs are integers or hashes decided by bytes.
- Shape family: B 1..16 scenarios, Hq to 32, Hkv 1..4, D {64,128},
  P {8,16,32}, C {16,32,64}; empty steps, fork-only rows (0 appends),
  release+fork of different rows in one step, pad-slot garbage.
- Input guarantees (finite ranges, op preconditions, pool sufficiency) are
  stated in the header; the harness never schedules row sets whose logical
  outcome depends on row order.
- Determinism testable and tested: bit-exact replay; row-permutation
  invariance for everything except page_state_checksum (allocation
  interleaving is row-ordered by contract, as documented).
