# v2-pl-80-paged_sink_e4m3_decode — DESIGN

Vein: attention / quantized-KV / decode pipelines (stateful, perf-gated).
Validated 2026-07-04 on RTX 5080 (sm_120, CUDA 13.0), WSL2.

## What the task is

A stateful paged FP8(E4M3) KV-cache decode engine with attention sinks and a
sliding window (StreamingLLM-style), grouped-query attention, and a
deterministic page lifecycle. One `solution_run` = one step: chunked-prefill
appends (0..8 tokens per active row) with normative per-token E4M3
quantization (power-of-two scale, RNE with round-half-to-even, signed-zero
preservation, saturation at 448), lowest-free-id page allocation, dead-page
reclamation between sink and window, then flash attention over the live set
[0,sink_end) U [win_start,L) read back from the quantized pool, plus exact
per-sequence FNV-1a-64 KV hashes (non-canonical basis) and a global
page-state checksum. State persists across steps; `solution_reset` restores
the initial configuration.

Sources: StreamingLLM (sinks+window), vLLM paged KV, flash-decoding split-K;
Pro-authoring vein notes in metadata/pmpp-model-evals/v2-pro-authoring/.

## Coupled cruxes

1. **Byte-exact E4M3 quantization feeding both attention and hashes.** The
   scale is a clamped power of two derived from amax's exponent; encode is
   RNE-to-3-bit-mantissa with subnormal handling and sign-of-zero rules. Any
   off-by-one in rounding changes stored bytes, which changes kv_hash
   (exact) *and* attention outputs. The quant_adversarial scenario feeds
   exact ties, subnormal-range magnitudes, +-0, and 448-saturating values.
2. **Deterministic page lifecycle coupled to the live set.** Lowest-free-id
   allocation in strict (row, token) order, then dead-page reclamation in
   (seq, lp) order with the post-append lengths; free_pages / total_allocs /
   total_frees / page_state_checksum are exact. Reclamation interacts with
   the sink/window geometry: a page is dead only when wholly between
   sink_end and win_start. churn_tight_pool recycles ids continuously so a
   wrong allocator order shows up in the global checksum within a few steps.
3. **Fast attention over a discontiguous quantized live set.** GQA
   (Hq/Hkv up to 32/1), live positions spread across non-adjacent pages,
   K/V stored as E4M3 bytes + per-(token,head) scale exponents. The perf
   gate demands flash-decoding-style splitting with online softmax and
   vectorized byte loads plus inline E4M3 decode; a clean two-pass
   re-reading implementation is ~48x slower on the bench sequence.
4. **Checksums must be parallel.** kv_hash folds ~(S+W)*Hkv*(2D+2) bytes
   per sequence per step. The reference uses the affine-FNV chunk maps
   (FNV(chunk,h) = A*(h&~0xFF) + T[h&0xFF]) cached per physical slot with
   dirty tracking plus page-level map composition, so the serial chain is
   ~pages, not ~bytes. A serial per-byte fold dominates the step and blows
   the gate on its own.

## Reference architecture

- Admin kernel (single block): appends + lowest-free-id allocation via a
  shared-memory bitmap with a monotone cursor (frees happen strictly after
  appends within a step), dead-page scan, counter updates.
- Quantize kernel: one block per (row, token, kvh); amax by warp reduction,
  bit-twiddled E4M3 encode, coalesced byte stores; marks slots dirty.
- Token/page map kernels: rebuild 256-entry affine-FNV tables only for
  dirty slots; full in-window pages get composed page-level tables.
- Attention: flash-decoding grid (row, kv-head x group tile, split); one
  warp per query head; online softmax over a contiguous share of live
  positions with 4-token ILP prefetch, uchar4/uchar2 K/V loads, inline
  E4M3 decode, warp-shuffle reductions; merge kernel combines per-split
  (m, l, acc) partials by LSE rescaling.
- Final hash kernel: per-sequence combine walks page maps then per-token
  maps (~(S+W)/P + O(P) dependent steps); global checksum folds the small
  metadata serially (few hundred bytes only).

naive_ref.cu (authoring artifact, not shipped): single-thread admin with
linear allocator scans, per-(row,token) quantize blocks, per-(row,head)
two-pass softmax re-reading K, one thread per sequence folding every live
byte serially, single-thread global checksum fold.

## Validation evidence (real outputs, RTX 5080)

- `make test_reference && ./test_reference` -> `passed 930 / 930`
  (8 shape-family scenarios x {base, exact_replay, reversed_active_order},
  930 per-step full-output checks; scenarios: boundary_small_S4_W16_P8,
  quant_adversarial_ties_saturation, group1_nosink_S0_W64_P16,
  group32_smallwin_S16_W16_P32, churn_tight_pool_S8_W40_P8,
  full_attn_S64_W1024_P16, sink_heavy_S64_W16_P8,
  bench_like_mid_S32_W480_P16).
- `make test_naive && ./test_naive` -> `passed 930 / 930` (clean GPU impl
  agrees with the independent CPU oracle — 3-way cross-validation).
- Stub `solution.cu` compiles; `./test_student` -> `passed 0 / 930`.
- `./bench_reference` twice: `avg_ms=27.292714`, `avg_ms=27.096055`
  (avg per full 112-step sequence, B=48 Hq=16 Hkv=4 D=128 P=16 S=32 W=480
  pages=2048, warm-started, event-timed).
- `./bench_naive`: `avg_ms=1305.877551` => **ratio 48.2x** (>= 5x bar).

## Gate rationale

Per bench step the attention alone is ~200M fp32 MAC (48 rows x 16 heads x
~512 live x 128 dims x 2) plus ~25M inline E4M3 decodes; over the 112-step
timed sequence that is ~22 GFLOP + hash folding of ~340 MB of KV bytes.
A single-core host replay is O(minutes) against a 27 ms GPU budget
(>1000x), and `solution_run` may not synchronize or copy to host anyway.
Serial floors in the reference are ~pages-long dependent chains (dozens of
steps), negligible next to the arithmetic.

## Fairness notes

- Every rounding rule, order, and fold is normative in
  paged_sink_e4m3_decode_common.h (E4M3 encode incl. tie and signed-zero
  cases; append/alloc/reclaim orders; live-set definition; both checksum
  recipes with the non-canonical FNV basis spelled out).
- Shape family: B 1..64, Hq/Hkv 1..32/1..16 (any legal GQA group), D
  {64,128}, P {8,16,32}, S 0..64, W 16..1024, pool sizes down to the
  guaranteed-sufficient minimum; scenarios cover S=0 (no sink), W covering
  the whole sequence (dense attention), sink-dominated, page churn, group
  sizes 1 and 32, empty steps, new_token_count=0 rows, and L=0 (empty live
  set => y=0, lse=0).
- y/lse compared at ATOL=RTOL=2e-3 (generous for any numerically stable
  order); everything else exact. Input guarantees (finite ranges, normal
  amax, pool sufficiency) are stated in the header, so no hidden traps.
- Determinism requirements are testable and tested: exact replay bit-exact;
  active-row permutation invariance for all outputs except
  page_state_checksum (allocation interleaving is row-ordered by contract).
