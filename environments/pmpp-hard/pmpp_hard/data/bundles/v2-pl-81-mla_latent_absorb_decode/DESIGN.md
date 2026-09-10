# v2-pl-81-mla_latent_absorb_decode — DESIGN

Vein: attention / quantized-KV / decode pipelines (stateful, perf-gated).
Validated 2026-07-04 on RTX 5080 (sm_120, CUDA 13.0), WSL2.

## What the task is

A stateful MLA-style (multi-head latent attention, DeepSeek-V2/V3 lineage)
decode engine with an append-only compressed KV cache. Each token stores ONE
shared latent vector c_t (d_c in {128,192,256}) plus a decoupled positional
key r_t (d_r in {32,64}), int8-quantized with per-32-channel power-of-two
group scales (normative rne/clamp recipe with saturation counting). Per-head
keys/values exist only through persistent up-projections W_uk[Hq,d_h,d_c] /
W_uv[Hq,d_v,d_c] captured at solution_init (harness poisons and frees the
weight buffers after init — implementations must copy). One solution_run =
one step: chunked prefill (1..8 tokens per active row) and decode share the
entry point; every appended token emits its own causal attention output
(position p attends to [0, p] including itself). Rotation of the stored raw
r bytes by harness-provided cos/sin tables happens at scoring time, so the
byte-exact cache hash never depends on trig precision. Exact outputs:
per-sequence FNV-1a-64 cache_hash (non-canonical basis; seq_len folded
LAST), global meta_checksum, sat_count, total_tokens, seq_len.

Sources: DeepSeek-V2 MLA + weight absorption (FlashMLA), StreamingLLM-style
serving loops, Pro-authoring specs new-02 (fp8-paged-mla-decode) and
hard-v18-mla-dual-cache-decode in metadata/pmpp-model-evals/v2-pro-authoring/.

## Coupled cruxes

1. **Absorption, both directions.** score(t) = (W_uk[h]^T q)·c~_t needs the
   query pushed through W_uk ONCE per (token, head) — materializing per-token
   per-head keys is O(L·d_h·d_c) per query instead of O(d_h·d_c); the value
   path must compute z = sum_t p_t c~_t first and up-project once
   (y = W_uv z), not per attended token. Both wrong orders are numerically
   fine (tolerance covers them) but catastrophically slow — the perf gate is
   what forces the MLA insight.
2. **Byte-exact int8 group quantization feeding exact hashes.** Power-of-two
   scale from the group amax exponent, rne with ties-to-even, clamp at +-127
   with exact saturation counting (127.5 -> 128 tie SATURATES and must be
   counted; the adversarial scenario hits it, plus 2.5/3.5/0.5 integer ties,
   all-zero groups, exact powers of two, 2^-40 anchors). One rounding slip
   changes cache_hash (exact) and sat_count (exact).
3. **Incremental hashing by construction.** cache_hash folds every stored
   byte of the token stream and folds seq_len LAST, so a per-sequence
   running hash extended by only this step's appended bytes is valid; a
   from-scratch refold each step is O(L * 330 bytes) serial per sequence
   and dominates the naive time. (Parallel affine-FNV maps are an equally
   valid alternative; both require noticing the structure.)
4. **Causal chunked prefill + weight custody.** Each of up to 8 appended
   tokens per row has a distinct causal prefix ([0..p] including itself,
   post-append), pad slots must be EXACT zeros while carrying adversarial
   garbage inputs, and W_uk/W_uv/rope tables are only borrowed during init
   (the harness poisons + frees them right after) — keeping the pointers
   fails every scenario.

## Reference architecture

- Quantize kernel: block per (row, token), warp per 32-channel group; amax
  via shuffle butterflies, exponent-extracted power-of-two scale,
  __float2int_rn, ballot/popc saturation aggregation, one atomicAdd per block.
- Hash-extend kernel: one thread per active row folds only this step's
  appended bytes into the persistent running hash (<= ~2.7 KB serial per row
  per step).
- Admin kernel (single small block): seq_len updates, step/total counters,
  final per-sequence hash (running hash + 4 length bytes), meta_checksum.
- Absorb kernel: grid (token-slot, head); q staged in shared, ql[j] rows
  dotted against a W_uk transpose built once at init (float4 loads).
- Attention kernel: flash-decoding grid (query, split); 4 warps,
  warp-interleaved tokens, char4 quantized-latent loads with inline
  power-of-two dequant, fused decoupled-rope scoring via float2 cos/sin
  loads, warp-shuffle score reduction, per-warp online (m, l, z[d_c])
  recurrence held in registers, deterministic block combine, per-split
  partials.
- Merge kernel: LSE-rescaled split merge into z, then one W_uv GEMV per
  (token, head) with float4 rows; writes y and lse.

naive_ref.cu (authoring artifact, not shipped): scalar thread per token for
quantization, single-thread admin, per-sequence full refold of the entire
cache hash every step, and one scalar thread per (row, token, head) doing
absorb + two-pass softmax (recomputing scores) + per-token latent
accumulation + serial up-projection.

## Validation evidence (real outputs, RTX 5080)

- `make test_reference && ./test_reference` -> `passed 495 / 495`
  (8 shape-family scenarios x {base, exact_replay, reversed_active_order},
  165 steps per pass; scenarios: boundary_min_B2_H2,
  quant_adversarial_ties_saturation, heads32_asym_dv, rope_deep_prefill
  (L to 326), singleton_dc192_empty_steps, asym_dc256_dv64, ragged_rates_B8,
  bench_like_small).
- `make test_naive && ./test_naive` -> `passed 495 / 495` (clean GPU impl
  agrees with the independent double-precision CPU oracle — 3-way).
- Stub `solution.cu` compiles; `./test_student` -> `passed 0 / 495`.
- `./bench_reference` twice: `avg_ms=121.179988`, `avg_ms=121.629927`
  (avg per full 162-step sequence: 100 prefill x 8 tokens + 30 mixed + 32
  decode; B=32 Hq=16 d_c=256 d_r=64 d_h=128 d_v=128 msl=2048, L to 1072).
- `./bench_naive 2`: `avg_ms=5589.128662` => **ratio 46.0x** (>= 5x bar).

## Gate rationale

A prefill step scores 4096 (token, head) queries against prefixes of up to
~1000 tokens x 320 quantized dims plus a 134M-MAC absorb GEMM and 8M-MAC
up-projection; the timed sequence is ~85 GMAC of fp32 work over ~1 GB of
int8 cache traffic. Host replay of one sequence is minutes of single-core
work against a 121 ms budget (>1000x), and solution_run may not synchronize
or copy to host. Serial floors in the reference: the per-row incremental
hash fold (~2.7 KB, ~microseconds) and the tiny admin block — negligible.
The 46x naive gap is pure optimization headroom on identical semantics
(naive passes the full suite).

## Fairness notes

- Every byte-level rule is normative in mla_latent_absorb_decode_common.h:
  quantization (amax exponent scale, rne ties-to-even, clamp/sat counting),
  fold orders for both checksums (with the non-canonical FNV basis spelled
  out), append/attention ordering, pad-slot zeroing, and the weight-pointer
  lifetime at init.
- The attention math itself is tolerance-checked (ATOL=RTOL=2.5e-3) and the
  contract explicitly allows any numerically stable order, including
  non-absorbed evaluation — slow-but-correct solutions fail only the perf
  gate, never correctness.
- Shape family: B 1..32, Hq 1..32, d_c {128,192,256} (192 exercises the
  non-power-of-two path), d_r {32,64}, d_h {64,128}, d_v {64,128} with
  d_v != d_h cases, msl 64..8192; scenarios cover empty steps, single-row
  bursts, ragged per-row rates, deep rope positions, and max-heads.
- Input guarantees (finite ranges, normal group amax, no seq overflow, pads
  arbitrary-but-finite) are stated in the header; no hidden thresholds.
- Determinism is testable and tested: bit-exact replay, and FULL output
  invariance under active-row permutation (appends touch disjoint
  sequences; counters are commutative integer sums).
