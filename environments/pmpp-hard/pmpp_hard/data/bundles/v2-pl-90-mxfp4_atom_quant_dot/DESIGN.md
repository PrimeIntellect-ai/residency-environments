# v2-pl-90-mxfp4_atom_quant_dot — DESIGN

## One-line
Exact MXFP4 (E2M1 payload + per-32 E8M0 scales, OCP-MX style) quantization of an
fp32 matrix into CUTLASS-style swizzled payload/scale atoms, with a fused
pinned-order fp32 dequant-dot epilogue, per-row saturation counts and per-row
two-level FNV-1a-64 digests. Everything graded exactly (byte/bit/u64), no
tolerances.

## Coupled cruxes
1. **Exact low-bit numerics.** E8M0 scale derivation is pinned at the bit level
   (`floor_log2` from the fp32 bit pattern including the subnormal `msb_index`
   path, `sexp = max(e-2, -127)`), and E2M1 rounding is a normative
   round-to-nearest / ties-to-even-code threshold table (`0.25,0.75,...,5.0`
   scaled by `2^sexp`, all exactly representable). Adversarial distributions
   hit every tie midpoint, saturation boundary (`> 6*2^sexp` strict), all-zero
   blocks (sf byte 0x00 + sign-preserving `-0.0` nibbles), subnormal amax
   (clamp to -127), power-of-two binade edges (`2^t`, `2^t ± ulp`), and random
   finite bit patterns. One wrong compare direction, tie rule, clamp, or
   denormal path flips bytes/digests.
2. **Swizzled atom layout discipline.** 128x64 payload atoms
   (`payload_off(rr,u) = (u/8)*1024 + (rr%32)*32 + (rr/32)*8 + (u%8)`, nibble
   pairing (u, u+32)) and 512-byte scale atoms — while the digest stream uses
   the *linear* logical packing with (2j, 2j+1) nibble pairing and a zero high
   nibble for odd C. Every padding byte must be written 0x00 (guard bytes are
   0xA5, so lazy padding fails). Ragged tails in the 32/64/128-element and
   128-row directions all occur.
3. **Pinned deterministic reduction + fused epilogue.** `row_dot` uses 128
   pinned partial lanes with fmul_rn/fadd_rn (no FMA, no FTZ) and a fixed
   binary tree; graded bit-exact. `row_digest` is a two-level non-canonical
   FNV-1a-64 (8 substreams then little-endian full-width fold of the 8
   sub-digests). Getting all of this *and* speed requires fusing: quantize,
   pack, scatter, dot and digest in one pass without materializing codes or
   the dequant matrix in global memory.

## Sources
- OCP Microscaling (MX) spec: 32-wide blocks, E8M0 scales, FP4 E2M1 elements.
- CUTLASS SM120 block-scaled layouts / idea-bank specs
  `hard-v18-nvfp4-payload-atom-repack.md` (atom offset formulas),
  `new-16-mxfp4-decode...` (E2M1 decode table), adapted to the v2 stateful ABI
  with quantization (not just repack) and an exact fused epilogue.

## Reference implementation
Single fused kernel, 128 threads / 4 rows per block (warp per row):
coalesced 32-lane block loads batched 4 blocks per iteration for MLP,
`__reduce_max_sync` u32 amax (positive-fp32/u32 order isomorphism),
scaled-domain compares (`y = |x| * 2^-sexp` exact wherever a threshold
decision can be affected — proof in code comment), dot partials held as 4
registers/lane mapping exactly onto the pinned 128-lane order, tree collapsed
to in-register + shfl butterfly in contract order, atoms written as aligned
u64/u32 words straight from shared logical bytes, digests as 32 parallel FNV
chains (4 rows x 8 substreams) on warp 0. Grid covers pad rows, so padding is
written on the same path (no pre-memset). x is read exactly once; no
workspace.

## Naive baseline (naive_ref.cu)
Clean independent multi-pass decomposition: memset padding; per-(row,block)
scale kernel; per-element quantize into materialized codes; per-byte logical
pack; per-atom-byte scatter; per-element dequant materialized fp32; block-per-
row pinned-tree dot from materialized dq; per-(row,substream) sub-digests;
per-row combine. Bit-exact identical outputs (validated 54/54).

## Validation (RTX 5080, sm_120, CUDA 13.0, 2026-07-02)
- `test_reference`: **passed 54 / 54** (9 distributions x 6 shapes incl.
  ragged R/C; guard sentinels; input immutability).
- `test_naive`: passed 54 / 54 (independent impl agreement).
- `test_student` (shipped stub): compiles, runs, **passed 0 / 54**.
- `bench_reference` (30 iters x 4 cases, 94.3M elems total):
  avg_ms = 0.142819 / 0.142868 / 0.142950 (stable).
- `bench_naive`: avg_ms = 0.876623 / 0.865681.
- **naive/ref ratio ≈ 6.1x** (gate requirement >= 5x met).

## Host-replay estimate
Inputs average ~94 MB per call; D2H alone at ~25 GB/s PCIe is ~3.8 ms vs
0.143 ms reference (~27x), before any CPU compute (CPU oracle takes ~1 s for
the 54 test cases). Contract additionally forbids host<->device copies in
`solution_run`. Hopeless.

## Fairness notes
- Domain guarantee added after calibration: RANDBITS exponent field capped at
  0xF0 so stage-4 products/partial sums can never overflow to inf (NaN bit
  patterns are not portable); guarantee stated in the contract.
- All rounding rules are decidable from the contract text alone; the oracle
  reproduces them with bit-level integer scalar code (no libm scaling).
- No hidden thresholds; all outputs exact; single normative algorithm.
