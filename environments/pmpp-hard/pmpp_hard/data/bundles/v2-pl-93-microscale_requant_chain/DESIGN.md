# v2-pl-93-microscale_requant_chain — DESIGN

## One-line
Deterministic two-stage microscaling requant chain: fp32 -> per-32 E8M0
scale + FP8 E4M3 codes (bit-level RNE, 448 saturation, subnormals, signed
zeros), then the EXACT stage-1 values -> per-32 msb-based second scale +
signed int4 codes through a non-uniform 8-level LUT with ties to the
LARGER index; plus exact int64 squared-error rows, strict saturation
counts, and word-folded FNV digests. Everything graded exactly.

## Coupled cruxes
1. **Chained exact low-bit rounding with two different tie rules.** Stage 1
   is grid-RNE on the E4M3 lattice (even-significand ties, binade carries,
   the 2^-10 underflow tie, saturation pinned to 448 with a STRICT u>448
   count that differs from "code == max"); stage 2 rounds the stage-1
   OUTPUT (as integer magnitudes M = value*2^9) onto the LUT
   {0,1/8,1/4,3/8,5/8,1,3/2,2} with ties to the larger index. The LUT
   midpoints {1/16,3/16,5/16,1/2,13/16,5/4,7/4} are all E4M3-representable
   BY DESIGN, so stage-2 exact ties occur constantly on quantized data —
   one wrong tie direction in either stage flips bytes, digests, and the
   error sums. Every decision is exactly decidable in integer arithmetic;
   fp32 shortcuts (dividing by the scale, fp comparisons) double-round and
   fail the DENORM/TIES distributions.
2. **Two chained in-block reductions + derived scales.** sexp1 comes from
   the fp32 bit pattern of amax (including the subnormal msb path and the
   -127 clamp — DENORM blocks where nonzero inputs quantize to all-zero
   codes force the sf2=0xFF sentinel path); sexp2 is the msb of the integer
   qmax over the block's E4M3 magnitudes. Both reductions and all four
   packed output regions (E4M3 bytes, 2-per-byte int4 nibbles with odd-C
   zero padding, sf1, sf2) have ragged tails.
3. **Fusion pressure.** row_err (fixed point 2^-12: (8M - p<<msb)^2 summed
   in int64), sat counts, and the per-row word-folded digest over the
   concatenated byte stream [e4m3|q4|sf1|sf2] all consume intermediate
   values that a multi-pass solution must materialize; the reference does
   the whole chain in one kernel with one read of x and nothing
   intermediate in global memory (the gate reflects that).

## Sources
- OCP MX / FP8: E4M3 format (bias 7, subnormals 2^-9 step, max 448, NaN
  code excluded by pinning saturation), 32-wide block scaling with E8M0
  shared exponents (floor_log2(amax) - emax_elem), as in v2-pl-90 but for
  the 8-bit element type and with a second requant stage.
- LUT-quantization (NF4/AWQ-style non-uniform int4 codebooks), adapted so
  every codebook boundary is exactly representable in the source format.

## Reference implementation
Single fused kernel, 128 threads / 4 rows per block (warp per row):
coalesced 32-lane x loads with a one-block register prefetch, u32 amax via
__reduce_max_sync, pure-integer E4M3 RNE (decompose mantissa/exponent,
binade-relative shift-round with even ties, carry, saturation, strict-sat
compare), integer qmax reduction + __clz msb, LUT index as a count of 7
integer boundary compares (ties-to-larger by construction), int64 error
accumulation in registers, int4 nibble pairing via shfl. Each row's four
output regions are staged in a contiguous shared buffer laid out exactly
like the digest stream; warp 0 folds 32 digest chains (4 rows x 8
round-robin substreams) from aligned u64 shared words; each warp copies
its regions out with alignment-fixed u32 body stores. x is read exactly
once; workspace unused.

## Naive baseline (naive_ref.cu)
Clean independent multi-pass decomposition: per-(row,block) scale kernels
(both stages, with materialized sexp1/msb maps), per-element quantize
kernels (materialized int M matrix and nibble matrix in workspace),
separate packing pass, atomicAdd row accumulators, digest chains re-reading
the four output regions from global. Bit-exact identical outputs (54/54).

## Validation (RTX 5080, sm_120, CUDA 13.0, 2026-07-04)
- `test_reference`: **passed 54 / 54**, two consecutive runs (9
  distributions x 6 shapes incl. ragged R/C; guard sentinels; input
  immutability). Distributions: smooth, stage-1 grid midpoints (normal +
  subnormal), near-448 saturation probes (440..511 incl. exact 448/464),
  stage-2 LUT-midpoint landings + off-by-one neighbours, all-zero blocks
  with -0.0, subnormal-amax blocks, +-0 mixtures, binade edges +-ulp,
  random finite bit patterns.
- `test_naive`: passed 54 / 54 (independent impl agreement).
- `test_student` (shipped stub): compiles, runs, **passed 0 / 54**.
- `bench_reference` (30 iters x 4 cases, 86M elements total):
  avg_ms = 0.314873 / 0.314791 / 0.314725 (stable).
- `bench_naive`: avg_ms = 8.853202 / 8.873876.
- **naive/ref ratio ≈ 28x** (gate requirement >= 5x met).

## Host-replay estimate
Inputs are ~345 MB per bench iteration; D2H alone at ~25 GB/s PCIe is
~14 ms vs 1.26 ms reference for the same four calls, before any CPU
compute (the integer oracle takes seconds for the 54-case suite). The
contract additionally forbids host<->device copies in `solution_run`.
Hopeless.

## Fairness notes
- All rounding rules are decidable from the contract text alone; the
  oracle reproduces them with bit-level integer scalar code (no libm, no
  fp arithmetic in any decision path).
- No fp accumulation anywhere in the task: no FTZ/FMA traps to trip over
  accidentally — the difficulty is concentrated in the two rounding rule
  sets, the sentinel scale paths, and the packing/stream disciplines.
- No hidden thresholds; all outputs exact; single normative algorithm.
