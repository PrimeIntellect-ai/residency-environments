# v2-pl-91-awq_actorder_repack_gemv — DESIGN

## One-line
AWQ-packed int4 weights + grouped int4 zeros + fp32 scales are repacked
GPTQ-act-order style (stable group argsort of `g_idx`, graded as an output)
into marlin-flavoured 16x64 swizzled tiles, fused with an exact 16-lane
pinned dequant-dot per output column, exact int32 column sums, and
per-column two-level word-folded FNV-1a-64 digests over the permuted
`(q - z)` byte stream plus the scale bytes. Everything graded exactly
(byte/bit/u64/int), no tolerances.

## Coupled cruxes
1. **Deterministic act-order permutation.** `perm` is the STABLE ascending
   argsort of `g_idx` (rank formula normative, ties by channel index) and is
   itself graded. It must be computed on device (no CUB/Thrust) — histogram +
   exclusive offsets + stable in-chunk scatter — and everything downstream
   (repack bytes, dot order, digest stream) consumes it, so an unstable or
   off-by-one sort corrupts all five outputs. Distributions include reversed
   block maps, heavy skew with empty groups, and all-equal `g_idx` (pure
   tie-break test).
2. **Three-layout packing discipline.** Source weights pack 8 output channels
   per u32 in AWQ weight order (`wlane`: 0,2,4,6,1,3,5,7 along n); zeros use
   AWQ zero order (`zlane`: 0,4,1,5,2,6,3,7); the destination packs 8
   PERMUTED input channels per u32 (`wlane` along j) inside 16x64 atoms with
   an h/i row split and the `((nt&7)<<3)|(nt>>3)` column swizzle (an
   involution). Ragged K (vs 16-row tiles) and ragged N (vs 8-lane words and
   64-column atoms) both occur; every padding word must be written 0.
3. **Exact fp32 numerics under fusion pressure.** The dot is a pinned
   16-lane chain over PERMUTED positions with
   `fadd_rn(acc, fmul_rn(fmul_rn((float)qz, s), x))` — no FMA contraction,
   no FTZ (subnormal scales occur), signed zeros live (`±0.0` scales/x,
   q==z blocks), then a fixed stride-8/4/2/1 tree; graded bit-exact. The
   digest re-consumes the same `qz` decode as two's-complement bytes packed
   LE into u64 words (round-robin substreams, straddle word at ragged K
   mixing qz and scale bytes), so shortcuts that get one output right and
   the other wrong are caught.

## Sources
- AWQ / GPTQ / marlin repack reality: act-order `g_idx`, per-group zeros +
  scales, both AWQ lane orders taken verbatim from
  `v2-pro-authoring/new/new-19-int4-awq-zeropoint-repack--dequant--dot.md`
  (weight_lane / zero_lane), adapted to the v2 stateful ABI with a
  from-scratch tile/swizzle destination, stable-argsort crux and exact
  fused epilogues.

## Reference implementation
- K1a/b/c: deterministic stable counting rank (256-wide chunks: shared
  histograms -> cross-chunk exclusive offsets -> stable in-chunk scatter);
  writes `perm` (output) plus `gsorted[j]`, `xp[j]` staging.
- K1d: scale transpose `sc_t[n][g]` (dot chains and digest tails walk g
  monotonically at fixed n — turns 16-way scattered lane gathers into
  single-line hits; this was worth ~25% end-to-end).
- K2: ONE fused kernel, block = 256 threads per 32-column strip
  (half atom), software-pipelined 64-row chunks: gathers for chunk c+1 and
  the perm tile for chunk c+2 are issued before the compute of chunk c
  (double-buffered q/qz tiles, triple-buffered perm tiles). Per chunk it
  unpacks wlane/zlane once into shared, then feeds: repack words (one per
  thread, swizzle guarded per-tile), two pinned dot chains per thread
  (`__fmul_rn/__fadd_rn` only) + int zsum, and 8 round-robin digest word
  chains per column (one u64 fold per chain per chunk, straddle word
  carried into the scale tail). Ends with exact-order shared-memory trees.
  qweight is read exactly once; the unpacked q matrix never touches global
  memory; workspace is only the rank scratch + transposed scales.

## Naive baseline (naive_ref.cu)
Clean independent multi-pass decomposition: literal O(K^2) rank formula;
materialized `qmat` u8[K][N], `zmat` u8[G][N], permuted `qzmat` i8[K][N];
thread-per-word repack gathering via perm; block-per-column 16-lane dot with
strided global reads; thread-per-column zsum; thread-per-(column, substream)
word digests; per-column combine. Bit-exact identical outputs (54/54).

## Validation (RTX 5080, sm_120, CUDA 13.0, 2026-07-04)
- `test_reference`: **passed 54 / 54** (9 distributions x 6 shapes incl.
  ragged K=259/1023/8191, ragged N=65/127/255/1027, G=5..512, empty groups,
  all-same groups, q==z, ±0 scales, subnormal scales, pow2, capped randbits;
  guard sentinels; input immutability on all five input buffers).
- `test_naive`: passed 54 / 54 (independent impl agreement).
- `test_student` (shipped stub): compiles, runs, **passed 0 / 54**.
- `bench_reference` (50 iters x 4 cases: K8192/N4096/G128, K8191/N2048/G512,
  K4096/N4096/G64, K2048/N1024/G32):
  avg_ms = 0.138163 / 0.137802 / 0.137775 (stable).
- `bench_naive`: avg_ms = 0.780404 / 0.791053.
- **naive/ref ratio ≈ 5.7x** (gate requirement >= 5x met).

## Host-replay estimate
The bench mix moves ~45 MB of inputs per iteration; D2H alone at ~25 GB/s
PCIe is ~1.8 ms vs 0.55 ms reference for the same four calls, before any
CPU compute (the CPU oracle needs seconds for the 54-case suite, dominated
by the K*N dot/digest loops). The contract additionally forbids
host<->device copies in `solution_run`. Hopeless.

## Fairness notes
- All rounding/ordering rules are decidable from the contract text alone;
  the oracle reproduces them with volatile-fp32 scalar code and a counting
  rank (no library sort).
- Domain guarantee: scales and x have biased exponent <= 0x9F, so no
  product or partial sum can overflow to inf; ±0 and subnormals are legal
  and adversarially generated.
- The digest word-fold and round-robin substream layout were chosen so a
  well-fused submission is not artificially serialized (each chunk of 64
  permuted rows is exactly 8 stream words, one per substream); the same
  structure is available to students.
- No hidden thresholds; all outputs exact; single normative algorithm.
