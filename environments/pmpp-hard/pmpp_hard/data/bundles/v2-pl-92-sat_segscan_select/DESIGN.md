# v2-pl-92-sat_segscan_select — DESIGN

## One-line
Exact SEGMENTED SATURATING inclusive scan over an int32 stream with packed
head-flag bits, plus a packed saturation bitmap, the ordered compaction of
all saturated positions, and per-segment last values — everything graded
exactly, and the update rule is deliberately NON-associative.

## Coupled cruxes
1. **Non-associative scan made parallel.** The normative semantics are a
   sequential loop: `p = clamp(p + v, LO, HI)` with head-flag resets.
   Saturating addition is not associative, so the textbook value-passing
   scan / lookback is simply wrong (and fails loudly on the TIGHT/SATRUN
   distributions). The hidden insight: the per-element maps
   `x -> clamp((flag ? 0 : x) + v)` live in the family
   `x -> min(max(x + a, lo), hi)`, which is CLOSED under composition
   (`(a1+a2, clamp(l1+a2, l2, h2), clamp(h1+a2, l2, h2))`, `a` in int64) —
   function composition is associative, so the scan runs over triples.
   Heads become constant maps `(0, c, c)` that absorb every prefix, which
   subsumes segmented-scan flag logic for free.
2. **Hand-rolled decoupled lookback (no CUB).** Segments spanning millions
   of elements (bench mixes have 64–1024 segments over 2^25 elements; a
   single whole-stream segment occurs in tests) rule out per-segment
   parallelism, and the perf gate rules out per-segment sequential
   fallbacks (the honest naive is >100x slower). A single-pass solution
   needs the full CUB-internals toolbox: ticket-ordered tiles, AGG/PREFIX
   state words with `__threadfence`/volatile publication (separate payload
   slots so nothing is overwritten), windowed warp-parallel spin, and the
   flag-count prefix carried through the same payloads for segment
   ordinals.
3. **Packed flag streams + ordered compaction.** Head flags arrive as
   packed little-endian bits (padding guaranteed 0); the saturation bitmap
   must be produced in the same packed format (byte-exact, padding 0), the
   saturated indices must be compacted in exact ascending order
   (sel_count exact), and seg_last[ord] requires the exact segment ordinal
   at every segment end — including ends that sit on tile boundaries.
   ALLSEG (every element a segment), ONESEG, ragged log-uniform segments,
   exact rail landings without clamping (sat is VALUE equality), and
   ±rail off-by-one probes all occur.

## Sources
- CUB DeviceScan/DeviceSelect decoupled-lookback internals (state machine,
  tile tickets, windowed lookback), re-derived from scratch — CUB itself is
  banned by the contract.
- Classic function-composition treatment of clamped/saturating parallel
  recurrences (monotone unit-slope clamp maps form a monoid).

## Reference implementation
K1: one fused kernel, 4096-element tiles (256 threads x 16, vectorized int4
loads/stores): per-thread composition of 16 element maps, shfl warp scan +
shared block scan of triples (+ flag counts), AGG publish, windowed
lookback on warp 0 (spin with `__nanosleep` backoff, `__ffs` on the ballot
of PREFIX states, serial shfl fold of at most 32 payloads in exact tile
order), PREFIX publish, then an exact sequential REPLAY of each thread's 16
elements from its exact start value: y (int4 stores), sat mask bits
(assembled 2 threads/word in shared), ordered seg_last scatter, per-tile
sat counts. K2: single-block exclusive scan of tile sat counts ->
sel offsets + sel_count. K3: per-tile ordered sel_idx emission straight
from sat_bits. v is read exactly once; workspace is ~200KB of lookback
state (no O(N) staging).

## Naive baseline (naive_ref.cu)
Clean independent multi-pass decomposition: flag word popcounts ->
single-block scan -> ordered segment starts; grid-stride over segments with
the normative sequential loop per segment (the honest direct implementation
of a non-associative recurrence); sat word popcounts -> scan -> ordered
sel_idx emission. Bit-exact identical outputs (54/54).

## Validation (RTX 5080, sm_120, CUDA 13.0, 2026-07-04)
- `test_reference`: **passed 54 / 54**, three consecutive runs (9
  distributions x 6 ragged sizes 65537..2097151; guard sentinels; input
  immutability; S from 1 to N).
- `test_naive`: passed 54 / 54 (independent impl agreement).
- `test_student` (shipped stub): compiles, runs, **passed 0 / 54**.
- `bench_reference` (30 iters x 4 cases, 117M elements total):
  avg_ms = 0.452329 / 0.461196 / 0.452526 (stable).
- `bench_naive`: avg_ms = 52.201215 / 52.265399.
- **naive/ref ratio ≈ 114x** (gate requirement >= 5x met with a wide
  margin; the gap is structural — sequential-per-segment vs single-pass).

## Host-replay estimate
Inputs are ~470MB per bench iteration (v + flags across the 4 cases); D2H
alone at ~25 GB/s PCIe is ~19 ms vs 1.8 ms reference for the same calls,
before any CPU compute. The contract additionally forbids host<->device
copies in `solution_run`. Hopeless.

## Fairness notes
- The contract is a complete sequential specification; no hidden
  thresholds, no tolerances, deterministic outputs required.
- The clamp-map composition is derivable from the contract text with
  pencil and paper; the contract explicitly warns that saturating addition
  is non-associative and that per-segment sequential fallbacks will not
  survive the perf gate, so the search is pointed at the right question.
- Domain guarantee (|v| <= 2^20, rails within ±2^24) makes all arithmetic
  exact in int32/int64 with no overflow traps.
- No FNV digest in this task: every output is already dense and exactly
  graded (y, packed bitmap, ordered compaction, per-segment values); a
  digest would add serialization without adding discrimination.
