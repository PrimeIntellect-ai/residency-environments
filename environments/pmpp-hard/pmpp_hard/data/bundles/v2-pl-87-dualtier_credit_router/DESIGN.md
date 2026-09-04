# v2-pl-87-dualtier_credit_router — DESIGN

Vein: MoE routing / scatter-combine / flow-controlled dispatch systems
(stateful, perf-gated). Validated 2026-07-04 on RTX 5080 (sm_120,
CUDA 13.0), WSL2.

## What the task is

A stateful GPU-resident token router with credit-based flow control.
Persistent state: per-expert credits, per-expert bounded FIFO backlogs
(each entry stores the token's global id AND its full int8 feature row),
and a global token counter. One `solution_run` = one routing round over N
new tokens: tier-1 node scoring GEMM -> top-2 nodes; tier-2 expert scoring
GEMM (full s2 matrix is an output); then a four-phase exactly-specified
admission pipeline — backlog drain (FIFO), primary admission, backup
admission, enqueue/drop — every phase per-expert ranked by
(score desc, gid asc) and bounded by credits (or backlog space). Every
delivered token is transformed by the DELIVERING run's expert FFN weights
(backlog tokens deliver under later runs' weights and qshift). Outputs:
full s2 matrix, per-token node/expert selections, a byte-exact event log
(one word per event with per-event credit_after bookkeeping), expert-major
packed deliveries with FFN outputs, post-run credits, and a three-level
FNV checksum over the post-run state (credits + backlog gids + STORED x
rows). All exact; zero tolerance.

## Coupled cruxes

1. **Exact multi-phase credit admission under tie storms.** Three
   distinct ranked admissions plus a ranked enqueue, all tie-broken by
   gid, coupled through credit residues (c1 = cred0 - a0, etc.) and
   through backlog space (free = bq - (len - a0)). TIES/ZERO_X
   distributions collapse scores so entire batches contend for one
   expert; any rank/residue error shifts the event log, the packing, the
   FFN inputs, the credits, and every later round.
2. **Cross-run state you must actually store.** Queued tokens' feature
   rows must persist (they are FFN-transformed in a LATER run under that
   run's weights/qshift and checksummed byte-exactly from state), gids
   chain across runs, credits and ring FIFOs evolve; reset semantics and
   the fresh-epoch credits are observable through the log's credit_after
   fields.
3. **Launch-graph systems engineering + heavy arithmetic.** A correct
   implementation is ~70 small kernels (two routing GEMMs up to
   16384x128x128 int8 MAC, a grouped per-expert FFN up to 8192 routes x
   2*D*H = 537M MAC, three stable segmented sorts, emissions, checksums).
   The reference needs BOTH dp4a/shared-memory tiling for the arithmetic
   AND CUDA-graph capture (explicitly permitted by the contract, which
   warns the pipeline is launch-bound) with device-resident gid/fresh
   state so the graph replays exactly. Scalar arithmetic OR per-kernel
   launching alone each cost several x.

## Reference architecture

- Tier-1: warp per token, packed-key (biased-score<<7 | 127-p) shfl-xor
  butterflies for top-2 nodes with exact tie rules.
- Tier-2 GEMM: 32 tokens x 32 experts per 256-thread block, x and wexp
  tiles staged as packed u32 in shared, dp4a, coalesced stores.
- Selection: warp per token over the owned expert slice (S <= 32), same
  packed-key argmax; builds 48-bit phase-1 keys
  (expert(7) | inverted-biased-score(22) | token(14)).
- Admission ranking: stable multi-block LSD radix sort (8-bit digits, 6
  passes): digit-major per-block histograms -> one 1024-thread scan ->
  stable scatter via __match_any_sync per-warp ranking. Sentinel-padded
  key arrays; phase-2/3 keys appended with atomics (order sorted away —
  token id is in the key, so the sort is a strict total order).
- Bookkeeping: tiny single-block kernels derive per-expert admissions,
  residues, segment starts, packing offsets, log bases, free slots, and
  update persistent credits/ring heads.
- Emission fully parallel from the sorted arrays; credit_after computed
  from phase-start credit snapshots.
- FFN: block per packed route; expert's w1 then w2 staged coalesced into
  padded shared rows (the naive's stride-D per-thread weight walks waste
  ~8x of every fetched sector), dp4a both layers, requantized hidden
  bytes packed in shared.
- Whole pipeline captured into a CUDA graph cached on (RunSpec, all
  pointers); gid base + fresh flag live in device memory and are
  finalized by the last kernel, so replay is byte-exact. Falls back to
  direct launches if capture is unavailable.
- State checksum: per-entry digests (thread each), per-expert digests,
  root — all short chains per the normative three-level recipe.

## Validation evidence (real outputs, RTX 5080)

- `make test_reference && ./test_reference` -> `passed 60 / 60`
  (60 stateful scenarios x 1-4 rounds: 5 distributions x 7 shape combos,
  2 max-shape, 15 seed-shifted, refill-to-zero starvation, mass-drop
  saturation, zero-x storm onto expert 0, tie storm, mid-scenario reset,
  qshift sweep across backlog generations, bursty hot-node churn, minimum
  N; 3 scenarios fully replayed for determinism; guards + input
  immutability checked every round; weights regenerate every round).
- `make test_naive && ./test_naive` -> `passed 60 / 60` (independent
  clean impl agrees with the independent sequential CPU oracle).
- Stub `solution.cu` compiles; `./test_student` -> `passed 0 / 60`.
- `./bench_reference 30` twice: `avg_ms=0.387354`, `avg_ms=0.387414`.
- `./bench_naive 10` twice: `avg_ms=3.346735`, `avg_ms=3.420842`
  => **ratio 8.6x** (>= 5x gate bar).
- Per-case ref vs naive (ms): router_max_uniform 0.632/5.757 (9.1x),
  router_max_hot 0.444/3.569 (8.0x), router_mid_ties 0.249/2.536 (10.2x),
  router_wide_bursty 0.421/2.619 (6.2x), router_small_churn 0.191/2.253
  (11.8x).

naive_ref.cu = clean scalar GEMMs (thread per output, byte loads), global
bitonic sorts (~105 launches each), scalar two-pass FFN, per-kernel
launches throughout. It passes the full suite, so the 8.6x gap is pure
optimization headroom, not correctness slack.

## Gate rationale

The gate deliberately couples arithmetic and systems overhead: scalar
GEMM/FFN loses ~4x (dp4a + coalesced staged weights), and per-kernel
launching of the ~70-kernel pipeline loses another ~0.3 ms/run on this
host (measured: 3.7 us per cudaLaunchKernel x 71 launches) — the contract
explicitly tells solvers that CUDA graphs / kernel fusion are legal and
load-bearing, so the crux is fair. Host replay is impossible:
`solution_run` may not perform sync host copies, and the routing +
admission + FFN of a max round is ~0.8G MAC plus three global sorts
against a 0.63 ms budget.

## Fairness notes

- No hidden thresholds: every order (node rank, expert rank, all four
  phase ranks, log order, packing order, checksum levels, log word bit
  layout) is normative in `dualtier_credit_router_common.h`; the header
  is the complete spec. The launch-bound nature of the pipeline and the
  legality of CUDA graphs are stated in the contract rules.
- Shape family: N in [1024,16384] (any integer, may change per run),
  D/H/E/P/ccap/bq enumerated with reset-scoped semantics, refill in
  [0,ccap] and qshift in [4,8] per run; five named distributions; edge
  scenarios include starvation, saturation, full tie cascades, mid-reset,
  and cross-generation backlog delivery.
- Unspecified regions (event_log beyond log_len, packed arrays beyond
  offsets[E]) are never compared; guard bytes police capacity bounds;
  inputs immutable.
- No overflow anywhere in the legal family: |scores| <= 2,064,512
  (22 bits biased); FFN layer-2 |acc| <= 256*127*127; credits <= 64 fit
  the log's uint16 credit_after field; gids < 2^31 by grader guarantee.
