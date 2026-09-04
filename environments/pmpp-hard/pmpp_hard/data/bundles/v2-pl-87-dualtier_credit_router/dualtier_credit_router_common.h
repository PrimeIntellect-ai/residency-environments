// ============================================================================
// file: dualtier_credit_router_common.h
// ============================================================================

#ifndef DUALTIER_CREDIT_ROUTER_COMMON_H_
#define DUALTIER_CREDIT_ROUTER_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define DTR_ABI_VERSION 1

#define DTR_MIN_N 1024
#define DTR_MAX_N 16384
#define DTR_MAX_D 128
#define DTR_MAX_H 256
#define DTR_MAX_E 128
#define DTR_MAX_P 16
#define DTR_MAX_CCAP 64
#define DTR_MAX_BQ 128
#define DTR_MIN_QSHIFT 4
#define DTR_MAX_QSHIFT 8

// |score| <= MAX_D * 127 * 127 = 2064512 < 2^21, so score + DTR_SCORE_BIAS
// always fits in 22 unsigned bits.
#define DTR_SCORE_BIAS (1 << 21)

// Event actions (event_log / terminal outcome of each routing attempt).
#define DTR_ACT_DELIV_BACKLOG 0
#define DTR_ACT_DELIV_PRIMARY 1
#define DTR_ACT_DELIV_BACKUP 2
#define DTR_ACT_QUEUED 3
#define DTR_ACT_DROPPED 4

// FNV-1a-64 (NON-canonical basis used across this project).
#define DTR_FNV_BASIS 1469598103934665603ULL
#define DTR_FNV_PRIME 1099511628211ULL

enum DtrDistributionId : int32_t {
    DTR_DIST_UNIFORM = 0,
    DTR_DIST_HOT_NODE = 1,
    DTR_DIST_TIES = 2,
    DTR_DIST_ZERO_X = 3,
    DTR_DIST_BURSTY = 4
};

/*
CONTRACT: dualtier_credit_router

A stateful GPU-resident token router with credit-based flow control.
Persistent state: one uint32 credit counter per expert, one bounded FIFO
backlog queue per expert (each queued entry stores the token's global id AND
its full int8 feature vector), and a global token counter. Each solution_run
ingests a batch of N new tokens, routes them through a two-tier hierarchy
(P nodes, each owning S = E/P consecutive experts), admits deliveries
subject to per-expert credits, transforms every delivered token with the
DELIVERING run's expert FFN weights, and emits a byte-exact event log.
All arithmetic is exact integer; there is no tolerance anywhere.

Legal shape family (RunSpec):
  N     in [1024, 16384]           (new tokens this run; may change per run)
  D     in {64, 128}               (feature dim)
  H     in {128, 256}              (expert FFN hidden dim)
  E     in {32, 64, 128}           (expert count)
  P     in {4, 8, 16}              (nodes; E % P == 0, E/P >= 2)
  ccap  in {8, 16, 32, 64}         (per-expert credit cap)
  bq    in {32, 64, 128}           (per-expert backlog capacity)
  refill in [0, ccap]              (credits added per run; may change per run)
  qshift in [4, 8]                 (FFN requant shift; may change per run)

State lifecycle:
  solution_init : allocate persistent state for the ProblemSpec maxima.
  solution_reset: credit[e] := ccap for all e, all backlogs empty,
                  global token counter := 0.
  solution_run  : execute one routing round (see below).
  The grader calls solution_reset before changing any of D, H, E, P, ccap,
  bq; between resets those are constant while N, refill, qshift, and all
  input bytes may change every run. The grader never ingests more than 2^31
  tokens between resets (global ids always fit in int32/uint32).

Global token ids:
  The i-th token ever ingested since the last reset (counting every run's
  tokens in run order, i starting at 0) has gid == i. Within a run, new
  token t (0 <= t < N) has gid == base + t where base is the number of
  tokens ingested by all previous runs since reset.

Named input distributions (test/bench data generation only; semantics
identical for all): DTR_DIST_UNIFORM (uniform int8), DTR_DIST_HOT_NODE
(one or two nodes win most tokens -> credit exhaustion), DTR_DIST_TIES
(coarse values -> massive score ties), DTR_DIST_ZERO_X (many all-zero
token rows -> full tie cascades onto expert 0), DTR_DIST_BURSTY (hot node
rotates between runs -> backlog churn).

Inputs (device pointers, fully initialized by the harness; ALL of them,
including the weights, may hold different bytes on every run):
  x     : int8_t [N, D] row-major. New token features.
  wnode : int8_t [P, D] row-major. Tier-1 (node) router rows.
  wexp  : int8_t [E, D] row-major. Tier-2 (expert) router rows.
  w1    : int8_t [E, H, D] row-major. w1[((e*H)+j)*D + d].
  w2    : int8_t [E, D, H] row-major. w2[((e*D)+d)*H + j].

ROUND SEMANTICS (normative, exact). All scores are exact int32.

Credit refill (start of run):
  cred0[e] = min(credit[e] + refill, ccap)      for every expert e.

Tier-1 / tier-2 routing for every new token t:
  s1[t][p] = sum over d of int32(x[t,d]) * int32(wnode[p,d])
  Rank nodes by (s1 descending, node id ascending):
    pn(t) = rank-0 node, bn(t) = rank-1 node (P >= 4, so both exist).
  s2[t][e] = sum over d of int32(x[t,d]) * int32(wexp[e,d])
  The FULL s2 matrix must be written to the s2_logits output, row-major.
  Node p owns experts [p*S, (p+1)*S), S = E/P.
  pe(t) = expert of node pn(t) maximizing (s2[t][e], -e)   (s2 desc, e asc)
  be(t) = expert of node bn(t) maximizing (s2[t][e], -e)
  (pn != bn, hence pe != be.)

PHASE 0 -- backlog drain (per expert e, independently):
  Let the backlog of e hold entries in FIFO order (oldest first).
  a0[e] = min(backlog_len[e], cred0[e]). The first a0[e] entries are
  delivered (action DTR_ACT_DELIV_BACKLOG) and removed; the rest remain in
  FIFO order. c1[e] = cred0[e] - a0[e].

PHASE 1 -- primary admission:
  Every new token t attempts expert pe(t) with key s2[t][pe(t)].
  Per expert e: rank its attempts by (key descending, gid ascending).
  The first min(count, c1[e]) are delivered (DTR_ACT_DELIV_PRIMARY);
  a1[e] = that number; c2[e] = c1[e] - a1[e]. The rest go to phase 2.

PHASE 2 -- backup admission:
  Each phase-1-rejected token t attempts expert be(t) with key
  s2[t][be(t)]. Per expert e: rank by (key descending, gid ascending),
  deliver the first min(count, c2[e]) (DTR_ACT_DELIV_BACKUP);
  a2[e] = that number; c3[e] = c2[e] - a2[e]. The rest go to phase 3.

PHASE 3 -- enqueue / drop:
  Each twice-rejected token t targets its PRIMARY expert pe(t) with key
  s2[t][pe(t)]. Per expert e: rank by (key descending, gid ascending).
  free[e] = bq - (backlog_len[e] - a0[e]). Tokens with rank < free[e] are
  appended to e's backlog in rank order (DTR_ACT_QUEUED), storing their gid
  and their full x row; the rest are discarded (DTR_ACT_DROPPED).
  Phase 3 consumes no credits.

End of run: credit[e] := c3[e]; global token counter += N.

Delivered-token FFN (every delivery, all phases, using THIS run's weights
of the delivering expert e and THIS run's qshift; xv = the delivered
token's feature row -- for phase-0 deliveries the vector stored in the
backlog at enqueue time, otherwise the current x row):
  hraw[j] = sum over d of int32(w1[((e*H)+j)*D + d]) * int32(xv[d])
  h[j]    = min( max(hraw[j], 0) >> qshift, 127 )        (j in [0, H))
  out[d]  = sum over j of int32(w2[((e*D)+d)*H + j]) * h[j]
  (Exact int32; cannot overflow within the legal family.)

Outputs after EVERY run (regions not listed as unspecified are compared
byte-exactly):
  s2_logits  : int32 [N, E] full tier-2 score matrix.
  route_nodes: int32 [N], (pn(t) << 16) | bn(t).
  route_pe_be: int32 [N], (pe(t) << 16) | be(t).
  log_len    : int32 [1], L = N + sum over e of a0[e].
  event_log  : uint64 [N + E*ccap capacity], entries [0, L) specified.
      One entry per phase-0 delivery and one per new token (its terminal
      event). Order: ascending (phase, expert, rank) where rank is the
      entry's rank inside its (phase, expert) group as defined above
      (phase-3 QUEUED and DROPPED share one rank sequence). Entry word:
        gid            in bits  0..31  (uint32)
        expert         in bits 32..39
        action         in bits 40..43  (DTR_ACT_*)
        phase          in bits 44..47  (0,1,2,3; QUEUED/DROPPED use 3)
        credit_after   in bits 48..63  (uint16)
      credit_after: for a delivery of rank r in phase 0/1/2 at expert e it
      is (start-of-phase credit of e) - r - 1; for QUEUED/DROPPED it is
      c3[pe(t)].
  counts     : int32 [E], a0[e] + a1[e] + a2[e]  (always <= cred0[e]).
  offsets    : int32 [E + 1], exclusive prefix sum of counts.
  packed_gid : int32 [E*ccap capacity]; positions [0, offsets[E]) specified.
      Expert-major packing; within expert e: phase-0 deliveries in FIFO
      order, then phase-1 in rank order, then phase-2 in rank order.
  packed_out : int32 [E*ccap, D]; rows [0, offsets[E]) specified; row of a
      packed position = FFN out[0..D) of that delivered token.
  credit_out : uint32 [E], c3[e] (the post-run persistent credits).
  state_checksum : uint64 [1], three-level FNV-1a-64 over the POST-run
      state. FNV-1a-64 byte step: h = (h XOR byte) * DTR_FNV_PRIME
      (mod 2^64), seeded with DTR_FNV_BASIS; multi-byte values absorbed
      little-endian, full width.
      Level 1: for expert e and FIFO position q in [0, backlog_len[e]),
        entry_digest[e][q] = FNV over gid (4 bytes LE) then the stored x
        row (D bytes, d ascending).
      Level 2: expert_digest[e] = FNV over credit[e] (4 bytes LE),
        backlog_len[e] (4 bytes LE), then entry_digest[e][q] for q
        ascending (8 bytes LE each).
      Level 3: state_checksum = FNV over expert_digest[0..E) (8 bytes LE
        each, e ascending).
  Regions declared unspecified (event_log beyond L, packed_gid/packed_out
  beyond offsets[E]) are never inspected but must stay inside their
  capacity bounds.

ABI:
  The grader passes DtrRunSpec*, DtrInputs*, DtrOutputs* through the
  generic pipeline ABI. DtrProblemSpec carries maximum allocation bounds.

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync.
  - May use CUDA Graphs (e.g., stream capture of this run's own kernel
    sequence, cached across runs and re-captured when the RunSpec or the
    passed pointers change). This matters: a correct implementation is a
    long pipeline of small kernels, and per-launch overhead is a first-
    order cost. Fusing kernels is equally legal.
  - May not call cudaMalloc/cudaFree (persistent allocations belong in
    solution_init).
  - May not perform synchronous host/device copies and may not synchronize
    the stream to inspect device values on the host. (Tracking run counts /
    gid bases host-side from RunSpec values alone is fine.)
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must be deterministic: identical reset+run sequences must produce
    identical bytes in every specified output region.
*/

struct alignas(8) DtrProblemSpec {
    int32_t abi_version;
    int32_t max_N;
    int32_t max_D;
    int32_t max_H;
    int32_t max_E;
    int32_t max_P;
    int32_t max_ccap;
    int32_t max_bq;
    int32_t flags;
    int32_t reserved[7];
};

struct alignas(8) DtrRunSpec {
    int32_t abi_version;
    int32_t N;
    int32_t D;
    int32_t H;
    int32_t E;
    int32_t P;
    int32_t ccap;
    int32_t bq;
    int32_t refill;
    int32_t qshift;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[3];
};

struct alignas(8) DtrInputs {
    const int8_t* x;
    const int8_t* wnode;
    const int8_t* wexp;
    const int8_t* w1;
    const int8_t* w2;
    const void* reserved0;
};

struct alignas(8) DtrOutputs {
    int32_t* s2_logits;      // [N, E]
    int32_t* route_nodes;    // [N]
    int32_t* route_pe_be;    // [N]
    int32_t* log_len;        // [1]
    uint64_t* event_log;     // [N + E*ccap]
    int32_t* counts;         // [E]
    int32_t* offsets;        // [E + 1]
    int32_t* packed_gid;     // [E * ccap]
    int32_t* packed_out;     // [E * ccap, D]
    uint32_t* credit_out;    // [E]
    uint64_t* state_checksum;// [1]
    void* reserved0;
};

static_assert(sizeof(DtrProblemSpec) == 64, "DtrProblemSpec layout drift");
static_assert(sizeof(DtrRunSpec) == 64, "DtrRunSpec layout drift");
static_assert(sizeof(DtrInputs) == 48, "DtrInputs layout drift");
static_assert(sizeof(DtrOutputs) == 96, "DtrOutputs layout drift");

static_assert(offsetof(DtrRunSpec, N) == 4, "layout");
static_assert(offsetof(DtrRunSpec, D) == 8, "layout");
static_assert(offsetof(DtrRunSpec, H) == 12, "layout");
static_assert(offsetof(DtrRunSpec, E) == 16, "layout");
static_assert(offsetof(DtrRunSpec, P) == 20, "layout");
static_assert(offsetof(DtrRunSpec, ccap) == 24, "layout");
static_assert(offsetof(DtrRunSpec, bq) == 28, "layout");
static_assert(offsetof(DtrRunSpec, refill) == 32, "layout");
static_assert(offsetof(DtrRunSpec, qshift) == 36, "layout");

static inline size_t dtr_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int dtr_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int dtr_valid_D(int D) { return D == 64 || D == 128; }
static inline int dtr_valid_H(int H) { return H == 128 || H == 256; }
static inline int dtr_valid_E(int E) {
    return E == 32 || E == 64 || E == 128;
}
static inline int dtr_valid_P(int P) { return P == 4 || P == 8 || P == 16; }
static inline int dtr_valid_ccap(int c) {
    return c == 8 || c == 16 || c == 32 || c == 64;
}
static inline int dtr_valid_bq(int b) {
    return b == 32 || b == 64 || b == 128;
}

static inline int dtr_validate_problem_spec(const DtrProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != DTR_ABI_VERSION) return 0;
    if (spec->max_N < DTR_MIN_N || spec->max_N > DTR_MAX_N) return 0;
    if (!dtr_valid_D(spec->max_D)) return 0;
    if (!dtr_valid_H(spec->max_H)) return 0;
    if (!dtr_valid_E(spec->max_E)) return 0;
    if (!dtr_valid_P(spec->max_P)) return 0;
    if (!dtr_valid_ccap(spec->max_ccap)) return 0;
    if (!dtr_valid_bq(spec->max_bq)) return 0;
    return 1;
}

static inline int dtr_validate_run_spec(const DtrRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != DTR_ABI_VERSION) return 0;
    if (run->N < DTR_MIN_N || run->N > DTR_MAX_N) return 0;
    if (!dtr_valid_D(run->D)) return 0;
    if (!dtr_valid_H(run->H)) return 0;
    if (!dtr_valid_E(run->E)) return 0;
    if (!dtr_valid_P(run->P)) return 0;
    if (!dtr_valid_ccap(run->ccap)) return 0;
    if (!dtr_valid_bq(run->bq)) return 0;
    if (run->E % run->P != 0) return 0;
    if (run->E / run->P < 2) return 0;
    if (run->refill < 0 || run->refill > run->ccap) return 0;
    if (run->qshift < DTR_MIN_QSHIFT || run->qshift > DTR_MAX_QSHIFT) return 0;
    if (run->distribution_id < DTR_DIST_UNIFORM ||
        run->distribution_id > DTR_DIST_BURSTY) return 0;
    return 1;
}

static __host__ __device__ inline uint64_t dtr_make_log_word(
    uint32_t gid, int expert, int action, int phase, uint32_t credit_after) {
    return (uint64_t)gid |
           ((uint64_t)(uint32_t)expert << 32) |
           ((uint64_t)(uint32_t)action << 40) |
           ((uint64_t)(uint32_t)phase << 44) |
           ((uint64_t)(credit_after & 0xFFFFu) << 48);
}

extern "C" size_t solution_workspace_bytes(const DtrProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const DtrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const DtrRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // DUALTIER_CREDIT_ROUTER_COMMON_H_
