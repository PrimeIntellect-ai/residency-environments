// ============================================================================
// file: moe_grouped_ffn_reroute_common.h
// ============================================================================

#ifndef MOE_GROUPED_FFN_REROUTE_COMMON_H_
#define MOE_GROUPED_FFN_REROUTE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MGF_ABI_VERSION 1

#define MGF_MIN_N 4096
#define MGF_MAX_N 32768
#define MGF_MIN_D 32
#define MGF_MAX_D 128
#define MGF_MIN_H 64
#define MGF_MAX_H 256
#define MGF_MIN_E 16
#define MGF_MAX_E 128
#define MGF_MIN_K 2
#define MGF_MAX_K 4
#define MGF_MIN_CAP 16
#define MGF_MAX_CAP 256
#define MGF_MIN_QSHIFT 4
#define MGF_MAX_QSHIFT 8

// |logit| <= MAX_D * 127 * 127 = 2064512 < 2^21, so logit + MGF_LOGIT_BIAS
// always fits in 22 unsigned bits.
#define MGF_LOGIT_BIAS (1 << 21)

// Route status codes (route_status output).
#define MGF_RS_KEPT_PRIMARY 0
#define MGF_RS_KEPT_REROUTED 1
#define MGF_RS_DROP_NO_BACKUP 2
#define MGF_RS_DROP_OVERFLOW 3

// y_checksum chunking: one digest per MGF_CSUM_ROWS consecutive token rows.
#define MGF_CSUM_ROWS 16

// FNV-1a-64 (NON-canonical basis used across this project).
#define MGF_FNV_BASIS 1469598103934665603ULL
#define MGF_FNV_PRIME 1099511628211ULL

enum MgfDistributionId : int32_t {
    MGF_DIST_UNIFORM = 0,
    MGF_DIST_HOT_EXPERT = 1,
    MGF_DIST_TIES = 2,
    MGF_DIST_ZERO_X = 3,
    MGF_DIST_SATURATE = 4
};

/*
CONTRACT: moe_grouped_ffn_reroute

One solution_run executes a full exact-integer MoE layer over a batch of
tokens: router GEMM, group-limited top-K routing, two-phase capacity dispatch
with deterministic overflow re-routing, a per-expert int8 two-layer FFN
(grouped GEMM) over every kept route, and an int64 combine. Every output is an
exact integer; there is no tolerance anywhere.

Legal shape family (RunSpec):
  N   in [4096, 32768]            (token count; any integer in range)
  D   in {32, 64, 128}            (model dim)
  H   in {64, 128, 256}           (expert hidden dim)
  E   in {16, 32, 64, 128}        (expert count)
  G   in {4, 8, 16}               (routing groups; E % G == 0, E/G >= 2)
  g_sel in {1, 2, 4}              (selected groups; g_sel <= G and
                                   g_sel * (E/G) >= K)
  K   in {2, 4}                   (route slots per token)
  cap in {16, 32, 64, 128, 256}   (per-expert TOTAL capacity, both phases)
  qshift in [4, 8]                (FFN requant shift)

Named input distributions (test/bench data generation; semantics identical
for all of them):
  MGF_DIST_UNIFORM   : x / wr / w1 / w2 approximately uniform int8.
  MGF_DIST_HOT_EXPERT: router rows biased so a few experts win most tokens,
                       forcing heavy phase-1 overflow and re-routing.
  MGF_DIST_TIES      : coarse-valued x and wr produce massive logit ties.
  MGF_DIST_ZERO_X    : many all-zero token rows (logits 0, full tie cascades).
  MGF_DIST_SATURATE  : large-magnitude weights, FFN hidden units saturate.

Inputs (all device pointers, fully initialized by the harness):
  x  : int8_t [N, D] row-major. Token features.
  wr : int8_t [E, D] row-major. Router weight rows.
  w1 : int8_t [E, H, D] row-major. w1[((e*H)+j)*D + d] is layer-1 weight of
       expert e, hidden unit j, input channel d.
  w2 : int8_t [E, D, H] row-major. w2[((e*D)+d)*H + j] is layer-2 weight of
       expert e, output channel d, hidden channel j.

STEP 1 -- router logits (exact int32):
  logit[t, e] = sum over d in [0, D) of int32(x[t,d]) * int32(wr[e,d])
  The full matrix must be written to the logits output, row-major [N, E].

STEP 2 -- group-limited candidate ranking (per token t):
  Groups are contiguous: group g owns experts [g*S, (g+1)*S), S = E/G.
  group_score[g] = max over its experts of logit[t, e].
  Selected groups: the g_sel best groups ranked by
    (group_score descending, group id ascending).
  Candidate list cand[t][0..g_sel*S) = ALL experts of selected groups ranked by
    (logit[t,e] descending, expert id ascending).

STEP 3 -- primary routes:
  Token t, slot k in [0, K): primary expert = cand[t][k] (always exists since
  g_sel*S >= K). Its gate is gate1 = logit[t, cand[t][k]] (int32, may be
  negative). The backup expert of slot k is cand[t][K + k] if K + k < g_sel*S,
  otherwise there is no backup.

STEP 4 -- phase-1 capacity dispatch:
  For each expert e, rank all primary routes assigned to e by
    (gate1 descending, token id ascending, slot id ascending).
  The first min(count, cap) routes are KEPT (status MGF_RS_KEPT_PRIMARY).
  Overflowed routes re-route to their backup expert if it exists; routes
  without a backup get status MGF_RS_DROP_NO_BACKUP.

STEP 5 -- phase-2 capacity dispatch (re-routed routes only):
  A re-routed route (t, k) targets backup expert b with gate
  gate2 = logit[t, b]. For each expert b, the remaining capacity is
  rcap[b] = cap - kept1[b] (kept1 = phase-1 kept count). Rank the phase-2
  routes of b by (gate2 descending, token id ascending, slot id ascending)
  and keep the first min(count2, rcap[b]) (status MGF_RS_KEPT_REROUTED).
  Phase-1 kept routes are NEVER displaced by phase-2 routes, even when a
  phase-2 gate is larger. Phase-2 overflow gets MGF_RS_DROP_OVERFLOW.
  There is no third phase. A token never has two routes to the same expert
  (candidate positions are distinct experts).

STEP 6 -- packing (expert-major):
  counts[e]  = kept1[e] + kept2[e]  (always <= cap)
  offsets[E+1]: exclusive prefix sum of counts; offsets[0] == 0;
    offsets[E] == total kept routes.
  The packed slice of expert e is offsets[e]..offsets[e+1]):
    FIRST all phase-1 kept routes in their phase-1 rank order,
    THEN all phase-2 kept routes in their phase-2 rank order.
  packed_token[pos] : token id (int32)
  packed_slot[pos]  : slot id (int32)
  packed_gate[pos]  : the route's final gate (int32; gate1 for phase-1,
                      gate2 for phase-2)
  packed_phase[pos] : 0 for phase-1 kept, 1 for phase-2 kept (uint8)
  Entries at positions >= offsets[E] are unspecified.

STEP 7 -- per-route status outputs:
  route_expert[t*K + k] : final expert id (int16) of the kept route, or -1
                          for both drop statuses.
  route_status[t*K + k] : MGF_RS_* code (uint8).

STEP 8 -- expert FFN (exact integers) for every kept packed position pos in
[0, offsets[E]) owned by expert e, token t:
  hraw[j] = sum over d of int32(w1[((e*H)+j)*D + d]) * int32(x[t,d])
  h[j]    = min( max(hraw[j], 0) >> qshift, 127 )      (j in [0, H))
  out[d]  = sum over j of int32(w2[((e*D)+d)*H + j]) * h[j]
  packed_y[pos*D + d] = out[d]  (int32; rows >= offsets[E] unspecified)
  All sums are exact in int32 (they cannot overflow within the legal shape
  family). ">>" is a shift of a non-negative value.

STEP 9 -- combine (int64, two's-complement wraparound):
  y[t*D + d] = sum over kept routes (t, k) of
      int64(packed_gate_of_route) * int64(out_of_route[d])
  computed modulo 2^64. Tokens with no kept routes have y[t, :] == 0.

STEP 10 -- y_checksum (uint64, two-level FNV-1a-64):
  FNV-1a-64 byte step: h = (h XOR byte) * MGF_FNV_PRIME  (mod 2^64), seeded
  with MGF_FNV_BASIS. Multi-byte values are absorbed little-endian, full
  width, no length prefix.
  Let C = ceil(N / MGF_CSUM_ROWS). For chunk c in [0, C):
    digest[c] = FNV over the raw bytes of y rows
                [c*MGF_CSUM_ROWS, min((c+1)*MGF_CSUM_ROWS, N)) in row-major
                order (token ascending, d ascending, 8 bytes little-endian
                per int64), seeded with MGF_FNV_BASIS.
  y_checksum[0] = FNV over the bytes of digest[0..C) (each uint64 absorbed
                little-endian, ascending c), seeded with MGF_FNV_BASIS.

ABI:
  The grader passes MgfRunSpec*, MgfInputs*, MgfOutputs* through the generic
  pipeline ABI. MgfProblemSpec carries maximum allocation bounds.

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync.
  - May not call cudaMalloc/cudaFree (persistent allocations belong in
    solution_init).
  - May not perform synchronous host/device copies and may not synchronize
    the stream to inspect device values on the host.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must be deterministic: identical inputs must produce identical bytes in
    every specified output region across repeated calls.
*/

struct alignas(8) MgfProblemSpec {
    int32_t abi_version;
    int32_t max_N;
    int32_t max_D;
    int32_t max_H;
    int32_t max_E;
    int32_t max_K;
    int32_t max_cap;
    int32_t flags;
    int32_t reserved[8];
};

struct alignas(8) MgfRunSpec {
    int32_t abi_version;
    int32_t N;
    int32_t D;
    int32_t H;
    int32_t E;
    int32_t G;
    int32_t g_sel;
    int32_t K;
    int32_t cap;
    int32_t qshift;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[3];
};

struct alignas(8) MgfInputs {
    const int8_t* x;
    const int8_t* wr;
    const int8_t* w1;
    const int8_t* w2;
};

struct alignas(8) MgfOutputs {
    int32_t* logits;        // [N, E]
    int32_t* counts;        // [E]
    int32_t* offsets;       // [E + 1]
    int32_t* packed_token;  // [E * cap]
    int32_t* packed_slot;   // [E * cap]
    int32_t* packed_gate;   // [E * cap]
    uint8_t* packed_phase;  // [E * cap]
    int16_t* route_expert;  // [N, K]
    uint8_t* route_status;  // [N, K]
    int32_t* packed_y;      // [E * cap, D]
    int64_t* y;             // [N, D]
    uint64_t* y_checksum;   // [1]
};

static_assert(sizeof(MgfProblemSpec) == 64, "MgfProblemSpec layout drift");
static_assert(sizeof(MgfRunSpec) == 64, "MgfRunSpec layout drift");
static_assert(sizeof(MgfInputs) == 32, "MgfInputs layout drift");
static_assert(sizeof(MgfOutputs) == 96, "MgfOutputs layout drift");

static_assert(offsetof(MgfRunSpec, N) == 4, "layout");
static_assert(offsetof(MgfRunSpec, D) == 8, "layout");
static_assert(offsetof(MgfRunSpec, H) == 12, "layout");
static_assert(offsetof(MgfRunSpec, E) == 16, "layout");
static_assert(offsetof(MgfRunSpec, G) == 20, "layout");
static_assert(offsetof(MgfRunSpec, g_sel) == 24, "layout");
static_assert(offsetof(MgfRunSpec, K) == 28, "layout");
static_assert(offsetof(MgfRunSpec, cap) == 32, "layout");
static_assert(offsetof(MgfRunSpec, qshift) == 36, "layout");

static inline size_t mgf_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int mgf_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int mgf_valid_D(int D) {
    return D == 32 || D == 64 || D == 128;
}

static inline int mgf_valid_H(int H) {
    return H == 64 || H == 128 || H == 256;
}

static inline int mgf_valid_E(int E) {
    return E == 16 || E == 32 || E == 64 || E == 128;
}

static inline int mgf_valid_G(int G) {
    return G == 4 || G == 8 || G == 16;
}

static inline int mgf_valid_gsel(int g) {
    return g == 1 || g == 2 || g == 4;
}

static inline int mgf_valid_K(int K) {
    return K == 2 || K == 4;
}

static inline int mgf_valid_cap(int cap) {
    return cap == 16 || cap == 32 || cap == 64 || cap == 128 || cap == 256;
}

static inline int mgf_validate_problem_spec(const MgfProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MGF_ABI_VERSION) return 0;
    if (spec->max_N < MGF_MIN_N || spec->max_N > MGF_MAX_N) return 0;
    if (!mgf_valid_D(spec->max_D)) return 0;
    if (!mgf_valid_H(spec->max_H)) return 0;
    if (!mgf_valid_E(spec->max_E)) return 0;
    if (!mgf_valid_K(spec->max_K)) return 0;
    if (!mgf_valid_cap(spec->max_cap)) return 0;
    return 1;
}

static inline int mgf_validate_run_spec(const MgfRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != MGF_ABI_VERSION) return 0;
    if (run->N < MGF_MIN_N || run->N > MGF_MAX_N) return 0;
    if (!mgf_valid_D(run->D)) return 0;
    if (!mgf_valid_H(run->H)) return 0;
    if (!mgf_valid_E(run->E)) return 0;
    if (!mgf_valid_G(run->G)) return 0;
    if (!mgf_valid_gsel(run->g_sel)) return 0;
    if (!mgf_valid_K(run->K)) return 0;
    if (!mgf_valid_cap(run->cap)) return 0;
    if (run->qshift < MGF_MIN_QSHIFT || run->qshift > MGF_MAX_QSHIFT) return 0;
    if (run->E % run->G != 0) return 0;
    if (run->E / run->G < 2) return 0;
    if (run->g_sel > run->G) return 0;
    if (run->g_sel * (run->E / run->G) < run->K) return 0;
    if (run->distribution_id < MGF_DIST_UNIFORM ||
        run->distribution_id > MGF_DIST_SATURATE) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MgfProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MgfProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MgfRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MOE_GROUPED_FFN_REROUTE_COMMON_H_
