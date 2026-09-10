// ============================================================================
// file: route_compact_reduce_common.h
// ============================================================================

#ifndef ROUTE_COMPACT_REDUCE_COMMON_H_
#define ROUTE_COMPACT_REDUCE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define RCR_ABI_VERSION 1

#define RCR_MIN_N 4096
#define RCR_MAX_N 131072
#define RCR_MIN_C 16
#define RCR_MAX_C 128
#define RCR_MIN_E 8
#define RCR_MAX_E 128
#define RCR_MIN_K 2
#define RCR_MAX_K 4

#define RCR_CHUNK_TOKENS 256

enum RcrDistributionId : int32_t {
    RCR_DIST_UNIFORM = 0,
    RCR_DIST_ZIPF_1_2 = 1,
    RCR_DIST_SINGLE_HOT = 2,
    RCR_DIST_FEW_HOT = 3,
    RCR_DIST_MANY_INVALID = 4,
    RCR_DIST_MANY_DUPLICATE = 5
};

/*
CONTRACT: route_compact_reduce_v1

Legal shape family:
  N ∈ [4096, 131072]
  C ∈ {16, 32, 64, 128}
  E ∈ {8, 16, 32, 64, 128}
  K ∈ {2, 4}

Named input distributions:
  RCR_DIST_UNIFORM:
    valid[t] usually 1; each route expert is uniform in [0, E); duplicate
    routes are uncommon.

  RCR_DIST_ZIPF_1_2:
    expert IDs follow approximately Zipf exponent 1.2, causing moderate-to-high
    expert skew.

  RCR_DIST_SINGLE_HOT:
    70%-95% of non-empty route slots go to one hot expert; remaining slots are
    uniform over the other experts.

  RCR_DIST_FEW_HOT:
    80%-95% of non-empty route slots go to h hot experts, h ∈ {2, 4, 8};
    remaining slots are uniform.

  RCR_DIST_MANY_INVALID:
    30%-75% of tokens have valid[t] == 0, and 20%-60% of route slots use
    expert == -1.

  RCR_DIST_MANY_DUPLICATE:
    50%-90% of valid tokens repeat at least one expert inside the K route slots;
    duplicate weights may combine to a positive, negative, or zero value.

Input layout:
  x:
    int16_t x[N, C], contiguous row-major.
    x[t * C + c] is the feature value for token t, channel c.

  expert:
    int16_t expert[N, K], contiguous row-major.
    expert[t * K + k] is either -1 for unused or in [0, E).
    Generated tests do not use expert < -1 or expert >= E.

  weight:
    int16_t weight[N, K], contiguous row-major.
    weight[t * K + k] is the signed route weight for that slot.

  valid:
    uint8_t valid[N].
    If valid[t] == 0, all K routes for token t are ignored.

Duplicate-expert coalescing:
  For each valid token t, scan k = 0..K-1.
  Ignore slots with expert == -1 or weight == 0.
  If the same expert appears multiple times for token t, combine those route
  weights by signed integer addition.
  The coalesced route keeps the first occurrence position for local ordering.
  If the final combined weight is zero, that token/expert route is omitted.

Outputs:
  counts[E]:
    counts[e] is the number of surviving coalesced token routes assigned to
    expert e.

  offsets[E + 1]:
    Exclusive prefix sum of counts.
    offsets[0] == 0.
    offsets[e + 1] == offsets[e] + counts[e].
    offsets[E] == total surviving coalesced routes.

  packed_token[N * K]:
    Only entries [0, offsets[E]) are defined.
    For each expert e, packed_token[offsets[e]..offsets[e+1]) contains token
    IDs assigned to expert e, strictly increasing by token ID.

  packed_weight[N * K]:
    packed_weight[i] is the coalesced signed weight corresponding to
    packed_token[i].

  sum[E, C]:
    int64_t sum[e, c] is:
      Σ over i in offsets[e]..offsets[e+1)-1:
        packed_weight[i] * x[packed_token[i], c]

    Arithmetic is modulo 2^64 and interpreted as two's-complement int64_t.
    The generated shape/value family keeps ordinary products in range, but the
    oracle and check semantics are wraparound.

  argmax_abs[E, C]:
    For each expert e and channel c:
      argmax_abs[e, c] is the token ID of the first occurrence maximizing:
        abs(int64(packed_weight[i]) * int64(x[packed_token[i], c]))

    Because packed_token is strictly increasing within an expert, "first
    occurrence" is equivalent to smallest token ID among ties.
    If counts[e] == 0, then:
      sum[e, c] == 0
      argmax_abs[e, c] == -1

ABI:
  The grader passes RcrRunSpec*, RcrInputs*, RcrOutputs* through the generic
  pipeline ABI. ProblemSpec describes maximum allocation bounds. RunSpec
  describes the current case.

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync.
  - May not call cudaMalloc/cudaFree.
  - May not perform synchronous host/device copies.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must use only the provided workspace plus persistent state allocated in
    solution_init.
*/

struct alignas(8) RcrProblemSpec {
    int32_t abi_version;
    int32_t max_N;
    int32_t max_C;
    int32_t max_E;
    int32_t max_K;
    int32_t flags;
    int32_t reserved[10];
};

struct alignas(8) RcrRunSpec {
    int32_t abi_version;
    int32_t N;
    int32_t C;
    int32_t E;
    int32_t K;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[8];
};

struct alignas(8) RcrInputs {
    const int16_t* x;
    const int16_t* expert;
    const int16_t* weight;
    const uint8_t* valid;
};

struct alignas(8) RcrOutputs {
    int32_t* counts;
    int32_t* offsets;
    int32_t* packed_token;
    int32_t* packed_weight;
    int64_t* sum;
    int32_t* argmax_abs;
};

static_assert(sizeof(RcrProblemSpec) == 64, "RcrProblemSpec layout drift");
static_assert(sizeof(RcrRunSpec) == 64, "RcrRunSpec layout drift");
static_assert(sizeof(RcrInputs) == 32, "RcrInputs layout drift");
static_assert(sizeof(RcrOutputs) == 48, "RcrOutputs layout drift");

static_assert(offsetof(RcrProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(RcrProblemSpec, max_N) == 4, "layout");
static_assert(offsetof(RcrProblemSpec, max_C) == 8, "layout");
static_assert(offsetof(RcrProblemSpec, max_E) == 12, "layout");
static_assert(offsetof(RcrProblemSpec, max_K) == 16, "layout");
static_assert(offsetof(RcrProblemSpec, flags) == 20, "layout");

static_assert(offsetof(RcrRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(RcrRunSpec, N) == 4, "layout");
static_assert(offsetof(RcrRunSpec, C) == 8, "layout");
static_assert(offsetof(RcrRunSpec, E) == 12, "layout");
static_assert(offsetof(RcrRunSpec, K) == 16, "layout");
static_assert(offsetof(RcrRunSpec, seed_id) == 20, "layout");
static_assert(offsetof(RcrRunSpec, distribution_id) == 24, "layout");
static_assert(offsetof(RcrRunSpec, case_id) == 28, "layout");

static inline size_t rcr_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int rcr_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int rcr_valid_C(int C) {
    return C == 16 || C == 32 || C == 64 || C == 128;
}

static inline int rcr_valid_E(int E) {
    return E == 8 || E == 16 || E == 32 || E == 64 || E == 128;
}

static inline int rcr_valid_K(int K) {
    return K == 2 || K == 4;
}

static inline int rcr_validate_problem_spec(const RcrProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != RCR_ABI_VERSION) return 0;
    if (spec->max_N < RCR_MIN_N || spec->max_N > RCR_MAX_N) return 0;
    if (!rcr_valid_C(spec->max_C)) return 0;
    if (!rcr_valid_E(spec->max_E)) return 0;
    if (!rcr_valid_K(spec->max_K)) return 0;
    return 1;
}

static inline int rcr_validate_run_spec(const RcrRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != RCR_ABI_VERSION) return 0;
    if (run->N < RCR_MIN_N || run->N > RCR_MAX_N) return 0;
    if (!rcr_valid_C(run->C)) return 0;
    if (!rcr_valid_E(run->E)) return 0;
    if (!rcr_valid_K(run->K)) return 0;
    if (run->distribution_id < RCR_DIST_UNIFORM ||
        run->distribution_id > RCR_DIST_MANY_DUPLICATE) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const RcrProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const RcrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const RcrRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // ROUTE_COMPACT_REDUCE_COMMON_H_
