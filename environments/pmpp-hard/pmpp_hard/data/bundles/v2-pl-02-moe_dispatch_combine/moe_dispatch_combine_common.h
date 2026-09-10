// ============================================================================
// file: moe_dispatch_combine_common.h
// ============================================================================

#ifndef MOE_DISPATCH_COMBINE_COMMON_H_
#define MOE_DISPATCH_COMBINE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MDC_ABI_VERSION 1

#define MDC_MIN_N 4096
#define MDC_MAX_N 131072
#define MDC_MIN_D 16
#define MDC_MAX_D 128
#define MDC_MIN_E 8
#define MDC_MAX_E 128
#define MDC_MIN_K 2
#define MDC_MAX_K 4
#define MDC_MIN_CAP 16
#define MDC_MAX_CAP 256

#define MDC_CHUNK_TOKENS 256

#define MDC_GATE_MIN (-127)
#define MDC_GATE_MAX 127
#define MDC_NUM_GATES (MDC_GATE_MAX - MDC_GATE_MIN + 1)

enum MdcDistributionId : int32_t {
    MDC_DIST_UNIFORM = 0,
    MDC_DIST_ZIPF_1_2 = 1,
    MDC_DIST_SINGLE_HOT = 2,
    MDC_DIST_FEW_HOT = 3,
    MDC_DIST_MANY_INVALID = 4,
    MDC_DIST_MANY_DUPLICATE = 5
};

/*
CONTRACT: moe_dispatch_combine

Legal shape family:
  N ∈ [4096, 131072]
  D ∈ {16, 32, 64, 128}
  E ∈ {8, 16, 32, 64, 128}
  K ∈ {2, 4}
  cap ∈ {16, 32, 64, 128, 256}

Named input distributions:
  MDC_DIST_UNIFORM:
    valid[t] usually 1. Expert IDs are approximately uniform.

  MDC_DIST_ZIPF_1_2:
    Expert IDs follow approximately Zipf exponent 1.2.

  MDC_DIST_SINGLE_HOT:
    Most routes target one hot expert, causing heavy overflow.

  MDC_DIST_FEW_HOT:
    Most routes target a small hot set of experts.

  MDC_DIST_MANY_INVALID:
    Many tokens have valid[t] == 0 and many route slots use expert == -1.

  MDC_DIST_MANY_DUPLICATE:
    Many valid tokens repeat an expert across multiple slots. Duplicate slots
    are distinct routes, not coalesced.

Input layout:
  expert:
    int16_t expert[N, K], contiguous row-major.
    expert[t * K + k] is either -1 for unused or in [0, E).

  gate:
    int16_t gate[N, K], contiguous row-major.
    Legal generated gate values are in [MDC_GATE_MIN, MDC_GATE_MAX].
    Gate values are signed. "Highest gate" means numerically largest.

  valid:
    uint8_t valid[N].
    If valid[t] == 0, all K slots for token t are ignored.

  expert_out:
    int16_t expert_out[E, D], contiguous row-major.
    expert_out[e * D + d] is the dense expert output value for expert e,
    channel d.

Route identity:
  A route is identified by token t and slot k.
  If valid[t] != 0, expert[t, k] in [0, E), and gate[t, k] is legal, the slot
  is an assigned route.
  Multiple slots from the same token to the same expert are distinct routes.

DISPATCH semantics:
  For each expert e, consider all assigned routes whose expert is e.

  Rank routes by:
    1. higher gate first,
    2. smaller token id first,
    3. smaller slot id first.

  Keep the first min(route_count[e], cap) routes.
  Drop the rest.

  counts[e]:
    number of kept routes for expert e. Always 0 <= counts[e] <= cap.

  offsets[E + 1]:
    exclusive prefix sum of counts.
    offsets[0] == 0.
    offsets[e + 1] == offsets[e] + counts[e].
    offsets[E] == total kept routes.

  packed_token[offsets[e]..offsets[e+1]):
    token ids for kept routes of expert e in the exact rank order above.

  packed_slot[offsets[e]..offsets[e+1]):
    slot ids corresponding to packed_token.

  packed_gate[offsets[e]..offsets[e+1]):
    gate weights corresponding to packed_token/packed_slot.

  dropped[N, K]:
    dropped[t * K + k] == 1 iff slot (t, k) is an assigned route but is not
    kept due to capacity.
    dropped is 0 for invalid tokens, expert == -1 slots, illegal expert slots,
    and kept routes.

COMBINE semantics:
  For each token t and channel d:
    y[t, d] =
      Σ over kept routes (t, k):
        int64(gate[t, k]) * int64(expert_out[expert[t, k], d])

  Invalid tokens have y[t, d] == 0.
  Tokens with no kept routes have y[t, d] == 0.
  Arithmetic is modulo 2^64 and interpreted as two's-complement int64_t.

Empty expert rules:
  If expert e keeps no routes:
    counts[e] == 0
    offsets[e] == offsets[e + 1]
    there is no packed slice for e.

ABI:
  The grader passes MdcRunSpec*, MdcInputs*, MdcOutputs* through the generic
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

struct alignas(8) MdcProblemSpec {
    int32_t abi_version;
    int32_t max_N;
    int32_t max_D;
    int32_t max_E;
    int32_t max_K;
    int32_t max_cap;
    int32_t flags;
    int32_t reserved[9];
};

struct alignas(8) MdcRunSpec {
    int32_t abi_version;
    int32_t N;
    int32_t D;
    int32_t E;
    int32_t K;
    int32_t cap;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[7];
};

struct alignas(8) MdcInputs {
    const int16_t* expert;
    const int16_t* gate;
    const uint8_t* valid;
    const int16_t* expert_out;
};

struct alignas(8) MdcOutputs {
    int32_t* counts;
    int32_t* offsets;
    int32_t* packed_token;
    int32_t* packed_slot;
    int16_t* packed_gate;
    uint8_t* dropped;
    int64_t* y;
};

static_assert(sizeof(MdcProblemSpec) == 64, "MdcProblemSpec layout drift");
static_assert(sizeof(MdcRunSpec) == 64, "MdcRunSpec layout drift");
static_assert(sizeof(MdcInputs) == 32, "MdcInputs layout drift");
static_assert(sizeof(MdcOutputs) == 56, "MdcOutputs layout drift");

static_assert(offsetof(MdcProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(MdcProblemSpec, max_N) == 4, "layout");
static_assert(offsetof(MdcProblemSpec, max_D) == 8, "layout");
static_assert(offsetof(MdcProblemSpec, max_E) == 12, "layout");
static_assert(offsetof(MdcProblemSpec, max_K) == 16, "layout");
static_assert(offsetof(MdcProblemSpec, max_cap) == 20, "layout");
static_assert(offsetof(MdcProblemSpec, flags) == 24, "layout");

static_assert(offsetof(MdcRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(MdcRunSpec, N) == 4, "layout");
static_assert(offsetof(MdcRunSpec, D) == 8, "layout");
static_assert(offsetof(MdcRunSpec, E) == 12, "layout");
static_assert(offsetof(MdcRunSpec, K) == 16, "layout");
static_assert(offsetof(MdcRunSpec, cap) == 20, "layout");
static_assert(offsetof(MdcRunSpec, seed_id) == 24, "layout");
static_assert(offsetof(MdcRunSpec, distribution_id) == 28, "layout");
static_assert(offsetof(MdcRunSpec, case_id) == 32, "layout");

static inline size_t mdc_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int mdc_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int mdc_valid_D(int D) {
    return D == 16 || D == 32 || D == 64 || D == 128;
}

static inline int mdc_valid_E(int E) {
    return E == 8 || E == 16 || E == 32 || E == 64 || E == 128;
}

static inline int mdc_valid_K(int K) {
    return K == 2 || K == 4;
}

static inline int mdc_valid_cap(int cap) {
    return cap == 16 || cap == 32 || cap == 64 ||
           cap == 128 || cap == 256;
}

static inline int mdc_validate_problem_spec(const MdcProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MDC_ABI_VERSION) return 0;
    if (spec->max_N < MDC_MIN_N || spec->max_N > MDC_MAX_N) return 0;
    if (!mdc_valid_D(spec->max_D)) return 0;
    if (!mdc_valid_E(spec->max_E)) return 0;
    if (!mdc_valid_K(spec->max_K)) return 0;
    if (!mdc_valid_cap(spec->max_cap)) return 0;
    return 1;
}

static inline int mdc_validate_run_spec(const MdcRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != MDC_ABI_VERSION) return 0;
    if (run->N < MDC_MIN_N || run->N > MDC_MAX_N) return 0;
    if (!mdc_valid_D(run->D)) return 0;
    if (!mdc_valid_E(run->E)) return 0;
    if (!mdc_valid_K(run->K)) return 0;
    if (!mdc_valid_cap(run->cap)) return 0;
    if (run->distribution_id < MDC_DIST_UNIFORM ||
        run->distribution_id > MDC_DIST_MANY_DUPLICATE) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MdcProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MdcProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MdcRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MOE_DISPATCH_COMBINE_COMMON_H_
