// ============================================================================
// file: segmented_sort_topm_common.h
// ============================================================================

#ifndef SEGMENTED_SORT_TOPM_COMMON_H_
#define SEGMENTED_SORT_TOPM_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define SST_ABI_VERSION 1

#define SST_MIN_N 4096
#define SST_MAX_N 131072
#define SST_MIN_S 32
#define SST_MAX_S 4096
#define SST_MAX_M 64

enum SstDistributionId : int32_t {
    SST_DIST_UNIFORM = 0,
    SST_DIST_POWERLAW = 1,
    SST_DIST_ALL_SIZE_ONE = 2,
    SST_DIST_ONE_GIANT = 3,
    SST_DIST_MANY_TIES = 4
};

/*
CONTRACT: segmented_sort_topm

Exact-integer segmented stable sort + top-M + segmented reduce.

Shape family:
  total N ∈ [4096, 131072]
  S ∈ {32, 256, 4096}
  M ∈ {4, 16, 64}

Inputs:
  seg_offsets[S + 1]:
    Segment offsets into item arrays.
    seg_offsets[0] == 0.
    seg_offsets[S] == N.
    Segment s contains items [seg_offsets[s], seg_offsets[s + 1]).

  item_key[N]:
    int32 sort key.

  item_value[N]:
    int32 payload value.

Per-segment order:
  Items are sorted within each segment by:
    1. key descending,
    2. original index within the segment ascending.

  original-within-segment index is:
    origidx = global_item_index - seg_offsets[segment]

Top-M:
  topm_count[s] = min(M, segment_length).

  The kept rows for each segment are the first topm_count[s] items in the
  stable sorted order.

Compacted output layout:
  topm_offsets[S + 1] is the exclusive prefix of topm_count.
  The packed top-M arrays are segment-major:
    segment s occupies [topm_offsets[s], topm_offsets[s + 1]).

Outputs:
  topm_count[S]:
    Exact kept count per segment.

  topm_offsets[S + 1]:
    Exclusive prefix of topm_count.
    topm_offsets[0] == 0.
    topm_offsets[S] == total kept rows.

  packed_topm_key[N]:
    Kept item keys in segment-major stable sorted order.
    Only entries [0, topm_offsets[S]) are defined/graded.

  packed_topm_value[N]:
    Kept item values in the same order.

  packed_topm_origidx[N]:
    Kept original-within-segment indices in the same order.

  seg_sum[S]:
    int64 two's-complement wrapping sum of the kept top-M values.

  seg_max[S]:
    int32 maximum value among the kept top-M values.
    Empty segment rule: INT32_MIN.

  seg_argmax[S]:
    original-within-segment index of the first maximum value among the kept
    top-M values. Ties use smaller original-within-segment index.
    Empty segment rule: -1.

Empty segment rules:
  topm_count[s] == 0
  topm_offsets[s] == topm_offsets[s + 1]
  seg_sum[s] == 0
  seg_max[s] == INT32_MIN
  seg_argmax[s] == -1

M > segment length:
  Keep the entire segment, sorted by the same key-desc/origidx rule.

Rules for solution_run:
  - May launch kernels and use cudaMemsetAsync / device-to-device copies.
  - May not call cudaMalloc/cudaFree.
  - May not perform synchronous host/device copies.
  - May not use CUB, Thrust, cuBLAS, cuDNN, or CUTLASS.
  - Must use only the provided workspace plus persistent state allocated in
    solution_init.
*/

struct alignas(8) SstProblemSpec {
    int32_t abi_version;
    int32_t max_N;
    int32_t max_S;
    int32_t flags;
    int32_t reserved[12];
};

struct alignas(8) SstRunSpec {
    int32_t abi_version;
    int32_t N;
    int32_t S;
    int32_t M;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[9];
};

struct alignas(8) SstInputs {
    const int32_t* seg_offsets;
    const int32_t* item_key;
    const int32_t* item_value;
};

struct alignas(8) SstOutputs {
    int32_t* topm_count;
    int32_t* topm_offsets;
    int32_t* packed_topm_key;
    int32_t* packed_topm_value;
    int32_t* packed_topm_origidx;
    int64_t* seg_sum;
    int32_t* seg_max;
    int32_t* seg_argmax;
};

static_assert(sizeof(SstProblemSpec) == 64, "SstProblemSpec layout drift");
static_assert(sizeof(SstRunSpec) == 64, "SstRunSpec layout drift");
static_assert(sizeof(SstInputs) == 24, "SstInputs layout drift");
static_assert(sizeof(SstOutputs) == 64, "SstOutputs layout drift");

static_assert(offsetof(SstProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(SstProblemSpec, max_N) == 4, "layout");
static_assert(offsetof(SstProblemSpec, max_S) == 8, "layout");
static_assert(offsetof(SstProblemSpec, flags) == 12, "layout");

static_assert(offsetof(SstRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(SstRunSpec, N) == 4, "layout");
static_assert(offsetof(SstRunSpec, S) == 8, "layout");
static_assert(offsetof(SstRunSpec, M) == 12, "layout");
static_assert(offsetof(SstRunSpec, seed_id) == 16, "layout");
static_assert(offsetof(SstRunSpec, distribution_id) == 20, "layout");
static_assert(offsetof(SstRunSpec, case_id) == 24, "layout");

static inline size_t sst_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int sst_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int sst_valid_S(int S) {
    return S == 32 || S == 256 || S == 4096;
}

static inline int sst_valid_M(int M) {
    return M == 4 || M == 16 || M == 64;
}

static inline int sst_validate_problem_spec(const SstProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != SST_ABI_VERSION) return 0;
    if (spec->max_N < SST_MIN_N || spec->max_N > SST_MAX_N) return 0;
    if (!sst_valid_S(spec->max_S)) return 0;
    return 1;
}

static inline int sst_validate_run_spec(const SstRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != SST_ABI_VERSION) return 0;
    if (run->N < SST_MIN_N || run->N > SST_MAX_N) return 0;
    if (!sst_valid_S(run->S)) return 0;
    if (!sst_valid_M(run->M)) return 0;
    if (run->distribution_id < SST_DIST_UNIFORM ||
        run->distribution_id > SST_DIST_MANY_TIES) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const SstProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const SstProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const SstRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // SEGMENTED_SORT_TOPM_COMMON_H_
