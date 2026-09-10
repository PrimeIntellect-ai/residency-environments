// file: segmented_groupby_topk_common.h

#ifndef SEGMENTED_GROUPBY_TOPK_COMMON_H_
#define SEGMENTED_GROUPBY_TOPK_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define SGTK_ABI_VERSION 1

#define SGTK_MIN_N 4096
#define SGTK_MAX_N 131072
#define SGTK_MIN_G 8
#define SGTK_MAX_G 1024
#define SGTK_MAX_M 64
#define SGTK_BLOCK_THREADS 256

enum SgtDistId : int32_t {
    SGTK_DIST_UNIFORM = 0,
    SGTK_DIST_ZIPF_HOT = 1,
    SGTK_DIST_MANY_TIES = 2,
    SGTK_DIST_SINGLE_HOT = 3
};

/*
CONTRACT: segmented_groupby_topk

Input rows:
  group_id[N] int32
  key[N]      int32
  value[N]    int32

Filter:
  keep row i iff value[i] > 0 and 0 <= group_id[i] < G.

Per group:
  group_counts[g] = number of filtered rows in group g.

  Select TOP-M filtered rows by:
    1. key descending
    2. original row index ascending for equal keys

  kept_count[g] = min(M, group_counts[g])

Compacted output:
  group_offsets[G+1] is the exclusive prefix of kept_count.
  group_offsets[0] == 0.
  group_offsets[G] == total kept rows.
  packed_topk_origidx[group_offsets[g] : group_offsets[g+1]]
    contains original row indices for group g in TOP-M order.

Per-group reductions over kept TOP-M rows:
  per_group_sum[g]    int64 two's-complement wrapping sum of value[idx]
  per_group_max[g]    maximum value[idx], or INT32_MIN if kept_count[g] == 0
  per_group_argmax[g] original row index of first max value among kept TOP-M,
                      tie broken by smaller original index; -1 if empty.

Shape family:
  N in [4096, 131072]
  G in {8, 64, 1024}
  M in {4, 16, 64}
  distributions: uniform, zipf-hot, single-hot, many key ties.

Rules:
  - solution_run may not call cudaMalloc/cudaFree.
  - solution_run may use provided workspace.
  - No CUB/Thrust.
  - All graded outputs are exact.
*/

struct alignas(8) SgtProblemSpec {
    int32_t abi_version;
    int32_t max_N;
    int32_t max_G;
    int32_t flags;
    int32_t reserved[12];
};

struct alignas(8) SgtRunSpec {
    int32_t abi_version;
    int32_t N;
    int32_t G;
    int32_t M;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[9];
};

struct alignas(8) SgtInputs {
    const int32_t* group_id;
    const int32_t* key;
    const int32_t* value;
};

struct alignas(8) SgtOutputs {
    int32_t* group_counts;
    int32_t* group_offsets;
    int32_t* packed_topk_origidx;
    int64_t* per_group_sum;
    int32_t* per_group_max;
    int32_t* per_group_argmax;
    int32_t* kept_count;
};

static_assert(sizeof(SgtProblemSpec) == 64, "SgtProblemSpec layout drift");
static_assert(sizeof(SgtRunSpec) == 64, "SgtRunSpec layout drift");
static_assert(sizeof(SgtInputs) == 24, "SgtInputs layout drift");
static_assert(sizeof(SgtOutputs) == 56, "SgtOutputs layout drift");

static inline size_t sgtk_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int sgtk_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int sgtk_next_pow2_int(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

static inline int sgtk_valid_G(int G) {
    return G == 8 || G == 64 || G == 1024;
}

static inline int sgtk_valid_M(int M) {
    return M == 4 || M == 16 || M == 64;
}

static inline int sgtk_validate_problem_spec(const SgtProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != SGTK_ABI_VERSION) return 0;
    if (spec->max_N < SGTK_MIN_N || spec->max_N > SGTK_MAX_N) return 0;
    if (!sgtk_valid_G(spec->max_G)) return 0;
    return 1;
}

static inline int sgtk_validate_run_spec(const SgtRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != SGTK_ABI_VERSION) return 0;
    if (run->N < SGTK_MIN_N || run->N > SGTK_MAX_N) return 0;
    if (!sgtk_valid_G(run->G)) return 0;
    if (!sgtk_valid_M(run->M)) return 0;
    if (run->distribution_id < SGTK_DIST_UNIFORM ||
        run->distribution_id > SGTK_DIST_SINGLE_HOT) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const SgtProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const SgtProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const SgtRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // SEGMENTED_GROUPBY_TOPK_COMMON_H_
