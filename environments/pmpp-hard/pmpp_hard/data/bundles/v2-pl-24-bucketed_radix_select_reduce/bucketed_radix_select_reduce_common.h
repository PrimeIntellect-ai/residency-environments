// file: bucketed_radix_select_reduce_common.h

#ifndef BUCKETED_RADIX_SELECT_REDUCE_COMMON_H_
#define BUCKETED_RADIX_SELECT_REDUCE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define BRSR_ABI_VERSION 1

#define BRSR_MIN_N 4096
#define BRSR_MAX_N (1 << 20)
#define BRSR_MAX_T 4096
#define BRSR_NUM_PASSES 4
#define BRSR_RADIX_BITS 8
#define BRSR_BUCKETS 256

enum BrsrDistributionId : int32_t {
    BRSR_DIST_UNIFORM = 0,
    BRSR_DIST_CLUSTERED = 1,
    BRSR_DIST_MANY_TIES = 2,
    BRSR_DIST_HIGH_BUCKET_HOT = 3
};

/*
CONTRACT: bucketed_radix_select_reduce

Given N items:
  key[N]   uint32
  value[N] int32

Select the global top T items by:
  1. key descending
  2. original index ascending for equal keys

T is clamped to min(T, N). Generated T values are {16, 256, 4096}.

Radix select metadata:
  The algorithm is specified as four MSB-to-LSB byte passes.

  pass 0 examines key bits [31:24] over all items.
  pass 1 examines key bits [23:16] over items matching pass-0 prefix.
  pass 2 examines key bits [15:8]  over items matching pass-0..1 prefix.
  pass 3 examines key bits [7:0]   over items matching pass-0..2 prefix.

  For each pass p:
    pass_histograms[p, bucket] stores the count of active items whose byte
    digit equals bucket.

    chosen_bucket[p] is the descending bucket containing the carried rank.

    carried_rank[p] is the 1-based rank inside chosen_bucket[p], after
    subtracting counts of all higher buckets. This becomes the rank carried
    into the next pass.

    prefix_after_pass[p] is the compact MSB prefix after selecting the bucket:
      p=0: 0x000000bb
      p=1: 0x0000bbbb
      p=2: 0x00bbbbbb
      p=3: full threshold key

  threshold_key[0] == prefix_after_pass[3].
  It is the key of topT_indices[count-1].

Selected output:
  count[0] = min(T, N)

  topT_indices[0:count] are original indices of the top count items in
  descending key order, stable by original index for equal keys.

Reductions over the selected top-T set:
  topT_sum[0]:
    int64 two's-complement wrapping sum of selected value[index].

  topT_max[0]:
    maximum int32 value over the selected set.

  topT_argmax[0]:
    original index of the first occurrence of topT_max by original index
    among selected items. Ties use smaller original index.

Rules:
  - solution_run may not call cudaMalloc/cudaFree.
  - solution_run may launch kernels and use provided workspace.
  - No CUB/Thrust.
*/

struct alignas(8) BrsrProblemSpec {
    int32_t abi_version;
    int32_t max_N;
    int32_t max_T;
    int32_t flags;
    int32_t reserved[12];
};

struct alignas(8) BrsrRunSpec {
    int32_t abi_version;
    int32_t N;
    int32_t T;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[10];
};

struct alignas(8) BrsrInputs {
    const uint32_t* key;
    const int32_t* value;
};

struct alignas(8) BrsrOutputs {
    uint32_t* threshold_key;
    int32_t* count;
    int32_t* topT_indices;
    int64_t* topT_sum;
    int32_t* topT_max;
    int32_t* topT_argmax;
    int32_t* pass_histograms;
    int32_t* chosen_bucket;
    int32_t* carried_rank;
    uint32_t* prefix_after_pass;
};

static_assert(sizeof(BrsrProblemSpec) == 64, "BrsrProblemSpec layout drift");
static_assert(sizeof(BrsrRunSpec) == 64, "BrsrRunSpec layout drift");
static_assert(sizeof(BrsrInputs) == 16, "BrsrInputs layout drift");
static_assert(sizeof(BrsrOutputs) == 80, "BrsrOutputs layout drift");

static inline size_t brsr_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int brsr_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int brsr_validate_problem_spec(const BrsrProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != BRSR_ABI_VERSION) return 0;
    if (spec->max_N < BRSR_MIN_N || spec->max_N > BRSR_MAX_N) return 0;
    if (spec->max_T < 1 || spec->max_T > BRSR_MAX_T) return 0;
    return 1;
}

static inline int brsr_validate_run_spec(const BrsrRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != BRSR_ABI_VERSION) return 0;
    if (run->N < BRSR_MIN_N || run->N > BRSR_MAX_N) return 0;
    if (run->T < 1 || run->T > BRSR_MAX_T) return 0;
    if (run->distribution_id < BRSR_DIST_UNIFORM ||
        run->distribution_id > BRSR_DIST_HIGH_BUCKET_HOT) return 0;
    return 1;
}

static inline int brsr_count_host(int N, int T) {
    return T < N ? T : N;
}

extern "C" size_t solution_workspace_bytes(const BrsrProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const BrsrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const BrsrRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // BUCKETED_RADIX_SELECT_REDUCE_COMMON_H_
