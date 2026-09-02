// ============================================================================
// file: onesweep_partition_reduce_common.h
// ============================================================================

#ifndef ONESWEEP_PARTITION_REDUCE_COMMON_H_
#define ONESWEEP_PARTITION_REDUCE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define OPR_ABI_VERSION 1

#define OPR_MIN_N 4096
#define OPR_MAX_N 131072
#define OPR_MIN_RADIX_BITS 4
#define OPR_MAX_RADIX_BITS 8
#define OPR_TILE_ITEMS 256
#define OPR_MAX_DIGITS 256

enum OprDistributionId : int32_t {
    OPR_DIST_UNIFORM = 0,
    OPR_DIST_SINGLE_HOT = 1,
    OPR_DIST_FEW_HOT = 2,
    OPR_DIST_ZIPF_1_2 = 3,
    OPR_DIST_NEARLY_SORTED = 4,
    OPR_DIST_REVERSE = 5
};

/*
CONTRACT: onesweep_partition_reduce

Legal shape family:
  N ∈ [4096, 131072]
  radix_bits r ∈ {4, 8}
  num_digits = 1 << r
  num_tiles = ceil(N / OPR_TILE_ITEMS)
  OPR_TILE_ITEMS = 256

Digit definition:
  digit(key) = top r bits of uint32 key:
    key >> (32 - r)

Input layout:
  key:
    uint32_t key[N]

  val:
    int32_t val[N]

Named input distributions:
  OPR_DIST_UNIFORM:
    top-r-bit digits are approximately uniform.

  OPR_DIST_SINGLE_HOT:
    70%-95% of items belong to one hot digit; the rest are distributed over
    other digits.

  OPR_DIST_FEW_HOT:
    80%-95% of items belong to a small hot digit set.

  OPR_DIST_ZIPF_1_2:
    digits follow approximately Zipf exponent 1.2.

  OPR_DIST_NEARLY_SORTED:
    input is mostly bucket-sorted by digit with local inversions/noise.

  OPR_DIST_REVERSE:
    input is mostly reverse bucket-sorted by digit.

Stage 1:
  Compute per-tile digit histograms:
    tile_count[tile, digit] =
      number of items i in that tile whose digit(key[i]) == digit

  tile_count is not an output, but it defines the required metadata.

Stage 2:
  Compute the carried per-digit exclusive look-back offsets across tiles and
  global digit metadata.

  digit_counts[d]:
    total number of input items whose digit is d.

  digit_offsets[num_digits + 1]:
    exclusive prefix sum of digit_counts.
    digit_offsets[0] == 0.
    digit_offsets[d + 1] == digit_offsets[d] + digit_counts[d].
    digit_offsets[num_digits] == N.

  tile_digit_offsets[num_tiles, num_digits]:
    global bucket-major base offset for each tile and digit:
      tile_digit_offsets[tile, d] =
        digit_offsets[d] + Σ_{tile' < tile} tile_count[tile', d]

    This metadata is graded explicitly. A solution that only produces final
    packed arrays but not the correct carried tile offsets is incorrect.

Stage 3:
  Stable scatter into bucket-major arrays:
    packed_key[N]
    packed_val[N]
    packed_src[N]

  For each digit d, the slice:
    [digit_offsets[d], digit_offsets[d + 1])

  contains exactly the input items with digit d, in stable input order.
  packed_src is the original input index.

Stage 4:
  Per-bucket segmented reduction over packed_val:

  bucket_sum[d]:
    int64 sum of packed_val over bucket d.
    Addition is modulo 2^64 and interpreted as two's-complement int64_t.

  bucket_max[d]:
    maximum int64 value in the bucket.
    If digit_counts[d] == 0, bucket_max[d] == INT64_MIN.

  bucket_argmax[d]:
    original input index of the first occurrence of bucket_max[d] within the
    stable bucket order.
    Because bucket order is stable, ties choose the smallest original input
    index among equal maximum values in that bucket.
    If digit_counts[d] == 0, bucket_argmax[d] == -1.

ABI:
  The grader passes OprRunSpec*, OprInputs*, OprOutputs* through the generic
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

struct alignas(8) OprProblemSpec {
    int32_t abi_version;
    int32_t max_N;
    int32_t max_radix_bits;
    int32_t flags;
    int32_t reserved[12];
};

struct alignas(8) OprRunSpec {
    int32_t abi_version;
    int32_t N;
    int32_t radix_bits;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[10];
};

struct alignas(8) OprInputs {
    const uint32_t* key;
    const int32_t* val;
};

struct alignas(8) OprOutputs {
    int32_t* tile_digit_offsets;
    int32_t* digit_counts;
    int32_t* digit_offsets;
    uint32_t* packed_key;
    int32_t* packed_val;
    int32_t* packed_src;
    int64_t* bucket_sum;
    int64_t* bucket_max;
    int32_t* bucket_argmax;
};

static_assert(sizeof(OprProblemSpec) == 64, "OprProblemSpec layout drift");
static_assert(sizeof(OprRunSpec) == 64, "OprRunSpec layout drift");
static_assert(sizeof(OprInputs) == 16, "OprInputs layout drift");
static_assert(sizeof(OprOutputs) == 72, "OprOutputs layout drift");

static_assert(offsetof(OprProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(OprProblemSpec, max_N) == 4, "layout");
static_assert(offsetof(OprProblemSpec, max_radix_bits) == 8, "layout");
static_assert(offsetof(OprProblemSpec, flags) == 12, "layout");

static_assert(offsetof(OprRunSpec, abi_version) == 0, "layout");
static_assert(offsetof(OprRunSpec, N) == 4, "layout");
static_assert(offsetof(OprRunSpec, radix_bits) == 8, "layout");
static_assert(offsetof(OprRunSpec, seed_id) == 12, "layout");
static_assert(offsetof(OprRunSpec, distribution_id) == 16, "layout");
static_assert(offsetof(OprRunSpec, case_id) == 20, "layout");

static inline size_t opr_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int opr_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int opr_valid_radix_bits(int r) {
    return r == 4 || r == 8;
}

static inline int opr_num_digits(int radix_bits) {
    return 1 << radix_bits;
}

static inline int opr_num_tiles(int N) {
    return opr_ceil_div_int(N, OPR_TILE_ITEMS);
}

static inline int opr_validate_problem_spec(const OprProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != OPR_ABI_VERSION) return 0;
    if (spec->max_N < OPR_MIN_N || spec->max_N > OPR_MAX_N) return 0;
    if (!opr_valid_radix_bits(spec->max_radix_bits)) return 0;
    return 1;
}

static inline int opr_validate_run_spec(const OprRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != OPR_ABI_VERSION) return 0;
    if (run->N < OPR_MIN_N || run->N > OPR_MAX_N) return 0;
    if (!opr_valid_radix_bits(run->radix_bits)) return 0;
    if (run->distribution_id < OPR_DIST_UNIFORM ||
        run->distribution_id > OPR_DIST_REVERSE) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const OprProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const OprProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const OprRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // ONESWEEP_PARTITION_REDUCE_COMMON_H_
