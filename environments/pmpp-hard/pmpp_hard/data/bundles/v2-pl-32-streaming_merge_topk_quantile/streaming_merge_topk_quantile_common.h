// file: streaming_merge_topk_quantile_common.h

#ifndef STREAMING_MERGE_TOPK_QUANTILE_COMMON_H_
#define STREAMING_MERGE_TOPK_QUANTILE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define SMTQ_ABI_VERSION 1

#define SMTQ_MIN_GROUPS 1
#define SMTQ_MAX_GROUPS 1024
#define SMTQ_MIN_K 1
#define SMTQ_MAX_K 64
#define SMTQ_MIN_BINS 2
#define SMTQ_MAX_BINS 4096
#define SMTQ_MAX_BATCH 4096
#define SMTQ_MAX_STEPS 64

/*
CONTRACT: streaming_merge_topk_quantile

Stateful exact-int streaming top-K + fixed-bin quantile.

Persistent state per group:
  top-K items by:
    1. key descending
    2. insertion_order ascending for equal keys

  histogram[num_bins] over all valid ingested items for that group
  group_total count over all valid ingested items for that group

Global state:
  total_ingested: int64 count of valid events processed so far.
  insertion_order for a valid event is total_ingested before increment.

Inputs per step:
  group[batch_size] int32
  key[batch_size]   int32
  value[batch_size] int32

Valid event:
  0 <= group < G. Invalid events are ignored and do not increment
  total_ingested.

Per valid event:
  - update fixed-bin histogram for its group
  - increment group_total
  - merge into that group's top-K

Histogram binning:
  key range is [key_min, key_max], inclusive.
  bin_width = ceil((key_max - key_min + 1) / num_bins).
  key <= key_min maps to bin 0.
  key >= key_max maps to bin num_bins - 1.
  otherwise bin = (key - key_min) / bin_width, clamped.

Quantile query:
  If run.is_query != 0:
    for each group g with group_total[g] = n:
      if n == 0: quantile_key[g] = INT32_MIN
      else:
        q_num/q_den is clamped into [0,1].
        rank = ceil(q_num * n / q_den), clamped into [1,n].
        quantile_key[g] is the LOWER BOUNDARY of the first histogram bin
        whose cumulative count >= rank.
  If run.is_query == 0:
    quantile_key[g] = INT32_MIN for every group.

Outputs after every step:
  topk_keys[G,K]:
    sorted by key desc, insertion_order asc. Empty slots are INT32_MIN.

  topk_values[G,K]:
    values aligned with topk_keys. Empty slots are 0.

  topk_count[G]:
    current number of kept top-K items.

  topk_value_sum[G]:
    int64 two's-complement wrapping sum of values currently in top-K.

  histogram_checksum[0]:
    FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
      the agent MUST use this exact basis or every checksum fails):
        offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
          the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
        prime        = 1099511628211  (0x100000001B3).
        fold: start h = offset basis; absorb each field's raw bytes little-endian at
          its full declared width (u8=1, u32=4, u64=8, i64=8 two's-complement); per
          byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator / final mix.

    FNV-1a over G, num_bins, then every histogram count in group-major order.

  quantile_key[G]:
    per-group query answer, or INT32_MIN for non-query steps.

  total_ingested[0]:
    total number of valid events processed so far.

  state_checksum[0]:
    FNV-1a over G,K,num_bins,key_min,key_max,total_ingested, then per group:
      group_total, topk_count,
      all topk_key/topk_value/topk_order slots,
      all histogram counts.

Rules:
  - solution_init may allocate persistent state.
  - solution_run may not call cudaMalloc/cudaFree.
  - solution_run may launch kernels and use provided workspace.
  - All graded outputs are exact.
*/

struct alignas(8) SmtqProblemSpec {
    int32_t abi_version;
    int32_t G;
    int32_t K;
    int32_t num_bins;
    int32_t key_min;
    int32_t key_max;
    int32_t max_batch;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[7];
};

struct alignas(8) SmtqRunSpec {
    int32_t abi_version;
    int32_t batch_size;
    int32_t is_query;
    int32_t q_num;
    int32_t q_den;
    int32_t step_id;
    int32_t reserved[10];
};

struct alignas(8) SmtqInputs {
    const int32_t* group;
    const int32_t* key;
    const int32_t* value;
};

struct alignas(8) SmtqOutputs {
    int32_t* topk_keys;
    int32_t* topk_values;
    int32_t* topk_count;
    int64_t* topk_value_sum;
    uint64_t* histogram_checksum;
    int32_t* quantile_key;
    int64_t* total_ingested;
    uint64_t* state_checksum;
};

static_assert(sizeof(SmtqProblemSpec) == 64, "SmtqProblemSpec layout drift");
static_assert(sizeof(SmtqRunSpec) == 64, "SmtqRunSpec layout drift");
static_assert(sizeof(SmtqInputs) == 24, "SmtqInputs layout drift");
static_assert(sizeof(SmtqOutputs) == 64, "SmtqOutputs layout drift");

static inline size_t smtq_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int smtq_validate_problem_spec(const SmtqProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != SMTQ_ABI_VERSION) return 0;
    if (spec->G < SMTQ_MIN_GROUPS || spec->G > SMTQ_MAX_GROUPS) return 0;
    if (spec->K < SMTQ_MIN_K || spec->K > SMTQ_MAX_K) return 0;
    if (spec->num_bins < SMTQ_MIN_BINS || spec->num_bins > SMTQ_MAX_BINS) return 0;
    if (spec->key_max <= spec->key_min) return 0;
    if (spec->max_batch < 0 || spec->max_batch > SMTQ_MAX_BATCH) return 0;
    if (spec->max_steps < 1 || spec->max_steps > SMTQ_MAX_STEPS) return 0;
    return 1;
}

static inline int smtq_validate_run_spec(const SmtqRunSpec* run, const SmtqProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != SMTQ_ABI_VERSION) return 0;
    if (run->batch_size < 0 || run->batch_size > spec->max_batch) return 0;
    if (run->q_den <= 0) return 0;
    if (run->q_num < 0) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const SmtqProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const SmtqProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const SmtqRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // STREAMING_MERGE_TOPK_QUANTILE_COMMON_H_
