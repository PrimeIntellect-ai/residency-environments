// common.h — det_reduce / dcrs_allreduce_stream_v2
// Extracted verbatim from §2 of design file.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define DCR_ABI_VERSION 2u
#define DCR_MAX_OPS_PER_RUN 64u
#define DCR_TILE_ITEMS 256u
#define DCR_WARP_SIZE 32u

#define DCR_MODE_SUM  0u
#define DCR_MODE_SEG  1u
#define DCR_MODE_NORM 2u

#define DCR_DTYPE_F32  0u
#define DCR_DTYPE_BF16 1u

#define DCR_OUT_F32  0u
#define DCR_OUT_BF16 1u

#define DCR_CLASS_FINITE  0u
#define DCR_CLASS_POS_INF 1u
#define DCR_CLASS_NEG_INF 2u
#define DCR_CLASS_NAN     3u

#define DCR_STATUS_OK              0u
#define DCR_STATUS_INVALID_OP      1u
#define DCR_STATUS_MODE_MISMATCH   2u
#define DCR_STATUS_OOB_INPUT       3u
#define DCR_STATUS_OOB_OUTPUT      4u
#define DCR_STATUS_UNSUPPORTED     5u

#define DCR_INVALID_F32_BITS    0x7FC0BAD1u
#define DCR_CANONICAL_NAN_F32_BITS 0x7FC00000u
#define DCR_CANONICAL_BF16_NAN  0x7FC0u
#define DCR_INVALID_BF16        0x7FC1u

// FNV-1a-64 constants (project basis, NOT canonical FNV basis)
#define DCR_FNV_BASIS 1469598103934665603ull
#define DCR_FNV_PRIME 1099511628211ull

struct DcrSpec {
    uint32_t abi_version;          // must be DCR_ABI_VERSION
    uint32_t max_accumulators;     // visible <= 16, hidden <= 128
    uint32_t max_ranks;            // visible <= 8, hidden <= 16
    uint32_t max_cells;            // vector cells or segment count
    uint32_t max_items_per_rank;   // per-rank items M
    uint32_t max_ops_per_run;      // <= DCR_MAX_OPS_PER_RUN
    uint32_t allow_bf16;           // 0 or 1
    uint32_t reserved0;

    uint64_t max_input_bytes;
    uint64_t max_output_bytes;
    uint64_t flags;                // must be zero in v2
};

struct DcrOpDesc {
    uint32_t mode;             // SUM, SEG, NORM, or invalid
    uint32_t input_dtype;      // F32 or BF16
    uint32_t output_dtype;     // F32 or BF16
    uint32_t acc_id;           // persistent accumulator slot

    uint32_t logical_ranks;    // R
    uint32_t cells;            // SUM/NORM vector length, SEG segment count
    uint32_t items_per_rank;   // M
    uint32_t segment_count;    // must equal cells for SEG, zero otherwise

    uint64_t values_offset;    // byte offset into inputs
    uint64_t keys_offset;      // byte offset into inputs, SEG only
    uint64_t tensor_offset;    // byte offset into outputs
    uint64_t state_offset;     // byte offset into outputs

    uint64_t reserved0;
    uint64_t reserved1;
};

struct DcrRunSpec {
    uint32_t abi_version;
    uint32_t op_count;

    uint64_t input_bytes;
    uint64_t output_bytes;

    uint64_t header_offset;          // normally 0
    uint64_t tensor_region_offset;
    uint64_t tensor_region_bytes;
    uint64_t timeline_region_offset;
    uint64_t timeline_region_bytes;  // op_count * sizeof(DcrTimelineRecord)
    uint64_t state_region_offset;
    uint64_t state_region_bytes;
    uint64_t counters_offset;
    uint64_t counters_bytes;

    DcrOpDesc ops[DCR_MAX_OPS_PER_RUN];
};

struct DcrOutputHeader {
    uint64_t magic;          // 0x44535253414C4C32 = "DSRSALL2"
    uint32_t abi_version;
    uint32_t run_status;

    uint64_t run_index_before;
    uint64_t run_index_after;

    uint32_t op_count;
    uint32_t reserved0;

    uint64_t tensor_bytes;
    uint64_t timeline_bytes;
    uint64_t state_bytes;
    uint64_t counters_bytes;

    uint64_t fnv_tensor;
    uint64_t fnv_timeline;
    uint64_t fnv_state;
    uint64_t fnv_counters;
    uint64_t fnv_all;
};

struct DcrTimelineRecord {
    uint64_t run_index;
    uint32_t op_index;
    uint32_t status;

    uint32_t mode;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t acc_id;

    uint32_t logical_ranks;
    uint32_t cells;
    uint32_t items_per_rank;
    uint32_t segment_count;

    uint32_t tile_items;              // always 256
    uint32_t warp_size;               // always 32
    uint32_t cross_rank_policy;       // always 1: owner-rotated order
    uint32_t shard_policy;            // always 1: contiguous balanced shards

    uint64_t values_offset;
    uint64_t keys_offset;
    uint64_t tensor_offset;
    uint64_t state_offset;
};

struct DcrStateRecord {
    uint32_t acc_id;
    uint32_t cell;
    uint32_t mode;
    uint32_t cls;

    uint32_t sum_bits;
    uint32_t comp_bits;
    uint32_t value_bits;      // finalized running value, f32 bits
    uint32_t reserved0;

    uint64_t update_count;
    uint64_t finite_count;
    uint64_t nan_count;
    uint64_t pos_inf_count;
    uint64_t neg_inf_count;
};

struct DcrCountersRecord {
    uint64_t magic;            // 0x44535253434E5452 = "DSRSCNTR"
    uint32_t abi_version;
    uint32_t reserved0;

    uint64_t run_count;
    uint64_t op_count;
    uint64_t valid_op_count;
    uint64_t invalid_op_count;

    uint64_t finite_input_count;
    uint64_t nan_input_count;
    uint64_t pos_inf_input_count;
    uint64_t neg_inf_input_count;

    uint64_t tensor_bytes_emitted;
    uint64_t state_records_emitted;

    uint64_t fnv_basis;        // 1469598103934665603
    uint64_t fnv_prime;        // 1099511628211
};
