// file: common.h
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Stateful PMPP-Pipeline v2 ABI
// -----------------------------------------------------------------------------
struct Spec;
struct RunSpec;

size_t solution_workspace_bytes(const Spec* spec);

cudaError_t solution_init(
    const Spec* spec,
    void** state,
    cudaStream_t stream);

cudaError_t solution_run(
    void* state,
    const RunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

void solution_destroy(void* state);

#ifdef __cplusplus
}
#endif

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
static constexpr uint64_t LNRBWD_MAGIC = 0x4C4E524257443201ull; // "LNRBWD2\1"
static constexpr uint32_t LNRBWD_VERSION = 1;

static constexpr uint32_t LNR_ROWS_PER_PARTIAL = 8;
static constexpr uint32_t LNR_SEGMENT_COLS = 256;
static constexpr uint32_t LNR_MAX_SEGMENTS = 128;       // hidden_size <= 32768
static constexpr uint32_t LNR_STAGE2_LEAVES = 1024;     // max partials per run/op reduction

static constexpr uint64_t LNR_FNV_BASIS = 1469598103934665603ull; // 0x14650FB0739D0383
static constexpr uint64_t LNR_FNV_PRIME = 1099511628211ull;       // 0x00000100000001B3

// -----------------------------------------------------------------------------
// Dtypes
// -----------------------------------------------------------------------------
enum LnrStorageDType : uint32_t {
    LNR_DTYPE_BF16 = 1, // x, dy, gamma, dx are raw IEEE bf16 uint16 payloads
    LNR_DTYPE_F32  = 2  // x, dy, gamma, dx are raw IEEE float32 payloads
};

// -----------------------------------------------------------------------------
// Operation kinds
// -----------------------------------------------------------------------------
enum LnrOpKind : uint8_t {
    LNR_OP_SAVE_LN   = 1, // compute and cache LN mean+rstd from x
    LNR_OP_SAVE_RMS  = 2, // compute and cache RMS rstd from x; cached mean := +0.0f
    LNR_OP_BWD_LN    = 3, // use cached LN mean+rstd, compute dx and accumulate dgamma/dbeta
    LNR_OP_BWD_RMS   = 4, // use cached RMS rstd, compute dx and accumulate dgamma; dbeta += 0
    LNR_OP_FLUSH     = 5  // emit accumulated dgamma/dbeta for one param_id and zero that accumulator
};

enum LnrCacheKind : uint32_t {
    LNR_CACHE_INVALID = 0,
    LNR_CACHE_LN      = 1,
    LNR_CACHE_RMS     = 2
};

enum LnrStatus : uint8_t {
    LNR_STATUS_OK               = 0,
    LNR_STATUS_INVALID_RANGE    = 1,
    LNR_STATUS_INVALID_PARAM    = 2,
    LNR_STATUS_CACHE_MISS       = 3,
    LNR_STATUS_PARTIAL_OVERFLOW = 4,
    LNR_STATUS_UNSUPPORTED_OP   = 5,
    LNR_STATUS_OUTPUT_RANGE     = 6
};

// -----------------------------------------------------------------------------
// Spec: fixed for one solver state.
// -----------------------------------------------------------------------------
#pragma pack(push, 1)

struct Spec {
    uint64_t magic;                    // must be LNRBWD_MAGIC
    uint32_t version;                  // must be LNRBWD_VERSION

    uint32_t hidden_size;              // N, 1 <= N <= 32768
    uint32_t param_count;              // 1 <= param_count <= 16
    uint32_t max_cache_rows;           // 1 <= max_cache_rows <= 65536

    uint32_t max_input_rows_per_run;   // x/dy row capacity visible to a run
    uint32_t max_dx_rows_per_run;      // dx logical output rows visible to a run
    uint32_t max_backward_rows_per_run;// sum of rows over valid backward ops, <= 8192
    uint32_t max_ops_per_run;          // <= 512
    uint32_t max_flush_records_per_run;// <= 512

    uint32_t storage_dtype;            // LNR_DTYPE_BF16 or LNR_DTYPE_F32
    float eps_ln;                      // finite, >= 0
    float eps_rms;                     // finite, >= 0

    uint64_t counter_seed;             // all uint64 counters start here; enables wrap tests

    uint32_t flags;                    // must be 0 in v1
    uint32_t reserved[7];              // must be 0
};

// One op. All ranges are logical rows; element addressing uses strides in RunSpec.
struct OpDesc {
    uint8_t kind;          // LnrOpKind
    uint8_t reserved_kind; // must be 0
    uint16_t flags;        // must be 0 in v1

    uint32_t param_id;     // for BWD/FLUSH; ignored for SAVE
    uint32_t x_row_base;   // source row base in x for SAVE/BWD
    uint32_t dy_row_base;  // source row base in dy for BWD; ignored for SAVE/FLUSH
    uint32_t cache_base;   // first cache slot for SAVE/BWD
    uint32_t rows;         // row count for SAVE/BWD; ignored for FLUSH

    uint32_t dx_out_base;    // first logical dx output row for BWD
    uint32_t flush_out_base; // flush record index for FLUSH
    uint32_t reserved0;      // must be 0
};

// RunSpec lives in host memory. run->ops is a host pointer to op_count OpDesc values.
struct RunSpec {
    uint64_t run_id;

    uint32_t op_count;
    uint32_t input_rows;    // logical row capacity of inputs x/dy for this run
    uint32_t dx_rows;       // logical rows in outputs.dx folded by scorer
    uint32_t flush_records; // logical rows in outputs.flush_dgamma/dbeta folded by scorer

    uint32_t x_stride_elems;     // >= hidden_size
    uint32_t dy_stride_elems;    // >= hidden_size
    uint32_t gamma_stride_elems; // >= hidden_size

    uint32_t reserved0;          // must be 0
    const OpDesc* ops;           // host pointer
};

// Host struct of device pointers.
struct InputPtrs {
    const void* x;     // storage_dtype, shape [input_rows, x_stride_elems]
    const void* dy;    // storage_dtype, shape [input_rows, dy_stride_elems]
    const void* gamma; // storage_dtype, shape [param_count, gamma_stride_elems]
};

// Host struct of device pointers. All snapshots are mandatory and overwritten every run.
struct OutputPtrs {
    void* dx; // storage_dtype, contiguous logical [dx_rows, hidden_size]

    float* flush_dgamma; // fp32 contiguous [flush_records, hidden_size]
    float* flush_dbeta;  // fp32 contiguous [flush_records, hidden_size]

    float* accum_dgamma_snapshot; // fp32 contiguous [param_count, hidden_size], after run
    float* accum_dbeta_snapshot;  // fp32 contiguous [param_count, hidden_size], after run

    float* cache_mean_snapshot;   // fp32 [max_cache_rows], after run
    float* cache_rstd_snapshot;   // fp32 [max_cache_rows], after run
    uint32_t* cache_kind_snapshot;// u32  [max_cache_rows], after run
    uint64_t* cache_gen_snapshot; // u64  [max_cache_rows], after run

    struct TimelineRecord* timeline; // [op_count], packed records below
    struct CounterSnapshot* counters; // [1]
};

// Packed event record. The scorer folds fields in exactly this order and width.
struct TimelineRecord {
    uint64_t run_id;
    uint64_t global_op_index; // cumulative op counter value after increment for this op

    uint32_t op_index_in_run;
    uint8_t kind;
    uint8_t status;
    uint8_t cache_kind_required; // 0 for SAVE/FLUSH, 1 LN, 2 RMS for backward
    uint8_t reserved_a;          // 0

    uint32_t param_id;
    uint32_t rows;
    uint32_t x_row_base;
    uint32_t dy_row_base;
    uint32_t cache_base;
    uint32_t dx_out_base;
    uint32_t flush_out_base;

    uint32_t partial_base;
    uint32_t partial_count;

    uint64_t cache_generation_first; // 0 if not applicable
    uint64_t cache_generation_last;  // 0 if not applicable

    uint64_t counter_snapshot_after_op; // same as global_op_index in v1
};

// All counters are cumulative since init/reset and wrap modulo 2^64.
struct CounterSnapshot {
    uint64_t run_count;
    uint64_t op_count;

    uint64_t save_ln_rows;
    uint64_t save_rms_rows;
    uint64_t bwd_ln_rows;
    uint64_t bwd_rms_rows;

    uint64_t partial_blocks;
    uint64_t flush_count;

    uint64_t invalid_ops;
    uint64_t cache_miss_ops;
    uint64_t cache_overwrite_rows;

    uint64_t cache_generation_counter;
    uint64_t last_run_id;
};

#pragma pack(pop)
