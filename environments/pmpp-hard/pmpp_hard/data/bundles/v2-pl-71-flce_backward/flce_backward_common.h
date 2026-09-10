// common.h — agent-visible ABI for FLCB-StateTrain v1.
//
// The starter solution.cu shipped to solvers MUST be an empty no-op ABI stub.
// It must not contain reference code, helper kernels, hidden constants, or
// checksum answers.

#pragma once
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#define FLCB_MAGIC   0x464C4342u      // "FLCB"
#define FLCB_VERSION 1u

// Project FNV basis, intentionally NOT canonical FNV offset.
#define FLCB_FNV_BASIS 1469598103934665603ull  // 0x14650FB0739D0383
#define FLCB_FNV_PRIME 1099511628211ull

enum FLCBOpCode : uint32_t {
    FLCB_OP_MICRO = 1,   // process a micro-batch row range
    FLCB_OP_FLUSH = 2,   // emit accumulated dW/dbias snapshot, optionally zero
    FLCB_OP_BUMP  = 3    // adversarial counter bump for wrap testing
};

enum FLCBReduction : uint32_t {
    FLCB_RED_SUM        = 0, // grad_scale = op.loss_scale
    FLCB_RED_MEAN_VALID = 1  // grad_scale = op.loss_scale / valid_count; zero if valid_count == 0
};

enum FLCBFlushMode : uint32_t {
    FLCB_FLUSH_SNAPSHOT      = 0, // emit current accumulators, keep state
    FLCB_FLUSH_EMIT_AND_ZERO = 1  // emit current accumulators, then zero accumulators
};

enum FLCBBumpCounter : uint32_t {
    FLCB_BUMP_ROWS_TOTAL         = 0,
    FLCB_BUMP_VALID_ROWS_TOTAL   = 1,
    FLCB_BUMP_VOCAB_TILE_UPDATES = 2,
    FLCB_BUMP_EVENTS_TOTAL       = 3
};

enum FLCBEventType : uint32_t {
    FLCB_EV_RUN_BEGIN  = 1,
    FLCB_EV_MICRO_BEGIN= 2,
    FLCB_EV_CHUNK_DONE = 3,
    FLCB_EV_MICRO_DONE = 4,
    FLCB_EV_FLUSH      = 5,
    FLCB_EV_BUMP       = 6,
    FLCB_EV_INVALID_OP = 7,
    FLCB_EV_RUN_END    = 8
};

enum FLCBInvalidFlag : uint32_t {
    FLCB_BAD_OPCODE    = 1u << 0,
    FLCB_BAD_RANGE     = 1u << 1,
    FLCB_BAD_EPS       = 1u << 2,
    FLCB_BAD_REDUCTION = 1u << 3,
    FLCB_BAD_FLUSH     = 1u << 4,
    FLCB_BAD_BUMP      = 1u << 5,
    FLCB_BAD_CAPACITY  = 1u << 6
};

// All structs use natural C layout on little-endian x86_64 + CUDA.
// Reserved fields must be zero in the test harness and must be preserved as zero in outputs.

struct alignas(8) FLCBSpec {
    uint32_t magic;          // FLCB_MAGIC
    uint32_t version;        // FLCB_VERSION
    uint32_t H;              // hidden dimension
    uint32_t V;              // vocabulary/classes
    uint32_t max_run_rows;   // maximum RunSpec.input_rows
    uint32_t max_ops;        // maximum RunSpec.op_count
    uint32_t row_chunk;      // rows per deterministic chunk
    uint32_t vocab_tile;     // vocab tile for online softmax
    int32_t  ignore_index;   // e.g. -100
    uint32_t has_bias;       // 0 or 1
    uint32_t max_flushes_per_run;
    uint32_t flags;          // must be 0 for v1
    uint32_t reserved[8];    // must be 0
};

struct alignas(8) FLCBRunSpec {
    uint64_t run_id;              // arbitrary, copied into event timeline
    uint32_t input_rows;          // rows available in x_bf16/target
    uint32_t output_row_capacity; // capacity for appended dX/loss rows
    uint32_t op_count;            // number of ops in inputs.ops
    uint32_t event_capacity;      // capacity of outputs.events
    uint32_t flush_capacity;      // capacity of outputs.flush_dW/flush_dbias snapshots
    uint32_t flags;               // must be 0 for v1
    uint32_t reserved[4];         // must be 0
};

// One op. Interpretation depends on opcode.
// For MICRO:
//   row_offset, row_count, reduction, label_smoothing_q16, loss_scale_bits, tag are used.
//   aux and bump_amount must be ignored.
// For FLUSH:
//   aux is FLCBFlushMode; tag is used.
// For BUMP:
//   aux is FLCBBumpCounter; bump_amount is added modulo 2^64; tag is used.
struct alignas(8) FLCBOp {
    uint32_t opcode;
    uint32_t row_offset;
    uint32_t row_count;
    uint32_t reduction;
    uint32_t label_smoothing_q16; // epsilon = q / 65536.0, valid q <= 65536
    uint32_t loss_scale_bits;     // raw IEEE-754 binary32 upstream scalar
    uint32_t aux;
    uint32_t reserved0;           // must be 0
    uint64_t bump_amount;
    uint64_t tag;
};

// inputs/outputs are host-side pointer packets.
// The pointed-to buffers are device buffers unless explicitly stated otherwise.
struct alignas(8) FLCBInputs {
    const FLCBOp*  ops;       // device array [run.op_count]
    const uint16_t* x_bf16;   // device [run.input_rows, H], row-major BF16 bits
    const int32_t*  target;   // device [run.input_rows]
    const uint16_t* w_bf16;   // device [V, H], row-major BF16 bits
    const uint16_t* bias_bf16;// device [V] if spec.has_bias != 0, else may be null
};

struct alignas(8) FLCBEvent {
    uint32_t type;
    uint32_t op_index;
    uint32_t chunk_index; // 0xffffffff when not applicable
    uint32_t flags;

    uint64_t run_id;
    uint64_t tag;

    uint64_t row_begin;
    uint64_t row_count;
    uint64_t valid_count;
    uint64_t ignored_count;
    uint64_t invalid_count;

    uint64_t counter_rows_after;
    uint64_t counter_valid_after;
    uint64_t flush_generation_after;
    uint64_t event_serial;
};

struct alignas(8) FLCBCounters {
    uint64_t runs;
    uint64_t ops_seen;
    uint64_t micro_ops;
    uint64_t flush_ops;
    uint64_t invalid_ops;

    uint64_t row_chunks;
    uint64_t vocab_tile_updates; // per-row online tile updates
    uint64_t rows_total;
    uint64_t valid_rows_total;
    uint64_t ignored_rows_total;
    uint64_t invalid_targets_total;
    uint64_t output_rows_total;

    uint64_t flush_generation;
    uint64_t events_total;
    uint64_t last_run_id;

    uint64_t reserved[8]; // must stay 0
};

struct alignas(8) FLCBRunReport {
    uint32_t status;             // 0 on valid structural run; invalid-op events do not set this
    uint32_t output_rows;        // rows actually appended to dx/loss
    uint32_t flushes_written;    // flush snapshots written
    uint32_t events_written;     // events written
    uint64_t required_workspace_bytes;
    uint64_t reserved[5];        // must stay 0
};

struct alignas(8) FLCBOutputs {
    uint16_t*      dx_bf16;       // device [run.output_row_capacity, H]
    float*         loss_f32;      // device [run.output_row_capacity]
    float*         flush_dW_f32;  // device [run.flush_capacity, V, H]
    float*         flush_dbias_f32;//device [run.flush_capacity, V], zero if no bias
    FLCBEvent*     events;        // device [run.event_capacity]
    FLCBCounters*  counters;      // device [1]
    FLCBRunReport* report;        // device [1]
};

// Required struct sizes for v1:
//   sizeof(FLCBSpec)      == 80
//   sizeof(FLCBRunSpec)   == 48
//   sizeof(FLCBOp)        == 48
//   sizeof(FLCBEvent)     == 104
//   sizeof(FLCBCounters)  == 184
//   sizeof(FLCBRunReport) == 64
// A submission is invalid if a compiler produces different sizes.

extern "C" size_t solution_workspace_bytes(const FLCBSpec* spec);

extern "C" cudaError_t solution_init(
    const FLCBSpec* spec,
    void** state,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const FLCBRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream);

extern "C" void solution_destroy(void* state);
