// common.h -- bwd_scatter / DSG-MoE-BWD contract
// Extracted verbatim from §2 of the task design.
// Solver: submit solution.cu implementing the five ABI functions below.

#pragma once
#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

// §2.2 Constants
static constexpr uint32_t DSG_SPEC_MAGIC = 0x44534753u; // "SGSD" little-endian marker
static constexpr uint32_t DSG_RUN_MAGIC  = 0x44534752u; // "RGSD"
static constexpr uint32_t DSG_OUT_MAGIC  = 0x4453474Fu; // "OGSD"
static constexpr uint32_t DSG_VERSION    = 1;

static constexpr uint32_t DSG_DTYPE_BF16 = 0;
static constexpr uint32_t DSG_DTYPE_F32  = 1;

static constexpr uint32_t DSG_FLAG_EMIT_TABLE        = 1u << 0;
static constexpr uint32_t DSG_FLAG_EMIT_EXPERT       = 1u << 1;
static constexpr uint32_t DSG_FLAG_CLEAR_AFTER_FLUSH = 1u << 2;

static constexpr uint32_t DSG_VALID_RUN_FLAGS =
    DSG_FLAG_EMIT_TABLE |
    DSG_FLAG_EMIT_EXPERT |
    DSG_FLAG_CLEAR_AFTER_FLUSH;

static constexpr uint32_t DSG_CAPACITY_USE_SPEC = 0xFFFFFFFFu;

static constexpr uint64_t DSG_FNV_BASIS = 1469598103934665603ull; // 0x14650FB0739D0383
static constexpr uint64_t DSG_FNV_PRIME = 1099511628211ull;

// §2.3 Structs -- all integer fields little-endian for FNV folding

struct Spec {
    uint32_t magic;        // DSG_SPEC_MAGIC
    uint32_t version;      // DSG_VERSION

    uint32_t table_rows;   // R
    uint32_t dim;          // D, embedding dim and MoE input dim
    uint32_t out_dim;      // O, MoE expert output dim
    uint32_t experts;      // E
    uint32_t top_k;        // K, 1..4
    uint32_t max_tokens;   // max T per run

    uint32_t capacity;     // default per-expert capacity per run
    uint32_t input_dtype;  // DSG_DTYPE_BF16 or DSG_DTYPE_F32

    int32_t  padding_idx;  // if index == padding_idx, table update is skipped before bounds check
    uint32_t flags;        // must be 0 for v1

    uint64_t weight_seed;  // deterministic expert-weight seed

    uint64_t reserved0;    // must be 0
    uint64_t reserved1;    // must be 0
};

struct RunSpec {
    uint32_t magic;             // DSG_RUN_MAGIC
    uint32_t version;           // DSG_VERSION
    uint32_t tokens;            // T, 0..spec.max_tokens
    uint32_t flags;             // DSG_FLAG_*
    uint32_t capacity_override; // DSG_CAPACITY_USE_SPEC or explicit per-expert capacity
    uint32_t reserved0;         // must be 0
    uint64_t user_tag;          // folded into START event, no semantic effect
};

// Output begins with this header, followed by tensor payloads.
// Reserved fields must be zero.
struct OutHeader {
    uint32_t magic;          // DSG_OUT_MAGIC
    uint16_t version;        // DSG_VERSION
    uint16_t header_bytes;   // sizeof(OutHeader)

    uint64_t run_id;         // zero-based id before this run
    uint64_t completed_runs; // run_id + 1 after successful run

    uint32_t tokens;
    uint32_t flags;
    uint32_t effective_capacity;
    uint32_t top_k;

    uint32_t accepted;
    uint32_t dropped_capacity;
    uint32_t dropped_invalid_expert;
    uint32_t skipped_padding;
    uint32_t skipped_invalid_index;
    uint32_t table_segments;
    uint32_t expert_segments;
    uint32_t flush_happened;

    uint64_t total_tokens;
    uint64_t total_accepted;
    uint64_t total_dropped_capacity;
    uint64_t total_dropped_invalid_expert;
    uint64_t total_skipped_padding;
    uint64_t total_skipped_invalid_index;
    uint64_t total_table_segments;
    uint64_t total_expert_segments;
    uint64_t total_flushes;

    uint64_t event_hash;      // per-run canonical event timeline hash
    uint64_t tensor_hash;     // per-run emitted tensor payload hash
    uint64_t state_hash;      // post-run persistent state hash
    uint64_t step_hash;       // FNV over summary fields, defined in §2.14

    uint64_t reserved0;       // 0
    uint64_t reserved1;       // 0
};

// §2.1 ABI -- implemented in solution.cu
extern "C" size_t solution_workspace_bytes(const Spec* spec);

extern "C" cudaError_t solution_init(
    const Spec* spec,
    void** state,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const RunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);
