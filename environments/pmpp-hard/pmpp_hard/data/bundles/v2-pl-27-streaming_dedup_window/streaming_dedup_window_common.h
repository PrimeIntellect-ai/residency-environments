// file: streaming_dedup_window_common.h

#ifndef STREAMING_DEDUP_WINDOW_COMMON_H_
#define STREAMING_DEDUP_WINDOW_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define SDW_ABI_VERSION 1

#define SDW_MIN_KEY_SPACE 1
#define SDW_MAX_KEY_SPACE 65536
#define SDW_MIN_CAPACITY 1
#define SDW_MAX_CAPACITY 65536
#define SDW_MIN_WINDOW 1
#define SDW_MAX_WINDOW 1048576
#define SDW_MAX_BATCH 4096
#define SDW_MAX_STEPS 64

/*
CONTRACT: streaming_dedup_window

Stateful sliding-window deduplicator + aggregator.

Persistent state:
  For each key in [0, key_space):
    active[key]    int32, 0/1
    last_pos[key]  int32, position of most recent ingested event for that key
    lru_stamp[key] int32, recency stamp for LRU eviction
    agg[key]       int64, wrapping aggregate sum of merged values

  Global:
    current_pos    int32, monotonically increasing event position, starts at 0
    active_count   int32

Event position:
  Every ingested event advances current_pos by 1, including invalid-key events.
  The first event after reset has position 1.

Expiry:
  Before processing each event after incrementing current_pos, expire active
  keys whose:
    current_pos - last_pos[key] > window_size

  Expiry finalizes the key into the evicted/finalized output stream.
  Expired keys are scanned/finalized in ascending key order.

Dedup/insert:
  For an event (key, value):
    if key is invalid, it only advances time and triggers expiry.

    else if key is active after expiry:
      DUP: agg[key] += value with int64 two's-complement wrap,
           last_pos[key] = current_pos,
           lru_stamp[key] = current_pos

    else:
      NEW: if active_count == capacity, evict the LRU active key first.
           LRU order: smaller lru_stamp wins; ties use smaller key.
           Capacity eviction also finalizes the evicted key.
           Then insert key with:
             active = 1
             last_pos = current_pos
             lru_stamp = current_pos
             agg = int64(value)

      If capacity is positive but no active key can be found, insertion is a no-op.
      Generated specs always have capacity >= 1.

Step outputs:
  active_count[0]:
    active keys after processing the whole batch and final expiry.

  num_new[0]:
    number of valid events that inserted a new active key.

  num_dup[0]:
    number of valid events that merged into an already active key.

  num_evicted[0]:
    number of finalized keys from both window expiry and capacity eviction.

  evicted_key_checksum[0]:
    FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
      the agent MUST use this exact basis or every checksum fails):
        offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
          the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
        prime        = 1099511628211  (0x100000001B3).
        fold: start h = offset basis; absorb each field's raw bytes little-endian at
          its full declared width (u8=1, u32=4, u64=8, i64=8 two's-complement); per
          byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator / final mix.

    FNV-1a 64-bit hash, initialized to offset basis. For each finalized key
    in deterministic finalize order, hash:
      key int32
      aggregate int64
      last_pos int32
    If no key is finalized, this is the offset basis.

  live_agg_sum[0]:
    int64 two's-complement wrapping sum of agg[key] over all active keys.

  state_checksum[0]:
    FNV-1a over:
      key_space, capacity, window_size, current_pos, active_count,
      then for every key in ascending key order:
        active, last_pos, lru_stamp, agg.

Rules:
  - solution_init may allocate persistent state.
  - solution_run may not call cudaMalloc/cudaFree.
  - solution_run may launch kernels and use provided workspace.
  - All outputs are exact.
*/

struct alignas(8) SdwProblemSpec {
    int32_t abi_version;
    int32_t key_space;
    int32_t capacity;
    int32_t window_size;
    int32_t max_batch;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[9];
};

struct alignas(8) SdwRunSpec {
    int32_t abi_version;
    int32_t batch_size;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) SdwInputs {
    const int32_t* key;
    const int32_t* value;
};

struct alignas(8) SdwOutputs {
    int32_t* active_count;
    int32_t* num_new;
    int32_t* num_dup;
    int32_t* num_evicted;
    uint64_t* evicted_key_checksum;
    int64_t* live_agg_sum;
    uint64_t* state_checksum;
};

static_assert(sizeof(SdwProblemSpec) == 64, "SdwProblemSpec layout drift");
static_assert(sizeof(SdwRunSpec) == 64, "SdwRunSpec layout drift");
static_assert(sizeof(SdwInputs) == 16, "SdwInputs layout drift");
static_assert(sizeof(SdwOutputs) == 56, "SdwOutputs layout drift");

static inline size_t sdw_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int sdw_validate_problem_spec(const SdwProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != SDW_ABI_VERSION) return 0;
    if (spec->key_space < SDW_MIN_KEY_SPACE || spec->key_space > SDW_MAX_KEY_SPACE) return 0;
    if (spec->capacity < SDW_MIN_CAPACITY || spec->capacity > SDW_MAX_CAPACITY) return 0;
    if (spec->capacity > spec->key_space) return 0;
    if (spec->window_size < SDW_MIN_WINDOW || spec->window_size > SDW_MAX_WINDOW) return 0;
    if (spec->max_batch < 0 || spec->max_batch > SDW_MAX_BATCH) return 0;
    if (spec->max_steps < 1 || spec->max_steps > SDW_MAX_STEPS) return 0;
    return 1;
}

static inline int sdw_validate_run_spec(const SdwRunSpec* run, const SdwProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != SDW_ABI_VERSION) return 0;
    if (run->batch_size < 0 || run->batch_size > spec->max_batch) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const SdwProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const SdwProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const SdwRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // STREAMING_DEDUP_WINDOW_COMMON_H_
