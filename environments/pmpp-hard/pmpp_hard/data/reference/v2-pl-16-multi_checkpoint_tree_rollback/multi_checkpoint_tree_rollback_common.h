// ============================================================================
// file: multi_checkpoint_tree_rollback_common.h
// ============================================================================

#ifndef MULTI_CHECKPOINT_TREE_ROLLBACK_COMMON_H_
#define MULTI_CHECKPOINT_TREE_ROLLBACK_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MCTR_ABI_VERSION 1

#define MCTR_MIN_B 1
#define MCTR_MAX_B 64
#define MCTR_MIN_MAX_LEN 16
#define MCTR_MAX_MAX_LEN 4096
#define MCTR_MIN_MAX_STEPS 16
#define MCTR_MAX_MAX_STEPS 64
#define MCTR_MAX_K 4
#define MCTR_MAX_DEPTH 8
#define MCTR_TAIL_VALUES 32

enum MctrOpCode : int32_t {
    MCTR_OP_NOOP = 0,
    MCTR_OP_APPEND = 1,
    MCTR_OP_SAVE_CHECKPOINT = 2,
    MCTR_OP_ROLLBACK_TO = 3,
    MCTR_OP_ACCEPT_PREFIX = 4
};

/*
CONTRACT: multi_checkpoint_tree_rollback

Stateful exact-integer checkpointed token cache.

Persistent state per sequence:
  cache[max_len] int32
  length int32
  checkpoint stack with up to max_depth entries:
    checkpoint_id int32
    checkpoint_len int32
    checkpoint_hash uint64

solution_reset clears cache, length, and checkpoint stack.

Run model:
  Each solution_run processes one step. The step contains active_count active
  sequence rows. active_seq entries are generated unique within a step.

Inputs per active row:
  active_seq[row]      logical sequence id in [0,B)
  op_code[row]         one of MctrOpCode
  token_count[row]     k in [0,4], clamped to [0,4]
  accept_count[row]    a for ACCEPT_PREFIX, clamped to [0,k]
  checkpoint_id[row]   id for SAVE_CHECKPOINT / ROLLBACK_TO
  token_values[row,4]  draft / append values
  correction_value[row] correction token for ACCEPT_PREFIX

Operations:
  NOOP:
    No state change.

  APPEND:
    Append token_values[row,0:k) to the live cache until max_len is reached.
    Overflow tokens are discarded.

  SAVE_CHECKPOINT:
    If stack depth < max_depth, push:
      checkpoint_id,
      current length,
      current live_cache_tail_hash.
    If stack is full, the operation is a no-op.

  ROLLBACK_TO:
    Find the newest stack entry whose checkpoint_id matches. If found, restore
    length to that checkpoint_len and discard entries above it. The matching
    checkpoint remains on the stack. Cache bytes above the new length are stale
    and must not contribute to outputs. If no matching id exists, no-op.

  ACCEPT_PREFIX:
    Let k=token_count and a=accept_count. This is the exact net effect of
    speculative append + rollback:
      append the first a draft tokens token_values[row,0:a)
      then append one correction_value
    Capacity rule:
      accepted draft values are physically written while length < max_len.
      The correction is appended iff old_length + a < max_len.
      Final length is min(max_len, old_length + a + 1).

Outputs after every step, for all B sequences:
  length[B]:
    exact live cache length.

  num_checkpoints[B]:
    exact stack depth.

  top_checkpoint_len[B]:
    checkpoint_len of the stack top, or -1 if stack is empty.

  live_cache_sum[B]:
    int64 two's-complement wrapping sum over cache[0:length).

  live_cache_tail_hash[B]:
    FNV-1a 64-bit hash over:
      tail_count int32 bytes,
      then the last tail_count int32 values,
    where tail_count = min(length, 32).

  state_checksum[1]:
    FNV-1a 64-bit checksum over all sequence lengths, checkpoint stacks, and
    live cache contents. Used for deterministic replay/state grading.
*/

struct alignas(8) MctrProblemSpec {
    int32_t abi_version;
    int32_t B;
    int32_t max_len;
    int32_t max_steps;
    int32_t max_depth;
    int32_t flags;
    int32_t reserved[10];
};

struct alignas(8) MctrRunSpec {
    int32_t abi_version;
    int32_t active_count;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) MctrInputs {
    const int32_t* active_seq;
    const int32_t* op_code;
    const int32_t* token_count;
    const int32_t* accept_count;
    const int32_t* checkpoint_id;
    const int32_t* token_values;
    const int32_t* correction_value;
};

struct alignas(8) MctrOutputs {
    int64_t* live_cache_sum;
    uint64_t* live_cache_tail_hash;
    int32_t* length;
    int32_t* num_checkpoints;
    int32_t* top_checkpoint_len;
    uint64_t* state_checksum;
};

static_assert(sizeof(MctrProblemSpec) == 64, "MctrProblemSpec layout drift");
static_assert(sizeof(MctrRunSpec) == 64, "MctrRunSpec layout drift");
static_assert(sizeof(MctrInputs) == 56, "MctrInputs layout drift");
static_assert(sizeof(MctrOutputs) == 48, "MctrOutputs layout drift");

static inline size_t mctr_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int mctr_validate_problem_spec(const MctrProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MCTR_ABI_VERSION) return 0;
    if (spec->B < MCTR_MIN_B || spec->B > MCTR_MAX_B) return 0;
    if (spec->max_len < MCTR_MIN_MAX_LEN || spec->max_len > MCTR_MAX_MAX_LEN) return 0;
    if (spec->max_steps < MCTR_MIN_MAX_STEPS || spec->max_steps > MCTR_MAX_MAX_STEPS) return 0;
    if (spec->max_depth < 1 || spec->max_depth > MCTR_MAX_DEPTH) return 0;
    return 1;
}

static inline int mctr_validate_run_spec(const MctrRunSpec* run, const MctrProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != MCTR_ABI_VERSION) return 0;
    if (run->active_count < 0 || run->active_count > spec->B) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MctrProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MctrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MctrRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MULTI_CHECKPOINT_TREE_ROLLBACK_COMMON_H_
