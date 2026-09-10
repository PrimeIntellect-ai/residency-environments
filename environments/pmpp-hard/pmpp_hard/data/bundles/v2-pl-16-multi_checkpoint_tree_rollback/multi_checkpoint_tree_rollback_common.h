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

// === DETERMINISM & EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section is SELF-CONTAINED. Every grader-enforced checksum below is
// fully specified here; nothing is deferred to any other file. A solver that
// reproduces exactly these byte streams and FNV folds reproduces every hash.
//
// ---- FNV-1a-64 primitive ----
//   basis  = 1469598103934665603  (0x14650FB0739D0383)
//   prime  = 1099511628211        (0x00000100000001B3)
//   fold one byte b into running hash h:  h = (h XOR b) * prime   (mod 2^64)
//   A multi-byte field is folded byte-by-byte in LITTLE-ENDIAN order
//   (i.e. exactly the in-memory byte order of the value on the device:
//   the value is reinterpreted as its raw bytes, index 0..size-1, and each
//   byte is folded in that order). int32 is folded as its 4 raw bytes;
//   uint64 as its 8 raw bytes.
//
//   All hashes start from h = basis (no length prefix, no trailing pad).
//
// ---- tail-hash helper T(seq,len) ----  (used by live_cache_tail_hash AND by
//   the checkpoint_hash stored on SAVE_CHECKPOINT)
//   Let tail_count = min(len, MCTR_TAIL_VALUES=32) (an int32),
//       tail_start = len - tail_count.
//   h = basis
//   fold tail_count           as int32 (4 bytes, little-endian)
//   if tail_count > 0:
//     fold the int32 array cache[seq][tail_start .. tail_start+tail_count-1]
//     as one contiguous byte run of (4 * tail_count) bytes, i.e. each cache
//     entry folded as int32 in increasing index order.
//   Result is T(seq,len).
//   NOTE: SAVE_CHECKPOINT stores checkpoint_hash = T(seq, current_length)
//   computed at push time over the CURRENT cache contents and length.
//
// ---- live_cache_tail_hash[seq] (per step, per sequence) ----
//   = T(seq, length[seq]) using the live length AFTER this step's ops.
//
// ---- live_cache_sum[seq] ----
//   sum_bits is a uint64 accumulator initialized to 0; for i in [0,length):
//     sum_bits += (uint64_t)(int64_t)cache[seq][i]   (two's-complement,
//     sign-extend each int32 to int64 then add mod 2^64, wrapping).
//   live_cache_sum[seq] = (int64_t)sum_bits.
//
// ---- top_checkpoint_len[seq] ----
//   = ckpt_len of the top-of-stack entry (index depth-1) if depth>0, else -1.
//
// ---- state_checksum[1] (single global hash over ALL sequences) ----
//   h = basis. Iterate seq = 0,1,...,B-1 (ascending, ALL B sequences,
//   including inactive/empty ones). For each seq let len=length[seq],
//   dep=depth[seq] and fold IN THIS EXACT ORDER:
//     1. fold len  as int32
//     2. fold dep  as int32
//     3. for d = 0,1,...,dep-1 (checkpoint stack, bottom-to-top):
//          fold ckpt_id[seq][d]    as int32
//          fold ckpt_len[seq][d]   as int32
//          fold ckpt_hash[seq][d]  as uint64   (the stored T() value)
//     4. fold len AGAIN as int32   (yes, len is folded a second time here)
//     5. if len > 0: fold cache[seq][0 .. len-1] as a contiguous run of
//          (4 * len) bytes (each entry as int32 in increasing index order).
//   state_checksum[0] = h after all B sequences.
//
// ---- op-semantics edge rules enforced by the grader ----
//   * token_count k is clamped to [0, MCTR_MAX_K=4]; accept_count a clamped to
//     [0, k] (after k is clamped).
//   * APPEND / ACCEPT_PREFIX writes use append-one: a value is written only
//     while length < max_len; length increments only on an actual write;
//     overflow values are silently discarded (NOT counted, NOT written).
//   * ACCEPT_PREFIX: writes token_values[0..a) first (each via append-one),
//     THEN appends correction_value via append-one IFF old_length + a < max_len
//     (old_length captured BEFORE any of this op's writes). Note the correction
//     gate uses old_length+a, independent of how many draft writes actually
//     landed.
//   * SAVE_CHECKPOINT: if depth < max_depth push (id, current length, T(seq,
//     length)); else no-op (stack full).
//   * ROLLBACK_TO: scan stack from top (index depth-1) DOWN to 0; first entry
//     with matching checkpoint_id wins; restore length := that entry's
//     ckpt_len and set depth := found_index+1 (the matching entry stays).
//     No match => no-op. Cache bytes above the restored length are stale and
//     are excluded from all outputs (outputs only read [0,length)).
//   * A row whose active_seq is out of [0,B) is ignored (no state change).
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
