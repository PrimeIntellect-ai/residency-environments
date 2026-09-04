// file: deterministic_mergeable_quantile_sketch_common.h

#ifndef DETERMINISTIC_MERGEABLE_QUANTILE_SKETCH_COMMON_H_
#define DETERMINISTIC_MERGEABLE_QUANTILE_SKETCH_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define DMQS_ABI_VERSION 1

// k is the per-level buffer capacity. Must be even (clean 2:1 compaction).
#define DMQS_MIN_K 2
#define DMQS_MAX_K 64

// Maximum number of compactor levels. Items at level L carry weight 2^L.
// With L < 32 the weight 2^L fits in int32 and the cumulative weight fits in
// int64 for all batch/step bounds below.
#define DMQS_MIN_LEVELS 4
#define DMQS_MAX_LEVELS 24

#define DMQS_MAX_BATCH 4096
#define DMQS_MAX_STEPS 64

// Operation codes carried in DmqsRunSpec.op.
#define DMQS_OP_INGEST 0  // append in.keys[0..batch_size) at level 0, cascade-compact.
#define DMQS_OP_MERGE  1  // merge the foreign sketch described by in.merge_* , cascade-compact.
#define DMQS_OP_QUERY  2  // weighted-rank quantile query using q_num/q_den; no state mutation.

/*
CONTRACT: deterministic_mergeable_quantile_sketch

A fully deterministic, exact-integer KLL-style mergeable quantile sketch over
int32 keys. NO randomness anywhere: compaction parity is a pure function of a
per-level compaction counter.

------------------------------------------------------------------------------
PERSISTENT STATE
------------------------------------------------------------------------------
  level_size[L]                for L in [0, num_levels_cap):
      number of live items currently stored in level L's buffer.

  level_buf[L][0 .. level_size[L])  int32 keys.
      The buffer is NOT required to be kept sorted between operations. It is a
      plain append region. Item ordering inside a level is "insertion order":
      the slot index 0,1,2,... in which items were appended. Insertion order is
      the documented tie-break for stable sorting (see below).

  compaction_counter[L]        int64 number of times level L has been compacted
      since the last reset. Parity for the NEXT compaction of level L is
      (compaction_counter[L] & 1).

  num_levels_cap = spec.num_levels (allocated levels). A level L is "active" if
      level_size[L] > 0 OR compaction_counter[L] > 0 OR there exists an active
      level above it. num_levels (reported) = (highest active level)+1, or 0 if
      no level is active and all counters are 0 (i.e. the empty sketch).

  Capacity invariant: 0 <= level_size[L] <= k for every active level after every
  operation completes (cascade compaction guarantees this).

------------------------------------------------------------------------------
WEIGHT MODEL
------------------------------------------------------------------------------
  An item stored at level L represents 2^L underlying ingested keys.
  total_weight = sum over all live items of 2^L(item).
  total_weight is also exactly the number of INGESTed keys that the sketch
  summarizes (compaction preserves total weight: dropping k/2 of k equally
  weighted items and doubling the survivors' weight is weight-preserving when
  k is even).

------------------------------------------------------------------------------
DETERMINISTIC COMPACTION  (the core hard rule)
------------------------------------------------------------------------------
  compact_level(L):
    precondition: level_size[L] == k  (compaction is triggered only at full).
    1. Take the k keys in level L. Produce a STABLE ASCENDING sort of them,
       ordered by (key ascending, insertion_order ascending). Call the result
       s[0..k).  (Insertion order = original slot index within level L.)
    2. parity = compaction_counter[L] & 1.
         if parity == 0: KEEP odd indices  -> s[1], s[3], s[5], ...   (drop even)
         if parity == 1: KEEP even indices -> s[0], s[2], s[4], ...   (drop odd)
       Exactly k/2 survivors result. Survivors retain their relative ascending
       order.
    3. compaction_counter[L] += 1.
    4. level_size[L] = 0  (level L is emptied).
    5. APPEND the k/2 survivors, in their ascending order, to level L+1's
       buffer (each gains weight: it now lives at level L+1 = weight 2^(L+1)).
       Appending uses the next free slots of level L+1 (insertion order
       continues).
    If L+1 == num_levels_cap and a compaction would need it, that is a hard
    capacity error (the harness sizes num_levels so this never happens).

  cascade_compact():
    for L = 0, 1, 2, ... while L < num_levels_cap:
      while level_size[L] == k:
        compact_level(L)   // may push into L+1, handled when the loop reaches it
    The cascade ALWAYS scans levels in ascending order. A compaction at L can
    only raise level_size[L+1], which is visited later, so a single ascending
    pass that re-checks each level until it is under capacity suffices.

------------------------------------------------------------------------------
OPERATIONS (one per step, in DmqsRunSpec.op)
------------------------------------------------------------------------------
  DMQS_OP_INGEST:
    Append in.keys[0 .. batch_size) to level 0's buffer in array order (slot
    order = insertion order). Then cascade_compact().
    Note: if batch_size > k the level-0 buffer can momentarily exceed k during
    the append; compaction is applied AFTER the whole batch is appended, and
    repeatedly while level_size[0] >= k. Define overflow handling precisely:
      append all batch_size keys first (buffer may hold up to k-1+batch_size),
      then while level_size[0] >= k: take the FIRST k keys currently in level 0
      (slots 0..k) as the compaction input, compact them per the rule, and the
      remaining keys shift down to slots 0.. (their relative order preserved,
      insertion order re-indexed to their new slot positions). Continue until
      level_size[0] < k, then cascade upward.
    (Equivalently: level 0 is a FIFO; compaction always consumes the oldest k.)

  DMQS_OP_MERGE:
    A foreign sketch is provided via inputs:
      merge_num_levels                 int32 (# foreign levels, <= num_levels_cap)
      merge_level_size[0..merge_num_levels)   int32
      merge_keys[ L*k + j ]            foreign level L slot j, j < merge_level_size[L]
    For L = 0 .. merge_num_levels-1, in ascending L, append the foreign level-L
    keys (in foreign slot order 0,1,...) to OUR level L buffer (continuing our
    insertion order). After ALL foreign levels are appended, cascade_compact().
    Foreign compaction counters are NOT imported; only our counters advance.

  DMQS_OP_QUERY:
    Does NOT mutate state. Uses q_num/q_den (q_den > 0, q_num clamped to
    [0,q_den]).
      if total_weight == 0: query_result = INT32_MIN.
      else:
        Build the multiset of all live items as (key, weight=2^level). Sort
        ascending by (key ascending, level ascending, insertion_order ascending).
        target = ceil(q_num * total_weight / q_den), clamped to [1, total_weight].
        Walk the sorted items accumulating weight; return the key of the FIRST
        item whose cumulative weight >= target.
    For non-QUERY steps, query_result = INT32_MIN.

------------------------------------------------------------------------------
OUTPUTS AFTER EVERY STEP
------------------------------------------------------------------------------
  total_weight[0]        int64  sum of 2^level over all live items.
  num_levels[0]          int32  (highest active level + 1), or 0 if empty sketch.
  num_retained_items[0]  int32  sum of level_size[L] over all levels.
  query_result[0]        int32  query answer (QUERY step) or INT32_MIN otherwise.

  FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
    the agent MUST use this exact basis or every checksum fails):
      offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
        the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
      prime        = 1099511628211  (0x100000001B3).
      fold: start h = offset basis; absorb each field's raw bytes little-endian at
        its full declared width (u8=1, u32=4, u64=8, i64=8 two's-complement); per
        byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator / final mix.

  sketch_checksum[0]     uint64 FNV-1a-64 over, for L = 0 .. num_levels_cap-1:
      feed int32 L, int32 level_size[L], then for each item j in
      0..level_size[L): the SORTED-ASCENDING (key, insertion_order) pair as two
      int32s in (key asc, insertion_order asc) order, then int32 weight=2^L.
      (i.e. per level we hash the canonical sorted view, not raw slot order.)
      Levels with level_size==0 contribute (L, 0) only.

  state_checksum[0]      uint64 FNV-1a-64 over k, num_levels_cap, total_weight
      (as int64), num_retained_items (int32), then for L = 0..num_levels_cap-1:
      int32 L, int32 level_size[L], int64 compaction_counter[L], then the
      level's keys in RAW SLOT ORDER (slot 0,1,...) as int32s. state_checksum
      therefore captures BOTH the live contents AND the compaction history and
      the raw slot layout.

------------------------------------------------------------------------------
RULES
------------------------------------------------------------------------------
  - solution_init may allocate persistent state.
  - solution_run may NOT call cudaMalloc / cudaFree.
  - solution_run may launch kernels and use the provided workspace.
  - All graded outputs are EXACT integers (no float, no tolerance).
  - Determinism: identical op streams from a reset state produce identical
    outputs every time.
*/

struct alignas(8) DmqsProblemSpec {
    int32_t abi_version;
    int32_t k;            // per-level buffer capacity (even)
    int32_t num_levels;   // allocated level count (num_levels_cap)
    int32_t max_batch;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[10];
};

struct alignas(8) DmqsRunSpec {
    int32_t abi_version;
    int32_t op;            // DMQS_OP_*
    int32_t batch_size;    // # of keys for INGEST (0 for MERGE/QUERY)
    int32_t q_num;
    int32_t q_den;
    int32_t merge_num_levels;  // # of foreign levels for MERGE (0 otherwise)
    int32_t step_id;
    int32_t reserved[9];
};

struct alignas(8) DmqsInputs {
    const int32_t* keys;            // INGEST keys [batch_size]
    const int32_t* merge_level_size;  // MERGE foreign per-level size [merge_num_levels]
    const int32_t* merge_keys;        // MERGE foreign keys [num_levels * k], row-major L*k+j
};

struct alignas(8) DmqsOutputs {
    int64_t* total_weight;
    int32_t* num_levels;
    int32_t* num_retained_items;
    int32_t* query_result;
    uint64_t* sketch_checksum;
    uint64_t* state_checksum;
};

static_assert(sizeof(DmqsProblemSpec) == 64, "DmqsProblemSpec layout drift");
static_assert(sizeof(DmqsRunSpec) == 64, "DmqsRunSpec layout drift");
static_assert(sizeof(DmqsInputs) == 24, "DmqsInputs layout drift");
static_assert(sizeof(DmqsOutputs) == 48, "DmqsOutputs layout drift");

static inline size_t dmqs_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int dmqs_validate_problem_spec(const DmqsProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != DMQS_ABI_VERSION) return 0;
    if (spec->k < DMQS_MIN_K || spec->k > DMQS_MAX_K) return 0;
    if ((spec->k & 1) != 0) return 0;  // k must be even
    if (spec->num_levels < DMQS_MIN_LEVELS || spec->num_levels > DMQS_MAX_LEVELS) return 0;
    if (spec->max_batch < 0 || spec->max_batch > DMQS_MAX_BATCH) return 0;
    if (spec->max_steps < 1 || spec->max_steps > DMQS_MAX_STEPS) return 0;
    return 1;
}

static inline int dmqs_validate_run_spec(const DmqsRunSpec* run, const DmqsProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != DMQS_ABI_VERSION) return 0;
    if (run->op != DMQS_OP_INGEST && run->op != DMQS_OP_MERGE && run->op != DMQS_OP_QUERY) return 0;
    if (run->op == DMQS_OP_INGEST) {
        if (run->batch_size < 0 || run->batch_size > spec->max_batch) return 0;
    } else {
        if (run->batch_size != 0) return 0;
    }
    if (run->op == DMQS_OP_MERGE) {
        if (run->merge_num_levels < 0 || run->merge_num_levels > spec->num_levels) return 0;
    } else {
        if (run->merge_num_levels != 0) return 0;
    }
    if (run->op == DMQS_OP_QUERY) {
        if (run->q_den <= 0) return 0;
        if (run->q_num < 0) return 0;
    }
    return 1;
}

extern "C" size_t solution_workspace_bytes(const DmqsProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const DmqsProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const DmqsRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // DETERMINISTIC_MERGEABLE_QUANTILE_SKETCH_COMMON_H_
