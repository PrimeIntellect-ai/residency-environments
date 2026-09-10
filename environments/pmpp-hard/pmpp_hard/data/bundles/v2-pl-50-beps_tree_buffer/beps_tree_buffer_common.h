// file: beps_tree_buffer_common.h

#ifndef BEPS_TREE_BUFFER_COMMON_H_
#define BEPS_TREE_BUFFER_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define BEPS_ABI_VERSION 1

// Capacity bounds (compile-time maxima used to size persistent buffers).
#define BEPS_MIN_NODES 4
#define BEPS_MAX_NODES 65536
#define BEPS_MAX_OPS 4096
#define BEPS_MAX_STEPS 256
#define BEPS_MAX_INTERNAL_BUFFER_CAP 1024
#define BEPS_MAX_LEAF_RECORD_CAP 1024
#define BEPS_MAX_CHILDREN 64
#define BEPS_MAX_FLUSH_MESSAGE_CAP 4096
#define BEPS_MAX_RANGE_RESULTS 4096
// Hard cap on total messages/records the persistent store can ever hold.
#define BEPS_MAX_TOTAL_SLOTS 1048576

// Operation kinds (BepsOp.kind).
#define BEPS_OP_PUT 0
#define BEPS_OP_ADD 1
#define BEPS_OP_DELETE 2
#define BEPS_OP_POINT_QUERY 3
#define BEPS_OP_RANGE_QUERY 4
#define BEPS_OP_FLUSH 5

// Message kinds (stored in node buffers / applied to leaves).
#define BEPS_MSG_SET 0
#define BEPS_MSG_ADD 1
#define BEPS_MSG_DEL 2

// message_event_hash event_kind:u8 values.
#define BEPS_EV_ROOT_BUFFER_SET 0
#define BEPS_EV_ROOT_BUFFER_ADD 1
#define BEPS_EV_ROOT_BUFFER_DEL 2
#define BEPS_EV_WRITE_STALL 3
#define BEPS_EV_FLUSH_MESSAGE 4
#define BEPS_EV_LEAF_APPLY_SET 5
#define BEPS_EV_LEAF_APPLY_ADD 6
#define BEPS_EV_LEAF_APPLY_DEL 7
#define BEPS_EV_FLUSH_EMPTY 8
#define BEPS_EV_INVALID 9

// query_hash record_kind:u8 values.
#define BEPS_REC_POINT_RESULT 0
#define BEPS_REC_RANGE_RESULT 1
#define BEPS_REC_RANGE_END 2

/*
CONTRACT: beps_tree_buffer  (Bε-Tree Message Buffer with Flush Cascades / Splits)

A persistent write-optimized Bε-tree. Updates are buffered as messages in
internal-node buffers and cascade downward in deterministic flush groups;
point/range visibility composes leaf records with all pending path messages
ordered globally by message sequence; overflowing leaves and internal nodes
split with range + buffer redistribution. Grounded in Bε-trees (Stony Brook).

Every graded output is an EXACT integer (cumulative counts + FNV-1a-64
checksums). No floats, no tolerance.

================================ STATE =================================

Persistent scalars: event_seq=0; msg_seq_next=1; node_id_next=1.
The tree starts as a single leaf root: node_id=1, range [0, UINT64_MAX],
depth 0, is_leaf=1, empty record map.

Node (common): node_id; parent_or_ZERO; low_key; high_key; depth; is_leaf.
Internal node: ordered child list (by child low_key ascending; child ranges
  partition the parent range inclusively, prev.high+1 == next.low) and a
  message buffer ordered by msg_seq ascending.
Leaf node: record map keyed by key; record = (value:i64, latest_seq:u64,
  deleted:u8). Deleted records (tombstones) remain in the map and DO count
  toward leaf_record_cap.
Message: (msg_seq, kind in {SET,ADD,DEL}, key, value:i64).

============================== OPERATIONS ==============================
(Applied strictly in op array order. op_index is the op's position in step.)

PUT(key,value) / ADD(key,delta) / DELETE(key):
  kind = SET / ADD / DEL respectively (DELETE value forced to 0).
  - If root is a leaf: consume msg_seq (= msg_seq_next++), apply directly to
    the leaf record map, emit LEAF_APPLY_{SET,ADD,DEL}. Then if the leaf's
    record count > leaf_record_cap, split it.
  - If root is internal: if root buffer length >= internal_buffer_cap (full)
    BEFORE append, first execute exactly ONE automatic flush group from root
    (same flush-group semantics as FLUSH below, unbounded by op budgets but
    still capped by flush_message_cap). If after that the root buffer is STILL
    full, emit WRITE_STALL and do NOT consume msg_seq. Otherwise consume
    msg_seq (= msg_seq_next++), append message to root buffer, emit
    ROOT_BUFFER_{SET,ADD,DEL}.

  Leaf application of a message (in moved/applied order):
    SET: record.value = value; latest_seq = msg_seq; deleted = 0 (create if
         absent).
    ADD: if no record OR record deleted -> base value 0; new value =
         (base + delta) mod 2^64 (stored as signed two's-complement i64);
         latest_seq = msg_seq; deleted = 0 (create if absent).
    DEL: record.value = 0; latest_seq = msg_seq; deleted = 1 (create tombstone
         if absent).

POINT_QUERY(read_id,key):
  Find the leaf whose range contains key. Seed state from the leaf record if
  present (found with its value/latest_seq), else missing with virtual seq 0.
  Collect ALL buffered messages for `key` on the root->leaf path (root first,
  deepest internal last). Merge the seeded leaf record (if present) and the
  path messages, then apply in ASCENDING msg_seq globally (NOT by depth):
    SET -> found, value set. ADD -> if missing/deleted start 0 then add, found.
    DEL -> deleted/missing.
  The seeded leaf record participates with its own latest_seq. Emit
  POINT_RESULT(found, value_or_INT64_MIN, latest_seq_or_0).

RANGE_QUERY(read_id,lo,hi,limit):
  Invalid if lo > hi (emit nothing, increment invalid_count, emit no records,
  no RANGE_END). Else effective_limit = min(limit, max_range_results).
  Candidate keys = every key in [lo,hi] that appears in some leaf record OR in
  any internal buffer message. For each candidate (ascending key) compute
  visibility exactly as POINT_QUERY; emit visible non-deleted keys as
  RANGE_RESULT(key,value,latest_seq) in ascending key order until
  effective_limit reached. Then emit RANGE_END(emitted_count).

FLUSH(max_flush_nodes,max_messages_total):
  If either budget <= 0: valid no-op (no FLUSH_EMPTY).
  While both budgets remain (>0):
    Choose eligible source = internal node with nonempty buffer maximizing
    buffer length; tie smallest depth; tie smallest low_key; tie smallest
    node_id. If none: emit FLUSH_EMPTY and stop.
    Within the source choose the child receiving the most buffered messages by
    key range (count of buffered msgs whose key in child's [low,high]); tie
    smallest child low_key then child node_id.
    Move that child's messages from source buffer to the child in msg_seq
    ascending order, capped by remaining max_messages_total and by
    flush_message_cap. Emit FLUSH_MESSAGE per moved message. If child internal:
    append in moved order. If child leaf: apply in moved order, emitting
    LEAF_APPLY_* each; then if leaf record count > leaf_record_cap split it.
    An internal child whose buffer now exceeds internal_buffer_cap is NOT
    recursively flushed; it just becomes eligible next loop iteration.
    Decrement max_flush_nodes by 1 after each source flush group. Decrement
    max_messages_total by the number of messages actually moved.

LEAF SPLIT (record_count > leaf_record_cap):
  sorted keys ascending; split_key = key at index record_count/2. New right
  leaf id = node_id_next++. Left keeps keys < split_key (range [old_low,
  split_key-1]); right gets keys >= split_key (range [split_key, old_high]).
  If split leaf was root: new internal root id = node_id_next++, depth 0,
  two children [left,right], empty buffer; old leaves depth becomes 1; the new
  root inherits parent (ZERO). Else insert right sibling into parent child list
  immediately after left. Emit LEAF_SPLIT. If parent now has > max_children,
  split the parent (cascades upward).

INTERNAL SPLIT (child count m > max_children):
  split_child_index = m/2. New right internal id = node_id_next++. Left keeps
  children [0,split_child_index); right gets [split_child_index,m). Ranges set
  to cover their children (left.low=old.low, left.high=children[sci-1].high;
  right.low=children[sci].low, right.high=old.high). Redistribute splitting
  node's buffer by key range (left range stays, right range moves), msg_seq
  order preserved within each side. If splitting node was root, create a new
  internal root with two children; else insert right after left in parent.
  Emit INTERNAL_SPLIT. Cascade upward until no parent exceeds max_children.

Depths after a root split: the new root takes the old root's depth; both
former-root halves get depth = new_root.depth + 1, and ALL descendants have
their depth incremented by 1 (the subtree shifts down one level).

msg_seq_next, node_id_next, event_seq all wrap modulo 2^64.

============================== OUTPUTS =================================
Cumulative counts (BepsCounts), accumulated across the WHOLE run (persist):
  root_buffered_set/add/del; write_stall; flush_groups; flush_messages;
  leaf_apply_set/add/del; leaf_splits; internal_splits; point_found;
  point_missing; range_results; range_end_count; flush_empty; invalid_count.

message_event_hash (FNV-1a-64, emission order) absorbs per event:
  event_kind:u8; event_seq:u64; op_index:u32; node_id:u64;
  msg_seq_or_UINT64_MAX:u64; kind_or_255:u8; key_or_UINT64_MAX:u64; value:i64;
  child_node_or_UINT64_MAX:u64.

query_hash (FNV-1a-64, op order) absorbs per record:
  POINT_RESULT: record_kind:u8; read_id:u64; op_index:u32; key:u64; found:u8;
    value_or_INT64_MIN:i64; latest_seq_or_ZERO:u64.
  RANGE_RESULT: record_kind:u8; read_id:u64; op_index:u32; key:u64; value:i64;
    latest_seq:u64.
  RANGE_END: record_kind:u8; read_id:u64; op_index:u32; emitted_count:u64.

tree_shape_hash (nodes by depth asc, then low_key, then node_id) absorbs:
  node_id:u64; parent_or_ZERO:u64; is_leaf:u8; depth:u32; low_key:u64;
  high_key:u64; child_count:u64; buffer_count:u64; record_count:u64.

buffer_hash (internal nodes in tree_shape canonical order, msg_seq asc):
  node_id:u64; msg_seq:u64; kind:u8; key:u64; value:i64.

leaf_hash (leaves by low_key then node_id; records by key ascending):
  leaf_node:u64; key:u64; value:i64; latest_seq:u64; deleted:u8.

All hashes begin from the FNV-1a-64 offset basis 1469598103934665603 and use
prime 1099511628211. Multi-byte fields are absorbed little-endian (native).

================================ RULES ================================
  - solution_init may allocate persistent device state.
  - solution_run may launch kernels and use the provided workspace but may NOT
    call cudaMalloc/cudaFree, and may NOT mutate its inputs.
  - solution_reset restores the single-leaf-root initial state and zeroes all
    cumulative counts and persistent scalars.
*/

struct alignas(8) BepsProblemSpec {
    int32_t abi_version;
    int32_t max_nodes;              // [BEPS_MIN_NODES, BEPS_MAX_NODES]
    int32_t internal_buffer_cap;    // buffer "full" threshold
    int32_t leaf_record_cap;        // leaf split threshold
    int32_t max_children_per_internal;
    int32_t flush_message_cap;      // per flush-group cap on moved msgs
    int32_t max_range_results;
    int32_t max_ops;                // max ops per step
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[6];
};

// One operation within a step.
//   PUT(key=u_key, value=value)            kind=BEPS_OP_PUT
//   ADD(key=u_key, delta=value)            kind=BEPS_OP_ADD
//   DELETE(key=u_key)                      kind=BEPS_OP_DELETE
//   POINT_QUERY(read_id=u_aux, key=u_key)  kind=BEPS_OP_POINT_QUERY
//   RANGE_QUERY(read_id=u_aux, lo=u_key, hi=u_key2, limit=value)
//                                          kind=BEPS_OP_RANGE_QUERY
//   FLUSH(max_flush_nodes=value, max_messages_total=value2)
//                                          kind=BEPS_OP_FLUSH
struct alignas(8) BepsOp {
    int32_t kind;
    int32_t pad0;
    uint64_t u_key;     // key / lo_key
    uint64_t u_key2;    // hi_key (range)
    uint64_t u_aux;     // read_id
    int64_t value;      // value/delta/limit/max_flush_nodes
    int64_t value2;     // max_messages_total (flush)
};

struct alignas(8) BepsRunSpec {
    int32_t abi_version;
    int32_t num_ops;
    int32_t step_id;
    int32_t flags;
    int32_t reserved[4];
};

struct alignas(8) BepsInputs {
    const BepsOp* ops;   // [num_ops]
};

struct alignas(8) BepsCounts {
    uint64_t root_buffered_set;
    uint64_t root_buffered_add;
    uint64_t root_buffered_del;
    uint64_t write_stall;
    uint64_t flush_groups;
    uint64_t flush_messages;
    uint64_t leaf_apply_set;
    uint64_t leaf_apply_add;
    uint64_t leaf_apply_del;
    uint64_t leaf_splits;
    uint64_t internal_splits;
    uint64_t point_found;
    uint64_t point_missing;
    uint64_t range_results;
    uint64_t range_end_count;
    uint64_t flush_empty;
    uint64_t invalid_count;
    uint64_t reserved;
};

struct alignas(8) BepsOutputs {
    BepsCounts* counts;            // [1] cumulative
    uint64_t* message_event_hash;  // [1]
    uint64_t* query_hash;          // [1]
    uint64_t* tree_shape_hash;     // [1]
    uint64_t* buffer_hash;         // [1]
    uint64_t* leaf_hash;           // [1]
};

static_assert(sizeof(BepsProblemSpec) == 64, "BepsProblemSpec layout drift");
static_assert(sizeof(BepsOp) == 48, "BepsOp layout drift");
static_assert(sizeof(BepsRunSpec) == 32, "BepsRunSpec layout drift");
static_assert(sizeof(BepsInputs) == 8, "BepsInputs layout drift");
static_assert(sizeof(BepsCounts) == 144, "BepsCounts layout drift");
static_assert(sizeof(BepsOutputs) == 48, "BepsOutputs layout drift");

static inline int beps_validate_problem_spec(const BepsProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != BEPS_ABI_VERSION) return 0;
    if (spec->max_nodes < BEPS_MIN_NODES || spec->max_nodes > BEPS_MAX_NODES) return 0;
    if (spec->internal_buffer_cap < 1 ||
        spec->internal_buffer_cap > BEPS_MAX_INTERNAL_BUFFER_CAP) return 0;
    if (spec->leaf_record_cap < 1 ||
        spec->leaf_record_cap > BEPS_MAX_LEAF_RECORD_CAP) return 0;
    if (spec->max_children_per_internal < 2 ||
        spec->max_children_per_internal > BEPS_MAX_CHILDREN) return 0;
    if (spec->flush_message_cap < 1 ||
        spec->flush_message_cap > BEPS_MAX_FLUSH_MESSAGE_CAP) return 0;
    if (spec->max_range_results < 0 ||
        spec->max_range_results > BEPS_MAX_RANGE_RESULTS) return 0;
    if (spec->max_ops < 0 || spec->max_ops > BEPS_MAX_OPS) return 0;
    if (spec->max_steps < 0 || spec->max_steps > BEPS_MAX_STEPS) return 0;
    return 1;
}

static inline int beps_validate_run_spec(const BepsRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != BEPS_ABI_VERSION) return 0;
    if (run->num_ops < 0 || run->num_ops > BEPS_MAX_OPS) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const BepsProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const BepsProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const BepsRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // BEPS_TREE_BUFFER_COMMON_H_
