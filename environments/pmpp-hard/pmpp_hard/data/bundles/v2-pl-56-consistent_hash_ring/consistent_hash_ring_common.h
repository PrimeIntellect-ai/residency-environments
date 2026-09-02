// file: consistent_hash_ring_common.h

#ifndef CONSISTENT_HASH_RING_COMMON_H_
#define CONSISTENT_HASH_RING_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define CHR_ABI_VERSION 1

// Capacity limits (hard ceilings used to size persistent buffers).
#define CHR_MAX_NODES        256
#define CHR_MAX_VNODES       8192
#define CHR_MAX_KEYS         4096
#define CHR_MAX_REPLICAS     8     /* hard ceiling on max_replicas_per_key */
#define CHR_MAX_MOVES        65536

// Node states.
#define CHR_NODE_JOINING 0
#define CHR_NODE_ACTIVE  1
#define CHR_NODE_LEAVING 2
#define CHR_NODE_DOWN    3
#define CHR_NODE_REMOVED 4

// Replica kinds.
#define CHR_KIND_PRIMARY 0
#define CHR_KIND_NORMAL  1
#define CHR_KIND_HINTED  2
#define CHR_KIND_NONE    255

// Move task kinds.
#define CHR_TASK_ADD_REPLICA    0
#define CHR_TASK_DROP_REPLICA   1
#define CHR_TASK_HANDOFF_HINT   2
#define CHR_TASK_DELETE_REPLICA 3

// Created reasons.
#define CHR_REASON_JOIN              0
#define CHR_REASON_LEAVE             1
#define CHR_REASON_FAIL             2
#define CHR_REASON_RECOVER          3
#define CHR_REASON_PUT_REPAIR       4
#define CHR_REASON_DELETE_REPAIR    5
#define CHR_REASON_EXPLICIT_REBALANCE 6

// Operation types (one op per solution_run).
#define CHR_OP_ADD_NODE      0   /* a=node_id, b=vnode_count, c=capacity */
#define CHR_OP_ACTIVATE_NODE 1   /* a=node_id */
#define CHR_OP_START_LEAVE   2   /* a=node_id */
#define CHR_OP_FAIL_NODE     3   /* a=node_id */
#define CHR_OP_RECOVER_NODE  4   /* a=node_id */
#define CHR_OP_REMOVE_NODE   5   /* a=node_id */
#define CHR_OP_PUT_KEY       6   /* a=key_id, b=value(i64) */
#define CHR_OP_DELETE_KEY    7   /* a=key_id */
#define CHR_OP_LOOKUP        8   /* a=read_id, b=key_id */
#define CHR_OP_REBALANCE     9   /* a=limit */

// Event kinds (emission-order hashed into ring_event_hash).
#define CHR_EV_VNODE_ADD             0
#define CHR_EV_NODE_ADD              1
#define CHR_EV_NODE_ACTIVATE         2
#define CHR_EV_NODE_LEAVE            3
#define CHR_EV_LEAVE_NO_TARGET       4
#define CHR_EV_NODE_FAIL             5
#define CHR_EV_FAIL_NO_HINT_TARGET   6
#define CHR_EV_NODE_RECOVER          7
#define CHR_EV_VNODE_REMOVE          8
#define CHR_EV_NODE_REMOVE           9
#define CHR_EV_REMOVE_STALL_REPLICAS 10
#define CHR_EV_KEY_PUT               11
#define CHR_EV_KEY_DELETE            12
#define CHR_EV_KEY_DELETE_MISS       13
#define CHR_EV_REPLICA_DIRECT_ADD    14
#define CHR_EV_MOVE_ENQUEUE          15
#define CHR_EV_MOVE_OBSOLETE         16
#define CHR_EV_MOVE_STALL            17
#define CHR_EV_REPLICA_ADD           18
#define CHR_EV_REPLICA_DROP          19
#define CHR_EV_HINT_HANDOFF_ADD      20
#define CHR_EV_HINT_DROP             21
#define CHR_EV_REPLICA_DELETE        22
#define CHR_EV_LOOKUP_MISSING        23
#define CHR_EV_LOOKUP_REPLICA        24
#define CHR_EV_LOOKUP_END            25
#define CHR_EV_INVALID               26

// Lookup record kinds.
#define CHR_LK_REPLICA 0
#define CHR_LK_END     1
#define CHR_LK_MISSING 2

/*
CONTRACT: consistent_hash_ring  (T56 — Dynamo-Style Consistent Ring)

A persistent consistent-hashing placement engine. Each solution_run applies
exactly ONE operation (the op fields in ChrRunSpec) and emits the updated
cumulative counters plus six FNV-1a-64 state checksums.

Counters are cumulative since the last solution_reset. Hashes describe the full
persistent state AFTER the op completes. All graded outputs are exact integers,
and the six hash outputs are exact FNV-1a-64 checksums. The complete normative
semantics + output serialization are inlined below in this header (search for
"=== CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===").
Both reference.cu and naive.cu must reproduce these outputs byte-for-byte.

Spec fields (ChrProblemSpec):
  replication_factor, preference_list_extra, max_nodes, max_vnodes, max_keys,
  max_replicas_per_key, max_move_tasks, default_node_capacity,
  ring_seed, key_seed.

solution_run may not call cudaMalloc/cudaFree; it launches a single-block kernel.
*/

struct alignas(8) ChrProblemSpec {
    int32_t abi_version;
    int32_t replication_factor;
    int32_t preference_list_extra;
    int32_t max_nodes;
    int32_t max_vnodes;
    int32_t max_keys;
    int32_t max_replicas_per_key;
    int32_t max_move_tasks;
    uint64_t default_node_capacity;
    uint64_t ring_seed;
    uint64_t key_seed;
    int32_t reserved[6];
};

struct alignas(8) ChrRunSpec {
    int32_t abi_version;
    int32_t op_type;
    int32_t op_index;     // monotonically increasing index of this op since reset
    int32_t reserved0;
    int64_t arg_a;        // node_id / key_id / read_id / limit
    int64_t arg_b;        // vnode_count / value / key_id
    int64_t arg_c;        // capacity
    int32_t reserved[8];
};

// Inputs carries no host arrays in this task (all op args are in ChrRunSpec),
// but is kept for ABI symmetry; pointer may be null.
struct alignas(8) ChrInputs {
    const void* reserved;
};

struct alignas(8) ChrCounters {
    uint64_t nodes_added;
    uint64_t nodes_activated;
    uint64_t nodes_leaving;
    uint64_t nodes_failed;
    uint64_t nodes_recovered;
    uint64_t nodes_removed;
    uint64_t vnode_added;
    uint64_t vnode_removed;
    uint64_t put_ok;
    uint64_t put_oom;
    uint64_t delete_ok;
    uint64_t delete_miss;
    uint64_t direct_replica_added;
    uint64_t move_tasks_enqueued;
    uint64_t move_obsolete;
    uint64_t move_stall;
    uint64_t replica_added;
    uint64_t replica_dropped;
    uint64_t hint_handoff_added;
    uint64_t hint_dropped;
    uint64_t replica_deleted;
    uint64_t lookup_found_replicas;
    uint64_t lookup_missing;
    uint64_t remove_stalls;
    uint64_t no_target_count;
    uint64_t invalid_count;
};

struct alignas(8) ChrOutputs {
    ChrCounters* counters;       // cumulative counters (1 element)
    uint64_t* ring_event_hash;   // emission-order event hash (cumulative since reset)
    uint64_t* lookup_hash;       // op-order lookup-record hash (cumulative since reset)
    uint64_t* ring_hash;         // virtual tokens in ring order (current state)
    uint64_t* node_hash;         // nodes by node_id ascending (current state)
    uint64_t* key_replica_hash;  // keys by key_hash,key_id; replicas by rank,node (current)
    uint64_t* move_hash;         // move tasks by move_seq (current state)
};

static_assert(sizeof(ChrProblemSpec) == 80, "ChrProblemSpec layout drift");
static_assert(sizeof(ChrRunSpec) == 72, "ChrRunSpec layout drift");
static_assert(sizeof(ChrCounters) == 26 * 8, "ChrCounters layout drift");
static_assert(sizeof(ChrOutputs) == 7 * 8, "ChrOutputs layout drift");

static inline int chr_validate_problem_spec(const ChrProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != CHR_ABI_VERSION) return 0;
    if (spec->replication_factor < 1 || spec->replication_factor > CHR_MAX_REPLICAS) return 0;
    if (spec->preference_list_extra < 0 || spec->preference_list_extra > CHR_MAX_NODES) return 0;
    if (spec->max_nodes < 1 || spec->max_nodes > CHR_MAX_NODES) return 0;
    if (spec->max_vnodes < 1 || spec->max_vnodes > CHR_MAX_VNODES) return 0;
    if (spec->max_keys < 1 || spec->max_keys > CHR_MAX_KEYS) return 0;
    if (spec->max_replicas_per_key < spec->replication_factor ||
        spec->max_replicas_per_key > CHR_MAX_REPLICAS) return 0;
    if (spec->max_move_tasks < 1 || spec->max_move_tasks > CHR_MAX_MOVES) return 0;
    if (spec->default_node_capacity == 0) return 0;
    return 1;
}

static inline int chr_validate_run_spec(const ChrRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != CHR_ABI_VERSION) return 0;
    if (run->op_type < 0 || run->op_type > CHR_OP_REBALANCE) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const ChrProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const ChrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const ChrRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

// === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section is the COMPLETE, self-contained specification. Everything that a
// graded solution must reproduce is spelled out here exactly as the reference
// computes it. Nothing is deferred to any held-out file. All multi-byte values
// are hashed as LITTLE-ENDIAN raw bytes (x86/CUDA native), via FNV-1a-64.
//
// --------------------------------------------------------------------------
// 0. FNV-1a-64 PRIMITIVE (used for ALL hashes: positions, key hashes, and the
//    six output checksums).
// --------------------------------------------------------------------------
//   offset basis = 1469598103934665603  (0x14650FB0739D0383)
//   prime        = 1099511628211        (0x 00000100000001b3)
//   Per byte b (processed in increasing memory address order, i.e. LE for an
//   integer): h = (h XOR b) * prime   (all arithmetic mod 2^64, unsigned wrap).
//   fnv_u64(h,x): fold the 8 LE bytes of x.   fnv_u32(h,x): 4 LE bytes.
//   fnv_u8(h,x): 1 byte.   fnv_i64(h,x): the int64 reinterpreted as 8 LE bytes.
//   "Start a hash" = initialize accumulator to the offset basis above, then fold.
//
// --------------------------------------------------------------------------
// 1. THE TWO RING/KEY HASH FORMULAS (EXACT). These place tokens on the ring and
//    locate keys. The argument order is seed FIRST, then the id fields.
// --------------------------------------------------------------------------
//   token_pos (virtual-node ring position) for (node_id, ordinal):
//       h = OFFSET_BASIS
//       h = fnv_u64(h, ring_seed)      // spec.ring_seed, 8 LE bytes
//       h = fnv_u64(h, node_id)        // 8 LE bytes
//       h = fnv_u64(h, (uint64_t)ord)  // ordinal 0..vnode_count-1, 8 LE bytes
//       token_pos = h
//     i.e. token_pos = FNV1a64(OFFSET_BASIS; ring_seed, node_id, ordinal).
//
//   key_hash (lookup position of a key) for key_id:
//       h = OFFSET_BASIS
//       h = fnv_u64(h, key_seed)       // spec.key_seed, 8 LE bytes
//       h = fnv_u64(h, key_id)         // 8 LE bytes
//       key_hash = h
//     i.e. key_hash = FNV1a64(OFFSET_BASIS; key_seed, key_id).
//
//   COLLISION RULE for token_pos: after computing token_pos, while ANY existing
//   token (across all nodes) already has that exact token_pos, do token_pos += 1
//   (unsigned wrap mod 2^64). Probe steps emit NO event. The final (free)
//   position is stored. Ordinals are assigned 0,1,...,vnode_count-1 in order.
//
// --------------------------------------------------------------------------
// 2. CANONICAL ORDERS (used by semantics AND by output serialization).
// --------------------------------------------------------------------------
//   RING ORDER (token traversal order): ascending lexicographic over the tuple
//       (token_pos, token_seq, node_id, vnode_ordinal), all UNSIGNED.
//       This is the tie-break when two tokens hash to the same ring position:
//       lower token_seq first (i.e. earlier insertion), then lower node_id,
//       then lower vnode_ordinal.
//   KEYS ORDER: ascending (key_hash, then key_id), both unsigned.
//   NODES ORDER: ascending node_id.
//   MOVES ORDER: ascending move_seq (== append order).
//   REPLICA ORDER for key_replica_hash and lookup tie-breaks: see sections below.
//
// --------------------------------------------------------------------------
// 3. SEQUENCE COUNTERS (monotonic, reset by solution_reset). After reset:
//    event_seq=0; node_seq_next=1; token_seq_next=1; key_seq_next=1;
//    version_seq_next=1; move_seq_next=1. Each *_next is consumed (post-
//    increment) when a node/token/key/version/move is created. event_seq is
//    PRE-incremented inside emit() (so the first emitted event has event_seq=1).
//
// --------------------------------------------------------------------------
// 4. PREFERENCE LIST + DESIRED SET (the "walk to N successors" logic).
// --------------------------------------------------------------------------
//   preference_list(key_hash):
//     - If there are no tokens, empty.
//     - Compute RING ORDER. Find start = first index whose token_pos >= key_hash
//       (clockwise from key_hash). If none (key_hash above all positions), wrap:
//       start = 0.
//     - cap = replication_factor + preference_list_extra.
//     - Walk clockwise (start+step) % m for step=0..m-1 (exactly one full wrap):
//         look up the token's physical node_id; require node found AND state in
//         {ACTIVE, JOINING}; skip otherwise. Skip DUPLICATE physical node_ids
//         (only first occurrence counts). Append node_id. Stop once cap distinct
//         nodes collected (or after the full single wrap).
//     Result is the ordered list of DISTINCT physical node ids.
//   desired_replicas(key_hash):
//     - Walk preference_list in order; keep only nodes whose state == ACTIVE;
//       stop after replication_factor of them. (JOINING nodes occupy pref slots
//       but are NOT desired servers.) The 0-based position in this ACTIVE-only
//       sequence is the replica's "rank". rank 0 => kind PRIMARY, rank>0 =>
//       kind NORMAL (HINTED is set only via hint_for_node, see ops).
//   first_free_rank(key, desired): smallest r in [0, |desired|) such that NO
//     serving replica of the key currently has stored rank == r; if all such
//     ranks are occupied, returns the count of serving replicas of the key.
//   node_full(node): used_slots >= capacity. A serving replica consumes exactly
//     one used_slot on its physical node (HINTED replicas included).
//
// --------------------------------------------------------------------------
// 5. PER-OP SEMANTICS. op_index = run.op_index. emit(kind,...) folds an event
//    (section 9) into ring_event_hash and bumps event_seq. "invalid" =>
//    invalid_count += 1 and emit(CHR_EV_INVALID, op_index, node=0, key=0,
//    token_pos=UINT64_MAX, rank=UINT32_MAX, value=INT64_MIN, aux=0).
// --------------------------------------------------------------------------
//   ADD_NODE (a=node_id, b=vnode_count, c=capacity):
//     INVALID if: a non-REMOVED node with this id exists; OR node_count >=
//     max_nodes; OR vnode_count == 0; OR vnode_count < 0; OR
//     token_count + vnode_count > max_vnodes. Otherwise:
//       cap = (capacity==0 ? default_node_capacity : capacity).
//       If a REMOVED record with this id exists, reuse/overwrite that slot; else
//       append a new node. Set state=JOINING, used_slots=0, last_state_seq=0,
//       node_seq = node_seq_next++. For ord=0..vnode_count-1: compute token_pos
//       (section 1 + collision rule), token_seq = token_seq_next++, store token,
//       vnode_added += 1, emit(CHR_EV_VNODE_ADD, op_index, node_id, key=0,
//       token_pos=pos, rank=ord, value=0, aux=token_seq). Then nodes_added += 1,
//       emit(CHR_EV_NODE_ADD, op_index, node_id, key=0, token_pos=UINT64_MAX,
//       rank=UINT32_MAX, value=0, aux=node_seq).
//
//   ACTIVATE_NODE (a=node_id):
//     INVALID unless node exists AND state == JOINING. Set state=ACTIVE;
//     nodes_activated += 1; emit(CHR_EV_NODE_ACTIVATE, op_index, node_id,0,
//     UINT64_MAX,UINT32_MAX,0,0); last_state_seq = event_seq (post-emit).
//     Then for each non-deleted key in KEYS ORDER: if node_id is in
//     desired_replicas(key_hash) and not already serving the key, enqueue_move
//     ADD_REPLICA to=node_id, rank = node_id's desired rank, hint_for=0,
//     target_version=key.version_seq, reason=JOIN.
//
//   START_LEAVE (a=node_id):
//     INVALID unless node exists AND state == ACTIVE. Set state=LEAVING;
//     nodes_leaving += 1; emit(CHR_EV_NODE_LEAVE,...); last_state_seq=event_seq.
//     For each non-deleted key in KEYS ORDER currently SERVED by node_id: let
//     des = desired_replicas; target = first des[r] (rank order) not currently
//     serving the key. If none: no_target_count += 1; emit(CHR_EV_LEAVE_NO_TARGET,
//     op_index, node_id, key_id, UINT64_MAX, UINT32_MAX,0,0). Else enqueue_move
//     ADD_REPLICA to=target, rank=r, hint_for=0, target_version=version_seq,
//     reason=LEAVE.
//
//   FAIL_NODE (a=node_id):
//     INVALID if node missing OR state in {DOWN, REMOVED}. Set state=DOWN;
//     nodes_failed += 1; emit(CHR_EV_NODE_FAIL,...); last_state_seq=event_seq.
//     PHASE 1 (KEYS ORDER): for each key with a replica record on node_id that
//     is serving: set serving=0 and, if node used_slots>0, used_slots -= 1.
//     PHASE 2 (KEYS ORDER): for each non-deleted key that still has a replica
//     RECORD on node_id: target = first node in preference_list that is ACTIVE
//     and not already serving the key. If none: no_target_count += 1;
//     emit(CHR_EV_FAIL_NO_HINT_TARGET, op_index, node_id, key_id,...). Else
//     rank = first_free_rank(key, desired_replicas); enqueue_move ADD_REPLICA
//     to=target, rank, hint_for_node=node_id (the failed node), reason=FAIL.
//     (hint_for_node != 0 => the installed replica becomes kind HINTED.)
//
//   RECOVER_NODE (a=node_id):
//     INVALID unless node exists AND state == DOWN. Set state=ACTIVE;
//     nodes_recovered += 1; emit(CHR_EV_NODE_RECOVER,...); last_state_seq=
//     event_seq. For each key in KEYS ORDER, for each of its replicas in
//     STORED order with kind==HINTED and hint_for_node==node_id: enqueue_move
//     HANDOFF_HINT from=replica.node_id, to=node_id, rank=replica.rank,
//     hint_for_node=node_id, target_version=version_seq, reason=RECOVER.
//
//   REMOVE_NODE (a=node_id):
//     INVALID unless node exists AND state == LEAVING. If ANY key (scan in key
//     storage order) still has a SERVING replica on node_id: remove_stalls += 1;
//     emit(CHR_EV_REMOVE_STALL_REPLICAS, op_index, node_id,...); RETURN (node
//     stays LEAVING). Else: in RING ORDER, for each token of node_id:
//     vnode_removed += 1; emit(CHR_EV_VNODE_REMOVE, op_index, node_id, key=0,
//     token_pos, rank=vnode_ordinal, value=0, aux=token_seq). Then erase all of
//     node_id's tokens (token_count shrinks). Set state=REMOVED; nodes_removed
//     += 1; emit(CHR_EV_NODE_REMOVE,...); last_state_seq=event_seq.
//
//   PUT_KEY (a=key_id, b=value i64):
//     If key absent AND key_count >= max_keys: put_oom += 1; RETURN (no event).
//     If absent: create key with key_hash (section 1), value, version_seq=0,
//     deleted=0, key_seq = key_seq_next++, no replicas. Then (existing or new):
//     deleted=0; value=b; version_seq = version_seq_next++; put_ok += 1;
//     emit(CHR_EV_KEY_PUT, op_index, node=0, key_id, UINT64_MAX, UINT32_MAX,
//     value, aux=version_seq). Let serving_count = #serving replicas.
//       If serving_count == 0: for r=0..|desired|-1, nid=desired[r]: require
//       node ACTIVE and not full; install a NEW serving replica rank=r,
//       kind=(r==0?PRIMARY:NORMAL), hint_for=0, serving=1; emit
//       (CHR_EV_REPLICA_DIRECT_ADD, op_index, nid, key_id, UINT64_MAX, r, 0,
//       version_seq); repair_seq=event_seq; node.used_slots += 1;
//       direct_replica_added += 1.
//       Else (serving_count>0): for r=0..|desired|-1, if desired[r] not serving,
//       enqueue_move ADD_REPLICA to=desired[r], rank=r, hint_for=0,
//       target_version=version_seq, reason=PUT_REPAIR.
//
//   DELETE_KEY (a=key_id):
//     If key missing OR already deleted: delete_miss += 1; emit
//     (CHR_EV_KEY_DELETE_MISS, op_index, 0, key_id,...); RETURN. Else deleted=1;
//     version_seq = version_seq_next++; delete_ok += 1; emit(CHR_EV_KEY_DELETE,
//     op_index, 0, key_id, UINT64_MAX, UINT32_MAX, 0, version_seq). Then for
//     EVERY replica record, in ASCENDING node_id order, enqueue_move
//     DELETE_REPLICA from=replica.node_id, to=0, rank=replica.rank,
//     hint_for_node=replica.hint_for_node, target_version=version_seq,
//     reason=DELETE_REPAIR. (node_ids are unique per key.)
//
//   LOOKUP (a=read_id, b=key_id): also folds into lookup_hash (section 10).
//     If key missing OR deleted: lookup_missing += 1; emit(CHR_EV_LOOKUP_MISSING,
//     op_index, 0, key_id,...); lk_record(read_id, key_id, CHR_LK_MISSING,
//     rank=UINT32_MAX, node_id=0, kind=CHR_KIND_NONE, hint_for=0, value=INT64_MIN,
//     version=UINT64_MAX); RETURN. Else: order SERVING replicas by (rank asc,
//     kind asc, node_id asc) and emit the first replication_factor of them:
//     for each, lookup_found_replicas += 1; emit(CHR_EV_LOOKUP_REPLICA, op_index,
//     replica.node_id, key_id, UINT64_MAX, replica.rank, value=key.value,
//     aux=version_seq); lk_record(read_id, key_id, CHR_LK_REPLICA, replica.rank,
//     replica.node_id, replica.kind, replica.hint_for_node, key.value,
//     version_seq). After the loop: emit(CHR_EV_LOOKUP_END, op_index, 0, key_id,
//     UINT64_MAX, UINT32_MAX,0,0); lk_record(read_id, key_id, CHR_LK_END,
//     UINT32_MAX, 0, CHR_KIND_NONE, 0, INT64_MIN, UINT64_MAX).
//
//   REBALANCE (a=limit i64): if limit==0, valid no-op (RETURN). Otherwise scan
//     the moves queue from i=0. For moves[i] call process_task (below). If it
//     STALLS (kept): ++i, continue. If it is REMOVED: erase moves[i] (do NOT
//     advance i; later-appended tasks stay at the tail). If the removal was
//     "counted" (an actual ADD/DROP/HANDOFF/DELETE completion, not an obsolete
//     skip): if the key still exists, run enqueue_drop_extras for it, then
//     ++completed. Stop once completed >= limit or the queue end is reached.
//     enqueue_drop_extras(key): des = desired_replicas; extras = (#serving) -
//     replication_factor; if extras<=0 nothing. Else pick `extras` serving
//     replicas ordered by (rank DESC, then node_id ASC) and enqueue_move
//     DROP_REPLICA from=replica.node_id, to=0, rank=replica.rank, hint_for=0,
//     target_version=version_seq, reason=EXPLICIT_REBALANCE.
//
//   process_task(mv) result classes: STALL(keep), REMOVE-uncounted(obsolete),
//   REMOVE-counted(completed). Per task_kind:
//     ADD_REPLICA: if key missing/deleted OR key.version_seq != mv.target_version
//       => move_obsolete += 1; emit(CHR_EV_MOVE_OBSOLETE, op, mv.to_node, key,
//       UINT64_MAX, mv.rank, 0, mv.move_seq); REMOVE-uncounted. Else let dest =
//       mv.to_node; dest_has = (replica record already on dest). If node missing
//       OR not ACTIVE OR (full AND !dest_has) => move_stall += 1; emit
//       (CHR_EV_MOVE_STALL,...); STALL. Else install: kind = (mv.hint_for_node
//       ? HINTED : (mv.rank==0 ? PRIMARY : NORMAL)). If a record on dest exists:
//       if it was non-serving add a used_slot; set rank/kind/hint_for/serving=1;
//       emit(CHR_EV_REPLICA_ADD, op, dest, key, UINT64_MAX, mv.rank, 0,
//       version_seq); repair_seq=event_seq. Else append new serving replica;
//       emit(CHR_EV_REPLICA_ADD,...); used_slots += 1. replica_added += 1;
//       REMOVE-counted.
//     DROP_REPLICA: if key missing => REMOVE-uncounted. If a SERVING replica on
//       mv.from_node exists: if node used_slots>0, used_slots -= 1; emit
//       (CHR_EV_REPLICA_DROP, op, mv.from_node, key, UINT64_MAX, replica.rank, 0,
//       version_seq); erase the replica; replica_dropped += 1. REMOVE-counted
//       (counted regardless of whether a serving replica was found).
//     HANDOFF_HINT: same obsolete/stall checks as ADD_REPLICA (using mv.to_node).
//       Install on dest with kind=(mv.rank==0?PRIMARY:NORMAL), hint_for=0,
//       serving=1; emit(CHR_EV_HINT_HANDOFF_ADD,...); repair_seq=event_seq;
//       used_slots accounting as ADD. hint_handoff_added += 1. Then if mv.from_node
//       still has a HINTED replica: if serving and source used_slots>0,
//       used_slots -= 1; emit(CHR_EV_HINT_DROP, op, mv.from_node, key,
//       UINT64_MAX, src.rank, 0, version_seq); erase it; hint_dropped += 1.
//       REMOVE-counted.
//     DELETE_REPLICA: if key exists and a replica record on mv.from_node exists:
//       if serving and node used_slots>0, used_slots -= 1; emit
//       (CHR_EV_REPLICA_DELETE, op, mv.from_node, key, UINT64_MAX, replica.rank,
//       0, version_seq); erase it; replica_deleted += 1. REMOVE-counted.
//     enqueue_move ALWAYS: move_seq = move_seq_next++; append; move_tasks_enqueued
//       += 1; emit(CHR_EV_MOVE_ENQUEUE, op, node=(to_node?to_node:from_node),
//       key, UINT64_MAX, rank, 0, aux=move_seq). If the queue is full
//       (move_count >= max_move_tasks) the enqueue is silently dropped (no
//       counter, no event) — reference guards this in enqueue_move.
//
// --------------------------------------------------------------------------
// 6. COUNTER (ChrCounters) FIELD MEANINGS / GRADED INDEX. The struct is compared
//    field-by-field (all u64, cumulative since reset). Index = struct order:
//      [0]  nodes_added            (+1 per successful ADD_NODE)
//      [1]  nodes_activated        (+1 per successful ACTIVATE_NODE)
//      [2]  nodes_leaving          (+1 per successful START_LEAVE)
//      [3]  nodes_failed           (+1 per successful FAIL_NODE)
//      [4]  nodes_recovered        (+1 per successful RECOVER_NODE)
//      [5]  nodes_removed          (+1 per successful REMOVE_NODE)
//      [6]  vnode_added            (+1 per virtual token created in ADD_NODE)
//      [7]  vnode_removed          (+1 per token removed in REMOVE_NODE)
//      [8]  put_ok                 (+1 per PUT_KEY that stores)
//      [9]  put_oom                (+1 per PUT_KEY rejected for full key table)
//      [10] delete_ok             (+1 per DELETE_KEY that deletes)
//      [11] delete_miss           (+1 per DELETE_KEY missing/already-deleted)
//      [12] direct_replica_added  (+1 per direct serving replica in PUT_KEY)
//      [13] move_tasks_enqueued   (+1 per enqueue_move)
//      [14] move_obsolete         (+1 per task dropped obsolete in process_task)
//      [15] move_stall            (+1 per ADD/HANDOFF task that stalls)
//      [16] replica_added         (+1 per completed ADD_REPLICA)
//      [17] replica_dropped       (+1 per DROP_REPLICA that removed a serving rep)
//      [18] hint_handoff_added    (+1 per completed HANDOFF_HINT)
//      [19] hint_dropped          (+1 per hinted source replica removed in handoff)
//      [20] replica_deleted       (+1 per DELETE_REPLICA that removed a record)
//      [21] lookup_found_replicas (+1 per LOOKUP_REPLICA emitted)
//      [22] lookup_missing        (+1 per LOOKUP on missing/deleted key)
//      [23] remove_stalls         (+1 per REMOVE_NODE stalled by serving replicas)
//      [24] no_target_count       (+1 per LEAVE_NO_TARGET and per FAIL_NO_HINT_TARGET)
//      [25] invalid_count         (+1 per invalid op / unknown op_type)
//
// --------------------------------------------------------------------------
// 7. ring_hash SERIALIZATION (current ring state). Start FNV. Iterate tokens in
//    RING ORDER (section 2). Per token fold: fnv_u64(token_pos), fnv_u64(token_seq),
//    fnv_u64(node_id), fnv_u32(vnode_ordinal). No separators.
//
// 8. node_hash SERIALIZATION. Start FNV. Iterate nodes in NODES ORDER (node_id
//    asc). Per node fold: fnv_u64(node_id), fnv_u8((uint8_t)state),
//    fnv_u64(capacity), fnv_u64(used_slots), fnv_u64(node_seq),
//    fnv_u64(last_state_seq). (REMOVED nodes remain in the table and ARE hashed.)
//
// 8b. key_replica_hash SERIALIZATION. Start FNV. Iterate keys in KEYS ORDER
//    (key_hash asc, then key_id asc). Per key fold the key fields in this order:
//    fnv_u64(key_id), fnv_u64(key_hash), fnv_i64(value), fnv_u64(version_seq),
//    fnv_u8(deleted), fnv_u64(key_seq). Then iterate that key's replicas in
//    (rank ASC, then node_id ASC) order; per replica fold: fnv_u32(rank),
//    fnv_u64(node_id), fnv_u8(kind), fnv_u64(hint_for_node), fnv_u8(serving),
//    fnv_u64(repair_seq). (Deleted keys and their replicas are still hashed.)
//
// 8c. move_hash SERIALIZATION. Start FNV. Iterate moves in MOVES ORDER
//    (move_seq asc == append order). Per move fold: fnv_u64(move_seq),
//    fnv_u64(key_id), fnv_u8(task_kind), fnv_u64(from_node), fnv_u64(to_node),
//    fnv_u32(rank), fnv_u64(hint_for_node), fnv_u64(target_version_seq),
//    fnv_u8(created_reason).
//
// --------------------------------------------------------------------------
// 9. ring_event_hash SERIALIZATION (cumulative emission stream). Initialized to
//    OFFSET_BASIS at reset; never re-initialized between ops. emit() does, IN
//    ORDER: event_seq += 1; then fold fnv_u8(kind), fnv_u64(event_seq),
//    fnv_u32(op_index), fnv_u64(node_id), fnv_u64(key_id), fnv_u64(token_pos),
//    fnv_u32(rank), fnv_i64(value), fnv_u64(aux). Sentinels: token_pos=UINT64_MAX,
//    rank=UINT32_MAX, value=INT64_MIN mark "not applicable". The aux field carries
//    node_seq / token_seq / version_seq / move_seq depending on the event.
//
// 10. lookup_hash SERIALIZATION (cumulative LOOKUP record stream). Initialized to
//    OFFSET_BASIS at reset; never re-initialized. Each lk_record folds, IN ORDER:
//    fnv_u64(read_id), fnv_u64(key_id), fnv_u8(record_kind in {CHR_LK_REPLICA,
//    CHR_LK_END, CHR_LK_MISSING}), fnv_u32(rank), fnv_u64(node_id), fnv_u8(kind),
//    fnv_u64(hint_for), fnv_i64(value), fnv_u64(version_seq). MISSING/END records
//    use rank=UINT32_MAX, node_id=0, kind=CHR_KIND_NONE, hint_for=0,
//    value=INT64_MIN, version=UINT64_MAX.
//
// --------------------------------------------------------------------------
// 11. The six hashes are ALL seeded with the same FNV offset basis (section 0).
//    counters[] is compared field-by-field as in section 6. No field is omitted;
//    no extra padding bytes are hashed (each field is folded explicitly).
// === END CONTRACT ===

#endif  // CONSISTENT_HASH_RING_COMMON_H_
