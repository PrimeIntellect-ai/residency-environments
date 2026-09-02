// file: rcu_epoch_reclaimer_common.h

#ifndef RCU_EPOCH_RECLAIMER_COMMON_H_
#define RCU_EPOCH_RECLAIMER_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define RCU_ABI_VERSION 1

// Limits (compile-time hard caps used by validation and fixed buffers).
#define RCU_MIN_THREADS 1
#define RCU_MAX_THREADS 256
#define RCU_MIN_ROOTS 1
#define RCU_MAX_ROOTS 256
#define RCU_MIN_OBJECTS 1
#define RCU_MAX_OBJECTS 4096
#define RCU_MIN_EDGES 1
#define RCU_MAX_EDGES 16
#define RCU_MAX_RETIRED 4096
#define RCU_MAX_FREE_IDS 4096
#define RCU_MAX_OPS 8192

// Operation opcodes (op.kind).
#define RCU_OP_ALLOC 0
#define RCU_OP_SET_ROOT 1
#define RCU_OP_LINK 2
#define RCU_OP_READ_LOCK 3
#define RCU_OP_READ_UNLOCK 4
#define RCU_OP_QUIESCE 5
#define RCU_OP_ADVANCE_EPOCH 6
#define RCU_OP_READ_CHAIN 7
#define RCU_OP_RETIRE 8
#define RCU_OP_RECLAIM 9
#define RCU_OP_FORCE_DROP_FREE_IDS 10

// Event kinds (must match emission order in the contract).
#define RCU_EV_ALLOC_OK 0
#define RCU_EV_ALLOC_OOM 1
#define RCU_EV_ROOT_SET 2
#define RCU_EV_EDGE_SET 3
#define RCU_EV_READ_LOCK_OK 4
#define RCU_EV_READ_UNLOCK_QUIESCENT 5
#define RCU_EV_READ_UNLOCK_NESTED 6
#define RCU_EV_QUIESCE_OK 7
#define RCU_EV_EPOCH_ADVANCE 8
#define RCU_EV_READ_NODE 9
#define RCU_EV_READ_EMPTY 10
#define RCU_EV_RETIRE_OK 11
#define RCU_EV_RETIRE_REJECT_FULL 12
#define RCU_EV_RCU_CALLBACK 13
#define RCU_EV_OBJECT_RECLAIM 14
#define RCU_EV_FREE_ID_DROPPED 15
#define RCU_EV_INVALID 16

/*
CONTRACT: rcu_epoch_reclaimer

A persistent RCU-like object graph. Each solution_run processes a batch of
operations against persistent state and emits exact-integer counts and
order-sensitive FNV-1a-64 checksums over the event stream plus structural
checksums over the final state.

PERSISTENT STATE
  event_seq        u64 (= 0 at reset). Incremented and assigned to each emitted
                   event BEFORE that event is hashed. The first event of the
                   whole run carries event_seq = 1.
  obj_id_next      u64 (= 1)
  read_seq_next    u64 (= 1)
  retire_seq_next  u64 (= 1)
  reclaim_seq_next u64 (= 1)
  global_epoch     u64 (= 0)

  root[root_id]      : 0 or object id, root_id in [0, root_count)
  thread[t]          : read_nest:u64, outer_read_seq:u64, active_epoch:u64,
                       root_view[root_count]
                       (read_nest == 0  =>  outer_read_seq = UINT64_MAX)
  object table keyed by obj_id (live and retired-but-unreclaimed):
    present:u8, value:i64, edges[max_edges_per_object] (0 or obj id),
    alloc_seq:u64, retired:u8, retire_seq:u64 (UINT64_MAX if not retired),
    retire_epoch:u64 (UINT64_MAX if not retired), callback_tag:u32

  retired queue : object ids with retired=1, canonical order
                  (retire_epoch, retire_seq, obj_id)
  free-id list  : (obj_id, reclaim_seq), canonical order (reclaim_seq, obj_id)

// === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
// The following fully specifies every enforced rule. Nothing is deferred to
// external files. All of it is derived directly from the reference and must be
// reproduced bit-for-bit. Every checksum is FNV-1a-64 (offset
// 1469598103934665603, prime 1099511628211), hashing the raw little-endian
// bytes of each listed field, in the listed order.
//
// GENERAL CONVENTIONS
//   - The kernel processes ops in array order, single-threaded, deterministic.
//   - op_index is a persistent u32 counter starting at 0 at reset; it is the
//     value BEFORE processing the op, and is incremented by 1 after each op
//     (every op, including invalid ones).
//   - All object-id args are read as (uint64_t)(uint32_t)arg (i.e. arg
//     reinterpreted as unsigned 32-bit then widened). Object id 0 = null.
//   - RCU_REF_ID_CAP = RCU_MAX_OBJECTS + RCU_MAX_OPS + 4. An id is "present"
//     iff id != 0 && id < RCU_REF_ID_CAP && o_present[id] != 0. An id is "live"
//     iff present and o_retired[id] == 0.
//   - rref_emit_event: BEFORE hashing, event_seq (sc[0]) += 1. Then the event
//     record (see rcu_event_hash below) is folded into event_hash.
//   - INVALID handling (rref_emit_invalid): emits an event with
//       kind=RCU_EV_INVALID, thread=UINT32_MAX, obj=UINT64_MAX,
//       root=UINT32_MAX, slot=UINT32_MAX, epoch=current global_epoch (sc[5]),
//       aux=opcode (the op.kind, or the offending RCU_OP_* constant), and
//       increments counts[RCU_COUNT_INVALID]. Any unknown op.kind emits INVALID
//       with aux=op.kind.
//
// OPERATIONS (exact semantics)
//
//   ALLOC(request_id=arg_a, value=arg_i64):
//     request_id = (uint64_t)(uint32_t)arg_a; value = arg_i64.
//     full := present_total >= spec.max_objects; fc := free_count.
//     If full && fc==0: emit ALLOC_OOM (obj=UINT64_MAX, epoch=sc[5],
//       aux=request_id); counts[ALLOC_OOM]++; return.
//     Else choose an id: if fc>0, sort the free list canonically
//       (reclaim_seq asc, then obj_id asc), take free_obj[0], pop it from the
//       front (shift remaining down), free_count--. Else chosen = obj_id_next
//       (sc[1]); obj_id_next += 1.
//     Emit ALLOC_OK (obj=chosen, epoch=sc[5], aux=request_id). alloc_seq :=
//       sc[0] (the event_seq just assigned). Initialize object: present=1,
//       value=value, alloc_seq=alloc_seq, retired=0, retire_seq=UINT64_MAX,
//       retire_epoch=UINT64_MAX, callback_tag=0, all edges=0.
//       present_total++; counts[ALLOC_OK]++.
//
//   SET_ROOT(root_id=arg_a, obj_id=arg_b):
//     obj = (uint64_t)(uint32_t)arg_b. If root_id<0 || root_id>=root_count:
//       INVALID(RCU_OP_SET_ROOT). If obj!=0 && !live(obj): INVALID. Else
//       root[root_id]=obj; emit ROOT_SET (obj=obj, root=root_id, epoch=sc[5],
//       aux=0); counts[ROOT_SETS]++.
//
//   LINK(src_obj=arg_a, slot=arg_b, dst_obj=arg_c):
//     src=(u64)(u32)arg_a, dst=(u64)(u32)arg_c. If !live(src): INVALID. If
//     slot<0 || slot>=max_edges_per_object: INVALID. If dst!=0 && !live(dst):
//     INVALID. Else edges[src][slot]=dst; emit EDGE_SET (obj=src,
//     slot=slot, epoch=sc[5], aux=dst); counts[EDGE_SETS]++.
//
//   READ_LOCK(thread=arg_a):
//     t=arg_a. If t<0||t>=thread_count: INVALID. If read_nest[t]==0 (outermost
//     lock): snapshot root view (root_view[t][r]=root[r] for all r);
//     outer_read_seq[t]=read_seq_next (sc[2]); read_seq_next+=1;
//     active_epoch[t]=sc[5]. If read_nest[t]+1 would overflow to 0: INVALID.
//     read_nest[t]+=1. Emit READ_LOCK_OK (thread=t, obj=UINT64_MAX,
//     epoch=active_epoch[t], aux=outer_read_seq[t]); counts[READ_LOCKS]++.
//
//   READ_UNLOCK(thread=arg_a):
//     t=arg_a. If t<0||t>=thread_count: INVALID. If read_nest[t]==0: INVALID.
//     read_nest[t]-=1. If now 0 (quiescent): outer_read_seq[t]=UINT64_MAX;
//       active_epoch[t]=sc[5]; emit READ_UNLOCK_QUIESCENT (thread=t,
//       obj=UINT64_MAX, epoch=active_epoch[t], aux=0). Else emit
//       READ_UNLOCK_NESTED (thread=t, obj=UINT64_MAX, epoch=active_epoch[t],
//       aux=read_nest[t]). counts[READ_UNLOCKS]++.
//
//   QUIESCE(thread=arg_a):
//     t=arg_a. If t<0||t>=thread_count: INVALID. If read_nest[t]!=0: INVALID.
//     active_epoch[t]=sc[5]; emit QUIESCE_OK (thread=t, obj=UINT64_MAX,
//     epoch=active_epoch[t], aux=0); counts[QUIESCES]++.
//
//   ADVANCE_EPOCH():
//     global_epoch (sc[5]) += 1; emit EPOCH_ADVANCE (obj=UINT64_MAX,
//     epoch=sc[5] [the new epoch], aux=0); counts[EPOCH_ADVANCES]++.
//
//   READ_CHAIN(thread=arg_a, root_id=arg_b, max_hops=arg_c):
//     t=arg_a, root_id=arg_b, max_hops=arg_c. If t<0||t>=thread_count ||
//       root_id<0||root_id>=root_count: INVALID. If read_nest[t]==0: INVALID.
//     cur := root_view[t][root_id] (the snapshot taken at outermost lock).
//     epoch := active_epoch[t].
//     If cur==0 || !present(cur): emit READ_EMPTY (thread=t, obj=UINT64_MAX,
//       root=root_id, epoch=epoch, aux=0); counts[READ_EMPTY]++; return.
//     pos := 0; hop_cap := (max_hops<0)?0:(int64)max_hops. Loop:
//       if !present(cur): break.
//       emit READ_NODE (thread=t, obj=cur, root=root_id, epoch=epoch, aux=pos);
//         counts[READ_NODES]++.
//       append one read_hash record (see read_hash below) using cur's current
//         fields: value, retired flag, alloc_seq, retire_seq.
//       if pos >= hop_cap: break.
//       nxt := edges[cur][0]; if nxt==0 || !present(nxt): break.
//       cur=nxt; pos+=1.
//     (Note: traversal follows slot 0 only; visits retired-but-present nodes.)
//
//   RETIRE(obj_id=arg_a, callback_tag=arg_b):
//     id=(u64)(u32)arg_a, tag=(u32)arg_b. If !present(id) || already retired:
//       INVALID. If id has any incoming PUBLISHED reference: INVALID. Incoming
//       published means: some root[r]==id, OR some present non-retired object k
//       has an edge edges[k][s]==id (scan all ids 1..RCU_REF_ID_CAP, all slots).
//     If retired_count+1 > spec.max_retired: emit RETIRE_REJECT_FULL (obj=id,
//       epoch=sc[5], aux=tag); counts[RETIRE_REJECT_FULL]++; return.
//     Else: retired=1; retire_seq=retire_seq_next (sc[3]); retire_seq_next+=1;
//       retire_epoch=sc[5]; callback_tag=tag; append id to retired queue. Emit
//       RETIRE_OK (obj=id, epoch=retire_epoch, aux=retire_seq);
//       counts[RETIRE_OK]++.
//
//   RECLAIM(limit=arg_a):
//     If limit<=0: no-op (no events, op_index still advances).
//     Sort the retired queue canonically by (retire_epoch asc, retire_seq asc,
//       obj_id asc). Walk it in that order with reclaimed=0:
//       For each id: if reclaimed>=limit, keep it in the queue (compact) and
//         continue. Else check grace-period eligibility: id is eligible iff NO
//         thread t has read_nest[t]>0 with outer_read_seq[t] < id.retire_seq
//         (a reader that started before this retirement still blocks it). If
//         not eligible, keep in queue and continue.
//       Eligible+within-limit: emit RCU_CALLBACK (obj=id, epoch=sc[5],
//         aux=callback_tag); counts[CALLBACKS_READY]++.
//         reclaim_seq_used := UINT64_MAX. If free_count >= spec.max_free_ids:
//           emit FREE_ID_DROPPED (obj=id, epoch=sc[5], aux=id);
//           counts[FREE_IDS_DROPPED]++ (id is NOT added to free list).
//         Else: reclaim_seq_used := reclaim_seq_next (sc[4]); reclaim_seq_next+=1;
//           append (id, reclaim_seq_used) to the free list.
//         Remove object: o_present[id]=0; present_total--.
//         Emit OBJECT_RECLAIM (obj=id, epoch=sc[5], aux=reclaim_seq_used);
//           counts[OBJECTS_RECLAIMED]++. reclaimed+=1.
//       After the walk, the queue holds exactly the kept (not reclaimed) ids.
//
//   FORCE_DROP_FREE_IDS(limit=arg_a):
//     If limit<=0: no-op. Sort free list canonically (reclaim_seq asc, obj_id
//     asc). Drop the first `limit` entries (or all if fewer): for each dropped
//     entry emit FREE_ID_DROPPED (obj=id, epoch=sc[5], aux=id);
//     counts[FREE_IDS_DROPPED]++. Remaining entries are kept (compacted).
//
// CANONICAL ORDERS
//   retired queue canonical order : (retire_epoch asc, retire_seq asc, obj_id
//     asc). Re-sorted (stable insertion sort) at the start of RECLAIM.
//   free list canonical order     : (reclaim_seq asc, obj_id asc). Re-sorted at
//     ALLOC (when popping), at FORCE_DROP, and at finalize (free_hash).

DETERMINISM CONTRACT FOR THE GRADED OUTPUTS
  - rcu_event_hash : FNV over events in emission order. Per event hash:
      event_kind:u8, event_seq:u64, op_index:u32,
      thread_or_UINT32_MAX:u32, obj_id_or_UINT64_MAX:u64,
      root_or_UINT32_MAX:u32, slot_or_UINT32_MAX:u32, epoch:u64, aux_u64
  - read_hash : FNV over READ_CHAIN output records, in operation order:
      op_index:u32, thread:u32, root_id:u32, position:u64,
      obj_id_or_ZERO:u64, value_or_INT64_MIN:i64, retired:u8,
      alloc_seq_or_UINT64_MAX:u64, retire_seq_or_UINT64_MAX:u64
  - root_hash : FNV by root id ascending: root_id:u32, obj_id_or_ZERO:u64
  - object_hash : FNV over live+retired objects by obj id ascending:
      obj_id:u64, value:i64, alloc_seq:u64, retired:u8,
      retire_epoch_or_UINT64_MAX:u64, retire_seq_or_UINT64_MAX:u64,
      callback_tag:u32, then edges by slot ascending (each u64)
  - thread_hash : FNV over threads ascending:
      thread:u32, read_nest:u64, outer_read_seq:u64, active_epoch:u64,
      then root_view by root id (each u64)
  - free_hash : FNV over reusable ids in canonical free order:
      obj_id:u64, reclaim_seq:u64
  - state scalars: final event_seq, obj_id_next, read_seq_next,
    retire_seq_next, reclaim_seq_next, global_epoch.

Rules:
  - solution_init may allocate persistent state.
  - solution_run may not call cudaMalloc / cudaFree.
  - solution_run may launch kernels and use provided workspace.
  - All graded outputs are exact integers.
*/

struct alignas(8) RcuProblemSpec {
    int32_t abi_version;
    int32_t thread_count;
    int32_t root_count;
    int32_t max_objects;
    int32_t max_edges_per_object;
    int32_t max_retired;
    int32_t max_free_ids;
    int32_t max_ops;
    int32_t reserved[8];
};

struct alignas(8) RcuRunSpec {
    int32_t abi_version;
    int32_t op_count;
    int32_t step_id;
    int32_t flags;
    int32_t reserved[12];
};

// One operation. Unused arg slots are ignored by the op semantics.
struct alignas(8) RcuOp {
    int32_t kind;
    int32_t arg_a;
    int32_t arg_b;
    int32_t arg_c;
    int64_t arg_i64;
};

struct alignas(8) RcuInputs {
    const RcuOp* ops;
};

struct alignas(8) RcuOutputs {
    uint64_t* counts;        // 16 entries, see RCU_COUNT_* below
    uint64_t* event_hash;    // 1
    uint64_t* read_hash;     // 1
    uint64_t* root_hash;     // 1
    uint64_t* object_hash;   // 1
    uint64_t* thread_hash;   // 1
    uint64_t* free_hash;     // 1
    uint64_t* state_scalars; // 6: event_seq,obj_id_next,read_seq_next,
                             //    retire_seq_next,reclaim_seq_next,global_epoch
};

// counts[] indices
#define RCU_COUNT_ALLOC_OK 0
#define RCU_COUNT_ALLOC_OOM 1
#define RCU_COUNT_ROOT_SETS 2
#define RCU_COUNT_EDGE_SETS 3
#define RCU_COUNT_READ_LOCKS 4
#define RCU_COUNT_READ_UNLOCKS 5
#define RCU_COUNT_QUIESCES 6
#define RCU_COUNT_EPOCH_ADVANCES 7
#define RCU_COUNT_READ_NODES 8
#define RCU_COUNT_READ_EMPTY 9
#define RCU_COUNT_RETIRE_OK 10
#define RCU_COUNT_RETIRE_REJECT_FULL 11
#define RCU_COUNT_CALLBACKS_READY 12
#define RCU_COUNT_OBJECTS_RECLAIMED 13
#define RCU_COUNT_FREE_IDS_DROPPED 14
#define RCU_COUNT_INVALID 15
#define RCU_COUNT_TOTAL 16

static_assert(sizeof(RcuProblemSpec) == 64, "RcuProblemSpec layout drift");
static_assert(sizeof(RcuRunSpec) == 64, "RcuRunSpec layout drift");
static_assert(sizeof(RcuOp) == 24, "RcuOp layout drift");
static_assert(sizeof(RcuInputs) == 8, "RcuInputs layout drift");
static_assert(sizeof(RcuOutputs) == 64, "RcuOutputs layout drift");

static inline size_t rcu_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int rcu_validate_problem_spec(const RcuProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != RCU_ABI_VERSION) return 0;
    if (spec->thread_count < RCU_MIN_THREADS || spec->thread_count > RCU_MAX_THREADS) return 0;
    if (spec->root_count < RCU_MIN_ROOTS || spec->root_count > RCU_MAX_ROOTS) return 0;
    if (spec->max_objects < RCU_MIN_OBJECTS || spec->max_objects > RCU_MAX_OBJECTS) return 0;
    if (spec->max_edges_per_object < RCU_MIN_EDGES ||
        spec->max_edges_per_object > RCU_MAX_EDGES) return 0;
    if (spec->max_retired < 0 || spec->max_retired > RCU_MAX_RETIRED) return 0;
    if (spec->max_free_ids < 0 || spec->max_free_ids > RCU_MAX_FREE_IDS) return 0;
    if (spec->max_ops < 0 || spec->max_ops > RCU_MAX_OPS) return 0;
    return 1;
}

static inline int rcu_validate_run_spec(const RcuRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != RCU_ABI_VERSION) return 0;
    if (run->op_count < 0 || run->op_count > RCU_MAX_OPS) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const RcuProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const RcuProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const RcuRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // RCU_EPOCH_RECLAIMER_COMMON_H_
