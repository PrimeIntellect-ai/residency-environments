// PMPP_CANARY_49_437f2a44a8 -- held-out canary; MUST NOT appear in any submission
// file: rcu_epoch_reclaimer_reference.cu
//
// Reference implementation. Single-thread device kernel that maintains the
// persistent RCU state in dense device arrays keyed directly by object id.
// Independent algorithm from naive.cu (which uses a linked-list / scan model)
// and from the host oracle (std::map based).
//
// FAST reference: the op stream is an inherently serial state machine (each
// event folds into a running FNV-1a-64 over all prior events, in emission
// order), so ops cannot be parallelized. The dominant cost of the naive
// version was dependent GLOBAL-memory round-trips on the hot persistent state
// (the two streaming hashes, the six state scalars, the 16 counters, and the
// present/retired/free counters), which are touched on every event. This
// version caches all of that hot state in registers for the lifetime of the
// run kernel (loaded from global at entry, written back at exit), so the
// serial dependency chain runs at register latency instead of global-memory
// latency. The object/thread/root/queue arrays (indexed by dynamic id) stay in
// global. The four independent structural-hash walks run on four concurrent
// lanes; the object-table reset is spread across the grid. Byte-identical
// output: no algorithm, ordering, coefficient, or rounding change -- execution
// locality only.

#include "rcu_epoch_reclaimer_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Max distinct object ids over a run: fresh ids come from obj_id_next which is
// incremented at most once per ALLOC op (<= max_ops), starting at 1.
#define RCU_REF_ID_CAP (RCU_MAX_OBJECTS + RCU_MAX_OPS + 4)

struct RcuRefState {
    RcuProblemSpec spec;

    // persistent scalars (1 element each, device).
    uint64_t* sc;  // [0..5] event_seq,obj_id_next,read_seq_next,retire_seq_next,
                   //        reclaim_seq_next,global_epoch
    uint64_t* op_index_global;  // 1
    uint64_t* event_hash;       // 1
    uint64_t* read_hash;        // 1
    uint64_t* counts;           // RCU_COUNT_TOTAL

    // object table keyed by id in [0, RCU_REF_ID_CAP). index 0 unused (id 0 = null).
    uint8_t*  o_present;
    int64_t*  o_value;
    uint64_t* o_alloc_seq;
    uint8_t*  o_retired;
    uint64_t* o_retire_seq;
    uint64_t* o_retire_epoch;
    uint32_t* o_callback_tag;
    uint64_t* o_edges;  // RCU_REF_ID_CAP * max_edges
    // Reverse index: number of PUBLISHED incoming references to each id =
    // (#root slots pointing to id) + (#edge slots of present-non-retired
    // objects pointing to id). Maintained incrementally so RETIRE can test
    // "has incoming published" in O(1) instead of an O(RCU_REF_ID_CAP) scan.
    uint32_t* o_incoming;  // RCU_REF_ID_CAP

    uint64_t* root;         // root_count

    // thread table
    uint64_t* th_read_nest;     // thread_count
    uint64_t* th_outer_seq;     // thread_count
    uint64_t* th_active_epoch;  // thread_count
    uint64_t* th_root_view;     // thread_count * root_count

    // retired queue: list of obj ids.
    uint64_t* retired_ids;   // RCU_MAX_RETIRED
    int32_t*  retired_count; // 1

    // free list entries.
    uint64_t* free_obj;      // RCU_MAX_FREE_IDS
    uint64_t* free_seq;      // RCU_MAX_FREE_IDS
    int32_t*  free_count;    // 1

    int32_t*  present_total; // 1 : number of present objects
};

// ---- value-returning FNV-1a-64 helpers (keep the running hash in a register) ----
__device__ __forceinline__ uint64_t rref_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= 1099511628211ULL;
    return h;
}
__device__ __forceinline__ uint64_t rref_fnv_bytes(uint64_t h, const void* p, int n) {
    const uint8_t* b = (const uint8_t*)p;
    for (int i = 0; i < n; ++i) h = rref_fnv_byte(h, b[i]);
    return h;
}
__device__ __forceinline__ uint64_t rref_h_u8 (uint64_t h, uint8_t v ){ return rref_fnv_bytes(h,&v,1); }
__device__ __forceinline__ uint64_t rref_h_u32(uint64_t h, uint32_t v){ return rref_fnv_bytes(h,&v,4); }
__device__ __forceinline__ uint64_t rref_h_u64(uint64_t h, uint64_t v){ return rref_fnv_bytes(h,&v,8); }
__device__ __forceinline__ uint64_t rref_h_i64(uint64_t h, int64_t v ){ return rref_fnv_bytes(h,&v,8); }

// pointer-style helpers retained for the finalize kernel.
__device__ __forceinline__ void rref_u8 (uint64_t* h, uint8_t v ){ *h = rref_h_u8 (*h, v); }
__device__ __forceinline__ void rref_u32(uint64_t* h, uint32_t v){ *h = rref_h_u32(*h, v); }
__device__ __forceinline__ void rref_u64(uint64_t* h, uint64_t v){ *h = rref_h_u64(*h, v); }
__device__ __forceinline__ void rref_i64(uint64_t* h, int64_t v ){ *h = rref_h_i64(*h, v); }

// Cached execution context: array pointers stay global, hot scalars are local.
struct RrefCtx {
    RcuRefState s;
    int max_edges;
    int root_count;
    int thread_count;

    // hot persistent state cached in registers for the whole run.
    uint64_t sc[6];
    uint64_t op_index;
    uint64_t event_hash;
    uint64_t read_hash;
    uint64_t counts[RCU_COUNT_TOTAL];
    int present_total;
    int retired_count;
    int free_count;
};

__device__ __forceinline__ void rref_emit_event(RrefCtx& c, uint8_t kind, uint32_t op_index,
                                 uint32_t thr, uint64_t obj, uint32_t root,
                                 uint32_t slot, uint64_t epoch, uint64_t aux) {
    c.sc[0] += 1;  // event_seq
    uint64_t h = c.event_hash;
    h = rref_h_u8 (h, kind);
    h = rref_h_u64(h, c.sc[0]);
    h = rref_h_u32(h, op_index);
    h = rref_h_u32(h, thr);
    h = rref_h_u64(h, obj);
    h = rref_h_u32(h, root);
    h = rref_h_u32(h, slot);
    h = rref_h_u64(h, epoch);
    h = rref_h_u64(h, aux);
    c.event_hash = h;
}

__device__ __forceinline__ void rref_emit_invalid(RrefCtx& c, uint32_t op_index, uint32_t opcode) {
    rref_emit_event(c, RCU_EV_INVALID, op_index, UINT32_MAX, UINT64_MAX,
                    UINT32_MAX, UINT32_MAX, c.sc[5], (uint64_t)opcode);
    c.counts[RCU_COUNT_INVALID] += 1;
}

__device__ __forceinline__ bool rref_present(RrefCtx& c, uint64_t id) {
    return id != 0 && id < (uint64_t)RCU_REF_ID_CAP && c.s.o_present[id] != 0;
}
__device__ __forceinline__ bool rref_live(RrefCtx& c, uint64_t id) {
    return rref_present(c, id) && c.s.o_retired[id] == 0;
}

// insertion-sort the retired queue into canonical order in place.
__device__ void rref_sort_retired(RrefCtx& c) {
    int n = c.retired_count;
    uint64_t* a = c.s.retired_ids;
    for (int i = 1; i < n; ++i) {
        uint64_t id = a[i];
        uint64_t re = c.s.o_retire_epoch[id];
        uint64_t rs = c.s.o_retire_seq[id];
        int j = i - 1;
        while (j >= 0) {
            uint64_t bid = a[j];
            uint64_t bre = c.s.o_retire_epoch[bid];
            uint64_t brs = c.s.o_retire_seq[bid];
            bool greater = (bre > re) ||
                           (bre == re && brs > rs) ||
                           (bre == re && brs == rs && bid > id);
            if (!greater) break;
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = id;
    }
}

__device__ void rref_sort_free(RrefCtx& c) {
    int n = c.free_count;
    uint64_t* fo = c.s.free_obj;
    uint64_t* fs = c.s.free_seq;
    for (int i = 1; i < n; ++i) {
        uint64_t oid = fo[i], osq = fs[i];
        int j = i - 1;
        while (j >= 0) {
            bool greater = (fs[j] > osq) || (fs[j] == osq && fo[j] > oid);
            if (!greater) break;
            fo[j + 1] = fo[j];
            fs[j + 1] = fs[j];
            --j;
        }
        fo[j + 1] = oid;
        fs[j + 1] = osq;
    }
}

__device__ void rref_op_alloc(RrefCtx& c, uint32_t op_index, const RcuOp& op) {
    const uint64_t request_id = (uint64_t)(uint32_t)op.arg_a;
    const int64_t value = op.arg_i64;
    const bool full = c.present_total >= c.s.spec.max_objects;
    const int fc = c.free_count;

    if (full && fc == 0) {
        rref_emit_event(c, RCU_EV_ALLOC_OOM, op_index, UINT32_MAX, UINT64_MAX,
                        UINT32_MAX, UINT32_MAX, c.sc[5], request_id);
        c.counts[RCU_COUNT_ALLOC_OOM] += 1;
        return;
    }

    uint64_t chosen;
    if (fc > 0) {
        rref_sort_free(c);
        chosen = c.s.free_obj[0];
        // pop front
        for (int i = 1; i < fc; ++i) {
            c.s.free_obj[i - 1] = c.s.free_obj[i];
            c.s.free_seq[i - 1] = c.s.free_seq[i];
        }
        c.free_count = fc - 1;
    } else {
        chosen = c.sc[1];     // obj_id_next
        c.sc[1] += 1;
    }

    rref_emit_event(c, RCU_EV_ALLOC_OK, op_index, UINT32_MAX, chosen,
                    UINT32_MAX, UINT32_MAX, c.sc[5], request_id);
    const uint64_t alloc_seq = c.sc[0];

    c.s.o_present[chosen] = 1;
    c.s.o_value[chosen] = value;
    c.s.o_alloc_seq[chosen] = alloc_seq;
    c.s.o_retired[chosen] = 0;
    c.s.o_retire_seq[chosen] = UINT64_MAX;
    c.s.o_retire_epoch[chosen] = UINT64_MAX;
    c.s.o_callback_tag[chosen] = 0;
    for (int s = 0; s < c.max_edges; ++s)
        c.s.o_edges[chosen * c.max_edges + s] = 0;

    c.present_total += 1;
    c.counts[RCU_COUNT_ALLOC_OK] += 1;
}

__device__ void rref_op_set_root(RrefCtx& c, uint32_t op_index, const RcuOp& op) {
    const int32_t root_id = op.arg_a;
    const uint64_t obj = (uint64_t)(uint32_t)op.arg_b;
    if (root_id < 0 || root_id >= c.root_count) { rref_emit_invalid(c, op_index, RCU_OP_SET_ROOT); return; }
    if (obj != 0 && !rref_live(c, obj)) { rref_emit_invalid(c, op_index, RCU_OP_SET_ROOT); return; }
    // maintain incoming-ref counts: this root slot moves from old target to obj.
    const uint64_t old_root = c.s.root[root_id];
    if (old_root != 0) c.s.o_incoming[old_root] -= 1;
    if (obj != 0) c.s.o_incoming[obj] += 1;
    c.s.root[root_id] = obj;
    rref_emit_event(c, RCU_EV_ROOT_SET, op_index, UINT32_MAX, obj,
                    (uint32_t)root_id, UINT32_MAX, c.sc[5], 0);
    c.counts[RCU_COUNT_ROOT_SETS] += 1;
}

__device__ void rref_op_link(RrefCtx& c, uint32_t op_index, const RcuOp& op) {
    const uint64_t src = (uint64_t)(uint32_t)op.arg_a;
    const int32_t slot = op.arg_b;
    const uint64_t dst = (uint64_t)(uint32_t)op.arg_c;
    if (!rref_live(c, src)) { rref_emit_invalid(c, op_index, RCU_OP_LINK); return; }
    if (slot < 0 || slot >= c.max_edges) { rref_emit_invalid(c, op_index, RCU_OP_LINK); return; }
    if (dst != 0 && !rref_live(c, dst)) { rref_emit_invalid(c, op_index, RCU_OP_LINK); return; }
    // maintain incoming-ref counts: src is live, so this edge slot moves from
    // its old target to dst (both count as published incoming references).
    const uint64_t old_edge = c.s.o_edges[src * c.max_edges + slot];
    if (old_edge != 0) c.s.o_incoming[old_edge] -= 1;
    if (dst != 0) c.s.o_incoming[dst] += 1;
    c.s.o_edges[src * c.max_edges + slot] = dst;
    rref_emit_event(c, RCU_EV_EDGE_SET, op_index, UINT32_MAX, src,
                    UINT32_MAX, (uint32_t)slot, c.sc[5], dst);
    c.counts[RCU_COUNT_EDGE_SETS] += 1;
}

__device__ void rref_op_read_lock(RrefCtx& c, uint32_t op_index, const RcuOp& op) {
    const int32_t t = op.arg_a;
    if (t < 0 || t >= c.thread_count) { rref_emit_invalid(c, op_index, RCU_OP_READ_LOCK); return; }
    if (c.s.th_read_nest[t] == 0) {
        for (int r = 0; r < c.root_count; ++r)
            c.s.th_root_view[t * c.root_count + r] = c.s.root[r];
        c.s.th_outer_seq[t] = c.sc[2];  // read_seq_next
        c.sc[2] += 1;
        c.s.th_active_epoch[t] = c.sc[5];
    }
    if (c.s.th_read_nest[t] + 1 == 0) { rref_emit_invalid(c, op_index, RCU_OP_READ_LOCK); return; }
    c.s.th_read_nest[t] += 1;
    rref_emit_event(c, RCU_EV_READ_LOCK_OK, op_index, (uint32_t)t, UINT64_MAX,
                    UINT32_MAX, UINT32_MAX, c.s.th_active_epoch[t], c.s.th_outer_seq[t]);
    c.counts[RCU_COUNT_READ_LOCKS] += 1;
}

__device__ void rref_op_read_unlock(RrefCtx& c, uint32_t op_index, const RcuOp& op) {
    const int32_t t = op.arg_a;
    if (t < 0 || t >= c.thread_count) { rref_emit_invalid(c, op_index, RCU_OP_READ_UNLOCK); return; }
    if (c.s.th_read_nest[t] == 0) { rref_emit_invalid(c, op_index, RCU_OP_READ_UNLOCK); return; }
    c.s.th_read_nest[t] -= 1;
    if (c.s.th_read_nest[t] == 0) {
        c.s.th_outer_seq[t] = UINT64_MAX;
        c.s.th_active_epoch[t] = c.sc[5];
        rref_emit_event(c, RCU_EV_READ_UNLOCK_QUIESCENT, op_index, (uint32_t)t, UINT64_MAX,
                        UINT32_MAX, UINT32_MAX, c.s.th_active_epoch[t], 0);
    } else {
        rref_emit_event(c, RCU_EV_READ_UNLOCK_NESTED, op_index, (uint32_t)t, UINT64_MAX,
                        UINT32_MAX, UINT32_MAX, c.s.th_active_epoch[t], c.s.th_read_nest[t]);
    }
    c.counts[RCU_COUNT_READ_UNLOCKS] += 1;
}

__device__ void rref_op_quiesce(RrefCtx& c, uint32_t op_index, const RcuOp& op) {
    const int32_t t = op.arg_a;
    if (t < 0 || t >= c.thread_count) { rref_emit_invalid(c, op_index, RCU_OP_QUIESCE); return; }
    if (c.s.th_read_nest[t] != 0) { rref_emit_invalid(c, op_index, RCU_OP_QUIESCE); return; }
    c.s.th_active_epoch[t] = c.sc[5];
    rref_emit_event(c, RCU_EV_QUIESCE_OK, op_index, (uint32_t)t, UINT64_MAX,
                    UINT32_MAX, UINT32_MAX, c.s.th_active_epoch[t], 0);
    c.counts[RCU_COUNT_QUIESCES] += 1;
}

__device__ void rref_op_advance(RrefCtx& c, uint32_t op_index) {
    c.sc[5] += 1;
    rref_emit_event(c, RCU_EV_EPOCH_ADVANCE, op_index, UINT32_MAX, UINT64_MAX,
                    UINT32_MAX, UINT32_MAX, c.sc[5], 0);
    c.counts[RCU_COUNT_EPOCH_ADVANCES] += 1;
}

__device__ __forceinline__ void rref_emit_read_rec(RrefCtx& c, uint32_t op_index, uint32_t t,
                                    uint32_t root_id, uint64_t pos, uint64_t obj,
                                    int64_t val, uint8_t retired, uint64_t alloc,
                                    uint64_t retire) {
    uint64_t h = c.read_hash;
    h = rref_h_u32(h, op_index);
    h = rref_h_u32(h, t);
    h = rref_h_u32(h, root_id);
    h = rref_h_u64(h, pos);
    h = rref_h_u64(h, obj);
    h = rref_h_i64(h, val);
    h = rref_h_u8 (h, retired);
    h = rref_h_u64(h, alloc);
    h = rref_h_u64(h, retire);
    c.read_hash = h;
}

__device__ void rref_op_read_chain(RrefCtx& c, uint32_t op_index, const RcuOp& op) {
    const int32_t t = op.arg_a;
    const int32_t root_id = op.arg_b;
    const int32_t max_hops = op.arg_c;
    if (t < 0 || t >= c.thread_count || root_id < 0 || root_id >= c.root_count) {
        rref_emit_invalid(c, op_index, RCU_OP_READ_CHAIN); return;
    }
    if (c.s.th_read_nest[t] == 0) { rref_emit_invalid(c, op_index, RCU_OP_READ_CHAIN); return; }

    uint64_t cur = c.s.th_root_view[t * c.root_count + root_id];
    const uint64_t epoch = c.s.th_active_epoch[t];

    if (cur == 0 || !rref_present(c, cur)) {
        rref_emit_event(c, RCU_EV_READ_EMPTY, op_index, (uint32_t)t, UINT64_MAX,
                        (uint32_t)root_id, UINT32_MAX, epoch, 0);
        c.counts[RCU_COUNT_READ_EMPTY] += 1;
        return;
    }

    uint64_t pos = 0;
    const int64_t hop_cap = (max_hops < 0) ? 0 : (int64_t)max_hops;
    while (true) {
        if (!rref_present(c, cur)) break;
        rref_emit_event(c, RCU_EV_READ_NODE, op_index, (uint32_t)t, cur,
                        (uint32_t)root_id, UINT32_MAX, epoch, pos);
        c.counts[RCU_COUNT_READ_NODES] += 1;
        rref_emit_read_rec(c, op_index, (uint32_t)t, (uint32_t)root_id, pos, cur,
                           c.s.o_value[cur], c.s.o_retired[cur],
                           c.s.o_alloc_seq[cur], c.s.o_retire_seq[cur]);
        if ((int64_t)pos >= hop_cap) break;
        uint64_t nxt = c.s.o_edges[cur * c.max_edges + 0];
        if (nxt == 0 || !rref_present(c, nxt)) break;
        cur = nxt;
        pos += 1;
    }
}

__device__ bool rref_has_incoming_published(RrefCtx& c, uint64_t id) {
    // O(1) via the maintained reverse-index (identical predicate to the
    // original O(RCU_REF_ID_CAP) scan over roots + live objects' edges).
    return c.s.o_incoming[id] != 0;
}

__device__ void rref_op_retire(RrefCtx& c, uint32_t op_index, const RcuOp& op) {
    const uint64_t id = (uint64_t)(uint32_t)op.arg_a;
    const uint32_t tag = (uint32_t)op.arg_b;
    if (!rref_present(c, id) || c.s.o_retired[id] != 0) { rref_emit_invalid(c, op_index, RCU_OP_RETIRE); return; }
    if (rref_has_incoming_published(c, id)) { rref_emit_invalid(c, op_index, RCU_OP_RETIRE); return; }

    if (c.retired_count + 1 > c.s.spec.max_retired) {
        rref_emit_event(c, RCU_EV_RETIRE_REJECT_FULL, op_index, UINT32_MAX, id,
                        UINT32_MAX, UINT32_MAX, c.sc[5], (uint64_t)tag);
        c.counts[RCU_COUNT_RETIRE_REJECT_FULL] += 1;
        return;
    }

    // id transitions live -> retired: its outgoing edges no longer count as
    // published incoming references for their targets. Drop those counts.
    for (int s = 0; s < c.max_edges; ++s) {
        uint64_t e = c.s.o_edges[id * c.max_edges + s];
        if (e != 0) c.s.o_incoming[e] -= 1;
    }

    c.s.o_retired[id] = 1;
    c.s.o_retire_seq[id] = c.sc[3];  // retire_seq_next
    c.sc[3] += 1;
    c.s.o_retire_epoch[id] = c.sc[5];
    c.s.o_callback_tag[id] = tag;
    int rc = c.retired_count;
    c.s.retired_ids[rc] = id;
    c.retired_count = rc + 1;

    rref_emit_event(c, RCU_EV_RETIRE_OK, op_index, UINT32_MAX, id,
                    UINT32_MAX, UINT32_MAX, c.s.o_retire_epoch[id], c.s.o_retire_seq[id]);
    c.counts[RCU_COUNT_RETIRE_OK] += 1;
}

__device__ bool rref_eligible(RrefCtx& c, uint64_t retire_seq) {
    for (int t = 0; t < c.thread_count; ++t) {
        if (c.s.th_read_nest[t] > 0 && c.s.th_outer_seq[t] < retire_seq) return false;
    }
    return true;
}

__device__ void rref_op_reclaim(RrefCtx& c, uint32_t op_index, const RcuOp& op) {
    const int32_t limit = op.arg_a;
    if (limit <= 0) return;

    rref_sort_retired(c);
    int n = c.retired_count;
    int reclaimed = 0;
    int write = 0;  // compaction write index for remaining

    for (int qi = 0; qi < n; ++qi) {
        uint64_t id = c.s.retired_ids[qi];
        if (reclaimed >= limit) { c.s.retired_ids[write++] = id; continue; }
        if (!rref_eligible(c, c.s.o_retire_seq[id])) { c.s.retired_ids[write++] = id; continue; }

        uint32_t tag = c.s.o_callback_tag[id];
        rref_emit_event(c, RCU_EV_RCU_CALLBACK, op_index, UINT32_MAX, id,
                        UINT32_MAX, UINT32_MAX, c.sc[5], (uint64_t)tag);
        c.counts[RCU_COUNT_CALLBACKS_READY] += 1;

        uint64_t reclaim_seq_used = UINT64_MAX;
        if (c.free_count >= c.s.spec.max_free_ids) {
            rref_emit_event(c, RCU_EV_FREE_ID_DROPPED, op_index, UINT32_MAX, id,
                            UINT32_MAX, UINT32_MAX, c.sc[5], id);
            c.counts[RCU_COUNT_FREE_IDS_DROPPED] += 1;
        } else {
            reclaim_seq_used = c.sc[4];  // reclaim_seq_next
            c.sc[4] += 1;
            int fc = c.free_count;
            c.s.free_obj[fc] = id;
            c.s.free_seq[fc] = reclaim_seq_used;
            c.free_count = fc + 1;
        }

        // remove object from table.
        c.s.o_present[id] = 0;
        c.present_total -= 1;

        rref_emit_event(c, RCU_EV_OBJECT_RECLAIM, op_index, UINT32_MAX, id,
                        UINT32_MAX, UINT32_MAX, c.sc[5], reclaim_seq_used);
        c.counts[RCU_COUNT_OBJECTS_RECLAIMED] += 1;
        reclaimed += 1;
    }
    c.retired_count = write;
}

__device__ void rref_op_force_drop(RrefCtx& c, uint32_t op_index, const RcuOp& op) {
    const int32_t limit = op.arg_a;
    if (limit <= 0) return;
    rref_sort_free(c);
    int n = c.free_count;
    int removed = 0, write = 0;
    for (int i = 0; i < n; ++i) {
        if (removed < limit) {
            uint64_t id = c.s.free_obj[i];
            rref_emit_event(c, RCU_EV_FREE_ID_DROPPED, op_index, UINT32_MAX, id,
                            UINT32_MAX, UINT32_MAX, c.sc[5], id);
            c.counts[RCU_COUNT_FREE_IDS_DROPPED] += 1;
            removed += 1;
        } else {
            c.s.free_obj[write] = c.s.free_obj[i];
            c.s.free_seq[write] = c.s.free_seq[i];
            ++write;
        }
    }
    c.free_count = write;
}

__global__ void rcu_ref_run_kernel(RcuRefState st, const RcuOp* ops, int op_count,
                                    int max_edges, int root_count, int thread_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    RrefCtx c;
    c.s = st;
    c.max_edges = max_edges;
    c.root_count = root_count;
    c.thread_count = thread_count;

    // load hot persistent state into registers.
    #pragma unroll
    for (int i = 0; i < 6; ++i) c.sc[i] = st.sc[i];
    c.op_index = *st.op_index_global;
    c.event_hash = *st.event_hash;
    c.read_hash = *st.read_hash;
    #pragma unroll
    for (int i = 0; i < RCU_COUNT_TOTAL; ++i) c.counts[i] = st.counts[i];
    c.present_total = *st.present_total;
    c.retired_count = *st.retired_count;
    c.free_count = *st.free_count;

    for (int i = 0; i < op_count; ++i) {
        const RcuOp op = ops[i];
        const uint32_t op_index = (uint32_t)c.op_index;
        switch (op.kind) {
            case RCU_OP_ALLOC: rref_op_alloc(c, op_index, op); break;
            case RCU_OP_SET_ROOT: rref_op_set_root(c, op_index, op); break;
            case RCU_OP_LINK: rref_op_link(c, op_index, op); break;
            case RCU_OP_READ_LOCK: rref_op_read_lock(c, op_index, op); break;
            case RCU_OP_READ_UNLOCK: rref_op_read_unlock(c, op_index, op); break;
            case RCU_OP_QUIESCE: rref_op_quiesce(c, op_index, op); break;
            case RCU_OP_ADVANCE_EPOCH: rref_op_advance(c, op_index); break;
            case RCU_OP_READ_CHAIN: rref_op_read_chain(c, op_index, op); break;
            case RCU_OP_RETIRE: rref_op_retire(c, op_index, op); break;
            case RCU_OP_RECLAIM: rref_op_reclaim(c, op_index, op); break;
            case RCU_OP_FORCE_DROP_FREE_IDS: rref_op_force_drop(c, op_index, op); break;
            default: rref_emit_invalid(c, op_index, (uint32_t)op.kind); break;
        }
        c.op_index += 1;
    }

    // write hot persistent state back to global.
    #pragma unroll
    for (int i = 0; i < 6; ++i) st.sc[i] = c.sc[i];
    *st.op_index_global = c.op_index;
    *st.event_hash = c.event_hash;
    *st.read_hash = c.read_hash;
    #pragma unroll
    for (int i = 0; i < RCU_COUNT_TOTAL; ++i) st.counts[i] = c.counts[i];
    *st.present_total = c.present_total;
    *st.retired_count = c.retired_count;
    *st.free_count = c.free_count;
}

// Structural-checksum kernel: writes the 6 structural hashes + scalars to
// outputs. The four hashes are independent and each is order-sensitive, so
// they run on four concurrent lanes (threads 0..3). Byte-identical to a serial
// walk; only the four walks are executed in parallel.
__global__ void rcu_ref_finalize_kernel(RcuRefState st, int max_edges, int root_count,
                                         int thread_count,
                                         uint64_t* out_root_hash,
                                         uint64_t* out_object_hash,
                                         uint64_t* out_thread_hash,
                                         uint64_t* out_free_hash) {
    const int lane = threadIdx.x;
    if (lane == 0) {
        // root_hash
        uint64_t h = 1469598103934665603ULL;
        for (int r = 0; r < root_count; ++r) {
            rref_u32(&h, (uint32_t)r);
            rref_u64(&h, st.root[r]);
        }
        *out_root_hash = h;
    } else if (lane == 1) {
        // thread_hash
        uint64_t h = 1469598103934665603ULL;
        for (int t = 0; t < thread_count; ++t) {
            rref_u32(&h, (uint32_t)t);
            rref_u64(&h, st.th_read_nest[t]);
            rref_u64(&h, st.th_outer_seq[t]);
            rref_u64(&h, st.th_active_epoch[t]);
            for (int r = 0; r < root_count; ++r)
                rref_u64(&h, st.th_root_view[t * root_count + r]);
        }
        *out_thread_hash = h;
    } else if (lane == 2) {
        // free_hash : sort then hash.
        int n = *st.free_count;
        for (int i = 1; i < n; ++i) {
            uint64_t oid = st.free_obj[i], osq = st.free_seq[i];
            int j = i - 1;
            while (j >= 0) {
                bool greater = (st.free_seq[j] > osq) || (st.free_seq[j] == osq && st.free_obj[j] > oid);
                if (!greater) break;
                st.free_obj[j + 1] = st.free_obj[j];
                st.free_seq[j + 1] = st.free_seq[j];
                --j;
            }
            st.free_obj[j + 1] = oid;
            st.free_seq[j + 1] = osq;
        }
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < n; ++i) {
            rref_u64(&h, st.free_obj[i]);
            rref_u64(&h, st.free_seq[i]);
        }
        *out_free_hash = h;
    } else if (lane == 3) {
        // object_hash : objects ascending by id.
        uint64_t h = 1469598103934665603ULL;
        for (uint64_t id = 1; id < (uint64_t)RCU_REF_ID_CAP; ++id) {
            if (st.o_present[id] == 0) continue;
            rref_u64(&h, id);
            rref_i64(&h, st.o_value[id]);
            rref_u64(&h, st.o_alloc_seq[id]);
            rref_u8(&h, st.o_retired[id]);
            rref_u64(&h, st.o_retire_epoch[id]);
            rref_u64(&h, st.o_retire_seq[id]);
            rref_u32(&h, st.o_callback_tag[id]);
            for (int s = 0; s < max_edges; ++s)
                rref_u64(&h, st.o_edges[id * max_edges + s]);
        }
        *out_object_hash = h;
    }
}

// ---------------- host plumbing ----------------

static cudaError_t rref_alloc_all(RcuRefState* st) {
    cudaError_t e;
    const int ME = st->spec.max_edges_per_object;
    const int RC = st->spec.root_count;
    const int TC = st->spec.thread_count;
    const size_t CAP = RCU_REF_ID_CAP;

#define RREF_MALLOC(p, bytes) do { e = cudaMalloc((void**)&(p), (bytes)); if (e != cudaSuccess) return e; } while(0)
    RREF_MALLOC(st->sc, sizeof(uint64_t) * 6);
    RREF_MALLOC(st->op_index_global, sizeof(uint64_t));
    RREF_MALLOC(st->event_hash, sizeof(uint64_t));
    RREF_MALLOC(st->read_hash, sizeof(uint64_t));
    RREF_MALLOC(st->counts, sizeof(uint64_t) * RCU_COUNT_TOTAL);

    RREF_MALLOC(st->o_present, sizeof(uint8_t) * CAP);
    RREF_MALLOC(st->o_value, sizeof(int64_t) * CAP);
    RREF_MALLOC(st->o_alloc_seq, sizeof(uint64_t) * CAP);
    RREF_MALLOC(st->o_retired, sizeof(uint8_t) * CAP);
    RREF_MALLOC(st->o_retire_seq, sizeof(uint64_t) * CAP);
    RREF_MALLOC(st->o_retire_epoch, sizeof(uint64_t) * CAP);
    RREF_MALLOC(st->o_callback_tag, sizeof(uint32_t) * CAP);
    RREF_MALLOC(st->o_edges, sizeof(uint64_t) * CAP * (size_t)ME);
    RREF_MALLOC(st->o_incoming, sizeof(uint32_t) * CAP);

    RREF_MALLOC(st->root, sizeof(uint64_t) * (size_t)RC);
    RREF_MALLOC(st->th_read_nest, sizeof(uint64_t) * (size_t)TC);
    RREF_MALLOC(st->th_outer_seq, sizeof(uint64_t) * (size_t)TC);
    RREF_MALLOC(st->th_active_epoch, sizeof(uint64_t) * (size_t)TC);
    RREF_MALLOC(st->th_root_view, sizeof(uint64_t) * (size_t)TC * (size_t)RC);

    RREF_MALLOC(st->retired_ids, sizeof(uint64_t) * (size_t)RCU_MAX_RETIRED);
    RREF_MALLOC(st->retired_count, sizeof(int32_t));
    RREF_MALLOC(st->free_obj, sizeof(uint64_t) * (size_t)RCU_MAX_FREE_IDS);
    RREF_MALLOC(st->free_seq, sizeof(uint64_t) * (size_t)RCU_MAX_FREE_IDS);
    RREF_MALLOC(st->free_count, sizeof(int32_t));
    RREF_MALLOC(st->present_total, sizeof(int32_t));
#undef RREF_MALLOC
    return cudaSuccess;
}

// Parallel reset: the object table dominates (RCU_REF_ID_CAP * fields); spread
// it across the grid. Byte-identical initial state to a serial reset.
__global__ void rref_reset_kernel(RcuRefState st, int max_edges, int root_count, int thread_count) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    if (gid == 0) {
        st.sc[0] = 0; st.sc[1] = 1; st.sc[2] = 1; st.sc[3] = 1; st.sc[4] = 1; st.sc[5] = 0;
        *st.op_index_global = 0;
        *st.event_hash = 1469598103934665603ULL;
        *st.read_hash = 1469598103934665603ULL;
        *st.retired_count = 0;
        *st.free_count = 0;
        *st.present_total = 0;
    }
    if (gid < (uint64_t)RCU_COUNT_TOTAL) st.counts[gid] = 0;
    for (uint64_t r = gid; r < (uint64_t)root_count; r += stride) st.root[r] = 0;
    for (uint64_t t = gid; t < (uint64_t)thread_count; t += stride) {
        st.th_read_nest[t] = 0;
        st.th_outer_seq[t] = UINT64_MAX;
        st.th_active_epoch[t] = 0;
        for (int r = 0; r < root_count; ++r) st.th_root_view[t * root_count + r] = 0;
    }
    for (uint64_t id = gid; id < (uint64_t)RCU_REF_ID_CAP; id += stride) {
        st.o_present[id] = 0;
        st.o_retired[id] = 0;
        st.o_retire_seq[id] = UINT64_MAX;
        st.o_retire_epoch[id] = UINT64_MAX;
        st.o_callback_tag[id] = 0;
        st.o_value[id] = 0;
        st.o_alloc_seq[id] = 0;
        st.o_incoming[id] = 0;
        for (int s = 0; s < max_edges; ++s) st.o_edges[id * max_edges + s] = 0;
    }
}

extern "C" size_t solution_workspace_bytes(const RcuProblemSpec* spec) {
    if (!rcu_validate_problem_spec(spec)) return 0;
    return 256;
}

extern "C" cudaError_t solution_init(const RcuProblemSpec* spec, void** state_out,
                                     cudaStream_t stream) {
    if (!rcu_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    RcuRefState* st = (RcuRefState*)malloc(sizeof(RcuRefState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(RcuRefState));
    memcpy(&st->spec, spec, sizeof(RcuProblemSpec));

    cudaError_t e = rref_alloc_all(st);
    if (e != cudaSuccess) { solution_destroy(st); return e; }

    rref_reset_kernel<<<256, 256, 0, stream>>>(*st, spec->max_edges_per_object,
                                               spec->root_count, spec->thread_count);
    e = cudaPeekAtLastError();
    if (e != cudaSuccess) { solution_destroy(st); return e; }

    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(void* state, const RcuRunSpec* run,
                                    const void* inputs_void, void* outputs_void,
                                    void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    (void)workspace;
    if (!state || !rcu_validate_run_spec(run) || !inputs_void || !outputs_void)
        return cudaErrorInvalidValue;
    if (workspace_bytes < 1) return cudaErrorInvalidValue;

    RcuRefState* st = (RcuRefState*)state;
    const RcuInputs* in = (const RcuInputs*)inputs_void;
    RcuOutputs* out = (RcuOutputs*)outputs_void;

    if (run->op_count > st->spec.max_ops) return cudaErrorInvalidValue;
    if (run->op_count > 0 && !in->ops) return cudaErrorInvalidValue;
    if (!out->counts || !out->event_hash || !out->read_hash || !out->root_hash ||
        !out->object_hash || !out->thread_hash || !out->free_hash || !out->state_scalars)
        return cudaErrorInvalidValue;

    const int ME = st->spec.max_edges_per_object;
    const int RC = st->spec.root_count;
    const int TC = st->spec.thread_count;

    if (run->op_count > 0) {
        rcu_ref_run_kernel<<<1, 1, 0, stream>>>(*st, in->ops, run->op_count, ME, RC, TC);
        cudaError_t e = cudaPeekAtLastError();
        if (e != cudaSuccess) return e;
    }

    rcu_ref_finalize_kernel<<<1, 4, 0, stream>>>(*st, ME, RC, TC,
                                                 out->root_hash, out->object_hash,
                                                 out->thread_hash, out->free_hash);
    cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) return e;

    // copy scalars + counts + event/read hashes to outputs.
    e = cudaMemcpyAsync(out->counts, st->counts, sizeof(uint64_t) * RCU_COUNT_TOTAL,
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(out->event_hash, st->event_hash, sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(out->read_hash, st->read_hash, sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(out->state_scalars, st->sc, sizeof(uint64_t) * 6,
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    RcuRefState* st = (RcuRefState*)state;
    rref_reset_kernel<<<256, 256, 0, stream>>>(*st, st->spec.max_edges_per_object,
                                               st->spec.root_count, st->spec.thread_count);
    return cudaPeekAtLastError();
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    RcuRefState* st = (RcuRefState*)state;
    cudaFree(st->sc); cudaFree(st->op_index_global); cudaFree(st->event_hash);
    cudaFree(st->read_hash); cudaFree(st->counts);
    cudaFree(st->o_present); cudaFree(st->o_value); cudaFree(st->o_alloc_seq);
    cudaFree(st->o_retired); cudaFree(st->o_retire_seq); cudaFree(st->o_retire_epoch);
    cudaFree(st->o_callback_tag); cudaFree(st->o_edges); cudaFree(st->o_incoming);
    cudaFree(st->root);
    cudaFree(st->th_read_nest); cudaFree(st->th_outer_seq);
    cudaFree(st->th_active_epoch); cudaFree(st->th_root_view);
    cudaFree(st->retired_ids); cudaFree(st->retired_count);
    cudaFree(st->free_obj); cudaFree(st->free_seq); cudaFree(st->free_count);
    cudaFree(st->present_total);
    free(st);
}
