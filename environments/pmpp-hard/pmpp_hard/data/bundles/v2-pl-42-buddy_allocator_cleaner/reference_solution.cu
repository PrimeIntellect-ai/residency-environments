// PMPP_CANARY_42_67c208025c -- held-out canary; MUST NOT appear in any submission
// file: buddy_allocator_cleaner_reference.cu
//
// Reference (implementation #2): on-device incremental model, cooperative block.
// Persistent state lives in device memory; each solution_run launches one
// <<<1,256>>> kernel. Phase 1 (thread 0): apply the op batch in stream order,
// mutating persistent state and emitting the cumulative event hashes/counters.
// This op replay is an inherently-serial data-dependent chain (each op reads
// the free-list/segment/object state the previous op mutated), so it stays on
// a single thread. Phase 2 (all 256 threads): recompute the per-step structural
// hashes (object / segment / buddy) by PARALLEL RANK-SORT -- rank[i] = #elements
// with key < key[i]. Each hashed set is a strict total order (obj_id unique;
// (base,seg_id) unique; (order,base) unique), so rank-sort reproduces the exact
// same ordered permutation as a serial selection sort, and the dependent FNV
// fold over that order yields byte-identical hashes. Only the O(n^2) ordering
// is parallelized; the fold stays serial. This collapses the structural-hash
// cost (~87% of the old <<<1,1>>> runtime) while remaining bit-exact. Free list
// is a flat {order,base} set scanned linearly; segment/object tables are flat
// arrays with linear search. Algorithmically independent from the naive replay
// model and from the host oracle.

#include "buddy_allocator_cleaner_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---- event-kind constants (must match oracle byte values) ----
#define REF_EK_ALLOC_OK 0
#define REF_EK_ALLOC_OOM 1
#define REF_EK_RELOCATE_OBJECT 2
#define REF_FK_FREE_DEFERRED 0
#define REF_FK_FREE_DEFERRED_DUP 1
#define REF_FK_OBJECT_FINALIZE 2
#define REF_FK_SEAL_IMPLICIT 3
#define REF_FK_SEAL_EXPLICIT 4
#define REF_FK_SEAL_EMPTY 5
#define REF_FK_CLEAN_BLOCKED_SEGMENT 6
#define REF_FK_SEGMENT_RECLAIM 7
#define REF_RSN_NONE 0
#define REF_RSN_FREE_IMMEDIATE 1
#define REF_RSN_UNPIN_FINALIZE 2
#define REF_U64MAX 0xFFFFFFFFFFFFFFFFULL
#define REF_ORDER_NONE 255
#define REF_FNV_BASIS 1469598103934665603ULL
#define REF_FNV_PRIME 1099511628211ULL

struct BacReferenceState {
    BacProblemSpec spec;
    int max_order;
    int segment_order;
    uint64_t total_pages;
    uint64_t segment_pages;
    uint64_t num_units;          // total_pages / segment_pages
    int free_cap;
    int seg_cap;                 // max_segments
    int obj_cap;                 // max_objects

    // free list (flat): orders[i], bases[i], for i in [0,free_count)
    uint8_t* d_free_order;
    uint64_t* d_free_base;
    int32_t* d_free_count;

    // segment table (flat, alive flag)
    uint8_t* d_seg_alive;
    uint64_t* d_seg_id;
    uint32_t* d_seg_class;
    uint64_t* d_seg_base;
    uint64_t* d_seg_aoff;
    uint64_t* d_seg_live;
    uint64_t* d_seg_dead;
    uint8_t* d_seg_sealed;

    // object table (flat, alive flag)
    uint8_t* d_obj_alive;
    uint64_t* d_obj_id;
    uint32_t* d_obj_class;
    uint64_t* d_obj_seg;
    uint64_t* d_obj_base;
    uint8_t* d_obj_order;
    uint64_t* d_obj_block;
    uint64_t* d_obj_req;
    uint64_t* d_obj_pin;
    uint8_t* d_obj_pending;
    uint64_t* d_obj_aseq;
    uint64_t* d_obj_mseq;

    // scalars / globals
    uint64_t* d_active_seg;       // [num_classes]
    uint64_t* d_next_seg_id;      // [1]
    uint64_t* d_event_seq;        // [1]
    uint32_t* d_op_index;         // [1]
    uint64_t* d_counters;         // [16]
    uint64_t* d_hashes;           // [5] alloc_event,finalize,buddy,segment,object

    // scratch for clean sorting (movable obj indices)
    int32_t* d_scratch;           // [obj_cap]

    // ordered-index scratch for parallel rank-sort structural hashes
    int32_t* d_ord_obj;           // [obj_cap]
    int32_t* d_ord_seg;           // [seg_cap]
    int32_t* d_ord_free;          // [free_cap]

    // device-side copy of op batch
    BacOp* d_ops;
};

// counter indices
enum {
    C_ALLOC_OK=0, C_ALLOC_OOM, C_FREE_FIN, C_FREE_DEF, C_PIN, C_UNPIN,
    C_SEAL_EXP, C_SEAL_IMP, C_SEAL_EMP, C_RELOC, C_BLOCKED, C_SEGREC,
    C_SPLIT, C_MERGE, C_PAD, C_INVALID
};

// ---- FNV device helpers ----
__device__ __forceinline__ uint64_t ref_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= REF_FNV_PRIME; return h;
}
__device__ void ref_fnv_u8(uint64_t* h, uint8_t v) { *h = ref_fnv_byte(*h, v); }
__device__ void ref_fnv_u32(uint64_t* h, uint32_t v) {
    uint64_t x = *h; const uint8_t* p = (const uint8_t*)&v;
    for (int i = 0; i < 4; ++i) x = ref_fnv_byte(x, p[i]); *h = x;
}
__device__ void ref_fnv_u64(uint64_t* h, uint64_t v) {
    uint64_t x = *h; const uint8_t* p = (const uint8_t*)&v;
    for (int i = 0; i < 8; ++i) x = ref_fnv_byte(x, p[i]); *h = x;
}

// device-resident "this state" pointers grouped for kernel
struct RefDev {
    int max_order, segment_order, num_classes, max_objects, max_segments;
    uint64_t total_pages, segment_pages;
    uint8_t* free_order; uint64_t* free_base; int32_t* free_count;
    uint8_t* seg_alive; uint64_t* seg_id; uint32_t* seg_class; uint64_t* seg_base;
    uint64_t* seg_aoff; uint64_t* seg_live; uint64_t* seg_dead; uint8_t* seg_sealed;
    int seg_cap;
    uint8_t* obj_alive; uint64_t* obj_id; uint32_t* obj_class; uint64_t* obj_seg;
    uint64_t* obj_base; uint8_t* obj_order; uint64_t* obj_block; uint64_t* obj_req;
    uint64_t* obj_pin; uint8_t* obj_pending; uint64_t* obj_aseq; uint64_t* obj_mseq;
    int obj_cap;
    uint64_t* active_seg; uint64_t* next_seg_id; uint64_t* event_seq; uint32_t* op_index;
    uint64_t* counters; uint64_t* hashes;
    int32_t* scratch;
};

__device__ int ref_ceil_log2(uint64_t n) {
    int o = 0; uint64_t p = 1;
    while (p < n) { p <<= 1; ++o; }
    return o;
}

// find segment slot by id; -1 if none
__device__ int ref_find_seg(RefDev* d, uint64_t sid) {
    for (int i = 0; i < d->seg_cap; ++i)
        if (d->seg_alive[i] && d->seg_id[i] == sid) return i;
    return -1;
}
__device__ int ref_find_obj(RefDev* d, uint64_t oid) {
    for (int i = 0; i < d->obj_cap; ++i)
        if (d->obj_alive[i] && d->obj_id[i] == oid) return i;
    return -1;
}
__device__ int ref_alloc_seg_slot(RefDev* d) {
    for (int i = 0; i < d->seg_cap; ++i) if (!d->seg_alive[i]) return i;
    return -1;
}
__device__ int ref_alloc_obj_slot(RefDev* d) {
    for (int i = 0; i < d->obj_cap; ++i) if (!d->obj_alive[i]) return i;
    return -1;
}
__device__ int ref_count_objs(RefDev* d) {
    int c = 0; for (int i = 0; i < d->obj_cap; ++i) if (d->obj_alive[i]) ++c; return c;
}
__device__ int ref_count_segs(RefDev* d) {
    int c = 0; for (int i = 0; i < d->seg_cap; ++i) if (d->seg_alive[i]) ++c; return c;
}

// free list contains (order, base)? return index or -1
__device__ int ref_free_find(RefDev* d, int order, uint64_t base) {
    int n = d->free_count[0];
    for (int i = 0; i < n; ++i)
        if (d->free_order[i] == (uint8_t)order && d->free_base[i] == base) return i;
    return -1;
}
__device__ void ref_free_insert(RefDev* d, int order, uint64_t base) {
    int n = d->free_count[0];
    d->free_order[n] = (uint8_t)order;
    d->free_base[n] = base;
    d->free_count[0] = n + 1;
}
__device__ void ref_free_remove_idx(RefDev* d, int idx) {
    int n = d->free_count[0];
    d->free_order[idx] = d->free_order[n-1];
    d->free_base[idx] = d->free_base[n-1];
    d->free_count[0] = n - 1;
}

// buddy_alloc at want_order: choose smallest order>=want with a block, lowest
// base, split down keeping left, inserting right halves. Returns base or marks
// fail via ok flag.
__device__ bool ref_buddy_alloc(RefDev* d, int want_order, uint64_t* base_out) {
    int best_idx = -1; int best_order = 1<<30; uint64_t best_base = 0;
    int n = d->free_count[0];
    for (int i = 0; i < n; ++i) {
        int o = d->free_order[i];
        if (o < want_order) continue;
        if (o < best_order || (o == best_order && d->free_base[i] < best_base)) {
            best_order = o; best_base = d->free_base[i]; best_idx = i;
        }
    }
    if (best_idx < 0) return false;
    int k = best_order; uint64_t base = best_base;
    ref_free_remove_idx(d, best_idx);
    while (k > want_order) {
        int child = k - 1;
        uint64_t right = base + ((uint64_t)1 << child);
        ref_free_insert(d, child, right);
        d->counters[C_SPLIT] += 1;
        k = child;
    }
    *base_out = base;
    return true;
}

__device__ void ref_buddy_free(RefDev* d, uint64_t base, int order) {
    int o = order; uint64_t b = base;
    while (o < d->max_order) {
        uint64_t buddy = b ^ ((uint64_t)1 << o);
        int idx = ref_free_find(d, o, buddy);
        if (idx >= 0) {
            ref_free_remove_idx(d, idx);
            b = (b < buddy) ? b : buddy;
            o += 1;
            d->counters[C_MERGE] += 1;
        } else break;
    }
    ref_free_insert(d, o, b);
}

__device__ void ref_emit_alloc(RefDev* d, uint8_t kind, uint64_t seq, uint32_t opidx,
                               uint64_t obj_id, uint32_t cls, uint64_t seg, uint64_t base,
                               uint8_t order, uint64_t req, uint64_t block) {
    uint64_t h = d->hashes[0];
    ref_fnv_u8(&h, kind); ref_fnv_u64(&h, seq); ref_fnv_u32(&h, opidx);
    ref_fnv_u64(&h, obj_id); ref_fnv_u32(&h, cls); ref_fnv_u64(&h, seg);
    ref_fnv_u64(&h, base); ref_fnv_u8(&h, order); ref_fnv_u64(&h, req);
    ref_fnv_u64(&h, block);
    d->hashes[0] = h;
}
__device__ void ref_emit_final(RefDev* d, uint8_t kind, uint64_t seq, uint32_t opidx,
                               uint64_t obj_id, uint64_t seg, uint64_t base,
                               uint64_t pages, uint8_t reason) {
    uint64_t h = d->hashes[1];
    ref_fnv_u8(&h, kind); ref_fnv_u64(&h, seq); ref_fnv_u32(&h, opidx);
    ref_fnv_u64(&h, obj_id); ref_fnv_u64(&h, seg); ref_fnv_u64(&h, base);
    ref_fnv_u64(&h, pages); ref_fnv_u8(&h, reason);
    d->hashes[1] = h;
}

__device__ void ref_seal_segment(RefDev* d, int sidx, bool implicit, uint64_t seq, uint32_t opidx) {
    uint64_t tail = d->segment_pages - d->seg_aoff[sidx];
    d->seg_dead[sidx] += tail;
    d->seg_aoff[sidx] = d->segment_pages;
    d->seg_sealed[sidx] = 1;
    if (implicit) {
        d->counters[C_SEAL_IMP] += 1;
        ref_emit_final(d, REF_FK_SEAL_IMPLICIT, seq, opidx, REF_U64MAX,
                       d->seg_id[sidx], d->seg_base[sidx], tail, REF_RSN_NONE);
    } else {
        d->counters[C_SEAL_EXP] += 1;
        ref_emit_final(d, REF_FK_SEAL_EXPLICIT, seq, opidx, REF_U64MAX,
                       d->seg_id[sidx], d->seg_base[sidx], tail, REF_RSN_NONE);
    }
}

// append allocator: returns true and sets *seg_id/*base on success.
__device__ bool ref_append_allocate(RefDev* d, uint32_t cls, int order, uint64_t block,
                                     uint64_t seq, uint32_t opidx,
                                     uint64_t* seg_id_out, uint64_t* base_out) {
    uint64_t active_id = d->active_seg[cls];
    bool need_new = false; uint64_t aligned_offset = 0; int active_idx = -1;
    if (active_id == 0) {
        need_new = true;
    } else {
        active_idx = ref_find_seg(d, active_id);
        uint64_t off = d->seg_aoff[active_idx];
        aligned_offset = (off + block - 1) & ~(block - 1);
        if (aligned_offset + block > d->segment_pages) need_new = true;
    }

    if (need_new) {
        if (active_id != 0) {
            ref_seal_segment(d, active_idx, true, seq, opidx);
            d->active_seg[cls] = 0;
            active_id = 0;
        }
        if (ref_count_segs(d) >= d->max_segments) return false;
        uint64_t new_base = 0;
        if (!ref_buddy_alloc(d, d->segment_order, &new_base)) return false;
        int slot = ref_alloc_seg_slot(d);
        // slot guaranteed (count<max_segments)
        uint64_t sid = d->next_seg_id[0];
        d->next_seg_id[0] = sid + 1;
        d->seg_alive[slot] = 1;
        d->seg_id[slot] = sid;
        d->seg_class[slot] = cls;
        d->seg_base[slot] = new_base;
        d->seg_aoff[slot] = 0;
        d->seg_live[slot] = 0;
        d->seg_dead[slot] = 0;
        d->seg_sealed[slot] = 0;
        d->active_seg[cls] = sid;
        active_id = sid; active_idx = slot; aligned_offset = 0;
    }

    int sidx = (active_idx >= 0) ? active_idx : ref_find_seg(d, active_id);
    uint64_t padding = aligned_offset - d->seg_aoff[sidx];
    uint64_t base_page = d->seg_base[sidx] + aligned_offset;
    d->seg_aoff[sidx] = aligned_offset + block;
    d->seg_live[sidx] += block;
    d->seg_dead[sidx] += padding;
    d->counters[C_PAD] += padding;
    *seg_id_out = d->seg_id[sidx];
    *base_out = base_page;
    return true;
}

__device__ void ref_finalize_object(RefDev* d, int oidx, uint64_t seq, uint32_t opidx, uint8_t reason) {
    uint64_t sid = d->obj_seg[oidx];
    int sidx = ref_find_seg(d, sid);
    d->seg_live[sidx] -= d->obj_block[oidx];
    d->seg_dead[sidx] += d->obj_block[oidx];
    d->counters[C_FREE_FIN] += 1;
    ref_emit_final(d, REF_FK_OBJECT_FINALIZE, seq, opidx, d->obj_id[oidx], sid,
                   d->obj_base[oidx], d->obj_block[oidx], reason);
    d->obj_alive[oidx] = 0;
}

__device__ void ref_reclaim_segment(RefDev* d, int sidx, uint64_t seq, uint32_t opidx) {
    uint64_t sid = d->seg_id[sidx];
    uint32_t cls = d->seg_class[sidx];
    uint64_t base = d->seg_base[sidx];
    if (d->active_seg[cls] == sid) d->active_seg[cls] = 0;
    d->seg_alive[sidx] = 0;
    ref_buddy_free(d, base, d->segment_order);
    d->counters[C_SEGREC] += 1;
    ref_emit_final(d, REF_FK_SEGMENT_RECLAIM, seq, opidx, REF_U64MAX, sid, base,
                   d->segment_pages, REF_RSN_NONE);
}

__device__ void ref_op_alloc(RefDev* d, const BacOp& op, uint64_t seq, uint32_t opidx) {
    uint32_t cls = (uint32_t)op.class_id;
    uint64_t req = op.a;
    if ((int)cls >= d->num_classes || req == 0 || req > d->segment_pages ||
        ref_find_obj(d, op.obj_id) >= 0) {
        d->counters[C_INVALID] += 1; return;
    }
    if (ref_count_objs(d) >= d->max_objects) {
        d->counters[C_ALLOC_OOM] += 1;
        ref_emit_alloc(d, REF_EK_ALLOC_OOM, seq, opidx, op.obj_id, cls,
                       REF_U64MAX, REF_U64MAX, REF_ORDER_NONE, req, 0);
        return;
    }
    int order = ref_ceil_log2(req);
    uint64_t block = (uint64_t)1 << order;
    uint64_t seg_id = 0, base = 0;
    if (!ref_append_allocate(d, cls, order, block, seq, opidx, &seg_id, &base)) {
        d->counters[C_ALLOC_OOM] += 1;
        ref_emit_alloc(d, REF_EK_ALLOC_OOM, seq, opidx, op.obj_id, cls,
                       REF_U64MAX, REF_U64MAX, REF_ORDER_NONE, req, block);
        return;
    }
    int slot = ref_alloc_obj_slot(d);
    d->obj_alive[slot] = 1;
    d->obj_id[slot] = op.obj_id;
    d->obj_class[slot] = cls;
    d->obj_seg[slot] = seg_id;
    d->obj_base[slot] = base;
    d->obj_order[slot] = (uint8_t)order;
    d->obj_block[slot] = block;
    d->obj_req[slot] = req;
    d->obj_pin[slot] = 0;
    d->obj_pending[slot] = 0;
    d->obj_aseq[slot] = seq;
    d->obj_mseq[slot] = seq;
    d->counters[C_ALLOC_OK] += 1;
    ref_emit_alloc(d, REF_EK_ALLOC_OK, seq, opidx, op.obj_id, cls, seg_id, base,
                   (uint8_t)order, req, block);
}

__device__ void ref_op_free(RefDev* d, const BacOp& op, uint64_t seq, uint32_t opidx) {
    int oidx = ref_find_obj(d, op.obj_id);
    if (oidx < 0) { d->counters[C_INVALID] += 1; return; }
    if (d->obj_pin[oidx] > 0) {
        if (d->obj_pending[oidx] == 1) {
            d->counters[C_FREE_DEF] += 1;
            ref_emit_final(d, REF_FK_FREE_DEFERRED_DUP, seq, opidx, d->obj_id[oidx],
                           d->obj_seg[oidx], d->obj_base[oidx], d->obj_block[oidx], REF_RSN_NONE);
            return;
        }
        d->obj_pending[oidx] = 1;
        d->counters[C_FREE_DEF] += 1;
        ref_emit_final(d, REF_FK_FREE_DEFERRED, seq, opidx, d->obj_id[oidx],
                       d->obj_seg[oidx], d->obj_base[oidx], d->obj_block[oidx], REF_RSN_NONE);
        return;
    }
    ref_finalize_object(d, oidx, seq, opidx, REF_RSN_FREE_IMMEDIATE);
}

__device__ void ref_op_pin(RefDev* d, const BacOp& op, uint64_t seq, uint32_t opidx) {
    int oidx = ref_find_obj(d, op.obj_id);
    if (oidx < 0) { d->counters[C_INVALID] += 1; return; }
    if (d->obj_pending[oidx] == 1) { d->counters[C_INVALID] += 1; return; }
    if (d->obj_pin[oidx] == REF_U64MAX) { d->counters[C_INVALID] += 1; return; }
    d->obj_pin[oidx] += 1;
    d->counters[C_PIN] += 1;
}

__device__ void ref_op_unpin(RefDev* d, const BacOp& op, uint64_t seq, uint32_t opidx) {
    int oidx = ref_find_obj(d, op.obj_id);
    if (oidx < 0) { d->counters[C_INVALID] += 1; return; }
    if (d->obj_pin[oidx] == 0) { d->counters[C_INVALID] += 1; return; }
    d->obj_pin[oidx] -= 1;
    d->counters[C_UNPIN] += 1;
    if (d->obj_pin[oidx] == 0 && d->obj_pending[oidx] == 1) {
        ref_finalize_object(d, oidx, seq, opidx, REF_RSN_UNPIN_FINALIZE);
    }
}

__device__ void ref_op_seal(RefDev* d, const BacOp& op, uint64_t seq, uint32_t opidx) {
    uint32_t cls = (uint32_t)op.class_id;
    if ((int)cls >= d->num_classes) { d->counters[C_INVALID] += 1; return; }
    uint64_t active_id = d->active_seg[cls];
    if (active_id == 0) {
        d->counters[C_SEAL_EMP] += 1;
        ref_emit_final(d, REF_FK_SEAL_EMPTY, seq, opidx, REF_U64MAX, REF_U64MAX, 0, 0, REF_RSN_NONE);
        return;
    }
    int sidx = ref_find_seg(d, active_id);
    ref_seal_segment(d, sidx, false, seq, opidx);
    d->active_seg[cls] = 0;
}

// victim selection: among sealed, dead>0. key: largest dead; tie smallest live;
// tie smallest base; tie smallest segment_id. Returns slot index or -1.
__device__ int ref_select_victim(RefDev* d) {
    int best = -1;
    for (int i = 0; i < d->seg_cap; ++i) {
        if (!d->seg_alive[i] || d->seg_sealed[i] == 0 || d->seg_dead[i] == 0) continue;
        if (best < 0) { best = i; continue; }
        bool better;
        if (d->seg_dead[i] != d->seg_dead[best]) better = d->seg_dead[i] > d->seg_dead[best];
        else if (d->seg_live[i] != d->seg_live[best]) better = d->seg_live[i] < d->seg_live[best];
        else if (d->seg_base[i] != d->seg_base[best]) better = d->seg_base[i] < d->seg_base[best];
        else better = d->seg_id[i] < d->seg_id[best];
        if (better) best = i;
    }
    return best;
}

__device__ void ref_op_clean(RefDev* d, const BacOp& op, uint64_t seq, uint32_t opidx) {
    uint64_t max_segs = op.a;
    uint64_t budget = op.b;
    if (max_segs == 0 || budget == 0) return;

    while (max_segs > 0 && budget > 0) {
        int vic = ref_select_victim(d);
        if (vic < 0) break;
        uint64_t vic_id = d->seg_id[vic];

        if (d->seg_live[vic] == 0) {
            ref_reclaim_segment(d, vic, seq, opidx);
            max_segs -= 1;
            continue;
        }

        // gather movable obj slots; sort by (base,obj_id)
        int m = 0;
        for (int i = 0; i < d->obj_cap; ++i) {
            if (!d->obj_alive[i]) continue;
            if (d->obj_seg[i] != vic_id) continue;
            if (d->obj_pin[i] != 0 || d->obj_pending[i] != 0) continue;
            d->scratch[m++] = i;
        }
        // insertion sort (m small)
        for (int a = 1; a < m; ++a) {
            int cur = d->scratch[a]; int b = a - 1;
            while (b >= 0) {
                int prev = d->scratch[b];
                bool gt = (d->obj_base[prev] > d->obj_base[cur]) ||
                          (d->obj_base[prev] == d->obj_base[cur] && d->obj_id[prev] > d->obj_id[cur]);
                if (!gt) break;
                d->scratch[b+1] = prev; --b;
            }
            d->scratch[b+1] = cur;
        }

        if (m == 0) {
            d->counters[C_BLOCKED] += 1;
            ref_emit_final(d, REF_FK_CLEAN_BLOCKED_SEGMENT, seq, opidx, REF_U64MAX,
                           vic_id, d->seg_base[vic], d->seg_live[vic], REF_RSN_NONE);
            max_segs -= 1;
            continue;
        }

        bool stop_clean = false;
        for (int t = 0; t < m; ++t) {
            int oidx = d->scratch[t];
            uint64_t block = d->obj_block[oidx];
            if (block > budget) { stop_clean = true; break; }
            int order = (int)d->obj_order[oidx];
            uint32_t cls = d->obj_class[oidx];
            uint64_t dst_seg = 0, dst_base = 0;
            if (!ref_append_allocate(d, cls, order, block, seq, opidx, &dst_seg, &dst_base)) {
                stop_clean = true; break;
            }
            // re-find victim slot (append_allocate may have moved nothing, but
            // seg array indices are stable; vic still valid since vic not freed here)
            d->seg_live[vic] -= block;
            d->seg_dead[vic] += block;
            d->obj_seg[oidx] = dst_seg;
            d->obj_base[oidx] = dst_base;
            d->obj_mseq[oidx] = seq;
            d->counters[C_RELOC] += 1;
            ref_emit_alloc(d, REF_EK_RELOCATE_OBJECT, seq, opidx, d->obj_id[oidx], cls,
                           dst_seg, dst_base, (uint8_t)order, d->obj_req[oidx], block);
            budget -= block;
            if (d->seg_live[vic] == 0) {
                ref_reclaim_segment(d, vic, seq, opidx);
                max_segs -= 1;
                break;
            }
        }
        if (stop_clean) break;
        // re-select on next loop iteration (faithful to contract).
    }
}

__global__ void ref_step_kernel(
    int max_order, int segment_order, int num_classes, int max_objects, int max_segments,
    uint64_t total_pages, uint64_t segment_pages,
    uint8_t* free_order, uint64_t* free_base, int32_t* free_count,
    uint8_t* seg_alive, uint64_t* seg_id, uint32_t* seg_class, uint64_t* seg_base,
    uint64_t* seg_aoff, uint64_t* seg_live, uint64_t* seg_dead, uint8_t* seg_sealed, int seg_cap,
    uint8_t* obj_alive, uint64_t* obj_id, uint32_t* obj_class, uint64_t* obj_seg,
    uint64_t* obj_base, uint8_t* obj_order, uint64_t* obj_block, uint64_t* obj_req,
    uint64_t* obj_pin, uint8_t* obj_pending, uint64_t* obj_aseq, uint64_t* obj_mseq, int obj_cap,
    uint64_t* active_seg, uint64_t* next_seg_id, uint64_t* event_seq, uint32_t* op_index,
    uint64_t* counters, uint64_t* hashes, int32_t* scratch,
    int32_t* ord_obj, int32_t* ord_seg, int32_t* ord_free,
    const BacOp* ops, int num_ops,
    // outputs
    uint64_t* o_counters, uint64_t* o_alloc_hash, uint64_t* o_final_hash,
    uint64_t* o_buddy_hash, uint64_t* o_seg_hash, uint64_t* o_obj_hash,
    uint64_t* o_live_obj, uint64_t* o_live_seg) {
    if (blockIdx.x != 0) return;
    const int tid = threadIdx.x;
    const int NT = blockDim.x;
    __shared__ int s_nobj;
    __shared__ int s_nseg;

    RefDev d;
    d.max_order = max_order; d.segment_order = segment_order; d.num_classes = num_classes;
    d.max_objects = max_objects; d.max_segments = max_segments;
    d.total_pages = total_pages; d.segment_pages = segment_pages;
    d.free_order = free_order; d.free_base = free_base; d.free_count = free_count;
    d.seg_alive = seg_alive; d.seg_id = seg_id; d.seg_class = seg_class; d.seg_base = seg_base;
    d.seg_aoff = seg_aoff; d.seg_live = seg_live; d.seg_dead = seg_dead; d.seg_sealed = seg_sealed;
    d.seg_cap = seg_cap;
    d.obj_alive = obj_alive; d.obj_id = obj_id; d.obj_class = obj_class; d.obj_seg = obj_seg;
    d.obj_base = obj_base; d.obj_order = obj_order; d.obj_block = obj_block; d.obj_req = obj_req;
    d.obj_pin = obj_pin; d.obj_pending = obj_pending; d.obj_aseq = obj_aseq; d.obj_mseq = obj_mseq;
    d.obj_cap = obj_cap;
    d.active_seg = active_seg; d.next_seg_id = next_seg_id; d.event_seq = event_seq;
    d.op_index = op_index; d.counters = counters; d.hashes = hashes; d.scratch = scratch;

    // ---- Phase 1: serial op replay (inherently serial dependency chain) ----
    if (tid == 0) {
    for (int i = 0; i < num_ops; ++i) {
        event_seq[0] += 1;
        uint64_t seq = event_seq[0];
        uint32_t opidx = op_index[0];
        op_index[0] += 1;
        const BacOp op = ops[i];
        switch (op.op_type) {
            case BAC_OP_ALLOC: ref_op_alloc(&d, op, seq, opidx); break;
            case BAC_OP_FREE:  ref_op_free(&d, op, seq, opidx); break;
            case BAC_OP_PIN:   ref_op_pin(&d, op, seq, opidx); break;
            case BAC_OP_UNPIN: ref_op_unpin(&d, op, seq, opidx); break;
            case BAC_OP_SEAL:  ref_op_seal(&d, op, seq, opidx); break;
            case BAC_OP_CLEAN: ref_op_clean(&d, op, seq, opidx); break;
            default: counters[C_INVALID] += 1; break;
        }
    }
        s_nobj = ref_count_objs(&d);
        s_nseg = ref_count_segs(&d);
    }
    __syncthreads();

    // ---- Phase 2: cooperative structural hashes (byte-exact rank-sort) ----
    // Each structural hash enumerates a strict-total-order key set. Rank-sort
    // (rank[i] = #elements with key < key[i]) reproduces the exact same ordered
    // permutation as the oracle's selection sort, so the serial FNV fold over
    // that order yields byte-identical hashes. Only the O(n^2) rank computation
    // is parallelized across the block; the dependent fold stays serial on t0.
    int fn = free_count[0];

    // Object rank: order by obj_id ascending.
    for (int i = tid; i < obj_cap; i += NT) {
        if (!obj_alive[i]) continue;
        uint64_t id = obj_id[i];
        int rank = 0;
        for (int j = 0; j < obj_cap; ++j)
            if (obj_alive[j] && obj_id[j] < id) ++rank;
        ord_obj[rank] = i;
    }
    // Segment rank: order by (base ascending, then seg_id ascending).
    for (int i = tid; i < seg_cap; i += NT) {
        if (!seg_alive[i]) continue;
        uint64_t b = seg_base[i], id = seg_id[i];
        int rank = 0;
        for (int j = 0; j < seg_cap; ++j) {
            if (!seg_alive[j]) continue;
            uint64_t bj = seg_base[j], idj = seg_id[j];
            if (bj < b || (bj == b && idj < id)) ++rank;
        }
        ord_seg[rank] = i;
    }
    // Buddy rank: order by (order ascending, then base ascending).
    for (int i = tid; i < fn; i += NT) {
        int o = free_order[i]; uint64_t b = free_base[i];
        int rank = 0;
        for (int j = 0; j < fn; ++j) {
            int oj = free_order[j]; uint64_t bj = free_base[j];
            if (oj < o || (oj == o && bj < b)) ++rank;
        }
        ord_free[rank] = i;
    }
    __syncthreads();

    if (tid == 0) {
        // buddy fold: (order asc, base asc)
        uint64_t bh = REF_FNV_BASIS;
        for (int r = 0; r < fn; ++r) {
            int i = ord_free[r];
            ref_fnv_u8(&bh, free_order[i]);
            ref_fnv_u64(&bh, free_base[i]);
        }
        o_buddy_hash[0] = bh;

        // segment fold: (base asc, seg_id asc)
        uint64_t sh = REF_FNV_BASIS;
        for (int r = 0; r < s_nseg; ++r) {
            int i = ord_seg[r];
            uint8_t is_active = (active_seg[seg_class[i]] == seg_id[i]) ? 1 : 0;
            ref_fnv_u64(&sh, seg_id[i]);
            ref_fnv_u32(&sh, seg_class[i]);
            ref_fnv_u64(&sh, seg_base[i]);
            ref_fnv_u64(&sh, seg_aoff[i]);
            ref_fnv_u64(&sh, seg_live[i]);
            ref_fnv_u64(&sh, seg_dead[i]);
            ref_fnv_u8(&sh, seg_sealed[i]);
            ref_fnv_u8(&sh, is_active);
        }
        o_seg_hash[0] = sh;

        // object fold: obj_id asc
        uint64_t oh = REF_FNV_BASIS;
        for (int r = 0; r < s_nobj; ++r) {
            int i = ord_obj[r];
            ref_fnv_u64(&oh, obj_id[i]);
            ref_fnv_u32(&oh, obj_class[i]);
            ref_fnv_u64(&oh, obj_seg[i]);
            ref_fnv_u64(&oh, obj_base[i]);
            ref_fnv_u8(&oh, obj_order[i]);
            ref_fnv_u64(&oh, obj_req[i]);
            ref_fnv_u64(&oh, obj_block[i]);
            ref_fnv_u64(&oh, obj_pin[i]);
            ref_fnv_u8(&oh, obj_pending[i]);
            ref_fnv_u64(&oh, obj_aseq[i]);
            ref_fnv_u64(&oh, obj_mseq[i]);
        }
        o_obj_hash[0] = oh;

        for (int i = 0; i < 16; ++i) o_counters[i] = counters[i];
        o_alloc_hash[0] = hashes[0];
        o_final_hash[0] = hashes[1];
        o_live_obj[0] = (uint64_t)s_nobj;
        o_live_seg[0] = (uint64_t)s_nseg;
    }
}

// ---------------------------------------------------------------------------
// Host glue.
// ---------------------------------------------------------------------------
static cudaError_t ref_reset_state(BacReferenceState* st, cudaStream_t stream) {
    cudaError_t err;
    // zero everything, then set initial free block + scalars via a small kernel.
    err = cudaMemsetAsync(st->d_free_count, 0, sizeof(int32_t), stream); if (err) return err;
    err = cudaMemsetAsync(st->d_seg_alive, 0, sizeof(uint8_t) * st->seg_cap, stream); if (err) return err;
    err = cudaMemsetAsync(st->d_obj_alive, 0, sizeof(uint8_t) * st->obj_cap, stream); if (err) return err;
    err = cudaMemsetAsync(st->d_active_seg, 0, sizeof(uint64_t) * st->spec.num_classes, stream); if (err) return err;
    err = cudaMemsetAsync(st->d_event_seq, 0, sizeof(uint64_t), stream); if (err) return err;
    err = cudaMemsetAsync(st->d_op_index, 0, sizeof(uint32_t), stream); if (err) return err;
    err = cudaMemsetAsync(st->d_counters, 0, sizeof(uint64_t) * 16, stream); if (err) return err;

    // next_seg_id = 1
    uint64_t one = 1;
    err = cudaMemcpyAsync(st->d_next_seg_id, &one, sizeof(uint64_t), cudaMemcpyHostToDevice, stream); if (err) return err;
    // hashes = FNV basis
    uint64_t hb[5] = {REF_FNV_BASIS, REF_FNV_BASIS, REF_FNV_BASIS, REF_FNV_BASIS, REF_FNV_BASIS};
    err = cudaMemcpyAsync(st->d_hashes, hb, sizeof(hb), cudaMemcpyHostToDevice, stream); if (err) return err;
    // free list: one block at order max_order, base 0
    int32_t fc = 1;
    uint8_t fo = (uint8_t)st->max_order;
    uint64_t fb = 0;
    err = cudaMemcpyAsync(st->d_free_count, &fc, sizeof(int32_t), cudaMemcpyHostToDevice, stream); if (err) return err;
    err = cudaMemcpyAsync(st->d_free_order, &fo, sizeof(uint8_t), cudaMemcpyHostToDevice, stream); if (err) return err;
    err = cudaMemcpyAsync(st->d_free_base, &fb, sizeof(uint64_t), cudaMemcpyHostToDevice, stream); if (err) return err;
    return cudaSuccess;
}

extern "C" size_t solution_workspace_bytes(const BacProblemSpec* spec) {
    if (!bac_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(const BacProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!bac_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    BacReferenceState* st = (BacReferenceState*)malloc(sizeof(BacReferenceState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(*st));
    memcpy(&st->spec, spec, sizeof(BacProblemSpec));
    st->max_order = spec->max_order;
    st->segment_order = spec->segment_order;
    st->total_pages = (uint64_t)1 << spec->max_order;
    st->segment_pages = (uint64_t)1 << spec->segment_order;
    st->num_units = st->total_pages / st->segment_pages;
    // free list capacity: number of distinct buddy blocks <= num_units, plus
    // headroom for transient split states (each alloc adds <= max_order entries).
    uint64_t fcap = st->num_units + (uint64_t)st->max_order + 4;
    if (fcap > 4194304ULL) fcap = 4194304ULL;
    st->free_cap = (int)fcap;
    st->seg_cap = spec->max_segments;
    st->obj_cap = spec->max_objects;

    cudaError_t err = cudaSuccess;
#define ALLOC(p, bytes) do { err = cudaMalloc((void**)&(p), (bytes)); if (err) goto fail; } while(0)
    ALLOC(st->d_free_order, sizeof(uint8_t) * st->free_cap);
    ALLOC(st->d_free_base, sizeof(uint64_t) * st->free_cap);
    ALLOC(st->d_free_count, sizeof(int32_t));
    ALLOC(st->d_seg_alive, sizeof(uint8_t) * st->seg_cap);
    ALLOC(st->d_seg_id, sizeof(uint64_t) * st->seg_cap);
    ALLOC(st->d_seg_class, sizeof(uint32_t) * st->seg_cap);
    ALLOC(st->d_seg_base, sizeof(uint64_t) * st->seg_cap);
    ALLOC(st->d_seg_aoff, sizeof(uint64_t) * st->seg_cap);
    ALLOC(st->d_seg_live, sizeof(uint64_t) * st->seg_cap);
    ALLOC(st->d_seg_dead, sizeof(uint64_t) * st->seg_cap);
    ALLOC(st->d_seg_sealed, sizeof(uint8_t) * st->seg_cap);
    ALLOC(st->d_obj_alive, sizeof(uint8_t) * st->obj_cap);
    ALLOC(st->d_obj_id, sizeof(uint64_t) * st->obj_cap);
    ALLOC(st->d_obj_class, sizeof(uint32_t) * st->obj_cap);
    ALLOC(st->d_obj_seg, sizeof(uint64_t) * st->obj_cap);
    ALLOC(st->d_obj_base, sizeof(uint64_t) * st->obj_cap);
    ALLOC(st->d_obj_order, sizeof(uint8_t) * st->obj_cap);
    ALLOC(st->d_obj_block, sizeof(uint64_t) * st->obj_cap);
    ALLOC(st->d_obj_req, sizeof(uint64_t) * st->obj_cap);
    ALLOC(st->d_obj_pin, sizeof(uint64_t) * st->obj_cap);
    ALLOC(st->d_obj_pending, sizeof(uint8_t) * st->obj_cap);
    ALLOC(st->d_obj_aseq, sizeof(uint64_t) * st->obj_cap);
    ALLOC(st->d_obj_mseq, sizeof(uint64_t) * st->obj_cap);
    ALLOC(st->d_active_seg, sizeof(uint64_t) * spec->num_classes);
    ALLOC(st->d_next_seg_id, sizeof(uint64_t));
    ALLOC(st->d_event_seq, sizeof(uint64_t));
    ALLOC(st->d_op_index, sizeof(uint32_t));
    ALLOC(st->d_counters, sizeof(uint64_t) * 16);
    ALLOC(st->d_hashes, sizeof(uint64_t) * 5);
    ALLOC(st->d_scratch, sizeof(int32_t) * st->obj_cap);
    ALLOC(st->d_ord_obj, sizeof(int32_t) * st->obj_cap);
    ALLOC(st->d_ord_seg, sizeof(int32_t) * st->seg_cap);
    ALLOC(st->d_ord_free, sizeof(int32_t) * st->free_cap);
    {
        size_t ops_bytes = sizeof(BacOp) * (size_t)(spec->max_ops_per_step > 0 ? spec->max_ops_per_step : 1);
        ALLOC(st->d_ops, ops_bytes);
    }
#undef ALLOC

    err = ref_reset_state(st, stream);
    if (err) goto fail;
    *state_out = st;
    return cudaSuccess;

fail:
    solution_destroy(st);
    return err;
}

extern "C" cudaError_t solution_run(void* state, const BacRunSpec* run, const void* inputs_void,
                                    void* outputs_void, void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    (void)workspace;
    if (!state || !inputs_void || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 1) return cudaErrorInvalidValue;
    BacReferenceState* st = (BacReferenceState*)state;
    if (!bac_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;
    const BacInputs* in = (const BacInputs*)inputs_void;
    BacOutputs* out = (BacOutputs*)outputs_void;
    if (run->num_ops > 0 && !in->ops) return cudaErrorInvalidValue;

    cudaError_t err;
    if (run->num_ops > 0) {
        err = cudaMemcpyAsync(st->d_ops, in->ops, sizeof(BacOp) * (size_t)run->num_ops,
                              cudaMemcpyDeviceToDevice, stream);
        if (err) return err;
    }

    // outputs are 16 separate counter scalars; gather into a temp via kernel
    // writing directly. We pass an output counter buffer (use st->d_counters'
    // mirror?). Simpler: write counters to a device staging then scatter. We
    // instead write each output pointer from the kernel via a small staging
    // array, then a copy kernel. To keep it simple, allocate staging is not
    // allowed in run. So we write to the 16 user counter pointers directly by
    // packing them — but they're separate device pointers. Use a scatter.
    // Approach: kernel writes into st->d_counters (already updated) AND we then
    // copy each into the user-provided scalar via 16 DtoD copies.

    ref_step_kernel<<<1,256,0,stream>>>(
        st->max_order, st->segment_order, st->spec.num_classes, st->spec.max_objects, st->spec.max_segments,
        st->total_pages, st->segment_pages,
        st->d_free_order, st->d_free_base, st->d_free_count,
        st->d_seg_alive, st->d_seg_id, st->d_seg_class, st->d_seg_base,
        st->d_seg_aoff, st->d_seg_live, st->d_seg_dead, st->d_seg_sealed, st->seg_cap,
        st->d_obj_alive, st->d_obj_id, st->d_obj_class, st->d_obj_seg,
        st->d_obj_base, st->d_obj_order, st->d_obj_block, st->d_obj_req,
        st->d_obj_pin, st->d_obj_pending, st->d_obj_aseq, st->d_obj_mseq, st->obj_cap,
        st->d_active_seg, st->d_next_seg_id, st->d_event_seq, st->d_op_index,
        st->d_counters, st->d_hashes, st->d_scratch,
        st->d_ord_obj, st->d_ord_seg, st->d_ord_free,
        st->d_ops, run->num_ops,
        st->d_counters /*reuse: o_counters*/, out->alloc_event_hash, out->finalize_hash,
        out->buddy_hash, out->segment_hash, out->object_hash,
        out->live_object_count, out->live_segment_count);
    err = cudaPeekAtLastError();
    if (err) return err;

    // scatter the 16 counters from st->d_counters to user pointers
    uint64_t* cdst[16] = {
        out->alloc_ok, out->alloc_oom, out->free_finalized, out->free_deferred,
        out->pin_ok, out->unpin_ok, out->seal_explicit, out->seal_implicit,
        out->seal_empty, out->relocated_objects, out->clean_blocked_segments,
        out->segments_reclaimed, out->buddy_splits, out->buddy_merges,
        out->padding_pages_added, out->invalid_count
    };
    for (int i = 0; i < 16; ++i) {
        err = cudaMemcpyAsync(cdst[i], st->d_counters + i, sizeof(uint64_t),
                              cudaMemcpyDeviceToDevice, stream);
        if (err) return err;
    }
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return ref_reset_state((BacReferenceState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    BacReferenceState* st = (BacReferenceState*)state;
#define FREE(p) do { if (p) cudaFree(p); } while(0)
    FREE(st->d_free_order); FREE(st->d_free_base); FREE(st->d_free_count);
    FREE(st->d_seg_alive); FREE(st->d_seg_id); FREE(st->d_seg_class); FREE(st->d_seg_base);
    FREE(st->d_seg_aoff); FREE(st->d_seg_live); FREE(st->d_seg_dead); FREE(st->d_seg_sealed);
    FREE(st->d_obj_alive); FREE(st->d_obj_id); FREE(st->d_obj_class); FREE(st->d_obj_seg);
    FREE(st->d_obj_base); FREE(st->d_obj_order); FREE(st->d_obj_block); FREE(st->d_obj_req);
    FREE(st->d_obj_pin); FREE(st->d_obj_pending); FREE(st->d_obj_aseq); FREE(st->d_obj_mseq);
    FREE(st->d_active_seg); FREE(st->d_next_seg_id); FREE(st->d_event_seq); FREE(st->d_op_index);
    FREE(st->d_counters); FREE(st->d_hashes); FREE(st->d_scratch);
    FREE(st->d_ord_obj); FREE(st->d_ord_seg); FREE(st->d_ord_free); FREE(st->d_ops);
#undef FREE
    free(st);
}
