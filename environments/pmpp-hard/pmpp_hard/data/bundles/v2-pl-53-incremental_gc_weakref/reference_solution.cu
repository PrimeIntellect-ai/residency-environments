// PMPP_CANARY_53_3d32e38c4b -- held-out canary; MUST NOT appear in any submission
// file: incremental_gc_weakref_reference.cu
//
// Reference implementation of T53. The full persistent GC state lives in
// device memory; a single-thread device kernel executes exactly one
// operation per solution_run and writes the count vector + six checksums.
//
// This is an independent implementation (raw device arrays, manual FIFOs).

#include "incremental_gc_weakref_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define IGCW_FNV_INIT 1469598103934665603ULL

// ----- device FNV -----
__device__ __forceinline__ uint64_t rfnv_b(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= 1099511628211ULL; return h;
}
__device__ void rfnv(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p; uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = rfnv_b(v, b[i]); *h = v;
}
__device__ __forceinline__ void rfnv_u8 (uint64_t* h, uint8_t v){ rfnv(h,&v,1); }
__device__ __forceinline__ void rfnv_u32(uint64_t* h, uint32_t v){ rfnv(h,&v,4); }
__device__ __forceinline__ void rfnv_u64(uint64_t* h, uint64_t v){ rfnv(h,&v,8); }

// ----- device state struct (pointers into one big arena) -----
struct IgcwDevState {
    int max_objects, root_count, strong_slots, weak_slots;
    int max_ephemerons, max_mark_queue, max_finalizer_queue, young_survive_threshold;

    uint64_t event_seq, obj_id_next, alloc_seq_next, gc_cycle_id;

    uint8_t*  present;
    uint64_t* o_size;
    uint8_t*  o_gen;
    uint8_t*  o_age;
    uint8_t*  o_color;
    uint8_t*  o_adgc;
    uint64_t* o_alloc_seq;
    uint32_t* o_fin_tag;
    uint32_t* o_fin_slot;
    uint8_t*  o_finalized;
    uint8_t*  o_in_fq;
    int32_t*  o_strong;
    int32_t*  o_weak;

    int32_t*  root;

    uint8_t*  eph_present;
    int32_t*  eph_owner;
    int32_t*  eph_key;
    int32_t*  eph_value;
    uint64_t* eph_cseq;

    int32_t*  rem_src;   // remembered set arrays (kept sorted), length rem_len
    int32_t*  rem_slot;
    int32_t   rem_len;

    int32_t*  free_ids;  // ascending, length free_len
    int32_t   free_len;

    int32_t*  fin_queue; // FIFO, fin_head..fin_tail
    int32_t   fin_head, fin_tail; // ring not needed; use compaction style: head index, count
    int32_t   fin_count;

    int       phase, mode;
    int32_t*  mark_queue;
    int32_t   mq_head, mq_count;
    uint64_t  scan_cursor_obj;
    uint32_t  scan_cursor_field;
    uint64_t  sweep_cursor_obj;
    uint8_t   ephemeron_changed;
    uint8_t*  in_cset;

    uint64_t  event_hash;
    int32_t   op_invalid;
    int32_t*  counts; // IGCW_NUM_COUNTS
    uint32_t  cur_op_index;
};

// Host mirror to hold device pointers for malloc/free.
struct IgcwRefHost {
    IgcwProblemSpec spec;
    IgcwDevState* d_state;     // device copy of the struct
    IgcwDevState  h_state;     // host shadow holding device pointers
    void* arena;               // single device allocation backing all arrays
};

// ---- device helpers operating on st ----
__device__ bool valid_obj(IgcwDevState* st, int32_t id) {
    return id >= 1 && id <= st->max_objects && st->present[id-1] != 0;
}
__device__ bool in_cset_fn(IgcwDevState* st, int32_t id) {
    return id >= 1 && id <= st->max_objects && st->present[id-1] && st->in_cset[id-1];
}
__device__ bool gc_active(IgcwDevState* st) { return st->phase != IGCW_PHASE_IDLE; }
__device__ int32_t* sp(IgcwDevState* st, int32_t id){ return st->o_strong + (size_t)(id-1)*st->strong_slots; }
__device__ int32_t* wp(IgcwDevState* st, int32_t id){ return st->o_weak   + (size_t)(id-1)*st->weak_slots; }

__device__ void emit(IgcwDevState* st, int kind, uint64_t obj, uint32_t slot,
                      uint32_t rootid, uint64_t eph, uint64_t aux) {
    uint64_t* h = &st->event_hash;
    rfnv_u8 (h,(uint8_t)kind);
    rfnv_u64(h, st->event_seq);
    rfnv_u32(h, st->cur_op_index);
    rfnv_u64(h, st->gc_cycle_id);
    rfnv_u64(h, obj);
    rfnv_u32(h, slot);
    rfnv_u32(h, rootid);
    rfnv_u64(h, eph);
    rfnv_u64(h, aux);
    st->event_seq += 1;
}
__device__ void mark_invalid(IgcwDevState* st) {
    st->op_invalid = 1;
    emit(st, IGCW_EV_INVALID, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
}

__device__ void mq_push(IgcwDevState* st, int32_t id) {
    st->mark_queue[(st->mq_head + st->mq_count) % st->max_mark_queue] = id;
    st->mq_count += 1;
}
__device__ int32_t mq_pop(IgcwDevState* st) {
    int32_t v = st->mark_queue[st->mq_head];
    st->mq_head = (st->mq_head + 1) % st->max_mark_queue;
    st->mq_count -= 1;
    return v;
}

// remembered helpers (kept sorted by (src,slot))
__device__ bool rem_contains(IgcwDevState* st, int32_t src, int32_t slot) {
    for (int i = 0; i < st->rem_len; ++i)
        if (st->rem_src[i] == src && st->rem_slot[i] == slot) return true;
    return false;
}
__device__ void rem_insert_sorted(IgcwDevState* st, int32_t src, int32_t slot) {
    int pos = st->rem_len;
    for (int i = 0; i < st->rem_len; ++i) {
        if (src < st->rem_src[i] || (src == st->rem_src[i] && slot < st->rem_slot[i])) { pos = i; break; }
    }
    for (int i = st->rem_len; i > pos; --i) { st->rem_src[i]=st->rem_src[i-1]; st->rem_slot[i]=st->rem_slot[i-1]; }
    st->rem_src[pos]=src; st->rem_slot[pos]=slot; st->rem_len += 1;
}
__device__ void rem_remove_at(IgcwDevState* st, int idx) {
    for (int i = idx; i < st->rem_len-1; ++i){ st->rem_src[i]=st->rem_src[i+1]; st->rem_slot[i]=st->rem_slot[i+1]; }
    st->rem_len -= 1;
}

__device__ void free_insert_sorted(IgcwDevState* st, int32_t id) {
    int pos = st->free_len;
    for (int i = 0; i < st->free_len; ++i) if (id < st->free_ids[i]) { pos = i; break; }
    for (int i = st->free_len; i > pos; --i) st->free_ids[i]=st->free_ids[i-1];
    st->free_ids[pos]=id; st->free_len += 1;
}

__device__ int eph_used(IgcwDevState* st) {
    int u = 0; for (int i = 0; i < st->max_ephemerons; ++i) if (st->eph_present[i]) ++u; return u;
}

// ===== ops =====
__device__ void do_alloc(IgcwDevState* st, const IgcwRunSpec& r) {
    int gen = r.a0; uint32_t tag = (uint32_t)r.a1; uint32_t slot = (uint32_t)r.a2;
    uint64_t size = (uint64_t)r.size_arg;
    if (gen != IGCW_GEN_YOUNG && gen != IGCW_GEN_OLD) { mark_invalid(st); return; }
    if (tag == 0) slot = IGCW_U32_MAX;
    else if (slot != IGCW_U32_MAX && slot >= (uint32_t)st->root_count) { mark_invalid(st); return; }

    int32_t id;
    if (st->free_len > 0) {
        id = st->free_ids[0];
        for (int i = 0; i < st->free_len-1; ++i) st->free_ids[i]=st->free_ids[i+1];
        st->free_len -= 1;
    } else {
        if ((int64_t)st->obj_id_next > st->max_objects) {
            emit(st, IGCW_EV_ALLOC_OOM, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
            st->counts[2]++; return;
        }
        id = (int32_t)st->obj_id_next; st->obj_id_next += 1;
    }
    int k = id-1;
    st->present[k]=1; st->o_size[k]=size; st->o_gen[k]=(uint8_t)gen; st->o_age[k]=0;
    st->o_alloc_seq[k]=st->alloc_seq_next; st->alloc_seq_next += 1;
    st->o_fin_tag[k]=tag; st->o_fin_slot[k]=slot; st->o_finalized[k]=0; st->o_in_fq[k]=0;
    for (int s=0;s<st->strong_slots;++s) sp(st,id)[s]=0;
    for (int s=0;s<st->weak_slots;++s) wp(st,id)[s]=0;

    if (!gc_active(st)) {
        st->o_color[k]=IGCW_WHITE; st->o_adgc[k]=0; st->in_cset[k]=0;
        emit(st, IGCW_EV_ALLOC_OK, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, size);
        st->counts[0]++;
    } else {
        st->in_cset[k]=1; st->o_color[k]=IGCW_BLACK; st->o_adgc[k]=1;
        emit(st, IGCW_EV_ALLOC_BLACK, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, size);
        st->counts[1]++;
        emit(st, IGCW_EV_ALLOC_OK, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, size);
        st->counts[0]++;
    }
}

__device__ void do_set_root(IgcwDevState* st, const IgcwRunSpec& r) {
    int32_t rid = r.a0, obj = r.a1;
    if (rid < 0 || rid >= st->root_count) { mark_invalid(st); return; }
    if (obj != 0 && !valid_obj(st,obj)) { mark_invalid(st); return; }
    st->root[rid] = obj;
    if (gc_active(st) && obj != 0 && in_cset_fn(st,obj) && st->o_color[obj-1]==IGCW_WHITE) {
        st->o_color[obj-1]=IGCW_GREY; mq_push(st,obj);
        emit(st, IGCW_EV_BARRIER_MARK, (uint64_t)obj, IGCW_U32_MAX, (uint32_t)rid, IGCW_U64_MAX, 0);
        st->counts[30]++;
    }
    emit(st, IGCW_EV_ROOT_SET, (uint64_t)obj, IGCW_U32_MAX, (uint32_t)rid, IGCW_U64_MAX, 0);
    st->counts[3]++;
}

__device__ void do_clear_root(IgcwDevState* st, const IgcwRunSpec& r) {
    int32_t rid = r.a0;
    if (rid < 0 || rid >= st->root_count) { mark_invalid(st); return; }
    st->root[rid]=0;
    emit(st, IGCW_EV_ROOT_CLEAR, 0, IGCW_U32_MAX, (uint32_t)rid, IGCW_U64_MAX, 0);
    st->counts[4]++;
}

__device__ void do_set_strong(IgcwDevState* st, const IgcwRunSpec& r) {
    int32_t src=r.a0, slot=r.a1, dst=r.a2;
    if (!valid_obj(st,src)) { mark_invalid(st); return; }
    if (slot < 0 || slot >= st->strong_slots) { mark_invalid(st); return; }
    if (dst != 0 && !valid_obj(st,dst)) { mark_invalid(st); return; }
    sp(st,src)[slot]=dst;
    if (dst != 0 && st->o_gen[src-1]==IGCW_GEN_OLD && st->o_gen[dst-1]==IGCW_GEN_YOUNG) {
        if (!rem_contains(st,src,slot)) {
            rem_insert_sorted(st,src,slot);
            emit(st, IGCW_EV_REMEMBERED_ADD, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, 0);
            st->counts[11]++;
        }
    }
    if (gc_active(st) && st->o_color[src-1]==IGCW_BLACK && dst != 0 &&
        in_cset_fn(st,dst) && st->o_color[dst-1]==IGCW_WHITE) {
        st->o_color[dst-1]=IGCW_GREY; mq_push(st,dst);
        emit(st, IGCW_EV_BARRIER_MARK, (uint64_t)dst, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, 0);
        st->counts[30]++;
    }
    emit(st, IGCW_EV_STRONG_SET, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, (uint64_t)(uint32_t)dst);
    st->counts[5]++;
}

__device__ void do_clear_strong(IgcwDevState* st, const IgcwRunSpec& r) {
    int32_t src=r.a0, slot=r.a1;
    if (!valid_obj(st,src)) { mark_invalid(st); return; }
    if (slot < 0 || slot >= st->strong_slots) { mark_invalid(st); return; }
    sp(st,src)[slot]=0;
    emit(st, IGCW_EV_STRONG_CLEAR, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, 0);
    st->counts[6]++;
}

__device__ void do_set_weak(IgcwDevState* st, const IgcwRunSpec& r) {
    int32_t src=r.a0, slot=r.a1, dst=r.a2;
    if (!valid_obj(st,src)) { mark_invalid(st); return; }
    if (slot < 0 || slot >= st->weak_slots) { mark_invalid(st); return; }
    if (dst != 0 && !valid_obj(st,dst)) { mark_invalid(st); return; }
    wp(st,src)[slot]=dst;
    emit(st, IGCW_EV_WEAK_SET, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, (uint64_t)(uint32_t)dst);
    st->counts[7]++;
}

__device__ void do_set_ephemeron(IgcwDevState* st, const IgcwRunSpec& r) {
    uint64_t eid = (uint64_t)(uint32_t)r.a0;
    int32_t owner=r.a1, key=r.a2, value=r.a3;
    if (!valid_obj(st,owner)) { mark_invalid(st); return; }
    if (key != 0 && !valid_obj(st,key)) { mark_invalid(st); return; }
    if (value != 0 && !valid_obj(st,value)) { mark_invalid(st); return; }
    if ((int64_t)eid < 0 || (int64_t)eid >= st->max_ephemerons) { mark_invalid(st); return; }
    bool exists = st->eph_present[eid] != 0;
    if (exists && st->eph_owner[eid] != owner) { mark_invalid(st); return; }
    if (!exists) {
        if (eph_used(st) >= st->max_ephemerons) {
            emit(st, IGCW_EV_EPHEMERON_OOM, 0, IGCW_U32_MAX, IGCW_U32_MAX, eid, 0);
            st->counts[10]++; return;
        }
        st->eph_present[eid]=1;
        st->eph_cseq[eid]=st->alloc_seq_next; st->alloc_seq_next += 1;
    }
    st->eph_owner[eid]=owner; st->eph_key[eid]=key; st->eph_value[eid]=value;
    if (gc_active(st) && key != 0 && st->o_color[key-1]==IGCW_BLACK &&
        value != 0 && in_cset_fn(st,value) && st->o_color[value-1]==IGCW_WHITE) {
        st->o_color[value-1]=IGCW_GREY; mq_push(st,value);
        emit(st, IGCW_EV_BARRIER_MARK, (uint64_t)value, IGCW_U32_MAX, IGCW_U32_MAX, eid, 0);
        st->counts[30]++;
    }
    emit(st, IGCW_EV_EPHEMERON_SET, (uint64_t)owner, IGCW_U32_MAX, IGCW_U32_MAX, eid, (uint64_t)(uint32_t)value);
    st->counts[8]++;
}

__device__ void do_delete_ephemeron(IgcwDevState* st, const IgcwRunSpec& r) {
    uint64_t eid = (uint64_t)(uint32_t)r.a0;
    if ((int64_t)eid < 0 || (int64_t)eid >= st->max_ephemerons || !st->eph_present[eid]) { mark_invalid(st); return; }
    st->eph_present[eid]=0; st->eph_owner[eid]=0; st->eph_key[eid]=0; st->eph_value[eid]=0; st->eph_cseq[eid]=0;
    emit(st, IGCW_EV_EPHEMERON_DELETE, 0, IGCW_U32_MAX, IGCW_U32_MAX, eid, 0);
    st->counts[9]++;
}

__device__ void do_start_minor(IgcwDevState* st) {
    if (st->phase != IGCW_PHASE_IDLE) { mark_invalid(st); return; }
    st->gc_cycle_id += 1; st->mode=IGCW_MODE_MINOR; st->phase=IGCW_PHASE_MARK;
    for (int i = 0; i < st->max_objects; ++i) {
        if (st->present[i] && st->o_gen[i]==IGCW_GEN_YOUNG) {
            st->in_cset[i]=1; st->o_color[i]=IGCW_WHITE; st->o_adgc[i]=0;
        } else st->in_cset[i]=0;
    }
    st->mq_head=0; st->mq_count=0;
    st->scan_cursor_obj=0; st->scan_cursor_field=0; st->sweep_cursor_obj=0; st->ephemeron_changed=0;
    for (int rid = 0; rid < st->root_count; ++rid) {
        int32_t obj = st->root[rid];
        if (obj != 0 && in_cset_fn(st,obj) && st->o_color[obj-1]==IGCW_WHITE) {
            st->o_color[obj-1]=IGCW_GREY; mq_push(st,obj);
        }
    }
    // remembered scan canonical order
    int i = 0;
    while (i < st->rem_len) {
        int32_t src=st->rem_src[i], slot=st->rem_slot[i];
        bool drop=false;
        if (!valid_obj(st,src)) drop=true;
        else {
            int32_t tgt = sp(st,src)[slot];
            if (tgt != 0 && in_cset_fn(st,tgt)) {
                if (st->o_color[tgt-1]==IGCW_WHITE){ st->o_color[tgt-1]=IGCW_GREY; mq_push(st,tgt); }
            } else drop=true;
        }
        if (drop) {
            emit(st, IGCW_EV_REMEMBERED_DROP, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, 0);
            st->counts[12]++;
            rem_remove_at(st, i);
        } else ++i;
    }
    emit(st, IGCW_EV_GC_START_MINOR, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, st->gc_cycle_id);
    st->counts[13]++;
}

__device__ void do_start_full(IgcwDevState* st) {
    if (st->phase != IGCW_PHASE_IDLE) { mark_invalid(st); return; }
    st->gc_cycle_id += 1; st->mode=IGCW_MODE_FULL; st->phase=IGCW_PHASE_MARK;
    for (int i = 0; i < st->max_objects; ++i) {
        if (st->present[i]) { st->in_cset[i]=1; st->o_color[i]=IGCW_WHITE; st->o_adgc[i]=0; }
        else st->in_cset[i]=0;
    }
    st->mq_head=0; st->mq_count=0;
    st->scan_cursor_obj=0; st->scan_cursor_field=0; st->sweep_cursor_obj=0; st->ephemeron_changed=0;
    for (int rid = 0; rid < st->root_count; ++rid) {
        int32_t obj = st->root[rid];
        if (obj != 0 && in_cset_fn(st,obj) && st->o_color[obj-1]==IGCW_WHITE) {
            st->o_color[obj-1]=IGCW_GREY; mq_push(st,obj);
        }
    }
    emit(st, IGCW_EV_GC_START_FULL, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, st->gc_cycle_id);
    st->counts[14]++;
}

__device__ void step_mark(IgcwDevState* st, int budget) {
    int processed = 0;
    while (processed < budget && st->mq_count > 0) {
        int32_t id = mq_pop(st); ++processed;
        if (!valid_obj(st,id) || !in_cset_fn(st,id)) continue;
        st->o_color[id-1]=IGCW_BLACK;
        emit(st, IGCW_EV_MARK_BLACK, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
        st->counts[15]++;
        for (int s = 0; s < st->strong_slots; ++s) {
            int32_t tgt = sp(st,id)[s];
            if (tgt != 0 && in_cset_fn(st,tgt) && st->o_color[tgt-1]==IGCW_WHITE) {
                st->o_color[tgt-1]=IGCW_GREY; mq_push(st,tgt);
                emit(st, IGCW_EV_MARK_GREY, (uint64_t)tgt, (uint32_t)s, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                st->counts[16]++;
            }
        }
    }
    if (st->mq_count == 0) {
        st->phase=IGCW_PHASE_EPHEMERON;
        emit(st, IGCW_EV_PHASE_EPHEMERON, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
    }
}

__device__ void step_ephemeron(IgcwDevState* st) {
    st->ephemeron_changed = 0;
    for (int i = 0; i < st->max_ephemerons; ++i) {
        if (!st->eph_present[i]) continue;
        int32_t key=st->eph_key[i], val=st->eph_value[i];
        bool key_reach=false;
        if (key != 0 && valid_obj(st,key)) {
            if (in_cset_fn(st,key)) key_reach = (st->o_color[key-1]==IGCW_BLACK);
            else key_reach = true;
        }
        if (key_reach && val != 0 && in_cset_fn(st,val) && st->o_color[val-1]==IGCW_WHITE) {
            st->o_color[val-1]=IGCW_GREY; mq_push(st,val); st->ephemeron_changed=1;
            emit(st, IGCW_EV_EPHEMERON_MARK, (uint64_t)val, IGCW_U32_MAX, IGCW_U32_MAX, (uint64_t)i, 0);
            st->counts[17]++;
        }
    }
    if (st->ephemeron_changed == 1) {
        st->phase=IGCW_PHASE_MARK;
        emit(st, IGCW_EV_PHASE_MARK, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
    } else {
        st->phase=IGCW_PHASE_FINALIZE_SCAN; st->scan_cursor_obj=1;
        emit(st, IGCW_EV_PHASE_FINALIZE_SCAN, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
    }
}

__device__ void step_finalize_scan(IgcwDevState* st) {
    for (int i = 0; i < st->max_objects; ++i) {
        int32_t id = i+1;
        if (!in_cset_fn(st,id)) continue;
        if (st->o_color[i]==IGCW_WHITE && st->o_fin_tag[i]!=0 && st->o_finalized[i]==0 && st->o_in_fq[i]==0) {
            if (st->fin_count >= st->max_finalizer_queue) {
                emit(st, IGCW_EV_FINALIZER_QUEUE_FULL, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                st->counts[19]++;
            } else {
                st->o_finalized[i]=1; st->o_in_fq[i]=1;
                st->fin_queue[(st->fin_head + st->fin_count) % st->max_finalizer_queue] = id;
                st->fin_count += 1;
                st->o_color[i]=IGCW_GREY; mq_push(st,id);
                emit(st, IGCW_EV_FINALIZER_ENQUEUE, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                st->counts[18]++;
            }
        }
    }
    st->scan_cursor_obj=0;
    if (st->mq_count > 0) {
        st->phase=IGCW_PHASE_RESURRECT_MARK;
        emit(st, IGCW_EV_PHASE_RESURRECT_MARK, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
    } else {
        st->phase=IGCW_PHASE_WEAK_CLEAR;
        emit(st, IGCW_EV_PHASE_WEAK_CLEAR, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
    }
}

__device__ void step_weak_clear(IgcwDevState* st) {
    for (int i = 0; i < st->max_objects; ++i) {
        int32_t id=i+1;
        if (!st->present[i]) continue;
        for (int s = 0; s < st->weak_slots; ++s) {
            int32_t tgt = wp(st,id)[s];
            if (tgt != 0 && in_cset_fn(st,tgt) && st->o_color[tgt-1]==IGCW_WHITE) {
                wp(st,id)[s]=0;
                emit(st, IGCW_EV_WEAK_CLEAR_FIELD, (uint64_t)id, (uint32_t)s, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                st->counts[20]++;
            }
        }
    }
    for (int i = 0; i < st->max_ephemerons; ++i) {
        if (!st->eph_present[i]) continue;
        int32_t key=st->eph_key[i];
        if (key != 0 && in_cset_fn(st,key) && st->o_color[key-1]==IGCW_WHITE) {
            st->eph_key[i]=0;
            emit(st, IGCW_EV_EPHEMERON_CLEAR_SIDE, 0, 0, IGCW_U32_MAX, (uint64_t)i, 0);
            st->counts[21]++;
        }
        int32_t val=st->eph_value[i];
        if (val != 0 && in_cset_fn(st,val) && st->o_color[val-1]==IGCW_WHITE) {
            st->eph_value[i]=0;
            emit(st, IGCW_EV_EPHEMERON_CLEAR_SIDE, 0, 1, IGCW_U32_MAX, (uint64_t)i, 1);
            st->counts[21]++;
        }
    }
    st->phase=IGCW_PHASE_SWEEP; st->sweep_cursor_obj=1;
    emit(st, IGCW_EV_PHASE_SWEEP, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
}

__device__ void step_sweep(IgcwDevState* st, int budget) {
    int processed = 0;
    int64_t cur = (int64_t)st->sweep_cursor_obj; if (cur < 1) cur = 1;
    for (; cur <= st->max_objects && processed < budget; ++cur) {
        int32_t id=(int32_t)cur; int k=id-1;
        if (!st->present[k] || !st->in_cset[k]) continue;
        ++processed;
        if (st->o_color[k]==IGCW_WHITE) {
            for (int e = 0; e < st->max_ephemerons; ++e) {
                if (st->eph_present[e] && st->eph_owner[e]==id) {
                    st->eph_present[e]=0; st->eph_owner[e]=0; st->eph_key[e]=0; st->eph_value[e]=0; st->eph_cseq[e]=0;
                    emit(st, IGCW_EV_EPHEMERON_OWNER_FREE, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, (uint64_t)e, 0);
                    st->counts[25]++;
                }
            }
            int j = 0;
            while (j < st->rem_len) {
                if (st->rem_src[j]==id) rem_remove_at(st, j);
                else ++j;
            }
            st->present[k]=0; st->in_cset[k]=0; st->o_size[k]=0; st->o_gen[k]=IGCW_GEN_YOUNG; st->o_age[k]=0;
            st->o_color[k]=IGCW_WHITE; st->o_adgc[k]=0; st->o_alloc_seq[k]=0;
            st->o_fin_tag[k]=0; st->o_fin_slot[k]=IGCW_U32_MAX; st->o_finalized[k]=0; st->o_in_fq[k]=0;
            for (int s=0;s<st->strong_slots;++s) sp(st,id)[s]=0;
            for (int s=0;s<st->weak_slots;++s) wp(st,id)[s]=0;
            free_insert_sorted(st, id);
            emit(st, IGCW_EV_SWEEP_FREE, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
            st->counts[22]++;
        } else {
            if (st->mode==IGCW_MODE_MINOR && st->o_gen[k]==IGCW_GEN_YOUNG) {
                st->o_age[k] = (uint8_t)(st->o_age[k]+1);
                if (st->o_age[k] >= st->young_survive_threshold) {
                    st->o_gen[k]=IGCW_GEN_OLD;
                    emit(st, IGCW_EV_PROMOTE_OLD, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                    st->counts[24]++;
                }
            }
            st->o_color[k]=IGCW_WHITE; st->o_adgc[k]=0;
            emit(st, IGCW_EV_SWEEP_KEEP, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
            st->counts[23]++;
        }
    }
    st->sweep_cursor_obj=(uint64_t)cur;
    bool done = true;
    for (int64_t j = cur; j <= st->max_objects; ++j) {
        int k=(int)j-1;
        if (st->present[k] && st->in_cset[k]) { done=false; break; }
    }
    if (done) {
        int i = 0;
        while (i < st->rem_len) {
            int32_t src=st->rem_src[i], slot=st->rem_slot[i];
            bool drop=false;
            if (!valid_obj(st,src)) drop=true;
            else {
                int32_t tgt = sp(st,src)[slot];
                if (!(st->o_gen[src-1]==IGCW_GEN_OLD && tgt != 0 && valid_obj(st,tgt) && st->o_gen[tgt-1]==IGCW_GEN_YOUNG))
                    drop=true;
            }
            if (drop) {
                emit(st, IGCW_EV_REMEMBERED_DROP, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                st->counts[12]++;
                rem_remove_at(st, i);
            } else ++i;
        }
        st->phase=IGCW_PHASE_IDLE; st->mode=IGCW_MODE_NONE;
        for (int q = 0; q < st->max_objects; ++q) st->in_cset[q]=0;
        st->mq_head=0; st->mq_count=0; st->sweep_cursor_obj=0;
        emit(st, IGCW_EV_GC_END, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, st->gc_cycle_id);
        st->counts[26]++;
    }
}

__device__ void do_gc_step(IgcwDevState* st, const IgcwRunSpec& r) {
    int mark_budget=r.a0, sweep_budget=r.a1;
    if (st->phase==IGCW_PHASE_IDLE) { mark_invalid(st); return; }
    if (mark_budget==0 && sweep_budget==0) { mark_invalid(st); return; }
    bool reached_sweep=false;
    switch (st->phase) {
        case IGCW_PHASE_MARK:
        case IGCW_PHASE_RESURRECT_MARK: step_mark(st, mark_budget); break;
        case IGCW_PHASE_EPHEMERON: step_ephemeron(st); break;
        case IGCW_PHASE_FINALIZE_SCAN: step_finalize_scan(st); break;
        case IGCW_PHASE_WEAK_CLEAR: step_weak_clear(st); reached_sweep=(st->phase==IGCW_PHASE_SWEEP); break;
        case IGCW_PHASE_SWEEP: step_sweep(st, sweep_budget); break;
        default: break;
    }
    if (reached_sweep) step_sweep(st, sweep_budget);
}

__device__ void do_run_finalizers(IgcwDevState* st, const IgcwRunSpec& r) {
    int limit=r.a0;
    if (limit==0) return;
    if (limit < 0) { mark_invalid(st); return; }
    int popped=0;
    while (popped < limit && st->fin_count > 0) {
        int32_t id = st->fin_queue[st->fin_head];
        st->fin_head = (st->fin_head + 1) % st->max_finalizer_queue;
        st->fin_count -= 1;
        ++popped;
        if (!valid_obj(st,id)) {
            emit(st, IGCW_EV_FINALIZER_SKIP_ABSENT, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
            st->counts[29]++;
            continue;
        }
        st->o_in_fq[id-1]=0;
        uint32_t slot = st->o_fin_slot[id-1];
        if (slot != IGCW_U32_MAX && slot < (uint32_t)st->root_count) {
            st->root[slot]=id;
            emit(st, IGCW_EV_FINALIZER_RESURRECT, (uint64_t)id, IGCW_U32_MAX, slot, IGCW_U64_MAX, 0);
            st->counts[28]++;
        } else {
            emit(st, IGCW_EV_FINALIZER_RUN, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
            st->counts[27]++;
        }
    }
}

// ----- hashes -----
__device__ uint64_t heap_hash(IgcwDevState* st) {
    uint64_t h=IGCW_FNV_INIT;
    for (int i = 0; i < st->max_objects; ++i) {
        if (!st->present[i]) continue;
        int32_t id=i+1;
        rfnv_u64(&h,(uint64_t)id);
        rfnv_u64(&h,st->o_size[i]);
        rfnv_u8(&h,st->o_gen[i]); rfnv_u8(&h,st->o_age[i]); rfnv_u8(&h,st->o_color[i]); rfnv_u8(&h,st->o_adgc[i]);
        rfnv_u64(&h,st->o_alloc_seq[i]);
        rfnv_u32(&h,st->o_fin_tag[i]); rfnv_u32(&h,st->o_fin_slot[i]);
        rfnv_u8(&h,st->o_finalized[i]); rfnv_u8(&h,st->o_in_fq[i]);
        for (int s=0;s<st->strong_slots;++s) rfnv_u64(&h,(uint64_t)(uint32_t)sp(st,id)[s]);
        for (int s=0;s<st->weak_slots;++s) rfnv_u64(&h,(uint64_t)(uint32_t)wp(st,id)[s]);
    }
    return h;
}
__device__ uint64_t root_hash(IgcwDevState* st) {
    uint64_t h=IGCW_FNV_INIT;
    for (int rid=0; rid<st->root_count; ++rid) { rfnv_u32(&h,(uint32_t)rid); rfnv_u64(&h,(uint64_t)(uint32_t)st->root[rid]); }
    return h;
}
__device__ uint64_t ephemeron_hash(IgcwDevState* st) {
    uint64_t h=IGCW_FNV_INIT;
    for (int i=0;i<st->max_ephemerons;++i) {
        if (!st->eph_present[i]) continue;
        rfnv_u64(&h,(uint64_t)i);
        rfnv_u64(&h,(uint64_t)(uint32_t)st->eph_owner[i]);
        rfnv_u64(&h,(uint64_t)(uint32_t)st->eph_key[i]);
        rfnv_u64(&h,(uint64_t)(uint32_t)st->eph_value[i]);
        rfnv_u64(&h,st->eph_cseq[i]);
    }
    return h;
}
__device__ uint64_t remembered_hash(IgcwDevState* st) {
    uint64_t h=IGCW_FNV_INIT;
    for (int i=0;i<st->rem_len;++i){ rfnv_u64(&h,(uint64_t)(uint32_t)st->rem_src[i]); rfnv_u32(&h,(uint32_t)st->rem_slot[i]); }
    return h;
}
__device__ uint64_t controller_hash(IgcwDevState* st) {
    uint64_t h=IGCW_FNV_INIT;
    rfnv_u8(&h,(uint8_t)st->phase); rfnv_u8(&h,(uint8_t)st->mode);
    rfnv_u64(&h,st->gc_cycle_id); rfnv_u64(&h,st->scan_cursor_obj);
    rfnv_u32(&h,st->scan_cursor_field); rfnv_u64(&h,st->sweep_cursor_obj);
    rfnv_u8(&h,st->ephemeron_changed);
    for (int i=0;i<st->mq_count;++i) rfnv_u64(&h,(uint64_t)(uint32_t)st->mark_queue[(st->mq_head+i)%st->max_mark_queue]);
    for (int i=0;i<st->fin_count;++i) rfnv_u64(&h,(uint64_t)(uint32_t)st->fin_queue[(st->fin_head+i)%st->max_finalizer_queue]);
    for (int i=0;i<st->free_len;++i) rfnv_u64(&h,(uint64_t)(uint32_t)st->free_ids[i]);
    return h;
}

__global__ void igcw_ref_kernel(IgcwDevState* st, IgcwRunSpec run,
                                int32_t* out_counts, uint64_t* out_eh, uint64_t* out_hh,
                                uint64_t* out_rh, uint64_t* out_eph, uint64_t* out_rem,
                                uint64_t* out_ch, int32_t* out_inv) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < IGCW_NUM_COUNTS; ++i) st->counts[i]=0;
    st->op_invalid=0;
    st->cur_op_index=(uint32_t)run.op_index;
    switch (run.opcode) {
        case IGCW_OP_ALLOC: do_alloc(st, run); break;
        case IGCW_OP_SET_ROOT: do_set_root(st, run); break;
        case IGCW_OP_CLEAR_ROOT: do_clear_root(st, run); break;
        case IGCW_OP_SET_STRONG: do_set_strong(st, run); break;
        case IGCW_OP_CLEAR_STRONG: do_clear_strong(st, run); break;
        case IGCW_OP_SET_WEAK: do_set_weak(st, run); break;
        case IGCW_OP_SET_EPHEMERON: do_set_ephemeron(st, run); break;
        case IGCW_OP_DELETE_EPHEMERON: do_delete_ephemeron(st, run); break;
        case IGCW_OP_START_MINOR: do_start_minor(st); break;
        case IGCW_OP_START_FULL: do_start_full(st); break;
        case IGCW_OP_GC_STEP: do_gc_step(st, run); break;
        case IGCW_OP_RUN_FINALIZERS: do_run_finalizers(st, run); break;
        default: mark_invalid(st); break;
    }
    for (int i = 0; i < IGCW_NUM_COUNTS; ++i) out_counts[i]=st->counts[i];
    *out_eh = st->event_hash;
    *out_hh = heap_hash(st);
    *out_rh = root_hash(st);
    *out_eph = ephemeron_hash(st);
    *out_rem = remembered_hash(st);
    *out_ch = controller_hash(st);
    *out_inv = st->op_invalid;
}

// ---- arena layout & allocation ----
static size_t align8(size_t x){ return (x+7)&~((size_t)7); }

extern "C" size_t solution_workspace_bytes(const IgcwProblemSpec* spec) {
    if (!igcw_validate_problem_spec(spec)) return 0;
    return 64;
}

static cudaError_t alloc_arena(IgcwRefHost* H) {
    const IgcwProblemSpec& s = H->spec;
    int MO = s.max_objects, RC = s.root_count, SS = s.strong_slots_per_object, WS = s.weak_slots_per_object;
    int ME = s.max_ephemerons, MQ = s.max_mark_queue, FQ = s.max_finalizer_queue;

    size_t off = 0;
    auto reserve = [&](size_t bytes)->size_t { size_t o = off; off = align8(off + bytes); return o; };

    size_t off_present = reserve((size_t)MO * 1);
    size_t off_size    = reserve((size_t)MO * 8);
    size_t off_gen     = reserve((size_t)MO * 1);
    size_t off_age     = reserve((size_t)MO * 1);
    size_t off_color   = reserve((size_t)MO * 1);
    size_t off_adgc    = reserve((size_t)MO * 1);
    size_t off_aseq    = reserve((size_t)MO * 8);
    size_t off_ftag    = reserve((size_t)MO * 4);
    size_t off_fslot   = reserve((size_t)MO * 4);
    size_t off_fin     = reserve((size_t)MO * 1);
    size_t off_infq    = reserve((size_t)MO * 1);
    size_t off_strong  = reserve((size_t)MO * (size_t)(SS>0?SS:1) * 4);
    size_t off_weak    = reserve((size_t)MO * (size_t)(WS>0?WS:1) * 4);
    size_t off_root    = reserve((size_t)(RC>0?RC:1) * 4);
    size_t off_ephp    = reserve((size_t)(ME>0?ME:1) * 1);
    size_t off_epho    = reserve((size_t)(ME>0?ME:1) * 4);
    size_t off_ephk    = reserve((size_t)(ME>0?ME:1) * 4);
    size_t off_ephv    = reserve((size_t)(ME>0?ME:1) * 4);
    size_t off_ephc    = reserve((size_t)(ME>0?ME:1) * 8);
    size_t off_remsrc  = reserve((size_t)(MO+1) * 4 * (size_t)(SS>0?SS:1)); // generous bound
    size_t off_remslot = reserve((size_t)(MO+1) * 4 * (size_t)(SS>0?SS:1));
    size_t off_free    = reserve((size_t)(MO+1) * 4);
    size_t off_finq    = reserve((size_t)(FQ>0?FQ:1) * 4);
    size_t off_markq   = reserve((size_t)MQ * 4);
    size_t off_cset    = reserve((size_t)MO * 1);
    size_t off_counts  = reserve((size_t)IGCW_NUM_COUNTS * 4);

    size_t total = off;
    void* arena = nullptr;
    cudaError_t err = cudaMalloc(&arena, total);
    if (err != cudaSuccess) return err;
    H->arena = arena;
    char* base = (char*)arena;

    IgcwDevState& h = H->h_state;
    memset(&h, 0, sizeof(h));
    h.max_objects=MO; h.root_count=RC; h.strong_slots=SS; h.weak_slots=WS;
    h.max_ephemerons=ME; h.max_mark_queue=MQ; h.max_finalizer_queue=FQ;
    h.young_survive_threshold=s.young_survive_threshold;
    h.present     = (uint8_t*)(base+off_present);
    h.o_size      = (uint64_t*)(base+off_size);
    h.o_gen       = (uint8_t*)(base+off_gen);
    h.o_age       = (uint8_t*)(base+off_age);
    h.o_color     = (uint8_t*)(base+off_color);
    h.o_adgc      = (uint8_t*)(base+off_adgc);
    h.o_alloc_seq = (uint64_t*)(base+off_aseq);
    h.o_fin_tag   = (uint32_t*)(base+off_ftag);
    h.o_fin_slot  = (uint32_t*)(base+off_fslot);
    h.o_finalized = (uint8_t*)(base+off_fin);
    h.o_in_fq     = (uint8_t*)(base+off_infq);
    h.o_strong    = (int32_t*)(base+off_strong);
    h.o_weak      = (int32_t*)(base+off_weak);
    h.root        = (int32_t*)(base+off_root);
    h.eph_present = (uint8_t*)(base+off_ephp);
    h.eph_owner   = (int32_t*)(base+off_epho);
    h.eph_key     = (int32_t*)(base+off_ephk);
    h.eph_value   = (int32_t*)(base+off_ephv);
    h.eph_cseq    = (uint64_t*)(base+off_ephc);
    h.rem_src     = (int32_t*)(base+off_remsrc);
    h.rem_slot    = (int32_t*)(base+off_remslot);
    h.free_ids    = (int32_t*)(base+off_free);
    h.fin_queue   = (int32_t*)(base+off_finq);
    h.mark_queue  = (int32_t*)(base+off_markq);
    h.in_cset     = (uint8_t*)(base+off_cset);
    h.counts      = (int32_t*)(base+off_counts);

    err = cudaMalloc((void**)&H->d_state, sizeof(IgcwDevState));
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

__global__ void igcw_ref_reset_kernel(IgcwDevState* st) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    st->event_seq=0; st->obj_id_next=1; st->alloc_seq_next=1; st->gc_cycle_id=0;
    int MO=st->max_objects, SS=st->strong_slots, WS=st->weak_slots, ME=st->max_ephemerons;
    for (int i=0;i<MO;++i){
        st->present[i]=0; st->o_size[i]=0; st->o_gen[i]=IGCW_GEN_YOUNG; st->o_age[i]=0;
        st->o_color[i]=IGCW_WHITE; st->o_adgc[i]=0; st->o_alloc_seq[i]=0;
        st->o_fin_tag[i]=0; st->o_fin_slot[i]=IGCW_U32_MAX; st->o_finalized[i]=0; st->o_in_fq[i]=0;
        st->in_cset[i]=0;
    }
    for (long i=0;i<(long)MO*SS;++i) st->o_strong[i]=0;
    for (long i=0;i<(long)MO*WS;++i) st->o_weak[i]=0;
    for (int i=0;i<st->root_count;++i) st->root[i]=0;
    for (int i=0;i<ME;++i){ st->eph_present[i]=0; st->eph_owner[i]=0; st->eph_key[i]=0; st->eph_value[i]=0; st->eph_cseq[i]=0; }
    st->rem_len=0; st->free_len=0;
    st->fin_head=0; st->fin_count=0;
    st->phase=IGCW_PHASE_IDLE; st->mode=IGCW_MODE_NONE;
    st->mq_head=0; st->mq_count=0;
    st->scan_cursor_obj=0; st->scan_cursor_field=0; st->sweep_cursor_obj=0; st->ephemeron_changed=0;
    st->event_hash=0; st->op_invalid=0;
}

extern "C" cudaError_t solution_init(const IgcwProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!igcw_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    IgcwRefHost* H = (IgcwRefHost*)malloc(sizeof(IgcwRefHost));
    if (!H) return cudaErrorMemoryAllocation;
    memset(H, 0, sizeof(IgcwRefHost));
    memcpy(&H->spec, spec, sizeof(IgcwProblemSpec));
    cudaError_t err = alloc_arena(H);
    if (err != cudaSuccess) { if (H->arena) cudaFree(H->arena); if (H->d_state) cudaFree(H->d_state); free(H); return err; }
    err = cudaMemcpyAsync(H->d_state, &H->h_state, sizeof(IgcwDevState), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    igcw_ref_reset_kernel<<<1,1,0,stream>>>(H->d_state);
    err = cudaPeekAtLastError(); if (err != cudaSuccess) return err;
    *state_out = H;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    IgcwRefHost* H = (IgcwRefHost*)state;
    igcw_ref_reset_kernel<<<1,1,0,stream>>>(H->d_state);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_run(void* state, const IgcwRunSpec* run, const void* inputs,
                                    void* outputs, void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    (void)inputs; (void)workspace;
    if (!state || !igcw_validate_run_spec(run) || !outputs) return cudaErrorInvalidValue;
    if (workspace_bytes < 64) return cudaErrorInvalidValue;
    IgcwRefHost* H = (IgcwRefHost*)state;
    IgcwOutputs* out = (IgcwOutputs*)outputs;
    if (!out->counts || !out->gc_event_hash || !out->heap_hash || !out->root_hash ||
        !out->ephemeron_hash || !out->remembered_hash || !out->gc_controller_hash || !out->invalid_flag)
        return cudaErrorInvalidValue;
    igcw_ref_kernel<<<1,1,0,stream>>>(H->d_state, *run,
        out->counts, out->gc_event_hash, out->heap_hash, out->root_hash,
        out->ephemeron_hash, out->remembered_hash, out->gc_controller_hash, out->invalid_flag);
    return cudaPeekAtLastError();
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    IgcwRefHost* H = (IgcwRefHost*)state;
    if (H->arena) cudaFree(H->arena);
    if (H->d_state) cudaFree(H->d_state);
    free(H);
}
