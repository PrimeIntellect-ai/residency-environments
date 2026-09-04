// PMPP_CANARY_61_0d96e8312d -- held-out canary; MUST NOT appear in any submission
// file: mk_chunked_dep_counters_reference.cu
//
// Reference implementation of MK2 (Chunked Dependency-Counter Runtime).
// Single-thread device kernel that keeps the persistent state in dense device
// arrays.  Cells are addressed directly by (edge * max_chunks + chunk); each
// cell owns a small fixed-capacity wait-queue array of waiter ids ordered by
// wait_seq.  The ready queue and pending-store queue are dense arrays processed
// from the front.  This data organization (per-cell wait-queue arrays, direct
// cell indexing) is deliberately different from naive.cu (flat waiter pool with
// on-the-fly filtering) and the host oracle (std::vector pools), but the three
// produce byte-identical graded outputs.

#include "mk_chunked_dep_counters_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MK_REF_CELLS (MK_MAX_EDGES * MK_MAX_CHUNKS_PER_EDGE)
// per-cell wait-queue capacity (waiters concurrently queued on one cell).
#define MK_REF_WQ_CAP MK_MAX_WAITERS

struct MkRefState {
    MkProblemSpec spec;
    uint64_t epoch_mod;

    // scalars: clock, event_seq, store_seq_next, wait_seq_next, ready_seq_next, epoch
    uint64_t* sc;
    uint64_t* op_index_global;  // 1
    uint64_t* event_hash;       // 1
    uint64_t* counts;           // MK_COUNT_TOTAL

    // edge table
    uint8_t*  e_defined;        // edge_count
    int32_t*  e_chunk_count;    // edge_count
    uint64_t* e_epoch;          // edge_count

    // cell table, indexed by edge*max_chunks + chunk
    uint64_t* c_counter;
    uint8_t*  c_page;
    uint64_t* c_payload;
    uint64_t* c_producer;
    uint64_t* c_store_seq;
    uint64_t* c_release_seq;
    int32_t*  c_wq;             // MK_REF_CELLS * MK_REF_WQ_CAP : waiter ids
    int32_t*  c_wq_len;         // MK_REF_CELLS

    // waiter pool
    uint64_t* w_consumer;
    int32_t*  w_edge;
    int32_t*  w_chunk;
    uint64_t* w_target;
    uint64_t* w_wait_seq;
    uint64_t* w_seed;
    uint64_t* w_armed_epoch;
    uint8_t*  w_state;
    int32_t*  w_count;          // 1 : number of waiters in pool

    // ready queue (dense array, front = index 0)
    uint64_t* r_ready_seq;
    int32_t*  r_waiter;
    uint64_t* r_observed_epoch;
    int32_t*  r_count;          // 1

    // pending store queue (sorted by due,seq,edge,chunk; front = index 0)
    uint64_t* p_producer;
    int32_t*  p_edge;
    int32_t*  p_chunk;
    uint64_t* p_increment;
    uint64_t* p_payload;
    uint64_t* p_store_seq;
    uint64_t* p_due;
    int32_t*  p_count;          // 1
};

__device__ __forceinline__ uint64_t rref_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= 1099511628211ULL; return h;
}
__device__ void rref_fnv(uint64_t* h, const void* p, int n) {
    const uint8_t* b = (const uint8_t*)p; uint64_t v = *h;
    for (int i = 0; i < n; ++i) v = rref_fnv_byte(v, b[i]); *h = v;
}
__device__ void rref_u8 (uint64_t* h, uint8_t v ){ rref_fnv(h,&v,1); }
__device__ void rref_u32(uint64_t* h, uint32_t v){ rref_fnv(h,&v,4); }
__device__ void rref_u64(uint64_t* h, uint64_t v){ rref_fnv(h,&v,8); }

__device__ uint64_t rref_payload_hash(uint64_t seed, uint64_t producer,
                                       uint32_t edge, uint32_t chunk,
                                       uint64_t store_seq, uint64_t epoch) {
    uint64_t h = 1469598103934665603ULL;
    rref_u64(&h, seed); rref_u64(&h, producer); rref_u32(&h, edge);
    rref_u32(&h, chunk); rref_u64(&h, store_seq); rref_u64(&h, epoch);
    return h;
}
__device__ uint64_t rref_consume_value(uint64_t seed, uint64_t consumer,
                                        uint32_t edge, uint32_t chunk,
                                        uint64_t counter, uint64_t payload,
                                        uint64_t wait_seq) {
    uint64_t h = 1469598103934665603ULL;
    rref_u64(&h, seed); rref_u64(&h, consumer); rref_u32(&h, edge);
    rref_u32(&h, chunk); rref_u64(&h, counter); rref_u64(&h, payload);
    rref_u64(&h, wait_seq);
    return h;
}

struct RrefCtx {
    MkRefState s;
    int edge_count;
    int max_chunks;
    uint64_t epoch_mod;
};

__device__ __forceinline__ int rref_cell_idx(RrefCtx& c, int e, int chunk) {
    return e * c.max_chunks + chunk;
}

__device__ void rref_emit(RrefCtx& c, uint8_t kind, uint32_t op_index,
                          uint64_t producer, uint64_t consumer, uint32_t edge,
                          uint32_t chunk_or_max, uint64_t counter_or_max,
                          uint64_t payload) {
    c.s.sc[1] += 1;  // event_seq
    uint64_t* h = c.s.event_hash;
    rref_u8(h, kind);
    rref_u64(h, c.s.sc[1]);
    rref_u32(h, op_index);
    rref_u64(h, c.s.sc[0]);  // clock
    rref_u64(h, producer);
    rref_u64(h, consumer);
    rref_u32(h, edge);
    rref_u32(h, chunk_or_max);
    rref_u64(h, counter_or_max);
    rref_u64(h, payload);
}

__device__ void rref_emit_invalid(RrefCtx& c, uint32_t op_index, uint32_t opcode) {
    rref_emit(c, MK_EV_INVALID, op_index, 0, 0, UINT32_MAX, UINT32_MAX,
              UINT64_MAX, (uint64_t)opcode);
    c.s.counts[MK_COUNT_INVALID] += 1;
}

__device__ __forceinline__ bool rref_edge_defined(RrefCtx& c, int e) {
    return e >= 0 && e < c.edge_count && c.s.e_defined[e] != 0;
}

// remove a waiter id from its cell wait queue (compact).
__device__ void rref_drop_from_wq(RrefCtx& c, int wid) {
    int e = c.s.w_edge[wid];
    int ch = c.s.w_chunk[wid];
    int ci = rref_cell_idx(c, e, ch);
    int n = c.s.c_wq_len[ci];
    int* q = &c.s.c_wq[ci * MK_REF_WQ_CAP];
    int w = 0;
    for (int i = 0; i < n; ++i) {
        if (q[i] == wid) continue;
        q[w++] = q[i];
    }
    c.s.c_wq_len[ci] = w;
}

// insertion of waiter id into a cell wait queue keeping wait_seq order.
__device__ void rref_wq_insert(RrefCtx& c, int ci, int wid) {
    int n = c.s.c_wq_len[ci];
    int* q = &c.s.c_wq[ci * MK_REF_WQ_CAP];
    uint64_t ws = c.s.w_wait_seq[wid];
    int pos = n;
    for (int i = 0; i < n; ++i) {
        if (c.s.w_wait_seq[q[i]] > ws) { pos = i; break; }
    }
    for (int i = n; i > pos; --i) q[i] = q[i - 1];
    q[pos] = wid;
    c.s.c_wq_len[ci] = n + 1;
}

__device__ void rref_evaluate_waiters(RrefCtx& c, int e, int chunk, uint32_t op_index) {
    int ci = rref_cell_idx(c, e, chunk);
    int n = c.s.c_wq_len[ci];
    int* q = &c.s.c_wq[ci * MK_REF_WQ_CAP];
    uint64_t edge_epoch = c.s.e_epoch[e];
    uint64_t counter = c.s.c_counter[ci];
    int w = 0;
    for (int i = 0; i < n; ++i) {
        int wid = q[i];
        uint8_t st = c.s.w_state[wid];
        if (st == MK_WAIT_CANCELLED || st == MK_WAIT_REMOVED || st == MK_WAIT_CONSUMED) {
            continue;  // dropped
        }
        if (c.s.w_armed_epoch[wid] != edge_epoch) {
            c.s.w_state[wid] = MK_WAIT_REMOVED;
            continue;
        }
        if (st == MK_WAIT_WAITING && c.s.w_target[wid] <= counter) {
            c.s.w_state[wid] = MK_WAIT_READY;
            int rc = *c.s.r_count;
            c.s.r_ready_seq[rc] = c.s.sc[4];  // ready_seq_next
            c.s.sc[4] += 1;
            c.s.r_waiter[rc] = wid;
            c.s.r_observed_epoch[rc] = edge_epoch;
            *c.s.r_count = rc + 1;
            rref_emit(c, MK_EV_WAITER_READY, op_index, 0, c.s.w_consumer[wid],
                      (uint32_t)e, (uint32_t)chunk, counter, c.s.w_target[wid]);
            c.s.counts[MK_COUNT_WAITER_READY] += 1;
            q[w++] = wid;
        } else {
            q[w++] = wid;
        }
    }
    c.s.c_wq_len[ci] = w;
}

__device__ void rref_clear_cell(RrefCtx& c, int ci) {
    c.s.c_counter[ci] = 0;
    c.s.c_page[ci] = MK_PAGE_EMPTY;
    c.s.c_payload[ci] = 0;
    c.s.c_producer[ci] = 0;
    c.s.c_store_seq[ci] = 0;
    c.s.c_release_seq[ci] = 0;
    c.s.c_wq_len[ci] = 0;
}

__device__ void rref_lazy_remove_edge_waiters(RrefCtx& c, int e) {
    int n = *c.s.w_count;
    for (int i = 0; i < n; ++i) {
        if (c.s.w_edge[i] == e &&
            (c.s.w_state[i] == MK_WAIT_WAITING || c.s.w_state[i] == MK_WAIT_READY)) {
            c.s.w_state[i] = MK_WAIT_REMOVED;
        }
    }
    for (int ch = 0; ch < c.max_chunks; ++ch)
        c.s.c_wq_len[rref_cell_idx(c, e, ch)] = 0;
}

__device__ void rref_op_define_edge(RrefCtx& c, uint32_t op_index, const MkOp& op) {
    int e = op.arg_a;
    int cc = op.arg_b;
    if (e < 0 || e >= c.edge_count || cc <= 0 || cc > c.max_chunks) {
        rref_emit_invalid(c, op_index, MK_OP_DEFINE_EDGE); return;
    }
    if (c.s.e_defined[e] != 0) {
        int oldcc = c.s.e_chunk_count[e];
        for (int ci2 = 0; ci2 < oldcc; ++ci2) {
            uint8_t ps = c.s.c_page[rref_cell_idx(c, e, ci2)];
            if (ps != MK_PAGE_EMPTY && ps != MK_PAGE_RELEASED) {
                rref_emit_invalid(c, op_index, MK_OP_DEFINE_EDGE); return;
            }
        }
    }
    rref_lazy_remove_edge_waiters(c, e);
    c.s.e_defined[e] = 1;
    c.s.e_chunk_count[e] = cc;
    c.s.e_epoch[e] = (c.s.e_epoch[e] + 1) % c.epoch_mod;
    for (int ch = 0; ch < c.max_chunks; ++ch) rref_clear_cell(c, rref_cell_idx(c, e, ch));
    rref_emit(c, MK_EV_EDGE_DEFINE, op_index, 0, 0, (uint32_t)e,
              UINT32_MAX, UINT64_MAX, 0);
    c.s.counts[MK_COUNT_EDGE_DEFINED] += 1;
}

__device__ void rref_op_reset_edge(RrefCtx& c, uint32_t op_index, const MkOp& op) {
    int e = op.arg_a;
    if (!rref_edge_defined(c, e)) { rref_emit_invalid(c, op_index, MK_OP_RESET_EDGE); return; }
    int cc = c.s.e_chunk_count[e];
    bool busy = false;
    for (int ch = 0; ch < cc; ++ch) {
        uint8_t ps = c.s.c_page[rref_cell_idx(c, e, ch)];
        if (ps == MK_PAGE_STORING || ps == MK_PAGE_READY || ps == MK_PAGE_CONSUMING) { busy = true; break; }
    }
    if (busy) {
        rref_emit(c, MK_EV_RESET_STALL, op_index, 0, 0, (uint32_t)e, UINT32_MAX, UINT64_MAX, 0);
        c.s.counts[MK_COUNT_RESET_STALL] += 1;
        return;
    }
    bool has_waiter = false;
    int wn = *c.s.w_count;
    for (int i = 0; i < wn; ++i) {
        if (c.s.w_edge[i] == e &&
            (c.s.w_state[i] == MK_WAIT_WAITING || c.s.w_state[i] == MK_WAIT_READY)) {
            has_waiter = true; break;
        }
    }
    if (has_waiter) {
        rref_emit(c, MK_EV_RESET_STALL, op_index, 0, 0, (uint32_t)e, UINT32_MAX, UINT64_MAX, 0);
        c.s.counts[MK_COUNT_RESET_STALL] += 1;
        return;
    }
    c.s.e_epoch[e] = (c.s.e_epoch[e] + 1) % c.epoch_mod;
    for (int ch = 0; ch < c.max_chunks; ++ch) rref_clear_cell(c, rref_cell_idx(c, e, ch));
    rref_emit(c, MK_EV_EDGE_RESET, op_index, 0, 0, (uint32_t)e, UINT32_MAX, UINT64_MAX, 0);
    c.s.counts[MK_COUNT_EDGE_RESET] += 1;
}

__device__ void rref_pending_insert(RrefCtx& c, uint64_t producer, int e, int chunk,
                                     uint64_t increment, uint64_t payload,
                                     uint64_t store_seq, uint64_t due) {
    int n = *c.s.p_count;
    int pos = n;
    for (int i = 0; i < n; ++i) {
        uint64_t qd = c.s.p_due[i], qs = c.s.p_store_seq[i];
        int qe = c.s.p_edge[i], qc = c.s.p_chunk[i];
        bool greater;  // is queue[i] greater than the new entry?
        if (qd != due) greater = qd > due;
        else if (qs != store_seq) greater = qs > store_seq;
        else if (qe != e) greater = qe > e;
        else greater = qc > chunk;
        if (greater) { pos = i; break; }
    }
    for (int i = n; i > pos; --i) {
        c.s.p_producer[i] = c.s.p_producer[i - 1];
        c.s.p_edge[i] = c.s.p_edge[i - 1];
        c.s.p_chunk[i] = c.s.p_chunk[i - 1];
        c.s.p_increment[i] = c.s.p_increment[i - 1];
        c.s.p_payload[i] = c.s.p_payload[i - 1];
        c.s.p_store_seq[i] = c.s.p_store_seq[i - 1];
        c.s.p_due[i] = c.s.p_due[i - 1];
    }
    c.s.p_producer[pos] = producer;
    c.s.p_edge[pos] = e;
    c.s.p_chunk[pos] = chunk;
    c.s.p_increment[pos] = increment;
    c.s.p_payload[pos] = payload;
    c.s.p_store_seq[pos] = store_seq;
    c.s.p_due[pos] = due;
    *c.s.p_count = n + 1;
}

__device__ void rref_op_produce(RrefCtx& c, uint32_t op_index, const MkOp& op) {
    uint64_t producer_id = (uint64_t)(uint32_t)op.arg_a;
    int e = op.arg_b;
    int first_chunk = op.arg_c;
    int chunk_count = op.arg_d;
    uint64_t increment = (uint64_t)op.arg_i64;
    uint64_t payload_seed = (uint64_t)(uint32_t)op.arg_e;
    uint64_t store_latency = (uint64_t)(uint32_t)op.arg_f;

    if (!rref_edge_defined(c, e) || chunk_count <= 0 || first_chunk < 0) {
        rref_emit_invalid(c, op_index, MK_OP_PRODUCE); return;
    }
    if ((int64_t)first_chunk + (int64_t)chunk_count > (int64_t)c.s.e_chunk_count[e]) {
        rref_emit_invalid(c, op_index, MK_OP_PRODUCE); return;
    }
    for (int k = 0; k < chunk_count; ++k) {
        int ch = first_chunk + k;
        int ci = rref_cell_idx(c, e, ch);
        uint8_t ps = c.s.c_page[ci];
        if (ps == MK_PAGE_STORING || ps == MK_PAGE_READY || ps == MK_PAGE_CONSUMING) {
            rref_emit(c, MK_EV_PRODUCE_STALL_CHUNK, op_index, producer_id, 0,
                      (uint32_t)e, (uint32_t)ch, UINT64_MAX, 0);
            c.s.counts[MK_COUNT_PRODUCE_STALL_CHUNK] += 1;
            return;
        }
        uint64_t store_seq = c.s.sc[2]; c.s.sc[2] += 1;
        uint64_t ph = rref_payload_hash(payload_seed, producer_id, (uint32_t)e,
                                        (uint32_t)ch, store_seq, c.s.sc[5]);
        c.s.c_page[ci] = MK_PAGE_STORING;
        c.s.c_producer[ci] = producer_id;
        c.s.c_store_seq[ci] = store_seq;
        c.s.c_payload[ci] = ph;
        c.s.c_release_seq[ci] = 0;
        rref_pending_insert(c, producer_id, e, ch, increment, ph, store_seq,
                            c.s.sc[0] + store_latency);
        rref_emit(c, MK_EV_STORE_ISSUE, op_index, producer_id, 0,
                  (uint32_t)e, (uint32_t)ch, UINT64_MAX, ph);
        c.s.counts[MK_COUNT_STORE_ISSUED] += 1;
    }
}

__device__ void rref_pending_erase(RrefCtx& c, int idx) {
    int n = *c.s.p_count;
    for (int i = idx; i < n - 1; ++i) {
        c.s.p_producer[i] = c.s.p_producer[i + 1];
        c.s.p_edge[i] = c.s.p_edge[i + 1];
        c.s.p_chunk[i] = c.s.p_chunk[i + 1];
        c.s.p_increment[i] = c.s.p_increment[i + 1];
        c.s.p_payload[i] = c.s.p_payload[i + 1];
        c.s.p_store_seq[i] = c.s.p_store_seq[i + 1];
        c.s.p_due[i] = c.s.p_due[i + 1];
    }
    *c.s.p_count = n - 1;
}

__device__ void rref_op_advance(RrefCtx& c, uint32_t op_index, const MkOp& op) {
    uint64_t delta = (uint64_t)(uint32_t)op.arg_a;
    int max_completions = op.arg_b;
    c.s.sc[0] = c.s.sc[0] + delta;
    if (max_completions <= 0) return;
    int processed = 0;
    int i = 0;
    while (i < *c.s.p_count && processed < max_completions) {
        if (c.s.p_due[i] > c.s.sc[0]) { ++i; continue; }
        uint64_t producer = c.s.p_producer[i];
        int e = c.s.p_edge[i];
        int ch = c.s.p_chunk[i];
        uint64_t increment = c.s.p_increment[i];
        uint64_t payload = c.s.p_payload[i];
        uint64_t store_seq = c.s.p_store_seq[i];
        rref_pending_erase(c, i);
        ++processed;
        int ci = rref_cell_idx(c, e, ch);
        if (c.s.c_store_seq[ci] == store_seq && c.s.c_page[ci] == MK_PAGE_STORING) {
            c.s.c_page[ci] = MK_PAGE_READY;
            c.s.c_payload[ci] = payload;
            c.s.c_counter[ci] = c.s.c_counter[ci] + increment;
            rref_emit(c, MK_EV_STORE_COMPLETE, op_index, producer, 0,
                      (uint32_t)e, (uint32_t)ch, c.s.c_counter[ci], c.s.c_payload[ci]);
            c.s.counts[MK_COUNT_STORE_COMPLETE] += 1;
            rref_evaluate_waiters(c, e, ch, op_index);
        } else {
            rref_emit(c, MK_EV_STORE_STALE_DROP, op_index, producer, 0,
                      (uint32_t)e, (uint32_t)ch, UINT64_MAX, store_seq);
            c.s.counts[MK_COUNT_STORE_STALE_DROP] += 1;
        }
    }
}

__device__ int rref_find_nonterminal(RrefCtx& c, uint64_t consumer, int e, int chunk) {
    int n = *c.s.w_count;
    for (int i = 0; i < n; ++i) {
        if (c.s.w_consumer[i] == consumer && c.s.w_edge[i] == e && c.s.w_chunk[i] == chunk &&
            (c.s.w_state[i] == MK_WAIT_WAITING || c.s.w_state[i] == MK_WAIT_READY))
            return i;
    }
    return -1;
}

__device__ void rref_op_arm_wait(RrefCtx& c, uint32_t op_index, const MkOp& op) {
    uint64_t consumer_id = (uint64_t)(uint32_t)op.arg_a;
    int e = op.arg_b;
    int chunk = op.arg_c;
    uint64_t target = (uint64_t)op.arg_i64;
    uint64_t seed = (uint64_t)(uint32_t)op.arg_e;

    if (!rref_edge_defined(c, e) || chunk < 0 || chunk >= c.s.e_chunk_count[e]) {
        rref_emit_invalid(c, op_index, MK_OP_ARM_WAIT); return;
    }
    if (rref_find_nonterminal(c, consumer_id, e, chunk) >= 0) {
        rref_emit_invalid(c, op_index, MK_OP_ARM_WAIT); return;
    }
    int ci = rref_cell_idx(c, e, chunk);
    int wid = *c.s.w_count;
    c.s.w_consumer[wid] = consumer_id;
    c.s.w_edge[wid] = e;
    c.s.w_chunk[wid] = chunk;
    c.s.w_target[wid] = target;
    c.s.w_wait_seq[wid] = c.s.sc[3]; c.s.sc[3] += 1;
    c.s.w_seed[wid] = seed;
    c.s.w_armed_epoch[wid] = c.s.e_epoch[e];
    *c.s.w_count = wid + 1;

    if (c.s.c_page[ci] == MK_PAGE_READY && c.s.c_counter[ci] >= target) {
        c.s.w_state[wid] = MK_WAIT_READY;
        int rc = *c.s.r_count;
        c.s.r_ready_seq[rc] = c.s.sc[4]; c.s.sc[4] += 1;
        c.s.r_waiter[rc] = wid;
        c.s.r_observed_epoch[rc] = c.s.e_epoch[e];
        *c.s.r_count = rc + 1;
        rref_emit(c, MK_EV_WAITER_READY_IMMEDIATE, op_index, 0, consumer_id,
                  (uint32_t)e, (uint32_t)chunk, c.s.c_counter[ci], target);
        c.s.counts[MK_COUNT_WAITER_READY_IMMEDIATE] += 1;
    } else {
        c.s.w_state[wid] = MK_WAIT_WAITING;
        rref_wq_insert(c, ci, wid);
        rref_emit(c, MK_EV_WAITER_ARM, op_index, 0, consumer_id,
                  (uint32_t)e, (uint32_t)chunk, UINT64_MAX, target);
        c.s.counts[MK_COUNT_WAITER_ARMED] += 1;
    }
}

__device__ void rref_ready_pop_front(RrefCtx& c) {
    int n = *c.s.r_count;
    for (int i = 0; i < n - 1; ++i) {
        c.s.r_ready_seq[i] = c.s.r_ready_seq[i + 1];
        c.s.r_waiter[i] = c.s.r_waiter[i + 1];
        c.s.r_observed_epoch[i] = c.s.r_observed_epoch[i + 1];
    }
    *c.s.r_count = n - 1;
}

__device__ void rref_op_consume(RrefCtx& c, uint32_t op_index, const MkOp& op) {
    int limit = op.arg_a;
    if (limit <= 0) return;
    int valid = 0;
    while (valid < limit && *c.s.r_count > 0) {
        int wid = c.s.r_waiter[0];
        uint64_t observed = c.s.r_observed_epoch[0];
        int e = c.s.w_edge[wid];
        int ch = c.s.w_chunk[wid];
        int ci = rref_cell_idx(c, e, ch);
        uint64_t consumer = c.s.w_consumer[wid];

        if (c.s.w_state[wid] != MK_WAIT_READY || observed != c.s.e_epoch[e]) {
            rref_emit(c, MK_EV_READY_STALE_DROP, op_index, 0, consumer,
                      (uint32_t)e, (uint32_t)ch, UINT64_MAX, 0);
            c.s.counts[MK_COUNT_READY_STALE_DROP] += 1;
            rref_ready_pop_front(c);
            continue;
        }
        if (c.s.c_page[ci] != MK_PAGE_READY) {
            rref_emit(c, MK_EV_READY_STALE_DROP, op_index, 0, consumer,
                      (uint32_t)e, (uint32_t)ch, UINT64_MAX, 0);
            c.s.counts[MK_COUNT_READY_STALE_DROP] += 1;
            rref_ready_pop_front(c);
            continue;
        }
        if (c.s.c_counter[ci] < c.s.w_target[wid]) {
            c.s.w_state[wid] = MK_WAIT_WAITING;
            rref_wq_insert(c, ci, wid);
            rref_emit(c, MK_EV_READY_REQUEUE, op_index, 0, consumer,
                      (uint32_t)e, (uint32_t)ch, c.s.c_counter[ci], c.s.w_target[wid]);
            c.s.counts[MK_COUNT_READY_REQUEUE] += 1;
            rref_ready_pop_front(c);
            continue;
        }
        // consume
        c.s.c_page[ci] = MK_PAGE_CONSUMING;
        uint64_t cval = rref_consume_value(c.s.w_seed[wid], consumer, (uint32_t)e,
                                           (uint32_t)ch, c.s.c_counter[ci],
                                           c.s.c_payload[ci], c.s.w_wait_seq[wid]);
        rref_emit(c, MK_EV_CONSUME_CHUNK, op_index, 0, consumer,
                  (uint32_t)e, (uint32_t)ch, c.s.c_counter[ci], cval);
        c.s.counts[MK_COUNT_CONSUME_CHUNK] += 1;

        c.s.c_page[ci] = MK_PAGE_RELEASED;
        c.s.c_producer[ci] = 0;
        c.s.c_store_seq[ci] = 0;
        uint64_t release_seq = c.s.sc[1] + 1;  // event_seq this release will carry
        c.s.c_release_seq[ci] = release_seq;
        rref_emit(c, MK_EV_CHUNK_RELEASE, op_index, 0, consumer,
                  (uint32_t)e, (uint32_t)ch, c.s.c_counter[ci], release_seq);
        c.s.counts[MK_COUNT_CHUNK_RELEASE] += 1;

        c.s.w_state[wid] = MK_WAIT_CONSUMED;
        rref_ready_pop_front(c);
        ++valid;
    }
}

__device__ void rref_op_cancel_wait(RrefCtx& c, uint32_t op_index, const MkOp& op) {
    uint64_t consumer_id = (uint64_t)(uint32_t)op.arg_a;
    int e = op.arg_b;
    int chunk = op.arg_c;
    if (e < 0 || e >= c.edge_count) { rref_emit_invalid(c, op_index, MK_OP_CANCEL_WAIT); return; }
    int wid = rref_find_nonterminal(c, consumer_id, e, chunk);
    if (wid < 0) { rref_emit_invalid(c, op_index, MK_OP_CANCEL_WAIT); return; }
    c.s.w_state[wid] = MK_WAIT_CANCELLED;
    rref_emit(c, MK_EV_WAITER_CANCEL, op_index, 0, consumer_id,
              (uint32_t)e, (uint32_t)chunk, UINT64_MAX, 0);
    c.s.counts[MK_COUNT_WAITER_CANCEL] += 1;
}

__device__ void rref_op_force_counter(RrefCtx& c, uint32_t op_index, const MkOp& op) {
    int e = op.arg_a;
    int chunk = op.arg_b;
    uint64_t amount = (uint64_t)op.arg_i64;
    if (!rref_edge_defined(c, e) || chunk < 0 || chunk >= c.s.e_chunk_count[e]) {
        rref_emit_invalid(c, op_index, MK_OP_FORCE_COUNTER); return;
    }
    int ci = rref_cell_idx(c, e, chunk);
    c.s.c_counter[ci] = c.s.c_counter[ci] + amount;
    rref_emit(c, MK_EV_COUNTER_FORCE, op_index, 0, 0, (uint32_t)e,
              (uint32_t)chunk, c.s.c_counter[ci], amount);
    c.s.counts[MK_COUNT_COUNTER_FORCE] += 1;
    rref_evaluate_waiters(c, e, chunk, op_index);
}

__global__ void mk_ref_run_kernel(MkRefState st, const MkOp* ops, int op_count,
                                  int edge_count, int max_chunks, uint64_t epoch_mod) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    RrefCtx c;
    c.s = st;
    c.edge_count = edge_count;
    c.max_chunks = max_chunks;
    c.epoch_mod = epoch_mod;
    for (int i = 0; i < op_count; ++i) {
        const MkOp op = ops[i];
        const uint32_t op_index = (uint32_t)(*c.s.op_index_global);
        switch (op.kind) {
            case MK_OP_DEFINE_EDGE: rref_op_define_edge(c, op_index, op); break;
            case MK_OP_RESET_EDGE: rref_op_reset_edge(c, op_index, op); break;
            case MK_OP_PRODUCE: rref_op_produce(c, op_index, op); break;
            case MK_OP_ADVANCE: rref_op_advance(c, op_index, op); break;
            case MK_OP_ARM_WAIT: rref_op_arm_wait(c, op_index, op); break;
            case MK_OP_CONSUME: rref_op_consume(c, op_index, op); break;
            case MK_OP_CANCEL_WAIT: rref_op_cancel_wait(c, op_index, op); break;
            case MK_OP_FORCE_COUNTER: rref_op_force_counter(c, op_index, op); break;
            default: rref_emit_invalid(c, op_index, (uint32_t)op.kind); break;
        }
        *c.s.op_index_global += 1;
    }
}

__global__ void mk_ref_finalize_kernel(MkRefState st, int edge_count, int max_chunks,
                                        uint64_t* out_cell, uint64_t* out_waiter,
                                        uint64_t* out_ready, uint64_t* out_pending) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // cell_hash : defined edges ascending, chunks ascending.
    {
        uint64_t h = 1469598103934665603ULL;
        for (int e = 0; e < edge_count; ++e) {
            if (st.e_defined[e] == 0) continue;
            int cc = st.e_chunk_count[e];
            for (int ch = 0; ch < cc; ++ch) {
                int ci = e * max_chunks + ch;
                rref_u32(&h, (uint32_t)e);
                rref_u64(&h, st.e_epoch[e]);
                rref_u32(&h, (uint32_t)ch);
                rref_u64(&h, st.c_counter[ci]);
                rref_u8(&h, st.c_page[ci]);
                rref_u64(&h, st.c_payload[ci]);
                rref_u64(&h, st.c_producer[ci]);
                rref_u64(&h, st.c_store_seq[ci]);
                rref_u64(&h, st.c_release_seq[ci]);
            }
        }
        *out_cell = h;
    }
    // waiter_hash : nonterminal waiters sorted by (edge,chunk,wait_seq,consumer).
    {
        int n = *st.w_count;
        // selection sort indices of nonterminal waiters.
        // collect into a temporary order array stored in r? we cannot allocate;
        // do an index selection over the pool directly.
        // Build a boolean of nonterminal then repeatedly pick the min.
        uint64_t h = 1469598103934665603ULL;
        // Mark a "used" via a local scan; since n<=MK_MAX_WAITERS we just do
        // O(n^2) selection emitting in sorted order.
        // Use w_armed_epoch? no, must not modify. Use a small approach: for
        // each output slot find global min not yet emitted. Track emitted via
        // a high bit is unsafe; instead use the fact wait_seq is unique. We use
        // a "last emitted key" cursor.
        // key ordering: (edge,chunk,wait_seq,consumer); wait_seq unique => key
        // is total order. Emit in increasing key by repeatedly scanning for the
        // smallest key strictly greater than the previous one.
        bool first = true;
        int pe = 0, pc = 0; uint64_t pw = 0, pcons = 0;
        for (;;) {
            int best = -1;
            for (int i = 0; i < n; ++i) {
                if (st.w_state[i] != MK_WAIT_WAITING && st.w_state[i] != MK_WAIT_READY) continue;
                int ie = st.w_edge[i], ic = st.w_chunk[i];
                uint64_t iw = st.w_wait_seq[i], icons = st.w_consumer[i];
                // must be strictly greater than previous emitted key.
                if (!first) {
                    bool gt;
                    if (ie != pe) gt = ie > pe;
                    else if (ic != pc) gt = ic > pc;
                    else if (iw != pw) gt = iw > pw;
                    else gt = icons > pcons;
                    if (!gt) continue;
                }
                if (best < 0) { best = i; continue; }
                int be = st.w_edge[best], bc = st.w_chunk[best];
                uint64_t bw = st.w_wait_seq[best], bcons = st.w_consumer[best];
                bool less;
                if (ie != be) less = ie < be;
                else if (ic != bc) less = ic < bc;
                else if (iw != bw) less = iw < bw;
                else less = icons < bcons;
                if (less) best = i;
            }
            if (best < 0) break;
            rref_u64(&h, st.w_consumer[best]);
            rref_u32(&h, (uint32_t)st.w_edge[best]);
            rref_u32(&h, (uint32_t)st.w_chunk[best]);
            rref_u64(&h, st.w_target[best]);
            rref_u64(&h, st.w_wait_seq[best]);
            rref_u64(&h, st.w_seed[best]);
            rref_u8(&h, st.w_state[best]);
            pe = st.w_edge[best]; pc = st.w_chunk[best];
            pw = st.w_wait_seq[best]; pcons = st.w_consumer[best];
            first = false;
        }
        *out_waiter = h;
    }
    // ready_hash : ready entries by ready queue order.
    {
        uint64_t h = 1469598103934665603ULL;
        int n = *st.r_count;
        for (int i = 0; i < n; ++i) {
            int wid = st.r_waiter[i];
            rref_u64(&h, st.r_ready_seq[i]);
            rref_u64(&h, st.w_consumer[wid]);
            rref_u32(&h, (uint32_t)st.w_edge[wid]);
            rref_u32(&h, (uint32_t)st.w_chunk[wid]);
            rref_u64(&h, st.r_observed_epoch[i]);
        }
        *out_ready = h;
    }
    // pending_hash : pending stores by queue order.
    {
        uint64_t h = 1469598103934665603ULL;
        int n = *st.p_count;
        for (int i = 0; i < n; ++i) {
            rref_u64(&h, st.p_due[i]);
            rref_u64(&h, st.p_store_seq[i]);
            rref_u64(&h, st.p_producer[i]);
            rref_u32(&h, (uint32_t)st.p_edge[i]);
            rref_u32(&h, (uint32_t)st.p_chunk[i]);
            rref_u64(&h, st.p_increment[i]);
            rref_u64(&h, st.p_payload[i]);
        }
        *out_pending = h;
    }
}

// ---------------- host plumbing ----------------

static cudaError_t rref_alloc_all(MkRefState* st) {
    cudaError_t e;
    const size_t CELLS = MK_REF_CELLS;
    const int EC = st->spec.edge_count;
#define RREF_MALLOC(p, bytes) do { e = cudaMalloc((void**)&(p), (bytes)); if (e != cudaSuccess) return e; } while(0)
    RREF_MALLOC(st->sc, sizeof(uint64_t) * 6);
    RREF_MALLOC(st->op_index_global, sizeof(uint64_t));
    RREF_MALLOC(st->event_hash, sizeof(uint64_t));
    RREF_MALLOC(st->counts, sizeof(uint64_t) * MK_COUNT_TOTAL);

    RREF_MALLOC(st->e_defined, sizeof(uint8_t) * (size_t)EC);
    RREF_MALLOC(st->e_chunk_count, sizeof(int32_t) * (size_t)EC);
    RREF_MALLOC(st->e_epoch, sizeof(uint64_t) * (size_t)EC);

    RREF_MALLOC(st->c_counter, sizeof(uint64_t) * CELLS);
    RREF_MALLOC(st->c_page, sizeof(uint8_t) * CELLS);
    RREF_MALLOC(st->c_payload, sizeof(uint64_t) * CELLS);
    RREF_MALLOC(st->c_producer, sizeof(uint64_t) * CELLS);
    RREF_MALLOC(st->c_store_seq, sizeof(uint64_t) * CELLS);
    RREF_MALLOC(st->c_release_seq, sizeof(uint64_t) * CELLS);
    RREF_MALLOC(st->c_wq, sizeof(int32_t) * CELLS * (size_t)MK_REF_WQ_CAP);
    RREF_MALLOC(st->c_wq_len, sizeof(int32_t) * CELLS);

    RREF_MALLOC(st->w_consumer, sizeof(uint64_t) * (size_t)MK_MAX_WAITERS);
    RREF_MALLOC(st->w_edge, sizeof(int32_t) * (size_t)MK_MAX_WAITERS);
    RREF_MALLOC(st->w_chunk, sizeof(int32_t) * (size_t)MK_MAX_WAITERS);
    RREF_MALLOC(st->w_target, sizeof(uint64_t) * (size_t)MK_MAX_WAITERS);
    RREF_MALLOC(st->w_wait_seq, sizeof(uint64_t) * (size_t)MK_MAX_WAITERS);
    RREF_MALLOC(st->w_seed, sizeof(uint64_t) * (size_t)MK_MAX_WAITERS);
    RREF_MALLOC(st->w_armed_epoch, sizeof(uint64_t) * (size_t)MK_MAX_WAITERS);
    RREF_MALLOC(st->w_state, sizeof(uint8_t) * (size_t)MK_MAX_WAITERS);
    RREF_MALLOC(st->w_count, sizeof(int32_t));

    RREF_MALLOC(st->r_ready_seq, sizeof(uint64_t) * (size_t)MK_MAX_READY);
    RREF_MALLOC(st->r_waiter, sizeof(int32_t) * (size_t)MK_MAX_READY);
    RREF_MALLOC(st->r_observed_epoch, sizeof(uint64_t) * (size_t)MK_MAX_READY);
    RREF_MALLOC(st->r_count, sizeof(int32_t));

    RREF_MALLOC(st->p_producer, sizeof(uint64_t) * (size_t)MK_MAX_STORE_EVENTS);
    RREF_MALLOC(st->p_edge, sizeof(int32_t) * (size_t)MK_MAX_STORE_EVENTS);
    RREF_MALLOC(st->p_chunk, sizeof(int32_t) * (size_t)MK_MAX_STORE_EVENTS);
    RREF_MALLOC(st->p_increment, sizeof(uint64_t) * (size_t)MK_MAX_STORE_EVENTS);
    RREF_MALLOC(st->p_payload, sizeof(uint64_t) * (size_t)MK_MAX_STORE_EVENTS);
    RREF_MALLOC(st->p_store_seq, sizeof(uint64_t) * (size_t)MK_MAX_STORE_EVENTS);
    RREF_MALLOC(st->p_due, sizeof(uint64_t) * (size_t)MK_MAX_STORE_EVENTS);
    RREF_MALLOC(st->p_count, sizeof(int32_t));
#undef RREF_MALLOC
    return cudaSuccess;
}

__global__ void mk_ref_reset_kernel(MkRefState st, int edge_count, int max_chunks) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    st.sc[0] = 0; st.sc[1] = 0; st.sc[2] = 1; st.sc[3] = 1; st.sc[4] = 1; st.sc[5] = 0;
    *st.op_index_global = 0;
    *st.event_hash = 1469598103934665603ULL;
    for (int i = 0; i < MK_COUNT_TOTAL; ++i) st.counts[i] = 0;
    for (int e = 0; e < edge_count; ++e) {
        st.e_defined[e] = 0;
        st.e_chunk_count[e] = 0;
        st.e_epoch[e] = 0;
    }
    int cells = edge_count * max_chunks;
    for (int ci = 0; ci < cells; ++ci) {
        st.c_counter[ci] = 0;
        st.c_page[ci] = MK_PAGE_EMPTY;
        st.c_payload[ci] = 0;
        st.c_producer[ci] = 0;
        st.c_store_seq[ci] = 0;
        st.c_release_seq[ci] = 0;
        st.c_wq_len[ci] = 0;
    }
    *st.w_count = 0;
    *st.r_count = 0;
    *st.p_count = 0;
}

extern "C" size_t solution_workspace_bytes(const MkProblemSpec* spec) {
    if (!mk_validate_problem_spec(spec)) return 0;
    return 256;
}

extern "C" cudaError_t solution_init(const MkProblemSpec* spec, void** state_out,
                                     cudaStream_t stream) {
    if (!mk_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    MkRefState* st = (MkRefState*)malloc(sizeof(MkRefState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(MkRefState));
    memcpy(&st->spec, spec, sizeof(MkProblemSpec));
    st->epoch_mod = (uint64_t)spec->max_epoch + 1;

    cudaError_t e = rref_alloc_all(st);
    if (e != cudaSuccess) { solution_destroy(st); return e; }

    mk_ref_reset_kernel<<<1, 1, 0, stream>>>(*st, spec->edge_count, spec->max_chunks_per_edge);
    e = cudaPeekAtLastError();
    if (e != cudaSuccess) { solution_destroy(st); return e; }

    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(void* state, const MkRunSpec* run,
                                    const void* inputs_void, void* outputs_void,
                                    void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    (void)workspace;
    if (!state || !mk_validate_run_spec(run) || !inputs_void || !outputs_void)
        return cudaErrorInvalidValue;
    if (workspace_bytes < 1) return cudaErrorInvalidValue;

    MkRefState* st = (MkRefState*)state;
    const MkInputs* in = (const MkInputs*)inputs_void;
    MkOutputs* out = (MkOutputs*)outputs_void;

    if (run->op_count > st->spec.max_ops) return cudaErrorInvalidValue;
    if (run->op_count > 0 && !in->ops) return cudaErrorInvalidValue;
    if (!out->counts || !out->event_hash || !out->cell_hash || !out->waiter_hash ||
        !out->ready_hash || !out->pending_hash || !out->state_scalars)
        return cudaErrorInvalidValue;

    const int EC = st->spec.edge_count;
    const int MC = st->spec.max_chunks_per_edge;

    if (run->op_count > 0) {
        mk_ref_run_kernel<<<1, 1, 0, stream>>>(*st, in->ops, run->op_count, EC, MC, st->epoch_mod);
        cudaError_t e = cudaPeekAtLastError();
        if (e != cudaSuccess) return e;
    }

    mk_ref_finalize_kernel<<<1, 1, 0, stream>>>(*st, EC, MC, out->cell_hash,
                                                out->waiter_hash, out->ready_hash,
                                                out->pending_hash);
    cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) return e;

    e = cudaMemcpyAsync(out->counts, st->counts, sizeof(uint64_t) * MK_COUNT_TOTAL,
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(out->event_hash, st->event_hash, sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(out->state_scalars, st->sc, sizeof(uint64_t) * 6,
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    MkRefState* st = (MkRefState*)state;
    mk_ref_reset_kernel<<<1, 1, 0, stream>>>(*st, st->spec.edge_count,
                                             st->spec.max_chunks_per_edge);
    return cudaPeekAtLastError();
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    MkRefState* st = (MkRefState*)state;
    cudaFree(st->sc); cudaFree(st->op_index_global); cudaFree(st->event_hash);
    cudaFree(st->counts);
    cudaFree(st->e_defined); cudaFree(st->e_chunk_count); cudaFree(st->e_epoch);
    cudaFree(st->c_counter); cudaFree(st->c_page); cudaFree(st->c_payload);
    cudaFree(st->c_producer); cudaFree(st->c_store_seq); cudaFree(st->c_release_seq);
    cudaFree(st->c_wq); cudaFree(st->c_wq_len);
    cudaFree(st->w_consumer); cudaFree(st->w_edge); cudaFree(st->w_chunk);
    cudaFree(st->w_target); cudaFree(st->w_wait_seq); cudaFree(st->w_seed);
    cudaFree(st->w_armed_epoch); cudaFree(st->w_state); cudaFree(st->w_count);
    cudaFree(st->r_ready_seq); cudaFree(st->r_waiter); cudaFree(st->r_observed_epoch);
    cudaFree(st->r_count);
    cudaFree(st->p_producer); cudaFree(st->p_edge); cudaFree(st->p_chunk);
    cudaFree(st->p_increment); cudaFree(st->p_payload); cudaFree(st->p_store_seq);
    cudaFree(st->p_due); cudaFree(st->p_count);
    free(st);
}
