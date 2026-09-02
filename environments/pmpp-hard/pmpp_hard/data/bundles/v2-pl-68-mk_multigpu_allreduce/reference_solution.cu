// PMPP_CANARY_68_c40e9a6600 -- held-out canary; MUST NOT appear in any submission
// file: mk_multigpu_allreduce_reference.cu
//
// Reference device implementation of T68 (MK9). A single-thread persistent
// kernel operating on device-resident state. Scheduler queues are flat arrays
// processed in append order; remote events are kept in a flat pool and selected
// by a canonical-min scan. Independent of naive.cu and the host oracle.

#include "mk_multigpu_allreduce_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace {

struct RCell {
    int64_t local_value;
    int64_t accum_value;
    int64_t final_value;
    uint64_t last_update_seq;
    uint8_t have_local;
    uint8_t local_ready;
    uint8_t reduced_owner_ready;
    uint8_t final_ready;
    uint8_t pad[4];
};

struct RColl {
    uint64_t coll_id;
    uint64_t collective_seq;
    int32_t chunk_count;
    int32_t active_rank_count;
    int32_t owners_reduced;
    uint8_t used;
    uint8_t status;
    uint8_t phase;
    uint8_t rs_started;
    uint8_t ag_started;
    uint8_t pad[7];
};

struct RAction {
    uint64_t coll_id;
    uint64_t phase_seq;
    int64_t value;
    int32_t coll_slot;
    int32_t chunk;
    int32_t phase_index;
    int32_t src;
    uint8_t action_kind;
    uint8_t pad[7];
};

struct RRemote {
    uint64_t due_clock;
    uint64_t remote_seq;
    uint64_t coll_id;
    int64_t value;
    int32_t src;
    int32_t dst;
    int32_t coll_slot;
    int32_t phase_index;
    int32_t chunk;
    int32_t credit_return_src;
    int32_t credit_return_dst;
    uint8_t phase_kind;
    uint8_t pad[3];
};

}  // namespace

struct MgaReferenceState {
    MgaProblemSpec spec;
    int rank_count, chunk_max, max_colls, max_remote, max_credit, remote_latency, max_sched;
    int phidx_bound;

    RColl* colls;                 // max_colls
    // active_ranks[slot * rank_count + i], pos_of_rank[slot * rank_count + rank]
    int32_t* active_ranks;        // max_colls * rank_count
    int32_t* pos_of_rank;         // max_colls * rank_count
    int64_t* credit;              // max_colls * rank_count  (edge pos->next)
    RCell* cells;                 // max_colls * rank_count * chunk_max

    RAction* sched;               // rank_count * max_sched
    int32_t* sched_count;         // rank_count

    RRemote* pending;             // max_remote
    int32_t* pending_count;       // 1

    uint64_t* signal;             // rank_count*max_colls*3*phidx_bound*chunk_max
    // recv buffers folded into signal-write effect; we keep a small struct array.
    int64_t* recv_value;          // rank_count*max_colls*chunk_max*3
    int32_t* recv_src;
    uint64_t* recv_seq;
    uint8_t* recv_valid;

    int64_t* counters;            // 19
    uint64_t* scalars;            // [clock, event_seq, coll_seq_next, remote_seq_next, phase_seq_next, ev_hash]
};

// ---------- device helpers ----------

__device__ __forceinline__ uint64_t rfnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= 1099511628211ULL; return h;
}
__device__ void rfnv(uint64_t* h, const void* p, int n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t v = *h;
    for (int i = 0; i < n; ++i) v = rfnv_byte(v, b[i]);
    *h = v;
}

struct RCtx {
    int rank_count, chunk_max, max_colls, max_remote, max_credit, remote_latency, max_sched;
    int phidx_bound;
    RColl* colls;
    int32_t* active_ranks;
    int32_t* pos_of_rank;
    int64_t* credit;
    RCell* cells;
    RAction* sched;
    int32_t* sched_count;
    RRemote* pending;
    int32_t* pending_count;
    uint64_t* signal;
    int64_t* recv_value;
    int32_t* recv_src;
    uint64_t* recv_seq;
    uint8_t* recv_valid;
    int64_t* counters;
    uint64_t clock;
    uint64_t event_seq;
    uint64_t coll_seq_next;
    uint64_t remote_seq_next;
    uint64_t phase_seq_next;
    uint64_t ev_hash;
    uint32_t op_index;
};

__device__ __forceinline__ int r_sig_index(RCtx* c, int rank, int slot, int pk, int pidx, int ch) {
    return (((rank * c->max_colls + slot) * 3 + pk) * c->phidx_bound + pidx) * c->chunk_max + ch;
}
__device__ __forceinline__ int r_recv_index(RCtx* c, int dst, int slot, int ch, int pk) {
    return ((dst * c->max_colls + slot) * c->chunk_max + ch) * 3 + pk;
}
__device__ __forceinline__ int r_cell_index(RCtx* c, int slot, int rank, int ch) {
    return (slot * c->rank_count + rank) * c->chunk_max + ch;
}

__device__ int r_find_coll(RCtx* c, uint64_t id) {
    for (int i = 0; i < c->max_colls; ++i)
        if (c->colls[i].used && c->colls[i].coll_id == id) return i;
    return -1;
}

__device__ int r_popcount(RCtx* c, uint32_t m) {
    int cnt = 0;
    for (int i = 0; i < c->rank_count; ++i) if (m & (1u << i)) ++cnt;
    return cnt;
}

__device__ int r_next_rank(RCtx* c, int slot, int rank) {
    RColl* coll = &c->colls[slot];
    int pos = c->pos_of_rank[slot * c->rank_count + rank];
    int np = (pos + 1) % coll->active_rank_count;
    return c->active_ranks[slot * c->rank_count + np];
}

__device__ int r_ring_slot(RCtx* c, int slot, int rank, int p) {
    RColl* coll = &c->colls[slot];
    int pos = c->pos_of_rank[slot * c->rank_count + rank];
    int R = coll->active_rank_count;
    return ((pos - p) % R + R) % R;
}

__device__ void r_emit(RCtx* c, uint8_t kind, uint64_t coll_or, uint32_t src_or,
                        uint32_t dst_or, uint8_t pk_or, uint32_t pidx_or,
                        uint32_t chunk_or, int64_t value_or) {
    uint64_t es = c->event_seq;
    uint64_t* h = &c->ev_hash;
    rfnv(h, &kind, 1);
    rfnv(h, &es, 8);
    rfnv(h, &c->op_index, 4);
    uint64_t clk = c->clock; rfnv(h, &clk, 8);
    rfnv(h, &coll_or, 8);
    rfnv(h, &src_or, 4);
    rfnv(h, &dst_or, 4);
    rfnv(h, &pk_or, 1);
    rfnv(h, &pidx_or, 4);
    rfnv(h, &chunk_or, 4);
    rfnv(h, &value_or, 8);
    c->event_seq = es + 1;
}

__device__ void r_enqueue(RCtx* c, int rank, RAction* a) {
    int cnt = c->sched_count[rank];
    a->phase_seq = c->phase_seq_next++;
    if (cnt >= c->max_sched) return;  // bounded
    c->sched[(size_t)rank * c->max_sched + cnt] = *a;
    c->sched_count[rank] = cnt + 1;
}

__device__ void r_sched_pop_front(RCtx* c, int rank) {
    int cnt = c->sched_count[rank];
    RAction* base = c->sched + (size_t)rank * c->max_sched;
    for (int i = 1; i < cnt; ++i) base[i - 1] = base[i];
    c->sched_count[rank] = cnt - 1;
}

__device__ bool r_all_local_ready(RCtx* c, int slot) {
    RColl* coll = &c->colls[slot];
    for (int ai = 0; ai < coll->active_rank_count; ++ai) {
        int r = c->active_ranks[slot * c->rank_count + ai];
        for (int ch = 0; ch < coll->chunk_count; ++ch)
            if (!c->cells[r_cell_index(c, slot, r, ch)].local_ready) return false;
    }
    return true;
}

__device__ bool r_all_final_ready(RCtx* c, int slot) {
    RColl* coll = &c->colls[slot];
    for (int ai = 0; ai < coll->active_rank_count; ++ai) {
        int r = c->active_ranks[slot * c->rank_count + ai];
        for (int ch = 0; ch < coll->chunk_count; ++ch)
            if (!c->cells[r_cell_index(c, slot, r, ch)].final_ready) return false;
    }
    return true;
}

__device__ void r_start_rs(RCtx* c, int slot) {
    RColl* coll = &c->colls[slot];
    coll->phase = MGA_PH_REDUCE_SCATTER;
    coll->rs_started = 1;
    c->counters[2]++;
    r_emit(c, MGA_EV_PHASE_ADVANCE_RS, coll->coll_id, UINT32_MAX, UINT32_MAX,
           MGA_PK_RS, UINT32_MAX, UINT32_MAX, INT64_MIN);
    for (int ai = 0; ai < coll->active_rank_count; ++ai) {
        int r = c->active_ranks[slot * c->rank_count + ai];
        int s = r_ring_slot(c, slot, r, 0);
        int ch = s % coll->chunk_count;
        RAction a;
        a.coll_id = coll->coll_id; a.coll_slot = slot;
        a.action_kind = MGA_ACT_REDUCE_SCATTER_SEND;
        a.chunk = ch; a.phase_index = 0; a.src = -1; a.value = 0;
        r_enqueue(c, r, &a);
    }
}

__device__ void r_start_ag(RCtx* c, int slot) {
    RColl* coll = &c->colls[slot];
    coll->phase = MGA_PH_ALLGATHER;
    coll->ag_started = 1;
    c->counters[3]++;
    r_emit(c, MGA_EV_PHASE_ADVANCE_AG, coll->coll_id, UINT32_MAX, UINT32_MAX,
           MGA_PK_AG, UINT32_MAX, UINT32_MAX, INT64_MIN);
    for (int ai = 0; ai < coll->active_rank_count; ++ai) {
        int r = c->active_ranks[slot * c->rank_count + ai];
        for (int ch = 0; ch < coll->chunk_count; ++ch) {
            RCell* cell = &c->cells[r_cell_index(c, slot, r, ch)];
            if (!cell->reduced_owner_ready) continue;
            cell->final_value = cell->accum_value;
            cell->final_ready = 1;
            RAction a;
            a.coll_id = coll->coll_id; a.coll_slot = slot;
            a.action_kind = MGA_ACT_ALLGATHER_SEND;
            a.chunk = ch; a.phase_index = 0; a.src = -1; a.value = 0;
            r_enqueue(c, r, &a);
        }
    }
}

__device__ void r_do_send(RCtx* c, int slot, int rank, RAction* a, uint8_t pk) {
    RColl* coll = &c->colls[slot];
    int dst = r_next_rank(c, slot, rank);
    int pos = c->pos_of_rank[slot * c->rank_count + rank];
    if (c->credit[slot * c->rank_count + pos] == 0) {
        c->counters[5]++;
        r_emit(c, MGA_EV_SEND_CREDIT_STALL, coll->coll_id, (uint32_t)rank, (uint32_t)dst,
               pk, (uint32_t)a->phase_index, (uint32_t)a->chunk, INT64_MIN);
        RAction re = *a;
        re.src = -1; re.value = 0;
        r_enqueue(c, rank, &re);
        return;
    }
    c->credit[slot * c->rank_count + pos] -= 1;
    int64_t val = c->cells[r_cell_index(c, slot, rank, a->chunk)].accum_value;
    if (c->max_remote > 0 && c->pending_count[0] < c->max_remote) {
        int k = c->pending_count[0];
        RRemote ev;
        ev.due_clock = c->clock + (uint64_t)c->remote_latency;
        ev.remote_seq = c->remote_seq_next++;
        ev.src = rank; ev.dst = dst;
        ev.coll_id = coll->coll_id; ev.coll_slot = slot;
        ev.phase_kind = pk;
        ev.phase_index = a->phase_index;
        ev.chunk = a->chunk;
        ev.value = val;
        ev.credit_return_src = dst;
        ev.credit_return_dst = rank;
        c->pending[k] = ev;
        c->pending_count[0] = k + 1;
    }
    c->counters[4]++;
    r_emit(c, MGA_EV_REMOTE_SEND, coll->coll_id, (uint32_t)rank, (uint32_t)dst,
           pk, (uint32_t)a->phase_index, (uint32_t)a->chunk, val);
}

__device__ void r_process_action(RCtx* c, int rank, RAction* a) {
    int slot = r_find_coll(c, a->coll_id);
    if (slot < 0 || c->colls[slot].status != MGA_ST_ACTIVE) {
        c->counters[16]++;
        r_emit(c, MGA_EV_ACTION_STALE_DROP, a->coll_id, (uint32_t)rank, UINT32_MAX,
               MGA_PK_NONE, (uint32_t)a->phase_index, (uint32_t)a->chunk, INT64_MIN);
        return;
    }
    RColl* coll = &c->colls[slot];
    int R = coll->active_rank_count;

    if (a->action_kind == MGA_ACT_LOCAL_REDUCE) {
        RCell* cell = &c->cells[r_cell_index(c, slot, rank, a->chunk)];
        if (cell->have_local) {
            cell->local_ready = 1;
            c->counters[1]++;
            r_emit(c, MGA_EV_LOCAL_READY, coll->coll_id, (uint32_t)rank, UINT32_MAX,
                   MGA_PK_LOCAL, UINT32_MAX, (uint32_t)a->chunk, INT64_MIN);
        }
        if (!coll->rs_started && r_all_local_ready(c, slot)) {
            r_start_rs(c, slot);
        }
    } else if (a->action_kind == MGA_ACT_REDUCE_SCATTER_SEND) {
        r_do_send(c, slot, rank, a, MGA_PK_RS);
    } else if (a->action_kind == MGA_ACT_ALLGATHER_SEND) {
        r_do_send(c, slot, rank, a, MGA_PK_AG);
    } else if (a->action_kind == MGA_ACT_REMOTE_RECV_REDUCE) {
        RCell* cell = &c->cells[r_cell_index(c, slot, rank, a->chunk)];
        cell->accum_value = (int64_t)((uint64_t)cell->accum_value + (uint64_t)a->value);
        cell->last_update_seq = c->event_seq;
        c->counters[10]++;
        r_emit(c, MGA_EV_REDUCE_APPLY, coll->coll_id, (uint32_t)a->src, (uint32_t)rank,
               MGA_PK_RS, (uint32_t)a->phase_index, (uint32_t)a->chunk, cell->accum_value);
        int p = a->phase_index;
        if (p + 1 < R - 1) {
            int s = r_ring_slot(c, slot, rank, p + 1);
            int ch = s % coll->chunk_count;
            RAction na;
            na.coll_id = coll->coll_id; na.coll_slot = slot;
            na.action_kind = MGA_ACT_REDUCE_SCATTER_SEND;
            na.chunk = ch; na.phase_index = p + 1; na.src = -1; na.value = 0;
            r_enqueue(c, rank, &na);
        } else {
            if (!cell->reduced_owner_ready) {
                cell->reduced_owner_ready = 1;
                coll->owners_reduced++;
                c->counters[11]++;
                r_emit(c, MGA_EV_OWNER_REDUCED, coll->coll_id, UINT32_MAX, (uint32_t)rank,
                       MGA_PK_RS, (uint32_t)a->phase_index, (uint32_t)a->chunk, cell->accum_value);
                if (!coll->ag_started && coll->owners_reduced >= R) {
                    r_start_ag(c, slot);
                }
            }
        }
    } else if (a->action_kind == MGA_ACT_REMOTE_RECV_GATHER) {
        RCell* cell = &c->cells[r_cell_index(c, slot, rank, a->chunk)];
        cell->final_value = a->value;
        cell->final_ready = 1;
        cell->last_update_seq = c->event_seq;
        c->counters[12]++;
        r_emit(c, MGA_EV_GATHER_APPLY, coll->coll_id, (uint32_t)a->src, (uint32_t)rank,
               MGA_PK_AG, (uint32_t)a->phase_index, (uint32_t)a->chunk, cell->final_value);
        int p = a->phase_index;
        if (p + 1 < R - 1) {
            RAction na;
            na.coll_id = coll->coll_id; na.coll_slot = slot;
            na.action_kind = MGA_ACT_ALLGATHER_SEND;
            na.chunk = a->chunk; na.phase_index = p + 1; na.src = -1; na.value = 0;
            r_enqueue(c, rank, &na);
        }
        if (coll->status == MGA_ST_ACTIVE && r_all_final_ready(c, slot)) {
            coll->status = MGA_ST_DONE;
            coll->phase = MGA_PH_DONE;
            c->counters[13]++;
            r_emit(c, MGA_EV_COLL_DONE, coll->coll_id, UINT32_MAX, UINT32_MAX,
                   MGA_PK_NONE, UINT32_MAX, UINT32_MAX, INT64_MIN);
        }
    }
}

// ---------- kernel ----------

__global__ void r_kernel(MgaReferenceState st, int op, int op_index,
                         int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    (void)a6; (void)a7;
    RCtx ctx; RCtx* c = &ctx;
    c->rank_count = st.rank_count; c->chunk_max = st.chunk_max;
    c->max_colls = st.max_colls; c->max_remote = st.max_remote;
    c->max_credit = st.max_credit; c->remote_latency = st.remote_latency;
    c->max_sched = st.max_sched; c->phidx_bound = st.phidx_bound;
    c->colls = st.colls; c->active_ranks = st.active_ranks; c->pos_of_rank = st.pos_of_rank;
    c->credit = st.credit; c->cells = st.cells; c->sched = st.sched;
    c->sched_count = st.sched_count; c->pending = st.pending; c->pending_count = st.pending_count;
    c->signal = st.signal; c->recv_value = st.recv_value; c->recv_src = st.recv_src;
    c->recv_seq = st.recv_seq; c->recv_valid = st.recv_valid; c->counters = st.counters;
    c->clock = st.scalars[0]; c->event_seq = st.scalars[1];
    c->coll_seq_next = st.scalars[2]; c->remote_seq_next = st.scalars[3];
    c->phase_seq_next = st.scalars[4];
    c->ev_hash = 1469598103934665603ULL;
    c->op_index = (uint32_t)op_index;

    if (op == MGA_OP_BEGIN_ALLREDUCE) {
        uint64_t coll_id = (uint64_t)(uint32_t)a0;
        uint32_t mask = (uint32_t)a1;
        int cc = a2;
        uint64_t seed = (uint64_t)(uint32_t)a3;
        bool invalid = false;
        int existing = r_find_coll(c, coll_id);
        if (existing >= 0 && c->colls[existing].status == MGA_ST_ACTIVE) invalid = true;
        if (r_popcount(c, mask) < 2) invalid = true;
        if (cc < MGA_MIN_CHUNKS || cc > c->chunk_max) invalid = true;
        int slot = -1;
        if (!invalid) {
            if (existing >= 0) slot = existing;
            else {
                for (int i = 0; i < c->max_colls; ++i)
                    if (!c->colls[i].used) { slot = i; break; }
                if (slot < 0)
                    for (int i = 0; i < c->max_colls; ++i)
                        if (c->colls[i].status != MGA_ST_ACTIVE) { slot = i; break; }
            }
            if (slot < 0) invalid = true;
        }
        if (invalid) {
            c->counters[18]++;
            r_emit(c, MGA_EV_INVALID, coll_id, UINT32_MAX, UINT32_MAX, MGA_PK_NONE,
                   UINT32_MAX, UINT32_MAX, INT64_MIN);
        } else {
            RColl* coll = &c->colls[slot];
            coll->used = 1; coll->coll_id = coll_id;
            coll->collective_seq = c->coll_seq_next++;
            coll->status = MGA_ST_ACTIVE; coll->phase = MGA_PH_LOCAL_REDUCE;
            coll->chunk_count = cc; coll->rs_started = 0; coll->ag_started = 0;
            coll->owners_reduced = 0;
            int arc = 0;
            for (int r = 0; r < c->rank_count; ++r) c->pos_of_rank[slot * c->rank_count + r] = -1;
            for (int r = 0; r < c->rank_count; ++r) {
                if (mask & (1u << r)) {
                    c->active_ranks[slot * c->rank_count + arc] = r;
                    c->pos_of_rank[slot * c->rank_count + r] = arc;
                    arc++;
                }
            }
            coll->active_rank_count = arc;
            // init every active directed ring-link credit to max.
            for (int pos = 0; pos < c->rank_count; ++pos)
                c->credit[slot * c->rank_count + pos] = (pos < arc) ? c->max_credit : 0;
            for (int r = 0; r < c->rank_count; ++r)
                for (int ch = 0; ch < c->chunk_max; ++ch) {
                    RCell* cell = &c->cells[r_cell_index(c, slot, r, ch)];
                    cell->local_value = 0; cell->accum_value = 0; cell->final_value = 0;
                    cell->last_update_seq = 0; cell->have_local = 0; cell->local_ready = 0;
                    cell->reduced_owner_ready = 0; cell->final_ready = 0;
                }
            for (int ai = 0; ai < arc; ++ai) {
                int r = c->active_ranks[slot * c->rank_count + ai];
                for (int ch = 0; ch < cc; ++ch) {
                    RCell* cell = &c->cells[r_cell_index(c, slot, r, ch)];
                    uint64_t vals[4] = { seed, coll_id, (uint64_t)r, (uint64_t)ch };
                    uint64_t hh = 1469598103934665603ULL;
                    for (int k = 0; k < 4; ++k) rfnv(&hh, &vals[k], 8);
                    cell->local_value = (int64_t)hh;
                    cell->accum_value = cell->local_value;
                    cell->have_local = 1;
                }
            }
            for (int ai = 0; ai < arc; ++ai) {
                int r = c->active_ranks[slot * c->rank_count + ai];
                for (int ch = 0; ch < cc; ++ch) {
                    RAction a;
                    a.coll_id = coll_id; a.coll_slot = slot;
                    a.action_kind = MGA_ACT_LOCAL_REDUCE;
                    a.chunk = ch; a.phase_index = 0; a.src = -1; a.value = 0;
                    r_enqueue(c, r, &a);
                }
            }
            c->counters[0]++;
            r_emit(c, MGA_EV_COLL_BEGIN, coll_id, UINT32_MAX, UINT32_MAX, MGA_PK_NONE,
                   UINT32_MAX, UINT32_MAX, INT64_MIN);
        }
    } else if (op == MGA_OP_RANK_STEP) {
        int rank = a0, action_limit = a1;
        if (rank < 0 || rank >= c->rank_count || action_limit == 0) {
            c->counters[18]++;
            r_emit(c, MGA_EV_INVALID, 0, (uint32_t)(rank >= 0 ? rank : -1), UINT32_MAX,
                   MGA_PK_NONE, UINT32_MAX, UINT32_MAX, INT64_MIN);
        } else {
            int processed = 0;
            while (processed < action_limit && c->sched_count[rank] > 0) {
                RAction a = c->sched[(size_t)rank * c->max_sched + 0];
                r_sched_pop_front(c, rank);
                r_process_action(c, rank, &a);
                ++processed;
            }
        }
    } else if (op == MGA_OP_ADVANCE) {
        int delta = a0, max_r = a1;
        if (delta < 0 || max_r < 0) {
            c->counters[18]++;
            r_emit(c, MGA_EV_INVALID, 0, UINT32_MAX, UINT32_MAX, MGA_PK_NONE,
                   UINT32_MAX, UINT32_MAX, INT64_MIN);
        } else {
            c->clock = c->clock + (uint64_t)delta;
            int processed = 0;
            while (processed < max_r) {
                int best = -1;
                int total = c->pending_count[0];
                for (int i = 0; i < total; ++i) {
                    RRemote* r = &c->pending[i];
                    if (r->due_clock > c->clock) continue;
                    if (best < 0) { best = i; continue; }
                    RRemote* b = &c->pending[best];
                    if (r->due_clock != b->due_clock) { if (r->due_clock < b->due_clock) best = i; continue; }
                    if (r->remote_seq < b->remote_seq) best = i;
                }
                if (best < 0) break;
                RRemote ev = c->pending[best];
                // remove by shifting
                for (int i = best + 1; i < total; ++i) c->pending[i - 1] = c->pending[i];
                c->pending_count[0] = total - 1;
                ++processed;

                int slot = r_find_coll(c, ev.coll_id);
                bool stale = (slot < 0) || c->colls[slot].status != MGA_ST_ACTIVE;
                if (stale) {
                    c->counters[7]++;
                    r_emit(c, MGA_EV_REMOTE_STALE_DROP, ev.coll_id, (uint32_t)ev.src,
                           (uint32_t)ev.dst, ev.phase_kind, (uint32_t)ev.phase_index,
                           (uint32_t)ev.chunk, ev.value);
                    continue;
                }
                RColl* coll = &c->colls[slot];
                int ri = r_recv_index(c, ev.dst, slot, ev.chunk, ev.phase_kind);
                c->recv_value[ri] = ev.value; c->recv_src[ri] = ev.src;
                c->recv_seq[ri] = ev.remote_seq; c->recv_valid[ri] = 1;
                int si = r_sig_index(c, ev.dst, slot, ev.phase_kind, ev.phase_index, ev.chunk);
                c->signal[si] += 1;
                c->counters[6]++;
                r_emit(c, MGA_EV_REMOTE_ARRIVE, ev.coll_id, (uint32_t)ev.src, (uint32_t)ev.dst,
                       ev.phase_kind, (uint32_t)ev.phase_index, (uint32_t)ev.chunk, ev.value);
                int rpos = c->pos_of_rank[slot * c->rank_count + ev.credit_return_src];
                if (rpos >= 0) {
                    int64_t nv = c->credit[slot * c->rank_count + rpos] + 1;
                    if (nv > c->max_credit) nv = c->max_credit;
                    c->credit[slot * c->rank_count + rpos] = nv;
                }
                c->counters[8]++;
                r_emit(c, MGA_EV_CREDIT_RETURN, ev.coll_id, (uint32_t)ev.credit_return_src,
                       (uint32_t)ev.credit_return_dst, ev.phase_kind,
                       (uint32_t)ev.phase_index, (uint32_t)ev.chunk, INT64_MIN);
                RAction a;
                a.coll_id = ev.coll_id; a.coll_slot = slot;
                a.action_kind = (ev.phase_kind == MGA_PK_RS) ? MGA_ACT_REMOTE_RECV_REDUCE
                                                             : MGA_ACT_REMOTE_RECV_GATHER;
                a.chunk = ev.chunk; a.phase_index = ev.phase_index;
                a.src = ev.src; a.value = ev.value;
                r_enqueue(c, ev.dst, &a);
                (void)coll;
            }
        }
    } else if (op == MGA_OP_POLL_SIGNAL) {
        int rank = a0;
        uint64_t coll_id = (uint64_t)(uint32_t)a1;
        int pk = a2, pidx = a3, chunk = a4;
        uint64_t target = (uint64_t)(uint32_t)a5;
        int slot = r_find_coll(c, coll_id);
        bool invalid = false;
        if (rank < 0 || rank >= c->rank_count) invalid = true;
        if (slot < 0) invalid = true;
        if (pk < 0 || pk > 2) invalid = true;
        if (pidx < 0 || pidx >= c->phidx_bound) invalid = true;
        if (chunk < 0 || chunk >= c->chunk_max) invalid = true;
        if (invalid) {
            c->counters[18]++;
            r_emit(c, MGA_EV_INVALID, coll_id, (uint32_t)(rank >= 0 ? rank : -1), UINT32_MAX,
                   (uint8_t)((pk >= 0 && pk <= 2) ? pk : MGA_PK_NONE),
                   (uint32_t)pidx, (uint32_t)chunk, INT64_MIN);
        } else {
            int si = r_sig_index(c, rank, slot, pk, pidx, chunk);
            uint64_t cur = c->signal[si];
            if (cur >= target) {
                c->counters[14]++;
                r_emit(c, MGA_EV_SIGNAL_READY, coll_id, (uint32_t)rank, UINT32_MAX,
                       (uint8_t)pk, (uint32_t)pidx, (uint32_t)chunk, (int64_t)cur);
            } else {
                c->counters[15]++;
                r_emit(c, MGA_EV_SIGNAL_WAIT, coll_id, (uint32_t)rank, UINT32_MAX,
                       (uint8_t)pk, (uint32_t)pidx, (uint32_t)chunk, (int64_t)cur);
            }
        }
    } else if (op == MGA_OP_CANCEL_COLL) {
        uint64_t coll_id = (uint64_t)(uint32_t)a0;
        int slot = r_find_coll(c, coll_id);
        if (slot < 0 || c->colls[slot].status != MGA_ST_ACTIVE) {
            c->counters[18]++;
            r_emit(c, MGA_EV_INVALID, coll_id, UINT32_MAX, UINT32_MAX, MGA_PK_NONE,
                   UINT32_MAX, UINT32_MAX, INT64_MIN);
        } else {
            c->colls[slot].status = MGA_ST_CANCELLED;
            c->counters[17]++;
            r_emit(c, MGA_EV_COLL_CANCEL, coll_id, UINT32_MAX, UINT32_MAX, MGA_PK_NONE,
                   UINT32_MAX, UINT32_MAX, INT64_MIN);
        }
    } else if (op == MGA_OP_FORCE_CREDIT) {
        int src = a0, dst = a1, amount = a2;
        bool invalid = false;
        if (src < 0 || src >= c->rank_count || dst < 0 || dst >= c->rank_count || src == dst)
            invalid = true;
        bool any_edge = false;
        if (!invalid) {
            for (int i = 0; i < c->max_colls; ++i) {
                RColl* coll = &c->colls[i];
                if (!coll->used || coll->status != MGA_ST_ACTIVE) continue;
                int pos = c->pos_of_rank[i * c->rank_count + src];
                if (pos < 0) continue;
                if (r_next_rank(c, i, src) == dst) any_edge = true;
            }
            if (!any_edge) invalid = true;
        }
        if (invalid) {
            c->counters[18]++;
            r_emit(c, MGA_EV_INVALID, 0, (uint32_t)(src >= 0 ? src : -1),
                   (uint32_t)(dst >= 0 ? dst : -1), MGA_PK_NONE, UINT32_MAX,
                   UINT32_MAX, INT64_MIN);
        } else {
            for (int i = 0; i < c->max_colls; ++i) {
                RColl* coll = &c->colls[i];
                if (!coll->used || coll->status != MGA_ST_ACTIVE) continue;
                int pos = c->pos_of_rank[i * c->rank_count + src];
                if (pos < 0) continue;
                if (r_next_rank(c, i, src) != dst) continue;
                int64_t nv = c->credit[i * c->rank_count + pos] + (int64_t)amount;
                if (nv > c->max_credit) nv = c->max_credit;
                if (nv < 0) nv = 0;
                c->credit[i * c->rank_count + pos] = nv;
            }
            c->counters[9]++;
            r_emit(c, MGA_EV_CREDIT_FORCE, 0, (uint32_t)src, (uint32_t)dst, MGA_PK_NONE,
                   UINT32_MAX, UINT32_MAX, (int64_t)amount);
        }
    } else {
        c->counters[18]++;
        r_emit(c, MGA_EV_INVALID, 0, UINT32_MAX, UINT32_MAX, MGA_PK_NONE,
               UINT32_MAX, UINT32_MAX, INT64_MIN);
    }

    st.scalars[0] = c->clock;
    st.scalars[1] = c->event_seq;
    st.scalars[2] = c->coll_seq_next;
    st.scalars[3] = c->remote_seq_next;
    st.scalars[4] = c->phase_seq_next;
    st.scalars[5] = c->ev_hash;
}

// ---------- hashing kernel ----------

__global__ void r_hash_kernel(MgaReferenceState st,
                              int64_t* out_counters, uint64_t* out_evhash,
                              uint64_t* out_chash, uint64_t* out_shash,
                              uint64_t* out_crhash, uint64_t* out_phash,
                              uint64_t* out_schhash, uint64_t* out_clock,
                              uint64_t* out_evseq) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    RCtx ctx; RCtx* c = &ctx;
    c->rank_count = st.rank_count; c->chunk_max = st.chunk_max; c->max_colls = st.max_colls;
    c->max_remote = st.max_remote; c->phidx_bound = st.phidx_bound;
    c->colls = st.colls; c->active_ranks = st.active_ranks; c->pos_of_rank = st.pos_of_rank;
    c->credit = st.credit; c->cells = st.cells; c->sched = st.sched;
    c->sched_count = st.sched_count; c->pending = st.pending; c->pending_count = st.pending_count;
    c->signal = st.signal;

    for (int i = 0; i < MGA_COUNTER_COUNT; ++i) out_counters[i] = st.counters[i];
    out_evhash[0] = st.scalars[5];
    out_clock[0] = st.scalars[0];
    out_evseq[0] = st.scalars[1];

    // collective hash: used colls by coll_id asc, then rank, then chunk.
    uint64_t h = 1469598103934665603ULL;
    // selection over used colls by coll_id
    int done_mask = 0;  // up to 8 colls bound
    for (int it = 0; it < c->max_colls; ++it) {
        int best = -1;
        for (int i = 0; i < c->max_colls; ++i) {
            if (!c->colls[i].used) continue;
            if (done_mask & (1 << i)) continue;
            if (best < 0 || c->colls[i].coll_id < c->colls[best].coll_id) best = i;
        }
        if (best < 0) break;
        done_mask |= (1 << best);
        RColl* coll = &c->colls[best];
        for (int ai = 0; ai < coll->active_rank_count; ++ai) {
            int r = c->active_ranks[best * c->rank_count + ai];
            for (int ch = 0; ch < coll->chunk_count; ++ch) {
                RCell* cell = &c->cells[r_cell_index(c, best, r, ch)];
                uint64_t cid = coll->coll_id, cseq = coll->collective_seq;
                uint8_t stt = coll->status, ph = coll->phase;
                uint32_t ru = (uint32_t)r, chu = (uint32_t)ch;
                rfnv(&h, &cid, 8); rfnv(&h, &cseq, 8);
                rfnv(&h, &stt, 1); rfnv(&h, &ph, 1);
                rfnv(&h, &ru, 4); rfnv(&h, &chu, 4);
                rfnv(&h, &cell->local_value, 8);
                rfnv(&h, &cell->accum_value, 8);
                rfnv(&h, &cell->final_value, 8);
                rfnv(&h, &cell->reduced_owner_ready, 1);
                rfnv(&h, &cell->final_ready, 1);
                rfnv(&h, &cell->last_update_seq, 8);
            }
        }
    }
    out_chash[0] = h;

    // signal hash
    h = 1469598103934665603ULL;
    for (int rank = 0; rank < c->rank_count; ++rank) {
        for (int slot = 0; slot < c->max_colls; ++slot) {
            if (!c->colls[slot].used) continue;
            uint64_t cid = c->colls[slot].coll_id;
            for (int pk = 0; pk < 3; ++pk) {
                for (int pidx = 0; pidx < c->phidx_bound; ++pidx) {
                    for (int ch = 0; ch < c->chunk_max; ++ch) {
                        uint64_t v = c->signal[r_sig_index(c, rank, slot, pk, pidx, ch)];
                        if (v == 0) continue;
                        uint32_t ru = (uint32_t)rank, pidu = (uint32_t)pidx, chu = (uint32_t)ch;
                        uint8_t pku = (uint8_t)pk;
                        rfnv(&h, &ru, 4); rfnv(&h, &cid, 8); rfnv(&h, &pku, 1);
                        rfnv(&h, &pidu, 4); rfnv(&h, &chu, 4); rfnv(&h, &v, 8);
                    }
                }
            }
        }
    }
    out_shash[0] = h;

    // credit hash: used colls by coll_id asc, edges pos->next.
    h = 1469598103934665603ULL;
    done_mask = 0;
    for (int it = 0; it < c->max_colls; ++it) {
        int best = -1;
        for (int i = 0; i < c->max_colls; ++i) {
            if (!c->colls[i].used) continue;
            if (done_mask & (1 << i)) continue;
            if (best < 0 || c->colls[i].coll_id < c->colls[best].coll_id) best = i;
        }
        if (best < 0) break;
        done_mask |= (1 << best);
        RColl* coll = &c->colls[best];
        int arc = coll->active_rank_count;
        for (int pos = 0; pos < arc; ++pos) {
            int src = c->active_ranks[best * c->rank_count + pos];
            int dst = c->active_ranks[best * c->rank_count + ((pos + 1) % arc)];
            uint64_t cid = coll->coll_id;
            uint32_t su = (uint32_t)src, du = (uint32_t)dst;
            uint64_t cr = (uint64_t)c->credit[best * c->rank_count + pos];
            rfnv(&h, &cid, 8); rfnv(&h, &su, 4); rfnv(&h, &du, 4); rfnv(&h, &cr, 8);
        }
    }
    out_crhash[0] = h;

    // pending hash: canonical order (due_clock, remote_seq), selection sort.
    int total = c->pending_count[0];
    h = 1469598103934665603ULL;
    // use a visited array stored in pad? We instead mark via a local bitmap is too
    // small (total up to 4096). Use a transient field: reuse credit_return_dst sign?
    // Safer: selection with a per-record visited flag stored in pad[0].
    for (int i = 0; i < total; ++i) c->pending[i].pad[0] = 0;
    for (;;) {
        int best = -1;
        for (int i = 0; i < total; ++i) {
            RRemote* r = &c->pending[i];
            if (r->pad[0]) continue;
            if (best < 0) { best = i; continue; }
            RRemote* b = &c->pending[best];
            if (r->due_clock != b->due_clock) { if (r->due_clock < b->due_clock) best = i; continue; }
            if (r->remote_seq < b->remote_seq) best = i;
        }
        if (best < 0) break;
        RRemote* e = &c->pending[best];
        e->pad[0] = 1;
        uint32_t su = (uint32_t)e->src, du = (uint32_t)e->dst;
        uint64_t cid = e->coll_id;
        uint8_t pk = e->phase_kind;
        uint32_t pidu = (uint32_t)e->phase_index, chu = (uint32_t)e->chunk;
        rfnv(&h, &su, 4); rfnv(&h, &du, 4); rfnv(&h, &cid, 8);
        rfnv(&h, &pk, 1); rfnv(&h, &pidu, 4); rfnv(&h, &chu, 4);
        rfnv(&h, &e->value, 8); rfnv(&h, &e->due_clock, 8); rfnv(&h, &e->remote_seq, 8);
    }
    out_phash[0] = h;

    // scheduler hash: per-rank append order.
    h = 1469598103934665603ULL;
    for (int rank = 0; rank < c->rank_count; ++rank) {
        int cnt = c->sched_count[rank];
        RAction* base = c->sched + (size_t)rank * st.max_sched;
        for (int pos = 0; pos < cnt; ++pos) {
            RAction* a = &base[pos];
            uint32_t ru = (uint32_t)rank, posu = (uint32_t)pos;
            uint64_t cid = a->coll_id;
            uint8_t ak = a->action_kind;
            uint32_t chu = (uint32_t)a->chunk, pidu = (uint32_t)a->phase_index;
            uint64_t ps = a->phase_seq;
            rfnv(&h, &ru, 4); rfnv(&h, &posu, 4); rfnv(&h, &cid, 8);
            rfnv(&h, &ak, 1); rfnv(&h, &chu, 4); rfnv(&h, &pidu, 4); rfnv(&h, &ps, 8);
        }
    }
    out_schhash[0] = h;
}

// ---------- host ABI ----------

static cudaError_t r_reset(MgaReferenceState* st, cudaStream_t s) {
    cudaError_t e;
    size_t ncells = (size_t)st->max_colls * st->rank_count * st->chunk_max;
    size_t nsig = (size_t)st->rank_count * st->max_colls * 3 * st->phidx_bound * st->chunk_max;
    size_t nrecv = (size_t)st->rank_count * st->max_colls * st->chunk_max * 3;
    e = cudaMemsetAsync(st->colls, 0, sizeof(RColl) * st->max_colls, s); if (e) return e;
    e = cudaMemsetAsync(st->cells, 0, sizeof(RCell) * ncells, s); if (e) return e;
    e = cudaMemsetAsync(st->sched_count, 0, sizeof(int32_t) * st->rank_count, s); if (e) return e;
    e = cudaMemsetAsync(st->pending_count, 0, sizeof(int32_t), s); if (e) return e;
    e = cudaMemsetAsync(st->signal, 0, sizeof(uint64_t) * nsig, s); if (e) return e;
    e = cudaMemsetAsync(st->recv_value, 0, sizeof(int64_t) * nrecv, s); if (e) return e;
    e = cudaMemsetAsync(st->recv_src, 0, sizeof(int32_t) * nrecv, s); if (e) return e;
    e = cudaMemsetAsync(st->recv_seq, 0, sizeof(uint64_t) * nrecv, s); if (e) return e;
    e = cudaMemsetAsync(st->recv_valid, 0, sizeof(uint8_t) * nrecv, s); if (e) return e;
    e = cudaMemsetAsync(st->counters, 0, sizeof(int64_t) * MGA_COUNTER_COUNT, s); if (e) return e;
    e = cudaMemsetAsync(st->credit, 0, sizeof(int64_t) * st->max_colls * st->rank_count, s); if (e) return e;
    uint64_t sc[6] = {0, 0, 1, 1, 1, 0};
    e = cudaMemcpyAsync(st->scalars, sc, sizeof(sc), cudaMemcpyHostToDevice, s); if (e) return e;
    return cudaStreamSynchronize(s);
}

extern "C" size_t solution_workspace_bytes(const MgaProblemSpec* spec) {
    if (!mga_validate_problem_spec(spec)) return 0;
    return 256;
}

extern "C" cudaError_t solution_init(const MgaProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!mga_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    MgaReferenceState* st = (MgaReferenceState*)malloc(sizeof(MgaReferenceState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(*st));
    memcpy(&st->spec, spec, sizeof(MgaProblemSpec));
    st->rank_count = spec->rank_count;
    st->chunk_max = spec->chunk_count_max;
    st->max_colls = spec->max_collectives;
    st->max_remote = spec->max_remote_events;
    st->max_credit = spec->max_send_credits_per_link;
    st->remote_latency = spec->remote_latency;
    st->max_sched = spec->max_scheduler_queue_per_rank;
    st->phidx_bound = spec->rank_count;

    size_t ncells = (size_t)st->max_colls * st->rank_count * st->chunk_max;
    size_t nsig = (size_t)st->rank_count * st->max_colls * 3 * st->phidx_bound * st->chunk_max;
    size_t nrecv = (size_t)st->rank_count * st->max_colls * st->chunk_max * 3;

    cudaError_t e = cudaSuccess;
    e = cudaMalloc(&st->colls, sizeof(RColl) * st->max_colls); if (e) goto fail;
    e = cudaMalloc(&st->active_ranks, sizeof(int32_t) * st->max_colls * st->rank_count); if (e) goto fail;
    e = cudaMalloc(&st->pos_of_rank, sizeof(int32_t) * st->max_colls * st->rank_count); if (e) goto fail;
    e = cudaMalloc(&st->credit, sizeof(int64_t) * st->max_colls * st->rank_count); if (e) goto fail;
    e = cudaMalloc(&st->cells, sizeof(RCell) * ncells); if (e) goto fail;
    e = cudaMalloc(&st->sched, sizeof(RAction) * (size_t)st->rank_count * st->max_sched); if (e) goto fail;
    e = cudaMalloc(&st->sched_count, sizeof(int32_t) * st->rank_count); if (e) goto fail;
    e = cudaMalloc(&st->pending, sizeof(RRemote) * st->max_remote); if (e) goto fail;
    e = cudaMalloc(&st->pending_count, sizeof(int32_t)); if (e) goto fail;
    e = cudaMalloc(&st->signal, sizeof(uint64_t) * nsig); if (e) goto fail;
    e = cudaMalloc(&st->recv_value, sizeof(int64_t) * nrecv); if (e) goto fail;
    e = cudaMalloc(&st->recv_src, sizeof(int32_t) * nrecv); if (e) goto fail;
    e = cudaMalloc(&st->recv_seq, sizeof(uint64_t) * nrecv); if (e) goto fail;
    e = cudaMalloc(&st->recv_valid, sizeof(uint8_t) * nrecv); if (e) goto fail;
    e = cudaMalloc(&st->counters, sizeof(int64_t) * MGA_COUNTER_COUNT); if (e) goto fail;
    e = cudaMalloc(&st->scalars, sizeof(uint64_t) * 6); if (e) goto fail;

    e = r_reset(st, stream); if (e) goto fail;
    *state_out = st;
    return cudaSuccess;
fail:
    solution_destroy(st);
    return e;
}

extern "C" cudaError_t solution_run(void* state, const MgaRunSpec* run, const void* inputs,
                                    void* outputs, void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    (void)inputs; (void)workspace;
    if (!state || !mga_validate_run_spec(run) || !outputs) return cudaErrorInvalidValue;
    if (workspace_bytes < 1) return cudaErrorInvalidValue;
    MgaReferenceState* st = (MgaReferenceState*)state;
    MgaOutputs* out = (MgaOutputs*)outputs;
    if (!out->counters || !out->remote_event_hash || !out->collective_hash ||
        !out->signal_hash || !out->credit_hash || !out->pending_remote_hash ||
        !out->scheduler_hash || !out->clock_out || !out->event_seq_out)
        return cudaErrorInvalidValue;

    r_kernel<<<1, 1, 0, stream>>>(*st, run->op, run->op_index, run->a0, run->a1, run->a2,
                                  run->a3, run->a4, run->a5, run->a6, run->a7);
    cudaError_t e = cudaPeekAtLastError(); if (e) return e;

    r_hash_kernel<<<1, 1, 0, stream>>>(*st, out->counters, out->remote_event_hash,
                                       out->collective_hash, out->signal_hash, out->credit_hash,
                                       out->pending_remote_hash, out->scheduler_hash,
                                       out->clock_out, out->event_seq_out);
    e = cudaPeekAtLastError(); if (e) return e;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return r_reset((MgaReferenceState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    MgaReferenceState* st = (MgaReferenceState*)state;
    if (st->colls) cudaFree(st->colls);
    if (st->active_ranks) cudaFree(st->active_ranks);
    if (st->pos_of_rank) cudaFree(st->pos_of_rank);
    if (st->credit) cudaFree(st->credit);
    if (st->cells) cudaFree(st->cells);
    if (st->sched) cudaFree(st->sched);
    if (st->sched_count) cudaFree(st->sched_count);
    if (st->pending) cudaFree(st->pending);
    if (st->pending_count) cudaFree(st->pending_count);
    if (st->signal) cudaFree(st->signal);
    if (st->recv_value) cudaFree(st->recv_value);
    if (st->recv_src) cudaFree(st->recv_src);
    if (st->recv_seq) cudaFree(st->recv_seq);
    if (st->recv_valid) cudaFree(st->recv_valid);
    if (st->counters) cudaFree(st->counters);
    if (st->scalars) cudaFree(st->scalars);
    free(st);
}
