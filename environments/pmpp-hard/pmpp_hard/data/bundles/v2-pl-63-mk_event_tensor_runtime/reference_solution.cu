// PMPP_CANARY_63_a5bb72d8bd -- held-out canary; MUST NOT appear in any submission
// file: mk_event_tensor_runtime_reference.cu
//
// Reference GPU implementation of MK4 Event-Tensor Pop/Push Runtime. A single-
// threaded device kernel keeps the full persistent engine state in flat device
// arrays (struct-of-arrays): tensor table, a dense cell table indexed by
// tensor*max_cells_per_tensor + linear_cell_index, a slotted task table with
// per-task dep/output sub-arrays, and per-scheduler ready/popped ring buffers.
// Reverse-dependency order (task_seq) and ascending orderings are realized by
// explicit linear scans / selection. Persistent state lives across steps. Shares
// no algorithm code with the std::map oracle or the array-of-structs naive impl.

#include "mk_event_tensor_runtime_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---- FNV helpers ----
__device__ __forceinline__ uint64_t rf_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= MK_FNV_PRIME; return h;
}
__device__ void rf_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*)p; uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = rf_byte(v, q[i]);
    *h = v;
}
__device__ __forceinline__ void rf_u8(uint64_t* h, uint8_t v)   { rf_bytes(h, &v, 1); }
__device__ __forceinline__ void rf_u32(uint64_t* h, uint32_t v) { rf_bytes(h, &v, 4); }
__device__ __forceinline__ void rf_u64(uint64_t* h, uint64_t v) { rf_bytes(h, &v, 8); }

// counter indices
#define C_ES   0
#define C_TSK  1
#define C_CELL 2
#define C_RDY  3
#define C_POP  4
#define C_OPI  5

// count indices (output order)
#define K_TD   0
#define K_DS   1
#define K_TRR  2
#define K_TBR  3
#define K_CD   4
#define K_CZ   5
#define K_CAZ  6
#define K_TRP  7
#define K_TP   8
#define K_TS   9
#define K_TC   10
#define K_DC   11
#define K_CC   12
#define K_TCBC 13
#define K_TCE  14
#define K_RSD  15
#define K_PSD  16
#define K_INV  17

struct RefCounts { int32_t v[18]; };

struct MkRefState {
    MkProblemSpec spec;

    uint64_t* counters;  // 6

    // Tensor table (capacity tensor_count).
    uint8_t*  t_defined;
    uint8_t*  t_rank;
    uint32_t* t_dim0;
    uint32_t* t_dim1;
    uint64_t* t_generation;

    // Cell table: tensor_count * max_cells_per_tensor entries.
    uint64_t* c_initial;
    uint64_t* c_remaining;
    uint8_t*  c_zero_pushed;
    uint64_t* c_cell_seq;
    uint64_t* c_last_dec_seq;
    uint8_t*  c_cancelled;

    // Task table (capacity max_tasks, slotted).
    uint8_t*  tk_used;
    uint64_t* tk_id;
    uint64_t* tk_seq;
    uint32_t* tk_sched;
    uint8_t*  tk_status;
    uint64_t* tk_mask;
    uint64_t* tk_payload;
    uint64_t* tk_ready_seq;
    uint64_t* tk_pop_seq;
    uint32_t* tk_dep_count;
    uint32_t* tk_out_count;
    // dep/output sub-arrays: max_tasks * max_task_deps / max_task_outputs.
    uint32_t* tk_dep_tensor;
    uint32_t* tk_dep_i;
    uint32_t* tk_dep_j;
    uint32_t* tk_out_tensor;
    uint32_t* tk_out_i;
    uint32_t* tk_out_j;
    uint64_t* tk_out_amount;

    // Ready/popped ring buffers per scheduler.
    int32_t*  rq_head;   // per scheduler
    int32_t*  rq_count;  // per scheduler
    uint64_t* rq_task;   // [sched * max_ready + slot]
    uint64_t* rq_seq;
    int32_t*  pq_head;
    int32_t*  pq_count;
    uint64_t* pq_task;
    uint64_t* pq_seq;
};

struct MkRefDev {
    int32_t tensor_count, max_cells, max_tasks, max_deps, max_outs;
    int32_t scheduler_count, max_ready, max_popped;

    uint64_t* counters;
    uint8_t* t_defined; uint8_t* t_rank; uint32_t* t_dim0; uint32_t* t_dim1; uint64_t* t_generation;
    uint64_t* c_initial; uint64_t* c_remaining; uint8_t* c_zero_pushed;
    uint64_t* c_cell_seq; uint64_t* c_last_dec_seq; uint8_t* c_cancelled;
    uint8_t* tk_used; uint64_t* tk_id; uint64_t* tk_seq; uint32_t* tk_sched;
    uint8_t* tk_status; uint64_t* tk_mask; uint64_t* tk_payload;
    uint64_t* tk_ready_seq; uint64_t* tk_pop_seq;
    uint32_t* tk_dep_count; uint32_t* tk_out_count;
    uint32_t* tk_dep_tensor; uint32_t* tk_dep_i; uint32_t* tk_dep_j;
    uint32_t* tk_out_tensor; uint32_t* tk_out_i; uint32_t* tk_out_j; uint64_t* tk_out_amount;
    int32_t* rq_head; int32_t* rq_count; uint64_t* rq_task; uint64_t* rq_seq;
    int32_t* pq_head; int32_t* pq_count; uint64_t* pq_task; uint64_t* pq_seq;
};

// ---- cell helpers ----
__device__ __forceinline__ uint32_t rd_d1(const MkRefDev& s, uint32_t t) {
    uint32_t d1 = s.t_dim1[t];
    return d1 > 0 ? d1 : 1u;
}
// Linear cell index inside tensor; -1 if out of range. Returns global cell slot.
__device__ int rd_cell_slot(const MkRefDev& s, uint32_t t, uint32_t i, uint32_t j) {
    if (t >= (uint32_t)s.tensor_count) return -1;
    if (!s.t_defined[t]) return -1;
    uint32_t d1 = rd_d1(s, t);
    if (i >= s.t_dim0[t]) return -1;
    if (j >= d1) return -1;
    uint64_t idx = (uint64_t)i * (uint64_t)d1 + (uint64_t)j;
    if (idx >= (uint64_t)s.max_cells) return -1;  // safety
    return (int)((uint64_t)t * (uint64_t)s.max_cells + idx);
}

__device__ int rd_find_task(const MkRefDev& s, uint64_t task_id) {
    for (int i = 0; i < s.max_tasks; ++i)
        if (s.tk_used[i] && s.tk_id[i] == task_id) return i;
    return -1;
}
__device__ int rd_task_count(const MkRefDev& s) {
    int c = 0;
    for (int i = 0; i < s.max_tasks; ++i) if (s.tk_used[i]) ++c;
    return c;
}

// Are all dependency cells of task-slot ti at zero & not cancelled?
__device__ bool rd_all_deps_zero(const MkRefDev& s, int ti) {
    uint32_t dc = s.tk_dep_count[ti];
    const int base = ti * s.max_deps;
    for (uint32_t k = 0; k < dc; ++k) {
        uint32_t t = s.tk_dep_tensor[base + k];
        uint32_t i = s.tk_dep_i[base + k];
        uint32_t j = s.tk_dep_j[base + k];
        int cs = rd_cell_slot(s, t, i, j);
        if (cs < 0) return false;
        if (s.c_cancelled[cs]) return false;
        if (s.c_remaining[cs] != 0) return false;
    }
    return true;
}

// Does task-slot ti depend on cell (t,i,j)?
__device__ bool rd_task_depends_on(const MkRefDev& s, int ti, uint32_t t, uint32_t i, uint32_t j) {
    uint32_t dc = s.tk_dep_count[ti];
    const int base = ti * s.max_deps;
    for (uint32_t k = 0; k < dc; ++k) {
        if (s.tk_dep_tensor[base + k] == t && s.tk_dep_i[base + k] == i &&
            s.tk_dep_j[base + k] == j) return true;
    }
    return false;
}

__device__ void rd_emit(const MkRefDev& s, uint64_t* eh, uint8_t kind, uint32_t op_idx,
        uint32_t tensor, uint32_t i, uint32_t j, uint64_t task_id, uint32_t sched,
        uint64_t count) {
    uint64_t seq = s.counters[C_ES];
    rf_u8(eh, kind);
    rf_u64(eh, seq);
    rf_u32(eh, op_idx);
    rf_u32(eh, tensor);
    rf_u32(eh, i);
    rf_u32(eh, j);
    rf_u64(eh, task_id);
    rf_u32(eh, sched);
    rf_u64(eh, count);
    s.counters[C_ES] = seq + 1;
}

__device__ void rd_emit_invalid(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t task_id) {
    rd_emit(s, eh, MK_EVT_INVALID, op_idx, MK_U32_MAX, MK_U32_MAX, MK_U32_MAX,
            task_id, MK_U32_MAX, MK_U64_MAX);
    c->v[K_INV] += 1;
}

// Find next task in ascending task_seq order with seq > cur (or smallest if !have).
// Returns slot or -1.
__device__ int rd_next_task_by_seq(const MkRefDev& s, bool have, uint64_t cur) {
    int sel = -1; uint64_t best = 0;
    for (int i = 0; i < s.max_tasks; ++i) {
        if (!s.tk_used[i]) continue;
        uint64_t sq = s.tk_seq[i];
        if (have && sq <= cur) continue;
        if (sel < 0 || sq < best) { sel = i; best = sq; }
    }
    return sel;
}

// Scan dependents of a freshly-zeroed cell (t,i,j) in task_seq order; push ready.
__device__ void rd_scan_and_push(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint32_t t, uint32_t i, uint32_t j) {
    bool have = false; uint64_t cur = 0;
    for (;;) {
        int ti = rd_next_task_by_seq(s, have, cur);
        if (ti < 0) break;
        have = true; cur = s.tk_seq[ti];
        if (!rd_task_depends_on(s, ti, t, i, j)) continue;
        if (s.tk_status[ti] != MK_ST_BLOCKED) continue;
        if (!rd_all_deps_zero(s, ti)) continue;
        uint64_t rs = s.counters[C_RDY]; s.counters[C_RDY] = rs + 1;
        s.tk_status[ti] = MK_ST_READY;
        s.tk_ready_seq[ti] = rs;
        s.tk_pop_seq[ti] = 0;
        uint32_t sc = s.tk_sched[ti];
        int hcnt = s.rq_count[sc];
        int hslot = (s.rq_head[sc] + hcnt) % s.max_ready;
        const int rb = sc * s.max_ready;
        s.rq_task[rb + hslot] = s.tk_id[ti];
        s.rq_seq[rb + hslot] = rs;
        s.rq_count[sc] = hcnt + 1;
        rd_emit(s, eh, MK_EVT_TASK_READY_PUSH, op_idx, MK_U32_MAX, MK_U32_MAX,
                MK_U32_MAX, s.tk_id[ti], sc, rs);
        c->v[K_TRP] += 1;
    }
}

// Apply one decrement to existing, non-cancelled cell-slot cs at coords (t,i,j).
__device__ void rd_apply_decrement(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, int cs, uint32_t t, uint32_t i, uint32_t j, uint64_t amount,
        uint8_t event_kind, uint64_t task_for_event) {
    bool was_zero = (s.c_remaining[cs] == 0);
    if (amount >= s.c_remaining[cs]) s.c_remaining[cs] = 0;
    else s.c_remaining[cs] -= amount;
    s.c_last_dec_seq[cs] = s.counters[C_ES];
    rd_emit(s, eh, event_kind, op_idx, t, i, j, task_for_event, MK_U32_MAX,
            s.c_remaining[cs]);
    c->v[K_CD] += 1;

    if (s.c_remaining[cs] == 0 && !was_zero && s.c_zero_pushed[cs] == 0) {
        s.c_zero_pushed[cs] = 1;
        rd_emit(s, eh, MK_EVT_CELL_ZERO, op_idx, t, i, j, 0, MK_U32_MAX, 0);
        c->v[K_CZ] += 1;
        rd_scan_and_push(s, eh, c, op_idx, t, i, j);
    } else if (was_zero) {
        rd_emit(s, eh, MK_EVT_CELL_ALREADY_ZERO, op_idx, t, i, j, 0, MK_U32_MAX, 0);
        c->v[K_CAZ] += 1;
    }
}

// ---- op handlers ----
__device__ void rd_define_tensor(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint32_t t, uint32_t rank, uint32_t dim0, uint32_t dim1,
        uint64_t initial_count) {
    if (t >= (uint32_t)s.tensor_count) { rd_emit_invalid(s, eh, c, op_idx, 0); return; }
    if (rank != 1 && rank != 2) { rd_emit_invalid(s, eh, c, op_idx, 0); return; }
    if (dim0 == 0) { rd_emit_invalid(s, eh, c, op_idx, 0); return; }
    if (rank == 2 && dim1 == 0) { rd_emit_invalid(s, eh, c, op_idx, 0); return; }
    uint32_t d1 = (rank == 2) ? dim1 : 1u;
    uint64_t total = (uint64_t)dim0 * (uint64_t)d1;
    if (total > (uint64_t)s.max_cells) { rd_emit_invalid(s, eh, c, op_idx, 0); return; }

    if (s.t_defined[t]) {
        // Stall if any cell of t has a dependent nonterminal task.
        uint32_t od1 = rd_d1(s, t);
        uint64_t ncells = (uint64_t)s.t_dim0[t] * (uint64_t)od1;
        bool stall = false;
        for (int ti = 0; ti < s.max_tasks && !stall; ++ti) {
            if (!s.tk_used[ti]) continue;
            uint8_t st = s.tk_status[ti];
            if (!(st == MK_ST_BLOCKED || st == MK_ST_READY || st == MK_ST_POPPED || st == MK_ST_RUNNING))
                continue;
            // does this task depend on any cell of tensor t?
            uint32_t dc = s.tk_dep_count[ti];
            const int base = ti * s.max_deps;
            for (uint32_t k = 0; k < dc; ++k) {
                if (s.tk_dep_tensor[base + k] != t) continue;
                uint32_t ci = s.tk_dep_i[base + k], cj = s.tk_dep_j[base + k];
                uint64_t lin = (uint64_t)ci * (uint64_t)od1 + (uint64_t)cj;
                if (ci < s.t_dim0[t] && cj < od1 && lin < ncells) { stall = true; break; }
            }
        }
        if (stall) {
            rd_emit(s, eh, MK_EVT_DEFINE_STALL, op_idx, t, MK_U32_MAX, MK_U32_MAX, 0,
                    MK_U32_MAX, MK_U64_MAX);
            c->v[K_DS] += 1;
            return;
        }
    }

    s.t_defined[t] = 1;
    s.t_rank[t] = (uint8_t)rank;
    s.t_dim0[t] = dim0;
    s.t_dim1[t] = (rank == 2) ? dim1 : 0u;
    s.t_generation[t] += 1;
    uint32_t d1s = (rank == 2) ? dim1 : 1u;
    uint64_t ncells = (uint64_t)dim0 * (uint64_t)d1s;
    const uint64_t cbase = (uint64_t)t * (uint64_t)s.max_cells;
    for (uint64_t k = 0; k < ncells; ++k) {
        int cs = (int)(cbase + k);
        s.c_initial[cs] = initial_count;
        s.c_remaining[cs] = initial_count;
        s.c_zero_pushed[cs] = (initial_count == 0) ? 1 : 0;
        uint64_t cseq = s.counters[C_CELL]; s.counters[C_CELL] = cseq + 1;
        s.c_cell_seq[cs] = cseq;
        s.c_last_dec_seq[cs] = 0;
        s.c_cancelled[cs] = 0;
    }
    rd_emit(s, eh, MK_EVT_TENSOR_DEFINE, op_idx, t, MK_U32_MAX, MK_U32_MAX, 0,
            MK_U32_MAX, s.t_generation[t]);
    c->v[K_TD] += 1;
}

__device__ void rd_register_task(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t task_id, uint32_t sched_hint, uint32_t dep_count,
        const uint32_t* dep_t, const uint32_t* dep_i, const uint32_t* dep_j,
        uint32_t out_count, const uint32_t* out_t, const uint32_t* out_i,
        const uint32_t* out_j, const uint64_t* out_amt, uint64_t dynamic_mask,
        uint64_t payload_seed) {
    if (rd_find_task(s, task_id) >= 0) { rd_emit_invalid(s, eh, c, op_idx, task_id); return; }
    if (rd_task_count(s) >= s.max_tasks) { rd_emit_invalid(s, eh, c, op_idx, task_id); return; }
    if (sched_hint >= (uint32_t)s.scheduler_count) { rd_emit_invalid(s, eh, c, op_idx, task_id); return; }
    if ((int)dep_count > s.max_deps) { rd_emit_invalid(s, eh, c, op_idx, task_id); return; }
    if ((int)out_count > s.max_outs) { rd_emit_invalid(s, eh, c, op_idx, task_id); return; }
    for (uint32_t k = 0; k < dep_count; ++k) {
        int cs = rd_cell_slot(s, dep_t[k], dep_i[k], dep_j[k]);
        if (cs < 0 || s.c_cancelled[cs]) { rd_emit_invalid(s, eh, c, op_idx, task_id); return; }
    }
    for (uint32_t k = 0; k < out_count; ++k) {
        int cs = rd_cell_slot(s, out_t[k], out_i[k], out_j[k]);
        if (cs < 0 || s.c_cancelled[cs]) { rd_emit_invalid(s, eh, c, op_idx, task_id); return; }
    }

    int slot = -1;
    for (int i = 0; i < s.max_tasks; ++i) if (!s.tk_used[i]) { slot = i; break; }
    uint64_t tseq = s.counters[C_TSK]; s.counters[C_TSK] = tseq + 1;
    s.tk_used[slot] = 1;
    s.tk_id[slot] = task_id;
    s.tk_seq[slot] = tseq;
    s.tk_sched[slot] = sched_hint;
    s.tk_mask[slot] = dynamic_mask;
    s.tk_payload[slot] = payload_seed;
    s.tk_ready_seq[slot] = 0;
    s.tk_pop_seq[slot] = 0;
    s.tk_dep_count[slot] = dep_count;
    s.tk_out_count[slot] = out_count;
    const int db = slot * s.max_deps;
    for (uint32_t k = 0; k < dep_count; ++k) {
        s.tk_dep_tensor[db + k] = dep_t[k]; s.tk_dep_i[db + k] = dep_i[k]; s.tk_dep_j[db + k] = dep_j[k];
    }
    const int ob = slot * s.max_outs;
    for (uint32_t k = 0; k < out_count; ++k) {
        s.tk_out_tensor[ob + k] = out_t[k]; s.tk_out_i[ob + k] = out_i[k];
        s.tk_out_j[ob + k] = out_j[k]; s.tk_out_amount[ob + k] = out_amt[k];
    }

    bool ready = rd_all_deps_zero(s, slot);
    if (ready) {
        uint64_t rs = s.counters[C_RDY]; s.counters[C_RDY] = rs + 1;
        s.tk_status[slot] = MK_ST_READY;
        s.tk_ready_seq[slot] = rs;
        int hcnt = s.rq_count[sched_hint];
        int hslot = (s.rq_head[sched_hint] + hcnt) % s.max_ready;
        const int rb = sched_hint * s.max_ready;
        s.rq_task[rb + hslot] = task_id; s.rq_seq[rb + hslot] = rs;
        s.rq_count[sched_hint] = hcnt + 1;
        rd_emit(s, eh, MK_EVT_TASK_READY_REGISTER, op_idx, MK_U32_MAX, MK_U32_MAX,
                MK_U32_MAX, task_id, sched_hint, rs);
        c->v[K_TRR] += 1;
    } else {
        s.tk_status[slot] = MK_ST_BLOCKED;
        rd_emit(s, eh, MK_EVT_TASK_BLOCKED_REGISTER, op_idx, MK_U32_MAX, MK_U32_MAX,
                MK_U32_MAX, task_id, sched_hint, MK_U64_MAX);
        c->v[K_TBR] += 1;
    }
}

__device__ void rd_signal_cell(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint32_t t, uint32_t i, uint32_t j, uint64_t amount) {
    int cs = rd_cell_slot(s, t, i, j);
    if (cs < 0 || s.c_cancelled[cs] || amount == 0) { rd_emit_invalid(s, eh, c, op_idx, 0); return; }
    rd_apply_decrement(s, eh, c, op_idx, cs, t, i, j, amount, MK_EVT_CELL_DECREMENT, 0);
}

__device__ void rd_pop_tasks(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint32_t scheduler, uint64_t limit) {
    if (scheduler >= (uint32_t)s.scheduler_count) { rd_emit_invalid(s, eh, c, op_idx, 0); return; }
    if (limit == 0) return;
    const int rb = scheduler * s.max_ready;
    uint64_t popped = 0;
    while (popped < limit && s.rq_count[scheduler] > 0) {
        int head = s.rq_head[scheduler];
        uint64_t tid = s.rq_task[rb + head];
        uint64_t eseq = s.rq_seq[rb + head];
        int ti = rd_find_task(s, tid);
        bool valid = (ti >= 0) && (s.tk_status[ti] == MK_ST_READY) && (s.tk_ready_seq[ti] == eseq);
        // pop front
        s.rq_head[scheduler] = (head + 1) % s.max_ready;
        s.rq_count[scheduler] -= 1;
        if (!valid) {
            rd_emit(s, eh, MK_EVT_READY_STALE_DROP, op_idx, MK_U32_MAX, MK_U32_MAX,
                    MK_U32_MAX, tid, scheduler, eseq);
            c->v[K_RSD] += 1;
            continue;
        }
        uint64_t ps = s.counters[C_POP]; s.counters[C_POP] = ps + 1;
        s.tk_status[ti] = MK_ST_POPPED;
        s.tk_pop_seq[ti] = ps;
        int pcnt = s.pq_count[scheduler];
        int pslot = (s.pq_head[scheduler] + pcnt) % s.max_popped;
        const int pb = scheduler * s.max_popped;
        s.pq_task[pb + pslot] = tid; s.pq_seq[pb + pslot] = ps;
        s.pq_count[scheduler] = pcnt + 1;
        rd_emit(s, eh, MK_EVT_TASK_POP, op_idx, MK_U32_MAX, MK_U32_MAX, MK_U32_MAX,
                tid, scheduler, ps);
        c->v[K_TP] += 1;
        popped += 1;
    }
}

__device__ void rd_start_popped(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint32_t scheduler, uint64_t limit) {
    if (scheduler >= (uint32_t)s.scheduler_count) { rd_emit_invalid(s, eh, c, op_idx, 0); return; }
    if (limit == 0) return;
    const int pb = scheduler * s.max_popped;
    uint64_t started = 0;
    while (started < limit && s.pq_count[scheduler] > 0) {
        int head = s.pq_head[scheduler];
        uint64_t tid = s.pq_task[pb + head];
        uint64_t eseq = s.pq_seq[pb + head];
        int ti = rd_find_task(s, tid);
        bool valid = (ti >= 0) && (s.tk_status[ti] == MK_ST_POPPED) && (s.tk_pop_seq[ti] == eseq);
        s.pq_head[scheduler] = (head + 1) % s.max_popped;
        s.pq_count[scheduler] -= 1;
        if (!valid) {
            rd_emit(s, eh, MK_EVT_POPPED_STALE_DROP, op_idx, MK_U32_MAX, MK_U32_MAX,
                    MK_U32_MAX, tid, scheduler, eseq);
            c->v[K_PSD] += 1;
            continue;
        }
        s.tk_status[ti] = MK_ST_RUNNING;
        rd_emit(s, eh, MK_EVT_TASK_START, op_idx, MK_U32_MAX, MK_U32_MAX, MK_U32_MAX,
                tid, scheduler, s.tk_pop_seq[ti]);
        c->v[K_TS] += 1;
        started += 1;
    }
}

__device__ void rd_complete_task(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t task_id, uint64_t observed_mask) {
    int ti = rd_find_task(s, task_id);
    if (ti < 0 || s.tk_status[ti] != MK_ST_RUNNING) { rd_emit_invalid(s, eh, c, op_idx, task_id); return; }
    uint64_t dm = s.tk_mask[ti];
    if ((observed_mask & dm) != dm) {
        s.tk_status[ti] = MK_ST_CANCELLED;
        rd_emit(s, eh, MK_EVT_TASK_DYNAMIC_CANCEL, op_idx, MK_U32_MAX, MK_U32_MAX,
                MK_U32_MAX, task_id, MK_U32_MAX, observed_mask);
        c->v[K_DC] += 1;
        return;
    }
    s.tk_status[ti] = MK_ST_DONE;
    rd_emit(s, eh, MK_EVT_TASK_COMPLETE, op_idx, MK_U32_MAX, MK_U32_MAX, MK_U32_MAX,
            task_id, MK_U32_MAX, MK_U64_MAX);
    c->v[K_TC] += 1;
    uint32_t oc = s.tk_out_count[ti];
    const int ob = ti * s.max_outs;
    for (uint32_t k = 0; k < oc; ++k) {
        uint32_t ot = s.tk_out_tensor[ob + k], oi = s.tk_out_i[ob + k], oj = s.tk_out_j[ob + k];
        uint64_t amt = s.tk_out_amount[ob + k];
        int cs = rd_cell_slot(s, ot, oi, oj);
        if (cs < 0 || s.c_cancelled[cs] || amt == 0) continue;  // skip silently
        rd_apply_decrement(s, eh, c, op_idx, cs, ot, oi, oj, amt, MK_EVT_OUTPUT_DECREMENT, task_id);
    }
}

__device__ void rd_cancel_cell(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint32_t t, uint32_t i, uint32_t j) {
    int cs = rd_cell_slot(s, t, i, j);
    if (cs < 0) { rd_emit_invalid(s, eh, c, op_idx, 0); return; }
    s.c_cancelled[cs] = 1;
    rd_emit(s, eh, MK_EVT_CELL_CANCEL, op_idx, t, i, j, 0, MK_U32_MAX, MK_U64_MAX);
    c->v[K_CC] += 1;
    // Cancel BLOCKED/READY dependents in task_seq order.
    bool have = false; uint64_t cur = 0;
    for (;;) {
        int ti = rd_next_task_by_seq(s, have, cur);
        if (ti < 0) break;
        have = true; cur = s.tk_seq[ti];
        if (!rd_task_depends_on(s, ti, t, i, j)) continue;
        uint8_t st = s.tk_status[ti];
        if (st == MK_ST_BLOCKED || st == MK_ST_READY) {
            s.tk_status[ti] = MK_ST_CANCELLED;
            rd_emit(s, eh, MK_EVT_TASK_CANCEL_BY_CELL, op_idx, t, i, j, s.tk_id[ti],
                    MK_U32_MAX, MK_U64_MAX);
            c->v[K_TCBC] += 1;
        }
    }
}

__device__ void rd_cancel_task(const MkRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t task_id) {
    int ti = rd_find_task(s, task_id);
    if (ti < 0) { rd_emit_invalid(s, eh, c, op_idx, task_id); return; }
    uint8_t st = s.tk_status[ti];
    if (st == MK_ST_DONE || st == MK_ST_CANCELLED) { rd_emit_invalid(s, eh, c, op_idx, task_id); return; }
    s.tk_status[ti] = MK_ST_CANCELLED;
    rd_emit(s, eh, MK_EVT_TASK_CANCEL_EXPLICIT, op_idx, MK_U32_MAX, MK_U32_MAX,
            MK_U32_MAX, task_id, MK_U32_MAX, MK_U64_MAX);
    c->v[K_TCE] += 1;
}

// ---- state hashes ----
__device__ uint64_t rd_cell_hash(const MkRefDev& s) {
    uint64_t h = MK_FNV_OFFSET;
    for (uint32_t t = 0; t < (uint32_t)s.tensor_count; ++t) {
        if (!s.t_defined[t]) continue;
        uint32_t d1 = rd_d1(s, t);
        uint64_t cbase = (uint64_t)t * (uint64_t)s.max_cells;
        for (uint32_t i = 0; i < s.t_dim0[t]; ++i) {
            for (uint32_t j = 0; j < d1; ++j) {
                int cs = (int)(cbase + (uint64_t)i * d1 + j);
                rf_u32(&h, t);
                rf_u64(&h, s.t_generation[t]);
                rf_u32(&h, i);
                rf_u32(&h, j);
                rf_u64(&h, s.c_initial[cs]);
                rf_u64(&h, s.c_remaining[cs]);
                rf_u8(&h, s.c_zero_pushed[cs]);
                rf_u64(&h, s.c_cell_seq[cs]);
                rf_u64(&h, s.c_last_dec_seq[cs]);
                rf_u8(&h, s.c_cancelled[cs]);
            }
        }
    }
    return h;
}

__device__ uint64_t rd_task_hash(const MkRefDev& s) {
    uint64_t h = MK_FNV_OFFSET;
    bool have = false; uint64_t cur = 0;
    for (;;) {
        int sel = -1; uint64_t best = 0;
        for (int i = 0; i < s.max_tasks; ++i) {
            if (!s.tk_used[i]) continue;
            uint64_t id = s.tk_id[i];
            if (have && id <= cur) continue;
            if (sel < 0 || id < best) { sel = i; best = id; }
        }
        if (sel < 0) break;
        have = true; cur = best;
        rf_u64(&h, s.tk_id[sel]);
        rf_u64(&h, s.tk_seq[sel]);
        rf_u32(&h, s.tk_sched[sel]);
        rf_u8(&h, s.tk_status[sel]);
        rf_u64(&h, s.tk_mask[sel]);
        rf_u64(&h, s.tk_payload[sel]);
        rf_u64(&h, s.tk_ready_seq[sel]);
        rf_u64(&h, s.tk_pop_seq[sel]);
        uint32_t dc = s.tk_dep_count[sel];
        const int db = sel * s.max_deps;
        for (uint32_t k = 0; k < dc; ++k) {
            rf_u32(&h, s.tk_dep_tensor[db + k]);
            rf_u32(&h, s.tk_dep_i[db + k]);
            rf_u32(&h, s.tk_dep_j[db + k]);
        }
        uint32_t oc = s.tk_out_count[sel];
        const int ob = sel * s.max_outs;
        for (uint32_t k = 0; k < oc; ++k) {
            rf_u32(&h, s.tk_out_tensor[ob + k]);
            rf_u32(&h, s.tk_out_i[ob + k]);
            rf_u32(&h, s.tk_out_j[ob + k]);
            rf_u64(&h, s.tk_out_amount[ob + k]);
        }
    }
    return h;
}

__device__ uint64_t rd_queue_hash(const MkRefDev& s) {
    uint64_t h = MK_FNV_OFFSET;
    for (uint32_t sc = 0; sc < (uint32_t)s.scheduler_count; ++sc) {
        const int rb = sc * s.max_ready;
        int cnt = s.rq_count[sc];
        for (int p = 0; p < cnt; ++p) {
            int slot = (s.rq_head[sc] + p) % s.max_ready;
            rf_u8(&h, 0);
            rf_u32(&h, sc);
            rf_u64(&h, (uint64_t)p);
            rf_u64(&h, s.rq_task[rb + slot]);
            rf_u64(&h, s.rq_seq[rb + slot]);
        }
    }
    for (uint32_t sc = 0; sc < (uint32_t)s.scheduler_count; ++sc) {
        const int pb = sc * s.max_popped;
        int cnt = s.pq_count[sc];
        for (int p = 0; p < cnt; ++p) {
            int slot = (s.pq_head[sc] + p) % s.max_popped;
            rf_u8(&h, 1);
            rf_u32(&h, sc);
            rf_u64(&h, (uint64_t)p);
            rf_u64(&h, s.pq_task[pb + slot]);
            rf_u64(&h, s.pq_seq[pb + slot]);
        }
    }
    return h;
}

__device__ uint64_t rd_reverse_dep_hash(const MkRefDev& s) {
    uint64_t h = MK_FNV_OFFSET;
    for (uint32_t t = 0; t < (uint32_t)s.tensor_count; ++t) {
        if (!s.t_defined[t]) continue;
        uint32_t d1 = rd_d1(s, t);
        for (uint32_t i = 0; i < s.t_dim0[t]; ++i) {
            for (uint32_t j = 0; j < d1; ++j) {
                // collect dependents in task_seq order; skip if none.
                // first detect any.
                bool any = false;
                for (int ti = 0; ti < s.max_tasks; ++ti) {
                    if (s.tk_used[ti] && rd_task_depends_on(s, ti, t, i, j)) { any = true; break; }
                }
                if (!any) continue;
                rf_u32(&h, t); rf_u32(&h, i); rf_u32(&h, j);
                bool have = false; uint64_t cur = 0;
                for (;;) {
                    int ti = rd_next_task_by_seq(s, have, cur);
                    if (ti < 0) break;
                    have = true; cur = s.tk_seq[ti];
                    if (rd_task_depends_on(s, ti, t, i, j)) rf_u64(&h, s.tk_id[ti]);
                }
            }
        }
    }
    return h;
}

__global__ void mk_ref_step_kernel(
        MkRefDev s, int batch_size,
        const int32_t* __restrict__ op_kind,
        const uint32_t* __restrict__ in_tensor,
        const uint32_t* __restrict__ in_ci,
        const uint32_t* __restrict__ in_cj,
        const uint32_t* __restrict__ in_rank,
        const uint32_t* __restrict__ in_dim0,
        const uint32_t* __restrict__ in_dim1,
        const uint64_t* __restrict__ in_task,
        const uint32_t* __restrict__ in_sched,
        const uint64_t* __restrict__ in_amount,
        const uint64_t* __restrict__ in_mask,
        const uint64_t* __restrict__ in_payload,
        const uint32_t* __restrict__ in_dep_off,
        const uint32_t* __restrict__ in_dep_count,
        const uint32_t* __restrict__ in_out_off,
        const uint32_t* __restrict__ in_out_count,
        const uint32_t* __restrict__ in_dep_tensor,
        const uint32_t* __restrict__ in_dep_i,
        const uint32_t* __restrict__ in_dep_j,
        const uint32_t* __restrict__ in_out_tensor,
        const uint32_t* __restrict__ in_out_i,
        const uint32_t* __restrict__ in_out_j,
        const uint64_t* __restrict__ in_out_amount,
        int32_t* __restrict__ out_counts,
        uint64_t* __restrict__ out_evt,
        uint64_t* __restrict__ out_cell,
        uint64_t* __restrict__ out_task,
        uint64_t* __restrict__ out_queue,
        uint64_t* __restrict__ out_rdep) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    RefCounts c; for (int i = 0; i < 18; ++i) c.v[i] = 0;
    uint64_t eh = MK_FNV_OFFSET;

    for (int r = 0; r < batch_size; ++r) {
        const uint32_t op_idx = (uint32_t)s.counters[C_OPI];
        const int32_t kind = op_kind[r];
        switch (kind) {
            case MK_OP_DEFINE_TENSOR:
                rd_define_tensor(s, &eh, &c, op_idx, in_tensor[r], in_rank[r],
                                 in_dim0[r], in_dim1[r], in_amount[r]);
                break;
            case MK_OP_REGISTER_TASK: {
                uint32_t doff = in_dep_off[r], dcnt = in_dep_count[r];
                uint32_t ooff = in_out_off[r], ocnt = in_out_count[r];
                rd_register_task(s, &eh, &c, op_idx, in_task[r], in_sched[r], dcnt,
                                 in_dep_tensor + doff, in_dep_i + doff, in_dep_j + doff,
                                 ocnt, in_out_tensor + ooff, in_out_i + ooff,
                                 in_out_j + ooff, in_out_amount + ooff, in_mask[r],
                                 in_payload[r]);
                break;
            }
            case MK_OP_SIGNAL_CELL:
                rd_signal_cell(s, &eh, &c, op_idx, in_tensor[r], in_ci[r], in_cj[r], in_amount[r]);
                break;
            case MK_OP_POP_TASKS:
                rd_pop_tasks(s, &eh, &c, op_idx, in_sched[r], in_amount[r]);
                break;
            case MK_OP_START_POPPED:
                rd_start_popped(s, &eh, &c, op_idx, in_sched[r], in_amount[r]);
                break;
            case MK_OP_COMPLETE_TASK:
                rd_complete_task(s, &eh, &c, op_idx, in_task[r], in_mask[r]);
                break;
            case MK_OP_CANCEL_CELL:
                rd_cancel_cell(s, &eh, &c, op_idx, in_tensor[r], in_ci[r], in_cj[r]);
                break;
            case MK_OP_CANCEL_TASK:
                rd_cancel_task(s, &eh, &c, op_idx, in_task[r]);
                break;
            default:
                rd_emit_invalid(s, &eh, &c, op_idx, in_task[r]);
                break;
        }
        s.counters[C_OPI] = (uint64_t)op_idx + 1;
    }

    for (int i = 0; i < 18; ++i) out_counts[i] = c.v[i];
    out_evt[0] = eh;
    out_cell[0] = rd_cell_hash(s);
    out_task[0] = rd_task_hash(s);
    out_queue[0] = rd_queue_hash(s);
    out_rdep[0] = rd_reverse_dep_hash(s);
}

static MkRefDev ref_make_dev(const MkRefState* st) {
    MkRefDev s;
    s.tensor_count = st->spec.tensor_count;
    s.max_cells = st->spec.max_cells_per_tensor;
    s.max_tasks = st->spec.max_tasks;
    s.max_deps = st->spec.max_task_deps;
    s.max_outs = st->spec.max_task_outputs;
    s.scheduler_count = st->spec.scheduler_count;
    s.max_ready = st->spec.max_ready_per_scheduler;
    s.max_popped = st->spec.max_popped_per_scheduler;
    s.counters = st->counters;
    s.t_defined = st->t_defined; s.t_rank = st->t_rank; s.t_dim0 = st->t_dim0;
    s.t_dim1 = st->t_dim1; s.t_generation = st->t_generation;
    s.c_initial = st->c_initial; s.c_remaining = st->c_remaining;
    s.c_zero_pushed = st->c_zero_pushed; s.c_cell_seq = st->c_cell_seq;
    s.c_last_dec_seq = st->c_last_dec_seq; s.c_cancelled = st->c_cancelled;
    s.tk_used = st->tk_used; s.tk_id = st->tk_id; s.tk_seq = st->tk_seq;
    s.tk_sched = st->tk_sched; s.tk_status = st->tk_status; s.tk_mask = st->tk_mask;
    s.tk_payload = st->tk_payload; s.tk_ready_seq = st->tk_ready_seq;
    s.tk_pop_seq = st->tk_pop_seq; s.tk_dep_count = st->tk_dep_count;
    s.tk_out_count = st->tk_out_count; s.tk_dep_tensor = st->tk_dep_tensor;
    s.tk_dep_i = st->tk_dep_i; s.tk_dep_j = st->tk_dep_j;
    s.tk_out_tensor = st->tk_out_tensor; s.tk_out_i = st->tk_out_i;
    s.tk_out_j = st->tk_out_j; s.tk_out_amount = st->tk_out_amount;
    s.rq_head = st->rq_head; s.rq_count = st->rq_count; s.rq_task = st->rq_task; s.rq_seq = st->rq_seq;
    s.pq_head = st->pq_head; s.pq_count = st->pq_count; s.pq_task = st->pq_task; s.pq_seq = st->pq_seq;
    return s;
}

__global__ void mk_ref_init_counters(uint64_t* counters) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    counters[C_ES] = 0; counters[C_TSK] = 1; counters[C_CELL] = 1;
    counters[C_RDY] = 1; counters[C_POP] = 1; counters[C_OPI] = 0;
}

static cudaError_t ref_reset_state(MkRefState* st, cudaStream_t stream) {
    cudaError_t err;
    const size_t T = (size_t)st->spec.tensor_count;
    const size_t TK = (size_t)st->spec.max_tasks;
    const size_t SC = (size_t)st->spec.scheduler_count;
    err = cudaMemsetAsync(st->t_defined, 0, sizeof(uint8_t) * T, stream); if (err) return err;
    err = cudaMemsetAsync(st->t_generation, 0, sizeof(uint64_t) * T, stream); if (err) return err;
    err = cudaMemsetAsync(st->t_dim0, 0, sizeof(uint32_t) * T, stream); if (err) return err;
    err = cudaMemsetAsync(st->t_dim1, 0, sizeof(uint32_t) * T, stream); if (err) return err;
    err = cudaMemsetAsync(st->t_rank, 0, sizeof(uint8_t) * T, stream); if (err) return err;
    err = cudaMemsetAsync(st->tk_used, 0, sizeof(uint8_t) * TK, stream); if (err) return err;
    err = cudaMemsetAsync(st->rq_head, 0, sizeof(int32_t) * SC, stream); if (err) return err;
    err = cudaMemsetAsync(st->rq_count, 0, sizeof(int32_t) * SC, stream); if (err) return err;
    err = cudaMemsetAsync(st->pq_head, 0, sizeof(int32_t) * SC, stream); if (err) return err;
    err = cudaMemsetAsync(st->pq_count, 0, sizeof(int32_t) * SC, stream); if (err) return err;
    mk_ref_init_counters<<<1, 1, 0, stream>>>(st->counters);
    return cudaPeekAtLastError();
}

extern "C" size_t solution_workspace_bytes(const MkProblemSpec* spec) {
    if (!mk_validate_problem_spec(spec)) return 0;
    return 256;  // 18 int32 counts fit
}

#define REF_MALLOC(p, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(p)), (bytes)); \
    if (err != cudaSuccess) goto fail; } while (0)

extern "C" cudaError_t solution_init(const MkProblemSpec* spec, void** state_out,
                                     cudaStream_t stream) {
    if (!mk_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    MkRefState* st = (MkRefState*)malloc(sizeof(MkRefState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(MkRefState));
    memcpy(&st->spec, spec, sizeof(MkProblemSpec));

    const size_t T = (size_t)spec->tensor_count;
    const size_t CELLS = T * (size_t)spec->max_cells_per_tensor;
    const size_t TK = (size_t)spec->max_tasks;
    const size_t DEPS = TK * (size_t)spec->max_task_deps;
    const size_t OUTS = TK * (size_t)spec->max_task_outputs;
    const size_t SC = (size_t)spec->scheduler_count;
    const size_t RQ = SC * (size_t)spec->max_ready_per_scheduler;
    const size_t PQ = SC * (size_t)spec->max_popped_per_scheduler;

    cudaError_t err = cudaSuccess;
    REF_MALLOC(st->counters, sizeof(uint64_t) * 6);
    REF_MALLOC(st->t_defined, sizeof(uint8_t) * T);
    REF_MALLOC(st->t_rank, sizeof(uint8_t) * T);
    REF_MALLOC(st->t_dim0, sizeof(uint32_t) * T);
    REF_MALLOC(st->t_dim1, sizeof(uint32_t) * T);
    REF_MALLOC(st->t_generation, sizeof(uint64_t) * T);
    REF_MALLOC(st->c_initial, sizeof(uint64_t) * CELLS);
    REF_MALLOC(st->c_remaining, sizeof(uint64_t) * CELLS);
    REF_MALLOC(st->c_zero_pushed, sizeof(uint8_t) * CELLS);
    REF_MALLOC(st->c_cell_seq, sizeof(uint64_t) * CELLS);
    REF_MALLOC(st->c_last_dec_seq, sizeof(uint64_t) * CELLS);
    REF_MALLOC(st->c_cancelled, sizeof(uint8_t) * CELLS);
    REF_MALLOC(st->tk_used, sizeof(uint8_t) * TK);
    REF_MALLOC(st->tk_id, sizeof(uint64_t) * TK);
    REF_MALLOC(st->tk_seq, sizeof(uint64_t) * TK);
    REF_MALLOC(st->tk_sched, sizeof(uint32_t) * TK);
    REF_MALLOC(st->tk_status, sizeof(uint8_t) * TK);
    REF_MALLOC(st->tk_mask, sizeof(uint64_t) * TK);
    REF_MALLOC(st->tk_payload, sizeof(uint64_t) * TK);
    REF_MALLOC(st->tk_ready_seq, sizeof(uint64_t) * TK);
    REF_MALLOC(st->tk_pop_seq, sizeof(uint64_t) * TK);
    REF_MALLOC(st->tk_dep_count, sizeof(uint32_t) * TK);
    REF_MALLOC(st->tk_out_count, sizeof(uint32_t) * TK);
    REF_MALLOC(st->tk_dep_tensor, sizeof(uint32_t) * DEPS);
    REF_MALLOC(st->tk_dep_i, sizeof(uint32_t) * DEPS);
    REF_MALLOC(st->tk_dep_j, sizeof(uint32_t) * DEPS);
    REF_MALLOC(st->tk_out_tensor, sizeof(uint32_t) * OUTS);
    REF_MALLOC(st->tk_out_i, sizeof(uint32_t) * OUTS);
    REF_MALLOC(st->tk_out_j, sizeof(uint32_t) * OUTS);
    REF_MALLOC(st->tk_out_amount, sizeof(uint64_t) * OUTS);
    REF_MALLOC(st->rq_head, sizeof(int32_t) * SC);
    REF_MALLOC(st->rq_count, sizeof(int32_t) * SC);
    REF_MALLOC(st->rq_task, sizeof(uint64_t) * RQ);
    REF_MALLOC(st->rq_seq, sizeof(uint64_t) * RQ);
    REF_MALLOC(st->pq_head, sizeof(int32_t) * SC);
    REF_MALLOC(st->pq_count, sizeof(int32_t) * SC);
    REF_MALLOC(st->pq_task, sizeof(uint64_t) * PQ);
    REF_MALLOC(st->pq_seq, sizeof(uint64_t) * PQ);

    err = ref_reset_state(st, stream);
    if (err != cudaSuccess) goto fail;
    *state_out = st;
    return cudaSuccess;
fail:
    solution_destroy(st);
    return err;
}

__global__ void mk_ref_scatter(const int32_t* d, int32_t* o0, int32_t* o1, int32_t* o2,
    int32_t* o3, int32_t* o4, int32_t* o5, int32_t* o6, int32_t* o7, int32_t* o8,
    int32_t* o9, int32_t* o10, int32_t* o11, int32_t* o12, int32_t* o13, int32_t* o14,
    int32_t* o15, int32_t* o16, int32_t* o17) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    o0[0]=d[0]; o1[0]=d[1]; o2[0]=d[2]; o3[0]=d[3]; o4[0]=d[4]; o5[0]=d[5];
    o6[0]=d[6]; o7[0]=d[7]; o8[0]=d[8]; o9[0]=d[9]; o10[0]=d[10]; o11[0]=d[11];
    o12[0]=d[12]; o13[0]=d[13]; o14[0]=d[14]; o15[0]=d[15]; o16[0]=d[16]; o17[0]=d[17];
}

extern "C" cudaError_t solution_run(void* state, const MkRunSpec* run,
        const void* inputs_void, void* outputs_void, void* workspace,
        size_t workspace_bytes, cudaStream_t stream) {
    if (!state || !inputs_void || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 256) return cudaErrorInvalidValue;
    MkRefState* st = (MkRefState*)state;
    if (!mk_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const MkInputs* in = (const MkInputs*)inputs_void;
    MkOutputs* out = (MkOutputs*)outputs_void;
    if (run->batch_size > 0 && (!in->op_kind || !in->tensor_id || !in->ci || !in->cj ||
        !in->rank || !in->dim0 || !in->dim1 || !in->task_id || !in->sched ||
        !in->amount || !in->mask || !in->payload || !in->dep_off || !in->dep_count ||
        !in->out_off || !in->out_count)) {
        return cudaErrorInvalidValue;
    }

    int32_t* d_counts = reinterpret_cast<int32_t*>(workspace);
    MkRefDev s = ref_make_dev(st);
    mk_ref_step_kernel<<<1, 1, 0, stream>>>(
        s, run->batch_size,
        in->op_kind, in->tensor_id, in->ci, in->cj, in->rank, in->dim0, in->dim1,
        in->task_id, in->sched, in->amount, in->mask, in->payload,
        in->dep_off, in->dep_count, in->out_off, in->out_count,
        in->dep_tensor, in->dep_i, in->dep_j,
        in->out_tensor, in->out_i, in->out_j, in->out_amount,
        d_counts, out->mk_event_hash, out->cell_hash, out->task_hash,
        out->queue_hash, out->reverse_dep_hash);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mk_ref_scatter<<<1, 1, 0, stream>>>(d_counts,
        out->tensors_defined, out->define_stall, out->tasks_ready_register,
        out->tasks_blocked_register, out->cell_decrements, out->cell_zero,
        out->cell_already_zero, out->tasks_ready_push, out->tasks_popped,
        out->tasks_started, out->tasks_completed, out->dynamic_cancelled,
        out->cell_cancelled, out->tasks_cancel_by_cell, out->tasks_cancel_explicit,
        out->ready_stale_drop, out->popped_stale_drop, out->invalid_count);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return ref_reset_state((MkRefState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    MkRefState* st = (MkRefState*)state;
    cudaFree(st->counters);
    cudaFree(st->t_defined); cudaFree(st->t_rank); cudaFree(st->t_dim0);
    cudaFree(st->t_dim1); cudaFree(st->t_generation);
    cudaFree(st->c_initial); cudaFree(st->c_remaining); cudaFree(st->c_zero_pushed);
    cudaFree(st->c_cell_seq); cudaFree(st->c_last_dec_seq); cudaFree(st->c_cancelled);
    cudaFree(st->tk_used); cudaFree(st->tk_id); cudaFree(st->tk_seq);
    cudaFree(st->tk_sched); cudaFree(st->tk_status); cudaFree(st->tk_mask);
    cudaFree(st->tk_payload); cudaFree(st->tk_ready_seq); cudaFree(st->tk_pop_seq);
    cudaFree(st->tk_dep_count); cudaFree(st->tk_out_count);
    cudaFree(st->tk_dep_tensor); cudaFree(st->tk_dep_i); cudaFree(st->tk_dep_j);
    cudaFree(st->tk_out_tensor); cudaFree(st->tk_out_i); cudaFree(st->tk_out_j);
    cudaFree(st->tk_out_amount);
    cudaFree(st->rq_head); cudaFree(st->rq_count); cudaFree(st->rq_task); cudaFree(st->rq_seq);
    cudaFree(st->pq_head); cudaFree(st->pq_count); cudaFree(st->pq_task); cudaFree(st->pq_seq);
    free(st);
}
