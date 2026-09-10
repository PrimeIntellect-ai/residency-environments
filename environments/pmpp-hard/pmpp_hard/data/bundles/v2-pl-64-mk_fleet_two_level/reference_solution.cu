// PMPP_CANARY_64_8be954d0ce -- held-out canary; MUST NOT appear in any submission
// file: mk_fleet_two_level_reference.cu
//
// Reference device implementation of T64 (MK5). Single-thread kernels over
// device-resident persistent state. Representation: tasks in a slot table, ready
// queues as ring buffers, mailboxes as ring buffers, local L2 counters as a dense
// [chiplet * max_tasks + slot] grid, running copies in a flat array. Independent
// of naive.cu and the host oracle.

#include "mk_fleet_two_level_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace {

struct RTask {
    uint64_t task_id;
    uint64_t task_seq;
    uint64_t worker_mask;
    uint64_t duration;
    uint64_t payload_seed;
    uint64_t output_increment;
    uint64_t ready_seq_or_zero;
    uint64_t result_hash;
    uint64_t global_parts_done;
    uint32_t home_chiplet;
    uint32_t output_event;
    uint32_t expected_local_parts;
    uint32_t expected_global_parts;
    uint32_t wait_ev[MKF_MAX_WAITS];
    uint32_t wait_target[MKF_MAX_WAITS];
    uint8_t scope;
    uint8_t status;
    uint8_t wait_count;
    uint8_t used;
    uint8_t mark;     // scratch for hashing selection
    uint8_t pad[3];
};

struct RReady {
    uint64_t task_id;
    uint64_t ready_seq;
    uint32_t chiplet;
    uint32_t pad;
};

struct RCopy {
    uint64_t task_id;
    uint64_t due_clock;
    uint64_t output_increment;
    uint64_t duration;
    uint32_t home_chiplet;
    uint32_t chiplet;
    uint32_t worker;
    uint32_t output_event;
    uint8_t scope;
    uint8_t pad[7];
};

}  // namespace

struct MkfReferenceState {
    MkfProblemSpec spec;
    int chiplet_count, wpc, total_workers, max_tasks, max_events, max_ready, max_wq, max_running;

    RTask* tasks;            // max_tasks
    RReady* ready_pool;      // chiplet_count * max_ready  (ring)
    int32_t* ready_head;     // chiplet_count
    int32_t* ready_count;    // chiplet_count
    int32_t* rr_cursor;      // chiplet_count
    RCopy* mbox_pool;        // total_workers * max_wq (ring)
    int32_t* mbox_head;      // total_workers
    int32_t* mbox_count;     // total_workers
    uint8_t* worker_running; // total_workers
    RCopy* running;          // max_running
    int32_t* running_count;  // 1
    int64_t* gcounter;       // max_events
    // dense local counters keyed by chiplet*max_tasks + slot
    int64_t* local_counter;  // chiplet_count * max_tasks
    int64_t* local_target;   // chiplet_count * max_tasks
    uint8_t* local_used;     // chiplet_count * max_tasks
    uint8_t* local_fence;    // chiplet_count * max_tasks
    uint8_t* part_dispatched;// chiplet_count * max_tasks
    int64_t* counters;       // MKF_COUNTER_COUNT
    // scalars: clock, event_seq, task_seq_next, dispatch_seq_next,
    //          local_count_seq_next, global_count_seq_next, ready_seq_next,
    //          last_local_seq, last_global_seq, run_event_hash
    uint64_t* scalars;       // 10
};

#define R_SC_CLOCK 0
#define R_SC_EVSEQ 1
#define R_SC_TASKSEQ 2
#define R_SC_DISPSEQ 3
#define R_SC_LOCALSEQ 4
#define R_SC_GLOBALSEQ 5
#define R_SC_READYSEQ 6
#define R_SC_LASTLOCAL 7
#define R_SC_LASTGLOBAL 8
#define R_SC_EVHASH 9

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
    int chiplet_count, wpc, total_workers, max_tasks, max_events, max_ready, max_wq, max_running;
    RTask* tasks;
    RReady* ready_pool; int32_t* ready_head; int32_t* ready_count; int32_t* rr_cursor;
    RCopy* mbox_pool; int32_t* mbox_head; int32_t* mbox_count; uint8_t* worker_running;
    RCopy* running; int32_t* running_count;
    int64_t* gcounter;
    int64_t* local_counter; int64_t* local_target; uint8_t* local_used; uint8_t* local_fence;
    uint8_t* part_dispatched;
    int64_t* counters;
    uint64_t clock, event_seq, task_seq_next, dispatch_seq_next;
    uint64_t local_count_seq_next, global_count_seq_next, ready_seq_next;
    uint64_t last_local_seq, last_global_seq, ev_hash;
    uint32_t op_index;
};

__device__ __forceinline__ int r_chof(RCtx* c, int w) { return w / c->wpc; }
__device__ __forceinline__ int r_gw(RCtx* c, int ch, int l) { return ch * c->wpc + l; }
__device__ __forceinline__ int r_lkey(RCtx* c, int ch, int slot) { return ch * c->max_tasks + slot; }

__device__ int r_find_task(RCtx* c, uint64_t id) {
    for (int i = 0; i < c->max_tasks; ++i)
        if (c->tasks[i].used && c->tasks[i].task_id == id) return i;
    return -1;
}

__device__ void r_emit(RCtx* c, uint8_t kind, uint32_t chiplet_or, uint32_t worker_or,
                        uint64_t task_or, uint32_t event_or, uint64_t value_or) {
    uint64_t es = c->event_seq;
    uint64_t* h = &c->ev_hash;
    rfnv(h, &kind, 1);
    rfnv(h, &es, 8);
    rfnv(h, &c->op_index, 4);
    uint64_t cyc = c->clock; rfnv(h, &cyc, 8);
    rfnv(h, &chiplet_or, 4);
    rfnv(h, &worker_or, 4);
    rfnv(h, &task_or, 8);
    rfnv(h, &event_or, 4);
    rfnv(h, &value_or, 8);
    c->event_seq = es + 1;
}

__device__ bool r_chiplet_participates(RCtx* c, uint64_t mask, int ch) {
    for (int l = 0; l < c->wpc; ++l) {
        int w = r_gw(c, ch, l);
        if (w < 64 && ((mask >> w) & 1ULL)) return true;
    }
    return false;
}
__device__ int r_count_home_selected(RCtx* c, uint64_t mask, int home) {
    int cnt = 0;
    for (int l = 0; l < c->wpc; ++l) {
        int w = r_gw(c, home, l);
        if (w < 64 && ((mask >> w) & 1ULL)) cnt++;
    }
    return cnt;
}

__device__ bool r_waits_satisfied(RCtx* c, RTask* t) {
    for (int k = 0; k < t->wait_count; ++k)
        if ((uint64_t)c->gcounter[t->wait_ev[k]] < (uint64_t)t->wait_target[k]) return false;
    return true;
}

// ready ring ops
__device__ void r_ready_push(RCtx* c, int ch, const RReady* r) {
    if (c->ready_count[ch] >= c->max_ready) return;
    int h = c->ready_head[ch], cnt = c->ready_count[ch];
    int slot = (h + cnt) % c->max_ready;
    c->ready_pool[(size_t)ch * c->max_ready + slot] = *r;
    c->ready_count[ch] = cnt + 1;
}
__device__ bool r_ready_front(RCtx* c, int ch, RReady* out) {
    if (c->ready_count[ch] == 0) return false;
    int h = c->ready_head[ch];
    *out = c->ready_pool[(size_t)ch * c->max_ready + h];
    return true;
}
__device__ void r_ready_pop(RCtx* c, int ch) {
    c->ready_head[ch] = (c->ready_head[ch] + 1) % c->max_ready;
    c->ready_count[ch] -= 1;
}

// mailbox ring ops
__device__ void r_mbox_push(RCtx* c, int w, const RCopy* cp) {
    int h = c->mbox_head[w], cnt = c->mbox_count[w];
    int slot = (h + cnt) % c->max_wq;
    c->mbox_pool[(size_t)w * c->max_wq + slot] = *cp;
    c->mbox_count[w] = cnt + 1;
}
__device__ bool r_mbox_front(RCtx* c, int w, RCopy* out) {
    if (c->mbox_count[w] == 0) return false;
    int h = c->mbox_head[w];
    *out = c->mbox_pool[(size_t)w * c->max_wq + h];
    return true;
}
__device__ void r_mbox_pop(RCtx* c, int w) {
    c->mbox_head[w] = (c->mbox_head[w] + 1) % c->max_wq;
    c->mbox_count[w] -= 1;
}

__device__ void r_set_result_hash(RCtx* c, RTask* t) {
    uint64_t h = 1469598103934665603ULL;
    uint64_t tid=t->task_id, ps=t->payload_seed, du=t->duration, oi=t->output_increment, ck=c->clock;
    uint32_t oe=t->output_event;
    rfnv(&h, &tid, 8); rfnv(&h, &ps, 8); rfnv(&h, &du, 8);
    rfnv(&h, &oe, 4); rfnv(&h, &oi, 8); rfnv(&h, &ck, 8);
    t->result_hash = h;
}

__device__ void r_enqueue_ready(RCtx* c, int slot) {
    RTask* t = &c->tasks[slot];
    t->status = MKF_ST_READY;
    if (t->scope == MKF_SCOPE_DEVICE) {
        for (int ch = 0; ch < c->chiplet_count; ++ch) {
            if (!r_chiplet_participates(c, t->worker_mask, ch)) continue;
            RReady r; r.task_id = t->task_id; r.ready_seq = c->ready_seq_next++; r.chiplet = (uint32_t)ch; r.pad = 0;
            r_ready_push(c, ch, &r);
            t->ready_seq_or_zero = r.ready_seq;
        }
    } else {
        int ch = (int)t->home_chiplet;
        RReady r; r.task_id = t->task_id; r.ready_seq = c->ready_seq_next++; r.chiplet = (uint32_t)ch; r.pad = 0;
        r_ready_push(c, ch, &r);
        t->ready_seq_or_zero = r.ready_seq;
    }
    c->counters[MKF_C_TASK_READY]++;
    r_emit(c, MKF_EV_TASK_READY, t->home_chiplet, UINT32_MAX, t->task_id, t->output_event,
           t->ready_seq_or_zero);
}

__device__ void r_ready_scan(RCtx* c) {
    // scan WAITING tasks by task_seq ascending via selection.
    for (int i = 0; i < c->max_tasks; ++i) c->tasks[i].mark = 0;
    for (;;) {
        int best = -1;
        for (int i = 0; i < c->max_tasks; ++i) {
            RTask* t = &c->tasks[i];
            if (!t->used || t->status != MKF_ST_WAITING || t->mark) continue;
            if (best < 0 || t->task_seq < c->tasks[best].task_seq) best = i;
        }
        if (best < 0) break;
        c->tasks[best].mark = 1;
        if (r_waits_satisfied(c, &c->tasks[best])) r_enqueue_ready(c, best);
    }
}

__device__ void r_global_inc(RCtx* c, uint32_t ev, uint64_t amount) {
    c->gcounter[ev] += (int64_t)amount;
    c->last_global_seq = c->global_count_seq_next++;
}

// ---------- ops ----------

__device__ void r_op_submit(RCtx* c, int a0, int a1, int a2, int a3,
                            uint64_t mask, int seed, int oe, int oinc, int wc,
                            const int* wev, const int* wtg) {
    uint64_t tid = (uint64_t)(uint32_t)a0;
    int scope = a1, home = a2, duration = a3;
    bool invalid = false;
    if (r_find_task(c, tid) >= 0) invalid = true;
    int slot = -1;
    for (int i = 0; i < c->max_tasks; ++i) if (!c->tasks[i].used) { slot = i; break; }
    if (slot < 0) invalid = true;
    if (home < 0 || home >= c->chiplet_count) invalid = true;
    if (scope < 0 || scope >= MKF_SCOPE_COUNT) invalid = true;
    if (oe < 0 || oe >= c->max_events) invalid = true;
    if (wc < 0 || wc > MKF_MAX_WAITS) invalid = true;
    if (!invalid) for (int k = 0; k < wc; ++k)
        if (wev[k] < 0 || wev[k] >= c->max_events) invalid = true;
    if (mask == 0) invalid = true;
    if (invalid) {
        c->counters[MKF_C_INVALID]++;
        r_emit(c, MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, tid, UINT32_MAX, UINT64_MAX);
        return;
    }
    RTask* t = &c->tasks[slot];
    memset(t, 0, sizeof(RTask));
    t->used = 1;
    t->task_id = tid;
    t->task_seq = c->task_seq_next++;
    t->scope = (uint8_t)scope;
    t->home_chiplet = (uint32_t)home;
    t->worker_mask = mask;
    t->duration = (uint64_t)(uint32_t)duration;
    t->payload_seed = (uint64_t)(uint32_t)seed;
    t->output_event = (uint32_t)oe;
    t->output_increment = (uint64_t)(uint32_t)oinc;
    t->wait_count = (uint8_t)wc;
    for (int k = 0; k < wc; ++k) { t->wait_ev[k] = (uint32_t)wev[k]; t->wait_target[k] = (uint32_t)wtg[k]; }
    if (scope == MKF_SCOPE_WAVE || scope == MKF_SCOPE_CU) {
        t->expected_local_parts = 0; t->expected_global_parts = 1;
    } else if (scope == MKF_SCOPE_CHIPLET) {
        t->expected_local_parts = (uint32_t)r_count_home_selected(c, mask, home);
        t->expected_global_parts = 1;
    } else {
        int pc = 0;
        for (int ch = 0; ch < c->chiplet_count; ++ch) if (r_chiplet_participates(c, mask, ch)) pc++;
        t->expected_local_parts = (uint32_t)(pc * c->wpc);
        t->expected_global_parts = (uint32_t)pc;
    }
    t->global_parts_done = 0;
    if (r_waits_satisfied(c, t)) {
        r_enqueue_ready(c, slot);
    } else {
        t->status = MKF_ST_WAITING;
        c->counters[MKF_C_TASK_WAIT]++;
        r_emit(c, MKF_EV_TASK_WAIT, t->home_chiplet, UINT32_MAX, t->task_id, t->output_event, UINT64_MAX);
    }
}

__device__ bool r_front_stale(RCtx* c, int ch) {
    RReady r;
    if (!r_ready_front(c, ch, &r)) return false;
    int idx = r_find_task(c, r.task_id);
    if (idx < 0) { r_ready_pop(c, ch); return true; }
    RTask* t = &c->tasks[idx];
    if (t->status == MKF_ST_CANCELLED) { r_ready_pop(c, ch); return true; }
    if (t->scope == MKF_SCOPE_DEVICE) {
        if (c->part_dispatched[r_lkey(c, (int)r.chiplet, idx)]) { r_ready_pop(c, ch); return true; }
        return false;
    } else {
        if (t->status != MKF_ST_READY) { r_ready_pop(c, ch); return true; }
        return false;
    }
}

__device__ void r_op_sched(RCtx* c, int chiplet, int limit) {
    if (chiplet < 0 || chiplet >= c->chiplet_count || limit == 0) {
        c->counters[MKF_C_INVALID]++;
        r_emit(c, MKF_EV_INVALID, (uint32_t)(chiplet >= 0 ? chiplet : -1), UINT32_MAX,
               UINT64_MAX, UINT32_MAX, UINT64_MAX);
        return;
    }
    int done = 0;
    while (done < limit) {
        while (r_front_stale(c, chiplet)) { }
        RReady r;
        if (!r_ready_front(c, chiplet, &r)) {
            c->counters[MKF_C_SCHED_IDLE]++;
            r_emit(c, MKF_EV_SCHED_IDLE, (uint32_t)chiplet, UINT32_MAX, UINT64_MAX, UINT32_MAX, UINT64_MAX);
            break;
        }
        int idx = r_find_task(c, r.task_id);
        RTask* t = &c->tasks[idx];
        if (t->scope == MKF_SCOPE_WAVE || t->scope == MKF_SCOPE_CU) {
            int home = (int)t->home_chiplet;
            int chosen_local = -1;
            for (int s = 0; s < c->wpc; ++s) {
                int lw = (c->rr_cursor[home] + s) % c->wpc;
                int w = r_gw(c, home, lw);
                if (w >= 64 || !((t->worker_mask >> w) & 1ULL)) continue;
                if (c->mbox_count[w] < c->max_wq) { chosen_local = lw; break; }
            }
            if (chosen_local < 0) {
                c->counters[MKF_C_DISPATCH_STALL]++;
                r_emit(c, MKF_EV_DISPATCH_STALL, (uint32_t)chiplet, UINT32_MAX, t->task_id, t->output_event, UINT64_MAX);
                break;
            }
            int w = r_gw(c, home, chosen_local);
            RCopy cp; memset(&cp, 0, sizeof(cp));
            cp.task_id=t->task_id; cp.scope=t->scope; cp.home_chiplet=t->home_chiplet;
            cp.chiplet=(uint32_t)home; cp.worker=(uint32_t)w; cp.due_clock=0;
            cp.output_event=t->output_event; cp.output_increment=t->output_increment; cp.duration=t->duration;
            r_mbox_push(c, w, &cp);
            c->rr_cursor[home] = (chosen_local + 1) % c->wpc;
            t->status = MKF_ST_DISPATCHED;
            c->dispatch_seq_next++;
            c->counters[MKF_C_DISPATCH_SINGLE]++;
            r_emit(c, MKF_EV_DISPATCH_SINGLE, (uint32_t)home, (uint32_t)w, t->task_id, t->output_event, UINT64_MAX);
            r_ready_pop(c, chiplet);
            done++;
        } else if (t->scope == MKF_SCOPE_CHIPLET) {
            int home = (int)t->home_chiplet;
            int sel[MKF_MAX_WPC]; int ns = 0;
            for (int l = 0; l < c->wpc; ++l) {
                int w = r_gw(c, home, l);
                if (w < 64 && ((t->worker_mask >> w) & 1ULL)) sel[ns++] = w;
            }
            bool all_cap = (ns > 0);
            for (int i = 0; i < ns; ++i) if (c->mbox_count[sel[i]] >= c->max_wq) all_cap = false;
            if (!all_cap) {
                c->counters[MKF_C_DISPATCH_STALL]++;
                r_emit(c, MKF_EV_DISPATCH_STALL, (uint32_t)chiplet, UINT32_MAX, t->task_id, t->output_event, UINT64_MAX);
                break;
            }
            int lk = r_lkey(c, home, idx);
            c->local_counter[lk] = 0; c->local_target[lk] = ns;
            c->local_used[lk] = 1; c->local_fence[lk] = 0;
            for (int i = 0; i < ns; ++i) {
                int w = sel[i];
                RCopy cp; memset(&cp, 0, sizeof(cp));
                cp.task_id=t->task_id; cp.scope=t->scope; cp.home_chiplet=t->home_chiplet;
                cp.chiplet=(uint32_t)home; cp.worker=(uint32_t)w; cp.due_clock=0;
                cp.output_event=t->output_event; cp.output_increment=t->output_increment; cp.duration=t->duration;
                r_mbox_push(c, w, &cp);
                c->counters[MKF_C_DISPATCH_CHIPLET_PART]++;
                r_emit(c, MKF_EV_DISPATCH_CHIPLET_PART, (uint32_t)home, (uint32_t)w, t->task_id, t->output_event, UINT64_MAX);
            }
            t->status = MKF_ST_DISPATCHED;
            c->dispatch_seq_next++;
            r_ready_pop(c, chiplet);
            done++;
        } else { // DEVICE
            int ch = (int)r.chiplet;
            bool all_cap = true;
            for (int l = 0; l < c->wpc; ++l) {
                int w = r_gw(c, ch, l);
                if (c->mbox_count[w] >= c->max_wq) all_cap = false;
            }
            if (!all_cap) {
                c->counters[MKF_C_DISPATCH_STALL]++;
                r_emit(c, MKF_EV_DISPATCH_STALL, (uint32_t)ch, UINT32_MAX, t->task_id, t->output_event, UINT64_MAX);
                break;
            }
            int lk = r_lkey(c, ch, idx);
            c->local_counter[lk] = 0; c->local_target[lk] = c->wpc;
            c->local_used[lk] = 1; c->local_fence[lk] = 0;
            for (int l = 0; l < c->wpc; ++l) {
                int w = r_gw(c, ch, l);
                RCopy cp; memset(&cp, 0, sizeof(cp));
                cp.task_id=t->task_id; cp.scope=t->scope; cp.home_chiplet=t->home_chiplet;
                cp.chiplet=(uint32_t)ch; cp.worker=(uint32_t)w; cp.due_clock=0;
                cp.output_event=t->output_event; cp.output_increment=t->output_increment; cp.duration=t->duration;
                r_mbox_push(c, w, &cp);
                c->counters[MKF_C_DISPATCH_DEVICE_PART]++;
                r_emit(c, MKF_EV_DISPATCH_DEVICE_PART, (uint32_t)ch, (uint32_t)w, t->task_id, t->output_event, UINT64_MAX);
            }
            c->part_dispatched[r_lkey(c, ch, idx)] = 1;
            if (t->status == MKF_ST_READY) t->status = MKF_ST_DISPATCHED;
            c->dispatch_seq_next++;
            r_ready_pop(c, chiplet);
            done++;
        }
    }
}

__device__ void r_op_worker(RCtx* c, int worker, int limit) {
    if (worker < 0 || worker >= c->total_workers || limit == 0) {
        c->counters[MKF_C_INVALID]++;
        r_emit(c, MKF_EV_INVALID, UINT32_MAX, (uint32_t)(worker >= 0 ? worker : -1),
               UINT64_MAX, UINT32_MAX, UINT64_MAX);
        return;
    }
    int done = 0;
    while (done < limit) {
        if (c->worker_running[worker]) {
            c->counters[MKF_C_WORKER_BUSY]++;
            r_emit(c, MKF_EV_WORKER_BUSY, (uint32_t)r_chof(c, worker), (uint32_t)worker,
                   UINT64_MAX, UINT32_MAX, UINT64_MAX);
            break;
        }
        // drop cancelled-task copies at front
        for (;;) {
            RCopy f;
            if (!r_mbox_front(c, worker, &f)) break;
            int idx = r_find_task(c, f.task_id);
            if (idx < 0 || c->tasks[idx].status == MKF_ST_CANCELLED) r_mbox_pop(c, worker);
            else break;
        }
        RCopy cp;
        if (!r_mbox_front(c, worker, &cp)) {
            c->counters[MKF_C_WORKER_IDLE]++;
            r_emit(c, MKF_EV_WORKER_IDLE, (uint32_t)r_chof(c, worker), (uint32_t)worker,
                   UINT64_MAX, UINT32_MAX, UINT64_MAX);
            break;
        }
        r_mbox_pop(c, worker);
        cp.due_clock = c->clock + cp.duration;
        c->worker_running[worker] = 1;
        int rc = c->running_count[0];
        c->running[rc] = cp;
        c->running_count[0] = rc + 1;
        int idx = r_find_task(c, cp.task_id);
        if (idx >= 0 && (cp.scope == MKF_SCOPE_WAVE || cp.scope == MKF_SCOPE_CU))
            if (c->tasks[idx].status == MKF_ST_DISPATCHED) c->tasks[idx].status = MKF_ST_RUNNING;
        c->counters[MKF_C_WORKER_START_TASK]++;
        r_emit(c, MKF_EV_WORKER_START_TASK, cp.chiplet, (uint32_t)worker, cp.task_id,
               cp.output_event, cp.due_clock);
        done++;
    }
}

__device__ bool r_running_less(const RCopy* a, const RCopy* b) {
    if (a->due_clock != b->due_clock) return a->due_clock < b->due_clock;
    if (a->chiplet != b->chiplet) return a->chiplet < b->chiplet;
    if (a->worker != b->worker) return a->worker < b->worker;
    return a->task_id < b->task_id;
}

__device__ void r_op_advance(RCtx* c, int delta, int max_fin) {
    if (delta < 0) {
        c->counters[MKF_C_INVALID]++;
        r_emit(c, MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, UINT64_MAX, UINT32_MAX, UINT64_MAX);
        return;
    }
    c->clock += (uint64_t)(uint32_t)delta;
    int fin = 0;
    while (fin < max_fin) {
        int best = -1;
        for (int i = 0; i < c->running_count[0]; ++i) {
            if (c->running[i].due_clock > c->clock) continue;
            if (best < 0 || r_running_less(&c->running[i], &c->running[best])) best = i;
        }
        if (best < 0) break;
        RCopy cp = c->running[best];
        // remove from running array (shift down)
        int rc = c->running_count[0];
        for (int i = best; i < rc - 1; ++i) c->running[i] = c->running[i + 1];
        c->running_count[0] = rc - 1;
        c->worker_running[cp.worker] = 0;

        c->counters[MKF_C_WORKER_FINISH]++;
        r_emit(c, MKF_EV_WORKER_FINISH, cp.chiplet, cp.worker, cp.task_id, cp.output_event, cp.due_clock);

        int idx = r_find_task(c, cp.task_id);
        bool cancelled = (idx < 0 || c->tasks[idx].status == MKF_ST_CANCELLED);
        if (cancelled) {
            c->counters[MKF_C_CANCELLED_FINISH]++;
            r_emit(c, MKF_EV_CANCELLED_FINISH, cp.chiplet, cp.worker, cp.task_id, cp.output_event, UINT64_MAX);
            fin++;
            continue;
        }
        RTask* t = &c->tasks[idx];
        if (cp.scope == MKF_SCOPE_WAVE || cp.scope == MKF_SCOPE_CU) {
            r_global_inc(c, cp.output_event, cp.output_increment);
            c->counters[MKF_C_GLOBAL_ATOMIC_INC]++;
            r_emit(c, MKF_EV_GLOBAL_ATOMIC_INC, cp.chiplet, cp.worker, cp.task_id, cp.output_event, cp.output_increment);
            t->status = MKF_ST_GLOBAL_DONE;
            r_set_result_hash(c, t);
            r_ready_scan(c);
        } else if (cp.scope == MKF_SCOPE_CHIPLET) {
            int lk = r_lkey(c, (int)cp.chiplet, idx);
            if (c->local_used[lk]) {
                c->local_counter[lk] += 1;
                c->last_local_seq = c->local_count_seq_next++;
                c->counters[MKF_C_LOCAL_L2_INC]++;
                r_emit(c, MKF_EV_LOCAL_L2_INC, cp.chiplet, cp.worker, cp.task_id, cp.output_event, (uint64_t)c->local_counter[lk]);
                if (c->local_counter[lk] >= c->local_target[lk] && !c->local_fence[lk]) {
                    c->local_fence[lk] = 1;
                    c->counters[MKF_C_L2_FENCE]++;
                    r_emit(c, MKF_EV_L2_FENCE, cp.chiplet, UINT32_MAX, cp.task_id, cp.output_event, (uint64_t)c->local_counter[lk]);
                    r_global_inc(c, cp.output_event, cp.output_increment);
                    c->counters[MKF_C_GLOBAL_ATOMIC_INC]++;
                    r_emit(c, MKF_EV_GLOBAL_ATOMIC_INC, cp.chiplet, UINT32_MAX, cp.task_id, cp.output_event, cp.output_increment);
                    t->status = MKF_ST_GLOBAL_DONE;
                    r_set_result_hash(c, t);
                    r_ready_scan(c);
                }
            }
        } else { // DEVICE
            int lk = r_lkey(c, (int)cp.chiplet, idx);
            if (c->local_used[lk]) {
                c->local_counter[lk] += 1;
                c->last_local_seq = c->local_count_seq_next++;
                c->counters[MKF_C_LOCAL_L2_INC]++;
                r_emit(c, MKF_EV_LOCAL_L2_INC, cp.chiplet, cp.worker, cp.task_id, cp.output_event, (uint64_t)c->local_counter[lk]);
                if (c->local_counter[lk] >= c->local_target[lk] && !c->local_fence[lk]) {
                    c->local_fence[lk] = 1;
                    c->counters[MKF_C_L2_FENCE]++;
                    r_emit(c, MKF_EV_L2_FENCE, cp.chiplet, UINT32_MAX, cp.task_id, cp.output_event, (uint64_t)c->local_counter[lk]);
                    r_global_inc(c, cp.output_event, cp.output_increment);
                    c->counters[MKF_C_GLOBAL_ATOMIC_INC]++;
                    r_emit(c, MKF_EV_GLOBAL_ATOMIC_INC, cp.chiplet, UINT32_MAX, cp.task_id, cp.output_event, cp.output_increment);
                    t->global_parts_done += 1;
                    if (t->global_parts_done >= t->expected_global_parts) {
                        t->status = MKF_ST_GLOBAL_DONE; r_set_result_hash(c, t);
                    } else {
                        t->status = MKF_ST_LOCAL_DONE;
                    }
                    r_ready_scan(c);
                }
            }
        }
        fin++;
    }
}

__device__ void r_op_event_force(RCtx* c, int ev, int amount) {
    if (ev < 0 || ev >= c->max_events) {
        c->counters[MKF_C_INVALID]++;
        r_emit(c, MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, UINT64_MAX,
               (uint32_t)(ev >= 0 ? ev : -1), UINT64_MAX);
        return;
    }
    r_global_inc(c, (uint32_t)ev, (uint64_t)(uint32_t)amount);
    c->counters[MKF_C_EVENT_FORCE]++;
    r_emit(c, MKF_EV_EVENT_FORCE, UINT32_MAX, UINT32_MAX, UINT64_MAX, (uint32_t)ev, (uint64_t)(uint32_t)amount);
    r_ready_scan(c);
}

__device__ void r_op_cancel(RCtx* c, int a0) {
    uint64_t tid = (uint64_t)(uint32_t)a0;
    int idx = r_find_task(c, tid);
    if (idx < 0) {
        c->counters[MKF_C_INVALID]++;
        r_emit(c, MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, tid, UINT32_MAX, UINT64_MAX);
        return;
    }
    uint8_t st = c->tasks[idx].status;
    if (st == MKF_ST_GLOBAL_DONE || st == MKF_ST_CANCELLED) {
        c->counters[MKF_C_INVALID]++;
        r_emit(c, MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, tid, UINT32_MAX, UINT64_MAX);
        return;
    }
    c->tasks[idx].status = MKF_ST_CANCELLED;
    c->counters[MKF_C_TASK_CANCEL]++;
    r_emit(c, MKF_EV_TASK_CANCEL, c->tasks[idx].home_chiplet, UINT32_MAX, tid, c->tasks[idx].output_event, UINT64_MAX);
}

__global__ void r_kernel(MkfReferenceState st, int op, int op_index,
                         int a0, int a1, int a2, int a3,
                         uint64_t mask, int seed, int oe, int oinc, int wc,
                         int wev0, int wev1, int wtg0, int wtg1) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    RCtx ctx; RCtx* c = &ctx;
    c->chiplet_count = st.chiplet_count; c->wpc = st.wpc; c->total_workers = st.total_workers;
    c->max_tasks = st.max_tasks; c->max_events = st.max_events; c->max_ready = st.max_ready;
    c->max_wq = st.max_wq; c->max_running = st.max_running;
    c->tasks = st.tasks; c->ready_pool = st.ready_pool; c->ready_head = st.ready_head;
    c->ready_count = st.ready_count; c->rr_cursor = st.rr_cursor;
    c->mbox_pool = st.mbox_pool; c->mbox_head = st.mbox_head; c->mbox_count = st.mbox_count;
    c->worker_running = st.worker_running; c->running = st.running; c->running_count = st.running_count;
    c->gcounter = st.gcounter;
    c->local_counter = st.local_counter; c->local_target = st.local_target;
    c->local_used = st.local_used; c->local_fence = st.local_fence;
    c->part_dispatched = st.part_dispatched;
    c->counters = st.counters;
    c->clock = st.scalars[R_SC_CLOCK];
    c->event_seq = st.scalars[R_SC_EVSEQ];
    c->task_seq_next = st.scalars[R_SC_TASKSEQ];
    c->dispatch_seq_next = st.scalars[R_SC_DISPSEQ];
    c->local_count_seq_next = st.scalars[R_SC_LOCALSEQ];
    c->global_count_seq_next = st.scalars[R_SC_GLOBALSEQ];
    c->ready_seq_next = st.scalars[R_SC_READYSEQ];
    c->last_local_seq = st.scalars[R_SC_LASTLOCAL];
    c->last_global_seq = st.scalars[R_SC_LASTGLOBAL];
    c->ev_hash = 1469598103934665603ULL;
    c->op_index = (uint32_t)op_index;

    int wev[MKF_MAX_WAITS] = {wev0, wev1};
    int wtg[MKF_MAX_WAITS] = {wtg0, wtg1};

    if (op == MKF_OP_SUBMIT) r_op_submit(c, a0, a1, a2, a3, mask, seed, oe, oinc, wc, wev, wtg);
    else if (op == MKF_OP_SCHED) r_op_sched(c, a0, a1);
    else if (op == MKF_OP_WORKER) r_op_worker(c, a0, a1);
    else if (op == MKF_OP_ADVANCE) r_op_advance(c, a0, a1);
    else if (op == MKF_OP_EVENT_FORCE) r_op_event_force(c, a0, a1);
    else if (op == MKF_OP_CANCEL) r_op_cancel(c, a0);
    else { c->counters[MKF_C_INVALID]++; r_emit(c, MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, UINT64_MAX, UINT32_MAX, UINT64_MAX); }

    st.scalars[R_SC_CLOCK] = c->clock;
    st.scalars[R_SC_EVSEQ] = c->event_seq;
    st.scalars[R_SC_TASKSEQ] = c->task_seq_next;
    st.scalars[R_SC_DISPSEQ] = c->dispatch_seq_next;
    st.scalars[R_SC_LOCALSEQ] = c->local_count_seq_next;
    st.scalars[R_SC_GLOBALSEQ] = c->global_count_seq_next;
    st.scalars[R_SC_READYSEQ] = c->ready_seq_next;
    st.scalars[R_SC_LASTLOCAL] = c->last_local_seq;
    st.scalars[R_SC_LASTGLOBAL] = c->last_global_seq;
    st.scalars[R_SC_EVHASH] = c->ev_hash;
}

// ---------- hashing kernel ----------
__global__ void r_hash_kernel(MkfReferenceState st,
                              int64_t* out_counters, uint64_t* out_evhash,
                              uint64_t* out_thash, uint64_t* out_qhash,
                              uint64_t* out_chash, uint64_t* out_rhash,
                              uint64_t* out_clock, uint64_t* out_evseq) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    RCtx ctx; RCtx* c = &ctx;
    c->chiplet_count = st.chiplet_count; c->wpc = st.wpc; c->total_workers = st.total_workers;
    c->max_tasks = st.max_tasks; c->max_events = st.max_events; c->max_ready = st.max_ready;
    c->max_wq = st.max_wq; c->max_running = st.max_running;
    c->tasks = st.tasks; c->ready_pool = st.ready_pool; c->ready_head = st.ready_head;
    c->ready_count = st.ready_count; c->mbox_pool = st.mbox_pool; c->mbox_head = st.mbox_head;
    c->mbox_count = st.mbox_count; c->running = st.running; c->running_count = st.running_count;
    c->gcounter = st.gcounter; c->local_counter = st.local_counter; c->local_target = st.local_target;
    c->local_used = st.local_used;

    for (int i = 0; i < MKF_COUNTER_COUNT; ++i) out_counters[i] = st.counters[i];
    out_evhash[0] = st.scalars[R_SC_EVHASH];
    out_clock[0] = st.scalars[R_SC_CLOCK];
    out_evseq[0] = st.scalars[R_SC_EVSEQ];
    uint64_t last_local = st.scalars[R_SC_LASTLOCAL];
    uint64_t last_global = st.scalars[R_SC_LASTGLOBAL];

    // task_state_hash: by task_id asc via selection
    for (int i = 0; i < c->max_tasks; ++i) c->tasks[i].mark = 0;
    uint64_t h = 1469598103934665603ULL;
    for (;;) {
        int best = -1;
        for (int i = 0; i < c->max_tasks; ++i) {
            RTask* t = &c->tasks[i];
            if (!t->used || t->mark) continue;
            if (best < 0 || t->task_id < c->tasks[best].task_id) best = i;
        }
        if (best < 0) break;
        RTask* t = &c->tasks[best]; t->mark = 1;
        uint64_t tid=t->task_id, ts=t->task_seq, mask=t->worker_mask, du=t->duration;
        uint8_t sc=t->scope, stt=t->status;
        uint32_t hc=t->home_chiplet, el=t->expected_local_parts, eg=t->expected_global_parts;
        uint64_t rs=t->ready_seq_or_zero, rh=t->result_hash;
        rfnv(&h,&tid,8); rfnv(&h,&ts,8); rfnv(&h,&sc,1); rfnv(&h,&hc,4); rfnv(&h,&mask,8);
        rfnv(&h,&du,8); rfnv(&h,&stt,1); rfnv(&h,&el,4); rfnv(&h,&eg,4); rfnv(&h,&rs,8); rfnv(&h,&rh,8);
    }
    out_thash[0] = h;

    // queue_hash
    h = 1469598103934665603ULL;
    for (int ch = 0; ch < c->chiplet_count; ++ch) {
        uint8_t kind = 0; uint32_t cc = (uint32_t)ch;
        rfnv(&h, &kind, 1); rfnv(&h, &cc, 4);
        int cnt = c->ready_count[ch], head = c->ready_head[ch];
        for (int pos = 0; pos < cnt; ++pos) {
            int slot = (head + pos) % c->max_ready;
            RReady* r = &c->ready_pool[(size_t)ch * c->max_ready + slot];
            uint64_t p64 = (uint64_t)pos;
            rfnv(&h, &p64, 8); rfnv(&h, &r->task_id, 8); rfnv(&h, &r->ready_seq, 8);
            uint32_t rc = r->chiplet; rfnv(&h, &rc, 4);
        }
    }
    for (int w = 0; w < c->total_workers; ++w) {
        uint8_t kind = 1; uint32_t ww = (uint32_t)w;
        rfnv(&h, &kind, 1); rfnv(&h, &ww, 4);
        int cnt = c->mbox_count[w], head = c->mbox_head[w];
        for (int pos = 0; pos < cnt; ++pos) {
            int slot = (head + pos) % c->max_wq;
            RCopy* cp = &c->mbox_pool[(size_t)w * c->max_wq + slot];
            uint64_t p64 = (uint64_t)pos;
            rfnv(&h, &p64, 8); rfnv(&h, &cp->task_id, 8);
        }
    }
    out_qhash[0] = h;

    // counter_hash: local by (chiplet, task_id) asc then global by event
    h = 1469598103934665603ULL;
    // build list of used local entries; iterate chiplet asc, then task_id asc.
    for (int ch = 0; ch < c->chiplet_count; ++ch) {
        // selection over task slots by task_id among used local entries for this chiplet
        // need a visited marker; reuse a local stack array bounded by max_tasks.
        // We do O(n^2): repeatedly pick min unvisited.
        // Use tasks[i].pad[0] as visited marker.
        for (int i = 0; i < c->max_tasks; ++i) c->tasks[i].pad[0] = 0;
        for (;;) {
            int best = -1; uint64_t bestid = 0;
            for (int slot = 0; slot < c->max_tasks; ++slot) {
                int lk = ch * c->max_tasks + slot;
                if (!c->local_used[lk]) continue;
                if (c->tasks[slot].pad[0]) continue;
                uint64_t tid = c->tasks[slot].task_id;
                if (best < 0 || tid < bestid) { best = slot; bestid = tid; }
            }
            if (best < 0) break;
            c->tasks[best].pad[0] = 1;
            int lk = ch * c->max_tasks + best;
            uint8_t kind = 0; uint32_t cc = (uint32_t)ch;
            uint64_t tid = c->tasks[best].task_id;
            uint64_t cnt = (uint64_t)c->local_counter[lk], tg = (uint64_t)c->local_target[lk];
            rfnv(&h, &kind, 1); rfnv(&h, &cc, 4); rfnv(&h, &tid, 8);
            rfnv(&h, &cnt, 8); rfnv(&h, &tg, 8); rfnv(&h, &last_local, 8);
        }
    }
    for (int e = 0; e < c->max_events; ++e) {
        uint8_t kind = 1; uint32_t ev = (uint32_t)e;
        uint64_t cnt = (uint64_t)c->gcounter[e];
        rfnv(&h, &kind, 1); rfnv(&h, &ev, 4); rfnv(&h, &cnt, 8); rfnv(&h, &last_global, 8);
    }
    out_chash[0] = h;

    // running_hash: by (due_clock, chiplet, worker, task_id) via selection
    h = 1469598103934665603ULL;
    int rcount = c->running_count[0];
    // mark consumed in a temp: reuse copy due_clock? safer: visited array via tasks pad[1].
    // running entries are independent; use a local bool buffer of size max_running.
    {
        // selection sort without modifying array
        for (int i = 0; i < rcount; ++i) c->running[i].pad[0] = 0;
        for (;;) {
            int best = -1;
            for (int i = 0; i < rcount; ++i) {
                if (c->running[i].pad[0]) continue;
                if (best < 0 || r_running_less(&c->running[i], &c->running[best])) best = i;
            }
            if (best < 0) break;
            c->running[best].pad[0] = 1;
            RCopy* cp = &c->running[best];
            uint64_t dc = cp->due_clock, tid = cp->task_id;
            uint32_t ch = cp->chiplet, w = cp->worker; uint8_t sc = cp->scope;
            rfnv(&h, &dc, 8); rfnv(&h, &ch, 4); rfnv(&h, &w, 4); rfnv(&h, &tid, 8); rfnv(&h, &sc, 1);
        }
    }
    out_rhash[0] = h;
}

// ---------- host ABI ----------

static cudaError_t r_reset(MkfReferenceState* st, cudaStream_t s) {
    cudaError_t e;
    int nlocal = st->chiplet_count * st->max_tasks;
    e = cudaMemsetAsync(st->tasks, 0, sizeof(RTask) * st->max_tasks, s); if (e) return e;
    e = cudaMemsetAsync(st->ready_head, 0, sizeof(int32_t) * st->chiplet_count, s); if (e) return e;
    e = cudaMemsetAsync(st->ready_count, 0, sizeof(int32_t) * st->chiplet_count, s); if (e) return e;
    e = cudaMemsetAsync(st->rr_cursor, 0, sizeof(int32_t) * st->chiplet_count, s); if (e) return e;
    e = cudaMemsetAsync(st->mbox_head, 0, sizeof(int32_t) * st->total_workers, s); if (e) return e;
    e = cudaMemsetAsync(st->mbox_count, 0, sizeof(int32_t) * st->total_workers, s); if (e) return e;
    e = cudaMemsetAsync(st->worker_running, 0, sizeof(uint8_t) * st->total_workers, s); if (e) return e;
    e = cudaMemsetAsync(st->running_count, 0, sizeof(int32_t), s); if (e) return e;
    e = cudaMemsetAsync(st->gcounter, 0, sizeof(int64_t) * st->max_events, s); if (e) return e;
    e = cudaMemsetAsync(st->local_counter, 0, sizeof(int64_t) * nlocal, s); if (e) return e;
    e = cudaMemsetAsync(st->local_target, 0, sizeof(int64_t) * nlocal, s); if (e) return e;
    e = cudaMemsetAsync(st->local_used, 0, sizeof(uint8_t) * nlocal, s); if (e) return e;
    e = cudaMemsetAsync(st->local_fence, 0, sizeof(uint8_t) * nlocal, s); if (e) return e;
    e = cudaMemsetAsync(st->part_dispatched, 0, sizeof(uint8_t) * nlocal, s); if (e) return e;
    e = cudaMemsetAsync(st->counters, 0, sizeof(int64_t) * MKF_COUNTER_COUNT, s); if (e) return e;
    uint64_t sc[10] = {0, 0, 1, 1, 1, 1, 1, 0, 0, 0};
    e = cudaMemcpyAsync(st->scalars, sc, sizeof(sc), cudaMemcpyHostToDevice, s); if (e) return e;
    return cudaStreamSynchronize(s);
}

extern "C" size_t solution_workspace_bytes(const MkfProblemSpec* spec) {
    if (!mkf_validate_problem_spec(spec)) return 0;
    return 256;
}

extern "C" cudaError_t solution_init(const MkfProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!mkf_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    MkfReferenceState* st = (MkfReferenceState*)malloc(sizeof(MkfReferenceState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(*st));
    memcpy(&st->spec, spec, sizeof(MkfProblemSpec));
    st->chiplet_count = spec->chiplet_count;
    st->wpc = spec->workers_per_chiplet;
    st->total_workers = spec->chiplet_count * spec->workers_per_chiplet;
    st->max_tasks = spec->max_tasks;
    st->max_events = spec->max_events;
    st->max_ready = spec->max_ready_per_chiplet;
    st->max_wq = spec->max_worker_queue;
    st->max_running = spec->max_running;
    int nready = st->chiplet_count * st->max_ready;
    int nmbox = st->total_workers * st->max_wq;
    int nlocal = st->chiplet_count * st->max_tasks;

    cudaError_t e = cudaSuccess;
    e = cudaMalloc(&st->tasks, sizeof(RTask) * st->max_tasks); if (e) goto fail;
    e = cudaMalloc(&st->ready_pool, sizeof(RReady) * nready); if (e) goto fail;
    e = cudaMalloc(&st->ready_head, sizeof(int32_t) * st->chiplet_count); if (e) goto fail;
    e = cudaMalloc(&st->ready_count, sizeof(int32_t) * st->chiplet_count); if (e) goto fail;
    e = cudaMalloc(&st->rr_cursor, sizeof(int32_t) * st->chiplet_count); if (e) goto fail;
    e = cudaMalloc(&st->mbox_pool, sizeof(RCopy) * nmbox); if (e) goto fail;
    e = cudaMalloc(&st->mbox_head, sizeof(int32_t) * st->total_workers); if (e) goto fail;
    e = cudaMalloc(&st->mbox_count, sizeof(int32_t) * st->total_workers); if (e) goto fail;
    e = cudaMalloc(&st->worker_running, sizeof(uint8_t) * st->total_workers); if (e) goto fail;
    e = cudaMalloc(&st->running, sizeof(RCopy) * st->max_running); if (e) goto fail;
    e = cudaMalloc(&st->running_count, sizeof(int32_t)); if (e) goto fail;
    e = cudaMalloc(&st->gcounter, sizeof(int64_t) * st->max_events); if (e) goto fail;
    e = cudaMalloc(&st->local_counter, sizeof(int64_t) * nlocal); if (e) goto fail;
    e = cudaMalloc(&st->local_target, sizeof(int64_t) * nlocal); if (e) goto fail;
    e = cudaMalloc(&st->local_used, sizeof(uint8_t) * nlocal); if (e) goto fail;
    e = cudaMalloc(&st->local_fence, sizeof(uint8_t) * nlocal); if (e) goto fail;
    e = cudaMalloc(&st->part_dispatched, sizeof(uint8_t) * nlocal); if (e) goto fail;
    e = cudaMalloc(&st->counters, sizeof(int64_t) * MKF_COUNTER_COUNT); if (e) goto fail;
    e = cudaMalloc(&st->scalars, sizeof(uint64_t) * 10); if (e) goto fail;

    e = r_reset(st, stream); if (e) goto fail;
    *state_out = st;
    return cudaSuccess;
fail:
    solution_destroy(st);
    return e;
}

extern "C" cudaError_t solution_run(void* state, const MkfRunSpec* run, const void* inputs,
                                    void* outputs, void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    (void)inputs; (void)workspace;
    if (!state || !mkf_validate_run_spec(run) || !outputs) return cudaErrorInvalidValue;
    if (workspace_bytes < 1) return cudaErrorInvalidValue;
    MkfReferenceState* st = (MkfReferenceState*)state;
    MkfOutputs* out = (MkfOutputs*)outputs;
    if (!out->counters || !out->fleet_event_hash || !out->task_state_hash || !out->queue_hash ||
        !out->counter_hash || !out->running_hash || !out->clock_out || !out->event_seq_out)
        return cudaErrorInvalidValue;

    int wev0 = run->wait_ev[0], wev1 = run->wait_ev[1];
    int wtg0 = run->wait_target[0], wtg1 = run->wait_target[1];
    r_kernel<<<1, 1, 0, stream>>>(*st, run->op, run->op_index, run->a0, run->a1, run->a2, run->a3,
                                  run->worker_mask, run->payload_seed, run->output_event,
                                  run->output_increment, run->wait_count, wev0, wev1, wtg0, wtg1);
    cudaError_t e = cudaPeekAtLastError(); if (e) return e;

    r_hash_kernel<<<1, 1, 0, stream>>>(*st, out->counters, out->fleet_event_hash,
                                       out->task_state_hash, out->queue_hash, out->counter_hash,
                                       out->running_hash, out->clock_out, out->event_seq_out);
    e = cudaPeekAtLastError(); if (e) return e;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return r_reset((MkfReferenceState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    MkfReferenceState* st = (MkfReferenceState*)state;
    if (st->tasks) cudaFree(st->tasks);
    if (st->ready_pool) cudaFree(st->ready_pool);
    if (st->ready_head) cudaFree(st->ready_head);
    if (st->ready_count) cudaFree(st->ready_count);
    if (st->rr_cursor) cudaFree(st->rr_cursor);
    if (st->mbox_pool) cudaFree(st->mbox_pool);
    if (st->mbox_head) cudaFree(st->mbox_head);
    if (st->mbox_count) cudaFree(st->mbox_count);
    if (st->worker_running) cudaFree(st->worker_running);
    if (st->running) cudaFree(st->running);
    if (st->running_count) cudaFree(st->running_count);
    if (st->gcounter) cudaFree(st->gcounter);
    if (st->local_counter) cudaFree(st->local_counter);
    if (st->local_target) cudaFree(st->local_target);
    if (st->local_used) cudaFree(st->local_used);
    if (st->local_fence) cudaFree(st->local_fence);
    if (st->part_dispatched) cudaFree(st->part_dispatched);
    if (st->counters) cudaFree(st->counters);
    if (st->scalars) cudaFree(st->scalars);
    free(st);
}
