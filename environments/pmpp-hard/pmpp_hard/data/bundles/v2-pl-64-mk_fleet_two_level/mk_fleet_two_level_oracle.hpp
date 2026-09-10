// file: mk_fleet_two_level_oracle.hpp
//
// Host-side golden model for T64 (MK5). INDEPENDENT third implementation: a plain
// C++ event-driven simulation with std::vector / std::deque containers. Neither
// reference.cu nor naive.cu shares any code with this file.

#ifndef MK_FLEET_TWO_LEVEL_ORACLE_HPP_
#define MK_FLEET_TWO_LEVEL_ORACLE_HPP_

#include "mk_fleet_two_level_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <deque>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct MkfOracleResult {
    int64_t counters[MKF_COUNTER_COUNT] = {0};
    uint64_t fleet_event_hash = 0;
    uint64_t task_state_hash = 0;
    uint64_t queue_hash = 0;
    uint64_t counter_hash = 0;
    uint64_t running_hash = 0;
    uint64_t clock_out = 0;
    uint64_t event_seq_out = 0;
};

static inline uint64_t mkf_o_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= 1099511628211ULL; return h;
}
static inline void mkf_o_fnv(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = mkf_o_fnv_byte(v, b[i]);
    *h = v;
}

struct MkfOWait { uint32_t event_id; uint64_t target; };

struct MkfOTask {
    bool used = false;
    uint64_t task_id = 0;
    uint64_t task_seq = 0;
    uint8_t scope = 0;
    uint32_t home_chiplet = 0;
    uint64_t worker_mask = 0;
    uint64_t duration = 0;
    uint64_t payload_seed = 0;
    uint8_t status = MKF_ST_WAITING;
    std::vector<MkfOWait> waits;
    uint32_t output_event = 0;
    uint64_t output_increment = 0;
    uint32_t expected_local_parts = 0;
    uint32_t expected_global_parts = 0;
    uint64_t ready_seq_or_zero = 0;
    uint64_t result_hash = 0;
    uint64_t global_parts_done = 0;  // DEVICE: chiplets that issued global inc
};

// One ready descriptor in a chiplet's queue.
struct MkfOReady {
    uint64_t task_id;
    uint64_t ready_seq;
    uint32_t chiplet;  // descriptor's chiplet (DEVICE part)
};

// A mailbox copy / running copy.
struct MkfOCopy {
    uint64_t task_id;
    uint8_t scope;
    uint32_t home_chiplet;
    uint32_t chiplet;   // chiplet this copy executes on
    uint32_t worker;    // global worker id
    uint64_t due_clock;
    uint32_t output_event;
    uint64_t output_increment;
    uint64_t duration;
};

// Local L2 counter entry keyed by (chiplet, task_id).
struct MkfOLocal {
    uint32_t chiplet;
    uint64_t task_id;
    uint64_t counter;
    uint64_t target;
    bool fence_issued;
};

struct MkfOracle {
    MkfProblemSpec spec{};
    int chiplet_count = 0, wpc = 0, total_workers = 0;
    int max_tasks = 0, max_events = 0, max_ready = 0, max_wq = 0, max_running = 0;

    uint64_t clock = 0, event_seq = 0;
    uint64_t task_seq_next = 1, dispatch_seq_next = 1;
    uint64_t local_count_seq_next = 1, global_count_seq_next = 1;
    uint64_t ready_seq_next = 1;
    uint64_t last_local_seq = 0, last_global_seq = 0;

    std::vector<MkfOTask> tasks;                 // [max_tasks]
    std::vector<std::deque<MkfOReady>> ready;    // [chiplet]
    std::vector<int> rr_cursor;                  // [chiplet] local worker cursor
    std::vector<std::deque<MkfOCopy>> mailbox;   // [worker]
    std::vector<bool> worker_running;            // [worker]
    std::vector<MkfOCopy> running;               // running copies (unordered; sorted on use)
    std::vector<int64_t> gcounter;               // [max_events]
    // local counters keyed by (chiplet,task); kept in a flat list in init order.
    std::vector<MkfOLocal> locals;
    // DEVICE: which (task,chiplet) parts have been dispatched.
    std::map<std::pair<uint64_t,uint32_t>, bool> part_dispatched;

    int64_t counters[MKF_COUNTER_COUNT];
    uint64_t run_event_hash;

    void init(const MkfProblemSpec& s) {
        spec = s;
        chiplet_count = s.chiplet_count;
        wpc = s.workers_per_chiplet;
        total_workers = chiplet_count * wpc;
        max_tasks = s.max_tasks;
        max_events = s.max_events;
        max_ready = s.max_ready_per_chiplet;
        max_wq = s.max_worker_queue;
        max_running = s.max_running;
        reset();
    }

    void reset() {
        clock = 0; event_seq = 0;
        task_seq_next = 1; dispatch_seq_next = 1;
        local_count_seq_next = 1; global_count_seq_next = 1;
        ready_seq_next = 1; last_local_seq = 0; last_global_seq = 0;
        tasks.assign((size_t)max_tasks, MkfOTask());
        ready.assign((size_t)chiplet_count, std::deque<MkfOReady>());
        rr_cursor.assign((size_t)chiplet_count, 0);
        mailbox.assign((size_t)total_workers, std::deque<MkfOCopy>());
        worker_running.assign((size_t)total_workers, false);
        running.clear();
        gcounter.assign((size_t)max_events, 0);
        locals.clear();
        part_dispatched.clear();
        for (int i = 0; i < MKF_COUNTER_COUNT; ++i) counters[i] = 0;
    }

    int chiplet_of_worker(int w) const { return w / wpc; }
    int local_of_worker(int w) const { return w % wpc; }
    int gworker(int chiplet, int local) const { return chiplet * wpc + local; }

    int find_task(uint64_t id) const {
        for (int i = 0; i < max_tasks; ++i)
            if (tasks[(size_t)i].used && tasks[(size_t)i].task_id == id) return i;
        return -1;
    }

    MkfOLocal* find_local(uint32_t chiplet, uint64_t task_id) {
        for (auto& l : locals)
            if (l.chiplet == chiplet && l.task_id == task_id) return &l;
        return nullptr;
    }

    // ---- event emission ----
    void emit(uint8_t kind, uint32_t chiplet_or, uint32_t worker_or,
              uint64_t task_or, uint32_t event_or, uint64_t value_or,
              uint32_t op_index) {
        uint64_t es = event_seq;
        uint64_t* h = &run_event_hash;
        mkf_o_fnv(h, &kind, 1);
        mkf_o_fnv(h, &es, 8);
        mkf_o_fnv(h, &op_index, 4);
        uint64_t cyc = clock; mkf_o_fnv(h, &cyc, 8);
        mkf_o_fnv(h, &chiplet_or, 4);
        mkf_o_fnv(h, &worker_or, 4);
        mkf_o_fnv(h, &task_or, 8);
        mkf_o_fnv(h, &event_or, 4);
        mkf_o_fnv(h, &value_or, 8);
        event_seq = es + 1;
    }

    bool waits_satisfied(const MkfOTask& t) const {
        for (const MkfOWait& w : t.waits)
            if ((uint64_t)gcounter[(size_t)w.event_id] < w.target) return false;
        return true;
    }

    // enqueue ready descriptors per scope; emit one TASK_READY.
    void enqueue_ready(MkfOTask& t, uint32_t op_index) {
        t.status = MKF_ST_READY;
        if (t.scope == MKF_SCOPE_DEVICE) {
            for (int ch = 0; ch < chiplet_count; ++ch) {
                if (!chiplet_participates(t.worker_mask, ch)) continue;
                MkfOReady r; r.task_id = t.task_id; r.ready_seq = ready_seq_next++;
                r.chiplet = (uint32_t)ch;
                if ((int)ready[(size_t)ch].size() < max_ready) ready[(size_t)ch].push_back(r);
                t.ready_seq_or_zero = r.ready_seq;
            }
        } else {
            int ch = (int)t.home_chiplet;
            MkfOReady r; r.task_id = t.task_id; r.ready_seq = ready_seq_next++;
            r.chiplet = (uint32_t)ch;
            if ((int)ready[(size_t)ch].size() < max_ready) ready[(size_t)ch].push_back(r);
            t.ready_seq_or_zero = r.ready_seq;
        }
        counters[MKF_C_TASK_READY]++;
        emit(MKF_EV_TASK_READY, t.home_chiplet, UINT32_MAX, t.task_id, t.output_event,
             (uint64_t)t.ready_seq_or_zero, op_index);
    }

    bool chiplet_participates(uint64_t mask, int ch) const {
        for (int l = 0; l < wpc; ++l) {
            int w = gworker(ch, l);
            if (w < 64 && (mask >> w) & 1ULL) return true;
        }
        return false;
    }
    int count_home_selected(uint64_t mask, int home) const {
        int c = 0;
        for (int l = 0; l < wpc; ++l) {
            int w = gworker(home, l);
            if (w < 64 && (mask >> w) & 1ULL) c++;
        }
        return c;
    }

    void ready_scan(uint32_t op_index) {
        // scan WAITING tasks by task_seq ascending.
        std::vector<int> idxs;
        for (int i = 0; i < max_tasks; ++i)
            if (tasks[(size_t)i].used && tasks[(size_t)i].status == MKF_ST_WAITING)
                idxs.push_back(i);
        std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
            return tasks[(size_t)a].task_seq < tasks[(size_t)b].task_seq;
        });
        for (int i : idxs) {
            MkfOTask& t = tasks[(size_t)i];
            if (waits_satisfied(t)) {
                counters[MKF_C_TASK_WAIT] += 0;
                enqueue_ready(t, op_index);
            }
        }
    }

    void global_inc(uint32_t event_id, uint64_t amount) {
        gcounter[(size_t)event_id] += (int64_t)amount;
        last_global_seq = global_count_seq_next++;
    }

    void set_result_hash(MkfOTask& t) {
        uint64_t h = 1469598103934665603ULL;
        uint64_t tid = t.task_id, ps = t.payload_seed, du = t.duration;
        uint32_t oe = t.output_event; uint64_t oi = t.output_increment, ck = clock;
        mkf_o_fnv(&h, &tid, 8);
        mkf_o_fnv(&h, &ps, 8);
        mkf_o_fnv(&h, &du, 8);
        mkf_o_fnv(&h, &oe, 4);
        mkf_o_fnv(&h, &oi, 8);
        mkf_o_fnv(&h, &ck, 8);
        t.result_hash = h;
    }

    // ---- operations ----
    void op_submit(const MkfRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        uint64_t tid = (uint64_t)(uint32_t)run.a0;
        int scope = run.a1;
        int home = run.a2;
        int duration = run.a3;
        uint64_t mask = run.worker_mask;
        int oe = run.output_event;
        uint64_t oinc = (uint64_t)(uint32_t)run.output_increment;
        int wc = run.wait_count;

        bool invalid = false;
        if (find_task(tid) >= 0) invalid = true;
        // table full
        int slot = -1;
        for (int i = 0; i < max_tasks; ++i) if (!tasks[(size_t)i].used) { slot = i; break; }
        if (slot < 0) invalid = true;
        if (home < 0 || home >= chiplet_count) invalid = true;
        if (scope < 0 || scope >= MKF_SCOPE_COUNT) invalid = true;
        if (oe < 0 || oe >= max_events) invalid = true;
        if (wc < 0 || wc > MKF_MAX_WAITS) invalid = true;
        if (!invalid) {
            for (int k = 0; k < wc; ++k)
                if (run.wait_ev[k] < 0 || run.wait_ev[k] >= max_events) invalid = true;
        }
        if (mask == 0) invalid = true;
        if (invalid) {
            counters[MKF_C_INVALID]++;
            emit(MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, tid, UINT32_MAX, UINT64_MAX, op_index);
            return;
        }

        MkfOTask& t = tasks[(size_t)slot];
        t = MkfOTask();
        t.used = true;
        t.task_id = tid;
        t.task_seq = task_seq_next++;
        t.scope = (uint8_t)scope;
        t.home_chiplet = (uint32_t)home;
        t.worker_mask = mask;
        t.duration = (uint64_t)(uint32_t)duration;
        t.payload_seed = (uint64_t)(uint32_t)run.payload_seed;
        t.output_event = (uint32_t)oe;
        t.output_increment = oinc;
        for (int k = 0; k < wc; ++k) {
            MkfOWait w; w.event_id = (uint32_t)run.wait_ev[k];
            w.target = (uint64_t)(uint32_t)run.wait_target[k];
            t.waits.push_back(w);
        }
        // expected parts by scope
        if (scope == MKF_SCOPE_WAVE || scope == MKF_SCOPE_CU) {
            t.expected_local_parts = 0; t.expected_global_parts = 1;
        } else if (scope == MKF_SCOPE_CHIPLET) {
            t.expected_local_parts = (uint32_t)count_home_selected(mask, home);
            t.expected_global_parts = 1;
        } else { // DEVICE
            int pc = 0;
            for (int ch = 0; ch < chiplet_count; ++ch)
                if (chiplet_participates(mask, ch)) pc++;
            t.expected_local_parts = (uint32_t)(pc * wpc);
            t.expected_global_parts = (uint32_t)pc;
        }
        t.global_parts_done = 0;

        if (waits_satisfied(t)) {
            enqueue_ready(t, op_index);
        } else {
            t.status = MKF_ST_WAITING;
            counters[MKF_C_TASK_WAIT]++;
            emit(MKF_EV_TASK_WAIT, t.home_chiplet, UINT32_MAX, t.task_id, t.output_event,
                 UINT64_MAX, op_index);
        }
    }

    // returns true if a ready descriptor at the front is stale and was popped
    bool front_stale(int chiplet) {
        std::deque<MkfOReady>& q = ready[(size_t)chiplet];
        if (q.empty()) return false;
        const MkfOReady& r = q.front();
        int idx = find_task(r.task_id);
        if (idx < 0) { q.pop_front(); return true; }
        MkfOTask& t = tasks[(size_t)idx];
        if (t.status == MKF_ST_CANCELLED) { q.pop_front(); return true; }
        if (t.scope == MKF_SCOPE_DEVICE) {
            auto key = std::make_pair(r.task_id, r.chiplet);
            if (part_dispatched.count(key) && part_dispatched[key]) { q.pop_front(); return true; }
            // for DEVICE the relevant readiness is "not yet dispatched part"; status may be
            // DISPATCHED for other parts which is fine. Only stale if cancelled/missing/part done.
            return false;
        } else {
            if (t.status != MKF_ST_READY) { q.pop_front(); return true; }
            return false;
        }
    }

    void op_sched(const MkfRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        int chiplet = run.a0;
        int limit = run.a1;
        if (chiplet < 0 || chiplet >= chiplet_count || limit == 0) {
            counters[MKF_C_INVALID]++;
            emit(MKF_EV_INVALID, (uint32_t)(chiplet >= 0 ? chiplet : -1), UINT32_MAX,
                 UINT64_MAX, UINT32_MAX, UINT64_MAX, op_index);
            return;
        }
        int done = 0;
        while (done < limit) {
            while (front_stale(chiplet)) { /* pop stale */ }
            std::deque<MkfOReady>& q = ready[(size_t)chiplet];
            if (q.empty()) {
                counters[MKF_C_SCHED_IDLE]++;
                emit(MKF_EV_SCHED_IDLE, (uint32_t)chiplet, UINT32_MAX, UINT64_MAX,
                     UINT32_MAX, UINT64_MAX, op_index);
                break;
            }
            MkfOReady r = q.front();
            int idx = find_task(r.task_id);
            MkfOTask& t = tasks[(size_t)idx];

            if (t.scope == MKF_SCOPE_WAVE || t.scope == MKF_SCOPE_CU) {
                int home = (int)t.home_chiplet;
                int chosen_local = -1;
                for (int s = 0; s < wpc; ++s) {
                    int lw = (rr_cursor[(size_t)home] + s) % wpc;
                    int w = gworker(home, lw);
                    if (w >= 64 || !((t.worker_mask >> w) & 1ULL)) continue;
                    if ((int)mailbox[(size_t)w].size() < max_wq) { chosen_local = lw; break; }
                }
                if (chosen_local < 0) {
                    counters[MKF_C_DISPATCH_STALL]++;
                    emit(MKF_EV_DISPATCH_STALL, (uint32_t)chiplet, UINT32_MAX, t.task_id,
                         t.output_event, UINT64_MAX, op_index);
                    break;
                }
                int w = gworker(home, chosen_local);
                MkfOCopy cp; cp.task_id = t.task_id; cp.scope = t.scope; cp.home_chiplet = t.home_chiplet;
                cp.chiplet = (uint32_t)home; cp.worker = (uint32_t)w; cp.due_clock = 0;
                cp.output_event = t.output_event; cp.output_increment = t.output_increment;
                cp.duration = t.duration;
                mailbox[(size_t)w].push_back(cp);
                rr_cursor[(size_t)home] = (chosen_local + 1) % wpc;
                t.status = MKF_ST_DISPATCHED;
                dispatch_seq_next++;
                counters[MKF_C_DISPATCH_SINGLE]++;
                emit(MKF_EV_DISPATCH_SINGLE, (uint32_t)home, (uint32_t)w, t.task_id,
                     t.output_event, UINT64_MAX, op_index);
                q.pop_front();
                done++;
            } else if (t.scope == MKF_SCOPE_CHIPLET) {
                int home = (int)t.home_chiplet;
                // collect selected workers
                std::vector<int> sel;
                for (int l = 0; l < wpc; ++l) {
                    int w = gworker(home, l);
                    if (w < 64 && ((t.worker_mask >> w) & 1ULL)) sel.push_back(w);
                }
                bool all_cap = true;
                for (int w : sel) if ((int)mailbox[(size_t)w].size() >= max_wq) all_cap = false;
                if (!all_cap || sel.empty()) {
                    counters[MKF_C_DISPATCH_STALL]++;
                    emit(MKF_EV_DISPATCH_STALL, (uint32_t)chiplet, UINT32_MAX, t.task_id,
                         t.output_event, UINT64_MAX, op_index);
                    break;
                }
                MkfOLocal* lc = find_local((uint32_t)home, t.task_id);
                if (!lc) { MkfOLocal nl; nl.chiplet=(uint32_t)home; nl.task_id=t.task_id;
                    nl.counter=0; nl.target=sel.size(); nl.fence_issued=false; locals.push_back(nl); }
                else { lc->counter = 0; lc->target = sel.size(); lc->fence_issued = false; }
                for (int w : sel) {
                    MkfOCopy cp; cp.task_id=t.task_id; cp.scope=t.scope; cp.home_chiplet=t.home_chiplet;
                    cp.chiplet=(uint32_t)home; cp.worker=(uint32_t)w; cp.due_clock=0;
                    cp.output_event=t.output_event; cp.output_increment=t.output_increment;
                    cp.duration=t.duration;
                    mailbox[(size_t)w].push_back(cp);
                    counters[MKF_C_DISPATCH_CHIPLET_PART]++;
                    emit(MKF_EV_DISPATCH_CHIPLET_PART, (uint32_t)home, (uint32_t)w, t.task_id,
                         t.output_event, UINT64_MAX, op_index);
                }
                t.status = MKF_ST_DISPATCHED;
                dispatch_seq_next++;
                q.pop_front();
                done++;
            } else { // DEVICE
                int ch = (int)r.chiplet;
                std::vector<int> allw;
                for (int l = 0; l < wpc; ++l) allw.push_back(gworker(ch, l));
                bool all_cap = true;
                for (int w : allw) if ((int)mailbox[(size_t)w].size() >= max_wq) all_cap = false;
                if (!all_cap) {
                    counters[MKF_C_DISPATCH_STALL]++;
                    emit(MKF_EV_DISPATCH_STALL, (uint32_t)ch, UINT32_MAX, t.task_id,
                         t.output_event, UINT64_MAX, op_index);
                    break;
                }
                MkfOLocal* lc = find_local((uint32_t)ch, t.task_id);
                if (!lc) { MkfOLocal nl; nl.chiplet=(uint32_t)ch; nl.task_id=t.task_id;
                    nl.counter=0; nl.target=(uint64_t)wpc; nl.fence_issued=false; locals.push_back(nl); }
                else { lc->counter=0; lc->target=(uint64_t)wpc; lc->fence_issued=false; }
                for (int w : allw) {
                    MkfOCopy cp; cp.task_id=t.task_id; cp.scope=t.scope; cp.home_chiplet=t.home_chiplet;
                    cp.chiplet=(uint32_t)ch; cp.worker=(uint32_t)w; cp.due_clock=0;
                    cp.output_event=t.output_event; cp.output_increment=t.output_increment;
                    cp.duration=t.duration;
                    mailbox[(size_t)w].push_back(cp);
                    counters[MKF_C_DISPATCH_DEVICE_PART]++;
                    emit(MKF_EV_DISPATCH_DEVICE_PART, (uint32_t)ch, (uint32_t)w, t.task_id,
                         t.output_event, UINT64_MAX, op_index);
                }
                part_dispatched[std::make_pair(t.task_id, (uint32_t)ch)] = true;
                if (t.status == MKF_ST_READY) t.status = MKF_ST_DISPATCHED;
                dispatch_seq_next++;
                q.pop_front();
                done++;
            }
        }
    }

    void op_worker(const MkfRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        int worker = run.a0;
        int limit = run.a1;
        if (worker < 0 || worker >= total_workers || limit == 0) {
            counters[MKF_C_INVALID]++;
            emit(MKF_EV_INVALID, UINT32_MAX, (uint32_t)(worker >= 0 ? worker : -1),
                 UINT64_MAX, UINT32_MAX, UINT64_MAX, op_index);
            return;
        }
        int done = 0;
        while (done < limit) {
            if (worker_running[(size_t)worker]) {
                counters[MKF_C_WORKER_BUSY]++;
                emit(MKF_EV_WORKER_BUSY, (uint32_t)chiplet_of_worker(worker), (uint32_t)worker,
                     UINT64_MAX, UINT32_MAX, UINT64_MAX, op_index);
                break;
            }
            std::deque<MkfOCopy>& mb = mailbox[(size_t)worker];
            // drop cancelled-task copies at front
            while (!mb.empty()) {
                int idx = find_task(mb.front().task_id);
                if (idx < 0 || tasks[(size_t)idx].status == MKF_ST_CANCELLED) mb.pop_front();
                else break;
            }
            if (mb.empty()) {
                counters[MKF_C_WORKER_IDLE]++;
                emit(MKF_EV_WORKER_IDLE, (uint32_t)chiplet_of_worker(worker), (uint32_t)worker,
                     UINT64_MAX, UINT32_MAX, UINT64_MAX, op_index);
                break;
            }
            MkfOCopy cp = mb.front(); mb.pop_front();
            cp.due_clock = clock + cp.duration;
            worker_running[(size_t)worker] = true;
            running.push_back(cp);
            int idx = find_task(cp.task_id);
            if (idx >= 0 && (cp.scope == MKF_SCOPE_WAVE || cp.scope == MKF_SCOPE_CU)) {
                if (tasks[(size_t)idx].status == MKF_ST_DISPATCHED)
                    tasks[(size_t)idx].status = MKF_ST_RUNNING;
            }
            counters[MKF_C_WORKER_START_TASK]++;
            emit(MKF_EV_WORKER_START_TASK, cp.chiplet, (uint32_t)worker, cp.task_id,
                 cp.output_event, cp.due_clock, op_index);
            done++;
        }
    }

    static bool running_less(const MkfOCopy& a, const MkfOCopy& b) {
        if (a.due_clock != b.due_clock) return a.due_clock < b.due_clock;
        if (a.chiplet != b.chiplet) return a.chiplet < b.chiplet;
        if (a.worker != b.worker) return a.worker < b.worker;
        return a.task_id < b.task_id;
    }

    void op_advance(const MkfRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        int delta = run.a0;
        int max_fin = run.a1;
        if (delta < 0) {
            counters[MKF_C_INVALID]++;
            emit(MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, UINT64_MAX, UINT32_MAX,
                 UINT64_MAX, op_index);
            return;
        }
        clock += (uint64_t)(uint32_t)delta;
        int fin = 0;
        while (fin < max_fin) {
            // pick the minimum eligible running copy by canonical order
            int best = -1;
            for (int i = 0; i < (int)running.size(); ++i) {
                if (running[(size_t)i].due_clock > clock) continue;
                if (best < 0 || running_less(running[(size_t)i], running[(size_t)best])) best = i;
            }
            if (best < 0) break;
            MkfOCopy cp = running[(size_t)best];
            running.erase(running.begin() + best);
            worker_running[(size_t)cp.worker] = false;

            counters[MKF_C_WORKER_FINISH]++;
            emit(MKF_EV_WORKER_FINISH, cp.chiplet, cp.worker, cp.task_id, cp.output_event,
                 cp.due_clock, op_index);

            int idx = find_task(cp.task_id);
            bool cancelled = (idx < 0 || tasks[(size_t)idx].status == MKF_ST_CANCELLED);
            if (cancelled) {
                counters[MKF_C_CANCELLED_FINISH]++;
                emit(MKF_EV_CANCELLED_FINISH, cp.chiplet, cp.worker, cp.task_id,
                     cp.output_event, UINT64_MAX, op_index);
                fin++;
                continue;
            }
            MkfOTask& t = tasks[(size_t)idx];
            if (cp.scope == MKF_SCOPE_WAVE || cp.scope == MKF_SCOPE_CU) {
                global_inc(cp.output_event, cp.output_increment);
                counters[MKF_C_GLOBAL_ATOMIC_INC]++;
                emit(MKF_EV_GLOBAL_ATOMIC_INC, cp.chiplet, cp.worker, cp.task_id,
                     cp.output_event, cp.output_increment, op_index);
                t.status = MKF_ST_GLOBAL_DONE;
                set_result_hash(t);
                ready_scan(op_index);
            } else if (cp.scope == MKF_SCOPE_CHIPLET) {
                MkfOLocal* lc = find_local(cp.chiplet, cp.task_id);
                if (lc) {
                    lc->counter += 1;
                    last_local_seq = local_count_seq_next++;
                    counters[MKF_C_LOCAL_L2_INC]++;
                    emit(MKF_EV_LOCAL_L2_INC, cp.chiplet, cp.worker, cp.task_id,
                         cp.output_event, lc->counter, op_index);
                    if (lc->counter >= lc->target && !lc->fence_issued) {
                        lc->fence_issued = true;
                        counters[MKF_C_L2_FENCE]++;
                        emit(MKF_EV_L2_FENCE, cp.chiplet, UINT32_MAX, cp.task_id,
                             cp.output_event, lc->counter, op_index);
                        global_inc(cp.output_event, cp.output_increment);
                        counters[MKF_C_GLOBAL_ATOMIC_INC]++;
                        emit(MKF_EV_GLOBAL_ATOMIC_INC, cp.chiplet, UINT32_MAX, cp.task_id,
                             cp.output_event, cp.output_increment, op_index);
                        t.status = MKF_ST_GLOBAL_DONE;
                        set_result_hash(t);
                        ready_scan(op_index);
                    }
                }
            } else { // DEVICE
                MkfOLocal* lc = find_local(cp.chiplet, cp.task_id);
                if (lc) {
                    lc->counter += 1;
                    last_local_seq = local_count_seq_next++;
                    counters[MKF_C_LOCAL_L2_INC]++;
                    emit(MKF_EV_LOCAL_L2_INC, cp.chiplet, cp.worker, cp.task_id,
                         cp.output_event, lc->counter, op_index);
                    if (lc->counter >= lc->target && !lc->fence_issued) {
                        lc->fence_issued = true;
                        counters[MKF_C_L2_FENCE]++;
                        emit(MKF_EV_L2_FENCE, cp.chiplet, UINT32_MAX, cp.task_id,
                             cp.output_event, lc->counter, op_index);
                        global_inc(cp.output_event, cp.output_increment);
                        counters[MKF_C_GLOBAL_ATOMIC_INC]++;
                        emit(MKF_EV_GLOBAL_ATOMIC_INC, cp.chiplet, UINT32_MAX, cp.task_id,
                             cp.output_event, cp.output_increment, op_index);
                        t.global_parts_done += 1;
                        if (t.global_parts_done >= t.expected_global_parts) {
                            t.status = MKF_ST_GLOBAL_DONE;
                            set_result_hash(t);
                        } else {
                            t.status = MKF_ST_LOCAL_DONE;
                        }
                        ready_scan(op_index);
                    }
                }
            }
            fin++;
        }
    }

    void op_event_force(const MkfRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        int ev = run.a0;
        uint64_t amount = (uint64_t)(uint32_t)run.a1;
        if (ev < 0 || ev >= max_events) {
            counters[MKF_C_INVALID]++;
            emit(MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, UINT64_MAX,
                 (uint32_t)(ev >= 0 ? ev : -1), UINT64_MAX, op_index);
            return;
        }
        global_inc((uint32_t)ev, amount);
        counters[MKF_C_EVENT_FORCE]++;
        emit(MKF_EV_EVENT_FORCE, UINT32_MAX, UINT32_MAX, UINT64_MAX, (uint32_t)ev,
             amount, op_index);
        ready_scan(op_index);
    }

    void op_cancel(const MkfRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        uint64_t tid = (uint64_t)(uint32_t)run.a0;
        int idx = find_task(tid);
        if (idx < 0) {
            counters[MKF_C_INVALID]++;
            emit(MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, tid, UINT32_MAX, UINT64_MAX, op_index);
            return;
        }
        uint8_t st = tasks[(size_t)idx].status;
        if (st == MKF_ST_GLOBAL_DONE || st == MKF_ST_CANCELLED) {
            counters[MKF_C_INVALID]++;
            emit(MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, tid, UINT32_MAX, UINT64_MAX, op_index);
            return;
        }
        tasks[(size_t)idx].status = MKF_ST_CANCELLED;
        counters[MKF_C_TASK_CANCEL]++;
        emit(MKF_EV_TASK_CANCEL, tasks[(size_t)idx].home_chiplet, UINT32_MAX, tid,
             tasks[(size_t)idx].output_event, UINT64_MAX, op_index);
    }

    // ---- hashing ----
    uint64_t hash_tasks() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<int> idxs;
        for (int i = 0; i < max_tasks; ++i) if (tasks[(size_t)i].used) idxs.push_back(i);
        std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
            return tasks[(size_t)a].task_id < tasks[(size_t)b].task_id;
        });
        for (int i : idxs) {
            const MkfOTask& t = tasks[(size_t)i];
            uint64_t tid=t.task_id, ts=t.task_seq, mask=t.worker_mask, du=t.duration;
            uint8_t sc=t.scope, st=t.status;
            uint32_t hc=t.home_chiplet, el=t.expected_local_parts, eg=t.expected_global_parts;
            uint64_t rs=t.ready_seq_or_zero, rh=t.result_hash;
            mkf_o_fnv(&h, &tid, 8);
            mkf_o_fnv(&h, &ts, 8);
            mkf_o_fnv(&h, &sc, 1);
            mkf_o_fnv(&h, &hc, 4);
            mkf_o_fnv(&h, &mask, 8);
            mkf_o_fnv(&h, &du, 8);
            mkf_o_fnv(&h, &st, 1);
            mkf_o_fnv(&h, &el, 4);
            mkf_o_fnv(&h, &eg, 4);
            mkf_o_fnv(&h, &rs, 8);
            mkf_o_fnv(&h, &rh, 8);
        }
        return h;
    }

    uint64_t hash_queues() const {
        uint64_t h = 1469598103934665603ULL;
        for (int ch = 0; ch < chiplet_count; ++ch) {
            uint8_t kind = 0; uint32_t c = (uint32_t)ch;
            mkf_o_fnv(&h, &kind, 1);
            mkf_o_fnv(&h, &c, 4);
            uint64_t pos = 0;
            for (const MkfOReady& r : ready[(size_t)ch]) {
                mkf_o_fnv(&h, &pos, 8);
                mkf_o_fnv(&h, &r.task_id, 8);
                mkf_o_fnv(&h, &r.ready_seq, 8);
                uint32_t rc = r.chiplet; mkf_o_fnv(&h, &rc, 4);
                pos++;
            }
        }
        for (int w = 0; w < total_workers; ++w) {
            uint8_t kind = 1; uint32_t ww = (uint32_t)w;
            mkf_o_fnv(&h, &kind, 1);
            mkf_o_fnv(&h, &ww, 4);
            uint64_t pos = 0;
            for (const MkfOCopy& cp : mailbox[(size_t)w]) {
                mkf_o_fnv(&h, &pos, 8);
                mkf_o_fnv(&h, &cp.task_id, 8);
                pos++;
            }
        }
        return h;
    }

    uint64_t hash_counters() const {
        uint64_t h = 1469598103934665603ULL;
        // local counters by chiplet asc then task_id asc
        std::vector<int> order;
        for (int i = 0; i < (int)locals.size(); ++i) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (locals[(size_t)a].chiplet != locals[(size_t)b].chiplet)
                return locals[(size_t)a].chiplet < locals[(size_t)b].chiplet;
            return locals[(size_t)a].task_id < locals[(size_t)b].task_id;
        });
        for (int i : order) {
            const MkfOLocal& l = locals[(size_t)i];
            uint8_t kind = 0;
            uint32_t ch = l.chiplet; uint64_t tid = l.task_id, cnt = l.counter, tg = l.target;
            uint64_t lls = last_local_seq;
            mkf_o_fnv(&h, &kind, 1);
            mkf_o_fnv(&h, &ch, 4);
            mkf_o_fnv(&h, &tid, 8);
            mkf_o_fnv(&h, &cnt, 8);
            mkf_o_fnv(&h, &tg, 8);
            mkf_o_fnv(&h, &lls, 8);
        }
        for (int e = 0; e < max_events; ++e) {
            uint8_t kind = 1; uint32_t ev = (uint32_t)e;
            uint64_t cnt = (uint64_t)gcounter[(size_t)e];
            uint64_t lgs = last_global_seq;
            mkf_o_fnv(&h, &kind, 1);
            mkf_o_fnv(&h, &ev, 4);
            mkf_o_fnv(&h, &cnt, 8);
            mkf_o_fnv(&h, &lgs, 8);
        }
        return h;
    }

    uint64_t hash_running() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<MkfOCopy> sorted = running;
        std::sort(sorted.begin(), sorted.end(), running_less);
        for (const MkfOCopy& cp : sorted) {
            uint64_t dc = cp.due_clock, tid = cp.task_id;
            uint32_t ch = cp.chiplet, w = cp.worker;
            uint8_t sc = cp.scope;
            mkf_o_fnv(&h, &dc, 8);
            mkf_o_fnv(&h, &ch, 4);
            mkf_o_fnv(&h, &w, 4);
            mkf_o_fnv(&h, &tid, 8);
            mkf_o_fnv(&h, &sc, 1);
        }
        return h;
    }

    void step_once(const MkfRunSpec& run, MkfOracleResult* out) {
        run_event_hash = 1469598103934665603ULL;
        switch (run.op) {
            case MKF_OP_SUBMIT: op_submit(run); break;
            case MKF_OP_SCHED: op_sched(run); break;
            case MKF_OP_WORKER: op_worker(run); break;
            case MKF_OP_ADVANCE: op_advance(run); break;
            case MKF_OP_EVENT_FORCE: op_event_force(run); break;
            case MKF_OP_CANCEL: op_cancel(run); break;
            default:
                counters[MKF_C_INVALID]++;
                emit(MKF_EV_INVALID, UINT32_MAX, UINT32_MAX, UINT64_MAX, UINT32_MAX,
                     UINT64_MAX, (uint32_t)run.op_index);
                break;
        }
        for (int i = 0; i < MKF_COUNTER_COUNT; ++i) out->counters[i] = counters[i];
        out->fleet_event_hash = run_event_hash;
        out->task_state_hash = hash_tasks();
        out->queue_hash = hash_queues();
        out->counter_hash = hash_counters();
        out->running_hash = hash_running();
        out->clock_out = clock;
        out->event_seq_out = event_seq;
    }
};

static inline bool mkf_check(const MkfOracleResult& exp, const MkfOracleResult& got,
                             std::string* err) {
    static const char* names[MKF_COUNTER_COUNT] = {
        "task_ready", "task_wait", "sched_idle", "dispatch_single",
        "dispatch_chiplet_part", "dispatch_device_part", "dispatch_stall",
        "worker_start_task", "worker_busy", "worker_idle", "worker_finish",
        "cancelled_finish", "local_l2_inc", "l2_fence", "global_atomic_inc",
        "event_force", "task_cancel", "invalid_count"};
    for (int i = 0; i < MKF_COUNTER_COUNT; ++i) {
        if (exp.counters[i] != got.counters[i]) {
            if (err) {
                std::ostringstream o;
                o << "counter " << names[i] << " got " << got.counters[i]
                  << " expected " << exp.counters[i];
                *err = o.str();
            }
            return false;
        }
    }
#define MKF_CK(field) \
    if (exp.field != got.field) { \
        if (err) { std::ostringstream o; o << #field << " got 0x" << std::hex \
            << got.field << " expected 0x" << exp.field; *err = o.str(); } \
        return false; }
    MKF_CK(fleet_event_hash);
    MKF_CK(task_state_hash);
    MKF_CK(queue_hash);
    MKF_CK(counter_hash);
    MKF_CK(running_hash);
    MKF_CK(clock_out);
    MKF_CK(event_seq_out);
#undef MKF_CK
    return true;
}

#endif  // MK_FLEET_TWO_LEVEL_ORACLE_HPP_
