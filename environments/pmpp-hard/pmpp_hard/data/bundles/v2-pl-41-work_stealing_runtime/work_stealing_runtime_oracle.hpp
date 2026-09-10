// file: work_stealing_runtime_oracle.hpp

#ifndef WORK_STEALING_RUNTIME_ORACLE_HPP_
#define WORK_STEALING_RUNTIME_ORACLE_HPP_

#include "work_stealing_runtime_common.h"

#include <stdint.h>
#include <stddef.h>

#include <deque>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ------------------------------------------------------------------ FNV-1a-64
struct WsrFnv {
    uint64_t h = 1469598103934665603ULL;
    void byte(uint8_t b) {
        h ^= static_cast<uint64_t>(b);
        h *= 1099511628211ULL;
    }
    void bytes(const void* p, size_t n) {
        const uint8_t* q = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) byte(q[i]);
    }
    void u8(uint8_t v) { bytes(&v, sizeof(v)); }
    void u32(uint32_t v) { bytes(&v, sizeof(v)); }
    void u64(uint64_t v) { bytes(&v, sizeof(v)); }
};

struct WsrExpected {
    std::vector<int64_t> counts;
    int32_t op_index = 0;
    uint64_t clock = 0;
    uint64_t event_seq = 0;
    uint64_t sched_event_hash = 0;
    uint64_t ready_hash = 0;
    uint64_t running_hash = 0;
    uint64_t blocked_hash = 0;
    uint64_t sleep_hash = 0;
    uint64_t state_checksum = 0;
};

struct WsrHostOutputsView {
    const int64_t* counts;
    const int32_t* op_index_out;
    const uint64_t* clock_out;
    const uint64_t* event_seq_out;
    const uint64_t* sched_event_hash;
    const uint64_t* ready_hash;
    const uint64_t* running_hash;
    const uint64_t* blocked_hash;
    const uint64_t* sleep_hash;
    const uint64_t* state_checksum;
};

// ------------------------------------------------------------------ Oracle
struct WsrOracle {
    struct Task {
        uint64_t task_id = 0;
        int32_t home_worker = 0;
        int32_t priority = 0;       // 0 low, 1 high
        uint64_t remaining_work = 0;
        int32_t state = 0;          // READY_LOCAL/READY_GLOBAL/RUNNING/BLOCKED/SLEEPING
        uint64_t birth_seq = 0;
        uint64_t last_run_seq = 0;
        uint64_t block_seq = 0;
        uint64_t sleep_seq = 0;
        uint64_t wait_key = 0;
        uint64_t wake_tick = 0;
    };

    enum { ST_READY_LOCAL = 0, ST_READY_GLOBAL = 1, ST_RUNNING = 2, ST_BLOCKED = 3, ST_SLEEPING = 4 };

    WsrProblemSpec spec{};

    uint64_t clock = 0;
    uint64_t event_seq = 0;
    int32_t op_index = 0;
    uint64_t sched_hash = 1469598103934665603ULL;

    std::unordered_map<uint64_t, Task> tasks;          // live tasks by id

    // Deques hold task ids; head=front, tail=back.
    std::vector<std::deque<uint64_t>> local_high;
    std::vector<std::deque<uint64_t>> local_low;
    std::deque<uint64_t> global_high;
    std::deque<uint64_t> global_low;
    std::vector<int64_t> running;                      // -1 empty else task_id (store as int64 with -1 sentinel)
    std::vector<uint8_t> running_has;

    // blocked[wait_key] = FIFO of task ids ordered by block_seq.
    std::map<uint64_t, std::deque<uint64_t>> blocked;
    int64_t blocked_total = 0;

    // sleeping index sorted by (wake_tick, sleep_seq, task_id).
    struct SleepKey {
        uint64_t wake_tick, sleep_seq, task_id;
        bool operator<(const SleepKey& o) const {
            if (wake_tick != o.wake_tick) return wake_tick < o.wake_tick;
            if (sleep_seq != o.sleep_seq) return sleep_seq < o.sleep_seq;
            return task_id < o.task_id;
        }
    };
    std::set<SleepKey> sleeping;

    std::vector<int64_t> counts;

    void init(const WsrProblemSpec& s) {
        spec = s;
        local_high.assign((size_t)spec.W, {});
        local_low.assign((size_t)spec.W, {});
        running.assign((size_t)spec.W, -1);
        running_has.assign((size_t)spec.W, 0);
        counts.assign(WSR_COUNT_N, 0);
        reset();
    }

    void reset() {
        clock = 0;
        event_seq = 0;
        op_index = 0;
        sched_hash = 1469598103934665603ULL;
        tasks.clear();
        for (auto& d : local_high) d.clear();
        for (auto& d : local_low) d.clear();
        global_high.clear();
        global_low.clear();
        for (size_t w = 0; w < running.size(); ++w) { running[w] = -1; running_has[w] = 0; }
        blocked.clear();
        blocked_total = 0;
        sleeping.clear();
        for (auto& c : counts) c = 0;
    }

    Task* find(uint64_t id) {
        auto it = tasks.find(id);
        return it == tasks.end() ? nullptr : &it->second;
    }

    std::deque<uint64_t>& local_q(int w, int prio) { return prio ? local_high[(size_t)w] : local_low[(size_t)w]; }
    std::deque<uint64_t>& global_q(int prio) { return prio ? global_high : global_low; }

    // -------------------------------------------------------------- event emit
    void emit(uint8_t kind, int32_t worker, uint64_t task_id,
              uint64_t aux0, uint64_t aux1, int prio, uint64_t remaining_after) {
        const uint64_t seq = event_seq;  // this event's seq
        WsrFnv f; f.h = sched_hash;
        f.u8(kind);
        f.u64(seq);
        f.u32(static_cast<uint32_t>(op_index));
        f.u32(worker < 0 ? UINT32_MAX : static_cast<uint32_t>(worker));
        f.u64(task_id);                                 // already MAX sentinel if absent
        f.u64(aux0);
        f.u64(aux1);
        f.u8(prio < 0 ? static_cast<uint8_t>(255) : static_cast<uint8_t>(prio));
        f.u64(remaining_after);
        sched_hash = f.h;
        event_seq += 1;                                 // wraps mod 2^64
    }
    // returns the seq that the NEXT emitted event will use.
    uint64_t peek_seq() const { return event_seq; }

    // -------------------------------------------------------------- placement
    // Append task to home local deque tail; if full -> global tail; if full -> -1.
    // Returns: 0 local, 1 global, -1 fail. Does not emit.
    int place_ready(Task& t) {
        const int w = t.home_worker;
        const int prio = t.priority;
        if ((int)local_q(w, prio).size() < spec.local_cap_per_worker) {
            local_q(w, prio).push_back(t.task_id);
            t.state = ST_READY_LOCAL;
            return 0;
        }
        if ((int)global_q(prio).size() < spec.global_cap) {
            global_q(prio).push_back(t.task_id);
            t.state = ST_READY_GLOBAL;
            return 1;
        }
        return -1;
    }

    // remove a task id from whatever ready queue currently holds it.
    void remove_from_ready_queue(const Task& t) {
        auto try_rm = [&](std::deque<uint64_t>& d) -> bool {
            for (auto it = d.begin(); it != d.end(); ++it) {
                if (*it == t.task_id) { d.erase(it); return true; }
            }
            return false;
        };
        if (try_rm(local_high[(size_t)t.home_worker])) return;
        if (try_rm(local_low[(size_t)t.home_worker])) return;
        for (int w = 0; w < spec.W; ++w) {
            if (try_rm(local_high[(size_t)w])) return;
            if (try_rm(local_low[(size_t)w])) return;
        }
        if (try_rm(global_high)) return;
        try_rm(global_low);
    }

    // -------------------------------------------------------------- operations
    void op_spawn(uint64_t task_id, int home, int prio, uint64_t work) {
        if (home >= spec.W || home < 0 || prio > 1 || prio < 0 || work == 0) {
            counts[WSR_C_INVALID] += 1;
            emit(WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }
        if (find(task_id)) {
            counts[WSR_C_INVALID] += 1;
            emit(WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }
        if ((int64_t)tasks.size() >= spec.max_tasks) {
            counts[WSR_C_SPAWN_REJECT] += 1;
            emit(WSR_EV_SPAWN_REJECT, home, task_id, 0, 0, prio, UINT64_MAX);
            return;
        }
        Task t;
        t.task_id = task_id;
        t.home_worker = home;
        t.priority = prio;
        t.remaining_work = work;
        t.birth_seq = peek_seq();
        // Attempt placement.
        const int placed = place_ready(t);
        if (placed < 0) {
            counts[WSR_C_SPAWN_REJECT] += 1;
            emit(WSR_EV_SPAWN_REJECT, home, task_id, 0, 0, prio, UINT64_MAX);
            return;
        }
        tasks[task_id] = t;
        if (placed == 0) {
            counts[WSR_C_SPAWN_LOCAL] += 1;
            emit(WSR_EV_SPAWN_LOCAL, home, task_id, 0, 0, prio, work);
        } else {
            counts[WSR_C_SPAWN_GLOBAL] += 1;
            emit(WSR_EV_SPAWN_GLOBAL, home, task_id, 0, 0, prio, work);
        }
    }

    int64_t total_local_ready(int w) const {
        return (int64_t)local_high[(size_t)w].size() + (int64_t)local_low[(size_t)w].size();
    }

    // returns selected task id or UINT64_MAX. Sets *from_steal optionally.
    uint64_t pick_local_or_global(int worker) {
        if (!local_high[(size_t)worker].empty()) {
            uint64_t id = local_high[(size_t)worker].back();
            local_high[(size_t)worker].pop_back();
            return id;
        }
        if (!local_low[(size_t)worker].empty()) {
            uint64_t id = local_low[(size_t)worker].back();
            local_low[(size_t)worker].pop_back();
            return id;
        }
        if (!global_high.empty()) {
            uint64_t id = global_high.front();
            global_high.pop_front();
            return id;
        }
        if (!global_low.empty()) {
            uint64_t id = global_low.front();
            global_low.pop_front();
            return id;
        }
        return UINT64_MAX;
    }

    uint64_t pick_local_only(int worker) {
        if (!local_high[(size_t)worker].empty()) {
            uint64_t id = local_high[(size_t)worker].back();
            local_high[(size_t)worker].pop_back();
            return id;
        }
        if (!local_low[(size_t)worker].empty()) {
            uint64_t id = local_low[(size_t)worker].back();
            local_low[(size_t)worker].pop_back();
            return id;
        }
        return UINT64_MAX;
    }

    void do_steal(int thief) {
        int victim = -1;
        int64_t best = 0;
        for (int w = 0; w < spec.W; ++w) {
            if (w == thief) continue;
            int64_t n = total_local_ready(w);
            if (n <= 0) continue;
            if (n > best) { best = n; victim = w; }  // tie -> keep lowest id (first seen)
        }
        if (victim < 0) return;  // nobody to steal from

        // Combined order = victim high head->tail then low head->tail.
        std::vector<uint64_t> combined;
        for (uint64_t id : local_high[(size_t)victim]) combined.push_back(id);
        const size_t high_n = local_high[(size_t)victim].size();
        for (uint64_t id : local_low[(size_t)victim]) combined.push_back(id);

        const int64_t total = (int64_t)combined.size();
        const int64_t take = (total + 1) / 2;  // ceil(total/2)

        // Steal from FRONT of combined; remove from original priority deque.
        for (int64_t i = 0; i < take; ++i) {
            const uint64_t id = combined[(size_t)i];
            const bool is_high = (i < (int64_t)high_n);
            if (is_high) {
                local_high[(size_t)victim].pop_front();
                local_high[(size_t)thief].push_back(id);
            } else {
                local_low[(size_t)victim].pop_front();
                local_low[(size_t)thief].push_back(id);
            }
            Task* t = find(id);
            const int prio = is_high ? 1 : 0;
            counts[WSR_C_STOLEN_TASKS] += 1;
            emit(WSR_EV_STEAL_TASK, thief, id, (uint64_t)victim, 0, prio,
                 t ? t->remaining_work : UINT64_MAX);
        }
    }

    void op_run(int worker, uint64_t quantum) {
        if (worker >= spec.W || worker < 0 || quantum == 0) {
            counts[WSR_C_INVALID] += 1;
            emit(WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }

        uint64_t tid;
        if (running_has[(size_t)worker]) {
            tid = (uint64_t)running[(size_t)worker];
            // already scheduled; no SCHEDULE event.
        } else {
            tid = pick_local_or_global(worker);
            if (tid == UINT64_MAX) {
                do_steal(worker);
                tid = pick_local_only(worker);
            }
            if (tid == UINT64_MAX) {
                counts[WSR_C_IDLE] += 1;
                emit(WSR_EV_IDLE, worker, UINT64_MAX, 0, 0, -1, UINT64_MAX);
                return;
            }
            Task* t = find(tid);
            t->state = ST_RUNNING;
            running[(size_t)worker] = (int64_t)tid;
            running_has[(size_t)worker] = 1;
            counts[WSR_C_SCHEDULED] += 1;
            emit(WSR_EV_SCHEDULE, worker, tid, 0, 0, t->priority, t->remaining_work);
        }

        Task* t = find(tid);
        const uint64_t slice = quantum < t->remaining_work ? quantum : t->remaining_work;
        t->remaining_work -= slice;
        t->last_run_seq = peek_seq();
        counts[WSR_C_RUN_SLICES] += 1;
        emit(WSR_EV_RUN_SLICE, worker, tid, slice, 0, t->priority, t->remaining_work);

        if (t->remaining_work == 0) {
            const int prio = t->priority;
            running[(size_t)worker] = -1;
            running_has[(size_t)worker] = 0;
            tasks.erase(tid);
            counts[WSR_C_COMPLETED] += 1;
            emit(WSR_EV_COMPLETE, worker, tid, 0, 0, prio, 0);
        }
    }

    void op_yield(int worker) {
        if (worker >= spec.W || worker < 0) {
            counts[WSR_C_INVALID] += 1;
            emit(WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }
        if (!running_has[(size_t)worker]) {
            counts[WSR_C_EMPTY_OP] += 1;
            emit(WSR_EV_YIELD_EMPTY, worker, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }
        const uint64_t tid = (uint64_t)running[(size_t)worker];
        Task* t = find(tid);
        running[(size_t)worker] = -1;
        running_has[(size_t)worker] = 0;
        const int placed = place_ready(*t);
        if (placed < 0) {
            const int prio = t->priority;
            const uint64_t rem = t->remaining_work;
            tasks.erase(tid);
            counts[WSR_C_OVERFLOW_CANCELLED] += 1;
            emit(WSR_EV_CANCEL_OVERFLOW, worker, tid, WSR_REASON_YIELD_OVERFLOW, 0, prio, rem);
            return;
        }
        counts[WSR_C_YIELDED] += 1;
        emit(WSR_EV_YIELD, worker, tid, 0, 0, t->priority, t->remaining_work);
    }

    void op_block(int worker, uint64_t wait_key) {
        if (worker >= spec.W || worker < 0) {
            counts[WSR_C_INVALID] += 1;
            emit(WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }
        if (!running_has[(size_t)worker]) {
            counts[WSR_C_EMPTY_OP] += 1;
            emit(WSR_EV_BLOCK_EMPTY, worker, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }
        const uint64_t tid = (uint64_t)running[(size_t)worker];
        Task* t = find(tid);
        running[(size_t)worker] = -1;
        running_has[(size_t)worker] = 0;
        if (blocked_total >= spec.max_blocked) {
            const int prio = t->priority;
            const uint64_t rem = t->remaining_work;
            tasks.erase(tid);
            counts[WSR_C_OVERFLOW_CANCELLED] += 1;
            emit(WSR_EV_CANCEL_OVERFLOW, worker, tid, WSR_REASON_BLOCK_OVERFLOW, 0, prio, rem);
            return;
        }
        t->state = ST_BLOCKED;
        t->wait_key = wait_key;
        t->block_seq = peek_seq();
        blocked[wait_key].push_back(tid);
        blocked_total += 1;
        counts[WSR_C_BLOCKED] += 1;
        emit(WSR_EV_BLOCK, worker, tid, wait_key, 0, t->priority, t->remaining_work);
    }

    void op_sleep(int worker, uint64_t wake_tick) {
        if (worker >= spec.W || worker < 0) {
            counts[WSR_C_INVALID] += 1;
            emit(WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }
        if (!running_has[(size_t)worker]) {
            counts[WSR_C_EMPTY_OP] += 1;
            emit(WSR_EV_SLEEP_EMPTY, worker, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }
        const uint64_t tid = (uint64_t)running[(size_t)worker];
        Task* t = find(tid);
        running[(size_t)worker] = -1;
        running_has[(size_t)worker] = 0;
        if ((int64_t)sleeping.size() >= spec.max_sleeping) {
            const int prio = t->priority;
            const uint64_t rem = t->remaining_work;
            tasks.erase(tid);
            counts[WSR_C_OVERFLOW_CANCELLED] += 1;
            emit(WSR_EV_CANCEL_OVERFLOW, worker, tid, WSR_REASON_SLEEP_OVERFLOW, 0, prio, rem);
            return;
        }
        t->state = ST_SLEEPING;
        t->wake_tick = wake_tick;
        t->sleep_seq = peek_seq();
        sleeping.insert(SleepKey{wake_tick, t->sleep_seq, tid});
        counts[WSR_C_SLEPT] += 1;
        emit(WSR_EV_SLEEP, worker, tid, wake_tick, 0, t->priority, t->remaining_work);
    }

    void op_wake(uint64_t wait_key, int64_t limit) {
        if (limit <= 0) return;  // limit==0 no-op (negative treated as no-op too)
        auto it = blocked.find(wait_key);
        if (it == blocked.end()) return;
        std::deque<uint64_t>& fifo = it->second;
        int64_t done = 0;
        while (done < limit && !fifo.empty()) {
            const uint64_t tid = fifo.front();
            fifo.pop_front();
            Task* t = find(tid);
            const int placed = place_ready(*t);
            if (placed < 0) {
                // global overflow: put back at FRONT with original block_seq, stall.
                fifo.push_front(tid);
                t->state = ST_BLOCKED;
                emit(WSR_EV_WAKE_STALLED, t->home_worker, tid, wait_key, 0, t->priority, t->remaining_work);
                return;
            }
            blocked_total -= 1;
            if (placed == 0) {
                counts[WSR_C_WOKEN_LOCAL] += 1;
                emit(WSR_EV_WAKE_LOCAL, t->home_worker, tid, wait_key, 0, t->priority, t->remaining_work);
            } else {
                counts[WSR_C_WOKEN_GLOBAL] += 1;
                emit(WSR_EV_WAKE_GLOBAL, t->home_worker, tid, wait_key, 0, t->priority, t->remaining_work);
            }
            ++done;
        }
        if (fifo.empty()) blocked.erase(it);
    }

    void op_advance(uint64_t delta) {
        clock = clock + delta;  // wraps mod 2^64
        emit(WSR_EV_CLOCK_ADVANCE, -1, UINT64_MAX, delta, clock, -1, UINT64_MAX);

        // Wake sleeping tasks with wake_tick <= clock in sorted order.
        while (!sleeping.empty()) {
            const SleepKey k = *sleeping.begin();
            if (k.wake_tick > clock) break;
            Task* t = find(k.task_id);
            // tentatively remove from sleeping index
            const int placed = place_ready(*t);
            if (placed < 0) {
                // leave in sleeping index, stall, stop.
                emit(WSR_EV_WAKE_STALLED, t->home_worker, k.task_id, k.wake_tick, 0,
                     t->priority, t->remaining_work);
                return;
            }
            sleeping.erase(sleeping.begin());
            if (placed == 0) {
                counts[WSR_C_WOKEN_LOCAL] += 1;
                emit(WSR_EV_WAKE_LOCAL, t->home_worker, k.task_id, k.wake_tick, 0,
                     t->priority, t->remaining_work);
            } else {
                counts[WSR_C_WOKEN_GLOBAL] += 1;
                emit(WSR_EV_WAKE_GLOBAL, t->home_worker, k.task_id, k.wake_tick, 0,
                     t->priority, t->remaining_work);
            }
        }
    }

    void op_cancel(uint64_t task_id) {
        Task* t = find(task_id);
        if (!t) {
            counts[WSR_C_INVALID] += 1;
            emit(WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }
        const int home = t->home_worker;
        const int prio = t->priority;
        const uint64_t rem = t->remaining_work;
        const int state = t->state;
        if (state == ST_RUNNING) {
            for (int w = 0; w < spec.W; ++w) {
                if (running_has[(size_t)w] && (uint64_t)running[(size_t)w] == task_id) {
                    running[(size_t)w] = -1;
                    running_has[(size_t)w] = 0;
                    break;
                }
            }
        } else if (state == ST_READY_LOCAL || state == ST_READY_GLOBAL) {
            remove_from_ready_queue(*t);
        } else if (state == ST_BLOCKED) {
            auto it = blocked.find(t->wait_key);
            if (it != blocked.end()) {
                std::deque<uint64_t>& d = it->second;
                for (auto jt = d.begin(); jt != d.end(); ++jt) {
                    if (*jt == task_id) { d.erase(jt); blocked_total -= 1; break; }
                }
                if (d.empty()) blocked.erase(it);
            }
        } else if (state == ST_SLEEPING) {
            sleeping.erase(SleepKey{t->wake_tick, t->sleep_seq, task_id});
        }
        tasks.erase(task_id);
        counts[WSR_C_CANCELLED] += 1;
        emit(WSR_EV_CANCEL_EXPLICIT, home, task_id, 0, 0, prio, rem);
    }

    // -------------------------------------------------------------- snapshots
    uint64_t compute_ready_hash() const {
        WsrFnv f;
        for (int w = 0; w < spec.W; ++w) {
            uint64_t pos = 0;
            for (uint64_t id : local_high[(size_t)w]) {
                f.u8(0); f.u32((uint32_t)w); f.u8(1); f.u64(pos++); f.u64(id);
            }
            pos = 0;
            for (uint64_t id : local_low[(size_t)w]) {
                f.u8(0); f.u32((uint32_t)w); f.u8(0); f.u64(pos++); f.u64(id);
            }
        }
        uint64_t pos = 0;
        for (uint64_t id : global_high) {
            f.u8(1); f.u32(UINT32_MAX); f.u8(1); f.u64(pos++); f.u64(id);
        }
        pos = 0;
        for (uint64_t id : global_low) {
            f.u8(1); f.u32(UINT32_MAX); f.u8(0); f.u64(pos++); f.u64(id);
        }
        return f.h;
    }

    uint64_t compute_running_hash() const {
        WsrFnv f;
        for (int w = 0; w < spec.W; ++w) {
            f.u32((uint32_t)w);
            if (running_has[(size_t)w]) {
                const uint64_t id = (uint64_t)running[(size_t)w];
                const auto it = tasks.find(id);
                f.u8(1);
                f.u64(id);
                f.u64(it != tasks.end() ? it->second.remaining_work : UINT64_MAX);
            } else {
                f.u8(0);
                f.u64(UINT64_MAX);
                f.u64(UINT64_MAX);
            }
        }
        return f.h;
    }

    uint64_t compute_blocked_hash() const {
        WsrFnv f;
        for (const auto& kv : blocked) {           // map iterates wait_key ascending
            const uint64_t key = kv.first;
            for (uint64_t id : kv.second) {         // already block_seq order (FIFO append)
                const auto it = tasks.find(id);
                if (it == tasks.end()) continue;
                const Task& t = it->second;
                f.u64(key);
                f.u64(t.block_seq);
                f.u64(id);
                f.u32((uint32_t)t.home_worker);
                f.u8((uint8_t)t.priority);
                f.u64(t.remaining_work);
            }
        }
        return f.h;
    }

    uint64_t compute_sleep_hash() const {
        WsrFnv f;
        for (const SleepKey& k : sleeping) {        // set iterates sorted
            const auto it = tasks.find(k.task_id);
            if (it == tasks.end()) continue;
            const Task& t = it->second;
            f.u64(k.wake_tick);
            f.u64(k.sleep_seq);
            f.u64(k.task_id);
            f.u32((uint32_t)t.home_worker);
            f.u8((uint8_t)t.priority);
            f.u64(t.remaining_work);
        }
        return f.h;
    }

    uint64_t compute_state_checksum(uint64_t rh, uint64_t runh, uint64_t bh, uint64_t sh) const {
        WsrFnv f;
        f.u64(clock);
        f.u64(event_seq);
        f.u32((uint32_t)op_index);
        f.u64(sched_hash);
        f.u64(rh);
        f.u64(runh);
        f.u64(bh);
        f.u64(sh);
        for (int i = 0; i < WSR_COUNT_N; ++i) {
            f.u64((uint64_t)counts[(size_t)i]);
        }
        return f.h;
    }

    void step_once(const WsrRunSpec& run, WsrExpected* exp) {
        switch (run.op_kind) {
            case WSR_OP_SPAWN:
                op_spawn(run.a_task, run.a_worker, run.a_priority, run.a_work);
                break;
            case WSR_OP_RUN:
                op_run(run.a_worker, run.a_work);
                break;
            case WSR_OP_YIELD:
                op_yield(run.a_worker);
                break;
            case WSR_OP_BLOCK:
                op_block(run.a_worker, run.a_key);
                break;
            case WSR_OP_SLEEP:
                op_sleep(run.a_worker, run.a_tick);
                break;
            case WSR_OP_WAKE:
                op_wake(run.a_key, run.a_limit);
                break;
            case WSR_OP_ADVANCE:
                op_advance(run.a_delta);
                break;
            case WSR_OP_CANCEL:
                op_cancel(run.a_task);
                break;
            default:
                break;
        }

        const int32_t this_op = op_index;
        op_index += 1;

        const uint64_t rh = compute_ready_hash();
        const uint64_t runh = compute_running_hash();
        const uint64_t bh = compute_blocked_hash();
        const uint64_t sh = compute_sleep_hash();

        exp->counts = counts;
        exp->op_index = this_op;
        exp->clock = clock;
        exp->event_seq = event_seq;
        exp->sched_event_hash = sched_hash;
        exp->ready_hash = rh;
        exp->running_hash = runh;
        exp->blocked_hash = bh;
        exp->sleep_hash = sh;
        exp->state_checksum = compute_state_checksum(rh, runh, bh, sh);
    }
};

static inline bool wsr_check_outputs(
    const WsrExpected& e,
    const WsrHostOutputsView& g,
    std::string* err) {
    for (int i = 0; i < WSR_COUNT_N; ++i) {
        if (g.counts[i] != e.counts[(size_t)i]) {
            if (err) {
                std::ostringstream o;
                o << "count[" << i << "] mismatch: got " << g.counts[i]
                  << " expected " << e.counts[(size_t)i];
                *err = o.str();
            }
            return false;
        }
    }
    auto chk64 = [&](const char* nm, uint64_t got, uint64_t exp) -> bool {
        if (got != exp) {
            if (err) {
                std::ostringstream o;
                o << nm << " mismatch: got 0x" << std::hex << got
                  << " expected 0x" << exp;
                *err = o.str();
            }
            return false;
        }
        return true;
    };
    if (g.op_index_out[0] != e.op_index) {
        if (err) { std::ostringstream o; o << "op_index mismatch got " << g.op_index_out[0] << " exp " << e.op_index; *err = o.str(); }
        return false;
    }
    if (!chk64("clock", g.clock_out[0], e.clock)) return false;
    if (!chk64("event_seq", g.event_seq_out[0], e.event_seq)) return false;
    if (!chk64("sched_event_hash", g.sched_event_hash[0], e.sched_event_hash)) return false;
    if (!chk64("ready_hash", g.ready_hash[0], e.ready_hash)) return false;
    if (!chk64("running_hash", g.running_hash[0], e.running_hash)) return false;
    if (!chk64("blocked_hash", g.blocked_hash[0], e.blocked_hash)) return false;
    if (!chk64("sleep_hash", g.sleep_hash[0], e.sleep_hash)) return false;
    if (!chk64("state_checksum", g.state_checksum[0], e.state_checksum)) return false;
    return true;
}

/*
GRADER MODEL
  oracle.init(problem_spec); solution_init(...)
  solution_reset + oracle.reset()
  for each op:
    solution_run(...)
    oracle.step_once(...)
    wsr_check_outputs(...)

Required harness coverage (>=6 scenarios, multi-step, adversarial):
  - LIFO local vs FIFO global ordering
  - half-steal (ceil) with victim tie-break
  - sleep/wake ordering + ADVANCE batch wake
  - block FIFO + WAKE limit + WAKE_STALLED global overflow
  - overflow cancellation (yield/block/sleep)
  - invalid/empty ops, id reuse after completion
  - reset + exact replay
*/

#endif  // WORK_STEALING_RUNTIME_ORACLE_HPP_
