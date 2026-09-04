// PMPP_CANARY_41_9e5cb20284 -- held-out canary; MUST NOT appear in any submission
// file: work_stealing_runtime_reference.cu
//
// Reference implementation of the T41 deterministic work-stealing runtime.
//
// Data-structure strategy (independent of naive + oracle):
//   * Each local/global deque is a fixed-capacity RING BUFFER (head index +
//     size) storing task_ids. Owners pop from the tail (LIFO); FIFO queues pop
//     from the head.
//   * The task table is a flat slot array; lookup is a linear scan keyed by
//     task_id with a live flag. Slots are reused after termination.
//   * Blocked tasks live as per-key FIFO ring buffers indexed by a small set of
//     distinct wait_keys discovered on the fly.
//   * Sleeping tasks live in an unsorted slot pool; ordering by
//     (wake_tick, sleep_seq, task_id) is materialised by selection at use time.
// Everything runs in a single <<<1,1>>> kernel mutating persistent global state.

#include "work_stealing_runtime_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define WSR_FNV_OFFSET 1469598103934665603ULL
#define WSR_FNV_PRIME  1099511628211ULL

namespace wsr_ref {

enum { ST_READY_LOCAL = 0, ST_READY_GLOBAL = 1, ST_RUNNING = 2, ST_BLOCKED = 3, ST_SLEEPING = 4 };

// Persistent state laid out across several device allocations.
struct State {
    WsrProblemSpec spec;

    // Scalars (device): [clock, event_seq, op_index, sched_hash, blocked_total]
    uint64_t* scal;            // 5 u64

    // Task table (flat slots, capacity = max_tasks).
    uint8_t*  t_live;          // [cap]
    uint64_t* t_id;            // [cap]
    int32_t*  t_home;          // [cap]
    int32_t*  t_prio;          // [cap]
    uint64_t* t_rem;           // [cap]
    int32_t*  t_state;         // [cap]
    uint64_t* t_birth;         // [cap]
    uint64_t* t_lastrun;       // [cap]
    uint64_t* t_blockseq;      // [cap]
    uint64_t* t_sleepseq;      // [cap]
    uint64_t* t_waitkey;       // [cap]
    uint64_t* t_waketick;      // [cap]

    // Local deques: ring buffers. Layout per (worker, prio): base + cap entries.
    // index = ((worker*2)+prio)*local_cap.
    uint64_t* lq_buf;          // [W*2*local_cap]
    int32_t*  lq_head;         // [W*2]
    int32_t*  lq_size;         // [W*2]

    // Global deques (prio 0,1): ring buffers.
    uint64_t* gq_buf;          // [2*global_cap]
    int32_t*  gq_head;         // [2]
    int32_t*  gq_size;         // [2]

    // Running slot per worker.
    int32_t*  run_has;         // [W]
    uint64_t* run_id;          // [W]

    // Blocked index: up to max_tasks distinct keys, each a FIFO ring of slot ids.
    // We store key buckets compactly: bk_key[k], bk_used[k], ring of task_ids.
    int32_t   nblock_keys;     // host-mirrored capacity bound = max_tasks (bounded)
    uint64_t* bk_key;          // [maxkeys]
    int32_t*  bk_count;        // [maxkeys]
    int32_t*  bk_head;         // [maxkeys]
    uint64_t* bk_ring;         // [maxkeys * maxlen]  maxlen = max_blocked (bounded), but to bound we use per-key cap = max_blocked
    int32_t   bk_maxkeys;
    int32_t   bk_maxlen;

    // Sleeping pool: unsorted slot list of task ids.
    uint64_t* sl_ids;          // [max_sleeping]
    int32_t*  sl_count;        // [1] device

    // Output mirror buffers (counts live in scal? no — separate).
    int64_t*  counts;          // [WSR_COUNT_N] device
    int32_t*  idmap;           // [hcap] id->slot open-addressing, -1 empty
};

}  // namespace wsr_ref

// ---------------------------------------------------------------- device FNV
__device__ __forceinline__ uint64_t wsr_ref_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= WSR_FNV_PRIME;
    return h;
}
__device__ void wsr_ref_fnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*)p;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = wsr_ref_fnv_byte(v, q[i]);
    *h = v;
}

// ------------------------------------------------------------ scalar accessors
// scal: 0 clock, 1 event_seq, 2 op_index, 3 sched_hash, 4 blocked_total

struct WsrRefCtx {
    wsr_ref::State s;          // by value copy of pointers (device-usable)
    int W, lcap, gcap, cap;
    int maxkeys, maxlen, maxsleep, maxblocked;
    int hcap;
};

// ---------------- task table helpers (linear scan) ----------------
__device__ __forceinline__ uint32_t wsr_ref_mix_id(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return (uint32_t)x;
}
__device__ int wsr_ref_map_find(const WsrRefCtx& c, uint64_t id) {
    const uint32_t mask = (uint32_t)c.hcap - 1u;
    uint32_t i = wsr_ref_mix_id(id) & mask;
    while (c.s.idmap[i] != -1) {
        const int sl = c.s.idmap[i];
        if (c.s.t_id[sl] == id) return sl;
        i = (i + 1u) & mask;
    }
    return -1;
}
__device__ void wsr_ref_map_insert(const WsrRefCtx& c, uint64_t id, int slot) {
    const uint32_t mask = (uint32_t)c.hcap - 1u;
    uint32_t i = wsr_ref_mix_id(id) & mask;
    while (c.s.idmap[i] != -1) i = (i + 1u) & mask;
    c.s.idmap[i] = slot;
}
__device__ void wsr_ref_map_erase(const WsrRefCtx& c, uint64_t id) {
    const uint32_t mask = (uint32_t)c.hcap - 1u;
    uint32_t i = wsr_ref_mix_id(id) & mask;
    while (c.s.idmap[i] != -1 && c.s.t_id[c.s.idmap[i]] != id) i = (i + 1u) & mask;
    if (c.s.idmap[i] == -1) return;
    uint32_t j = i;
    for (;;) {
        c.s.idmap[i] = -1;
        for (;;) {
            j = (j + 1u) & mask;
            if (c.s.idmap[j] == -1) return;
            const uint32_t k = wsr_ref_mix_id(c.s.t_id[c.s.idmap[j]]) & mask;
            const bool between = (i <= j) ? (i < k && k <= j) : (i < k || k <= j);
            if (!between) break;
        }
        c.s.idmap[i] = c.s.idmap[j];
        i = j;
    }
}
__device__ int wsr_ref_find_slot(const WsrRefCtx& c, uint64_t id) {
    return wsr_ref_map_find(c, id);
}
__device__ int wsr_ref_alloc_slot(const WsrRefCtx& c) {
    for (int i = 0; i < c.cap; ++i) {
        if (!c.s.t_live[i]) return i;
    }
    return -1;
}
__device__ int wsr_ref_live_count(const WsrRefCtx& c) {
    int n = 0;
    for (int i = 0; i < c.cap; ++i) if (c.s.t_live[i]) ++n;
    return n;
}

// ---------------- ring deque helpers ----------------
// local index l = worker*2 + prio.
__device__ __forceinline__ int wsr_ref_lqi(int worker, int prio) { return worker * 2 + prio; }

__device__ int wsr_ref_local_size(const WsrRefCtx& c, int worker, int prio) {
    return c.s.lq_size[wsr_ref_lqi(worker, prio)];
}
__device__ void wsr_ref_local_push_tail(const WsrRefCtx& c, int worker, int prio, uint64_t id) {
    const int l = wsr_ref_lqi(worker, prio);
    const int base = l * c.lcap;
    const int head = c.s.lq_head[l];
    const int sz = c.s.lq_size[l];
    const int pos = (head + sz) % c.lcap;
    c.s.lq_buf[base + pos] = id;
    c.s.lq_size[l] = sz + 1;
}
__device__ uint64_t wsr_ref_local_pop_tail(const WsrRefCtx& c, int worker, int prio) {
    const int l = wsr_ref_lqi(worker, prio);
    const int base = l * c.lcap;
    const int head = c.s.lq_head[l];
    const int sz = c.s.lq_size[l];
    const int pos = (head + sz - 1) % c.lcap;
    const uint64_t id = c.s.lq_buf[base + pos];
    c.s.lq_size[l] = sz - 1;
    return id;
}
__device__ uint64_t wsr_ref_local_pop_head(const WsrRefCtx& c, int worker, int prio) {
    const int l = wsr_ref_lqi(worker, prio);
    const int base = l * c.lcap;
    const int head = c.s.lq_head[l];
    const int sz = c.s.lq_size[l];
    const uint64_t id = c.s.lq_buf[base + head];
    c.s.lq_head[l] = (head + 1) % c.lcap;
    c.s.lq_size[l] = sz - 1;
    return id;
}
__device__ uint64_t wsr_ref_local_at(const WsrRefCtx& c, int worker, int prio, int idx) {
    const int l = wsr_ref_lqi(worker, prio);
    const int base = l * c.lcap;
    const int head = c.s.lq_head[l];
    return c.s.lq_buf[base + (head + idx) % c.lcap];
}

__device__ int wsr_ref_global_size(const WsrRefCtx& c, int prio) { return c.s.gq_size[prio]; }
__device__ void wsr_ref_global_push_tail(const WsrRefCtx& c, int prio, uint64_t id) {
    const int base = prio * c.gcap;
    const int head = c.s.gq_head[prio];
    const int sz = c.s.gq_size[prio];
    const int pos = (head + sz) % c.gcap;
    c.s.gq_buf[base + pos] = id;
    c.s.gq_size[prio] = sz + 1;
}
__device__ uint64_t wsr_ref_global_pop_head(const WsrRefCtx& c, int prio) {
    const int base = prio * c.gcap;
    const int head = c.s.gq_head[prio];
    const int sz = c.s.gq_size[prio];
    const uint64_t id = c.s.gq_buf[base + head];
    c.s.gq_head[prio] = (head + 1) % c.gcap;
    c.s.gq_size[prio] = sz - 1;
    return id;
}
__device__ uint64_t wsr_ref_global_at(const WsrRefCtx& c, int prio, int idx) {
    const int base = prio * c.gcap;
    const int head = c.s.gq_head[prio];
    return c.s.gq_buf[base + (head + idx) % c.gcap];
}

// remove first occurrence of id from a local ring (by physical rebuild).
__device__ bool wsr_ref_local_remove(const WsrRefCtx& c, int worker, int prio, uint64_t id) {
    const int l = wsr_ref_lqi(worker, prio);
    const int sz = c.s.lq_size[l];
    int found = -1;
    for (int i = 0; i < sz; ++i) {
        if (wsr_ref_local_at(c, worker, prio, i) == id) { found = i; break; }
    }
    if (found < 0) return false;
    // Rebuild deque skipping found.
    uint64_t tmp[WSR_MAX_LOCAL_CAP];
    int n = 0;
    for (int i = 0; i < sz; ++i) {
        if (i == found) continue;
        tmp[n++] = wsr_ref_local_at(c, worker, prio, i);
    }
    c.s.lq_head[l] = 0;
    c.s.lq_size[l] = 0;
    for (int i = 0; i < n; ++i) wsr_ref_local_push_tail(c, worker, prio, tmp[i]);
    return true;
}
__device__ bool wsr_ref_global_remove(const WsrRefCtx& c, int prio, uint64_t id) {
    const int sz = c.s.gq_size[prio];
    int found = -1;
    for (int i = 0; i < sz; ++i) {
        if (wsr_ref_global_at(c, prio, i) == id) { found = i; break; }
    }
    if (found < 0) return false;
    static const int MAXG = WSR_MAX_GLOBAL_CAP;
    (void)MAXG;
    // Rebuild via temp on heap-free stack is too big; do in-place shift.
    for (int i = found; i < sz - 1; ++i) {
        const int base = prio * c.gcap;
        const int head = c.s.gq_head[prio];
        c.s.gq_buf[base + (head + i) % c.gcap] = wsr_ref_global_at(c, prio, i + 1);
    }
    c.s.gq_size[prio] = sz - 1;
    return true;
}

// ---------------- blocked index helpers ----------------
__device__ int wsr_ref_bk_find(const WsrRefCtx& c, uint64_t key) {
    for (int k = 0; k < c.maxkeys; ++k) {
        if (c.s.bk_count[k] > 0 && c.s.bk_key[k] == key) return k;
    }
    return -1;
}
__device__ int wsr_ref_bk_alloc(const WsrRefCtx& c, uint64_t key) {
    int existing = wsr_ref_bk_find(c, key);
    if (existing >= 0) return existing;
    for (int k = 0; k < c.maxkeys; ++k) {
        if (c.s.bk_count[k] == 0) {
            c.s.bk_key[k] = key;
            c.s.bk_head[k] = 0;
            c.s.bk_count[k] = 0;
            return k;
        }
    }
    return -1;
}
__device__ void wsr_ref_bk_push_tail(const WsrRefCtx& c, int k, uint64_t id) {
    const int base = k * c.maxlen;
    const int head = c.s.bk_head[k];
    const int cnt = c.s.bk_count[k];
    const int pos = (head + cnt) % c.maxlen;
    c.s.bk_ring[base + pos] = id;
    c.s.bk_count[k] = cnt + 1;
}
__device__ void wsr_ref_bk_push_front(const WsrRefCtx& c, int k, uint64_t id) {
    const int base = k * c.maxlen;
    int head = c.s.bk_head[k];
    head = (head - 1 + c.maxlen) % c.maxlen;
    c.s.bk_ring[base + head] = id;
    c.s.bk_head[k] = head;
    c.s.bk_count[k] = c.s.bk_count[k] + 1;
}
__device__ uint64_t wsr_ref_bk_front(const WsrRefCtx& c, int k) {
    const int base = k * c.maxlen;
    return c.s.bk_ring[base + c.s.bk_head[k]];
}
__device__ uint64_t wsr_ref_bk_pop_front(const WsrRefCtx& c, int k) {
    const int base = k * c.maxlen;
    const int head = c.s.bk_head[k];
    const uint64_t id = c.s.bk_ring[base + head];
    c.s.bk_head[k] = (head + 1) % c.maxlen;
    c.s.bk_count[k] = c.s.bk_count[k] - 1;
    return id;
}
__device__ uint64_t wsr_ref_bk_at(const WsrRefCtx& c, int k, int idx) {
    const int base = k * c.maxlen;
    return c.s.bk_ring[base + (c.s.bk_head[k] + idx) % c.maxlen];
}
__device__ bool wsr_ref_bk_remove(const WsrRefCtx& c, int k, uint64_t id) {
    const int cnt = c.s.bk_count[k];
    int found = -1;
    for (int i = 0; i < cnt; ++i) {
        if (wsr_ref_bk_at(c, k, i) == id) { found = i; break; }
    }
    if (found < 0) return false;
    // shift remaining forward within ring
    for (int i = found; i < cnt - 1; ++i) {
        const int base = k * c.maxlen;
        const int head = c.s.bk_head[k];
        c.s.bk_ring[base + (head + i) % c.maxlen] = wsr_ref_bk_at(c, k, i + 1);
    }
    c.s.bk_count[k] = cnt - 1;
    return true;
}

// ---------------- event emit ----------------
__device__ void wsr_ref_emit(WsrRefCtx& c, uint8_t kind, int worker, uint64_t task_id,
                             uint64_t aux0, uint64_t aux1, int prio, uint64_t rem_after) {
    uint64_t* scal = c.s.scal;
    const uint64_t seq = scal[1];
    uint64_t h = scal[3];
    const uint32_t w = (worker < 0) ? UINT32_MAX : (uint32_t)worker;
    const uint8_t p = (prio < 0) ? (uint8_t)255 : (uint8_t)prio;
    wsr_ref_fnv_bytes(&h, &kind, sizeof(uint8_t));
    wsr_ref_fnv_bytes(&h, &seq, sizeof(uint64_t));
    uint32_t opi = (uint32_t)scal[2];
    wsr_ref_fnv_bytes(&h, &opi, sizeof(uint32_t));
    wsr_ref_fnv_bytes(&h, &w, sizeof(uint32_t));
    wsr_ref_fnv_bytes(&h, &task_id, sizeof(uint64_t));
    wsr_ref_fnv_bytes(&h, &aux0, sizeof(uint64_t));
    wsr_ref_fnv_bytes(&h, &aux1, sizeof(uint64_t));
    wsr_ref_fnv_bytes(&h, &p, sizeof(uint8_t));
    wsr_ref_fnv_bytes(&h, &rem_after, sizeof(uint64_t));
    scal[3] = h;
    scal[1] = seq + 1;
}
__device__ __forceinline__ uint64_t wsr_ref_peek_seq(WsrRefCtx& c) { return c.s.scal[1]; }

// ---------------- placement ----------------
// returns 0 local, 1 global, -1 fail. updates state field.
__device__ int wsr_ref_place_ready(WsrRefCtx& c, int slot) {
    const int home = c.s.t_home[slot];
    const int prio = c.s.t_prio[slot];
    if (wsr_ref_local_size(c, home, prio) < c.lcap) {
        wsr_ref_local_push_tail(c, home, prio, c.s.t_id[slot]);
        c.s.t_state[slot] = wsr_ref::ST_READY_LOCAL;
        return 0;
    }
    if (wsr_ref_global_size(c, prio) < c.gcap) {
        wsr_ref_global_push_tail(c, prio, c.s.t_id[slot]);
        c.s.t_state[slot] = wsr_ref::ST_READY_GLOBAL;
        return 1;
    }
    return -1;
}

// ---------------- operations ----------------
__device__ void wsr_ref_op_spawn(WsrRefCtx& c, uint64_t task_id, int home, int prio, uint64_t work) {
    if (home < 0 || home >= c.W || prio < 0 || prio > 1 || work == 0) {
        c.s.counts[WSR_C_INVALID] += 1;
        wsr_ref_emit(c, WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
        return;
    }
    if (wsr_ref_find_slot(c, task_id) >= 0) {
        c.s.counts[WSR_C_INVALID] += 1;
        wsr_ref_emit(c, WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
        return;
    }
    if (wsr_ref_live_count(c) >= c.cap) {
        c.s.counts[WSR_C_SPAWN_REJECT] += 1;
        wsr_ref_emit(c, WSR_EV_SPAWN_REJECT, home, task_id, 0, 0, prio, UINT64_MAX);
        return;
    }
    const int slot = wsr_ref_alloc_slot(c);
    c.s.t_live[slot] = 1;
    c.s.t_id[slot] = task_id;
    wsr_ref_map_insert(c, task_id, slot);
    c.s.t_home[slot] = home;
    c.s.t_prio[slot] = prio;
    c.s.t_rem[slot] = work;
    c.s.t_birth[slot] = wsr_ref_peek_seq(c);
    c.s.t_state[slot] = wsr_ref::ST_READY_LOCAL;
    const int placed = wsr_ref_place_ready(c, slot);
    if (placed < 0) {
        wsr_ref_map_erase(c, c.s.t_id[slot]); c.s.t_live[slot] = 0;
        c.s.counts[WSR_C_SPAWN_REJECT] += 1;
        wsr_ref_emit(c, WSR_EV_SPAWN_REJECT, home, task_id, 0, 0, prio, UINT64_MAX);
        return;
    }
    if (placed == 0) {
        c.s.counts[WSR_C_SPAWN_LOCAL] += 1;
        wsr_ref_emit(c, WSR_EV_SPAWN_LOCAL, home, task_id, 0, 0, prio, work);
    } else {
        c.s.counts[WSR_C_SPAWN_GLOBAL] += 1;
        wsr_ref_emit(c, WSR_EV_SPAWN_GLOBAL, home, task_id, 0, 0, prio, work);
    }
}

__device__ uint64_t wsr_ref_pick_local_or_global(WsrRefCtx& c, int worker) {
    if (wsr_ref_local_size(c, worker, 1) > 0) return wsr_ref_local_pop_tail(c, worker, 1);
    if (wsr_ref_local_size(c, worker, 0) > 0) return wsr_ref_local_pop_tail(c, worker, 0);
    if (wsr_ref_global_size(c, 1) > 0) return wsr_ref_global_pop_head(c, 1);
    if (wsr_ref_global_size(c, 0) > 0) return wsr_ref_global_pop_head(c, 0);
    return UINT64_MAX;
}
__device__ uint64_t wsr_ref_pick_local_only(WsrRefCtx& c, int worker) {
    if (wsr_ref_local_size(c, worker, 1) > 0) return wsr_ref_local_pop_tail(c, worker, 1);
    if (wsr_ref_local_size(c, worker, 0) > 0) return wsr_ref_local_pop_tail(c, worker, 0);
    return UINT64_MAX;
}

__device__ void wsr_ref_do_steal(WsrRefCtx& c, int thief) {
    int victim = -1;
    int best = 0;
    for (int w = 0; w < c.W; ++w) {
        if (w == thief) continue;
        const int n = wsr_ref_local_size(c, w, 1) + wsr_ref_local_size(c, w, 0);
        if (n <= 0) continue;
        if (n > best) { best = n; victim = w; }
    }
    if (victim < 0) return;

    const int high_n = wsr_ref_local_size(c, victim, 1);
    const int low_n = wsr_ref_local_size(c, victim, 0);
    const int total = high_n + low_n;
    const int take = (total + 1) / 2;

    for (int i = 0; i < take; ++i) {
        const bool is_high = (i < high_n);
        const int prio = is_high ? 1 : 0;
        uint64_t id;
        if (is_high) {
            id = wsr_ref_local_pop_head(c, victim, 1);
            wsr_ref_local_push_tail(c, thief, 1, id);
        } else {
            id = wsr_ref_local_pop_head(c, victim, 0);
            wsr_ref_local_push_tail(c, thief, 0, id);
        }
        const int slot = wsr_ref_find_slot(c, id);
        const uint64_t rem = (slot >= 0) ? c.s.t_rem[slot] : UINT64_MAX;
        c.s.counts[WSR_C_STOLEN_TASKS] += 1;
        wsr_ref_emit(c, WSR_EV_STEAL_TASK, thief, id, (uint64_t)victim, 0, prio, rem);
    }
}

__device__ void wsr_ref_op_run(WsrRefCtx& c, int worker, uint64_t quantum) {
    if (worker < 0 || worker >= c.W || quantum == 0) {
        c.s.counts[WSR_C_INVALID] += 1;
        wsr_ref_emit(c, WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
        return;
    }
    uint64_t tid;
    if (c.s.run_has[worker]) {
        tid = c.s.run_id[worker];
    } else {
        tid = wsr_ref_pick_local_or_global(c, worker);
        if (tid == UINT64_MAX) {
            wsr_ref_do_steal(c, worker);
            tid = wsr_ref_pick_local_only(c, worker);
        }
        if (tid == UINT64_MAX) {
            c.s.counts[WSR_C_IDLE] += 1;
            wsr_ref_emit(c, WSR_EV_IDLE, worker, UINT64_MAX, 0, 0, -1, UINT64_MAX);
            return;
        }
        const int slot = wsr_ref_find_slot(c, tid);
        c.s.t_state[slot] = wsr_ref::ST_RUNNING;
        c.s.run_has[worker] = 1;
        c.s.run_id[worker] = tid;
        c.s.counts[WSR_C_SCHEDULED] += 1;
        wsr_ref_emit(c, WSR_EV_SCHEDULE, worker, tid, 0, 0, c.s.t_prio[slot], c.s.t_rem[slot]);
    }
    const int slot = wsr_ref_find_slot(c, tid);
    const uint64_t rem = c.s.t_rem[slot];
    const uint64_t slice = (quantum < rem) ? quantum : rem;
    c.s.t_rem[slot] = rem - slice;
    c.s.t_lastrun[slot] = wsr_ref_peek_seq(c);
    c.s.counts[WSR_C_RUN_SLICES] += 1;
    wsr_ref_emit(c, WSR_EV_RUN_SLICE, worker, tid, slice, 0, c.s.t_prio[slot], c.s.t_rem[slot]);
    if (c.s.t_rem[slot] == 0) {
        const int prio = c.s.t_prio[slot];
        c.s.run_has[worker] = 0;
        wsr_ref_map_erase(c, c.s.t_id[slot]); c.s.t_live[slot] = 0;
        c.s.counts[WSR_C_COMPLETED] += 1;
        wsr_ref_emit(c, WSR_EV_COMPLETE, worker, tid, 0, 0, prio, 0);
    }
}

__device__ void wsr_ref_op_yield(WsrRefCtx& c, int worker) {
    if (worker < 0 || worker >= c.W) {
        c.s.counts[WSR_C_INVALID] += 1;
        wsr_ref_emit(c, WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
        return;
    }
    if (!c.s.run_has[worker]) {
        c.s.counts[WSR_C_EMPTY_OP] += 1;
        wsr_ref_emit(c, WSR_EV_YIELD_EMPTY, worker, UINT64_MAX, 0, 0, -1, UINT64_MAX);
        return;
    }
    const uint64_t tid = c.s.run_id[worker];
    const int slot = wsr_ref_find_slot(c, tid);
    c.s.run_has[worker] = 0;
    const int placed = wsr_ref_place_ready(c, slot);
    if (placed < 0) {
        const int prio = c.s.t_prio[slot];
        const uint64_t rem = c.s.t_rem[slot];
        wsr_ref_map_erase(c, c.s.t_id[slot]); c.s.t_live[slot] = 0;
        c.s.counts[WSR_C_OVERFLOW_CANCELLED] += 1;
        wsr_ref_emit(c, WSR_EV_CANCEL_OVERFLOW, worker, tid, WSR_REASON_YIELD_OVERFLOW, 0, prio, rem);
        return;
    }
    c.s.counts[WSR_C_YIELDED] += 1;
    wsr_ref_emit(c, WSR_EV_YIELD, worker, tid, 0, 0, c.s.t_prio[slot], c.s.t_rem[slot]);
}

__device__ void wsr_ref_op_block(WsrRefCtx& c, int worker, uint64_t wait_key) {
    if (worker < 0 || worker >= c.W) {
        c.s.counts[WSR_C_INVALID] += 1;
        wsr_ref_emit(c, WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
        return;
    }
    if (!c.s.run_has[worker]) {
        c.s.counts[WSR_C_EMPTY_OP] += 1;
        wsr_ref_emit(c, WSR_EV_BLOCK_EMPTY, worker, UINT64_MAX, 0, 0, -1, UINT64_MAX);
        return;
    }
    const uint64_t tid = c.s.run_id[worker];
    const int slot = wsr_ref_find_slot(c, tid);
    c.s.run_has[worker] = 0;
    if ((int64_t)c.s.scal[4] >= c.maxblocked) {
        const int prio = c.s.t_prio[slot];
        const uint64_t rem = c.s.t_rem[slot];
        wsr_ref_map_erase(c, c.s.t_id[slot]); c.s.t_live[slot] = 0;
        c.s.counts[WSR_C_OVERFLOW_CANCELLED] += 1;
        wsr_ref_emit(c, WSR_EV_CANCEL_OVERFLOW, worker, tid, WSR_REASON_BLOCK_OVERFLOW, 0, prio, rem);
        return;
    }
    c.s.t_state[slot] = wsr_ref::ST_BLOCKED;
    c.s.t_waitkey[slot] = wait_key;
    c.s.t_blockseq[slot] = wsr_ref_peek_seq(c);
    const int k = wsr_ref_bk_alloc(c, wait_key);
    wsr_ref_bk_push_tail(c, k, tid);
    c.s.scal[4] += 1;
    c.s.counts[WSR_C_BLOCKED] += 1;
    wsr_ref_emit(c, WSR_EV_BLOCK, worker, tid, wait_key, 0, c.s.t_prio[slot], c.s.t_rem[slot]);
}

__device__ void wsr_ref_op_sleep(WsrRefCtx& c, int worker, uint64_t wake_tick) {
    if (worker < 0 || worker >= c.W) {
        c.s.counts[WSR_C_INVALID] += 1;
        wsr_ref_emit(c, WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
        return;
    }
    if (!c.s.run_has[worker]) {
        c.s.counts[WSR_C_EMPTY_OP] += 1;
        wsr_ref_emit(c, WSR_EV_SLEEP_EMPTY, worker, UINT64_MAX, 0, 0, -1, UINT64_MAX);
        return;
    }
    const uint64_t tid = c.s.run_id[worker];
    const int slot = wsr_ref_find_slot(c, tid);
    c.s.run_has[worker] = 0;
    if (*c.s.sl_count >= c.maxsleep) {
        const int prio = c.s.t_prio[slot];
        const uint64_t rem = c.s.t_rem[slot];
        wsr_ref_map_erase(c, c.s.t_id[slot]); c.s.t_live[slot] = 0;
        c.s.counts[WSR_C_OVERFLOW_CANCELLED] += 1;
        wsr_ref_emit(c, WSR_EV_CANCEL_OVERFLOW, worker, tid, WSR_REASON_SLEEP_OVERFLOW, 0, prio, rem);
        return;
    }
    c.s.t_state[slot] = wsr_ref::ST_SLEEPING;
    c.s.t_waketick[slot] = wake_tick;
    c.s.t_sleepseq[slot] = wsr_ref_peek_seq(c);
    c.s.sl_ids[*c.s.sl_count] = tid;
    *c.s.sl_count = *c.s.sl_count + 1;
    c.s.counts[WSR_C_SLEPT] += 1;
    wsr_ref_emit(c, WSR_EV_SLEEP, worker, tid, wake_tick, 0, c.s.t_prio[slot], c.s.t_rem[slot]);
}

__device__ void wsr_ref_op_wake(WsrRefCtx& c, uint64_t wait_key, int64_t limit) {
    if (limit <= 0) return;
    const int k = wsr_ref_bk_find(c, wait_key);
    if (k < 0) return;
    int64_t done = 0;
    while (done < limit && c.s.bk_count[k] > 0) {
        const uint64_t tid = wsr_ref_bk_pop_front(c, k);
        const int slot = wsr_ref_find_slot(c, tid);
        const int placed = wsr_ref_place_ready(c, slot);
        if (placed < 0) {
            wsr_ref_bk_push_front(c, k, tid);
            c.s.t_state[slot] = wsr_ref::ST_BLOCKED;
            wsr_ref_emit(c, WSR_EV_WAKE_STALLED, c.s.t_home[slot], tid, wait_key, 0,
                         c.s.t_prio[slot], c.s.t_rem[slot]);
            return;
        }
        c.s.scal[4] -= 1;
        if (placed == 0) {
            c.s.counts[WSR_C_WOKEN_LOCAL] += 1;
            wsr_ref_emit(c, WSR_EV_WAKE_LOCAL, c.s.t_home[slot], tid, wait_key, 0,
                         c.s.t_prio[slot], c.s.t_rem[slot]);
        } else {
            c.s.counts[WSR_C_WOKEN_GLOBAL] += 1;
            wsr_ref_emit(c, WSR_EV_WAKE_GLOBAL, c.s.t_home[slot], tid, wait_key, 0,
                         c.s.t_prio[slot], c.s.t_rem[slot]);
        }
        ++done;
    }
}

// Find sleeping pool index of the minimum (wake_tick, sleep_seq, task_id).
__device__ int wsr_ref_sleep_min_idx(WsrRefCtx& c) {
    const int n = *c.s.sl_count;
    if (n == 0) return -1;
    int best = -1;
    uint64_t bw = 0, bs = 0, bt = 0;
    for (int i = 0; i < n; ++i) {
        const uint64_t id = c.s.sl_ids[i];
        const int slot = wsr_ref_find_slot(c, id);
        const uint64_t wt = c.s.t_waketick[slot];
        const uint64_t ss = c.s.t_sleepseq[slot];
        bool take = false;
        if (best < 0) take = true;
        else if (wt < bw) take = true;
        else if (wt == bw && ss < bs) take = true;
        else if (wt == bw && ss == bs && id < bt) take = true;
        if (take) { best = i; bw = wt; bs = ss; bt = id; }
    }
    return best;
}

__device__ void wsr_ref_sleep_erase_idx(WsrRefCtx& c, int idx) {
    const int n = *c.s.sl_count;
    for (int i = idx; i < n - 1; ++i) c.s.sl_ids[i] = c.s.sl_ids[i + 1];
    *c.s.sl_count = n - 1;
}

__device__ void wsr_ref_op_advance(WsrRefCtx& c, uint64_t delta) {
    c.s.scal[0] += delta;  // clock wraps
    const uint64_t clk = c.s.scal[0];
    wsr_ref_emit(c, WSR_EV_CLOCK_ADVANCE, -1, UINT64_MAX, delta, clk, -1, UINT64_MAX);

    for (;;) {
        const int idx = wsr_ref_sleep_min_idx(c);
        if (idx < 0) break;
        const uint64_t id = c.s.sl_ids[idx];
        const int slot = wsr_ref_find_slot(c, id);
        if (c.s.t_waketick[slot] > clk) break;
        const int placed = wsr_ref_place_ready(c, slot);
        if (placed < 0) {
            wsr_ref_emit(c, WSR_EV_WAKE_STALLED, c.s.t_home[slot], id, c.s.t_waketick[slot], 0,
                         c.s.t_prio[slot], c.s.t_rem[slot]);
            return;
        }
        wsr_ref_sleep_erase_idx(c, idx);
        if (placed == 0) {
            c.s.counts[WSR_C_WOKEN_LOCAL] += 1;
            wsr_ref_emit(c, WSR_EV_WAKE_LOCAL, c.s.t_home[slot], id, c.s.t_waketick[slot], 0,
                         c.s.t_prio[slot], c.s.t_rem[slot]);
        } else {
            c.s.counts[WSR_C_WOKEN_GLOBAL] += 1;
            wsr_ref_emit(c, WSR_EV_WAKE_GLOBAL, c.s.t_home[slot], id, c.s.t_waketick[slot], 0,
                         c.s.t_prio[slot], c.s.t_rem[slot]);
        }
    }
}

__device__ void wsr_ref_op_cancel(WsrRefCtx& c, uint64_t task_id) {
    const int slot = wsr_ref_find_slot(c, task_id);
    if (slot < 0) {
        c.s.counts[WSR_C_INVALID] += 1;
        wsr_ref_emit(c, WSR_EV_INVALID, -1, UINT64_MAX, 0, 0, -1, UINT64_MAX);
        return;
    }
    const int home = c.s.t_home[slot];
    const int prio = c.s.t_prio[slot];
    const uint64_t rem = c.s.t_rem[slot];
    const int state = c.s.t_state[slot];
    if (state == wsr_ref::ST_RUNNING) {
        for (int w = 0; w < c.W; ++w) {
            if (c.s.run_has[w] && c.s.run_id[w] == task_id) { c.s.run_has[w] = 0; break; }
        }
    } else if (state == wsr_ref::ST_READY_LOCAL || state == wsr_ref::ST_READY_GLOBAL) {
        if (!wsr_ref_local_remove(c, home, prio, task_id)) {
            if (!wsr_ref_global_remove(c, prio, task_id)) {
                // scan all locals as fallback
                for (int w = 0; w < c.W; ++w) {
                    if (wsr_ref_local_remove(c, w, 0, task_id)) break;
                    if (wsr_ref_local_remove(c, w, 1, task_id)) break;
                }
            }
        }
    } else if (state == wsr_ref::ST_BLOCKED) {
        const int k = wsr_ref_bk_find(c, c.s.t_waitkey[slot]);
        if (k >= 0) {
            if (wsr_ref_bk_remove(c, k, task_id)) c.s.scal[4] -= 1;
        }
    } else if (state == wsr_ref::ST_SLEEPING) {
        const int n = *c.s.sl_count;
        for (int i = 0; i < n; ++i) {
            if (c.s.sl_ids[i] == task_id) { wsr_ref_sleep_erase_idx(c, i); break; }
        }
    }
    wsr_ref_map_erase(c, c.s.t_id[slot]); c.s.t_live[slot] = 0;
    c.s.counts[WSR_C_CANCELLED] += 1;
    wsr_ref_emit(c, WSR_EV_CANCEL_EXPLICIT, home, task_id, 0, 0, prio, rem);
}

// ---------------- snapshot hashes ----------------
__device__ uint64_t wsr_ref_ready_hash(WsrRefCtx& c) {
    uint64_t h = WSR_FNV_OFFSET;
    for (int w = 0; w < c.W; ++w) {
        uint64_t pos = 0;
        const int hn = wsr_ref_local_size(c, w, 1);
        for (int i = 0; i < hn; ++i) {
            uint8_t sk = 0; uint32_t wu = (uint32_t)w; uint8_t p = 1; uint64_t id = wsr_ref_local_at(c, w, 1, i);
            wsr_ref_fnv_bytes(&h, &sk, 1); wsr_ref_fnv_bytes(&h, &wu, 4); wsr_ref_fnv_bytes(&h, &p, 1);
            wsr_ref_fnv_bytes(&h, &pos, 8); wsr_ref_fnv_bytes(&h, &id, 8); pos++;
        }
        pos = 0;
        const int ln = wsr_ref_local_size(c, w, 0);
        for (int i = 0; i < ln; ++i) {
            uint8_t sk = 0; uint32_t wu = (uint32_t)w; uint8_t p = 0; uint64_t id = wsr_ref_local_at(c, w, 0, i);
            wsr_ref_fnv_bytes(&h, &sk, 1); wsr_ref_fnv_bytes(&h, &wu, 4); wsr_ref_fnv_bytes(&h, &p, 1);
            wsr_ref_fnv_bytes(&h, &pos, 8); wsr_ref_fnv_bytes(&h, &id, 8); pos++;
        }
    }
    uint64_t pos = 0;
    int gn = wsr_ref_global_size(c, 1);
    for (int i = 0; i < gn; ++i) {
        uint8_t sk = 1; uint32_t wu = UINT32_MAX; uint8_t p = 1; uint64_t id = wsr_ref_global_at(c, 1, i);
        wsr_ref_fnv_bytes(&h, &sk, 1); wsr_ref_fnv_bytes(&h, &wu, 4); wsr_ref_fnv_bytes(&h, &p, 1);
        wsr_ref_fnv_bytes(&h, &pos, 8); wsr_ref_fnv_bytes(&h, &id, 8); pos++;
    }
    pos = 0;
    gn = wsr_ref_global_size(c, 0);
    for (int i = 0; i < gn; ++i) {
        uint8_t sk = 1; uint32_t wu = UINT32_MAX; uint8_t p = 0; uint64_t id = wsr_ref_global_at(c, 0, i);
        wsr_ref_fnv_bytes(&h, &sk, 1); wsr_ref_fnv_bytes(&h, &wu, 4); wsr_ref_fnv_bytes(&h, &p, 1);
        wsr_ref_fnv_bytes(&h, &pos, 8); wsr_ref_fnv_bytes(&h, &id, 8); pos++;
    }
    return h;
}

__device__ uint64_t wsr_ref_running_hash(WsrRefCtx& c) {
    uint64_t h = WSR_FNV_OFFSET;
    for (int w = 0; w < c.W; ++w) {
        uint32_t wu = (uint32_t)w;
        wsr_ref_fnv_bytes(&h, &wu, 4);
        if (c.s.run_has[w]) {
            uint8_t has = 1; uint64_t id = c.s.run_id[w];
            const int slot = wsr_ref_find_slot(c, id);
            uint64_t rem = (slot >= 0) ? c.s.t_rem[slot] : UINT64_MAX;
            wsr_ref_fnv_bytes(&h, &has, 1); wsr_ref_fnv_bytes(&h, &id, 8); wsr_ref_fnv_bytes(&h, &rem, 8);
        } else {
            uint8_t has = 0; uint64_t id = UINT64_MAX; uint64_t rem = UINT64_MAX;
            wsr_ref_fnv_bytes(&h, &has, 1); wsr_ref_fnv_bytes(&h, &id, 8); wsr_ref_fnv_bytes(&h, &rem, 8);
        }
    }
    return h;
}

__device__ uint64_t wsr_ref_blocked_hash(WsrRefCtx& c) {
    uint64_t h = WSR_FNV_OFFSET;
    // wait keys ascending. Selection sort over distinct active keys.
    uint64_t last_key = 0; bool have_last = false;
    for (;;) {
        // find smallest active key strictly greater than last_key (or any first).
        int best_k = -1; uint64_t best_key = 0;
        for (int k = 0; k < c.maxkeys; ++k) {
            if (c.s.bk_count[k] <= 0) continue;
            const uint64_t key = c.s.bk_key[k];
            if (have_last && key <= last_key) continue;
            if (best_k < 0 || key < best_key) { best_k = k; best_key = key; }
        }
        if (best_k < 0) break;
        // within key, block_seq ascending == FIFO order (push_tail), so iterate ring.
        const int cnt = c.s.bk_count[best_k];
        for (int i = 0; i < cnt; ++i) {
            const uint64_t id = wsr_ref_bk_at(c, best_k, i);
            const int slot = wsr_ref_find_slot(c, id);
            uint64_t key = best_key;
            uint64_t bseq = c.s.t_blockseq[slot];
            uint32_t home = (uint32_t)c.s.t_home[slot];
            uint8_t p = (uint8_t)c.s.t_prio[slot];
            uint64_t rem = c.s.t_rem[slot];
            wsr_ref_fnv_bytes(&h, &key, 8); wsr_ref_fnv_bytes(&h, &bseq, 8); wsr_ref_fnv_bytes(&h, &id, 8);
            wsr_ref_fnv_bytes(&h, &home, 4); wsr_ref_fnv_bytes(&h, &p, 1); wsr_ref_fnv_bytes(&h, &rem, 8);
        }
        last_key = best_key; have_last = true;
    }
    return h;
}

__device__ uint64_t wsr_ref_sleep_hash(WsrRefCtx& c) {
    uint64_t h = WSR_FNV_OFFSET;
    const int n = *c.s.sl_count;
    // selection over (wake_tick, sleep_seq, task_id) ascending.
    bool used_mark[WSR_MAX_MAX_SLEEPING];
    // can't statically size to 8192 on stack; use a different approach:
    // repeatedly find global min among not-yet-emitted using a "previous" cursor.
    (void)used_mark;
    uint64_t pw = 0, ps = 0, pt = 0; bool have_prev = false;
    for (int emitted = 0; emitted < n; ++emitted) {
        int best = -1; uint64_t bw = 0, bs = 0, bt = 0;
        for (int i = 0; i < n; ++i) {
            const uint64_t id = c.s.sl_ids[i];
            const int slot = wsr_ref_find_slot(c, id);
            const uint64_t wt = c.s.t_waketick[slot];
            const uint64_t ss = c.s.t_sleepseq[slot];
            // skip entries <= prev (already emitted)
            if (have_prev) {
                bool gt = (wt > pw) || (wt == pw && ss > ps) || (wt == pw && ss == ps && id > pt);
                if (!gt) continue;
            }
            bool take = false;
            if (best < 0) take = true;
            else if (wt < bw) take = true;
            else if (wt == bw && ss < bs) take = true;
            else if (wt == bw && ss == bs && id < bt) take = true;
            if (take) { best = i; bw = wt; bs = ss; bt = id; }
        }
        if (best < 0) break;
        uint64_t wt = bw, ss = bs, id = bt;
        const int slot = wsr_ref_find_slot(c, id);
        uint32_t home = (uint32_t)c.s.t_home[slot];
        uint8_t p = (uint8_t)c.s.t_prio[slot];
        uint64_t rem = c.s.t_rem[slot];
        wsr_ref_fnv_bytes(&h, &wt, 8); wsr_ref_fnv_bytes(&h, &ss, 8); wsr_ref_fnv_bytes(&h, &id, 8);
        wsr_ref_fnv_bytes(&h, &home, 4); wsr_ref_fnv_bytes(&h, &p, 1); wsr_ref_fnv_bytes(&h, &rem, 8);
        pw = wt; ps = ss; pt = id; have_prev = true;
    }
    return h;
}

// ---------------- main step kernel ----------------
__global__ void wsr_ref_step_kernel(
    wsr_ref::State s,
    int W, int lcap, int gcap, int cap, int maxkeys, int maxlen, int maxsleep, int maxblocked,
    int op_kind, int a_worker, int a_priority, int a_limit,
    uint64_t a_task, uint64_t a_work, uint64_t a_key, uint64_t a_tick, uint64_t a_delta,
    int64_t* out_counts, int32_t* out_opidx, uint64_t* out_clock, uint64_t* out_eseq,
    uint64_t* out_sched, uint64_t* out_ready, uint64_t* out_run, uint64_t* out_blocked,
    uint64_t* out_sleep, uint64_t* out_state) {
    if (blockIdx.x != 0 || threadIdx.x >= 128) return;
    const int tid = threadIdx.x;

    WsrRefCtx c;
    c.s = s;
    c.W = W; c.lcap = lcap; c.gcap = gcap; c.cap = cap;
    c.maxkeys = maxkeys; c.maxlen = maxlen; c.maxsleep = maxsleep; c.maxblocked = maxblocked;
    { c.hcap = 1; while (c.hcap < cap * 2) c.hcap <<= 1; }

    // Persistent scalars/counters cached in shared memory for the op (thread 0
    // owns the serial state machine; threads 1-3 compute independent snapshot
    // hashes in parallel to overlap their FNV fold chains and memory latency).
    __shared__ uint64_t s_scal[5];
    __shared__ int64_t s_counts[WSR_COUNT_N];
    __shared__ uint64_t s_hash[4];
    __shared__ int32_t s_thisop;
    uint64_t* g_scal = s.scal;
    int64_t* g_counts = s.counts;

    if (tid == 0) {
        for (int i = 0; i < 5; ++i) s_scal[i] = g_scal[i];
        for (int i = 0; i < WSR_COUNT_N; ++i) s_counts[i] = g_counts[i];
        c.s.scal = s_scal;
        c.s.counts = s_counts;

        switch (op_kind) {
            case WSR_OP_SPAWN:   wsr_ref_op_spawn(c, a_task, a_worker, a_priority, a_work); break;
            case WSR_OP_RUN:     wsr_ref_op_run(c, a_worker, a_work); break;
            case WSR_OP_YIELD:   wsr_ref_op_yield(c, a_worker); break;
            case WSR_OP_BLOCK:   wsr_ref_op_block(c, a_worker, a_key); break;
            case WSR_OP_SLEEP:   wsr_ref_op_sleep(c, a_worker, a_tick); break;
            case WSR_OP_WAKE:    wsr_ref_op_wake(c, a_key, (int64_t)a_limit); break;
            case WSR_OP_ADVANCE: wsr_ref_op_advance(c, a_delta); break;
            case WSR_OP_CANCEL:  wsr_ref_op_cancel(c, a_task); break;
            default: break;
        }

        s_thisop = (int32_t)c.s.scal[2];
        c.s.scal[2] += 1;
    }
    __syncthreads();

    // Each snapshot hash is an independent sequential FNV fold over post-op
    // state; run them on separate threads. They read only global structures
    // (ready deques, running slots, blocked rings, sleep pool), which thread 0
    // has finished mutating before the barrier above.
    if (tid == 0)       s_hash[0] = wsr_ref_ready_hash(c);
    else if (tid == 32) s_hash[1] = wsr_ref_running_hash(c);
    else if (tid == 64) s_hash[2] = wsr_ref_blocked_hash(c);
    else if (tid == 96) s_hash[3] = wsr_ref_sleep_hash(c);
    __syncthreads();

    if (tid == 0) {
        const uint64_t rh = s_hash[0];
        const uint64_t runh = s_hash[1];
        const uint64_t bh = s_hash[2];
        const uint64_t sh = s_hash[3];

        for (int i = 0; i < WSR_COUNT_N; ++i) out_counts[i] = c.s.counts[i];
        *out_opidx = s_thisop;
        *out_clock = c.s.scal[0];
        *out_eseq = c.s.scal[1];
        *out_sched = c.s.scal[3];
        *out_ready = rh;
        *out_run = runh;
        *out_blocked = bh;
        *out_sleep = sh;

        uint64_t mh = WSR_FNV_OFFSET;
        wsr_ref_fnv_bytes(&mh, &c.s.scal[0], 8);
        wsr_ref_fnv_bytes(&mh, &c.s.scal[1], 8);
        uint32_t opu = (uint32_t)c.s.scal[2];
        wsr_ref_fnv_bytes(&mh, &opu, 4);
        wsr_ref_fnv_bytes(&mh, &c.s.scal[3], 8);
        wsr_ref_fnv_bytes(&mh, &rh, 8);
        wsr_ref_fnv_bytes(&mh, &runh, 8);
        wsr_ref_fnv_bytes(&mh, &bh, 8);
        wsr_ref_fnv_bytes(&mh, &sh, 8);
        for (int i = 0; i < WSR_COUNT_N; ++i) {
            uint64_t cv = (uint64_t)c.s.counts[i];
            wsr_ref_fnv_bytes(&mh, &cv, 8);
        }
        *out_state = mh;

        for (int i = 0; i < 5; ++i) g_scal[i] = s_scal[i];
        for (int i = 0; i < WSR_COUNT_N; ++i) g_counts[i] = s_counts[i];
    }
}

__global__ void wsr_ref_reset_kernel(
    wsr_ref::State s, int W, int lcap, int gcap, int cap, int maxkeys, int maxlen, int maxsleep) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < 5; ++i) s.scal[i] = 0;
    s.scal[3] = WSR_FNV_OFFSET;  // sched_hash seed
    for (int i = 0; i < cap; ++i) s.t_live[i] = 0;
    { int hc = 1; while (hc < cap * 2) hc <<= 1; for (int i = 0; i < hc; ++i) s.idmap[i] = -1; }
    for (int i = 0; i < W * 2; ++i) { s.lq_head[i] = 0; s.lq_size[i] = 0; }
    for (int i = 0; i < 2; ++i) { s.gq_head[i] = 0; s.gq_size[i] = 0; }
    for (int i = 0; i < W; ++i) { s.run_has[i] = 0; s.run_id[i] = 0; }
    for (int i = 0; i < maxkeys; ++i) { s.bk_count[i] = 0; s.bk_head[i] = 0; s.bk_key[i] = 0; }
    *s.sl_count = 0;
    for (int i = 0; i < WSR_COUNT_N; ++i) s.counts[i] = 0;
    (void)maxlen; (void)maxsleep;
}

// ---------------- host ABI ----------------
static int wsr_ref_maxkeys(const WsrProblemSpec* spec) {
    int mk = spec->max_blocked > 0 ? spec->max_blocked : 1;
    if (mk > spec->max_tasks) mk = spec->max_tasks;
    if (mk < 1) mk = 1;
    return mk;
}
static int wsr_ref_maxlen(const WsrProblemSpec* spec) {
    int ml = spec->max_blocked > 0 ? spec->max_blocked : 1;
    if (ml < 1) ml = 1;
    return ml;
}
static int wsr_ref_maxsleep(const WsrProblemSpec* spec) {
    int ms = spec->max_sleeping > 0 ? spec->max_sleeping : 1;
    if (ms < 1) ms = 1;
    return ms;
}

extern "C" size_t solution_workspace_bytes(const WsrProblemSpec* spec) {
    if (!wsr_validate_problem_spec(spec)) return 0;
    return 256;
}

static cudaError_t wsr_ref_do_reset(wsr_ref::State* st, cudaStream_t stream) {
    const WsrProblemSpec& sp = st->spec;
    wsr_ref_reset_kernel<<<1, 1, 0, stream>>>(
        *st, sp.W, sp.local_cap_per_worker, sp.global_cap, sp.max_tasks,
        wsr_ref_maxkeys(&sp), wsr_ref_maxlen(&sp), wsr_ref_maxsleep(&sp));
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_init(const WsrProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!wsr_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;

    wsr_ref::State* st = (wsr_ref::State*)malloc(sizeof(wsr_ref::State));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(wsr_ref::State));
    memcpy(&st->spec, spec, sizeof(WsrProblemSpec));

    const int W = spec->W;
    const int lcap = spec->local_cap_per_worker;
    const int gcap = spec->global_cap;
    const int cap = spec->max_tasks;
    const int maxkeys = wsr_ref_maxkeys(spec);
    const int maxlen = wsr_ref_maxlen(spec);
    const int maxsleep = wsr_ref_maxsleep(spec);
    int hcap = 1; while (hcap < cap * 2) hcap <<= 1;

    cudaError_t err = cudaSuccess;
    #define WSR_M(p, bytes) do { err = cudaMalloc((void**)&(p), (bytes)); if (err) goto fail; } while (0)
    WSR_M(st->scal, sizeof(uint64_t) * 5);
    WSR_M(st->t_live, sizeof(uint8_t) * cap);
    WSR_M(st->t_id, sizeof(uint64_t) * cap);
    WSR_M(st->t_home, sizeof(int32_t) * cap);
    WSR_M(st->t_prio, sizeof(int32_t) * cap);
    WSR_M(st->t_rem, sizeof(uint64_t) * cap);
    WSR_M(st->t_state, sizeof(int32_t) * cap);
    WSR_M(st->t_birth, sizeof(uint64_t) * cap);
    WSR_M(st->t_lastrun, sizeof(uint64_t) * cap);
    WSR_M(st->t_blockseq, sizeof(uint64_t) * cap);
    WSR_M(st->t_sleepseq, sizeof(uint64_t) * cap);
    WSR_M(st->t_waitkey, sizeof(uint64_t) * cap);
    WSR_M(st->t_waketick, sizeof(uint64_t) * cap);
    WSR_M(st->lq_buf, sizeof(uint64_t) * (size_t)W * 2 * lcap);
    WSR_M(st->lq_head, sizeof(int32_t) * W * 2);
    WSR_M(st->lq_size, sizeof(int32_t) * W * 2);
    WSR_M(st->gq_buf, sizeof(uint64_t) * 2 * gcap);
    WSR_M(st->gq_head, sizeof(int32_t) * 2);
    WSR_M(st->gq_size, sizeof(int32_t) * 2);
    WSR_M(st->run_has, sizeof(int32_t) * W);
    WSR_M(st->run_id, sizeof(uint64_t) * W);
    WSR_M(st->bk_key, sizeof(uint64_t) * maxkeys);
    WSR_M(st->bk_count, sizeof(int32_t) * maxkeys);
    WSR_M(st->bk_head, sizeof(int32_t) * maxkeys);
    WSR_M(st->bk_ring, sizeof(uint64_t) * (size_t)maxkeys * maxlen);
    WSR_M(st->sl_ids, sizeof(uint64_t) * maxsleep);
    WSR_M(st->sl_count, sizeof(int32_t) * 1);
    WSR_M(st->counts, sizeof(int64_t) * WSR_COUNT_N);
    WSR_M(st->idmap, sizeof(int32_t) * hcap);
    #undef WSR_M

    st->bk_maxkeys = maxkeys;
    st->bk_maxlen = maxlen;
    st->nblock_keys = maxkeys;

    err = wsr_ref_do_reset(st, stream);
    if (err) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    solution_destroy(st);
    return err ? err : cudaErrorMemoryAllocation;
}

extern "C" cudaError_t solution_run(
    void* state, const WsrRunSpec* run, const void* inputs_void, void* outputs_void,
    void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    (void)inputs_void; (void)workspace;
    if (!state || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 256) return cudaErrorInvalidValue;

    wsr_ref::State* st = (wsr_ref::State*)state;
    if (!wsr_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    WsrOutputs* out = (WsrOutputs*)outputs_void;
    if (!out->counts || !out->op_index_out || !out->clock_out || !out->event_seq_out ||
        !out->sched_event_hash || !out->ready_hash || !out->running_hash ||
        !out->blocked_hash || !out->sleep_hash || !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    const WsrProblemSpec& sp = st->spec;
    wsr_ref_step_kernel<<<1, 128, 0, stream>>>(
        *st, sp.W, sp.local_cap_per_worker, sp.global_cap, sp.max_tasks,
        wsr_ref_maxkeys(&sp), wsr_ref_maxlen(&sp), wsr_ref_maxsleep(&sp), sp.max_blocked,
        run->op_kind, run->a_worker, run->a_priority, run->a_limit,
        run->a_task, run->a_work, run->a_key, run->a_tick, run->a_delta,
        out->counts, out->op_index_out, out->clock_out, out->event_seq_out,
        out->sched_event_hash, out->ready_hash, out->running_hash, out->blocked_hash,
        out->sleep_hash, out->state_checksum);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return wsr_ref_do_reset((wsr_ref::State*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    wsr_ref::State* st = (wsr_ref::State*)state;
    cudaFree(st->scal); cudaFree(st->t_live); cudaFree(st->t_id); cudaFree(st->t_home);
    cudaFree(st->t_prio); cudaFree(st->t_rem); cudaFree(st->t_state); cudaFree(st->t_birth);
    cudaFree(st->t_lastrun); cudaFree(st->t_blockseq); cudaFree(st->t_sleepseq);
    cudaFree(st->t_waitkey); cudaFree(st->t_waketick);
    cudaFree(st->lq_buf); cudaFree(st->lq_head); cudaFree(st->lq_size);
    cudaFree(st->gq_buf); cudaFree(st->gq_head); cudaFree(st->gq_size);
    cudaFree(st->run_has); cudaFree(st->run_id);
    cudaFree(st->bk_key); cudaFree(st->bk_count); cudaFree(st->bk_head); cudaFree(st->bk_ring);
    cudaFree(st->sl_ids); cudaFree(st->sl_count); cudaFree(st->counts); cudaFree(st->idmap);
    free(st);
}
