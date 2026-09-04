// PMPP_CANARY_51_4ff1c4d0e3 -- held-out canary; MUST NOT appear in any submission
// file: mesi_directory_reference.cu
//
// Reference implementation: direct in-place mutation of persistent device
// state on a single (block 0, thread 0) lane. Sharer / target sets are u64
// bitmasks (core_count <= 64). All outputs are exact integers.

#include "mesi_directory_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct MesiRefState {
    MesiProblemSpec spec;

    int64_t* memory_value;   // [L]

    int32_t* cache_line;     // [core*cap+slot]  (-1 empty)
    uint8_t* cache_state;    // [..]  (MESI_STATE_NONE empty)
    int64_t* cache_value;    // [..]
    uint64_t* cache_touch;   // [..]

    int32_t* owner;          // [L] (-1 none)
    uint64_t* sharer_mask;   // [L]

    uint8_t* pend_active;    // [L]
    uint64_t* pend_txn;      // [L]
    int32_t* pend_req;       // [L]
    int64_t* pend_newval;    // [L]
    int64_t* pend_supplier;  // [L]
    uint64_t* pend_start;    // [L]
    uint64_t* pend_target;   // [L] target mask

    uint64_t* scalars;       // [4] = {event_seq, touch_seq_next, txn_seq_next, run_event_hash}
    int64_t* counts;         // [MESI_COUNT_FIELDS]
};

#define REF_SC_EVENT_SEQ 0
#define REF_SC_TOUCH 1
#define REF_SC_TXN 2
#define REF_SC_EVHASH 3

__device__ __forceinline__ uint64_t mref_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= 1099511628211ULL;
    return h;
}

__device__ void mref_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = (const uint8_t*)ptr;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = mref_fnv_byte(v, p[i]);
    *h = v;
}

struct MrefCtx {
    MesiRefState st;
    int C, L, CAP, MAXP;
    uint64_t event_seq;
    uint64_t touch_next;
    uint64_t txn_next;
    uint64_t evhash;
};

__device__ void mref_emit(MrefCtx* x, uint8_t kind, uint32_t op_index, uint32_t core,
                          uint64_t line, uint8_t state, int64_t value, uint64_t txn,
                          uint64_t aux) {
    uint64_t h = x->evhash;
    uint64_t seq = x->event_seq;
    mref_fnv_bytes(&h, &kind, sizeof(uint8_t));
    mref_fnv_bytes(&h, &seq, sizeof(uint64_t));
    mref_fnv_bytes(&h, &op_index, sizeof(uint32_t));
    mref_fnv_bytes(&h, &core, sizeof(uint32_t));
    mref_fnv_bytes(&h, &line, sizeof(uint64_t));
    mref_fnv_bytes(&h, &state, sizeof(uint8_t));
    mref_fnv_bytes(&h, &value, sizeof(int64_t));
    mref_fnv_bytes(&h, &txn, sizeof(uint64_t));
    mref_fnv_bytes(&h, &aux, sizeof(uint64_t));
    x->evhash = h;
    x->event_seq += 1;
}

#define NO_CORE 0xFFFFFFFFu
#define NO_STATE 255
#define NO_VAL ((int64_t)INT64_MIN)
#define NO_TXN 0xFFFFFFFFFFFFFFFFULL
#define NO_LINE 0xFFFFFFFFFFFFFFFFULL

__device__ int mref_find_slot(MrefCtx* x, int core, int line) {
    const int base = core * x->CAP;
    for (int s = 0; s < x->CAP; ++s) {
        if (x->st.cache_state[base + s] != MESI_STATE_NONE &&
            x->st.cache_line[base + s] == line) {
            return base + s;
        }
    }
    return -1;
}

__device__ int mref_free_slot(MrefCtx* x, int core) {
    const int base = core * x->CAP;
    for (int s = 0; s < x->CAP; ++s) {
        if (x->st.cache_state[base + s] == MESI_STATE_NONE) return base + s;
    }
    return -1;
}

__device__ int mref_count_used(MrefCtx* x, int core) {
    const int base = core * x->CAP;
    int n = 0;
    for (int s = 0; s < x->CAP; ++s) {
        if (x->st.cache_state[base + s] != MESI_STATE_NONE) ++n;
    }
    return n;
}

__device__ int mref_active_pending(MrefCtx* x) {
    int n = 0;
    for (int l = 0; l < x->L; ++l) if (x->st.pend_active[l]) ++n;
    return n;
}

__device__ void mref_clear_slot(MrefCtx* x, int slot) {
    x->st.cache_line[slot] = -1;
    x->st.cache_state[slot] = MESI_STATE_NONE;
    x->st.cache_value[slot] = 0;
    x->st.cache_touch[slot] = 0;
}

__device__ void mref_install(MrefCtx* x, int core, int line, uint8_t state, int64_t value) {
    int slot = mref_find_slot(x, core, line);
    if (slot < 0) slot = mref_free_slot(x, core);
    x->st.cache_line[slot] = line;
    x->st.cache_state[slot] = state;
    x->st.cache_value[slot] = value;
    x->st.cache_touch[slot] = x->touch_next++;
}

__device__ void mref_capacity_evict(MrefCtx* x, uint32_t op_index, int core) {
    const int base = core * x->CAP;
    int victim = -1;
    uint64_t best_touch = 0;
    int best_line = 0;
    for (int s = 0; s < x->CAP; ++s) {
        if (x->st.cache_state[base + s] == MESI_STATE_NONE) continue;
        uint64_t t = x->st.cache_touch[base + s];
        int ln = x->st.cache_line[base + s];
        if (victim < 0 || t < best_touch || (t == best_touch && ln < best_line)) {
            victim = base + s;
            best_touch = t;
            best_line = ln;
        }
    }
    if (victim < 0) return;
    int vline = x->st.cache_line[victim];
    uint8_t vstate = x->st.cache_state[victim];
    int64_t vvalue = x->st.cache_value[victim];
    uint64_t vtouch = x->st.cache_touch[victim];

    if (vstate == MESI_STATE_M) {
        x->st.memory_value[vline] = vvalue;
        if (x->st.owner[vline] == core) x->st.owner[vline] = -1;
    } else if (vstate == MESI_STATE_E) {
        if (x->st.owner[vline] == core) x->st.owner[vline] = -1;
    } else {
        x->st.sharer_mask[vline] &= ~(1ULL << core);
    }
    mref_clear_slot(x, victim);
    x->st.counts[MC_capacity_evictions] += 1;
    mref_emit(x, EV_CAPACITY_EVICT, op_index, (uint32_t)core, (uint64_t)vline,
              vstate, vvalue, NO_TXN, vtouch);
}

__device__ void mref_load(MrefCtx* x, uint32_t op_index, int core, int line) {
    if (core < 0 || core >= x->C || line < 0 || line >= x->L) {
        x->st.counts[MC_invalid_count] += 1;
        mref_emit(x, EV_INVALID, op_index,
                  (core < 0 || core >= x->C) ? NO_CORE : (uint32_t)core,
                  (line < 0 || line >= x->L) ? NO_LINE : (uint64_t)line,
                  NO_STATE, NO_VAL, NO_TXN, 0);
        return;
    }
    if (x->st.pend_active[line]) {
        x->st.counts[MC_load_stall_pending] += 1;
        mref_emit(x, EV_LOAD_STALL_PENDING, op_index, (uint32_t)core, (uint64_t)line,
                  NO_STATE, NO_VAL, NO_TXN, 0);
        return;
    }
    int slot = mref_find_slot(x, core, line);
    if (slot >= 0) {
        x->st.cache_touch[slot] = x->touch_next++;
        x->st.counts[MC_load_hit] += 1;
        mref_emit(x, EV_LOAD_HIT, op_index, (uint32_t)core, (uint64_t)line,
                  x->st.cache_state[slot], x->st.cache_value[slot], NO_TXN, 0);
        return;
    }
    if (mref_count_used(x, core) >= x->CAP) mref_capacity_evict(x, op_index, core);

    int own = x->st.owner[line];
    if (own >= 0) {
        int oslot = mref_find_slot(x, own, line);
        if (oslot >= 0 && x->st.cache_state[oslot] == MESI_STATE_M) {
            int64_t sv = x->st.cache_value[oslot];
            x->st.memory_value[line] = sv;
            x->st.cache_state[oslot] = MESI_STATE_S;
            x->st.owner[line] = -1;
            x->st.sharer_mask[line] |= (1ULL << own);
            x->st.sharer_mask[line] |= (1ULL << core);
            mref_install(x, core, line, MESI_STATE_S, sv);
            x->st.counts[MC_downgrade_writeback] += 1;
            mref_emit(x, EV_DOWNGRADE_WRITEBACK, op_index, (uint32_t)own, (uint64_t)line,
                      MESI_STATE_S, sv, NO_TXN, 0);
            x->st.counts[MC_load_miss_shared] += 1;
            mref_emit(x, EV_LOAD_MISS_SHARED, op_index, (uint32_t)core, (uint64_t)line,
                      MESI_STATE_S, sv, NO_TXN, 0);
            return;
        } else if (oslot >= 0 && x->st.cache_state[oslot] == MESI_STATE_E) {
            int64_t sv = x->st.cache_value[oslot];
            x->st.cache_state[oslot] = MESI_STATE_S;
            x->st.owner[line] = -1;
            x->st.sharer_mask[line] |= (1ULL << own);
            x->st.sharer_mask[line] |= (1ULL << core);
            mref_install(x, core, line, MESI_STATE_S, sv);
            x->st.counts[MC_downgrade_clean] += 1;
            mref_emit(x, EV_DOWNGRADE_CLEAN, op_index, (uint32_t)own, (uint64_t)line,
                      MESI_STATE_S, sv, NO_TXN, 0);
            x->st.counts[MC_load_miss_shared] += 1;
            mref_emit(x, EV_LOAD_MISS_SHARED, op_index, (uint32_t)core, (uint64_t)line,
                      MESI_STATE_S, sv, NO_TXN, 0);
            return;
        }
    }
    if (x->st.sharer_mask[line] != 0ULL) {
        int64_t val = x->st.memory_value[line];
        mref_install(x, core, line, MESI_STATE_S, val);
        x->st.sharer_mask[line] |= (1ULL << core);
        x->st.counts[MC_load_miss_shared] += 1;
        mref_emit(x, EV_LOAD_MISS_SHARED, op_index, (uint32_t)core, (uint64_t)line,
                  MESI_STATE_S, val, NO_TXN, 0);
        return;
    }
    int64_t val = x->st.memory_value[line];
    mref_install(x, core, line, MESI_STATE_E, val);
    x->st.owner[line] = core;
    x->st.counts[MC_load_miss_exclusive] += 1;
    mref_emit(x, EV_LOAD_MISS_EXCLUSIVE, op_index, (uint32_t)core, (uint64_t)line,
              MESI_STATE_E, val, NO_TXN, 0);
}

__device__ int mref_popcount(uint64_t m) {
    int n = 0;
    while (m) { n += (int)(m & 1ULL); m >>= 1; }
    return n;
}

__device__ void mref_store(MrefCtx* x, uint32_t op_index, int core, int line, int64_t value) {
    if (core < 0 || core >= x->C || line < 0 || line >= x->L) {
        x->st.counts[MC_invalid_count] += 1;
        mref_emit(x, EV_INVALID, op_index,
                  (core < 0 || core >= x->C) ? NO_CORE : (uint32_t)core,
                  (line < 0 || line >= x->L) ? NO_LINE : (uint64_t)line,
                  NO_STATE, NO_VAL, NO_TXN, 0);
        return;
    }
    if (x->st.pend_active[line]) {
        x->st.counts[MC_store_stall_pending] += 1;
        mref_emit(x, EV_STORE_STALL_PENDING, op_index, (uint32_t)core, (uint64_t)line,
                  NO_STATE, value, NO_TXN, 0);
        return;
    }
    int slot = mref_find_slot(x, core, line);
    if (slot >= 0 && x->st.cache_state[slot] == MESI_STATE_M) {
        x->st.cache_value[slot] = value;
        x->st.cache_touch[slot] = x->touch_next++;
        x->st.counts[MC_store_hit_modified] += 1;
        mref_emit(x, EV_STORE_HIT_MODIFIED, op_index, (uint32_t)core, (uint64_t)line,
                  MESI_STATE_M, value, NO_TXN, 0);
        return;
    }
    if (slot >= 0 && x->st.cache_state[slot] == MESI_STATE_E) {
        x->st.cache_state[slot] = MESI_STATE_M;
        x->st.cache_value[slot] = value;
        x->st.cache_touch[slot] = x->touch_next++;
        x->st.owner[line] = core;
        x->st.counts[MC_store_hit_exclusive] += 1;
        mref_emit(x, EV_STORE_HIT_EXCLUSIVE, op_index, (uint32_t)core, (uint64_t)line,
                  MESI_STATE_M, value, NO_TXN, 0);
        return;
    }
    // S or absent: build target mask = sharers\{core} U {owner if owner!=core}
    uint64_t targets = x->st.sharer_mask[line] & ~(1ULL << core);
    if (x->st.owner[line] >= 0 && x->st.owner[line] != core) {
        targets |= (1ULL << x->st.owner[line]);
    }
    if (targets == 0ULL) {
        if (mref_find_slot(x, core, line) < 0 && mref_count_used(x, core) >= x->CAP) {
            mref_capacity_evict(x, op_index, core);
        }
        mref_install(x, core, line, MESI_STATE_M, value);
        x->st.sharer_mask[line] = 0ULL;
        x->st.owner[line] = core;
        x->st.counts[MC_store_committed] += 1;
        mref_emit(x, EV_STORE_COMMIT, op_index, (uint32_t)core, (uint64_t)line,
                  MESI_STATE_M, value, NO_TXN, 0);
        return;
    }
    if (mref_active_pending(x) >= x->MAXP) {
        x->st.counts[MC_store_stall_pending] += 1;
        mref_emit(x, EV_STORE_STALL_PENDING, op_index, (uint32_t)core, (uint64_t)line,
                  NO_STATE, value, NO_TXN, 0);
        return;
    }
    uint64_t tid = x->txn_next++;
    x->st.pend_active[line] = 1;
    x->st.pend_txn[line] = tid;
    x->st.pend_req[line] = core;
    x->st.pend_newval[line] = value;
    x->st.pend_supplier[line] = NO_VAL;
    x->st.pend_start[line] = x->event_seq;
    x->st.pend_target[line] = targets;

    x->st.counts[MC_store_pending] += 1;
    int tcount = mref_popcount(targets);
    for (int t = 0; t < x->C; ++t) {
        if (targets & (1ULL << t)) {
            x->st.counts[MC_inv_sent] += 1;
            mref_emit(x, EV_INV_SEND, op_index, (uint32_t)t, (uint64_t)line,
                      MESI_STATE_NONE, value, tid, (uint64_t)core);
        }
    }
    mref_emit(x, EV_STORE_PENDING, op_index, (uint32_t)core, (uint64_t)line,
              MESI_STATE_NONE, value, tid, (uint64_t)tcount);
}

__device__ void mref_ack(MrefCtx* x, uint32_t op_index, int core, int line, uint64_t txn) {
    bool valid = true;
    if (core < 0 || core >= x->C || line < 0 || line >= x->L) valid = false;
    else if (!x->st.pend_active[line]) valid = false;
    else if (x->st.pend_txn[line] != txn) valid = false;
    else if (!(x->st.pend_target[line] & (1ULL << core))) valid = false;

    if (!valid) {
        x->st.counts[MC_invalid_count] += 1;
        mref_emit(x, EV_INVALID, op_index,
                  (core < 0 || core >= x->C) ? NO_CORE : (uint32_t)core,
                  (line < 0 || line >= x->L) ? NO_LINE : (uint64_t)line,
                  NO_STATE, NO_VAL, txn, 0);
        return;
    }
    int slot = mref_find_slot(x, core, line);
    if (slot >= 0) {
        if (x->st.cache_state[slot] == MESI_STATE_M) {
            int64_t v = x->st.cache_value[slot];
            x->st.pend_supplier[line] = v;
            x->st.memory_value[line] = v;
            mref_emit(x, EV_DATA_SUPPLY_DIRTY, op_index, (uint32_t)core, (uint64_t)line,
                      MESI_STATE_M, v, x->st.pend_txn[line], 0);
            x->st.counts[MC_data_supply_dirty] += 1;
        } else if (x->st.cache_state[slot] == MESI_STATE_E) {
            int64_t v = x->st.cache_value[slot];
            x->st.pend_supplier[line] = v;
            mref_emit(x, EV_DATA_SUPPLY_CLEAN, op_index, (uint32_t)core, (uint64_t)line,
                      MESI_STATE_E, v, x->st.pend_txn[line], 0);
            x->st.counts[MC_data_supply_clean] += 1;
        }
        mref_clear_slot(x, slot);
    }
    x->st.sharer_mask[line] &= ~(1ULL << core);
    if (x->st.owner[line] == core) x->st.owner[line] = -1;

    x->st.pend_target[line] &= ~(1ULL << core);
    x->st.counts[MC_inv_acked] += 1;
    int remaining = mref_popcount(x->st.pend_target[line]);
    mref_emit(x, EV_INV_ACK, op_index, (uint32_t)core, (uint64_t)line,
              NO_STATE, NO_VAL, x->st.pend_txn[line], (uint64_t)remaining);

    if (x->st.pend_target[line] == 0ULL) {
        int req = x->st.pend_req[line];
        int64_t nv = x->st.pend_newval[line];
        uint64_t tid = x->st.pend_txn[line];
        if (mref_find_slot(x, req, line) < 0 && mref_count_used(x, req) >= x->CAP) {
            mref_capacity_evict(x, op_index, req);
        }
        mref_install(x, req, line, MESI_STATE_M, nv);
        x->st.sharer_mask[line] = 0ULL;
        x->st.owner[line] = req;
        x->st.counts[MC_store_committed] += 1;
        mref_emit(x, EV_STORE_COMMIT, op_index, (uint32_t)req, (uint64_t)line,
                  MESI_STATE_M, nv, tid, 0);
        // delete pending
        x->st.pend_active[line] = 0;
        x->st.pend_txn[line] = 0;
        x->st.pend_req[line] = -1;
        x->st.pend_newval[line] = 0;
        x->st.pend_supplier[line] = NO_VAL;
        x->st.pend_start[line] = 0;
        x->st.pend_target[line] = 0ULL;
    }
}

__device__ void mref_evict(MrefCtx* x, uint32_t op_index, int core, int line) {
    if (core < 0 || core >= x->C || line < 0 || line >= x->L) {
        x->st.counts[MC_invalid_count] += 1;
        mref_emit(x, EV_INVALID, op_index,
                  (core < 0 || core >= x->C) ? NO_CORE : (uint32_t)core,
                  (line < 0 || line >= x->L) ? NO_LINE : (uint64_t)line,
                  NO_STATE, NO_VAL, NO_TXN, 0);
        return;
    }
    if (x->st.pend_active[line] && (x->st.pend_target[line] & (1ULL << core))) {
        x->st.counts[MC_evict_stall_pending] += 1;
        mref_emit(x, EV_EVICT_STALL_PENDING, op_index, (uint32_t)core, (uint64_t)line,
                  NO_STATE, NO_VAL, NO_TXN, 0);
        return;
    }
    int slot = mref_find_slot(x, core, line);
    if (slot < 0) {
        x->st.counts[MC_evict_miss] += 1;
        mref_emit(x, EV_EVICT_MISS, op_index, (uint32_t)core, (uint64_t)line,
                  NO_STATE, NO_VAL, NO_TXN, 0);
        return;
    }
    uint8_t state = x->st.cache_state[slot];
    if (state == MESI_STATE_M) {
        int64_t wb = x->st.cache_value[slot];
        x->st.memory_value[line] = wb;
        if (x->st.owner[line] == core) x->st.owner[line] = -1;
        mref_clear_slot(x, slot);
        x->st.counts[MC_evict_writeback] += 1;
        mref_emit(x, EV_EVICT_WRITEBACK, op_index, (uint32_t)core, (uint64_t)line,
                  MESI_STATE_M, wb, NO_TXN, 0);
    } else if (state == MESI_STATE_E) {
        if (x->st.owner[line] == core) x->st.owner[line] = -1;
        mref_clear_slot(x, slot);
        x->st.counts[MC_evict_clean] += 1;
        mref_emit(x, EV_EVICT_CLEAN, op_index, (uint32_t)core, (uint64_t)line,
                  MESI_STATE_E, NO_VAL, NO_TXN, 0);
    } else {
        x->st.sharer_mask[line] &= ~(1ULL << core);
        mref_clear_slot(x, slot);
        x->st.counts[MC_evict_shared] += 1;
        mref_emit(x, EV_EVICT_SHARED, op_index, (uint32_t)core, (uint64_t)line,
                  MESI_STATE_S, NO_VAL, NO_TXN, 0);
    }
}

__device__ void mref_flush(MrefCtx* x, uint32_t op_index, int line) {
    if (line < 0 || line >= x->L || x->st.pend_active[line]) {
        x->st.counts[MC_invalid_count] += 1;
        mref_emit(x, EV_INVALID, op_index, NO_CORE,
                  (line < 0 || line >= x->L) ? NO_LINE : (uint64_t)line,
                  NO_STATE, NO_VAL, NO_TXN, 0);
        return;
    }
    int own = x->st.owner[line];
    if (own >= 0) {
        int oslot = mref_find_slot(x, own, line);
        if (oslot >= 0 && x->st.cache_state[oslot] == MESI_STATE_M) {
            int64_t wb = x->st.cache_value[oslot];
            x->st.memory_value[line] = wb;
            x->st.cache_state[oslot] = MESI_STATE_S;
            x->st.owner[line] = -1;
            x->st.sharer_mask[line] |= (1ULL << own);
            x->st.counts[MC_flush_writeback] += 1;
            mref_emit(x, EV_FLUSH_WRITEBACK, op_index, (uint32_t)own, (uint64_t)line,
                      MESI_STATE_S, wb, NO_TXN, 0);
            return;
        } else if (oslot >= 0 && x->st.cache_state[oslot] == MESI_STATE_E) {
            x->st.cache_state[oslot] = MESI_STATE_S;
            x->st.owner[line] = -1;
            x->st.sharer_mask[line] |= (1ULL << own);
            x->st.counts[MC_flush_clean] += 1;
            mref_emit(x, EV_FLUSH_CLEAN, op_index, (uint32_t)own, (uint64_t)line,
                      MESI_STATE_S, NO_VAL, NO_TXN, 0);
            return;
        }
    }
    x->st.counts[MC_flush_noop] += 1;
    mref_emit(x, EV_FLUSH_NOOP, op_index, NO_CORE, (uint64_t)line,
              NO_STATE, NO_VAL, NO_TXN, 0);
}

// ---- hashes ----
__device__ uint64_t mref_cache_hash(MrefCtx* x) {
    uint64_t h = 1469598103934665603ULL;
    for (int c = 0; c < x->C; ++c) {
        const int base = c * x->CAP;
        // emit entries in ascending line order via repeated min-scan
        int prev_line = INT_MIN;
        for (;;) {
            int best = -1;
            int best_line = 0;
            for (int s = 0; s < x->CAP; ++s) {
                if (x->st.cache_state[base + s] == MESI_STATE_NONE) continue;
                int ln = x->st.cache_line[base + s];
                if (ln <= prev_line) continue;
                if (best < 0 || ln < best_line) { best = base + s; best_line = ln; }
            }
            if (best < 0) break;
            uint32_t core = (uint32_t)c;
            uint64_t line = (uint64_t)best_line;
            uint8_t state = x->st.cache_state[best];
            int64_t value = x->st.cache_value[best];
            uint64_t touch = x->st.cache_touch[best];
            mref_fnv_bytes(&h, &core, sizeof(uint32_t));
            mref_fnv_bytes(&h, &line, sizeof(uint64_t));
            mref_fnv_bytes(&h, &state, sizeof(uint8_t));
            mref_fnv_bytes(&h, &value, sizeof(int64_t));
            mref_fnv_bytes(&h, &touch, sizeof(uint64_t));
            prev_line = best_line;
        }
    }
    return h;
}

__device__ uint64_t mref_directory_hash(MrefCtx* x) {
    uint64_t h = 1469598103934665603ULL;
    for (int l = 0; l < x->L; ++l) {
        uint64_t line = (uint64_t)l;
        int64_t mv = x->st.memory_value[l];
        uint32_t own = (x->st.owner[l] < 0) ? 0xFFFFFFFFu : (uint32_t)x->st.owner[l];
        uint64_t mask = x->st.sharer_mask[l];
        uint64_t sc = (uint64_t)mref_popcount(mask);
        mref_fnv_bytes(&h, &line, sizeof(uint64_t));
        mref_fnv_bytes(&h, &mv, sizeof(int64_t));
        mref_fnv_bytes(&h, &own, sizeof(uint32_t));
        mref_fnv_bytes(&h, &sc, sizeof(uint64_t));
        for (int c = 0; c < x->C; ++c) {
            if (mask & (1ULL << c)) {
                uint32_t su = (uint32_t)c;
                mref_fnv_bytes(&h, &su, sizeof(uint32_t));
            }
        }
    }
    return h;
}

__device__ uint64_t mref_pending_hash(MrefCtx* x) {
    uint64_t h = 1469598103934665603ULL;
    for (int l = 0; l < x->L; ++l) {
        if (!x->st.pend_active[l]) continue;
        uint64_t line = (uint64_t)l;
        uint64_t tid = x->st.pend_txn[l];
        uint32_t req = (uint32_t)x->st.pend_req[l];
        int64_t nv = x->st.pend_newval[l];
        int64_t sv = x->st.pend_supplier[l];
        uint64_t ss = x->st.pend_start[l];
        mref_fnv_bytes(&h, &line, sizeof(uint64_t));
        mref_fnv_bytes(&h, &tid, sizeof(uint64_t));
        mref_fnv_bytes(&h, &req, sizeof(uint32_t));
        mref_fnv_bytes(&h, &nv, sizeof(int64_t));
        mref_fnv_bytes(&h, &sv, sizeof(int64_t));
        mref_fnv_bytes(&h, &ss, sizeof(uint64_t));
        uint64_t mask = x->st.pend_target[l];
        for (int c = 0; c < x->C; ++c) {
            if (mask & (1ULL << c)) {
                uint32_t tu = (uint32_t)c;
                mref_fnv_bytes(&h, &tu, sizeof(uint32_t));
            }
        }
    }
    return h;
}

__device__ uint64_t mref_state_checksum(MrefCtx* x, uint64_t ch, uint64_t dh, uint64_t ph) {
    uint64_t h = 1469598103934665603ULL;
    int cc = x->C, lc = x->L, cap = x->CAP, mp = x->MAXP;
    mref_fnv_bytes(&h, &cc, sizeof(int32_t));
    mref_fnv_bytes(&h, &lc, sizeof(int32_t));
    mref_fnv_bytes(&h, &cap, sizeof(int32_t));
    mref_fnv_bytes(&h, &mp, sizeof(int32_t));
    mref_fnv_bytes(&h, &x->event_seq, sizeof(uint64_t));
    mref_fnv_bytes(&h, &x->touch_next, sizeof(uint64_t));
    mref_fnv_bytes(&h, &x->txn_next, sizeof(uint64_t));
    mref_fnv_bytes(&h, &ch, sizeof(uint64_t));
    mref_fnv_bytes(&h, &dh, sizeof(uint64_t));
    mref_fnv_bytes(&h, &ph, sizeof(uint64_t));
    for (int i = 0; i < MESI_COUNT_FIELDS; ++i) {
        int64_t c = x->st.counts[i];
        mref_fnv_bytes(&h, &c, sizeof(int64_t));
    }
    return h;
}

__global__ void mref_reset_kernel(MesiRefState st, int C, int L, int CAP) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int l = 0; l < L; ++l) {
        st.memory_value[l] = (int64_t)(1000 + l * 7);
        st.owner[l] = -1;
        st.sharer_mask[l] = 0ULL;
        st.pend_active[l] = 0;
        st.pend_txn[l] = 0;
        st.pend_req[l] = -1;
        st.pend_newval[l] = 0;
        st.pend_supplier[l] = (int64_t)INT64_MIN;
        st.pend_start[l] = 0;
        st.pend_target[l] = 0ULL;
    }
    for (int i = 0; i < C * CAP; ++i) {
        st.cache_line[i] = -1;
        st.cache_state[i] = MESI_STATE_NONE;
        st.cache_value[i] = 0;
        st.cache_touch[i] = 0;
    }
    for (int i = 0; i < MESI_COUNT_FIELDS; ++i) st.counts[i] = 0;
    st.scalars[REF_SC_EVENT_SEQ] = 0;
    st.scalars[REF_SC_TOUCH] = 1;
    st.scalars[REF_SC_TXN] = 1;
    st.scalars[REF_SC_EVHASH] = 1469598103934665603ULL;
}

__global__ void mref_step_kernel(
    MesiRefState st, int C, int L, int CAP, int MAXP, int batch_size,
    const int32_t* op, const int32_t* arg_core, const int32_t* arg_line,
    const int64_t* arg_value, const uint64_t* arg_txn,
    int64_t* out_counts, uint64_t* out_coh, uint64_t* out_cache,
    uint64_t* out_dir, uint64_t* out_pend, uint64_t* out_evseq,
    uint64_t* out_state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    MrefCtx x;
    x.st = st;
    x.C = C; x.L = L; x.CAP = CAP; x.MAXP = MAXP;
    x.event_seq = st.scalars[REF_SC_EVENT_SEQ];
    x.touch_next = st.scalars[REF_SC_TOUCH];
    x.txn_next = st.scalars[REF_SC_TXN];
    x.evhash = st.scalars[REF_SC_EVHASH];

    for (int i = 0; i < batch_size; ++i) {
        int o = op[i];
        int core = arg_core[i];
        int line = arg_line[i];
        int64_t value = arg_value[i];
        uint64_t txn = arg_txn[i];
        if (o == MESI_OP_LOAD) mref_load(&x, (uint32_t)i, core, line);
        else if (o == MESI_OP_STORE) mref_store(&x, (uint32_t)i, core, line, value);
        else if (o == MESI_OP_ACK_INV) mref_ack(&x, (uint32_t)i, core, line, txn);
        else if (o == MESI_OP_EVICT) mref_evict(&x, (uint32_t)i, core, line);
        else if (o == MESI_OP_FLUSH) mref_flush(&x, (uint32_t)i, line);
        else {
            x.st.counts[MC_invalid_count] += 1;
            mref_emit(&x, EV_INVALID, (uint32_t)i, NO_CORE, NO_LINE,
                      NO_STATE, NO_VAL, NO_TXN, 0);
        }
    }

    st.scalars[REF_SC_EVENT_SEQ] = x.event_seq;
    st.scalars[REF_SC_TOUCH] = x.touch_next;
    st.scalars[REF_SC_TXN] = x.txn_next;
    st.scalars[REF_SC_EVHASH] = x.evhash;

    uint64_t ch = mref_cache_hash(&x);
    uint64_t dh = mref_directory_hash(&x);
    uint64_t ph = mref_pending_hash(&x);

    for (int i = 0; i < MESI_COUNT_FIELDS; ++i) out_counts[i] = x.st.counts[i];
    out_coh[0] = x.evhash;
    out_cache[0] = ch;
    out_dir[0] = dh;
    out_pend[0] = ph;
    out_evseq[0] = x.event_seq;
    out_state[0] = mref_state_checksum(&x, ch, dh, ph);
}

static cudaError_t mref_do_reset(MesiRefState* st, cudaStream_t stream) {
    mref_reset_kernel<<<1, 1, 0, stream>>>(*st, st->spec.core_count,
                                           st->spec.line_count,
                                           st->spec.cache_capacity_per_core);
    return cudaPeekAtLastError();
}

extern "C" size_t solution_workspace_bytes(const MesiProblemSpec* spec) {
    if (!mesi_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const MesiProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!mesi_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;

    MesiRefState* st = (MesiRefState*)malloc(sizeof(MesiRefState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(MesiRefState));
    memcpy(&st->spec, spec, sizeof(MesiProblemSpec));

    const size_t L = (size_t)spec->line_count;
    const size_t CC = (size_t)spec->core_count * (size_t)spec->cache_capacity_per_core;
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&st->memory_value, sizeof(int64_t) * L); if (err) goto fail;
    err = cudaMalloc((void**)&st->cache_line, sizeof(int32_t) * CC); if (err) goto fail;
    err = cudaMalloc((void**)&st->cache_state, sizeof(uint8_t) * CC); if (err) goto fail;
    err = cudaMalloc((void**)&st->cache_value, sizeof(int64_t) * CC); if (err) goto fail;
    err = cudaMalloc((void**)&st->cache_touch, sizeof(uint64_t) * CC); if (err) goto fail;
    err = cudaMalloc((void**)&st->owner, sizeof(int32_t) * L); if (err) goto fail;
    err = cudaMalloc((void**)&st->sharer_mask, sizeof(uint64_t) * L); if (err) goto fail;
    err = cudaMalloc((void**)&st->pend_active, sizeof(uint8_t) * L); if (err) goto fail;
    err = cudaMalloc((void**)&st->pend_txn, sizeof(uint64_t) * L); if (err) goto fail;
    err = cudaMalloc((void**)&st->pend_req, sizeof(int32_t) * L); if (err) goto fail;
    err = cudaMalloc((void**)&st->pend_newval, sizeof(int64_t) * L); if (err) goto fail;
    err = cudaMalloc((void**)&st->pend_supplier, sizeof(int64_t) * L); if (err) goto fail;
    err = cudaMalloc((void**)&st->pend_start, sizeof(uint64_t) * L); if (err) goto fail;
    err = cudaMalloc((void**)&st->pend_target, sizeof(uint64_t) * L); if (err) goto fail;
    err = cudaMalloc((void**)&st->scalars, sizeof(uint64_t) * 4); if (err) goto fail;
    err = cudaMalloc((void**)&st->counts, sizeof(int64_t) * MESI_COUNT_FIELDS); if (err) goto fail;

    err = mref_do_reset(st, stream);
    if (err) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    solution_destroy(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state, const MesiRunSpec* run, const void* inputs_void,
    void* outputs_void, void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;
    if (!state || !inputs_void || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 1) return cudaErrorInvalidValue;

    MesiRefState* st = (MesiRefState*)state;
    if (!mesi_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const MesiInputs* in = (const MesiInputs*)inputs_void;
    MesiOutputs* out = (MesiOutputs*)outputs_void;

    if (run->batch_size > 0 &&
        (!in->op || !in->arg_core || !in->arg_line || !in->arg_value || !in->arg_txn)) {
        return cudaErrorInvalidValue;
    }
    if (!out->counts || !out->coh_event_hash || !out->cache_hash ||
        !out->directory_hash || !out->pending_hash || !out->event_seq_out ||
        !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    mref_step_kernel<<<1, 1, 0, stream>>>(
        *st, st->spec.core_count, st->spec.line_count,
        st->spec.cache_capacity_per_core, st->spec.max_pending_lines,
        run->batch_size, in->op, in->arg_core, in->arg_line, in->arg_value,
        in->arg_txn, out->counts, out->coh_event_hash, out->cache_hash,
        out->directory_hash, out->pending_hash, out->event_seq_out,
        out->state_checksum);

    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return mref_do_reset((MesiRefState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    MesiRefState* st = (MesiRefState*)state;
    if (st->memory_value) cudaFree(st->memory_value);
    if (st->cache_line) cudaFree(st->cache_line);
    if (st->cache_state) cudaFree(st->cache_state);
    if (st->cache_value) cudaFree(st->cache_value);
    if (st->cache_touch) cudaFree(st->cache_touch);
    if (st->owner) cudaFree(st->owner);
    if (st->sharer_mask) cudaFree(st->sharer_mask);
    if (st->pend_active) cudaFree(st->pend_active);
    if (st->pend_txn) cudaFree(st->pend_txn);
    if (st->pend_req) cudaFree(st->pend_req);
    if (st->pend_newval) cudaFree(st->pend_newval);
    if (st->pend_supplier) cudaFree(st->pend_supplier);
    if (st->pend_start) cudaFree(st->pend_start);
    if (st->pend_target) cudaFree(st->pend_target);
    if (st->scalars) cudaFree(st->scalars);
    if (st->counts) cudaFree(st->counts);
    free(st);
}
