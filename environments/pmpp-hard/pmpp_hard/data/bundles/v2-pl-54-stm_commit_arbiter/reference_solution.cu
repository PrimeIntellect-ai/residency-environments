// PMPP_CANARY_54_f1ce07a0c9 -- held-out canary; MUST NOT appear in any submission
// file: stm_commit_arbiter_reference.cu
//
// Reference GPU implementation of the T54 STM Commit Arbiter. A single-threaded
// device kernel keeps the full persistent engine state in flat device arrays:
//   - location table (open slots, looked up by linear scan over addr)
//   - transaction table (slotted; per-txn read/write sets as fixed sub-arrays)
//   - lock wait queues + retry watch queues (per-location ring of entries kept
//     in seq order by append)
// Orderings (ascending addr, queue order, prepared arbitration) are realized by
// explicit linear scans / selection. Persistent state lives in device memory
// across steps. Shares no algorithm code with the std::map oracle or the naive
// snapshot/rebuild implementation.

#include "stm_commit_arbiter_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---- FNV helpers ----
__device__ __forceinline__ uint64_t rf_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= STM_FNV_PRIME; return h;
}
__device__ void rf_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*)p; uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = rf_byte(v, q[i]);
    *h = v;
}
__device__ __forceinline__ void rf_u8(uint64_t* h, uint8_t v)   { rf_bytes(h, &v, 1); }
__device__ __forceinline__ void rf_u32(uint64_t* h, uint32_t v) { rf_bytes(h, &v, 4); }
__device__ __forceinline__ void rf_u64(uint64_t* h, uint64_t v) { rf_bytes(h, &v, 8); }
__device__ __forceinline__ void rf_i64(uint64_t* h, int64_t v)  { rf_bytes(h, &v, 8); }

struct StmRefState {
    StmProblemSpec spec;

    // Global counters: [global_version, event_seq, xid_next, begin_seq_next,
    //                   write_seq_next, wait_seq_next, retry_seq_next]
    uint64_t* counters;  // 7
    uint64_t* op_index;  // 1

    // Location table (capacity max_locations). Slot used iff loc_used==1.
    uint8_t*  loc_used;
    uint64_t* loc_addr;
    int64_t*  loc_value;
    uint64_t* loc_version;
    uint64_t* loc_lock_owner;  // 0 = unlocked
    uint64_t* loc_write_seq;

    // Transaction table (capacity max_txns).
    uint8_t*  txn_used;
    uint64_t* txn_id;
    uint64_t* txn_xid;
    uint64_t* txn_begin_seq;
    uint64_t* txn_attempt_no;
    uint32_t* txn_priority;
    uint64_t* txn_start_version;
    uint8_t*  txn_status;
    uint64_t* txn_wait_addr;
    uint64_t* txn_wait_seq;
    uint64_t* txn_prepare_seq;

    // Read set per txn (max_txns * max_read_set).
    uint8_t*  rs_used;
    uint64_t* rs_addr;
    uint64_t* rs_read_version;
    int64_t*  rs_read_value;
    uint8_t*  rs_from_own_write;

    // Write set per txn (max_txns * max_write_set).
    uint8_t*  ws_used;
    uint64_t* ws_addr;
    int64_t*  ws_value;
    uint64_t* ws_write_set_seq;

    // Lock wait queues per location (max_locations * max_waiters). Stored in
    // queue order via a count + contiguous fill. Lock waits only ever target
    // already-materialized locations, so embedding them per location slot is OK.
    int32_t*  lq_count;     // per addr-slot count
    uint64_t* lq_addr;      // addr key per addr-slot (mirrors loc table index)
    uint64_t* lq_txn;       // [addr_slot * max_waiters + pos]
    uint64_t* lq_seq;

    // Retry watch queues: standalone addr-keyed entries (NOT tied to the
    // location table; watching an absent address must not materialize it).
    // Capacity = max_txns * max_watch_set. Order within an addr = seq asc, ins
    // asc.
    uint8_t*  rq_used;
    uint64_t* rq_eaddr;
    uint64_t* rq_etxn;
    uint64_t* rq_eseq;
    uint64_t* rq_eins;
    uint64_t* qins;         // [1] monotone insertion counter for retry queue
};

// Device-side flat view.
struct StmRefDev {
    int32_t max_txns, max_locations, max_read_set, max_write_set, max_watch_set;
    int32_t max_waiters, max_retry_watchers;

    uint64_t* counters;
    uint64_t* op_index;

    uint8_t*  loc_used;
    uint64_t* loc_addr;
    int64_t*  loc_value;
    uint64_t* loc_version;
    uint64_t* loc_lock_owner;
    uint64_t* loc_write_seq;

    uint8_t*  txn_used;
    uint64_t* txn_id;
    uint64_t* txn_xid;
    uint64_t* txn_begin_seq;
    uint64_t* txn_attempt_no;
    uint32_t* txn_priority;
    uint64_t* txn_start_version;
    uint8_t*  txn_status;
    uint64_t* txn_wait_addr;
    uint64_t* txn_wait_seq;
    uint64_t* txn_prepare_seq;

    uint8_t*  rs_used;
    uint64_t* rs_addr;
    uint64_t* rs_read_version;
    int64_t*  rs_read_value;
    uint8_t*  rs_from_own_write;

    uint8_t*  ws_used;
    uint64_t* ws_addr;
    int64_t*  ws_value;
    uint64_t* ws_write_set_seq;

    int32_t*  lq_count;
    uint64_t* lq_addr;
    uint64_t* lq_txn;
    uint64_t* lq_seq;

    uint8_t*  rq_used;
    uint64_t* rq_eaddr;
    uint64_t* rq_etxn;
    uint64_t* rq_eseq;
    uint64_t* rq_eins;
    uint64_t* qins;
};

// counter indices
#define C_GV   0
#define C_ES   1
#define C_XID  2
#define C_BS   3
#define C_WS   4
#define C_WAIT 5
#define C_RTY  6

struct RefCounts {
    int32_t v[23];
};
// count indices match output order
#define K_BEGUN 0
#define K_ROW   1
#define K_RSH   2
#define K_WST   3
#define K_VOK   4
#define K_ROC   5
#define K_PREP  6
#define K_WL    7
#define K_WTL   8
#define K_CD    9
#define K_LW    10
#define K_NTW   11
#define K_NTS   12
#define K_RSU   13
#define K_RIM   14
#define K_RWO   15
#define K_WR    16
#define K_WKL   17
#define K_ABT   18
#define K_PU    19
#define K_CU    20
#define K_AU    21
#define K_INV   22

// ---- location helpers ----
__device__ int rd_find_loc(const StmRefDev& s, uint64_t addr) {
    for (int i = 0; i < s.max_locations; ++i) {
        if (!s.loc_used[i]) break;  // loc slots form a used-prefix (never freed)
        if (s.loc_addr[i] == addr) return i;
    }
    return -1;
}
__device__ int rd_loc_count(const StmRefDev& s) {
    int c = 0;
    for (int i = 0; i < s.max_locations; ++i) { if (!s.loc_used[i]) break; ++c; }
    return c;
}
__device__ int rd_alloc_loc(const StmRefDev& s, uint64_t addr) {
    for (int i = 0; i < s.max_locations; ++i) {
        if (!s.loc_used[i]) {
            s.loc_used[i] = 1; s.loc_addr[i] = addr; s.loc_value[i] = 0;
            s.loc_version[i] = 0; s.loc_lock_owner[i] = 0; s.loc_write_seq[i] = 0;
            // reset its lock-wait queue (retry queue is addr-keyed, separate).
            s.lq_count[i] = 0; s.lq_addr[i] = addr;
            return i;
        }
    }
    return -1;
}
// returns existing or creates; -1 if no room.
__device__ int rd_get_or_create_loc(const StmRefDev& s, uint64_t addr) {
    int i = rd_find_loc(s, addr);
    if (i >= 0) return i;
    return rd_alloc_loc(s, addr);
}

__device__ int rd_find_txn(const StmRefDev& s, uint64_t txn_id) {
    for (int i = 0; i < s.max_txns; ++i)
        if (s.txn_used[i] && s.txn_id[i] == txn_id) return i;
    return -1;
}
__device__ int rd_txn_count(const StmRefDev& s) {
    int c = 0;
    for (int i = 0; i < s.max_txns; ++i) if (s.txn_used[i]) ++c;
    return c;
}

__device__ int rd_find_rs(const StmRefDev& s, int tslot, uint64_t addr) {
    const int base = tslot * s.max_read_set;
    for (int j = 0; j < s.max_read_set; ++j)
        if (s.rs_used[base + j] && s.rs_addr[base + j] == addr) return base + j;
    return -1;
}
__device__ int rd_alloc_rs(const StmRefDev& s, int tslot) {
    const int base = tslot * s.max_read_set;
    for (int j = 0; j < s.max_read_set; ++j)
        if (!s.rs_used[base + j]) return base + j;
    return -1;
}
__device__ int rd_find_ws(const StmRefDev& s, int tslot, uint64_t addr) {
    const int base = tslot * s.max_write_set;
    for (int j = 0; j < s.max_write_set; ++j)
        if (s.ws_used[base + j] && s.ws_addr[base + j] == addr) return base + j;
    return -1;
}
__device__ int rd_ws_count(const StmRefDev& s, int tslot) {
    const int base = tslot * s.max_write_set;
    int c = 0;
    for (int j = 0; j < s.max_write_set; ++j) if (s.ws_used[base + j]) ++c;
    return c;
}
__device__ int rd_alloc_ws(const StmRefDev& s, int tslot) {
    const int base = tslot * s.max_write_set;
    for (int j = 0; j < s.max_write_set; ++j)
        if (!s.ws_used[base + j]) return base + j;
    return -1;
}

__device__ void rd_clear_txn_slot(const StmRefDev& s, int tslot) {
    s.txn_used[tslot] = 0;
    const int rb = tslot * s.max_read_set;
    for (int j = 0; j < s.max_read_set; ++j) s.rs_used[rb + j] = 0;
    const int wb = tslot * s.max_write_set;
    for (int j = 0; j < s.max_write_set; ++j) s.ws_used[wb + j] = 0;
}

// Remove txn from all lock + retry queues (compact in place, preserving order).
__device__ void rd_remove_from_queues(const StmRefDev& s, uint64_t txn_id) {
    for (int i = 0; i < s.max_locations; ++i) {
        if (!s.loc_used[i]) break;
        // lock queue
        int lc = s.lq_count[i];
        int w = 0;
        const int lb = i * s.max_waiters;
        for (int p = 0; p < lc; ++p) {
            if (s.lq_txn[lb + p] == txn_id) continue;
            if (w != p) { s.lq_txn[lb + w] = s.lq_txn[lb + p]; s.lq_seq[lb + w] = s.lq_seq[lb + p]; }
            ++w;
        }
        s.lq_count[i] = w;
    }
    // retry queue: standalone addr-keyed entries.
    const int RN = s.max_txns * s.max_watch_set;
    for (int i = 0; i < RN; ++i)
        if (s.rq_used[i] && s.rq_etxn[i] == txn_id) s.rq_used[i] = 0;
}

// Count retry entries for an addr.
__device__ int rd_retryq_count(const StmRefDev& s, uint64_t addr) {
    const int RN = s.max_txns * s.max_watch_set; int c = 0;
    for (int i = 0; i < RN; ++i) if (s.rq_used[i] && s.rq_eaddr[i] == addr) ++c;
    return c;
}
// Find retry entry slot for addr at queue position pos (seq asc, ins asc).
__device__ int rd_retryq_at(const StmRefDev& s, uint64_t addr, int pos) {
    const int RN = s.max_txns * s.max_watch_set;
    int found = 0; bool have = false; uint64_t cs = 0, ci = 0;
    for (;;) {
        int sel = -1; uint64_t bs = 0, bi = 0;
        for (int i = 0; i < RN; ++i) {
            if (!s.rq_used[i] || s.rq_eaddr[i] != addr) continue;
            uint64_t sq = s.rq_eseq[i], ins = s.rq_eins[i];
            if (have && (sq < cs || (sq == cs && ins <= ci))) continue;
            if (sel < 0 || sq < bs || (sq == bs && ins < bi)) { sel = i; bs = sq; bi = ins; }
        }
        if (sel < 0) return -1;
        if (found == pos) return sel;
        ++found; have = true; cs = bs; ci = bi;
    }
}

// forward declaration (defined after rd_wake_retry_watchers)
__device__ void rd_clear_txn_slot_sets(const StmRefDev& s, int tslot);

__device__ void rd_emit(const StmRefDev& s, uint64_t* eh, uint8_t kind,
        uint32_t op_idx, uint64_t txn_id, uint64_t addr, int64_t value,
        uint64_t version, uint8_t reason, uint64_t aux) {
    uint64_t seq = s.counters[C_ES];
    rf_u8(eh, kind);
    rf_u64(eh, seq);
    rf_u32(eh, op_idx);
    rf_u64(eh, txn_id);
    rf_u64(eh, addr);
    rf_i64(eh, value);
    rf_u64(eh, version);
    rf_u8(eh, reason);
    rf_u64(eh, aux);
    s.counters[C_ES] = seq + 1;
}

__device__ void rd_emit_read_result(uint64_t* rh, uint64_t read_id, uint64_t txn_id,
        uint64_t addr, uint8_t kind, int64_t value, uint64_t version, uint64_t attempt) {
    rf_u64(rh, read_id);
    rf_u64(rh, txn_id);
    rf_u64(rh, addr);
    rf_u8(rh, kind);
    rf_i64(rh, value);
    rf_u64(rh, version);
    rf_u64(rh, attempt);
}

__device__ void rd_emit_invalid(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id) {
    rd_emit(s, eh, STM_EVT_INVALID, op_idx, txn_id, STM_U64_MAX, STM_I64_MIN,
            STM_U64_MAX, STM_RS_NONE, 0);
    c->v[K_INV] += 1;
}

// Wake at most one lock waiter for location-slot li (addr). Drops stale silently.
__device__ bool rd_wake_one_lock_waiter(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, int li) {
    if (li < 0) return false;
    const int lb = li * s.max_waiters;
    while (s.lq_count[li] > 0) {
        uint64_t cand = s.lq_txn[lb + 0];
        int ts = rd_find_txn(s, cand);
        bool valid = (ts >= 0) && (s.txn_status[ts] == STM_ST_WAITING_LOCK) &&
                     (s.txn_wait_addr[ts] == s.loc_addr[li]);
        // pop front
        int cnt = s.lq_count[li];
        for (int p = 1; p < cnt; ++p) {
            s.lq_txn[lb + p - 1] = s.lq_txn[lb + p];
            s.lq_seq[lb + p - 1] = s.lq_seq[lb + p];
        }
        s.lq_count[li] = cnt - 1;
        if (!valid) continue;  // stale: silent
        s.txn_status[ts] = STM_ST_ACTIVE;
        s.txn_wait_addr[ts] = STM_U64_MAX;
        s.txn_wait_seq[ts] = STM_U64_MAX;
        rd_emit(s, eh, STM_EVT_TXN_WAKE_LOCK, op_idx, cand, s.loc_addr[li],
                STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, 0);
        c->v[K_WKL] += 1;
        return true;
    }
    return false;
}

// Wake all retry watchers for an addr in queue order (seq asc, ins asc).
__device__ void rd_wake_retry_watchers(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t addr) {
    int cnt = rd_retryq_count(s, addr);
    // Snapshot candidate txn ids in queue order (waking removes from the queue).
    uint64_t snap[STM_MAX_WATCH_SET];
    int nsnap = 0;
    for (int p = 0; p < cnt; ++p) {
        int qi = rd_retryq_at(s, addr, p);
        if (qi < 0) break;
        snap[nsnap++] = s.rq_etxn[qi];
    }
    for (int p = 0; p < nsnap; ++p) {
        uint64_t cand = snap[p];
        int ts = rd_find_txn(s, cand);
        if (ts < 0) continue;                              // absent
        if (s.txn_status[ts] != STM_ST_SUSPENDED_RETRY) continue;  // active/removed
        rd_remove_from_queues(s, cand);
        rd_clear_txn_slot_sets(s, ts);
        s.txn_attempt_no[ts] += 1;
        s.txn_start_version[ts] = s.counters[C_GV];
        s.txn_status[ts] = STM_ST_ACTIVE;
        rd_emit(s, eh, STM_EVT_TXN_WAKE_RETRY, op_idx, cand, addr,
                STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, s.txn_attempt_no[ts]);
        c->v[K_WR] += 1;
    }
}

// (forward) clear only the read+write sets, keep txn slot used.
__device__ void rd_clear_txn_slot_sets(const StmRefDev& s, int tslot) {
    const int rb = tslot * s.max_read_set;
    for (int j = 0; j < s.max_read_set; ++j) s.rs_used[rb + j] = 0;
    const int wb = tslot * s.max_write_set;
    for (int j = 0; j < s.max_write_set; ++j) s.ws_used[wb + j] = 0;
}

// Validate read set for txn slot. Returns true if OK; else *fail = addr.
__device__ bool rd_validate(const StmRefDev& s, int tslot, uint64_t self_txn, uint64_t* fail) {
    // iterate read entries ascending addr by selection.
    bool have = false; uint64_t cur = 0;
    for (;;) {
        // find smallest read-set addr > cur (or any if !have)
        int sel = -1; uint64_t best = 0;
        const int rb = tslot * s.max_read_set;
        for (int j = 0; j < s.max_read_set; ++j) {
            if (!s.rs_used[rb + j]) continue;
            uint64_t a = s.rs_addr[rb + j];
            if (have && a <= cur) continue;
            if (sel < 0 || a < best) { sel = rb + j; best = a; }
        }
        if (sel < 0) break;
        have = true; cur = best;
        if (s.rs_from_own_write[sel]) continue;
        if (rd_find_ws(s, tslot, best) >= 0) continue;
        int li = rd_find_loc(s, best);
        uint64_t lver = (li >= 0) ? s.loc_version[li] : 0;
        uint64_t lown = (li >= 0) ? s.loc_lock_owner[li] : 0;
        if (lown != 0 && lown != self_txn) { *fail = best; return false; }
        if (lver != s.rs_read_version[sel]) { *fail = best; return false; }
    }
    return true;
}

// Release all locks held by txn ascending addr (abort path): emit
// WRITE_UNLOCK_ABORT + wake-lock per location.
__device__ void rd_release_all_locks_abort(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id) {
    bool have = false; uint64_t cur = 0;
    for (;;) {
        int sel = -1; uint64_t best = 0;
        for (int i = 0; i < s.max_locations; ++i) {
            if (!s.loc_used[i]) break;
            if (s.loc_lock_owner[i] != txn_id) continue;
            uint64_t a = s.loc_addr[i];
            if (have && a <= cur) continue;
            if (sel < 0 || a < best) { sel = i; best = a; }
        }
        if (sel < 0) break;
        have = true; cur = best;
        s.loc_lock_owner[sel] = 0;
        rd_emit(s, eh, STM_EVT_WRITE_UNLOCK_ABORT, op_idx, txn_id, best, STM_I64_MIN,
                STM_U64_MAX, STM_RS_NONE, 0);
        c->v[K_AU] += 1;
        rd_wake_one_lock_waiter(s, eh, c, op_idx, sel);
    }
}

// Abort txn: release locks (+wakes), remove from queues, emit TXN_ABORT(reason).
__device__ void rd_abort_txn(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id, int tslot, uint8_t reason, uint64_t fail_addr) {
    rd_release_all_locks_abort(s, eh, c, op_idx, txn_id);
    rd_remove_from_queues(s, txn_id);
    rd_emit(s, eh, STM_EVT_TXN_ABORT, op_idx, txn_id, fail_addr, STM_I64_MIN,
            STM_U64_MAX, reason, 0);
    c->v[K_ABT] += 1;
    rd_clear_txn_slot(s, tslot);
}

// ---- op handlers ----
__device__ void rd_begin(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id, uint64_t priority) {
    if (rd_find_txn(s, txn_id) >= 0 || rd_txn_count(s) >= s.max_txns) {
        rd_emit_invalid(s, eh, c, op_idx, txn_id);
        return;
    }
    int slot = -1;
    for (int i = 0; i < s.max_txns; ++i) if (!s.txn_used[i]) { slot = i; break; }
    uint64_t xid = s.counters[C_XID]; s.counters[C_XID] = xid + 1;
    uint64_t bs = s.counters[C_BS]; s.counters[C_BS] = bs + 1;
    uint64_t gv = s.counters[C_GV];
    s.txn_used[slot] = 1;
    s.txn_id[slot] = txn_id;
    s.txn_xid[slot] = xid;
    s.txn_begin_seq[slot] = bs;
    s.txn_attempt_no[slot] = 0;
    s.txn_priority[slot] = (uint32_t)priority;
    s.txn_start_version[slot] = gv;
    s.txn_status[slot] = STM_ST_ACTIVE;
    s.txn_wait_addr[slot] = STM_U64_MAX;
    s.txn_wait_seq[slot] = STM_U64_MAX;
    s.txn_prepare_seq[slot] = STM_U64_MAX;
    rd_clear_txn_slot_sets(s, slot);
    rd_emit(s, eh, STM_EVT_TXN_BEGIN, op_idx, txn_id, STM_U64_MAX, STM_I64_MIN,
            gv, STM_RS_NONE, (uint64_t)(uint32_t)priority);
    c->v[K_BEGUN] += 1;
}

__device__ void rd_tx_read(const StmRefDev& s, uint64_t* eh, uint64_t* rh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id, int tslot, uint64_t read_id, uint64_t addr) {
    uint64_t attempt = s.txn_attempt_no[tslot];
    int wj = rd_find_ws(s, tslot, addr);
    if (wj >= 0) {
        int64_t v = s.ws_value[wj];
        // set/update read entry from_own_write=1
        int rj = rd_find_rs(s, tslot, addr);
        if (rj < 0) rj = rd_alloc_rs(s, tslot);
        s.rs_used[rj] = 1; s.rs_addr[rj] = addr; s.rs_from_own_write[rj] = 1;
        s.rs_read_version[rj] = STM_U64_MAX; s.rs_read_value[rj] = v;
        rd_emit(s, eh, STM_EVT_READ_OWN_WRITE, op_idx, txn_id, addr, v, STM_U64_MAX,
                STM_RS_NONE, read_id);
        rd_emit_read_result(rh, read_id, txn_id, addr, STM_READ_RES_OWN_WRITE, v,
                            STM_U64_MAX, attempt);
        c->v[K_ROW] += 1;
        return;
    }
    int li = rd_find_loc(s, addr);
    int64_t lval = (li >= 0) ? s.loc_value[li] : 0;
    uint64_t lver = (li >= 0) ? s.loc_version[li] : 0;
    uint64_t lown = (li >= 0) ? s.loc_lock_owner[li] : 0;
    if (lown != 0 && lown != txn_id) {
        rd_emit_read_result(rh, read_id, txn_id, addr, STM_READ_RES_ABORT_LOCK,
                            STM_I64_MIN, STM_U64_MAX, attempt);
        rd_abort_txn(s, eh, c, op_idx, txn_id, tslot, STM_RS_READ_LOCK_CONFLICT, addr);
        return;
    }
    if (lver > s.txn_start_version[tslot]) {
        rd_emit_read_result(rh, read_id, txn_id, addr, STM_READ_RES_ABORT_VER,
                            STM_I64_MIN, STM_U64_MAX, attempt);
        rd_abort_txn(s, eh, c, op_idx, txn_id, tslot, STM_RS_READ_VERSION_CONFLICT, addr);
        return;
    }
    int rj = rd_find_rs(s, tslot, addr);
    if (rj < 0) rj = rd_alloc_rs(s, tslot);
    s.rs_used[rj] = 1; s.rs_addr[rj] = addr; s.rs_from_own_write[rj] = 0;
    s.rs_read_version[rj] = lver; s.rs_read_value[rj] = lval;
    rd_emit(s, eh, STM_EVT_READ_SHARED, op_idx, txn_id, addr, lval, lver,
            STM_RS_NONE, read_id);
    rd_emit_read_result(rh, read_id, txn_id, addr, STM_READ_RES_SHARED, lval,
                        lver, attempt);
    c->v[K_RSH] += 1;
}

__device__ void rd_tx_write(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id, int tslot, uint64_t addr, int64_t value) {
    int wj = rd_find_ws(s, tslot, addr);
    uint64_t wseq;
    if (wj >= 0) {
        s.ws_value[wj] = value;
        wseq = s.ws_write_set_seq[wj];
    } else {
        if (rd_ws_count(s, tslot) >= s.max_write_set) {
            rd_emit_invalid(s, eh, c, op_idx, txn_id);
            return;
        }
        wj = rd_alloc_ws(s, tslot);
        s.ws_used[wj] = 1; s.ws_addr[wj] = addr; s.ws_value[wj] = value;
        wseq = s.counters[C_WS]; s.counters[C_WS] = wseq + 1;
        s.ws_write_set_seq[wj] = wseq;
    }
    rd_emit(s, eh, STM_EVT_WRITE_STAGE, op_idx, txn_id, addr, value, STM_U64_MAX,
            STM_RS_NONE, wseq);
    c->v[K_WST] += 1;
}

__device__ void rd_validate_op(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id, int tslot) {
    uint64_t fail = STM_U64_MAX;
    if (!rd_validate(s, tslot, txn_id, &fail)) {
        rd_abort_txn(s, eh, c, op_idx, txn_id, tslot, STM_RS_VALIDATE_FAIL, fail);
        return;
    }
    rd_emit(s, eh, STM_EVT_VALIDATE_OK, op_idx, txn_id, STM_U64_MAX, STM_I64_MIN,
            STM_U64_MAX, STM_RS_NONE, 0);
    c->v[K_VOK] += 1;
}

// Release locks acquired this prepare (descending addr) + wakes.
__device__ void rd_release_partial(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id, uint64_t* acq, int nacq) {
    // sort ascending (insertion, nacq small <= max_write_set)
    for (int i = 1; i < nacq; ++i) {
        uint64_t key = acq[i]; int j = i - 1;
        while (j >= 0 && acq[j] > key) { acq[j + 1] = acq[j]; --j; }
        acq[j + 1] = key;
    }
    for (int k = nacq - 1; k >= 0; --k) {
        uint64_t addr = acq[k];
        int li = rd_find_loc(s, addr);
        if (li >= 0 && s.loc_lock_owner[li] == txn_id) s.loc_lock_owner[li] = 0;
        rd_emit(s, eh, STM_EVT_WRITE_UNLOCK_PARTIAL, op_idx, txn_id, addr, STM_I64_MIN,
                STM_U64_MAX, STM_RS_NONE, 0);
        c->v[K_PU] += 1;
        rd_wake_one_lock_waiter(s, eh, c, op_idx, li);
    }
}

__device__ void rd_try_prepare(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id, int tslot) {
    int wcount = rd_ws_count(s, tslot);
    if (wcount == 0) {
        rd_emit(s, eh, STM_EVT_COMMIT_READONLY, op_idx, txn_id, STM_U64_MAX,
                STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, 0);
        c->v[K_ROC] += 1;
        rd_remove_from_queues(s, txn_id);
        rd_clear_txn_slot(s, tslot);
        return;
    }

    uint64_t acq[STM_MAX_WRITE_SET];
    int nacq = 0;

    // Lock write-set addrs ascending order (selection).
    bool have = false; uint64_t cur = 0;
    for (;;) {
        int sel = -1; uint64_t best = 0;
        const int wb = tslot * s.max_write_set;
        for (int j = 0; j < s.max_write_set; ++j) {
            if (!s.ws_used[wb + j]) continue;
            uint64_t a = s.ws_addr[wb + j];
            if (have && a <= cur) continue;
            if (sel < 0 || a < best) { sel = wb + j; best = a; }
        }
        if (sel < 0) break;
        have = true; cur = best;
        uint64_t addr = best;

        int li = rd_find_loc(s, addr);
        if (li < 0) {
            // absent: can lock only if room to materialize.
            if (rd_loc_count(s) >= s.max_locations) {
                rd_release_partial(s, eh, c, op_idx, txn_id, acq, nacq);
                rd_abort_txn(s, eh, c, op_idx, txn_id, tslot, STM_RS_PREPARE_OOM, addr);
                return;
            }
            li = rd_alloc_loc(s, addr);
            s.loc_lock_owner[li] = txn_id;
            acq[nacq++] = addr;
            rd_emit(s, eh, STM_EVT_WRITE_LOCK, op_idx, txn_id, addr, STM_I64_MIN,
                    s.loc_version[li], STM_RS_NONE, 0);
            c->v[K_WL] += 1;
            continue;
        }
        uint64_t own = s.loc_lock_owner[li];
        if (own == 0 || own == txn_id) {
            s.loc_lock_owner[li] = txn_id;
            acq[nacq++] = addr;
            rd_emit(s, eh, STM_EVT_WRITE_LOCK, op_idx, txn_id, addr, STM_I64_MIN,
                    s.loc_version[li], STM_RS_NONE, 0);
            c->v[K_WL] += 1;
            continue;
        }
        // locked by another -> wait.
        if (s.lq_count[li] >= s.max_waiters) {
            rd_release_partial(s, eh, c, op_idx, txn_id, acq, nacq);
            rd_abort_txn(s, eh, c, op_idx, txn_id, tslot, STM_RS_LOCK_WAIT_OVERFLOW, addr);
            return;
        }
        uint64_t wseq = s.counters[C_WAIT]; s.counters[C_WAIT] = wseq + 1;
        s.txn_status[tslot] = STM_ST_WAITING_LOCK;
        s.txn_wait_addr[tslot] = addr;
        s.txn_wait_seq[tslot] = wseq;
        const int lb = li * s.max_waiters;
        int pos = s.lq_count[li];
        s.lq_txn[lb + pos] = txn_id; s.lq_seq[lb + pos] = wseq;
        s.lq_count[li] = pos + 1;
        rd_emit(s, eh, STM_EVT_TXN_WAIT_LOCK, op_idx, txn_id, addr, STM_I64_MIN,
                STM_U64_MAX, STM_RS_NONE, wseq);
        c->v[K_WTL] += 1;
        rd_release_partial(s, eh, c, op_idx, txn_id, acq, nacq);
        return;
    }

    // all locks held: validate.
    uint64_t fail = STM_U64_MAX;
    if (!rd_validate(s, tslot, txn_id, &fail)) {
        rd_abort_txn(s, eh, c, op_idx, txn_id, tslot, STM_RS_PREPARE_VALIDATE_FAIL, fail);
        return;
    }
    s.txn_status[tslot] = STM_ST_PREPARED;
    s.txn_prepare_seq[tslot] = s.counters[C_ES];  // seq TXN_PREPARED consumes
    rd_emit(s, eh, STM_EVT_TXN_PREPARED, op_idx, txn_id, STM_U64_MAX, STM_I64_MIN,
            STM_U64_MAX, STM_RS_NONE, s.txn_prepare_seq[tslot]);
    c->v[K_PREP] += 1;
}

__device__ int rd_pick_prepared(const StmRefDev& s) {
    int best = -1; uint32_t bpri = 0; uint64_t bpseq = 0, bid = 0;
    for (int i = 0; i < s.max_txns; ++i) {
        if (!s.txn_used[i] || s.txn_status[i] != STM_ST_PREPARED) continue;
        uint32_t pri = s.txn_priority[i];
        uint64_t pseq = s.txn_prepare_seq[i];
        uint64_t id = s.txn_id[i];
        if (best < 0 || pri > bpri || (pri == bpri && pseq < bpseq) ||
            (pri == bpri && pseq == bpseq && id < bid)) {
            best = i; bpri = pri; bpseq = pseq; bid = id;
        }
    }
    return best;
}

__device__ void rd_drain(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t limit) {
    if (limit == 0) return;
    uint64_t budget = limit;
    while (budget > 0) {
        int ti = rd_pick_prepared(s);
        if (ti < 0) break;
        uint64_t cid = s.txn_id[ti];
        uint64_t commit_version = s.counters[C_GV] + 1; s.counters[C_GV] = commit_version;

        // Apply writes ascending addr; collect written addrs (ascending).
        uint64_t written[STM_MAX_WRITE_SET];
        int nwritten = 0;
        bool have = false; uint64_t cur = 0;
        for (;;) {
            int sel = -1; uint64_t best = 0;
            const int wb = ti * s.max_write_set;
            for (int j = 0; j < s.max_write_set; ++j) {
                if (!s.ws_used[wb + j]) continue;
                uint64_t a = s.ws_addr[wb + j];
                if (have && a <= cur) continue;
                if (sel < 0 || a < best) { sel = wb + j; best = a; }
            }
            if (sel < 0) break;
            have = true; cur = best;
            int64_t val = s.ws_value[sel];
            int li = rd_get_or_create_loc(s, best);  // materialize if needed
            s.loc_value[li] = val;
            s.loc_version[li] = commit_version;
            s.loc_write_seq[li] = s.counters[C_ES];  // seq LOCATION_WRITE consumes
            rd_emit(s, eh, STM_EVT_LOCATION_WRITE, op_idx, cid, best, val,
                    commit_version, STM_RS_NONE, s.loc_write_seq[li]);
            c->v[K_LW] += 1;
            written[nwritten++] = best;
        }

        // Collect locks held by cid ascending (these are the write-set addrs).
        uint64_t held[STM_MAX_WRITE_SET];
        int nheld = 0;
        have = false; cur = 0;
        for (;;) {
            int sel = -1; uint64_t best = 0;
            for (int i = 0; i < s.max_locations; ++i) {
                if (!s.loc_used[i]) break;
                if (s.loc_lock_owner[i] != cid) continue;
                uint64_t a = s.loc_addr[i];
                if (have && a <= cur) continue;
                if (sel < 0 || a < best) { sel = i; best = a; }
            }
            if (sel < 0) break;
            have = true; cur = best;
            held[nheld++] = best;
        }

        // Remove txn first.
        rd_remove_from_queues(s, cid);
        rd_clear_txn_slot(s, ti);

        // Release locks ascending addr (WRITE_UNLOCK_COMMIT).
        for (int k = 0; k < nheld; ++k) {
            int li = rd_find_loc(s, held[k]);
            if (li >= 0) s.loc_lock_owner[li] = 0;
            rd_emit(s, eh, STM_EVT_WRITE_UNLOCK_COMMIT, op_idx, cid, held[k],
                    STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, 0);
            c->v[K_CU] += 1;
        }

        // Wake retry watchers for each written addr ascending.
        for (int k = 0; k < nwritten; ++k) {
            rd_wake_retry_watchers(s, eh, c, op_idx, written[k]);
        }
        // Wake at most one lock waiter per unlocked addr ascending.
        for (int k = 0; k < nheld; ++k) {
            int li = rd_find_loc(s, held[k]);
            rd_wake_one_lock_waiter(s, eh, c, op_idx, li);
        }

        rd_emit(s, eh, STM_EVT_COMMIT_DONE, op_idx, cid, STM_U64_MAX, STM_I64_MIN,
                commit_version, STM_RS_NONE, 0);
        c->v[K_CD] += 1;
        budget -= 1;
    }
}

__device__ void rd_retry(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id, int tslot, uint64_t watch_count,
        const uint64_t* watch_addrs) {
    if (watch_count == 0 || watch_count > (uint64_t)s.max_watch_set) {
        rd_emit_invalid(s, eh, c, op_idx, txn_id);
        return;
    }
    // dedup ascending into local buffer.
    uint64_t ws[STM_MAX_WATCH_SET];
    int n = 0;
    // copy
    for (uint64_t i = 0; i < watch_count; ++i) ws[i] = watch_addrs[i];
    int wc = (int)watch_count;
    // insertion sort
    for (int i = 1; i < wc; ++i) {
        uint64_t key = ws[i]; int j = i - 1;
        while (j >= 0 && ws[j] > key) { ws[j + 1] = ws[j]; --j; }
        ws[j + 1] = key;
    }
    // unique
    for (int i = 0; i < wc; ++i) {
        if (n == 0 || ws[n - 1] != ws[i]) ws[n++] = ws[i];
    }
    if ((uint64_t)n > (uint64_t)s.max_watch_set) {
        rd_emit_invalid(s, eh, c, op_idx, txn_id);
        return;
    }

    uint64_t fail = STM_U64_MAX;
    if (!rd_validate(s, tslot, txn_id, &fail)) {
        rd_clear_txn_slot_sets(s, tslot);
        s.txn_attempt_no[tslot] += 1;
        s.txn_start_version[tslot] = s.counters[C_GV];
        rd_emit(s, eh, STM_EVT_RETRY_IMMEDIATE, op_idx, txn_id, STM_U64_MAX,
                STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, s.txn_attempt_no[tslot]);
        c->v[K_RIM] += 1;
        return;
    }

    // Append to each watched addr's retry queue (addr-keyed, independent of the
    // location table; watching an absent address does NOT materialize it).
    // Single retry_seq for all; capacity is per-addr max_retry_watchers.
    uint64_t rseq = s.counters[C_RTY];  // tentative
    int added_slots[STM_MAX_WATCH_SET];
    int nadded = 0;
    const int RN = s.max_txns * s.max_watch_set;
    bool overflow = false; uint64_t ovf_addr = STM_U64_MAX;
    for (int i = 0; i < n; ++i) {
        uint64_t addr = ws[i];
        if (rd_retryq_count(s, addr) >= s.max_retry_watchers) { overflow = true; ovf_addr = addr; break; }
        int qslot = -1;
        for (int q = 0; q < RN; ++q) if (!s.rq_used[q]) { qslot = q; break; }
        uint64_t ins = s.qins[0]; s.qins[0] = ins + 1;
        s.rq_used[qslot] = 1; s.rq_eaddr[qslot] = addr; s.rq_etxn[qslot] = txn_id;
        s.rq_eseq[qslot] = rseq; s.rq_eins[qslot] = ins;
        added_slots[nadded++] = qslot;
    }
    if (overflow) {
        for (int k = nadded - 1; k >= 0; --k) s.rq_used[added_slots[k]] = 0;
        rd_emit(s, eh, STM_EVT_RETRY_WATCH_OVERFLOW, op_idx, txn_id, ovf_addr,
                STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, 0);
        c->v[K_RWO] += 1;
        return;
    }
    s.counters[C_RTY] = rseq + 1;
    // clear write set, keep read set.
    const int wb = tslot * s.max_write_set;
    for (int j = 0; j < s.max_write_set; ++j) s.ws_used[wb + j] = 0;
    s.txn_status[tslot] = STM_ST_SUSPENDED_RETRY;
    rd_emit(s, eh, STM_EVT_TXN_SUSPEND_RETRY, op_idx, txn_id, STM_U64_MAX,
            STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, rseq);
    c->v[K_RSU] += 1;
}

__device__ void rd_non_tx_write(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t addr, int64_t value) {
    int li = rd_find_loc(s, addr);
    if (li >= 0 && s.loc_lock_owner[li] != 0) {
        rd_emit(s, eh, STM_EVT_NON_TX_STALL_LOCKED, op_idx, 0, addr, value,
                STM_U64_MAX, STM_RS_NONE, s.loc_lock_owner[li]);
        c->v[K_NTS] += 1;
        return;
    }
    if (li < 0 && rd_loc_count(s) >= s.max_locations) {
        rd_emit(s, eh, STM_EVT_NON_TX_OOM, op_idx, 0, addr, value, STM_U64_MAX,
                STM_RS_NONE, 0);
        return;
    }
    if (li < 0) li = rd_alloc_loc(s, addr);
    uint64_t nv = s.counters[C_GV] + 1; s.counters[C_GV] = nv;
    s.loc_version[li] = nv;
    s.loc_value[li] = value;
    s.loc_write_seq[li] = s.counters[C_ES];  // seq NON_TX_WRITE consumes
    rd_emit(s, eh, STM_EVT_NON_TX_WRITE, op_idx, 0, addr, value, nv, STM_RS_NONE, 0);
    c->v[K_NTW] += 1;
    rd_wake_retry_watchers(s, eh, c, op_idx, addr);
}

__device__ void rd_abort_op(const StmRefDev& s, uint64_t* eh, RefCounts* c,
        uint32_t op_idx, uint64_t txn_id, int tslot) {
    rd_abort_txn(s, eh, c, op_idx, txn_id, tslot, STM_RS_EXPLICIT_ABORT, STM_U64_MAX);
}

// ---- state hashes ----
__device__ uint64_t rd_location_hash(const StmRefDev& s) {
    uint64_t h = STM_FNV_OFFSET;
    int idx[STM_MAX_LOCATIONS];
    uint64_t key[STM_MAX_LOCATIONS];
    int n = 0;
    for (int i = 0; i < s.max_locations; ++i) {
        if (!s.loc_used[i]) break;
        idx[n] = i; key[n] = s.loc_addr[i]; ++n;
    }
    for (int a = 1; a < n; ++a) {
        int ki = idx[a]; uint64_t kk = key[a]; int b = a - 1;
        while (b >= 0 && key[b] > kk) { key[b+1]=key[b]; idx[b+1]=idx[b]; --b; }
        key[b+1]=kk; idx[b+1]=ki;
    }
    for (int k = 0; k < n; ++k) {
        int sel = idx[k];
        rf_u64(&h, s.loc_addr[sel]);
        rf_i64(&h, s.loc_value[sel]);
        rf_u64(&h, s.loc_version[sel]);
        rf_u64(&h, s.loc_lock_owner[sel]);
        rf_u64(&h, s.loc_write_seq[sel]);
    }
    return h;
}

__device__ uint64_t rd_txn_hash(const StmRefDev& s) {
    uint64_t h = STM_FNV_OFFSET;
    // txns ascending txn_id (unique): compact used slots + insertion sort.
    int tidx[STM_MAX_TXNS]; uint64_t tkey[STM_MAX_TXNS]; int nt = 0;
    for (int i = 0; i < s.max_txns; ++i)
        if (s.txn_used[i]) { tidx[nt] = i; tkey[nt] = s.txn_id[i]; ++nt; }
    for (int a = 1; a < nt; ++a) {
        int ki = tidx[a]; uint64_t kk = tkey[a]; int b = a - 1;
        while (b >= 0 && tkey[b] > kk) { tkey[b+1]=tkey[b]; tidx[b+1]=tidx[b]; --b; }
        tkey[b+1]=kk; tidx[b+1]=ki;
    }
    for (int t = 0; t < nt; ++t) {
        int sel = tidx[t];
        rf_u64(&h, s.txn_id[sel]);
        rf_u64(&h, s.txn_xid[sel]);
        rf_u64(&h, s.txn_begin_seq[sel]);
        rf_u64(&h, s.txn_attempt_no[sel]);
        rf_u32(&h, s.txn_priority[sel]);
        rf_u64(&h, s.txn_start_version[sel]);
        rf_u8(&h, s.txn_status[sel]);
        rf_u64(&h, s.txn_wait_addr[sel]);
        rf_u64(&h, s.txn_prepare_seq[sel]);
        // read set ascending addr (unique): compact + sort.
        int ridx[STM_MAX_READ_SET]; uint64_t rkey[STM_MAX_READ_SET]; int nr = 0;
        const int rb = sel * s.max_read_set;
        for (int j = 0; j < s.max_read_set; ++j)
            if (s.rs_used[rb + j]) { ridx[nr] = rb + j; rkey[nr] = s.rs_addr[rb + j]; ++nr; }
        for (int a = 1; a < nr; ++a) {
            int ki = ridx[a]; uint64_t kk = rkey[a]; int b = a - 1;
            while (b >= 0 && rkey[b] > kk) { rkey[b+1]=rkey[b]; ridx[b+1]=ridx[b]; --b; }
            rkey[b+1]=kk; ridx[b+1]=ki;
        }
        for (int k = 0; k < nr; ++k) {
            int rs = ridx[k];
            rf_u64(&h, s.rs_addr[rs]);
            rf_u64(&h, s.rs_read_version[rs]);
            rf_i64(&h, s.rs_read_value[rs]);
            rf_u8(&h, s.rs_from_own_write[rs]);
        }
        // write set ascending addr (unique): compact + sort.
        int widx[STM_MAX_WRITE_SET]; uint64_t wkey[STM_MAX_WRITE_SET]; int nw = 0;
        const int wb = sel * s.max_write_set;
        for (int j = 0; j < s.max_write_set; ++j)
            if (s.ws_used[wb + j]) { widx[nw] = wb + j; wkey[nw] = s.ws_addr[wb + j]; ++nw; }
        for (int a = 1; a < nw; ++a) {
            int ki = widx[a]; uint64_t kk = wkey[a]; int b = a - 1;
            while (b >= 0 && wkey[b] > kk) { wkey[b+1]=wkey[b]; widx[b+1]=widx[b]; --b; }
            wkey[b+1]=kk; widx[b+1]=ki;
        }
        for (int k = 0; k < nw; ++k) {
            int ws = widx[k];
            rf_u64(&h, s.ws_addr[ws]);
            rf_i64(&h, s.ws_value[ws]);
            rf_u64(&h, s.ws_write_set_seq[ws]);
        }
    }
    return h;
}

__device__ uint64_t rd_queue_hash(const StmRefDev& s) {
    uint64_t h = STM_FNV_OFFSET;
    // lock wait queues by addr ascending (unique), then stored queue order.
    int lidx[STM_MAX_LOCATIONS]; uint64_t lkey[STM_MAX_LOCATIONS]; int nl = 0;
    for (int i = 0; i < s.max_locations; ++i) {
        if (!s.loc_used[i]) break;
        if (s.lq_count[i] > 0) { lidx[nl] = i; lkey[nl] = s.loc_addr[i]; ++nl; }
    }
    for (int a = 1; a < nl; ++a) {
        int ki = lidx[a]; uint64_t kk = lkey[a]; int b = a - 1;
        while (b >= 0 && lkey[b] > kk) { lkey[b+1]=lkey[b]; lidx[b+1]=lidx[b]; --b; }
        lkey[b+1]=kk; lidx[b+1]=ki;
    }
    for (int t = 0; t < nl; ++t) {
        int sel = lidx[t];
        const int lb = sel * s.max_waiters;
        for (int p = 0; p < s.lq_count[sel]; ++p) {
            rf_u8(&h, (uint8_t)STM_QK_LOCK_WAIT);
            rf_u64(&h, s.loc_addr[sel]);
            rf_u64(&h, (uint64_t)p);
            rf_u64(&h, s.lq_txn[lb + p]);
            rf_u64(&h, s.lq_seq[lb + p]);
        }
    }
    // retry watch queues by addr ascending, then (seq asc, ins asc). Addr-keyed;
    // per-addr count is bounded by max_retry_watchers. Find next distinct addr by
    // scan, then compact that addr's entries and sort by (seq, ins).
    const int RN = s.max_txns * s.max_watch_set;
    bool have = false; uint64_t cur = 0;
    for (;;) {
        int selA = -1; uint64_t best = 0;
        for (int i = 0; i < RN; ++i) {
            if (!s.rq_used[i]) continue;
            uint64_t a = s.rq_eaddr[i];
            if (have && a <= cur) continue;
            if (selA < 0 || a < best) { selA = i; best = a; }
        }
        if (selA < 0) break;
        have = true; cur = best;
        int qarr[STM_MAX_RETRY_WATCHERS];
        uint64_t sq[STM_MAX_RETRY_WATCHERS], ins[STM_MAX_RETRY_WATCHERS];
        int cnt = 0;
        for (int i = 0; i < RN; ++i)
            if (s.rq_used[i] && s.rq_eaddr[i] == best) {
                qarr[cnt] = i; sq[cnt] = s.rq_eseq[i]; ins[cnt] = s.rq_eins[i]; ++cnt;
            }
        for (int a = 1; a < cnt; ++a) {
            int ki = qarr[a]; uint64_t ks = sq[a], kn = ins[a]; int b = a - 1;
            while (b >= 0 && (sq[b] > ks || (sq[b] == ks && ins[b] > kn))) {
                sq[b+1]=sq[b]; ins[b+1]=ins[b]; qarr[b+1]=qarr[b]; --b;
            }
            sq[b+1]=ks; ins[b+1]=kn; qarr[b+1]=ki;
        }
        for (int p = 0; p < cnt; ++p) {
            int qi = qarr[p];
            rf_u8(&h, (uint8_t)STM_QK_RETRY_WATCH);
            rf_u64(&h, best);
            rf_u64(&h, (uint64_t)p);
            rf_u64(&h, s.rq_etxn[qi]);
            rf_u64(&h, s.rq_eseq[qi]);
        }
    }
    return h;
}

__global__ void stm_ref_step_kernel(
        StmRefDev s, int batch_size,
        const int32_t* __restrict__ op_kind,
        const uint64_t* __restrict__ in_txn,
        const uint64_t* __restrict__ in_read_id,
        const uint64_t* __restrict__ in_addr,
        const int64_t* __restrict__ in_value,
        const uint64_t* __restrict__ in_aux,
        const uint64_t* __restrict__ in_woff,
        const uint64_t* __restrict__ in_watch,
        int32_t* __restrict__ out_counts,
        uint64_t* __restrict__ out_evt,
        uint64_t* __restrict__ out_read,
        uint64_t* __restrict__ out_loc,
        uint64_t* __restrict__ out_txn,
        uint64_t* __restrict__ out_queue) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    RefCounts c; for (int i = 0; i < 23; ++i) c.v[i] = 0;
    uint64_t eh = STM_FNV_OFFSET;
    uint64_t rh = STM_FNV_OFFSET;

    for (int r = 0; r < batch_size; ++r) {
        const uint32_t op_idx = (uint32_t)s.op_index[0];
        const int32_t kind = op_kind[r];
        const uint64_t txn_id = in_txn[r];
        const uint64_t read_id = in_read_id[r];
        const uint64_t addr = in_addr[r];
        const int64_t value = in_value[r];
        const uint64_t aux = in_aux[r];
        const uint64_t woff = in_woff[r];

        int tslot = rd_find_txn(s, txn_id);
        bool active = (tslot >= 0) && (s.txn_status[tslot] == STM_ST_ACTIVE);

        switch (kind) {
            case STM_OP_BEGIN:
                rd_begin(s, &eh, &c, op_idx, txn_id, aux);
                break;
            case STM_OP_TX_READ:
                if (!active) rd_emit_invalid(s, &eh, &c, op_idx, txn_id);
                else rd_tx_read(s, &eh, &rh, &c, op_idx, txn_id, tslot, read_id, addr);
                break;
            case STM_OP_TX_WRITE:
                if (!active) rd_emit_invalid(s, &eh, &c, op_idx, txn_id);
                else rd_tx_write(s, &eh, &c, op_idx, txn_id, tslot, addr, value);
                break;
            case STM_OP_VALIDATE:
                if (!active) rd_emit_invalid(s, &eh, &c, op_idx, txn_id);
                else rd_validate_op(s, &eh, &c, op_idx, txn_id, tslot);
                break;
            case STM_OP_TRY_PREPARE:
                if (!active) rd_emit_invalid(s, &eh, &c, op_idx, txn_id);
                else rd_try_prepare(s, &eh, &c, op_idx, txn_id, tslot);
                break;
            case STM_OP_DRAIN_COMMITS:
                rd_drain(s, &eh, &c, op_idx, aux);
                break;
            case STM_OP_RETRY:
                if (!active) rd_emit_invalid(s, &eh, &c, op_idx, txn_id);
                else rd_retry(s, &eh, &c, op_idx, txn_id, tslot, aux, in_watch + woff);
                break;
            case STM_OP_NON_TX_WRITE:
                rd_non_tx_write(s, &eh, &c, op_idx, addr, value);
                break;
            case STM_OP_ABORT:
                if (tslot < 0) rd_emit_invalid(s, &eh, &c, op_idx, txn_id);
                else rd_abort_op(s, &eh, &c, op_idx, txn_id, tslot);
                break;
            default:
                rd_emit_invalid(s, &eh, &c, op_idx, txn_id);
                break;
        }
        s.op_index[0] = (uint64_t)op_idx + 1;
    }

    for (int i = 0; i < 23; ++i) out_counts[i] = c.v[i];
    out_evt[0] = eh;
    out_read[0] = rh;
    out_loc[0] = rd_location_hash(s);
    out_txn[0] = rd_txn_hash(s);
    out_queue[0] = rd_queue_hash(s);
}

static StmRefDev ref_make_dev(const StmRefState* st) {
    StmRefDev s;
    s.max_txns = st->spec.max_txns;
    s.max_locations = st->spec.max_locations;
    s.max_read_set = st->spec.max_read_set;
    s.max_write_set = st->spec.max_write_set;
    s.max_watch_set = st->spec.max_watch_set;
    s.max_waiters = st->spec.max_waiters_per_location;
    s.max_retry_watchers = st->spec.max_retry_watchers_per_location;
    s.counters = st->counters; s.op_index = st->op_index;
    s.loc_used = st->loc_used; s.loc_addr = st->loc_addr; s.loc_value = st->loc_value;
    s.loc_version = st->loc_version; s.loc_lock_owner = st->loc_lock_owner;
    s.loc_write_seq = st->loc_write_seq;
    s.txn_used = st->txn_used; s.txn_id = st->txn_id; s.txn_xid = st->txn_xid;
    s.txn_begin_seq = st->txn_begin_seq; s.txn_attempt_no = st->txn_attempt_no;
    s.txn_priority = st->txn_priority; s.txn_start_version = st->txn_start_version;
    s.txn_status = st->txn_status; s.txn_wait_addr = st->txn_wait_addr;
    s.txn_wait_seq = st->txn_wait_seq; s.txn_prepare_seq = st->txn_prepare_seq;
    s.rs_used = st->rs_used; s.rs_addr = st->rs_addr; s.rs_read_version = st->rs_read_version;
    s.rs_read_value = st->rs_read_value; s.rs_from_own_write = st->rs_from_own_write;
    s.ws_used = st->ws_used; s.ws_addr = st->ws_addr; s.ws_value = st->ws_value;
    s.ws_write_set_seq = st->ws_write_set_seq;
    s.lq_count = st->lq_count; s.lq_addr = st->lq_addr; s.lq_txn = st->lq_txn; s.lq_seq = st->lq_seq;
    s.rq_used = st->rq_used; s.rq_eaddr = st->rq_eaddr; s.rq_etxn = st->rq_etxn;
    s.rq_eseq = st->rq_eseq; s.rq_eins = st->rq_eins; s.qins = st->qins;
    return s;
}

__global__ void stm_ref_init_counters(uint64_t* counters, uint64_t* op_index, uint64_t* qins) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    counters[C_GV] = 0; counters[C_ES] = 0; counters[C_XID] = 1; counters[C_BS] = 1;
    counters[C_WS] = 1; counters[C_WAIT] = 1; counters[C_RTY] = 1;
    op_index[0] = 0; qins[0] = 0;
}

static cudaError_t ref_reset_state(StmRefState* st, cudaStream_t stream) {
    cudaError_t err;
    const size_t L = (size_t)st->spec.max_locations;
    const size_t T = (size_t)st->spec.max_txns;
    err = cudaMemsetAsync(st->loc_used, 0, sizeof(uint8_t) * L, stream); if (err) return err;
    err = cudaMemsetAsync(st->txn_used, 0, sizeof(uint8_t) * T, stream); if (err) return err;
    const size_t RN = T * (size_t)st->spec.max_watch_set;
    err = cudaMemsetAsync(st->lq_count, 0, sizeof(int32_t) * L, stream); if (err) return err;
    err = cudaMemsetAsync(st->rq_used, 0, sizeof(uint8_t) * RN, stream); if (err) return err;
    stm_ref_init_counters<<<1, 1, 0, stream>>>(st->counters, st->op_index, st->qins);
    return cudaPeekAtLastError();
}

extern "C" size_t solution_workspace_bytes(const StmProblemSpec* spec) {
    if (!stm_validate_problem_spec(spec)) return 0;
    return 256;  // 23 int32 counts fit
}

#define REF_MALLOC(p, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(p)), (bytes)); \
    if (err != cudaSuccess) goto fail; } while (0)

extern "C" cudaError_t solution_init(const StmProblemSpec* spec, void** state_out,
                                     cudaStream_t stream) {
    if (!stm_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    StmRefState* st = (StmRefState*)malloc(sizeof(StmRefState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(StmRefState));
    memcpy(&st->spec, spec, sizeof(StmProblemSpec));

    const size_t L = (size_t)spec->max_locations;
    const size_t T = (size_t)spec->max_txns;
    const size_t RS = T * (size_t)spec->max_read_set;
    const size_t WS = T * (size_t)spec->max_write_set;
    const size_t LQ = L * (size_t)spec->max_waiters_per_location;
    const size_t RN = T * (size_t)spec->max_watch_set;  // retry queue entries

    cudaError_t err = cudaSuccess;
    REF_MALLOC(st->counters, sizeof(uint64_t) * 7);
    REF_MALLOC(st->op_index, sizeof(uint64_t));
    REF_MALLOC(st->loc_used, sizeof(uint8_t) * L);
    REF_MALLOC(st->loc_addr, sizeof(uint64_t) * L);
    REF_MALLOC(st->loc_value, sizeof(int64_t) * L);
    REF_MALLOC(st->loc_version, sizeof(uint64_t) * L);
    REF_MALLOC(st->loc_lock_owner, sizeof(uint64_t) * L);
    REF_MALLOC(st->loc_write_seq, sizeof(uint64_t) * L);
    REF_MALLOC(st->txn_used, sizeof(uint8_t) * T);
    REF_MALLOC(st->txn_id, sizeof(uint64_t) * T);
    REF_MALLOC(st->txn_xid, sizeof(uint64_t) * T);
    REF_MALLOC(st->txn_begin_seq, sizeof(uint64_t) * T);
    REF_MALLOC(st->txn_attempt_no, sizeof(uint64_t) * T);
    REF_MALLOC(st->txn_priority, sizeof(uint32_t) * T);
    REF_MALLOC(st->txn_start_version, sizeof(uint64_t) * T);
    REF_MALLOC(st->txn_status, sizeof(uint8_t) * T);
    REF_MALLOC(st->txn_wait_addr, sizeof(uint64_t) * T);
    REF_MALLOC(st->txn_wait_seq, sizeof(uint64_t) * T);
    REF_MALLOC(st->txn_prepare_seq, sizeof(uint64_t) * T);
    REF_MALLOC(st->rs_used, sizeof(uint8_t) * RS);
    REF_MALLOC(st->rs_addr, sizeof(uint64_t) * RS);
    REF_MALLOC(st->rs_read_version, sizeof(uint64_t) * RS);
    REF_MALLOC(st->rs_read_value, sizeof(int64_t) * RS);
    REF_MALLOC(st->rs_from_own_write, sizeof(uint8_t) * RS);
    REF_MALLOC(st->ws_used, sizeof(uint8_t) * WS);
    REF_MALLOC(st->ws_addr, sizeof(uint64_t) * WS);
    REF_MALLOC(st->ws_value, sizeof(int64_t) * WS);
    REF_MALLOC(st->ws_write_set_seq, sizeof(uint64_t) * WS);
    REF_MALLOC(st->lq_count, sizeof(int32_t) * L);
    REF_MALLOC(st->lq_addr, sizeof(uint64_t) * L);
    REF_MALLOC(st->lq_txn, sizeof(uint64_t) * LQ);
    REF_MALLOC(st->lq_seq, sizeof(uint64_t) * LQ);
    REF_MALLOC(st->qins, sizeof(uint64_t));
    REF_MALLOC(st->rq_used, sizeof(uint8_t) * RN);
    REF_MALLOC(st->rq_eaddr, sizeof(uint64_t) * RN);
    REF_MALLOC(st->rq_etxn, sizeof(uint64_t) * RN);
    REF_MALLOC(st->rq_eseq, sizeof(uint64_t) * RN);
    REF_MALLOC(st->rq_eins, sizeof(uint64_t) * RN);

    err = ref_reset_state(st, stream);
    if (err != cudaSuccess) goto fail;
    *state_out = st;
    return cudaSuccess;
fail:
    solution_destroy(st);
    return err;
}

__global__ void stm_ref_scatter(const int32_t* d, int32_t* o0, int32_t* o1, int32_t* o2,
    int32_t* o3, int32_t* o4, int32_t* o5, int32_t* o6, int32_t* o7, int32_t* o8,
    int32_t* o9, int32_t* o10, int32_t* o11, int32_t* o12, int32_t* o13, int32_t* o14,
    int32_t* o15, int32_t* o16, int32_t* o17, int32_t* o18, int32_t* o19, int32_t* o20,
    int32_t* o21, int32_t* o22) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    o0[0]=d[0]; o1[0]=d[1]; o2[0]=d[2]; o3[0]=d[3]; o4[0]=d[4]; o5[0]=d[5];
    o6[0]=d[6]; o7[0]=d[7]; o8[0]=d[8]; o9[0]=d[9]; o10[0]=d[10]; o11[0]=d[11];
    o12[0]=d[12]; o13[0]=d[13]; o14[0]=d[14]; o15[0]=d[15]; o16[0]=d[16];
    o17[0]=d[17]; o18[0]=d[18]; o19[0]=d[19]; o20[0]=d[20]; o21[0]=d[21]; o22[0]=d[22];
}

extern "C" cudaError_t solution_run(void* state, const StmRunSpec* run,
        const void* inputs_void, void* outputs_void, void* workspace,
        size_t workspace_bytes, cudaStream_t stream) {
    if (!state || !inputs_void || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 256) return cudaErrorInvalidValue;
    StmRefState* st = (StmRefState*)state;
    if (!stm_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const StmInputs* in = (const StmInputs*)inputs_void;
    StmOutputs* out = (StmOutputs*)outputs_void;
    if (run->batch_size > 0 && (!in->op_kind || !in->txn_id || !in->read_id ||
        !in->addr || !in->value || !in->aux || !in->watch_off)) {
        return cudaErrorInvalidValue;
    }

    int32_t* d_counts = reinterpret_cast<int32_t*>(workspace);

    StmRefDev s = ref_make_dev(st);
    stm_ref_step_kernel<<<1, 1, 0, stream>>>(
        s, run->batch_size,
        in->op_kind, in->txn_id, in->read_id, in->addr, in->value, in->aux,
        in->watch_off, in->watch_addrs,
        d_counts, out->stm_event_hash, out->read_result_hash,
        out->location_hash, out->txn_hash, out->queue_hash);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    stm_ref_scatter<<<1, 1, 0, stream>>>(d_counts,
        out->txn_begun, out->read_own_write, out->read_shared, out->write_staged,
        out->validate_ok, out->readonly_commits, out->txn_prepared, out->write_locks,
        out->wait_locks, out->commits_done, out->location_writes, out->non_tx_writes,
        out->non_tx_stalls, out->retry_suspended, out->retry_immediate,
        out->retry_watch_overflow, out->wake_retry, out->wake_lock, out->aborts,
        out->partial_unlocks, out->commit_unlocks, out->abort_unlocks, out->invalid_count);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return ref_reset_state((StmRefState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    StmRefState* st = (StmRefState*)state;
    cudaFree(st->counters); cudaFree(st->op_index);
    cudaFree(st->loc_used); cudaFree(st->loc_addr); cudaFree(st->loc_value);
    cudaFree(st->loc_version); cudaFree(st->loc_lock_owner); cudaFree(st->loc_write_seq);
    cudaFree(st->txn_used); cudaFree(st->txn_id); cudaFree(st->txn_xid);
    cudaFree(st->txn_begin_seq); cudaFree(st->txn_attempt_no); cudaFree(st->txn_priority);
    cudaFree(st->txn_start_version); cudaFree(st->txn_status); cudaFree(st->txn_wait_addr);
    cudaFree(st->txn_wait_seq); cudaFree(st->txn_prepare_seq);
    cudaFree(st->rs_used); cudaFree(st->rs_addr); cudaFree(st->rs_read_version);
    cudaFree(st->rs_read_value); cudaFree(st->rs_from_own_write);
    cudaFree(st->ws_used); cudaFree(st->ws_addr); cudaFree(st->ws_value);
    cudaFree(st->ws_write_set_seq);
    cudaFree(st->lq_count); cudaFree(st->lq_addr); cudaFree(st->lq_txn); cudaFree(st->lq_seq);
    cudaFree(st->qins);
    cudaFree(st->rq_used); cudaFree(st->rq_eaddr); cudaFree(st->rq_etxn);
    cudaFree(st->rq_eseq); cudaFree(st->rq_eins);
    free(st);
}
