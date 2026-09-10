// file: stm_commit_arbiter_oracle.hpp
//
// CPU reference (oracle) for T54 STM Commit Arbiter. Uses std:: containers
// (std::map / std::vector) and shares no algorithm code with the GPU reference
// or naive implementations. This is the source of truth for the coupled
// semantics: read/write-set validation, version checks, commit-time locking,
// deterministic commit-order arbitration, retry watchsets, and wake ordering.
//
// CANONICAL FIELD CONVENTIONS for stm_event_hash (a record is emitted for every
// event; event_seq consumed per event):
//   event_kind:u8; event_seq:u64; op_index:u32; txn_id_or_ZERO:u64;
//   addr_or_U64MAX:u64; value_or_I64MIN:i64; version_or_U64MAX:u64;
//   reason_or_255:u8; aux_u64.
// Per-event population (any field not meaningful -> its sentinel):
//   TXN_BEGIN:            txn; addr=MAX; value=MIN; version=start_version;
//                        reason=255; aux=priority.
//   READ_OWN_WRITE:      txn; addr; value=staged value; version=MAX;
//                        reason=255; aux=read_id.
//   READ_SHARED:         txn; addr; value=loc value; version=loc version;
//                        reason=255; aux=read_id.
//   WRITE_STAGE:         txn; addr; value=value; version=MAX; reason=255;
//                        aux=write_set_seq.
//   VALIDATE_OK:         txn; addr=MAX; value=MIN; version=MAX; reason=255; aux=0.
//   WRITE_LOCK:          txn; addr; value=MIN; version=loc version; reason=255;
//                        aux=0.
//   TXN_WAIT_LOCK:       txn; addr; value=MIN; version=MAX; reason=255;
//                        aux=wait_seq.
//   WRITE_UNLOCK_PARTIAL:txn; addr; value=MIN; version=MAX; reason=255; aux=0.
//   TXN_PREPARED:        txn; addr=MAX; value=MIN; version=MAX; reason=255;
//                        aux=prepare_seq.
//   COMMIT_READONLY:     txn; addr=MAX; value=MIN; version=MAX; reason=255; aux=0.
//   LOCATION_WRITE:      txn; addr; value=value; version=commit_version;
//                        reason=255; aux=write_seq.
//   WRITE_UNLOCK_COMMIT: txn; addr; value=MIN; version=MAX; reason=255; aux=0.
//   TXN_WAKE_RETRY:      txn; addr=wake addr; value=MIN; version=MAX;
//                        reason=255; aux=attempt_no (post-increment).
//   TXN_WAKE_LOCK:       txn; addr=wake addr; value=MIN; version=MAX;
//                        reason=255; aux=0.
//   COMMIT_DONE:         txn (committed txn); addr=MAX; value=MIN;
//                        version=commit_version; reason=255; aux=0.
//   TXN_SUSPEND_RETRY:   txn; addr=MAX; value=MIN; version=MAX; reason=255;
//                        aux=retry_seq.
//   RETRY_IMMEDIATE:     txn; addr=MAX; value=MIN; version=MAX; reason=255;
//                        aux=attempt_no (post-increment).
//   RETRY_WATCH_OVERFLOW:txn; addr=failing addr; value=MIN; version=MAX;
//                        reason=255; aux=0.
//   NON_TX_WRITE:        txn=0; addr; value=value; version=new version;
//                        reason=255; aux=0.
//   NON_TX_STALL_LOCKED: txn=0; addr; value=value; version=MAX; reason=255;
//                        aux=lock_owner.
//   NON_TX_OOM:          txn=0; addr; value=value; version=MAX; reason=255; aux=0.
//   WRITE_UNLOCK_ABORT:  txn; addr; value=MIN; version=MAX; reason=255; aux=0.
//   TXN_ABORT:           txn; addr=fail addr or MAX; value=MIN; version=MAX;
//                        reason=<reason>; aux=0.
//   INVALID:             txn; addr=MAX; value=MIN; version=MAX; reason=255; aux=0.

#ifndef STM_COMMIT_ARBITER_ORACLE_HPP_
#define STM_COMMIT_ARBITER_ORACLE_HPP_

#include "stm_commit_arbiter_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <deque>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct StmHostInputsView {
    const int32_t*  op_kind;
    const uint64_t* txn_id;
    const uint64_t* read_id;
    const uint64_t* addr;
    const int64_t*  value;
    const uint64_t* aux;
    const uint64_t* watch_off;
    const uint64_t* watch_addrs;
};

struct StmHostOutputsView {
    const int32_t* txn_begun;
    const int32_t* read_own_write;
    const int32_t* read_shared;
    const int32_t* write_staged;
    const int32_t* validate_ok;
    const int32_t* readonly_commits;
    const int32_t* txn_prepared;
    const int32_t* write_locks;
    const int32_t* wait_locks;
    const int32_t* commits_done;
    const int32_t* location_writes;
    const int32_t* non_tx_writes;
    const int32_t* non_tx_stalls;
    const int32_t* retry_suspended;
    const int32_t* retry_immediate;
    const int32_t* retry_watch_overflow;
    const int32_t* wake_retry;
    const int32_t* wake_lock;
    const int32_t* aborts;
    const int32_t* partial_unlocks;
    const int32_t* commit_unlocks;
    const int32_t* abort_unlocks;
    const int32_t* invalid_count;
    const uint64_t* stm_event_hash;
    const uint64_t* read_result_hash;
    const uint64_t* location_hash;
    const uint64_t* txn_hash;
    const uint64_t* queue_hash;
};

struct StmExpected {
    int32_t txn_begun = 0, read_own_write = 0, read_shared = 0, write_staged = 0;
    int32_t validate_ok = 0, readonly_commits = 0, txn_prepared = 0, write_locks = 0;
    int32_t wait_locks = 0, commits_done = 0, location_writes = 0, non_tx_writes = 0;
    int32_t non_tx_stalls = 0, retry_suspended = 0, retry_immediate = 0;
    int32_t retry_watch_overflow = 0, wake_retry = 0, wake_lock = 0, aborts = 0;
    int32_t partial_unlocks = 0, commit_unlocks = 0, abort_unlocks = 0, invalid_count = 0;
    uint64_t stm_event_hash = STM_FNV_OFFSET;
    uint64_t read_result_hash = STM_FNV_OFFSET;
    uint64_t location_hash = STM_FNV_OFFSET;
    uint64_t txn_hash = STM_FNV_OFFSET;
    uint64_t queue_hash = STM_FNV_OFFSET;
};

static inline uint64_t stm_o_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= STM_FNV_PRIME; return h;
}
static inline void stm_o_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*)p; uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = stm_o_fnv_byte(v, q[i]);
    *h = v;
}
static inline void stm_o_u8(uint64_t* h, uint8_t v)   { stm_o_bytes(h, &v, 1); }
static inline void stm_o_u32(uint64_t* h, uint32_t v) { stm_o_bytes(h, &v, 4); }
static inline void stm_o_u64(uint64_t* h, uint64_t v) { stm_o_bytes(h, &v, 8); }
static inline void stm_o_i64(uint64_t* h, int64_t v)  { stm_o_bytes(h, &v, 8); }

struct StmOLocation {
    int64_t  value = 0;
    uint64_t version = 0;
    uint64_t lock_owner = 0;  // 0 = unlocked
    uint64_t write_seq = 0;
};

struct StmOReadEntry {
    uint64_t read_version = 0;
    int64_t  read_value = 0;
    uint8_t  from_own_write = 0;
};

struct StmOWriteEntry {
    int64_t  value = 0;
    uint64_t write_set_seq = 0;
};

struct StmOTxn {
    uint64_t xid = 0;
    uint64_t begin_seq = 0;
    uint64_t attempt_no = 0;
    uint32_t priority = 0;
    uint64_t start_version = 0;
    uint8_t  status = STM_ST_ACTIVE;
    uint64_t wait_addr = STM_U64_MAX;
    uint64_t wait_seq = STM_U64_MAX;
    uint64_t prepare_seq = STM_U64_MAX;
    std::map<uint64_t, StmOReadEntry> read_set;
    std::map<uint64_t, StmOWriteEntry> write_set;
};

struct StmOQueueEntry {
    uint64_t txn_id;
    uint64_t seq;
};

struct StmOracleState {
    StmProblemSpec spec{};

    uint64_t global_version = 0;
    uint64_t event_seq = 0;
    uint64_t xid_next = 1;
    uint64_t begin_seq_next = 1;
    uint64_t write_seq_next = 1;
    uint64_t wait_seq_next = 1;
    uint64_t retry_seq_next = 1;

    uint64_t op_index = 0;

    std::map<uint64_t, StmOLocation> locs;            // addr -> location
    std::map<uint64_t, StmOTxn> txns;                 // txn_id -> txn
    std::map<uint64_t, std::deque<StmOQueueEntry>> lock_q;   // addr -> wait queue
    std::map<uint64_t, std::deque<StmOQueueEntry>> retry_q;  // addr -> retry queue

    void init(const StmProblemSpec& s) { spec = s; reset(); }

    void reset() {
        global_version = 0; event_seq = 0; xid_next = 1; begin_seq_next = 1;
        write_seq_next = 1; wait_seq_next = 1; retry_seq_next = 1; op_index = 0;
        locs.clear(); txns.clear(); lock_q.clear(); retry_q.clear();
    }

    // ---- event emission -------------------------------------------------
    void emit(StmExpected* e, uint8_t kind, uint32_t op_idx, uint64_t txn_id,
              uint64_t addr, int64_t value, uint64_t version, uint8_t reason,
              uint64_t aux) {
        uint64_t* h = &e->stm_event_hash;
        stm_o_u8(h, kind);
        stm_o_u64(h, event_seq);
        stm_o_u32(h, op_idx);
        stm_o_u64(h, txn_id);
        stm_o_u64(h, addr);
        stm_o_i64(h, value);
        stm_o_u64(h, version);
        stm_o_u8(h, reason);
        stm_o_u64(h, aux);
        event_seq += 1;
    }

    void emit_read_result(StmExpected* e, uint64_t read_id, uint64_t txn_id,
                          uint64_t addr, uint8_t result_kind, int64_t value,
                          uint64_t version, uint64_t attempt_no) {
        uint64_t* h = &e->read_result_hash;
        stm_o_u64(h, read_id);
        stm_o_u64(h, txn_id);
        stm_o_u64(h, addr);
        stm_o_u8(h, result_kind);
        stm_o_i64(h, value);
        stm_o_u64(h, version);
        stm_o_u64(h, attempt_no);
    }

    void emit_invalid(StmExpected* e, uint32_t op_idx, uint64_t txn_id) {
        emit(e, STM_EVT_INVALID, op_idx, txn_id, STM_U64_MAX, STM_I64_MIN,
             STM_U64_MAX, STM_RS_NONE, 0);
        e->invalid_count += 1;
    }

    // location lookup (absent default)
    StmOLocation get_loc(uint64_t addr) const {
        auto it = locs.find(addr);
        if (it == locs.end()) return StmOLocation();
        return it->second;
    }

    // Remove txn from all lock + retry queues (silently).
    void remove_from_all_queues(uint64_t txn_id) {
        for (auto& kv : lock_q) {
            auto& q = kv.second;
            for (size_t i = 0; i < q.size();) {
                if (q[i].txn_id == txn_id) q.erase(q.begin() + i);
                else ++i;
            }
        }
        for (auto& kv : retry_q) {
            auto& q = kv.second;
            for (size_t i = 0; i < q.size();) {
                if (q[i].txn_id == txn_id) q.erase(q.begin() + i);
                else ++i;
            }
        }
    }

    // Wake at most one lock waiter for addr (commit/abort rule). Drops stale
    // entries silently. Returns true if a waiter was woken; emits TXN_WAKE_LOCK.
    bool wake_one_lock_waiter(StmExpected* e, uint32_t op_idx, uint64_t addr) {
        auto it = lock_q.find(addr);
        if (it == lock_q.end()) return false;
        auto& q = it->second;
        while (!q.empty()) {
            uint64_t cand = q.front().txn_id;
            auto tit = txns.find(cand);
            if (tit == txns.end() || tit->second.status != STM_ST_WAITING_LOCK ||
                tit->second.wait_addr != addr) {
                q.pop_front();  // stale, silent
                continue;
            }
            // wake it
            q.pop_front();
            StmOTxn& t = tit->second;
            t.status = STM_ST_ACTIVE;
            t.wait_addr = STM_U64_MAX;
            t.wait_seq = STM_U64_MAX;
            emit(e, STM_EVT_TXN_WAKE_LOCK, op_idx, cand, addr, STM_I64_MIN,
                 STM_U64_MAX, STM_RS_NONE, 0);
            e->wake_lock += 1;
            return true;
        }
        return false;
    }

    // Wake all retry watchers for addr in queue order (commit/non-tx rule).
    void wake_retry_watchers(StmExpected* e, uint32_t op_idx, uint64_t addr) {
        auto it = retry_q.find(addr);
        if (it == retry_q.end()) return;
        // Snapshot the queue order for this addr; process front-to-back.
        std::deque<StmOQueueEntry> q = it->second;
        for (size_t i = 0; i < q.size(); ++i) {
            uint64_t cand = q[i].txn_id;
            auto tit = txns.find(cand);
            if (tit == txns.end()) continue;            // absent
            StmOTxn& t = tit->second;
            if (t.status != STM_ST_SUSPENDED_RETRY) continue;  // already active/removed
            // wake: remove from ALL retry queues, clear sets, attempt_no++,
            // start_version=global_version, status ACTIVE, emit TXN_WAKE_RETRY.
            // Remove from all retry queues.
            for (auto& kv : retry_q) {
                auto& qq = kv.second;
                for (size_t j = 0; j < qq.size();) {
                    if (qq[j].txn_id == cand) qq.erase(qq.begin() + j);
                    else ++j;
                }
            }
            t.read_set.clear();
            t.write_set.clear();
            t.attempt_no += 1;
            t.start_version = global_version;
            t.status = STM_ST_ACTIVE;
            emit(e, STM_EVT_TXN_WAKE_RETRY, op_idx, cand, addr, STM_I64_MIN,
                 STM_U64_MAX, STM_RS_NONE, t.attempt_no);
            e->wake_retry += 1;
        }
    }

    // Release all locks held by txn ascending addr, emit unlock_kind per
    // location, and (for abort) emit wake-lock per released location.
    // Returns nothing; updates counts via caller-provided count selection.
    void release_all_locks_abort(StmExpected* e, uint32_t op_idx, uint64_t txn_id) {
        std::vector<uint64_t> held;
        for (auto& kv : locs) {
            if (kv.second.lock_owner == txn_id) held.push_back(kv.first);
        }
        std::sort(held.begin(), held.end());
        for (uint64_t addr : held) {
            locs[addr].lock_owner = 0;
            emit(e, STM_EVT_WRITE_UNLOCK_ABORT, op_idx, txn_id, addr, STM_I64_MIN,
                 STM_U64_MAX, STM_RS_NONE, 0);
            e->abort_unlocks += 1;
            wake_one_lock_waiter(e, op_idx, addr);
        }
    }

    // ---- op handlers ----------------------------------------------------

    void op_begin(StmExpected* e, uint32_t op_idx, uint64_t txn_id, uint64_t priority) {
        if (txns.count(txn_id) != 0 || (int)txns.size() >= spec.max_txns) {
            emit_invalid(e, op_idx, txn_id);
            return;
        }
        StmOTxn t;
        t.xid = xid_next++;
        t.begin_seq = begin_seq_next++;
        t.attempt_no = 0;
        t.priority = (uint32_t)priority;
        t.start_version = global_version;
        t.status = STM_ST_ACTIVE;
        t.wait_addr = STM_U64_MAX;
        t.wait_seq = STM_U64_MAX;
        t.prepare_seq = STM_U64_MAX;
        emit(e, STM_EVT_TXN_BEGIN, op_idx, txn_id, STM_U64_MAX, STM_I64_MIN,
             t.start_version, STM_RS_NONE, (uint64_t)t.priority);
        txns[txn_id] = std::move(t);
        e->txn_begun += 1;
    }

    // Abort a transaction (common path used by TX_READ conflicts). Removes sets,
    // releases locks (WRITE_UNLOCK_ABORT + wake), emits TXN_ABORT(reason).
    void abort_txn(StmExpected* e, uint32_t op_idx, uint64_t txn_id, uint8_t reason,
                   uint64_t fail_addr) {
        auto it = txns.find(txn_id);
        if (it == txns.end()) return;
        // release locks held (ascending) with wake events
        release_all_locks_abort(e, op_idx, txn_id);
        // remove from queues
        remove_from_all_queues(txn_id);
        // clear sets implicitly by erasing txn
        emit(e, STM_EVT_TXN_ABORT, op_idx, txn_id, fail_addr, STM_I64_MIN,
             STM_U64_MAX, reason, 0);
        e->aborts += 1;
        txns.erase(it);
    }

    void op_tx_read(StmExpected* e, uint32_t op_idx, uint64_t txn_id,
                    uint64_t read_id, uint64_t addr) {
        auto it = txns.find(txn_id);
        if (it == txns.end() || it->second.status != STM_ST_ACTIVE) {
            emit_invalid(e, op_idx, txn_id);
            return;
        }
        StmOTxn& t = it->second;
        uint64_t attempt = t.attempt_no;

        auto wit = t.write_set.find(addr);
        if (wit != t.write_set.end()) {
            int64_t v = wit->second.value;
            StmOReadEntry re;
            re.from_own_write = 1;
            re.read_version = STM_U64_MAX;
            re.read_value = v;
            t.read_set[addr] = re;
            emit(e, STM_EVT_READ_OWN_WRITE, op_idx, txn_id, addr, v, STM_U64_MAX,
                 STM_RS_NONE, read_id);
            emit_read_result(e, read_id, txn_id, addr, STM_READ_RES_OWN_WRITE, v,
                             STM_U64_MAX, attempt);
            e->read_own_write += 1;
            return;
        }

        StmOLocation loc = get_loc(addr);
        if (loc.lock_owner != 0 && loc.lock_owner != txn_id) {
            // abort READ_LOCK_CONFLICT
            emit_read_result(e, read_id, txn_id, addr, STM_READ_RES_ABORT_LOCK,
                             STM_I64_MIN, STM_U64_MAX, attempt);
            abort_txn(e, op_idx, txn_id, STM_RS_READ_LOCK_CONFLICT, addr);
            return;
        }
        if (loc.version > t.start_version) {
            emit_read_result(e, read_id, txn_id, addr, STM_READ_RES_ABORT_VER,
                             STM_I64_MIN, STM_U64_MAX, attempt);
            abort_txn(e, op_idx, txn_id, STM_RS_READ_VERSION_CONFLICT, addr);
            return;
        }
        StmOReadEntry re;
        re.from_own_write = 0;
        re.read_version = loc.version;
        re.read_value = loc.value;
        t.read_set[addr] = re;
        emit(e, STM_EVT_READ_SHARED, op_idx, txn_id, addr, loc.value, loc.version,
             STM_RS_NONE, read_id);
        emit_read_result(e, read_id, txn_id, addr, STM_READ_RES_SHARED, loc.value,
                         loc.version, attempt);
        e->read_shared += 1;
    }

    void op_tx_write(StmExpected* e, uint32_t op_idx, uint64_t txn_id,
                     uint64_t addr, int64_t value) {
        auto it = txns.find(txn_id);
        if (it == txns.end() || it->second.status != STM_ST_ACTIVE) {
            emit_invalid(e, op_idx, txn_id);
            return;
        }
        StmOTxn& t = it->second;
        auto wit = t.write_set.find(addr);
        uint64_t wseq;
        if (wit != t.write_set.end()) {
            wit->second.value = value;
            wseq = wit->second.write_set_seq;
        } else {
            if ((int)t.write_set.size() >= spec.max_write_set) {
                emit_invalid(e, op_idx, txn_id);
                return;
            }
            StmOWriteEntry we;
            we.value = value;
            we.write_set_seq = write_seq_next++;
            wseq = we.write_set_seq;
            t.write_set[addr] = we;
        }
        emit(e, STM_EVT_WRITE_STAGE, op_idx, txn_id, addr, value, STM_U64_MAX,
             STM_RS_NONE, wseq);
        e->write_staged += 1;
    }

    // Validation that checks lock-by-other and version equality. Returns true
    // if OK; on first failure sets *fail_addr.
    bool validate_full(const StmOTxn& t, uint64_t self_txn, uint64_t* fail_addr) const {
        for (auto it = t.read_set.begin(); it != t.read_set.end(); ++it) {
            uint64_t addr = it->first;
            const StmOReadEntry& re = it->second;
            if (re.from_own_write) continue;
            if (t.write_set.find(addr) != t.write_set.end()) continue;
            StmOLocation loc = get_loc(addr);
            if (loc.lock_owner != 0 && loc.lock_owner != self_txn) {
                *fail_addr = addr; return false;
            }
            if (loc.version != re.read_version) { *fail_addr = addr; return false; }
        }
        return true;
    }

    void op_validate(StmExpected* e, uint32_t op_idx, uint64_t txn_id) {
        auto it = txns.find(txn_id);
        if (it == txns.end() || it->second.status != STM_ST_ACTIVE) {
            emit_invalid(e, op_idx, txn_id);
            return;
        }
        StmOTxn& t = it->second;
        uint64_t fail_addr = STM_U64_MAX;
        if (!validate_full(t, txn_id, &fail_addr)) {
            abort_txn(e, op_idx, txn_id, STM_RS_VALIDATE_FAIL, fail_addr);
            return;
        }
        emit(e, STM_EVT_VALIDATE_OK, op_idx, txn_id, STM_U64_MAX, STM_I64_MIN,
             STM_U64_MAX, STM_RS_NONE, 0);
        e->validate_ok += 1;
    }

    int count_materialized_locations() const { return (int)locs.size(); }

    void op_try_prepare(StmExpected* e, uint32_t op_idx, uint64_t txn_id) {
        auto it = txns.find(txn_id);
        if (it == txns.end() || it->second.status != STM_ST_ACTIVE) {
            emit_invalid(e, op_idx, txn_id);
            return;
        }
        StmOTxn& t = it->second;

        if (t.write_set.empty()) {
            emit(e, STM_EVT_COMMIT_READONLY, op_idx, txn_id, STM_U64_MAX,
                 STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, 0);
            e->readonly_commits += 1;
            remove_from_all_queues(txn_id);
            txns.erase(it);
            return;
        }

        // Lock write-set addrs ascending. Track locks acquired this prepare.
        std::vector<uint64_t> waddrs;
        for (auto wit = t.write_set.begin(); wit != t.write_set.end(); ++wit)
            waddrs.push_back(wit->first);
        // std::map already ascending, but be explicit.
        std::sort(waddrs.begin(), waddrs.end());

        std::vector<uint64_t> acquired;  // locks acquired in this prepare attempt

        for (size_t i = 0; i < waddrs.size(); ++i) {
            uint64_t addr = waddrs[i];
            auto lit = locs.find(addr);
            bool absent = (lit == locs.end());
            if (absent) {
                // can lock only if room to materialize later
                int materialized = count_materialized_locations();
                // count addresses we've already newly-created this prepare
                // (none materialized yet; we only set lock_owner on existing or
                //  create a placeholder). We create a placeholder location so it
                //  occupies a slot.
                if (materialized >= spec.max_locations) {
                    // PREPARE_OOM: release acquired locks descending, then abort.
                    release_acquired_partial(e, op_idx, txn_id, acquired);
                    abort_txn(e, op_idx, txn_id, STM_RS_PREPARE_OOM, addr);
                    return;
                }
                // materialize placeholder (value 0, version 0) and lock it.
                StmOLocation nl;
                nl.value = 0; nl.version = 0; nl.lock_owner = txn_id; nl.write_seq = 0;
                locs[addr] = nl;
                acquired.push_back(addr);
                emit(e, STM_EVT_WRITE_LOCK, op_idx, txn_id, addr, STM_I64_MIN,
                     nl.version, STM_RS_NONE, 0);
                e->write_locks += 1;
                continue;
            }
            StmOLocation& loc = lit->second;
            if (loc.lock_owner == 0 || loc.lock_owner == txn_id) {
                loc.lock_owner = txn_id;
                acquired.push_back(addr);
                emit(e, STM_EVT_WRITE_LOCK, op_idx, txn_id, addr, STM_I64_MIN,
                     loc.version, STM_RS_NONE, 0);
                e->write_locks += 1;
                continue;
            }
            // locked by another txn -> wait.
            // Check queue capacity.
            auto& q = lock_q[addr];
            if ((int)q.size() >= spec.max_waiters_per_location) {
                // LOCK_WAIT_OVERFLOW: release acquired descending then abort.
                release_acquired_partial(e, op_idx, txn_id, acquired);
                abort_txn(e, op_idx, txn_id, STM_RS_LOCK_WAIT_OVERFLOW, addr);
                return;
            }
            t.status = STM_ST_WAITING_LOCK;
            t.wait_addr = addr;
            t.wait_seq = wait_seq_next++;
            StmOQueueEntry qe; qe.txn_id = txn_id; qe.seq = t.wait_seq;
            q.push_back(qe);
            emit(e, STM_EVT_TXN_WAIT_LOCK, op_idx, txn_id, addr, STM_I64_MIN,
                 STM_U64_MAX, STM_RS_NONE, t.wait_seq);
            e->wait_locks += 1;
            // release locks acquired earlier in THIS prepare in descending order.
            release_acquired_partial(e, op_idx, txn_id, acquired);
            return;  // stop operation
        }

        // All write locks held. Validate read set as VALIDATE.
        uint64_t fail_addr = STM_U64_MAX;
        if (!validate_full(t, txn_id, &fail_addr)) {
            abort_txn(e, op_idx, txn_id, STM_RS_PREPARE_VALIDATE_FAIL, fail_addr);
            return;
        }
        t.status = STM_ST_PREPARED;
        t.prepare_seq = event_seq;  // seq the TXN_PREPARED event consumes
        emit(e, STM_EVT_TXN_PREPARED, op_idx, txn_id, STM_U64_MAX, STM_I64_MIN,
             STM_U64_MAX, STM_RS_NONE, t.prepare_seq);
        e->txn_prepared += 1;
    }

    // Release locks acquired in this prepare in descending addr order; emit
    // WRITE_UNLOCK_PARTIAL + wake-lock per location.
    void release_acquired_partial(StmExpected* e, uint32_t op_idx, uint64_t txn_id,
                                  std::vector<uint64_t>& acquired) {
        std::sort(acquired.begin(), acquired.end());
        for (size_t k = acquired.size(); k-- > 0;) {
            uint64_t addr = acquired[k];
            auto lit = locs.find(addr);
            if (lit != locs.end() && lit->second.lock_owner == txn_id) {
                lit->second.lock_owner = 0;
            }
            emit(e, STM_EVT_WRITE_UNLOCK_PARTIAL, op_idx, txn_id, addr,
                 STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, 0);
            e->partial_unlocks += 1;
            wake_one_lock_waiter(e, op_idx, addr);
        }
        acquired.clear();
    }

    // Select prepared txn: priority desc, prepare_seq asc, txn_id asc.
    uint64_t pick_prepared() const {
        bool have = false;
        uint64_t best_id = 0; uint32_t best_pri = 0; uint64_t best_pseq = 0;
        for (auto it = txns.begin(); it != txns.end(); ++it) {
            if (it->second.status != STM_ST_PREPARED) continue;
            uint32_t pri = it->second.priority;
            uint64_t pseq = it->second.prepare_seq;
            uint64_t id = it->first;
            if (!have) { have = true; best_id = id; best_pri = pri; best_pseq = pseq; continue; }
            if (pri > best_pri ||
                (pri == best_pri && pseq < best_pseq) ||
                (pri == best_pri && pseq == best_pseq && id < best_id)) {
                best_id = id; best_pri = pri; best_pseq = pseq;
            }
        }
        return have ? best_id : STM_U64_MAX;
    }

    void op_drain_commits(StmExpected* e, uint32_t op_idx, uint64_t limit) {
        if (limit == 0) return;  // valid no-op
        uint64_t budget = limit;
        while (budget > 0) {
            uint64_t cid = pick_prepared();
            if (cid == STM_U64_MAX) break;
            StmOTxn& t = txns[cid];
            uint64_t commit_version = ++global_version;

            // Apply writes ascending addr.
            std::vector<uint64_t> waddrs;
            for (auto wit = t.write_set.begin(); wit != t.write_set.end(); ++wit)
                waddrs.push_back(wit->first);
            std::sort(waddrs.begin(), waddrs.end());

            for (uint64_t addr : waddrs) {
                StmOWriteEntry& we = t.write_set[addr];
                StmOLocation& loc = locs[addr];  // materialize if needed
                loc.value = we.value;
                loc.version = commit_version;
                loc.write_seq = event_seq;  // seq the LOCATION_WRITE consumes
                // lock_owner unchanged here (still owned by cid); released below.
                emit(e, STM_EVT_LOCATION_WRITE, op_idx, cid, addr, we.value,
                     commit_version, STM_RS_NONE, loc.write_seq);
                e->location_writes += 1;
            }

            // Remove txn, release its write locks ascending addr.
            // The locks held by this txn are exactly the write-set addrs (locked
            // in prepare). Release ascending.
            std::vector<uint64_t> held;
            for (auto& kv : locs) if (kv.second.lock_owner == cid) held.push_back(kv.first);
            std::sort(held.begin(), held.end());
            // Remove txn first from table so wake logic sees it gone.
            remove_from_all_queues(cid);
            txns.erase(cid);
            for (uint64_t addr : held) {
                locs[addr].lock_owner = 0;
                emit(e, STM_EVT_WRITE_UNLOCK_COMMIT, op_idx, cid, addr, STM_I64_MIN,
                     STM_U64_MAX, STM_RS_NONE, 0);
                e->commit_unlocks += 1;
            }

            // Wake retry watchers for each written addr ascending.
            for (uint64_t addr : waddrs) {
                wake_retry_watchers(e, op_idx, addr);
            }
            // Wake at most one lock waiter for each unlocked addr ascending.
            for (uint64_t addr : held) {
                wake_one_lock_waiter(e, op_idx, addr);
            }

            emit(e, STM_EVT_COMMIT_DONE, op_idx, cid, STM_U64_MAX, STM_I64_MIN,
                 commit_version, STM_RS_NONE, 0);
            e->commits_done += 1;
            budget -= 1;
        }
    }

    void op_retry(StmExpected* e, uint32_t op_idx, uint64_t txn_id,
                  uint64_t watch_count, const uint64_t* watch_addrs) {
        auto it = txns.find(txn_id);
        if (it == txns.end() || it->second.status != STM_ST_ACTIVE ||
            watch_count == 0 || watch_count > (uint64_t)spec.max_watch_set) {
            emit_invalid(e, op_idx, txn_id);
            return;
        }
        StmOTxn& t = it->second;
        // Dedup ascending.
        std::vector<uint64_t> ws(watch_addrs, watch_addrs + watch_count);
        std::sort(ws.begin(), ws.end());
        ws.erase(std::unique(ws.begin(), ws.end()), ws.end());
        if ((uint64_t)ws.size() > (uint64_t)spec.max_watch_set) {
            emit_invalid(e, op_idx, txn_id);
            return;
        }

        // Validate read set.
        uint64_t fail_addr = STM_U64_MAX;
        if (!validate_full(t, txn_id, &fail_addr)) {
            t.read_set.clear();
            t.write_set.clear();
            t.attempt_no += 1;
            t.start_version = global_version;
            // status stays ACTIVE
            emit(e, STM_EVT_RETRY_IMMEDIATE, op_idx, txn_id, STM_U64_MAX,
                 STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, t.attempt_no);
            e->retry_immediate += 1;
            return;
        }

        // Append to each watched location's retry watch queue ascending addr,
        // single retry_seq for all. Check capacity first per address; if any
        // lacks capacity, roll back all added entries.
        uint64_t rseq = retry_seq_next;  // tentative
        // First pass: capacity check + record additions.
        bool overflow = false;
        uint64_t overflow_addr = STM_U64_MAX;
        std::vector<uint64_t> added;
        for (uint64_t addr : ws) {
            auto& q = retry_q[addr];
            if ((int)q.size() >= spec.max_retry_watchers_per_location) {
                overflow = true; overflow_addr = addr; break;
            }
            StmOQueueEntry qe; qe.txn_id = txn_id; qe.seq = rseq;
            q.push_back(qe);
            added.push_back(addr);
        }
        if (overflow) {
            // roll back additions
            for (uint64_t addr : added) {
                auto& q = retry_q[addr];
                // remove the last entry we added with this txn_id+rseq
                for (size_t j = q.size(); j-- > 0;) {
                    if (q[j].txn_id == txn_id && q[j].seq == rseq) {
                        q.erase(q.begin() + j);
                        break;
                    }
                }
            }
            emit(e, STM_EVT_RETRY_WATCH_OVERFLOW, op_idx, txn_id, overflow_addr,
                 STM_I64_MIN, STM_U64_MAX, STM_RS_NONE, 0);
            e->retry_watch_overflow += 1;
            return;
        }
        // success: consume retry_seq, clear write set, KEEP read set, suspend.
        retry_seq_next++;
        t.write_set.clear();
        t.status = STM_ST_SUSPENDED_RETRY;
        emit(e, STM_EVT_TXN_SUSPEND_RETRY, op_idx, txn_id, STM_U64_MAX, STM_I64_MIN,
             STM_U64_MAX, STM_RS_NONE, rseq);
        e->retry_suspended += 1;
    }

    void op_non_tx_write(StmExpected* e, uint32_t op_idx, uint64_t addr, int64_t value) {
        auto lit = locs.find(addr);
        bool absent = (lit == locs.end());
        if (!absent && lit->second.lock_owner != 0) {
            emit(e, STM_EVT_NON_TX_STALL_LOCKED, op_idx, 0, addr, value,
                 STM_U64_MAX, STM_RS_NONE, lit->second.lock_owner);
            e->non_tx_stalls += 1;
            return;
        }
        if (absent && (int)locs.size() >= spec.max_locations) {
            emit(e, STM_EVT_NON_TX_OOM, op_idx, 0, addr, value, STM_U64_MAX,
                 STM_RS_NONE, 0);
            // non_tx_stalls? no -- this is OOM, no dedicated count. Per spec
            // count list has no non_tx_oom; it is implicitly observable via
            // event hash. We do not bump non_tx_writes.
            return;
        }
        StmOLocation& loc = locs[addr];  // materialize if needed
        loc.version = ++global_version;
        loc.value = value;
        loc.write_seq = event_seq;  // seq the NON_TX_WRITE event consumes
        emit(e, STM_EVT_NON_TX_WRITE, op_idx, 0, addr, value, loc.version,
             STM_RS_NONE, 0);
        e->non_tx_writes += 1;
        // wake retry watchers for addr
        wake_retry_watchers(e, op_idx, addr);
    }

    void op_abort(StmExpected* e, uint32_t op_idx, uint64_t txn_id) {
        auto it = txns.find(txn_id);
        if (it == txns.end()) {
            emit_invalid(e, op_idx, txn_id);
            return;
        }
        abort_txn(e, op_idx, txn_id, STM_RS_EXPLICIT_ABORT, STM_U64_MAX);
    }

    // ---- state hashes ---------------------------------------------------

    uint64_t location_hash() const {
        uint64_t h = STM_FNV_OFFSET;
        for (auto it = locs.begin(); it != locs.end(); ++it) {
            stm_o_u64(&h, it->first);
            stm_o_i64(&h, it->second.value);
            stm_o_u64(&h, it->second.version);
            stm_o_u64(&h, it->second.lock_owner);
            stm_o_u64(&h, it->second.write_seq);
        }
        return h;
    }

    uint64_t txn_hash() const {
        uint64_t h = STM_FNV_OFFSET;
        for (auto it = txns.begin(); it != txns.end(); ++it) {
            const StmOTxn& t = it->second;
            stm_o_u64(&h, it->first);
            stm_o_u64(&h, t.xid);
            stm_o_u64(&h, t.begin_seq);
            stm_o_u64(&h, t.attempt_no);
            stm_o_u32(&h, t.priority);
            stm_o_u64(&h, t.start_version);
            stm_o_u8(&h, t.status);
            stm_o_u64(&h, t.wait_addr);
            stm_o_u64(&h, t.prepare_seq);
            for (auto rit = t.read_set.begin(); rit != t.read_set.end(); ++rit) {
                stm_o_u64(&h, rit->first);
                stm_o_u64(&h, rit->second.read_version);
                stm_o_i64(&h, rit->second.read_value);
                stm_o_u8(&h, rit->second.from_own_write);
            }
            for (auto wit = t.write_set.begin(); wit != t.write_set.end(); ++wit) {
                stm_o_u64(&h, wit->first);
                stm_o_i64(&h, wit->second.value);
                stm_o_u64(&h, wit->second.write_set_seq);
            }
        }
        return h;
    }

    uint64_t queue_hash() const {
        uint64_t h = STM_FNV_OFFSET;
        for (auto it = lock_q.begin(); it != lock_q.end(); ++it) {
            const std::deque<StmOQueueEntry>& q = it->second;
            for (size_t pos = 0; pos < q.size(); ++pos) {
                stm_o_u8(&h, (uint8_t)STM_QK_LOCK_WAIT);
                stm_o_u64(&h, it->first);
                stm_o_u64(&h, (uint64_t)pos);
                stm_o_u64(&h, q[pos].txn_id);
                stm_o_u64(&h, q[pos].seq);
            }
        }
        for (auto it = retry_q.begin(); it != retry_q.end(); ++it) {
            const std::deque<StmOQueueEntry>& q = it->second;
            for (size_t pos = 0; pos < q.size(); ++pos) {
                stm_o_u8(&h, (uint8_t)STM_QK_RETRY_WATCH);
                stm_o_u64(&h, it->first);
                stm_o_u64(&h, (uint64_t)pos);
                stm_o_u64(&h, q[pos].txn_id);
                stm_o_u64(&h, q[pos].seq);
            }
        }
        return h;
    }

    void step_once(const StmRunSpec& run, const StmHostInputsView& in, StmExpected* exp) {
        *exp = StmExpected();
        for (int r = 0; r < run.batch_size; ++r) {
            const uint32_t op_idx = (uint32_t)op_index;
            const int32_t kind = in.op_kind[r];
            const uint64_t txn_id = in.txn_id[r];
            const uint64_t read_id = in.read_id[r];
            const uint64_t addr = in.addr[r];
            const int64_t value = in.value[r];
            const uint64_t aux = in.aux[r];
            const uint64_t woff = in.watch_off[r];

            switch (kind) {
                case STM_OP_BEGIN:
                    op_begin(exp, op_idx, txn_id, aux);
                    break;
                case STM_OP_TX_READ:
                    op_tx_read(exp, op_idx, txn_id, read_id, addr);
                    break;
                case STM_OP_TX_WRITE:
                    op_tx_write(exp, op_idx, txn_id, addr, value);
                    break;
                case STM_OP_VALIDATE:
                    op_validate(exp, op_idx, txn_id);
                    break;
                case STM_OP_TRY_PREPARE:
                    op_try_prepare(exp, op_idx, txn_id);
                    break;
                case STM_OP_DRAIN_COMMITS:
                    op_drain_commits(exp, op_idx, aux);
                    break;
                case STM_OP_RETRY:
                    op_retry(exp, op_idx, txn_id, aux, in.watch_addrs + woff);
                    break;
                case STM_OP_NON_TX_WRITE:
                    op_non_tx_write(exp, op_idx, addr, value);
                    break;
                case STM_OP_ABORT:
                    op_abort(exp, op_idx, txn_id);
                    break;
                default:
                    emit_invalid(exp, op_idx, txn_id);
                    break;
            }
            op_index += 1;
        }
        exp->location_hash = location_hash();
        exp->txn_hash = txn_hash();
        exp->queue_hash = queue_hash();
    }
};

static inline bool stm_check_all_outputs(const StmExpected& e,
                                         const StmHostOutputsView& g,
                                         std::string* error) {
#define STM_CHECK_INT(field)                                                   \
    if (g.field[0] != e.field) {                                               \
        if (error) { std::ostringstream o; o << #field " mismatch: got "       \
            << g.field[0] << ", expected " << e.field; *error = o.str(); }      \
        return false; }
#define STM_CHECK_HASH(field)                                                  \
    if (g.field[0] != e.field) {                                               \
        if (error) { std::ostringstream o; o << #field " mismatch: got 0x"     \
            << std::hex << g.field[0] << ", expected 0x" << e.field;            \
            *error = o.str(); }                                                 \
        return false; }

    STM_CHECK_INT(txn_begun);
    STM_CHECK_INT(read_own_write);
    STM_CHECK_INT(read_shared);
    STM_CHECK_INT(write_staged);
    STM_CHECK_INT(validate_ok);
    STM_CHECK_INT(readonly_commits);
    STM_CHECK_INT(txn_prepared);
    STM_CHECK_INT(write_locks);
    STM_CHECK_INT(wait_locks);
    STM_CHECK_INT(commits_done);
    STM_CHECK_INT(location_writes);
    STM_CHECK_INT(non_tx_writes);
    STM_CHECK_INT(non_tx_stalls);
    STM_CHECK_INT(retry_suspended);
    STM_CHECK_INT(retry_immediate);
    STM_CHECK_INT(retry_watch_overflow);
    STM_CHECK_INT(wake_retry);
    STM_CHECK_INT(wake_lock);
    STM_CHECK_INT(aborts);
    STM_CHECK_INT(partial_unlocks);
    STM_CHECK_INT(commit_unlocks);
    STM_CHECK_INT(abort_unlocks);
    STM_CHECK_INT(invalid_count);
    STM_CHECK_HASH(stm_event_hash);
    STM_CHECK_HASH(read_result_hash);
    STM_CHECK_HASH(location_hash);
    STM_CHECK_HASH(txn_hash);
    STM_CHECK_HASH(queue_hash);
#undef STM_CHECK_INT
#undef STM_CHECK_HASH
    return true;
}

#endif  // STM_COMMIT_ARBITER_ORACLE_HPP_
