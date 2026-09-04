// file: mesi_directory_oracle.hpp

#ifndef MESI_DIRECTORY_ORACLE_HPP_
#define MESI_DIRECTORY_ORACLE_HPP_

#include "mesi_directory_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

/*
 * Host oracle: the canonical, exact MESI directory semantics. The reference
 * and naive CUDA kernels must reproduce every output bit-for-bit.
 *
 * EVENT TUPLE FIELD CONVENTIONS (the most specific deterministic reading of
 * T51; identical across all three implementations):
 *
 *   Each emitted event consumes one event_seq (the current value is stamped,
 *   then event_seq is post-incremented). Unused fields take their sentinel:
 *     core   -> UINT32_MAX, state -> 255, value -> INT64_MIN, txn -> UINT64_MAX,
 *     aux    -> 0.
 *
 *   LOAD_HIT          core, line, state(current), value(current cache value)
 *   LOAD_STALL_PENDING core, line
 *   CAPACITY_EVICT    core, line(victim), state(victim state), value(victim
 *                     cache value); aux = victim touch_seq
 *   DOWNGRADE_WRITEBACK supplier core, line, state=S(new), value=supplier value
 *   DOWNGRADE_CLEAN   supplier core, line, state=S(new), value=supplier value
 *   LOAD_MISS_SHARED  requester core, line, state=S, value(installed value)
 *   LOAD_MISS_EXCLUSIVE requester core, line, state=E, value(installed value)
 *   STORE_HIT_MODIFIED core, line, state=M, value(new value)
 *   STORE_HIT_EXCLUSIVE core, line, state=M(new), value(new value)
 *   STORE_STALL_PENDING core, line, value(attempted new value)
 *   INV_SEND          target core, line, txn, value(requester new value);
 *                     aux = requester core
 *   STORE_PENDING     requester core, line, txn, value(new value);
 *                     aux = target count
 *   DATA_SUPPLY_DIRTY target core, line, state=M(old), value(dirty value), txn
 *   DATA_SUPPLY_CLEAN target core, line, state=E(old), value(clean value), txn
 *   INV_ACK           target core, line, txn; aux = remaining target count
 *   STORE_COMMIT      requester core, line, state=M, value(new value), txn
 *   EVICT_STALL_PENDING core, line
 *   EVICT_MISS        core, line
 *   EVICT_WRITEBACK   core, line, state=M(old), value(written-back value)
 *   EVICT_CLEAN       core, line, state=E(old)
 *   EVICT_SHARED      core, line, state=S(old)
 *   FLUSH_WRITEBACK   owner core, line, state=S(new), value(written-back value)
 *   FLUSH_CLEAN       owner core, line, state=S(new)
 *   FLUSH_NOOP        line
 *   INVALID           op_index only (core/line as supplied if in range else
 *                     sentinel); kind=INVALID
 *
 * COUNTERS are cumulative across all steps (never reset between steps, only on
 * solution_reset). coh_event_hash and event_seq_out are also cumulative-by-
 * construction: coh_event_hash folds in step order from the persistent running
 * hash, and event_seq_out is the persistent event_seq.
 */

struct MesiHostInputsView {
    const int32_t* op;
    const int32_t* arg_core;
    const int32_t* arg_line;
    const int64_t* arg_value;
    const uint64_t* arg_txn;
};

struct MesiHostOutputsView {
    const int64_t* counts;
    const uint64_t* coh_event_hash;
    const uint64_t* cache_hash;
    const uint64_t* directory_hash;
    const uint64_t* pending_hash;
    const uint64_t* event_seq_out;
    const uint64_t* state_checksum;
};

struct MesiExpected {
    std::vector<int64_t> counts;
    uint64_t coh_event_hash = 0;
    uint64_t cache_hash = 0;
    uint64_t directory_hash = 0;
    uint64_t pending_hash = 0;
    uint64_t event_seq_out = 0;
    uint64_t state_checksum = 0;
};

static inline uint64_t mesi_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void mesi_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = mesi_oracle_fnv_byte(v, p[i]);
    *h = v;
}

struct MesiCacheEntry {
    int32_t line = -1;
    uint8_t state = MESI_STATE_NONE;
    int64_t value = 0;
    uint64_t touch_seq = 0;
};

struct MesiPending {
    bool active = false;
    uint64_t txn_id = 0;
    int32_t requester = -1;
    int32_t line = -1;
    int64_t new_value = 0;
    int64_t supplier_value = INT64_MIN;
    uint64_t start_seq = 0;
    std::set<int32_t> targets;
};

struct MesiOracleState {
    MesiProblemSpec spec{};

    uint64_t event_seq = 0;
    uint64_t touch_seq_next = 1;
    uint64_t txn_seq_next = 1;

    std::vector<int64_t> memory_value;            // [line]
    std::vector<MesiCacheEntry> cache;            // [core][cap]  (line<0 => empty)
    std::vector<int32_t> owner;                   // [line], -1 = none
    std::vector<std::set<int32_t>> sharers;       // [line]
    std::vector<MesiPending> pending;             // [line]

    std::vector<int64_t> counts;                  // cumulative

    uint64_t running_event_hash = 1469598103934665603ULL;

    int cap() const { return spec.cache_capacity_per_core; }

    void init(const MesiProblemSpec& s) {
        spec = s;
        memory_value.assign((size_t)spec.line_count, 0);
        cache.assign((size_t)spec.core_count * (size_t)spec.cache_capacity_per_core,
                     MesiCacheEntry{});
        owner.assign((size_t)spec.line_count, -1);
        sharers.assign((size_t)spec.line_count, std::set<int32_t>{});
        pending.assign((size_t)spec.line_count, MesiPending{});
        counts.assign(MESI_COUNT_FIELDS, 0);
        seed_memory();
        event_seq = 0;
        touch_seq_next = 1;
        txn_seq_next = 1;
        running_event_hash = 1469598103934665603ULL;
    }

    void seed_memory() {
        // Deterministic non-trivial initial memory so dirty/clean paths differ.
        for (int l = 0; l < spec.line_count; ++l) {
            memory_value[(size_t)l] = (int64_t)(1000 + l * 7);
        }
    }

    void reset() {
        for (auto& e : cache) e = MesiCacheEntry{};
        std::fill(owner.begin(), owner.end(), -1);
        for (auto& s : sharers) s.clear();
        for (auto& p : pending) p = MesiPending{};
        std::fill(counts.begin(), counts.end(), 0);
        seed_memory();
        event_seq = 0;
        touch_seq_next = 1;
        txn_seq_next = 1;
        running_event_hash = 1469598103934665603ULL;
    }

    // ---- cache helpers ----
    MesiCacheEntry* find_entry(int core, int line) {
        const int base = core * cap();
        for (int s = 0; s < cap(); ++s) {
            MesiCacheEntry& e = cache[(size_t)base + (size_t)s];
            if (e.line == line && e.state != MESI_STATE_NONE) return &e;
        }
        return nullptr;
    }

    int free_slot(int core) {
        const int base = core * cap();
        for (int s = 0; s < cap(); ++s) {
            if (cache[(size_t)base + (size_t)s].state == MESI_STATE_NONE) return base + s;
        }
        return -1;
    }

    int count_used(int core) {
        const int base = core * cap();
        int n = 0;
        for (int s = 0; s < cap(); ++s) {
            if (cache[(size_t)base + (size_t)s].state != MESI_STATE_NONE) ++n;
        }
        return n;
    }

    int active_pending_count() {
        int n = 0;
        for (int l = 0; l < spec.line_count; ++l) if (pending[(size_t)l].active) ++n;
        return n;
    }

    // ---- event emission ----
    void emit(uint8_t kind, uint32_t op_index, uint32_t core, uint64_t line,
              uint8_t state, int64_t value, uint64_t txn, uint64_t aux) {
        uint64_t h = running_event_hash;
        uint64_t seq = event_seq;
        mesi_oracle_fnv_bytes(&h, &kind, sizeof(uint8_t));
        mesi_oracle_fnv_bytes(&h, &seq, sizeof(uint64_t));
        mesi_oracle_fnv_bytes(&h, &op_index, sizeof(uint32_t));
        mesi_oracle_fnv_bytes(&h, &core, sizeof(uint32_t));
        mesi_oracle_fnv_bytes(&h, &line, sizeof(uint64_t));
        mesi_oracle_fnv_bytes(&h, &state, sizeof(uint8_t));
        mesi_oracle_fnv_bytes(&h, &value, sizeof(int64_t));
        mesi_oracle_fnv_bytes(&h, &txn, sizeof(uint64_t));
        mesi_oracle_fnv_bytes(&h, &aux, sizeof(uint64_t));
        running_event_hash = h;
        event_seq += 1;  // wraps mod 2^64 naturally
    }

    static constexpr uint32_t NO_CORE = 0xFFFFFFFFu;
    static constexpr uint8_t NO_STATE = 255;
    static constexpr int64_t NO_VAL = INT64_MIN;
    static constexpr uint64_t NO_TXN = 0xFFFFFFFFFFFFFFFFULL;

    // Deterministic capacity eviction. Victim: smallest touch_seq, tie smallest
    // line. Emits CAPACITY_EVICT, applies EVICT effects. Returns nothing.
    void capacity_evict(uint32_t op_index, int core) {
        const int base = core * cap();
        int victim = -1;
        uint64_t best_touch = 0;
        int32_t best_line = 0;
        for (int s = 0; s < cap(); ++s) {
            MesiCacheEntry& e = cache[(size_t)base + (size_t)s];
            if (e.state == MESI_STATE_NONE) continue;
            if (victim < 0 || e.touch_seq < best_touch ||
                (e.touch_seq == best_touch && e.line < best_line)) {
                victim = s;
                best_touch = e.touch_seq;
                best_line = e.line;
            }
        }
        if (victim < 0) return;  // cache not full; nothing to evict
        MesiCacheEntry& e = cache[(size_t)base + (size_t)victim];
        const int32_t vline = e.line;
        const uint8_t vstate = e.state;
        const int64_t vvalue = e.value;
        const uint64_t vtouch = e.touch_seq;

        // EVICT effects mirror EVICT op, but event kind is CAPACITY_EVICT.
        if (vstate == MESI_STATE_M) {
            memory_value[(size_t)vline] = vvalue;
            if (owner[(size_t)vline] == core) owner[(size_t)vline] = -1;
        } else if (vstate == MESI_STATE_E) {
            if (owner[(size_t)vline] == core) owner[(size_t)vline] = -1;
        } else {  // S
            sharers[(size_t)vline].erase(core);
        }
        e = MesiCacheEntry{};  // remove cache entry
        counts[MC_capacity_evictions] += 1;
        emit(EV_CAPACITY_EVICT, op_index, (uint32_t)core, (uint64_t)vline,
             vstate, vvalue, NO_TXN, vtouch);
    }

    void install_entry(int core, int32_t line, uint8_t state, int64_t value) {
        // Update existing or use free slot. Caller guarantees a slot exists.
        MesiCacheEntry* ex = find_entry(core, line);
        if (ex) {
            ex->state = state;
            ex->value = value;
            ex->touch_seq = touch_seq_next++;
            return;
        }
        int slot = free_slot(core);
        MesiCacheEntry& e = cache[(size_t)slot];
        e.line = line;
        e.state = state;
        e.value = value;
        e.touch_seq = touch_seq_next++;
    }

    // ---- operations ----
    void do_load(uint32_t op_index, int core, int line) {
        if (core < 0 || core >= spec.core_count || line < 0 || line >= spec.line_count) {
            counts[MC_invalid_count] += 1;
            emit(EV_INVALID, op_index,
                 (core < 0 || core >= spec.core_count) ? NO_CORE : (uint32_t)core,
                 (line < 0 || line >= spec.line_count) ? 0xFFFFFFFFFFFFFFFFULL : (uint64_t)line,
                 NO_STATE, NO_VAL, NO_TXN, 0);
            return;
        }
        if (pending[(size_t)line].active) {
            counts[MC_load_stall_pending] += 1;
            emit(EV_LOAD_STALL_PENDING, op_index, (uint32_t)core, (uint64_t)line,
                 NO_STATE, NO_VAL, NO_TXN, 0);
            return;
        }
        MesiCacheEntry* ce = find_entry(core, line);
        if (ce) {
            ce->touch_seq = touch_seq_next++;
            counts[MC_load_hit] += 1;
            emit(EV_LOAD_HIT, op_index, (uint32_t)core, (uint64_t)line,
                 ce->state, ce->value, NO_TXN, 0);
            return;
        }
        // miss; evict first if full
        if (count_used(core) >= cap()) capacity_evict(op_index, core);

        const int own = owner[(size_t)line];
        if (own >= 0) {
            MesiCacheEntry* oe = find_entry(own, line);
            // owner must hold M or E
            if (oe && oe->state == MESI_STATE_M) {
                memory_value[(size_t)line] = oe->value;
                const int64_t supplier_val = oe->value;
                oe->state = MESI_STATE_S;  // M -> S
                // directory: owner none, sharers = {prev owner, requester}
                owner[(size_t)line] = -1;
                sharers[(size_t)line].insert(own);
                sharers[(size_t)line].insert(core);
                install_entry(core, line, MESI_STATE_S, supplier_val);
                counts[MC_downgrade_writeback] += 1;
                emit(EV_DOWNGRADE_WRITEBACK, op_index, (uint32_t)own, (uint64_t)line,
                     MESI_STATE_S, supplier_val, NO_TXN, 0);
                counts[MC_load_miss_shared] += 1;
                emit(EV_LOAD_MISS_SHARED, op_index, (uint32_t)core, (uint64_t)line,
                     MESI_STATE_S, supplier_val, NO_TXN, 0);
                return;
            } else if (oe && oe->state == MESI_STATE_E) {
                const int64_t supplier_val = oe->value;
                oe->state = MESI_STATE_S;  // E -> S
                owner[(size_t)line] = -1;
                sharers[(size_t)line].insert(own);
                sharers[(size_t)line].insert(core);
                install_entry(core, line, MESI_STATE_S, supplier_val);
                counts[MC_downgrade_clean] += 1;
                emit(EV_DOWNGRADE_CLEAN, op_index, (uint32_t)own, (uint64_t)line,
                     MESI_STATE_S, supplier_val, NO_TXN, 0);
                counts[MC_load_miss_shared] += 1;
                emit(EV_LOAD_MISS_SHARED, op_index, (uint32_t)core, (uint64_t)line,
                     MESI_STATE_S, supplier_val, NO_TXN, 0);
                return;
            }
            // owner recorded but no cache entry: should not happen; fall through
        }
        if (!sharers[(size_t)line].empty()) {
            const int64_t val = memory_value[(size_t)line];
            install_entry(core, line, MESI_STATE_S, val);
            sharers[(size_t)line].insert(core);
            counts[MC_load_miss_shared] += 1;
            emit(EV_LOAD_MISS_SHARED, op_index, (uint32_t)core, (uint64_t)line,
                 MESI_STATE_S, val, NO_TXN, 0);
            return;
        }
        // no owner, no sharers -> exclusive
        const int64_t val = memory_value[(size_t)line];
        install_entry(core, line, MESI_STATE_E, val);
        owner[(size_t)line] = core;
        counts[MC_load_miss_exclusive] += 1;
        emit(EV_LOAD_MISS_EXCLUSIVE, op_index, (uint32_t)core, (uint64_t)line,
             MESI_STATE_E, val, NO_TXN, 0);
    }

    void do_store(uint32_t op_index, int core, int line, int64_t value) {
        if (core < 0 || core >= spec.core_count || line < 0 || line >= spec.line_count) {
            counts[MC_invalid_count] += 1;
            emit(EV_INVALID, op_index,
                 (core < 0 || core >= spec.core_count) ? NO_CORE : (uint32_t)core,
                 (line < 0 || line >= spec.line_count) ? 0xFFFFFFFFFFFFFFFFULL : (uint64_t)line,
                 NO_STATE, NO_VAL, NO_TXN, 0);
            return;
        }
        if (pending[(size_t)line].active) {
            counts[MC_store_stall_pending] += 1;
            emit(EV_STORE_STALL_PENDING, op_index, (uint32_t)core, (uint64_t)line,
                 NO_STATE, value, NO_TXN, 0);
            return;
        }
        MesiCacheEntry* ce = find_entry(core, line);
        if (ce && ce->state == MESI_STATE_M) {
            ce->value = value;
            ce->touch_seq = touch_seq_next++;
            counts[MC_store_hit_modified] += 1;
            emit(EV_STORE_HIT_MODIFIED, op_index, (uint32_t)core, (uint64_t)line,
                 MESI_STATE_M, value, NO_TXN, 0);
            return;
        }
        if (ce && ce->state == MESI_STATE_E) {
            ce->state = MESI_STATE_M;
            ce->value = value;
            ce->touch_seq = touch_seq_next++;
            owner[(size_t)line] = core;  // remains core
            counts[MC_store_hit_exclusive] += 1;
            emit(EV_STORE_HIT_EXCLUSIVE, op_index, (uint32_t)core, (uint64_t)line,
                 MESI_STATE_M, value, NO_TXN, 0);
            return;
        }
        // core holds S or absent. Build invalidation targets.
        std::set<int32_t> targets;
        for (int32_t s : sharers[(size_t)line]) if (s != core) targets.insert(s);
        if (owner[(size_t)line] >= 0 && owner[(size_t)line] != core) {
            targets.insert(owner[(size_t)line]);
        }

        if (targets.empty()) {
            if (!find_entry(core, line) && count_used(core) >= cap()) {
                capacity_evict(op_index, core);
            }
            install_entry(core, line, MESI_STATE_M, value);
            sharers[(size_t)line].erase(core);
            owner[(size_t)line] = core;
            // sharer set empty (only core could have been there)
            sharers[(size_t)line].clear();
            counts[MC_store_committed] += 1;
            emit(EV_STORE_COMMIT, op_index, (uint32_t)core, (uint64_t)line,
                 MESI_STATE_M, value, NO_TXN, 0);
            return;
        }

        // target set nonempty -> pending transaction needed
        if (active_pending_count() >= spec.max_pending_lines) {
            counts[MC_store_stall_pending] += 1;
            emit(EV_STORE_STALL_PENDING, op_index, (uint32_t)core, (uint64_t)line,
                 NO_STATE, value, NO_TXN, 0);
            return;
        }
        MesiPending& p = pending[(size_t)line];
        p.active = true;
        p.txn_id = txn_seq_next++;
        p.requester = core;
        p.line = line;
        p.new_value = value;
        p.supplier_value = INT64_MIN;
        p.start_seq = event_seq;  // seq at which the txn begins (before INV_SEND)
        p.targets = targets;

        counts[MC_store_pending] += 1;
        for (int32_t t : targets) {  // std::set iterates ascending
            counts[MC_inv_sent] += 1;
            emit(EV_INV_SEND, op_index, (uint32_t)t, (uint64_t)line,
                 MESI_STATE_NONE, value, p.txn_id, (uint64_t)core);
        }
        emit(EV_STORE_PENDING, op_index, (uint32_t)core, (uint64_t)line,
             MESI_STATE_NONE, value, p.txn_id, (uint64_t)targets.size());
    }

    void do_ack(uint32_t op_index, int core, int line, uint64_t txn) {
        bool valid = true;
        if (core < 0 || core >= spec.core_count || line < 0 || line >= spec.line_count) {
            valid = false;
        } else if (!pending[(size_t)line].active) {
            valid = false;
        } else if (pending[(size_t)line].txn_id != txn) {
            valid = false;
        } else if (pending[(size_t)line].targets.find(core) ==
                   pending[(size_t)line].targets.end()) {
            valid = false;
        }
        if (!valid) {
            counts[MC_invalid_count] += 1;
            emit(EV_INVALID, op_index,
                 (core < 0 || core >= spec.core_count) ? NO_CORE : (uint32_t)core,
                 (line < 0 || line >= spec.line_count) ? 0xFFFFFFFFFFFFFFFFULL : (uint64_t)line,
                 NO_STATE, NO_VAL, txn, 0);
            return;
        }
        MesiPending& p = pending[(size_t)line];
        MesiCacheEntry* te = find_entry(core, line);
        if (te) {
            if (te->state == MESI_STATE_M) {
                p.supplier_value = te->value;
                memory_value[(size_t)line] = te->value;
                emit(EV_DATA_SUPPLY_DIRTY, op_index, (uint32_t)core, (uint64_t)line,
                     MESI_STATE_M, te->value, p.txn_id, 0);
                counts[MC_data_supply_dirty] += 1;
            } else if (te->state == MESI_STATE_E) {
                p.supplier_value = te->value;
                emit(EV_DATA_SUPPLY_CLEAN, op_index, (uint32_t)core, (uint64_t)line,
                     MESI_STATE_E, te->value, p.txn_id, 0);
                counts[MC_data_supply_clean] += 1;
            }
            *te = MesiCacheEntry{};  // remove cache entry
        }
        sharers[(size_t)line].erase(core);
        if (owner[(size_t)line] == core) owner[(size_t)line] = -1;

        p.targets.erase(core);
        counts[MC_inv_acked] += 1;
        emit(EV_INV_ACK, op_index, (uint32_t)core, (uint64_t)line,
             NO_STATE, NO_VAL, p.txn_id, (uint64_t)p.targets.size());

        if (p.targets.empty()) {
            const int req = p.requester;
            const int64_t nv = p.new_value;
            const uint64_t tid = p.txn_id;
            if (!find_entry(req, line) && count_used(req) >= cap()) {
                capacity_evict(op_index, req);
            }
            install_entry(req, line, MESI_STATE_M, nv);
            sharers[(size_t)line].erase(req);
            sharers[(size_t)line].clear();
            owner[(size_t)line] = req;
            counts[MC_store_committed] += 1;
            emit(EV_STORE_COMMIT, op_index, (uint32_t)req, (uint64_t)line,
                 MESI_STATE_M, nv, tid, 0);
            p = MesiPending{};  // delete pending transaction
        }
    }

    void do_evict(uint32_t op_index, int core, int line) {
        if (core < 0 || core >= spec.core_count || line < 0 || line >= spec.line_count) {
            counts[MC_invalid_count] += 1;
            emit(EV_INVALID, op_index,
                 (core < 0 || core >= spec.core_count) ? NO_CORE : (uint32_t)core,
                 (line < 0 || line >= spec.line_count) ? 0xFFFFFFFFFFFFFFFFULL : (uint64_t)line,
                 NO_STATE, NO_VAL, NO_TXN, 0);
            return;
        }
        if (pending[(size_t)line].active &&
            pending[(size_t)line].targets.find(core) !=
                pending[(size_t)line].targets.end()) {
            counts[MC_evict_stall_pending] += 1;
            emit(EV_EVICT_STALL_PENDING, op_index, (uint32_t)core, (uint64_t)line,
                 NO_STATE, NO_VAL, NO_TXN, 0);
            return;
        }
        MesiCacheEntry* ce = find_entry(core, line);
        if (!ce) {
            counts[MC_evict_miss] += 1;
            emit(EV_EVICT_MISS, op_index, (uint32_t)core, (uint64_t)line,
                 NO_STATE, NO_VAL, NO_TXN, 0);
            return;
        }
        if (ce->state == MESI_STATE_M) {
            memory_value[(size_t)line] = ce->value;
            const int64_t wb = ce->value;
            if (owner[(size_t)line] == core) owner[(size_t)line] = -1;
            *ce = MesiCacheEntry{};
            counts[MC_evict_writeback] += 1;
            emit(EV_EVICT_WRITEBACK, op_index, (uint32_t)core, (uint64_t)line,
                 MESI_STATE_M, wb, NO_TXN, 0);
        } else if (ce->state == MESI_STATE_E) {
            if (owner[(size_t)line] == core) owner[(size_t)line] = -1;
            *ce = MesiCacheEntry{};
            counts[MC_evict_clean] += 1;
            emit(EV_EVICT_CLEAN, op_index, (uint32_t)core, (uint64_t)line,
                 MESI_STATE_E, NO_VAL, NO_TXN, 0);
        } else {  // S
            sharers[(size_t)line].erase(core);
            *ce = MesiCacheEntry{};
            counts[MC_evict_shared] += 1;
            emit(EV_EVICT_SHARED, op_index, (uint32_t)core, (uint64_t)line,
                 MESI_STATE_S, NO_VAL, NO_TXN, 0);
        }
    }

    void do_flush(uint32_t op_index, int line) {
        if (line < 0 || line >= spec.line_count || pending[(size_t)line].active) {
            counts[MC_invalid_count] += 1;
            emit(EV_INVALID, op_index, NO_CORE,
                 (line < 0 || line >= spec.line_count) ? 0xFFFFFFFFFFFFFFFFULL : (uint64_t)line,
                 NO_STATE, NO_VAL, NO_TXN, 0);
            return;
        }
        const int own = owner[(size_t)line];
        if (own >= 0) {
            MesiCacheEntry* oe = find_entry(own, line);
            if (oe && oe->state == MESI_STATE_M) {
                memory_value[(size_t)line] = oe->value;
                const int64_t wb = oe->value;
                oe->state = MESI_STATE_S;
                owner[(size_t)line] = -1;
                sharers[(size_t)line].insert(own);
                counts[MC_flush_writeback] += 1;
                emit(EV_FLUSH_WRITEBACK, op_index, (uint32_t)own, (uint64_t)line,
                     MESI_STATE_S, wb, NO_TXN, 0);
                return;
            } else if (oe && oe->state == MESI_STATE_E) {
                oe->state = MESI_STATE_S;
                owner[(size_t)line] = -1;
                sharers[(size_t)line].insert(own);
                counts[MC_flush_clean] += 1;
                emit(EV_FLUSH_CLEAN, op_index, (uint32_t)own, (uint64_t)line,
                     MESI_STATE_S, NO_VAL, NO_TXN, 0);
                return;
            }
        }
        counts[MC_flush_noop] += 1;
        emit(EV_FLUSH_NOOP, op_index, NO_CORE, (uint64_t)line,
             NO_STATE, NO_VAL, NO_TXN, 0);
    }

    // ---- checksums ----
    uint64_t cache_hash_compute() const {
        uint64_t h = 1469598103934665603ULL;
        for (int c = 0; c < spec.core_count; ++c) {
            // gather entries, sort by line ascending
            std::vector<const MesiCacheEntry*> ents;
            const int base = c * cap();
            for (int s = 0; s < cap(); ++s) {
                const MesiCacheEntry& e = cache[(size_t)base + (size_t)s];
                if (e.state != MESI_STATE_NONE) ents.push_back(&e);
            }
            std::sort(ents.begin(), ents.end(),
                      [](const MesiCacheEntry* a, const MesiCacheEntry* b) {
                          return a->line < b->line;
                      });
            for (const MesiCacheEntry* e : ents) {
                uint32_t core = (uint32_t)c;
                uint64_t line = (uint64_t)e->line;
                uint8_t state = e->state;
                int64_t value = e->value;
                uint64_t touch = e->touch_seq;
                mesi_oracle_fnv_bytes(&h, &core, sizeof(uint32_t));
                mesi_oracle_fnv_bytes(&h, &line, sizeof(uint64_t));
                mesi_oracle_fnv_bytes(&h, &state, sizeof(uint8_t));
                mesi_oracle_fnv_bytes(&h, &value, sizeof(int64_t));
                mesi_oracle_fnv_bytes(&h, &touch, sizeof(uint64_t));
            }
        }
        return h;
    }

    uint64_t directory_hash_compute() const {
        uint64_t h = 1469598103934665603ULL;
        for (int l = 0; l < spec.line_count; ++l) {
            uint64_t line = (uint64_t)l;
            int64_t mv = memory_value[(size_t)l];
            uint32_t own = (owner[(size_t)l] < 0) ? 0xFFFFFFFFu : (uint32_t)owner[(size_t)l];
            uint64_t sc = (uint64_t)sharers[(size_t)l].size();
            mesi_oracle_fnv_bytes(&h, &line, sizeof(uint64_t));
            mesi_oracle_fnv_bytes(&h, &mv, sizeof(int64_t));
            mesi_oracle_fnv_bytes(&h, &own, sizeof(uint32_t));
            mesi_oracle_fnv_bytes(&h, &sc, sizeof(uint64_t));
            for (int32_t s : sharers[(size_t)l]) {
                uint32_t su = (uint32_t)s;
                mesi_oracle_fnv_bytes(&h, &su, sizeof(uint32_t));
            }
        }
        return h;
    }

    uint64_t pending_hash_compute() const {
        uint64_t h = 1469598103934665603ULL;
        for (int l = 0; l < spec.line_count; ++l) {
            const MesiPending& p = pending[(size_t)l];
            if (!p.active) continue;
            uint64_t line = (uint64_t)l;
            uint64_t tid = p.txn_id;
            uint32_t req = (uint32_t)p.requester;
            int64_t nv = p.new_value;
            int64_t sv = p.supplier_value;
            uint64_t ss = p.start_seq;
            mesi_oracle_fnv_bytes(&h, &line, sizeof(uint64_t));
            mesi_oracle_fnv_bytes(&h, &tid, sizeof(uint64_t));
            mesi_oracle_fnv_bytes(&h, &req, sizeof(uint32_t));
            mesi_oracle_fnv_bytes(&h, &nv, sizeof(int64_t));
            mesi_oracle_fnv_bytes(&h, &sv, sizeof(int64_t));
            mesi_oracle_fnv_bytes(&h, &ss, sizeof(uint64_t));
            for (int32_t t : p.targets) {
                uint32_t tu = (uint32_t)t;
                mesi_oracle_fnv_bytes(&h, &tu, sizeof(uint32_t));
            }
        }
        return h;
    }

    uint64_t state_checksum_compute() const {
        uint64_t h = 1469598103934665603ULL;
        mesi_oracle_fnv_bytes(&h, &spec.core_count, sizeof(int32_t));
        mesi_oracle_fnv_bytes(&h, &spec.line_count, sizeof(int32_t));
        mesi_oracle_fnv_bytes(&h, &spec.cache_capacity_per_core, sizeof(int32_t));
        mesi_oracle_fnv_bytes(&h, &spec.max_pending_lines, sizeof(int32_t));
        mesi_oracle_fnv_bytes(&h, &event_seq, sizeof(uint64_t));
        mesi_oracle_fnv_bytes(&h, &touch_seq_next, sizeof(uint64_t));
        mesi_oracle_fnv_bytes(&h, &txn_seq_next, sizeof(uint64_t));
        uint64_t ch = cache_hash_compute();
        uint64_t dh = directory_hash_compute();
        uint64_t ph = pending_hash_compute();
        mesi_oracle_fnv_bytes(&h, &ch, sizeof(uint64_t));
        mesi_oracle_fnv_bytes(&h, &dh, sizeof(uint64_t));
        mesi_oracle_fnv_bytes(&h, &ph, sizeof(uint64_t));
        for (int i = 0; i < MESI_COUNT_FIELDS; ++i) {
            int64_t c = counts[(size_t)i];
            mesi_oracle_fnv_bytes(&h, &c, sizeof(int64_t));
        }
        return h;
    }

    void step_once(const MesiRunSpec& run, const MesiHostInputsView& in,
                   MesiExpected* expected) {
        for (int i = 0; i < run.batch_size; ++i) {
            const int op = in.op[i];
            const int core = in.arg_core[i];
            const int line = in.arg_line[i];
            const int64_t value = in.arg_value[i];
            const uint64_t txn = in.arg_txn[i];

            if (op == MESI_OP_LOAD) {
                do_load((uint32_t)i, core, line);
            } else if (op == MESI_OP_STORE) {
                do_store((uint32_t)i, core, line, value);
            } else if (op == MESI_OP_ACK_INV) {
                do_ack((uint32_t)i, core, line, txn);
            } else if (op == MESI_OP_EVICT) {
                do_evict((uint32_t)i, core, line);
            } else if (op == MESI_OP_FLUSH) {
                do_flush((uint32_t)i, line);
            } else {
                counts[MC_invalid_count] += 1;
                emit(EV_INVALID, (uint32_t)i, NO_CORE,
                     0xFFFFFFFFFFFFFFFFULL, NO_STATE, NO_VAL, NO_TXN, 0);
            }
        }

        expected->counts = counts;
        expected->coh_event_hash = running_event_hash;
        expected->cache_hash = cache_hash_compute();
        expected->directory_hash = directory_hash_compute();
        expected->pending_hash = pending_hash_compute();
        expected->event_seq_out = event_seq;
        expected->state_checksum = state_checksum_compute();
    }
};

static inline bool mesi_check_all_outputs(
    const MesiProblemSpec& spec,
    const MesiExpected& expected,
    const MesiHostOutputsView& got,
    std::string* error) {
    (void)spec;
    for (int i = 0; i < MESI_COUNT_FIELDS; ++i) {
        if (got.counts[i] != expected.counts[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "counts[" << i << "] mismatch: got " << got.counts[i]
                    << ", expected " << expected.counts[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }
    struct HashField { const char* name; uint64_t got; uint64_t exp; };
    HashField fields[] = {
        {"coh_event_hash", got.coh_event_hash[0], expected.coh_event_hash},
        {"cache_hash", got.cache_hash[0], expected.cache_hash},
        {"directory_hash", got.directory_hash[0], expected.directory_hash},
        {"pending_hash", got.pending_hash[0], expected.pending_hash},
        {"event_seq_out", got.event_seq_out[0], expected.event_seq_out},
        {"state_checksum", got.state_checksum[0], expected.state_checksum},
    };
    for (const HashField& f : fields) {
        if (f.got != f.exp) {
            if (error) {
                std::ostringstream oss;
                oss << f.name << " mismatch: got 0x" << std::hex << f.got
                    << ", expected 0x" << f.exp;
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

/*
GRADER MODEL
  oracle.init(problem_spec)
  solution_reset + oracle.reset()
  for each step:
    solution_run(...)
    oracle.step_once(...)
    mesi_check_all_outputs(...)

Required adversarial coverage:
  - pending store with multiple dirty/clean/shared targets, partial acks
  - load/store/flush stalls against pending lines, valid acks proceed
  - dirty owner downgrade on read miss (memory updated before share)
  - capacity eviction during miss install and during store commit
  - invalid ops (bad core/line/txn, ack from non-target, double ack)
  - reset and exact replay; pending-table-full stalls
*/

#endif  // MESI_DIRECTORY_ORACLE_HPP_
