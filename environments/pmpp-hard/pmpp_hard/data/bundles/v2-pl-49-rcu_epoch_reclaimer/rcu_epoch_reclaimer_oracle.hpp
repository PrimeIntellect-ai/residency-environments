// file: rcu_epoch_reclaimer_oracle.hpp
//
// Host reference model = canonical semantics for T49 (RCU Epoch Reclaimer).
// The device reference.cu and naive.cu must reproduce these exact outputs.
//
// DETERMINISTIC INTERPRETATIONS (ambiguities resolved to the most specific
// deterministic reading of designs/T49.md):
//
//  E1. event_seq starts at 0; for every emitted event we first do
//      event_seq += 1 (mod 2^64) and that new value is the event's event_seq.
//      So the first event of a run has event_seq == 1.
//  E2. "object table is full" means (#present objects) == max_objects, where
//      present counts both live and retired-but-unreclaimed objects.
//  E3. Per-event hash fields. For each event we fill the 9-tuple
//      (event_kind, event_seq, op_index, thread_or_MAX, obj_id_or_MAX,
//       root_or_MAX, slot_or_MAX, epoch, aux_u64). Fields not naturally
//      produced by an event use their sentinel (UINT32_MAX / UINT64_MAX) and
//      epoch defaults to current global_epoch unless the event is about a
//      specific epoch. The exact per-event mapping is in emit_* helpers below
//      and is part of the contract.
//  E4. op_index is the global operation index across the whole run, starting
//      at 0 for the first op of the first solution_run and continuing to grow
//      across solution_run calls (it is persistent run-state, like event_seq).
//  E5. INVALID events carry aux_u64 = op.kind so the offending opcode is
//      distinguishable; obj/root/slot/thread sentinels otherwise.
//  E6. ALLOC: aux_u64 = request_id (arg_a). obj_id_or_MAX = chosen id (OK) or
//      UINT64_MAX (OOM). epoch = global_epoch.
//  E7. READ_NODE aux_u64 = position (0-based hop index along the chain).
//      obj_id = the node id, epoch = thread.active_epoch of the reader.
//  E8. READ_EMPTY: obj_id = UINT64_MAX, root set, thread set, aux = 0.
//  E9. RECLAIM emits RCU_CALLBACK then OBJECT_RECLAIM for each reclaimed
//      object; RCU_CALLBACK aux = callback_tag, OBJECT_RECLAIM aux =
//      reclaim_seq assigned (or UINT64_MAX if the free id was dropped).
//      FREE_ID_DROPPED (from full free list during reclaim) is emitted
//      between RCU_CALLBACK and OBJECT_RECLAIM with aux = obj_id.
// E10. FORCE_DROP_FREE_IDS emits one FREE_ID_DROPPED per removed entry,
//      aux = obj_id, obj_id field = obj_id, in free-list order.
// E11. callbacks_ready counts RCU_CALLBACK events (one per reclaimed object).
// E12. ADVANCE_EPOCH event epoch field = the NEW global_epoch.

#ifndef RCU_EPOCH_RECLAIMER_ORACLE_HPP_
#define RCU_EPOCH_RECLAIMER_ORACLE_HPP_

#include "rcu_epoch_reclaimer_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static inline uint64_t rcu_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void rcu_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = rcu_oracle_fnv_byte(v, p[i]);
    *h = v;
}

static inline void rcu_fnv_u8(uint64_t* h, uint8_t v)  { rcu_oracle_fnv_bytes(h, &v, 1); }
static inline void rcu_fnv_u32(uint64_t* h, uint32_t v){ rcu_oracle_fnv_bytes(h, &v, 4); }
static inline void rcu_fnv_u64(uint64_t* h, uint64_t v){ rcu_oracle_fnv_bytes(h, &v, 8); }
static inline void rcu_fnv_i64(uint64_t* h, int64_t v) { rcu_oracle_fnv_bytes(h, &v, 8); }

struct RcuExpected {
    uint64_t counts[RCU_COUNT_TOTAL] = {0};
    uint64_t event_hash = 0;
    uint64_t read_hash = 0;
    uint64_t root_hash = 0;
    uint64_t object_hash = 0;
    uint64_t thread_hash = 0;
    uint64_t free_hash = 0;
    uint64_t state_scalars[6] = {0};
};

struct RcuOracleObject {
    bool present = false;
    int64_t value = 0;
    uint64_t alloc_seq = 0;
    uint8_t retired = 0;
    uint64_t retire_seq = UINT64_MAX;
    uint64_t retire_epoch = UINT64_MAX;
    uint32_t callback_tag = 0;
    std::vector<uint64_t> edges;  // size = max_edges_per_object
};

struct RcuOracleThread {
    uint64_t read_nest = 0;
    uint64_t outer_read_seq = UINT64_MAX;
    uint64_t active_epoch = 0;
    std::vector<uint64_t> root_view;  // size = root_count
};

struct RcuOracleFreeEntry {
    uint64_t obj_id;
    uint64_t reclaim_seq;
};

struct RcuOracleState {
    RcuProblemSpec spec{};
    int thread_count = 0;
    int root_count = 0;
    int max_edges = 0;

    uint64_t event_seq = 0;
    uint64_t obj_id_next = 1;
    uint64_t read_seq_next = 1;
    uint64_t retire_seq_next = 1;
    uint64_t reclaim_seq_next = 1;
    uint64_t global_epoch = 0;
    uint64_t op_index_global = 0;

    std::vector<uint64_t> root;                  // root_count
    std::vector<RcuOracleThread> threads;        // thread_count
    std::map<uint64_t, RcuOracleObject> objects; // keyed by obj id (present ones)
    std::vector<uint64_t> retired_queue;         // obj ids, retired=1
    std::vector<RcuOracleFreeEntry> free_list;   // canonical order

    // accumulators carried across solution_run calls within one scenario.
    uint64_t counts[RCU_COUNT_TOTAL] = {0};
    uint64_t event_hash = 1469598103934665603ULL;
    uint64_t read_hash = 1469598103934665603ULL;

    void init(const RcuProblemSpec& s) {
        spec = s;
        thread_count = s.thread_count;
        root_count = s.root_count;
        max_edges = s.max_edges_per_object;
        full_reset();
    }

    void full_reset() {
        event_seq = 0;
        obj_id_next = 1;
        read_seq_next = 1;
        retire_seq_next = 1;
        reclaim_seq_next = 1;
        global_epoch = 0;
        op_index_global = 0;

        root.assign((size_t)root_count, 0);
        threads.assign((size_t)thread_count, RcuOracleThread{});
        for (auto& t : threads) {
            t.read_nest = 0;
            t.outer_read_seq = UINT64_MAX;
            t.active_epoch = 0;
            t.root_view.assign((size_t)root_count, 0);
        }
        objects.clear();
        retired_queue.clear();
        free_list.clear();

        for (int i = 0; i < RCU_COUNT_TOTAL; ++i) counts[i] = 0;
        event_hash = 1469598103934665603ULL;
        read_hash = 1469598103934665603ULL;
    }

    int present_count() const { return (int)objects.size(); }

    bool object_present(uint64_t id) const {
        auto it = objects.find(id);
        return it != objects.end() && it->second.present;
    }
    bool object_live(uint64_t id) const {
        auto it = objects.find(id);
        return it != objects.end() && it->second.present && it->second.retired == 0;
    }

    // ---- event emission -------------------------------------------------
    void emit_event(uint8_t kind, uint32_t op_index, uint32_t thread_or_max,
                    uint64_t obj_or_max, uint32_t root_or_max, uint32_t slot_or_max,
                    uint64_t epoch, uint64_t aux) {
        event_seq += 1;  // E1
        uint64_t* h = &event_hash;
        rcu_fnv_u8(h, kind);
        rcu_fnv_u64(h, event_seq);
        rcu_fnv_u32(h, op_index);
        rcu_fnv_u32(h, thread_or_max);
        rcu_fnv_u64(h, obj_or_max);
        rcu_fnv_u32(h, root_or_max);
        rcu_fnv_u32(h, slot_or_max);
        rcu_fnv_u64(h, epoch);
        rcu_fnv_u64(h, aux);
    }

    void emit_invalid(uint32_t op_index, uint32_t opcode) {
        emit_event(RCU_EV_INVALID, op_index, UINT32_MAX, UINT64_MAX,
                   UINT32_MAX, UINT32_MAX, global_epoch, (uint64_t)opcode);
        counts[RCU_COUNT_INVALID] += 1;
    }

    // ---- retired queue canonical ordering -------------------------------
    void sort_retired_queue() {
        std::sort(retired_queue.begin(), retired_queue.end(),
                  [this](uint64_t a, uint64_t b) {
                      const RcuOracleObject& oa = objects.at(a);
                      const RcuOracleObject& ob = objects.at(b);
                      if (oa.retire_epoch != ob.retire_epoch)
                          return oa.retire_epoch < ob.retire_epoch;
                      if (oa.retire_seq != ob.retire_seq)
                          return oa.retire_seq < ob.retire_seq;
                      return a < b;
                  });
    }

    void sort_free_list() {
        std::sort(free_list.begin(), free_list.end(),
                  [](const RcuOracleFreeEntry& a, const RcuOracleFreeEntry& b) {
                      if (a.reclaim_seq != b.reclaim_seq)
                          return a.reclaim_seq < b.reclaim_seq;
                      return a.obj_id < b.obj_id;
                  });
    }

    // ---- operations -----------------------------------------------------
    void op_alloc(uint32_t op_index, const RcuOp& op) {
        const uint64_t request_id = (uint64_t)(uint32_t)op.arg_a;
        const int64_t value = op.arg_i64;

        const bool full = present_count() >= spec.max_objects;
        if (full && free_list.empty()) {
            emit_event(RCU_EV_ALLOC_OOM, op_index, UINT32_MAX, UINT64_MAX,
                       UINT32_MAX, UINT32_MAX, global_epoch, request_id);
            counts[RCU_COUNT_ALLOC_OOM] += 1;
            return;
        }

        uint64_t chosen;
        if (!free_list.empty()) {
            sort_free_list();
            chosen = free_list.front().obj_id;  // lowest (reclaim_seq, obj_id)
            free_list.erase(free_list.begin());
        } else {
            chosen = obj_id_next;
            obj_id_next += 1;
        }

        // alloc_seq = event_seq of the ALLOC_OK event.
        emit_event(RCU_EV_ALLOC_OK, op_index, UINT32_MAX, chosen,
                   UINT32_MAX, UINT32_MAX, global_epoch, request_id);
        const uint64_t alloc_seq = event_seq;

        RcuOracleObject o;
        o.present = true;
        o.value = value;
        o.alloc_seq = alloc_seq;
        o.retired = 0;
        o.retire_seq = UINT64_MAX;
        o.retire_epoch = UINT64_MAX;
        o.callback_tag = 0;
        o.edges.assign((size_t)max_edges, 0);
        objects[chosen] = o;

        counts[RCU_COUNT_ALLOC_OK] += 1;
    }

    void op_set_root(uint32_t op_index, const RcuOp& op) {
        const int32_t root_id = op.arg_a;
        const uint64_t obj_id = (uint64_t)(uint32_t)op.arg_b;

        if (root_id < 0 || root_id >= root_count) { emit_invalid(op_index, RCU_OP_SET_ROOT); return; }
        if (obj_id != 0 && !object_live(obj_id)) { emit_invalid(op_index, RCU_OP_SET_ROOT); return; }

        root[(size_t)root_id] = obj_id;
        emit_event(RCU_EV_ROOT_SET, op_index, UINT32_MAX, obj_id,
                   (uint32_t)root_id, UINT32_MAX, global_epoch, 0);
        counts[RCU_COUNT_ROOT_SETS] += 1;
    }

    void op_link(uint32_t op_index, const RcuOp& op) {
        const uint64_t src = (uint64_t)(uint32_t)op.arg_a;
        const int32_t slot = op.arg_b;
        const uint64_t dst = (uint64_t)(uint32_t)op.arg_c;

        if (!object_live(src)) { emit_invalid(op_index, RCU_OP_LINK); return; }
        if (slot < 0 || slot >= max_edges) { emit_invalid(op_index, RCU_OP_LINK); return; }
        if (dst != 0 && !object_live(dst)) { emit_invalid(op_index, RCU_OP_LINK); return; }

        objects[src].edges[(size_t)slot] = dst;
        emit_event(RCU_EV_EDGE_SET, op_index, UINT32_MAX, src,
                   UINT32_MAX, (uint32_t)slot, global_epoch, dst);
        counts[RCU_COUNT_EDGE_SETS] += 1;
    }

    void op_read_lock(uint32_t op_index, const RcuOp& op) {
        const int32_t t = op.arg_a;
        if (t < 0 || t >= thread_count) { emit_invalid(op_index, RCU_OP_READ_LOCK); return; }
        RcuOracleThread& th = threads[(size_t)t];

        if (th.read_nest == 0) {
            th.root_view = root;  // snapshot all roots in root id order
            th.outer_read_seq = read_seq_next;
            read_seq_next += 1;
            th.active_epoch = global_epoch;
        }

        // increment read_nest; wrap to zero is invalid.
        if (th.read_nest + 1 == 0) { emit_invalid(op_index, RCU_OP_READ_LOCK); return; }
        th.read_nest += 1;

        emit_event(RCU_EV_READ_LOCK_OK, op_index, (uint32_t)t, UINT64_MAX,
                   UINT32_MAX, UINT32_MAX, th.active_epoch, th.outer_read_seq);
        counts[RCU_COUNT_READ_LOCKS] += 1;
    }

    void op_read_unlock(uint32_t op_index, const RcuOp& op) {
        const int32_t t = op.arg_a;
        if (t < 0 || t >= thread_count) { emit_invalid(op_index, RCU_OP_READ_UNLOCK); return; }
        RcuOracleThread& th = threads[(size_t)t];
        if (th.read_nest == 0) { emit_invalid(op_index, RCU_OP_READ_UNLOCK); return; }

        th.read_nest -= 1;
        if (th.read_nest == 0) {
            th.outer_read_seq = UINT64_MAX;
            th.active_epoch = global_epoch;
            emit_event(RCU_EV_READ_UNLOCK_QUIESCENT, op_index, (uint32_t)t, UINT64_MAX,
                       UINT32_MAX, UINT32_MAX, th.active_epoch, 0);
        } else {
            emit_event(RCU_EV_READ_UNLOCK_NESTED, op_index, (uint32_t)t, UINT64_MAX,
                       UINT32_MAX, UINT32_MAX, th.active_epoch, th.read_nest);
        }
        counts[RCU_COUNT_READ_UNLOCKS] += 1;
    }

    void op_quiesce(uint32_t op_index, const RcuOp& op) {
        const int32_t t = op.arg_a;
        if (t < 0 || t >= thread_count) { emit_invalid(op_index, RCU_OP_QUIESCE); return; }
        RcuOracleThread& th = threads[(size_t)t];
        if (th.read_nest != 0) { emit_invalid(op_index, RCU_OP_QUIESCE); return; }

        th.active_epoch = global_epoch;
        emit_event(RCU_EV_QUIESCE_OK, op_index, (uint32_t)t, UINT64_MAX,
                   UINT32_MAX, UINT32_MAX, th.active_epoch, 0);
        counts[RCU_COUNT_QUIESCES] += 1;
    }

    void op_advance_epoch(uint32_t op_index, const RcuOp& /*op*/) {
        global_epoch += 1;  // mod 2^64
        emit_event(RCU_EV_EPOCH_ADVANCE, op_index, UINT32_MAX, UINT64_MAX,
                   UINT32_MAX, UINT32_MAX, global_epoch, 0);  // E12: new epoch
        counts[RCU_COUNT_EPOCH_ADVANCES] += 1;
    }

    void emit_read_record(uint32_t op_index, uint32_t thread, uint32_t root_id,
                          uint64_t position, uint64_t obj_or_zero, int64_t value_or_min,
                          uint8_t retired, uint64_t alloc_or_max, uint64_t retire_or_max) {
        uint64_t* h = &read_hash;
        rcu_fnv_u32(h, op_index);
        rcu_fnv_u32(h, thread);
        rcu_fnv_u32(h, root_id);
        rcu_fnv_u64(h, position);
        rcu_fnv_u64(h, obj_or_zero);
        rcu_fnv_i64(h, value_or_min);
        rcu_fnv_u8(h, retired);
        rcu_fnv_u64(h, alloc_or_max);
        rcu_fnv_u64(h, retire_or_max);
    }

    void op_read_chain(uint32_t op_index, const RcuOp& op) {
        const int32_t t = op.arg_a;
        const int32_t root_id = op.arg_b;
        const int32_t max_hops = op.arg_c;

        if (t < 0 || t >= thread_count ||
            root_id < 0 || root_id >= root_count) { emit_invalid(op_index, RCU_OP_READ_CHAIN); return; }
        RcuOracleThread& th = threads[(size_t)t];
        if (th.read_nest == 0) { emit_invalid(op_index, RCU_OP_READ_CHAIN); return; }

        uint64_t cur = th.root_view[(size_t)root_id];

        // Starting pointer 0 or absent -> READ_EMPTY.
        if (cur == 0 || !object_present(cur)) {
            emit_event(RCU_EV_READ_EMPTY, op_index, (uint32_t)t, UINT64_MAX,
                       (uint32_t)root_id, UINT32_MAX, th.active_epoch, 0);
            counts[RCU_COUNT_READ_EMPTY] += 1;
            return;
        }

        uint64_t position = 0;
        const int hop_cap = (max_hops < 0) ? 0 : max_hops;
        while (true) {
            if (!object_present(cur)) break;  // reused/absent boundary stops the walk
            const RcuOracleObject& o = objects.at(cur);

            emit_event(RCU_EV_READ_NODE, op_index, (uint32_t)t, cur,
                       (uint32_t)root_id, UINT32_MAX, th.active_epoch, position);
            counts[RCU_COUNT_READ_NODES] += 1;
            emit_read_record(op_index, (uint32_t)t, (uint32_t)root_id, position,
                             cur, o.value, o.retired, o.alloc_seq, o.retire_seq);

            if (position >= (uint64_t)hop_cap) break;  // up to max_hops edges
            uint64_t nxt = o.edges.empty() ? 0 : o.edges[0];
            if (nxt == 0 || !object_present(nxt)) break;  // stop at 0 or absent
            cur = nxt;
            position += 1;
        }
    }

    bool has_incoming_published(uint64_t id) const {
        for (size_t r = 0; r < root.size(); ++r) if (root[r] == id) return true;
        for (const auto& kv : objects) {
            const RcuOracleObject& o = kv.second;
            if (o.retired != 0) continue;  // only non-retired object edges count
            for (uint64_t e : o.edges) if (e == id) return true;
        }
        return false;
    }

    void op_retire(uint32_t op_index, const RcuOp& op) {
        const uint64_t id = (uint64_t)(uint32_t)op.arg_a;
        const uint32_t tag = (uint32_t)op.arg_b;

        auto it = objects.find(id);
        if (it == objects.end() || !it->second.present || it->second.retired != 0) {
            emit_invalid(op_index, RCU_OP_RETIRE); return;
        }
        if (has_incoming_published(id)) { emit_invalid(op_index, RCU_OP_RETIRE); return; }

        // If retired count would exceed max_retired, reject (do not mark).
        if ((int)retired_queue.size() + 1 > spec.max_retired) {
            emit_event(RCU_EV_RETIRE_REJECT_FULL, op_index, UINT32_MAX, id,
                       UINT32_MAX, UINT32_MAX, global_epoch, (uint64_t)tag);
            counts[RCU_COUNT_RETIRE_REJECT_FULL] += 1;
            return;
        }

        RcuOracleObject& o = it->second;
        o.retired = 1;
        o.retire_seq = retire_seq_next;
        retire_seq_next += 1;
        o.retire_epoch = global_epoch;
        o.callback_tag = tag;
        retired_queue.push_back(id);

        emit_event(RCU_EV_RETIRE_OK, op_index, UINT32_MAX, id,
                   UINT32_MAX, UINT32_MAX, o.retire_epoch, o.retire_seq);
        counts[RCU_COUNT_RETIRE_OK] += 1;
    }

    bool retire_eligible(uint64_t retire_seq) const {
        // eligible iff no thread with read_nest > 0 and outer_read_seq < retire_seq
        for (const RcuOracleThread& th : threads) {
            if (th.read_nest > 0 && th.outer_read_seq < retire_seq) return false;
        }
        return true;
    }

    void op_reclaim(uint32_t op_index, const RcuOp& op) {
        const int32_t limit = op.arg_a;
        if (limit <= 0) return;  // limit==0 valid no-op; negative treated as no-op

        sort_retired_queue();
        int reclaimed = 0;
        std::vector<uint64_t> remaining;
        remaining.reserve(retired_queue.size());

        for (size_t qi = 0; qi < retired_queue.size(); ++qi) {
            const uint64_t id = retired_queue[qi];
            if (reclaimed >= limit) { remaining.push_back(id); continue; }

            const RcuOracleObject& o = objects.at(id);
            if (!retire_eligible(o.retire_seq)) {
                remaining.push_back(id);  // skip, keep scanning
                continue;
            }

            const uint32_t tag = o.callback_tag;

            // RCU_CALLBACK first.
            emit_event(RCU_EV_RCU_CALLBACK, op_index, UINT32_MAX, id,
                       UINT32_MAX, UINT32_MAX, global_epoch, (uint64_t)tag);
            counts[RCU_COUNT_CALLBACKS_READY] += 1;

            // Append to free list (or drop if full).
            uint64_t reclaim_seq_used = UINT64_MAX;
            if ((int)free_list.size() >= spec.max_free_ids) {
                emit_event(RCU_EV_FREE_ID_DROPPED, op_index, UINT32_MAX, id,
                           UINT32_MAX, UINT32_MAX, global_epoch, id);
                counts[RCU_COUNT_FREE_IDS_DROPPED] += 1;
            } else {
                reclaim_seq_used = reclaim_seq_next;
                reclaim_seq_next += 1;
                free_list.push_back(RcuOracleFreeEntry{id, reclaim_seq_used});
            }

            // Remove from object table.
            objects.erase(id);

            emit_event(RCU_EV_OBJECT_RECLAIM, op_index, UINT32_MAX, id,
                       UINT32_MAX, UINT32_MAX, global_epoch, reclaim_seq_used);
            counts[RCU_COUNT_OBJECTS_RECLAIMED] += 1;
            reclaimed += 1;
        }

        retired_queue = remaining;
        // remaining keeps canonical order since we scanned canonical order.
    }

    void op_force_drop_free_ids(uint32_t op_index, const RcuOp& op) {
        const int32_t limit = op.arg_a;
        if (limit <= 0) return;

        sort_free_list();
        int removed = 0;
        std::vector<RcuOracleFreeEntry> kept;
        for (size_t i = 0; i < free_list.size(); ++i) {
            if (removed < limit) {
                const uint64_t id = free_list[i].obj_id;
                emit_event(RCU_EV_FREE_ID_DROPPED, op_index, UINT32_MAX, id,
                           UINT32_MAX, UINT32_MAX, global_epoch, id);
                counts[RCU_COUNT_FREE_IDS_DROPPED] += 1;
                removed += 1;
            } else {
                kept.push_back(free_list[i]);
            }
        }
        free_list = kept;
    }

    void apply_op(const RcuOp& op) {
        const uint32_t op_index = (uint32_t)op_index_global;
        switch (op.kind) {
            case RCU_OP_ALLOC: op_alloc(op_index, op); break;
            case RCU_OP_SET_ROOT: op_set_root(op_index, op); break;
            case RCU_OP_LINK: op_link(op_index, op); break;
            case RCU_OP_READ_LOCK: op_read_lock(op_index, op); break;
            case RCU_OP_READ_UNLOCK: op_read_unlock(op_index, op); break;
            case RCU_OP_QUIESCE: op_quiesce(op_index, op); break;
            case RCU_OP_ADVANCE_EPOCH: op_advance_epoch(op_index, op); break;
            case RCU_OP_READ_CHAIN: op_read_chain(op_index, op); break;
            case RCU_OP_RETIRE: op_retire(op_index, op); break;
            case RCU_OP_RECLAIM: op_reclaim(op_index, op); break;
            case RCU_OP_FORCE_DROP_FREE_IDS: op_force_drop_free_ids(op_index, op); break;
            default: emit_invalid(op_index, (uint32_t)op.kind); break;
        }
        op_index_global += 1;
    }

    // ---- structural checksums (final state) -----------------------------
    uint64_t compute_root_hash() const {
        uint64_t h = 1469598103934665603ULL;
        for (int r = 0; r < root_count; ++r) {
            rcu_fnv_u32(&h, (uint32_t)r);
            rcu_fnv_u64(&h, root[(size_t)r]);
        }
        return h;
    }

    uint64_t compute_object_hash() const {
        uint64_t h = 1469598103934665603ULL;
        // objects map is already sorted ascending by obj id.
        for (const auto& kv : objects) {
            const uint64_t id = kv.first;
            const RcuOracleObject& o = kv.second;
            rcu_fnv_u64(&h, id);
            rcu_fnv_i64(&h, o.value);
            rcu_fnv_u64(&h, o.alloc_seq);
            rcu_fnv_u8(&h, o.retired);
            rcu_fnv_u64(&h, o.retire_epoch);
            rcu_fnv_u64(&h, o.retire_seq);
            rcu_fnv_u32(&h, o.callback_tag);
            for (int s = 0; s < max_edges; ++s) {
                rcu_fnv_u64(&h, o.edges[(size_t)s]);
            }
        }
        return h;
    }

    uint64_t compute_thread_hash() const {
        uint64_t h = 1469598103934665603ULL;
        for (int t = 0; t < thread_count; ++t) {
            const RcuOracleThread& th = threads[(size_t)t];
            rcu_fnv_u32(&h, (uint32_t)t);
            rcu_fnv_u64(&h, th.read_nest);
            rcu_fnv_u64(&h, th.outer_read_seq);
            rcu_fnv_u64(&h, th.active_epoch);
            for (int r = 0; r < root_count; ++r) {
                rcu_fnv_u64(&h, th.root_view[(size_t)r]);
            }
        }
        return h;
    }

    uint64_t compute_free_hash() {
        sort_free_list();
        uint64_t h = 1469598103934665603ULL;
        for (const RcuOracleFreeEntry& e : free_list) {
            rcu_fnv_u64(&h, e.obj_id);
            rcu_fnv_u64(&h, e.reclaim_seq);
        }
        return h;
    }

    void run_ops(const RcuOp* ops, int op_count) {
        for (int i = 0; i < op_count; ++i) apply_op(ops[i]);
    }

    void snapshot(RcuExpected* exp) {
        for (int i = 0; i < RCU_COUNT_TOTAL; ++i) exp->counts[i] = counts[i];
        exp->event_hash = event_hash;
        exp->read_hash = read_hash;
        exp->root_hash = compute_root_hash();
        exp->object_hash = compute_object_hash();
        exp->thread_hash = compute_thread_hash();
        exp->free_hash = compute_free_hash();
        exp->state_scalars[0] = event_seq;
        exp->state_scalars[1] = obj_id_next;
        exp->state_scalars[2] = read_seq_next;
        exp->state_scalars[3] = retire_seq_next;
        exp->state_scalars[4] = reclaim_seq_next;
        exp->state_scalars[5] = global_epoch;
    }
};

static const char* rcu_count_name(int i) {
    switch (i) {
        case RCU_COUNT_ALLOC_OK: return "alloc_ok";
        case RCU_COUNT_ALLOC_OOM: return "alloc_oom";
        case RCU_COUNT_ROOT_SETS: return "root_sets";
        case RCU_COUNT_EDGE_SETS: return "edge_sets";
        case RCU_COUNT_READ_LOCKS: return "read_locks";
        case RCU_COUNT_READ_UNLOCKS: return "read_unlocks";
        case RCU_COUNT_QUIESCES: return "quiesces";
        case RCU_COUNT_EPOCH_ADVANCES: return "epoch_advances";
        case RCU_COUNT_READ_NODES: return "read_nodes";
        case RCU_COUNT_READ_EMPTY: return "read_empty";
        case RCU_COUNT_RETIRE_OK: return "retire_ok";
        case RCU_COUNT_RETIRE_REJECT_FULL: return "retire_reject_full";
        case RCU_COUNT_CALLBACKS_READY: return "callbacks_ready";
        case RCU_COUNT_OBJECTS_RECLAIMED: return "objects_reclaimed";
        case RCU_COUNT_FREE_IDS_DROPPED: return "free_ids_dropped";
        case RCU_COUNT_INVALID: return "invalid_count";
        default: return "?";
    }
}

static const char* rcu_scalar_name(int i) {
    switch (i) {
        case 0: return "event_seq";
        case 1: return "obj_id_next";
        case 2: return "read_seq_next";
        case 3: return "retire_seq_next";
        case 4: return "reclaim_seq_next";
        case 5: return "global_epoch";
        default: return "?";
    }
}

struct RcuHostOutputsView {
    const uint64_t* counts;
    uint64_t event_hash;
    uint64_t read_hash;
    uint64_t root_hash;
    uint64_t object_hash;
    uint64_t thread_hash;
    uint64_t free_hash;
    const uint64_t* state_scalars;
};

static inline bool rcu_check_outputs(const RcuExpected& exp,
                                     const RcuHostOutputsView& got,
                                     std::string* error) {
    for (int i = 0; i < RCU_COUNT_TOTAL; ++i) {
        if (got.counts[i] != exp.counts[i]) {
            if (error) {
                std::ostringstream oss;
                oss << "count " << rcu_count_name(i) << " mismatch: got "
                    << got.counts[i] << ", expected " << exp.counts[i];
                *error = oss.str();
            }
            return false;
        }
    }
    struct HPair { const char* n; uint64_t g; uint64_t e; };
    HPair pairs[] = {
        {"event_hash", got.event_hash, exp.event_hash},
        {"read_hash", got.read_hash, exp.read_hash},
        {"root_hash", got.root_hash, exp.root_hash},
        {"object_hash", got.object_hash, exp.object_hash},
        {"thread_hash", got.thread_hash, exp.thread_hash},
        {"free_hash", got.free_hash, exp.free_hash},
    };
    for (const HPair& p : pairs) {
        if (p.g != p.e) {
            if (error) {
                std::ostringstream oss;
                oss << p.n << " mismatch: got 0x" << std::hex << p.g
                    << ", expected 0x" << p.e;
                *error = oss.str();
            }
            return false;
        }
    }
    for (int i = 0; i < 6; ++i) {
        if (got.state_scalars[i] != exp.state_scalars[i]) {
            if (error) {
                std::ostringstream oss;
                oss << "scalar " << rcu_scalar_name(i) << " mismatch: got "
                    << got.state_scalars[i] << ", expected " << exp.state_scalars[i];
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

#endif  // RCU_EPOCH_RECLAIMER_ORACLE_HPP_
