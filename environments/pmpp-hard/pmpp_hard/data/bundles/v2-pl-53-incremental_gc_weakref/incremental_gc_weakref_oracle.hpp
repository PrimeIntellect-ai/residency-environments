// file: incremental_gc_weakref_oracle.hpp
//
// Authoritative host-side executable semantics for T53. The reference and
// naive CUDA implementations are validated against this oracle. This is a
// fully independent implementation (pure host C++ std::vector data model).
//
// COUNT VECTOR ORDER (index in IgcwOutputs.counts), exactly matching the
// spec's "Counts:" list:
//   0  alloc_ok            1  alloc_black          2  alloc_oom
//   3  root_sets           4  root_clears          5  strong_sets
//   6  strong_clears       7  weak_sets            8  ephemeron_sets
//   9  ephemeron_deletes  10  ephemeron_oom       11  remembered_added
//  12  remembered_dropped 13  gc_start_minor      14  gc_start_full
//  15  mark_black         16  mark_grey           17  ephemeron_marked
//  18  finalizer_enqueued 19  finalizer_queue_full 20 weak_fields_cleared
//  21  ephemeron_sides_cleared 22 sweep_freed      23  sweep_kept
//  24  promoted_old       25  ephemeron_owner_freed 26 gc_end
//  27  finalizer_run      28  finalizer_resurrect 29  finalizer_skip_absent
//  30  barrier_marks
// (invalid_count is reported separately via IgcwOutputs.invalid_flag, which
//  the harness accumulates; we also keep a running invalid_count for the
//  controller, but it is not part of the per-op count vector.)

#ifndef INCREMENTAL_GC_WEAKREF_ORACLE_HPP_
#define INCREMENTAL_GC_WEAKREF_ORACLE_HPP_

#include "incremental_gc_weakref_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

// ----- FNV-1a-64 ------------------------------------------------------------

static inline uint64_t igcw_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}
static inline void igcw_fnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = igcw_fnv_byte(v, b[i]);
    *h = v;
}
static inline void igcw_fnv_u8(uint64_t* h, uint8_t v)  { igcw_fnv_bytes(h, &v, 1); }
static inline void igcw_fnv_u32(uint64_t* h, uint32_t v){ igcw_fnv_bytes(h, &v, 4); }
static inline void igcw_fnv_u64(uint64_t* h, uint64_t v){ igcw_fnv_bytes(h, &v, 8); }

#define IGCW_FNV_INIT 1469598103934665603ULL

// ----- Oracle state ---------------------------------------------------------

struct IgcwOracle {
    // Capacities.
    int max_objects = 0;
    int root_count = 0;
    int strong_slots = 0;
    int weak_slots = 0;
    int max_ephemerons = 0;
    int max_mark_queue = 0;
    int max_finalizer_queue = 0;
    int young_survive_threshold = 1;

    // Persistent scalars.
    uint64_t event_seq = 0;
    uint64_t obj_id_next = 1;
    uint64_t alloc_seq_next = 1;
    uint64_t gc_cycle_id = 0;

    // Object table, indexed by obj_id-1.
    std::vector<uint8_t>  present;        // present[i] => obj_id i+1 allocated
    std::vector<uint64_t> o_size;
    std::vector<uint8_t>  o_gen;
    std::vector<uint8_t>  o_age;
    std::vector<uint8_t>  o_color;
    std::vector<uint8_t>  o_alloc_during_gc;
    std::vector<uint64_t> o_alloc_seq;
    std::vector<uint32_t> o_fin_tag;
    std::vector<uint32_t> o_fin_root_slot; // UINT32_MAX = none
    std::vector<uint8_t>  o_finalized;
    std::vector<uint8_t>  o_in_fin_queue;
    std::vector<int32_t>  o_strong;        // [obj][slot], 0 = none
    std::vector<int32_t>  o_weak;          // [obj][slot], 0 = none

    // Roots.
    std::vector<int32_t> root; // 0 = none

    // Ephemerons (slot-indexed dense; eph_present[id]).
    std::vector<uint8_t>  eph_present;
    std::vector<int32_t>  eph_owner;
    std::vector<int32_t>  eph_key;
    std::vector<int32_t>  eph_value;
    std::vector<uint64_t> eph_create_seq;

    // Remembered set: canonical ordered list of (src,slot) pairs (unique).
    std::vector<std::pair<int32_t,int32_t>> remembered;

    // Free-id list, ascending.
    std::vector<int32_t> free_ids;

    // Finalizer queue (FIFO).
    std::vector<int32_t> fin_queue;

    // GC controller.
    int phase = IGCW_PHASE_IDLE;
    int mode = IGCW_MODE_NONE;
    std::vector<int32_t> mark_queue;       // FIFO
    uint64_t scan_cursor_obj = 0;
    uint32_t scan_cursor_field = 0;
    uint64_t sweep_cursor_obj = 0;
    uint8_t ephemeron_changed = 0;
    std::vector<uint8_t> in_cset;          // collection-set membership, obj_id-1

    // Per-op accumulators.
    int32_t counts[IGCW_NUM_COUNTS];
    uint64_t event_hash = 0;
    uint64_t invalid_count = 0;
    int op_invalid = 0;

    // -------- lifecycle --------

    void init(const IgcwProblemSpec& s) {
        max_objects = s.max_objects;
        root_count = s.root_count;
        strong_slots = s.strong_slots_per_object;
        weak_slots = s.weak_slots_per_object;
        max_ephemerons = s.max_ephemerons;
        max_mark_queue = s.max_mark_queue;
        max_finalizer_queue = s.max_finalizer_queue;
        young_survive_threshold = s.young_survive_threshold;
        reset();
    }

    void reset() {
        event_seq = 0; obj_id_next = 1; alloc_seq_next = 1; gc_cycle_id = 0;
        present.assign((size_t)max_objects, 0);
        o_size.assign((size_t)max_objects, 0);
        o_gen.assign((size_t)max_objects, IGCW_GEN_YOUNG);
        o_age.assign((size_t)max_objects, 0);
        o_color.assign((size_t)max_objects, IGCW_WHITE);
        o_alloc_during_gc.assign((size_t)max_objects, 0);
        o_alloc_seq.assign((size_t)max_objects, 0);
        o_fin_tag.assign((size_t)max_objects, 0);
        o_fin_root_slot.assign((size_t)max_objects, IGCW_U32_MAX);
        o_finalized.assign((size_t)max_objects, 0);
        o_in_fin_queue.assign((size_t)max_objects, 0);
        o_strong.assign((size_t)max_objects * (size_t)std::max(strong_slots,0), 0);
        o_weak.assign((size_t)max_objects * (size_t)std::max(weak_slots,0), 0);
        root.assign((size_t)std::max(root_count,0), 0);
        eph_present.assign((size_t)std::max(max_ephemerons,0), 0);
        eph_owner.assign((size_t)std::max(max_ephemerons,0), 0);
        eph_key.assign((size_t)std::max(max_ephemerons,0), 0);
        eph_value.assign((size_t)std::max(max_ephemerons,0), 0);
        eph_create_seq.assign((size_t)std::max(max_ephemerons,0), 0);
        remembered.clear();
        free_ids.clear();
        fin_queue.clear();
        phase = IGCW_PHASE_IDLE; mode = IGCW_MODE_NONE;
        mark_queue.clear();
        scan_cursor_obj = 0; scan_cursor_field = 0; sweep_cursor_obj = 0;
        ephemeron_changed = 0;
        in_cset.assign((size_t)max_objects, 0);
        invalid_count = 0;
    }

    // -------- helpers --------

    bool valid_obj(int32_t id) const {
        return id >= 1 && id <= max_objects && present[(size_t)(id-1)] != 0;
    }
    int32_t* strong_ptr(int32_t id) { return &o_strong[(size_t)(id-1)*(size_t)strong_slots]; }
    int32_t* weak_ptr(int32_t id)   { return &o_weak[(size_t)(id-1)*(size_t)weak_slots]; }
    bool in_collection_set(int32_t id) const {
        return id >= 1 && id <= max_objects && present[(size_t)(id-1)] && in_cset[(size_t)(id-1)];
    }
    bool gc_active() const { return phase != IGCW_PHASE_IDLE; }

    int find_eph(uint64_t eid) const {
        // ephemeron_id is the dense slot index.
        if ((int64_t)eid < 0 || (int64_t)eid >= max_ephemerons) return -1;
        return eph_present[(size_t)eid] ? (int)eid : -1;
    }

    // Emit an event into the hash, in emission order.
    void emit(int kind, uint64_t obj, uint32_t slot, uint32_t root_id,
              uint64_t eph, uint64_t aux) {
        igcw_fnv_u8 (&event_hash, (uint8_t)kind);
        igcw_fnv_u64(&event_hash, event_seq);
        igcw_fnv_u32(&event_hash, (uint32_t)cur_op_index);
        igcw_fnv_u64(&event_hash, gc_cycle_id);
        igcw_fnv_u64(&event_hash, obj);
        igcw_fnv_u32(&event_hash, slot);
        igcw_fnv_u32(&event_hash, root_id);
        igcw_fnv_u64(&event_hash, eph);
        igcw_fnv_u64(&event_hash, aux);
        event_seq = event_seq + 1; // wraps mod 2^64
    }

    uint32_t cur_op_index = 0;

    void mark_invalid() {
        op_invalid = 1;
        invalid_count = invalid_count + 1;
        emit(IGCW_EV_INVALID, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
    }

    // Enqueue into mark queue (no capacity guard in spec for mark queue ops
    // other than the controller bound; mark queue capacity is large).
    void mq_push(int32_t id) { mark_queue.push_back(id); }

    // -------- per-op dispatch --------

    void run_op(const IgcwRunSpec& r) {
        cur_op_index = (uint32_t)r.op_index;
        for (int i = 0; i < IGCW_NUM_COUNTS; ++i) counts[i] = 0;
        op_invalid = 0;

        switch (r.opcode) {
            case IGCW_OP_ALLOC:           op_alloc(r); break;
            case IGCW_OP_SET_ROOT:        op_set_root(r); break;
            case IGCW_OP_CLEAR_ROOT:      op_clear_root(r); break;
            case IGCW_OP_SET_STRONG:      op_set_strong(r); break;
            case IGCW_OP_CLEAR_STRONG:    op_clear_strong(r); break;
            case IGCW_OP_SET_WEAK:        op_set_weak(r); break;
            case IGCW_OP_SET_EPHEMERON:   op_set_ephemeron(r); break;
            case IGCW_OP_DELETE_EPHEMERON:op_delete_ephemeron(r); break;
            case IGCW_OP_START_MINOR:     op_start_minor(r); break;
            case IGCW_OP_START_FULL:      op_start_full(r); break;
            case IGCW_OP_GC_STEP:         op_gc_step(r); break;
            case IGCW_OP_RUN_FINALIZERS:  op_run_finalizers(r); break;
            default: mark_invalid(); break;
        }
    }

    // ===== ALLOC =====
    void op_alloc(const IgcwRunSpec& r) {
        const int gen = r.a0;
        const uint32_t fin_tag = (uint32_t)r.a1;
        uint32_t fin_slot = (uint32_t)r.a2;
        const uint64_t size = (uint64_t)r.size_arg;

        if (gen != IGCW_GEN_YOUNG && gen != IGCW_GEN_OLD) { mark_invalid(); return; }

        if (fin_tag == 0) {
            fin_slot = IGCW_U32_MAX;
        } else {
            if (fin_slot != IGCW_U32_MAX && fin_slot >= (uint32_t)root_count) {
                mark_invalid(); return;
            }
        }

        // Choose id.
        int32_t id;
        if (!free_ids.empty()) {
            id = free_ids.front(); // lowest (list kept ascending)
        } else {
            if ((int64_t)obj_id_next > max_objects) {
                emit(IGCW_EV_ALLOC_OOM, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                counts[2]++; // alloc_oom
                return;
            }
            id = (int32_t)obj_id_next;
        }
        // Pop the chosen id.
        if (!free_ids.empty()) {
            free_ids.erase(free_ids.begin());
        } else {
            obj_id_next = obj_id_next + 1;
        }

        size_t k = (size_t)(id-1);
        present[k] = 1;
        o_size[k] = size;
        o_gen[k] = (uint8_t)gen;
        o_age[k] = 0;
        o_alloc_seq[k] = alloc_seq_next; alloc_seq_next = alloc_seq_next + 1;
        o_fin_tag[k] = fin_tag;
        o_fin_root_slot[k] = fin_slot;
        o_finalized[k] = 0;
        o_in_fin_queue[k] = 0;
        for (int s = 0; s < strong_slots; ++s) strong_ptr(id)[s] = 0;
        for (int s = 0; s < weak_slots; ++s) weak_ptr(id)[s] = 0;

        if (!gc_active()) {
            o_color[k] = IGCW_WHITE;
            o_alloc_during_gc[k] = 0;
            in_cset[k] = 0;
            emit(IGCW_EV_ALLOC_OK, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, size);
            counts[0]++; // alloc_ok
        } else {
            in_cset[k] = 1; // object is in the active collection set
            o_color[k] = IGCW_BLACK;
            o_alloc_during_gc[k] = 1;
            emit(IGCW_EV_ALLOC_BLACK, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, size);
            counts[1]++; // alloc_black
            // "Always emit ALLOC_OK for successful allocation."
            emit(IGCW_EV_ALLOC_OK, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, size);
            counts[0]++; // alloc_ok
        }
    }

    // ===== SET_ROOT =====
    void op_set_root(const IgcwRunSpec& r) {
        const int32_t root_id = r.a0;
        const int32_t obj = r.a1;
        if (root_id < 0 || root_id >= root_count) { mark_invalid(); return; }
        if (obj != 0 && !valid_obj(obj)) { mark_invalid(); return; }
        root[(size_t)root_id] = obj;

        if (gc_active() && obj != 0 && in_collection_set(obj) &&
            o_color[(size_t)(obj-1)] == IGCW_WHITE) {
            o_color[(size_t)(obj-1)] = IGCW_GREY;
            mq_push(obj);
            emit(IGCW_EV_BARRIER_MARK, (uint64_t)obj, IGCW_U32_MAX, (uint32_t)root_id, IGCW_U64_MAX, 0);
            counts[30]++; // barrier_marks
        }
        emit(IGCW_EV_ROOT_SET, (uint64_t)obj, IGCW_U32_MAX, (uint32_t)root_id, IGCW_U64_MAX, 0);
        counts[3]++; // root_sets
    }

    // ===== CLEAR_ROOT =====
    void op_clear_root(const IgcwRunSpec& r) {
        const int32_t root_id = r.a0;
        if (root_id < 0 || root_id >= root_count) { mark_invalid(); return; }
        root[(size_t)root_id] = 0;
        emit(IGCW_EV_ROOT_CLEAR, 0, IGCW_U32_MAX, (uint32_t)root_id, IGCW_U64_MAX, 0);
        counts[4]++; // root_clears
    }

    // ===== SET_STRONG =====
    void op_set_strong(const IgcwRunSpec& r) {
        const int32_t src = r.a0;
        const int32_t slot = r.a1;
        const int32_t dst = r.a2;
        if (!valid_obj(src)) { mark_invalid(); return; }
        if (slot < 0 || slot >= strong_slots) { mark_invalid(); return; }
        if (dst != 0 && !valid_obj(dst)) { mark_invalid(); return; }

        strong_ptr(src)[slot] = dst;

        // Remembered set: src OLD, dst YOUNG.
        if (dst != 0 && o_gen[(size_t)(src-1)] == IGCW_GEN_OLD &&
            o_gen[(size_t)(dst-1)] == IGCW_GEN_YOUNG) {
            std::pair<int32_t,int32_t> pr(src, slot);
            bool found = false;
            for (auto& p : remembered) if (p == pr) { found = true; break; }
            if (!found) {
                remembered.push_back(pr);
                std::sort(remembered.begin(), remembered.end());
                emit(IGCW_EV_REMEMBERED_ADD, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                counts[11]++; // remembered_added
            }
        }

        // Write barrier: src BLACK, dst in cset and WHITE.
        if (gc_active() && o_color[(size_t)(src-1)] == IGCW_BLACK &&
            dst != 0 && in_collection_set(dst) &&
            o_color[(size_t)(dst-1)] == IGCW_WHITE) {
            o_color[(size_t)(dst-1)] = IGCW_GREY;
            mq_push(dst);
            emit(IGCW_EV_BARRIER_MARK, (uint64_t)dst, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, 0);
            counts[30]++; // barrier_marks
        }
        emit(IGCW_EV_STRONG_SET, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, (uint64_t)(uint32_t)dst);
        counts[5]++; // strong_sets
    }

    // ===== CLEAR_STRONG =====
    void op_clear_strong(const IgcwRunSpec& r) {
        const int32_t src = r.a0;
        const int32_t slot = r.a1;
        if (!valid_obj(src)) { mark_invalid(); return; }
        if (slot < 0 || slot >= strong_slots) { mark_invalid(); return; }
        strong_ptr(src)[slot] = 0;
        emit(IGCW_EV_STRONG_CLEAR, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, 0);
        counts[6]++; // strong_clears
    }

    // ===== SET_WEAK =====
    void op_set_weak(const IgcwRunSpec& r) {
        const int32_t src = r.a0;
        const int32_t slot = r.a1;
        const int32_t dst = r.a2;
        if (!valid_obj(src)) { mark_invalid(); return; }
        if (slot < 0 || slot >= weak_slots) { mark_invalid(); return; }
        if (dst != 0 && !valid_obj(dst)) { mark_invalid(); return; }
        weak_ptr(src)[slot] = dst;
        emit(IGCW_EV_WEAK_SET, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, (uint64_t)(uint32_t)dst);
        counts[7]++; // weak_sets
    }

    // ===== SET_EPHEMERON =====
    void op_set_ephemeron(const IgcwRunSpec& r) {
        const uint64_t eid = (uint64_t)(uint32_t)r.a0;
        const int32_t owner = r.a1;
        const int32_t key = r.a2;
        const int32_t value = r.a3;
        if (!valid_obj(owner)) { mark_invalid(); return; }
        if (key != 0 && !valid_obj(key)) { mark_invalid(); return; }
        if (value != 0 && !valid_obj(value)) { mark_invalid(); return; }
        // id range check: ephemeron_id is dense slot index.
        if ((int64_t)eid < 0 || (int64_t)eid >= max_ephemerons) { mark_invalid(); return; }

        const bool exists = eph_present[(size_t)eid] != 0;
        if (exists && eph_owner[(size_t)eid] != owner) { mark_invalid(); return; }

        if (!exists) {
            // count current ephemerons to detect "full".
            int used = 0;
            for (int i = 0; i < max_ephemerons; ++i) if (eph_present[(size_t)i]) ++used;
            if (used >= max_ephemerons) {
                emit(IGCW_EV_EPHEMERON_OOM, 0, IGCW_U32_MAX, IGCW_U32_MAX, eid, 0);
                counts[10]++; // ephemeron_oom
                return;
            }
            eph_present[(size_t)eid] = 1;
            eph_create_seq[(size_t)eid] = alloc_seq_next; // create_seq from alloc_seq stream
            alloc_seq_next = alloc_seq_next + 1;
        }
        eph_owner[(size_t)eid] = owner;
        eph_key[(size_t)eid] = key;
        eph_value[(size_t)eid] = value;

        if (gc_active() && key != 0 && o_color[(size_t)(key-1)] == IGCW_BLACK &&
            value != 0 && in_collection_set(value) &&
            o_color[(size_t)(value-1)] == IGCW_WHITE) {
            o_color[(size_t)(value-1)] = IGCW_GREY;
            mq_push(value);
            emit(IGCW_EV_BARRIER_MARK, (uint64_t)value, IGCW_U32_MAX, IGCW_U32_MAX, eid, 0);
            counts[30]++; // barrier_marks
        }
        emit(IGCW_EV_EPHEMERON_SET, (uint64_t)owner, IGCW_U32_MAX, IGCW_U32_MAX, eid, (uint64_t)(uint32_t)value);
        counts[8]++; // ephemeron_sets
    }

    // ===== DELETE_EPHEMERON =====
    void op_delete_ephemeron(const IgcwRunSpec& r) {
        const uint64_t eid = (uint64_t)(uint32_t)r.a0;
        if (find_eph(eid) < 0) { mark_invalid(); return; }
        eph_present[(size_t)eid] = 0;
        eph_owner[(size_t)eid] = 0;
        eph_key[(size_t)eid] = 0;
        eph_value[(size_t)eid] = 0;
        eph_create_seq[(size_t)eid] = 0;
        emit(IGCW_EV_EPHEMERON_DELETE, 0, IGCW_U32_MAX, IGCW_U32_MAX, eid, 0);
        counts[9]++; // ephemeron_deletes
    }

    // ===== START_MINOR =====
    void op_start_minor(const IgcwRunSpec& r) {
        (void)r;
        if (phase != IGCW_PHASE_IDLE) { mark_invalid(); return; }
        gc_cycle_id = gc_cycle_id + 1;
        mode = IGCW_MODE_MINOR;
        phase = IGCW_PHASE_MARK;

        // Collection set = all currently young objects.
        for (int i = 0; i < max_objects; ++i) {
            if (present[(size_t)i] && o_gen[(size_t)i] == IGCW_GEN_YOUNG) {
                in_cset[(size_t)i] = 1;
                o_color[(size_t)i] = IGCW_WHITE;
                o_alloc_during_gc[(size_t)i] = 0;
            } else {
                in_cset[(size_t)i] = 0;
                // old objects: leave color unchanged, treated as black for cycle.
            }
        }
        mark_queue.clear();
        scan_cursor_obj = 0; scan_cursor_field = 0; sweep_cursor_obj = 0;
        ephemeron_changed = 0;

        // Enqueue all young root targets.
        for (int rid = 0; rid < root_count; ++rid) {
            int32_t obj = root[(size_t)rid];
            if (obj != 0 && in_collection_set(obj) &&
                o_color[(size_t)(obj-1)] == IGCW_WHITE) {
                o_color[(size_t)(obj-1)] = IGCW_GREY;
                mq_push(obj);
            }
        }

        // Scan remembered set in canonical order.
        std::vector<std::pair<int32_t,int32_t>> kept;
        for (auto& p : remembered) {
            int32_t src = p.first, slot = p.second;
            bool drop = false;
            if (!valid_obj(src)) {
                drop = true;
            } else {
                int32_t tgt = strong_ptr(src)[slot];
                if (tgt != 0 && in_collection_set(tgt)) {
                    if (o_color[(size_t)(tgt-1)] == IGCW_WHITE) {
                        o_color[(size_t)(tgt-1)] = IGCW_GREY;
                        mq_push(tgt);
                    }
                } else {
                    drop = true;
                }
            }
            if (drop) {
                emit(IGCW_EV_REMEMBERED_DROP, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                counts[12]++; // remembered_dropped
            } else {
                kept.push_back(p);
            }
        }
        remembered.swap(kept);

        emit(IGCW_EV_GC_START_MINOR, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, gc_cycle_id);
        counts[13]++; // gc_start_minor
    }

    // ===== START_FULL =====
    void op_start_full(const IgcwRunSpec& r) {
        (void)r;
        if (phase != IGCW_PHASE_IDLE) { mark_invalid(); return; }
        gc_cycle_id = gc_cycle_id + 1;
        mode = IGCW_MODE_FULL;
        phase = IGCW_PHASE_MARK;

        for (int i = 0; i < max_objects; ++i) {
            if (present[(size_t)i]) {
                in_cset[(size_t)i] = 1;
                o_color[(size_t)i] = IGCW_WHITE;
                o_alloc_during_gc[(size_t)i] = 0;
            } else {
                in_cset[(size_t)i] = 0;
            }
        }
        mark_queue.clear();
        scan_cursor_obj = 0; scan_cursor_field = 0; sweep_cursor_obj = 0;
        ephemeron_changed = 0;

        for (int rid = 0; rid < root_count; ++rid) {
            int32_t obj = root[(size_t)rid];
            if (obj != 0 && in_collection_set(obj) &&
                o_color[(size_t)(obj-1)] == IGCW_WHITE) {
                o_color[(size_t)(obj-1)] = IGCW_GREY;
                mq_push(obj);
            }
        }

        emit(IGCW_EV_GC_START_FULL, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, gc_cycle_id);
        counts[14]++; // gc_start_full
    }

    // ===== GC_STEP =====
    void op_gc_step(const IgcwRunSpec& r) {
        const int mark_budget = r.a0;
        const int sweep_budget = r.a1;
        if (phase == IGCW_PHASE_IDLE) { mark_invalid(); return; }
        if (mark_budget == 0 && sweep_budget == 0) { mark_invalid(); return; }

        // "Advance at most one phase transition at a time, but may consume both
        //  mark and sweep budget if the current phase reaches sweep during this
        //  call." We implement: run the current phase to its single transition
        //  (bounded by mark_budget for mark-like phases), then if the resulting
        //  phase is SWEEP, also run the sweep using sweep_budget.

        bool reached_sweep = false;
        switch (phase) {
            case IGCW_PHASE_MARK:
            case IGCW_PHASE_RESURRECT_MARK:
                step_mark(mark_budget);
                break;
            case IGCW_PHASE_EPHEMERON:
                step_ephemeron();
                break;
            case IGCW_PHASE_FINALIZE_SCAN:
                step_finalize_scan();
                break;
            case IGCW_PHASE_WEAK_CLEAR:
                step_weak_clear();
                reached_sweep = (phase == IGCW_PHASE_SWEEP);
                break;
            case IGCW_PHASE_SWEEP:
                step_sweep(sweep_budget);
                break;
            default: break;
        }
        if (reached_sweep) {
            step_sweep(sweep_budget);
        }
    }

    // MARK / RESURRECT_MARK
    void step_mark(int budget) {
        int processed = 0;
        while (processed < budget && !mark_queue.empty()) {
            int32_t id = mark_queue.front();
            mark_queue.erase(mark_queue.begin());
            ++processed;
            if (!valid_obj(id) || !in_collection_set(id)) continue;
            o_color[(size_t)(id-1)] = IGCW_BLACK;
            emit(IGCW_EV_MARK_BLACK, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
            counts[15]++; // mark_black
            for (int s = 0; s < strong_slots; ++s) {
                int32_t tgt = strong_ptr(id)[s];
                if (tgt != 0 && in_collection_set(tgt) &&
                    o_color[(size_t)(tgt-1)] == IGCW_WHITE) {
                    o_color[(size_t)(tgt-1)] = IGCW_GREY;
                    mq_push(tgt);
                    emit(IGCW_EV_MARK_GREY, (uint64_t)tgt, (uint32_t)s, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                    counts[16]++; // mark_grey
                }
            }
        }
        if (mark_queue.empty()) {
            phase = IGCW_PHASE_EPHEMERON;
            emit(IGCW_EV_PHASE_EPHEMERON, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
        }
    }

    // EPHEMERON
    void step_ephemeron() {
        ephemeron_changed = 0;
        for (int i = 0; i < max_ephemerons; ++i) {
            if (!eph_present[(size_t)i]) continue;
            int32_t key = eph_key[(size_t)i];
            int32_t val = eph_value[(size_t)i];
            // key is black OR (outside collection set but allocated).
            bool key_reachable = false;
            if (key != 0 && valid_obj(key)) {
                if (in_collection_set(key)) {
                    key_reachable = (o_color[(size_t)(key-1)] == IGCW_BLACK);
                } else {
                    key_reachable = true; // outside cset but allocated
                }
            }
            if (key_reachable && val != 0 && in_collection_set(val) &&
                o_color[(size_t)(val-1)] == IGCW_WHITE) {
                o_color[(size_t)(val-1)] = IGCW_GREY;
                mq_push(val);
                ephemeron_changed = 1;
                emit(IGCW_EV_EPHEMERON_MARK, (uint64_t)val, IGCW_U32_MAX, IGCW_U32_MAX, (uint64_t)i, 0);
                counts[17]++; // ephemeron_marked
            }
        }
        if (ephemeron_changed == 1) {
            phase = IGCW_PHASE_MARK;
            emit(IGCW_EV_PHASE_MARK, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
        } else {
            phase = IGCW_PHASE_FINALIZE_SCAN;
            scan_cursor_obj = 1; // start scanning at lowest object id
            emit(IGCW_EV_PHASE_FINALIZE_SCAN, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
        }
    }

    // FINALIZE_SCAN (runs to completion of the scan in one step)
    void step_finalize_scan() {
        for (int i = 0; i < max_objects; ++i) {
            int32_t id = i + 1;
            if (!in_collection_set(id)) continue;
            size_t k = (size_t)i;
            if (o_color[k] == IGCW_WHITE && o_fin_tag[k] != 0 &&
                o_finalized[k] == 0 && o_in_fin_queue[k] == 0) {
                if ((int)fin_queue.size() >= max_finalizer_queue) {
                    emit(IGCW_EV_FINALIZER_QUEUE_FULL, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                    counts[19]++; // finalizer_queue_full
                    // leave white and not finalized
                } else {
                    o_finalized[k] = 1;
                    o_in_fin_queue[k] = 1;
                    fin_queue.push_back(id);
                    o_color[k] = IGCW_GREY;
                    mq_push(id);
                    emit(IGCW_EV_FINALIZER_ENQUEUE, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                    counts[18]++; // finalizer_enqueued
                }
            }
        }
        scan_cursor_obj = 0;
        if (!mark_queue.empty()) {
            phase = IGCW_PHASE_RESURRECT_MARK;
            emit(IGCW_EV_PHASE_RESURRECT_MARK, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
        } else {
            phase = IGCW_PHASE_WEAK_CLEAR;
            emit(IGCW_EV_PHASE_WEAK_CLEAR, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
        }
    }

    // WEAK_CLEAR (runs to completion, then transitions to SWEEP)
    void step_weak_clear() {
        // Weak fields: all allocated objects by id, weak slot ascending.
        for (int i = 0; i < max_objects; ++i) {
            int32_t id = i + 1;
            if (!present[(size_t)i]) continue;
            for (int s = 0; s < weak_slots; ++s) {
                int32_t tgt = weak_ptr(id)[s];
                if (tgt != 0 && in_collection_set(tgt) &&
                    o_color[(size_t)(tgt-1)] == IGCW_WHITE) {
                    weak_ptr(id)[s] = 0;
                    emit(IGCW_EV_WEAK_CLEAR_FIELD, (uint64_t)id, (uint32_t)s, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                    counts[20]++; // weak_fields_cleared
                }
            }
        }
        // Ephemerons by id: clear key/value sides pointing at white cset objs.
        for (int i = 0; i < max_ephemerons; ++i) {
            if (!eph_present[(size_t)i]) continue;
            int32_t key = eph_key[(size_t)i];
            if (key != 0 && in_collection_set(key) &&
                o_color[(size_t)(key-1)] == IGCW_WHITE) {
                eph_key[(size_t)i] = 0;
                emit(IGCW_EV_EPHEMERON_CLEAR_SIDE, 0, 0, IGCW_U32_MAX, (uint64_t)i, 0); // aux 0 => key side
                counts[21]++; // ephemeron_sides_cleared
            }
            int32_t val = eph_value[(size_t)i];
            if (val != 0 && in_collection_set(val) &&
                o_color[(size_t)(val-1)] == IGCW_WHITE) {
                eph_value[(size_t)i] = 0;
                emit(IGCW_EV_EPHEMERON_CLEAR_SIDE, 0, 1, IGCW_U32_MAX, (uint64_t)i, 1); // aux 1 => value side
                counts[21]++; // ephemeron_sides_cleared
            }
        }
        phase = IGCW_PHASE_SWEEP;
        sweep_cursor_obj = 1; // lowest object id
        emit(IGCW_EV_PHASE_SWEEP, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
    }

    // SWEEP
    void step_sweep(int budget) {
        int processed = 0;
        // Scan obj ids ascending from sweep_cursor_obj; only process
        // allocated objects that are in the collection set, up to budget.
        int64_t cur = (int64_t)sweep_cursor_obj;
        if (cur < 1) cur = 1;
        for (; cur <= max_objects && processed < budget; ++cur) {
            int32_t id = (int32_t)cur;
            size_t k = (size_t)(id-1);
            if (!present[k] || !in_cset[k]) continue;
            ++processed;
            if (o_color[k] == IGCW_WHITE) {
                // Remove the object.
                // Remove ephemerons owned by it.
                for (int e = 0; e < max_ephemerons; ++e) {
                    if (eph_present[(size_t)e] && eph_owner[(size_t)e] == id) {
                        eph_present[(size_t)e] = 0;
                        eph_owner[(size_t)e] = 0;
                        eph_key[(size_t)e] = 0;
                        eph_value[(size_t)e] = 0;
                        eph_create_seq[(size_t)e] = 0;
                        emit(IGCW_EV_EPHEMERON_OWNER_FREE, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, (uint64_t)e, 0);
                        counts[25]++; // ephemeron_owner_freed
                    }
                }
                // Remove remembered-set pairs whose source is it.
                {
                    std::vector<std::pair<int32_t,int32_t>> kept;
                    for (auto& p : remembered) if (p.first != id) kept.push_back(p);
                    remembered.swap(kept);
                }
                // Clear object record; append id to free list (ascending).
                present[k] = 0;
                in_cset[k] = 0;
                o_size[k] = 0; o_gen[k] = IGCW_GEN_YOUNG; o_age[k] = 0;
                o_color[k] = IGCW_WHITE; o_alloc_during_gc[k] = 0; o_alloc_seq[k] = 0;
                o_fin_tag[k] = 0; o_fin_root_slot[k] = IGCW_U32_MAX;
                o_finalized[k] = 0; o_in_fin_queue[k] = 0;
                for (int s = 0; s < strong_slots; ++s) strong_ptr(id)[s] = 0;
                for (int s = 0; s < weak_slots; ++s) weak_ptr(id)[s] = 0;
                free_ids.push_back(id);
                std::sort(free_ids.begin(), free_ids.end());
                emit(IGCW_EV_SWEEP_FREE, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                counts[22]++; // sweep_freed
            } else { // BLACK
                if (mode == IGCW_MODE_MINOR && o_gen[k] == IGCW_GEN_YOUNG) {
                    o_age[k] = (uint8_t)(o_age[k] + 1);
                    if (o_age[k] >= young_survive_threshold) {
                        o_gen[k] = IGCW_GEN_OLD;
                        emit(IGCW_EV_PROMOTE_OLD, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                        counts[24]++; // promoted_old
                    }
                }
                o_color[k] = IGCW_WHITE;
                o_alloc_during_gc[k] = 0;
                emit(IGCW_EV_SWEEP_KEEP, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                counts[23]++; // sweep_kept
            }
        }
        sweep_cursor_obj = (uint64_t)cur;

        // Have all collection-set objects been scanned? cur > max_objects means
        // the cursor has passed the last id.
        bool done = true;
        for (int64_t j = cur; j <= max_objects; ++j) {
            size_t k = (size_t)(j-1);
            if (present[k] && in_cset[k]) { done = false; break; }
        }
        if (done) {
            // Clean remembered set: remove stale or no-longer-old-to-young.
            std::vector<std::pair<int32_t,int32_t>> kept;
            for (auto& p : remembered) {
                int32_t src = p.first, slot = p.second;
                bool drop = false;
                if (!valid_obj(src)) {
                    drop = true;
                } else {
                    int32_t tgt = strong_ptr(src)[slot];
                    if (!(o_gen[(size_t)(src-1)] == IGCW_GEN_OLD &&
                          tgt != 0 && valid_obj(tgt) &&
                          o_gen[(size_t)(tgt-1)] == IGCW_GEN_YOUNG)) {
                        drop = true;
                    }
                }
                if (drop) {
                    emit(IGCW_EV_REMEMBERED_DROP, (uint64_t)src, (uint32_t)slot, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                    counts[12]++; // remembered_dropped
                } else {
                    kept.push_back(p);
                }
            }
            remembered.swap(kept);

            phase = IGCW_PHASE_IDLE;
            mode = IGCW_MODE_NONE;
            for (int i = 0; i < max_objects; ++i) in_cset[(size_t)i] = 0;
            mark_queue.clear();
            sweep_cursor_obj = 0;
            emit(IGCW_EV_GC_END, 0, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, gc_cycle_id);
            counts[26]++; // gc_end
        }
    }

    // ===== RUN_FINALIZERS =====
    void op_run_finalizers(const IgcwRunSpec& r) {
        const int limit = r.a0;
        if (limit == 0) return; // valid no-op
        if (limit < 0) { mark_invalid(); return; }
        int popped = 0;
        while (popped < limit && !fin_queue.empty()) {
            int32_t id = fin_queue.front();
            fin_queue.erase(fin_queue.begin());
            ++popped;
            if (!valid_obj(id)) {
                emit(IGCW_EV_FINALIZER_SKIP_ABSENT, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                counts[29]++; // finalizer_skip_absent
                continue;
            }
            o_in_fin_queue[(size_t)(id-1)] = 0;
            uint32_t slot = o_fin_root_slot[(size_t)(id-1)];
            if (slot != IGCW_U32_MAX && slot < (uint32_t)root_count) {
                root[(size_t)slot] = id;
                emit(IGCW_EV_FINALIZER_RESURRECT, (uint64_t)id, IGCW_U32_MAX, slot, IGCW_U64_MAX, 0);
                counts[28]++; // finalizer_resurrect
            } else {
                emit(IGCW_EV_FINALIZER_RUN, (uint64_t)id, IGCW_U32_MAX, IGCW_U32_MAX, IGCW_U64_MAX, 0);
                counts[27]++; // finalizer_run
            }
        }
    }

    // -------- hashes --------

    uint64_t heap_hash() const {
        uint64_t h = IGCW_FNV_INIT;
        for (int i = 0; i < max_objects; ++i) {
            if (!present[(size_t)i]) continue;
            int32_t id = i + 1;
            igcw_fnv_u64(&h, (uint64_t)id);
            igcw_fnv_u64(&h, o_size[(size_t)i]);
            igcw_fnv_u8 (&h, o_gen[(size_t)i]);
            igcw_fnv_u8 (&h, o_age[(size_t)i]);
            igcw_fnv_u8 (&h, o_color[(size_t)i]);
            igcw_fnv_u8 (&h, o_alloc_during_gc[(size_t)i]);
            igcw_fnv_u64(&h, o_alloc_seq[(size_t)i]);
            igcw_fnv_u32(&h, o_fin_tag[(size_t)i]);
            igcw_fnv_u32(&h, o_fin_root_slot[(size_t)i]);
            igcw_fnv_u8 (&h, o_finalized[(size_t)i]);
            igcw_fnv_u8 (&h, o_in_fin_queue[(size_t)i]);
            for (int s = 0; s < strong_slots; ++s)
                igcw_fnv_u64(&h, (uint64_t)(uint32_t)o_strong[(size_t)i*(size_t)strong_slots + s]);
            for (int s = 0; s < weak_slots; ++s)
                igcw_fnv_u64(&h, (uint64_t)(uint32_t)o_weak[(size_t)i*(size_t)weak_slots + s]);
        }
        return h;
    }

    uint64_t root_hash() const {
        uint64_t h = IGCW_FNV_INIT;
        for (int rid = 0; rid < root_count; ++rid) {
            igcw_fnv_u32(&h, (uint32_t)rid);
            igcw_fnv_u64(&h, (uint64_t)(uint32_t)root[(size_t)rid]);
        }
        return h;
    }

    uint64_t ephemeron_hash() const {
        uint64_t h = IGCW_FNV_INIT;
        for (int i = 0; i < max_ephemerons; ++i) {
            if (!eph_present[(size_t)i]) continue;
            igcw_fnv_u64(&h, (uint64_t)i);
            igcw_fnv_u64(&h, (uint64_t)(uint32_t)eph_owner[(size_t)i]);
            igcw_fnv_u64(&h, (uint64_t)(uint32_t)eph_key[(size_t)i]);
            igcw_fnv_u64(&h, (uint64_t)(uint32_t)eph_value[(size_t)i]);
            igcw_fnv_u64(&h, eph_create_seq[(size_t)i]);
        }
        return h;
    }

    uint64_t remembered_hash() const {
        uint64_t h = IGCW_FNV_INIT;
        for (auto& p : remembered) {
            igcw_fnv_u64(&h, (uint64_t)(uint32_t)p.first);
            igcw_fnv_u32(&h, (uint32_t)p.second);
        }
        return h;
    }

    uint64_t controller_hash() const {
        uint64_t h = IGCW_FNV_INIT;
        igcw_fnv_u8 (&h, (uint8_t)phase);
        igcw_fnv_u8 (&h, (uint8_t)mode);
        igcw_fnv_u64(&h, gc_cycle_id);
        igcw_fnv_u64(&h, scan_cursor_obj);
        igcw_fnv_u32(&h, scan_cursor_field);
        igcw_fnv_u64(&h, sweep_cursor_obj);
        igcw_fnv_u8 (&h, ephemeron_changed);
        for (int32_t v : mark_queue) igcw_fnv_u64(&h, (uint64_t)(uint32_t)v);
        for (int32_t v : fin_queue)  igcw_fnv_u64(&h, (uint64_t)(uint32_t)v);
        for (int32_t v : free_ids)   igcw_fnv_u64(&h, (uint64_t)(uint32_t)v);
        return h;
    }
};

// -------- expected snapshot for one op --------

struct IgcwExpected {
    int32_t counts[IGCW_NUM_COUNTS];
    uint64_t gc_event_hash = 0;
    uint64_t heap_hash = 0;
    uint64_t root_hash = 0;
    uint64_t ephemeron_hash = 0;
    uint64_t remembered_hash = 0;
    uint64_t gc_controller_hash = 0;
    int32_t invalid_flag = 0;
};

static inline void igcw_oracle_step(IgcwOracle* o, const IgcwRunSpec& r, IgcwExpected* e) {
    o->run_op(r);
    for (int i = 0; i < IGCW_NUM_COUNTS; ++i) e->counts[i] = o->counts[i];
    e->gc_event_hash = o->event_hash;
    e->heap_hash = o->heap_hash();
    e->root_hash = o->root_hash();
    e->ephemeron_hash = o->ephemeron_hash();
    e->remembered_hash = o->remembered_hash();
    e->gc_controller_hash = o->controller_hash();
    e->invalid_flag = o->op_invalid;
}

struct IgcwGotView {
    const int32_t* counts;
    uint64_t gc_event_hash;
    uint64_t heap_hash;
    uint64_t root_hash;
    uint64_t ephemeron_hash;
    uint64_t remembered_hash;
    uint64_t gc_controller_hash;
    int32_t invalid_flag;
};

static inline bool igcw_check(const IgcwExpected& e, const IgcwGotView& g, std::string* err) {
    for (int i = 0; i < IGCW_NUM_COUNTS; ++i) {
        if (g.counts[i] != e.counts[i]) {
            if (err) { std::ostringstream o; o << "count["<<i<<"] got "<<g.counts[i]<<" exp "<<e.counts[i]; *err=o.str(); }
            return false;
        }
    }
    if (g.invalid_flag != e.invalid_flag) { if (err) *err="invalid_flag mismatch"; return false; }
    if (g.gc_event_hash != e.gc_event_hash)   { if(err) *err="gc_event_hash mismatch"; return false; }
    if (g.heap_hash != e.heap_hash)           { if(err) *err="heap_hash mismatch"; return false; }
    if (g.root_hash != e.root_hash)           { if(err) *err="root_hash mismatch"; return false; }
    if (g.ephemeron_hash != e.ephemeron_hash) { if(err) *err="ephemeron_hash mismatch"; return false; }
    if (g.remembered_hash != e.remembered_hash){ if(err) *err="remembered_hash mismatch"; return false; }
    if (g.gc_controller_hash != e.gc_controller_hash){ if(err) *err="gc_controller_hash mismatch"; return false; }
    return true;
}

#endif  // INCREMENTAL_GC_WEAKREF_ORACLE_HPP_
