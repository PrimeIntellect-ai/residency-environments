// file: cuckoo_tombstone_table_oracle.hpp

#ifndef CUCKOO_TOMBSTONE_TABLE_ORACLE_HPP_
#define CUCKOO_TOMBSTONE_TABLE_ORACLE_HPP_

#include "cuckoo_tombstone_table_common.h"

#include <stdint.h>
#include <stddef.h>

#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// Host oracle: independent, straightforward implementation of the full T47
// semantics. This is the source of truth for the test harness.
// ===========================================================================

struct CktHostInputsView {
    const int32_t* op_type;
    const int64_t* a0;
    const int64_t* a1;
};

struct CktExpected {
    int32_t counts[CKT_NUM_COUNTS] = {0};
    uint64_t op_event_hash = CKT_FNV_OFFSET;
    uint64_t read_hash = CKT_FNV_OFFSET;
    uint64_t slot_state_hash = CKT_FNV_OFFSET;
    uint64_t stash_hash = CKT_FNV_OFFSET;
    uint64_t page_hash = CKT_FNV_OFFSET;
};

struct CktHostOutputsView {
    const int32_t* counts;
    const uint64_t* op_event_hash;
    const uint64_t* read_hash;
    const uint64_t* slot_state_hash;
    const uint64_t* stash_hash;
    const uint64_t* page_hash;
};

static inline uint64_t ckt_o_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= CKT_FNV_PRIME;
    return h;
}

static inline void ckt_o_fnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = ckt_o_fnv_byte(v, b[i]);
    *h = v;
}

static inline uint64_t ckt_o_fnv_seed_key(uint64_t seed, uint64_t key) {
    uint64_t h = CKT_FNV_OFFSET;
    ckt_o_fnv_bytes(&h, &seed, sizeof(uint64_t));
    ckt_o_fnv_bytes(&h, &key, sizeof(uint64_t));
    return h;
}

struct CktOracleState {
    CktProblemSpec spec{};
    int slot_count = 0;
    int page_size = 0;
    int neighborhood = 0;
    int max_disp = 0;
    int stash_capacity = 0;
    int max_tombstones = 0;
    int n_pages = 0;

    // slot SoA
    std::vector<uint8_t> st;       // state
    std::vector<uint64_t> key;
    std::vector<int64_t> val;
    std::vector<uint8_t> home_kind;
    std::vector<uint64_t> home_slot;
    std::vector<uint64_t> iseq;
    std::vector<uint64_t> aux;      // version_seq (LIVE) / tomb_seq (TOMB)

    std::vector<uint64_t> pin_count;

    // stash, kept ordered by insert_seq ascending
    std::vector<uint64_t> s_key;
    std::vector<int64_t> s_val;
    std::vector<uint64_t> s_iseq;
    std::vector<uint64_t> s_vseq;

    uint64_t event_seq = 0;
    uint64_t insert_seq_next = 1;

    int tomb_count = 0;

    void init(const CktProblemSpec& s) {
        spec = s;
        slot_count = s.slot_count;
        page_size = s.page_size;
        neighborhood = s.neighborhood;
        max_disp = s.max_displacements_per_home;
        stash_capacity = s.stash_capacity;
        max_tombstones = s.max_tombstones;
        n_pages = ckt_n_pages(&s);
        reset();
    }

    void reset() {
        st.assign((size_t)slot_count, CKT_EMPTY);
        key.assign((size_t)slot_count, 0);
        val.assign((size_t)slot_count, 0);
        home_kind.assign((size_t)slot_count, 0);
        home_slot.assign((size_t)slot_count, 0);
        iseq.assign((size_t)slot_count, 0);
        aux.assign((size_t)slot_count, 0);
        pin_count.assign((size_t)n_pages, 0);
        s_key.clear(); s_val.clear(); s_iseq.clear(); s_vseq.clear();
        event_seq = 0;
        insert_seq_next = 1;
        tomb_count = 0;
    }

    int page_of(int slot) const { return slot / page_size; }
    int dist(int home, int slot) const {
        return (int)(((int64_t)slot + slot_count - home) % slot_count);
    }
    int home0_of(uint64_t k) const {
        return (int)(ckt_o_fnv_seed_key(spec.seed0, k) % (uint64_t)slot_count);
    }
    int home1_of(uint64_t k) const {
        int h0 = home0_of(k);
        int raw1 = (int)(ckt_o_fnv_seed_key(spec.seed1, k) % (uint64_t)slot_count);
        if (raw1 != h0) return raw1;
        return (raw1 + 1) % slot_count;
    }

    uint64_t next_event_seq() { return ++event_seq; }

    // ---- event hashing ----
    void hash_event(uint64_t* h, uint8_t kind, uint64_t ev, uint32_t op_index,
                    uint64_t key_f, uint64_t slot_f, uint8_t hk_f,
                    uint64_t hslot_f, int64_t val_f, uint64_t aux_f) const {
        ckt_o_fnv_bytes(h, &kind, sizeof(uint8_t));
        ckt_o_fnv_bytes(h, &ev, sizeof(uint64_t));
        ckt_o_fnv_bytes(h, &op_index, sizeof(uint32_t));
        ckt_o_fnv_bytes(h, &key_f, sizeof(uint64_t));
        ckt_o_fnv_bytes(h, &slot_f, sizeof(uint64_t));
        ckt_o_fnv_bytes(h, &hk_f, sizeof(uint8_t));
        ckt_o_fnv_bytes(h, &hslot_f, sizeof(uint64_t));
        ckt_o_fnv_bytes(h, &val_f, sizeof(int64_t));
        ckt_o_fnv_bytes(h, &aux_f, sizeof(uint64_t));
    }

    void hash_read(uint64_t* h, uint64_t read_id, uint64_t k, uint8_t found,
                   uint8_t src, uint64_t slot_f, uint64_t stash_iseq_f,
                   int64_t val_f, uint64_t vseq_f) const {
        ckt_o_fnv_bytes(h, &read_id, sizeof(uint64_t));
        ckt_o_fnv_bytes(h, &k, sizeof(uint64_t));
        ckt_o_fnv_bytes(h, &found, sizeof(uint8_t));
        ckt_o_fnv_bytes(h, &src, sizeof(uint8_t));
        ckt_o_fnv_bytes(h, &slot_f, sizeof(uint64_t));
        ckt_o_fnv_bytes(h, &stash_iseq_f, sizeof(uint64_t));
        ckt_o_fnv_bytes(h, &val_f, sizeof(int64_t));
        ckt_o_fnv_bytes(h, &vseq_f, sizeof(uint64_t));
    }

    // ---- table live search using GET table order; returns slot or -1 ----
    int find_live_table(uint64_t k) const {
        int h0 = home0_of(k);
        for (int off = 0; off < neighborhood; ++off) {
            int s = (h0 + off) % slot_count;
            if (st[(size_t)s] == CKT_LIVE && key[(size_t)s] == k) return s;
        }
        int h1 = home1_of(k);
        for (int off = 0; off < neighborhood; ++off) {
            int s = (h1 + off) % slot_count;
            // second physical visit skipped: equivalent to first-match, so a
            // re-visit cannot change the result; just scan.
            if (st[(size_t)s] == CKT_LIVE && key[(size_t)s] == k) return s;
        }
        return -1;
    }

    // stash index of first matching key (ascending iseq == ascending index)
    int find_stash(uint64_t k) const {
        for (size_t i = 0; i < s_key.size(); ++i) {
            if (s_key[i] == k) return (int)i;
        }
        return -1;
    }

    void stash_erase(int idx) {
        s_key.erase(s_key.begin() + idx);
        s_val.erase(s_val.begin() + idx);
        s_iseq.erase(s_iseq.begin() + idx);
        s_vseq.erase(s_vseq.begin() + idx);
    }

    // Find first global tombstone with matching key in tombstone-index order
    // (tomb_seq ascending, then slot ascending). Returns slot or -1.
    int find_tombstone_for_key(uint64_t k) const {
        int best = -1;
        uint64_t best_ts = 0;
        for (int s = 0; s < slot_count; ++s) {
            if (st[(size_t)s] != CKT_TOMBSTONE || key[(size_t)s] != k) continue;
            uint64_t ts = aux[(size_t)s];
            if (best < 0 || ts < best_ts || (ts == best_ts && s < best)) {
                best = s; best_ts = ts;
            }
        }
        return best;
    }

    // INSERT_WITH_HOME. Returns slot of final placement on success, -1 on fail.
    // Relocations and tombstone overwrites performed are NOT rolled back.
    // place_key/place_val/place_iseq/place_vseq describe the entry to insert.
    int insert_with_home(int home_kind_attempt, uint64_t place_key,
                         int64_t place_val, uint64_t place_iseq,
                         bool use_existing_vseq, uint64_t existing_vseq,
                         uint32_t op_index, uint64_t* ev_h,
                         CktExpected* exp) {
        int target_home = (home_kind_attempt == CKT_HOME0)
                              ? home0_of(place_key) : home1_of(place_key);

        // find first vacancy scanning offsets 0..slot_count-1 from target_home
        int vacancy = -1;
        for (int off = 0; off < slot_count; ++off) {
            int s = (target_home + off) % slot_count;
            if ((st[(size_t)s] == CKT_EMPTY || st[(size_t)s] == CKT_TOMBSTONE) &&
                pin_count[(size_t)page_of(s)] == 0) {
                vacancy = s; break;
            }
        }
        if (vacancy < 0) return -1;

        int disp = 0;
        while (dist(target_home, vacancy) >= neighborhood) {
            if (disp >= max_disp) return -1;
            // scan candidates vacancy-1, vacancy-2, ... vacancy-(neighborhood-1)
            int chosen = -1;
            for (int d = 1; d < neighborhood; ++d) {
                int c = ((vacancy - d) % slot_count + slot_count) % slot_count;
                if (st[(size_t)c] != CKT_LIVE) continue;
                if (pin_count[(size_t)page_of(c)] != 0) continue;
                int chome = (int)home_slot[(size_t)c];
                if (dist(chome, vacancy) < neighborhood) { chosen = c; break; }
            }
            if (chosen < 0) return -1;

            // move chosen -> vacancy (overwrite). chosen becomes EMPTY.
            // If vacancy was a tombstone it is destroyed (no event here).
            if (st[(size_t)vacancy] == CKT_TOMBSTONE) tomb_count -= 1;

            st[(size_t)vacancy] = CKT_LIVE;
            key[(size_t)vacancy] = key[(size_t)chosen];
            val[(size_t)vacancy] = val[(size_t)chosen];
            home_kind[(size_t)vacancy] = home_kind[(size_t)chosen];
            home_slot[(size_t)vacancy] = home_slot[(size_t)chosen];
            iseq[(size_t)vacancy] = iseq[(size_t)chosen];
            aux[(size_t)vacancy] = aux[(size_t)chosen];

            uint64_t ev = next_event_seq();
            hash_event(ev_h, CKT_EV_RELOCATE_SLOT, ev, op_index,
                       key[(size_t)vacancy], (uint64_t)vacancy,
                       home_kind[(size_t)vacancy], home_slot[(size_t)vacancy],
                       val[(size_t)vacancy], (uint64_t)chosen);
            exp->counts[10] += 1;  // slot_relocations

            st[(size_t)chosen] = CKT_EMPTY;
            key[(size_t)chosen] = 0; val[(size_t)chosen] = 0;
            home_kind[(size_t)chosen] = 0; home_slot[(size_t)chosen] = 0;
            iseq[(size_t)chosen] = 0; aux[(size_t)chosen] = 0;

            vacancy = chosen;
            disp += 1;
        }

        // place new entry at vacancy
        if (st[(size_t)vacancy] == CKT_TOMBSTONE) {
            uint64_t old_ts = aux[(size_t)vacancy];
            uint64_t ev = next_event_seq();
            hash_event(ev_h, CKT_EV_REUSE_TOMBSTONE, ev, op_index,
                       place_key, (uint64_t)vacancy,
                       (uint8_t)home_kind_attempt, (uint64_t)target_home,
                       CKT_I64_MIN, old_ts);
            exp->counts[11] += 1;  // tombstone_reused
            tomb_count -= 1;
        }

        uint64_t ev = next_event_seq();
        uint64_t vseq = use_existing_vseq ? existing_vseq : ev;
        st[(size_t)vacancy] = CKT_LIVE;
        key[(size_t)vacancy] = place_key;
        val[(size_t)vacancy] = place_val;
        home_kind[(size_t)vacancy] = (uint8_t)home_kind_attempt;
        home_slot[(size_t)vacancy] = (uint64_t)target_home;
        iseq[(size_t)vacancy] = place_iseq;
        aux[(size_t)vacancy] = vseq;

        hash_event(ev_h, CKT_EV_PUT_INSERT, ev, op_index, place_key,
                   (uint64_t)vacancy, (uint8_t)home_kind_attempt,
                   (uint64_t)target_home, place_val, place_iseq);
        exp->counts[2] += 1;  // put_inserted
        return vacancy;
    }

    // Sweep tombstones in tombstone-index order. Returns nothing; emits events.
    // limit < 0 => unlimited (used by auto-sweep-until-threshold via target).
    // For auto-sweep we instead loop until tomb_count <= max_tombstones.
    void sweep_step(uint32_t op_index, uint64_t* ev_h, CktExpected* exp,
                    int limit, bool until_threshold) {
        int removed = 0;
        while (true) {
            if (until_threshold) {
                if (tomb_count <= max_tombstones) break;
            } else {
                if (removed >= limit) break;
            }
            // find next removable tombstone in index order (tomb_seq asc,
            // then slot asc) that is on an unpinned page.
            int target = -1;
            uint64_t best_ts = 0;
            for (int s = 0; s < slot_count; ++s) {
                if (st[(size_t)s] != CKT_TOMBSTONE) continue;
                if (pin_count[(size_t)page_of(s)] != 0) continue;  // skip pinned
                uint64_t ts = aux[(size_t)s];
                if (target < 0 || ts < best_ts || (ts == best_ts && s < target)) {
                    target = s; best_ts = ts;
                }
            }
            if (target < 0) break;  // no removable tombstone remains

            uint64_t ev = next_event_seq();
            hash_event(ev_h, CKT_EV_TOMBSTONE_SWEEP, ev, op_index,
                       key[(size_t)target], (uint64_t)target,
                       home_kind[(size_t)target], home_slot[(size_t)target],
                       val[(size_t)target], aux[(size_t)target]);
            exp->counts[12] += 1;  // tombstone_swept

            st[(size_t)target] = CKT_EMPTY;
            key[(size_t)target] = 0; val[(size_t)target] = 0;
            home_kind[(size_t)target] = 0; home_slot[(size_t)target] = 0;
            iseq[(size_t)target] = 0; aux[(size_t)target] = 0;
            tomb_count -= 1;
            removed += 1;
        }
    }

    // ---- per-op handlers ----
    void op_get(uint32_t op_index, uint64_t read_id, uint64_t k,
                uint64_t* ev_h, uint64_t* rd_h, CktExpected* exp) {
        int slot = find_live_table(k);
        uint8_t found; uint8_t src;
        uint64_t slot_f; uint64_t stash_iseq_f; int64_t val_f; uint64_t vseq_f;
        if (slot >= 0) {
            found = 1; src = CKT_SRC_TABLE;
            slot_f = (uint64_t)slot; stash_iseq_f = CKT_U64_MAX;
            val_f = val[(size_t)slot]; vseq_f = aux[(size_t)slot];
            exp->counts[0] += 1;  // get_found
        } else {
            int si = find_stash(k);
            if (si >= 0) {
                found = 1; src = CKT_SRC_STASH;
                slot_f = CKT_U64_MAX; stash_iseq_f = s_iseq[(size_t)si];
                val_f = s_val[(size_t)si]; vseq_f = s_vseq[(size_t)si];
                exp->counts[0] += 1;  // get_found
            } else {
                found = 0; src = CKT_SRC_NONE;
                slot_f = CKT_U64_MAX; stash_iseq_f = CKT_U64_MAX;
                val_f = CKT_I64_MIN; vseq_f = CKT_U64_MAX;
                exp->counts[1] += 1;  // get_missing
            }
        }
        uint64_t ev = next_event_seq();
        hash_event(ev_h, CKT_EV_GET_RESULT, ev, op_index, k, slot_f,
                   CKT_HK_NONE, CKT_U64_MAX, val_f, (uint64_t)src);
        hash_read(rd_h, read_id, k, found, src, slot_f, stash_iseq_f,
                  val_f, vseq_f);
    }

    void op_put(uint32_t op_index, uint64_t k, int64_t value,
                uint64_t* ev_h, CktExpected* exp) {
        int slot = find_live_table(k);
        if (slot >= 0) {
            uint64_t ev = next_event_seq();
            val[(size_t)slot] = value;
            aux[(size_t)slot] = ev;  // version_seq
            hash_event(ev_h, CKT_EV_UPDATE_EXISTING, ev, op_index, k,
                       (uint64_t)slot, home_kind[(size_t)slot],
                       home_slot[(size_t)slot], value, ev);
            exp->counts[3] += 1;  // put_updated
            return;
        }
        int si = find_stash(k);
        if (si >= 0) {
            uint64_t ev = next_event_seq();
            s_val[(size_t)si] = value;
            s_vseq[(size_t)si] = ev;
            hash_event(ev_h, CKT_EV_UPDATE_STASH, ev, op_index, k,
                       CKT_U64_MAX, CKT_HK_NONE, CKT_U64_MAX, value,
                       s_iseq[(size_t)si]);
            exp->counts[3] += 1;  // put_updated
            return;
        }
        int ts = find_tombstone_for_key(k);
        if (ts >= 0) {
            uint64_t ev = next_event_seq();
            st[(size_t)ts] = CKT_LIVE;
            // key, home_kind, home_slot, iseq preserved; value set; vseq=ev
            val[(size_t)ts] = value;
            aux[(size_t)ts] = ev;
            tomb_count -= 1;
            hash_event(ev_h, CKT_EV_RESURRECT_TOMBSTONE, ev, op_index, k,
                       (uint64_t)ts, home_kind[(size_t)ts],
                       home_slot[(size_t)ts], value, iseq[(size_t)ts]);
            exp->counts[4] += 1;  // put_resurrected
            return;
        }
        // fresh insert
        uint64_t new_iseq = insert_seq_next++;
        int r = insert_with_home(CKT_HOME0, k, value, new_iseq, false, 0,
                                 op_index, ev_h, exp);
        if (r < 0) {
            r = insert_with_home(CKT_HOME1, k, value, new_iseq, false, 0,
                                 op_index, ev_h, exp);
        }
        if (r >= 0) return;  // PUT_INSERT already emitted/counted
        // both failed -> stash or oom
        if ((int)s_key.size() < stash_capacity) {
            uint64_t ev = next_event_seq();
            s_key.push_back(k);
            s_val.push_back(value);
            s_iseq.push_back(new_iseq);
            s_vseq.push_back(ev);
            hash_event(ev_h, CKT_EV_PUT_STASH, ev, op_index, k, CKT_U64_MAX,
                       CKT_HK_NONE, CKT_U64_MAX, value, new_iseq);
            exp->counts[5] += 1;  // put_stashed
        } else {
            uint64_t ev = next_event_seq();
            hash_event(ev_h, CKT_EV_PUT_OOM, ev, op_index, k, CKT_U64_MAX,
                       CKT_HK_NONE, CKT_U64_MAX, value, new_iseq);
            exp->counts[6] += 1;  // put_oom
        }
    }

    void op_delete(uint32_t op_index, uint64_t k, uint64_t* ev_h,
                   CktExpected* exp) {
        int slot = find_live_table(k);
        if (slot >= 0) {
            uint64_t ev = next_event_seq();
            // convert to tombstone; old_value = current value
            st[(size_t)slot] = CKT_TOMBSTONE;
            aux[(size_t)slot] = ev;  // tomb_seq
            tomb_count += 1;
            hash_event(ev_h, CKT_EV_DELETE_TABLE, ev, op_index, k,
                       (uint64_t)slot, home_kind[(size_t)slot],
                       home_slot[(size_t)slot], val[(size_t)slot], ev);
            exp->counts[7] += 1;  // delete_table
            if (tomb_count > max_tombstones) {
                sweep_step(op_index, ev_h, exp, 0, /*until_threshold=*/true);
            }
            return;
        }
        int si = find_stash(k);
        if (si >= 0) {
            uint64_t ev = next_event_seq();
            int64_t removed_val = s_val[(size_t)si];
            uint64_t removed_iseq = s_iseq[(size_t)si];
            hash_event(ev_h, CKT_EV_DELETE_STASH, ev, op_index, k, CKT_U64_MAX,
                       CKT_HK_NONE, CKT_U64_MAX, removed_val, removed_iseq);
            exp->counts[8] += 1;  // delete_stash
            stash_erase(si);
            return;
        }
        uint64_t ev = next_event_seq();
        hash_event(ev_h, CKT_EV_DELETE_MISS, ev, op_index, k, CKT_U64_MAX,
                   CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN, CKT_U64_MAX);
        exp->counts[9] += 1;  // delete_miss
    }

    void op_pin(uint32_t op_index, int64_t page_arg, uint64_t* ev_h,
                CktExpected* exp) {
        if (page_arg < 0 || page_arg >= n_pages ||
            pin_count[(size_t)page_arg] == CKT_U64_MAX) {
            uint64_t ev = next_event_seq();
            hash_event(ev_h, CKT_EV_INVALID, ev, op_index, CKT_U64_MAX,
                       CKT_U64_MAX, CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN,
                       (uint64_t)page_arg);
            exp->counts[16] += 1;  // invalid_count
            return;
        }
        pin_count[(size_t)page_arg] += 1;
        uint64_t ev = next_event_seq();
        hash_event(ev_h, CKT_EV_PIN_PAGE_OK, ev, op_index, CKT_U64_MAX,
                   CKT_U64_MAX, CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN,
                   (uint64_t)page_arg);
        exp->counts[14] += 1;  // page_pins
    }

    void op_unpin(uint32_t op_index, int64_t page_arg, uint64_t* ev_h,
                  CktExpected* exp) {
        if (page_arg < 0 || page_arg >= n_pages ||
            pin_count[(size_t)page_arg] == 0) {
            uint64_t ev = next_event_seq();
            hash_event(ev_h, CKT_EV_INVALID, ev, op_index, CKT_U64_MAX,
                       CKT_U64_MAX, CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN,
                       (uint64_t)page_arg);
            exp->counts[16] += 1;
            return;
        }
        pin_count[(size_t)page_arg] -= 1;
        uint64_t ev = next_event_seq();
        hash_event(ev_h, CKT_EV_UNPIN_PAGE_OK, ev, op_index, CKT_U64_MAX,
                   CKT_U64_MAX, CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN,
                   (uint64_t)page_arg);
        exp->counts[15] += 1;  // page_unpins
    }

    void op_sweep(uint32_t op_index, int64_t limit, uint64_t* ev_h,
                  CktExpected* exp) {
        int lim = (limit < 0) ? 0 : (int)limit;
        sweep_step(op_index, ev_h, exp, lim, /*until_threshold=*/false);
    }

    void op_replay(uint32_t op_index, int64_t limit, uint64_t* ev_h,
                   CktExpected* exp) {
        int lim = (limit < 0) ? 0 : (int)limit;
        int success = 0;
        size_t i = 0;
        while (i < s_key.size()) {
            if (success >= lim) break;
            uint64_t k = s_key[i];
            int64_t v = s_val[i];
            uint64_t es = s_iseq[i];
            uint64_t vs = s_vseq[i];
            int r = insert_with_home(CKT_HOME0, k, v, es, true, vs,
                                     op_index, ev_h, exp);
            if (r < 0) {
                r = insert_with_home(CKT_HOME1, k, v, es, true, vs,
                                     op_index, ev_h, exp);
            }
            if (r >= 0) {
                uint64_t ev = next_event_seq();
                hash_event(ev_h, CKT_EV_STASH_REPLAY_OK, ev, op_index, k,
                           (uint64_t)r, home_kind[(size_t)r],
                           home_slot[(size_t)r], v, es);
                exp->counts[13] += 1;  // stash_replayed
                stash_erase((int)i);  // do not advance i
                success += 1;
            } else {
                i += 1;  // keep, advance
            }
        }
    }

    // ---- final state hashes ----
    void finalize_hashes(CktExpected* exp) const {
        // slot_state_hash
        uint64_t h = CKT_FNV_OFFSET;
        for (int s = 0; s < slot_count; ++s) {
            uint64_t su = (uint64_t)s;
            uint8_t state = st[(size_t)s];
            ckt_o_fnv_bytes(&h, &su, sizeof(uint64_t));
            ckt_o_fnv_bytes(&h, &state, sizeof(uint8_t));
            if (state == CKT_EMPTY) continue;
            uint64_t kk = key[(size_t)s];
            int64_t vv = val[(size_t)s];
            uint8_t hk = home_kind[(size_t)s];
            uint64_t hs = home_slot[(size_t)s];
            uint64_t is = iseq[(size_t)s];
            uint64_t ax = aux[(size_t)s];
            ckt_o_fnv_bytes(&h, &kk, sizeof(uint64_t));
            ckt_o_fnv_bytes(&h, &vv, sizeof(int64_t));
            ckt_o_fnv_bytes(&h, &hk, sizeof(uint8_t));
            ckt_o_fnv_bytes(&h, &hs, sizeof(uint64_t));
            ckt_o_fnv_bytes(&h, &is, sizeof(uint64_t));
            ckt_o_fnv_bytes(&h, &ax, sizeof(uint64_t));
        }
        exp->slot_state_hash = h;

        // stash_hash (already in insert_seq ascending order)
        uint64_t hs2 = CKT_FNV_OFFSET;
        for (size_t i = 0; i < s_key.size(); ++i) {
            uint64_t kk = s_key[i];
            int64_t vv = s_val[i];
            uint64_t is = s_iseq[i];
            uint64_t vs = s_vseq[i];
            ckt_o_fnv_bytes(&hs2, &kk, sizeof(uint64_t));
            ckt_o_fnv_bytes(&hs2, &vv, sizeof(int64_t));
            ckt_o_fnv_bytes(&hs2, &is, sizeof(uint64_t));
            ckt_o_fnv_bytes(&hs2, &vs, sizeof(uint64_t));
        }
        exp->stash_hash = hs2;

        // page_hash
        uint64_t hp = CKT_FNV_OFFSET;
        for (int p = 0; p < n_pages; ++p) {
            uint64_t pu = (uint64_t)p;
            uint64_t pc = pin_count[(size_t)p];
            ckt_o_fnv_bytes(&hp, &pu, sizeof(uint64_t));
            ckt_o_fnv_bytes(&hp, &pc, sizeof(uint64_t));
        }
        exp->page_hash = hp;
    }

    void step_once(const CktRunSpec& run, const CktHostInputsView& in,
                   CktExpected* exp) {
        for (int i = 0; i < CKT_NUM_COUNTS; ++i) exp->counts[i] = 0;
        exp->op_event_hash = CKT_FNV_OFFSET;
        exp->read_hash = CKT_FNV_OFFSET;

        uint64_t ev_h = CKT_FNV_OFFSET;
        uint64_t rd_h = CKT_FNV_OFFSET;

        for (int o = 0; o < run.num_ops; ++o) {
            uint32_t op_index = (uint32_t)o;
            int32_t ot = in.op_type[o];
            int64_t a0 = in.a0[o];
            int64_t a1 = in.a1[o];
            switch (ot) {
                case CKT_OP_GET:
                    op_get(op_index, (uint64_t)a0, (uint64_t)a1, &ev_h, &rd_h, exp);
                    break;
                case CKT_OP_PUT:
                    op_put(op_index, (uint64_t)a0, a1, &ev_h, exp);
                    break;
                case CKT_OP_DELETE:
                    op_delete(op_index, (uint64_t)a0, &ev_h, exp);
                    break;
                case CKT_OP_PIN_PAGE:
                    op_pin(op_index, a0, &ev_h, exp);
                    break;
                case CKT_OP_UNPIN_PAGE:
                    op_unpin(op_index, a0, &ev_h, exp);
                    break;
                case CKT_OP_SWEEP_TOMBSTONES:
                    op_sweep(op_index, a0, &ev_h, exp);
                    break;
                case CKT_OP_REPLAY_STASH:
                    op_replay(op_index, a0, &ev_h, exp);
                    break;
                default:
                    // unknown op -> INVALID
                    {
                        uint64_t ev = next_event_seq();
                        hash_event(&ev_h, CKT_EV_INVALID, ev, op_index,
                                   CKT_U64_MAX, CKT_U64_MAX, CKT_HK_NONE,
                                   CKT_U64_MAX, CKT_I64_MIN, CKT_U64_MAX);
                        exp->counts[16] += 1;
                    }
                    break;
            }
        }

        exp->op_event_hash = ev_h;
        exp->read_hash = rd_h;
        finalize_hashes(exp);
    }
};

static inline bool ckt_check_all_outputs(const CktExpected& e,
                                         const CktHostOutputsView& g,
                                         std::string* error) {
    static const char* names[CKT_NUM_COUNTS] = {
        "get_found", "get_missing", "put_inserted", "put_updated",
        "put_resurrected", "put_stashed", "put_oom", "delete_table",
        "delete_stash", "delete_miss", "slot_relocations", "tombstone_reused",
        "tombstone_swept", "stash_replayed", "page_pins", "page_unpins",
        "invalid_count"};
    for (int i = 0; i < CKT_NUM_COUNTS; ++i) {
        if (g.counts[i] != e.counts[i]) {
            if (error) {
                std::ostringstream oss;
                oss << "count[" << i << "] (" << names[i] << ") mismatch: got "
                    << g.counts[i] << ", expected " << e.counts[i];
                *error = oss.str();
            }
            return false;
        }
    }
#define CKT_CHK_HASH(field, label)                                          \
    if (g.field[0] != e.field) {                                            \
        if (error) {                                                        \
            std::ostringstream oss;                                         \
            oss << label " mismatch: got 0x" << std::hex << g.field[0]      \
                << ", expected 0x" << e.field;                              \
            *error = oss.str();                                             \
        }                                                                   \
        return false;                                                       \
    }
    CKT_CHK_HASH(op_event_hash, "op_event_hash")
    CKT_CHK_HASH(read_hash, "read_hash")
    CKT_CHK_HASH(slot_state_hash, "slot_state_hash")
    CKT_CHK_HASH(stash_hash, "stash_hash")
    CKT_CHK_HASH(page_hash, "page_hash")
#undef CKT_CHK_HASH
    return true;
}

/*
GRADER MODEL
  oracle.init(problem_spec)
  solution_reset + oracle.reset()
  for each step:
    solution_run(...)
    oracle.step_once(...)
    ckt_check_all_outputs(...)

Required harness coverage:
  - cuckoo displacement chains (hopscotch relocation)
  - failed HOME0 placement that still mutates table, then HOME1 / stash
  - tombstone resurrection (same key) vs tombstone reuse (different key)
  - paginated pin/unpin blocking placement & relocation but not delete/resurrect
  - auto-sweep after delete with pinned tombstones skipped
  - explicit SWEEP_TOMBSTONES and REPLAY_STASH
  - num_ops = 0
  - reset and exact replay
*/

#endif  // CUCKOO_TOMBSTONE_TABLE_ORACLE_HPP_
