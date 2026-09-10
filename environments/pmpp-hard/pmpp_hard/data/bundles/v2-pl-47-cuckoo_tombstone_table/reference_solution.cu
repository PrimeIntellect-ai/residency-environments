// PMPP_CANARY_47_e3306a0566 -- held-out canary; MUST NOT appear in any submission
// file: cuckoo_tombstone_table_reference.cu
//
// Reference implementation: a single device thread mutates the persistent
// device-resident table state directly, op by op, in one kernel launch.

#include "cuckoo_tombstone_table_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct CktRefState {
    CktProblemSpec spec;
    int slot_count, page_size, neighborhood, max_disp;
    int stash_capacity, max_tombstones, n_pages;

    // slot SoA (device)
    uint8_t* st;
    uint64_t* key;
    int64_t* val;
    uint8_t* home_kind;
    uint64_t* home_slot;
    uint64_t* iseq;
    uint64_t* aux;
    // page
    uint64_t* pin_count;
    // stash
    uint64_t* s_key;
    int64_t* s_val;
    uint64_t* s_iseq;
    uint64_t* s_vseq;
    // scalars (device): [0]=stash_size [1]=tomb_count
    int32_t* scalars_i;
    uint64_t* scalars_u;  // [0]=event_seq [1]=insert_seq_next
};

__device__ __forceinline__ uint64_t rfnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= CKT_FNV_PRIME;
    return h;
}
__device__ void rfnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = rfnv_byte(v, b[i]);
    *h = v;
}
__device__ uint64_t rfnv_seed_key(uint64_t seed, uint64_t key) {
    uint64_t h = CKT_FNV_OFFSET;
    rfnv_bytes(&h, &seed, sizeof(uint64_t));
    rfnv_bytes(&h, &key, sizeof(uint64_t));
    return h;
}

// Context shared with op handlers running inside the single-thread kernel.
struct RCtx {
    int slot_count, page_size, neighborhood, max_disp;
    int stash_capacity, max_tombstones, n_pages;
    uint64_t seed0, seed1;

    uint8_t* st;
    uint64_t* key;
    int64_t* val;
    uint8_t* home_kind;
    uint64_t* home_slot;
    uint64_t* iseq;
    uint64_t* aux;
    uint64_t* pin_count;
    uint64_t* s_key;
    int64_t* s_val;
    uint64_t* s_iseq;
    uint64_t* s_vseq;

    int stash_size;
    int tomb_count;
    uint64_t event_seq;
    uint64_t insert_seq_next;

    int32_t* counts;
    uint64_t ev_h;
    uint64_t rd_h;
};

__device__ __forceinline__ int rc_page_of(RCtx* c, int slot) { return slot / c->page_size; }
__device__ __forceinline__ int rc_dist(RCtx* c, int home, int slot) {
    return (int)(((int64_t)slot + c->slot_count - home) % c->slot_count);
}
__device__ __forceinline__ int rc_home0(RCtx* c, uint64_t k) {
    return (int)(rfnv_seed_key(c->seed0, k) % (uint64_t)c->slot_count);
}
__device__ __forceinline__ int rc_home1(RCtx* c, uint64_t k) {
    int h0 = rc_home0(c, k);
    int raw1 = (int)(rfnv_seed_key(c->seed1, k) % (uint64_t)c->slot_count);
    if (raw1 != h0) return raw1;
    return (raw1 + 1) % c->slot_count;
}
__device__ __forceinline__ uint64_t rc_next_ev(RCtx* c) { return ++c->event_seq; }

__device__ void rc_hash_event(RCtx* c, uint8_t kind, uint64_t ev, uint32_t op_index,
                              uint64_t key_f, uint64_t slot_f, uint8_t hk_f,
                              uint64_t hslot_f, int64_t val_f, uint64_t aux_f) {
    uint64_t h = c->ev_h;
    rfnv_bytes(&h, &kind, sizeof(uint8_t));
    rfnv_bytes(&h, &ev, sizeof(uint64_t));
    rfnv_bytes(&h, &op_index, sizeof(uint32_t));
    rfnv_bytes(&h, &key_f, sizeof(uint64_t));
    rfnv_bytes(&h, &slot_f, sizeof(uint64_t));
    rfnv_bytes(&h, &hk_f, sizeof(uint8_t));
    rfnv_bytes(&h, &hslot_f, sizeof(uint64_t));
    rfnv_bytes(&h, &val_f, sizeof(int64_t));
    rfnv_bytes(&h, &aux_f, sizeof(uint64_t));
    c->ev_h = h;
}
__device__ void rc_hash_read(RCtx* c, uint64_t read_id, uint64_t k, uint8_t found,
                             uint8_t src, uint64_t slot_f, uint64_t stash_iseq_f,
                             int64_t val_f, uint64_t vseq_f) {
    uint64_t h = c->rd_h;
    rfnv_bytes(&h, &read_id, sizeof(uint64_t));
    rfnv_bytes(&h, &k, sizeof(uint64_t));
    rfnv_bytes(&h, &found, sizeof(uint8_t));
    rfnv_bytes(&h, &src, sizeof(uint8_t));
    rfnv_bytes(&h, &slot_f, sizeof(uint64_t));
    rfnv_bytes(&h, &stash_iseq_f, sizeof(uint64_t));
    rfnv_bytes(&h, &val_f, sizeof(int64_t));
    rfnv_bytes(&h, &vseq_f, sizeof(uint64_t));
    c->rd_h = h;
}

__device__ int rc_find_live_table(RCtx* c, uint64_t k) {
    int h0 = rc_home0(c, k);
    for (int off = 0; off < c->neighborhood; ++off) {
        int s = (h0 + off) % c->slot_count;
        if (c->st[s] == CKT_LIVE && c->key[s] == k) return s;
    }
    int h1 = rc_home1(c, k);
    for (int off = 0; off < c->neighborhood; ++off) {
        int s = (h1 + off) % c->slot_count;
        if (c->st[s] == CKT_LIVE && c->key[s] == k) return s;
    }
    return -1;
}

__device__ int rc_find_stash(RCtx* c, uint64_t k) {
    for (int i = 0; i < c->stash_size; ++i)
        if (c->s_key[i] == k) return i;
    return -1;
}

__device__ void rc_stash_erase(RCtx* c, int idx) {
    for (int i = idx; i < c->stash_size - 1; ++i) {
        c->s_key[i] = c->s_key[i + 1];
        c->s_val[i] = c->s_val[i + 1];
        c->s_iseq[i] = c->s_iseq[i + 1];
        c->s_vseq[i] = c->s_vseq[i + 1];
    }
    c->stash_size -= 1;
}

__device__ int rc_find_tombstone_for_key(RCtx* c, uint64_t k) {
    int best = -1; uint64_t best_ts = 0;
    for (int s = 0; s < c->slot_count; ++s) {
        if (c->st[s] != CKT_TOMBSTONE || c->key[s] != k) continue;
        uint64_t ts = c->aux[s];
        if (best < 0 || ts < best_ts || (ts == best_ts && s < best)) {
            best = s; best_ts = ts;
        }
    }
    return best;
}

__device__ int rc_insert_with_home(RCtx* c, int hk_attempt, uint64_t pk,
                                    int64_t pv, uint64_t p_iseq,
                                    bool use_existing_vseq, uint64_t existing_vseq,
                                    uint32_t op_index) {
    int target_home = (hk_attempt == CKT_HOME0) ? rc_home0(c, pk) : rc_home1(c, pk);

    int vacancy = -1;
    for (int off = 0; off < c->slot_count; ++off) {
        int s = (target_home + off) % c->slot_count;
        if ((c->st[s] == CKT_EMPTY || c->st[s] == CKT_TOMBSTONE) &&
            c->pin_count[rc_page_of(c, s)] == 0) {
            vacancy = s; break;
        }
    }
    if (vacancy < 0) return -1;

    int disp = 0;
    while (rc_dist(c, target_home, vacancy) >= c->neighborhood) {
        if (disp >= c->max_disp) return -1;
        int chosen = -1;
        for (int d = 1; d < c->neighborhood; ++d) {
            int cc = (((vacancy - d) % c->slot_count) + c->slot_count) % c->slot_count;
            if (c->st[cc] != CKT_LIVE) continue;
            if (c->pin_count[rc_page_of(c, cc)] != 0) continue;
            int chome = (int)c->home_slot[cc];
            if (rc_dist(c, chome, vacancy) < c->neighborhood) { chosen = cc; break; }
        }
        if (chosen < 0) return -1;

        if (c->st[vacancy] == CKT_TOMBSTONE) c->tomb_count -= 1;

        c->st[vacancy] = CKT_LIVE;
        c->key[vacancy] = c->key[chosen];
        c->val[vacancy] = c->val[chosen];
        c->home_kind[vacancy] = c->home_kind[chosen];
        c->home_slot[vacancy] = c->home_slot[chosen];
        c->iseq[vacancy] = c->iseq[chosen];
        c->aux[vacancy] = c->aux[chosen];

        uint64_t ev = rc_next_ev(c);
        rc_hash_event(c, CKT_EV_RELOCATE_SLOT, ev, op_index, c->key[vacancy],
                      (uint64_t)vacancy, c->home_kind[vacancy],
                      c->home_slot[vacancy], c->val[vacancy], (uint64_t)chosen);
        c->counts[10] += 1;

        c->st[chosen] = CKT_EMPTY;
        c->key[chosen] = 0; c->val[chosen] = 0;
        c->home_kind[chosen] = 0; c->home_slot[chosen] = 0;
        c->iseq[chosen] = 0; c->aux[chosen] = 0;

        vacancy = chosen;
        disp += 1;
    }

    if (c->st[vacancy] == CKT_TOMBSTONE) {
        uint64_t old_ts = c->aux[vacancy];
        uint64_t ev = rc_next_ev(c);
        rc_hash_event(c, CKT_EV_REUSE_TOMBSTONE, ev, op_index, pk,
                      (uint64_t)vacancy, (uint8_t)hk_attempt,
                      (uint64_t)target_home, CKT_I64_MIN, old_ts);
        c->counts[11] += 1;
        c->tomb_count -= 1;
    }

    uint64_t ev = rc_next_ev(c);
    uint64_t vseq = use_existing_vseq ? existing_vseq : ev;
    c->st[vacancy] = CKT_LIVE;
    c->key[vacancy] = pk;
    c->val[vacancy] = pv;
    c->home_kind[vacancy] = (uint8_t)hk_attempt;
    c->home_slot[vacancy] = (uint64_t)target_home;
    c->iseq[vacancy] = p_iseq;
    c->aux[vacancy] = vseq;

    rc_hash_event(c, CKT_EV_PUT_INSERT, ev, op_index, pk, (uint64_t)vacancy,
                  (uint8_t)hk_attempt, (uint64_t)target_home, pv, p_iseq);
    c->counts[2] += 1;
    return vacancy;
}

__device__ void rc_sweep_step(RCtx* c, uint32_t op_index, int limit,
                              bool until_threshold) {
    int removed = 0;
    while (true) {
        if (until_threshold) {
            if (c->tomb_count <= c->max_tombstones) break;
        } else {
            if (removed >= limit) break;
        }
        int target = -1; uint64_t best_ts = 0;
        for (int s = 0; s < c->slot_count; ++s) {
            if (c->st[s] != CKT_TOMBSTONE) continue;
            if (c->pin_count[rc_page_of(c, s)] != 0) continue;
            uint64_t ts = c->aux[s];
            if (target < 0 || ts < best_ts || (ts == best_ts && s < target)) {
                target = s; best_ts = ts;
            }
        }
        if (target < 0) break;

        uint64_t ev = rc_next_ev(c);
        rc_hash_event(c, CKT_EV_TOMBSTONE_SWEEP, ev, op_index, c->key[target],
                      (uint64_t)target, c->home_kind[target],
                      c->home_slot[target], c->val[target], c->aux[target]);
        c->counts[12] += 1;

        c->st[target] = CKT_EMPTY;
        c->key[target] = 0; c->val[target] = 0;
        c->home_kind[target] = 0; c->home_slot[target] = 0;
        c->iseq[target] = 0; c->aux[target] = 0;
        c->tomb_count -= 1;
        removed += 1;
    }
}

__device__ void rc_op_get(RCtx* c, uint32_t op_index, uint64_t read_id, uint64_t k) {
    int slot = rc_find_live_table(c, k);
    uint8_t found, src;
    uint64_t slot_f, stash_iseq_f, vseq_f; int64_t val_f;
    if (slot >= 0) {
        found = 1; src = CKT_SRC_TABLE;
        slot_f = (uint64_t)slot; stash_iseq_f = CKT_U64_MAX;
        val_f = c->val[slot]; vseq_f = c->aux[slot];
        c->counts[0] += 1;
    } else {
        int si = rc_find_stash(c, k);
        if (si >= 0) {
            found = 1; src = CKT_SRC_STASH;
            slot_f = CKT_U64_MAX; stash_iseq_f = c->s_iseq[si];
            val_f = c->s_val[si]; vseq_f = c->s_vseq[si];
            c->counts[0] += 1;
        } else {
            found = 0; src = CKT_SRC_NONE;
            slot_f = CKT_U64_MAX; stash_iseq_f = CKT_U64_MAX;
            val_f = CKT_I64_MIN; vseq_f = CKT_U64_MAX;
            c->counts[1] += 1;
        }
    }
    uint64_t ev = rc_next_ev(c);
    rc_hash_event(c, CKT_EV_GET_RESULT, ev, op_index, k, slot_f, CKT_HK_NONE,
                  CKT_U64_MAX, val_f, (uint64_t)src);
    rc_hash_read(c, read_id, k, found, src, slot_f, stash_iseq_f, val_f, vseq_f);
}

__device__ void rc_op_put(RCtx* c, uint32_t op_index, uint64_t k, int64_t value) {
    int slot = rc_find_live_table(c, k);
    if (slot >= 0) {
        uint64_t ev = rc_next_ev(c);
        c->val[slot] = value;
        c->aux[slot] = ev;
        rc_hash_event(c, CKT_EV_UPDATE_EXISTING, ev, op_index, k, (uint64_t)slot,
                      c->home_kind[slot], c->home_slot[slot], value, ev);
        c->counts[3] += 1;
        return;
    }
    int si = rc_find_stash(c, k);
    if (si >= 0) {
        uint64_t ev = rc_next_ev(c);
        c->s_val[si] = value;
        c->s_vseq[si] = ev;
        rc_hash_event(c, CKT_EV_UPDATE_STASH, ev, op_index, k, CKT_U64_MAX,
                      CKT_HK_NONE, CKT_U64_MAX, value, c->s_iseq[si]);
        c->counts[3] += 1;
        return;
    }
    int ts = rc_find_tombstone_for_key(c, k);
    if (ts >= 0) {
        uint64_t ev = rc_next_ev(c);
        c->st[ts] = CKT_LIVE;
        c->val[ts] = value;
        c->aux[ts] = ev;
        c->tomb_count -= 1;
        rc_hash_event(c, CKT_EV_RESURRECT_TOMBSTONE, ev, op_index, k,
                      (uint64_t)ts, c->home_kind[ts], c->home_slot[ts], value,
                      c->iseq[ts]);
        c->counts[4] += 1;
        return;
    }
    uint64_t new_iseq = c->insert_seq_next++;
    int r = rc_insert_with_home(c, CKT_HOME0, k, value, new_iseq, false, 0, op_index);
    if (r < 0) r = rc_insert_with_home(c, CKT_HOME1, k, value, new_iseq, false, 0, op_index);
    if (r >= 0) return;
    if (c->stash_size < c->stash_capacity) {
        uint64_t ev = rc_next_ev(c);
        int idx = c->stash_size;
        c->s_key[idx] = k; c->s_val[idx] = value;
        c->s_iseq[idx] = new_iseq; c->s_vseq[idx] = ev;
        c->stash_size += 1;
        rc_hash_event(c, CKT_EV_PUT_STASH, ev, op_index, k, CKT_U64_MAX,
                      CKT_HK_NONE, CKT_U64_MAX, value, new_iseq);
        c->counts[5] += 1;
    } else {
        uint64_t ev = rc_next_ev(c);
        rc_hash_event(c, CKT_EV_PUT_OOM, ev, op_index, k, CKT_U64_MAX,
                      CKT_HK_NONE, CKT_U64_MAX, value, new_iseq);
        c->counts[6] += 1;
    }
}

__device__ void rc_op_delete(RCtx* c, uint32_t op_index, uint64_t k) {
    int slot = rc_find_live_table(c, k);
    if (slot >= 0) {
        uint64_t ev = rc_next_ev(c);
        c->st[slot] = CKT_TOMBSTONE;
        c->aux[slot] = ev;
        c->tomb_count += 1;
        rc_hash_event(c, CKT_EV_DELETE_TABLE, ev, op_index, k, (uint64_t)slot,
                      c->home_kind[slot], c->home_slot[slot], c->val[slot], ev);
        c->counts[7] += 1;
        if (c->tomb_count > c->max_tombstones)
            rc_sweep_step(c, op_index, 0, true);
        return;
    }
    int si = rc_find_stash(c, k);
    if (si >= 0) {
        uint64_t ev = rc_next_ev(c);
        int64_t removed_val = c->s_val[si];
        uint64_t removed_iseq = c->s_iseq[si];
        rc_hash_event(c, CKT_EV_DELETE_STASH, ev, op_index, k, CKT_U64_MAX,
                      CKT_HK_NONE, CKT_U64_MAX, removed_val, removed_iseq);
        c->counts[8] += 1;
        rc_stash_erase(c, si);
        return;
    }
    uint64_t ev = rc_next_ev(c);
    rc_hash_event(c, CKT_EV_DELETE_MISS, ev, op_index, k, CKT_U64_MAX,
                  CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN, CKT_U64_MAX);
    c->counts[9] += 1;
}

__device__ void rc_op_pin(RCtx* c, uint32_t op_index, int64_t page_arg) {
    if (page_arg < 0 || page_arg >= c->n_pages ||
        c->pin_count[page_arg] == CKT_U64_MAX) {
        uint64_t ev = rc_next_ev(c);
        rc_hash_event(c, CKT_EV_INVALID, ev, op_index, CKT_U64_MAX, CKT_U64_MAX,
                      CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN, (uint64_t)page_arg);
        c->counts[16] += 1;
        return;
    }
    c->pin_count[page_arg] += 1;
    uint64_t ev = rc_next_ev(c);
    rc_hash_event(c, CKT_EV_PIN_PAGE_OK, ev, op_index, CKT_U64_MAX, CKT_U64_MAX,
                  CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN, (uint64_t)page_arg);
    c->counts[14] += 1;
}

__device__ void rc_op_unpin(RCtx* c, uint32_t op_index, int64_t page_arg) {
    if (page_arg < 0 || page_arg >= c->n_pages || c->pin_count[page_arg] == 0) {
        uint64_t ev = rc_next_ev(c);
        rc_hash_event(c, CKT_EV_INVALID, ev, op_index, CKT_U64_MAX, CKT_U64_MAX,
                      CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN, (uint64_t)page_arg);
        c->counts[16] += 1;
        return;
    }
    c->pin_count[page_arg] -= 1;
    uint64_t ev = rc_next_ev(c);
    rc_hash_event(c, CKT_EV_UNPIN_PAGE_OK, ev, op_index, CKT_U64_MAX, CKT_U64_MAX,
                  CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN, (uint64_t)page_arg);
    c->counts[15] += 1;
}

__device__ void rc_op_replay(RCtx* c, uint32_t op_index, int64_t limit) {
    int lim = (limit < 0) ? 0 : (int)limit;
    int success = 0;
    int i = 0;
    while (i < c->stash_size) {
        if (success >= lim) break;
        uint64_t k = c->s_key[i];
        int64_t v = c->s_val[i];
        uint64_t es = c->s_iseq[i];
        uint64_t vs = c->s_vseq[i];
        int r = rc_insert_with_home(c, CKT_HOME0, k, v, es, true, vs, op_index);
        if (r < 0) r = rc_insert_with_home(c, CKT_HOME1, k, v, es, true, vs, op_index);
        if (r >= 0) {
            uint64_t ev = rc_next_ev(c);
            rc_hash_event(c, CKT_EV_STASH_REPLAY_OK, ev, op_index, k,
                          (uint64_t)r, c->home_kind[r], c->home_slot[r], v, es);
            c->counts[13] += 1;
            rc_stash_erase(c, i);
            success += 1;
        } else {
            i += 1;
        }
    }
}

__global__ void ckt_ref_kernel(
    int slot_count, int page_size, int neighborhood, int max_disp,
    int stash_capacity, int max_tombstones, int n_pages,
    uint64_t seed0, uint64_t seed1,
    int num_ops,
    const int32_t* __restrict__ op_type,
    const int64_t* __restrict__ a0,
    const int64_t* __restrict__ a1,
    uint8_t* st, uint64_t* key, int64_t* val, uint8_t* home_kind,
    uint64_t* home_slot, uint64_t* iseq, uint64_t* aux,
    uint64_t* pin_count,
    uint64_t* s_key, int64_t* s_val, uint64_t* s_iseq, uint64_t* s_vseq,
    int32_t* scalars_i, uint64_t* scalars_u,
    int32_t* out_counts, uint64_t* out_ev_h, uint64_t* out_rd_h,
    uint64_t* out_slot_h, uint64_t* out_stash_h, uint64_t* out_page_h) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    RCtx c;
    c.slot_count = slot_count; c.page_size = page_size;
    c.neighborhood = neighborhood; c.max_disp = max_disp;
    c.stash_capacity = stash_capacity; c.max_tombstones = max_tombstones;
    c.n_pages = n_pages; c.seed0 = seed0; c.seed1 = seed1;
    c.st = st; c.key = key; c.val = val; c.home_kind = home_kind;
    c.home_slot = home_slot; c.iseq = iseq; c.aux = aux;
    c.pin_count = pin_count;
    c.s_key = s_key; c.s_val = s_val; c.s_iseq = s_iseq; c.s_vseq = s_vseq;
    c.stash_size = scalars_i[0];
    c.tomb_count = scalars_i[1];
    c.event_seq = scalars_u[0];
    c.insert_seq_next = scalars_u[1];
    c.counts = out_counts;
    c.ev_h = CKT_FNV_OFFSET;
    c.rd_h = CKT_FNV_OFFSET;

    for (int i = 0; i < CKT_NUM_COUNTS; ++i) out_counts[i] = 0;

    for (int o = 0; o < num_ops; ++o) {
        uint32_t op_index = (uint32_t)o;
        int32_t ot = op_type[o];
        int64_t x0 = a0[o];
        int64_t x1 = a1[o];
        switch (ot) {
            case CKT_OP_GET: rc_op_get(&c, op_index, (uint64_t)x0, (uint64_t)x1); break;
            case CKT_OP_PUT: rc_op_put(&c, op_index, (uint64_t)x0, x1); break;
            case CKT_OP_DELETE: rc_op_delete(&c, op_index, (uint64_t)x0); break;
            case CKT_OP_PIN_PAGE: rc_op_pin(&c, op_index, x0); break;
            case CKT_OP_UNPIN_PAGE: rc_op_unpin(&c, op_index, x0); break;
            case CKT_OP_SWEEP_TOMBSTONES:
                rc_sweep_step(&c, op_index, (x0 < 0) ? 0 : (int)x0, false); break;
            case CKT_OP_REPLAY_STASH: rc_op_replay(&c, op_index, x0); break;
            default: {
                uint64_t ev = rc_next_ev(&c);
                rc_hash_event(&c, CKT_EV_INVALID, ev, op_index, CKT_U64_MAX,
                              CKT_U64_MAX, CKT_HK_NONE, CKT_U64_MAX, CKT_I64_MIN,
                              CKT_U64_MAX);
                c.counts[16] += 1;
            } break;
        }
    }

    scalars_i[0] = c.stash_size;
    scalars_i[1] = c.tomb_count;
    scalars_u[0] = c.event_seq;
    scalars_u[1] = c.insert_seq_next;

    out_ev_h[0] = c.ev_h;
    out_rd_h[0] = c.rd_h;

    // slot_state_hash
    uint64_t h = CKT_FNV_OFFSET;
    for (int s = 0; s < slot_count; ++s) {
        uint64_t su = (uint64_t)s; uint8_t state = st[s];
        rfnv_bytes(&h, &su, sizeof(uint64_t));
        rfnv_bytes(&h, &state, sizeof(uint8_t));
        if (state == CKT_EMPTY) continue;
        rfnv_bytes(&h, &key[s], sizeof(uint64_t));
        rfnv_bytes(&h, &val[s], sizeof(int64_t));
        rfnv_bytes(&h, &home_kind[s], sizeof(uint8_t));
        rfnv_bytes(&h, &home_slot[s], sizeof(uint64_t));
        rfnv_bytes(&h, &iseq[s], sizeof(uint64_t));
        rfnv_bytes(&h, &aux[s], sizeof(uint64_t));
    }
    out_slot_h[0] = h;

    uint64_t hs = CKT_FNV_OFFSET;
    for (int i = 0; i < c.stash_size; ++i) {
        rfnv_bytes(&hs, &s_key[i], sizeof(uint64_t));
        rfnv_bytes(&hs, &s_val[i], sizeof(int64_t));
        rfnv_bytes(&hs, &s_iseq[i], sizeof(uint64_t));
        rfnv_bytes(&hs, &s_vseq[i], sizeof(uint64_t));
    }
    out_stash_h[0] = hs;

    uint64_t hp = CKT_FNV_OFFSET;
    for (int p = 0; p < n_pages; ++p) {
        uint64_t pu = (uint64_t)p; uint64_t pc = pin_count[p];
        rfnv_bytes(&hp, &pu, sizeof(uint64_t));
        rfnv_bytes(&hp, &pc, sizeof(uint64_t));
    }
    out_page_h[0] = hp;
}

// ---------------------------------------------------------------------------
static cudaError_t ckt_ref_reset(CktRefState* st, cudaStream_t stream) {
    cudaError_t e;
    e = cudaMemsetAsync(st->st, 0, sizeof(uint8_t) * st->slot_count, stream); if (e) return e;
    e = cudaMemsetAsync(st->key, 0, sizeof(uint64_t) * st->slot_count, stream); if (e) return e;
    e = cudaMemsetAsync(st->val, 0, sizeof(int64_t) * st->slot_count, stream); if (e) return e;
    e = cudaMemsetAsync(st->home_kind, 0, sizeof(uint8_t) * st->slot_count, stream); if (e) return e;
    e = cudaMemsetAsync(st->home_slot, 0, sizeof(uint64_t) * st->slot_count, stream); if (e) return e;
    e = cudaMemsetAsync(st->iseq, 0, sizeof(uint64_t) * st->slot_count, stream); if (e) return e;
    e = cudaMemsetAsync(st->aux, 0, sizeof(uint64_t) * st->slot_count, stream); if (e) return e;
    e = cudaMemsetAsync(st->pin_count, 0, sizeof(uint64_t) * st->n_pages, stream); if (e) return e;
    // scalars: stash_size=0, tomb_count=0, event_seq=0, insert_seq_next=1
    int32_t si_init[2] = {0, 0};
    uint64_t su_init[2] = {0, 1};
    e = cudaMemcpyAsync(st->scalars_i, si_init, sizeof(si_init), cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(st->scalars_u, su_init, sizeof(su_init), cudaMemcpyHostToDevice, stream); if (e) return e;
    return cudaSuccess;
}

extern "C" size_t solution_workspace_bytes(const CktProblemSpec* spec) {
    if (!ckt_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(const CktProblemSpec* spec, void** state_out,
                                     cudaStream_t stream) {
    if (!ckt_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;

    CktRefState* st = (CktRefState*)malloc(sizeof(CktRefState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(CktRefState));
    memcpy(&st->spec, spec, sizeof(CktProblemSpec));
    st->slot_count = spec->slot_count;
    st->page_size = spec->page_size;
    st->neighborhood = spec->neighborhood;
    st->max_disp = spec->max_displacements_per_home;
    st->stash_capacity = spec->stash_capacity;
    st->max_tombstones = spec->max_tombstones;
    st->n_pages = ckt_n_pages(spec);

    const int sc = st->slot_count;
    const int cap = st->stash_capacity > 0 ? st->stash_capacity : 1;

    cudaError_t e = cudaSuccess;
    e = cudaMalloc((void**)&st->st, sizeof(uint8_t) * sc); if (e) goto fail;
    e = cudaMalloc((void**)&st->key, sizeof(uint64_t) * sc); if (e) goto fail;
    e = cudaMalloc((void**)&st->val, sizeof(int64_t) * sc); if (e) goto fail;
    e = cudaMalloc((void**)&st->home_kind, sizeof(uint8_t) * sc); if (e) goto fail;
    e = cudaMalloc((void**)&st->home_slot, sizeof(uint64_t) * sc); if (e) goto fail;
    e = cudaMalloc((void**)&st->iseq, sizeof(uint64_t) * sc); if (e) goto fail;
    e = cudaMalloc((void**)&st->aux, sizeof(uint64_t) * sc); if (e) goto fail;
    e = cudaMalloc((void**)&st->pin_count, sizeof(uint64_t) * st->n_pages); if (e) goto fail;
    e = cudaMalloc((void**)&st->s_key, sizeof(uint64_t) * cap); if (e) goto fail;
    e = cudaMalloc((void**)&st->s_val, sizeof(int64_t) * cap); if (e) goto fail;
    e = cudaMalloc((void**)&st->s_iseq, sizeof(uint64_t) * cap); if (e) goto fail;
    e = cudaMalloc((void**)&st->s_vseq, sizeof(uint64_t) * cap); if (e) goto fail;
    e = cudaMalloc((void**)&st->scalars_i, sizeof(int32_t) * 2); if (e) goto fail;
    e = cudaMalloc((void**)&st->scalars_u, sizeof(uint64_t) * 2); if (e) goto fail;

    e = ckt_ref_reset(st, stream); if (e) goto fail;

    *state_out = st;
    return cudaSuccess;
fail:
    if (st->st) cudaFree(st->st);
    if (st->key) cudaFree(st->key);
    if (st->val) cudaFree(st->val);
    if (st->home_kind) cudaFree(st->home_kind);
    if (st->home_slot) cudaFree(st->home_slot);
    if (st->iseq) cudaFree(st->iseq);
    if (st->aux) cudaFree(st->aux);
    if (st->pin_count) cudaFree(st->pin_count);
    if (st->s_key) cudaFree(st->s_key);
    if (st->s_val) cudaFree(st->s_val);
    if (st->s_iseq) cudaFree(st->s_iseq);
    if (st->s_vseq) cudaFree(st->s_vseq);
    if (st->scalars_i) cudaFree(st->scalars_i);
    if (st->scalars_u) cudaFree(st->scalars_u);
    free(st);
    return e;
}

extern "C" cudaError_t solution_run(void* state, const CktRunSpec* run,
                                    const void* inputs_void, void* outputs_void,
                                    void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    (void)workspace;
    if (!state || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 1) return cudaErrorInvalidValue;

    CktRefState* st = (CktRefState*)state;
    if (!ckt_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const CktInputs* in = (const CktInputs*)inputs_void;
    CktOutputs* out = (CktOutputs*)outputs_void;
    if (run->num_ops > 0 && (!in || !in->op_type || !in->a0 || !in->a1))
        return cudaErrorInvalidValue;
    if (!out->counts || !out->op_event_hash || !out->read_hash ||
        !out->slot_state_hash || !out->stash_hash || !out->page_hash)
        return cudaErrorInvalidValue;

    ckt_ref_kernel<<<1, 1, 0, stream>>>(
        st->slot_count, st->page_size, st->neighborhood, st->max_disp,
        st->stash_capacity, st->max_tombstones, st->n_pages,
        st->spec.seed0, st->spec.seed1, run->num_ops,
        run->num_ops > 0 ? in->op_type : nullptr,
        run->num_ops > 0 ? in->a0 : nullptr,
        run->num_ops > 0 ? in->a1 : nullptr,
        st->st, st->key, st->val, st->home_kind, st->home_slot, st->iseq, st->aux,
        st->pin_count, st->s_key, st->s_val, st->s_iseq, st->s_vseq,
        st->scalars_i, st->scalars_u,
        out->counts, out->op_event_hash, out->read_hash,
        out->slot_state_hash, out->stash_hash, out->page_hash);

    cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) return e;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return ckt_ref_reset((CktRefState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    CktRefState* st = (CktRefState*)state;
    if (st->st) cudaFree(st->st);
    if (st->key) cudaFree(st->key);
    if (st->val) cudaFree(st->val);
    if (st->home_kind) cudaFree(st->home_kind);
    if (st->home_slot) cudaFree(st->home_slot);
    if (st->iseq) cudaFree(st->iseq);
    if (st->aux) cudaFree(st->aux);
    if (st->pin_count) cudaFree(st->pin_count);
    if (st->s_key) cudaFree(st->s_key);
    if (st->s_val) cudaFree(st->s_val);
    if (st->s_iseq) cudaFree(st->s_iseq);
    if (st->s_vseq) cudaFree(st->s_vseq);
    if (st->scalars_i) cudaFree(st->scalars_i);
    if (st->scalars_u) cudaFree(st->scalars_u);
    free(st);
}
