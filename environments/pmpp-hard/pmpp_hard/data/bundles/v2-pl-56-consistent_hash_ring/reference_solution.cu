// PMPP_CANARY_56_f194c72499 -- held-out canary; MUST NOT appear in any submission
// file: consistent_hash_ring_reference.cu
//
// Reference device implementation of T56. Single-block, single-thread kernel
// processing exactly one op per solution_run over persistent device state.
// Independent of naive.cu (different data layout: this keeps a tokens array kept
// sorted on demand via index sort with an explicit comparator helper, replicas
// stored inline per key, moves in a compacting array).

#include "consistent_hash_ring_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace chr_ref {

struct Node {
    uint64_t node_id;
    int32_t state;
    uint64_t capacity;
    uint64_t used_slots;
    uint64_t node_seq;
    uint64_t last_state_seq;
};
struct Token {
    uint64_t token_pos;
    uint64_t token_seq;
    uint64_t node_id;
    uint32_t vnode_ordinal;
};
struct Replica {
    uint32_t rank;
    uint64_t node_id;
    uint8_t kind;
    uint64_t hint_for_node;
    uint8_t serving;
    uint64_t repair_seq;
};
struct Move {
    uint64_t move_seq;
    uint64_t key_id;
    uint8_t task_kind;
    uint64_t from_node;
    uint64_t to_node;
    uint32_t rank;
    uint64_t hint_for_node;
    uint64_t target_version_seq;
    uint8_t created_reason;
};

// Persistent device state. Replicas stored in a flat per-key block.
struct State {
    // spec mirror
    int32_t replication_factor;
    int32_t preference_list_extra;
    int32_t max_nodes;
    int32_t max_vnodes;
    int32_t max_keys;
    int32_t max_replicas_per_key;
    int32_t max_move_tasks;
    uint64_t default_node_capacity;
    uint64_t ring_seed;
    uint64_t key_seed;

    uint64_t event_seq;
    uint64_t node_seq_next;
    uint64_t token_seq_next;
    uint64_t key_seq_next;
    uint64_t version_seq_next;
    uint64_t move_seq_next;

    int32_t node_count;
    int32_t token_count;
    int32_t key_count;
    int32_t move_count;

    Node* nodes;          // max_nodes
    Token* tokens;        // max_vnodes

    // keys
    uint64_t* k_key_id;
    uint64_t* k_key_hash;
    int64_t* k_value;
    uint64_t* k_version_seq;
    uint8_t* k_deleted;
    uint64_t* k_key_seq;
    int32_t* k_rep_count;   // per key
    Replica* replicas;      // max_keys * max_replicas_per_key

    Move* moves;            // max_move_tasks

    ChrCounters* ctr;
    uint64_t* ring_event_hash;
    uint64_t* lookup_hash;

    // scratch
    int32_t* scratch_idx;   // generic index scratch, size max(max_vnodes,max_keys,max_move_tasks)

    // Cooperative-sort caches (filled by the whole block at kernel entry; the
    // ordered set is CONSTANT for the remainder of the op, so thread 0 reuses
    // these instead of re-running an O(n^2) selection sort per lookup). Because
    // every one of these is a strict TOTAL order, the parallel rank-sort below
    // yields the exact same permutation a selection sort would -> byte-identical.
    int32_t* ring_cache;    // token indices in RING ORDER
    int32_t* keys_cache;    // key indices in KEYS ORDER
    int32_t* node_cache;    // node indices by node_id asc (hash_kernel only)
    int32_t* move_cache;    // move indices by move_seq asc (hash_kernel only)
};

__device__ __forceinline__ uint64_t fnvb(uint64_t h, uint8_t b) { h ^= (uint64_t)b; h *= 1099511628211ULL; return h; }
__device__ void fnv(uint64_t* h, const void* p, int n) {
    const uint8_t* b = (const uint8_t*)p; uint64_t v = *h;
    for (int i = 0; i < n; ++i) v = fnvb(v, b[i]);
    *h = v;
}
__device__ __forceinline__ void fu64(uint64_t* h, uint64_t x) { fnv(h, &x, 8); }
__device__ __forceinline__ void fu32(uint64_t* h, uint32_t x) { fnv(h, &x, 4); }
__device__ __forceinline__ void fu8(uint64_t* h, uint8_t x) { fnv(h, &x, 1); }
__device__ __forceinline__ void fi64(uint64_t* h, int64_t x) { fnv(h, &x, 8); }

__device__ uint64_t hash_pos(uint64_t seed, uint64_t a, uint64_t b) {
    uint64_t h = 1469598103934665603ULL; fu64(&h, seed); fu64(&h, a); fu64(&h, b); return h;
}
__device__ uint64_t hash_key(uint64_t seed, uint64_t a) {
    uint64_t h = 1469598103934665603ULL; fu64(&h, seed); fu64(&h, a); return h;
}

__device__ int find_node(State* s, uint64_t id) {
    for (int i = 0; i < s->node_count; ++i) if (s->nodes[i].node_id == id) return i;
    return -1;
}
__device__ int find_key(State* s, uint64_t id) {
    for (int i = 0; i < s->key_count; ++i) if (s->k_key_id[i] == id) return i;
    return -1;
}
__device__ bool pos_taken(State* s, uint64_t pos) {
    for (int i = 0; i < s->token_count; ++i) if (s->tokens[i].token_pos == pos) return true;
    return false;
}
__device__ Replica* rep_base(State* s, int ki) { return s->replicas + (size_t)ki * s->max_replicas_per_key; }
__device__ int rep_on(State* s, int ki, uint64_t nid) {
    Replica* r = rep_base(s, ki);
    for (int i = 0; i < s->k_rep_count[ki]; ++i) if (r[i].node_id == nid) return i;
    return -1;
}
__device__ int serving_on(State* s, int ki, uint64_t nid) {
    Replica* r = rep_base(s, ki);
    for (int i = 0; i < s->k_rep_count[ki]; ++i) if (r[i].node_id == nid && r[i].serving) return i;
    return -1;
}
__device__ void rep_erase(State* s, int ki, int idx) {
    Replica* r = rep_base(s, ki);
    int n = s->k_rep_count[ki];
    for (int i = idx; i < n - 1; ++i) r[i] = r[i + 1];
    s->k_rep_count[ki] = n - 1;
}

__device__ bool token_less(const Token& x, const Token& y) {
    if (x.token_pos != y.token_pos) return x.token_pos < y.token_pos;
    if (x.token_seq != y.token_seq) return x.token_seq < y.token_seq;
    if (x.node_id != y.node_id) return x.node_id < y.node_id;
    return x.vnode_ordinal < y.vnode_ordinal;
}

__device__ __forceinline__ bool key_less(State* s, int a, int b) {
    if (s->k_key_hash[a] != s->k_key_hash[b]) return s->k_key_hash[a] < s->k_key_hash[b];
    return s->k_key_id[a] < s->k_key_id[b];
}

// ---- Cooperative rank-sorts (called by EVERY thread in the single block) -----
// For a strict total order, rank(i) = #{ j : elem[j] < elem[i] } is a bijection
// onto [0,n); out[rank(i)] = i reproduces the exact selection-sort permutation.
// Work is O(n^2) but spread across the block, so parallel depth is ~O(n).
__device__ void coop_ring_order(State* s, int32_t* out) {
    int n = s->token_count;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        Token ti = s->tokens[i];
        int rank = 0;
        for (int j = 0; j < n; ++j) if (token_less(s->tokens[j], ti)) ++rank;
        out[rank] = i;
    }
}
__device__ void coop_keys_order(State* s, int32_t* out) {
    int n = s->key_count;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int rank = 0;
        for (int j = 0; j < n; ++j) if (key_less(s, j, i)) ++rank;
        out[rank] = i;
    }
}
__device__ void coop_node_order(State* s, int32_t* out) {
    int n = s->node_count;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        uint64_t id = s->nodes[i].node_id;
        int rank = 0;
        for (int j = 0; j < n; ++j) if (s->nodes[j].node_id < id) ++rank;
        out[rank] = i;
    }
}
__device__ void coop_move_order(State* s, int32_t* out) {
    int n = s->move_count;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        uint64_t sq = s->moves[i].move_seq;
        int rank = 0;
        for (int j = 0; j < n; ++j) if (s->moves[j].move_seq < sq) ++rank;
        out[rank] = i;
    }
}

// ring_order/keys_order are now thread-0 copies from the block-filled caches
// (the ordered set is constant across the rest of the op). Interface preserved
// so the op logic below is byte-for-byte unchanged.
__device__ void ring_order(State* s, int32_t* out) {
    int n = s->token_count;
    for (int i = 0; i < n; ++i) out[i] = s->ring_cache[i];
}
__device__ void keys_order(State* s, int32_t* out) {
    int n = s->key_count;
    for (int i = 0; i < n; ++i) out[i] = s->keys_cache[i];
}

__device__ void emit(State* s, uint8_t kind, uint32_t op_index, uint64_t node_id, uint64_t key_id,
                      uint64_t token_pos, uint32_t rank, int64_t value, uint64_t aux) {
    s->event_seq += 1;
    uint64_t* h = s->ring_event_hash;
    fu8(h, kind); fu64(h, s->event_seq); fu32(h, op_index); fu64(h, node_id);
    fu64(h, key_id); fu64(h, token_pos); fu32(h, rank); fi64(h, value); fu64(h, aux);
}

__device__ void lk_record(State* s, uint64_t read_id, uint64_t key_id, uint8_t rk, uint32_t rank,
                           uint64_t node_id, uint8_t kind, uint64_t hint_for, int64_t value, uint64_t version) {
    uint64_t* h = s->lookup_hash;
    fu64(h, read_id); fu64(h, key_id); fu8(h, rk); fu32(h, rank); fu64(h, node_id);
    fu8(h, kind); fu64(h, hint_for); fi64(h, value); fu64(h, version);
}

// preference list -> pref[] node ids, returns count
__device__ int preference_list(State* s, uint64_t key_hash, int32_t* order_scratch, uint64_t* pref) {
    int m = s->token_count;
    if (m == 0) return 0;
    ring_order(s, order_scratch);
    int start = 0; bool found = false;
    for (int i = 0; i < m; ++i) {
        if (s->tokens[order_scratch[i]].token_pos >= key_hash) { start = i; found = true; break; }
    }
    if (!found) start = 0;
    int cap = s->replication_factor + s->preference_list_extra;
    int cnt = 0;
    for (int step = 0; step < m; ++step) {
        int ti = order_scratch[(start + step) % m];
        uint64_t nid = s->tokens[ti].node_id;
        int ni = find_node(s, nid);
        if (ni < 0) continue;
        int st = s->nodes[ni].state;
        if (st != CHR_NODE_ACTIVE && st != CHR_NODE_JOINING) continue;
        bool dup = false;
        for (int i = 0; i < cnt; ++i) if (pref[i] == nid) { dup = true; break; }
        if (dup) continue;
        pref[cnt++] = nid;
        if (cnt >= cap) break;
    }
    return cnt;
}

// desired = first RF ACTIVE nodes from pref
__device__ int desired_replicas(State* s, uint64_t key_hash, int32_t* order_scratch, uint64_t* pref, uint64_t* des) {
    int pc = preference_list(s, key_hash, order_scratch, pref);
    int dc = 0;
    for (int i = 0; i < pc; ++i) {
        int ni = find_node(s, pref[i]);
        if (ni >= 0 && s->nodes[ni].state == CHR_NODE_ACTIVE) {
            des[dc++] = pref[i];
            if (dc >= s->replication_factor) break;
        }
    }
    return dc;
}

__device__ bool node_full(State* s, int ni) { return s->nodes[ni].used_slots >= s->nodes[ni].capacity; }

__device__ uint32_t first_free_rank(State* s, int ki, uint64_t* des, int dc) {
    Replica* r = rep_base(s, ki);
    int rc = s->k_rep_count[ki];
    for (uint32_t rr = 0; rr < (uint32_t)dc; ++rr) {
        bool occ = false;
        for (int i = 0; i < rc; ++i) if (r[i].serving && r[i].rank == rr) { occ = true; break; }
        if (!occ) return rr;
    }
    uint32_t sc = 0;
    for (int i = 0; i < rc; ++i) if (r[i].serving) ++sc;
    return sc;
}

__device__ void enqueue_move(State* s, uint32_t op_index, uint64_t key_id, uint8_t task_kind,
                             uint64_t from_node, uint64_t to_node, uint32_t rank,
                             uint64_t hint_for_node, uint64_t target_version, uint8_t reason) {
    if (s->move_count >= s->max_move_tasks) return;
    Move& mv = s->moves[s->move_count];
    mv.move_seq = s->move_seq_next++;
    mv.key_id = key_id; mv.task_kind = task_kind; mv.from_node = from_node; mv.to_node = to_node;
    mv.rank = rank; mv.hint_for_node = hint_for_node; mv.target_version_seq = target_version;
    mv.created_reason = reason;
    s->move_count += 1;
    s->ctr->move_tasks_enqueued += 1;
    emit(s, CHR_EV_MOVE_ENQUEUE, op_index, to_node ? to_node : from_node, key_id,
         UINT64_MAX, rank, 0, mv.move_seq);
}

__device__ void invalid(State* s, uint32_t op_index) {
    s->ctr->invalid_count += 1;
    emit(s, CHR_EV_INVALID, op_index, 0, 0, UINT64_MAX, UINT32_MAX, INT64_MIN, 0);
}

__device__ void op_add_node(State* s, uint32_t op_index, uint64_t node_id, int64_t vnode_count, uint64_t capacity) {
    int ni = find_node(s, node_id);
    bool exists_nonrem = (ni >= 0 && s->nodes[ni].state != CHR_NODE_REMOVED);
    if (exists_nonrem || s->node_count >= s->max_nodes || vnode_count == 0 || vnode_count < 0 ||
        s->token_count + (int)vnode_count > s->max_vnodes) {
        invalid(s, op_index); return;
    }
    uint64_t cap = capacity == 0 ? s->default_node_capacity : capacity;
    int slot;
    if (ni >= 0) slot = ni; else { slot = s->node_count; s->node_count += 1; }
    Node& nd = s->nodes[slot];
    nd.node_id = node_id; nd.state = CHR_NODE_JOINING; nd.capacity = cap; nd.used_slots = 0;
    nd.node_seq = s->node_seq_next++; nd.last_state_seq = 0;

    for (uint32_t ord = 0; ord < (uint32_t)vnode_count; ++ord) {
        uint64_t pos = hash_pos(s->ring_seed, node_id, ord);
        while (pos_taken(s, pos)) pos += 1;
        Token& tk = s->tokens[s->token_count];
        tk.token_pos = pos; tk.token_seq = s->token_seq_next++; tk.node_id = node_id; tk.vnode_ordinal = ord;
        s->token_count += 1;
        s->ctr->vnode_added += 1;
        emit(s, CHR_EV_VNODE_ADD, op_index, node_id, 0, pos, ord, 0, tk.token_seq);
    }
    s->ctr->nodes_added += 1;
    emit(s, CHR_EV_NODE_ADD, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, nd.node_seq);
}

__device__ void op_activate(State* s, uint32_t op_index, uint64_t node_id, int32_t* ord_scr, int32_t* korder,
                            uint64_t* pref, uint64_t* des) {
    int ni = find_node(s, node_id);
    if (ni < 0 || s->nodes[ni].state != CHR_NODE_JOINING) { invalid(s, op_index); return; }
    s->nodes[ni].state = CHR_NODE_ACTIVE;
    s->ctr->nodes_activated += 1;
    emit(s, CHR_EV_NODE_ACTIVATE, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
    s->nodes[ni].last_state_seq = s->event_seq;

    keys_order(s, korder);
    for (int idx = 0; idx < s->key_count; ++idx) {
        int ki = korder[idx];
        if (s->k_deleted[ki]) continue;
        int dc = desired_replicas(s, s->k_key_hash[ki], ord_scr, pref, des);
        int rank = -1;
        for (int r = 0; r < dc; ++r) if (des[r] == node_id) { rank = r; break; }
        if (rank < 0) continue;
        if (serving_on(s, ki, node_id) >= 0) continue;
        enqueue_move(s, op_index, s->k_key_id[ki], CHR_TASK_ADD_REPLICA, 0, node_id, (uint32_t)rank,
                     0, s->k_version_seq[ki], CHR_REASON_JOIN);
    }
}

__device__ void op_start_leave(State* s, uint32_t op_index, uint64_t node_id, int32_t* ord_scr, int32_t* korder,
                               uint64_t* pref, uint64_t* des) {
    int ni = find_node(s, node_id);
    if (ni < 0 || s->nodes[ni].state != CHR_NODE_ACTIVE) { invalid(s, op_index); return; }
    s->nodes[ni].state = CHR_NODE_LEAVING;
    s->ctr->nodes_leaving += 1;
    emit(s, CHR_EV_NODE_LEAVE, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
    s->nodes[ni].last_state_seq = s->event_seq;

    keys_order(s, korder);
    for (int idx = 0; idx < s->key_count; ++idx) {
        int ki = korder[idx];
        if (s->k_deleted[ki]) continue;
        if (serving_on(s, ki, node_id) < 0) continue;
        int dc = desired_replicas(s, s->k_key_hash[ki], ord_scr, pref, des);
        int target_rank = -1; uint64_t target = 0;
        for (int r = 0; r < dc; ++r) if (serving_on(s, ki, des[r]) < 0) { target = des[r]; target_rank = r; break; }
        if (target_rank < 0) {
            s->ctr->no_target_count += 1;
            emit(s, CHR_EV_LEAVE_NO_TARGET, op_index, node_id, s->k_key_id[ki], UINT64_MAX, UINT32_MAX, 0, 0);
            continue;
        }
        enqueue_move(s, op_index, s->k_key_id[ki], CHR_TASK_ADD_REPLICA, 0, target, (uint32_t)target_rank,
                     0, s->k_version_seq[ki], CHR_REASON_LEAVE);
    }
}

__device__ void op_fail(State* s, uint32_t op_index, uint64_t node_id, int32_t* ord_scr, int32_t* korder,
                        uint64_t* pref, uint64_t* des) {
    int ni = find_node(s, node_id);
    if (ni < 0 || s->nodes[ni].state == CHR_NODE_DOWN || s->nodes[ni].state == CHR_NODE_REMOVED) {
        invalid(s, op_index); return;
    }
    s->nodes[ni].state = CHR_NODE_DOWN;
    s->ctr->nodes_failed += 1;
    emit(s, CHR_EV_NODE_FAIL, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
    s->nodes[ni].last_state_seq = s->event_seq;

    keys_order(s, korder);
    // release serving slots on failed node first
    for (int idx = 0; idx < s->key_count; ++idx) {
        int ki = korder[idx];
        int ri = rep_on(s, ki, node_id);
        if (ri >= 0 && rep_base(s, ki)[ri].serving) {
            rep_base(s, ki)[ri].serving = 0;
            if (s->nodes[ni].used_slots > 0) s->nodes[ni].used_slots -= 1;
        }
    }
    // enqueue hinted handoff
    for (int idx = 0; idx < s->key_count; ++idx) {
        int ki = korder[idx];
        if (s->k_deleted[ki]) continue;
        int ri = rep_on(s, ki, node_id);
        if (ri < 0) continue;
        int pc = preference_list(s, s->k_key_hash[ki], ord_scr, pref);
        uint64_t target = 0; bool have = false;
        for (int i = 0; i < pc; ++i) {
            int xi = find_node(s, pref[i]);
            if (xi < 0 || s->nodes[xi].state != CHR_NODE_ACTIVE) continue;
            if (serving_on(s, ki, pref[i]) >= 0) continue;
            target = pref[i]; have = true; break;
        }
        if (!have) {
            s->ctr->no_target_count += 1;
            emit(s, CHR_EV_FAIL_NO_HINT_TARGET, op_index, node_id, s->k_key_id[ki], UINT64_MAX, UINT32_MAX, 0, 0);
            continue;
        }
        int dc = desired_replicas(s, s->k_key_hash[ki], ord_scr, pref, des);
        uint32_t rank = first_free_rank(s, ki, des, dc);
        enqueue_move(s, op_index, s->k_key_id[ki], CHR_TASK_ADD_REPLICA, 0, target, rank, node_id,
                     s->k_version_seq[ki], CHR_REASON_FAIL);
    }
}

__device__ void op_recover(State* s, uint32_t op_index, uint64_t node_id, int32_t* korder) {
    int ni = find_node(s, node_id);
    if (ni < 0 || s->nodes[ni].state != CHR_NODE_DOWN) { invalid(s, op_index); return; }
    s->nodes[ni].state = CHR_NODE_ACTIVE;
    s->ctr->nodes_recovered += 1;
    emit(s, CHR_EV_NODE_RECOVER, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
    s->nodes[ni].last_state_seq = s->event_seq;

    keys_order(s, korder);
    for (int idx = 0; idx < s->key_count; ++idx) {
        int ki = korder[idx];
        Replica* r = rep_base(s, ki);
        for (int i = 0; i < s->k_rep_count[ki]; ++i) {
            if (r[i].kind == CHR_KIND_HINTED && r[i].hint_for_node == node_id) {
                enqueue_move(s, op_index, s->k_key_id[ki], CHR_TASK_HANDOFF_HINT, r[i].node_id, node_id,
                             r[i].rank, node_id, s->k_version_seq[ki], CHR_REASON_RECOVER);
            }
        }
    }
}

__device__ void op_remove(State* s, uint32_t op_index, uint64_t node_id, int32_t* ord_scr) {
    int ni = find_node(s, node_id);
    if (ni < 0 || s->nodes[ni].state != CHR_NODE_LEAVING) { invalid(s, op_index); return; }
    for (int ki = 0; ki < s->key_count; ++ki) {
        if (serving_on(s, ki, node_id) >= 0) {
            s->ctr->remove_stalls += 1;
            emit(s, CHR_EV_REMOVE_STALL_REPLICAS, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
            return;
        }
    }
    ring_order(s, ord_scr);
    for (int i = 0; i < s->token_count; ++i) {
        int ti = ord_scr[i];
        if (s->tokens[ti].node_id == node_id) {
            s->ctr->vnode_removed += 1;
            emit(s, CHR_EV_VNODE_REMOVE, op_index, node_id, 0, s->tokens[ti].token_pos,
                 s->tokens[ti].vnode_ordinal, 0, s->tokens[ti].token_seq);
        }
    }
    // compact tokens removing node_id
    int w = 0;
    for (int i = 0; i < s->token_count; ++i) {
        if (s->tokens[i].node_id != node_id) { if (w != i) s->tokens[w] = s->tokens[i]; ++w; }
    }
    s->token_count = w;

    s->nodes[ni].state = CHR_NODE_REMOVED;
    s->ctr->nodes_removed += 1;
    emit(s, CHR_EV_NODE_REMOVE, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
    s->nodes[ni].last_state_seq = s->event_seq;
}

__device__ void op_put(State* s, uint32_t op_index, uint64_t key_id, int64_t value,
                       int32_t* ord_scr, uint64_t* pref, uint64_t* des) {
    int ki = find_key(s, key_id);
    if (ki < 0 && s->key_count >= s->max_keys) { s->ctr->put_oom += 1; return; }
    if (ki < 0) {
        ki = s->key_count;
        s->k_key_id[ki] = key_id;
        s->k_key_hash[ki] = hash_key(s->key_seed, key_id);
        s->k_value[ki] = value;
        s->k_version_seq[ki] = 0;
        s->k_deleted[ki] = 0;
        s->k_key_seq[ki] = s->key_seq_next++;
        s->k_rep_count[ki] = 0;
        s->key_count += 1;
    }
    s->k_deleted[ki] = 0;
    s->k_value[ki] = value;
    s->k_version_seq[ki] = s->version_seq_next++;
    s->ctr->put_ok += 1;
    emit(s, CHR_EV_KEY_PUT, op_index, 0, key_id, UINT64_MAX, UINT32_MAX, value, s->k_version_seq[ki]);

    int serving_count = 0;
    Replica* r = rep_base(s, ki);
    for (int i = 0; i < s->k_rep_count[ki]; ++i) if (r[i].serving) ++serving_count;

    if (serving_count == 0) {
        int dc = desired_replicas(s, s->k_key_hash[ki], ord_scr, pref, des);
        for (int rr = 0; rr < dc; ++rr) {
            uint64_t nid = des[rr];
            int xi = find_node(s, nid);
            if (xi < 0 || s->nodes[xi].state != CHR_NODE_ACTIVE) continue;
            if (node_full(s, xi)) continue;
            int rc = s->k_rep_count[ki];
            Replica& rep = rep_base(s, ki)[rc];
            rep.rank = rr; rep.node_id = nid;
            rep.kind = (rr == 0) ? CHR_KIND_PRIMARY : CHR_KIND_NORMAL;
            rep.hint_for_node = 0; rep.serving = 1;
            emit(s, CHR_EV_REPLICA_DIRECT_ADD, op_index, nid, key_id, UINT64_MAX, rr, 0, s->k_version_seq[ki]);
            rep.repair_seq = s->event_seq;
            s->k_rep_count[ki] = rc + 1;
            s->nodes[xi].used_slots += 1;
            s->ctr->direct_replica_added += 1;
        }
    } else {
        int dc = desired_replicas(s, s->k_key_hash[ki], ord_scr, pref, des);
        for (int rr = 0; rr < dc; ++rr) {
            if (serving_on(s, ki, des[rr]) >= 0) continue;
            enqueue_move(s, op_index, key_id, CHR_TASK_ADD_REPLICA, 0, des[rr], (uint32_t)rr, 0,
                         s->k_version_seq[ki], CHR_REASON_PUT_REPAIR);
        }
    }
}

__device__ void op_delete(State* s, uint32_t op_index, uint64_t key_id) {
    int ki = find_key(s, key_id);
    if (ki < 0 || s->k_deleted[ki]) {
        s->ctr->delete_miss += 1;
        emit(s, CHR_EV_KEY_DELETE_MISS, op_index, 0, key_id, UINT64_MAX, UINT32_MAX, 0, 0);
        return;
    }
    s->k_deleted[ki] = 1;
    s->k_version_seq[ki] = s->version_seq_next++;
    s->ctr->delete_ok += 1;
    emit(s, CHR_EV_KEY_DELETE, op_index, 0, key_id, UINT64_MAX, UINT32_MAX, 0, s->k_version_seq[ki]);

    // DELETE_REPLICA for every replica record in node-id ascending order.
    // node ids are unique per key, so pick strictly increasing ids.
    int rc = s->k_rep_count[ki];
    Replica* r = rep_base(s, ki);
    uint64_t prev = 0; bool have_prev = false;
    for (int a = 0; a < rc; ++a) {
        uint64_t cur = 0; bool found = false;
        for (int b = 0; b < rc; ++b) {
            uint64_t nid = r[b].node_id;
            if (have_prev && nid <= prev) continue;
            if (!found || nid < cur) { cur = nid; found = true; }
        }
        if (!found) break;
        int bi = rep_on(s, ki, cur);
        enqueue_move(s, op_index, key_id, CHR_TASK_DELETE_REPLICA, cur, 0, r[bi].rank,
                     r[bi].hint_for_node, s->k_version_seq[ki], CHR_REASON_DELETE_REPAIR);
        prev = cur; have_prev = true;
    }
}

__device__ void op_lookup(State* s, uint32_t op_index, uint64_t read_id, uint64_t key_id) {
    int ki = find_key(s, key_id);
    if (ki < 0 || s->k_deleted[ki]) {
        s->ctr->lookup_missing += 1;
        emit(s, CHR_EV_LOOKUP_MISSING, op_index, 0, key_id, UINT64_MAX, UINT32_MAX, 0, 0);
        lk_record(s, read_id, key_id, CHR_LK_MISSING, UINT32_MAX, 0, CHR_KIND_NONE, 0, INT64_MIN, UINT64_MAX);
        return;
    }
    int rc = s->k_rep_count[ki];
    Replica* r = rep_base(s, ki);
    // order serving replicas by rank asc, kind asc, node id asc via repeated selection
    // mark consumed by tracking last picked tuple.
    int emitted = 0;
    bool used[CHR_MAX_REPLICAS];
    for (int i = 0; i < rc; ++i) used[i] = false;
    while (emitted < s->replication_factor) {
        int best = -1;
        for (int i = 0; i < rc; ++i) {
            if (used[i] || !r[i].serving) continue;
            if (best < 0) { best = i; continue; }
            // compare r[i] vs r[best]
            if (r[i].rank != r[best].rank) { if (r[i].rank < r[best].rank) best = i; continue; }
            if (r[i].kind != r[best].kind) { if (r[i].kind < r[best].kind) best = i; continue; }
            if (r[i].node_id < r[best].node_id) best = i;
        }
        if (best < 0) break;
        used[best] = true;
        s->ctr->lookup_found_replicas += 1;
        emit(s, CHR_EV_LOOKUP_REPLICA, op_index, r[best].node_id, key_id, UINT64_MAX, r[best].rank,
             s->k_value[ki], s->k_version_seq[ki]);
        lk_record(s, read_id, key_id, CHR_LK_REPLICA, r[best].rank, r[best].node_id, r[best].kind,
                  r[best].hint_for_node, s->k_value[ki], s->k_version_seq[ki]);
        ++emitted;
    }
    emit(s, CHR_EV_LOOKUP_END, op_index, 0, key_id, UINT64_MAX, UINT32_MAX, 0, 0);
    lk_record(s, read_id, key_id, CHR_LK_END, UINT32_MAX, 0, CHR_KIND_NONE, 0, INT64_MIN, UINT64_MAX);
}

__device__ void enqueue_drop_extras(State* s, uint32_t op_index, int ki, int32_t* ord_scr, uint64_t* pref, uint64_t* des) {
    int dc = desired_replicas(s, s->k_key_hash[ki], ord_scr, pref, des);
    (void)dc;
    int rc = s->k_rep_count[ki];
    Replica* r = rep_base(s, ki);
    int serving_count = 0;
    for (int i = 0; i < rc; ++i) if (r[i].serving) ++serving_count;
    int extras = serving_count - s->replication_factor;
    if (extras <= 0) return;
    // pick worst-rank desc then node asc, `extras` of them
    bool used[CHR_MAX_REPLICAS];
    for (int i = 0; i < rc; ++i) used[i] = false;
    for (int e = 0; e < extras; ++e) {
        int best = -1;
        for (int i = 0; i < rc; ++i) {
            if (used[i] || !r[i].serving) continue;
            if (best < 0) { best = i; continue; }
            if (r[i].rank != r[best].rank) { if (r[i].rank > r[best].rank) best = i; continue; }
            if (r[i].node_id < r[best].node_id) best = i;
        }
        if (best < 0) break;
        used[best] = true;
        enqueue_move(s, op_index, s->k_key_id[ki], CHR_TASK_DROP_REPLICA, r[best].node_id, 0,
                     r[best].rank, 0, s->k_version_seq[ki], CHR_REASON_EXPLICIT_REBALANCE);
    }
}

// returns: 0=keep(stall), 1=remove-uncounted, 2=remove-counted
__device__ int process_task(State* s, uint32_t op_index, Move& mv, int32_t* ord_scr, uint64_t* pref, uint64_t* des) {
    int ki = find_key(s, mv.key_id);

    if (mv.task_kind == CHR_TASK_ADD_REPLICA) {
        if (ki < 0 || s->k_deleted[ki] || s->k_version_seq[ki] != mv.target_version_seq) {
            s->ctr->move_obsolete += 1;
            emit(s, CHR_EV_MOVE_OBSOLETE, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, mv.move_seq);
            return 1;
        }
        int xi = find_node(s, mv.to_node);
        bool dest_has = (rep_on(s, ki, mv.to_node) >= 0);
        if (xi < 0 || s->nodes[xi].state != CHR_NODE_ACTIVE || (node_full(s, xi) && !dest_has)) {
            s->ctr->move_stall += 1;
            emit(s, CHR_EV_MOVE_STALL, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, mv.move_seq);
            return 0;
        }
        int ri = rep_on(s, ki, mv.to_node);
        uint8_t kind = mv.hint_for_node ? CHR_KIND_HINTED : ((mv.rank == 0) ? CHR_KIND_PRIMARY : CHR_KIND_NORMAL);
        Replica* r = rep_base(s, ki);
        if (ri >= 0) {
            if (!r[ri].serving) s->nodes[xi].used_slots += 1;
            r[ri].rank = mv.rank; r[ri].kind = kind; r[ri].hint_for_node = mv.hint_for_node; r[ri].serving = 1;
            emit(s, CHR_EV_REPLICA_ADD, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, s->k_version_seq[ki]);
            r[ri].repair_seq = s->event_seq;
        } else {
            int rc = s->k_rep_count[ki];
            Replica& rep = r[rc];
            rep.rank = mv.rank; rep.node_id = mv.to_node; rep.kind = kind;
            rep.hint_for_node = mv.hint_for_node; rep.serving = 1;
            emit(s, CHR_EV_REPLICA_ADD, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, s->k_version_seq[ki]);
            rep.repair_seq = s->event_seq;
            s->k_rep_count[ki] = rc + 1;
            s->nodes[xi].used_slots += 1;
        }
        s->ctr->replica_added += 1;
        return 2;
    }

    if (mv.task_kind == CHR_TASK_DROP_REPLICA) {
        if (ki < 0) return 1;
        int ri = rep_on(s, ki, mv.from_node);
        Replica* r = rep_base(s, ki);
        if (ri >= 0 && r[ri].serving) {
            int xi = find_node(s, mv.from_node);
            if (xi >= 0 && s->nodes[xi].used_slots > 0) s->nodes[xi].used_slots -= 1;
            emit(s, CHR_EV_REPLICA_DROP, op_index, mv.from_node, mv.key_id, UINT64_MAX, r[ri].rank, 0, s->k_version_seq[ki]);
            rep_erase(s, ki, ri);
            s->ctr->replica_dropped += 1;
        }
        return 2;
    }

    if (mv.task_kind == CHR_TASK_HANDOFF_HINT) {
        if (ki < 0 || s->k_deleted[ki] || s->k_version_seq[ki] != mv.target_version_seq) {
            s->ctr->move_obsolete += 1;
            emit(s, CHR_EV_MOVE_OBSOLETE, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, mv.move_seq);
            return 1;
        }
        int xi = find_node(s, mv.to_node);
        bool dest_has = (rep_on(s, ki, mv.to_node) >= 0);
        if (xi < 0 || s->nodes[xi].state != CHR_NODE_ACTIVE || (node_full(s, xi) && !dest_has)) {
            s->ctr->move_stall += 1;
            emit(s, CHR_EV_MOVE_STALL, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, mv.move_seq);
            return 0;
        }
        Replica* r = rep_base(s, ki);
        int ri = rep_on(s, ki, mv.to_node);
        uint8_t kind = (mv.rank == 0) ? CHR_KIND_PRIMARY : CHR_KIND_NORMAL;
        if (ri >= 0) {
            if (!r[ri].serving) s->nodes[xi].used_slots += 1;
            r[ri].rank = mv.rank; r[ri].kind = kind; r[ri].hint_for_node = 0; r[ri].serving = 1;
            emit(s, CHR_EV_HINT_HANDOFF_ADD, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, s->k_version_seq[ki]);
            r[ri].repair_seq = s->event_seq;
        } else {
            int rc = s->k_rep_count[ki];
            Replica& rep = r[rc];
            rep.rank = mv.rank; rep.node_id = mv.to_node; rep.kind = kind; rep.hint_for_node = 0; rep.serving = 1;
            emit(s, CHR_EV_HINT_HANDOFF_ADD, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, s->k_version_seq[ki]);
            rep.repair_seq = s->event_seq;
            s->k_rep_count[ki] = rc + 1;
            s->nodes[xi].used_slots += 1;
        }
        s->ctr->hint_handoff_added += 1;
        int si = rep_on(s, ki, mv.from_node);
        if (si >= 0 && rep_base(s, ki)[si].kind == CHR_KIND_HINTED) {
            int sxi = find_node(s, mv.from_node);
            Replica* rr = rep_base(s, ki);
            if (rr[si].serving && sxi >= 0 && s->nodes[sxi].used_slots > 0) s->nodes[sxi].used_slots -= 1;
            emit(s, CHR_EV_HINT_DROP, op_index, mv.from_node, mv.key_id, UINT64_MAX, rr[si].rank, 0, s->k_version_seq[ki]);
            rep_erase(s, ki, si);
            s->ctr->hint_dropped += 1;
        }
        return 2;
    }

    if (mv.task_kind == CHR_TASK_DELETE_REPLICA) {
        if (ki >= 0) {
            int ri = rep_on(s, ki, mv.from_node);
            Replica* r = rep_base(s, ki);
            if (ri >= 0) {
                if (r[ri].serving) {
                    int xi = find_node(s, mv.from_node);
                    if (xi >= 0 && s->nodes[xi].used_slots > 0) s->nodes[xi].used_slots -= 1;
                }
                emit(s, CHR_EV_REPLICA_DELETE, op_index, mv.from_node, mv.key_id, UINT64_MAX, r[ri].rank, 0, s->k_version_seq[ki]);
                rep_erase(s, ki, ri);
                s->ctr->replica_deleted += 1;
            }
        }
        return 2;
    }
    return 1;
}

__device__ void move_erase(State* s, int idx) {
    for (int i = idx; i < s->move_count - 1; ++i) s->moves[i] = s->moves[i + 1];
    s->move_count -= 1;
}

__device__ void op_rebalance(State* s, uint32_t op_index, int64_t limit, int32_t* ord_scr, uint64_t* pref, uint64_t* des) {
    if (limit == 0) return;
    int completed = 0;
    int i = 0;
    while (i < s->move_count) {
        if (completed >= (int)limit) break;
        int rc = process_task(s, op_index, s->moves[i], ord_scr, pref, des);
        if (rc == 0) { ++i; continue; }
        uint64_t key_id = s->moves[i].key_id;
        bool counted = (rc == 2);
        move_erase(s, i);
        if (counted) {
            int ki = find_key(s, key_id);
            if (ki >= 0) enqueue_drop_extras(s, op_index, ki, ord_scr, pref, des);
            ++completed;
        }
        // do not advance i
    }
}

__global__ void step_kernel(State* s, int32_t op_type, uint32_t op_index, int64_t a, int64_t b, int64_t c) {
    if (blockIdx.x != 0) return;
    // Whole block pre-sorts the ring + key sets (constant for this op). Every
    // ring_order()/keys_order() call in the op logic then just copies the cache.
    coop_ring_order(s, s->ring_cache);
    coop_keys_order(s, s->keys_cache);
    __syncthreads();
    if (threadIdx.x != 0) return;
    int32_t* ord_scr = s->scratch_idx;
    // korder uses a separate region of scratch after token region
    int32_t* korder = s->scratch_idx + s->max_vnodes;
    // pref/des local buffers
    uint64_t pref[CHR_MAX_NODES];
    uint64_t des[CHR_MAX_REPLICAS];

    switch (op_type) {
        case CHR_OP_ADD_NODE:      op_add_node(s, op_index, (uint64_t)a, b, (uint64_t)c); break;
        case CHR_OP_ACTIVATE_NODE: op_activate(s, op_index, (uint64_t)a, ord_scr, korder, pref, des); break;
        case CHR_OP_START_LEAVE:   op_start_leave(s, op_index, (uint64_t)a, ord_scr, korder, pref, des); break;
        case CHR_OP_FAIL_NODE:     op_fail(s, op_index, (uint64_t)a, ord_scr, korder, pref, des); break;
        case CHR_OP_RECOVER_NODE:  op_recover(s, op_index, (uint64_t)a, korder); break;
        case CHR_OP_REMOVE_NODE:   op_remove(s, op_index, (uint64_t)a, ord_scr); break;
        case CHR_OP_PUT_KEY:       op_put(s, op_index, (uint64_t)a, b, ord_scr, pref, des); break;
        case CHR_OP_DELETE_KEY:    op_delete(s, op_index, (uint64_t)a); break;
        case CHR_OP_LOOKUP:        op_lookup(s, op_index, (uint64_t)a, (uint64_t)b); break;
        case CHR_OP_REBALANCE:     op_rebalance(s, op_index, a, ord_scr, pref, des); break;
        default: invalid(s, op_index); break;
    }
}

// -------- state hash kernels (write into output) --------
__global__ void hash_kernel(State* s, uint64_t* ring_h, uint64_t* node_h, uint64_t* kr_h, uint64_t* move_h) {
    if (blockIdx.x != 0) return;
    // Whole block computes all four canonical orders in parallel; thread 0 then
    // folds the FNV streams sequentially (order-sensitive, but cheap O(n)).
    coop_ring_order(s, s->ring_cache);
    coop_keys_order(s, s->keys_cache);
    coop_node_order(s, s->node_cache);
    coop_move_order(s, s->move_cache);
    __syncthreads();
    if (threadIdx.x != 0) return;

    // ring hash
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < s->token_count; ++i) {
        int ti = s->ring_cache[i];
        fu64(&h, s->tokens[ti].token_pos); fu64(&h, s->tokens[ti].token_seq);
        fu64(&h, s->tokens[ti].node_id); fu32(&h, s->tokens[ti].vnode_ordinal);
    }
    *ring_h = h;

    // node hash by node_id asc
    h = 1469598103934665603ULL;
    for (int i = 0; i < s->node_count; ++i) {
        Node& nd = s->nodes[s->node_cache[i]];
        fu64(&h, nd.node_id); fu8(&h, (uint8_t)nd.state); fu64(&h, nd.capacity);
        fu64(&h, nd.used_slots); fu64(&h, nd.node_seq); fu64(&h, nd.last_state_seq);
    }
    *node_h = h;

    // key replica hash
    h = 1469598103934665603ULL;
    for (int idx = 0; idx < s->key_count; ++idx) {
        int ki = s->keys_cache[idx];
        fu64(&h, s->k_key_id[ki]); fu64(&h, s->k_key_hash[ki]); fi64(&h, s->k_value[ki]);
        fu64(&h, s->k_version_seq[ki]); fu8(&h, s->k_deleted[ki]); fu64(&h, s->k_key_seq[ki]);
        int rc = s->k_rep_count[ki];
        Replica* r = rep_base(s, ki);
        // replicas by rank asc then node asc (selection with used mask)
        bool used[CHR_MAX_REPLICAS];
        for (int i = 0; i < rc; ++i) used[i] = false;
        for (int a = 0; a < rc; ++a) {
            int best = -1;
            for (int i = 0; i < rc; ++i) {
                if (used[i]) continue;
                if (best < 0) { best = i; continue; }
                if (r[i].rank != r[best].rank) { if (r[i].rank < r[best].rank) best = i; continue; }
                if (r[i].node_id < r[best].node_id) best = i;
            }
            if (best < 0) break;
            used[best] = true;
            fu32(&h, r[best].rank); fu64(&h, r[best].node_id); fu8(&h, r[best].kind);
            fu64(&h, r[best].hint_for_node); fu8(&h, r[best].serving); fu64(&h, r[best].repair_seq);
        }
    }
    *kr_h = h;

    // move hash by move_seq (order precomputed cooperatively into move_cache)
    h = 1469598103934665603ULL;
    for (int i = 0; i < s->move_count; ++i) {
        Move& mv = s->moves[s->move_cache[i]];
        fu64(&h, mv.move_seq); fu64(&h, mv.key_id); fu8(&h, mv.task_kind);
        fu64(&h, mv.from_node); fu64(&h, mv.to_node); fu32(&h, mv.rank);
        fu64(&h, mv.hint_for_node); fu64(&h, mv.target_version_seq); fu8(&h, mv.created_reason);
    }
    *move_h = h;
}

} // namespace chr_ref

// ======================= host wrappers =======================
struct ChrRefHandle {
    ChrProblemSpec spec;
    chr_ref::State* d_state;        // device State struct
    chr_ref::State h_state;         // host mirror of pointers
};

static cudaError_t chr_ref_reset(ChrRefHandle* H, cudaStream_t stream) {
    chr_ref::State& s = H->h_state;
    s.event_seq = 0; s.node_seq_next = 1; s.token_seq_next = 1;
    s.key_seq_next = 1; s.version_seq_next = 1; s.move_seq_next = 1;
    s.node_count = 0; s.token_count = 0; s.key_count = 0; s.move_count = 0;

    cudaError_t err;
    err = cudaMemsetAsync(s.ctr, 0, sizeof(ChrCounters), stream); if (err) return err;
    uint64_t basis = 1469598103934665603ULL;
    err = cudaMemcpyAsync(s.ring_event_hash, &basis, 8, cudaMemcpyHostToDevice, stream); if (err) return err;
    err = cudaMemcpyAsync(s.lookup_hash, &basis, 8, cudaMemcpyHostToDevice, stream); if (err) return err;
    // push the State struct (counts/seqs) to device
    err = cudaMemcpyAsync(H->d_state, &H->h_state, sizeof(chr_ref::State), cudaMemcpyHostToDevice, stream);
    return err;
}

extern "C" size_t solution_workspace_bytes(const ChrProblemSpec* spec) {
    if (!chr_validate_problem_spec(spec)) return 0;
    return 256;
}

extern "C" cudaError_t solution_init(const ChrProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!chr_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    ChrRefHandle* H = (ChrRefHandle*)malloc(sizeof(ChrRefHandle));
    if (!H) return cudaErrorMemoryAllocation;
    memset(H, 0, sizeof(*H));
    memcpy(&H->spec, spec, sizeof(ChrProblemSpec));

    chr_ref::State& s = H->h_state;
    memset(&s, 0, sizeof(s));
    s.replication_factor = spec->replication_factor;
    s.preference_list_extra = spec->preference_list_extra;
    s.max_nodes = spec->max_nodes;
    s.max_vnodes = spec->max_vnodes;
    s.max_keys = spec->max_keys;
    s.max_replicas_per_key = spec->max_replicas_per_key;
    s.max_move_tasks = spec->max_move_tasks;
    s.default_node_capacity = spec->default_node_capacity;
    s.ring_seed = spec->ring_seed;
    s.key_seed = spec->key_seed;

    cudaError_t err = cudaSuccess;
    #define ALLOC(p, bytes) do { err = cudaMalloc((void**)&(p), (bytes)); if (err) goto fail; } while (0)
    ALLOC(s.nodes, sizeof(chr_ref::Node) * spec->max_nodes);
    ALLOC(s.tokens, sizeof(chr_ref::Token) * spec->max_vnodes);
    ALLOC(s.k_key_id, sizeof(uint64_t) * spec->max_keys);
    ALLOC(s.k_key_hash, sizeof(uint64_t) * spec->max_keys);
    ALLOC(s.k_value, sizeof(int64_t) * spec->max_keys);
    ALLOC(s.k_version_seq, sizeof(uint64_t) * spec->max_keys);
    ALLOC(s.k_deleted, sizeof(uint8_t) * spec->max_keys);
    ALLOC(s.k_key_seq, sizeof(uint64_t) * spec->max_keys);
    ALLOC(s.k_rep_count, sizeof(int32_t) * spec->max_keys);
    ALLOC(s.replicas, sizeof(chr_ref::Replica) * (size_t)spec->max_keys * spec->max_replicas_per_key);
    ALLOC(s.moves, sizeof(chr_ref::Move) * spec->max_move_tasks);
    ALLOC(s.ctr, sizeof(ChrCounters));
    ALLOC(s.ring_event_hash, sizeof(uint64_t));
    ALLOC(s.lookup_hash, sizeof(uint64_t));
    {
        size_t scratch_n = (size_t)spec->max_vnodes + spec->max_keys + spec->max_move_tasks + 16;
        ALLOC(s.scratch_idx, sizeof(int32_t) * scratch_n);
    }
    ALLOC(s.ring_cache, sizeof(int32_t) * spec->max_vnodes);
    ALLOC(s.keys_cache, sizeof(int32_t) * spec->max_keys);
    ALLOC(s.node_cache, sizeof(int32_t) * spec->max_nodes);
    ALLOC(s.move_cache, sizeof(int32_t) * spec->max_move_tasks);
    #undef ALLOC

    err = cudaMalloc((void**)&H->d_state, sizeof(chr_ref::State));
    if (err) goto fail;

    err = chr_ref_reset(H, stream);
    if (err) goto fail;
    err = cudaStreamSynchronize(stream);
    if (err) goto fail;

    *state_out = H;
    return cudaSuccess;

fail:
    if (s.nodes) cudaFree(s.nodes);
    if (s.tokens) cudaFree(s.tokens);
    if (s.k_key_id) cudaFree(s.k_key_id);
    if (s.k_key_hash) cudaFree(s.k_key_hash);
    if (s.k_value) cudaFree(s.k_value);
    if (s.k_version_seq) cudaFree(s.k_version_seq);
    if (s.k_deleted) cudaFree(s.k_deleted);
    if (s.k_key_seq) cudaFree(s.k_key_seq);
    if (s.k_rep_count) cudaFree(s.k_rep_count);
    if (s.replicas) cudaFree(s.replicas);
    if (s.moves) cudaFree(s.moves);
    if (s.ctr) cudaFree(s.ctr);
    if (s.ring_event_hash) cudaFree(s.ring_event_hash);
    if (s.lookup_hash) cudaFree(s.lookup_hash);
    if (s.scratch_idx) cudaFree(s.scratch_idx);
    if (s.ring_cache) cudaFree(s.ring_cache);
    if (s.keys_cache) cudaFree(s.keys_cache);
    if (s.node_cache) cudaFree(s.node_cache);
    if (s.move_cache) cudaFree(s.move_cache);
    if (H->d_state) cudaFree(H->d_state);
    free(H);
    return err;
}

extern "C" cudaError_t solution_run(void* state, const ChrRunSpec* run, const void* inputs,
                                    void* outputs_void, void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    (void)inputs; (void)workspace; (void)workspace_bytes;
    if (!state || !chr_validate_run_spec(run) || !outputs_void) return cudaErrorInvalidValue;
    ChrRefHandle* H = (ChrRefHandle*)state;
    ChrOutputs* out = (ChrOutputs*)outputs_void;
    if (!out->counters || !out->ring_event_hash || !out->lookup_hash || !out->ring_hash ||
        !out->node_hash || !out->key_replica_hash || !out->move_hash) return cudaErrorInvalidValue;

    chr_ref::step_kernel<<<1, 256, 0, stream>>>(H->d_state, run->op_type, (uint32_t)run->op_index,
                                              run->arg_a, run->arg_b, run->arg_c);
    cudaError_t err = cudaPeekAtLastError(); if (err) return err;

    chr_ref::hash_kernel<<<1, 256, 0, stream>>>(H->d_state, out->ring_hash, out->node_hash,
                                              out->key_replica_hash, out->move_hash);
    err = cudaPeekAtLastError(); if (err) return err;

    // copy cumulative counters + event/lookup hashes out
    err = cudaMemcpyAsync(out->counters, H->h_state.ctr, sizeof(ChrCounters), cudaMemcpyDeviceToDevice, stream); if (err) return err;
    err = cudaMemcpyAsync(out->ring_event_hash, H->h_state.ring_event_hash, 8, cudaMemcpyDeviceToDevice, stream); if (err) return err;
    err = cudaMemcpyAsync(out->lookup_hash, H->h_state.lookup_hash, 8, cudaMemcpyDeviceToDevice, stream); if (err) return err;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return chr_ref_reset((ChrRefHandle*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    ChrRefHandle* H = (ChrRefHandle*)state;
    chr_ref::State& s = H->h_state;
    if (s.nodes) cudaFree(s.nodes);
    if (s.tokens) cudaFree(s.tokens);
    if (s.k_key_id) cudaFree(s.k_key_id);
    if (s.k_key_hash) cudaFree(s.k_key_hash);
    if (s.k_value) cudaFree(s.k_value);
    if (s.k_version_seq) cudaFree(s.k_version_seq);
    if (s.k_deleted) cudaFree(s.k_deleted);
    if (s.k_key_seq) cudaFree(s.k_key_seq);
    if (s.k_rep_count) cudaFree(s.k_rep_count);
    if (s.replicas) cudaFree(s.replicas);
    if (s.moves) cudaFree(s.moves);
    if (s.ctr) cudaFree(s.ctr);
    if (s.ring_event_hash) cudaFree(s.ring_event_hash);
    if (s.lookup_hash) cudaFree(s.lookup_hash);
    if (s.scratch_idx) cudaFree(s.scratch_idx);
    if (s.ring_cache) cudaFree(s.ring_cache);
    if (s.keys_cache) cudaFree(s.keys_cache);
    if (s.node_cache) cudaFree(s.node_cache);
    if (s.move_cache) cudaFree(s.move_cache);
    if (H->d_state) cudaFree(H->d_state);
    free(H);
}
