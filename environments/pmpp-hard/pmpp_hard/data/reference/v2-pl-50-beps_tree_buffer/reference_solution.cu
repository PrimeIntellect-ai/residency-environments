// PMPP_CANARY_50_1200b21dfd -- held-out canary; MUST NOT appear in any submission
// file: beps_tree_buffer_reference.cu
//
// Reference GPU implementation of the Bε-tree message-buffer contract.
// The whole persistent tree lives in flat device SoA arrays; a single-thread
// kernel replays the op batch deterministically. Per-node buffers and leaf
// records live in fixed-stride slabs (node-indexed) kept sorted (buffer by
// msg_seq ascending, records by key ascending), which is a representation
// distinct from both the CPU oracle (std::map / std::vector) and the naive
// implementation (which uses unsorted append slabs + scan).

#include "beps_tree_buffer_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct BepsRefState {
    BepsProblemSpec spec;

    // node SoA (index by node_id; 0 unused)
    uint64_t* n_parent;
    uint64_t* n_low;
    uint64_t* n_high;
    uint32_t* n_depth;
    uint8_t*  n_isleaf;
    uint8_t*  n_alive;
    int32_t*  n_childcnt;
    int32_t*  n_bufcnt;
    int32_t*  n_reccnt;

    // children slab: child id per (node*max_children + i)
    uint64_t* child;       // node_cap * max_children

    // GLOBAL message arena (append-only in msg_seq order). Each live message is
    // tagged with its owner internal node. A node's buffer is the subsequence
    // of live arena entries with owner==node, in arena (== msg_seq) order. This
    // lets an internal buffer grow without a fixed per-node bound.
    uint64_t* buf_owner;   // owner node id (0 == dead/applied)
    uint64_t* buf_seq;
    uint8_t*  buf_kind;
    uint64_t* buf_key;
    int64_t*  buf_val;

    // record slab (sorted by key): per node*rec_stride
    uint64_t* rec_key;
    int64_t*  rec_val;
    uint64_t* rec_seq;
    uint8_t*  rec_del;

    // scalars [1]
    uint64_t* event_seq;
    uint64_t* msg_seq_next;
    uint64_t* node_id_next;
    uint64_t* root_id;
    int32_t*  buf_used;     // [1] number of arena slots appended so far

    // accumulators
    BepsCounts* counts;
    uint64_t* meh;  // message_event_hash
    uint64_t* qh;   // query_hash

    int32_t node_cap;
    int32_t max_children;
    int32_t buf_cap;       // arena capacity
    int32_t rec_stride;
};

__device__ __forceinline__ uint64_t beps_r_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= 1099511628211ULL; return h;
}
__device__ void beps_r_fnv(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p; uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = beps_r_fnv_byte(v, b[i]);
    *h = v;
}

// ---- device helpers operating on the SoA via a context struct ----
struct Ctx {
    BepsRefState s;
    uint64_t event_seq;
    uint64_t msg_seq_next;
    uint64_t node_id_next;
    uint64_t root_id;
    int32_t buf_used;
};

__device__ uint64_t r_alloc_node(Ctx* c) {
    uint64_t id = c->node_id_next++;
    c->s.n_parent[id] = 0;
    c->s.n_low[id] = 0;
    c->s.n_high[id] = 0;
    c->s.n_depth[id] = 0;
    c->s.n_isleaf[id] = 1;
    c->s.n_alive[id] = 1;
    c->s.n_childcnt[id] = 0;
    c->s.n_bufcnt[id] = 0;
    c->s.n_reccnt[id] = 0;
    return id;
}

__device__ void r_emit_event(Ctx* c, uint8_t ekind, uint32_t op_index,
                              uint64_t node_id, uint64_t msg_seq, uint8_t kind,
                              uint64_t key, int64_t value, uint64_t child_node) {
    uint64_t es = c->event_seq++;
    uint64_t* h = c->s.meh;
    beps_r_fnv(h, &ekind, 1);
    beps_r_fnv(h, &es, 8);
    beps_r_fnv(h, &op_index, 4);
    beps_r_fnv(h, &node_id, 8);
    beps_r_fnv(h, &msg_seq, 8);
    beps_r_fnv(h, &kind, 1);
    beps_r_fnv(h, &key, 8);
    beps_r_fnv(h, &value, 8);
    beps_r_fnv(h, &child_node, 8);
}

__device__ void r_emit_point(Ctx* c, uint64_t read_id, uint32_t op_index,
                             uint64_t key, uint8_t found, int64_t value,
                             uint64_t latest_seq) {
    uint8_t rk = BEPS_REC_POINT_RESULT; uint64_t* h = c->s.qh;
    beps_r_fnv(h, &rk, 1); beps_r_fnv(h, &read_id, 8); beps_r_fnv(h, &op_index, 4);
    beps_r_fnv(h, &key, 8); beps_r_fnv(h, &found, 1); beps_r_fnv(h, &value, 8);
    beps_r_fnv(h, &latest_seq, 8);
}
__device__ void r_emit_rr(Ctx* c, uint64_t read_id, uint32_t op_index,
                          uint64_t key, int64_t value, uint64_t latest_seq) {
    uint8_t rk = BEPS_REC_RANGE_RESULT; uint64_t* h = c->s.qh;
    beps_r_fnv(h, &rk, 1); beps_r_fnv(h, &read_id, 8); beps_r_fnv(h, &op_index, 4);
    beps_r_fnv(h, &key, 8); beps_r_fnv(h, &value, 8); beps_r_fnv(h, &latest_seq, 8);
}
__device__ void r_emit_re(Ctx* c, uint64_t read_id, uint32_t op_index, uint64_t cnt) {
    uint8_t rk = BEPS_REC_RANGE_END; uint64_t* h = c->s.qh;
    beps_r_fnv(h, &rk, 1); beps_r_fnv(h, &read_id, 8); beps_r_fnv(h, &op_index, 4);
    beps_r_fnv(h, &cnt, 8);
}

// child whose [low,high] contains key
__device__ uint64_t r_child_for_key(Ctx* c, uint64_t n, uint64_t key) {
    int cnt = c->s.n_childcnt[n];
    uint64_t base = (uint64_t)n * c->s.max_children;
    for (int i = 0; i < cnt; ++i) {
        uint64_t ch = c->s.child[base + i];
        if (key >= c->s.n_low[ch] && key <= c->s.n_high[ch]) return ch;
    }
    return 0;
}
__device__ uint64_t r_find_leaf(Ctx* c, uint64_t key) {
    uint64_t cur = c->root_id;
    while (!c->s.n_isleaf[cur]) cur = r_child_for_key(c, cur, key);
    return cur;
}

// Apply message to a leaf's sorted record slab. emit controls LEAF_APPLY_*.
__device__ void r_apply_leaf(Ctx* c, uint64_t leaf, uint64_t seq, uint8_t kind,
                             uint64_t key, int64_t value, uint32_t op_index,
                             bool emit) {
    uint64_t base = (uint64_t)leaf * c->s.rec_stride;
    int cnt = c->s.n_reccnt[leaf];
    // binary search for key
    int lo = 0, hi = cnt, idx = cnt; bool found = false;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        uint64_t mk = c->s.rec_key[base + mid];
        if (mk == key) { idx = mid; found = true; break; }
        else if (mk < key) lo = mid + 1;
        else hi = mid;
    }
    if (!found) idx = lo;  // insertion point

    int64_t newval; uint8_t newdel;
    if (kind == BEPS_MSG_SET) { newval = value; newdel = 0; }
    else if (kind == BEPS_MSG_ADD) {
        int64_t b = 0;
        if (found && c->s.rec_del[base + idx] == 0) b = c->s.rec_val[base + idx];
        newval = (int64_t)((uint64_t)b + (uint64_t)value); newdel = 0;
    } else { newval = 0; newdel = 1; }

    if (found) {
        c->s.rec_val[base + idx] = newval;
        c->s.rec_seq[base + idx] = seq;
        c->s.rec_del[base + idx] = newdel;
    } else {
        // shift up to insert at idx
        for (int j = cnt; j > idx; --j) {
            c->s.rec_key[base + j] = c->s.rec_key[base + j - 1];
            c->s.rec_val[base + j] = c->s.rec_val[base + j - 1];
            c->s.rec_seq[base + j] = c->s.rec_seq[base + j - 1];
            c->s.rec_del[base + j] = c->s.rec_del[base + j - 1];
        }
        c->s.rec_key[base + idx] = key;
        c->s.rec_val[base + idx] = newval;
        c->s.rec_seq[base + idx] = seq;
        c->s.rec_del[base + idx] = newdel;
        c->s.n_reccnt[leaf] = cnt + 1;
    }
    if (emit) {
        uint8_t ek = (kind == BEPS_MSG_SET) ? BEPS_EV_LEAF_APPLY_SET
                   : (kind == BEPS_MSG_ADD) ? BEPS_EV_LEAF_APPLY_ADD
                                            : BEPS_EV_LEAF_APPLY_DEL;
        r_emit_event(c, ek, op_index, leaf, seq, kind, key, value, UINT64_MAX);
    }
}

__device__ void r_insert_child_after(Ctx* c, uint64_t parent, uint64_t left,
                                     uint64_t right) {
    uint64_t base = (uint64_t)parent * c->s.max_children;
    int cnt = c->s.n_childcnt[parent];
    int pos = cnt;
    for (int i = 0; i < cnt; ++i) if (c->s.child[base + i] == left) { pos = i + 1; break; }
    for (int j = cnt; j > pos; --j) c->s.child[base + j] = c->s.child[base + j - 1];
    c->s.child[base + pos] = right;
    c->s.n_childcnt[parent] = cnt + 1;
}

// recursively set depth of subtree
__device__ void r_set_depths(Ctx* c, uint64_t n, uint32_t depth) {
    // iterative stack-free via explicit recursion is risky on device; use an
    // explicit stack array sized to node_cap is overkill. Trees here are
    // shallow; use bounded recursion through a manual stack.
    // manual DFS stack
    uint64_t stack[64]; uint32_t dstack[64]; int sp = 0;
    stack[sp] = n; dstack[sp] = depth; sp++;
    while (sp > 0) {
        --sp;
        uint64_t cur = stack[sp]; uint32_t d = dstack[sp];
        c->s.n_depth[cur] = d;
        if (!c->s.n_isleaf[cur]) {
            uint64_t base = (uint64_t)cur * c->s.max_children;
            int cc = c->s.n_childcnt[cur];
            for (int i = 0; i < cc; ++i) {
                stack[sp] = c->s.child[base + i]; dstack[sp] = d + 1; sp++;
            }
        }
    }
}

__device__ void r_split_internal(Ctx* c, uint64_t node, uint32_t op_index);

__device__ void r_split_leaf(Ctx* c, uint64_t leaf, uint32_t op_index) {
    uint64_t lbase = (uint64_t)leaf * c->s.rec_stride;
    int rc = c->s.n_reccnt[leaf];
    int split_idx = rc / 2;
    uint64_t split_key = c->s.rec_key[lbase + split_idx];

    uint64_t old_low = c->s.n_low[leaf];
    uint64_t old_high = c->s.n_high[leaf];
    uint64_t old_parent = c->s.n_parent[leaf];
    uint32_t old_depth = c->s.n_depth[leaf];

    uint64_t right = r_alloc_node(c);
    uint64_t rbase = (uint64_t)right * c->s.rec_stride;
    c->s.n_isleaf[right] = 1;
    c->s.n_low[right] = split_key;
    c->s.n_high[right] = old_high;
    // move records [split_idx, rc) to right
    int rcount = 0;
    for (int i = split_idx; i < rc; ++i) {
        c->s.rec_key[rbase + rcount] = c->s.rec_key[lbase + i];
        c->s.rec_val[rbase + rcount] = c->s.rec_val[lbase + i];
        c->s.rec_seq[rbase + rcount] = c->s.rec_seq[lbase + i];
        c->s.rec_del[rbase + rcount] = c->s.rec_del[lbase + i];
        rcount++;
    }
    c->s.n_reccnt[right] = rcount;
    c->s.n_reccnt[leaf] = split_idx;
    c->s.n_high[leaf] = split_key - 1;

    c->s.counts->leaf_splits += 1;

    if (leaf == c->root_id) {
        uint64_t nroot = r_alloc_node(c);
        c->s.n_isleaf[nroot] = 0;
        c->s.n_low[nroot] = old_low;
        c->s.n_high[nroot] = old_high;
        c->s.n_depth[nroot] = old_depth;
        c->s.n_parent[nroot] = 0;
        uint64_t cb = (uint64_t)nroot * c->s.max_children;
        c->s.child[cb + 0] = leaf;
        c->s.child[cb + 1] = right;
        c->s.n_childcnt[nroot] = 2;
        c->s.n_parent[leaf] = nroot;
        c->s.n_parent[right] = nroot;
        c->s.n_depth[leaf] = old_depth + 1;
        c->s.n_depth[right] = old_depth + 1;
        c->root_id = nroot;
    } else {
        c->s.n_parent[right] = old_parent;
        c->s.n_depth[right] = old_depth;
        r_insert_child_after(c, old_parent, leaf, right);
        if (c->s.n_childcnt[old_parent] > c->s.spec.max_children_per_internal)
            r_split_internal(c, old_parent, op_index);
    }
}

__device__ void r_split_internal(Ctx* c, uint64_t node, uint32_t op_index) {
    int m = c->s.n_childcnt[node];
    int sci = m / 2;
    uint64_t old_low = c->s.n_low[node];
    uint64_t old_high = c->s.n_high[node];
    uint64_t old_parent = c->s.n_parent[node];
    uint32_t old_depth = c->s.n_depth[node];

    uint64_t cbase = (uint64_t)node * c->s.max_children;
    uint64_t left_high = c->s.n_high[c->s.child[cbase + sci - 1]];
    uint64_t right_low = c->s.n_low[c->s.child[cbase + sci]];

    uint64_t right = r_alloc_node(c);
    c->s.n_isleaf[right] = 0;
    c->s.n_low[right] = right_low;
    c->s.n_high[right] = old_high;
    uint64_t rcbase = (uint64_t)right * c->s.max_children;
    int rcc = 0;
    for (int i = sci; i < m; ++i) {
        uint64_t ch = c->s.child[cbase + i];
        c->s.child[rcbase + rcc++] = ch;
        c->s.n_parent[ch] = right;
    }
    c->s.n_childcnt[right] = rcc;
    c->s.n_childcnt[node] = sci;
    c->s.n_high[node] = left_high;

    // redistribute buffer by range: retag arena entries owned by `node` whose
    // key falls in the right range to be owned by `right`. Arena order (==
    // msg_seq order) is preserved automatically; no copying needed.
    int lc = 0, rcn = 0;
    for (int i = 0; i < c->buf_used; ++i) {
        if (c->s.buf_owner[i] != node) continue;
        uint64_t k = c->s.buf_key[i];
        if (k >= right_low && k <= old_high) { c->s.buf_owner[i] = right; rcn++; }
        else lc++;
    }
    c->s.n_bufcnt[node] = lc;
    c->s.n_bufcnt[right] = rcn;

    c->s.counts->internal_splits += 1;

    if (node == c->root_id) {
        uint64_t nroot = r_alloc_node(c);
        c->s.n_isleaf[nroot] = 0;
        c->s.n_low[nroot] = old_low;
        c->s.n_high[nroot] = old_high;
        c->s.n_depth[nroot] = old_depth;
        c->s.n_parent[nroot] = 0;
        uint64_t ncb = (uint64_t)nroot * c->s.max_children;
        c->s.child[ncb + 0] = node;
        c->s.child[ncb + 1] = right;
        c->s.n_childcnt[nroot] = 2;
        c->s.n_parent[node] = nroot;
        c->s.n_parent[right] = nroot;
        c->root_id = nroot;
        r_set_depths(c, node, old_depth + 1);
        r_set_depths(c, right, old_depth + 1);
    } else {
        c->s.n_parent[right] = old_parent;
        r_set_depths(c, node, old_depth);
        r_set_depths(c, right, old_depth);
        r_insert_child_after(c, old_parent, node, right);
        if (c->s.n_childcnt[old_parent] > c->s.spec.max_children_per_internal)
            r_split_internal(c, old_parent, op_index);
    }
}

// append a message to an internal node buffer = append a new live arena entry
// owned by `node`. msg_seq ascending order is the arena order.
__device__ void r_buf_append(Ctx* c, uint64_t node, uint64_t seq, uint8_t kind,
                             uint64_t key, int64_t value) {
    int idx = c->buf_used++;
    c->s.buf_owner[idx] = node;
    c->s.buf_seq[idx] = seq;
    c->s.buf_kind[idx] = kind;
    c->s.buf_key[idx] = key;
    c->s.buf_val[idx] = value;
    c->s.n_bufcnt[node] += 1;
}

// flush one group from source. returns msgs moved.
__device__ int64_t r_flush_group(Ctx* c, uint64_t source, int64_t max_messages,
                                 uint32_t op_index) {
    int nc = c->s.n_childcnt[source];
    uint64_t cb = (uint64_t)source * c->s.max_children;

    // count source's live messages per child by range (single arena scan).
    int best = -1; int64_t best_cnt = 0; uint64_t best_low = 0, best_id = 0;
    for (int ci = 0; ci < nc; ++ci) {
        uint64_t ch = c->s.child[cb + ci];
        uint64_t clo = c->s.n_low[ch], chi = c->s.n_high[ch];
        int64_t cnt = 0;
        for (int i = 0; i < c->buf_used; ++i) {
            if (c->s.buf_owner[i] != source) continue;
            uint64_t k = c->s.buf_key[i];
            if (k >= clo && k <= chi) cnt++;
        }
        if (cnt <= 0) continue;
        if (best < 0 || cnt > best_cnt ||
            (cnt == best_cnt && (clo < best_low ||
              (clo == best_low && ch < best_id)))) {
            best = ci; best_cnt = cnt; best_low = clo; best_id = ch;
        }
    }
    if (best < 0) return 0;

    uint64_t child = c->s.child[cb + best];
    uint64_t clo = c->s.n_low[child], chi = c->s.n_high[child];
    int64_t cap = max_messages;
    if ((int64_t)c->s.spec.flush_message_cap < cap) cap = c->s.spec.flush_message_cap;
    bool child_leaf = c->s.n_isleaf[child];

    // Pass 1: emit ALL FLUSH_MESSAGE events (the move) in msg_seq (arena) order.
    int taken = 0;
    for (int i = 0; i < c->buf_used && taken < cap; ++i) {
        if (c->s.buf_owner[i] != source) continue;
        uint64_t k = c->s.buf_key[i];
        if (k >= clo && k <= chi) {
            r_emit_event(c, BEPS_EV_FLUSH_MESSAGE, op_index, source,
                         c->s.buf_seq[i], c->s.buf_kind[i], k, c->s.buf_val[i], child);
            taken++;
        }
    }
    int64_t moved = taken;
    c->s.counts->flush_messages += (uint64_t)moved;

    // Pass 2: deliver in moved order: retag arena owner to child (internal) or
    // mark dead + apply (leaf). Updates bufcnt.
    int taken_d = 0;
    for (int i = 0; i < c->buf_used && taken_d < cap; ++i) {
        if (c->s.buf_owner[i] != source) continue;
        uint64_t k = c->s.buf_key[i];
        if (k < clo || k > chi) continue;
        uint64_t seq = c->s.buf_seq[i];
        uint8_t kind = c->s.buf_kind[i];
        int64_t val = c->s.buf_val[i];
        c->s.n_bufcnt[source] -= 1;
        if (child_leaf) {
            c->s.buf_owner[i] = 0;  // dead
            r_apply_leaf(c, child, seq, kind, k, val, op_index, true);
            if (kind == BEPS_MSG_SET) c->s.counts->leaf_apply_set += 1;
            else if (kind == BEPS_MSG_ADD) c->s.counts->leaf_apply_add += 1;
            else c->s.counts->leaf_apply_del += 1;
        } else {
            c->s.buf_owner[i] = child;
            c->s.n_bufcnt[child] += 1;
        }
        taken_d++;
    }

    if (child_leaf && c->s.n_reccnt[child] > c->s.spec.leaf_record_cap)
        r_split_leaf(c, child, op_index);

    return moved;
}

__device__ uint64_t r_choose_source(Ctx* c) {
    uint64_t best = 0;
    for (uint64_t i = 1; i < c->node_id_next; ++i) {
        if (!c->s.n_alive[i]) continue;
        if (c->s.n_isleaf[i]) continue;
        if (c->s.n_bufcnt[i] <= 0) continue;
        if (best == 0) { best = i; continue; }
        int bl = c->s.n_bufcnt[best], cl = c->s.n_bufcnt[i];
        bool take = false;
        if (cl > bl) take = true;
        else if (cl == bl) {
            if (c->s.n_depth[i] < c->s.n_depth[best]) take = true;
            else if (c->s.n_depth[i] == c->s.n_depth[best]) {
                if (c->s.n_low[i] < c->s.n_low[best]) take = true;
                else if (c->s.n_low[i] == c->s.n_low[best] && i < best) take = true;
            }
        }
        if (take) best = i;
    }
    return best;
}

__device__ void r_do_flush(Ctx* c, int64_t mfn, int64_t mmt, uint32_t op_index) {
    if (mfn <= 0 || mmt <= 0) return;
    while (mfn > 0 && mmt > 0) {
        uint64_t src = r_choose_source(c);
        if (src == 0) {
            r_emit_event(c, BEPS_EV_FLUSH_EMPTY, op_index, 0, UINT64_MAX, 255,
                         UINT64_MAX, 0, UINT64_MAX);
            c->s.counts->flush_empty += 1;
            break;
        }
        int64_t moved = r_flush_group(c, src, mmt, op_index);
        c->s.counts->flush_groups += 1;
        mfn -= 1;
        mmt -= moved;
    }
}

__device__ void r_auto_flush_root(Ctx* c, uint32_t op_index) {
    if (c->s.n_bufcnt[c->root_id] <= 0) return;
    r_flush_group(c, c->root_id, (int64_t)INT64_MAX, op_index);
    c->s.counts->flush_groups += 1;
}

__device__ void r_do_mutation(Ctx* c, uint8_t kind, uint64_t key, int64_t value,
                              uint32_t op_index) {
    uint64_t root = c->root_id;
    if (c->s.n_isleaf[root]) {
        uint64_t seq = c->msg_seq_next++;
        int64_t v = (kind == BEPS_MSG_DEL) ? 0 : value;
        r_apply_leaf(c, root, seq, kind, key, v, op_index, true);
        if (kind == BEPS_MSG_SET) c->s.counts->leaf_apply_set += 1;
        else if (kind == BEPS_MSG_ADD) c->s.counts->leaf_apply_add += 1;
        else c->s.counts->leaf_apply_del += 1;
        if (c->s.n_reccnt[root] > c->s.spec.leaf_record_cap)
            r_split_leaf(c, root, op_index);
        return;
    }
    if (c->s.n_bufcnt[root] >= c->s.spec.internal_buffer_cap)
        r_auto_flush_root(c, op_index);
    root = c->root_id;  // may have changed if root split during flush
    if (c->s.n_bufcnt[root] >= c->s.spec.internal_buffer_cap) {
        int64_t v = (kind == BEPS_MSG_DEL) ? 0 : value;
        r_emit_event(c, BEPS_EV_WRITE_STALL, op_index, root, UINT64_MAX, kind,
                     key, v, UINT64_MAX);
        c->s.counts->write_stall += 1;
        return;
    }
    uint64_t seq = c->msg_seq_next++;
    int64_t v = (kind == BEPS_MSG_DEL) ? 0 : value;
    r_buf_append(c, root, seq, kind, key, v);
    if (kind == BEPS_MSG_SET) {
        r_emit_event(c, BEPS_EV_ROOT_BUFFER_SET, op_index, root, seq, BEPS_MSG_SET,
                     key, v, UINT64_MAX);
        c->s.counts->root_buffered_set += 1;
    } else if (kind == BEPS_MSG_ADD) {
        r_emit_event(c, BEPS_EV_ROOT_BUFFER_ADD, op_index, root, seq, BEPS_MSG_ADD,
                     key, v, UINT64_MAX);
        c->s.counts->root_buffered_add += 1;
    } else {
        r_emit_event(c, BEPS_EV_ROOT_BUFFER_DEL, op_index, root, seq, BEPS_MSG_DEL,
                     key, v, UINT64_MAX);
        c->s.counts->root_buffered_del += 1;
    }
}

// visibility for a key
__device__ void r_visible(Ctx* c, uint64_t key, uint8_t* found, int64_t* value,
                          uint64_t* latest_seq) {
    // gather path msgs for key into a temp buffer, plus leaf record, then
    // selection-apply in ascending seq. Path depth small; total path msgs
    // bounded but can be large -> we do a streaming "min-seq" selection instead
    // of sorting, applying one event at a time.
    // We implement by repeatedly finding the next-smallest unused seq among
    // path msgs and the (single) leaf record event.
    uint64_t cur = c->root_id;
    // first, find leaf and collect path nodes
    uint64_t path[64]; int pn = 0;
    while (!c->s.n_isleaf[cur]) { path[pn++] = cur; cur = r_child_for_key(c, cur, key); }
    uint64_t leaf = cur;

    // leaf record event
    bool have_leaf = false; uint64_t leaf_seq = 0; uint8_t leaf_kind = 0; int64_t leaf_val = 0;
    {
        uint64_t lb = (uint64_t)leaf * c->s.rec_stride;
        int cnt = c->s.n_reccnt[leaf];
        int lo = 0, hi = cnt;
        while (lo < hi) { int mid=(lo+hi)>>1; uint64_t mk=c->s.rec_key[lb+mid];
            if (mk==key){lo=mid;hi=mid;have_leaf=true;
                leaf_seq=c->s.rec_seq[lb+mid];
                if (c->s.rec_del[lb+mid]) { leaf_kind=BEPS_MSG_DEL; leaf_val=0; }
                else { leaf_kind=BEPS_MSG_SET; leaf_val=c->s.rec_val[lb+mid]; }
                break;}
            else if (mk<key) lo=mid+1; else hi=mid; }
    }

    // Iterate ALL events (leaf record + path msgs for this key) in ascending
    // msg_seq. All seqs are unique (monotonic), so repeatedly select the
    // smallest seq strictly greater than the previously applied one (prev),
    // starting from prev=0 (seqs begin at 1).
    bool missing = true; uint8_t f = 0; int64_t v = 0; uint64_t ls = 0;
    uint64_t prev = 0;
    while (true) {
        uint64_t bestseq = 0; bool have = false; uint8_t bk = 0; int64_t bv = 0;
        if (have_leaf && leaf_seq > prev) {
            have = true; bestseq = leaf_seq; bk = leaf_kind; bv = leaf_val;
        }
        for (int i = 0; i < c->buf_used; ++i) {
            uint64_t owner = c->s.buf_owner[i];
            if (owner == 0) continue;
            if (c->s.buf_key[i] != key) continue;
            // owner must be on the path
            bool on_path = false;
            for (int p = 0; p < pn; ++p) if (path[p] == owner) { on_path = true; break; }
            if (!on_path) continue;
            uint64_t sq = c->s.buf_seq[i];
            if (sq <= prev) continue;
            if (!have || sq < bestseq) {
                have = true; bestseq = sq; bk = c->s.buf_kind[i]; bv = c->s.buf_val[i];
            }
        }
        if (!have) break;
        if (bk == BEPS_MSG_SET) { f=1; v=bv; ls=bestseq; missing=false; }
        else if (bk == BEPS_MSG_ADD) {
            int64_t base = missing ? 0 : v;
            v = (int64_t)((uint64_t)base + (uint64_t)bv); f=1; ls=bestseq; missing=false;
        } else { f=0; v=0; ls=bestseq; missing=true; }
        prev = bestseq;
    }
    *found = f; *value = v; *latest_seq = ls;
}

__device__ void r_do_point(Ctx* c, uint64_t read_id, uint64_t key, uint32_t op_index) {
    uint8_t found; int64_t value; uint64_t ls;
    r_visible(c, key, &found, &value, &ls);
    if (found) { c->s.counts->point_found += 1; r_emit_point(c, read_id, op_index, key, 1, value, ls); }
    else { c->s.counts->point_missing += 1; r_emit_point(c, read_id, op_index, key, 0, INT64_MIN, 0); }
}

__device__ void r_do_range(Ctx* c, uint64_t read_id, uint64_t lo, uint64_t hi,
                           int64_t limit, uint32_t op_index, uint64_t* scratch,
                           int64_t scratch_cap) {
    if (lo > hi) { c->s.counts->invalid_count += 1; return; }
    int64_t eff = limit;
    if ((int64_t)c->s.spec.max_range_results < eff) eff = c->s.spec.max_range_results;
    if (eff < 0) eff = 0;
    // collect candidate keys into scratch, then sort+unique.
    int64_t cnt = 0;
    for (uint64_t n = 1; n < c->node_id_next; ++n) {
        if (!c->s.n_alive[n]) continue;
        if (!c->s.n_isleaf[n]) continue;
        uint64_t lb = (uint64_t)n * c->s.rec_stride;
        int rc = c->s.n_reccnt[n];
        for (int i = 0; i < rc; ++i) {
            uint64_t k = c->s.rec_key[lb + i];
            if (k >= lo && k <= hi && cnt < scratch_cap) scratch[cnt++] = k;
        }
    }
    // buffered candidate keys: one arena scan over live entries.
    for (int i = 0; i < c->buf_used; ++i) {
        if (c->s.buf_owner[i] == 0) continue;
        uint64_t k = c->s.buf_key[i];
        if (k >= lo && k <= hi && cnt < scratch_cap) scratch[cnt++] = k;
    }
    // insertion sort (cnt is modest in tests)
    for (int64_t i = 1; i < cnt; ++i) {
        uint64_t x = scratch[i]; int64_t j = i - 1;
        while (j >= 0 && scratch[j] > x) { scratch[j+1] = scratch[j]; --j; }
        scratch[j+1] = x;
    }
    uint64_t emitted = 0; uint64_t lastk = 0; bool haslast = false;
    for (int64_t i = 0; i < cnt; ++i) {
        uint64_t k = scratch[i];
        if (haslast && k == lastk) continue;  // dedup
        lastk = k; haslast = true;
        if ((int64_t)emitted >= eff) continue;  // keep dedup but stop emitting
        uint8_t found; int64_t value; uint64_t ls;
        r_visible(c, k, &found, &value, &ls);
        if (found) { r_emit_rr(c, read_id, op_index, k, value, ls); c->s.counts->range_results += 1; emitted++; }
    }
    r_emit_re(c, read_id, op_index, emitted);
    c->s.counts->range_end_count += 1;
}

__global__ void beps_ref_kernel(BepsRefState s, const BepsOp* __restrict__ ops,
                                int num_ops, uint64_t* __restrict__ scratch,
                                int64_t scratch_cap) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    Ctx c; c.s = s;
    c.event_seq = s.event_seq[0];
    c.msg_seq_next = s.msg_seq_next[0];
    c.node_id_next = s.node_id_next[0];
    c.root_id = s.root_id[0];
    c.buf_used = s.buf_used[0];

    for (int i = 0; i < num_ops; ++i) {
        const BepsOp o = ops[i];
        uint32_t op_index = (uint32_t)i;
        switch (o.kind) {
            case BEPS_OP_PUT: r_do_mutation(&c, BEPS_MSG_SET, o.u_key, o.value, op_index); break;
            case BEPS_OP_ADD: r_do_mutation(&c, BEPS_MSG_ADD, o.u_key, o.value, op_index); break;
            case BEPS_OP_DELETE: r_do_mutation(&c, BEPS_MSG_DEL, o.u_key, 0, op_index); break;
            case BEPS_OP_POINT_QUERY: r_do_point(&c, o.u_aux, o.u_key, op_index); break;
            case BEPS_OP_RANGE_QUERY: r_do_range(&c, o.u_aux, o.u_key, o.u_key2, o.value, op_index, scratch, scratch_cap); break;
            case BEPS_OP_FLUSH: r_do_flush(&c, o.value, o.value2, op_index); break;
            default:
                c.s.counts->invalid_count += 1;
                r_emit_event(&c, BEPS_EV_INVALID, op_index, 0, UINT64_MAX, 255, UINT64_MAX, 0, UINT64_MAX);
                break;
        }
    }

    s.event_seq[0] = c.event_seq;
    s.msg_seq_next[0] = c.msg_seq_next;
    s.node_id_next[0] = c.node_id_next;
    s.root_id[0] = c.root_id;
    s.buf_used[0] = c.buf_used;
}

// ---- hashing kernels (recompute from persistent state each step) ----
// canonical order: depth asc, low asc, node_id asc. Single thread.
__global__ void beps_ref_hash_kernel(BepsRefState s, uint64_t node_id_next,
                                     uint64_t* out_tsh, uint64_t* out_bh,
                                     uint64_t* out_lh) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    // build sorted list of alive ids by (depth, low, id) using selection.
    // node count small relative to caps. We do O(N^2) selection by repeatedly
    // finding the next node in canonical order greater than the previous.
    uint64_t tsh = 1469598103934665603ULL;
    uint64_t bh = 1469598103934665603ULL;
    // tree_shape + buffer hashes in canonical order
    // previous tuple sentinel
    bool have_prev = false; uint32_t pdep = 0; uint64_t plow = 0, pid = 0;
    for (;;) {
        bool found = false; uint32_t bdep = 0; uint64_t blow = 0, bid = 0;
        for (uint64_t n = 1; n < node_id_next; ++n) {
            if (!s.n_alive[n]) continue;
            uint32_t d = s.n_depth[n]; uint64_t lw = s.n_low[n];
            // must be strictly greater than prev tuple
            if (have_prev) {
                bool gt = (d > pdep) || (d == pdep && lw > plow) ||
                          (d == pdep && lw == plow && n > pid);
                if (!gt) continue;
            }
            // pick smallest tuple among candidates
            if (!found) { found = true; bdep = d; blow = lw; bid = n; }
            else {
                bool lt = (d < bdep) || (d == bdep && lw < blow) ||
                          (d == bdep && lw == blow && n < bid);
                if (lt) { bdep = d; blow = lw; bid = n; }
            }
        }
        if (!found) break;
        uint64_t id = bid;
        uint64_t parent = s.n_parent[id];
        uint8_t is_leaf = s.n_isleaf[id];
        uint32_t depth = s.n_depth[id];
        uint64_t low = s.n_low[id], high = s.n_high[id];
        uint64_t cc = is_leaf ? 0 : (uint64_t)s.n_childcnt[id];
        uint64_t bc = is_leaf ? 0 : (uint64_t)s.n_bufcnt[id];
        uint64_t rc = is_leaf ? (uint64_t)s.n_reccnt[id] : 0;
        beps_r_fnv(&tsh, &id, 8); beps_r_fnv(&tsh, &parent, 8);
        beps_r_fnv(&tsh, &is_leaf, 1); beps_r_fnv(&tsh, &depth, 4);
        beps_r_fnv(&tsh, &low, 8); beps_r_fnv(&tsh, &high, 8);
        beps_r_fnv(&tsh, &cc, 8); beps_r_fnv(&tsh, &bc, 8); beps_r_fnv(&tsh, &rc, 8);
        if (!is_leaf) {
            // arena scan in index (== msg_seq) order for entries owned by id.
            for (int i = 0; i < s.buf_used[0]; ++i) {
                if (s.buf_owner[i] != id) continue;
                uint64_t nid = id; uint64_t seq = s.buf_seq[i];
                uint8_t kind = s.buf_kind[i]; uint64_t key = s.buf_key[i];
                int64_t val = s.buf_val[i];
                beps_r_fnv(&bh, &nid, 8); beps_r_fnv(&bh, &seq, 8);
                beps_r_fnv(&bh, &kind, 1); beps_r_fnv(&bh, &key, 8);
                beps_r_fnv(&bh, &val, 8);
            }
        }
        have_prev = true; pdep = bdep; plow = blow; pid = bid;
    }
    *out_tsh = tsh;
    *out_bh = bh;

    // leaf hash: leaves by (low, id) ascending
    uint64_t lh = 1469598103934665603ULL;
    bool hp = false; uint64_t plw = 0, pi = 0;
    for (;;) {
        bool found = false; uint64_t blw = 0, bi = 0;
        for (uint64_t n = 1; n < node_id_next; ++n) {
            if (!s.n_alive[n] || !s.n_isleaf[n]) continue;
            uint64_t lw = s.n_low[n];
            if (hp) {
                bool gt = (lw > plw) || (lw == plw && n > pi);
                if (!gt) continue;
            }
            if (!found) { found = true; blw = lw; bi = n; }
            else { bool lt = (lw < blw) || (lw == blw && n < bi); if (lt) { blw = lw; bi = n; } }
        }
        if (!found) break;
        uint64_t id = bi;
        uint64_t lb = id * s.rec_stride;
        int rc = s.n_reccnt[id];
        for (int i = 0; i < rc; ++i) {
            uint64_t leaf_node = id; uint64_t key = s.rec_key[lb + i];
            int64_t value = s.rec_val[lb + i]; uint64_t ls = s.rec_seq[lb + i];
            uint8_t del = s.rec_del[lb + i];
            beps_r_fnv(&lh, &leaf_node, 8); beps_r_fnv(&lh, &key, 8);
            beps_r_fnv(&lh, &value, 8); beps_r_fnv(&lh, &ls, 8);
            beps_r_fnv(&lh, &del, 1);
        }
        hp = true; plw = blw; pi = bi;
    }
    *out_lh = lh;
}

// add the missing spec accessor used above (compile shim removed)
// (we referenced c->spec_dummy() erroneously; provide via macro shim removed)

// ============================ host side ============================

static cudaError_t beps_ref_reset(BepsRefState* st, cudaStream_t stream) {
    size_t ncap = (size_t)st->node_cap;
    cudaError_t e;
    e = cudaMemsetAsync(st->n_alive, 0, sizeof(uint8_t)*ncap, stream); if (e) return e;
    e = cudaMemsetAsync(st->n_childcnt, 0, sizeof(int32_t)*ncap, stream); if (e) return e;
    e = cudaMemsetAsync(st->n_bufcnt, 0, sizeof(int32_t)*ncap, stream); if (e) return e;
    e = cudaMemsetAsync(st->n_reccnt, 0, sizeof(int32_t)*ncap, stream); if (e) return e;
    e = cudaMemsetAsync(st->counts, 0, sizeof(BepsCounts), stream); if (e) return e;

    // initialize root node id=1 on host then copy.
    uint64_t one = 1, zero = 0, maxk = UINT64_MAX;
    uint32_t z32 = 0; uint8_t leaf1 = 1, alive1 = 1;
    e = cudaMemcpyAsync(&st->n_parent[1], &zero, 8, cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(&st->n_low[1], &zero, 8, cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(&st->n_high[1], &maxk, 8, cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(&st->n_depth[1], &z32, 4, cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(&st->n_isleaf[1], &leaf1, 1, cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(&st->n_alive[1], &alive1, 1, cudaMemcpyHostToDevice, stream); if (e) return e;

    uint64_t msn = 1, nin = 2, rid = 1;  // node_id_next=2 (1 already used), root=1
    uint64_t es = 0;
    uint64_t basis = 1469598103934665603ULL;
    e = cudaMemcpyAsync(st->event_seq, &es, 8, cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(st->msg_seq_next, &msn, 8, cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(st->node_id_next, &nin, 8, cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(st->root_id, &rid, 8, cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(st->meh, &basis, 8, cudaMemcpyHostToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(st->qh, &basis, 8, cudaMemcpyHostToDevice, stream); if (e) return e;
    int32_t z = 0;
    e = cudaMemcpyAsync(st->buf_used, &z, sizeof(int32_t), cudaMemcpyHostToDevice, stream); if (e) return e;
    return cudaSuccess;
}

extern "C" size_t solution_workspace_bytes(const BepsProblemSpec* spec) {
    if (!beps_validate_problem_spec(spec)) return 0;
    // scratch for range candidate keys
    size_t cands = (size_t)BEPS_MAX_TOTAL_SLOTS;
    return cands * sizeof(uint64_t);
}

extern "C" cudaError_t solution_init(const BepsProblemSpec* spec, void** state_out,
                                     cudaStream_t stream) {
    if (!beps_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    BepsRefState* st = (BepsRefState*)malloc(sizeof(BepsRefState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(BepsRefState));
    memcpy(&st->spec, spec, sizeof(BepsProblemSpec));

    st->node_cap = spec->max_nodes;
    st->max_children = spec->max_children_per_internal + 2;  // headroom for transient overflow
    st->rec_stride = spec->leaf_record_cap + spec->flush_message_cap + 2;
    // Arena capacity: a buffered message is created per root-buffer append, so
    // the live+dead arena never exceeds total mutations across the run, bounded
    // by max_ops*max_steps. Clamp to a global ceiling.
    {
        long long cap = (long long)(spec->max_ops > 0 ? spec->max_ops : 1) *
                        (long long)(spec->max_steps > 0 ? spec->max_steps : 1) + 16;
        if (cap > (long long)BEPS_MAX_TOTAL_SLOTS) cap = (long long)BEPS_MAX_TOTAL_SLOTS;
        if (cap < 16) cap = 16;
        st->buf_cap = (int32_t)cap;
    }

    size_t ncap = (size_t)st->node_cap;
    size_t bcap = (size_t)st->buf_cap;
    cudaError_t e;
#define ALLOC(p, bytes) do { e = cudaMalloc((void**)&st->p, (bytes)); if (e) goto fail; } while(0)
    ALLOC(n_parent, sizeof(uint64_t)*ncap);
    ALLOC(n_low, sizeof(uint64_t)*ncap);
    ALLOC(n_high, sizeof(uint64_t)*ncap);
    ALLOC(n_depth, sizeof(uint32_t)*ncap);
    ALLOC(n_isleaf, sizeof(uint8_t)*ncap);
    ALLOC(n_alive, sizeof(uint8_t)*ncap);
    ALLOC(n_childcnt, sizeof(int32_t)*ncap);
    ALLOC(n_bufcnt, sizeof(int32_t)*ncap);
    ALLOC(n_reccnt, sizeof(int32_t)*ncap);
    ALLOC(child, sizeof(uint64_t)*ncap*(size_t)st->max_children);
    ALLOC(buf_owner, sizeof(uint64_t)*bcap);
    ALLOC(buf_seq, sizeof(uint64_t)*bcap);
    ALLOC(buf_kind, sizeof(uint8_t)*bcap);
    ALLOC(buf_key, sizeof(uint64_t)*bcap);
    ALLOC(buf_val, sizeof(int64_t)*bcap);
    ALLOC(rec_key, sizeof(uint64_t)*ncap*(size_t)st->rec_stride);
    ALLOC(rec_val, sizeof(int64_t)*ncap*(size_t)st->rec_stride);
    ALLOC(rec_seq, sizeof(uint64_t)*ncap*(size_t)st->rec_stride);
    ALLOC(rec_del, sizeof(uint8_t)*ncap*(size_t)st->rec_stride);
    ALLOC(event_seq, sizeof(uint64_t));
    ALLOC(msg_seq_next, sizeof(uint64_t));
    ALLOC(node_id_next, sizeof(uint64_t));
    ALLOC(root_id, sizeof(uint64_t));
    ALLOC(buf_used, sizeof(int32_t));
    ALLOC(counts, sizeof(BepsCounts));
    ALLOC(meh, sizeof(uint64_t));
    ALLOC(qh, sizeof(uint64_t));
#undef ALLOC

    e = beps_ref_reset(st, stream);
    if (e) goto fail;
    *state_out = st;
    return cudaSuccess;
fail:
    solution_destroy(st);
    return e ? e : cudaErrorMemoryAllocation;
}

extern "C" cudaError_t solution_run(void* state, const BepsRunSpec* run,
                                    const void* inputs_void, void* outputs_void,
                                    void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    if (!state || !beps_validate_run_spec(run) || !inputs_void || !outputs_void)
        return cudaErrorInvalidValue;
    BepsRefState* st = (BepsRefState*)state;
    const BepsInputs* in = (const BepsInputs*)inputs_void;
    BepsOutputs* out = (BepsOutputs*)outputs_void;
    if (run->num_ops > st->spec.max_ops) return cudaErrorInvalidValue;

    int64_t scratch_cap = (int64_t)(workspace_bytes / sizeof(uint64_t));

    if (run->num_ops > 0) {
        if (!in->ops) return cudaErrorInvalidValue;
        beps_ref_kernel<<<1,1,0,stream>>>(*st, in->ops, run->num_ops,
                                          (uint64_t*)workspace, scratch_cap);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    // copy persistent accumulators into outputs and compute structural hashes
    cudaError_t e;
    e = cudaMemcpyAsync(out->counts, st->counts, sizeof(BepsCounts),
                        cudaMemcpyDeviceToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(out->message_event_hash, st->meh, sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream); if (e) return e;
    e = cudaMemcpyAsync(out->query_hash, st->qh, sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream); if (e) return e;

    // need node_id_next on host to bound hash loops; read it back.
    uint64_t nin = 0;
    e = cudaMemcpyAsync(&nin, st->node_id_next, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost, stream); if (e) return e;
    e = cudaStreamSynchronize(stream); if (e) return e;

    beps_ref_hash_kernel<<<1,1,0,stream>>>(*st, nin, out->tree_shape_hash,
                                          out->buffer_hash, out->leaf_hash);
    e = cudaPeekAtLastError(); if (e) return e;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return beps_ref_reset((BepsRefState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    BepsRefState* st = (BepsRefState*)state;
#define FREE(p) do { if (st->p) cudaFree(st->p); } while(0)
    FREE(n_parent); FREE(n_low); FREE(n_high); FREE(n_depth); FREE(n_isleaf);
    FREE(n_alive); FREE(n_childcnt); FREE(n_bufcnt); FREE(n_reccnt); FREE(child);
    FREE(buf_owner); FREE(buf_seq); FREE(buf_kind); FREE(buf_key); FREE(buf_val);
    FREE(rec_key); FREE(rec_val); FREE(rec_seq); FREE(rec_del);
    FREE(event_seq); FREE(msg_seq_next); FREE(node_id_next); FREE(root_id);
    FREE(buf_used); FREE(counts); FREE(meh); FREE(qh);
#undef FREE
    free(st);
}
