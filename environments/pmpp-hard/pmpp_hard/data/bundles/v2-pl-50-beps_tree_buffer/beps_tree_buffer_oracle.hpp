// file: beps_tree_buffer_oracle.hpp
//
// Independent CPU reference (the source of truth) for beps_tree_buffer.
// Implemented with std:: containers and explicit node structs. Children are
// stored as a vector kept sorted by low_key; buffers as a vector kept sorted
// by msg_seq; leaf records as a std::map keyed by key. Canonical hashing
// traversals exactly follow the contract.

#ifndef BEPS_TREE_BUFFER_ORACLE_HPP_
#define BEPS_TREE_BUFFER_ORACLE_HPP_

#include "beps_tree_buffer_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct BepsHostInputsView {
    const BepsOp* ops;
};

struct BepsHostOutputsView {
    const BepsCounts* counts;
    const uint64_t* message_event_hash;
    const uint64_t* query_hash;
    const uint64_t* tree_shape_hash;
    const uint64_t* buffer_hash;
    const uint64_t* leaf_hash;
};

static inline uint64_t beps_o_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= 1099511628211ULL;
    return h;
}
static inline void beps_o_fnv(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = beps_o_fnv_byte(v, b[i]);
    *h = v;
}

struct BepsORecord {
    int64_t value = 0;
    uint64_t latest_seq = 0;
    uint8_t deleted = 0;
};

struct BepsOMsg {
    uint64_t msg_seq = 0;
    uint8_t kind = 0;
    uint64_t key = 0;
    int64_t value = 0;
};

struct BepsONode {
    uint64_t node_id = 0;
    uint64_t parent = 0;   // ZERO == none
    uint64_t low_key = 0;
    uint64_t high_key = 0;
    uint32_t depth = 0;
    uint8_t is_leaf = 1;
    std::vector<uint64_t> children;     // child node ids, ascending low_key
    std::vector<BepsOMsg> buffer;       // ascending msg_seq
    std::map<uint64_t, BepsORecord> records;  // leaf records by key
    bool alive = false;
};

struct BepsOracleState {
    BepsProblemSpec spec{};
    std::vector<BepsONode> nodes;  // index by node_id (0 unused)
    uint64_t event_seq = 0;
    uint64_t msg_seq_next = 1;
    uint64_t node_id_next = 1;
    uint64_t root_id = 0;

    // accumulators (persist across steps)
    BepsCounts counts{};
    uint64_t message_event_hash = 0;
    uint64_t query_hash = 0;

    void init(const BepsProblemSpec& s) { spec = s; reset(); }

    void reset() {
        nodes.clear();
        // Reserve full capacity up front so the vector NEVER reallocates while
        // node references are held across alloc_node() during splits. Node ids
        // are monotonic and bounded by spec.max_nodes, so this reservation is
        // sufficient and keeps all BepsONode& references stable.
        nodes.reserve((size_t)spec.max_nodes + 8);
        nodes.resize(1);  // index 0 unused
        event_seq = 0;
        msg_seq_next = 1;
        node_id_next = 1;
        // single leaf root id=1, range [0, UINT64_MAX]
        uint64_t r = alloc_node();
        BepsONode& root = nodes[r];
        root.parent = 0;
        root.low_key = 0;
        root.high_key = UINT64_MAX;
        root.depth = 0;
        root.is_leaf = 1;
        root_id = r;
        std::memset(&counts, 0, sizeof(counts));
        message_event_hash = 1469598103934665603ULL;
        query_hash = 1469598103934665603ULL;
    }

    uint64_t alloc_node() {
        uint64_t id = node_id_next++;
        if (id >= nodes.size()) nodes.resize((size_t)id + 1);
        nodes[(size_t)id] = BepsONode{};
        nodes[(size_t)id].node_id = id;
        nodes[(size_t)id].alive = true;
        return id;
    }

    // ---- emission helpers ----
    void emit_event(uint8_t ekind, uint32_t op_index, uint64_t node_id,
                    uint64_t msg_seq, uint8_t kind, uint64_t key,
                    int64_t value, uint64_t child_node) {
        uint64_t es = event_seq++;
        uint64_t* h = &message_event_hash;
        beps_o_fnv(h, &ekind, 1);
        beps_o_fnv(h, &es, 8);
        beps_o_fnv(h, &op_index, 4);
        beps_o_fnv(h, &node_id, 8);
        beps_o_fnv(h, &msg_seq, 8);
        beps_o_fnv(h, &kind, 1);
        beps_o_fnv(h, &key, 8);
        beps_o_fnv(h, &value, 8);
        beps_o_fnv(h, &child_node, 8);
    }

    void emit_point(uint64_t read_id, uint32_t op_index, uint64_t key,
                    uint8_t found, int64_t value, uint64_t latest_seq) {
        uint8_t rk = BEPS_REC_POINT_RESULT;
        uint64_t* h = &query_hash;
        beps_o_fnv(h, &rk, 1);
        beps_o_fnv(h, &read_id, 8);
        beps_o_fnv(h, &op_index, 4);
        beps_o_fnv(h, &key, 8);
        beps_o_fnv(h, &found, 1);
        beps_o_fnv(h, &value, 8);
        beps_o_fnv(h, &latest_seq, 8);
    }
    void emit_range_result(uint64_t read_id, uint32_t op_index, uint64_t key,
                           int64_t value, uint64_t latest_seq) {
        uint8_t rk = BEPS_REC_RANGE_RESULT;
        uint64_t* h = &query_hash;
        beps_o_fnv(h, &rk, 1);
        beps_o_fnv(h, &read_id, 8);
        beps_o_fnv(h, &op_index, 4);
        beps_o_fnv(h, &key, 8);
        beps_o_fnv(h, &value, 8);
        beps_o_fnv(h, &latest_seq, 8);
    }
    void emit_range_end(uint64_t read_id, uint32_t op_index, uint64_t cnt) {
        uint8_t rk = BEPS_REC_RANGE_END;
        uint64_t* h = &query_hash;
        beps_o_fnv(h, &rk, 1);
        beps_o_fnv(h, &read_id, 8);
        beps_o_fnv(h, &op_index, 4);
        beps_o_fnv(h, &cnt, 8);
    }

    // ---- leaf application ----
    // Applies one message to a leaf's record map. emit_leaf controls whether a
    // LEAF_APPLY_* event is produced (true for flush/direct apply).
    void apply_to_leaf(uint64_t leaf, const BepsOMsg& m, uint32_t op_index,
                       bool emit_leaf) {
        BepsONode& L = nodes[(size_t)leaf];
        auto it = L.records.find(m.key);
        if (m.kind == BEPS_MSG_SET) {
            BepsORecord& r = L.records[m.key];
            r.value = m.value;
            r.latest_seq = m.msg_seq;
            r.deleted = 0;
            if (emit_leaf)
                emit_event(BEPS_EV_LEAF_APPLY_SET, op_index, leaf, m.msg_seq,
                           BEPS_MSG_SET, m.key, m.value, UINT64_MAX);
        } else if (m.kind == BEPS_MSG_ADD) {
            int64_t base = 0;
            if (it != L.records.end() && it->second.deleted == 0)
                base = it->second.value;
            uint64_t nv = (uint64_t)base + (uint64_t)m.value;  // mod 2^64
            BepsORecord& r = L.records[m.key];
            r.value = (int64_t)nv;
            r.latest_seq = m.msg_seq;
            r.deleted = 0;
            if (emit_leaf)
                emit_event(BEPS_EV_LEAF_APPLY_ADD, op_index, leaf, m.msg_seq,
                           BEPS_MSG_ADD, m.key, m.value, UINT64_MAX);
        } else {  // DEL
            BepsORecord& r = L.records[m.key];
            r.value = 0;
            r.latest_seq = m.msg_seq;
            r.deleted = 1;
            if (emit_leaf)
                emit_event(BEPS_EV_LEAF_APPLY_DEL, op_index, leaf, m.msg_seq,
                           BEPS_MSG_DEL, m.key, m.value, UINT64_MAX);
        }
    }

    // ---- routing ----
    // descend internal node `n`: pick child whose [low,high] contains key.
    uint64_t child_for_key(uint64_t n, uint64_t key) const {
        const BepsONode& N = nodes[(size_t)n];
        for (uint64_t c : N.children) {
            const BepsONode& C = nodes[(size_t)c];
            if (key >= C.low_key && key <= C.high_key) return c;
        }
        return 0;  // should not happen (ranges partition)
    }

    uint64_t find_leaf(uint64_t key) const {
        uint64_t cur = root_id;
        while (!nodes[(size_t)cur].is_leaf) {
            cur = child_for_key(cur, key);
        }
        return cur;
    }

    // ---- splits ----
    void update_depths_subtree(uint64_t n, uint32_t depth) {
        nodes[(size_t)n].depth = depth;
        if (!nodes[(size_t)n].is_leaf) {
            // copy children to avoid reference invalidation (no alloc here)
            std::vector<uint64_t> ch = nodes[(size_t)n].children;
            for (uint64_t c : ch) update_depths_subtree(c, depth + 1);
        }
    }

    void insert_child_after(uint64_t parent, uint64_t left, uint64_t right) {
        BepsONode& P = nodes[(size_t)parent];
        for (size_t i = 0; i < P.children.size(); ++i) {
            if (P.children[i] == left) {
                P.children.insert(P.children.begin() + (i + 1), right);
                return;
            }
        }
        P.children.push_back(right);
    }

    void split_leaf(uint64_t leaf, uint32_t op_index) {
        // gather sorted keys (std::map already sorted ascending)
        std::vector<uint64_t> keys;
        keys.reserve(nodes[(size_t)leaf].records.size());
        for (auto& kv : nodes[(size_t)leaf].records) keys.push_back(kv.first);
        const size_t rc = keys.size();
        const uint64_t split_key = keys[rc / 2];

        const uint64_t old_low = nodes[(size_t)leaf].low_key;
        const uint64_t old_high = nodes[(size_t)leaf].high_key;
        const uint64_t old_parent = nodes[(size_t)leaf].parent;
        const uint32_t old_depth = nodes[(size_t)leaf].depth;

        const uint64_t right = alloc_node();
        // move records >= split_key to right
        {
            BepsONode& Lref = nodes[(size_t)leaf];
            BepsONode& Rref = nodes[(size_t)right];
            Rref.is_leaf = 1;
            Rref.low_key = split_key;
            Rref.high_key = old_high;
            std::vector<uint64_t> moved;
            for (auto& kv : Lref.records)
                if (kv.first >= split_key) moved.push_back(kv.first);
            for (uint64_t k : moved) {
                Rref.records[k] = Lref.records[k];
                Lref.records.erase(k);
            }
            Lref.high_key = split_key - 1;
        }

        // NOTE: the contract names LEAF_SPLIT / INTERNAL_SPLIT as structural
        // events, but the message_event_hash event-kind enumeration does NOT
        // include them. Only the leaf_splits / internal_splits COUNTS capture
        // splits; no hashed mutation/flush event is emitted for a split.
        counts.leaf_splits += 1;

        if (leaf == root_id) {
            // new internal root
            const uint64_t nroot = alloc_node();
            BepsONode& NR = nodes[(size_t)nroot];
            NR.is_leaf = 0;
            NR.low_key = old_low;
            NR.high_key = old_high;
            NR.depth = old_depth;       // new root takes old root depth
            NR.parent = 0;
            NR.children.push_back(leaf);
            NR.children.push_back(right);
            nodes[(size_t)leaf].parent = nroot;
            nodes[(size_t)right].parent = nroot;
            nodes[(size_t)leaf].depth = old_depth + 1;
            nodes[(size_t)right].depth = old_depth + 1;
            root_id = nroot;
        } else {
            nodes[(size_t)right].parent = old_parent;
            nodes[(size_t)right].depth = old_depth;
            insert_child_after(old_parent, leaf, right);
            if (nodes[(size_t)old_parent].children.size() >
                (size_t)spec.max_children_per_internal) {
                split_internal(old_parent, op_index);
            }
        }
    }

    void split_internal(uint64_t node, uint32_t op_index) {
        BepsONode& N = nodes[(size_t)node];
        const size_t m = N.children.size();
        const size_t sci = m / 2;

        const uint64_t old_low = N.low_key;
        const uint64_t old_high = N.high_key;
        const uint64_t old_parent = N.parent;
        const uint32_t old_depth = N.depth;

        // snapshot children + buffer (alloc may invalidate references)
        std::vector<uint64_t> ch = N.children;
        std::vector<BepsOMsg> buf = N.buffer;

        const uint64_t left_high = nodes[(size_t)ch[sci - 1]].high_key;
        const uint64_t right_low = nodes[(size_t)ch[sci]].low_key;

        const uint64_t right = alloc_node();
        {
            BepsONode& Lref = nodes[(size_t)node];
            BepsONode& Rref = nodes[(size_t)right];
            Rref.is_leaf = 0;
            Rref.low_key = right_low;
            Rref.high_key = old_high;
            Rref.depth = old_depth;   // sibling at same depth as node (fixed up if root)
            // children split
            Rref.children.assign(ch.begin() + sci, ch.end());
            Lref.children.assign(ch.begin(), ch.begin() + sci);
            Lref.high_key = left_high;
            // reparent right's children
            for (uint64_t c : Rref.children) nodes[(size_t)c].parent = right;
            // redistribute buffer by key range, preserving msg_seq order
            std::vector<BepsOMsg> lbuf, rbuf;
            for (const BepsOMsg& mm : buf) {
                if (mm.key >= right_low && mm.key <= old_high) rbuf.push_back(mm);
                else lbuf.push_back(mm);
            }
            Lref.buffer = lbuf;
            Rref.buffer = rbuf;
        }

        counts.internal_splits += 1;

        if (node == root_id) {
            const uint64_t nroot = alloc_node();
            BepsONode& NR = nodes[(size_t)nroot];
            NR.is_leaf = 0;
            NR.low_key = old_low;
            NR.high_key = old_high;
            NR.depth = old_depth;
            NR.parent = 0;
            NR.children.push_back(node);
            NR.children.push_back(right);
            nodes[(size_t)node].parent = nroot;
            nodes[(size_t)right].parent = nroot;
            root_id = nroot;
            // shift both subtrees down by one level
            update_depths_subtree(node, old_depth + 1);
            update_depths_subtree(right, old_depth + 1);
        } else {
            nodes[(size_t)right].parent = old_parent;
            update_depths_subtree(node, old_depth);
            update_depths_subtree(right, old_depth);
            insert_child_after(old_parent, node, right);
            if (nodes[(size_t)old_parent].children.size() >
                (size_t)spec.max_children_per_internal) {
                split_internal(old_parent, op_index);
            }
        }
    }

    // ---- buffer insert (keeps msg_seq ascending; appended msgs already are) ----
    void buffer_append(uint64_t node, const BepsOMsg& m) {
        nodes[(size_t)node].buffer.push_back(m);  // appended in ascending seq
    }

    // ---- flush a single group from a chosen source; returns msgs moved ----
    // Returns number of messages moved. Caller handles budgets/eligibility.
    int64_t flush_group(uint64_t source, int64_t max_messages, uint32_t op_index) {
        BepsONode& S = nodes[(size_t)source];
        // choose child receiving most messages by key range.
        // counts per child index.
        const size_t nc = S.children.size();
        std::vector<int64_t> cnt(nc, 0);
        for (const BepsOMsg& mm : S.buffer) {
            for (size_t ci = 0; ci < nc; ++ci) {
                const BepsONode& C = nodes[(size_t)S.children[ci]];
                if (mm.key >= C.low_key && mm.key <= C.high_key) { cnt[ci] += 1; break; }
            }
        }
        // pick max count; tie smallest child low_key then node_id.
        int best = -1;
        for (size_t ci = 0; ci < nc; ++ci) {
            if (cnt[ci] <= 0) continue;
            if (best < 0) { best = (int)ci; continue; }
            const BepsONode& Cb = nodes[(size_t)S.children[best]];
            const BepsONode& Cc = nodes[(size_t)S.children[ci]];
            if (cnt[ci] > cnt[best] ||
                (cnt[ci] == cnt[best] && (Cc.low_key < Cb.low_key ||
                  (Cc.low_key == Cb.low_key && Cc.node_id < Cb.node_id)))) {
                best = (int)ci;
            }
        }
        if (best < 0) return 0;  // no child has messages (shouldn't happen if buffer nonempty)

        const uint64_t child = S.children[best];
        const uint64_t clow = nodes[(size_t)child].low_key;
        const uint64_t chigh = nodes[(size_t)child].high_key;

        // collect messages targeting child in msg_seq ascending order, capped.
        int64_t cap = max_messages;
        if ((int64_t)spec.flush_message_cap < cap) cap = spec.flush_message_cap;
        std::vector<BepsOMsg> moved;
        std::vector<BepsOMsg> remain;
        for (const BepsOMsg& mm : S.buffer) {
            if (mm.key >= clow && mm.key <= chigh && (int64_t)moved.size() < cap)
                moved.push_back(mm);
            else
                remain.push_back(mm);
        }
        S.buffer = remain;

        const bool child_leaf = nodes[(size_t)child].is_leaf;
        for (const BepsOMsg& mm : moved) {
            emit_event(BEPS_EV_FLUSH_MESSAGE, op_index, source, mm.msg_seq,
                       mm.kind, mm.key, mm.value, child);
        }
        counts.flush_messages += (uint64_t)moved.size();

        if (child_leaf) {
            for (const BepsOMsg& mm : moved) {
                apply_to_leaf(child, mm, op_index, true);
                if (mm.kind == BEPS_MSG_SET) counts.leaf_apply_set += 1;
                else if (mm.kind == BEPS_MSG_ADD) counts.leaf_apply_add += 1;
                else counts.leaf_apply_del += 1;
            }
            if (nodes[(size_t)child].records.size() > (size_t)spec.leaf_record_cap)
                split_leaf(child, op_index);
        } else {
            for (const BepsOMsg& mm : moved) buffer_append(child, mm);
        }
        return (int64_t)moved.size();
    }

    // choose eligible source; returns 0 if none.
    uint64_t choose_flush_source() const {
        uint64_t best = 0;
        for (size_t i = 1; i < nodes.size(); ++i) {
            if (!nodes[i].alive) continue;
            if (nodes[i].is_leaf) continue;
            if (nodes[i].buffer.empty()) continue;
            if (best == 0) { best = (uint64_t)i; continue; }
            const BepsONode& B = nodes[(size_t)best];
            const BepsONode& C = nodes[i];
            const size_t bl = B.buffer.size(), cl = C.buffer.size();
            bool take = false;
            if (cl > bl) take = true;
            else if (cl == bl) {
                if (C.depth < B.depth) take = true;
                else if (C.depth == B.depth) {
                    if (C.low_key < B.low_key) take = true;
                    else if (C.low_key == B.low_key && C.node_id < B.node_id) take = true;
                }
            }
            if (take) best = (uint64_t)i;
        }
        return best;
    }

    // execute FLUSH op
    void do_flush(int64_t max_flush_nodes, int64_t max_messages_total,
                  uint32_t op_index) {
        if (max_flush_nodes <= 0 || max_messages_total <= 0) return;  // no-op
        while (max_flush_nodes > 0 && max_messages_total > 0) {
            uint64_t src = choose_flush_source();
            if (src == 0) {
                emit_event(BEPS_EV_FLUSH_EMPTY, op_index, 0, UINT64_MAX, 255,
                           UINT64_MAX, 0, UINT64_MAX);
                counts.flush_empty += 1;
                break;
            }
            int64_t moved = flush_group(src, max_messages_total, op_index);
            counts.flush_groups += 1;
            max_flush_nodes -= 1;
            max_messages_total -= moved;
        }
    }

    // one automatic flush group from root (PUT-when-full). Unbounded by op
    // budgets but capped by flush_message_cap (cap handled inside flush_group).
    void auto_flush_root(uint32_t op_index) {
        if (nodes[(size_t)root_id].buffer.empty()) return;
        flush_group(root_id, (int64_t)INT64_MAX, op_index);
        counts.flush_groups += 1;
    }

    // ---- mutations ----
    void do_mutation(uint8_t kind, uint64_t key, int64_t value, uint32_t op_index) {
        if (nodes[(size_t)root_id].is_leaf) {
            const uint64_t seq = msg_seq_next++;
            BepsOMsg m; m.msg_seq = seq; m.kind = kind; m.key = key;
            m.value = (kind == BEPS_MSG_DEL) ? 0 : value;
            apply_to_leaf(root_id, m, op_index, true);
            if (kind == BEPS_MSG_SET) counts.leaf_apply_set += 1;
            else if (kind == BEPS_MSG_ADD) counts.leaf_apply_add += 1;
            else counts.leaf_apply_del += 1;
            if (nodes[(size_t)root_id].records.size() > (size_t)spec.leaf_record_cap)
                split_leaf(root_id, op_index);
            return;
        }
        // root internal
        if (nodes[(size_t)root_id].buffer.size() >= (size_t)spec.internal_buffer_cap) {
            auto_flush_root(op_index);
        }
        if (nodes[(size_t)root_id].buffer.size() >= (size_t)spec.internal_buffer_cap) {
            // still full -> WRITE_STALL, no msg_seq consumed
            emit_event(BEPS_EV_WRITE_STALL, op_index, root_id, UINT64_MAX,
                       kind, key, (kind == BEPS_MSG_DEL) ? 0 : value, UINT64_MAX);
            counts.write_stall += 1;
            return;
        }
        const uint64_t seq = msg_seq_next++;
        BepsOMsg m; m.msg_seq = seq; m.kind = kind; m.key = key;
        m.value = (kind == BEPS_MSG_DEL) ? 0 : value;
        buffer_append(root_id, m);
        if (kind == BEPS_MSG_SET) {
            emit_event(BEPS_EV_ROOT_BUFFER_SET, op_index, root_id, seq,
                       BEPS_MSG_SET, key, m.value, UINT64_MAX);
            counts.root_buffered_set += 1;
        } else if (kind == BEPS_MSG_ADD) {
            emit_event(BEPS_EV_ROOT_BUFFER_ADD, op_index, root_id, seq,
                       BEPS_MSG_ADD, key, m.value, UINT64_MAX);
            counts.root_buffered_add += 1;
        } else {
            emit_event(BEPS_EV_ROOT_BUFFER_DEL, op_index, root_id, seq,
                       BEPS_MSG_DEL, key, m.value, UINT64_MAX);
            counts.root_buffered_del += 1;
        }
    }

    // ---- visibility (point semantics) for a key ----
    // Returns found, value, latest_seq.
    void visible(uint64_t key, uint8_t* found, int64_t* value, uint64_t* latest_seq) const {
        // gather path from root to leaf for key; collect buffered messages for key
        std::vector<BepsOMsg> msgs;
        uint64_t cur = root_id;
        uint64_t leaf = 0;
        while (true) {
            const BepsONode& N = nodes[(size_t)cur];
            if (N.is_leaf) { leaf = cur; break; }
            for (const BepsOMsg& mm : N.buffer)
                if (mm.key == key) msgs.push_back(mm);
            cur = child_for_key(cur, key);
        }
        // seed leaf record
        uint8_t f = 0; int64_t v = 0; uint64_t ls = 0;
        const BepsONode& Lf = nodes[(size_t)leaf];
        auto it = Lf.records.find(key);
        bool have_leaf = (it != Lf.records.end());
        uint64_t leaf_seq = 0;
        if (have_leaf) {
            leaf_seq = it->second.latest_seq;
        }
        // build combined event list: leaf record (as a pseudo-event at its
        // latest_seq) + path msgs, then sort by msg_seq ascending and apply.
        // Represent leaf record as: if deleted -> DEL-like; else SET-like value.
        struct Ev { uint64_t seq; uint8_t kind; int64_t value; };
        std::vector<Ev> evs;
        if (have_leaf) {
            Ev e; e.seq = leaf_seq;
            if (it->second.deleted) { e.kind = BEPS_MSG_DEL; e.value = 0; }
            else { e.kind = BEPS_MSG_SET; e.value = it->second.value; }
            evs.push_back(e);
        }
        for (const BepsOMsg& mm : msgs) {
            Ev e; e.seq = mm.msg_seq; e.kind = mm.kind; e.value = mm.value;
            evs.push_back(e);
        }
        std::stable_sort(evs.begin(), evs.end(),
                         [](const Ev& a, const Ev& b){ return a.seq < b.seq; });
        // apply globally
        bool missing = true;  // missing or deleted -> not found
        for (const Ev& e : evs) {
            if (e.kind == BEPS_MSG_SET) { f = 1; v = e.value; ls = e.seq; missing = false; }
            else if (e.kind == BEPS_MSG_ADD) {
                int64_t base = missing ? 0 : v;
                v = (int64_t)((uint64_t)base + (uint64_t)e.value);
                f = 1; ls = e.seq; missing = false;
            } else {  // DEL
                f = 0; v = 0; ls = e.seq; missing = true;
            }
        }
        *found = f; *value = v; *latest_seq = ls;
    }

    void do_point_query(uint64_t read_id, uint64_t key, uint32_t op_index) {
        uint8_t found; int64_t value; uint64_t latest_seq;
        visible(key, &found, &value, &latest_seq);
        if (found) {
            counts.point_found += 1;
            emit_point(read_id, op_index, key, 1, value, latest_seq);
        } else {
            counts.point_missing += 1;
            emit_point(read_id, op_index, key, 0, INT64_MIN, 0);
        }
    }

    void do_range_query(uint64_t read_id, uint64_t lo, uint64_t hi, int64_t limit,
                        uint32_t op_index) {
        if (lo > hi) { counts.invalid_count += 1; return; }
        int64_t eff = limit;
        if ((int64_t)spec.max_range_results < eff) eff = spec.max_range_results;
        if (eff < 0) eff = 0;
        // candidate keys: in any leaf record OR any internal buffer message,
        // within [lo,hi]. Collect distinct ascending.
        std::vector<uint64_t> cand;
        for (size_t i = 1; i < nodes.size(); ++i) {
            if (!nodes[i].alive) continue;
            if (nodes[i].is_leaf) {
                for (auto& kv : nodes[i].records)
                    if (kv.first >= lo && kv.first <= hi) cand.push_back(kv.first);
            } else {
                for (const BepsOMsg& mm : nodes[i].buffer)
                    if (mm.key >= lo && mm.key <= hi) cand.push_back(mm.key);
            }
        }
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

        uint64_t emitted = 0;
        for (uint64_t k : cand) {
            if ((int64_t)emitted >= eff) break;
            uint8_t found; int64_t value; uint64_t latest_seq;
            visible(k, &found, &value, &latest_seq);
            if (found) {
                emit_range_result(read_id, op_index, k, value, latest_seq);
                counts.range_results += 1;
                emitted += 1;
            }
        }
        emit_range_end(read_id, op_index, emitted);
        counts.range_end_count += 1;
    }

    // ---- canonical traversal order: depth asc, low_key asc, node_id asc ----
    std::vector<uint64_t> canonical_nodes() const {
        std::vector<uint64_t> ids;
        for (size_t i = 1; i < nodes.size(); ++i)
            if (nodes[i].alive) ids.push_back((uint64_t)i);
        std::sort(ids.begin(), ids.end(), [&](uint64_t a, uint64_t b){
            const BepsONode& A = nodes[(size_t)a];
            const BepsONode& B = nodes[(size_t)b];
            if (A.depth != B.depth) return A.depth < B.depth;
            if (A.low_key != B.low_key) return A.low_key < B.low_key;
            return A.node_id < B.node_id;
        });
        return ids;
    }

    uint64_t tree_shape_hash() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<uint64_t> ids = canonical_nodes();
        for (uint64_t id : ids) {
            const BepsONode& N = nodes[(size_t)id];
            uint64_t node_id = N.node_id;
            uint64_t parent = N.parent;
            uint8_t is_leaf = N.is_leaf;
            uint32_t depth = N.depth;
            uint64_t low = N.low_key, high = N.high_key;
            uint64_t cc = N.is_leaf ? 0 : (uint64_t)N.children.size();
            uint64_t bc = N.is_leaf ? 0 : (uint64_t)N.buffer.size();
            uint64_t rc = N.is_leaf ? (uint64_t)N.records.size() : 0;
            beps_o_fnv(&h, &node_id, 8);
            beps_o_fnv(&h, &parent, 8);
            beps_o_fnv(&h, &is_leaf, 1);
            beps_o_fnv(&h, &depth, 4);
            beps_o_fnv(&h, &low, 8);
            beps_o_fnv(&h, &high, 8);
            beps_o_fnv(&h, &cc, 8);
            beps_o_fnv(&h, &bc, 8);
            beps_o_fnv(&h, &rc, 8);
        }
        return h;
    }

    uint64_t buffer_hash() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<uint64_t> ids = canonical_nodes();
        for (uint64_t id : ids) {
            const BepsONode& N = nodes[(size_t)id];
            if (N.is_leaf) continue;
            for (const BepsOMsg& mm : N.buffer) {  // already msg_seq ascending
                uint64_t node_id = N.node_id;
                uint64_t seq = mm.msg_seq;
                uint8_t kind = mm.kind;
                uint64_t key = mm.key;
                int64_t value = mm.value;
                beps_o_fnv(&h, &node_id, 8);
                beps_o_fnv(&h, &seq, 8);
                beps_o_fnv(&h, &kind, 1);
                beps_o_fnv(&h, &key, 8);
                beps_o_fnv(&h, &value, 8);
            }
        }
        return h;
    }

    uint64_t leaf_hash() const {
        uint64_t h = 1469598103934665603ULL;
        // leaves by low_key then node_id
        std::vector<uint64_t> ids;
        for (size_t i = 1; i < nodes.size(); ++i)
            if (nodes[i].alive && nodes[i].is_leaf) ids.push_back((uint64_t)i);
        std::sort(ids.begin(), ids.end(), [&](uint64_t a, uint64_t b){
            const BepsONode& A = nodes[(size_t)a];
            const BepsONode& B = nodes[(size_t)b];
            if (A.low_key != B.low_key) return A.low_key < B.low_key;
            return A.node_id < B.node_id;
        });
        for (uint64_t id : ids) {
            const BepsONode& N = nodes[(size_t)id];
            for (auto& kv : N.records) {  // ascending key
                uint64_t leaf_node = N.node_id;
                uint64_t key = kv.first;
                int64_t value = kv.second.value;
                uint64_t ls = kv.second.latest_seq;
                uint8_t deleted = kv.second.deleted;
                beps_o_fnv(&h, &leaf_node, 8);
                beps_o_fnv(&h, &key, 8);
                beps_o_fnv(&h, &value, 8);
                beps_o_fnv(&h, &ls, 8);
                beps_o_fnv(&h, &deleted, 1);
            }
        }
        return h;
    }

    void step(const BepsRunSpec& run, const BepsHostInputsView& in) {
        for (int i = 0; i < run.num_ops; ++i) {
            const BepsOp& o = in.ops[i];
            const uint32_t op_index = (uint32_t)i;
            switch (o.kind) {
                case BEPS_OP_PUT:
                    do_mutation(BEPS_MSG_SET, o.u_key, o.value, op_index); break;
                case BEPS_OP_ADD:
                    do_mutation(BEPS_MSG_ADD, o.u_key, o.value, op_index); break;
                case BEPS_OP_DELETE:
                    do_mutation(BEPS_MSG_DEL, o.u_key, 0, op_index); break;
                case BEPS_OP_POINT_QUERY:
                    do_point_query(o.u_aux, o.u_key, op_index); break;
                case BEPS_OP_RANGE_QUERY:
                    do_range_query(o.u_aux, o.u_key, o.u_key2, o.value, op_index); break;
                case BEPS_OP_FLUSH:
                    do_flush(o.value, o.value2, op_index); break;
                default:
                    counts.invalid_count += 1;
                    emit_event(BEPS_EV_INVALID, op_index, 0, UINT64_MAX, 255,
                               UINT64_MAX, 0, UINT64_MAX);
                    break;
            }
        }
    }

    void fill_expected(BepsCounts* out_counts, uint64_t* meh, uint64_t* qh,
                       uint64_t* tsh, uint64_t* bh, uint64_t* lh) const {
        *out_counts = counts;
        *meh = message_event_hash;
        *qh = query_hash;
        *tsh = tree_shape_hash();
        *bh = buffer_hash();
        *lh = leaf_hash();
    }
};

static inline bool beps_check_outputs(
    const BepsOracleState& oracle,
    const BepsHostOutputsView& got,
    std::string* error) {
    BepsCounts ec; uint64_t meh, qh, tsh, bh, lh;
    oracle.fill_expected(&ec, &meh, &qh, &tsh, &bh, &lh);

    const uint64_t* e = (const uint64_t*)&ec;
    const uint64_t* g = (const uint64_t*)got.counts;
    const char* names[] = {
        "root_buffered_set","root_buffered_add","root_buffered_del",
        "write_stall","flush_groups","flush_messages","leaf_apply_set",
        "leaf_apply_add","leaf_apply_del","leaf_splits","internal_splits",
        "point_found","point_missing","range_results","range_end_count",
        "flush_empty","invalid_count","reserved"};
    for (int i = 0; i < 18; ++i) {
        if (e[i] != g[i]) {
            if (error) { std::ostringstream o; o << "count " << names[i]
                << " mismatch: got " << g[i] << " expected " << e[i]; *error = o.str(); }
            return false;
        }
    }
#define BEPS_CMP_H(field, ev) \
    if (*got.field != ev) { if (error) { std::ostringstream o; \
        o << #field " mismatch: got 0x" << std::hex << *got.field \
          << " expected 0x" << ev; *error = o.str(); } return false; }
    BEPS_CMP_H(message_event_hash, meh)
    BEPS_CMP_H(query_hash, qh)
    BEPS_CMP_H(tree_shape_hash, tsh)
    BEPS_CMP_H(buffer_hash, bh)
    BEPS_CMP_H(leaf_hash, lh)
#undef BEPS_CMP_H
    return true;
}

#endif  // BEPS_TREE_BUFFER_ORACLE_HPP_
