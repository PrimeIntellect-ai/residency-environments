// file: consistent_hash_ring_oracle.hpp
//
// Authoritative host reference model for T56 (Dynamo-style consistent ring).
// reference.cu and naive.cu must reproduce these outputs byte-for-byte.
//
// DETERMINISTIC INTERPRETATIONS of T56 ambiguities (documented):
//
//  * FNV1a64(seed, a, b, ...) hashes the 8 little-endian bytes of `seed`, then
//    of each subsequent u64 argument, with the standard FNV-1a-64 basis/prime.
//      token_pos     = FNV1a64(ring_seed, node_id, ordinal)
//      key_hash      = FNV1a64(key_seed, key_id)
//
//  * Ring order is (token_pos, token_seq, node_id, vnode_ordinal) ascending,
//    all compared as unsigned.
//
//  * Token collision: if FNV pos already taken by ANY existing token, increment
//    pos modulo 2^64 until free. Collision steps emit no event.
//
//  * Preference list for key K: start at the first ring token whose token_pos is
//    >= key_hash (clockwise from key_hash); if none, wrap to ring start. Walk
//    clockwise collecting DISTINCT physical node_ids whose node state is ACTIVE
//    or JOINING. Skip duplicate physical nodes and nodes in LEAVING/DOWN/REMOVED.
//    Stop after (replication_factor + preference_list_extra) distinct nodes, or
//    after a full single wrap of the ring.
//
//  * Desired replicas = the first replication_factor ACTIVE nodes (in pref-list
//    order). JOINING nodes occupy pref-list slots but are not desired servers.
//    desired rank = position in the desired (ACTIVE-only) sequence, 0-based.
//    rank 0 kind PRIMARY; later ranks kind NORMAL.
//
//  * "first desired active node not already serving" (LEAVE/REBALANCE-add) walks
//    the desired list in rank order and picks the first ACTIVE desired node that
//    has no serving replica for the key.
//
//  * FAIL hint target: first ACTIVE node in the preference list (pref-list order)
//    that is not already serving the key. Its rank is that node's desired rank if
//    it is a desired node, else the next free desired rank == current desired set
//    size is not used; we assign rank = its index among desired nodes if present,
//    otherwise rank = replication_factor + (position among non-desired pref nodes).
//    To keep this fully deterministic and simple, the hint target's rank is the
//    smallest desired rank not currently occupied by a serving replica; if all
//    desired ranks are occupied, rank = number of distinct serving replicas.
//    (See chr_first_free_rank.)
//
//  * Capacity: a serving replica consumes one used_slot on its physical node. A
//    HINTED replica consumes one slot on the hint holder. "full" means
//    used_slots >= capacity.
//
//  * Counters are cumulative since reset; the six hashes describe full state /
//    cumulative emission streams.

#ifndef CONSISTENT_HASH_RING_ORACLE_HPP_
#define CONSISTENT_HASH_RING_ORACLE_HPP_

#include "consistent_hash_ring_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// ---------------- FNV-1a-64 ----------------
static inline uint64_t chr_o_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= 1099511628211ULL;
    return h;
}
static inline void chr_o_fnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = chr_o_fnv_byte(v, b[i]);
    *h = v;
}
static inline void chr_o_fnv_u64(uint64_t* h, uint64_t x) { chr_o_fnv_bytes(h, &x, 8); }
static inline void chr_o_fnv_u32(uint64_t* h, uint32_t x) { chr_o_fnv_bytes(h, &x, 4); }
static inline void chr_o_fnv_u8(uint64_t* h, uint8_t x) { chr_o_fnv_bytes(h, &x, 1); }
static inline void chr_o_fnv_i64(uint64_t* h, int64_t x) { chr_o_fnv_bytes(h, &x, 8); }

// FNV1a64(seed, args...) for token/key hashing.
static inline uint64_t chr_o_hash_pos(uint64_t seed, uint64_t a, uint64_t b) {
    uint64_t h = 1469598103934665603ULL;
    chr_o_fnv_u64(&h, seed);
    chr_o_fnv_u64(&h, a);
    chr_o_fnv_u64(&h, b);
    return h;
}
static inline uint64_t chr_o_hash_key(uint64_t seed, uint64_t a) {
    uint64_t h = 1469598103934665603ULL;
    chr_o_fnv_u64(&h, seed);
    chr_o_fnv_u64(&h, a);
    return h;
}

// ---------------- State structures ----------------
struct ChrNode {
    uint64_t node_id;
    int32_t state;
    uint64_t capacity;
    uint64_t used_slots;
    uint64_t node_seq;
    uint64_t last_state_seq;
};

struct ChrToken {
    uint64_t token_pos;
    uint64_t token_seq;
    uint64_t node_id;
    uint32_t vnode_ordinal;
};

struct ChrReplica {
    uint32_t rank;
    uint64_t node_id;
    uint8_t kind;            // PRIMARY/NORMAL/HINTED
    uint64_t hint_for_node;  // or 0
    uint8_t serving;         // 0/1
    uint64_t repair_seq;
};

struct ChrKey {
    uint64_t key_id;
    uint64_t key_hash;
    int64_t value;
    uint64_t version_seq;
    uint8_t deleted;
    uint64_t key_seq;
    std::vector<ChrReplica> replicas;
};

struct ChrMove {
    uint64_t move_seq;
    uint64_t key_id;
    uint8_t task_kind;
    uint64_t from_node;   // or 0
    uint64_t to_node;     // or 0
    uint32_t rank;
    uint64_t hint_for_node; // or 0
    uint64_t target_version_seq;
    uint8_t created_reason;
};

struct ChrOracle {
    ChrProblemSpec spec{};

    uint64_t event_seq = 0;
    uint64_t node_seq_next = 1;
    uint64_t token_seq_next = 1;
    uint64_t key_seq_next = 1;
    uint64_t version_seq_next = 1;
    uint64_t move_seq_next = 1;

    std::vector<ChrNode> nodes;     // keyed by node_id (unordered; we search)
    std::vector<ChrToken> tokens;   // unsorted; ring order computed on demand
    std::vector<ChrKey> keys;       // keyed by key_id
    std::vector<ChrMove> moves;     // ordered by move_seq (append order == seq order)

    ChrCounters ctr{};
    uint64_t ring_event_hash = 1469598103934665603ULL;
    uint64_t lookup_hash = 1469598103934665603ULL;

    // ---- helpers ----
    void init(const ChrProblemSpec& s) { spec = s; reset(); }

    void reset() {
        event_seq = 0;
        node_seq_next = 1;
        token_seq_next = 1;
        key_seq_next = 1;
        version_seq_next = 1;
        move_seq_next = 1;
        nodes.clear();
        tokens.clear();
        keys.clear();
        moves.clear();
        std::memset(&ctr, 0, sizeof(ctr));
        ring_event_hash = 1469598103934665603ULL;
        lookup_hash = 1469598103934665603ULL;
    }

    int find_node(uint64_t id) const {
        for (size_t i = 0; i < nodes.size(); ++i) if (nodes[i].node_id == id) return (int)i;
        return -1;
    }
    int find_key(uint64_t id) const {
        for (size_t i = 0; i < keys.size(); ++i) if (keys[i].key_id == id) return (int)i;
        return -1;
    }

    bool pos_taken(uint64_t pos) const {
        for (const auto& t : tokens) if (t.token_pos == pos) return true;
        return false;
    }

    // Emit one event into ring_event_hash (cumulative stream).
    void emit(uint8_t kind, uint32_t op_index, uint64_t node_id, uint64_t key_id,
              uint64_t token_pos, uint32_t rank, int64_t value, uint64_t aux) {
        event_seq += 1;
        uint64_t* h = &ring_event_hash;
        chr_o_fnv_u8(h, kind);
        chr_o_fnv_u64(h, event_seq);
        chr_o_fnv_u32(h, op_index);
        chr_o_fnv_u64(h, node_id);
        chr_o_fnv_u64(h, key_id);
        chr_o_fnv_u64(h, token_pos);
        chr_o_fnv_u32(h, rank);
        chr_o_fnv_i64(h, value);
        chr_o_fnv_u64(h, aux);
    }

    // Ring order: indices into tokens[] sorted (pos, seq, node_id, ordinal).
    std::vector<int> ring_order() const {
        std::vector<int> idx(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) idx[i] = (int)i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            const ChrToken& x = tokens[a];
            const ChrToken& y = tokens[b];
            if (x.token_pos != y.token_pos) return x.token_pos < y.token_pos;
            if (x.token_seq != y.token_seq) return x.token_seq < y.token_seq;
            if (x.node_id != y.node_id) return x.node_id < y.node_id;
            return x.vnode_ordinal < y.vnode_ordinal;
        });
        return idx;
    }

    // Preference list for a key_hash: distinct physical node_ids (ACTIVE|JOINING),
    // clockwise from key_hash, wrapping once, capped at RF+extra.
    std::vector<uint64_t> preference_list(uint64_t key_hash) const {
        std::vector<int> order = ring_order();
        const int m = (int)order.size();
        std::vector<uint64_t> pref;
        if (m == 0) return pref;

        // find first index with token_pos >= key_hash
        int start = 0;
        bool found = false;
        for (int i = 0; i < m; ++i) {
            if (tokens[order[i]].token_pos >= key_hash) { start = i; found = true; break; }
        }
        if (!found) start = 0; // wrap to ring start

        const int cap = spec.replication_factor + spec.preference_list_extra;

        for (int step = 0; step < m; ++step) {
            int ti = order[(start + step) % m];
            uint64_t nid = tokens[ti].node_id;
            int ni = find_node(nid);
            if (ni < 0) continue;
            int st = nodes[ni].state;
            if (st != CHR_NODE_ACTIVE && st != CHR_NODE_JOINING) continue;
            // distinct physical node only
            bool dup = false;
            for (uint64_t e : pref) if (e == nid) { dup = true; break; }
            if (dup) continue;
            pref.push_back(nid);
            if ((int)pref.size() >= cap) break;
        }
        return pref;
    }

    // Desired replicas = first RF ACTIVE nodes from pref list, in order.
    std::vector<uint64_t> desired_replicas(uint64_t key_hash) const {
        std::vector<uint64_t> pref = preference_list(key_hash);
        std::vector<uint64_t> des;
        for (uint64_t nid : pref) {
            int ni = find_node(nid);
            if (ni >= 0 && nodes[ni].state == CHR_NODE_ACTIVE) {
                des.push_back(nid);
                if ((int)des.size() >= spec.replication_factor) break;
            }
        }
        return des;
    }

    bool node_full(int ni) const {
        return nodes[ni].used_slots >= nodes[ni].capacity;
    }

    int key_replica_on(const ChrKey& k, uint64_t nid) const {
        for (size_t i = 0; i < k.replicas.size(); ++i) if (k.replicas[i].node_id == nid) return (int)i;
        return -1;
    }
    int key_serving_on(const ChrKey& k, uint64_t nid) const {
        for (size_t i = 0; i < k.replicas.size(); ++i)
            if (k.replicas[i].node_id == nid && k.replicas[i].serving) return (int)i;
        return -1;
    }

    // smallest desired rank with no serving replica occupying it; if all occupied,
    // returns number of distinct serving replicas.
    uint32_t first_free_rank(const ChrKey& k, const std::vector<uint64_t>& des) const {
        // ranks occupied by serving replicas mapped via their stored rank
        for (uint32_t r = 0; r < (uint32_t)des.size(); ++r) {
            bool occupied = false;
            for (const auto& rep : k.replicas) {
                if (rep.serving && rep.rank == r) { occupied = true; break; }
            }
            if (!occupied) return r;
        }
        uint32_t serving_count = 0;
        for (const auto& rep : k.replicas) if (rep.serving) ++serving_count;
        return serving_count;
    }

    void enqueue_move(uint32_t op_index, uint64_t key_id, uint8_t task_kind,
                      uint64_t from_node, uint64_t to_node, uint32_t rank,
                      uint64_t hint_for_node, uint64_t target_version, uint8_t reason) {
        ChrMove mv;
        mv.move_seq = move_seq_next++;
        mv.key_id = key_id;
        mv.task_kind = task_kind;
        mv.from_node = from_node;
        mv.to_node = to_node;
        mv.rank = rank;
        mv.hint_for_node = hint_for_node;
        mv.target_version_seq = target_version;
        mv.created_reason = reason;
        moves.push_back(mv);
        ctr.move_tasks_enqueued += 1;
        emit(CHR_EV_MOVE_ENQUEUE, op_index, to_node ? to_node : from_node, key_id,
             UINT64_MAX, rank, 0, mv.move_seq);
    }

    // -------- Desired-set rebalance enqueue for a key after activation --------
    // Enqueue ADD_REPLICA tasks for desired ACTIVE nodes not yet serving.
    void enqueue_desired_repairs(uint32_t op_index, ChrKey& k, uint8_t reason) {
        std::vector<uint64_t> des = desired_replicas(k.key_hash);
        for (uint32_t r = 0; r < (uint32_t)des.size(); ++r) {
            uint64_t nid = des[r];
            if (key_serving_on(k, nid) >= 0) continue;
            enqueue_move(op_index, k.key_id, CHR_TASK_ADD_REPLICA, 0, nid, r, 0,
                         k.version_seq, reason);
        }
    }

    // After a completed task, drop extra serving replicas beyond RF, worst rank
    // descending then node id ascending.
    void enqueue_drop_extras(uint32_t op_index, ChrKey& k) {
        std::vector<uint64_t> des = desired_replicas(k.key_hash);
        // serving replicas whose node is NOT in current desired set, or beyond RF.
        // Per spec: if extra serving replicas remain beyond replication_factor.
        int serving_count = 0;
        for (const auto& rep : k.replicas) if (rep.serving) ++serving_count;
        int extras = serving_count - spec.replication_factor;
        if (extras <= 0) return;

        // candidate serving replicas, sorted worst-rank desc then node asc
        std::vector<int> cand;
        for (size_t i = 0; i < k.replicas.size(); ++i)
            if (k.replicas[i].serving) cand.push_back((int)i);
        std::sort(cand.begin(), cand.end(), [&](int a, int b) {
            if (k.replicas[a].rank != k.replicas[b].rank)
                return k.replicas[a].rank > k.replicas[b].rank; // worst (highest) rank first
            return k.replicas[a].node_id < k.replicas[b].node_id;
        });
        for (int e = 0; e < extras && e < (int)cand.size(); ++e) {
            const ChrReplica& rep = k.replicas[cand[e]];
            enqueue_move(op_index, k.key_id, CHR_TASK_DROP_REPLICA, rep.node_id, 0,
                         rep.rank, 0, k.version_seq, CHR_REASON_EXPLICIT_REBALANCE);
        }
    }

    // ================= Operation dispatch =================
    void step(const ChrRunSpec& run) {
        const uint32_t op_index = (uint32_t)run.op_index;
        switch (run.op_type) {
            case CHR_OP_ADD_NODE:      op_add_node(op_index, (uint64_t)run.arg_a, run.arg_b, (uint64_t)run.arg_c); break;
            case CHR_OP_ACTIVATE_NODE: op_activate(op_index, (uint64_t)run.arg_a); break;
            case CHR_OP_START_LEAVE:   op_start_leave(op_index, (uint64_t)run.arg_a); break;
            case CHR_OP_FAIL_NODE:     op_fail(op_index, (uint64_t)run.arg_a); break;
            case CHR_OP_RECOVER_NODE:  op_recover(op_index, (uint64_t)run.arg_a); break;
            case CHR_OP_REMOVE_NODE:   op_remove(op_index, (uint64_t)run.arg_a); break;
            case CHR_OP_PUT_KEY:       op_put(op_index, (uint64_t)run.arg_a, run.arg_b); break;
            case CHR_OP_DELETE_KEY:    op_delete(op_index, (uint64_t)run.arg_a); break;
            case CHR_OP_LOOKUP:        op_lookup(op_index, (uint64_t)run.arg_a, (uint64_t)run.arg_b); break;
            case CHR_OP_REBALANCE:     op_rebalance(op_index, run.arg_a); break;
            default: invalid(op_index); break;
        }
    }

    void invalid(uint32_t op_index) {
        ctr.invalid_count += 1;
        emit(CHR_EV_INVALID, op_index, 0, 0, UINT64_MAX, UINT32_MAX, INT64_MIN, 0);
    }

    // ---- ADD_NODE ----
    void op_add_node(uint32_t op_index, uint64_t node_id, int64_t vnode_count, uint64_t capacity) {
        int ni = find_node(node_id);
        bool exists_nonremoved = (ni >= 0 && nodes[ni].state != CHR_NODE_REMOVED);
        if (exists_nonremoved || (int)nodes.size() >= spec.max_nodes ||
            vnode_count == 0 ||
            (int)tokens.size() + (int)vnode_count > spec.max_vnodes ||
            vnode_count < 0) {
            invalid(op_index);
            return;
        }
        uint64_t cap = capacity == 0 ? spec.default_node_capacity : capacity;

        // If a REMOVED node record with same id exists, reuse the slot (overwrite).
        ChrNode nd;
        nd.node_id = node_id;
        nd.state = CHR_NODE_JOINING;
        nd.capacity = cap;
        nd.used_slots = 0;
        nd.node_seq = node_seq_next++;
        nd.last_state_seq = 0;
        if (ni >= 0) nodes[ni] = nd; else nodes.push_back(nd);

        for (uint32_t ord = 0; ord < (uint32_t)vnode_count; ++ord) {
            uint64_t pos = chr_o_hash_pos(spec.ring_seed, node_id, ord);
            while (pos_taken(pos)) pos += 1; // mod 2^64 wrap is natural for u64
            ChrToken tk;
            tk.token_pos = pos;
            tk.token_seq = token_seq_next++;
            tk.node_id = node_id;
            tk.vnode_ordinal = ord;
            tokens.push_back(tk);
            ctr.vnode_added += 1;
            emit(CHR_EV_VNODE_ADD, op_index, node_id, 0, pos, ord, 0, tk.token_seq);
        }
        ctr.nodes_added += 1;
        emit(CHR_EV_NODE_ADD, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, nd.node_seq);
    }

    // ---- ACTIVATE_NODE ----
    void op_activate(uint32_t op_index, uint64_t node_id) {
        int ni = find_node(node_id);
        if (ni < 0 || nodes[ni].state != CHR_NODE_JOINING) { invalid(op_index); return; }
        nodes[ni].state = CHR_NODE_ACTIVE;
        ctr.nodes_activated += 1;
        emit(CHR_EV_NODE_ACTIVATE, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
        nodes[ni].last_state_seq = event_seq;

        // rebalance for nondeleted keys whose desired set changes; key_hash then key_id.
        std::vector<int> order = keys_order();
        for (int ki : order) {
            ChrKey& k = keys[ki];
            if (k.deleted) continue;
            // does node_id become a desired server now?
            std::vector<uint64_t> des = desired_replicas(k.key_hash);
            bool in_des = false;
            for (uint64_t d : des) if (d == node_id) { in_des = true; break; }
            if (!in_des) continue;
            if (key_serving_on(k, node_id) >= 0) continue;
            // enqueue add at this node's desired rank
            for (uint32_t r = 0; r < (uint32_t)des.size(); ++r) {
                if (des[r] == node_id) {
                    enqueue_move(op_index, k.key_id, CHR_TASK_ADD_REPLICA, 0, node_id, r,
                                 0, k.version_seq, CHR_REASON_JOIN);
                    break;
                }
            }
        }
    }

    // ---- START_LEAVE ----
    void op_start_leave(uint32_t op_index, uint64_t node_id) {
        int ni = find_node(node_id);
        if (ni < 0 || nodes[ni].state != CHR_NODE_ACTIVE) { invalid(op_index); return; }
        nodes[ni].state = CHR_NODE_LEAVING;
        ctr.nodes_leaving += 1;
        emit(CHR_EV_NODE_LEAVE, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
        nodes[ni].last_state_seq = event_seq;

        std::vector<int> order = keys_order();
        for (int ki : order) {
            ChrKey& k = keys[ki];
            if (k.deleted) continue;
            if (key_serving_on(k, node_id) < 0) continue;
            // first desired active node not already serving
            std::vector<uint64_t> des = desired_replicas(k.key_hash);
            int target_rank = -1; uint64_t target = 0;
            for (uint32_t r = 0; r < (uint32_t)des.size(); ++r) {
                if (key_serving_on(k, des[r]) < 0) { target = des[r]; target_rank = (int)r; break; }
            }
            if (target_rank < 0) {
                ctr.no_target_count += 1;
                emit(CHR_EV_LEAVE_NO_TARGET, op_index, node_id, k.key_id, UINT64_MAX, UINT32_MAX, 0, 0);
                continue;
            }
            enqueue_move(op_index, k.key_id, CHR_TASK_ADD_REPLICA, 0, target, (uint32_t)target_rank,
                         0, k.version_seq, CHR_REASON_LEAVE);
        }
    }

    // ---- FAIL_NODE ----
    void op_fail(uint32_t op_index, uint64_t node_id) {
        int ni = find_node(node_id);
        if (ni < 0 || nodes[ni].state == CHR_NODE_DOWN || nodes[ni].state == CHR_NODE_REMOVED) {
            invalid(op_index); return;
        }
        nodes[ni].state = CHR_NODE_DOWN;
        ctr.nodes_failed += 1;
        emit(CHR_EV_NODE_FAIL, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
        nodes[ni].last_state_seq = event_seq;

        // Mark old serving replicas on failed node nonserving first (release slots).
        std::vector<int> order = keys_order();
        for (int ki : order) {
            ChrKey& k = keys[ki];
            int ri = key_replica_on(k, node_id);
            if (ri >= 0 && k.replicas[ri].serving) {
                k.replicas[ri].serving = 0;
                if (nodes[ni].used_slots > 0) nodes[ni].used_slots -= 1;
            }
        }
        // Then enqueue hinted handoff for each affected nondeleted key.
        for (int ki : order) {
            ChrKey& k = keys[ki];
            if (k.deleted) continue;
            int ri = key_replica_on(k, node_id);
            if (ri < 0) continue; // had a replica record on failed node
            // first active node in preference list not already serving
            std::vector<uint64_t> pref = preference_list(k.key_hash);
            uint64_t target = 0; bool have = false;
            for (uint64_t nid : pref) {
                int xi = find_node(nid);
                if (xi < 0 || nodes[xi].state != CHR_NODE_ACTIVE) continue;
                if (key_serving_on(k, nid) >= 0) continue;
                target = nid; have = true; break;
            }
            if (!have) {
                ctr.no_target_count += 1;
                emit(CHR_EV_FAIL_NO_HINT_TARGET, op_index, node_id, k.key_id, UINT64_MAX, UINT32_MAX, 0, 0);
                continue;
            }
            uint32_t rank = first_free_rank(k, desired_replicas(k.key_hash));
            enqueue_move(op_index, k.key_id, CHR_TASK_ADD_REPLICA, 0, target, rank, node_id,
                         k.version_seq, CHR_REASON_FAIL);
        }
    }

    // ---- RECOVER_NODE ----
    void op_recover(uint32_t op_index, uint64_t node_id) {
        int ni = find_node(node_id);
        if (ni < 0 || nodes[ni].state != CHR_NODE_DOWN) { invalid(op_index); return; }
        nodes[ni].state = CHR_NODE_ACTIVE;
        ctr.nodes_recovered += 1;
        emit(CHR_EV_NODE_RECOVER, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
        nodes[ni].last_state_seq = event_seq;

        // For every hinted replica with hint_for_node == node_id, enqueue HANDOFF
        // from hint holder to recovered node, in key order.
        std::vector<int> order = keys_order();
        for (int ki : order) {
            ChrKey& k = keys[ki];
            for (size_t r = 0; r < k.replicas.size(); ++r) {
                const ChrReplica& rep = k.replicas[r];
                if (rep.kind == CHR_KIND_HINTED && rep.hint_for_node == node_id) {
                    enqueue_move(op_index, k.key_id, CHR_TASK_HANDOFF_HINT, rep.node_id, node_id,
                                 rep.rank, node_id, k.version_seq, CHR_REASON_RECOVER);
                }
            }
        }
    }

    // ---- REMOVE_NODE ----
    void op_remove(uint32_t op_index, uint64_t node_id) {
        int ni = find_node(node_id);
        if (ni < 0 || nodes[ni].state != CHR_NODE_LEAVING) { invalid(op_index); return; }
        // any serving replica still on node?
        for (const auto& k : keys) {
            if (key_serving_on(k, node_id) >= 0) {
                ctr.remove_stalls += 1;
                emit(CHR_EV_REMOVE_STALL_REPLICAS, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
                return;
            }
        }
        // delete tokens in ring order for this node
        std::vector<int> order = ring_order();
        std::vector<uint64_t> to_remove_pos;
        for (int oi : order) {
            if (tokens[oi].node_id == node_id) {
                ctr.vnode_removed += 1;
                emit(CHR_EV_VNODE_REMOVE, op_index, node_id, 0, tokens[oi].token_pos,
                     tokens[oi].vnode_ordinal, 0, tokens[oi].token_seq);
                to_remove_pos.push_back(tokens[oi].token_pos);
            }
        }
        // erase
        std::vector<ChrToken> kept;
        for (const auto& t : tokens) if (t.node_id != node_id) kept.push_back(t);
        tokens.swap(kept);

        nodes[ni].state = CHR_NODE_REMOVED;
        ctr.nodes_removed += 1;
        emit(CHR_EV_NODE_REMOVE, op_index, node_id, 0, UINT64_MAX, UINT32_MAX, 0, 0);
        nodes[ni].last_state_seq = event_seq;
    }

    // ---- PUT_KEY ----
    void op_put(uint32_t op_index, uint64_t key_id, int64_t value) {
        int ki = find_key(key_id);
        if (ki < 0 && (int)keys.size() >= spec.max_keys) {
            // PUT_OOM: no ring-event enum slot exists; counter only (deterministic).
            ctr.put_oom += 1;
            return;
        }
        if (ki < 0) {
            ChrKey k;
            k.key_id = key_id;
            k.key_hash = chr_o_hash_key(spec.key_seed, key_id);
            k.value = value;
            k.version_seq = 0;
            k.deleted = 0;
            k.key_seq = key_seq_next++;
            keys.push_back(k);
            ki = (int)keys.size() - 1;
        }
        ChrKey& k = keys[ki];
        // resurrect if deleted: same key_hash and key_seq retained
        k.deleted = 0;
        k.value = value;
        k.version_seq = version_seq_next++;
        ctr.put_ok += 1;
        emit(CHR_EV_KEY_PUT, op_index, 0, key_id, UINT64_MAX, UINT32_MAX, value, k.version_seq);

        // direct assignment if no serving replicas and possible
        int serving_count = 0;
        for (const auto& rep : k.replicas) if (rep.serving) ++serving_count;

        if (serving_count == 0) {
            std::vector<uint64_t> des = desired_replicas(k.key_hash);
            for (uint32_t r = 0; r < (uint32_t)des.size(); ++r) {
                uint64_t nid = des[r];
                int xi = find_node(nid);
                if (xi < 0 || nodes[xi].state != CHR_NODE_ACTIVE) continue;
                if (node_full(xi)) continue;
                // install serving replica
                ChrReplica rep;
                rep.rank = r;
                rep.node_id = nid;
                rep.kind = (r == 0) ? CHR_KIND_PRIMARY : CHR_KIND_NORMAL;
                rep.hint_for_node = 0;
                rep.serving = 1;
                emit(CHR_EV_REPLICA_DIRECT_ADD, op_index, nid, key_id, UINT64_MAX, r, 0, k.version_seq);
                rep.repair_seq = event_seq;
                k.replicas.push_back(rep);
                nodes[xi].used_slots += 1;
                ctr.direct_replica_added += 1;
            }
        } else {
            // converge via repair tasks
            enqueue_desired_repairs(op_index, k, CHR_REASON_PUT_REPAIR);
        }
    }

    // ---- DELETE_KEY ----
    void op_delete(uint32_t op_index, uint64_t key_id) {
        int ki = find_key(key_id);
        if (ki < 0 || keys[ki].deleted) {
            ctr.delete_miss += 1;
            emit(CHR_EV_KEY_DELETE_MISS, op_index, 0, key_id, UINT64_MAX, UINT32_MAX, 0, 0);
            return;
        }
        ChrKey& k = keys[ki];
        k.deleted = 1;
        k.version_seq = version_seq_next++;
        ctr.delete_ok += 1;
        emit(CHR_EV_KEY_DELETE, op_index, 0, key_id, UINT64_MAX, UINT32_MAX, 0, k.version_seq);

        // DELETE_REPLICA for every replica record in node id order
        std::vector<int> ridx(k.replicas.size());
        for (size_t i = 0; i < k.replicas.size(); ++i) ridx[i] = (int)i;
        std::sort(ridx.begin(), ridx.end(), [&](int a, int b) {
            return k.replicas[a].node_id < k.replicas[b].node_id;
        });
        for (int r : ridx) {
            enqueue_move(op_index, key_id, CHR_TASK_DELETE_REPLICA, k.replicas[r].node_id, 0,
                         k.replicas[r].rank, k.replicas[r].hint_for_node, k.version_seq,
                         CHR_REASON_DELETE_REPAIR);
        }
    }

    // ---- LOOKUP ----
    void op_lookup(uint32_t op_index, uint64_t read_id, uint64_t key_id) {
        int ki = find_key(key_id);
        if (ki < 0 || keys[ki].deleted) {
            ctr.lookup_missing += 1;
            emit(CHR_EV_LOOKUP_MISSING, op_index, 0, key_id, UINT64_MAX, UINT32_MAX, 0, 0);
            // lookup record: MISSING
            lk_record(read_id, key_id, CHR_LK_MISSING, UINT32_MAX, 0, CHR_KIND_NONE, 0, INT64_MIN, UINT64_MAX);
            return;
        }
        const ChrKey& k = keys[ki];
        // serving replicas ordered by rank asc, kind(PRIMARY,NORMAL,HINTED), node id
        std::vector<int> ridx;
        for (size_t i = 0; i < k.replicas.size(); ++i) if (k.replicas[i].serving) ridx.push_back((int)i);
        std::sort(ridx.begin(), ridx.end(), [&](int a, int b) {
            if (k.replicas[a].rank != k.replicas[b].rank) return k.replicas[a].rank < k.replicas[b].rank;
            if (k.replicas[a].kind != k.replicas[b].kind) return k.replicas[a].kind < k.replicas[b].kind;
            return k.replicas[a].node_id < k.replicas[b].node_id;
        });
        int emitted = 0;
        for (int r : ridx) {
            if (emitted >= spec.replication_factor) break;
            const ChrReplica& rep = k.replicas[r];
            ctr.lookup_found_replicas += 1;
            emit(CHR_EV_LOOKUP_REPLICA, op_index, rep.node_id, key_id, UINT64_MAX, rep.rank, k.value, k.version_seq);
            lk_record(read_id, key_id, CHR_LK_REPLICA, rep.rank, rep.node_id, rep.kind,
                      rep.hint_for_node, k.value, k.version_seq);
            ++emitted;
        }
        emit(CHR_EV_LOOKUP_END, op_index, 0, key_id, UINT64_MAX, UINT32_MAX, 0, 0);
        lk_record(read_id, key_id, CHR_LK_END, UINT32_MAX, 0, CHR_KIND_NONE, 0, INT64_MIN, UINT64_MAX);
    }

    void lk_record(uint64_t read_id, uint64_t key_id, uint8_t record_kind, uint32_t rank,
                   uint64_t node_id, uint8_t kind, uint64_t hint_for, int64_t value,
                   uint64_t version_seq) {
        uint64_t* h = &lookup_hash;
        chr_o_fnv_u64(h, read_id);
        chr_o_fnv_u64(h, key_id);
        chr_o_fnv_u8(h, record_kind);
        chr_o_fnv_u32(h, rank);
        chr_o_fnv_u64(h, node_id);
        chr_o_fnv_u8(h, kind);
        chr_o_fnv_u64(h, hint_for);
        chr_o_fnv_i64(h, value);
        chr_o_fnv_u64(h, version_seq);
    }

    // ---- REBALANCE ----
    void op_rebalance(uint32_t op_index, int64_t limit) {
        if (limit == 0) return; // valid no-op
        int completed = 0;
        size_t i = 0;
        while (i < moves.size()) {
            if (completed >= (int)limit) break;
            bool remove = false;
            bool counted = false;
            bool stalled = false;
            process_task(op_index, moves[i], &remove, &counted, &stalled);
            if (remove) {
                // after completion recompute desired placement / drop extras
                if (counted) {
                    int ki = find_key(moves[i].key_id);
                    if (ki >= 0) enqueue_drop_extras(op_index, keys[ki]);
                }
                moves.erase(moves.begin() + i);
                if (counted) ++completed;
                // do not advance i (erased element); but new tasks were appended at end
            } else {
                ++i; // stalled: keep, scan later tasks
            }
        }
    }

    void process_task(uint32_t op_index, ChrMove& mv, bool* remove, bool* counted, bool* stalled) {
        *remove = false; *counted = false; *stalled = false;
        int ki = find_key(mv.key_id);

        if (mv.task_kind == CHR_TASK_ADD_REPLICA) {
            if (ki < 0 || keys[ki].deleted || keys[ki].version_seq != mv.target_version_seq) {
                ctr.move_obsolete += 1;
                emit(CHR_EV_MOVE_OBSOLETE, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, mv.move_seq);
                *remove = true; return;
            }
            int xi = find_node(mv.to_node);
            if (xi < 0 || nodes[xi].state != CHR_NODE_ACTIVE || node_full(xi)) {
                // full check only matters if dest not already holding replica
                bool dest_has = (key_replica_on(keys[ki], mv.to_node) >= 0);
                if (xi < 0 || nodes[xi].state != CHR_NODE_ACTIVE || (node_full(xi) && !dest_has)) {
                    ctr.move_stall += 1;
                    emit(CHR_EV_MOVE_STALL, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, mv.move_seq);
                    *stalled = true; return;
                }
            }
            ChrKey& k = keys[ki];
            int ri = key_replica_on(k, mv.to_node);
            uint8_t kind = mv.hint_for_node ? CHR_KIND_HINTED : ((mv.rank == 0) ? CHR_KIND_PRIMARY : CHR_KIND_NORMAL);
            if (ri >= 0) {
                // update kind/hint/rank, no capacity if already serving; if was nonserving, consume
                if (!k.replicas[ri].serving) { nodes[xi].used_slots += 1; }
                k.replicas[ri].rank = mv.rank;
                k.replicas[ri].kind = kind;
                k.replicas[ri].hint_for_node = mv.hint_for_node;
                k.replicas[ri].serving = 1;
                emit(CHR_EV_REPLICA_ADD, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, k.version_seq);
                k.replicas[ri].repair_seq = event_seq;
            } else {
                ChrReplica rep;
                rep.rank = mv.rank;
                rep.node_id = mv.to_node;
                rep.kind = kind;
                rep.hint_for_node = mv.hint_for_node;
                rep.serving = 1;
                emit(CHR_EV_REPLICA_ADD, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, k.version_seq);
                rep.repair_seq = event_seq;
                k.replicas.push_back(rep);
                nodes[xi].used_slots += 1;
            }
            ctr.replica_added += 1;
            *remove = true; *counted = true; return;
        }

        if (mv.task_kind == CHR_TASK_DROP_REPLICA) {
            if (ki < 0) { *remove = true; return; }
            ChrKey& k = keys[ki];
            int ri = key_replica_on(k, mv.from_node);
            if (ri >= 0 && k.replicas[ri].serving) {
                int xi = find_node(mv.from_node);
                if (xi >= 0 && nodes[xi].used_slots > 0) nodes[xi].used_slots -= 1;
                emit(CHR_EV_REPLICA_DROP, op_index, mv.from_node, mv.key_id, UINT64_MAX, k.replicas[ri].rank, 0, k.version_seq);
                k.replicas.erase(k.replicas.begin() + ri);
                ctr.replica_dropped += 1;
            }
            *remove = true; *counted = true; return;
        }

        if (mv.task_kind == CHR_TASK_HANDOFF_HINT) {
            if (ki < 0 || keys[ki].deleted || keys[ki].version_seq != mv.target_version_seq) {
                ctr.move_obsolete += 1;
                emit(CHR_EV_MOVE_OBSOLETE, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, mv.move_seq);
                *remove = true; return;
            }
            int xi = find_node(mv.to_node);
            bool dest_has = (key_replica_on(keys[ki], mv.to_node) >= 0);
            if (xi < 0 || nodes[xi].state != CHR_NODE_ACTIVE || (node_full(xi) && !dest_has)) {
                ctr.move_stall += 1;
                emit(CHR_EV_MOVE_STALL, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, mv.move_seq);
                *stalled = true; return;
            }
            ChrKey& k = keys[ki];
            // install normal serving replica on destination
            int ri = key_replica_on(k, mv.to_node);
            uint8_t kind = (mv.rank == 0) ? CHR_KIND_PRIMARY : CHR_KIND_NORMAL;
            if (ri >= 0) {
                if (!k.replicas[ri].serving) nodes[xi].used_slots += 1;
                k.replicas[ri].rank = mv.rank;
                k.replicas[ri].kind = kind;
                k.replicas[ri].hint_for_node = 0;
                k.replicas[ri].serving = 1;
                emit(CHR_EV_HINT_HANDOFF_ADD, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, k.version_seq);
                k.replicas[ri].repair_seq = event_seq;
            } else {
                ChrReplica rep;
                rep.rank = mv.rank; rep.node_id = mv.to_node; rep.kind = kind;
                rep.hint_for_node = 0; rep.serving = 1;
                emit(CHR_EV_HINT_HANDOFF_ADD, op_index, mv.to_node, mv.key_id, UINT64_MAX, mv.rank, 0, k.version_seq);
                rep.repair_seq = event_seq;
                k.replicas.push_back(rep);
                nodes[xi].used_slots += 1;
            }
            ctr.hint_handoff_added += 1;
            // remove hinted replica from source if still exists
            int si = key_replica_on(k, mv.from_node);
            if (si >= 0 && k.replicas[si].kind == CHR_KIND_HINTED) {
                int sxi = find_node(mv.from_node);
                if (k.replicas[si].serving && sxi >= 0 && nodes[sxi].used_slots > 0) nodes[sxi].used_slots -= 1;
                emit(CHR_EV_HINT_DROP, op_index, mv.from_node, mv.key_id, UINT64_MAX, k.replicas[si].rank, 0, k.version_seq);
                k.replicas.erase(k.replicas.begin() + si);
                ctr.hint_dropped += 1;
            }
            *remove = true; *counted = true; return;
        }

        if (mv.task_kind == CHR_TASK_DELETE_REPLICA) {
            if (ki >= 0) {
                ChrKey& k = keys[ki];
                int ri = key_replica_on(k, mv.from_node);
                if (ri >= 0) {
                    if (k.replicas[ri].serving) {
                        int xi = find_node(mv.from_node);
                        if (xi >= 0 && nodes[xi].used_slots > 0) nodes[xi].used_slots -= 1;
                    }
                    emit(CHR_EV_REPLICA_DELETE, op_index, mv.from_node, mv.key_id, UINT64_MAX, k.replicas[ri].rank, 0, k.version_seq);
                    k.replicas.erase(k.replicas.begin() + ri);
                    ctr.replica_deleted += 1;
                }
            }
            *remove = true; *counted = true; return;
        }
        *remove = true;
    }

    // keys ordered by key_hash then key_id
    std::vector<int> keys_order() const {
        std::vector<int> idx(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) idx[i] = (int)i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            if (keys[a].key_hash != keys[b].key_hash) return keys[a].key_hash < keys[b].key_hash;
            return keys[a].key_id < keys[b].key_id;
        });
        return idx;
    }

    // ================= State hashes =================
    uint64_t compute_ring_hash() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<int> order = ring_order();
        for (int oi : order) {
            chr_o_fnv_u64(&h, tokens[oi].token_pos);
            chr_o_fnv_u64(&h, tokens[oi].token_seq);
            chr_o_fnv_u64(&h, tokens[oi].node_id);
            chr_o_fnv_u32(&h, tokens[oi].vnode_ordinal);
        }
        return h;
    }
    uint64_t compute_node_hash() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<int> idx(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) idx[i] = (int)i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return nodes[a].node_id < nodes[b].node_id; });
        for (int ni : idx) {
            chr_o_fnv_u64(&h, nodes[ni].node_id);
            chr_o_fnv_u8(&h, (uint8_t)nodes[ni].state);
            chr_o_fnv_u64(&h, nodes[ni].capacity);
            chr_o_fnv_u64(&h, nodes[ni].used_slots);
            chr_o_fnv_u64(&h, nodes[ni].node_seq);
            chr_o_fnv_u64(&h, nodes[ni].last_state_seq);
        }
        return h;
    }
    uint64_t compute_key_replica_hash() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<int> order = keys_order();
        for (int ki : order) {
            const ChrKey& k = keys[ki];
            chr_o_fnv_u64(&h, k.key_id);
            chr_o_fnv_u64(&h, k.key_hash);
            chr_o_fnv_i64(&h, k.value);
            chr_o_fnv_u64(&h, k.version_seq);
            chr_o_fnv_u8(&h, k.deleted);
            chr_o_fnv_u64(&h, k.key_seq);
            std::vector<int> ridx(k.replicas.size());
            for (size_t i = 0; i < k.replicas.size(); ++i) ridx[i] = (int)i;
            std::sort(ridx.begin(), ridx.end(), [&](int a, int b) {
                if (k.replicas[a].rank != k.replicas[b].rank) return k.replicas[a].rank < k.replicas[b].rank;
                return k.replicas[a].node_id < k.replicas[b].node_id;
            });
            for (int r : ridx) {
                const ChrReplica& rep = k.replicas[r];
                chr_o_fnv_u32(&h, rep.rank);
                chr_o_fnv_u64(&h, rep.node_id);
                chr_o_fnv_u8(&h, rep.kind);
                chr_o_fnv_u64(&h, rep.hint_for_node);
                chr_o_fnv_u8(&h, rep.serving);
                chr_o_fnv_u64(&h, rep.repair_seq);
            }
        }
        return h;
    }
    uint64_t compute_move_hash() const {
        uint64_t h = 1469598103934665603ULL;
        // moves vector is already in move_seq order (append order)
        std::vector<int> idx(moves.size());
        for (size_t i = 0; i < moves.size(); ++i) idx[i] = (int)i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return moves[a].move_seq < moves[b].move_seq; });
        for (int mi : idx) {
            const ChrMove& mv = moves[mi];
            chr_o_fnv_u64(&h, mv.move_seq);
            chr_o_fnv_u64(&h, mv.key_id);
            chr_o_fnv_u8(&h, mv.task_kind);
            chr_o_fnv_u64(&h, mv.from_node);
            chr_o_fnv_u64(&h, mv.to_node);
            chr_o_fnv_u32(&h, mv.rank);
            chr_o_fnv_u64(&h, mv.hint_for_node);
            chr_o_fnv_u64(&h, mv.target_version_seq);
            chr_o_fnv_u8(&h, mv.created_reason);
        }
        return h;
    }
};

struct ChrExpected {
    ChrCounters counters{};
    uint64_t ring_event_hash = 0;
    uint64_t lookup_hash = 0;
    uint64_t ring_hash = 0;
    uint64_t node_hash = 0;
    uint64_t key_replica_hash = 0;
    uint64_t move_hash = 0;
};

static inline void chr_oracle_step(ChrOracle& o, const ChrRunSpec& run, ChrExpected* exp) {
    o.step(run);
    exp->counters = o.ctr;
    exp->ring_event_hash = o.ring_event_hash;
    exp->lookup_hash = o.lookup_hash;
    exp->ring_hash = o.compute_ring_hash();
    exp->node_hash = o.compute_node_hash();
    exp->key_replica_hash = o.compute_key_replica_hash();
    exp->move_hash = o.compute_move_hash();
}

struct ChrGotView {
    const ChrCounters* counters;
    const uint64_t* ring_event_hash;
    const uint64_t* lookup_hash;
    const uint64_t* ring_hash;
    const uint64_t* node_hash;
    const uint64_t* key_replica_hash;
    const uint64_t* move_hash;
};

static inline bool chr_check_outputs(const ChrExpected& e, const ChrGotView& g, std::string* err) {
    #define CHR_CMP_U64(field, gv, ev) do { \
        if ((gv) != (ev)) { if (err) { std::ostringstream o; o << field " mismatch: got 0x" \
            << std::hex << (gv) << " expected 0x" << (ev); *err = o.str(); } return false; } } while (0)

    const ChrCounters& gc = *g.counters;
    const ChrCounters& ec = e.counters;
    #define CHR_CMP_CTR(name) do { if (gc.name != ec.name) { if (err) { std::ostringstream o; \
        o << #name " mismatch: got " << gc.name << " expected " << ec.name; *err = o.str(); } return false; } } while (0)
    CHR_CMP_CTR(nodes_added); CHR_CMP_CTR(nodes_activated); CHR_CMP_CTR(nodes_leaving);
    CHR_CMP_CTR(nodes_failed); CHR_CMP_CTR(nodes_recovered); CHR_CMP_CTR(nodes_removed);
    CHR_CMP_CTR(vnode_added); CHR_CMP_CTR(vnode_removed); CHR_CMP_CTR(put_ok);
    CHR_CMP_CTR(put_oom); CHR_CMP_CTR(delete_ok); CHR_CMP_CTR(delete_miss);
    CHR_CMP_CTR(direct_replica_added); CHR_CMP_CTR(move_tasks_enqueued); CHR_CMP_CTR(move_obsolete);
    CHR_CMP_CTR(move_stall); CHR_CMP_CTR(replica_added); CHR_CMP_CTR(replica_dropped);
    CHR_CMP_CTR(hint_handoff_added); CHR_CMP_CTR(hint_dropped); CHR_CMP_CTR(replica_deleted);
    CHR_CMP_CTR(lookup_found_replicas); CHR_CMP_CTR(lookup_missing); CHR_CMP_CTR(remove_stalls);
    CHR_CMP_CTR(no_target_count); CHR_CMP_CTR(invalid_count);
    #undef CHR_CMP_CTR

    CHR_CMP_U64("ring_event_hash", *g.ring_event_hash, e.ring_event_hash);
    CHR_CMP_U64("lookup_hash", *g.lookup_hash, e.lookup_hash);
    CHR_CMP_U64("ring_hash", *g.ring_hash, e.ring_hash);
    CHR_CMP_U64("node_hash", *g.node_hash, e.node_hash);
    CHR_CMP_U64("key_replica_hash", *g.key_replica_hash, e.key_replica_hash);
    CHR_CMP_U64("move_hash", *g.move_hash, e.move_hash);
    #undef CHR_CMP_U64
    return true;
}

#endif  // CONSISTENT_HASH_RING_ORACLE_HPP_
