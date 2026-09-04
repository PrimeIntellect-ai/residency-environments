// file: radix_prefix_cache_tree_oracle.hpp

#ifndef RADIX_PREFIX_CACHE_TREE_ORACLE_HPP_
#define RADIX_PREFIX_CACHE_TREE_ORACLE_HPP_

#include "radix_prefix_cache_tree_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct RpctHostInputsView {
    const int32_t* op_kind;
    const int32_t* op_arg_a;
    const int32_t* op_arg_b;
    const int32_t* op_token_offset;
    const int32_t* op_token_len;
    const int32_t* op_tokens;
};

struct RpctHostOutputsView {
    const int32_t* matched_prefix_len;
    const int32_t* num_nodes;
    const int32_t* num_tokens_cached;
    const int32_t* num_evicted_nodes;
    const int32_t* evicted_tokens;
    const uint64_t* tree_checksum;
    const uint64_t* state_checksum;
};

struct RpctOpExpected {
    int32_t matched_prefix_len = -1;
    int32_t num_nodes = 0;
    int32_t num_tokens_cached = 0;
    int32_t num_evicted_nodes = 0;
    int32_t evicted_tokens = 0;
    uint64_t tree_checksum = 0;
    uint64_t state_checksum = 0;
};

static inline uint64_t rpct_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void rpct_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = rpct_oracle_fnv_byte(v, p[i]);
    *h = v;
}

// Independent CPU reference implemented with std:: containers and explicit
// node structs. Children are kept in an ordered map keyed by first token so
// that "find child by first token" is direct; canonical traversal is by id.
struct RpctOracleNode {
    int32_t id = 0;
    int32_t parent_id = -1;
    int32_t ref_count = 0;
    int64_t lru_timestamp = 0;
    std::vector<int32_t> segment;            // edge label from parent
    std::map<int32_t, int32_t> children;     // first_token -> child id
    bool alive = false;
};

struct RpctOracleState {
    RpctProblemSpec spec{};
    std::vector<RpctOracleNode> nodes;       // index by id
    int64_t clock = 0;
    int32_t next_node_id = 1;
    std::unordered_map<int32_t, int32_t> req_terminal;  // request_id -> node id

    void init(const RpctProblemSpec& s) {
        spec = s;
        reset();
    }

    void reset() {
        nodes.clear();
        nodes.resize(1);
        nodes[0] = RpctOracleNode{};
        nodes[0].id = 0;
        nodes[0].parent_id = -1;
        nodes[0].alive = true;
        clock = 0;
        next_node_id = 1;
        req_terminal.clear();
    }

    int32_t alloc_node() {
        int32_t id = next_node_id++;
        if ((size_t)id >= nodes.size()) nodes.resize((size_t)id + 1);
        nodes[(size_t)id] = RpctOracleNode{};
        nodes[(size_t)id].id = id;
        nodes[(size_t)id].alive = true;
        return id;
    }

    int32_t num_nodes() const {
        int32_t c = 0;
        for (size_t i = 1; i < nodes.size(); ++i) if (nodes[i].alive) ++c;
        return c;
    }

    int32_t num_tokens_cached() const {
        int32_t c = 0;
        for (size_t i = 1; i < nodes.size(); ++i)
            if (nodes[i].alive) c += (int32_t)nodes[i].segment.size();
        return c;
    }

    // INSERT. Returns matched_prefix_len.
    int32_t do_insert(int32_t request_id, const int32_t* tokens, int32_t len) {
        clock += 1;
        const int64_t now = clock;

        int32_t cur = 0;     // current node id, start at root
        int32_t pos = 0;     // matched token count

        while (pos < len) {
            const int32_t first = tokens[pos];
            auto it = nodes[(size_t)cur].children.find(first);
            if (it == nodes[(size_t)cur].children.end()) break;

            const int32_t child = it->second;
            // copy out the segment so no reference is held across alloc_node()
            // (alloc_node may resize `nodes` and invalidate references).
            const std::vector<int32_t> seg = nodes[(size_t)child].segment;
            int32_t k = 0;
            const int32_t seglen = (int32_t)seg.size();
            while (k < seglen && pos + k < len && seg[(size_t)k] == tokens[pos + k]) {
                ++k;
            }

            if (k == seglen) {
                // full edge consumed; descend
                cur = child;
                pos += seglen;
                continue;
            }

            // partial match k in [1, seglen): split child at k
            // (k>=1 guaranteed because first token matched)
            const int32_t inherit_ref = nodes[(size_t)child].ref_count;
            const int64_t inherit_ts = nodes[(size_t)child].lru_timestamp;
            std::vector<int32_t> prefix(seg.begin(), seg.begin() + k);
            std::vector<int32_t> suffix(seg.begin() + k, seg.end());

            const int32_t pnode = alloc_node();  // may realloc nodes
            nodes[(size_t)pnode].parent_id = cur;
            nodes[(size_t)pnode].ref_count = inherit_ref;
            nodes[(size_t)pnode].lru_timestamp = inherit_ts;
            nodes[(size_t)pnode].segment = prefix;
            // child keeps suffix
            nodes[(size_t)child].segment = suffix;
            nodes[(size_t)child].parent_id = pnode;
            // rewire: cur's child[first] now -> P; P's child[suffix[0]] -> C
            nodes[(size_t)cur].children[first] = pnode;
            nodes[(size_t)pnode].children[suffix[0]] = child;

            cur = pnode;
            pos += k;
            break;  // diverged inside the edge; matching cannot continue
        }

        const int32_t matched_prefix_len = pos;

        // append remainder as one leaf
        if (pos < len) {
            const int32_t leaf = alloc_node();  // may realloc nodes
            nodes[(size_t)leaf].parent_id = cur;
            nodes[(size_t)leaf].ref_count = 0;
            nodes[(size_t)leaf].lru_timestamp = now;
            nodes[(size_t)leaf].segment.assign(tokens + pos, tokens + len);
            nodes[(size_t)cur].children[tokens[pos]] = leaf;
            cur = leaf;
        }

        // ref-count + touch along path from terminal up to (excluding) root
        int32_t walk = cur;
        while (walk != 0) {
            nodes[(size_t)walk].ref_count += 1;
            nodes[(size_t)walk].lru_timestamp = now;
            walk = nodes[(size_t)walk].parent_id;
        }

        req_terminal[request_id] = cur;
        return matched_prefix_len;
    }

    void do_release(int32_t request_id) {
        clock += 1;
        auto it = req_terminal.find(request_id);
        if (it == req_terminal.end()) return;
        int32_t walk = it->second;
        while (walk != 0) {
            if (nodes[(size_t)walk].ref_count > 0) nodes[(size_t)walk].ref_count -= 1;
            walk = nodes[(size_t)walk].parent_id;
        }
        req_terminal.erase(it);
    }

    bool is_leaf(int32_t id) const {
        return nodes[(size_t)id].children.empty();
    }

    void do_evict(int32_t num_tokens_target, int32_t* evicted_nodes, int32_t* evicted_tokens) {
        clock += 1;
        int32_t en = 0;
        int32_t et = 0;

        if (num_tokens_target <= 0) {
            *evicted_nodes = 0;
            *evicted_tokens = 0;
            return;
        }

        while (et < num_tokens_target) {
            // find evictable leaf with smallest (lru_timestamp, id)
            int32_t best = -1;
            int64_t best_ts = 0;
            for (size_t i = 1; i < nodes.size(); ++i) {
                if (!nodes[i].alive) continue;
                if (nodes[i].ref_count != 0) continue;
                if (!nodes[i].children.empty()) continue;  // must be leaf
                const int64_t ts = nodes[i].lru_timestamp;
                if (best < 0 || ts < best_ts || (ts == best_ts && (int32_t)i < best)) {
                    best = (int32_t)i;
                    best_ts = ts;
                }
            }
            if (best < 0) break;

            RpctOracleNode& B = nodes[(size_t)best];
            const int32_t parent = B.parent_id;
            et += (int32_t)B.segment.size();
            en += 1;
            // unlink from parent
            if (parent >= 0) {
                int32_t first = B.segment.empty() ? -1 : B.segment[0];
                auto pit = nodes[(size_t)parent].children.find(first);
                if (pit != nodes[(size_t)parent].children.end() && pit->second == best) {
                    nodes[(size_t)parent].children.erase(pit);
                }
            }
            B.alive = false;
            B.children.clear();
        }

        *evicted_nodes = en;
        *evicted_tokens = et;
    }

    uint64_t tree_checksum() const {
        uint64_t h = 1469598103934665603ULL;
        int32_t cnt = num_nodes();
        rpct_oracle_fnv_bytes(&h, &cnt, sizeof(int32_t));
        for (size_t i = 1; i < nodes.size(); ++i) {
            if (!nodes[i].alive) continue;
            const RpctOracleNode& n = nodes[i];
            int32_t id = n.id;
            int32_t parent_id = n.parent_id;
            int32_t ref_count = n.ref_count;
            int32_t segment_len = (int32_t)n.segment.size();
            rpct_oracle_fnv_bytes(&h, &id, sizeof(int32_t));
            rpct_oracle_fnv_bytes(&h, &parent_id, sizeof(int32_t));
            rpct_oracle_fnv_bytes(&h, &ref_count, sizeof(int32_t));
            rpct_oracle_fnv_bytes(&h, &segment_len, sizeof(int32_t));
            for (int32_t t : n.segment) {
                rpct_oracle_fnv_bytes(&h, &t, sizeof(int32_t));
            }
        }
        return h;
    }

    uint64_t state_checksum(uint64_t tree_cs) const {
        uint64_t h = 1469598103934665603ULL;
        int64_t cl = clock;
        int32_t nn = num_nodes();
        int32_t nt = num_tokens_cached();
        int32_t nid = next_node_id;
        rpct_oracle_fnv_bytes(&h, &cl, sizeof(int64_t));
        rpct_oracle_fnv_bytes(&h, &nn, sizeof(int32_t));
        rpct_oracle_fnv_bytes(&h, &nt, sizeof(int32_t));
        rpct_oracle_fnv_bytes(&h, &nid, sizeof(int32_t));
        rpct_oracle_fnv_bytes(&h, &tree_cs, sizeof(uint64_t));
        return h;
    }

    void step(const RpctRunSpec& run, const RpctHostInputsView& in,
              std::vector<RpctOpExpected>* out) {
        out->clear();
        out->reserve((size_t)run.op_count);
        for (int i = 0; i < run.op_count; ++i) {
            RpctOpExpected e;
            const int32_t kind = in.op_kind[i];
            if (kind == RPCT_OP_INSERT) {
                const int32_t req = in.op_arg_a[i];
                const int32_t off = in.op_token_offset[i];
                const int32_t ln = in.op_token_len[i];
                e.matched_prefix_len = do_insert(req, in.op_tokens + off, ln);
                e.num_evicted_nodes = 0;
                e.evicted_tokens = 0;
            } else if (kind == RPCT_OP_RELEASE) {
                do_release(in.op_arg_a[i]);
                e.matched_prefix_len = -1;
                e.num_evicted_nodes = 0;
                e.evicted_tokens = 0;
            } else {
                int32_t en = 0, et = 0;
                do_evict(in.op_arg_a[i], &en, &et);
                e.matched_prefix_len = -1;
                e.num_evicted_nodes = en;
                e.evicted_tokens = et;
            }
            e.num_nodes = num_nodes();
            e.num_tokens_cached = num_tokens_cached();
            e.tree_checksum = tree_checksum();
            e.state_checksum = state_checksum(e.tree_checksum);
            out->push_back(e);
        }
    }
};

static inline bool rpct_check_all_outputs(
    const std::vector<RpctOpExpected>& expected,
    const RpctHostOutputsView& got,
    int op_count,
    std::string* error) {
    for (int i = 0; i < op_count; ++i) {
        const RpctOpExpected& e = expected[(size_t)i];
#define RPCT_CMP_I(field)                                                     \
        if (got.field[i] != e.field) {                                       \
            if (error) {                                                     \
                std::ostringstream oss;                                      \
                oss << #field " mismatch at op " << i << ": got "            \
                    << got.field[i] << ", expected " << e.field;            \
                *error = oss.str();                                          \
            }                                                               \
            return false;                                                   \
        }
        RPCT_CMP_I(matched_prefix_len)
        RPCT_CMP_I(num_nodes)
        RPCT_CMP_I(num_tokens_cached)
        RPCT_CMP_I(num_evicted_nodes)
        RPCT_CMP_I(evicted_tokens)
#undef RPCT_CMP_I
        if (got.tree_checksum[i] != e.tree_checksum) {
            if (error) {
                std::ostringstream oss;
                oss << "tree_checksum mismatch at op " << i << ": got 0x"
                    << std::hex << got.tree_checksum[i] << ", expected 0x"
                    << e.tree_checksum;
                *error = oss.str();
            }
            return false;
        }
        if (got.state_checksum[i] != e.state_checksum) {
            if (error) {
                std::ostringstream oss;
                oss << "state_checksum mismatch at op " << i << ": got 0x"
                    << std::hex << got.state_checksum[i] << ", expected 0x"
                    << e.state_checksum;
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

#endif  // RADIX_PREFIX_CACHE_TREE_ORACLE_HPP_
