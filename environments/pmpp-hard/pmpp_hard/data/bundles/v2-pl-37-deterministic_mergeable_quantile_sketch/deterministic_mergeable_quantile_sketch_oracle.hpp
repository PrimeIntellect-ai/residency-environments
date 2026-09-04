// file: deterministic_mergeable_quantile_sketch_oracle.hpp

#ifndef DETERMINISTIC_MERGEABLE_QUANTILE_SKETCH_ORACLE_HPP_
#define DETERMINISTIC_MERGEABLE_QUANTILE_SKETCH_ORACLE_HPP_

#include "deterministic_mergeable_quantile_sketch_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

struct DmqsHostInputsView {
    const int32_t* keys;
    const int32_t* merge_level_size;
    const int32_t* merge_keys;
};

struct DmqsHostOutputsView {
    const int64_t* total_weight;
    const int32_t* num_levels;
    const int32_t* num_retained_items;
    const int32_t* query_result;
    const uint64_t* sketch_checksum;
    const uint64_t* state_checksum;
};

struct DmqsExpected {
    int64_t total_weight = 0;
    int32_t num_levels = 0;
    int32_t num_retained_items = 0;
    int32_t query_result = INT_MIN;
    uint64_t sketch_checksum = 0;
    uint64_t state_checksum = 0;
};

static inline uint64_t dmqs_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void dmqs_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = dmqs_oracle_fnv_byte(v, p[i]);
    *h = v;
}

// Item used for canonical sorting: (key asc, insertion_order asc).
struct DmqsSortItem {
    int32_t key;
    int32_t order;  // slot index within the level
};

static inline bool dmqs_sort_less(const DmqsSortItem& a, const DmqsSortItem& b) {
    if (a.key != b.key) return a.key < b.key;
    return a.order < b.order;
}

struct DmqsOracleState {
    DmqsProblemSpec spec{};

    int k = 0;
    int num_levels_cap = 0;

    // level_buf[L] holds the raw slot contents (size = level_size[L]).
    std::vector<std::vector<int32_t>> level_buf;
    std::vector<int64_t> compaction_counter;

    void init(const DmqsProblemSpec& s) {
        spec = s;
        k = s.k;
        num_levels_cap = s.num_levels;
        level_buf.assign((size_t)num_levels_cap, std::vector<int32_t>());
        compaction_counter.assign((size_t)num_levels_cap, 0);
    }

    void reset() {
        for (auto& b : level_buf) b.clear();
        std::fill(compaction_counter.begin(), compaction_counter.end(), 0);
    }

    // Canonical ascending sort of a level's contents by (key, slot index).
    std::vector<DmqsSortItem> sorted_level(int L) const {
        std::vector<DmqsSortItem> v;
        const std::vector<int32_t>& buf = level_buf[(size_t)L];
        v.reserve(buf.size());
        for (size_t j = 0; j < buf.size(); ++j) {
            v.push_back(DmqsSortItem{buf[j], (int32_t)j});
        }
        std::stable_sort(v.begin(), v.end(), dmqs_sort_less);
        return v;
    }

    // Compact level L: precondition level_buf[L].size() == k.
    // Produces k/2 survivors appended to level L+1.
    void compact_level(int L) {
        std::vector<DmqsSortItem> s = sorted_level(L);  // size == k, ascending
        const int parity = (int)(compaction_counter[(size_t)L] & 1);
        // parity==0: keep ODD indices (1,3,5,...). parity==1: keep EVEN (0,2,4,...).
        const int start = (parity == 0) ? 1 : 0;

        std::vector<int32_t> survivors;
        survivors.reserve((size_t)(k / 2));
        for (int i = start; i < k; i += 2) {
            survivors.push_back(s[(size_t)i].key);
        }

        compaction_counter[(size_t)L] += 1;
        level_buf[(size_t)L].clear();

        // Append survivors (already ascending) to level L+1.
        std::vector<int32_t>& up = level_buf[(size_t)(L + 1)];
        for (int32_t key : survivors) up.push_back(key);
    }

    // Cascade compaction: ascending pass, re-checking each level until under k.
    void cascade_compact() {
        for (int L = 0; L < num_levels_cap; ++L) {
            while ((int)level_buf[(size_t)L].size() >= k) {
                // For level 0 a batch may have pushed size beyond k. Compaction
                // consumes the FIRST k slots (FIFO); the remainder shifts down.
                if ((int)level_buf[(size_t)L].size() == k) {
                    compact_level(L);
                } else {
                    // size > k: split off the first k as a temporary level, compact
                    // those, and keep the remainder as the new level-L buffer.
                    std::vector<int32_t>& buf = level_buf[(size_t)L];
                    std::vector<int32_t> first_k(buf.begin(), buf.begin() + k);
                    std::vector<int32_t> rest(buf.begin() + k, buf.end());

                    // Build sorted view of the first k and select survivors.
                    std::vector<DmqsSortItem> sv;
                    sv.reserve((size_t)k);
                    for (int j = 0; j < k; ++j) sv.push_back(DmqsSortItem{first_k[(size_t)j], j});
                    std::stable_sort(sv.begin(), sv.end(), dmqs_sort_less);

                    const int parity = (int)(compaction_counter[(size_t)L] & 1);
                    const int start = (parity == 0) ? 1 : 0;
                    std::vector<int32_t> survivors;
                    survivors.reserve((size_t)(k / 2));
                    for (int i = start; i < k; i += 2) survivors.push_back(sv[(size_t)i].key);

                    compaction_counter[(size_t)L] += 1;

                    // Remainder becomes level L (insertion order re-indexed by slot).
                    level_buf[(size_t)L] = rest;

                    std::vector<int32_t>& up = level_buf[(size_t)(L + 1)];
                    for (int32_t key : survivors) up.push_back(key);
                }
            }
        }
    }

    void do_ingest(const int32_t* keys, int batch_size) {
        std::vector<int32_t>& l0 = level_buf[0];
        for (int i = 0; i < batch_size; ++i) l0.push_back(keys[i]);
        cascade_compact();
    }

    void do_merge(const DmqsRunSpec& run, const DmqsHostInputsView& in) {
        for (int L = 0; L < run.merge_num_levels; ++L) {
            const int sz = in.merge_level_size[L];
            for (int j = 0; j < sz; ++j) {
                level_buf[(size_t)L].push_back(in.merge_keys[(size_t)L * (size_t)k + (size_t)j]);
            }
        }
        cascade_compact();
    }

    int64_t compute_total_weight() const {
        int64_t tw = 0;
        for (int L = 0; L < num_levels_cap; ++L) {
            tw += (int64_t)level_buf[(size_t)L].size() * (int64_t)(1LL << L);
        }
        return tw;
    }

    int compute_num_retained() const {
        int n = 0;
        for (int L = 0; L < num_levels_cap; ++L) n += (int)level_buf[(size_t)L].size();
        return n;
    }

    int compute_num_levels() const {
        int highest = -1;
        for (int L = 0; L < num_levels_cap; ++L) {
            if (!level_buf[(size_t)L].empty() || compaction_counter[(size_t)L] > 0) highest = L;
        }
        return highest + 1;
    }

    int32_t query(int q_num, int q_den) const {
        const int64_t total = compute_total_weight();
        if (total <= 0) return INT_MIN;

        if (q_num < 0) q_num = 0;
        if (q_num > q_den) q_num = q_den;

        int64_t target = ((int64_t)q_num * total + q_den - 1) / q_den;
        if (target < 1) target = 1;
        if (target > total) target = total;

        // Build global list (key, weight, level, order) and sort.
        struct QItem { int32_t key; int32_t level; int32_t order; };
        std::vector<QItem> items;
        items.reserve((size_t)compute_num_retained());
        for (int L = 0; L < num_levels_cap; ++L) {
            const std::vector<int32_t>& buf = level_buf[(size_t)L];
            for (size_t j = 0; j < buf.size(); ++j) {
                items.push_back(QItem{buf[j], (int32_t)L, (int32_t)j});
            }
        }
        std::stable_sort(items.begin(), items.end(), [](const QItem& a, const QItem& b) {
            if (a.key != b.key) return a.key < b.key;
            if (a.level != b.level) return a.level < b.level;
            return a.order < b.order;
        });

        int64_t accum = 0;
        for (const QItem& it : items) {
            accum += (int64_t)(1LL << it.level);
            if (accum >= target) return it.key;
        }
        return items.back().key;  // unreachable: target <= total
    }

    uint64_t sketch_checksum() const {
        uint64_t h = 1469598103934665603ULL;
        for (int L = 0; L < num_levels_cap; ++L) {
            const int32_t lv = L;
            const int32_t sz = (int32_t)level_buf[(size_t)L].size();
            dmqs_oracle_fnv_bytes(&h, &lv, sizeof(int32_t));
            dmqs_oracle_fnv_bytes(&h, &sz, sizeof(int32_t));

            std::vector<DmqsSortItem> s = sorted_level(L);
            const int32_t weight = (int32_t)(1LL << L);
            for (const DmqsSortItem& it : s) {
                dmqs_oracle_fnv_bytes(&h, &it.key, sizeof(int32_t));
                dmqs_oracle_fnv_bytes(&h, &it.order, sizeof(int32_t));
                dmqs_oracle_fnv_bytes(&h, &weight, sizeof(int32_t));
            }
        }
        return h;
    }

    uint64_t state_checksum() const {
        uint64_t h = 1469598103934665603ULL;
        const int32_t kk = k;
        const int32_t nl = num_levels_cap;
        const int64_t tw = compute_total_weight();
        const int32_t nr = compute_num_retained();
        dmqs_oracle_fnv_bytes(&h, &kk, sizeof(int32_t));
        dmqs_oracle_fnv_bytes(&h, &nl, sizeof(int32_t));
        dmqs_oracle_fnv_bytes(&h, &tw, sizeof(int64_t));
        dmqs_oracle_fnv_bytes(&h, &nr, sizeof(int32_t));

        for (int L = 0; L < num_levels_cap; ++L) {
            const int32_t lv = L;
            const int32_t sz = (int32_t)level_buf[(size_t)L].size();
            const int64_t cc = compaction_counter[(size_t)L];
            dmqs_oracle_fnv_bytes(&h, &lv, sizeof(int32_t));
            dmqs_oracle_fnv_bytes(&h, &sz, sizeof(int32_t));
            dmqs_oracle_fnv_bytes(&h, &cc, sizeof(int64_t));
            const std::vector<int32_t>& buf = level_buf[(size_t)L];
            for (size_t j = 0; j < buf.size(); ++j) {
                const int32_t kv = buf[j];
                dmqs_oracle_fnv_bytes(&h, &kv, sizeof(int32_t));
            }
        }
        return h;
    }

    void step_once(
        const DmqsRunSpec& run,
        const DmqsHostInputsView& in,
        DmqsExpected* expected) {
        if (run.op == DMQS_OP_INGEST) {
            do_ingest(in.keys, run.batch_size);
        } else if (run.op == DMQS_OP_MERGE) {
            do_merge(run, in);
        }
        // QUERY does not mutate.

        expected->total_weight = compute_total_weight();
        expected->num_levels = compute_num_levels();
        expected->num_retained_items = compute_num_retained();
        expected->query_result =
            (run.op == DMQS_OP_QUERY) ? query(run.q_num, run.q_den) : INT_MIN;
        expected->sketch_checksum = sketch_checksum();
        expected->state_checksum = state_checksum();
    }
};

static inline bool dmqs_check_all_outputs(
    const DmqsProblemSpec& spec,
    const DmqsExpected& expected,
    const DmqsHostOutputsView& got,
    std::string* error) {
    (void)spec;

    if (got.total_weight[0] != expected.total_weight) {
        if (error) {
            std::ostringstream oss;
            oss << "total_weight mismatch: got " << got.total_weight[0]
                << ", expected " << expected.total_weight;
            *error = oss.str();
        }
        return false;
    }

    if (got.num_levels[0] != expected.num_levels) {
        if (error) {
            std::ostringstream oss;
            oss << "num_levels mismatch: got " << got.num_levels[0]
                << ", expected " << expected.num_levels;
            *error = oss.str();
        }
        return false;
    }

    if (got.num_retained_items[0] != expected.num_retained_items) {
        if (error) {
            std::ostringstream oss;
            oss << "num_retained_items mismatch: got " << got.num_retained_items[0]
                << ", expected " << expected.num_retained_items;
            *error = oss.str();
        }
        return false;
    }

    if (got.query_result[0] != expected.query_result) {
        if (error) {
            std::ostringstream oss;
            oss << "query_result mismatch: got " << got.query_result[0]
                << ", expected " << expected.query_result;
            *error = oss.str();
        }
        return false;
    }

    if (got.sketch_checksum[0] != expected.sketch_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "sketch_checksum mismatch: got 0x" << std::hex << got.sketch_checksum[0]
                << ", expected 0x" << expected.sketch_checksum;
            *error = oss.str();
        }
        return false;
    }

    if (got.state_checksum[0] != expected.state_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "state_checksum mismatch: got 0x" << std::hex << got.state_checksum[0]
                << ", expected 0x" << expected.state_checksum;
            *error = oss.str();
        }
        return false;
    }

    return true;
}

/*
GRADER MODEL

Use:
  oracle.init(problem_spec)
  solution_reset + oracle.reset()
  for each step:
    solution_run(...)
    oracle.step_once(...)
    dmqs_check_all_outputs(...)

Required harness coverage:
  - k in {2, 4, 8, 64}
  - overflow cascades (single ingest forces multi-level compaction)
  - merges that cascade-compact
  - exact-rank boundary queries (q=0, q=1, midpoints)
  - duplicate keys / heavy ties
  - empty-sketch queries
  - reset and exact replay
*/

#endif  // DETERMINISTIC_MERGEABLE_QUANTILE_SKETCH_ORACLE_HPP_
