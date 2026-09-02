// file: streaming_merge_topk_quantile_oracle.hpp

#ifndef STREAMING_MERGE_TOPK_QUANTILE_ORACLE_HPP_
#define STREAMING_MERGE_TOPK_QUANTILE_ORACLE_HPP_

#include "streaming_merge_topk_quantile_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

struct SmtqHostInputsView {
    const int32_t* group;
    const int32_t* key;
    const int32_t* value;
};

struct SmtqHostOutputsView {
    const int32_t* topk_keys;
    const int32_t* topk_values;
    const int32_t* topk_count;
    const int64_t* topk_value_sum;
    const uint64_t* histogram_checksum;
    const int32_t* quantile_key;
    const int64_t* total_ingested;
    const uint64_t* state_checksum;
};

struct SmtqItem {
    int32_t key = INT_MIN;
    int32_t value = 0;
    int64_t order = INT64_MAX;
};

struct SmtqExpected {
    std::vector<int32_t> topk_keys;
    std::vector<int32_t> topk_values;
    std::vector<int32_t> topk_count;
    std::vector<int64_t> topk_value_sum;
    uint64_t histogram_checksum = 0;
    std::vector<int32_t> quantile_key;
    int64_t total_ingested = 0;
    uint64_t state_checksum = 0;
};

static inline uint64_t smtq_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void smtq_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = smtq_oracle_fnv_byte(v, p[i]);
    }
    *h = v;
}

static inline int64_t smtq_oracle_add_i64_i32(int64_t a, int32_t b) {
    uint64_t ua = static_cast<uint64_t>(a);
    ua += static_cast<uint64_t>(static_cast<int64_t>(b));
    return static_cast<int64_t>(ua);
}

static inline bool smtq_item_less(const SmtqItem& a, const SmtqItem& b) {
    if (a.key != b.key) return a.key > b.key;
    return a.order < b.order;
}

struct SmtqOracleState {
    SmtqProblemSpec spec{};

    std::vector<SmtqItem> topk;
    std::vector<int32_t> topk_count;
    std::vector<int32_t> hist;
    std::vector<int32_t> group_total;
    int64_t total_ingested = 0;

    void init(const SmtqProblemSpec& s) {
        spec = s;
        topk.assign((size_t)spec.G * (size_t)spec.K, SmtqItem{});
        topk_count.assign((size_t)spec.G, 0);
        hist.assign((size_t)spec.G * (size_t)spec.num_bins, 0);
        group_total.assign((size_t)spec.G, 0);
        total_ingested = 0;
    }

    void reset() {
        std::fill(topk.begin(), topk.end(), SmtqItem{});
        std::fill(topk_count.begin(), topk_count.end(), 0);
        std::fill(hist.begin(), hist.end(), 0);
        std::fill(group_total.begin(), group_total.end(), 0);
        total_ingested = 0;
    }

    int bin_width() const {
        const int64_t range =
            static_cast<int64_t>(spec.key_max) - static_cast<int64_t>(spec.key_min) + 1;
        return static_cast<int>((range + spec.num_bins - 1) / spec.num_bins);
    }

    int bin_for_key(int32_t key) const {
        if (key <= spec.key_min) return 0;
        if (key >= spec.key_max) return spec.num_bins - 1;

        int bin = static_cast<int>(
            (static_cast<int64_t>(key) - spec.key_min) / bin_width());

        if (bin < 0) bin = 0;
        if (bin >= spec.num_bins) bin = spec.num_bins - 1;
        return bin;
    }

    int32_t bin_lower_bound(int bin) const {
        int64_t lo = static_cast<int64_t>(spec.key_min) + static_cast<int64_t>(bin) * bin_width();

        if (lo < spec.key_min) lo = spec.key_min;
        if (lo > spec.key_max) lo = spec.key_max;

        return static_cast<int32_t>(lo);
    }

    void insert_topk(int group, int32_t key, int32_t value, int64_t order) {
        const int base = group * spec.K;
        int count = topk_count[(size_t)group];

        if (count < spec.K) {
            topk[(size_t)base + (size_t)count] = SmtqItem{key, value, order};
            topk_count[(size_t)group] = count + 1;
        } else {
            SmtqItem& worst = topk[(size_t)base + (size_t)spec.K - 1];

            if (smtq_item_less(SmtqItem{key, value, order}, worst)) {
                worst = SmtqItem{key, value, order};
            }
        }

        std::sort(
            topk.begin() + base,
            topk.begin() + base + spec.K,
            smtq_item_less);
    }

    int64_t topk_sum(int group) const {
        const int base = group * spec.K;
        const int count = topk_count[(size_t)group];

        int64_t sum = 0;

        for (int i = 0; i < count; ++i) {
            sum = smtq_oracle_add_i64_i32(sum, topk[(size_t)base + (size_t)i].value);
        }

        return sum;
    }

    uint64_t histogram_checksum() const {
        uint64_t h = 1469598103934665603ULL;

        smtq_oracle_fnv_bytes(&h, &spec.G, sizeof(int32_t));
        smtq_oracle_fnv_bytes(&h, &spec.num_bins, sizeof(int32_t));

        for (int g = 0; g < spec.G; ++g) {
            for (int b = 0; b < spec.num_bins; ++b) {
                const int32_t v = hist[(size_t)g * (size_t)spec.num_bins + (size_t)b];
                smtq_oracle_fnv_bytes(&h, &v, sizeof(int32_t));
            }
        }

        return h;
    }

    uint64_t state_checksum() const {
        uint64_t h = 1469598103934665603ULL;

        smtq_oracle_fnv_bytes(&h, &spec.G, sizeof(int32_t));
        smtq_oracle_fnv_bytes(&h, &spec.K, sizeof(int32_t));
        smtq_oracle_fnv_bytes(&h, &spec.num_bins, sizeof(int32_t));
        smtq_oracle_fnv_bytes(&h, &spec.key_min, sizeof(int32_t));
        smtq_oracle_fnv_bytes(&h, &spec.key_max, sizeof(int32_t));
        smtq_oracle_fnv_bytes(&h, &total_ingested, sizeof(int64_t));

        for (int g = 0; g < spec.G; ++g) {
            const int32_t gt = group_total[(size_t)g];
            const int32_t tc = topk_count[(size_t)g];

            smtq_oracle_fnv_bytes(&h, &gt, sizeof(int32_t));
            smtq_oracle_fnv_bytes(&h, &tc, sizeof(int32_t));

            for (int i = 0; i < spec.K; ++i) {
                const SmtqItem& item = topk[(size_t)g * (size_t)spec.K + (size_t)i];
                smtq_oracle_fnv_bytes(&h, &item.key, sizeof(int32_t));
                smtq_oracle_fnv_bytes(&h, &item.value, sizeof(int32_t));
                smtq_oracle_fnv_bytes(&h, &item.order, sizeof(int64_t));
            }

            for (int b = 0; b < spec.num_bins; ++b) {
                const int32_t hv = hist[(size_t)g * (size_t)spec.num_bins + (size_t)b];
                smtq_oracle_fnv_bytes(&h, &hv, sizeof(int32_t));
            }
        }

        return h;
    }

    int32_t quantile_for_group(int group, int q_num, int q_den) const {
        const int total = group_total[(size_t)group];
        if (total <= 0) return INT_MIN;

        if (q_num < 0) q_num = 0;
        if (q_num > q_den) q_num = q_den;

        int64_t rank = (static_cast<int64_t>(q_num) * total + q_den - 1) / q_den;
        if (rank < 1) rank = 1;
        if (rank > total) rank = total;

        int64_t accum = 0;
        for (int b = 0; b < spec.num_bins; ++b) {
            accum += hist[(size_t)group * (size_t)spec.num_bins + (size_t)b];
            if (accum >= rank) return bin_lower_bound(b);
        }

        return spec.key_max;
    }

    void step_once(
        const SmtqRunSpec& run,
        const SmtqHostInputsView& in,
        SmtqExpected* expected) {
        for (int i = 0; i < run.batch_size; ++i) {
            const int g = in.group[i];

            if (g < 0 || g >= spec.G) {
                continue;
            }

            const int32_t key = in.key[i];
            const int32_t value = in.value[i];

            const int bin = bin_for_key(key);

            hist[(size_t)g * (size_t)spec.num_bins + (size_t)bin] += 1;
            group_total[(size_t)g] += 1;

            insert_topk(g, key, value, total_ingested);
            total_ingested += 1;
        }

        expected->topk_keys.assign((size_t)spec.G * (size_t)spec.K, INT_MIN);
        expected->topk_values.assign((size_t)spec.G * (size_t)spec.K, 0);
        expected->topk_count.assign((size_t)spec.G, 0);
        expected->topk_value_sum.assign((size_t)spec.G, 0);
        expected->quantile_key.assign((size_t)spec.G, INT_MIN);

        for (int g = 0; g < spec.G; ++g) {
            expected->topk_count[(size_t)g] = topk_count[(size_t)g];
            expected->topk_value_sum[(size_t)g] = topk_sum(g);

            for (int i = 0; i < spec.K; ++i) {
                const size_t idx = (size_t)g * (size_t)spec.K + (size_t)i;
                expected->topk_keys[idx] = topk[idx].key;
                expected->topk_values[idx] = topk[idx].value;
            }

            expected->quantile_key[(size_t)g] =
                run.is_query ? quantile_for_group(g, run.q_num, run.q_den) : INT_MIN;
        }

        expected->histogram_checksum = histogram_checksum();
        expected->total_ingested = total_ingested;
        expected->state_checksum = state_checksum();
    }
};

static inline bool smtq_check_all_outputs(
    const SmtqProblemSpec& spec,
    const SmtqExpected& expected,
    const SmtqHostOutputsView& got,
    std::string* error) {
    const int total_topk = spec.G * spec.K;

    for (int i = 0; i < total_topk; ++i) {
        if (got.topk_keys[i] != expected.topk_keys[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "topk_keys mismatch i=" << i
                    << ": got " << got.topk_keys[i]
                    << ", expected " << expected.topk_keys[(size_t)i];
                *error = oss.str();
            }
            return false;
        }

        if (got.topk_values[i] != expected.topk_values[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "topk_values mismatch i=" << i
                    << ": got " << got.topk_values[i]
                    << ", expected " << expected.topk_values[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int g = 0; g < spec.G; ++g) {
        if (got.topk_count[g] != expected.topk_count[(size_t)g]) {
            if (error) {
                std::ostringstream oss;
                oss << "topk_count mismatch g=" << g
                    << ": got " << got.topk_count[g]
                    << ", expected " << expected.topk_count[(size_t)g];
                *error = oss.str();
            }
            return false;
        }

        if (got.topk_value_sum[g] != expected.topk_value_sum[(size_t)g]) {
            if (error) {
                std::ostringstream oss;
                oss << "topk_value_sum mismatch g=" << g
                    << ": got " << got.topk_value_sum[g]
                    << ", expected " << expected.topk_value_sum[(size_t)g];
                *error = oss.str();
            }
            return false;
        }

        if (got.quantile_key[g] != expected.quantile_key[(size_t)g]) {
            if (error) {
                std::ostringstream oss;
                oss << "quantile_key mismatch g=" << g
                    << ": got " << got.quantile_key[g]
                    << ", expected " << expected.quantile_key[(size_t)g];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.histogram_checksum[0] != expected.histogram_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "histogram_checksum mismatch: got 0x"
                << std::hex << got.histogram_checksum[0]
                << ", expected 0x" << expected.histogram_checksum;
            *error = oss.str();
        }
        return false;
    }

    if (got.total_ingested[0] != expected.total_ingested) {
        if (error) {
            std::ostringstream oss;
            oss << "total_ingested mismatch: got " << got.total_ingested[0]
                << ", expected " << expected.total_ingested;
            *error = oss.str();
        }
        return false;
    }

    if (got.state_checksum[0] != expected.state_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "state_checksum mismatch: got 0x"
                << std::hex << got.state_checksum[0]
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
    smtq_check_all_outputs(...)

Required harness coverage:
  - K in {4,16,64}
  - G in {8,64,1024}
  - uniform and zipf-hot group distributions
  - many key ties
  - interleaved query and ingest steps
  - empty batch query
  - invalid groups
  - reset and replay
*/

#endif  // STREAMING_MERGE_TOPK_QUANTILE_ORACLE_HPP_
