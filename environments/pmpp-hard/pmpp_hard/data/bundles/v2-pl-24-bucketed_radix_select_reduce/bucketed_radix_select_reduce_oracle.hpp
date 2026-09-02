// file: bucketed_radix_select_reduce_oracle.hpp

#ifndef BUCKETED_RADIX_SELECT_REDUCE_ORACLE_HPP_
#define BUCKETED_RADIX_SELECT_REDUCE_ORACLE_HPP_

#include "bucketed_radix_select_reduce_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

struct BrsrHostInputsView {
    const uint32_t* key;
    const int32_t* value;
};

struct BrsrHostOutputsView {
    const uint32_t* threshold_key;
    const int32_t* count;
    const int32_t* topT_indices;
    const int64_t* topT_sum;
    const int32_t* topT_max;
    const int32_t* topT_argmax;
    const int32_t* pass_histograms;
    const int32_t* chosen_bucket;
    const int32_t* carried_rank;
    const uint32_t* prefix_after_pass;
};

struct BrsrExpected {
    uint32_t threshold_key = 0;
    int32_t count = 0;
    std::vector<int32_t> topT_indices;
    int64_t topT_sum = 0;
    int32_t topT_max = INT_MIN;
    int32_t topT_argmax = -1;
    std::vector<int32_t> pass_histograms;
    std::vector<int32_t> chosen_bucket;
    std::vector<int32_t> carried_rank;
    std::vector<uint32_t> prefix_after_pass;
};

struct BrsrItem {
    uint32_t key;
    int32_t index;
};

static inline bool brsr_item_less(const BrsrItem& a, const BrsrItem& b) {
    if (a.key != b.key) return a.key > b.key;
    return a.index < b.index;
}

static inline int brsr_digit_for_pass_host(uint32_t key, int pass) {
    const int shift = 24 - 8 * pass;
    return static_cast<int>((key >> shift) & 0xffu);
}

static inline bool brsr_active_for_pass_host(uint32_t key, int pass, uint32_t prefix) {
    if (pass == 0) return true;
    return (key >> (32 - 8 * pass)) == prefix;
}

static inline void brsr_cpu_oracle(
    const BrsrRunSpec& run,
    const BrsrHostInputsView& in,
    BrsrExpected* expected) {
    const int N = run.N;
    const int count = brsr_count_host(run.N, run.T);

    expected->count = count;
    expected->pass_histograms.assign(BRSR_NUM_PASSES * BRSR_BUCKETS, 0);
    expected->chosen_bucket.assign(BRSR_NUM_PASSES, 0);
    expected->carried_rank.assign(BRSR_NUM_PASSES, 0);
    expected->prefix_after_pass.assign(BRSR_NUM_PASSES, 0);

    for (int pass = 0; pass < BRSR_NUM_PASSES; ++pass) {
        const uint32_t prefix = pass == 0 ? 0u : expected->prefix_after_pass[(size_t)pass - 1];

        for (int i = 0; i < N; ++i) {
            const uint32_t k = in.key[i];

            if (brsr_active_for_pass_host(k, pass, prefix)) {
                const int digit = brsr_digit_for_pass_host(k, pass);
                ++expected->pass_histograms[(size_t)pass * BRSR_BUCKETS + (size_t)digit];
            }
        }

        int rank = pass == 0 ? count : expected->carried_rank[(size_t)pass - 1];
        int chosen = 0;

        for (int b = BRSR_BUCKETS - 1; b >= 0; --b) {
            const int h = expected->pass_histograms[(size_t)pass * BRSR_BUCKETS + (size_t)b];

            if (rank > h) {
                rank -= h;
            } else {
                chosen = b;
                break;
            }
        }

        expected->chosen_bucket[(size_t)pass] = chosen;
        expected->carried_rank[(size_t)pass] = rank;
        expected->prefix_after_pass[(size_t)pass] =
            (prefix << 8) | static_cast<uint32_t>(chosen);
    }

    expected->threshold_key = expected->prefix_after_pass[(size_t)BRSR_NUM_PASSES - 1];

    std::vector<BrsrItem> items;
    items.reserve((size_t)N);

    for (int i = 0; i < N; ++i) {
        items.push_back(BrsrItem{in.key[i], i});
    }

    std::sort(items.begin(), items.end(), brsr_item_less);

    expected->topT_indices.assign((size_t)count, 0);

    for (int i = 0; i < count; ++i) {
        expected->topT_indices[(size_t)i] = items[(size_t)i].index;
    }

    uint64_t sum_bits = 0;
    int32_t max_v = INT_MIN;
    int32_t argmax = -1;

    for (int i = 0; i < count; ++i) {
        const int idx = expected->topT_indices[(size_t)i];
        const int32_t v = in.value[idx];

        sum_bits += static_cast<uint64_t>(static_cast<int64_t>(v));

        if (v > max_v || (v == max_v && (argmax < 0 || idx < argmax))) {
            max_v = v;
            argmax = idx;
        }
    }

    expected->topT_sum = static_cast<int64_t>(sum_bits);
    expected->topT_max = max_v;
    expected->topT_argmax = argmax;
}

static inline bool brsr_check_scalar_outputs(
    const BrsrExpected& expected,
    const BrsrHostOutputsView& got,
    std::string* error) {
    if (got.threshold_key[0] != expected.threshold_key) {
        if (error) {
            std::ostringstream oss;
            oss << "threshold_key mismatch: got " << got.threshold_key[0]
                << ", expected " << expected.threshold_key;
            *error = oss.str();
        }
        return false;
    }

    if (got.count[0] != expected.count) {
        if (error) {
            std::ostringstream oss;
            oss << "count mismatch: got " << got.count[0]
                << ", expected " << expected.count;
            *error = oss.str();
        }
        return false;
    }

    if (got.topT_sum[0] != expected.topT_sum) {
        if (error) {
            std::ostringstream oss;
            oss << "topT_sum mismatch: got " << got.topT_sum[0]
                << ", expected " << expected.topT_sum;
            *error = oss.str();
        }
        return false;
    }

    if (got.topT_max[0] != expected.topT_max) {
        if (error) {
            std::ostringstream oss;
            oss << "topT_max mismatch: got " << got.topT_max[0]
                << ", expected " << expected.topT_max;
            *error = oss.str();
        }
        return false;
    }

    if (got.topT_argmax[0] != expected.topT_argmax) {
        if (error) {
            std::ostringstream oss;
            oss << "topT_argmax mismatch: got " << got.topT_argmax[0]
                << ", expected " << expected.topT_argmax;
            *error = oss.str();
        }
        return false;
    }

    return true;
}

static inline bool brsr_check_topT_indices(
    const BrsrExpected& expected,
    const BrsrHostOutputsView& got,
    std::string* error) {
    for (int i = 0; i < expected.count; ++i) {
        if (got.topT_indices[i] != expected.topT_indices[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "topT_indices mismatch at i=" << i
                    << ": got " << got.topT_indices[i]
                    << ", expected " << expected.topT_indices[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool brsr_check_pass_metadata(
    const BrsrExpected& expected,
    const BrsrHostOutputsView& got,
    std::string* error) {
    for (int pass = 0; pass < BRSR_NUM_PASSES; ++pass) {
        if (got.chosen_bucket[pass] != expected.chosen_bucket[(size_t)pass]) {
            if (error) {
                std::ostringstream oss;
                oss << "chosen_bucket mismatch pass=" << pass
                    << ": got " << got.chosen_bucket[pass]
                    << ", expected " << expected.chosen_bucket[(size_t)pass];
                *error = oss.str();
            }
            return false;
        }

        if (got.carried_rank[pass] != expected.carried_rank[(size_t)pass]) {
            if (error) {
                std::ostringstream oss;
                oss << "carried_rank mismatch pass=" << pass
                    << ": got " << got.carried_rank[pass]
                    << ", expected " << expected.carried_rank[(size_t)pass];
                *error = oss.str();
            }
            return false;
        }

        if (got.prefix_after_pass[pass] != expected.prefix_after_pass[(size_t)pass]) {
            if (error) {
                std::ostringstream oss;
                oss << "prefix_after_pass mismatch pass=" << pass
                    << ": got " << got.prefix_after_pass[pass]
                    << ", expected " << expected.prefix_after_pass[(size_t)pass];
                *error = oss.str();
            }
            return false;
        }

        for (int b = 0; b < BRSR_BUCKETS; ++b) {
            const size_t idx = (size_t)pass * BRSR_BUCKETS + (size_t)b;

            if (got.pass_histograms[idx] != expected.pass_histograms[idx]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "pass_histograms mismatch pass=" << pass
                        << ", bucket=" << b
                        << ": got " << got.pass_histograms[idx]
                        << ", expected " << expected.pass_histograms[idx];
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

static inline bool brsr_check_all_outputs(
    const BrsrExpected& expected,
    const BrsrHostOutputsView& got,
    std::string* error) {
    if (!brsr_check_scalar_outputs(expected, got, error)) return false;
    if (!brsr_check_topT_indices(expected, got, error)) return false;
    if (!brsr_check_pass_metadata(expected, got, error)) return false;
    return true;
}

/*
GRADER CHECKS

Exact:
  - threshold_key[0]
  - count[0]
  - topT_indices[0:count]
  - topT_sum[0]
  - topT_max[0]
  - topT_argmax[0]
  - pass_histograms[4,256]
  - chosen_bucket[4]
  - carried_rank[4]
  - prefix_after_pass[4]

Additional harness checks:
  - guard sentinels
  - input immutability
  - held-out seeds
  - uniform, clustered, hot high-bucket, and many-tie key distributions
  - T in {16,256,4096}
  - N up to 1<<20
*/

#endif  // BUCKETED_RADIX_SELECT_REDUCE_ORACLE_HPP_
