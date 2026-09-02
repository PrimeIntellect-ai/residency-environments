// ============================================================================
// file: segmented_sort_topm_oracle.hpp
// Independent CPU oracle + exact check helpers.
// ============================================================================

#ifndef SEGMENTED_SORT_TOPM_ORACLE_HPP_
#define SEGMENTED_SORT_TOPM_ORACLE_HPP_

#include "segmented_sort_topm_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

struct SstHostInputsView {
    const int32_t* seg_offsets;
    const int32_t* item_key;
    const int32_t* item_value;
};

struct SstHostOutputsView {
    const int32_t* topm_count;
    const int32_t* topm_offsets;
    const int32_t* packed_topm_key;
    const int32_t* packed_topm_value;
    const int32_t* packed_topm_origidx;
    const int64_t* seg_sum;
    const int32_t* seg_max;
    const int32_t* seg_argmax;
};

struct SstItem {
    int32_t key;
    int32_t value;
    int32_t origidx;
};

struct SstExpected {
    std::vector<int32_t> topm_count;
    std::vector<int32_t> topm_offsets;
    std::vector<int32_t> packed_topm_key;
    std::vector<int32_t> packed_topm_value;
    std::vector<int32_t> packed_topm_origidx;
    std::vector<int64_t> seg_sum;
    std::vector<int32_t> seg_max;
    std::vector<int32_t> seg_argmax;
};

static inline bool sst_item_less(const SstItem& a, const SstItem& b) {
    if (a.key != b.key) return a.key > b.key;
    return a.origidx < b.origidx;
}

static inline int64_t sst_u64_to_i64(uint64_t x) {
    return static_cast<int64_t>(x);
}

static inline void sst_cpu_oracle(
    const SstRunSpec& run,
    const SstHostInputsView& in,
    SstExpected* expected) {
    const int S = run.S;
    const int M = run.M;

    expected->topm_count.assign((size_t)S, 0);
    expected->topm_offsets.assign((size_t)S + 1, 0);
    expected->packed_topm_key.clear();
    expected->packed_topm_value.clear();
    expected->packed_topm_origidx.clear();
    expected->seg_sum.assign((size_t)S, 0);
    expected->seg_max.assign((size_t)S, INT_MIN);
    expected->seg_argmax.assign((size_t)S, -1);

    int total = 0;
    expected->topm_offsets[0] = 0;

    for (int s = 0; s < S; ++s) {
        const int begin = in.seg_offsets[s];
        const int end = in.seg_offsets[s + 1];
        int len = end - begin;
        if (len < 0) len = 0;

        const int count = std::min(M, len);
        expected->topm_count[(size_t)s] = count;

        total += count;
        expected->topm_offsets[(size_t)s + 1] = total;
    }

    expected->packed_topm_key.assign((size_t)total, 0);
    expected->packed_topm_value.assign((size_t)total, 0);
    expected->packed_topm_origidx.assign((size_t)total, 0);

    std::vector<SstItem> items;

    for (int s = 0; s < S; ++s) {
        const int begin = in.seg_offsets[s];
        const int end = in.seg_offsets[s + 1];
        int len = end - begin;
        if (len < 0) len = 0;

        const int count = expected->topm_count[(size_t)s];
        const int out_base = expected->topm_offsets[(size_t)s];

        if (count == 0) {
            expected->seg_sum[(size_t)s] = 0;
            expected->seg_max[(size_t)s] = INT_MIN;
            expected->seg_argmax[(size_t)s] = -1;
            continue;
        }

        items.clear();
        items.reserve((size_t)len);

        for (int idx = begin; idx < end; ++idx) {
            items.push_back(SstItem{
                in.item_key[idx],
                in.item_value[idx],
                idx - begin
            });
        }

        std::stable_sort(items.begin(), items.end(), sst_item_less);

        uint64_t sum_bits = 0;
        int max_v = INT_MIN;
        int argmax_orig = -1;

        for (int i = 0; i < count; ++i) {
            const SstItem& item = items[(size_t)i];
            const int out = out_base + i;

            expected->packed_topm_key[(size_t)out] = item.key;
            expected->packed_topm_value[(size_t)out] = item.value;
            expected->packed_topm_origidx[(size_t)out] = item.origidx;

            sum_bits += static_cast<uint64_t>(static_cast<int64_t>(item.value));

            if (item.value > max_v ||
                (item.value == max_v && (argmax_orig < 0 || item.origidx < argmax_orig))) {
                max_v = item.value;
                argmax_orig = item.origidx;
            }
        }

        expected->seg_sum[(size_t)s] = sst_u64_to_i64(sum_bits);
        expected->seg_max[(size_t)s] = max_v;
        expected->seg_argmax[(size_t)s] = argmax_orig;
    }
}

static inline bool sst_check_counts_offsets(
    const SstRunSpec& run,
    const SstExpected& expected,
    const SstHostOutputsView& got,
    std::string* error) {
    const int S = run.S;

    for (int s = 0; s < S; ++s) {
        if (got.topm_count[s] != expected.topm_count[(size_t)s]) {
            if (error) {
                std::ostringstream oss;
                oss << "topm_count mismatch at s=" << s
                    << ": got " << got.topm_count[s]
                    << ", expected " << expected.topm_count[(size_t)s];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int s = 0; s <= S; ++s) {
        if (got.topm_offsets[s] != expected.topm_offsets[(size_t)s]) {
            if (error) {
                std::ostringstream oss;
                oss << "topm_offsets mismatch at s=" << s
                    << ": got " << got.topm_offsets[s]
                    << ", expected " << expected.topm_offsets[(size_t)s];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.topm_offsets[0] != 0) {
        if (error) *error = "topm_offsets[0] must be 0";
        return false;
    }

    for (int s = 0; s < S; ++s) {
        if (got.topm_offsets[s + 1] - got.topm_offsets[s] != got.topm_count[s]) {
            if (error) {
                std::ostringstream oss;
                oss << "topm offset/count invariant failed at s=" << s;
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool sst_check_packed(
    const SstRunSpec& run,
    const SstExpected& expected,
    const SstHostOutputsView& got,
    std::string* error) {
    const int total = expected.topm_offsets[(size_t)run.S];

    for (int i = 0; i < total; ++i) {
        if (got.packed_topm_key[i] != expected.packed_topm_key[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_topm_key mismatch at i=" << i
                    << ": got " << got.packed_topm_key[i]
                    << ", expected " << expected.packed_topm_key[(size_t)i];
                *error = oss.str();
            }
            return false;
        }

        if (got.packed_topm_value[i] != expected.packed_topm_value[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_topm_value mismatch at i=" << i
                    << ": got " << got.packed_topm_value[i]
                    << ", expected " << expected.packed_topm_value[(size_t)i];
                *error = oss.str();
            }
            return false;
        }

        if (got.packed_topm_origidx[i] != expected.packed_topm_origidx[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_topm_origidx mismatch at i=" << i
                    << ": got " << got.packed_topm_origidx[i]
                    << ", expected " << expected.packed_topm_origidx[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool sst_check_segment_reductions(
    const SstRunSpec& run,
    const SstExpected& expected,
    const SstHostOutputsView& got,
    std::string* error) {
    const int S = run.S;

    for (int s = 0; s < S; ++s) {
        if (got.seg_sum[s] != expected.seg_sum[(size_t)s]) {
            if (error) {
                std::ostringstream oss;
                oss << "seg_sum mismatch at s=" << s
                    << ": got " << got.seg_sum[s]
                    << ", expected " << expected.seg_sum[(size_t)s];
                *error = oss.str();
            }
            return false;
        }

        if (got.seg_max[s] != expected.seg_max[(size_t)s]) {
            if (error) {
                std::ostringstream oss;
                oss << "seg_max mismatch at s=" << s
                    << ": got " << got.seg_max[s]
                    << ", expected " << expected.seg_max[(size_t)s];
                *error = oss.str();
            }
            return false;
        }

        if (got.seg_argmax[s] != expected.seg_argmax[(size_t)s]) {
            if (error) {
                std::ostringstream oss;
                oss << "seg_argmax mismatch at s=" << s
                    << ": got " << got.seg_argmax[s]
                    << ", expected " << expected.seg_argmax[(size_t)s];
                *error = oss.str();
            }
            return false;
        }

        if (expected.topm_count[(size_t)s] == 0) {
            if (got.seg_sum[s] != 0 || got.seg_max[s] != INT_MIN || got.seg_argmax[s] != -1) {
                if (error) {
                    std::ostringstream oss;
                    oss << "empty segment rule failed at s=" << s;
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

static inline bool sst_check_all_outputs(
    const SstRunSpec& run,
    const SstExpected& expected,
    const SstHostOutputsView& got,
    std::string* error) {
    if (!sst_check_counts_offsets(run, expected, got, error)) return false;
    if (!sst_check_packed(run, expected, got, error)) return false;
    if (!sst_check_segment_reductions(run, expected, got, error)) return false;
    return true;
}

/*
GRADER CHECKS

Exact:
  - topm_count[S]
  - topm_offsets[S+1]
  - packed_topm_key/value/origidx[0:topm_offsets[S]]
  - seg_sum[S]
  - seg_max[S]
  - seg_argmax[S]

Additional harness checks:
  - guard sentinels around every output allocation
  - input immutability
  - held-out segment-size distributions:
      uniform,
      power-law,
      all-size-1,
      one-giant-rest-tiny,
      many ties
  - M > segment length and empty segment edges
*/

#endif  // SEGMENTED_SORT_TOPM_ORACLE_HPP_
