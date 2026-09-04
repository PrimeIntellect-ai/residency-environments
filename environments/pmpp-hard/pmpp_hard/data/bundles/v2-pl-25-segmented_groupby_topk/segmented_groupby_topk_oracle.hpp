// file: segmented_groupby_topk_oracle.hpp

#ifndef SEGMENTED_GROUPBY_TOPK_ORACLE_HPP_
#define SEGMENTED_GROUPBY_TOPK_ORACLE_HPP_

#include "segmented_groupby_topk_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

struct SgtHostInputsView {
    const int32_t* group_id;
    const int32_t* key;
    const int32_t* value;
};

struct SgtHostOutputsView {
    const int32_t* group_counts;
    const int32_t* group_offsets;
    const int32_t* packed_topk_origidx;
    const int64_t* per_group_sum;
    const int32_t* per_group_max;
    const int32_t* per_group_argmax;
    const int32_t* kept_count;
};

struct SgtItem {
    int32_t key;
    int32_t value;
    int32_t orig;
};

struct SgtExpected {
    std::vector<int32_t> group_counts;
    std::vector<int32_t> group_offsets;
    std::vector<int32_t> packed_topk_origidx;
    std::vector<int64_t> per_group_sum;
    std::vector<int32_t> per_group_max;
    std::vector<int32_t> per_group_argmax;
    std::vector<int32_t> kept_count;
};

static inline bool sgtk_item_less(const SgtItem& a, const SgtItem& b) {
    if (a.key != b.key) return a.key > b.key;
    return a.orig < b.orig;
}

static inline void sgtk_cpu_oracle(
    const SgtRunSpec& run,
    const SgtHostInputsView& in,
    SgtExpected* expected) {
    const int N = run.N;
    const int G = run.G;
    const int M = run.M;

    expected->group_counts.assign((size_t)G, 0);
    expected->group_offsets.assign((size_t)G + 1, 0);
    expected->per_group_sum.assign((size_t)G, 0);
    expected->per_group_max.assign((size_t)G, INT_MIN);
    expected->per_group_argmax.assign((size_t)G, -1);
    expected->kept_count.assign((size_t)G, 0);

    std::vector<std::vector<SgtItem>> groups((size_t)G);

    for (int i = 0; i < N; ++i) {
        const int g = in.group_id[i];
        const int v = in.value[i];

        if (v > 0 && g >= 0 && g < G) {
            groups[(size_t)g].push_back(SgtItem{in.key[i], v, i});
            ++expected->group_counts[(size_t)g];
        }
    }

    int total_kept = 0;
    expected->group_offsets[0] = 0;

    for (int g = 0; g < G; ++g) {
        const int cnt = expected->group_counts[(size_t)g];
        const int kept = cnt < M ? cnt : M;

        expected->kept_count[(size_t)g] = kept;
        total_kept += kept;
        expected->group_offsets[(size_t)g + 1] = total_kept;
    }

    expected->packed_topk_origidx.assign((size_t)total_kept, 0);

    for (int g = 0; g < G; ++g) {
        std::vector<SgtItem>& rows = groups[(size_t)g];
        std::sort(rows.begin(), rows.end(), sgtk_item_less);

        const int kept = expected->kept_count[(size_t)g];
        const int out_base = expected->group_offsets[(size_t)g];

        if (kept == 0) {
            expected->per_group_sum[(size_t)g] = 0;
            expected->per_group_max[(size_t)g] = INT_MIN;
            expected->per_group_argmax[(size_t)g] = -1;
            continue;
        }

        uint64_t sum_bits = 0;
        int32_t max_v = INT_MIN;
        int32_t argmax = -1;

        for (int i = 0; i < kept; ++i) {
            const SgtItem& item = rows[(size_t)i];
            expected->packed_topk_origidx[(size_t)out_base + (size_t)i] = item.orig;

            sum_bits += static_cast<uint64_t>(static_cast<int64_t>(item.value));

            if (item.value > max_v ||
                (item.value == max_v && (argmax < 0 || item.orig < argmax))) {
                max_v = item.value;
                argmax = item.orig;
            }
        }

        expected->per_group_sum[(size_t)g] = static_cast<int64_t>(sum_bits);
        expected->per_group_max[(size_t)g] = max_v;
        expected->per_group_argmax[(size_t)g] = argmax;
    }
}

static inline bool sgtk_check_counts_offsets(
    const SgtRunSpec& run,
    const SgtExpected& expected,
    const SgtHostOutputsView& got,
    std::string* error) {
    const int G = run.G;

    for (int g = 0; g < G; ++g) {
        if (got.group_counts[g] != expected.group_counts[(size_t)g]) {
            if (error) {
                std::ostringstream oss;
                oss << "group_counts mismatch g=" << g
                    << ": got " << got.group_counts[g]
                    << ", expected " << expected.group_counts[(size_t)g];
                *error = oss.str();
            }
            return false;
        }

        if (got.kept_count[g] != expected.kept_count[(size_t)g]) {
            if (error) {
                std::ostringstream oss;
                oss << "kept_count mismatch g=" << g
                    << ": got " << got.kept_count[g]
                    << ", expected " << expected.kept_count[(size_t)g];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int g = 0; g <= G; ++g) {
        if (got.group_offsets[g] != expected.group_offsets[(size_t)g]) {
            if (error) {
                std::ostringstream oss;
                oss << "group_offsets mismatch g=" << g
                    << ": got " << got.group_offsets[g]
                    << ", expected " << expected.group_offsets[(size_t)g];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.group_offsets[0] != 0) {
        if (error) *error = "group_offsets[0] must be 0";
        return false;
    }

    for (int g = 0; g < G; ++g) {
        if (got.group_offsets[g + 1] - got.group_offsets[g] != got.kept_count[g]) {
            if (error) {
                std::ostringstream oss;
                oss << "offset/kept invariant failed at g=" << g;
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool sgtk_check_packed(
    const SgtRunSpec& run,
    const SgtExpected& expected,
    const SgtHostOutputsView& got,
    std::string* error) {
    const int total = expected.group_offsets[(size_t)run.G];

    for (int i = 0; i < total; ++i) {
        if (got.packed_topk_origidx[i] != expected.packed_topk_origidx[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_topk_origidx mismatch i=" << i
                    << ": got " << got.packed_topk_origidx[i]
                    << ", expected " << expected.packed_topk_origidx[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool sgtk_check_summaries(
    const SgtRunSpec& run,
    const SgtExpected& expected,
    const SgtHostOutputsView& got,
    std::string* error) {
    const int G = run.G;

    for (int g = 0; g < G; ++g) {
        if (got.per_group_sum[g] != expected.per_group_sum[(size_t)g]) {
            if (error) {
                std::ostringstream oss;
                oss << "per_group_sum mismatch g=" << g
                    << ": got " << got.per_group_sum[g]
                    << ", expected " << expected.per_group_sum[(size_t)g];
                *error = oss.str();
            }
            return false;
        }

        if (got.per_group_max[g] != expected.per_group_max[(size_t)g]) {
            if (error) {
                std::ostringstream oss;
                oss << "per_group_max mismatch g=" << g
                    << ": got " << got.per_group_max[g]
                    << ", expected " << expected.per_group_max[(size_t)g];
                *error = oss.str();
            }
            return false;
        }

        if (got.per_group_argmax[g] != expected.per_group_argmax[(size_t)g]) {
            if (error) {
                std::ostringstream oss;
                oss << "per_group_argmax mismatch g=" << g
                    << ": got " << got.per_group_argmax[g]
                    << ", expected " << expected.per_group_argmax[(size_t)g];
                *error = oss.str();
            }
            return false;
        }

        if (expected.kept_count[(size_t)g] == 0) {
            if (got.per_group_sum[g] != 0 ||
                got.per_group_max[g] != INT_MIN ||
                got.per_group_argmax[g] != -1) {
                if (error) {
                    std::ostringstream oss;
                    oss << "empty-group summary rule failed at g=" << g;
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

static inline bool sgtk_check_all_outputs(
    const SgtRunSpec& run,
    const SgtExpected& expected,
    const SgtHostOutputsView& got,
    std::string* error) {
    if (!sgtk_check_counts_offsets(run, expected, got, error)) return false;
    if (!sgtk_check_packed(run, expected, got, error)) return false;
    if (!sgtk_check_summaries(run, expected, got, error)) return false;
    return true;
}

/*
GRADER CHECKS

Exact:
  - group_counts[G]
  - kept_count[G]
  - group_offsets[G+1]
  - packed_topk_origidx[0:group_offsets[G]]
  - per_group_sum[G]
  - per_group_max[G]
  - per_group_argmax[G]

Additional harness checks:
  - guard sentinels
  - input immutability
  - uniform, zipf-hot, single-hot, many key-ties
  - all-filtered groups
  - M in {4,16,64}
*/

#endif  // SEGMENTED_GROUPBY_TOPK_ORACLE_HPP_
