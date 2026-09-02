// file: fused_csr_spmm_topk_oracle.hpp

#ifndef FUSED_CSR_SPMM_TOPK_ORACLE_HPP_
#define FUSED_CSR_SPMM_TOPK_ORACLE_HPP_

#include "fused_csr_spmm_topk_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <sstream>
#include <string>
#include <vector>

struct FcstHostInputsView {
    const int32_t* row_offsets;
    const int32_t* col_indices;
    const int32_t* vals;
    const int32_t* dense_b;
};

struct FcstHostOutputsView {
    const int32_t* topm_cols;
    const int64_t* topm_vals;
    const int32_t* topm_count;
    const int64_t* row_sum;
    const int64_t* row_max;
    const int32_t* row_argmax;
    const int32_t* row_nnz;
    const int32_t* row_nnz_prefix;
};

struct FcstItem {
    int32_t col = -1;
    int64_t value = INT64_MIN;
};

struct FcstExpected {
    std::vector<int32_t> topm_cols;
    std::vector<int64_t> topm_vals;
    std::vector<int32_t> topm_count;
    std::vector<int64_t> row_sum;
    std::vector<int64_t> row_max;
    std::vector<int32_t> row_argmax;
    std::vector<int32_t> row_nnz;
    std::vector<int32_t> row_nnz_prefix;
};

static inline bool fcst_item_less(const FcstItem& a, const FcstItem& b) {
    if (a.value != b.value) return a.value > b.value;
    return a.col < b.col;
}

static inline int64_t fcst_add_i64_wrap_host(int64_t a, int64_t b) {
    uint64_t ua = static_cast<uint64_t>(a);
    ua += static_cast<uint64_t>(b);
    return static_cast<int64_t>(ua);
}

static inline int64_t fcst_spmm_value_host(
    int row_begin,
    int row_end,
    int K,
    int N,
    int col_n,
    const int32_t* col_indices,
    const int32_t* vals,
    const int32_t* dense_b) {
    uint64_t acc = 0;

    for (int e = row_begin; e < row_end; ++e) {
        const int k = col_indices[e];

        if (k >= 0 && k < K) {
            const int64_t product =
                static_cast<int64_t>(vals[e]) *
                static_cast<int64_t>(dense_b[(size_t)k * (size_t)N + (size_t)col_n]);

            acc += static_cast<uint64_t>(product);
        }
    }

    return static_cast<int64_t>(acc);
}

static inline void fcst_cpu_oracle(
    const FcstRunSpec& run,
    const FcstHostInputsView& in,
    FcstExpected* expected) {
    const int rows = run.rows;
    const int K = run.K;
    const int N = run.N;
    const int M = run.M;
    const int count = fcst_topm_count_host(N, M);

    expected->topm_cols.assign((size_t)rows * (size_t)M, -1);
    expected->topm_vals.assign((size_t)rows * (size_t)M, INT64_MIN);
    expected->topm_count.assign((size_t)rows, count);
    expected->row_sum.assign((size_t)rows, 0);
    expected->row_max.assign((size_t)rows, INT64_MIN);
    expected->row_argmax.assign((size_t)rows, -1);
    expected->row_nnz.assign((size_t)rows, 0);
    expected->row_nnz_prefix.assign((size_t)rows + 1, 0);

    for (int r = 0; r <= rows; ++r) {
        expected->row_nnz_prefix[(size_t)r] = in.row_offsets[r];
    }

    std::vector<FcstItem> items;
    items.reserve((size_t)N);

    for (int r = 0; r < rows; ++r) {
        const int row_begin = in.row_offsets[r];
        const int row_end = in.row_offsets[r + 1];

        expected->row_nnz[(size_t)r] = row_end - row_begin;
        items.clear();

        for (int n = 0; n < N; ++n) {
            const int64_t v = fcst_spmm_value_host(
                row_begin,
                row_end,
                K,
                N,
                n,
                in.col_indices,
                in.vals,
                in.dense_b);

            items.push_back(FcstItem{n, v});
        }

        std::sort(items.begin(), items.end(), fcst_item_less);

        uint64_t sum_bits = 0;
        int64_t max_v = INT64_MIN;
        int32_t argmax = -1;

        for (int i = 0; i < M; ++i) {
            const size_t out_idx = (size_t)r * (size_t)M + (size_t)i;

            if (i < count) {
                expected->topm_cols[out_idx] = items[(size_t)i].col;
                expected->topm_vals[out_idx] = items[(size_t)i].value;

                sum_bits += static_cast<uint64_t>(items[(size_t)i].value);

                if (items[(size_t)i].value > max_v ||
                    (items[(size_t)i].value == max_v &&
                     (argmax < 0 || items[(size_t)i].col < argmax))) {
                    max_v = items[(size_t)i].value;
                    argmax = items[(size_t)i].col;
                }
            }
        }

        expected->row_sum[(size_t)r] = static_cast<int64_t>(sum_bits);
        expected->row_max[(size_t)r] = count > 0 ? max_v : INT64_MIN;
        expected->row_argmax[(size_t)r] = count > 0 ? argmax : -1;
    }
}

static inline bool fcst_check_topm(
    const FcstRunSpec& run,
    const FcstExpected& expected,
    const FcstHostOutputsView& got,
    std::string* error) {
    const int rows = run.rows;
    const int M = run.M;

    for (int r = 0; r < rows; ++r) {
        if (got.topm_count[r] != expected.topm_count[(size_t)r]) {
            if (error) {
                std::ostringstream oss;
                oss << "topm_count mismatch row=" << r
                    << ": got " << got.topm_count[r]
                    << ", expected " << expected.topm_count[(size_t)r];
                *error = oss.str();
            }
            return false;
        }

        for (int i = 0; i < M; ++i) {
            const size_t idx = (size_t)r * (size_t)M + (size_t)i;

            if (got.topm_cols[idx] != expected.topm_cols[idx]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "topm_cols mismatch row=" << r
                        << ", i=" << i
                        << ": got " << got.topm_cols[idx]
                        << ", expected " << expected.topm_cols[idx];
                    *error = oss.str();
                }
                return false;
            }

            if (got.topm_vals[idx] != expected.topm_vals[idx]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "topm_vals mismatch row=" << r
                        << ", i=" << i
                        << ": got " << got.topm_vals[idx]
                        << ", expected " << expected.topm_vals[idx];
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

static inline bool fcst_check_summaries(
    const FcstRunSpec& run,
    const FcstExpected& expected,
    const FcstHostOutputsView& got,
    std::string* error) {
    const int rows = run.rows;

    for (int r = 0; r < rows; ++r) {
        if (got.row_sum[r] != expected.row_sum[(size_t)r]) {
            if (error) {
                std::ostringstream oss;
                oss << "row_sum mismatch row=" << r
                    << ": got " << got.row_sum[r]
                    << ", expected " << expected.row_sum[(size_t)r];
                *error = oss.str();
            }
            return false;
        }

        if (got.row_max[r] != expected.row_max[(size_t)r]) {
            if (error) {
                std::ostringstream oss;
                oss << "row_max mismatch row=" << r
                    << ": got " << got.row_max[r]
                    << ", expected " << expected.row_max[(size_t)r];
                *error = oss.str();
            }
            return false;
        }

        if (got.row_argmax[r] != expected.row_argmax[(size_t)r]) {
            if (error) {
                std::ostringstream oss;
                oss << "row_argmax mismatch row=" << r
                    << ": got " << got.row_argmax[r]
                    << ", expected " << expected.row_argmax[(size_t)r];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool fcst_check_metadata(
    const FcstRunSpec& run,
    const FcstExpected& expected,
    const FcstHostOutputsView& got,
    std::string* error) {
    const int rows = run.rows;

    for (int r = 0; r < rows; ++r) {
        if (got.row_nnz[r] != expected.row_nnz[(size_t)r]) {
            if (error) {
                std::ostringstream oss;
                oss << "row_nnz mismatch row=" << r
                    << ": got " << got.row_nnz[r]
                    << ", expected " << expected.row_nnz[(size_t)r];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int r = 0; r <= rows; ++r) {
        if (got.row_nnz_prefix[r] != expected.row_nnz_prefix[(size_t)r]) {
            if (error) {
                std::ostringstream oss;
                oss << "row_nnz_prefix mismatch index=" << r
                    << ": got " << got.row_nnz_prefix[r]
                    << ", expected " << expected.row_nnz_prefix[(size_t)r];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool fcst_check_all_outputs(
    const FcstRunSpec& run,
    const FcstExpected& expected,
    const FcstHostOutputsView& got,
    std::string* error) {
    if (!fcst_check_metadata(run, expected, got, error)) return false;
    if (!fcst_check_topm(run, expected, got, error)) return false;
    if (!fcst_check_summaries(run, expected, got, error)) return false;
    return true;
}

/*
GRADER CHECKS

Exact:
  - row_nnz[rows]
  - row_nnz_prefix[rows+1]
  - topm_count[rows]
  - topm_cols[rows,M]
  - topm_vals[rows,M]
  - row_sum[rows]
  - row_max[rows]
  - row_argmax[rows]

Harness should also enforce:
  - guard sentinels
  - input immutability
  - uniform, grid, many-tie, power-law/hub row-degree distributions
  - rows up to 65536
  - N in {256,1024,4096}
  - avg nnz/row in {4,16,64}
  - M in {4,16,64}
*/

#endif  // FUSED_CSR_SPMM_TOPK_ORACLE_HPP_
