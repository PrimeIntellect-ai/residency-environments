// ============================================================================
// file: penalty_filter_sample_oracle.hpp
// Independent CPU oracle + check helpers.
// ============================================================================

#ifndef PENALTY_FILTER_SAMPLE_ORACLE_HPP_
#define PENALTY_FILTER_SAMPLE_ORACLE_HPP_

#include "penalty_filter_sample_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

struct PfsHostInputsView {
    const float* logits;
    const int32_t* history_token;
    const int32_t* history_len;
    const float* repetition_penalty;
    const float* frequency_penalty;
    const float* presence_penalty;
    const float* temperature;
    const float* min_p;
    const float* uniform_u;
};

struct PfsHostOutputsView {
    const int32_t* selected_token;
    const int32_t* survivor_count;
    const int32_t* packed_cand_token;
    const float* packed_cand_prob;
};

struct PfsExpected {
    std::vector<int32_t> selected_token;
    std::vector<int32_t> survivor_count;
    std::vector<int32_t> packed_cand_token;
    std::vector<float> packed_cand_prob;
};

struct PfsScoreId {
    float score;
    int32_t id;
};

static inline bool pfs_oracle_better(const PfsScoreId& a, const PfsScoreId& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.id < b.id;
}

static inline float pfs_oracle_fadd(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float cv = av + bv;
    return cv;
}

static inline float pfs_oracle_expf(float x) {
    return ::expf(x);
}

static inline float pfs_oracle_sanitize_rp(float rp) {
    return rp > 1.0f ? rp : 1.0f;
}

static inline float pfs_oracle_sanitize_temp(float t) {
    return t > 1.0e-6f ? t : 1.0e-6f;
}

static inline float pfs_oracle_sanitize_min_p(float p) {
    if (p < 0.0f) return 0.0f;
    if (p >= 1.0f) return 0.9999999403953552f;
    return p;
}

static inline float pfs_oracle_sanitize_u(float u) {
    if (u < 0.0f) return 0.0f;
    if (u >= 1.0f) return 0.9999999403953552f;
    return u;
}

static inline float pfs_oracle_adjust_score(
    float logit,
    int count,
    float repetition_penalty,
    float frequency_penalty,
    float presence_penalty,
    float temperature) {
    float x = logit;

    if (count > 0) {
        const float rp = pfs_oracle_sanitize_rp(repetition_penalty);
        if (x > 0.0f) {
            x = x / rp;
        } else {
            x = x * rp;
        }

        x -= frequency_penalty * static_cast<float>(count);
        x -= presence_penalty;
    }

    return x / pfs_oracle_sanitize_temp(temperature);
}

static inline void pfs_cpu_oracle(
    const PfsRunSpec& run,
    const PfsHostInputsView& in,
    PfsExpected* expected) {
    const int B = run.B;
    const int V = run.V;
    const int H = run.H;

    expected->selected_token.assign((size_t)B, 0);
    expected->survivor_count.assign((size_t)B, 0);
    expected->packed_cand_token.assign((size_t)B * (size_t)V, 0);
    expected->packed_cand_prob.assign((size_t)B * (size_t)V, 0.0f);

    std::vector<int32_t> counts((size_t)V, 0);
    std::vector<PfsScoreId> items((size_t)V);

    for (int row = 0; row < B; ++row) {
        std::fill(counts.begin(), counts.end(), 0);

        int hist_len = in.history_len[row];
        if (hist_len < 0) hist_len = 0;
        if (hist_len > H) hist_len = H;

        if (H > 0 && in.history_token) {
            const int32_t* row_hist = in.history_token + (size_t)row * (size_t)H;

            for (int h = 0; h < hist_len; ++h) {
                const int token = row_hist[h];
                if (token >= 0 && token < V) {
                    ++counts[(size_t)token];
                }
            }
        }

        const float* row_logits = in.logits + (size_t)row * (size_t)V;

        for (int token = 0; token < V; ++token) {
            items[(size_t)token] = PfsScoreId{
                pfs_oracle_adjust_score(
                    row_logits[token],
                    counts[(size_t)token],
                    in.repetition_penalty[row],
                    in.frequency_penalty[row],
                    in.presence_penalty[row],
                    in.temperature[row]),
                token
            };
        }

        std::sort(items.begin(), items.end(), pfs_oracle_better);

        const float max_score = items[0].score;
        const float threshold = pfs_oracle_sanitize_min_p(in.min_p[row]);

        int count = 0;

        if (threshold <= 0.0f) {
            count = V;
        } else {
            for (int i = 0; i < V; ++i) {
                const float rel = pfs_oracle_expf(items[(size_t)i].score - max_score);
                if (rel >= threshold) {
                    ++count;
                } else {
                    break;
                }
            }

            if (count <= 0) count = 1;
        }

        float denom = 0.0f;
        for (int i = 0; i < count; ++i) {
            denom = pfs_oracle_fadd(
                denom,
                pfs_oracle_expf(items[(size_t)i].score - max_score));
        }
        if (denom <= 0.0f) denom = 1.0f;

        int32_t* row_tokens =
            expected->packed_cand_token.data() + (size_t)row * (size_t)V;
        float* row_probs =
            expected->packed_cand_prob.data() + (size_t)row * (size_t)V;

        for (int i = 0; i < count; ++i) {
            row_tokens[i] = items[(size_t)i].id;
            row_probs[i] =
                pfs_oracle_expf(items[(size_t)i].score - max_score) / denom;
        }

        const float u = pfs_oracle_sanitize_u(in.uniform_u[row]);
        float cdf = 0.0f;
        int selected = row_tokens[count - 1];

        for (int i = 0; i < count; ++i) {
            cdf = pfs_oracle_fadd(cdf, row_probs[i]);

            if (cdf > u) {
                selected = row_tokens[i];
                break;
            }
        }

        expected->selected_token[(size_t)row] = selected;
        expected->survivor_count[(size_t)row] = count;
    }
}

static inline bool pfs_check_selected_and_count(
    const PfsRunSpec& run,
    const PfsExpected& expected,
    const PfsHostOutputsView& got,
    std::string* error) {
    const int B = run.B;
    const int V = run.V;

    for (int row = 0; row < B; ++row) {
        const int exp_count = expected.survivor_count[(size_t)row];

        if (got.survivor_count[row] != exp_count) {
            if (error) {
                std::ostringstream oss;
                oss << "survivor_count mismatch at row=" << row
                    << ": got " << got.survivor_count[row]
                    << ", expected " << exp_count;
                *error = oss.str();
            }
            return false;
        }

        if (got.survivor_count[row] < 1 || got.survivor_count[row] > V) {
            if (error) {
                std::ostringstream oss;
                oss << "survivor_count out of range at row=" << row
                    << ": got " << got.survivor_count[row]
                    << ", V=" << V;
                *error = oss.str();
            }
            return false;
        }

        if (got.selected_token[row] != expected.selected_token[(size_t)row]) {
            if (error) {
                std::ostringstream oss;
                oss << "selected_token mismatch at row=" << row
                    << ": got " << got.selected_token[row]
                    << ", expected " << expected.selected_token[(size_t)row];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool pfs_check_candidates(
    const PfsRunSpec& run,
    const PfsExpected& expected,
    const PfsHostOutputsView& got,
    std::string* error) {
    const int B = run.B;
    const int V = run.V;

    for (int row = 0; row < B; ++row) {
        const int count = expected.survivor_count[(size_t)row];

        const int32_t* exp_tokens =
            expected.packed_cand_token.data() + (size_t)row * (size_t)V;
        const float* exp_probs =
            expected.packed_cand_prob.data() + (size_t)row * (size_t)V;

        const int32_t* got_tokens = got.packed_cand_token + (size_t)row * (size_t)V;
        const float* got_probs = got.packed_cand_prob + (size_t)row * (size_t)V;

        float prob_sum = 0.0f;

        for (int i = 0; i < count; ++i) {
            if (got_tokens[i] != exp_tokens[i]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_cand_token mismatch at row=" << row
                        << ", i=" << i
                        << ": got " << got_tokens[i]
                        << ", expected " << exp_tokens[i];
                    *error = oss.str();
                }
                return false;
            }

            const float diff = std::fabs(got_probs[i] - exp_probs[i]);
            const float tol = PFS_PROB_ATOL + PFS_PROB_RTOL * std::fabs(exp_probs[i]);

            if (!(diff <= tol)) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_cand_prob mismatch at row=" << row
                        << ", i=" << i
                        << ": got " << got_probs[i]
                        << ", expected " << exp_probs[i]
                        << ", diff=" << diff
                        << ", tol=" << tol;
                    *error = oss.str();
                }
                return false;
            }

            prob_sum = pfs_oracle_fadd(prob_sum, got_probs[i]);

            if (got_tokens[i] < 0 || got_tokens[i] >= V) {
                if (error) {
                    std::ostringstream oss;
                    oss << "candidate token out of range at row=" << row
                        << ", i=" << i
                        << ": token=" << got_tokens[i];
                    *error = oss.str();
                }
                return false;
            }
        }

        const float sum_diff = std::fabs(prob_sum - 1.0f);
        if (!(sum_diff <= 5.0e-4f)) {
            if (error) {
                std::ostringstream oss;
                oss << "renormalized probability sum mismatch at row=" << row
                    << ": sum=" << prob_sum
                    << ", diff=" << sum_diff;
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool pfs_check_all_outputs(
    const PfsRunSpec& run,
    const PfsExpected& expected,
    const PfsHostOutputsView& got,
    std::string* error) {
    if (!pfs_check_selected_and_count(run, expected, got, error)) return false;
    if (!pfs_check_candidates(run, expected, got, error)) return false;
    return true;
}

/*
INTERMEDIATE/CORRECTNESS CHECKS REQUIRED BY GRADER

The grader should copy back and verify:

  1. selected_token[B]
     - exact equality against CPU oracle.
     - deterministic under the specified strict cumsum > u rule.

  2. survivor_count[B]
     - exact equality against CPU oracle.
     - 1 <= count <= V.

  3. packed_cand_token[B,V]
     - exact equality for the first survivor_count[row] entries.
     - order must be:
         adjusted score descending,
         token id ascending for ties.

  4. packed_cand_prob[B,V]
     - final renormalized survivor weights after penalties and min-p.
     - tolerance:
         abs_error <= 5e-5 + 5e-5 * abs(expected)
     - sum should be close to 1.

The grader should additionally enforce:
  - Sentinels around every output allocation.
  - Input immutability hash before/after solution_run.
  - Held-out seed-shifted distributions from the declared shape family.
  - Rows covering:
      empty history,
      repeated-token history,
      all tokens in history from a small set,
      min_p = 0,
      min_p large,
      all-equal adjusted logits,
      peaked and heavy-tail logits.
*/

#endif  // PENALTY_FILTER_SAMPLE_ORACLE_HPP_
