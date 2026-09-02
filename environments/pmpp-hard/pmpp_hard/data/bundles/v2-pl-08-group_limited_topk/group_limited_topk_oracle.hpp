// ============================================================================
// file: group_limited_topk_oracle.hpp
// Independent CPU oracle + check helpers.
// ============================================================================

#ifndef GROUP_LIMITED_TOPK_ORACLE_HPP_
#define GROUP_LIMITED_TOPK_ORACLE_HPP_

#include "group_limited_topk_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct GltHostInputsView {
    const float* score;
    const float* uniform_u;
};

struct GltHostOutputsView {
    const int32_t* selected_group_ids;
    const int32_t* survivor_expert_ids;
    const int32_t* survivor_count;
    const float* weights;
    const float* group_scores;
};

struct GltExpected {
    std::vector<int32_t> selected_group_ids;
    std::vector<int32_t> survivor_expert_ids;
    std::vector<int32_t> survivor_count;
    std::vector<float> weights;
    std::vector<float> group_scores;
};

struct GltScoreId {
    float score;
    int32_t id;
};

static inline bool glt_oracle_expert_better(const GltScoreId& a, const GltScoreId& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.id < b.id;
}

static inline bool glt_oracle_group_better(const GltScoreId& a, const GltScoreId& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.id < b.id;
}

static inline void glt_insert_expert_cpu(
    const GltScoreId& cand,
    std::vector<GltScoreId>* top,
    int k) {
    size_t pos = top->size();

    for (size_t i = 0; i < top->size(); ++i) {
        if (glt_oracle_expert_better(cand, (*top)[i])) {
            pos = i;
            break;
        }
    }

    if (pos >= static_cast<size_t>(k)) {
        return;
    }

    top->insert(top->begin() + static_cast<std::ptrdiff_t>(pos), cand);

    if (top->size() > static_cast<size_t>(k)) {
        top->pop_back();
    }
}

static inline void glt_insert_group_cpu(
    const GltScoreId& cand,
    std::vector<GltScoreId>* top,
    int k) {
    size_t pos = top->size();

    for (size_t i = 0; i < top->size(); ++i) {
        if (glt_oracle_group_better(cand, (*top)[i])) {
            pos = i;
            break;
        }
    }

    if (pos >= static_cast<size_t>(k)) {
        return;
    }

    top->insert(top->begin() + static_cast<std::ptrdiff_t>(pos), cand);

    if (top->size() > static_cast<size_t>(k)) {
        top->pop_back();
    }
}

static inline void glt_cpu_oracle(
    const GltRunSpec& run,
    const GltHostInputsView& in,
    GltExpected* expected) {
    const int B = run.B;
    const int V = run.V;
    const int G = run.G;
    const int group_size = V / G;
    const int group_k = std::min(run.group_k, group_size);
    const int n_groups = run.n_groups;
    const int final_k = run.final_k;

    expected->selected_group_ids.assign((size_t)B * (size_t)n_groups, 0);
    expected->survivor_expert_ids.assign((size_t)B * (size_t)final_k, 0);
    expected->survivor_count.assign((size_t)B, 0);
    expected->weights.assign((size_t)B * (size_t)final_k, 0.0f);
    expected->group_scores.assign((size_t)B * (size_t)G, 0.0f);

    for (int row = 0; row < B; ++row) {
        const float* row_score = in.score + (size_t)row * (size_t)V;
        float* row_group_scores =
            expected->group_scores.data() + (size_t)row * (size_t)G;

        for (int g = 0; g < G; ++g) {
            std::vector<GltScoreId> top;
            top.reserve((size_t)group_k);

            const int start = g * group_size;
            for (int off = 0; off < group_size; ++off) {
                const int expert = start + off;
                glt_insert_expert_cpu(GltScoreId{row_score[expert], expert}, &top, group_k);
            }

            float sum = 0.0f;
            for (const GltScoreId& item : top) {
                sum += item.score;
            }
            row_group_scores[g] = sum;
        }

        std::vector<GltScoreId> top_groups;
        top_groups.reserve((size_t)n_groups);

        for (int g = 0; g < G; ++g) {
            glt_insert_group_cpu(GltScoreId{row_group_scores[g], g}, &top_groups, n_groups);
        }

        int32_t* row_selected =
            expected->selected_group_ids.data() + (size_t)row * (size_t)n_groups;

        for (int i = 0; i < n_groups; ++i) {
            row_selected[i] = top_groups[(size_t)i].id;
        }

        const int possible = n_groups * group_size;
        const int count = std::min(final_k, possible);

        std::vector<GltScoreId> top_experts;
        top_experts.reserve((size_t)count);

        for (int gi = 0; gi < n_groups; ++gi) {
            const int g = row_selected[gi];
            const int start = g * group_size;

            for (int off = 0; off < group_size; ++off) {
                const int expert = start + off;
                glt_insert_expert_cpu(GltScoreId{row_score[expert], expert}, &top_experts, count);
            }
        }

        int32_t* row_experts =
            expected->survivor_expert_ids.data() + (size_t)row * (size_t)final_k;
        float* row_weights =
            expected->weights.data() + (size_t)row * (size_t)final_k;

        expected->survivor_count[(size_t)row] = count;

        float max_s = -std::numeric_limits<float>::infinity();
        for (const GltScoreId& item : top_experts) {
            if (item.score > max_s) {
                max_s = item.score;
            }
        }

        float denom = 0.0f;
        for (const GltScoreId& item : top_experts) {
            denom += std::exp(item.score - max_s);
        }
        if (denom <= 0.0f) denom = 1.0f;

        for (int i = 0; i < count; ++i) {
            row_experts[i] = top_experts[(size_t)i].id;
            row_weights[i] = std::exp(top_experts[(size_t)i].score - max_s) / denom;
        }
    }
}

static inline bool glt_check_group_scores(
    const GltRunSpec& run,
    const GltExpected& expected,
    const GltHostOutputsView& got,
    std::string* error) {
    const int B = run.B;
    const int G = run.G;

    for (int row = 0; row < B; ++row) {
        for (int g = 0; g < G; ++g) {
            const size_t idx = (size_t)row * (size_t)G + (size_t)g;
            const float exp_v = expected.group_scores[idx];
            const float got_v = got.group_scores[idx];
            const float diff = std::fabs(got_v - exp_v);
            const float tol =
                GLT_GROUP_SCORE_ATOL + GLT_GROUP_SCORE_RTOL * std::fabs(exp_v);

            if (!(diff <= tol)) {
                if (error) {
                    std::ostringstream oss;
                    oss << "group_scores mismatch at row=" << row
                        << ", g=" << g
                        << ": got " << got_v
                        << ", expected " << exp_v
                        << ", diff=" << diff
                        << ", tol=" << tol;
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

static inline bool glt_check_selected_groups(
    const GltRunSpec& run,
    const GltExpected& expected,
    const GltHostOutputsView& got,
    std::string* error) {
    const int B = run.B;
    const int n_groups = run.n_groups;

    for (int row = 0; row < B; ++row) {
        for (int i = 0; i < n_groups; ++i) {
            const size_t idx = (size_t)row * (size_t)n_groups + (size_t)i;

            if (got.selected_group_ids[idx] != expected.selected_group_ids[idx]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "selected_group_ids mismatch at row=" << row
                        << ", i=" << i
                        << ": got " << got.selected_group_ids[idx]
                        << ", expected " << expected.selected_group_ids[idx];
                    *error = oss.str();
                }
                return false;
            }

            if (got.selected_group_ids[idx] < 0 || got.selected_group_ids[idx] >= run.G) {
                if (error) {
                    std::ostringstream oss;
                    oss << "selected_group_ids out of range at row=" << row
                        << ", i=" << i
                        << ": got " << got.selected_group_ids[idx];
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

static inline bool glt_check_survivors_and_weights(
    const GltRunSpec& run,
    const GltExpected& expected,
    const GltHostOutputsView& got,
    std::string* error) {
    const int B = run.B;
    const int V = run.V;
    const int final_k = run.final_k;

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

        if (got.survivor_count[row] < 1 || got.survivor_count[row] > final_k) {
            if (error) {
                std::ostringstream oss;
                oss << "survivor_count out of range at row=" << row
                    << ": got " << got.survivor_count[row];
                *error = oss.str();
            }
            return false;
        }

        float sum_w = 0.0f;

        for (int i = 0; i < exp_count; ++i) {
            const size_t idx = (size_t)row * (size_t)final_k + (size_t)i;

            if (got.survivor_expert_ids[idx] != expected.survivor_expert_ids[idx]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "survivor_expert_ids mismatch at row=" << row
                        << ", i=" << i
                        << ": got " << got.survivor_expert_ids[idx]
                        << ", expected " << expected.survivor_expert_ids[idx];
                    *error = oss.str();
                }
                return false;
            }

            if (got.survivor_expert_ids[idx] < 0 || got.survivor_expert_ids[idx] >= V) {
                if (error) {
                    std::ostringstream oss;
                    oss << "survivor expert out of range at row=" << row
                        << ", i=" << i
                        << ": got " << got.survivor_expert_ids[idx];
                    *error = oss.str();
                }
                return false;
            }

            const float exp_w = expected.weights[idx];
            const float got_w = got.weights[idx];
            const float diff = std::fabs(got_w - exp_w);
            const float tol = GLT_WEIGHT_ATOL + GLT_WEIGHT_RTOL * std::fabs(exp_w);

            if (!(diff <= tol)) {
                if (error) {
                    std::ostringstream oss;
                    oss << "weights mismatch at row=" << row
                        << ", i=" << i
                        << ": got " << got_w
                        << ", expected " << exp_w
                        << ", diff=" << diff
                        << ", tol=" << tol;
                    *error = oss.str();
                }
                return false;
            }

            sum_w += got_w;
        }

        const float sum_diff = std::fabs(sum_w - 1.0f);
        if (!(sum_diff <= 5.0e-4f)) {
            if (error) {
                std::ostringstream oss;
                oss << "weight sum mismatch at row=" << row
                    << ": sum=" << sum_w
                    << ", diff=" << sum_diff;
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool glt_check_all_outputs(
    const GltRunSpec& run,
    const GltExpected& expected,
    const GltHostOutputsView& got,
    std::string* error) {
    if (!glt_check_group_scores(run, expected, got, error)) return false;
    if (!glt_check_selected_groups(run, expected, got, error)) return false;
    if (!glt_check_survivors_and_weights(run, expected, got, error)) return false;
    return true;
}

/*
INTERMEDIATE/CORRECTNESS CHECKS REQUIRED BY GRADER

The grader should copy back and verify:

  1. group_scores[B, G]
     - fp32 tolerance:
         abs_error <= 1e-4 + 1e-4 * abs(expected)
     - score is sum of top group_k scores inside the group.

  2. selected_group_ids[B, n_groups]
     - exact equality.
     - sorted by group_score descending, tie smaller group id.

  3. survivor_count[B]
     - exact equality.
     - equals min(final_k, n_groups * group_size).

  4. survivor_expert_ids[B, final_k]
     - exact equality for entries [0, survivor_count[row]).
     - sorted by score descending, tie smaller expert id.
     - experts must come only from selected groups.

  5. weights[B, final_k]
     - fp32 tolerance:
         abs_error <= 5e-5 + 5e-5 * abs(expected)
     - probabilities softmax over selected final-k scores.
     - checked sum close to 1.

The grader should additionally enforce:
  - Sentinels around every output allocation.
  - Input immutability hash before/after solution_run.
  - Held-out seed-shifted distributions from the declared shape family.
  - Rows covering:
      uniform scores,
      peaked experts,
      group-skewed scores,
      many exact ties,
      ties at group boundary,
      ties at final-k boundary.
*/

#endif  // GROUP_LIMITED_TOPK_ORACLE_HPP_
