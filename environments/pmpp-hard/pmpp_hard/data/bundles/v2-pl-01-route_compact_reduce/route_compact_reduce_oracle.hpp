// ============================================================================
// file: route_compact_reduce_oracle.hpp
// Independent CPU oracle + intermediate/final check helpers.
// ============================================================================

#ifndef ROUTE_COMPACT_REDUCE_ORACLE_HPP_
#define ROUTE_COMPACT_REDUCE_ORACLE_HPP_

#include "route_compact_reduce_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct RcrHostInputsView {
    const int16_t* x;
    const int16_t* expert;
    const int16_t* weight;
    const uint8_t* valid;
};

struct RcrHostOutputsView {
    const int32_t* counts;
    const int32_t* offsets;
    const int32_t* packed_token;
    const int32_t* packed_weight;
    const int64_t* sum;
    const int32_t* argmax_abs;
};

struct RcrExpected {
    std::vector<int32_t> counts;
    std::vector<int32_t> offsets;
    std::vector<int32_t> packed_token;
    std::vector<int32_t> packed_weight;
    std::vector<int64_t> sum;
    std::vector<int32_t> argmax_abs;
};

struct RcrRoute {
    int32_t token;
    int32_t weight;
};

static inline uint64_t rcr_oracle_abs_i64(int64_t v) {
    const uint64_t u = static_cast<uint64_t>(v);
    return v < 0 ? (~u + 1ULL) : u;
}

static inline int64_t rcr_oracle_mul_i32_i16_to_i64(int32_t a, int16_t b) {
    return static_cast<int64_t>(a) * static_cast<int64_t>(b);
}

static inline void rcr_cpu_oracle(
    const RcrRunSpec& run,
    const RcrHostInputsView& in,
    RcrExpected* expected) {
    const int N = run.N;
    const int C = run.C;
    const int E = run.E;
    const int K = run.K;

    expected->counts.assign(E, 0);
    expected->offsets.assign(E + 1, 0);
    expected->packed_token.clear();
    expected->packed_weight.clear();
    expected->sum.assign((size_t)E * (size_t)C, 0);
    expected->argmax_abs.assign((size_t)E * (size_t)C, -1);

    std::vector<std::vector<RcrRoute>> by_expert((size_t)E);

    for (int t = 0; t < N; ++t) {
        if (in.valid[t] == 0) continue;

        int loc_exp[RCR_MAX_K];
        int loc_w[RCR_MAX_K];
        int loc_n = 0;

        for (int i = 0; i < RCR_MAX_K; ++i) {
            loc_exp[i] = -1;
            loc_w[i] = 0;
        }

        for (int k = 0; k < K; ++k) {
            const int e = static_cast<int>(in.expert[t * K + k]);
            const int w = static_cast<int>(in.weight[t * K + k]);

            if (e < 0 || e >= E || w == 0) continue;

            int found = -1;
            for (int j = 0; j < loc_n; ++j) {
                if (loc_exp[j] == e) {
                    found = j;
                    break;
                }
            }

            if (found >= 0) {
                loc_w[found] += w;
            } else {
                loc_exp[loc_n] = e;
                loc_w[loc_n] = w;
                ++loc_n;
            }
        }

        for (int j = 0; j < loc_n; ++j) {
            if (loc_w[j] == 0) continue;
            by_expert[(size_t)loc_exp[j]].push_back(RcrRoute{t, loc_w[j]});
        }
    }

    int32_t total = 0;
    expected->offsets[0] = 0;
    for (int e = 0; e < E; ++e) {
        const int32_t count = static_cast<int32_t>(by_expert[(size_t)e].size());
        expected->counts[e] = count;
        total += count;
        expected->offsets[e + 1] = total;
    }

    expected->packed_token.resize((size_t)total);
    expected->packed_weight.resize((size_t)total);

    for (int e = 0; e < E; ++e) {
        const int32_t begin = expected->offsets[e];
        const std::vector<RcrRoute>& routes = by_expert[(size_t)e];

        for (size_t i = 0; i < routes.size(); ++i) {
            expected->packed_token[(size_t)begin + i] = routes[i].token;
            expected->packed_weight[(size_t)begin + i] = routes[i].weight;
        }
    }

    for (int e = 0; e < E; ++e) {
        const int32_t begin = expected->offsets[e];
        const int32_t end = expected->offsets[e + 1];

        for (int c = 0; c < C; ++c) {
            uint64_t sum_bits = 0;
            uint64_t best_abs = 0;
            int32_t best_token = -1;
            bool any = false;

            for (int32_t i = begin; i < end; ++i) {
                const int32_t token = expected->packed_token[(size_t)i];
                const int32_t w = expected->packed_weight[(size_t)i];
                const int16_t xv = in.x[(size_t)token * (size_t)C + (size_t)c];
                const int64_t prod = rcr_oracle_mul_i32_i16_to_i64(w, xv);
                const uint64_t abs_v = rcr_oracle_abs_i64(prod);

                sum_bits += static_cast<uint64_t>(prod);

                if (!any ||
                    abs_v > best_abs ||
                    (abs_v == best_abs && token < best_token)) {
                    any = true;
                    best_abs = abs_v;
                    best_token = token;
                }
            }

            const size_t out_idx = (size_t)e * (size_t)C + (size_t)c;
            expected->sum[out_idx] = static_cast<int64_t>(sum_bits);
            expected->argmax_abs[out_idx] = any ? best_token : -1;
        }
    }
}

static inline bool rcr_check_counts_offsets(
    const RcrRunSpec& run,
    const RcrExpected& expected,
    const RcrHostOutputsView& got,
    std::string* error) {
    const int E = run.E;

    for (int e = 0; e < E; ++e) {
        if (got.counts[e] != expected.counts[(size_t)e]) {
            if (error) {
                std::ostringstream oss;
                oss << "counts mismatch at e=" << e
                    << ": got " << got.counts[e]
                    << ", expected " << expected.counts[(size_t)e];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int e = 0; e <= E; ++e) {
        if (got.offsets[e] != expected.offsets[(size_t)e]) {
            if (error) {
                std::ostringstream oss;
                oss << "offsets mismatch at e=" << e
                    << ": got " << got.offsets[e]
                    << ", expected " << expected.offsets[(size_t)e];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.offsets[0] != 0) {
        if (error) *error = "offsets[0] must be zero";
        return false;
    }

    for (int e = 0; e < E; ++e) {
        if (got.offsets[e + 1] - got.offsets[e] != got.counts[e]) {
            if (error) {
                std::ostringstream oss;
                oss << "offset/count invariant failed at e=" << e;
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool rcr_check_packed_intermediates(
    const RcrRunSpec& run,
    const RcrExpected& expected,
    const RcrHostOutputsView& got,
    std::string* error) {
    const int E = run.E;
    const int32_t total = expected.offsets[(size_t)E];

    for (int32_t i = 0; i < total; ++i) {
        if (got.packed_token[i] != expected.packed_token[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_token mismatch at i=" << i
                    << ": got " << got.packed_token[i]
                    << ", expected " << expected.packed_token[(size_t)i];
                *error = oss.str();
            }
            return false;
        }

        if (got.packed_weight[i] != expected.packed_weight[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_weight mismatch at i=" << i
                    << ": got " << got.packed_weight[i]
                    << ", expected " << expected.packed_weight[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int e = 0; e < E; ++e) {
        const int32_t begin = got.offsets[e];
        const int32_t end = got.offsets[e + 1];

        int32_t prev_token = -1;
        for (int32_t i = begin; i < end; ++i) {
            const int32_t tok = got.packed_token[i];
            if (tok <= prev_token) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_token is not strictly increasing in expert "
                        << e << " at packed index " << i
                        << ": prev=" << prev_token << ", current=" << tok;
                    *error = oss.str();
                }
                return false;
            }
            if (tok < 0 || tok >= run.N) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_token out of range at i=" << i
                        << ": token=" << tok;
                    *error = oss.str();
                }
                return false;
            }
            prev_token = tok;
        }
    }

    return true;
}

static inline bool rcr_check_final_outputs(
    const RcrRunSpec& run,
    const RcrExpected& expected,
    const RcrHostOutputsView& got,
    std::string* error) {
    const int E = run.E;
    const int C = run.C;
    const size_t EC = (size_t)E * (size_t)C;

    for (size_t i = 0; i < EC; ++i) {
        if (got.sum[i] != expected.sum[i]) {
            if (error) {
                const int e = static_cast<int>(i / (size_t)C);
                const int c = static_cast<int>(i % (size_t)C);
                std::ostringstream oss;
                oss << "sum mismatch at e=" << e << ", c=" << c
                    << ": got " << got.sum[i]
                    << ", expected " << expected.sum[i];
                *error = oss.str();
            }
            return false;
        }

        if (got.argmax_abs[i] != expected.argmax_abs[i]) {
            if (error) {
                const int e = static_cast<int>(i / (size_t)C);
                const int c = static_cast<int>(i % (size_t)C);
                std::ostringstream oss;
                oss << "argmax_abs mismatch at e=" << e << ", c=" << c
                    << ": got " << got.argmax_abs[i]
                    << ", expected " << expected.argmax_abs[i];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool rcr_check_all_outputs(
    const RcrRunSpec& run,
    const RcrExpected& expected,
    const RcrHostOutputsView& got,
    std::string* error) {
    if (!rcr_check_counts_offsets(run, expected, got, error)) return false;
    if (!rcr_check_packed_intermediates(run, expected, got, error)) return false;
    if (!rcr_check_final_outputs(run, expected, got, error)) return false;
    return true;
}

/*
INTERMEDIATE CHECKS REQUIRED BY GRADER

The grader should copy back and verify:

  1. counts[E]
     - exact equality against CPU oracle.
     - counts[e] must equal offsets[e+1] - offsets[e].

  2. offsets[E + 1]
     - exact equality against CPU oracle.
     - offsets[0] == 0.
     - monotone nondecreasing.
     - offsets[E] == total surviving coalesced routes.

  3. packed_token[0:offsets[E]]
     - exact equality against CPU oracle.
     - for each expert slice [offsets[e], offsets[e+1]), tokens must be
       strictly increasing.
     - each token must be in [0, N).

  4. packed_weight[0:offsets[E]]
     - exact equality against CPU oracle.
     - verifies duplicate-expert coalescing and zero-combined-route omission.

  5. sum[E, C]
     - exact int64 two's-complement equality against CPU oracle.

  6. argmax_abs[E, C]
     - exact equality against CPU oracle.
     - empty experts must output -1 for every channel.

The grader should additionally enforce:
  - Sentinels around every output allocation.
  - Input immutability hash before/after solution_run.
  - Determinism replay with the same case repeated at least twice.
  - Held-out seed-shifted distributions from the declared shape family.
*/

#endif  // ROUTE_COMPACT_REDUCE_ORACLE_HPP_
