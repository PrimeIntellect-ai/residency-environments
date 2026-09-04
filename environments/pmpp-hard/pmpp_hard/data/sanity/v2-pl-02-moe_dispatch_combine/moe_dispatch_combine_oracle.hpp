// ============================================================================
// file: moe_dispatch_combine_oracle.hpp
// Independent CPU oracle + intermediate/final check helpers.
// ============================================================================

#ifndef MOE_DISPATCH_COMBINE_ORACLE_HPP_
#define MOE_DISPATCH_COMBINE_ORACLE_HPP_

#include "moe_dispatch_combine_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

struct MdcHostInputsView {
    const int16_t* expert;
    const int16_t* gate;
    const uint8_t* valid;
    const int16_t* expert_out;
};

struct MdcHostOutputsView {
    const int32_t* counts;
    const int32_t* offsets;
    const int32_t* packed_token;
    const int32_t* packed_slot;
    const int16_t* packed_gate;
    const uint8_t* dropped;
    const int64_t* y;
};

struct MdcExpected {
    std::vector<int32_t> counts;
    std::vector<int32_t> offsets;
    std::vector<int32_t> packed_token;
    std::vector<int32_t> packed_slot;
    std::vector<int16_t> packed_gate;
    std::vector<uint8_t> dropped;
    std::vector<int64_t> y;
};

struct MdcRoute {
    int32_t token;
    int32_t slot;
    int32_t expert;
    int32_t gate;
};

static inline bool mdc_oracle_gate_is_legal(int gate) {
    return gate >= MDC_GATE_MIN && gate <= MDC_GATE_MAX;
}

static inline bool mdc_oracle_route_less(const MdcRoute& a, const MdcRoute& b) {
    if (a.gate != b.gate) {
        return a.gate > b.gate;
    }
    if (a.token != b.token) {
        return a.token < b.token;
    }
    return a.slot < b.slot;
}

static inline void mdc_cpu_oracle(
    const MdcRunSpec& run,
    const MdcHostInputsView& in,
    MdcExpected* expected) {
    const int N = run.N;
    const int D = run.D;
    const int E = run.E;
    const int K = run.K;
    const int cap = run.cap;

    expected->counts.assign(E, 0);
    expected->offsets.assign(E + 1, 0);
    expected->packed_token.clear();
    expected->packed_slot.clear();
    expected->packed_gate.clear();
    expected->dropped.assign((size_t)N * (size_t)K, 0);
    expected->y.assign((size_t)N * (size_t)D, 0);

    std::vector<std::vector<MdcRoute>> by_expert((size_t)E);

    for (int t = 0; t < N; ++t) {
        if (in.valid[t] == 0) continue;

        const int base = t * K;
        for (int k = 0; k < K; ++k) {
            const int e = static_cast<int>(in.expert[base + k]);
            const int g = static_cast<int>(in.gate[base + k]);

            if (e < 0 || e >= E || !mdc_oracle_gate_is_legal(g)) {
                continue;
            }

            by_expert[(size_t)e].push_back(MdcRoute{t, k, e, g});
        }
    }

    for (int e = 0; e < E; ++e) {
        std::vector<MdcRoute>& routes = by_expert[(size_t)e];
        std::sort(routes.begin(), routes.end(), mdc_oracle_route_less);

        const int kept = std::min(static_cast<int>(routes.size()), cap);
        expected->counts[(size_t)e] = kept;

        for (size_t i = (size_t)kept; i < routes.size(); ++i) {
            const MdcRoute& r = routes[i];
            expected->dropped[(size_t)r.token * (size_t)K + (size_t)r.slot] = 1;
        }
    }

    int32_t total = 0;
    expected->offsets[0] = 0;

    for (int e = 0; e < E; ++e) {
        total += expected->counts[(size_t)e];
        expected->offsets[(size_t)e + 1] = total;
    }

    expected->packed_token.resize((size_t)total);
    expected->packed_slot.resize((size_t)total);
    expected->packed_gate.resize((size_t)total);

    for (int e = 0; e < E; ++e) {
        const int begin = expected->offsets[(size_t)e];
        const int kept = expected->counts[(size_t)e];

        const std::vector<MdcRoute>& routes = by_expert[(size_t)e];

        for (int i = 0; i < kept; ++i) {
            const MdcRoute& r = routes[(size_t)i];
            const size_t out_idx = (size_t)begin + (size_t)i;

            expected->packed_token[out_idx] = r.token;
            expected->packed_slot[out_idx] = r.slot;
            expected->packed_gate[out_idx] = static_cast<int16_t>(r.gate);
        }
    }

    for (int e = 0; e < E; ++e) {
        const int begin = expected->offsets[(size_t)e];
        const int end = expected->offsets[(size_t)e + 1];

        for (int i = begin; i < end; ++i) {
            const int token = expected->packed_token[(size_t)i];
            const int gate = static_cast<int>(expected->packed_gate[(size_t)i]);

            for (int d = 0; d < D; ++d) {
                const int out_v = static_cast<int>(in.expert_out[(size_t)e * (size_t)D + (size_t)d]);
                const int64_t prod =
                    static_cast<int64_t>(gate) * static_cast<int64_t>(out_v);

                const size_t y_idx = (size_t)token * (size_t)D + (size_t)d;
                const uint64_t acc =
                    static_cast<uint64_t>(expected->y[y_idx]) +
                    static_cast<uint64_t>(prod);

                expected->y[y_idx] = static_cast<int64_t>(acc);
            }
        }
    }
}

static inline bool mdc_check_counts_offsets(
    const MdcRunSpec& run,
    const MdcExpected& expected,
    const MdcHostOutputsView& got,
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

        if (got.counts[e] < 0 || got.counts[e] > run.cap) {
            if (error) {
                std::ostringstream oss;
                oss << "counts out of range at e=" << e
                    << ": got " << got.counts[e]
                    << ", cap " << run.cap;
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

static inline bool mdc_check_packed_intermediates(
    const MdcRunSpec& run,
    const MdcExpected& expected,
    const MdcHostOutputsView& got,
    std::string* error) {
    const int E = run.E;
    const int total = expected.offsets[(size_t)E];

    for (int i = 0; i < total; ++i) {
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

        if (got.packed_slot[i] != expected.packed_slot[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_slot mismatch at i=" << i
                    << ": got " << got.packed_slot[i]
                    << ", expected " << expected.packed_slot[(size_t)i];
                *error = oss.str();
            }
            return false;
        }

        if (got.packed_gate[i] != expected.packed_gate[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_gate mismatch at i=" << i
                    << ": got " << got.packed_gate[i]
                    << ", expected " << expected.packed_gate[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int e = 0; e < E; ++e) {
        const int begin = got.offsets[e];
        const int end = got.offsets[e + 1];

        int prev_gate = MDC_GATE_MAX + 1;
        int prev_token = -1;
        int prev_slot = -1;

        for (int i = begin; i < end; ++i) {
            const int token = got.packed_token[i];
            const int slot = got.packed_slot[i];
            const int gate = static_cast<int>(got.packed_gate[i]);

            if (token < 0 || token >= run.N) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_token out of range at i=" << i
                        << ": token=" << token;
                    *error = oss.str();
                }
                return false;
            }

            if (slot < 0 || slot >= run.K) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_slot out of range at i=" << i
                        << ": slot=" << slot;
                    *error = oss.str();
                }
                return false;
            }

            if (!mdc_oracle_gate_is_legal(gate)) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_gate out of legal range at i=" << i
                        << ": gate=" << gate;
                    *error = oss.str();
                }
                return false;
            }

            bool ordered = false;
            if (i == begin) {
                ordered = true;
            } else if (gate < prev_gate) {
                ordered = true;
            } else if (gate == prev_gate && token > prev_token) {
                ordered = true;
            } else if (gate == prev_gate && token == prev_token && slot > prev_slot) {
                ordered = true;
            }

            if (!ordered) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed route order violation at expert=" << e
                        << ", i=" << i
                        << ": previous(gate,token,slot)=("
                        << prev_gate << "," << prev_token << "," << prev_slot
                        << "), current=(" << gate << "," << token << "," << slot << ")";
                    *error = oss.str();
                }
                return false;
            }

            prev_gate = gate;
            prev_token = token;
            prev_slot = slot;
        }
    }

    return true;
}

static inline bool mdc_check_dropped(
    const MdcRunSpec& run,
    const MdcExpected& expected,
    const MdcHostOutputsView& got,
    std::string* error) {
    const size_t total_routes = (size_t)run.N * (size_t)run.K;

    for (size_t i = 0; i < total_routes; ++i) {
        const uint8_t got_v = got.dropped[i] ? 1 : 0;
        const uint8_t exp_v = expected.dropped[i] ? 1 : 0;

        if (got_v != exp_v) {
            if (error) {
                const int token = static_cast<int>(i / (size_t)run.K);
                const int slot = static_cast<int>(i % (size_t)run.K);
                std::ostringstream oss;
                oss << "dropped mismatch at token=" << token
                    << ", slot=" << slot
                    << ": got " << static_cast<int>(got_v)
                    << ", expected " << static_cast<int>(exp_v);
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool mdc_check_y(
    const MdcRunSpec& run,
    const MdcExpected& expected,
    const MdcHostOutputsView& got,
    std::string* error) {
    const size_t total = (size_t)run.N * (size_t)run.D;

    for (size_t i = 0; i < total; ++i) {
        if (got.y[i] != expected.y[i]) {
            if (error) {
                const int token = static_cast<int>(i / (size_t)run.D);
                const int d = static_cast<int>(i % (size_t)run.D);
                std::ostringstream oss;
                oss << "y mismatch at token=" << token
                    << ", d=" << d
                    << ": got " << got.y[i]
                    << ", expected " << expected.y[i];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool mdc_check_all_outputs(
    const MdcRunSpec& run,
    const MdcExpected& expected,
    const MdcHostOutputsView& got,
    std::string* error) {
    if (!mdc_check_counts_offsets(run, expected, got, error)) return false;
    if (!mdc_check_packed_intermediates(run, expected, got, error)) return false;
    if (!mdc_check_dropped(run, expected, got, error)) return false;
    if (!mdc_check_y(run, expected, got, error)) return false;
    return true;
}

/*
INTERMEDIATE CHECKS REQUIRED BY GRADER

The grader should copy back and verify:

  1. counts[E]
     - exact equality against CPU oracle.
     - 0 <= counts[e] <= cap.

  2. offsets[E + 1]
     - exact equality against CPU oracle.
     - offsets[0] == 0.
     - offsets[e + 1] - offsets[e] == counts[e].
     - offsets[E] == total kept routes.

  3. packed_token[0:offsets[E]]
     - exact equality against CPU oracle.
     - token ids must be in [0, N).

  4. packed_slot[0:offsets[E]]
     - exact equality against CPU oracle.
     - slot ids must be in [0, K).

  5. packed_gate[0:offsets[E]]
     - exact equality against CPU oracle.
     - gate values must be in [MDC_GATE_MIN, MDC_GATE_MAX].

  6. Packed route order per expert:
     - descending gate,
     - then ascending token,
     - then ascending slot.

  7. dropped[N, K]
     - exact equality against CPU oracle.
     - value is 1 only for assigned routes dropped due to capacity.

  8. y[N, D]
     - exact int64 two's-complement equality against CPU oracle.

The grader should additionally enforce:
  - Sentinels around every output allocation.
  - Input immutability hash before/after solution_run.
  - Determinism replay with the same case repeated at least twice.
  - Held-out seed-shifted distributions from the declared shape family.
*/

#endif  // MOE_DISPATCH_COMBINE_ORACLE_HPP_
