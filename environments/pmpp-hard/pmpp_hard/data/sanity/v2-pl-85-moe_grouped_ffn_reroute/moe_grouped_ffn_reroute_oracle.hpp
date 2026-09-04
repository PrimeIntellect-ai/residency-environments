// ============================================================================
// file: moe_grouped_ffn_reroute_oracle.hpp
// Independent CPU oracle + output check helpers.
// ============================================================================

#ifndef MOE_GROUPED_FFN_REROUTE_ORACLE_HPP_
#define MOE_GROUPED_FFN_REROUTE_ORACLE_HPP_

#include "moe_grouped_ffn_reroute_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

struct MgfHostInputsView {
    const int8_t* x;
    const int8_t* wr;
    const int8_t* w1;
    const int8_t* w2;
};

struct MgfHostOutputsView {
    const int32_t* logits;
    const int32_t* counts;
    const int32_t* offsets;
    const int32_t* packed_token;
    const int32_t* packed_slot;
    const int32_t* packed_gate;
    const uint8_t* packed_phase;
    const int16_t* route_expert;
    const uint8_t* route_status;
    const int32_t* packed_y;
    const int64_t* y;
    const uint64_t* y_checksum;
};

struct MgfExpected {
    std::vector<int32_t> logits;
    std::vector<int32_t> counts;
    std::vector<int32_t> offsets;
    std::vector<int32_t> packed_token;
    std::vector<int32_t> packed_slot;
    std::vector<int32_t> packed_gate;
    std::vector<uint8_t> packed_phase;
    std::vector<int16_t> route_expert;
    std::vector<uint8_t> route_status;
    std::vector<int32_t> packed_y;
    std::vector<int64_t> y;
    uint64_t y_checksum;
};

struct MgfOracleRoute {
    int32_t gate;
    int32_t token;
    int32_t slot;
};

static inline bool mgf_oracle_route_less(
    const MgfOracleRoute& a,
    const MgfOracleRoute& b) {
    if (a.gate != b.gate) return a.gate > b.gate;
    if (a.token != b.token) return a.token < b.token;
    return a.slot < b.slot;
}

static inline uint64_t mgf_oracle_fnv_byte(uint64_t h, uint8_t b) {
    return (h ^ static_cast<uint64_t>(b)) * MGF_FNV_PRIME;
}

static inline void mgf_cpu_oracle(
    const MgfRunSpec& run,
    const MgfHostInputsView& in,
    MgfExpected* expected) {
    const int N = run.N;
    const int D = run.D;
    const int H = run.H;
    const int E = run.E;
    const int G = run.G;
    const int S = E / G;
    const int g_sel = run.g_sel;
    const int K = run.K;
    const int cap = run.cap;
    const int qshift = run.qshift;
    const int n_sel_experts = g_sel * S;

    expected->logits.assign((size_t)N * (size_t)E, 0);
    expected->counts.assign((size_t)E, 0);
    expected->offsets.assign((size_t)E + 1, 0);
    expected->route_expert.assign((size_t)N * (size_t)K, -1);
    expected->route_status.assign((size_t)N * (size_t)K, 0);
    expected->y.assign((size_t)N * (size_t)D, 0);

    // STEP 1: router logits.
    for (int t = 0; t < N; ++t) {
        const int8_t* xr = in.x + (size_t)t * (size_t)D;
        for (int e = 0; e < E; ++e) {
            const int8_t* wrr = in.wr + (size_t)e * (size_t)D;
            int32_t acc = 0;
            for (int d = 0; d < D; ++d) {
                acc += (int32_t)xr[d] * (int32_t)wrr[d];
            }
            expected->logits[(size_t)t * (size_t)E + (size_t)e] = acc;
        }
    }

    // STEP 2 + 3: per-token candidate lists, primary and backup experts.
    // cand[t*n_sel_experts + j] holds expert ids of the selected groups
    // ranked by (logit desc, expert id asc).
    std::vector<int16_t> cand((size_t)N * (size_t)n_sel_experts, -1);

    for (int t = 0; t < N; ++t) {
        const int32_t* lrow = expected->logits.data() + (size_t)t * (size_t)E;

        // Group scores: max logit inside each contiguous group.
        std::vector<int32_t> gscore((size_t)G);
        for (int g = 0; g < G; ++g) {
            int32_t best = lrow[(size_t)g * (size_t)S];
            for (int j = 1; j < S; ++j) {
                const int32_t v = lrow[(size_t)g * (size_t)S + (size_t)j];
                if (v > best) best = v;
            }
            gscore[(size_t)g] = best;
        }

        // Top g_sel groups by (score desc, id asc).
        std::vector<int> gids((size_t)G);
        for (int g = 0; g < G; ++g) gids[(size_t)g] = g;
        std::sort(gids.begin(), gids.end(), [&](int a, int b) {
            if (gscore[(size_t)a] != gscore[(size_t)b]) {
                return gscore[(size_t)a] > gscore[(size_t)b];
            }
            return a < b;
        });

        // Candidate experts from the selected groups, ranked.
        std::vector<int> ce;
        ce.reserve((size_t)n_sel_experts);
        for (int i = 0; i < g_sel; ++i) {
            const int g = gids[(size_t)i];
            for (int j = 0; j < S; ++j) {
                ce.push_back(g * S + j);
            }
        }
        std::sort(ce.begin(), ce.end(), [&](int a, int b) {
            if (lrow[(size_t)a] != lrow[(size_t)b]) {
                return lrow[(size_t)a] > lrow[(size_t)b];
            }
            return a < b;
        });

        for (int j = 0; j < n_sel_experts; ++j) {
            cand[(size_t)t * (size_t)n_sel_experts + (size_t)j] =
                static_cast<int16_t>(ce[(size_t)j]);
        }
    }

    // STEP 4: phase-1 dispatch.
    std::vector<std::vector<MgfOracleRoute>> by_e1((size_t)E);
    for (int t = 0; t < N; ++t) {
        for (int k = 0; k < K; ++k) {
            const int e =
                cand[(size_t)t * (size_t)n_sel_experts + (size_t)k];
            const int32_t gate =
                expected->logits[(size_t)t * (size_t)E + (size_t)e];
            by_e1[(size_t)e].push_back(MgfOracleRoute{gate, t, k});
        }
    }

    std::vector<int> kept1((size_t)E, 0);
    std::vector<std::vector<MgfOracleRoute>> by_e2((size_t)E);

    for (int e = 0; e < E; ++e) {
        std::vector<MgfOracleRoute>& routes = by_e1[(size_t)e];
        std::sort(routes.begin(), routes.end(), mgf_oracle_route_less);
        kept1[(size_t)e] =
            std::min(static_cast<int>(routes.size()), cap);

        for (size_t i = (size_t)kept1[(size_t)e]; i < routes.size(); ++i) {
            const MgfOracleRoute& r = routes[i];
            const int backup_pos = K + r.slot;
            const int b = backup_pos < n_sel_experts
                ? cand[(size_t)r.token * (size_t)n_sel_experts +
                       (size_t)backup_pos]
                : -1;
            const size_t rid = (size_t)r.token * (size_t)K + (size_t)r.slot;
            if (b < 0) {
                expected->route_expert[rid] = -1;
                expected->route_status[rid] = MGF_RS_DROP_NO_BACKUP;
            } else {
                const int32_t gate2 =
                    expected->logits[(size_t)r.token * (size_t)E + (size_t)b];
                by_e2[(size_t)b].push_back(
                    MgfOracleRoute{gate2, r.token, r.slot});
            }
        }
    }

    // STEP 5: phase-2 dispatch.
    std::vector<int> kept2((size_t)E, 0);
    for (int e = 0; e < E; ++e) {
        std::vector<MgfOracleRoute>& routes = by_e2[(size_t)e];
        std::sort(routes.begin(), routes.end(), mgf_oracle_route_less);
        const int rcap = cap - kept1[(size_t)e];
        kept2[(size_t)e] =
            std::min(static_cast<int>(routes.size()), rcap);

        for (size_t i = (size_t)kept2[(size_t)e]; i < routes.size(); ++i) {
            const MgfOracleRoute& r = routes[i];
            const size_t rid = (size_t)r.token * (size_t)K + (size_t)r.slot;
            expected->route_expert[rid] = -1;
            expected->route_status[rid] = MGF_RS_DROP_OVERFLOW;
        }
    }

    // STEP 6 + 7: packing.
    int32_t total = 0;
    expected->offsets[0] = 0;
    for (int e = 0; e < E; ++e) {
        expected->counts[(size_t)e] = kept1[(size_t)e] + kept2[(size_t)e];
        total += expected->counts[(size_t)e];
        expected->offsets[(size_t)e + 1] = total;
    }

    expected->packed_token.assign((size_t)total, 0);
    expected->packed_slot.assign((size_t)total, 0);
    expected->packed_gate.assign((size_t)total, 0);
    expected->packed_phase.assign((size_t)total, 0);
    expected->packed_y.assign((size_t)total * (size_t)D, 0);

    std::vector<int16_t> packed_expert((size_t)total, 0);

    for (int e = 0; e < E; ++e) {
        int pos = expected->offsets[(size_t)e];
        for (int i = 0; i < kept1[(size_t)e]; ++i, ++pos) {
            const MgfOracleRoute& r = by_e1[(size_t)e][(size_t)i];
            expected->packed_token[(size_t)pos] = r.token;
            expected->packed_slot[(size_t)pos] = r.slot;
            expected->packed_gate[(size_t)pos] = r.gate;
            expected->packed_phase[(size_t)pos] = 0;
            packed_expert[(size_t)pos] = static_cast<int16_t>(e);
            const size_t rid = (size_t)r.token * (size_t)K + (size_t)r.slot;
            expected->route_expert[rid] = static_cast<int16_t>(e);
            expected->route_status[rid] = MGF_RS_KEPT_PRIMARY;
        }
        for (int i = 0; i < kept2[(size_t)e]; ++i, ++pos) {
            const MgfOracleRoute& r = by_e2[(size_t)e][(size_t)i];
            expected->packed_token[(size_t)pos] = r.token;
            expected->packed_slot[(size_t)pos] = r.slot;
            expected->packed_gate[(size_t)pos] = r.gate;
            expected->packed_phase[(size_t)pos] = 1;
            packed_expert[(size_t)pos] = static_cast<int16_t>(e);
            const size_t rid = (size_t)r.token * (size_t)K + (size_t)r.slot;
            expected->route_expert[rid] = static_cast<int16_t>(e);
            expected->route_status[rid] = MGF_RS_KEPT_REROUTED;
        }
    }

    // STEP 8 + 9: expert FFN + combine.
    std::vector<int32_t> h((size_t)H, 0);
    for (int pos = 0; pos < total; ++pos) {
        const int e = packed_expert[(size_t)pos];
        const int t = expected->packed_token[(size_t)pos];
        const int64_t gate =
            static_cast<int64_t>(expected->packed_gate[(size_t)pos]);
        const int8_t* xr = in.x + (size_t)t * (size_t)D;
        const int8_t* w1e = in.w1 + (size_t)e * (size_t)H * (size_t)D;
        const int8_t* w2e = in.w2 + (size_t)e * (size_t)D * (size_t)H;

        for (int j = 0; j < H; ++j) {
            const int8_t* w1r = w1e + (size_t)j * (size_t)D;
            int32_t acc = 0;
            for (int d = 0; d < D; ++d) {
                acc += (int32_t)w1r[d] * (int32_t)xr[d];
            }
            if (acc < 0) acc = 0;
            acc >>= qshift;
            h[(size_t)j] = acc < 127 ? acc : 127;
        }

        for (int d = 0; d < D; ++d) {
            const int8_t* w2r = w2e + (size_t)d * (size_t)H;
            int32_t acc = 0;
            for (int j = 0; j < H; ++j) {
                acc += (int32_t)w2r[j] * h[(size_t)j];
            }
            expected->packed_y[(size_t)pos * (size_t)D + (size_t)d] = acc;

            const size_t y_idx = (size_t)t * (size_t)D + (size_t)d;
            const uint64_t prod = static_cast<uint64_t>(
                gate * static_cast<int64_t>(acc));
            expected->y[y_idx] = static_cast<int64_t>(
                static_cast<uint64_t>(expected->y[y_idx]) + prod);
        }
    }

    // STEP 10: two-level checksum over y.
    const int C = mgf_ceil_div_int(N, MGF_CSUM_ROWS);
    std::vector<uint64_t> digests((size_t)C, 0);
    for (int c = 0; c < C; ++c) {
        const int row_begin = c * MGF_CSUM_ROWS;
        int row_end = row_begin + MGF_CSUM_ROWS;
        if (row_end > N) row_end = N;

        uint64_t hsh = MGF_FNV_BASIS;
        for (int t = row_begin; t < row_end; ++t) {
            for (int d = 0; d < D; ++d) {
                const uint64_t v = static_cast<uint64_t>(
                    expected->y[(size_t)t * (size_t)D + (size_t)d]);
                for (int b = 0; b < 8; ++b) {
                    hsh = mgf_oracle_fnv_byte(
                        hsh, static_cast<uint8_t>((v >> (8 * b)) & 0xFF));
                }
            }
        }
        digests[(size_t)c] = hsh;
    }

    uint64_t root = MGF_FNV_BASIS;
    for (int c = 0; c < C; ++c) {
        const uint64_t v = digests[(size_t)c];
        for (int b = 0; b < 8; ++b) {
            root = mgf_oracle_fnv_byte(
                root, static_cast<uint8_t>((v >> (8 * b)) & 0xFF));
        }
    }
    expected->y_checksum = root;
}

// ---------------------------------------------------------------------------
// Checkers.
// ---------------------------------------------------------------------------

template <typename T>
static inline bool mgf_check_array(
    const char* name,
    const T* got,
    const std::vector<T>& expected,
    size_t count,
    std::string* error) {
    for (size_t i = 0; i < count; ++i) {
        if (got[i] != expected[i]) {
            if (error) {
                std::ostringstream oss;
                oss << name << " mismatch at index " << i
                    << ": got " << (long long)got[i]
                    << ", expected " << (long long)expected[i];
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

static inline bool mgf_check_all_outputs(
    const MgfRunSpec& run,
    const MgfExpected& expected,
    const MgfHostOutputsView& got,
    std::string* error) {
    const int N = run.N;
    const int D = run.D;
    const int E = run.E;
    const int K = run.K;

    if (!mgf_check_array("logits", got.logits, expected.logits,
                         (size_t)N * (size_t)E, error)) return false;
    if (!mgf_check_array("counts", got.counts, expected.counts,
                         (size_t)E, error)) return false;

    for (int e = 0; e < E; ++e) {
        if (got.counts[e] < 0 || got.counts[e] > run.cap) {
            if (error) {
                std::ostringstream oss;
                oss << "counts out of range at e=" << e
                    << ": got " << got.counts[e] << ", cap " << run.cap;
                *error = oss.str();
            }
            return false;
        }
    }

    if (!mgf_check_array("offsets", got.offsets, expected.offsets,
                         (size_t)E + 1, error)) return false;
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

    const size_t total = (size_t)expected.offsets[(size_t)E];

    if (!mgf_check_array("packed_token", got.packed_token,
                         expected.packed_token, total, error)) return false;
    if (!mgf_check_array("packed_slot", got.packed_slot,
                         expected.packed_slot, total, error)) return false;
    if (!mgf_check_array("packed_gate", got.packed_gate,
                         expected.packed_gate, total, error)) return false;
    if (!mgf_check_array("packed_phase", got.packed_phase,
                         expected.packed_phase, total, error)) return false;
    if (!mgf_check_array("route_expert", got.route_expert,
                         expected.route_expert,
                         (size_t)N * (size_t)K, error)) return false;
    if (!mgf_check_array("route_status", got.route_status,
                         expected.route_status,
                         (size_t)N * (size_t)K, error)) return false;
    if (!mgf_check_array("packed_y", got.packed_y, expected.packed_y,
                         total * (size_t)D, error)) return false;
    if (!mgf_check_array("y", got.y, expected.y,
                         (size_t)N * (size_t)D, error)) return false;

    if (got.y_checksum[0] != expected.y_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "y_checksum mismatch: got " << got.y_checksum[0]
                << ", expected " << expected.y_checksum;
            *error = oss.str();
        }
        return false;
    }

    return true;
}

/*
INTERMEDIATE CHECKS REQUIRED BY GRADER

The grader should copy back and verify (all exact):
  1. logits[N, E]                 int32 equality.
  2. counts[E] / offsets[E + 1]   equality + prefix invariants + counts <= cap.
  3. packed_token/slot/gate/phase[0:offsets[E]]  equality (expert-major,
     phase-1 slice before phase-2 slice, documented rank orders).
  4. route_expert / route_status [N, K]          equality.
  5. packed_y[0:offsets[E]*D]     int32 equality.
  6. y[N, D]                      int64 equality.
  7. y_checksum[0]                two-level FNV equality.

The grader should additionally enforce:
  - Sentinels around every output allocation.
  - Input immutability before/after solution_run.
  - Determinism replay (same case twice, byte-identical specified regions).
  - Held-out seed-shifted cases from the declared shape family.
*/

#endif  // MOE_GROUPED_FFN_REROUTE_ORACLE_HPP_
