// ============================================================================
// file: taskgraph_wavefront_gemm_oracle.hpp
// Independent CPU oracle (stateful) + output check helpers.
// ============================================================================

#ifndef TASKGRAPH_WAVEFRONT_GEMM_ORACLE_HPP_
#define TASKGRAPH_WAVEFRONT_GEMM_ORACLE_HPP_

#include "taskgraph_wavefront_gemm_common.h"

#include <stdint.h>
#include <stddef.h>

#include <sstream>
#include <string>
#include <vector>

struct TwgExpected {
    int32_t round_out;
    std::vector<uint32_t> border_row;  // [C*32]
    std::vector<uint32_t> border_col;  // [R*32]
    uint64_t state_checksum;
    std::vector<uint32_t> acc_dump;    // [R*32 * C*32] (always filled)
};

static inline uint64_t twg_oracle_fnv_byte(uint64_t h, uint8_t b) {
    return (h ^ static_cast<uint64_t>(b)) * TWG_FNV_PRIME;
}

// Stateful CPU oracle mirroring the persistent device state.
struct TwgOracle {
    int R = 0;
    int C = 0;
    int round = 0;
    std::vector<uint32_t> acc;  // [R*32 * C*32] row-major global layout

    void reset(int R_, int C_) {
        R = R_;
        C = C_;
        round = 0;
        acc.assign((size_t)R * 32 * (size_t)C * 32, 0u);
    }

    // Executes one round and fills the expected outputs.
    void run_round(
        const TwgRunSpec& run,
        const int8_t* U,
        const int8_t* V,
        TwgExpected* expected) {
        const int K = run.K;
        const uint32_t P1 = (uint32_t)run.P1;
        const uint32_t P2 = (uint32_t)run.P2;
        const int W = C * 32;  // global row width

        // Wavefront order: any order with (i-1,j),(i,j-1) first. Row-major
        // works because left/top are always earlier.
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) {
                uint32_t rowvec[32];
                uint32_t colvec[32];
                uint32_t sum_row = 0;
                uint32_t sum_col = 0;

                for (int b = 0; b < 32; ++b) {
                    rowvec[b] = i > 0
                        ? acc[(size_t)((i - 1) * 32 + 31) * W + (size_t)j * 32 + b]
                        : 0u;
                    sum_row += rowvec[b];
                }
                for (int a = 0; a < 32; ++a) {
                    colvec[a] = j > 0
                        ? acc[(size_t)(i * 32 + a) * W + (size_t)(j - 1) * 32 + 31]
                        : 0u;
                    sum_col += colvec[a];
                }

                const uint32_t s = TWG_SALT_ROW * sum_row +
                                   TWG_SALT_COL * sum_col +
                                   TWG_SALT_ROUND * (uint32_t)round +
                                   ((uint32_t)i << 16) + (uint32_t)j;
                uint8_t sb[4];
                sb[0] = (uint8_t)(s & 0xFF);
                sb[1] = (uint8_t)((s >> 8) & 0xFF);
                sb[2] = (uint8_t)((s >> 16) & 0xFF);
                sb[3] = (uint8_t)((s >> 24) & 0xFF);

                for (int a = 0; a < 32; ++a) {
                    const int8_t* urow = U + (size_t)(i * 32 + a) * K;
                    for (int b = 0; b < 32; ++b) {
                        const int8_t* vrow = V + (size_t)(j * 32 + b) * K;
                        int32_t g = 0;
                        for (int k = 0; k < K; ++k) {
                            const int8_t mixed = (int8_t)(
                                (uint8_t)urow[k] ^ sb[k & 3]);
                            g += (int32_t)mixed * (int32_t)vrow[k];
                        }
                        uint32_t* cell =
                            &acc[(size_t)(i * 32 + a) * W + (size_t)j * 32 + b];
                        *cell = *cell + (uint32_t)g + P1 * rowvec[b] +
                                P2 * colvec[a];
                    }
                }
            }
        }

        ++round;

        expected->round_out = round;
        expected->border_row.assign((size_t)C * 32, 0u);
        expected->border_col.assign((size_t)R * 32, 0u);
        for (int x = 0; x < C * 32; ++x) {
            expected->border_row[(size_t)x] =
                acc[(size_t)(R * 32 - 1) * W + x];
        }
        for (int y = 0; y < R * 32; ++y) {
            expected->border_col[(size_t)y] =
                acc[(size_t)y * W + (C * 32 - 1)];
        }

        // Two-level checksum.
        std::vector<uint64_t> digests((size_t)R * C, 0ULL);
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) {
                uint64_t h = TWG_FNV_BASIS;
                for (int a = 0; a < 32; ++a) {
                    for (int b = 0; b < 32; ++b) {
                        const uint32_t v =
                            acc[(size_t)(i * 32 + a) * W + (size_t)j * 32 + b];
                        for (int m = 0; m < 4; ++m) {
                            h = twg_oracle_fnv_byte(
                                h, (uint8_t)((v >> (8 * m)) & 0xFF));
                        }
                    }
                }
                digests[(size_t)i * C + j] = h;
            }
        }
        uint64_t root = TWG_FNV_BASIS;
        for (size_t d = 0; d < digests.size(); ++d) {
            const uint64_t v = digests[d];
            for (int m = 0; m < 8; ++m) {
                root = twg_oracle_fnv_byte(
                    root, (uint8_t)((v >> (8 * m)) & 0xFF));
            }
        }
        expected->state_checksum = root;

        expected->acc_dump = acc;
    }
};

struct TwgHostOutputsView {
    const int32_t* round_out;
    const uint32_t* border_row;
    const uint32_t* border_col;
    const uint64_t* state_checksum;
    const uint32_t* acc_dump;  // may be null when dump == 0
};

static inline bool twg_check_outputs(
    const TwgRunSpec& run,
    const TwgExpected& expected,
    const TwgHostOutputsView& got,
    std::string* error) {
    if (got.round_out[0] != expected.round_out) {
        if (error) {
            std::ostringstream oss;
            oss << "round_out mismatch: got " << got.round_out[0]
                << ", expected " << expected.round_out;
            *error = oss.str();
        }
        return false;
    }

    for (int x = 0; x < run.C * 32; ++x) {
        if (got.border_row[x] != expected.border_row[(size_t)x]) {
            if (error) {
                std::ostringstream oss;
                oss << "border_row mismatch at x=" << x
                    << ": got " << got.border_row[x]
                    << ", expected " << expected.border_row[(size_t)x];
                *error = oss.str();
            }
            return false;
        }
    }
    for (int y = 0; y < run.R * 32; ++y) {
        if (got.border_col[y] != expected.border_col[(size_t)y]) {
            if (error) {
                std::ostringstream oss;
                oss << "border_col mismatch at y=" << y
                    << ": got " << got.border_col[y]
                    << ", expected " << expected.border_col[(size_t)y];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.state_checksum[0] != expected.state_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "state_checksum mismatch: got " << got.state_checksum[0]
                << ", expected " << expected.state_checksum;
            *error = oss.str();
        }
        return false;
    }

    if (run.dump == 1 && got.acc_dump) {
        const size_t total = (size_t)run.R * 32 * (size_t)run.C * 32;
        for (size_t idx = 0; idx < total; ++idx) {
            if (got.acc_dump[idx] != expected.acc_dump[idx]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "acc_dump mismatch at flat index " << idx
                        << ": got " << got.acc_dump[idx]
                        << ", expected " << expected.acc_dump[idx];
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

/*
INTERMEDIATE CHECKS REQUIRED BY GRADER

After every solution_run the grader should verify (all exact):
  1. round_out          == oracle round counter (statefulness probe).
  2. border_row/col     == oracle borders (cheap per-round probes).
  3. state_checksum     == oracle two-level FNV over the FULL state
                           (any error in any tile of any earlier round
                            cascades through the salt into every later
                            checksum).
  4. acc_dump           == full oracle state on dump rounds.

The grader should additionally enforce:
  - Sentinels around every output allocation; the acc_dump buffer must be
    untouched on dump == 0 rounds.
  - Input immutability before/after solution_run.
  - Determinism replay: identical reset+round sequences twice must produce
    identical bytes.
  - Multi-round chains (state persistence) and resets mid-case.
*/

#endif  // TASKGRAPH_WAVEFRONT_GEMM_ORACLE_HPP_
