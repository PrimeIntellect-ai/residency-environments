// file: sat_segscan_select_oracle.hpp
//
// Independent CPU oracle for sat_segscan_select. A direct transliteration of
// the normative sequential loop -- no parallel decomposition, no function
// composition; the contract's single source of truth executed literally.

#ifndef SAT_SEGSCAN_SELECT_ORACLE_HPP_
#define SAT_SEGSCAN_SELECT_ORACLE_HPP_

#include "sat_segscan_select_common.h"

#include <stdint.h>
#include <stddef.h>

#include <sstream>
#include <string>
#include <vector>

struct SssHostInputsView {
    const int32_t* v;
    const uint32_t* flags;
};

struct SssHostOutputsView {
    const int32_t* y;
    const uint32_t* sat_bits;
    const int32_t* sel_idx;
    const int32_t* sel_count;
    const int32_t* seg_last;
};

struct SssExpected {
    std::vector<int32_t> y;
    std::vector<uint32_t> sat_bits;
    std::vector<int32_t> sel_idx;
    int32_t sel_count = 0;
    std::vector<int32_t> seg_last;
};

static inline int sss_oracle_flag(const uint32_t* flags, int i) {
    return (int)((flags[(size_t)(i >> 5)] >> (i & 31)) & 1u);
}

static inline void sss_cpu_oracle(
    const SssRunSpec& run,
    const SssHostInputsView& in,
    SssExpected* expected) {
    const int N = run.N;
    const int Nw = sss_Nw(N);
    const int64_t LO = run.lo;
    const int64_t HI = run.hi;

    expected->y.assign((size_t)N, 0);
    expected->sat_bits.assign((size_t)Nw, 0u);
    expected->sel_idx.clear();
    expected->seg_last.clear();
    expected->sel_count = 0;

    int64_t p = 0;

    for (int i = 0; i < N; ++i) {
        if (sss_oracle_flag(in.flags, i)) {
            p = 0;
            expected->seg_last.push_back(0);  // placeholder, overwritten below
        }
        int64_t t = p + (int64_t)in.v[(size_t)i];
        if (t < LO) t = LO;
        if (t > HI) t = HI;
        p = t;
        expected->y[(size_t)i] = (int32_t)p;

        const bool sat = (p == LO || p == HI);
        if (sat) {
            expected->sat_bits[(size_t)(i >> 5)] |= (1u << (i & 31));
            expected->sel_idx.push_back(i);
        }
        expected->seg_last.back() = (int32_t)p;
    }

    expected->sel_count = (int32_t)expected->sel_idx.size();
}

// ---------------------------------------------------------------------------
// Checkers (exact; no tolerances).
// ---------------------------------------------------------------------------

static inline bool sss_check_all_outputs(
    const SssRunSpec& run,
    const SssExpected& expected,
    const SssHostOutputsView& got,
    std::string* error) {
    const int N = run.N;
    const int Nw = sss_Nw(N);

    for (int i = 0; i < N; ++i) {
        if (got.y[i] != expected.y[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "y mismatch at i=" << i
                    << ": got " << got.y[i]
                    << ", expected " << expected.y[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int w = 0; w < Nw; ++w) {
        if (got.sat_bits[w] != expected.sat_bits[(size_t)w]) {
            if (error) {
                std::ostringstream oss;
                oss << "sat_bits mismatch at word " << w
                    << ": got 0x" << std::hex << got.sat_bits[w]
                    << ", expected 0x" << expected.sat_bits[(size_t)w];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.sel_count[0] != expected.sel_count) {
        if (error) {
            std::ostringstream oss;
            oss << "sel_count mismatch: got " << got.sel_count[0]
                << ", expected " << expected.sel_count;
            *error = oss.str();
        }
        return false;
    }

    for (int j = 0; j < expected.sel_count; ++j) {
        if (got.sel_idx[j] != expected.sel_idx[(size_t)j]) {
            if (error) {
                std::ostringstream oss;
                oss << "sel_idx mismatch at j=" << j
                    << ": got " << got.sel_idx[j]
                    << ", expected " << expected.sel_idx[(size_t)j];
                *error = oss.str();
            }
            return false;
        }
    }

    for (size_t s = 0; s < expected.seg_last.size(); ++s) {
        if (got.seg_last[s] != expected.seg_last[s]) {
            if (error) {
                std::ostringstream oss;
                oss << "seg_last mismatch at segment " << s
                    << ": got " << got.seg_last[s]
                    << ", expected " << expected.seg_last[s];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

/*
GRADER CHECKS (all exact):
  - y         exact int32, dense
  - sat_bits  byte-exact over the whole packed buffer (padding bits 0)
  - sel_idx   exact int32 for the first sel_count entries
  - sel_count exact int32
  - seg_last  exact int32, one per segment

Harness should also enforce:
  - output guard sentinels
  - input immutability
  - held-out seeds
  - all nine distributions, ragged N, single-segment and all-segment runs
*/

#endif  // SAT_SEGSCAN_SELECT_ORACLE_HPP_
