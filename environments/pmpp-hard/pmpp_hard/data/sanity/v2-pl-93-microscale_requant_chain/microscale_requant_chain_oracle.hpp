// file: microscale_requant_chain_oracle.hpp
//
// Independent CPU oracle for microscale_requant_chain. Every rounding rule
// of the contract is reproduced in scalar host code using ONLY bit-level
// integer constructions: the E4M3 RNE (including subnormals, binade
// carries, the 2^-10 underflow tie and 448 saturation), the strict
// saturation count, the msb-based second scale, the LUT boundary table
// with ties-to-larger-index, the fixed-point squared error, and the
// word-folded digest. No floating-point arithmetic is used anywhere in
// the decision paths.

#ifndef MICROSCALE_REQUANT_CHAIN_ORACLE_HPP_
#define MICROSCALE_REQUANT_CHAIN_ORACLE_HPP_

#include "microscale_requant_chain_common.h"

#include <stdint.h>
#include <stddef.h>

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

struct MrqHostInputsView {
    const float* x;
};

struct MrqHostOutputsView {
    const uint8_t* e4m3_codes;
    const uint8_t* q4_packed;
    const uint8_t* sf1;
    const uint8_t* sf2;
    const int64_t* row_err;
    const uint64_t* row_digest;
    const int32_t* sat1_count;
};

struct MrqExpected {
    std::vector<uint8_t> e4m3_codes;
    std::vector<uint8_t> q4_packed;
    std::vector<uint8_t> sf1;
    std::vector<uint8_t> sf2;
    std::vector<int64_t> row_err;
    std::vector<uint64_t> row_digest;
    std::vector<int32_t> sat1_count;
};

static inline uint32_t mrq_oracle_f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

static inline int mrq_oracle_msb32(uint32_t v) {
    int i = 0;
    while (v >> (i + 1)) ++i;
    return i;  // v != 0
}

// Decompose |x| into (mant, ex) with |x| = mant * 2^ex, mant integer != 0.
// Returns 0 for +-0.0.
static inline int mrq_oracle_decompose(
    uint32_t xbits, uint32_t* mant, int* ex) {
    const uint32_t E = (xbits >> 23) & 0xffu;
    const uint32_t Mn = xbits & 0x7fffffu;
    if (E == 0) {
        if (Mn == 0) return 0;
        *mant = Mn;
        *ex = -149;
        return 1;
    }
    *mant = 0x800000u | Mn;
    *ex = (int)E - 127 - 23;
    return 1;
}

// floor_log2(amax) from bits (amax > 0).
static inline int mrq_oracle_floor_log2(uint32_t abits) {
    const int E = (int)((abits >> 23) & 0xffu);
    const uint32_t Mn = abits & 0x7fffffu;
    if (E > 0) return E - 127;
    return mrq_oracle_msb32(Mn) - 149;
}

// Stage 1: quantize |x| (as mant*2^ex) scaled by 2^-sexp1 to E4M3.
// Outputs the unsigned code (E<<3 | m) and the integer magnitude M.
// Also reports strict saturation (u > 448).
static inline void mrq_oracle_e4m3(
    uint32_t mant, int ex, int sexp1,
    int* code_out, int* M_out, int* sat_out) {
    const int exp = ex - sexp1;             // u = mant * 2^exp
    const int msb = mrq_oracle_msb32(mant);
    const int k = msb + exp;                // floor_log2(u)

    // strict u > 448 test (448 = 7 * 2^6)
    int sat = 0;
    if (k > 8) {
        sat = 1;
    } else if (k == 8) {
        const int sh2 = exp - 5;            // u/32 = mant * 2^sh2 vs 14
        if (sh2 >= 0) {
            sat = ((unsigned long long)mant << sh2) > 14ull;
        } else {
            sat = (unsigned long long)mant > (14ull << (-sh2));
        }
    }
    *sat_out = sat;

    const int step_exp = (k >= -6) ? (k - 3) : -9;
    const int sh = exp - step_exp;          // n = mant * 2^sh

    unsigned long long n;
    if (sh >= 0) {
        n = (unsigned long long)mant << sh;
    } else {
        const int s = -sh;
        if (s >= 25) {
            n = 0;  // mant < 2^24 => u < half of the step
        } else {
            const unsigned long long n0 = (unsigned long long)mant >> s;
            const unsigned long long rem =
                (unsigned long long)mant & ((1ull << s) - 1ull);
            const unsigned long long half = 1ull << (s - 1);
            if (rem > half) n = n0 + 1;
            else if (rem == half) n = n0 + (n0 & 1ull);
            else n = n0;
        }
    }

    int kk = k;
    if (k >= -6) {
        if (n == 16) { n = 8; kk = k + 1; }
        if (kk > 8 || (kk == 8 && n > 14)) {   // saturate to 448
            *code_out = (15 << 3) | 6;
            *M_out = 229376;
            return;
        }
        *code_out = ((kk + 7) << 3) | (int)(n - 8);
        *M_out = (int)(n << (kk + 6));
        return;
    }
    // denormal region
    if (n >= 8) {                             // rounds up to 2^-6
        *code_out = (1 << 3) | 0;
        *M_out = 8;
        return;
    }
    *code_out = (int)n;                       // E=0, m=n (n may be 0)
    *M_out = (int)n;
}

static inline uint64_t mrq_oracle_fnv_word(uint64_t h, uint64_t w) {
    h ^= w;
    h *= MRQ_FNV_PRIME;
    return h;
}

static const int MRQ_ORACLE_BP[7] = { 1, 3, 5, 8, 13, 20, 28 };  // 16*B
static const int MRQ_ORACLE_LP[8] = { 0, 1, 2, 3, 5, 8, 12, 16 };  // 8*L

static inline void mrq_cpu_oracle(
    const MrqRunSpec& run,
    const MrqHostInputsView& in,
    MrqExpected* expected) {
    const int R = run.R;
    const int C = run.C;
    const int S = mrq_S(C);
    const int Kb = mrq_Kb(C);
    const int Lb = mrq_stream_bytes(C);
    const int W = (Lb + 7) / 8;

    expected->e4m3_codes.assign((size_t)R * (size_t)C, 0);
    expected->q4_packed.assign((size_t)R * (size_t)Kb, 0);
    expected->sf1.assign((size_t)R * (size_t)S, 0);
    expected->sf2.assign((size_t)R * (size_t)S, 0);
    expected->row_err.assign((size_t)R, 0);
    expected->row_digest.assign((size_t)R, 0);
    expected->sat1_count.assign((size_t)R, 0);

    std::vector<int> Mrow((size_t)C);
    std::vector<uint8_t> stream((size_t)((W * 8)));

    for (int r = 0; r < R; ++r) {
        int32_t sat = 0;
        int64_t errsum = 0;

        uint8_t* e4row = expected->e4m3_codes.data() + (size_t)r * (size_t)C;
        uint8_t* q4row = expected->q4_packed.data() + (size_t)r * (size_t)Kb;
        uint8_t* sf1row = expected->sf1.data() + (size_t)r * (size_t)S;
        uint8_t* sf2row = expected->sf2.data() + (size_t)r * (size_t)S;

        std::vector<int> q4n((size_t)C);

        for (int b = 0; b < S; ++b) {
            const int k0 = 32 * b;
            const int k1 = (k0 + 32 < C) ? (k0 + 32) : C;

            // ---- stage 1: amax + sexp1 ----
            uint32_t amaxbits = 0;
            for (int k = k0; k < k1; ++k) {
                const uint32_t ab =
                    mrq_oracle_f2u(in.x[(size_t)r * (size_t)C + (size_t)k]) &
                    0x7fffffffu;
                if (ab > amaxbits) amaxbits = ab;
            }

            int sexp1;
            uint8_t s1;
            if (amaxbits == 0) {
                sexp1 = -127;
                s1 = 0x00;
            } else {
                const int e = mrq_oracle_floor_log2(amaxbits);
                sexp1 = e - 8;
                if (sexp1 < -127) sexp1 = -127;
                s1 = (uint8_t)(sexp1 + 127);
            }
            sf1row[(size_t)b] = s1;

            // ---- stage 1: elements ----
            int qmax = 0;
            for (int k = k0; k < k1; ++k) {
                const uint32_t xb =
                    mrq_oracle_f2u(in.x[(size_t)r * (size_t)C + (size_t)k]);
                const int sign = (int)(xb >> 31);
                uint32_t mant;
                int ex;
                int code = 0, Mv = 0, st = 0;
                if (mrq_oracle_decompose(xb, &mant, &ex)) {
                    mrq_oracle_e4m3(mant, ex, sexp1, &code, &Mv, &st);
                }
                e4row[(size_t)k] = (uint8_t)((sign << 7) | code);
                Mrow[(size_t)k] = Mv;
                sat += st;
                if (Mv > qmax) qmax = Mv;
            }

            // ---- stage 2 ----
            if (qmax == 0) {
                sf2row[(size_t)b] = 0xFF;
                for (int k = k0; k < k1; ++k) {
                    q4n[(size_t)k] = (int)((e4row[(size_t)k] >> 7) << 3);
                }
                // err contribution 0
            } else {
                const int msb = mrq_oracle_msb32((uint32_t)qmax);
                sf2row[(size_t)b] = (uint8_t)msb;
                for (int k = k0; k < k1; ++k) {
                    const long long w16 = 16ll * (long long)Mrow[(size_t)k];
                    int idx = 0;
                    for (int j = 0; j < 7; ++j) {
                        if (w16 >= ((long long)MRQ_ORACLE_BP[j] << msb)) ++idx;
                    }
                    q4n[(size_t)k] =
                        (int)(((e4row[(size_t)k] >> 7) << 3) | idx);
                    const long long a = 8ll * (long long)Mrow[(size_t)k];
                    const long long rt = (long long)MRQ_ORACLE_LP[idx] << msb;
                    const long long err = a - rt;
                    errsum += err * err;
                }
            }
        }

        expected->sat1_count[(size_t)r] = sat;
        expected->row_err[(size_t)r] = errsum;

        // ---- pack q4 ----
        for (int j = 0; j < Kb; ++j) {
            const int lo = q4n[(size_t)(2 * j)];
            const int hi = (2 * j + 1 < C) ? q4n[(size_t)(2 * j + 1)] : 0;
            q4row[(size_t)j] = (uint8_t)(lo | (hi << 4));
        }

        // ---- digest ----
        std::memset(stream.data(), 0, stream.size());
        std::memcpy(stream.data(), e4row, (size_t)C);
        std::memcpy(stream.data() + C, q4row, (size_t)Kb);
        std::memcpy(stream.data() + C + Kb, sf1row, (size_t)S);
        std::memcpy(stream.data() + C + Kb + S, sf2row, (size_t)S);

        uint64_t h = MRQ_FNV_BASIS;
        for (int t = 0; t < MRQ_DIGEST_SUBSTREAMS; ++t) {
            uint64_t sub = MRQ_FNV_BASIS;
            for (int w = t; w < W; w += MRQ_DIGEST_SUBSTREAMS) {
                uint64_t word = 0;
                for (int bb = 0; bb < 8; ++bb) {
                    word |= (uint64_t)stream[(size_t)(8 * w + bb)] << (8 * bb);
                }
                sub = mrq_oracle_fnv_word(sub, word);
            }
            h = mrq_oracle_fnv_word(h, sub);
        }
        expected->row_digest[(size_t)r] = h;
    }
}

// ---------------------------------------------------------------------------
// Checkers (exact; no tolerances).
// ---------------------------------------------------------------------------

static inline bool mrq_check_bytes(
    const char* name,
    const std::vector<uint8_t>& expected,
    const uint8_t* got,
    std::string* error) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (got[i] != expected[i]) {
            if (error) {
                std::ostringstream oss;
                oss << name << " mismatch at byte " << i
                    << ": got 0x" << std::hex << (int)got[i]
                    << ", expected 0x" << (int)expected[i];
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

static inline bool mrq_check_all_outputs(
    const MrqRunSpec& run,
    const MrqExpected& expected,
    const MrqHostOutputsView& got,
    std::string* error) {
    const int R = run.R;

    if (!mrq_check_bytes("e4m3_codes", expected.e4m3_codes, got.e4m3_codes, error)) return false;
    if (!mrq_check_bytes("q4_packed", expected.q4_packed, got.q4_packed, error)) return false;
    if (!mrq_check_bytes("sf1", expected.sf1, got.sf1, error)) return false;
    if (!mrq_check_bytes("sf2", expected.sf2, got.sf2, error)) return false;

    for (int r = 0; r < R; ++r) {
        if (got.row_err[r] != expected.row_err[(size_t)r]) {
            if (error) {
                std::ostringstream oss;
                oss << "row_err mismatch row=" << r
                    << ": got " << got.row_err[r]
                    << ", expected " << expected.row_err[(size_t)r];
                *error = oss.str();
            }
            return false;
        }
        if (got.row_digest[r] != expected.row_digest[(size_t)r]) {
            if (error) {
                std::ostringstream oss;
                oss << "row_digest mismatch row=" << r
                    << ": got 0x" << std::hex << got.row_digest[r]
                    << ", expected 0x" << expected.row_digest[(size_t)r];
                *error = oss.str();
            }
            return false;
        }
        if (got.sat1_count[r] != expected.sat1_count[(size_t)r]) {
            if (error) {
                std::ostringstream oss;
                oss << "sat1_count mismatch row=" << r
                    << ": got " << got.sat1_count[r]
                    << ", expected " << expected.sat1_count[(size_t)r];
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

/*
GRADER CHECKS (all exact):
  - e4m3_codes byte-exact
  - q4_packed  byte-exact (odd-C high nibble 0)
  - sf1 / sf2  byte-exact (0x00 zero-input blocks; 0xFF zero-code blocks)
  - row_err    exact int64
  - row_digest exact u64
  - sat1_count exact int32

Harness should also enforce:
  - output guard sentinels
  - input immutability
  - held-out seeds
  - all nine distributions, ragged R/C tails
*/

#endif  // MICROSCALE_REQUANT_CHAIN_ORACLE_HPP_
