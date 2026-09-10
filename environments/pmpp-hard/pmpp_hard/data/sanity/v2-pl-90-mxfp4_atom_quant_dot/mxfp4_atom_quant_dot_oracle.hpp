// file: mxfp4_atom_quant_dot_oracle.hpp
//
// Independent CPU oracle for mxfp4_atom_quant_dot. Every rounding rule of
// the contract is reproduced in scalar host code, using only bit-level
// integer constructions and volatile fp32 arithmetic (no libm scaling).

#ifndef MXFP4_ATOM_QUANT_DOT_ORACLE_HPP_
#define MXFP4_ATOM_QUANT_DOT_ORACLE_HPP_

#include "mxfp4_atom_quant_dot_common.h"

#include <stdint.h>
#include <stddef.h>

#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

struct MxqHostInputsView {
    const float* x;
    const float* v;
};

struct MxqHostOutputsView {
    const uint8_t* pay_atoms;
    const uint8_t* sf_atoms;
    const float* row_dot;
    const uint64_t* row_digest;
    const int32_t* sat_count;
};

struct MxqExpected {
    std::vector<uint8_t> pay_atoms;
    std::vector<uint8_t> sf_atoms;
    std::vector<float> row_dot;
    std::vector<uint64_t> row_digest;
    std::vector<int32_t> sat_count;
};

// ---------------------------------------------------------------------------
// Exact scalar helpers (bit-level; independent of libm).
// ---------------------------------------------------------------------------

static inline uint32_t mxq_oracle_f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

static inline float mxq_oracle_u2f(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// floor(log2(a)) for finite a > 0 (normal or subnormal), from the bits.
static inline int mxq_oracle_floor_log2(float a) {
    const uint32_t u = mxq_oracle_f2u(a);
    const int E = (int)((u >> 23) & 0xffu);
    const uint32_t M = u & 0x7fffffu;
    if (E > 0) return E - 127;
    int il = 0;
    while ((M >> (il + 1)) != 0u) ++il;  // msb_index(M); M != 0 here
    return il - 149;
}

// t * 2^p, exact. t is a positive fp32 normal with few significand bits
// (lattice values / thresholds); every use in this task is exactly
// representable, possibly as a subnormal.
static inline float mxq_oracle_pow2_scale(float t, int p) {
    const uint32_t u = mxq_oracle_f2u(t);
    const int E = (int)((u >> 23) & 0xffu);
    const uint32_t M = u & 0x7fffffu;
    const int e = E - 127 + p;
    if (e >= -126) {
        return mxq_oracle_u2f(((uint32_t)(e + 127) << 23) | M);
    }
    const uint32_t mant24 = 0x800000u | M;
    const int shift = -126 - e;  // >= 1; exact for all task uses
    return mxq_oracle_u2f(mant24 >> shift);
}

static inline float mxq_oracle_fadd(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float cv = av + bv;
    return cv;
}

static inline float mxq_oracle_fmul(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float cv = av * bv;
    return cv;
}

static inline uint64_t mxq_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= MXQ_FNV_PRIME;
    return h;
}

static inline uint64_t mxq_oracle_fnv_u64(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h = mxq_oracle_fnv_byte(h, (uint8_t)((v >> (8 * i)) & 0xffu));
    }
    return h;
}

// E2M1 magnitude code per the contract threshold table. thr[i] holds
// T(0.25), T(0.75), T(1.25), T(1.75), T(2.5), T(3.5), T(5.0).
static inline int mxq_oracle_mag_code(float a, const float* thr) {
    if (a <= thr[0]) return 0;
    if (a < thr[1]) return 1;
    if (a <= thr[2]) return 2;
    if (a < thr[3]) return 3;
    if (a <= thr[4]) return 4;
    if (a < thr[5]) return 5;
    if (a <= thr[6]) return 6;
    return 7;
}

static const float MXQ_ORACLE_LAT[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

static const float MXQ_ORACLE_THR_BASE[7] = {
    0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f
};

static inline void mxq_cpu_oracle(
    const MxqRunSpec& run,
    const MxqHostInputsView& in,
    MxqExpected* expected) {
    const int R = run.R;
    const int C = run.C;
    const int S = mxq_S(C);
    const int Kb = mxq_Kb(C);
    const int AM = mxq_AM(R);
    const int AK = mxq_AK(C);
    const int AS = mxq_AS(C);

    expected->pay_atoms.assign(mxq_pay_atom_bytes(R, C), 0);
    expected->sf_atoms.assign(mxq_sf_atom_bytes(R, C), 0);
    expected->row_dot.assign((size_t)R, 0.0f);
    expected->row_digest.assign((size_t)R, 0);
    expected->sat_count.assign((size_t)R, 0);

    std::vector<uint8_t> codes((size_t)C);      // per-row element nibbles
    std::vector<uint8_t> logical((size_t)Kb);   // per-row logical bytes
    std::vector<uint8_t> sf_row((size_t)S);     // per-row scale bytes
    std::vector<int> sexp_row((size_t)S);       // per-row block exponents

    for (int r = 0; r < R; ++r) {
        int32_t sat = 0;

        // ------ stages 1+2: scales and codes ------
        for (int b = 0; b < S; ++b) {
            const int k0 = 32 * b;
            const int k1 = (k0 + 32 < C) ? (k0 + 32) : C;

            float amax = 0.0f;
            for (int k = k0; k < k1; ++k) {
                const float a = fabsf(in.x[(size_t)r * (size_t)C + (size_t)k]);
                if (a > amax) amax = a;
            }

            int sexp;
            uint8_t sf;
            if (amax == 0.0f) {
                sexp = -127;
                sf = 0x00;
            } else {
                const int e = mxq_oracle_floor_log2(amax);
                sexp = e - 2;
                if (sexp < -127) sexp = -127;
                sf = (uint8_t)(sexp + 127);
            }
            sf_row[(size_t)b] = sf;
            sexp_row[(size_t)b] = sexp;

            float thr[7];
            for (int i = 0; i < 7; ++i) {
                thr[i] = mxq_oracle_pow2_scale(MXQ_ORACLE_THR_BASE[i], sexp);
            }
            const float t6 = mxq_oracle_pow2_scale(6.0f, sexp);

            for (int k = k0; k < k1; ++k) {
                const float xv = in.x[(size_t)r * (size_t)C + (size_t)k];
                const float a = fabsf(xv);
                const int sign = std::signbit(xv) ? 1 : 0;
                const int m = mxq_oracle_mag_code(a, thr);
                codes[(size_t)k] = (uint8_t)((sign << 3) | m);
                if (a > t6) ++sat;
            }
        }
        expected->sat_count[(size_t)r] = sat;

        // ------ stage 3: logical bytes ------
        for (int j = 0; j < Kb; ++j) {
            const uint8_t lo = codes[(size_t)(2 * j)];
            const uint8_t hi = (2 * j + 1 < C) ? codes[(size_t)(2 * j + 1)] : 0;
            logical[(size_t)j] = (uint8_t)(lo | (hi << 4));
        }

        // ------ stage 3: atom scatter ------
        const int am = r / 128;
        const int rr = r % 128;
        for (int ak = 0; ak < AK; ++ak) {
            const size_t base = (size_t)4096 * ((size_t)am * (size_t)AK + (size_t)ak);
            for (int u = 0; u < 32; ++u) {
                const int klo = 64 * ak + u;
                const int khi = klo + 32;
                const uint8_t lo = (klo < C) ? codes[(size_t)klo] : 0;
                const uint8_t hi = (khi < C) ? codes[(size_t)khi] : 0;
                const size_t off = (size_t)((u / 8) * 1024 +
                                            (rr % 32) * 32 +
                                            (rr / 32) * 8 +
                                            (u % 8));
                expected->pay_atoms[base + off] = (uint8_t)(lo | (hi << 4));
            }
        }
        for (int s = 0; s < S; ++s) {
            const int as = s / 4;
            const int c = s % 4;
            const size_t base = (size_t)512 * ((size_t)am * (size_t)AS + (size_t)as);
            const size_t off = (size_t)((rr % 32) * 16 + (rr / 32) * 4 + c);
            expected->sf_atoms[base + off] = sf_row[(size_t)s];
        }

        // ------ stage 4: pinned-tree dequant dot ------
        float partial[MXQ_DOT_LANES];
        for (int l = 0; l < MXQ_DOT_LANES; ++l) partial[l] = 0.0f;

        for (int l = 0; l < MXQ_DOT_LANES; ++l) {
            for (int k = l; k < C; k += MXQ_DOT_LANES) {
                const uint8_t code = codes[(size_t)k];
                const int m = code & 7;
                const int sign = (code >> 3) & 1;
                float dq;
                if (m == 0) {
                    dq = sign ? -0.0f : 0.0f;
                } else {
                    const float mag = mxq_oracle_pow2_scale(
                        MXQ_ORACLE_LAT[m], sexp_row[(size_t)(k / 32)]);
                    dq = sign ? -mag : mag;
                }
                partial[l] = mxq_oracle_fadd(
                    partial[l], mxq_oracle_fmul(dq, in.v[(size_t)k]));
            }
        }
        for (int stride = MXQ_DOT_LANES / 2; stride > 0; stride >>= 1) {
            for (int l = 0; l < stride; ++l) {
                partial[l] = mxq_oracle_fadd(partial[l], partial[l + stride]);
            }
        }
        expected->row_dot[(size_t)r] = partial[0];

        // ------ stage 5: two-level digest ------
        const int total_len = Kb + S;
        const int sub_len = (total_len + MXQ_DIGEST_SUBSTREAMS - 1) /
                            MXQ_DIGEST_SUBSTREAMS;
        uint64_t h = MXQ_FNV_BASIS;
        for (int g = 0; g < MXQ_DIGEST_SUBSTREAMS; ++g) {
            const int lo = g * sub_len;
            int hi = lo + sub_len;
            if (hi > total_len) hi = total_len;
            uint64_t sub = MXQ_FNV_BASIS;
            for (int i = lo; i < hi; ++i) {
                const uint8_t byte = (i < Kb)
                    ? logical[(size_t)i]
                    : sf_row[(size_t)(i - Kb)];
                sub = mxq_oracle_fnv_byte(sub, byte);
            }
            h = mxq_oracle_fnv_u64(h, sub);
        }
        expected->row_digest[(size_t)r] = h;
    }
}

// ---------------------------------------------------------------------------
// Checkers (exact; no tolerances).
// ---------------------------------------------------------------------------

static inline bool mxq_check_bytes(
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

static inline bool mxq_check_all_outputs(
    const MxqRunSpec& run,
    const MxqExpected& expected,
    const MxqHostOutputsView& got,
    std::string* error) {
    const int R = run.R;

    if (!mxq_check_bytes("pay_atoms", expected.pay_atoms, got.pay_atoms, error)) {
        return false;
    }
    if (!mxq_check_bytes("sf_atoms", expected.sf_atoms, got.sf_atoms, error)) {
        return false;
    }

    for (int r = 0; r < R; ++r) {
        const uint32_t eb = mxq_oracle_f2u(expected.row_dot[(size_t)r]);
        const uint32_t gb = mxq_oracle_f2u(got.row_dot[r]);
        if (eb != gb) {
            if (error) {
                std::ostringstream oss;
                oss << "row_dot mismatch row=" << r
                    << ": got " << got.row_dot[r]
                    << " (0x" << std::hex << gb << std::dec << ")"
                    << ", expected " << expected.row_dot[(size_t)r]
                    << " (0x" << std::hex << eb << ")";
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
        if (got.sat_count[r] != expected.sat_count[(size_t)r]) {
            if (error) {
                std::ostringstream oss;
                oss << "sat_count mismatch row=" << r
                    << ": got " << got.sat_count[r]
                    << ", expected " << expected.sat_count[(size_t)r];
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

/*
GRADER CHECKS (all exact):
  - pay_atoms  byte-exact, including all padding bytes (0x00)
  - sf_atoms   byte-exact, including all padding bytes (0x00)
  - row_dot    bit-exact fp32 (u32 pattern compare)
  - row_digest exact u64
  - sat_count  exact int32

Harness should also enforce:
  - output guard sentinels
  - input immutability
  - held-out seeds
  - all nine distributions, ragged R/C tails
*/

#endif  // MXFP4_ATOM_QUANT_DOT_ORACLE_HPP_
