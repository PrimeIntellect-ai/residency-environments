// file: awq_actorder_repack_gemv_oracle.hpp
//
// Independent CPU oracle for awq_actorder_repack_gemv. Every rule of the
// contract is reproduced in scalar host code: the stable rank formula is
// evaluated by counting (not by a library sort), the dequant-dot uses
// volatile fp32 arithmetic (no FMA contraction, no reassociation), and all
// packing/unpacking is bit-level integer code.

#ifndef AWQ_ACTORDER_REPACK_GEMV_ORACLE_HPP_
#define AWQ_ACTORDER_REPACK_GEMV_ORACLE_HPP_

#include "awq_actorder_repack_gemv_common.h"

#include <stdint.h>
#include <stddef.h>

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

struct AwqHostInputsView {
    const uint32_t* qweight;
    const uint32_t* qzeros;
    const float* scales;
    const int32_t* g_idx;
    const float* x;
};

struct AwqHostOutputsView {
    const uint8_t* rq_atoms;
    const float* col_dot;
    const uint64_t* col_digest;
    const int32_t* col_zsum;
    const int32_t* perm;
};

struct AwqExpected {
    std::vector<uint8_t> rq_atoms;
    std::vector<float> col_dot;
    std::vector<uint64_t> col_digest;
    std::vector<int32_t> col_zsum;
    std::vector<int32_t> perm;
};

static inline uint32_t awq_oracle_f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

static inline float awq_oracle_fadd(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float cv = av + bv;
    return cv;
}

static inline float awq_oracle_fmul(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float cv = av * bv;
    return cv;
}

static inline uint64_t awq_oracle_fnv_word(uint64_t h, uint64_t w) {
    h ^= w;
    h *= AWQ_FNV_PRIME;
    return h;
}

static inline int awq_oracle_q(
    const uint32_t* qweight, int Nw, int k, int n) {
    const uint32_t word = qweight[(size_t)k * (size_t)Nw + (size_t)(n >> 3)];
    return (int)((word >> (4 * awq_wlane(n & 7))) & 0xFu);
}

static inline int awq_oracle_z(
    const uint32_t* qzeros, int Nw, int g, int n) {
    const uint32_t word = qzeros[(size_t)g * (size_t)Nw + (size_t)(n >> 3)];
    return (int)((word >> (4 * awq_zlane(n & 7))) & 0xFu);
}

static inline void awq_cpu_oracle(
    const AwqRunSpec& run,
    const AwqHostInputsView& in,
    AwqExpected* expected) {
    const int K = run.K;
    const int N = run.N;
    const int G = run.G;
    const int Nw = awq_Nw(N);
    const int TA = awq_TA(K);
    const int TN = awq_TN(N);

    expected->rq_atoms.assign(awq_rq_bytes(K, N), 0);
    expected->col_dot.assign((size_t)N, 0.0f);
    expected->col_digest.assign((size_t)N, 0);
    expected->col_zsum.assign((size_t)N, 0);
    expected->perm.assign((size_t)K, 0);

    // ------ stage 1: stable rank by direct counting (O(K^2) but exact
    // to the normative formula; K <= 8192 keeps this manageable) ------
    std::vector<int> rank((size_t)K, 0);
    {
        // counting sort formulation, equivalent to the contract formula:
        // rank(k) = (#k' with smaller group) + (#k' < k with equal group).
        std::vector<int> count((size_t)G + 1, 0);
        for (int k = 0; k < K; ++k) count[(size_t)in.g_idx[k] + 1]++;
        for (int g = 0; g < G; ++g) count[(size_t)g + 1] += count[(size_t)g];
        std::vector<int> next(count.begin(), count.end() - 1);
        for (int k = 0; k < K; ++k) {
            rank[(size_t)k] = next[(size_t)in.g_idx[k]]++;
            expected->perm[(size_t)rank[(size_t)k]] = k;
        }
    }

    uint32_t* rq_words = reinterpret_cast<uint32_t*>(expected->rq_atoms.data());

    // Per-position decode caches for the current column n.
    std::vector<int8_t> qz_perm((size_t)K);

    const int L = K + 4 * G;
    const int W = (L + 7) / 8;

    for (int n = 0; n < N; ++n) {
        // ------ decode column n in permuted order ------
        int32_t zsum = 0;
        for (int j = 0; j < K; ++j) {
            const int k = expected->perm[(size_t)j];
            const int g = in.g_idx[k];
            const int qv = awq_oracle_q(in.qweight, Nw, k, n);
            const int zv = awq_oracle_z(in.qzeros, Nw, g, n);
            const int d = qv - zv;
            qz_perm[(size_t)j] = (int8_t)d;
            zsum += d;
        }
        expected->col_zsum[(size_t)n] = zsum;

        // ------ stage 3: pinned 16-lane dot ------
        float partial[AWQ_DOT_LANES];
        for (int l = 0; l < AWQ_DOT_LANES; ++l) partial[l] = 0.0f;
        for (int l = 0; l < AWQ_DOT_LANES; ++l) {
            for (int j = l; j < K; j += AWQ_DOT_LANES) {
                const int k = expected->perm[(size_t)j];
                const int g = in.g_idx[k];
                const float s = in.scales[(size_t)g * (size_t)N + (size_t)n];
                const float w = awq_oracle_fmul((float)qz_perm[(size_t)j], s);
                const float p = awq_oracle_fmul(w, in.x[(size_t)k]);
                partial[l] = awq_oracle_fadd(partial[l], p);
            }
        }
        for (int stride = AWQ_DOT_LANES / 2; stride > 0; stride >>= 1) {
            for (int l = 0; l < stride; ++l) {
                partial[l] = awq_oracle_fadd(partial[l], partial[l + stride]);
            }
        }
        expected->col_dot[(size_t)n] = partial[0];

        // ------ stage 4: two-level word-folded digest ------
        uint64_t h = AWQ_FNV_BASIS;
        for (int r = 0; r < AWQ_DIGEST_SUBSTREAMS; ++r) {
            uint64_t sub = AWQ_FNV_BASIS;
            for (int w = r; w < W; w += AWQ_DIGEST_SUBSTREAMS) {
                uint64_t word = 0;
                for (int b = 0; b < 8; ++b) {
                    const int i = 8 * w + b;
                    uint8_t byte = 0;
                    if (i < K) {
                        byte = (uint8_t)qz_perm[(size_t)i];
                    } else if (i < L) {
                        const int idx = i - K;
                        const uint32_t sb = awq_oracle_f2u(
                            in.scales[(size_t)(idx >> 2) * (size_t)N + (size_t)n]);
                        byte = (uint8_t)((sb >> (8 * (idx & 3))) & 0xffu);
                    }
                    word |= (uint64_t)byte << (8 * b);
                }
                sub = awq_oracle_fnv_word(sub, word);
            }
            h = awq_oracle_fnv_word(h, sub);
        }
        expected->col_digest[(size_t)n] = h;
    }

    // ------ stage 2: tile repack ------
    for (int ta = 0; ta < TA; ++ta) {
        for (int tn = 0; tn < TN; ++tn) {
            const size_t base = (size_t)128 * ((size_t)ta * (size_t)TN + (size_t)tn);
            for (int h2 = 0; h2 < 2; ++h2) {
                for (int nt = 0; nt < 64; ++nt) {
                    const int n = 64 * tn + nt;
                    uint32_t word = 0;
                    for (int i = 0; i < 8; ++i) {
                        const int j = 16 * ta + 8 * h2 + i;
                        int qv = 0;
                        if (j < K && n < N) {
                            const int k = expected->perm[(size_t)j];
                            qv = awq_oracle_q(in.qweight, Nw, k, n);
                        }
                        word |= (uint32_t)qv << (4 * awq_wlane(i));
                    }
                    const size_t off =
                        base + (size_t)(h2 * 64 + (((nt & 7) << 3) | (nt >> 3)));
                    rq_words[off] = word;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Checkers (exact; no tolerances).
// ---------------------------------------------------------------------------

static inline bool awq_check_bytes(
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

static inline bool awq_check_all_outputs(
    const AwqRunSpec& run,
    const AwqExpected& expected,
    const AwqHostOutputsView& got,
    std::string* error) {
    const int K = run.K;
    const int N = run.N;

    for (int k = 0; k < K; ++k) {
        if (got.perm[k] != expected.perm[(size_t)k]) {
            if (error) {
                std::ostringstream oss;
                oss << "perm mismatch at j=" << k
                    << ": got " << got.perm[k]
                    << ", expected " << expected.perm[(size_t)k];
                *error = oss.str();
            }
            return false;
        }
    }

    if (!awq_check_bytes("rq_atoms", expected.rq_atoms, got.rq_atoms, error)) {
        return false;
    }

    for (int n = 0; n < N; ++n) {
        const uint32_t eb = awq_oracle_f2u(expected.col_dot[(size_t)n]);
        const uint32_t gb = awq_oracle_f2u(got.col_dot[n]);
        if (eb != gb) {
            if (error) {
                std::ostringstream oss;
                oss << "col_dot mismatch col=" << n
                    << ": got " << got.col_dot[n]
                    << " (0x" << std::hex << gb << std::dec << ")"
                    << ", expected " << expected.col_dot[(size_t)n]
                    << " (0x" << std::hex << eb << ")";
                *error = oss.str();
            }
            return false;
        }
        if (got.col_digest[n] != expected.col_digest[(size_t)n]) {
            if (error) {
                std::ostringstream oss;
                oss << "col_digest mismatch col=" << n
                    << ": got 0x" << std::hex << got.col_digest[n]
                    << ", expected 0x" << expected.col_digest[(size_t)n];
                *error = oss.str();
            }
            return false;
        }
        if (got.col_zsum[n] != expected.col_zsum[(size_t)n]) {
            if (error) {
                std::ostringstream oss;
                oss << "col_zsum mismatch col=" << n
                    << ": got " << got.col_zsum[n]
                    << ", expected " << expected.col_zsum[(size_t)n];
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

/*
GRADER CHECKS (all exact):
  - perm       exact int32 (stable group argsort)
  - rq_atoms   byte-exact, including all padding words (0x00000000)
  - col_dot    bit-exact fp32 (u32 pattern compare)
  - col_digest exact u64
  - col_zsum   exact int32

Harness should also enforce:
  - output guard sentinels
  - input immutability
  - held-out seeds
  - all nine distributions, ragged K/N tails, empty groups
*/

#endif  // AWQ_ACTORDER_REPACK_GEMV_ORACLE_HPP_
