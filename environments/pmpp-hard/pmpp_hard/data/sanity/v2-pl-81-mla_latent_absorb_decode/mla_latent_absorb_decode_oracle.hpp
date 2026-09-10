// ============================================================================
// file: mla_latent_absorb_decode_oracle.hpp
//
// Independent host-side oracle for the mla_latent_absorb_decode contract.
// Pure CPU, double-precision attention math over an exactly modelled
// quantized cache. Shares no code with any GPU implementation.
// ============================================================================

#ifndef MLA_LATENT_ABSORB_DECODE_ORACLE_HPP_
#define MLA_LATENT_ABSORB_DECODE_ORACLE_HPP_

#include "mla_latent_absorb_decode_common.h"

#include <stdint.h>

#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct MlaHostInputsView {
    const int32_t* active_seq;
    const int32_t* new_token_count;
    const float* new_c;
    const float* new_r;
    const float* q;
    const float* q_rope;
};

struct MlaHostOutputsView {
    const float* y;
    const float* lse;
    const int32_t* seq_len;
    const uint64_t* cache_hash;
    const uint64_t* meta_checksum;
    const int32_t* sat_count;
    const int32_t* total_tokens;
};

struct MlaExpected {
    std::vector<float> y;         // [A * 8 * Hq * d_v], pad slots exact 0
    std::vector<float> lse;       // [A * 8 * Hq], pad slots exact 0
    std::vector<int32_t> seq_len; // [B]
    std::vector<uint64_t> cache_hash;
    uint64_t meta_checksum = 0;
    int32_t sat_count = 0;
    int32_t total_tokens = 0;
};

static const uint64_t kMlaOracleFnvBasis = 1469598103934665603ULL;
static const uint64_t kMlaOracleFnvPrime = 1099511628211ULL;

static inline uint64_t mla_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= kMlaOracleFnvPrime;
    return h;
}

static inline uint64_t mla_oracle_fnv_i32(uint64_t h, int32_t v) {
    const uint32_t u = static_cast<uint32_t>(v);
    h = mla_oracle_fnv_byte(h, static_cast<uint8_t>(u & 0xFF));
    h = mla_oracle_fnv_byte(h, static_cast<uint8_t>((u >> 8) & 0xFF));
    h = mla_oracle_fnv_byte(h, static_cast<uint8_t>((u >> 16) & 0xFF));
    h = mla_oracle_fnv_byte(h, static_cast<uint8_t>((u >> 24) & 0xFF));
    return h;
}

struct MlaOracleState {
    MlaProblemSpec spec;
    int gc = 0;  // d_c / 32
    int gr = 0;  // d_r / 32

    // Quantized cache, exact byte model.
    // c_scale[b][t][g], c_byte[b][t][d], r_scale[b][t][g], r_byte[b][t][d].
    std::vector<int8_t> c_scale;
    std::vector<int8_t> c_byte;
    std::vector<int8_t> r_scale;
    std::vector<int8_t> r_byte;

    std::vector<int32_t> seq_len;
    int32_t step_counter = 0;
    int32_t sat_count = 0;
    int32_t total_tokens = 0;

    void init(const MlaProblemSpec& s) {
        if (!mla_validate_problem_spec(&s)) {
            throw std::runtime_error("oracle: invalid problem spec");
        }
        spec = s;
        gc = spec.d_c / MLA_QUANT_GROUP;
        gr = spec.d_r / MLA_QUANT_GROUP;

        const size_t toks = static_cast<size_t>(spec.B) * spec.max_seq_len;
        c_scale.assign(toks * gc, 0);
        c_byte.assign(toks * spec.d_c, 0);
        r_scale.assign(toks * gr, 0);
        r_byte.assign(toks * spec.d_r, 0);
        seq_len.assign(spec.B, 0);
        step_counter = 0;
        sat_count = 0;
        total_tokens = 0;
    }

    void reset() {
        std::fill(c_scale.begin(), c_scale.end(), 0);
        std::fill(c_byte.begin(), c_byte.end(), 0);
        std::fill(r_scale.begin(), r_scale.end(), 0);
        std::fill(r_byte.begin(), r_byte.end(), 0);
        std::fill(seq_len.begin(), seq_len.end(), 0);
        step_counter = 0;
        sat_count = 0;
        total_tokens = 0;
    }

    // Weights, captured once (host copies).
    std::vector<float> W_uk;      // [Hq * d_h * d_c]
    std::vector<float> W_uv;      // [Hq * d_v * d_c]
    std::vector<float> rope_cos;  // [max_seq_len * d_r/2]
    std::vector<float> rope_sin;

    void set_weights(
        const std::vector<float>& wuk,
        const std::vector<float>& wuv,
        const std::vector<float>& rc,
        const std::vector<float>& rs) {
        W_uk = wuk;
        W_uv = wuv;
        rope_cos = rc;
        rope_sin = rs;
    }

    // NORMATIVE group quantizer, exact integer/bit model (no libm rounding
    // ambiguity: rne implemented arithmetically on exact values).
    void quantize_group(const float* x, int n, int8_t* scale_out, int8_t* bytes_out) {
        float amax = 0.0f;
        for (int d = 0; d < n; ++d) {
            const float a = std::fabs(x[d]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) {
            *scale_out = 0;
            for (int d = 0; d < n; ++d) bytes_out[d] = 0;
            return;
        }

        uint32_t bits;
        std::memcpy(&bits, &amax, sizeof(bits));
        const int kfloor = static_cast<int>((bits >> 23) & 0xFF) - 127;
        int se = kfloor - 6;
        if (se < -110) se = -110;
        if (se > 110) se = 110;
        *scale_out = static_cast<int8_t>(se);

        for (int d = 0; d < n; ++d) {
            // Exact power-of-two scaling in double, then exact rne.
            const double z = static_cast<double>(x[d]) * std::ldexp(1.0, -se);
            double fl = std::floor(z);
            const double frac = z - fl;
            long long nval;
            if (frac > 0.5) {
                nval = static_cast<long long>(fl) + 1;
            } else if (frac < 0.5) {
                nval = static_cast<long long>(fl);
            } else {
                long long lo = static_cast<long long>(fl);
                nval = (lo % 2 == 0) ? lo : lo + 1;
            }
            if (nval > 127) {
                sat_count += 1;
                nval = 127;
            } else if (nval < -127) {
                sat_count += 1;
                nval = -127;
            }
            bytes_out[d] = static_cast<int8_t>(nval);
        }
    }

    size_t tok_index(int b, int t) const {
        return static_cast<size_t>(b) * spec.max_seq_len + static_cast<size_t>(t);
    }

    void step(
        const MlaRunSpec& run,
        const MlaHostInputsView& in,
        MlaExpected* expected) {
        if (!mla_validate_run_spec(&run, &spec)) {
            throw std::runtime_error("oracle: invalid run spec");
        }

        const int A = run.active_count;
        const int Hq = spec.Hq;
        const int d_c = spec.d_c;
        const int d_r = spec.d_r;
        const int d_h = spec.d_h;
        const int d_v = spec.d_v;
        const int half_r = d_r / 2;

        step_counter += 1;

        std::vector<int32_t> len_before(seq_len);

        // ---- Phase 2: appends -------------------------------------------
        for (int a = 0; a < A; ++a) {
            const int b = in.active_seq[a];
            const int cnt = in.new_token_count[a];
            if (b < 0 || b >= spec.B) throw std::runtime_error("oracle: bad seq id");
            if (cnt < 1 || cnt > MLA_MAX_NEW_TOKENS) {
                throw std::runtime_error("oracle: bad new_token_count");
            }
            if (seq_len[b] + cnt > spec.max_seq_len) {
                throw std::runtime_error("oracle: seq overflow");
            }

            for (int nt = 0; nt < cnt; ++nt) {
                const int pos = len_before[b] + nt;
                const size_t ti = tok_index(b, pos);
                const float* cvec =
                    in.new_c + (static_cast<size_t>(a) * MLA_MAX_NEW_TOKENS + nt) * d_c;
                const float* rvec =
                    in.new_r + (static_cast<size_t>(a) * MLA_MAX_NEW_TOKENS + nt) * d_r;

                for (int g = 0; g < gc; ++g) {
                    quantize_group(
                        cvec + g * MLA_QUANT_GROUP,
                        MLA_QUANT_GROUP,
                        &c_scale[ti * gc + g],
                        &c_byte[ti * d_c + g * MLA_QUANT_GROUP]);
                }
                for (int g = 0; g < gr; ++g) {
                    quantize_group(
                        rvec + g * MLA_QUANT_GROUP,
                        MLA_QUANT_GROUP,
                        &r_scale[ti * gr + g],
                        &r_byte[ti * d_r + g * MLA_QUANT_GROUP]);
                }
            }

            seq_len[b] += cnt;
            total_tokens += cnt;
        }

        // ---- Phase 3: attention -----------------------------------------
        const size_t y_count =
            static_cast<size_t>(A) * MLA_MAX_NEW_TOKENS * Hq * d_v;
        const size_t lse_count =
            static_cast<size_t>(A) * MLA_MAX_NEW_TOKENS * Hq;
        expected->y.assign(y_count, 0.0f);
        expected->lse.assign(lse_count, 0.0f);

        const double scale = 1.0 / std::sqrt(static_cast<double>(d_h + d_r));

        std::vector<double> ql(d_c);
        std::vector<double> z(d_c);
        std::vector<double> ctil(d_c);
        std::vector<double> scores;

        for (int a = 0; a < A; ++a) {
            const int b = in.active_seq[a];
            const int cnt = in.new_token_count[a];

            for (int nt = 0; nt < cnt; ++nt) {
                const int p = len_before[b] + nt;
                const float* qvec_base =
                    in.q + ((static_cast<size_t>(a) * MLA_MAX_NEW_TOKENS + nt) * Hq) *
                               static_cast<size_t>(d_h);
                const float* qr_base =
                    in.q_rope +
                    ((static_cast<size_t>(a) * MLA_MAX_NEW_TOKENS + nt) * Hq) *
                        static_cast<size_t>(d_r);

                for (int h = 0; h < Hq; ++h) {
                    const float* qv = qvec_base + static_cast<size_t>(h) * d_h;
                    const float* qr = qr_base + static_cast<size_t>(h) * d_r;
                    const float* wuk = W_uk.data() +
                        static_cast<size_t>(h) * d_h * d_c;

                    for (int j = 0; j < d_c; ++j) ql[j] = 0.0;
                    for (int i = 0; i < d_h; ++i) {
                        const double qi = qv[i];
                        const float* wrow = wuk + static_cast<size_t>(i) * d_c;
                        for (int j = 0; j < d_c; ++j) {
                            ql[j] += qi * static_cast<double>(wrow[j]);
                        }
                    }

                    const int L = p + 1;
                    scores.assign(L, 0.0);

                    double m = -1.0e300;
                    for (int t = 0; t < L; ++t) {
                        const size_t ti = tok_index(b, t);
                        double s = 0.0;
                        for (int g = 0; g < gc; ++g) {
                            const double sc =
                                std::ldexp(1.0, c_scale[ti * gc + g]);
                            for (int d = 0; d < MLA_QUANT_GROUP; ++d) {
                                const int dd = g * MLA_QUANT_GROUP + d;
                                s += ql[dd] *
                                     (static_cast<double>(c_byte[ti * d_c + dd]) * sc);
                            }
                        }
                        for (int j = 0; j < half_r; ++j) {
                            const int d0 = 2 * j;
                            const int d1 = 2 * j + 1;
                            const double sc0 =
                                std::ldexp(1.0, r_scale[ti * gr + d0 / MLA_QUANT_GROUP]);
                            const double sc1 =
                                std::ldexp(1.0, r_scale[ti * gr + d1 / MLA_QUANT_GROUP]);
                            const double r0 =
                                static_cast<double>(r_byte[ti * d_r + d0]) * sc0;
                            const double r1 =
                                static_cast<double>(r_byte[ti * d_r + d1]) * sc1;
                            const double co =
                                rope_cos[static_cast<size_t>(t) * half_r + j];
                            const double si =
                                rope_sin[static_cast<size_t>(t) * half_r + j];
                            const double rr0 = r0 * co - r1 * si;
                            const double rr1 = r0 * si + r1 * co;
                            s += static_cast<double>(qr[d0]) * rr0;
                            s += static_cast<double>(qr[d1]) * rr1;
                        }
                        s *= scale;
                        scores[t] = s;
                        if (s > m) m = s;
                    }

                    double l = 0.0;
                    for (int j = 0; j < d_c; ++j) z[j] = 0.0;
                    for (int t = 0; t < L; ++t) {
                        const double w = std::exp(scores[t] - m);
                        l += w;
                        const size_t ti = tok_index(b, t);
                        for (int g = 0; g < gc; ++g) {
                            const double sc =
                                std::ldexp(1.0, c_scale[ti * gc + g]);
                            for (int d = 0; d < MLA_QUANT_GROUP; ++d) {
                                const int dd = g * MLA_QUANT_GROUP + d;
                                z[dd] += w *
                                    (static_cast<double>(c_byte[ti * d_c + dd]) * sc);
                            }
                        }
                    }
                    const double inv_l = 1.0 / l;
                    for (int j = 0; j < d_c; ++j) z[j] *= inv_l;

                    const float* wuv = W_uv.data() +
                        static_cast<size_t>(h) * d_v * d_c;
                    const size_t y_base =
                        ((static_cast<size_t>(a) * MLA_MAX_NEW_TOKENS + nt) * Hq + h) *
                        static_cast<size_t>(d_v);
                    for (int i = 0; i < d_v; ++i) {
                        double acc = 0.0;
                        const float* wrow = wuv + static_cast<size_t>(i) * d_c;
                        for (int j = 0; j < d_c; ++j) {
                            acc += static_cast<double>(wrow[j]) * z[j];
                        }
                        expected->y[y_base + i] = static_cast<float>(acc);
                    }
                    expected->lse[(static_cast<size_t>(a) * MLA_MAX_NEW_TOKENS + nt) * Hq + h] =
                        static_cast<float>(m + std::log(l));
                }
            }
        }

        // ---- Phase 5: exact outputs --------------------------------------
        expected->seq_len = seq_len;

        expected->cache_hash.assign(spec.B, 0);
        for (int b = 0; b < spec.B; ++b) {
            uint64_t h = kMlaOracleFnvBasis;
            for (int t = 0; t < seq_len[b]; ++t) {
                const size_t ti = tok_index(b, t);
                for (int g = 0; g < gc; ++g) {
                    h = mla_oracle_fnv_byte(h, static_cast<uint8_t>(c_scale[ti * gc + g]));
                }
                for (int d = 0; d < d_c; ++d) {
                    h = mla_oracle_fnv_byte(h, static_cast<uint8_t>(c_byte[ti * d_c + d]));
                }
                for (int g = 0; g < gr; ++g) {
                    h = mla_oracle_fnv_byte(h, static_cast<uint8_t>(r_scale[ti * gr + g]));
                }
                for (int d = 0; d < d_r; ++d) {
                    h = mla_oracle_fnv_byte(h, static_cast<uint8_t>(r_byte[ti * d_r + d]));
                }
            }
            h = mla_oracle_fnv_i32(h, seq_len[b]);
            expected->cache_hash[b] = h;
        }

        uint64_t mh = kMlaOracleFnvBasis;
        mh = mla_oracle_fnv_i32(mh, spec.B);
        mh = mla_oracle_fnv_i32(mh, spec.Hq);
        mh = mla_oracle_fnv_i32(mh, spec.d_c);
        mh = mla_oracle_fnv_i32(mh, spec.d_r);
        mh = mla_oracle_fnv_i32(mh, spec.d_h);
        mh = mla_oracle_fnv_i32(mh, spec.d_v);
        mh = mla_oracle_fnv_i32(mh, spec.max_seq_len);
        mh = mla_oracle_fnv_i32(mh, step_counter);
        for (int b = 0; b < spec.B; ++b) {
            mh = mla_oracle_fnv_i32(mh, seq_len[b]);
        }
        mh = mla_oracle_fnv_i32(mh, sat_count);
        mh = mla_oracle_fnv_i32(mh, total_tokens);
        expected->meta_checksum = mh;

        expected->sat_count = sat_count;
        expected->total_tokens = total_tokens;
    }
};

static inline bool mla_check_all_outputs(
    const MlaRunSpec& run,
    const MlaProblemSpec& spec,
    const MlaExpected& expected,
    const MlaHostOutputsView& got,
    std::string* error) {
    const int A = run.active_count;
    const int Hq = spec.Hq;
    const int d_v = spec.d_v;

    const size_t y_count = static_cast<size_t>(A) * MLA_MAX_NEW_TOKENS * Hq * d_v;
    for (size_t i = 0; i < y_count; ++i) {
        const float e = expected.y[i];
        const float g = got.y[i];
        if (e == 0.0f) {
            // Pad slots and true zeros: pad slots must be written as exact
            // 0.0f, and a true-zero expected value within tolerance of 0
            // still passes the same check below; distinguish only pads.
        }
        const float diff = std::fabs(g - e);
        const float tol = MLA_Y_ATOL + MLA_Y_RTOL * std::fabs(e);
        if (!(diff <= tol)) {
            if (error) {
                std::ostringstream oss;
                oss << "y mismatch at flat index " << i << ": got " << g
                    << ", expected " << e << ", diff " << diff << ", tol " << tol;
                *error = oss.str();
            }
            return false;
        }
    }

    // Pad slots must be EXACT zeros.
    for (int a = 0; a < A; ++a) {
        // new_token_count is not visible here; rely on expected: pad slots
        // in expected are exactly 0 for whole (nt,h) rows. A row is a pad
        // row iff its expected lse is exactly 0 AND every expected y in the
        // slot is exactly 0 -- true zeros of lse (= log l + m = 0) with an
        // all-zero y row cannot occur for a real query because l >= 1 and
        // y = 0 exactly requires W_uv z = 0 in double, measure-zero and
        // never generated by the harness.
        for (int nt = 0; nt < MLA_MAX_NEW_TOKENS; ++nt) {
            const size_t lse_idx =
                (static_cast<size_t>(a) * MLA_MAX_NEW_TOKENS + nt) * Hq;
            bool pad = true;
            for (int h = 0; h < Hq && pad; ++h) {
                if (expected.lse[lse_idx + h] != 0.0f) pad = false;
            }
            if (!pad) continue;
            const size_t y_base = lse_idx * static_cast<size_t>(d_v);
            for (size_t k = 0; k < static_cast<size_t>(Hq) * d_v; ++k) {
                if (got.y[y_base + k] != 0.0f) {
                    if (error) {
                        std::ostringstream oss;
                        oss << "pad y slot not exactly zero at a=" << a
                            << " nt=" << nt << " flat=" << (y_base + k);
                        *error = oss.str();
                    }
                    return false;
                }
            }
            for (int h = 0; h < Hq; ++h) {
                if (got.lse[lse_idx + h] != 0.0f) {
                    if (error) {
                        std::ostringstream oss;
                        oss << "pad lse slot not exactly zero at a=" << a
                            << " nt=" << nt << " h=" << h;
                        *error = oss.str();
                    }
                    return false;
                }
            }
        }
    }

    const size_t lse_count = static_cast<size_t>(A) * MLA_MAX_NEW_TOKENS * Hq;
    for (size_t i = 0; i < lse_count; ++i) {
        const float e = expected.lse[i];
        const float g = got.lse[i];
        const float diff = std::fabs(g - e);
        const float tol = MLA_LSE_ATOL + MLA_LSE_RTOL * std::fabs(e);
        if (!(diff <= tol)) {
            if (error) {
                std::ostringstream oss;
                oss << "lse mismatch at flat index " << i << ": got " << g
                    << ", expected " << e << ", diff " << diff << ", tol " << tol;
                *error = oss.str();
            }
            return false;
        }
    }

    for (int b = 0; b < spec.B; ++b) {
        if (got.seq_len[b] != expected.seq_len[b]) {
            if (error) {
                std::ostringstream oss;
                oss << "seq_len mismatch at b=" << b << ": got " << got.seq_len[b]
                    << ", expected " << expected.seq_len[b];
                *error = oss.str();
            }
            return false;
        }
        if (got.cache_hash[b] != expected.cache_hash[b]) {
            if (error) {
                std::ostringstream oss;
                oss << "cache_hash mismatch at b=" << b << ": got 0x" << std::hex
                    << got.cache_hash[b] << ", expected 0x" << expected.cache_hash[b];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.meta_checksum[0] != expected.meta_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "meta_checksum mismatch: got 0x" << std::hex
                << got.meta_checksum[0] << ", expected 0x" << expected.meta_checksum;
            *error = oss.str();
        }
        return false;
    }

    if (got.sat_count[0] != expected.sat_count) {
        if (error) {
            std::ostringstream oss;
            oss << "sat_count mismatch: got " << got.sat_count[0]
                << ", expected " << expected.sat_count;
            *error = oss.str();
        }
        return false;
    }

    if (got.total_tokens[0] != expected.total_tokens) {
        if (error) {
            std::ostringstream oss;
            oss << "total_tokens mismatch: got " << got.total_tokens[0]
                << ", expected " << expected.total_tokens;
            *error = oss.str();
        }
        return false;
    }

    return true;
}

/*
Usage:
  1. Construct MlaOracleState, call init(spec) and set_weights(...) with the
     same host weight vectors uploaded to the solution's solution_init.
  2. For every solution_run call, call step(run, host_inputs, &expected) and
     compare with mla_check_all_outputs.
  3. reset() mirrors solution_reset (weights retained).
*/

#endif  // MLA_LATENT_ABSORB_DECODE_ORACLE_HPP_
