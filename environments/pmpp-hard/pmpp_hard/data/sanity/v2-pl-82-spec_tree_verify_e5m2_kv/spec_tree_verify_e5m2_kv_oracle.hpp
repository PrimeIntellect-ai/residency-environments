// ============================================================================
// file: spec_tree_verify_e5m2_kv_oracle.hpp
//
// Independent host-side oracle for the spec_tree_verify_e5m2_kv contract.
// Pure CPU: exact E5M2 byte model, exact page-pool simulation, double
// precision attention. Shares no code with any GPU implementation.
// ============================================================================

#ifndef SPEC_TREE_VERIFY_E5M2_KV_ORACLE_HPP_
#define SPEC_TREE_VERIFY_E5M2_KV_ORACLE_HPP_

#include "spec_tree_verify_e5m2_kv_common.h"

#include <stdint.h>

#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct StvHostInputsView {
    const int32_t* active_seq;
    const int32_t* node_count;
    const int32_t* parent;
    const float* draft_k;
    const float* draft_v;
    const float* q;
    const uint64_t* target_sig;
    const float* bonus_k;
    const float* bonus_v;
};

struct StvHostOutputsView {
    const float* y;
    const float* lse;
    const int32_t* accepted_tail;
    const int32_t* accepted_len;
    const int32_t* seq_len;
    const uint64_t* kv_hash;
    const uint64_t* page_state_checksum;
    const int32_t* free_pages;
    const int32_t* total_allocs;
    const int32_t* total_frees;
};

struct StvExpected {
    std::vector<float> y;
    std::vector<float> lse;
    std::vector<int32_t> accepted_tail;
    std::vector<int32_t> accepted_len;
    std::vector<int32_t> seq_len;
    std::vector<uint64_t> kv_hash;
    uint64_t page_state_checksum = 0;
    int32_t free_pages = 0;
    int32_t total_allocs = 0;
    int32_t total_frees = 0;
};

static const uint64_t kStvOracleFnvBasis = 1469598103934665603ULL;
static const uint64_t kStvOracleFnvPrime = 1099511628211ULL;

static inline uint64_t stv_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= kStvOracleFnvPrime;
    return h;
}

static inline uint64_t stv_oracle_fnv_i32(uint64_t h, int32_t v) {
    const uint32_t u = static_cast<uint32_t>(v);
    h = stv_oracle_fnv_byte(h, static_cast<uint8_t>(u & 0xFF));
    h = stv_oracle_fnv_byte(h, static_cast<uint8_t>((u >> 8) & 0xFF));
    h = stv_oracle_fnv_byte(h, static_cast<uint8_t>((u >> 16) & 0xFF));
    h = stv_oracle_fnv_byte(h, static_cast<uint8_t>((u >> 24) & 0xFF));
    return h;
}

// Exact E5M2 decode.
static inline double stv_oracle_e5m2_decode(uint8_t b) {
    const int sign = (b & 0x80u) ? -1 : 1;
    const int E = (b >> 2) & 0x1F;
    const int M = b & 0x3;
    if (E == 0) {
        return sign * static_cast<double>(M) * std::ldexp(1.0, -16);
    }
    return sign * std::ldexp(1.0 + M / 4.0, E - 15);
}

// Exact E5M2 encode per the contract (operates on double holding an exactly
// scaled fp32 value).
static inline uint8_t stv_oracle_e5m2_encode(float zf) {
    uint32_t bits;
    std::memcpy(&bits, &zf, sizeof(bits));
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const double a = std::fabs(static_cast<double>(zf));
    if (a == 0.0) return sign;
    if (a > 57344.0) return sign | 0x7B;

    if (a >= std::ldexp(1.0, -14)) {
        // Normal target: significand in [1,2), 2-bit mantissa, RNE.
        int e = 0;
        const double frac = std::frexp(a, &e);  // frac in [0.5, 1)
        // a = frac * 2^e = (2*frac) * 2^(e-1); significand s = 2*frac in [1,2)
        const int ee = e - 1;                   // unbiased exponent
        const double s = 2.0 * frac;            // [1, 2)
        const double m_real = (s - 1.0) * 4.0;  // [0, 4)
        const double fl = std::floor(m_real);
        const double rem = m_real - fl;
        long long m = static_cast<long long>(fl);
        if (rem > 0.5) {
            m += 1;
        } else if (rem == 0.5) {
            if (m % 2 != 0) m += 1;
        }
        int e_out = ee;
        if (m == 4) {
            m = 0;
            e_out += 1;
        }
        if (e_out > 15 || (e_out == 15 && m > 3)) {
            return sign | 0x7B;  // beyond max finite (defensive; a<=57344)
        }
        return static_cast<uint8_t>(
            sign | ((e_out + 15) << 2) | static_cast<int>(m));
    }

    // Subnormal target: round a / 2^-16 to an integer in [0, 4], RNE.
    const double t = a * 65536.0;
    const double fl = std::floor(t);
    const double rem = t - fl;
    long long i = static_cast<long long>(fl);
    if (rem > 0.5) {
        i += 1;
    } else if (rem == 0.5) {
        if (i % 2 != 0) i += 1;
    }
    if (i >= 4) return static_cast<uint8_t>(sign | 0x04);
    return static_cast<uint8_t>(sign | static_cast<int>(i));
}

struct StvOracleToken {
    // Per KV head: scale exps + quantized bytes.
    std::vector<int8_t> k_exp;   // [Hkv]
    std::vector<int8_t> v_exp;   // [Hkv]
    std::vector<uint8_t> k_byte; // [Hkv * D]
    std::vector<uint8_t> v_byte; // [Hkv * D]
};

struct StvOracleState {
    StvProblemSpec spec;
    int max_lp = 0;

    // Committed cache as token list per sequence (bytes are what counts;
    // page ids are tracked separately for the page-state checksum).
    std::vector<std::vector<StvOracleToken>> committed;  // [B][tokens]
    std::vector<std::vector<int32_t>> page_table;        // [B][max_lp]
    std::vector<uint8_t> page_used;                      // [max_pages]
    std::vector<int32_t> seq_len;
    int32_t step_counter = 0;
    int32_t total_allocs = 0;
    int32_t total_frees = 0;

    void init(const StvProblemSpec& s) {
        if (!stv_validate_problem_spec(&s)) {
            throw std::runtime_error("oracle: invalid problem spec");
        }
        spec = s;
        max_lp = stv_max_logical_pages(spec.max_seq_len, spec.page_size);
        reset();
    }

    void reset() {
        committed.assign(spec.B, {});
        page_table.assign(spec.B, std::vector<int32_t>(max_lp, -1));
        page_used.assign(spec.max_pages, 0);
        seq_len.assign(spec.B, 0);
        step_counter = 0;
        total_allocs = 0;
        total_frees = 0;
    }

    int alloc_lowest() {
        for (int p = 0; p < spec.max_pages; ++p) {
            if (!page_used[p]) {
                page_used[p] = 1;
                total_allocs += 1;
                return p;
            }
        }
        throw std::runtime_error("oracle: page pool exhausted");
    }

    // NORMATIVE quantizer for one head vector.
    void quantize_vec(const float* x, int D, int8_t* exp_out, uint8_t* bytes_out) {
        float amax = 0.0f;
        for (int d = 0; d < D; ++d) {
            const float a = std::fabs(x[d]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) {
            *exp_out = 0;
            for (int d = 0; d < D; ++d) {
                uint32_t bits;
                std::memcpy(&bits, &x[d], sizeof(bits));
                bytes_out[d] = (bits & 0x80000000u) ? 0x80 : 0x00;
            }
            return;
        }
        uint32_t bits;
        std::memcpy(&bits, &amax, sizeof(bits));
        const int kfloor = static_cast<int>((bits >> 23) & 0xFF) - 127;
        int se = kfloor - 15;
        if (se < -110) se = -110;
        if (se > 110) se = 110;
        *exp_out = static_cast<int8_t>(se);
        for (int d = 0; d < D; ++d) {
            const float z = std::ldexp(x[d], -se);  // exact
            bytes_out[d] = stv_oracle_e5m2_encode(z);
        }
    }

    StvOracleToken quantize_token(const float* k, const float* v) {
        const int Hkv = spec.Hkv;
        const int D = spec.D;
        StvOracleToken tok;
        tok.k_exp.resize(Hkv);
        tok.v_exp.resize(Hkv);
        tok.k_byte.resize((size_t)Hkv * D);
        tok.v_byte.resize((size_t)Hkv * D);
        for (int h = 0; h < Hkv; ++h) {
            quantize_vec(k + (size_t)h * D, D, &tok.k_exp[h], tok.k_byte.data() + (size_t)h * D);
            quantize_vec(v + (size_t)h * D, D, &tok.v_exp[h], tok.v_byte.data() + (size_t)h * D);
        }
        return tok;
    }

    static uint64_t token_fold(uint64_t h, const StvOracleToken& tok, int Hkv, int D) {
        for (int hh = 0; hh < Hkv; ++hh) {
            h = stv_oracle_fnv_byte(h, static_cast<uint8_t>(tok.k_exp[hh]));
            for (int d = 0; d < D; ++d) {
                h = stv_oracle_fnv_byte(h, tok.k_byte[(size_t)hh * D + d]);
            }
            h = stv_oracle_fnv_byte(h, static_cast<uint8_t>(tok.v_exp[hh]));
            for (int d = 0; d < D; ++d) {
                h = stv_oracle_fnv_byte(h, tok.v_byte[(size_t)hh * D + d]);
            }
        }
        return h;
    }

    void step(
        const StvRunSpec& run,
        const StvHostInputsView& in,
        StvExpected* expected) {
        if (!stv_validate_run_spec(&run, &spec)) {
            throw std::runtime_error("oracle: invalid run spec");
        }
        const int A = run.active_count;
        const int N = spec.max_nodes;
        const int Hq = spec.Hq;
        const int Hkv = spec.Hkv;
        const int D = spec.D;
        const int P = spec.page_size;
        const int group = Hq / Hkv;

        step_counter += 1;

        // Phase 2: scratch allocation (row order).
        std::vector<std::vector<int>> scratch_pages(A);
        for (int a = 0; a < A; ++a) {
            const int n = in.node_count[a];
            const int npg = stv_ceil_div_int(n, P);
            for (int g = 0; g < npg; ++g) {
                scratch_pages[a].push_back(alloc_lowest());
            }
        }

        // Phase 3: quantize nodes + bonus.
        std::vector<std::vector<StvOracleToken>> nodes(A);
        std::vector<StvOracleToken> bonus(A);
        for (int a = 0; a < A; ++a) {
            const int n = in.node_count[a];
            nodes[a].reserve(n);
            for (int i = 0; i < n; ++i) {
                const float* k =
                    in.draft_k + (((size_t)a * N + i) * Hkv) * (size_t)D;
                const float* v =
                    in.draft_v + (((size_t)a * N + i) * Hkv) * (size_t)D;
                nodes[a].push_back(quantize_token(k, v));
            }
            bonus[a] = quantize_token(
                in.bonus_k + (size_t)a * Hkv * D,
                in.bonus_v + (size_t)a * Hkv * D);
        }

        // Phase 4: attention.
        expected->y.assign((size_t)A * N * Hq * D, 0.0f);
        expected->lse.assign((size_t)A * N * Hq, 0.0f);

        const double scale = 1.0 / std::sqrt(static_cast<double>(D));

        std::vector<int> path;
        std::vector<double> scores;

        for (int a = 0; a < A; ++a) {
            const int b = in.active_seq[a];
            const int n = in.node_count[a];
            const int L = seq_len[b];

            for (int i = 0; i < n; ++i) {
                // Build ancestor path root..i.
                path.clear();
                int cur = i;
                while (cur != -1) {
                    path.push_back(cur);
                    cur = in.parent[(size_t)a * N + cur];
                }
                std::vector<int> rpath(path.rbegin(), path.rend());
                const int plen = static_cast<int>(rpath.size());
                const int total_tok = L + plen;

                for (int hq = 0; hq < Hq; ++hq) {
                    const int kvh = hq / group;
                    const float* qv =
                        in.q + (((size_t)a * N + i) * Hq + hq) * (size_t)D;

                    scores.assign(total_tok, 0.0);
                    double m = -1.0e300;
                    for (int t = 0; t < total_tok; ++t) {
                        const StvOracleToken& tok =
                            t < L ? committed[b][t] : nodes[a][rpath[t - L]];
                        const double se = std::ldexp(1.0, tok.k_exp[kvh]);
                        double s = 0.0;
                        for (int d = 0; d < D; ++d) {
                            s += static_cast<double>(qv[d]) *
                                 (stv_oracle_e5m2_decode(
                                      tok.k_byte[(size_t)kvh * D + d]) * se);
                        }
                        s *= scale;
                        scores[t] = s;
                        if (s > m) m = s;
                    }

                    double l = 0.0;
                    std::vector<double> acc(D, 0.0);
                    for (int t = 0; t < total_tok; ++t) {
                        const double w = std::exp(scores[t] - m);
                        l += w;
                        const StvOracleToken& tok =
                            t < L ? committed[b][t] : nodes[a][rpath[t - L]];
                        const double se = std::ldexp(1.0, tok.v_exp[kvh]);
                        for (int d = 0; d < D; ++d) {
                            acc[d] += w *
                                (stv_oracle_e5m2_decode(
                                     tok.v_byte[(size_t)kvh * D + d]) * se);
                        }
                    }
                    const size_t y_base =
                        (((size_t)a * N + i) * Hq + hq) * (size_t)D;
                    for (int d = 0; d < D; ++d) {
                        expected->y[y_base + d] =
                            static_cast<float>(acc[d] / l);
                    }
                    expected->lse[((size_t)a * N + i) * Hq + hq] =
                        static_cast<float>(m + std::log(l));
                }
            }
        }

        // Phase 5: verification.
        expected->accepted_tail.assign(A, -1);
        expected->accepted_len.assign(A, 0);
        std::vector<std::vector<int>> commit_path(A);
        for (int a = 0; a < A; ++a) {
            const int n = in.node_count[a];
            std::vector<bool> acc(n, false);
            std::vector<int> depth(n, 0);
            int best = -1;
            int best_depth = 0;
            for (int i = 0; i < n; ++i) {
                const uint64_t sig = token_fold(
                    kStvOracleFnvBasis, nodes[a][i], Hkv, D);
                const int par = in.parent[(size_t)a * N + i];
                const bool par_ok = (par == -1) ? true : acc[par];
                acc[i] = par_ok && (sig == in.target_sig[(size_t)a * N + i]);
                depth[i] = (par == -1 ? 1 : depth[par] + 1);
                if (acc[i] && depth[i] > best_depth) {
                    best_depth = depth[i];
                    best = i;
                }
            }
            expected->accepted_tail[a] = best;
            expected->accepted_len[a] = best < 0 ? 0 : best_depth;
            if (best >= 0) {
                std::vector<int> rev;
                int cur = best;
                while (cur != -1) {
                    rev.push_back(cur);
                    cur = in.parent[(size_t)a * N + cur];
                }
                commit_path[a].assign(rev.rbegin(), rev.rend());
            }
        }

        // Phase 6: commit (row order, scratch still held).
        for (int a = 0; a < A; ++a) {
            const int b = in.active_seq[a];
            for (size_t ci = 0; ci <= commit_path[a].size(); ++ci) {
                const StvOracleToken& tok =
                    ci < commit_path[a].size()
                        ? nodes[a][commit_path[a][ci]]
                        : bonus[a];
                const int pos = seq_len[b];
                if (pos >= spec.max_seq_len) {
                    throw std::runtime_error("oracle: committed overflow");
                }
                const int lp = pos / P;
                if (page_table[b][lp] < 0) {
                    page_table[b][lp] = alloc_lowest();
                }
                committed[b].push_back(tok);
                seq_len[b] += 1;
            }
        }

        // Phase 7: release scratch.
        for (int a = 0; a < A; ++a) {
            for (int p : scratch_pages[a]) {
                page_used[p] = 0;
                total_frees += 1;
            }
        }

        // Phase 8: outputs.
        expected->seq_len = seq_len;
        expected->kv_hash.assign(spec.B, 0);
        for (int b = 0; b < spec.B; ++b) {
            uint64_t h = kStvOracleFnvBasis;
            for (const StvOracleToken& tok : committed[b]) {
                h = token_fold(h, tok, Hkv, D);
            }
            h = stv_oracle_fnv_i32(h, seq_len[b]);
            expected->kv_hash[b] = h;
        }

        uint64_t ph = kStvOracleFnvBasis;
        ph = stv_oracle_fnv_i32(ph, spec.B);
        ph = stv_oracle_fnv_i32(ph, spec.Hq);
        ph = stv_oracle_fnv_i32(ph, spec.Hkv);
        ph = stv_oracle_fnv_i32(ph, spec.D);
        ph = stv_oracle_fnv_i32(ph, spec.page_size);
        ph = stv_oracle_fnv_i32(ph, spec.max_nodes);
        ph = stv_oracle_fnv_i32(ph, spec.max_seq_len);
        ph = stv_oracle_fnv_i32(ph, spec.max_pages);
        ph = stv_oracle_fnv_i32(ph, step_counter);
        int used = 0;
        for (int p = 0; p < spec.max_pages; ++p) used += page_used[p] ? 1 : 0;
        for (int b = 0; b < spec.B; ++b) {
            ph = stv_oracle_fnv_i32(ph, seq_len[b]);
            const int n_lp = stv_ceil_div_int(seq_len[b], P);
            ph = stv_oracle_fnv_i32(ph, n_lp);
            for (int lp = 0; lp < n_lp; ++lp) {
                ph = stv_oracle_fnv_i32(ph, page_table[b][lp]);
            }
        }
        const int freep = spec.max_pages - used;
        ph = stv_oracle_fnv_i32(ph, total_allocs);
        ph = stv_oracle_fnv_i32(ph, total_frees);
        ph = stv_oracle_fnv_i32(ph, freep);
        expected->page_state_checksum = ph;
        expected->free_pages = freep;
        expected->total_allocs = total_allocs;
        expected->total_frees = total_frees;
    }
};

static inline bool stv_check_all_outputs(
    const StvRunSpec& run,
    const StvProblemSpec& spec,
    const StvExpected& expected,
    const StvHostOutputsView& got,
    const int32_t* node_count,   // host copy, for pad checking
    std::string* error) {
    const int A = run.active_count;
    const int N = spec.max_nodes;
    const int Hq = spec.Hq;
    const int D = spec.D;

    for (int a = 0; a < A; ++a) {
        const int n = node_count[a];
        for (int i = 0; i < N; ++i) {
            const bool pad = i >= n;
            for (int hq = 0; hq < Hq; ++hq) {
                const size_t li = ((size_t)a * N + i) * Hq + hq;
                if (pad) {
                    if (got.lse[li] != 0.0f) {
                        if (error) {
                            std::ostringstream oss;
                            oss << "pad lse not exactly zero at a=" << a
                                << " node=" << i << " hq=" << hq;
                            *error = oss.str();
                        }
                        return false;
                    }
                } else {
                    const float e = expected.lse[li];
                    const float g = got.lse[li];
                    const float diff = std::fabs(g - e);
                    const float tol = STV_LSE_ATOL + STV_LSE_RTOL * std::fabs(e);
                    if (!(diff <= tol)) {
                        if (error) {
                            std::ostringstream oss;
                            oss << "lse mismatch at a=" << a << " node=" << i
                                << " hq=" << hq << ": got " << g
                                << ", expected " << e;
                            *error = oss.str();
                        }
                        return false;
                    }
                }
                for (int d = 0; d < D; ++d) {
                    const size_t yi = li * (size_t)D + d;
                    if (pad) {
                        if (got.y[yi] != 0.0f) {
                            if (error) {
                                std::ostringstream oss;
                                oss << "pad y not exactly zero at a=" << a
                                    << " node=" << i << " flat=" << yi;
                                *error = oss.str();
                            }
                            return false;
                        }
                    } else {
                        const float e = expected.y[yi];
                        const float g = got.y[yi];
                        const float diff = std::fabs(g - e);
                        const float tol = STV_Y_ATOL + STV_Y_RTOL * std::fabs(e);
                        if (!(diff <= tol)) {
                            if (error) {
                                std::ostringstream oss;
                                oss << "y mismatch at a=" << a << " node=" << i
                                    << " hq=" << hq << " d=" << d << ": got "
                                    << g << ", expected " << e << ", diff "
                                    << diff << ", tol " << tol;
                                *error = oss.str();
                            }
                            return false;
                        }
                    }
                }
            }
        }

        if (got.accepted_tail[a] != expected.accepted_tail[a]) {
            if (error) {
                std::ostringstream oss;
                oss << "accepted_tail mismatch at a=" << a << ": got "
                    << got.accepted_tail[a] << ", expected "
                    << expected.accepted_tail[a];
                *error = oss.str();
            }
            return false;
        }
        if (got.accepted_len[a] != expected.accepted_len[a]) {
            if (error) {
                std::ostringstream oss;
                oss << "accepted_len mismatch at a=" << a << ": got "
                    << got.accepted_len[a] << ", expected "
                    << expected.accepted_len[a];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int b = 0; b < spec.B; ++b) {
        if (got.seq_len[b] != expected.seq_len[b]) {
            if (error) {
                std::ostringstream oss;
                oss << "seq_len mismatch at b=" << b << ": got "
                    << got.seq_len[b] << ", expected " << expected.seq_len[b];
                *error = oss.str();
            }
            return false;
        }
        if (got.kv_hash[b] != expected.kv_hash[b]) {
            if (error) {
                std::ostringstream oss;
                oss << "kv_hash mismatch at b=" << b << ": got 0x" << std::hex
                    << got.kv_hash[b] << ", expected 0x" << expected.kv_hash[b];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.page_state_checksum[0] != expected.page_state_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "page_state_checksum mismatch: got 0x" << std::hex
                << got.page_state_checksum[0] << ", expected 0x"
                << expected.page_state_checksum;
            *error = oss.str();
        }
        return false;
    }
    if (got.free_pages[0] != expected.free_pages) {
        if (error) {
            std::ostringstream oss;
            oss << "free_pages mismatch: got " << got.free_pages[0]
                << ", expected " << expected.free_pages;
            *error = oss.str();
        }
        return false;
    }
    if (got.total_allocs[0] != expected.total_allocs) {
        if (error) {
            std::ostringstream oss;
            oss << "total_allocs mismatch: got " << got.total_allocs[0]
                << ", expected " << expected.total_allocs;
            *error = oss.str();
        }
        return false;
    }
    if (got.total_frees[0] != expected.total_frees) {
        if (error) {
            std::ostringstream oss;
            oss << "total_frees mismatch: got " << got.total_frees[0]
                << ", expected " << expected.total_frees;
            *error = oss.str();
        }
        return false;
    }

    return true;
}

/*
Usage:
  1. Construct StvOracleState with the same StvProblemSpec as the solution.
  2. For every solution_run call, call step(run, host_inputs, &expected) and
     compare with stv_check_all_outputs.
  3. reset() mirrors solution_reset.

Test authoring note: to build target_sig values, replicate the oracle
quantizer + TOKEN FOLD on the host (token_fold(quantize_token(...))); pass
the exact value to accept a node, or any different value to reject it.
*/

#endif  // SPEC_TREE_VERIFY_E5M2_KV_ORACLE_HPP_
