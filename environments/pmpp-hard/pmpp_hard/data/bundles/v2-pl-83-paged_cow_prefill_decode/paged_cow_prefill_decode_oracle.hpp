// ============================================================================
// file: paged_cow_prefill_decode_oracle.hpp
//
// Independent host-side oracle for the paged_cow_prefill_decode contract.
// Pure CPU: bit-exact fp32->fp16 RNE conversion, exact refcounted page-pool
// simulation, double precision attention. Shares no code with any GPU
// implementation.
// ============================================================================

#ifndef PAGED_COW_PREFILL_DECODE_ORACLE_HPP_
#define PAGED_COW_PREFILL_DECODE_ORACLE_HPP_

#include "paged_cow_prefill_decode_common.h"

#include <stdint.h>

#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct CpdHostInputsView {
    const int32_t* active_seq;
    const int32_t* op_kind;
    const int32_t* fork_src;
    const int32_t* new_token_count;
    const float* new_k;
    const float* new_v;
    const float* q;
};

struct CpdHostOutputsView {
    const float* y;
    const float* lse;
    const int32_t* seq_len;
    const uint64_t* kv_hash;
    const uint64_t* page_state_checksum;
    const int32_t* free_pages;
    const int32_t* total_allocs;
    const int32_t* total_frees;
    const int32_t* total_forks;
    const int32_t* total_releases;
};

struct CpdExpected {
    std::vector<float> y;
    std::vector<float> lse;
    std::vector<int32_t> seq_len;
    std::vector<uint64_t> kv_hash;
    uint64_t page_state_checksum = 0;
    int32_t free_pages = 0;
    int32_t total_allocs = 0;
    int32_t total_frees = 0;
    int32_t total_forks = 0;
    int32_t total_releases = 0;
};

static const uint64_t kCpdOracleFnvBasis = 1469598103934665603ULL;
static const uint64_t kCpdOracleFnvPrime = 1099511628211ULL;

static inline uint64_t cpd_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= kCpdOracleFnvPrime;
    return h;
}

static inline uint64_t cpd_oracle_fnv_u16(uint64_t h, uint16_t v) {
    h = cpd_oracle_fnv_byte(h, static_cast<uint8_t>(v & 0xFF));
    h = cpd_oracle_fnv_byte(h, static_cast<uint8_t>((v >> 8) & 0xFF));
    return h;
}

static inline uint64_t cpd_oracle_fnv_i32(uint64_t h, int32_t v) {
    const uint32_t u = static_cast<uint32_t>(v);
    h = cpd_oracle_fnv_byte(h, static_cast<uint8_t>(u & 0xFF));
    h = cpd_oracle_fnv_byte(h, static_cast<uint8_t>((u >> 8) & 0xFF));
    h = cpd_oracle_fnv_byte(h, static_cast<uint8_t>((u >> 16) & 0xFF));
    h = cpd_oracle_fnv_byte(h, static_cast<uint8_t>((u >> 24) & 0xFF));
    return h;
}

// Bit-exact IEEE binary16 round-to-nearest-even conversion (== CUDA
// __float2half_rn for finite inputs within range).
static inline uint16_t cpd_oracle_f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
    const uint32_t absx = x & 0x7FFFFFFFu;

    if (absx >= 0x47800000u) {
        // >= 65536 (or inf/nan): saturates to inf; unreachable per input
        // bounds, kept for totality.
        return sign | 0x7C00u;
    }
    if (absx >= 0x38800000u) {
        // Normal half range [2^-14, 65536): rebias and round 13 bits.
        const uint32_t sub = absx - 0x38000000u;
        uint16_t h = static_cast<uint16_t>(sub >> 13);
        const uint32_t rest = sub & 0x1FFFu;
        if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) h++;
        return sign | h;
    }
    if (absx < 0x33000000u) {
        // < 2^-25: rounds to (signed) zero (2^-25 itself ties to even 0 and
        // is handled by the subnormal path below producing 0).
        return sign;
    }
    // Subnormal half: value = m * 2^(e-23), unit 2^-24.
    const int e = static_cast<int>(absx >> 23) - 127;  // in [-25, -15]
    const uint32_t m = (absx & 0x7FFFFFu) | 0x800000u;
    const uint32_t sh = static_cast<uint32_t>(-e - 1);  // in [14, 24]
    uint32_t i = m >> sh;
    const uint32_t rest = m & ((1u << sh) - 1u);
    const uint32_t half = 1u << (sh - 1);
    if (rest > half || (rest == half && (i & 1u))) i++;
    return static_cast<uint16_t>(sign | i);  // carry into 0x0400 is correct
}

static inline double cpd_oracle_f16_to_f64(uint16_t h) {
    const int sign = (h & 0x8000u) ? -1 : 1;
    const int E = (h >> 10) & 0x1F;
    const int M = h & 0x3FF;
    if (E == 0) {
        return sign * static_cast<double>(M) * std::ldexp(1.0, -24);
    }
    if (E == 31) {
        return sign * 1.0e300;  // inf; unreachable
    }
    return sign * std::ldexp(1.0 + M / 1024.0, E - 15);
}

struct CpdOracleToken {
    std::vector<uint16_t> k;  // [Hkv * D]
    std::vector<uint16_t> v;  // [Hkv * D]
};

struct CpdOracleState {
    CpdProblemSpec spec;
    int max_lp = 0;

    std::vector<std::vector<CpdOracleToken>> tokens;  // [B][seq_len] logical
    std::vector<std::vector<int32_t>> page_table;     // [B][max_lp]
    std::vector<int32_t> refcount;                    // [max_pages]
    std::vector<int32_t> seq_len;
    int32_t step_counter = 0;
    int32_t total_allocs = 0;
    int32_t total_frees = 0;
    int32_t total_forks = 0;
    int32_t total_releases = 0;

    void init(const CpdProblemSpec& s) {
        if (!cpd_validate_problem_spec(&s)) {
            throw std::runtime_error("oracle: invalid problem spec");
        }
        spec = s;
        max_lp = cpd_max_logical_pages(spec.max_seq_len, spec.page_size);
        reset();
    }

    void reset() {
        tokens.assign(spec.B, {});
        page_table.assign(spec.B, std::vector<int32_t>(max_lp, -1));
        refcount.assign(spec.max_pages, 0);
        seq_len.assign(spec.B, 0);
        step_counter = 0;
        total_allocs = 0;
        total_frees = 0;
        total_forks = 0;
        total_releases = 0;
    }

    int alloc_lowest() {
        for (int p = 0; p < spec.max_pages; ++p) {
            if (refcount[p] == 0) {
                refcount[p] = 1;
                total_allocs += 1;
                return p;
            }
        }
        throw std::runtime_error("oracle: page pool exhausted");
    }

    CpdOracleToken convert_token(const float* k, const float* v) {
        const int n = spec.Hkv * spec.D;
        CpdOracleToken tok;
        tok.k.resize(n);
        tok.v.resize(n);
        for (int i = 0; i < n; ++i) {
            tok.k[i] = cpd_oracle_f32_to_f16(k[i]);
            tok.v[i] = cpd_oracle_f32_to_f16(v[i]);
        }
        return tok;
    }

    static uint64_t token_fold(
        uint64_t h, const CpdOracleToken& tok, int Hkv, int D) {
        for (int hh = 0; hh < Hkv; ++hh) {
            for (int d = 0; d < D; ++d) {
                h = cpd_oracle_fnv_u16(h, tok.k[(size_t)hh * D + d]);
            }
            for (int d = 0; d < D; ++d) {
                h = cpd_oracle_fnv_u16(h, tok.v[(size_t)hh * D + d]);
            }
        }
        return h;
    }

    void step(
        const CpdRunSpec& run,
        const CpdHostInputsView& in,
        CpdExpected* expected) {
        if (!cpd_validate_run_spec(&run, &spec)) {
            throw std::runtime_error("oracle: invalid run spec");
        }
        const int A = run.active_count;
        const int C = spec.max_chunk;
        const int Hq = spec.Hq;
        const int Hkv = spec.Hkv;
        const int D = spec.D;
        const int P = spec.page_size;
        const int group = Hq / Hkv;

        step_counter += 1;

        // Phase 2: releases (row order).
        for (int a = 0; a < A; ++a) {
            if (in.op_kind[a] != CPD_OP_RELEASE) continue;
            const int b = in.active_seq[a];
            if (seq_len[b] < 1) throw std::runtime_error("oracle: release of empty seq");
            const int n_lp = cpd_ceil_div_int(seq_len[b], P);
            for (int lp = 0; lp < n_lp; ++lp) {
                const int p = page_table[b][lp];
                page_table[b][lp] = -1;
                refcount[p] -= 1;
                if (refcount[p] == 0) total_frees += 1;
                if (refcount[p] < 0) throw std::runtime_error("oracle: refcount underflow");
            }
            seq_len[b] = 0;
            tokens[b].clear();
            total_releases += 1;
        }

        // Phase 3: forks (row order); observe post-release, pre-append state.
        for (int a = 0; a < A; ++a) {
            if (in.op_kind[a] != CPD_OP_FORK_APPEND) continue;
            const int b = in.active_seq[a];
            const int s = in.fork_src[a];
            if (seq_len[b] != 0) throw std::runtime_error("oracle: fork into live seq");
            if (seq_len[s] < 1) throw std::runtime_error("oracle: fork from empty seq");

            seq_len[b] = seq_len[s];
            tokens[b] = tokens[s];
            total_forks += 1;

            const int full_lp = seq_len[s] / P;
            for (int lp = 0; lp < full_lp; ++lp) {
                page_table[b][lp] = page_table[s][lp];
                refcount[page_table[s][lp]] += 1;
            }
            if (seq_len[s] % P != 0) {
                page_table[b][full_lp] = alloc_lowest();
                // Copy of tail bytes is implicit in tokens[b] = tokens[s].
            }
        }

        // Phase 4: appends (row order).
        std::vector<int32_t> base_len(A, 0);
        for (int a = 0; a < A; ++a) {
            const int op = in.op_kind[a];
            if (op == CPD_OP_RELEASE) continue;
            const int b = in.active_seq[a];
            const int cnt = in.new_token_count[a];
            base_len[a] = seq_len[b];
            for (int nt = 0; nt < cnt; ++nt) {
                const int pos = seq_len[b];
                if (pos >= spec.max_seq_len) {
                    throw std::runtime_error("oracle: seq overflow");
                }
                const int lp = pos / P;
                if (page_table[b][lp] < 0) {
                    page_table[b][lp] = alloc_lowest();
                }
                tokens[b].push_back(convert_token(
                    in.new_k + (((size_t)a * C + nt) * Hkv) * (size_t)D,
                    in.new_v + (((size_t)a * C + nt) * Hkv) * (size_t)D));
                seq_len[b] += 1;
            }
        }

        // Phase 5: attention.
        expected->y.assign((size_t)A * C * Hq * D, 0.0f);
        expected->lse.assign((size_t)A * C * Hq, 0.0f);

        const double scale = 1.0 / std::sqrt(static_cast<double>(D));
        std::vector<double> scores;

        for (int a = 0; a < A; ++a) {
            const int op = in.op_kind[a];
            if (op == CPD_OP_RELEASE) continue;
            const int b = in.active_seq[a];
            const int cnt = in.new_token_count[a];

            for (int nt = 0; nt < cnt; ++nt) {
                const int p = base_len[a] + nt;
                const int L = p + 1;

                for (int hq = 0; hq < Hq; ++hq) {
                    const int kvh = hq / group;
                    const float* qv =
                        in.q + (((size_t)a * C + nt) * Hq + hq) * (size_t)D;

                    scores.assign(L, 0.0);
                    double m = -1.0e300;
                    for (int t = 0; t < L; ++t) {
                        const CpdOracleToken& tok = tokens[b][t];
                        double s = 0.0;
                        for (int d = 0; d < D; ++d) {
                            s += static_cast<double>(qv[d]) *
                                 cpd_oracle_f16_to_f64(tok.k[(size_t)kvh * D + d]);
                        }
                        s *= scale;
                        scores[t] = s;
                        if (s > m) m = s;
                    }

                    double l = 0.0;
                    std::vector<double> acc(D, 0.0);
                    for (int t = 0; t < L; ++t) {
                        const double w = std::exp(scores[t] - m);
                        l += w;
                        const CpdOracleToken& tok = tokens[b][t];
                        for (int d = 0; d < D; ++d) {
                            acc[d] += w *
                                cpd_oracle_f16_to_f64(tok.v[(size_t)kvh * D + d]);
                        }
                    }
                    const size_t y_base =
                        (((size_t)a * C + nt) * Hq + hq) * (size_t)D;
                    for (int d = 0; d < D; ++d) {
                        expected->y[y_base + d] = static_cast<float>(acc[d] / l);
                    }
                    expected->lse[((size_t)a * C + nt) * Hq + hq] =
                        static_cast<float>(m + std::log(l));
                }
            }
        }

        // Phase 7: exact outputs.
        expected->seq_len = seq_len;
        expected->kv_hash.assign(spec.B, 0);
        for (int b = 0; b < spec.B; ++b) {
            uint64_t h = kCpdOracleFnvBasis;
            for (const CpdOracleToken& tok : tokens[b]) {
                h = token_fold(h, tok, Hkv, D);
            }
            h = cpd_oracle_fnv_i32(h, seq_len[b]);
            expected->kv_hash[b] = h;
        }

        uint64_t ph = kCpdOracleFnvBasis;
        ph = cpd_oracle_fnv_i32(ph, spec.B);
        ph = cpd_oracle_fnv_i32(ph, spec.Hq);
        ph = cpd_oracle_fnv_i32(ph, spec.Hkv);
        ph = cpd_oracle_fnv_i32(ph, spec.D);
        ph = cpd_oracle_fnv_i32(ph, spec.page_size);
        ph = cpd_oracle_fnv_i32(ph, spec.max_chunk);
        ph = cpd_oracle_fnv_i32(ph, spec.max_seq_len);
        ph = cpd_oracle_fnv_i32(ph, spec.max_pages);
        ph = cpd_oracle_fnv_i32(ph, step_counter);
        for (int b = 0; b < spec.B; ++b) {
            ph = cpd_oracle_fnv_i32(ph, seq_len[b]);
            const int n_lp = cpd_ceil_div_int(seq_len[b], P);
            ph = cpd_oracle_fnv_i32(ph, n_lp);
            for (int lp = 0; lp < n_lp; ++lp) {
                ph = cpd_oracle_fnv_i32(ph, page_table[b][lp]);
            }
        }
        int freep = 0;
        for (int p = 0; p < spec.max_pages; ++p) freep += refcount[p] == 0 ? 1 : 0;
        ph = cpd_oracle_fnv_i32(ph, total_allocs);
        ph = cpd_oracle_fnv_i32(ph, total_frees);
        ph = cpd_oracle_fnv_i32(ph, total_forks);
        ph = cpd_oracle_fnv_i32(ph, total_releases);
        ph = cpd_oracle_fnv_i32(ph, freep);
        expected->page_state_checksum = ph;
        expected->free_pages = freep;
        expected->total_allocs = total_allocs;
        expected->total_frees = total_frees;
        expected->total_forks = total_forks;
        expected->total_releases = total_releases;
    }
};

static inline bool cpd_check_all_outputs(
    const CpdRunSpec& run,
    const CpdProblemSpec& spec,
    const CpdExpected& expected,
    const CpdHostOutputsView& got,
    const int32_t* op_kind,          // host copies, for pad checking
    const int32_t* new_token_count,
    std::string* error) {
    const int A = run.active_count;
    const int C = spec.max_chunk;
    const int Hq = spec.Hq;
    const int D = spec.D;

    for (int a = 0; a < A; ++a) {
        const bool rel = op_kind[a] == CPD_OP_RELEASE;
        const int cnt = rel ? 0 : new_token_count[a];
        for (int nt = 0; nt < C; ++nt) {
            const bool pad = nt >= cnt;
            for (int hq = 0; hq < Hq; ++hq) {
                const size_t li = ((size_t)a * C + nt) * Hq + hq;
                if (pad) {
                    if (got.lse[li] != 0.0f) {
                        if (error) {
                            std::ostringstream oss;
                            oss << "pad lse not exactly zero at a=" << a
                                << " nt=" << nt << " hq=" << hq;
                            *error = oss.str();
                        }
                        return false;
                    }
                } else {
                    const float e = expected.lse[li];
                    const float g = got.lse[li];
                    const float diff = std::fabs(g - e);
                    const float tol = CPD_LSE_ATOL + CPD_LSE_RTOL * std::fabs(e);
                    if (!(diff <= tol)) {
                        if (error) {
                            std::ostringstream oss;
                            oss << "lse mismatch at a=" << a << " nt=" << nt
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
                                    << " nt=" << nt << " flat=" << yi;
                                *error = oss.str();
                            }
                            return false;
                        }
                    } else {
                        const float e = expected.y[yi];
                        const float g = got.y[yi];
                        const float diff = std::fabs(g - e);
                        const float tol = CPD_Y_ATOL + CPD_Y_RTOL * std::fabs(e);
                        if (!(diff <= tol)) {
                            if (error) {
                                std::ostringstream oss;
                                oss << "y mismatch at a=" << a << " nt=" << nt
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

    struct IntCheck {
        const char* name;
        int32_t got;
        int32_t exp;
    };
    const IntCheck checks[] = {
        {"free_pages", got.free_pages[0], expected.free_pages},
        {"total_allocs", got.total_allocs[0], expected.total_allocs},
        {"total_frees", got.total_frees[0], expected.total_frees},
        {"total_forks", got.total_forks[0], expected.total_forks},
        {"total_releases", got.total_releases[0], expected.total_releases},
    };
    for (const IntCheck& c : checks) {
        if (c.got != c.exp) {
            if (error) {
                std::ostringstream oss;
                oss << c.name << " mismatch: got " << c.got
                    << ", expected " << c.exp;
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

/*
Usage:
  1. Construct CpdOracleState with the same CpdProblemSpec as the solution.
  2. For every solution_run call, call step(run, host_inputs, &expected) and
     compare with cpd_check_all_outputs.
  3. reset() mirrors solution_reset.
*/

#endif  // PAGED_COW_PREFILL_DECODE_ORACLE_HPP_
