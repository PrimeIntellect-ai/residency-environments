// ============================================================================
// file: paged_kv_decode_oracle.hpp
// Independent CPU oracle/state model + check helpers.
// ============================================================================

#ifndef PAGED_KV_DECODE_ORACLE_HPP_
#define PAGED_KV_DECODE_ORACLE_HPP_

#include "paged_kv_decode_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct PkdHostInputsView {
    const int32_t* active_seq;
    const int32_t* new_token_count;
    const float* new_k;
    const float* new_v;
    const float* new_scale;
    const float* q;
};

struct PkdHostOutputsView {
    const float* y;
    const int32_t* lengths;
    const uint64_t* state_checksum;
};

struct PkdExpected {
    std::vector<float> y;
    std::vector<int32_t> lengths;
    uint64_t state_checksum = 0;
};

static inline uint64_t pkd_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void pkd_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = pkd_oracle_fnv_byte(v, p[i]);
    }
    *h = v;
}

static inline float pkd_oracle_sanitize_scale(float s) {
    return s > 1.0e-6f ? s : 1.0f;
}

static inline int8_t pkd_oracle_quantize(float x, float scale) {
    const float z = x / scale;
    int qi = 0;

    if (z >= 0.0f) {
        qi = static_cast<int>(std::floor(z + 0.5f));
    } else {
        qi = static_cast<int>(std::ceil(z - 0.5f));
    }

    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;

    return static_cast<int8_t>(qi);
}

struct PkdOracleState {
    PkdProblemSpec spec{};
    int pages_per_seq = 0;
    int next_page = 0;

    std::vector<int8_t> k_cache;
    std::vector<int8_t> v_cache;
    std::vector<float> page_scale;
    std::vector<int32_t> page_table;
    std::vector<int32_t> lengths;

    void init(const PkdProblemSpec& s) {
        spec = s;
        pages_per_seq = pkd_pages_per_seq(spec.max_seq_len, spec.page_size);

        const size_t cache_elems =
            (size_t)spec.max_pages *
            (size_t)spec.Hkv *
            (size_t)spec.page_size *
            (size_t)spec.D;

        k_cache.assign(cache_elems, 0);
        v_cache.assign(cache_elems, 0);
        page_scale.assign((size_t)spec.max_pages * (size_t)spec.Hkv, 1.0f);
        page_table.assign((size_t)spec.B * (size_t)pages_per_seq, -1);
        lengths.assign((size_t)spec.B, 0);
        next_page = 0;
    }

    void reset() {
        std::fill(k_cache.begin(), k_cache.end(), static_cast<int8_t>(0));
        std::fill(v_cache.begin(), v_cache.end(), static_cast<int8_t>(0));
        std::fill(page_scale.begin(), page_scale.end(), 1.0f);
        std::fill(page_table.begin(), page_table.end(), -1);
        std::fill(lengths.begin(), lengths.end(), 0);
        next_page = 0;
    }

    size_t cache_index(int page, int h, int offset, int d) const {
        return (((size_t)page * (size_t)spec.Hkv + (size_t)h) *
                (size_t)spec.page_size + (size_t)offset) *
                (size_t)spec.D + (size_t)d;
    }

    uint64_t checksum() const {
        uint64_t h = 1469598103934665603ULL;

        pkd_oracle_fnv_bytes(&h, &next_page, sizeof(int32_t));

        pkd_oracle_fnv_bytes(
            &h,
            lengths.data(),
            sizeof(int32_t) * (size_t)spec.B);

        pkd_oracle_fnv_bytes(
            &h,
            page_table.data(),
            sizeof(int32_t) * (size_t)spec.B * (size_t)pages_per_seq);

        int used_pages = next_page;
        if (used_pages < 0) used_pages = 0;
        if (used_pages > spec.max_pages) used_pages = spec.max_pages;

        const size_t page_cache_elems =
            (size_t)spec.Hkv *
            (size_t)spec.page_size *
            (size_t)spec.D;

        for (int p = 0; p < used_pages; ++p) {
            pkd_oracle_fnv_bytes(
                &h,
                &page_scale[(size_t)p * (size_t)spec.Hkv],
                sizeof(float) * (size_t)spec.Hkv);

            pkd_oracle_fnv_bytes(
                &h,
                &k_cache[(size_t)p * page_cache_elems],
                sizeof(int8_t) * page_cache_elems);

            pkd_oracle_fnv_bytes(
                &h,
                &v_cache[(size_t)p * page_cache_elems],
                sizeof(int8_t) * page_cache_elems);
        }

        return h;
    }

    void step(
        const PkdRunSpec& run,
        const PkdHostInputsView& in,
        PkdExpected* expected) {
        const int B = spec.B;
        const int Hq = spec.Hq;
        const int Hkv = spec.Hkv;
        const int D = spec.D;
        const int P = spec.page_size;
        const int active_count = run.active_count;

        expected->y.assign((size_t)active_count * (size_t)Hq * (size_t)D, 0.0f);

        for (int a = 0; a < active_count; ++a) {
            const int seq = in.active_seq[a];
            if (seq < 0 || seq >= B) {
                continue;
            }

            int ntokens = in.new_token_count[a];
            if (ntokens < 0) ntokens = 0;
            if (ntokens > PKD_MAX_NEW_TOKENS) ntokens = PKD_MAX_NEW_TOKENS;

            for (int nt = 0; nt < ntokens; ++nt) {
                const int old_len = lengths[(size_t)seq];
                if (old_len >= spec.max_seq_len) {
                    continue;
                }

                const int page_slot = old_len / P;
                const int page_offset = old_len - page_slot * P;
                int page = -1;

                if (page_offset == 0) {
                    page = next_page;
                    ++next_page;

                    if (page >= spec.max_pages) {
                        continue;
                    }

                    page_table[(size_t)seq * (size_t)pages_per_seq + (size_t)page_slot] = page;

                    for (int h = 0; h < Hkv; ++h) {
                        const size_t scale_idx =
                            ((size_t)a * (size_t)PKD_MAX_NEW_TOKENS + (size_t)nt) *
                            (size_t)Hkv + (size_t)h;
                        page_scale[(size_t)page * (size_t)Hkv + (size_t)h] =
                            pkd_oracle_sanitize_scale(in.new_scale[scale_idx]);
                    }
                } else {
                    page = page_table[(size_t)seq * (size_t)pages_per_seq + (size_t)page_slot];
                }

                if (page < 0 || page >= spec.max_pages) {
                    continue;
                }

                for (int h = 0; h < Hkv; ++h) {
                    const float scale = page_scale[(size_t)page * (size_t)Hkv + (size_t)h];

                    for (int d = 0; d < D; ++d) {
                        const size_t input_idx =
                            ((size_t)a * (size_t)PKD_MAX_NEW_TOKENS + (size_t)nt) *
                            (size_t)Hkv * (size_t)D +
                            (size_t)h * (size_t)D +
                            (size_t)d;

                        const size_t cache_idx = cache_index(page, h, page_offset, d);

                        k_cache[cache_idx] = pkd_oracle_quantize(in.new_k[input_idx], scale);
                        v_cache[cache_idx] = pkd_oracle_quantize(in.new_v[input_idx], scale);
                    }
                }

                lengths[(size_t)seq] = old_len + 1;
            }
        }

        for (int a = 0; a < active_count; ++a) {
            const int seq = in.active_seq[a];
            if (seq < 0 || seq >= B) {
                continue;
            }

            const int len = lengths[(size_t)seq];
            int W = run.window_size;
            if (W > spec.max_seq_len) W = spec.max_seq_len;

            int start = len - W;
            if (start < 0) start = 0;

            if (len <= start) {
                continue;
            }

            const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
            const int group = Hq / Hkv;

            for (int hq = 0; hq < Hq; ++hq) {
                const int kvh = hq / group;
                float max_logit = -std::numeric_limits<float>::infinity();

                for (int pos = start; pos < len; ++pos) {
                    const int page_slot = pos / P;
                    const int offset = pos - page_slot * P;
                    const int page =
                        page_table[(size_t)seq * (size_t)pages_per_seq + (size_t)page_slot];
                    const float scale =
                        page_scale[(size_t)page * (size_t)Hkv + (size_t)kvh];

                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        const size_t q_idx =
                            ((size_t)a * (size_t)Hq + (size_t)hq) *
                            (size_t)D + (size_t)d;
                        const size_t k_idx = cache_index(page, kvh, offset, d);
                        dot += in.q[q_idx] *
                               (static_cast<float>(k_cache[k_idx]) * scale);
                    }

                    const float logit = dot * inv_sqrt_d;
                    if (logit > max_logit) {
                        max_logit = logit;
                    }
                }

                float denom = 0.0f;
                std::vector<float> acc((size_t)D, 0.0f);

                for (int pos = start; pos < len; ++pos) {
                    const int page_slot = pos / P;
                    const int offset = pos - page_slot * P;
                    const int page =
                        page_table[(size_t)seq * (size_t)pages_per_seq + (size_t)page_slot];
                    const float scale =
                        page_scale[(size_t)page * (size_t)Hkv + (size_t)kvh];

                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        const size_t q_idx =
                            ((size_t)a * (size_t)Hq + (size_t)hq) *
                            (size_t)D + (size_t)d;
                        const size_t k_idx = cache_index(page, kvh, offset, d);
                        dot += in.q[q_idx] *
                               (static_cast<float>(k_cache[k_idx]) * scale);
                    }

                    const float w = std::exp(dot * inv_sqrt_d - max_logit);
                    denom += w;

                    for (int d = 0; d < D; ++d) {
                        const size_t v_idx = cache_index(page, kvh, offset, d);
                        acc[(size_t)d] += w * (static_cast<float>(v_cache[v_idx]) * scale);
                    }
                }

                for (int d = 0; d < D; ++d) {
                    const size_t y_idx =
                        ((size_t)a * (size_t)Hq + (size_t)hq) *
                        (size_t)D + (size_t)d;
                    expected->y[y_idx] = denom > 0.0f ? acc[(size_t)d] / denom : 0.0f;
                }
            }
        }

        expected->lengths = lengths;
        expected->state_checksum = checksum();
    }
};

static inline bool pkd_check_y(
    const PkdRunSpec& run,
    const PkdProblemSpec& spec,
    const PkdExpected& expected,
    const PkdHostOutputsView& got,
    std::string* error) {
    const size_t total =
        (size_t)run.active_count * (size_t)spec.Hq * (size_t)spec.D;

    for (size_t i = 0; i < total; ++i) {
        const float exp_v = expected.y[i];
        const float got_v = got.y[i];

        const float diff = std::fabs(got_v - exp_v);
        const float tol = PKD_Y_ATOL + PKD_Y_RTOL * std::fabs(exp_v);

        if (!(diff <= tol)) {
            if (error) {
                const size_t d = i % (size_t)spec.D;
                const size_t tmp = i / (size_t)spec.D;
                const size_t hq = tmp % (size_t)spec.Hq;
                const size_t a = tmp / (size_t)spec.Hq;

                std::ostringstream oss;
                oss << "y mismatch at active=" << a
                    << ", hq=" << hq
                    << ", d=" << d
                    << ": got " << got_v
                    << ", expected " << exp_v
                    << ", diff " << diff
                    << ", tol " << tol;
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool pkd_check_lengths(
    const PkdProblemSpec& spec,
    const PkdExpected& expected,
    const PkdHostOutputsView& got,
    std::string* error) {
    for (int b = 0; b < spec.B; ++b) {
        if (got.lengths[b] != expected.lengths[(size_t)b]) {
            if (error) {
                std::ostringstream oss;
                oss << "length mismatch at seq=" << b
                    << ": got " << got.lengths[b]
                    << ", expected " << expected.lengths[(size_t)b];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool pkd_check_checksum(
    const PkdExpected& expected,
    const PkdHostOutputsView& got,
    std::string* error) {
    if (got.state_checksum[0] != expected.state_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "state checksum mismatch: got 0x"
                << std::hex << got.state_checksum[0]
                << ", expected 0x" << expected.state_checksum;
            *error = oss.str();
        }
        return false;
    }

    return true;
}

static inline bool pkd_check_all_outputs(
    const PkdRunSpec& run,
    const PkdProblemSpec& spec,
    const PkdExpected& expected,
    const PkdHostOutputsView& got,
    std::string* error) {
    if (!pkd_check_y(run, spec, expected, got, error)) return false;
    if (!pkd_check_lengths(spec, expected, got, error)) return false;
    if (!pkd_check_checksum(expected, got, error)) return false;
    return true;
}

/*
PIPELINE GRADING MODEL SUPPORTED BY THIS ORACLE

A test harness should:

  1. Construct PkdOracleState with the same PkdProblemSpec as the solution.

  2. Call solution_reset and oracle.reset.

  3. Run T decode steps, T roughly 16-32:
       - mutate active_seq each step,
       - use new_token_count in {0,1,2},
       - create sequence-length skew,
       - choose windows that cross page boundaries,
       - include steps with no append but non-empty query.

  4. After every step:
       - compare y with tolerance:
           abs_error <= 2e-3 + 2e-3 * abs(expected)
       - compare lengths exactly,
       - compare state_checksum exactly.

  5. Repeat after reset with:
       - different active batch order,
       - different page pressure,
       - different new-token pattern.

This catches:
  - first-step-only implementations,
  - stale lengths,
  - wrong GQA head mapping,
  - wrong page allocation or page-table update,
  - quantization/layout mismatch,
  - cache overwrite across page boundaries,
  - stale state after reset,
  - sequence-order nondeterminism.
*/

#endif  // PAGED_KV_DECODE_ORACLE_HPP_
