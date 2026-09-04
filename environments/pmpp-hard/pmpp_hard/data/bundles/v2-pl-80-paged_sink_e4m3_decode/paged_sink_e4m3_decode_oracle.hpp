// ============================================================================
// file: paged_sink_e4m3_decode_oracle.hpp
// Independent CPU oracle/state model + check helpers.
//
// Deliberately different code path from any GPU implementation:
//  - E4M3 encoding is a nearest-value search over the 256-entry decode table
//    (no bit-manipulation rounding).
//  - Quantized K/V are stored per logical position (no physical page layout).
//  - The page allocator is modeled with plain arrays and linear scans.
//  - Attention is a double-precision two-pass softmax.
// ============================================================================

#ifndef PAGED_SINK_E4M3_DECODE_ORACLE_HPP_
#define PAGED_SINK_E4M3_DECODE_ORACLE_HPP_

#include "paged_sink_e4m3_decode_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct PseHostInputsView {
    const int32_t* active_seq;
    const int32_t* new_token_count;
    const float* new_k;
    const float* new_v;
    const float* q;
};

struct PseHostOutputsView {
    const float* y;
    const float* lse;
    const int32_t* seq_len;
    const uint64_t* kv_hash;
    const uint64_t* page_state_checksum;
    const int32_t* free_pages;
    const int32_t* total_allocs;
    const int32_t* total_frees;
};

struct PseExpected {
    std::vector<float> y;
    std::vector<float> lse;
    std::vector<int32_t> seq_len;
    std::vector<uint64_t> kv_hash;
    uint64_t page_state_checksum = 0;
    int32_t free_pages = 0;
    int32_t total_allocs = 0;
    int32_t total_frees = 0;
};

static inline uint64_t pse_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void pse_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = pse_oracle_fnv_byte(v, p[i]);
    }
    *h = v;
}

static inline void pse_oracle_fnv_i32(uint64_t* h, int32_t v) {
    pse_oracle_fnv_bytes(h, &v, sizeof(int32_t));
}

// Exact E4M3 decode of one byte (finite values only; the NaN slot 0x7F/0xFF
// is never stored by a conforming implementation).
static inline double pse_oracle_e4m3_decode(uint8_t byte) {
    const int s = (byte >> 7) & 1;
    const int E = (byte >> 3) & 0xF;
    const int M = byte & 0x7;
    double v;
    if (E == 0) {
        v = std::ldexp(static_cast<double>(M), -9);
    } else {
        v = std::ldexp(1.0 + static_cast<double>(M) / 8.0, E - 7);
    }
    return s ? -v : v;
}

// Nearest-value-search encoder. Ties break toward even mantissa field M.
struct PseE4m3Table {
    // 127 non-negative magnitudes (byte 0x00..0x7E); 0x7F is NaN, skipped.
    double value[127];

    PseE4m3Table() {
        for (int b = 0; b < 127; ++b) {
            value[b] = pse_oracle_e4m3_decode(static_cast<uint8_t>(b));
        }
    }

    uint8_t encode(float zf) const {
        const bool neg = std::signbit(zf);
        const uint8_t sign = neg ? 0x80 : 0x00;
        const double a = std::fabs(static_cast<double>(zf));
        if (a == 0.0) return sign;
        if (a > 448.0) return sign | 0x7E;

        int best = 0;
        double best_dist = std::numeric_limits<double>::infinity();
        for (int b = 0; b < 127; ++b) {
            const double dist = std::fabs(a - value[b]);
            if (dist < best_dist) {
                best = b;
                best_dist = dist;
            } else if (dist == best_dist) {
                // tie: prefer even mantissa field
                const int m_new = b & 7;
                const int m_old = best & 7;
                if ((m_new & 1) == 0 && (m_old & 1) == 1) {
                    best = b;
                }
            }
        }
        return sign | static_cast<uint8_t>(best);
    }
};

struct PseOracleState {
    PseProblemSpec spec{};
    int max_lp = 0;

    // Full signed decode table (256 entries) for the attention hot loop.
    double dec[256];

    // Quantized token storage keyed by logical position (NOT by page).
    // token index = b * max_seq_len + pos; per token: Hkv heads.
    std::vector<uint8_t> k_bytes;   // [B * max_seq_len * Hkv * D]
    std::vector<uint8_t> v_bytes;
    std::vector<int8_t> k_exp;      // [B * max_seq_len * Hkv]
    std::vector<int8_t> v_exp;

    // Page allocator model.
    std::vector<int32_t> page_table;  // [B * max_lp], -1 absent
    std::vector<uint8_t> page_used;   // [max_pages]
    std::vector<int32_t> seq_len;     // [B]
    int32_t step_counter = 0;
    int32_t total_allocs = 0;
    int32_t total_frees = 0;

    PseE4m3Table table;

    void init(const PseProblemSpec& s) {
        spec = s;
        max_lp = pse_max_logical_pages(spec.max_seq_len, spec.page_size);
        for (int b = 0; b < 256; ++b) {
            dec[b] = pse_oracle_e4m3_decode(static_cast<uint8_t>(b));
        }
        k_bytes.assign((size_t)spec.B * spec.max_seq_len * spec.Hkv * spec.D, 0);
        v_bytes.assign(k_bytes.size(), 0);
        k_exp.assign((size_t)spec.B * spec.max_seq_len * spec.Hkv, 0);
        v_exp.assign(k_exp.size(), 0);
        page_table.assign((size_t)spec.B * max_lp, -1);
        page_used.assign((size_t)spec.max_pages, 0);
        seq_len.assign((size_t)spec.B, 0);
        step_counter = 0;
        total_allocs = 0;
        total_frees = 0;
    }

    void reset() {
        std::fill(k_bytes.begin(), k_bytes.end(), 0);
        std::fill(v_bytes.begin(), v_bytes.end(), 0);
        std::fill(k_exp.begin(), k_exp.end(), 0);
        std::fill(v_exp.begin(), v_exp.end(), 0);
        std::fill(page_table.begin(), page_table.end(), -1);
        std::fill(page_used.begin(), page_used.end(), 0);
        std::fill(seq_len.begin(), seq_len.end(), 0);
        step_counter = 0;
        total_allocs = 0;
        total_frees = 0;
    }

    int sink_end(int b) const {
        return std::min(spec.n_sink, seq_len[(size_t)b]);
    }

    int win_start(int b) const {
        const int se = sink_end(b);
        return std::max(se, seq_len[(size_t)b] - spec.window);
    }

    // scale_exp per the contract: floor(log2(amax)) - 8, clamped to [-110,110]
    static int8_t scale_exp_of(float amax) {
        if (amax == 0.0f) return 0;
        int e = 0;
        std::frexp(amax, &e);          // amax = f * 2^e with f in [0.5, 1)
        int se = (e - 1) - 8;          // floor(log2(amax)) = e - 1
        if (se < -110) se = -110;
        if (se > 110) se = 110;
        return static_cast<int8_t>(se);
    }

    void quantize_vec(const float* x, int8_t* exp_out, uint8_t* bytes_out) const {
        const int D = spec.D;
        float amax = 0.0f;
        for (int d = 0; d < D; ++d) {
            const float a = std::fabs(x[d]);
            if (a > amax) amax = a;
        }
        const int8_t se = scale_exp_of(amax);
        *exp_out = se;
        for (int d = 0; d < D; ++d) {
            const float z = std::ldexp(x[d], -static_cast<int>(se));
            bytes_out[d] = table.encode(z);
        }
    }

    size_t tok_head_index(int b, int pos, int h) const {
        return ((size_t)b * spec.max_seq_len + (size_t)pos) * (size_t)spec.Hkv + (size_t)h;
    }

    int alloc_page() {
        for (int p = 0; p < spec.max_pages; ++p) {
            if (!page_used[(size_t)p]) {
                page_used[(size_t)p] = 1;
                total_allocs += 1;
                return p;
            }
        }
        return -1;  // harness invariant: never happens
    }

    void append_token(int seq, const float* k_in, const float* v_in) {
        const int P = spec.page_size;
        const int pos = seq_len[(size_t)seq];
        const int lp = pos / P;
        int32_t& entry = page_table[(size_t)seq * max_lp + lp];
        if (entry < 0) {
            entry = alloc_page();
        }
        for (int h = 0; h < spec.Hkv; ++h) {
            const size_t th = tok_head_index(seq, pos, h);
            quantize_vec(k_in + (size_t)h * spec.D, &k_exp[th], &k_bytes[th * spec.D]);
            quantize_vec(v_in + (size_t)h * spec.D, &v_exp[th], &v_bytes[th * spec.D]);
        }
        seq_len[(size_t)seq] = pos + 1;
    }

    void reclaim_dead_pages() {
        const int P = spec.page_size;
        for (int b = 0; b < spec.B; ++b) {
            const int se = sink_end(b);
            const int ws = win_start(b);
            const int n_lp = (seq_len[(size_t)b] + P - 1) / P;
            for (int lp = 0; lp < n_lp; ++lp) {
                int32_t& entry = page_table[(size_t)b * max_lp + lp];
                if (entry < 0) continue;
                const int lo = lp * P;
                const int hi = lo + P;
                if (lo >= se && hi <= ws) {
                    page_used[(size_t)entry] = 0;
                    entry = -1;
                    total_frees += 1;
                }
            }
        }
    }

    // Appends live positions of b, ascending, into out.
    void live_positions(int b, std::vector<int>* out) const {
        out->clear();
        const int L = seq_len[(size_t)b];
        const int se = sink_end(b);
        const int ws = win_start(b);
        for (int t = 0; t < se; ++t) out->push_back(t);
        for (int t = ws; t < L; ++t) out->push_back(t);
    }

    uint64_t kv_hash_of(int b) const {
        uint64_t h = 1469598103934665603ULL;
        pse_oracle_fnv_i32(&h, seq_len[(size_t)b]);
        pse_oracle_fnv_i32(&h, sink_end(b));
        pse_oracle_fnv_i32(&h, win_start(b));

        std::vector<int> live;
        live_positions(b, &live);
        for (int pos : live) {
            for (int hk = 0; hk < spec.Hkv; ++hk) {
                const size_t th = tok_head_index(b, pos, hk);
                pse_oracle_fnv_bytes(&h, &k_exp[th], 1);
                pse_oracle_fnv_bytes(&h, &k_bytes[th * spec.D], (size_t)spec.D);
                pse_oracle_fnv_bytes(&h, &v_exp[th], 1);
                pse_oracle_fnv_bytes(&h, &v_bytes[th * spec.D], (size_t)spec.D);
            }
        }
        return h;
    }

    int32_t count_free_pages() const {
        int32_t n = 0;
        for (int p = 0; p < spec.max_pages; ++p) {
            if (!page_used[(size_t)p]) ++n;
        }
        return n;
    }

    uint64_t page_state_checksum() const {
        const int P = spec.page_size;
        uint64_t h = 1469598103934665603ULL;
        pse_oracle_fnv_i32(&h, spec.B);
        pse_oracle_fnv_i32(&h, spec.Hq);
        pse_oracle_fnv_i32(&h, spec.Hkv);
        pse_oracle_fnv_i32(&h, spec.D);
        pse_oracle_fnv_i32(&h, spec.page_size);
        pse_oracle_fnv_i32(&h, spec.n_sink);
        pse_oracle_fnv_i32(&h, spec.window);
        pse_oracle_fnv_i32(&h, spec.max_pages);
        pse_oracle_fnv_i32(&h, spec.max_seq_len);
        pse_oracle_fnv_i32(&h, step_counter);
        for (int b = 0; b < spec.B; ++b) {
            pse_oracle_fnv_i32(&h, seq_len[(size_t)b]);
            const int n_lp = (seq_len[(size_t)b] + P - 1) / P;
            pse_oracle_fnv_i32(&h, n_lp);
            for (int lp = 0; lp < n_lp; ++lp) {
                pse_oracle_fnv_i32(&h, page_table[(size_t)b * max_lp + lp]);
            }
        }
        pse_oracle_fnv_i32(&h, total_allocs);
        pse_oracle_fnv_i32(&h, total_frees);
        pse_oracle_fnv_i32(&h, count_free_pages());
        return h;
    }

    void attention_row(int b, const float* q_row, float* y_out, float* lse_out) const {
        const int Hq = spec.Hq;
        const int Hkv = spec.Hkv;
        const int D = spec.D;
        const int group = Hq / Hkv;

        std::vector<int> live;
        live_positions(b, &live);

        if (live.empty()) {
            for (int hq = 0; hq < Hq; ++hq) {
                for (int d = 0; d < D; ++d) y_out[(size_t)hq * D + d] = 0.0f;
                lse_out[hq] = 0.0f;
            }
            return;
        }

        const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(D));

        for (int hq = 0; hq < Hq; ++hq) {
            const int kvh = hq / group;
            const float* qh = q_row + (size_t)hq * D;

            std::vector<double> scores(live.size(), 0.0);
            double m = -std::numeric_limits<double>::infinity();

            for (size_t j = 0; j < live.size(); ++j) {
                const int pos = live[j];
                const size_t th = tok_head_index(b, pos, kvh);
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dot += static_cast<double>(qh[d]) * dec[k_bytes[th * D + d]];
                }
                const double s =
                    dot * std::ldexp(1.0, k_exp[th]) * inv_sqrt_d;
                scores[j] = s;
                if (s > m) m = s;
            }

            double denom = 0.0;
            std::vector<double> acc((size_t)D, 0.0);
            for (size_t j = 0; j < live.size(); ++j) {
                const int pos = live[j];
                const size_t th = tok_head_index(b, pos, kvh);
                const double w = std::exp(scores[j] - m);
                denom += w;
                const double wv = w * std::ldexp(1.0, v_exp[th]);
                for (int d = 0; d < D; ++d) {
                    acc[(size_t)d] += wv * dec[v_bytes[th * D + d]];
                }
            }

            for (int d = 0; d < D; ++d) {
                y_out[(size_t)hq * D + d] =
                    static_cast<float>(acc[(size_t)d] / denom);
            }
            lse_out[hq] = static_cast<float>(m + std::log(denom));
        }
    }

    void step(
        const PseRunSpec& run,
        const PseHostInputsView& in,
        PseExpected* expected) {
        const int A = run.active_count;
        const int Hq = spec.Hq;
        const int Hkv = spec.Hkv;
        const int D = spec.D;

        step_counter += 1;

        // Phase 2: appends in row/token order.
        for (int a = 0; a < A; ++a) {
            const int seq = in.active_seq[a];
            int cnt = in.new_token_count[a];
            if (cnt < 0) cnt = 0;
            if (cnt > PSE_MAX_NEW_TOKENS) cnt = PSE_MAX_NEW_TOKENS;
            for (int nt = 0; nt < cnt; ++nt) {
                const size_t base =
                    ((size_t)a * PSE_MAX_NEW_TOKENS + (size_t)nt) *
                    (size_t)Hkv * (size_t)D;
                append_token(seq, in.new_k + base, in.new_v + base);
            }
        }

        // Phase 3: reclamation.
        reclaim_dead_pages();

        // Phase 4: outputs.
        expected->y.assign((size_t)A * Hq * D, 0.0f);
        expected->lse.assign((size_t)A * Hq, 0.0f);
        for (int a = 0; a < A; ++a) {
            const int seq = in.active_seq[a];
            attention_row(
                seq,
                in.q + (size_t)a * Hq * D,
                expected->y.data() + (size_t)a * Hq * D,
                expected->lse.data() + (size_t)a * Hq);
        }

        expected->seq_len = seq_len;
        expected->kv_hash.resize((size_t)spec.B);
        for (int b = 0; b < spec.B; ++b) {
            expected->kv_hash[(size_t)b] = kv_hash_of(b);
        }
        expected->page_state_checksum = page_state_checksum();
        expected->free_pages = count_free_pages();
        expected->total_allocs = total_allocs;
        expected->total_frees = total_frees;
    }
};

static inline bool pse_check_float_array(
    const char* name,
    const float* got,
    const float* expected,
    size_t count,
    float atol,
    float rtol,
    std::string* error) {
    for (size_t i = 0; i < count; ++i) {
        const float e = expected[i];
        const float g = got[i];
        const float diff = std::fabs(g - e);
        const float tol = atol + rtol * std::fabs(e);
        if (!(diff <= tol)) {
            if (error) {
                std::ostringstream oss;
                oss << name << " mismatch at flat index " << i
                    << ": got " << g << ", expected " << e
                    << ", diff " << diff << ", tol " << tol;
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

static inline bool pse_check_all_outputs(
    const PseRunSpec& run,
    const PseProblemSpec& spec,
    const PseExpected& expected,
    const PseHostOutputsView& got,
    std::string* error) {
    const size_t y_count = (size_t)run.active_count * spec.Hq * spec.D;
    const size_t lse_count = (size_t)run.active_count * spec.Hq;

    if (!pse_check_float_array("y", got.y, expected.y.data(), y_count,
                               PSE_Y_ATOL, PSE_Y_RTOL, error)) {
        return false;
    }
    if (!pse_check_float_array("lse", got.lse, expected.lse.data(), lse_count,
                               PSE_LSE_ATOL, PSE_LSE_RTOL, error)) {
        return false;
    }

    for (int b = 0; b < spec.B; ++b) {
        if (got.seq_len[b] != expected.seq_len[(size_t)b]) {
            if (error) {
                std::ostringstream oss;
                oss << "seq_len mismatch at seq=" << b
                    << ": got " << got.seq_len[b]
                    << ", expected " << expected.seq_len[(size_t)b];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int b = 0; b < spec.B; ++b) {
        if (got.kv_hash[b] != expected.kv_hash[(size_t)b]) {
            if (error) {
                std::ostringstream oss;
                oss << "kv_hash mismatch at seq=" << b
                    << ": got 0x" << std::hex << got.kv_hash[b]
                    << ", expected 0x" << expected.kv_hash[(size_t)b];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.page_state_checksum[0] != expected.page_state_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "page_state_checksum mismatch: got 0x"
                << std::hex << got.page_state_checksum[0]
                << ", expected 0x" << expected.page_state_checksum;
            *error = oss.str();
        }
        return false;
    }

    if (got.free_pages[0] != expected.free_pages ||
        got.total_allocs[0] != expected.total_allocs ||
        got.total_frees[0] != expected.total_frees) {
        if (error) {
            std::ostringstream oss;
            oss << "page counter mismatch: got free/allocs/frees = "
                << got.free_pages[0] << "/" << got.total_allocs[0]
                << "/" << got.total_frees[0]
                << ", expected " << expected.free_pages
                << "/" << expected.total_allocs
                << "/" << expected.total_frees;
            *error = oss.str();
        }
        return false;
    }

    return true;
}

/*
PIPELINE GRADING MODEL SUPPORTED BY THIS ORACLE

A test harness should:

  1. Construct PseOracleState with the same PseProblemSpec as the solution.
  2. Call solution_reset and oracle.reset.
  3. Run T steps that cross every interesting boundary:
       - chunked prefill (up to 8 tokens/row/step) and 0/1/2-token decode,
       - L crossing n_sink, n_sink+window, page-size boundaries,
       - page death and physical-page reuse under a tight pool,
       - empty steps and inactive rows,
       - adversarial quantization values (ties, saturation, signed zeros,
         exact powers of two, subnormal-region magnitudes).
  4. After every step compare y/lse with tolerance and everything else exactly.
  5. Re-run exactly (bit-identical checksums) and with permuted row order
     (identical y/lse/seq_len/kv_hash; page_state_checksum may differ).

This catches:
  - wrong E4M3 rounding/tie/saturation behavior,
  - wrong scale exponents (off-by-one in floor(log2)),
  - stale/evicted tokens attended or hashed,
  - sink/window boundary off-by-ones,
  - wrong GQA head mapping,
  - allocator nondeterminism or wrong lowest-free-id policy,
  - missed or premature page reclamation,
  - reset bugs and cross-step state corruption.
*/

#endif  // PAGED_SINK_E4M3_DECODE_ORACLE_HPP_
