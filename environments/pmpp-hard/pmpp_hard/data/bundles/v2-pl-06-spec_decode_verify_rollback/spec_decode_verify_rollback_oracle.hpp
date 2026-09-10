// ============================================================================
// file: spec_decode_verify_rollback_oracle.hpp
// Independent CPU oracle/state model + exact check helpers.
// ============================================================================

#ifndef SPEC_DECODE_VERIFY_ROLLBACK_ORACLE_HPP_
#define SPEC_DECODE_VERIFY_ROLLBACK_ORACLE_HPP_

#include "spec_decode_verify_rollback_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

struct SdvHostInputsView {
    const int32_t* active_seq;
    const int32_t* draft_value;
    const int32_t* correction_value;
    const uint32_t* p_target;
    const uint32_t* p_draft;
    const uint32_t* uniform_u32;
};

struct SdvHostOutputsView {
    const int32_t* accepted_count;
    const int32_t* new_length;
    const int64_t* live_cache_sum;
    const uint64_t* live_cache_tail_hash;
    const uint64_t* state_checksum;
};

struct SdvExpected {
    std::vector<int32_t> accepted_count;
    std::vector<int32_t> new_length;
    std::vector<int64_t> live_cache_sum;
    std::vector<uint64_t> live_cache_tail_hash;
    uint64_t state_checksum = 0;
};

static inline uint64_t sdv_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void sdv_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = sdv_oracle_fnv_byte(v, p[i]);
    }
    *h = v;
}

static inline bool sdv_oracle_accept(
    uint32_t p_target,
    uint32_t p_draft,
    uint32_t uniform_u32) {
    if (p_draft == 0U) {
        return p_target > 0U;
    }

    const uint64_t lhs =
        static_cast<uint64_t>(uniform_u32) *
        static_cast<uint64_t>(p_draft);

    const uint64_t rhs =
        static_cast<uint64_t>(p_target) *
        4294967295ULL;

    return lhs <= rhs;
}

struct SdvOracleState {
    SdvProblemSpec spec{};

    std::vector<int32_t> cache;
    std::vector<int32_t> length;
    std::vector<int32_t> checkpoint_length;

    void init(const SdvProblemSpec& s) {
        spec = s;
        cache.assign((size_t)spec.B * (size_t)spec.max_len, 0);
        length.assign((size_t)spec.B, 0);
        checkpoint_length.assign((size_t)spec.B, 0);
    }

    void reset() {
        std::fill(cache.begin(), cache.end(), 0);
        std::fill(length.begin(), length.end(), 0);
        std::fill(checkpoint_length.begin(), checkpoint_length.end(), 0);
    }

    uint64_t tail_hash_for_seq(int seq) const {
        const int live_len = length[(size_t)seq];
        const int tail_count = std::min(live_len, SDV_TAIL_VALUES);
        const int tail_start = live_len - tail_count;

        uint64_t h = 1469598103934665603ULL;

        sdv_oracle_fnv_bytes(&h, &tail_count, sizeof(int32_t));

        if (tail_count > 0) {
            sdv_oracle_fnv_bytes(
                &h,
                &cache[(size_t)seq * (size_t)spec.max_len + (size_t)tail_start],
                sizeof(int32_t) * (size_t)tail_count);
        }

        return h;
    }

    int64_t sum_for_seq(int seq) const {
        uint64_t sum_bits = 0;
        const int live_len = length[(size_t)seq];

        for (int i = 0; i < live_len; ++i) {
            const int32_t v = cache[(size_t)seq * (size_t)spec.max_len + (size_t)i];
            sum_bits += static_cast<uint64_t>(static_cast<int64_t>(v));
        }

        return static_cast<int64_t>(sum_bits);
    }

    uint64_t checksum() const {
        uint64_t h = 1469598103934665603ULL;

        sdv_oracle_fnv_bytes(
            &h,
            length.data(),
            sizeof(int32_t) * (size_t)spec.B);

        sdv_oracle_fnv_bytes(
            &h,
            checkpoint_length.data(),
            sizeof(int32_t) * (size_t)spec.B);

        for (int b = 0; b < spec.B; ++b) {
            int live_len = length[(size_t)b];
            if (live_len < 0) live_len = 0;
            if (live_len > spec.max_len) live_len = spec.max_len;

            sdv_oracle_fnv_bytes(&h, &live_len, sizeof(int32_t));

            if (live_len > 0) {
                sdv_oracle_fnv_bytes(
                    &h,
                    &cache[(size_t)b * (size_t)spec.max_len],
                    sizeof(int32_t) * (size_t)live_len);
            }
        }

        return h;
    }

    int apply_one_active(
        int seq,
        int row,
        const SdvRunSpec& run,
        const SdvHostInputsView& in,
        int32_t* accepted_out,
        int32_t* new_length_out,
        int64_t* sum_out,
        uint64_t* tail_hash_out) {
        if (seq < 0 || seq >= spec.B) {
            *accepted_out = 0;
            *new_length_out = 0;
            *sum_out = 0;
            *tail_hash_out = 1469598103934665603ULL;
            return 0;
        }

        const int old_len = length[(size_t)seq];
        checkpoint_length[(size_t)seq] = old_len;

        int prefix = 0;

        for (int i = 0; i < run.draft_len; ++i) {
            const size_t idx = (size_t)row * (size_t)run.draft_len + (size_t)i;
            if (!sdv_oracle_accept(in.p_target[idx], in.p_draft[idx], in.uniform_u32[idx])) {
                break;
            }
            ++prefix;
        }

        int final_len = old_len;
        int retained_accepted = 0;

        for (int i = 0; i < prefix; ++i) {
            if (final_len >= spec.max_len) {
                break;
            }

            cache[(size_t)seq * (size_t)spec.max_len + (size_t)final_len] =
                in.draft_value[(size_t)row * (size_t)run.draft_len + (size_t)i];

            ++final_len;
            ++retained_accepted;
        }

        if (old_len + prefix < spec.max_len && final_len < spec.max_len) {
            cache[(size_t)seq * (size_t)spec.max_len + (size_t)final_len] =
                in.correction_value[row];
            ++final_len;
        }

        length[(size_t)seq] = final_len;

        *accepted_out = retained_accepted;
        *new_length_out = final_len;
        *sum_out = sum_for_seq(seq);
        *tail_hash_out = tail_hash_for_seq(seq);

        return retained_accepted;
    }

    void step(
        const SdvRunSpec& run,
        const SdvHostInputsView& in,
        SdvExpected* expected) {
        expected->accepted_count.assign((size_t)run.active_count, 0);
        expected->new_length.assign((size_t)run.active_count, 0);
        expected->live_cache_sum.assign((size_t)run.active_count, 0);
        expected->live_cache_tail_hash.assign((size_t)run.active_count, 1469598103934665603ULL);

        for (int a = 0; a < run.active_count; ++a) {
            const int seq = in.active_seq[a];

            apply_one_active(
                seq,
                a,
                run,
                in,
                &expected->accepted_count[(size_t)a],
                &expected->new_length[(size_t)a],
                &expected->live_cache_sum[(size_t)a],
                &expected->live_cache_tail_hash[(size_t)a]);
        }

        expected->state_checksum = checksum();
    }
};

static inline bool sdv_check_accepted_count(
    const SdvRunSpec& run,
    const SdvExpected& expected,
    const SdvHostOutputsView& got,
    std::string* error) {
    for (int a = 0; a < run.active_count; ++a) {
        if (got.accepted_count[a] != expected.accepted_count[(size_t)a]) {
            if (error) {
                std::ostringstream oss;
                oss << "accepted_count mismatch at active=" << a
                    << ": got " << got.accepted_count[a]
                    << ", expected " << expected.accepted_count[(size_t)a];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool sdv_check_new_length(
    const SdvRunSpec& run,
    const SdvExpected& expected,
    const SdvHostOutputsView& got,
    std::string* error) {
    for (int a = 0; a < run.active_count; ++a) {
        if (got.new_length[a] != expected.new_length[(size_t)a]) {
            if (error) {
                std::ostringstream oss;
                oss << "new_length mismatch at active=" << a
                    << ": got " << got.new_length[a]
                    << ", expected " << expected.new_length[(size_t)a];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool sdv_check_live_cache_sum(
    const SdvRunSpec& run,
    const SdvExpected& expected,
    const SdvHostOutputsView& got,
    std::string* error) {
    for (int a = 0; a < run.active_count; ++a) {
        if (got.live_cache_sum[a] != expected.live_cache_sum[(size_t)a]) {
            if (error) {
                std::ostringstream oss;
                oss << "live_cache_sum mismatch at active=" << a
                    << ": got " << got.live_cache_sum[a]
                    << ", expected " << expected.live_cache_sum[(size_t)a];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool sdv_check_live_cache_tail_hash(
    const SdvRunSpec& run,
    const SdvExpected& expected,
    const SdvHostOutputsView& got,
    std::string* error) {
    for (int a = 0; a < run.active_count; ++a) {
        if (got.live_cache_tail_hash[a] != expected.live_cache_tail_hash[(size_t)a]) {
            if (error) {
                std::ostringstream oss;
                oss << "live_cache_tail_hash mismatch at active=" << a
                    << ": got 0x" << std::hex << got.live_cache_tail_hash[a]
                    << ", expected 0x" << expected.live_cache_tail_hash[(size_t)a];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static inline bool sdv_check_state_checksum(
    const SdvExpected& expected,
    const SdvHostOutputsView& got,
    std::string* error) {
    if (got.state_checksum[0] != expected.state_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "state_checksum mismatch: got 0x"
                << std::hex << got.state_checksum[0]
                << ", expected 0x" << expected.state_checksum;
            *error = oss.str();
        }
        return false;
    }

    return true;
}

static inline bool sdv_check_all_outputs(
    const SdvRunSpec& run,
    const SdvExpected& expected,
    const SdvHostOutputsView& got,
    std::string* error) {
    if (!sdv_check_accepted_count(run, expected, got, error)) return false;
    if (!sdv_check_new_length(run, expected, got, error)) return false;
    if (!sdv_check_live_cache_sum(run, expected, got, error)) return false;
    if (!sdv_check_live_cache_tail_hash(run, expected, got, error)) return false;
    if (!sdv_check_state_checksum(expected, got, error)) return false;
    return true;
}

/*
PIPELINE GRADING MODEL SUPPORTED BY THIS ORACLE

A test harness should:

  1. Construct SdvOracleState with the same SdvProblemSpec as the solution.

  2. Call solution_reset and oracle.reset.

  3. Run T exact-int verification steps, T roughly 16-32:
       - mutate active_seq each step,
       - choose draft_len in {2,4,8},
       - include accept patterns:
           all accepted,
           all rejected,
           alternating,
           first-token reject,
           late reject,
       - include cache-near-max_len cases.

  4. After every step, compare exactly:
       - accepted_count[active],
       - new_length[active],
       - live_cache_sum[active],
       - live_cache_tail_hash[active],
       - state_checksum.

  5. Repeat after reset with:
       - different active batch order,
       - different pressure near max_len.

This catches:
  - keeping rejected draft suffixes,
  - forgetting to append correction,
  - off-by-one at accepted prefix length,
  - wrong cap behavior near max_len,
  - stale lengths/checkpoints,
  - incorrect live-cache summaries,
  - replay/order nondeterminism.
*/

#endif  // SPEC_DECODE_VERIFY_ROLLBACK_ORACLE_HPP_
