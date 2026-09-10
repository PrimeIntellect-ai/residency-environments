// ============================================================================
// file: multi_checkpoint_tree_rollback_oracle.hpp
// Independent CPU oracle/state model + exact check helpers.
// ============================================================================

#ifndef MULTI_CHECKPOINT_TREE_ROLLBACK_ORACLE_HPP_
#define MULTI_CHECKPOINT_TREE_ROLLBACK_ORACLE_HPP_

#include "multi_checkpoint_tree_rollback_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

struct MctrHostInputsView {
    const int32_t* active_seq;
    const int32_t* op_code;
    const int32_t* token_count;
    const int32_t* accept_count;
    const int32_t* checkpoint_id;
    const int32_t* token_values;
    const int32_t* correction_value;
};

struct MctrHostOutputsView {
    const int64_t* live_cache_sum;
    const uint64_t* live_cache_tail_hash;
    const int32_t* length;
    const int32_t* num_checkpoints;
    const int32_t* top_checkpoint_len;
    const uint64_t* state_checksum;
};

struct MctrCheckpoint {
    int32_t id = 0;
    int32_t len = 0;
    uint64_t hash = 0;
};

struct MctrExpected {
    std::vector<int64_t> live_cache_sum;
    std::vector<uint64_t> live_cache_tail_hash;
    std::vector<int32_t> length;
    std::vector<int32_t> num_checkpoints;
    std::vector<int32_t> top_checkpoint_len;
    uint64_t state_checksum = 0;
};

static inline uint64_t mctr_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void mctr_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;

    for (size_t i = 0; i < n; ++i) {
        v = mctr_oracle_fnv_byte(v, p[i]);
    }

    *h = v;
}

struct MctrOracleState {
    MctrProblemSpec spec{};
    std::vector<int32_t> cache;
    std::vector<int32_t> length;
    std::vector<std::vector<MctrCheckpoint>> stack;

    void init(const MctrProblemSpec& s) {
        spec = s;
        cache.assign((size_t)spec.B * (size_t)spec.max_len, 0);
        length.assign((size_t)spec.B, 0);
        stack.assign((size_t)spec.B, std::vector<MctrCheckpoint>());
    }

    void reset() {
        std::fill(cache.begin(), cache.end(), 0);
        std::fill(length.begin(), length.end(), 0);
        for (std::vector<MctrCheckpoint>& v : stack) {
            v.clear();
        }
    }

    uint64_t tail_hash_for_seq(int seq) const {
        const int len = length[(size_t)seq];
        const int tail_count = std::min(len, MCTR_TAIL_VALUES);
        const int tail_start = len - tail_count;

        uint64_t h = 1469598103934665603ULL;

        mctr_oracle_fnv_bytes(&h, &tail_count, sizeof(int32_t));

        if (tail_count > 0) {
            mctr_oracle_fnv_bytes(
                &h,
                &cache[(size_t)seq * (size_t)spec.max_len + (size_t)tail_start],
                sizeof(int32_t) * (size_t)tail_count);
        }

        return h;
    }

    int64_t sum_for_seq(int seq) const {
        uint64_t sum_bits = 0;
        const int len = length[(size_t)seq];

        for (int i = 0; i < len; ++i) {
            const int32_t v = cache[(size_t)seq * (size_t)spec.max_len + (size_t)i];
            sum_bits += static_cast<uint64_t>(static_cast<int64_t>(v));
        }

        return static_cast<int64_t>(sum_bits);
    }

    void append_one(int seq, int32_t value) {
        int len = length[(size_t)seq];
        if (len < spec.max_len) {
            cache[(size_t)seq * (size_t)spec.max_len + (size_t)len] = value;
            length[(size_t)seq] = len + 1;
        }
    }

    void apply_one(
        int seq,
        int op,
        int k,
        int a,
        int checkpoint_id,
        const int32_t* values4,
        int32_t correction) {
        if (seq < 0 || seq >= spec.B) return;

        if (k < 0) k = 0;
        if (k > MCTR_MAX_K) k = MCTR_MAX_K;

        if (op == MCTR_OP_APPEND) {
            for (int i = 0; i < k; ++i) append_one(seq, values4[i]);
            return;
        }

        if (op == MCTR_OP_SAVE_CHECKPOINT) {
            if ((int)stack[(size_t)seq].size() < spec.max_depth) {
                stack[(size_t)seq].push_back(MctrCheckpoint{
                    checkpoint_id,
                    length[(size_t)seq],
                    tail_hash_for_seq(seq)
                });
            }
            return;
        }

        if (op == MCTR_OP_ROLLBACK_TO) {
            std::vector<MctrCheckpoint>& st = stack[(size_t)seq];
            int found = -1;

            for (int i = (int)st.size() - 1; i >= 0; --i) {
                if (st[(size_t)i].id == checkpoint_id) {
                    found = i;
                    break;
                }
            }

            if (found >= 0) {
                length[(size_t)seq] = st[(size_t)found].len;
                st.resize((size_t)found + 1);
            }
            return;
        }

        if (op == MCTR_OP_ACCEPT_PREFIX) {
            if (a < 0) a = 0;
            if (a > k) a = k;

            const int old_len = length[(size_t)seq];

            for (int i = 0; i < a; ++i) append_one(seq, values4[i]);

            if (old_len + a < spec.max_len) append_one(seq, correction);
            return;
        }
    }

    uint64_t checksum() const {
        uint64_t h = 1469598103934665603ULL;

        for (int seq = 0; seq < spec.B; ++seq) {
            const int len = length[(size_t)seq];
            const int dep = (int)stack[(size_t)seq].size();

            mctr_oracle_fnv_bytes(&h, &len, sizeof(int32_t));
            mctr_oracle_fnv_bytes(&h, &dep, sizeof(int32_t));

            for (int d = 0; d < dep; ++d) {
                const MctrCheckpoint& c = stack[(size_t)seq][(size_t)d];
                mctr_oracle_fnv_bytes(&h, &c.id, sizeof(int32_t));
                mctr_oracle_fnv_bytes(&h, &c.len, sizeof(int32_t));
                mctr_oracle_fnv_bytes(&h, &c.hash, sizeof(uint64_t));
            }

            mctr_oracle_fnv_bytes(&h, &len, sizeof(int32_t));

            if (len > 0) {
                mctr_oracle_fnv_bytes(
                    &h,
                    &cache[(size_t)seq * (size_t)spec.max_len],
                    sizeof(int32_t) * (size_t)len);
            }
        }

        return h;
    }

    void step(
        const MctrRunSpec& run,
        const MctrHostInputsView& in,
        MctrExpected* expected) {
        for (int r = 0; r < run.active_count; ++r) {
            int32_t vals[MCTR_MAX_K];

            for (int i = 0; i < MCTR_MAX_K; ++i) {
                vals[i] = in.token_values[(size_t)r * (size_t)MCTR_MAX_K + (size_t)i];
            }

            apply_one(
                in.active_seq[r],
                in.op_code[r],
                in.token_count[r],
                in.accept_count[r],
                in.checkpoint_id[r],
                vals,
                in.correction_value[r]);
        }

        expected->live_cache_sum.assign((size_t)spec.B, 0);
        expected->live_cache_tail_hash.assign((size_t)spec.B, 0);
        expected->length.assign((size_t)spec.B, 0);
        expected->num_checkpoints.assign((size_t)spec.B, 0);
        expected->top_checkpoint_len.assign((size_t)spec.B, -1);

        for (int seq = 0; seq < spec.B; ++seq) {
            const int dep = (int)stack[(size_t)seq].size();

            expected->live_cache_sum[(size_t)seq] = sum_for_seq(seq);
            expected->live_cache_tail_hash[(size_t)seq] = tail_hash_for_seq(seq);
            expected->length[(size_t)seq] = length[(size_t)seq];
            expected->num_checkpoints[(size_t)seq] = dep;
            expected->top_checkpoint_len[(size_t)seq] =
                dep > 0 ? stack[(size_t)seq][(size_t)dep - 1].len : -1;
        }

        expected->state_checksum = checksum();
    }
};

static inline bool mctr_check_all_outputs(
    const MctrProblemSpec& spec,
    const MctrExpected& expected,
    const MctrHostOutputsView& got,
    std::string* error) {
    for (int seq = 0; seq < spec.B; ++seq) {
        if (got.length[seq] != expected.length[(size_t)seq]) {
            if (error) {
                std::ostringstream oss;
                oss << "length mismatch seq=" << seq
                    << ": got " << got.length[seq]
                    << ", expected " << expected.length[(size_t)seq];
                *error = oss.str();
            }
            return false;
        }

        if (got.num_checkpoints[seq] != expected.num_checkpoints[(size_t)seq]) {
            if (error) {
                std::ostringstream oss;
                oss << "num_checkpoints mismatch seq=" << seq
                    << ": got " << got.num_checkpoints[seq]
                    << ", expected " << expected.num_checkpoints[(size_t)seq];
                *error = oss.str();
            }
            return false;
        }

        if (got.top_checkpoint_len[seq] != expected.top_checkpoint_len[(size_t)seq]) {
            if (error) {
                std::ostringstream oss;
                oss << "top_checkpoint_len mismatch seq=" << seq
                    << ": got " << got.top_checkpoint_len[seq]
                    << ", expected " << expected.top_checkpoint_len[(size_t)seq];
                *error = oss.str();
            }
            return false;
        }

        if (got.live_cache_sum[seq] != expected.live_cache_sum[(size_t)seq]) {
            if (error) {
                std::ostringstream oss;
                oss << "live_cache_sum mismatch seq=" << seq
                    << ": got " << got.live_cache_sum[seq]
                    << ", expected " << expected.live_cache_sum[(size_t)seq];
                *error = oss.str();
            }
            return false;
        }

        if (got.live_cache_tail_hash[seq] != expected.live_cache_tail_hash[(size_t)seq]) {
            if (error) {
                std::ostringstream oss;
                oss << "live_cache_tail_hash mismatch seq=" << seq
                    << ": got 0x" << std::hex << got.live_cache_tail_hash[seq]
                    << ", expected 0x" << expected.live_cache_tail_hash[(size_t)seq];
                *error = oss.str();
            }
            return false;
        }
    }

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

/*
GRADER MODEL

Use:
  oracle.init(problem_spec)
  solution_reset + oracle.reset
  for each step:
    solution_run(...)
    oracle.step(...)
    mctr_check_all_outputs(...)

Required scenario coverage:
  - nested SAVE_CHECKPOINT
  - rollback to old checkpoint then re-append
  - rollback-to-zero via checkpoint saved at length 0
  - invalid/no-op rollback id
  - ACCEPT_PREFIX with a=0, a=k, and cache-near-max_len
  - stack-full SAVE_CHECKPOINT no-op
  - replay after reset with permuted active row order where active sequences are
    independent in each step
*/

#endif  // MULTI_CHECKPOINT_TREE_ROLLBACK_ORACLE_HPP_
