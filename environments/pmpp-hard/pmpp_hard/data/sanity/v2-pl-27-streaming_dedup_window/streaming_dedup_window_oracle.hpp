// file: streaming_dedup_window_oracle.hpp

#ifndef STREAMING_DEDUP_WINDOW_ORACLE_HPP_
#define STREAMING_DEDUP_WINDOW_ORACLE_HPP_

#include "streaming_dedup_window_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

struct SdwHostInputsView {
    const int32_t* key;
    const int32_t* value;
};

struct SdwHostOutputsView {
    const int32_t* active_count;
    const int32_t* num_new;
    const int32_t* num_dup;
    const int32_t* num_evicted;
    const uint64_t* evicted_key_checksum;
    const int64_t* live_agg_sum;
    const uint64_t* state_checksum;
};

struct SdwExpected {
    int32_t active_count = 0;
    int32_t num_new = 0;
    int32_t num_dup = 0;
    int32_t num_evicted = 0;
    uint64_t evicted_key_checksum = 1469598103934665603ULL;
    int64_t live_agg_sum = 0;
    uint64_t state_checksum = 0;
};

static inline uint64_t sdw_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void sdw_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = sdw_oracle_fnv_byte(v, p[i]);
    }
    *h = v;
}

static inline int64_t sdw_oracle_add_i64_i32(int64_t a, int32_t b) {
    uint64_t ua = static_cast<uint64_t>(a);
    ua += static_cast<uint64_t>(static_cast<int64_t>(b));
    return static_cast<int64_t>(ua);
}

static inline int64_t sdw_oracle_add_i64_i64(int64_t a, int64_t b) {
    uint64_t ua = static_cast<uint64_t>(a);
    ua += static_cast<uint64_t>(b);
    return static_cast<int64_t>(ua);
}

struct SdwOracleState {
    SdwProblemSpec spec{};
    int32_t current_pos = 0;
    int32_t active_count = 0;

    std::vector<int32_t> active;
    std::vector<int32_t> last_pos;
    std::vector<int32_t> lru_stamp;
    std::vector<int64_t> agg;

    void init(const SdwProblemSpec& s) {
        spec = s;
        active.assign((size_t)spec.key_space, 0);
        last_pos.assign((size_t)spec.key_space, 0);
        lru_stamp.assign((size_t)spec.key_space, 0);
        agg.assign((size_t)spec.key_space, 0);
        current_pos = 0;
        active_count = 0;
    }

    void reset() {
        std::fill(active.begin(), active.end(), 0);
        std::fill(last_pos.begin(), last_pos.end(), 0);
        std::fill(lru_stamp.begin(), lru_stamp.end(), 0);
        std::fill(agg.begin(), agg.end(), 0);
        current_pos = 0;
        active_count = 0;
    }

    void hash_finalized_key(uint64_t* h, int32_t key) const {
        sdw_oracle_fnv_bytes(h, &key, sizeof(int32_t));
        sdw_oracle_fnv_bytes(h, &agg[(size_t)key], sizeof(int64_t));
        sdw_oracle_fnv_bytes(h, &last_pos[(size_t)key], sizeof(int32_t));
    }

    void finalize_key(
        int key,
        uint64_t* evicted_hash,
        int32_t* num_evicted) {
        if (active[(size_t)key] == 0) return;

        hash_finalized_key(evicted_hash, key);
        *num_evicted += 1;

        active[(size_t)key] = 0;
        last_pos[(size_t)key] = 0;
        lru_stamp[(size_t)key] = 0;
        agg[(size_t)key] = 0;
        active_count -= 1;
    }

    void expire_old_keys(uint64_t* evicted_hash, int32_t* num_evicted) {
        for (int key = 0; key < spec.key_space; ++key) {
            if (active[(size_t)key] != 0 &&
                current_pos - last_pos[(size_t)key] > spec.window_size) {
                finalize_key(key, evicted_hash, num_evicted);
            }
        }
    }

    int find_lru_key() const {
        int best_key = -1;
        int best_stamp = 2147483647;

        for (int key = 0; key < spec.key_space; ++key) {
            if (active[(size_t)key] == 0) continue;

            const int stamp = lru_stamp[(size_t)key];
            if (best_key < 0 || stamp < best_stamp ||
                (stamp == best_stamp && key < best_key)) {
                best_key = key;
                best_stamp = stamp;
            }
        }

        return best_key;
    }

    uint64_t checksum() const {
        uint64_t h = 1469598103934665603ULL;

        sdw_oracle_fnv_bytes(&h, &spec.key_space, sizeof(int32_t));
        sdw_oracle_fnv_bytes(&h, &spec.capacity, sizeof(int32_t));
        sdw_oracle_fnv_bytes(&h, &spec.window_size, sizeof(int32_t));
        sdw_oracle_fnv_bytes(&h, &current_pos, sizeof(int32_t));
        sdw_oracle_fnv_bytes(&h, &active_count, sizeof(int32_t));

        for (int key = 0; key < spec.key_space; ++key) {
            sdw_oracle_fnv_bytes(&h, &active[(size_t)key], sizeof(int32_t));
            sdw_oracle_fnv_bytes(&h, &last_pos[(size_t)key], sizeof(int32_t));
            sdw_oracle_fnv_bytes(&h, &lru_stamp[(size_t)key], sizeof(int32_t));
            sdw_oracle_fnv_bytes(&h, &agg[(size_t)key], sizeof(int64_t));
        }

        return h;
    }

    void step_once(
        const SdwRunSpec& run,
        const SdwHostInputsView& in,
        SdwExpected* expected) {
        expected->num_new = 0;
        expected->num_dup = 0;
        expected->num_evicted = 0;
        expected->evicted_key_checksum = 1469598103934665603ULL;

        for (int r = 0; r < run.batch_size; ++r) {
            current_pos += 1;

            expire_old_keys(&expected->evicted_key_checksum, &expected->num_evicted);

            const int key = in.key[r];
            const int32_t value = in.value[r];

            if (key < 0 || key >= spec.key_space) {
                continue;
            }

            if (active[(size_t)key] != 0) {
                agg[(size_t)key] = sdw_oracle_add_i64_i32(agg[(size_t)key], value);
                last_pos[(size_t)key] = current_pos;
                lru_stamp[(size_t)key] = current_pos;
                expected->num_dup += 1;
            } else {
                if (active_count >= spec.capacity) {
                    const int victim = find_lru_key();
                    if (victim >= 0) {
                        finalize_key(victim, &expected->evicted_key_checksum, &expected->num_evicted);
                    }
                }

                if (active_count < spec.capacity) {
                    active[(size_t)key] = 1;
                    last_pos[(size_t)key] = current_pos;
                    lru_stamp[(size_t)key] = current_pos;
                    agg[(size_t)key] = static_cast<int64_t>(value);
                    active_count += 1;
                    expected->num_new += 1;
                }
            }
        }

        expire_old_keys(&expected->evicted_key_checksum, &expected->num_evicted);

        int64_t live_sum = 0;
        for (int key = 0; key < spec.key_space; ++key) {
            if (active[(size_t)key] != 0) {
                live_sum = sdw_oracle_add_i64_i64(live_sum, agg[(size_t)key]);
            }
        }

        expected->active_count = active_count;
        expected->live_agg_sum = live_sum;
        expected->state_checksum = checksum();
    }
};

static inline bool sdw_check_all_outputs(
    const SdwExpected& expected,
    const SdwHostOutputsView& got,
    std::string* error) {
    if (got.active_count[0] != expected.active_count) {
        if (error) {
            std::ostringstream oss;
            oss << "active_count mismatch: got " << got.active_count[0]
                << ", expected " << expected.active_count;
            *error = oss.str();
        }
        return false;
    }

    if (got.num_new[0] != expected.num_new) {
        if (error) {
            std::ostringstream oss;
            oss << "num_new mismatch: got " << got.num_new[0]
                << ", expected " << expected.num_new;
            *error = oss.str();
        }
        return false;
    }

    if (got.num_dup[0] != expected.num_dup) {
        if (error) {
            std::ostringstream oss;
            oss << "num_dup mismatch: got " << got.num_dup[0]
                << ", expected " << expected.num_dup;
            *error = oss.str();
        }
        return false;
    }

    if (got.num_evicted[0] != expected.num_evicted) {
        if (error) {
            std::ostringstream oss;
            oss << "num_evicted mismatch: got " << got.num_evicted[0]
                << ", expected " << expected.num_evicted;
            *error = oss.str();
        }
        return false;
    }

    if (got.evicted_key_checksum[0] != expected.evicted_key_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "evicted_key_checksum mismatch: got 0x"
                << std::hex << got.evicted_key_checksum[0]
                << ", expected 0x" << expected.evicted_key_checksum;
            *error = oss.str();
        }
        return false;
    }

    if (got.live_agg_sum[0] != expected.live_agg_sum) {
        if (error) {
            std::ostringstream oss;
            oss << "live_agg_sum mismatch: got " << got.live_agg_sum[0]
                << ", expected " << expected.live_agg_sum;
            *error = oss.str();
        }
        return false;
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
  solution_reset + oracle.reset()
  for each step:
    solution_run(...)
    oracle.step_once(...)
    sdw_check_all_outputs(...)

Required harness coverage:
  - high duplicate rate
  - low duplicate rate
  - capacity pressure causing LRU eviction
  - window expiry without capacity pressure
  - invalid keys that advance time
  - batch_size = 0
  - reset and replay
*/

#endif  // STREAMING_DEDUP_WINDOW_ORACLE_HPP_
