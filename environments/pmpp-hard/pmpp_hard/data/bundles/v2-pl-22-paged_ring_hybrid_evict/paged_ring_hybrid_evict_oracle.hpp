// file: paged_ring_hybrid_evict_oracle.hpp

#ifndef PAGED_RING_HYBRID_EVICT_ORACLE_HPP_
#define PAGED_RING_HYBRID_EVICT_ORACLE_HPP_

#include "paged_ring_hybrid_evict_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

struct PrheHostInputsView {
    const int32_t* active_seq;
    const int32_t* append_count;
    const int32_t* token_values;
};

struct PrheHostOutputsView {
    const int32_t* live_count;
    const int64_t* live_sum;
    const uint64_t* live_hash;
    const uint64_t* page_table_checksum;
    const int32_t* evicted_count;
    const int32_t* free_pages;
};

struct PrheExpected {
    std::vector<int32_t> live_count;
    std::vector<int64_t> live_sum;
    std::vector<uint64_t> live_hash;
    uint64_t page_table_checksum = 0;
    int32_t evicted_count = 0;
    int32_t free_pages = 0;
};

static inline uint64_t prhe_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void prhe_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = prhe_oracle_fnv_byte(v, p[i]);
    }
    *h = v;
}

struct PrheOracleState {
    PrheProblemSpec spec{};
    int max_logical_pages = 0;
    int step = 0;

    std::vector<int32_t> page_data;
    std::vector<int32_t> page_owner;
    std::vector<int32_t> page_logical;
    std::vector<int32_t> page_last_used;
    std::vector<int32_t> page_table;

    std::vector<int32_t> length;
    std::vector<int32_t> ring_head;
    std::vector<int32_t> ring_count;

    void init(const PrheProblemSpec& s) {
        spec = s;
        max_logical_pages = prhe_max_logical_pages_host(spec.max_len, spec.page_size);

        page_data.assign((size_t)spec.max_pages * (size_t)spec.page_size, 0);
        page_owner.assign((size_t)spec.max_pages, -1);
        page_logical.assign((size_t)spec.max_pages, -1);
        page_last_used.assign((size_t)spec.max_pages, 0);
        page_table.assign((size_t)spec.B * (size_t)max_logical_pages, -1);

        length.assign((size_t)spec.B, 0);
        ring_head.assign((size_t)spec.B, 0);
        ring_count.assign((size_t)spec.B, 0);
        step = 0;
    }

    void reset() {
        std::fill(page_data.begin(), page_data.end(), 0);
        std::fill(page_owner.begin(), page_owner.end(), -1);
        std::fill(page_logical.begin(), page_logical.end(), -1);
        std::fill(page_last_used.begin(), page_last_used.end(), 0);
        std::fill(page_table.begin(), page_table.end(), -1);
        std::fill(length.begin(), length.end(), 0);
        std::fill(ring_head.begin(), ring_head.end(), 0);
        std::fill(ring_count.begin(), ring_count.end(), 0);
        step = 0;
    }

    bool page_intersects_live(int seq, int logical_page) const {
        const int len = length[(size_t)seq];
        if (len <= 0) return false;

        int start = len - spec.window_size;
        if (start < 0) start = 0;

        const int end = len - 1;
        const int page_start = logical_page * spec.page_size;
        const int page_end = page_start + spec.page_size - 1;

        return page_start <= end && page_end >= start;
    }

    int alloc_page(int* evicted) {
        for (int p = 0; p < spec.max_pages; ++p) {
            if (page_owner[(size_t)p] < 0) return p;
        }

        int best = -1;
        int best_last = 2147483647;

        for (int p = 0; p < spec.max_pages; ++p) {
            const int owner = page_owner[(size_t)p];
            if (owner < 0 || owner >= spec.B) continue;

            const int lp = page_logical[(size_t)p];
            if (page_intersects_live(owner, lp)) continue;

            const int lu = page_last_used[(size_t)p];
            if (best < 0 || lu < best_last || (lu == best_last && p < best)) {
                best = p;
                best_last = lu;
            }
        }

        if (best < 0) return -1;

        const int old_owner = page_owner[(size_t)best];
        const int old_lp = page_logical[(size_t)best];

        if (old_owner >= 0 && old_owner < spec.B &&
            old_lp >= 0 && old_lp < max_logical_pages) {
            const size_t idx = (size_t)old_owner * (size_t)max_logical_pages + (size_t)old_lp;
            if (page_table[idx] == best) page_table[idx] = -1;
        }

        page_owner[(size_t)best] = -1;
        page_logical[(size_t)best] = -1;
        page_last_used[(size_t)best] = step;

        *evicted += 1;
        return best;
    }

    void append_one(int seq, int32_t value, int* evicted) {
        if (seq < 0 || seq >= spec.B) return;

        int len = length[(size_t)seq];
        if (len >= spec.max_len) return;

        const int lp = len / spec.page_size;
        const int off = len - lp * spec.page_size;
        const size_t table_idx = (size_t)seq * (size_t)max_logical_pages + (size_t)lp;

        int phys = page_table[table_idx];

        if (phys < 0) {
            phys = alloc_page(evicted);
            if (phys < 0) return;

            page_table[table_idx] = phys;
            page_owner[(size_t)phys] = seq;
            page_logical[(size_t)phys] = lp;
            page_last_used[(size_t)phys] = step;
        }

        page_data[(size_t)phys * (size_t)spec.page_size + (size_t)off] = value;
        page_last_used[(size_t)phys] = step;

        ++len;
        length[(size_t)seq] = len;
        ring_head[(size_t)seq] = len % spec.window_size;
        ring_count[(size_t)seq] = len < spec.window_size ? len : spec.window_size;
    }

    uint64_t live_hash_for_seq(int seq, int32_t* count_out, int64_t* sum_out) {
        const int len = length[(size_t)seq];
        int start = len - spec.window_size;
        if (start < 0) start = 0;

        int count = 0;
        uint64_t sum_bits = 0;

        for (int pos = start; pos < len; ++pos) {
            const int lp = pos / spec.page_size;
            const int off = pos - lp * spec.page_size;
            const int phys = page_table[(size_t)seq * (size_t)max_logical_pages + (size_t)lp];

            if (phys >= 0) {
                ++count;
                const int32_t v = page_data[(size_t)phys * (size_t)spec.page_size + (size_t)off];
                sum_bits += static_cast<uint64_t>(static_cast<int64_t>(v));
            }
        }

        uint64_t h = 1469598103934665603ULL;
        prhe_oracle_fnv_bytes(&h, &count, sizeof(int32_t));

        for (int pos = start; pos < len; ++pos) {
            const int lp = pos / spec.page_size;
            const int off = pos - lp * spec.page_size;
            const int phys = page_table[(size_t)seq * (size_t)max_logical_pages + (size_t)lp];

            if (phys >= 0) {
                const int32_t p32 = static_cast<int32_t>(pos);
                const int32_t v = page_data[(size_t)phys * (size_t)spec.page_size + (size_t)off];

                prhe_oracle_fnv_bytes(&h, &p32, sizeof(int32_t));
                prhe_oracle_fnv_bytes(&h, &v, sizeof(int32_t));
                page_last_used[(size_t)phys] = step;
            }
        }

        *count_out = count;
        *sum_out = static_cast<int64_t>(sum_bits);
        return h;
    }

    uint64_t checksum() const {
        uint64_t h = 1469598103934665603ULL;

        prhe_oracle_fnv_bytes(&h, &spec.B, sizeof(int32_t));
        prhe_oracle_fnv_bytes(&h, &spec.max_len, sizeof(int32_t));
        prhe_oracle_fnv_bytes(&h, &spec.page_size, sizeof(int32_t));
        prhe_oracle_fnv_bytes(&h, &spec.window_size, sizeof(int32_t));
        prhe_oracle_fnv_bytes(&h, &spec.max_pages, sizeof(int32_t));
        prhe_oracle_fnv_bytes(&h, &step, sizeof(int32_t));

        for (int s = 0; s < spec.B; ++s) {
            prhe_oracle_fnv_bytes(&h, &length[(size_t)s], sizeof(int32_t));
            prhe_oracle_fnv_bytes(&h, &ring_head[(size_t)s], sizeof(int32_t));
            prhe_oracle_fnv_bytes(&h, &ring_count[(size_t)s], sizeof(int32_t));

            for (int lp = 0; lp < max_logical_pages; ++lp) {
                const int x = page_table[(size_t)s * (size_t)max_logical_pages + (size_t)lp];
                prhe_oracle_fnv_bytes(&h, &x, sizeof(int32_t));
            }
        }

        for (int p = 0; p < spec.max_pages; ++p) {
            prhe_oracle_fnv_bytes(&h, &page_owner[(size_t)p], sizeof(int32_t));
            prhe_oracle_fnv_bytes(&h, &page_logical[(size_t)p], sizeof(int32_t));
            prhe_oracle_fnv_bytes(&h, &page_last_used[(size_t)p], sizeof(int32_t));
        }

        return h;
    }

    void step_once(
        const PrheRunSpec& run,
        const PrheHostInputsView& in,
        PrheExpected* expected) {
        (void)run;

        step += 1;

        int evicted = 0;

        for (int r = 0; r < run.active_count; ++r) {
            const int seq = in.active_seq[r];

            int cnt = in.append_count[r];
            if (cnt < 0) cnt = 0;
            if (cnt > spec.max_new_tokens) cnt = spec.max_new_tokens;
            if (cnt > PRHE_MAX_NEW_TOKENS) cnt = PRHE_MAX_NEW_TOKENS;

            for (int i = 0; i < cnt; ++i) {
                append_one(
                    seq,
                    in.token_values[(size_t)r * (size_t)spec.max_new_tokens + (size_t)i],
                    &evicted);
            }
        }

        expected->live_count.assign((size_t)spec.B, 0);
        expected->live_sum.assign((size_t)spec.B, 0);
        expected->live_hash.assign((size_t)spec.B, 0);

        for (int seq = 0; seq < spec.B; ++seq) {
            int32_t count = 0;
            int64_t sum = 0;
            const uint64_t h = live_hash_for_seq(seq, &count, &sum);

            expected->live_count[(size_t)seq] = count;
            expected->live_sum[(size_t)seq] = sum;
            expected->live_hash[(size_t)seq] = h;
        }

        int free_count = 0;
        for (int p = 0; p < spec.max_pages; ++p) {
            if (page_owner[(size_t)p] < 0) ++free_count;
        }

        expected->page_table_checksum = checksum();
        expected->evicted_count = evicted;
        expected->free_pages = free_count;
    }
};

static inline bool prhe_check_all_outputs(
    const PrheProblemSpec& spec,
    const PrheExpected& expected,
    const PrheHostOutputsView& got,
    std::string* error) {
    for (int seq = 0; seq < spec.B; ++seq) {
        if (got.live_count[seq] != expected.live_count[(size_t)seq]) {
            if (error) {
                std::ostringstream oss;
                oss << "live_count mismatch seq=" << seq
                    << ": got " << got.live_count[seq]
                    << ", expected " << expected.live_count[(size_t)seq];
                *error = oss.str();
            }
            return false;
        }

        if (got.live_sum[seq] != expected.live_sum[(size_t)seq]) {
            if (error) {
                std::ostringstream oss;
                oss << "live_sum mismatch seq=" << seq
                    << ": got " << got.live_sum[seq]
                    << ", expected " << expected.live_sum[(size_t)seq];
                *error = oss.str();
            }
            return false;
        }

        if (got.live_hash[seq] != expected.live_hash[(size_t)seq]) {
            if (error) {
                std::ostringstream oss;
                oss << "live_hash mismatch seq=" << seq
                    << ": got 0x" << std::hex << got.live_hash[seq]
                    << ", expected 0x" << expected.live_hash[(size_t)seq];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.page_table_checksum[0] != expected.page_table_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "page_table_checksum mismatch: got 0x"
                << std::hex << got.page_table_checksum[0]
                << ", expected 0x" << expected.page_table_checksum;
            *error = oss.str();
        }
        return false;
    }

    if (got.evicted_count[0] != expected.evicted_count) {
        if (error) {
            std::ostringstream oss;
            oss << "evicted_count mismatch: got " << got.evicted_count[0]
                << ", expected " << expected.evicted_count;
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
    prhe_check_all_outputs(...)

Required scenario coverage:
  - append sizes 0..max_new_tokens
  - page boundary crossing
  - window sliding across page boundaries
  - fixed page budget forcing cold-page eviction
  - no-op append at max_len
  - invalid active sequence ids
  - reset and permuted replay where active sequence rows are independent
*/

#endif  // PAGED_RING_HYBRID_EVICT_ORACLE_HPP_
