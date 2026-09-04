// ============================================================================
// file: onesweep_partition_reduce_oracle.hpp
// Independent CPU oracle + intermediate/final check helpers.
// ============================================================================

#ifndef ONESWEEP_PARTITION_REDUCE_ORACLE_HPP_
#define ONESWEEP_PARTITION_REDUCE_ORACLE_HPP_

#include "onesweep_partition_reduce_common.h"

#include <stdint.h>
#include <stddef.h>

#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct OprHostInputsView {
    const uint32_t* key;
    const int32_t* val;
};

struct OprHostOutputsView {
    const int32_t* tile_digit_offsets;
    const int32_t* digit_counts;
    const int32_t* digit_offsets;
    const uint32_t* packed_key;
    const int32_t* packed_val;
    const int32_t* packed_src;
    const int64_t* bucket_sum;
    const int64_t* bucket_max;
    const int32_t* bucket_argmax;
};

struct OprExpected {
    std::vector<int32_t> tile_digit_offsets;
    std::vector<int32_t> digit_counts;
    std::vector<int32_t> digit_offsets;
    std::vector<uint32_t> packed_key;
    std::vector<int32_t> packed_val;
    std::vector<int32_t> packed_src;
    std::vector<int64_t> bucket_sum;
    std::vector<int64_t> bucket_max;
    std::vector<int32_t> bucket_argmax;
};

static inline int opr_oracle_digit(uint32_t key, int radix_bits) {
    return static_cast<int>(key >> (32 - radix_bits));
}

static inline int64_t opr_oracle_int64_min() {
    return std::numeric_limits<int64_t>::min();
}

static inline void opr_cpu_oracle(
    const OprRunSpec& run,
    const OprHostInputsView& in,
    OprExpected* expected) {
    const int N = run.N;
    const int radix_bits = run.radix_bits;
    const int num_digits = opr_num_digits(radix_bits);
    const int num_tiles = opr_num_tiles(N);

    expected->tile_digit_offsets.assign((size_t)num_tiles * (size_t)num_digits, 0);
    expected->digit_counts.assign(num_digits, 0);
    expected->digit_offsets.assign(num_digits + 1, 0);
    expected->packed_key.assign(N, 0);
    expected->packed_val.assign(N, 0);
    expected->packed_src.assign(N, 0);
    expected->bucket_sum.assign(num_digits, 0);
    expected->bucket_max.assign(num_digits, opr_oracle_int64_min());
    expected->bucket_argmax.assign(num_digits, -1);

    std::vector<int32_t> tile_counts((size_t)num_tiles * (size_t)num_digits, 0);

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int start = tile * OPR_TILE_ITEMS;
        int end = start + OPR_TILE_ITEMS;
        if (end > N) end = N;

        for (int i = start; i < end; ++i) {
            const int d = opr_oracle_digit(in.key[i], radix_bits);
            ++tile_counts[(size_t)tile * (size_t)num_digits + (size_t)d];
        }
    }

    int total = 0;
    expected->digit_offsets[0] = 0;

    for (int d = 0; d < num_digits; ++d) {
        int count = 0;
        for (int tile = 0; tile < num_tiles; ++tile) {
            count += tile_counts[(size_t)tile * (size_t)num_digits + (size_t)d];
        }

        expected->digit_counts[(size_t)d] = count;
        total += count;
        expected->digit_offsets[(size_t)d + 1] = total;
    }

    for (int d = 0; d < num_digits; ++d) {
        int running = expected->digit_offsets[(size_t)d];

        for (int tile = 0; tile < num_tiles; ++tile) {
            const size_t idx = (size_t)tile * (size_t)num_digits + (size_t)d;
            expected->tile_digit_offsets[idx] = running;
            running += tile_counts[idx];
        }
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int start = tile * OPR_TILE_ITEMS;
        int end = start + OPR_TILE_ITEMS;
        if (end > N) end = N;

        std::vector<int32_t> local_counts((size_t)num_digits, 0);

        for (int i = start; i < end; ++i) {
            const uint32_t k = in.key[i];
            const int d = opr_oracle_digit(k, radix_bits);
            const size_t off_idx = (size_t)tile * (size_t)num_digits + (size_t)d;
            const int pos = expected->tile_digit_offsets[off_idx] + local_counts[(size_t)d];

            expected->packed_key[(size_t)pos] = k;
            expected->packed_val[(size_t)pos] = in.val[i];
            expected->packed_src[(size_t)pos] = i;

            ++local_counts[(size_t)d];
        }
    }

    for (int d = 0; d < num_digits; ++d) {
        const int begin = expected->digit_offsets[(size_t)d];
        const int end = expected->digit_offsets[(size_t)d + 1];

        uint64_t sum_bits = 0;
        int64_t max_v = opr_oracle_int64_min();
        int32_t arg = -1;
        bool any = false;

        for (int i = begin; i < end; ++i) {
            const int64_t v = static_cast<int64_t>(expected->packed_val[(size_t)i]);
            const int32_t src = expected->packed_src[(size_t)i];

            sum_bits += static_cast<uint64_t>(v);

            if (!any || v > max_v || (v == max_v && src < arg)) {
                any = true;
                max_v = v;
                arg = src;
            }
        }

        expected->bucket_sum[(size_t)d] = static_cast<int64_t>(sum_bits);
        expected->bucket_max[(size_t)d] = any ? max_v : opr_oracle_int64_min();
        expected->bucket_argmax[(size_t)d] = any ? arg : -1;
    }
}

static inline bool opr_check_metadata(
    const OprRunSpec& run,
    const OprExpected& expected,
    const OprHostOutputsView& got,
    std::string* error) {
    const int N = run.N;
    const int radix_bits = run.radix_bits;
    const int num_digits = opr_num_digits(radix_bits);
    const int num_tiles = opr_num_tiles(N);

    for (int d = 0; d < num_digits; ++d) {
        if (got.digit_counts[d] != expected.digit_counts[(size_t)d]) {
            if (error) {
                std::ostringstream oss;
                oss << "digit_counts mismatch at d=" << d
                    << ": got " << got.digit_counts[d]
                    << ", expected " << expected.digit_counts[(size_t)d];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int d = 0; d <= num_digits; ++d) {
        if (got.digit_offsets[d] != expected.digit_offsets[(size_t)d]) {
            if (error) {
                std::ostringstream oss;
                oss << "digit_offsets mismatch at d=" << d
                    << ": got " << got.digit_offsets[d]
                    << ", expected " << expected.digit_offsets[(size_t)d];
                *error = oss.str();
            }
            return false;
        }
    }

    if (got.digit_offsets[0] != 0) {
        if (error) *error = "digit_offsets[0] must be zero";
        return false;
    }

    if (got.digit_offsets[num_digits] != N) {
        if (error) {
            std::ostringstream oss;
            oss << "digit_offsets[num_digits] must equal N: got "
                << got.digit_offsets[num_digits] << ", N=" << N;
            *error = oss.str();
        }
        return false;
    }

    for (int d = 0; d < num_digits; ++d) {
        if (got.digit_offsets[d + 1] - got.digit_offsets[d] != got.digit_counts[d]) {
            if (error) {
                std::ostringstream oss;
                oss << "digit offset/count invariant failed at d=" << d;
                *error = oss.str();
            }
            return false;
        }
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        for (int d = 0; d < num_digits; ++d) {
            const size_t idx = (size_t)tile * (size_t)num_digits + (size_t)d;

            if (got.tile_digit_offsets[idx] != expected.tile_digit_offsets[idx]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "tile_digit_offsets mismatch at tile=" << tile
                        << ", d=" << d
                        << ": got " << got.tile_digit_offsets[idx]
                        << ", expected " << expected.tile_digit_offsets[idx];
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

static inline bool opr_check_packed(
    const OprRunSpec& run,
    const OprExpected& expected,
    const OprHostOutputsView& got,
    std::string* error) {
    const int N = run.N;
    const int radix_bits = run.radix_bits;
    const int num_digits = opr_num_digits(radix_bits);

    for (int i = 0; i < N; ++i) {
        if (got.packed_key[i] != expected.packed_key[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_key mismatch at i=" << i
                    << ": got " << got.packed_key[i]
                    << ", expected " << expected.packed_key[(size_t)i];
                *error = oss.str();
            }
            return false;
        }

        if (got.packed_val[i] != expected.packed_val[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_val mismatch at i=" << i
                    << ": got " << got.packed_val[i]
                    << ", expected " << expected.packed_val[(size_t)i];
                *error = oss.str();
            }
            return false;
        }

        if (got.packed_src[i] != expected.packed_src[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_src mismatch at i=" << i
                    << ": got " << got.packed_src[i]
                    << ", expected " << expected.packed_src[(size_t)i];
                *error = oss.str();
            }
            return false;
        }

        if (got.packed_src[i] < 0 || got.packed_src[i] >= N) {
            if (error) {
                std::ostringstream oss;
                oss << "packed_src out of range at i=" << i
                    << ": src=" << got.packed_src[i];
                *error = oss.str();
            }
            return false;
        }
    }

    for (int d = 0; d < num_digits; ++d) {
        const int begin = got.digit_offsets[d];
        const int end = got.digit_offsets[d + 1];

        int prev_src = -1;

        for (int i = begin; i < end; ++i) {
            const int digit = opr_oracle_digit(got.packed_key[i], radix_bits);

            if (digit != d) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_key digit mismatch at i=" << i
                        << ": expected bucket " << d
                        << ", got digit " << digit;
                    *error = oss.str();
                }
                return false;
            }

            if (got.packed_src[i] <= prev_src) {
                if (error) {
                    std::ostringstream oss;
                    oss << "stability violation in bucket " << d
                        << " at packed index " << i
                        << ": prev_src=" << prev_src
                        << ", src=" << got.packed_src[i];
                    *error = oss.str();
                }
                return false;
            }

            prev_src = got.packed_src[i];
        }
    }

    return true;
}

static inline bool opr_check_reductions(
    const OprRunSpec& run,
    const OprExpected& expected,
    const OprHostOutputsView& got,
    std::string* error) {
    const int num_digits = opr_num_digits(run.radix_bits);

    for (int d = 0; d < num_digits; ++d) {
        if (got.bucket_sum[d] != expected.bucket_sum[(size_t)d]) {
            if (error) {
                std::ostringstream oss;
                oss << "bucket_sum mismatch at d=" << d
                    << ": got " << got.bucket_sum[d]
                    << ", expected " << expected.bucket_sum[(size_t)d];
                *error = oss.str();
            }
            return false;
        }

        if (got.bucket_max[d] != expected.bucket_max[(size_t)d]) {
            if (error) {
                std::ostringstream oss;
                oss << "bucket_max mismatch at d=" << d
                    << ": got " << got.bucket_max[d]
                    << ", expected " << expected.bucket_max[(size_t)d];
                *error = oss.str();
            }
            return false;
        }

        if (got.bucket_argmax[d] != expected.bucket_argmax[(size_t)d]) {
            if (error) {
                std::ostringstream oss;
                oss << "bucket_argmax mismatch at d=" << d
                    << ": got " << got.bucket_argmax[d]
                    << ", expected " << expected.bucket_argmax[(size_t)d];
                *error = oss.str();
            }
            return false;
        }

        if (expected.digit_counts[(size_t)d] == 0) {
            if (got.bucket_max[d] != opr_oracle_int64_min() || got.bucket_argmax[d] != -1) {
                if (error) {
                    std::ostringstream oss;
                    oss << "empty bucket rule failed at d=" << d;
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

static inline bool opr_check_all_outputs(
    const OprRunSpec& run,
    const OprExpected& expected,
    const OprHostOutputsView& got,
    std::string* error) {
    if (!opr_check_metadata(run, expected, got, error)) return false;
    if (!opr_check_packed(run, expected, got, error)) return false;
    if (!opr_check_reductions(run, expected, got, error)) return false;
    return true;
}

/*
INTERMEDIATE CHECKS REQUIRED BY GRADER

The grader should copy back and verify:

  1. tile_digit_offsets[num_tiles * num_digits]
     - exact equality against CPU oracle.
     - each entry is the bucket-major base:
         digit_offsets[d] + sum of previous tile counts for digit d.
     - this is mandatory graded metadata.

  2. digit_counts[num_digits]
     - exact equality against CPU oracle.

  3. digit_offsets[num_digits + 1]
     - exact equality against CPU oracle.
     - digit_offsets[0] == 0.
     - digit_offsets[d + 1] - digit_offsets[d] == digit_counts[d].
     - digit_offsets[num_digits] == N.

  4. packed_key[N], packed_val[N], packed_src[N]
     - exact equality against CPU oracle.
     - every bucket slice contains only that digit.
     - packed_src is strictly increasing within each bucket.

  5. bucket_sum[num_digits]
     - exact int64 two's-complement equality.

  6. bucket_max[num_digits]
     - exact int64 equality.
     - empty buckets must use INT64_MIN.

  7. bucket_argmax[num_digits]
     - exact equality.
     - first-occurrence max in stable bucket order.
     - empty buckets must use -1.

The grader should additionally enforce:
  - Sentinels around every output allocation.
  - Input immutability hash before/after solution_run.
  - Determinism replay with the same case repeated at least twice.
  - Held-out seed-shifted distributions from the declared shape family.
*/

#endif  // ONESWEEP_PARTITION_REDUCE_ORACLE_HPP_
