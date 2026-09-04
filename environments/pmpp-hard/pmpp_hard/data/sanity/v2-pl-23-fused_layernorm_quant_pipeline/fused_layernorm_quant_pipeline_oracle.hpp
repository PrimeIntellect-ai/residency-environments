// file: fused_layernorm_quant_pipeline_oracle.hpp

#ifndef FUSED_LAYERNORM_QUANT_PIPELINE_ORACLE_HPP_
#define FUSED_LAYERNORM_QUANT_PIPELINE_ORACLE_HPP_

#include "fused_layernorm_quant_pipeline_common.h"

#include <stdint.h>
#include <stddef.h>

#include <cmath>
#include <cfenv>
#include <sstream>
#include <string>
#include <vector>

struct FlqpHostInputsView {
    const float* x;
    const float* weight;
    const float* bias;
};

struct FlqpHostOutputsView {
    const int8_t* q_int8;
    const float* scale;
    const float* dequant;
    const int64_t* code_sum;
};

struct FlqpExpected {
    std::vector<int8_t> q_int8;
    std::vector<float> scale;
    std::vector<float> dequant;
    std::vector<int64_t> code_sum;
};

static inline float flqp_oracle_fadd(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float cv = av + bv;
    return cv;
}

static inline float flqp_oracle_fsub(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float cv = av - bv;
    return cv;
}

static inline float flqp_oracle_fmul(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float cv = av * bv;
    return cv;
}

static inline float flqp_oracle_fdiv(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float cv = av / bv;
    return cv;
}

static inline int32_t flqp_oracle_quantize(float y, float scale) {
    const float qf = flqp_oracle_fdiv(y, scale);

    // nearbyintf uses current rounding mode. The default C/C++ mode is
    // round-to-nearest-even, matching __float2int_rn.
    int32_t qi = static_cast<int32_t>(nearbyintf(qf));

    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;

    return qi;
}

static inline void flqp_cpu_oracle(
    const FlqpRunSpec& run,
    const FlqpHostInputsView& in,
    FlqpExpected* expected) {
    const int N = run.N;
    const int D = run.D;

    expected->q_int8.assign((size_t)N * (size_t)D, 0);
    expected->scale.assign((size_t)N, 1.0f);
    expected->dequant.assign((size_t)N * (size_t)D, 0.0f);
    expected->code_sum.assign((size_t)N, 0);

    for (int row = 0; row < N; ++row) {
        float partial_sum[FLQP_BLOCK_THREADS];
        float partial_sumsq[FLQP_BLOCK_THREADS];
        float partial_amax[FLQP_BLOCK_THREADS];
        int64_t partial_code_sum[FLQP_BLOCK_THREADS];

        for (int tid = 0; tid < FLQP_BLOCK_THREADS; ++tid) {
            partial_sum[tid] = 0.0f;
            partial_sumsq[tid] = 0.0f;

            for (int d = tid; d < D; d += FLQP_BLOCK_THREADS) {
                const float v = in.x[(size_t)row * (size_t)D + (size_t)d];
                partial_sum[tid] = flqp_oracle_fadd(partial_sum[tid], v);
                partial_sumsq[tid] =
                    flqp_oracle_fadd(partial_sumsq[tid], flqp_oracle_fmul(v, v));
            }
        }

        for (int stride = FLQP_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
            for (int tid = 0; tid < stride; ++tid) {
                partial_sum[tid] = flqp_oracle_fadd(partial_sum[tid], partial_sum[tid + stride]);
                partial_sumsq[tid] =
                    flqp_oracle_fadd(partial_sumsq[tid], partial_sumsq[tid + stride]);
            }
        }

        const float inv_D = flqp_oracle_fdiv(1.0f, static_cast<float>(D));
        const float mean = flqp_oracle_fmul(partial_sum[0], inv_D);

        float var = flqp_oracle_fsub(
            flqp_oracle_fmul(partial_sumsq[0], inv_D),
            flqp_oracle_fmul(mean, mean));
        if (var < 0.0f) var = 0.0f;

        const float inv = flqp_oracle_fdiv(1.0f, sqrtf(var + run.eps));

        for (int tid = 0; tid < FLQP_BLOCK_THREADS; ++tid) {
            partial_amax[tid] = 0.0f;

            for (int d = tid; d < D; d += FLQP_BLOCK_THREADS) {
                const float x = in.x[(size_t)row * (size_t)D + (size_t)d];
                const float centered = flqp_oracle_fsub(x, mean);
                const float normed = flqp_oracle_fmul(centered, inv);
                const float weighted = flqp_oracle_fmul(normed, in.weight[d]);
                const float y = flqp_oracle_fadd(weighted, in.bias[d]);
                const float ay = fabsf(y);

                if (ay > partial_amax[tid]) partial_amax[tid] = ay;
            }
        }

        for (int stride = FLQP_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
            for (int tid = 0; tid < stride; ++tid) {
                if (partial_amax[tid + stride] > partial_amax[tid]) {
                    partial_amax[tid] = partial_amax[tid + stride];
                }
            }
        }

        const float row_scale = (partial_amax[0] > 0.0f)
            ? flqp_oracle_fdiv(partial_amax[0], 127.0f)
            : 1.0f;

        expected->scale[(size_t)row] = row_scale;

        for (int tid = 0; tid < FLQP_BLOCK_THREADS; ++tid) {
            partial_code_sum[tid] = 0;

            for (int d = tid; d < D; d += FLQP_BLOCK_THREADS) {
                const size_t idx = (size_t)row * (size_t)D + (size_t)d;

                const float centered = flqp_oracle_fsub(in.x[idx], mean);
                const float normed = flqp_oracle_fmul(centered, inv);
                const float weighted = flqp_oracle_fmul(normed, in.weight[d]);
                const float y = flqp_oracle_fadd(weighted, in.bias[d]);

                const int32_t qi = flqp_oracle_quantize(y, row_scale);
                expected->q_int8[idx] = static_cast<int8_t>(qi);
                expected->dequant[idx] = flqp_oracle_fmul(static_cast<float>(qi), row_scale);
                partial_code_sum[tid] += static_cast<int64_t>(qi);
            }
        }

        for (int stride = FLQP_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
            for (int tid = 0; tid < stride; ++tid) {
                partial_code_sum[tid] += partial_code_sum[tid + stride];
            }
        }

        expected->code_sum[(size_t)row] = partial_code_sum[0];
    }
}

static inline bool flqp_check_q_and_sum(
    const FlqpRunSpec& run,
    const FlqpExpected& expected,
    const FlqpHostOutputsView& got,
    std::string* error) {
    const int N = run.N;
    const int D = run.D;

    for (int row = 0; row < N; ++row) {
        if (got.code_sum[row] != expected.code_sum[(size_t)row]) {
            if (error) {
                std::ostringstream oss;
                oss << "code_sum mismatch row=" << row
                    << ": got " << got.code_sum[row]
                    << ", expected " << expected.code_sum[(size_t)row];
                *error = oss.str();
            }
            return false;
        }

        for (int d = 0; d < D; ++d) {
            const size_t idx = (size_t)row * (size_t)D + (size_t)d;

            if (got.q_int8[idx] != expected.q_int8[idx]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "q_int8 mismatch row=" << row
                        << ", d=" << d
                        << ": got " << static_cast<int>(got.q_int8[idx])
                        << ", expected " << static_cast<int>(expected.q_int8[idx]);
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

static inline bool flqp_check_scale_and_dequant(
    const FlqpRunSpec& run,
    const FlqpExpected& expected,
    const FlqpHostOutputsView& got,
    std::string* error) {
    const int N = run.N;
    const int D = run.D;

    for (int row = 0; row < N; ++row) {
        const float exp_s = expected.scale[(size_t)row];
        const float got_s = got.scale[row];

        const float sdiff = fabsf(got_s - exp_s);
        const float stol = FLQP_SCALE_ATOL + FLQP_SCALE_RTOL * fabsf(exp_s);

        if (!(sdiff <= stol)) {
            if (error) {
                std::ostringstream oss;
                oss << "scale mismatch row=" << row
                    << ": got " << got_s
                    << ", expected " << exp_s
                    << ", diff=" << sdiff
                    << ", tol=" << stol;
                *error = oss.str();
            }
            return false;
        }

        for (int d = 0; d < D; ++d) {
            const size_t idx = (size_t)row * (size_t)D + (size_t)d;

            const float exp_v = expected.dequant[idx];
            const float got_v = got.dequant[idx];

            const float diff = fabsf(got_v - exp_v);
            const float tol = FLQP_DEQUANT_ATOL + FLQP_DEQUANT_RTOL * fabsf(exp_v);

            if (!(diff <= tol)) {
                if (error) {
                    std::ostringstream oss;
                    oss << "dequant mismatch row=" << row
                        << ", d=" << d
                        << ": got " << got_v
                        << ", expected " << exp_v
                        << ", diff=" << diff
                        << ", tol=" << tol;
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

static inline bool flqp_check_all_outputs(
    const FlqpRunSpec& run,
    const FlqpExpected& expected,
    const FlqpHostOutputsView& got,
    std::string* error) {
    if (!flqp_check_q_and_sum(run, expected, got, error)) return false;
    if (!flqp_check_scale_and_dequant(run, expected, got, error)) return false;
    return true;
}

/*
GRADER CHECKS

Exact:
  - q_int8[N,D]
  - code_sum[N]

Tolerance:
  - scale[N] with FLQP_SCALE_ATOL / FLQP_SCALE_RTOL
  - dequant[N,D] with FLQP_DEQUANT_ATOL / FLQP_DEQUANT_RTOL

Harness should also enforce:
  - output guard sentinels
  - input immutability
  - held-out seeds
  - outlier rows stressing amax
  - all-equal rows where variance is zero
  - D = 256, 1024, 4096
*/

#endif  // FUSED_LAYERNORM_QUANT_PIPELINE_ORACLE_HPP_
