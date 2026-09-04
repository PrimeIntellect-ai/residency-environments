// ============================================================================
// file: fused_gemm_nvfp4_epilogue_oracle.hpp
// Independent exact CPU oracle + check helpers.
// ============================================================================

#ifndef FUSED_GEMM_NVFP4_EPILOGUE_ORACLE_HPP_
#define FUSED_GEMM_NVFP4_EPILOGUE_ORACLE_HPP_

#include "fused_gemm_nvfp4_epilogue_common.h"

#include <stdint.h>
#include <stddef.h>

#include <sstream>
#include <string>
#include <vector>

struct FgeHostInputsView {
    const uint8_t* a_packed;
    const uint8_t* b_packed;
    const int16_t* a_scale_q;
    const int16_t* b_scale_q;
    const int32_t* bias;
};

struct FgeHostOutputsView {
    const int32_t* c_out;
};

struct FgeExpected {
    std::vector<int32_t> c_out;
};

static inline int64_t fge_oracle_apply_epilogue(
    int32_t acc,
    int16_t a_scale,
    int16_t b_scale,
    int32_t bias,
    int shift,
    int activation) {
    int64_t y = static_cast<int64_t>(acc);
    y *= static_cast<int64_t>(a_scale);
    y *= static_cast<int64_t>(b_scale);
    y += static_cast<int64_t>(bias);

    y = fge_div_pow2_toward_zero_host(y, shift);

    if (activation == FGE_ACT_RELU && y < 0) {
        y = 0;
    } else if (activation == FGE_ACT_CLAMP_INT8_RANGE) {
        if (y > 127) y = 127;
        if (y < -128) y = -128;
    }

    return y;
}

static inline void fge_cpu_oracle(
    const FgeRunSpec& run,
    const FgeHostInputsView& in,
    FgeExpected* expected) {
    const int M = run.M;
    const int N = run.N;
    const int K = run.K;

    expected->c_out.assign((size_t)M * (size_t)N, 0);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;

            for (int k = 0; k < K; ++k) {
                const size_t a_index = (size_t)m * (size_t)K + (size_t)k;
                const size_t b_index = (size_t)k * (size_t)N + (size_t)n;

                const int32_t av = static_cast<int32_t>(
                    fge_decode_code_host(fge_get_packed_code_host(in.a_packed, a_index)));
                const int32_t bv = static_cast<int32_t>(
                    fge_decode_code_host(fge_get_packed_code_host(in.b_packed, b_index)));

                acc += av * bv;
            }

            const int64_t y = fge_oracle_apply_epilogue(
                acc,
                in.a_scale_q[m],
                in.b_scale_q[n],
                in.bias[n],
                run.epilogue_shift,
                run.activation);

            expected->c_out[(size_t)m * (size_t)N + (size_t)n] = fge_clamp_i32_host(y);
        }
    }
}

static inline bool fge_check_all_outputs(
    const FgeRunSpec& run,
    const FgeExpected& expected,
    const FgeHostOutputsView& got,
    std::string* error) {
    const int M = run.M;
    const int N = run.N;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const size_t idx = (size_t)m * (size_t)N + (size_t)n;

            if (got.c_out[idx] != expected.c_out[idx]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "c_out mismatch at m=" << m
                        << ", n=" << n
                        << ": got " << got.c_out[idx]
                        << ", expected " << expected.c_out[idx];
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

/*
GRADER CHECKS

Exact:
  - c_out[M,N] int32

Harness should also enforce:
  - output guard sentinels
  - input immutability
  - held-out shapes/distributions
  - cases with activation NONE/RELU/CLAMP_INT8_RANGE
  - many-zero and many-tie packed FP4 code distributions
  - bias and scale sign/magnitude variation
*/

#endif  // FUSED_GEMM_NVFP4_EPILOGUE_ORACLE_HPP_
