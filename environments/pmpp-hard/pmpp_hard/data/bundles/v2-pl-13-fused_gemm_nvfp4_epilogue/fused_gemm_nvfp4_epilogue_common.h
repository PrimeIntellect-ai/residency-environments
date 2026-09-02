// ============================================================================
// file: fused_gemm_nvfp4_epilogue_common.h
// ============================================================================

#ifndef FUSED_GEMM_NVFP4_EPILOGUE_COMMON_H_
#define FUSED_GEMM_NVFP4_EPILOGUE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define FGE_ABI_VERSION 1

#define FGE_MIN_M 64
#define FGE_MAX_M 4096
#define FGE_MIN_N 64
#define FGE_MAX_N 4096
#define FGE_MIN_K 64
#define FGE_MAX_K 4096

enum FgeActivation : int32_t {
    FGE_ACT_NONE = 0,
    FGE_ACT_RELU = 1,
    FGE_ACT_CLAMP_INT8_RANGE = 2
};

enum FgeDistributionId : int32_t {
    FGE_DIST_UNIFORM = 0,
    FGE_DIST_PEAKED = 1,
    FGE_DIST_MANY_ZERO = 2,
    FGE_DIST_MANY_TIES = 3
};

/*
CONTRACT: fused_gemm_nvfp4_epilogue

This is a deterministic FP4-code GEMM + fused epilogue task. It is intended as
a cuTLASS/CUTLASS task: the strong reference unpacks packed 4-bit E2M1-style
codes to int8 codebook values, runs a CUTLASS int8 GEMM, then applies an exact
integer epilogue.

The arithmetic is deliberately exact. There is no floating-point oracle.

Shape family:
  M, N, K in [64, 4096].
  Generated shapes are multiples of 64.
  K is even.

Input packing:
  a_packed stores M*K 4-bit codes in row-major order.
    logical code index = m*K + k

  b_packed stores K*N 4-bit codes in row-major order.
    logical code index = k*N + n

  Two codes per byte:
    even logical index -> low nibble
    odd logical index  -> high nibble

FP4-codebook:
  Code nibble c in [0,15] maps to an exact int8 value:

    c : value
    0 :   0
    1 :   1
    2 :   2
    3 :   3
    4 :   4
    5 :   6
    6 :   8
    7 :  12
    8 :   0
    9 :  -1
   10 :  -2
   11 :  -3
   12 :  -4
   13 :  -6
   14 :  -8
   15 : -12

GEMM:
  acc[m,n] = sum_{k=0..K-1} decode(a[m,k]) * decode(b[k,n])
  Accumulator is int32. Bounds in generated tests keep it safe.

Fused epilogue:
  Inputs:
    a_scale_q[M] int16
    b_scale_q[N] int16
    bias[N] int32

  y64 = int64(acc[m,n]) * int64(a_scale_q[m]) * int64(b_scale_q[n])
        + int64(bias[n])

  shifted = div_pow2_toward_zero(y64, epilogue_shift)

  epilogue_shift is in [0, 24] in generated tests.

  Activation:
    FGE_ACT_NONE:
      no activation

    FGE_ACT_RELU:
      if shifted < 0, shifted = 0

    FGE_ACT_CLAMP_INT8_RANGE:
      clamp shifted to [-128, 127]

  Final output:
    c_out[m,n] = clamp shifted to int32 range.

Outputs:
  c_out[M,N] int32 row-major.

Rules:
  - solution_run may not cudaMalloc/cudaFree.
  - solution_run may not perform synchronous host/device copies.
  - Reference uses CUTLASS for GEMM and custom kernels for unpack/epilogue.
  - Naive baseline is plain CUDA direct decode + dot product.
*/

struct alignas(8) FgeProblemSpec {
    int32_t abi_version;
    int32_t max_M;
    int32_t max_N;
    int32_t max_K;
    int32_t flags;
    int32_t reserved[11];
};

struct alignas(8) FgeRunSpec {
    int32_t abi_version;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t epilogue_shift;
    int32_t activation;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    int32_t reserved[7];
};

struct alignas(8) FgeInputs {
    const uint8_t* a_packed;
    const uint8_t* b_packed;
    const int16_t* a_scale_q;
    const int16_t* b_scale_q;
    const int32_t* bias;
};

struct alignas(8) FgeOutputs {
    int32_t* c_out;
};

static_assert(sizeof(FgeProblemSpec) == 64, "FgeProblemSpec layout drift");
static_assert(sizeof(FgeRunSpec) == 64, "FgeRunSpec layout drift");
static_assert(sizeof(FgeInputs) == 40, "FgeInputs layout drift");
static_assert(sizeof(FgeOutputs) == 8, "FgeOutputs layout drift");

static_assert(offsetof(FgeProblemSpec, abi_version) == 0, "layout");
static_assert(offsetof(FgeProblemSpec, max_M) == 4, "layout");
static_assert(offsetof(FgeProblemSpec, max_N) == 8, "layout");
static_assert(offsetof(FgeProblemSpec, max_K) == 12, "layout");
static_assert(offsetof(FgeProblemSpec, flags) == 16, "layout");

static inline size_t fge_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int fge_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline size_t fge_packed_bytes_for(size_t elements) {
    return (elements + 1) / 2;
}

static inline int fge_validate_problem_spec(const FgeProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != FGE_ABI_VERSION) return 0;
    if (spec->max_M < FGE_MIN_M || spec->max_M > FGE_MAX_M) return 0;
    if (spec->max_N < FGE_MIN_N || spec->max_N > FGE_MAX_N) return 0;
    if (spec->max_K < FGE_MIN_K || spec->max_K > FGE_MAX_K) return 0;
    if ((spec->max_K & 1) != 0) return 0;
    return 1;
}

static inline int fge_validate_run_spec(const FgeRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != FGE_ABI_VERSION) return 0;
    if (run->M < FGE_MIN_M || run->M > FGE_MAX_M) return 0;
    if (run->N < FGE_MIN_N || run->N > FGE_MAX_N) return 0;
    if (run->K < FGE_MIN_K || run->K > FGE_MAX_K) return 0;
    if ((run->K & 1) != 0) return 0;
    if (run->epilogue_shift < 0 || run->epilogue_shift > 24) return 0;
    if (run->activation < FGE_ACT_NONE || run->activation > FGE_ACT_CLAMP_INT8_RANGE) return 0;
    if (run->distribution_id < FGE_DIST_UNIFORM || run->distribution_id > FGE_DIST_MANY_TIES) return 0;
    return 1;
}

static inline int8_t fge_decode_code_host(uint8_t code) {
    static const int8_t table[16] = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12
    };
    return table[code & 15u];
}

static inline uint8_t fge_get_packed_code_host(const uint8_t* ptr, size_t logical_index) {
    const uint8_t byte = ptr[logical_index >> 1];
    if ((logical_index & 1u) == 0u) return byte & 0x0fu;
    return (byte >> 4) & 0x0fu;
}

static inline int64_t fge_div_pow2_toward_zero_host(int64_t x, int shift) {
    if (shift <= 0) return x;
    if (x >= 0) return x >> shift;
    return -(((-x) >> shift));
}

static inline int32_t fge_clamp_i32_host(int64_t x) {
    if (x > 2147483647LL) return 2147483647;
    if (x < -2147483647LL - 1LL) return static_cast<int32_t>(0x80000000u);
    return static_cast<int32_t>(x);
}

extern "C" size_t solution_workspace_bytes(const FgeProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const FgeProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const FgeRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // FUSED_GEMM_NVFP4_EPILOGUE_COMMON_H_
