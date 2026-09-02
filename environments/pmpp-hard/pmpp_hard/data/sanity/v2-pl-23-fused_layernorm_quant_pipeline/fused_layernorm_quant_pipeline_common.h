// file: fused_layernorm_quant_pipeline_common.h

#ifndef FUSED_LAYERNORM_QUANT_PIPELINE_COMMON_H_
#define FUSED_LAYERNORM_QUANT_PIPELINE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define FLQP_ABI_VERSION 1

#define FLQP_MIN_N 4096
#define FLQP_MAX_N 131072
#define FLQP_MIN_D 256
#define FLQP_MAX_D 4096
#define FLQP_BLOCK_THREADS 256

#define FLQP_SCALE_ATOL 1.0e-5f
#define FLQP_SCALE_RTOL 1.0e-5f
#define FLQP_DEQUANT_ATOL 2.0e-4f
#define FLQP_DEQUANT_RTOL 2.0e-4f

enum FlqpDistributionId : int32_t {
    FLQP_DIST_UNIFORM = 0,
    FLQP_DIST_OUTLIERS = 1,
    FLQP_DIST_ALL_EQUAL = 2,
    FLQP_DIST_MANY_TIES = 3,
    FLQP_DIST_ZEROISH = 4
};

/*
CONTRACT: fused_layernorm_quant_pipeline

Per-row fused layernorm + symmetric int8 quantization + dequant/checksum.

Inputs:
  x[N, D] fp32 row-major
  weight[D] fp32
  bias[D] fp32

Shape family:
  N in [4096, 131072]
  D in {256, 1024, 4096}

Layernorm:
  mean = sum(x[row, d]) / D
  var  = sum(x[row, d]^2) / D - mean^2
  inv  = 1 / sqrt(max(var, 0) + eps)

  ln[d] = (x[row,d] - mean) * inv * weight[d] + bias[d]

Reduction convention:
  The reference uses one CUDA block per row with 256 threads.
  Per-thread strided partial sums are accumulated in fp32, then reduced by
  a deterministic binary-tree order. The CPU oracle mirrors this order.
  This pins exact int8-code grading despite fp32 reductions.

Quantization:
  amax = max_d abs(ln[d])
  scale[row] = amax / 127 if amax > 0 else 1.0

  q_raw = round_to_nearest_even(ln[d] / scale[row])
  q = clamp(q_raw, -127, 127)
  q_int8[row,d] = int8(q)

  dequant[row,d] = float(q) * scale[row]

  code_sum[row] = int64 sum_d q_int8[row,d]

Outputs:
  q_int8[N,D]      exact
  scale[N]         fp32 tolerance
  dequant[N,D]     fp32 tolerance
  code_sum[N]      exact

Rules:
  - solution_init may allocate persistent state.
  - solution_run may not cudaMalloc/cudaFree.
  - solution_run may launch kernels and use provided workspace.
  - Reference is fused: no full-precision normalized matrix is materialized.
  - Naive is an independent multi-pass implementation using workspace.
*/

struct alignas(8) FlqpProblemSpec {
    int32_t abi_version;
    int32_t max_N;
    int32_t max_D;
    int32_t flags;
    int32_t reserved[12];
};

struct alignas(8) FlqpRunSpec {
    int32_t abi_version;
    int32_t N;
    int32_t D;
    int32_t seed_id;
    int32_t distribution_id;
    int32_t case_id;
    float eps;
    int32_t reserved[9];
};

struct alignas(8) FlqpInputs {
    const float* x;
    const float* weight;
    const float* bias;
};

struct alignas(8) FlqpOutputs {
    int8_t* q_int8;
    float* scale;
    float* dequant;
    int64_t* code_sum;
};

static_assert(sizeof(FlqpProblemSpec) == 64, "FlqpProblemSpec layout drift");
static_assert(sizeof(FlqpRunSpec) == 64, "FlqpRunSpec layout drift");
static_assert(sizeof(FlqpInputs) == 24, "FlqpInputs layout drift");
static_assert(sizeof(FlqpOutputs) == 32, "FlqpOutputs layout drift");

static inline size_t flqp_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int flqp_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int flqp_valid_D(int D) {
    return D == 256 || D == 1024 || D == 4096;
}

static inline int flqp_validate_problem_spec(const FlqpProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != FLQP_ABI_VERSION) return 0;
    if (spec->max_N < FLQP_MIN_N || spec->max_N > FLQP_MAX_N) return 0;
    if (!flqp_valid_D(spec->max_D)) return 0;
    return 1;
}

static inline int flqp_validate_run_spec(const FlqpRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != FLQP_ABI_VERSION) return 0;
    if (run->N < FLQP_MIN_N || run->N > FLQP_MAX_N) return 0;
    if (!flqp_valid_D(run->D)) return 0;
    if (!(run->eps >= 0.0f)) return 0;
    if (run->distribution_id < FLQP_DIST_UNIFORM ||
        run->distribution_id > FLQP_DIST_ZEROISH) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const FlqpProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const FlqpProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const FlqpRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // FUSED_LAYERNORM_QUANT_PIPELINE_COMMON_H_
