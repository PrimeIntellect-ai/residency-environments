// PMPP_CANARY_23_d482ae0072 -- held-out canary; MUST NOT appear in any submission
// file: fused_layernorm_quant_pipeline_reference.cu

#include "fused_layernorm_quant_pipeline_common.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct FlqpReferenceState {
    FlqpProblemSpec spec;
};

__device__ __forceinline__ float flqp_fadd_rn(float a, float b) {
    return __fadd_rn(a, b);
}

__device__ __forceinline__ float flqp_fsub_rn(float a, float b) {
    return __fadd_rn(a, -b);
}

__device__ __forceinline__ float flqp_fmul_rn(float a, float b) {
    return __fmul_rn(a, b);
}

__device__ __forceinline__ int32_t flqp_quantize_device(float y, float scale) {
    float qf = y / scale;
    int32_t qi = __float2int_rn(qf);

    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;

    return qi;
}

__global__ void flqp_reference_kernel(
    int N,
    int D,
    float eps,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int8_t* __restrict__ q_int8,
    float* __restrict__ scale,
    float* __restrict__ dequant,
    int64_t* __restrict__ code_sum) {
    __shared__ float sh_sum[FLQP_BLOCK_THREADS];
    __shared__ float sh_sumsq[FLQP_BLOCK_THREADS];
    __shared__ float sh_amax[FLQP_BLOCK_THREADS];
    __shared__ int64_t sh_code_sum[FLQP_BLOCK_THREADS];
    __shared__ float row_mean;
    __shared__ float row_inv;
    __shared__ float row_scale;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= N) return;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    const int base = row * D;

    for (int d = tid; d < D; d += blockDim.x) {
        const float v = x[base + d];
        local_sum = flqp_fadd_rn(local_sum, v);
        local_sumsq = flqp_fadd_rn(local_sumsq, flqp_fmul_rn(v, v));
    }

    sh_sum[tid] = local_sum;
    sh_sumsq[tid] = local_sumsq;
    __syncthreads();

    for (int stride = FLQP_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid] = flqp_fadd_rn(sh_sum[tid], sh_sum[tid + stride]);
            sh_sumsq[tid] = flqp_fadd_rn(sh_sumsq[tid], sh_sumsq[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float inv_D = 1.0f / static_cast<float>(D);
        const float mean = flqp_fmul_rn(sh_sum[0], inv_D);
        float var = flqp_fsub_rn(flqp_fmul_rn(sh_sumsq[0], inv_D), flqp_fmul_rn(mean, mean));
        if (var < 0.0f) var = 0.0f;

        row_mean = mean;
        row_inv = 1.0f / sqrtf(var + eps);
    }
    __syncthreads();

    float local_amax = 0.0f;

    for (int d = tid; d < D; d += blockDim.x) {
        const float centered = flqp_fsub_rn(x[base + d], row_mean);
        const float normed = flqp_fmul_rn(centered, row_inv);
        const float weighted = flqp_fmul_rn(normed, weight[d]);
        const float y = flqp_fadd_rn(weighted, bias[d]);
        const float ay = fabsf(y);
        if (ay > local_amax) local_amax = ay;
    }

    sh_amax[tid] = local_amax;
    __syncthreads();

    for (int stride = FLQP_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sh_amax[tid + stride] > sh_amax[tid]) {
                sh_amax[tid] = sh_amax[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float amax = sh_amax[0];
        row_scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        scale[row] = row_scale;
    }
    __syncthreads();

    int64_t local_code_sum = 0;

    for (int d = tid; d < D; d += blockDim.x) {
        const float centered = flqp_fsub_rn(x[base + d], row_mean);
        const float normed = flqp_fmul_rn(centered, row_inv);
        const float weighted = flqp_fmul_rn(normed, weight[d]);
        const float y = flqp_fadd_rn(weighted, bias[d]);

        const int32_t qi = flqp_quantize_device(y, row_scale);
        const int8_t q8 = static_cast<int8_t>(qi);

        q_int8[base + d] = q8;
        dequant[base + d] = flqp_fmul_rn(static_cast<float>(qi), row_scale);
        local_code_sum += static_cast<int64_t>(qi);
    }

    sh_code_sum[tid] = local_code_sum;
    __syncthreads();

    for (int stride = FLQP_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_code_sum[tid] += sh_code_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        code_sum[row] = sh_code_sum[0];
    }
}

extern "C" size_t solution_workspace_bytes(const FlqpProblemSpec* spec) {
    if (!flqp_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const FlqpProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!flqp_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    FlqpReferenceState* st =
        static_cast<FlqpReferenceState*>(malloc(sizeof(FlqpReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memcpy(&st->spec, spec, sizeof(FlqpProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const FlqpRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;

    if (!state || !flqp_validate_run_spec(run) || !inputs_void || !outputs_void) {
        return cudaErrorInvalidValue;
    }

    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    FlqpReferenceState* st = static_cast<FlqpReferenceState*>(state);
    const FlqpInputs* in = static_cast<const FlqpInputs*>(inputs_void);
    FlqpOutputs* out = static_cast<FlqpOutputs*>(outputs_void);

    if (run->N > st->spec.max_N || run->D > st->spec.max_D) {
        return cudaErrorInvalidValue;
    }

    if (!in->x || !in->weight || !in->bias ||
        !out->q_int8 || !out->scale || !out->dequant || !out->code_sum) {
        return cudaErrorInvalidValue;
    }

    flqp_reference_kernel<<<run->N, FLQP_BLOCK_THREADS, 0, stream>>>(
        run->N,
        run->D,
        run->eps,
        in->x,
        in->weight,
        in->bias,
        out->q_int8,
        out->scale,
        out->dequant,
        out->code_sum);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    (void)state;
    (void)stream;
    return cudaSuccess;
}

extern "C" void solution_destroy(void* state) {
    free(state);
}
