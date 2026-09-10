// PMPP_CANARY_13_caa4e59080 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: fused_gemm_nvfp4_epilogue_reference.cu
// CUTLASS GEMM reference path:
//   packed FP4-code unpack -> CUTLASS int8 GEMM -> exact fused int epilogue.
// ============================================================================

#include "fused_gemm_nvfp4_epilogue_common.h"

#include <cuda_runtime.h>

#include <stdlib.h>
#include <string.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/arch/mma.h>

struct FgeReferenceState {
    FgeProblemSpec spec;
};

struct FgeReferenceWorkspace {
    int8_t* a_i8;              // max_M * max_K row-major
    int8_t* b_i8_colmajor;     // max_N * max_K column-major view of KxN B
    int32_t* acc_i32;          // max_M * max_N row-major
    void* cutlass_workspace;
    size_t required_bytes;
};

static size_t fge_reference_workspace_bytes_for(
    int max_M,
    int max_N,
    int max_K) {
    size_t off = 0;

    off = fge_align_up_size(off, 128);
    off += sizeof(int8_t) * (size_t)max_M * (size_t)max_K;

    off = fge_align_up_size(off, 128);
    off += sizeof(int8_t) * (size_t)max_N * (size_t)max_K;

    off = fge_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_M * (size_t)max_N;

    off = fge_align_up_size(off, 128);
    off += 4u * 1024u * 1024u;

    off = fge_align_up_size(off, 128);
    return off;
}

static FgeReferenceWorkspace fge_reference_make_workspace(
    void* workspace,
    int max_M,
    int max_N,
    int max_K) {
    FgeReferenceWorkspace layout{};
    char* base = static_cast<char*>(workspace);
    size_t off = 0;

    off = fge_align_up_size(off, 128);
    layout.a_i8 = reinterpret_cast<int8_t*>(base + off);
    off += sizeof(int8_t) * (size_t)max_M * (size_t)max_K;

    off = fge_align_up_size(off, 128);
    layout.b_i8_colmajor = reinterpret_cast<int8_t*>(base + off);
    off += sizeof(int8_t) * (size_t)max_N * (size_t)max_K;

    off = fge_align_up_size(off, 128);
    layout.acc_i32 = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_M * (size_t)max_N;

    off = fge_align_up_size(off, 128);
    layout.cutlass_workspace = reinterpret_cast<void*>(base + off);
    off += 4u * 1024u * 1024u;

    off = fge_align_up_size(off, 128);
    layout.required_bytes = off;
    return layout;
}

__device__ __forceinline__ int8_t fge_decode_code_device(uint8_t code) {
    static const int8_t table[16] = {
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12
    };
    return table[code & 15u];
}

__device__ __forceinline__ uint8_t fge_get_code_device(
    const uint8_t* ptr,
    size_t logical_index) {
    const uint8_t byte = ptr[logical_index >> 1];
    if ((logical_index & 1u) == 0u) return byte & 0x0fu;
    return (byte >> 4) & 0x0fu;
}

__device__ __forceinline__ int64_t fge_div_pow2_toward_zero_device(int64_t x, int shift) {
    if (shift <= 0) return x;
    if (x >= 0) return x >> shift;
    return -(((-x) >> shift));
}

__device__ __forceinline__ int32_t fge_clamp_i32_device(int64_t x) {
    if (x > 2147483647LL) return 2147483647;
    if (x < -2147483647LL - 1LL) return static_cast<int32_t>(0x80000000u);
    return static_cast<int32_t>(x);
}

__global__ void fge_ref_unpack_a_kernel(
    int M,
    int K,
    const uint8_t* __restrict__ a_packed,
    int8_t* __restrict__ a_i8) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * K;
    if (idx >= total) return;

    const uint8_t code = fge_get_code_device(a_packed, static_cast<size_t>(idx));
    a_i8[idx] = fge_decode_code_device(code);
}

__global__ void fge_ref_unpack_b_colmajor_kernel(
    int K,
    int N,
    const uint8_t* __restrict__ b_packed,
    int8_t* __restrict__ b_i8_colmajor) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = K * N;
    if (idx >= total) return;

    const int k = idx / N;
    const int n = idx - k * N;

    const uint8_t code = fge_get_code_device(b_packed, static_cast<size_t>(idx));
    b_i8_colmajor[(size_t)n * (size_t)K + (size_t)k] = fge_decode_code_device(code);
}

__global__ void fge_ref_epilogue_kernel(
    int M,
    int N,
    int shift,
    int activation,
    const int32_t* __restrict__ acc_i32,
    const int16_t* __restrict__ a_scale_q,
    const int16_t* __restrict__ b_scale_q,
    const int32_t* __restrict__ bias,
    int32_t* __restrict__ c_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int m = idx / N;
    const int n = idx - m * N;

    int64_t y = static_cast<int64_t>(acc_i32[idx]);
    y *= static_cast<int64_t>(a_scale_q[m]);
    y *= static_cast<int64_t>(b_scale_q[n]);
    y += static_cast<int64_t>(bias[n]);

    y = fge_div_pow2_toward_zero_device(y, shift);

    if (activation == FGE_ACT_RELU && y < 0) {
        y = 0;
    } else if (activation == FGE_ACT_CLAMP_INT8_RANGE) {
        if (y > 127) y = 127;
        if (y < -128) y = -128;
    }

    c_out[idx] = fge_clamp_i32_device(y);
}

using FgeCutlassGemm = cutlass::gemm::device::Gemm<
    int8_t,
    cutlass::layout::RowMajor,
    int8_t,
    cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4>;

extern "C" size_t solution_workspace_bytes(const FgeProblemSpec* spec) {
    if (!fge_validate_problem_spec(spec)) return 0;
    return fge_reference_workspace_bytes_for(spec->max_M, spec->max_N, spec->max_K);
}

extern "C" cudaError_t solution_init(
    const FgeProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!fge_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    FgeReferenceState* st = static_cast<FgeReferenceState*>(malloc(sizeof(FgeReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memcpy(&st->spec, spec, sizeof(FgeProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const FgeRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !fge_validate_run_spec(run) || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    FgeReferenceState* st = static_cast<FgeReferenceState*>(state);
    const FgeInputs* in = static_cast<const FgeInputs*>(inputs_void);
    FgeOutputs* out = static_cast<FgeOutputs*>(outputs_void);

    if (run->M > st->spec.max_M || run->N > st->spec.max_N || run->K > st->spec.max_K) {
        return cudaErrorInvalidValue;
    }

    if (!in->a_packed || !in->b_packed || !in->a_scale_q ||
        !in->b_scale_q || !in->bias || !out->c_out) {
        return cudaErrorInvalidValue;
    }

    FgeReferenceWorkspace layout =
        fge_reference_make_workspace(workspace, st->spec.max_M, st->spec.max_N, st->spec.max_K);

    if (workspace_bytes < layout.required_bytes) return cudaErrorInvalidValue;

    const int M = run->M;
    const int N = run->N;
    const int K = run->K;

    cudaError_t err = cudaSuccess;

    const int block = 256;
    const int a_total = M * K;
    const int b_total = K * N;

    fge_ref_unpack_a_kernel<<<fge_ceil_div_int(a_total, block), block, 0, stream>>>(
        M, K, in->a_packed, layout.a_i8);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    fge_ref_unpack_b_colmajor_kernel<<<fge_ceil_div_int(b_total, block), block, 0, stream>>>(
        K, N, in->b_packed, layout.b_i8_colmajor);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    FgeCutlassGemm gemm_op;

    FgeCutlassGemm::Arguments args(
        {M, N, K},
        {layout.a_i8, K},
        {layout.b_i8_colmajor, K},
        {layout.acc_i32, N},
        {layout.acc_i32, N},
        {1, 0});

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    size_t gemm_workspace_bytes = FgeCutlassGemm::get_workspace_size(args);
    const size_t cutlass_region_bytes = 4u * 1024u * 1024u;

    if (gemm_workspace_bytes > cutlass_region_bytes) {
        return cudaErrorMemoryAllocation;
    }

    status = gemm_op(args, layout.cutlass_workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    const int c_total = M * N;

    fge_ref_epilogue_kernel<<<fge_ceil_div_int(c_total, block), block, 0, stream>>>(
        M,
        N,
        run->epilogue_shift,
        run->activation,
        layout.acc_i32,
        in->a_scale_q,
        in->b_scale_q,
        in->bias,
        out->c_out);
    err = cudaPeekAtLastError();
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
