// PMPP_CANARY_03_742613fc69 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: onesweep_partition_reduce_reference.cu
// Strong reference pipeline implementation.
// ============================================================================

#include "onesweep_partition_reduce_common.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct OprReferenceState {
    OprProblemSpec spec;
};

struct OprWorkspaceLayout {
    int32_t* tile_counts;   // max_tiles * max_digits
    size_t required_bytes;
};

static size_t opr_reference_workspace_bytes_for(int max_N, int max_radix_bits) {
    const int max_tiles = opr_num_tiles(max_N);
    const int max_digits = opr_num_digits(max_radix_bits);

    size_t off = 0;

    off = opr_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_tiles * (size_t)max_digits;

    off = opr_align_up_size(off, 128);
    return off;
}

static OprWorkspaceLayout opr_reference_make_layout(
    void* workspace,
    int max_N,
    int max_radix_bits) {
    OprWorkspaceLayout layout{};
    char* base = static_cast<char*>(workspace);
    const int max_tiles = opr_num_tiles(max_N);
    const int max_digits = opr_num_digits(max_radix_bits);

    size_t off = 0;

    off = opr_align_up_size(off, 128);
    layout.tile_counts = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_tiles * (size_t)max_digits;

    off = opr_align_up_size(off, 128);
    layout.required_bytes = off;
    return layout;
}

__device__ __forceinline__ int opr_digit_device(uint32_t key, int radix_bits) {
    return static_cast<int>(key >> (32 - radix_bits));
}

__global__ void opr_ref_tile_hist_kernel(
    int N,
    int radix_bits,
    const uint32_t* __restrict__ key,
    int32_t* __restrict__ tile_counts) {
    __shared__ int32_t s_counts[OPR_MAX_DIGITS];

    const int num_digits = 1 << radix_bits;
    const int tile = blockIdx.x;
    const int start = tile * OPR_TILE_ITEMS;
    int end = start + OPR_TILE_ITEMS;
    if (end > N) end = N;

    for (int d = threadIdx.x; d < num_digits; d += blockDim.x) {
        s_counts[d] = 0;
    }
    __syncthreads();

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const int d = opr_digit_device(key[i], radix_bits);
        atomicAdd(&s_counts[d], 1);
    }
    __syncthreads();

    for (int d = threadIdx.x; d < num_digits; d += blockDim.x) {
        tile_counts[(size_t)tile * (size_t)num_digits + (size_t)d] = s_counts[d];
    }
}

__global__ void opr_ref_offsets_kernel(
    int N,
    int radix_bits,
    const int32_t* __restrict__ tile_counts,
    int32_t* __restrict__ tile_digit_offsets,
    int32_t* __restrict__ digit_counts,
    int32_t* __restrict__ digit_offsets) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int num_digits = 1 << radix_bits;
    const int num_tiles = (N + OPR_TILE_ITEMS - 1) / OPR_TILE_ITEMS;

    int total = 0;
    digit_offsets[0] = 0;

    for (int d = 0; d < num_digits; ++d) {
        int count = 0;
        for (int tile = 0; tile < num_tiles; ++tile) {
            count += tile_counts[(size_t)tile * (size_t)num_digits + (size_t)d];
        }

        digit_counts[d] = count;
        total += count;
        digit_offsets[d + 1] = total;
    }

    for (int d = 0; d < num_digits; ++d) {
        int running = digit_offsets[d];

        for (int tile = 0; tile < num_tiles; ++tile) {
            const size_t idx = (size_t)tile * (size_t)num_digits + (size_t)d;
            tile_digit_offsets[idx] = running;
            running += tile_counts[idx];
        }
    }
}

__global__ void opr_ref_stable_scatter_kernel(
    int N,
    int radix_bits,
    const uint32_t* __restrict__ key,
    const int32_t* __restrict__ val,
    const int32_t* __restrict__ tile_digit_offsets,
    uint32_t* __restrict__ packed_key,
    int32_t* __restrict__ packed_val,
    int32_t* __restrict__ packed_src) {
    const int tile = blockIdx.x;
    if (threadIdx.x != 0) return;

    const int num_digits = 1 << radix_bits;
    const int start = tile * OPR_TILE_ITEMS;
    int end = start + OPR_TILE_ITEMS;
    if (end > N) end = N;

    int local_counts[OPR_MAX_DIGITS];

    #pragma unroll
    for (int d = 0; d < OPR_MAX_DIGITS; ++d) {
        if (d < num_digits) {
            local_counts[d] = 0;
        }
    }

    for (int i = start; i < end; ++i) {
        const uint32_t k = key[i];
        const int d = opr_digit_device(k, radix_bits);
        const size_t off_idx = (size_t)tile * (size_t)num_digits + (size_t)d;
        const int pos = tile_digit_offsets[off_idx] + local_counts[d];

        packed_key[pos] = k;
        packed_val[pos] = val[i];
        packed_src[pos] = i;

        ++local_counts[d];
    }
}

__device__ __forceinline__ long long opr_int64_min_device() {
    return (-9223372036854775807LL - 1LL);
}

__global__ void opr_ref_bucket_reduce_kernel(
    int radix_bits,
    const int32_t* __restrict__ digit_offsets,
    const int32_t* __restrict__ packed_val,
    const int32_t* __restrict__ packed_src,
    int64_t* __restrict__ bucket_sum,
    int64_t* __restrict__ bucket_max,
    int32_t* __restrict__ bucket_argmax) {
    constexpr int BLOCK = 256;

    __shared__ unsigned long long s_sum[BLOCK];
    __shared__ long long s_max[BLOCK];
    __shared__ int s_arg[BLOCK];
    __shared__ int s_any[BLOCK];

    const int d = blockIdx.x;
    const int begin = digit_offsets[d];
    const int end = digit_offsets[d + 1];

    unsigned long long local_sum = 0ULL;
    long long local_max = opr_int64_min_device();
    int local_arg = -1;
    int local_any = 0;

    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const long long v = static_cast<long long>(packed_val[i]);
        const int src = packed_src[i];

        local_sum += static_cast<unsigned long long>(v);

        if (!local_any || v > local_max || (v == local_max && src < local_arg)) {
            local_any = 1;
            local_max = v;
            local_arg = src;
        }
    }

    s_sum[threadIdx.x] = local_sum;
    s_max[threadIdx.x] = local_max;
    s_arg[threadIdx.x] = local_arg;
    s_any[threadIdx.x] = local_any;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];

            const int other_any = s_any[threadIdx.x + stride];
            const long long other_max = s_max[threadIdx.x + stride];
            const int other_arg = s_arg[threadIdx.x + stride];

            if (other_any &&
                (!s_any[threadIdx.x] ||
                 other_max > s_max[threadIdx.x] ||
                 (other_max == s_max[threadIdx.x] && other_arg < s_arg[threadIdx.x]))) {
                s_any[threadIdx.x] = 1;
                s_max[threadIdx.x] = other_max;
                s_arg[threadIdx.x] = other_arg;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        bucket_sum[d] = static_cast<int64_t>(s_sum[0]);
        bucket_max[d] = s_any[0] ? static_cast<int64_t>(s_max[0]) : static_cast<int64_t>(opr_int64_min_device());
        bucket_argmax[d] = s_any[0] ? s_arg[0] : -1;
    }
}

extern "C" size_t solution_workspace_bytes(const OprProblemSpec* spec) {
    if (!opr_validate_problem_spec(spec)) return 0;
    return opr_reference_workspace_bytes_for(spec->max_N, spec->max_radix_bits);
}

extern "C" cudaError_t solution_init(
    const OprProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!opr_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    OprReferenceState* st = static_cast<OprReferenceState*>(malloc(sizeof(OprReferenceState)));
    if (!st) {
        return cudaErrorMemoryAllocation;
    }

    memcpy(&st->spec, spec, sizeof(OprProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const OprRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !opr_validate_run_spec(run) || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    OprReferenceState* st = static_cast<OprReferenceState*>(state);
    const OprInputs* in = static_cast<const OprInputs*>(inputs_void);
    OprOutputs* out = static_cast<OprOutputs*>(outputs_void);

    if (run->N > st->spec.max_N ||
        run->radix_bits > st->spec.max_radix_bits) {
        return cudaErrorInvalidValue;
    }

    if (!in->key || !in->val ||
        !out->tile_digit_offsets || !out->digit_counts ||
        !out->digit_offsets || !out->packed_key ||
        !out->packed_val || !out->packed_src ||
        !out->bucket_sum || !out->bucket_max || !out->bucket_argmax) {
        return cudaErrorInvalidValue;
    }

    OprWorkspaceLayout layout = opr_reference_make_layout(
        workspace,
        st->spec.max_N,
        st->spec.max_radix_bits);

    if (workspace_bytes < layout.required_bytes) {
        return cudaErrorInvalidValue;
    }

    const int N = run->N;
    const int radix_bits = run->radix_bits;
    const int num_tiles = opr_num_tiles(N);
    const int num_digits = opr_num_digits(radix_bits);

    cudaError_t err = cudaSuccess;

    opr_ref_tile_hist_kernel<<<num_tiles, 256, 0, stream>>>(
        N,
        radix_bits,
        in->key,
        layout.tile_counts);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    opr_ref_offsets_kernel<<<1, 1, 0, stream>>>(
        N,
        radix_bits,
        layout.tile_counts,
        out->tile_digit_offsets,
        out->digit_counts,
        out->digit_offsets);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    opr_ref_stable_scatter_kernel<<<num_tiles, 1, 0, stream>>>(
        N,
        radix_bits,
        in->key,
        in->val,
        out->tile_digit_offsets,
        out->packed_key,
        out->packed_val,
        out->packed_src);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    opr_ref_bucket_reduce_kernel<<<num_digits, 256, 0, stream>>>(
        radix_bits,
        out->digit_offsets,
        out->packed_val,
        out->packed_src,
        out->bucket_sum,
        out->bucket_max,
        out->bucket_argmax);
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
