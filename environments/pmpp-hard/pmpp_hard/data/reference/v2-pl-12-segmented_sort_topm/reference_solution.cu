// PMPP_CANARY_12_49d360a8e5 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: segmented_sort_topm_reference.cu
// Parallel per-segment repeated top-M selection. Exact, deterministic, stable.
// ============================================================================

#include "segmented_sort_topm_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct SstReferenceState {
    SstProblemSpec spec;
};

__device__ __forceinline__ int sst_better_pair_device(
    int cand_key,
    int cand_orig,
    int best_key,
    int best_orig) {
    return (cand_key > best_key) ||
           (cand_key == best_key && cand_orig < best_orig);
}

__device__ __forceinline__ int sst_after_prev_device(
    int key,
    int orig,
    int prev_key,
    int prev_orig,
    int rank) {
    if (rank == 0) return 1;
    return (key < prev_key) || (key == prev_key && orig > prev_orig);
}

__global__ void sst_ref_count_kernel(
    int S,
    int M,
    const int32_t* __restrict__ seg_offsets,
    int32_t* __restrict__ topm_count) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    const int begin = seg_offsets[s];
    const int end = seg_offsets[s + 1];
    int len = end - begin;
    if (len < 0) len = 0;

    topm_count[s] = len < M ? len : M;
}

__global__ void sst_ref_offsets_kernel(
    int S,
    const int32_t* __restrict__ topm_count,
    int32_t* __restrict__ topm_offsets) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int acc = 0;
    topm_offsets[0] = 0;

    for (int s = 0; s < S; ++s) {
        acc += topm_count[s];
        topm_offsets[s + 1] = acc;
    }
}

__global__ void sst_ref_select_kernel(
    int S,
    int M,
    const int32_t* __restrict__ seg_offsets,
    const int32_t* __restrict__ item_key,
    const int32_t* __restrict__ item_value,
    const int32_t* __restrict__ topm_offsets,
    int32_t* __restrict__ packed_topm_key,
    int32_t* __restrict__ packed_topm_value,
    int32_t* __restrict__ packed_topm_origidx,
    int64_t* __restrict__ seg_sum,
    int32_t* __restrict__ seg_max,
    int32_t* __restrict__ seg_argmax) {
    __shared__ int s_key[256];
    __shared__ int s_orig[256];
    __shared__ int s_val[256];
    __shared__ int s_prev_key;
    __shared__ int s_prev_orig;

    const int s = blockIdx.x;
    const int tid = threadIdx.x;

    if (s >= S) return;

    const int begin = seg_offsets[s];
    const int end = seg_offsets[s + 1];
    int len = end - begin;
    if (len < 0) len = 0;

    int count = len < M ? len : M;
    const int out_base = topm_offsets[s];

    unsigned long long sum_bits = 0ULL;
    int max_v = INT_MIN;
    int argmax_orig = -1;

    if (tid == 0) {
        s_prev_key = INT_MAX;
        s_prev_orig = -1;
    }
    __syncthreads();

    if (count == 0) {
        if (tid == 0) {
            seg_sum[s] = 0;
            seg_max[s] = INT_MIN;
            seg_argmax[s] = -1;
        }
        return;
    }

    for (int rank = 0; rank < count; ++rank) {
        const int prev_key = s_prev_key;
        const int prev_orig = s_prev_orig;

        int best_key = INT_MIN;
        int best_orig = INT_MAX;
        int best_val = 0;

        for (int idx = begin + tid; idx < end; idx += blockDim.x) {
            const int orig = idx - begin;
            const int k = item_key[idx];

            if (!sst_after_prev_device(k, orig, prev_key, prev_orig, rank)) {
                continue;
            }

            if (sst_better_pair_device(k, orig, best_key, best_orig)) {
                best_key = k;
                best_orig = orig;
                best_val = item_value[idx];
            }
        }

        s_key[tid] = best_key;
        s_orig[tid] = best_orig;
        s_val[tid] = best_val;
        __syncthreads();

        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const int other_key = s_key[tid + stride];
                const int other_orig = s_orig[tid + stride];

                if (sst_better_pair_device(other_key, other_orig, s_key[tid], s_orig[tid])) {
                    s_key[tid] = other_key;
                    s_orig[tid] = other_orig;
                    s_val[tid] = s_val[tid + stride];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int out = out_base + rank;
            const int selected_key = s_key[0];
            const int selected_orig = s_orig[0];
            const int selected_val = s_val[0];

            packed_topm_key[out] = selected_key;
            packed_topm_value[out] = selected_val;
            packed_topm_origidx[out] = selected_orig;

            sum_bits += static_cast<unsigned long long>(static_cast<long long>(selected_val));

            if (selected_val > max_v ||
                (selected_val == max_v && (argmax_orig < 0 || selected_orig < argmax_orig))) {
                max_v = selected_val;
                argmax_orig = selected_orig;
            }

            s_prev_key = selected_key;
            s_prev_orig = selected_orig;
        }

        __syncthreads();
    }

    if (tid == 0) {
        seg_sum[s] = static_cast<int64_t>(sum_bits);
        seg_max[s] = max_v;
        seg_argmax[s] = argmax_orig;
    }
}

extern "C" size_t solution_workspace_bytes(const SstProblemSpec* spec) {
    if (!sst_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const SstProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!sst_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    SstReferenceState* st = static_cast<SstReferenceState*>(malloc(sizeof(SstReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memcpy(&st->spec, spec, sizeof(SstProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const SstRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !sst_validate_run_spec(run) || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    SstReferenceState* st = static_cast<SstReferenceState*>(state);
    const SstInputs* in = static_cast<const SstInputs*>(inputs_void);
    SstOutputs* out = static_cast<SstOutputs*>(outputs_void);

    if (run->N > st->spec.max_N || run->S > st->spec.max_S) {
        return cudaErrorInvalidValue;
    }

    if (!in->seg_offsets || !in->item_key || !in->item_value ||
        !out->topm_count || !out->topm_offsets ||
        !out->packed_topm_key || !out->packed_topm_value ||
        !out->packed_topm_origidx || !out->seg_sum ||
        !out->seg_max || !out->seg_argmax) {
        return cudaErrorInvalidValue;
    }

    const int block = 256;
    const int grid_s = sst_ceil_div_int(run->S, block);

    cudaError_t err = cudaSuccess;

    sst_ref_count_kernel<<<grid_s, block, 0, stream>>>(
        run->S,
        run->M,
        in->seg_offsets,
        out->topm_count);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    sst_ref_offsets_kernel<<<1, 1, 0, stream>>>(
        run->S,
        out->topm_count,
        out->topm_offsets);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    sst_ref_select_kernel<<<run->S, 256, 0, stream>>>(
        run->S,
        run->M,
        in->seg_offsets,
        in->item_key,
        in->item_value,
        out->topm_offsets,
        out->packed_topm_key,
        out->packed_topm_value,
        out->packed_topm_origidx,
        out->seg_sum,
        out->seg_max,
        out->seg_argmax);
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
