// PMPP_CANARY_25_db4747bf0b -- held-out canary; MUST NOT appear in any submission
// file: segmented_groupby_topk_reference.cu

#include "segmented_groupby_topk_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct SgtReferenceState {
    SgtProblemSpec spec;
};

struct SgtReferenceWorkspace {
    int32_t* filtered_offsets;
    int32_t* scatter_cursor;
    int32_t* compact_key;
    int32_t* compact_value;
    int32_t* compact_orig;
    size_t required_bytes;
};

static size_t sgtk_reference_workspace_bytes_for(int max_N, int max_G) {
    size_t off = 0;

    off = sgtk_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)(max_G + 1);

    off = sgtk_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_G;

    off = sgtk_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_N;

    off = sgtk_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_N;

    off = sgtk_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_N;

    off = sgtk_align_up_size(off, 128);
    return off;
}

static SgtReferenceWorkspace sgtk_reference_make_workspace(
    void* workspace,
    int max_N,
    int max_G) {
    SgtReferenceWorkspace layout{};
    char* base = static_cast<char*>(workspace);
    size_t off = 0;

    off = sgtk_align_up_size(off, 128);
    layout.filtered_offsets = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)(max_G + 1);

    off = sgtk_align_up_size(off, 128);
    layout.scatter_cursor = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_G;

    off = sgtk_align_up_size(off, 128);
    layout.compact_key = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_N;

    off = sgtk_align_up_size(off, 128);
    layout.compact_value = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_N;

    off = sgtk_align_up_size(off, 128);
    layout.compact_orig = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_N;

    off = sgtk_align_up_size(off, 128);
    layout.required_bytes = off;
    return layout;
}

__device__ __forceinline__ int sgtk_better_pair_device(
    int cand_key,
    int cand_orig,
    int best_key,
    int best_orig) {
    return (cand_key > best_key) ||
           (cand_key == best_key && cand_orig < best_orig);
}

__device__ __forceinline__ int sgtk_after_prev_device(
    int key,
    int orig,
    int prev_key,
    int prev_orig,
    int rank) {
    if (rank == 0) return 1;
    return (key < prev_key) || (key == prev_key && orig > prev_orig);
}

__global__ void sgtk_ref_count_kernel(
    int N,
    int G,
    const int32_t* __restrict__ group_id,
    const int32_t* __restrict__ value,
    int32_t* __restrict__ group_counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int g = group_id[idx];
    const int v = value[idx];

    if (v > 0 && g >= 0 && g < G) {
        atomicAdd(&group_counts[g], 1);
    }
}

__global__ void sgtk_ref_offsets_kernel(
    int G,
    int M,
    const int32_t* __restrict__ group_counts,
    int32_t* __restrict__ group_offsets,
    int32_t* __restrict__ kept_count,
    int32_t* __restrict__ filtered_offsets) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int filtered_acc = 0;
    int kept_acc = 0;

    filtered_offsets[0] = 0;
    group_offsets[0] = 0;

    for (int g = 0; g < G; ++g) {
        const int cnt = group_counts[g];
        const int kept = cnt < M ? cnt : M;

        kept_count[g] = kept;

        filtered_acc += cnt;
        kept_acc += kept;

        filtered_offsets[g + 1] = filtered_acc;
        group_offsets[g + 1] = kept_acc;
    }
}

__global__ void sgtk_ref_scatter_kernel(
    int N,
    int G,
    const int32_t* __restrict__ group_id,
    const int32_t* __restrict__ key,
    const int32_t* __restrict__ value,
    const int32_t* __restrict__ filtered_offsets,
    int32_t* __restrict__ scatter_cursor,
    int32_t* __restrict__ compact_key,
    int32_t* __restrict__ compact_value,
    int32_t* __restrict__ compact_orig) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int g = group_id[idx];
    const int v = value[idx];

    if (v <= 0 || g < 0 || g >= G) return;

    const int local = atomicAdd(&scatter_cursor[g], 1);
    const int pos = filtered_offsets[g] + local;

    compact_key[pos] = key[idx];
    compact_value[pos] = v;
    compact_orig[pos] = idx;
}

__global__ void sgtk_ref_select_reduce_kernel(
    int G,
    int M,
    const int32_t* __restrict__ group_counts,
    const int32_t* __restrict__ group_offsets,
    const int32_t* __restrict__ filtered_offsets,
    const int32_t* __restrict__ compact_key,
    const int32_t* __restrict__ compact_value,
    const int32_t* __restrict__ compact_orig,
    int32_t* __restrict__ packed_topk_origidx,
    int64_t* __restrict__ per_group_sum,
    int32_t* __restrict__ per_group_max,
    int32_t* __restrict__ per_group_argmax) {
    __shared__ int sh_key[SGTK_BLOCK_THREADS];
    __shared__ int sh_orig[SGTK_BLOCK_THREADS];
    __shared__ int sh_val[SGTK_BLOCK_THREADS];
    __shared__ int sh_prev_key;
    __shared__ int sh_prev_orig;

    const int g = blockIdx.x;
    const int tid = threadIdx.x;

    if (g >= G) return;

    const int cnt = group_counts[g];
    const int kept = cnt < M ? cnt : M;
    const int filtered_begin = filtered_offsets[g];
    const int filtered_end = filtered_offsets[g + 1];
    const int out_base = group_offsets[g];

    if (tid == 0) {
        sh_prev_key = INT_MAX;
        sh_prev_orig = -1;
    }
    __syncthreads();

    if (kept == 0) {
        if (tid == 0) {
            per_group_sum[g] = 0;
            per_group_max[g] = INT_MIN;
            per_group_argmax[g] = -1;
        }
        return;
    }

    uint64_t sum_bits = 0;
    int max_v = INT_MIN;
    int argmax = -1;

    for (int rank = 0; rank < kept; ++rank) {
        const int prev_key = sh_prev_key;
        const int prev_orig = sh_prev_orig;

        int best_key = INT_MIN;
        int best_orig = INT_MAX;
        int best_val = 0;

        for (int pos = filtered_begin + tid; pos < filtered_end; pos += blockDim.x) {
            const int k = compact_key[pos];
            const int orig = compact_orig[pos];

            if (!sgtk_after_prev_device(k, orig, prev_key, prev_orig, rank)) {
                continue;
            }

            if (sgtk_better_pair_device(k, orig, best_key, best_orig)) {
                best_key = k;
                best_orig = orig;
                best_val = compact_value[pos];
            }
        }

        sh_key[tid] = best_key;
        sh_orig[tid] = best_orig;
        sh_val[tid] = best_val;
        __syncthreads();

        for (int stride = SGTK_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const int other_key = sh_key[tid + stride];
                const int other_orig = sh_orig[tid + stride];

                if (sgtk_better_pair_device(other_key, other_orig, sh_key[tid], sh_orig[tid])) {
                    sh_key[tid] = other_key;
                    sh_orig[tid] = other_orig;
                    sh_val[tid] = sh_val[tid + stride];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int selected_orig = sh_orig[0];
            const int selected_key = sh_key[0];
            const int selected_val = sh_val[0];

            packed_topk_origidx[out_base + rank] = selected_orig;

            sum_bits += static_cast<uint64_t>(static_cast<int64_t>(selected_val));

            if (selected_val > max_v ||
                (selected_val == max_v && (argmax < 0 || selected_orig < argmax))) {
                max_v = selected_val;
                argmax = selected_orig;
            }

            sh_prev_key = selected_key;
            sh_prev_orig = selected_orig;
        }
        __syncthreads();
    }

    if (tid == 0) {
        per_group_sum[g] = static_cast<int64_t>(sum_bits);
        per_group_max[g] = max_v;
        per_group_argmax[g] = argmax;
    }
}

extern "C" size_t solution_workspace_bytes(const SgtProblemSpec* spec) {
    if (!sgtk_validate_problem_spec(spec)) return 0;
    return sgtk_reference_workspace_bytes_for(spec->max_N, spec->max_G);
}

extern "C" cudaError_t solution_init(
    const SgtProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!sgtk_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    SgtReferenceState* st =
        static_cast<SgtReferenceState*>(malloc(sizeof(SgtReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memcpy(&st->spec, spec, sizeof(SgtProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const SgtRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !sgtk_validate_run_spec(run) ||
        !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    SgtReferenceState* st = static_cast<SgtReferenceState*>(state);
    const SgtInputs* in = static_cast<const SgtInputs*>(inputs_void);
    SgtOutputs* out = static_cast<SgtOutputs*>(outputs_void);

    if (run->N > st->spec.max_N || run->G > st->spec.max_G) {
        return cudaErrorInvalidValue;
    }

    if (!in->group_id || !in->key || !in->value ||
        !out->group_counts || !out->group_offsets || !out->packed_topk_origidx ||
        !out->per_group_sum || !out->per_group_max || !out->per_group_argmax ||
        !out->kept_count) {
        return cudaErrorInvalidValue;
    }

    SgtReferenceWorkspace layout =
        sgtk_reference_make_workspace(workspace, st->spec.max_N, st->spec.max_G);

    if (workspace_bytes < layout.required_bytes) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(out->group_counts, 0, sizeof(int32_t) * (size_t)run->G, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(layout.scatter_cursor, 0, sizeof(int32_t) * (size_t)run->G, stream);
    if (err != cudaSuccess) return err;

    const int block = 256;
    const int grid = sgtk_ceil_div_int(run->N, block);

    sgtk_ref_count_kernel<<<grid, block, 0, stream>>>(
        run->N,
        run->G,
        in->group_id,
        in->value,
        out->group_counts);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    sgtk_ref_offsets_kernel<<<1, 1, 0, stream>>>(
        run->G,
        run->M,
        out->group_counts,
        out->group_offsets,
        out->kept_count,
        layout.filtered_offsets);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    sgtk_ref_scatter_kernel<<<grid, block, 0, stream>>>(
        run->N,
        run->G,
        in->group_id,
        in->key,
        in->value,
        layout.filtered_offsets,
        layout.scatter_cursor,
        layout.compact_key,
        layout.compact_value,
        layout.compact_orig);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    sgtk_ref_select_reduce_kernel<<<run->G, SGTK_BLOCK_THREADS, 0, stream>>>(
        run->G,
        run->M,
        out->group_counts,
        out->group_offsets,
        layout.filtered_offsets,
        layout.compact_key,
        layout.compact_value,
        layout.compact_orig,
        out->packed_topk_origidx,
        out->per_group_sum,
        out->per_group_max,
        out->per_group_argmax);
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
