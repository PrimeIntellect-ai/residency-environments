// PMPP_CANARY_09_2167a4f8b5 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: penalty_filter_sample_reference.cu
// Bitonic-sort reference implementation.
// ============================================================================

#include "penalty_filter_sample_common.h"

#include <cuda_runtime.h>

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct PfsReferenceState {
    PfsProblemSpec spec;
};

struct PfsWorkspaceLayout {
    int32_t* hist_count;      // max_B * max_V
    float* sorted_score;      // max_B * max_V
    int32_t* sorted_token;    // max_B * max_V
    size_t required_bytes;
};

static size_t pfs_reference_workspace_bytes_for(int max_B, int max_V) {
    size_t off = 0;

    off = pfs_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_B * (size_t)max_V;

    off = pfs_align_up_size(off, 128);
    off += sizeof(float) * (size_t)max_B * (size_t)max_V;

    off = pfs_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_B * (size_t)max_V;

    off = pfs_align_up_size(off, 128);
    return off;
}

static PfsWorkspaceLayout pfs_reference_make_layout(
    void* workspace,
    int max_B,
    int max_V) {
    PfsWorkspaceLayout layout{};
    char* base = static_cast<char*>(workspace);
    size_t off = 0;

    off = pfs_align_up_size(off, 128);
    layout.hist_count = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_B * (size_t)max_V;

    off = pfs_align_up_size(off, 128);
    layout.sorted_score = reinterpret_cast<float*>(base + off);
    off += sizeof(float) * (size_t)max_B * (size_t)max_V;

    off = pfs_align_up_size(off, 128);
    layout.sorted_token = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_B * (size_t)max_V;

    off = pfs_align_up_size(off, 128);
    layout.required_bytes = off;
    return layout;
}

__device__ __forceinline__ float pfs_sanitize_rp_device(float rp) {
    return rp > 1.0f ? rp : 1.0f;
}

__device__ __forceinline__ float pfs_sanitize_temp_device(float t) {
    return t > 1.0e-6f ? t : 1.0e-6f;
}

__device__ __forceinline__ float pfs_sanitize_min_p_device(float p) {
    if (p < 0.0f) return 0.0f;
    if (p >= 1.0f) return 0.9999999403953552f;
    return p;
}

__device__ __forceinline__ float pfs_sanitize_u_device(float u) {
    if (u < 0.0f) return 0.0f;
    if (u >= 1.0f) return 0.9999999403953552f;
    return u;
}

__device__ __forceinline__ int pfs_better_device(float a_score, int a_id, float b_score, int b_id) {
    return (a_score > b_score) || (a_score == b_score && a_id < b_id);
}

__device__ __forceinline__ float pfs_adjust_score_device(
    float logit,
    int count,
    float repetition_penalty,
    float frequency_penalty,
    float presence_penalty,
    float temperature) {
    float x = logit;

    if (count > 0) {
        const float rp = pfs_sanitize_rp_device(repetition_penalty);
        if (x > 0.0f) {
            x = x / rp;
        } else {
            x = x * rp;
        }

        x -= frequency_penalty * static_cast<float>(count);
        x -= presence_penalty;
    }

    return x / pfs_sanitize_temp_device(temperature);
}

__global__ void pfs_ref_count_history_kernel(
    int B,
    int V,
    int H,
    const int32_t* __restrict__ history_token,
    const int32_t* __restrict__ history_len,
    int32_t* __restrict__ hist_count) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * H;
    if (linear >= total) return;

    const int row = linear / H;
    const int h = linear - row * H;

    int len = history_len[row];
    if (len < 0) len = 0;
    if (len > H) len = H;

    if (h >= len) return;

    const int token = history_token[(size_t)row * (size_t)H + (size_t)h];
    if (token < 0 || token >= V) return;

    atomicAdd(&hist_count[(size_t)row * (size_t)V + (size_t)token], 1);
}

__global__ void pfs_ref_adjust_kernel(
    int B,
    int V,
    const float* __restrict__ logits,
    const int32_t* __restrict__ hist_count,
    const float* __restrict__ repetition_penalty,
    const float* __restrict__ frequency_penalty,
    const float* __restrict__ presence_penalty,
    const float* __restrict__ temperature,
    float* __restrict__ sorted_score,
    int32_t* __restrict__ sorted_token) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * V;
    if (linear >= total) return;

    const int row = linear / V;
    const int token = linear - row * V;

    const int count = hist_count[linear];

    sorted_score[linear] = pfs_adjust_score_device(
        logits[linear],
        count,
        repetition_penalty[row],
        frequency_penalty[row],
        presence_penalty[row],
        temperature[row]);

    sorted_token[linear] = token;
}

__global__ void pfs_ref_bitonic_stage_kernel(
    int B,
    int V,
    int k,
    int j,
    float* __restrict__ sorted_score,
    int32_t* __restrict__ sorted_token) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * V;
    if (linear >= total) return;

    const int row = linear / V;
    const int local = linear - row * V;
    const int partner_local = local ^ j;

    if (partner_local <= local || partner_local >= V) {
        return;
    }

    const int partner = row * V + partner_local;

    const float a_score = sorted_score[linear];
    const int a_id = sorted_token[linear];
    const float b_score = sorted_score[partner];
    const int b_id = sorted_token[partner];

    const int descending = ((local & k) == 0);

    int do_swap = 0;
    if (descending) {
        do_swap = pfs_better_device(b_score, b_id, a_score, a_id);
    } else {
        do_swap = pfs_better_device(a_score, a_id, b_score, b_id);
    }

    if (do_swap) {
        sorted_score[linear] = b_score;
        sorted_token[linear] = b_id;
        sorted_score[partner] = a_score;
        sorted_token[partner] = a_id;
    }
}

__global__ void pfs_ref_finalize_kernel(
    int B,
    int V,
    const float* __restrict__ min_p,
    const float* __restrict__ uniform_u,
    const float* __restrict__ sorted_score,
    const int32_t* __restrict__ sorted_token,
    int32_t* __restrict__ selected_token,
    int32_t* __restrict__ survivor_count,
    int32_t* __restrict__ packed_cand_token,
    float* __restrict__ packed_cand_prob) {
    const int row = blockIdx.x;
    if (row >= B || threadIdx.x != 0) return;

    const float* row_score = sorted_score + (size_t)row * (size_t)V;
    const int32_t* row_token = sorted_token + (size_t)row * (size_t)V;
    int32_t* row_out_token = packed_cand_token + (size_t)row * (size_t)V;
    float* row_out_prob = packed_cand_prob + (size_t)row * (size_t)V;

    const float threshold = pfs_sanitize_min_p_device(min_p[row]);
    const float max_score = row_score[0];

    int count = 0;

    if (threshold <= 0.0f) {
        count = V;
    } else {
        for (int i = 0; i < V; ++i) {
            const float rel = expf(row_score[i] - max_score);
            if (rel >= threshold) {
                ++count;
            } else {
                break;
            }
        }

        if (count <= 0) count = 1;
    }

    float denom = 0.0f;
    for (int i = 0; i < count; ++i) {
        denom += expf(row_score[i] - max_score);
    }
    if (denom <= 0.0f) denom = 1.0f;

    for (int i = 0; i < count; ++i) {
        row_out_token[i] = row_token[i];
        row_out_prob[i] = expf(row_score[i] - max_score) / denom;
    }

    const float u = pfs_sanitize_u_device(uniform_u[row]);
    float cdf = 0.0f;
    int selected = row_out_token[count - 1];

    for (int i = 0; i < count; ++i) {
        cdf += row_out_prob[i];

        if (cdf > u) {
            selected = row_out_token[i];
            break;
        }
    }

    survivor_count[row] = count;
    selected_token[row] = selected;
}

extern "C" size_t solution_workspace_bytes(const PfsProblemSpec* spec) {
    if (!pfs_validate_problem_spec(spec)) return 0;
    return pfs_reference_workspace_bytes_for(spec->max_B, spec->max_V);
}

extern "C" cudaError_t solution_init(
    const PfsProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!pfs_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    PfsReferenceState* st = static_cast<PfsReferenceState*>(malloc(sizeof(PfsReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memcpy(&st->spec, spec, sizeof(PfsProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const PfsRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !pfs_validate_run_spec(run) || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    PfsReferenceState* st = static_cast<PfsReferenceState*>(state);
    const PfsInputs* in = static_cast<const PfsInputs*>(inputs_void);
    PfsOutputs* out = static_cast<PfsOutputs*>(outputs_void);

    if (run->B > st->spec.max_B ||
        run->V > st->spec.max_V ||
        run->H > st->spec.max_H) {
        return cudaErrorInvalidValue;
    }

    if (!in->logits || !in->history_len ||
        !in->repetition_penalty || !in->frequency_penalty ||
        !in->presence_penalty || !in->temperature ||
        !in->min_p || !in->uniform_u ||
        !out->selected_token || !out->survivor_count ||
        !out->packed_cand_token || !out->packed_cand_prob) {
        return cudaErrorInvalidValue;
    }

    if (run->H > 0 && !in->history_token) {
        return cudaErrorInvalidValue;
    }

    PfsWorkspaceLayout layout = pfs_reference_make_layout(
        workspace,
        st->spec.max_B,
        st->spec.max_V);

    if (workspace_bytes < layout.required_bytes) {
        return cudaErrorInvalidValue;
    }

    const int B = run->B;
    const int V = run->V;
    const int H = run->H;
    const int total = B * V;

    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(
        layout.hist_count,
        0,
        sizeof(int32_t) * (size_t)B * (size_t)V,
        stream);
    if (err != cudaSuccess) return err;

    if (H > 0) {
        const int hist_total = B * H;
        const int block = 256;
        const int grid = pfs_ceil_div_int(hist_total, block);

        pfs_ref_count_history_kernel<<<grid, block, 0, stream>>>(
            B,
            V,
            H,
            in->history_token,
            in->history_len,
            layout.hist_count);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    {
        const int block = 256;
        const int grid = pfs_ceil_div_int(total, block);

        pfs_ref_adjust_kernel<<<grid, block, 0, stream>>>(
            B,
            V,
            in->logits,
            layout.hist_count,
            in->repetition_penalty,
            in->frequency_penalty,
            in->presence_penalty,
            in->temperature,
            layout.sorted_score,
            layout.sorted_token);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    const int sort_block = 256;
    const int sort_grid = pfs_ceil_div_int(total, sort_block);

    for (int k = 2; k <= V; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            pfs_ref_bitonic_stage_kernel<<<sort_grid, sort_block, 0, stream>>>(
                B,
                V,
                k,
                j,
                layout.sorted_score,
                layout.sorted_token);
            err = cudaPeekAtLastError();
            if (err != cudaSuccess) return err;
        }
    }

    pfs_ref_finalize_kernel<<<B, 1, 0, stream>>>(
        B,
        V,
        in->min_p,
        in->uniform_u,
        layout.sorted_score,
        layout.sorted_token,
        out->selected_token,
        out->survivor_count,
        out->packed_cand_token,
        out->packed_cand_prob);
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
