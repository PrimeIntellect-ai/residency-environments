// PMPP_CANARY_08_e3d8367d59 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: group_limited_topk_reference.cu
// Parallel group-score reference + per-row hierarchical selection.
// ============================================================================

#include "group_limited_topk_common.h"

#include <cuda_runtime.h>

#include <float.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define GLT_GROUP_SCORE_BLOCK 256

struct GltReferenceState {
    GltProblemSpec spec;
};

static size_t glt_reference_workspace_bytes_for(const GltProblemSpec* spec) {
    (void)spec;
    return 128;
}

__device__ __forceinline__ int glt_expert_better_device(
    float cand_score,
    int cand_id,
    float old_score,
    int old_id) {
    return (cand_score > old_score) ||
           (cand_score == old_score && cand_id < old_id);
}

__device__ __forceinline__ int glt_group_better_device(
    float cand_score,
    int cand_id,
    float old_score,
    int old_id) {
    return (cand_score > old_score) ||
           (cand_score == old_score && cand_id < old_id);
}

__device__ __forceinline__ void glt_insert_expert_device(
    float score,
    int id,
    float* top_score,
    int* top_id,
    int k) {
    int pos = -1;

    for (int i = 0; i < k; ++i) {
        if (glt_expert_better_device(score, id, top_score[i], top_id[i])) {
            pos = i;
            break;
        }
    }

    if (pos < 0) return;

    for (int j = k - 1; j > pos; --j) {
        top_score[j] = top_score[j - 1];
        top_id[j] = top_id[j - 1];
    }

    top_score[pos] = score;
    top_id[pos] = id;
}

__device__ __forceinline__ void glt_insert_group_device(
    float score,
    int id,
    float* top_score,
    int* top_id,
    int k) {
    int pos = -1;

    for (int i = 0; i < k; ++i) {
        if (glt_group_better_device(score, id, top_score[i], top_id[i])) {
            pos = i;
            break;
        }
    }

    if (pos < 0) return;

    for (int j = k - 1; j > pos; --j) {
        top_score[j] = top_score[j - 1];
        top_id[j] = top_id[j - 1];
    }

    top_score[pos] = score;
    top_id[pos] = id;
}

__global__ void glt_ref_group_scores_kernel(
    int B,
    int V,
    int G,
    int group_k,
    const float* __restrict__ score,
    float* __restrict__ group_scores) {
    __shared__ float s_score[GLT_GROUP_SCORE_BLOCK * GLT_MAX_GROUP_K];
    __shared__ int s_id[GLT_GROUP_SCORE_BLOCK * GLT_MAX_GROUP_K];

    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;

    const int group_size = V / G;
    int k_eff = group_k < group_size ? group_k : group_size;
    if (k_eff > GLT_MAX_GROUP_K) k_eff = GLT_MAX_GROUP_K;

    float local_score[GLT_MAX_GROUP_K];
    int local_id[GLT_MAX_GROUP_K];

    #pragma unroll
    for (int i = 0; i < GLT_MAX_GROUP_K; ++i) {
        local_score[i] = -FLT_MAX;
        local_id[i] = INT_MAX;
    }

    const int start = group * group_size;
    const float* row_score = score + (size_t)row * (size_t)V;

    for (int off = tid; off < group_size; off += blockDim.x) {
        const int expert = start + off;
        const float s = row_score[expert];
        glt_insert_expert_device(s, expert, local_score, local_id, k_eff);
    }

    #pragma unroll
    for (int i = 0; i < GLT_MAX_GROUP_K; ++i) {
        s_score[tid * GLT_MAX_GROUP_K + i] = local_score[i];
        s_id[tid * GLT_MAX_GROUP_K + i] = local_id[i];
    }
    __syncthreads();

    if (tid == 0) {
        float top_score[GLT_MAX_GROUP_K];
        int top_id[GLT_MAX_GROUP_K];

        #pragma unroll
        for (int i = 0; i < GLT_MAX_GROUP_K; ++i) {
            top_score[i] = -FLT_MAX;
            top_id[i] = INT_MAX;
        }

        for (int t = 0; t < blockDim.x; ++t) {
            for (int i = 0; i < k_eff; ++i) {
                const float s = s_score[t * GLT_MAX_GROUP_K + i];
                const int id = s_id[t * GLT_MAX_GROUP_K + i];
                if (id != INT_MAX) {
                    glt_insert_expert_device(s, id, top_score, top_id, k_eff);
                }
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < k_eff; ++i) {
            sum += top_score[i];
        }

        group_scores[(size_t)row * (size_t)G + (size_t)group] = sum;
    }
}

__global__ void glt_ref_select_groups_kernel(
    int B,
    int G,
    int n_groups,
    const float* __restrict__ group_scores,
    int32_t* __restrict__ selected_group_ids) {
    const int row = blockIdx.x;
    if (row >= B || threadIdx.x != 0) return;

    float top_score[GLT_MAX_N_GROUPS];
    int top_id[GLT_MAX_N_GROUPS];

    #pragma unroll
    for (int i = 0; i < GLT_MAX_N_GROUPS; ++i) {
        top_score[i] = -FLT_MAX;
        top_id[i] = INT_MAX;
    }

    const float* row_scores = group_scores + (size_t)row * (size_t)G;

    for (int g = 0; g < G; ++g) {
        glt_insert_group_device(row_scores[g], g, top_score, top_id, n_groups);
    }

    int32_t* row_selected = selected_group_ids + (size_t)row * (size_t)n_groups;
    for (int i = 0; i < n_groups; ++i) {
        row_selected[i] = top_id[i];
    }
}

__global__ void glt_ref_final_select_kernel(
    int B,
    int V,
    int G,
    int n_groups,
    int final_k,
    const float* __restrict__ score,
    const int32_t* __restrict__ selected_group_ids,
    int32_t* __restrict__ survivor_expert_ids,
    int32_t* __restrict__ survivor_count,
    float* __restrict__ weights) {
    const int row = blockIdx.x;
    if (row >= B || threadIdx.x != 0) return;

    const int group_size = V / G;
    const int possible = n_groups * group_size;
    const int count = possible < final_k ? possible : final_k;

    float top_score[GLT_MAX_FINAL_K];
    int top_id[GLT_MAX_FINAL_K];

    #pragma unroll
    for (int i = 0; i < GLT_MAX_FINAL_K; ++i) {
        top_score[i] = -FLT_MAX;
        top_id[i] = INT_MAX;
    }

    const float* row_score = score + (size_t)row * (size_t)V;
    const int32_t* row_groups =
        selected_group_ids + (size_t)row * (size_t)n_groups;

    for (int gi = 0; gi < n_groups; ++gi) {
        const int group = row_groups[gi];
        const int start = group * group_size;

        for (int off = 0; off < group_size; ++off) {
            const int expert = start + off;
            glt_insert_expert_device(
                row_score[expert],
                expert,
                top_score,
                top_id,
                count);
        }
    }

    float max_s = -FLT_MAX;
    for (int i = 0; i < count; ++i) {
        if (top_score[i] > max_s) {
            max_s = top_score[i];
        }
    }

    float denom = 0.0f;
    for (int i = 0; i < count; ++i) {
        denom += expf(top_score[i] - max_s);
    }
    if (denom <= 0.0f) denom = 1.0f;

    int32_t* row_experts = survivor_expert_ids + (size_t)row * (size_t)final_k;
    float* row_weights = weights + (size_t)row * (size_t)final_k;

    survivor_count[row] = count;

    for (int i = 0; i < count; ++i) {
        row_experts[i] = top_id[i];
        row_weights[i] = expf(top_score[i] - max_s) / denom;
    }
}

extern "C" size_t solution_workspace_bytes(const GltProblemSpec* spec) {
    if (!glt_validate_problem_spec(spec)) return 0;
    return glt_reference_workspace_bytes_for(spec);
}

extern "C" cudaError_t solution_init(
    const GltProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!glt_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    GltReferenceState* st = static_cast<GltReferenceState*>(malloc(sizeof(GltReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memcpy(&st->spec, spec, sizeof(GltProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const GltRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !glt_validate_run_spec(run) || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    if (workspace_bytes < 128) {
        return cudaErrorInvalidValue;
    }

    GltReferenceState* st = static_cast<GltReferenceState*>(state);
    const GltInputs* in = static_cast<const GltInputs*>(inputs_void);
    GltOutputs* out = static_cast<GltOutputs*>(outputs_void);

    if (run->B > st->spec.max_B ||
        run->V > st->spec.max_V ||
        run->G > st->spec.max_G ||
        run->n_groups > st->spec.max_n_groups ||
        run->final_k > st->spec.max_final_k) {
        return cudaErrorInvalidValue;
    }

    if (!in->score || !in->uniform_u ||
        !out->selected_group_ids || !out->survivor_expert_ids ||
        !out->survivor_count || !out->weights || !out->group_scores) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaSuccess;

    dim3 group_grid(run->B, run->G, 1);
    glt_ref_group_scores_kernel<<<group_grid, GLT_GROUP_SCORE_BLOCK, 0, stream>>>(
        run->B,
        run->V,
        run->G,
        run->group_k,
        in->score,
        out->group_scores);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    glt_ref_select_groups_kernel<<<run->B, 1, 0, stream>>>(
        run->B,
        run->G,
        run->n_groups,
        out->group_scores,
        out->selected_group_ids);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    glt_ref_final_select_kernel<<<run->B, 1, 0, stream>>>(
        run->B,
        run->V,
        run->G,
        run->n_groups,
        run->final_k,
        in->score,
        out->selected_group_ids,
        out->survivor_expert_ids,
        out->survivor_count,
        out->weights);
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
