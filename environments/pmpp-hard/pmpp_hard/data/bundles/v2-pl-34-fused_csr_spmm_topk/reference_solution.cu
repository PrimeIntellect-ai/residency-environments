// PMPP_CANARY_34_32f9ff8a0e -- held-out canary; MUST NOT appear in any submission
// file: fused_csr_spmm_topk_reference.cu

#include "fused_csr_spmm_topk_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct FcstReferenceState {
    FcstProblemSpec spec;
};

__device__ __forceinline__ int fcst_better_pair_device(
    int64_t cand_val,
    int32_t cand_col,
    int64_t best_val,
    int32_t best_col) {
    if (best_col < 0) return 1;
    return (cand_val > best_val) ||
           (cand_val == best_val && cand_col < best_col);
}

__device__ __forceinline__ int64_t fcst_add_i64_wrap_device(int64_t a, int64_t b) {
    uint64_t ua = static_cast<uint64_t>(a);
    ua += static_cast<uint64_t>(b);
    return static_cast<int64_t>(ua);
}

__device__ __forceinline__ int64_t fcst_spmm_value_device(
    int row_begin,
    int row_end,
    int K,
    int N,
    int col_n,
    const int32_t* __restrict__ col_indices,
    const int32_t* __restrict__ vals,
    const int32_t* __restrict__ dense_b) {
    uint64_t acc = 0;

    for (int e = row_begin; e < row_end; ++e) {
        const int k = col_indices[e];

        if (k >= 0 && k < K) {
            const int64_t product =
                static_cast<int64_t>(vals[e]) *
                static_cast<int64_t>(dense_b[(size_t)k * (size_t)N + (size_t)col_n]);

            acc += static_cast<uint64_t>(product);
        }
    }

    return static_cast<int64_t>(acc);
}

__device__ void fcst_insert_top_device(
    int M,
    int32_t cand_col,
    int64_t cand_val,
    int32_t* __restrict__ cols,
    int64_t* __restrict__ values) {
    if (cand_col < 0) return;

    int pos = M;

    for (int i = 0; i < M; ++i) {
        if (fcst_better_pair_device(cand_val, cand_col, values[i], cols[i])) {
            pos = i;
            break;
        }
    }

    if (pos >= M) return;

    for (int j = M - 1; j > pos; --j) {
        values[j] = values[j - 1];
        cols[j] = cols[j - 1];
    }

    values[pos] = cand_val;
    cols[pos] = cand_col;
}


// Strict total order over (value,col) with padding (col<0) ranked last.
// Real items always outrank padding, matching fcst_insert_top_device semantics
// (a padding candidate is never inserted; any real candidate beats a padding slot).
__device__ __forceinline__ bool fcst_rank_before(
    int64_t cv, int32_t cc, int64_t ov, int32_t oc) {
    if (cc < 0) return false;   // candidate is padding: never before anything
    if (oc < 0) return true;    // other is padding, candidate real: before
    return (cv > ov) || (cv == ov && cc < oc);
}

// Merge two sorted-descending top-M lists (a, b) into the top-M of their union.
__device__ __forceinline__ void fcst_merge_two(
    int M,
    const int32_t* __restrict__ aC, const int64_t* __restrict__ aV,
    const int32_t* __restrict__ bC, const int64_t* __restrict__ bV,
    int32_t* __restrict__ outC, int64_t* __restrict__ outV) {
    int i = 0;
    int j = 0;
    for (int k = 0; k < M; ++k) {
        const int64_t av = aV[i];
        const int32_t ac = aC[i];
        const int64_t bv = bV[j];
        const int32_t bc = bC[j];
        if (fcst_rank_before(bv, bc, av, ac)) {
            outC[k] = bc;
            outV[k] = bv;
            ++j;
        } else {
            outC[k] = ac;
            outV[k] = av;
            ++i;
        }
    }
}

__global__ void fcst_reference_kernel(
    int rows,
    int K,
    int N,
    int M,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const int32_t* __restrict__ vals,
    const int32_t* __restrict__ dense_b,
    int32_t* __restrict__ topm_cols,
    int64_t* __restrict__ topm_vals,
    int32_t* __restrict__ topm_count,
    int64_t* __restrict__ row_sum,
    int64_t* __restrict__ row_max,
    int32_t* __restrict__ row_argmax,
    int32_t* __restrict__ row_nnz,
    int32_t* __restrict__ row_nnz_prefix) {
    __shared__ int64_t sh_vals[FCST_REF_THREADS * FCST_MAX_M];
    __shared__ int32_t sh_cols[FCST_REF_THREADS * FCST_MAX_M];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= rows || tid >= FCST_REF_THREADS) return;

    const int row_begin = row_offsets[row];
    const int row_end = row_offsets[row + 1];

    if (tid == 0) {
        row_nnz[row] = row_end - row_begin;
        row_nnz_prefix[row] = row_begin;
        if (row == rows - 1) {
            row_nnz_prefix[rows] = row_end;
        }
    }

    int32_t local_cols[FCST_MAX_M];
    int64_t local_vals[FCST_MAX_M];

    for (int i = 0; i < FCST_MAX_M; ++i) {
        local_cols[i] = -1;
        local_vals[i] = INT64_MIN;
    }

    for (int n = tid; n < N; n += FCST_REF_THREADS) {
        const int64_t v = fcst_spmm_value_device(
            row_begin,
            row_end,
            K,
            N,
            n,
            col_indices,
            vals,
            dense_b);

        fcst_insert_top_device(M, n, v, local_cols, local_vals);
    }

    for (int i = 0; i < M; ++i) {
        const int slot = tid * FCST_MAX_M + i;
        sh_cols[slot] = local_cols[i];
        sh_vals[slot] = local_vals[i];
    }

    __syncthreads();

    // Parallel tree reduction: merge the FCST_REF_THREADS per-thread sorted
    // top-M lists pairwise down to a single global top-M held in list 0.
    // Byte-exact with the serial insertion merge because the ranking is a
    // strict total order (unique col tiebreak) with padding ranked last.
    for (int half = FCST_REF_THREADS >> 1; half >= 1; half >>= 1) {
        int32_t merged_cols[FCST_MAX_M];
        int64_t merged_vals[FCST_MAX_M];

        const bool active = (tid < half);
        if (active) {
            const int a_base = tid * FCST_MAX_M;
            const int b_base = (tid + half) * FCST_MAX_M;
            fcst_merge_two(
                M,
                &sh_cols[a_base], &sh_vals[a_base],
                &sh_cols[b_base], &sh_vals[b_base],
                merged_cols, merged_vals);
        }

        __syncthreads();

        if (active) {
            const int a_base = tid * FCST_MAX_M;
            for (int i = 0; i < M; ++i) {
                sh_cols[a_base + i] = merged_cols[i];
                sh_vals[a_base + i] = merged_vals[i];
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        const int32_t* final_cols = &sh_cols[0];
        const int64_t* final_vals = &sh_vals[0];

        const int count = M < N ? M : N;
        topm_count[row] = count;

        uint64_t sum_bits = 0;
        int64_t max_v = INT64_MIN;
        int32_t argmax = -1;

        for (int i = 0; i < M; ++i) {
            const size_t out_idx = (size_t)row * (size_t)M + (size_t)i;

            if (i < count) {
                topm_cols[out_idx] = final_cols[i];
                topm_vals[out_idx] = final_vals[i];

                sum_bits += static_cast<uint64_t>(final_vals[i]);

                if (fcst_better_pair_device(final_vals[i], final_cols[i], max_v, argmax)) {
                    max_v = final_vals[i];
                    argmax = final_cols[i];
                }
            } else {
                topm_cols[out_idx] = -1;
                topm_vals[out_idx] = INT64_MIN;
            }
        }

        row_sum[row] = static_cast<int64_t>(sum_bits);
        row_max[row] = count > 0 ? max_v : INT64_MIN;
        row_argmax[row] = count > 0 ? argmax : -1;
    }
}

extern "C" size_t solution_workspace_bytes(const FcstProblemSpec* spec) {
    if (!fcst_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const FcstProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!fcst_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    FcstReferenceState* st =
        static_cast<FcstReferenceState*>(malloc(sizeof(FcstReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memcpy(&st->spec, spec, sizeof(FcstProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const FcstRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;

    if (!state || !fcst_validate_run_spec(run) || !inputs_void || !outputs_void) {
        return cudaErrorInvalidValue;
    }

    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    FcstReferenceState* st = static_cast<FcstReferenceState*>(state);
    const FcstInputs* in = static_cast<const FcstInputs*>(inputs_void);
    FcstOutputs* out = static_cast<FcstOutputs*>(outputs_void);

    if (run->rows > st->spec.max_rows ||
        run->K > st->spec.max_K ||
        run->N > st->spec.max_N ||
        run->nnz > st->spec.max_nnz ||
        run->M > st->spec.max_M) {
        return cudaErrorInvalidValue;
    }

    if (!in->row_offsets || !in->col_indices || !in->vals || !in->dense_b ||
        !out->topm_cols || !out->topm_vals || !out->topm_count ||
        !out->row_sum || !out->row_max || !out->row_argmax ||
        !out->row_nnz || !out->row_nnz_prefix) {
        return cudaErrorInvalidValue;
    }

    fcst_reference_kernel<<<run->rows, FCST_REF_THREADS, 0, stream>>>(
        run->rows,
        run->K,
        run->N,
        run->M,
        in->row_offsets,
        in->col_indices,
        in->vals,
        in->dense_b,
        out->topm_cols,
        out->topm_vals,
        out->topm_count,
        out->row_sum,
        out->row_max,
        out->row_argmax,
        out->row_nnz,
        out->row_nnz_prefix);

    cudaError_t err = cudaPeekAtLastError();
    return err;
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
