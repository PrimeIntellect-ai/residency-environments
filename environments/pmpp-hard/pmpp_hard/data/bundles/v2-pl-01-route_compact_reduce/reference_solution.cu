// PMPP_CANARY_01_f186c8bed2 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: route_compact_reduce_reference.cu
// Strong reference pipeline implementation.
// ============================================================================

#include "route_compact_reduce_common.h"

#include <stdlib.h>
#include <string.h>
#include <limits.h>

struct RcrReferenceState {
    RcrProblemSpec spec;
};

struct RcrWorkspaceLayout {
    int16_t* norm_exp;       // max_N * max_K
    int32_t* norm_weight;    // max_N * max_K
    int32_t* chunk_counts;   // max_chunks * max_E
    int32_t* chunk_bases;    // max_chunks * max_E
    size_t required_bytes;
};

static size_t rcr_reference_workspace_bytes_for(int max_N, int max_E, int max_K) {
    const int max_chunks = rcr_ceil_div_int(max_N, RCR_CHUNK_TOKENS);

    size_t off = 0;
    off = rcr_align_up_size(off, 128);
    off += sizeof(int16_t) * (size_t)max_N * (size_t)max_K;

    off = rcr_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_N * (size_t)max_K;

    off = rcr_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_chunks * (size_t)max_E;

    off = rcr_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_chunks * (size_t)max_E;

    off = rcr_align_up_size(off, 128);
    return off;
}

static RcrWorkspaceLayout rcr_reference_make_layout(
    void* workspace,
    int max_N,
    int max_E,
    int max_K) {
    RcrWorkspaceLayout layout{};
    char* base = static_cast<char*>(workspace);
    size_t off = 0;
    const int max_chunks = rcr_ceil_div_int(max_N, RCR_CHUNK_TOKENS);

    off = rcr_align_up_size(off, 128);
    layout.norm_exp = reinterpret_cast<int16_t*>(base + off);
    off += sizeof(int16_t) * (size_t)max_N * (size_t)max_K;

    off = rcr_align_up_size(off, 128);
    layout.norm_weight = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_N * (size_t)max_K;

    off = rcr_align_up_size(off, 128);
    layout.chunk_counts = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_chunks * (size_t)max_E;

    off = rcr_align_up_size(off, 128);
    layout.chunk_bases = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_chunks * (size_t)max_E;

    off = rcr_align_up_size(off, 128);
    layout.required_bytes = off;
    return layout;
}

__device__ __forceinline__ unsigned long long rcr_abs_i64_device(long long v) {
    unsigned long long u = static_cast<unsigned long long>(v);
    return v < 0 ? (~u + 1ULL) : u;
}

__global__ void rcr_ref_count_normalize_kernel(
    int N,
    int E,
    int K,
    const int16_t* __restrict__ expert,
    const int16_t* __restrict__ weight,
    const uint8_t* __restrict__ valid,
    int16_t* __restrict__ norm_exp,
    int32_t* __restrict__ norm_weight,
    int32_t* __restrict__ chunk_counts) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    const int base = t * K;

    #pragma unroll
    for (int i = 0; i < RCR_MAX_K; ++i) {
        if (i < K) {
            norm_exp[base + i] = -1;
            norm_weight[base + i] = 0;
        }
    }

    if (valid[t] == 0) return;

    int loc_exp[RCR_MAX_K];
    int loc_w[RCR_MAX_K];
    int loc_n = 0;

    #pragma unroll
    for (int i = 0; i < RCR_MAX_K; ++i) {
        loc_exp[i] = -1;
        loc_w[i] = 0;
    }

    for (int k = 0; k < K; ++k) {
        const int e = static_cast<int>(expert[base + k]);
        const int w = static_cast<int>(weight[base + k]);

        if (e < 0 || e >= E || w == 0) continue;

        int found = -1;
        #pragma unroll
        for (int j = 0; j < RCR_MAX_K; ++j) {
            if (j < loc_n && loc_exp[j] == e) {
                found = j;
            }
        }

        if (found >= 0) {
            loc_w[found] += w;
        } else if (loc_n < RCR_MAX_K) {
            loc_exp[loc_n] = e;
            loc_w[loc_n] = w;
            ++loc_n;
        }
    }

    const int chunk = t / RCR_CHUNK_TOKENS;
    int out = 0;
    for (int j = 0; j < loc_n; ++j) {
        if (loc_w[j] == 0) continue;
        norm_exp[base + out] = static_cast<int16_t>(loc_exp[j]);
        norm_weight[base + out] = loc_w[j];
        atomicAdd(&chunk_counts[chunk * E + loc_exp[j]], 1);
        ++out;
    }
}

__global__ void rcr_ref_counts_offsets_kernel(
    const int32_t* __restrict__ chunk_counts,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int num_chunks,
    int E) {
    extern __shared__ int32_t s_counts[];

    const int tid = threadIdx.x;
    if (tid < E) {
        int total = 0;
        for (int ch = 0; ch < num_chunks; ++ch) {
            total += chunk_counts[ch * E + tid];
        }
        s_counts[tid] = total;
        counts[tid] = total;
    }

    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        offsets[0] = 0;
        for (int e = 0; e < E; ++e) {
            acc += s_counts[e];
            offsets[e + 1] = acc;
        }
    }
}

__global__ void rcr_ref_chunk_bases_kernel(
    const int32_t* __restrict__ chunk_counts,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ chunk_bases,
    int num_chunks,
    int E) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int running = offsets[e];
    for (int ch = 0; ch < num_chunks; ++ch) {
        chunk_bases[ch * E + e] = running;
        running += chunk_counts[ch * E + e];
    }
}

__global__ void rcr_ref_stable_scatter_kernel(
    int N,
    int E,
    int K,
    const int16_t* __restrict__ norm_exp,
    const int32_t* __restrict__ norm_weight,
    const int32_t* __restrict__ chunk_bases,
    int32_t* __restrict__ packed_token,
    int32_t* __restrict__ packed_weight) {
    const int ch = blockIdx.x;
    const int e = threadIdx.x;
    if (e >= E) return;

    const int start_t = ch * RCR_CHUNK_TOKENS;
    int end_t = start_t + RCR_CHUNK_TOKENS;
    if (end_t > N) end_t = N;

    int out = chunk_bases[ch * E + e];

    for (int t = start_t; t < end_t; ++t) {
        const int base = t * K;
        for (int r = 0; r < K; ++r) {
            if (static_cast<int>(norm_exp[base + r]) == e) {
                packed_token[out] = t;
                packed_weight[out] = norm_weight[base + r];
                ++out;
            }
        }
    }
}

__global__ void rcr_ref_segmented_reduce_kernel(
    int C,
    const int16_t* __restrict__ x,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ packed_token,
    const int32_t* __restrict__ packed_weight,
    int64_t* __restrict__ sum,
    int32_t* __restrict__ argmax_abs) {
    constexpr int BLOCK = 256;

    __shared__ unsigned long long s_sum[BLOCK];
    __shared__ unsigned long long s_abs[BLOCK];
    __shared__ int s_tok[BLOCK];
    __shared__ int s_any[BLOCK];

    const int pair = blockIdx.x;
    const int e = pair / C;
    const int c = pair - e * C;

    const int begin = offsets[e];
    const int end = offsets[e + 1];

    unsigned long long local_sum = 0ULL;
    unsigned long long local_abs = 0ULL;
    int local_tok = INT_MAX;
    int local_any = 0;

    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const int tok = packed_token[i];
        const int w = packed_weight[i];
        const int xv = static_cast<int>(x[tok * C + c]);
        const long long prod = static_cast<long long>(w) * static_cast<long long>(xv);
        const unsigned long long abs_v = rcr_abs_i64_device(prod);

        local_sum += static_cast<unsigned long long>(prod);

        if (!local_any ||
            abs_v > local_abs ||
            (abs_v == local_abs && tok < local_tok)) {
            local_any = 1;
            local_abs = abs_v;
            local_tok = tok;
        }
    }

    s_sum[threadIdx.x] = local_sum;
    s_abs[threadIdx.x] = local_abs;
    s_tok[threadIdx.x] = local_tok;
    s_any[threadIdx.x] = local_any;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];

            const int other_any = s_any[threadIdx.x + stride];
            const unsigned long long other_abs = s_abs[threadIdx.x + stride];
            const int other_tok = s_tok[threadIdx.x + stride];

            if (other_any &&
                (!s_any[threadIdx.x] ||
                 other_abs > s_abs[threadIdx.x] ||
                 (other_abs == s_abs[threadIdx.x] && other_tok < s_tok[threadIdx.x]))) {
                s_any[threadIdx.x] = 1;
                s_abs[threadIdx.x] = other_abs;
                s_tok[threadIdx.x] = other_tok;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum[pair] = static_cast<int64_t>(s_sum[0]);
        argmax_abs[pair] = s_any[0] ? s_tok[0] : -1;
    }
}

extern "C" size_t solution_workspace_bytes(const RcrProblemSpec* spec) {
    if (!rcr_validate_problem_spec(spec)) return 0;
    return rcr_reference_workspace_bytes_for(spec->max_N, spec->max_E, spec->max_K);
}

extern "C" cudaError_t solution_init(
    const RcrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!rcr_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    RcrReferenceState* st = static_cast<RcrReferenceState*>(malloc(sizeof(RcrReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memcpy(&st->spec, spec, sizeof(RcrProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const RcrRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !rcr_validate_run_spec(run) || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    RcrReferenceState* st = static_cast<RcrReferenceState*>(state);
    const RcrInputs* in = static_cast<const RcrInputs*>(inputs_void);
    RcrOutputs* out = static_cast<RcrOutputs*>(outputs_void);

    if (run->N > st->spec.max_N ||
        run->C > st->spec.max_C ||
        run->E > st->spec.max_E ||
        run->K > st->spec.max_K) {
        return cudaErrorInvalidValue;
    }

    if (!in->x || !in->expert || !in->weight || !in->valid ||
        !out->counts || !out->offsets || !out->packed_token ||
        !out->packed_weight || !out->sum || !out->argmax_abs) {
        return cudaErrorInvalidValue;
    }

    RcrWorkspaceLayout layout = rcr_reference_make_layout(
        workspace,
        st->spec.max_N,
        st->spec.max_E,
        st->spec.max_K);

    if (workspace_bytes < layout.required_bytes) {
        return cudaErrorInvalidValue;
    }

    const int N = run->N;
    const int C = run->C;
    const int E = run->E;
    const int K = run->K;
    const int num_chunks = rcr_ceil_div_int(N, RCR_CHUNK_TOKENS);

    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(
        layout.chunk_counts,
        0,
        sizeof(int32_t) * (size_t)num_chunks * (size_t)E,
        stream);
    if (err != cudaSuccess) return err;

    const int block = 256;
    const int grid = rcr_ceil_div_int(N, block);

    rcr_ref_count_normalize_kernel<<<grid, block, 0, stream>>>(
        N,
        E,
        K,
        in->expert,
        in->weight,
        in->valid,
        layout.norm_exp,
        layout.norm_weight,
        layout.chunk_counts);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    rcr_ref_counts_offsets_kernel<<<1, 128, sizeof(int32_t) * (size_t)E, stream>>>(
        layout.chunk_counts,
        out->counts,
        out->offsets,
        num_chunks,
        E);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    rcr_ref_chunk_bases_kernel<<<1, 128, 0, stream>>>(
        layout.chunk_counts,
        out->offsets,
        layout.chunk_bases,
        num_chunks,
        E);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    rcr_ref_stable_scatter_kernel<<<num_chunks, 128, 0, stream>>>(
        N,
        E,
        K,
        layout.norm_exp,
        layout.norm_weight,
        layout.chunk_bases,
        out->packed_token,
        out->packed_weight);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    const int reduce_blocks = E * C;
    rcr_ref_segmented_reduce_kernel<<<reduce_blocks, 256, 0, stream>>>(
        C,
        in->x,
        out->offsets,
        out->packed_token,
        out->packed_weight,
        out->sum,
        out->argmax_abs);
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
