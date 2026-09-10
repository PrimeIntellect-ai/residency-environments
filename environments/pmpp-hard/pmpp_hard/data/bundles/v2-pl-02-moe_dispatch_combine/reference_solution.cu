// PMPP_CANARY_02_091f610f42 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: moe_dispatch_combine_reference.cu
// Strong reference pipeline implementation.
// ============================================================================

#include "moe_dispatch_combine_common.h"

#include <stdlib.h>
#include <string.h>

struct MdcReferenceState {
    MdcProblemSpec spec;
};

struct MdcWorkspaceLayout {
    int32_t* chunk_counts;      // max_chunks * max_E * MDC_NUM_GATES
    int32_t* chunk_bases;       // max_chunks * max_E * MDC_NUM_GATES
    int32_t* local_gate_base;   // max_E * MDC_NUM_GATES
    size_t required_bytes;
};

static size_t mdc_reference_workspace_bytes_for(int max_N, int max_E) {
    const int max_chunks = mdc_ceil_div_int(max_N, MDC_CHUNK_TOKENS);

    size_t off = 0;

    off = mdc_align_up_size(off, 128);
    off += sizeof(int32_t) *
           (size_t)max_chunks * (size_t)max_E * (size_t)MDC_NUM_GATES;

    off = mdc_align_up_size(off, 128);
    off += sizeof(int32_t) *
           (size_t)max_chunks * (size_t)max_E * (size_t)MDC_NUM_GATES;

    off = mdc_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_E * (size_t)MDC_NUM_GATES;

    off = mdc_align_up_size(off, 128);
    return off;
}

static MdcWorkspaceLayout mdc_reference_make_layout(
    void* workspace,
    int max_N,
    int max_E) {
    MdcWorkspaceLayout layout{};
    char* base = static_cast<char*>(workspace);
    const int max_chunks = mdc_ceil_div_int(max_N, MDC_CHUNK_TOKENS);

    size_t off = 0;

    off = mdc_align_up_size(off, 128);
    layout.chunk_counts = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) *
           (size_t)max_chunks * (size_t)max_E * (size_t)MDC_NUM_GATES;

    off = mdc_align_up_size(off, 128);
    layout.chunk_bases = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) *
           (size_t)max_chunks * (size_t)max_E * (size_t)MDC_NUM_GATES;

    off = mdc_align_up_size(off, 128);
    layout.local_gate_base = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_E * (size_t)MDC_NUM_GATES;

    off = mdc_align_up_size(off, 128);
    layout.required_bytes = off;
    return layout;
}

__device__ __forceinline__ int mdc_gate_to_index_device(int gate) {
    return gate - MDC_GATE_MIN;
}

__device__ __forceinline__ int mdc_gate_is_legal_device(int gate) {
    return gate >= MDC_GATE_MIN && gate <= MDC_GATE_MAX;
}

__global__ void mdc_ref_count_chunks_kernel(
    int N,
    int E,
    int K,
    const int16_t* __restrict__ expert,
    const int16_t* __restrict__ gate,
    const uint8_t* __restrict__ valid,
    int32_t* __restrict__ chunk_counts) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    if (valid[t] == 0) return;

    const int chunk = t / MDC_CHUNK_TOKENS;
    const int base = t * K;

    for (int k = 0; k < K; ++k) {
        const int e = static_cast<int>(expert[base + k]);
        const int g = static_cast<int>(gate[base + k]);

        if (e < 0 || e >= E || !mdc_gate_is_legal_device(g)) {
            continue;
        }

        const int gi = mdc_gate_to_index_device(g);
        const size_t idx =
            ((size_t)chunk * (size_t)E + (size_t)e) *
            (size_t)MDC_NUM_GATES + (size_t)gi;

        atomicAdd(&chunk_counts[idx], 1);
    }
}

__global__ void mdc_ref_gate_bases_kernel(
    const int32_t* __restrict__ chunk_counts,
    int32_t* __restrict__ local_gate_base,
    int32_t* __restrict__ counts,
    int num_chunks,
    int E,
    int cap) {
    const int e = blockIdx.x;
    if (e >= E || threadIdx.x != 0) return;

    int running = 0;

    for (int gi = MDC_NUM_GATES - 1; gi >= 0; --gi) {
        local_gate_base[e * MDC_NUM_GATES + gi] = running;

        int gate_total = 0;
        for (int ch = 0; ch < num_chunks; ++ch) {
            const size_t idx =
                ((size_t)ch * (size_t)E + (size_t)e) *
                (size_t)MDC_NUM_GATES + (size_t)gi;
            gate_total += chunk_counts[idx];
        }

        running += gate_total;
    }

    counts[e] = running < cap ? running : cap;
}

__global__ void mdc_ref_offsets_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int E) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int acc = 0;
        offsets[0] = 0;
        for (int e = 0; e < E; ++e) {
            acc += counts[e];
            offsets[e + 1] = acc;
        }
    }
}

__global__ void mdc_ref_chunk_bases_kernel(
    const int32_t* __restrict__ chunk_counts,
    const int32_t* __restrict__ local_gate_base,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ chunk_bases,
    int num_chunks,
    int E) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = E * MDC_NUM_GATES;
    if (linear >= total) return;

    const int e = linear / MDC_NUM_GATES;
    const int gi = linear - e * MDC_NUM_GATES;

    int running = offsets[e] + local_gate_base[e * MDC_NUM_GATES + gi];

    for (int ch = 0; ch < num_chunks; ++ch) {
        const size_t idx =
            ((size_t)ch * (size_t)E + (size_t)e) *
            (size_t)MDC_NUM_GATES + (size_t)gi;

        chunk_bases[idx] = running;
        running += chunk_counts[idx];
    }
}

__global__ void mdc_ref_stable_scatter_kernel(
    int N,
    int E,
    int K,
    const int16_t* __restrict__ expert,
    const int16_t* __restrict__ gate,
    const uint8_t* __restrict__ valid,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ chunk_bases,
    int32_t* __restrict__ packed_token,
    int32_t* __restrict__ packed_slot,
    int16_t* __restrict__ packed_gate,
    uint8_t* __restrict__ dropped) {
    const int block = blockIdx.x;
    const int ch = block / E;
    const int e = block - ch * E;

    if (threadIdx.x != 0) return;

    int gate_seen[MDC_NUM_GATES];

    #pragma unroll
    for (int gi = 0; gi < MDC_NUM_GATES; ++gi) {
        gate_seen[gi] = 0;
    }

    const int start_t = ch * MDC_CHUNK_TOKENS;
    int end_t = start_t + MDC_CHUNK_TOKENS;
    if (end_t > N) end_t = N;

    const int expert_end = offsets[e + 1];

    for (int t = start_t; t < end_t; ++t) {
        if (valid[t] == 0) continue;

        const int route_base = t * K;
        for (int k = 0; k < K; ++k) {
            const int route_e = static_cast<int>(expert[route_base + k]);
            if (route_e != e) continue;

            const int g = static_cast<int>(gate[route_base + k]);
            if (!mdc_gate_is_legal_device(g)) continue;

            const int gi = mdc_gate_to_index_device(g);
            const size_t base_idx =
                ((size_t)ch * (size_t)E + (size_t)e) *
                (size_t)MDC_NUM_GATES + (size_t)gi;

            const int rank = chunk_bases[base_idx] + gate_seen[gi];
            ++gate_seen[gi];

            if (rank < expert_end) {
                packed_token[rank] = t;
                packed_slot[rank] = k;
                packed_gate[rank] = static_cast<int16_t>(g);
            } else {
                dropped[route_base + k] = 1;
            }
        }
    }
}

__global__ void mdc_ref_combine_kernel(
    int D,
    const int16_t* __restrict__ expert_out,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ packed_token,
    const int16_t* __restrict__ packed_gate,
    int64_t* __restrict__ y) {
    const int e = blockIdx.x;
    const int d = blockIdx.y;

    const int begin = offsets[e];
    const int end = offsets[e + 1];
    const int out_v = static_cast<int>(expert_out[e * D + d]);

    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const int t = packed_token[i];
        const int g = static_cast<int>(packed_gate[i]);

        const long long prod =
            static_cast<long long>(g) * static_cast<long long>(out_v);

        atomicAdd(
            reinterpret_cast<unsigned long long*>(&y[(size_t)t * (size_t)D + (size_t)d]),
            static_cast<unsigned long long>(prod));
    }
}

extern "C" size_t solution_workspace_bytes(const MdcProblemSpec* spec) {
    if (!mdc_validate_problem_spec(spec)) return 0;
    return mdc_reference_workspace_bytes_for(spec->max_N, spec->max_E);
}

extern "C" cudaError_t solution_init(
    const MdcProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!mdc_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    MdcReferenceState* st = static_cast<MdcReferenceState*>(malloc(sizeof(MdcReferenceState)));
    if (!st) {
        return cudaErrorMemoryAllocation;
    }

    memcpy(&st->spec, spec, sizeof(MdcProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const MdcRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !mdc_validate_run_spec(run) || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    MdcReferenceState* st = static_cast<MdcReferenceState*>(state);
    const MdcInputs* in = static_cast<const MdcInputs*>(inputs_void);
    MdcOutputs* out = static_cast<MdcOutputs*>(outputs_void);

    if (run->N > st->spec.max_N ||
        run->D > st->spec.max_D ||
        run->E > st->spec.max_E ||
        run->K > st->spec.max_K ||
        run->cap > st->spec.max_cap) {
        return cudaErrorInvalidValue;
    }

    if (!in->expert || !in->gate || !in->valid || !in->expert_out ||
        !out->counts || !out->offsets || !out->packed_token ||
        !out->packed_slot || !out->packed_gate || !out->dropped || !out->y) {
        return cudaErrorInvalidValue;
    }

    MdcWorkspaceLayout layout = mdc_reference_make_layout(
        workspace,
        st->spec.max_N,
        st->spec.max_E);

    if (workspace_bytes < layout.required_bytes) {
        return cudaErrorInvalidValue;
    }

    const int N = run->N;
    const int D = run->D;
    const int E = run->E;
    const int K = run->K;
    const int cap = run->cap;
    const int num_chunks = mdc_ceil_div_int(N, MDC_CHUNK_TOKENS);

    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(
        layout.chunk_counts,
        0,
        sizeof(int32_t) *
            (size_t)num_chunks * (size_t)E * (size_t)MDC_NUM_GATES,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(
        out->dropped,
        0,
        sizeof(uint8_t) * (size_t)N * (size_t)K,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(
        out->y,
        0,
        sizeof(int64_t) * (size_t)N * (size_t)D,
        stream);
    if (err != cudaSuccess) return err;

    const int token_block = 256;
    const int token_grid = mdc_ceil_div_int(N, token_block);

    mdc_ref_count_chunks_kernel<<<token_grid, token_block, 0, stream>>>(
        N,
        E,
        K,
        in->expert,
        in->gate,
        in->valid,
        layout.chunk_counts);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mdc_ref_gate_bases_kernel<<<E, 1, 0, stream>>>(
        layout.chunk_counts,
        layout.local_gate_base,
        out->counts,
        num_chunks,
        E,
        cap);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mdc_ref_offsets_kernel<<<1, 1, 0, stream>>>(
        out->counts,
        out->offsets,
        E);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    const int eg_total = E * MDC_NUM_GATES;
    const int eg_block = 256;
    const int eg_grid = mdc_ceil_div_int(eg_total, eg_block);

    mdc_ref_chunk_bases_kernel<<<eg_grid, eg_block, 0, stream>>>(
        layout.chunk_counts,
        layout.local_gate_base,
        out->offsets,
        layout.chunk_bases,
        num_chunks,
        E);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mdc_ref_stable_scatter_kernel<<<num_chunks * E, 1, 0, stream>>>(
        N,
        E,
        K,
        in->expert,
        in->gate,
        in->valid,
        out->offsets,
        layout.chunk_bases,
        out->packed_token,
        out->packed_slot,
        out->packed_gate,
        out->dropped);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    dim3 combine_grid(E, D, 1);
    mdc_ref_combine_kernel<<<combine_grid, 256, 0, stream>>>(
        D,
        in->expert_out,
        out->offsets,
        out->packed_token,
        out->packed_gate,
        out->y);
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
