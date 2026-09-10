// file: naive_ref.cu (sat_segscan_select)
//
// Independent, clean multi-pass baseline used ONLY to calibrate the perf
// gate. Straightforward decomposition of the contract:
//   pass 0: memset sat_bits (atomicOr target)
//   pass 1: per-word flag popcounts
//   pass 2: single-block exclusive scan of flag popcounts
//   pass 3: per-word emission of ordered segment starts
//   pass 4: grid-stride over SEGMENTS; thread 0 of a block executes the
//           normative sequential loop over each of its segments (the
//           saturating scan is non-associative, so the direct honest
//           implementation runs each segment serially): y, sat bits via
//           atomicOr, seg_last.
//   pass 5: per-word sat popcounts
//   pass 6: single-block exclusive scan -> sel offsets + sel_count
//   pass 7: per-word ordered sel_idx scatter
// Bit-exact identical outputs to the reference.

#include "sat_segscan_select_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct SssNaiveState {
    SssProblemSpec spec;
};

__global__ void nv_wordcnt_kernel(
    int Nw,
    const uint32_t* __restrict__ words,
    unsigned* __restrict__ wcnt) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= Nw) return;
    wcnt[w] = (unsigned)__popc(words[w]);
}

// Single-block exclusive scan over up to ~1M counters (chunked).
__global__ void nv_scan_kernel(
    int Nw,
    const unsigned* __restrict__ wcnt,
    unsigned* __restrict__ woff,
    int32_t* __restrict__ total_out) {
    __shared__ unsigned sh[1024];
    __shared__ unsigned sh_carry;

    const int tid = threadIdx.x;
    if (tid == 0) sh_carry = 0;
    __syncthreads();

    for (int b0 = 0; b0 < Nw; b0 += 1024) {
        const int w = b0 + tid;
        const unsigned val = (w < Nw) ? wcnt[w] : 0u;
        sh[tid] = val;
        __syncthreads();

        for (int off = 1; off < 1024; off <<= 1) {
            unsigned add = 0;
            if (tid >= off) add = sh[tid - off];
            __syncthreads();
            sh[tid] += add;
            __syncthreads();
        }

        if (w < Nw) woff[w] = sh_carry + sh[tid] - val;
        __syncthreads();
        if (tid == 0) sh_carry += sh[1023];
        __syncthreads();
    }

    if (total_out && tid == 0) total_out[0] = (int32_t)sh_carry;
}

__global__ void nv_emit_kernel(
    int Nw,
    const uint32_t* __restrict__ words,
    const unsigned* __restrict__ woff,
    int32_t* __restrict__ out_idx) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= Nw) return;
    uint32_t rest = words[w];
    unsigned o = woff[w];
    while (rest) {
        const int b = __ffs((int)rest) - 1;
        rest &= rest - 1;
        out_idx[o++] = 32 * w + b;
    }
}

// Grid-stride over segments; thread 0 of each block runs the normative
// sequential loop for its segments.
__global__ void nv_segrun_kernel(
    int N,
    const int32_t* __restrict__ S_ptr,
    int LO,
    int HI,
    const int32_t* __restrict__ v,
    const int32_t* __restrict__ seg_starts,
    int32_t* __restrict__ y,
    uint32_t* __restrict__ sat_bits,
    int32_t* __restrict__ seg_last) {
    if (threadIdx.x != 0) return;
    const int S = S_ptr[0];

    for (int s = blockIdx.x; s < S; s += gridDim.x) {
        const int lo_i = seg_starts[s];
        const int hi_i = (s + 1 < S) ? seg_starts[s + 1] : N;

        long long p = 0;
        for (int i = lo_i; i < hi_i; ++i) {
            long long t = p + (long long)v[i];
            if (t < (long long)LO) t = LO;
            if (t > (long long)HI) t = HI;
            p = t;
            y[i] = (int32_t)p;
            if (p == (long long)LO || p == (long long)HI) {
                atomicOr(&sat_bits[i >> 5], 1u << (i & 31));
            }
        }
        seg_last[s] = (int32_t)p;
    }
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

static size_t nv_align256(size_t x) { return (x + 255u) & ~(size_t)255u; }

static void nv_ws_layout(
    int max_N,
    size_t* off_wcnt, size_t* off_woff, size_t* off_starts, size_t* off_S,
    size_t* total) {
    const int Nw = sss_Nw(max_N);
    size_t off = 0;
    *off_wcnt = off;   off += nv_align256((size_t)Nw * 4);
    *off_woff = off;   off += nv_align256((size_t)Nw * 4);
    *off_starts = off; off += nv_align256((size_t)max_N * 4);
    *off_S = off;      off += nv_align256(4);
    *total = off;
}

extern "C" size_t solution_workspace_bytes(const SssProblemSpec* spec) {
    if (!sss_validate_problem_spec(spec)) return 0;
    size_t a, b, c, d, total;
    nv_ws_layout(spec->max_N, &a, &b, &c, &d, &total);
    return total;
}

extern "C" cudaError_t solution_init(
    const SssProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;
    if (!sss_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }
    SssNaiveState* st =
        static_cast<SssNaiveState*>(malloc(sizeof(SssNaiveState)));
    if (!st) return cudaErrorMemoryAllocation;
    memcpy(&st->spec, spec, sizeof(SssProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const SssRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !sss_validate_run_spec(run) || !inputs_void || !outputs_void ||
        !workspace) {
        return cudaErrorInvalidValue;
    }

    SssNaiveState* st = static_cast<SssNaiveState*>(state);
    const SssInputs* in = static_cast<const SssInputs*>(inputs_void);
    SssOutputs* out = static_cast<SssOutputs*>(outputs_void);

    if (run->N > st->spec.max_N) return cudaErrorInvalidValue;

    size_t off_wcnt, off_woff, off_starts, off_S, need;
    nv_ws_layout(st->spec.max_N, &off_wcnt, &off_woff, &off_starts, &off_S,
                 &need);
    if (workspace_bytes < need) return cudaErrorInvalidValue;

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    unsigned* wcnt = reinterpret_cast<unsigned*>(ws + off_wcnt);
    unsigned* woff = reinterpret_cast<unsigned*>(ws + off_woff);
    int32_t* seg_starts = reinterpret_cast<int32_t*>(ws + off_starts);
    int32_t* ws_S = reinterpret_cast<int32_t*>(ws + off_S);

    const int N = run->N;
    const int Nw = sss_Nw(N);
    const int T = 256;

    cudaError_t err = cudaMemsetAsync(out->sat_bits, 0, (size_t)Nw * 4, stream);
    if (err != cudaSuccess) return err;

    nv_wordcnt_kernel<<<sss_ceil_div_int(Nw, T), T, 0, stream>>>(
        Nw, in->flags, wcnt);
    nv_scan_kernel<<<1, 1024, 0, stream>>>(Nw, wcnt, woff, ws_S);
    nv_emit_kernel<<<sss_ceil_div_int(Nw, T), T, 0, stream>>>(
        Nw, in->flags, woff, seg_starts);

    nv_segrun_kernel<<<8192, 32, 0, stream>>>(
        N, ws_S, run->lo, run->hi,
        in->v, seg_starts, out->y, out->sat_bits, out->seg_last);

    nv_wordcnt_kernel<<<sss_ceil_div_int(Nw, T), T, 0, stream>>>(
        Nw, out->sat_bits, wcnt);
    nv_scan_kernel<<<1, 1024, 0, stream>>>(Nw, wcnt, woff, out->sel_count);
    nv_emit_kernel<<<sss_ceil_div_int(Nw, T), T, 0, stream>>>(
        Nw, out->sat_bits, woff, out->sel_idx);

    return cudaPeekAtLastError();
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
