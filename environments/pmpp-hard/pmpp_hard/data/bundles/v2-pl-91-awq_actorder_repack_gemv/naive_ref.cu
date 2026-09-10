// file: naive_ref.cu (awq_actorder_repack_gemv)
//
// Independent, clean multi-pass baseline used ONLY to calibrate the perf
// gate. Direct decomposition of the contract, one kernel per stage, scalar
// loads, materialized intermediates in workspace:
//   pass 1: rank(k) by the literal counting formula (thread per k, O(K^2))
//   pass 2: unpack q  -> qmat  u8[K][N]   (word re-read per element)
//   pass 3: unpack z  -> zmat  u8[G][N]
//   pass 4: qz in permuted order -> qzmat i8[K][N] (j-major)
//   pass 5: repack: one thread per destination word, gathers from qmat
//   pass 6: block-per-column pinned 16-lane dot from qzmat
//   pass 7: thread-per-column zsum loop
//   pass 8: thread-per-(column, substream) FNV sub-digests
//   pass 9: thread-per-column digest combine
// Bit-exact identical outputs to the reference.

#include "awq_actorder_repack_gemv_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct AwqNaiveState {
    AwqProblemSpec spec;
};

__device__ __forceinline__ int nv_wlane(int t) {
    return ((t & 3) << 1) | (t >> 2);
}

__device__ __forceinline__ int nv_zlane(int t) {
    return ((t & 1) << 2) | (t >> 1);
}

__device__ __forceinline__ uint64_t nv_fnv_word(uint64_t h, uint64_t w) {
    h ^= w;
    h *= AWQ_FNV_PRIME;
    return h;
}

// pass 1: literal stable rank.
__global__ void nv_rank_kernel(
    int K,
    const int32_t* __restrict__ g_idx,
    int32_t* __restrict__ perm) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const int g = g_idx[k];
    int rank = 0;
    for (int kp = 0; kp < K; ++kp) {
        const int gp = g_idx[kp];
        if (gp < g || (gp == g && kp < k)) ++rank;
    }
    perm[rank] = k;
}

// pass 2: unpack weights.
__global__ void nv_unpack_q_kernel(
    int K, int N, int Nw,
    const uint32_t* __restrict__ qweight,
    uint8_t* __restrict__ qmat) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)K * N) return;
    const int k = (int)(idx / N);
    const int n = (int)(idx - (long long)k * N);
    const uint32_t word = qweight[(size_t)k * Nw + (size_t)(n >> 3)];
    qmat[idx] = (uint8_t)((word >> (4 * nv_wlane(n & 7))) & 0xFu);
}

// pass 3: unpack zeros.
__global__ void nv_unpack_z_kernel(
    int G, int N, int Nw,
    const uint32_t* __restrict__ qzeros,
    uint8_t* __restrict__ zmat) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)G * N) return;
    const int g = (int)(idx / N);
    const int n = (int)(idx - (long long)g * N);
    const uint32_t word = qzeros[(size_t)g * Nw + (size_t)(n >> 3)];
    zmat[idx] = (uint8_t)((word >> (4 * nv_zlane(n & 7))) & 0xFu);
}

// pass 4: permuted qz matrix.
__global__ void nv_qz_kernel(
    int K, int N,
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ g_idx,
    const uint8_t* __restrict__ qmat,
    const uint8_t* __restrict__ zmat,
    int8_t* __restrict__ qzmat) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)K * N) return;
    const int j = (int)(idx / N);
    const int n = (int)(idx - (long long)j * N);
    const int k = perm[j];
    const int g = g_idx[k];
    qzmat[idx] = (int8_t)((int)qmat[(size_t)k * N + n] -
                          (int)zmat[(size_t)g * N + n]);
}

// pass 5: repack, one thread per destination word.
__global__ void nv_repack_kernel(
    int K, int N, int TA, int TN,
    const int32_t* __restrict__ perm,
    const uint8_t* __restrict__ qmat,
    uint32_t* __restrict__ rq_words) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)128 * TA * TN) return;
    const int atom = (int)(idx >> 7);
    const int off = (int)(idx & 127);
    const int ta = atom / TN;
    const int tn = atom - ta * TN;
    const int h = off >> 6;
    const int s = off & 63;
    const int nt = ((s & 7) << 3) | (s >> 3);  // swz involution
    const int n = 64 * tn + nt;

    uint32_t word = 0;
    for (int i = 0; i < 8; ++i) {
        const int j = 16 * ta + 8 * h + i;
        uint32_t qv = 0;
        if (j < K && n < N) {
            qv = qmat[(size_t)perm[j] * N + n];
        }
        word |= qv << (4 * nv_wlane(i));
    }
    rq_words[idx] = word;
}

// pass 6: pinned 16-lane dot, one block (16 threads) per column.
__global__ void nv_dot_kernel(
    int K, int N,
    const int8_t* __restrict__ qzmat,
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ g_idx,
    const float* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ col_dot) {
    __shared__ float sh[AWQ_DOT_LANES];

    const int n = blockIdx.x;
    if (n >= N) return;
    const int l = threadIdx.x;

    float acc = 0.0f;
    for (int j = l; j < K; j += AWQ_DOT_LANES) {
        const int k = perm[j];
        const int g = g_idx[k];
        const float s = scales[(size_t)g * N + n];
        const float w = __fmul_rn((float)qzmat[(size_t)j * N + n], s);
        acc = __fadd_rn(acc, __fmul_rn(w, x[k]));
    }
    sh[l] = acc;
    __syncthreads();

    for (int stride = AWQ_DOT_LANES / 2; stride > 0; stride >>= 1) {
        if (l < stride) sh[l] = __fadd_rn(sh[l], sh[l + stride]);
        __syncthreads();
    }
    if (l == 0) col_dot[n] = sh[0];
}

// pass 7: zsum.
__global__ void nv_zsum_kernel(
    int K, int N,
    const int8_t* __restrict__ qzmat,
    int32_t* __restrict__ col_zsum) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int32_t acc = 0;
    for (int j = 0; j < K; ++j) acc += qzmat[(size_t)j * N + n];
    col_zsum[n] = acc;
}

// pass 8: sub-digests.
__global__ void nv_subdigest_kernel(
    int K, int N, int G,
    const int8_t* __restrict__ qzmat,
    const float* __restrict__ scales,
    uint64_t* __restrict__ subs) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)N * AWQ_DIGEST_SUBSTREAMS) return;
    const int n = (int)(idx / AWQ_DIGEST_SUBSTREAMS);
    const int r = (int)(idx - (long long)n * AWQ_DIGEST_SUBSTREAMS);

    const int L = K + 4 * G;
    const int W = (L + 7) / 8;

    uint64_t sub = AWQ_FNV_BASIS;
    for (int w = r; w < W; w += AWQ_DIGEST_SUBSTREAMS) {
        uint64_t word = 0;
        for (int b = 0; b < 8; ++b) {
            const int i = 8 * w + b;
            uint8_t byte = 0;
            if (i < K) {
                byte = (uint8_t)qzmat[(size_t)i * N + n];
            } else if (i < L) {
                const int t = i - K;
                const uint32_t sb =
                    __float_as_uint(scales[(size_t)(t >> 2) * N + n]);
                byte = (uint8_t)((sb >> (8 * (t & 3))) & 0xffu);
            }
            word |= (uint64_t)byte << (8 * b);
        }
        sub = nv_fnv_word(sub, word);
    }
    subs[idx] = sub;
}

// pass 9: combine.
__global__ void nv_combine_kernel(
    int N,
    const uint64_t* __restrict__ subs,
    uint64_t* __restrict__ col_digest) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    uint64_t h = AWQ_FNV_BASIS;
    for (int r = 0; r < AWQ_DIGEST_SUBSTREAMS; ++r) {
        h = nv_fnv_word(h, subs[(size_t)n * AWQ_DIGEST_SUBSTREAMS + r]);
    }
    col_digest[n] = h;
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

static size_t nv_align256(size_t x) { return (x + 255u) & ~(size_t)255u; }

static void nv_ws_layout(
    int max_K, int max_N, int max_G,
    size_t* off_qmat, size_t* off_zmat, size_t* off_qzmat, size_t* off_subs,
    size_t* total) {
    size_t off = 0;
    *off_qmat = off;  off += nv_align256((size_t)max_K * (size_t)max_N);
    *off_zmat = off;  off += nv_align256((size_t)max_G * (size_t)max_N);
    *off_qzmat = off; off += nv_align256((size_t)max_K * (size_t)max_N);
    *off_subs = off;  off += nv_align256((size_t)max_N * 8 * 8);
    *total = off;
}

extern "C" size_t solution_workspace_bytes(const AwqProblemSpec* spec) {
    if (!awq_validate_problem_spec(spec)) return 0;
    size_t a, b, c, d, total;
    nv_ws_layout(spec->max_K, spec->max_N, spec->max_G, &a, &b, &c, &d, &total);
    return total;
}

extern "C" cudaError_t solution_init(
    const AwqProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;
    if (!awq_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }
    AwqNaiveState* st =
        static_cast<AwqNaiveState*>(malloc(sizeof(AwqNaiveState)));
    if (!st) return cudaErrorMemoryAllocation;
    memcpy(&st->spec, spec, sizeof(AwqProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const AwqRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !awq_validate_run_spec(run) || !inputs_void || !outputs_void ||
        !workspace) {
        return cudaErrorInvalidValue;
    }

    AwqNaiveState* st = static_cast<AwqNaiveState*>(state);
    const AwqInputs* in = static_cast<const AwqInputs*>(inputs_void);
    AwqOutputs* out = static_cast<AwqOutputs*>(outputs_void);

    if (run->K > st->spec.max_K || run->N > st->spec.max_N ||
        run->G > st->spec.max_G) {
        return cudaErrorInvalidValue;
    }

    size_t off_qmat, off_zmat, off_qzmat, off_subs, need;
    nv_ws_layout(st->spec.max_K, st->spec.max_N, st->spec.max_G,
                 &off_qmat, &off_zmat, &off_qzmat, &off_subs, &need);
    if (workspace_bytes < need) return cudaErrorInvalidValue;

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    uint8_t* qmat = ws + off_qmat;
    uint8_t* zmat = ws + off_zmat;
    int8_t* qzmat = reinterpret_cast<int8_t*>(ws + off_qzmat);
    uint64_t* subs = reinterpret_cast<uint64_t*>(ws + off_subs);

    const int K = run->K;
    const int N = run->N;
    const int G = run->G;
    const int Nw = awq_Nw(N);
    const int TA = awq_TA(K);
    const int TN = awq_TN(N);

    const int T = 256;
    const long long KN = (long long)K * N;
    const long long GN = (long long)G * N;
    const long long W = (long long)128 * TA * TN;

    nv_rank_kernel<<<awq_ceil_div_int(K, T), T, 0, stream>>>(
        K, in->g_idx, out->perm);
    nv_unpack_q_kernel<<<(unsigned)((KN + T - 1) / T), T, 0, stream>>>(
        K, N, Nw, in->qweight, qmat);
    nv_unpack_z_kernel<<<(unsigned)((GN + T - 1) / T), T, 0, stream>>>(
        G, N, Nw, in->qzeros, zmat);
    nv_qz_kernel<<<(unsigned)((KN + T - 1) / T), T, 0, stream>>>(
        K, N, out->perm, in->g_idx, qmat, zmat, qzmat);
    nv_repack_kernel<<<(unsigned)((W + T - 1) / T), T, 0, stream>>>(
        K, N, TA, TN, out->perm, qmat,
        reinterpret_cast<uint32_t*>(out->rq_atoms));
    nv_dot_kernel<<<N, AWQ_DOT_LANES, 0, stream>>>(
        K, N, qzmat, out->perm, in->g_idx, in->scales, in->x, out->col_dot);
    nv_zsum_kernel<<<awq_ceil_div_int(N, T), T, 0, stream>>>(
        K, N, qzmat, out->col_zsum);
    nv_subdigest_kernel<<<awq_ceil_div_int(N * 8, T), T, 0, stream>>>(
        K, N, G, qzmat, in->scales, subs);
    nv_combine_kernel<<<awq_ceil_div_int(N, T), T, 0, stream>>>(
        N, subs, out->col_digest);

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
