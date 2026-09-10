// PMPP_CANARY_91_7b3e9d40aa -- held-out canary; MUST NOT appear in any submission
// file: awq_actorder_repack_gemv_reference.cu
//
// Two-phase pipeline:
//   K1a/b/c: deterministic STABLE counting rank (per-chunk shared histograms,
//            cross-chunk exclusive offsets, in-chunk stable local ranks) --
//            no library sort, exact contract formula; writes perm (output),
//            gsorted[j] = g_idx[perm[j]] and xp[j] = x[perm[j]] (workspace).
//   K2:      ONE fused kernel does everything else. A block owns a 32-column
//            strip (half a payload atom) and walks the permuted rows in
//            64-row chunks: it gathers the strip's 4 source words per row
//            (16B contiguous per row -> well-used sectors), unpacks
//            wlane/zlane exactly once, stages q and qz tiles in shared, and
//            feeds four consumers in the same pass: the repacked atom words
//            (assembled from shared, guarded per-tile), the 16-lane pinned
//            dequant-dot (two chains per thread for ILP; __fmul_rn/__fadd_rn
//            only -- no FMA, no FTZ), the int32 column sums, and the eight
//            per-column FNV substream chains (which consume the staged tiles
//            strictly in stream order, then the scale-byte tail from L2).
//            The unpacked q matrix is never materialized in global memory,
//            qweight is read exactly once, and the only workspace is the
//            tiny rank scratch (no K*N staging tensor). Shared-memory trees
//            reproduce the contract reduction orders exactly.

#include "awq_actorder_repack_gemv_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define AWQ_RANK_CHUNK 256

struct AwqReferenceState {
    AwqProblemSpec spec;
};

__device__ __forceinline__ int awq_d_wlane(int t) {
    return ((t & 3) << 1) | (t >> 2);
}

__device__ __forceinline__ int awq_d_zlane(int t) {
    return ((t & 1) << 2) | (t >> 1);
}

__device__ __forceinline__ uint64_t awq_d_fnv_word(uint64_t h, uint64_t w) {
    h ^= w;
    h *= AWQ_FNV_PRIME;
    return h;
}

// ---------------------------------------------------------------------------
// K1a: per-chunk group histograms. ws_hist[b*G + g].
// ---------------------------------------------------------------------------
__global__ void awq_ref_hist_kernel(
    int K,
    int G,
    const int32_t* __restrict__ g_idx,
    int32_t* __restrict__ ws_hist) {
    extern __shared__ int32_t sh_hist[];  // G

    const int b = blockIdx.x;
    const int k0 = b * AWQ_RANK_CHUNK;

    for (int g = threadIdx.x; g < G; g += blockDim.x) sh_hist[g] = 0;
    __syncthreads();

    const int k = k0 + threadIdx.x;
    if (k < K) atomicAdd(&sh_hist[g_idx[k]], 1);
    __syncthreads();

    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        ws_hist[(size_t)b * (size_t)G + (size_t)g] = sh_hist[g];
    }
}

// ---------------------------------------------------------------------------
// K1b: exclusive offsets. ws_base[b*G + g] = start of (chunk b, group g)
//      = sum_{g'<g} total(g') + sum_{b'<b} hist(b', g).
// ---------------------------------------------------------------------------
__global__ void awq_ref_offsets_kernel(
    int B,
    int G,
    const int32_t* __restrict__ ws_hist,
    int32_t* __restrict__ ws_base,
    int32_t* __restrict__ ws_gstart) {
    extern __shared__ int32_t sh_total[];  // G

    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        int32_t t = 0;
        for (int b = 0; b < B; ++b) t += ws_hist[(size_t)b * (size_t)G + (size_t)g];
        sh_total[g] = t;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t run = 0;
        for (int g = 0; g < G; ++g) {
            ws_gstart[g] = run;
            run += sh_total[g];
        }
    }
    __syncthreads();

    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        int32_t run = ws_gstart[g];
        for (int b = 0; b < B; ++b) {
            ws_base[(size_t)b * (size_t)G + (size_t)g] = run;
            run += ws_hist[(size_t)b * (size_t)G + (size_t)g];
        }
    }
}

// ---------------------------------------------------------------------------
// K1c: stable scatter. rank(k) = base(chunk, g) + #earlier-in-chunk equals.
// ---------------------------------------------------------------------------
__global__ void awq_ref_scatter_kernel(
    int K,
    int G,
    const int32_t* __restrict__ g_idx,
    const float* __restrict__ x,
    const int32_t* __restrict__ ws_base,
    int32_t* __restrict__ perm,
    int32_t* __restrict__ gsorted,
    float* __restrict__ xp) {
    __shared__ int32_t sh_g[AWQ_RANK_CHUNK];

    const int b = blockIdx.x;
    const int k = b * AWQ_RANK_CHUNK + threadIdx.x;
    const int g = (k < K) ? g_idx[k] : 0;

    sh_g[threadIdx.x] = g;
    __syncthreads();

    if (k >= K) return;

    int local = 0;
    for (int t = 0; t < (int)threadIdx.x; ++t) {
        local += (sh_g[t] == g) ? 1 : 0;
    }

    const int j = ws_base[(size_t)b * (size_t)G + (size_t)g] + local;
    perm[j] = k;
    gsorted[j] = g;
    xp[j] = x[k];
}

// ---------------------------------------------------------------------------
// K1d: scale transpose. sc_t[n*G + g] = scales[g*N + n]. The dot chains and
//      digest tails walk g monotonically at fixed n, so the transposed rows
//      turn 16-way scattered lane gathers into single-line hits.
// ---------------------------------------------------------------------------
__global__ void awq_ref_sctrans_kernel(
    int N,
    int G,
    const float* __restrict__ scales,
    float* __restrict__ sc_t) {
    __shared__ float tile[32][33];

    const int g0 = blockIdx.x * 32;
    const int n0 = blockIdx.y * 32;

    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const int g = g0 + r;
        const int n = n0 + threadIdx.x;
        tile[r][threadIdx.x] = (g < G && n < N)
            ? scales[(size_t)g * (size_t)N + (size_t)n] : 0.0f;
    }
    __syncthreads();

    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const int n = n0 + r;
        const int g = g0 + threadIdx.x;
        if (n < N && g < G) {
            sc_t[(size_t)n * (size_t)G + (size_t)g] = tile[threadIdx.x][r];
        }
    }
}

// ---------------------------------------------------------------------------
// K2: fused unpack + repack + pinned dot + zsum + digests.
//     grid.x = 2*TN strips of 32 columns; block = 256 threads.
//     Chunks of 64 permuted rows (= 4 repack tiles) staged in shared.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
awq_ref_fused_kernel(
    int K,
    int N,
    int G,
    int Nw,
    int TA,
    int TN,
    const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros,
    const float* __restrict__ sc_t,
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ gsorted,
    const float* __restrict__ xp,
    uint32_t* __restrict__ rq_words,
    float* __restrict__ col_dot,
    uint64_t* __restrict__ col_digest,
    int32_t* __restrict__ col_zsum) {
    __shared__ uint8_t sh_q[2][64][32];    // double-buffered chunk tiles
    __shared__ int8_t sh_qz[2][64][32];
    __shared__ int32_t sh_k[3][64];        // triple-buffered perm tiles
    __shared__ int32_t sh_g[3][64];
    __shared__ float sh_xp[3][64];
    __shared__ float sh_dot[16][32];
    __shared__ int32_t sh_z[16][32];
    __shared__ uint64_t sh_sub[32][8];

    const int tid = threadIdx.x;
    const int strip = blockIdx.x;          // 32-column strip
    const int n0 = 32 * strip;
    const int tnA = strip >> 1;            // atom column tile
    const int ntBase = 32 * (strip & 1);   // n-offset inside the atom

    // unpack mapping: 64 rows x 4 words, one word per thread
    const int u_jj = tid >> 2;
    const int u_cw = tid & 3;
    const int u_c = 4 * strip + u_cw;

    // dot mapping: two chains per thread, (nl, l) and (nl + 16, l)
    const int d_l = tid & 15;
    const int d_nl = tid >> 4;             // [0, 16)

    // digest mapping: one chain per thread, (g_nl, g_r)
    const int g_r = tid & 7;
    const int g_nl = tid >> 3;             // [0, 32)

    const int L = K + 4 * G;
    const int W = (L + 7) / 8;
    const bool dig_live = (n0 + g_nl) < N;

    float acc0 = 0.0f, acc1 = 0.0f;
    int32_t zac0 = 0, zac1 = 0;
    uint64_t sub = AWQ_FNV_BASIS;
    uint64_t wacc = 0;  // straddle word carry (qz bytes at the K boundary)

    // ---- software pipeline: the gathers for chunk c+1 and the perm tile
    //      for chunk c+2 are issued BEFORE the compute of chunk c, so their
    //      latency is hidden behind the repack/dot/digest work. sh_q/sh_qz
    //      are double-buffered; the perm tiles are triple-buffered. ----
    const int NC = (K + 63) / 64;

    if (tid < 64) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int j = 64 * c + tid;
            const bool v = (j < K);
            sh_k[c][tid] = v ? perm[j] : -1;
            sh_g[c][tid] = v ? gsorted[j] : 0;
            sh_xp[c][tid] = v ? xp[j] : 0.0f;
        }
    }
    __syncthreads();

    uint32_t gq, gz;
    {
        const int k = sh_k[0][u_jj];
        const bool live = (k >= 0) && (u_c < Nw);
        gq = live ? qweight[(size_t)k * (size_t)Nw + (size_t)u_c] : 0u;
        gz = live ? qzeros[(size_t)sh_g[0][u_jj] * (size_t)Nw + (size_t)u_c] : 0u;
    }

    for (int c = 0; c < NC; ++c) {
        const int j0 = 64 * c;
        const int cur = c & 1;
        const int slot = c % 3;

        // unpack this chunk's prefetched words into the current buffers
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int nl = 8 * u_cw + t;
            uint32_t qv = (gq >> (4 * awq_d_wlane(t))) & 0xFu;
            uint32_t zv = (gz >> (4 * awq_d_zlane(t))) & 0xFu;
            if (n0 + nl >= N) { qv = 0u; zv = 0u; }
            sh_q[cur][u_jj][nl] = (uint8_t)qv;
            sh_qz[cur][u_jj][nl] = (int8_t)((int)qv - (int)zv);
        }

        // prefetch the perm tile for chunk c+2 into registers
        int rk = -1, rg = 0;
        float rx = 0.0f;
        const bool pfk = (tid < 64) && (c + 2 < NC);
        if (pfk) {
            const int j = 64 * (c + 2) + tid;
            if (j < K) {
                rk = perm[j];
                rg = gsorted[j];
                rx = xp[j];
            }
        }

        __syncthreads();  // chunk c published

        // issue the gathers for chunk c+1 (completes during compute below)
        if (c + 1 < NC) {
            const int ns = (c + 1) % 3;
            const int k = sh_k[ns][u_jj];
            const bool live = (k >= 0) && (u_c < Nw);
            gq = live ? qweight[(size_t)k * (size_t)Nw + (size_t)u_c] : 0u;
            gz = live
                ? qzeros[(size_t)sh_g[ns][u_jj] * (size_t)Nw + (size_t)u_c] : 0u;
        }

        // ---- repack: 4 tiles x (2h x 32nt) words, one per thread ----
        {
            const int a = tid >> 6;
            const int w = tid & 63;
            const int h = w >> 5;
            const int ntl = w & 31;
            const int ta = (j0 >> 4) + a;
            if (ta < TA) {
                uint32_t word = 0;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    word |= (uint32_t)sh_q[cur][16 * a + 8 * h + i][ntl]
                            << (4 * awq_d_wlane(i));
                }
                const int nt = ntBase + ntl;
                const int off = h * 64 + (((nt & 7) << 3) | (nt >> 3));
                rq_words[(size_t)128 * ((size_t)ta * (size_t)TN + (size_t)tnA) +
                         (size_t)off] = word;
            }
        }

        // ---- pinned dot chains (ascending j inside each lane) ----
        {
            const int n_a = n0 + d_nl;
            const int n_b = n_a + 16;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int jj = d_l + 16 * e;
                if (j0 + jj < K) {
                    const int g = sh_g[slot][jj];
                    const float xv = sh_xp[slot][jj];
                    if (n_a < N) {
                        const int qz = (int)sh_qz[cur][jj][d_nl];
                        const float s =
                            sc_t[(size_t)n_a * (size_t)G + (size_t)g];
                        acc0 = __fadd_rn(acc0,
                            __fmul_rn(__fmul_rn((float)qz, s), xv));
                        zac0 += qz;
                    }
                    if (n_b < N) {
                        const int qz = (int)sh_qz[cur][jj][d_nl + 16];
                        const float s =
                            sc_t[(size_t)n_b * (size_t)G + (size_t)g];
                        acc1 = __fadd_rn(acc1,
                            __fmul_rn(__fmul_rn((float)qz, s), xv));
                        zac1 += qz;
                    }
                }
            }
        }

        // ---- digest chains: the chunk holds exactly 8 stream words and the
        //      substreams are round-robin (word w -> substream w mod 8), so
        //      thread r owns the single word starting at byte j0 + 8*r.
        //      All 8 chains of every column fold one word per chunk, fully
        //      in parallel; the ragged-K straddle word is carried in wacc. ----
        if (dig_live) {
            const int i0 = j0 + 8 * g_r;
            if (i0 + 8 <= K) {
                uint64_t word = 0;
                #pragma unroll
                for (int b = 0; b < 8; ++b) {
                    word |= (uint64_t)(uint8_t)sh_qz[cur][i0 + b - j0][g_nl]
                            << (8 * b);
                }
                sub = awq_d_fnv_word(sub, word);
            } else if (i0 < K) {
                for (int i = i0; i < K; ++i) {
                    wacc |= (uint64_t)(uint8_t)sh_qz[cur][i - j0][g_nl]
                            << (8 * (i - i0));
                }
            }
        }

        // stage the prefetched perm tile into slot (c+2)%3 (idle since c-1)
        if (pfk) {
            const int s2 = (c + 2) % 3;
            sh_k[s2][tid] = rk;
            sh_g[s2][tid] = rg;
            sh_xp[s2][tid] = rx;
        }
        __syncthreads();
    }

    // ---- digest tail: words containing scale bytes (starting with the
    //      straddle word carried in wacc), ascending within the substream ----
    if (dig_live) {
        const int n = n0 + g_nl;
        const int wS = K / 8;  // first word containing scale bytes
        int last_g = -1;
        uint32_t sbits = 0;
        for (int w = wS + (((g_r - wS) % 8) + 8) % 8; w < W; w += 8) {
            const int b0 = 8 * w;
            uint64_t word = (b0 < K) ? wacc : 0;
            for (int b = (b0 < K) ? (K - b0) : 0; b < 8; ++b) {
                const int i = b0 + b;
                if (i >= L) break;
                const int idx = i - K;
                const int gg = idx >> 2;
                if (gg != last_g) {
                    last_g = gg;
                    sbits = __float_as_uint(
                        sc_t[(size_t)n * (size_t)G + (size_t)gg]);
                }
                word |= (uint64_t)((sbits >> (8 * (idx & 3))) & 0xffu) << (8 * b);
            }
            sub = awq_d_fnv_word(sub, word);
        }
    }

    // ---- dot + zsum trees (exact contract order) ----
    sh_dot[d_l][d_nl] = acc0;
    sh_dot[d_l][d_nl + 16] = acc1;
    sh_z[d_l][d_nl] = zac0;
    sh_z[d_l][d_nl + 16] = zac1;
    sh_sub[g_nl][g_r] = sub;
    __syncthreads();

    {
        const int tl = tid >> 5;            // [0, 8)
        const int tn = tid & 31;            // [0, 32)
        #pragma unroll
        for (int stride = 8; stride > 0; stride >>= 1) {
            if (tl < stride) {
                sh_dot[tl][tn] = __fadd_rn(sh_dot[tl][tn], sh_dot[tl + stride][tn]);
                sh_z[tl][tn] += sh_z[tl + stride][tn];
            }
            __syncthreads();
        }
        if (tl == 0 && (n0 + tn) < N) {
            col_dot[n0 + tn] = sh_dot[0][tn];
            col_zsum[n0 + tn] = sh_z[0][tn];
        }
    }

    // ---- digest combine ----
    if (tid < 32 && (n0 + tid) < N) {
        uint64_t h = AWQ_FNV_BASIS;
        #pragma unroll
        for (int rr = 0; rr < AWQ_DIGEST_SUBSTREAMS; ++rr) {
            h = awq_d_fnv_word(h, sh_sub[tid][rr]);
        }
        col_digest[n0 + tid] = h;
    }
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

static size_t awq_align256(size_t x) { return (x + 255u) & ~(size_t)255u; }

static void awq_ws_layout(
    int max_K, int max_N, int max_G,
    size_t* off_xp, size_t* off_gsorted,
    size_t* off_hist, size_t* off_base, size_t* off_gstart,
    size_t* off_sct, size_t* total) {
    const int Kp = 16 * awq_TA(max_K);
    const int B = awq_ceil_div_int(max_K, AWQ_RANK_CHUNK);
    size_t off = 0;
    *off_xp = off;      off += awq_align256((size_t)Kp * 4);
    *off_gsorted = off; off += awq_align256((size_t)Kp * 4);
    *off_hist = off;    off += awq_align256((size_t)B * (size_t)max_G * 4);
    *off_base = off;    off += awq_align256((size_t)B * (size_t)max_G * 4);
    *off_gstart = off;  off += awq_align256((size_t)max_G * 4);
    *off_sct = off;     off += awq_align256((size_t)max_N * (size_t)max_G * 4);
    *total = off;
}

extern "C" size_t solution_workspace_bytes(const AwqProblemSpec* spec) {
    if (!awq_validate_problem_spec(spec)) return 0;
    size_t a, b, c, d, e, f, total;
    awq_ws_layout(spec->max_K, spec->max_N, spec->max_G,
                  &a, &b, &c, &d, &e, &f, &total);
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
    AwqReferenceState* st =
        static_cast<AwqReferenceState*>(malloc(sizeof(AwqReferenceState)));
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

    AwqReferenceState* st = static_cast<AwqReferenceState*>(state);
    const AwqInputs* in = static_cast<const AwqInputs*>(inputs_void);
    AwqOutputs* out = static_cast<AwqOutputs*>(outputs_void);

    if (run->K > st->spec.max_K || run->N > st->spec.max_N ||
        run->G > st->spec.max_G) {
        return cudaErrorInvalidValue;
    }
    if (!in->qweight || !in->qzeros || !in->scales || !in->g_idx || !in->x ||
        !out->rq_atoms || !out->col_dot || !out->col_digest || !out->col_zsum ||
        !out->perm) {
        return cudaErrorInvalidValue;
    }

    size_t off_xp, off_gsorted, off_hist, off_base, off_gstart, off_sct, need;
    awq_ws_layout(st->spec.max_K, st->spec.max_N, st->spec.max_G,
                  &off_xp, &off_gsorted, &off_hist, &off_base, &off_gstart,
                  &off_sct, &need);
    if (workspace_bytes < need) return cudaErrorInvalidValue;

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    float* xp = reinterpret_cast<float*>(ws + off_xp);
    int32_t* gsorted = reinterpret_cast<int32_t*>(ws + off_gsorted);
    int32_t* ws_hist = reinterpret_cast<int32_t*>(ws + off_hist);
    int32_t* ws_base = reinterpret_cast<int32_t*>(ws + off_base);
    int32_t* ws_gstart = reinterpret_cast<int32_t*>(ws + off_gstart);
    float* ws_sct = reinterpret_cast<float*>(ws + off_sct);

    const int K = run->K;
    const int N = run->N;
    const int G = run->G;
    const int Nw = awq_Nw(N);
    const int TA = awq_TA(K);
    const int TN = awq_TN(N);
    const int B = awq_ceil_div_int(K, AWQ_RANK_CHUNK);

    awq_ref_hist_kernel<<<B, AWQ_RANK_CHUNK, (size_t)G * 4, stream>>>(
        K, G, in->g_idx, ws_hist);
    awq_ref_offsets_kernel<<<1, 512, (size_t)G * 4, stream>>>(
        B, G, ws_hist, ws_base, ws_gstart);
    awq_ref_scatter_kernel<<<B, AWQ_RANK_CHUNK, 0, stream>>>(
        K, G, in->g_idx, in->x, ws_base, out->perm, gsorted, xp);

    {
        dim3 tgrid((unsigned)awq_ceil_div_int(G, 32),
                   (unsigned)awq_ceil_div_int(N, 32));
        dim3 tblock(32, 8);
        awq_ref_sctrans_kernel<<<tgrid, tblock, 0, stream>>>(
            N, G, in->scales, ws_sct);
    }

    awq_ref_fused_kernel<<<2 * TN, 256, 0, stream>>>(
        K, N, G, Nw, TA, TN,
        in->qweight, in->qzeros, ws_sct,
        out->perm, gsorted, xp,
        reinterpret_cast<uint32_t*>(out->rq_atoms),
        out->col_dot, out->col_digest, out->col_zsum);

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
