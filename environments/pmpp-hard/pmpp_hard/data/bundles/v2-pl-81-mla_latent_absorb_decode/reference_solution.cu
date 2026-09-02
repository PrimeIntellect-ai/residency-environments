// PMPP_CANARY_81_9e2c47fa50 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: mla_latent_absorb_decode_reference.cu
//
// Optimization notes (held out from solvers):
//
//  * Absorption both ways: ql = W_uk^T q is computed once per (token, head)
//    against a transposed weight copy built at init (contiguous fp32/float4
//    rows), and the value up-projection runs once per (token, head) on the
//    softmax-weighted latent sum z, never per attended position.
//  * Attention is flash-decoding style: grid (query, split); 4 warps per
//    block, warp-interleaved tokens, char4 quantized latent loads with
//    inline power-of-two dequant, warp-shuffle score reduction, per-warp
//    online (m, l, z[d_c]) recurrence in registers, block combine, per-split
//    partial records merged by LSE rescaling in the merge kernel.
//  * The per-sequence FNV cache hash is INCREMENTAL: a running hash per
//    sequence is extended only by this step's appended bytes (the contract
//    folds seq_len LAST, so the append-only token stream keeps a valid
//    running prefix); the final output folds 4 length bytes. A from-scratch
//    refold per step is O(L * bytes/token) serial and blows the budget.
//  * Quantization: one block per appended token, one warp per 32-channel
//    group, amax via shuffle butterflies, __float2int_rn for the normative
//    rne, ballot/popc saturation counting aggregated once per block.
// ============================================================================

#include "mla_latent_absorb_decode_common.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t kFnvBasis = 1469598103934665603ULL;
static const uint64_t kFnvPrime = 1099511628211ULL;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct MlaRefState {
    MlaProblemSpec spec;
    int gc;        // d_c / 32
    int gr;        // d_r / 32
    int half_r;    // d_r / 2
    int n_splits;

    // Quantized cache.
    int8_t* c_scale;   // [B * msl * gc]
    int8_t* c_byte;    // [B * msl * d_c]
    int8_t* r_scale;   // [B * msl * gr]
    int8_t* r_byte;    // [B * msl * d_r]

    int32_t* seq_len;  // [B]
    uint64_t* h_run;   // [B] running FNV over appended token bytes
    int32_t* counters; // [0]=step_counter [1]=sat_count [2]=total_tokens

    // Weights.
    float* W_ukT;      // [Hq * d_c * d_h] transposed: (h, j, i)
    float* W_uv;       // [Hq * d_v * d_c] original layout
    float* rope_cos;   // [msl * half_r]
    float* rope_sin;   // [msl * half_r]
};

static int mla_ref_splits(const MlaProblemSpec* spec) {
    if (spec->max_seq_len >= 2048) return 4;
    if (spec->max_seq_len >= 512) return 2;
    return 1;
}

struct MlaRefWs {
    float* ql;        // [B*8*Hq * d_c]
    float* partials;  // [B*8*Hq * n_splits * (d_c + 2)]
    size_t required_bytes;
};

static MlaRefWs mla_ref_layout(void* ws, const MlaProblemSpec* spec) {
    MlaRefWs lay{};
    char* base = static_cast<char*>(ws);
    const size_t nq =
        (size_t)spec->B * MLA_MAX_NEW_TOKENS * (size_t)spec->Hq;
    const int S = mla_ref_splits(spec);
    size_t off = 0;

    off = mla_align_up_size(off, 128);
    lay.ql = reinterpret_cast<float*>(base + off);
    off += sizeof(float) * nq * (size_t)spec->d_c;

    off = mla_align_up_size(off, 128);
    lay.partials = reinterpret_cast<float*>(base + off);
    off += sizeof(float) * nq * (size_t)S * (size_t)(spec->d_c + 2);

    off = mla_align_up_size(off, 128);
    lay.required_bytes = off;
    return lay;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t mla_fnv_byte_d(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ __forceinline__ uint64_t mla_fnv_i32_d(uint64_t h, int32_t v) {
    const uint32_t u = static_cast<uint32_t>(v);
    h = mla_fnv_byte_d(h, (uint8_t)(u & 0xFF));
    h = mla_fnv_byte_d(h, (uint8_t)((u >> 8) & 0xFF));
    h = mla_fnv_byte_d(h, (uint8_t)((u >> 16) & 0xFF));
    h = mla_fnv_byte_d(h, (uint8_t)((u >> 24) & 0xFF));
    return h;
}

// 2^e as fp32 for e in [-126, 127] (scale exponents are in [-110, 110]).
__device__ __forceinline__ float mla_pow2f(int e) {
    return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

// ---------------------------------------------------------------------------
// Init kernels
// ---------------------------------------------------------------------------

__global__ void mla_transpose_wuk_kernel(
    const float* __restrict__ W_uk,  // [Hq, d_h, d_c]
    float* __restrict__ W_ukT,       // [Hq, d_c, d_h]
    int Hq,
    int d_h,
    int d_c) {
    const int h = blockIdx.z;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= Hq || j >= d_c || i >= d_h) return;
    W_ukT[((size_t)h * d_c + j) * d_h + i] =
        W_uk[((size_t)h * d_h + i) * d_c + j];
}

__global__ void mla_reset_kernel(
    int32_t* __restrict__ seq_len,
    uint64_t* __restrict__ h_run,
    int32_t* __restrict__ counters,
    int B) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) {
        seq_len[i] = 0;
        h_run[i] = kFnvBasis;
    }
    if (i < 3) counters[i] = 0;
}

// ---------------------------------------------------------------------------
// Kernel 1: quantize + append. Block per (a, nt); warp per group.
// ---------------------------------------------------------------------------

__global__ void mla_quantize_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ new_token_count,
    const float* __restrict__ new_c,
    const float* __restrict__ new_r,
    const int32_t* __restrict__ seq_len,   // pre-update
    int8_t* __restrict__ c_scale,
    int8_t* __restrict__ c_byte,
    int8_t* __restrict__ r_scale,
    int8_t* __restrict__ r_byte,
    int32_t* __restrict__ sat_counter,
    int d_c,
    int d_r,
    int gc,
    int gr,
    int msl) {
    const int a = blockIdx.x / MLA_MAX_NEW_TOKENS;
    const int nt = blockIdx.x - a * MLA_MAX_NEW_TOKENS;
    const int cnt = new_token_count[a];
    if (nt >= cnt) return;

    const int b = active_seq[a];
    const int pos = seq_len[b] + nt;
    const size_t ti = (size_t)b * msl + (size_t)pos;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;

    __shared__ int sh_sat;
    if (threadIdx.x == 0) sh_sat = 0;
    __syncthreads();

    const int n_groups = gc + gr;
    const float* cvec = new_c + ((size_t)a * MLA_MAX_NEW_TOKENS + nt) * d_c;
    const float* rvec = new_r + ((size_t)a * MLA_MAX_NEW_TOKENS + nt) * d_r;

    for (int g = warp; g < n_groups; g += nwarps) {
        const bool is_c = g < gc;
        const int gg = is_c ? g : g - gc;
        const float x = is_c ? cvec[gg * MLA_QUANT_GROUP + lane]
                             : rvec[gg * MLA_QUANT_GROUP + lane];

        float amax = fabsf(x);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        }

        int8_t se8 = 0;
        int8_t byte = 0;
        bool sat = false;
        if (amax != 0.0f) {
            const int kfloor =
                (int)((__float_as_uint(amax) >> 23) & 0xFF) - 127;
            int se = kfloor - 6;
            se = se < -110 ? -110 : (se > 110 ? 110 : se);
            se8 = (int8_t)se;
            const float z = x * mla_pow2f(-se);
            int n = __float2int_rn(z);
            if (n > 127) { n = 127; sat = true; }
            if (n < -127) { n = -127; sat = true; }
            byte = (int8_t)n;
        }

        const unsigned mask = __ballot_sync(0xFFFFFFFFu, sat);
        if (lane == 0 && mask != 0u) {
            atomicAdd(&sh_sat, __popc(mask));
        }

        if (is_c) {
            c_byte[ti * d_c + gg * MLA_QUANT_GROUP + lane] = byte;
            if (lane == 0) c_scale[ti * gc + gg] = se8;
        } else {
            r_byte[ti * d_r + gg * MLA_QUANT_GROUP + lane] = byte;
            if (lane == 0) r_scale[ti * gr + gg] = se8;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && sh_sat != 0) {
        atomicAdd(sat_counter, sh_sat);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: extend running per-sequence hashes with this step's bytes.
// One thread per active row; at most 8 * (d_c + gc + d_r + gr) folds each.
// ---------------------------------------------------------------------------

__global__ void mla_hash_extend_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ new_token_count,
    const int32_t* __restrict__ seq_len,   // pre-update
    const int8_t* __restrict__ c_scale,
    const int8_t* __restrict__ c_byte,
    const int8_t* __restrict__ r_scale,
    const int8_t* __restrict__ r_byte,
    uint64_t* __restrict__ h_run,
    int active_count,
    int d_c,
    int d_r,
    int gc,
    int gr,
    int msl) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= active_count) return;

    const int b = active_seq[a];
    const int cnt = new_token_count[a];
    const int len0 = seq_len[b];

    uint64_t h = h_run[b];
    for (int nt = 0; nt < cnt; ++nt) {
        const size_t ti = (size_t)b * msl + (size_t)(len0 + nt);
        for (int g = 0; g < gc; ++g) {
            h = mla_fnv_byte_d(h, (uint8_t)c_scale[ti * gc + g]);
        }
        for (int d = 0; d < d_c; ++d) {
            h = mla_fnv_byte_d(h, (uint8_t)c_byte[ti * d_c + d]);
        }
        for (int g = 0; g < gr; ++g) {
            h = mla_fnv_byte_d(h, (uint8_t)r_scale[ti * gr + g]);
        }
        for (int d = 0; d < d_r; ++d) {
            h = mla_fnv_byte_d(h, (uint8_t)r_byte[ti * d_r + d]);
        }
    }
    h_run[b] = h;
}

// ---------------------------------------------------------------------------
// Kernel 3: admin (single block): lengths, counters, exact outputs.
// ---------------------------------------------------------------------------

__global__ void mla_admin_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ new_token_count,
    int32_t* __restrict__ seq_len,
    const uint64_t* __restrict__ h_run,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ out_seq_len,
    uint64_t* __restrict__ out_cache_hash,
    uint64_t* __restrict__ out_meta,
    int32_t* __restrict__ out_sat,
    int32_t* __restrict__ out_total,
    int active_count,
    int B,
    int Hq,
    int d_c,
    int d_r,
    int d_h,
    int d_v,
    int msl) {
    const int tid = threadIdx.x;

    if (tid < active_count) {
        seq_len[active_seq[tid]] += new_token_count[tid];
    }
    __syncthreads();

    if (tid == 0) {
        counters[0] += 1;  // step_counter
        int tot = 0;
        for (int a = 0; a < active_count; ++a) tot += new_token_count[a];
        counters[2] += tot;
    }
    __syncthreads();

    if (tid < B) {
        const int32_t len = seq_len[tid];
        out_seq_len[tid] = len;
        out_cache_hash[tid] = mla_fnv_i32_d(h_run[tid], len);
    }
    __syncthreads();

    if (tid == 0) {
        uint64_t mh = kFnvBasis;
        mh = mla_fnv_i32_d(mh, B);
        mh = mla_fnv_i32_d(mh, Hq);
        mh = mla_fnv_i32_d(mh, d_c);
        mh = mla_fnv_i32_d(mh, d_r);
        mh = mla_fnv_i32_d(mh, d_h);
        mh = mla_fnv_i32_d(mh, d_v);
        mh = mla_fnv_i32_d(mh, msl);
        mh = mla_fnv_i32_d(mh, counters[0]);
        for (int b = 0; b < B; ++b) {
            mh = mla_fnv_i32_d(mh, seq_len[b]);
        }
        mh = mla_fnv_i32_d(mh, counters[1]);
        mh = mla_fnv_i32_d(mh, counters[2]);
        out_meta[0] = mh;
        out_sat[0] = counters[1];
        out_total[0] = counters[2];
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: query absorption ql = W_uk^T q. Grid (a*8+nt, h), 128 threads.
// ---------------------------------------------------------------------------

__global__ void mla_absorb_kernel(
    const int32_t* __restrict__ new_token_count,
    const float* __restrict__ q,           // [A, 8, Hq, d_h]
    const float* __restrict__ W_ukT,       // [Hq, d_c, d_h]
    float* __restrict__ ql_out,            // [A*8*Hq, d_c]
    int Hq,
    int d_h,
    int d_c) {
    const int slot = blockIdx.x;            // a*8 + nt
    const int a = slot / MLA_MAX_NEW_TOKENS;
    const int nt = slot - a * MLA_MAX_NEW_TOKENS;
    if (nt >= new_token_count[a]) return;
    const int h = blockIdx.y;

    extern __shared__ float sh_q[];  // [d_h]
    const float* qv = q + ((size_t)slot * Hq + h) * d_h;
    for (int i = threadIdx.x; i < d_h; i += blockDim.x) {
        sh_q[i] = qv[i];
    }
    __syncthreads();

    float* out = ql_out + ((size_t)slot * Hq + h) * d_c;
    for (int j = threadIdx.x; j < d_c; j += blockDim.x) {
        const float4* wrow = reinterpret_cast<const float4*>(
            W_ukT + ((size_t)h * d_c + j) * d_h);
        float acc = 0.0f;
        const int n4 = d_h >> 2;
        #pragma unroll 4
        for (int i4 = 0; i4 < n4; ++i4) {
            const float4 w = wrow[i4];
            const float4 qq = reinterpret_cast<const float4*>(sh_q)[i4];
            acc += w.x * qq.x + w.y * qq.y + w.z * qq.z + w.w * qq.w;
        }
        out[j] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: split flash attention over the quantized latent cache.
// Grid (query = (a*8+nt)*Hq + h, split). 128 threads = 4 warps.
// Warp-interleaved tokens; per-warp online (m, l, z) with char4 loads.
// ---------------------------------------------------------------------------

#define MLA_ATTN_WARPS 4

__global__ void mla_attention_impl_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ new_token_count,
    const float* __restrict__ q_rope,
    const float* __restrict__ ql_in,
    const int32_t* __restrict__ seq_len,
    const int8_t* __restrict__ c_scale,
    const int8_t* __restrict__ c_byte,
    const int8_t* __restrict__ r_scale,
    const int8_t* __restrict__ r_byte,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    float* __restrict__ partials,
    int Hq,
    int d_c,
    int d_r,
    int gc,
    int gr,
    int half_r,
    int msl,
    int n_splits,
    float sm_scale) {
    const int qidx = blockIdx.x;
    const int split = blockIdx.y;
    const int h = qidx % Hq;
    const int slot = qidx / Hq;
    const int a = slot / MLA_MAX_NEW_TOKENS;
    const int nt = slot - a * MLA_MAX_NEW_TOKENS;
    const int cnt = new_token_count[a];
    if (nt >= cnt) return;

    const int b = active_seq[a];
    const int p = seq_len[b] - cnt + nt;
    const int L = p + 1;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    extern __shared__ float smem[];
    float* sh_ql = smem;
    float* sh_qr = sh_ql + d_c;
    float* sh_z = sh_qr + d_r;
    float* sh_ml = sh_z + MLA_ATTN_WARPS * d_c;

    {
        const float* ql = ql_in + (size_t)qidx * d_c;
        for (int j = threadIdx.x; j < d_c; j += blockDim.x) sh_ql[j] = ql[j];
        const float* qr = q_rope + ((size_t)slot * Hq + h) * d_r;
        for (int j = threadIdx.x; j < d_r; j += blockDim.x) sh_qr[j] = qr[j];
    }
    __syncthreads();

    const int chunk = (L + n_splits - 1) / n_splits;
    const int t0 = split * chunk;
    const int t1 = (t0 + chunk) < L ? (t0 + chunk) : L;

    // Lane covers latent dims ic*128 + lane*4 .. +3 for ic in [0, nIC).
    const int nIC = (d_c + 127) >> 7;   // 1 for 128, 2 for 192/256

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float zacc[2][4];
    #pragma unroll
    for (int ic = 0; ic < 2; ++ic) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) zacc[ic][k] = 0.0f;
    }

    const size_t seq_base = (size_t)b * msl;

    for (int t = t0 + warp; t < t1; t += MLA_ATTN_WARPS) {
        const size_t ti = seq_base + (size_t)t;
        float part = 0.0f;
        float cval[2][4];

        #pragma unroll
        for (int ic = 0; ic < 2; ++ic) {
            const int d0 = (ic << 7) + (lane << 2);
            if (ic < nIC && d0 < d_c) {
                const char4 cb = *reinterpret_cast<const char4*>(
                    c_byte + ti * d_c + d0);
                const float sc = mla_pow2f((int)c_scale[ti * gc + (d0 >> 5)]);
                cval[ic][0] = (float)cb.x * sc;
                cval[ic][1] = (float)cb.y * sc;
                cval[ic][2] = (float)cb.z * sc;
                cval[ic][3] = (float)cb.w * sc;
                part += cval[ic][0] * sh_ql[d0 + 0];
                part += cval[ic][1] * sh_ql[d0 + 1];
                part += cval[ic][2] * sh_ql[d0 + 2];
                part += cval[ic][3] * sh_ql[d0 + 3];
            } else {
                cval[ic][0] = 0.0f;
                cval[ic][1] = 0.0f;
                cval[ic][2] = 0.0f;
                cval[ic][3] = 0.0f;
            }
        }

        {
            const int d0 = lane << 2;
            if (d0 < d_r) {
                const char4 rb = *reinterpret_cast<const char4*>(
                    r_byte + ti * d_r + d0);
                const float sc = mla_pow2f((int)r_scale[ti * gr + (d0 >> 5)]);
                const float r0 = (float)rb.x * sc;
                const float r1 = (float)rb.y * sc;
                const float r2 = (float)rb.z * sc;
                const float r3 = (float)rb.w * sc;
                const int j0 = d0 >> 1;
                const float2 cs01 = *reinterpret_cast<const float2*>(
                    rope_cos + (size_t)t * half_r + j0);
                const float2 sn01 = *reinterpret_cast<const float2*>(
                    rope_sin + (size_t)t * half_r + j0);
                const float rr0 = r0 * cs01.x - r1 * sn01.x;
                const float rr1 = r0 * sn01.x + r1 * cs01.x;
                const float rr2 = r2 * cs01.y - r3 * sn01.y;
                const float rr3 = r2 * sn01.y + r3 * cs01.y;
                part += sh_qr[d0 + 0] * rr0;
                part += sh_qr[d0 + 1] * rr1;
                part += sh_qr[d0 + 2] * rr2;
                part += sh_qr[d0 + 3] * rr3;
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            part += __shfl_xor_sync(0xFFFFFFFFu, part, off);
        }
        const float score = part * sm_scale;

        const float nm = fmaxf(m, score);
        const float alpha = (m == -CUDART_INF_F) ? 0.0f : __expf(m - nm);
        const float beta = __expf(score - nm);
        #pragma unroll
        for (int ic = 0; ic < 2; ++ic) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                zacc[ic][k] = zacc[ic][k] * alpha + beta * cval[ic][k];
            }
        }
        l = l * alpha + beta;
        m = nm;
    }

    // Stage per-warp results.
    #pragma unroll
    for (int ic = 0; ic < 2; ++ic) {
        const int d0 = (ic << 7) + (lane << 2);
        if (ic < nIC && d0 < d_c) {
            sh_z[warp * d_c + d0 + 0] = zacc[ic][0];
            sh_z[warp * d_c + d0 + 1] = zacc[ic][1];
            sh_z[warp * d_c + d0 + 2] = zacc[ic][2];
            sh_z[warp * d_c + d0 + 3] = zacc[ic][3];
        }
    }
    if (lane == 0) {
        sh_ml[warp * 2 + 0] = m;
        sh_ml[warp * 2 + 1] = l;
    }
    __syncthreads();

    // Block combine (fixed order, deterministic).
    __shared__ float sh_M, sh_L;
    if (threadIdx.x == 0) {
        float M = -CUDART_INF_F;
        #pragma unroll
        for (int w = 0; w < MLA_ATTN_WARPS; ++w) {
            M = fmaxf(M, sh_ml[w * 2 + 0]);
        }
        float Lc = 0.0f;
        #pragma unroll
        for (int w = 0; w < MLA_ATTN_WARPS; ++w) {
            const float mw = sh_ml[w * 2 + 0];
            const float lw = sh_ml[w * 2 + 1];
            const float f = (mw == -CUDART_INF_F) ? 0.0f : __expf(mw - M);
            Lc += lw * f;
            sh_ml[w * 2 + 1] = f;  // reuse: per-warp rescale factor
        }
        sh_M = M;
        sh_L = Lc;
    }
    __syncthreads();

    float* rec = partials + ((size_t)qidx * n_splits + split) * (d_c + 2);
    for (int j = threadIdx.x; j < d_c; j += blockDim.x) {
        float zj = 0.0f;
        #pragma unroll
        for (int w = 0; w < MLA_ATTN_WARPS; ++w) {
            zj += sh_z[w * d_c + j] * sh_ml[w * 2 + 1];
        }
        rec[2 + j] = zj;
    }
    if (threadIdx.x == 0) {
        rec[0] = sh_M;
        rec[1] = sh_L;
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: merge splits + value up-projection. Grid (slot, h), 128 threads.
// ---------------------------------------------------------------------------

__global__ void mla_merge_proj_kernel(
    const int32_t* __restrict__ new_token_count,
    const float* __restrict__ partials,   // [nq, S, d_c+2]
    const float* __restrict__ W_uv,       // [Hq, d_v, d_c]
    float* __restrict__ y,                // [A, 8, Hq, d_v]
    float* __restrict__ lse,              // [A, 8, Hq]
    int Hq,
    int d_c,
    int d_v,
    int n_splits) {
    const int slot = blockIdx.x;
    const int a = slot / MLA_MAX_NEW_TOKENS;
    const int nt = slot - a * MLA_MAX_NEW_TOKENS;
    if (nt >= new_token_count[a]) return;
    const int h = blockIdx.y;
    const int qidx = slot * Hq + h;

    extern __shared__ float sh_zf[];  // [d_c]

    __shared__ float sh_M, sh_L;
    __shared__ float sh_f[8];
    if (threadIdx.x == 0) {
        const float* base = partials + (size_t)qidx * n_splits * (d_c + 2);
        float M = -CUDART_INF_F;
        for (int s = 0; s < n_splits; ++s) {
            M = fmaxf(M, base[(size_t)s * (d_c + 2)]);
        }
        float Lc = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float ms = base[(size_t)s * (d_c + 2)];
            const float ls = base[(size_t)s * (d_c + 2) + 1];
            const float f = (ms == -CUDART_INF_F) ? 0.0f : __expf(ms - M);
            sh_f[s] = f;
            Lc += ls * f;
        }
        sh_M = M;
        sh_L = Lc;
    }
    __syncthreads();

    {
        const float* base = partials + (size_t)qidx * n_splits * (d_c + 2);
        const float inv_l = 1.0f / sh_L;
        for (int j = threadIdx.x; j < d_c; j += blockDim.x) {
            float zj = 0.0f;
            for (int s = 0; s < n_splits; ++s) {
                zj += base[(size_t)s * (d_c + 2) + 2 + j] * sh_f[s];
            }
            sh_zf[j] = zj * inv_l;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < d_v; i += blockDim.x) {
        const float4* wrow = reinterpret_cast<const float4*>(
            W_uv + ((size_t)h * d_v + i) * d_c);
        float acc = 0.0f;
        const int n4 = d_c >> 2;
        #pragma unroll 4
        for (int j4 = 0; j4 < n4; ++j4) {
            const float4 w = wrow[j4];
            const float4 zz = reinterpret_cast<const float4*>(sh_zf)[j4];
            acc += w.x * zz.x + w.y * zz.y + w.z * zz.z + w.w * zz.w;
        }
        y[(size_t)qidx * d_v + i] = acc;
    }
    if (threadIdx.x == 0) {
        lse[qidx] = sh_M + logf(sh_L);
    }
}

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------

extern "C" size_t solution_workspace_bytes(const MlaProblemSpec* spec) {
    if (!mla_validate_problem_spec(spec)) return 0;
    return mla_ref_layout(nullptr, spec).required_bytes;
}

extern "C" cudaError_t solution_init(
    const MlaProblemSpec* spec,
    const MlaInitInputs* init_inputs,
    void** state_out,
    cudaStream_t stream) {
    if (!state_out) return cudaErrorInvalidValue;
    *state_out = nullptr;
    if (!mla_validate_problem_spec(spec) || !init_inputs) {
        return cudaErrorInvalidValue;
    }

    MlaRefState* st = static_cast<MlaRefState*>(malloc(sizeof(MlaRefState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(*st));
    st->spec = *spec;
    st->gc = spec->d_c / MLA_QUANT_GROUP;
    st->gr = spec->d_r / MLA_QUANT_GROUP;
    st->half_r = spec->d_r / 2;
    st->n_splits = mla_ref_splits(spec);

    const int B = spec->B;
    const int msl = spec->max_seq_len;
    const size_t toks = (size_t)B * msl;

    cudaError_t err = cudaSuccess;
    #define MLA_ALLOC(ptr, bytes)                                             \
        do {                                                                  \
            err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes));      \
            if (err != cudaSuccess) goto fail;                                \
        } while (0)

    MLA_ALLOC(st->c_scale, toks * st->gc);
    MLA_ALLOC(st->c_byte, toks * spec->d_c);
    MLA_ALLOC(st->r_scale, toks * st->gr);
    MLA_ALLOC(st->r_byte, toks * spec->d_r);
    MLA_ALLOC(st->seq_len, sizeof(int32_t) * B);
    MLA_ALLOC(st->h_run, sizeof(uint64_t) * B);
    MLA_ALLOC(st->counters, sizeof(int32_t) * 4);
    MLA_ALLOC(st->W_ukT, sizeof(float) * (size_t)spec->Hq * spec->d_c * spec->d_h);
    MLA_ALLOC(st->W_uv, sizeof(float) * (size_t)spec->Hq * spec->d_v * spec->d_c);
    MLA_ALLOC(st->rope_cos, sizeof(float) * (size_t)msl * st->half_r);
    MLA_ALLOC(st->rope_sin, sizeof(float) * (size_t)msl * st->half_r);
    #undef MLA_ALLOC

    {
        dim3 blk(32, 8, 1);
        dim3 grd((spec->d_h + 31) / 32, (spec->d_c + 7) / 8, spec->Hq);
        mla_transpose_wuk_kernel<<<grd, blk, 0, stream>>>(
            init_inputs->W_uk, st->W_ukT, spec->Hq, spec->d_h, spec->d_c);

        err = cudaMemcpyAsync(
            st->W_uv, init_inputs->W_uv,
            sizeof(float) * (size_t)spec->Hq * spec->d_v * spec->d_c,
            cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpyAsync(
            st->rope_cos, init_inputs->rope_cos,
            sizeof(float) * (size_t)msl * st->half_r,
            cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpyAsync(
            st->rope_sin, init_inputs->rope_sin,
            sizeof(float) * (size_t)msl * st->half_r,
            cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) goto fail;

        mla_reset_kernel<<<1, 64, 0, stream>>>(
            st->seq_len, st->h_run, st->counters, B);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto fail;
    }

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->c_scale) cudaFree(st->c_scale);
    if (st->c_byte) cudaFree(st->c_byte);
    if (st->r_scale) cudaFree(st->r_scale);
    if (st->r_byte) cudaFree(st->r_byte);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->h_run) cudaFree(st->h_run);
    if (st->counters) cudaFree(st->counters);
    if (st->W_ukT) cudaFree(st->W_ukT);
    if (st->W_uv) cudaFree(st->W_uv);
    if (st->rope_cos) cudaFree(st->rope_cos);
    if (st->rope_sin) cudaFree(st->rope_sin);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const MlaRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    MlaRefState* st = static_cast<MlaRefState*>(state);
    if (!st || !run || !inputs || !outputs) return cudaErrorInvalidValue;
    if (!mla_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const MlaInputs* in = static_cast<const MlaInputs*>(inputs);
    MlaOutputs* out = static_cast<MlaOutputs*>(outputs);
    const MlaProblemSpec& sp = st->spec;
    const int A = run->active_count;

    MlaRefWs ws = mla_ref_layout(workspace, &sp);
    if (workspace_bytes < ws.required_bytes) return cudaErrorInvalidValue;

    cudaError_t err;

    if (A > 0) {
        const size_t y_floats =
            (size_t)A * MLA_MAX_NEW_TOKENS * sp.Hq * sp.d_v;
        const size_t lse_floats = (size_t)A * MLA_MAX_NEW_TOKENS * sp.Hq;
        err = cudaMemsetAsync(out->y, 0, y_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;
        err = cudaMemsetAsync(out->lse, 0, lse_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;

        mla_quantize_kernel<<<A * MLA_MAX_NEW_TOKENS, 128, 0, stream>>>(
            in->active_seq, in->new_token_count, in->new_c, in->new_r,
            st->seq_len,
            st->c_scale, st->c_byte, st->r_scale, st->r_byte,
            st->counters + 1,
            sp.d_c, sp.d_r, st->gc, st->gr, sp.max_seq_len);

        mla_hash_extend_kernel<<<1, 32, 0, stream>>>(
            in->active_seq, in->new_token_count, st->seq_len,
            st->c_scale, st->c_byte, st->r_scale, st->r_byte,
            st->h_run, A, sp.d_c, sp.d_r, st->gc, st->gr, sp.max_seq_len);
    }

    mla_admin_kernel<<<1, 64, 0, stream>>>(
        in->active_seq, in->new_token_count, st->seq_len, st->h_run,
        st->counters,
        out->seq_len, out->cache_hash, out->meta_checksum,
        out->sat_count, out->total_tokens,
        A, sp.B, sp.Hq, sp.d_c, sp.d_r, sp.d_h, sp.d_v, sp.max_seq_len);

    if (A > 0) {
        {
            dim3 grd(A * MLA_MAX_NEW_TOKENS, sp.Hq);
            const size_t shmem = sizeof(float) * sp.d_h;
            mla_absorb_kernel<<<grd, 128, shmem, stream>>>(
                in->new_token_count, in->q, st->W_ukT, ws.ql,
                sp.Hq, sp.d_h, sp.d_c);
        }

        {
            dim3 grd(A * MLA_MAX_NEW_TOKENS * sp.Hq, st->n_splits);
            const size_t shmem =
                sizeof(float) *
                (sp.d_c + sp.d_r + MLA_ATTN_WARPS * sp.d_c + 2 * MLA_ATTN_WARPS);
            const float sm_scale = rsqrtf((float)(sp.d_h + sp.d_r));
            mla_attention_impl_kernel<<<grd, 32 * MLA_ATTN_WARPS, shmem, stream>>>(
                in->active_seq, in->new_token_count, in->q_rope, ws.ql,
                st->seq_len,
                st->c_scale, st->c_byte, st->r_scale, st->r_byte,
                st->rope_cos, st->rope_sin,
                ws.partials,
                sp.Hq, sp.d_c, sp.d_r, st->gc, st->gr, st->half_r,
                sp.max_seq_len, st->n_splits, sm_scale);
        }

        {
            dim3 grd(A * MLA_MAX_NEW_TOKENS, sp.Hq);
            const size_t shmem = sizeof(float) * sp.d_c;
            mla_merge_proj_kernel<<<grd, 128, shmem, stream>>>(
                in->new_token_count, ws.partials, st->W_uv,
                out->y, out->lse,
                sp.Hq, sp.d_c, sp.d_v, st->n_splits);
        }
    }

    return cudaGetLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    MlaRefState* st = static_cast<MlaRefState*>(state);
    if (!st) return cudaErrorInvalidValue;
    mla_reset_kernel<<<1, 64, 0, stream>>>(
        st->seq_len, st->h_run, st->counters, st->spec.B);
    return cudaGetLastError();
}

extern "C" void solution_destroy(void* state) {
    MlaRefState* st = static_cast<MlaRefState*>(state);
    if (!st) return;
    if (st->c_scale) cudaFree(st->c_scale);
    if (st->c_byte) cudaFree(st->c_byte);
    if (st->r_scale) cudaFree(st->r_scale);
    if (st->r_byte) cudaFree(st->r_byte);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->h_run) cudaFree(st->h_run);
    if (st->counters) cudaFree(st->counters);
    if (st->W_ukT) cudaFree(st->W_ukT);
    if (st->W_uv) cudaFree(st->W_uv);
    if (st->rope_cos) cudaFree(st->rope_cos);
    if (st->rope_sin) cudaFree(st->rope_sin);
    free(st);
}
