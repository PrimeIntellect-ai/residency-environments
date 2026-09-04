// ============================================================================
// file: naive_ref.cu  (authoring artifact, NOT shipped to solvers)
//
// Clean, obviously-correct but unoptimized implementation of the
// mla_latent_absorb_decode contract:
//   - one scalar thread per appended token for quantization,
//   - single-thread admin kernel,
//   - one thread per sequence refolding the ENTIRE cache hash from scratch
//     every step (no incremental running hash),
//   - one thread per (row, token, head) computing the absorbed query, a
//     two-pass softmax re-reading the cache, the weighted latent sum, and
//     the value up-projection, all serially.
// Used to measure the naive/reference performance ratio.
// ============================================================================

#include "mla_latent_absorb_decode_common.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t kNvFnvBasis = 1469598103934665603ULL;

struct MlaNaiveState {
    MlaProblemSpec spec;
    int gc;
    int gr;
    int half_r;

    int8_t* c_scale;
    int8_t* c_byte;
    int8_t* r_scale;
    int8_t* r_byte;

    int32_t* seq_len;
    int32_t* counters;  // [0]=step_counter [1]=sat [2]=total_tokens

    float* W_uk;       // [Hq, d_h, d_c] original layout
    float* W_uv;       // [Hq, d_v, d_c]
    float* rope_cos;
    float* rope_sin;
};

struct MlaNaiveWs {
    float* ql;   // [B*8*Hq * d_c]
    float* z;    // [B*8*Hq * d_c]
    size_t required_bytes;
};

static MlaNaiveWs mla_nv_layout(void* ws, const MlaProblemSpec* spec) {
    MlaNaiveWs lay{};
    char* base = static_cast<char*>(ws);
    const size_t nq = (size_t)spec->B * MLA_MAX_NEW_TOKENS * (size_t)spec->Hq;
    size_t off = 0;

    off = mla_align_up_size(off, 128);
    lay.ql = reinterpret_cast<float*>(base + off);
    off += sizeof(float) * nq * (size_t)spec->d_c;

    off = mla_align_up_size(off, 128);
    lay.z = reinterpret_cast<float*>(base + off);
    off += sizeof(float) * nq * (size_t)spec->d_c;

    off = mla_align_up_size(off, 128);
    lay.required_bytes = off;
    return lay;
}

__device__ __forceinline__ uint64_t nv_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ __forceinline__ uint64_t nv_fnv_i32(uint64_t h, int32_t v) {
    const uint32_t u = static_cast<uint32_t>(v);
    h = nv_fnv_byte(h, (uint8_t)(u & 0xFF));
    h = nv_fnv_byte(h, (uint8_t)((u >> 8) & 0xFF));
    h = nv_fnv_byte(h, (uint8_t)((u >> 16) & 0xFF));
    h = nv_fnv_byte(h, (uint8_t)((u >> 24) & 0xFF));
    return h;
}

__device__ __forceinline__ float nv_pow2f(int e) {
    return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

// One scalar thread per (a, nt): quantize both sections of one token.
__global__ void mla_nv_quantize_kernel(
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
    const int a = blockIdx.x;
    const int nt = threadIdx.x;
    if (nt >= new_token_count[a]) return;

    const int b = active_seq[a];
    const int pos = seq_len[b] + nt;
    const size_t ti = (size_t)b * msl + (size_t)pos;

    int local_sat = 0;
    for (int sec = 0; sec < 2; ++sec) {
        const int dims = sec == 0 ? d_c : d_r;
        const int ng = sec == 0 ? gc : gr;
        const float* x = sec == 0
            ? new_c + ((size_t)a * MLA_MAX_NEW_TOKENS + nt) * d_c
            : new_r + ((size_t)a * MLA_MAX_NEW_TOKENS + nt) * d_r;
        int8_t* sc_out = sec == 0 ? c_scale + ti * gc : r_scale + ti * gr;
        int8_t* by_out = sec == 0 ? c_byte + ti * d_c : r_byte + ti * d_r;
        (void)dims;

        for (int g = 0; g < ng; ++g) {
            float amax = 0.0f;
            for (int d = 0; d < MLA_QUANT_GROUP; ++d) {
                const float v = fabsf(x[g * MLA_QUANT_GROUP + d]);
                if (v > amax) amax = v;
            }
            if (amax == 0.0f) {
                sc_out[g] = 0;
                for (int d = 0; d < MLA_QUANT_GROUP; ++d) {
                    by_out[g * MLA_QUANT_GROUP + d] = 0;
                }
                continue;
            }
            const int kfloor =
                (int)((__float_as_uint(amax) >> 23) & 0xFF) - 127;
            int se = kfloor - 6;
            se = se < -110 ? -110 : (se > 110 ? 110 : se);
            sc_out[g] = (int8_t)se;
            for (int d = 0; d < MLA_QUANT_GROUP; ++d) {
                const float z = x[g * MLA_QUANT_GROUP + d] * nv_pow2f(-se);
                int n = __float2int_rn(z);
                if (n > 127) { n = 127; local_sat += 1; }
                if (n < -127) { n = -127; local_sat += 1; }
                by_out[g * MLA_QUANT_GROUP + d] = (int8_t)n;
            }
        }
    }
    if (local_sat != 0) atomicAdd(sat_counter, local_sat);
}

// Single-thread admin.
__global__ void mla_nv_admin_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ new_token_count,
    int32_t* __restrict__ seq_len,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ out_seq_len,
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
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    counters[0] += 1;
    for (int a = 0; a < active_count; ++a) {
        seq_len[active_seq[a]] += new_token_count[a];
        counters[2] += new_token_count[a];
    }
    for (int b = 0; b < B; ++b) out_seq_len[b] = seq_len[b];

    uint64_t mh = kNvFnvBasis;
    mh = nv_fnv_i32(mh, B);
    mh = nv_fnv_i32(mh, Hq);
    mh = nv_fnv_i32(mh, d_c);
    mh = nv_fnv_i32(mh, d_r);
    mh = nv_fnv_i32(mh, d_h);
    mh = nv_fnv_i32(mh, d_v);
    mh = nv_fnv_i32(mh, msl);
    mh = nv_fnv_i32(mh, counters[0]);
    for (int b = 0; b < B; ++b) mh = nv_fnv_i32(mh, seq_len[b]);
    mh = nv_fnv_i32(mh, counters[1]);
    mh = nv_fnv_i32(mh, counters[2]);
    out_meta[0] = mh;
    out_sat[0] = counters[1];
    out_total[0] = counters[2];
}

// One thread per sequence: full from-scratch refold every step.
__global__ void mla_nv_hash_kernel(
    const int32_t* __restrict__ seq_len,   // post-update
    const int8_t* __restrict__ c_scale,
    const int8_t* __restrict__ c_byte,
    const int8_t* __restrict__ r_scale,
    const int8_t* __restrict__ r_byte,
    uint64_t* __restrict__ out_cache_hash,
    int B,
    int d_c,
    int d_r,
    int gc,
    int gr,
    int msl) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    uint64_t h = kNvFnvBasis;
    const int L = seq_len[b];
    for (int t = 0; t < L; ++t) {
        const size_t ti = (size_t)b * msl + (size_t)t;
        for (int g = 0; g < gc; ++g) h = nv_fnv_byte(h, (uint8_t)c_scale[ti * gc + g]);
        for (int d = 0; d < d_c; ++d) h = nv_fnv_byte(h, (uint8_t)c_byte[ti * d_c + d]);
        for (int g = 0; g < gr; ++g) h = nv_fnv_byte(h, (uint8_t)r_scale[ti * gr + g]);
        for (int d = 0; d < d_r; ++d) h = nv_fnv_byte(h, (uint8_t)r_byte[ti * d_r + d]);
    }
    h = nv_fnv_i32(h, L);
    out_cache_hash[b] = h;
}

// One scalar thread per (slot, head): absorb + two-pass softmax + upproj.
__global__ void mla_nv_attention_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ new_token_count,
    const float* __restrict__ q,
    const float* __restrict__ q_rope,
    const int32_t* __restrict__ seq_len,   // post-update
    const int8_t* __restrict__ c_scale,
    const int8_t* __restrict__ c_byte,
    const int8_t* __restrict__ r_scale,
    const int8_t* __restrict__ r_byte,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    const float* __restrict__ W_uk,
    const float* __restrict__ W_uv,
    float* __restrict__ ql_ws,
    float* __restrict__ z_ws,
    float* __restrict__ y,
    float* __restrict__ lse,
    int Hq,
    int d_c,
    int d_r,
    int d_h,
    int d_v,
    int gc,
    int gr,
    int half_r,
    int msl,
    int nq) {
    const int qidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qidx >= nq) return;
    const int h = qidx % Hq;
    const int slot = qidx / Hq;
    const int a = slot / MLA_MAX_NEW_TOKENS;
    const int nt = slot - a * MLA_MAX_NEW_TOKENS;
    const int cnt = new_token_count[a];
    if (nt >= cnt) return;

    const int b = active_seq[a];
    const int p = seq_len[b] - cnt + nt;
    const int L = p + 1;
    const float sm_scale = rsqrtf((float)(d_h + d_r));

    float* ql = ql_ws + (size_t)qidx * d_c;
    float* z = z_ws + (size_t)qidx * d_c;

    const float* qv = q + ((size_t)slot * Hq + h) * d_h;
    const float* qr = q_rope + ((size_t)slot * Hq + h) * d_r;
    const float* wuk = W_uk + (size_t)h * d_h * d_c;

    for (int j = 0; j < d_c; ++j) ql[j] = 0.0f;
    for (int i = 0; i < d_h; ++i) {
        const float qi = qv[i];
        for (int j = 0; j < d_c; ++j) {
            ql[j] += qi * wuk[(size_t)i * d_c + j];
        }
    }

    const size_t seq_base = (size_t)b * msl;

    // Pass 1: max score.
    float m = -CUDART_INF_F;
    for (int t = 0; t < L; ++t) {
        const size_t ti = seq_base + (size_t)t;
        float s = 0.0f;
        for (int d = 0; d < d_c; ++d) {
            const float sc = nv_pow2f((int)c_scale[ti * gc + (d >> 5)]);
            s += ql[d] * ((float)c_byte[ti * d_c + d] * sc);
        }
        for (int j = 0; j < half_r; ++j) {
            const int d0 = 2 * j;
            const float sc0 = nv_pow2f((int)r_scale[ti * gr + (d0 >> 5)]);
            const float sc1 = nv_pow2f((int)r_scale[ti * gr + ((d0 + 1) >> 5)]);
            const float r0 = (float)r_byte[ti * d_r + d0] * sc0;
            const float r1 = (float)r_byte[ti * d_r + d0 + 1] * sc1;
            const float co = rope_cos[(size_t)t * half_r + j];
            const float si = rope_sin[(size_t)t * half_r + j];
            s += qr[d0] * (r0 * co - r1 * si);
            s += qr[d0 + 1] * (r0 * si + r1 * co);
        }
        s *= sm_scale;
        if (s > m) m = s;
    }

    // Pass 2: denominator and weighted latent sum (scores recomputed).
    for (int j = 0; j < d_c; ++j) z[j] = 0.0f;
    float l = 0.0f;
    for (int t = 0; t < L; ++t) {
        const size_t ti = seq_base + (size_t)t;
        float s = 0.0f;
        for (int d = 0; d < d_c; ++d) {
            const float sc = nv_pow2f((int)c_scale[ti * gc + (d >> 5)]);
            s += ql[d] * ((float)c_byte[ti * d_c + d] * sc);
        }
        for (int j = 0; j < half_r; ++j) {
            const int d0 = 2 * j;
            const float sc0 = nv_pow2f((int)r_scale[ti * gr + (d0 >> 5)]);
            const float sc1 = nv_pow2f((int)r_scale[ti * gr + ((d0 + 1) >> 5)]);
            const float r0 = (float)r_byte[ti * d_r + d0] * sc0;
            const float r1 = (float)r_byte[ti * d_r + d0 + 1] * sc1;
            const float co = rope_cos[(size_t)t * half_r + j];
            const float si = rope_sin[(size_t)t * half_r + j];
            s += qr[d0] * (r0 * co - r1 * si);
            s += qr[d0 + 1] * (r0 * si + r1 * co);
        }
        s *= sm_scale;
        const float w = expf(s - m);
        l += w;
        for (int d = 0; d < d_c; ++d) {
            const float sc = nv_pow2f((int)c_scale[ti * gc + (d >> 5)]);
            z[d] += w * ((float)c_byte[ti * d_c + d] * sc);
        }
    }
    const float inv_l = 1.0f / l;
    for (int j = 0; j < d_c; ++j) z[j] *= inv_l;

    const float* wuv = W_uv + (size_t)h * d_v * d_c;
    for (int i = 0; i < d_v; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < d_c; ++j) {
            acc += wuv[(size_t)i * d_c + j] * z[j];
        }
        y[(size_t)qidx * d_v + i] = acc;
    }
    lse[qidx] = m + logf(l);
}

__global__ void mla_nv_reset_kernel(
    int32_t* __restrict__ seq_len,
    int32_t* __restrict__ counters,
    int B) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) seq_len[i] = 0;
    if (i < 3) counters[i] = 0;
}

extern "C" size_t solution_workspace_bytes(const MlaProblemSpec* spec) {
    if (!mla_validate_problem_spec(spec)) return 0;
    return mla_nv_layout(nullptr, spec).required_bytes;
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

    MlaNaiveState* st =
        static_cast<MlaNaiveState*>(malloc(sizeof(MlaNaiveState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(*st));
    st->spec = *spec;
    st->gc = spec->d_c / MLA_QUANT_GROUP;
    st->gr = spec->d_r / MLA_QUANT_GROUP;
    st->half_r = spec->d_r / 2;

    const int B = spec->B;
    const int msl = spec->max_seq_len;
    const size_t toks = (size_t)B * msl;

    cudaError_t err = cudaSuccess;
    #define NV_ALLOC(ptr, bytes)                                              \
        do {                                                                  \
            err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes));      \
            if (err != cudaSuccess) goto fail;                                \
        } while (0)

    NV_ALLOC(st->c_scale, toks * st->gc);
    NV_ALLOC(st->c_byte, toks * spec->d_c);
    NV_ALLOC(st->r_scale, toks * st->gr);
    NV_ALLOC(st->r_byte, toks * spec->d_r);
    NV_ALLOC(st->seq_len, sizeof(int32_t) * B);
    NV_ALLOC(st->counters, sizeof(int32_t) * 4);
    NV_ALLOC(st->W_uk, sizeof(float) * (size_t)spec->Hq * spec->d_h * spec->d_c);
    NV_ALLOC(st->W_uv, sizeof(float) * (size_t)spec->Hq * spec->d_v * spec->d_c);
    NV_ALLOC(st->rope_cos, sizeof(float) * (size_t)msl * st->half_r);
    NV_ALLOC(st->rope_sin, sizeof(float) * (size_t)msl * st->half_r);
    #undef NV_ALLOC

    err = cudaMemcpyAsync(
        st->W_uk, init_inputs->W_uk,
        sizeof(float) * (size_t)spec->Hq * spec->d_h * spec->d_c,
        cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) goto fail;
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

    mla_nv_reset_kernel<<<1, 64, 0, stream>>>(st->seq_len, st->counters, B);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->c_scale) cudaFree(st->c_scale);
    if (st->c_byte) cudaFree(st->c_byte);
    if (st->r_scale) cudaFree(st->r_scale);
    if (st->r_byte) cudaFree(st->r_byte);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->counters) cudaFree(st->counters);
    if (st->W_uk) cudaFree(st->W_uk);
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
    MlaNaiveState* st = static_cast<MlaNaiveState*>(state);
    if (!st || !run || !inputs || !outputs) return cudaErrorInvalidValue;
    if (!mla_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const MlaInputs* in = static_cast<const MlaInputs*>(inputs);
    MlaOutputs* out = static_cast<MlaOutputs*>(outputs);
    const MlaProblemSpec& sp = st->spec;
    const int A = run->active_count;

    MlaNaiveWs ws = mla_nv_layout(workspace, &sp);
    if (workspace_bytes < ws.required_bytes) return cudaErrorInvalidValue;

    cudaError_t err;

    if (A > 0) {
        const size_t y_floats = (size_t)A * MLA_MAX_NEW_TOKENS * sp.Hq * sp.d_v;
        const size_t lse_floats = (size_t)A * MLA_MAX_NEW_TOKENS * sp.Hq;
        err = cudaMemsetAsync(out->y, 0, y_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;
        err = cudaMemsetAsync(out->lse, 0, lse_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;

        mla_nv_quantize_kernel<<<A, MLA_MAX_NEW_TOKENS, 0, stream>>>(
            in->active_seq, in->new_token_count, in->new_c, in->new_r,
            st->seq_len,
            st->c_scale, st->c_byte, st->r_scale, st->r_byte,
            st->counters + 1,
            sp.d_c, sp.d_r, st->gc, st->gr, sp.max_seq_len);
    }

    mla_nv_admin_kernel<<<1, 1, 0, stream>>>(
        in->active_seq, in->new_token_count, st->seq_len, st->counters,
        out->seq_len, out->meta_checksum, out->sat_count, out->total_tokens,
        A, sp.B, sp.Hq, sp.d_c, sp.d_r, sp.d_h, sp.d_v, sp.max_seq_len);

    mla_nv_hash_kernel<<<(sp.B + 31) / 32, 32, 0, stream>>>(
        st->seq_len,
        st->c_scale, st->c_byte, st->r_scale, st->r_byte,
        out->cache_hash,
        sp.B, sp.d_c, sp.d_r, st->gc, st->gr, sp.max_seq_len);

    if (A > 0) {
        const int nq = A * MLA_MAX_NEW_TOKENS * sp.Hq;
        mla_nv_attention_kernel<<<(nq + 31) / 32, 32, 0, stream>>>(
            in->active_seq, in->new_token_count, in->q, in->q_rope,
            st->seq_len,
            st->c_scale, st->c_byte, st->r_scale, st->r_byte,
            st->rope_cos, st->rope_sin,
            st->W_uk, st->W_uv,
            ws.ql, ws.z,
            out->y, out->lse,
            sp.Hq, sp.d_c, sp.d_r, sp.d_h, sp.d_v,
            st->gc, st->gr, st->half_r, sp.max_seq_len, nq);
    }

    return cudaGetLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    MlaNaiveState* st = static_cast<MlaNaiveState*>(state);
    if (!st) return cudaErrorInvalidValue;
    mla_nv_reset_kernel<<<1, 64, 0, stream>>>(
        st->seq_len, st->counters, st->spec.B);
    return cudaGetLastError();
}

extern "C" void solution_destroy(void* state) {
    MlaNaiveState* st = static_cast<MlaNaiveState*>(state);
    if (!st) return;
    if (st->c_scale) cudaFree(st->c_scale);
    if (st->c_byte) cudaFree(st->c_byte);
    if (st->r_scale) cudaFree(st->r_scale);
    if (st->r_byte) cudaFree(st->r_byte);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->counters) cudaFree(st->counters);
    if (st->W_uk) cudaFree(st->W_uk);
    if (st->W_uv) cudaFree(st->W_uv);
    if (st->rope_cos) cudaFree(st->rope_cos);
    if (st->rope_sin) cudaFree(st->rope_sin);
    free(st);
}
