// ============================================================================
// file: naive_ref.cu  (authoring artifact, NOT shipped to solvers)
//
// Clean, obviously-correct but unoptimized implementation of the
// paged_sink_e4m3_decode contract:
//   - single-thread admin kernel with linear allocator scans,
//   - per-(row,token) quantize blocks,
//   - per-(row,head) two-pass softmax attention re-reading K,
//   - one thread per sequence folding every live byte serially for kv_hash,
//   - single-thread global checksum fold.
// Used to measure the naive/reference performance ratio.
// ============================================================================

#include "paged_sink_e4m3_decode_common.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct PseNaiveState {
    PseProblemSpec spec;
    int max_lp;

    uint8_t* k_pool;   // [(phys*Hkv + h)*P + off]*D + d
    uint8_t* v_pool;
    int8_t* k_exp;     // [(phys*Hkv + h)*P + off]
    int8_t* v_exp;

    int32_t* page_table;  // [B * max_lp]
    int32_t* seq_len;     // [B]
    uint8_t* page_used;   // [max_pages]
    int32_t* counters;    // [0]=step [1]=allocs [2]=frees
};

struct PseNaiveLayout {
    int32_t* dest_slot;  // [B * 8]
    size_t required_bytes;
};

static PseNaiveLayout pse_naive_layout(void* workspace, const PseProblemSpec* spec) {
    PseNaiveLayout lay{};
    char* base = static_cast<char*>(workspace);
    size_t off = 0;
    off = pse_align_up_size(off, 128);
    lay.dest_slot = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)spec->B * PSE_MAX_NEW_TOKENS;
    off = pse_align_up_size(off, 128);
    lay.required_bytes = off;
    return lay;
}

__device__ __forceinline__ uint64_t pse_nv_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ __forceinline__ float pse_nv_pow2f(int e) {
    return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

__device__ __forceinline__ float pse_nv_decode(uint8_t b) {
    const int s = (b >> 7) & 1;
    const int E = (b >> 3) & 0xF;
    const int M = b & 0x7;
    float v;
    if (E == 0) {
        v = static_cast<float>(M) * 0.001953125f;
    } else {
        v = (1.0f + static_cast<float>(M) * 0.125f) * pse_nv_pow2f(E - 7);
    }
    return s ? (v == 0.0f ? -0.0f : -v) : v;
}

__device__ uint8_t pse_nv_encode(float z) {
    const uint32_t bits = __float_as_uint(z);
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const float a = fabsf(z);
    if (a == 0.0f) return sign;
    if (a > 448.0f) return sign | 0x7E;

    const int e = static_cast<int>((bits >> 23) & 0xFF) - 127;
    if (e >= -6) {
        const uint32_t mant = bits & 0x7FFFFFu;
        uint32_t keep = mant >> 20;
        const uint32_t rest = mant & 0xFFFFFu;
        const uint32_t half = 0x80000u;
        int ee = e;
        if (rest > half || (rest == half && (keep & 1u))) keep++;
        if (keep == 8u) { keep = 0u; ee++; }
        if (ee > 8 || (ee == 8 && keep == 7u)) return sign | 0x7E;
        return sign | static_cast<uint8_t>((ee + 7) << 3) | static_cast<uint8_t>(keep);
    }
    const float t = a * 512.0f;
    int i = static_cast<int>(t);
    const float f = t - static_cast<float>(i);
    if (f > 0.5f || (f == 0.5f && (i & 1))) i++;
    if (i >= 8) return sign | 0x08;
    return sign | static_cast<uint8_t>(i);
}

__device__ __forceinline__ size_t pse_nv_head_index(
    int phys, int kvh, int off, int Hkv, int P) {
    return ((size_t)phys * Hkv + (size_t)kvh) * (size_t)P + (size_t)off;
}

// ---------------------------------------------------------------------------

__global__ void pse_nv_admin_kernel(
    int B, int P, int S, int W, int max_lp, int max_pages,
    int active_count,
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ new_token_count,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ seq_len,
    uint8_t* __restrict__ page_used,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ dest_slot,
    int32_t* __restrict__ out_seq_len,
    int32_t* __restrict__ out_free_pages,
    int32_t* __restrict__ out_total_allocs,
    int32_t* __restrict__ out_total_frees) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    counters[0] += 1;

    for (int i = 0; i < B * PSE_MAX_NEW_TOKENS; ++i) {
        dest_slot[i] = -1;
    }

    for (int a = 0; a < active_count; ++a) {
        const int seq = active_seq[a];
        int cnt = new_token_count[a];
        if (cnt < 0) cnt = 0;
        if (cnt > PSE_MAX_NEW_TOKENS) cnt = PSE_MAX_NEW_TOKENS;
        for (int nt = 0; nt < cnt; ++nt) {
            const int pos = seq_len[seq];
            const int lp = pos / P;
            const int off = pos - lp * P;
            int entry = page_table[(size_t)seq * max_lp + lp];
            if (entry < 0) {
                for (int p = 0; p < max_pages; ++p) {
                    if (!page_used[p]) {
                        entry = p;
                        break;
                    }
                }
                page_used[entry] = 1;
                page_table[(size_t)seq * max_lp + lp] = entry;
                counters[1] += 1;
            }
            dest_slot[a * PSE_MAX_NEW_TOKENS + nt] = entry * P + off;
            seq_len[seq] = pos + 1;
        }
    }

    for (int b = 0; b < B; ++b) {
        const int L = seq_len[b];
        const int se = min(S, L);
        const int ws = max(se, L - W);
        const int n_lp = (L + P - 1) / P;
        for (int lp = 0; lp < n_lp; ++lp) {
            const int entry = page_table[(size_t)b * max_lp + lp];
            if (entry < 0) continue;
            if (lp * P >= se && lp * P + P <= ws) {
                page_used[entry] = 0;
                page_table[(size_t)b * max_lp + lp] = -1;
                counters[2] += 1;
            }
        }
    }

    int free_cnt = 0;
    for (int p = 0; p < max_pages; ++p) {
        if (!page_used[p]) ++free_cnt;
    }
    out_free_pages[0] = free_cnt;
    out_total_allocs[0] = counters[1];
    out_total_frees[0] = counters[2];
    for (int b = 0; b < B; ++b) {
        out_seq_len[b] = seq_len[b];
    }
}

__global__ void pse_nv_quantize_kernel(
    int Hkv, int P, int D,
    const int32_t* __restrict__ dest_slot,
    const float* __restrict__ new_k,
    const float* __restrict__ new_v,
    uint8_t* __restrict__ k_pool,
    uint8_t* __restrict__ v_pool,
    int8_t* __restrict__ k_exp,
    int8_t* __restrict__ v_exp) {
    const int slot = dest_slot[blockIdx.x];
    if (slot < 0) return;
    if (threadIdx.x != 0) return;

    const int phys = slot / P;
    const int off = slot - phys * P;

    for (int h = 0; h < Hkv; ++h) {
        const size_t in_base = ((size_t)blockIdx.x * Hkv + (size_t)h) * (size_t)D;
        const size_t hb = pse_nv_head_index(phys, h, off, Hkv, P);

        for (int which = 0; which < 2; ++which) {
            const float* src = which == 0 ? new_k : new_v;
            uint8_t* pool = which == 0 ? k_pool : v_pool;
            int8_t* exps = which == 0 ? k_exp : v_exp;

            float amax = 0.0f;
            for (int d = 0; d < D; ++d) {
                const float a = fabsf(src[in_base + d]);
                if (a > amax) amax = a;
            }
            int se = 0;
            if (amax != 0.0f) {
                const int kfloor =
                    static_cast<int>((__float_as_uint(amax) >> 23) & 0xFF) - 127;
                se = kfloor - 8;
                if (se < -110) se = -110;
                if (se > 110) se = 110;
            }
            exps[hb] = static_cast<int8_t>(se);
            for (int d = 0; d < D; ++d) {
                const float z = src[in_base + d] * pse_nv_pow2f(-se);
                pool[hb * D + d] = pse_nv_encode(z);
            }
        }
    }
}

// two-pass softmax, one block per (row, query head)
__global__ void pse_nv_attention_kernel(
    int Hq, int Hkv, int P, int S, int W, int D, int max_lp,
    const int32_t* __restrict__ active_seq,
    const float* __restrict__ q,
    const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool,
    const int8_t* __restrict__ k_exp,
    const int8_t* __restrict__ v_exp,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seq_len,
    float* __restrict__ y,
    float* __restrict__ lse) {
    const int a = blockIdx.x;
    const int hq = blockIdx.y;
    const int tid = threadIdx.x;  // blockDim == D

    __shared__ float sh_part[128];
    __shared__ float sh_scalar;

    const int seq = active_seq[a];
    const int L = seq_len[seq];
    const int se = min(S, L);
    const int ws = max(se, L - W);
    const int n_live = se + (L - ws);
    const int group = Hq / Hkv;
    const int kvh = hq / group;

    float* y_out = y + ((size_t)a * Hq + (size_t)hq) * D;

    if (n_live == 0) {
        y_out[tid] = 0.0f;
        if (tid == 0) lse[(size_t)a * Hq + hq] = 0.0f;
        return;
    }

    const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    const float qv = q[((size_t)a * Hq + (size_t)hq) * D + tid];

    // pass 1: max logit
    float m = -3.4028234663852886e38f;
    for (int j = 0; j < n_live; ++j) {
        const int pos = j < se ? j : ws + (j - se);
        const int lp = pos / P;
        const int off = pos - lp * P;
        const int phys = page_table[(size_t)seq * max_lp + lp];
        const size_t hb = pse_nv_head_index(phys, kvh, off, Hkv, P);

        sh_part[tid] = qv * pse_nv_decode(k_pool[hb * D + tid]);
        __syncthreads();
        for (int stride = D / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sh_part[tid] += sh_part[tid + stride];
            __syncthreads();
        }
        const float s = sh_part[0] * pse_nv_pow2f(k_exp[hb]) * inv_sqrt_d;
        if (s > m) m = s;
        __syncthreads();
    }

    // pass 2: weighted accumulate
    float denom = 0.0f;
    float acc = 0.0f;
    for (int j = 0; j < n_live; ++j) {
        const int pos = j < se ? j : ws + (j - se);
        const int lp = pos / P;
        const int off = pos - lp * P;
        const int phys = page_table[(size_t)seq * max_lp + lp];
        const size_t hb = pse_nv_head_index(phys, kvh, off, Hkv, P);

        sh_part[tid] = qv * pse_nv_decode(k_pool[hb * D + tid]);
        __syncthreads();
        for (int stride = D / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sh_part[tid] += sh_part[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float s = sh_part[0] * pse_nv_pow2f(k_exp[hb]) * inv_sqrt_d;
            sh_scalar = expf(s - m);
        }
        __syncthreads();
        const float w = sh_scalar;
        denom += w;
        acc += w * pse_nv_pow2f(v_exp[hb]) * pse_nv_decode(v_pool[hb * D + tid]);
        __syncthreads();
    }

    y_out[tid] = acc / denom;
    if (tid == 0) {
        lse[(size_t)a * Hq + hq] = m + logf(denom);
    }
}

// one thread per sequence, serial fold of every live byte
__global__ void pse_nv_kv_hash_kernel(
    int B, int Hkv, int P, int S, int W, int D, int max_lp,
    const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool,
    const int8_t* __restrict__ k_exp,
    const int8_t* __restrict__ v_exp,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seq_len,
    uint64_t* __restrict__ kv_hash) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const int L = seq_len[b];
    const int se = min(S, L);
    const int ws = max(se, L - W);

    uint64_t h = 1469598103934665603ULL;
    const int32_t pre[3] = {L, se, ws};
    for (int i = 0; i < 3; ++i) {
        const uint32_t v = static_cast<uint32_t>(pre[i]);
        for (int k = 0; k < 4; ++k) {
            h = pse_nv_fnv_byte(h, static_cast<uint8_t>((v >> (8 * k)) & 0xFF));
        }
    }

    const int n_live = se + (L - ws);
    for (int j = 0; j < n_live; ++j) {
        const int pos = j < se ? j : ws + (j - se);
        const int lp = pos / P;
        const int off = pos - lp * P;
        const int phys = page_table[(size_t)b * max_lp + lp];
        for (int hk = 0; hk < Hkv; ++hk) {
            const size_t hb = pse_nv_head_index(phys, hk, off, Hkv, P);
            h = pse_nv_fnv_byte(h, static_cast<uint8_t>(k_exp[hb]));
            for (int d = 0; d < D; ++d) {
                h = pse_nv_fnv_byte(h, k_pool[hb * D + d]);
            }
            h = pse_nv_fnv_byte(h, static_cast<uint8_t>(v_exp[hb]));
            for (int d = 0; d < D; ++d) {
                h = pse_nv_fnv_byte(h, v_pool[hb * D + d]);
            }
        }
    }
    kv_hash[b] = h;
}

__global__ void pse_nv_global_checksum_kernel(
    int B, int Hq, int Hkv, int P, int S, int W, int D,
    int max_pages, int max_seq_len, int max_lp,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seq_len,
    const int32_t* __restrict__ counters,
    const int32_t* __restrict__ out_free_pages,
    uint64_t* __restrict__ page_state_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint64_t h = 1469598103934665603ULL;
    const int32_t header[10] = {
        B, Hq, Hkv, D, P, S, W, max_pages, max_seq_len, counters[0]};
    for (int i = 0; i < 10; ++i) {
        const uint32_t v = static_cast<uint32_t>(header[i]);
        for (int k = 0; k < 4; ++k) {
            h = pse_nv_fnv_byte(h, static_cast<uint8_t>((v >> (8 * k)) & 0xFF));
        }
    }
    for (int b = 0; b < B; ++b) {
        const int L = seq_len[b];
        const int n_lp = (L + P - 1) / P;
        int32_t fields[2] = {L, n_lp};
        for (int i = 0; i < 2; ++i) {
            const uint32_t v = static_cast<uint32_t>(fields[i]);
            for (int k = 0; k < 4; ++k) {
                h = pse_nv_fnv_byte(h, static_cast<uint8_t>((v >> (8 * k)) & 0xFF));
            }
        }
        for (int lp = 0; lp < n_lp; ++lp) {
            const uint32_t v =
                static_cast<uint32_t>(page_table[(size_t)b * max_lp + lp]);
            for (int k = 0; k < 4; ++k) {
                h = pse_nv_fnv_byte(h, static_cast<uint8_t>((v >> (8 * k)) & 0xFF));
            }
        }
    }
    const int32_t tail[3] = {counters[1], counters[2], out_free_pages[0]};
    for (int i = 0; i < 3; ++i) {
        const uint32_t v = static_cast<uint32_t>(tail[i]);
        for (int k = 0; k < 4; ++k) {
            h = pse_nv_fnv_byte(h, static_cast<uint8_t>((v >> (8 * k)) & 0xFF));
        }
    }
    page_state_checksum[0] = h;
}

// ---------------------------------------------------------------------------

static cudaError_t pse_naive_reset(PseNaiveState* st, cudaStream_t stream) {
    const PseProblemSpec& s = st->spec;
    const size_t pool_elems =
        (size_t)s.max_pages * s.Hkv * s.page_size * (size_t)s.D;
    const size_t exp_elems = (size_t)s.max_pages * s.Hkv * s.page_size;
    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(st->k_pool, 0, pool_elems, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->v_pool, 0, pool_elems, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->k_exp, 0, exp_elems, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->v_exp, 0, exp_elems, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->page_table, 0xFF,
                          sizeof(int32_t) * (size_t)s.B * st->max_lp, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->seq_len, 0, sizeof(int32_t) * (size_t)s.B, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->page_used, 0, (size_t)s.max_pages, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->counters, 0, sizeof(int32_t) * 4, stream);
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

extern "C" size_t solution_workspace_bytes(const PseProblemSpec* spec) {
    if (!pse_validate_problem_spec(spec)) return 0;
    PseNaiveLayout lay = pse_naive_layout(nullptr, spec);
    return lay.required_bytes;
}

extern "C" cudaError_t solution_init(
    const PseProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!pse_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    PseNaiveState* st = static_cast<PseNaiveState*>(malloc(sizeof(PseNaiveState)));
    if (!st) {
        return cudaErrorMemoryAllocation;
    }
    memset(st, 0, sizeof(PseNaiveState));
    memcpy(&st->spec, spec, sizeof(PseProblemSpec));
    st->max_lp = pse_max_logical_pages(spec->max_seq_len, spec->page_size);

    const size_t pool_elems =
        (size_t)spec->max_pages * spec->Hkv * spec->page_size * (size_t)spec->D;
    const size_t exp_elems = (size_t)spec->max_pages * spec->Hkv * spec->page_size;
    cudaError_t err = cudaSuccess;

    err = cudaMalloc(reinterpret_cast<void**>(&st->k_pool), pool_elems);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->v_pool), pool_elems);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->k_exp), exp_elems);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->v_exp), exp_elems);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->page_table),
                     sizeof(int32_t) * (size_t)spec->B * st->max_lp);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->seq_len),
                     sizeof(int32_t) * (size_t)spec->B);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->page_used), (size_t)spec->max_pages);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->counters), sizeof(int32_t) * 4);
    if (err != cudaSuccess) goto fail;

    err = pse_naive_reset(st, stream);
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->k_pool) cudaFree(st->k_pool);
    if (st->v_pool) cudaFree(st->v_pool);
    if (st->k_exp) cudaFree(st->k_exp);
    if (st->v_exp) cudaFree(st->v_exp);
    if (st->page_table) cudaFree(st->page_table);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->page_used) cudaFree(st->page_used);
    if (st->counters) cudaFree(st->counters);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const PseRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    PseNaiveState* st = static_cast<PseNaiveState*>(state);
    if (!pse_validate_run_spec(run, &st->spec)) {
        return cudaErrorInvalidValue;
    }

    const PseInputs* in = static_cast<const PseInputs*>(inputs_void);
    PseOutputs* out = static_cast<PseOutputs*>(outputs_void);

    PseNaiveLayout lay = pse_naive_layout(workspace, &st->spec);
    if (workspace_bytes < lay.required_bytes) {
        return cudaErrorInvalidValue;
    }

    const PseProblemSpec& s = st->spec;
    const int A = run->active_count;
    cudaError_t err = cudaSuccess;

    pse_nv_admin_kernel<<<1, 1, 0, stream>>>(
        s.B, s.page_size, s.n_sink, s.window, st->max_lp, s.max_pages,
        A,
        in->active_seq,
        in->new_token_count,
        st->page_table,
        st->seq_len,
        st->page_used,
        st->counters,
        lay.dest_slot,
        out->seq_len,
        out->free_pages,
        out->total_allocs,
        out->total_frees);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    if (A > 0) {
        pse_nv_quantize_kernel<<<A * PSE_MAX_NEW_TOKENS, 32, 0, stream>>>(
            s.Hkv, s.page_size, s.D,
            lay.dest_slot,
            in->new_k,
            in->new_v,
            st->k_pool,
            st->v_pool,
            st->k_exp,
            st->v_exp);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        dim3 attn_grid(A, s.Hq, 1);
        pse_nv_attention_kernel<<<attn_grid, s.D, 0, stream>>>(
            s.Hq, s.Hkv, s.page_size, s.n_sink, s.window, s.D, st->max_lp,
            in->active_seq,
            in->q,
            st->k_pool,
            st->v_pool,
            st->k_exp,
            st->v_exp,
            st->page_table,
            st->seq_len,
            out->y,
            out->lse);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    pse_nv_kv_hash_kernel<<<pse_ceil_div_int(s.B, 64), 64, 0, stream>>>(
        s.B, s.Hkv, s.page_size, s.n_sink, s.window, s.D, st->max_lp,
        st->k_pool,
        st->v_pool,
        st->k_exp,
        st->v_exp,
        st->page_table,
        st->seq_len,
        out->kv_hash);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    pse_nv_global_checksum_kernel<<<1, 1, 0, stream>>>(
        s.B, s.Hq, s.Hkv, s.page_size, s.n_sink, s.window, s.D,
        s.max_pages, s.max_seq_len, st->max_lp,
        st->page_table,
        st->seq_len,
        st->counters,
        out->free_pages,
        out->page_state_checksum);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return pse_naive_reset(static_cast<PseNaiveState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    PseNaiveState* st = static_cast<PseNaiveState*>(state);
    if (st->k_pool) cudaFree(st->k_pool);
    if (st->v_pool) cudaFree(st->v_pool);
    if (st->k_exp) cudaFree(st->k_exp);
    if (st->v_exp) cudaFree(st->v_exp);
    if (st->page_table) cudaFree(st->page_table);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->page_used) cudaFree(st->page_used);
    if (st->counters) cudaFree(st->counters);
    free(st);
}
