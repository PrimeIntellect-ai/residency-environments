// ============================================================================
// file: naive_ref.cu  (authoring artifact, NOT shipped to solvers)
//
// Clean, obviously-correct but unoptimized implementation of the
// paged_cow_prefill_decode contract:
//   - single-thread admin (releases, forks, appends planning) with linear
//     refcount scans and a single-thread tail copy,
//   - one scalar thread per appended token for the fp16 conversion,
//   - one scalar thread per (row, token, head) two-pass attention re-reading
//     the paged cache,
//   - full from-scratch kv_hash refold per sequence every step.
// Used to measure the naive/reference performance ratio.
// ============================================================================

#include "paged_cow_prefill_decode_common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t kNvFnvBasis = 1469598103934665603ULL;

struct CpdNaiveState {
    CpdProblemSpec spec;
    int max_lp;

    uint16_t* k_pool;
    uint16_t* v_pool;
    int32_t* page_table;
    int32_t* refcount;
    int32_t* seq_len;
    int32_t* counters;  // [0..4]
};

struct CpdNaiveWs {
    int32_t* base_len;      // [B]
    int32_t* append_page;   // [B * C]
    int32_t* append_slot;   // [B * C]
    size_t required_bytes;
};

static CpdNaiveWs cpd_nv_layout(void* ws, const CpdProblemSpec* spec) {
    CpdNaiveWs lay{};
    char* base = static_cast<char*>(ws);
    const int B = spec->B;
    const int C = spec->max_chunk;
    size_t off = 0;

    #define NV_WS(field, type, count)                                         \
        off = cpd_align_up_size(off, 128);                                    \
        lay.field = reinterpret_cast<type*>(base + off);                      \
        off += sizeof(type) * (count)

    NV_WS(base_len, int32_t, (size_t)B);
    NV_WS(append_page, int32_t, (size_t)B * C);
    NV_WS(append_slot, int32_t, (size_t)B * C);
    #undef NV_WS

    off = cpd_align_up_size(off, 128);
    lay.required_bytes = off;
    return lay;
}

__device__ __forceinline__ uint64_t nv_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ __forceinline__ uint64_t nv_fnv_u16(uint64_t h, uint16_t v) {
    h = nv_fnv_byte(h, (uint8_t)(v & 0xFF));
    h = nv_fnv_byte(h, (uint8_t)((v >> 8) & 0xFF));
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

__device__ __forceinline__ size_t nv_slot_head(
    int page, int slot, int kvh, int Hkv, int P) {
    return ((size_t)page * Hkv + (size_t)kvh) * (size_t)P + (size_t)slot;
}

// Single thread: releases, forks (incl. tail copy), append planning.
__global__ void nv_admin_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ op_kind,
    const int32_t* __restrict__ fork_src,
    const int32_t* __restrict__ new_token_count,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ refcount,
    int32_t* __restrict__ seq_len,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ base_len,
    int32_t* __restrict__ append_page,
    int32_t* __restrict__ append_slot,
    uint16_t* __restrict__ k_pool,
    uint16_t* __restrict__ v_pool,
    int active_count,
    int Hkv,
    int D,
    int P,
    int C,
    int max_lp,
    int max_pages) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    counters[0] += 1;

    for (int a = 0; a < active_count; ++a) {
        if (op_kind[a] != CPD_OP_RELEASE) continue;
        const int b = active_seq[a];
        const int n_lp = (seq_len[b] + P - 1) / P;
        for (int lp = 0; lp < n_lp; ++lp) {
            const int p = page_table[(size_t)b * max_lp + lp];
            page_table[(size_t)b * max_lp + lp] = -1;
            refcount[p] -= 1;
            if (refcount[p] == 0) counters[2] += 1;
        }
        seq_len[b] = 0;
        counters[4] += 1;
    }

    for (int a = 0; a < active_count; ++a) {
        if (op_kind[a] != CPD_OP_FORK_APPEND) continue;
        const int b = active_seq[a];
        const int s = fork_src[a];
        seq_len[b] = seq_len[s];
        counters[3] += 1;
        const int full_lp = seq_len[s] / P;
        for (int lp = 0; lp < full_lp; ++lp) {
            const int p = page_table[(size_t)s * max_lp + lp];
            page_table[(size_t)b * max_lp + lp] = p;
            refcount[p] += 1;
        }
        const int rem = seq_len[s] % P;
        if (rem != 0) {
            int page = -1;
            for (int p = 0; p < max_pages; ++p) {
                if (refcount[p] == 0) { page = p; break; }
            }
            refcount[page] = 1;
            counters[1] += 1;
            page_table[(size_t)b * max_lp + full_lp] = page;
            const int src_page = page_table[(size_t)s * max_lp + full_lp];
            for (int hh = 0; hh < Hkv; ++hh) {
                for (int slot = 0; slot < rem; ++slot) {
                    const size_t si = nv_slot_head(src_page, slot, hh, Hkv, P) * D;
                    const size_t di = nv_slot_head(page, slot, hh, Hkv, P) * D;
                    for (int d = 0; d < D; ++d) {
                        k_pool[di + d] = k_pool[si + d];
                        v_pool[di + d] = v_pool[si + d];
                    }
                }
            }
        }
    }

    for (int a = 0; a < active_count; ++a) {
        if (op_kind[a] == CPD_OP_RELEASE) continue;
        const int b = active_seq[a];
        const int cnt = new_token_count[a];
        base_len[a] = seq_len[b];
        for (int nt = 0; nt < cnt; ++nt) {
            const int pos = seq_len[b] + nt;
            const int lp = pos / P;
            if (page_table[(size_t)b * max_lp + lp] < 0) {
                int page = -1;
                for (int p = 0; p < max_pages; ++p) {
                    if (refcount[p] == 0) { page = p; break; }
                }
                refcount[page] = 1;
                counters[1] += 1;
                page_table[(size_t)b * max_lp + lp] = page;
            }
            append_page[(size_t)a * C + nt] = page_table[(size_t)b * max_lp + lp];
            append_slot[(size_t)a * C + nt] = pos % P;
        }
        seq_len[b] += cnt;
    }
}

// One scalar thread per (a, nt): fp16 conversion.
__global__ void nv_append_kernel(
    const int32_t* __restrict__ op_kind,
    const int32_t* __restrict__ new_token_count,
    const float* __restrict__ new_k,
    const float* __restrict__ new_v,
    const int32_t* __restrict__ append_page,
    const int32_t* __restrict__ append_slot,
    uint16_t* __restrict__ k_pool,
    uint16_t* __restrict__ v_pool,
    int C,
    int Hkv,
    int D,
    int P) {
    const int a = blockIdx.x;
    const int nt = threadIdx.x;
    if (nt >= C) return;
    if (op_kind[a] == CPD_OP_RELEASE) return;
    if (nt >= new_token_count[a]) return;

    const int page = append_page[(size_t)a * C + nt];
    const int slot = append_slot[(size_t)a * C + nt];
    const float* kk = new_k + (((size_t)a * C + nt) * Hkv) * (size_t)D;
    const float* vv = new_v + (((size_t)a * C + nt) * Hkv) * (size_t)D;

    for (int hh = 0; hh < Hkv; ++hh) {
        const size_t hs = nv_slot_head(page, slot, hh, Hkv, P) * D;
        for (int d = 0; d < D; ++d) {
            k_pool[hs + d] = __half_as_ushort(__float2half_rn(kk[(size_t)hh * D + d]));
            v_pool[hs + d] = __half_as_ushort(__float2half_rn(vv[(size_t)hh * D + d]));
        }
    }
}

// One thread per sequence: full refold.
__global__ void nv_hash_kernel(
    const int32_t* __restrict__ seq_len,
    const int32_t* __restrict__ page_table,
    const uint16_t* __restrict__ k_pool,
    const uint16_t* __restrict__ v_pool,
    uint64_t* __restrict__ out_kv_hash,
    int B,
    int Hkv,
    int D,
    int P,
    int max_lp) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    uint64_t h = kNvFnvBasis;
    const int L = seq_len[b];
    for (int t = 0; t < L; ++t) {
        const int page = page_table[(size_t)b * max_lp + t / P];
        for (int hh = 0; hh < Hkv; ++hh) {
            const size_t hs = nv_slot_head(page, t % P, hh, Hkv, P) * D;
            for (int d = 0; d < D; ++d) h = nv_fnv_u16(h, k_pool[hs + d]);
            for (int d = 0; d < D; ++d) h = nv_fnv_u16(h, v_pool[hs + d]);
        }
    }
    h = nv_fnv_i32(h, L);
    out_kv_hash[b] = h;
}

// One scalar thread per (a, nt, hq): two-pass attention.
__global__ void nv_attention_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ op_kind,
    const int32_t* __restrict__ new_token_count,
    const float* __restrict__ q,
    const int32_t* __restrict__ base_len,
    const int32_t* __restrict__ page_table,
    const uint16_t* __restrict__ k_pool,
    const uint16_t* __restrict__ v_pool,
    float* __restrict__ y,
    float* __restrict__ lse,
    int C,
    int Hq,
    int Hkv,
    int D,
    int P,
    int max_lp,
    int nq) {
    const int qidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qidx >= nq) return;
    const int hq = qidx % Hq;
    const int slot_id = qidx / Hq;
    const int a = slot_id / C;
    const int nt = slot_id - a * C;
    if (op_kind[a] == CPD_OP_RELEASE) return;
    if (nt >= new_token_count[a]) return;

    const int b = active_seq[a];
    const int L = base_len[a] + nt + 1;
    const int kvh = hq / (Hq / Hkv);
    const float sm_scale = rsqrtf((float)D);
    const float* qv = q + (((size_t)a * C + nt) * Hq + hq) * (size_t)D;

    float m = -CUDART_INF_F;
    for (int t = 0; t < L; ++t) {
        const int page = page_table[(size_t)b * max_lp + t / P];
        const size_t hs = nv_slot_head(page, t % P, kvh, Hkv, P) * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) {
            s += qv[d] * __half2float(__ushort_as_half(k_pool[hs + d]));
        }
        s *= sm_scale;
        if (s > m) m = s;
    }

    float l = 0.0f;
    for (int d = 0; d < D; ++d) y[(size_t)qidx * D + d] = 0.0f;
    for (int t = 0; t < L; ++t) {
        const int page = page_table[(size_t)b * max_lp + t / P];
        const size_t hs = nv_slot_head(page, t % P, kvh, Hkv, P) * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) {
            s += qv[d] * __half2float(__ushort_as_half(k_pool[hs + d]));
        }
        s *= sm_scale;
        const float w = expf(s - m);
        l += w;
        for (int d = 0; d < D; ++d) {
            y[(size_t)qidx * D + d] +=
                w * __half2float(__ushort_as_half(v_pool[hs + d]));
        }
    }
    const float inv_l = 1.0f / l;
    for (int d = 0; d < D; ++d) y[(size_t)qidx * D + d] *= inv_l;
    lse[qidx] = m + logf(l);
}

// Single thread: exact outputs.
__global__ void nv_final_kernel(
    const int32_t* __restrict__ seq_len,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ refcount,
    const int32_t* __restrict__ counters,
    int32_t* __restrict__ out_seq_len,
    uint64_t* __restrict__ out_pcs,
    int32_t* __restrict__ out_free_pages,
    int32_t* __restrict__ out_total_allocs,
    int32_t* __restrict__ out_total_frees,
    int32_t* __restrict__ out_total_forks,
    int32_t* __restrict__ out_total_releases,
    int B,
    int Hq,
    int Hkv,
    int D,
    int P,
    int C,
    int msl,
    int max_pages,
    int max_lp) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int b = 0; b < B; ++b) out_seq_len[b] = seq_len[b];

    uint64_t ph = kNvFnvBasis;
    ph = nv_fnv_i32(ph, B);
    ph = nv_fnv_i32(ph, Hq);
    ph = nv_fnv_i32(ph, Hkv);
    ph = nv_fnv_i32(ph, D);
    ph = nv_fnv_i32(ph, P);
    ph = nv_fnv_i32(ph, C);
    ph = nv_fnv_i32(ph, msl);
    ph = nv_fnv_i32(ph, max_pages);
    ph = nv_fnv_i32(ph, counters[0]);
    for (int b = 0; b < B; ++b) {
        ph = nv_fnv_i32(ph, seq_len[b]);
        const int n_lp = (seq_len[b] + P - 1) / P;
        ph = nv_fnv_i32(ph, n_lp);
        for (int lp = 0; lp < n_lp; ++lp) {
            ph = nv_fnv_i32(ph, page_table[(size_t)b * max_lp + lp]);
        }
    }
    int freep = 0;
    for (int p = 0; p < max_pages; ++p) freep += refcount[p] == 0 ? 1 : 0;
    ph = nv_fnv_i32(ph, counters[1]);
    ph = nv_fnv_i32(ph, counters[2]);
    ph = nv_fnv_i32(ph, counters[3]);
    ph = nv_fnv_i32(ph, counters[4]);
    ph = nv_fnv_i32(ph, freep);
    out_pcs[0] = ph;
    out_free_pages[0] = freep;
    out_total_allocs[0] = counters[1];
    out_total_frees[0] = counters[2];
    out_total_forks[0] = counters[3];
    out_total_releases[0] = counters[4];
}

__global__ void nv_reset_kernel(
    int32_t* __restrict__ seq_len,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ refcount,
    int32_t* __restrict__ counters,
    int B,
    int max_lp,
    int max_pages) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    if (i < B) seq_len[i] = 0;
    for (int j = i; j < B * max_lp; j += stride) page_table[j] = -1;
    for (int p = i; p < max_pages; p += stride) refcount[p] = 0;
    if (i < 5) counters[i] = 0;
}

extern "C" size_t solution_workspace_bytes(const CpdProblemSpec* spec) {
    if (!cpd_validate_problem_spec(spec)) return 0;
    return cpd_nv_layout(nullptr, spec).required_bytes;
}

extern "C" cudaError_t solution_init(
    const CpdProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!state_out) return cudaErrorInvalidValue;
    *state_out = nullptr;
    if (!cpd_validate_problem_spec(spec)) return cudaErrorInvalidValue;

    CpdNaiveState* st =
        static_cast<CpdNaiveState*>(malloc(sizeof(CpdNaiveState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(*st));
    st->spec = *spec;
    st->max_lp = cpd_max_logical_pages(spec->max_seq_len, spec->page_size);

    const size_t elems =
        (size_t)spec->max_pages * spec->Hkv * spec->page_size * spec->D;

    cudaError_t err = cudaSuccess;
    #define NV_ALLOC(ptr, bytes)                                              \
        do {                                                                  \
            err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes));      \
            if (err != cudaSuccess) goto fail;                                \
        } while (0)

    NV_ALLOC(st->k_pool, elems * sizeof(uint16_t));
    NV_ALLOC(st->v_pool, elems * sizeof(uint16_t));
    NV_ALLOC(st->page_table, sizeof(int32_t) * (size_t)spec->B * st->max_lp);
    NV_ALLOC(st->refcount, sizeof(int32_t) * spec->max_pages);
    NV_ALLOC(st->seq_len, sizeof(int32_t) * spec->B);
    NV_ALLOC(st->counters, sizeof(int32_t) * 8);
    #undef NV_ALLOC

    nv_reset_kernel<<<8, 128, 0, stream>>>(
        st->seq_len, st->page_table, st->refcount, st->counters,
        spec->B, st->max_lp, spec->max_pages);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->k_pool) cudaFree(st->k_pool);
    if (st->v_pool) cudaFree(st->v_pool);
    if (st->page_table) cudaFree(st->page_table);
    if (st->refcount) cudaFree(st->refcount);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->counters) cudaFree(st->counters);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const CpdRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    CpdNaiveState* st = static_cast<CpdNaiveState*>(state);
    if (!st || !run || !inputs || !outputs) return cudaErrorInvalidValue;
    if (!cpd_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const CpdInputs* in = static_cast<const CpdInputs*>(inputs);
    CpdOutputs* out = static_cast<CpdOutputs*>(outputs);
    const CpdProblemSpec& sp = st->spec;
    const int A = run->active_count;
    const int C = sp.max_chunk;

    CpdNaiveWs ws = cpd_nv_layout(workspace, &sp);
    if (workspace_bytes < ws.required_bytes) return cudaErrorInvalidValue;

    cudaError_t err;

    nv_admin_kernel<<<1, 1, 0, stream>>>(
        in->active_seq, in->op_kind, in->fork_src, in->new_token_count,
        st->page_table, st->refcount, st->seq_len, st->counters,
        ws.base_len, ws.append_page, ws.append_slot,
        st->k_pool, st->v_pool,
        A, sp.Hkv, sp.D, sp.page_size, C, st->max_lp, sp.max_pages);

    if (A > 0) {
        const size_t y_floats = (size_t)A * C * sp.Hq * sp.D;
        const size_t lse_floats = (size_t)A * C * sp.Hq;
        err = cudaMemsetAsync(out->y, 0, y_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;
        err = cudaMemsetAsync(out->lse, 0, lse_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;

        nv_append_kernel<<<A, C, 0, stream>>>(
            in->op_kind, in->new_token_count, in->new_k, in->new_v,
            ws.append_page, ws.append_slot,
            st->k_pool, st->v_pool,
            C, sp.Hkv, sp.D, sp.page_size);

        const int nq = A * C * sp.Hq;
        nv_attention_kernel<<<(nq + 31) / 32, 32, 0, stream>>>(
            in->active_seq, in->op_kind, in->new_token_count, in->q,
            ws.base_len, st->page_table,
            st->k_pool, st->v_pool,
            out->y, out->lse,
            C, sp.Hq, sp.Hkv, sp.D, sp.page_size, st->max_lp, nq);
    }

    nv_hash_kernel<<<(sp.B + 31) / 32, 32, 0, stream>>>(
        st->seq_len, st->page_table,
        st->k_pool, st->v_pool,
        out->kv_hash,
        sp.B, sp.Hkv, sp.D, sp.page_size, st->max_lp);

    nv_final_kernel<<<1, 1, 0, stream>>>(
        st->seq_len, st->page_table, st->refcount, st->counters,
        out->seq_len, out->page_state_checksum,
        out->free_pages, out->total_allocs, out->total_frees,
        out->total_forks, out->total_releases,
        sp.B, sp.Hq, sp.Hkv, sp.D, sp.page_size, C, sp.max_seq_len,
        sp.max_pages, st->max_lp);

    return cudaGetLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    CpdNaiveState* st = static_cast<CpdNaiveState*>(state);
    if (!st) return cudaErrorInvalidValue;
    nv_reset_kernel<<<8, 128, 0, stream>>>(
        st->seq_len, st->page_table, st->refcount, st->counters,
        st->spec.B, st->max_lp, st->spec.max_pages);
    return cudaGetLastError();
}

extern "C" void solution_destroy(void* state) {
    CpdNaiveState* st = static_cast<CpdNaiveState*>(state);
    if (!st) return;
    if (st->k_pool) cudaFree(st->k_pool);
    if (st->v_pool) cudaFree(st->v_pool);
    if (st->page_table) cudaFree(st->page_table);
    if (st->refcount) cudaFree(st->refcount);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->counters) cudaFree(st->counters);
    free(st);
}
