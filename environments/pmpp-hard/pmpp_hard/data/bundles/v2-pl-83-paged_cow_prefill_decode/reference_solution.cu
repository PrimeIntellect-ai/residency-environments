// PMPP_CANARY_83_e1b7a52f04 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: paged_cow_prefill_decode_reference.cu
//
// Optimization notes (held out from solvers):
//
//  * Prefill and decode share one attention kernel: grid (appended token,
//    kv-head, split); 4 warps round-robin the query heads of the kv group,
//    each running a per-warp online (m, l, acc[D]) recurrence with fp16
//    loads and shuffle score reductions; per-split partials merged by LSE
//    rescaling. Chunked prefill contributes A*C queries, decode 1 -- the
//    grid covers the mixed batch uniformly with no host-side branching.
//  * All lifecycle mutation (releases -> forks -> append planning) runs in
//    one tiny single-thread admin kernel over a refcount array + free
//    bitmap; the tail copy-on-write and the fp16 conversions are
//    block-parallel kernels driven by the recorded plan.
//  * kv_hash is INCREMENTAL: per-sequence running hashes are extended only
//    by this step's appended halves; forks INHERIT the source's running
//    hash verbatim (the shared prefix and the copied tail are
//    byte-identical); releases reset it. A from-scratch refold is O(L * 2KB)
//    serial per sequence per step.
// ============================================================================

#include "paged_cow_prefill_decode_common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t kFnvBasis = 1469598103934665603ULL;

// ---------------------------------------------------------------------------
// State + workspace
// ---------------------------------------------------------------------------

struct CpdRefState {
    CpdProblemSpec spec;
    int max_lp;
    int n_splits;
    int n_words;

    uint16_t* k_pool;   // [((page*Hkv + h)*P + slot)*D + d] fp16 bits
    uint16_t* v_pool;

    int32_t* page_table;   // [B * max_lp]
    int32_t* refcount;     // [max_pages]
    uint64_t* free_words;  // bit=1 free
    int32_t* seq_len;      // [B]
    uint64_t* h_run;       // [B]
    int32_t* counters;     // [0]=step [1]=allocs [2]=frees [3]=forks [4]=releases
};

static int cpd_ref_splits(const CpdProblemSpec* spec) {
    if (spec->max_seq_len >= 2048) return 4;
    if (spec->max_seq_len >= 512) return 2;
    return 1;
}

struct CpdRefWs {
    int32_t* base_len;       // [B]
    int32_t* append_page;    // [B * C]
    int32_t* append_slot;    // [B * C]
    int32_t* fork_src_page;  // [B]
    int32_t* fork_dst_page;  // [B]
    int32_t* fork_nslots;    // [B]
    float* partials;         // [B*C*Hq * S * (D+2)]
    size_t required_bytes;
};

static CpdRefWs cpd_ref_layout(void* ws, const CpdProblemSpec* spec) {
    CpdRefWs lay{};
    char* base = static_cast<char*>(ws);
    const int B = spec->B;
    const int C = spec->max_chunk;
    const int S = cpd_ref_splits(spec);
    size_t off = 0;

    #define CPD_WS(field, type, count)                                        \
        off = cpd_align_up_size(off, 128);                                    \
        lay.field = reinterpret_cast<type*>(base + off);                      \
        off += sizeof(type) * (count)

    CPD_WS(base_len, int32_t, (size_t)B);
    CPD_WS(append_page, int32_t, (size_t)B * C);
    CPD_WS(append_slot, int32_t, (size_t)B * C);
    CPD_WS(fork_src_page, int32_t, (size_t)B);
    CPD_WS(fork_dst_page, int32_t, (size_t)B);
    CPD_WS(fork_nslots, int32_t, (size_t)B);
    CPD_WS(partials, float,
           (size_t)B * C * spec->Hq * S * (size_t)(spec->D + 2));
    #undef CPD_WS

    off = cpd_align_up_size(off, 128);
    lay.required_bytes = off;
    return lay;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t cpd_fnv_byte_d(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ __forceinline__ uint64_t cpd_fnv_u16_d(uint64_t h, uint16_t v) {
    h = cpd_fnv_byte_d(h, (uint8_t)(v & 0xFF));
    h = cpd_fnv_byte_d(h, (uint8_t)((v >> 8) & 0xFF));
    return h;
}

__device__ __forceinline__ uint64_t cpd_fnv_i32_d(uint64_t h, int32_t v) {
    const uint32_t u = static_cast<uint32_t>(v);
    h = cpd_fnv_byte_d(h, (uint8_t)(u & 0xFF));
    h = cpd_fnv_byte_d(h, (uint8_t)((u >> 8) & 0xFF));
    h = cpd_fnv_byte_d(h, (uint8_t)((u >> 16) & 0xFF));
    h = cpd_fnv_byte_d(h, (uint8_t)((u >> 24) & 0xFF));
    return h;
}

__device__ __forceinline__ size_t cpd_slot_head(
    int page, int slot, int kvh, int Hkv, int P) {
    return ((size_t)page * Hkv + (size_t)kvh) * (size_t)P + (size_t)slot;
}

// ---------------------------------------------------------------------------
// Kernel 1: admin (single thread): releases -> forks -> append planning.
// ---------------------------------------------------------------------------

__global__ void cpd_admin_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ op_kind,
    const int32_t* __restrict__ fork_src,
    const int32_t* __restrict__ new_token_count,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ refcount,
    uint64_t* __restrict__ free_words,
    int32_t* __restrict__ seq_len,
    uint64_t* __restrict__ h_run,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ base_len,
    int32_t* __restrict__ append_page,
    int32_t* __restrict__ append_slot,
    int32_t* __restrict__ fork_src_page,
    int32_t* __restrict__ fork_dst_page,
    int32_t* __restrict__ fork_nslots,
    int active_count,
    int P,
    int C,
    int max_lp,
    int n_words) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    counters[0] += 1;

    // Phase 2: releases.
    for (int a = 0; a < active_count; ++a) {
        if (op_kind[a] != CPD_OP_RELEASE) continue;
        const int b = active_seq[a];
        const int n_lp = (seq_len[b] + P - 1) / P;
        for (int lp = 0; lp < n_lp; ++lp) {
            const int p = page_table[(size_t)b * max_lp + lp];
            page_table[(size_t)b * max_lp + lp] = -1;
            refcount[p] -= 1;
            if (refcount[p] == 0) {
                free_words[p >> 6] |= (1ULL << (p & 63));
                counters[2] += 1;
            }
        }
        seq_len[b] = 0;
        h_run[b] = kFnvBasis;
        counters[4] += 1;
    }

    // Phase 3: forks.
    for (int a = 0; a < active_count; ++a) {
        fork_nslots[a] = 0;
        if (op_kind[a] != CPD_OP_FORK_APPEND) continue;
        const int b = active_seq[a];
        const int s = fork_src[a];

        seq_len[b] = seq_len[s];
        h_run[b] = h_run[s];
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
            for (int w = 0; w < n_words; ++w) {
                const uint64_t word = free_words[w];
                if (word != 0ULL) {
                    const int bit = __ffsll(static_cast<long long>(word)) - 1;
                    page = w * 64 + bit;
                    free_words[w] = word & ~(1ULL << bit);
                    break;
                }
            }
            refcount[page] = 1;
            counters[1] += 1;
            page_table[(size_t)b * max_lp + full_lp] = page;
            fork_src_page[a] = page_table[(size_t)s * max_lp + full_lp];
            fork_dst_page[a] = page;
            fork_nslots[a] = rem;
        }
    }

    // Phase 4 planning: appends.
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
                for (int w = 0; w < n_words; ++w) {
                    const uint64_t word = free_words[w];
                    if (word != 0ULL) {
                        const int bit = __ffsll(static_cast<long long>(word)) - 1;
                        page = w * 64 + bit;
                        free_words[w] = word & ~(1ULL << bit);
                        break;
                    }
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

// ---------------------------------------------------------------------------
// Kernel 2: copy-on-write tail copies. Block per fork row.
// ---------------------------------------------------------------------------

__global__ void cpd_fork_copy_kernel(
    const int32_t* __restrict__ fork_src_page,
    const int32_t* __restrict__ fork_dst_page,
    const int32_t* __restrict__ fork_nslots,
    uint16_t* __restrict__ k_pool,
    uint16_t* __restrict__ v_pool,
    int Hkv,
    int D,
    int P) {
    const int a = blockIdx.x;
    const int nslots = fork_nslots[a];
    if (nslots == 0) return;

    const int src = fork_src_page[a];
    const int dst = fork_dst_page[a];
    const int total = Hkv * nslots * D;

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int d = i % D;
        const int rest = i / D;
        const int slot = rest % nslots;
        const int h = rest / nslots;
        const size_t si = cpd_slot_head(src, slot, h, Hkv, P) * D + d;
        const size_t di = cpd_slot_head(dst, slot, h, Hkv, P) * D + d;
        k_pool[di] = k_pool[si];
        v_pool[di] = v_pool[si];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: fp16 conversion + append. Block per (row, token).
// ---------------------------------------------------------------------------

__global__ void cpd_append_kernel(
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
    const int a = blockIdx.x / C;
    const int nt = blockIdx.x - a * C;
    if (op_kind[a] == CPD_OP_RELEASE) return;
    if (nt >= new_token_count[a]) return;

    const int page = append_page[(size_t)a * C + nt];
    const int slot = append_slot[(size_t)a * C + nt];
    const int total = Hkv * D;

    const float* kk = new_k + (((size_t)a * C + nt) * Hkv) * (size_t)D;
    const float* vv = new_v + (((size_t)a * C + nt) * Hkv) * (size_t)D;

    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int h = i / D;
        const int d = i % D;
        const size_t idx = cpd_slot_head(page, slot, h, Hkv, P) * D + d;
        k_pool[idx] = __half_as_ushort(__float2half_rn(kk[i]));
        v_pool[idx] = __half_as_ushort(__float2half_rn(vv[i]));
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: extend running hashes with this step's appended tokens.
// ---------------------------------------------------------------------------

__global__ void cpd_hash_extend_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ op_kind,
    const int32_t* __restrict__ new_token_count,
    const int32_t* __restrict__ append_page,
    const int32_t* __restrict__ append_slot,
    const uint16_t* __restrict__ k_pool,
    const uint16_t* __restrict__ v_pool,
    uint64_t* __restrict__ h_run,
    int active_count,
    int C,
    int Hkv,
    int D,
    int P) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= active_count) return;
    if (op_kind[a] == CPD_OP_RELEASE) return;

    const int b = active_seq[a];
    const int cnt = new_token_count[a];
    uint64_t h = h_run[b];
    for (int nt = 0; nt < cnt; ++nt) {
        const int page = append_page[(size_t)a * C + nt];
        const int slot = append_slot[(size_t)a * C + nt];
        for (int hh = 0; hh < Hkv; ++hh) {
            const size_t hs = cpd_slot_head(page, slot, hh, Hkv, P);
            for (int d = 0; d < D; ++d) {
                h = cpd_fnv_u16_d(h, k_pool[hs * D + d]);
            }
            for (int d = 0; d < D; ++d) {
                h = cpd_fnv_u16_d(h, v_pool[hs * D + d]);
            }
        }
    }
    h_run[b] = h;
}

// ---------------------------------------------------------------------------
// Kernel 5: attention. Grid ((a*C + nt), kvh, split); 4 warps round-robin
// the query heads of the kv group.
// ---------------------------------------------------------------------------

#define CPD_ATTN_WARPS 4

__global__ void cpd_attention_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ op_kind,
    const int32_t* __restrict__ new_token_count,
    const float* __restrict__ q,
    const int32_t* __restrict__ base_len,
    const int32_t* __restrict__ page_table,
    const uint16_t* __restrict__ k_pool,
    const uint16_t* __restrict__ v_pool,
    float* __restrict__ partials,
    int C,
    int Hq,
    int Hkv,
    int D,
    int P,
    int max_lp,
    int n_splits,
    float sm_scale) {
    const int slot_id = blockIdx.x;   // a*C + nt
    const int a = slot_id / C;
    const int nt = slot_id - a * C;
    if (op_kind[a] == CPD_OP_RELEASE) return;
    if (nt >= new_token_count[a]) return;
    const int kvh = blockIdx.y;
    const int split = blockIdx.z;

    const int b = active_seq[a];
    const int p = base_len[a] + nt;
    const int L = p + 1;
    const int group = Hq / Hkv;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int per_lane = D >> 5;

    const int chunk = (L + n_splits - 1) / n_splits;
    const int t0 = split * chunk;
    const int t1 = (t0 + chunk) < L ? (t0 + chunk) : L;

    for (int hq_off = warp; hq_off < group; hq_off += CPD_ATTN_WARPS) {
        const int hq = kvh * group + hq_off;

        float qv[4];
        {
            const float* qp = q + (((size_t)a * C + nt) * Hq + hq) * (size_t)D;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int d = lane + (r << 5);
                qv[r] = (r < per_lane) ? qp[d] : 0.0f;
            }
        }

        float m = -CUDART_INF_F;
        float l = 0.0f;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int t = t0; t < t1; ++t) {
            const int page = page_table[(size_t)b * max_lp + t / P];
            const int slot = t % P;
            const size_t hs = cpd_slot_head(page, slot, kvh, Hkv, P);

            float part = 0.0f;
            float vvv[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r < per_lane) {
                    const int d = lane + (r << 5);
                    const float kv = __half2float(
                        __ushort_as_half(k_pool[hs * D + d]));
                    vvv[r] = __half2float(
                        __ushort_as_half(v_pool[hs * D + d]));
                    part += qv[r] * kv;
                } else {
                    vvv[r] = 0.0f;
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
            for (int r = 0; r < 4; ++r) {
                acc[r] = acc[r] * alpha + beta * vvv[r];
            }
            l = l * alpha + beta;
            m = nm;
        }

        float* rec = partials +
            (((size_t)slot_id * Hq + hq) * n_splits + split) * (size_t)(D + 2);
        if (lane == 0) {
            rec[0] = m;
            rec[1] = l;
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r < per_lane) {
                rec[2 + lane + (r << 5)] = acc[r];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: merge splits, write y / lse.
// ---------------------------------------------------------------------------

__global__ void cpd_merge_kernel(
    const int32_t* __restrict__ op_kind,
    const int32_t* __restrict__ new_token_count,
    const float* __restrict__ partials,
    float* __restrict__ y,
    float* __restrict__ lse,
    int C,
    int Hq,
    int D,
    int n_splits) {
    const int flat = blockIdx.x;   // (a*C + nt)*Hq + hq
    const int slot_id = flat / Hq;
    const int a = slot_id / C;
    const int nt = slot_id - a * C;
    if (op_kind[a] == CPD_OP_RELEASE) return;
    if (nt >= new_token_count[a]) return;

    __shared__ float sh_M, sh_L;
    __shared__ float sh_f[8];

    const float* base = partials + (size_t)flat * n_splits * (D + 2);

    if (threadIdx.x == 0) {
        float M = -CUDART_INF_F;
        for (int s = 0; s < n_splits; ++s) {
            M = fmaxf(M, base[(size_t)s * (D + 2)]);
        }
        float Lc = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float ms = base[(size_t)s * (D + 2)];
            const float ls = base[(size_t)s * (D + 2) + 1];
            const float f = (ms == -CUDART_INF_F) ? 0.0f : __expf(ms - M);
            sh_f[s] = f;
            Lc += ls * f;
        }
        sh_M = M;
        sh_L = Lc;
        lse[flat] = M + logf(Lc);
    }
    __syncthreads();

    const float inv_l = 1.0f / sh_L;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            v += base[(size_t)s * (D + 2) + 2 + d] * sh_f[s];
        }
        y[(size_t)flat * D + d] = v * inv_l;
    }
}

// ---------------------------------------------------------------------------
// Kernel 7: final exact outputs.
// ---------------------------------------------------------------------------

__global__ void cpd_final_kernel(
    const int32_t* __restrict__ seq_len,
    const uint64_t* __restrict__ h_run,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ refcount,
    const int32_t* __restrict__ counters,
    int32_t* __restrict__ out_seq_len,
    uint64_t* __restrict__ out_kv_hash,
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
    const int tid = threadIdx.x;

    if (tid < B) {
        out_seq_len[tid] = seq_len[tid];
        out_kv_hash[tid] = cpd_fnv_i32_d(h_run[tid], seq_len[tid]);
    }

    if (tid == 0) {
        uint64_t ph = kFnvBasis;
        ph = cpd_fnv_i32_d(ph, B);
        ph = cpd_fnv_i32_d(ph, Hq);
        ph = cpd_fnv_i32_d(ph, Hkv);
        ph = cpd_fnv_i32_d(ph, D);
        ph = cpd_fnv_i32_d(ph, P);
        ph = cpd_fnv_i32_d(ph, C);
        ph = cpd_fnv_i32_d(ph, msl);
        ph = cpd_fnv_i32_d(ph, max_pages);
        ph = cpd_fnv_i32_d(ph, counters[0]);
        for (int b = 0; b < B; ++b) {
            ph = cpd_fnv_i32_d(ph, seq_len[b]);
            const int n_lp = (seq_len[b] + P - 1) / P;
            ph = cpd_fnv_i32_d(ph, n_lp);
            for (int lp = 0; lp < n_lp; ++lp) {
                ph = cpd_fnv_i32_d(ph, page_table[(size_t)b * max_lp + lp]);
            }
        }
        int freep = 0;
        for (int p = 0; p < max_pages; ++p) freep += refcount[p] == 0 ? 1 : 0;
        ph = cpd_fnv_i32_d(ph, counters[1]);
        ph = cpd_fnv_i32_d(ph, counters[2]);
        ph = cpd_fnv_i32_d(ph, counters[3]);
        ph = cpd_fnv_i32_d(ph, counters[4]);
        ph = cpd_fnv_i32_d(ph, freep);
        out_pcs[0] = ph;
        out_free_pages[0] = freep;
        out_total_allocs[0] = counters[1];
        out_total_frees[0] = counters[2];
        out_total_forks[0] = counters[3];
        out_total_releases[0] = counters[4];
    }
}

__global__ void cpd_reset_kernel(
    int32_t* __restrict__ seq_len,
    uint64_t* __restrict__ h_run,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ refcount,
    uint64_t* __restrict__ free_words,
    int32_t* __restrict__ counters,
    int B,
    int max_lp,
    int max_pages,
    int n_words) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    if (i < B) {
        seq_len[i] = 0;
        h_run[i] = kFnvBasis;
    }
    for (int j = i; j < B * max_lp; j += stride) page_table[j] = -1;
    for (int p = i; p < max_pages; p += stride) refcount[p] = 0;
    for (int w = i; w < n_words; w += stride) {
        const int base = w * 64;
        uint64_t word = 0ULL;
        for (int bit = 0; bit < 64; ++bit) {
            if (base + bit < max_pages) word |= (1ULL << bit);
        }
        free_words[w] = word;
    }
    if (i < 5) counters[i] = 0;
}

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------

extern "C" size_t solution_workspace_bytes(const CpdProblemSpec* spec) {
    if (!cpd_validate_problem_spec(spec)) return 0;
    return cpd_ref_layout(nullptr, spec).required_bytes;
}

extern "C" cudaError_t solution_init(
    const CpdProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!state_out) return cudaErrorInvalidValue;
    *state_out = nullptr;
    if (!cpd_validate_problem_spec(spec)) return cudaErrorInvalidValue;

    CpdRefState* st = static_cast<CpdRefState*>(malloc(sizeof(CpdRefState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(*st));
    st->spec = *spec;
    st->max_lp = cpd_max_logical_pages(spec->max_seq_len, spec->page_size);
    st->n_splits = cpd_ref_splits(spec);
    st->n_words = (spec->max_pages + 63) / 64;

    const size_t elems =
        (size_t)spec->max_pages * spec->Hkv * spec->page_size * spec->D;

    cudaError_t err = cudaSuccess;
    #define CPD_ALLOC(ptr, bytes)                                             \
        do {                                                                  \
            err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes));      \
            if (err != cudaSuccess) goto fail;                                \
        } while (0)

    CPD_ALLOC(st->k_pool, elems * sizeof(uint16_t));
    CPD_ALLOC(st->v_pool, elems * sizeof(uint16_t));
    CPD_ALLOC(st->page_table, sizeof(int32_t) * (size_t)spec->B * st->max_lp);
    CPD_ALLOC(st->refcount, sizeof(int32_t) * spec->max_pages);
    CPD_ALLOC(st->free_words, sizeof(uint64_t) * st->n_words);
    CPD_ALLOC(st->seq_len, sizeof(int32_t) * spec->B);
    CPD_ALLOC(st->h_run, sizeof(uint64_t) * spec->B);
    CPD_ALLOC(st->counters, sizeof(int32_t) * 8);
    #undef CPD_ALLOC

    cpd_reset_kernel<<<8, 128, 0, stream>>>(
        st->seq_len, st->h_run, st->page_table, st->refcount, st->free_words,
        st->counters, spec->B, st->max_lp, spec->max_pages, st->n_words);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->k_pool) cudaFree(st->k_pool);
    if (st->v_pool) cudaFree(st->v_pool);
    if (st->page_table) cudaFree(st->page_table);
    if (st->refcount) cudaFree(st->refcount);
    if (st->free_words) cudaFree(st->free_words);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->h_run) cudaFree(st->h_run);
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
    CpdRefState* st = static_cast<CpdRefState*>(state);
    if (!st || !run || !inputs || !outputs) return cudaErrorInvalidValue;
    if (!cpd_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const CpdInputs* in = static_cast<const CpdInputs*>(inputs);
    CpdOutputs* out = static_cast<CpdOutputs*>(outputs);
    const CpdProblemSpec& sp = st->spec;
    const int A = run->active_count;
    const int C = sp.max_chunk;

    CpdRefWs ws = cpd_ref_layout(workspace, &sp);
    if (workspace_bytes < ws.required_bytes) return cudaErrorInvalidValue;

    cudaError_t err;

    cpd_admin_kernel<<<1, 1, 0, stream>>>(
        in->active_seq, in->op_kind, in->fork_src, in->new_token_count,
        st->page_table, st->refcount, st->free_words, st->seq_len, st->h_run,
        st->counters,
        ws.base_len, ws.append_page, ws.append_slot,
        ws.fork_src_page, ws.fork_dst_page, ws.fork_nslots,
        A, sp.page_size, C, st->max_lp, st->n_words);

    if (A > 0) {
        const size_t y_floats = (size_t)A * C * sp.Hq * sp.D;
        const size_t lse_floats = (size_t)A * C * sp.Hq;
        err = cudaMemsetAsync(out->y, 0, y_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;
        err = cudaMemsetAsync(out->lse, 0, lse_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;

        cpd_fork_copy_kernel<<<A, 128, 0, stream>>>(
            ws.fork_src_page, ws.fork_dst_page, ws.fork_nslots,
            st->k_pool, st->v_pool, sp.Hkv, sp.D, sp.page_size);

        cpd_append_kernel<<<A * C, 128, 0, stream>>>(
            in->op_kind, in->new_token_count, in->new_k, in->new_v,
            ws.append_page, ws.append_slot,
            st->k_pool, st->v_pool,
            C, sp.Hkv, sp.D, sp.page_size);

        cpd_hash_extend_kernel<<<1, 32, 0, stream>>>(
            in->active_seq, in->op_kind, in->new_token_count,
            ws.append_page, ws.append_slot,
            st->k_pool, st->v_pool, st->h_run,
            A, C, sp.Hkv, sp.D, sp.page_size);

        {
            dim3 grd(A * C, sp.Hkv, st->n_splits);
            const float sm_scale = rsqrtf((float)sp.D);
            cpd_attention_kernel<<<grd, 32 * CPD_ATTN_WARPS, 0, stream>>>(
                in->active_seq, in->op_kind, in->new_token_count, in->q,
                ws.base_len, st->page_table,
                st->k_pool, st->v_pool,
                ws.partials,
                C, sp.Hq, sp.Hkv, sp.D, sp.page_size, st->max_lp,
                st->n_splits, sm_scale);
        }

        cpd_merge_kernel<<<A * C * sp.Hq, 64, 0, stream>>>(
            in->op_kind, in->new_token_count, ws.partials,
            out->y, out->lse,
            C, sp.Hq, sp.D, st->n_splits);
    }

    cpd_final_kernel<<<1, 64, 0, stream>>>(
        st->seq_len, st->h_run, st->page_table, st->refcount, st->counters,
        out->seq_len, out->kv_hash, out->page_state_checksum,
        out->free_pages, out->total_allocs, out->total_frees,
        out->total_forks, out->total_releases,
        sp.B, sp.Hq, sp.Hkv, sp.D, sp.page_size, C, sp.max_seq_len,
        sp.max_pages, st->max_lp);

    return cudaGetLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    CpdRefState* st = static_cast<CpdRefState*>(state);
    if (!st) return cudaErrorInvalidValue;
    cpd_reset_kernel<<<8, 128, 0, stream>>>(
        st->seq_len, st->h_run, st->page_table, st->refcount, st->free_words,
        st->counters, st->spec.B, st->max_lp, st->spec.max_pages, st->n_words);
    return cudaGetLastError();
}

extern "C" void solution_destroy(void* state) {
    CpdRefState* st = static_cast<CpdRefState*>(state);
    if (!st) return;
    if (st->k_pool) cudaFree(st->k_pool);
    if (st->v_pool) cudaFree(st->v_pool);
    if (st->page_table) cudaFree(st->page_table);
    if (st->refcount) cudaFree(st->refcount);
    if (st->free_words) cudaFree(st->free_words);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->h_run) cudaFree(st->h_run);
    if (st->counters) cudaFree(st->counters);
    free(st);
}
