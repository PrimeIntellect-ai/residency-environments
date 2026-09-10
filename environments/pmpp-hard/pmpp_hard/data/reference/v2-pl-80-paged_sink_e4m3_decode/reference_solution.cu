// PMPP_CANARY_80_7c4e19a2d3 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: paged_sink_e4m3_decode_reference.cu
//
// Optimization notes (held out from solvers):
//
//  * FNV-1a is parallelized exactly with affine chunk maps. Because each byte
//    only XORs into the LOW 8 bits before the multiply, folding a fixed chunk
//    is  FNV(chunk, h) = A * h + C(h & 0xFF)  (mod 2^64) with A = p^len.
//    Per chunk we cache T[v] = FNV(chunk, v) for v in [0,256) and combine
//    with  h' = A * (h & ~0xFF) + T[h & 0xFF].
//  * Chunk = one live token (all Hkv heads: k_exp, K bytes, v_exp, V bytes).
//    Token maps are cached persistently per PHYSICAL slot and recomputed only
//    for slots written this step (dirty tracking): appends are the only way
//    slot bytes change.
//  * Full pages inside the live window additionally get a PAGE map
//    (trajectory continuation of the P token maps), so the serial per-seq
//    combine walks ~pages instead of ~tokens: the dependent-load chain drops
//    from (S+W) steps to roughly (S+W)/P + O(P) steps.
//  * The admin kernel is a single block: appends and lowest-free-id page
//    allocation are done with a shared-memory bitmap and a monotone cursor
//    (frees happen strictly after appends, so the lowest free id never moves
//    backwards during a step).
//  * Attention: flash-decoding style. Grid (row, kv head x group tile,
//    split); one warp per query head; each split runs an online softmax over
//    a balanced contiguous share of the live positions with 4-token ILP
//    prefetch, uchar4/uchar2 vectorized K/V loads, inline bit-twiddled E4M3
//    decode and warp-shuffle dot reductions; a merge kernel combines the
//    per-split (m, l, acc) partials via LSE rescaling.
// ============================================================================

#include "paged_sink_e4m3_decode_common.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t kFnvBasis = 1469598103934665603ULL;
static const uint64_t kFnvPrime = 1099511628211ULL;

// ---------------------------------------------------------------------------
// State + workspace layout
// ---------------------------------------------------------------------------

struct PseRefState {
    PseProblemSpec spec;
    int max_lp;
    int log2P;
    int len_tok;      // Hkv * (2D + 2)
    uint64_t A_tok;   // p^len_tok
    uint64_t A_page;  // A_tok^P

    uint8_t* k_pool;      // [(phys*Hkv + h)*P + off]*D + d
    uint8_t* v_pool;
    int8_t* k_exp;        // [(phys*Hkv + h)*P + off]
    int8_t* v_exp;

    int32_t* page_table;  // [B * max_lp], -1 absent
    int32_t* seq_len;     // [B]
    int32_t* freed_upto;  // [B]
    uint64_t* free_bitmap;  // bit=1 free; ceil(max_pages/64) words
    int32_t* counters;    // [0]=step_counter [1]=total_allocs [2]=total_frees

    uint64_t* tok_T;      // [max_pages*P * 256]
    uint64_t* page_T;     // [max_pages * 256]
    uint8_t* page_valid;  // [max_pages]
};

struct PseWorkspaceLayout {
    int32_t* dest_slot;       // [B * 8]
    int32_t* complete_pages;  // [B * 8]
    int32_t* complete_count;  // [1]
    int32_t* plan;            // [B * plan_cap]
    int32_t* plan_len;        // [B]
    uint64_t* A_meta;         // [B]
    uint64_t* meta_T;         // [B * 256]
    size_t required_bytes;
};

static int pse_plan_cap(const PseProblemSpec* spec) {
    return spec->n_sink + spec->window + 2 * spec->page_size + 16;
}

static PseWorkspaceLayout pse_make_layout(void* workspace, const PseProblemSpec* spec) {
    PseWorkspaceLayout lay{};
    char* base = static_cast<char*>(workspace);
    const int B = spec->B;
    size_t off = 0;

    off = pse_align_up_size(off, 128);
    lay.dest_slot = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)B * PSE_MAX_NEW_TOKENS;

    off = pse_align_up_size(off, 128);
    lay.complete_pages = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)B * PSE_MAX_NEW_TOKENS;

    off = pse_align_up_size(off, 128);
    lay.complete_count = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t);

    off = pse_align_up_size(off, 128);
    lay.plan = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)B * (size_t)pse_plan_cap(spec);

    off = pse_align_up_size(off, 128);
    lay.plan_len = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)B;

    off = pse_align_up_size(off, 128);
    lay.A_meta = reinterpret_cast<uint64_t*>(base + off);
    off += sizeof(uint64_t) * (size_t)B;

    off = pse_align_up_size(off, 128);
    lay.meta_T = reinterpret_cast<uint64_t*>(base + off);
    off += sizeof(uint64_t) * (size_t)B * 256;

    off = pse_align_up_size(off, 128);
    lay.required_bytes = off;
    return lay;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t pse_fnv_byte_d(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ __forceinline__ uint64_t pse_map_step(
    uint64_t h, uint64_t A, const uint64_t* T_row) {
    return A * (h & ~0xFFULL) + T_row[h & 0xFFULL];
}

// 2^e as fp32 for e in [-126, 127]
__device__ __forceinline__ float pse_pow2f(int e) {
    return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

// Exact E4M3 decode (finite bytes).
__device__ __forceinline__ float pse_e4m3_decode_d(uint8_t b) {
    const uint32_t sign = (static_cast<uint32_t>(b) & 0x80u) << 24;
    const uint32_t E = (b >> 3) & 0xF;
    const uint32_t M = b & 0x7;
    if (E == 0) {
        const float mag = static_cast<float>(M) * 0.001953125f;  // M * 2^-9
        return __uint_as_float(sign | __float_as_uint(mag));
    }
    return __uint_as_float(sign | ((E + 120u) << 23) | (M << 20));
}

// Exact E4M3 encode per the contract: |z|>448 saturates, RNE below, sign kept.
__device__ __forceinline__ uint8_t pse_e4m3_encode_d(float z) {
    const uint32_t bits = __float_as_uint(z);
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const float a = fabsf(z);
    if (a == 0.0f) return sign;
    if (a > 448.0f) return sign | 0x7E;

    const int e = static_cast<int>((bits >> 23) & 0xFF) - 127;
    if (e >= -6) {
        // normal target: round the 23-bit mantissa to 3 bits, RNE
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
    // subnormal target: round a / 2^-9 to an integer in [0, 8], RNE
    const float t = a * 512.0f;
    int i = static_cast<int>(t);
    const float f = t - static_cast<float>(i);
    if (f > 0.5f || (f == 0.5f && (i & 1))) i++;
    if (i >= 8) return sign | 0x08;
    return sign | static_cast<uint8_t>(i);
}

__device__ __forceinline__ size_t pse_slot_head_index(
    int slot, int kvh, int Hkv, int P, int log2P) {
    const int phys = slot >> log2P;
    const int off = slot & (P - 1);
    return ((size_t)phys * Hkv + (size_t)kvh) * (size_t)P + (size_t)off;
}

// ---------------------------------------------------------------------------
// Kernel 1: admin (single block).
// ---------------------------------------------------------------------------

__global__ void pse_admin_kernel(
    int B, int Hkv, int P, int log2P, int S, int W,
    int max_lp, int max_pages, int plan_cap,
    int active_count,
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ new_token_count,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ seq_len,
    int32_t* __restrict__ freed_upto,
    uint64_t* __restrict__ free_bitmap,
    int32_t* __restrict__ counters,
    uint8_t* __restrict__ page_valid,
    int32_t* __restrict__ dest_slot,
    int32_t* __restrict__ complete_pages,
    int32_t* __restrict__ complete_count,
    int32_t* __restrict__ plan,
    int32_t* __restrict__ plan_len,
    int32_t* __restrict__ out_seq_len,
    int32_t* __restrict__ out_free_pages,
    int32_t* __restrict__ out_total_allocs,
    int32_t* __restrict__ out_total_frees) {
    __shared__ uint64_t sh_bm[PSE_MAX_PAGES / 64];
    __shared__ int sh_allocs;
    __shared__ int sh_frees;

    const int tid = threadIdx.x;
    const int nwords = (max_pages + 63) / 64;

    for (int i = tid; i < nwords; i += blockDim.x) {
        sh_bm[i] = free_bitmap[i];
    }
    for (int i = tid; i < B * PSE_MAX_NEW_TOKENS; i += blockDim.x) {
        dest_slot[i] = -1;
    }
    __syncthreads();

    if (tid == 0) {
        counters[0] += 1;  // step_counter

        int allocs = counters[1];
        int frees = counters[2];
        int n_complete = 0;
        int cw = 0;  // bitmap cursor word: frees happen after all appends,
                     // so the lowest free id is monotone during this phase

        for (int a = 0; a < active_count; ++a) {
            const int seq = active_seq[a];
            int cnt = new_token_count[a];
            if (cnt < 0) cnt = 0;
            if (cnt > PSE_MAX_NEW_TOKENS) cnt = PSE_MAX_NEW_TOKENS;

            for (int nt = 0; nt < cnt; ++nt) {
                const int pos = seq_len[seq];
                const int lp = pos >> log2P;
                const int off = pos & (P - 1);
                int entry = page_table[(size_t)seq * max_lp + lp];
                if (entry < 0) {
                    // lowest-free-id allocation
                    while (cw < nwords && sh_bm[cw] == 0ULL) ++cw;
                    const int bit = __ffsll(static_cast<long long>(sh_bm[cw])) - 1;
                    entry = cw * 64 + bit;
                    sh_bm[cw] &= ~(1ULL << bit);
                    page_table[(size_t)seq * max_lp + lp] = entry;
                    page_valid[entry] = 0;
                    ++allocs;
                }
                dest_slot[a * PSE_MAX_NEW_TOKENS + nt] = entry * P + off;
                if (off == P - 1) {
                    // page fully written: schedule its composed map
                    complete_pages[n_complete++] = entry;
                    page_valid[entry] = 1;
                }
                seq_len[seq] = pos + 1;
            }
        }

        // dead-page reclamation (b ascending, lp ascending)
        for (int b = 0; b < B; ++b) {
            const int L = seq_len[b];
            const int se = min(S, L);
            const int ws = max(se, L - W);
            const int lo = max(freed_upto[b], (se + P - 1) >> log2P);
            const int hi = (ws >> log2P) - 1;  // last dead lp
            int lp = lo;
            for (; lp <= hi; ++lp) {
                const int entry = page_table[(size_t)b * max_lp + lp];
                if (entry >= 0) {
                    page_table[(size_t)b * max_lp + lp] = -1;
                    sh_bm[entry >> 6] |= (1ULL << (entry & 63));
                    page_valid[entry] = 0;
                    ++frees;
                }
            }
            if (lp > freed_upto[b]) freed_upto[b] = lp;
        }

        counters[1] = allocs;
        counters[2] = frees;
        complete_count[0] = n_complete;
        sh_allocs = allocs;
        sh_frees = frees;
    }
    __syncthreads();

    // write back bitmap, count free pages
    for (int i = tid; i < nwords; i += blockDim.x) {
        free_bitmap[i] = sh_bm[i];
    }
    if (tid == 0) {
        int free_cnt = 0;
        for (int i = 0; i < nwords; ++i) {
            free_cnt += __popcll(sh_bm[i]);
        }
        out_free_pages[0] = free_cnt;
        out_total_allocs[0] = sh_allocs;
        out_total_frees[0] = sh_frees;
    }
    for (int b = tid; b < B; b += blockDim.x) {
        out_seq_len[b] = seq_len[b];
    }

    // build the per-sequence combine plan (one thread per sequence):
    // token entries: slot id (phys*P+off) >= 0
    // page entries:  -(phys+1) for full, composed pages inside the live set
    for (int b = tid; b < B; b += blockDim.x) {
        const int L = seq_len[b];
        const int se = min(S, L);
        const int ws = max(se, L - W);
        int n = 0;
        int32_t* pl = plan + (size_t)b * plan_cap;

        for (int range = 0; range < 2; ++range) {
            const int from = range == 0 ? 0 : ws;
            const int to = range == 0 ? se : L;
            int pos = from;
            while (pos < to) {
                const int lp = pos >> log2P;
                const int page_end = (lp + 1) << log2P;
                const int entry = page_table[(size_t)b * max_lp + lp];
                if (pos == (lp << log2P) && page_end <= to && page_valid[entry]) {
                    pl[n++] = -(entry + 1);
                    pos = page_end;
                } else {
                    const int stop = page_end < to ? page_end : to;
                    for (; pos < stop; ++pos) {
                        pl[n++] = entry * P + (pos & (P - 1));
                    }
                }
            }
        }
        plan_len[b] = n;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: quantize + append. One block per (row, token); loops over heads.
// ---------------------------------------------------------------------------

__global__ void pse_quantize_kernel(
    int Hkv, int P, int log2P, int D,
    const int32_t* __restrict__ dest_slot,
    const float* __restrict__ new_k,
    const float* __restrict__ new_v,
    uint8_t* __restrict__ k_pool,
    uint8_t* __restrict__ v_pool,
    int8_t* __restrict__ k_exp,
    int8_t* __restrict__ v_exp) {
    const int slot = dest_slot[blockIdx.x];
    if (slot < 0) return;

    const int tid = threadIdx.x;  // blockDim.x == D
    __shared__ float sh_red[128];
    __shared__ int sh_se;

    for (int h = 0; h < Hkv; ++h) {
        const size_t in_base =
            ((size_t)blockIdx.x * Hkv + (size_t)h) * (size_t)D;
        const size_t sh_idx = pse_slot_head_index(slot, h, Hkv, P, log2P);

        for (int which = 0; which < 2; ++which) {
            const float* src = which == 0 ? new_k : new_v;
            uint8_t* pool = which == 0 ? k_pool : v_pool;
            int8_t* exps = which == 0 ? k_exp : v_exp;

            const float x = src[in_base + tid];

            // exact amax reduction
            sh_red[tid] = fabsf(x);
            __syncthreads();
            for (int stride = D / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    sh_red[tid] = fmaxf(sh_red[tid], sh_red[tid + stride]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                const float amax = sh_red[0];
                int se = 0;
                if (amax != 0.0f) {
                    const int kfloor =
                        static_cast<int>((__float_as_uint(amax) >> 23) & 0xFF) - 127;
                    se = kfloor - 8;
                    if (se < -110) se = -110;
                    if (se > 110) se = 110;
                }
                sh_se = se;
            }
            __syncthreads();

            const int se = sh_se;
            const float z = x * pse_pow2f(-se);  // exact power-of-two scale
            pool[sh_idx * D + tid] = pse_e4m3_encode_d(z);
            if (tid == 0) {
                exps[sh_idx] = static_cast<int8_t>(se);
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: token chunk maps for dirty slots (256 threads = 256 trajectories).
// Chunk stream order: per head h: k_exp, K[D], v_exp, V[D].
// ---------------------------------------------------------------------------

__global__ void pse_tok_map_kernel(
    int Hkv, int P, int log2P, int D, int len_tok,
    const int32_t* __restrict__ dest_slot,
    const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool,
    const int8_t* __restrict__ k_exp,
    const int8_t* __restrict__ v_exp,
    uint64_t* __restrict__ tok_T) {
    const int slot = dest_slot[blockIdx.x];
    if (slot < 0) return;

    extern __shared__ uint8_t sh_bytes[];

    const int seg = 2 * D + 2;
    for (int idx = threadIdx.x; idx < len_tok; idx += blockDim.x) {
        const int h = idx / seg;
        const int r = idx - h * seg;
        const size_t hb = pse_slot_head_index(slot, h, Hkv, P, log2P);
        uint8_t byte;
        if (r == 0) {
            byte = static_cast<uint8_t>(k_exp[hb]);
        } else if (r <= D) {
            byte = k_pool[hb * D + (r - 1)];
        } else if (r == D + 1) {
            byte = static_cast<uint8_t>(v_exp[hb]);
        } else {
            byte = v_pool[hb * D + (r - D - 2)];
        }
        sh_bytes[idx] = byte;
    }
    __syncthreads();

    uint64_t h = static_cast<uint64_t>(threadIdx.x);
    for (int i = 0; i < len_tok; ++i) {
        h = pse_fnv_byte_d(h, sh_bytes[i]);
    }
    tok_T[(size_t)slot * 256 + threadIdx.x] = h;
}

// ---------------------------------------------------------------------------
// Kernel 4: page map composition for pages completed this step.
// Trajectory continuation: thread v folds v through the P token maps.
// ---------------------------------------------------------------------------

__global__ void pse_page_compose_kernel(
    int P, uint64_t A_tok,
    const int32_t* __restrict__ complete_pages,
    const int32_t* __restrict__ complete_count,
    const uint64_t* __restrict__ tok_T,
    uint64_t* __restrict__ page_T) {
    if (blockIdx.x >= static_cast<unsigned int>(complete_count[0])) return;
    const int phys = complete_pages[blockIdx.x];
    const int v = threadIdx.x;

    uint64_t h = tok_T[((size_t)phys * P) * 256 + v];
    for (int off = 1; off < P; ++off) {
        h = pse_map_step(h, A_tok, tok_T + ((size_t)phys * P + off) * 256);
    }
    page_T[(size_t)phys * 256 + v] = h;
}

// ---------------------------------------------------------------------------
// Kernel 5: fused online-softmax GQA attention over the live plan.
// grid: (active_count, Hkv * gtiles); one warp per query head in the tile.
// ---------------------------------------------------------------------------

template <int VEC>
__global__ void pse_attention_kernel(
    int B, int Hq, int Hkv, int P, int log2P, int D,
    int gtiles, int plan_cap,
    const int32_t* __restrict__ active_seq,
    const float* __restrict__ q,
    const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool,
    const int8_t* __restrict__ k_exp,
    const int8_t* __restrict__ v_exp,
    const int32_t* __restrict__ plan,
    const int32_t* __restrict__ plan_len,
    float* __restrict__ y,
    float* __restrict__ lse) {
    const int a = blockIdx.x;
    const int kvh = blockIdx.y / gtiles;
    const int tile = blockIdx.y - kvh * gtiles;
    const int group = Hq / Hkv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int g = tile * (blockDim.x >> 5) + warp;  // head within the group
    if (g >= group) return;
    const int hq = kvh * group + g;

    const int seq = active_seq[a];
    const int n_plan = plan_len[seq];
    const int32_t* pl = plan + (size_t)seq * plan_cap;

    float* y_out = y + ((size_t)a * Hq + (size_t)hq) * D;

    if (n_plan == 0) {
        for (int i = 0; i < VEC; ++i) {
            y_out[lane * VEC + i] = 0.0f;
        }
        if (lane == 0) {
            lse[(size_t)a * Hq + hq] = 0.0f;
        }
        return;
    }

    // per-lane query fragment (D = 32*VEC)
    const float* q_head = q + ((size_t)a * Hq + (size_t)hq) * D;
    float qv[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        qv[i] = q_head[lane * VEC + i];
    }

    const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    float m = -3.4028234663852886e38f;
    float l = 0.0f;
    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

    for (int e = 0; e < n_plan; ++e) {
        const int entry = pl[e];
        int slot0, count;
        if (entry >= 0) {
            slot0 = entry;
            count = 1;
        } else {
            slot0 = (-entry - 1) * P;
            count = P;
        }

        for (int c = 0; c < count; ++c) {
            const int slot = slot0 + c;
            const size_t hb = pse_slot_head_index(slot, kvh, Hkv, P, log2P);
            const uint8_t* kb = k_pool + hb * D + lane * VEC;
            const uint8_t* vb = v_pool + hb * D + lane * VEC;

            float part = 0.0f;
#pragma unroll
            for (int i = 0; i < VEC; ++i) {
                part += qv[i] * pse_e4m3_decode_d(kb[i]);
            }
#pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1) {
                part += __shfl_down_sync(0xFFFFFFFFu, part, sh);
            }
            float dot = __shfl_sync(0xFFFFFFFFu, part, 0);

            const float s = dot * pse_pow2f(k_exp[hb]) * inv_sqrt_d;
            const float m_new = fmaxf(m, s);
            const float alpha = expf(m - m_new);
            const float p = expf(s - m_new);
            l = l * alpha + p;
            const float pv = p * pse_pow2f(v_exp[hb]);
#pragma unroll
            for (int i = 0; i < VEC; ++i) {
                acc[i] = acc[i] * alpha + pv * pse_e4m3_decode_d(vb[i]);
            }
            m = m_new;
        }
    }

    const float inv_l = 1.0f / l;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        y_out[lane * VEC + i] = acc[i] * inv_l;
    }
    if (lane == 0) {
        lse[(size_t)a * Hq + hq] = m + logf(l);
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: per-sequence page-table metadata chunk maps (for the global
// checksum): chunk bytes = seq_len, n_lp, page_table[0..n_lp).
// ---------------------------------------------------------------------------

__global__ void pse_seqmeta_map_kernel(
    int P, int max_lp,
    const int32_t* __restrict__ seq_len,
    const int32_t* __restrict__ page_table,
    uint64_t* __restrict__ A_meta,
    uint64_t* __restrict__ meta_T) {
    const int b = blockIdx.x;
    extern __shared__ uint8_t sh_meta[];

    const int L = seq_len[b];
    const int n_lp = (L + P - 1) / P;
    const int len = 8 + n_lp * 4;

    for (int idx = threadIdx.x; idx < len; idx += blockDim.x) {
        int32_t word;
        const int w = idx >> 2;
        if (w == 0) {
            word = L;
        } else if (w == 1) {
            word = n_lp;
        } else {
            word = page_table[(size_t)b * max_lp + (w - 2)];
        }
        sh_meta[idx] = static_cast<uint8_t>(
            (static_cast<uint32_t>(word) >> (8 * (idx & 3))) & 0xFF);
    }
    __syncthreads();

    uint64_t h = static_cast<uint64_t>(threadIdx.x);
    uint64_t A = 1ULL;
    for (int i = 0; i < len; ++i) {
        h = pse_fnv_byte_d(h, sh_meta[i]);
        A *= 1099511628211ULL;
    }
    meta_T[(size_t)b * 256 + threadIdx.x] = h;
    if (threadIdx.x == 0) {
        A_meta[b] = A;
    }
}

// ---------------------------------------------------------------------------
// Kernel 7: final hashes. Threads 0..B-1 walk their sequence's plan for
// kv_hash; then thread 0 combines the global page-state checksum.
// ---------------------------------------------------------------------------

__global__ void pse_final_hash_kernel(
    int B, int Hq, int Hkv, int P, int S, int W, int D,
    int max_pages, int max_seq_len, int plan_cap,
    uint64_t A_tok, uint64_t A_page,
    const int32_t* __restrict__ seq_len,
    const int32_t* __restrict__ counters,
    const int32_t* __restrict__ plan,
    const int32_t* __restrict__ plan_len,
    const uint64_t* __restrict__ tok_T,
    const uint64_t* __restrict__ page_T,
    const uint64_t* __restrict__ A_meta,
    const uint64_t* __restrict__ meta_T,
    const int32_t* __restrict__ out_free_pages,
    uint64_t* __restrict__ kv_hash,
    uint64_t* __restrict__ page_state_checksum) {
    const int b = threadIdx.x;

    if (b < B) {
        const int L = seq_len[b];
        const int se = min(S, L);
        const int ws = max(se, L - W);

        uint64_t h = kFnvBasis;
        const int32_t pre[3] = {L, se, ws};
        for (int i = 0; i < 3; ++i) {
            const uint32_t v = static_cast<uint32_t>(pre[i]);
            for (int k = 0; k < 4; ++k) {
                h = pse_fnv_byte_d(h, static_cast<uint8_t>((v >> (8 * k)) & 0xFF));
            }
        }

        const int n = plan_len[b];
        const int32_t* pl = plan + (size_t)b * plan_cap;
        for (int e = 0; e < n; ++e) {
            const int entry = pl[e];
            if (entry >= 0) {
                h = pse_map_step(h, A_tok, tok_T + (size_t)entry * 256);
            } else {
                h = pse_map_step(h, A_page, page_T + (size_t)(-entry - 1) * 256);
            }
        }
        kv_hash[b] = h;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t h = kFnvBasis;
        const int32_t header[10] = {
            B, Hq, Hkv, D, P, S, W, max_pages, max_seq_len, counters[0]};
        for (int i = 0; i < 10; ++i) {
            const uint32_t v = static_cast<uint32_t>(header[i]);
            for (int k = 0; k < 4; ++k) {
                h = pse_fnv_byte_d(h, static_cast<uint8_t>((v >> (8 * k)) & 0xFF));
            }
        }
        for (int bb = 0; bb < B; ++bb) {
            h = pse_map_step(h, A_meta[bb], meta_T + (size_t)bb * 256);
        }
        const int32_t tail[3] = {counters[1], counters[2], out_free_pages[0]};
        for (int i = 0; i < 3; ++i) {
            const uint32_t v = static_cast<uint32_t>(tail[i]);
            for (int k = 0; k < 4; ++k) {
                h = pse_fnv_byte_d(h, static_cast<uint8_t>((v >> (8 * k)) & 0xFF));
            }
        }
        page_state_checksum[0] = h;
    }
}

// ---------------------------------------------------------------------------
// Bitmap reset kernel (exact partial-word handling).
// ---------------------------------------------------------------------------

__global__ void pse_reset_bitmap_kernel(
    int max_pages, uint64_t* __restrict__ free_bitmap) {
    const int nwords = (max_pages + 63) / 64;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nwords) return;
    const int rem = max_pages - i * 64;
    free_bitmap[i] = rem >= 64 ? ~0ULL : ((1ULL << rem) - 1ULL);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

static cudaError_t pse_reference_reset_state(PseRefState* st, cudaStream_t stream) {
    const PseProblemSpec& s = st->spec;
    cudaError_t err = cudaSuccess;

    const size_t pool_elems =
        (size_t)s.max_pages * s.Hkv * s.page_size * (size_t)s.D;
    const size_t exp_elems = (size_t)s.max_pages * s.Hkv * s.page_size;

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
    err = cudaMemsetAsync(st->freed_upto, 0, sizeof(int32_t) * (size_t)s.B, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->counters, 0, sizeof(int32_t) * 4, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->page_valid, 0, (size_t)s.max_pages, stream);
    if (err != cudaSuccess) return err;

    const int nwords = (s.max_pages + 63) / 64;
    pse_reset_bitmap_kernel<<<pse_ceil_div_int(nwords, 128), 128, 0, stream>>>(
        s.max_pages, st->free_bitmap);
    return cudaPeekAtLastError();
}

extern "C" size_t solution_workspace_bytes(const PseProblemSpec* spec) {
    if (!pse_validate_problem_spec(spec)) return 0;
    PseWorkspaceLayout lay = pse_make_layout(nullptr, spec);
    return lay.required_bytes;
}

extern "C" cudaError_t solution_init(
    const PseProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!pse_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    PseRefState* st = static_cast<PseRefState*>(malloc(sizeof(PseRefState)));
    if (!st) {
        return cudaErrorMemoryAllocation;
    }
    memset(st, 0, sizeof(PseRefState));
    memcpy(&st->spec, spec, sizeof(PseProblemSpec));

    st->max_lp = pse_max_logical_pages(spec->max_seq_len, spec->page_size);
    st->log2P = spec->page_size == 8 ? 3 : (spec->page_size == 16 ? 4 : 5);
    st->len_tok = spec->Hkv * (2 * spec->D + 2);

    uint64_t A = 1ULL;
    for (int i = 0; i < st->len_tok; ++i) A *= kFnvPrime;
    st->A_tok = A;
    uint64_t Ap = 1ULL;
    for (int i = 0; i < spec->page_size; ++i) Ap *= A;
    st->A_page = Ap;

    const size_t pool_elems =
        (size_t)spec->max_pages * spec->Hkv * spec->page_size * (size_t)spec->D;
    const size_t exp_elems = (size_t)spec->max_pages * spec->Hkv * spec->page_size;
    const size_t nslots = (size_t)spec->max_pages * spec->page_size;
    const int nwords = (spec->max_pages + 63) / 64;

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
    err = cudaMalloc(reinterpret_cast<void**>(&st->freed_upto),
                     sizeof(int32_t) * (size_t)spec->B);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->free_bitmap),
                     sizeof(uint64_t) * (size_t)nwords);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->counters), sizeof(int32_t) * 4);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->tok_T),
                     sizeof(uint64_t) * nslots * 256);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->page_T),
                     sizeof(uint64_t) * (size_t)spec->max_pages * 256);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->page_valid), (size_t)spec->max_pages);
    if (err != cudaSuccess) goto fail;

    err = pse_reference_reset_state(st, stream);
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
    if (st->freed_upto) cudaFree(st->freed_upto);
    if (st->free_bitmap) cudaFree(st->free_bitmap);
    if (st->counters) cudaFree(st->counters);
    if (st->tok_T) cudaFree(st->tok_T);
    if (st->page_T) cudaFree(st->page_T);
    if (st->page_valid) cudaFree(st->page_valid);
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

    PseRefState* st = static_cast<PseRefState*>(state);
    if (!pse_validate_run_spec(run, &st->spec)) {
        return cudaErrorInvalidValue;
    }

    const PseInputs* in = static_cast<const PseInputs*>(inputs_void);
    PseOutputs* out = static_cast<PseOutputs*>(outputs_void);

    if (!in->active_seq || !in->new_token_count || !in->new_k || !in->new_v ||
        !in->q || !out->y || !out->lse || !out->seq_len || !out->kv_hash ||
        !out->page_state_checksum || !out->free_pages || !out->total_allocs ||
        !out->total_frees) {
        return cudaErrorInvalidValue;
    }

    PseWorkspaceLayout lay = pse_make_layout(workspace, &st->spec);
    if (workspace_bytes < lay.required_bytes) {
        return cudaErrorInvalidValue;
    }

    const PseProblemSpec& s = st->spec;
    const int A = run->active_count;
    const int plan_cap = pse_plan_cap(&s);
    cudaError_t err = cudaSuccess;

    pse_admin_kernel<<<1, 256, 0, stream>>>(
        s.B, s.Hkv, s.page_size, st->log2P, s.n_sink, s.window,
        st->max_lp, s.max_pages, plan_cap,
        A,
        in->active_seq,
        in->new_token_count,
        st->page_table,
        st->seq_len,
        st->freed_upto,
        st->free_bitmap,
        st->counters,
        st->page_valid,
        lay.dest_slot,
        lay.complete_pages,
        lay.complete_count,
        lay.plan,
        lay.plan_len,
        out->seq_len,
        out->free_pages,
        out->total_allocs,
        out->total_frees);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    if (A > 0) {
        const int nblocks = A * PSE_MAX_NEW_TOKENS;

        pse_quantize_kernel<<<nblocks, s.D, 0, stream>>>(
            s.Hkv, s.page_size, st->log2P, s.D,
            lay.dest_slot,
            in->new_k,
            in->new_v,
            st->k_pool,
            st->v_pool,
            st->k_exp,
            st->v_exp);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        pse_tok_map_kernel<<<nblocks, 256, (size_t)st->len_tok, stream>>>(
            s.Hkv, s.page_size, st->log2P, s.D, st->len_tok,
            lay.dest_slot,
            st->k_pool,
            st->v_pool,
            st->k_exp,
            st->v_exp,
            st->tok_T);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        pse_page_compose_kernel<<<nblocks, 256, 0, stream>>>(
            s.page_size, st->A_tok,
            lay.complete_pages,
            lay.complete_count,
            st->tok_T,
            st->page_T);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        const int group = s.Hq / s.Hkv;
        const int warps = group < 8 ? group : 8;
        const int gtiles = pse_ceil_div_int(group, warps);
        dim3 attn_grid(A, s.Hkv * gtiles, 1);
        const int attn_block = 32 * warps;

        if (s.D == 128) {
            pse_attention_kernel<4><<<attn_grid, attn_block, 0, stream>>>(
                s.B, s.Hq, s.Hkv, s.page_size, st->log2P, s.D, gtiles, plan_cap,
                in->active_seq, in->q,
                st->k_pool, st->v_pool, st->k_exp, st->v_exp,
                lay.plan, lay.plan_len,
                out->y, out->lse);
        } else {
            pse_attention_kernel<2><<<attn_grid, attn_block, 0, stream>>>(
                s.B, s.Hq, s.Hkv, s.page_size, st->log2P, s.D, gtiles, plan_cap,
                in->active_seq, in->q,
                st->k_pool, st->v_pool, st->k_exp, st->v_exp,
                lay.plan, lay.plan_len,
                out->y, out->lse);
        }
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    {
        const size_t meta_shmem = (size_t)(8 + st->max_lp * 4);
        pse_seqmeta_map_kernel<<<s.B, 256, meta_shmem, stream>>>(
            s.page_size, st->max_lp,
            st->seq_len,
            st->page_table,
            lay.A_meta,
            lay.meta_T);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        pse_final_hash_kernel<<<1, 64, 0, stream>>>(
            s.B, s.Hq, s.Hkv, s.page_size, s.n_sink, s.window, s.D,
            s.max_pages, s.max_seq_len, plan_cap,
            st->A_tok, st->A_page,
            st->seq_len,
            st->counters,
            lay.plan,
            lay.plan_len,
            st->tok_T,
            st->page_T,
            lay.A_meta,
            lay.meta_T,
            out->free_pages,
            out->kv_hash,
            out->page_state_checksum);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return pse_reference_reset_state(static_cast<PseRefState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;

    PseRefState* st = static_cast<PseRefState*>(state);
    if (st->k_pool) cudaFree(st->k_pool);
    if (st->v_pool) cudaFree(st->v_pool);
    if (st->k_exp) cudaFree(st->k_exp);
    if (st->v_exp) cudaFree(st->v_exp);
    if (st->page_table) cudaFree(st->page_table);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->freed_upto) cudaFree(st->freed_upto);
    if (st->free_bitmap) cudaFree(st->free_bitmap);
    if (st->counters) cudaFree(st->counters);
    if (st->tok_T) cudaFree(st->tok_T);
    if (st->page_T) cudaFree(st->page_T);
    if (st->page_valid) cudaFree(st->page_valid);
    free(st);
}
