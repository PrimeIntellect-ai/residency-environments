// ============================================================================
// file: naive_ref.cu  (authoring artifact, NOT shipped to solvers)
//
// Clean, obviously-correct but unoptimized implementation of the
// spec_tree_verify_e5m2_kv contract:
//   - single-thread admin with linear byte-array page scans,
//   - one scalar thread per node for quantization and signatures,
//   - one scalar thread per (row, node, head) two-pass attention re-reading
//     the paged cache and walking the ancestor path per pass,
//   - full from-scratch kv_hash refold per sequence every step,
//   - single-thread commit planning + verification cascade.
// Used to measure the naive/reference performance ratio.
// ============================================================================

#include "spec_tree_verify_e5m2_kv_common.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t kNvFnvBasis = 1469598103934665603ULL;

struct StvNaiveState {
    StvProblemSpec spec;
    int max_lp;
    int maxnpg;

    uint8_t* k_pool;
    uint8_t* v_pool;
    int8_t* k_exp;
    int8_t* v_exp;

    int32_t* page_table;
    int32_t* seq_len;
    uint8_t* page_used;
    int32_t* counters;  // [0]=step [1]=allocs [2]=frees
};

struct StvNaiveWs {
    int32_t* scratch_pages;  // [B * maxnpg]
    int8_t* bonus_exp;       // [B * Hkv * 2]
    uint8_t* bonus_bytes;    // [B * Hkv * 2 * D]
    int32_t* commit_src;     // [B * (N+1)]
    int32_t* commit_page;
    int32_t* commit_slot;
    int32_t* commit_count;   // [B]
    size_t required_bytes;
};

static StvNaiveWs stv_nv_layout(void* ws, const StvProblemSpec* spec) {
    StvNaiveWs lay{};
    char* base = static_cast<char*>(ws);
    const int B = spec->B;
    const int N = spec->max_nodes;
    const int maxnpg = stv_ceil_div_int(N, spec->page_size);
    size_t off = 0;

    #define NV_WS(field, type, count)                                         \
        off = stv_align_up_size(off, 128);                                    \
        lay.field = reinterpret_cast<type*>(base + off);                      \
        off += sizeof(type) * (count)

    NV_WS(scratch_pages, int32_t, (size_t)B * maxnpg);
    NV_WS(bonus_exp, int8_t, (size_t)B * spec->Hkv * 2);
    NV_WS(bonus_bytes, uint8_t, (size_t)B * spec->Hkv * 2 * spec->D);
    NV_WS(commit_src, int32_t, (size_t)B * (N + 1));
    NV_WS(commit_page, int32_t, (size_t)B * (N + 1));
    NV_WS(commit_slot, int32_t, (size_t)B * (N + 1));
    NV_WS(commit_count, int32_t, (size_t)B);
    #undef NV_WS

    off = stv_align_up_size(off, 128);
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

__device__ __forceinline__ float nv_e5m2_decode(uint8_t b) {
    const uint32_t sign = (static_cast<uint32_t>(b) & 0x80u) << 24;
    const uint32_t E = (b >> 2) & 0x1F;
    const uint32_t M = b & 0x3;
    if (E == 0) {
        const float mag = static_cast<float>(M) * 0.0000152587890625f;
        return __uint_as_float(sign | __float_as_uint(mag));
    }
    return __uint_as_float(sign | ((E + 112u) << 23) | (M << 21));
}

__device__ __forceinline__ uint8_t nv_e5m2_encode(float z) {
    const uint32_t bits = __float_as_uint(z);
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const float a = fabsf(z);
    if (a == 0.0f) return sign;
    if (a > 57344.0f) return sign | 0x7B;
    const int e = static_cast<int>((bits >> 23) & 0xFF) - 127;
    if (e >= -14) {
        const uint32_t mant = bits & 0x7FFFFFu;
        uint32_t keep = mant >> 21;
        const uint32_t rest = mant & 0x1FFFFFu;
        const uint32_t half = 0x100000u;
        int ee = e;
        if (rest > half || (rest == half && (keep & 1u))) keep++;
        if (keep == 4u) { keep = 0u; ee++; }
        if (ee > 15) return sign | 0x7B;
        return sign | static_cast<uint8_t>((ee + 15) << 2) | static_cast<uint8_t>(keep);
    }
    const float t = a * 65536.0f;
    int i = static_cast<int>(t);
    const float f = t - static_cast<float>(i);
    if (f > 0.5f || (f == 0.5f && (i & 1))) i++;
    if (i >= 4) return sign | 0x04;
    return sign | static_cast<uint8_t>(i);
}

__device__ __forceinline__ size_t nv_slot_head(
    int page, int slot, int kvh, int Hkv, int P) {
    return ((size_t)page * Hkv + (size_t)kvh) * (size_t)P + (size_t)slot;
}

// Single-thread: step counter + scratch allocation (linear scans).
__global__ void nv_admin_pre_kernel(
    const int32_t* __restrict__ node_count,
    uint8_t* __restrict__ page_used,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ scratch_pages,
    int active_count,
    int P,
    int maxnpg,
    int max_pages) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    counters[0] += 1;
    for (int a = 0; a < active_count; ++a) {
        const int npg = (node_count[a] + P - 1) / P;
        for (int g = 0; g < npg; ++g) {
            int page = -1;
            for (int p = 0; p < max_pages; ++p) {
                if (!page_used[p]) { page = p; page_used[p] = 1; break; }
            }
            scratch_pages[a * maxnpg + g] = page;
            counters[1] += 1;
        }
    }
}

// One scalar thread per (a, node): quantize; nt beyond node_count skipped.
__global__ void nv_quantize_kernel(
    const int32_t* __restrict__ node_count,
    const float* __restrict__ draft_k,
    const float* __restrict__ draft_v,
    const float* __restrict__ bonus_k,
    const float* __restrict__ bonus_v,
    const int32_t* __restrict__ scratch_pages,
    uint8_t* __restrict__ k_pool,
    uint8_t* __restrict__ v_pool,
    int8_t* __restrict__ k_exp,
    int8_t* __restrict__ v_exp,
    int8_t* __restrict__ bonus_exp,
    uint8_t* __restrict__ bonus_bytes,
    int N,
    int Hkv,
    int D,
    int P,
    int maxnpg) {
    const int a = blockIdx.x;
    const int i = threadIdx.x;   // node index or N for bonus
    if (i > N) return;

    const bool is_bonus = (i == N);
    if (!is_bonus && i >= node_count[a]) return;

    for (int hh = 0; hh < Hkv; ++hh) {
        for (int sec = 0; sec < 2; ++sec) {
            const float* x = is_bonus
                ? ((sec == 0 ? bonus_k : bonus_v) + ((size_t)a * Hkv + hh) * D)
                : ((sec == 0 ? draft_k : draft_v) +
                   (((size_t)a * N + i) * Hkv + hh) * (size_t)D);

            float amax = 0.0f;
            for (int d = 0; d < D; ++d) {
                const float v = fabsf(x[d]);
                if (v > amax) amax = v;
            }

            int8_t se8 = 0;
            float inv_scale = 0.0f;
            if (amax != 0.0f) {
                const int kfloor =
                    (int)((__float_as_uint(amax) >> 23) & 0xFF) - 127;
                int se = kfloor - 15;
                se = se < -110 ? -110 : (se > 110 ? 110 : se);
                se8 = (int8_t)se;
                inv_scale = nv_pow2f(-se);
            }

            if (is_bonus) {
                bonus_exp[((size_t)a * Hkv + hh) * 2 + sec] = se8;
                for (int d = 0; d < D; ++d) {
                    uint8_t byte;
                    if (amax == 0.0f) {
                        byte = (__float_as_uint(x[d]) & 0x80000000u) ? 0x80 : 0x00;
                    } else {
                        byte = nv_e5m2_encode(x[d] * inv_scale);
                    }
                    bonus_bytes[(((size_t)a * Hkv + hh) * 2 + sec) * D + d] = byte;
                }
            } else {
                const int page = scratch_pages[a * maxnpg + i / P];
                const size_t hs = nv_slot_head(page, i % P, hh, Hkv, P);
                if (sec == 0) k_exp[hs] = se8; else v_exp[hs] = se8;
                uint8_t* pool = sec == 0 ? k_pool : v_pool;
                for (int d = 0; d < D; ++d) {
                    uint8_t byte;
                    if (amax == 0.0f) {
                        byte = (__float_as_uint(x[d]) & 0x80000000u) ? 0x80 : 0x00;
                    } else {
                        byte = nv_e5m2_encode(x[d] * inv_scale);
                    }
                    pool[hs * D + d] = byte;
                }
            }
        }
    }
}

// Single thread per row: signatures + acceptance + commit source list.
__global__ void nv_verify_kernel(
    const int32_t* __restrict__ node_count,
    const int32_t* __restrict__ parent,
    const uint64_t* __restrict__ target_sig,
    const int32_t* __restrict__ scratch_pages,
    const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool,
    const int8_t* __restrict__ k_exp,
    const int8_t* __restrict__ v_exp,
    int32_t* __restrict__ commit_src,
    int32_t* __restrict__ commit_count,
    int32_t* __restrict__ out_accepted_tail,
    int32_t* __restrict__ out_accepted_len,
    int N,
    int Hkv,
    int D,
    int P,
    int maxnpg) {
    const int a = blockIdx.x;
    if (threadIdx.x != 0) return;

    bool acc[STV_MAX_NODES_LIMIT];
    int depth[STV_MAX_NODES_LIMIT];
    const int n = node_count[a];
    int best = -1;
    int best_depth = 0;

    for (int i = 0; i < n; ++i) {
        const int page = scratch_pages[a * maxnpg + i / P];
        uint64_t h = kNvFnvBasis;
        for (int hh = 0; hh < Hkv; ++hh) {
            const size_t hs = nv_slot_head(page, i % P, hh, Hkv, P);
            h = nv_fnv_byte(h, (uint8_t)k_exp[hs]);
            for (int d = 0; d < D; ++d) h = nv_fnv_byte(h, k_pool[hs * D + d]);
            h = nv_fnv_byte(h, (uint8_t)v_exp[hs]);
            for (int d = 0; d < D; ++d) h = nv_fnv_byte(h, v_pool[hs * D + d]);
        }
        const int par = parent[(size_t)a * N + i];
        const bool par_ok = (par == -1) ? true : acc[par];
        acc[i] = par_ok && (h == target_sig[(size_t)a * N + i]);
        depth[i] = (par == -1) ? 1 : depth[par] + 1;
        if (acc[i] && depth[i] > best_depth) {
            best_depth = depth[i];
            best = i;
        }
    }

    out_accepted_tail[a] = best;
    out_accepted_len[a] = best < 0 ? 0 : best_depth;

    int32_t* cs = commit_src + (size_t)a * (N + 1);
    int cnt = 0;
    if (best >= 0) {
        int rev[STV_MAX_NODES_LIMIT];
        int len = 0;
        int cur = best;
        while (cur != -1) {
            rev[len++] = cur;
            cur = parent[(size_t)a * N + cur];
        }
        for (int k = len - 1; k >= 0; --k) cs[cnt++] = rev[k];
    }
    cs[cnt++] = -2;
    commit_count[a] = cnt;
}

// One scalar thread per (a, node, hq): two-pass attention.
__global__ void nv_attention_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ node_count,
    const int32_t* __restrict__ parent,
    const float* __restrict__ q,
    const int32_t* __restrict__ seq_len,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ scratch_pages,
    const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool,
    const int8_t* __restrict__ k_exp,
    const int8_t* __restrict__ v_exp,
    float* __restrict__ y,
    float* __restrict__ lse,
    int N,
    int Hq,
    int Hkv,
    int D,
    int P,
    int max_lp,
    int maxnpg,
    int nq) {
    const int qidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (qidx >= nq) return;
    const int hq = qidx % Hq;
    const int slot_id = qidx / Hq;
    const int a = slot_id / N;
    const int i = slot_id - a * N;
    if (i >= node_count[a]) return;

    const int b = active_seq[a];
    const int L = seq_len[b];
    const int kvh = hq / (Hq / Hkv);
    const float sm_scale = rsqrtf((float)D);

    int path[STV_MAX_NODES_LIMIT];
    int plen = 0;
    {
        int cur = i;
        while (cur != -1) {
            path[plen++] = cur;
            cur = parent[(size_t)a * N + cur];
        }
        // reverse
        for (int k = 0; k < plen / 2; ++k) {
            const int tmp = path[k];
            path[k] = path[plen - 1 - k];
            path[plen - 1 - k] = tmp;
        }
    }

    const float* qv = q + (((size_t)a * N + i) * Hq + hq) * (size_t)D;
    const int total = L + plen;

    auto token_slot = [&](int t, int* page, int* slot) {
        if (t < L) {
            *page = page_table[(size_t)b * max_lp + t / P];
            *slot = t % P;
        } else {
            const int node = path[t - L];
            *page = scratch_pages[a * maxnpg + node / P];
            *slot = node % P;
        }
    };

    float m = -CUDART_INF_F;
    for (int t = 0; t < total; ++t) {
        int page, slot;
        token_slot(t, &page, &slot);
        const size_t hs = nv_slot_head(page, slot, kvh, Hkv, P);
        const float sc = nv_pow2f((int)k_exp[hs]);
        float s = 0.0f;
        for (int d = 0; d < D; ++d) {
            s += qv[d] * (nv_e5m2_decode(k_pool[hs * D + d]) * sc);
        }
        s *= sm_scale;
        if (s > m) m = s;
    }

    float l = 0.0f;
    for (int d = 0; d < D; ++d) {
        y[(size_t)qidx * D + d] = 0.0f;
    }
    for (int t = 0; t < total; ++t) {
        int page, slot;
        token_slot(t, &page, &slot);
        const size_t hs = nv_slot_head(page, slot, kvh, Hkv, P);
        const float ksc = nv_pow2f((int)k_exp[hs]);
        float s = 0.0f;
        for (int d = 0; d < D; ++d) {
            s += qv[d] * (nv_e5m2_decode(k_pool[hs * D + d]) * ksc);
        }
        s *= sm_scale;
        const float w = expf(s - m);
        l += w;
        const float vsc = nv_pow2f((int)v_exp[hs]);
        for (int d = 0; d < D; ++d) {
            y[(size_t)qidx * D + d] += w * (nv_e5m2_decode(v_pool[hs * D + d]) * vsc);
        }
    }
    const float inv_l = 1.0f / l;
    for (int d = 0; d < D; ++d) {
        y[(size_t)qidx * D + d] *= inv_l;
    }
    lse[qidx] = m + logf(l);
}

// Single thread: commit planning + scratch release.
__global__ void nv_commit_plan_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ node_count,
    const int32_t* __restrict__ commit_src,
    const int32_t* __restrict__ commit_count,
    int32_t* __restrict__ seq_len,
    int32_t* __restrict__ page_table,
    uint8_t* __restrict__ page_used,
    int32_t* __restrict__ counters,
    const int32_t* __restrict__ scratch_pages,
    int32_t* __restrict__ commit_page,
    int32_t* __restrict__ commit_slot,
    int active_count,
    int N,
    int P,
    int max_lp,
    int maxnpg,
    int max_pages) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int a = 0; a < active_count; ++a) {
        const int b = active_seq[a];
        const int cnt = commit_count[a];
        for (int c = 0; c < cnt; ++c) {
            const int pos = seq_len[b];
            const int lp = pos / P;
            if (page_table[(size_t)b * max_lp + lp] < 0) {
                int page = -1;
                for (int p = 0; p < max_pages; ++p) {
                    if (!page_used[p]) { page = p; page_used[p] = 1; break; }
                }
                page_table[(size_t)b * max_lp + lp] = page;
                counters[1] += 1;
            }
            commit_page[(size_t)a * (N + 1) + c] =
                page_table[(size_t)b * max_lp + lp];
            commit_slot[(size_t)a * (N + 1) + c] = pos % P;
            seq_len[b] += 1;
        }
    }
    for (int a = 0; a < active_count; ++a) {
        const int npg = (node_count[a] + P - 1) / P;
        for (int g = 0; g < npg; ++g) {
            page_used[scratch_pages[a * maxnpg + g]] = 0;
            counters[2] += 1;
        }
    }
}

// Block per (a, c): copy committed bytes.
__global__ void nv_commit_copy_kernel(
    const int32_t* __restrict__ commit_src,
    const int32_t* __restrict__ commit_count,
    const int32_t* __restrict__ commit_page,
    const int32_t* __restrict__ commit_slot,
    const int32_t* __restrict__ scratch_pages,
    const int8_t* __restrict__ bonus_exp,
    const uint8_t* __restrict__ bonus_bytes,
    uint8_t* __restrict__ k_pool,
    uint8_t* __restrict__ v_pool,
    int8_t* __restrict__ k_exp,
    int8_t* __restrict__ v_exp,
    int N,
    int Hkv,
    int D,
    int P,
    int maxnpg) {
    const int a = blockIdx.x;
    const int c = blockIdx.y;
    if (c >= commit_count[a]) return;

    const int src = commit_src[(size_t)a * (N + 1) + c];
    const int dpage = commit_page[(size_t)a * (N + 1) + c];
    const int dslot = commit_slot[(size_t)a * (N + 1) + c];

    for (int hh = 0; hh < Hkv; ++hh) {
        const size_t dst = nv_slot_head(dpage, dslot, hh, Hkv, P);
        if (src >= 0) {
            const size_t s =
                nv_slot_head(scratch_pages[a * maxnpg + src / P], src % P, hh, Hkv, P);
            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                k_pool[dst * D + d] = k_pool[s * D + d];
                v_pool[dst * D + d] = v_pool[s * D + d];
            }
            if (threadIdx.x == 0) {
                k_exp[dst] = k_exp[s];
                v_exp[dst] = v_exp[s];
            }
        } else {
            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                k_pool[dst * D + d] =
                    bonus_bytes[(((size_t)a * Hkv + hh) * 2 + 0) * D + d];
                v_pool[dst * D + d] =
                    bonus_bytes[(((size_t)a * Hkv + hh) * 2 + 1) * D + d];
            }
            if (threadIdx.x == 0) {
                k_exp[dst] = bonus_exp[((size_t)a * Hkv + hh) * 2 + 0];
                v_exp[dst] = bonus_exp[((size_t)a * Hkv + hh) * 2 + 1];
            }
        }
    }
}

// One thread per sequence: full kv_hash refold from scratch.
__global__ void nv_hash_kernel(
    const int32_t* __restrict__ seq_len,
    const int32_t* __restrict__ page_table,
    const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool,
    const int8_t* __restrict__ k_exp,
    const int8_t* __restrict__ v_exp,
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
            const size_t hs = nv_slot_head(page, t % P, hh, Hkv, P);
            h = nv_fnv_byte(h, (uint8_t)k_exp[hs]);
            for (int d = 0; d < D; ++d) h = nv_fnv_byte(h, k_pool[hs * D + d]);
            h = nv_fnv_byte(h, (uint8_t)v_exp[hs]);
            for (int d = 0; d < D; ++d) h = nv_fnv_byte(h, v_pool[hs * D + d]);
        }
    }
    h = nv_fnv_i32(h, L);
    out_kv_hash[b] = h;
}

// Single thread: remaining exact outputs.
__global__ void nv_final_kernel(
    const int32_t* __restrict__ seq_len,
    const int32_t* __restrict__ page_table,
    const uint8_t* __restrict__ page_used,
    const int32_t* __restrict__ counters,
    int32_t* __restrict__ out_seq_len,
    uint64_t* __restrict__ out_pcs,
    int32_t* __restrict__ out_free_pages,
    int32_t* __restrict__ out_total_allocs,
    int32_t* __restrict__ out_total_frees,
    int B,
    int Hq,
    int Hkv,
    int D,
    int P,
    int N,
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
    ph = nv_fnv_i32(ph, N);
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
    for (int p = 0; p < max_pages; ++p) freep += page_used[p] ? 0 : 1;
    ph = nv_fnv_i32(ph, counters[1]);
    ph = nv_fnv_i32(ph, counters[2]);
    ph = nv_fnv_i32(ph, freep);
    out_pcs[0] = ph;
    out_free_pages[0] = freep;
    out_total_allocs[0] = counters[1];
    out_total_frees[0] = counters[2];
}

__global__ void nv_reset_kernel(
    int32_t* __restrict__ seq_len,
    int32_t* __restrict__ page_table,
    uint8_t* __restrict__ page_used,
    int32_t* __restrict__ counters,
    int B,
    int max_lp,
    int max_pages) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    if (i < B) seq_len[i] = 0;
    for (int j = i; j < B * max_lp; j += stride) page_table[j] = -1;
    for (int p = i; p < max_pages; p += stride) page_used[p] = 0;
    if (i < 3) counters[i] = 0;
}

extern "C" size_t solution_workspace_bytes(const StvProblemSpec* spec) {
    if (!stv_validate_problem_spec(spec)) return 0;
    return stv_nv_layout(nullptr, spec).required_bytes;
}

extern "C" cudaError_t solution_init(
    const StvProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!state_out) return cudaErrorInvalidValue;
    *state_out = nullptr;
    if (!stv_validate_problem_spec(spec)) return cudaErrorInvalidValue;

    StvNaiveState* st =
        static_cast<StvNaiveState*>(malloc(sizeof(StvNaiveState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(*st));
    st->spec = *spec;
    st->max_lp = stv_max_logical_pages(spec->max_seq_len, spec->page_size);
    st->maxnpg = stv_ceil_div_int(spec->max_nodes, spec->page_size);

    const size_t slots =
        (size_t)spec->max_pages * spec->Hkv * spec->page_size;

    cudaError_t err = cudaSuccess;
    #define NV_ALLOC(ptr, bytes)                                              \
        do {                                                                  \
            err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes));      \
            if (err != cudaSuccess) goto fail;                                \
        } while (0)

    NV_ALLOC(st->k_pool, slots * spec->D);
    NV_ALLOC(st->v_pool, slots * spec->D);
    NV_ALLOC(st->k_exp, slots);
    NV_ALLOC(st->v_exp, slots);
    NV_ALLOC(st->page_table, sizeof(int32_t) * (size_t)spec->B * st->max_lp);
    NV_ALLOC(st->seq_len, sizeof(int32_t) * spec->B);
    NV_ALLOC(st->page_used, (size_t)spec->max_pages);
    NV_ALLOC(st->counters, sizeof(int32_t) * 4);
    #undef NV_ALLOC

    nv_reset_kernel<<<8, 128, 0, stream>>>(
        st->seq_len, st->page_table, st->page_used, st->counters,
        spec->B, st->max_lp, spec->max_pages);
    err = cudaGetLastError();
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
    const StvRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    StvNaiveState* st = static_cast<StvNaiveState*>(state);
    if (!st || !run || !inputs || !outputs) return cudaErrorInvalidValue;
    if (!stv_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const StvInputs* in = static_cast<const StvInputs*>(inputs);
    StvOutputs* out = static_cast<StvOutputs*>(outputs);
    const StvProblemSpec& sp = st->spec;
    const int A = run->active_count;
    const int N = sp.max_nodes;

    StvNaiveWs ws = stv_nv_layout(workspace, &sp);
    if (workspace_bytes < ws.required_bytes) return cudaErrorInvalidValue;

    cudaError_t err;

    nv_admin_pre_kernel<<<1, 1, 0, stream>>>(
        in->node_count, st->page_used, st->counters, ws.scratch_pages,
        A, sp.page_size, st->maxnpg, sp.max_pages);

    if (A > 0) {
        const size_t y_floats = (size_t)A * N * sp.Hq * sp.D;
        const size_t lse_floats = (size_t)A * N * sp.Hq;
        err = cudaMemsetAsync(out->y, 0, y_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;
        err = cudaMemsetAsync(out->lse, 0, lse_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;

        nv_quantize_kernel<<<A, N + 1, 0, stream>>>(
            in->node_count, in->draft_k, in->draft_v, in->bonus_k, in->bonus_v,
            ws.scratch_pages,
            st->k_pool, st->v_pool, st->k_exp, st->v_exp,
            ws.bonus_exp, ws.bonus_bytes,
            N, sp.Hkv, sp.D, sp.page_size, st->maxnpg);

        nv_verify_kernel<<<A, 1, 0, stream>>>(
            in->node_count, in->parent, in->target_sig, ws.scratch_pages,
            st->k_pool, st->v_pool, st->k_exp, st->v_exp,
            ws.commit_src, ws.commit_count,
            out->accepted_tail, out->accepted_len,
            N, sp.Hkv, sp.D, sp.page_size, st->maxnpg);

        {
            const int nq = A * N * sp.Hq;
            nv_attention_kernel<<<(nq + 31) / 32, 32, 0, stream>>>(
                in->active_seq, in->node_count, in->parent, in->q,
                st->seq_len, st->page_table, ws.scratch_pages,
                st->k_pool, st->v_pool, st->k_exp, st->v_exp,
                out->y, out->lse,
                N, sp.Hq, sp.Hkv, sp.D, sp.page_size, st->max_lp,
                st->maxnpg, nq);
        }

        nv_commit_plan_kernel<<<1, 1, 0, stream>>>(
            in->active_seq, in->node_count, ws.commit_src, ws.commit_count,
            st->seq_len, st->page_table, st->page_used, st->counters,
            ws.scratch_pages, ws.commit_page, ws.commit_slot,
            A, N, sp.page_size, st->max_lp, st->maxnpg, sp.max_pages);

        {
            dim3 grd(A, N + 1);
            nv_commit_copy_kernel<<<grd, 64, 0, stream>>>(
                ws.commit_src, ws.commit_count, ws.commit_page, ws.commit_slot,
                ws.scratch_pages, ws.bonus_exp, ws.bonus_bytes,
                st->k_pool, st->v_pool, st->k_exp, st->v_exp,
                N, sp.Hkv, sp.D, sp.page_size, st->maxnpg);
        }
    }

    nv_hash_kernel<<<(sp.B + 31) / 32, 32, 0, stream>>>(
        st->seq_len, st->page_table,
        st->k_pool, st->v_pool, st->k_exp, st->v_exp,
        out->kv_hash,
        sp.B, sp.Hkv, sp.D, sp.page_size, st->max_lp);

    nv_final_kernel<<<1, 1, 0, stream>>>(
        st->seq_len, st->page_table, st->page_used, st->counters,
        out->seq_len, out->page_state_checksum,
        out->free_pages, out->total_allocs, out->total_frees,
        sp.B, sp.Hq, sp.Hkv, sp.D, sp.page_size, N, sp.max_seq_len,
        sp.max_pages, st->max_lp);

    return cudaGetLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    StvNaiveState* st = static_cast<StvNaiveState*>(state);
    if (!st) return cudaErrorInvalidValue;
    nv_reset_kernel<<<8, 128, 0, stream>>>(
        st->seq_len, st->page_table, st->page_used, st->counters,
        st->spec.B, st->max_lp, st->spec.max_pages);
    return cudaGetLastError();
}

extern "C" void solution_destroy(void* state) {
    StvNaiveState* st = static_cast<StvNaiveState*>(state);
    if (!st) return;
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
