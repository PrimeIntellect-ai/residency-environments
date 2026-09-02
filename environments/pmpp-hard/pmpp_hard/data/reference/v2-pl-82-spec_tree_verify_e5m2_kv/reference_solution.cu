// PMPP_CANARY_82_4a8f31c6d9 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: spec_tree_verify_e5m2_kv_reference.cu
//
// Optimization notes (held out from solvers):
//
//  * All heavy phases are fully parallel: one block per (row, node) for
//    quantization (warp per head/section, shuffle amax, bit-twiddled E5M2
//    encode), one warp per (node, query-head) for attention with char4
//    quantized loads and inline dequant, per-split partial records merged by
//    LSE rescaling, and block-parallel commit byte copies driven by a
//    precomputed plan.
//  * Node signatures are folded one thread per node (~1 KB serial each, all
//    nodes in parallel); the committed kv_hash is INCREMENTAL: a per-sequence
//    running hash is extended only by this step's committed tokens (the
//    contract folds seq_len LAST), never refolded from scratch.
//  * The serial floor is confined to the tiny admin kernels: lowest-free-id
//    scratch/commit allocation over a 64-bit bitmap and the acceptance
//    cascade over <= 64 nodes.
// ============================================================================

#include "spec_tree_verify_e5m2_kv_common.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t kFnvBasis = 1469598103934665603ULL;

// ---------------------------------------------------------------------------
// State + workspace
// ---------------------------------------------------------------------------

struct StvRefState {
    StvProblemSpec spec;
    int max_lp;
    int maxnpg;      // ceil(max_nodes / P)
    int n_splits;
    int group;       // Hq / Hkv

    uint8_t* k_pool;   // [((page*Hkv + h)*P + slot)*D + d]
    uint8_t* v_pool;
    int8_t* k_exp;     // [(page*Hkv + h)*P + slot]
    int8_t* v_exp;

    int32_t* page_table;   // [B * max_lp]
    int32_t* seq_len;      // [B]
    uint64_t* h_run;       // [B]
    uint64_t* free_words;  // bitmap, bit=1 free
    int32_t* counters;     // [0]=step [1]=allocs [2]=frees
    int n_words;
};

static int stv_ref_splits(const StvProblemSpec* spec) {
    if (spec->max_seq_len >= 2048) return 4;
    if (spec->max_seq_len >= 512) return 2;
    return 1;
}

struct StvRefWs {
    int32_t* scratch_pages;  // [B * maxnpg]
    int8_t* bonus_exp;       // [B * Hkv * 2]
    uint8_t* bonus_bytes;    // [B * Hkv * 2 * D]
    uint64_t* sig;           // [B * N]
    int32_t* pathbuf;        // [B * N * N]
    int32_t* path_len;       // [B * N]
    int32_t* commit_src;     // [B * (N+1)]  node idx or -2 (bonus)
    int32_t* commit_page;    // [B * (N+1)]
    int32_t* commit_slot;    // [B * (N+1)]
    int32_t* commit_count;   // [B]
    float* partials;         // [B*N*Hq * S * (D+2)]
    size_t required_bytes;
};

static StvRefWs stv_ref_layout(void* ws, const StvProblemSpec* spec) {
    StvRefWs lay{};
    char* base = static_cast<char*>(ws);
    const int B = spec->B;
    const int N = spec->max_nodes;
    const int Hkv = spec->Hkv;
    const int D = spec->D;
    const int S = stv_ref_splits(spec);
    const int maxnpg = stv_ceil_div_int(N, spec->page_size);
    size_t off = 0;

    #define STV_WS(field, type, count)                                        \
        off = stv_align_up_size(off, 128);                                    \
        lay.field = reinterpret_cast<type*>(base + off);                      \
        off += sizeof(type) * (count)

    STV_WS(scratch_pages, int32_t, (size_t)B * maxnpg);
    STV_WS(bonus_exp, int8_t, (size_t)B * Hkv * 2);
    STV_WS(bonus_bytes, uint8_t, (size_t)B * Hkv * 2 * D);
    STV_WS(sig, uint64_t, (size_t)B * N);
    STV_WS(pathbuf, int32_t, (size_t)B * N * N);
    STV_WS(path_len, int32_t, (size_t)B * N);
    STV_WS(commit_src, int32_t, (size_t)B * (N + 1));
    STV_WS(commit_page, int32_t, (size_t)B * (N + 1));
    STV_WS(commit_slot, int32_t, (size_t)B * (N + 1));
    STV_WS(commit_count, int32_t, (size_t)B);
    STV_WS(partials, float,
           (size_t)B * N * spec->Hq * S * (size_t)(D + 2));
    #undef STV_WS

    off = stv_align_up_size(off, 128);
    lay.required_bytes = off;
    return lay;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t stv_fnv_byte_d(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ __forceinline__ uint64_t stv_fnv_i32_d(uint64_t h, int32_t v) {
    const uint32_t u = static_cast<uint32_t>(v);
    h = stv_fnv_byte_d(h, (uint8_t)(u & 0xFF));
    h = stv_fnv_byte_d(h, (uint8_t)((u >> 8) & 0xFF));
    h = stv_fnv_byte_d(h, (uint8_t)((u >> 16) & 0xFF));
    h = stv_fnv_byte_d(h, (uint8_t)((u >> 24) & 0xFF));
    return h;
}

__device__ __forceinline__ float stv_pow2f(int e) {
    return __uint_as_float(static_cast<uint32_t>(e + 127) << 23);
}

// Exact E5M2 decode (finite bytes).
__device__ __forceinline__ float stv_e5m2_decode_d(uint8_t b) {
    const uint32_t sign = (static_cast<uint32_t>(b) & 0x80u) << 24;
    const uint32_t E = (b >> 2) & 0x1F;
    const uint32_t M = b & 0x3;
    if (E == 0) {
        const float mag = static_cast<float>(M) * 0.0000152587890625f;  // M * 2^-16
        return __uint_as_float(sign | __float_as_uint(mag));
    }
    return __uint_as_float(sign | ((E + 112u) << 23) | (M << 21));
}

// Exact E5M2 encode per the contract.
__device__ __forceinline__ uint8_t stv_e5m2_encode_d(float z) {
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
    // subnormal target: round a / 2^-16 to an integer in [0, 4], RNE
    const float t = a * 65536.0f;
    int i = static_cast<int>(t);
    const float f = t - static_cast<float>(i);
    if (f > 0.5f || (f == 0.5f && (i & 1))) i++;
    if (i >= 4) return sign | 0x04;
    return sign | static_cast<uint8_t>(i);
}

__device__ __forceinline__ size_t stv_slot_head(
    int page, int slot, int kvh, int Hkv, int P) {
    return ((size_t)page * Hkv + (size_t)kvh) * (size_t)P + (size_t)slot;
}

// ---------------------------------------------------------------------------
// Kernel 1: admin-pre (single thread): step counter + scratch allocation.
// ---------------------------------------------------------------------------

__global__ void stv_admin_pre_kernel(
    const int32_t* __restrict__ node_count,
    uint64_t* __restrict__ free_words,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ scratch_pages,
    int active_count,
    int P,
    int maxnpg,
    int n_words) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    counters[0] += 1;

    for (int a = 0; a < active_count; ++a) {
        const int npg = (node_count[a] + P - 1) / P;
        for (int g = 0; g < npg; ++g) {
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
            scratch_pages[a * maxnpg + g] = page;
            counters[1] += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: quantize nodes into scratch. Block per (a, node), warp per
// (head, K/V section) round-robin.
// ---------------------------------------------------------------------------

__global__ void stv_quantize_nodes_kernel(
    const int32_t* __restrict__ node_count,
    const float* __restrict__ draft_k,
    const float* __restrict__ draft_v,
    const int32_t* __restrict__ scratch_pages,
    uint8_t* __restrict__ k_pool,
    uint8_t* __restrict__ v_pool,
    int8_t* __restrict__ k_exp,
    int8_t* __restrict__ v_exp,
    int N,
    int Hkv,
    int D,
    int P,
    int maxnpg) {
    const int a = blockIdx.x / N;
    const int i = blockIdx.x - a * N;
    if (i >= node_count[a]) return;

    const int page = scratch_pages[a * maxnpg + i / P];
    const int slot = i % P;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int per_lane = D >> 5;  // 2 for D=64, 4 for D=128

    const int n_jobs = Hkv * 2;
    for (int job = warp; job < n_jobs; job += nwarps) {
        const int h = job >> 1;
        const bool is_k = (job & 1) == 0;
        const float* x = (is_k ? draft_k : draft_v) +
            (((size_t)a * N + i) * Hkv + h) * (size_t)D;

        float amax = 0.0f;
        float vals[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int d = lane + (r << 5);
            vals[r] = (r < per_lane) ? x[d] : 0.0f;
            amax = fmaxf(amax, fabsf(vals[r]));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        }

        const size_t head_slot = stv_slot_head(page, slot, h, Hkv, P);
        uint8_t* pool = is_k ? k_pool : v_pool;
        int8_t* exps = is_k ? k_exp : v_exp;

        int8_t se8 = 0;
        float inv_scale = 0.0f;
        if (amax != 0.0f) {
            const int kfloor =
                (int)((__float_as_uint(amax) >> 23) & 0xFF) - 127;
            int se = kfloor - 15;
            se = se < -110 ? -110 : (se > 110 ? 110 : se);
            se8 = (int8_t)se;
            inv_scale = stv_pow2f(-se);
        }
        if (lane == 0) exps[head_slot] = se8;

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r < per_lane) {
                const int d = lane + (r << 5);
                uint8_t byte;
                if (amax == 0.0f) {
                    byte = (__float_as_uint(vals[r]) & 0x80000000u) ? 0x80 : 0x00;
                } else {
                    byte = stv_e5m2_encode_d(vals[r] * inv_scale);
                }
                pool[head_slot * D + d] = byte;
            }
        }
    }
}

// Bonus tokens quantized into workspace staging. Block per row.
__global__ void stv_quantize_bonus_kernel(
    const float* __restrict__ bonus_k,
    const float* __restrict__ bonus_v,
    int8_t* __restrict__ bonus_exp,     // [A, Hkv, 2]
    uint8_t* __restrict__ bonus_bytes,  // [A, Hkv, 2, D]
    int Hkv,
    int D) {
    const int a = blockIdx.x;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int per_lane = D >> 5;

    const int n_jobs = Hkv * 2;
    for (int job = warp; job < n_jobs; job += nwarps) {
        const int h = job >> 1;
        const bool is_k = (job & 1) == 0;
        const float* x = (is_k ? bonus_k : bonus_v) +
            ((size_t)a * Hkv + h) * (size_t)D;

        float amax = 0.0f;
        float vals[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int d = lane + (r << 5);
            vals[r] = (r < per_lane) ? x[d] : 0.0f;
            amax = fmaxf(amax, fabsf(vals[r]));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        }

        int8_t se8 = 0;
        float inv_scale = 0.0f;
        if (amax != 0.0f) {
            const int kfloor =
                (int)((__float_as_uint(amax) >> 23) & 0xFF) - 127;
            int se = kfloor - 15;
            se = se < -110 ? -110 : (se > 110 ? 110 : se);
            se8 = (int8_t)se;
            inv_scale = stv_pow2f(-se);
        }
        const int sec = is_k ? 0 : 1;
        if (lane == 0) bonus_exp[((size_t)a * Hkv + h) * 2 + sec] = se8;

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (r < per_lane) {
                const int d = lane + (r << 5);
                uint8_t byte;
                if (amax == 0.0f) {
                    byte = (__float_as_uint(vals[r]) & 0x80000000u) ? 0x80 : 0x00;
                } else {
                    byte = stv_e5m2_encode_d(vals[r] * inv_scale);
                }
                bonus_bytes[(((size_t)a * Hkv + h) * 2 + sec) * D + d] = byte;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: per-node signature + ancestor paths + acceptance cascade.
// Block per row; thread per node for sig/path, thread 0 for the cascade.
// ---------------------------------------------------------------------------

__global__ void stv_sig_verify_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ node_count,
    const int32_t* __restrict__ parent,
    const uint64_t* __restrict__ target_sig,
    const int32_t* __restrict__ scratch_pages,
    const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool,
    const int8_t* __restrict__ k_exp,
    const int8_t* __restrict__ v_exp,
    uint64_t* __restrict__ sig_out,
    int32_t* __restrict__ pathbuf,
    int32_t* __restrict__ path_len,
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
    const int n = node_count[a];
    const int i = threadIdx.x;

    __shared__ bool sh_acc[STV_MAX_NODES_LIMIT];
    __shared__ int sh_depth[STV_MAX_NODES_LIMIT];

    if (i < n) {
        // Signature: TOKEN FOLD over this node's scratch bytes.
        const int page = scratch_pages[a * maxnpg + i / P];
        const int slot = i % P;
        uint64_t h = kFnvBasis;
        for (int hh = 0; hh < Hkv; ++hh) {
            const size_t hs = stv_slot_head(page, slot, hh, Hkv, P);
            h = stv_fnv_byte_d(h, (uint8_t)k_exp[hs]);
            for (int d = 0; d < D; ++d) {
                h = stv_fnv_byte_d(h, k_pool[hs * D + d]);
            }
            h = stv_fnv_byte_d(h, (uint8_t)v_exp[hs]);
            for (int d = 0; d < D; ++d) {
                h = stv_fnv_byte_d(h, v_pool[hs * D + d]);
            }
        }
        sig_out[(size_t)a * N + i] = h;

        // Ancestor path root..i.
        int chain[STV_MAX_NODES_LIMIT];
        int len = 0;
        int cur = i;
        while (cur != -1) {
            chain[len++] = cur;
            cur = parent[(size_t)a * N + cur];
        }
        path_len[(size_t)a * N + i] = len;
        int32_t* pb = pathbuf + ((size_t)a * N + i) * N;
        for (int k = 0; k < len; ++k) {
            pb[k] = chain[len - 1 - k];
        }
    }
    __syncthreads();

    if (i == 0) {
        int best = -1;
        int best_depth = 0;
        for (int j = 0; j < n; ++j) {
            const int par = parent[(size_t)a * N + j];
            const bool par_ok = (par == -1) ? true : sh_acc[par];
            const bool ok = par_ok &&
                (sig_out[(size_t)a * N + j] == target_sig[(size_t)a * N + j]);
            sh_acc[j] = ok;
            sh_depth[j] = (par == -1) ? 1 : sh_depth[par] + 1;
            if (ok && sh_depth[j] > best_depth) {
                best_depth = sh_depth[j];
                best = j;
            }
        }
        out_accepted_tail[a] = best;
        out_accepted_len[a] = best < 0 ? 0 : best_depth;

        // Commit source list: accepted path root..tail, then bonus (-2).
        int32_t* cs = commit_src + (size_t)a * (N + 1);
        int cnt = 0;
        if (best >= 0) {
            const int32_t* pb = pathbuf + ((size_t)a * N + best) * N;
            for (int k = 0; k < best_depth; ++k) cs[cnt++] = pb[k];
        }
        cs[cnt++] = -2;
        commit_count[a] = cnt;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: attention. Grid ((a*N + node), kvh, split); 4 warps; warps
// round-robin the query heads of the kv group. Split 0 also covers the
// ancestor path (scratch tokens).
// ---------------------------------------------------------------------------

#define STV_ATTN_WARPS 4

__global__ void stv_attention_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ node_count,
    const float* __restrict__ q,
    const int32_t* __restrict__ seq_len,     // pre-commit
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ scratch_pages,
    const int32_t* __restrict__ pathbuf,
    const int32_t* __restrict__ path_len,
    const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool,
    const int8_t* __restrict__ k_exp,
    const int8_t* __restrict__ v_exp,
    float* __restrict__ partials,            // [(a*N+i)*Hq + hq][S][D+2]
    int N,
    int Hq,
    int Hkv,
    int D,
    int P,
    int max_lp,
    int maxnpg,
    int n_splits,
    float sm_scale) {
    const int slot_id = blockIdx.x;           // a*N + i
    const int a = slot_id / N;
    const int i = slot_id - a * N;
    if (i >= node_count[a]) return;
    const int kvh = blockIdx.y;
    const int split = blockIdx.z;

    const int b = active_seq[a];
    const int L = seq_len[b];
    const int group = Hq / Hkv;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int per_lane = D >> 5;

    const int chunk = (L + n_splits - 1) / n_splits;
    const int t0 = split * chunk;
    const int t1 = (t0 + chunk) < L ? (t0 + chunk) : L;

    const int plen = path_len[(size_t)a * N + i];
    const int32_t* pb = pathbuf + ((size_t)a * N + i) * N;

    for (int hq_off = warp; hq_off < group; hq_off += STV_ATTN_WARPS) {
        const int hq = kvh * group + hq_off;

        // Query registers (lane covers dims lane + 32*r).
        float qv[4];
        {
            const float* qp = q + (((size_t)a * N + i) * Hq + hq) * (size_t)D;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int d = lane + (r << 5);
                qv[r] = (r < per_lane) ? qp[d] : 0.0f;
            }
        }

        float m = -CUDART_INF_F;
        float l = 0.0f;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        const int total_here = (t1 - t0) + (split == 0 ? plen : 0);
        for (int tt = 0; tt < total_here; ++tt) {
            int page, slot;
            if (tt < t1 - t0) {
                const int t = t0 + tt;
                page = page_table[(size_t)b * max_lp + t / P];
                slot = t % P;
            } else {
                const int node = pb[tt - (t1 - t0)];
                page = scratch_pages[a * maxnpg + node / P];
                slot = node % P;
            }
            const size_t hs = stv_slot_head(page, slot, kvh, Hkv, P);
            const float ksc = stv_pow2f((int)k_exp[hs]);

            float part = 0.0f;
            float kv[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r < per_lane) {
                    const int d = lane + (r << 5);
                    kv[r] = stv_e5m2_decode_d(k_pool[hs * D + d]) * ksc;
                    part += qv[r] * kv[r];
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

            const float vsc = stv_pow2f((int)v_exp[hs]);
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r < per_lane) {
                    const int d = lane + (r << 5);
                    const float vv = stv_e5m2_decode_d(v_pool[hs * D + d]) * vsc;
                    acc[r] = acc[r] * alpha + beta * vv;
                }
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
// Kernel 5: merge splits, write y / lse.
// ---------------------------------------------------------------------------

__global__ void stv_merge_kernel(
    const int32_t* __restrict__ node_count,
    const float* __restrict__ partials,
    float* __restrict__ y,
    float* __restrict__ lse,
    int N,
    int Hq,
    int D,
    int n_splits) {
    const int flat = blockIdx.x;               // (a*N + i) * Hq + hq
    const int slot_id = flat / Hq;
    const int a = slot_id / N;
    const int i = slot_id - a * N;
    if (i >= node_count[a]) return;

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
// Kernel 6: commit plan (single thread): committed page allocation with
// scratch still held, seq_len updates, then scratch release.
// ---------------------------------------------------------------------------

__global__ void stv_commit_plan_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ node_count,
    const int32_t* __restrict__ commit_src,
    const int32_t* __restrict__ commit_count,
    int32_t* __restrict__ seq_len,
    int32_t* __restrict__ page_table,
    uint64_t* __restrict__ free_words,
    int32_t* __restrict__ counters,
    const int32_t* __restrict__ scratch_pages,
    int32_t* __restrict__ commit_page,
    int32_t* __restrict__ commit_slot,
    int active_count,
    int N,
    int P,
    int max_lp,
    int maxnpg,
    int n_words) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int a = 0; a < active_count; ++a) {
        const int b = active_seq[a];
        const int cnt = commit_count[a];
        for (int c = 0; c < cnt; ++c) {
            const int pos = seq_len[b];
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
                page_table[(size_t)b * max_lp + lp] = page;
                counters[1] += 1;
            }
            commit_page[(size_t)a * (N + 1) + c] =
                page_table[(size_t)b * max_lp + lp];
            commit_slot[(size_t)a * (N + 1) + c] = pos % P;
            seq_len[b] += 1;
        }
    }

    // Release scratch.
    for (int a = 0; a < active_count; ++a) {
        const int npg = (node_count[a] + P - 1) / P;
        for (int g = 0; g < npg; ++g) {
            const int page = scratch_pages[a * maxnpg + g];
            free_words[page >> 6] |= (1ULL << (page & 63));
            counters[2] += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 7: commit byte copies (block per (row, committed token)).
// ---------------------------------------------------------------------------

__global__ void stv_commit_copy_kernel(
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
        const size_t dst_hs = stv_slot_head(dpage, dslot, hh, Hkv, P);
        if (src >= 0) {
            const int spage = scratch_pages[a * maxnpg + src / P];
            const int sslot = src % P;
            const size_t src_hs = stv_slot_head(spage, sslot, hh, Hkv, P);
            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                k_pool[dst_hs * D + d] = k_pool[src_hs * D + d];
                v_pool[dst_hs * D + d] = v_pool[src_hs * D + d];
            }
            if (threadIdx.x == 0) {
                k_exp[dst_hs] = k_exp[src_hs];
                v_exp[dst_hs] = v_exp[src_hs];
            }
        } else {
            for (int d = threadIdx.x; d < D; d += blockDim.x) {
                k_pool[dst_hs * D + d] =
                    bonus_bytes[(((size_t)a * Hkv + hh) * 2 + 0) * D + d];
                v_pool[dst_hs * D + d] =
                    bonus_bytes[(((size_t)a * Hkv + hh) * 2 + 1) * D + d];
            }
            if (threadIdx.x == 0) {
                k_exp[dst_hs] = bonus_exp[((size_t)a * Hkv + hh) * 2 + 0];
                v_exp[dst_hs] = bonus_exp[((size_t)a * Hkv + hh) * 2 + 1];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 8: extend running hashes with this step's committed tokens.
// One thread per active row.
// ---------------------------------------------------------------------------

__global__ void stv_hash_extend_kernel(
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ commit_count,
    const int32_t* __restrict__ commit_page,
    const int32_t* __restrict__ commit_slot,
    const uint8_t* __restrict__ k_pool,
    const uint8_t* __restrict__ v_pool,
    const int8_t* __restrict__ k_exp,
    const int8_t* __restrict__ v_exp,
    uint64_t* __restrict__ h_run,
    int active_count,
    int N,
    int Hkv,
    int D,
    int P) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= active_count) return;

    const int b = active_seq[a];
    uint64_t h = h_run[b];
    const int cnt = commit_count[a];
    for (int c = 0; c < cnt; ++c) {
        const int page = commit_page[(size_t)a * (N + 1) + c];
        const int slot = commit_slot[(size_t)a * (N + 1) + c];
        for (int hh = 0; hh < Hkv; ++hh) {
            const size_t hs = stv_slot_head(page, slot, hh, Hkv, P);
            h = stv_fnv_byte_d(h, (uint8_t)k_exp[hs]);
            for (int d = 0; d < D; ++d) {
                h = stv_fnv_byte_d(h, k_pool[hs * D + d]);
            }
            h = stv_fnv_byte_d(h, (uint8_t)v_exp[hs]);
            for (int d = 0; d < D; ++d) {
                h = stv_fnv_byte_d(h, v_pool[hs * D + d]);
            }
        }
    }
    h_run[b] = h;
}

// ---------------------------------------------------------------------------
// Kernel 9: final admin: exact outputs.
// ---------------------------------------------------------------------------

__global__ void stv_final_kernel(
    const int32_t* __restrict__ seq_len,
    const uint64_t* __restrict__ h_run,
    const int32_t* __restrict__ page_table,
    const uint64_t* __restrict__ free_words,
    const int32_t* __restrict__ counters,
    int32_t* __restrict__ out_seq_len,
    uint64_t* __restrict__ out_kv_hash,
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
    int max_lp,
    int n_words) {
    const int tid = threadIdx.x;

    if (tid < B) {
        out_seq_len[tid] = seq_len[tid];
        out_kv_hash[tid] = stv_fnv_i32_d(h_run[tid], seq_len[tid]);
    }

    if (tid == 0) {
        uint64_t ph = kFnvBasis;
        ph = stv_fnv_i32_d(ph, B);
        ph = stv_fnv_i32_d(ph, Hq);
        ph = stv_fnv_i32_d(ph, Hkv);
        ph = stv_fnv_i32_d(ph, D);
        ph = stv_fnv_i32_d(ph, P);
        ph = stv_fnv_i32_d(ph, N);
        ph = stv_fnv_i32_d(ph, msl);
        ph = stv_fnv_i32_d(ph, max_pages);
        ph = stv_fnv_i32_d(ph, counters[0]);
        for (int b = 0; b < B; ++b) {
            ph = stv_fnv_i32_d(ph, seq_len[b]);
            const int n_lp = (seq_len[b] + P - 1) / P;
            ph = stv_fnv_i32_d(ph, n_lp);
            for (int lp = 0; lp < n_lp; ++lp) {
                ph = stv_fnv_i32_d(ph, page_table[(size_t)b * max_lp + lp]);
            }
        }

        int freep = 0;
        for (int w = 0; w < n_words; ++w) {
            freep += __popcll(free_words[w]);
        }
        // Bits beyond max_pages are never set (init clears them).

        ph = stv_fnv_i32_d(ph, counters[1]);
        ph = stv_fnv_i32_d(ph, counters[2]);
        ph = stv_fnv_i32_d(ph, freep);
        out_pcs[0] = ph;
        out_free_pages[0] = freep;
        out_total_allocs[0] = counters[1];
        out_total_frees[0] = counters[2];
    }
}

__global__ void stv_reset_kernel(
    int32_t* __restrict__ seq_len,
    uint64_t* __restrict__ h_run,
    int32_t* __restrict__ page_table,
    uint64_t* __restrict__ free_words,
    int32_t* __restrict__ counters,
    int B,
    int max_lp,
    int max_pages,
    int n_words) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) {
        seq_len[i] = 0;
        h_run[i] = kFnvBasis;
    }
    for (int j = i; j < B * max_lp; j += gridDim.x * blockDim.x) {
        page_table[j] = -1;
    }
    for (int w = i; w < n_words; w += gridDim.x * blockDim.x) {
        const int base = w * 64;
        uint64_t word = 0ULL;
        for (int bit = 0; bit < 64; ++bit) {
            if (base + bit < max_pages) word |= (1ULL << bit);
        }
        free_words[w] = word;
    }
    if (i < 3) counters[i] = 0;
}

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------

extern "C" size_t solution_workspace_bytes(const StvProblemSpec* spec) {
    if (!stv_validate_problem_spec(spec)) return 0;
    return stv_ref_layout(nullptr, spec).required_bytes;
}

extern "C" cudaError_t solution_init(
    const StvProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!state_out) return cudaErrorInvalidValue;
    *state_out = nullptr;
    if (!stv_validate_problem_spec(spec)) return cudaErrorInvalidValue;

    StvRefState* st = static_cast<StvRefState*>(malloc(sizeof(StvRefState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(*st));
    st->spec = *spec;
    st->max_lp = stv_max_logical_pages(spec->max_seq_len, spec->page_size);
    st->maxnpg = stv_ceil_div_int(spec->max_nodes, spec->page_size);
    st->n_splits = stv_ref_splits(spec);
    st->group = spec->Hq / spec->Hkv;
    st->n_words = (spec->max_pages + 63) / 64;

    const size_t slots =
        (size_t)spec->max_pages * spec->Hkv * spec->page_size;

    cudaError_t err = cudaSuccess;
    #define STV_ALLOC(ptr, bytes)                                             \
        do {                                                                  \
            err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes));      \
            if (err != cudaSuccess) goto fail;                                \
        } while (0)

    STV_ALLOC(st->k_pool, slots * spec->D);
    STV_ALLOC(st->v_pool, slots * spec->D);
    STV_ALLOC(st->k_exp, slots);
    STV_ALLOC(st->v_exp, slots);
    STV_ALLOC(st->page_table, sizeof(int32_t) * (size_t)spec->B * st->max_lp);
    STV_ALLOC(st->seq_len, sizeof(int32_t) * spec->B);
    STV_ALLOC(st->h_run, sizeof(uint64_t) * spec->B);
    STV_ALLOC(st->free_words, sizeof(uint64_t) * st->n_words);
    STV_ALLOC(st->counters, sizeof(int32_t) * 4);
    #undef STV_ALLOC

    stv_reset_kernel<<<8, 128, 0, stream>>>(
        st->seq_len, st->h_run, st->page_table, st->free_words, st->counters,
        spec->B, st->max_lp, spec->max_pages, st->n_words);
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
    if (st->h_run) cudaFree(st->h_run);
    if (st->free_words) cudaFree(st->free_words);
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
    StvRefState* st = static_cast<StvRefState*>(state);
    if (!st || !run || !inputs || !outputs) return cudaErrorInvalidValue;
    if (!stv_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const StvInputs* in = static_cast<const StvInputs*>(inputs);
    StvOutputs* out = static_cast<StvOutputs*>(outputs);
    const StvProblemSpec& sp = st->spec;
    const int A = run->active_count;
    const int N = sp.max_nodes;

    StvRefWs ws = stv_ref_layout(workspace, &sp);
    if (workspace_bytes < ws.required_bytes) return cudaErrorInvalidValue;

    cudaError_t err;

    stv_admin_pre_kernel<<<1, 1, 0, stream>>>(
        in->node_count, st->free_words, st->counters, ws.scratch_pages,
        A, sp.page_size, st->maxnpg, st->n_words);

    if (A > 0) {
        const size_t y_floats = (size_t)A * N * sp.Hq * sp.D;
        const size_t lse_floats = (size_t)A * N * sp.Hq;
        err = cudaMemsetAsync(out->y, 0, y_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;
        err = cudaMemsetAsync(out->lse, 0, lse_floats * sizeof(float), stream);
        if (err != cudaSuccess) return err;

        stv_quantize_nodes_kernel<<<A * N, 128, 0, stream>>>(
            in->node_count, in->draft_k, in->draft_v, ws.scratch_pages,
            st->k_pool, st->v_pool, st->k_exp, st->v_exp,
            N, sp.Hkv, sp.D, sp.page_size, st->maxnpg);

        stv_quantize_bonus_kernel<<<A, 128, 0, stream>>>(
            in->bonus_k, in->bonus_v, ws.bonus_exp, ws.bonus_bytes,
            sp.Hkv, sp.D);

        stv_sig_verify_kernel<<<A, ((N + 31) / 32) * 32, 0, stream>>>(
            in->active_seq, in->node_count, in->parent, in->target_sig,
            ws.scratch_pages,
            st->k_pool, st->v_pool, st->k_exp, st->v_exp,
            ws.sig, ws.pathbuf, ws.path_len,
            ws.commit_src, ws.commit_count,
            out->accepted_tail, out->accepted_len,
            N, sp.Hkv, sp.D, sp.page_size, st->maxnpg);

        {
            dim3 grd(A * N, sp.Hkv, st->n_splits);
            const float sm_scale = rsqrtf((float)sp.D);
            stv_attention_kernel<<<grd, 32 * STV_ATTN_WARPS, 0, stream>>>(
                in->active_seq, in->node_count, in->q,
                st->seq_len, st->page_table, ws.scratch_pages,
                ws.pathbuf, ws.path_len,
                st->k_pool, st->v_pool, st->k_exp, st->v_exp,
                ws.partials,
                N, sp.Hq, sp.Hkv, sp.D, sp.page_size, st->max_lp,
                st->maxnpg, st->n_splits, sm_scale);
        }

        stv_merge_kernel<<<A * N * sp.Hq, 64, 0, stream>>>(
            in->node_count, ws.partials, out->y, out->lse,
            N, sp.Hq, sp.D, st->n_splits);

        stv_commit_plan_kernel<<<1, 1, 0, stream>>>(
            in->active_seq, in->node_count, ws.commit_src, ws.commit_count,
            st->seq_len, st->page_table, st->free_words, st->counters,
            ws.scratch_pages, ws.commit_page, ws.commit_slot,
            A, N, sp.page_size, st->max_lp, st->maxnpg, st->n_words);

        {
            dim3 grd(A, N + 1);
            stv_commit_copy_kernel<<<grd, 128, 0, stream>>>(
                ws.commit_src, ws.commit_count, ws.commit_page, ws.commit_slot,
                ws.scratch_pages, ws.bonus_exp, ws.bonus_bytes,
                st->k_pool, st->v_pool, st->k_exp, st->v_exp,
                N, sp.Hkv, sp.D, sp.page_size, st->maxnpg);
        }

        stv_hash_extend_kernel<<<1, 32, 0, stream>>>(
            in->active_seq, ws.commit_count, ws.commit_page, ws.commit_slot,
            st->k_pool, st->v_pool, st->k_exp, st->v_exp,
            st->h_run, A, N, sp.Hkv, sp.D, sp.page_size);
    }

    stv_final_kernel<<<1, 64, 0, stream>>>(
        st->seq_len, st->h_run, st->page_table, st->free_words, st->counters,
        out->seq_len, out->kv_hash, out->page_state_checksum,
        out->free_pages, out->total_allocs, out->total_frees,
        sp.B, sp.Hq, sp.Hkv, sp.D, sp.page_size, N, sp.max_seq_len,
        sp.max_pages, st->max_lp, st->n_words);

    return cudaGetLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    StvRefState* st = static_cast<StvRefState*>(state);
    if (!st) return cudaErrorInvalidValue;
    stv_reset_kernel<<<8, 128, 0, stream>>>(
        st->seq_len, st->h_run, st->page_table, st->free_words, st->counters,
        st->spec.B, st->max_lp, st->spec.max_pages, st->n_words);
    return cudaGetLastError();
}

extern "C" void solution_destroy(void* state) {
    StvRefState* st = static_cast<StvRefState*>(state);
    if (!st) return;
    if (st->k_pool) cudaFree(st->k_pool);
    if (st->v_pool) cudaFree(st->v_pool);
    if (st->k_exp) cudaFree(st->k_exp);
    if (st->v_exp) cudaFree(st->v_exp);
    if (st->page_table) cudaFree(st->page_table);
    if (st->seq_len) cudaFree(st->seq_len);
    if (st->h_run) cudaFree(st->h_run);
    if (st->free_words) cudaFree(st->free_words);
    if (st->counters) cudaFree(st->counters);
    free(st);
}
