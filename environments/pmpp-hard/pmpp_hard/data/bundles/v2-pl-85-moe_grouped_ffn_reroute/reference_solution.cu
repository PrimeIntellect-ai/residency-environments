// PMPP_CANARY_85_d48e8bfe17 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: moe_grouped_ffn_reroute_reference.cu
// Strong reference pipeline implementation.
//
// Architecture:
//   1. Router logits: shared-staged int8 GEMM with dp4a (templated on D).
//   2. Routing: one warp per token; group max scores, top-g_sel groups,
//      warp-argmax candidate ranking with exact (logit desc, id asc) ties.
//   3. Dispatch: stable LSD radix sort (6x8-bit passes) over 46-bit route
//      keys (expert | inverted-gate | token | slot), run once per phase.
//      Chunked histograms + warp-synchronous stable scatter.
//   4. Grouped FFN: per-expert tile GEMM (32 routes x {H,D} tiles) with
//      dp4a, shared-memory staging of x rows, w1/w2 tiles and requantized
//      hidden activations; combine via wrapping int64 atomics.
//   5. y_checksum: per-chunk digests in parallel, root folded via the exact
//      affine-FNV chunk composition (never a serial byte fold of the data).
// ============================================================================

#include "moe_grouped_ffn_reroute_common.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Workspace layout.
// ---------------------------------------------------------------------------

#define MGF_SORT_CHUNK 2048
#define MGF_ROOT_CHUNK 256  // digests per affine root chunk

struct MgfReferenceState {
    MgfProblemSpec spec;
};

struct MgfWorkspaceLayout {
    uint64_t* rec_a;
    uint64_t* rec_b;
    uint64_t* rec2_a;
    uint64_t* rec2_b;
    uint32_t* hist;
    uint32_t* scanned;
    int16_t* cand;          // [max_N, 8]
    uint32_t* cnt1;
    uint32_t* kept1;
    uint32_t* seg1;
    uint32_t* cnt2;
    uint32_t* kept2;
    uint32_t* seg2;
    uint32_t* tiles_before; // [max_E + 1]
    uint32_t* total_tiles;  // [1]
    uint32_t* m2;           // [1]
    uint64_t* digests;
    uint64_t* root_T;       // [nRootChunksMax * 256]
    uint64_t* root_A;       // [nRootChunksMax]
    size_t required_bytes;
};

static MgfWorkspaceLayout mgf_reference_make_layout(
    void* workspace,
    const MgfProblemSpec* spec) {
    MgfWorkspaceLayout L{};
    char* base = static_cast<char*>(workspace);
    const size_t maxM = (size_t)spec->max_N * (size_t)spec->max_K;
    const size_t maxChunks = (maxM + MGF_SORT_CHUNK - 1) / MGF_SORT_CHUNK;
    const size_t maxDig = (size_t)mgf_ceil_div_int(spec->max_N, MGF_CSUM_ROWS);
    const size_t maxRoot = (maxDig + MGF_ROOT_CHUNK - 1) / MGF_ROOT_CHUNK;

    size_t off = 0;
#define MGF_WS_FIELD(field, type, count)                                    \
    off = mgf_align_up_size(off, 128);                                      \
    L.field = reinterpret_cast<type*>(base + off);                          \
    off += sizeof(type) * (count);

    MGF_WS_FIELD(rec_a, uint64_t, maxM)
    MGF_WS_FIELD(rec_b, uint64_t, maxM)
    MGF_WS_FIELD(rec2_a, uint64_t, maxM)
    MGF_WS_FIELD(rec2_b, uint64_t, maxM)
    MGF_WS_FIELD(hist, uint32_t, maxChunks * 256)
    MGF_WS_FIELD(scanned, uint32_t, maxChunks * 256)
    MGF_WS_FIELD(cand, int16_t, (size_t)spec->max_N * 8)
    MGF_WS_FIELD(cnt1, uint32_t, (size_t)spec->max_E)
    MGF_WS_FIELD(kept1, uint32_t, (size_t)spec->max_E)
    MGF_WS_FIELD(seg1, uint32_t, (size_t)spec->max_E)
    MGF_WS_FIELD(cnt2, uint32_t, (size_t)spec->max_E)
    MGF_WS_FIELD(kept2, uint32_t, (size_t)spec->max_E)
    MGF_WS_FIELD(seg2, uint32_t, (size_t)spec->max_E)
    MGF_WS_FIELD(tiles_before, uint32_t, (size_t)spec->max_E + 1)
    MGF_WS_FIELD(total_tiles, uint32_t, 1)
    MGF_WS_FIELD(m2, uint32_t, 1)
    MGF_WS_FIELD(digests, uint64_t, maxDig)
    MGF_WS_FIELD(root_T, uint64_t, maxRoot * 256)
    MGF_WS_FIELD(root_A, uint64_t, maxRoot)
#undef MGF_WS_FIELD

    off = mgf_align_up_size(off, 128);
    L.required_bytes = off;
    return L;
}

// ---------------------------------------------------------------------------
// Key encode/decode. 46-bit key:
//   bits [39,45] expert, [17,38] inverted biased gate, [2,16] token, [0,1] slot
// Sorted ascending == (expert asc, gate desc, token asc, slot asc).
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t mgf_make_key(
    int e, int32_t gate, int t, int k) {
    const uint32_t biased = (uint32_t)(gate + MGF_LOGIT_BIAS);
    const uint32_t inv = 0x3FFFFFu - biased;
    return ((uint64_t)e << 39) | ((uint64_t)inv << 17) |
           ((uint64_t)t << 2) | (uint64_t)k;
}

__device__ __forceinline__ void mgf_decode_key(
    uint64_t key, int* e, int32_t* gate, int* t, int* k) {
    *k = (int)(key & 3u);
    *t = (int)((key >> 2) & 0x7FFFu);
    const uint32_t inv = (uint32_t)((key >> 17) & 0x3FFFFFu);
    *gate = (int32_t)(0x3FFFFFu - inv) - MGF_LOGIT_BIAS;
    *e = (int)((key >> 39) & 0x7Fu);
}

// ---------------------------------------------------------------------------
// 1. Router logits GEMM.
// ---------------------------------------------------------------------------

template <int D4>
__global__ void mgf_ref_logits_kernel(
    int N,
    int E,
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ wr,
    int32_t* __restrict__ logits) {
    extern __shared__ uint32_t s_mem[];
    uint32_t* wr_s = s_mem;                     // [E * D4]
    uint32_t* xs = s_mem + E * D4;              // [128 * (D4 + 1)]

    const int base = blockIdx.x * 128;
    const uint32_t* x_u = reinterpret_cast<const uint32_t*>(x);
    const uint32_t* wr_u = reinterpret_cast<const uint32_t*>(wr);

    for (int idx = threadIdx.x; idx < E * D4; idx += 128) {
        wr_s[idx] = wr_u[idx];
    }
    for (int idx = threadIdx.x; idx < 128 * D4; idx += 128) {
        const int r = idx / D4;
        const int i = idx - r * D4;
        const int t = base + r;
        xs[r * (D4 + 1) + i] = t < N ? x_u[(size_t)t * D4 + i] : 0u;
    }
    __syncthreads();

    const int t = base + threadIdx.x;
    if (t >= N) return;

    uint32_t xreg[D4];
#pragma unroll
    for (int i = 0; i < D4; ++i) {
        xreg[i] = xs[threadIdx.x * (D4 + 1) + i];
    }

    int32_t* out_row = logits + (size_t)t * E;
    for (int e0 = 0; e0 < E; e0 += 4) {
        int4 acc4;
        int32_t* acc = reinterpret_cast<int32_t*>(&acc4);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            int32_t a = 0;
            const uint32_t* w_row = wr_s + (e0 + q) * D4;
#pragma unroll
            for (int i = 0; i < D4; ++i) {
                a = __dp4a((int)xreg[i], (int)w_row[i], a);
            }
            acc[q] = a;
        }
        *reinterpret_cast<int4*>(out_row + e0) = acc4;
    }
}

// ---------------------------------------------------------------------------
// 2. Routing: one warp per token.
// ---------------------------------------------------------------------------

__global__ void mgf_ref_route_kernel(
    int N,
    int E,
    int G,
    int g_sel,
    int K,
    const int32_t* __restrict__ logits,
    int16_t* __restrict__ cand,
    uint64_t* __restrict__ rec1,
    uint32_t* __restrict__ cnt1) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int t = blockIdx.x * 4 + warp;
    if (t >= N) return;

    const int S = E / G;
    const int n_sel = g_sel * S;
    const int n_store = min(2 * K, n_sel);

    const int32_t* lrow = logits + (size_t)t * E;

    // Per-lane experts: lane, lane+32, lane+64, lane+96.
    int32_t le[4];
    int avail = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = lane + 32 * i;
        le[i] = e < E ? lrow[e] : INT_MIN;
    }

    // Group scores (max logit per contiguous group), replicated in all lanes.
    int32_t gs[16];
    for (int g = 0; g < G; ++g) {
        int32_t local = INT_MIN;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int e = lane + 32 * i;
            if (e < E && e / S == g && le[i] > local) local = le[i];
        }
#pragma unroll
        for (int s = 16; s >= 1; s >>= 1) {
            const int32_t o = __shfl_xor_sync(0xffffffffu, local, s);
            if (o > local) local = o;
        }
        gs[g] = local;
    }

    // Top g_sel groups by (score desc, id asc), replicated deterministic scan.
    uint32_t sel_mask = 0;
    for (int r = 0; r < g_sel; ++r) {
        int best_g = -1;
        int32_t best_s = INT_MIN;
        for (int g = 0; g < G; ++g) {
            if (sel_mask & (1u << g)) continue;
            if (best_g < 0 || gs[g] > best_s) {
                best_s = gs[g];
                best_g = g;
            }
        }
        sel_mask |= 1u << best_g;
    }

    // Per-lane availability of experts in selected groups.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = lane + 32 * i;
        if (e < E && (sel_mask & (1u << (e / S)))) avail |= 1 << i;
    }

    // Candidate ranking: repeated warp argmax over key
    //   (biased gate << 7) | (127 - expert)  -- maximization gives
    //   (gate desc, id asc). Key 0 means "no candidate on this lane".
    for (int j = 0; j < n_store; ++j) {
        uint32_t local_key = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (avail & (1 << i)) {
                const int e = lane + 32 * i;
                const uint32_t biased = (uint32_t)(le[i] + MGF_LOGIT_BIAS);
                const uint32_t key = (biased << 7) | (uint32_t)(127 - e);
                if (key > local_key) local_key = key;
            }
        }
        uint32_t best = local_key;
#pragma unroll
        for (int s = 16; s >= 1; s >>= 1) {
            const uint32_t o = __shfl_xor_sync(0xffffffffu, best, s);
            if (o > best) best = o;
        }

        const int e_best = 127 - (int)(best & 0x7Fu);
        if ((e_best & 31) == lane) {
            avail &= ~(1 << (e_best >> 5));
        }

        if (lane == 0) {
            cand[(size_t)t * 8 + j] = (int16_t)e_best;
            if (j < K) {
                const int32_t gate =
                    (int32_t)(best >> 7) - MGF_LOGIT_BIAS;
                rec1[(size_t)t * K + j] = mgf_make_key(e_best, gate, t, j);
                atomicAdd(&cnt1[e_best], 1u);
            }
        }
    }
    if (lane == 0) {
        for (int j = n_store; j < 8; ++j) {
            cand[(size_t)t * 8 + j] = -1;
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Stable LSD radix sort (8-bit digits).
// ---------------------------------------------------------------------------

__device__ __forceinline__ int mgf_sort_count(
    int host_count, const uint32_t* dev_count) {
    return dev_count ? (int)*dev_count : host_count;
}

__global__ void mgf_ref_sort_hist_kernel(
    const uint64_t* __restrict__ keys,
    uint32_t* __restrict__ hist,
    int shift,
    int host_count,
    const uint32_t* __restrict__ dev_count) {
    __shared__ uint32_t h_s[256];
    const int count = mgf_sort_count(host_count, dev_count);
    const int chunk = blockIdx.x;
    const int begin = chunk * MGF_SORT_CHUNK;

    for (int i = threadIdx.x; i < 256; i += 256) h_s[i] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < MGF_SORT_CHUNK; i += 256) {
        const int g = begin + i;
        if (g < count) {
            const int b = (int)((keys[g] >> shift) & 0xFFu);
            atomicAdd(&h_s[b], 1u);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 256; i += 256) {
        hist[(size_t)chunk * 256 + i] = h_s[i];
    }
}

__global__ void mgf_ref_sort_scan_kernel(
    const uint32_t* __restrict__ hist,
    uint32_t* __restrict__ scanned,
    int nchunks) {
    __shared__ uint32_t tot_s[256];
    const int b = threadIdx.x;  // one thread per bucket

    uint32_t run = 0;
    for (int c = 0; c < nchunks; ++c) {
        scanned[(size_t)c * 256 + b] = run;
        run += hist[(size_t)c * 256 + b];
    }
    tot_s[b] = run;
    __syncthreads();

    // Exclusive scan of totals across the 256 buckets.
    uint32_t base = 0;
    for (int i = 0; i < b; ++i) base += tot_s[i];
    __syncthreads();

    for (int c = 0; c < nchunks; ++c) {
        scanned[(size_t)c * 256 + b] += base;
    }
}

__global__ void mgf_ref_sort_scatter_kernel(
    const uint64_t* __restrict__ keys_in,
    uint64_t* __restrict__ keys_out,
    const uint32_t* __restrict__ scanned,
    int shift,
    int host_count,
    const uint32_t* __restrict__ dev_count) {
    __shared__ uint32_t warp_hist[4][256];
    __shared__ uint32_t next_s[4][256];

    const int count = mgf_sort_count(host_count, dev_count);
    const int chunk = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int chunk_begin = chunk * MGF_SORT_CHUNK;
    const int seg_begin = chunk_begin + warp * (MGF_SORT_CHUNK / 4);

    for (int i = threadIdx.x; i < 4 * 256; i += 128) {
        (&warp_hist[0][0])[i] = 0;
    }
    __syncthreads();

    // Phase A: per-warp digit counts.
    for (int i = lane; i < MGF_SORT_CHUNK / 4; i += 32) {
        const int g = seg_begin + i;
        if (g < count) {
            const int b = (int)((keys_in[g] >> shift) & 0xFFu);
            atomicAdd(&warp_hist[warp][b], 1u);
        }
    }
    __syncthreads();

    // Phase B: per-warp stable bases.
    for (int b = threadIdx.x; b < 256; b += 128) {
        uint32_t run = scanned[(size_t)chunk * 256 + b];
        for (int w = 0; w < 4; ++w) {
            next_s[w][b] = run;
            run += warp_hist[w][b];
        }
    }
    __syncthreads();

    // Phase C: warp-synchronous stable scatter, 32 elements per batch.
    for (int batch = 0; batch < (MGF_SORT_CHUNK / 4) / 32; ++batch) {
        const int g = seg_begin + batch * 32 + lane;
        const bool v = g < count;
        const unsigned amask = __ballot_sync(0xffffffffu, v);
        uint64_t key = 0;
        int b = 0;
        if (v) {
            key = keys_in[g];
            b = (int)((key >> shift) & 0xFFu);
        }
        const unsigned peers = __match_any_sync(0xffffffffu, b) & amask;
        const int rank = __popc(peers & ((1u << lane) - 1u));
        uint32_t pos_base = 0;
        if (v) pos_base = next_s[warp][b];
        __syncwarp();
        if (v) {
            keys_out[pos_base + rank] = key;
            if (lane == __ffs(peers) - 1) {
                next_s[warp][b] = pos_base + __popc(peers);
            }
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// 4. Segment bookkeeping.
// ---------------------------------------------------------------------------

__global__ void mgf_ref_seg1_kernel(
    int E,
    int cap,
    const uint32_t* __restrict__ cnt1,
    uint32_t* __restrict__ kept1,
    uint32_t* __restrict__ seg1) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t run = 0;
    for (int e = 0; e < E; ++e) {
        seg1[e] = run;
        run += cnt1[e];
        kept1[e] = cnt1[e] < (uint32_t)cap ? cnt1[e] : (uint32_t)cap;
    }
}

// Phase-1 classification: overflow routes either become phase-2 records or
// are dropped with MGF_RS_DROP_NO_BACKUP.
__global__ void mgf_ref_classify_kernel(
    int M,
    int E,
    int K,
    int n_sel,
    const uint64_t* __restrict__ rec1_sorted,
    const uint32_t* __restrict__ kept1,
    const uint32_t* __restrict__ seg1,
    const int16_t* __restrict__ cand,
    const int32_t* __restrict__ logits,
    uint64_t* __restrict__ rec2,
    uint32_t* __restrict__ cnt2,
    uint32_t* __restrict__ m2,
    int16_t* __restrict__ route_expert,
    uint8_t* __restrict__ route_status) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    int e, t, k;
    int32_t gate;
    mgf_decode_key(rec1_sorted[i], &e, &gate, &t, &k);

    const uint32_t r = (uint32_t)i - seg1[e];
    if (r < kept1[e]) return;  // kept phase-1; packed later.

    const int bpos = K + k;
    const size_t rid = (size_t)t * K + k;
    if (bpos >= n_sel) {
        route_expert[rid] = -1;
        route_status[rid] = MGF_RS_DROP_NO_BACKUP;
        return;
    }
    const int b = (int)cand[(size_t)t * 8 + bpos];
    const int32_t gate2 = logits[(size_t)t * E + b];
    const uint32_t idx = atomicAdd(m2, 1u);
    rec2[idx] = mgf_make_key(b, gate2, t, k);
    atomicAdd(&cnt2[b], 1u);
}

__global__ void mgf_ref_finalize_kernel(
    int E,
    int cap,
    const uint32_t* __restrict__ cnt2,
    const uint32_t* __restrict__ kept1,
    uint32_t* __restrict__ kept2,
    uint32_t* __restrict__ seg2,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    uint32_t* __restrict__ tiles_before,
    uint32_t* __restrict__ total_tiles) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t run2 = 0;
    int32_t off = 0;
    uint32_t tiles = 0;
    offsets[0] = 0;
    for (int e = 0; e < E; ++e) {
        seg2[e] = run2;
        run2 += cnt2[e];
        const uint32_t rcap = (uint32_t)cap - kept1[e];
        kept2[e] = cnt2[e] < rcap ? cnt2[e] : rcap;
        const int32_t c = (int32_t)(kept1[e] + kept2[e]);
        counts[e] = c;
        off += c;
        offsets[e + 1] = off;
        tiles_before[e] = tiles;
        tiles += (uint32_t)((c + 31) / 32);
    }
    tiles_before[E] = tiles;
    total_tiles[0] = tiles;
}

__global__ void mgf_ref_pack1_kernel(
    int M,
    const uint64_t* __restrict__ rec1_sorted,
    const uint32_t* __restrict__ kept1,
    const uint32_t* __restrict__ seg1,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ packed_token,
    int32_t* __restrict__ packed_slot,
    int32_t* __restrict__ packed_gate,
    uint8_t* __restrict__ packed_phase,
    int16_t* __restrict__ route_expert,
    uint8_t* __restrict__ route_status,
    int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    int e, t, k;
    int32_t gate;
    mgf_decode_key(rec1_sorted[i], &e, &gate, &t, &k);

    const uint32_t r = (uint32_t)i - seg1[e];
    if (r >= kept1[e]) return;

    const int pos = offsets[e] + (int)r;
    packed_token[pos] = t;
    packed_slot[pos] = k;
    packed_gate[pos] = gate;
    packed_phase[pos] = 0;
    const size_t rid = (size_t)t * K + k;
    route_expert[rid] = (int16_t)e;
    route_status[rid] = MGF_RS_KEPT_PRIMARY;
}

__global__ void mgf_ref_pack2_kernel(
    const uint64_t* __restrict__ rec2_sorted,
    const uint32_t* __restrict__ m2,
    const uint32_t* __restrict__ kept1,
    const uint32_t* __restrict__ kept2,
    const uint32_t* __restrict__ seg2,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ packed_token,
    int32_t* __restrict__ packed_slot,
    int32_t* __restrict__ packed_gate,
    uint8_t* __restrict__ packed_phase,
    int16_t* __restrict__ route_expert,
    uint8_t* __restrict__ route_status,
    int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)*m2) return;

    int e, t, k;
    int32_t gate;
    mgf_decode_key(rec2_sorted[i], &e, &gate, &t, &k);

    const uint32_t r = (uint32_t)i - seg2[e];
    const size_t rid = (size_t)t * K + k;
    if (r < kept2[e]) {
        const int pos = offsets[e] + (int)kept1[e] + (int)r;
        packed_token[pos] = t;
        packed_slot[pos] = k;
        packed_gate[pos] = gate;
        packed_phase[pos] = 1;
        route_expert[rid] = (int16_t)e;
        route_status[rid] = MGF_RS_KEPT_REROUTED;
    } else {
        route_expert[rid] = -1;
        route_status[rid] = MGF_RS_DROP_OVERFLOW;
    }
}

// ---------------------------------------------------------------------------
// 5. Grouped FFN + combine. One block = one 32-route tile of one expert.
// ---------------------------------------------------------------------------

template <int D4, int H4>
__global__ void mgf_ref_ffn_kernel(
    int E,
    int qshift,
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ w1,
    const int8_t* __restrict__ w2,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ counts,
    const uint32_t* __restrict__ tiles_before,
    const uint32_t* __restrict__ total_tiles,
    const int32_t* __restrict__ packed_token,
    const int32_t* __restrict__ packed_gate,
    int32_t* __restrict__ packed_y,
    int64_t* __restrict__ y) {
    constexpr int D = D4 * 4;
    constexpr int H = H4 * 4;

    const int tile = blockIdx.x;
    if (tile >= (int)*total_tiles) return;

    // Binary search the owning expert: tiles_before[e] <= tile < [e+1].
    int lo = 0, hi = E;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if ((int)tiles_before[mid] <= tile) lo = mid;
        else hi = mid;
    }
    const int e = lo;
    const int tile_local = tile - (int)tiles_before[e];
    const int pos_base = offsets[e] + tile_local * 32;
    const int nr = min(32, counts[e] - tile_local * 32);
    if (nr <= 0) return;

    extern __shared__ uint32_t s_mem[];
    uint32_t* xs = s_mem;                              // [32][D4+1]
    uint32_t* hs = xs + 32 * (D4 + 1);                 // [32][H4+1]
    uint32_t* ws = hs + 32 * (H4 + 1);                 // [64*max(D4,H4)]
    __shared__ int tok_s[32];
    __shared__ int32_t gate_s[32];

    if (threadIdx.x < 32) {
        const int r = threadIdx.x;
        tok_s[r] = r < nr ? packed_token[pos_base + r] : 0;
        gate_s[r] = r < nr ? packed_gate[pos_base + r] : 0;
    }
    __syncthreads();

    const uint32_t* x_u = reinterpret_cast<const uint32_t*>(x);
    for (int idx = threadIdx.x; idx < 32 * D4; idx += 128) {
        const int r = idx / D4;
        const int i = idx - r * D4;
        xs[r * (D4 + 1) + i] =
            r < nr ? x_u[(size_t)tok_s[r] * D4 + i] : 0u;
    }
    __syncthreads();

    // Fixed route per thread: r = tid & 31; its x row lives in registers.
    const int r = threadIdx.x & 31;
    uint32_t xreg[D4];
#pragma unroll
    for (int i = 0; i < D4; ++i) {
        xreg[i] = xs[r * (D4 + 1) + i];
    }

    const uint32_t* w1_u = reinterpret_cast<const uint32_t*>(w1) +
                           (size_t)e * H * D4;
    const uint32_t* w2_u = reinterpret_cast<const uint32_t*>(w2) +
                           (size_t)e * D * H4;
    char* hsb = reinterpret_cast<char*>(hs);

    // Stage A: hidden activations, 64 hidden units per w1 tile.
    for (int hb = 0; hb < H; hb += 64) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < 64 * D4; idx += 128) {
            ws[idx] = w1_u[(size_t)(hb) * D4 + idx];
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < 16; ++p) {
            const int j = p * 4 + (threadIdx.x >> 5);
            int32_t acc = 0;
            const uint32_t* w_row = ws + j * D4;
#pragma unroll
            for (int i = 0; i < D4; ++i) {
                acc = __dp4a((int)xreg[i], (int)w_row[i], acc);
            }
            if (acc < 0) acc = 0;
            acc >>= qshift;
            if (acc > 127) acc = 127;
            hsb[r * (H + 4) + hb + j] = (char)acc;
        }
    }
    __syncthreads();

    // Stage B: outputs, 64 output channels per w2 tile.
    for (int db = 0; db < D; db += 64) {
        const int rows_eff = min(64, D - db);
        __syncthreads();
        for (int idx = threadIdx.x; idx < rows_eff * H4; idx += 128) {
            ws[idx] = w2_u[(size_t)db * H4 + idx];
        }
        __syncthreads();
        for (int p = 0; p < 16; ++p) {
            const int dd = p * 4 + (threadIdx.x >> 5);
            if (dd >= rows_eff) break;
            int32_t acc = 0;
            const uint32_t* w_row = ws + dd * H4;
            const uint32_t* h_row = hs + r * (H4 + 1);
#pragma unroll
            for (int i = 0; i < H4; ++i) {
                acc = __dp4a((int)h_row[i], (int)w_row[i], acc);
            }
            if (r < nr) {
                const int pos = pos_base + r;
                const int d = db + dd;
                packed_y[(size_t)pos * D + d] = acc;
                const unsigned long long add = (unsigned long long)(
                    (int64_t)gate_s[r] * (int64_t)acc);
                atomicAdd(reinterpret_cast<unsigned long long*>(
                              &y[(size_t)tok_s[r] * D + d]),
                          add);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Two-level y checksum. Digests fully parallel; root via exact affine-FNV
//    chunk composition:  FNV(chunk, h) = A * (h & ~0xFF) + T[h & 0xFF].
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t mgf_fnv_byte_dev(uint64_t h, uint8_t b) {
    return (h ^ (uint64_t)b) * MGF_FNV_PRIME;
}

__global__ void mgf_ref_digest_kernel(
    int N,
    int D,
    const int64_t* __restrict__ y,
    uint64_t* __restrict__ digests) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int C = (N + MGF_CSUM_ROWS - 1) / MGF_CSUM_ROWS;
    if (c >= C) return;

    const int row_begin = c * MGF_CSUM_ROWS;
    const int row_end = min(row_begin + MGF_CSUM_ROWS, N);

    uint64_t h = MGF_FNV_BASIS;
    for (int t = row_begin; t < row_end; ++t) {
        const uint64_t* row =
            reinterpret_cast<const uint64_t*>(y) + (size_t)t * D;
        for (int d = 0; d < D; ++d) {
            const uint64_t v = row[d];
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                h = mgf_fnv_byte_dev(h, (uint8_t)(v >> (8 * b)));
            }
        }
    }
    digests[c] = h;
}

__global__ void mgf_ref_root_table_kernel(
    int C,
    const uint64_t* __restrict__ digests,
    uint64_t* __restrict__ root_T,
    uint64_t* __restrict__ root_A) {
    const int chunk = blockIdx.x;
    const int v = threadIdx.x;  // incoming low byte

    const int dig_begin = chunk * MGF_ROOT_CHUNK;
    const int dig_end = min(dig_begin + MGF_ROOT_CHUNK, C);

    uint64_t h = (uint64_t)v;
    uint64_t m = 1ULL;
    for (int i = dig_begin; i < dig_end; ++i) {
        const uint64_t d = digests[i];
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            h = mgf_fnv_byte_dev(h, (uint8_t)(d >> (8 * b)));
            m *= MGF_FNV_PRIME;
        }
    }
    root_T[(size_t)chunk * 256 + v] = h;
    if (v == 0) root_A[chunk] = m;
}

__global__ void mgf_ref_root_combine_kernel(
    int nchunks,
    const uint64_t* __restrict__ root_T,
    const uint64_t* __restrict__ root_A,
    uint64_t* __restrict__ y_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t h = MGF_FNV_BASIS;
    for (int c = 0; c < nchunks; ++c) {
        const uint64_t A = root_A[c];
        const uint64_t T = root_T[(size_t)c * 256 + (h & 0xFFULL)];
        h = A * (h & ~0xFFULL) + T;
    }
    y_checksum[0] = h;
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

static size_t mgf_reference_workspace_bytes_for(const MgfProblemSpec* spec) {
    MgfWorkspaceLayout L = mgf_reference_make_layout(nullptr, spec);
    return L.required_bytes;
}

extern "C" size_t solution_workspace_bytes(const MgfProblemSpec* spec) {
    if (!mgf_validate_problem_spec(spec)) return 0;
    return mgf_reference_workspace_bytes_for(spec);
}

extern "C" cudaError_t solution_init(
    const MgfProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;
    if (!mgf_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }
    MgfReferenceState* st =
        static_cast<MgfReferenceState*>(malloc(sizeof(MgfReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;
    memcpy(&st->spec, spec, sizeof(MgfProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

static cudaError_t mgf_run_sort(
    uint64_t* rec_a,
    uint64_t* rec_b,
    uint32_t* hist,
    uint32_t* scanned,
    int nchunks,
    int host_count,
    const uint32_t* dev_count,
    cudaStream_t stream) {
    uint64_t* src = rec_a;
    uint64_t* dst = rec_b;
    for (int pass = 0; pass < 6; ++pass) {
        const int shift = pass * 8;
        mgf_ref_sort_hist_kernel<<<nchunks, 256, 0, stream>>>(
            src, hist, shift, host_count, dev_count);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        mgf_ref_sort_scan_kernel<<<1, 256, 0, stream>>>(
            hist, scanned, nchunks);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        mgf_ref_sort_scatter_kernel<<<nchunks, 128, 0, stream>>>(
            src, dst, scanned, shift, host_count, dev_count);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        uint64_t* tmp = src;
        src = dst;
        dst = tmp;
    }
    // 6 passes: result is back in rec_a.
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const MgfRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !mgf_validate_run_spec(run) || !inputs_void ||
        !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    MgfReferenceState* st = static_cast<MgfReferenceState*>(state);
    const MgfInputs* in = static_cast<const MgfInputs*>(inputs_void);
    MgfOutputs* out = static_cast<MgfOutputs*>(outputs_void);

    if (run->N > st->spec.max_N || run->D > st->spec.max_D ||
        run->H > st->spec.max_H || run->E > st->spec.max_E ||
        run->K > st->spec.max_K || run->cap > st->spec.max_cap) {
        return cudaErrorInvalidValue;
    }
    if (!in->x || !in->wr || !in->w1 || !in->w2 || !out->logits ||
        !out->counts || !out->offsets || !out->packed_token ||
        !out->packed_slot || !out->packed_gate || !out->packed_phase ||
        !out->route_expert || !out->route_status || !out->packed_y ||
        !out->y || !out->y_checksum) {
        return cudaErrorInvalidValue;
    }

    MgfWorkspaceLayout L = mgf_reference_make_layout(workspace, &st->spec);
    if (workspace_bytes < L.required_bytes) return cudaErrorInvalidValue;

    const int N = run->N;
    const int D = run->D;
    const int H = run->H;
    const int E = run->E;
    const int G = run->G;
    const int g_sel = run->g_sel;
    const int K = run->K;
    const int cap = run->cap;
    const int S = E / G;
    const int n_sel = g_sel * S;
    const int M = N * K;
    const int nchunks = (M + MGF_SORT_CHUNK - 1) / MGF_SORT_CHUNK;
    const int D4 = D / 4;
    const int H4 = H / 4;

    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(L.cnt1, 0, sizeof(uint32_t) * E, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.cnt2, 0, sizeof(uint32_t) * E, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.m2, 0, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(out->y, 0, sizeof(int64_t) * (size_t)N * D, stream);
    if (err != cudaSuccess) return err;

    // 1. Router logits.
    {
        const int grid = mgf_ceil_div_int(N, 128);
        const size_t shmem =
            sizeof(uint32_t) * ((size_t)E * D4 + 128 * (D4 + 1));
        switch (D4) {
            case 8:
                mgf_ref_logits_kernel<8><<<grid, 128, shmem, stream>>>(
                    N, E, in->x, in->wr, out->logits);
                break;
            case 16:
                mgf_ref_logits_kernel<16><<<grid, 128, shmem, stream>>>(
                    N, E, in->x, in->wr, out->logits);
                break;
            case 32:
                mgf_ref_logits_kernel<32><<<grid, 128, shmem, stream>>>(
                    N, E, in->x, in->wr, out->logits);
                break;
            default:
                return cudaErrorInvalidValue;
        }
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    // 2. Routing.
    mgf_ref_route_kernel<<<mgf_ceil_div_int(N, 4), 128, 0, stream>>>(
        N, E, G, g_sel, K, out->logits, L.cand, L.rec_a, L.cnt1);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    // 3. Phase-1 sort.
    err = mgf_run_sort(L.rec_a, L.rec_b, L.hist, L.scanned,
                       nchunks, M, nullptr, stream);
    if (err != cudaSuccess) return err;

    mgf_ref_seg1_kernel<<<1, 1, 0, stream>>>(E, cap, L.cnt1, L.kept1, L.seg1);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    // 4. Classify overflow -> phase-2 records / no-backup drops.
    mgf_ref_classify_kernel<<<mgf_ceil_div_int(M, 256), 256, 0, stream>>>(
        M, E, K, n_sel, L.rec_a, L.kept1, L.seg1, L.cand, out->logits,
        L.rec2_a, L.cnt2, L.m2, out->route_expert, out->route_status);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    // 5. Phase-2 sort (device-resident count).
    err = mgf_run_sort(L.rec2_a, L.rec2_b, L.hist, L.scanned,
                       nchunks, 0, L.m2, stream);
    if (err != cudaSuccess) return err;

    mgf_ref_finalize_kernel<<<1, 1, 0, stream>>>(
        E, cap, L.cnt2, L.kept1, L.kept2, L.seg2, out->counts, out->offsets,
        L.tiles_before, L.total_tiles);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    // 6. Pack both phases.
    mgf_ref_pack1_kernel<<<mgf_ceil_div_int(M, 256), 256, 0, stream>>>(
        M, L.rec_a, L.kept1, L.seg1, out->offsets, out->packed_token,
        out->packed_slot, out->packed_gate, out->packed_phase,
        out->route_expert, out->route_status, K);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mgf_ref_pack2_kernel<<<mgf_ceil_div_int(M, 256), 256, 0, stream>>>(
        L.rec2_a, L.m2, L.kept1, L.kept2, L.seg2, out->offsets,
        out->packed_token, out->packed_slot, out->packed_gate,
        out->packed_phase, out->route_expert, out->route_status, K);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    // 7. Grouped FFN + combine.
    {
        const int max_tiles = E * mgf_ceil_div_int(cap, 32);
        const size_t shmem = sizeof(uint32_t) *
            ((size_t)32 * (D4 + 1) + 32 * (H4 + 1) +
             64 * (size_t)(D4 > H4 ? D4 : H4));
#define MGF_FFN_LAUNCH(D4C, H4C)                                            \
    mgf_ref_ffn_kernel<D4C, H4C><<<max_tiles, 128, shmem, stream>>>(        \
        E, run->qshift, in->x, in->w1, in->w2, out->offsets, out->counts,   \
        L.tiles_before, L.total_tiles, out->packed_token, out->packed_gate, \
        out->packed_y, out->y);
        if (D4 == 8 && H4 == 16) { MGF_FFN_LAUNCH(8, 16) }
        else if (D4 == 8 && H4 == 32) { MGF_FFN_LAUNCH(8, 32) }
        else if (D4 == 8 && H4 == 64) { MGF_FFN_LAUNCH(8, 64) }
        else if (D4 == 16 && H4 == 16) { MGF_FFN_LAUNCH(16, 16) }
        else if (D4 == 16 && H4 == 32) { MGF_FFN_LAUNCH(16, 32) }
        else if (D4 == 16 && H4 == 64) { MGF_FFN_LAUNCH(16, 64) }
        else if (D4 == 32 && H4 == 16) { MGF_FFN_LAUNCH(32, 16) }
        else if (D4 == 32 && H4 == 32) { MGF_FFN_LAUNCH(32, 32) }
        else if (D4 == 32 && H4 == 64) { MGF_FFN_LAUNCH(32, 64) }
        else return cudaErrorInvalidValue;
#undef MGF_FFN_LAUNCH
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    // 8. y checksum.
    {
        const int C = mgf_ceil_div_int(N, MGF_CSUM_ROWS);
        mgf_ref_digest_kernel<<<mgf_ceil_div_int(C, 128), 128, 0, stream>>>(
            N, D, out->y, L.digests);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        const int nroot = mgf_ceil_div_int(C, MGF_ROOT_CHUNK);
        mgf_ref_root_table_kernel<<<nroot, 256, 0, stream>>>(
            C, L.digests, L.root_T, L.root_A);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        mgf_ref_root_combine_kernel<<<1, 1, 0, stream>>>(
            nroot, L.root_T, L.root_A, out->y_checksum);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    (void)state;
    (void)stream;
    return cudaSuccess;
}

extern "C" void solution_destroy(void* state) {
    free(state);
}
