// PMPP_CANARY_87_29aa91a531 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: dualtier_credit_router_reference.cu
// Strong reference implementation.
//
// Architecture:
//   - Routing: warp-per-token tier-1 (packed-key shfl butterflies for the
//     top-2 nodes), tiled dp4a s2 GEMM (32 tokens x 32 experts per block,
//     operands staged as packed u32 in shared), warp-per-token tier-2
//     expert argmax over the owned slice.
//   - Admission: per-phase 48-bit keys (expert | inverted-biased-score |
//     token) ranked with a stable multi-block LSD radix sort (8-bit
//     digits, digit-major block histograms + single-block scan +
//     __match_any_sync stable scatter). Credit arithmetic, packing
//     offsets and event-log bases derived with tiny device scans.
//   - Event log / packed outputs emitted fully in parallel from the
//     sorted arrays; per-event credit_after computed from phase-start
//     credit snapshots.
//   - Delivered-token FFN: one block per packed route, dp4a layer 1 into
//     shared requantized bytes, dp4a layer 2 (weights L2-resident since
//     packing is expert-major).
//   - State: persistent per-expert credit + ring-buffer backlog (gid +
//     stored x rows). Checksums per contract: per-entry digests, then
//     per-expert digests, then a root fold (all short serial chains).
// ============================================================================

#include "dualtier_credit_router_common.h"

#include <stdlib.h>
#include <string.h>

#define DTR_KEY_SENTINEL (~0ULL)
#define DTR_RADIX_BLOCK 256

struct DtrReferenceState {
    DtrProblemSpec spec;
    uint32_t* credit;        // [max_E]
    uint32_t* bl_head;       // [max_E]
    uint32_t* bl_len;        // [max_E]
    uint32_t* bl_gid;        // [max_E * max_bq]
    int8_t* bl_x;            // [max_E * max_bq * max_D]
    uint32_t* dev_ctr;       // [2]: [0] = gid base, [1] = fresh flag
    // Cached executable graph of the whole run pipeline (the pipeline is
    // many small kernels; replay kills per-launch overhead). Re-captured
    // whenever the RunSpec or any passed pointer changes.
    cudaGraphExec_t graph_exec;
    int graph_valid;
    DtrRunSpec cached_run;
    DtrInputs cached_in;
    DtrOutputs cached_out;
    void* cached_ws;
    size_t cached_wsb;
};

struct DtrRefWorkspace {
    uint32_t* snap_head;   // [E]
    uint32_t* snap_len;    // [E]
    uint32_t* cred0;       // [E]
    uint32_t* c1;          // [E]
    uint32_t* c2;          // [E]
    uint32_t* c3;          // [E]
    uint32_t* a0;          // [E]
    uint32_t* a1;          // [E]
    uint32_t* a2;          // [E]
    uint32_t* att1;        // [E]
    uint32_t* att2;        // [E]
    uint32_t* att3;        // [E]
    uint32_t* m23;         // [2]
    uint32_t* seg1;        // [E]
    uint32_t* seg2;        // [E]
    uint32_t* seg3;        // [E]
    uint32_t* lb0;         // [E]
    uint32_t* lb1;         // [E]
    uint32_t* lb2;         // [E]
    uint32_t* lb3;         // [E]
    uint32_t* freec;       // [E]
    uint32_t* pn;          // [maxN]
    uint32_t* bn;          // [maxN]
    uint32_t* pe;          // [maxN]
    uint32_t* be;          // [maxN]
    uint64_t* keys1;       // [padN]
    uint64_t* keys2;       // [padN]
    uint64_t* keys3;       // [padN]
    uint64_t* ktmp;        // [padN]
    uint32_t* hist;        // [256 * padN/256]
    uint32_t* dsrc;        // [max_E * max_ccap]
    uint32_t* dexp;        // [max_E * max_ccap]
    uint64_t* edig;        // [max_E * max_bq]
    uint64_t* xdig;        // [max_E]
    size_t required_bytes;
};

static DtrRefWorkspace dtr_ref_make_layout(
    void* workspace,
    const DtrProblemSpec* spec) {
    DtrRefWorkspace L{};
    char* base = static_cast<char*>(workspace);
    const size_t E = (size_t)spec->max_E;
    const size_t maxN = (size_t)spec->max_N;
    const size_t padN = dtr_align_up_size(maxN, DTR_RADIX_BLOCK);
    const size_t cap = (size_t)spec->max_E * (size_t)spec->max_ccap;

    size_t off = 0;
#define DTR_WS_FIELD(field, type, count)                                     \
    off = dtr_align_up_size(off, 128);                                       \
    L.field = reinterpret_cast<type*>(base + off);                           \
    off += sizeof(type) * (count);

    DTR_WS_FIELD(snap_head, uint32_t, E)
    DTR_WS_FIELD(snap_len, uint32_t, E)
    DTR_WS_FIELD(cred0, uint32_t, E)
    DTR_WS_FIELD(c1, uint32_t, E)
    DTR_WS_FIELD(c2, uint32_t, E)
    DTR_WS_FIELD(c3, uint32_t, E)
    DTR_WS_FIELD(a0, uint32_t, E)
    DTR_WS_FIELD(a1, uint32_t, E)
    DTR_WS_FIELD(a2, uint32_t, E)
    DTR_WS_FIELD(att1, uint32_t, E)
    DTR_WS_FIELD(att2, uint32_t, E)
    DTR_WS_FIELD(att3, uint32_t, E)
    DTR_WS_FIELD(m23, uint32_t, 2)
    DTR_WS_FIELD(seg1, uint32_t, E)
    DTR_WS_FIELD(seg2, uint32_t, E)
    DTR_WS_FIELD(seg3, uint32_t, E)
    DTR_WS_FIELD(lb0, uint32_t, E)
    DTR_WS_FIELD(lb1, uint32_t, E)
    DTR_WS_FIELD(lb2, uint32_t, E)
    DTR_WS_FIELD(lb3, uint32_t, E)
    DTR_WS_FIELD(freec, uint32_t, E)
    DTR_WS_FIELD(pn, uint32_t, maxN)
    DTR_WS_FIELD(bn, uint32_t, maxN)
    DTR_WS_FIELD(pe, uint32_t, maxN)
    DTR_WS_FIELD(be, uint32_t, maxN)
    DTR_WS_FIELD(keys1, uint64_t, padN)
    DTR_WS_FIELD(keys2, uint64_t, padN)
    DTR_WS_FIELD(keys3, uint64_t, padN)
    DTR_WS_FIELD(ktmp, uint64_t, padN)
    DTR_WS_FIELD(hist, uint32_t, 256 * (padN / DTR_RADIX_BLOCK))
    DTR_WS_FIELD(dsrc, uint32_t, cap)
    DTR_WS_FIELD(dexp, uint32_t, cap)
    DTR_WS_FIELD(edig, uint64_t, (size_t)spec->max_E * spec->max_bq)
    DTR_WS_FIELD(xdig, uint64_t, E)
#undef DTR_WS_FIELD

    off = dtr_align_up_size(off, 128);
    L.required_bytes = off;
    return L;
}

// ---------------------------------------------------------------------------
// Small device helpers.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t dtr_pack_key(int e, uint32_t inv, int t) {
    return ((uint64_t)(uint32_t)e << 36) | ((uint64_t)inv << 14) |
           (uint64_t)(uint32_t)t;
}

__device__ __forceinline__ uint32_t dtr_inv_score(int32_t s) {
    return 0x3FFFFFu - (uint32_t)(s + DTR_SCORE_BIAS);
}

__device__ __forceinline__ uint64_t dtr_fnv_byte_dev(uint64_t h, uint8_t b) {
    return (h ^ (uint64_t)b) * DTR_FNV_PRIME;
}

__device__ __forceinline__ uint64_t dtr_fnv_u32_dev(uint64_t h, uint32_t v) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        h = dtr_fnv_byte_dev(h, (uint8_t)(v >> (8 * m)));
    }
    return h;
}

__device__ __forceinline__ uint64_t dtr_fnv_u64_dev(uint64_t h, uint64_t v) {
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        h = dtr_fnv_byte_dev(h, (uint8_t)(v >> (8 * m)));
    }
    return h;
}

// ---------------------------------------------------------------------------
// Refill + snapshots.
// ---------------------------------------------------------------------------

__global__ void dtr_ref_refill_kernel(
    int E,
    int ccap,
    int refill,
    const uint32_t* __restrict__ dev_ctr,
    const uint32_t* __restrict__ credit,
    const uint32_t* __restrict__ bl_head,
    const uint32_t* __restrict__ bl_len,
    uint32_t* __restrict__ snap_head,
    uint32_t* __restrict__ snap_len,
    uint32_t* __restrict__ cred0,
    uint32_t* __restrict__ a0,
    uint32_t* __restrict__ c1) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const int fresh = dev_ctr[1] != 0u;
    const uint32_t head = fresh ? 0u : bl_head[e];
    const uint32_t len = fresh ? 0u : bl_len[e];
    uint32_t c = fresh ? (uint32_t)ccap
                       : credit[e] + (uint32_t)refill;
    if (c > (uint32_t)ccap) c = (uint32_t)ccap;
    const uint32_t a = len < c ? len : c;
    snap_head[e] = head;
    snap_len[e] = len;
    cred0[e] = c;
    a0[e] = a;
    c1[e] = c - a;
}

// ---------------------------------------------------------------------------
// Tier-1: s1 scores + top-2 nodes (one warp per token).
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256)
dtr_ref_s1_kernel(
    int N,
    int P,
    int D4,
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ wnode,
    int32_t* __restrict__ route_nodes,
    uint32_t* __restrict__ pn_arr,
    uint32_t* __restrict__ bn_arr) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int t = blockIdx.x * 8 + warp;
    if (t >= N) return;

    const uint32_t* xu = reinterpret_cast<const uint32_t*>(x) +
                         (size_t)t * D4;
    uint32_t key = 0;
    if (lane < P) {
        const uint32_t* wu = reinterpret_cast<const uint32_t*>(wnode) +
                             (size_t)lane * D4;
        int32_t s = 0;
        for (int w = 0; w < D4; ++w) {
            s = __dp4a((int)__ldg(&xu[w]), (int)__ldg(&wu[w]), s);
        }
        key = ((uint32_t)(s + DTR_SCORE_BIAS) << 7) | (uint32_t)(127 - lane);
    }

    uint32_t k1 = key;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const uint32_t o = __shfl_xor_sync(0xffffffffu, k1, off);
        if (o > k1) k1 = o;
    }
    const int pn = 127 - (int)(k1 & 0x7Fu);

    uint32_t k2 = lane == pn ? 0u : key;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const uint32_t o = __shfl_xor_sync(0xffffffffu, k2, off);
        if (o > k2) k2 = o;
    }
    const int bn = 127 - (int)(k2 & 0x7Fu);

    if (lane == 0) {
        route_nodes[t] = (pn << 16) | bn;
        pn_arr[t] = (uint32_t)pn;
        bn_arr[t] = (uint32_t)bn;
    }
}

// ---------------------------------------------------------------------------
// Tier-2 GEMM: s2[t][e] tiled 32 tokens x 32 experts, dp4a.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256)
dtr_ref_s2_kernel(
    int N,
    int E,
    int D4,
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ wexp,
    int32_t* __restrict__ s2_logits) {
    __shared__ uint32_t xs[32][33];
    __shared__ uint32_t ws[32][33];

    const int t0 = blockIdx.x * 32;
    const int e0 = blockIdx.y * 32;
    const uint32_t* xu = reinterpret_cast<const uint32_t*>(x);
    const uint32_t* wu = reinterpret_cast<const uint32_t*>(wexp);

    for (int idx = threadIdx.x; idx < 32 * D4; idx += 256) {
        const int row = idx / D4;
        const int w = idx - row * D4;
        xs[row][w] = t0 + row < N ? xu[(size_t)(t0 + row) * D4 + w] : 0u;
        ws[row][w] = wu[(size_t)(e0 + row) * D4 + w];
    }
    __syncthreads();

    const int e_sub = threadIdx.x & 31;
    const int tquad = threadIdx.x >> 5;

    int32_t acc[4] = {0, 0, 0, 0};
    for (int w = 0; w < D4; ++w) {
        const uint32_t wv = ws[e_sub][w];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            acc[q] = __dp4a((int)xs[tquad * 4 + q][w], (int)wv, acc[q]);
        }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int t = t0 + tquad * 4 + q;
        if (t < N) {
            s2_logits[(size_t)t * E + e0 + e_sub] = acc[q];
        }
    }
}

// ---------------------------------------------------------------------------
// Tier-2 selection + phase-1 keys (one warp per token).
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256)
dtr_ref_select_kernel(
    int N,
    int E,
    int S,
    const int32_t* __restrict__ s2_logits,
    const uint32_t* __restrict__ pn_arr,
    const uint32_t* __restrict__ bn_arr,
    int32_t* __restrict__ route_pe_be,
    uint32_t* __restrict__ pe_arr,
    uint32_t* __restrict__ be_arr,
    uint64_t* __restrict__ keys1,
    uint32_t* __restrict__ att1) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int t = blockIdx.x * 8 + warp;
    if (t >= N) return;

    const int pn = (int)pn_arr[t];
    const int bn = (int)bn_arr[t];
    const int32_t* row = s2_logits + (size_t)t * E;

    uint32_t kp = 0, kb = 0;
    if (lane < S) {
        const int32_t sp = __ldg(&row[pn * S + lane]);
        const int32_t sb = __ldg(&row[bn * S + lane]);
        kp = ((uint32_t)(sp + DTR_SCORE_BIAS) << 7) | (uint32_t)(127 - lane);
        kb = ((uint32_t)(sb + DTR_SCORE_BIAS) << 7) | (uint32_t)(127 - lane);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const uint32_t op = __shfl_xor_sync(0xffffffffu, kp, off);
        const uint32_t ob = __shfl_xor_sync(0xffffffffu, kb, off);
        if (op > kp) kp = op;
        if (ob > kb) kb = ob;
    }
    const int pe = pn * S + (127 - (int)(kp & 0x7Fu));
    const int be = bn * S + (127 - (int)(kb & 0x7Fu));

    if (lane == 0) {
        route_pe_be[t] = (pe << 16) | be;
        pe_arr[t] = (uint32_t)pe;
        be_arr[t] = (uint32_t)be;
        keys1[t] = dtr_pack_key(pe, dtr_inv_score(__ldg(&row[pe])), t);
        atomicAdd(&att1[pe], 1u);
    }
}

// ---------------------------------------------------------------------------
// Stable LSD radix sort (8-bit digits over bits 0..47).
// ---------------------------------------------------------------------------

__global__ void dtr_ref_radix_hist_kernel(
    const uint64_t* __restrict__ keys,
    int shift,
    int nb,
    uint32_t* __restrict__ hist) {
    __shared__ uint32_t sh[256];
    sh[threadIdx.x] = 0;
    __syncthreads();
    const uint64_t key = keys[(size_t)blockIdx.x * 256 + threadIdx.x];
    const int digit = (int)((key >> shift) & 0xFF);
    atomicAdd(&sh[digit], 1u);
    __syncthreads();
    hist[(size_t)threadIdx.x * nb + blockIdx.x] = sh[threadIdx.x];
}

__global__ void dtr_ref_radix_scan_kernel(
    int nb,
    uint32_t* __restrict__ hist) {
    __shared__ uint32_t partial[1024];
    const int M = 256 * nb;
    const int chunk = (M + 1023) / 1024;  // per-thread contiguous chunk
    const int lo = threadIdx.x * chunk;

    uint32_t sum = 0;
    for (int i = 0; i < chunk && lo + i < M; ++i) {
        sum += hist[lo + i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    // Exclusive scan of partials (Hillis-Steele).
    uint32_t v = partial[threadIdx.x];
    for (int off = 1; off < 1024; off <<= 1) {
        uint32_t add = 0;
        if (threadIdx.x >= off) add = partial[threadIdx.x - off];
        __syncthreads();
        partial[threadIdx.x] = v = v + add;
        __syncthreads();
    }
    uint32_t run = threadIdx.x == 0 ? 0u : partial[threadIdx.x - 1];

    for (int i = 0; i < chunk && lo + i < M; ++i) {
        const uint32_t h = hist[lo + i];
        hist[lo + i] = run;
        run += h;
    }
}

__global__ void dtr_ref_radix_scatter_kernel(
    const uint64_t* __restrict__ keys_in,
    int shift,
    int nb,
    const uint32_t* __restrict__ hist,
    uint64_t* __restrict__ keys_out) {
    __shared__ uint32_t warp_hist[8][256];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    for (int i = threadIdx.x; i < 8 * 256; i += 256) {
        (&warp_hist[0][0])[i] = 0;
    }
    __syncthreads();

    const uint64_t key = keys_in[(size_t)blockIdx.x * 256 + threadIdx.x];
    const int digit = (int)((key >> shift) & 0xFF);

    const uint32_t mask = __match_any_sync(0xffffffffu, digit);
    const uint32_t lane_rank = __popc(mask & ((1u << lane) - 1u));
    const int leader = __ffs(mask) - 1;
    if (lane == leader) {
        warp_hist[warp][digit] = __popc(mask);
    }
    __syncthreads();

    uint32_t base = 0;
    for (int w = 0; w < 8; ++w) {
        if (w < warp) base += warp_hist[w][digit];
    }
    const uint32_t pos =
        hist[(size_t)digit * nb + blockIdx.x] + base + lane_rank;
    keys_out[pos] = key;
}

static void dtr_ref_radix_sort(
    uint64_t* keys,
    uint64_t* tmp,
    uint32_t* hist,
    int padN,
    cudaStream_t stream) {
    const int nb = padN / DTR_RADIX_BLOCK;
    uint64_t* a = keys;
    uint64_t* b = tmp;
    for (int pass = 0; pass < 6; ++pass) {
        const int shift = pass * 8;
        dtr_ref_radix_hist_kernel<<<nb, 256, 0, stream>>>(a, shift, nb, hist);
        dtr_ref_radix_scan_kernel<<<1, 1024, 0, stream>>>(nb, hist);
        dtr_ref_radix_scatter_kernel<<<nb, 256, 0, stream>>>(
            a, shift, nb, hist, b);
        uint64_t* t = a;
        a = b;
        b = t;
    }
    // 6 passes: result is back in `keys`.
}

// ---------------------------------------------------------------------------
// Per-phase credit/segment bookkeeping (tiny single-block kernels).
// ---------------------------------------------------------------------------

__global__ void dtr_ref_mini_kernel(
    int E,
    const uint32_t* __restrict__ att,
    const uint32_t* __restrict__ cred_in,
    uint32_t* __restrict__ seg,
    uint32_t* __restrict__ adm,
    uint32_t* __restrict__ cred_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t run = 0;
    for (int e = 0; e < E; ++e) {
        seg[e] = run;
        run += att[e];
        const uint32_t a = att[e] < cred_in[e] ? att[e] : cred_in[e];
        adm[e] = a;
        cred_out[e] = cred_in[e] - a;
    }
}

__global__ void dtr_ref_reject_kernel(
    int N,
    int E,
    uint32_t sentinel_guard,  // unused; keeps signature obvious
    const uint64_t* __restrict__ sorted_keys,
    const uint32_t* __restrict__ seg,
    const uint32_t* __restrict__ adm,
    const int32_t* __restrict__ s2_logits,
    const uint32_t* __restrict__ next_target,  // be (phase1) or pe (phase2)
    uint64_t* __restrict__ next_keys,
    uint32_t* __restrict__ next_att,
    uint32_t* __restrict__ mcounter) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t key = sorted_keys[g];
    if (key == DTR_KEY_SENTINEL) return;
    const int e = (int)((key >> 36) & 0x7F);
    const uint32_t r = (uint32_t)g - seg[e];
    if (r < adm[e]) return;  // admitted
    const int t = (int)(key & 0x3FFF);
    const int ne = (int)next_target[t];
    const int32_t s = __ldg(&s2_logits[(size_t)t * E + ne]);
    const uint32_t pos = atomicAdd(mcounter, 1u);
    next_keys[pos] = dtr_pack_key(ne, dtr_inv_score(s), t);
    atomicAdd(&next_att[ne], 1u);
}

// Big single-block bookkeeping kernel: counts/offsets outputs, log bases,
// free slots, log_len, credit_out, persistent state update.
__global__ void dtr_ref_offsets_kernel(
    int E,
    int N,
    int bq,
    const uint32_t* __restrict__ a0,
    const uint32_t* __restrict__ a1,
    const uint32_t* __restrict__ a2,
    const uint32_t* __restrict__ att3,
    const uint32_t* __restrict__ c3,
    const uint32_t* __restrict__ snap_head,
    const uint32_t* __restrict__ snap_len,
    uint32_t* __restrict__ seg3,
    uint32_t* __restrict__ lb0,
    uint32_t* __restrict__ lb1,
    uint32_t* __restrict__ lb2,
    uint32_t* __restrict__ lb3,
    uint32_t* __restrict__ freec,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int32_t* __restrict__ log_len,
    uint32_t* __restrict__ credit_out,
    uint32_t* __restrict__ credit,
    uint32_t* __restrict__ bl_head,
    uint32_t* __restrict__ bl_len) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t tot0 = 0, tot1 = 0, tot2 = 0, tot3 = 0;
    for (int e = 0; e < E; ++e) {
        tot0 += a0[e];
        tot1 += a1[e];
        tot2 += a2[e];
        tot3 += att3[e];
    }

    uint32_t r0 = 0, r1 = tot0, r2 = tot0 + tot1, r3 = tot0 + tot1 + tot2;
    uint32_t rs = 0, ro = 0;
    for (int e = 0; e < E; ++e) {
        lb0[e] = r0;
        lb1[e] = r1;
        lb2[e] = r2;
        lb3[e] = r3;
        r0 += a0[e];
        r1 += a1[e];
        r2 += a2[e];
        r3 += att3[e];

        seg3[e] = rs;
        rs += att3[e];

        const uint32_t cnt = a0[e] + a1[e] + a2[e];
        counts[e] = (int32_t)cnt;
        offsets[e] = (int32_t)ro;
        ro += cnt;

        const uint32_t remain = snap_len[e] - a0[e];
        const uint32_t fr = (uint32_t)bq - remain;
        freec[e] = fr;
        const uint32_t enq = att3[e] < fr ? att3[e] : fr;

        credit_out[e] = c3[e];
        credit[e] = c3[e];
        bl_head[e] = (snap_head[e] + a0[e]) % (uint32_t)bq;
        bl_len[e] = remain + enq;
    }
    offsets[E] = (int32_t)ro;
    log_len[0] = (int32_t)(tot0 + tot1 + tot2 + tot3);
}

// ---------------------------------------------------------------------------
// Emission.
// ---------------------------------------------------------------------------

__global__ void dtr_ref_emit_phase0_kernel(
    int E,
    int maxBQ,
    int bq,
    const uint32_t* __restrict__ a0,
    const uint32_t* __restrict__ cred0,
    const uint32_t* __restrict__ snap_head,
    const uint32_t* __restrict__ lb0,
    const uint32_t* __restrict__ bl_gid,
    const int32_t* __restrict__ offsets,
    uint64_t* __restrict__ event_log,
    int32_t* __restrict__ packed_gid,
    uint32_t* __restrict__ dsrc,
    uint32_t* __restrict__ dexp) {
    const int e = blockIdx.x;
    const uint32_t r = threadIdx.x;
    if (e >= E || r >= a0[e]) return;
    const uint32_t slot = (snap_head[e] + r) % (uint32_t)bq;
    const uint32_t gid = bl_gid[(size_t)e * maxBQ + slot];
    event_log[lb0[e] + r] = dtr_make_log_word(
        gid, e, DTR_ACT_DELIV_BACKLOG, 0, cred0[e] - r - 1u);
    const uint32_t pos = (uint32_t)offsets[e] + r;
    packed_gid[pos] = (int32_t)gid;
    dsrc[pos] = 0x80000000u | ((uint32_t)e * maxBQ + slot);
    dexp[pos] = (uint32_t)e;
}

__global__ void dtr_ref_emit_deliv_kernel(
    int phase,  // 1 or 2
    const uint32_t* __restrict__ dev_ctr,
    const uint64_t* __restrict__ sorted_keys,
    const uint32_t* __restrict__ seg,
    const uint32_t* __restrict__ adm,
    const uint32_t* __restrict__ cred_start,
    const uint32_t* __restrict__ lb,
    const uint32_t* __restrict__ a0,
    const uint32_t* __restrict__ a1,  // null for phase 1
    const int32_t* __restrict__ offsets,
    uint64_t* __restrict__ event_log,
    int32_t* __restrict__ packed_gid,
    uint32_t* __restrict__ dsrc,
    uint32_t* __restrict__ dexp) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t key = sorted_keys[g];
    if (key == DTR_KEY_SENTINEL) return;
    const int e = (int)((key >> 36) & 0x7F);
    const uint32_t r = (uint32_t)g - seg[e];
    if (r >= adm[e]) return;  // rejected (handled by the next phase)
    const int t = (int)(key & 0x3FFF);
    const uint32_t gid = dev_ctr[0] + (uint32_t)t;
    const int action =
        phase == 1 ? DTR_ACT_DELIV_PRIMARY : DTR_ACT_DELIV_BACKUP;
    event_log[lb[e] + r] = dtr_make_log_word(
        gid, e, action, phase, cred_start[e] - r - 1u);
    uint32_t pos = (uint32_t)offsets[e] + a0[e] + r;
    if (phase == 2) pos += a1[e];
    packed_gid[pos] = (int32_t)gid;
    dsrc[pos] = (uint32_t)t;
    dexp[pos] = (uint32_t)e;
}

__global__ void dtr_ref_emit_phase3_kernel(
    const uint32_t* __restrict__ dev_ctr,
    int D4,
    int maxBQ,
    int maxD,
    int bq,
    const uint64_t* __restrict__ sorted_keys,
    const uint32_t* __restrict__ seg3,
    const uint32_t* __restrict__ freec,
    const uint32_t* __restrict__ c3,
    const uint32_t* __restrict__ lb3,
    const uint32_t* __restrict__ snap_head,
    const uint32_t* __restrict__ snap_len,
    const int8_t* __restrict__ x,
    uint64_t* __restrict__ event_log,
    uint32_t* __restrict__ bl_gid,
    int8_t* __restrict__ bl_x) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.x * 8 + warp;
    const uint64_t key = sorted_keys[g];
    if (key == DTR_KEY_SENTINEL) return;
    const int e = (int)((key >> 36) & 0x7F);
    const uint32_t r = (uint32_t)g - seg3[e];
    const int t = (int)(key & 0x3FFF);
    const uint32_t gid = dev_ctr[0] + (uint32_t)t;
    const bool queued = r < freec[e];

    if (lane == 0) {
        event_log[lb3[e] + r] = dtr_make_log_word(
            gid, e, queued ? DTR_ACT_QUEUED : DTR_ACT_DROPPED, 3,
            c3[e] & 0xFFFFu);
    }
    if (!queued) return;

    const uint32_t slot = (snap_head[e] + snap_len[e] + r) % (uint32_t)bq;
    if (lane == 0) {
        bl_gid[(size_t)e * maxBQ + slot] = gid;
    }
    const uint32_t* xu = reinterpret_cast<const uint32_t*>(x) +
                         (size_t)t * D4;
    uint32_t* dst = reinterpret_cast<uint32_t*>(
        bl_x + ((size_t)e * maxBQ + slot) * maxD);
    if (lane < D4) {
        dst[lane] = __ldg(&xu[lane]);
    }
}

// ---------------------------------------------------------------------------
// Delivered-token FFN (one block per packed route).
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256)
dtr_ref_ffn_kernel(
    int E,
    int D,
    int H,
    int qshift,
    int maxD,
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ dsrc,
    const uint32_t* __restrict__ dexp,
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ bl_x,
    const int8_t* __restrict__ w1,
    const int8_t* __restrict__ w2,
    int32_t* __restrict__ packed_out) {
    const int pos = blockIdx.x;
    if (pos >= __ldg(&offsets[E])) return;

    __shared__ uint32_t xv[32];
    __shared__ uint32_t hw[64];      // H requantized bytes as packed u32
    __shared__ uint32_t wsh[256 * 33];  // padded weight tile (33.8 KB)

    const int D4 = D / 4;
    const int H4 = H / 4;
    const uint32_t src = dsrc[pos];
    const int e = (int)dexp[pos];

    if (threadIdx.x < (uint32_t)D4) {
        const uint32_t* srcp;
        if (src & 0x80000000u) {
            srcp = reinterpret_cast<const uint32_t*>(
                bl_x + (size_t)(src & 0x7FFFFFFFu) * maxD);
        } else {
            srcp = reinterpret_cast<const uint32_t*>(x) + (size_t)src * D4;
        }
        xv[threadIdx.x] = __ldg(&srcp[threadIdx.x]);
    }

    // Stage w1[e] (H x D4 words) coalesced into padded shared rows.
    {
        const uint32_t* w1u = reinterpret_cast<const uint32_t*>(w1) +
                              (size_t)e * H * D4;
        for (int i = threadIdx.x; i < H * D4; i += 256) {
            const int row = i / D4;
            const int w = i - row * D4;
            wsh[row * (D4 + 1) + w] = __ldg(&w1u[i]);
        }
    }
    __syncthreads();

    if (threadIdx.x < (uint32_t)H) {
        const int j = threadIdx.x;
        int32_t acc = 0;
        for (int w = 0; w < D4; ++w) {
            acc = __dp4a((int)xv[w], (int)wsh[j * (D4 + 1) + w], acc);
        }
        acc = acc > 0 ? acc : 0;
        acc >>= qshift;
        if (acc > 127) acc = 127;
        reinterpret_cast<uint8_t*>(hw)[j] = (uint8_t)acc;
    }
    __syncthreads();

    // Stage w2[e] (D x H4 words) the same way.
    {
        const uint32_t* w2u = reinterpret_cast<const uint32_t*>(w2) +
                              (size_t)e * D * H4;
        for (int i = threadIdx.x; i < D * H4; i += 256) {
            const int row = i / H4;
            const int k = i - row * H4;
            wsh[row * (H4 + 1) + k] = __ldg(&w2u[i]);
        }
    }
    __syncthreads();

    if (threadIdx.x < (uint32_t)D) {
        const int d = threadIdx.x;
        int32_t acc = 0;
        for (int k = 0; k < H4; ++k) {
            acc = __dp4a((int)hw[k], (int)wsh[d * (H4 + 1) + k], acc);
        }
        packed_out[(size_t)pos * D + d] = acc;
    }
}

// ---------------------------------------------------------------------------
// Post-state checksum (three levels, all short chains).
// ---------------------------------------------------------------------------

__global__ void dtr_ref_csum_entries_kernel(
    int E,
    int D,
    int maxBQ,
    int maxD,
    int bq,
    const uint32_t* __restrict__ bl_head,
    const uint32_t* __restrict__ bl_len,
    const uint32_t* __restrict__ bl_gid,
    const int8_t* __restrict__ bl_x,
    uint64_t* __restrict__ edig) {
    const int e = blockIdx.x;
    const uint32_t q = threadIdx.x;
    if (e >= E || q >= bl_len[e]) return;
    const uint32_t slot = (bl_head[e] + q) % (uint32_t)bq;
    uint64_t h = DTR_FNV_BASIS;
    h = dtr_fnv_u32_dev(h, bl_gid[(size_t)e * maxBQ + slot]);
    const int8_t* row = bl_x + ((size_t)e * maxBQ + slot) * maxD;
    for (int d = 0; d < D; ++d) {
        h = dtr_fnv_byte_dev(h, (uint8_t)row[d]);
    }
    edig[(size_t)e * maxBQ + q] = h;
}

__global__ void dtr_ref_csum_expert_kernel(
    int E,
    int maxBQ,
    const uint32_t* __restrict__ credit,
    const uint32_t* __restrict__ bl_len,
    const uint64_t* __restrict__ edig,
    uint64_t* __restrict__ xdig) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    uint64_t h = DTR_FNV_BASIS;
    h = dtr_fnv_u32_dev(h, credit[e]);
    h = dtr_fnv_u32_dev(h, bl_len[e]);
    const uint32_t len = bl_len[e];
    for (uint32_t q = 0; q < len; ++q) {
        h = dtr_fnv_u64_dev(h, edig[(size_t)e * maxBQ + q]);
    }
    xdig[e] = h;
}

__global__ void dtr_ref_csum_root_kernel(
    int E,
    int N,
    const uint64_t* __restrict__ xdig,
    uint64_t* __restrict__ state_checksum,
    uint32_t* __restrict__ dev_ctr) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t h = DTR_FNV_BASIS;
    for (int e = 0; e < E; ++e) {
        h = dtr_fnv_u64_dev(h, xdig[e]);
    }
    state_checksum[0] = h;
    // Finalize run-scoped device state (graph-replay safe).
    dev_ctr[0] += (uint32_t)N;
    dev_ctr[1] = 0u;
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

extern "C" size_t solution_workspace_bytes(const DtrProblemSpec* spec) {
    if (!dtr_validate_problem_spec(spec)) return 0;
    DtrRefWorkspace L = dtr_ref_make_layout(nullptr, spec);
    return L.required_bytes;
}

extern "C" cudaError_t solution_init(
    const DtrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!dtr_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    DtrReferenceState* st = static_cast<DtrReferenceState*>(
        malloc(sizeof(DtrReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(DtrReferenceState));
    memcpy(&st->spec, spec, sizeof(DtrProblemSpec));

    const size_t E = (size_t)spec->max_E;
    const size_t bqcap = E * (size_t)spec->max_bq;

    cudaError_t err = cudaSuccess;
#define DTR_INIT_ALLOC(field, type, count)                                   \
    if (err == cudaSuccess) {                                                \
        err = cudaMalloc(reinterpret_cast<void**>(&st->field),               \
                         sizeof(type) * (count));                            \
    }
    DTR_INIT_ALLOC(credit, uint32_t, E)
    DTR_INIT_ALLOC(bl_head, uint32_t, E)
    DTR_INIT_ALLOC(bl_len, uint32_t, E)
    DTR_INIT_ALLOC(bl_gid, uint32_t, bqcap)
    DTR_INIT_ALLOC(bl_x, int8_t, bqcap* (size_t)spec->max_D)
    DTR_INIT_ALLOC(dev_ctr, uint32_t, 2)
#undef DTR_INIT_ALLOC
    if (err != cudaSuccess) {
        solution_destroy(st);
        return err;
    }

    err = solution_reset(st, stream);
    if (err != cudaSuccess) {
        solution_destroy(st);
        return err;
    }

    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    DtrReferenceState* st = static_cast<DtrReferenceState*>(state);
    const size_t E = (size_t)st->spec.max_E;
    cudaError_t err =
        cudaMemsetAsync(st->bl_head, 0, sizeof(uint32_t) * E, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->bl_len, 0, sizeof(uint32_t) * E, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->dev_ctr, 0, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;
    // dev_ctr[1] != 0 marks the fresh (credits full, empty backlog) epoch.
    return cudaMemsetAsync(st->dev_ctr + 1, 0x01, sizeof(uint32_t), stream);
}

static cudaError_t dtr_ref_enqueue(
    DtrReferenceState* st,
    const DtrRunSpec* run,
    const DtrInputs* in,
    DtrOutputs* out,
    const DtrRefWorkspace& L,
    cudaStream_t stream) {
    const int N = run->N;
    const int E = run->E;
    const int P = run->P;
    const int S = E / P;
    const int D4 = run->D / 4;
    const int padN = (int)dtr_align_up_size((size_t)N, DTR_RADIX_BLOCK);
    const int nb = padN / DTR_RADIX_BLOCK;

    cudaError_t err = cudaSuccess;
#define DTR_CHECK_LAUNCH()                                                   \
    err = cudaPeekAtLastError();                                             \
    if (err != cudaSuccess) return err;

    // Zero per-run counters (att1..att3 + m23 are laid out contiguously),
    // sentinel-fill the three key arrays (contiguous as well).
    err = cudaMemsetAsync(
        L.att1, 0,
        (size_t)(reinterpret_cast<char*>(L.m23 + 2) -
                 reinterpret_cast<char*>(L.att1)),
        stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(
        L.keys1, 0xFF,
        (size_t)(reinterpret_cast<char*>(L.ktmp) -
                 reinterpret_cast<char*>(L.keys1)),
        stream);
    if (err != cudaSuccess) return err;

    dtr_ref_refill_kernel<<<dtr_ceil_div_int(E, 128), 128, 0, stream>>>(
        E, run->ccap, run->refill, st->dev_ctr, st->credit, st->bl_head,
        st->bl_len, L.snap_head, L.snap_len, L.cred0, L.a0, L.c1);
    DTR_CHECK_LAUNCH()

    dtr_ref_s1_kernel<<<dtr_ceil_div_int(N, 8), 256, 0, stream>>>(
        N, P, D4, in->x, in->wnode, out->route_nodes, L.pn, L.bn);
    DTR_CHECK_LAUNCH()

    {
        dim3 grid(dtr_ceil_div_int(N, 32), E / 32);
        dtr_ref_s2_kernel<<<grid, 256, 0, stream>>>(
            N, E, D4, in->x, in->wexp, out->s2_logits);
        DTR_CHECK_LAUNCH()
    }

    dtr_ref_select_kernel<<<dtr_ceil_div_int(N, 8), 256, 0, stream>>>(
        N, E, S, out->s2_logits, L.pn, L.bn, out->route_pe_be, L.pe, L.be,
        L.keys1, L.att1);
    DTR_CHECK_LAUNCH()

    // Phase 1.
    dtr_ref_radix_sort(L.keys1, L.ktmp, L.hist, padN, stream);
    DTR_CHECK_LAUNCH()
    dtr_ref_mini_kernel<<<1, 1, 0, stream>>>(
        E, L.att1, L.c1, L.seg1, L.a1, L.c2);
    DTR_CHECK_LAUNCH()
    dtr_ref_reject_kernel<<<nb, 256, 0, stream>>>(
        N, E, 0, L.keys1, L.seg1, L.a1, out->s2_logits, L.be, L.keys2,
        L.att2, &L.m23[0]);
    DTR_CHECK_LAUNCH()

    // Phase 2.
    dtr_ref_radix_sort(L.keys2, L.ktmp, L.hist, padN, stream);
    DTR_CHECK_LAUNCH()
    dtr_ref_mini_kernel<<<1, 1, 0, stream>>>(
        E, L.att2, L.c2, L.seg2, L.a2, L.c3);
    DTR_CHECK_LAUNCH()
    dtr_ref_reject_kernel<<<nb, 256, 0, stream>>>(
        N, E, 0, L.keys2, L.seg2, L.a2, out->s2_logits, L.pe, L.keys3,
        L.att3, &L.m23[1]);
    DTR_CHECK_LAUNCH()

    // Phase 3 ranking.
    dtr_ref_radix_sort(L.keys3, L.ktmp, L.hist, padN, stream);
    DTR_CHECK_LAUNCH()

    dtr_ref_offsets_kernel<<<1, 1, 0, stream>>>(
        E, N, run->bq, L.a0, L.a1, L.a2, L.att3, L.c3, L.snap_head,
        L.snap_len, L.seg3, L.lb0, L.lb1, L.lb2, L.lb3, L.freec, out->counts,
        out->offsets, out->log_len, out->credit_out, st->credit, st->bl_head,
        st->bl_len);
    DTR_CHECK_LAUNCH()

    dtr_ref_emit_phase0_kernel<<<E, st->spec.max_ccap, 0, stream>>>(
        E, st->spec.max_bq, run->bq, L.a0, L.cred0, L.snap_head, L.lb0,
        st->bl_gid, out->offsets, out->event_log, out->packed_gid, L.dsrc,
        L.dexp);
    DTR_CHECK_LAUNCH()

    dtr_ref_emit_deliv_kernel<<<nb, 256, 0, stream>>>(
        1, st->dev_ctr, L.keys1, L.seg1, L.a1, L.c1, L.lb1, L.a0, nullptr,
        out->offsets, out->event_log, out->packed_gid, L.dsrc, L.dexp);
    DTR_CHECK_LAUNCH()

    dtr_ref_emit_deliv_kernel<<<nb, 256, 0, stream>>>(
        2, st->dev_ctr, L.keys2, L.seg2, L.a2, L.c2, L.lb2, L.a0, L.a1,
        out->offsets, out->event_log, out->packed_gid, L.dsrc, L.dexp);
    DTR_CHECK_LAUNCH()

    // FFN BEFORE phase-3 enqueue: phase-0 deliveries read their stored x
    // rows from backlog slots that new enqueues may reuse.
    dtr_ref_ffn_kernel<<<E * run->ccap, 256, 0, stream>>>(
        E, run->D, run->H, run->qshift, st->spec.max_D, out->offsets, L.dsrc,
        L.dexp, in->x, st->bl_x, in->w1, in->w2, out->packed_out);
    DTR_CHECK_LAUNCH()

    dtr_ref_emit_phase3_kernel<<<dtr_ceil_div_int(padN, 8), 256, 0, stream>>>(
        st->dev_ctr, D4, st->spec.max_bq, st->spec.max_D, run->bq, L.keys3,
        L.seg3, L.freec, L.c3, L.lb3, L.snap_head, L.snap_len, in->x,
        out->event_log, st->bl_gid, st->bl_x);
    DTR_CHECK_LAUNCH()

    dtr_ref_csum_entries_kernel<<<E, st->spec.max_bq, 0, stream>>>(
        E, run->D, st->spec.max_bq, st->spec.max_D, run->bq, st->bl_head,
        st->bl_len, st->bl_gid, st->bl_x, L.edig);
    DTR_CHECK_LAUNCH()

    dtr_ref_csum_expert_kernel<<<dtr_ceil_div_int(E, 128), 128, 0, stream>>>(
        E, st->spec.max_bq, st->credit, st->bl_len, L.edig, L.xdig);
    DTR_CHECK_LAUNCH()

    dtr_ref_csum_root_kernel<<<1, 1, 0, stream>>>(
        E, N, L.xdig, out->state_checksum, st->dev_ctr);
    DTR_CHECK_LAUNCH()

#undef DTR_CHECK_LAUNCH

    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const DtrRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !dtr_validate_run_spec(run) || !inputs_void ||
        !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    DtrReferenceState* st = static_cast<DtrReferenceState*>(state);
    const DtrInputs* in = static_cast<const DtrInputs*>(inputs_void);
    DtrOutputs* out = static_cast<DtrOutputs*>(outputs_void);

    if (run->N > st->spec.max_N || run->D > st->spec.max_D ||
        run->H > st->spec.max_H || run->E > st->spec.max_E ||
        run->P > st->spec.max_P || run->ccap > st->spec.max_ccap ||
        run->bq > st->spec.max_bq) {
        return cudaErrorInvalidValue;
    }
    if (!in->x || !in->wnode || !in->wexp || !in->w1 || !in->w2 ||
        !out->s2_logits || !out->route_nodes || !out->route_pe_be ||
        !out->log_len || !out->event_log || !out->counts || !out->offsets ||
        !out->packed_gid || !out->packed_out || !out->credit_out ||
        !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    DtrRefWorkspace L = dtr_ref_make_layout(workspace, &st->spec);
    if (workspace_bytes < L.required_bytes) return cudaErrorInvalidValue;

    // The pipeline is dozens of small kernels; replaying a cached CUDA
    // graph removes the per-launch overhead. Everything run-varying that
    // is not part of the cache key (gid base, fresh flag) lives in device
    // memory, so replay is exact.
    const bool cache_hit =
        st->graph_valid &&
        memcmp(&st->cached_run, run, sizeof(DtrRunSpec)) == 0 &&
        memcmp(&st->cached_in, in, sizeof(DtrInputs)) == 0 &&
        memcmp(&st->cached_out, out, sizeof(DtrOutputs)) == 0 &&
        st->cached_ws == workspace && st->cached_wsb == workspace_bytes;

    if (!cache_hit) {
        if (st->graph_valid) {
            cudaGraphExecDestroy(st->graph_exec);
            st->graph_valid = 0;
        }
        cudaError_t err = cudaStreamBeginCapture(
            stream, cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess) {
            // Capture unavailable (e.g., legacy stream): run directly.
            cudaGetLastError();
            return dtr_ref_enqueue(st, run, in, out, L, stream);
        }
        err = dtr_ref_enqueue(st, run, in, out, L, stream);
        cudaGraph_t graph = nullptr;
        cudaError_t err_end = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess || err_end != cudaSuccess) {
            if (graph) cudaGraphDestroy(graph);
            return err != cudaSuccess ? err : err_end;
        }
        err = cudaGraphInstantiate(&st->graph_exec, graph, 0);
        cudaGraphDestroy(graph);
        if (err != cudaSuccess) return err;
        st->graph_valid = 1;
        memcpy(&st->cached_run, run, sizeof(DtrRunSpec));
        memcpy(&st->cached_in, in, sizeof(DtrInputs));
        memcpy(&st->cached_out, out, sizeof(DtrOutputs));
        st->cached_ws = workspace;
        st->cached_wsb = workspace_bytes;
    }

    return cudaGraphLaunch(st->graph_exec, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    DtrReferenceState* st = static_cast<DtrReferenceState*>(state);
    if (st->graph_valid) cudaGraphExecDestroy(st->graph_exec);
    if (st->credit) cudaFree(st->credit);
    if (st->bl_head) cudaFree(st->bl_head);
    if (st->bl_len) cudaFree(st->bl_len);
    if (st->bl_gid) cudaFree(st->bl_gid);
    if (st->bl_x) cudaFree(st->bl_x);
    if (st->dev_ctr) cudaFree(st->dev_ctr);
    free(st);
}
