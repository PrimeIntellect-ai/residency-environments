// ============================================================================
// file: naive_ref.cu
// Clean, correct, parallel-but-unoptimized implementation used to calibrate
// the perf gate. Scalar per-byte routing GEMMs (no dp4a, no staging), global
// bitonic sorts for the admission rankings, scalar two-pass FFN, serial
// checksum folds per level.
// ============================================================================

#include "dualtier_credit_router_common.h"

#include <stdlib.h>
#include <string.h>

#define DTR_NAIVE_SENTINEL (~0ULL)

struct DtrNaiveState {
    DtrProblemSpec spec;
    uint32_t token_counter;
    int fresh;
    uint32_t* credit;
    uint32_t* bl_head;
    uint32_t* bl_len;
    uint32_t* bl_gid;
    int8_t* bl_x;
};

static size_t dtr_naive_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

struct DtrNaiveWorkspace {
    int32_t* s1;          // [maxN * maxP]
    uint32_t* snap_head;  // [E]
    uint32_t* snap_len;   // [E]
    uint32_t* cred0;      // [E]
    uint32_t* c1;         // [E]
    uint32_t* c2;         // [E]
    uint32_t* c3;         // [E]
    uint32_t* a0;         // [E]
    uint32_t* a1;         // [E]
    uint32_t* a2;         // [E]
    uint32_t* att1;       // [E]
    uint32_t* att2;       // [E]
    uint32_t* att3;       // [E]
    uint32_t* m23;        // [2]
    uint32_t* seg1;       // [E]
    uint32_t* seg2;       // [E]
    uint32_t* seg3;       // [E]
    uint32_t* lb0;        // [E]
    uint32_t* lb1;        // [E]
    uint32_t* lb2;        // [E]
    uint32_t* lb3;        // [E]
    uint32_t* freec;      // [E]
    uint32_t* pn;         // [maxN]
    uint32_t* bn;         // [maxN]
    uint32_t* pe;         // [maxN]
    uint32_t* be;         // [maxN]
    uint64_t* keys1;      // [pow2N]
    uint64_t* keys2;      // [pow2N]
    uint64_t* keys3;      // [pow2N]
    uint32_t* dsrc;       // [max_E * max_ccap]
    uint32_t* dexp;       // [max_E * max_ccap]
    uint8_t* hbuf;        // [max_E * max_ccap * max_H]
    uint64_t* edig;       // [max_E * max_bq]
    uint64_t* xdig;       // [E]
    size_t required_bytes;
};

static DtrNaiveWorkspace dtr_naive_make_layout(
    void* workspace,
    const DtrProblemSpec* spec) {
    DtrNaiveWorkspace L{};
    char* base = static_cast<char*>(workspace);
    const size_t E = (size_t)spec->max_E;
    const size_t maxN = (size_t)spec->max_N;
    const size_t p2 = dtr_naive_pow2(maxN);
    const size_t cap = (size_t)spec->max_E * (size_t)spec->max_ccap;

    size_t off = 0;
#define DTR_WS_FIELD(field, type, count)                                     \
    off = dtr_align_up_size(off, 128);                                       \
    L.field = reinterpret_cast<type*>(base + off);                           \
    off += sizeof(type) * (count);

    DTR_WS_FIELD(s1, int32_t, maxN*(size_t)spec->max_P)
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
    DTR_WS_FIELD(keys1, uint64_t, p2)
    DTR_WS_FIELD(keys2, uint64_t, p2)
    DTR_WS_FIELD(keys3, uint64_t, p2)
    DTR_WS_FIELD(dsrc, uint32_t, cap)
    DTR_WS_FIELD(dexp, uint32_t, cap)
    DTR_WS_FIELD(hbuf, uint8_t, cap*(size_t)spec->max_H)
    DTR_WS_FIELD(edig, uint64_t, (size_t)spec->max_E * spec->max_bq)
    DTR_WS_FIELD(xdig, uint64_t, E)
#undef DTR_WS_FIELD

    off = dtr_align_up_size(off, 128);
    L.required_bytes = off;
    return L;
}

__device__ __forceinline__ uint64_t dtr_naive_pack_key(
    int e, uint32_t inv, int t) {
    return ((uint64_t)(uint32_t)e << 36) | ((uint64_t)inv << 14) |
           (uint64_t)(uint32_t)t;
}

__device__ __forceinline__ uint32_t dtr_naive_inv_score(int32_t s) {
    return 0x3FFFFFu - (uint32_t)(s + DTR_SCORE_BIAS);
}

__device__ __forceinline__ uint64_t dtr_naive_fnv_byte(uint64_t h, uint8_t b) {
    return (h ^ (uint64_t)b) * DTR_FNV_PRIME;
}

__device__ __forceinline__ uint64_t dtr_naive_fnv_u32(uint64_t h, uint32_t v) {
    for (int m = 0; m < 4; ++m) {
        h = dtr_naive_fnv_byte(h, (uint8_t)(v >> (8 * m)));
    }
    return h;
}

__device__ __forceinline__ uint64_t dtr_naive_fnv_u64(uint64_t h, uint64_t v) {
    for (int m = 0; m < 8; ++m) {
        h = dtr_naive_fnv_byte(h, (uint8_t)(v >> (8 * m)));
    }
    return h;
}

// ---------------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------------

__global__ void dtr_naive_refill_kernel(
    int E, int ccap, int refill, int fresh,
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
    const uint32_t head = fresh ? 0u : bl_head[e];
    const uint32_t len = fresh ? 0u : bl_len[e];
    uint32_t c = fresh ? (uint32_t)ccap : credit[e] + (uint32_t)refill;
    if (c > (uint32_t)ccap) c = (uint32_t)ccap;
    const uint32_t a = len < c ? len : c;
    snap_head[e] = head;
    snap_len[e] = len;
    cred0[e] = c;
    a0[e] = a;
    c1[e] = c - a;
}

__global__ void dtr_naive_s1_kernel(
    int N, int P, int D,
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ wnode,
    int32_t* __restrict__ s1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * P) return;
    const int t = idx / P;
    const int p = idx - t * P;
    int32_t s = 0;
    for (int d = 0; d < D; ++d) {
        s += (int32_t)x[(size_t)t * D + d] * (int32_t)wnode[(size_t)p * D + d];
    }
    s1[idx] = s;
}

__global__ void dtr_naive_top2_kernel(
    int N, int P,
    const int32_t* __restrict__ s1,
    int32_t* __restrict__ route_nodes,
    uint32_t* __restrict__ pn_arr,
    uint32_t* __restrict__ bn_arr) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    int pn = -1, bn = -1;
    int32_t pn_s = 0, bn_s = 0;
    for (int p = 0; p < P; ++p) {
        const int32_t s = s1[(size_t)t * P + p];
        if (pn < 0 || s > pn_s) {
            bn = pn;
            bn_s = pn_s;
            pn = p;
            pn_s = s;
        } else if (bn < 0 || s > bn_s) {
            bn = p;
            bn_s = s;
        }
    }
    route_nodes[t] = (pn << 16) | bn;
    pn_arr[t] = (uint32_t)pn;
    bn_arr[t] = (uint32_t)bn;
}

__global__ void dtr_naive_s2_kernel(
    int N, int E, int D,
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ wexp,
    int32_t* __restrict__ s2_logits) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * E) return;
    const int t = idx / E;
    const int e = idx - t * E;
    int32_t s = 0;
    for (int d = 0; d < D; ++d) {
        s += (int32_t)x[(size_t)t * D + d] * (int32_t)wexp[(size_t)e * D + d];
    }
    s2_logits[idx] = s;
}

__global__ void dtr_naive_select_kernel(
    int N, int E, int S,
    const int32_t* __restrict__ s2_logits,
    const uint32_t* __restrict__ pn_arr,
    const uint32_t* __restrict__ bn_arr,
    int32_t* __restrict__ route_pe_be,
    uint32_t* __restrict__ pe_arr,
    uint32_t* __restrict__ be_arr,
    uint64_t* __restrict__ keys1,
    uint32_t* __restrict__ att1) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    const int pn = (int)pn_arr[t];
    const int bn = (int)bn_arr[t];
    const int32_t* row = s2_logits + (size_t)t * E;
    int pe = pn * S, be = bn * S;
    for (int q = 1; q < S; ++q) {
        if (row[pn * S + q] > row[pe]) pe = pn * S + q;
        if (row[bn * S + q] > row[be]) be = bn * S + q;
    }
    route_pe_be[t] = (pe << 16) | be;
    pe_arr[t] = (uint32_t)pe;
    be_arr[t] = (uint32_t)be;
    keys1[t] = dtr_naive_pack_key(pe, dtr_naive_inv_score(row[pe]), t);
    atomicAdd(&att1[pe], 1u);
}

__global__ void dtr_naive_bitonic_kernel(
    uint64_t* __restrict__ keys, int n, int k, int j) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int l = i ^ j;
    if (l <= i) return;
    const uint64_t a = keys[i];
    const uint64_t b = keys[l];
    const bool up = (i & k) == 0;
    if (up ? (a > b) : (a < b)) {
        keys[i] = b;
        keys[l] = a;
    }
}

static void dtr_naive_sort(
    uint64_t* keys, int p2, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (p2 + threads - 1) / threads;
    for (int k = 2; k <= p2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            dtr_naive_bitonic_kernel<<<blocks, threads, 0, stream>>>(
                keys, p2, k, j);
        }
    }
}

__global__ void dtr_naive_mini_kernel(
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

__global__ void dtr_naive_reject_kernel(
    int p2, int E,
    const uint64_t* __restrict__ sorted_keys,
    const uint32_t* __restrict__ seg,
    const uint32_t* __restrict__ adm,
    const int32_t* __restrict__ s2_logits,
    const uint32_t* __restrict__ next_target,
    uint64_t* __restrict__ next_keys,
    uint32_t* __restrict__ next_att,
    uint32_t* __restrict__ mcounter) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= p2) return;
    const uint64_t key = sorted_keys[g];
    if (key == DTR_NAIVE_SENTINEL) return;
    const int e = (int)((key >> 36) & 0x7F);
    const uint32_t r = (uint32_t)g - seg[e];
    if (r < adm[e]) return;
    const int t = (int)(key & 0x3FFF);
    const int ne = (int)next_target[t];
    const int32_t s = s2_logits[(size_t)t * E + ne];
    const uint32_t pos = atomicAdd(mcounter, 1u);
    next_keys[pos] = dtr_naive_pack_key(ne, dtr_naive_inv_score(s), t);
    atomicAdd(&next_att[ne], 1u);
}

__global__ void dtr_naive_offsets_kernel(
    int E, int N, int bq,
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

__global__ void dtr_naive_emit_phase0_kernel(
    int E, int maxBQ, int bq,
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

__global__ void dtr_naive_emit_deliv_kernel(
    int phase, uint32_t base_gid, int p2,
    const uint64_t* __restrict__ sorted_keys,
    const uint32_t* __restrict__ seg,
    const uint32_t* __restrict__ adm,
    const uint32_t* __restrict__ cred_start,
    const uint32_t* __restrict__ lb,
    const uint32_t* __restrict__ a0,
    const uint32_t* __restrict__ a1,
    const int32_t* __restrict__ offsets,
    uint64_t* __restrict__ event_log,
    int32_t* __restrict__ packed_gid,
    uint32_t* __restrict__ dsrc,
    uint32_t* __restrict__ dexp) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= p2) return;
    const uint64_t key = sorted_keys[g];
    if (key == DTR_NAIVE_SENTINEL) return;
    const int e = (int)((key >> 36) & 0x7F);
    const uint32_t r = (uint32_t)g - seg[e];
    if (r >= adm[e]) return;
    const int t = (int)(key & 0x3FFF);
    const uint32_t gid = base_gid + (uint32_t)t;
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

__global__ void dtr_naive_emit_phase3_kernel(
    uint32_t base_gid, int D, int maxBQ, int maxD, int bq, int p2,
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
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= p2) return;
    const uint64_t key = sorted_keys[g];
    if (key == DTR_NAIVE_SENTINEL) return;
    const int e = (int)((key >> 36) & 0x7F);
    const uint32_t r = (uint32_t)g - seg3[e];
    const int t = (int)(key & 0x3FFF);
    const uint32_t gid = base_gid + (uint32_t)t;
    const bool queued = r < freec[e];
    event_log[lb3[e] + r] = dtr_make_log_word(
        gid, e, queued ? DTR_ACT_QUEUED : DTR_ACT_DROPPED, 3,
        c3[e] & 0xFFFFu);
    if (!queued) return;
    const uint32_t slot = (snap_head[e] + snap_len[e] + r) % (uint32_t)bq;
    bl_gid[(size_t)e * maxBQ + slot] = gid;
    int8_t* dst = bl_x + ((size_t)e * maxBQ + slot) * maxD;
    for (int d = 0; d < D; ++d) {
        dst[d] = x[(size_t)t * D + d];
    }
}

__global__ void dtr_naive_ffn_h_kernel(
    int E, int D, int H, int qshift, int maxD, int maxH,
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ dsrc,
    const uint32_t* __restrict__ dexp,
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ bl_x,
    const int8_t* __restrict__ w1,
    uint8_t* __restrict__ hbuf) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = offsets[E];
    if (idx >= total * H) return;
    const int pos = idx / H;
    const int j = idx - pos * H;
    const uint32_t src = dsrc[pos];
    const int e = (int)dexp[pos];
    const int8_t* xv = (src & 0x80000000u)
        ? bl_x + (size_t)(src & 0x7FFFFFFFu) * maxD
        : x + (size_t)src * D;
    int32_t acc = 0;
    const int8_t* w1row = w1 + ((size_t)e * H + j) * D;
    for (int d = 0; d < D; ++d) {
        acc += (int32_t)w1row[d] * (int32_t)xv[d];
    }
    acc = acc > 0 ? acc : 0;
    acc >>= qshift;
    if (acc > 127) acc = 127;
    hbuf[(size_t)pos * maxH + j] = (uint8_t)acc;
}

__global__ void dtr_naive_ffn_out_kernel(
    int E, int D, int H, int maxH,
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ dexp,
    const int8_t* __restrict__ w2,
    const uint8_t* __restrict__ hbuf,
    int32_t* __restrict__ packed_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = offsets[E];
    if (idx >= total * D) return;
    const int pos = idx / D;
    const int d = idx - pos * D;
    const int e = (int)dexp[pos];
    const int8_t* w2row = w2 + ((size_t)e * D + d) * H;
    const uint8_t* h = hbuf + (size_t)pos * maxH;
    int32_t acc = 0;
    for (int j = 0; j < H; ++j) {
        acc += (int32_t)w2row[j] * (int32_t)h[j];
    }
    packed_out[(size_t)pos * D + d] = acc;
}

__global__ void dtr_naive_csum_entries_kernel(
    int E, int D, int maxBQ, int maxD, int bq,
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
    h = dtr_naive_fnv_u32(h, bl_gid[(size_t)e * maxBQ + slot]);
    const int8_t* row = bl_x + ((size_t)e * maxBQ + slot) * maxD;
    for (int d = 0; d < D; ++d) {
        h = dtr_naive_fnv_byte(h, (uint8_t)row[d]);
    }
    edig[(size_t)e * maxBQ + q] = h;
}

__global__ void dtr_naive_csum_expert_kernel(
    int E, int maxBQ,
    const uint32_t* __restrict__ credit,
    const uint32_t* __restrict__ bl_len,
    const uint64_t* __restrict__ edig,
    uint64_t* __restrict__ xdig) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    uint64_t h = DTR_FNV_BASIS;
    h = dtr_naive_fnv_u32(h, credit[e]);
    h = dtr_naive_fnv_u32(h, bl_len[e]);
    const uint32_t len = bl_len[e];
    for (uint32_t q = 0; q < len; ++q) {
        h = dtr_naive_fnv_u64(h, edig[(size_t)e * maxBQ + q]);
    }
    xdig[e] = h;
}

__global__ void dtr_naive_csum_root_kernel(
    int E,
    const uint64_t* __restrict__ xdig,
    uint64_t* __restrict__ state_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t h = DTR_FNV_BASIS;
    for (int e = 0; e < E; ++e) {
        h = dtr_naive_fnv_u64(h, xdig[e]);
    }
    state_checksum[0] = h;
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

extern "C" size_t solution_workspace_bytes(const DtrProblemSpec* spec) {
    if (!dtr_validate_problem_spec(spec)) return 0;
    DtrNaiveWorkspace L = dtr_naive_make_layout(nullptr, spec);
    return L.required_bytes;
}

extern "C" cudaError_t solution_init(
    const DtrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!dtr_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }
    DtrNaiveState* st =
        static_cast<DtrNaiveState*>(malloc(sizeof(DtrNaiveState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(DtrNaiveState));
    memcpy(&st->spec, spec, sizeof(DtrProblemSpec));
    st->token_counter = 0;
    st->fresh = 1;

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
    DTR_INIT_ALLOC(bl_x, int8_t, bqcap*(size_t)spec->max_D)
#undef DTR_INIT_ALLOC
    if (err != cudaSuccess) {
        solution_destroy(st);
        return err;
    }
    err = cudaMemsetAsync(st->bl_head, 0, sizeof(uint32_t) * E, stream);
    if (err == cudaSuccess) {
        err = cudaMemsetAsync(st->bl_len, 0, sizeof(uint32_t) * E, stream);
    }
    if (err != cudaSuccess) {
        solution_destroy(st);
        return err;
    }
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    DtrNaiveState* st = static_cast<DtrNaiveState*>(state);
    st->token_counter = 0;
    st->fresh = 1;
    const size_t E = (size_t)st->spec.max_E;
    cudaError_t err =
        cudaMemsetAsync(st->bl_head, 0, sizeof(uint32_t) * E, stream);
    if (err != cudaSuccess) return err;
    return cudaMemsetAsync(st->bl_len, 0, sizeof(uint32_t) * E, stream);
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
    DtrNaiveState* st = static_cast<DtrNaiveState*>(state);
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

    DtrNaiveWorkspace L = dtr_naive_make_layout(workspace, &st->spec);
    if (workspace_bytes < L.required_bytes) return cudaErrorInvalidValue;

    const int N = run->N;
    const int E = run->E;
    const int P = run->P;
    const int S = E / P;
    const int D = run->D;
    const int H = run->H;
    const int p2 = (int)dtr_naive_pow2((size_t)N);
    const uint32_t base_gid = st->token_counter;
    const int cap = E * run->ccap;

    cudaError_t err = cudaSuccess;
#define DTR_CHECK_LAUNCH()                                                   \
    err = cudaPeekAtLastError();                                             \
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(L.att1, 0, sizeof(uint32_t) * E, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.att2, 0, sizeof(uint32_t) * E, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.att3, 0, sizeof(uint32_t) * E, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.m23, 0, sizeof(uint32_t) * 2, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.keys1, 0xFF, sizeof(uint64_t) * p2, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.keys2, 0xFF, sizeof(uint64_t) * p2, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.keys3, 0xFF, sizeof(uint64_t) * p2, stream);
    if (err != cudaSuccess) return err;

    dtr_naive_refill_kernel<<<dtr_ceil_div_int(E, 128), 128, 0, stream>>>(
        E, run->ccap, run->refill, st->fresh, st->credit, st->bl_head,
        st->bl_len, L.snap_head, L.snap_len, L.cred0, L.a0, L.c1);
    DTR_CHECK_LAUNCH()

    dtr_naive_s1_kernel<<<dtr_ceil_div_int(N * P, 256), 256, 0, stream>>>(
        N, P, D, in->x, in->wnode, L.s1);
    DTR_CHECK_LAUNCH()
    dtr_naive_top2_kernel<<<dtr_ceil_div_int(N, 256), 256, 0, stream>>>(
        N, P, L.s1, out->route_nodes, L.pn, L.bn);
    DTR_CHECK_LAUNCH()
    dtr_naive_s2_kernel<<<dtr_ceil_div_int(N * E, 256), 256, 0, stream>>>(
        N, E, D, in->x, in->wexp, out->s2_logits);
    DTR_CHECK_LAUNCH()
    dtr_naive_select_kernel<<<dtr_ceil_div_int(N, 256), 256, 0, stream>>>(
        N, E, S, out->s2_logits, L.pn, L.bn, out->route_pe_be, L.pe, L.be,
        L.keys1, L.att1);
    DTR_CHECK_LAUNCH()

    dtr_naive_sort(L.keys1, p2, stream);
    DTR_CHECK_LAUNCH()
    dtr_naive_mini_kernel<<<1, 1, 0, stream>>>(
        E, L.att1, L.c1, L.seg1, L.a1, L.c2);
    DTR_CHECK_LAUNCH()
    dtr_naive_reject_kernel<<<dtr_ceil_div_int(p2, 256), 256, 0, stream>>>(
        p2, E, L.keys1, L.seg1, L.a1, out->s2_logits, L.be, L.keys2, L.att2,
        &L.m23[0]);
    DTR_CHECK_LAUNCH()

    dtr_naive_sort(L.keys2, p2, stream);
    DTR_CHECK_LAUNCH()
    dtr_naive_mini_kernel<<<1, 1, 0, stream>>>(
        E, L.att2, L.c2, L.seg2, L.a2, L.c3);
    DTR_CHECK_LAUNCH()
    dtr_naive_reject_kernel<<<dtr_ceil_div_int(p2, 256), 256, 0, stream>>>(
        p2, E, L.keys2, L.seg2, L.a2, out->s2_logits, L.pe, L.keys3, L.att3,
        &L.m23[1]);
    DTR_CHECK_LAUNCH()

    dtr_naive_sort(L.keys3, p2, stream);
    DTR_CHECK_LAUNCH()

    dtr_naive_offsets_kernel<<<1, 1, 0, stream>>>(
        E, N, run->bq, L.a0, L.a1, L.a2, L.att3, L.c3, L.snap_head,
        L.snap_len, L.seg3, L.lb0, L.lb1, L.lb2, L.lb3, L.freec, out->counts,
        out->offsets, out->log_len, out->credit_out, st->credit, st->bl_head,
        st->bl_len);
    DTR_CHECK_LAUNCH()

    dtr_naive_emit_phase0_kernel<<<E, st->spec.max_ccap, 0, stream>>>(
        E, st->spec.max_bq, run->bq, L.a0, L.cred0, L.snap_head, L.lb0,
        st->bl_gid, out->offsets, out->event_log, out->packed_gid, L.dsrc,
        L.dexp);
    DTR_CHECK_LAUNCH()
    dtr_naive_emit_deliv_kernel<<<dtr_ceil_div_int(p2, 256), 256, 0,
                                  stream>>>(
        1, base_gid, p2, L.keys1, L.seg1, L.a1, L.c1, L.lb1, L.a0, nullptr,
        out->offsets, out->event_log, out->packed_gid, L.dsrc, L.dexp);
    DTR_CHECK_LAUNCH()
    dtr_naive_emit_deliv_kernel<<<dtr_ceil_div_int(p2, 256), 256, 0,
                                  stream>>>(
        2, base_gid, p2, L.keys2, L.seg2, L.a2, L.c2, L.lb2, L.a0, L.a1,
        out->offsets, out->event_log, out->packed_gid, L.dsrc, L.dexp);
    DTR_CHECK_LAUNCH()

    // FFN before phase-3 enqueue (phase-0 rows are read from backlog slots
    // that new enqueues may reuse).
    dtr_naive_ffn_h_kernel<<<dtr_ceil_div_int(cap * H, 256), 256, 0,
                             stream>>>(
        E, D, H, run->qshift, st->spec.max_D, st->spec.max_H, out->offsets,
        L.dsrc, L.dexp, in->x, st->bl_x, in->w1, L.hbuf);
    DTR_CHECK_LAUNCH()
    dtr_naive_ffn_out_kernel<<<dtr_ceil_div_int(cap * D, 256), 256, 0,
                               stream>>>(
        E, D, H, st->spec.max_H, out->offsets, L.dexp, in->w2, L.hbuf,
        out->packed_out);
    DTR_CHECK_LAUNCH()

    dtr_naive_emit_phase3_kernel<<<dtr_ceil_div_int(p2, 256), 256, 0,
                                   stream>>>(
        base_gid, D, st->spec.max_bq, st->spec.max_D, run->bq, p2, L.keys3,
        L.seg3, L.freec, L.c3, L.lb3, L.snap_head, L.snap_len, in->x,
        out->event_log, st->bl_gid, st->bl_x);
    DTR_CHECK_LAUNCH()

    dtr_naive_csum_entries_kernel<<<E, st->spec.max_bq, 0, stream>>>(
        E, D, st->spec.max_bq, st->spec.max_D, run->bq, st->bl_head,
        st->bl_len, st->bl_gid, st->bl_x, L.edig);
    DTR_CHECK_LAUNCH()
    dtr_naive_csum_expert_kernel<<<dtr_ceil_div_int(E, 128), 128, 0,
                                   stream>>>(
        E, st->spec.max_bq, st->credit, st->bl_len, L.edig, L.xdig);
    DTR_CHECK_LAUNCH()
    dtr_naive_csum_root_kernel<<<1, 1, 0, stream>>>(
        E, L.xdig, out->state_checksum);
    DTR_CHECK_LAUNCH()

#undef DTR_CHECK_LAUNCH

    st->token_counter += (uint32_t)N;
    st->fresh = 0;
    return cudaSuccess;
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    DtrNaiveState* st = static_cast<DtrNaiveState*>(state);
    if (st->credit) cudaFree(st->credit);
    if (st->bl_head) cudaFree(st->bl_head);
    if (st->bl_len) cudaFree(st->bl_len);
    if (st->bl_gid) cudaFree(st->bl_gid);
    if (st->bl_x) cudaFree(st->bl_x);
    free(st);
}
