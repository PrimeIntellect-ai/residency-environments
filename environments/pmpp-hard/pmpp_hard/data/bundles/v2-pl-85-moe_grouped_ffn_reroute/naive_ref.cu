// ============================================================================
// file: naive_ref.cu
// Clean, correct, deliberately unoptimized implementation of the
// moe_grouped_ffn_reroute contract. Used only to calibrate the perf gate
// (reference must be >= 5x faster). Scalar GEMMs (no dp4a, no tiling),
// global-memory bitonic sort for dispatch, straightforward scalar kernels
// elsewhere, serial root checksum fold.
// ============================================================================

#include "moe_grouped_ffn_reroute_common.h"

#include <stdlib.h>
#include <string.h>

struct MgfNaiveState {
    MgfProblemSpec spec;
};

static size_t mgf_naive_pow2(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

struct MgfNaiveLayout {
    uint64_t* rec1;   // [pow2(maxM)]
    uint64_t* rec2;   // [pow2(maxM)]
    int16_t* cand;    // [max_N, 8]
    uint32_t* cnt1;
    uint32_t* kept1;
    uint32_t* seg1;
    uint32_t* cnt2;
    uint32_t* kept2;
    uint32_t* seg2;
    uint32_t* m2;
    uint8_t* hbuf;    // [max_E * max_cap * max_H]
    uint64_t* digests;
    size_t required_bytes;
};

static MgfNaiveLayout mgf_naive_make_layout(
    void* workspace,
    const MgfProblemSpec* spec) {
    MgfNaiveLayout L{};
    char* base = static_cast<char*>(workspace);
    const size_t maxM = (size_t)spec->max_N * (size_t)spec->max_K;
    const size_t maxMp2 = mgf_naive_pow2(maxM);
    const size_t maxDig = (size_t)mgf_ceil_div_int(spec->max_N, MGF_CSUM_ROWS);

    size_t off = 0;
#define MGF_NWS_FIELD(field, type, count)                                   \
    off = mgf_align_up_size(off, 128);                                      \
    L.field = reinterpret_cast<type*>(base + off);                          \
    off += sizeof(type) * (count);

    MGF_NWS_FIELD(rec1, uint64_t, maxMp2)
    MGF_NWS_FIELD(rec2, uint64_t, maxMp2)
    MGF_NWS_FIELD(cand, int16_t, (size_t)spec->max_N * 8)
    MGF_NWS_FIELD(cnt1, uint32_t, (size_t)spec->max_E)
    MGF_NWS_FIELD(kept1, uint32_t, (size_t)spec->max_E)
    MGF_NWS_FIELD(seg1, uint32_t, (size_t)spec->max_E)
    MGF_NWS_FIELD(cnt2, uint32_t, (size_t)spec->max_E)
    MGF_NWS_FIELD(kept2, uint32_t, (size_t)spec->max_E)
    MGF_NWS_FIELD(seg2, uint32_t, (size_t)spec->max_E)
    MGF_NWS_FIELD(m2, uint32_t, 1)
    MGF_NWS_FIELD(hbuf, uint8_t,
                  (size_t)spec->max_E * (size_t)spec->max_cap *
                      (size_t)spec->max_H)
    MGF_NWS_FIELD(digests, uint64_t, maxDig)
#undef MGF_NWS_FIELD

    off = mgf_align_up_size(off, 128);
    L.required_bytes = off;
    return L;
}

__device__ __forceinline__ uint64_t mgf_naive_make_key(
    int e, int32_t gate, int t, int k) {
    const uint32_t biased = (uint32_t)(gate + MGF_LOGIT_BIAS);
    const uint32_t inv = 0x3FFFFFu - biased;
    return ((uint64_t)e << 39) | ((uint64_t)inv << 17) |
           ((uint64_t)t << 2) | (uint64_t)k;
}

__device__ __forceinline__ void mgf_naive_decode_key(
    uint64_t key, int* e, int32_t* gate, int* t, int* k) {
    *k = (int)(key & 3u);
    *t = (int)((key >> 2) & 0x7FFFu);
    const uint32_t inv = (uint32_t)((key >> 17) & 0x3FFFFFu);
    *gate = (int32_t)(0x3FFFFFu - inv) - MGF_LOGIT_BIAS;
    *e = (int)((key >> 39) & 0x7Fu);
}

// 1. Scalar router logits: one thread per (t, e).
__global__ void mgf_naive_logits_kernel(
    int N,
    int D,
    int E,
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ wr,
    int32_t* __restrict__ logits) {
    const long long idx =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)N * E) return;
    const int t = (int)(idx / E);
    const int e = (int)(idx - (long long)t * E);
    int32_t acc = 0;
    for (int d = 0; d < D; ++d) {
        acc += (int32_t)x[(size_t)t * D + d] * (int32_t)wr[(size_t)e * D + d];
    }
    logits[(size_t)t * E + e] = acc;
}

// 2. Scalar routing: one thread per token.
__global__ void mgf_naive_route_kernel(
    int N,
    int E,
    int G,
    int g_sel,
    int K,
    const int32_t* __restrict__ logits,
    int16_t* __restrict__ cand,
    uint64_t* __restrict__ rec1,
    uint32_t* __restrict__ cnt1) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    const int S = E / G;
    const int n_sel = g_sel * S;
    const int n_store = min(2 * K, n_sel);
    const int32_t* lrow = logits + (size_t)t * E;

    // Group scores.
    int32_t gs[16];
    for (int g = 0; g < G; ++g) {
        int32_t best = lrow[g * S];
        for (int j = 1; j < S; ++j) {
            const int32_t v = lrow[g * S + j];
            if (v > best) best = v;
        }
        gs[g] = best;
    }

    // Top g_sel groups.
    uint32_t sel_mask = 0;
    for (int r = 0; r < g_sel; ++r) {
        int best_g = -1;
        int32_t best_s = 0;
        for (int g = 0; g < G; ++g) {
            if (sel_mask & (1u << g)) continue;
            if (best_g < 0 || gs[g] > best_s) {
                best_s = gs[g];
                best_g = g;
            }
        }
        sel_mask |= 1u << best_g;
    }

    // Candidate ranking within selected groups.
    uint64_t chosen[2] = {0ULL, 0ULL};
    for (int j = 0; j < n_store; ++j) {
        int best_e = -1;
        int32_t best_l = 0;
        for (int g = 0; g < G; ++g) {
            if (!(sel_mask & (1u << g))) continue;
            for (int q = 0; q < S; ++q) {
                const int e = g * S + q;
                if (chosen[e >> 6] & (1ULL << (e & 63))) continue;
                const int32_t v = lrow[e];
                if (best_e < 0 || v > best_l) {
                    best_l = v;
                    best_e = e;
                }
            }
        }
        chosen[best_e >> 6] |= 1ULL << (best_e & 63);
        cand[(size_t)t * 8 + j] = (int16_t)best_e;
        if (j < K) {
            rec1[(size_t)t * K + j] = mgf_naive_make_key(best_e, best_l, t, j);
            atomicAdd(&cnt1[best_e], 1u);
        }
    }
    for (int j = n_store; j < 8; ++j) {
        cand[(size_t)t * 8 + j] = -1;
    }
}

// 3. Global bitonic compare-swap (ascending), padded to a power of two.
__global__ void mgf_naive_bitonic_kernel(
    uint64_t* __restrict__ a,
    int n,
    int k,
    int j) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int ixj = i ^ j;
    if (ixj <= i) return;
    const bool up = (i & k) == 0;
    const uint64_t vi = a[i];
    const uint64_t vx = a[ixj];
    if (up ? (vi > vx) : (vi < vx)) {
        a[i] = vx;
        a[ixj] = vi;
    }
}

__global__ void mgf_naive_seg1_kernel(
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

__global__ void mgf_naive_classify_kernel(
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
    mgf_naive_decode_key(rec1_sorted[i], &e, &gate, &t, &k);

    const uint32_t r = (uint32_t)i - seg1[e];
    if (r < kept1[e]) return;

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
    rec2[idx] = mgf_naive_make_key(b, gate2, t, k);
    atomicAdd(&cnt2[b], 1u);
}

__global__ void mgf_naive_finalize_kernel(
    int E,
    int cap,
    const uint32_t* __restrict__ cnt2,
    const uint32_t* __restrict__ kept1,
    uint32_t* __restrict__ kept2,
    uint32_t* __restrict__ seg2,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t run2 = 0;
    int32_t off = 0;
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
    }
}

__global__ void mgf_naive_pack1_kernel(
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
    mgf_naive_decode_key(rec1_sorted[i], &e, &gate, &t, &k);

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

__global__ void mgf_naive_pack2_kernel(
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
    mgf_naive_decode_key(rec2_sorted[i], &e, &gate, &t, &k);

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

// Expert id per packed position (linear scan over offsets).
__device__ __forceinline__ int mgf_naive_expert_of_pos(
    const int32_t* offsets, int E, int pos) {
    for (int e = 0; e < E; ++e) {
        if (pos < offsets[e + 1]) return e;
    }
    return E - 1;
}

// 4a. Scalar FFN layer 1: one thread per (pos, j).
__global__ void mgf_naive_ffn1_kernel(
    int D,
    int H,
    int E,
    int qshift,
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ w1,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ packed_token,
    uint8_t* __restrict__ hbuf) {
    const long long idx =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int total = offsets[E];
    if (idx >= (long long)total * H) return;
    const int pos = (int)(idx / H);
    const int j = (int)(idx - (long long)pos * H);
    const int e = mgf_naive_expert_of_pos(offsets, E, pos);
    const int t = packed_token[pos];

    int32_t acc = 0;
    const int8_t* w1r = w1 + ((size_t)e * H + j) * D;
    const int8_t* xr = x + (size_t)t * D;
    for (int d = 0; d < D; ++d) {
        acc += (int32_t)w1r[d] * (int32_t)xr[d];
    }
    if (acc < 0) acc = 0;
    acc >>= qshift;
    hbuf[(size_t)pos * H + j] = (uint8_t)(acc < 127 ? acc : 127);
}

// 4b. Scalar FFN layer 2 + combine: one thread per (pos, d).
__global__ void mgf_naive_ffn2_kernel(
    int D,
    int H,
    int E,
    const int8_t* __restrict__ w2,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ packed_token,
    const int32_t* __restrict__ packed_gate,
    const uint8_t* __restrict__ hbuf,
    int32_t* __restrict__ packed_y,
    int64_t* __restrict__ y) {
    const long long idx =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int total = offsets[E];
    if (idx >= (long long)total * D) return;
    const int pos = (int)(idx / D);
    const int d = (int)(idx - (long long)pos * D);
    const int e = mgf_naive_expert_of_pos(offsets, E, pos);
    const int t = packed_token[pos];

    int32_t acc = 0;
    const int8_t* w2r = w2 + ((size_t)e * D + d) * H;
    const uint8_t* hr = hbuf + (size_t)pos * H;
    for (int j = 0; j < H; ++j) {
        acc += (int32_t)w2r[j] * (int32_t)hr[j];
    }
    packed_y[(size_t)pos * D + d] = acc;
    const unsigned long long add = (unsigned long long)(
        (int64_t)packed_gate[pos] * (int64_t)acc);
    atomicAdd(reinterpret_cast<unsigned long long*>(&y[(size_t)t * D + d]),
              add);
}

// 5. Checksum: parallel digests, serial root fold.
__device__ __forceinline__ uint64_t mgf_naive_fnv_byte(uint64_t h, uint8_t b) {
    return (h ^ (uint64_t)b) * MGF_FNV_PRIME;
}

__global__ void mgf_naive_digest_kernel(
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
        for (int d = 0; d < D; ++d) {
            const uint64_t v =
                (uint64_t)y[(size_t)t * D + d];
            for (int b = 0; b < 8; ++b) {
                h = mgf_naive_fnv_byte(h, (uint8_t)(v >> (8 * b)));
            }
        }
    }
    digests[c] = h;
}

__global__ void mgf_naive_root_kernel(
    int C,
    const uint64_t* __restrict__ digests,
    uint64_t* __restrict__ y_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t h = MGF_FNV_BASIS;
    for (int c = 0; c < C; ++c) {
        const uint64_t v = digests[c];
        for (int b = 0; b < 8; ++b) {
            h = mgf_naive_fnv_byte(h, (uint8_t)(v >> (8 * b)));
        }
    }
    y_checksum[0] = h;
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

extern "C" size_t solution_workspace_bytes(const MgfProblemSpec* spec) {
    if (!mgf_validate_problem_spec(spec)) return 0;
    MgfNaiveLayout L = mgf_naive_make_layout(nullptr, spec);
    return L.required_bytes;
}

extern "C" cudaError_t solution_init(
    const MgfProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;
    if (!mgf_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }
    MgfNaiveState* st =
        static_cast<MgfNaiveState*>(malloc(sizeof(MgfNaiveState)));
    if (!st) return cudaErrorMemoryAllocation;
    memcpy(&st->spec, spec, sizeof(MgfProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

static void mgf_naive_bitonic_sort(
    uint64_t* a,
    int n_pow2,
    cudaStream_t stream) {
    const int block = 256;
    const int grid = (n_pow2 + block - 1) / block;
    for (int k = 2; k <= n_pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            mgf_naive_bitonic_kernel<<<grid, block, 0, stream>>>(
                a, n_pow2, k, j);
        }
    }
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

    MgfNaiveState* st = static_cast<MgfNaiveState*>(state);
    const MgfInputs* in = static_cast<const MgfInputs*>(inputs_void);
    MgfOutputs* out = static_cast<MgfOutputs*>(outputs_void);

    MgfNaiveLayout L = mgf_naive_make_layout(workspace, &st->spec);
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
    const int Mp2 = (int)mgf_naive_pow2((size_t)M);

    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(L.cnt1, 0, sizeof(uint32_t) * E, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.cnt2, 0, sizeof(uint32_t) * E, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.m2, 0, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;
    // Pad both record buffers with UINT64_MAX so bitonic pushes pads to the end.
    err = cudaMemsetAsync(L.rec1, 0xFF, sizeof(uint64_t) * Mp2, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.rec2, 0xFF, sizeof(uint64_t) * Mp2, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(out->y, 0, sizeof(int64_t) * (size_t)N * D, stream);
    if (err != cudaSuccess) return err;

    const long long ne = (long long)N * E;
    mgf_naive_logits_kernel<<<(unsigned)((ne + 255) / 256), 256, 0, stream>>>(
        N, D, E, in->x, in->wr, out->logits);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mgf_naive_route_kernel<<<mgf_ceil_div_int(N, 256), 256, 0, stream>>>(
        N, E, G, g_sel, K, out->logits, L.cand, L.rec1, L.cnt1);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mgf_naive_bitonic_sort(L.rec1, Mp2, stream);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mgf_naive_seg1_kernel<<<1, 1, 0, stream>>>(E, cap, L.cnt1, L.kept1, L.seg1);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mgf_naive_classify_kernel<<<mgf_ceil_div_int(M, 256), 256, 0, stream>>>(
        M, E, K, n_sel, L.rec1, L.kept1, L.seg1, L.cand, out->logits,
        L.rec2, L.cnt2, L.m2, out->route_expert, out->route_status);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mgf_naive_bitonic_sort(L.rec2, Mp2, stream);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mgf_naive_finalize_kernel<<<1, 1, 0, stream>>>(
        E, cap, L.cnt2, L.kept1, L.kept2, L.seg2, out->counts, out->offsets);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mgf_naive_pack1_kernel<<<mgf_ceil_div_int(M, 256), 256, 0, stream>>>(
        M, L.rec1, L.kept1, L.seg1, out->offsets, out->packed_token,
        out->packed_slot, out->packed_gate, out->packed_phase,
        out->route_expert, out->route_status, K);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mgf_naive_pack2_kernel<<<mgf_ceil_div_int(M, 256), 256, 0, stream>>>(
        L.rec2, L.m2, L.kept1, L.kept2, L.seg2, out->offsets,
        out->packed_token, out->packed_slot, out->packed_gate,
        out->packed_phase, out->route_expert, out->route_status, K);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    // FFN over an upper bound of positions (guarded by offsets[E] on device).
    {
        const long long max_total = (long long)E * cap;
        const long long n1 = max_total * H;
        mgf_naive_ffn1_kernel<<<(unsigned)((n1 + 255) / 256), 256, 0, stream>>>(
            D, H, E, run->qshift, in->x, in->w1, out->offsets,
            out->packed_token, L.hbuf);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        const long long n2 = max_total * D;
        mgf_naive_ffn2_kernel<<<(unsigned)((n2 + 255) / 256), 256, 0, stream>>>(
            D, H, E, in->w2, out->offsets, out->packed_token,
            out->packed_gate, L.hbuf, out->packed_y, out->y);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    {
        const int C = mgf_ceil_div_int(N, MGF_CSUM_ROWS);
        mgf_naive_digest_kernel<<<mgf_ceil_div_int(C, 128), 128, 0, stream>>>(
            N, D, out->y, L.digests);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        mgf_naive_root_kernel<<<1, 1, 0, stream>>>(
            C, L.digests, out->y_checksum);
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
