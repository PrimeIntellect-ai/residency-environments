// PMPP_CANARY_92_e4c19a55d0 -- held-out canary; MUST NOT appear in any submission
// file: sat_segscan_select_reference.cu
//
// Single-pass decoupled-lookback implementation. The key identity: the
// per-element update x -> clamp((flag ? 0 : x) + v, LO, HI) belongs to the
// family F = { x -> min(max(x + a, lo), hi) } of monotone unit-slope clamp
// maps, and F is CLOSED under composition:
//   (f then g) = (f.a + g.a,
//                 clamp(f.lo + g.a, g.lo, g.hi),
//                 clamp(f.hi + g.a, g.lo, g.hi))
// Function composition is associative, so the non-associative saturating
// scan becomes an ordinary scan over (a, lo, hi) triples (a in int64; a head
// element is the constant map (0, c, c), which absorbs everything before
// it). Identity is represented as (0, -2^28, +2^28): all values that ever
// flow through a map lie in [-2^24, 2^24] u {0}, so the widened rails are
// exact on the reachable domain.
//
// K1 (one kernel, one read of v): 4096-element tiles, 256 threads x 16.
//   Tickets give tiles their rank in launch order. Each thread composes its
//   16 element maps, a shfl warp scan + shared block scan produce thread-
//   exclusive triples and the tile aggregate, the tile publishes AGG,
//   performs a warp-parallel WINDOWED lookback (spin on predecessor states,
//   payload protected by __threadfence + volatile state words, AGG/PREFIX
//   in separate slots so nothing is ever overwritten), publishes PREFIX,
//   then every thread REPLAYS its 16 elements sequentially from its exact
//   start value: y stores (vectorized), sat bit mask, ordered seg_last
//   writes (segment ordinals from the flag-count scan carried through the
//   same lookback payloads), and the per-tile saturation count.
// K2: single-block exclusive scan of per-tile sat counts -> sel offsets.
// K3: per-tile ordered compaction of sel_idx straight from sat_bits.

#include "sat_segscan_select_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SSS_TILE 4096
#define SSS_THREADS 256
#define SSS_PER_THREAD 16
#define SSS_FULL 0xffffffffu

#define SSS_ID_LO (-(1 << 28))
#define SSS_ID_HI (1 << 28)

struct SssReferenceState {
    SssProblemSpec spec;
};

struct SssFn {
    long long a;
    int lo;
    int hi;
};

struct alignas(16) SssPay {
    long long a;
    int lo;
    int hi;
    unsigned fcnt;
};

__device__ __forceinline__ SssFn sss_identity() {
    SssFn f;
    f.a = 0;
    f.lo = SSS_ID_LO;
    f.hi = SSS_ID_HI;
    return f;
}

__device__ __forceinline__ int sss_clamp_ll(long long t, int lo, int hi) {
    if (t < (long long)lo) return lo;
    if (t > (long long)hi) return hi;
    return (int)t;
}

// f then g (f applied first).
__device__ __forceinline__ SssFn sss_compose(const SssFn& f, const SssFn& g) {
    SssFn r;
    r.a = f.a + g.a;
    r.lo = sss_clamp_ll((long long)f.lo + g.a, g.lo, g.hi);
    r.hi = sss_clamp_ll((long long)f.hi + g.a, g.lo, g.hi);
    return r;
}

__device__ __forceinline__ int sss_eval(const SssFn& f, int x) {
    return sss_clamp_ll((long long)x + f.a, f.lo, f.hi);
}

__device__ __forceinline__ SssFn sss_shfl_up(const SssFn& f, int delta) {
    SssFn r;
    r.a = __shfl_up_sync(SSS_FULL, f.a, delta);
    r.lo = __shfl_up_sync(SSS_FULL, f.lo, delta);
    r.hi = __shfl_up_sync(SSS_FULL, f.hi, delta);
    return r;
}

__device__ __forceinline__ SssFn sss_shfl_idx(const SssFn& f, int src) {
    SssFn r;
    r.a = __shfl_sync(SSS_FULL, f.a, src);
    r.lo = __shfl_sync(SSS_FULL, f.lo, src);
    r.hi = __shfl_sync(SSS_FULL, f.hi, src);
    return r;
}

// ---------------------------------------------------------------------------
// K1: the fused lookback scan.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(SSS_THREADS)
sss_ref_scan_kernel(
    int N,
    int Nw,
    int T,
    int LO,
    int HI,
    const int32_t* __restrict__ v,
    const uint32_t* __restrict__ flags,
    unsigned* ws_ticket,
    volatile unsigned* ws_state,
    SssPay* ws_agg,
    SssPay* ws_pre,
    unsigned* ws_satcnt,
    int32_t* __restrict__ y,
    uint32_t* __restrict__ sat_bits,
    int32_t* __restrict__ seg_last) {
    __shared__ unsigned sh_tile;
    __shared__ uint32_t sh_fw[SSS_TILE / 32];      // 128 flag words
    __shared__ SssFn sh_wagg[8];
    __shared__ unsigned sh_wfc[8];
    __shared__ SssFn sh_wexcl[8];
    __shared__ unsigned sh_wfcexcl[8];
    __shared__ SssFn sh_inc;
    __shared__ unsigned sh_incfc;
    __shared__ uint32_t sh_m[SSS_THREADS];         // 16-bit sat masks
    __shared__ unsigned sh_satred[8];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (tid == 0) sh_tile = atomicAdd(ws_ticket, 1u);
    __syncthreads();

    const int tile = (int)sh_tile;
    const int base = tile * SSS_TILE;
    const int wbase = base >> 5;

    // ---- stage flag words ----
    if (tid < SSS_TILE / 32) {
        const int w = wbase + tid;
        sh_fw[tid] = (w < Nw) ? flags[w] : 0u;
    }
    __syncthreads();

    // ---- load my 16 values ----
    const int e0 = base + SSS_PER_THREAD * tid;
    int myv[SSS_PER_THREAD];
    if (e0 + SSS_PER_THREAD <= N) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int4 t4 = *reinterpret_cast<const int4*>(v + e0 + 4 * q);
            myv[4 * q + 0] = t4.x;
            myv[4 * q + 1] = t4.y;
            myv[4 * q + 2] = t4.z;
            myv[4 * q + 3] = t4.w;
        }
    } else {
        #pragma unroll
        for (int e = 0; e < SSS_PER_THREAD; ++e) {
            myv[e] = (e0 + e < N) ? v[e0 + e] : 0;
        }
    }

    // ---- compose my 16 element maps ----
    SssFn f = sss_identity();
    unsigned fcnt = 0;
    #pragma unroll
    for (int e = 0; e < SSS_PER_THREAD; ++e) {
        const int i = e0 + e;
        if (i < N) {
            const int li = i - base;
            const int fl = (int)((sh_fw[li >> 5] >> (li & 31)) & 1u);
            SssFn ef;
            if (fl) {
                const int c = sss_clamp_ll((long long)myv[e], LO, HI);
                ef.a = 0;
                ef.lo = c;
                ef.hi = c;
                ++fcnt;
            } else {
                ef.a = myv[e];
                ef.lo = LO;
                ef.hi = HI;
            }
            f = sss_compose(f, ef);
        }
    }

    // ---- warp inclusive scan ----
    SssFn incf = f;
    unsigned incc = fcnt;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const SssFn of = sss_shfl_up(incf, off);
        const unsigned oc = __shfl_up_sync(SSS_FULL, incc, off);
        if (lane >= off) {
            incf = sss_compose(of, incf);
            incc += oc;
        }
    }
    if (lane == 31) {
        sh_wagg[warp] = incf;
        sh_wfc[warp] = incc;
    }
    __syncthreads();

    // ---- warp-aggregate exclusive prefixes (serial over 8, exact order) ----
    if (tid == 0) {
        SssFn run = sss_identity();
        unsigned rc = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            sh_wexcl[w] = run;
            sh_wfcexcl[w] = rc;
            run = sss_compose(run, sh_wagg[w]);
            rc += sh_wfc[w];
        }
        sh_wagg[7] = run;   // tile aggregate parked here
        sh_wfc[7] = rc;
    }
    __syncthreads();

    const SssFn tile_agg = sh_wagg[7];
    const unsigned tile_fc = sh_wfc[7];

    // thread-exclusive triple within the tile
    SssFn lane_excl = sss_shfl_up(incf, 1);
    unsigned lane_exclc = __shfl_up_sync(SSS_FULL, incc, 1);
    if (lane == 0) {
        lane_excl = sss_identity();
        lane_exclc = 0;
    }
    const SssFn thr_excl = sss_compose(sh_wexcl[warp], lane_excl);
    const unsigned thr_exclc = sh_wfcexcl[warp] + lane_exclc;

    // ---- publish AGG ----
    if (tid == 0 && tile > 0) {
        SssPay p;
        p.a = tile_agg.a;
        p.lo = tile_agg.lo;
        p.hi = tile_agg.hi;
        p.fcnt = tile_fc;
        ws_agg[tile] = p;
        __threadfence();
        ws_state[tile] = 1u;
    }

    // ---- windowed lookback (warp 0) ----
    if (warp == 0) {
        SssFn inc = sss_identity();
        unsigned infc = 0;

        if (tile > 0) {
            int end = tile;
            for (;;) {
                const int idx = end - 1 - lane;
                const bool active = (idx >= 0);
                unsigned st = 2u;
                if (active) {
                    st = ws_state[idx];
                    while (st == 0u) {
                        __nanosleep(40);
                        st = ws_state[idx];
                    }
                }
                __threadfence();
                const unsigned pmask =
                    __ballot_sync(SSS_FULL, active && st == 2u);
                const int j = pmask ? (__ffs((int)pmask) - 1) : -1;

                SssFn pf = sss_identity();
                unsigned pfc = 0;
                const int lim = (j >= 0) ? j : 31;
                if (active && lane <= lim) {
                    const SssPay p = (lane == j) ? ws_pre[idx] : ws_agg[idx];
                    pf.a = p.a;
                    pf.lo = p.lo;
                    pf.hi = p.hi;
                    pfc = p.fcnt;
                }

                // fold lanes lim..0 (ascending tile order) on lane 0
                SssFn w = sss_shfl_idx(pf, lim);
                unsigned wc = __shfl_sync(SSS_FULL, pfc, lim);
                for (int l = lim - 1; l >= 0; --l) {
                    const SssFn nf = sss_shfl_idx(pf, l);
                    const unsigned nc = __shfl_sync(SSS_FULL, pfc, l);
                    if (lane == 0) {
                        w = sss_compose(w, nf);
                        wc += nc;
                    }
                }
                if (lane == 0) {
                    inc = sss_compose(w, inc);
                    infc = wc + infc;
                }
                if (j >= 0) break;
                end -= 32;
            }
        }

        if (lane == 0) {
            sh_inc = inc;
            sh_incfc = infc;

            // publish PREFIX
            SssPay p;
            const SssFn pre = sss_compose(inc, tile_agg);
            p.a = pre.a;
            p.lo = pre.lo;
            p.hi = pre.hi;
            p.fcnt = infc + tile_fc;
            ws_pre[tile] = p;
            __threadfence();
            ws_state[tile] = 2u;
        }
    }
    __syncthreads();

    const SssFn inc = sh_inc;
    const unsigned F0 = sh_incfc;
    const int p0 = sss_eval(inc, 0);

    // ---- replay: exact sequential semantics from the exact start value ----
    int p = sss_eval(thr_excl, p0);
    int ord = (int)(F0 + thr_exclc) - 1;
    uint32_t mask16 = 0;
    int out[SSS_PER_THREAD];

    const int nvalid = (e0 < N)
        ? ((e0 + SSS_PER_THREAD <= N) ? SSS_PER_THREAD : (N - e0)) : 0;

    #pragma unroll
    for (int e = 0; e < SSS_PER_THREAD; ++e) {
        if (e < nvalid) {
            const int i = e0 + e;
            const int li = i - base;
            const int fl = (int)((sh_fw[li >> 5] >> (li & 31)) & 1u);
            if (fl) {
                p = sss_clamp_ll((long long)myv[e], LO, HI);
                ++ord;
            } else {
                p = sss_clamp_ll((long long)p + myv[e], LO, HI);
            }
            out[e] = p;
            if (p == LO || p == HI) mask16 |= (1u << e);

            // segment end?
            bool endseg;
            if (i == N - 1) {
                endseg = true;
            } else if (li + 1 < SSS_TILE) {
                endseg = ((sh_fw[(li + 1) >> 5] >> ((li + 1) & 31)) & 1u) != 0u;
            } else {
                endseg = ((flags[(i + 1) >> 5] >> ((i + 1) & 31)) & 1u) != 0u;
            }
            if (endseg) seg_last[ord] = p;
        }
    }

    // ---- y stores ----
    if (nvalid == SSS_PER_THREAD) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            int4 t4;
            t4.x = out[4 * q + 0];
            t4.y = out[4 * q + 1];
            t4.z = out[4 * q + 2];
            t4.w = out[4 * q + 3];
            *reinterpret_cast<int4*>(y + e0 + 4 * q) = t4;
        }
    } else {
        #pragma unroll
        for (int e = 0; e < SSS_PER_THREAD; ++e) {
            if (e < nvalid) y[e0 + e] = out[e];
        }
    }

    // ---- sat_bits assembly + tile count ----
    sh_m[tid] = mask16;
    __syncthreads();

    unsigned satacc = 0;
    if (tid < SSS_TILE / 32) {
        const int w = wbase + tid;
        if (w < Nw) {
            const uint32_t word = sh_m[2 * tid] | (sh_m[2 * tid + 1] << 16);
            sat_bits[w] = word;
            satacc = (unsigned)__popc(word);
        }
    }
    // block reduce satacc (order-free integer sum)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        satacc += __shfl_down_sync(SSS_FULL, satacc, off);
    }
    if (lane == 0) sh_satred[warp] = satacc;
    __syncthreads();
    if (tid == 0) {
        unsigned t = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) t += sh_satred[w];
        ws_satcnt[tile] = t;
    }
}

// ---------------------------------------------------------------------------
// K2: single-block exclusive scan of per-tile sat counts.
// ---------------------------------------------------------------------------
__global__ void sss_ref_offsets_kernel(
    int T,
    const unsigned* __restrict__ ws_satcnt,
    unsigned* __restrict__ ws_seloff,
    int32_t* __restrict__ sel_count) {
    __shared__ unsigned sh[1024];
    __shared__ unsigned sh_carry;

    const int tid = threadIdx.x;
    if (tid == 0) sh_carry = 0;
    __syncthreads();

    for (int b0 = 0; b0 < T; b0 += 1024) {
        const int t = b0 + tid;
        unsigned val = (t < T) ? ws_satcnt[t] : 0u;
        sh[tid] = val;
        __syncthreads();

        // Hillis-Steele inclusive scan in shared
        for (int off = 1; off < 1024; off <<= 1) {
            unsigned add = 0;
            if (tid >= off) add = sh[tid - off];
            __syncthreads();
            sh[tid] += add;
            __syncthreads();
        }

        if (t < T) ws_seloff[t] = sh_carry + sh[tid] - val;  // exclusive
        __syncthreads();
        if (tid == 0) sh_carry += sh[1023];
        __syncthreads();
    }

    if (tid == 0) sel_count[0] = (int32_t)sh_carry;
}

// ---------------------------------------------------------------------------
// K3: per-tile ordered compaction from sat_bits.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128)
sss_ref_select_kernel(
    int N,
    int Nw,
    const uint32_t* __restrict__ sat_bits,
    const unsigned* __restrict__ ws_seloff,
    int32_t* __restrict__ sel_idx) {
    __shared__ unsigned sh_c[128];

    const int tile = blockIdx.x;
    const int tid = threadIdx.x;
    const int w = tile * 128 + tid;

    const uint32_t word = (w < Nw) ? sat_bits[w] : 0u;
    unsigned cnt = (unsigned)__popc(word);

    // block exclusive scan of 128 word counts
    unsigned inc = cnt;
    const int lane = tid & 31;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const unsigned o = __shfl_up_sync(SSS_FULL, inc, off);
        if (lane >= off) inc += o;
    }
    if (lane == 31) sh_c[tid >> 5] = inc;
    __syncthreads();
    if (tid == 0) {
        unsigned run = 0;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const unsigned t = sh_c[q];
            sh_c[q] = run;
            run += t;
        }
    }
    __syncthreads();
    unsigned off0 = ws_seloff[tile] + sh_c[tid >> 5] + inc - cnt;

    uint32_t rest = word;
    while (rest) {
        const int b = __ffs((int)rest) - 1;
        rest &= rest - 1;
        sel_idx[off0++] = 32 * w + b;
    }
    (void)N;
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

static size_t sss_align256(size_t x) { return (x + 255u) & ~(size_t)255u; }

static void sss_ws_layout(
    int max_N,
    size_t* off_state, size_t* off_ticket,
    size_t* off_agg, size_t* off_pre,
    size_t* off_satcnt, size_t* off_seloff,
    size_t* total) {
    const int T = sss_ceil_div_int(max_N, SSS_TILE);
    size_t off = 0;
    *off_state = off;  off += sss_align256((size_t)T * 4);
    *off_ticket = off; off += sss_align256(4);
    *off_agg = off;    off += sss_align256((size_t)T * sizeof(SssPay));
    *off_pre = off;    off += sss_align256((size_t)T * sizeof(SssPay));
    *off_satcnt = off; off += sss_align256((size_t)T * 4);
    *off_seloff = off; off += sss_align256((size_t)T * 4);
    *total = off;
}

extern "C" size_t solution_workspace_bytes(const SssProblemSpec* spec) {
    if (!sss_validate_problem_spec(spec)) return 0;
    size_t a, b, c, d, e, f, total;
    sss_ws_layout(spec->max_N, &a, &b, &c, &d, &e, &f, &total);
    return total;
}

extern "C" cudaError_t solution_init(
    const SssProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;
    if (!sss_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }
    SssReferenceState* st =
        static_cast<SssReferenceState*>(malloc(sizeof(SssReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;
    memcpy(&st->spec, spec, sizeof(SssProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const SssRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !sss_validate_run_spec(run) || !inputs_void || !outputs_void ||
        !workspace) {
        return cudaErrorInvalidValue;
    }

    SssReferenceState* st = static_cast<SssReferenceState*>(state);
    const SssInputs* in = static_cast<const SssInputs*>(inputs_void);
    SssOutputs* out = static_cast<SssOutputs*>(outputs_void);

    if (run->N > st->spec.max_N) return cudaErrorInvalidValue;
    if (!in->v || !in->flags ||
        !out->y || !out->sat_bits || !out->sel_idx || !out->sel_count ||
        !out->seg_last) {
        return cudaErrorInvalidValue;
    }

    size_t off_state, off_ticket, off_agg, off_pre, off_satcnt, off_seloff, need;
    sss_ws_layout(st->spec.max_N,
                  &off_state, &off_ticket, &off_agg, &off_pre,
                  &off_satcnt, &off_seloff, &need);
    if (workspace_bytes < need) return cudaErrorInvalidValue;

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    unsigned* ws_state = reinterpret_cast<unsigned*>(ws + off_state);
    unsigned* ws_ticket = reinterpret_cast<unsigned*>(ws + off_ticket);
    SssPay* ws_agg = reinterpret_cast<SssPay*>(ws + off_agg);
    SssPay* ws_pre = reinterpret_cast<SssPay*>(ws + off_pre);
    unsigned* ws_satcnt = reinterpret_cast<unsigned*>(ws + off_satcnt);
    unsigned* ws_seloff = reinterpret_cast<unsigned*>(ws + off_seloff);

    const int N = run->N;
    const int Nw = sss_Nw(N);
    const int T = sss_ceil_div_int(N, SSS_TILE);

    cudaError_t err = cudaMemsetAsync(ws_state, 0, (size_t)T * 4, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(ws_ticket, 0, 4, stream);
    if (err != cudaSuccess) return err;

    sss_ref_scan_kernel<<<T, SSS_THREADS, 0, stream>>>(
        N, Nw, T, run->lo, run->hi,
        in->v, in->flags,
        ws_ticket, ws_state, ws_agg, ws_pre, ws_satcnt,
        out->y, out->sat_bits, out->seg_last);

    sss_ref_offsets_kernel<<<1, 1024, 0, stream>>>(
        T, ws_satcnt, ws_seloff, out->sel_count);

    sss_ref_select_kernel<<<T, 128, 0, stream>>>(
        N, Nw, out->sat_bits, ws_seloff, out->sel_idx);

    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    (void)state;
    (void)stream;
    return cudaSuccess;
}

extern "C" void solution_destroy(void* state) {
    free(state);
}
