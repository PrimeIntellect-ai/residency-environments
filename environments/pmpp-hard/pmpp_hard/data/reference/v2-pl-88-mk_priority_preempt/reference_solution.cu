// PMPP_CANARY_88_d1aba4bfa2 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: mk_priority_preempt_reference.cu
// Strong reference implementation.
//
// Architecture:
//   - Persistent job table (class / rem / arrival / stored weight rows)
//     indexed directly by gjid (< MKP_MAX_JOBS_TOTAL per reset epoch).
//   - Per round: ONE single-block scheduler kernel (folds the previous
//     round's execution digests, parallel min-class, ordered compaction
//     via a warp-shuffle two-level scan, n_s / rem / trace-word updates,
//     trace-base bookkeeping) + ONE grid-strided work kernel (256-thread
//     blocks, weight row staged in shared, whole-word salt XOR,
//     warp-per-row lane-strided dp4a with shuffle reduction so X loads
//     are fully coalesced, hierarchical slice digests, commutative u64
//     y atomics).
//   - The full R-round pipeline (~100 small kernels) is captured into a
//     CUDA graph cached on (RunSpec, pointers); the gid base lives in
//     device memory so replay is byte-exact. Falls back to direct
//     launches when capture is unavailable.
//   - Final state: parallel per-job digests, ordered queue dumps via the
//     same block-scan compaction, short class/root folds.
// ============================================================================

#include "mk_priority_preempt_common.h"

#include <stdlib.h>
#include <string.h>

#define MKP_WORK_GRID 1024

struct MkpReferenceState {
    MkpProblemSpec spec;
    uint32_t* j_class;   // [MKP_MAX_JOBS_TOTAL]
    uint32_t* j_rem;     // [MKP_MAX_JOBS_TOTAL]
    uint32_t* j_arr;     // [MKP_MAX_JOBS_TOTAL]
    int8_t* j_w;         // [MKP_MAX_JOBS_TOTAL * max_K]
    uint32_t* dev_ctr;   // [1]: job base
    cudaGraphExec_t graph_exec;
    int graph_valid;
    MkpRunSpec cached_run;
    MkpInputs cached_in;
    MkpOutputs cached_out;
    void* cached_ws;
    size_t cached_wsb;
};

struct MkpRefWorkspace {
    uint32_t* execlist;   // [MKP_MAX_JOBS_TOTAL]
    uint32_t* exec_ns;    // [MKP_MAX_JOBS_TOTAL]
    uint32_t* nexec;      // [1]
    int32_t* active_c;    // [1]
    uint32_t* trace_base; // [1]
    uint32_t* base_this;  // [1]
    uint64_t* sdig;       // [MKP_MAX_JOBS_TOTAL * 8]
    uint64_t* jdig;       // [MKP_MAX_JOBS_TOTAL]
    size_t required_bytes;
};

static MkpRefWorkspace mkp_ref_make_layout(void* workspace) {
    MkpRefWorkspace L{};
    char* base = static_cast<char*>(workspace);
    size_t off = 0;
#define MKP_WS_FIELD(field, type, count)                                     \
    off = mkp_align_up_size(off, 128);                                       \
    L.field = reinterpret_cast<type*>(base + off);                           \
    off += sizeof(type) * (count);
    MKP_WS_FIELD(execlist, uint32_t, MKP_MAX_JOBS_TOTAL)
    MKP_WS_FIELD(exec_ns, uint32_t, MKP_MAX_JOBS_TOTAL)
    MKP_WS_FIELD(nexec, uint32_t, 1)
    MKP_WS_FIELD(active_c, int32_t, 1)
    MKP_WS_FIELD(trace_base, uint32_t, 1)
    MKP_WS_FIELD(base_this, uint32_t, 1)
    MKP_WS_FIELD(sdig, uint64_t, (size_t)MKP_MAX_JOBS_TOTAL * 8)
    MKP_WS_FIELD(jdig, uint64_t, MKP_MAX_JOBS_TOTAL)
#undef MKP_WS_FIELD
    off = mkp_align_up_size(off, 128);
    L.required_bytes = off;
    return L;
}

__device__ __forceinline__ uint64_t mkp_fnv_byte_dev(uint64_t h, uint8_t b) {
    return (h ^ (uint64_t)b) * MKP_FNV_PRIME;
}

__device__ __forceinline__ uint64_t mkp_fnv_u32_dev(uint64_t h, uint32_t v) {
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        h = mkp_fnv_byte_dev(h, (uint8_t)(v >> (8 * m)));
    }
    return h;
}

__device__ __forceinline__ uint64_t mkp_fnv_u64_dev(uint64_t h, uint64_t v) {
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        h = mkp_fnv_byte_dev(h, (uint8_t)(v >> (8 * m)));
    }
    return h;
}

// ---------------------------------------------------------------------------
// Ingestion.
// ---------------------------------------------------------------------------

__global__ void mkp_ref_ingest_kernel(
    int K4,
    int maxK4,
    const uint32_t* __restrict__ dev_ctr,
    const int32_t* __restrict__ jarrival,
    const int32_t* __restrict__ jclass,
    const int32_t* __restrict__ jlen,
    const int8_t* __restrict__ jw,
    uint32_t* __restrict__ j_class,
    uint32_t* __restrict__ j_rem,
    uint32_t* __restrict__ j_arr,
    int8_t* __restrict__ j_w) {
    const int j = blockIdx.x;
    const uint32_t slot = dev_ctr[0] + (uint32_t)j;
    if (threadIdx.x == 0) {
        j_class[slot] = (uint32_t)jclass[j];
        j_rem[slot] = (uint32_t)jlen[j];
        j_arr[slot] = (uint32_t)jarrival[j];
    }
    const uint32_t* src = reinterpret_cast<const uint32_t*>(jw) +
                          (size_t)j * K4;
    uint32_t* dst = reinterpret_cast<uint32_t*>(j_w) + (size_t)slot * maxK4;
    for (int k = threadIdx.x; k < K4; k += 64) {
        dst[k] = __ldg(&src[k]);
    }
}

// ---------------------------------------------------------------------------
// Per-round scheduler (single 1024-thread block).
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(1024)
mkp_ref_sched_kernel(
    int r,
    int quantum,
    int njobs,
    const uint32_t* __restrict__ dev_ctr,
    const uint32_t* __restrict__ j_class,
    uint32_t* __restrict__ j_rem,
    const uint32_t* __restrict__ j_arr,
    uint32_t* __restrict__ execlist,
    uint32_t* __restrict__ exec_ns,
    uint32_t* __restrict__ nexec,
    int32_t* __restrict__ active_c,
    uint32_t* __restrict__ trace_base,
    uint32_t* __restrict__ base_this,
    const uint64_t* __restrict__ sdig,
    uint64_t* __restrict__ trace,
    uint64_t* __restrict__ exec_digest) {
    __shared__ uint32_t wsum[32];
    __shared__ int s_c;

    const int lane = (int)(threadIdx.x & 31u);
    const int wid = (int)(threadIdx.x >> 5);

    // Fold the PREVIOUS round's execution digests (r > 0) while the old
    // nexec / base_this values are still in place.
    if (r > 0) {
        const int pne = (int)nexec[0];
        const uint32_t pbase = base_this[0];
        for (int e = (int)threadIdx.x; e < pne; e += 1024) {
            const int ns = (int)exec_ns[e];
            uint64_t h = MKP_FNV_BASIS;
            for (int s = 0; s < ns; ++s) {
                h = mkp_fnv_u64_dev(h, sdig[(size_t)e * 8 + s]);
            }
            exec_digest[pbase + (uint32_t)e] = h;
        }
    }
    if (threadIdx.x == 0) s_c = 0x7FFFFFFF;
    __syncthreads();

    const int total = (int)dev_ctr[0] + njobs;
    const int chunk = (total + 1023) / 1024;
    const int lo = (int)threadIdx.x * chunk;
    const int hi = lo + chunk < total ? lo + chunk : total;

    int local_min = 0x7FFFFFFF;
    for (int i = lo; i < hi; ++i) {
        if (j_rem[i] > 0u && j_arr[i] <= (uint32_t)r) {
            const int c = (int)j_class[i];
            if (c < local_min) local_min = c;
        }
    }
    if (local_min != 0x7FFFFFFF) atomicMin(&s_c, local_min);
    __syncthreads();
    const int c = s_c;

    uint32_t cnt = 0;
    if (c != 0x7FFFFFFF) {
        for (int i = lo; i < hi; ++i) {
            if (j_rem[i] > 0u && j_arr[i] <= (uint32_t)r &&
                (int)j_class[i] == c) {
                ++cnt;
            }
        }
    }
    // Two-level exclusive scan (warp shuffles + one 32-entry pass).
    uint32_t v = cnt;
    for (int off = 1; off < 32; off <<= 1) {
        const uint32_t n = __shfl_up_sync(0xffffffffu, v, off);
        if (lane >= off) v += n;
    }
    if (lane == 31) wsum[wid] = v;
    __syncthreads();
    if (wid == 0) {
        uint32_t w = wsum[lane];
        for (int off = 1; off < 32; off <<= 1) {
            const uint32_t n = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w += n;
        }
        wsum[lane] = w;
    }
    __syncthreads();
    const uint32_t tb = trace_base[0];
    uint32_t pos = v - cnt + (wid > 0 ? wsum[wid - 1] : 0u);
    if (c != 0x7FFFFFFF) {
        for (int i = lo; i < hi; ++i) {
            const uint32_t rem = j_rem[i];
            if (rem > 0u && j_arr[i] <= (uint32_t)r &&
                (int)j_class[i] == c) {
                const uint32_t ns =
                    (uint32_t)quantum < rem ? (uint32_t)quantum : rem;
                execlist[pos] = (uint32_t)i;
                exec_ns[pos] = ns;
                trace[tb + pos] = mkp_trace_word(
                    (uint32_t)i, r, c, (int)ns, rem - ns);
                j_rem[i] = rem - ns;
                ++pos;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t ne = c == 0x7FFFFFFF ? 0u : wsum[31];
        nexec[0] = ne;
        active_c[0] = c == 0x7FFFFFFF ? -1 : c;
        base_this[0] = tb;
        trace_base[0] = tb + ne;
    }
}

// ---------------------------------------------------------------------------
// Per-round work kernel (grid-strided over execution x slice units).
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256)
mkp_ref_work_kernel(
    int r,
    int quantum,
    int B,
    int K4,
    int maxK4,
    const uint32_t* __restrict__ execlist,
    const uint32_t* __restrict__ exec_ns,
    const uint32_t* __restrict__ nexec,
    const uint32_t* __restrict__ base_this,
    const int8_t* __restrict__ j_w,
    const int8_t* __restrict__ X,
    uint64_t* __restrict__ y,
    uint64_t* __restrict__ sdig) {
    __shared__ uint32_t wsh[64];
    __shared__ uint32_t res_sh[256];
    __shared__ uint64_t sub_sh[16];

    const int units = (int)*nexec * quantum;
    const uint32_t gbase = *base_this;
    const uint32_t* Xu = reinterpret_cast<const uint32_t*>(X);

    for (int u = blockIdx.x; u < units; u += gridDim.x) {
        const int e = u / quantum;
        const int s = u - e * quantum;
        const int ns = (int)exec_ns[e];
        if (s < ns) {
            const int job = (int)execlist[e];
            const uint32_t g = gbase + (uint32_t)e;
            const uint32_t salt = MKP_SALT_ROUND * (uint32_t)r +
                                  MKP_SALT_TRACE * g +
                                  MKP_SALT_JOB * (uint32_t)job +
                                  MKP_SALT_SLICE * (uint32_t)s;
            const int b = (int)((g + (uint32_t)s) % (uint32_t)B);

            if (threadIdx.x < (uint32_t)K4) {
                wsh[threadIdx.x] = reinterpret_cast<const uint32_t*>(
                    j_w)[(size_t)job * maxK4 + threadIdx.x];
            }
            __syncthreads();

            // Warp-per-row: lanes stride along k so X loads are fully
            // coalesced (thread-per-row would waste ~8x of every sector).
            const int warp = (int)(threadIdx.x >> 5);
            const int lane = (int)(threadIdx.x & 31u);
            for (int i0 = 0; i0 < 32; ++i0) {
                const int i = warp * 32 + i0;
                const int row = b * MKP_ROWS + i;
                const uint32_t* xrow = Xu + (size_t)row * K4;
                int32_t acc = 0;
                for (int k = lane; k < K4; k += 32) {
                    acc = __dp4a((int)(__ldg(&xrow[k]) ^ salt),
                                 (int)wsh[k], acc);
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    acc += __shfl_xor_sync(0xffffffffu, acc, off);
                }
                if (lane == 0) {
                    res_sh[i] = (uint32_t)acc;
                    atomicAdd(
                        reinterpret_cast<unsigned long long*>(&y[row]),
                        (unsigned long long)(uint32_t)acc);
                }
            }
            __syncthreads();

            if ((threadIdx.x & 15u) == 0u) {
                const int t = (int)threadIdx.x >> 4;
                uint64_t sub = MKP_FNV_BASIS;
                for (int i = 16 * t; i < 16 * t + 16; ++i) {
                    sub = mkp_fnv_u32_dev(sub, res_sh[i]);
                }
                sub_sh[t] = sub;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                uint64_t sd = MKP_FNV_BASIS;
#pragma unroll
                for (int t = 0; t < 16; ++t) {
                    sd = mkp_fnv_u64_dev(sd, sub_sh[t]);
                }
                sdig[(size_t)e * 8 + s] = sd;
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Fold the LAST round's execution digests (all earlier rounds are folded
// by the next round's scheduler kernel).
// ---------------------------------------------------------------------------

__global__ void mkp_ref_lastfold_kernel(
    const uint32_t* __restrict__ nexec,
    const uint32_t* __restrict__ base_this,
    const uint32_t* __restrict__ exec_ns,
    const uint64_t* __restrict__ sdig,
    uint64_t* __restrict__ exec_digest) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (int)*nexec) return;
    const int ns = (int)exec_ns[e];
    uint64_t h = MKP_FNV_BASIS;
    for (int s = 0; s < ns; ++s) {
        h = mkp_fnv_u64_dev(h, sdig[(size_t)e * 8 + s]);
    }
    exec_digest[*base_this + (uint32_t)e] = h;
}

// ---------------------------------------------------------------------------
// Finalization.
// ---------------------------------------------------------------------------

__global__ void mkp_ref_carryover_kernel(
    int njobs,
    const uint32_t* __restrict__ dev_ctr,
    uint32_t* __restrict__ j_arr,
    const uint32_t* __restrict__ trace_base,
    int32_t* __restrict__ trace_len) {
    const int total = (int)dev_ctr[0] + njobs;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) j_arr[i] = 0;
    if (i == 0) trace_len[0] = (int32_t)trace_base[0];
}

__global__ void mkp_ref_bump_kernel(
    int njobs,
    uint32_t* __restrict__ dev_ctr) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        dev_ctr[0] += (uint32_t)njobs;
    }
}

__global__ void mkp_ref_jdig_kernel(
    int K,
    int maxK,
    const uint32_t* __restrict__ dev_ctr,
    const uint32_t* __restrict__ j_class,
    const uint32_t* __restrict__ j_rem,
    const int8_t* __restrict__ j_w,
    uint64_t* __restrict__ jdig) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)dev_ctr[0]) return;
    if (j_rem[i] == 0u) return;
    uint64_t h = MKP_FNV_BASIS;
    h = mkp_fnv_u32_dev(h, (uint32_t)i);
    h = mkp_fnv_u32_dev(h, j_class[i]);
    h = mkp_fnv_u32_dev(h, j_rem[i]);
    const int8_t* w = j_w + (size_t)i * maxK;
    for (int k = 0; k < K; ++k) {
        h = mkp_fnv_byte_dev(h, (uint8_t)w[k]);
    }
    jdig[i] = h;
}

__global__ void __launch_bounds__(1024)
mkp_ref_dump_kernel(
    int Q,
    const uint32_t* __restrict__ dev_ctr,
    const uint32_t* __restrict__ j_class,
    const uint32_t* __restrict__ j_rem,
    int32_t* __restrict__ queue_len,
    uint64_t* __restrict__ queue_dump) {
    __shared__ uint32_t part[1024];
    __shared__ uint32_t s_off;
    const int total = (int)dev_ctr[0];
    const int chunk = (total + 1023) / 1024;
    const int lo = threadIdx.x * chunk;
    const int hi = lo + chunk < total ? lo + chunk : total;
    if (threadIdx.x == 0) s_off = 0;

    for (int c = 0; c < Q; ++c) {
        __syncthreads();
        uint32_t cnt = 0;
        for (int i = lo; i < hi; ++i) {
            if (j_rem[i] > 0u && (int)j_class[i] == c) ++cnt;
        }
        part[threadIdx.x] = cnt;
        __syncthreads();
        uint32_t v = part[threadIdx.x];
        for (int off = 1; off < 1024; off <<= 1) {
            uint32_t add = 0;
            if (threadIdx.x >= (uint32_t)off) add = part[threadIdx.x - off];
            __syncthreads();
            part[threadIdx.x] = v = v + add;
            __syncthreads();
        }
        uint32_t pos = s_off + part[threadIdx.x] - cnt;
        for (int i = lo; i < hi; ++i) {
            if (j_rem[i] > 0u && (int)j_class[i] == c) {
                queue_dump[pos++] = mkp_dump_word((uint32_t)i, c, j_rem[i]);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            queue_len[c] = (int32_t)part[1023];
            s_off += part[1023];
        }
    }
}

__global__ void mkp_ref_csum_kernel(
    int Q,
    int K,
    const uint32_t* __restrict__ dev_ctr,
    const uint32_t* __restrict__ j_class,
    const uint32_t* __restrict__ j_rem,
    const int32_t* __restrict__ queue_len,
    const uint64_t* __restrict__ jdig,
    uint64_t* __restrict__ state_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int total = (int)dev_ctr[0];
    uint64_t root = MKP_FNV_BASIS;
    for (int c = 0; c < Q; ++c) {
        uint64_t cd = MKP_FNV_BASIS;
        cd = mkp_fnv_u32_dev(cd, (uint32_t)queue_len[c]);
        for (int i = 0; i < total; ++i) {
            if (j_rem[i] > 0u && (int)j_class[i] == c) {
                cd = mkp_fnv_u64_dev(cd, jdig[i]);
            }
        }
        root = mkp_fnv_u64_dev(root, cd);
    }
    state_checksum[0] = root;
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

extern "C" size_t solution_workspace_bytes(const MkpProblemSpec* spec) {
    if (!mkp_validate_problem_spec(spec)) return 0;
    return mkp_ref_make_layout(nullptr).required_bytes;
}

extern "C" cudaError_t solution_init(
    const MkpProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!mkp_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }
    MkpReferenceState* st = static_cast<MkpReferenceState*>(
        malloc(sizeof(MkpReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(MkpReferenceState));
    memcpy(&st->spec, spec, sizeof(MkpProblemSpec));

    cudaError_t err = cudaSuccess;
#define MKP_INIT_ALLOC(field, type, count)                                   \
    if (err == cudaSuccess) {                                                \
        err = cudaMalloc(reinterpret_cast<void**>(&st->field),               \
                         sizeof(type) * (count));                            \
    }
    MKP_INIT_ALLOC(j_class, uint32_t, MKP_MAX_JOBS_TOTAL)
    MKP_INIT_ALLOC(j_rem, uint32_t, MKP_MAX_JOBS_TOTAL)
    MKP_INIT_ALLOC(j_arr, uint32_t, MKP_MAX_JOBS_TOTAL)
    MKP_INIT_ALLOC(j_w, int8_t, (size_t)MKP_MAX_JOBS_TOTAL * spec->max_K)
    MKP_INIT_ALLOC(dev_ctr, uint32_t, 1)
#undef MKP_INIT_ALLOC
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
    MkpReferenceState* st = static_cast<MkpReferenceState*>(state);
    cudaError_t err = cudaMemsetAsync(
        st->j_rem, 0, sizeof(uint32_t) * MKP_MAX_JOBS_TOTAL, stream);
    if (err != cudaSuccess) return err;
    return cudaMemsetAsync(st->dev_ctr, 0, sizeof(uint32_t), stream);
}

static cudaError_t mkp_ref_enqueue(
    MkpReferenceState* st,
    const MkpRunSpec* run,
    const MkpInputs* in,
    MkpOutputs* out,
    const MkpRefWorkspace& L,
    cudaStream_t stream) {
    const int K4 = run->K / 4;
    const int maxK4 = st->spec.max_K / 4;
    const int B = run->M / MKP_ROWS;

    cudaError_t err = cudaSuccess;
#define MKP_CHECK_LAUNCH()                                                   \
    err = cudaPeekAtLastError();                                             \
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(out->y, 0, sizeof(uint64_t) * run->M, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(L.trace_base, 0, sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;

    mkp_ref_ingest_kernel<<<run->njobs, 64, 0, stream>>>(
        K4, maxK4, st->dev_ctr, in->jarrival, in->jclass, in->jlen, in->jw,
        st->j_class, st->j_rem, st->j_arr, st->j_w);
    MKP_CHECK_LAUNCH()

    for (int r = 0; r < run->R; ++r) {
        mkp_ref_sched_kernel<<<1, 1024, 0, stream>>>(
            r, run->quantum, run->njobs, st->dev_ctr, st->j_class, st->j_rem,
            st->j_arr, L.execlist, L.exec_ns, L.nexec, L.active_c,
            L.trace_base, L.base_this, L.sdig, out->trace, out->exec_digest);
        MKP_CHECK_LAUNCH()
        mkp_ref_work_kernel<<<MKP_WORK_GRID, 256, 0, stream>>>(
            r, run->quantum, B, K4, maxK4, L.execlist, L.exec_ns, L.nexec,
            L.base_this, st->j_w, in->X, out->y, L.sdig);
        MKP_CHECK_LAUNCH()
    }
    mkp_ref_lastfold_kernel<<<MKP_MAX_JOBS_TOTAL / 256, 256, 0, stream>>>(
        L.nexec, L.base_this, L.exec_ns, L.sdig, out->exec_digest);
    MKP_CHECK_LAUNCH()

    mkp_ref_carryover_kernel<<<MKP_MAX_JOBS_TOTAL / 256, 256, 0, stream>>>(
        run->njobs, st->dev_ctr, st->j_arr, L.trace_base, out->trace_len);
    MKP_CHECK_LAUNCH()
    mkp_ref_bump_kernel<<<1, 1, 0, stream>>>(run->njobs, st->dev_ctr);
    MKP_CHECK_LAUNCH()

    mkp_ref_jdig_kernel<<<MKP_MAX_JOBS_TOTAL / 256, 256, 0, stream>>>(
        run->K, st->spec.max_K, st->dev_ctr, st->j_class, st->j_rem, st->j_w,
        L.jdig);
    MKP_CHECK_LAUNCH()
    mkp_ref_dump_kernel<<<1, 1024, 0, stream>>>(
        run->Q, st->dev_ctr, st->j_class, st->j_rem, out->queue_len,
        out->queue_dump);
    MKP_CHECK_LAUNCH()
    mkp_ref_csum_kernel<<<1, 1, 0, stream>>>(
        run->Q, run->K, st->dev_ctr, st->j_class, st->j_rem, out->queue_len,
        L.jdig, out->state_checksum);
    MKP_CHECK_LAUNCH()

#undef MKP_CHECK_LAUNCH
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const MkpRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !mkp_validate_run_spec(run) || !inputs_void ||
        !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }
    MkpReferenceState* st = static_cast<MkpReferenceState*>(state);
    const MkpInputs* in = static_cast<const MkpInputs*>(inputs_void);
    MkpOutputs* out = static_cast<MkpOutputs*>(outputs_void);

    if (run->M > st->spec.max_M || run->K > st->spec.max_K ||
        run->Q > st->spec.max_Q || run->R > st->spec.max_R ||
        run->njobs > st->spec.max_njobs) {
        return cudaErrorInvalidValue;
    }
    if (!in->X || !in->jarrival || !in->jclass || !in->jlen || !in->jw ||
        !out->trace_len || !out->trace || !out->exec_digest || !out->y ||
        !out->queue_len || !out->queue_dump || !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    MkpRefWorkspace L = mkp_ref_make_layout(workspace);
    if (workspace_bytes < L.required_bytes) return cudaErrorInvalidValue;

    const bool cache_hit =
        st->graph_valid &&
        memcmp(&st->cached_run, run, sizeof(MkpRunSpec)) == 0 &&
        memcmp(&st->cached_in, in, sizeof(MkpInputs)) == 0 &&
        memcmp(&st->cached_out, out, sizeof(MkpOutputs)) == 0 &&
        st->cached_ws == workspace && st->cached_wsb == workspace_bytes;

    if (!cache_hit) {
        if (st->graph_valid) {
            cudaGraphExecDestroy(st->graph_exec);
            st->graph_valid = 0;
        }
        cudaError_t err = cudaStreamBeginCapture(
            stream, cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess) {
            cudaGetLastError();
            return mkp_ref_enqueue(st, run, in, out, L, stream);
        }
        err = mkp_ref_enqueue(st, run, in, out, L, stream);
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
        memcpy(&st->cached_run, run, sizeof(MkpRunSpec));
        memcpy(&st->cached_in, in, sizeof(MkpInputs));
        memcpy(&st->cached_out, out, sizeof(MkpOutputs));
        st->cached_ws = workspace;
        st->cached_wsb = workspace_bytes;
    }

    return cudaGraphLaunch(st->graph_exec, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    MkpReferenceState* st = static_cast<MkpReferenceState*>(state);
    if (st->graph_valid) cudaGraphExecDestroy(st->graph_exec);
    if (st->j_class) cudaFree(st->j_class);
    if (st->j_rem) cudaFree(st->j_rem);
    if (st->j_arr) cudaFree(st->j_arr);
    if (st->j_w) cudaFree(st->j_w);
    if (st->dev_ctr) cudaFree(st->dev_ctr);
    free(st);
}
