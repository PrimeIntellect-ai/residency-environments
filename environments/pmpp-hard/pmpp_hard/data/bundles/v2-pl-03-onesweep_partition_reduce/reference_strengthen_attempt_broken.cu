// PMPP_CANARY_03_742613fc69 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: onesweep_partition_reduce_reference.cu
// Strong reference pipeline implementation.
//
// Two fused kernels (the workload is launch-latency dominated at N <= 128K):
//   K_A: per-tile histograms (one block per tile, shared-memory counters,
//        block 0 also zeroes the reduce accumulators for K_B). The LAST
//        block to finish (threadfence + global ticket) computes ALL the
//        metadata in-kernel: per-digit totals, a block-wide exclusive scan
//        for digit_offsets, and the carried per-tile look-back offsets
//        (coalesced column walks: thread d handles digit d). The ticket
//        resets itself for the next call.
//   K_B: stable scatter (warp-level __match_any ranks + per-warp digit
//        histograms) FUSED with the bucket reduction: per-tile partial sums
//        and (max, smallest-src) results are combined in shared memory and
//        pushed with one global atomic per present digit; the encoding
//        ((v ^ 0x80000000) << 32) | (INT32_MAX - src) makes a single 64-bit
//        atomicMax implement "largest value, then smallest source index"
//        exactly. The last block decodes the winners, writes bucket_sum /
//        bucket_max / bucket_argmax (empty buckets: 0 / INT64_MIN / -1),
//        and resets its ticket.
// The tickets live in persistent state allocated in solution_init, so no
// per-call memset launches are needed; every kernel leaves them zeroed.
// All combining operators are commutative integer ops (add / max on
// uint64), so outputs are bit-identical to the sequential specification.
// ============================================================================

#include "onesweep_partition_reduce_common.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define OPR_FULL_MASK 0xffffffffu

struct OprReferenceState {
    OprProblemSpec spec;
    unsigned* d_tickets;   // [2], persistent, self-cleaning
};

struct OprWorkspaceLayout {
    int32_t* tile_counts;             // max_tiles * max_digits
    unsigned long long* gsum;         // OPR_MAX_DIGITS
    unsigned long long* genc;         // OPR_MAX_DIGITS
    size_t required_bytes;
};

static size_t opr_reference_workspace_bytes_for(int max_N, int max_radix_bits) {
    const int max_tiles = opr_num_tiles(max_N);
    const int max_digits = opr_num_digits(max_radix_bits);

    size_t off = 0;
    off = opr_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)max_tiles * (size_t)max_digits;
    off = opr_align_up_size(off, 128);
    off += sizeof(unsigned long long) * (size_t)OPR_MAX_DIGITS;
    off = opr_align_up_size(off, 128);
    off += sizeof(unsigned long long) * (size_t)OPR_MAX_DIGITS;
    off = opr_align_up_size(off, 128);
    return off;
}

static OprWorkspaceLayout opr_reference_make_layout(
    void* workspace,
    int max_N,
    int max_radix_bits) {
    OprWorkspaceLayout layout{};
    char* base = static_cast<char*>(workspace);
    const int max_tiles = opr_num_tiles(max_N);
    const int max_digits = opr_num_digits(max_radix_bits);

    size_t off = 0;
    off = opr_align_up_size(off, 128);
    layout.tile_counts = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)max_tiles * (size_t)max_digits;
    off = opr_align_up_size(off, 128);
    layout.gsum = reinterpret_cast<unsigned long long*>(base + off);
    off += sizeof(unsigned long long) * (size_t)OPR_MAX_DIGITS;
    off = opr_align_up_size(off, 128);
    layout.genc = reinterpret_cast<unsigned long long*>(base + off);
    off += sizeof(unsigned long long) * (size_t)OPR_MAX_DIGITS;
    off = opr_align_up_size(off, 128);
    layout.required_bytes = off;
    return layout;
}

__device__ __forceinline__ int opr_digit_device(uint32_t key, int radix_bits) {
    return static_cast<int>(key >> (32 - radix_bits));
}

// (max value, then smallest src) as one monotone 64-bit key; > 0 always.
__device__ __forceinline__ unsigned long long opr_enc_device(int v, int src) {
    return ((unsigned long long)((unsigned)v ^ 0x80000000u) << 32) |
           (unsigned long long)(unsigned)(0x7fffffff - src);
}

// ---------------------------------------------------------------------------
// K_A: tile histograms + last-block metadata.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
opr_ref_hist_meta_kernel(
    int N,
    int radix_bits,
    int num_tiles,
    const uint32_t* __restrict__ key,
    int32_t* __restrict__ tile_counts,
    int32_t* __restrict__ tile_digit_offsets,
    int32_t* __restrict__ digit_counts,
    int32_t* __restrict__ digit_offsets,
    unsigned long long* __restrict__ gsum,
    unsigned long long* __restrict__ genc,
    unsigned* tickets) {
    __shared__ int32_t s_hist[OPR_MAX_DIGITS];
    __shared__ int32_t s_warp[8];
    __shared__ int s_last;

    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    const int num_digits = 1 << radix_bits;

    if (tid < num_digits) s_hist[tid] = 0;

    // block 0 zeroes the K_B accumulators (K_B only starts after K_A ends)
    if (tile == 0) {
        gsum[tid] = 0ull;
        genc[tid] = 0ull;
    }
    __syncthreads();

    const int i = tile * OPR_TILE_ITEMS + tid;
    if (i < N) {
        atomicAdd(&s_hist[opr_digit_device(key[i], radix_bits)], 1);
    }
    __syncthreads();

    if (tid < num_digits) {
        tile_counts[(size_t)tile * (size_t)num_digits + (size_t)tid] =
            s_hist[tid];
    }

    __threadfence();
    if (tid == 0) {
        s_last = (atomicAdd(&tickets[0], 1u) == (unsigned)(num_tiles - 1));
    }
    __syncthreads();
    if (!s_last) return;

    // ---- last block: all per-tile counts are visible now ----
    __threadfence();

    // per-digit totals (coalesced column walk)
    int total = 0;
    if (tid < num_digits) {
        for (int t = 0; t < num_tiles; ++t) {
            total += tile_counts[(size_t)t * (size_t)num_digits + (size_t)tid];
        }
        digit_counts[tid] = total;
    }

    // block-wide exclusive scan of totals -> digit_offsets
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int incl = total;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int o = __shfl_up_sync(OPR_FULL_MASK, incl, off);
        if (lane >= off) incl += o;
    }
    if (lane == 31) s_warp[warp] = incl;
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            const int t = s_warp[w];
            s_warp[w] = run;
            run += t;
        }
    }
    __syncthreads();
    const int excl = s_warp[warp] + incl - total;

    if (tid < num_digits) {
        digit_offsets[tid] = excl;
        if (tid == num_digits - 1) digit_offsets[num_digits] = excl + total;
    }

    // carried per-tile look-back offsets (coalesced column walk)
    if (tid < num_digits) {
        int running = excl;
        for (int t = 0; t < num_tiles; ++t) {
            const size_t idx = (size_t)t * (size_t)num_digits + (size_t)tid;
            tile_digit_offsets[idx] = running;
            running += tile_counts[idx];
        }
    }

    if (tid == 0) tickets[0] = 0u;   // self-clean for the next call
}

// ---------------------------------------------------------------------------
// K_B: stable scatter + fused bucket reduce + last-block finalize.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
opr_ref_scatter_reduce_kernel(
    int N,
    int radix_bits,
    int num_tiles,
    const uint32_t* __restrict__ key,
    const int32_t* __restrict__ val,
    const int32_t* __restrict__ tile_digit_offsets,
    const int32_t* __restrict__ digit_counts,
    uint32_t* __restrict__ packed_key,
    int32_t* __restrict__ packed_val,
    int32_t* __restrict__ packed_src,
    unsigned long long* __restrict__ gsum,
    unsigned long long* __restrict__ genc,
    int64_t* __restrict__ bucket_sum,
    int64_t* __restrict__ bucket_max,
    int32_t* __restrict__ bucket_argmax,
    unsigned* tickets) {
    __shared__ int32_t s_warp_hist[8 * OPR_MAX_DIGITS];
    __shared__ unsigned long long s_sum[OPR_MAX_DIGITS];
    __shared__ unsigned long long s_enc[OPR_MAX_DIGITS];
    __shared__ int s_last;

    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    const int num_digits = 1 << radix_bits;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int j = tid; j < 8 * num_digits; j += 256) s_warp_hist[j] = 0;
    if (tid < num_digits) {
        s_sum[tid] = 0ull;
        s_enc[tid] = 0ull;
    }
    __syncthreads();

    const int i = tile * OPR_TILE_ITEMS + tid;
    const bool active = (i < N);
    uint32_t k = 0;
    int32_t v = 0;
    int dig = 0;

    if (active) {
        k = key[i];
        v = val[i];
        dig = opr_digit_device(k, radix_bits);
        atomicAdd(&s_warp_hist[warp * num_digits + dig], 1);
    }
    __syncthreads();

    if (active) {
        int rank = 0;
        for (int w = 0; w < warp; ++w) {
            rank += s_warp_hist[w * num_digits + dig];
        }
        const unsigned amask = __ballot_sync(OPR_FULL_MASK, active);
        const unsigned mm = __match_any_sync(amask, dig);
        rank += __popc(mm & ((lane == 0) ? 0u : ((1u << lane) - 1u)));

        const int pos =
            tile_digit_offsets[(size_t)tile * (size_t)num_digits + (size_t)dig] +
            rank;
        packed_key[pos] = k;
        packed_val[pos] = v;
        packed_src[pos] = i;

        atomicAdd(&s_sum[dig], (unsigned long long)(long long)v);
        atomicMax(&s_enc[dig], opr_enc_device(v, i));
    }
    __syncthreads();

    if (tid < num_digits && s_enc[tid] != 0ull) {
        atomicAdd(&gsum[tid], s_sum[tid]);
        atomicMax(&genc[tid], s_enc[tid]);
    }

    __threadfence();
    if (tid == 0) {
        s_last = (atomicAdd(&tickets[1], 1u) == (unsigned)(num_tiles - 1));
    }
    __syncthreads();
    if (!s_last) return;

    // ---- last block: finalize the bucket outputs ----
    __threadfence();

    if (tid < num_digits) {
        bucket_sum[tid] = (int64_t)gsum[tid];   // empty bucket -> 0
        if (digit_counts[tid] == 0) {
            bucket_max[tid] = (int64_t)(-9223372036854775807LL - 1LL);
            bucket_argmax[tid] = -1;
        } else {
            const unsigned long long e = genc[tid];
            const int mv = (int)((unsigned)(e >> 32) ^ 0x80000000u);
            const int src = 0x7fffffff - (int)(unsigned)(e & 0xffffffffull);
            bucket_max[tid] = (int64_t)mv;
            bucket_argmax[tid] = src;
        }
    }

    if (tid == 0) tickets[1] = 0u;   // self-clean for the next call
}

extern "C" size_t solution_workspace_bytes(const OprProblemSpec* spec) {
    if (!opr_validate_problem_spec(spec)) return 0;
    return opr_reference_workspace_bytes_for(spec->max_N, spec->max_radix_bits);
}

extern "C" cudaError_t solution_init(
    const OprProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!opr_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    OprReferenceState* st = static_cast<OprReferenceState*>(malloc(sizeof(OprReferenceState)));
    if (!st) {
        return cudaErrorMemoryAllocation;
    }

    memcpy(&st->spec, spec, sizeof(OprProblemSpec));
    st->d_tickets = nullptr;

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&st->d_tickets),
                                 2 * sizeof(unsigned));
    if (err != cudaSuccess) {
        free(st);
        return err;
    }
    err = cudaMemsetAsync(st->d_tickets, 0, 2 * sizeof(unsigned), stream);
    if (err != cudaSuccess) {
        cudaFree(st->d_tickets);
        free(st);
        return err;
    }

    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const OprRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !opr_validate_run_spec(run) || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    OprReferenceState* st = static_cast<OprReferenceState*>(state);
    const OprInputs* in = static_cast<const OprInputs*>(inputs_void);
    OprOutputs* out = static_cast<OprOutputs*>(outputs_void);

    if (run->N > st->spec.max_N ||
        run->radix_bits > st->spec.max_radix_bits) {
        return cudaErrorInvalidValue;
    }

    if (!in->key || !in->val ||
        !out->tile_digit_offsets || !out->digit_counts ||
        !out->digit_offsets || !out->packed_key ||
        !out->packed_val || !out->packed_src ||
        !out->bucket_sum || !out->bucket_max || !out->bucket_argmax) {
        return cudaErrorInvalidValue;
    }

    OprWorkspaceLayout layout = opr_reference_make_layout(
        workspace,
        st->spec.max_N,
        st->spec.max_radix_bits);

    if (workspace_bytes < layout.required_bytes) {
        return cudaErrorInvalidValue;
    }

    const int N = run->N;
    const int radix_bits = run->radix_bits;
    const int num_tiles = opr_num_tiles(N);

    opr_ref_hist_meta_kernel<<<num_tiles, 256, 0, stream>>>(
        N,
        radix_bits,
        num_tiles,
        in->key,
        layout.tile_counts,
        out->tile_digit_offsets,
        out->digit_counts,
        out->digit_offsets,
        layout.gsum,
        layout.genc,
        st->d_tickets);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    opr_ref_scatter_reduce_kernel<<<num_tiles, 256, 0, stream>>>(
        N,
        radix_bits,
        num_tiles,
        in->key,
        in->val,
        out->tile_digit_offsets,
        out->digit_counts,
        out->packed_key,
        out->packed_val,
        out->packed_src,
        layout.gsum,
        layout.genc,
        out->bucket_sum,
        out->bucket_max,
        out->bucket_argmax,
        st->d_tickets);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    OprReferenceState* st = static_cast<OprReferenceState*>(state);
    if (st && st->d_tickets) {
        return cudaMemsetAsync(st->d_tickets, 0, 2 * sizeof(unsigned), stream);
    }
    return cudaSuccess;
}

extern "C" void solution_destroy(void* state) {
    OprReferenceState* st = static_cast<OprReferenceState*>(state);
    if (st) {
        if (st->d_tickets) cudaFree(st->d_tickets);
        free(st);
    }
}
