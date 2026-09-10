// PMPP_CANARY_24_2c749af13b -- held-out canary; MUST NOT appear in any submission
// file: bucketed_radix_select_reduce_reference.cu

#include "bucketed_radix_select_reduce_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct BrsrReferenceState {
    BrsrProblemSpec spec;
};

// ---------------------------------------------------------------------------
// Metadata passes (byte-identical to the deterministic oracle): already
// parallel histogram + trivial serial bucket-choose. Unchanged.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int brsr_active_for_pass_device(
    uint32_t key,
    int pass,
    uint32_t prefix) {
    if (pass == 0) return 1;
    const uint32_t key_prefix = key >> (32 - 8 * pass);
    return key_prefix == prefix;
}

__device__ __forceinline__ int brsr_digit_for_pass_device(uint32_t key, int pass) {
    const int shift = 24 - 8 * pass;
    return static_cast<int>((key >> shift) & 0xffu);
}

__global__ void brsr_ref_hist_kernel(
    int N,
    int pass,
    const uint32_t* __restrict__ key,
    const uint32_t* __restrict__ prefix_after_pass,
    int32_t* __restrict__ pass_histograms) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint32_t prefix = 0;
    if (pass > 0) {
        prefix = prefix_after_pass[pass - 1];
    }

    const uint32_t k = key[idx];

    if (!brsr_active_for_pass_device(k, pass, prefix)) return;

    const int digit = brsr_digit_for_pass_device(k, pass);
    atomicAdd(&pass_histograms[pass * BRSR_BUCKETS + digit], 1);
}

__global__ void brsr_ref_choose_kernel(
    int count,
    int pass,
    const int32_t* __restrict__ pass_histograms,
    int32_t* __restrict__ chosen_bucket,
    int32_t* __restrict__ carried_rank,
    uint32_t* __restrict__ prefix_after_pass,
    uint32_t* __restrict__ threshold_key) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int rank = (pass == 0) ? count : carried_rank[pass - 1];
    uint32_t prev_prefix = (pass == 0) ? 0u : prefix_after_pass[pass - 1];

    int chosen = 0;

    for (int b = BRSR_BUCKETS - 1; b >= 0; --b) {
        const int h = pass_histograms[pass * BRSR_BUCKETS + b];

        if (rank > h) {
            rank -= h;
        } else {
            chosen = b;
            break;
        }
    }

    chosen_bucket[pass] = chosen;
    carried_rank[pass] = rank;

    const uint32_t prefix = (prev_prefix << 8) | static_cast<uint32_t>(chosen);
    prefix_after_pass[pass] = prefix;

    if (pass == BRSR_NUM_PASSES - 1) {
        threshold_key[0] = prefix;
    }
}

// ---------------------------------------------------------------------------
// Parallel select + reduce (replaces the old <<<1,1>>> scan / insertion-sort /
// fold). Byte-identical because the selected set ordered by (key desc, index
// asc) is a strict total order (unique indices), so any correct sort yields the
// exact same permutation; and sum/max/argmax are order-independent.
//
// Selection predicate (matches oracle stable top-count):
//   key[i] > threshold                          -> always selected
//   key[i] == threshold && eqrank[i] < need     -> selected
// where eqrank[i] = #{j < i : key[j] == threshold} and need = carried_rank[3].
// ---------------------------------------------------------------------------

// Per-block count of items whose key == threshold (in valid range).
__global__ void brsr_count_equal_per_block_kernel(
    int N,
    const uint32_t* __restrict__ key,
    const uint32_t* __restrict__ threshold_key,
    int32_t* __restrict__ block_equal_count) {
    __shared__ int cnt;
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thr = threshold_key[0];
    if (i < N && key[i] == thr) atomicAdd(&cnt, 1);
    __syncthreads();

    if (threadIdx.x == 0) block_equal_count[blockIdx.x] = cnt;
}

// Serial exclusive scan of the (small) per-block equal counts. num_blocks is
// ceil(N/256) <= 4096, so this is O(num_blocks), not O(N).
__global__ void brsr_excl_scan_kernel(
    int num_blocks,
    const int32_t* __restrict__ in,
    int32_t* __restrict__ out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int acc = 0;
    for (int i = 0; i < num_blocks; ++i) {
        out[i] = acc;
        acc += in[i];
    }
}

// Gather selected original indices (order irrelevant: sorted afterwards).
// Block size must be 256 (matches the shared scan buffer below).
__global__ void brsr_gather_selected_kernel(
    int N,
    const uint32_t* __restrict__ key,
    const uint32_t* __restrict__ threshold_key,
    const int32_t* __restrict__ carried_rank,
    const int32_t* __restrict__ block_offset,
    int32_t* __restrict__ sel_idx,
    int32_t* __restrict__ sel_counter) {
    __shared__ int tmp[256];

    const int t = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + t;
    const uint32_t thr = threshold_key[0];
    const int need = carried_rank[BRSR_NUM_PASSES - 1];

    const bool valid = i < N;
    const uint32_t k = valid ? key[i] : 0u;
    const int is_equal = (valid && k == thr) ? 1 : 0;

    // Inclusive scan of the equal mask within the block (Hillis-Steele).
    tmp[t] = is_equal;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        const int v = (t >= off) ? tmp[t - off] : 0;
        __syncthreads();
        tmp[t] += v;
        __syncthreads();
    }
    const int excl_rank = tmp[t] - is_equal;  // # equal before i within block

    bool sel = false;
    if (valid) {
        if (k > thr) {
            sel = true;
        } else if (k == thr) {
            const int eqrank = block_offset[blockIdx.x] + excl_rank;
            if (eqrank < need) sel = true;
        }
    }

    if (sel) {
        const int pos = atomicAdd(sel_counter, 1);
        sel_idx[pos] = i;
    }
}

// Single-block sort (bitonic) of the count selected items by
// (key desc, index asc), then write outputs + block reductions.
// Composite ascending sort key: ((uint64)(~key) << 32) | (uint32)index.
__global__ void brsr_sort_reduce_kernel(
    int count,
    int m_pow2,
    const uint32_t* __restrict__ key,
    const int32_t* __restrict__ value,
    const int32_t* __restrict__ sel_idx,
    int32_t* __restrict__ out_count,
    int32_t* __restrict__ topT_indices,
    int64_t* __restrict__ topT_sum,
    int32_t* __restrict__ topT_max,
    int32_t* __restrict__ topT_argmax) {
    extern __shared__ uint64_t s[];  // m_pow2 composite keys

    const int t = threadIdx.x;
    const int nt = blockDim.x;

    for (int i = t; i < m_pow2; i += nt) {
        if (i < count) {
            const int idx = sel_idx[i];
            const uint32_t k = key[idx];
            s[i] = (static_cast<uint64_t>(~k) << 32) |
                   static_cast<uint64_t>(static_cast<uint32_t>(idx));
        } else {
            s[i] = 0xffffffffffffffffULL;  // pad -> sorts to the end
        }
    }
    __syncthreads();

    for (int k2 = 2; k2 <= m_pow2; k2 <<= 1) {
        for (int j = k2 >> 1; j > 0; j >>= 1) {
            for (int i = t; i < m_pow2; i += nt) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const uint64_t a = s[i];
                    const uint64_t b = s[ixj];
                    const bool ascending = ((i & k2) == 0);
                    if (ascending == (a > b)) {
                        s[i] = b;
                        s[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write sorted indices.
    for (int i = t; i < count; i += nt) {
        topT_indices[i] = static_cast<int32_t>(
            static_cast<uint32_t>(s[i] & 0xffffffffULL));
    }

    // Block reductions over the selected set (order-independent).
    __shared__ int64_t r_sum[512];
    __shared__ int32_t r_max[512];
    __shared__ int32_t r_arg[512];

    int64_t local_sum = 0;
    int32_t local_max = INT_MIN;
    int32_t local_arg = -1;

    for (int i = t; i < count; i += nt) {
        const int idx = static_cast<int32_t>(
            static_cast<uint32_t>(s[i] & 0xffffffffULL));
        const int32_t v = value[idx];
        local_sum += static_cast<int64_t>(v);
        if (v > local_max || (v == local_max && (local_arg < 0 || idx < local_arg))) {
            local_max = v;
            local_arg = idx;
        }
    }

    r_sum[t] = local_sum;
    r_max[t] = local_max;
    r_arg[t] = local_arg;
    __syncthreads();

    for (int stride = nt >> 1; stride > 0; stride >>= 1) {
        if (t < stride) {
            r_sum[t] += r_sum[t + stride];
            const int32_t ov = r_max[t + stride];
            const int32_t oa = r_arg[t + stride];
            const int32_t cv = r_max[t];
            const int32_t ca = r_arg[t];
            if (ov > cv || (ov == cv && oa >= 0 && (ca < 0 || oa < ca))) {
                r_max[t] = ov;
                r_arg[t] = oa;
            }
        }
        __syncthreads();
    }

    if (t == 0) {
        out_count[0] = count;
        topT_sum[0] = r_sum[0];
        topT_max[0] = r_max[0];
        topT_argmax[0] = r_arg[0];
    }
}

// ---------------------------------------------------------------------------
// Workspace layout (all int32): [sel_counter][block_equal_count][block_offset]
// [sel_idx]. Sized off max_N / max_T at init.
// ---------------------------------------------------------------------------

static inline void brsr_layout(
    int max_N,
    int max_T,
    size_t* off_sel_counter,
    size_t* off_bec,
    size_t* off_bo,
    size_t* off_idx,
    size_t* total) {
    const int num_blocks = brsr_ceil_div_int(max_N, 256);
    size_t o = 0;
    *off_sel_counter = o;
    o += brsr_align_up_size(sizeof(int32_t) * 1, 256);
    *off_bec = o;
    o += brsr_align_up_size(sizeof(int32_t) * (size_t)num_blocks, 256);
    *off_bo = o;
    o += brsr_align_up_size(sizeof(int32_t) * (size_t)num_blocks, 256);
    *off_idx = o;
    o += brsr_align_up_size(sizeof(int32_t) * (size_t)max_T, 256);
    *total = (o < 128) ? 128 : o;
}

extern "C" size_t solution_workspace_bytes(const BrsrProblemSpec* spec) {
    if (!brsr_validate_problem_spec(spec)) return 0;
    size_t a, b, c, d, total;
    brsr_layout(spec->max_N, spec->max_T, &a, &b, &c, &d, &total);
    return total;
}

extern "C" cudaError_t solution_init(
    const BrsrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!brsr_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    BrsrReferenceState* st =
        static_cast<BrsrReferenceState*>(malloc(sizeof(BrsrReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memcpy(&st->spec, spec, sizeof(BrsrProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const BrsrRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !brsr_validate_run_spec(run) ||
        !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    BrsrReferenceState* st = static_cast<BrsrReferenceState*>(state);
    const BrsrInputs* in = static_cast<const BrsrInputs*>(inputs_void);
    BrsrOutputs* out = static_cast<BrsrOutputs*>(outputs_void);

    if (run->N > st->spec.max_N || run->T > st->spec.max_T) {
        return cudaErrorInvalidValue;
    }

    size_t off_sel_counter, off_bec, off_bo, off_idx, total;
    brsr_layout(st->spec.max_N, st->spec.max_T,
                &off_sel_counter, &off_bec, &off_bo, &off_idx, &total);
    if (workspace_bytes < total) return cudaErrorInvalidValue;

    if (!in->key || !in->value ||
        !out->threshold_key || !out->count || !out->topT_indices ||
        !out->topT_sum || !out->topT_max || !out->topT_argmax ||
        !out->pass_histograms || !out->chosen_bucket ||
        !out->carried_rank || !out->prefix_after_pass) {
        return cudaErrorInvalidValue;
    }

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    int32_t* sel_counter = reinterpret_cast<int32_t*>(ws + off_sel_counter);
    int32_t* block_equal_count = reinterpret_cast<int32_t*>(ws + off_bec);
    int32_t* block_offset = reinterpret_cast<int32_t*>(ws + off_bo);
    int32_t* sel_idx = reinterpret_cast<int32_t*>(ws + off_idx);

    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(
        out->pass_histograms, 0,
        sizeof(int32_t) * BRSR_NUM_PASSES * BRSR_BUCKETS, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(sel_counter, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) return err;

    const int count = brsr_count_host(run->N, run->T);
    const int block = 256;
    const int grid = brsr_ceil_div_int(run->N, block);

    for (int pass = 0; pass < BRSR_NUM_PASSES; ++pass) {
        brsr_ref_hist_kernel<<<grid, block, 0, stream>>>(
            run->N, pass, in->key, out->prefix_after_pass, out->pass_histograms);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        brsr_ref_choose_kernel<<<1, 1, 0, stream>>>(
            count, pass, out->pass_histograms, out->chosen_bucket,
            out->carried_rank, out->prefix_after_pass, out->threshold_key);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    // Parallel select.
    brsr_count_equal_per_block_kernel<<<grid, block, 0, stream>>>(
        run->N, in->key, out->threshold_key, block_equal_count);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    brsr_excl_scan_kernel<<<1, 1, 0, stream>>>(
        grid, block_equal_count, block_offset);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    brsr_gather_selected_kernel<<<grid, block, 0, stream>>>(
        run->N, in->key, out->threshold_key, out->carried_rank,
        block_offset, sel_idx, sel_counter);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    // Single-block sort + reduce.
    int m_pow2 = 1;
    while (m_pow2 < count) m_pow2 <<= 1;
    const int sort_block = 512;
    const size_t shmem = sizeof(uint64_t) * (size_t)m_pow2;

    brsr_sort_reduce_kernel<<<1, sort_block, shmem, stream>>>(
        count, m_pow2, in->key, in->value, sel_idx,
        out->count, out->topT_indices,
        out->topT_sum, out->topT_max, out->topT_argmax);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
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
