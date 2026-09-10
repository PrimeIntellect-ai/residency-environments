// PMPP_CANARY_32_0e747549ed -- held-out canary; MUST NOT appear in any submission
// file: streaming_merge_topk_quantile_reference.cu
//
// Fast byte-exact GPU reference. Output bytes (topk arrays, counts, sums,
// quantiles, FNV checksums, total_ingested) are IDENTICAL to the serial oracle;
// only execution is parallelized:
//   * ingest / top-K insertion  -> one thread per group (the SAME serial
//                                  insertion logic per group, partitioned by
//                                  group -> byte-exact by construction)
//   * quantile / finalize        -> one thread per group (same serial bin scan)
//   * FNV-1a-64 checksums        -> exact parallel FNV via the chunk-affine
//                                  identity  FNV(chunk, h) = A*(h & ~0xFF)
//                                  + T[h & 0xFF],  A = prime^len,
//                                  T[v] = FNV(chunk, v).

#include "streaming_merge_topk_quantile_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// FNV-1a-64 (NON-CANONICAL basis, per contract)
// ---------------------------------------------------------------------------
static constexpr uint64_t SMTQ_FNV_OFFSET = 1469598103934665603ULL;
static constexpr uint64_t SMTQ_FNV_PRIME = 1099511628211ULL;

// Bytes per FNV chunk for the parallel-affine checksum. Tuned: balances the
// serial combine pass (cost ~ num_chunks, latency-bound) against the parallel
// table build (cost ~ 256*len / #threads, threads = num_chunks*256).
static constexpr int SMTQ_FNV_CHUNK = 1024;

struct SmtqReferenceState {
    SmtqProblemSpec spec;

    int32_t* topk_key;
    int32_t* topk_value;
    int64_t* topk_order;
    int32_t* topk_count;
    int32_t* hist;
    int32_t* group_total;
    int64_t* total_ingested;

    // ingest scratch
    int64_t* order_buf;  // [max_batch] per-event insertion order (valid events)
    int cand_w;          // per-group candidate stride = max_batch + K
    int32_t* cand_key;   // [G * cand_w]
    int32_t* cand_value; // [G * cand_w]
    int64_t* cand_order; // [G * cand_w]

    // parallel-FNV scratch
    uint8_t* packed;     // packed byte stream (reused for hist then state)
    uint64_t* fnv_A;     // [num_chunks_cap] A = prime^len per chunk
    uint64_t* fnv_T;     // [num_chunks_cap * 256] chunk table
    size_t packed_cap;   // bytes
    int num_chunks_cap;
};

// ---------------------------------------------------------------------------
// Device helpers (identical semantics to the oracle)
// ---------------------------------------------------------------------------
__device__ __forceinline__ int64_t smtq_add_i64_i32_wrap_device(int64_t a, int32_t b) {
    uint64_t ua = static_cast<uint64_t>(a);
    ua += static_cast<uint64_t>(static_cast<int64_t>(b));
    return static_cast<int64_t>(ua);
}

__device__ __forceinline__ int smtq_better_device(
    int32_t cand_key,
    int64_t cand_order,
    int32_t best_key,
    int64_t best_order) {
    return (cand_key > best_key) ||
           (cand_key == best_key && cand_order < best_order);
}

__device__ __forceinline__ int smtq_bin_width_device(
    int32_t key_min,
    int32_t key_max,
    int num_bins) {
    const int64_t range = static_cast<int64_t>(key_max) - static_cast<int64_t>(key_min) + 1;
    return static_cast<int>((range + num_bins - 1) / num_bins);
}

__device__ int smtq_bin_for_key_device(
    int32_t key,
    int32_t key_min,
    int32_t key_max,
    int num_bins) {
    if (key <= key_min) return 0;
    if (key >= key_max) return num_bins - 1;

    const int width = smtq_bin_width_device(key_min, key_max, num_bins);
    int bin = static_cast<int>((static_cast<int64_t>(key) - key_min) / width);

    if (bin < 0) bin = 0;
    if (bin >= num_bins) bin = num_bins - 1;

    return bin;
}

__device__ int32_t smtq_bin_lower_bound_device(
    int bin,
    int32_t key_min,
    int32_t key_max,
    int num_bins) {
    const int width = smtq_bin_width_device(key_min, key_max, num_bins);
    int64_t lo = static_cast<int64_t>(key_min) + static_cast<int64_t>(bin) * width;

    if (lo < key_min) lo = key_min;
    if (lo > key_max) lo = key_max;

    return static_cast<int32_t>(lo);
}

__device__ void smtq_sort_group_topk_device(
    int K,
    int32_t* __restrict__ topk_key,
    int32_t* __restrict__ topk_value,
    int64_t* __restrict__ topk_order,
    int base) {
    for (int i = 1; i < K; ++i) {
        int32_t k = topk_key[base + i];
        int32_t v = topk_value[base + i];
        int64_t o = topk_order[base + i];

        int j = i - 1;
        while (j >= 0 && smtq_better_device(k, o, topk_key[base + j], topk_order[base + j])) {
            topk_key[base + j + 1] = topk_key[base + j];
            topk_value[base + j + 1] = topk_value[base + j];
            topk_order[base + j + 1] = topk_order[base + j];
            --j;
        }

        topk_key[base + j + 1] = k;
        topk_value[base + j + 1] = v;
        topk_order[base + j + 1] = o;
    }
}

__device__ void smtq_insert_topk_device(
    int K,
    int group,
    int32_t key,
    int32_t value,
    int64_t order,
    int32_t* __restrict__ topk_key,
    int32_t* __restrict__ topk_value,
    int64_t* __restrict__ topk_order,
    int32_t* __restrict__ topk_count) {
    const int base = group * K;
    int count = topk_count[group];

    if (count < K) {
        topk_key[base + count] = key;
        topk_value[base + count] = value;
        topk_order[base + count] = order;
        topk_count[group] = count + 1;
        smtq_sort_group_topk_device(K, topk_key, topk_value, topk_order, base);
        return;
    }

    const int worst = base + K - 1;
    if (smtq_better_device(key, order, topk_key[worst], topk_order[worst])) {
        topk_key[worst] = key;
        topk_value[worst] = value;
        topk_order[worst] = order;
        smtq_sort_group_topk_device(K, topk_key, topk_value, topk_order, base);
    }
}

__device__ int32_t smtq_quantile_device(
    int group,
    int q_num,
    int q_den,
    int num_bins,
    int32_t key_min,
    int32_t key_max,
    const int32_t* __restrict__ hist,
    const int32_t* __restrict__ group_total) {
    const int total = group_total[group];
    if (total <= 0) return INT_MIN;

    if (q_num < 0) q_num = 0;
    if (q_num > q_den) q_num = q_den;

    int64_t rank = (static_cast<int64_t>(q_num) * total + q_den - 1) / q_den;
    if (rank < 1) rank = 1;
    if (rank > total) rank = total;

    int64_t accum = 0;
    for (int b = 0; b < num_bins; ++b) {
        accum += hist[group * num_bins + b];
        if (accum >= rank) {
            return smtq_bin_lower_bound_device(b, key_min, key_max, num_bins);
        }
    }

    return key_max;
}

// ---------------------------------------------------------------------------
// Reset (parallel across groups)
// ---------------------------------------------------------------------------
__global__ void smtq_reference_reset_kernel(
    int G,
    int K,
    int num_bins,
    int32_t* __restrict__ topk_key,
    int32_t* __restrict__ topk_value,
    int64_t* __restrict__ topk_order,
    int32_t* __restrict__ topk_count,
    int32_t* __restrict__ hist,
    int32_t* __restrict__ group_total,
    int64_t* __restrict__ total_ingested) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;

    topk_count[g] = 0;
    group_total[g] = 0;

    for (int i = 0; i < K; ++i) {
        const int idx = g * K + i;
        topk_key[idx] = INT_MIN;
        topk_value[idx] = 0;
        topk_order[idx] = INT64_MAX;
    }

    for (int b = 0; b < num_bins; ++b) {
        hist[g * num_bins + b] = 0;
    }

    if (g == 0) total_ingested[0] = 0;
}

// ---------------------------------------------------------------------------
// Ingest: assign per-event order (serial prefix over the batch, cheap), then
// one thread per group performs the exact serial insertion for its own group.
// ---------------------------------------------------------------------------
__global__ void smtq_assign_order_kernel(
    int G,
    int batch_size,
    const int32_t* __restrict__ in_group,
    int64_t* __restrict__ order_buf,
    int64_t* __restrict__ total_ingested) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int64_t order = total_ingested[0];
    for (int i = 0; i < batch_size; ++i) {
        const int g = in_group[i];
        if (g < 0 || g >= G) {
            order_buf[i] = -1;
            continue;
        }
        order_buf[i] = order;
        order += 1;
    }
    total_ingested[0] = order;
}

// Rank-select ingest: one BLOCK per group. The final top-K for a group is the
// K best of (prior real top-K items) U (this step's valid events for the
// group) under the strict total order (key desc, insertion_order asc). Because
// insertion_order is globally unique (prior orders < new orders, all distinct),
// this ordered set is UNIQUE and identical to what the serial insertion
// produces -- so slot r receives the candidate that has exactly r strictly
// better candidates. Byte-exact; parallel even for hot (single-group) batches.
__global__ void smtq_ingest_ranksel_kernel(
    int G,
    int K,
    int num_bins,
    int32_t key_min,
    int32_t key_max,
    int batch_size,
    int cand_w,
    const int32_t* __restrict__ in_group,
    const int32_t* __restrict__ in_key,
    const int32_t* __restrict__ in_value,
    const int64_t* __restrict__ order_buf,
    int32_t* __restrict__ topk_key,
    int32_t* __restrict__ topk_value,
    int64_t* __restrict__ topk_order,
    int32_t* __restrict__ topk_count,
    int32_t* __restrict__ hist,
    int32_t* __restrict__ group_total,
    int32_t* __restrict__ cand_key,
    int32_t* __restrict__ cand_value,
    int64_t* __restrict__ cand_order) {
    const int g = blockIdx.x;
    if (g >= G) return;

    const int t = threadIdx.x;
    const int nt = blockDim.x;
    const size_t cbase = static_cast<size_t>(g) * cand_w;

    __shared__ int s_count_old;
    __shared__ int s_new;
    if (t == 0) {
        s_count_old = topk_count[g];
        s_new = 0;
    }
    __syncthreads();

    const int count_old = s_count_old;

    // gather prior real items as candidates [0, count_old)
    for (int i = t; i < count_old; i += nt) {
        const int idx = g * K + i;
        cand_key[cbase + i] = topk_key[idx];
        cand_value[cbase + i] = topk_value[idx];
        cand_order[cbase + i] = topk_order[idx];
    }
    __syncthreads();

    // scan batch, append this group's valid events, accumulate histogram
    for (int i = t; i < batch_size; i += nt) {
        if (in_group[i] != g) continue;  // only own, valid group (0<=g<G)

        const int32_t k = in_key[i];
        const int32_t v = in_value[i];
        const int64_t o = order_buf[i];

        const int bin = smtq_bin_for_key_device(k, key_min, key_max, num_bins);
        atomicAdd(&hist[g * num_bins + bin], 1);

        const int pos = atomicAdd(&s_new, 1);
        const size_t ci = cbase + count_old + pos;
        cand_key[ci] = k;
        cand_value[ci] = v;
        cand_order[ci] = o;
    }
    __syncthreads();

    const int n_new = s_new;
    const int C = count_old + n_new;
    const int new_count = C < K ? C : K;

    if (t == 0) {
        group_total[g] += n_new;
        topk_count[g] = new_count;
    }

    // rank each candidate; write those landing in the kept top-K
    for (int a = t; a < C; a += nt) {
        const int32_t ka = cand_key[cbase + a];
        const int64_t oa = cand_order[cbase + a];

        int rank = 0;
        for (int b = 0; b < C; ++b) {
            const int32_t kb = cand_key[cbase + b];
            const int64_t ob = cand_order[cbase + b];
            // b strictly better than a  (b outranks a)
            if (kb > ka || (kb == ka && ob < oa)) ++rank;
        }

        if (rank < new_count) {
            const int idx = g * K + rank;
            topk_key[idx] = ka;
            topk_value[idx] = cand_value[cbase + a];
            topk_order[idx] = oa;
        }
    }
    __syncthreads();

    // fill unused slots with the empty sentinel
    for (int i = new_count + t; i < K; i += nt) {
        const int idx = g * K + i;
        topk_key[idx] = INT_MIN;
        topk_value[idx] = 0;
        topk_order[idx] = INT64_MAX;
    }
}

// ---------------------------------------------------------------------------
// Finalize per-group outputs (topk arrays, count, value_sum, quantile).
// ---------------------------------------------------------------------------
__global__ void smtq_finalize_kernel(
    int G,
    int K,
    int num_bins,
    int32_t key_min,
    int32_t key_max,
    int is_query,
    int q_num,
    int q_den,
    const int32_t* __restrict__ topk_key,
    const int32_t* __restrict__ topk_value,
    const int32_t* __restrict__ topk_count,
    const int32_t* __restrict__ hist,
    const int32_t* __restrict__ group_total,
    int32_t* __restrict__ out_topk_keys,
    int32_t* __restrict__ out_topk_values,
    int32_t* __restrict__ out_topk_count,
    int64_t* __restrict__ out_topk_value_sum,
    int32_t* __restrict__ out_quantile_key) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;

    const int count = topk_count[g];
    out_topk_count[g] = count;

    int64_t sum = 0;
    for (int i = 0; i < count; ++i) {
        sum = smtq_add_i64_i32_wrap_device(sum, topk_value[g * K + i]);
    }
    out_topk_value_sum[g] = sum;

    for (int i = 0; i < K; ++i) {
        const int idx = g * K + i;
        out_topk_keys[idx] = topk_key[idx];
        out_topk_values[idx] = topk_value[idx];
    }

    out_quantile_key[g] = is_query
        ? smtq_quantile_device(g, q_num, q_den, num_bins, key_min, key_max, hist, group_total)
        : INT_MIN;
}

// ---------------------------------------------------------------------------
// Packing kernels: lay out the exact FNV byte streams in a contiguous buffer.
// ---------------------------------------------------------------------------
// Histogram stream: [G(i32)] [num_bins(i32)] [hist group-major (i32...)]
__global__ void smtq_pack_hist_kernel(
    int G,
    int num_bins,
    const int32_t* __restrict__ hist,
    uint8_t* __restrict__ packed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        reinterpret_cast<int32_t*>(packed)[0] = G;
        reinterpret_cast<int32_t*>(packed)[1] = num_bins;
    }
    const int total = G * num_bins;
    int32_t* dst = reinterpret_cast<int32_t*>(packed + 8);
    for (int i = tid; i < total; i += gridDim.x * blockDim.x) {
        dst[i] = hist[i];
    }
}

// State stream:
//   header(28): G,K,num_bins,key_min,key_max (i32 x5), total_ingested (i64)
//   per group : group_total(i32), topk_count(i32),
//               for i in K: key(i32), value(i32), order(i64)   [16*K bytes]
//               for b in num_bins: hist(i32)                    [4*num_bins bytes]
// Every field lands at a 4-byte-aligned offset in the packed stream, so all
// writes are plain int32 stores (int64 fields are split into two 4-aligned
// int32 halves, little-endian). Fully parallel: thread-per-element over the
// three regions (gt/tc, top-K slots, histogram).
__device__ __forceinline__ void smtq_put_i32(uint8_t* base, size_t off, int32_t v) {
    *reinterpret_cast<int32_t*>(base + off) = v;
}
__device__ __forceinline__ void smtq_put_i64(uint8_t* base, size_t off, int64_t v) {
    const uint64_t u = static_cast<uint64_t>(v);
    *reinterpret_cast<int32_t*>(base + off) = static_cast<int32_t>(u & 0xFFFFFFFFu);
    *reinterpret_cast<int32_t*>(base + off + 4) = static_cast<int32_t>(u >> 32);
}

__global__ void smtq_pack_state_kernel(
    int G,
    int K,
    int num_bins,
    int32_t key_min,
    int32_t key_max,
    const int64_t* __restrict__ total_ingested,
    const int32_t* __restrict__ topk_key,
    const int32_t* __restrict__ topk_value,
    const int64_t* __restrict__ topk_order,
    const int32_t* __restrict__ topk_count,
    const int32_t* __restrict__ hist,
    const int32_t* __restrict__ group_total,
    uint8_t* __restrict__ packed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    const size_t seg = static_cast<size_t>(8) +
                       static_cast<size_t>(16) * K +
                       static_cast<size_t>(4) * num_bins;

    // header (28 bytes)
    if (tid == 0) {
        smtq_put_i32(packed, 0, G);
        smtq_put_i32(packed, 4, K);
        smtq_put_i32(packed, 8, num_bins);
        smtq_put_i32(packed, 12, key_min);
        smtq_put_i32(packed, 16, key_max);
        smtq_put_i64(packed, 20, total_ingested[0]);
    }

    // per-group group_total + topk_count
    for (int g = tid; g < G; g += stride) {
        const size_t base = 28 + static_cast<size_t>(g) * seg;
        smtq_put_i32(packed, base, group_total[g]);
        smtq_put_i32(packed, base + 4, topk_count[g]);
    }

    // top-K slots: 16 bytes each (key,value,order)
    const int nslots = G * K;
    for (int s = tid; s < nslots; s += stride) {
        const int g = s / K;
        const int i = s % K;
        const size_t base = 28 + static_cast<size_t>(g) * seg + 8 + static_cast<size_t>(16) * i;
        smtq_put_i32(packed, base, topk_key[s]);
        smtq_put_i32(packed, base + 4, topk_value[s]);
        smtq_put_i64(packed, base + 8, topk_order[s]);
    }

    // histogram: 4 bytes each
    const int ncells = G * num_bins;
    const size_t hist_off = 8 + static_cast<size_t>(16) * K;
    for (int e = tid; e < ncells; e += stride) {
        const int g = e / num_bins;
        const int b = e % num_bins;
        const size_t base = 28 + static_cast<size_t>(g) * seg + hist_off + static_cast<size_t>(4) * b;
        smtq_put_i32(packed, base, hist[e]);
    }
}

// ---------------------------------------------------------------------------
// Parallel exact FNV over a contiguous buffer of `len` bytes.
//   For each chunk c (SMTQ_FNV_CHUNK bytes, last may be shorter):
//     A[c] = prime^len_c
//     T[c*256 + v] = FNV(chunk_c, v)   for v in 0..255
//   Then a single serial pass combines chunks in stream order:
//     h = A[c]*(h & ~0xFF) + T[c*256 + (h & 0xFF)]
// ---------------------------------------------------------------------------
__global__ void smtq_fnv_tables_kernel(
    const uint8_t* __restrict__ buf,
    size_t len,
    int num_chunks,
    uint64_t* __restrict__ fnv_A,
    uint64_t* __restrict__ fnv_T) {
    const long tid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(num_chunks) * 256;
    const long stride = static_cast<long>(gridDim.x) * blockDim.x;

    for (long t = tid; t < total; t += stride) {
        const int c = static_cast<int>(t >> 8);       // chunk index
        const int v = static_cast<int>(t & 0xFF);     // seed low byte value

        const size_t start = static_cast<size_t>(c) * SMTQ_FNV_CHUNK;
        size_t clen = SMTQ_FNV_CHUNK;
        if (start + clen > len) clen = len - start;

        uint64_t h = static_cast<uint64_t>(v);
        for (size_t i = 0; i < clen; ++i) {
            h ^= static_cast<uint64_t>(buf[start + i]);
            h *= SMTQ_FNV_PRIME;
        }
        fnv_T[static_cast<size_t>(c) * 256 + v] = h;

        if (v == 0) {
            uint64_t A = 1ULL;
            for (size_t i = 0; i < clen; ++i) A *= SMTQ_FNV_PRIME;
            fnv_A[c] = A;
        }
    }
}

__global__ void smtq_fnv_combine_kernel(
    int num_chunks,
    const uint64_t* __restrict__ fnv_A,
    const uint64_t* __restrict__ fnv_T,
    uint64_t* __restrict__ out_hash) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t h = SMTQ_FNV_OFFSET;
    for (int c = 0; c < num_chunks; ++c) {
        const uint64_t A = fnv_A[c];
        const uint64_t lo = h & 0xFFULL;
        h = A * (h & ~0xFFULL) + fnv_T[static_cast<size_t>(c) * 256 + lo];
    }
    out_hash[0] = h;
}

// ---------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------
static cudaError_t smtq_launch_fnv(
    SmtqReferenceState* st,
    size_t len,
    uint64_t* out_hash,
    cudaStream_t stream) {
    const int num_chunks = static_cast<int>((len + SMTQ_FNV_CHUNK - 1) / SMTQ_FNV_CHUNK);

    const long units = static_cast<long>(num_chunks) * 256;
    const int threads = 256;
    long blk = (units + threads - 1) / threads;
    if (blk < 1) blk = 1;
    if (blk > 65535) blk = 65535;
    const int blocks = static_cast<int>(blk);

    smtq_fnv_tables_kernel<<<blocks, threads, 0, stream>>>(
        st->packed, len, num_chunks, st->fnv_A, st->fnv_T);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    smtq_fnv_combine_kernel<<<1, 1, 0, stream>>>(
        num_chunks, st->fnv_A, st->fnv_T, out_hash);
    return cudaPeekAtLastError();
}

static cudaError_t smtq_reference_reset_state(SmtqReferenceState* st, cudaStream_t stream) {
    const int G = st->spec.G;
    const int threads = 256;
    const int blocks = (G + threads - 1) / threads;
    smtq_reference_reset_kernel<<<blocks, threads, 0, stream>>>(
        G, st->spec.K, st->spec.num_bins,
        st->topk_key, st->topk_value, st->topk_order, st->topk_count,
        st->hist, st->group_total, st->total_ingested);
    return cudaPeekAtLastError();
}

extern "C" size_t solution_workspace_bytes(const SmtqProblemSpec* spec) {
    if (!smtq_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const SmtqProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!smtq_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    SmtqReferenceState* st =
        static_cast<SmtqReferenceState*>(malloc(sizeof(SmtqReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memset(st, 0, sizeof(SmtqReferenceState));
    memcpy(&st->spec, spec, sizeof(SmtqProblemSpec));

    const size_t G = (size_t)spec->G;
    const size_t K = (size_t)spec->K;
    const size_t num_bins = (size_t)spec->num_bins;
    const size_t max_batch = (size_t)(spec->max_batch > 0 ? spec->max_batch : 1);

    // FNV stream sizes
    const size_t len_hist = 8 + 4 * G * num_bins;
    const size_t seg = 8 + 16 * K + 4 * num_bins;
    const size_t len_state = 28 + G * seg;
    st->packed_cap = len_hist > len_state ? len_hist : len_state;
    st->num_chunks_cap =
        static_cast<int>((st->packed_cap + SMTQ_FNV_CHUNK - 1) / SMTQ_FNV_CHUNK);

    st->cand_w = static_cast<int>(max_batch + K);
    const size_t cand_n = G * (size_t)st->cand_w;

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(reinterpret_cast<void**>(&st->topk_key), sizeof(int32_t) * G * K);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->topk_value), sizeof(int32_t) * G * K);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->topk_order), sizeof(int64_t) * G * K);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->topk_count), sizeof(int32_t) * G);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->hist), sizeof(int32_t) * G * num_bins);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->group_total), sizeof(int32_t) * G);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->total_ingested), sizeof(int64_t));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->order_buf), sizeof(int64_t) * max_batch);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->cand_key), sizeof(int32_t) * cand_n);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->cand_value), sizeof(int32_t) * cand_n);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->cand_order), sizeof(int64_t) * cand_n);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->packed), st->packed_cap);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->fnv_A), sizeof(uint64_t) * (size_t)st->num_chunks_cap);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(reinterpret_cast<void**>(&st->fnv_T), sizeof(uint64_t) * (size_t)st->num_chunks_cap * 256);
    if (err != cudaSuccess) goto fail;

    err = smtq_reference_reset_state(st, stream);
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->topk_key) cudaFree(st->topk_key);
    if (st->topk_value) cudaFree(st->topk_value);
    if (st->topk_order) cudaFree(st->topk_order);
    if (st->topk_count) cudaFree(st->topk_count);
    if (st->hist) cudaFree(st->hist);
    if (st->group_total) cudaFree(st->group_total);
    if (st->total_ingested) cudaFree(st->total_ingested);
    if (st->order_buf) cudaFree(st->order_buf);
    if (st->cand_key) cudaFree(st->cand_key);
    if (st->cand_value) cudaFree(st->cand_value);
    if (st->cand_order) cudaFree(st->cand_order);
    if (st->packed) cudaFree(st->packed);
    if (st->fnv_A) cudaFree(st->fnv_A);
    if (st->fnv_T) cudaFree(st->fnv_T);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const SmtqRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;

    if (!state || !inputs_void || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    SmtqReferenceState* st = static_cast<SmtqReferenceState*>(state);
    if (!smtq_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const SmtqInputs* in = static_cast<const SmtqInputs*>(inputs_void);
    SmtqOutputs* out = static_cast<SmtqOutputs*>(outputs_void);

    if (run->batch_size > 0 && (!in->group || !in->key || !in->value)) {
        return cudaErrorInvalidValue;
    }
    if (!out->topk_keys || !out->topk_values || !out->topk_count ||
        !out->topk_value_sum || !out->histogram_checksum ||
        !out->quantile_key || !out->total_ingested || !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    const int G = st->spec.G;
    const int K = st->spec.K;
    const int num_bins = st->spec.num_bins;
    const int threads = 256;
    const int gblocks = (G + threads - 1) / threads;

    cudaError_t err;

    // 1. per-event insertion order (serial prefix over the batch) + advance total
    smtq_assign_order_kernel<<<1, 1, 0, stream>>>(
        G, run->batch_size, in->group, st->order_buf, st->total_ingested);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    // 2. ingest: one block per group, byte-exact rank-select of the new top-K
    if (run->batch_size > 0) {
        smtq_ingest_ranksel_kernel<<<G, threads, 0, stream>>>(
            G, K, num_bins, st->spec.key_min, st->spec.key_max, run->batch_size,
            st->cand_w,
            in->group, in->key, in->value, st->order_buf,
            st->topk_key, st->topk_value, st->topk_order, st->topk_count,
            st->hist, st->group_total,
            st->cand_key, st->cand_value, st->cand_order);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    // 3. finalize per-group outputs (topk arrays, count, sum, quantile)
    smtq_finalize_kernel<<<gblocks, threads, 0, stream>>>(
        G, K, num_bins, st->spec.key_min, st->spec.key_max,
        run->is_query, run->q_num, run->q_den,
        st->topk_key, st->topk_value, st->topk_count, st->hist, st->group_total,
        out->topk_keys, out->topk_values, out->topk_count,
        out->topk_value_sum, out->quantile_key);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    // 4. total_ingested passthrough
    cudaMemcpyAsync(out->total_ingested, st->total_ingested, sizeof(int64_t),
                    cudaMemcpyDeviceToDevice, stream);

    // 5. histogram checksum (parallel exact FNV)
    {
        const size_t len_hist = 8 + 4 * (size_t)G * (size_t)num_bins;
        const int total = G * num_bins;
        int pblocks = (total + threads - 1) / threads;
        if (pblocks < 1) pblocks = 1;
        smtq_pack_hist_kernel<<<pblocks, threads, 0, stream>>>(
            G, num_bins, st->hist, st->packed);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        err = smtq_launch_fnv(st, len_hist, out->histogram_checksum, stream);
        if (err != cudaSuccess) return err;
    }

    // 6. state checksum (parallel exact FNV)
    {
        const size_t seg = 8 + 16 * (size_t)K + 4 * (size_t)num_bins;
        const size_t len_state = 28 + (size_t)G * seg;

        const int cells = G * num_bins;
        int sblocks = (cells + threads - 1) / threads;
        if (sblocks < 1) sblocks = 1;
        smtq_pack_state_kernel<<<sblocks, threads, 0, stream>>>(
            G, K, num_bins, st->spec.key_min, st->spec.key_max,
            st->total_ingested,
            st->topk_key, st->topk_value, st->topk_order, st->topk_count,
            st->hist, st->group_total, st->packed);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;

        err = smtq_launch_fnv(st, len_state, out->state_checksum, stream);
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return smtq_reference_reset_state(static_cast<SmtqReferenceState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    SmtqReferenceState* st = static_cast<SmtqReferenceState*>(state);
    if (st->topk_key) cudaFree(st->topk_key);
    if (st->topk_value) cudaFree(st->topk_value);
    if (st->topk_order) cudaFree(st->topk_order);
    if (st->topk_count) cudaFree(st->topk_count);
    if (st->hist) cudaFree(st->hist);
    if (st->group_total) cudaFree(st->group_total);
    if (st->total_ingested) cudaFree(st->total_ingested);
    if (st->order_buf) cudaFree(st->order_buf);
    if (st->cand_key) cudaFree(st->cand_key);
    if (st->cand_value) cudaFree(st->cand_value);
    if (st->cand_order) cudaFree(st->cand_order);
    if (st->packed) cudaFree(st->packed);
    if (st->fnv_A) cudaFree(st->fnv_A);
    if (st->fnv_T) cudaFree(st->fnv_T);
    free(st);
}
