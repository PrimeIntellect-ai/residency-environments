// PMPP_CANARY_37_ab6036b8e1 -- held-out canary; MUST NOT appear in any submission
// file: deterministic_mergeable_quantile_sketch_reference.cu

#include "deterministic_mergeable_quantile_sketch_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Per-level slab capacity. Level 0 can transiently hold (k-1)+max_batch keys;
// higher levels hold at most (k-1)+(k/2). slab = k + max_batch covers both.
static inline size_t dmqs_ref_slab(const DmqsProblemSpec* spec) {
    return (size_t)spec->k + (size_t)spec->max_batch + (size_t)spec->k;
}

struct DmqsReferenceState {
    DmqsProblemSpec spec;
    size_t slab;

    int32_t* level_buf;        // num_levels * slab
    int32_t* level_size;       // num_levels
    int64_t* compaction_counter;  // num_levels

    int32_t* scratch_key;      // slab (sort scratch)
    int32_t* scratch_ord;      // slab
};

__device__ __forceinline__ uint64_t dmqs_fnv_byte_dev(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= 1099511628211ULL;
    return h;
}

__device__ void dmqs_fnv_bytes_dev(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = (const uint8_t*)ptr;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = dmqs_fnv_byte_dev(v, p[i]);
    *h = v;
}

// Stable insertion sort of (key,order) pairs by (key asc, order asc).
__device__ void dmqs_stable_sort_dev(int32_t* key, int32_t* ord, int n) {
    for (int i = 1; i < n; ++i) {
        int32_t ck = key[i];
        int32_t co = ord[i];
        int j = i - 1;
        while (j >= 0 && (key[j] > ck || (key[j] == ck && ord[j] > co))) {
            key[j + 1] = key[j];
            ord[j + 1] = ord[j];
            --j;
        }
        key[j + 1] = ck;
        ord[j + 1] = co;
    }
}

// Compact the FIRST k slots of level L (FIFO). Remainder shifts down. Survivors
// appended to level L+1. Caller guarantees level_size[L] >= k.
__device__ void dmqs_compact_first_k_dev(
    int L,
    int k,
    size_t slab,
    int32_t* __restrict__ level_buf,
    int32_t* __restrict__ level_size,
    int64_t* __restrict__ compaction_counter,
    int32_t* __restrict__ scratch_key,
    int32_t* __restrict__ scratch_ord) {
    int32_t* base = level_buf + (size_t)L * slab;
    const int sz = level_size[L];

    // Load first k slots with insertion order = slot index.
    for (int j = 0; j < k; ++j) {
        scratch_key[j] = base[j];
        scratch_ord[j] = j;
    }
    dmqs_stable_sort_dev(scratch_key, scratch_ord, k);

    const int parity = (int)(compaction_counter[L] & 1);
    const int start = (parity == 0) ? 1 : 0;  // parity0 keeps odd, parity1 keeps even

    // Append survivors (ascending) to level L+1.
    int32_t* up = level_buf + (size_t)(L + 1) * slab;
    int up_sz = level_size[L + 1];
    for (int i = start; i < k; i += 2) {
        up[up_sz++] = scratch_key[i];
    }
    level_size[L + 1] = up_sz;

    compaction_counter[L] += 1;

    // Shift remainder (slots k..sz) down to slots 0..(sz-k).
    const int rem = sz - k;
    for (int j = 0; j < rem; ++j) {
        base[j] = base[k + j];
    }
    level_size[L] = rem;
}

__device__ void dmqs_cascade_dev(
    int num_levels,
    int k,
    size_t slab,
    int32_t* __restrict__ level_buf,
    int32_t* __restrict__ level_size,
    int64_t* __restrict__ compaction_counter,
    int32_t* __restrict__ scratch_key,
    int32_t* __restrict__ scratch_ord) {
    for (int L = 0; L < num_levels; ++L) {
        while (level_size[L] >= k) {
            dmqs_compact_first_k_dev(
                L, k, slab, level_buf, level_size, compaction_counter,
                scratch_key, scratch_ord);
        }
    }
}

__device__ int64_t dmqs_total_weight_dev(
    int num_levels, const int32_t* __restrict__ level_size) {
    int64_t tw = 0;
    for (int L = 0; L < num_levels; ++L) {
        tw += (int64_t)level_size[L] * (int64_t)(1LL << L);
    }
    return tw;
}

__device__ int dmqs_num_retained_dev(
    int num_levels, const int32_t* __restrict__ level_size) {
    int n = 0;
    for (int L = 0; L < num_levels; ++L) n += level_size[L];
    return n;
}

__device__ int dmqs_num_levels_dev(
    int num_levels,
    const int32_t* __restrict__ level_size,
    const int64_t* __restrict__ compaction_counter) {
    int highest = -1;
    for (int L = 0; L < num_levels; ++L) {
        if (level_size[L] > 0 || compaction_counter[L] > 0) highest = L;
    }
    return highest + 1;
}

__device__ uint64_t dmqs_sketch_checksum_dev(
    int num_levels,
    size_t slab,
    const int32_t* __restrict__ level_buf,
    const int32_t* __restrict__ level_size,
    int32_t* __restrict__ scratch_key,
    int32_t* __restrict__ scratch_ord) {
    uint64_t h = 1469598103934665603ULL;
    for (int L = 0; L < num_levels; ++L) {
        const int32_t lv = L;
        const int32_t sz = level_size[L];
        dmqs_fnv_bytes_dev(&h, &lv, sizeof(int32_t));
        dmqs_fnv_bytes_dev(&h, &sz, sizeof(int32_t));

        const int32_t* base = level_buf + (size_t)L * slab;
        for (int j = 0; j < sz; ++j) {
            scratch_key[j] = base[j];
            scratch_ord[j] = j;
        }
        dmqs_stable_sort_dev(scratch_key, scratch_ord, sz);

        const int32_t weight = (int32_t)(1LL << L);
        for (int j = 0; j < sz; ++j) {
            dmqs_fnv_bytes_dev(&h, &scratch_key[j], sizeof(int32_t));
            dmqs_fnv_bytes_dev(&h, &scratch_ord[j], sizeof(int32_t));
            dmqs_fnv_bytes_dev(&h, &weight, sizeof(int32_t));
        }
    }
    return h;
}

__device__ uint64_t dmqs_state_checksum_dev(
    int k,
    int num_levels,
    size_t slab,
    const int32_t* __restrict__ level_buf,
    const int32_t* __restrict__ level_size,
    const int64_t* __restrict__ compaction_counter) {
    uint64_t h = 1469598103934665603ULL;
    const int32_t kk = k;
    const int32_t nl = num_levels;
    const int64_t tw = dmqs_total_weight_dev(num_levels, level_size);
    const int32_t nr = dmqs_num_retained_dev(num_levels, level_size);
    dmqs_fnv_bytes_dev(&h, &kk, sizeof(int32_t));
    dmqs_fnv_bytes_dev(&h, &nl, sizeof(int32_t));
    dmqs_fnv_bytes_dev(&h, &tw, sizeof(int64_t));
    dmqs_fnv_bytes_dev(&h, &nr, sizeof(int32_t));

    for (int L = 0; L < num_levels; ++L) {
        const int32_t lv = L;
        const int32_t sz = level_size[L];
        const int64_t cc = compaction_counter[L];
        dmqs_fnv_bytes_dev(&h, &lv, sizeof(int32_t));
        dmqs_fnv_bytes_dev(&h, &sz, sizeof(int32_t));
        dmqs_fnv_bytes_dev(&h, &cc, sizeof(int64_t));
        const int32_t* base = level_buf + (size_t)L * slab;
        for (int j = 0; j < sz; ++j) {
            dmqs_fnv_bytes_dev(&h, &base[j], sizeof(int32_t));
        }
    }
    return h;
}

// Weighted-rank query. Uses scratch_key/scratch_ord/scratch_lvl as a flat
// global sort buffer of all retained items.
__device__ int32_t dmqs_query_dev(
    int q_num,
    int q_den,
    int num_levels,
    size_t slab,
    const int32_t* __restrict__ level_buf,
    const int32_t* __restrict__ level_size,
    int32_t* __restrict__ s_key,
    int32_t* __restrict__ s_lvl,
    int32_t* __restrict__ s_ord) {
    const int64_t total = dmqs_total_weight_dev(num_levels, level_size);
    if (total <= 0) return INT_MIN;

    if (q_num < 0) q_num = 0;
    if (q_num > q_den) q_num = q_den;

    int64_t target = ((int64_t)q_num * total + q_den - 1) / q_den;
    if (target < 1) target = 1;
    if (target > total) target = total;

    // Gather all items into s_key/s_lvl/s_ord.
    int n = 0;
    for (int L = 0; L < num_levels; ++L) {
        const int32_t* base = level_buf + (size_t)L * slab;
        const int sz = level_size[L];
        for (int j = 0; j < sz; ++j) {
            s_key[n] = base[j];
            s_lvl[n] = L;
            s_ord[n] = j;
            ++n;
        }
    }

    // Stable insertion sort by (key asc, level asc, order asc).
    for (int i = 1; i < n; ++i) {
        int32_t ck = s_key[i], cl = s_lvl[i], co = s_ord[i];
        int j = i - 1;
        while (j >= 0 &&
               (s_key[j] > ck ||
                (s_key[j] == ck && s_lvl[j] > cl) ||
                (s_key[j] == ck && s_lvl[j] == cl && s_ord[j] > co))) {
            s_key[j + 1] = s_key[j];
            s_lvl[j + 1] = s_lvl[j];
            s_ord[j + 1] = s_ord[j];
            --j;
        }
        s_key[j + 1] = ck;
        s_lvl[j + 1] = cl;
        s_ord[j + 1] = co;
    }

    int64_t accum = 0;
    for (int i = 0; i < n; ++i) {
        accum += (int64_t)(1LL << s_lvl[i]);
        if (accum >= target) return s_key[i];
    }
    return s_key[n - 1];
}

__global__ void dmqs_reference_reset_kernel(
    int num_levels,
    int32_t* __restrict__ level_size,
    int64_t* __restrict__ compaction_counter) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int L = 0; L < num_levels; ++L) {
        level_size[L] = 0;
        compaction_counter[L] = 0;
    }
}

__global__ void dmqs_reference_step_kernel(
    int k,
    int num_levels,
    size_t slab,
    int op,
    int batch_size,
    int q_num,
    int q_den,
    int merge_num_levels,
    const int32_t* __restrict__ in_keys,
    const int32_t* __restrict__ in_merge_size,
    const int32_t* __restrict__ in_merge_keys,
    int32_t* __restrict__ level_buf,
    int32_t* __restrict__ level_size,
    int64_t* __restrict__ compaction_counter,
    int32_t* __restrict__ scratch_key,
    int32_t* __restrict__ scratch_ord,
    int64_t* __restrict__ out_total_weight,
    int32_t* __restrict__ out_num_levels,
    int32_t* __restrict__ out_num_retained,
    int32_t* __restrict__ out_query_result,
    uint64_t* __restrict__ out_sketch_checksum,
    uint64_t* __restrict__ out_state_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    if (op == DMQS_OP_INGEST) {
        int32_t* l0 = level_buf;  // level 0 base
        int sz = level_size[0];
        for (int i = 0; i < batch_size; ++i) {
            l0[sz++] = in_keys[i];
        }
        level_size[0] = sz;
        dmqs_cascade_dev(num_levels, k, slab, level_buf, level_size,
                         compaction_counter, scratch_key, scratch_ord);
    } else if (op == DMQS_OP_MERGE) {
        for (int L = 0; L < merge_num_levels; ++L) {
            int32_t* base = level_buf + (size_t)L * slab;
            int sz = level_size[L];
            const int msz = in_merge_size[L];
            for (int j = 0; j < msz; ++j) {
                base[sz++] = in_merge_keys[(size_t)L * (size_t)k + (size_t)j];
            }
            level_size[L] = sz;
        }
        dmqs_cascade_dev(num_levels, k, slab, level_buf, level_size,
                         compaction_counter, scratch_key, scratch_ord);
    }

    out_total_weight[0] = dmqs_total_weight_dev(num_levels, level_size);
    out_num_levels[0] = dmqs_num_levels_dev(num_levels, level_size, compaction_counter);
    out_num_retained[0] = dmqs_num_retained_dev(num_levels, level_size);

    if (op == DMQS_OP_QUERY) {
        // For query we need a 3-field global sort buffer. Reuse scratch_key as
        // s_key, scratch_ord as s_ord, and pack levels into a separate region:
        // we lay s_lvl right after s_key/s_ord using the (k+max_batch) headroom
        // is insufficient for ALL items; instead the host sizes scratch to hold
        // total retained. We use scratch_key[0..n) and need s_lvl,s_ord too.
        // The host allocates scratch as 3 contiguous regions of `slab*num_levels`.
        // Here scratch_key points to region0, scratch_ord to region1; region2 is
        // contiguous after scratch_ord.
        int32_t* s_key = scratch_key;
        int32_t* s_ord = scratch_ord;
        int32_t* s_lvl = scratch_ord + (size_t)slab * (size_t)num_levels;
        out_query_result[0] = dmqs_query_dev(
            q_num, q_den, num_levels, slab, level_buf, level_size,
            s_key, s_lvl, s_ord);
    } else {
        out_query_result[0] = INT_MIN;
    }

    out_sketch_checksum[0] = dmqs_sketch_checksum_dev(
        num_levels, slab, level_buf, level_size, scratch_key, scratch_ord);
    out_state_checksum[0] = dmqs_state_checksum_dev(
        k, num_levels, slab, level_buf, level_size, compaction_counter);
}

static cudaError_t dmqs_reference_reset_state(DmqsReferenceState* st, cudaStream_t stream) {
    dmqs_reference_reset_kernel<<<1, 1, 0, stream>>>(
        st->spec.num_levels, st->level_size, st->compaction_counter);
    return cudaPeekAtLastError();
}

extern "C" size_t solution_workspace_bytes(const DmqsProblemSpec* spec) {
    if (!dmqs_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const DmqsProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!dmqs_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;

    DmqsReferenceState* st =
        (DmqsReferenceState*)malloc(sizeof(DmqsReferenceState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(DmqsReferenceState));
    memcpy(&st->spec, spec, sizeof(DmqsProblemSpec));
    st->slab = dmqs_ref_slab(spec);

    const size_t nlev = (size_t)spec->num_levels;
    const size_t buf_elems = nlev * st->slab;
    // scratch must hold the full global query sort: 3 arrays of buf_elems each.
    const size_t scratch_elems = buf_elems;  // scratch_key
    const size_t scratch_elems2 = buf_elems * 2;  // scratch_ord + s_lvl region

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&st->level_buf, sizeof(int32_t) * buf_elems);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&st->level_size, sizeof(int32_t) * nlev);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&st->compaction_counter, sizeof(int64_t) * nlev);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&st->scratch_key, sizeof(int32_t) * scratch_elems);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc((void**)&st->scratch_ord, sizeof(int32_t) * scratch_elems2);
    if (err != cudaSuccess) goto fail;

    err = dmqs_reference_reset_state(st, stream);
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->level_buf) cudaFree(st->level_buf);
    if (st->level_size) cudaFree(st->level_size);
    if (st->compaction_counter) cudaFree(st->compaction_counter);
    if (st->scratch_key) cudaFree(st->scratch_key);
    if (st->scratch_ord) cudaFree(st->scratch_ord);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const DmqsRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;

    if (!state || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    DmqsReferenceState* st = (DmqsReferenceState*)state;
    if (!dmqs_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const DmqsInputs* in = (const DmqsInputs*)inputs_void;
    DmqsOutputs* out = (DmqsOutputs*)outputs_void;

    if (run->op == DMQS_OP_INGEST && run->batch_size > 0 && (!in || !in->keys)) {
        return cudaErrorInvalidValue;
    }
    if (run->op == DMQS_OP_MERGE && run->merge_num_levels > 0 &&
        (!in || !in->merge_level_size || !in->merge_keys)) {
        return cudaErrorInvalidValue;
    }

    if (!out->total_weight || !out->num_levels || !out->num_retained_items ||
        !out->query_result || !out->sketch_checksum || !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    const int32_t* in_keys = (run->op == DMQS_OP_INGEST && in) ? in->keys : nullptr;
    const int32_t* in_msize = (run->op == DMQS_OP_MERGE && in) ? in->merge_level_size : nullptr;
    const int32_t* in_mkeys = (run->op == DMQS_OP_MERGE && in) ? in->merge_keys : nullptr;

    dmqs_reference_step_kernel<<<1, 1, 0, stream>>>(
        st->spec.k,
        st->spec.num_levels,
        st->slab,
        run->op,
        run->batch_size,
        run->q_num,
        run->q_den,
        run->merge_num_levels,
        in_keys,
        in_msize,
        in_mkeys,
        st->level_buf,
        st->level_size,
        st->compaction_counter,
        st->scratch_key,
        st->scratch_ord,
        out->total_weight,
        out->num_levels,
        out->num_retained_items,
        out->query_result,
        out->sketch_checksum,
        out->state_checksum);

    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return dmqs_reference_reset_state((DmqsReferenceState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    DmqsReferenceState* st = (DmqsReferenceState*)state;
    if (st->level_buf) cudaFree(st->level_buf);
    if (st->level_size) cudaFree(st->level_size);
    if (st->compaction_counter) cudaFree(st->compaction_counter);
    if (st->scratch_key) cudaFree(st->scratch_key);
    if (st->scratch_ord) cudaFree(st->scratch_ord);
    free(st);
}
