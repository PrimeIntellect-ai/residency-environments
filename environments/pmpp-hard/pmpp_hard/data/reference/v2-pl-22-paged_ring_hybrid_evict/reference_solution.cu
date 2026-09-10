// PMPP_CANARY_22_9c6942a857 -- held-out canary; MUST NOT appear in any submission
// file: paged_ring_hybrid_evict_reference.cu

#include "paged_ring_hybrid_evict_common.h"

#include <cuda_runtime.h>

#include <stdlib.h>
#include <string.h>

struct PrheReferenceState {
    PrheProblemSpec spec;
    int32_t max_logical_pages;

    int32_t* page_data;       // max_pages * page_size
    int32_t* page_owner;      // max_pages, -1 free
    int32_t* page_logical;    // max_pages
    int32_t* page_last_used;  // max_pages
    int32_t* page_table;      // B * max_logical_pages, -1 absent

    int32_t* length;          // B
    int32_t* ring_head;       // B
    int32_t* ring_count;      // B
    int32_t* step_counter;    // 1
};

__device__ __forceinline__ uint64_t prhe_fnv_byte_device(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ void prhe_fnv_bytes_device(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = prhe_fnv_byte_device(v, p[i]);
    }
    *h = v;
}

__device__ __forceinline__ int prhe_page_intersects_live_device(
    int seq,
    int logical_page,
    int page_size,
    int window_size,
    const int32_t* __restrict__ length) {
    const int len = length[seq];
    if (len <= 0) return 0;

    int start = len - window_size;
    if (start < 0) start = 0;
    const int end = len - 1;

    const int page_start = logical_page * page_size;
    const int page_end = page_start + page_size - 1;

    return page_start <= end && page_end >= start;
}

__device__ int prhe_alloc_page_device(
    int B,
    int max_pages,
    int max_logical_pages,
    int page_size,
    int window_size,
    int step,
    int* evicted,
    int32_t* __restrict__ page_owner,
    int32_t* __restrict__ page_logical,
    int32_t* __restrict__ page_last_used,
    int32_t* __restrict__ page_table,
    const int32_t* __restrict__ length) {
    for (int p = 0; p < max_pages; ++p) {
        if (page_owner[p] < 0) {
            return p;
        }
    }

    int best = -1;
    int best_last = 2147483647;

    for (int p = 0; p < max_pages; ++p) {
        const int owner = page_owner[p];
        if (owner < 0 || owner >= B) continue;

        const int lp = page_logical[p];
        if (prhe_page_intersects_live_device(owner, lp, page_size, window_size, length)) {
            continue;
        }

        const int lu = page_last_used[p];
        if (best < 0 || lu < best_last || (lu == best_last && p < best)) {
            best = p;
            best_last = lu;
        }
    }

    if (best < 0) {
        return -1;
    }

    const int old_owner = page_owner[best];
    const int old_lp = page_logical[best];

    if (old_owner >= 0 && old_owner < B && old_lp >= 0 && old_lp < max_logical_pages) {
        const int idx = old_owner * max_logical_pages + old_lp;
        if (page_table[idx] == best) {
            page_table[idx] = -1;
        }
    }

    page_owner[best] = -1;
    page_logical[best] = -1;
    page_last_used[best] = step;

    *evicted += 1;
    return best;
}

__device__ void prhe_append_one_device(
    int B,
    int max_len,
    int page_size,
    int window_size,
    int max_pages,
    int max_logical_pages,
    int step,
    int seq,
    int32_t value,
    int* evicted,
    int32_t* __restrict__ page_data,
    int32_t* __restrict__ page_owner,
    int32_t* __restrict__ page_logical,
    int32_t* __restrict__ page_last_used,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ length,
    int32_t* __restrict__ ring_head,
    int32_t* __restrict__ ring_count) {
    if (seq < 0 || seq >= B) return;

    int len = length[seq];
    if (len >= max_len) return;

    const int logical_page = len / page_size;
    const int page_offset = len - logical_page * page_size;
    const int table_idx = seq * max_logical_pages + logical_page;

    int phys = page_table[table_idx];

    if (phys < 0) {
        phys = prhe_alloc_page_device(
            B,
            max_pages,
            max_logical_pages,
            page_size,
            window_size,
            step,
            evicted,
            page_owner,
            page_logical,
            page_last_used,
            page_table,
            length);

        if (phys < 0) {
            return;
        }

        page_table[table_idx] = phys;
        page_owner[phys] = seq;
        page_logical[phys] = logical_page;
        page_last_used[phys] = step;
    }

    page_data[phys * page_size + page_offset] = value;
    page_last_used[phys] = step;

    len += 1;
    length[seq] = len;

    ring_head[seq] = len % window_size;
    ring_count[seq] = len < window_size ? len : window_size;
}

__device__ uint64_t prhe_live_hash_device(
    int seq,
    int max_logical_pages,
    int page_size,
    int window_size,
    const int32_t* __restrict__ page_data,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ length,
    int32_t* __restrict__ page_last_used,
    int step,
    int32_t* live_count_out,
    int64_t* live_sum_out) {
    const int len = length[seq];
    int start = len - window_size;
    if (start < 0) start = 0;

    int count = 0;
    uint64_t sum_bits = 0;

    for (int pos = start; pos < len; ++pos) {
        const int lp = pos / page_size;
        const int off = pos - lp * page_size;
        const int phys = page_table[seq * max_logical_pages + lp];

        if (phys >= 0) {
            ++count;
            const int32_t v = page_data[phys * page_size + off];
            sum_bits += static_cast<uint64_t>(static_cast<int64_t>(v));
        }
    }

    uint64_t h = 1469598103934665603ULL;
    prhe_fnv_bytes_device(&h, &count, sizeof(int32_t));

    for (int pos = start; pos < len; ++pos) {
        const int lp = pos / page_size;
        const int off = pos - lp * page_size;
        const int phys = page_table[seq * max_logical_pages + lp];

        if (phys >= 0) {
            const int32_t p32 = static_cast<int32_t>(pos);
            const int32_t v = page_data[phys * page_size + off];
            prhe_fnv_bytes_device(&h, &p32, sizeof(int32_t));
            prhe_fnv_bytes_device(&h, &v, sizeof(int32_t));
            page_last_used[phys] = step;
        }
    }

    *live_count_out = count;
    *live_sum_out = static_cast<int64_t>(sum_bits);
    return h;
}

__device__ uint64_t prhe_state_checksum_device(
    int B,
    int max_len,
    int page_size,
    int window_size,
    int max_pages,
    int max_logical_pages,
    int step,
    const int32_t* __restrict__ page_owner,
    const int32_t* __restrict__ page_logical,
    const int32_t* __restrict__ page_last_used,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ length,
    const int32_t* __restrict__ ring_head,
    const int32_t* __restrict__ ring_count) {
    uint64_t h = 1469598103934665603ULL;

    prhe_fnv_bytes_device(&h, &B, sizeof(int32_t));
    prhe_fnv_bytes_device(&h, &max_len, sizeof(int32_t));
    prhe_fnv_bytes_device(&h, &page_size, sizeof(int32_t));
    prhe_fnv_bytes_device(&h, &window_size, sizeof(int32_t));
    prhe_fnv_bytes_device(&h, &max_pages, sizeof(int32_t));
    prhe_fnv_bytes_device(&h, &step, sizeof(int32_t));

    for (int s = 0; s < B; ++s) {
        prhe_fnv_bytes_device(&h, &length[s], sizeof(int32_t));
        prhe_fnv_bytes_device(&h, &ring_head[s], sizeof(int32_t));
        prhe_fnv_bytes_device(&h, &ring_count[s], sizeof(int32_t));

        const int base = s * max_logical_pages;
        for (int lp = 0; lp < max_logical_pages; ++lp) {
            prhe_fnv_bytes_device(&h, &page_table[base + lp], sizeof(int32_t));
        }
    }

    for (int p = 0; p < max_pages; ++p) {
        prhe_fnv_bytes_device(&h, &page_owner[p], sizeof(int32_t));
        prhe_fnv_bytes_device(&h, &page_logical[p], sizeof(int32_t));
        prhe_fnv_bytes_device(&h, &page_last_used[p], sizeof(int32_t));
    }

    return h;
}

__global__ void prhe_ref_step_kernel(
    int B,
    int max_len,
    int page_size,
    int window_size,
    int max_pages,
    int max_logical_pages,
    int max_new_tokens,
    int active_count,
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ append_count,
    const int32_t* __restrict__ token_values,
    int32_t* __restrict__ page_data,
    int32_t* __restrict__ page_owner,
    int32_t* __restrict__ page_logical,
    int32_t* __restrict__ page_last_used,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ length,
    int32_t* __restrict__ ring_head,
    int32_t* __restrict__ ring_count,
    int32_t* __restrict__ step_counter,
    int32_t* __restrict__ live_count,
    int64_t* __restrict__ live_sum,
    uint64_t* __restrict__ live_hash,
    uint64_t* __restrict__ page_table_checksum,
    int32_t* __restrict__ evicted_count,
    int32_t* __restrict__ free_pages) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int step = step_counter[0] + 1;
    step_counter[0] = step;

    int evicted = 0;

    for (int r = 0; r < active_count; ++r) {
        const int seq = active_seq[r];

        int cnt = append_count[r];
        if (cnt < 0) cnt = 0;
        if (cnt > max_new_tokens) cnt = max_new_tokens;
        if (cnt > PRHE_MAX_NEW_TOKENS) cnt = PRHE_MAX_NEW_TOKENS;

        for (int i = 0; i < cnt; ++i) {
            prhe_append_one_device(
                B,
                max_len,
                page_size,
                window_size,
                max_pages,
                max_logical_pages,
                step,
                seq,
                token_values[(size_t)r * (size_t)max_new_tokens + (size_t)i],
                &evicted,
                page_data,
                page_owner,
                page_logical,
                page_last_used,
                page_table,
                length,
                ring_head,
                ring_count);
        }
    }

    for (int seq = 0; seq < B; ++seq) {
        int32_t cnt = 0;
        int64_t sum = 0;
        const uint64_t h = prhe_live_hash_device(
            seq,
            max_logical_pages,
            page_size,
            window_size,
            page_data,
            page_table,
            length,
            page_last_used,
            step,
            &cnt,
            &sum);

        live_count[seq] = cnt;
        live_sum[seq] = sum;
        live_hash[seq] = h;
    }

    int free_count = 0;
    for (int p = 0; p < max_pages; ++p) {
        if (page_owner[p] < 0) ++free_count;
    }

    page_table_checksum[0] = prhe_state_checksum_device(
        B,
        max_len,
        page_size,
        window_size,
        max_pages,
        max_logical_pages,
        step,
        page_owner,
        page_logical,
        page_last_used,
        page_table,
        length,
        ring_head,
        ring_count);

    evicted_count[0] = evicted;
    free_pages[0] = free_count;
}

static cudaError_t prhe_reference_reset_state(PrheReferenceState* st, cudaStream_t stream) {
    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(
        st->page_data,
        0,
        sizeof(int32_t) * (size_t)st->spec.max_pages * (size_t)st->spec.page_size,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->page_owner, 0xff, sizeof(int32_t) * (size_t)st->spec.max_pages, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->page_logical, 0xff, sizeof(int32_t) * (size_t)st->spec.max_pages, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->page_last_used, 0, sizeof(int32_t) * (size_t)st->spec.max_pages, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(
        st->page_table,
        0xff,
        sizeof(int32_t) * (size_t)st->spec.B * (size_t)st->max_logical_pages,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->length, 0, sizeof(int32_t) * (size_t)st->spec.B, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->ring_head, 0, sizeof(int32_t) * (size_t)st->spec.B, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->ring_count, 0, sizeof(int32_t) * (size_t)st->spec.B, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->step_counter, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" size_t solution_workspace_bytes(const PrheProblemSpec* spec) {
    if (!prhe_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const PrheProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!prhe_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    PrheReferenceState* st =
        static_cast<PrheReferenceState*>(malloc(sizeof(PrheReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memset(st, 0, sizeof(PrheReferenceState));
    memcpy(&st->spec, spec, sizeof(PrheProblemSpec));
    st->max_logical_pages = prhe_max_logical_pages_host(spec->max_len, spec->page_size);

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(
        reinterpret_cast<void**>(&st->page_data),
        sizeof(int32_t) * (size_t)spec->max_pages * (size_t)spec->page_size);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->page_owner), sizeof(int32_t) * (size_t)spec->max_pages);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->page_logical), sizeof(int32_t) * (size_t)spec->max_pages);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->page_last_used), sizeof(int32_t) * (size_t)spec->max_pages);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(
        reinterpret_cast<void**>(&st->page_table),
        sizeof(int32_t) * (size_t)spec->B * (size_t)st->max_logical_pages);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->length), sizeof(int32_t) * (size_t)spec->B);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->ring_head), sizeof(int32_t) * (size_t)spec->B);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->ring_count), sizeof(int32_t) * (size_t)spec->B);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->step_counter), sizeof(int32_t));
    if (err != cudaSuccess) goto fail;

    err = prhe_reference_reset_state(st, stream);
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->page_data) cudaFree(st->page_data);
    if (st->page_owner) cudaFree(st->page_owner);
    if (st->page_logical) cudaFree(st->page_logical);
    if (st->page_last_used) cudaFree(st->page_last_used);
    if (st->page_table) cudaFree(st->page_table);
    if (st->length) cudaFree(st->length);
    if (st->ring_head) cudaFree(st->ring_head);
    if (st->ring_count) cudaFree(st->ring_count);
    if (st->step_counter) cudaFree(st->step_counter);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const PrheRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;

    if (!state || !inputs_void || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    PrheReferenceState* st = static_cast<PrheReferenceState*>(state);
    if (!prhe_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const PrheInputs* in = static_cast<const PrheInputs*>(inputs_void);
    PrheOutputs* out = static_cast<PrheOutputs*>(outputs_void);

    if (run->active_count > 0) {
        if (!in->active_seq || !in->append_count || !in->token_values) {
            return cudaErrorInvalidValue;
        }
    }

    if (!out->live_count || !out->live_sum || !out->live_hash ||
        !out->page_table_checksum || !out->evicted_count || !out->free_pages) {
        return cudaErrorInvalidValue;
    }

    prhe_ref_step_kernel<<<1, 1, 0, stream>>>(
        st->spec.B,
        st->spec.max_len,
        st->spec.page_size,
        st->spec.window_size,
        st->spec.max_pages,
        st->max_logical_pages,
        st->spec.max_new_tokens,
        run->active_count,
        in->active_seq,
        in->append_count,
        in->token_values,
        st->page_data,
        st->page_owner,
        st->page_logical,
        st->page_last_used,
        st->page_table,
        st->length,
        st->ring_head,
        st->ring_count,
        st->step_counter,
        out->live_count,
        out->live_sum,
        out->live_hash,
        out->page_table_checksum,
        out->evicted_count,
        out->free_pages);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return prhe_reference_reset_state(static_cast<PrheReferenceState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;

    PrheReferenceState* st = static_cast<PrheReferenceState*>(state);
    if (st->page_data) cudaFree(st->page_data);
    if (st->page_owner) cudaFree(st->page_owner);
    if (st->page_logical) cudaFree(st->page_logical);
    if (st->page_last_used) cudaFree(st->page_last_used);
    if (st->page_table) cudaFree(st->page_table);
    if (st->length) cudaFree(st->length);
    if (st->ring_head) cudaFree(st->ring_head);
    if (st->ring_count) cudaFree(st->ring_count);
    if (st->step_counter) cudaFree(st->step_counter);
    free(st);
}
