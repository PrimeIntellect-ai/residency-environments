// PMPP_CANARY_06_94123577ea -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: spec_decode_verify_rollback_reference.cu
// ============================================================================

#include "spec_decode_verify_rollback_common.h"

#include <cuda_runtime.h>

#include <stdlib.h>
#include <string.h>

struct SdvReferenceState {
    SdvProblemSpec spec;
    int32_t* cache;
    int32_t* length;
    int32_t* checkpoint_length;
};

__device__ __forceinline__ int sdv_accept_device(
    uint32_t p_target,
    uint32_t p_draft,
    uint32_t uniform_u32) {
    if (p_draft == 0U) {
        return p_target > 0U;
    }

    const unsigned long long lhs =
        static_cast<unsigned long long>(uniform_u32) *
        static_cast<unsigned long long>(p_draft);

    const unsigned long long rhs =
        static_cast<unsigned long long>(p_target) *
        4294967295ULL;

    return lhs <= rhs;
}

__device__ __forceinline__ uint64_t sdv_fnv_byte_device(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ void sdv_fnv_bytes_device(uint64_t* h, const uint8_t* p, size_t n) {
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = sdv_fnv_byte_device(v, p[i]);
    }
    *h = v;
}

static cudaError_t sdv_reference_reset_state(SdvReferenceState* st, cudaStream_t stream) {
    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(
        st->cache,
        0,
        sizeof(int32_t) * (size_t)st->spec.B * (size_t)st->spec.max_len,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(
        st->length,
        0,
        sizeof(int32_t) * (size_t)st->spec.B,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(
        st->checkpoint_length,
        0,
        sizeof(int32_t) * (size_t)st->spec.B,
        stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

__global__ void sdv_ref_step_kernel(
    int B,
    int max_len,
    int active_count,
    int draft_len,
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ draft_value,
    const int32_t* __restrict__ correction_value,
    const uint32_t* __restrict__ p_target,
    const uint32_t* __restrict__ p_draft,
    const uint32_t* __restrict__ uniform_u32,
    int32_t* __restrict__ cache,
    int32_t* __restrict__ length,
    int32_t* __restrict__ checkpoint_length,
    int32_t* __restrict__ accepted_count,
    int32_t* __restrict__ new_length,
    int64_t* __restrict__ live_cache_sum,
    uint64_t* __restrict__ live_cache_tail_hash) {
    const int a = blockIdx.x;
    if (a >= active_count || threadIdx.x != 0) return;

    const int seq = active_seq[a];

    if (seq < 0 || seq >= B) {
        accepted_count[a] = 0;
        new_length[a] = 0;
        live_cache_sum[a] = 0;
        live_cache_tail_hash[a] = 1469598103934665603ULL;
        return;
    }

    const int old_len = length[seq];
    checkpoint_length[seq] = old_len;

    int prefix = 0;
    for (int i = 0; i < draft_len; ++i) {
        const size_t pidx = (size_t)a * (size_t)draft_len + (size_t)i;
        const int accept = sdv_accept_device(
            p_target[pidx],
            p_draft[pidx],
            uniform_u32[pidx]);

        if (!accept) {
            break;
        }

        ++prefix;
    }

    int final_len = old_len;
    int retained_accepted = 0;

    for (int i = 0; i < prefix; ++i) {
        if (final_len >= max_len) {
            break;
        }

        cache[(size_t)seq * (size_t)max_len + (size_t)final_len] =
            draft_value[(size_t)a * (size_t)draft_len + (size_t)i];

        ++final_len;
        ++retained_accepted;
    }

    if (old_len + prefix < max_len && final_len < max_len) {
        cache[(size_t)seq * (size_t)max_len + (size_t)final_len] =
            correction_value[a];
        ++final_len;
    }

    length[seq] = final_len;

    unsigned long long sum_bits = 0ULL;

    for (int i = 0; i < final_len; ++i) {
        const int32_t v = cache[(size_t)seq * (size_t)max_len + (size_t)i];
        sum_bits += static_cast<unsigned long long>(static_cast<long long>(v));
    }

    int tail_count = final_len < SDV_TAIL_VALUES ? final_len : SDV_TAIL_VALUES;
    int tail_start = final_len - tail_count;

    uint64_t h = 1469598103934665603ULL;
    sdv_fnv_bytes_device(
        &h,
        reinterpret_cast<const uint8_t*>(&tail_count),
        sizeof(int32_t));

    if (tail_count > 0) {
        sdv_fnv_bytes_device(
            &h,
            reinterpret_cast<const uint8_t*>(
                &cache[(size_t)seq * (size_t)max_len + (size_t)tail_start]),
            sizeof(int32_t) * (size_t)tail_count);
    }

    accepted_count[a] = retained_accepted;
    new_length[a] = final_len;
    live_cache_sum[a] = static_cast<int64_t>(sum_bits);
    live_cache_tail_hash[a] = h;
}

__global__ void sdv_ref_checksum_kernel(
    int B,
    int max_len,
    const int32_t* __restrict__ cache,
    const int32_t* __restrict__ length,
    const int32_t* __restrict__ checkpoint_length,
    uint64_t* __restrict__ state_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint64_t h = 1469598103934665603ULL;

    sdv_fnv_bytes_device(
        &h,
        reinterpret_cast<const uint8_t*>(length),
        sizeof(int32_t) * (size_t)B);

    sdv_fnv_bytes_device(
        &h,
        reinterpret_cast<const uint8_t*>(checkpoint_length),
        sizeof(int32_t) * (size_t)B);

    for (int b = 0; b < B; ++b) {
        int live_len = length[b];
        if (live_len < 0) live_len = 0;
        if (live_len > max_len) live_len = max_len;

        sdv_fnv_bytes_device(
            &h,
            reinterpret_cast<const uint8_t*>(&live_len),
            sizeof(int32_t));

        if (live_len > 0) {
            sdv_fnv_bytes_device(
                &h,
                reinterpret_cast<const uint8_t*>(
                    &cache[(size_t)b * (size_t)max_len]),
                sizeof(int32_t) * (size_t)live_len);
        }
    }

    state_checksum[0] = h;
}

extern "C" size_t solution_workspace_bytes(const SdvProblemSpec* spec) {
    if (!sdv_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const SdvProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!sdv_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    SdvReferenceState* st = static_cast<SdvReferenceState*>(malloc(sizeof(SdvReferenceState)));
    if (!st) {
        return cudaErrorMemoryAllocation;
    }

    memset(st, 0, sizeof(SdvReferenceState));
    memcpy(&st->spec, spec, sizeof(SdvProblemSpec));

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(
        reinterpret_cast<void**>(&st->cache),
        sizeof(int32_t) * (size_t)spec->B * (size_t)spec->max_len);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(
        reinterpret_cast<void**>(&st->length),
        sizeof(int32_t) * (size_t)spec->B);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(
        reinterpret_cast<void**>(&st->checkpoint_length),
        sizeof(int32_t) * (size_t)spec->B);
    if (err != cudaSuccess) goto fail;

    err = sdv_reference_reset_state(st, stream);
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->cache) cudaFree(st->cache);
    if (st->length) cudaFree(st->length);
    if (st->checkpoint_length) cudaFree(st->checkpoint_length);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const SdvRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    if (workspace_bytes < 128) {
        return cudaErrorInvalidValue;
    }

    SdvReferenceState* st = static_cast<SdvReferenceState*>(state);

    if (!sdv_validate_run_spec(run, &st->spec)) {
        return cudaErrorInvalidValue;
    }

    const SdvInputs* in = static_cast<const SdvInputs*>(inputs_void);
    SdvOutputs* out = static_cast<SdvOutputs*>(outputs_void);

    if (!in->active_seq || !in->draft_value || !in->correction_value ||
        !in->p_target || !in->p_draft || !in->uniform_u32 ||
        !out->accepted_count || !out->new_length ||
        !out->live_cache_sum || !out->live_cache_tail_hash ||
        !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaSuccess;

    if (run->active_count > 0) {
        sdv_ref_step_kernel<<<run->active_count, 1, 0, stream>>>(
            st->spec.B,
            st->spec.max_len,
            run->active_count,
            run->draft_len,
            in->active_seq,
            in->draft_value,
            in->correction_value,
            in->p_target,
            in->p_draft,
            in->uniform_u32,
            st->cache,
            st->length,
            st->checkpoint_length,
            out->accepted_count,
            out->new_length,
            out->live_cache_sum,
            out->live_cache_tail_hash);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    sdv_ref_checksum_kernel<<<1, 1, 0, stream>>>(
        st->spec.B,
        st->spec.max_len,
        st->cache,
        st->length,
        st->checkpoint_length,
        out->state_checksum);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return sdv_reference_reset_state(static_cast<SdvReferenceState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;

    SdvReferenceState* st = static_cast<SdvReferenceState*>(state);
    if (st->cache) cudaFree(st->cache);
    if (st->length) cudaFree(st->length);
    if (st->checkpoint_length) cudaFree(st->checkpoint_length);
    free(st);
}
