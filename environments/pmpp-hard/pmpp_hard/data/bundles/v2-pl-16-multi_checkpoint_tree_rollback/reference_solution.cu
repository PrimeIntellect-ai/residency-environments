// PMPP_CANARY_16_b8002112a0 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: multi_checkpoint_tree_rollback_reference.cu
// Incremental stateful reference.
// ============================================================================

#include "multi_checkpoint_tree_rollback_common.h"

#include <cuda_runtime.h>

#include <stdlib.h>
#include <string.h>

struct MctrReferenceState {
    MctrProblemSpec spec;
    int32_t* cache;       // B * max_len
    int32_t* length;      // B
    int32_t* depth;       // B
    int32_t* ckpt_id;     // B * max_depth
    int32_t* ckpt_len;    // B * max_depth
    uint64_t* ckpt_hash;  // B * max_depth
};

__device__ __forceinline__ uint64_t mctr_fnv_byte_device(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ void mctr_fnv_bytes_device(uint64_t* h, const uint8_t* p, size_t n) {
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = mctr_fnv_byte_device(v, p[i]);
    }
    *h = v;
}

__device__ uint64_t mctr_tail_hash_device(
    const int32_t* cache,
    int max_len,
    int seq,
    int len) {
    int tail_count = len < MCTR_TAIL_VALUES ? len : MCTR_TAIL_VALUES;
    int tail_start = len - tail_count;

    uint64_t h = 1469598103934665603ULL;

    mctr_fnv_bytes_device(&h, reinterpret_cast<const uint8_t*>(&tail_count), sizeof(int32_t));

    if (tail_count > 0) {
        mctr_fnv_bytes_device(
            &h,
            reinterpret_cast<const uint8_t*>(
                &cache[(size_t)seq * (size_t)max_len + (size_t)tail_start]),
            sizeof(int32_t) * (size_t)tail_count);
    }

    return h;
}

__device__ void mctr_append_one_device(
    int32_t* cache,
    int32_t* length,
    int max_len,
    int seq,
    int32_t value) {
    int len = length[seq];
    if (len < max_len) {
        cache[(size_t)seq * (size_t)max_len + (size_t)len] = value;
        length[seq] = len + 1;
    }
}

__global__ void mctr_ref_apply_step_kernel(
    int B,
    int max_len,
    int max_depth,
    int active_count,
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ op_code,
    const int32_t* __restrict__ token_count,
    const int32_t* __restrict__ accept_count,
    const int32_t* __restrict__ checkpoint_id,
    const int32_t* __restrict__ token_values,
    const int32_t* __restrict__ correction_value,
    int32_t* __restrict__ cache,
    int32_t* __restrict__ length,
    int32_t* __restrict__ depth,
    int32_t* __restrict__ ckpt_id,
    int32_t* __restrict__ ckpt_len,
    uint64_t* __restrict__ ckpt_hash) {
    const int row = blockIdx.x;
    if (row >= active_count || threadIdx.x != 0) return;

    const int seq = active_seq[row];
    if (seq < 0 || seq >= B) return;

    const int op = op_code[row];

    int k = token_count[row];
    if (k < 0) k = 0;
    if (k > MCTR_MAX_K) k = MCTR_MAX_K;

    if (op == MCTR_OP_APPEND) {
        for (int i = 0; i < k; ++i) {
            mctr_append_one_device(
                cache,
                length,
                max_len,
                seq,
                token_values[(size_t)row * (size_t)MCTR_MAX_K + (size_t)i]);
        }
        return;
    }

    if (op == MCTR_OP_SAVE_CHECKPOINT) {
        const int d = depth[seq];
        if (d < max_depth) {
            ckpt_id[(size_t)seq * (size_t)max_depth + (size_t)d] = checkpoint_id[row];
            ckpt_len[(size_t)seq * (size_t)max_depth + (size_t)d] = length[seq];
            ckpt_hash[(size_t)seq * (size_t)max_depth + (size_t)d] =
                mctr_tail_hash_device(cache, max_len, seq, length[seq]);
            depth[seq] = d + 1;
        }
        return;
    }

    if (op == MCTR_OP_ROLLBACK_TO) {
        const int target = checkpoint_id[row];
        int found = -1;
        const int d = depth[seq];

        for (int i = d - 1; i >= 0; --i) {
            if (ckpt_id[(size_t)seq * (size_t)max_depth + (size_t)i] == target) {
                found = i;
                break;
            }
        }

        if (found >= 0) {
            length[seq] = ckpt_len[(size_t)seq * (size_t)max_depth + (size_t)found];
            depth[seq] = found + 1;
        }
        return;
    }

    if (op == MCTR_OP_ACCEPT_PREFIX) {
        int a = accept_count[row];
        if (a < 0) a = 0;
        if (a > k) a = k;

        const int old_len = length[seq];

        for (int i = 0; i < a; ++i) {
            mctr_append_one_device(
                cache,
                length,
                max_len,
                seq,
                token_values[(size_t)row * (size_t)MCTR_MAX_K + (size_t)i]);
        }

        if (old_len + a < max_len) {
            mctr_append_one_device(
                cache,
                length,
                max_len,
                seq,
                correction_value[row]);
        }
        return;
    }
}

__global__ void mctr_ref_emit_outputs_kernel(
    int B,
    int max_len,
    int max_depth,
    const int32_t* __restrict__ cache,
    const int32_t* __restrict__ length,
    const int32_t* __restrict__ depth,
    const int32_t* __restrict__ ckpt_len,
    int64_t* __restrict__ live_cache_sum,
    uint64_t* __restrict__ live_cache_tail_hash,
    int32_t* __restrict__ out_length,
    int32_t* __restrict__ num_checkpoints,
    int32_t* __restrict__ top_checkpoint_len) {
    const int seq = blockIdx.x;
    if (seq >= B || threadIdx.x != 0) return;

    const int len = length[seq];
    uint64_t sum_bits = 0;

    for (int i = 0; i < len; ++i) {
        const int32_t v = cache[(size_t)seq * (size_t)max_len + (size_t)i];
        sum_bits += static_cast<uint64_t>(static_cast<int64_t>(v));
    }

    const int d = depth[seq];

    live_cache_sum[seq] = static_cast<int64_t>(sum_bits);
    live_cache_tail_hash[seq] = mctr_tail_hash_device(cache, max_len, seq, len);
    out_length[seq] = len;
    num_checkpoints[seq] = d;
    top_checkpoint_len[seq] =
        d > 0 ? ckpt_len[(size_t)seq * (size_t)max_depth + (size_t)(d - 1)] : -1;
}

__global__ void mctr_ref_checksum_kernel(
    int B,
    int max_len,
    int max_depth,
    const int32_t* __restrict__ cache,
    const int32_t* __restrict__ length,
    const int32_t* __restrict__ depth,
    const int32_t* __restrict__ ckpt_id,
    const int32_t* __restrict__ ckpt_len,
    const uint64_t* __restrict__ ckpt_hash,
    uint64_t* __restrict__ state_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint64_t h = 1469598103934665603ULL;

    for (int seq = 0; seq < B; ++seq) {
        const int len = length[seq];
        const int dep = depth[seq];

        mctr_fnv_bytes_device(&h, reinterpret_cast<const uint8_t*>(&len), sizeof(int32_t));
        mctr_fnv_bytes_device(&h, reinterpret_cast<const uint8_t*>(&dep), sizeof(int32_t));

        for (int d = 0; d < dep; ++d) {
            const size_t idx = (size_t)seq * (size_t)max_depth + (size_t)d;
            mctr_fnv_bytes_device(&h, reinterpret_cast<const uint8_t*>(&ckpt_id[idx]), sizeof(int32_t));
            mctr_fnv_bytes_device(&h, reinterpret_cast<const uint8_t*>(&ckpt_len[idx]), sizeof(int32_t));
            mctr_fnv_bytes_device(&h, reinterpret_cast<const uint8_t*>(&ckpt_hash[idx]), sizeof(uint64_t));
        }

        mctr_fnv_bytes_device(&h, reinterpret_cast<const uint8_t*>(&len), sizeof(int32_t));

        if (len > 0) {
            mctr_fnv_bytes_device(
                &h,
                reinterpret_cast<const uint8_t*>(&cache[(size_t)seq * (size_t)max_len]),
                sizeof(int32_t) * (size_t)len);
        }
    }

    state_checksum[0] = h;
}

static cudaError_t mctr_reference_reset_state(MctrReferenceState* st, cudaStream_t stream) {
    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(
        st->cache,
        0,
        sizeof(int32_t) * (size_t)st->spec.B * (size_t)st->spec.max_len,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->length, 0, sizeof(int32_t) * (size_t)st->spec.B, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->depth, 0, sizeof(int32_t) * (size_t)st->spec.B, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(
        st->ckpt_id,
        0,
        sizeof(int32_t) * (size_t)st->spec.B * (size_t)st->spec.max_depth,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(
        st->ckpt_len,
        0,
        sizeof(int32_t) * (size_t)st->spec.B * (size_t)st->spec.max_depth,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(
        st->ckpt_hash,
        0,
        sizeof(uint64_t) * (size_t)st->spec.B * (size_t)st->spec.max_depth,
        stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" size_t solution_workspace_bytes(const MctrProblemSpec* spec) {
    if (!mctr_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const MctrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!mctr_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    MctrReferenceState* st =
        static_cast<MctrReferenceState*>(malloc(sizeof(MctrReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memset(st, 0, sizeof(MctrReferenceState));
    memcpy(&st->spec, spec, sizeof(MctrProblemSpec));

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(
        reinterpret_cast<void**>(&st->cache),
        sizeof(int32_t) * (size_t)spec->B * (size_t)spec->max_len);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->length), sizeof(int32_t) * (size_t)spec->B);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->depth), sizeof(int32_t) * (size_t)spec->B);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(
        reinterpret_cast<void**>(&st->ckpt_id),
        sizeof(int32_t) * (size_t)spec->B * (size_t)spec->max_depth);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(
        reinterpret_cast<void**>(&st->ckpt_len),
        sizeof(int32_t) * (size_t)spec->B * (size_t)spec->max_depth);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(
        reinterpret_cast<void**>(&st->ckpt_hash),
        sizeof(uint64_t) * (size_t)spec->B * (size_t)spec->max_depth);
    if (err != cudaSuccess) goto fail;

    err = mctr_reference_reset_state(st, stream);
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->cache) cudaFree(st->cache);
    if (st->length) cudaFree(st->length);
    if (st->depth) cudaFree(st->depth);
    if (st->ckpt_id) cudaFree(st->ckpt_id);
    if (st->ckpt_len) cudaFree(st->ckpt_len);
    if (st->ckpt_hash) cudaFree(st->ckpt_hash);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const MctrRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;

    if (!state || !inputs_void || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    MctrReferenceState* st = static_cast<MctrReferenceState*>(state);

    if (!mctr_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const MctrInputs* in = static_cast<const MctrInputs*>(inputs_void);
    MctrOutputs* out = static_cast<MctrOutputs*>(outputs_void);

    if (!out->live_cache_sum || !out->live_cache_tail_hash || !out->length ||
        !out->num_checkpoints || !out->top_checkpoint_len || !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    if (run->active_count > 0) {
        if (!in->active_seq || !in->op_code || !in->token_count || !in->accept_count ||
            !in->checkpoint_id || !in->token_values || !in->correction_value) {
            return cudaErrorInvalidValue;
        }

        mctr_ref_apply_step_kernel<<<run->active_count, 1, 0, stream>>>(
            st->spec.B,
            st->spec.max_len,
            st->spec.max_depth,
            run->active_count,
            in->active_seq,
            in->op_code,
            in->token_count,
            in->accept_count,
            in->checkpoint_id,
            in->token_values,
            in->correction_value,
            st->cache,
            st->length,
            st->depth,
            st->ckpt_id,
            st->ckpt_len,
            st->ckpt_hash);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    mctr_ref_emit_outputs_kernel<<<st->spec.B, 1, 0, stream>>>(
        st->spec.B,
        st->spec.max_len,
        st->spec.max_depth,
        st->cache,
        st->length,
        st->depth,
        st->ckpt_len,
        out->live_cache_sum,
        out->live_cache_tail_hash,
        out->length,
        out->num_checkpoints,
        out->top_checkpoint_len);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    mctr_ref_checksum_kernel<<<1, 1, 0, stream>>>(
        st->spec.B,
        st->spec.max_len,
        st->spec.max_depth,
        st->cache,
        st->length,
        st->depth,
        st->ckpt_id,
        st->ckpt_len,
        st->ckpt_hash,
        out->state_checksum);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return mctr_reference_reset_state(static_cast<MctrReferenceState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;

    MctrReferenceState* st = static_cast<MctrReferenceState*>(state);
    if (st->cache) cudaFree(st->cache);
    if (st->length) cudaFree(st->length);
    if (st->depth) cudaFree(st->depth);
    if (st->ckpt_id) cudaFree(st->ckpt_id);
    if (st->ckpt_len) cudaFree(st->ckpt_len);
    if (st->ckpt_hash) cudaFree(st->ckpt_hash);
    free(st);
}
