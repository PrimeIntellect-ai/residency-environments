// PMPP_CANARY_27_efd38af2e2 -- held-out canary; MUST NOT appear in any submission
// file: streaming_dedup_window_reference.cu

#include "streaming_dedup_window_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Fast, byte-identical GPU reference.
//
// The oracle is a single-thread <<<1,1>>> kernel whose runtime is ~92% dominated
// by the per-event "expire old keys" scan over the whole key_space, repeated for
// every ingested event. The streaming state machine is genuinely sequential
// across events (dedup / LRU-evict / finalize order all depend on event order),
// so we keep the event loop sequential but cooperatively parallelize the three
// O(key_space) hotspots inside a single <<<1, SDW_BLK>>> block:
//   * per-event expiry: a block scan finds the keys that expire, an ordered
//     (ascending-key) compaction gathers them, then thread 0 finalizes exactly
//     those few keys in the same order the serial oracle would -> identical
//     evicted-hash / num_evicted / mutations.
//   * capacity LRU eviction: a block argmin (min lru_stamp, tie -> min key)
//     reproduces the serial find-LRU result exactly.
//   * final live_agg_sum: a block reduction; int64 two's-complement wrapping
//     addition is associative+commutative, so any order gives identical bytes.
// state_checksum stays serial on thread 0 (small; ~7% of oracle time).

#define SDW_BLK 256

struct SdwReferenceState {
    SdwProblemSpec spec;

    int32_t* active;
    int32_t* last_pos;
    int32_t* lru_stamp;
    int64_t* agg;

    int32_t* current_pos;
    int32_t* active_count_state;

    int32_t* cand;  // scratch: expiring-key candidate list (size key_space)
};

__device__ __forceinline__ uint64_t sdw_fnv_byte_device(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ void sdw_fnv_bytes_device(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = sdw_fnv_byte_device(v, p[i]);
    }
    *h = v;
}

__device__ __forceinline__ int64_t sdw_add_i64_i32_wrap_device(int64_t a, int32_t b) {
    uint64_t ua = static_cast<uint64_t>(a);
    ua += static_cast<uint64_t>(static_cast<int64_t>(b));
    return static_cast<int64_t>(ua);
}

// Finalize one active key: fold (key, agg, last_pos) into evicted_hash in the
// exact byte order the oracle uses, bump num_evicted, then clear the key and
// decrement active_count. Values are read before clearing. Called by thread 0.
__device__ void sdw_finalize_inline(
    int key,
    uint64_t* evicted_hash,
    int* num_evicted,
    int* active_count,
    int32_t* __restrict__ active,
    int32_t* __restrict__ last_pos,
    int32_t* __restrict__ lru_stamp,
    int64_t* __restrict__ agg) {
    if (active[key] == 0) return;

    uint64_t h = *evicted_hash;
    int32_t kk = key;
    sdw_fnv_bytes_device(&h, &kk, sizeof(int32_t));
    sdw_fnv_bytes_device(&h, &agg[key], sizeof(int64_t));
    sdw_fnv_bytes_device(&h, &last_pos[key], sizeof(int32_t));
    *evicted_hash = h;
    *num_evicted += 1;

    active[key] = 0;
    last_pos[key] = 0;
    lru_stamp[key] = 0;
    agg[key] = 0;
    *active_count -= 1;
}

// Cooperative expiry for one event position `cur`. All threads participate.
// Blocked ranges [lo,hi) preserve ascending-key order across the block; an
// exclusive prefix sum of per-thread candidate counts places every thread's
// candidates before the next thread's, so `cand` is globally ascending. Thread
// 0 then finalizes exactly those keys in order.
__device__ void sdw_do_expire(
    int key_space,
    int window_size,
    int cur,
    int lo,
    int hi,
    int nthreads,
    int tid,
    int32_t* __restrict__ active,
    int32_t* __restrict__ last_pos,
    int32_t* __restrict__ lru_stamp,
    int64_t* __restrict__ agg,
    int32_t* __restrict__ cand,
    int* s_scan,
    int* s_off,
    int* s_total,
    uint64_t* s_evicted_hash,
    int* s_num_evicted,
    int* s_active_count) {
    int cnt = 0;
    for (int k = lo; k < hi; ++k) {
        if (active[k] != 0 && cur - last_pos[k] > window_size) ++cnt;
    }
    s_scan[tid] = cnt;
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < nthreads; ++i) {
            int c = s_scan[i];
            s_off[i] = acc;
            acc += c;
        }
        *s_total = acc;
    }
    __syncthreads();

    int w = s_off[tid];
    for (int k = lo; k < hi; ++k) {
        if (active[k] != 0 && cur - last_pos[k] > window_size) cand[w++] = k;
    }
    __syncthreads();

    if (tid == 0) {
        int tot = *s_total;
        for (int i = 0; i < tot; ++i) {
            sdw_finalize_inline(
                cand[i],
                s_evicted_hash,
                s_num_evicted,
                s_active_count,
                active,
                last_pos,
                lru_stamp,
                agg);
        }
    }
    __syncthreads();
}

__device__ uint64_t sdw_state_checksum_device(
    int key_space,
    int capacity,
    int window_size,
    int current_pos,
    int active_count,
    const int32_t* __restrict__ active,
    const int32_t* __restrict__ last_pos,
    const int32_t* __restrict__ lru_stamp,
    const int64_t* __restrict__ agg) {
    uint64_t h = 1469598103934665603ULL;

    sdw_fnv_bytes_device(&h, &key_space, sizeof(int32_t));
    sdw_fnv_bytes_device(&h, &capacity, sizeof(int32_t));
    sdw_fnv_bytes_device(&h, &window_size, sizeof(int32_t));
    sdw_fnv_bytes_device(&h, &current_pos, sizeof(int32_t));
    sdw_fnv_bytes_device(&h, &active_count, sizeof(int32_t));

    for (int key = 0; key < key_space; ++key) {
        sdw_fnv_bytes_device(&h, &active[key], sizeof(int32_t));
        sdw_fnv_bytes_device(&h, &last_pos[key], sizeof(int32_t));
        sdw_fnv_bytes_device(&h, &lru_stamp[key], sizeof(int32_t));
        sdw_fnv_bytes_device(&h, &agg[key], sizeof(int64_t));
    }

    return h;
}

__global__ void sdw_reference_step_kernel(
    int key_space,
    int capacity,
    int window_size,
    int batch_size,
    const int32_t* __restrict__ in_key,
    const int32_t* __restrict__ in_value,
    int32_t* __restrict__ active,
    int32_t* __restrict__ last_pos,
    int32_t* __restrict__ lru_stamp,
    int64_t* __restrict__ agg,
    int32_t* __restrict__ cand,
    int32_t* __restrict__ current_pos_ptr,
    int32_t* __restrict__ active_count_state,
    int32_t* __restrict__ out_active_count,
    int32_t* __restrict__ out_num_new,
    int32_t* __restrict__ out_num_dup,
    int32_t* __restrict__ out_num_evicted,
    uint64_t* __restrict__ out_evicted_hash,
    int64_t* __restrict__ out_live_agg_sum,
    uint64_t* __restrict__ out_state_checksum) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ int s_current_pos;
    __shared__ int s_active_count;
    __shared__ int s_num_new;
    __shared__ int s_num_dup;
    __shared__ int s_num_evicted;
    __shared__ uint64_t s_evicted_hash;

    __shared__ int s_scan[SDW_BLK];
    __shared__ int s_off[SDW_BLK];
    __shared__ int s_total;

    __shared__ int s_rk[SDW_BLK];
    __shared__ int s_rs[SDW_BLK];

    __shared__ uint64_t s_sum[SDW_BLK];

    __shared__ int s_key;
    __shared__ int s_val;
    __shared__ int s_valid;
    __shared__ int s_active_key;
    __shared__ int s_need_lru;
    __shared__ int s_lru_key;

    // Blocked, contiguous key range per thread (ascending order across threads).
    int per = (key_space + nthreads - 1) / nthreads;
    int lo = tid * per;
    int hi = lo + per;
    if (lo > key_space) lo = key_space;
    if (hi > key_space) hi = key_space;

    if (tid == 0) {
        s_current_pos = current_pos_ptr[0];
        s_active_count = active_count_state[0];
        s_num_new = 0;
        s_num_dup = 0;
        s_num_evicted = 0;
        s_evicted_hash = 1469598103934665603ULL;
    }
    __syncthreads();

    for (int r = 0; r < batch_size; ++r) {
        if (tid == 0) s_current_pos += 1;
        __syncthreads();
        int cur = s_current_pos;

        sdw_do_expire(
            key_space, window_size, cur, lo, hi, nthreads, tid,
            active, last_pos, lru_stamp, agg, cand,
            s_scan, s_off, &s_total,
            &s_evicted_hash, &s_num_evicted, &s_active_count);

        if (tid == 0) {
            int key = in_key[r];
            int val = in_value[r];
            s_key = key;
            s_val = val;
            int valid = (key >= 0 && key < key_space) ? 1 : 0;
            s_valid = valid;
            s_active_key = valid ? active[key] : 0;
            s_need_lru = (valid && s_active_key == 0 && s_active_count >= capacity) ? 1 : 0;
        }
        __syncthreads();

        if (s_need_lru) {
            // Block argmin over active keys: min lru_stamp, ties -> min key.
            int bk = -1;
            int bs = 2147483647;
            for (int k = tid; k < key_space; k += nthreads) {
                if (active[k] == 0) continue;
                int st = lru_stamp[k];
                if (bk < 0 || st < bs || (st == bs && k < bk)) {
                    bk = k;
                    bs = st;
                }
            }
            s_rk[tid] = bk;
            s_rs[tid] = bs;
            __syncthreads();
            for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    int ak = s_rk[tid];
                    int as = s_rs[tid];
                    int b2k = s_rk[tid + stride];
                    int b2s = s_rs[tid + stride];
                    int wk, ws;
                    if (ak < 0) {
                        wk = b2k; ws = b2s;
                    } else if (b2k < 0) {
                        wk = ak; ws = as;
                    } else if (b2s < as || (b2s == as && b2k < ak)) {
                        wk = b2k; ws = b2s;
                    } else {
                        wk = ak; ws = as;
                    }
                    s_rk[tid] = wk;
                    s_rs[tid] = ws;
                }
                __syncthreads();
            }
            if (tid == 0) s_lru_key = s_rk[0];
            __syncthreads();
        }

        if (tid == 0) {
            if (s_valid) {
                int key = s_key;
                int val = s_val;
                if (s_active_key != 0) {
                    agg[key] = sdw_add_i64_i32_wrap_device(agg[key], val);
                    last_pos[key] = cur;
                    lru_stamp[key] = cur;
                    s_num_dup += 1;
                } else {
                    if (s_active_count >= capacity) {
                        int lk = s_lru_key;
                        if (lk >= 0) {
                            sdw_finalize_inline(
                                lk, &s_evicted_hash, &s_num_evicted, &s_active_count,
                                active, last_pos, lru_stamp, agg);
                        }
                    }
                    if (s_active_count < capacity) {
                        active[key] = 1;
                        last_pos[key] = cur;
                        lru_stamp[key] = cur;
                        agg[key] = static_cast<int64_t>(val);
                        s_active_count += 1;
                        s_num_new += 1;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Final expiry at the last position.
    {
        int cur = s_current_pos;
        sdw_do_expire(
            key_space, window_size, cur, lo, hi, nthreads, tid,
            active, last_pos, lru_stamp, agg, cand,
            s_scan, s_off, &s_total,
            &s_evicted_hash, &s_num_evicted, &s_active_count);
    }

    // live_agg_sum: block reduction (wrapping int64 add is assoc+comm).
    uint64_t ls = 0;
    for (int k = tid; k < key_space; k += nthreads) {
        if (active[k] != 0) ls += static_cast<uint64_t>(agg[k]);
    }
    s_sum[tid] = ls;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_sum[tid] += s_sum[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        current_pos_ptr[0] = s_current_pos;
        active_count_state[0] = s_active_count;

        out_active_count[0] = s_active_count;
        out_num_new[0] = s_num_new;
        out_num_dup[0] = s_num_dup;
        out_num_evicted[0] = s_num_evicted;
        out_evicted_hash[0] = s_evicted_hash;
        out_live_agg_sum[0] = static_cast<int64_t>(s_sum[0]);
        out_state_checksum[0] = sdw_state_checksum_device(
            key_space,
            capacity,
            window_size,
            s_current_pos,
            s_active_count,
            active,
            last_pos,
            lru_stamp,
            agg);
    }
}

static cudaError_t sdw_reference_reset_state(SdwReferenceState* st, cudaStream_t stream) {
    cudaError_t err = cudaSuccess;

    err = cudaMemsetAsync(st->active, 0, sizeof(int32_t) * (size_t)st->spec.key_space, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->last_pos, 0, sizeof(int32_t) * (size_t)st->spec.key_space, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->lru_stamp, 0, sizeof(int32_t) * (size_t)st->spec.key_space, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->agg, 0, sizeof(int64_t) * (size_t)st->spec.key_space, stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->current_pos, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->active_count_state, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" size_t solution_workspace_bytes(const SdwProblemSpec* spec) {
    if (!sdw_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const SdwProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!sdw_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    SdwReferenceState* st =
        static_cast<SdwReferenceState*>(malloc(sizeof(SdwReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memset(st, 0, sizeof(SdwReferenceState));
    memcpy(&st->spec, spec, sizeof(SdwProblemSpec));

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(reinterpret_cast<void**>(&st->active), sizeof(int32_t) * (size_t)spec->key_space);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->last_pos), sizeof(int32_t) * (size_t)spec->key_space);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->lru_stamp), sizeof(int32_t) * (size_t)spec->key_space);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->agg), sizeof(int64_t) * (size_t)spec->key_space);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->current_pos), sizeof(int32_t));
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->active_count_state), sizeof(int32_t));
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->cand), sizeof(int32_t) * (size_t)spec->key_space);
    if (err != cudaSuccess) goto fail;

    err = sdw_reference_reset_state(st, stream);
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->active) cudaFree(st->active);
    if (st->last_pos) cudaFree(st->last_pos);
    if (st->lru_stamp) cudaFree(st->lru_stamp);
    if (st->agg) cudaFree(st->agg);
    if (st->current_pos) cudaFree(st->current_pos);
    if (st->active_count_state) cudaFree(st->active_count_state);
    if (st->cand) cudaFree(st->cand);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const SdwRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;

    if (!state || !inputs_void || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    SdwReferenceState* st = static_cast<SdwReferenceState*>(state);
    if (!sdw_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const SdwInputs* in = static_cast<const SdwInputs*>(inputs_void);
    SdwOutputs* out = static_cast<SdwOutputs*>(outputs_void);

    if (run->batch_size > 0 && (!in->key || !in->value)) {
        return cudaErrorInvalidValue;
    }

    if (!out->active_count || !out->num_new || !out->num_dup ||
        !out->num_evicted || !out->evicted_key_checksum ||
        !out->live_agg_sum || !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    sdw_reference_step_kernel<<<1, SDW_BLK, 0, stream>>>(
        st->spec.key_space,
        st->spec.capacity,
        st->spec.window_size,
        run->batch_size,
        in->key,
        in->value,
        st->active,
        st->last_pos,
        st->lru_stamp,
        st->agg,
        st->cand,
        st->current_pos,
        st->active_count_state,
        out->active_count,
        out->num_new,
        out->num_dup,
        out->num_evicted,
        out->evicted_key_checksum,
        out->live_agg_sum,
        out->state_checksum);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return sdw_reference_reset_state(static_cast<SdwReferenceState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;

    SdwReferenceState* st = static_cast<SdwReferenceState*>(state);
    if (st->active) cudaFree(st->active);
    if (st->last_pos) cudaFree(st->last_pos);
    if (st->lru_stamp) cudaFree(st->lru_stamp);
    if (st->agg) cudaFree(st->agg);
    if (st->current_pos) cudaFree(st->current_pos);
    if (st->active_count_state) cudaFree(st->active_count_state);
    if (st->cand) cudaFree(st->cand);
    free(st);
}
