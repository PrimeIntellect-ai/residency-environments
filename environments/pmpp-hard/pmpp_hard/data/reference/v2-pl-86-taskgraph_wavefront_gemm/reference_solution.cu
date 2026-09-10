// PMPP_CANARY_86_bea02ece42 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: taskgraph_wavefront_gemm_reference.cu
// Strong reference implementation.
//
// Architecture:
//   - Persistent GPU-resident scheduler: one kernel launch per round. Blocks
//     claim tasks from a global MPMC queue (atomic head), spin (with
//     __nanosleep backoff) until their claimed slot is published, execute the
//     salted tile GEMM, then release-fence and decrement the dependency
//     counters of the right/down neighbors, publishing newly-ready tasks.
//     ACC traffic inside the round uses .cg (L2) loads/stores so
//     producer/consumer visibility does not depend on non-coherent L1.
//   - Tile GEMM: U/V tiles staged to padded shared memory as packed u32,
//     salt XOR applied word-wise, dp4a inner loop, 4x2 register micro-tiles.
//   - Checksum: per-tile digests in parallel, root folded via the exact
//     affine-FNV chunk composition (no serial byte fold of the data).
// ============================================================================

#include "taskgraph_wavefront_gemm_common.h"

#include <stdlib.h>
#include <string.h>

#define TWG_QUEUE_EMPTY 0xFFFFFFFFu
#define TWG_ROOT_CHUNK 256
#define TWG_K4MAX (TWG_MAX_K / 4)

struct TwgReferenceState {
    TwgProblemSpec spec;
    uint32_t* acc;        // persistent [max_R*32 * max_C*32]
    uint32_t* round_ctr;  // persistent [1]
    int max_resident_blocks;
};

struct TwgWorkspaceLayout {
    uint32_t* dep;      // [max_R * max_C]
    uint32_t* queue;    // [max_R * max_C]
    uint32_t* head;     // [1]
    uint32_t* tail;     // [1]
    uint64_t* digests;  // [max_R * max_C]
    uint64_t* root_T;   // [nroot * 256]
    uint64_t* root_A;   // [nroot]
    size_t required_bytes;
};

static TwgWorkspaceLayout twg_reference_make_layout(
    void* workspace,
    const TwgProblemSpec* spec) {
    TwgWorkspaceLayout L{};
    char* base = static_cast<char*>(workspace);
    const size_t maxTasks = (size_t)spec->max_R * (size_t)spec->max_C;
    const size_t nroot = (maxTasks + TWG_ROOT_CHUNK - 1) / TWG_ROOT_CHUNK;

    size_t off = 0;
#define TWG_WS_FIELD(field, type, count)                                    \
    off = twg_align_up_size(off, 128);                                      \
    L.field = reinterpret_cast<type*>(base + off);                          \
    off += sizeof(type) * (count);

    TWG_WS_FIELD(dep, uint32_t, maxTasks)
    TWG_WS_FIELD(queue, uint32_t, maxTasks)
    TWG_WS_FIELD(head, uint32_t, 1)
    TWG_WS_FIELD(tail, uint32_t, 1)
    TWG_WS_FIELD(digests, uint64_t, maxTasks)
    TWG_WS_FIELD(root_T, uint64_t, nroot * 256)
    TWG_WS_FIELD(root_A, uint64_t, nroot)
#undef TWG_WS_FIELD

    off = twg_align_up_size(off, 128);
    L.required_bytes = off;
    return L;
}

// ---------------------------------------------------------------------------
// Scheduler bookkeeping.
// ---------------------------------------------------------------------------

__global__ void twg_ref_sched_init_kernel(
    int R,
    int C,
    uint32_t* __restrict__ dep,
    uint32_t* __restrict__ queue,
    uint32_t* __restrict__ head,
    uint32_t* __restrict__ tail) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = R * C;
    if (t >= total) return;
    const int i = t / C;
    const int j = t - i * C;
    dep[t] = (uint32_t)((i > 0 ? 1 : 0) + (j > 0 ? 1 : 0));
    queue[t] = t == 0 ? 0u : TWG_QUEUE_EMPTY;  // task (0,0) pre-published
    if (t == 0) {
        *head = 0u;
        *tail = 1u;
    }
}

// ---------------------------------------------------------------------------
// Persistent wavefront executor.
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(128)
twg_ref_persistent_kernel(
    int R,
    int C,
    int K,
    uint32_t P1,
    uint32_t P2,
    const int8_t* __restrict__ U,
    const int8_t* __restrict__ V,
    uint32_t* __restrict__ acc,
    const uint32_t* __restrict__ round_ctr,
    uint32_t* __restrict__ dep,
    uint32_t* __restrict__ queue,
    uint32_t* __restrict__ head,
    uint32_t* __restrict__ tail) {
    __shared__ uint32_t Us[32][TWG_K4MAX + 1];
    __shared__ uint32_t Vs[32][TWG_K4MAX + 1];
    __shared__ uint32_t rv[32];
    __shared__ uint32_t cv[32];
    __shared__ uint32_t s_task;
    __shared__ uint32_t s_salt;

    const int total = R * C;
    const int K4 = K / 4;
    const int W = C * 32;
    const uint32_t r = __ldcg(round_ctr);
    const uint32_t* U_u = reinterpret_cast<const uint32_t*>(U);
    const uint32_t* V_u = reinterpret_cast<const uint32_t*>(V);

    for (;;) {
        // Claim a queue slot; wait for it to be published.
        if (threadIdx.x == 0) {
            const uint32_t my = atomicAdd(head, 1u);
            uint32_t task = TWG_QUEUE_EMPTY;
            if (my < (uint32_t)total) {
                while ((task = atomicAdd(&queue[my], 0u)) == TWG_QUEUE_EMPTY) {
                    __nanosleep(64);
                }
            }
            s_task = task;
        }
        __syncthreads();
        const uint32_t task = s_task;
        if (task == TWG_QUEUE_EMPTY) return;
        __syncthreads();

        const int i = (int)task / C;
        const int j = (int)task - i * C;

        // Neighbor borders (updated this round) -> shared; L2-scoped loads.
        if (threadIdx.x < 32) {
            const int b = threadIdx.x;
            rv[b] = i > 0
                ? __ldcg(&acc[(size_t)((i - 1) * 32 + 31) * W + (size_t)j * 32 + b])
                : 0u;
        } else if (threadIdx.x < 64) {
            const int a = threadIdx.x - 32;
            cv[a] = j > 0
                ? __ldcg(&acc[(size_t)(i * 32 + a) * W + (size_t)(j - 1) * 32 + 31])
                : 0u;
        }

        // Stage U and V tiles as packed u32 words.
        for (int idx = threadIdx.x; idx < 32 * K4; idx += 128) {
            const int row = idx / K4;
            const int kk = idx - row * K4;
            Us[row][kk] = U_u[((size_t)(i * 32 + row) * K) / 4 + kk];
            Vs[row][kk] = V_u[((size_t)(j * 32 + row) * K) / 4 + kk];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            uint32_t sum_row = 0;
            uint32_t sum_col = 0;
            for (int q = 0; q < 32; ++q) {
                sum_row += rv[q];
                sum_col += cv[q];
            }
            s_salt = TWG_SALT_ROW * sum_row + TWG_SALT_COL * sum_col +
                     TWG_SALT_ROUND * r + ((uint32_t)i << 16) + (uint32_t)j;
        }
        __syncthreads();
        const uint32_t s = s_salt;

        // 4x2 register micro-tile per thread.
        const int a0 = (threadIdx.x >> 4) * 4;
        const int b0 = (threadIdx.x & 15) * 2;

        int32_t g00 = 0, g01 = 0, g10 = 0, g11 = 0;
        int32_t g20 = 0, g21 = 0, g30 = 0, g31 = 0;

        for (int k4 = 0; k4 < K4; ++k4) {
            const uint32_t u0 = Us[a0 + 0][k4] ^ s;
            const uint32_t u1 = Us[a0 + 1][k4] ^ s;
            const uint32_t u2 = Us[a0 + 2][k4] ^ s;
            const uint32_t u3 = Us[a0 + 3][k4] ^ s;
            const uint32_t v0 = Vs[b0 + 0][k4];
            const uint32_t v1 = Vs[b0 + 1][k4];
            g00 = __dp4a((int)u0, (int)v0, g00);
            g01 = __dp4a((int)u0, (int)v1, g01);
            g10 = __dp4a((int)u1, (int)v0, g10);
            g11 = __dp4a((int)u1, (int)v1, g11);
            g20 = __dp4a((int)u2, (int)v0, g20);
            g21 = __dp4a((int)u2, (int)v1, g21);
            g30 = __dp4a((int)u3, (int)v0, g30);
            g31 = __dp4a((int)u3, (int)v1, g31);
        }

        int32_t g[4][2] = {{g00, g01}, {g10, g11}, {g20, g21}, {g30, g31}};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int a = a0 + q;
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int b = b0 + p;
                uint32_t* cell =
                    &acc[(size_t)(i * 32 + a) * W + (size_t)j * 32 + b];
                const uint32_t updated = __ldcg(cell) + (uint32_t)g[q][p] +
                                         P1 * rv[b] + P2 * cv[a];
                __stcg(cell, updated);
            }
        }

        // Release: make this tile visible, then wake dependents.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            if (i + 1 < R) {
                const int t2 = (i + 1) * C + j;
                if (atomicSub(&dep[t2], 1u) == 1u) {
                    const uint32_t pos = atomicAdd(tail, 1u);
                    atomicExch(&queue[pos], (uint32_t)t2);
                }
            }
            if (j + 1 < C) {
                const int t2 = i * C + (j + 1);
                if (atomicSub(&dep[t2], 1u) == 1u) {
                    const uint32_t pos = atomicAdd(tail, 1u);
                    atomicExch(&queue[pos], (uint32_t)t2);
                }
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Round bookkeeping + outputs.
// ---------------------------------------------------------------------------

__global__ void twg_ref_round_kernel(
    uint32_t* __restrict__ round_ctr,
    int32_t* __restrict__ round_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint32_t nr = *round_ctr + 1u;
    *round_ctr = nr;
    round_out[0] = (int32_t)nr;
}

__global__ void twg_ref_border_kernel(
    int R,
    int C,
    const uint32_t* __restrict__ acc,
    uint32_t* __restrict__ border_row,
    uint32_t* __restrict__ border_col) {
    const int W = C * 32;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < W) {
        border_row[idx] = acc[(size_t)(R * 32 - 1) * W + idx];
    }
    const int y = idx - W;
    if (y >= 0 && y < R * 32) {
        border_col[y] = acc[(size_t)y * W + (W - 1)];
    }
}

__device__ __forceinline__ uint64_t twg_fnv_byte_dev(uint64_t h, uint8_t b) {
    return (h ^ (uint64_t)b) * TWG_FNV_PRIME;
}

__global__ void twg_ref_digest_kernel(
    int R,
    int C,
    const uint32_t* __restrict__ acc,
    uint64_t* __restrict__ digests) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= R * C) return;
    const int i = t / C;
    const int j = t - i * C;
    const int W = C * 32;

    uint64_t h = TWG_FNV_BASIS;
    for (int a = 0; a < 32; ++a) {
        const uint32_t* row = acc + (size_t)(i * 32 + a) * W + (size_t)j * 32;
        for (int b = 0; b < 32; ++b) {
            const uint32_t v = row[b];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                h = twg_fnv_byte_dev(h, (uint8_t)(v >> (8 * m)));
            }
        }
    }
    digests[t] = h;
}

__global__ void twg_ref_root_table_kernel(
    int C,
    const uint64_t* __restrict__ digests,
    uint64_t* __restrict__ root_T,
    uint64_t* __restrict__ root_A) {
    const int chunk = blockIdx.x;
    const int v = threadIdx.x;

    const int begin = chunk * TWG_ROOT_CHUNK;
    const int end = min(begin + TWG_ROOT_CHUNK, C);

    uint64_t h = (uint64_t)v;
    uint64_t m = 1ULL;
    for (int i = begin; i < end; ++i) {
        const uint64_t d = digests[i];
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            h = twg_fnv_byte_dev(h, (uint8_t)(d >> (8 * b)));
            m *= TWG_FNV_PRIME;
        }
    }
    root_T[(size_t)chunk * 256 + v] = h;
    if (v == 0) root_A[chunk] = m;
}

__global__ void twg_ref_root_combine_kernel(
    int nchunks,
    const uint64_t* __restrict__ root_T,
    const uint64_t* __restrict__ root_A,
    uint64_t* __restrict__ state_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t h = TWG_FNV_BASIS;
    for (int c = 0; c < nchunks; ++c) {
        const uint64_t A = root_A[c];
        const uint64_t T = root_T[(size_t)c * 256 + (h & 0xFFULL)];
        h = A * (h & ~0xFFULL) + T;
    }
    state_checksum[0] = h;
}

__global__ void twg_ref_dump_kernel(
    int total_elems,
    const uint32_t* __restrict__ acc,
    uint32_t* __restrict__ acc_dump) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) {
        acc_dump[idx] = acc[idx];
    }
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

extern "C" size_t solution_workspace_bytes(const TwgProblemSpec* spec) {
    if (!twg_validate_problem_spec(spec)) return 0;
    TwgWorkspaceLayout L = twg_reference_make_layout(nullptr, spec);
    return L.required_bytes;
}

static cudaError_t twg_reference_reset_state(
    TwgReferenceState* st,
    cudaStream_t stream) {
    const size_t elems = (size_t)st->spec.max_R * 32 *
                         (size_t)st->spec.max_C * 32;
    cudaError_t err =
        cudaMemsetAsync(st->acc, 0, elems * sizeof(uint32_t), stream);
    if (err != cudaSuccess) return err;
    return cudaMemsetAsync(st->round_ctr, 0, sizeof(uint32_t), stream);
}

extern "C" cudaError_t solution_init(
    const TwgProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!twg_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    TwgReferenceState* st = static_cast<TwgReferenceState*>(
        malloc(sizeof(TwgReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(TwgReferenceState));
    memcpy(&st->spec, spec, sizeof(TwgProblemSpec));

    const size_t elems = (size_t)spec->max_R * 32 * (size_t)spec->max_C * 32;

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&st->acc),
                                 elems * sizeof(uint32_t));
    if (err != cudaSuccess) {
        free(st);
        return err;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&st->round_ctr),
                     sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(st->acc);
        free(st);
        return err;
    }

    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int bpsm = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bpsm, twg_ref_persistent_kernel, 128, 0);
    if (err != cudaSuccess || bpsm < 1) bpsm = 1;
    st->max_resident_blocks = sm_count > 0 ? sm_count * bpsm : bpsm;

    err = twg_reference_reset_state(st, stream);
    if (err != cudaSuccess) {
        cudaFree(st->acc);
        cudaFree(st->round_ctr);
        free(st);
        return err;
    }

    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const TwgRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !twg_validate_run_spec(run) || !inputs_void ||
        !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    TwgReferenceState* st = static_cast<TwgReferenceState*>(state);
    const TwgInputs* in = static_cast<const TwgInputs*>(inputs_void);
    TwgOutputs* out = static_cast<TwgOutputs*>(outputs_void);

    if (run->R > st->spec.max_R || run->C > st->spec.max_C ||
        run->K > st->spec.max_K) {
        return cudaErrorInvalidValue;
    }
    if (!in->U || !in->V || !out->round_out || !out->border_row ||
        !out->border_col || !out->state_checksum ||
        (run->dump == 1 && !out->acc_dump)) {
        return cudaErrorInvalidValue;
    }

    TwgWorkspaceLayout L = twg_reference_make_layout(workspace, &st->spec);
    if (workspace_bytes < L.required_bytes) return cudaErrorInvalidValue;

    const int R = run->R;
    const int C = run->C;
    const int total = R * C;

    cudaError_t err = cudaSuccess;

    twg_ref_sched_init_kernel<<<twg_ceil_div_int(total, 128), 128, 0,
                                stream>>>(
        R, C, L.dep, L.queue, L.head, L.tail);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    const int nblocks = total < st->max_resident_blocks
        ? total
        : st->max_resident_blocks;
    twg_ref_persistent_kernel<<<nblocks, 128, 0, stream>>>(
        R, C, run->K, (uint32_t)run->P1, (uint32_t)run->P2, in->U, in->V,
        st->acc, st->round_ctr, L.dep, L.queue, L.head, L.tail);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    twg_ref_round_kernel<<<1, 1, 0, stream>>>(st->round_ctr, out->round_out);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    twg_ref_border_kernel<<<twg_ceil_div_int(C * 32 + R * 32, 128), 128, 0,
                            stream>>>(
        R, C, st->acc, out->border_row, out->border_col);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    twg_ref_digest_kernel<<<twg_ceil_div_int(total, 128), 128, 0, stream>>>(
        R, C, st->acc, L.digests);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    const int nroot = twg_ceil_div_int(total, TWG_ROOT_CHUNK);
    twg_ref_root_table_kernel<<<nroot, 256, 0, stream>>>(
        total, L.digests, L.root_T, L.root_A);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    twg_ref_root_combine_kernel<<<1, 1, 0, stream>>>(
        nroot, L.root_T, L.root_A, out->state_checksum);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    if (run->dump == 1) {
        const int elems = R * 32 * C * 32;
        twg_ref_dump_kernel<<<twg_ceil_div_int(elems, 256), 256, 0, stream>>>(
            elems, st->acc, out->acc_dump);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return twg_reference_reset_state(
        static_cast<TwgReferenceState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    TwgReferenceState* st = static_cast<TwgReferenceState*>(state);
    if (st->acc) cudaFree(st->acc);
    if (st->round_ctr) cudaFree(st->round_ctr);
    free(st);
}
