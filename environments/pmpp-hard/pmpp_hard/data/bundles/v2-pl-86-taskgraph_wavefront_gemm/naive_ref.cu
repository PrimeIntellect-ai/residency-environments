// ============================================================================
// file: naive_ref.cu
// Clean, correct, parallel-but-unoptimized implementation used to calibrate
// the perf gate. One kernel launch per wavefront anti-diagonal (dependency
// order enforced by launch boundaries), scalar byte arithmetic (no dp4a, no
// operand staging), one thread per tile for digests, serial root fold.
// ============================================================================

#include "taskgraph_wavefront_gemm_common.h"

#include <stdlib.h>
#include <string.h>

struct TwgNaiveState {
    TwgProblemSpec spec;
    uint32_t* acc;        // persistent [max_R*32 * max_C*32]
    uint32_t* round_ctr;  // persistent [1]
};

struct TwgNaiveWorkspaceLayout {
    uint64_t* digests;  // [max_R * max_C]
    size_t required_bytes;
};

static TwgNaiveWorkspaceLayout twg_naive_make_layout(
    void* workspace,
    const TwgProblemSpec* spec) {
    TwgNaiveWorkspaceLayout L{};
    char* base = static_cast<char*>(workspace);
    const size_t maxTasks = (size_t)spec->max_R * (size_t)spec->max_C;

    size_t off = 0;
    off = twg_align_up_size(off, 128);
    L.digests = reinterpret_cast<uint64_t*>(base + off);
    off += sizeof(uint64_t) * maxTasks;

    off = twg_align_up_size(off, 128);
    L.required_bytes = off;
    return L;
}

// One block per tile of the current anti-diagonal d (i + j == d). The
// previous diagonal's tiles are complete because they were written by an
// earlier kernel launch on the same stream.
__global__ void twg_naive_diag_kernel(
    int R,
    int C,
    int K,
    int d,
    uint32_t P1,
    uint32_t P2,
    const int8_t* __restrict__ U,
    const int8_t* __restrict__ V,
    uint32_t* __restrict__ acc,
    const uint32_t* __restrict__ round_ctr) {
    const int i_lo = d - (C - 1) > 0 ? d - (C - 1) : 0;
    const int i = i_lo + blockIdx.x;
    const int j = d - i;
    if (i >= R || j < 0 || j >= C) return;

    const int W = C * 32;

    __shared__ uint32_t rv[32];
    __shared__ uint32_t cv[32];
    __shared__ uint32_t s_salt;

    if (threadIdx.x < 32) {
        const int b = threadIdx.x;
        rv[b] = i > 0
            ? acc[(size_t)((i - 1) * 32 + 31) * W + (size_t)j * 32 + b]
            : 0u;
    } else if (threadIdx.x < 64) {
        const int a = threadIdx.x - 32;
        cv[a] = j > 0
            ? acc[(size_t)(i * 32 + a) * W + (size_t)(j - 1) * 32 + 31]
            : 0u;
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
                 TWG_SALT_ROUND * (*round_ctr) +
                 ((uint32_t)i << 16) + (uint32_t)j;
    }
    __syncthreads();

    const uint32_t s = s_salt;
    uint8_t sb[4];
    sb[0] = (uint8_t)(s & 0xFF);
    sb[1] = (uint8_t)((s >> 8) & 0xFF);
    sb[2] = (uint8_t)((s >> 16) & 0xFF);
    sb[3] = (uint8_t)((s >> 24) & 0xFF);

    for (int c = threadIdx.x; c < 32 * 32; c += blockDim.x) {
        const int a = c >> 5;
        const int b = c & 31;
        const int8_t* urow = U + (size_t)(i * 32 + a) * K;
        const int8_t* vrow = V + (size_t)(j * 32 + b) * K;
        int32_t g = 0;
        for (int k = 0; k < K; ++k) {
            const int8_t mixed = (int8_t)((uint8_t)urow[k] ^ sb[k & 3]);
            g += (int32_t)mixed * (int32_t)vrow[k];
        }
        uint32_t* cell = &acc[(size_t)(i * 32 + a) * W + (size_t)j * 32 + b];
        *cell = *cell + (uint32_t)g + P1 * rv[b] + P2 * cv[a];
    }
}

__global__ void twg_naive_round_kernel(
    uint32_t* __restrict__ round_ctr,
    int32_t* __restrict__ round_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint32_t nr = *round_ctr + 1u;
    *round_ctr = nr;
    round_out[0] = (int32_t)nr;
}

__global__ void twg_naive_border_kernel(
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

__device__ __forceinline__ uint64_t twg_naive_fnv_byte(uint64_t h, uint8_t b) {
    return (h ^ (uint64_t)b) * TWG_FNV_PRIME;
}

__global__ void twg_naive_digest_kernel(
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
            for (int m = 0; m < 4; ++m) {
                h = twg_naive_fnv_byte(h, (uint8_t)(v >> (8 * m)));
            }
        }
    }
    digests[t] = h;
}

__global__ void twg_naive_root_kernel(
    int total,
    const uint64_t* __restrict__ digests,
    uint64_t* __restrict__ state_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t h = TWG_FNV_BASIS;
    for (int t = 0; t < total; ++t) {
        const uint64_t d = digests[t];
        for (int m = 0; m < 8; ++m) {
            h = twg_naive_fnv_byte(h, (uint8_t)(d >> (8 * m)));
        }
    }
    state_checksum[0] = h;
}

__global__ void twg_naive_dump_kernel(
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
    TwgNaiveWorkspaceLayout L = twg_naive_make_layout(nullptr, spec);
    return L.required_bytes;
}

static cudaError_t twg_naive_reset_state(
    TwgNaiveState* st,
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

    TwgNaiveState* st =
        static_cast<TwgNaiveState*>(malloc(sizeof(TwgNaiveState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(TwgNaiveState));
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

    err = twg_naive_reset_state(st, stream);
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

    TwgNaiveState* st = static_cast<TwgNaiveState*>(state);
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

    TwgNaiveWorkspaceLayout L = twg_naive_make_layout(workspace, &st->spec);
    if (workspace_bytes < L.required_bytes) return cudaErrorInvalidValue;

    const int R = run->R;
    const int C = run->C;
    const int total = R * C;

    cudaError_t err = cudaSuccess;

    for (int d = 0; d <= R + C - 2; ++d) {
        const int i_lo = d - (C - 1) > 0 ? d - (C - 1) : 0;
        const int i_hi = d < R - 1 ? d : R - 1;
        const int ntiles = i_hi - i_lo + 1;
        twg_naive_diag_kernel<<<ntiles, 256, 0, stream>>>(
            R, C, run->K, d, (uint32_t)run->P1, (uint32_t)run->P2, in->U,
            in->V, st->acc, st->round_ctr);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    twg_naive_round_kernel<<<1, 1, 0, stream>>>(st->round_ctr,
                                                out->round_out);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    twg_naive_border_kernel<<<twg_ceil_div_int(C * 32 + R * 32, 128), 128, 0,
                              stream>>>(
        R, C, st->acc, out->border_row, out->border_col);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    twg_naive_digest_kernel<<<twg_ceil_div_int(total, 128), 128, 0, stream>>>(
        R, C, st->acc, L.digests);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    twg_naive_root_kernel<<<1, 1, 0, stream>>>(
        total, L.digests, out->state_checksum);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    if (run->dump == 1) {
        const int elems = R * 32 * C * 32;
        twg_naive_dump_kernel<<<twg_ceil_div_int(elems, 256), 256, 0,
                                stream>>>(elems, st->acc, out->acc_dump);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return twg_naive_reset_state(static_cast<TwgNaiveState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    TwgNaiveState* st = static_cast<TwgNaiveState*>(state);
    if (st->acc) cudaFree(st->acc);
    if (st->round_ctr) cudaFree(st->round_ctr);
    free(st);
}
