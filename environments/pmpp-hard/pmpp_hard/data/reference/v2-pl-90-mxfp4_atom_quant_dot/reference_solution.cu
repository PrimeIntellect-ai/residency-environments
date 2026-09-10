// PMPP_CANARY_90_5c6e8f1c91 -- held-out canary; MUST NOT appear in any submission
// file: mxfp4_atom_quant_dot_reference.cu
//
// Fully fused single-kernel pipeline. One 128-thread block handles four
// consecutive rows (one warp per row for quantization):
//   phase 1: warp-per-row MXFP4 quantization FUSED with the pinned-order
//            dequant-dot. Coalesced 32-lane loads (one scale block per
//            iteration), u32-bit amax butterfly, then scaled-domain compares:
//            y = |x| * 2^-sexp is exact fp32 everywhere it can affect a
//            threshold decision (both operands are normal and the product is
//            only rounded when subnormal, far below the 0.25 threshold), so
//            constant-threshold compares are exactly equivalent to the
//            contract table. The 128 pinned dot partials for row r live as
//            4 register accumulators per lane: lane l, acc[j] holds
//            partial[32*j + l] because block b assigns k = 32*b + l and
//            k mod 128 = 32*(b mod 4) + l; ascending b preserves the pinned
//            ascending-k order per partial. The reduction tree collapses to
//            two in-register fadds (strides 64, 32) plus a shfl butterfly
//            (strides 16..1) in the exact contract order.
//   phase 2: atom stores. Payload bytes are assembled 8-at-a-time from the
//            shared logical view and written as aligned u64 words directly
//            into the swizzled atom offsets; scale atoms as u32 words.
//            Padding rows/columns are written as zeros by the same path
//            (grid covers all 128*AM tile rows), so no pre-memset is needed.
//   phase 3: two-level FNV digests: 32 chains (4 rows x 8 substreams) run in
//            parallel on warp 0 from shared memory, combined via shuffles.
// No workspace, no intermediate global traffic: x is read exactly once.

#include "mxfp4_atom_quant_dot_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct MxqReferenceState {
    MxqProblemSpec spec;
};

#define MXQ_ROWS_PER_BLOCK 4
#define MXQ_FULL_MASK 0xffffffffu

// Bit patterns of the E2M1 lattice magnitudes (index 1..7).
__device__ __constant__ uint32_t kMxqLatBits[8] = {
    0x00000000u, 0x3f000000u, 0x3f800000u, 0x3fc00000u,
    0x40000000u, 0x40400000u, 0x40800000u, 0x40c00000u
};

// floor(log2(.)) of a positive finite fp32 given as bits.
__device__ __forceinline__ int mxq_floor_log2_bits(uint32_t u) {
    const int E = (int)((u >> 23) & 0xffu);
    if (E > 0) return E - 127;
    const uint32_t M = u & 0x7fffffu;  // != 0 when this is called
    return (31 - __clz(M)) - 149;
}

__device__ __forceinline__ uint64_t mxq_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= MXQ_FNV_PRIME;
    return h;
}

__global__ void __launch_bounds__(128)
mxq_reference_kernel(
    int R,
    int C,
    int S,
    int Kb,
    int AK,
    int AS,
    const float* __restrict__ x,
    const float* __restrict__ v,
    uint8_t* __restrict__ pay_atoms,
    uint8_t* __restrict__ sf_atoms,
    float* __restrict__ row_dot,
    uint64_t* __restrict__ row_digest,
    int32_t* __restrict__ sat_count) {
    extern __shared__ uint8_t smem[];

    const int KbPad = (Kb + 15) & ~15;
    const int SPad = (S + 15) & ~15;

    uint8_t* sh_logical = smem;                                   // 4 x KbPad
    uint8_t* sh_sf = smem + MXQ_ROWS_PER_BLOCK * KbPad;           // 4 x SPad

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int r0 = blockIdx.x * MXQ_ROWS_PER_BLOCK;
    const int r = r0 + warp;
    const bool valid_row = (r < R);

    // -------- phase 1: quantize one row per warp, dot fused in ----------
    {
        const float* xr = x + (size_t)r * (size_t)C;
        uint8_t* lrow = sh_logical + warp * KbPad;
        uint8_t* srow = sh_sf + warp * SPad;
        int satacc = 0;

        // acc[j] == pinned dot partial[32*j + lane] (see header comment).
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Process scale blocks in batches of 4 so that 8 independent loads
        // are in flight per warp iteration (memory-level parallelism), then
        // 4 independent redux/shuffle chains overlap.
        for (int b0 = 0; b0 < S; b0 += 4) {
            const int nb = (S - b0 < 4) ? (S - b0) : 4;

            float xv[4], vv[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = 32 * (b0 + j) + lane;
                const bool validk = (j < nb) && valid_row && (k < C);
                xv[j] = validk ? xr[k] : 0.0f;
                vv[j] = validk ? v[k] : 0.0f;
            }

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j >= nb) break;
                const int b = b0 + j;
                const int k = 32 * b + lane;
                const bool validk = valid_row && (k < C);

                const uint32_t xbits = __float_as_uint(xv[j]);
                const uint32_t abits = xbits & 0x7fffffffu;

                // amax of the block: hw redux (u32 order == positive fp32).
                const uint32_t mb = __reduce_max_sync(MXQ_FULL_MASK, abits);

                uint32_t sf;
                if (mb == 0u) {
                    sf = 0u;
                } else {
                    const int e = mxq_floor_log2_bits(mb);
                    int sexp = e - 2;
                    if (sexp < -127) sexp = -127;
                    sf = (uint32_t)(sexp + 127);
                }

                // y = |x| * 2^-sexp: exact wherever a threshold decision can
                // be affected, so constant compares match the contract table.
                const float inv = __uint_as_float((254u - sf) << 23);
                const float y = __fmul_rn(fabsf(xv[j]), inv);

                int m;
                if (y <= 0.25f) m = 0;
                else if (y < 0.75f) m = 1;
                else if (y <= 1.25f) m = 2;
                else if (y < 1.75f) m = 3;
                else if (y <= 2.5f) m = 4;
                else if (y < 3.5f) m = 5;
                else if (y <= 5.0f) m = 6;
                else m = 7;

                const int code = (int)((xbits >> 31) << 3) | m;

                satacc += (validk && (y > 6.0f)) ? 1 : 0;

                const int nxt = __shfl_down_sync(MXQ_FULL_MASK, code, 1);
                if ((lane & 1) == 0 && k < C) {
                    const int hi = (k + 1 < C) ? nxt : 0;
                    lrow[16 * b + (lane >> 1)] = (uint8_t)(code | (hi << 4));
                }
                if (lane == 0) srow[b] = (uint8_t)sf;

                // Fused pinned-order dot contribution (exact dq).
                if (validk) {
                    const float sfpow = (sf != 0u)
                        ? __uint_as_float(sf << 23)
                        : __uint_as_float(0x400000u);  // 2^-127
                    const float dq = __fmul_rn(
                        __uint_as_float(kMxqLatBits[m] | ((xbits >> 31) << 31)),
                        sfpow);
                    acc[b & 3] = __fadd_rn(acc[b & 3], __fmul_rn(dq, vv[j]));
                }
            }
        }

        satacc = (int)__reduce_add_sync(MXQ_FULL_MASK, (uint32_t)satacc);
        if (lane == 0 && valid_row) sat_count[r] = satacc;

        // Pinned tree: strides 64, 32 in-register; strides 16..1 via shfl.
        acc[0] = __fadd_rn(acc[0], acc[2]);
        acc[1] = __fadd_rn(acc[1], acc[3]);
        acc[0] = __fadd_rn(acc[0], acc[1]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float o = __shfl_down_sync(MXQ_FULL_MASK, acc[0], off);
            acc[0] = __fadd_rn(acc[0], o);
        }
        if (lane == 0 && valid_row) row_dot[r] = acc[0];
    }

    __syncthreads();

    // ---------------- phase 2: atom stores (whole block) ----------------
    const int am = r0 >> 7;                    // all 4 rows share one am tile
    {
        // Payload: one u64 (8 swizzled bytes) per thread per iteration.
        const int groups = MXQ_ROWS_PER_BLOCK * AK * 4;
        for (int idx = tid; idx < groups; idx += 128) {
            const int wrow = idx / (AK * 4);
            const int rem = idx - wrow * (AK * 4);
            const int ak = rem >> 2;
            const int ug = rem & 3;

            const int rr = (r0 + wrow) & 127;
            const bool vr = (r0 + wrow) < R;
            const uint8_t* lrow = sh_logical + wrow * KbPad;

            uint64_t word = 0;
            if (vr) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int u = 8 * ug + i;
                    const int klo = 64 * ak + u;
                    const int khi = klo + 32;
                    uint32_t lo = 0, hi = 0;
                    if (klo < C) {
                        lo = (uint32_t)(lrow[klo >> 1] >> ((klo & 1) * 4)) & 0xfu;
                    }
                    if (khi < C) {
                        hi = (uint32_t)(lrow[khi >> 1] >> ((khi & 1) * 4)) & 0xfu;
                    }
                    word |= (uint64_t)(lo | (hi << 4)) << (8 * i);
                }
            }
            const size_t off =
                (size_t)4096 * ((size_t)am * AK + (size_t)ak) +
                (size_t)(ug * 1024 + (rr & 31) * 32 + (rr >> 5) * 8);
            *reinterpret_cast<uint64_t*>(pay_atoms + off) = word;
        }

        // Scales: one u32 (4 scale columns) per thread per iteration.
        const int sgroups = MXQ_ROWS_PER_BLOCK * AS;
        for (int idx = tid; idx < sgroups; idx += 128) {
            const int wrow = idx / AS;
            const int as = idx - wrow * AS;

            const int rr = (r0 + wrow) & 127;
            const bool vr = (r0 + wrow) < R;
            const uint8_t* srow = sh_sf + wrow * SPad;

            uint32_t word = 0;
            if (vr) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int s = 4 * as + c;
                    if (s < S) word |= (uint32_t)srow[s] << (8 * c);
                }
            }
            const size_t off =
                (size_t)512 * ((size_t)am * AS + (size_t)as) +
                (size_t)((rr & 31) * 16 + (rr >> 5) * 4);
            *reinterpret_cast<uint32_t*>(sf_atoms + off) = word;
        }
    }

    // ---------------- phase 3: digests (warp 0: 4 rows x 8 substreams) --
    if (warp == 0) {
        const int wrow = lane >> 3;
        const int g = lane & 7;
        const uint8_t* lrow = sh_logical + wrow * KbPad;
        const uint8_t* srow = sh_sf + wrow * SPad;

        const int total_len = Kb + S;
        const int sub_len =
            (total_len + MXQ_DIGEST_SUBSTREAMS - 1) / MXQ_DIGEST_SUBSTREAMS;
        const int lo = g * sub_len;
        int hi = lo + sub_len;
        if (hi > total_len) hi = total_len;

        uint64_t sub = MXQ_FNV_BASIS;
        for (int i = lo; i < hi; ++i) {
            const uint8_t byte = (i < Kb) ? lrow[i] : srow[i - Kb];
            sub = mxq_fnv_byte(sub, byte);
        }

        uint64_t h = MXQ_FNV_BASIS;
        #pragma unroll
        for (int gg = 0; gg < MXQ_DIGEST_SUBSTREAMS; ++gg) {
            const uint64_t s =
                __shfl_sync(MXQ_FULL_MASK, sub, (wrow << 3) + gg);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                h = mxq_fnv_byte(h, (uint8_t)((s >> (8 * i)) & 0xffu));
            }
        }
        if (g == 0 && (r0 + wrow) < R) row_digest[r0 + wrow] = h;
    }
}

extern "C" size_t solution_workspace_bytes(const MxqProblemSpec* spec) {
    if (!mxq_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const MxqProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!mxq_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    MxqReferenceState* st =
        static_cast<MxqReferenceState*>(malloc(sizeof(MxqReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;

    memcpy(&st->spec, spec, sizeof(MxqProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const MxqRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;

    if (!state || !mxq_validate_run_spec(run) || !inputs_void || !outputs_void) {
        return cudaErrorInvalidValue;
    }
    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    MxqReferenceState* st = static_cast<MxqReferenceState*>(state);
    const MxqInputs* in = static_cast<const MxqInputs*>(inputs_void);
    MxqOutputs* out = static_cast<MxqOutputs*>(outputs_void);

    if (run->R > st->spec.max_R || run->C > st->spec.max_C) {
        return cudaErrorInvalidValue;
    }
    if (!in->x || !in->v ||
        !out->pay_atoms || !out->sf_atoms || !out->row_dot ||
        !out->row_digest || !out->sat_count) {
        return cudaErrorInvalidValue;
    }

    const int R = run->R;
    const int C = run->C;
    const int S = mxq_S(C);
    const int Kb = mxq_Kb(C);
    const int AM = mxq_AM(R);
    const int AK = mxq_AK(C);
    const int AS = mxq_AS(C);

    const int KbPad = (Kb + 15) & ~15;
    const int SPad = (S + 15) & ~15;
    const size_t shmem =
        (size_t)MXQ_ROWS_PER_BLOCK * (size_t)(KbPad + SPad) + 16;

    const int grid = 32 * AM;  // ceil(128*AM / MXQ_ROWS_PER_BLOCK)

    mxq_reference_kernel<<<grid, 128, shmem, stream>>>(
        R, C, S, Kb, AK, AS,
        in->x, in->v,
        out->pay_atoms, out->sf_atoms,
        out->row_dot, out->row_digest, out->sat_count);

    return cudaPeekAtLastError();
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
