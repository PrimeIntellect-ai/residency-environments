// PMPP_CANARY_93_1f8a2c67b4 -- held-out canary; MUST NOT appear in any submission
// file: microscale_requant_chain_reference.cu
//
// Fully fused single-kernel pipeline, 128 threads / 4 rows per block (one
// warp per row). Per 32-element scale block: coalesced x loads (with a
// one-block register prefetch to keep the read stream ahead of the two
// chained reductions), u32-bit amax via __reduce_max_sync (positive fp32 /
// u32 order isomorphism), then a PURE INTEGER E4M3 RNE (mantissa decompose,
// binade-relative shift-round with even-tie, carry, 448 saturation, strict
// u>448 count) -- no floating-point rounding is trusted anywhere. The
// second reduction takes qmax over the integer E4M3 magnitudes, msb gives
// the stage-2 scale, and the LUT index is a count of seven exact integer
// boundary comparisons (16*M >= p<<msb), which implements nearest-with-
// ties-to-larger by construction. Squared error accumulates per lane in
// int64. Each row's bytes (e4m3 codes, packed int4 nibbles via shfl
// pairing, sf1, sf2) are staged in a contiguous shared STREAM buffer laid
// out exactly like the digest stream; warp 0 then runs the 32 digest word
// chains (4 rows x 8 round-robin substreams, aligned u64 shared reads) and
// each warp copies its row's regions out with alignment-fixed u32 stores.
// x is read exactly once; no workspace; nothing intermediate touches
// global memory.

#include "microscale_requant_chain_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MRQ_ROWS_PER_BLOCK 4
#define MRQ_FULL 0xffffffffu

struct MrqReferenceState {
    MrqProblemSpec spec;
};

__device__ __constant__ int kMrqBP[7] = { 1, 3, 5, 8, 13, 20, 28 };   // 16*B
__device__ __constant__ int kMrqLP[8] = { 0, 1, 2, 3, 5, 8, 12, 16 }; // 8*L

__device__ __forceinline__ uint64_t mrq_fnv_word(uint64_t h, uint64_t w) {
    h ^= w;
    h *= MRQ_FNV_PRIME;
    return h;
}

__device__ __forceinline__ int mrq_floor_log2_bits(uint32_t u) {
    const int E = (int)((u >> 23) & 0xffu);
    if (E > 0) return E - 127;
    const uint32_t Mn = u & 0x7fffffu;   // != 0 when called
    return (31 - __clz(Mn)) - 149;
}

// Integer E4M3 RNE of |x| * 2^-sexp1. Returns unsigned code (E<<3|m),
// integer magnitude M = value*2^9, and strict u>448 flag.
__device__ __forceinline__ void mrq_e4m3_rne(
    uint32_t xbits, int sexp1, int* code_out, int* M_out, int* sat_out) {
    const uint32_t E = (xbits >> 23) & 0xffu;
    const uint32_t Mn = xbits & 0x7fffffu;

    uint32_t mant;
    int ex;
    if (E == 0) {
        if (Mn == 0) { *code_out = 0; *M_out = 0; *sat_out = 0; return; }
        mant = Mn;
        ex = -149;
    } else {
        mant = 0x800000u | Mn;
        ex = (int)E - 150;
    }

    const int exp = ex - sexp1;
    const int msb = 31 - __clz(mant);
    const int k = msb + exp;

    int sat = 0;
    if (k > 8) {
        sat = 1;
    } else if (k == 8) {
        const int sh2 = exp - 5;
        if (sh2 >= 0) sat = (((unsigned long long)mant << sh2) > 14ull);
        else sat = ((unsigned long long)mant > (14ull << (-sh2)));
    }
    *sat_out = sat;

    const int step_exp = (k >= -6) ? (k - 3) : -9;
    const int sh = exp - step_exp;

    unsigned long long n;
    if (sh >= 0) {
        n = (unsigned long long)mant << sh;
    } else {
        const int s = -sh;
        if (s >= 25) {
            n = 0;
        } else {
            const unsigned long long n0 = (unsigned long long)mant >> s;
            const unsigned long long rem =
                (unsigned long long)mant & ((1ull << s) - 1ull);
            const unsigned long long half = 1ull << (s - 1);
            n = n0 + ((rem > half) ? 1ull : ((rem == half) ? (n0 & 1ull) : 0ull));
        }
    }

    int kk = k;
    if (k >= -6) {
        if (n == 16) { n = 8; kk = k + 1; }
        if (kk > 8 || (kk == 8 && n > 14)) {
            *code_out = (15 << 3) | 6;
            *M_out = 229376;
            return;
        }
        *code_out = ((kk + 7) << 3) | (int)(n - 8);
        *M_out = (int)(n << (kk + 6));
        return;
    }
    if (n >= 8) {
        *code_out = (1 << 3) | 0;
        *M_out = 8;
        return;
    }
    *code_out = (int)n;
    *M_out = (int)n;
}

// Warp-cooperative copy of len bytes from a shared buffer to global with
// 4-byte body stores (head/tail bytes handle arbitrary dst alignment).
__device__ __forceinline__ void mrq_warp_copy(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ src_sh,
    int len,
    int lane) {
    const int mis = (int)((uintptr_t)dst & 3u);
    int head = (mis == 0) ? 0 : (4 - mis);
    if (head > len) head = len;
    if (lane < head) dst[lane] = src_sh[lane];

    const int body4 = (len - head) >> 2;
    for (int i = lane; i < body4; i += 32) {
        const int o = head + 4 * i;
        const uint32_t w = (uint32_t)src_sh[o] |
                           ((uint32_t)src_sh[o + 1] << 8) |
                           ((uint32_t)src_sh[o + 2] << 16) |
                           ((uint32_t)src_sh[o + 3] << 24);
        *reinterpret_cast<uint32_t*>(dst + o) = w;
    }

    const int done = head + 4 * body4;
    const int tail = len - done;
    if (lane < tail) dst[done + lane] = src_sh[done + lane];
}

__global__ void __launch_bounds__(128)
mrq_reference_kernel(
    int R,
    int C,
    int S,
    int Kb,
    int Lb,
    int Wd,        // ceil(Lb/8)
    int stride,    // 8*Wd (row stream buffer stride, 8-aligned)
    const float* __restrict__ x,
    uint8_t* __restrict__ e4m3_codes,
    uint8_t* __restrict__ q4_packed,
    uint8_t* __restrict__ sf1,
    uint8_t* __restrict__ sf2,
    int64_t* __restrict__ row_err,
    uint64_t* __restrict__ row_digest,
    int32_t* __restrict__ sat1_count) {
    extern __shared__ uint8_t smem[];   // MRQ_ROWS_PER_BLOCK * stride

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int r0 = blockIdx.x * MRQ_ROWS_PER_BLOCK;
    const int r = r0 + warp;
    const bool valid_row = (r < R);

    uint8_t* rowbuf = smem + warp * stride;

    // zero the padded tail of the stream buffer
    for (int i = Lb + lane; i < stride; i += 32) rowbuf[i] = 0;

    {
        const float* xr = x + (size_t)r * (size_t)C;
        long long erracc = 0;
        unsigned satacc = 0;

        // register prefetch of the first block
        uint32_t xb_next = 0;
        {
            const int k = lane;
            xb_next = (valid_row && k < C) ? __float_as_uint(xr[k]) : 0u;
        }

        for (int b = 0; b < S; ++b) {
            const uint32_t xb = xb_next;
            const int kbase = 32 * b;
            const int k = kbase + lane;
            const bool valid = valid_row && (k < C);

            // prefetch next block while this one computes
            {
                const int kn = kbase + 32 + lane;
                xb_next = (valid_row && kn < C && b + 1 < S)
                    ? __float_as_uint(xr[kn]) : 0u;
            }

            const uint32_t abits = xb & 0x7fffffffu;
            const uint32_t amax = __reduce_max_sync(MRQ_FULL, abits);

            int sexp1;
            uint32_t s1;
            if (amax == 0u) {
                sexp1 = -127;
                s1 = 0u;
            } else {
                const int e = mrq_floor_log2_bits(amax);
                sexp1 = e - 8;
                if (sexp1 < -127) sexp1 = -127;
                s1 = (uint32_t)(sexp1 + 127);
            }

            int code, Mv, st;
            mrq_e4m3_rne(xb, sexp1, &code, &Mv, &st);
            const int sign = (int)(xb >> 31);

            if (valid) {
                rowbuf[k] = (uint8_t)((sign << 7) | code);
                satacc += (unsigned)st;
            }

            const uint32_t qmax =
                __reduce_max_sync(MRQ_FULL, valid ? (uint32_t)Mv : 0u);

            int q4n;
            if (qmax == 0u) {
                q4n = sign << 3;
                if (lane == 0 && valid_row) {
                    rowbuf[C + Kb + b] = s1;
                    rowbuf[C + Kb + S + b] = 0xFF;
                }
            } else {
                const int msb = 31 - __clz(qmax);
                const int w16 = 16 * Mv;
                int idx = 0;
                #pragma unroll
                for (int j = 0; j < 7; ++j) {
                    idx += (w16 >= (kMrqBP[j] << msb)) ? 1 : 0;
                }
                q4n = (sign << 3) | idx;
                const long long a = (long long)(Mv << 3);
                const long long rt = (long long)(kMrqLP[idx] << msb);
                const long long err = a - rt;
                if (valid) erracc += err * err;
                if (lane == 0 && valid_row) {
                    rowbuf[C + Kb + b] = s1;
                    rowbuf[C + Kb + S + b] = (uint8_t)msb;
                }
            }
            if (!valid) q4n = 0;   // padding nibble (sign bit included)

            // pack two nibbles per byte (even lane takes odd lane's nibble)
            const int nxt = __shfl_down_sync(MRQ_FULL, q4n, 1);
            if ((lane & 1) == 0 && k < C && valid_row) {
                const int hi = (k + 1 < C) ? nxt : 0;
                rowbuf[C + (k >> 1)] = (uint8_t)(q4n | (hi << 4));
            }
        }

        // warp reductions (order-free integer sums)
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            erracc += __shfl_down_sync(MRQ_FULL, erracc, off);
        }
        satacc = __reduce_add_sync(MRQ_FULL, satacc);
        if (lane == 0 && valid_row) {
            row_err[r] = (int64_t)erracc;
            sat1_count[r] = (int32_t)satacc;
        }
    }

    __syncthreads();

    // ---- digests: warp 0 runs 4 rows x 8 substream word chains ----
    if (warp == 0) {
        const int wrow = lane >> 3;
        const int t = lane & 7;
        const uint64_t* wbuf =
            reinterpret_cast<const uint64_t*>(smem + wrow * stride);

        uint64_t sub = MRQ_FNV_BASIS;
        for (int w = t; w < Wd; w += MRQ_DIGEST_SUBSTREAMS) {
            sub = mrq_fnv_word(sub, wbuf[w]);
        }

        uint64_t h = MRQ_FNV_BASIS;
        #pragma unroll
        for (int tt = 0; tt < MRQ_DIGEST_SUBSTREAMS; ++tt) {
            const uint64_t sv = __shfl_sync(MRQ_FULL, sub, (wrow << 3) + tt);
            h = mrq_fnv_word(h, sv);
        }
        if (t == 0 && (r0 + wrow) < R) row_digest[r0 + wrow] = h;
    }

    // ---- copy-out: each warp writes its row's four regions ----
    if (valid_row) {
        mrq_warp_copy(e4m3_codes + (size_t)r * (size_t)C, rowbuf, C, lane);
        mrq_warp_copy(q4_packed + (size_t)r * (size_t)Kb, rowbuf + C, Kb, lane);
        mrq_warp_copy(sf1 + (size_t)r * (size_t)S, rowbuf + C + Kb, S, lane);
        mrq_warp_copy(sf2 + (size_t)r * (size_t)S, rowbuf + C + Kb + S, S, lane);
    }
}

extern "C" size_t solution_workspace_bytes(const MrqProblemSpec* spec) {
    if (!mrq_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const MrqProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;
    if (!mrq_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }
    MrqReferenceState* st =
        static_cast<MrqReferenceState*>(malloc(sizeof(MrqReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;
    memcpy(&st->spec, spec, sizeof(MrqProblemSpec));
    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const MrqRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;

    if (!state || !mrq_validate_run_spec(run) || !inputs_void || !outputs_void) {
        return cudaErrorInvalidValue;
    }
    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    MrqReferenceState* st = static_cast<MrqReferenceState*>(state);
    const MrqInputs* in = static_cast<const MrqInputs*>(inputs_void);
    MrqOutputs* out = static_cast<MrqOutputs*>(outputs_void);

    if (run->R > st->spec.max_R || run->C > st->spec.max_C) {
        return cudaErrorInvalidValue;
    }
    if (!in->x || !out->e4m3_codes || !out->q4_packed || !out->sf1 ||
        !out->sf2 || !out->row_err || !out->row_digest || !out->sat1_count) {
        return cudaErrorInvalidValue;
    }

    const int R = run->R;
    const int C = run->C;
    const int S = mrq_S(C);
    const int Kb = mrq_Kb(C);
    const int Lb = mrq_stream_bytes(C);
    const int Wd = (Lb + 7) / 8;
    const int stride = 8 * Wd;

    const size_t shmem = (size_t)MRQ_ROWS_PER_BLOCK * (size_t)stride;
    const int grid = mrq_ceil_div_int(R, MRQ_ROWS_PER_BLOCK);

    mrq_reference_kernel<<<grid, 128, shmem, stream>>>(
        R, C, S, Kb, Lb, Wd, stride,
        in->x,
        out->e4m3_codes, out->q4_packed, out->sf1, out->sf2,
        out->row_err, out->row_digest, out->sat1_count);

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
