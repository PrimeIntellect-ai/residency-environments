// file: naive_ref.cu (microscale_requant_chain)
//
// Independent, clean multi-pass baseline used ONLY to calibrate the perf
// gate. Straightforward decomposition of the contract into one kernel per
// stage, scalar loads, materialized intermediates in workspace:
//   pass 0: memset row_err / sat1_count accumulators
//   pass 1: per-(row,block) stage-1 scale (amax) -> sf1 + sexp1 (i32)
//   pass 2: per-element E4M3 quantize -> e4m3_codes + M matrix (i32) +
//           per-row saturation counts via atomicAdd
//   pass 3: per-(row,block) stage-2 scale (qmax over M) -> sf2 + msb (i32)
//   pass 4: per-element LUT index -> nibble matrix (u8) + row_err via
//           64-bit atomicAdd of the exact squared error
//   pass 5: per-(row,byte) q4 packing from the nibble matrix
//   pass 6: per-(row,substream) digest word chains reading the four
//           output regions back from global memory
//   pass 7: per-row digest combine
// Bit-exact identical outputs to the reference.

#include "microscale_requant_chain_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct MrqNaiveState {
    MrqProblemSpec spec;
};

__device__ __constant__ int kNvBP[7] = { 1, 3, 5, 8, 13, 20, 28 };
__device__ __constant__ int kNvLP[8] = { 0, 1, 2, 3, 5, 8, 12, 16 };

__device__ __forceinline__ uint64_t nv_fnv_word(uint64_t h, uint64_t w) {
    h ^= w;
    h *= MRQ_FNV_PRIME;
    return h;
}

__device__ __forceinline__ int nv_floor_log2_bits(uint32_t u) {
    const int E = (int)((u >> 23) & 0xffu);
    if (E > 0) return E - 127;
    const uint32_t Mn = u & 0x7fffffu;
    return (31 - __clz(Mn)) - 149;
}

__device__ __forceinline__ void nv_e4m3_rne(
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

// pass 1
__global__ void nv_scale1_kernel(
    int R, int C, int S,
    const float* __restrict__ x,
    uint8_t* __restrict__ sf1,
    int32_t* __restrict__ sexp1m) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)R * S) return;
    const int r = (int)(idx / S);
    const int b = (int)(idx - (long long)r * S);
    const int k0 = 32 * b;
    const int k1 = (k0 + 32 < C) ? (k0 + 32) : C;

    uint32_t amax = 0;
    for (int k = k0; k < k1; ++k) {
        const uint32_t ab =
            __float_as_uint(x[(size_t)r * C + k]) & 0x7fffffffu;
        if (ab > amax) amax = ab;
    }

    int sexp1;
    uint8_t s1;
    if (amax == 0u) {
        sexp1 = -127;
        s1 = 0;
    } else {
        const int e = nv_floor_log2_bits(amax);
        sexp1 = e - 8;
        if (sexp1 < -127) sexp1 = -127;
        s1 = (uint8_t)(sexp1 + 127);
    }
    sf1[idx] = s1;
    sexp1m[idx] = sexp1;
}

// pass 2
__global__ void nv_quant1_kernel(
    int R, int C, int S,
    const float* __restrict__ x,
    const int32_t* __restrict__ sexp1m,
    uint8_t* __restrict__ e4m3_codes,
    int32_t* __restrict__ Mmat,
    int32_t* __restrict__ sat1_count) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)R * C) return;
    const int r = (int)(idx / C);
    const int k = (int)(idx - (long long)r * C);
    const int b = k >> 5;

    const uint32_t xb = __float_as_uint(x[idx]);
    int code, Mv, st;
    nv_e4m3_rne(xb, sexp1m[(size_t)r * S + b], &code, &Mv, &st);
    e4m3_codes[idx] = (uint8_t)(((xb >> 31) << 7) | code);
    Mmat[idx] = Mv;
    if (st) atomicAdd(&sat1_count[r], 1);
}

// pass 3
__global__ void nv_scale2_kernel(
    int R, int C, int S,
    const int32_t* __restrict__ Mmat,
    uint8_t* __restrict__ sf2,
    int32_t* __restrict__ msbm) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)R * S) return;
    const int r = (int)(idx / S);
    const int b = (int)(idx - (long long)r * S);
    const int k0 = 32 * b;
    const int k1 = (k0 + 32 < C) ? (k0 + 32) : C;

    int qmax = 0;
    for (int k = k0; k < k1; ++k) {
        const int Mv = Mmat[(size_t)r * C + k];
        if (Mv > qmax) qmax = Mv;
    }

    if (qmax == 0) {
        sf2[idx] = 0xFF;
        msbm[idx] = -1;
    } else {
        const int msb = 31 - __clz((uint32_t)qmax);
        sf2[idx] = (uint8_t)msb;
        msbm[idx] = msb;
    }
}

// pass 4
__global__ void nv_quant2_kernel(
    int R, int C, int S,
    const uint8_t* __restrict__ e4m3_codes,
    const int32_t* __restrict__ Mmat,
    const int32_t* __restrict__ msbm,
    uint8_t* __restrict__ nib,
    unsigned long long* __restrict__ row_err) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)R * C) return;
    const int r = (int)(idx / C);
    const int k = (int)(idx - (long long)r * C);
    const int b = k >> 5;

    const int sign = e4m3_codes[idx] >> 7;
    const int msb = msbm[(size_t)r * S + b];

    if (msb < 0) {
        nib[idx] = (uint8_t)(sign << 3);
        return;
    }

    const int Mv = Mmat[idx];
    const int w16 = 16 * Mv;
    int lut = 0;
    for (int j = 0; j < 7; ++j) {
        if (w16 >= (kNvBP[j] << msb)) ++lut;
    }
    nib[idx] = (uint8_t)((sign << 3) | lut);

    const long long a = (long long)(Mv << 3);
    const long long rt = (long long)(kNvLP[lut] << msb);
    const long long err = a - rt;
    atomicAdd(&row_err[r], (unsigned long long)(err * err));
}

// pass 5
__global__ void nv_pack_kernel(
    int R, int C, int Kb,
    const uint8_t* __restrict__ nib,
    uint8_t* __restrict__ q4_packed) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)R * Kb) return;
    const int r = (int)(idx / Kb);
    const int j = (int)(idx - (long long)r * Kb);

    const uint8_t lo = nib[(size_t)r * C + 2 * j];
    const uint8_t hi = (2 * j + 1 < C) ? nib[(size_t)r * C + 2 * j + 1] : 0;
    q4_packed[idx] = (uint8_t)(lo | (hi << 4));
}

// pass 6: digest sub-chains reading output regions back from global.
__global__ void nv_subdigest_kernel(
    int R, int C, int S, int Kb,
    const uint8_t* __restrict__ e4m3_codes,
    const uint8_t* __restrict__ q4_packed,
    const uint8_t* __restrict__ sf1,
    const uint8_t* __restrict__ sf2,
    uint64_t* __restrict__ subs) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)R * MRQ_DIGEST_SUBSTREAMS) return;
    const int r = (int)(idx / MRQ_DIGEST_SUBSTREAMS);
    const int t = (int)(idx - (long long)r * MRQ_DIGEST_SUBSTREAMS);

    const int Lb = C + Kb + 2 * S;
    const int W = (Lb + 7) / 8;

    uint64_t sub = MRQ_FNV_BASIS;
    for (int w = t; w < W; w += MRQ_DIGEST_SUBSTREAMS) {
        uint64_t word = 0;
        for (int bb = 0; bb < 8; ++bb) {
            const int i = 8 * w + bb;
            uint8_t byte = 0;
            if (i < C) {
                byte = e4m3_codes[(size_t)r * C + i];
            } else if (i < C + Kb) {
                byte = q4_packed[(size_t)r * Kb + (i - C)];
            } else if (i < C + Kb + S) {
                byte = sf1[(size_t)r * S + (i - C - Kb)];
            } else if (i < Lb) {
                byte = sf2[(size_t)r * S + (i - C - Kb - S)];
            }
            word |= (uint64_t)byte << (8 * bb);
        }
        sub = nv_fnv_word(sub, word);
    }
    subs[idx] = sub;
}

// pass 7
__global__ void nv_combine_kernel(
    int R,
    const uint64_t* __restrict__ subs,
    uint64_t* __restrict__ row_digest) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    uint64_t h = MRQ_FNV_BASIS;
    for (int t = 0; t < MRQ_DIGEST_SUBSTREAMS; ++t) {
        h = nv_fnv_word(h, subs[(size_t)r * MRQ_DIGEST_SUBSTREAMS + t]);
    }
    row_digest[r] = h;
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

static size_t nv_align256(size_t x) { return (x + 255u) & ~(size_t)255u; }

static void nv_ws_layout(
    int max_R, int max_C,
    size_t* off_sexp1, size_t* off_M, size_t* off_msb, size_t* off_nib,
    size_t* off_subs, size_t* total) {
    const int S = mrq_S(max_C);
    size_t off = 0;
    *off_sexp1 = off; off += nv_align256((size_t)max_R * S * 4);
    *off_M = off;     off += nv_align256((size_t)max_R * max_C * 4);
    *off_msb = off;   off += nv_align256((size_t)max_R * S * 4);
    *off_nib = off;   off += nv_align256((size_t)max_R * max_C);
    *off_subs = off;  off += nv_align256((size_t)max_R * 8 * 8);
    *total = off;
}

extern "C" size_t solution_workspace_bytes(const MrqProblemSpec* spec) {
    if (!mrq_validate_problem_spec(spec)) return 0;
    size_t a, b, c, d, e, total;
    nv_ws_layout(spec->max_R, spec->max_C, &a, &b, &c, &d, &e, &total);
    return total;
}

extern "C" cudaError_t solution_init(
    const MrqProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;
    if (!mrq_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }
    MrqNaiveState* st =
        static_cast<MrqNaiveState*>(malloc(sizeof(MrqNaiveState)));
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
    if (!state || !mrq_validate_run_spec(run) || !inputs_void || !outputs_void ||
        !workspace) {
        return cudaErrorInvalidValue;
    }

    MrqNaiveState* st = static_cast<MrqNaiveState*>(state);
    const MrqInputs* in = static_cast<const MrqInputs*>(inputs_void);
    MrqOutputs* out = static_cast<MrqOutputs*>(outputs_void);

    if (run->R > st->spec.max_R || run->C > st->spec.max_C) {
        return cudaErrorInvalidValue;
    }

    size_t off_sexp1, off_M, off_msb, off_nib, off_subs, need;
    nv_ws_layout(st->spec.max_R, st->spec.max_C,
                 &off_sexp1, &off_M, &off_msb, &off_nib, &off_subs, &need);
    if (workspace_bytes < need) return cudaErrorInvalidValue;

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    int32_t* sexp1m = reinterpret_cast<int32_t*>(ws + off_sexp1);
    int32_t* Mmat = reinterpret_cast<int32_t*>(ws + off_M);
    int32_t* msbm = reinterpret_cast<int32_t*>(ws + off_msb);
    uint8_t* nib = ws + off_nib;
    uint64_t* subs = reinterpret_cast<uint64_t*>(ws + off_subs);

    const int R = run->R;
    const int C = run->C;
    const int S = mrq_S(C);
    const int Kb = mrq_Kb(C);
    const int T = 256;

    cudaError_t err;
    err = cudaMemsetAsync(out->row_err, 0, (size_t)R * 8, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(out->sat1_count, 0, (size_t)R * 4, stream);
    if (err != cudaSuccess) return err;

    const long long RS = (long long)R * S;
    const long long RC = (long long)R * C;
    const long long RKb = (long long)R * Kb;

    nv_scale1_kernel<<<(unsigned)((RS + T - 1) / T), T, 0, stream>>>(
        R, C, S, in->x, out->sf1, sexp1m);
    nv_quant1_kernel<<<(unsigned)((RC + T - 1) / T), T, 0, stream>>>(
        R, C, S, in->x, sexp1m, out->e4m3_codes, Mmat, out->sat1_count);
    nv_scale2_kernel<<<(unsigned)((RS + T - 1) / T), T, 0, stream>>>(
        R, C, S, Mmat, out->sf2, msbm);
    nv_quant2_kernel<<<(unsigned)((RC + T - 1) / T), T, 0, stream>>>(
        R, C, S, out->e4m3_codes, Mmat, msbm, nib,
        reinterpret_cast<unsigned long long*>(out->row_err));
    nv_pack_kernel<<<(unsigned)((RKb + T - 1) / T), T, 0, stream>>>(
        R, C, Kb, nib, out->q4_packed);
    nv_subdigest_kernel<<<(unsigned)(((long long)R * 8 + T - 1) / T), T, 0, stream>>>(
        R, C, S, Kb, out->e4m3_codes, out->q4_packed, out->sf1, out->sf2,
        subs);
    nv_combine_kernel<<<mrq_ceil_div_int(R, T), T, 0, stream>>>(
        R, subs, out->row_digest);

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
