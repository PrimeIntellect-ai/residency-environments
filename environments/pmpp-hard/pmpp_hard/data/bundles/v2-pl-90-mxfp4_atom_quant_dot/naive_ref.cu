// file: naive_ref.cu (mxfp4_atom_quant_dot)
//
// Independent, clean multi-pass baseline used ONLY to calibrate the perf
// gate. Straightforward decomposition of the contract into one kernel per
// stage, scalar loads, materialized intermediates in workspace:
//   pass 0: memset atom outputs (padding)
//   pass 1: per-(row,block) scale derivation + sat counting (atomicAdd)
//   pass 2: per-element quantization -> codes (u8)
//   pass 3a: per-(row,logical byte) packing -> logical bytes (u8)
//   pass 3b: per-(row,atom byte) payload atom scatter
//   pass 3c: per-(row,scale) sf atom scatter
//   pass 4: per-element dequantize -> dq (fp32, materialized)
//   pass 5: block-per-row pinned-tree dot from dq
//   pass 6: per-(row,substream) FNV sub-digests
//   pass 7: per-row digest combine
// Bit-exact identical outputs to the reference.

#include "mxfp4_atom_quant_dot_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct MxqNaiveState {
    MxqProblemSpec spec;
};

__device__ __constant__ uint32_t kNvThrBits[7] = {
    0x3e800000u, 0x3f400000u, 0x3fa00000u, 0x3fe00000u,
    0x40200000u, 0x40600000u, 0x40a00000u
};
__device__ __constant__ uint32_t kNvSixBits = 0x40c00000u;
__device__ __constant__ uint32_t kNvLatBits[8] = {
    0x00000000u, 0x3f000000u, 0x3f800000u, 0x3fc00000u,
    0x40000000u, 0x40400000u, 0x40800000u, 0x40c00000u
};

__device__ __forceinline__ uint32_t nv_pow2_bits(uint32_t tbits, int p) {
    const int E = (int)((tbits >> 23) & 0xffu);
    const uint32_t M = tbits & 0x7fffffu;
    const int e = E - 127 + p;
    if (e >= -126) return ((uint32_t)(e + 127) << 23) | M;
    return (0x800000u | M) >> (-126 - e);
}

__device__ __forceinline__ int nv_floor_log2_bits(uint32_t u) {
    const int E = (int)((u >> 23) & 0xffu);
    if (E > 0) return E - 127;
    const uint32_t M = u & 0x7fffffu;
    return (31 - __clz(M)) - 149;
}

__device__ __forceinline__ uint64_t nv_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= MXQ_FNV_PRIME;
    return h;
}

// pass 1: one thread per (row, scale block)
__global__ void nv_scale_kernel(
    int R, int C, int S, int AS,
    const float* __restrict__ x,
    uint8_t* __restrict__ ws_sf,
    uint8_t* __restrict__ sf_atoms,
    int32_t* __restrict__ sat_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * S) return;

    const int r = idx / S;
    const int b = idx - r * S;

    const int k0 = 32 * b;
    const int k1 = (k0 + 32 < C) ? (k0 + 32) : C;

    uint32_t mb = 0;
    for (int k = k0; k < k1; ++k) {
        const uint32_t a = __float_as_uint(x[(size_t)r * C + k]) & 0x7fffffffu;
        if (a > mb) mb = a;
    }

    int sexp;
    uint8_t sf;
    if (mb == 0u) {
        sexp = -127;
        sf = 0x00;
    } else {
        const int e = nv_floor_log2_bits(mb);
        sexp = e - 2;
        if (sexp < -127) sexp = -127;
        sf = (uint8_t)(sexp + 127);
    }
    ws_sf[(size_t)r * S + b] = sf;

    const uint32_t t6 = nv_pow2_bits(kNvSixBits, sexp);
    int sat = 0;
    for (int k = k0; k < k1; ++k) {
        const uint32_t a = __float_as_uint(x[(size_t)r * C + k]) & 0x7fffffffu;
        if (a > t6) ++sat;
    }
    if (sat) atomicAdd(&sat_count[r], sat);

    const int am = r / 128;
    const int rr = r % 128;
    const int as = b / 4;
    const int c = b % 4;
    sf_atoms[(size_t)512 * ((size_t)am * AS + as) +
             (size_t)((rr % 32) * 16 + (rr / 32) * 4 + c)] = sf;
}

// pass 2: one thread per element
__global__ void nv_quant_kernel(
    int R, int C, int S,
    const float* __restrict__ x,
    const uint8_t* __restrict__ ws_sf,
    uint8_t* __restrict__ ws_codes) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)R * C) return;

    const int r = (int)(idx / C);
    const int k = (int)(idx - (size_t)r * C);

    const uint32_t xb = __float_as_uint(x[idx]);
    const uint32_t abits = xb & 0x7fffffffu;
    const int sexp = (int)ws_sf[(size_t)r * S + (k >> 5)] - 127;

    uint32_t thr[7];
    for (int i = 0; i < 7; ++i) thr[i] = nv_pow2_bits(kNvThrBits[i], sexp);

    int m;
    if (abits <= thr[0]) m = 0;
    else if (abits < thr[1]) m = 1;
    else if (abits <= thr[2]) m = 2;
    else if (abits < thr[3]) m = 3;
    else if (abits <= thr[4]) m = 4;
    else if (abits < thr[5]) m = 5;
    else if (abits <= thr[6]) m = 6;
    else m = 7;

    ws_codes[idx] = (uint8_t)(((xb >> 31) << 3) | (uint32_t)m);
}

// pass 3a: one thread per logical byte
__global__ void nv_pack_logical_kernel(
    int R, int C, int Kb,
    const uint8_t* __restrict__ ws_codes,
    uint8_t* __restrict__ ws_logical) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)R * Kb) return;

    const int r = (int)(idx / Kb);
    const int j = (int)(idx - (size_t)r * Kb);

    const uint8_t lo = ws_codes[(size_t)r * C + 2 * j];
    const uint8_t hi = (2 * j + 1 < C) ? ws_codes[(size_t)r * C + 2 * j + 1] : 0;
    ws_logical[idx] = (uint8_t)(lo | (hi << 4));
}

// pass 3b: one thread per (row, payload atom byte)
__global__ void nv_pay_atom_kernel(
    int R, int C, int AK,
    const uint8_t* __restrict__ ws_codes,
    uint8_t* __restrict__ pay_atoms) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)R * AK * 32) return;

    const int r = (int)(idx / ((size_t)AK * 32));
    const int rem = (int)(idx - (size_t)r * AK * 32);
    const int ak = rem / 32;
    const int u = rem - ak * 32;

    const int klo = 64 * ak + u;
    const int khi = klo + 32;
    const uint8_t lo = (klo < C) ? ws_codes[(size_t)r * C + klo] : 0;
    const uint8_t hi = (khi < C) ? ws_codes[(size_t)r * C + khi] : 0;

    const int am = r / 128;
    const int rr = r % 128;
    pay_atoms[(size_t)4096 * ((size_t)am * AK + ak) +
              (size_t)((u / 8) * 1024 + (rr % 32) * 32 + (rr / 32) * 8 + (u % 8))] =
        (uint8_t)(lo | (hi << 4));
}

// pass 4: one thread per element, materialize dq
__global__ void nv_dequant_kernel(
    int R, int C, int S,
    const uint8_t* __restrict__ ws_codes,
    const uint8_t* __restrict__ ws_sf,
    float* __restrict__ ws_dq) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)R * C) return;

    const int r = (int)(idx / C);
    const int k = (int)(idx - (size_t)r * C);

    const uint32_t code = ws_codes[idx];
    const uint32_t m = code & 7u;
    const uint32_t sign = (code >> 3) & 1u;

    uint32_t dqb = sign << 31;
    if (m != 0u) {
        const int sexp = (int)ws_sf[(size_t)r * S + (k >> 5)] - 127;
        dqb |= nv_pow2_bits(kNvLatBits[m], sexp);
    }
    ws_dq[idx] = __uint_as_float(dqb);
}

// pass 5: one 128-thread block per row, pinned tree
__global__ void nv_dot_kernel(
    int R, int C,
    const float* __restrict__ ws_dq,
    const float* __restrict__ v,
    float* __restrict__ row_dot) {
    __shared__ float tree[MXQ_DOT_LANES];

    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    if (r >= R) return;

    float p = 0.0f;
    for (int k = tid; k < C; k += MXQ_DOT_LANES) {
        p = __fadd_rn(p, __fmul_rn(ws_dq[(size_t)r * C + k], v[k]));
    }

    tree[tid] = p;
    __syncthreads();
    for (int stride = MXQ_DOT_LANES / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tree[tid] = __fadd_rn(tree[tid], tree[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) row_dot[r] = tree[0];
}

// pass 6: one thread per (row, substream)
__global__ void nv_subdigest_kernel(
    int R, int Kb, int S,
    const uint8_t* __restrict__ ws_logical,
    const uint8_t* __restrict__ ws_sf,
    uint64_t* __restrict__ ws_sub) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * MXQ_DIGEST_SUBSTREAMS) return;

    const int r = idx / MXQ_DIGEST_SUBSTREAMS;
    const int g = idx - r * MXQ_DIGEST_SUBSTREAMS;

    const int total_len = Kb + S;
    const int sub_len =
        (total_len + MXQ_DIGEST_SUBSTREAMS - 1) / MXQ_DIGEST_SUBSTREAMS;
    const int lo = g * sub_len;
    int hi = lo + sub_len;
    if (hi > total_len) hi = total_len;

    uint64_t sub = MXQ_FNV_BASIS;
    for (int i = lo; i < hi; ++i) {
        const uint8_t byte = (i < Kb)
            ? ws_logical[(size_t)r * Kb + i]
            : ws_sf[(size_t)r * S + (i - Kb)];
        sub = nv_fnv_byte(sub, byte);
    }
    ws_sub[idx] = sub;
}

// pass 7: one thread per row
__global__ void nv_digest_kernel(
    int R,
    const uint64_t* __restrict__ ws_sub,
    uint64_t* __restrict__ row_digest) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;

    uint64_t h = MXQ_FNV_BASIS;
    for (int g = 0; g < MXQ_DIGEST_SUBSTREAMS; ++g) {
        const uint64_t s = ws_sub[(size_t)r * MXQ_DIGEST_SUBSTREAMS + g];
        for (int i = 0; i < 8; ++i) {
            h = nv_fnv_byte(h, (uint8_t)((s >> (8 * i)) & 0xffu));
        }
    }
    row_digest[r] = h;
}

// ---------------------------------------------------------------------------
// Workspace layout: [sf u8 R*S][codes u8 R*C][logical u8 R*Kb][dq f32 R*C]
// [sub u64 R*8]
// ---------------------------------------------------------------------------

static void nv_layout(
    int max_R, int max_C,
    size_t* off_sf, size_t* off_codes, size_t* off_logical,
    size_t* off_dq, size_t* off_sub, size_t* total) {
    const size_t R = (size_t)max_R;
    const size_t C = (size_t)max_C;
    const size_t S = (size_t)mxq_S(max_C);
    const size_t Kb = (size_t)mxq_Kb(max_C);

    size_t o = 0;
    *off_sf = o;
    o += mxq_align_up_size(R * S, 256);
    *off_codes = o;
    o += mxq_align_up_size(R * C, 256);
    *off_logical = o;
    o += mxq_align_up_size(R * Kb, 256);
    *off_dq = o;
    o += mxq_align_up_size(R * C * sizeof(float), 256);
    *off_sub = o;
    o += mxq_align_up_size(R * MXQ_DIGEST_SUBSTREAMS * sizeof(uint64_t), 256);
    *total = (o < 128) ? 128 : o;
}

extern "C" size_t solution_workspace_bytes(const MxqProblemSpec* spec) {
    if (!mxq_validate_problem_spec(spec)) return 0;
    size_t a, b, c, d, e, total;
    nv_layout(spec->max_R, spec->max_C, &a, &b, &c, &d, &e, &total);
    return total;
}

extern "C" cudaError_t solution_init(
    const MxqProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    (void)stream;

    if (!mxq_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    MxqNaiveState* st =
        static_cast<MxqNaiveState*>(malloc(sizeof(MxqNaiveState)));
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
    if (!state || !mxq_validate_run_spec(run) ||
        !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    MxqNaiveState* st = static_cast<MxqNaiveState*>(state);
    const MxqInputs* in = static_cast<const MxqInputs*>(inputs_void);
    MxqOutputs* out = static_cast<MxqOutputs*>(outputs_void);

    if (run->R > st->spec.max_R || run->C > st->spec.max_C) {
        return cudaErrorInvalidValue;
    }

    size_t off_sf, off_codes, off_logical, off_dq, off_sub, total;
    nv_layout(st->spec.max_R, st->spec.max_C,
              &off_sf, &off_codes, &off_logical, &off_dq, &off_sub, &total);
    if (workspace_bytes < total) return cudaErrorInvalidValue;

    const int R = run->R;
    const int C = run->C;
    const int S = mxq_S(C);
    const int Kb = mxq_Kb(C);
    const int AK = mxq_AK(C);
    const int AS = mxq_AS(C);

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    uint8_t* ws_sf = ws + off_sf;
    uint8_t* ws_codes = ws + off_codes;
    uint8_t* ws_logical = ws + off_logical;
    float* ws_dq = reinterpret_cast<float*>(ws + off_dq);
    uint64_t* ws_sub = reinterpret_cast<uint64_t*>(ws + off_sub);

    cudaError_t err;

    err = cudaMemsetAsync(out->pay_atoms, 0, mxq_pay_atom_bytes(R, C), stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(out->sf_atoms, 0, mxq_sf_atom_bytes(R, C), stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(out->sat_count, 0, sizeof(int32_t) * (size_t)R, stream);
    if (err != cudaSuccess) return err;

    const int TB = 256;

    nv_scale_kernel<<<mxq_ceil_div_int(R * S, TB), TB, 0, stream>>>(
        R, C, S, AS, in->x, ws_sf, out->sf_atoms, out->sat_count);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    const size_t elems = (size_t)R * (size_t)C;
    const int egrid = (int)((elems + TB - 1) / TB);

    nv_quant_kernel<<<egrid, TB, 0, stream>>>(R, C, S, in->x, ws_sf, ws_codes);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    nv_pack_logical_kernel<<<
        (int)(((size_t)R * Kb + TB - 1) / TB), TB, 0, stream>>>(
        R, C, Kb, ws_codes, ws_logical);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    nv_pay_atom_kernel<<<
        (int)(((size_t)R * AK * 32 + TB - 1) / TB), TB, 0, stream>>>(
        R, C, AK, ws_codes, out->pay_atoms);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    nv_dequant_kernel<<<egrid, TB, 0, stream>>>(
        R, C, S, ws_codes, ws_sf, ws_dq);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    nv_dot_kernel<<<R, MXQ_DOT_LANES, 0, stream>>>(
        R, C, ws_dq, in->v, out->row_dot);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    nv_subdigest_kernel<<<
        mxq_ceil_div_int(R * MXQ_DIGEST_SUBSTREAMS, TB), TB, 0, stream>>>(
        R, Kb, S, ws_logical, ws_sf, ws_sub);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    nv_digest_kernel<<<mxq_ceil_div_int(R, TB), TB, 0, stream>>>(
        R, ws_sub, out->row_digest);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
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
