// PMPP_CANARY_72_d00b550927 -- held-out canary; MUST NOT appear in any submission
// file: reference.cu
#include "norm_backward_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstring>
#include <new>
#include <vector>

#define LNR_CHECK_CUDA(expr)                         \
    do {                                             \
        cudaError_t _e = (expr);                     \
        if (_e != cudaSuccess) return _e;            \
    } while (0)

static inline uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

static inline size_t dtype_size(uint32_t dtype) {
    return dtype == LNR_DTYPE_BF16 ? 2u : 4u;
}

static bool valid_spec_host(const Spec* s) {
    if (!s) return false;
    if (s->magic != LNRBWD_MAGIC || s->version != LNRBWD_VERSION) return false;
    if (s->hidden_size < 1u || s->hidden_size > LNR_SEGMENT_COLS * LNR_MAX_SEGMENTS) return false;
    if (s->param_count < 1u || s->param_count > 16u) return false;
    if (s->max_cache_rows < 1u || s->max_cache_rows > 65536u) return false;
    if (s->max_backward_rows_per_run > LNR_ROWS_PER_PARTIAL * LNR_STAGE2_LEAVES) return false;
    if (s->max_ops_per_run > 512u) return false;
    if (s->max_flush_records_per_run > 512u) return false;
    if (s->storage_dtype != LNR_DTYPE_BF16 && s->storage_dtype != LNR_DTYPE_F32) return false;
    if (s->eps_ln < 0.0f || s->eps_rms < 0.0f) return false;
    if (s->flags != 0u) return false;
    for (uint32_t i = 0; i < 7; ++i) {
        if (s->reserved[i] != 0u) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Device exact-fp helpers
// -----------------------------------------------------------------------------
__device__ __forceinline__ float lnr_f32_from_bits(uint32_t u) {
    union { uint32_t u; float f; } v;
    v.u = u;
    return v.f;
}

__device__ __forceinline__ uint32_t lnr_f32_bits(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    return v.u;
}

__device__ __forceinline__ bool lnr_bits_is_nan(uint32_t u) {
    return ((u & 0x7f800000u) == 0x7f800000u) && ((u & 0x007fffffu) != 0u);
}

__device__ __forceinline__ float lnr_canon_nan(float x) {
    uint32_t u = lnr_f32_bits(x);
    if (lnr_bits_is_nan(u)) return lnr_f32_from_bits(0x7fc00000u);
    return x;
}

__device__ __forceinline__ float rn_add(float a, float b) {
    float r;
    asm volatile ("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return lnr_canon_nan(r);
}

__device__ __forceinline__ float rn_sub(float a, float b) {
    float r;
    asm volatile ("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return lnr_canon_nan(r);
}

__device__ __forceinline__ float rn_mul(float a, float b) {
    float r;
    asm volatile ("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return lnr_canon_nan(r);
}

__device__ __forceinline__ float rn_div(float a, float b) {
    float r;
    asm volatile ("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return lnr_canon_nan(r);
}

__device__ __forceinline__ float rn_sqrt(float a) {
    float r;
    asm volatile ("sqrt.rn.f32 %0, %1;" : "=f"(r) : "f"(a));
    return lnr_canon_nan(r);
}

__device__ __forceinline__ float u32_to_f32(uint32_t x) {
    return static_cast<float>(x);
}

__device__ __forceinline__ float load_storage_elem(
    const void* base,
    uint32_t dtype,
    size_t index) {

    if (dtype == LNR_DTYPE_BF16) {
        uint16_t h = reinterpret_cast<const uint16_t*>(base)[index];
        uint32_t u = uint32_t(h) << 16;
        if (lnr_bits_is_nan(u)) u = 0x7fc00000u;
        return lnr_f32_from_bits(u);
    }

    uint32_t u = reinterpret_cast<const uint32_t*>(base)[index];
    if (lnr_bits_is_nan(u)) u = 0x7fc00000u;
    return lnr_f32_from_bits(u);
}

__device__ __forceinline__ uint16_t f32_to_bf16_rne(float x) {
    x = lnr_canon_nan(x);
    uint32_t u = lnr_f32_bits(x);
    if (lnr_bits_is_nan(u)) return uint16_t(0x7fc0u);

    uint32_t lsb = (u >> 16) & 1u;
    uint32_t bias = 0x7fffu + lsb;
    return uint16_t((u + bias) >> 16);
}

__device__ __forceinline__ void store_storage_elem(
    void* base,
    uint32_t dtype,
    size_t index,
    float value) {

    value = lnr_canon_nan(value);

    if (dtype == LNR_DTYPE_BF16) {
        reinterpret_cast<uint16_t*>(base)[index] = f32_to_bf16_rne(value);
    } else {
        reinterpret_cast<uint32_t*>(base)[index] = lnr_f32_bits(value);
    }
}

__device__ __forceinline__ void store_f32(float* base, size_t index, float value) {
    value = lnr_canon_nan(value);
    base[index] = value;
}

struct WelfordTriple {
    uint32_t n;
    float mean;
    float m2;
};

__device__ __forceinline__ WelfordTriple welford_combine(
    WelfordTriple a,
    WelfordTriple b) {

    if (a.n == 0u) return b;
    if (b.n == 0u) return a;

    WelfordTriple c;
    c.n = a.n + b.n;

    float delta = rn_sub(b.mean, a.mean);
    float ratio = rn_div(u32_to_f32(b.n), u32_to_f32(c.n));
    c.mean = rn_add(a.mean, rn_mul(delta, ratio));

    float ab = rn_mul(u32_to_f32(a.n), u32_to_f32(b.n));
    float scale = rn_div(ab, u32_to_f32(c.n));
    float term = rn_mul(rn_mul(delta, delta), scale);
    c.m2 = rn_add(rn_add(a.m2, b.m2), term);

    return c;
}

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------
__global__ void zero_dx_kernel(
    void* dx,
    uint32_t dtype,
    uint32_t n,
    uint32_t rows,
    uint32_t row_base) {

    uint64_t total = uint64_t(rows) * uint64_t(n);
    uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = uint64_t(blockDim.x) * gridDim.x;

    for (uint64_t t = tid; t < total; t += stride) {
        uint32_t r = uint32_t(t / n);
        uint32_t c = uint32_t(t - uint64_t(r) * n);
        size_t idx = size_t(row_base + r) * n + c;
        store_storage_elem(dx, dtype, idx, 0.0f);
    }
}

__global__ void save_stats_kernel(
    uint8_t op_kind,
    const void* x,
    uint32_t dtype,
    uint32_t n,
    uint32_t x_stride,
    uint32_t x_row_base,
    uint32_t rows,
    uint32_t cache_base,
    float eps,
    uint64_t gen_first,
    float* cache_mean,
    float* cache_rstd,
    uint32_t* cache_kind,
    uint64_t* cache_gen) {

    uint32_t local_row = blockIdx.x;
    if (local_row >= rows) return;

    uint32_t tid = threadIdx.x;
    uint32_t row = x_row_base + local_row;
    uint32_t slot = cache_base + local_row;

    __shared__ float sh_val[LNR_SEGMENT_COLS];
    __shared__ uint32_t sh_n[LNR_SEGMENT_COLS];
    __shared__ float sh_mean[LNR_SEGMENT_COLS];
    __shared__ float sh_m2[LNR_SEGMENT_COLS];

    __shared__ float seg_val[LNR_MAX_SEGMENTS];
    __shared__ uint32_t seg_n[LNR_MAX_SEGMENTS];
    __shared__ float seg_mean[LNR_MAX_SEGMENTS];
    __shared__ float seg_m2[LNR_MAX_SEGMENTS];

    uint32_t num_segments = (n + LNR_SEGMENT_COLS - 1u) / LNR_SEGMENT_COLS;

    if (op_kind == LNR_OP_SAVE_LN) {
        for (uint32_t seg = 0; seg < num_segments; ++seg) {
            uint32_t col = seg * LNR_SEGMENT_COLS + tid;
            if (col < n) {
                float xv = load_storage_elem(x, dtype, size_t(row) * x_stride + col);
                sh_n[tid] = 1u;
                sh_mean[tid] = xv;
                sh_m2[tid] = 0.0f;
            } else {
                sh_n[tid] = 0u;
                sh_mean[tid] = 0.0f;
                sh_m2[tid] = 0.0f;
            }
            __syncthreads();

            for (uint32_t stride = 1u; stride < LNR_SEGMENT_COLS; stride <<= 1u) {
                if ((tid % (2u * stride)) == 0u) {
                    uint32_t other = tid + stride;
                    WelfordTriple a{sh_n[tid], sh_mean[tid], sh_m2[tid]};
                    WelfordTriple b{sh_n[other], sh_mean[other], sh_m2[other]};
                    WelfordTriple c = welford_combine(a, b);
                    sh_n[tid] = c.n;
                    sh_mean[tid] = c.mean;
                    sh_m2[tid] = c.m2;
                }
                __syncthreads();
            }

            if (tid == 0u) {
                seg_n[seg] = sh_n[0];
                seg_mean[seg] = sh_mean[0];
                seg_m2[seg] = sh_m2[0];
            }
            __syncthreads();
        }

        if (tid < LNR_MAX_SEGMENTS) {
            if (tid >= num_segments) {
                seg_n[tid] = 0u;
                seg_mean[tid] = 0.0f;
                seg_m2[tid] = 0.0f;
            }
        }
        __syncthreads();

        for (uint32_t stride = 1u; stride < LNR_MAX_SEGMENTS; stride <<= 1u) {
            if (tid < LNR_MAX_SEGMENTS && (tid % (2u * stride)) == 0u) {
                uint32_t other = tid + stride;
                WelfordTriple a{seg_n[tid], seg_mean[tid], seg_m2[tid]};
                WelfordTriple b{seg_n[other], seg_mean[other], seg_m2[other]};
                WelfordTriple c = welford_combine(a, b);
                seg_n[tid] = c.n;
                seg_mean[tid] = c.mean;
                seg_m2[tid] = c.m2;
            }
            __syncthreads();
        }

        if (tid == 0u) {
            float mean = seg_mean[0];
            float var = rn_div(seg_m2[0], u32_to_f32(n));
            float rstd = rn_div(1.0f, rn_sqrt(rn_add(var, eps)));

            cache_mean[slot] = lnr_canon_nan(mean);
            cache_rstd[slot] = lnr_canon_nan(rstd);
            cache_kind[slot] = LNR_CACHE_LN;
            cache_gen[slot] = gen_first + uint64_t(local_row);
        }
    } else {
        for (uint32_t seg = 0; seg < num_segments; ++seg) {
            uint32_t col = seg * LNR_SEGMENT_COLS + tid;
            if (col < n) {
                float xv = load_storage_elem(x, dtype, size_t(row) * x_stride + col);
                sh_val[tid] = rn_mul(xv, xv);
            } else {
                sh_val[tid] = 0.0f;
            }
            __syncthreads();

            for (uint32_t stride = 1u; stride < LNR_SEGMENT_COLS; stride <<= 1u) {
                if ((tid % (2u * stride)) == 0u) {
                    sh_val[tid] = rn_add(sh_val[tid], sh_val[tid + stride]);
                }
                __syncthreads();
            }

            if (tid == 0u) seg_val[seg] = sh_val[0];
            __syncthreads();
        }

        if (tid < LNR_MAX_SEGMENTS) {
            if (tid >= num_segments) seg_val[tid] = 0.0f;
        }
        __syncthreads();

        for (uint32_t stride = 1u; stride < LNR_MAX_SEGMENTS; stride <<= 1u) {
            if (tid < LNR_MAX_SEGMENTS && (tid % (2u * stride)) == 0u) {
                seg_val[tid] = rn_add(seg_val[tid], seg_val[tid + stride]);
            }
            __syncthreads();
        }

        if (tid == 0u) {
            float ms = rn_div(seg_val[0], u32_to_f32(n));
            float rstd = rn_div(1.0f, rn_sqrt(rn_add(ms, eps)));

            cache_mean[slot] = 0.0f;
            cache_rstd[slot] = lnr_canon_nan(rstd);
            cache_kind[slot] = LNR_CACHE_RMS;
            cache_gen[slot] = gen_first + uint64_t(local_row);
        }
    }
}

__global__ void backward_dx_kernel(
    uint8_t op_kind,
    const void* x,
    const void* dy,
    const void* gamma,
    void* dx,
    uint32_t dtype,
    uint32_t n,
    uint32_t x_stride,
    uint32_t dy_stride,
    uint32_t gamma_stride,
    uint32_t param_id,
    uint32_t x_row_base,
    uint32_t dy_row_base,
    uint32_t cache_base,
    uint32_t rows,
    uint32_t dx_out_base,
    const float* cache_mean,
    const float* cache_rstd) {

    uint32_t local_row = blockIdx.x;
    if (local_row >= rows) return;

    uint32_t tid = threadIdx.x;
    uint32_t x_row = x_row_base + local_row;
    uint32_t dy_row = dy_row_base + local_row;
    uint32_t slot = cache_base + local_row;

    float mean = cache_mean[slot];
    float rstd = cache_rstd[slot];

    __shared__ float sh1[LNR_SEGMENT_COLS];
    __shared__ float sh2[LNR_SEGMENT_COLS];
    __shared__ float seg1[LNR_MAX_SEGMENTS];
    __shared__ float seg2[LNR_MAX_SEGMENTS];
    __shared__ float c1_shared;
    __shared__ float c2_shared;

    uint32_t num_segments = (n + LNR_SEGMENT_COLS - 1u) / LNR_SEGMENT_COLS;

    for (uint32_t seg = 0; seg < num_segments; ++seg) {
        uint32_t col = seg * LNR_SEGMENT_COLS + tid;

        float v1 = 0.0f;
        float v2 = 0.0f;

        if (col < n) {
            float xv = load_storage_elem(x, dtype, size_t(x_row) * x_stride + col);
            float dyv = load_storage_elem(dy, dtype, size_t(dy_row) * dy_stride + col);
            float gv = load_storage_elem(gamma, dtype, size_t(param_id) * gamma_stride + col);

            float xhat;
            float v = rn_mul(dyv, gv);

            if (op_kind == LNR_OP_BWD_LN) {
                xhat = rn_mul(rn_sub(xv, mean), rstd);
                v1 = v;
                v2 = rn_mul(v, xhat);
            } else {
                xhat = rn_mul(xv, rstd);
                v1 = 0.0f;
                v2 = rn_mul(v, xhat);
            }
        }

        sh1[tid] = v1;
        sh2[tid] = v2;
        __syncthreads();

        for (uint32_t stride = 1u; stride < LNR_SEGMENT_COLS; stride <<= 1u) {
            if ((tid % (2u * stride)) == 0u) {
                sh1[tid] = rn_add(sh1[tid], sh1[tid + stride]);
                sh2[tid] = rn_add(sh2[tid], sh2[tid + stride]);
            }
            __syncthreads();
        }

        if (tid == 0u) {
            seg1[seg] = sh1[0];
            seg2[seg] = sh2[0];
        }
        __syncthreads();
    }

    if (tid < LNR_MAX_SEGMENTS) {
        if (tid >= num_segments) {
            seg1[tid] = 0.0f;
            seg2[tid] = 0.0f;
        }
    }
    __syncthreads();

    for (uint32_t stride = 1u; stride < LNR_MAX_SEGMENTS; stride <<= 1u) {
        if (tid < LNR_MAX_SEGMENTS && (tid % (2u * stride)) == 0u) {
            seg1[tid] = rn_add(seg1[tid], seg1[tid + stride]);
            seg2[tid] = rn_add(seg2[tid], seg2[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0u) {
        c1_shared = seg1[0];
        c2_shared = seg2[0];
    }
    __syncthreads();

    float c1 = c1_shared;
    float c2 = c2_shared;
    float inv_n = rn_div(1.0f, u32_to_f32(n));
    float nf = u32_to_f32(n);

    for (uint32_t col = tid; col < n; col += LNR_SEGMENT_COLS) {
        float xv = load_storage_elem(x, dtype, size_t(x_row) * x_stride + col);
        float dyv = load_storage_elem(dy, dtype, size_t(dy_row) * dy_stride + col);
        float gv = load_storage_elem(gamma, dtype, size_t(param_id) * gamma_stride + col);

        float v = rn_mul(dyv, gv);
        float out;

        if (op_kind == LNR_OP_BWD_LN) {
            float xhat = rn_mul(rn_sub(xv, mean), rstd);
            float term0 = rn_mul(nf, v);
            float term1 = rn_sub(term0, c1);
            float term2 = rn_sub(term1, rn_mul(xhat, c2));
            float scale = rn_mul(rstd, inv_n);
            out = rn_mul(scale, term2);
        } else {
            float xhat = rn_mul(xv, rstd);
            float coeff = rn_mul(inv_n, c2);
            float inner = rn_sub(v, rn_mul(xhat, coeff));
            out = rn_mul(rstd, inner);
        }

        size_t out_idx = size_t(dx_out_base + local_row) * n + col;
        store_storage_elem(dx, dtype, out_idx, out);
    }
}

__global__ void partials_stage1_kernel(
    uint8_t op_kind,
    const void* x,
    const void* dy,
    uint32_t dtype,
    uint32_t n,
    uint32_t x_stride,
    uint32_t dy_stride,
    uint32_t x_row_base,
    uint32_t dy_row_base,
    uint32_t cache_base,
    uint32_t rows,
    uint32_t partial_base,
    const float* cache_mean,
    const float* cache_rstd,
    float* partial_dgamma,
    float* partial_dbeta) {

    uint32_t col = blockIdx.x;
    uint32_t p = blockIdx.y;
    uint32_t tid = threadIdx.x; // 0..7

    __shared__ float dg[LNR_ROWS_PER_PARTIAL];
    __shared__ float db[LNR_ROWS_PER_PARTIAL];

    float dg_val = 0.0f;
    float db_val = 0.0f;

    uint32_t local_row = p * LNR_ROWS_PER_PARTIAL + tid;

    if (col < n && local_row < rows) {
        uint32_t x_row = x_row_base + local_row;
        uint32_t dy_row = dy_row_base + local_row;
        uint32_t slot = cache_base + local_row;

        float xv = load_storage_elem(x, dtype, size_t(x_row) * x_stride + col);
        float dyv = load_storage_elem(dy, dtype, size_t(dy_row) * dy_stride + col);

        float mean = cache_mean[slot];
        float rstd = cache_rstd[slot];

        if (op_kind == LNR_OP_BWD_LN) {
            float xhat = rn_mul(rn_sub(xv, mean), rstd);
            dg_val = rn_mul(dyv, xhat);
            db_val = dyv;
        } else {
            float xhat = rn_mul(xv, rstd);
            dg_val = rn_mul(dyv, xhat);
            db_val = 0.0f;
        }
    }

    dg[tid] = dg_val;
    db[tid] = db_val;
    __syncthreads();

    for (uint32_t stride = 1u; stride < LNR_ROWS_PER_PARTIAL; stride <<= 1u) {
        if ((tid % (2u * stride)) == 0u) {
            dg[tid] = rn_add(dg[tid], dg[tid + stride]);
            db[tid] = rn_add(db[tid], db[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0u && col < n) {
        size_t idx = size_t(partial_base + p) * n + col;
        partial_dgamma[idx] = lnr_canon_nan(dg[0]);
        partial_dbeta[idx] = lnr_canon_nan(db[0]);
    }
}

__global__ void partials_stage2_accum_kernel(
    uint32_t n,
    uint32_t param_id,
    uint32_t partial_base,
    uint32_t partial_count,
    const float* partial_dgamma,
    const float* partial_dbeta,
    float* accum_dgamma,
    float* accum_dbeta) {

    uint32_t col = blockIdx.x;
    uint32_t tid = threadIdx.x; // 0..1023

    __shared__ float dg[LNR_STAGE2_LEAVES];
    __shared__ float db[LNR_STAGE2_LEAVES];

    if (tid < LNR_STAGE2_LEAVES) {
        if (tid < partial_count) {
            size_t pidx = size_t(partial_base + tid) * n + col;
            dg[tid] = lnr_canon_nan(partial_dgamma[pidx]);
            db[tid] = lnr_canon_nan(partial_dbeta[pidx]);
        } else {
            dg[tid] = 0.0f;
            db[tid] = 0.0f;
        }
    }
    __syncthreads();

    for (uint32_t stride = 1u; stride < LNR_STAGE2_LEAVES; stride <<= 1u) {
        if ((tid % (2u * stride)) == 0u) {
            dg[tid] = rn_add(dg[tid], dg[tid + stride]);
            db[tid] = rn_add(db[tid], db[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0u && col < n) {
        size_t idx = size_t(param_id) * n + col;
        accum_dgamma[idx] = rn_add(accum_dgamma[idx], dg[0]);
        accum_dbeta[idx] = rn_add(accum_dbeta[idx], db[0]);
    }
}

__global__ void flush_kernel(
    uint32_t n,
    uint32_t param_id,
    uint32_t flush_out_base,
    float* accum_dgamma,
    float* accum_dbeta,
    float* flush_dgamma,
    float* flush_dbeta) {

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t col = tid; col < n; col += stride) {
        size_t acc_idx = size_t(param_id) * n + col;
        size_t out_idx = size_t(flush_out_base) * n + col;

        float dg = lnr_canon_nan(accum_dgamma[acc_idx]);
        float db = lnr_canon_nan(accum_dbeta[acc_idx]);

        flush_dgamma[out_idx] = dg;
        flush_dbeta[out_idx] = db;

        accum_dgamma[acc_idx] = 0.0f;
        accum_dbeta[acc_idx] = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// Host persistent state
// -----------------------------------------------------------------------------
struct RefState {
    Spec spec;

    float* d_accum_dgamma = nullptr;
    float* d_accum_dbeta = nullptr;

    float* d_cache_mean = nullptr;
    float* d_cache_rstd = nullptr;
    uint32_t* d_cache_kind = nullptr;
    uint64_t* d_cache_gen = nullptr;

    std::vector<uint32_t> h_cache_kind;
    std::vector<uint64_t> h_cache_gen;

    CounterSnapshot counters;
    std::vector<TimelineRecord> events;
};

static cudaError_t alloc_state_arrays(RefState* st) {
    const Spec& s = st->spec;
    size_t pn = size_t(s.param_count) * s.hidden_size;

    LNR_CHECK_CUDA(cudaMalloc(&st->d_accum_dgamma, pn * sizeof(float)));
    LNR_CHECK_CUDA(cudaMalloc(&st->d_accum_dbeta,  pn * sizeof(float)));

    LNR_CHECK_CUDA(cudaMalloc(&st->d_cache_mean, s.max_cache_rows * sizeof(float)));
    LNR_CHECK_CUDA(cudaMalloc(&st->d_cache_rstd, s.max_cache_rows * sizeof(float)));
    LNR_CHECK_CUDA(cudaMalloc(&st->d_cache_kind, s.max_cache_rows * sizeof(uint32_t)));
    LNR_CHECK_CUDA(cudaMalloc(&st->d_cache_gen,  s.max_cache_rows * sizeof(uint64_t)));

    st->h_cache_kind.assign(s.max_cache_rows, LNR_CACHE_INVALID);
    st->h_cache_gen.assign(s.max_cache_rows, 0ull);
    st->events.resize(s.max_ops_per_run);

    return cudaSuccess;
}

static cudaError_t reset_state_arrays(RefState* st, cudaStream_t stream) {
    const Spec& s = st->spec;
    size_t pn = size_t(s.param_count) * s.hidden_size;

    LNR_CHECK_CUDA(cudaMemsetAsync(st->d_accum_dgamma, 0, pn * sizeof(float), stream));
    LNR_CHECK_CUDA(cudaMemsetAsync(st->d_accum_dbeta,  0, pn * sizeof(float), stream));

    LNR_CHECK_CUDA(cudaMemsetAsync(st->d_cache_mean, 0, s.max_cache_rows * sizeof(float), stream));
    LNR_CHECK_CUDA(cudaMemsetAsync(st->d_cache_rstd, 0, s.max_cache_rows * sizeof(float), stream));
    LNR_CHECK_CUDA(cudaMemsetAsync(st->d_cache_kind, 0, s.max_cache_rows * sizeof(uint32_t), stream));
    LNR_CHECK_CUDA(cudaMemsetAsync(st->d_cache_gen,  0, s.max_cache_rows * sizeof(uint64_t), stream));

    std::fill(st->h_cache_kind.begin(), st->h_cache_kind.end(), LNR_CACHE_INVALID);
    std::fill(st->h_cache_gen.begin(), st->h_cache_gen.end(), 0ull);

    uint64_t seed = s.counter_seed;
    st->counters.run_count = seed;
    st->counters.op_count = seed;
    st->counters.save_ln_rows = seed;
    st->counters.save_rms_rows = seed;
    st->counters.bwd_ln_rows = seed;
    st->counters.bwd_rms_rows = seed;
    st->counters.partial_blocks = seed;
    st->counters.flush_count = seed;
    st->counters.invalid_ops = seed;
    st->counters.cache_miss_ops = seed;
    st->counters.cache_overwrite_rows = seed;
    st->counters.cache_generation_counter = seed;
    st->counters.last_run_id = 0ull;

    return cudaSuccess;
}

static void free_state_arrays(RefState* st) {
    if (!st) return;
    cudaFree(st->d_accum_dgamma);
    cudaFree(st->d_accum_dbeta);
    cudaFree(st->d_cache_mean);
    cudaFree(st->d_cache_rstd);
    cudaFree(st->d_cache_kind);
    cudaFree(st->d_cache_gen);
}

// -----------------------------------------------------------------------------
// ABI implementation
// -----------------------------------------------------------------------------
extern "C" size_t solution_workspace_bytes(const Spec* spec) {
    if (!valid_spec_host(spec)) return 0u;

    uint32_t max_partials = ceil_div_u32(
        spec->max_backward_rows_per_run,
        LNR_ROWS_PER_PARTIAL);

    size_t bytes = size_t(2) * max_partials * spec->hidden_size * sizeof(float);
    bytes = (bytes + 255u) & ~size_t(255u);
    return bytes;
}

extern "C" cudaError_t solution_init(
    const Spec* spec,
    void** state,
    cudaStream_t stream) {

    if (!state || !valid_spec_host(spec)) return cudaErrorInvalidValue;

    RefState* st = new (std::nothrow) RefState();
    if (!st) return cudaErrorMemoryAllocation;

    st->spec = *spec;

    cudaError_t e = alloc_state_arrays(st);
    if (e != cudaSuccess) {
        free_state_arrays(st);
        delete st;
        return e;
    }

    e = reset_state_arrays(st, stream);
    if (e != cudaSuccess) {
        free_state_arrays(st);
        delete st;
        return e;
    }

    *state = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {

    if (!state) return cudaErrorInvalidValue;
    RefState* st = reinterpret_cast<RefState*>(state);
    return reset_state_arrays(st, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    RefState* st = reinterpret_cast<RefState*>(state);
    free_state_arrays(st);
    delete st;
}

static bool top_level_run_valid(
    const RefState* st,
    const RunSpec* run,
    const InputPtrs* in,
    const OutputPtrs* out,
    const void* workspace,
    size_t workspace_bytes) {

    const Spec& s = st->spec;

    if (!run || !in || !out) return false;
    if (workspace_bytes < solution_workspace_bytes(&s)) return false;
    if (solution_workspace_bytes(&s) > 0u && !workspace) return false;

    if (run->op_count > s.max_ops_per_run) return false;
    if (run->input_rows > s.max_input_rows_per_run) return false;
    if (run->dx_rows > s.max_dx_rows_per_run) return false;
    if (run->flush_records > s.max_flush_records_per_run) return false;

    if (run->x_stride_elems < s.hidden_size) return false;
    if (run->dy_stride_elems < s.hidden_size) return false;
    if (run->gamma_stride_elems < s.hidden_size) return false;

    if (run->op_count > 0u && !run->ops) return false;
    if (!in->x || !in->dy || !in->gamma) return false;

    if (run->dx_rows > 0u && !out->dx) return false;
    if (run->flush_records > 0u && (!out->flush_dgamma || !out->flush_dbeta)) return false;

    if (!out->accum_dgamma_snapshot || !out->accum_dbeta_snapshot) return false;
    if (!out->cache_mean_snapshot || !out->cache_rstd_snapshot) return false;
    if (!out->cache_kind_snapshot || !out->cache_gen_snapshot) return false;
    if (run->op_count > 0u && !out->timeline) return false;
    if (!out->counters) return false;

    return true;
}

extern "C" cudaError_t solution_run(
    void* state,
    const RunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {

    if (!state) return cudaErrorInvalidValue;

    RefState* st = reinterpret_cast<RefState*>(state);
    const Spec& s = st->spec;
    const InputPtrs* in = reinterpret_cast<const InputPtrs*>(inputs);
    OutputPtrs* out = reinterpret_cast<OutputPtrs*>(outputs);

    if (!top_level_run_valid(st, run, in, out, workspace, workspace_bytes)) {
        return cudaErrorInvalidValue;
    }

    const uint32_t n = s.hidden_size;
    const uint32_t dtype = s.storage_dtype;
    const size_t elem_bytes = dtype_size(dtype);

    float* partial_dgamma = reinterpret_cast<float*>(workspace);
    uint32_t max_partials = ceil_div_u32(s.max_backward_rows_per_run, LNR_ROWS_PER_PARTIAL);
    float* partial_dbeta = partial_dgamma + size_t(max_partials) * n;

    // Required output zeroing.
    if (run->dx_rows > 0u) {
        LNR_CHECK_CUDA(cudaMemsetAsync(
            out->dx, 0, size_t(run->dx_rows) * n * elem_bytes, stream));
    }
    if (run->flush_records > 0u) {
        LNR_CHECK_CUDA(cudaMemsetAsync(
            out->flush_dgamma, 0, size_t(run->flush_records) * n * sizeof(float), stream));
        LNR_CHECK_CUDA(cudaMemsetAsync(
            out->flush_dbeta, 0, size_t(run->flush_records) * n * sizeof(float), stream));
    }

    st->counters.run_count += 1ull;
    st->counters.last_run_id = run->run_id;

    uint32_t partial_cursor = 0u;
    uint32_t valid_bwd_rows_this_run = 0u;

    for (uint32_t oi = 0; oi < run->op_count; ++oi) {
        const OpDesc& op = run->ops[oi];

        TimelineRecord ev{};
        ev.run_id = run->run_id;
        ev.op_index_in_run = oi;
        ev.kind = op.kind;
        ev.status = LNR_STATUS_OK;
        ev.param_id = op.param_id;
        ev.rows = op.rows;
        ev.x_row_base = op.x_row_base;
        ev.dy_row_base = op.dy_row_base;
        ev.cache_base = op.cache_base;
        ev.dx_out_base = op.dx_out_base;
        ev.flush_out_base = op.flush_out_base;
        ev.partial_base = 0u;
        ev.partial_count = 0u;
        ev.cache_generation_first = 0ull;
        ev.cache_generation_last = 0ull;

        st->counters.op_count += 1ull;
        ev.global_op_index = st->counters.op_count;
        ev.counter_snapshot_after_op = st->counters.op_count;

        auto mark_invalid = [&](uint8_t status) {
            ev.status = status;
            st->counters.invalid_ops += 1ull;
        };

        if (op.kind == LNR_OP_SAVE_LN || op.kind == LNR_OP_SAVE_RMS) {
            bool ok = true;
            if (op.flags != 0u || op.reserved_kind != 0u || op.reserved0 != 0u) ok = false;
            if (op.x_row_base + op.rows < op.x_row_base) ok = false;
            if (op.cache_base + op.rows < op.cache_base) ok = false;
            if (op.x_row_base + op.rows > run->input_rows) ok = false;
            if (op.cache_base + op.rows > s.max_cache_rows) ok = false;

            if (!ok) {
                mark_invalid(LNR_STATUS_INVALID_RANGE);
            } else {
                uint32_t cache_kind = (op.kind == LNR_OP_SAVE_LN) ? LNR_CACHE_LN : LNR_CACHE_RMS;

                for (uint32_t r = 0; r < op.rows; ++r) {
                    uint32_t slot = op.cache_base + r;
                    if (st->h_cache_kind[slot] != LNR_CACHE_INVALID) {
                        st->counters.cache_overwrite_rows += 1ull;
                    }
                }

                uint64_t gen_first = st->counters.cache_generation_counter + 1ull;
                for (uint32_t r = 0; r < op.rows; ++r) {
                    uint32_t slot = op.cache_base + r;
                    st->counters.cache_generation_counter += 1ull;
                    st->h_cache_kind[slot] = cache_kind;
                    st->h_cache_gen[slot] = st->counters.cache_generation_counter;
                }

                ev.cache_generation_first = (op.rows == 0u) ? 0ull : gen_first;
                ev.cache_generation_last = (op.rows == 0u) ? 0ull : st->counters.cache_generation_counter;

                if (op.kind == LNR_OP_SAVE_LN) {
                    st->counters.save_ln_rows += uint64_t(op.rows);
                } else {
                    st->counters.save_rms_rows += uint64_t(op.rows);
                }

                if (op.rows > 0u) {
                    save_stats_kernel<<<op.rows, LNR_SEGMENT_COLS, 0, stream>>>(
                        op.kind,
                        in->x,
                        dtype,
                        n,
                        run->x_stride_elems,
                        op.x_row_base,
                        op.rows,
                        op.cache_base,
                        (op.kind == LNR_OP_SAVE_LN) ? s.eps_ln : s.eps_rms,
                        gen_first,
                        st->d_cache_mean,
                        st->d_cache_rstd,
                        st->d_cache_kind,
                        st->d_cache_gen);
                    LNR_CHECK_CUDA(cudaGetLastError());
                }
            }
        } else if (op.kind == LNR_OP_BWD_LN || op.kind == LNR_OP_BWD_RMS) {
            uint32_t required_cache =
                (op.kind == LNR_OP_BWD_LN) ? LNR_CACHE_LN : LNR_CACHE_RMS;
            ev.cache_kind_required = uint8_t(required_cache);

            bool ok = true;
            if (op.flags != 0u || op.reserved_kind != 0u || op.reserved0 != 0u) ok = false;
            if (op.param_id >= s.param_count) {
                ok = false;
                ev.status = LNR_STATUS_INVALID_PARAM;
            }
            if (op.x_row_base + op.rows < op.x_row_base) ok = false;
            if (op.dy_row_base + op.rows < op.dy_row_base) ok = false;
            if (op.cache_base + op.rows < op.cache_base) ok = false;
            if (op.dx_out_base + op.rows < op.dx_out_base) ok = false;

            if (op.x_row_base + op.rows > run->input_rows) ok = false;
            if (op.dy_row_base + op.rows > run->input_rows) ok = false;
            if (op.cache_base + op.rows > s.max_cache_rows) ok = false;

            bool output_range_ok = (op.dx_out_base + op.rows <= run->dx_rows);
            if (!output_range_ok) ok = false;

            uint32_t pcount = ceil_div_u32(op.rows, LNR_ROWS_PER_PARTIAL);
            bool partial_ok = (partial_cursor + pcount <= max_partials);
            bool row_budget_ok = (valid_bwd_rows_this_run + op.rows <= s.max_backward_rows_per_run);

            if (!partial_ok || !row_budget_ok) {
                ok = false;
                ev.status = LNR_STATUS_PARTIAL_OVERFLOW;
            }

            bool cache_ok = true;
            if (ok) {
                for (uint32_t r = 0; r < op.rows; ++r) {
                    uint32_t slot = op.cache_base + r;
                    if (st->h_cache_kind[slot] != required_cache) {
                        cache_ok = false;
                        break;
                    }
                }
            }

            if (!ok) {
                if (ev.status == LNR_STATUS_OK) {
                    mark_invalid(output_range_ok ? LNR_STATUS_INVALID_RANGE
                                                 : LNR_STATUS_OUTPUT_RANGE);
                } else {
                    st->counters.invalid_ops += 1ull;
                }
                if (output_range_ok && op.rows > 0u) {
                    uint64_t total = uint64_t(op.rows) * n;
                    uint32_t blocks = uint32_t(std::min<uint64_t>(1024, (total + 255u) / 256u));
                    zero_dx_kernel<<<blocks, 256, 0, stream>>>(
                        out->dx, dtype, n, op.rows, op.dx_out_base);
                    LNR_CHECK_CUDA(cudaGetLastError());
                }
            } else if (!cache_ok) {
                mark_invalid(LNR_STATUS_CACHE_MISS);
                st->counters.cache_miss_ops += 1ull;

                if (op.rows > 0u) {
                    uint64_t total = uint64_t(op.rows) * n;
                    uint32_t blocks = uint32_t(std::min<uint64_t>(1024, (total + 255u) / 256u));
                    zero_dx_kernel<<<blocks, 256, 0, stream>>>(
                        out->dx, dtype, n, op.rows, op.dx_out_base);
                    LNR_CHECK_CUDA(cudaGetLastError());
                }
            } else {
                ev.partial_base = partial_cursor;
                ev.partial_count = pcount;

                if (op.rows > 0u) {
                    ev.cache_generation_first = st->h_cache_gen[op.cache_base];
                    ev.cache_generation_last = st->h_cache_gen[op.cache_base + op.rows - 1u];
                }

                valid_bwd_rows_this_run += op.rows;
                partial_cursor += pcount;

                st->counters.partial_blocks += uint64_t(pcount);
                if (op.kind == LNR_OP_BWD_LN) {
                    st->counters.bwd_ln_rows += uint64_t(op.rows);
                } else {
                    st->counters.bwd_rms_rows += uint64_t(op.rows);
                }

                if (op.rows > 0u) {
                    backward_dx_kernel<<<op.rows, LNR_SEGMENT_COLS, 0, stream>>>(
                        op.kind,
                        in->x,
                        in->dy,
                        in->gamma,
                        out->dx,
                        dtype,
                        n,
                        run->x_stride_elems,
                        run->dy_stride_elems,
                        run->gamma_stride_elems,
                        op.param_id,
                        op.x_row_base,
                        op.dy_row_base,
                        op.cache_base,
                        op.rows,
                        op.dx_out_base,
                        st->d_cache_mean,
                        st->d_cache_rstd);
                    LNR_CHECK_CUDA(cudaGetLastError());

                    dim3 grid_stage1(n, pcount, 1);
                    partials_stage1_kernel<<<grid_stage1, LNR_ROWS_PER_PARTIAL, 0, stream>>>(
                        op.kind,
                        in->x,
                        in->dy,
                        dtype,
                        n,
                        run->x_stride_elems,
                        run->dy_stride_elems,
                        op.x_row_base,
                        op.dy_row_base,
                        op.cache_base,
                        op.rows,
                        ev.partial_base,
                        st->d_cache_mean,
                        st->d_cache_rstd,
                        partial_dgamma,
                        partial_dbeta);
                    LNR_CHECK_CUDA(cudaGetLastError());

                    partials_stage2_accum_kernel<<<n, LNR_STAGE2_LEAVES, 0, stream>>>(
                        n,
                        op.param_id,
                        ev.partial_base,
                        ev.partial_count,
                        partial_dgamma,
                        partial_dbeta,
                        st->d_accum_dgamma,
                        st->d_accum_dbeta);
                    LNR_CHECK_CUDA(cudaGetLastError());
                }
            }
        } else if (op.kind == LNR_OP_FLUSH) {
            bool ok = true;
            if (op.flags != 0u || op.reserved_kind != 0u || op.reserved0 != 0u) ok = false;
            if (op.param_id >= s.param_count) {
                ok = false;
                ev.status = LNR_STATUS_INVALID_PARAM;
            }
            if (op.flush_out_base >= run->flush_records) {
                ok = false;
                ev.status = LNR_STATUS_OUTPUT_RANGE;
            }

            if (!ok) {
                if (ev.status == LNR_STATUS_OK) mark_invalid(LNR_STATUS_INVALID_RANGE);
                else st->counters.invalid_ops += 1ull;
            } else {
                st->counters.flush_count += 1ull;

                uint32_t blocks = std::min<uint32_t>(1024u, ceil_div_u32(n, 256u));
                flush_kernel<<<blocks, 256, 0, stream>>>(
                    n,
                    op.param_id,
                    op.flush_out_base,
                    st->d_accum_dgamma,
                    st->d_accum_dbeta,
                    out->flush_dgamma,
                    out->flush_dbeta);
                LNR_CHECK_CUDA(cudaGetLastError());
            }
        } else {
            mark_invalid(LNR_STATUS_UNSUPPORTED_OP);
        }

        st->events[oi] = ev;
    }

    // State snapshots after all ops.
    size_t pn = size_t(s.param_count) * n;
    LNR_CHECK_CUDA(cudaMemcpyAsync(
        out->accum_dgamma_snapshot,
        st->d_accum_dgamma,
        pn * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));

    LNR_CHECK_CUDA(cudaMemcpyAsync(
        out->accum_dbeta_snapshot,
        st->d_accum_dbeta,
        pn * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));

    LNR_CHECK_CUDA(cudaMemcpyAsync(
        out->cache_mean_snapshot,
        st->d_cache_mean,
        size_t(s.max_cache_rows) * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));

    LNR_CHECK_CUDA(cudaMemcpyAsync(
        out->cache_rstd_snapshot,
        st->d_cache_rstd,
        size_t(s.max_cache_rows) * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream));

    LNR_CHECK_CUDA(cudaMemcpyAsync(
        out->cache_kind_snapshot,
        st->d_cache_kind,
        size_t(s.max_cache_rows) * sizeof(uint32_t),
        cudaMemcpyDeviceToDevice,
        stream));

    LNR_CHECK_CUDA(cudaMemcpyAsync(
        out->cache_gen_snapshot,
        st->d_cache_gen,
        size_t(s.max_cache_rows) * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice,
        stream));

    if (run->op_count > 0u) {
        LNR_CHECK_CUDA(cudaMemcpyAsync(
            out->timeline,
            st->events.data(),
            size_t(run->op_count) * sizeof(TimelineRecord),
            cudaMemcpyHostToDevice,
            stream));
    }

    LNR_CHECK_CUDA(cudaMemcpyAsync(
        out->counters,
        &st->counters,
        sizeof(CounterSnapshot),
        cudaMemcpyHostToDevice,
        stream));

    // Correctness reference: synchronize so pageable host timeline/counter copies are complete.
    LNR_CHECK_CUDA(cudaStreamSynchronize(stream));
    return cudaSuccess;
}
