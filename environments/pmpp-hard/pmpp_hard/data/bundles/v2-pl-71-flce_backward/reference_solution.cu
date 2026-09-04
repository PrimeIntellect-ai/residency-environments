// PMPP_CANARY_71_c37de04230 -- held-out canary; MUST NOT appear in any submission
// solution.cu — FLCB-StateTrain v1 implementation (independent blind solver).
// Implements the agent-visible ABI in common.h. All math follows the contract
// exactly: exact 32-lane reduction trees, chunked online softmax, BF16
// decode/encode, persistent dW/dbias accumulation, event timeline, and modulo
// 2^64 counters. Transcendentals run on-device; every +,-,*,/ uses the *_rn
// intrinsics so ordering/rounding is bit-exact under --fmad=false.

#include "flce_backward_common.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

static_assert(sizeof(FLCBSpec)      == 80,  "FLCBSpec size");
static_assert(sizeof(FLCBRunSpec)   == 48,  "FLCBRunSpec size");
static_assert(sizeof(FLCBOp)        == 48,  "FLCBOp size");
static_assert(sizeof(FLCBEvent)     == 104, "FLCBEvent size");
static_assert(sizeof(FLCBCounters)  == 184, "FLCBCounters size");
static_assert(sizeof(FLCBRunReport) == 64,  "FLCBRunReport size");

// ── BF16 primitives ────────────────────────────────────────────────────────
__device__ __forceinline__ float decode_bf16(uint16_t b) {
    uint32_t exp  = (b >> 7) & 0xFFu;
    uint32_t mant = b & 0x7Fu;
    if (exp == 0xFFu) {
        if (mant != 0u) return 0.0f;                 // NaN -> +0
        return (b & 0x8000u) ? -16.0f : 16.0f;       // -Inf -> -16, +Inf -> +16
    }
    uint32_t u = ((uint32_t)b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ uint16_t encode_bf16_rne(float f) {
    if (isnan(f)) return 0x7fc0u;
    uint32_t u = __float_as_uint(f);
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t round_bias = 0x7fffu + lsb;
    return (uint16_t)((u + round_bias) >> 16);
}

// ── Exact warp reductions ([16,8,4,2,1] tree; lane l += lane l+stride) ───────
__device__ __forceinline__ float warp_tree_sum(float p, int lane) {
    #pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
        float o = __shfl_down_sync(0xffffffffu, p, s);
        if (lane < s) p = __fadd_rn(p, o);
    }
    return __shfl_sync(0xffffffffu, p, 0);
}
__device__ __forceinline__ void warp_tree_argmax(float& bv, uint32_t& bi, int lane) {
    #pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
        float    ov = __shfl_down_sync(0xffffffffu, bv, s);
        uint32_t oi = __shfl_down_sync(0xffffffffu, bi, s);
        if (lane < s) {
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
    }
    bv = __shfl_sync(0xffffffffu, bv, 0);
    bi = __shfl_sync(0xffffffffu, bi, 0);
}

#define WPB 8                 // warps per block
#define BLK (WPB * 32)        // 256 threads

__device__ __forceinline__ int row_is_valid(int32_t t, int ignore_index, int V) {
    if (t == ignore_index) return 0;          // ignored
    if (t >= 0 && t < V)   return 1;          // valid
    return 0;                                 // invalid (treated like ignored for math)
}

// ── Kernel: logits[local_row, v] = dot(X[row], W[v]) (+ bias) ────────────────
// One block per v: stage W[v] (H bf16) in shared once, reuse across all
// chunk rows (warp w handles rows w, w+WPB, ...). The per-output reduction is
// unchanged (lane l owns h = l, l+32, ...), so bytes are identical; only W's
// global traffic drops from chunk_rows re-reads to a single staged read.
__global__ void k_logits(const uint16_t* X, const uint16_t* W, const uint16_t* bias,
                         int has_bias, int H, int V, int chunk_rows, int input_base,
                         float* logits_ws) {
    extern __shared__ uint16_t sW[];   // [H]
    int v = blockIdx.x;
    if (v >= V) return;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int h = threadIdx.x; h < H; h += blockDim.x) sW[h] = W[(size_t)v * H + h];
    __syncthreads();
    float bv = has_bias ? decode_bf16(bias[v]) : 0.0f;
    for (int local_row = warp; local_row < chunk_rows; local_row += WPB) {
        int xrow = input_base + local_row;
        const uint16_t* xr = X + (size_t)xrow * H;
        float p = 0.0f;
        for (int h = lane; h < H; h += 32)
            p = __fadd_rn(p, __fmul_rn(decode_bf16(xr[h]), decode_bf16(sW[h])));
        float dot = warp_tree_sum(p, lane);
        if (lane == 0) {
            float logit = has_bias ? __fadd_rn(dot, bv) : dot;
            logits_ws[(size_t)local_row * V + v] = logit;
        }
    }
}

// ── Kernel: online softmax per row -> loss + lse ─────────────────────────────
__global__ void k_softmax(const int32_t* target, int ignore_index, int V, int vocab_tile,
                          int chunk_rows, int input_base, int out_base, int chunk_begin,
                          int q16, const float* logits_ws, float* loss_out, float* lse_out) {
    int warp = blockIdx.x * WPB + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (warp >= chunk_rows) return;
    int local_row = warp;
    int out_index = out_base + chunk_begin + local_row;
    int32_t t = target[input_base + local_row];
    if (!row_is_valid(t, ignore_index, V)) {
        if (lane == 0) { loss_out[out_index] = 0.0f; lse_out[local_row] = 0.0f; }
        return;
    }
    int y = t;
    const float NEG_INF = -INFINITY;
    float m = NEG_INF, s = 0.0f, suml = 0.0f;
    int ntiles = (V + vocab_tile - 1) / vocab_tile;
    const float* row_logits = logits_ws + (size_t)local_row * V;
    for (int tIdx = 0; tIdx < ntiles; ++tIdx) {
        int begin = tIdx * vocab_tile;
        int end   = begin + vocab_tile; if (end > V) end = V;
        int ts = end - begin;
        // tile argmax
        float pv = NEG_INF; uint32_t pi = 0xffffffffu;
        for (int i = lane; i < ts; i += 32) {
            float v = row_logits[begin + i];
            uint32_t c = (uint32_t)(begin + i);
            if (v > pv || (v == pv && c < pi)) { pv = v; pi = c; }
        }
        warp_tree_argmax(pv, pi, lane);
        float tmax = pv;
        // tile exp-sum
        float es = 0.0f;
        for (int i = lane; i < ts; i += 32)
            es = __fadd_rn(es, expf(__fsub_rn(row_logits[begin + i], tmax)));
        es = warp_tree_sum(es, lane);
        // tile logit-sum
        float ls = 0.0f;
        for (int i = lane; i < ts; i += 32)
            ls = __fadd_rn(ls, row_logits[begin + i]);
        ls = warp_tree_sum(ls, lane);
        // online combine
        float new_m = (tmax > m) ? tmax : m;
        float left  = (m == NEG_INF) ? 0.0f : __fmul_rn(s, expf(__fsub_rn(m, new_m)));
        float right = __fmul_rn(es, expf(__fsub_rn(tmax, new_m)));
        s    = __fadd_rn(left, right);
        suml = __fadd_rn(suml, ls);
        m    = new_m;
    }
    float lse  = __fadd_rn(m, logf(s));
    float mean = __fdiv_rn(suml, (float)V);
    float eps  = __fdiv_rn((float)q16, 65536.0f);
    float ome  = __fsub_rn(1.0f, eps);
    float tlogit = row_logits[y];
    float nll    = __fsub_rn(lse, tlogit);
    float smooth = __fsub_rn(lse, mean);
    float loss = __fadd_rn(__fmul_rn(ome, nll), __fmul_rn(eps, smooth));
    if (lane == 0) { loss_out[out_index] = loss; lse_out[local_row] = lse; }
}

// ── Kernel: dlogits[local_row, v] (elementwise) ──────────────────────────────
__global__ void k_dlogits(const int32_t* target, int ignore_index, int V, int chunk_rows,
                          int input_base, int q16, float grad_scale,
                          const float* logits_ws, const float* lse_arr, float* dlogits_ws) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ntasks = chunk_rows * V;
    if (idx >= ntasks) return;
    int local_row = idx / V;
    int v         = idx % V;
    int32_t t = target[input_base + local_row];
    if (!row_is_valid(t, ignore_index, V)) {
        dlogits_ws[(size_t)local_row * V + v] = 0.0f;
        return;
    }
    int y = t;
    float eps = __fdiv_rn((float)q16, 65536.0f);
    float ome = __fsub_rn(1.0f, eps);
    float inv_v = __fdiv_rn(1.0f, (float)V);
    float logit = logits_ws[(size_t)local_row * V + v];
    float lse = lse_arr[local_row];
    float prob = expf(__fsub_rn(logit, lse));
    float base = __fmul_rn(eps, inv_v);
    float tp = (v == y) ? __fadd_rn(base, ome) : base;
    float raw = __fsub_rn(prob, tp);
    dlogits_ws[(size_t)local_row * V + v] = __fmul_rn(grad_scale, raw);
}

// ── Transpose: WT[h, v] = W[v, h]  (once per run; enables coalesced k_dx) ─────
// Tiled 32x32 shared transpose with padding to avoid bank conflicts. Pure bit
// copy of the BF16 payload, so downstream decode is byte-identical.
__global__ void k_transpose_W(const uint16_t* __restrict__ W, uint16_t* __restrict__ WT,
                              int V, int H) {
    __shared__ uint16_t tile[32][33];
    int v0 = blockIdx.x * 32;   // vocab tile
    int h0 = blockIdx.y * 32;   // hidden tile
    int vx = v0 + threadIdx.x;
    int hy = h0 + threadIdx.y;
    if (vx < V && hy < H) tile[threadIdx.y][threadIdx.x] = W[(size_t)vx * H + hy];
    __syncthreads();
    int hxo = h0 + threadIdx.x;
    int vyo = v0 + threadIdx.y;
    if (hxo < H && vyo < V) WT[(size_t)hxo * V + vyo] = tile[threadIdx.x][threadIdx.y];
}

// ── Transpose: dlogitsT[v, i] = dlogits[i, v]  (per chunk; coalesces k_dW) ────
__global__ void k_transpose_dlog(const float* __restrict__ dl, float* __restrict__ dlT,
                                 int chunk_rows, int V) {
    __shared__ float tile[32][33];
    int v0 = blockIdx.x * 32;
    int i0 = blockIdx.y * 32;
    int vx = v0 + threadIdx.x;
    int iy = i0 + threadIdx.y;
    if (vx < V && iy < chunk_rows) tile[threadIdx.y][threadIdx.x] = dl[(size_t)iy * V + vx];
    __syncthreads();
    int ixo = i0 + threadIdx.x;
    int vyo = v0 + threadIdx.y;
    if (ixo < chunk_rows && vyo < V) dlT[(size_t)vyo * chunk_rows + ixo] = tile[threadIdx.x][threadIdx.y];
}

// ── Transpose: XT[h, i] = X[input_base + i, h]  (per chunk; coalesces k_dW) ───
__global__ void k_transpose_X(const uint16_t* __restrict__ X, uint16_t* __restrict__ XT,
                              int H, int chunk_rows, int input_base) {
    __shared__ uint16_t tile[32][33];
    int h0 = blockIdx.x * 32;
    int i0 = blockIdx.y * 32;
    int hx = h0 + threadIdx.x;
    int iy = i0 + threadIdx.y;
    if (hx < H && iy < chunk_rows) tile[threadIdx.y][threadIdx.x] = X[(size_t)(input_base + iy) * H + hx];
    __syncthreads();
    int ixo = i0 + threadIdx.x;
    int hyo = h0 + threadIdx.y;
    if (ixo < chunk_rows && hyo < H) XT[(size_t)hyo * chunk_rows + ixo] = tile[threadIdx.x][threadIdx.y];
}

// ── Kernel: dX[out_row, h] = sum_v dlogit[v] * W[v,h]  (coalesced via WT) ─────
// Reduction order preserved: lane l owns v = l, l+32, ... reads WT[h, v] which
// is a bit-identical copy of W[v, h]; identical sum -> identical bytes.
__global__ void k_dx(const uint16_t* WT, const int32_t* target, int ignore_index,
                     int H, int V, int chunk_rows, int input_base, int out_base,
                     int chunk_begin, const float* dlogits_ws, uint16_t* dx_out) {
    int warp = blockIdx.x * WPB + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int ntasks = chunk_rows * H;
    if (warp >= ntasks) return;
    int local_row = warp / H;
    int h         = warp % H;
    int out_index = out_base + chunk_begin + local_row;
    int32_t t = target[input_base + local_row];
    if (!row_is_valid(t, ignore_index, V)) {
        if (lane == 0) dx_out[(size_t)out_index * H + h] = 0x0000u;
        return;
    }
    const float* row_dl = dlogits_ws + (size_t)local_row * V;
    const uint16_t* wrow = WT + (size_t)h * V;   // WT[h, :] contiguous in v
    float p = 0.0f;
    for (int v = lane; v < V; v += 32) {
        float wv = decode_bf16(wrow[v]);
        p = __fadd_rn(p, __fmul_rn(row_dl[v], wv));
    }
    float dx = warp_tree_sum(p, lane);
    if (lane == 0) dx_out[(size_t)out_index * H + h] = encode_bf16_rne(dx);
}

// ── Kernel: dW accumulate  (one warp per v, loops all h) ─────────────────────
// Collapses V*H warp-launches to V warps: each warp loads its dlT[v] partials
// once (lane l owns i = l, l+32, ...) and reuses them across every h, so the
// per-(v,h) reduction is the identical lane-strided accumulate + [16..1] tree.
// XT[h] (tiny) streams from L2. Byte-identical to the elementwise formulation.
__global__ void k_dW(const uint16_t* XT, int H, int V, int chunk_rows,
                     const float* dlogitsT, float* acc_dW) {
    int v = blockIdx.x * WPB + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (v >= V) return;
    const float* dlrow = dlogitsT + (size_t)v * chunk_rows;  // dlT[v, :]

    // Cache this lane's owned dlogit elements (i = lane, lane+32, ...).
    float dl_cache[4];   // chunk_rows <= 128 -> at most 4 per lane
    int   ncache = 0;
    for (int i = lane; i < chunk_rows; i += 32) dl_cache[ncache++] = dlrow[i];

    float* accW_row = acc_dW + (size_t)v * H;
    for (int h = 0; h < H; ++h) {
        const uint16_t* xrow = XT + (size_t)h * chunk_rows;
        float p = 0.0f;
        int k = 0;
        for (int i = lane; i < chunk_rows; i += 32)
            p = __fadd_rn(p, __fmul_rn(dl_cache[k++], decode_bf16(xrow[i])));
        float g = warp_tree_sum(p, lane);
        if (lane == 0) accW_row[h] = __fadd_rn(accW_row[h], g);
    }
}

// ── Kernel: dbias accumulate (per chunk, per v) ──────────────────────────────
__global__ void k_dbias(int V, int chunk_rows, const float* dlogits_ws, float* acc_dB) {
    int warp = blockIdx.x * WPB + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (warp >= V) return;
    int v = warp;
    float p = 0.0f;
    for (int i = lane; i < chunk_rows; i += 32)
        p = __fadd_rn(p, dlogits_ws[(size_t)i * V + v]);
    float g = warp_tree_sum(p, lane);
    if (lane == 0) acc_dB[v] = __fadd_rn(acc_dB[v], g);
}

// ── State ───────────────────────────────────────────────────────────────────
struct State {
    FLCBSpec     spec;
    float*       d_accW;   // [V*H]
    float*       d_accB;   // [V]
    float*       d_lse;    // [row_chunk]
    uint16_t*    d_WT;     // [H*V] transposed W (bit copy), rebuilt per run
    float*       d_dlogT;  // [row_chunk*V] transposed dlogits, per chunk
    uint16_t*    d_XT;     // [row_chunk*H] transposed X chunk, per chunk
    FLCBCounters ctr;      // persistent host copy
};

static bool spec_valid(const FLCBSpec* s) {
    if (!s) return false;
    if (s->magic != FLCB_MAGIC || s->version != FLCB_VERSION) return false;
    if (s->H < 1 || s->V < 2 || s->max_ops < 1 || s->row_chunk < 1 || s->vocab_tile < 1) return false;
    if (s->has_bias != 0 && s->has_bias != 1) return false;
    if (s->flags != 0) return false;
    for (int i = 0; i < 8; ++i) if (s->reserved[i] != 0) return false;
    return true;
}

static size_t required_ws(const FLCBSpec* s) {
    size_t raw = (size_t)2 * s->row_chunk * s->V * sizeof(float);
    return (raw + 255) & ~((size_t)255);
}

extern "C" size_t solution_workspace_bytes(const FLCBSpec* spec) {
    if (!spec_valid(spec)) return 0;
    return required_ws(spec);
}

extern "C" cudaError_t solution_init(const FLCBSpec* spec, void** state, cudaStream_t stream) {
    if (!spec_valid(spec) || !state) return cudaErrorInvalidValue;
    State* st = new State();
    st->spec = *spec;
    memset(&st->ctr, 0, sizeof(st->ctr));
    size_t wbytes = (size_t)spec->V * spec->H * sizeof(float);
    cudaError_t e;
    if ((e = cudaMalloc(&st->d_accW, wbytes)) != cudaSuccess) { delete st; return e; }
    if ((e = cudaMalloc(&st->d_accB, (size_t)spec->V * sizeof(float))) != cudaSuccess) { cudaFree(st->d_accW); delete st; return e; }
    if ((e = cudaMalloc(&st->d_lse, (size_t)spec->row_chunk * sizeof(float))) != cudaSuccess) { cudaFree(st->d_accW); cudaFree(st->d_accB); delete st; return e; }
    st->d_WT = nullptr; st->d_dlogT = nullptr; st->d_XT = nullptr;
    if ((e = cudaMalloc(&st->d_WT, (size_t)spec->V * spec->H * sizeof(uint16_t))) != cudaSuccess) { cudaFree(st->d_accW); cudaFree(st->d_accB); cudaFree(st->d_lse); delete st; return e; }
    if ((e = cudaMalloc(&st->d_dlogT, (size_t)spec->row_chunk * spec->V * sizeof(float))) != cudaSuccess) { cudaFree(st->d_accW); cudaFree(st->d_accB); cudaFree(st->d_lse); cudaFree(st->d_WT); delete st; return e; }
    if ((e = cudaMalloc(&st->d_XT, (size_t)spec->row_chunk * spec->H * sizeof(uint16_t))) != cudaSuccess) { cudaFree(st->d_accW); cudaFree(st->d_accB); cudaFree(st->d_lse); cudaFree(st->d_WT); cudaFree(st->d_dlogT); delete st; return e; }
    cudaMemsetAsync(st->d_accW, 0, wbytes, stream);
    cudaMemsetAsync(st->d_accB, 0, (size_t)spec->V * sizeof(float), stream);
    *state = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    State* st = (State*)state;
    memset(&st->ctr, 0, sizeof(st->ctr));
    cudaMemsetAsync(st->d_accW, 0, (size_t)st->spec.V * st->spec.H * sizeof(float), stream);
    cudaMemsetAsync(st->d_accB, 0, (size_t)st->spec.V * sizeof(float), stream);
    return cudaSuccess;
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    State* st = (State*)state;
    cudaFree(st->d_accW); cudaFree(st->d_accB); cudaFree(st->d_lse);
    cudaFree(st->d_WT); cudaFree(st->d_dlogT); cudaFree(st->d_XT);
    delete st;
}

// ── Host helpers ─────────────────────────────────────────────────────────────
static float sanitize_scale(uint32_t bits) {
    float f; memcpy(&f, &bits, 4);
    if (isnan(f)) return 0.0f;
    if (isinf(f)) return f > 0 ? 1.0f : -1.0f;
    return f;
}
static inline int classify(int32_t t, int32_t ignore_index, uint32_t V) {
    if (t == ignore_index) return 1;             // ignored
    if (t >= 0 && (uint32_t)t < V) return 0;     // valid
    return 2;                                    // invalid
}

extern "C" cudaError_t solution_run(void* state, const FLCBRunSpec* run,
                                    const void* inputs, void* outputs,
                                    void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    if (!state || !run || !inputs || !outputs) return cudaErrorInvalidValue;
    State* st = (State*)state;
    const FLCBSpec& sp = st->spec;
    const FLCBInputs* in = (const FLCBInputs*)inputs;
    FLCBOutputs* out = (FLCBOutputs*)outputs;

    // structural checks on run spec
    if (run->flags != 0) return cudaErrorInvalidValue;
    for (int i = 0; i < 4; ++i) if (run->reserved[i] != 0) return cudaErrorInvalidValue;
    if (run->op_count > sp.max_ops) return cudaErrorInvalidValue;
    if (run->input_rows > sp.max_run_rows) return cudaErrorInvalidValue;

    size_t req = required_ws(&sp);
    if (workspace_bytes < req) return cudaErrorInvalidValue;
    if (req > 0 && !workspace) return cudaErrorInvalidValue;

    if (run->op_count > 0 && !in->ops) return cudaErrorInvalidValue;
    if (!in->x_bf16 || !in->target || !in->w_bf16) return cudaErrorInvalidValue;
    if (sp.has_bias && !in->bias_bf16) return cudaErrorInvalidValue;
    if (!out->report || !out->counters || !out->events) return cudaErrorInvalidValue;

    // fetch ops + targets to host
    std::vector<FLCBOp> ops(run->op_count);
    if (run->op_count > 0)
        if (cudaMemcpy(ops.data(), in->ops, (size_t)run->op_count * sizeof(FLCBOp), cudaMemcpyDeviceToHost) != cudaSuccess)
            return cudaErrorInvalidValue;
    std::vector<int32_t> tgt(run->input_rows);
    if (run->input_rows > 0)
        if (cudaMemcpy(tgt.data(), in->target, (size_t)run->input_rows * sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            return cudaErrorInvalidValue;

    const uint32_t rc = sp.row_chunk;
    const uint32_t V  = sp.V;
    const uint32_t H  = sp.H;
    const uint32_t num_vocab_tiles = (V + sp.vocab_tile - 1) / sp.vocab_tile;

    // preflight: classify ops, count needed outputs/events/flushes
    struct OpInfo { int kind; bool valid; uint32_t flags; };
    std::vector<OpInfo> info(run->op_count);
    uint32_t need_out = 0, need_flush = 0, need_events = 2;
    for (uint32_t j = 0; j < run->op_count; ++j) {
        const FLCBOp& op = ops[j];
        OpInfo oi{}; oi.flags = 0;
        if (op.opcode == FLCB_OP_MICRO) {
            oi.kind = 1;
            uint64_t end = (uint64_t)op.row_offset + op.row_count;
            if (end > run->input_rows) oi.flags |= FLCB_BAD_RANGE;
            if (op.label_smoothing_q16 > 65536u) oi.flags |= FLCB_BAD_EPS;
            if (op.reduction != FLCB_RED_SUM && op.reduction != FLCB_RED_MEAN_VALID) oi.flags |= FLCB_BAD_REDUCTION;
            oi.valid = (oi.flags == 0);
            if (oi.valid) { need_out += op.row_count; need_events += 2u + (op.row_count + rc - 1) / rc; }
            else          { need_events += 1u; }
        } else if (op.opcode == FLCB_OP_FLUSH) {
            oi.kind = 2;
            if (op.aux != FLCB_FLUSH_SNAPSHOT && op.aux != FLCB_FLUSH_EMIT_AND_ZERO) oi.flags |= FLCB_BAD_FLUSH;
            oi.valid = (oi.flags == 0);
            if (oi.valid) need_flush += 1;
            need_events += 1u;
        } else if (op.opcode == FLCB_OP_BUMP) {
            oi.kind = 3;
            if (op.aux > 3u) oi.flags |= FLCB_BAD_BUMP;
            oi.valid = (oi.flags == 0);
            need_events += 1u;
        } else {
            oi.kind = 0; oi.flags = FLCB_BAD_OPCODE; oi.valid = false;
            need_events += 1u;
        }
        info[j] = oi;
    }

    if (need_out > run->output_row_capacity) return cudaErrorInvalidValue;
    if (need_flush > run->flush_capacity)    return cudaErrorInvalidValue;
    if (need_events > run->event_capacity)   return cudaErrorInvalidValue;
    if (need_out > 0 && (!out->dx_bf16 || !out->loss_f32)) return cudaErrorInvalidValue;
    if (need_flush > 0 && (!out->flush_dW_f32 || !out->flush_dbias_f32)) return cudaErrorInvalidValue;

    // ── execute ──────────────────────────────────────────────────────────────
    float* logits_ws = (float*)workspace;
    float* dlog_ws   = logits_ws + (size_t)rc * V;

    FLCBCounters ctr = st->ctr;
    std::vector<FLCBEvent> evs;
    evs.reserve(need_events);
    uint32_t out_cursor = 0, flush_cursor = 0;

    auto emit = [&](uint32_t type, uint32_t op_index, uint32_t chunk_index, uint32_t flags,
                    uint64_t tag, uint64_t row_begin, uint64_t row_count,
                    uint64_t valid_c, uint64_t ignored_c, uint64_t invalid_c) {
        ctr.events_total += 1ull;
        FLCBEvent e{};
        e.type = type; e.op_index = op_index; e.chunk_index = chunk_index; e.flags = flags;
        e.run_id = run->run_id; e.tag = tag;
        e.row_begin = row_begin; e.row_count = row_count;
        e.valid_count = valid_c; e.ignored_count = ignored_c; e.invalid_count = invalid_c;
        e.counter_rows_after = ctr.rows_total;
        e.counter_valid_after = ctr.valid_rows_total;
        e.flush_generation_after = ctr.flush_generation;
        e.event_serial = ctr.events_total;
        evs.push_back(e);
    };

    ctr.runs += 1ull;
    ctr.last_run_id = run->run_id;
    emit(FLCB_EV_RUN_BEGIN, 0xffffffffu, 0xffffffffu, 0, 0, 0, 0, 0, 0, 0);

    // WT[h,v] = W[v,h] bit-copy, built once per run (reused by every k_dx).
    // Only needed when at least one valid MICRO op will run.
    bool any_micro = false;
    for (uint32_t j = 0; j < run->op_count; ++j) if (info[j].kind == 1 && info[j].valid) { any_micro = true; break; }
    if (any_micro) {
        dim3 tb(32, 32);
        dim3 tg((V + 31) / 32, (H + 31) / 32);
        k_transpose_W<<<tg, tb, 0, stream>>>(in->w_bf16, st->d_WT, V, H);
    }

    for (uint32_t j = 0; j < run->op_count; ++j) {
        const FLCBOp& op = ops[j];
        const OpInfo& oi = info[j];
        ctr.ops_seen += 1ull;

        if (oi.kind == 1 && oi.valid) {
            ctr.micro_ops += 1ull;
            // op-level counts
            uint64_t vc = 0, ic = 0, xc = 0;
            for (uint32_t r = 0; r < op.row_count; ++r) {
                int k = classify(tgt[op.row_offset + r], sp.ignore_index, V);
                if (k == 0) vc++; else if (k == 1) ic++; else xc++;
            }
            emit(FLCB_EV_MICRO_BEGIN, j, 0xffffffffu, 0, op.tag,
                 op.row_offset, op.row_count, vc, ic, xc);

            // grad scale
            float gs;
            if (op.reduction == FLCB_RED_SUM) gs = sanitize_scale(op.loss_scale_bits);
            else gs = (vc == 0) ? 0.0f : (sanitize_scale(op.loss_scale_bits) / (float)vc);

            uint32_t nchunks = (op.row_count + rc - 1) / rc;
            for (uint32_t c = 0; c < nchunks; ++c) {
                uint32_t chunk_begin = c * rc;
                uint32_t chunk_rows  = op.row_count - chunk_begin;
                if (chunk_rows > rc) chunk_rows = rc;
                uint32_t input_base = op.row_offset + chunk_begin;

                int crv = chunk_rows * V;
                k_logits<<<V, BLK, (size_t)H * sizeof(uint16_t), stream>>>(
                    in->x_bf16, in->w_bf16, in->bias_bf16, sp.has_bias, H, V,
                    chunk_rows, input_base, logits_ws);
                k_softmax<<<(chunk_rows + WPB - 1) / WPB, BLK, 0, stream>>>(
                    in->target, sp.ignore_index, V, sp.vocab_tile, chunk_rows,
                    input_base, out_cursor, chunk_begin, op.label_smoothing_q16,
                    logits_ws, out->loss_f32, st->d_lse);
                k_dlogits<<<(crv + BLK - 1) / BLK, BLK, 0, stream>>>(
                    in->target, sp.ignore_index, V, chunk_rows, input_base,
                    op.label_smoothing_q16, gs, logits_ws, st->d_lse, dlog_ws);
                int crh = chunk_rows * H;
                // dX via WT (coalesced over v).
                k_dx<<<(crh + WPB - 1) / WPB, BLK, 0, stream>>>(
                    st->d_WT, in->target, sp.ignore_index, H, V, chunk_rows,
                    input_base, out_cursor, chunk_begin, dlog_ws, out->dx_bf16);
                // Transpose dlogits and the X chunk so dW reduces over i coalesced.
                {
                    dim3 tb(32, 32);
                    k_transpose_dlog<<<dim3((V + 31) / 32, (chunk_rows + 31) / 32), tb, 0, stream>>>(
                        dlog_ws, st->d_dlogT, chunk_rows, V);
                    k_transpose_X<<<dim3((H + 31) / 32, (chunk_rows + 31) / 32), tb, 0, stream>>>(
                        in->x_bf16, st->d_XT, H, chunk_rows, input_base);
                }
                k_dW<<<(V + WPB - 1) / WPB, BLK, 0, stream>>>(
                    st->d_XT, H, V, chunk_rows, st->d_dlogT, st->d_accW);
                if (sp.has_bias)
                    k_dbias<<<(V + WPB - 1) / WPB, BLK, 0, stream>>>(
                        V, chunk_rows, dlog_ws, st->d_accB);

                ctr.row_chunks += 1ull;
                ctr.vocab_tile_updates += (uint64_t)chunk_rows * num_vocab_tiles;

                // chunk-local counts
                uint64_t cvc = 0, cic = 0, cxc = 0;
                for (uint32_t r = 0; r < chunk_rows; ++r) {
                    int k = classify(tgt[input_base + r], sp.ignore_index, V);
                    if (k == 0) cvc++; else if (k == 1) cic++; else cxc++;
                }
                emit(FLCB_EV_CHUNK_DONE, j, c, 0, op.tag,
                     input_base, chunk_rows, cvc, cic, cxc);
            }

            ctr.rows_total += op.row_count;
            ctr.valid_rows_total += vc;
            ctr.ignored_rows_total += ic;
            ctr.invalid_targets_total += xc;
            ctr.output_rows_total += op.row_count;
            emit(FLCB_EV_MICRO_DONE, j, 0xffffffffu, 0, op.tag,
                 op.row_offset, op.row_count, vc, ic, xc);
            out_cursor += op.row_count;

        } else if (oi.kind == 2 && oi.valid) {
            cudaMemcpyAsync(out->flush_dW_f32 + (size_t)flush_cursor * V * H,
                            st->d_accW, (size_t)V * H * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out->flush_dbias_f32 + (size_t)flush_cursor * V,
                            st->d_accB, (size_t)V * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            ctr.flush_ops += 1ull;
            flush_cursor += 1;
            if (op.aux == FLCB_FLUSH_EMIT_AND_ZERO) {
                cudaMemsetAsync(st->d_accW, 0, (size_t)V * H * sizeof(float), stream);
                cudaMemsetAsync(st->d_accB, 0, (size_t)V * sizeof(float), stream);
                ctr.flush_generation += 1ull;
            }
            emit(FLCB_EV_FLUSH, j, 0xffffffffu, op.aux, op.tag, 0, 0, 0, 0, 0);

        } else if (oi.kind == 3 && oi.valid) {
            switch (op.aux) {
                case FLCB_BUMP_ROWS_TOTAL:         ctr.rows_total += op.bump_amount; break;
                case FLCB_BUMP_VALID_ROWS_TOTAL:   ctr.valid_rows_total += op.bump_amount; break;
                case FLCB_BUMP_VOCAB_TILE_UPDATES: ctr.vocab_tile_updates += op.bump_amount; break;
                case FLCB_BUMP_EVENTS_TOTAL:       ctr.events_total += op.bump_amount; break;
            }
            emit(FLCB_EV_BUMP, j, 0xffffffffu, op.aux, op.tag, 0, op.bump_amount, 0, 0, 0);

        } else {
            ctr.invalid_ops += 1ull;
            emit(FLCB_EV_INVALID_OP, j, 0xffffffffu, oi.flags, op.tag,
                 op.row_offset, op.row_count, 0, 0, 0);
        }
    }

    emit(FLCB_EV_RUN_END, 0xffffffffu, 0xffffffffu, 0, 0, 0, 0, 0, 0, 0);

    st->ctr = ctr;

    // write outputs
    FLCBRunReport rep{};
    rep.status = 0;
    rep.output_rows = out_cursor;
    rep.flushes_written = flush_cursor;
    rep.events_written = (uint32_t)evs.size();
    rep.required_workspace_bytes = req;

    cudaMemcpyAsync(out->counters, &ctr, sizeof(ctr), cudaMemcpyHostToDevice, stream);
    if (!evs.empty())
        cudaMemcpyAsync(out->events, evs.data(), evs.size() * sizeof(FLCBEvent),
                        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(out->report, &rep, sizeof(rep), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return cudaSuccess;
}
