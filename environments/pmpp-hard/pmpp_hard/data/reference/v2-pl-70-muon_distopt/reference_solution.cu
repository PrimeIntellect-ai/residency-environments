// PMPP_CANARY_70_9427c377eb -- held-out canary; MUST NOT appear in any submission
// file: reference_solution.cu
// FAST, BIT-EXACT GPU reference for muon_distopt_stateful_v2.
//
// This reproduces the EXACT golden bytes of the slow single-thread oracle
// (kept as reference_oracle.cu), but parallelizes execution so it is fast
// enough to serve as the perf-gate reference on large Muon params.
//
// Bit-exactness strategy (determinism is about ORDER, not serialization):
//   * Every FP32 primitive (rn_add/sub/mul/div/sqrt) is identical to the
//     oracle, so per-op rounding matches.
//   * The Newton-Schulz quintic matmuls are computed one-thread-per-output
//     element, each thread performing the SAME canonical pairwise reduction
//     tree over the contraction dimension (an explicit balanced-tree stack
//     that is provably identical to the oracle's level-by-level pairwise_sum,
//     including the +0.0 zero padding to next_pow2(K)).  This is embarrassingly
//     parallel across output elements and params.
//   * Elementwise steps (momentum, candidate, normalize, B = b*A + c*AA,
//     X_new = a*X + BX, weight-decay apply, AdamW) are parallel one-thread-
//     per-element, identical arithmetic.
//   * All ORDER-sensitive bookkeeping (op validation/timeline, simulated
//     reduce-scatter accumulation, Frobenius pairwise norms, every FNV hash,
//     counters, output header) is computed serially in <<<1,1>>> kernels with
//     the EXACT same byte stream as the oracle -- these are O(numel), not the
//     O(numel*K) matmul cost, so they are not the bottleneck.
//
// No coefficient, iteration count, rounding point, transpose rule,
// reduce-scatter order, or AdamW math is changed -- only execution is
// parallelized.

#include "muon_distopt_common.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define CUDA_CHECK_RET(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) return _e; } while (0)

// ---------------------------------------------------------------------------
// Persistent device state (extended with scratch + cross-kernel scalars).
// ---------------------------------------------------------------------------
struct DeviceState {
    Spec spec;
    uint32_t path[MD_MAX_PARAMS];
    uint32_t offsets[MD_MAX_PARAMS];
    uint32_t numel[MD_MAX_PARAMS];

    uint64_t total_numel;
    uint64_t max_param_numel;
    uint64_t max_muon_work_elems;
    uint64_t max_red_elems;
    uint64_t max_MM;   // max over Muon params of M*M (M = min(rows,cols))

    uint16_t* weights;
    float* muon_m;
    float* adam_m;
    float* adam_v;

    uint64_t param_step[MD_MAX_PARAMS];
    float beta1_pow[MD_MAX_PARAMS];
    float beta2_pow[MD_MAX_PARAMS];

    uint64_t attempted_runs;
    uint64_t successful_runs;
    uint64_t reset_count;
    uint64_t total_valid_ops;
    uint64_t total_invalid_ops;
    uint64_t total_duplicate_ops;
    uint64_t total_nan_inputs;
    uint64_t total_inf_inputs;
    uint64_t total_muon_param_updates;
    uint64_t total_adam_param_updates;

    // ---- scratch buffers (own allocations, NOT the passed workspace) ----
    float*    grad;      // [total_numel]
    uint16_t* X;         // [max_param_numel]
    uint16_t* A;         // [max_MM]
    uint16_t* AA;        // [max_MM]
    uint16_t* B;         // [max_MM]
    uint16_t* BX;        // [max_param_numel]
    uint16_t* oriented;  // [max_param_numel]
    float*    red;       // [next_pow2(max_red_elems)]
    uint16_t* snapX;     // [5 * max_param_numel]  per-iter X snapshots for hashing

    // ---- cross-kernel scalars (threaded through the kernel sequence) ----
    uint64_t timeline;
    uint32_t last_valid, last_invalid, last_dup, last_nan, last_inf, last_muon, last_adam;
    float    g_denom;
    uint16_t g_normq;
    uint16_t iterq[5];
    float    g_shape_scale;
    float    g_bc1, g_bc2;

    // ---- precomputed hash slots (filled concurrently, folded serially) ----
    uint64_t h_mom, h_x[5], h_ortho, h_weight, h_am, h_av;
    uint64_t h_rank[MD_MAX_RANKS];
    uint64_t h_wall, h_state, h_counter;
};

struct HostState {
    DeviceState* d;
    uint16_t* weights;
    float* muon_m;
    float* adam_m;
    float* adam_v;
    float* grad;
    uint16_t* X;
    uint16_t* A;
    uint16_t* AA;
    uint16_t* B;
    uint16_t* BX;
    uint16_t* oriented;
    float* red;
    uint16_t* snapX;
    size_t workspace_required;
    uint64_t total_numel;
};

// ---------------------------------------------------------------------------
// Bit helpers / FP32 primitives -- IDENTICAL to the oracle.
// ---------------------------------------------------------------------------
__host__ __device__ static inline uint32_t as_u32(float x) {
    union { float f; uint32_t u; } v; v.f = x; return v.u;
}
__host__ __device__ static inline float as_f32(uint32_t x) {
    union { float f; uint32_t u; } v; v.u = x; return v.f;
}
__host__ __device__ static inline float canon_f32(float x) {
    uint32_t u = as_u32(x);
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu)) return as_f32(0x7FC00000u);
    return x;
}
__device__ static inline float rn_add(float a, float b) { return canon_f32(__fadd_rn(a, b)); }
__device__ static inline float rn_sub(float a, float b) { return canon_f32(__fadd_rn(a, -b)); }
__device__ static inline float rn_mul(float a, float b) { return canon_f32(__fmul_rn(a, b)); }
__device__ static inline float rn_div(float a, float b) { return canon_f32(__fdividef(a, b)); }
__device__ static inline float rn_sqrt(float a) {
    float y; asm("sqrt.rn.f32 %0, %1;" : "=f"(y) : "f"(a)); return canon_f32(y);
}

__host__ __device__ static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__host__ __device__ static inline uint16_t init_weight_bf16(
    uint64_t global_seed, uint64_t init_seed, uint32_t pid, uint32_t e) {
    uint64_t x = global_seed ^ init_seed
        ^ (0x9E3779B97F4A7C15ull * (uint64_t)(pid + 1u))
        ^ (0xD1B54A32D192ED03ull * (uint64_t)(e + 1u));
    x = splitmix64(x);
    uint16_t sign = (uint16_t)((x >> 63) & 1ull);
    uint16_t exp_offset = (uint16_t)((x >> 58) & 7ull);
    uint16_t exp = (uint16_t)(124u + exp_offset);
    uint16_t mant = (uint16_t)((x >> 16) & 0x7Full);
    return (uint16_t)((sign << 15) | (exp << 7) | mant);
}

__device__ static inline float bf16_to_f32(uint16_t b) {
    uint32_t exp = (b >> 7) & 0xFFu;
    uint32_t mant = b & 0x7Fu;
    if (exp == 0xFFu && mant != 0u) return as_f32(0x7FC00000u);
    return as_f32(((uint32_t)b) << 16);
}
__device__ static inline uint16_t f32_to_bf16(float x) {
    x = canon_f32(x);
    uint32_t u = as_u32(x);
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu)) return 0x7FC0u;
    uint32_t upper = u >> 16;
    uint32_t lower = u & 0xFFFFu;
    if (lower > 0x8000u || (lower == 0x8000u && (upper & 1u))) upper += 1u;
    return (uint16_t)(upper & 0xFFFFu);
}

__device__ static inline uint64_t fnv_byte(uint64_t h, uint8_t b) { h ^= (uint64_t)b; h *= MD_FNV_PRIME; return h; }
__device__ static inline uint64_t fnv_u8(uint64_t h, uint8_t v) { return fnv_byte(h, v); }
__device__ static inline uint64_t fnv_u16(uint64_t h, uint16_t v) {
    h = fnv_byte(h, (uint8_t)(v & 0xFFu));
    h = fnv_byte(h, (uint8_t)((v >> 8) & 0xFFu));
    return h;
}
__device__ static inline uint64_t fnv_u32(uint64_t h, uint32_t v) {
    for (int i = 0; i < 4; ++i) h = fnv_byte(h, (uint8_t)((v >> (8 * i)) & 0xFFu));
    return h;
}
__device__ static inline uint64_t fnv_u64(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) h = fnv_byte(h, (uint8_t)((v >> (8 * i)) & 0xFFu));
    return h;
}

__host__ __device__ static inline uint32_t path_for(const ParamDesc& p) {
    if (p.flags & MD_PARAM_FORCE_ADAMW) return MD_PATH_ADAMW;
    if ((p.flags & MD_PARAM_FORCE_MUON) && p.rows >= 2u && p.cols >= 2u) return MD_PATH_MUON;
    if (p.rows >= 2u && p.cols >= 2u) return MD_PATH_MUON;
    return MD_PATH_ADAMW;
}

__host__ __device__ static inline uint32_t shard_start(uint32_t numel, uint32_t rank, uint32_t R) {
    return (uint32_t)(((uint64_t)numel * (uint64_t)rank) / (uint64_t)R);
}
__host__ __device__ static inline uint32_t shard_end(uint32_t numel, uint32_t rank, uint32_t R) {
    return (uint32_t)(((uint64_t)numel * (uint64_t)(rank + 1u)) / (uint64_t)R);
}

__device__ static inline uint32_t op_reason(const DeviceState* st, const GradOp& op, uint32_t gvc) {
    const uint32_t R = st->spec.num_ranks;
    if (op.param_id >= st->spec.num_params) return 1u;
    if (op.src_rank >= R) return 2u;
    if (op.dst_rank >= R) return 3u;
    uint32_t n = st->numel[op.param_id];
    uint32_t ss = shard_start(n, op.dst_rank, R);
    uint32_t se = shard_end(n, op.dst_rank, R);
    uint32_t slen = se - ss;
    if (op.owner_offset > slen) return 4u;
    if (op.elem_count > slen - op.owner_offset) return 4u;
    if (op.grad_offset > gvc) return 5u;
    if (op.elem_count > gvc - op.grad_offset) return 5u;
    return 0u;
}
__device__ static inline bool op_duplicate(const DeviceState* st, const GradOp* ops, uint32_t i, uint32_t gvc) {
    const GradOp& oi = ops[i];
    if (op_reason(st, oi, gvc) != 0u) return false;
    for (uint32_t j = 0; j < i; ++j) {
        const GradOp& oj = ops[j];
        if (op_reason(st, oj, gvc) != 0u) continue;
        if (oj.param_id == oi.param_id && oj.src_rank == oi.src_rank &&
            oj.dst_rank == oi.dst_rank && oj.owner_offset == oi.owner_offset &&
            oj.elem_count == oi.elem_count && oj.grad_offset == oi.grad_offset) return true;
    }
    return false;
}

__host__ __device__ static inline uint64_t next_pow2_u64(uint64_t x) {
    if (x <= 1ull) return 1ull;
    --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32;
    return x + 1ull;
}

__device__ static uint64_t hash_bf16_array(const uint16_t* x, uint64_t n) {
    uint64_t h = MD_FNV_BASIS;
    for (uint64_t i = 0; i < n; ++i) h = fnv_u16(h, x[i]);
    return h;
}
__device__ static uint64_t hash_f32_array(const float* x, uint64_t n) {
    uint64_t h = MD_FNV_BASIS;
    for (uint64_t i = 0; i < n; ++i) h = fnv_u32(h, as_u32(canon_f32(x[i])));
    return h;
}

__device__ static uint64_t hash_rank_grad(float* grad, uint32_t base, uint32_t start, uint32_t len) {
    uint64_t h = MD_FNV_BASIS;
    for (uint32_t i = 0; i < len; ++i) h = fnv_u32(h, as_u32(canon_f32(grad[base + start + i])));
    return h;
}

// ===========================================================================
// init / reset kernels (oracle-identical semantics; offsets etc. set on host).
// ===========================================================================
__global__ static void init_kernel(DeviceState* st) {
    if (threadIdx.x || blockIdx.x) return;
    for (uint32_t pid = 0; pid < st->spec.num_params; ++pid) {
        ParamDesc& p = st->spec.params[pid];
        uint32_t n = st->numel[pid];
        uint32_t base = st->offsets[pid];
        for (uint32_t e = 0; e < n; ++e) {
            st->weights[base + e] = init_weight_bf16(st->spec.global_seed, p.init_seed, pid, e);
            st->muon_m[base + e] = as_f32(0x00000000u);
            st->adam_m[base + e] = as_f32(0x00000000u);
            st->adam_v[base + e] = as_f32(0x00000000u);
        }
        st->param_step[pid] = 0;
        st->beta1_pow[pid] = as_f32(0x3F800000u);
        st->beta2_pow[pid] = as_f32(0x3F800000u);
    }
    st->attempted_runs = 0; st->successful_runs = 0; st->reset_count = 0;
    st->total_valid_ops = 0; st->total_invalid_ops = 0; st->total_duplicate_ops = 0;
    st->total_nan_inputs = 0; st->total_inf_inputs = 0;
    st->total_muon_param_updates = 0; st->total_adam_param_updates = 0;
}

__global__ static void reset_kernel(DeviceState* st) {
    if (threadIdx.x || blockIdx.x) return;
    uint64_t old_reset = st->reset_count;
    for (uint32_t pid = 0; pid < st->spec.num_params; ++pid) {
        const ParamDesc& p = st->spec.params[pid];
        uint32_t base = st->offsets[pid];
        uint32_t n = st->numel[pid];
        for (uint32_t e = 0; e < n; ++e) {
            st->weights[base + e] = init_weight_bf16(st->spec.global_seed, p.init_seed, pid, e);
            st->muon_m[base + e] = as_f32(0x00000000u);
            st->adam_m[base + e] = as_f32(0x00000000u);
            st->adam_v[base + e] = as_f32(0x00000000u);
        }
        st->param_step[pid] = 0;
        st->beta1_pow[pid] = as_f32(0x3F800000u);
        st->beta2_pow[pid] = as_f32(0x3F800000u);
    }
    st->attempted_runs = 0; st->successful_runs = 0; st->reset_count = old_reset + 1ull;
    st->total_valid_ops = 0; st->total_invalid_ops = 0; st->total_duplicate_ops = 0;
    st->total_nan_inputs = 0; st->total_inf_inputs = 0;
    st->total_muon_param_updates = 0; st->total_adam_param_updates = 0;
}

// ===========================================================================
// Hashing helpers reused for OOM + run_end (serial).
// ===========================================================================
__device__ static uint64_t compute_weight_hash(DeviceState* st) {
    uint64_t h = MD_FNV_BASIS;
    for (uint32_t pid = 0; pid < st->spec.num_params; ++pid) {
        uint32_t base = st->offsets[pid];
        for (uint32_t e = 0; e < st->numel[pid]; ++e) h = fnv_u16(h, st->weights[base + e]);
    }
    return h;
}
__device__ static uint64_t compute_counter_hash(DeviceState* st) {
    uint64_t h = MD_FNV_BASIS;
    h = fnv_u64(h, st->attempted_runs);
    h = fnv_u64(h, st->successful_runs);
    h = fnv_u64(h, st->reset_count);
    h = fnv_u64(h, st->total_valid_ops);
    h = fnv_u64(h, st->total_invalid_ops);
    h = fnv_u64(h, st->total_duplicate_ops);
    h = fnv_u64(h, st->total_nan_inputs);
    h = fnv_u64(h, st->total_inf_inputs);
    h = fnv_u64(h, st->total_muon_param_updates);
    h = fnv_u64(h, st->total_adam_param_updates);
    return h;
}
__device__ static uint64_t compute_state_hash(DeviceState* st) {
    uint64_t h = MD_FNV_BASIS;
    h = fnv_u64(h, MD_MAGIC);
    h = fnv_u32(h, MD_VERSION);
    h = fnv_u32(h, st->spec.num_params);
    h = fnv_u32(h, st->spec.num_ranks);
    for (uint32_t pid = 0; pid < st->spec.num_params; ++pid) {
        const ParamDesc& p = st->spec.params[pid];
        uint32_t base = st->offsets[pid];
        uint32_t n = st->numel[pid];
        h = fnv_u32(h, pid);
        h = fnv_u32(h, st->path[pid]);
        h = fnv_u32(h, p.rows);
        h = fnv_u32(h, p.cols);
        h = fnv_u64(h, st->param_step[pid]);
        for (uint32_t e = 0; e < n; ++e) h = fnv_u16(h, st->weights[base + e]);
        if (st->path[pid] == MD_PATH_MUON) {
            for (uint32_t e = 0; e < n; ++e) h = fnv_u32(h, as_u32(canon_f32(st->muon_m[base + e])));
        } else {
            h = fnv_u32(h, as_u32(canon_f32(st->beta1_pow[pid])));
            h = fnv_u32(h, as_u32(canon_f32(st->beta2_pow[pid])));
            for (uint32_t e = 0; e < n; ++e) h = fnv_u32(h, as_u32(canon_f32(st->adam_m[base + e])));
            for (uint32_t e = 0; e < n; ++e) h = fnv_u32(h, as_u32(canon_f32(st->adam_v[base + e])));
        }
    }
    return h;
}

// ===========================================================================
// OOM header (oracle-identical, see README §4.1).
// ===========================================================================
__global__ static void write_oom_kernel(DeviceState* st, const RunSpec run, void* outputs) {
    if (threadIdx.x || blockIdx.x) return;
    st->attempted_runs += 1ull;
    if (!outputs) return;
    RunOutputHeader* h = (RunOutputHeader*)outputs;
    h->magic = MD_MAGIC; h->version = MD_VERSION; h->status = MD_STATUS_OOM;
    h->attempted_runs = st->attempted_runs;
    h->successful_runs = st->successful_runs;
    h->reset_count = st->reset_count;
    h->run_tag = run.run_tag;
    h->weight_hash = compute_weight_hash(st);
    h->state_hash = compute_state_hash(st);
    h->counter_hash = compute_counter_hash(st);
    h->timeline_hash = MD_FNV_BASIS;
    h->final_hash = MD_FNV_BASIS;
    h->total_valid_ops = st->total_valid_ops;
    h->total_invalid_ops = st->total_invalid_ops;
    h->total_duplicate_ops = st->total_duplicate_ops;
    h->total_nan_inputs = st->total_nan_inputs;
    h->total_inf_inputs = st->total_inf_inputs;
    h->total_muon_param_updates = st->total_muon_param_updates;
    h->total_adam_param_updates = st->total_adam_param_updates;
}

// ===========================================================================
// RUN_BEGIN + op validation/timeline events (serial, oracle byte-stream).
// ===========================================================================
__global__ static void run_begin_kernel(DeviceState* st, const RunSpec run, const void* inputs) {
    if (threadIdx.x || blockIdx.x) return;
    st->attempted_runs += 1ull;

    const GradOp* ops = (const GradOp*)inputs;

    uint32_t last_valid = 0, last_invalid = 0, last_dup = 0;

    uint64_t timeline = MD_FNV_BASIS;
    timeline = fnv_u8(timeline, 0xA0u);
    timeline = fnv_u64(timeline, run.run_tag);
    timeline = fnv_u64(timeline, st->successful_runs);
    timeline = fnv_u32(timeline, run.op_count);
    timeline = fnv_u32(timeline, run.grad_value_count);
    timeline = fnv_u32(timeline, st->spec.num_ranks);

    for (uint32_t i = 0; i < run.op_count; ++i) {
        uint32_t reason = op_reason(st, ops[i], run.grad_value_count);
        bool dup = op_duplicate(st, ops, i, run.grad_value_count);
        if (reason == 0u) {
            last_valid++;
            if (dup) last_dup++;
            timeline = fnv_u8(timeline, 0xA1u);
            timeline = fnv_u32(timeline, i);
            timeline = fnv_u32(timeline, ops[i].param_id);
            timeline = fnv_u32(timeline, ops[i].src_rank);
            timeline = fnv_u32(timeline, ops[i].dst_rank);
            timeline = fnv_u32(timeline, ops[i].owner_offset);
            timeline = fnv_u32(timeline, ops[i].elem_count);
            timeline = fnv_u32(timeline, ops[i].grad_offset);
            timeline = fnv_u32(timeline, ops[i].flags);
            timeline = fnv_u32(timeline, dup ? 1u : 0u);
        } else {
            last_invalid++;
            timeline = fnv_u8(timeline, 0xA2u);
            timeline = fnv_u32(timeline, i);
            timeline = fnv_u32(timeline, ops[i].param_id);
            timeline = fnv_u32(timeline, ops[i].src_rank);
            timeline = fnv_u32(timeline, ops[i].dst_rank);
            timeline = fnv_u32(timeline, ops[i].owner_offset);
            timeline = fnv_u32(timeline, ops[i].elem_count);
            timeline = fnv_u32(timeline, ops[i].grad_offset);
            timeline = fnv_u32(timeline, ops[i].flags);
            timeline = fnv_u32(timeline, reason);
        }
    }

    st->timeline = timeline;
    st->last_valid = last_valid;
    st->last_invalid = last_invalid;
    st->last_dup = last_dup;
    st->last_nan = 0;
    st->last_inf = 0;
    st->last_muon = 0;
    st->last_adam = 0;
}

// Simulated reduce-scatter (serial, src-then-op order, nan/inf counting).
__global__ static void reduce_scatter_kernel(DeviceState* st, const RunSpec run, const void* inputs) {
    if (threadIdx.x || blockIdx.x) return;
    const GradOp* ops = (const GradOp*)inputs;
    const uint16_t* grad_values = (const uint16_t*)((const uint8_t*)inputs + sizeof(GradOp) * run.op_count);

    uint32_t last_nan = 0, last_inf = 0;
    for (uint32_t src = 0; src < st->spec.num_ranks; ++src) {
        for (uint32_t i = 0; i < run.op_count; ++i) {
            const GradOp& op = ops[i];
            if (op.src_rank != src) continue;
            if (op_reason(st, op, run.grad_value_count) != 0u) continue;
            uint32_t pid = op.param_id;
            uint32_t n = st->numel[pid];
            uint32_t ss = shard_start(n, op.dst_rank, st->spec.num_ranks);
            uint32_t local = ss + op.owner_offset;
            uint32_t base = st->offsets[pid];
            for (uint32_t k = 0; k < op.elem_count; ++k) {
                uint16_t gb = grad_values[op.grad_offset + k];
                uint32_t exp = (gb >> 7) & 0xFFu;
                uint32_t mant = gb & 0x7Fu;
                if (exp == 0xFFu && mant != 0u) last_nan++;
                else if (exp == 0xFFu && mant == 0u) last_inf++;
                float g = bf16_to_f32(gb);
                st->grad[base + local + k] = rn_add(st->grad[base + local + k], g);
            }
        }
    }
    st->last_nan = last_nan;
    st->last_inf = last_inf;
}

// PARAM_BEGIN + REDUCE_SCATTER_RANK rank-hash events (serial).
// Concurrent per-rank grad-shard hashes -> h_rank[r] (one thread per rank).
__global__ static void rank_hash_kernel(DeviceState* st, uint32_t pid) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= st->spec.num_ranks) return;
    uint32_t base = st->offsets[pid];
    uint32_t n = st->numel[pid];
    uint32_t ss = shard_start(n, r, st->spec.num_ranks);
    uint32_t se = shard_end(n, r, st->spec.num_ranks);
    st->h_rank[r] = hash_rank_grad(st->grad, base, ss, se - ss);
}

// PARAM_BEGIN + REDUCE_SCATTER_RANK events (serial; reads precomputed h_rank).
__global__ static void param_begin_fold_kernel(DeviceState* st, uint32_t pid) {
    if (threadIdx.x || blockIdx.x) return;
    const ParamDesc& p = st->spec.params[pid];
    uint32_t n = st->numel[pid];
    uint64_t timeline = st->timeline;
    timeline = fnv_u8(timeline, 0xA3u);
    timeline = fnv_u32(timeline, pid);
    timeline = fnv_u32(timeline, st->path[pid]);
    timeline = fnv_u32(timeline, p.rows);
    timeline = fnv_u32(timeline, p.cols);
    timeline = fnv_u64(timeline, st->param_step[pid]);
    for (uint32_t r = 0; r < st->spec.num_ranks; ++r) {
        uint32_t ss = shard_start(n, r, st->spec.num_ranks);
        uint32_t se = shard_end(n, r, st->spec.num_ranks);
        timeline = fnv_u8(timeline, 0xA4u);
        timeline = fnv_u32(timeline, pid);
        timeline = fnv_u32(timeline, r);
        timeline = fnv_u32(timeline, ss);
        timeline = fnv_u32(timeline, se - ss);
        timeline = fnv_u64(timeline, st->h_rank[r]);
    }
    st->timeline = timeline;
}

// PARAM_END (serial; reads precomputed final weight hash h_weight).
__global__ static void param_end_kernel(DeviceState* st, uint32_t pid) {
    if (threadIdx.x || blockIdx.x) return;
    uint64_t timeline = st->timeline;
    timeline = fnv_u8(timeline, 0xA9u);
    timeline = fnv_u32(timeline, pid);
    timeline = fnv_u64(timeline, st->param_step[pid]);
    timeline = fnv_u64(timeline, st->h_weight);
    st->timeline = timeline;
}

// ===========================================================================
// MUON numeric kernels (parallel) + fold kernels (serial).
// ===========================================================================
// momentum update: m_new = rn_add(rn_mul(mu, m_old), g).
__global__ static void muon_momentum_kernel(DeviceState* st, uint32_t pid, float mu) {
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = st->numel[pid];
    if (e >= n) return;
    uint32_t base = st->offsets[pid];
    float g = st->grad[base + e];
    float oldm = st->muon_m[base + e];
    st->muon_m[base + e] = rn_add(rn_mul(mu, oldm), g);
}

// build initial NS matrix X[i*N+j] = bf16(candidate), candidate = g + mu*m_new.
__global__ static void muon_build_X_kernel(DeviceState* st, uint32_t pid, float mu,
                                           uint32_t M, uint32_t N, uint32_t rows, uint32_t cols) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    uint32_t i = idx / N, j = idx % N;
    uint32_t orig = (rows <= cols) ? (i * cols + j) : (j * cols + i);
    uint32_t base = st->offsets[pid];
    float g = st->grad[base + orig];
    float m = st->muon_m[base + orig];
    float cand = rn_add(g, rn_mul(mu, m));
    st->X[idx] = f32_to_bf16(cand);
}

// ---- Parallel Frobenius norm: bit-exact adjacent-pair balanced-tree ----
// square kernel writes red[i] = rn_mul(x,x) for i<numel, +0.0 for i in [numel,P);
// then log2(P) reduce-level launches collapse it; red[0] holds the sum.  This is
// the SAME tree as the serial pairwise_sum, but parallel within each level.
__global__ static void frob_square_kernel(DeviceState* st, uint32_t numel, uint32_t P) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    if (i < numel) { float x = bf16_to_f32(st->X[i]); st->red[i] = rn_mul(x, x); }
    else st->red[i] = as_f32(0x00000000u);
}
__global__ static void reduce_level_kernel(DeviceState* st, uint32_t half) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    st->red[i] = rn_add(st->red[2u * i], st->red[2u * i + 1u]);
}
// finish kernels read red[0] (= sum of squares) and produce the derived scalar.
__global__ static void frob_pre_finish_kernel(DeviceState* st, float eps) {
    if (threadIdx.x || blockIdx.x) return;
    st->g_denom = rn_add(rn_sqrt(st->red[0]), eps);
}
__global__ static void frob_normq_finish_kernel(DeviceState* st) {
    if (threadIdx.x || blockIdx.x) return;
    st->g_normq = f32_to_bf16(rn_sqrt(st->red[0]));
}
__global__ static void frob_iterq_finish_kernel(DeviceState* st, uint32_t t) {
    if (threadIdx.x || blockIdx.x) return;
    st->iterq[t] = f32_to_bf16(rn_sqrt(st->red[0]));
}

// Snapshot X (M*N bf16) into snapX[t*max_param_numel] for deferred hashing.
__global__ static void snap_X_kernel(DeviceState* st, uint32_t t, uint32_t MN) {
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= MN) return;
    st->snapX[(uint64_t)t * st->max_param_numel + e] = st->X[e];
}

// normalize: X[e] = bf16(X[e]/denom).
__global__ static void muon_normalize_kernel(DeviceState* st, uint32_t MN) {
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= MN) return;
    float denom = st->g_denom;
    st->X[e] = f32_to_bf16(rn_div(bf16_to_f32(st->X[e]), denom));
}

// Concurrent Muon array hashes (one thread per independent hash chain):
//   thread 0     -> h_mom    = hash_f32(muon_m)
//   thread 1..5  -> h_x[t]   = hash_bf16(snapX[t])   (per-iter X snapshot)
//   thread 6     -> h_ortho  = hash_bf16(oriented)
//   thread 7     -> h_weight = hash_bf16(weights)    (final; used by APPLY+END)
// All source buffers are finalized before this launch, so the chains are
// independent and run concurrently (latency-hidden), replacing ~9 serial
// FNV passes with a single wave.
__global__ static void muon_hash_kernel(DeviceState* st, uint32_t pid, uint32_t MN) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = st->offsets[pid];
    uint32_t n = st->numel[pid];
    if (t == 0) st->h_mom = hash_f32_array(st->muon_m + base, n);
    else if (t >= 1 && t <= 5) st->h_x[t - 1] = hash_bf16_array(st->snapX + (uint64_t)(t - 1) * st->max_param_numel, MN);
    else if (t == 6) st->h_ortho = hash_bf16_array(st->oriented, n);
    else if (t == 7) st->h_weight = hash_bf16_array(st->weights + base, n);
}

// Serial Muon timeline fold (reads all precomputed hash slots + scalars).
__global__ static void muon_fold_all_kernel(DeviceState* st, uint32_t pid) {
    if (threadIdx.x || blockIdx.x) return;
    uint64_t timeline = st->timeline;
    // MUON_MOMENTUM
    timeline = fnv_u8(timeline, 0xA5u);
    timeline = fnv_u32(timeline, pid);
    timeline = fnv_u64(timeline, st->h_mom);
    timeline = fnv_u16(timeline, st->g_normq);
    // MUON_NS_ITER x5
    for (uint32_t t = 0; t < 5u; ++t) {
        timeline = fnv_u8(timeline, 0xA6u);
        timeline = fnv_u32(timeline, pid);
        timeline = fnv_u32(timeline, t);
        timeline = fnv_u16(timeline, st->iterq[t]);
        timeline = fnv_u64(timeline, st->h_x[t]);
    }
    // MUON_APPLY
    timeline = fnv_u8(timeline, 0xA7u);
    timeline = fnv_u32(timeline, pid);
    timeline = fnv_u32(timeline, as_u32(st->g_shape_scale));
    timeline = fnv_u64(timeline, st->h_ortho);
    timeline = fnv_u64(timeline, st->h_weight);
    st->timeline = timeline;
}

// Parallel bit-exact bf16 matmul: ONE WARP per output element (i,j).
//
// The K products are reduced with the EXACT canonical adjacent-pair balanced
// tree of the oracle pairwise_sum (over P = next_pow2(K) leaves, +0.0 padding),
// factored across the 32 warp lanes:
//   * When P >= 32: lane l reduces its contiguous chunk [l*G, l*G+G) of
//     G = P/32 leaves into a per-lane balanced subtree (tiny register stack,
//     depth <= log2(G)), then the 32 lane partials are combined by a 5-level
//     adjacent-pair tree via __shfl_down_sync (strides 1,2,4,8,16).
//   * When P < 32: G = 1, only P lanes are active, combined by a log2(P)-level
//     adjacent-pair shuffle tree.
// Because P = 32*G (or P itself) and the tree splits cleanly, this is bit-for-bit
// identical to the serial balanced tree, but uses only registers + shuffles
// (no local-memory spill, no shared memory) so it runs at full throughput.
// Verified zero-mismatch vs the serial oracle across ragged/prime/pow2/small
// shapes.  B_transposed selects B[j*K+k] (A = X*X^T) vs B[k*N+j] (standard).
// Launch: blockDim multiple of 32; grid covers M*N warps.
__global__ static void matmul_kernel(const uint16_t* A, const uint16_t* Bm, uint16_t* C,
                                     uint32_t M, uint32_t N, uint32_t K, int B_transposed) {
    uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    uint32_t lane = threadIdx.x & 31u;
    if (warp >= M * N) return;
    uint32_t i = warp / N, j = warp % N;

    uint32_t P = (uint32_t)next_pow2_u64(K);
    uint32_t active = (P >= 32u) ? 32u : P;
    uint32_t G = (P >= 32u) ? (P >> 5) : 1u;

    float part = as_f32(0x00000000u);
    if (lane < active) {
        float sval[8];
        int   slvl[8];
        int sp = 0;
        uint32_t base = lane * G;
        for (uint32_t g = 0; g < G; ++g) {
            uint32_t k = base + g;
            float v;
            if (k < K) {
                uint16_t av = A[(uint64_t)i * K + k];
                uint16_t bv = B_transposed ? Bm[(uint64_t)j * K + k] : Bm[(uint64_t)k * N + j];
                v = rn_mul(bf16_to_f32(av), bf16_to_f32(bv));
            } else {
                v = as_f32(0x00000000u);
            }
            int lvl = 0;
            while (sp > 0 && slvl[sp - 1] == lvl) { v = rn_add(sval[sp - 1], v); sp--; lvl++; }
            sval[sp] = v; slvl[sp] = lvl; sp++;
        }
        part = sval[0];  // balanced subtree over G leaves (G is a power of two)
    }
    // Combine the `active` lane partials via adjacent-pair balanced tree.
    for (uint32_t s = 1u; s < active; s <<= 1) {
        float o = __shfl_down_sync(0xffffffffu, part, s);
        if ((lane & ((2u * s) - 1u)) == 0u) part = rn_add(part, o);
    }
    if (lane == 0u) C[(uint64_t)i * N + j] = f32_to_bf16(part);
}

// B = bf16(b*A + c*AA), elementwise over M*M.
__global__ static void muon_B_combine_kernel(DeviceState* st, uint32_t MM, float b, float c) {
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= MM) return;
    float bv = rn_add(rn_mul(b, bf16_to_f32(st->A[e])), rn_mul(c, bf16_to_f32(st->AA[e])));
    st->B[e] = f32_to_bf16(bv);
}

// X_new = bf16(a*X + BX), elementwise over M*N.
__global__ static void muon_Xnew_combine_kernel(DeviceState* st, uint32_t MN, float a) {
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= MN) return;
    float nv = rn_add(rn_mul(a, bf16_to_f32(st->X[e])), bf16_to_f32(st->BX[e]));
    st->X[e] = f32_to_bf16(nv);
}

// shape_scale = rn_sqrt(max(rows,cols)) (serial scalar).
__global__ static void muon_shape_scale_kernel(DeviceState* st, uint32_t maxdim) {
    if (threadIdx.x || blockIdx.x) return;
    st->g_shape_scale = rn_sqrt((float)maxdim);
}

// weight-decay apply (parallel over numel) + write oriented snapshot.
__global__ static void muon_apply_kernel(DeviceState* st, uint32_t pid, uint32_t rows, uint32_t cols,
                                         uint32_t N, float lr, float wd) {
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = rows * cols;
    if (e >= n) return;
    uint32_t r = e / cols, cc = e % cols;
    uint32_t base = st->offsets[pid];
    float shape_scale = st->g_shape_scale;
    uint16_t ortho_b = (rows <= cols) ? st->X[r * N + cc] : st->X[cc * N + r];
    st->oriented[e] = ortho_b;
    float ortho = bf16_to_f32(ortho_b);
    float wold = bf16_to_f32(st->weights[base + e]);
    float delta = rn_add(rn_mul(shape_scale, ortho), rn_mul(wd, wold));
    float wnew = rn_sub(wold, rn_mul(lr, delta));
    st->weights[base + e] = f32_to_bf16(wnew);
}

__global__ static void muon_step_inc_kernel(DeviceState* st, uint32_t pid) {
    if (threadIdx.x || blockIdx.x) return;
    st->param_step[pid] += 1ull;
    st->last_muon += 1u;
}

// ===========================================================================
// ADAMW numeric kernels (parallel) + scalar/fold (serial).
// ===========================================================================
__global__ static void adam_scalar_kernel(DeviceState* st, uint32_t pid, float beta1, float beta2) {
    if (threadIdx.x || blockIdx.x) return;
    st->param_step[pid] += 1ull;
    st->beta1_pow[pid] = rn_mul(st->beta1_pow[pid], beta1);
    st->beta2_pow[pid] = rn_mul(st->beta2_pow[pid], beta2);
    float one = as_f32(0x3F800000u);
    st->g_bc1 = rn_sub(one, st->beta1_pow[pid]);
    st->g_bc2 = rn_sub(one, st->beta2_pow[pid]);
    st->last_adam += 1u;
}

__global__ static void adam_apply_kernel(DeviceState* st, uint32_t pid,
                                         float beta1, float beta2, float eps, float lr, float wd) {
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = st->numel[pid];
    if (e >= n) return;
    uint32_t base = st->offsets[pid];
    float one = as_f32(0x3F800000u);
    float bc1 = st->g_bc1, bc2 = st->g_bc2;

    float g = st->grad[base + e];
    float wold = bf16_to_f32(st->weights[base + e]);
    float m = st->adam_m[base + e];
    float v = st->adam_v[base + e];

    m = rn_add(rn_mul(beta1, m), rn_mul(rn_sub(one, beta1), g));
    float gg = rn_mul(g, g);
    v = rn_add(rn_mul(beta2, v), rn_mul(rn_sub(one, beta2), gg));

    st->adam_m[base + e] = m;
    st->adam_v[base + e] = v;

    float mhat = rn_div(m, bc1);
    float vhat = rn_div(v, bc2);
    float denom = rn_add(rn_sqrt(vhat), eps);
    float adam_update = rn_div(mhat, denom);
    float delta = rn_add(adam_update, rn_mul(wd, wold));
    float wnew = rn_sub(wold, rn_mul(lr, delta));
    st->weights[base + e] = f32_to_bf16(wnew);
}

// Concurrent AdamW array hashes: thread 0 -> h_am, 1 -> h_av, 2 -> h_weight.
__global__ static void adam_hash_kernel(DeviceState* st, uint32_t pid) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = st->offsets[pid];
    uint32_t n = st->numel[pid];
    if (t == 0) st->h_am = hash_f32_array(st->adam_m + base, n);
    else if (t == 1) st->h_av = hash_f32_array(st->adam_v + base, n);
    else if (t == 2) st->h_weight = hash_bf16_array(st->weights + base, n);
}

// ADAMW_APPLY event (serial; reads precomputed hash slots).
__global__ static void adam_fold_kernel(DeviceState* st, uint32_t pid) {
    if (threadIdx.x || blockIdx.x) return;
    uint64_t timeline = st->timeline;
    timeline = fnv_u8(timeline, 0xA8u);
    timeline = fnv_u32(timeline, pid);
    timeline = fnv_u32(timeline, as_u32(st->beta1_pow[pid]));
    timeline = fnv_u32(timeline, as_u32(st->beta2_pow[pid]));
    timeline = fnv_u64(timeline, st->h_am);
    timeline = fnv_u64(timeline, st->h_av);
    timeline = fnv_u64(timeline, st->h_weight);
    st->timeline = timeline;
}

// Concurrent final weight_hash / state_hash (2 independent big chains):
// thread 0 -> h_wall (weight_hash), thread 1 -> h_state (state_hash).
// Both depend only on final weights/state (not on counters), so they run
// before run_end.  counter_hash is cheap and stays inline in run_end.
__global__ static void run_hash_kernel(DeviceState* st) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0) st->h_wall = compute_weight_hash(st);
    else if (t == 1) st->h_state = compute_state_hash(st);
}

// ===========================================================================
// RUN_END: counters, final-hash fold, header (serial; reads precomputed hashes).
// ===========================================================================
__global__ static void run_end_kernel(DeviceState* st, const RunSpec run, void* outputs) {
    if (threadIdx.x || blockIdx.x) return;
    RunOutputHeader* out = (RunOutputHeader*)outputs;

    uint32_t last_valid = st->last_valid, last_invalid = st->last_invalid, last_dup = st->last_dup;
    uint32_t last_nan = st->last_nan, last_inf = st->last_inf;
    uint32_t last_muon = st->last_muon, last_adam = st->last_adam;

    st->successful_runs += 1ull;
    st->total_valid_ops += last_valid;
    st->total_invalid_ops += last_invalid;
    st->total_duplicate_ops += last_dup;
    st->total_nan_inputs += last_nan;
    st->total_inf_inputs += last_inf;
    st->total_muon_param_updates += last_muon;
    st->total_adam_param_updates += last_adam;

    uint64_t weight_hash = st->h_wall;
    uint64_t state_hash = st->h_state;
    uint64_t counter_hash = compute_counter_hash(st);

    uint64_t timeline = st->timeline;
    timeline = fnv_u8(timeline, 0xAAu);
    timeline = fnv_u32(timeline, last_valid);
    timeline = fnv_u32(timeline, last_invalid);
    timeline = fnv_u32(timeline, last_dup);
    timeline = fnv_u32(timeline, last_nan);
    timeline = fnv_u32(timeline, last_inf);
    timeline = fnv_u32(timeline, last_muon);
    timeline = fnv_u32(timeline, last_adam);
    timeline = fnv_u64(timeline, weight_hash);
    timeline = fnv_u64(timeline, state_hash);
    timeline = fnv_u64(timeline, counter_hash);

    uint64_t final_hash = MD_FNV_BASIS;
    final_hash = fnv_u64(final_hash, weight_hash);
    final_hash = fnv_u64(final_hash, state_hash);
    final_hash = fnv_u64(final_hash, timeline);
    final_hash = fnv_u64(final_hash, counter_hash);
    final_hash = fnv_u64(final_hash, st->successful_runs);
    final_hash = fnv_u64(final_hash, st->reset_count);

    out->magic = MD_MAGIC;
    out->version = MD_VERSION;
    out->status = MD_STATUS_OK;
    out->attempted_runs = st->attempted_runs;
    out->successful_runs = st->successful_runs;
    out->reset_count = st->reset_count;
    out->run_tag = run.run_tag;
    out->weight_hash = weight_hash;
    out->state_hash = state_hash;
    out->timeline_hash = timeline;
    out->counter_hash = counter_hash;
    out->final_hash = final_hash;
    out->total_valid_ops = st->total_valid_ops;
    out->total_invalid_ops = st->total_invalid_ops;
    out->total_duplicate_ops = st->total_duplicate_ops;
    out->total_nan_inputs = st->total_nan_inputs;
    out->total_inf_inputs = st->total_inf_inputs;
    out->total_muon_param_updates = st->total_muon_param_updates;
    out->total_adam_param_updates = st->total_adam_param_updates;
    out->last_valid_ops = last_valid;
    out->last_invalid_ops = last_invalid;
    out->last_duplicate_ops = last_dup;
    out->last_nan_inputs = last_nan;
    out->last_inf_inputs = last_inf;
    out->last_muon_param_updates = last_muon;
    out->last_adam_param_updates = last_adam;
    out->reserved = 0;
}

// copy weights -> output region (parallel).
__global__ static void copy_weights_kernel(const uint16_t* weights, uint16_t* out_weights, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_weights[i] = weights[i];
}

// ===========================================================================
// Host-side spec validation + workspace (oracle-identical contract values).
// ===========================================================================
static bool validate_spec_host(const Spec* spec, uint64_t* total_numel, uint64_t* max_param_numel,
                               uint64_t* max_muon_work, uint64_t* max_red, uint64_t* max_MM) {
    if (!spec) return false;
    if (spec->magic != MD_MAGIC || spec->version != MD_VERSION) return false;
    if (spec->num_params == 0 || spec->num_params > MD_MAX_PARAMS) return false;
    if (spec->num_ranks == 0 || spec->num_ranks > MD_MAX_RANKS) return false;
    if (spec->max_ops_per_run > MD_MAX_OPS_PER_RUN) return false;

    uint64_t total = 0, maxn = 1, maxwork = 1, maxr = 1, maxmm = 1;
    for (uint32_t pid = 0; pid < spec->num_params; ++pid) {
        const ParamDesc& p = spec->params[pid];
        if (p.rows == 0 || p.cols == 0) return false;
        if (p.storage_dtype != MD_DTYPE_BF16) return false;
        if ((p.flags & MD_PARAM_FORCE_ADAMW) && (p.flags & MD_PARAM_FORCE_MUON)) return false;
        if ((p.flags & MD_PARAM_FORCE_MUON) && (p.rows < 2 || p.cols < 2)) return false;

        uint64_t n = (uint64_t)p.rows * (uint64_t)p.cols;
        if (n > 0xFFFFFFFFull) return false;
        total += n;
        if (n > maxn) maxn = n;
        if (n > maxr) maxr = n;
        uint32_t maxdim = p.rows > p.cols ? p.rows : p.cols;
        if (maxdim > maxr) maxr = maxdim;

        bool muon = (!(p.flags & MD_PARAM_FORCE_ADAMW) && p.rows >= 2 && p.cols >= 2) ||
                    ((p.flags & MD_PARAM_FORCE_MUON) && p.rows >= 2 && p.cols >= 2);
        if (muon) {
            uint64_t M = p.rows < p.cols ? p.rows : p.cols;
            uint64_t N = p.rows > p.cols ? p.rows : p.cols;
            uint64_t work = 2ull * M * N + 3ull * M * M;
            if (work > maxwork) maxwork = work;
            if (M * M > maxmm) maxmm = M * M;
        }
    }
    *total_numel = total; *max_param_numel = maxn; *max_muon_work = maxwork;
    *max_red = maxr; *max_MM = maxmm;
    return true;
}

static inline uint32_t get_bits_or_default_host(uint32_t bits, uint32_t def_bits) {
    return bits ? bits : def_bits;
}
static size_t align64_host(size_t x) { return (x + 63ull) & ~63ull; }
static uint64_t next_pow2_host(uint64_t x) {
    if (x <= 1ull) return 1ull;
    --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32;
    return x + 1ull;
}

extern "C" size_t solution_workspace_bytes(const Spec* spec) {
    uint64_t total = 0, maxn = 1, maxwork = 1, maxred = 1, maxmm = 1;
    if (!validate_spec_host(spec, &total, &maxn, &maxwork, &maxred, &maxmm)) return 0;
    size_t bytes = 0;
    bytes += align64_host(sizeof(float) * (size_t)total);
    bytes += align64_host(sizeof(float) * (size_t)next_pow2_host(maxred));
    bytes += align64_host(sizeof(uint16_t) * (size_t)(maxwork * 5ull));
    bytes += align64_host(sizeof(uint8_t) * (size_t)spec->max_ops_per_run);
    return align64_host(bytes);
}

// ---------------------------------------------------------------------------
// kernel launch helpers
// ---------------------------------------------------------------------------
static inline uint32_t grid_for(uint64_t n, uint32_t block) {
    return (uint32_t)((n + block - 1) / block);
}

// Parallel Frobenius sum-of-squares of d->X[0..numel) into d->red[0], bit-exact
// to the serial pairwise tree (square + log2(P) adjacent-pair reduce levels).
static void launch_frob(DeviceState* d, uint32_t numel, cudaStream_t stream) {
    const uint32_t BLK = 256;
    uint32_t P = (uint32_t)next_pow2_host(numel);
    frob_square_kernel<<<grid_for(P, BLK), BLK, 0, stream>>>(d, numel, P);
    for (uint32_t p = P; p > 1u; p >>= 1) {
        uint32_t half = p >> 1;
        reduce_level_kernel<<<grid_for(half, BLK), BLK, 0, stream>>>(d, half);
    }
}

// One warp per output element; blockDim = 256 (8 warps/block).
static void launch_matmul(const uint16_t* A, const uint16_t* B, uint16_t* C,
                          uint32_t M, uint32_t N, uint32_t K, int B_transposed,
                          cudaStream_t stream) {
    const uint32_t BLK = 256;
    uint64_t warps = (uint64_t)M * N;
    uint32_t grid = (uint32_t)((warps * 32ull + BLK - 1) / BLK);
    matmul_kernel<<<grid, BLK, 0, stream>>>(A, B, C, M, N, K, B_transposed);
}

extern "C" cudaError_t solution_init(const Spec* spec, void** state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    *state = nullptr;

    uint64_t total = 0, maxn = 1, maxwork = 1, maxred = 1, maxmm = 1;
    if (!validate_spec_host(spec, &total, &maxn, &maxwork, &maxred, &maxmm)) return cudaErrorInvalidValue;

    HostState* hs = new HostState();
    hs->workspace_required = solution_workspace_bytes(spec);
    hs->total_numel = total;

    uint64_t red_cap = next_pow2_host(maxred);

    CUDA_CHECK_RET(cudaMalloc(&hs->d, sizeof(DeviceState)));
    CUDA_CHECK_RET(cudaMalloc(&hs->weights, sizeof(uint16_t) * (size_t)total));
    CUDA_CHECK_RET(cudaMalloc(&hs->muon_m, sizeof(float) * (size_t)total));
    CUDA_CHECK_RET(cudaMalloc(&hs->adam_m, sizeof(float) * (size_t)total));
    CUDA_CHECK_RET(cudaMalloc(&hs->adam_v, sizeof(float) * (size_t)total));
    CUDA_CHECK_RET(cudaMalloc(&hs->grad, sizeof(float) * (size_t)total));
    CUDA_CHECK_RET(cudaMalloc(&hs->X, sizeof(uint16_t) * (size_t)maxn));
    CUDA_CHECK_RET(cudaMalloc(&hs->A, sizeof(uint16_t) * (size_t)maxmm));
    CUDA_CHECK_RET(cudaMalloc(&hs->AA, sizeof(uint16_t) * (size_t)maxmm));
    CUDA_CHECK_RET(cudaMalloc(&hs->B, sizeof(uint16_t) * (size_t)maxmm));
    CUDA_CHECK_RET(cudaMalloc(&hs->BX, sizeof(uint16_t) * (size_t)maxn));
    CUDA_CHECK_RET(cudaMalloc(&hs->oriented, sizeof(uint16_t) * (size_t)maxn));
    CUDA_CHECK_RET(cudaMalloc(&hs->red, sizeof(float) * (size_t)red_cap));
    CUDA_CHECK_RET(cudaMalloc(&hs->snapX, sizeof(uint16_t) * (size_t)maxn * 5ull));

    DeviceState tmp = {};
    tmp.spec = *spec;
    uint64_t offset = 0;
    for (uint32_t pid = 0; pid < spec->num_params; ++pid) {
        const ParamDesc& p = spec->params[pid];
        uint32_t n = p.rows * p.cols;
        tmp.offsets[pid] = (uint32_t)offset;
        tmp.numel[pid] = n;
        tmp.path[pid] = path_for(p);
        offset += n;
    }
    tmp.total_numel = total;
    tmp.max_param_numel = maxn;
    tmp.max_muon_work_elems = maxwork;
    tmp.max_red_elems = maxred;
    tmp.max_MM = maxmm;
    tmp.weights = hs->weights;
    tmp.muon_m = hs->muon_m;
    tmp.adam_m = hs->adam_m;
    tmp.adam_v = hs->adam_v;
    tmp.grad = hs->grad;
    tmp.X = hs->X; tmp.A = hs->A; tmp.AA = hs->AA; tmp.B = hs->B; tmp.BX = hs->BX;
    tmp.oriented = hs->oriented; tmp.red = hs->red; tmp.snapX = hs->snapX;

    CUDA_CHECK_RET(cudaMemcpyAsync(hs->d, &tmp, sizeof(DeviceState), cudaMemcpyHostToDevice, stream));
    init_kernel<<<1, 1, 0, stream>>>(hs->d);
    CUDA_CHECK_RET(cudaGetLastError());

    *state = hs;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state, const RunSpec* run, const void* inputs, void* outputs,
    void* workspace, size_t workspace_bytes, cudaStream_t stream) {

    if (!state || !run || !inputs || !outputs || !workspace) return cudaErrorInvalidValue;
    HostState* hs = (HostState*)state;
    DeviceState* d = hs->d;

    if (run->op_count > MD_MAX_OPS_PER_RUN) return cudaErrorInvalidValue;

    if (workspace_bytes < hs->workspace_required) {
        write_oom_kernel<<<1, 1, 0, stream>>>(d, *run, outputs);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        return cudaErrorMemoryAllocation;
    }

    const uint32_t BLK = 256;

    // Per-param dispatch needs shapes/paths/hyperparams on the host.  Pull a
    // host copy of the persisted Spec once (MD_MAX_PARAMS is small).
    static thread_local Spec hspec;  // reused buffer
    CUDA_CHECK_RET(cudaMemcpyAsync(&hspec, &d->spec, sizeof(Spec), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_RET(cudaStreamSynchronize(stream));

    const uint32_t num_params = hspec.num_params;

    // grad_accum = +0.0
    CUDA_CHECK_RET(cudaMemsetAsync(hs->grad, 0, sizeof(float) * (size_t)hs->total_numel, stream));

    // RUN_BEGIN + op events, then reduce-scatter.
    run_begin_kernel<<<1, 1, 0, stream>>>(d, *run, inputs);
    reduce_scatter_kernel<<<1, 1, 0, stream>>>(d, *run, inputs);

    const float a = as_f32(MD_F32_MUON_A_BITS);
    const float b = as_f32(MD_F32_MUON_B_BITS);
    const float c = as_f32(MD_F32_MUON_C_BITS);
    const float eps = as_f32(MD_F32_NS_EPS_BITS);

    for (uint32_t pid = 0; pid < num_params; ++pid) {
        const ParamDesc& p = hspec.params[pid];
        uint32_t rows = p.rows, cols = p.cols, n = rows * cols;
        uint32_t path = path_for(p);

        // Concurrent per-rank grad-shard hashes, then the serial PARAM_BEGIN fold.
        rank_hash_kernel<<<1, hspec.num_ranks, 0, stream>>>(d, pid);
        param_begin_fold_kernel<<<1, 1, 0, stream>>>(d, pid);

        if (path == MD_PATH_MUON) {
            float mu = as_f32(get_bits_or_default_host(p.muon_momentum_bits, MD_F32_DEFAULT_MOM_BITS));
            float lr = as_f32(get_bits_or_default_host(p.lr_bits, 0x3CA3D70Au));
            float wd = as_f32(get_bits_or_default_host(p.weight_decay_bits, 0x3C23D70Au));
            uint32_t M = rows <= cols ? rows : cols;
            uint32_t N = rows <= cols ? cols : rows;
            uint32_t MN = M * N, MM = M * M;
            uint32_t maxdim = rows > cols ? rows : cols;

            muon_momentum_kernel<<<grid_for(n, BLK), BLK, 0, stream>>>(d, pid, mu);
            muon_build_X_kernel<<<grid_for(MN, BLK), BLK, 0, stream>>>(d, pid, mu, M, N, rows, cols);
            launch_frob(d, MN, stream);
            frob_pre_finish_kernel<<<1, 1, 0, stream>>>(d, eps);
            muon_normalize_kernel<<<grid_for(MN, BLK), BLK, 0, stream>>>(d, MN);
            launch_frob(d, MN, stream);
            frob_normq_finish_kernel<<<1, 1, 0, stream>>>(d);

            for (uint32_t t = 0; t < 5u; ++t) {
                // A = X * X^T  (M x M), K = N
                launch_matmul(hs->X, hs->X, hs->A, M, M, N, 1, stream);
                // AA = A * A   (M x M), K = M
                launch_matmul(hs->A, hs->A, hs->AA, M, M, M, 0, stream);
                muon_B_combine_kernel<<<grid_for(MM, BLK), BLK, 0, stream>>>(d, MM, b, c);
                // BX = B * X   (M x N), K = M
                launch_matmul(hs->B, hs->X, hs->BX, M, N, M, 0, stream);
                muon_Xnew_combine_kernel<<<grid_for(MN, BLK), BLK, 0, stream>>>(d, MN, a);
                launch_frob(d, MN, stream);
                frob_iterq_finish_kernel<<<1, 1, 0, stream>>>(d, t);
                snap_X_kernel<<<grid_for(MN, BLK), BLK, 0, stream>>>(d, t, MN);
            }

            muon_shape_scale_kernel<<<1, 1, 0, stream>>>(d, maxdim);
            muon_apply_kernel<<<grid_for(n, BLK), BLK, 0, stream>>>(d, pid, rows, cols, N, lr, wd);
            muon_step_inc_kernel<<<1, 1, 0, stream>>>(d, pid);
            // Concurrent hashes (mom, 5x X, ortho, weight) then a single serial fold.
            muon_hash_kernel<<<1, 8, 0, stream>>>(d, pid, MN);
            muon_fold_all_kernel<<<1, 1, 0, stream>>>(d, pid);
        } else {
            float lr = as_f32(get_bits_or_default_host(p.adam_lr_bits, 0x3A83126Fu));
            float wd = as_f32(get_bits_or_default_host(p.weight_decay_bits, 0x3C23D70Au));
            float beta1 = as_f32(get_bits_or_default_host(p.adam_beta1_bits, MD_F32_ADAM_B1_BITS));
            float beta2 = as_f32(get_bits_or_default_host(p.adam_beta2_bits, MD_F32_ADAM_B2_BITS));
            float aeps = as_f32(get_bits_or_default_host(p.adam_eps_bits, MD_F32_ADAM_EPS_BITS));

            adam_scalar_kernel<<<1, 1, 0, stream>>>(d, pid, beta1, beta2);
            adam_apply_kernel<<<grid_for(n, BLK), BLK, 0, stream>>>(d, pid, beta1, beta2, aeps, lr, wd);
            adam_hash_kernel<<<1, 3, 0, stream>>>(d, pid);
            adam_fold_kernel<<<1, 1, 0, stream>>>(d, pid);
        }

        param_end_kernel<<<1, 1, 0, stream>>>(d, pid);
    }

    run_hash_kernel<<<1, 2, 0, stream>>>(d);
    run_end_kernel<<<1, 1, 0, stream>>>(d, *run, outputs);

    uint16_t* out_weights = (uint16_t*)((uint8_t*)outputs + sizeof(RunOutputHeader));
    copy_weights_kernel<<<grid_for(hs->total_numel, BLK), BLK, 0, stream>>>(
        hs->weights, out_weights, hs->total_numel);

    return cudaGetLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    HostState* hs = (HostState*)state;
    reset_kernel<<<1, 1, 0, stream>>>(hs->d);
    return cudaGetLastError();
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    HostState* hs = (HostState*)state;
    cudaFree(hs->weights); cudaFree(hs->muon_m); cudaFree(hs->adam_m); cudaFree(hs->adam_v);
    cudaFree(hs->grad); cudaFree(hs->X); cudaFree(hs->A); cudaFree(hs->AA); cudaFree(hs->B);
    cudaFree(hs->BX); cudaFree(hs->oriented); cudaFree(hs->red); cudaFree(hs->snapX); cudaFree(hs->d);
    delete hs;
}
