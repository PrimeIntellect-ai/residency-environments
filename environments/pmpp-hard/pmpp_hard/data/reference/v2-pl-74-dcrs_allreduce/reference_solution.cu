// PMPP_CANARY_74_dacb4e97c8 -- held-out canary; MUST NOT appear in any submission
// file: reference.cu
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <vector>
#include <algorithm>

#define DCR_ABI_VERSION 2u
#define DCR_MAX_OPS_PER_RUN 64u
#define DCR_TILE_ITEMS 256u
#define DCR_WARP_SIZE 32u

#define DCR_MODE_SUM  0u
#define DCR_MODE_SEG  1u
#define DCR_MODE_NORM 2u

#define DCR_DTYPE_F32  0u
#define DCR_DTYPE_BF16 1u
#define DCR_OUT_F32  0u
#define DCR_OUT_BF16 1u

#define DCR_CLASS_FINITE  0u
#define DCR_CLASS_POS_INF 1u
#define DCR_CLASS_NEG_INF 2u
#define DCR_CLASS_NAN     3u

#define DCR_STATUS_OK              0u
#define DCR_STATUS_INVALID_OP      1u
#define DCR_STATUS_MODE_MISMATCH   2u
#define DCR_STATUS_OOB_INPUT       3u
#define DCR_STATUS_OOB_OUTPUT      4u
#define DCR_STATUS_UNSUPPORTED     5u

#define DCR_INVALID_F32_BITS 0x7FC0BAD1u
#define DCR_CANONICAL_NAN_F32_BITS 0x7FC00000u
#define DCR_CANONICAL_BF16_NAN 0x7FC0u
#define DCR_INVALID_BF16 0x7FC1u

static constexpr uint64_t DCR_FNV_BASIS = 1469598103934665603ull;
static constexpr uint64_t DCR_FNV_PRIME = 1099511628211ull;

struct DcrSpec {
    uint32_t abi_version;
    uint32_t max_accumulators;
    uint32_t max_ranks;
    uint32_t max_cells;
    uint32_t max_items_per_rank;
    uint32_t max_ops_per_run;
    uint32_t allow_bf16;
    uint32_t reserved0;
    uint64_t max_input_bytes;
    uint64_t max_output_bytes;
    uint64_t flags;
};

struct DcrOpDesc {
    uint32_t mode;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t acc_id;
    uint32_t logical_ranks;
    uint32_t cells;
    uint32_t items_per_rank;
    uint32_t segment_count;
    uint64_t values_offset;
    uint64_t keys_offset;
    uint64_t tensor_offset;
    uint64_t state_offset;
    uint64_t reserved0;
    uint64_t reserved1;
};

struct DcrRunSpec {
    uint32_t abi_version;
    uint32_t op_count;
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t header_offset;
    uint64_t tensor_region_offset;
    uint64_t tensor_region_bytes;
    uint64_t timeline_region_offset;
    uint64_t timeline_region_bytes;
    uint64_t state_region_offset;
    uint64_t state_region_bytes;
    uint64_t counters_offset;
    uint64_t counters_bytes;
    DcrOpDesc ops[DCR_MAX_OPS_PER_RUN];
};

struct DcrOutputHeader {
    uint64_t magic;
    uint32_t abi_version;
    uint32_t run_status;
    uint64_t run_index_before;
    uint64_t run_index_after;
    uint32_t op_count;
    uint32_t reserved0;
    uint64_t tensor_bytes;
    uint64_t timeline_bytes;
    uint64_t state_bytes;
    uint64_t counters_bytes;
    uint64_t fnv_tensor;
    uint64_t fnv_timeline;
    uint64_t fnv_state;
    uint64_t fnv_counters;
    uint64_t fnv_all;
};

struct DcrTimelineRecord {
    uint64_t run_index;
    uint32_t op_index;
    uint32_t status;
    uint32_t mode;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t acc_id;
    uint32_t logical_ranks;
    uint32_t cells;
    uint32_t items_per_rank;
    uint32_t segment_count;
    uint32_t tile_items;
    uint32_t warp_size;
    uint32_t cross_rank_policy;
    uint32_t shard_policy;
    uint64_t values_offset;
    uint64_t keys_offset;
    uint64_t tensor_offset;
    uint64_t state_offset;
};

struct DcrStateRecord {
    uint32_t acc_id;
    uint32_t cell;
    uint32_t mode;
    uint32_t cls;
    uint32_t sum_bits;
    uint32_t comp_bits;
    uint32_t value_bits;
    uint32_t reserved0;
    uint64_t update_count;
    uint64_t finite_count;
    uint64_t nan_count;
    uint64_t pos_inf_count;
    uint64_t neg_inf_count;
};

struct DcrCountersRecord {
    uint64_t magic;
    uint32_t abi_version;
    uint32_t reserved0;
    uint64_t run_count;
    uint64_t op_count;
    uint64_t valid_op_count;
    uint64_t invalid_op_count;
    uint64_t finite_input_count;
    uint64_t nan_input_count;
    uint64_t pos_inf_input_count;
    uint64_t neg_inf_input_count;
    uint64_t tensor_bytes_emitted;
    uint64_t state_records_emitted;
    uint64_t fnv_basis;
    uint64_t fnv_prime;
};

struct DcrPair {
    float sum;
    float comp;
    uint32_t cls;
    uint32_t pad;
    uint64_t finite_count;
    uint64_t nan_count;
    uint64_t pos_inf_count;
    uint64_t neg_inf_count;
};

struct DcrCellState {
    float sum;
    float comp;
    uint32_t cls;
    uint32_t mode;
    uint64_t update_count;
    uint64_t finite_count;
    uint64_t nan_count;
    uint64_t pos_inf_count;
    uint64_t neg_inf_count;
};

struct DcrDeviceCounters {
    uint64_t op_count;
    uint64_t valid_op_count;
    uint64_t invalid_op_count;
    uint64_t finite_input_count;
    uint64_t nan_input_count;
    uint64_t pos_inf_input_count;
    uint64_t neg_inf_input_count;
    uint64_t tensor_bytes_emitted;
    uint64_t state_records_emitted;
};

struct HostState {
    DcrSpec spec;
    DcrCellState* d_cells = nullptr;
    DcrDeviceCounters* d_counters = nullptr;
    std::vector<uint32_t> acc_mode;
    uint64_t run_count = 0;
    size_t workspace_required = 0;
};

__device__ __forceinline__ uint32_t f32_bits(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    return v.u;
}

__device__ __forceinline__ float bits_f32(uint32_t u) {
    union { float f; uint32_t u; } v;
    v.u = u;
    return v.f;
}

__device__ __forceinline__ float canonical_zero(float x) {
    return (x == 0.0f) ? 0.0f : x;
}

__device__ __forceinline__ DcrPair empty_pair() {
    DcrPair p;
    p.sum = 0.0f;
    p.comp = 0.0f;
    p.cls = DCR_CLASS_FINITE;
    p.pad = 0;
    p.finite_count = 0;
    p.nan_count = 0;
    p.pos_inf_count = 0;
    p.neg_inf_count = 0;
    return p;
}

__device__ __forceinline__ void set_nan_pair(DcrPair& p) {
    p.cls = DCR_CLASS_NAN;
    p.sum = bits_f32(DCR_CANONICAL_NAN_F32_BITS);
    p.comp = 0.0f;
}

__device__ __forceinline__ void pair_add_finite(DcrPair& a, float x) {
    x = canonical_zero(x);
    float t = __fadd_rn(a.sum, x);

    if (isinf(t)) {
        a.cls = signbit(t) ? DCR_CLASS_NEG_INF : DCR_CLASS_POS_INF;
        a.sum = t;
        a.comp = 0.0f;
        return;
    }

    float delta;
    if (fabsf(a.sum) >= fabsf(x)) {
        delta = __fadd_rn(__fsub_rn(a.sum, t), x);
    } else {
        delta = __fadd_rn(__fsub_rn(x, t), a.sum);
    }

    a.comp = canonical_zero(__fadd_rn(a.comp, delta));
    a.sum = canonical_zero(t);
}

__device__ __forceinline__ void pair_resolve_special(DcrPair& a) {
    if (a.nan_count != 0) {
        set_nan_pair(a);
        return;
    }

    if (a.pos_inf_count != 0 && a.neg_inf_count != 0) {
        a.nan_count += 1;
        set_nan_pair(a);
        return;
    }

    if (a.pos_inf_count != 0) {
        a.cls = DCR_CLASS_POS_INF;
        a.sum = CUDART_INF_F;
        a.comp = 0.0f;
        return;
    }

    if (a.neg_inf_count != 0) {
        a.cls = DCR_CLASS_NEG_INF;
        a.sum = -CUDART_INF_F;
        a.comp = 0.0f;
        return;
    }

    a.cls = DCR_CLASS_FINITE;
}

__device__ __forceinline__ void merge_pair(DcrPair& a, const DcrPair& b) {
    a.finite_count  += b.finite_count;
    a.nan_count     += b.nan_count;
    a.pos_inf_count += b.pos_inf_count;
    a.neg_inf_count += b.neg_inf_count;

    if (a.cls == DCR_CLASS_NAN || b.cls == DCR_CLASS_NAN) {
        a.nan_count += (a.nan_count == 0);
        set_nan_pair(a);
        return;
    }

    pair_resolve_special(a);
    if (a.cls != DCR_CLASS_FINITE) return;

    if (b.cls == DCR_CLASS_POS_INF) {
        a.pos_inf_count += 1;
        pair_resolve_special(a);
        return;
    }
    if (b.cls == DCR_CLASS_NEG_INF) {
        a.neg_inf_count += 1;
        pair_resolve_special(a);
        return;
    }
    if (b.cls == DCR_CLASS_NAN) {
        a.nan_count += 1;
        set_nan_pair(a);
        return;
    }

    pair_add_finite(a, b.sum);
    pair_add_finite(a, b.comp);
}

__device__ __forceinline__ DcrPair leaf_sum(float x) {
    DcrPair p = empty_pair();
    if (isnan(x)) {
        p.nan_count = 1;
        set_nan_pair(p);
    } else if (isinf(x)) {
        if (signbit(x)) {
            p.cls = DCR_CLASS_NEG_INF;
            p.sum = -CUDART_INF_F;
            p.neg_inf_count = 1;
        } else {
            p.cls = DCR_CLASS_POS_INF;
            p.sum = CUDART_INF_F;
            p.pos_inf_count = 1;
        }
    } else {
        p.sum = canonical_zero(x);
        p.finite_count = 1;
    }
    return p;
}

__device__ __forceinline__ DcrPair leaf_norm(float x) {
    DcrPair p = empty_pair();
    if (isnan(x)) {
        p.nan_count = 1;
        set_nan_pair(p);
    } else if (isinf(x)) {
        p.cls = DCR_CLASS_POS_INF;
        p.sum = CUDART_INF_F;
        p.pos_inf_count = 1;
    } else {
        float y = __fmul_rn(x, x);
        if (isinf(y)) {
            p.cls = DCR_CLASS_POS_INF;
            p.sum = CUDART_INF_F;
            p.pos_inf_count = 1;
        } else {
            p.sum = canonical_zero(y);
            p.finite_count = 1;
        }
    }
    return p;
}

__device__ __forceinline__ float load_value(
    const uint8_t* inputs,
    uint64_t values_offset,
    uint64_t linear_idx,
    uint32_t dtype) {
    if (dtype == DCR_DTYPE_F32) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(inputs + values_offset);
        return bits_f32(p[linear_idx]);
    } else {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(inputs + values_offset);
        uint32_t bits = uint32_t(p[linear_idx]) << 16;
        return bits_f32(bits);
    }
}

__device__ __forceinline__ uint32_t finalize_sum_bits(const DcrPair& p) {
    if (p.cls == DCR_CLASS_NAN) return DCR_CANONICAL_NAN_F32_BITS;
    if (p.cls == DCR_CLASS_POS_INF) return 0x7F800000u;
    if (p.cls == DCR_CLASS_NEG_INF) return 0xFF800000u;
    float v = canonical_zero(__fadd_rn(p.sum, p.comp));
    return f32_bits(v);
}

__device__ __forceinline__ uint32_t finalize_norm_bits(const DcrPair& p) {
    if (p.cls == DCR_CLASS_NAN) return DCR_CANONICAL_NAN_F32_BITS;
    if (p.cls == DCR_CLASS_POS_INF) return 0x7F800000u;
    if (p.cls == DCR_CLASS_NEG_INF) return DCR_CANONICAL_NAN_F32_BITS;

    float total = canonical_zero(__fadd_rn(p.sum, p.comp));
    if (total < 0.0f) total = 0.0f;
    float v = canonical_zero(sqrtf(total));
    return f32_bits(v);
}

__device__ __forceinline__ uint16_t f32_to_bf16_rne(uint32_t bits) {
    uint32_t exp = bits & 0x7F800000u;
    uint32_t mant = bits & 0x007FFFFFu;
    if (exp == 0x7F800000u && mant != 0) {
        return DCR_CANONICAL_BF16_NAN;
    }
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounded = bits + 0x7FFFu + lsb;
    return uint16_t(rounded >> 16);
}

__device__ __forceinline__ void store_output_value(
    uint8_t* outputs,
    uint64_t byte_offset,
    uint32_t output_dtype,
    uint32_t value_bits) {
    if (output_dtype == DCR_OUT_F32) {
        uint32_t* p = reinterpret_cast<uint32_t*>(outputs + byte_offset);
        *p = value_bits;
    } else {
        uint16_t* p = reinterpret_cast<uint16_t*>(outputs + byte_offset);
        *p = f32_to_bf16_rne(value_bits);
    }
}

__device__ __forceinline__ uint32_t owner_of_cell(uint32_t c, uint32_t C, uint32_t R) {
    if (R == 0) return 0;
    uint32_t base = C / R;
    uint32_t rem = C % R;

    for (uint32_t r = 0; r < R; ++r) {
        uint32_t begin = r * base + min(r, rem);
        uint32_t end = begin + base + ((r < rem) ? 1u : 0u);
        if (c >= begin && c < end) return r;
    }
    return R - 1;
}

__device__ DcrPair block_reduce_pair(DcrPair v) {
    __shared__ DcrPair warp_partials[8];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    for (int delta = 16; delta >= 1; delta >>= 1) {
        DcrPair other;
        other.sum = __shfl_down_sync(0xffffffffu, v.sum, delta);
        other.comp = __shfl_down_sync(0xffffffffu, v.comp, delta);
        other.cls = __shfl_down_sync(0xffffffffu, v.cls, delta);
        other.pad = 0;
        other.finite_count = __shfl_down_sync(0xffffffffu, v.finite_count, delta);
        other.nan_count = __shfl_down_sync(0xffffffffu, v.nan_count, delta);
        other.pos_inf_count = __shfl_down_sync(0xffffffffu, v.pos_inf_count, delta);
        other.neg_inf_count = __shfl_down_sync(0xffffffffu, v.neg_inf_count, delta);

        if (lane < delta) {
            merge_pair(v, other);
        }
    }

    if (lane == 0) warp_partials[warp] = v;
    __syncthreads();

    DcrPair w = empty_pair();
    if (warp == 0) {
        if (lane < 8) w = warp_partials[lane];

        for (int delta = 4; delta >= 1; delta >>= 1) {
            DcrPair other;
            other.sum = __shfl_down_sync(0xffffffffu, w.sum, delta);
            other.comp = __shfl_down_sync(0xffffffffu, w.comp, delta);
            other.cls = __shfl_down_sync(0xffffffffu, w.cls, delta);
            other.pad = 0;
            other.finite_count = __shfl_down_sync(0xffffffffu, w.finite_count, delta);
            other.nan_count = __shfl_down_sync(0xffffffffu, w.nan_count, delta);
            other.pos_inf_count = __shfl_down_sync(0xffffffffu, w.pos_inf_count, delta);
            other.neg_inf_count = __shfl_down_sync(0xffffffffu, w.neg_inf_count, delta);

            if (lane < delta) {
                merge_pair(w, other);
            }
        }
    }

    return w;
}

__global__ void tile_partial_kernel(
    const uint8_t* inputs,
    DcrPair* tile_partials,
    DcrOpDesc op,
    uint32_t tile_count) {
    uint32_t tile = blockIdx.x;
    uint32_t cell = blockIdx.y;
    uint32_t rank = blockIdx.z;
    uint32_t tid = threadIdx.x;

    uint32_t C = op.cells;
    uint32_t M = op.items_per_rank;
    uint32_t m = tile * DCR_TILE_ITEMS + tid;

    DcrPair leaf = empty_pair();

    if (m < M) {
        if (op.mode == DCR_MODE_SEG) {
            const uint32_t* keys =
                reinterpret_cast<const uint32_t*>(inputs + op.keys_offset);
            uint64_t km_idx = uint64_t(rank) * M + m;
            uint32_t key = keys[km_idx];

            if (key == cell) {
                float x = load_value(inputs, op.values_offset, km_idx, op.input_dtype);
                leaf = leaf_sum(x);
            }
        } else {
            uint64_t idx = (uint64_t(rank) * M + m) * C + cell;
            float x = load_value(inputs, op.values_offset, idx, op.input_dtype);
            leaf = (op.mode == DCR_MODE_NORM) ? leaf_norm(x) : leaf_sum(x);
        }
    }

    DcrPair reduced = block_reduce_pair(leaf);

    if (threadIdx.x == 0) {
        uint64_t out_idx = (uint64_t(rank) * C + cell) * tile_count + tile;
        tile_partials[out_idx] = reduced;
    }
}

__global__ void tile_second_pass_kernel(
    DcrPair* tile_partials,
    DcrPair* rank_partials,
    DcrOpDesc op,
    uint32_t tile_count) {
    uint32_t cell = blockIdx.x;
    uint32_t rank = blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t C = op.cells;

    DcrPair p = empty_pair();
    if (tid < tile_count) {
        uint64_t idx = (uint64_t(rank) * C + cell) * tile_count + tid;
        p = tile_partials[idx];
    }

    DcrPair reduced = block_reduce_pair(p);

    if (threadIdx.x == 0) {
        rank_partials[uint64_t(rank) * C + cell] = reduced;
    }
}

__device__ DcrPair cross_rank_reduce(
    const DcrPair* rank_partials,
    uint32_t cell,
    uint32_t C,
    uint32_t R) {
    uint32_t owner = owner_of_cell(cell, C, R);
    DcrPair lanes[16];

    for (uint32_t j = 0; j < 16; ++j) {
        if (j < R) {
            uint32_t rank = (owner + j) % R;
            lanes[j] = rank_partials[uint64_t(rank) * C + cell];
        } else {
            lanes[j] = empty_pair();
        }
    }

    uint32_t pow2 = 1;
    while ((pow2 << 1) <= R) pow2 <<= 1;

    for (uint32_t delta = pow2 >> 1; delta >= 1; delta >>= 1) {
        for (uint32_t lane = 0; lane < delta; ++lane) {
            if (lane + delta < R) {
                merge_pair(lanes[lane], lanes[lane + delta]);
            }
        }
        if (delta == 1) break;
    }

    return lanes[0];
}

__global__ void cross_state_output_kernel(
    DcrPair* rank_partials,
    DcrCellState* cells,
    DcrDeviceCounters* counters,
    uint8_t* outputs,
    DcrOpDesc op,
    uint32_t mode,
    uint32_t run_mode_status) {
    uint32_t cell = blockIdx.x;
    if (threadIdx.x != 0) return;

    uint32_t C = op.cells;
    uint32_t R = op.logical_ranks;
    uint32_t out_bytes = (op.output_dtype == DCR_OUT_F32) ? 4u : 2u;

    DcrPair step = cross_rank_reduce(rank_partials, cell, C, R);

    uint64_t state_idx = uint64_t(op.acc_id) * gridDim.x + cell;
    // The reference launches with gridDim.x == C, but state is actually max_cells-strided.
    // The host passes state_base already offset by acc_id * max_cells in production.
    DcrCellState* st = cells + state_idx;

    DcrPair running = empty_pair();
    running.sum = st->sum;
    running.comp = st->comp;
    running.cls = st->cls;
    running.finite_count = st->finite_count;
    running.nan_count = st->nan_count;
    running.pos_inf_count = st->pos_inf_count;
    running.neg_inf_count = st->neg_inf_count;

    merge_pair(running, step);

    st->sum = running.sum;
    st->comp = running.comp;
    st->cls = running.cls;
    st->mode = mode;
    st->update_count += 1;
    st->finite_count = running.finite_count;
    st->nan_count = running.nan_count;
    st->pos_inf_count = running.pos_inf_count;
    st->neg_inf_count = running.neg_inf_count;

    uint32_t step_bits =
        (mode == DCR_MODE_NORM) ? finalize_norm_bits(step) : finalize_sum_bits(step);

    for (uint32_t r = 0; r < R; ++r) {
        uint64_t byte_off = op.tensor_offset + (uint64_t(r) * C + cell) * out_bytes;
        store_output_value(outputs, byte_off, op.output_dtype, step_bits);
    }

    uint32_t running_bits =
        (mode == DCR_MODE_NORM) ? finalize_norm_bits(running) : finalize_sum_bits(running);

    DcrStateRecord rec;
    rec.acc_id = op.acc_id;
    rec.cell = cell;
    rec.mode = mode;
    rec.cls = running.cls;
    rec.sum_bits = f32_bits(running.sum);
    rec.comp_bits = f32_bits(running.comp);
    rec.value_bits = running_bits;
    rec.reserved0 = 0;
    rec.update_count = st->update_count;
    rec.finite_count = running.finite_count;
    rec.nan_count = running.nan_count;
    rec.pos_inf_count = running.pos_inf_count;
    rec.neg_inf_count = running.neg_inf_count;

    reinterpret_cast<DcrStateRecord*>(outputs + op.state_offset)[cell] = rec;

    atomicAdd(reinterpret_cast<unsigned long long*>(&counters->finite_input_count),
              static_cast<unsigned long long>(step.finite_count));
    atomicAdd(reinterpret_cast<unsigned long long*>(&counters->nan_input_count),
              static_cast<unsigned long long>(step.nan_count));
    atomicAdd(reinterpret_cast<unsigned long long*>(&counters->pos_inf_input_count),
              static_cast<unsigned long long>(step.pos_inf_count));
    atomicAdd(reinterpret_cast<unsigned long long*>(&counters->neg_inf_input_count),
              static_cast<unsigned long long>(step.neg_inf_count));
}

__global__ void write_timeline_kernel(
    uint8_t* outputs,
    DcrOpDesc op,
    uint64_t timeline_offset,
    uint64_t run_index,
    uint32_t op_index,
    uint32_t status) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    DcrTimelineRecord rec;
    rec.run_index = run_index;
    rec.op_index = op_index;
    rec.status = status;
    rec.mode = op.mode;
    rec.input_dtype = op.input_dtype;
    rec.output_dtype = op.output_dtype;
    rec.acc_id = op.acc_id;
    rec.logical_ranks = op.logical_ranks;
    rec.cells = op.cells;
    rec.items_per_rank = op.items_per_rank;
    rec.segment_count = op.segment_count;
    rec.tile_items = DCR_TILE_ITEMS;
    rec.warp_size = DCR_WARP_SIZE;
    rec.cross_rank_policy = 1;
    rec.shard_policy = 1;
    rec.values_offset = op.values_offset;
    rec.keys_offset = op.keys_offset;
    rec.tensor_offset = op.tensor_offset;
    rec.state_offset = op.state_offset;

    reinterpret_cast<DcrTimelineRecord*>(outputs + timeline_offset)[op_index] = rec;
}

__global__ void invalid_op_kernel(
    uint8_t* outputs,
    DcrDeviceCounters* counters,
    DcrOpDesc op,
    uint32_t effective_R,
    uint32_t effective_C) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t out_bytes = (op.output_dtype == DCR_OUT_BF16) ? 2u : 4u;
    uint64_t n = uint64_t(effective_R) * effective_C;

    for (uint64_t i = 0; i < n; ++i) {
        uint64_t byte_off = op.tensor_offset + i * out_bytes;
        if (op.output_dtype == DCR_OUT_BF16) {
            reinterpret_cast<uint16_t*>(outputs + byte_off)[0] = DCR_INVALID_BF16;
        } else {
            reinterpret_cast<uint32_t*>(outputs + byte_off)[0] = DCR_INVALID_F32_BITS;
        }
    }

    counters->invalid_op_count += 1;
}

__global__ void bump_valid_op_kernel(DcrDeviceCounters* counters, uint64_t tensor_bytes, uint64_t state_records) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    counters->op_count += 1;
    counters->valid_op_count += 1;
    counters->tensor_bytes_emitted += tensor_bytes;
    counters->state_records_emitted += state_records;
}

__global__ void bump_invalid_op_kernel(DcrDeviceCounters* counters, uint64_t tensor_bytes) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    counters->op_count += 1;
    counters->tensor_bytes_emitted += tensor_bytes;
}

__device__ __forceinline__ uint64_t fnv_byte(uint64_t h, uint8_t b) {
    h ^= uint64_t(b);
    h *= DCR_FNV_PRIME;
    return h;
}

__device__ uint64_t fnv_region(const uint8_t* p, uint64_t n) {
    uint64_t h = DCR_FNV_BASIS;
    for (uint64_t i = 0; i < n; ++i) h = fnv_byte(h, p[i]);
    return h;
}

__device__ uint64_t fnv_u32(uint64_t h, uint32_t x) {
    for (int i = 0; i < 4; ++i) h = fnv_byte(h, uint8_t((x >> (8 * i)) & 0xffu));
    return h;
}

__device__ uint64_t fnv_u64(uint64_t h, uint64_t x) {
    for (int i = 0; i < 8; ++i) h = fnv_byte(h, uint8_t((x >> (8 * i)) & 0xffu));
    return h;
}

__device__ uint64_t fnv_region_continue(uint64_t h, const uint8_t* p, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) h = fnv_byte(h, p[i]);
    return h;
}

__global__ void finalize_header_kernel(
    uint8_t* outputs,
    DcrRunSpec run,
    DcrDeviceCounters* counters,
    uint64_t run_before,
    uint64_t run_after) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    DcrCountersRecord cr;
    cr.magic = 0x44535253434E5452ull;
    cr.abi_version = DCR_ABI_VERSION;
    cr.reserved0 = 0;
    cr.run_count = run_after;
    cr.op_count = counters->op_count;
    cr.valid_op_count = counters->valid_op_count;
    cr.invalid_op_count = counters->invalid_op_count;
    cr.finite_input_count = counters->finite_input_count;
    cr.nan_input_count = counters->nan_input_count;
    cr.pos_inf_input_count = counters->pos_inf_input_count;
    cr.neg_inf_input_count = counters->neg_inf_input_count;
    cr.tensor_bytes_emitted = counters->tensor_bytes_emitted;
    cr.state_records_emitted = counters->state_records_emitted;
    cr.fnv_basis = DCR_FNV_BASIS;
    cr.fnv_prime = DCR_FNV_PRIME;

    *reinterpret_cast<DcrCountersRecord*>(outputs + run.counters_offset) = cr;

    const uint8_t* tensor = outputs + run.tensor_region_offset;
    const uint8_t* time = outputs + run.timeline_region_offset;
    const uint8_t* state = outputs + run.state_region_offset;
    const uint8_t* cntr = outputs + run.counters_offset;

    uint64_t ht = fnv_region(tensor, run.tensor_region_bytes);
    uint64_t htime = fnv_region(time, run.timeline_region_bytes);
    uint64_t hs = fnv_region(state, run.state_region_bytes);
    uint64_t hc = fnv_region(cntr, run.counters_bytes);

    uint64_t hall = DCR_FNV_BASIS;
    hall = fnv_u32(hall, 0x54454E53u);
    hall = fnv_u64(hall, run.tensor_region_bytes);
    hall = fnv_region_continue(hall, tensor, run.tensor_region_bytes);

    hall = fnv_u32(hall, 0x54494D45u);
    hall = fnv_u64(hall, run.timeline_region_bytes);
    hall = fnv_region_continue(hall, time, run.timeline_region_bytes);

    hall = fnv_u32(hall, 0x53544154u);
    hall = fnv_u64(hall, run.state_region_bytes);
    hall = fnv_region_continue(hall, state, run.state_region_bytes);

    hall = fnv_u32(hall, 0x434E5452u);
    hall = fnv_u64(hall, run.counters_bytes);
    hall = fnv_region_continue(hall, cntr, run.counters_bytes);

    DcrOutputHeader hdr;
    hdr.magic = 0x44535253414C4C32ull;
    hdr.abi_version = DCR_ABI_VERSION;
    hdr.run_status = DCR_STATUS_OK;
    hdr.run_index_before = run_before;
    hdr.run_index_after = run_after;
    hdr.op_count = run.op_count;
    hdr.reserved0 = 0;
    hdr.tensor_bytes = run.tensor_region_bytes;
    hdr.timeline_bytes = run.timeline_region_bytes;
    hdr.state_bytes = run.state_region_bytes;
    hdr.counters_bytes = run.counters_bytes;
    hdr.fnv_tensor = ht;
    hdr.fnv_timeline = htime;
    hdr.fnv_state = hs;
    hdr.fnv_counters = hc;
    hdr.fnv_all = hall;

    *reinterpret_cast<DcrOutputHeader*>(outputs + run.header_offset) = hdr;
}

static inline uint64_t value_bytes(const DcrOpDesc& op) {
    uint64_t elem = (op.input_dtype == DCR_DTYPE_F32) ? 4ull : 2ull;
    if (op.mode == DCR_MODE_SEG) {
        return uint64_t(op.logical_ranks) * op.items_per_rank * elem;
    }
    return uint64_t(op.logical_ranks) * op.items_per_rank * op.cells * elem;
}

static inline bool range_ok(uint64_t off, uint64_t len, uint64_t total) {
    return off <= total && len <= total - off;
}

extern "C" size_t solution_workspace_bytes(const DcrSpec* spec) {
    if (!spec) return 0;
    uint64_t tiles = (uint64_t(spec->max_items_per_rank) + DCR_TILE_ITEMS - 1) / DCR_TILE_ITEMS;
    if (tiles == 0) tiles = 1;
    uint64_t tile_pairs = uint64_t(spec->max_ranks) * spec->max_cells * tiles;
    uint64_t rank_pairs = uint64_t(spec->max_ranks) * spec->max_cells;
    return size_t((tile_pairs + rank_pairs) * sizeof(DcrPair));
}

extern "C" cudaError_t solution_init(const DcrSpec* spec, void** state, cudaStream_t stream) {
    if (!spec || !state) return cudaErrorInvalidValue;
    if (spec->abi_version != DCR_ABI_VERSION) return cudaErrorInvalidValue;
    if (spec->max_ops_per_run > DCR_MAX_OPS_PER_RUN) return cudaErrorInvalidValue;

    HostState* hs = new HostState();
    hs->spec = *spec;
    hs->workspace_required = solution_workspace_bytes(spec);
    hs->acc_mode.assign(spec->max_accumulators, 0xFFFFFFFFu);

    uint64_t cell_count = uint64_t(spec->max_accumulators) * spec->max_cells;
    cudaError_t err = cudaMalloc(&hs->d_cells, cell_count * sizeof(DcrCellState));
    if (err != cudaSuccess) { delete hs; return err; }

    err = cudaMalloc(&hs->d_counters, sizeof(DcrDeviceCounters));
    if (err != cudaSuccess) {
        cudaFree(hs->d_cells);
        delete hs;
        return err;
    }

    cudaMemsetAsync(hs->d_cells, 0, cell_count * sizeof(DcrCellState), stream);
    cudaMemsetAsync(hs->d_counters, 0, sizeof(DcrDeviceCounters), stream);

    *state = hs;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    HostState* hs = reinterpret_cast<HostState*>(state);
    hs->run_count = 0;
    std::fill(hs->acc_mode.begin(), hs->acc_mode.end(), 0xFFFFFFFFu);

    uint64_t cell_count = uint64_t(hs->spec.max_accumulators) * hs->spec.max_cells;
    cudaMemsetAsync(hs->d_cells, 0, cell_count * sizeof(DcrCellState), stream);
    cudaMemsetAsync(hs->d_counters, 0, sizeof(DcrDeviceCounters), stream);
    return cudaSuccess;
}

static uint32_t validate_op_host(
    const HostState* hs,
    const DcrRunSpec* run,
    const DcrOpDesc& op,
    uint64_t* tensor_bytes_out,
    uint64_t* state_bytes_out) {
    const DcrSpec& sp = hs->spec;

    uint32_t out_b = (op.output_dtype == DCR_OUT_BF16) ? 2u : 4u;
    uint32_t effective_R = std::max(1u, std::min(op.logical_ranks, sp.max_ranks));
    uint32_t effective_C = std::min(op.cells, sp.max_cells);
    *tensor_bytes_out = uint64_t(effective_R) * effective_C * out_b;
    *state_bytes_out = uint64_t(effective_C) * sizeof(DcrStateRecord);

    bool mode_ok = (op.mode == DCR_MODE_SUM || op.mode == DCR_MODE_SEG || op.mode == DCR_MODE_NORM);
    if (!mode_ok) return DCR_STATUS_INVALID_OP;
    if (!(op.input_dtype == DCR_DTYPE_F32 || op.input_dtype == DCR_DTYPE_BF16)) return DCR_STATUS_INVALID_OP;
    if (!(op.output_dtype == DCR_OUT_F32 || op.output_dtype == DCR_OUT_BF16)) return DCR_STATUS_INVALID_OP;
    if (op.input_dtype == DCR_DTYPE_BF16 && sp.allow_bf16 == 0) return DCR_STATUS_INVALID_OP;
    if (op.acc_id >= sp.max_accumulators) return DCR_STATUS_INVALID_OP;
    if (op.logical_ranks == 0 || op.logical_ranks > sp.max_ranks) return DCR_STATUS_INVALID_OP;
    if (op.cells > sp.max_cells) return DCR_STATUS_INVALID_OP;
    if (op.items_per_rank > sp.max_items_per_rank) return DCR_STATUS_INVALID_OP;
    if (op.mode == DCR_MODE_SEG && op.segment_count != op.cells) return DCR_STATUS_INVALID_OP;
    if (op.mode != DCR_MODE_SEG && op.segment_count != 0) return DCR_STATUS_INVALID_OP;

    uint64_t vb = value_bytes(op);
    if (!range_ok(op.values_offset, vb, run->input_bytes)) return DCR_STATUS_OOB_INPUT;

    if (op.mode == DCR_MODE_SEG) {
        uint64_t kb = uint64_t(op.logical_ranks) * op.items_per_rank * sizeof(uint32_t);
        if (!range_ok(op.keys_offset, kb, run->input_bytes)) return DCR_STATUS_OOB_INPUT;
    }

    if (!range_ok(op.tensor_offset, *tensor_bytes_out, run->output_bytes)) return DCR_STATUS_OOB_OUTPUT;
    if (!range_ok(op.state_offset, *state_bytes_out, run->output_bytes)) return DCR_STATUS_OOB_OUTPUT;

    uint32_t locked = hs->acc_mode[op.acc_id];
    if (locked != 0xFFFFFFFFu && locked != op.mode) return DCR_STATUS_MODE_MISMATCH;

    return DCR_STATUS_OK;
}

extern "C" cudaError_t solution_run(
    void* state,
    const DcrRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !run || !inputs || !outputs || !workspace) return cudaErrorInvalidValue;
    if (run->abi_version != DCR_ABI_VERSION) return cudaErrorInvalidValue;

    HostState* hs = reinterpret_cast<HostState*>(state);
    if (run->op_count > hs->spec.max_ops_per_run) return cudaErrorInvalidValue;
    if (run->op_count > DCR_MAX_OPS_PER_RUN) return cudaErrorInvalidValue;
    if (workspace_bytes < hs->workspace_required) return cudaErrorMemoryAllocation;

    uint8_t* out = reinterpret_cast<uint8_t*>(outputs);
    const uint8_t* in = reinterpret_cast<const uint8_t*>(inputs);

    uint64_t run_before = hs->run_count;
    uint64_t run_after = hs->run_count + 1;

    DcrPair* tile_partials = reinterpret_cast<DcrPair*>(workspace);
    uint64_t max_tiles = (uint64_t(hs->spec.max_items_per_rank) + DCR_TILE_ITEMS - 1) / DCR_TILE_ITEMS;
    if (max_tiles == 0) max_tiles = 1;
    DcrPair* rank_partials =
        tile_partials + uint64_t(hs->spec.max_ranks) * hs->spec.max_cells * max_tiles;

    for (uint32_t i = 0; i < run->op_count; ++i) {
        DcrOpDesc op = run->ops[i];

        uint64_t tensor_bytes = 0;
        uint64_t state_bytes = 0;
        uint32_t status = validate_op_host(hs, run, op, &tensor_bytes, &state_bytes);

        uint64_t timeline_off = run->timeline_region_offset + uint64_t(i) * sizeof(DcrTimelineRecord);
        write_timeline_kernel<<<1, 1, 0, stream>>>(out, op, timeline_off, run_before, i, status);

        if (status != DCR_STATUS_OK) {
            uint32_t eff_R = std::max(1u, std::min(op.logical_ranks, hs->spec.max_ranks));
            uint32_t eff_C = std::min(op.cells, hs->spec.max_cells);
            invalid_op_kernel<<<1, 1, 0, stream>>>(out, hs->d_counters, op, eff_R, eff_C);
            bump_invalid_op_kernel<<<1, 1, 0, stream>>>(hs->d_counters, tensor_bytes);
            continue;
        }

        if (hs->acc_mode[op.acc_id] == 0xFFFFFFFFu) {
            hs->acc_mode[op.acc_id] = op.mode;
        }

        uint32_t tile_count = (op.items_per_rank + DCR_TILE_ITEMS - 1) / DCR_TILE_ITEMS;
        if (tile_count == 0) tile_count = 1;

        dim3 tile_grid(tile_count, op.cells, op.logical_ranks);
        tile_partial_kernel<<<tile_grid, DCR_TILE_ITEMS, 0, stream>>>(
            in, tile_partials, op, tile_count);

        dim3 second_grid(op.cells, op.logical_ranks);
        tile_second_pass_kernel<<<second_grid, DCR_TILE_ITEMS, 0, stream>>>(
            tile_partials, rank_partials, op, tile_count);

        // Reference state indexing note:
        // production state is max_cells-strided. To keep the kernel simple, pass a cell pointer
        // already offset to acc_id * max_cells by pointer arithmetic here.
        DcrCellState* acc_base = hs->d_cells + uint64_t(op.acc_id) * hs->spec.max_cells;

        DcrOpDesc op_for_kernel = op;
        op_for_kernel.acc_id = op.acc_id;

        cross_state_output_kernel<<<op.cells, 1, 0, stream>>>(
            rank_partials,
            acc_base - uint64_t(op.acc_id) * op.cells,
            hs->d_counters,
            out,
            op_for_kernel,
            op.mode,
            status);

        bump_valid_op_kernel<<<1, 1, 0, stream>>>(
            hs->d_counters,
            tensor_bytes,
            uint64_t(op.cells));
    }

    finalize_header_kernel<<<1, 1, 0, stream>>>(
        out, *run, hs->d_counters, run_before, run_after);

    hs->run_count = run_after;
    return cudaGetLastError();
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    HostState* hs = reinterpret_cast<HostState*>(state);
    cudaFree(hs->d_cells);
    cudaFree(hs->d_counters);
    delete hs;
}
