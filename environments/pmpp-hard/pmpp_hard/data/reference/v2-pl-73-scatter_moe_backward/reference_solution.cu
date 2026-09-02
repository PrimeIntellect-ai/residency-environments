// PMPP_CANARY_73_dcff03cf5c -- held-out canary; MUST NOT appear in any submission
// file: reference.cu
#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

static constexpr uint32_t DSG_SPEC_MAGIC = 0x44534753u;
static constexpr uint32_t DSG_RUN_MAGIC  = 0x44534752u;
static constexpr uint32_t DSG_OUT_MAGIC  = 0x4453474Fu;
static constexpr uint32_t DSG_VERSION    = 1;

static constexpr uint32_t DSG_DTYPE_BF16 = 0;
static constexpr uint32_t DSG_DTYPE_F32  = 1;

static constexpr uint32_t DSG_FLAG_EMIT_TABLE        = 1u << 0;
static constexpr uint32_t DSG_FLAG_EMIT_EXPERT       = 1u << 1;
static constexpr uint32_t DSG_FLAG_CLEAR_AFTER_FLUSH = 1u << 2;

static constexpr uint32_t DSG_VALID_RUN_FLAGS =
    DSG_FLAG_EMIT_TABLE |
    DSG_FLAG_EMIT_EXPERT |
    DSG_FLAG_CLEAR_AFTER_FLUSH;

static constexpr uint32_t DSG_CAPACITY_USE_SPEC = 0xFFFFFFFFu;

static constexpr uint64_t DSG_FNV_BASIS = 1469598103934665603ull;
static constexpr uint64_t DSG_FNV_PRIME = 1099511628211ull;

struct Spec {
    uint32_t magic;
    uint32_t version;

    uint32_t table_rows;
    uint32_t dim;
    uint32_t out_dim;
    uint32_t experts;
    uint32_t top_k;
    uint32_t max_tokens;

    uint32_t capacity;
    uint32_t input_dtype;

    int32_t  padding_idx;
    uint32_t flags;

    uint64_t weight_seed;

    uint64_t reserved0;
    uint64_t reserved1;
};

struct RunSpec {
    uint32_t magic;
    uint32_t version;
    uint32_t tokens;
    uint32_t flags;
    uint32_t capacity_override;
    uint32_t reserved0;
    uint64_t user_tag;
};

struct OutHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;

    uint64_t run_id;
    uint64_t completed_runs;

    uint32_t tokens;
    uint32_t flags;
    uint32_t effective_capacity;
    uint32_t top_k;

    uint32_t accepted;
    uint32_t dropped_capacity;
    uint32_t dropped_invalid_expert;
    uint32_t skipped_padding;
    uint32_t skipped_invalid_index;
    uint32_t table_segments;
    uint32_t expert_segments;
    uint32_t flush_happened;

    uint64_t total_tokens;
    uint64_t total_accepted;
    uint64_t total_dropped_capacity;
    uint64_t total_dropped_invalid_expert;
    uint64_t total_skipped_padding;
    uint64_t total_skipped_invalid_index;
    uint64_t total_table_segments;
    uint64_t total_expert_segments;
    uint64_t total_flushes;

    uint64_t event_hash;
    uint64_t tensor_hash;
    uint64_t state_hash;
    uint64_t step_hash;

    uint64_t reserved0;
    uint64_t reserved1;
};

struct Persist {
    uint64_t completed_runs;

    uint64_t total_tokens;
    uint64_t total_accepted;
    uint64_t total_dropped_capacity;
    uint64_t total_dropped_invalid_expert;
    uint64_t total_skipped_padding;
    uint64_t total_skipped_invalid_index;
    uint64_t total_table_segments;
    uint64_t total_expert_segments;
    uint64_t total_flushes;

    uint64_t cumulative_event_hash;
};

struct DeviceStats {
    uint64_t event_hash;

    uint32_t accepted;
    uint32_t dropped_capacity;
    uint32_t dropped_invalid_expert;
    uint32_t skipped_padding;
    uint32_t skipped_invalid_index;

    uint32_t table_count;
    uint32_t table_segments;

    uint32_t expert_count;
    uint32_t expert_segments;

    uint32_t effective_capacity;
    uint32_t flush_happened;
};

struct RefState {
    Spec spec;

    Persist* d_persist;
    DeviceStats* d_stats;

    float* d_table;
    float* d_expert;
    float* d_weight;

    float* d_dx;

    uint32_t* d_capacity_used;
    uint8_t*  d_accepted;

    uint64_t* d_table_keys;
    uint32_t* d_table_tokens;

    uint64_t* d_expert_keys;
    uint32_t* d_expert_assigns;

    size_t table_cells;
    size_t expert_cells;
    size_t assign_cells;
    size_t expert_contrib_cells;
};

__host__ __device__ static inline size_t dsg_align8(size_t x) {
    return (x + 7u) & ~size_t(7u);
}

__host__ static bool mul_overflow_size(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
}

__host__ static bool valid_spec_host(const Spec* s) {
    if (!s) return false;
    if (s->magic != DSG_SPEC_MAGIC || s->version != DSG_VERSION) return false;
    if (s->flags != 0 || s->reserved0 != 0 || s->reserved1 != 0) return false;
    if (s->input_dtype != DSG_DTYPE_BF16 && s->input_dtype != DSG_DTYPE_F32) return false;
    if (s->top_k < 1 || s->top_k > 4) return false;
    if (s->table_rows == 0 || s->dim == 0 || s->out_dim == 0 || s->experts == 0) return false;
    if (s->max_tokens == 0) return false;

    size_t tmp = 0;
    size_t table_cells = 0;
    size_t expert_cells = 0;
    size_t assign_cells = 0;
    size_t expert_contrib = 0;

    if (mul_overflow_size(s->table_rows, s->dim, &table_cells)) return false;
    if (mul_overflow_size(s->experts, s->dim, &tmp)) return false;
    if (mul_overflow_size(tmp, s->out_dim, &expert_cells)) return false;
    if (mul_overflow_size(s->max_tokens, s->top_k, &assign_cells)) return false;
    if (mul_overflow_size(assign_cells, s->dim, &tmp)) return false;
    if (mul_overflow_size(tmp, s->out_dim, &expert_contrib)) return false;

    if (table_cells > 524288ull) return false;
    if (expert_cells > 524288ull) return false;
    if (expert_contrib > 4194304ull) return false;

    if (s->capacity > assign_cells) return false;

    return true;
}

__device__ static inline uint64_t fnv_byte(uint64_t h, uint8_t b) {
    return (h ^ uint64_t(b)) * DSG_FNV_PRIME;
}

__device__ static inline uint64_t fnv_u8(uint64_t h, uint8_t x) {
    return fnv_byte(h, x);
}

__device__ static inline uint64_t fnv_u16(uint64_t h, uint16_t x) {
    h = fnv_byte(h, uint8_t((x >> 0) & 0xffu));
    h = fnv_byte(h, uint8_t((x >> 8) & 0xffu));
    return h;
}

__device__ static inline uint64_t fnv_i16(uint64_t h, int16_t x) {
    return fnv_u16(h, uint16_t(x));
}

__device__ static inline uint64_t fnv_u32(uint64_t h, uint32_t x) {
    h = fnv_byte(h, uint8_t((x >> 0)  & 0xffu));
    h = fnv_byte(h, uint8_t((x >> 8)  & 0xffu));
    h = fnv_byte(h, uint8_t((x >> 16) & 0xffu));
    h = fnv_byte(h, uint8_t((x >> 24) & 0xffu));
    return h;
}

__device__ static inline uint64_t fnv_i32(uint64_t h, int32_t x) {
    return fnv_u32(h, uint32_t(x));
}

__device__ static inline uint64_t fnv_u64(uint64_t h, uint64_t x) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        h = fnv_byte(h, uint8_t((x >> (8 * i)) & 0xffull));
    }
    return h;
}

__device__ static inline uint32_t f32_bits(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    return v.u;
}

__device__ static inline float f32_from_bits(uint32_t x) {
    union { float f; uint32_t u; } v;
    v.u = x;
    return v.f;
}

__device__ static inline uint32_t canon_bits(uint32_t b) {
    uint32_t exp = b & 0x7F800000u;
    uint32_t man = b & 0x007FFFFFu;
    if (exp == 0x7F800000u && man != 0) return 0x7FC00000u;
    return b;
}

__device__ static inline float canon_f32(float x) {
    return f32_from_bits(canon_bits(f32_bits(x)));
}

__device__ static inline float fmul_exact(float a, float b) {
    return canon_f32(__fmul_rn(canon_f32(a), canon_f32(b)));
}

__device__ static inline float fadd_exact(float a, float b) {
    return canon_f32(__fadd_rn(canon_f32(a), canon_f32(b)));
}

__device__ static inline float q15_to_float(int16_t q) {
    float fq = __int2float_rn(int(q));
    return canon_f32(__fmul_rn(fq, 0.000030517578125f));
}

__device__ static inline uint64_t fnv_f32(uint64_t h, float x) {
    return fnv_u32(h, canon_bits(f32_bits(x)));
}

__device__ static inline size_t dtype_size(uint32_t dtype) {
    return dtype == DSG_DTYPE_BF16 ? sizeof(uint16_t) : sizeof(uint32_t);
}

__device__ static inline size_t input_off_expert(const Spec& s, const RunSpec& r) {
    return dsg_align8(size_t(r.tokens) * sizeof(int32_t));
}

__device__ static inline size_t input_off_gate(const Spec& s, const RunSpec& r) {
    return dsg_align8(input_off_expert(s, r) + size_t(r.tokens) * s.top_k * sizeof(int32_t));
}

__device__ static inline size_t input_off_x(const Spec& s, const RunSpec& r) {
    return dsg_align8(input_off_gate(s, r) + size_t(r.tokens) * s.top_k * sizeof(int16_t));
}

__device__ static inline size_t input_off_dy(const Spec& s, const RunSpec& r) {
    return dsg_align8(input_off_x(s, r) + size_t(r.tokens) * s.dim * dtype_size(s.input_dtype));
}

__device__ static inline size_t input_off_up(const Spec& s, const RunSpec& r) {
    return dsg_align8(input_off_dy(s, r) + size_t(r.tokens) * s.out_dim * dtype_size(s.input_dtype));
}

__device__ static inline float load_input_value(
    const void* base,
    size_t array_offset,
    uint32_t dtype,
    size_t idx) {

    const uint8_t* p = static_cast<const uint8_t*>(base) + array_offset;

    if (dtype == DSG_DTYPE_BF16) {
        uint16_t raw = reinterpret_cast<const uint16_t*>(p)[idx];
        return f32_from_bits(canon_bits(uint32_t(raw) << 16));
    }

    uint32_t raw = reinterpret_cast<const uint32_t*>(p)[idx];
    return f32_from_bits(canon_bits(raw));
}

__device__ static inline float make_weight(uint64_t seed, uint64_t cell) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ull + cell * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);

    int32_t q = int32_t(z & 0xFFFFu) - 32768;
    return canon_f32(__fmul_rn(__int2float_rn(q), 0.00006103515625f));
}

__global__ static void init_persist_kernel(Persist* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        p->completed_runs = 0;

        p->total_tokens = 0;
        p->total_accepted = 0;
        p->total_dropped_capacity = 0;
        p->total_dropped_invalid_expert = 0;
        p->total_skipped_padding = 0;
        p->total_skipped_invalid_index = 0;
        p->total_table_segments = 0;
        p->total_expert_segments = 0;
        p->total_flushes = 0;

        p->cumulative_event_hash = DSG_FNV_BASIS;
    }
}

__global__ static void init_arrays_kernel(
    Spec s,
    float* table,
    float* expert,
    float* weight) {

    size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(blockDim.x) * gridDim.x;

    size_t table_n = size_t(s.table_rows) * s.dim;
    size_t expert_n = size_t(s.experts) * s.dim * s.out_dim;

    for (size_t i = tid; i < table_n; i += stride) {
        table[i] = f32_from_bits(0u);
    }

    for (size_t i = tid; i < expert_n; i += stride) {
        expert[i] = f32_from_bits(0u);
        weight[i] = make_weight(s.weight_seed, i);
    }
}

__global__ static void reset_accum_kernel(
    Spec s,
    Persist* p,
    float* table,
    float* expert) {

    size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(blockDim.x) * gridDim.x;

    size_t table_n = size_t(s.table_rows) * s.dim;
    size_t expert_n = size_t(s.experts) * s.dim * s.out_dim;

    for (size_t i = tid; i < table_n; i += stride) {
        table[i] = f32_from_bits(0u);
    }

    for (size_t i = tid; i < expert_n; i += stride) {
        expert[i] = f32_from_bits(0u);
    }

    if (tid == 0) {
        p->completed_runs = 0;
        p->total_tokens = 0;
        p->total_accepted = 0;
        p->total_dropped_capacity = 0;
        p->total_dropped_invalid_expert = 0;
        p->total_skipped_padding = 0;
        p->total_skipped_invalid_index = 0;
        p->total_table_segments = 0;
        p->total_expert_segments = 0;
        p->total_flushes = 0;
        p->cumulative_event_hash = DSG_FNV_BASIS;
    }
}

__global__ static void begin_run_kernel(
    Spec s,
    RunSpec r,
    Persist* p,
    DeviceStats* st,
    uint32_t* capacity_used,
    uint8_t* accepted,
    float* dx) {

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t cap = (r.capacity_override == DSG_CAPACITY_USE_SPEC)
        ? s.capacity
        : r.capacity_override;

    st->event_hash = DSG_FNV_BASIS;

    st->accepted = 0;
    st->dropped_capacity = 0;
    st->dropped_invalid_expert = 0;
    st->skipped_padding = 0;
    st->skipped_invalid_index = 0;

    st->table_count = 0;
    st->table_segments = 0;
    st->expert_count = 0;
    st->expert_segments = 0;

    st->effective_capacity = cap;
    st->flush_happened =
        (r.flags & (DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT | DSG_FLAG_CLEAR_AFTER_FLUSH)) ? 1u : 0u;

    for (uint32_t e = 0; e < s.experts; ++e) {
        capacity_used[e] = 0;
    }

    size_t assign_n = size_t(r.tokens) * s.top_k;
    for (size_t i = 0; i < assign_n; ++i) {
        accepted[i] = 0;
    }

    size_t dx_n = size_t(r.tokens) * s.dim;
    for (size_t i = 0; i < dx_n; ++i) {
        dx[i] = f32_from_bits(0u);
    }

    uint64_t h = st->event_hash;
    h = fnv_u8(h, 0x01);
    h = fnv_u64(h, p->completed_runs);
    h = fnv_u64(h, r.user_tag);
    h = fnv_u32(h, r.tokens);
    h = fnv_u32(h, r.flags);
    h = fnv_u32(h, cap);
    h = fnv_u32(h, s.top_k);
    h = fnv_u32(h, s.table_rows);
    h = fnv_u32(h, s.dim);
    h = fnv_u32(h, s.out_dim);
    h = fnv_u32(h, s.experts);
    h = fnv_u32(h, s.input_dtype);
    st->event_hash = h;
}

__global__ static void route_kernel(
    Spec s,
    RunSpec r,
    const void* inputs,
    Persist* p,
    DeviceStats* st,
    uint32_t* capacity_used,
    uint8_t* accepted,
    float* dx,
    const float* weight) {

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint8_t* base = static_cast<const uint8_t*>(inputs);

    const int32_t* experts =
        reinterpret_cast<const int32_t*>(base + input_off_expert(s, r));
    const int16_t* gates =
        reinterpret_cast<const int16_t*>(base + input_off_gate(s, r));

    size_t off_dy = input_off_dy(s, r);

    uint64_t h = st->event_hash;

    for (uint32_t t = 0; t < r.tokens; ++t) {
        for (uint32_t k = 0; k < s.top_k; ++k) {
            uint32_t assign = t * s.top_k + k;
            int32_t e = experts[assign];
            int16_t q = gates[assign];

            if (e < 0 || e >= int32_t(s.experts)) {
                st->dropped_invalid_expert++;

                h = fnv_u8(h, 0x03);
                h = fnv_u32(h, assign);
                h = fnv_u32(h, t);
                h = fnv_u32(h, k);
                h = fnv_i32(h, e);
                h = fnv_i16(h, q);
                h = fnv_u8(h, 1);
                continue;
            }

            uint32_t ue = uint32_t(e);
            if (capacity_used[ue] >= st->effective_capacity) {
                st->dropped_capacity++;

                h = fnv_u8(h, 0x03);
                h = fnv_u32(h, assign);
                h = fnv_u32(h, t);
                h = fnv_u32(h, k);
                h = fnv_i32(h, e);
                h = fnv_i16(h, q);
                h = fnv_u8(h, 2);
                continue;
            }

            uint32_t local_pos = capacity_used[ue]++;
            accepted[assign] = 1;
            st->accepted++;

            h = fnv_u8(h, 0x02);
            h = fnv_u32(h, assign);
            h = fnv_u32(h, t);
            h = fnv_u32(h, k);
            h = fnv_i32(h, e);
            h = fnv_i16(h, q);
            h = fnv_u32(h, local_pos);

            float gate = q15_to_float(q);

            for (uint32_t d = 0; d < s.dim; ++d) {
                float sum = f32_from_bits(0u);

                for (uint32_t o = 0; o < s.out_dim; ++o) {
                    float dy = load_input_value(
                        inputs,
                        off_dy,
                        s.input_dtype,
                        size_t(t) * s.out_dim + o);

                    float w = weight[(size_t(ue) * s.dim + d) * s.out_dim + o];

                    sum = fadd_exact(sum, fmul_exact(dy, w));
                }

                float contrib = fmul_exact(gate, sum);
                size_t dx_idx = size_t(t) * s.dim + d;
                dx[dx_idx] = fadd_exact(dx[dx_idx], contrib);
            }
        }
    }

    st->event_hash = h;
}

__global__ static void build_table_kernel(
    Spec s,
    RunSpec r,
    const void* inputs,
    DeviceStats* st,
    uint64_t* table_keys,
    uint32_t* table_tokens) {

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint8_t* base = static_cast<const uint8_t*>(inputs);
    const int32_t* indices = reinterpret_cast<const int32_t*>(base);

    uint64_t h = st->event_hash;
    uint32_t count = 0;

    for (uint32_t t = 0; t < r.tokens; ++t) {
        int32_t idx = indices[t];

        if (idx == s.padding_idx) {
            st->skipped_padding++;

            h = fnv_u8(h, 0x04);
            h = fnv_u32(h, t);
            h = fnv_i32(h, idx);
            continue;
        }

        if (idx < 0 || idx >= int32_t(s.table_rows)) {
            st->skipped_invalid_index++;

            h = fnv_u8(h, 0x05);
            h = fnv_u32(h, t);
            h = fnv_i32(h, idx);
            continue;
        }

        uint32_t row = uint32_t(idx);
        table_keys[count] = (uint64_t(row) << 32) | uint64_t(t);
        table_tokens[count] = t;
        count++;
    }

    st->table_count = count;
    st->event_hash = h;
}

__global__ static void reduce_table_kernel(
    Spec s,
    RunSpec r,
    const void* inputs,
    DeviceStats* st,
    const uint64_t* table_keys,
    const uint32_t* table_tokens,
    const float* dx,
    float* table) {

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    size_t off_up = input_off_up(s, r);
    uint64_t h = st->event_hash;

    uint32_t i = 0;
    while (i < st->table_count) {
        uint32_t row = uint32_t(table_keys[i] >> 32);
        uint32_t start = i;
        uint32_t first_arrival = uint32_t(table_keys[i] & 0xffffffffu);

        uint32_t j = i + 1;
        while (j < st->table_count && uint32_t(table_keys[j] >> 32) == row) {
            j++;
        }

        uint32_t len = j - start;
        uint32_t last_arrival = uint32_t(table_keys[j - 1] & 0xffffffffu);

        h = fnv_u8(h, 0x10);
        h = fnv_u32(h, row);
        h = fnv_u32(h, start);
        h = fnv_u32(h, len);
        h = fnv_u32(h, first_arrival);
        h = fnv_u32(h, last_arrival);

        st->table_segments++;

        for (uint32_t d = 0; d < s.dim; ++d) {
            float seg_sum = f32_from_bits(0u);

            for (uint32_t p = start; p < j; ++p) {
                uint32_t t = table_tokens[p];

                float up = load_input_value(
                    inputs,
                    off_up,
                    s.input_dtype,
                    size_t(t) * s.dim + d);

                float contrib = fadd_exact(up, dx[size_t(t) * s.dim + d]);
                seg_sum = fadd_exact(seg_sum, contrib);
            }

            size_t idx = size_t(row) * s.dim + d;
            table[idx] = fadd_exact(table[idx], seg_sum);
        }

        i = j;
    }

    st->event_hash = h;
}

__global__ static void build_expert_kernel(
    Spec s,
    RunSpec r,
    const void* inputs,
    const uint8_t* accepted,
    DeviceStats* st,
    uint64_t* expert_keys,
    uint32_t* expert_assigns) {

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint8_t* base = static_cast<const uint8_t*>(inputs);
    const int32_t* experts =
        reinterpret_cast<const int32_t*>(base + input_off_expert(s, r));

    uint32_t count = 0;
    uint32_t assign_n = r.tokens * s.top_k;

    for (uint32_t assign = 0; assign < assign_n; ++assign) {
        if (!accepted[assign]) continue;

        uint32_t e = uint32_t(experts[assign]);

        for (uint32_t d = 0; d < s.dim; ++d) {
            for (uint32_t o = 0; o < s.out_dim; ++o) {
                uint32_t cell = ((e * s.dim + d) * s.out_dim + o);
                expert_keys[count] = (uint64_t(cell) << 32) | uint64_t(assign);
                expert_assigns[count] = assign;
                count++;
            }
        }
    }

    st->expert_count = count;
}

__global__ static void reduce_expert_kernel(
    Spec s,
    RunSpec r,
    const void* inputs,
    DeviceStats* st,
    const uint64_t* expert_keys,
    const uint32_t* expert_assigns,
    float* expert) {

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint8_t* base = static_cast<const uint8_t*>(inputs);
    const int16_t* gates =
        reinterpret_cast<const int16_t*>(base + input_off_gate(s, r));

    size_t off_x = input_off_x(s, r);
    size_t off_dy = input_off_dy(s, r);

    uint64_t h = st->event_hash;

    uint32_t i = 0;
    while (i < st->expert_count) {
        uint32_t cell = uint32_t(expert_keys[i] >> 32);
        uint32_t start = i;
        uint32_t first_assign = uint32_t(expert_keys[i] & 0xffffffffu);

        uint32_t j = i + 1;
        while (j < st->expert_count && uint32_t(expert_keys[j] >> 32) == cell) {
            j++;
        }

        uint32_t len = j - start;
        uint32_t last_assign = uint32_t(expert_keys[j - 1] & 0xffffffffu);

        uint32_t o = cell % s.out_dim;
        uint32_t tmp = cell / s.out_dim;
        uint32_t d = tmp % s.dim;
        uint32_t e = tmp / s.dim;

        h = fnv_u8(h, 0x20);
        h = fnv_u32(h, e);
        h = fnv_u32(h, d);
        h = fnv_u32(h, o);
        h = fnv_u32(h, start);
        h = fnv_u32(h, len);
        h = fnv_u32(h, first_assign);
        h = fnv_u32(h, last_assign);

        st->expert_segments++;

        float seg_sum = f32_from_bits(0u);

        for (uint32_t p = start; p < j; ++p) {
            uint32_t assign = expert_assigns[p];
            uint32_t t = assign / s.top_k;

            float gate = q15_to_float(gates[assign]);

            float x = load_input_value(
                inputs,
                off_x,
                s.input_dtype,
                size_t(t) * s.dim + d);

            float dy = load_input_value(
                inputs,
                off_dy,
                s.input_dtype,
                size_t(t) * s.out_dim + o);

            float prod = fmul_exact(x, dy);
            float contrib = fmul_exact(prod, gate);

            seg_sum = fadd_exact(seg_sum, contrib);
        }

        expert[cell] = fadd_exact(expert[cell], seg_sum);

        i = j;
    }

    st->event_hash = h;
}

__device__ static inline uint64_t hash_f32_array(uint64_t h, const float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        h = fnv_f32(h, p[i]);
    }
    return h;
}

__global__ static void emit_kernel(
    Spec s,
    RunSpec r,
    Persist* p,
    DeviceStats* st,
    const float* dx,
    float* table,
    float* expert,
    void* outputs) {

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint8_t* out = static_cast<uint8_t*>(outputs);
    OutHeader* hdr = reinterpret_cast<OutHeader*>(out);

    size_t dx_n = size_t(r.tokens) * s.dim;
    size_t table_n = size_t(s.table_rows) * s.dim;
    size_t expert_n = size_t(s.experts) * s.dim * s.out_dim;

    size_t off = sizeof(OutHeader);

    float* out_dx = reinterpret_cast<float*>(out + off);
    for (size_t i = 0; i < dx_n; ++i) {
        out_dx[i] = canon_f32(dx[i]);
    }

    uint64_t tensor_hash = DSG_FNV_BASIS;
    tensor_hash = hash_f32_array(tensor_hash, out_dx, dx_n);

    off = dsg_align8(off + dx_n * sizeof(float));

    uint64_t table_bytes = 0;
    uint64_t expert_bytes = 0;

    if (r.flags & DSG_FLAG_EMIT_TABLE) {
        float* out_table = reinterpret_cast<float*>(out + off);
        for (size_t i = 0; i < table_n; ++i) {
            out_table[i] = canon_f32(table[i]);
        }
        tensor_hash = hash_f32_array(tensor_hash, out_table, table_n);
        table_bytes = uint64_t(table_n * sizeof(float));
        off = dsg_align8(off + table_n * sizeof(float));
    }

    if (r.flags & DSG_FLAG_EMIT_EXPERT) {
        float* out_expert = reinterpret_cast<float*>(out + off);
        for (size_t i = 0; i < expert_n; ++i) {
            out_expert[i] = canon_f32(expert[i]);
        }
        tensor_hash = hash_f32_array(tensor_hash, out_expert, expert_n);
        expert_bytes = uint64_t(expert_n * sizeof(float));
        off = dsg_align8(off + expert_n * sizeof(float));
    }

    uint64_t event_hash = st->event_hash;

    if (st->flush_happened) {
        event_hash = fnv_u8(event_hash, 0x30);
        event_hash = fnv_u32(event_hash, r.flags);
        event_hash = fnv_u64(event_hash, table_bytes);
        event_hash = fnv_u64(event_hash, expert_bytes);
        event_hash = fnv_u8(event_hash, (r.flags & DSG_FLAG_CLEAR_AFTER_FLUSH) ? 1 : 0);
    }

    if (r.flags & DSG_FLAG_CLEAR_AFTER_FLUSH) {
        for (size_t i = 0; i < table_n; ++i) {
            table[i] = f32_from_bits(0u);
        }
        for (size_t i = 0; i < expert_n; ++i) {
            expert[i] = f32_from_bits(0u);
        }
    }

    uint64_t old_run_id = p->completed_runs;
    uint64_t completed_runs = old_run_id + 1ull;

    p->completed_runs = completed_runs;

    p->total_tokens += uint64_t(r.tokens);
    p->total_accepted += uint64_t(st->accepted);
    p->total_dropped_capacity += uint64_t(st->dropped_capacity);
    p->total_dropped_invalid_expert += uint64_t(st->dropped_invalid_expert);
    p->total_skipped_padding += uint64_t(st->skipped_padding);
    p->total_skipped_invalid_index += uint64_t(st->skipped_invalid_index);
    p->total_table_segments += uint64_t(st->table_segments);
    p->total_expert_segments += uint64_t(st->expert_segments);
    p->total_flushes += uint64_t(st->flush_happened);

    p->cumulative_event_hash = fnv_u64(p->cumulative_event_hash, event_hash);

    uint64_t state_hash = DSG_FNV_BASIS;
    state_hash = fnv_u32(state_hash, DSG_SPEC_MAGIC);
    state_hash = fnv_u32(state_hash, DSG_VERSION);
    state_hash = fnv_u32(state_hash, s.table_rows);
    state_hash = fnv_u32(state_hash, s.dim);
    state_hash = fnv_u32(state_hash, s.out_dim);
    state_hash = fnv_u32(state_hash, s.experts);
    state_hash = fnv_u32(state_hash, s.top_k);
    state_hash = fnv_u32(state_hash, s.max_tokens);
    state_hash = fnv_u32(state_hash, s.capacity);
    state_hash = fnv_u32(state_hash, s.input_dtype);
    state_hash = fnv_i32(state_hash, s.padding_idx);
    state_hash = fnv_u64(state_hash, s.weight_seed);

    state_hash = fnv_u64(state_hash, p->completed_runs);
    state_hash = fnv_u64(state_hash, p->total_tokens);
    state_hash = fnv_u64(state_hash, p->total_accepted);
    state_hash = fnv_u64(state_hash, p->total_dropped_capacity);
    state_hash = fnv_u64(state_hash, p->total_dropped_invalid_expert);
    state_hash = fnv_u64(state_hash, p->total_skipped_padding);
    state_hash = fnv_u64(state_hash, p->total_skipped_invalid_index);
    state_hash = fnv_u64(state_hash, p->total_table_segments);
    state_hash = fnv_u64(state_hash, p->total_expert_segments);
    state_hash = fnv_u64(state_hash, p->total_flushes);
    state_hash = fnv_u64(state_hash, p->cumulative_event_hash);

    state_hash = hash_f32_array(state_hash, table, table_n);
    state_hash = hash_f32_array(state_hash, expert, expert_n);

    uint64_t step_hash = DSG_FNV_BASIS;
    step_hash = fnv_u64(step_hash, event_hash);
    step_hash = fnv_u64(step_hash, tensor_hash);
    step_hash = fnv_u64(step_hash, state_hash);

    step_hash = fnv_u64(step_hash, old_run_id);
    step_hash = fnv_u64(step_hash, completed_runs);
    step_hash = fnv_u32(step_hash, r.tokens);
    step_hash = fnv_u32(step_hash, r.flags);
    step_hash = fnv_u32(step_hash, st->effective_capacity);

    step_hash = fnv_u32(step_hash, st->accepted);
    step_hash = fnv_u32(step_hash, st->dropped_capacity);
    step_hash = fnv_u32(step_hash, st->dropped_invalid_expert);
    step_hash = fnv_u32(step_hash, st->skipped_padding);
    step_hash = fnv_u32(step_hash, st->skipped_invalid_index);
    step_hash = fnv_u32(step_hash, st->table_segments);
    step_hash = fnv_u32(step_hash, st->expert_segments);
    step_hash = fnv_u32(step_hash, st->flush_happened);

    step_hash = fnv_u64(step_hash, p->total_tokens);
    step_hash = fnv_u64(step_hash, p->total_accepted);
    step_hash = fnv_u64(step_hash, p->total_dropped_capacity);
    step_hash = fnv_u64(step_hash, p->total_dropped_invalid_expert);
    step_hash = fnv_u64(step_hash, p->total_skipped_padding);
    step_hash = fnv_u64(step_hash, p->total_skipped_invalid_index);
    step_hash = fnv_u64(step_hash, p->total_table_segments);
    step_hash = fnv_u64(step_hash, p->total_expert_segments);
    step_hash = fnv_u64(step_hash, p->total_flushes);

    hdr->magic = DSG_OUT_MAGIC;
    hdr->version = uint16_t(DSG_VERSION);
    hdr->header_bytes = uint16_t(sizeof(OutHeader));

    hdr->run_id = old_run_id;
    hdr->completed_runs = completed_runs;

    hdr->tokens = r.tokens;
    hdr->flags = r.flags;
    hdr->effective_capacity = st->effective_capacity;
    hdr->top_k = s.top_k;

    hdr->accepted = st->accepted;
    hdr->dropped_capacity = st->dropped_capacity;
    hdr->dropped_invalid_expert = st->dropped_invalid_expert;
    hdr->skipped_padding = st->skipped_padding;
    hdr->skipped_invalid_index = st->skipped_invalid_index;
    hdr->table_segments = st->table_segments;
    hdr->expert_segments = st->expert_segments;
    hdr->flush_happened = st->flush_happened;

    hdr->total_tokens = p->total_tokens;
    hdr->total_accepted = p->total_accepted;
    hdr->total_dropped_capacity = p->total_dropped_capacity;
    hdr->total_dropped_invalid_expert = p->total_dropped_invalid_expert;
    hdr->total_skipped_padding = p->total_skipped_padding;
    hdr->total_skipped_invalid_index = p->total_skipped_invalid_index;
    hdr->total_table_segments = p->total_table_segments;
    hdr->total_expert_segments = p->total_expert_segments;
    hdr->total_flushes = p->total_flushes;

    hdr->event_hash = event_hash;
    hdr->tensor_hash = tensor_hash;
    hdr->state_hash = state_hash;
    hdr->step_hash = step_hash;

    hdr->reserved0 = 0;
    hdr->reserved1 = 0;
}

extern "C" size_t solution_workspace_bytes(const Spec* spec) {
    if (!valid_spec_host(spec)) return 0;
    return 0;
}

static cudaError_t free_ref_state(RefState* st) {
    if (!st) return cudaSuccess;

    cudaError_t first = cudaSuccess;
    auto free_one = [&](void* p) {
        if (p) {
            cudaError_t e = cudaFree(p);
            if (first == cudaSuccess && e != cudaSuccess) first = e;
        }
    };

    free_one(st->d_persist);
    free_one(st->d_stats);
    free_one(st->d_table);
    free_one(st->d_expert);
    free_one(st->d_weight);
    free_one(st->d_dx);
    free_one(st->d_capacity_used);
    free_one(st->d_accepted);
    free_one(st->d_table_keys);
    free_one(st->d_table_tokens);
    free_one(st->d_expert_keys);
    free_one(st->d_expert_assigns);

    free(st);
    return first;
}

extern "C" cudaError_t solution_init(
    const Spec* spec,
    void** state,
    cudaStream_t stream) {

    if (!state) return cudaErrorInvalidValue;
    *state = nullptr;

    if (!valid_spec_host(spec)) return cudaErrorInvalidValue;

    RefState* st = static_cast<RefState*>(calloc(1, sizeof(RefState)));
    if (!st) return cudaErrorMemoryAllocation;

    st->spec = *spec;

    st->table_cells = size_t(spec->table_rows) * spec->dim;
    st->expert_cells = size_t(spec->experts) * spec->dim * spec->out_dim;
    st->assign_cells = size_t(spec->max_tokens) * spec->top_k;
    st->expert_contrib_cells = st->assign_cells * spec->dim * spec->out_dim;

    cudaError_t err = cudaSuccess;

    auto alloc = [&](void** p, size_t bytes) -> bool {
        size_t n = bytes ? bytes : 1;
        err = cudaMalloc(p, n);
        return err == cudaSuccess;
    };

    if (!alloc(reinterpret_cast<void**>(&st->d_persist), sizeof(Persist))) goto fail;
    if (!alloc(reinterpret_cast<void**>(&st->d_stats), sizeof(DeviceStats))) goto fail;

    if (!alloc(reinterpret_cast<void**>(&st->d_table), st->table_cells * sizeof(float))) goto fail;
    if (!alloc(reinterpret_cast<void**>(&st->d_expert), st->expert_cells * sizeof(float))) goto fail;
    if (!alloc(reinterpret_cast<void**>(&st->d_weight), st->expert_cells * sizeof(float))) goto fail;

    if (!alloc(reinterpret_cast<void**>(&st->d_dx), size_t(spec->max_tokens) * spec->dim * sizeof(float))) goto fail;

    if (!alloc(reinterpret_cast<void**>(&st->d_capacity_used), size_t(spec->experts) * sizeof(uint32_t))) goto fail;
    if (!alloc(reinterpret_cast<void**>(&st->d_accepted), st->assign_cells * sizeof(uint8_t))) goto fail;

    if (!alloc(reinterpret_cast<void**>(&st->d_table_keys), size_t(spec->max_tokens) * sizeof(uint64_t))) goto fail;
    if (!alloc(reinterpret_cast<void**>(&st->d_table_tokens), size_t(spec->max_tokens) * sizeof(uint32_t))) goto fail;

    if (!alloc(reinterpret_cast<void**>(&st->d_expert_keys), st->expert_contrib_cells * sizeof(uint64_t))) goto fail;
    if (!alloc(reinterpret_cast<void**>(&st->d_expert_assigns), st->expert_contrib_cells * sizeof(uint32_t))) goto fail;

    init_persist_kernel<<<1, 1, 0, stream>>>(st->d_persist);

    {
        int threads = 256;
        size_t max_cells = st->table_cells > st->expert_cells ? st->table_cells : st->expert_cells;
        int blocks = int((max_cells + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        if (blocks > 4096) blocks = 4096;

        init_arrays_kernel<<<blocks, threads, 0, stream>>>(
            st->spec,
            st->d_table,
            st->d_expert,
            st->d_weight);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;

    *state = st;
    return cudaSuccess;

fail:
    free_ref_state(st);
    *state = nullptr;
    return err == cudaSuccess ? cudaErrorMemoryAllocation : err;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {

    if (!state) return cudaErrorInvalidValue;

    RefState* st = static_cast<RefState*>(state);

    int threads = 256;
    size_t max_cells = st->table_cells > st->expert_cells ? st->table_cells : st->expert_cells;
    int blocks = int((max_cells + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;

    reset_accum_kernel<<<blocks, threads, 0, stream>>>(
        st->spec,
        st->d_persist,
        st->d_table,
        st->d_expert);

    return cudaGetLastError();
}

extern "C" cudaError_t solution_run(
    void* state,
    const RunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {

    (void)workspace;
    (void)workspace_bytes;

    if (!state || !run || !inputs || !outputs) return cudaErrorInvalidValue;

    RefState* st = static_cast<RefState*>(state);
    const Spec& s = st->spec;

    if (run->magic != DSG_RUN_MAGIC || run->version != DSG_VERSION) return cudaErrorInvalidValue;
    if (run->reserved0 != 0) return cudaErrorInvalidValue;
    if (run->flags & ~DSG_VALID_RUN_FLAGS) return cudaErrorInvalidValue;
    if (run->tokens > s.max_tokens) return cudaErrorInvalidValue;

    size_t max_assign = size_t(s.max_tokens) * s.top_k;
    if (run->capacity_override != DSG_CAPACITY_USE_SPEC &&
        uint64_t(run->capacity_override) > uint64_t(max_assign)) {
        return cudaErrorInvalidValue;
    }

    begin_run_kernel<<<1, 1, 0, stream>>>(
        s,
        *run,
        st->d_persist,
        st->d_stats,
        st->d_capacity_used,
        st->d_accepted,
        st->d_dx);

    route_kernel<<<1, 1, 0, stream>>>(
        s,
        *run,
        inputs,
        st->d_persist,
        st->d_stats,
        st->d_capacity_used,
        st->d_accepted,
        st->d_dx,
        st->d_weight);

    build_table_kernel<<<1, 1, 0, stream>>>(
        s,
        *run,
        inputs,
        st->d_stats,
        st->d_table_keys,
        st->d_table_tokens);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    DeviceStats hstats;
    err = cudaMemcpyAsync(&hstats, st->d_stats, sizeof(DeviceStats), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    if (hstats.table_count > 1) {
        thrust::device_ptr<uint64_t> kptr(st->d_table_keys);
        thrust::device_ptr<uint32_t> vptr(st->d_table_tokens);

        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            kptr,
            kptr + hstats.table_count,
            vptr);
    }

    reduce_table_kernel<<<1, 1, 0, stream>>>(
        s,
        *run,
        inputs,
        st->d_stats,
        st->d_table_keys,
        st->d_table_tokens,
        st->d_dx,
        st->d_table);

    build_expert_kernel<<<1, 1, 0, stream>>>(
        s,
        *run,
        inputs,
        st->d_accepted,
        st->d_stats,
        st->d_expert_keys,
        st->d_expert_assigns);

    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(&hstats, st->d_stats, sizeof(DeviceStats), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    if (hstats.expert_count > 1) {
        thrust::device_ptr<uint64_t> kptr(st->d_expert_keys);
        thrust::device_ptr<uint32_t> vptr(st->d_expert_assigns);

        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            kptr,
            kptr + hstats.expert_count,
            vptr);
    }

    reduce_expert_kernel<<<1, 1, 0, stream>>>(
        s,
        *run,
        inputs,
        st->d_stats,
        st->d_expert_keys,
        st->d_expert_assigns,
        st->d_expert);

    emit_kernel<<<1, 1, 0, stream>>>(
        s,
        *run,
        st->d_persist,
        st->d_stats,
        st->d_dx,
        st->d_table,
        st->d_expert,
        outputs);

    return cudaGetLastError();
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    RefState* st = static_cast<RefState*>(state);
    free_ref_state(st);
}
