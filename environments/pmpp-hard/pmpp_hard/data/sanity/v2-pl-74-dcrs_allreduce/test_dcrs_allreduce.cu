// test_dcrs_allreduce.cu — self-contained correctness gate for dcrs_allreduce_stream_v2
// Generates deterministic inputs, calls the solution ABI, reads back the raw output
// bytes, and INDEPENDENTLY recomputes FNV-1a-64 (project basis 1469598103934665603)
// over those bytes. Each run's five recomputed checksums are compared against the
// EMBEDDED golden values (captured from the validated reference on RTX 5080 sm_120).
// A case PASSES only if every one of its runs matches all five golden checksums.
// Prints exactly `passed N / M` over the M=7 cases. Does NOT embed/call the reference.
// Unspecified knobs (LCG seed for generic cases) fixed at seed=42; documented below.

#include "dcrs_allreduce_common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <cstring>

// ---------------------------------------------------------------------------
// FNV-1a-64 helpers (host-side, project basis 1469598103934665603)
// ---------------------------------------------------------------------------
static uint64_t fnv_u8(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= (uint64_t)DCR_FNV_PRIME;
    return h;
}
static uint64_t fnv_buf(uint64_t h, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < len; ++i) h = fnv_u8(h, p[i]);
    return h;
}
static uint64_t fnv_u32_le(uint64_t h, uint32_t x) {
    for (int i = 0; i < 4; ++i) h = fnv_u8(h, (uint8_t)((x >> (8*i)) & 0xffu));
    return h;
}
static uint64_t fnv_u64_le(uint64_t h, uint64_t x) {
    for (int i = 0; i < 8; ++i) h = fnv_u8(h, (uint8_t)((x >> (8*i)) & 0xffu));
    return h;
}
static uint64_t fnv_region_fresh(const void* data, size_t len) {
    return fnv_buf(DCR_FNV_BASIS, data, len);
}
static uint64_t compute_fnv_all(
    const uint8_t* out,
    uint64_t tensor_off, uint64_t tensor_bytes,
    uint64_t timeline_off, uint64_t timeline_bytes,
    uint64_t state_off,   uint64_t state_bytes,
    uint64_t cntr_off,    uint64_t cntr_bytes)
{
    uint64_t h = DCR_FNV_BASIS;
    h = fnv_u32_le(h, 0x54454E53u); // "TENS"
    h = fnv_u64_le(h, tensor_bytes);
    h = fnv_buf(h, out + tensor_off, (size_t)tensor_bytes);
    h = fnv_u32_le(h, 0x54494D45u); // "TIME"
    h = fnv_u64_le(h, timeline_bytes);
    h = fnv_buf(h, out + timeline_off, (size_t)timeline_bytes);
    h = fnv_u32_le(h, 0x53544154u); // "STAT"
    h = fnv_u64_le(h, state_bytes);
    h = fnv_buf(h, out + state_off, (size_t)state_bytes);
    h = fnv_u32_le(h, 0x434E5452u); // "CNTR"
    h = fnv_u64_le(h, cntr_bytes);
    h = fnv_buf(h, out + cntr_off, (size_t)cntr_bytes);
    return h;
}

// ---------------------------------------------------------------------------
// Embedded golden checksums (captured from the validated reference on
// RTX 5080 sm_120, CUDA 13). One record per solution_run, flattened in the
// exact dispatch order used by main(). A run matches iff all five recomputed
// FNV values equal the golden record. g_gix walks this table run-by-run.
// AGGREGATE HASH golden: 0xb4e417a40265d49f  (informational; seed-shifted sanity variant)
// ---------------------------------------------------------------------------
struct GoldenRun {
    uint64_t tensor;
    uint64_t timeline;
    uint64_t state;
    uint64_t counters;
    uint64_t all;
};

static const GoldenRun g_gold[] = {
    // case_single_sum  (1 run)
    { 0x2fa183a08531045aULL, 0x3376d0846c9fe8f5ULL, 0x7ae2b41c3b2bd6a3ULL, 0x28146a175ba950c6ULL, 0x8018886457baf561ULL },
    // case_empty_sum  (1 run)
    { 0x43899d1d0dde10c3ULL, 0x0f635abcc07708b0ULL, 0x70ee56f20cc49ef5ULL, 0xddaa87db4a18b579ULL, 0xec84739624353cabULL },
    // case_bf16_sum  (1 run)
    { 0x39ab0d41754f8877ULL, 0xb943c858d733d6afULL, 0x8a64aa8031ae4009ULL, 0x6aacbeefd4617bc2ULL, 0xa8fab7796855ee60ULL },
    // case_seg_dup_keys  (1 run)
    { 0xa0881003fcb36903ULL, 0xf8df1f67432c7a8cULL, 0xf3e68076f1d57419ULL, 0xc65577b62bf8d017ULL, 0x468596e6721f8d08ULL },
    // case_norm_subnormals  (1 run)
    { 0x70fda04fef70caffULL, 0xbe0f8f702f10b2d4ULL, 0x4b883eb9530f0130ULL, 0xf249dce25c766469ULL, 0x0fefaf4c65cc6a1aULL },
    // case_multirun_sum  (3 runs)
    { 0x61c1966a8df36597ULL, 0x5bfc940fc65e098fULL, 0x9beebc450aa8a50cULL, 0x5cbb66607385e6e9ULL, 0xca71e26efae89ce1ULL },
    { 0x61c1966a8df36597ULL, 0x9f45179d5addf8faULL, 0xc16f14544ac1c6b1ULL, 0xb9ffd1406e3496bfULL, 0x6eb14d9da7301ef3ULL },
    { 0x61c1966a8df36597ULL, 0x4eeac6b52c2e5151ULL, 0xaf1f0c7f82d7dee6ULL, 0xa4559e5a4a7e0449ULL, 0x8b413d02241bcb49ULL },
    // case_mode_mismatch  (2 runs)
    { 0xb970d66a503c485bULL, 0x462093481437c4a4ULL, 0xf33257c6f2efd5fcULL, 0xdb81a64e059f3905ULL, 0x782360afc6c874e1ULL },
    { 0x1bcc89338cdd826fULL, 0xf1d05a721476e829ULL, 0xe0fa5a0e1b0abba3ULL, 0x7968a27898e52974ULL, 0xe8488b6c614feae6ULL },
};
static int g_gix = 0; // cursor into g_gold, advanced once per execute_one_run

// ---------------------------------------------------------------------------
// LCG for deterministic input generation.
// Seed for all generic cases: 42.  Seed for subnormal/special case: 123.
// ---------------------------------------------------------------------------
struct LCG {
    uint64_t s;
    explicit LCG(uint64_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963409ull;
        return (uint32_t)(s >> 32);
    }
    // Normal-range f32 in [-4, 4] from LCG bits
    float next_f32() {
        uint32_t u = next();
        float f = (float)(int32_t(u)) * (4.0f / 2147483648.0f);
        return f;
    }
    // BF16 bits (upper 16 of a small float)
    uint16_t next_bf16() {
        // Keep exponent in a reasonable range; just use upper 16 of a normal f32.
        union { float f; uint32_t u; } v;
        v.f = next_f32();
        return (uint16_t)(v.u >> 16);
    }
};

// ---------------------------------------------------------------------------
// Output layout helper — lays regions sequentially after the header.
// All ops are assumed valid (no poison-region sizing needed for gate cases).
// ---------------------------------------------------------------------------
struct OutputLayout {
    uint64_t header_offset;
    uint64_t tensor_region_offset;
    uint64_t tensor_region_bytes;
    uint64_t timeline_region_offset;
    uint64_t timeline_region_bytes;
    uint64_t state_region_offset;
    uint64_t state_region_bytes;
    uint64_t counters_offset;
    uint64_t counters_bytes;
    uint64_t total_bytes;
    // per-op offsets relative to region start
    uint64_t op_tensor_offset[DCR_MAX_OPS_PER_RUN];
    uint64_t op_state_offset[DCR_MAX_OPS_PER_RUN];
};

// Round up to 8-byte alignment (required for uint64_t fields in the struct types).
static uint64_t align8(uint64_t x) { return (x + 7u) & ~uint64_t(7); }

static OutputLayout make_layout(uint32_t op_count,
                                 const DcrOpDesc* ops) {
    OutputLayout L;
    memset(&L, 0, sizeof(L));
    // sizeof(DcrOutputHeader) = 112, which is 8-byte aligned, so header at 0 is fine.
    L.header_offset = 0;
    L.tensor_region_offset = sizeof(DcrOutputHeader); // 112, 8-byte aligned
    uint64_t tensor_cursor = 0;
    uint64_t state_cursor = 0;
    for (uint32_t i = 0; i < op_count; ++i) {
        uint32_t out_b = (ops[i].output_dtype == DCR_OUT_BF16) ? 2u : 4u;
        uint64_t tb = uint64_t(ops[i].logical_ranks) * ops[i].cells * out_b;
        uint64_t sb = uint64_t(ops[i].cells) * sizeof(DcrStateRecord);
        L.op_tensor_offset[i] = tensor_cursor;
        L.op_state_offset[i] = state_cursor;
        tensor_cursor += tb;
        state_cursor += sb;
    }
    L.tensor_region_bytes = tensor_cursor;
    // Align timeline region to 8 bytes (DcrTimelineRecord starts with uint64_t).
    L.timeline_region_offset = align8(L.tensor_region_offset + L.tensor_region_bytes);
    L.timeline_region_bytes = uint64_t(op_count) * sizeof(DcrTimelineRecord);
    // Align state region to 8 bytes (DcrStateRecord contains uint64_t fields).
    L.state_region_offset = align8(L.timeline_region_offset + L.timeline_region_bytes);
    L.state_region_bytes = state_cursor;
    // Align counters to 8 bytes (DcrCountersRecord starts with uint64_t magic).
    L.counters_offset = align8(L.state_region_offset + L.state_region_bytes);
    L.counters_bytes = sizeof(DcrCountersRecord);
    L.total_bytes = L.counters_offset + L.counters_bytes;
    return L;
}

// ---------------------------------------------------------------------------
// Solution ABI declarations
// ---------------------------------------------------------------------------
extern "C" {
    size_t       solution_workspace_bytes(const DcrSpec*);
    cudaError_t  solution_init(const DcrSpec*, void**, cudaStream_t);
    cudaError_t  solution_run(void*, const DcrRunSpec*, const void*, void*, void*, size_t, cudaStream_t);
    cudaError_t  solution_reset(void*, cudaStream_t);
    void         solution_destroy(void*);
}

// ---------------------------------------------------------------------------
// CUDA error check macro
// ---------------------------------------------------------------------------
#define CHK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    return false; } } while(0)

// ---------------------------------------------------------------------------
// Per-run execution: fills RunSpec from layout, calls solution_run, reads back,
// verifies FNV, prints results.  Returns false on any failure.
// ---------------------------------------------------------------------------
static bool execute_one_run(
    const char* case_name,
    uint32_t run_idx,
    void* state,
    const OutputLayout& L,
    uint32_t op_count,
    DcrOpDesc* ops,         // will have tensor_offset / state_offset filled in
    void* d_input,
    uint64_t input_bytes,
    void* d_output,
    void* d_workspace,
    size_t ws_bytes,
    std::vector<uint8_t>& h_output,
    cudaStream_t stream,
    uint64_t& aggregate_hash)
{
    // Clear output buffer so stale bytes don't pollute the FNV
    CHK(cudaMemsetAsync(d_output, 0, L.total_bytes, stream));

    // Build RunSpec
    DcrRunSpec run;
    memset(&run, 0, sizeof(run));
    run.abi_version            = DCR_ABI_VERSION;
    run.op_count               = op_count;
    run.input_bytes            = input_bytes;
    run.output_bytes           = L.total_bytes;
    run.header_offset          = L.header_offset;
    run.tensor_region_offset   = L.tensor_region_offset;
    run.tensor_region_bytes    = L.tensor_region_bytes;
    run.timeline_region_offset = L.timeline_region_offset;
    run.timeline_region_bytes  = L.timeline_region_bytes;
    run.state_region_offset    = L.state_region_offset;
    run.state_region_bytes     = L.state_region_bytes;
    run.counters_offset        = L.counters_offset;
    run.counters_bytes         = L.counters_bytes;

    for (uint32_t i = 0; i < op_count; ++i) {
        run.ops[i] = ops[i];
        run.ops[i].tensor_offset = L.tensor_region_offset + L.op_tensor_offset[i];
        run.ops[i].state_offset  = L.state_region_offset  + L.op_state_offset[i];
    }

    cudaError_t err = solution_run(state, &run, d_input, d_output, d_workspace, ws_bytes, stream);
    if (err != cudaSuccess) {
        printf("CASE %s run %u: solution_run error: %s\n", case_name, run_idx, cudaGetErrorString(err));
        return false;
    }
    CHK(cudaStreamSynchronize(stream));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CASE %s run %u: post-run CUDA error: %s\n", case_name, run_idx, cudaGetErrorString(err));
        return false;
    }

    // Copy output back to host
    CHK(cudaMemcpy(h_output.data(), d_output, L.total_bytes, cudaMemcpyDeviceToHost));

    // Read header
    DcrOutputHeader hdr;
    memcpy(&hdr, h_output.data() + L.header_offset, sizeof(DcrOutputHeader));

    // Harness independently computes FNV checksums from raw output bytes
    uint64_t hc_tensor   = fnv_region_fresh(h_output.data() + L.tensor_region_offset,   (size_t)L.tensor_region_bytes);
    uint64_t hc_timeline = fnv_region_fresh(h_output.data() + L.timeline_region_offset, (size_t)L.timeline_region_bytes);
    uint64_t hc_state    = fnv_region_fresh(h_output.data() + L.state_region_offset,    (size_t)L.state_region_bytes);
    uint64_t hc_counters = fnv_region_fresh(h_output.data() + L.counters_offset,        (size_t)L.counters_bytes);
    uint64_t hc_all      = compute_fnv_all(
        h_output.data(),
        L.tensor_region_offset,   L.tensor_region_bytes,
        L.timeline_region_offset, L.timeline_region_bytes,
        L.state_region_offset,    L.state_region_bytes,
        L.counters_offset,        L.counters_bytes);

    // Pass criterion: the INDEPENDENTLY-recomputed checksums (over the raw output
    // bytes the solution actually wrote) must equal the embedded golden values.
    // We also require the solution's self-reported header to be self-consistent
    // with those bytes, so a solution cannot fake the header checksums.
    const GoldenRun& gold =
        (g_gix < (int)(sizeof(g_gold) / sizeof(g_gold[0]))) ? g_gold[g_gix]
                                                            : g_gold[0];
    g_gix += 1;

    bool self_consistent =
        (hdr.fnv_tensor   == hc_tensor)   &&
        (hdr.fnv_timeline == hc_timeline) &&
        (hdr.fnv_state    == hc_state)    &&
        (hdr.fnv_counters == hc_counters) &&
        (hdr.fnv_all      == hc_all);

    bool t_ok  = (hc_tensor   == gold.tensor);
    bool tl_ok = (hc_timeline == gold.timeline);
    bool s_ok  = (hc_state    == gold.state);
    bool c_ok  = (hc_counters == gold.counters);
    bool a_ok  = (hc_all      == gold.all);

    printf("CASE %-36s  run=%u  fnv_tensor=0x%016llx  fnv_timeline=0x%016llx  fnv_state=0x%016llx  fnv_counters=0x%016llx  fnv_all=0x%016llx\n",
           case_name, run_idx,
           (unsigned long long)hc_tensor,
           (unsigned long long)hc_timeline,
           (unsigned long long)hc_state,
           (unsigned long long)hc_counters,
           (unsigned long long)hc_all);
    printf("  verify(vs golden): tensor=%s timeline=%s state=%s counters=%s all=%s  header=%s\n",
           t_ok ? "OK" : "MISMATCH",
           tl_ok ? "OK" : "MISMATCH",
           s_ok  ? "OK" : "MISMATCH",
           c_ok  ? "OK" : "MISMATCH",
           a_ok  ? "OK" : "MISMATCH",
           self_consistent ? "consistent" : "INCONSISTENT");

    if (!t_ok || !tl_ok || !s_ok || !c_ok || !a_ok || !self_consistent) return false;

    // Fold fnv_all into running aggregate via FNV step
    aggregate_hash = fnv_u8(aggregate_hash ^ (uint8_t)(hc_all & 0xffu), 0);
    aggregate_hash ^= hc_all;
    aggregate_hash *= DCR_FNV_PRIME;

    return true;
}

// ---------------------------------------------------------------------------
// Individual test cases
// ---------------------------------------------------------------------------

// Case 1: Single rank, single element SUM, f32->f32, input=[1.0f]
// Seed: N/A (explicit input)
static bool case_single_sum(uint64_t& agg) {
    const char* name = "single_sum_r1c1m1_f32";
    DcrSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.abi_version     = DCR_ABI_VERSION;
    spec.max_accumulators = 2;
    spec.max_ranks       = 1;
    spec.max_cells       = 1;
    spec.max_items_per_rank = 1;
    spec.max_ops_per_run = 8;
    spec.allow_bf16      = 1;
    spec.max_input_bytes  = 1 << 20;
    spec.max_output_bytes = 1 << 20;

    DcrOpDesc op;
    memset(&op, 0, sizeof(op));
    op.mode          = DCR_MODE_SUM;
    op.input_dtype   = DCR_DTYPE_F32;
    op.output_dtype  = DCR_OUT_F32;
    op.acc_id        = 0;
    op.logical_ranks = 1;
    op.cells         = 1;
    op.items_per_rank = 1;
    op.segment_count = 0;
    op.values_offset = 0;
    // tensor_offset and state_offset filled by execute_one_run

    OutputLayout L = make_layout(1, &op);

    // Input: [1.0f]
    float val = 1.0f;
    std::vector<uint8_t> h_input(4);
    memcpy(h_input.data(), &val, 4);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* state = nullptr;
    CHK(solution_init(&spec, &state, stream));
    size_t ws = solution_workspace_bytes(&spec);

    void *d_in = nullptr, *d_out = nullptr, *d_ws = nullptr;
    CHK(cudaMalloc(&d_in,  h_input.size()));
    CHK(cudaMalloc(&d_out, L.total_bytes));
    CHK(cudaMalloc(&d_ws,  ws > 0 ? ws : 1));
    CHK(cudaMemcpyAsync(d_in, h_input.data(), h_input.size(), cudaMemcpyHostToDevice, stream));

    std::vector<uint8_t> h_out(L.total_bytes, 0);
    bool ok = execute_one_run(name, 0, state, L, 1, &op,
                              d_in, h_input.size(), d_out, d_ws, ws, h_out, stream, agg);

    solution_destroy(state);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ws);
    cudaStreamDestroy(stream);
    return ok;
}

// Case 2: Empty SUM, R=4, C=7, M=0, f32->f32
// Seed: N/A (no input data)
static bool case_empty_sum(uint64_t& agg) {
    const char* name = "empty_sum_r4c7m0_f32";
    DcrSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.abi_version     = DCR_ABI_VERSION;
    spec.max_accumulators = 2;
    spec.max_ranks       = 4;
    spec.max_cells       = 7;
    spec.max_items_per_rank = 0;
    spec.max_ops_per_run = 8;
    spec.allow_bf16      = 1;
    spec.max_input_bytes  = 1 << 20;
    spec.max_output_bytes = 1 << 20;

    DcrOpDesc op;
    memset(&op, 0, sizeof(op));
    op.mode          = DCR_MODE_SUM;
    op.input_dtype   = DCR_DTYPE_F32;
    op.output_dtype  = DCR_OUT_F32;
    op.acc_id        = 0;
    op.logical_ranks = 4;
    op.cells         = 7;
    op.items_per_rank = 0;
    op.segment_count = 0;
    op.values_offset = 0;

    OutputLayout L = make_layout(1, &op);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* state = nullptr;
    CHK(solution_init(&spec, &state, stream));
    size_t ws = solution_workspace_bytes(&spec);

    void *d_in = nullptr, *d_out = nullptr, *d_ws = nullptr;
    CHK(cudaMalloc(&d_in,  1)); // no actual input bytes
    CHK(cudaMalloc(&d_out, L.total_bytes > 0 ? L.total_bytes : 1));
    CHK(cudaMalloc(&d_ws,  ws > 0 ? ws : 1));

    std::vector<uint8_t> h_out(L.total_bytes, 0);
    bool ok = execute_one_run(name, 0, state, L, 1, &op,
                              d_in, 0, d_out, d_ws, ws, h_out, stream, agg);

    solution_destroy(state);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ws);
    cudaStreamDestroy(stream);
    return ok;
}

// Case 3: BF16 input/output SUM, R=2, C=31, M=33
// Seed: 42 for bf16 values (R*M*C = 2046 bf16 elements)
static bool case_bf16_sum(uint64_t& agg) {
    const char* name = "bf16_sum_r2c31m33";
    const uint32_t R=2, C=31, M=33;
    DcrSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.abi_version     = DCR_ABI_VERSION;
    spec.max_accumulators = 2;
    spec.max_ranks       = R;
    spec.max_cells       = C;
    spec.max_items_per_rank = M;
    spec.max_ops_per_run = 8;
    spec.allow_bf16      = 1;
    spec.max_input_bytes  = 1 << 20;
    spec.max_output_bytes = 1 << 20;

    DcrOpDesc op;
    memset(&op, 0, sizeof(op));
    op.mode          = DCR_MODE_SUM;
    op.input_dtype   = DCR_DTYPE_BF16;
    op.output_dtype  = DCR_OUT_BF16;
    op.acc_id        = 0;
    op.logical_ranks = R;
    op.cells         = C;
    op.items_per_rank = M;
    op.segment_count = 0;
    op.values_offset = 0;

    OutputLayout L = make_layout(1, &op);

    // Generate R*M*C bf16 values, seed=42
    uint64_t n_elem = uint64_t(R) * M * C;
    std::vector<uint16_t> bf16_vals(n_elem);
    LCG lcg(42);
    for (uint64_t i = 0; i < n_elem; ++i) bf16_vals[i] = lcg.next_bf16();
    // Insert two half-way rounding cases at positions 0 and 1
    // half-way: bit 16 of f32 == 1, bits 15..0 == 0x8000 exactly
    // e.g. 0x3F808000 -> bf16 rounds to 0x3F81 (round up since lsb=1)
    //      0x3F800000 -> bf16 = 0x3F80 (exact)
    bf16_vals[0] = 0x3F80u; // exact 1.0 in bf16
    bf16_vals[1] = 0x3F81u; // 1.0078125 in bf16

    std::vector<uint8_t> h_input(n_elem * 2);
    memcpy(h_input.data(), bf16_vals.data(), n_elem * 2);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* state = nullptr;
    CHK(solution_init(&spec, &state, stream));
    size_t ws = solution_workspace_bytes(&spec);

    void *d_in = nullptr, *d_out = nullptr, *d_ws = nullptr;
    CHK(cudaMalloc(&d_in,  h_input.size()));
    CHK(cudaMalloc(&d_out, L.total_bytes));
    CHK(cudaMalloc(&d_ws,  ws > 0 ? ws : 1));
    CHK(cudaMemcpyAsync(d_in, h_input.data(), h_input.size(), cudaMemcpyHostToDevice, stream));

    std::vector<uint8_t> h_out(L.total_bytes, 0);
    bool ok = execute_one_run(name, 0, state, L, 1, &op,
                              d_in, h_input.size(), d_out, d_ws, ws, h_out, stream, agg);

    solution_destroy(state);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ws);
    cudaStreamDestroy(stream);
    return ok;
}

// Case 4: SEG, R=4, S=8 (cells=8,segment_count=8), M=257, f32->f32
// Key pattern (seed=42): 60% go to seg 3 (heavy dup), segs 1 and 6 intentionally empty,
// ~10% keys >= S=8 (out-of-range, ignored).
// Values: LCG seed=42 f32.
static bool case_seg_dup_keys(uint64_t& agg) {
    const char* name = "seg_r4s8m257_f32";
    const uint32_t R=4, S=8, M=257;
    DcrSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.abi_version     = DCR_ABI_VERSION;
    spec.max_accumulators = 2;
    spec.max_ranks       = R;
    spec.max_cells       = S;
    spec.max_items_per_rank = M;
    spec.max_ops_per_run = 8;
    spec.allow_bf16      = 1;
    spec.max_input_bytes  = 1 << 20;
    spec.max_output_bytes = 1 << 20;

    DcrOpDesc op;
    memset(&op, 0, sizeof(op));
    op.mode          = DCR_MODE_SEG;
    op.input_dtype   = DCR_DTYPE_F32;
    op.output_dtype  = DCR_OUT_F32;
    op.acc_id        = 0;
    op.logical_ranks = R;
    op.cells         = S;
    op.items_per_rank = M;
    op.segment_count = S;
    op.values_offset = 0;
    op.keys_offset   = uint64_t(R) * M * 4; // after f32 values

    OutputLayout L = make_layout(1, &op);

    uint64_t n_vals = uint64_t(R) * M;
    // values: R*M f32
    std::vector<float> vals(n_vals);
    LCG lcg_v(42);
    for (uint64_t i = 0; i < n_vals; ++i) vals[i] = lcg_v.next_f32();

    // keys: R*M uint32_t
    // Pattern: segs 1 and 6 empty; heavy dups on seg 3; ~10% out-of-range (key=9)
    // For rank r, item m:
    //   if (m + r) % 10 == 1 or == 6: key = 9 (out-of-range, ensures segs 1&6 see no valid items)
    //   elif (m + r) % 10 == 0: key = 0
    //   elif (m + r) % 10 == 2: key = 2
    //   elif (m + r) % 10 == 4: key = 4
    //   elif (m + r) % 10 == 5: key = 5
    //   elif (m + r) % 10 == 7: key = 7
    //   elif (m + r) % 10 == 8: key = 8  (out-of-range, >= S=8, also ignored)
    //   elif (m + r) % 10 == 9: key = 9  (out-of-range)
    //   else: key = 3  (everything else -> seg 3, heavy dup)
    std::vector<uint32_t> keys(n_vals);
    for (uint32_t r = 0; r < R; ++r) {
        for (uint32_t m = 0; m < M; ++m) {
            uint32_t bucket = (m + r) % 10;
            uint32_t key;
            switch(bucket) {
                case 0: key = 0; break;
                case 1: key = 9; break;  // would be seg 1, but forced out-of-range
                case 2: key = 2; break;
                case 4: key = 4; break;
                case 5: key = 5; break;
                case 6: key = 9; break;  // would be seg 6, but forced out-of-range
                case 7: key = 7; break;
                case 8: key = 9; break;  // out-of-range
                case 9: key = 9; break;  // out-of-range
                default: key = 3; break; // case 3 -> heavy dup
            }
            keys[r * M + m] = key;
        }
    }

    uint64_t val_bytes = n_vals * 4;
    uint64_t key_bytes = n_vals * 4;
    std::vector<uint8_t> h_input(val_bytes + key_bytes);
    memcpy(h_input.data(),            vals.data(), val_bytes);
    memcpy(h_input.data() + val_bytes, keys.data(), key_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* state = nullptr;
    CHK(solution_init(&spec, &state, stream));
    size_t ws = solution_workspace_bytes(&spec);

    void *d_in = nullptr, *d_out = nullptr, *d_ws = nullptr;
    CHK(cudaMalloc(&d_in,  h_input.size()));
    CHK(cudaMalloc(&d_out, L.total_bytes));
    CHK(cudaMalloc(&d_ws,  ws > 0 ? ws : 1));
    CHK(cudaMemcpyAsync(d_in, h_input.data(), h_input.size(), cudaMemcpyHostToDevice, stream));

    std::vector<uint8_t> h_out(L.total_bytes, 0);
    bool ok = execute_one_run(name, 0, state, L, 1, &op,
                              d_in, h_input.size(), d_out, d_ws, ws, h_out, stream, agg);

    solution_destroy(state);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ws);
    cudaStreamDestroy(stream);
    return ok;
}

// Case 5: NORM with signed zeros and subnormals, R=2, C=5, M=17, f32->f32
// Seed: 123. First 4 values per rank forced to +0, -0, min_subnorm, -min_subnorm.
static bool case_norm_subnormals(uint64_t& agg) {
    const char* name = "norm_r2c5m17_f32";
    const uint32_t R=2, C=5, M=17;
    DcrSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.abi_version     = DCR_ABI_VERSION;
    spec.max_accumulators = 2;
    spec.max_ranks       = R;
    spec.max_cells       = C;
    spec.max_items_per_rank = M;
    spec.max_ops_per_run = 8;
    spec.allow_bf16      = 1;
    spec.max_input_bytes  = 1 << 20;
    spec.max_output_bytes = 1 << 20;

    DcrOpDesc op;
    memset(&op, 0, sizeof(op));
    op.mode          = DCR_MODE_NORM;
    op.input_dtype   = DCR_DTYPE_F32;
    op.output_dtype  = DCR_OUT_F32;
    op.acc_id        = 0;
    op.logical_ranks = R;
    op.cells         = C;
    op.items_per_rank = M;
    op.segment_count = 0;
    op.values_offset = 0;

    OutputLayout L = make_layout(1, &op);

    // Input: values[r][m][c], layout: ((r*M+m)*C+c)
    uint64_t n_elem = uint64_t(R) * M * C;
    std::vector<uint32_t> raw(n_elem);
    LCG lcg(123);
    for (uint64_t i = 0; i < n_elem; ++i) {
        float f = lcg.next_f32();
        uint32_t u; memcpy(&u, &f, 4);
        raw[i] = u;
    }
    // Inject special values at specific positions (c=0, various r,m)
    // values[0][0][0] = +0.0f (index 0)
    raw[0] = 0x00000000u;
    // values[0][1][0] = -0.0f (index C = 5)
    raw[5] = 0x80000000u;
    // values[0][2][0] = min positive subnormal (index 2*C = 10)
    raw[10] = 0x00000001u;
    // values[0][3][0] = min negative subnormal (index 3*C = 15)
    raw[15] = 0x80000001u;
    // values[1][0][0] = largest subnormal positive (index R*M*C/2 area, at r=1,m=0,c=0 = M*C = 85)
    raw[M*C] = 0x007FFFFFu;

    std::vector<uint8_t> h_input(n_elem * 4);
    memcpy(h_input.data(), raw.data(), n_elem * 4);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* state = nullptr;
    CHK(solution_init(&spec, &state, stream));
    size_t ws = solution_workspace_bytes(&spec);

    void *d_in = nullptr, *d_out = nullptr, *d_ws = nullptr;
    CHK(cudaMalloc(&d_in,  h_input.size()));
    CHK(cudaMalloc(&d_out, L.total_bytes));
    CHK(cudaMalloc(&d_ws,  ws > 0 ? ws : 1));
    CHK(cudaMemcpyAsync(d_in, h_input.data(), h_input.size(), cudaMemcpyHostToDevice, stream));

    std::vector<uint8_t> h_out(L.total_bytes, 0);
    bool ok = execute_one_run(name, 0, state, L, 1, &op,
                              d_in, h_input.size(), d_out, d_ws, ws, h_out, stream, agg);

    solution_destroy(state);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ws);
    cudaStreamDestroy(stream);
    return ok;
}

// Case 6: Multi-run SUM, R=2, C=4, M=33, f32->f32, 3 runs, acc_id=0
// State accumulates across runs.
// Seed: 42 for values (same input buffer re-used each run).
static bool case_multirun_sum(uint64_t& agg) {
    const char* name = "multirun_sum_r2c4m33_3runs";
    const uint32_t R=2, C=4, M=33;
    DcrSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.abi_version     = DCR_ABI_VERSION;
    spec.max_accumulators = 2;
    spec.max_ranks       = R;
    spec.max_cells       = C;
    spec.max_items_per_rank = M;
    spec.max_ops_per_run = 8;
    spec.allow_bf16      = 1;
    spec.max_input_bytes  = 1 << 20;
    spec.max_output_bytes = 1 << 20;

    DcrOpDesc op;
    memset(&op, 0, sizeof(op));
    op.mode          = DCR_MODE_SUM;
    op.input_dtype   = DCR_DTYPE_F32;
    op.output_dtype  = DCR_OUT_F32;
    op.acc_id        = 0;
    op.logical_ranks = R;
    op.cells         = C;
    op.items_per_rank = M;
    op.segment_count = 0;
    op.values_offset = 0;

    OutputLayout L = make_layout(1, &op);

    // Generate R*M*C f32 values, seed=42; inject a large and tiny pair
    uint64_t n_elem = uint64_t(R) * M * C;
    std::vector<float> vals(n_elem);
    LCG lcg(42);
    for (uint64_t i = 0; i < n_elem; ++i) vals[i] = lcg.next_f32();
    // Inject alternating large/tiny at positions 0..3 (r=0,m=0..3,c=0)
    vals[0*C+0] =  1e7f;  // large
    vals[1*C+0] =  1e-7f; // tiny
    vals[2*C+0] = -1e7f;  // large negative
    vals[3*C+0] =  1e-7f; // tiny

    std::vector<uint8_t> h_input(n_elem * 4);
    memcpy(h_input.data(), vals.data(), n_elem * 4);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* state = nullptr;
    CHK(solution_init(&spec, &state, stream));
    size_t ws = solution_workspace_bytes(&spec);

    void *d_in = nullptr, *d_out = nullptr, *d_ws = nullptr;
    CHK(cudaMalloc(&d_in,  h_input.size()));
    CHK(cudaMalloc(&d_out, L.total_bytes > 0 ? L.total_bytes : 1));
    CHK(cudaMalloc(&d_ws,  ws > 0 ? ws : 1));
    CHK(cudaMemcpyAsync(d_in, h_input.data(), h_input.size(), cudaMemcpyHostToDevice, stream));

    std::vector<uint8_t> h_out(L.total_bytes, 0);
    bool ok = true;
    for (uint32_t run = 0; run < 3; ++run) {
        bool r = execute_one_run(name, run, state, L, 1, &op,
                                 d_in, h_input.size(), d_out, d_ws, ws, h_out, stream, agg);
        if (!r) { ok = false; break; }
    }

    solution_destroy(state);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ws);
    cudaStreamDestroy(stream);
    return ok;
}

// Case 7: Mode mismatch error path — run 1 SUM acc_id=1, run 2 NORM acc_id=1
// Harness verifies the second op produces poison tensor and MODE_MISMATCH timeline.
static bool case_mode_mismatch(uint64_t& agg) {
    const char* name = "mode_mismatch_acc1";
    const uint32_t R=2, C=3, M=4;
    DcrSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.abi_version     = DCR_ABI_VERSION;
    spec.max_accumulators = 4;
    spec.max_ranks       = R;
    spec.max_cells       = C;
    spec.max_items_per_rank = M;
    spec.max_ops_per_run = 8;
    spec.allow_bf16      = 1;
    spec.max_input_bytes  = 1 << 20;
    spec.max_output_bytes = 1 << 20;

    // Run 1: acc_id=1, SUM
    DcrOpDesc op1;
    memset(&op1, 0, sizeof(op1));
    op1.mode = DCR_MODE_SUM; op1.input_dtype = DCR_DTYPE_F32; op1.output_dtype = DCR_OUT_F32;
    op1.acc_id = 1; op1.logical_ranks = R; op1.cells = C; op1.items_per_rank = M;
    op1.values_offset = 0;

    // Run 2: acc_id=1, NORM (mode mismatch!)
    DcrOpDesc op2;
    memset(&op2, 0, sizeof(op2));
    op2.mode = DCR_MODE_NORM; op2.input_dtype = DCR_DTYPE_F32; op2.output_dtype = DCR_OUT_F32;
    op2.acc_id = 1; op2.logical_ranks = R; op2.cells = C; op2.items_per_rank = M;
    op2.values_offset = 0;

    // Run 1 layout (valid SUM)
    OutputLayout L1 = make_layout(1, &op1);
    // Run 2 layout (invalid NORM, but harness allocates tensor as if R*C*4)
    OutputLayout L2 = make_layout(1, &op2);

    uint64_t n_elem = uint64_t(R) * M * C;
    std::vector<float> vals(n_elem);
    LCG lcg(42);
    for (uint64_t i = 0; i < n_elem; ++i) vals[i] = lcg.next_f32();
    std::vector<uint8_t> h_input(n_elem * 4);
    memcpy(h_input.data(), vals.data(), n_elem * 4);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* state = nullptr;
    CHK(solution_init(&spec, &state, stream));
    size_t ws = solution_workspace_bytes(&spec);

    void *d_in = nullptr, *d_out = nullptr, *d_ws = nullptr;
    uint64_t max_out = (L1.total_bytes > L2.total_bytes) ? L1.total_bytes : L2.total_bytes;
    CHK(cudaMalloc(&d_in,  h_input.size()));
    CHK(cudaMalloc(&d_out, max_out));
    CHK(cudaMalloc(&d_ws,  ws > 0 ? ws : 1));
    CHK(cudaMemcpyAsync(d_in, h_input.data(), h_input.size(), cudaMemcpyHostToDevice, stream));

    std::vector<uint8_t> h_out(max_out, 0);
    // Run 1: valid SUM
    bool ok = execute_one_run(name, 0, state, L1, 1, &op1,
                              d_in, h_input.size(), d_out, d_ws, ws, h_out, stream, agg);
    // Run 2: NORM on same acc_id -> MODE_MISMATCH
    // The harness still verifies checksums match self-reported header values (solution must be internally consistent).
    if (ok) {
        h_out.resize(L2.total_bytes, 0);
        bool ok2 = execute_one_run(name, 1, state, L2, 1, &op2,
                                   d_in, h_input.size(), d_out, d_ws, ws, h_out, stream, agg);
        // Note: ok2 may fail checksum if solution doesn't handle mismatch correctly,
        // but for the reference gate we expect it to pass.
        ok = ok2;
    }

    solution_destroy(state);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ws);
    cudaStreamDestroy(stream);
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Verify CUDA device available
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    if (ndev == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s  sm_%d%d\n", prop.name, prop.major, prop.minor);

    // Running aggregate hash (FNV-folded across all fnv_all values)
    uint64_t aggregate = DCR_FNV_BASIS;

    struct { const char* name; bool(*fn)(uint64_t&); } cases[] = {
        { "case_single_sum",     case_single_sum     },
        { "case_empty_sum",      case_empty_sum      },
        { "case_bf16_sum",       case_bf16_sum       },
        { "case_seg_dup_keys",   case_seg_dup_keys   },
        { "case_norm_subnormals",case_norm_subnormals},
        { "case_multirun_sum",   case_multirun_sum   },
        { "case_mode_mismatch",  case_mode_mismatch  },
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);
    int n_pass = 0, n_fail = 0;

    for (int i = 0; i < n_cases; ++i) {
        printf("\n--- %s ---\n", cases[i].name);
        bool ok = cases[i].fn(aggregate);
        printf("  result: %s\n", ok ? "PASS" : "FAIL");
        if (ok) ++n_pass; else ++n_fail;
    }

    printf("\n=== AGGREGATE HASH: 0x%016llx ===\n", (unsigned long long)aggregate);
    printf("=== SUMMARY: %d/%d cases PASS ===\n", n_pass, n_cases);
    // Grader-facing line (greps `passed [0-9]+ / [0-9]+`):
    printf("passed %d / %d\n", n_pass, n_cases);
    return (n_fail == 0) ? 0 : 1;
}
