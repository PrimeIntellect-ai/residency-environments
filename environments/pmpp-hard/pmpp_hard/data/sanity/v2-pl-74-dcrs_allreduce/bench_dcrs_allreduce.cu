// bench_dcrs_allreduce.cu — THE PERF GATE for dcrs_allreduce_stream_v2.
//
// Drives the solution_* ABI on LARGE shapes (large per-rank item counts M, several
// ranks R, wide cell/segment counts C, repeated solution_run calls so the running
// accumulator + cross-rank merge dominate). Each call reduces tens of millions of
// leaf elements; a real device solution finishes in a few ms, while a serial CPU
// host-replay (copy input D->H, reduce on one thread, copy output H->D) is orders of
// magnitude slower and blows the gate.
//
// Does NOT check correctness (that is test_dcrs_allreduce). It only times.
// Prints exactly `avg_ms=%.6f\n` = total_ms / (iters * num_cases).
//
// Usage: ./bench_* [iters]   (default iters=20)

#include "dcrs_allreduce_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Solution ABI
// ---------------------------------------------------------------------------
extern "C" {
    size_t       solution_workspace_bytes(const DcrSpec*);
    cudaError_t  solution_init(const DcrSpec*, void**, cudaStream_t);
    cudaError_t  solution_run(void*, const DcrRunSpec*, const void*, void*, void*, size_t, cudaStream_t);
    cudaError_t  solution_reset(void*, cudaStream_t);
    void         solution_destroy(void*);
}

#define CHK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
    return 1; } } while(0)

// ---------------------------------------------------------------------------
// Deterministic input generator (SplitMix64 -> modest-magnitude f32 in [-2,2]).
// Magnitudes stay small so 20+ accumulating runs never overflow to Inf; timing
// is independent of the exact values anyway.
// ---------------------------------------------------------------------------
struct SM64 {
    uint64_t s;
    explicit SM64(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    float f32() {
        uint32_t u = (uint32_t)(next() >> 32);
        return (float)((int32_t)u) * (2.0f / 2147483648.0f);
    }
};

static uint64_t align8(uint64_t x) { return (x + 7u) & ~uint64_t(7); }

// ---------------------------------------------------------------------------
// One large bench case: a single op in a single run, with its own input/output
// device buffers but sharing the engine state + workspace.
// ---------------------------------------------------------------------------
struct BenchCase {
    std::string name;
    DcrOpDesc op;          // tensor_offset/state_offset filled here for the (single-op) layout
    uint64_t input_bytes;

    // layout (single op, header-first)
    uint64_t tensor_region_offset, tensor_region_bytes;
    uint64_t timeline_region_offset, timeline_region_bytes;
    uint64_t state_region_offset, state_region_bytes;
    uint64_t counters_offset, counters_bytes;
    uint64_t output_bytes;

    uint8_t* d_in  = nullptr;
    uint8_t* d_out = nullptr;
};

static void build_layout(BenchCase& c) {
    uint32_t out_b = (c.op.output_dtype == DCR_OUT_BF16) ? 2u : 4u;
    c.tensor_region_offset  = sizeof(DcrOutputHeader);               // 112, 8-aligned
    c.tensor_region_bytes   = (uint64_t)c.op.logical_ranks * c.op.cells * out_b;
    c.timeline_region_offset = align8(c.tensor_region_offset + c.tensor_region_bytes);
    c.timeline_region_bytes  = sizeof(DcrTimelineRecord);            // 1 op
    c.state_region_offset    = align8(c.timeline_region_offset + c.timeline_region_bytes);
    c.state_region_bytes     = (uint64_t)c.op.cells * sizeof(DcrStateRecord);
    c.counters_offset        = align8(c.state_region_offset + c.state_region_bytes);
    c.counters_bytes         = sizeof(DcrCountersRecord);
    c.output_bytes           = c.counters_offset + c.counters_bytes;
    c.op.tensor_offset = c.tensor_region_offset;
    c.op.state_offset  = c.state_region_offset;
}

// Fill h_input for a case (returns raw bytes). SUM/NORM: R*M*C values.
// SEG: R*M values followed by R*M uint32 keys (keys_offset already set).
static std::vector<uint8_t> make_input(const DcrOpDesc& op, uint64_t seed) {
    SM64 rng(seed);
    uint32_t R = op.logical_ranks, C = op.cells, M = op.items_per_rank;
    std::vector<uint8_t> buf;

    if (op.mode == DCR_MODE_SEG) {
        uint64_t n = (uint64_t)R * M;
        buf.resize(n * 4 + n * 4);
        uint8_t* vbase = buf.data();
        uint8_t* kbase = buf.data() + n * 4;
        for (uint64_t i = 0; i < n; ++i) {
            float f = rng.f32();
            memcpy(vbase + i * 4, &f, 4);
        }
        // keys spread across the C segments (plus ~10% out-of-range -> ignored)
        for (uint64_t i = 0; i < n; ++i) {
            uint32_t r = (uint32_t)(rng.next() % (C + (C / 10 + 1)));
            memcpy(kbase + i * 4, &r, 4);
        }
        return buf;
    }

    uint64_t n = (uint64_t)R * M * C;
    if (op.input_dtype == DCR_DTYPE_BF16) {
        buf.resize(n * 2);
        for (uint64_t i = 0; i < n; ++i) {
            float f = rng.f32();
            uint32_t u; memcpy(&u, &f, 4);
            uint16_t bf = (uint16_t)(u >> 16);
            memcpy(buf.data() + i * 2, &bf, 2);
        }
    } else {
        buf.resize(n * 4);
        for (uint64_t i = 0; i < n; ++i) {
            float f = rng.f32();
            memcpy(buf.data() + i * 4, &f, 4);
        }
    }
    return buf;
}

static int run_case(void* state, const BenchCase& c, uint8_t* d_ws, size_t ws,
                    cudaStream_t stream) {
    DcrRunSpec run;
    memset(&run, 0, sizeof(run));
    run.abi_version            = DCR_ABI_VERSION;
    run.op_count               = 1;
    run.input_bytes            = c.input_bytes;
    run.output_bytes           = c.output_bytes;
    run.header_offset          = 0;
    run.tensor_region_offset   = c.tensor_region_offset;
    run.tensor_region_bytes    = c.tensor_region_bytes;
    run.timeline_region_offset = c.timeline_region_offset;
    run.timeline_region_bytes  = c.timeline_region_bytes;
    run.state_region_offset    = c.state_region_offset;
    run.state_region_bytes     = c.state_region_bytes;
    run.counters_offset        = c.counters_offset;
    run.counters_bytes         = c.counters_bytes;
    run.ops[0]                 = c.op;
    cudaError_t e = solution_run(state, &run, c.d_in, c.d_out, d_ws, ws, stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "solution_run(%s) error: %s\n", c.name.c_str(), cudaGetErrorString(e));
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int iters = 20;
    if (argc >= 2) { iters = atoi(argv[1]); if (iters < 1) iters = 1; }

    int ndev = 0; cudaGetDeviceCount(&ndev);
    if (ndev == 0) { fprintf(stderr, "No CUDA devices.\n"); return 1; }
    CHK(cudaSetDevice(0));

    // ---- engine spec covering every case's maxes ----
    const uint32_t R = 8, C = 256, M = 16384;
    DcrSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.abi_version        = DCR_ABI_VERSION;
    spec.max_accumulators   = 4;
    spec.max_ranks          = R;
    spec.max_cells          = C;
    spec.max_items_per_rank = M;
    spec.max_ops_per_run    = 4;
    spec.allow_bf16         = 1;
    spec.max_input_bytes    = (uint64_t)1 << 31;
    spec.max_output_bytes   = (uint64_t)1 << 20;
    spec.flags              = 0;

    // ---- define the LARGE cases (distinct acc_id + fixed mode => always valid) ----
    std::vector<BenchCase> cases(4);

    auto mk_op = [&](uint32_t mode, uint32_t in_dt, uint32_t out_dt, uint32_t acc) {
        DcrOpDesc op; memset(&op, 0, sizeof(op));
        op.mode = mode; op.input_dtype = in_dt; op.output_dtype = out_dt; op.acc_id = acc;
        op.logical_ranks = R; op.cells = C; op.items_per_rank = M;
        op.segment_count = (mode == DCR_MODE_SEG) ? C : 0u;
        op.values_offset = 0;
        op.keys_offset   = (mode == DCR_MODE_SEG) ? (uint64_t)R * M * 4u : 0u;
        return op;
    };

    cases[0].name = "sum_f32_r8c256m16384";   cases[0].op = mk_op(DCR_MODE_SUM,  DCR_DTYPE_F32,  DCR_OUT_F32,  0);
    cases[1].name = "norm_f32_r8c256m16384";  cases[1].op = mk_op(DCR_MODE_NORM, DCR_DTYPE_F32,  DCR_OUT_F32,  1);
    cases[2].name = "seg_f32_r8s256m16384";   cases[2].op = mk_op(DCR_MODE_SEG,  DCR_DTYPE_F32,  DCR_OUT_F32,  2);
    cases[3].name = "sum_bf16_r8c256m16384";  cases[3].op = mk_op(DCR_MODE_SUM,  DCR_DTYPE_BF16, DCR_OUT_BF16, 3);

    cudaStream_t stream = nullptr;
    CHK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CHK(solution_init(&spec, &state, stream));
    CHK(cudaStreamSynchronize(stream));

    size_t ws = solution_workspace_bytes(&spec);
    uint8_t* d_ws = nullptr;
    CHK(cudaMalloc(&d_ws, ws > 0 ? ws : 1));

    uint64_t seed = 0x1234abcdULL ^ 0x000000000008ce7bULL;
    uint64_t total_in = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        BenchCase& c = cases[i];
        std::vector<uint8_t> h_in = make_input(c.op, seed + i);
        c.input_bytes = h_in.size();
        build_layout(c);

        CHK(cudaMalloc(&c.d_in,  c.input_bytes));
        CHK(cudaMalloc(&c.d_out, c.output_bytes));
        CHK(cudaMemcpyAsync(c.d_in, h_in.data(), c.input_bytes, cudaMemcpyHostToDevice, stream));
        total_in += c.input_bytes;
        printf("bench_case %-26s R=%u C=%u M=%u mode=%u in=%llu bytes\n",
               c.name.c_str(), c.op.logical_ranks, c.op.cells, c.op.items_per_rank,
               c.op.mode, (unsigned long long)c.input_bytes);
    }
    CHK(cudaStreamSynchronize(stream));
    printf("total_input=%.1f MB  workspace=%.2f MB  iters=%d  cases=%zu\n",
           total_in / 1048576.0, ws / 1048576.0, iters, cases.size());

    // ---- warmup (~3 passes over all cases) ----
    for (int w = 0; w < 3; ++w)
        for (size_t i = 0; i < cases.size(); ++i)
            if (run_case(state, cases[i], d_ws, ws, stream)) return 1;
    CHK(cudaStreamSynchronize(stream));

    // ---- timed loop ----
    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));
    CHK(cudaEventRecord(start, stream));

    for (int it = 0; it < iters; ++it)
        for (size_t i = 0; i < cases.size(); ++i)
            if (run_case(state, cases[i], d_ws, ws, stream)) return 1;

    CHK(cudaEventRecord(stop, stream));
    CHK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double calls = (double)iters * (double)cases.size();
    double avg_ms = (double)elapsed_ms / calls;
    printf("avg_ms=%.6f\n", avg_ms);

    // ---- teardown ----
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (size_t i = 0; i < cases.size(); ++i) {
        if (cases[i].d_in)  cudaFree(cases[i].d_in);
        if (cases[i].d_out) cudaFree(cases[i].d_out);
    }
    cudaFree(d_ws);
    solution_destroy(state);
    cudaStreamDestroy(stream);
    return 0;
}
