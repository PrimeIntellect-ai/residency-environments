// bench_norm_backward.cu — PERF GATE for stateful_norm_bwd_cachegrad_v2.
//
// Drives the solution_* ABI on a LARGE shape (large hidden_size N and many
// backward rows per run) so that the reduction work dominates. A correct GPU
// implementation runs in a few ms per run, while a serial CPU host-replay
// solution is orders of magnitude over budget. This harness does NOT check
// correctness (test_norm_backward.cu does that); it only times solution_run.
//
// Prints exactly one machine-readable line: avg_ms=<total_ms / K>

#include "norm_backward_common.h"
#include "pmpp_bench_digest.cuh"
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(expr) do {                                       \
    cudaError_t _e = (expr);                                        \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
            cudaGetErrorString(_e), __FILE__, __LINE__);            \
        exit(1);                                                    \
    }                                                               \
} while(0)

// ---------------------------------------------------------------------------
// Deterministic input generation (no reference math here)
// ---------------------------------------------------------------------------
static uint64_t g_rng = 0x243f6a8885a308d3ull;
static uint64_t splitmix64_h(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static uint64_t rng_next() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}
static float rng_f32() {
    uint32_t u = uint32_t(rng_next() >> 32);
    float f = float(u >> 8) / float(1u << 24);
    return f * 2.0f - 1.0f;
}
static uint16_t f32_to_bf16_rne_host(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if (((u & 0x7f800000u) == 0x7f800000u) && (u & 0x007fffffu)) return 0x7fc0u;
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t bias = 0x7fffu + lsb;
    return uint16_t((u + bias) >> 16);
}
static void fill_bf16(std::vector<uint8_t>& buf, uint32_t rows, uint32_t stride) {
    buf.resize(size_t(rows) * stride * 2);
    for (size_t i = 0; i < size_t(rows) * stride; ++i) {
        uint16_t h = f32_to_bf16_rne_host(rng_f32());
        memcpy(buf.data() + i * 2, &h, 2);
    }
}

int main() {
    // ----- LARGE bench shape (design §6.3 hard composition: N=8193 class,
    //       many backward rows; here N=8192 with the full per-run backward
    //       budget so reduction work dominates) -----
    const uint32_t N            = 8192;   // hidden_size
    const uint32_t PARAM_COUNT  = 4;
    const uint32_t ROWS         = 512;    // saved + backward rows per op
    const uint32_t N_BWD_OPS    = 4;      // 4 * 512 = 2048 backward rows / run
    const uint32_t STRIDE       = N;

    Spec s{};
    s.magic   = LNRBWD_MAGIC;
    s.version = LNRBWD_VERSION;
    s.hidden_size               = N;
    s.param_count               = PARAM_COUNT;
    s.max_cache_rows            = ROWS;
    s.max_input_rows_per_run    = ROWS;
    s.max_dx_rows_per_run       = ROWS;
    s.max_backward_rows_per_run = N_BWD_OPS * ROWS;   // 2048
    s.max_ops_per_run           = 16;
    s.max_flush_records_per_run = 4;
    s.storage_dtype             = LNR_DTYPE_BF16;
    s.eps_ln                    = 1e-5f;
    s.eps_rms                   = 1e-5f;
    s.counter_seed              = 0;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&s, &state, stream));

    size_t wbytes = solution_workspace_bytes(&s);
    void* workspace = nullptr;
    if (wbytes) CUDA_CHECK(cudaMalloc(&workspace, wbytes));

    // ----- Inputs -----
    // C3: every timed run k gets its OWN input variant (x/dy/gamma), derived from
    // PMPP_BENCH_SEED so the same seed yields a bit-identical variant sequence in
    // the student and reference benches. Variant K is reserved for warmup so
    // warmup can never pre-populate a digested region. Shapes stay fixed.
    const int K = 20;   // timed runs (one input+output variant each)
    g_rng = splitmix64_h(pmpp::bench_seed(0x243f6a8885a308d3ull)) | 1ull;

    std::vector<void*> d_x(K + 1), d_dy(K + 1), d_gamma(K + 1);
    for (int k = 0; k <= K; ++k) {
        std::vector<uint8_t> h_x, h_dy, h_gamma;
        fill_bf16(h_x,     ROWS,        STRIDE);
        fill_bf16(h_dy,    ROWS,        STRIDE);
        fill_bf16(h_gamma, PARAM_COUNT, STRIDE);
        CUDA_CHECK(cudaMalloc(&d_x[k],     h_x.size()));
        CUDA_CHECK(cudaMalloc(&d_dy[k],    h_dy.size()));
        CUDA_CHECK(cudaMalloc(&d_gamma[k], h_gamma.size()));
        CUDA_CHECK(cudaMemcpy(d_x[k],     h_x.data(),     h_x.size(),     cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dy[k],    h_dy.data(),    h_dy.size(),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma[k], h_gamma.data(), h_gamma.size(), cudaMemcpyHostToDevice));
    }

    InputPtrs inp{};

    // ----- Ops: SAVE_LN(512) + 4x BWD_LN(512) + FLUSH -----
    const uint32_t DX_ROWS        = ROWS;
    const uint32_t FLUSH_RECORDS  = 1;
    std::vector<OpDesc> ops;
    {
        OpDesc op{}; op.kind = LNR_OP_SAVE_LN;
        op.x_row_base = 0; op.cache_base = 0; op.rows = ROWS;
        ops.push_back(op);
    }
    for (uint32_t k = 0; k < N_BWD_OPS; ++k) {
        OpDesc op{}; op.kind = LNR_OP_BWD_LN;
        op.param_id = k % PARAM_COUNT;
        op.x_row_base = 0; op.dy_row_base = 0; op.cache_base = 0;
        op.rows = ROWS; op.dx_out_base = 0;
        ops.push_back(op);
    }
    {
        OpDesc op{}; op.kind = LNR_OP_FLUSH;
        op.param_id = 0; op.flush_out_base = 0;
        ops.push_back(op);
    }
    const uint32_t OP_COUNT = uint32_t(ops.size());

    RunSpec run{};
    run.run_id             = 1;
    run.op_count           = OP_COUNT;
    run.input_rows         = ROWS;
    run.dx_rows            = DX_ROWS;
    run.flush_records      = FLUSH_RECORDS;
    run.x_stride_elems     = STRIDE;
    run.dy_stride_elems    = STRIDE;
    run.gamma_stride_elems = STRIDE;
    run.reserved0          = 0;
    run.ops                = ops.data();

    // ----- Outputs -----
    // C3: one full output set per timed run (index K reserved for warmup) so the
    // digest can bind EVERY timed call's outputs; a no-op/cached run leaves its
    // region wrong and the digest cannot match the reference bench.
    std::vector<OutputPtrs> outs(K + 1);
    for (int k = 0; k <= K; ++k) {
        OutputPtrs& o = outs[k];
        o = OutputPtrs{};
        CUDA_CHECK(cudaMalloc(&o.dx, size_t(DX_ROWS) * N * 2));
        CUDA_CHECK(cudaMalloc(&o.flush_dgamma, size_t(FLUSH_RECORDS) * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&o.flush_dbeta,  size_t(FLUSH_RECORDS) * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&o.accum_dgamma_snapshot, size_t(PARAM_COUNT) * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&o.accum_dbeta_snapshot,  size_t(PARAM_COUNT) * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&o.cache_mean_snapshot, size_t(ROWS) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&o.cache_rstd_snapshot, size_t(ROWS) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&o.cache_kind_snapshot, size_t(ROWS) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&o.cache_gen_snapshot,  size_t(ROWS) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&o.timeline, size_t(OP_COUNT) * sizeof(TimelineRecord)));
        CUDA_CHECK(cudaMalloc(&o.counters, sizeof(CounterSnapshot)));
    }

    auto do_run = [&](uint64_t rid, int variant) {
        run.run_id = rid;
        inp.x = d_x[variant];
        inp.dy = d_dy[variant];
        inp.gamma = d_gamma[variant];
        cudaError_t e = solution_run(state, &run, &inp, &outs[variant],
                                     workspace, wbytes, stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "solution_run error: %s\n", cudaGetErrorString(e));
            exit(1);
        }
    };

    // ----- Warmup (dedicated variant K; its outputs are never folded) -----
    for (int i = 0; i < 3; ++i) do_run(uint64_t(100 + i), K);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ----- Timed loop: run i computes variant i into output set i -----
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < K; ++i) do_run(uint64_t(1000 + i), i);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, t0, t1));
    printf("avg_ms=%.6f\n", total_ms / float(K));

    // Anti-hack digest: fold every graded output buffer of EVERY timed run (all
    // outputs are byte-exact per contract; TimelineRecord and CounterSnapshot are
    // packed with all fields normatively specified). No untimed run remains, so
    // nothing a solution executes after the timer can touch a digested region.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    pmpp::OutFnv dg;
    for (int k = 0; k < K; ++k) {
        const OutputPtrs& o = outs[k];
        dg.dev(o.dx, size_t(DX_ROWS) * N * 2);
        dg.dev(o.flush_dgamma, size_t(FLUSH_RECORDS) * N * sizeof(float));
        dg.dev(o.flush_dbeta,  size_t(FLUSH_RECORDS) * N * sizeof(float));
        dg.dev(o.accum_dgamma_snapshot, size_t(PARAM_COUNT) * N * sizeof(float));
        dg.dev(o.accum_dbeta_snapshot,  size_t(PARAM_COUNT) * N * sizeof(float));
        dg.dev(o.cache_mean_snapshot, size_t(ROWS) * sizeof(float));
        dg.dev(o.cache_rstd_snapshot, size_t(ROWS) * sizeof(float));
        dg.dev(o.cache_kind_snapshot, size_t(ROWS) * sizeof(uint32_t));
        dg.dev(o.cache_gen_snapshot,  size_t(ROWS) * sizeof(uint64_t));
        dg.dev(o.timeline, size_t(OP_COUNT) * sizeof(TimelineRecord));
        dg.dev(o.counters, sizeof(CounterSnapshot));
    }
    dg.print();

    // ----- Teardown -----
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    for (int k = 0; k <= K; ++k) {
        OutputPtrs& o = outs[k];
        cudaFree(o.dx);
        cudaFree(o.flush_dgamma);
        cudaFree(o.flush_dbeta);
        cudaFree(o.accum_dgamma_snapshot);
        cudaFree(o.accum_dbeta_snapshot);
        cudaFree(o.cache_mean_snapshot);
        cudaFree(o.cache_rstd_snapshot);
        cudaFree(o.cache_kind_snapshot);
        cudaFree(o.cache_gen_snapshot);
        cudaFree(o.timeline);
        cudaFree(o.counters);
        cudaFree(d_x[k]); cudaFree(d_dy[k]); cudaFree(d_gamma[k]);
    }
    if (workspace) cudaFree(workspace);
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
