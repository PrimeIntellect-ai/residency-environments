// ============================================================================
// file: bench_flce_backward.cu
// ============================================================================
//
// Perf gate for FLCB-StateTrain v1 (fused linear cross-entropy backward).
//
// Drives the solution_* ABI on a single LARGE shape where the matmul-class work
// dominates: per micro row the solver materializes V logits (each an H-dim dot
// product), an exact online-softmax pass, dX (V x H) and a persistent dW (V x H)
// accumulation. With V=32768 and H=1024 over 128 rows this is ~4.3e9
// multiply-adds for the logits plus ~4.3e9 for dW per call — a few ms for a real
// parallel GPU kernel, but tens of seconds for a serial CPU host-replay.
//
// This bench does NOT check correctness (test_flce_backward.cu does that); it
// only times K solution_run calls and prints `avg_ms=<total_ms / K>`.
//
// Usage: bench [K] [warmup]
//   K       number of timed solution_run calls   (default 20)
//   warmup  number of untimed warmup calls        (default 3)
// A small K is intended for the slow serial baselines (CPU host-replay / a
// single-thread device kernel) so a measurement completes in tolerable wall time.

#include "flce_backward_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#define CUDA_CHECK(expr) do {                                                    \
    cudaError_t _e = (expr);                                                      \
    if (_e != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__,       \
                     cudaGetErrorString(_e));                                     \
        std::exit(1);                                                             \
    }                                                                            \
} while (0)

// xorshift64 PRNG — same bf16 magnitude band as the correctness harness.
struct RNG {
    uint64_t s;
    explicit RNG(uint64_t seed) : s(seed | 1ull) {}
    uint64_t next() { s^=s<<13; s^=s>>7; s^=s<<17; return s; }
    uint16_t small_bf16() {
        uint32_t exp  = 0x3Cu + (uint32_t)(next() % 7u);
        uint32_t mant = (uint32_t)(next() & 0x7Fu);
        uint32_t sign = (uint32_t)((next() >> 32) & 1u);
        return (uint16_t)((sign << 15) | (exp << 7) | mant);
    }
};

static uint32_t float_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
    int K      = (argc >= 2) ? std::max(1, std::atoi(argv[1])) : 20;
    int warmup = (argc >= 3) ? std::max(0, std::atoi(argv[2])) : 3;

    CUDA_CHECK(cudaSetDevice(0));

    // ── LARGE bench shape (matmul-class work dominates) ──────────────────────
    FLCBSpec spec{};
    spec.magic = FLCB_MAGIC; spec.version = FLCB_VERSION;
    spec.H = 1024; spec.V = 32768;
    spec.max_run_rows = 128; spec.max_ops = 8;
    spec.row_chunk = 64; spec.vocab_tile = 512;
    spec.ignore_index = -100; spec.has_bias = 1; spec.max_flushes_per_run = 4;

    const uint32_t H = spec.H, V = spec.V;
    const uint32_t rows = spec.max_run_rows;       // 128

    std::printf("bench_shape H=%u V=%u rows=%u row_chunk=%u vocab_tile=%u has_bias=%u K=%d warmup=%d\n",
                H, V, rows, spec.row_chunk, spec.vocab_tile, spec.has_bias, K, warmup);

    // ── Deterministic inputs ────────────────────────────────────────────────
    RNG rng(0xB1A5Eull + 0x8ce7bull);
    std::vector<uint16_t> h_x((uint64_t)rows * H); for (auto& v : h_x) v = rng.small_bf16();
    std::vector<uint16_t> h_w((uint64_t)V * H);    for (auto& v : h_w) v = rng.small_bf16();
    std::vector<uint16_t> h_b(V);                  for (auto& v : h_b) v = rng.small_bf16();
    std::vector<int32_t>  h_tgt(rows);
    for (auto& t : h_tgt) {
        uint64_t r = rng.next() % 16u;
        t = (r < 1) ? -100 : (int32_t)(rng.next() % V);  // ~6% ignored
    }

    // One micro op over all rows + one snapshot flush (materializes dW [V,H]).
    std::vector<FLCBOp> ops(2);
    ops[0] = {FLCB_OP_MICRO, 0, rows, FLCB_RED_SUM, 0, float_bits(1.0f), 0, 0, 0ull, 0x1ull};
    ops[1] = {FLCB_OP_FLUSH, 0, 0, 0, 0, 0, FLCB_FLUSH_SNAPSHOT, 0, 0ull, 0x2ull};

    FLCBRunSpec run{};
    run.run_id = 0xBEEFull;
    run.input_rows = rows;
    run.output_row_capacity = rows;
    run.op_count = (uint32_t)ops.size();
    run.event_capacity = 64u;
    run.flush_capacity = 1u;

    // ── Device buffers ──────────────────────────────────────────────────────
    FLCBOp*        d_ops   = nullptr;
    uint16_t*      d_x     = nullptr;
    int32_t*       d_tgt   = nullptr;
    uint16_t*      d_w     = nullptr;
    uint16_t*      d_bias  = nullptr;
    uint16_t*      d_dx    = nullptr;
    float*         d_loss  = nullptr;
    float*         d_dW    = nullptr;
    float*         d_dbias = nullptr;
    FLCBEvent*     d_evs   = nullptr;
    FLCBCounters*  d_ctr   = nullptr;
    FLCBRunReport* d_rep   = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ops,   ops.size() * sizeof(FLCBOp)));
    CUDA_CHECK(cudaMalloc(&d_x,     (uint64_t)rows * H * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_tgt,   (uint64_t)rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_w,     (uint64_t)V * H * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_bias,  (uint64_t)V * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_dx,    (uint64_t)rows * H * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_loss,  (uint64_t)rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW,    (uint64_t)run.flush_capacity * V * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dbias, (uint64_t)run.flush_capacity * V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_evs,   (uint64_t)run.event_capacity * sizeof(FLCBEvent)));
    CUDA_CHECK(cudaMalloc(&d_ctr,   sizeof(FLCBCounters)));
    CUDA_CHECK(cudaMalloc(&d_rep,   sizeof(FLCBRunReport)));

    CUDA_CHECK(cudaMemcpy(d_ops, ops.data(),   ops.size() * sizeof(FLCBOp), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,   h_x.data(),   h_x.size() * 2,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tgt, h_tgt.data(), h_tgt.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,   h_w.data(),   h_w.size() * 2,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias,h_b.data(),   h_b.size() * 2,  cudaMemcpyHostToDevice));

    FLCBInputs in{};
    in.ops = d_ops; in.x_bf16 = d_x; in.target = d_tgt; in.w_bf16 = d_w; in.bias_bf16 = d_bias;
    FLCBOutputs out{};
    out.dx_bf16 = d_dx; out.loss_f32 = d_loss;
    out.flush_dW_f32 = d_dW; out.flush_dbias_f32 = d_dbias;
    out.events = d_evs; out.counters = d_ctr; out.report = d_rep;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    size_t ws_bytes = solution_workspace_bytes(&spec);
    void* d_ws = nullptr;
    if (ws_bytes > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));

    // Each timed call starts from a clean accumulator so the flush snapshot is
    // identical across iterations (state otherwise persists across runs).
    auto one_call = [&]() {
        CUDA_CHECK(solution_reset(state, stream));
        cudaError_t r = solution_run(state, &run, &in, &out, d_ws, ws_bytes, stream);
        if (r != cudaSuccess)
            std::fprintf(stderr, "solution_run: %s\n", cudaGetErrorString(r));
    };

    for (int i = 0; i < warmup; ++i) one_call();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0, stream));
    for (int i = 0; i < K; ++i) one_call();
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev0, ev1));
    std::printf("avg_ms=%.6f\n", (double)total_ms / (double)K);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));

    cudaFree(d_ops); cudaFree(d_x); cudaFree(d_tgt); cudaFree(d_w); cudaFree(d_bias);
    cudaFree(d_dx); cudaFree(d_loss); cudaFree(d_dW); cudaFree(d_dbias);
    cudaFree(d_evs); cudaFree(d_ctr); cudaFree(d_rep);
    if (d_ws) cudaFree(d_ws);
    return 0;
}
