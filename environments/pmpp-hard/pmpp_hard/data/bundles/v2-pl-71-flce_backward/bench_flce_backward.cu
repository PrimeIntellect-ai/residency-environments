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
#include "pmpp_bench_digest.cuh"

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
    // w/b (fixed weights) derive from PMPP_BENCH_SEED so a memorized answer table is
    // wrong on a fresh seed. The per-row activations x and targets are regenerated as
    // one INDEPENDENT variant per timed call (variant 0 = warmup) from a per-variant
    // seed, keeping shapes and the ~6% ignored-target rate FIXED (timing comparable)
    // while every timed call's graded dx/loss/dW differ — and each call's outputs are
    // folded into out_fnv below. A compute-once-then-replay solution serves stale
    // outputs for calls 2..K → the folded digest diverges → perf FAIL.
    const uint64_t base_seed = pmpp::bench_seed(0xB1A5Eull);
    RNG rng(base_seed);
    std::vector<uint16_t> h_w((uint64_t)V * H);    for (auto& v : h_w) v = rng.small_bf16();
    std::vector<uint16_t> h_b(V);                  for (auto& v : h_b) v = rng.small_bf16();

    auto make_x = [&](uint64_t seed) {
        RNG r(seed);
        std::vector<uint16_t> hx((uint64_t)rows * H);
        for (auto& v : hx) v = r.small_bf16();
        return hx;
    };
    auto make_tgt = [&](uint64_t seed) {
        RNG r(seed ^ 0x7A46E70ull);
        std::vector<int32_t> ht(rows);
        for (auto& t : ht) {
            uint64_t rr = r.next() % 16u;
            t = (rr < 1) ? -100 : (int32_t)(r.next() % V);  // ~6% ignored
        }
        return ht;
    };

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
    CUDA_CHECK(cudaMemcpy(d_w,   h_w.data(),   h_w.size() * 2,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias,h_b.data(),   h_b.size() * 2,  cudaMemcpyHostToDevice));

    // Per-variant x/target device buffers (variant 0 = warmup, 1..K = timed calls).
    uint64_t vseed = base_seed ^ 0x700000000ull;
    auto next_vseed = [&vseed]() -> uint64_t {
        vseed += 0x9e3779b97f4a7c15ull;
        uint64_t z = vseed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    };
    std::vector<uint16_t*> dx_variants((size_t)K + 1, nullptr);
    std::vector<int32_t*>  dtgt_variants((size_t)K + 1, nullptr);
    for (int v = 0; v <= K; ++v) {
        const uint64_t vs = next_vseed();
        std::vector<uint16_t> hx = make_x(vs);
        std::vector<int32_t>  ht = make_tgt(vs);
        CUDA_CHECK(cudaMalloc(&dx_variants[(size_t)v],   (uint64_t)rows * H * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&dtgt_variants[(size_t)v], (uint64_t)rows * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(dx_variants[(size_t)v],   hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dtgt_variants[(size_t)v], ht.data(), ht.size() * 4, cudaMemcpyHostToDevice));
    }

    FLCBInputs in{};
    in.ops = d_ops; in.w_bf16 = d_w; in.bias_bf16 = d_bias;
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
    // identical across iterations (state otherwise persists across runs), and uses its
    // own x/target variant.
    auto one_call = [&](int variant) {
        in.x_bf16 = dx_variants[(size_t)variant];
        in.target = dtgt_variants[(size_t)variant];
        CUDA_CHECK(solution_reset(state, stream));
        cudaError_t r = solution_run(state, &run, &in, &out, d_ws, ws_bytes, stream);
        if (r != cudaSuccess)
            std::fprintf(stderr, "solution_run: %s\n", cudaGetErrorString(r));
    };

    for (int i = 0; i < warmup; ++i) one_call(0);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // Fold each timed call's graded outputs (outside the timed region) so a
    // stale/replayed output on any call breaks the digest.
    pmpp::OutFnv dg;
    float total_ms = 0.0f;

    for (int i = 0; i < K; ++i) {
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        one_call(i + 1);
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        total_ms += ms;

        FLCBRunReport rep{};
        CUDA_CHECK(cudaMemcpy(&rep, d_rep, sizeof(rep), cudaMemcpyDeviceToHost));
        if (rep.output_rows > 0) {
            dg.dev(d_dx,   (uint64_t)rep.output_rows * H * sizeof(uint16_t));
            dg.dev(d_loss, (uint64_t)rep.output_rows * sizeof(float));
        }
        if (rep.flushes_written > 0) {
            dg.dev(d_dW,    (uint64_t)rep.flushes_written * V * H * sizeof(float));
            dg.dev(d_dbias, (uint64_t)rep.flushes_written * V * sizeof(float));
        }
        if (rep.events_written > 0) {
            dg.dev(d_evs, (uint64_t)rep.events_written * sizeof(FLCBEvent));
        }
        dg.dev(d_ctr, sizeof(FLCBCounters));
        dg.dev(d_rep, sizeof(FLCBRunReport));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("avg_ms=%.6f\n", (double)total_ms / (double)K);
    dg.print();

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));

    cudaFree(d_ops); cudaFree(d_w); cudaFree(d_bias);
    for (uint16_t* p : dx_variants) cudaFree(p);
    for (int32_t* p : dtgt_variants) cudaFree(p);
    cudaFree(d_dx); cudaFree(d_loss); cudaFree(d_dW); cudaFree(d_dbias);
    cudaFree(d_evs); cudaFree(d_ctr); cudaFree(d_rep);
    if (d_ws) cudaFree(d_ws);
    return 0;
}
