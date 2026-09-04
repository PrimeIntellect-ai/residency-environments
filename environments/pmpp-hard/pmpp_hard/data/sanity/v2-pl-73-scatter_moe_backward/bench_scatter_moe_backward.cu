// bench_scatter_moe_backward.cu -- PERF GATE for bwd_scatter (DSG-MoE-BWD)
//
// Drives the solution_* ABI on a LARGE shape from the design bench family.
// Generates inputs once (deterministic seed), solution_init, ~3 warmup calls,
// then times K solution_run calls with CUDA events. Prints exactly:
//   avg_ms=%.6f
// Does NOT check correctness (that is test_scatter_moe_backward.cu's job).
//
// Build: nvcc -O3 -std=c++17 -arch=sm_120 bench_scatter_moe_backward.cu <impl>.cu -o bench
//
// Large-shape rationale: high token count with high index/expert collision so the
// sorted-segment duplicate reduction dominates. A serial CPU host-replay solution
// must D2H the full input, std::sort millions of contributions, and reduce them
// serially every run -- orders of magnitude over budget vs the GPU reference.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "scatter_moe_backward_common.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// ---- Sanity seed salt: shifts generated INPUTS so bench data matches the
//      seed-shifted sanity grader (timing shape unchanged). ----
static const uint64_t SANITY_SALT = 0xa5a5a5a5a5a5a5a5ull;

// ---- Splitmix64 deterministic RNG (same family as the test harness) ----
struct SM64 {
    uint64_t s;
    SM64(uint64_t seed) : s(seed ^ SANITY_SALT) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t next32() { return uint32_t(next() >> 32); }
    // Small-magnitude finite float bits (avoids NaN/Inf exponent).
    uint32_t small_f32_bits() {
        uint32_t r = next32();
        uint32_t exp = 1 + (r >> 25) % 126;  // 1..126
        return (r & 0x807FFFFFu) | (exp << 23);
    }
};

static size_t align8(size_t x) { return (x + 7u) & ~size_t(7u); }

struct InputLayout {
    size_t off_index, off_expert, off_gate, off_x, off_dy, off_up, total;
};

static InputLayout input_layout(const Spec& s, uint32_t T) {
    InputLayout L;
    size_t dsz = (s.input_dtype == DSG_DTYPE_BF16) ? 2u : 4u;
    L.off_index  = 0;
    L.off_expert = align8(L.off_index  + size_t(T) * 4u);
    L.off_gate   = align8(L.off_expert + size_t(T) * s.top_k * 4u);
    L.off_x      = align8(L.off_gate   + size_t(T) * s.top_k * 2u);
    L.off_dy     = align8(L.off_x      + size_t(T) * s.dim * dsz);
    L.off_up     = align8(L.off_dy     + size_t(T) * s.out_dim * dsz);
    L.total      = align8(L.off_up     + size_t(T) * s.dim * dsz);
    return L;
}

static size_t output_used_bytes(const Spec& s, uint32_t T, uint32_t flags) {
    size_t off = sizeof(OutHeader);
    off = align8(off + size_t(T) * s.dim * 4u);
    if (flags & DSG_FLAG_EMIT_TABLE)  off = align8(off + size_t(s.table_rows) * s.dim * 4u);
    if (flags & DSG_FLAG_EMIT_EXPERT) off = align8(off + size_t(s.experts) * s.dim * s.out_dim * 4u);
    return off;
}

int main() {
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) { fprintf(stderr, "No CUDA devices found\n"); return 1; }
    CUDA_CHECK(cudaSetDevice(0));

    // ---- LARGE bench shape (near the agent-visible caps) ----
    //   max_tokens*top_k*dim*out_dim = 16384*2*8*8 = 2,097,152 <= 4,194,304
    //   table_rows*dim              = 512*8         = 4,096     <= 524,288
    //   experts*dim*out_dim         = 64*8*8        = 4,096     <= 524,288
    //   capacity                    = 32768         <= max_tokens*top_k = 32768
    const uint32_t R = 512;
    const uint32_t D = 8;
    const uint32_t O = 8;
    const uint32_t E = 64;
    const uint32_t K = 2;
    const uint32_t T = 16384;          // tokens this run (== max_tokens)
    const uint32_t CAP = T * K;        // 32768: capacity high => all assignments accepted

    Spec spec = {};
    spec.magic      = DSG_SPEC_MAGIC;
    spec.version    = DSG_VERSION;
    spec.table_rows = R;
    spec.dim        = D;
    spec.out_dim    = O;
    spec.experts    = E;
    spec.top_k      = K;
    spec.max_tokens = T;
    spec.capacity   = CAP;
    spec.input_dtype = DSG_DTYPE_F32;
    spec.padding_idx = -1;
    spec.flags      = 0;
    spec.weight_seed = 0x9E3779B97F4A7C15ull ^ SANITY_SALT;  // sanity salt: shifts weights
    spec.reserved0  = 0;
    spec.reserved1  = 0;

    RunSpec run = {};
    run.magic             = DSG_RUN_MAGIC;
    run.version           = DSG_VERSION;
    run.tokens            = T;
    run.flags             = DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT;
    run.capacity_override = DSG_CAPACITY_USE_SPEC;
    run.reserved0         = 0;
    run.user_tag          = 0xBEEF;

    // ---- Build inputs once (deterministic), with HIGH index/expert collision ----
    InputLayout L = input_layout(spec, T);
    std::vector<uint8_t> hbuf(L.total, 0u);

    int32_t* emb_idx  = reinterpret_cast<int32_t*>(hbuf.data() + L.off_index);
    int32_t* route_e  = reinterpret_cast<int32_t*>(hbuf.data() + L.off_expert);
    int16_t* gate_q15 = reinterpret_cast<int16_t*>(hbuf.data() + L.off_gate);
    uint32_t* xv      = reinterpret_cast<uint32_t*>(hbuf.data() + L.off_x);
    uint32_t* dyv     = reinterpret_cast<uint32_t*>(hbuf.data() + L.off_dy);
    uint32_t* upv     = reinterpret_cast<uint32_t*>(hbuf.data() + L.off_up);

    SM64 rng(0xD5C12345ull);
    // High collision: only 16 distinct embedding rows across all T tokens, so the
    // table segment reduction is dominated by a handful of very long segments.
    for (uint32_t t = 0; t < T; ++t) {
        emb_idx[t] = int32_t(t % 16u);
        for (uint32_t k = 0; k < K; ++k) {
            route_e[t * K + k]  = int32_t((t + k) % E);   // valid experts, all used
            gate_q15[t * K + k] = int16_t(8192 + ((t + k) & 0x3FFF)); // varied Q15 gates
        }
    }
    for (uint32_t i = 0; i < T * D; ++i) xv[i]  = rng.small_f32_bits();
    for (uint32_t i = 0; i < T * O; ++i) dyv[i] = rng.small_f32_bits();
    for (uint32_t i = 0; i < T * D; ++i) upv[i] = rng.small_f32_bits();

    void* d_inp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_inp, L.total ? L.total : 1));
    CUDA_CHECK(cudaMemcpy(d_inp, hbuf.data(), L.total, cudaMemcpyHostToDevice));

    size_t out_bytes = output_used_bytes(spec, T, run.flags);
    void* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, out_bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    cudaError_t err = solution_init(&spec, &state, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "solution_init failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ---- Warmup ----
    const int WARMUP = 3;
    for (int i = 0; i < WARMUP; ++i) {
        err = solution_run(state, &run, d_inp, d_out, nullptr, 0, stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "solution_run (warmup) failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ---- Timed loop ----
    const int ITERS = 20;
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0, stream));
    for (int i = 0; i < ITERS; ++i) {
        err = solution_run(state, &run, d_inp, d_out, nullptr, 0, stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "solution_run (timed) failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev0, ev1));

    printf("avg_ms=%.6f\n", total_ms / float(ITERS));

    // ---- Teardown ----
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
