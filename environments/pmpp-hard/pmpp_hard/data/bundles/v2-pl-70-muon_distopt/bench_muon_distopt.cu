// bench_muon_distopt.cu - PERF GATE for muon_distopt_stateful_v2.
//
// Drives the solution_* ABI on a LARGE bench shape (many params, several
// square Muon params with large rows*cols + several wide Adam params, R ranks,
// hundreds of valid ops per run). The five-step Newton-Schulz orthogonalization
// on the Muon params dominates the per-run cost, so a serial CPU host-replay
// solution is far more expensive per run than a real device implementation.
//
// This file does NOT check correctness (that is test_muon_distopt.cu); it only
// times solution_run. It generates inputs once with a deterministic seed,
// solution_init, warms up, then times K solution_run calls with CUDA events and
// prints exactly:  avg_ms=%.6f
//
// Usage: ./bench_* [K]      (default K=20)

#include "muon_distopt_common.h"
#include "pmpp_bench_digest.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

// ---- Solution ABI (provided by solution.cu / reference_solution.cu) ----
extern "C" {
    size_t solution_workspace_bytes(const Spec* spec);
    cudaError_t solution_init(const Spec* spec, void** state, cudaStream_t stream);
    cudaError_t solution_run(void* state, const RunSpec* run, const void* inputs,
                             void* outputs, void* workspace, size_t workspace_bytes,
                             cudaStream_t stream);
    cudaError_t solution_reset(void* state, cudaStream_t stream);
    void solution_destroy(void* state);
}

#define CUDA_CHK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// ---- LARGE bench shape (NOT the tiny correctness families) ----
// Several large square Muon params (five-step Newton-Schulz cost ~ DIM^3 per
// param per run on a serial solver) + a couple of wide Adam vectors. The NS
// matmuls dominate the per-run cost, so a serial CPU host-replay is orders of
// magnitude slower than a real parallel device implementation.
static const uint32_t BENCH_NUM_MUON = 4;
static const uint32_t BENCH_MUON_DIM = 512;    // square 512x512 Muon params
static const uint32_t BENCH_NUM_ADAM = 2;
static const uint32_t BENCH_ADAM_LEN = 8192;   // 1x8192 Adam params
static const uint32_t BENCH_RANKS    = 4;

// ---- Host utilities (mirror the test harness generators) ----
static uint32_t shard_start_h(uint32_t numel, uint32_t rank, uint32_t R) {
    return (uint32_t)(((uint64_t)numel * (uint64_t)rank) / (uint64_t)R);
}
static uint32_t shard_end_h(uint32_t numel, uint32_t rank, uint32_t R) {
    return (uint32_t)(((uint64_t)numel * (uint64_t)(rank + 1u)) / (uint64_t)R);
}
static uint64_t spec_total_numel(const Spec* spec) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < spec->num_params; ++i)
        total += (uint64_t)spec->params[i].rows * spec->params[i].cols;
    return total;
}
static uint64_t splitmix64_h(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static uint16_t gen_grad_bf16(uint64_t seed) {
    uint64_t x = splitmix64_h(seed);
    uint16_t sign = (uint16_t)((x >> 63) & 1ull);
    uint16_t exp  = (uint16_t)(120u + (uint32_t)((x >> 56) & 0xFull));
    uint16_t mant = (uint16_t)((x >> 16) & 0x7Full);
    return (uint16_t)((sign << 15u) | (exp << 7u) | mant);
}

static void build_bench_spec(Spec* spec) {
    memset(spec, 0, sizeof(*spec));
    spec->magic           = MD_MAGIC;
    spec->version         = MD_VERSION;
    spec->num_ranks       = BENCH_RANKS;
    spec->global_seed     = 0x9E3779B97F4A7C15ull;

    uint32_t p = 0;
    for (uint32_t i = 0; i < BENCH_NUM_MUON; ++i, ++p) {
        spec->params[p].rows = BENCH_MUON_DIM;
        spec->params[p].cols = BENCH_MUON_DIM;
        spec->params[p].storage_dtype = MD_DTYPE_BF16;
        spec->params[p].init_seed = 0x1000000000000000ull + (uint64_t)p;
    }
    for (uint32_t i = 0; i < BENCH_NUM_ADAM; ++i, ++p) {
        spec->params[p].rows = 1;
        spec->params[p].cols = BENCH_ADAM_LEN;
        spec->params[p].storage_dtype = MD_DTYPE_BF16;
        spec->params[p].init_seed = 0x2000000000000000ull + (uint64_t)p;
    }
    spec->num_params = p;

    // Complete ops = one per (param, rank) with a nonzero shard.
    spec->max_ops_per_run = spec->num_params * spec->num_ranks + 16u;
}

// Generate complete-shard ops + grads for all params/ranks (deterministic).
// grad_seed_base folds PMPP_BENCH_SEED so grad VALUES vary per rollout while the
// op schedule / shard structure stays fixed; gen_grad_bf16 only emits finite
// normal bf16 (exp 120..135), so every seed stays in the legal input domain.
static void build_complete_ops(const Spec* spec,
                               uint64_t grad_seed_base,
                               std::vector<GradOp>& ops,
                               std::vector<uint16_t>& grads) {
    ops.clear(); grads.clear();
    uint32_t grad_offset = 0;
    for (uint32_t pid = 0; pid < spec->num_params; ++pid) {
        uint32_t numel = spec->params[pid].rows * spec->params[pid].cols;
        for (uint32_t r = 0; r < spec->num_ranks; ++r) {
            uint32_t ss = shard_start_h(numel, r, spec->num_ranks);
            uint32_t se = shard_end_h(numel, r, spec->num_ranks);
            uint32_t slen = se - ss;
            if (slen == 0) continue;
            GradOp op = {};
            op.param_id     = pid;
            op.src_rank     = r;
            op.dst_rank     = r;
            op.owner_offset = 0;
            op.elem_count   = slen;
            op.grad_offset  = grad_offset;
            ops.push_back(op);
            for (uint32_t k = 0; k < slen; ++k) {
                uint64_t seed = grad_seed_base
                              ^ ((uint64_t)pid << 40) ^ ((uint64_t)r << 32) ^ (uint64_t)k;
                grads.push_back(gen_grad_bf16(seed));
            }
            grad_offset += slen;
        }
    }
}

int main(int argc, char** argv) {
    int K = 20;
    if (argc >= 2) { K = atoi(argv[1]); if (K < 1) K = 1; }

    CUDA_CHK(cudaSetDevice(0));

    Spec spec; build_bench_spec(&spec);

    uint64_t total_numel = spec_total_numel(&spec);

    // C3: every timed call k gets its OWN grad batch, seeded from
    // (PMPP_BENCH_SEED, k). The op schedule / shard structure is identical for
    // every call (timing comparability); only the grad VALUES differ, so a
    // no-op/cached call produces stale weights for a grad batch it never applied
    // and the per-call weight-payload fold below cannot match the reference.
    // Variant K is reserved for warmup so warmup never feeds a folded call.
    const uint64_t seed_root = pmpp::bench_seed(0x70B3ull);
    std::vector<std::vector<GradOp>> ops_v(K + 1);
    std::vector<std::vector<uint16_t>> grads_v(K + 1);
    for (int k = 0; k <= K; ++k) {
        const uint64_t grad_seed_base =
            splitmix64_h(seed_root ^ (0x9E3779B97F4A7C15ull * (uint64_t)(k + 1)));
        build_complete_ops(&spec, grad_seed_base, ops_v[k], grads_v[k]);
    }

    printf("bench shape: num_params=%u (muon=%u %ux%u, adam=%u 1x%u) ranks=%u "
           "total_numel=%llu ops/run=%zu grads/run=%zu K=%d variants=%d\n",
           spec.num_params, BENCH_NUM_MUON, BENCH_MUON_DIM, BENCH_MUON_DIM,
           BENCH_NUM_ADAM, BENCH_ADAM_LEN, spec.num_ranks,
           (unsigned long long)total_numel, ops_v[0].size(), grads_v[0].size(),
           K, K);

    cudaStream_t stream; CUDA_CHK(cudaStreamCreate(&stream));

    // Allocate device buffers.
    size_t ws_bytes = solution_workspace_bytes(&spec);
    if (ws_bytes == 0) ws_bytes = 64; // stub returns 0; cudaMalloc needs >0
    size_t out_bytes = sizeof(RunOutputHeader) + sizeof(uint16_t) * (size_t)total_numel;
    size_t ops_bytes = sizeof(GradOp) * ops_v[0].size();
    size_t grads_bytes = sizeof(uint16_t) * grads_v[0].size();
    size_t in_bytes = ops_bytes + grads_bytes;

    // Unified (managed) memory so a single bench harness drives every ABI flavor:
    //  - a real device solution reads/writes these pointers from a kernel,
    //  - a host-replay solution cudaMemcpy's them D2H/H2D,
    //  - the host scalar baseline dereferences them directly.
    // A conformant device solution is unaffected; this does not weaken the gate.
    // One input blob per grad variant, and one OUTPUT snapshot per timed call so
    // the digest can bind every timed call's weight payload (a shared out buffer
    // would be overwritten, folding only the last call).
    void* d_ws = nullptr;
    CUDA_CHK(cudaMallocManaged(&d_ws, ws_bytes));
    std::vector<void*> d_in(K + 1, nullptr);
    for (int k = 0; k <= K; ++k) {
        CUDA_CHK(cudaMallocManaged(&d_in[k], in_bytes));
        memcpy((uint8_t*)d_in[k], ops_v[k].data(), ops_bytes);
        memcpy((uint8_t*)d_in[k] + ops_bytes, grads_v[k].data(), grads_bytes);
    }
    std::vector<void*> d_out(K, nullptr);
    for (int i = 0; i < K; ++i) {
        CUDA_CHK(cudaMallocManaged(&d_out[i], out_bytes));
    }
    void* d_out_warm = nullptr;
    CUDA_CHK(cudaMallocManaged(&d_out_warm, out_bytes));

    RunSpec rs; memset(&rs, 0, sizeof(rs));
    rs.run_tag          = 0xBE0011ull;
    rs.op_count         = (uint32_t)ops_v[0].size();
    rs.grad_value_count = (uint32_t)grads_v[0].size();

    void* state = nullptr;
    CUDA_CHK(solution_init(&spec, &state, stream));
    CUDA_CHK(cudaStreamSynchronize(stream));

    // Warmup (~3 calls) on the dedicated warmup variant + output buffer.
    for (int w = 0; w < 3; ++w) {
        CUDA_CHK(solution_run(state, &rs, d_in[K], d_out_warm, d_ws, ws_bytes, stream));
    }
    CUDA_CHK(cudaStreamSynchronize(stream));

    // Timed loop: K solution_run calls, each on its own grad variant + output.
    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));
    CUDA_CHK(cudaEventRecord(start, stream));
    for (int i = 0; i < K; ++i) {
        CUDA_CHK(solution_run(state, &rs, d_in[i], d_out[i], d_ws, ws_bytes, stream));
    }
    CUDA_CHK(cudaEventRecord(stop, stream));
    CUDA_CHK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHK(cudaEventElapsedTime(&total_ms, start, stop));
    printf("avg_ms=%.6f\n", (double)total_ms / (double)K);

    // Anti-hack digest: fold the graded RunOutputHeader fields AND the full
    // weights_bf16 output payload of EVERY timed call. The header hashes alone are
    // solution-computed (self-reported); folding each call's payload — layout pinned
    // by the contract (param-id ascending, row-major, bf16-exact per weight_hash),
    // and now on a per-call PMPP_BENCH_SEED-derived grad batch — forces the solution
    // to materialize the real seeded weights on every timed call, not replay a cached
    // snapshot. No untimed run remains, so nothing after the timer touches a digest.
    CUDA_CHK(cudaStreamSynchronize(stream));
    CUDA_CHK(cudaDeviceSynchronize());

    pmpp::OutFnv dg;
    for (int i = 0; i < K; ++i) {
        RunOutputHeader dh;
        memcpy(&dh, d_out[i], sizeof(dh));
        dg.bytes(&dh.status, sizeof(dh.status));
        dg.bytes(&dh.attempted_runs, sizeof(dh.attempted_runs));
        dg.bytes(&dh.successful_runs, sizeof(dh.successful_runs));
        dg.bytes(&dh.reset_count, sizeof(dh.reset_count));
        dg.bytes(&dh.weight_hash, sizeof(dh.weight_hash));
        dg.bytes(&dh.state_hash, sizeof(dh.state_hash));
        dg.bytes(&dh.timeline_hash, sizeof(dh.timeline_hash));
        dg.bytes(&dh.counter_hash, sizeof(dh.counter_hash));
        dg.bytes(&dh.final_hash, sizeof(dh.final_hash));
        dg.bytes(&dh.last_valid_ops, sizeof(dh.last_valid_ops));
        dg.bytes(&dh.last_invalid_ops, sizeof(dh.last_invalid_ops));
        dg.bytes((const uint8_t*)d_out[i] + sizeof(RunOutputHeader),
                 sizeof(uint16_t) * (size_t)total_numel);
    }
    dg.print();

    // Teardown.
    CUDA_CHK(cudaEventDestroy(start));
    CUDA_CHK(cudaEventDestroy(stop));
    solution_destroy(state);
    cudaFree(d_ws);
    for (int k = 0; k <= K; ++k) cudaFree(d_in[k]);
    for (int i = 0; i < K; ++i) cudaFree(d_out[i]);
    cudaFree(d_out_warm);
    CUDA_CHK(cudaStreamDestroy(stream));
    CUDA_CHK(cudaDeviceReset());
    return 0;
}
