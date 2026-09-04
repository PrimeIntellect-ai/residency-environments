// file: bench_consistent_hash_ring.cu

#include "consistent_hash_ring_common.h"
#include "pmpp_bench_digest.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                             \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                              \
        }                                                                       \
    } while (0)

struct Op { int32_t op_type; int64_t a, b, c; };

static void op(std::vector<Op>& v, int32_t t, int64_t a = 0, int64_t b = 0, int64_t c = 0) {
    Op o; o.op_type = t; o.a = a; o.b = b; o.c = c; v.push_back(o);
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 10;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        // Single-thread device kernel with O(n^2) ring/key scans: keep sizes modest
        // so the bench terminates promptly. This exercises every op path.
        ChrProblemSpec spec = {};
        spec.abi_version = CHR_ABI_VERSION;
        spec.replication_factor = 3;
        spec.preference_list_extra = 2;
        spec.max_nodes = 64;
        spec.max_vnodes = 512;
        spec.max_keys = 256;
        spec.max_replicas_per_key = 8;
        spec.max_move_tasks = 8192;
        spec.default_node_capacity = 4096;
        // ring_seed/key_seed derive from PMPP_BENCH_SEED (contract: they are plain
        // FNV inputs, any u64 legal) so every graded hash varies per rollout; the
        // op sequence/sizes stay fixed so the timed workload shape is comparable.
        const uint64_t bench_base = pmpp::bench_seed(0x9e3779b97f4a7c15ULL);
        spec.ring_seed = bench_base;
        spec.key_seed = (bench_base * 0xff51afd7ed558ccdULL) ^ 0xc2b2ae3d27d4eb4fULL;

        // One op-stream variant per timed iteration (plus variant 0 for warmup): the op
        // TYPES, COUNTS and ordering are identical across variants (timing comparable),
        // but the per-op arg VALUES (node capacities, key ids, put values) derive from a
        // per-variant seed, so the graded lookup/key/move hashes and counters differ
        // each iteration. Each iteration's outputs are folded into out_fnv, so a
        // compute-once-then-replay solution serves stale outputs for iterations 2..K →
        // the folded digest diverges → perf FAIL. Per-variant seeds come from a
        // SplitMix64 stream over PMPP_BENCH_SEED.
        auto build_ops = [](uint64_t seed) {
            uint64_t z = seed;
            auto next = [&z]() -> uint64_t {
                z += 0x9e3779b97f4a7c15ULL;
                uint64_t x = z;
                x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
                x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
                return x ^ (x >> 31);
            };
            std::vector<Op> ops;
            // Node capacity stays FIXED (16) so rebalance move-counts — and thus the
            // timed workload — are comparable across variants; only key ids/values vary.
            for (int n = 1; n <= 16; ++n) {
                op(ops, CHR_OP_ADD_NODE, n, 16, 0);
                op(ops, CHR_OP_ACTIVATE_NODE, n);
            }
            std::vector<int64_t> keys(201);
            for (int k = 1; k <= 200; ++k) keys[k] = 1 + (int64_t)(next() % 1000000);
            for (int k = 1; k <= 200; ++k) op(ops, CHR_OP_PUT_KEY, keys[k], (int64_t)(next() % 1000000));
            op(ops, CHR_OP_REBALANCE, 100000);
            op(ops, CHR_OP_FAIL_NODE, 7);
            op(ops, CHR_OP_REBALANCE, 100000);
            op(ops, CHR_OP_RECOVER_NODE, 7);
            op(ops, CHR_OP_REBALANCE, 100000);
            for (int k = 1; k <= 200; ++k) op(ops, CHR_OP_LOOKUP, 100000 + k, keys[k]);
            return ops;
        };

        uint64_t vseed = bench_base ^ 0x700000000ULL;
        auto next_vseed = [&vseed]() -> uint64_t {
            vseed += 0x9e3779b97f4a7c15ULL;
            uint64_t x = vseed;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            return x ^ (x >> 31);
        };
        std::vector<std::vector<Op>> op_variants((size_t)iters + 1);
        for (int v = 0; v <= iters; ++v) op_variants[(size_t)v] = build_ops(next_vseed());

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t ws = solution_workspace_bytes(&spec);
        uint8_t* d_ws = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_ws, std::max<size_t>(ws, 1)));

        // single output set reused
        ChrCounters* d_ctr; uint64_t *d_e, *d_l, *d_r, *d_n, *d_k, *d_m;
        CUDA_CHECK(cudaMalloc((void**)&d_ctr, sizeof(ChrCounters)));
        CUDA_CHECK(cudaMalloc((void**)&d_e, 8)); CUDA_CHECK(cudaMalloc((void**)&d_l, 8));
        CUDA_CHECK(cudaMalloc((void**)&d_r, 8)); CUDA_CHECK(cudaMalloc((void**)&d_n, 8));
        CUDA_CHECK(cudaMalloc((void**)&d_k, 8)); CUDA_CHECK(cudaMalloc((void**)&d_m, 8));
        ChrOutputs out = {};
        out.counters = d_ctr; out.ring_event_hash = d_e; out.lookup_hash = d_l;
        out.ring_hash = d_r; out.node_hash = d_n; out.key_replica_hash = d_k; out.move_hash = d_m;

        std::printf("bench_chr ops=%zu nodes=16 keys=200\n", op_variants[0].size());

        auto run_all = [&](const std::vector<Op>& ops) {
            CUDA_CHECK(solution_reset(state, stream));
            for (size_t i = 0; i < ops.size(); ++i) {
                ChrRunSpec run = {};
                run.abi_version = CHR_ABI_VERSION;
                run.op_type = ops[i].op_type; run.op_index = (int32_t)i;
                run.arg_a = ops[i].a; run.arg_b = ops[i].b; run.arg_c = ops[i].c;
                CUDA_CHECK(solution_run(state, &run, nullptr, &out, d_ws, ws, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        };

        for (int w = 0; w < 2; ++w) run_all(op_variants[0]);

        cudaEvent_t s0, s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
        double total_ms = 0;

        // Fold each timed iteration's graded outputs (outside the timed region) so a
        // stale/replayed output on any iteration breaks the digest.
        pmpp::OutFnv dg;

        for (int it = 0; it < iters; ++it) {
            const std::vector<Op>& ops = op_variants[(size_t)it + 1];
            CUDA_CHECK(cudaEventRecord(s0, stream));
            run_all(ops);
            CUDA_CHECK(cudaEventRecord(s1, stream));
            CUDA_CHECK(cudaEventSynchronize(s1));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1));
            total_ms += ms;

            dg.dev(d_ctr, sizeof(ChrCounters));
            dg.dev(d_e, sizeof(uint64_t));
            dg.dev(d_l, sizeof(uint64_t));
            dg.dev(d_r, sizeof(uint64_t));
            dg.dev(d_n, sizeof(uint64_t));
            dg.dev(d_k, sizeof(uint64_t));
            dg.dev(d_m, sizeof(uint64_t));
        }
        std::printf("avg_ms=%.6f\n", total_ms / iters);
        dg.print();

        cudaEventDestroy(s0); cudaEventDestroy(s1);
        cudaFree(d_ctr); cudaFree(d_e); cudaFree(d_l); cudaFree(d_r); cudaFree(d_n); cudaFree(d_k); cudaFree(d_m);
        cudaFree(d_ws);
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
