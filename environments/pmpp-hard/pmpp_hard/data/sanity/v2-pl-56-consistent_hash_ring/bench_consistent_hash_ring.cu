// file: bench_consistent_hash_ring.cu

#include "consistent_hash_ring_common.h"

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
        spec.ring_seed = 0x8e3779b97f4a7c15ULL;
        spec.key_seed = 0xc2b2ae3d27d4eb4fULL;

        std::vector<Op> ops;
        for (int n = 1; n <= 16; ++n) { op(ops, CHR_OP_ADD_NODE, n, 16, 0); op(ops, CHR_OP_ACTIVATE_NODE, n); }
        for (int k = 1; k <= 200; ++k) op(ops, CHR_OP_PUT_KEY, k, k);
        op(ops, CHR_OP_REBALANCE, 100000);
        op(ops, CHR_OP_FAIL_NODE, 7);
        op(ops, CHR_OP_REBALANCE, 100000);
        op(ops, CHR_OP_RECOVER_NODE, 7);
        op(ops, CHR_OP_REBALANCE, 100000);
        for (int k = 1; k <= 200; ++k) op(ops, CHR_OP_LOOKUP, 100000 + k, k);

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

        std::printf("bench_chr ops=%zu nodes=16 keys=200\n", ops.size());

        auto run_all = [&]() {
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

        for (int w = 0; w < 2; ++w) run_all();

        cudaEvent_t s0, s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
        double total_ms = 0;
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(cudaEventRecord(s0, stream));
            run_all();
            CUDA_CHECK(cudaEventRecord(s1, stream));
            CUDA_CHECK(cudaEventSynchronize(s1));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, s0, s1));
            total_ms += ms;
        }
        std::printf("avg_ms=%.6f\n", total_ms / iters);

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
