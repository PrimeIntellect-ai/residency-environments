// file: bench_beps_tree_buffer.cu
//
// Minimal timing harness for the beps_tree_buffer kernel ABI. Drives a fixed
// pseudo-random op-batch workload through solution_init / solution_run and
// reports wall time.

#include "beps_tree_buffer_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                   \
            throw std::runtime_error(_oss.str());                               \
        }                                                                       \
    } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t s) : state(s) {}
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        BepsProblemSpec spec = {};
        spec.abi_version = BEPS_ABI_VERSION;
        spec.max_nodes = 16384;
        spec.internal_buffer_cap = 8;
        spec.leaf_record_cap = 8;
        spec.max_children_per_internal = 8;
        spec.flush_message_cap = 8;
        spec.max_range_results = 256;
        spec.max_ops = 128;
        spec.max_steps = 64;
        if (!beps_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));

        size_t wb = solution_workspace_bytes(&spec);
        uint8_t* workspace = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace), std::max<size_t>(wb, 1)));

        BepsOp* d_ops = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ops), sizeof(BepsOp) * spec.max_ops));

        BepsCounts* d_counts = nullptr;
        uint64_t* d_hashes = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_counts), sizeof(BepsCounts)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hashes), sizeof(uint64_t) * 5));

        BepsOutputs out = {};
        out.counts = d_counts;
        out.message_event_hash = d_hashes + 0;
        out.query_hash = d_hashes + 1;
        out.tree_shape_hash = d_hashes + 2;
        out.buffer_hash = d_hashes + 3;
        out.leaf_hash = d_hashes + 4;

        SplitMix64 rng(0xBEE5C0FFEEULL);
        const int steps = 64;
        const uint64_t kKeyMod = 200;

        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0, stream));

        for (int s = 0; s < steps; ++s) {
            std::vector<BepsOp> ops;
            int n = 16 + (int)(rng.next() % 16);
            for (int k = 0; k < n; ++k) {
                BepsOp o = {};
                int sel = (int)(rng.next() % 6);
                uint64_t key = rng.next() % kKeyMod;
                switch (sel) {
                    case 0: o.kind = BEPS_OP_PUT; o.u_key = key; o.value = (int64_t)rng.next(); break;
                    case 1: o.kind = BEPS_OP_ADD; o.u_key = key; o.value = (int64_t)(rng.next() % 100); break;
                    case 2: o.kind = BEPS_OP_DELETE; o.u_key = key; break;
                    case 3: o.kind = BEPS_OP_POINT_QUERY; o.u_aux = rng.next(); o.u_key = key; break;
                    case 4: { uint64_t lo = rng.next() % kKeyMod, hi = rng.next() % kKeyMod;
                              if (lo > hi) { uint64_t t = lo; lo = hi; hi = t; }
                              o.kind = BEPS_OP_RANGE_QUERY; o.u_aux = rng.next(); o.u_key = lo; o.u_key2 = hi; o.value = 32; break; }
                    default: o.kind = BEPS_OP_FLUSH; o.value = (int64_t)(rng.next() % 8); o.value2 = (int64_t)(rng.next() % 32); break;
                }
                ops.push_back(o);
            }
            CUDA_CHECK(cudaMemcpyAsync(d_ops, ops.data(), sizeof(BepsOp) * ops.size(),
                                       cudaMemcpyHostToDevice, stream));

            BepsRunSpec run = {};
            run.abi_version = BEPS_ABI_VERSION;
            run.num_ops = (int32_t)ops.size();
            run.step_id = s;

            BepsInputs in = {};
            in.ops = d_ops;
            CUDA_CHECK(solution_run(state, &run, &in, &out, workspace, wb, stream));
        }

        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

        std::printf("beps_tree_buffer bench: %d steps in %.3f ms (%.4f ms/step)\n",
                    steps, ms, ms / steps);

        cudaFree(d_ops); cudaFree(d_counts); cudaFree(d_hashes); cudaFree(workspace);
        solution_destroy(state);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
