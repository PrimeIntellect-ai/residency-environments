// file: bench_raft_log_snapshot.cu
//
// Minimal timing harness for the raft_log_snapshot kernel ABI. Drives a fixed
// op-batch workload through solution_init / solution_run and reports wall time.

#include "raft_log_snapshot_common.h"

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

        RaftProblemSpec spec = {};
        spec.abi_version = RAFT_ABI_VERSION;
        spec.server_count = 5;
        spec.max_log_entries_per_server = 256;
        spec.max_pending_append_rpcs = 64;
        spec.max_entries_per_append = 32;
        spec.max_apply_per_op = 64;
        spec.max_ops = 64;
        spec.max_steps = 64;
        if (!raft_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));

        size_t wb = solution_workspace_bytes(&spec);
        uint8_t* workspace = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&workspace), std::max<size_t>(wb, 1)));

        RaftOp* d_ops = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ops), sizeof(RaftOp) * spec.max_ops));

        RaftCounts* d_counts = nullptr;
        uint64_t* d_hashes = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_counts), sizeof(RaftCounts)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hashes), sizeof(uint64_t) * 5));

        RaftOutputs out = {};
        out.counts = d_counts;
        out.raft_event_hash = d_hashes + 0;
        out.log_hash = d_hashes + 1;
        out.leader_state_hash = d_hashes + 2;
        out.pending_rpc_hash = d_hashes + 3;
        out.apply_hash = d_hashes + 4;

        SplitMix64 rng(0xC0FFEE1234ULL);
        const int steps = 64;

        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0, stream));

        uint64_t cmd = 1;
        for (int s = 0; s < steps; ++s) {
            std::vector<RaftOp> ops;
            if (s == 0) { RaftOp o = {}; o.kind = RAFT_OP_BECOME_LEADER; o.i_a = 0; o.u_a = 1; ops.push_back(o); }
            for (int k = 0; k < 8; ++k) {
                RaftOp o = {};
                int sel = (int)(rng.next() % 6);
                switch (sel) {
                    case 0: o.kind = RAFT_OP_CLIENT_APPEND; o.u_a = cmd++; o.value = (int64_t)rng.next(); break;
                    case 1: o.kind = RAFT_OP_SEND_APPEND; o.i_a = 1 + (int)(rng.next() % 4); o.i_b = 8; break;
                    case 2: o.kind = RAFT_OP_DELIVER_APPEND; o.u_a = 1 + (rng.next() % 32); break;
                    case 3: o.kind = RAFT_OP_ADVANCE_COMMIT; break;
                    case 4: o.kind = RAFT_OP_APPLY; o.i_a = (int)(rng.next() % 5); o.i_b = 8; break;
                    default: o.kind = RAFT_OP_TAKE_SNAPSHOT; o.i_a = (int)(rng.next() % 5); break;
                }
                ops.push_back(o);
            }
            CUDA_CHECK(cudaMemcpyAsync(d_ops, ops.data(), sizeof(RaftOp) * ops.size(),
                                       cudaMemcpyHostToDevice, stream));

            RaftRunSpec run = {};
            run.abi_version = RAFT_ABI_VERSION;
            run.num_ops = (int32_t)ops.size();
            run.step_id = s;

            RaftInputs in = {};
            in.ops = d_ops;
            CUDA_CHECK(solution_run(state, &run, &in, &out, workspace, wb, stream));
        }

        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

        std::printf("raft_log_snapshot bench: %d steps in %.3f ms (%.4f ms/step)\n",
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
