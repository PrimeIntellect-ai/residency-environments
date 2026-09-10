// file: bench_mk_multigpu_allreduce.cu
//
// Throughput benchmark for the linked T68 (MK9) solution. Drives a long stream
// of RANK_STEP / ADVANCE operations over loaded collectives and reports
// operations/sec of the persistent runtime.

#include "mk_multigpu_allreduce_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <chrono>
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
        }                                                                      \
    } while (0)

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) {
        count = n;
        if (n > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
    }
};

static MgaRunSpec rs(int op, int op_index, int a0 = 0, int a1 = 0, int a2 = 0, int a3 = 0,
                     int a4 = 0, int a5 = 0) {
    MgaRunSpec r = {};
    r.abi_version = MGA_ABI_VERSION;
    r.op = op; r.op_index = op_index;
    r.a0 = a0; r.a1 = a1; r.a2 = a2; r.a3 = a3; r.a4 = a4; r.a5 = a5;
    return r;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        MgaProblemSpec spec = {};
        spec.abi_version = MGA_ABI_VERSION;
        spec.rank_count = 8; spec.chunk_count_max = 8; spec.max_collectives = 8;
        spec.max_remote_events = 4096; spec.max_send_credits_per_link = 8;
        spec.remote_latency = 3; spec.max_scheduler_queue_per_rank = 512;

        size_t workspace_bytes = solution_workspace_bytes(&spec);
        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(std::max<size_t>(workspace_bytes, 1));

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(solution_reset(state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<int64_t> d_counters; d_counters.allocate(MGA_COUNTER_COUNT);
        DeviceBuffer<uint64_t> d_evh, d_ch, d_sh, d_crh, d_ph, d_sch, d_clk, d_es;
        d_evh.allocate(1); d_ch.allocate(1); d_sh.allocate(1); d_crh.allocate(1);
        d_ph.allocate(1); d_sch.allocate(1); d_clk.allocate(1); d_es.allocate(1);

        MgaOutputs out = {};
        out.counters = d_counters.ptr; out.remote_event_hash = d_evh.ptr;
        out.collective_hash = d_ch.ptr; out.signal_hash = d_sh.ptr; out.credit_hash = d_crh.ptr;
        out.pending_remote_hash = d_ph.ptr; out.scheduler_hash = d_sch.ptr;
        out.clock_out = d_clk.ptr; out.event_seq_out = d_es.ptr;

        int oi = 0;
        // start a few collectives
        for (int k = 0; k < 4; ++k) {
            int mask = 0xFF;  // all 8 ranks
            MgaRunSpec r = rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 1000 + k, mask, 1 + (k % 8), 7 * k + 1);
            CUDA_CHECK(solution_run(state, &r, nullptr, &out, workspace.ptr, workspace_bytes, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const int kBatches = 400;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < kBatches; ++b) {
            for (int r = 0; r < 8; ++r) {
                MgaRunSpec rstep = rs(MGA_OP_RANK_STEP, oi++, r, 8);
                CUDA_CHECK(solution_run(state, &rstep, nullptr, &out, workspace.ptr, workspace_bytes, stream));
            }
            MgaRunSpec adv = rs(MGA_OP_ADVANCE, oi++, 1, 16);
            CUDA_CHECK(solution_run(state, &adv, nullptr, &out, workspace.ptr, workspace_bytes, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        long total_ops = (long)kBatches * 9;
        std::printf("mk_multigpu_allreduce bench: %ld ops in %.4f s = %.1f ops/s\n",
                    total_ops, sec, total_ops / sec);

        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
