// file: bench_wormhole_mesh_credits.cu
//
// Throughput benchmark for the linked T52 solution. Drives a long TICK stream
// over a loaded fabric and reports cycles/sec of the persistent simulation.

#include "wormhole_mesh_credits_common.h"

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

static WmcRunSpec rs(int op, int op_index, int a0, int a1 = 0, int a2 = 0, int a3 = 0, int a4 = 0) {
    WmcRunSpec r = {};
    r.abi_version = WMC_ABI_VERSION;
    r.op = op; r.op_index = op_index;
    r.a0 = a0; r.a1 = a1; r.a2 = a2; r.a3 = a3; r.a4 = a4;
    return r;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        WmcProblemSpec spec = {};
        spec.abi_version = WMC_ABI_VERSION;
        spec.rows = 8; spec.cols = 8; spec.vc_count = 4; spec.buffer_cap_per_vc = 4;
        spec.credit_latency = 3; spec.max_packets = 256;
        spec.max_injection_queue_per_node = 8; spec.max_credit_events = 4096;

        size_t workspace_bytes = solution_workspace_bytes(&spec);
        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(std::max<size_t>(workspace_bytes, 1));

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(solution_reset(state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<int64_t> d_counters; d_counters.allocate(WMC_COUNTER_COUNT);
        DeviceBuffer<uint64_t> d_evh, d_ph, d_bh, d_ch, d_cqh, d_cyc, d_es;
        d_evh.allocate(1); d_ph.allocate(1); d_bh.allocate(1); d_ch.allocate(1);
        d_cqh.allocate(1); d_cyc.allocate(1); d_es.allocate(1);

        WmcOutputs out = {};
        out.counters = d_counters.ptr; out.fabric_event_hash = d_evh.ptr;
        out.packet_hash = d_ph.ptr; out.buffer_hash = d_bh.ptr; out.credit_hash = d_ch.ptr;
        out.credit_queue_hash = d_cqh.ptr; out.cycle_out = d_cyc.ptr; out.event_seq_out = d_es.ptr;

        int oi = 0;
        // load the fabric with traffic
        for (int k = 0; k < 64; ++k) {
            int src = (k * 7) % 64, dst = (k * 11 + 5) % 64;
            if (src == dst) dst = (dst + 1) % 64;
            WmcRunSpec r = rs(WMC_OP_INJECT, oi++, 1000 + k, src, dst, 4, k % 2);
            CUDA_CHECK(solution_run(state, &r, nullptr, &out, workspace.ptr, workspace_bytes, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const int kBatches = 200;
        const int kTicksPerBatch = 4;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < kBatches; ++b) {
            WmcRunSpec r = rs(WMC_OP_TICK, oi++, kTicksPerBatch, 0, 0, 0, 0);
            CUDA_CHECK(solution_run(state, &r, nullptr, &out, workspace.ptr, workspace_bytes, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        long total_cycles = (long)kBatches * kTicksPerBatch;
        std::printf("wormhole_mesh_credits bench: %ld cycles in %.4f s = %.1f cycles/s\n",
                    total_cycles, sec, total_cycles / sec);

        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
