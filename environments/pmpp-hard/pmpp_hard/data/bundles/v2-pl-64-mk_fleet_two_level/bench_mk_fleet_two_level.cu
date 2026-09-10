// file: bench_mk_fleet_two_level.cu
//
// Throughput benchmark for the linked T64 solution. Loads a fleet with mixed-scope
// tasks then drives a long stream of SCHED/WORKER/ADVANCE ops, reporting ops/sec
// of the persistent runtime.

#include "mk_fleet_two_level_common.h"

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

static MkfRunSpec op(int o, int oi, int a0, int a1 = 0, int a2 = 0, int a3 = 0) {
    MkfRunSpec r = {};
    r.abi_version = MKF_ABI_VERSION;
    r.op = o; r.op_index = oi; r.a0 = a0; r.a1 = a1; r.a2 = a2; r.a3 = a3;
    return r;
}

static MkfRunSpec submit(int oi, int task_id, int scope, int home, uint64_t mask,
                         int duration, int seed, int oe, int oinc) {
    MkfRunSpec r = {};
    r.abi_version = MKF_ABI_VERSION;
    r.op = MKF_OP_SUBMIT; r.op_index = oi;
    r.a0 = task_id; r.a1 = scope; r.a2 = home; r.a3 = duration;
    r.worker_mask = mask; r.payload_seed = seed; r.output_event = oe; r.output_increment = oinc;
    return r;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        MkfProblemSpec spec = {};
        spec.abi_version = MKF_ABI_VERSION;
        spec.chiplet_count = 8; spec.workers_per_chiplet = 8;
        spec.max_tasks = 256; spec.max_events = 64; spec.max_ready_per_chiplet = 256;
        spec.max_worker_queue = 16; spec.max_running = 256; spec.device_task_chiplet_count = 8;

        size_t workspace_bytes = solution_workspace_bytes(&spec);
        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(std::max<size_t>(workspace_bytes, 1));

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(solution_reset(state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<int64_t> d_counters; d_counters.allocate(MKF_COUNTER_COUNT);
        DeviceBuffer<uint64_t> d_evh, d_th, d_qh, d_ch, d_rh, d_clk, d_es;
        d_evh.allocate(1); d_th.allocate(1); d_qh.allocate(1); d_ch.allocate(1);
        d_rh.allocate(1); d_clk.allocate(1); d_es.allocate(1);

        MkfOutputs out = {};
        out.counters = d_counters.ptr; out.fleet_event_hash = d_evh.ptr;
        out.task_state_hash = d_th.ptr; out.queue_hash = d_qh.ptr; out.counter_hash = d_ch.ptr;
        out.running_hash = d_rh.ptr; out.clock_out = d_clk.ptr; out.event_seq_out = d_es.ptr;

        int oi = 0;
        int total_workers = spec.chiplet_count * spec.workers_per_chiplet;
        for (int k = 0; k < 128; ++k) {
            int scope = k % 4;
            int home = k % spec.chiplet_count;
            uint64_t mask;
            if (scope == MKF_SCOPE_DEVICE) {
                int c0 = k % spec.chiplet_count, c1 = (k + 1) % spec.chiplet_count;
                mask = (1ULL << (c0 * 8)) | (1ULL << (c1 * 8 + 1));
            } else {
                mask = (1ULL << (home * 8)) | (1ULL << (home * 8 + 1)) | (1ULL << (home * 8 + 2));
            }
            MkfRunSpec r = submit(oi++, k + 1, scope, home, mask, 3, 1000 + k, k % spec.max_events, 1);
            CUDA_CHECK(solution_run(state, &r, nullptr, &out, workspace.ptr, workspace_bytes, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const int kRounds = 200;
        auto t0 = std::chrono::high_resolution_clock::now();
        long ops = 0;
        for (int b = 0; b < kRounds; ++b) {
            for (int ch = 0; ch < spec.chiplet_count; ++ch) {
                MkfRunSpec r = op(MKF_OP_SCHED, oi++, ch, 8);
                CUDA_CHECK(solution_run(state, &r, nullptr, &out, workspace.ptr, workspace_bytes, stream));
                ops++;
            }
            for (int w = 0; w < total_workers; ++w) {
                MkfRunSpec r = op(MKF_OP_WORKER, oi++, w, 2);
                CUDA_CHECK(solution_run(state, &r, nullptr, &out, workspace.ptr, workspace_bytes, stream));
                ops++;
            }
            MkfRunSpec r = op(MKF_OP_ADVANCE, oi++, 3, 64);
            CUDA_CHECK(solution_run(state, &r, nullptr, &out, workspace.ptr, workspace_bytes, stream));
            ops++;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        std::printf("mk_fleet_two_level bench: %ld ops in %.4f s = %.1f ops/s\n",
                    ops, sec, ops / sec);

        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
