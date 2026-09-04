// file: bench_work_stealing_runtime.cu

#include "work_stealing_runtime_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

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
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                              \
        }                                                                       \
    } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x8e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    int uniform_int(int lo, int hi) { return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1)); }
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr; size_t count = 0;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) { count = n; if (n) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n)); }
};

static WsrRunSpec mk(int kind, int step) { WsrRunSpec r = {}; r.abi_version = WSR_ABI_VERSION; r.op_kind = kind; r.step_id = step; return r; }

static std::vector<WsrRunSpec> build_ops(const WsrProblemSpec& spec) {
    std::vector<WsrRunSpec> ops;
    SplitMix64 rng(0xabcdef0123456789ULL);
    uint64_t nid = 1;
    int t = 0;
    for (int i = 0; i < 64; ++i) {
        WsrRunSpec r = mk(WSR_OP_SPAWN, t++); r.a_task = nid++; r.a_worker = rng.uniform_int(0, spec.W - 1);
        r.a_priority = rng.uniform_int(0, 1); r.a_work = rng.uniform_int(1, 8); ops.push_back(r);
    }
    for (int i = 0; i + 64 < spec.max_steps; ++i) {
        int p = rng.uniform_int(0, 99); int w = rng.uniform_int(0, spec.W - 1);
        WsrRunSpec r;
        if (p < 45) { r = mk(WSR_OP_RUN, t++); r.a_worker = w; r.a_work = rng.uniform_int(1, 6); }
        else if (p < 58) { r = mk(WSR_OP_SPAWN, t++); r.a_task = nid++; r.a_worker = w; r.a_priority = rng.uniform_int(0, 1); r.a_work = rng.uniform_int(1, 6); }
        else if (p < 68) { r = mk(WSR_OP_YIELD, t++); r.a_worker = w; }
        else if (p < 78) { r = mk(WSR_OP_BLOCK, t++); r.a_worker = w; r.a_key = rng.uniform_int(1, 6); }
        else if (p < 86) { r = mk(WSR_OP_SLEEP, t++); r.a_worker = w; r.a_tick = rng.uniform_int(1, 200); }
        else if (p < 92) { r = mk(WSR_OP_WAKE, t++); r.a_key = rng.uniform_int(1, 6); r.a_limit = rng.uniform_int(0, 4); }
        else { r = mk(WSR_OP_ADVANCE, t++); r.a_delta = rng.uniform_int(1, 50); }
        ops.push_back(r);
    }
    return ops;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 10;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        WsrProblemSpec spec = {};
        spec.abi_version = WSR_ABI_VERSION; spec.W = 8; spec.local_cap_per_worker = 8;
        spec.global_cap = 32; spec.max_tasks = 256; spec.max_blocked = 64; spec.max_sleeping = 64;
        spec.max_steps = 1024;
        if (!wsr_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

        const size_t wb = solution_workspace_bytes(&spec);
        if (wb == 0) throw std::runtime_error("workspace 0");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace; workspace.allocate(wb);

        DeviceBuffer<int64_t> d_counts; d_counts.allocate(WSR_COUNT_N);
        DeviceBuffer<int32_t> d_opidx; d_opidx.allocate(1);
        DeviceBuffer<uint64_t> d_clock, d_eseq, d_sched, d_ready, d_run, d_blocked, d_sleep, d_state;
        d_clock.allocate(1); d_eseq.allocate(1); d_sched.allocate(1); d_ready.allocate(1);
        d_run.allocate(1); d_blocked.allocate(1); d_sleep.allocate(1); d_state.allocate(1);

        WsrOutputs out = {};
        out.counts = d_counts.ptr; out.op_index_out = d_opidx.ptr; out.clock_out = d_clock.ptr;
        out.event_seq_out = d_eseq.ptr; out.sched_event_hash = d_sched.ptr; out.ready_hash = d_ready.ptr;
        out.running_hash = d_run.ptr; out.blocked_hash = d_blocked.ptr; out.sleep_hash = d_sleep.ptr;
        out.state_checksum = d_state.ptr;

        WsrInputs in = {}; in.reserved = nullptr;
        const std::vector<WsrRunSpec> ops = build_ops(spec);

        std::printf("bench W=%d ops=%zu max_tasks=%d\n", spec.W, ops.size(), spec.max_tasks);

        for (int warm = 0; warm < 2; ++warm) {
            CUDA_CHECK(solution_reset(state, stream));
            for (const WsrRunSpec& op : ops) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, wb, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        double total_ms = 0;
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(start, stream));
            for (const WsrRunSpec& op : ops) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, wb, stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }
        std::printf("avg_ms=%.6f\n", total_ms / iters);

        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
