// file: bench_mk_chunked_dep_counters.cu

#include "mk_chunked_dep_counters_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::ostringstream _oss;                                           \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "   \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                             \
        }                                                                       \
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
    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("upload size mismatch");
        if (count > 0) CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
    }
};

static MkOp mk(int kind, int a = 0, int b = 0, int cc = 0, int d = 0, int e = 0,
               int f = 0, int64_t i64 = 0) {
    MkOp op;
    op.kind = kind; op.arg_a = a; op.arg_b = b; op.arg_c = cc; op.arg_d = d;
    op.arg_e = e; op.arg_f = f; op.pad = 0; op.arg_i64 = i64;
    return op;
}

// Build a long mixed op stream: define/produce/advance/arm/consume cycles.
static std::vector<MkOp> make_ops(int edges, int chunks, int total_ops) {
    std::vector<MkOp> ops;
    for (int e = 0; e < edges; ++e) ops.push_back(mk(MK_OP_DEFINE_EDGE, e, chunks));
    int clk = 0;
    int cons = 1;
    while ((int)ops.size() < total_ops) {
        int e = (int)ops.size() % edges;
        // PRODUCE: a=prod, b=edge, c=first, d=chunks, e=seed, f=latency, i64=increment.
        ops.push_back(mk(MK_OP_PRODUCE, 1000 + e, e, 0, chunks, e * 7 + 1, 2, /*increment*/3));
        // ARM_WAIT: a=cons, b=edge, c=chunk, e=seed, i64=target.
        for (int ch = 0; ch < chunks; ++ch)
            ops.push_back(mk(MK_OP_ARM_WAIT, cons++, e, ch, 0, ch + 1, 0, /*target*/2));
        ops.push_back(mk(MK_OP_ADVANCE, 5, chunks));
        clk += 5;
        ops.push_back(mk(MK_OP_CONSUME, chunks));
    }
    if ((int)ops.size() > total_ops) ops.resize(total_ops);
    return ops;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        // Keep the cumulative number of armed waiters within MK_MAX_WAITERS: the
        // waiter pool is append-only within a single run (wait_seq only grows).
        const int edges = 8, chunks = 8, total_ops = 900;

        MkProblemSpec spec{};
        spec.abi_version = MK_ABI_VERSION;
        spec.edge_count = edges;
        spec.max_chunks_per_edge = chunks;
        spec.max_waiters = MK_MAX_WAITERS;
        spec.max_ready_entries = MK_MAX_READY;
        spec.max_store_events = MK_MAX_STORE_EVENTS;
        spec.max_consumers = MK_MAX_CONSUMERS;
        spec.max_epoch = 32;
        spec.max_ops = MK_MAX_OPS;

        if (!mk_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

        size_t workspace_bytes = solution_workspace_bytes(&spec);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(std::max<size_t>(workspace_bytes, 1));

        std::vector<MkOp> ops = make_ops(edges, chunks, total_ops);
        DeviceBuffer<MkOp> d_ops;
        d_ops.allocate(ops.size());
        d_ops.upload(ops);

        DeviceBuffer<uint64_t> d_counts, d_eh, d_ch, d_wh, d_rh, d_ph, d_sc;
        d_counts.allocate(MK_COUNT_TOTAL);
        d_eh.allocate(1); d_ch.allocate(1); d_wh.allocate(1); d_rh.allocate(1);
        d_ph.allocate(1); d_sc.allocate(6);

        MkRunSpec run{};
        run.abi_version = MK_ABI_VERSION;
        run.op_count = (int32_t)ops.size();

        MkInputs inputs{}; inputs.ops = d_ops.ptr;
        MkOutputs outputs{};
        outputs.counts = d_counts.ptr; outputs.event_hash = d_eh.ptr;
        outputs.cell_hash = d_ch.ptr; outputs.waiter_hash = d_wh.ptr;
        outputs.ready_hash = d_rh.ptr; outputs.pending_hash = d_ph.ptr;
        outputs.state_scalars = d_sc.ptr;

        // warmup
        CUDA_CHECK(solution_reset(state, stream));
        CUDA_CHECK(solution_run(state, &run, &inputs, &outputs, workspace.ptr,
                                std::max<size_t>(workspace_bytes, 1), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const int iters = 50;
        cudaEvent_t a, b;
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
        CUDA_CHECK(cudaEventRecord(a, stream));
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(solution_run(state, &run, &inputs, &outputs, workspace.ptr,
                                    std::max<size_t>(workspace_bytes, 1), stream));
        }
        CUDA_CHECK(cudaEventRecord(b, stream));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));

        std::printf("bench: %d ops x %d iters in %.3f ms (%.3f us/run)\n",
                    (int)ops.size(), iters, ms, (ms * 1000.0) / iters);

        solution_destroy(state);
        CUDA_CHECK(cudaEventDestroy(a));
        CUDA_CHECK(cudaEventDestroy(b));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
