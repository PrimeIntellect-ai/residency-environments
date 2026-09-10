// file: bench_rcu_epoch_reclaimer.cu

#include "rcu_epoch_reclaimer_common.h"
#include "pmpp_bench_digest.cuh"

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

static RcuOp mk(int kind, int a = 0, int b = 0, int cc = 0, int64_t i64 = 0) {
    RcuOp op; op.kind = kind; op.arg_a = a; op.arg_b = b; op.arg_c = cc; op.arg_i64 = i64; return op;
}

// Build a long mixed op stream: alloc/link/lock/read/retire/reclaim cycles.
static std::vector<RcuOp> make_ops(int max_objects, int roots, int threads, int total_ops,
                                   uint64_t seed) {
    std::vector<RcuOp> ops;
    int next_alloc_req = 0;
    int live = 0;
    int t = 0;
    // ALLOC payload values derive from PMPP_BENCH_SEED (they feed object_hash /
    // read_hash), so graded hashes are not precomputable offline; the op-stream
    // structure stays fixed so the timed workload shape is comparable.
    uint64_t z = seed;
    auto next_i64 = [&z]() {
        z += 0x9e3779b97f4a7c15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return (int64_t)(x ^ (x >> 31));
    };
    while ((int)ops.size() < total_ops) {
        // alloc a batch
        for (int i = 0; i < 8 && live < max_objects - 1; ++i) {
            ops.push_back(mk(RCU_OP_ALLOC, next_alloc_req++, 0, 0, next_i64()));
            ++live;
        }
        // link a few (ids are unknown statically; just link 1->0 slot etc with low ids)
        ops.push_back(mk(RCU_OP_SET_ROOT, 0, 1));
        ops.push_back(mk(RCU_OP_LINK, 1, 0, 2));
        ops.push_back(mk(RCU_OP_LINK, 2, 0, 3));
        // reader
        ops.push_back(mk(RCU_OP_READ_LOCK, t % threads));
        ops.push_back(mk(RCU_OP_READ_CHAIN, t % threads, 0, 8));
        ops.push_back(mk(RCU_OP_READ_UNLOCK, t % threads));
        ops.push_back(mk(RCU_OP_ADVANCE_EPOCH));
        // retire + reclaim some high ids if present
        ops.push_back(mk(RCU_OP_LINK, 2, 0, 0));
        ops.push_back(mk(RCU_OP_RETIRE, 3, 9));
        ops.push_back(mk(RCU_OP_RECLAIM, 4));
        live = live > 1 ? live - 1 : live;
        ++t;
    }
    ops.resize(total_ops);
    return ops;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 50;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        RcuProblemSpec spec{};
        spec.abi_version = RCU_ABI_VERSION;
        spec.thread_count = 8;
        spec.root_count = 8;
        spec.max_objects = 256;
        spec.max_edges_per_object = 4;
        spec.max_retired = 256;
        spec.max_free_ids = 256;
        spec.max_ops = RCU_MAX_OPS;

        // One INDEPENDENT op-stream variant per timed iteration (plus variant 0 for
        // warmup): every variant has the SAME op-stream structure and length (timing
        // comparable) but the ALLOC arg_i64 payloads derive from a per-variant seed, so
        // the graded object/read/event hashes differ each iteration. Each iteration's
        // graded outputs are folded into out_fnv, so a compute-once-then-replay solution
        // serves stale outputs for iterations 2..K → the folded digest diverges → perf
        // FAIL. Per-variant seeds come from a SplitMix64 stream over PMPP_BENCH_SEED.
        uint64_t vseed = pmpp::bench_seed(0x8f14e45fceea167aULL) ^ 0x700000000ULL;
        auto next_seed = [&vseed]() -> uint64_t {
            vseed += 0x9e3779b97f4a7c15ULL;
            uint64_t x = vseed;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            return x ^ (x >> 31);
        };

        const size_t workspace_bytes = solution_workspace_bytes(&spec);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(std::max<size_t>(workspace_bytes, 1));

        std::vector<DeviceBuffer<RcuOp>*> d_ops_variants((size_t)iters + 1, nullptr);
        size_t op_count = 0;
        for (int v = 0; v <= iters; ++v) {
            std::vector<RcuOp> host_ops = make_ops(spec.max_objects, spec.root_count,
                                                   spec.thread_count, 2048, next_seed());
            op_count = host_ops.size();
            d_ops_variants[(size_t)v] = new DeviceBuffer<RcuOp>();
            d_ops_variants[(size_t)v]->allocate(host_ops.size());
            d_ops_variants[(size_t)v]->upload(host_ops);
        }

        DeviceBuffer<uint64_t> d_counts; d_counts.allocate(RCU_COUNT_TOTAL);
        DeviceBuffer<uint64_t> d_eh, d_rh, d_roh, d_oh, d_th, d_fh, d_sc;
        d_eh.allocate(1); d_rh.allocate(1); d_roh.allocate(1); d_oh.allocate(1);
        d_th.allocate(1); d_fh.allocate(1); d_sc.allocate(6);

        RcuInputs inputs{};
        RcuOutputs outputs{};
        outputs.counts = d_counts.ptr; outputs.event_hash = d_eh.ptr; outputs.read_hash = d_rh.ptr;
        outputs.root_hash = d_roh.ptr; outputs.object_hash = d_oh.ptr; outputs.thread_hash = d_th.ptr;
        outputs.free_hash = d_fh.ptr; outputs.state_scalars = d_sc.ptr;

        RcuRunSpec run{};
        run.abi_version = RCU_ABI_VERSION;
        run.op_count = static_cast<int32_t>(op_count);

        std::printf("bench_rcu ops=%zu objects=%d threads=%d\n",
                    op_count, spec.max_objects, spec.thread_count);

        for (int w = 0; w < 3; ++w) {
            inputs.ops = d_ops_variants[0]->ptr;
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(solution_run(state, &run, &inputs, &outputs,
                                    workspace.ptr, std::max<size_t>(workspace_bytes, 1), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start = nullptr, stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        double total_ms = 0.0;

        // Fold each timed iteration's graded outputs (outside the timed region) so a
        // stale/replayed output on any iteration breaks the digest.
        pmpp::OutFnv dg;

        for (int it = 0; it < iters; ++it) {
            inputs.ops = d_ops_variants[(size_t)it + 1]->ptr;

            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUDA_CHECK(solution_run(state, &run, &inputs, &outputs,
                                    workspace.ptr, std::max<size_t>(workspace_bytes, 1), stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += (double)ms;

            dg.dev(d_counts.ptr, (size_t)RCU_COUNT_TOTAL * sizeof(uint64_t));
            dg.dev(d_eh.ptr, sizeof(uint64_t));
            dg.dev(d_rh.ptr, sizeof(uint64_t));
            dg.dev(d_roh.ptr, sizeof(uint64_t));
            dg.dev(d_oh.ptr, sizeof(uint64_t));
            dg.dev(d_th.ptr, sizeof(uint64_t));
            dg.dev(d_fh.ptr, sizeof(uint64_t));
            dg.dev(d_sc.ptr, 6 * sizeof(uint64_t));
        }

        std::printf("avg_ms=%.6f\n", total_ms / (double)iters);
        dg.print();

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (DeviceBuffer<RcuOp>* p : d_ops_variants) delete p;
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
