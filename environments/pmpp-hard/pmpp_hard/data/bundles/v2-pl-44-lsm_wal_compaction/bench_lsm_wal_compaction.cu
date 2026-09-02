// file: bench_lsm_wal_compaction.cu

#include "lsm_wal_compaction_common.h"

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
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
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
    void allocate(size_t n) { count = n; if (n > 0) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T)*n)); }
    void upload(const std::vector<T>& h) { if (count) CUDA_CHECK(cudaMemcpy(ptr, h.data(), sizeof(T)*count, cudaMemcpyHostToDevice)); }
};

static LsmProblemSpec make_bench_spec() {
    LsmProblemSpec s = {};
    s.abi_version = LSM_ABI_VERSION;
    s.level_count = 4;
    s.max_files_per_level[0] = 6; s.max_files_per_level[1] = 6;
    s.max_files_per_level[2] = 6; s.max_files_per_level[3] = 6;
    s.memtable_record_cap = 6; s.sst_record_cap = 4; s.wal_segment_record_cap = 6;
    s.max_wal_segments = 10; s.max_snapshots = 8; s.max_ops = 200; s.max_steps = 48;
    if (!lsm_validate_problem_spec(&s)) throw std::runtime_error("bad bench spec");
    return s;
}

struct DeviceStep {
    LsmRunSpec run;
    DeviceBuffer<LsmOp> ops;
    LsmInputs inputs;
    DeviceBuffer<LsmCounts> counts;
    DeviceBuffer<uint64_t> read, write, compact, wal, state, snap;
    LsmOutputs outputs;
};

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 20;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        const LsmProblemSpec spec = make_bench_spec();
        size_t workspace_bytes = solution_workspace_bytes(&spec);
        if (workspace_bytes == 0) throw std::runtime_error("workspace_bytes 0");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(std::max<size_t>(workspace_bytes, 1));

        SplitMix64 rng(0x1234abcdULL);
        std::vector<DeviceStep*> steps;
        for (int s = 0; s < 40; ++s) {
            DeviceStep* ds = new DeviceStep();
            std::vector<LsmOp> ops;
            int n = rng.uniform_int(4, 12);
            for (int i = 0; i < n; ++i) {
                int r = rng.uniform_int(0, 99);
                LsmOp o = {};
                uint32_t key = (uint32_t)rng.uniform_int(0, 40);
                if (r < 50) { o.kind = LSM_OP_PUT; o.i_a = (int32_t)key; o.value = (int64_t)rng.next_u64(); }
                else if (r < 65) { o.kind = LSM_OP_DEL; o.i_a = (int32_t)key; }
                else if (r < 82) { o.kind = LSM_OP_GET; o.i_a = (int32_t)key; o.u_a = (uint64_t)i; o.u_b = 0; }
                else if (r < 88) { o.kind = LSM_OP_FLUSH; }
                else if (r < 94) { o.kind = LSM_OP_COMPACT; o.i_a = rng.uniform_int(0, 2); o.i_b = rng.uniform_int(1, 3); }
                else { o.kind = LSM_OP_CHECKPOINT_WAL; }
                ops.push_back(o);
            }
            ops.push_back(LsmOp{LSM_OP_FLUSH, 0, 0, 0, 0, 0, 0});
            ds->run = {}; ds->run.abi_version = LSM_ABI_VERSION; ds->run.num_ops = (int32_t)ops.size(); ds->run.step_id = s;
            ds->ops.allocate(ops.size()); ds->ops.upload(ops);
            ds->inputs = {}; ds->inputs.ops = ds->ops.ptr;
            ds->counts.allocate(1); ds->read.allocate(1); ds->write.allocate(1);
            ds->compact.allocate(1); ds->wal.allocate(1); ds->state.allocate(1); ds->snap.allocate(1);
            ds->outputs = {};
            ds->outputs.counts = ds->counts.ptr; ds->outputs.read_hash = ds->read.ptr;
            ds->outputs.write_hash = ds->write.ptr; ds->outputs.compaction_hash = ds->compact.ptr;
            ds->outputs.wal_hash = ds->wal.ptr; ds->outputs.lsm_state_hash = ds->state.ptr;
            ds->outputs.snapshot_hash = ds->snap.ptr;
            steps.push_back(ds);
        }

        std::printf("bench_sequence levels=%d steps=%zu workspace=%zu\n", spec.level_count, steps.size(), workspace_bytes);

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

        for (int w = 0; w < 3; ++w) {
            CUDA_CHECK(solution_reset(state, stream));
            for (DeviceStep* ds : steps)
                CUDA_CHECK(solution_run(state, &ds->run, &ds->inputs, &ds->outputs, workspace.ptr, workspace_bytes, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(solution_reset(state, stream));
            for (DeviceStep* ds : steps)
                CUDA_CHECK(solution_run(state, &ds->run, &ds->inputs, &ds->outputs, workspace.ptr, workspace_bytes, stream));
        }
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        std::printf("iters=%d total_ms=%.3f per_iter_ms=%.4f\n", iters, ms, ms / iters);

        for (DeviceStep* ds : steps) delete ds;
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
