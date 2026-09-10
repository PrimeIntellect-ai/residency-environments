// file: bench_mk_warp_pipeline_mbarrier.cu

#include "mk_warp_pipeline_mbarrier_common.h"

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

static MkwpRunSpec mk(int kind, int step) { MkwpRunSpec r = {}; r.abi_version = MKWP_ABI_VERSION; r.op_kind = kind; r.step_id = step; return r; }

static std::vector<MkwpRunSpec> build_ops(const MkwpProblemSpec& spec) {
    std::vector<MkwpRunSpec> ops;
    SplitMix64 rng(0xabcdef0123456789ULL);
    uint64_t nid = 1;
    int t = 0;
    for (int i = 0; i < 32; ++i) {
        MkwpRunSpec r = mk(MKWP_OP_ENQUEUE_TILE, t++);
        r.a_tile = nid++; r.a_load_bytes = rng.uniform_int(1, 8);
        r.a_compute_iters = rng.uniform_int(1, 8); r.a_store_bytes = rng.uniform_int(1, 8);
        r.a_seed = rng.next_u64(); ops.push_back(r);
    }
    for (int i = 0; i + 32 < spec.max_steps; ++i) {
        int p = rng.uniform_int(0, 99);
        MkwpRunSpec r;
        if (p < 14) { r = mk(MKWP_OP_ENQUEUE_TILE, t++); r.a_tile = nid++; r.a_load_bytes = rng.uniform_int(1, 8); r.a_compute_iters = rng.uniform_int(1, 8); r.a_store_bytes = rng.uniform_int(1, 8); r.a_seed = rng.next_u64(); }
        else if (p < 30) { r = mk(MKWP_OP_LOADER_STEP, t++); r.a_role_id = rng.uniform_int(0, spec.loader_warps - 1); r.a_limit = rng.uniform_int(1, 4); }
        else if (p < 46) { r = mk(MKWP_OP_COMPUTE_STEP, t++); r.a_role_id = rng.uniform_int(0, spec.compute_warps - 1); r.a_limit = rng.uniform_int(1, 4); }
        else if (p < 62) { r = mk(MKWP_OP_STORER_STEP, t++); r.a_role_id = rng.uniform_int(0, spec.storer_warps - 1); r.a_limit = rng.uniform_int(1, 4); }
        else if (p < 88) { r = mk(MKWP_OP_ADVANCE, t++); r.a_delta = rng.uniform_int(0, 12); r.a_limit = rng.uniform_int(1, 8); }
        else if (p < 95) { r = mk(MKWP_OP_CANCEL_TILE, t++); r.a_tile = rng.uniform_int(1, (int)nid); }
        else { r = mk(MKWP_OP_RESET_BARRIER, t++); r.a_barrier = rng.uniform_int(0, spec.barrier_count - 1); }
        ops.push_back(r);
    }
    return ops;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 10;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        MkwpProblemSpec spec = {};
        spec.abi_version = MKWP_ABI_VERSION;
        spec.buffer_count = 6; spec.loader_warps = 2; spec.compute_warps = 2; spec.storer_warps = 2;
        spec.barrier_count = 6 * MKWP_BARRIERS_PER_BUFFER;
        spec.max_tiles = 256; spec.max_pending_async = 1024; spec.max_role_queue = 256;
        spec.max_steps = 1024;
        if (!mkwp_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

        const size_t wb = solution_workspace_bytes(&spec);
        if (wb == 0) throw std::runtime_error("workspace 0");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace; workspace.allocate(std::max<size_t>(wb, 1));

        DeviceBuffer<int64_t> d_counts; d_counts.allocate(MKWP_COUNT_N);
        DeviceBuffer<int32_t> d_opidx; d_opidx.allocate(1);
        DeviceBuffer<uint64_t> d_clock, d_eseq, d_pipe, d_buf, d_bar, d_tile, d_async, d_state;
        d_clock.allocate(1); d_eseq.allocate(1); d_pipe.allocate(1); d_buf.allocate(1);
        d_bar.allocate(1); d_tile.allocate(1); d_async.allocate(1); d_state.allocate(1);

        MkwpOutputs out = {};
        out.counts = d_counts.ptr; out.op_index_out = d_opidx.ptr; out.clock_out = d_clock.ptr;
        out.event_seq_out = d_eseq.ptr; out.pipe_event_hash = d_pipe.ptr; out.buffer_hash = d_buf.ptr;
        out.barrier_hash = d_bar.ptr; out.tile_hash = d_tile.ptr; out.async_hash = d_async.ptr;
        out.state_checksum = d_state.ptr;

        MkwpInputs in = {}; in.reserved = nullptr;
        const std::vector<MkwpRunSpec> ops = build_ops(spec);

        std::printf("bench buffers=%d ops=%zu max_tiles=%d\n", spec.buffer_count, ops.size(), spec.max_tiles);

        for (int warm = 0; warm < 2; ++warm) {
            CUDA_CHECK(solution_reset(state, stream));
            for (const MkwpRunSpec& op : ops) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, wb, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        double total_ms = 0;
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(start, stream));
            for (const MkwpRunSpec& op : ops) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, wb, stream));
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
