// file: bench_lock_manager_deadlock.cu

#include "lock_manager_deadlock_common.h"

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

static LmdRunSpec mk(int kind, int step) { LmdRunSpec r = {}; r.abi_version = LMD_ABI_VERSION; r.op_kind = kind; r.step_id = step; r.a_partition=-1; r.a_row=-1; return r; }

static std::vector<LmdRunSpec> build_ops(const LmdProblemSpec& spec) {
    std::vector<LmdRunSpec> ops;
    SplitMix64 rng(0xabcdef0123456789ULL);
    int t = 0;
    for (uint64_t id = 1; id <= 12; ++id) {
        LmdRunSpec r = mk(LMD_OP_BEGIN, t++); r.a_txn = id; r.a_priority = rng.uniform_int(0,4); ops.push_back(r);
    }
    int modes[5] = {LMD_IS, LMD_IX, LMD_S, LMD_SIX, LMD_X};
    const int total_ops = 256;
    for (int i = 0; i + 12 < total_ops; ++i) {
        int p = rng.uniform_int(0, 99);
        uint64_t txn = rng.uniform_int(1, 12);
        int tb = rng.uniform_int(0, spec.table_count - 1);
        int pp = rng.uniform_int(0, spec.partitions_per_table - 1);
        int rr = rng.uniform_int(0, spec.rows_per_partition - 1);
        LmdRunSpec r;
        if (p < 45) {
            int g = rng.uniform_int(0, 2);
            r = mk(LMD_OP_LOCK, t++); r.a_txn = txn; r.a_mode = modes[rng.uniform_int(0,4)];
            if (g == 0) { r.a_res_kind = LMD_ROW; r.a_table=tb; r.a_partition=pp; r.a_row=rr; }
            else if (g == 1) { r.a_res_kind = LMD_PARTITION; r.a_table=tb; r.a_partition=pp; r.a_row=-1; }
            else { r.a_res_kind = LMD_TABLE; r.a_table=tb; r.a_partition=-1; r.a_row=-1; }
        } else if (p < 60) {
            r = mk(LMD_OP_CONVERT, t++); r.a_txn = txn; r.a_res_kind = LMD_ROW; r.a_table=tb; r.a_partition=pp; r.a_row=rr; r.a_mode = modes[rng.uniform_int(0,4)];
        } else if (p < 75) {
            r = mk(LMD_OP_UNLOCK, t++); r.a_txn = txn; r.a_res_kind = LMD_ROW; r.a_table=tb; r.a_partition=pp; r.a_row=rr;
        } else if (p < 82) {
            r = mk(LMD_OP_UNLOCK_ALL, t++); r.a_txn = txn;
        } else {
            r = mk(LMD_OP_DETECT_DEADLOCK, t++); r.a_limit = rng.uniform_int(0, 8);
        }
        ops.push_back(r);
    }
    return ops;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 10;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        LmdProblemSpec spec = {};
        spec.abi_version = LMD_ABI_VERSION; spec.table_count = 3; spec.partitions_per_table = 3;
        spec.rows_per_partition = 6; spec.max_txns = 24; spec.max_locks = 1024; spec.max_waiters = 256;
        spec.escalation_threshold = 4; spec.max_deadlock_cycles_per_detect = 16;
        if (!lmd_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

        const size_t wb = solution_workspace_bytes(&spec);
        if (wb == 0) throw std::runtime_error("workspace 0");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace; workspace.allocate(std::max<size_t>(wb, 1));

        DeviceBuffer<int64_t> d_counts; d_counts.allocate(LMD_COUNT_N);
        DeviceBuffer<int32_t> d_opidx; d_opidx.allocate(1);
        DeviceBuffer<uint64_t> d_eseq, d_ev, d_grant, d_wait, d_txn, d_state;
        d_eseq.allocate(1); d_ev.allocate(1); d_grant.allocate(1);
        d_wait.allocate(1); d_txn.allocate(1); d_state.allocate(1);

        LmdOutputs out = {};
        out.counts = d_counts.ptr; out.op_index_out = d_opidx.ptr; out.event_seq_out = d_eseq.ptr;
        out.lock_event_hash = d_ev.ptr; out.grant_hash = d_grant.ptr; out.wait_hash = d_wait.ptr;
        out.txn_lock_hash = d_txn.ptr; out.state_checksum = d_state.ptr;

        LmdInputs in = {}; in.reserved = nullptr;
        const std::vector<LmdRunSpec> ops = build_ops(spec);

        std::printf("bench tables=%d ops=%zu max_locks=%d\n", spec.table_count, ops.size(), spec.max_locks);

        for (int warm = 0; warm < 2; ++warm) {
            CUDA_CHECK(solution_reset(state, stream));
            for (const LmdRunSpec& op : ops) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, wb, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        double total_ms = 0;
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(start, stream));
            for (const LmdRunSpec& op : ops) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, wb, stream));
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
