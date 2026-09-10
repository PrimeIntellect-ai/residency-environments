// file: bench_mesi_directory.cu
//
// Minimal throughput bench: drives a fixed deterministic op stream through the
// linked solution and times solution_run. Not a correctness check.

#include "mesi_directory_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstdio>
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
    explicit SplitMix64(uint64_t s) : state(s) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    int uni(int lo, int hi) { return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1)); }
};

template <typename T>
struct DBuf {
    T* ptr = nullptr; size_t n = 0;
    ~DBuf() { if (ptr) cudaFree(ptr); }
    void alloc(size_t c) { n = c; if (c) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * c)); }
    void up(const std::vector<T>& h) { if (n) CUDA_CHECK(cudaMemcpy(ptr, h.data(), sizeof(T) * n, cudaMemcpyHostToDevice)); }
};

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        MesiProblemSpec spec = {};
        spec.abi_version = MESI_ABI_VERSION;
        spec.core_count = 16;
        spec.line_count = 64;
        spec.cache_capacity_per_core = 8;
        spec.max_pending_lines = 64;
        spec.max_batch = 256;
        spec.max_steps = 64;
        if (!mesi_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t wb = solution_workspace_bytes(&spec);
        DBuf<uint8_t> ws; ws.alloc(std::max<size_t>(wb, 1));

        const int steps = spec.max_steps;
        const int batch = spec.max_batch;
        SplitMix64 rng(0xBEEF1234ULL);

        std::vector<DBuf<int32_t>> d_op(steps), d_core(steps), d_line(steps);
        std::vector<DBuf<int64_t>> d_value(steps);
        std::vector<DBuf<uint64_t>> d_txn(steps);

        for (int s = 0; s < steps; ++s) {
            std::vector<int32_t> op(batch), core(batch), line(batch);
            std::vector<int64_t> val(batch);
            std::vector<uint64_t> txn(batch);
            for (int i = 0; i < batch; ++i) {
                int r = rng.uni(0, 99);
                op[i] = r < 40 ? MESI_OP_LOAD : (r < 70 ? MESI_OP_STORE : (r < 85 ? MESI_OP_ACK_INV : (r < 95 ? MESI_OP_EVICT : MESI_OP_FLUSH)));
                core[i] = rng.uni(0, spec.core_count - 1);
                line[i] = rng.uni(0, spec.line_count - 1);
                val[i] = (int64_t)rng.next_u64();
                txn[i] = (uint64_t)rng.uni(1, 16);
            }
            d_op[s].alloc(batch); d_op[s].up(op);
            d_core[s].alloc(batch); d_core[s].up(core);
            d_line[s].alloc(batch); d_line[s].up(line);
            d_value[s].alloc(batch); d_value[s].up(val);
            d_txn[s].alloc(batch); d_txn[s].up(txn);
        }

        DBuf<int64_t> d_counts; d_counts.alloc(MESI_COUNT_FIELDS);
        DBuf<uint64_t> d_coh, d_cache, d_dir, d_pend, d_evseq, d_state;
        d_coh.alloc(1); d_cache.alloc(1); d_dir.alloc(1); d_pend.alloc(1); d_evseq.alloc(1); d_state.alloc(1);

        MesiOutputs out = {};
        out.counts = d_counts.ptr; out.coh_event_hash = d_coh.ptr; out.cache_hash = d_cache.ptr;
        out.directory_hash = d_dir.ptr; out.pending_hash = d_pend.ptr;
        out.event_seq_out = d_evseq.ptr; out.state_checksum = d_state.ptr;

        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));

        // warmup
        for (int s = 0; s < steps; ++s) {
            MesiRunSpec run = {}; run.abi_version = MESI_ABI_VERSION; run.batch_size = batch; run.step_id = s;
            MesiInputs in = {}; in.op = d_op[s].ptr; in.arg_core = d_core[s].ptr; in.arg_line = d_line[s].ptr;
            in.arg_value = d_value[s].ptr; in.arg_txn = d_txn[s].ptr;
            CUDA_CHECK(solution_run(state, &run, &in, &out, ws.ptr, std::max<size_t>(wb, 1), stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(solution_reset(state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaEventRecord(e0, stream));
        for (int s = 0; s < steps; ++s) {
            MesiRunSpec run = {}; run.abi_version = MESI_ABI_VERSION; run.batch_size = batch; run.step_id = s;
            MesiInputs in = {}; in.op = d_op[s].ptr; in.arg_core = d_core[s].ptr; in.arg_line = d_line[s].ptr;
            in.arg_value = d_value[s].ptr; in.arg_txn = d_txn[s].ptr;
            CUDA_CHECK(solution_run(state, &run, &in, &out, ws.ptr, std::max<size_t>(wb, 1), stream));
        }
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("bench: %d steps x %d ops in %.3f ms (%.2f us/step)\n", steps, batch, ms, ms * 1000.0 / steps);

        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
