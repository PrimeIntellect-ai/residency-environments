// file: bench_mk_pipeline_scoreboard.cu
//
// Minimal throughput bench: drives a fixed deterministic op stream through the
// linked solution and times solution_run. Not a correctness check.

#include "mk_pipeline_scoreboard_common.h"

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
        MkpsProblemSpec spec = {};
        spec.abi_version = MKPS_ABI_VERSION;
        spec.buffer_count = 32;
        spec.tile_count = 64;
        spec.max_instrs = 128;
        spec.max_reads_per_instr = 4;
        spec.max_writes_per_instr = 4;
        spec.issue_window = 16;
        spec.max_pending_ops = 128;
        spec.counter_count = 16;
        spec.max_batch = 64;
        spec.max_steps = 64;
        if (!mkps_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

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

        std::vector<DBuf<int32_t>> d_op(steps);
        std::vector<DBuf<uint32_t>> d_a0(steps), d_a1(steps), d_a2(steps), d_a3(steps),
            d_a4(steps), d_a5(steps), d_a6(steps), d_a7(steps), d_tiles(steps);
        std::vector<DBuf<uint64_t>> d_a8(steps);

        uint32_t next_id = 1;
        for (int s = 0; s < steps; ++s) {
            std::vector<int32_t> op(batch);
            std::vector<uint32_t> a0(batch), a1(batch), a2(batch), a3(batch), a4(batch),
                a5(batch), a6(batch), a7(batch), tiles((size_t)batch * 16, 0);
            std::vector<uint64_t> a8(batch);
            for (int i = 0; i < batch; ++i) {
                int r = rng.uni(0, 99);
                if (r < 30) {
                    op[i] = MKPS_OP_ENQUEUE;
                    a0[i] = next_id++; a1[i] = (uint32_t)rng.uni(0, 4); a2[i] = (uint32_t)rng.uni(0, 4);
                    a3[i] = (uint32_t)rng.uni(0, 2); a4[i] = (uint32_t)rng.uni(1, 6);
                    a5[i] = (uint32_t)rng.uni(1, 5); a6[i] = (uint32_t)rng.uni(1, 5);
                    a7[i] = (uint32_t)rng.uni(0, spec.counter_count - 1); a8[i] = rng.next_u64();
                    for (uint32_t k = 0; k < a1[i] && k < 8; ++k) tiles[i * 16 + k] = (uint32_t)rng.uni(0, spec.tile_count - 1);
                    for (uint32_t k = 0; k < a2[i] && k < 8; ++k) tiles[i * 16 + 8 + k] = (uint32_t)rng.uni(0, spec.tile_count - 1);
                } else if (r < 45) { op[i] = MKPS_OP_ISSUE_LOADS; a0[i] = (uint32_t)rng.uni(0, 4); }
                else if (r < 58) { op[i] = MKPS_OP_ISSUE_COMPUTE; a0[i] = (uint32_t)rng.uni(0, 4); }
                else if (r < 70) { op[i] = MKPS_OP_ISSUE_STORES; a0[i] = (uint32_t)rng.uni(0, 4); }
                else if (r < 92) { op[i] = MKPS_OP_ADVANCE; a0[i] = (uint32_t)rng.uni(0, 7); a1[i] = (uint32_t)rng.uni(0, 8); }
                else { op[i] = MKPS_OP_HOST_COUNTER; a0[i] = (uint32_t)rng.uni(0, spec.counter_count - 1); a1[i] = (uint32_t)rng.uni(1, 9); }
            }
            d_op[s].alloc(batch); d_op[s].up(op);
            d_a0[s].alloc(batch); d_a0[s].up(a0); d_a1[s].alloc(batch); d_a1[s].up(a1);
            d_a2[s].alloc(batch); d_a2[s].up(a2); d_a3[s].alloc(batch); d_a3[s].up(a3);
            d_a4[s].alloc(batch); d_a4[s].up(a4); d_a5[s].alloc(batch); d_a5[s].up(a5);
            d_a6[s].alloc(batch); d_a6[s].up(a6); d_a7[s].alloc(batch); d_a7[s].up(a7);
            d_a8[s].alloc(batch); d_a8[s].up(a8);
            d_tiles[s].alloc((size_t)batch * 16); d_tiles[s].up(tiles);
        }

        DBuf<int64_t> d_counts; d_counts.alloc(MKPS_COUNT_FIELDS);
        DBuf<uint64_t> d_ev, d_ih, d_th, d_bh, d_ph, d_ch, d_evseq, d_state;
        d_ev.alloc(1); d_ih.alloc(1); d_th.alloc(1); d_bh.alloc(1); d_ph.alloc(1);
        d_ch.alloc(1); d_evseq.alloc(1); d_state.alloc(1);

        MkpsOutputs out = {};
        out.counts = d_counts.ptr; out.pipeline_event_hash = d_ev.ptr; out.instr_hash = d_ih.ptr;
        out.tile_hash = d_th.ptr; out.buffer_hash = d_bh.ptr; out.pending_hash = d_ph.ptr;
        out.counter_hash = d_ch.ptr; out.event_seq_out = d_evseq.ptr; out.state_checksum = d_state.ptr;

        auto bind = [&](int s, MkpsInputs& in) {
            in.op = d_op[s].ptr; in.a0 = d_a0[s].ptr; in.a1 = d_a1[s].ptr; in.a2 = d_a2[s].ptr;
            in.a3 = d_a3[s].ptr; in.a4 = d_a4[s].ptr; in.a5 = d_a5[s].ptr; in.a6 = d_a6[s].ptr;
            in.a7 = d_a7[s].ptr; in.a8 = d_a8[s].ptr; in.tiles = d_tiles[s].ptr;
        };

        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));

        for (int s = 0; s < steps; ++s) {
            MkpsRunSpec run = {}; run.abi_version = MKPS_ABI_VERSION; run.batch_size = batch; run.step_id = s;
            MkpsInputs in = {}; bind(s, in);
            CUDA_CHECK(solution_run(state, &run, &in, &out, ws.ptr, std::max<size_t>(wb, 1), stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(solution_reset(state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaEventRecord(e0, stream));
        for (int s = 0; s < steps; ++s) {
            MkpsRunSpec run = {}; run.abi_version = MKPS_ABI_VERSION; run.batch_size = batch; run.step_id = s;
            MkpsInputs in = {}; bind(s, in);
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
