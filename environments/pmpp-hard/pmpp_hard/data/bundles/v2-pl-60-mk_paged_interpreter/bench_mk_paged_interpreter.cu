// file: bench_mk_paged_interpreter.cu
//
// Throughput micro-bench for MK1. Builds one moderately sized program and a
// long mixed op stream, then times solution_run over many iterations. All
// outputs are ignored for timing; correctness is covered by the test harness.

#include "mk_paged_interpreter_common.h"

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
                 << cudaGetErrorString(_err);                                   \
            throw std::runtime_error(_oss.str());                               \
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
    uint64_t uniform_u64(uint64_t lo, uint64_t hi) { return lo + (next_u64() % (hi - lo + 1)); }
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr; size_t count = 0;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) { count = n; if (n) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n)); }
};

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 10;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        const int SM = 8, PAGES = 8, NCTR = 8, MAXP = 1024, MAXPROG = 8;
        SplitMix64 rng(0x1234abcd5678ULL);

        std::vector<int32_t> program_len(SM, 0), instr_offset(SM, 0);
        std::vector<MkInstr> instrs;
        std::vector<MkPageReq> reqs;
        std::vector<MkWaitRec> waits;
        uint64_t iid = 1;
        for (int sm = 0; sm < SM; ++sm) {
            instr_offset[sm] = (int32_t)instrs.size();
            int n = rng.uniform_int(3, MAXPROG);
            program_len[sm] = n;
            for (int k = 0; k < n; ++k) {
                MkInstr in = {};
                in.instr_id = iid++;
                in.req_offset = (uint32_t)reqs.size();
                in.wait_offset = (uint32_t)waits.size();
                int nreq = rng.uniform_int(0, 3);
                for (int q = 0; q < nreq; ++q) {
                    MkPageReq r = {}; r.tile_id = rng.uniform_u64(1, 8); r.mode = (uint8_t)rng.uniform_int(0, 2);
                    r.release_after_store = (uint8_t)rng.uniform_int(0, 1); reqs.push_back(r);
                }
                int nw = rng.uniform_int(0, 2);
                for (int q = 0; q < nw; ++q) { MkWaitRec w = {}; w.counter_id = (uint32_t)rng.uniform_int(0, NCTR - 1); w.target = rng.uniform_u64(1, 4); waits.push_back(w); }
                in.page_req_count = (uint32_t)nreq; in.wait_count = (uint32_t)nw;
                in.load_latency = rng.uniform_u64(0, 4); in.compute_latency = rng.uniform_u64(0, 4); in.store_latency = rng.uniform_u64(0, 4);
                in.result_seed = rng.next_u64();
                in.out_counter = rng.uniform_int(0, 1) ? (uint32_t)rng.uniform_int(0, NCTR - 1) : UINT32_MAX;
                in.out_increment = rng.uniform_u64(1, 3);
                instrs.push_back(in);
            }
        }

        MkProblemSpec spec = {};
        spec.abi_version = MK_ABI_VERSION; spec.sm_count = SM; spec.pages_per_sm = PAGES;
        spec.counter_count = NCTR; spec.max_pending_events = MAXP; spec.max_program_len_per_sm = MAXPROG;
        spec.total_instr = (int32_t)instrs.size(); spec.total_reqs = (int32_t)reqs.size(); spec.total_waits = (int32_t)waits.size();
        spec.program_len = program_len.data(); spec.instr_offset = instr_offset.data();
        spec.instrs = instrs.data(); spec.reqs = reqs.empty() ? nullptr : reqs.data(); spec.waits = waits.empty() ? nullptr : waits.data();
        if (!mk_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

        // op stream
        std::vector<MkRunSpec> ops;
        auto mk = [&](int kind) { MkRunSpec r = {}; r.abi_version = MK_ABI_VERSION; r.op_kind = kind; return r; };
        { MkRunSpec r = mk(MK_OP_BEGIN_PASS); r.a_pass_id = 1; ops.push_back(r); }
        for (int i = 0; i < 2000; ++i) {
            int p = rng.uniform_int(0, 99);
            if (p < 55) { MkRunSpec r = mk(MK_OP_STEP_SM); r.a_sm = rng.uniform_int(0, SM - 1); r.a_transition_limit = rng.uniform_int(1, 4); ops.push_back(r); }
            else if (p < 82) { MkRunSpec r = mk(MK_OP_ADVANCE); r.a_delta = rng.uniform_u64(0, 6); r.a_max_events = rng.uniform_int(0, 8); ops.push_back(r); }
            else { MkRunSpec r = mk(MK_OP_HOST_INC_COUNTER); r.a_counter = rng.uniform_int(0, NCTR - 1); r.a_amount = rng.uniform_u64(1, 4); ops.push_back(r); }
        }

        const size_t wb = solution_workspace_bytes(&spec);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace; workspace.allocate(std::max<size_t>(wb, 1));
        DeviceBuffer<int64_t> d_counts; d_counts.allocate(MK_COUNT_N);
        DeviceBuffer<int32_t> d_opidx; d_opidx.allocate(1);
        DeviceBuffer<uint64_t> d_clock, d_eseq, d_eh, d_ph, d_sh, d_ch, d_pend, d_state;
        d_clock.allocate(1); d_eseq.allocate(1); d_eh.allocate(1); d_ph.allocate(1);
        d_sh.allocate(1); d_ch.allocate(1); d_pend.allocate(1); d_state.allocate(1);

        MkOutputs out = {};
        out.counts = d_counts.ptr; out.op_index_out = d_opidx.ptr; out.clock_out = d_clock.ptr;
        out.event_seq_out = d_eseq.ptr; out.event_hash = d_eh.ptr; out.page_hash = d_ph.ptr;
        out.sm_hash = d_sh.ptr; out.counter_hash = d_ch.ptr; out.pending_hash = d_pend.ptr; out.state_checksum = d_state.ptr;

        MkInputs in = {}; in.reserved = nullptr;
        std::printf("bench SM=%d ops=%zu instrs=%zu\n", SM, ops.size(), instrs.size());

        for (int warm = 0; warm < 2; ++warm) {
            CUDA_CHECK(solution_reset(state, stream));
            for (const MkRunSpec& op : ops) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, std::max<size_t>(wb, 1), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        double total_ms = 0;
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(start, stream));
            for (const MkRunSpec& op : ops) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, std::max<size_t>(wb, 1), stream));
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
