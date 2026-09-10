// file: bench_mk_schedule_planner.cu
//
// Throughput benchmark for MK7. Replays a fixed multi-step op sequence and
// times the steady-state cost.

#include "mk_schedule_planner_common.h"

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
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                             \
            std::ostringstream _oss;                                           \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "   \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                             \
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
    int uniform_int(int lo, int hi) { return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1)); }
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr; size_t count = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) { count = n; if (!n) { ptr = nullptr; return; } CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T)*n)); }
    void upload(const std::vector<T>& h) { if (h.size()!=count) throw std::runtime_error("size"); if (count) CUDA_CHECK(cudaMemcpy(ptr,h.data(),sizeof(T)*count,cudaMemcpyHostToDevice)); }
};

static MkOp mk_add_instr(uint64_t id, uint64_t dur, uint64_t pcnt, uint64_t k0, uint64_t k1, uint64_t rdelay, uint64_t seed) {
    MkOp o={}; o.op_type=MK_OP_ADD_INSTR; o.id=id; o.a=dur; o.b=pcnt; o.c=rdelay; o.d=seed;
    o.page_keys[0]=k0; if (pcnt>1) o.page_keys[1]=k1; return o;
}
static MkOp mk_add_edge(uint64_t eid, uint64_t s, uint64_t d, uint32_t c, uint64_t t) {
    MkOp o={}; o.op_type=MK_OP_ADD_EDGE; o.id=eid; o.src=s; o.dst=d; o.a=c; o.b=t; return o;
}
static MkOp mk_plan(uint64_t l){ MkOp o={}; o.op_type=MK_OP_PLAN_NEXT; o.a=l; return o; }
static MkOp mk_commit(uint64_t m){ MkOp o={}; o.op_type=MK_OP_COMMIT_PLAN; o.a=m; return o; }
static MkOp mk_exec(uint64_t t, uint64_t m){ MkOp o={}; o.op_type=MK_OP_EXECUTE_UNTIL; o.a=t; o.b=m; return o; }

struct StepHost { MkRunSpec run; std::vector<MkOp> ops; };

struct DeviceStep {
    MkRunSpec run;
    DeviceBuffer<MkOp> ops;
    DeviceBuffer<uint64_t> scalars; // 20
    MkInputs inputs; MkOutputs out;
};

static MkProblemSpec make_spec() {
    MkProblemSpec s = {};
    s.abi_version = MK_ABI_VERSION;
    s.sm_count = 8; s.pages_per_sm = 4; s.wave_quantum = 5;
    s.max_instrs = 512; s.max_edges = 1024;
    s.max_ops_per_step = 128; s.max_steps = 48; s.flags = 0;
    if (!mk_validate_problem_spec(&s)) throw std::runtime_error("bad spec");
    return s;
}

static std::vector<StepHost> build_seq(const MkProblemSpec& spec) {
    SplitMix64 rng(0x55aa1234ULL);
    std::vector<StepHost> steps;
    uint64_t next_instr = 1, next_edge = 1;
    std::vector<uint64_t> live;
    for (int s = 0; s < 48; ++s) {
        StepHost st; st.run = {}; st.run.abi_version = MK_ABI_VERSION; st.run.step_id = s;
        std::vector<MkOp> ops;
        int n = rng.uniform_int(20, 80);
        for (int i = 0; i < n; ++i) {
            int r = rng.uniform_int(0, 99);
            if (r < 45 && (int)live.size() < spec.max_instrs - 4) {
                int pc = rng.uniform_int(1, 2);
                ops.push_back(mk_add_instr(next_instr, rng.uniform_int(1,12), pc,
                                           rng.uniform_int(1,8), rng.uniform_int(1,8),
                                           rng.uniform_int(0,5), rng.next_u64()));
                live.push_back(next_instr++);
            } else if (r < 60 && live.size() >= 2) {
                uint64_t a = live[rng.uniform_int(0,(int)live.size()-1)];
                uint64_t b = live[rng.uniform_int(0,(int)live.size()-1)];
                ops.push_back(mk_add_edge(next_edge++, a, b, rng.uniform_int(0,7), rng.uniform_int(1,4)));
            } else if (r < 80) ops.push_back(mk_plan(rng.uniform_int(0,8)));
            else if (r < 90) ops.push_back(mk_commit(rng.uniform_int(0,6)));
            else ops.push_back(mk_exec(rng.uniform_int(0,300), rng.uniform_int(0,8)));
        }
        st.run.num_ops = (int)ops.size();
        st.ops = ops;
        if (st.ops.empty()) st.ops.resize(1);
        steps.push_back(st);
    }
    return steps;
}

static DeviceStep* make_device_step(const StepHost& h) {
    DeviceStep* ds = new DeviceStep();
    ds->run = h.run;
    ds->ops.allocate(h.ops.size());
    ds->ops.upload(h.ops);
    ds->scalars.allocate(20);
    ds->inputs = {}; ds->inputs.ops = ds->ops.ptr;
    uint64_t* p = ds->scalars.ptr;
    MkOutputs o = {};
    o.instr_added=p+0; o.edge_added=p+1; o.plan_invalidated=p+2; o.plan_empty=p+3;
    o.plan_stall=p+4; o.plan_placed=p+5; o.plan_committed=p+6; o.instr_executed=p+7;
    o.edge_signaled=p+8; o.instr_cancelled=p+9; o.epoch_advanced=p+10; o.invalid_count=p+11;
    o.planner_event_hash=p+12; o.plan_hash=p+13; o.instr_hash=p+14; o.edge_hash=p+15;
    o.page_interval_hash=p+16; o.committed_epoch=p+17; o.live_instr_count=p+18;
    o.planned_interval_count=p+19;
    ds->out = o;
    return ds;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = (argc >= 2) ? std::max(1, atoi(argv[1])) : 20;
        MkProblemSpec spec = make_spec();
        size_t wb = solution_workspace_bytes(&spec);
        if (wb == 0) throw std::runtime_error("workspace_bytes 0");

        cudaStream_t stream = nullptr; CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr; CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        DeviceBuffer<uint8_t> workspace; workspace.allocate(std::max<size_t>(wb,1));

        std::vector<StepHost> hs = build_seq(spec);
        std::vector<DeviceStep*> steps;
        for (auto& h : hs) steps.push_back(make_device_step(h));

        std::printf("bench sm=%d pages=%d wq=%llu T=%zu\n",
                    spec.sm_count, spec.pages_per_sm, (unsigned long long)spec.wave_quantum, steps.size());

        for (int w = 0; w < 3; ++w) {
            CUDA_CHECK(solution_reset(state, stream));
            for (auto* ds : steps) CUDA_CHECK(solution_run(state,&ds->run,&ds->inputs,&ds->out,workspace.ptr,std::max<size_t>(wb,1),stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        double total_ms = 0;
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(solution_reset(state, stream)); CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(a, stream));
            for (auto* ds : steps) CUDA_CHECK(solution_run(state,&ds->run,&ds->inputs,&ds->out,workspace.ptr,std::max<size_t>(wb,1),stream));
            CUDA_CHECK(cudaEventRecord(b, stream)); CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); total_ms += ms;
        }
        std::printf("avg_ms=%.6f\n", total_ms / iters);

        CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
        for (auto* ds : steps) delete ds;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state); CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
