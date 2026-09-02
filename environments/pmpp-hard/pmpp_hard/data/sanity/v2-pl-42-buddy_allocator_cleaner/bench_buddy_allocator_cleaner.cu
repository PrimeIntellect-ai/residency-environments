// file: bench_buddy_allocator_cleaner.cu
//
// Throughput benchmark for T42. Replays a fixed multi-step op sequence and
// times the steady-state cost.

#include "buddy_allocator_cleaner_common.h"

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
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) { count = n; if (!n) { ptr = nullptr; return; } CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T)*n)); }
    void upload(const std::vector<T>& h) { if (h.size()!=count) throw std::runtime_error("size"); if (count) CUDA_CHECK(cudaMemcpy(ptr,h.data(),sizeof(T)*count,cudaMemcpyHostToDevice)); }
};

static BacOp mk_alloc(uint64_t o,int c,uint64_t r){BacOp x={};x.op_type=BAC_OP_ALLOC;x.class_id=c;x.obj_id=o;x.a=r;return x;}
static BacOp mk_free(uint64_t o){BacOp x={};x.op_type=BAC_OP_FREE;x.obj_id=o;return x;}
static BacOp mk_clean(uint64_t m,uint64_t b){BacOp x={};x.op_type=BAC_OP_CLEAN;x.a=m;x.b=b;return x;}
static BacOp mk_seal(int c){BacOp x={};x.op_type=BAC_OP_SEAL;x.class_id=c;return x;}

struct StepHost { BacRunSpec run; std::vector<BacOp> ops; };

struct DeviceStep {
    BacRunSpec run;
    DeviceBuffer<BacOp> ops;
    DeviceBuffer<uint64_t> scalars; // 23
    BacInputs inputs; BacOutputs out;
};

static BacProblemSpec make_spec() {
    BacProblemSpec s = {};
    s.abi_version = BAC_ABI_VERSION;
    s.max_order = 12; s.segment_order = 4; s.num_classes = 4;
    s.max_objects = 1024; s.max_segments = 256;
    s.max_ops_per_step = 128; s.max_steps = 48; s.flags = 0;
    if (!bac_validate_problem_spec(&s)) throw std::runtime_error("bad spec");
    return s;
}

static std::vector<StepHost> build_seq(const BacProblemSpec& spec) {
    SplitMix64 rng(0x1234abcdULL);
    std::vector<StepHost> steps;
    uint64_t next_obj = 1;
    std::vector<uint64_t> live;
    for (int s = 0; s < 48; ++s) {
        StepHost st; st.run = {}; st.run.abi_version = BAC_ABI_VERSION; st.run.step_id = s;
        std::vector<BacOp> ops;
        int n = rng.uniform_int(20, 100);
        for (int i = 0; i < n; ++i) {
            int r = rng.uniform_int(0, 99);
            if (r < 55) { ops.push_back(mk_alloc(next_obj, rng.uniform_int(0,3), rng.uniform_int(1,16))); live.push_back(next_obj++); }
            else if (r < 75 && !live.empty()) ops.push_back(mk_free(live[rng.uniform_int(0,(int)live.size()-1)]));
            else if (r < 88) ops.push_back(mk_seal(rng.uniform_int(0,3)));
            else ops.push_back(mk_clean(rng.uniform_int(0,8), rng.uniform_int(0,64)));
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
    ds->scalars.allocate(23);
    ds->inputs = {}; ds->inputs.ops = ds->ops.ptr;
    uint64_t* p = ds->scalars.ptr;
    BacOutputs o = {};
    o.alloc_ok=p+0; o.alloc_oom=p+1; o.free_finalized=p+2; o.free_deferred=p+3;
    o.pin_ok=p+4; o.unpin_ok=p+5; o.seal_explicit=p+6; o.seal_implicit=p+7; o.seal_empty=p+8;
    o.relocated_objects=p+9; o.clean_blocked_segments=p+10; o.segments_reclaimed=p+11;
    o.buddy_splits=p+12; o.buddy_merges=p+13; o.padding_pages_added=p+14; o.invalid_count=p+15;
    o.alloc_event_hash=p+16; o.finalize_hash=p+17; o.buddy_hash=p+18; o.segment_hash=p+19;
    o.object_hash=p+20; o.live_object_count=p+21; o.live_segment_count=p+22;
    ds->out = o;
    return ds;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = (argc >= 2) ? std::max(1, atoi(argv[1])) : 20;
        BacProblemSpec spec = make_spec();
        size_t wb = solution_workspace_bytes(&spec);
        if (wb == 0) throw std::runtime_error("workspace_bytes 0");

        cudaStream_t stream = nullptr; CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr; CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        DeviceBuffer<uint8_t> workspace; workspace.allocate(std::max<size_t>(wb,1));

        std::vector<StepHost> hs = build_seq(spec);
        std::vector<DeviceStep*> steps;
        for (auto& h : hs) steps.push_back(make_device_step(h));

        std::printf("bench max_order=%d seg_order=%d classes=%d T=%zu\n",
                    spec.max_order, spec.segment_order, spec.num_classes, steps.size());

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
