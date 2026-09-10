// file: bench_incremental_gc_weakref.cu
//
// Minimal throughput bench: drives a long mixed op stream through the
// solution under test and reports ops/sec. Not graded; correctness lives in
// the test harness.

#include "incremental_gc_weakref_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(expr)                                                        \
    do { cudaError_t _e=(expr); if(_e!=cudaSuccess){ std::ostringstream o;     \
        o<<"CUDA error "<<__FILE__<<":"<<__LINE__<<" "<<cudaGetErrorString(_e); \
        throw std::runtime_error(o.str()); } } while(0)

static IgcwRunSpec mk(int op,int idx,int a0=0,int a1=0,int a2=0,int a3=0,int64_t sz=0){
    IgcwRunSpec r; std::memset(&r,0,sizeof(r));
    r.abi_version=IGCW_ABI_VERSION; r.opcode=op; r.op_index=idx;
    r.a0=a0;r.a1=a1;r.a2=a2;r.a3=a3;r.size_arg=sz; return r;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        IgcwProblemSpec spec; std::memset(&spec,0,sizeof(spec));
        spec.abi_version=IGCW_ABI_VERSION; spec.max_objects=256; spec.root_count=16;
        spec.strong_slots_per_object=4; spec.weak_slots_per_object=2;
        spec.max_ephemerons=64; spec.max_mark_queue=1024; spec.max_finalizer_queue=128;
        spec.young_survive_threshold=2;

        std::vector<IgcwRunSpec> ops;
        int idx=0;
        for (int round=0; round<50; ++round) {
            for (int i=0;i<60;++i) ops.push_back(mk(IGCW_OP_ALLOC,idx++,IGCW_GEN_YOUNG,0,(int)0xFFFFFFFF,1+i));
            for (int r=0;r<16;++r) ops.push_back(mk(IGCW_OP_SET_ROOT,idx++,r,1+r));
            for (int s=0;s<40;++s) ops.push_back(mk(IGCW_OP_SET_STRONG,idx++,1+(s%50),s%4,1+((s*3)%60)));
            ops.push_back(mk(IGCW_OP_START_FULL,idx++));
            for (int s=0;s<80;++s) ops.push_back(mk(IGCW_OP_GC_STEP,idx++,4,4));
            ops.push_back(mk(IGCW_OP_RUN_FINALIZERS,idx++,16));
        }

        size_t wb = solution_workspace_bytes(&spec); if (wb==0) wb=1;
        cudaStream_t stream=nullptr; CUDA_CHECK(cudaStreamCreate(&stream));
        void* state=nullptr; CUDA_CHECK(solution_init(&spec,&state,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        uint8_t* ws=nullptr; CUDA_CHECK(cudaMalloc((void**)&ws, wb));

        int32_t* d_counts; uint64_t* d_h[6]; int32_t* d_inv;
        CUDA_CHECK(cudaMalloc((void**)&d_counts,sizeof(int32_t)*IGCW_NUM_COUNTS));
        for(int i=0;i<6;++i) CUDA_CHECK(cudaMalloc((void**)&d_h[i],8));
        CUDA_CHECK(cudaMalloc((void**)&d_inv,4));

        IgcwOutputs out; std::memset(&out,0,sizeof(out));
        out.counts=d_counts; out.gc_event_hash=d_h[0]; out.heap_hash=d_h[1]; out.root_hash=d_h[2];
        out.ephemeron_hash=d_h[3]; out.remembered_hash=d_h[4]; out.gc_controller_hash=d_h[5]; out.invalid_flag=d_inv;

        cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s,stream));
        for (auto& op : ops) {
            CUDA_CHECK(solution_run(state,&op,nullptr,&out,ws,wb,stream));
        }
        CUDA_CHECK(cudaEventRecord(e,stream));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
        std::printf("bench: %zu ops in %.2f ms (%.1f ops/s)\n", ops.size(), ms, ops.size()/(ms/1000.0));

        solution_destroy(state);
        cudaFree(ws); cudaFree(d_counts); for(int i=0;i<6;++i) cudaFree(d_h[i]); cudaFree(d_inv);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr,"fatal: %s\n", ex.what());
        return 1;
    }
}
