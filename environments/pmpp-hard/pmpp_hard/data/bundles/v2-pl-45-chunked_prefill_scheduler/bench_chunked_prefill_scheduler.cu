// file: bench_chunked_prefill_scheduler.cu
//
// Throughput micro-bench for T45. Replays a long random op stream and reports
// ops/sec. Not a correctness check (see test_*.cu for that).

#include "chunked_prefill_scheduler_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <chrono>
#include <cstdio>
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

struct SM64 {
    uint64_t s;
    explicit SM64(uint64_t x) : s(x) {}
    uint64_t n() { uint64_t z=(s+=0x9e3779b97f4a7c15ULL); z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31); }
    uint64_t r(uint64_t lo, uint64_t hi){ return lo + n()%(hi-lo+1); }
};

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        CpsProblemSpec spec{};
        spec.abi_version = CPS_ABI_VERSION;
        spec.num_tenants = 4; spec.kv_capacity_tokens = 64; spec.max_live_requests = 64;
        spec.default_chunk_max = 8; spec.max_batch_slots = 16; spec.num_experts = 8;
        spec.max_ops = 256; spec.moe_seed = 0xBEEF1234ULL;
        for (int t=0;t<4;++t){ spec.bucket_cap[t]=1000; spec.initial_bucket_tokens[t]=200; }
        for (int e=0;e<8;++e) spec.expert_capacity[e]=4;

        cudaStream_t stream=nullptr; CUDA_CHECK(cudaStreamCreate(&stream));
        void* state=nullptr; CUDA_CHECK(solution_init(&spec,&state,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        size_t wbytes = solution_workspace_bytes(&spec);
        uint8_t* ws=nullptr; if (wbytes) CUDA_CHECK(cudaMalloc((void**)&ws, wbytes));

        CpsCounts* d_counts; uint64_t* d_h[7];
        CUDA_CHECK(cudaMalloc((void**)&d_counts, sizeof(CpsCounts)));
        for (int i=0;i<7;++i) CUDA_CHECK(cudaMalloc((void**)&d_h[i], sizeof(uint64_t)));
        CpsOutputs out{}; out.counts=d_counts; out.batch_hash=d_h[0]; out.moe_hash=d_h[1];
        out.finalize_hash=d_h[2]; out.queue_hash=d_h[3]; out.request_hash=d_h[4];
        out.bucket_hash=d_h[5]; out.scalar_hash=d_h[6];

        SM64 rng(0xC0FFEEULL);
        std::vector<CpsOp> ops;
        uint64_t rid=1;
        for (int i=0;i<256;++i){
            CpsOp o{}; o.abi_version=CPS_ABI_VERSION; o.op_index=i;
            int roll=(int)(rng.n()%100);
            if (roll<30){ o.opcode=CPS_OP_ARRIVE; o.request_id=rid++; o.tenant=(uint32_t)rng.r(0,3); o.priority=(uint32_t)rng.r(0,255); o.prompt_len=rng.r(1,20); o.decode_len=rng.r(0,10); o.chunk_max=rng.r(1,8); }
            else if (roll<45){ o.opcode=CPS_OP_REFILL_TENANT; o.tenant=(uint32_t)rng.r(0,3); o.a=rng.r(1,50); }
            else if (roll<52){ o.opcode=CPS_OP_SET_KV_CAP; o.a=rng.r(0,80); }
            else if (roll<60){ o.opcode=CPS_OP_CANCEL; o.request_id=rng.r(1,rid); }
            else { o.opcode=CPS_OP_STEP_ITER; o.a=rng.r(0,40); o.b=rng.r(0,16); }
            ops.push_back(o);
        }

        const int reps=200;
        auto t0=std::chrono::high_resolution_clock::now();
        for (int r=0;r<reps;++r){
            CUDA_CHECK(solution_reset(state,stream));
            for (const CpsOp& o : ops){
                CUDA_CHECK(solution_run(state,&o,nullptr,&out,ws,wbytes,stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        auto t1=std::chrono::high_resolution_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        double total_ops=(double)reps*ops.size();
        std::printf("bench: %d reps x %zu ops = %.0f ops in %.3f s -> %.1f kops/s\n",
                    reps, ops.size(), total_ops, sec, total_ops/sec/1000.0);

        cudaFree(d_counts); for(int i=0;i<7;++i) cudaFree(d_h[i]); if(ws) cudaFree(ws);
        solution_destroy(state); CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex){ std::fprintf(stderr,"fatal: %s\n",ex.what()); return 1; }
}
