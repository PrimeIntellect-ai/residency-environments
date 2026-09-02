// file: bench_mk_instr_decoder_hazard.cu
//
// Throughput micro-benchmark for MK10: builds one large random stream, drives a
// long mixed op trace, and times the per-op solution_run dispatch.

#include "mk_instr_decoder_hazard_common.h"

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
    uint64_t state; explicit SplitMix64(uint64_t s):state(s){}
    uint64_t next(){ uint64_t z=(state+=0x9e3779b97f4a7c15ULL); z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31); }
    int ui(int lo,int hi){ return lo+(int)(next()%(uint64_t)(hi-lo+1)); }
};

template <typename T> struct DeviceBuffer {
    T* ptr=nullptr; size_t count=0; ~DeviceBuffer(){ if(ptr) cudaFree(ptr); }
    void allocate(size_t n){ count=n; if(n) CUDA_CHECK(cudaMalloc((void**)&ptr,sizeof(T)*n)); }
};

static MkInstr mk_nop(uint64_t uid,uint64_t lat){ MkInstr i={}; i.opcode=MK_NOP; i.instr_uid=uid; i.latency=lat; i.predicate_mask=~0ULL; i.dst_tile=-1; i.st_read_tile=-1; i.free_tile=-1; i.counter_id=-1; return i; }

int main(int argc,char** argv){
    try{
        CUDA_CHECK(cudaSetDevice(0));
        int iters=10; if(argc>=2) iters=std::max(1,std::atoi(argv[1]));

        MkProblemSpec spec={}; spec.abi_version=MK_ABI_VERSION;
        spec.sm_count=4; spec.stream_len_per_sm=64; spec.decode_width=4; spec.issue_width=4; spec.retire_width=4;
        spec.tile_count=32; spec.page_count_per_sm=8; spec.counter_count=8;
        spec.max_decode_queue=16; spec.max_issue_queue=16; spec.max_pending_units=512; spec.max_replay=16; spec.max_decode_records=2048;
        if(!mk_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

        const int SM=spec.sm_count, SL=spec.stream_len_per_sm;
        SplitMix64 rng(0xB0BACAFEULL);
        std::vector<MkInstr> stream((size_t)SM*SL, mk_nop(0,1));
        uint64_t uid=1;
        for(int sm=0;sm<SM;++sm) for(int pc=0;pc<SL;++pc){
            MkInstr in=mk_nop(uid++, (uint64_t)rng.ui(1,4));
            int k=rng.ui(0,9);
            if(k<=2){ in.opcode=MK_LD; in.dst_tile=rng.ui(0,31); in.page_key=rng.ui(1,9); }
            else if(k<=4){ in.opcode=MK_ALU; in.n_read=1; in.read_tiles[0]=rng.ui(0,31); in.n_write=1; in.write_tiles[0]=rng.ui(0,31); in.compute_seed=rng.ui(1,99); }
            else if(k==5){ in.opcode=MK_ST; in.st_read_tile=rng.ui(0,31); }
            else if(k==6){ in.opcode=MK_INC; in.counter_id=rng.ui(0,7); in.amount=rng.ui(1,5); }
            else if(k==7){ in.opcode=MK_FREE; in.free_tile=rng.ui(0,31); }
            stream[(size_t)sm*SL+pc]=in;
        }

        const size_t wb=solution_workspace_bytes(&spec);
        if(wb==0) throw std::runtime_error("workspace 0");

        cudaStream_t st=nullptr; CUDA_CHECK(cudaStreamCreate(&st));
        void* state=nullptr; CUDA_CHECK(solution_init(&spec,&state,st)); CUDA_CHECK(cudaStreamSynchronize(st));

        DeviceBuffer<uint8_t> ws; ws.allocate(wb);
        DeviceBuffer<MkInstr> d_stream; d_stream.allocate(stream.size());
        CUDA_CHECK(cudaMemcpy(d_stream.ptr,stream.data(),sizeof(MkInstr)*stream.size(),cudaMemcpyHostToDevice));

        DeviceBuffer<int64_t> d_counts; d_counts.allocate(MK_COUNT_N);
        DeviceBuffer<int32_t> d_opidx; d_opidx.allocate(1);
        DeviceBuffer<uint64_t> a,b,cc,dd,ee,ff,gg,hh,ii,jj; a.allocate(1);b.allocate(1);cc.allocate(1);dd.allocate(1);ee.allocate(1);ff.allocate(1);gg.allocate(1);hh.allocate(1);ii.allocate(1);jj.allocate(1);

        MkOutputs out={}; out.counts=d_counts.ptr; out.op_index_out=d_opidx.ptr; out.clock_out=a.ptr; out.event_seq_out=b.ptr;
        out.decoder_event_hash=cc.ptr; out.sm_stream_hash=dd.ptr; out.decoded_hash=ee.ptr; out.tile_scoreboard_hash=ff.ptr;
        out.page_hash=gg.ptr; out.pending_unit_hash=hh.ptr; out.counter_hash=ii.ptr; out.state_checksum=jj.ptr;

        MkInputs in={}; in.stream=d_stream.ptr; in.stream_count=(int)stream.size();

        // build op trace
        std::vector<MkRunSpec> ops;
        auto rk=[&](int kind){ MkRunSpec r={}; r.abi_version=MK_ABI_VERSION; r.op_kind=kind; return r; };
        { MkRunSpec r=rk(MK_OP_BEGIN_KERNEL); r.a_epoch_seed=0x1234; ops.push_back(r); }
        for(int round=0;round<60;++round){
            for(int sm=0;sm<SM;++sm){ MkRunSpec d=rk(MK_OP_DECODE_SM); d.a_sm=sm; d.a_limit=4; ops.push_back(d); MkRunSpec is=rk(MK_OP_ISSUE_SM); is.a_sm=sm; is.a_limit=4; ops.push_back(is); }
            { MkRunSpec ad=rk(MK_OP_ADVANCE); ad.a_delta=rng.ui(1,6); ad.a_max_units=8; ops.push_back(ad); }
            for(int sm=0;sm<SM;++sm){ MkRunSpec re=rk(MK_OP_RETIRE_SM); re.a_sm=sm; re.a_limit=4; ops.push_back(re); }
        }

        std::printf("bench SM=%d stream=%zu ops=%zu\n",SM,stream.size(),ops.size());

        for(int warm=0;warm<2;++warm){ CUDA_CHECK(solution_reset(state,st)); for(const MkRunSpec& op:ops) CUDA_CHECK(solution_run(state,&op,&in,&out,ws.ptr,wb,st)); CUDA_CHECK(cudaStreamSynchronize(st)); }

        cudaEvent_t s0,s1; CUDA_CHECK(cudaEventCreate(&s0)); CUDA_CHECK(cudaEventCreate(&s1));
        double total=0;
        for(int it=0;it<iters;++it){
            CUDA_CHECK(solution_reset(state,st)); CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaEventRecord(s0,st));
            for(const MkRunSpec& op:ops) CUDA_CHECK(solution_run(state,&op,&in,&out,ws.ptr,wb,st));
            CUDA_CHECK(cudaEventRecord(s1,st)); CUDA_CHECK(cudaEventSynchronize(s1));
            float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s0,s1)); total+=ms;
        }
        std::printf("avg_ms=%.6f\n",total/iters);

        CUDA_CHECK(cudaEventDestroy(s0)); CUDA_CHECK(cudaEventDestroy(s1));
        solution_destroy(state); CUDA_CHECK(cudaStreamDestroy(st));
        return 0;
    } catch(const std::exception& ex){ std::fprintf(stderr,"fatal: %s\n",ex.what()); return 1; }
}
