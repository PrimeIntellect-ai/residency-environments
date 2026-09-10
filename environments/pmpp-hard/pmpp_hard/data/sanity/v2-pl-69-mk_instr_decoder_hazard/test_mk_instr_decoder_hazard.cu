// file: test_mk_instr_decoder_hazard.cu
//
// 3-way validation harness for MK10.  Drives the device solution_run against
// the host golden oracle across >=6 deeply-adversarial multi-step scenarios,
// with guard buffers, input/run-spec immutability checks, and exact replay.

#include "mk_instr_decoder_hazard_common.h"
#include "mk_instr_decoder_hazard_oracle.hpp"

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

static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

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
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) { count = n; if (n > 0) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n)); }
};

template <typename T>
struct GuardedDeviceBuffer {
    uint8_t* raw = nullptr; T* ptr = nullptr; size_t count = 0, data_bytes = 0;
    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;
    ~GuardedDeviceBuffer() { if (raw) cudaFree(raw); }
    void allocate(size_t n) {
        count = n; data_bytes = sizeof(T) * count;
        CUDA_CHECK(cudaMalloc((void**)&raw, kGuardBytes + data_bytes + kGuardBytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, kGuardBytes + data_bytes + kGuardBytes));
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> left(kGuardBytes), right(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(left.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(right.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (left[i] != kGuardByte) { if (error){std::ostringstream o;o<<"left guard corrupted for "<<name<<" at "<<i;*error=o.str();} return false; }
            if (right[i] != kGuardByte) { if (error){std::ostringstream o;o<<"right guard corrupted for "<<name<<" at "<<i;*error=o.str();} return false; }
        }
        return true;
    }
};

struct StepResult {
    std::vector<int64_t> counts;
    int32_t op_index = 0;
    uint64_t clock=0,eseq=0,evh=0,smh=0,dh=0,th=0,ph=0,puh=0,ch=0,state=0;
};

struct Scenario {
    std::string name;
    MkProblemSpec spec;
    std::vector<MkInstr> stream;     // [sm_count*stream_len]
    std::vector<MkRunSpec> ops;
};

// ----------------------------------------------------------- builders
static MkProblemSpec make_spec(int sm,int sl,int dw,int iw,int rw,int tc,int pg,int cc,int mdq,int miq,int mpu,int mrp,int mdr){
    MkProblemSpec s={}; s.abi_version=MK_ABI_VERSION;
    s.sm_count=sm; s.stream_len_per_sm=sl; s.decode_width=dw; s.issue_width=iw; s.retire_width=rw;
    s.tile_count=tc; s.page_count_per_sm=pg; s.counter_count=cc;
    s.max_decode_queue=mdq; s.max_issue_queue=miq; s.max_pending_units=mpu; s.max_replay=mrp; s.max_decode_records=mdr;
    if(!mk_validate_problem_spec(&s)) throw std::runtime_error("invalid MkProblemSpec");
    return s;
}

static MkInstr mk_nop(uint64_t uid,uint64_t lat,uint64_t pred){ MkInstr i={}; i.opcode=MK_NOP; i.instr_uid=uid; i.latency=lat; i.predicate_mask=pred; i.dst_tile=-1; i.st_read_tile=-1; i.free_tile=-1; i.counter_id=-1; return i; }
static MkInstr mk_ld(uint64_t uid,uint64_t lat,uint64_t pred,int dst,uint64_t pkey){ MkInstr i=mk_nop(uid,lat,pred); i.opcode=MK_LD; i.dst_tile=dst; i.page_key=pkey; return i; }
static MkInstr mk_alu(uint64_t uid,uint64_t lat,uint64_t pred,std::vector<int> rd,std::vector<int> wr,uint64_t seed){ MkInstr i=mk_nop(uid,lat,pred); i.opcode=MK_ALU; i.n_read=(int)rd.size(); for(size_t k=0;k<rd.size()&&k<MK_MAX_RW;++k) i.read_tiles[k]=rd[k]; i.n_write=(int)wr.size(); for(size_t k=0;k<wr.size()&&k<MK_MAX_RW;++k) i.write_tiles[k]=wr[k]; i.compute_seed=seed; return i; }
static MkInstr mk_st(uint64_t uid,uint64_t lat,uint64_t pred,int rt,uint64_t seed){ MkInstr i=mk_nop(uid,lat,pred); i.opcode=MK_ST; i.st_read_tile=rt; i.store_seed=seed; return i; }
static MkInstr mk_wait(uint64_t uid,uint64_t lat,uint64_t pred,int cid,uint64_t tgt){ MkInstr i=mk_nop(uid,lat,pred); i.opcode=MK_WAIT; i.counter_id=cid; i.target=tgt; return i; }
static MkInstr mk_inc(uint64_t uid,uint64_t lat,uint64_t pred,int cid,uint64_t amt){ MkInstr i=mk_nop(uid,lat,pred); i.opcode=MK_INC; i.counter_id=cid; i.amount=amt; return i; }
static MkInstr mk_free(uint64_t uid,uint64_t lat,uint64_t pred,int tile){ MkInstr i=mk_nop(uid,lat,pred); i.opcode=MK_FREE; i.free_tile=tile; return i; }
static MkInstr mk_branch(uint64_t uid,uint64_t lat,uint64_t pred,int cid,uint64_t tgt,int taken,int fall){ MkInstr i=mk_nop(uid,lat,pred); i.opcode=MK_BRANCH; i.counter_id=cid; i.target=tgt; i.taken_pc=taken; i.fallthrough_pc=fall; return i; }

static MkRunSpec rk(int kind,int step){ MkRunSpec r={}; r.abi_version=MK_ABI_VERSION; r.op_kind=kind; r.step_id=step; return r; }
static MkRunSpec o_begin(int step,uint64_t seed){ MkRunSpec r=rk(MK_OP_BEGIN_KERNEL,step); r.a_epoch_seed=seed; return r; }
static MkRunSpec o_decode(int step,int sm,int limit){ MkRunSpec r=rk(MK_OP_DECODE_SM,step); r.a_sm=sm; r.a_limit=limit; return r; }
static MkRunSpec o_issue(int step,int sm,int limit){ MkRunSpec r=rk(MK_OP_ISSUE_SM,step); r.a_sm=sm; r.a_limit=limit; return r; }
static MkRunSpec o_advance(int step,uint64_t delta,int mu){ MkRunSpec r=rk(MK_OP_ADVANCE,step); r.a_delta=delta; r.a_max_units=mu; return r; }
static MkRunSpec o_retire(int step,int sm,int limit){ MkRunSpec r=rk(MK_OP_RETIRE_SM,step); r.a_sm=sm; r.a_limit=limit; return r; }
static MkRunSpec o_replay(int step,int sm,int limit){ MkRunSpec r=rk(MK_OP_REPLAY_SM,step); r.a_sm=sm; r.a_limit=limit; return r; }
static MkRunSpec o_hostinc(int step,int cid,uint64_t amt){ MkRunSpec r=rk(MK_OP_HOST_INC_COUNTER,step); r.a_counter_id=cid; r.a_amount=amt; return r; }
static MkRunSpec o_flush(int step,int sm){ MkRunSpec r=rk(MK_OP_FLUSH_SM,step); r.a_sm=sm; return r; }

// epoch predicate: BEGIN sets epoch = FNV(seed,sm,epoch_seq_next). To make predicate
// masks meaningful and deterministic across runs we just use predicate_mask = ~0
// (always pass) for most instrs, and a 0 mask to force PRED_SKIP.
static const uint64_t PALL = 0xFFFFFFFFFFFFFFFFULL;
static const uint64_t PNONE = 0ULL;

// S1: RAW/WAR/WAW head-of-line hazards + ordered retire.
static Scenario sc_hazards() {
    Scenario s; s.name="raw_war_waw_hazards";
    s.spec=make_spec(1,8,4,4,4,8,8,4,16,16,64,8,256);
    int SL=8;
    s.stream.assign(SL,mk_nop(0,1,PALL));
    // pc0: LD t0 ; pc1: ALU read t0 write t1 (RAW on t0 until LD retires) ;
    // pc2: LD t0 again (WAW + WAR vs reader at pc1) ; pc3: ST t1 ; pc4: FREE t0 ; pc5..7 NOP
    s.stream[0]=mk_ld(101,2,PALL,0,0xAA);
    s.stream[1]=mk_alu(102,2,PALL,{0},{1},0x1234);
    s.stream[2]=mk_ld(103,2,PALL,0,0xBB);
    s.stream[3]=mk_st(104,2,PALL,1,0x55);
    s.stream[4]=mk_free(105,1,PALL,0);
    s.stream[5]=mk_nop(106,1,PALL);
    s.stream[6]=mk_nop(107,1,PALL);
    s.stream[7]=mk_nop(108,1,PALL);
    int t=0;
    s.ops.push_back(o_begin(t++,0xDEAD));
    s.ops.push_back(o_decode(t++,0,4));      // decode pc0..3
    s.ops.push_back(o_issue(t++,0,4));       // LD t0 issues; ALU RAW stall(head-of-line) -> stop
    s.ops.push_back(o_advance(t++,5,8));     // LD_DONE -> complete
    s.ops.push_back(o_retire(t++,0,4));      // retire LD t0 (clears writer)
    s.ops.push_back(o_issue(t++,0,4));       // now ALU issues, LD#2 WAR/WAW stall
    s.ops.push_back(o_advance(t++,5,8));     // ALU_DONE
    s.ops.push_back(o_retire(t++,0,4));      // retire ALU (clears reader of t0, writer t1)
    s.ops.push_back(o_issue(t++,0,4));       // LD#2 + ST t1 issue
    s.ops.push_back(o_advance(t++,5,8));     // both done
    s.ops.push_back(o_retire(t++,0,4));
    s.ops.push_back(o_decode(t++,0,4));      // decode pc4..7 (FREE + NOPs)
    s.ops.push_back(o_issue(t++,0,4));
    s.ops.push_back(o_advance(t++,5,8));     // FREE_DONE frees t0
    s.ops.push_back(o_retire(t++,0,4));
    s.ops.push_back(o_decode(t++,0,4));      // pc8 halt
    return s;
}

// S2: FREE replay — FREE issues then fails late due to older reader, replays.
static Scenario sc_free_replay() {
    Scenario s; s.name="free_replay_older_reader";
    s.spec=make_spec(1,6,4,4,4,8,8,4,16,16,64,8,256);
    int SL=6;
    s.stream.assign(SL,mk_nop(0,1,PALL));
    // pc0 LD t0 ; pc1 ALU read t0 write t1 (reader of t0) ; pc2 FREE t0 (older reader at pc1)
    s.stream[0]=mk_ld(201,2,PALL,0,0x10);
    s.stream[1]=mk_alu(202,8,PALL,{0},{1},0x9);   // long latency reader
    s.stream[2]=mk_free(203,1,PALL,0);
    s.stream[3]=mk_nop(204,1,PALL);
    s.stream[4]=mk_nop(205,1,PALL);
    s.stream[5]=mk_nop(206,1,PALL);
    int t=0;
    s.ops.push_back(o_begin(t++,0x77));
    s.ops.push_back(o_decode(t++,0,3));
    s.ops.push_back(o_issue(t++,0,1));       // LD issues
    s.ops.push_back(o_advance(t++,3,4));      // LD done
    s.ops.push_back(o_retire(t++,0,1));       // retire LD
    s.ops.push_back(o_issue(t++,0,4));        // ALU issues (reads t0), FREE issues too (no RAW: FREE has no read tiles)
    s.ops.push_back(o_advance(t++,1,8));       // clock=4: FREE_DONE due(t=issueclock+1) fires while ALU still pending(due later)
    // FREE finds older reader (ALU at decode_seq < free) still unretired -> REPLAY
    s.ops.push_back(o_replay(t++,0,4));        // move FREE back to issue queue
    s.ops.push_back(o_issue(t++,0,4));         // FREE re-issues
    s.ops.push_back(o_advance(t++,10,8));      // ALU done, FREE_DONE again -> still older reader until ALU retired
    s.ops.push_back(o_retire(t++,0,4));        // retire ALU
    s.ops.push_back(o_replay(t++,0,4));
    s.ops.push_back(o_issue(t++,0,4));
    s.ops.push_back(o_advance(t++,5,8));        // FREE now succeeds
    s.ops.push_back(o_retire(t++,0,4));
    return s;
}

// S3: branch redirect cancels younger decoded instrs + new epoch + stale unit drop.
static Scenario sc_branch() {
    Scenario s; s.name="branch_redirect_epoch_cancel";
    s.spec=make_spec(1,10,6,6,6,8,8,4,32,32,64,8,256);
    int SL=10;
    s.stream.assign(SL,mk_nop(0,1,PALL));
    // pc0 INC counter0 +5 ; pc1 BRANCH if c0>=3 goto pc6 else fall pc2 ;
    // pc2,3,4,5 NOPs (speculative, will be cancelled) ; pc6 LD t0 ; pc7..9 NOP
    s.stream[0]=mk_inc(301,1,PALL,0,5);
    s.stream[1]=mk_branch(302,3,PALL,0,3,6,2);    // taken target pc6, fallthrough pc2
    s.stream[2]=mk_nop(303,1,PALL);
    s.stream[3]=mk_nop(304,1,PALL);
    s.stream[4]=mk_nop(305,1,PALL);
    s.stream[5]=mk_nop(306,1,PALL);
    s.stream[6]=mk_ld(307,2,PALL,0,0x99);
    s.stream[7]=mk_nop(308,1,PALL);
    s.stream[8]=mk_nop(309,1,PALL);
    s.stream[9]=mk_nop(310,1,PALL);
    int t=0;
    s.ops.push_back(o_begin(t++,0x55));
    s.ops.push_back(o_decode(t++,0,6));        // decode pc0(INC),pc1(BRANCH),pc2..5 (speculative fall)
    s.ops.push_back(o_issue(t++,0,6));          // INC + BRANCH + NOPs issue
    s.ops.push_back(o_advance(t++,2,8));         // INC_DONE (c0=5) ... others maybe
    s.ops.push_back(o_advance(t++,5,8));         // BRANCH_DONE: c0>=3 taken pc6 != fall pc2 -> cancel younger, new epoch
    s.ops.push_back(o_retire(t++,0,6));          // retire INC, BRANCH (cancelled ones skipped)
    s.ops.push_back(o_decode(t++,0,6));          // decode from pc6 LD under NEW epoch
    s.ops.push_back(o_issue(t++,0,6));
    s.ops.push_back(o_advance(t++,5,8));
    s.ops.push_back(o_retire(t++,0,6));
    return s;
}

// S3b: branch that resolves to the speculative fallthrough -> BRANCH_NOOP (no cancel).
static Scenario sc_branch_noop() {
    Scenario s; s.name="branch_noop_fallthrough";
    s.spec=make_spec(1,8,6,6,6,8,8,4,32,32,64,8,256);
    int SL=8;
    s.stream.assign(SL,mk_nop(0,1,PALL));
    // pc0 INC c0 +2 ; pc1 BRANCH if c0>=100 goto pc6 else fall pc2 (condition false -> fall == speculative)
    s.stream[0]=mk_inc(351,1,PALL,0,2);
    s.stream[1]=mk_branch(352,3,PALL,0,100,6,2);
    s.stream[2]=mk_nop(353,1,PALL);
    s.stream[3]=mk_nop(354,1,PALL);
    s.stream[4]=mk_nop(355,1,PALL);
    s.stream[5]=mk_nop(356,1,PALL);
    s.stream[6]=mk_nop(357,1,PALL);
    s.stream[7]=mk_nop(358,1,PALL);
    int t=0;
    s.ops.push_back(o_begin(t++,0x6677));
    s.ops.push_back(o_decode(t++,0,6));     // INC, BRANCH, pc2..5 speculative
    s.ops.push_back(o_issue(t++,0,6));
    s.ops.push_back(o_advance(t++,2,8));      // INC_DONE c0=2
    s.ops.push_back(o_advance(t++,5,8));      // BRANCH_DONE: c0<100 -> fall pc2 == speculative -> BRANCH_NOOP
    s.ops.push_back(o_retire(t++,0,6));       // all proceed, none cancelled
    s.ops.push_back(o_decode(t++,0,6));
    s.ops.push_back(o_issue(t++,0,6));
    s.ops.push_back(o_advance(t++,2,8));
    s.ops.push_back(o_retire(t++,0,6));
    return s;
}

// S4: WAIT rearm until counter target met (host increments counter).
static Scenario sc_wait() {
    Scenario s; s.name="wait_rearm_counter";
    s.spec=make_spec(1,4,4,4,4,8,8,4,16,16,64,8,256);
    int SL=4;
    s.stream.assign(SL,mk_nop(0,1,PALL));
    s.stream[0]=mk_wait(401,1,PALL,2,10);      // wait counter2 >= 10
    s.stream[1]=mk_nop(402,1,PALL);
    s.stream[2]=mk_nop(403,1,PALL);
    s.stream[3]=mk_nop(404,1,PALL);
    int t=0;
    s.ops.push_back(o_begin(t++,0x33));
    s.ops.push_back(o_decode(t++,0,1));
    s.ops.push_back(o_issue(t++,0,1));          // WAIT issues
    s.ops.push_back(o_advance(t++,2,4));          // WAIT_DONE fires, counter2=0 -> rearm
    s.ops.push_back(o_advance(t++,2,4));          // rearm again
    s.ops.push_back(o_hostinc(t++,2,4));          // counter2=4
    s.ops.push_back(o_advance(t++,2,4));          // still <10 -> rearm
    s.ops.push_back(o_hostinc(t++,2,8));          // counter2=12
    s.ops.push_back(o_advance(t++,2,4));          // now >=10 -> WAIT_DONE complete
    s.ops.push_back(o_retire(t++,0,4));
    return s;
}

// S5: page stalls (skippable) + multi-SM independent streams + FLUSH.
static Scenario sc_pages_flush() {
    Scenario s; s.name="page_stall_multi_sm_flush";
    s.spec=make_spec(2,6,4,4,4,8,2,4,16,16,64,8,256);  // only 2 pages/sm -> page pressure
    int SL=6;
    s.stream.assign((size_t)2*SL,mk_nop(0,1,PALL));
    // SM0: 3 LDs to t0,t1,t2 (only 2 pages -> 3rd page-stalls), then NOP
    s.stream[0*SL+0]=mk_ld(501,3,PALL,0,0x1);
    s.stream[0*SL+1]=mk_ld(502,3,PALL,1,0x2);
    s.stream[0*SL+2]=mk_ld(503,3,PALL,2,0x3);   // page stall while 0,1 loading
    s.stream[0*SL+3]=mk_nop(504,1,PALL);
    s.stream[0*SL+4]=mk_nop(505,1,PALL);
    s.stream[0*SL+5]=mk_nop(506,1,PALL);
    // SM1: LD t3, ALU read t3 write t4
    s.stream[1*SL+0]=mk_ld(511,2,PALL,3,0x11);
    s.stream[1*SL+1]=mk_alu(512,2,PALL,{3},{4},0x22);
    s.stream[1*SL+2]=mk_nop(513,1,PALL);
    s.stream[1*SL+3]=mk_nop(514,1,PALL);
    s.stream[1*SL+4]=mk_nop(515,1,PALL);
    s.stream[1*SL+5]=mk_nop(516,1,PALL);
    int t=0;
    s.ops.push_back(o_begin(t++,0xABCD));
    s.ops.push_back(o_decode(t++,0,4));
    s.ops.push_back(o_decode(t++,1,3));
    s.ops.push_back(o_issue(t++,0,4));          // 2 LDs issue, 3rd page-stalls (skipped), NOP after issues
    s.ops.push_back(o_issue(t++,1,4));
    s.ops.push_back(o_advance(t++,3,16));        // LD/ALU dones
    s.ops.push_back(o_issue(t++,0,4));           // now 3rd LD can issue (pages freed? no, still resident) -> still stall maybe
    s.ops.push_back(o_retire(t++,0,4));
    s.ops.push_back(o_retire(t++,1,4));
    s.ops.push_back(o_flush(t++,0));             // flush SM0: cancel remaining decoded, new epoch
    s.ops.push_back(o_advance(t++,5,16));         // any pending units of SM0 become stale-dropped
    s.ops.push_back(o_issue(t++,1,4));
    s.ops.push_back(o_retire(t++,1,4));
    return s;
}

// S6: invalid/edge ops + predicate skip + decode-queue-full + halt.
static Scenario sc_edges() {
    Scenario s; s.name="edges_pred_qfull_halt";
    s.spec=make_spec(1,8,8,8,8,8,8,4,3,16,64,8,256);  // max_decode_queue=3 small
    int SL=8;
    s.stream.assign(SL,mk_nop(0,1,PALL));
    s.stream[0]=mk_nop(601,1,PNONE);     // predicate skip
    s.stream[1]=mk_nop(602,1,PALL);
    s.stream[2]=mk_nop(603,1,PALL);
    s.stream[3]=mk_nop(604,1,PALL);
    s.stream[4]=mk_nop(605,1,PALL);
    s.stream[5]=mk_nop(606,1,PNONE);     // predicate skip
    s.stream[6]=mk_nop(607,1,PALL);
    s.stream[7]=mk_nop(608,1,PALL);
    int t=0;
    // invalid ops first (before any begin many ops still valid op_kind but invalid args)
    s.ops.push_back(o_decode(t++,5,4));   // sm oob -> invalid
    s.ops.push_back(o_issue(t++,0,0));    // limit 0 -> invalid
    s.ops.push_back(o_retire(t++,9,4));   // sm oob -> invalid
    s.ops.push_back(o_flush(t++,3));      // sm oob -> invalid
    s.ops.push_back(o_hostinc(t++,99,5)); // counter oob -> invalid
    s.ops.push_back(o_begin(t++,0x9));
    s.ops.push_back(o_decode(t++,0,8));   // skip pc0, decode pc1,2,3 -> queue full at 3, then DECODE_QUEUE_FULL
    s.ops.push_back(o_issue(t++,0,8));    // issue 3 NOPs
    s.ops.push_back(o_advance(t++,2,16));
    s.ops.push_back(o_retire(t++,0,8));   // retire 3
    s.ops.push_back(o_decode(t++,0,8));   // decode pc4, skip pc5, pc6,7 then halt at pc8
    s.ops.push_back(o_issue(t++,0,8));
    s.ops.push_back(o_advance(t++,2,16));
    s.ops.push_back(o_retire(t++,0,8));
    s.ops.push_back(o_decode(t++,0,8));   // already halted -> DECODE_HALTED
    s.ops.push_back(o_hostinc(t++,0,7));  // valid host inc
    s.ops.push_back(o_begin(t++,0x9));    // valid begin again (all retired)
    return s;
}

// S7: large pseudo-random multi-SM stress + clock wrap.
static Scenario sc_random() {
    Scenario s; s.name="random_stress_wrap";
    s.spec=make_spec(3,16,4,4,4,16,6,8,32,32,256,16,1024);
    int SM=3, SL=16;
    SplitMix64 rng(0xC0FFEE1234ULL);
    s.stream.assign((size_t)SM*SL, mk_nop(0,1,PALL));
    uint64_t uid=1000;
    for (int sm=0; sm<SM; ++sm) {
        for (int pc=0; pc<SL; ++pc) {
            int kind = rng.uniform_int(0,9);
            uint64_t lat = (uint64_t)rng.uniform_int(1,4);
            uint64_t pred = (rng.uniform_int(0,9)==0) ? PNONE : PALL;
            MkInstr in;
            if (kind<=2) in=mk_ld(uid++,lat,pred,rng.uniform_int(0,15),(uint64_t)rng.uniform_int(1,9));
            else if (kind<=4) in=mk_alu(uid++,lat,pred,{rng.uniform_int(0,15)},{rng.uniform_int(0,15)},(uint64_t)rng.uniform_int(1,99));
            else if (kind==5) in=mk_st(uid++,lat,pred,rng.uniform_int(0,15),(uint64_t)rng.uniform_int(1,9));
            else if (kind==6) in=mk_inc(uid++,lat,pred,rng.uniform_int(0,7),(uint64_t)rng.uniform_int(1,5));
            else if (kind==7) in=mk_wait(uid++,lat,pred,rng.uniform_int(0,7),(uint64_t)rng.uniform_int(1,10));
            else if (kind==8) in=mk_free(uid++,lat,pred,rng.uniform_int(0,15));
            else in=mk_nop(uid++,lat,pred);
            s.stream[(size_t)sm*SL+pc]=in;
        }
    }
    int t=0;
    s.ops.push_back(o_begin(t++,0x1357));
    for (int round=0; round<40; ++round) {
        for (int sm=0; sm<SM; ++sm) {
            s.ops.push_back(o_decode(t++,sm,rng.uniform_int(1,4)));
            s.ops.push_back(o_issue(t++,sm,rng.uniform_int(1,4)));
        }
        s.ops.push_back(o_advance(t++,(uint64_t)rng.uniform_int(1,6),rng.uniform_int(1,8)));
        for (int sm=0; sm<SM; ++sm) s.ops.push_back(o_retire(t++,sm,rng.uniform_int(1,4)));
        if (round%7==3) s.ops.push_back(o_hostinc(t++,rng.uniform_int(0,7),(uint64_t)rng.uniform_int(1,10)));
        if (round%11==5) s.ops.push_back(o_replay(t++,rng.uniform_int(0,SM-1),rng.uniform_int(1,4)));
        if (round%13==7) s.ops.push_back(o_flush(t++,rng.uniform_int(0,SM-1)));
    }
    // clock wrap near 2^64
    s.ops.push_back(o_advance(t++,0xFFFFFFFFFFFFFFF0ULL,8));
    s.ops.push_back(o_advance(t++,0x100ULL,8));
    for (int r=0;r<30;++r){ int sm=r%SM; s.ops.push_back(o_advance(t++,(uint64_t)(r+1),8)); s.ops.push_back(o_retire(t++,sm,4)); }
    return s;
}

static std::vector<Scenario> build_scenarios(){
    std::vector<Scenario> v;
    v.push_back(sc_hazards());
    v.push_back(sc_free_replay());
    v.push_back(sc_branch());
    v.push_back(sc_branch_noop());
    v.push_back(sc_wait());
    v.push_back(sc_pages_flush());
    v.push_back(sc_edges());
    v.push_back(sc_random());
    return v;
}

// ----------------------------------------------------------- runner
static bool run_one_op(const Scenario& sc, const MkRunSpec& op, const MkInstr* d_stream, int stream_count,
                       const std::vector<MkInstr>& host_stream, MkOracle* oracle,
                       void* state, void* workspace, size_t workspace_bytes, cudaStream_t stream,
                       StepResult* result, std::string* error) {
    GuardedDeviceBuffer<int64_t> d_counts;
    GuardedDeviceBuffer<int32_t> d_opidx;
    GuardedDeviceBuffer<uint64_t> d_clock,d_eseq,d_evh,d_smh,d_dh,d_th,d_ph,d_puh,d_ch,d_state;
    d_counts.allocate(MK_COUNT_N); d_opidx.allocate(1);
    d_clock.allocate(1);d_eseq.allocate(1);d_evh.allocate(1);d_smh.allocate(1);d_dh.allocate(1);
    d_th.allocate(1);d_ph.allocate(1);d_puh.allocate(1);d_ch.allocate(1);d_state.allocate(1);

    MkOutputs outputs={};
    outputs.counts=d_counts.ptr; outputs.op_index_out=d_opidx.ptr; outputs.clock_out=d_clock.ptr;
    outputs.event_seq_out=d_eseq.ptr; outputs.decoder_event_hash=d_evh.ptr; outputs.sm_stream_hash=d_smh.ptr;
    outputs.decoded_hash=d_dh.ptr; outputs.tile_scoreboard_hash=d_th.ptr; outputs.page_hash=d_ph.ptr;
    outputs.pending_unit_hash=d_puh.ptr; outputs.counter_hash=d_ch.ptr; outputs.state_checksum=d_state.ptr;

    MkRunSpec op_copy=op;
    MkInputs inputs={}; inputs.stream=d_stream; inputs.stream_count=stream_count;

    CUDA_CHECK(solution_run(state,&op_copy,&inputs,&outputs,workspace,workspace_bytes,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (std::memcmp(&op_copy,&op,sizeof(MkRunSpec))!=0) { if(error)*error="run spec mutated by solution_run"; return false; }

    // input immutability: re-download device stream and compare.
    {
        std::vector<MkInstr> back((size_t)stream_count);
        CUDA_CHECK(cudaMemcpy(back.data(), d_stream, sizeof(MkInstr)*(size_t)stream_count, cudaMemcpyDeviceToHost));
        if (std::memcmp(back.data(), host_stream.data(), sizeof(MkInstr)*(size_t)stream_count)!=0) {
            if(error)*error="input stream mutated by solution_run"; return false;
        }
    }

    if(!d_counts.check_guards("counts",error)) return false;
    if(!d_opidx.check_guards("op_index",error)) return false;
    if(!d_clock.check_guards("clock",error)) return false;
    if(!d_eseq.check_guards("event_seq",error)) return false;
    if(!d_evh.check_guards("decoder_event_hash",error)) return false;
    if(!d_smh.check_guards("sm_stream_hash",error)) return false;
    if(!d_dh.check_guards("decoded_hash",error)) return false;
    if(!d_th.check_guards("tile_scoreboard_hash",error)) return false;
    if(!d_ph.check_guards("page_hash",error)) return false;
    if(!d_puh.check_guards("pending_unit_hash",error)) return false;
    if(!d_ch.check_guards("counter_hash",error)) return false;
    if(!d_state.check_guards("state_checksum",error)) return false;

    auto hc=d_counts.download(); auto hoi=d_opidx.download();
    auto hcl=d_clock.download(); auto hes=d_eseq.download(); auto hev=d_evh.download(); auto hsm=d_smh.download();
    auto hdh=d_dh.download(); auto hth=d_th.download(); auto hph=d_ph.download(); auto hpu=d_puh.download(); auto hch=d_ch.download(); auto hst=d_state.download();

    MkExpected expected; oracle->step_once(op,&expected);
    MkHostOutputsView got={};
    got.counts=hc.data(); got.op_index_out=hoi.data(); got.clock_out=hcl.data(); got.event_seq_out=hes.data();
    got.decoder_event_hash=hev.data(); got.sm_stream_hash=hsm.data(); got.decoded_hash=hdh.data();
    got.tile_scoreboard_hash=hth.data(); got.page_hash=hph.data(); got.pending_unit_hash=hpu.data();
    got.counter_hash=hch.data(); got.state_checksum=hst.data();

    if(!mk_check_outputs(expected,got,error)) return false;

    if(result){
        result->counts=hc; result->op_index=hoi[0]; result->clock=hcl[0]; result->eseq=hes[0];
        result->evh=hev[0]; result->smh=hsm[0]; result->dh=hdh[0]; result->th=hth[0]; result->ph=hph[0];
        result->puh=hpu[0]; result->ch=hch[0]; result->state=hst[0];
    }
    return true;
}

static bool run_scenario_once(const Scenario& sc, bool verbose, std::vector<StepResult>* results,
                              int* passed, int* total, std::string* first_error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);

    cudaStream_t stream=nullptr; CUDA_CHECK(cudaStreamCreate(&stream));
    void* state=nullptr; CUDA_CHECK(solution_init(&sc.spec,&state,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    // upload stream once to a device buffer the test owns (acts as the immutable input).
    int stream_count=(int)sc.stream.size();
    DeviceBuffer<MkInstr> d_stream; d_stream.allocate((size_t)stream_count);
    CUDA_CHECK(cudaMemcpy(d_stream.ptr, sc.stream.data(), sizeof(MkInstr)*(size_t)stream_count, cudaMemcpyHostToDevice));

    MkOracle oracle; oracle.init(sc.spec, sc.stream.data());

    CUDA_CHECK(solution_reset(state,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if(results){ results->clear(); results->reserve(sc.ops.size()); }
    bool all_ok=true;
    for(size_t i=0;i<sc.ops.size();++i){
        StepResult result; std::string error;
        bool ok=run_one_op(sc, sc.ops[i], d_stream.ptr, stream_count, sc.stream, &oracle, state,
                           workspace.ptr, workspace_bytes, stream, results?&result:nullptr, &error);
        ++(*total); if(ok) ++(*passed);
        else { all_ok=false; if(first_error&&first_error->empty()){ std::ostringstream o; o<<sc.name<<" op "<<i<<" (kind="<<sc.ops[i].op_kind<<"): "<<error; *first_error=o.str(); } }
        if(results) results->push_back(result);
        if(verbose && (!ok || (i%64==0))){
            std::printf("scenario %-30s op %04zu/%04zu kind=%d %s%s%s\n",
                sc.name.c_str(),i,sc.ops.size(),sc.ops[i].op_kind, ok?"PASS":"FAIL", ok?"":"  ", ok?"":error.c_str());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(const std::vector<StepResult>& a, const std::vector<StepResult>& b, std::string* error){
    if(a.size()!=b.size()){ if(error)*error="result length mismatch"; return false; }
    for(size_t i=0;i<a.size();++i){
        if(a[i].counts!=b[i].counts||a[i].op_index!=b[i].op_index||a[i].clock!=b[i].clock||a[i].eseq!=b[i].eseq||
           a[i].evh!=b[i].evh||a[i].smh!=b[i].smh||a[i].dh!=b[i].dh||a[i].th!=b[i].th||a[i].ph!=b[i].ph||
           a[i].puh!=b[i].puh||a[i].ch!=b[i].ch||a[i].state!=b[i].state){
            if(error){ std::ostringstream o; o<<"replay mismatch at op "<<i<<" state a=0x"<<std::hex<<a[i].state<<" b=0x"<<b[i].state; *error=o.str(); }
            return false;
        }
    }
    return true;
}

static void coverage_dump(const std::vector<Scenario>& scenarios){
    static const char* nm[MK_COUNT_N]={"kernel_begin","instr_decoded","decode_pred_skip","decode_halted","decode_queue_full","instr_issue","issue_hazard","issue_page_stall","ld_done","alu_done","st_done","wait_done","wait_rearm","counter_inc","tile_free","free_replay","branch_redirect","branch_noop","epoch_cancel","unit_stale_drop","instr_complete","retire_blocked","read_retire","write_retire","instr_retire","replay_enqueue","host_counter_inc","sm_flush","invalid_count"};
    std::vector<int64_t> agg(MK_COUNT_N,0);
    for(const Scenario& sc : scenarios){
        MkOracle o; o.init(sc.spec, sc.stream.data()); o.reset();
        MkExpected e;
        for(const MkRunSpec& op : sc.ops) o.step_once(op,&e);
        for(int i=0;i<MK_COUNT_N;++i) agg[i]+=e.counts[i];
    }
    std::printf("=== aggregate event coverage across scenarios ===\n");
    for(int i=0;i<MK_COUNT_N;++i) std::printf("  %-20s %lld\n", nm[i], (long long)agg[i]);
}

int main(){
    try{
        CUDA_CHECK(cudaSetDevice(0));
        const std::vector<Scenario> scenarios=build_scenarios();
        if(getenv("MK_COVERAGE")){ coverage_dump(scenarios); }
        int passed=0,total=0; bool all_ok=true;
        for(const Scenario& sc : scenarios){
            std::vector<StepResult> base_results, replay_results; std::string error;
            bool ok_base=run_scenario_once(sc,true,&base_results,&passed,&total,&error);
            bool ok_replay=run_scenario_once(sc,false,&replay_results,&passed,&total,&error);
            if(ok_base&&ok_replay){
                std::string ce;
                if(compare_results(base_results,replay_results,&ce)) std::printf("scenario %-30s exact replay PASS (%zu ops)\n",sc.name.c_str(),sc.ops.size());
                else { all_ok=false; std::printf("scenario %-30s exact replay FAIL  %s\n",sc.name.c_str(),ce.c_str()); }
            } else { all_ok=false; std::printf("scenario %-30s FAIL  %s\n",sc.name.c_str(),error.c_str()); }
        }
        std::printf("passed %d / %d\n",passed,total);
        return (all_ok&&passed==total)?0:1;
    } catch(const std::exception& ex){ std::fprintf(stderr,"fatal: %s\n",ex.what()); return 1; }
}
