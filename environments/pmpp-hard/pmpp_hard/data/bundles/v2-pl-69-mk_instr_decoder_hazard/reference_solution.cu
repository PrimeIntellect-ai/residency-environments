// PMPP_CANARY_69_b4abcabeaa -- held-out canary; MUST NOT appear in any submission
// file: mk_instr_decoder_hazard_reference.cu
//
// Reference device implementation of MK10.  Runs the entire coarse op inside a
// single <<<1,1>>> kernel mutating persistent device state.
//
// Data-structure strategy (independent of naive + oracle):
//   * Decoded records live in a flat Structure-of-Arrays pool indexed directly
//     by (decode_seq-1); slots are never reclaimed within a kernel.
//   * SM decode/issue/replay queues are flat fixed-capacity arrays of decode_seqs
//     with explicit length counters (linear compaction on removal).
//   * The tile scoreboard, page table, counters, and pending-unit list are flat
//     device arrays; the pending list is scanned for the (due_clock,unit_seq)
//     minimum rather than kept sorted.
//   * oldest_unretired_reader is recomputed by scanning the record pool.

#include "mk_instr_decoder_hazard_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MK_FNV_OFFSET 1469598103934665603ULL
#define MK_FNV_PRIME  1099511628211ULL

namespace mk_ref {

struct State {
    MkProblemSpec spec;

    // scalars: [clock,event_seq,decode_seq_next,issue_seq_next,unit_seq_next,
    //           retire_seq_next,epoch_seq_next,op_index,event_hash]  (9 u64)
    uint64_t* scal;

    // device stream copy
    MkInstr* stream;          // [sm_count*stream_len]
    int32_t  stream_count;

    // SM state
    uint32_t* sm_pc;          // [SM]
    uint64_t* sm_epoch;       // [SM]
    uint8_t*  sm_halted;      // [SM]
    uint64_t* dq;             // [SM*max_decode_queue]
    int32_t*  dq_n;           // [SM]
    uint64_t* iq;             // [SM*max_issue_queue]
    int32_t*  iq_n;           // [SM]
    uint64_t* rq;             // [SM*max_replay]
    int32_t*  rq_n;           // [SM]

    // record pool (SoA), capacity = max_decode_records, slot = decode_seq-1
    int32_t*  r_used;         // 1 if slot allocated
    int32_t*  r_sm;
    uint32_t* r_pc;
    uint64_t* r_epoch;
    uint64_t* r_uid;
    int32_t*  r_op;
    int32_t*  r_status;
    int32_t*  r_hz;
    uint64_t* r_rhash;
    int32_t*  r_nread;  int32_t* r_read;   // r_read[slot*MK_MAX_RW]
    int32_t*  r_nwrite; int32_t* r_write;  // [slot*MK_MAX_RW]
    int32_t*  r_npage;  int32_t* r_page;   // [slot*MK_MAX_RW] (LD:1, ALU:nwrite)
    int32_t*  r_free_tile;
    int32_t*  r_counter;
    uint64_t* r_target; uint64_t* r_amount; uint64_t* r_cseed; uint64_t* r_pkey;
    int32_t*  r_dst; int32_t* r_taken; int32_t* r_fall; uint64_t* r_lat;

    // tile scoreboard
    uint64_t* t_ver; uint64_t* t_writer; uint64_t* t_oldr; uint64_t* t_rcount;
    uint32_t* t_resident; uint8_t* t_dirty; uint8_t* t_freed;

    // pages [SM*PG]
    int32_t*  p_state; uint64_t* p_key; uint32_t* p_tile; uint64_t* p_ver; uint64_t* p_owner;

    // counters [CC]
    uint64_t* ctr;

    // pending list [max_pending]
    int32_t*  u_kind; int32_t* u_sm; uint64_t* u_ds; uint64_t* u_epoch;
    uint64_t* u_due; uint64_t* u_seq; int32_t* u_n;

    // counts
    int64_t*  counts;
};

} // namespace mk_ref

// ---------------- device FNV ----------------
struct MkH { uint64_t h; };
__device__ __forceinline__ void mkh_init(MkH* f){ f->h = MK_FNV_OFFSET; }
__device__ __forceinline__ void mkh_seed(MkH* f, uint64_t s){ f->h = s; }
__device__ __forceinline__ void mkh_u8(MkH* f, uint8_t v){ f->h ^= (uint64_t)v; f->h *= MK_FNV_PRIME; }
__device__ __forceinline__ void mkh_u32(MkH* f, uint32_t v){ const uint8_t* p=(const uint8_t*)&v; for(int i=0;i<4;++i) mkh_u8(f,p[i]); }
__device__ __forceinline__ void mkh_u64(MkH* f, uint64_t v){ const uint8_t* p=(const uint8_t*)&v; for(int i=0;i<8;++i) mkh_u8(f,p[i]); }

#define U32MAXc 0xFFFFFFFFu

struct Ctx {
    mk_ref::State s;
    int SM, SL, DW, IW, RW, TC, PG, CC, MDQ, MIQ, MRP, MDR, MPU;
};

// scalar accessors
__device__ __forceinline__ uint64_t& CLK(Ctx&c){return c.s.scal[0];}
__device__ __forceinline__ uint64_t& ESEQ(Ctx&c){return c.s.scal[1];}
__device__ __forceinline__ uint64_t& DSN(Ctx&c){return c.s.scal[2];}
__device__ __forceinline__ uint64_t& ISN(Ctx&c){return c.s.scal[3];}
__device__ __forceinline__ uint64_t& USN(Ctx&c){return c.s.scal[4];}
__device__ __forceinline__ uint64_t& RSN(Ctx&c){return c.s.scal[5];}
__device__ __forceinline__ uint64_t& EPN(Ctx&c){return c.s.scal[6];}
__device__ __forceinline__ uint64_t& OPI(Ctx&c){return c.s.scal[7];}
__device__ __forceinline__ uint64_t& EVH(Ctx&c){return c.s.scal[8];}

__device__ void mk_emit(Ctx&c, uint8_t kind, uint32_t sm, uint32_t pc, uint64_t ds,
                        uint64_t uid, uint32_t tile, uint64_t aux) {
    MkH f; mkh_seed(&f, EVH(c));
    mkh_u8(&f, kind);
    mkh_u64(&f, ESEQ(c));
    mkh_u32(&f, (uint32_t)OPI(c));
    mkh_u64(&f, CLK(c));
    mkh_u32(&f, sm);
    mkh_u32(&f, pc);
    mkh_u64(&f, ds);
    mkh_u64(&f, uid);
    mkh_u32(&f, tile);
    mkh_u64(&f, aux);
    EVH(c) = f.h;
    ESEQ(c) += 1;
}
__device__ void mk_invalid(Ctx&c){ c.s.counts[MK_C_INVALID]+=1; mk_emit(c,MK_EV_INVALID,U32MAXc,U32MAXc,0,0,U32MAXc,0); }

__device__ __forceinline__ int RSLOT(uint64_t ds){ return (int)(ds-1); }

// queue helpers
__device__ void dq_push(Ctx&c,int sm,uint64_t v){ c.s.dq[sm*c.MDQ + c.s.dq_n[sm]]=v; c.s.dq_n[sm]++; }
__device__ void iq_push_back(Ctx&c,int sm,uint64_t v){ c.s.iq[sm*c.MIQ + c.s.iq_n[sm]]=v; c.s.iq_n[sm]++; }
__device__ void iq_push_front(Ctx&c,int sm,uint64_t v){ int base=sm*c.MIQ; int n=c.s.iq_n[sm]; for(int i=n;i>0;--i) c.s.iq[base+i]=c.s.iq[base+i-1]; c.s.iq[base]=v; c.s.iq_n[sm]=n+1; }
__device__ void rq_push(Ctx&c,int sm,uint64_t v){ c.s.rq[sm*c.MRP + c.s.rq_n[sm]]=v; c.s.rq_n[sm]++; }

__device__ void q_remove(uint64_t* q, int32_t* n, int sm, int cap, uint64_t v){
    int base=sm*cap; int cnt=n[sm]; int w=0;
    for(int i=0;i<cnt;++i){ if(q[base+i]==v) continue; q[base+w++]=q[base+i]; }
    n[sm]=w;
}
__device__ void remove_from_queues(Ctx&c,int sm,uint64_t ds){
    q_remove(c.s.dq,c.s.dq_n,sm,c.MDQ,ds);
    q_remove(c.s.iq,c.s.iq_n,sm,c.MIQ,ds);
    q_remove(c.s.rq,c.s.rq_n,sm,c.MRP,ds);
}

__device__ __forceinline__ int PGIDX(Ctx&c,int sm,int p){return sm*c.PG+p;}

// recompute oldest reader on sm for tile
__device__ void recompute_oldest(Ctx&c,int tile,int sm){
    uint64_t best=0;
    int n=(int)DSN(c)-1;
    for(int slot=0;slot<n;++slot){
        if(!c.s.r_used[slot]) continue;
        if(c.s.r_sm[slot]!=sm) continue;
        int st=c.s.r_status[slot];
        if(st==MK_ST_RETIRED||st==MK_ST_CANCELLED) continue;
        if(!(st==MK_ST_ISSUED||st==MK_ST_COMPLETED||st==MK_ST_REPLAY)) continue;
        bool reads=false; int nr=c.s.r_nread[slot];
        for(int k=0;k<nr;++k) if(c.s.r_read[slot*MK_MAX_RW+k]==tile){reads=true;break;}
        if(!reads) continue;
        uint64_t ds=(uint64_t)slot+1;
        if(best==0||ds<best) best=ds;
    }
    c.s.t_oldr[tile]=best;
}

__device__ void release_record_pages(Ctx&c,int slot){
    int np=c.s.r_npage[slot]; int sm=c.s.r_sm[slot];
    for(int k=0;k<np;++k){
        int p=c.s.r_page[slot*MK_MAX_RW+k]; int pi=PGIDX(c,sm,p);
        c.s.p_state[pi]=MK_PG_FREE; c.s.p_key[pi]=0; c.s.p_tile[pi]=U32MAXc; c.s.p_ver[pi]=0; c.s.p_owner[pi]=0;
    }
    c.s.r_npage[slot]=0;
}
__device__ void unwind_reservations(Ctx&c,int slot){
    uint64_t ds=(uint64_t)slot+1;
    int nw=c.s.r_nwrite[slot];
    for(int k=0;k<nw;++k){ int t=c.s.r_write[slot*MK_MAX_RW+k]; if(c.s.t_writer[t]==ds) c.s.t_writer[t]=0; }
    int nr=c.s.r_nread[slot];
    for(int k=0;k<nr;++k){ int t=c.s.r_read[slot*MK_MAX_RW+k]; if(c.s.t_rcount[t]>0) c.s.t_rcount[t]-=1; }
}

// ---------------- decode-time field fill ----------------
__device__ void fill_rec(Ctx&c,int slot,const MkInstr& in){
    c.s.r_op[slot]=in.opcode; c.s.r_uid[slot]=in.instr_uid; c.s.r_lat[slot]=in.latency;
    c.s.r_nread[slot]=0; c.s.r_nwrite[slot]=0; c.s.r_npage[slot]=0;
    c.s.r_free_tile[slot]=-1; c.s.r_counter[slot]=-1; c.s.r_target[slot]=0; c.s.r_amount[slot]=0;
    c.s.r_cseed[slot]=0; c.s.r_pkey[slot]=0; c.s.r_dst[slot]=-1; c.s.r_taken[slot]=0; c.s.r_fall[slot]=0;
    c.s.r_rhash[slot]=0;
    switch(in.opcode){
        case MK_LD:
            c.s.r_write[slot*MK_MAX_RW+0]=in.dst_tile; c.s.r_nwrite[slot]=1;
            c.s.r_dst[slot]=in.dst_tile; c.s.r_pkey[slot]=in.page_key; break;
        case MK_ALU:
            for(int i=0;i<in.n_read;++i) c.s.r_read[slot*MK_MAX_RW+i]=in.read_tiles[i];
            c.s.r_nread[slot]=in.n_read;
            for(int i=0;i<in.n_write;++i) c.s.r_write[slot*MK_MAX_RW+i]=in.write_tiles[i];
            c.s.r_nwrite[slot]=in.n_write;
            c.s.r_cseed[slot]=in.compute_seed; break;
        case MK_ST:
            c.s.r_read[slot*MK_MAX_RW+0]=in.st_read_tile; c.s.r_nread[slot]=1; break;
        case MK_WAIT:
            c.s.r_counter[slot]=in.counter_id; c.s.r_target[slot]=in.target; break;
        case MK_INC:
            c.s.r_counter[slot]=in.counter_id; c.s.r_amount[slot]=in.amount; break;
        case MK_FREE:
            c.s.r_free_tile[slot]=in.free_tile; break;
        case MK_BRANCH:
            c.s.r_counter[slot]=in.counter_id; c.s.r_target[slot]=in.target;
            c.s.r_taken[slot]=in.taken_pc; c.s.r_fall[slot]=in.fallthrough_pc; break;
        default: break;
    }
}

// ---------------- ops ----------------
__device__ bool any_outstanding(Ctx&c){
    int n=(int)DSN(c)-1;
    for(int slot=0;slot<n;++slot){ if(!c.s.r_used[slot]) continue; int st=c.s.r_status[slot]; if(st!=MK_ST_RETIRED&&st!=MK_ST_CANCELLED) return true; }
    if(c.s.u_n[0]>0) return true;
    return false;
}

__device__ uint64_t fnv_epoch(uint64_t seed,int sm,uint64_t ctr){
    MkH f; mkh_init(&f); mkh_u64(&f,seed); mkh_u32(&f,(uint32_t)sm); mkh_u64(&f,ctr); return f.h;
}

__device__ void op_begin(Ctx&c,uint64_t seed){
    if(any_outstanding(c)){ mk_invalid(c); return; }
    for(int s=0;s<c.SM;++s){
        c.s.sm_pc[s]=0; c.s.sm_epoch[s]=fnv_epoch(seed,s,EPN(c)); EPN(c)+=1; c.s.sm_halted[s]=0;
        c.s.dq_n[s]=0; c.s.iq_n[s]=0; c.s.rq_n[s]=0;
    }
    for(int i=0;i<c.CC;++i) c.s.ctr[i]=0;
    for(int t=0;t<c.TC;++t){ c.s.t_ver[t]=0; c.s.t_writer[t]=0; c.s.t_oldr[t]=0; c.s.t_rcount[t]=0; c.s.t_resident[t]=U32MAXc; c.s.t_dirty[t]=0; c.s.t_freed[t]=0; }
    int npages=c.SM*c.PG;
    for(int i=0;i<npages;++i){ c.s.p_state[i]=MK_PG_FREE; c.s.p_key[i]=0; c.s.p_tile[i]=U32MAXc; c.s.p_ver[i]=0; c.s.p_owner[i]=0; }
    int nrec=(int)DSN(c)-1; for(int slot=0;slot<nrec;++slot) c.s.r_used[slot]=0;
    c.s.counts[MK_C_KERNEL_BEGIN]+=1;
    mk_emit(c,MK_EV_KERNEL_BEGIN,U32MAXc,U32MAXc,0,0,U32MAXc,seed);
}

__device__ void op_decode(Ctx&c,int sm,int limit){
    if(sm<0||sm>=c.SM||limit==0){ mk_invalid(c); return; }
    int budget = limit<c.DW?limit:c.DW;
    for(int i=0;i<budget;++i){
        if(c.s.sm_halted[sm]){ c.s.counts[MK_C_DECODE_HALTED]+=1; mk_emit(c,MK_EV_DECODE_HALTED,sm,c.s.sm_pc[sm],0,0,U32MAXc,0); break; }
        if(c.s.dq_n[sm]>=c.MDQ){ c.s.counts[MK_C_DECODE_QUEUE_FULL]+=1; mk_emit(c,MK_EV_DECODE_QUEUE_FULL,sm,c.s.sm_pc[sm],0,0,U32MAXc,0); break; }
        uint32_t pc=c.s.sm_pc[sm];
        if(pc>=(uint32_t)c.SL){ c.s.sm_halted[sm]=1; c.s.counts[MK_C_DECODE_HALTED]+=1; mk_emit(c,MK_EV_DECODE_HALTED,sm,pc,0,0,U32MAXc,0); break; }
        const MkInstr& in=c.s.stream[(size_t)sm*c.SL+pc];
        if((in.predicate_mask & c.s.sm_epoch[sm])==0){ c.s.counts[MK_C_DECODE_PRED_SKIP]+=1; mk_emit(c,MK_EV_DECODE_PRED_SKIP,sm,pc,0,in.instr_uid,U32MAXc,0); c.s.sm_pc[sm]=pc+1; continue; }
        if((int)DSN(c)-1 >= c.MDR){ c.s.counts[MK_C_DECODE_QUEUE_FULL]+=1; mk_emit(c,MK_EV_DECODE_QUEUE_FULL,sm,pc,0,0,U32MAXc,0); break; }
        uint64_t ds=DSN(c); DSN(c)+=1; int slot=RSLOT(ds);
        c.s.r_used[slot]=1; c.s.r_sm[slot]=sm; c.s.r_pc[slot]=pc; c.s.r_epoch[slot]=c.s.sm_epoch[sm];
        c.s.r_status[slot]=MK_ST_DECODED; c.s.r_hz[slot]=MK_HZ_NONE;
        fill_rec(c,slot,in);
        dq_push(c,sm,ds); iq_push_back(c,sm,ds);
        c.s.counts[MK_C_INSTR_DECODED]+=1;
        mk_emit(c,MK_EV_INSTR_DECODE,sm,pc,ds,in.instr_uid,U32MAXc,0);
        if(in.opcode==MK_BRANCH) c.s.sm_pc[sm]=(uint32_t)in.fallthrough_pc; else c.s.sm_pc[sm]=pc+1;
    }
}

__device__ int count_free_pages(Ctx&c,int sm){ int n=0; for(int p=0;p<c.PG;++p) if(c.s.p_state[PGIDX(c,sm,p)]==MK_PG_FREE) ++n; return n; }
__device__ int lowest_free_page(Ctx&c,int sm){ for(int p=0;p<c.PG;++p) if(c.s.p_state[PGIDX(c,sm,p)]==MK_PG_FREE) return p; return -1; }

__device__ int hazard_check(Ctx&c,int slot){
    uint64_t ds=(uint64_t)slot+1;
    int nr=c.s.r_nread[slot];
    for(int k=0;k<nr;++k){ int t=c.s.r_read[slot*MK_MAX_RW+k]; if(c.s.t_freed[t]) return MK_HZ_RAW; uint64_t w=c.s.t_writer[t]; if(w!=0&&w<ds) return MK_HZ_RAW; }
    int nw=c.s.r_nwrite[slot];
    for(int k=0;k<nw;++k){ int t=c.s.r_write[slot*MK_MAX_RW+k]; uint64_t orr=c.s.t_oldr[t]; if(orr!=0&&orr<ds) return MK_HZ_WAR; }
    for(int k=0;k<nw;++k){ int t=c.s.r_write[slot*MK_MAX_RW+k]; uint64_t w=c.s.t_writer[t]; if(w!=0&&w<ds) return MK_HZ_WAW; }
    return MK_HZ_NONE;
}
__device__ bool page_check(Ctx&c,int slot){
    int op=c.s.r_op[slot]; int sm=c.s.r_sm[slot];
    if(op==MK_LD) return count_free_pages(c,sm)>=1;
    if(op==MK_ALU){ int nr=c.s.r_nread[slot]; for(int k=0;k<nr;++k){ int t=c.s.r_read[slot*MK_MAX_RW+k]; if(c.s.t_resident[t]==U32MAXc) return false; } return count_free_pages(c,sm)>=c.s.r_nwrite[slot]; }
    if(op==MK_ST){ if(c.s.r_nread[slot]==0) return true; int t=c.s.r_read[slot*MK_MAX_RW+0]; return c.s.t_resident[t]!=U32MAXc; }
    return true;
}

__device__ void reserve_and_issue(Ctx&c,int slot){
    uint64_t ds=(uint64_t)slot+1; int sm=c.s.r_sm[slot]; int op=c.s.r_op[slot];
    int nr=c.s.r_nread[slot];
    for(int k=0;k<nr;++k){ int t=c.s.r_read[slot*MK_MAX_RW+k]; c.s.t_rcount[t]+=1; if(c.s.t_oldr[t]==0||c.s.t_oldr[t]>ds) c.s.t_oldr[t]=ds; }
    int nw=c.s.r_nwrite[slot];
    for(int k=0;k<nw;++k){ int t=c.s.r_write[slot*MK_MAX_RW+k]; c.s.t_writer[t]=ds; c.s.t_ver[t]+=1; c.s.t_dirty[t]=1; }
    if(op==MK_LD){
        int p=lowest_free_page(c,sm); int pi=PGIDX(c,sm,p);
        c.s.p_state[pi]=MK_PG_LOADING; c.s.p_owner[pi]=ds; c.s.p_key[pi]=c.s.r_pkey[slot];
        int dst=c.s.r_dst[slot]; c.s.p_tile[pi]=(uint32_t)dst; c.s.p_ver[pi]=c.s.t_ver[dst];
        c.s.r_page[slot*MK_MAX_RW+0]=p; c.s.r_npage[slot]=1;
    } else if(op==MK_ALU){
        int np=0;
        for(int k=0;k<nw;++k){ int t=c.s.r_write[slot*MK_MAX_RW+k]; int p=lowest_free_page(c,sm); int pi=PGIDX(c,sm,p);
            c.s.p_state[pi]=MK_PG_COMPUTING; c.s.p_owner[pi]=ds; c.s.p_key[pi]=0; c.s.p_tile[pi]=(uint32_t)t; c.s.p_ver[pi]=c.s.t_ver[t];
            c.s.r_page[slot*MK_MAX_RW+np]=p; np++; }
        c.s.r_npage[slot]=np;
    }
    int ukind;
    switch(op){ case MK_LD:ukind=MK_U_LD_DONE;break; case MK_ALU:ukind=MK_U_ALU_DONE;break; case MK_ST:ukind=MK_U_ST_DONE;break;
        case MK_WAIT:ukind=MK_U_WAIT_DONE;break; case MK_INC:ukind=MK_U_INC_DONE;break; case MK_FREE:ukind=MK_U_FREE_DONE;break;
        case MK_BRANCH:ukind=MK_U_BRANCH_DONE;break; default:ukind=MK_U_ST_DONE;break; }
    int ui=c.s.u_n[0];
    c.s.u_kind[ui]=ukind; c.s.u_sm[ui]=sm; c.s.u_ds[ui]=ds; c.s.u_epoch[ui]=c.s.r_epoch[slot];
    c.s.u_due[ui]=CLK(c)+c.s.r_lat[slot]; c.s.u_seq[ui]=USN(c); USN(c)+=1; c.s.u_n[0]=ui+1;
    c.s.r_status[slot]=MK_ST_ISSUED;
    c.s.counts[MK_C_INSTR_ISSUE]+=1;
    mk_emit(c,MK_EV_INSTR_ISSUE,sm,c.s.r_pc[slot],ds,c.s.r_uid[slot],U32MAXc,0);
}

__device__ void op_issue(Ctx&c,int sm,int limit){
    if(sm<0||sm>=c.SM||limit==0){ mk_invalid(c); return; }
    int budget=limit<c.IW?limit:c.IW;
    int issued=0;
    for(;;){
        if(issued>=budget) break;
        int n=c.s.iq_n[sm]; if(n==0) break;
        bool progressed=false, stop=false;
        // process in ascending decode_seq order; find smallest unprocessed
        uint64_t prev=0; bool have_prev=false;
        for(;;){
            if(issued>=budget){ stop=true; break; }
            // find smallest iq entry > prev
            int base=sm*c.MIQ; n=c.s.iq_n[sm];
            uint64_t best=0; bool found=false;
            for(int i=0;i<n;++i){ uint64_t v=c.s.iq[base+i]; if(have_prev&&v<=prev) continue; if(!found||v<best){best=v;found=true;} }
            if(!found) break;
            prev=best; have_prev=true;
            uint64_t ds=best; int slot=RSLOT(ds);
            if(!c.s.r_used[slot]){ remove_from_queues(c,sm,ds); progressed=true; continue; }
            int st=c.s.r_status[slot];
            if(!(st==MK_ST_DECODED||st==MK_ST_REPLAY)){ q_remove(c.s.iq,c.s.iq_n,sm,c.MIQ,ds); progressed=true; continue; }
            int hz=hazard_check(c,slot);
            if(hz!=MK_HZ_NONE){ c.s.r_hz[slot]=hz; c.s.counts[MK_C_ISSUE_HAZARD]+=1; mk_emit(c,MK_EV_ISSUE_HAZARD,sm,c.s.r_pc[slot],ds,c.s.r_uid[slot],U32MAXc,(uint64_t)hz); stop=true; break; }
            if(!page_check(c,slot)){ c.s.counts[MK_C_ISSUE_PAGE_STALL]+=1; mk_emit(c,MK_EV_ISSUE_PAGE_STALL,sm,c.s.r_pc[slot],ds,c.s.r_uid[slot],U32MAXc,0); continue; }
            reserve_and_issue(c,slot);
            q_remove(c.s.iq,c.s.iq_n,sm,c.MIQ,ds);
            issued++; progressed=true;
        }
        if(stop) break;
        if(!progressed) break;
    }
}

__device__ uint64_t alu_rhash(Ctx&c,int slot){
    MkH f; mkh_init(&f); mkh_u64(&f,c.s.r_cseed[slot]);
    int nr=c.s.r_nread[slot]; for(int k=0;k<nr;++k){ int t=c.s.r_read[slot*MK_MAX_RW+k]; mkh_u32(&f,(uint32_t)t); mkh_u64(&f,c.s.t_ver[t]); }
    int nw=c.s.r_nwrite[slot]; for(int k=0;k<nw;++k){ int t=c.s.r_write[slot*MK_MAX_RW+k]; mkh_u32(&f,(uint32_t)t); mkh_u64(&f,c.s.t_ver[t]); }
    return f.h;
}

__device__ int pick_ready_unit(Ctx&c){
    int n=c.s.u_n[0]; int best=-1; uint64_t bd=0,bs=0;
    for(int i=0;i<n;++i){ if(c.s.u_due[i]>CLK(c)) continue; if(best<0||c.s.u_due[i]<bd||(c.s.u_due[i]==bd&&c.s.u_seq[i]<bs)){best=i;bd=c.s.u_due[i];bs=c.s.u_seq[i];} }
    return best;
}
__device__ void pending_erase(Ctx&c,int idx){
    int n=c.s.u_n[0];
    for(int i=idx;i<n-1;++i){ c.s.u_kind[i]=c.s.u_kind[i+1]; c.s.u_sm[i]=c.s.u_sm[i+1]; c.s.u_ds[i]=c.s.u_ds[i+1]; c.s.u_epoch[i]=c.s.u_epoch[i+1]; c.s.u_due[i]=c.s.u_due[i+1]; c.s.u_seq[i]=c.s.u_seq[i+1]; }
    c.s.u_n[0]=n-1;
}
__device__ void complete_record(Ctx&c,int slot){
    c.s.r_status[slot]=MK_ST_COMPLETED;
    c.s.counts[MK_C_INSTR_COMPLETE]+=1;
    mk_emit(c,MK_EV_INSTR_COMPLETE,c.s.r_sm[slot],c.s.r_pc[slot],(uint64_t)slot+1,c.s.r_uid[slot],U32MAXc,0);
}

__device__ void cancel_younger(Ctx&c,int sm,uint64_t ep,uint64_t after){
    int n=(int)DSN(c)-1;
    // descending order: iterate slot high->low
    for(int slot=n-1;slot>=0;--slot){
        if(!c.s.r_used[slot]) continue;
        if(c.s.r_sm[slot]!=sm) continue;
        if(c.s.r_epoch[slot]!=ep) continue;
        uint64_t ds=(uint64_t)slot+1; if(ds<=after) continue;
        int st=c.s.r_status[slot]; if(st==MK_ST_RETIRED||st==MK_ST_CANCELLED) continue;
        bool was_issued=(st==MK_ST_ISSUED||st==MK_ST_COMPLETED||st==MK_ST_REPLAY);
        release_record_pages(c,slot);
        if(was_issued) unwind_reservations(c,slot);
        c.s.r_status[slot]=MK_ST_CANCELLED;
        remove_from_queues(c,sm,ds);
        c.s.counts[MK_C_EPOCH_CANCEL]+=1;
        mk_emit(c,MK_EV_EPOCH_CANCEL,sm,c.s.r_pc[slot],ds,c.s.r_uid[slot],U32MAXc,0);
    }
    // recompute oldest readers for read tiles of cancelled records
    for(int slot=n-1;slot>=0;--slot){
        if(!c.s.r_used[slot]) continue; if(c.s.r_sm[slot]!=sm) continue;
        if(c.s.r_status[slot]!=MK_ST_CANCELLED) continue; if(c.s.r_epoch[slot]!=ep) continue;
        uint64_t ds=(uint64_t)slot+1; if(ds<=after) continue;
        int nr=c.s.r_nread[slot]; for(int k=0;k<nr;++k) recompute_oldest(c,c.s.r_read[slot*MK_MAX_RW+k],sm);
    }
}

__device__ void op_advance(Ctx&c,uint64_t delta,int max_units){
    CLK(c)+=delta;
    int processed=0;
    while(processed<max_units){
        int idx=pick_ready_unit(c); if(idx<0) break;
        int ukind=c.s.u_kind[idx]; int usm=c.s.u_sm[idx]; uint64_t uds=c.s.u_ds[idx]; uint64_t uep=c.s.u_epoch[idx];
        pending_erase(c,idx);
        int slot=RSLOT(uds);
        if(!c.s.r_used[slot]){ c.s.counts[MK_C_UNIT_STALE_DROP]+=1; mk_emit(c,MK_EV_UNIT_STALE_DROP,usm,U32MAXc,uds,0,U32MAXc,0); processed++; continue; }
        if(c.s.sm_epoch[usm]!=uep){ c.s.counts[MK_C_UNIT_STALE_DROP]+=1; mk_emit(c,MK_EV_UNIT_STALE_DROP,usm,c.s.r_pc[slot],uds,c.s.r_uid[slot],U32MAXc,0); processed++; continue; }
        bool completed=true;
        switch(ukind){
            case MK_U_LD_DONE:{ int p=(c.s.r_npage[slot]>0)?c.s.r_page[slot*MK_MAX_RW+0]:-1; if(p>=0){ int pi=PGIDX(c,usm,p); c.s.p_state[pi]=MK_PG_RESIDENT; int dst=c.s.r_dst[slot]; c.s.p_tile[pi]=(uint32_t)dst; c.s.p_ver[pi]=c.s.t_ver[dst]; c.s.t_resident[dst]=(uint32_t)p; } c.s.counts[MK_C_LD_DONE]+=1; mk_emit(c,MK_EV_LD_DONE,usm,c.s.r_pc[slot],uds,c.s.r_uid[slot],(uint32_t)c.s.r_dst[slot],0); break; }
            case MK_U_ALU_DONE:{ uint64_t rh=alu_rhash(c,slot); c.s.r_rhash[slot]=rh; int np=c.s.r_npage[slot]; for(int k=0;k<np;++k){ int p=c.s.r_page[slot*MK_MAX_RW+k]; int pi=PGIDX(c,usm,p); c.s.p_state[pi]=MK_PG_RESIDENT; int t=c.s.r_write[slot*MK_MAX_RW+k]; c.s.t_resident[t]=(uint32_t)p; c.s.p_tile[pi]=(uint32_t)t; c.s.p_ver[pi]=c.s.t_ver[t]; } c.s.counts[MK_C_ALU_DONE]+=1; mk_emit(c,MK_EV_ALU_DONE,usm,c.s.r_pc[slot],uds,c.s.r_uid[slot],U32MAXc,rh); break; }
            case MK_U_ST_DONE:{ c.s.counts[MK_C_ST_DONE]+=1; mk_emit(c,MK_EV_ST_DONE,usm,c.s.r_pc[slot],uds,c.s.r_uid[slot],U32MAXc,0); break; }
            case MK_U_WAIT_DONE:{ int cid=c.s.r_counter[slot]; uint64_t tgt=c.s.r_target[slot]; if(c.s.ctr[cid]>=tgt){ c.s.counts[MK_C_WAIT_DONE]+=1; mk_emit(c,MK_EV_WAIT_DONE,usm,c.s.r_pc[slot],uds,c.s.r_uid[slot],U32MAXc,c.s.ctr[cid]); } else { int ui=c.s.u_n[0]; c.s.u_kind[ui]=MK_U_WAIT_DONE; c.s.u_sm[ui]=usm; c.s.u_ds[ui]=uds; c.s.u_epoch[ui]=uep; c.s.u_due[ui]=CLK(c)+1; c.s.u_seq[ui]=USN(c); USN(c)+=1; c.s.u_n[0]=ui+1; c.s.counts[MK_C_WAIT_REARM]+=1; mk_emit(c,MK_EV_WAIT_REARM,usm,c.s.r_pc[slot],uds,c.s.r_uid[slot],U32MAXc,tgt); completed=false; } break; }
            case MK_U_INC_DONE:{ int cid=c.s.r_counter[slot]; c.s.ctr[cid]+=c.s.r_amount[slot]; c.s.counts[MK_C_COUNTER_INC]+=1; mk_emit(c,MK_EV_COUNTER_INC,usm,c.s.r_pc[slot],uds,c.s.r_uid[slot],U32MAXc,c.s.ctr[cid]); break; }
            case MK_U_FREE_DONE:{ int t=c.s.r_free_tile[slot]; bool older=false; if(t>=0){ if(c.s.t_writer[t]!=0&&c.s.t_writer[t]<uds) older=true; if(c.s.t_oldr[t]!=0&&c.s.t_oldr[t]<uds) older=true; } if(!older&&t>=0){ c.s.t_freed[t]=1; uint32_t rp=c.s.t_resident[t]; if(rp!=U32MAXc){ int pi=PGIDX(c,usm,(int)rp); c.s.p_state[pi]=MK_PG_FREE; c.s.p_key[pi]=0; c.s.p_tile[pi]=U32MAXc; c.s.p_ver[pi]=0; c.s.p_owner[pi]=0; c.s.t_resident[t]=U32MAXc; } c.s.counts[MK_C_TILE_FREE]+=1; mk_emit(c,MK_EV_TILE_FREE,usm,c.s.r_pc[slot],uds,c.s.r_uid[slot],(uint32_t)t,0); } else { c.s.r_status[slot]=MK_ST_REPLAY; rq_push(c,usm,uds); c.s.counts[MK_C_FREE_REPLAY]+=1; mk_emit(c,MK_EV_FREE_REPLAY,usm,c.s.r_pc[slot],uds,c.s.r_uid[slot],(uint32_t)(t<0?U32MAXc:(uint32_t)t),0); completed=false; } break; }
            case MK_U_BRANCH_DONE:{ int cid=c.s.r_counter[slot]; uint64_t tgt=c.s.r_target[slot]; uint64_t resolved=(c.s.ctr[cid]>=tgt)?(uint64_t)c.s.r_taken[slot]:(uint64_t)c.s.r_fall[slot]; if(resolved!=(uint64_t)c.s.r_fall[slot]){ cancel_younger(c,usm,c.s.r_epoch[slot],uds); c.s.sm_pc[usm]=(uint32_t)resolved; c.s.sm_epoch[usm]=EPN(c)++; c.s.counts[MK_C_BRANCH_REDIRECT]+=1; mk_emit(c,MK_EV_BRANCH_REDIRECT,usm,c.s.sm_pc[usm],uds,c.s.r_uid[slot],U32MAXc,resolved); } else { c.s.counts[MK_C_BRANCH_NOOP]+=1; mk_emit(c,MK_EV_BRANCH_NOOP,usm,c.s.r_pc[slot],uds,c.s.r_uid[slot],U32MAXc,resolved); } break; }
            default: break;
        }
        if(completed) complete_record(c,slot);
        processed++;
    }
}

__device__ uint64_t head_record(Ctx&c,int sm){
    int n=(int)DSN(c)-1; uint64_t best=0;
    for(int slot=0;slot<n;++slot){ if(!c.s.r_used[slot]) continue; if(c.s.r_sm[slot]!=sm) continue; int st=c.s.r_status[slot]; if(st==MK_ST_RETIRED||st==MK_ST_CANCELLED) continue; uint64_t ds=(uint64_t)slot+1; if(best==0||ds<best) best=ds; }
    return best;
}

__device__ void op_retire(Ctx&c,int sm,int limit){
    if(sm<0||sm>=c.SM||limit==0){ mk_invalid(c); return; }
    int budget=limit<c.RW?limit:c.RW; int retired=0;
    while(retired<budget){
        uint64_t head=head_record(c,sm); if(head==0) break; int slot=RSLOT(head);
        if(c.s.r_status[slot]!=MK_ST_COMPLETED){ c.s.counts[MK_C_RETIRE_BLOCKED]+=1; mk_emit(c,MK_EV_RETIRE_BLOCKED,sm,c.s.r_pc[slot],head,c.s.r_uid[slot],U32MAXc,0); break; }
        int nr=c.s.r_nread[slot];
        for(int k=0;k<nr;++k){ int t=c.s.r_read[slot*MK_MAX_RW+k]; if(c.s.t_rcount[t]>0) c.s.t_rcount[t]-=1; }
        c.s.r_status[slot]=MK_ST_RETIRED;
        for(int k=0;k<nr;++k){ int t=c.s.r_read[slot*MK_MAX_RW+k]; recompute_oldest(c,t,sm); c.s.counts[MK_C_READ_RETIRE]+=1; mk_emit(c,MK_EV_READ_RETIRE,sm,c.s.r_pc[slot],head,c.s.r_uid[slot],(uint32_t)t,0); }
        int nw=c.s.r_nwrite[slot];
        for(int k=0;k<nw;++k){ int t=c.s.r_write[slot*MK_MAX_RW+k]; if(c.s.t_writer[t]==head) c.s.t_writer[t]=0; c.s.counts[MK_C_WRITE_RETIRE]+=1; mk_emit(c,MK_EV_WRITE_RETIRE,sm,c.s.r_pc[slot],head,c.s.r_uid[slot],(uint32_t)t,0); }
        uint64_t rseq=RSN(c)++; c.s.counts[MK_C_INSTR_RETIRE]+=1; mk_emit(c,MK_EV_INSTR_RETIRE,sm,c.s.r_pc[slot],head,c.s.r_uid[slot],U32MAXc,rseq);
        remove_from_queues(c,sm,head);
        retired++;
    }
}

__device__ void op_replay(Ctx&c,int sm,int limit){
    if(sm<0||sm>=c.SM||limit==0){ mk_invalid(c); return; }
    int moved=0;
    // take up to limit from front of replay queue (rq order), prepend to iq front preserving order.
    uint64_t batch[MK_MAX_REPLAY]; int bn=0;
    int base=sm*c.MRP; int rn=c.s.rq_n[sm];
    while(moved<limit && moved<rn){ batch[bn++]=c.s.rq[base+moved]; moved++; }
    // remove first 'moved' from rq (shift)
    for(int i=moved;i<rn;++i) c.s.rq[base+i-moved]=c.s.rq[base+i];
    c.s.rq_n[sm]=rn-moved;
    // prepend batch (in order) to iq front: insert reversed so batch[0] ends first
    for(int i=bn-1;i>=0;--i) iq_push_front(c,sm,batch[i]);
    for(int i=0;i<bn;++i){ uint64_t ds=batch[i]; int slot=RSLOT(ds); c.s.counts[MK_C_REPLAY_ENQUEUE]+=1; mk_emit(c,MK_EV_REPLAY_ENQUEUE,sm,c.s.r_used[slot]?c.s.r_pc[slot]:U32MAXc,ds,c.s.r_used[slot]?c.s.r_uid[slot]:0,U32MAXc,0); }
}

__device__ void op_host_inc(Ctx&c,int cid,uint64_t amount){
    if(cid<0||cid>=c.CC){ mk_invalid(c); return; }
    c.s.ctr[cid]+=amount; c.s.counts[MK_C_HOST_COUNTER_INC]+=1;
    mk_emit(c,MK_EV_HOST_COUNTER_INC,U32MAXc,U32MAXc,0,0,U32MAXc,c.s.ctr[cid]);
}

__device__ void op_flush(Ctx&c,int sm){
    if(sm<0||sm>=c.SM){ mk_invalid(c); return; }
    int n=(int)DSN(c)-1;
    for(int slot=n-1;slot>=0;--slot){
        if(!c.s.r_used[slot]) continue; if(c.s.r_sm[slot]!=sm) continue;
        int st=c.s.r_status[slot]; if(st==MK_ST_RETIRED||st==MK_ST_CANCELLED) continue;
        bool was_issued=(st==MK_ST_ISSUED||st==MK_ST_COMPLETED||st==MK_ST_REPLAY);
        release_record_pages(c,slot); if(was_issued) unwind_reservations(c,slot);
        c.s.r_status[slot]=MK_ST_CANCELLED;
        c.s.counts[MK_C_EPOCH_CANCEL]+=1; mk_emit(c,MK_EV_EPOCH_CANCEL,sm,c.s.r_pc[slot],(uint64_t)slot+1,c.s.r_uid[slot],U32MAXc,0);
    }
    for(int slot=n-1;slot>=0;--slot){ if(!c.s.r_used[slot]) continue; if(c.s.r_sm[slot]!=sm) continue; if(c.s.r_status[slot]!=MK_ST_CANCELLED) continue; int nr=c.s.r_nread[slot]; for(int k=0;k<nr;++k) recompute_oldest(c,c.s.r_read[slot*MK_MAX_RW+k],sm); }
    c.s.dq_n[sm]=0; c.s.iq_n[sm]=0; c.s.rq_n[sm]=0;
    c.s.sm_epoch[sm]=EPN(c)++;
    c.s.counts[MK_C_SM_FLUSH]+=1; mk_emit(c,MK_EV_SM_FLUSH,sm,c.s.sm_pc[sm],0,0,U32MAXc,0);
}

// ---------------- hashes ----------------
__device__ uint64_t hash_sm_stream(Ctx&c){
    MkH f; mkh_init(&f);
    for(int s=0;s<c.SM;++s){
        mkh_u32(&f,(uint32_t)s); mkh_u32(&f,c.s.sm_pc[s]); mkh_u64(&f,c.s.sm_epoch[s]); mkh_u8(&f,c.s.sm_halted[s]);
        int dn=c.s.dq_n[s]; mkh_u32(&f,(uint32_t)dn); for(int i=0;i<dn;++i) mkh_u64(&f,c.s.dq[s*c.MDQ+i]);
        int in=c.s.iq_n[s]; mkh_u32(&f,(uint32_t)in); for(int i=0;i<in;++i) mkh_u64(&f,c.s.iq[s*c.MIQ+i]);
        int rn=c.s.rq_n[s]; mkh_u32(&f,(uint32_t)rn); for(int i=0;i<rn;++i) mkh_u64(&f,c.s.rq[s*c.MRP+i]);
    }
    return f.h;
}
__device__ uint64_t hash_decoded(Ctx&c){
    MkH f; mkh_init(&f); int n=(int)DSN(c)-1;
    for(int s=0;s<c.SM;++s){
        for(int slot=0;slot<n;++slot){
            if(!c.s.r_used[slot]) continue; if(c.s.r_sm[slot]!=s) continue;
            int st=c.s.r_status[slot]; if(st==MK_ST_RETIRED||st==MK_ST_CANCELLED) continue;
            mkh_u32(&f,(uint32_t)c.s.r_sm[slot]); mkh_u32(&f,c.s.r_pc[slot]); mkh_u64(&f,c.s.r_epoch[slot]); mkh_u64(&f,(uint64_t)slot+1);
            mkh_u64(&f,c.s.r_uid[slot]); mkh_u8(&f,(uint8_t)c.s.r_op[slot]); mkh_u8(&f,(uint8_t)c.s.r_status[slot]); mkh_u8(&f,(uint8_t)c.s.r_hz[slot]);
            mkh_u64(&f,c.s.r_rhash[slot]);
            int nr=c.s.r_nread[slot]; mkh_u32(&f,(uint32_t)nr); for(int k=0;k<nr;++k) mkh_u32(&f,(uint32_t)c.s.r_read[slot*MK_MAX_RW+k]);
            int nw=c.s.r_nwrite[slot]; mkh_u32(&f,(uint32_t)nw); for(int k=0;k<nw;++k) mkh_u32(&f,(uint32_t)c.s.r_write[slot*MK_MAX_RW+k]);
            int np=c.s.r_npage[slot]; mkh_u32(&f,(uint32_t)np); for(int k=0;k<np;++k) mkh_u32(&f,(uint32_t)c.s.r_page[slot*MK_MAX_RW+k]);
        }
    }
    return f.h;
}
__device__ uint64_t hash_tiles(Ctx&c){
    MkH f; mkh_init(&f);
    for(int t=0;t<c.TC;++t){ mkh_u32(&f,(uint32_t)t); mkh_u64(&f,c.s.t_ver[t]); mkh_u64(&f,c.s.t_writer[t]); mkh_u64(&f,c.s.t_oldr[t]); mkh_u64(&f,c.s.t_rcount[t]); mkh_u32(&f,c.s.t_resident[t]); mkh_u8(&f,c.s.t_dirty[t]); mkh_u8(&f,c.s.t_freed[t]); }
    return f.h;
}
__device__ uint64_t hash_pages(Ctx&c){
    MkH f; mkh_init(&f);
    for(int s=0;s<c.SM;++s) for(int p=0;p<c.PG;++p){ int pi=PGIDX(c,s,p); mkh_u32(&f,(uint32_t)s); mkh_u32(&f,(uint32_t)p); mkh_u8(&f,(uint8_t)c.s.p_state[pi]); mkh_u64(&f,c.s.p_key[pi]); mkh_u32(&f,c.s.p_tile[pi]); mkh_u64(&f,c.s.p_ver[pi]); mkh_u64(&f,c.s.p_owner[pi]); }
    return f.h;
}
__device__ uint64_t hash_pending(Ctx&c){
    MkH f; mkh_init(&f); int n=c.s.u_n[0];
    // emit in (due_clock,unit_seq) ascending order via repeated min-selection
    uint64_t pd=0,ps=0; bool have=false;
    for(int e=0;e<n;++e){
        int best=-1; uint64_t bd=0,bs=0;
        for(int i=0;i<n;++i){ uint64_t d=c.s.u_due[i], sq=c.s.u_seq[i];
            if(have){ bool gt=(d>pd)||(d==pd&&sq>ps); if(!gt) continue; }
            if(best<0||d<bd||(d==bd&&sq<bs)){best=i;bd=d;bs=sq;} }
        if(best<0) break;
        mkh_u8(&f,(uint8_t)c.s.u_kind[best]); mkh_u32(&f,(uint32_t)c.s.u_sm[best]); mkh_u64(&f,c.s.u_ds[best]); mkh_u64(&f,c.s.u_epoch[best]); mkh_u64(&f,c.s.u_due[best]); mkh_u64(&f,c.s.u_seq[best]);
        pd=bd; ps=bs; have=true;
    }
    return f.h;
}
__device__ uint64_t hash_counters(Ctx&c){ MkH f; mkh_init(&f); for(int i=0;i<c.CC;++i){ mkh_u32(&f,(uint32_t)i); mkh_u64(&f,c.s.ctr[i]); } return f.h; }

// ---------------- main kernel ----------------
__global__ void mk_ref_step(mk_ref::State s,
    int SM,int SL,int DW,int IW,int RW,int TC,int PG,int CC,int MDQ,int MIQ,int MRP,int MDR,int MPU,
    int op_kind,int a_sm,int a_limit,int a_max_units,int a_counter,
    uint64_t a_seed,uint64_t a_delta,uint64_t a_amount,
    int64_t* o_counts,int32_t* o_opidx,uint64_t* o_clock,uint64_t* o_eseq,
    uint64_t* o_evh,uint64_t* o_smh,uint64_t* o_dh,uint64_t* o_th,uint64_t* o_ph,uint64_t* o_puh,uint64_t* o_ch,uint64_t* o_state){
    if(blockIdx.x!=0||threadIdx.x!=0) return;
    Ctx c; c.s=s; c.SM=SM;c.SL=SL;c.DW=DW;c.IW=IW;c.RW=RW;c.TC=TC;c.PG=PG;c.CC=CC;c.MDQ=MDQ;c.MIQ=MIQ;c.MRP=MRP;c.MDR=MDR;c.MPU=MPU;
    switch(op_kind){
        case MK_OP_BEGIN_KERNEL: op_begin(c,a_seed); break;
        case MK_OP_DECODE_SM: op_decode(c,a_sm,a_limit); break;
        case MK_OP_ISSUE_SM: op_issue(c,a_sm,a_limit); break;
        case MK_OP_ADVANCE: op_advance(c,a_delta,a_max_units); break;
        case MK_OP_RETIRE_SM: op_retire(c,a_sm,a_limit); break;
        case MK_OP_REPLAY_SM: op_replay(c,a_sm,a_limit); break;
        case MK_OP_HOST_INC_COUNTER: op_host_inc(c,a_counter,a_amount); break;
        case MK_OP_FLUSH_SM: op_flush(c,a_sm); break;
        default: break;
    }
    int32_t this_op=(int32_t)OPI(c); OPI(c)+=1;
    uint64_t smh=hash_sm_stream(c), dh=hash_decoded(c), th=hash_tiles(c), ph=hash_pages(c), puh=hash_pending(c), ch=hash_counters(c);
    for(int i=0;i<MK_COUNT_N;++i) o_counts[i]=c.s.counts[i];
    *o_opidx=this_op; *o_clock=CLK(c); *o_eseq=ESEQ(c); *o_evh=EVH(c);
    *o_smh=smh; *o_dh=dh; *o_th=th; *o_ph=ph; *o_puh=puh; *o_ch=ch;
    MkH f; mkh_init(&f);
    mkh_u64(&f,CLK(c)); mkh_u64(&f,ESEQ(c)); mkh_u32(&f,(uint32_t)OPI(c));
    mkh_u64(&f,DSN(c)); mkh_u64(&f,ISN(c)); mkh_u64(&f,USN(c)); mkh_u64(&f,RSN(c)); mkh_u64(&f,EPN(c));
    mkh_u64(&f,EVH(c)); mkh_u64(&f,smh); mkh_u64(&f,dh); mkh_u64(&f,th); mkh_u64(&f,ph); mkh_u64(&f,puh); mkh_u64(&f,ch);
    for(int i=0;i<MK_COUNT_N;++i) mkh_u64(&f,(uint64_t)c.s.counts[i]);
    *o_state=f.h;
}

__global__ void mk_ref_reset(mk_ref::State s,int SM,int TC,int PG,int CC,int MDR,int MPU){
    if(blockIdx.x!=0||threadIdx.x!=0) return;
    for(int i=0;i<9;++i) s.scal[i]=0;
    s.scal[2]=1; s.scal[3]=1; s.scal[4]=1; s.scal[5]=1; s.scal[6]=1; // seq nexts
    s.scal[8]=MK_FNV_OFFSET;
    for(int i=0;i<SM;++i){ s.sm_pc[i]=0; s.sm_epoch[i]=0; s.sm_halted[i]=0; s.dq_n[i]=0; s.iq_n[i]=0; s.rq_n[i]=0; }
    for(int i=0;i<MDR;++i) s.r_used[i]=0;
    for(int t=0;t<TC;++t){ s.t_ver[t]=0; s.t_writer[t]=0; s.t_oldr[t]=0; s.t_rcount[t]=0; s.t_resident[t]=U32MAXc; s.t_dirty[t]=0; s.t_freed[t]=0; }
    int np=SM*PG; for(int i=0;i<np;++i){ s.p_state[i]=MK_PG_FREE; s.p_key[i]=0; s.p_tile[i]=U32MAXc; s.p_ver[i]=0; s.p_owner[i]=0; }
    for(int i=0;i<CC;++i) s.ctr[i]=0;
    s.u_n[0]=0;
    for(int i=0;i<MK_COUNT_N;++i) s.counts[i]=0;
    (void)MPU;
}

// ---------------- host ABI ----------------
extern "C" size_t solution_workspace_bytes(const MkProblemSpec* spec){
    if(!mk_validate_problem_spec(spec)) return 0;
    return 256;
}

static cudaError_t mk_ref_do_reset(mk_ref::State* st, cudaStream_t stream){
    const MkProblemSpec& sp=st->spec;
    mk_ref_reset<<<1,1,0,stream>>>(*st, sp.sm_count, sp.tile_count, sp.page_count_per_sm, sp.counter_count, sp.max_decode_records, sp.max_pending_units);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_init(const MkProblemSpec* spec, void** state_out, cudaStream_t stream){
    if(!mk_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    mk_ref::State* st=(mk_ref::State*)malloc(sizeof(mk_ref::State));
    if(!st) return cudaErrorMemoryAllocation;
    memset(st,0,sizeof(mk_ref::State));
    memcpy(&st->spec,spec,sizeof(MkProblemSpec));
    const int SM=spec->sm_count, SL=spec->stream_len_per_sm, TC=spec->tile_count, PG=spec->page_count_per_sm, CC=spec->counter_count;
    const int MDQ=spec->max_decode_queue, MIQ=spec->max_issue_queue, MRP=spec->max_replay, MDR=spec->max_decode_records, MPU=spec->max_pending_units;
    cudaError_t err=cudaSuccess;
    #define M(p,bytes) do{ err=cudaMalloc((void**)&(p),(bytes)); if(err) goto fail; }while(0)
    M(st->scal,sizeof(uint64_t)*9);
    M(st->stream,sizeof(MkInstr)*(size_t)SM*SL);
    M(st->sm_pc,sizeof(uint32_t)*SM); M(st->sm_epoch,sizeof(uint64_t)*SM); M(st->sm_halted,sizeof(uint8_t)*SM);
    M(st->dq,sizeof(uint64_t)*(size_t)SM*MDQ); M(st->dq_n,sizeof(int32_t)*SM);
    M(st->iq,sizeof(uint64_t)*(size_t)SM*MIQ); M(st->iq_n,sizeof(int32_t)*SM);
    M(st->rq,sizeof(uint64_t)*(size_t)SM*MRP); M(st->rq_n,sizeof(int32_t)*SM);
    M(st->r_used,sizeof(int32_t)*MDR); M(st->r_sm,sizeof(int32_t)*MDR); M(st->r_pc,sizeof(uint32_t)*MDR);
    M(st->r_epoch,sizeof(uint64_t)*MDR); M(st->r_uid,sizeof(uint64_t)*MDR); M(st->r_op,sizeof(int32_t)*MDR);
    M(st->r_status,sizeof(int32_t)*MDR); M(st->r_hz,sizeof(int32_t)*MDR); M(st->r_rhash,sizeof(uint64_t)*MDR);
    M(st->r_nread,sizeof(int32_t)*MDR); M(st->r_read,sizeof(int32_t)*(size_t)MDR*MK_MAX_RW);
    M(st->r_nwrite,sizeof(int32_t)*MDR); M(st->r_write,sizeof(int32_t)*(size_t)MDR*MK_MAX_RW);
    M(st->r_npage,sizeof(int32_t)*MDR); M(st->r_page,sizeof(int32_t)*(size_t)MDR*MK_MAX_RW);
    M(st->r_free_tile,sizeof(int32_t)*MDR); M(st->r_counter,sizeof(int32_t)*MDR);
    M(st->r_target,sizeof(uint64_t)*MDR); M(st->r_amount,sizeof(uint64_t)*MDR); M(st->r_cseed,sizeof(uint64_t)*MDR); M(st->r_pkey,sizeof(uint64_t)*MDR);
    M(st->r_dst,sizeof(int32_t)*MDR); M(st->r_taken,sizeof(int32_t)*MDR); M(st->r_fall,sizeof(int32_t)*MDR); M(st->r_lat,sizeof(uint64_t)*MDR);
    M(st->t_ver,sizeof(uint64_t)*TC); M(st->t_writer,sizeof(uint64_t)*TC); M(st->t_oldr,sizeof(uint64_t)*TC); M(st->t_rcount,sizeof(uint64_t)*TC);
    M(st->t_resident,sizeof(uint32_t)*TC); M(st->t_dirty,sizeof(uint8_t)*TC); M(st->t_freed,sizeof(uint8_t)*TC);
    M(st->p_state,sizeof(int32_t)*(size_t)SM*PG); M(st->p_key,sizeof(uint64_t)*(size_t)SM*PG); M(st->p_tile,sizeof(uint32_t)*(size_t)SM*PG);
    M(st->p_ver,sizeof(uint64_t)*(size_t)SM*PG); M(st->p_owner,sizeof(uint64_t)*(size_t)SM*PG);
    M(st->ctr,sizeof(uint64_t)*CC);
    M(st->u_kind,sizeof(int32_t)*MPU); M(st->u_sm,sizeof(int32_t)*MPU); M(st->u_ds,sizeof(uint64_t)*MPU);
    M(st->u_epoch,sizeof(uint64_t)*MPU); M(st->u_due,sizeof(uint64_t)*MPU); M(st->u_seq,sizeof(uint64_t)*MPU); M(st->u_n,sizeof(int32_t)*1);
    M(st->counts,sizeof(int64_t)*MK_COUNT_N);
    #undef M
    st->stream_count = SM*SL;
    err=mk_ref_do_reset(st,stream); if(err) goto fail;
    *state_out=st; return cudaSuccess;
fail:
    solution_destroy(st); return err?err:cudaErrorMemoryAllocation;
}

extern "C" cudaError_t solution_run(void* state,const MkRunSpec* run,const void* inputs_void,void* outputs_void,void* workspace,size_t workspace_bytes,cudaStream_t stream){
    (void)workspace;
    if(!state||!outputs_void) return cudaErrorInvalidValue;
    if(workspace_bytes<256) return cudaErrorInvalidValue;
    mk_ref::State* st=(mk_ref::State*)state;
    if(!mk_validate_run_spec(run,&st->spec)) return cudaErrorInvalidValue;
    const MkInputs* in=(const MkInputs*)inputs_void;
    if(in && in->stream && in->stream_count>0){
        size_t n=(size_t)st->spec.sm_count*st->spec.stream_len_per_sm;
        if((size_t)in->stream_count<n) n=(size_t)in->stream_count;
        cudaError_t ce=cudaMemcpyAsync(st->stream,in->stream,sizeof(MkInstr)*n,cudaMemcpyHostToDevice,stream);
        if(ce) return ce;
    }
    MkOutputs* out=(MkOutputs*)outputs_void;
    if(!out->counts||!out->op_index_out||!out->clock_out||!out->event_seq_out||!out->decoder_event_hash||!out->sm_stream_hash||!out->decoded_hash||!out->tile_scoreboard_hash||!out->page_hash||!out->pending_unit_hash||!out->counter_hash||!out->state_checksum) return cudaErrorInvalidValue;
    const MkProblemSpec& sp=st->spec;
    mk_ref_step<<<1,1,0,stream>>>(*st,
        sp.sm_count,sp.stream_len_per_sm,sp.decode_width,sp.issue_width,sp.retire_width,sp.tile_count,sp.page_count_per_sm,sp.counter_count,
        sp.max_decode_queue,sp.max_issue_queue,sp.max_replay,sp.max_decode_records,sp.max_pending_units,
        run->op_kind,run->a_sm,run->a_limit,run->a_max_units,run->a_counter_id,
        run->a_epoch_seed,run->a_delta,run->a_amount,
        out->counts,out->op_index_out,out->clock_out,out->event_seq_out,
        out->decoder_event_hash,out->sm_stream_hash,out->decoded_hash,out->tile_scoreboard_hash,out->page_hash,out->pending_unit_hash,out->counter_hash,out->state_checksum);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state,cudaStream_t stream){
    if(!state) return cudaErrorInvalidValue;
    return mk_ref_do_reset((mk_ref::State*)state,stream);
}

extern "C" void solution_destroy(void* state){
    if(!state) return;
    mk_ref::State* st=(mk_ref::State*)state;
    cudaFree(st->scal); cudaFree(st->stream);
    cudaFree(st->sm_pc); cudaFree(st->sm_epoch); cudaFree(st->sm_halted);
    cudaFree(st->dq); cudaFree(st->dq_n); cudaFree(st->iq); cudaFree(st->iq_n); cudaFree(st->rq); cudaFree(st->rq_n);
    cudaFree(st->r_used); cudaFree(st->r_sm); cudaFree(st->r_pc); cudaFree(st->r_epoch); cudaFree(st->r_uid);
    cudaFree(st->r_op); cudaFree(st->r_status); cudaFree(st->r_hz); cudaFree(st->r_rhash);
    cudaFree(st->r_nread); cudaFree(st->r_read); cudaFree(st->r_nwrite); cudaFree(st->r_write); cudaFree(st->r_npage); cudaFree(st->r_page);
    cudaFree(st->r_free_tile); cudaFree(st->r_counter); cudaFree(st->r_target); cudaFree(st->r_amount); cudaFree(st->r_cseed); cudaFree(st->r_pkey);
    cudaFree(st->r_dst); cudaFree(st->r_taken); cudaFree(st->r_fall); cudaFree(st->r_lat);
    cudaFree(st->t_ver); cudaFree(st->t_writer); cudaFree(st->t_oldr); cudaFree(st->t_rcount); cudaFree(st->t_resident); cudaFree(st->t_dirty); cudaFree(st->t_freed);
    cudaFree(st->p_state); cudaFree(st->p_key); cudaFree(st->p_tile); cudaFree(st->p_ver); cudaFree(st->p_owner);
    cudaFree(st->ctr);
    cudaFree(st->u_kind); cudaFree(st->u_sm); cudaFree(st->u_ds); cudaFree(st->u_epoch); cudaFree(st->u_due); cudaFree(st->u_seq); cudaFree(st->u_n);
    cudaFree(st->counts);
    free(st);
}
