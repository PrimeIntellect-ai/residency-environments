// PMPP_CANARY_55_37d36f61b6 -- held-out canary; MUST NOT appear in any submission
// file: lock_manager_deadlock_reference.cu
//
// Reference implementation of the T55 multi-granularity lock manager.
//
// Data-structure strategy (independent of naive + oracle):
//   * Persistent state mutated IN PLACE by a single <<<1,1>>> kernel per op.
//   * Transactions: flat slot array (capacity max_txns), linear scan by txn_id.
//   * Granted locks: flat slot array (capacity max_locks) of
//     (resource, txn, mode, explicit_count, auto_count, grant_seq,
//      last_convert_seq); linear scan keyed by (resource, txn). Dead entries are
//     reused.
//   * Wait queues: flat slot array (capacity max_waiters) of waiter records,
//     each tagged with its resource; per-resource ordering is materialised by
//     wait_seq selection at use time.
//   * Per-(txn,table) descendant counters live inside the transaction slot.
//   * Resource ordering (canonical = TABLE,PARTITION,ROW; release =
//     ROW,PARTITION,TABLE) is computed by comparator functions, not stored.
// All counters are recomputed from grants to stay self-consistent.

#include "lock_manager_deadlock_common.h"

#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LMD_FNV_OFFSET 1469598103934665603ULL
#define LMD_FNV_PRIME  1099511628211ULL
#define LMD_U64MAX (~0ULL)
#define LMD_U32MAX 0xFFFFFFFFu
// Bound on a single op's affected-resource set (initial releases + cascade
// appends). Kept modest to limit per-thread local memory.
#define LMD_REF_WK 256
#define LMD_REF_WK_SET 256

// -------- mode tables (device) --------
__device__ const uint8_t d_compat[6][6] = {
    { 1,1,1,1,1,1 },
    { 1,1,1,1,1,0 },
    { 1,1,1,0,0,0 },
    { 1,1,0,1,0,0 },
    { 1,1,0,0,0,0 },
    { 1,0,0,0,0,0 },
};
__device__ const uint8_t d_dom[6][6] = {
    { 1,0,0,0,0,0 },
    { 1,1,0,0,0,0 },
    { 1,1,1,0,0,0 },
    { 1,1,0,1,0,0 },
    { 1,1,1,1,1,0 },
    { 1,1,1,1,1,1 },
};
__device__ const uint8_t d_lub[6][6] = {
    { 0,1,2,3,4,5 },
    { 1,1,2,3,4,5 },
    { 2,2,2,4,4,5 },
    { 3,3,4,3,4,5 },
    { 4,4,4,4,4,5 },
    { 5,5,5,5,5,5 },
};

namespace lmd_ref {

struct State {
    LmdProblemSpec spec;

    // scalars: [0]=event_seq [1]=txn_seq_next [2]=request_seq_next
    //          [3]=wait_seq_next [4]=deadlock_seq_next [5]=op_index
    //          [6]=event_hash
    uint64_t* scal;            // 7 u64

    // transaction slots.
    uint8_t*  tx_live;         // [maxtxn]
    uint64_t* tx_id;           // [maxtxn]
    uint64_t* tx_seq;          // [maxtxn]
    uint32_t* tx_prio;         // [maxtxn]
    int32_t*  tx_status;       // [maxtxn]
    int32_t*  tx_bk_kind;      // [maxtxn] blocked res kind or 255
    int32_t*  tx_bk_t;         // [maxtxn]
    int32_t*  tx_bk_p;         // [maxtxn]
    int32_t*  tx_bk_r;         // [maxtxn]
    int32_t*  tx_bk_mode;      // [maxtxn]
    uint64_t* tx_wait_seq;     // [maxtxn]
    uint64_t* tx_locks;        // [maxtxn]
    uint64_t* tx_aborts;       // [maxtxn]
    // per-(txn,table) counters: 5 fields, layout slot*tables*5 + tab*5 + f.
    uint64_t* tx_ctr;          // [maxtxn*tables*5]

    // grant slots.
    uint8_t*  g_live;          // [maxlocks]
    int32_t*  g_kind;          // [maxlocks]
    int32_t*  g_t;             // [maxlocks]
    int32_t*  g_p;             // [maxlocks]
    int32_t*  g_r;             // [maxlocks]
    uint64_t* g_txn;           // [maxlocks]
    int32_t*  g_mode;          // [maxlocks]
    uint64_t* g_expl;          // [maxlocks]
    uint64_t* g_auto;          // [maxlocks]
    uint64_t* g_gseq;          // [maxlocks]
    uint64_t* g_cseq;          // [maxlocks]

    // waiter slots.
    uint8_t*  w_live;          // [maxwait]
    int32_t*  w_kind;          // [maxwait] resource of the waiter
    int32_t*  w_t;             // [maxwait]
    int32_t*  w_p;             // [maxwait]
    int32_t*  w_r;             // [maxwait]
    uint64_t* w_txn;           // [maxwait]
    int32_t*  w_reqmode;       // [maxwait]
    int32_t*  w_planidx;       // [maxwait]
    uint64_t* w_reqseq;        // [maxwait]
    uint8_t*  w_isconv;        // [maxwait]
    int32_t*  w_ot_kind;       // [maxwait] original target res
    int32_t*  w_ot_t;          // [maxwait]
    int32_t*  w_ot_p;          // [maxwait]
    int32_t*  w_ot_r;          // [maxwait]
    int32_t*  w_ot_mode;       // [maxwait]
    uint64_t* w_waitseq;       // [maxwait]

    int64_t*  counts;          // [LMD_COUNT_N]
};

}  // namespace lmd_ref

// -------------------------------------------------- device FNV
__device__ __forceinline__ uint64_t lmd_ref_fnv_b(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= LMD_FNV_PRIME; return h;
}
__device__ void lmd_ref_fnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*)p; uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = lmd_ref_fnv_b(v, q[i]);
    *h = v;
}
__device__ __forceinline__ void lmd_ref_h8(uint64_t* h, uint8_t v){ lmd_ref_fnv_bytes(h,&v,1); }
__device__ __forceinline__ void lmd_ref_h32(uint64_t* h, uint32_t v){ lmd_ref_fnv_bytes(h,&v,4); }
__device__ __forceinline__ void lmd_ref_h64(uint64_t* h, uint64_t v){ lmd_ref_fnv_bytes(h,&v,8); }

struct LmdRefCtx {
    lmd_ref::State s;
    int tables, parts, rows, maxtxn, maxlocks, maxwait, esc_thresh, max_dl;
};

// -------------------------------------------------- resource helpers
__device__ __forceinline__ int lmd_ref_kind_canon(int kind) {
    if (kind == LMD_TABLE) return 0;
    if (kind == LMD_PARTITION) return 1;
    return 2;
}
__device__ __forceinline__ int lmd_ref_kind_release(int kind) {
    if (kind == LMD_ROW) return 0;
    if (kind == LMD_PARTITION) return 1;
    return 2;
}
// returns <0,0,>0 for a<b,a==b,a>b under canonical order.
__device__ int lmd_ref_canon_cmp(int ak,int at,int ap,int ar,int bk,int bt,int bp,int br){
    int ra=lmd_ref_kind_canon(ak), rb=lmd_ref_kind_canon(bk);
    if(ra!=rb) return ra<rb?-1:1;
    if(at!=bt) return at<bt?-1:1;
    if(ap!=bp) return ap<bp?-1:1;
    if(ar!=br) return ar<br?-1:1;
    return 0;
}
__device__ int lmd_ref_release_cmp(int ak,int at,int ap,int ar,int bk,int bt,int bp,int br){
    int ra=lmd_ref_kind_release(ak), rb=lmd_ref_kind_release(bk);
    if(ra!=rb) return ra<rb?-1:1;
    if(at!=bt) return at<bt?-1:1;
    if(ap!=bp) return ap<bp?-1:1;
    if(ar!=br) return ar<br?-1:1;
    return 0;
}

// -------------------------------------------------- slot lookups
__device__ int lmd_ref_find_tx(LmdRefCtx& c, uint64_t id){
    for(int i=0;i<c.maxtxn;++i) if(c.s.tx_live[i] && c.s.tx_id[i]==id) return i;
    return -1;
}
__device__ int lmd_ref_count_tx(LmdRefCtx& c){
    int n=0; for(int i=0;i<c.maxtxn;++i) if(c.s.tx_live[i]) ++n; return n;
}
__device__ int lmd_ref_find_grant(LmdRefCtx& c,int kind,int t,int p,int r,uint64_t txn){
    for(int i=0;i<c.maxlocks;++i){
        if(!c.s.g_live[i]) continue;
        if(c.s.g_txn[i]!=txn) continue;
        if(c.s.g_kind[i]==kind && c.s.g_t[i]==t && c.s.g_p[i]==p && c.s.g_r[i]==r) return i;
    }
    return -1;
}
__device__ bool lmd_ref_grant_exists(LmdRefCtx& c,int gi){
    return gi>=0 && (c.s.g_expl[gi]!=0 || c.s.g_auto[gi]!=0);
}
__device__ int lmd_ref_alloc_grant(LmdRefCtx& c){
    for(int i=0;i<c.maxlocks;++i) if(!c.s.g_live[i]) return i;
    return -1;
}
__device__ int lmd_ref_alloc_wait(LmdRefCtx& c){
    for(int i=0;i<c.maxwait;++i) if(!c.s.w_live[i]) return i;
    return -1;
}

// -------------------------------------------------- emit
__device__ void lmd_ref_emit(LmdRefCtx& c, uint8_t kind, uint64_t txn,
                             int rk,int rt,int rp,int rr, int mode, uint64_t aux){
    uint64_t* sc=c.s.scal;
    uint64_t seq=sc[0];
    uint64_t h=sc[6];
    lmd_ref_h8(&h,kind);
    lmd_ref_h64(&h,seq);
    lmd_ref_h32(&h,(uint32_t)sc[5]);
    lmd_ref_h64(&h,txn);
    if(rk>=0){
        lmd_ref_h8(&h,(uint8_t)rk);
        lmd_ref_h32(&h, rt<0?LMD_U32MAX:(uint32_t)rt);
        lmd_ref_h32(&h, rp<0?LMD_U32MAX:(uint32_t)rp);
        lmd_ref_h32(&h, rr<0?LMD_U32MAX:(uint32_t)rr);
    } else {
        lmd_ref_h8(&h,(uint8_t)255);
        lmd_ref_h32(&h,LMD_U32MAX); lmd_ref_h32(&h,LMD_U32MAX); lmd_ref_h32(&h,LMD_U32MAX);
    }
    lmd_ref_h8(&h, mode<0?(uint8_t)255:(uint8_t)mode);
    lmd_ref_h64(&h,aux);
    sc[6]=h;
    sc[0]=seq+1;
}
__device__ __forceinline__ uint64_t lmd_ref_peek_seq(LmdRefCtx& c){ return c.s.scal[0]; }

// -------------------------------------------------- compatibility
__device__ bool lmd_ref_compat_others(LmdRefCtx& c,int kind,int t,int p,int r,uint64_t self,int req){
    for(int i=0;i<c.maxlocks;++i){
        if(!c.s.g_live[i]) continue;
        if(!(c.s.g_expl[i]!=0 || c.s.g_auto[i]!=0)) continue;
        if(c.s.g_kind[i]!=kind||c.s.g_t[i]!=t||c.s.g_p[i]!=p||c.s.g_r[i]!=r) continue;
        if(c.s.g_txn[i]==self) continue;
        if(!d_compat[req][c.s.g_mode[i]]) return false;
    }
    return true;
}

// -------------------------------------------------- counters
__device__ __forceinline__ uint64_t* lmd_ref_ctr(LmdRefCtx& c,int txslot,int tab){
    return &c.s.tx_ctr[((size_t)txslot*c.tables + tab)*5];
}
__device__ void lmd_ref_recompute_counters(LmdRefCtx& c,int txslot){
    uint64_t txn=c.s.tx_id[txslot];
    for(int tab=0;tab<c.tables;++tab){
        uint64_t* cc=lmd_ref_ctr(c,txslot,tab);
        cc[0]=0;cc[1]=0;cc[2]=0;cc[3]=0; // escalation_attempts cc[4] preserved
    }
    for(int i=0;i<c.maxlocks;++i){
        if(!c.s.g_live[i]) continue;
        if(c.s.g_txn[i]!=txn) continue;
        if(c.s.g_expl[i]==0) continue;
        if(c.s.g_kind[i]==LMD_TABLE) continue;
        int tab=c.s.g_t[i];
        int mode=c.s.g_mode[i];
        bool s_like=(mode==LMD_S||mode==LMD_SIX);
        bool x_like=(mode==LMD_X||mode==LMD_IX||mode==LMD_SIX);
        uint64_t* cc=lmd_ref_ctr(c,txslot,tab);
        if(c.s.g_kind[i]==LMD_ROW){ if(s_like)cc[0]++; if(x_like)cc[1]++; }
        else { if(s_like)cc[2]++; if(x_like)cc[3]++; }
    }
}
__device__ uint64_t lmd_ref_fine_count(LmdRefCtx& c,int txslot,int tab){
    uint64_t* cc=lmd_ref_ctr(c,txslot,tab);
    return cc[0]+cc[1]+cc[2]+cc[3];
}
__device__ bool lmd_ref_holds_xlike(LmdRefCtx& c,int txslot,int tab){
    uint64_t* cc=lmd_ref_ctr(c,txslot,tab);
    return (cc[1]+cc[3])>0;
}

// -------------------------------------------------- plan
// plan entries materialised into arrays. returns count.
struct LmdPlan { int kind[3],t[3],p[3],r[3],mode[3]; uint8_t is_target[3]; int n; };
__device__ LmdPlan lmd_ref_build_plan(int kind,int t,int p,int r,int mode){
    LmdPlan pl; pl.n=0;
    if(kind==LMD_TABLE){
        pl.kind[0]=LMD_TABLE;pl.t[0]=t;pl.p[0]=-1;pl.r[0]=-1;pl.mode[0]=mode;pl.is_target[0]=1;pl.n=1;
    } else if(kind==LMD_PARTITION){
        int anc=(mode==LMD_S||mode==LMD_IS)?LMD_IS:LMD_IX;
        pl.kind[0]=LMD_TABLE;pl.t[0]=t;pl.p[0]=-1;pl.r[0]=-1;pl.mode[0]=anc;pl.is_target[0]=0;
        pl.kind[1]=LMD_PARTITION;pl.t[1]=t;pl.p[1]=p;pl.r[1]=-1;pl.mode[1]=mode;pl.is_target[1]=1;
        pl.n=2;
    } else {
        int tanc=(mode==LMD_S)?LMD_IS:LMD_IX;
        int panc=(mode==LMD_S)?LMD_IS:LMD_IX;
        pl.kind[0]=LMD_TABLE;pl.t[0]=t;pl.p[0]=-1;pl.r[0]=-1;pl.mode[0]=tanc;pl.is_target[0]=0;
        pl.kind[1]=LMD_PARTITION;pl.t[1]=t;pl.p[1]=p;pl.r[1]=-1;pl.mode[1]=panc;pl.is_target[1]=0;
        pl.kind[2]=LMD_ROW;pl.t[2]=t;pl.p[2]=p;pl.r[2]=r;pl.mode[2]=mode;pl.is_target[2]=1;
        pl.n=3;
    }
    return pl;
}

// -------------------------------------------------- validation
__device__ bool lmd_ref_res_valid(LmdRefCtx& c,int kind,int t,int p,int r){
    if(kind==LMD_TABLE) return t>=0&&t<c.tables&&p<0&&r<0;
    if(kind==LMD_PARTITION) return t>=0&&t<c.tables&&p>=0&&p<c.parts&&r<0;
    if(kind==LMD_ROW) return t>=0&&t<c.tables&&p>=0&&p<c.parts&&r>=0&&r<c.rows;
    return false;
}
__device__ __forceinline__ bool lmd_ref_mode_req(int m){
    return m==LMD_IS||m==LMD_IX||m==LMD_S||m==LMD_SIX||m==LMD_X;
}
__device__ void lmd_ref_invalid(LmdRefCtx& c){
    c.s.counts[LMD_C_INVALID]+=1;
    lmd_ref_emit(c,LMD_EV_INVALID,0,-1,0,0,0,-1,0);
}

// -------------------------------------------------- grant/convert
__device__ void lmd_ref_grant_or_convert(LmdRefCtx& c,int txslot,int kind,int t,int p,int r,int reqmode,bool is_target){
    uint64_t txn=c.s.tx_id[txslot];
    int gi=lmd_ref_find_grant(c,kind,t,p,r,txn);
    if(!lmd_ref_grant_exists(c,gi)){
        if(gi<0) gi=lmd_ref_alloc_grant(c);
        c.s.g_live[gi]=1; c.s.g_kind[gi]=kind; c.s.g_t[gi]=t; c.s.g_p[gi]=p; c.s.g_r[gi]=r;
        c.s.g_txn[gi]=txn; c.s.g_mode[gi]=reqmode;
        c.s.g_expl[gi]=is_target?1:0; c.s.g_auto[gi]=is_target?0:1;
        c.s.g_gseq[gi]=lmd_ref_peek_seq(c); c.s.g_cseq[gi]=0;
        c.s.tx_locks[txslot]+=1;
        if(is_target) lmd_ref_recompute_counters(c,txslot);
        lmd_ref_emit(c,LMD_EV_LOCK_GRANT,txn,kind,t,p,r,reqmode,0);
    } else {
        int nm=d_lub[c.s.g_mode[gi]][reqmode];
        c.s.g_mode[gi]=nm;
        if(is_target) c.s.g_expl[gi]+=1; else c.s.g_auto[gi]+=1;
        c.s.g_cseq[gi]=lmd_ref_peek_seq(c);
        if(is_target) lmd_ref_recompute_counters(c,txslot);
        c.s.tx_locks[txslot]+=1;
        lmd_ref_emit(c,LMD_EV_LOCK_CONVERT,txn,kind,t,p,r,nm,0);
    }
}

// forward decls
__device__ void lmd_ref_wake_resources(LmdRefCtx& c,int* ak,int* at,int* ap,int* ar,int n);

// -------------------------------------------------- escalation
__device__ void lmd_ref_try_escalate(LmdRefCtx& c,int txslot,int tab){
    uint64_t txn=c.s.tx_id[txslot];
    lmd_ref_ctr(c,txslot,tab)[4]+=1; // escalation_attempts
    int target=lmd_ref_holds_xlike(c,txslot,tab)?LMD_X:LMD_S;
    if(!lmd_ref_compat_others(c,LMD_TABLE,tab,-1,-1,txn,target)){
        c.s.counts[LMD_C_ESCALATE_BLOCKED]+=1;
        lmd_ref_emit(c,LMD_EV_ESCALATE_BLOCKED,txn,LMD_TABLE,tab,-1,-1,target,0);
        return;
    }
    int gi=lmd_ref_find_grant(c,LMD_TABLE,tab,-1,-1,txn);
    if(!lmd_ref_grant_exists(c,gi)){
        if(gi<0) gi=lmd_ref_alloc_grant(c);
        c.s.g_live[gi]=1;c.s.g_kind[gi]=LMD_TABLE;c.s.g_t[gi]=tab;c.s.g_p[gi]=-1;c.s.g_r[gi]=-1;
        c.s.g_txn[gi]=txn;c.s.g_mode[gi]=target;c.s.g_expl[gi]=1;c.s.g_auto[gi]=0;
        c.s.g_gseq[gi]=lmd_ref_peek_seq(c);c.s.g_cseq[gi]=0;
        c.s.tx_locks[txslot]+=1;
        lmd_ref_emit(c,LMD_EV_ESCALATE_GRANT,txn,LMD_TABLE,tab,-1,-1,target,0);
    } else {
        int nm=d_lub[c.s.g_mode[gi]][target];
        c.s.g_mode[gi]=nm; c.s.g_expl[gi]+=1; c.s.g_cseq[gi]=lmd_ref_peek_seq(c);
        c.s.tx_locks[txslot]+=1;
        lmd_ref_emit(c,LMD_EV_ESCALATE_GRANT,txn,LMD_TABLE,tab,-1,-1,nm,0);
    }
    // release every non-table grant under tab in canonical order (cursor based).
    int ak[LMD_REF_WK_SET],at[LMD_REF_WK_SET],ap[LMD_REF_WK_SET],ar[LMD_REF_WK_SET]; int an=0;
    bool hp=false; int pk=0,pt=0,pp=0,pr=0;
    for(;;){
        // next live non-table grant of txn under tab in canonical order after prev.
        int sel=-1;
        for(int i=0;i<c.maxlocks;++i){
            if(!c.s.g_live[i]) continue;
            if(!(c.s.g_expl[i]!=0||c.s.g_auto[i]!=0)) continue;
            if(c.s.g_txn[i]!=txn) continue;
            if(c.s.g_kind[i]==LMD_TABLE) continue;
            if(c.s.g_t[i]!=tab) continue;
            if(hp){
                int cmp=lmd_ref_canon_cmp(c.s.g_kind[i],c.s.g_t[i],c.s.g_p[i],c.s.g_r[i],pk,pt,pp,pr);
                if(cmp<=0) continue;
            }
            if(sel<0){ sel=i; continue; }
            int cmp=lmd_ref_canon_cmp(c.s.g_kind[i],c.s.g_t[i],c.s.g_p[i],c.s.g_r[i],
                                      c.s.g_kind[sel],c.s.g_t[sel],c.s.g_p[sel],c.s.g_r[sel]);
            if(cmp<0) sel=i;
        }
        if(sel<0) break;
        int i=sel;
        bool had_expl=(c.s.g_expl[i]!=0);
        int rk=c.s.g_kind[i],rt=c.s.g_t[i],rp=c.s.g_p[i],rr=c.s.g_r[i];
        pk=rk;pt=rt;pp=rp;pr=rr;hp=true;
        c.s.g_expl[i]=0;c.s.g_auto[i]=0;c.s.g_mode[i]=LMD_NL;c.s.g_live[i]=0;
        if(had_expl){
            c.s.counts[LMD_C_ESCALATE_RELEASES]+=1;
            lmd_ref_emit(c,LMD_EV_ESCALATE_RELEASE,txn,rk,rt,rp,rr,-1,0);
        }
        if(an<LMD_REF_WK_SET){ ak[an]=rk;at[an]=rt;ap[an]=rp;ar[an]=rr;an++; }
    }
    lmd_ref_recompute_counters(c,txslot);
    if(an<LMD_REF_WK_SET){ ak[an]=LMD_TABLE;at[an]=tab;ap[an]=-1;ar[an]=-1;an++; }
    lmd_ref_wake_resources(c,ak,at,ap,ar,an);
}

// -------------------------------------------------- enqueue waiter
__device__ void lmd_ref_enqueue(LmdRefCtx& c,int txslot,int kind,int t,int p,int r,int reqmode,
                                int planidx,uint64_t reqseq,uint8_t isconv,
                                int otk,int ott,int otp,int otr,int otmode){
    int wi=lmd_ref_alloc_wait(c);
    c.s.w_live[wi]=1;
    c.s.w_kind[wi]=kind;c.s.w_t[wi]=t;c.s.w_p[wi]=p;c.s.w_r[wi]=r;
    c.s.w_txn[wi]=c.s.tx_id[txslot];
    c.s.w_reqmode[wi]=reqmode;c.s.w_planidx[wi]=planidx;c.s.w_reqseq[wi]=reqseq;
    c.s.w_isconv[wi]=isconv;
    c.s.w_ot_kind[wi]=otk;c.s.w_ot_t[wi]=ott;c.s.w_ot_p[wi]=otp;c.s.w_ot_r[wi]=otr;c.s.w_ot_mode[wi]=otmode;
    uint64_t ws=c.s.scal[3]++; c.s.w_waitseq[wi]=ws;
    c.s.tx_status[txslot]=LMD_WAITING;
    c.s.tx_bk_kind[txslot]=kind;c.s.tx_bk_t[txslot]=t;c.s.tx_bk_p[txslot]=p;c.s.tx_bk_r[txslot]=r;
    c.s.tx_bk_mode[txslot]=reqmode;
    c.s.tx_wait_seq[txslot]=ws;
}

// -------------------------------------------------- LOCK
__device__ void lmd_ref_op_lock(LmdRefCtx& c,uint64_t txn,int kind,int t,int p,int r,int mode){
    int ts=lmd_ref_find_tx(c,txn);
    if(ts<0||c.s.tx_status[ts]!=LMD_ACTIVE||!lmd_ref_res_valid(c,kind,t,p,r)||!lmd_ref_mode_req(mode)){
        lmd_ref_invalid(c); return;
    }
    uint64_t reqseq=c.s.scal[2]++;
    LmdPlan pl=lmd_ref_build_plan(kind,t,p,r,mode);
    for(int pi=0;pi<pl.n;++pi){
        int ek=pl.kind[pi],et=pl.t[pi],ep=pl.p[pi],er=pl.r[pi],em=pl.mode[pi];
        bool is_tgt=pl.is_target[pi];
        int gi=lmd_ref_find_grant(c,ek,et,ep,er,txn);
        if(lmd_ref_grant_exists(c,gi) && d_dom[c.s.g_mode[gi]][em]){
            if(is_tgt) c.s.g_expl[gi]+=1; else c.s.g_auto[gi]+=1;
            c.s.counts[LMD_C_LOCK_REENTERS]+=1;
            lmd_ref_emit(c,LMD_EV_LOCK_REENTER,txn,ek,et,ep,er,em,0);
            continue;
        }
        int eff=em;
        if(lmd_ref_grant_exists(c,gi)) eff=d_lub[c.s.g_mode[gi]][em];
        if(lmd_ref_compat_others(c,ek,et,ep,er,txn,eff)){
            lmd_ref_grant_or_convert(c,ts,ek,et,ep,er,em,is_tgt);
            if(is_tgt && (ek==LMD_ROW||ek==LMD_PARTITION)){
                if(lmd_ref_fine_count(c,ts,et)>=(uint64_t)c.esc_thresh) lmd_ref_try_escalate(c,ts,et);
            }
        } else {
            lmd_ref_enqueue(c,ts,ek,et,ep,er,em,pi,reqseq,0,kind,t,p,r,mode);
            c.s.counts[LMD_C_LOCK_WAITS]+=1;
            lmd_ref_emit(c,LMD_EV_LOCK_WAIT,txn,ek,et,ep,er,em,0);
            return;
        }
    }
}

// -------------------------------------------------- intent for descendants
__device__ int lmd_ref_intent(LmdRefCtx& c,int txslot,int tab){
    uint64_t* cc=lmd_ref_ctr(c,txslot,tab);
    bool x_like=(cc[1]+cc[3])>0;
    return x_like?LMD_IX:LMD_IS;
}

// -------------------------------------------------- UNLOCK
__device__ void lmd_ref_op_unlock(LmdRefCtx& c,uint64_t txn,int kind,int t,int p,int r){
    int ts=lmd_ref_find_tx(c,txn);
    if(ts<0){ lmd_ref_invalid(c); return; }
    int gi=lmd_ref_find_grant(c,kind,t,p,r,txn);
    if(!lmd_ref_grant_exists(c,gi)||c.s.g_expl[gi]==0){ lmd_ref_invalid(c); return; }
    c.s.g_expl[gi]-=1;
    if(c.s.g_expl[gi]==0){
        if(c.s.g_auto[gi]==0){
            c.s.g_mode[gi]=LMD_NL; c.s.g_live[gi]=0;
            if(kind==LMD_ROW||kind==LMD_PARTITION) lmd_ref_recompute_counters(c,ts);
        } else {
            if(kind==LMD_ROW||kind==LMD_PARTITION) lmd_ref_recompute_counters(c,ts);
            c.s.g_mode[gi]=lmd_ref_intent(c,ts,t);
        }
    } else {
        if(kind==LMD_ROW||kind==LMD_PARTITION) lmd_ref_recompute_counters(c,ts);
    }
    c.s.counts[LMD_C_LOCK_RELEASES]+=1;
    lmd_ref_emit(c,LMD_EV_LOCK_RELEASE,txn,kind,t,p,r,-1,0);
    int ak[1]={kind},at[1]={t},ap[1]={p},ar[1]={r};
    lmd_ref_wake_resources(c,ak,at,ap,ar,1);
}

// find the next live grant of txn in RELEASE order strictly after prev (cursor).
// have_prev=false to get the first. Returns slot index or -1.
__device__ int lmd_ref_next_grant_release(LmdRefCtx& c,uint64_t txn,bool have_prev,
                                          int pk,int pt,int pp,int pr){
    int sel=-1;
    for(int i=0;i<c.maxlocks;++i){
        if(!c.s.g_live[i]) continue;
        if(!(c.s.g_expl[i]!=0||c.s.g_auto[i]!=0)) continue;
        if(c.s.g_txn[i]!=txn) continue;
        if(have_prev){
            int cmp=lmd_ref_release_cmp(c.s.g_kind[i],c.s.g_t[i],c.s.g_p[i],c.s.g_r[i],pk,pt,pp,pr);
            if(cmp<=0) continue;
        }
        if(sel<0){ sel=i; continue; }
        int cmp=lmd_ref_release_cmp(c.s.g_kind[i],c.s.g_t[i],c.s.g_p[i],c.s.g_r[i],
                                    c.s.g_kind[sel],c.s.g_t[sel],c.s.g_p[sel],c.s.g_r[sel]);
        if(cmp<0) sel=i;
    }
    return sel;
}

// -------------------------------------------------- UNLOCK_ALL
__device__ void lmd_ref_op_unlock_all(LmdRefCtx& c,uint64_t txn){
    int ts=lmd_ref_find_tx(c,txn);
    if(ts<0){ lmd_ref_invalid(c); return; }
    int ak[LMD_REF_WK_SET],at[LMD_REF_WK_SET],ap[LMD_REF_WK_SET],ar[LMD_REF_WK_SET]; int an=0;
    bool hp=false; int pk=0,pt=0,pp=0,pr=0;
    for(;;){
        int i=lmd_ref_next_grant_release(c,txn,hp,pk,pt,pp,pr);
        if(i<0) break;
        int rk=c.s.g_kind[i],rt=c.s.g_t[i],rp=c.s.g_p[i],rr=c.s.g_r[i];
        pk=rk;pt=rt;pp=rp;pr=rr;hp=true;
        c.s.g_expl[i]=0;c.s.g_auto[i]=0;c.s.g_mode[i]=LMD_NL;c.s.g_live[i]=0;
        c.s.counts[LMD_C_LOCK_RELEASE_ALL]+=1;
        lmd_ref_emit(c,LMD_EV_LOCK_RELEASE_ALL,txn,rk,rt,rp,rr,-1,0);
        if(an<LMD_REF_WK_SET){ ak[an]=rk;at[an]=rt;ap[an]=rp;ar[an]=rr;an++; }
    }
    c.s.tx_live[ts]=0;  // remove transaction
    lmd_ref_wake_resources(c,ak,at,ap,ar,an);
}

// -------------------------------------------------- CONVERT
__device__ void lmd_ref_op_convert(LmdRefCtx& c,uint64_t txn,int kind,int t,int p,int r,int newmode){
    int ts=lmd_ref_find_tx(c,txn);
    if(ts<0||c.s.tx_status[ts]!=LMD_ACTIVE||!lmd_ref_res_valid(c,kind,t,p,r)||!lmd_ref_mode_req(newmode)){
        lmd_ref_invalid(c); return;
    }
    int gi=lmd_ref_find_grant(c,kind,t,p,r,txn);
    if(!lmd_ref_grant_exists(c,gi)){ lmd_ref_invalid(c); return; }
    if(d_dom[c.s.g_mode[gi]][newmode]){
        c.s.counts[LMD_C_CONVERT_NOOPS]+=1;
        lmd_ref_emit(c,LMD_EV_CONVERT_NOOP,txn,kind,t,p,r,newmode,0);
        return;
    }
    int eff=d_lub[c.s.g_mode[gi]][newmode];
    if(lmd_ref_compat_others(c,kind,t,p,r,txn,eff)){
        c.s.g_mode[gi]=eff; c.s.g_cseq[gi]=lmd_ref_peek_seq(c);
        if(kind==LMD_ROW||kind==LMD_PARTITION) lmd_ref_recompute_counters(c,ts);
        c.s.tx_locks[ts]+=1;
        c.s.counts[LMD_C_LOCK_CONVERTS]+=1;
        lmd_ref_emit(c,LMD_EV_LOCK_CONVERT,txn,kind,t,p,r,eff,0);
    } else {
        uint64_t rs=c.s.scal[2]++;
        lmd_ref_enqueue(c,ts,kind,t,p,r,newmode,0,rs,1,kind,t,p,r,newmode);
        c.s.counts[LMD_C_CONVERT_WAITS]+=1;
        lmd_ref_emit(c,LMD_EV_CONVERT_WAIT,txn,kind,t,p,r,newmode,0);
    }
}

// -------------------------------------------------- BEGIN
__device__ void lmd_ref_op_begin(LmdRefCtx& c,uint64_t txn,uint32_t prio){
    if(lmd_ref_find_tx(c,txn)>=0 || lmd_ref_count_tx(c)>=c.maxtxn){ lmd_ref_invalid(c); return; }
    int ts=-1; for(int i=0;i<c.maxtxn;++i) if(!c.s.tx_live[i]){ ts=i; break; }
    if(ts<0){ lmd_ref_invalid(c); return; }
    c.s.tx_live[ts]=1; c.s.tx_id[ts]=txn; c.s.tx_seq[ts]=c.s.scal[1]++;
    c.s.tx_prio[ts]=prio; c.s.tx_status[ts]=LMD_ACTIVE;
    c.s.tx_bk_kind[ts]=255; c.s.tx_bk_t[ts]=-1; c.s.tx_bk_p[ts]=-1; c.s.tx_bk_r[ts]=-1; c.s.tx_bk_mode[ts]=255;
    c.s.tx_wait_seq[ts]=LMD_U64MAX; c.s.tx_locks[ts]=0; c.s.tx_aborts[ts]=0;
    for(int tab=0;tab<c.tables;++tab){ uint64_t* cc=lmd_ref_ctr(c,ts,tab); cc[0]=cc[1]=cc[2]=cc[3]=cc[4]=0; }
    c.s.counts[LMD_C_TXN_BEGUN]+=1;
    lmd_ref_emit(c,LMD_EV_TXN_BEGIN,txn,-1,0,0,0,-1,prio);
}

// -------------------------------------------------- wake
// find head waiter (min wait_seq) on resource among live waiters.
__device__ int lmd_ref_wait_head(LmdRefCtx& c,int kind,int t,int p,int r){
    int best=-1; uint64_t bws=LMD_U64MAX;
    for(int i=0;i<c.maxwait;++i){
        if(!c.s.w_live[i]) continue;
        if(c.s.w_kind[i]!=kind||c.s.w_t[i]!=t||c.s.w_p[i]!=p||c.s.w_r[i]!=r) continue;
        if(best<0||c.s.w_waitseq[i]<bws){ best=i; bws=c.s.w_waitseq[i]; }
    }
    return best;
}

__device__ void lmd_ref_grant_woken(LmdRefCtx& c,int txslot,int kind,int t,int p,int r,int reqmode,bool is_target){
    uint64_t txn=c.s.tx_id[txslot];
    int gi=lmd_ref_find_grant(c,kind,t,p,r,txn);
    if(!lmd_ref_grant_exists(c,gi)){
        if(gi<0) gi=lmd_ref_alloc_grant(c);
        c.s.g_live[gi]=1;c.s.g_kind[gi]=kind;c.s.g_t[gi]=t;c.s.g_p[gi]=p;c.s.g_r[gi]=r;
        c.s.g_txn[gi]=txn;c.s.g_mode[gi]=reqmode;
        c.s.g_expl[gi]=is_target?1:0;c.s.g_auto[gi]=is_target?0:1;
        c.s.g_gseq[gi]=lmd_ref_peek_seq(c);c.s.g_cseq[gi]=0;
        c.s.tx_locks[txslot]+=1;
        if(is_target) lmd_ref_recompute_counters(c,txslot);
    } else {
        int nm=d_lub[c.s.g_mode[gi]][reqmode];
        c.s.g_mode[gi]=nm;
        if(is_target) c.s.g_expl[gi]+=1; else c.s.g_auto[gi]+=1;
        c.s.g_cseq[gi]=lmd_ref_peek_seq(c);
        if(is_target) lmd_ref_recompute_counters(c,txslot);
        c.s.tx_locks[txslot]+=1;
    }
}

// continue plan from index `from`; collects newly affected resources.
__device__ void lmd_ref_continue_plan(LmdRefCtx& c,int txslot,LmdPlan pl,int from,
                                       uint64_t reqseq,int otk,int ott,int otp,int otr,int otmode,
                                       int* ak,int* at,int* ap,int* ar,int* an){
    uint64_t txn=c.s.tx_id[txslot];
    for(int pi=from;pi<pl.n;++pi){
        int ek=pl.kind[pi],et=pl.t[pi],ep=pl.p[pi],er=pl.r[pi],em=pl.mode[pi];
        bool is_tgt=pl.is_target[pi];
        int gi=lmd_ref_find_grant(c,ek,et,ep,er,txn);
        if(lmd_ref_grant_exists(c,gi)&&d_dom[c.s.g_mode[gi]][em]){
            if(is_tgt) c.s.g_expl[gi]+=1; else c.s.g_auto[gi]+=1;
            c.s.counts[LMD_C_LOCK_REENTERS]+=1;
            lmd_ref_emit(c,LMD_EV_LOCK_REENTER,txn,ek,et,ep,er,em,0);
            continue;
        }
        int eff=em;
        if(lmd_ref_grant_exists(c,gi)) eff=d_lub[c.s.g_mode[gi]][em];
        if(lmd_ref_compat_others(c,ek,et,ep,er,txn,eff)){
            lmd_ref_grant_or_convert(c,txslot,ek,et,ep,er,em,is_tgt);
            ak[*an]=ek;at[*an]=et;ap[*an]=ep;ar[*an]=er;(*an)++;
            if(is_tgt&&(ek==LMD_ROW||ek==LMD_PARTITION)){
                if(lmd_ref_fine_count(c,txslot,et)>=(uint64_t)c.esc_thresh) lmd_ref_try_escalate(c,txslot,et);
            }
        } else {
            int wi=lmd_ref_alloc_wait(c);
            c.s.w_live[wi]=1;c.s.w_kind[wi]=ek;c.s.w_t[wi]=et;c.s.w_p[wi]=ep;c.s.w_r[wi]=er;
            c.s.w_txn[wi]=txn;c.s.w_reqmode[wi]=em;c.s.w_planidx[wi]=pi;c.s.w_reqseq[wi]=reqseq;
            c.s.w_isconv[wi]=0;c.s.w_ot_kind[wi]=otk;c.s.w_ot_t[wi]=ott;c.s.w_ot_p[wi]=otp;c.s.w_ot_r[wi]=otr;c.s.w_ot_mode[wi]=otmode;
            uint64_t ws=c.s.scal[3]++; c.s.w_waitseq[wi]=ws;
            c.s.tx_status[txslot]=LMD_WAITING;
            c.s.tx_bk_kind[txslot]=ek;c.s.tx_bk_t[txslot]=et;c.s.tx_bk_p[txslot]=ep;c.s.tx_bk_r[txslot]=er;
            c.s.tx_bk_mode[txslot]=em; c.s.tx_wait_seq[txslot]=ws;
            c.s.counts[LMD_C_WAKE_REBLOCKS]+=1;
            lmd_ref_emit(c,LMD_EV_WAKE_REBLOCK,txn,ek,et,ep,er,em,0);
            return;
        }
    }
}

__device__ void lmd_ref_wake_one(LmdRefCtx& c,int kind,int t,int p,int r,
                                  int* ak,int* at,int* ap,int* ar,int* an){
    for(;;){
        int wi=lmd_ref_wait_head(c,kind,t,p,r);
        if(wi<0) return;
        uint64_t htxn=c.s.w_txn[wi];
        int ts=lmd_ref_find_tx(c,htxn);
        if(ts<0){ c.s.w_live[wi]=0; continue; }  // stale
        int reqmode=c.s.w_reqmode[wi];
        int eff=reqmode;
        int gi=lmd_ref_find_grant(c,kind,t,p,r,htxn);
        if(lmd_ref_grant_exists(c,gi)) eff=d_lub[c.s.g_mode[gi]][reqmode];
        if(!lmd_ref_compat_others(c,kind,t,p,r,htxn,eff)) return; // head blocks -> stop
        // grant head.
        uint8_t isconv=c.s.w_isconv[wi];
        int planidx=c.s.w_planidx[wi];
        uint64_t reqseq=c.s.w_reqseq[wi];
        int otk=c.s.w_ot_kind[wi],ott=c.s.w_ot_t[wi],otp=c.s.w_ot_p[wi],otr=c.s.w_ot_r[wi],otmode=c.s.w_ot_mode[wi];
        LmdPlan pl=lmd_ref_build_plan(otk,ott,otp,otr,otmode);
        bool is_tgt = isconv ? true : (planidx+1==pl.n);
        c.s.w_live[wi]=0;  // dequeue
        lmd_ref_grant_woken(c,ts,kind,t,p,r,reqmode,is_tgt);
        c.s.counts[LMD_C_WAKE_GRANTS]+=1;
        lmd_ref_emit(c,LMD_EV_WAKE_GRANT,htxn,kind,t,p,r,reqmode,0);
        c.s.tx_status[ts]=LMD_ACTIVE;
        c.s.tx_bk_kind[ts]=255;c.s.tx_bk_t[ts]=-1;c.s.tx_bk_p[ts]=-1;c.s.tx_bk_r[ts]=-1;c.s.tx_bk_mode[ts]=255;
        c.s.tx_wait_seq[ts]=LMD_U64MAX;
        if(is_tgt&&(kind==LMD_ROW||kind==LMD_PARTITION)){
            if(lmd_ref_fine_count(c,ts,t)>=(uint64_t)c.esc_thresh) lmd_ref_try_escalate(c,ts,t);
        }
        if(!isconv && planidx+1<pl.n){
            lmd_ref_continue_plan(c,ts,pl,planidx+1,reqseq,otk,ott,otp,otr,otmode,ak,at,ap,ar,an);
        }
    }
}

// wake a set of resources; expands the affected set as wake cascades add more.
// Copies the (possibly small) initial list into an internal large buffer so
// continue_plan can append without overflowing the caller's array.
//
// Wake processing model (identical across all 3 implementations):
//   * sort+dedup the INITIAL resource set into canonical order.
//   * process by growing index ri from 0; wake_one may append new resources
//     (from continue_plan grants) at the tail WITHOUT sort or dedup.
//   * stop when ri reaches the (growing) end.
__device__ void lmd_ref_wake_resources(LmdRefCtx& c,int* ik,int* it,int* ip,int* ir,int n){
    int ak[LMD_REF_WK],at[LMD_REF_WK],ap[LMD_REF_WK],ar[LMD_REF_WK];
    int total=0;
    for(int i=0;i<n && total<LMD_REF_WK;++i){
        ak[total]=ik[i];at[total]=it[i];ap[total]=ip[i];ar[total]=ir[i];total++;
    }
    // sort initial canonical (selection sort).
    for(int a=0;a<total;++a){
        int best=a;
        for(int b=a+1;b<total;++b)
            if(lmd_ref_canon_cmp(ak[b],at[b],ap[b],ar[b],ak[best],at[best],ap[best],ar[best])<0) best=b;
        int t0=ak[a];ak[a]=ak[best];ak[best]=t0;
        int t1=at[a];at[a]=at[best];at[best]=t1;
        int t2=ap[a];ap[a]=ap[best];ap[best]=t2;
        int t3=ar[a];ar[a]=ar[best];ar[best]=t3;
    }
    // dedup initial (adjacent after sort).
    int w=0;
    for(int i=0;i<total;++i){
        if(w>0 && ak[i]==ak[w-1]&&at[i]==at[w-1]&&ap[i]==ap[w-1]&&ar[i]==ar[w-1]) continue;
        ak[w]=ak[i];at[w]=at[i];ap[w]=ap[i];ar[w]=ar[i];w++;
    }
    total=w;
    for(int ri=0; ri<total; ++ri){
        lmd_ref_wake_one(c,ak[ri],at[ri],ap[ri],ar[ri],ak,at,ap,ar,&total);
    }
}

// -------------------------------------------------- deadlock
// successors of waiting txn u (slot) -> blocker txn ids, sorted asc, deduped.
// returns count, fills out[].
__device__ int lmd_ref_succ(LmdRefCtx& c,int uslot,uint64_t* out,int cap){
    if(c.s.tx_status[uslot]!=LMD_WAITING||c.s.tx_bk_kind[uslot]==255) return 0;
    int kind=c.s.tx_bk_kind[uslot],t=c.s.tx_bk_t[uslot],p=c.s.tx_bk_p[uslot],r=c.s.tx_bk_r[uslot];
    int req=c.s.tx_bk_mode[uslot];
    uint64_t u=c.s.tx_id[uslot];
    int n=0;
    for(int i=0;i<c.maxlocks;++i){
        if(!c.s.g_live[i]) continue;
        if(!(c.s.g_expl[i]!=0||c.s.g_auto[i]!=0)) continue;
        if(c.s.g_kind[i]!=kind||c.s.g_t[i]!=t||c.s.g_p[i]!=p||c.s.g_r[i]!=r) continue;
        uint64_t b=c.s.g_txn[i];
        if(b==u) continue;
        if(d_compat[req][c.s.g_mode[i]]) continue;
        // insert b if not already present.
        bool dup=false; for(int j=0;j<n;++j) if(out[j]==b){dup=true;break;}
        if(!dup && n<cap) out[n++]=b;
    }
    // sort asc.
    for(int a=0;a<n;++a){ int best=a; for(int b=a+1;b<n;++b) if(out[b]<out[best]) best=b; uint64_t tmp=out[a];out[a]=out[best];out[best]=tmp; }
    return n;
}

// DFS for first cycle from start (txn id). Returns cycle length, fills cyc[].
// Successors are recomputed on demand by index (no per-node child cache) so the
// DFS keeps only a path stack and a per-frame child cursor.
__device__ int lmd_ref_dfs_cycle(LmdRefCtx& c,uint64_t start,uint64_t* cyc,int cap){
    uint64_t path[LMD_MAX_TXNS+1]; int pn=0;
    int fidx[LMD_MAX_TXNS+1];      // child cursor per frame
    uint64_t buf[LMD_MAX_TXNS];    // scratch for one node's successors
    int ss=lmd_ref_find_tx(c,start);
    if(ss<0) return 0;
    path[pn++]=start; fidx[0]=0;
    while(pn>0){
        int top=pn-1;
        int us=lmd_ref_find_tx(c,path[top]);
        int nch = (us<0)?0:lmd_ref_succ(c,us,buf,LMD_MAX_TXNS);
        if(fidx[top]<nch){
            uint64_t nxt=buf[fidx[top]++];
            int pos=-1; for(int i=0;i<pn;++i) if(path[i]==nxt){pos=i;break;}
            if(pos>=0){
                int len=0; for(int i=pos;i<pn && len<cap;++i) cyc[len++]=path[i];
                return len;
            }
            if(lmd_ref_find_tx(c,nxt)<0) continue;
            path[pn]=nxt; fidx[pn]=0; pn++;
        } else {
            pn--;
        }
    }
    return 0;
}

__device__ int lmd_ref_find_cycle(LmdRefCtx& c,uint64_t* cyc,int cap){
    // waiting txn ids ascending.
    uint64_t wids[LMD_MAX_TXNS]; int wn=0;
    for(int i=0;i<c.maxtxn;++i){
        if(c.s.tx_live[i]&&c.s.tx_status[i]==LMD_WAITING&&c.s.tx_bk_kind[i]!=255) wids[wn++]=c.s.tx_id[i];
    }
    for(int a=0;a<wn;++a){ int best=a; for(int b=a+1;b<wn;++b) if(wids[b]<wids[best]) best=b; uint64_t tmp=wids[a];wids[a]=wids[best];wids[best]=tmp; }
    for(int i=0;i<wn;++i){
        int len=lmd_ref_dfs_cycle(c,wids[i],cyc,cap);
        if(len>0) return len;
    }
    return 0;
}

__device__ void lmd_ref_abort_victim(LmdRefCtx& c,uint64_t victim){
    int vs=lmd_ref_find_tx(c,victim);
    if(vs<0) return;
    // remove from wait queue if waiting.
    if(c.s.tx_status[vs]==LMD_WAITING&&c.s.tx_bk_kind[vs]!=255){
        for(int i=0;i<c.maxwait;++i){
            if(c.s.w_live[i]&&c.s.w_txn[i]==victim&&c.s.w_waitseq[i]==c.s.tx_wait_seq[vs]){ c.s.w_live[i]=0; break; }
        }
    }
    // release all in RELEASE order (cursor based).
    int ak[LMD_REF_WK_SET],at[LMD_REF_WK_SET],ap[LMD_REF_WK_SET],ar[LMD_REF_WK_SET]; int an=0;
    bool hp=false; int pk=0,pt=0,pp=0,pr=0;
    for(;;){
        int i=lmd_ref_next_grant_release(c,victim,hp,pk,pt,pp,pr);
        if(i<0) break;
        int rk=c.s.g_kind[i],rt=c.s.g_t[i],rp=c.s.g_p[i],rr=c.s.g_r[i];
        pk=rk;pt=rt;pp=rp;pr=rr;hp=true;
        c.s.g_expl[i]=0;c.s.g_auto[i]=0;c.s.g_mode[i]=LMD_NL;c.s.g_live[i]=0;
        c.s.counts[LMD_C_VICTIM_RELEASES]+=1;
        lmd_ref_emit(c,LMD_EV_VICTIM_RELEASE,victim,rk,rt,rp,rr,-1,0);
        if(an<LMD_REF_WK_SET){ ak[an]=rk;at[an]=rt;ap[an]=rp;ar[an]=rr;an++; }
    }
    c.s.tx_live[vs]=0;
    c.s.counts[LMD_C_DEADLOCK_ABORTS]+=1;
    lmd_ref_emit(c,LMD_EV_DEADLOCK_ABORT,victim,-1,0,0,0,-1,0);
    c.s.scal[4]++; // deadlock_seq_next
    lmd_ref_wake_resources(c,ak,at,ap,ar,an);
}

__device__ void lmd_ref_op_detect(LmdRefCtx& c,int limit){
    if(limit==0) return;
    int repeat=limit<c.max_dl?limit:c.max_dl;
    for(int iter=0;iter<repeat;++iter){
        uint64_t cyc[LMD_MAX_TXNS+1];
        int len=lmd_ref_find_cycle(c,cyc,LMD_MAX_TXNS+1);
        if(len==0){
            c.s.counts[LMD_C_DEADLOCK_NONE]+=1;
            lmd_ref_emit(c,LMD_EV_DEADLOCK_NONE,0,-1,0,0,0,-1,0);
            return;
        }
        // victim selection.
        bool have=false; uint64_t best=0;
        uint32_t bp=0; uint64_t bl=0,bs=0,bi=0;
        for(int k=0;k<len;++k){
            int ts=lmd_ref_find_tx(c,cyc[k]);
            if(ts<0) continue;
            uint32_t pr=c.s.tx_prio[ts]; uint64_t lo=c.s.tx_locks[ts],se=c.s.tx_seq[ts],id=c.s.tx_id[ts];
            bool take=false;
            if(!have) take=true;
            else if(pr<bp) take=true;
            else if(pr==bp){
                if(lo>bl) take=true;
                else if(lo==bl){ if(se>bs) take=true; else if(se==bs){ if(id>bi) take=true; } }
            }
            if(take){ have=true;best=id;bp=pr;bl=lo;bs=se;bi=id; }
        }
        if(!have){
            c.s.counts[LMD_C_DEADLOCK_NONE]+=1;
            lmd_ref_emit(c,LMD_EV_DEADLOCK_NONE,0,-1,0,0,0,-1,0);
            return;
        }
        lmd_ref_abort_victim(c,best);
    }
}

// -------------------------------------------------- snapshot hashes
// grant_hash: canonical order, then txn ascending.
__device__ uint64_t lmd_ref_grant_hash(LmdRefCtx& c){
    uint64_t h=LMD_FNV_OFFSET;
    // selection over (canonical resource, txn) ascending among live grants.
    // Use prev-key cursor.
    bool have_prev=false;
    int pk=0,pt=0,pp=0,pr=0; uint64_t ptx=0;
    for(;;){
        int best=-1;
        for(int i=0;i<c.maxlocks;++i){
            if(!c.s.g_live[i]) continue;
            if(!(c.s.g_expl[i]!=0||c.s.g_auto[i]!=0)) continue;
            if(have_prev){
                int cmp=lmd_ref_canon_cmp(c.s.g_kind[i],c.s.g_t[i],c.s.g_p[i],c.s.g_r[i],pk,pt,pp,pr);
                if(cmp<0) continue;
                if(cmp==0 && c.s.g_txn[i]<=ptx) continue;
            }
            if(best<0){ best=i; continue; }
            int cmp=lmd_ref_canon_cmp(c.s.g_kind[i],c.s.g_t[i],c.s.g_p[i],c.s.g_r[i],
                                      c.s.g_kind[best],c.s.g_t[best],c.s.g_p[best],c.s.g_r[best]);
            if(cmp<0) best=i;
            else if(cmp==0 && c.s.g_txn[i]<c.s.g_txn[best]) best=i;
        }
        if(best<0) break;
        int i=best;
        lmd_ref_h8(&h,(uint8_t)c.s.g_kind[i]);
        lmd_ref_h32(&h,(uint32_t)c.s.g_t[i]);
        lmd_ref_h32(&h, c.s.g_p[i]<0?LMD_U32MAX:(uint32_t)c.s.g_p[i]);
        lmd_ref_h32(&h, c.s.g_r[i]<0?LMD_U32MAX:(uint32_t)c.s.g_r[i]);
        lmd_ref_h64(&h,c.s.g_txn[i]);
        lmd_ref_h8(&h,(uint8_t)c.s.g_mode[i]);
        lmd_ref_h64(&h,c.s.g_expl[i]);
        lmd_ref_h64(&h,c.s.g_auto[i]);
        lmd_ref_h64(&h,c.s.g_gseq[i]);
        lmd_ref_h64(&h,c.s.g_cseq[i]);
        pk=c.s.g_kind[i];pt=c.s.g_t[i];pp=c.s.g_p[i];pr=c.s.g_r[i];ptx=c.s.g_txn[i];have_prev=true;
    }
    return h;
}

// wait_hash: canonical resource order, then queue position (wait_seq asc).
__device__ uint64_t lmd_ref_wait_hash(LmdRefCtx& c){
    uint64_t h=LMD_FNV_OFFSET;
    // iterate distinct resources in canonical order, then waiters by wait_seq.
    bool have_prev=false; int pk=0,pt=0,pp=0,pr=0;
    for(;;){
        int sel=-1;
        for(int i=0;i<c.maxwait;++i){
            if(!c.s.w_live[i]) continue;
            if(have_prev){
                int cmp=lmd_ref_canon_cmp(c.s.w_kind[i],c.s.w_t[i],c.s.w_p[i],c.s.w_r[i],pk,pt,pp,pr);
                if(cmp<=0) continue;
            }
            if(sel<0){ sel=i; continue; }
            int cmp=lmd_ref_canon_cmp(c.s.w_kind[i],c.s.w_t[i],c.s.w_p[i],c.s.w_r[i],
                                      c.s.w_kind[sel],c.s.w_t[sel],c.s.w_p[sel],c.s.w_r[sel]);
            if(cmp<0) sel=i;
        }
        if(sel<0) break;
        int rk=c.s.w_kind[sel],rt=c.s.w_t[sel],rp=c.s.w_p[sel],rr=c.s.w_r[sel];
        // emit all waiters on this resource by wait_seq ascending.
        uint64_t pos=0; bool hp2=false; uint64_t pws=0;
        for(;;){
            int wsel=-1; uint64_t bws=LMD_U64MAX;
            for(int i=0;i<c.maxwait;++i){
                if(!c.s.w_live[i]) continue;
                if(c.s.w_kind[i]!=rk||c.s.w_t[i]!=rt||c.s.w_p[i]!=rp||c.s.w_r[i]!=rr) continue;
                if(hp2 && c.s.w_waitseq[i]<=pws) continue;
                if(c.s.w_waitseq[i]<bws){ bws=c.s.w_waitseq[i]; wsel=i; }
            }
            if(wsel<0) break;
            int i=wsel;
            lmd_ref_h8(&h,(uint8_t)rk);
            lmd_ref_h32(&h,(uint32_t)rt);
            lmd_ref_h32(&h, rp<0?LMD_U32MAX:(uint32_t)rp);
            lmd_ref_h32(&h, rr<0?LMD_U32MAX:(uint32_t)rr);
            lmd_ref_h64(&h,pos++);
            lmd_ref_h64(&h,c.s.w_txn[i]);
            lmd_ref_h8(&h,(uint8_t)c.s.w_reqmode[i]);
            lmd_ref_h64(&h,c.s.w_waitseq[i]);
            lmd_ref_h64(&h,c.s.w_reqseq[i]);
            lmd_ref_h8(&h,c.s.w_isconv[i]);
            lmd_ref_h8(&h,(uint8_t)c.s.w_ot_kind[i]);
            lmd_ref_h32(&h,(uint32_t)c.s.w_ot_t[i]);
            lmd_ref_h32(&h, c.s.w_ot_p[i]<0?LMD_U32MAX:(uint32_t)c.s.w_ot_p[i]);
            lmd_ref_h32(&h, c.s.w_ot_r[i]<0?LMD_U32MAX:(uint32_t)c.s.w_ot_r[i]);
            lmd_ref_h8(&h,(uint8_t)c.s.w_ot_mode[i]);
            pws=c.s.w_waitseq[i]; hp2=true;
        }
        pk=rk;pt=rt;pp=rp;pr=rr;have_prev=true;
    }
    return h;
}

// txn_lock_hash: txn_id ascending.
__device__ uint64_t lmd_ref_txn_hash(LmdRefCtx& c){
    uint64_t h=LMD_FNV_OFFSET;
    bool have_prev=false; uint64_t pid=0;
    for(;;){
        int sel=-1; uint64_t bid=LMD_U64MAX;
        for(int i=0;i<c.maxtxn;++i){
            if(!c.s.tx_live[i]) continue;
            if(have_prev && c.s.tx_id[i]<=pid) continue;
            if(c.s.tx_id[i]<bid){ bid=c.s.tx_id[i]; sel=i; }
        }
        if(sel<0) break;
        int i=sel;
        lmd_ref_h64(&h,c.s.tx_id[i]);
        lmd_ref_h64(&h,c.s.tx_seq[i]);
        lmd_ref_h32(&h,c.s.tx_prio[i]);
        lmd_ref_h8(&h,(uint8_t)c.s.tx_status[i]);
        if(c.s.tx_bk_kind[i]!=255){
            lmd_ref_h8(&h,(uint8_t)c.s.tx_bk_kind[i]);
            lmd_ref_h32(&h,(uint32_t)c.s.tx_bk_t[i]);
            lmd_ref_h32(&h, c.s.tx_bk_p[i]<0?LMD_U32MAX:(uint32_t)c.s.tx_bk_p[i]);
            lmd_ref_h32(&h, c.s.tx_bk_r[i]<0?LMD_U32MAX:(uint32_t)c.s.tx_bk_r[i]);
            lmd_ref_h8(&h,(uint8_t)c.s.tx_bk_mode[i]);
        } else {
            lmd_ref_h8(&h,(uint8_t)255);
            lmd_ref_h32(&h,LMD_U32MAX);lmd_ref_h32(&h,LMD_U32MAX);lmd_ref_h32(&h,LMD_U32MAX);
            lmd_ref_h8(&h,(uint8_t)255);
        }
        lmd_ref_h64(&h,c.s.tx_locks[i]);
        lmd_ref_h64(&h,c.s.tx_aborts[i]);
        for(int tab=0;tab<c.tables;++tab){
            uint64_t* cc=lmd_ref_ctr(c,i,tab);
            lmd_ref_h32(&h,(uint32_t)tab);
            lmd_ref_h64(&h,cc[0]);lmd_ref_h64(&h,cc[1]);lmd_ref_h64(&h,cc[2]);lmd_ref_h64(&h,cc[3]);lmd_ref_h64(&h,cc[4]);
        }
        pid=c.s.tx_id[i]; have_prev=true;
    }
    return h;
}

// -------------------------------------------------- main kernel
__global__ void lmd_ref_step_kernel(
    lmd_ref::State s,int tables,int parts,int rows,int maxtxn,int maxlocks,int maxwait,
    int esc_thresh,int max_dl,
    int op_kind,int a_res_kind,int a_table,int a_partition,int a_row,int a_mode,int a_priority,int a_limit,
    uint64_t a_txn,
    int64_t* out_counts,int32_t* out_opidx,uint64_t* out_eseq,uint64_t* out_ev,
    uint64_t* out_grant,uint64_t* out_wait,uint64_t* out_txn,uint64_t* out_state){
    if(blockIdx.x!=0||threadIdx.x!=0) return;
    LmdRefCtx c; c.s=s;
    c.tables=tables;c.parts=parts;c.rows=rows;c.maxtxn=maxtxn;c.maxlocks=maxlocks;c.maxwait=maxwait;
    c.esc_thresh=esc_thresh;c.max_dl=max_dl;

    switch(op_kind){
        case LMD_OP_BEGIN: lmd_ref_op_begin(c,a_txn,(uint32_t)a_priority); break;
        case LMD_OP_LOCK: lmd_ref_op_lock(c,a_txn,a_res_kind,a_table,a_partition,a_row,a_mode); break;
        case LMD_OP_UNLOCK: lmd_ref_op_unlock(c,a_txn,a_res_kind,a_table,a_partition,a_row); break;
        case LMD_OP_UNLOCK_ALL: lmd_ref_op_unlock_all(c,a_txn); break;
        case LMD_OP_CONVERT: lmd_ref_op_convert(c,a_txn,a_res_kind,a_table,a_partition,a_row,a_mode); break;
        case LMD_OP_DETECT_DEADLOCK: lmd_ref_op_detect(c,a_limit); break;
        default: break;
    }
    int32_t this_op=(int32_t)c.s.scal[5];
    c.s.scal[5]+=1;

    uint64_t gh=lmd_ref_grant_hash(c);
    uint64_t wh=lmd_ref_wait_hash(c);
    uint64_t th=lmd_ref_txn_hash(c);

    for(int i=0;i<LMD_COUNT_N;++i) out_counts[i]=c.s.counts[i];
    *out_opidx=this_op;
    *out_eseq=c.s.scal[0];
    *out_ev=c.s.scal[6];
    *out_grant=gh; *out_wait=wh; *out_txn=th;

    uint64_t mh=LMD_FNV_OFFSET;
    lmd_ref_h64(&mh,c.s.scal[0]); // event_seq
    lmd_ref_h64(&mh,c.s.scal[1]); // txn_seq_next
    lmd_ref_h64(&mh,c.s.scal[2]); // request_seq_next
    lmd_ref_h64(&mh,c.s.scal[3]); // wait_seq_next
    lmd_ref_h64(&mh,c.s.scal[4]); // deadlock_seq_next
    lmd_ref_h32(&mh,(uint32_t)c.s.scal[5]); // op_index (after inc)
    lmd_ref_h64(&mh,c.s.scal[6]); // event_hash
    lmd_ref_h64(&mh,gh); lmd_ref_h64(&mh,wh); lmd_ref_h64(&mh,th);
    for(int i=0;i<LMD_COUNT_N;++i){ uint64_t cv=(uint64_t)c.s.counts[i]; lmd_ref_h64(&mh,cv); }
    *out_state=mh;
}

__global__ void lmd_ref_reset_kernel(lmd_ref::State s,int tables,int maxtxn,int maxlocks,int maxwait){
    if(blockIdx.x!=0||threadIdx.x!=0) return;
    s.scal[0]=0; s.scal[1]=1; s.scal[2]=1; s.scal[3]=1; s.scal[4]=1; s.scal[5]=0;
    s.scal[6]=LMD_FNV_OFFSET;
    for(int i=0;i<maxtxn;++i) s.tx_live[i]=0;
    for(int i=0;i<maxlocks;++i) s.g_live[i]=0;
    for(int i=0;i<maxwait;++i) s.w_live[i]=0;
    for(int i=0;i<LMD_COUNT_N;++i) s.counts[i]=0;
    (void)tables;
}

// -------------------------------------------------- host ABI
extern "C" size_t solution_workspace_bytes(const LmdProblemSpec* spec){
    if(!lmd_validate_problem_spec(spec)) return 0;
    return 256;
}

static cudaError_t lmd_ref_do_reset(lmd_ref::State* st,cudaStream_t stream){
    const LmdProblemSpec& sp=st->spec;
    lmd_ref_reset_kernel<<<1,1,0,stream>>>(*st,sp.table_count,sp.max_txns,sp.max_locks,sp.max_waiters);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_init(const LmdProblemSpec* spec,void** state_out,cudaStream_t stream){
    if(!lmd_validate_problem_spec(spec)||!state_out) return cudaErrorInvalidValue;
    // The single-thread step kernel uses recursive wake/escalate/continue-plan
    // calls and sizeable local frames; raise the per-thread device stack.
    cudaDeviceSetLimit(cudaLimitStackSize, 256 * 1024);
    lmd_ref::State* st=(lmd_ref::State*)malloc(sizeof(lmd_ref::State));
    if(!st) return cudaErrorMemoryAllocation;
    memset(st,0,sizeof(lmd_ref::State));
    memcpy(&st->spec,spec,sizeof(LmdProblemSpec));
    const int tables=spec->table_count, maxtxn=spec->max_txns, maxlocks=spec->max_locks, maxwait=spec->max_waiters;
    cudaError_t err=cudaSuccess;
    #define LMD_M(p,bytes) do{ err=cudaMalloc((void**)&(p),(bytes)); if(err) goto fail; }while(0)
    LMD_M(st->scal,sizeof(uint64_t)*7);
    LMD_M(st->tx_live,sizeof(uint8_t)*maxtxn);
    LMD_M(st->tx_id,sizeof(uint64_t)*maxtxn);
    LMD_M(st->tx_seq,sizeof(uint64_t)*maxtxn);
    LMD_M(st->tx_prio,sizeof(uint32_t)*maxtxn);
    LMD_M(st->tx_status,sizeof(int32_t)*maxtxn);
    LMD_M(st->tx_bk_kind,sizeof(int32_t)*maxtxn);
    LMD_M(st->tx_bk_t,sizeof(int32_t)*maxtxn);
    LMD_M(st->tx_bk_p,sizeof(int32_t)*maxtxn);
    LMD_M(st->tx_bk_r,sizeof(int32_t)*maxtxn);
    LMD_M(st->tx_bk_mode,sizeof(int32_t)*maxtxn);
    LMD_M(st->tx_wait_seq,sizeof(uint64_t)*maxtxn);
    LMD_M(st->tx_locks,sizeof(uint64_t)*maxtxn);
    LMD_M(st->tx_aborts,sizeof(uint64_t)*maxtxn);
    LMD_M(st->tx_ctr,sizeof(uint64_t)*(size_t)maxtxn*tables*5);
    LMD_M(st->g_live,sizeof(uint8_t)*maxlocks);
    LMD_M(st->g_kind,sizeof(int32_t)*maxlocks);
    LMD_M(st->g_t,sizeof(int32_t)*maxlocks);
    LMD_M(st->g_p,sizeof(int32_t)*maxlocks);
    LMD_M(st->g_r,sizeof(int32_t)*maxlocks);
    LMD_M(st->g_txn,sizeof(uint64_t)*maxlocks);
    LMD_M(st->g_mode,sizeof(int32_t)*maxlocks);
    LMD_M(st->g_expl,sizeof(uint64_t)*maxlocks);
    LMD_M(st->g_auto,sizeof(uint64_t)*maxlocks);
    LMD_M(st->g_gseq,sizeof(uint64_t)*maxlocks);
    LMD_M(st->g_cseq,sizeof(uint64_t)*maxlocks);
    LMD_M(st->w_live,sizeof(uint8_t)*maxwait);
    LMD_M(st->w_kind,sizeof(int32_t)*maxwait);
    LMD_M(st->w_t,sizeof(int32_t)*maxwait);
    LMD_M(st->w_p,sizeof(int32_t)*maxwait);
    LMD_M(st->w_r,sizeof(int32_t)*maxwait);
    LMD_M(st->w_txn,sizeof(uint64_t)*maxwait);
    LMD_M(st->w_reqmode,sizeof(int32_t)*maxwait);
    LMD_M(st->w_planidx,sizeof(int32_t)*maxwait);
    LMD_M(st->w_reqseq,sizeof(uint64_t)*maxwait);
    LMD_M(st->w_isconv,sizeof(uint8_t)*maxwait);
    LMD_M(st->w_ot_kind,sizeof(int32_t)*maxwait);
    LMD_M(st->w_ot_t,sizeof(int32_t)*maxwait);
    LMD_M(st->w_ot_p,sizeof(int32_t)*maxwait);
    LMD_M(st->w_ot_r,sizeof(int32_t)*maxwait);
    LMD_M(st->w_ot_mode,sizeof(int32_t)*maxwait);
    LMD_M(st->w_waitseq,sizeof(uint64_t)*maxwait);
    LMD_M(st->counts,sizeof(int64_t)*LMD_COUNT_N);
    #undef LMD_M
    err=lmd_ref_do_reset(st,stream);
    if(err) goto fail;
    *state_out=st;
    return cudaSuccess;
fail:
    solution_destroy(st);
    return err?err:cudaErrorMemoryAllocation;
}

extern "C" cudaError_t solution_run(void* state,const LmdRunSpec* run,const void* inputs_void,
                                    void* outputs_void,void* workspace,size_t workspace_bytes,cudaStream_t stream){
    (void)inputs_void;(void)workspace;
    if(!state||!outputs_void) return cudaErrorInvalidValue;
    if(workspace_bytes<256) return cudaErrorInvalidValue;
    lmd_ref::State* st=(lmd_ref::State*)state;
    if(!lmd_validate_run_spec(run,&st->spec)) return cudaErrorInvalidValue;
    LmdOutputs* out=(LmdOutputs*)outputs_void;
    if(!out->counts||!out->op_index_out||!out->event_seq_out||!out->lock_event_hash||
       !out->grant_hash||!out->wait_hash||!out->txn_lock_hash||!out->state_checksum) return cudaErrorInvalidValue;
    const LmdProblemSpec& sp=st->spec;
    lmd_ref_step_kernel<<<1,1,0,stream>>>(
        *st,sp.table_count,sp.partitions_per_table,sp.rows_per_partition,sp.max_txns,sp.max_locks,sp.max_waiters,
        sp.escalation_threshold,sp.max_deadlock_cycles_per_detect,
        run->op_kind,run->a_res_kind,run->a_table,run->a_partition,run->a_row,run->a_mode,run->a_priority,run->a_limit,
        run->a_txn,
        out->counts,out->op_index_out,out->event_seq_out,out->lock_event_hash,
        out->grant_hash,out->wait_hash,out->txn_lock_hash,out->state_checksum);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state,cudaStream_t stream){
    if(!state) return cudaErrorInvalidValue;
    return lmd_ref_do_reset((lmd_ref::State*)state,stream);
}

extern "C" void solution_destroy(void* state){
    if(!state) return;
    lmd_ref::State* st=(lmd_ref::State*)state;
    cudaFree(st->scal);
    cudaFree(st->tx_live);cudaFree(st->tx_id);cudaFree(st->tx_seq);cudaFree(st->tx_prio);
    cudaFree(st->tx_status);cudaFree(st->tx_bk_kind);cudaFree(st->tx_bk_t);cudaFree(st->tx_bk_p);
    cudaFree(st->tx_bk_r);cudaFree(st->tx_bk_mode);cudaFree(st->tx_wait_seq);cudaFree(st->tx_locks);
    cudaFree(st->tx_aborts);cudaFree(st->tx_ctr);
    cudaFree(st->g_live);cudaFree(st->g_kind);cudaFree(st->g_t);cudaFree(st->g_p);cudaFree(st->g_r);
    cudaFree(st->g_txn);cudaFree(st->g_mode);cudaFree(st->g_expl);cudaFree(st->g_auto);cudaFree(st->g_gseq);cudaFree(st->g_cseq);
    cudaFree(st->w_live);cudaFree(st->w_kind);cudaFree(st->w_t);cudaFree(st->w_p);cudaFree(st->w_r);
    cudaFree(st->w_txn);cudaFree(st->w_reqmode);cudaFree(st->w_planidx);cudaFree(st->w_reqseq);cudaFree(st->w_isconv);
    cudaFree(st->w_ot_kind);cudaFree(st->w_ot_t);cudaFree(st->w_ot_p);cudaFree(st->w_ot_r);cudaFree(st->w_ot_mode);cudaFree(st->w_waitseq);
    cudaFree(st->counts);
    free(st);
}
