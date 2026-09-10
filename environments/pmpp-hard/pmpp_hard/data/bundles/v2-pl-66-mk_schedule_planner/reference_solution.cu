// PMPP_CANARY_66_bb1f870b76 -- held-out canary; MUST NOT appear in any submission
// file: mk_schedule_planner_reference.cu
//
// Reference (implementation #2): single-thread on-device INCREMENTAL model.
// Persistent state lives in device memory; each solution_run launches one
// kernel that applies the op batch in place, mutating persistent state and
// emitting the cumulative outputs.
//
// Data representation is deliberately distinct from the host oracle: flat
// parallel arrays with linear scans, an explicit edge array scanned linearly,
// and a flat plan array. Critical heights are recomputed by an iterative
// Bellman-style relaxation over the noncancelled DAG (independent of the
// oracle's memoized post-order DFS). This is algorithmically independent from
// the naive full-replay model and from the host oracle.

#include "mk_schedule_planner_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define REF_EK_INSTR_ADD 0
#define REF_EK_EDGE_ADD 1
#define REF_EK_PLAN_INVALIDATE 2
#define REF_EK_PLAN_EMPTY 3
#define REF_EK_PLAN_STALL 4
#define REF_EK_PLAN_PLACE 5
#define REF_EK_PLAN_COMMIT 6
#define REF_EK_EXEC_INSTR 7
#define REF_EK_EDGE_SIGNAL 8
#define REF_EK_INSTR_CANCEL 9
#define REF_EK_EPOCH_ADVANCE 10
#define REF_EK_INVALID 11

#define REF_ST_UNPLANNED 0
#define REF_ST_PLANNED   1
#define REF_ST_COMMITTED 2
#define REF_ST_EXECUTED  3
#define REF_ST_CANCELLED 4

#define REF_U64MAX 0xFFFFFFFFFFFFFFFFULL
#define REF_U32MAX 0xFFFFFFFFu
#define REF_FNV_BASIS 1469598103934665603ULL
#define REF_FNV_PRIME 1099511628211ULL

enum {
    RC_INSTR_ADD=0, RC_EDGE_ADD, RC_PLAN_INVAL, RC_PLAN_EMPTY, RC_PLAN_STALL,
    RC_PLAN_PLACE, RC_PLAN_COMMIT, RC_EXEC, RC_EDGE_SIG, RC_CANCEL,
    RC_EPOCH, RC_INVALID
};

struct MkReferenceState {
    MkProblemSpec spec;
    uint32_t sm_count, pages_per_sm;
    uint64_t wave_quantum;
    int instr_cap, edge_cap;

    // instruction table (flat, alive=in-use slot)
    uint8_t* d_in_used;
    uint64_t* d_in_id;
    uint64_t* d_in_seq;
    uint64_t* d_in_dur;
    uint64_t* d_in_pcnt;
    uint64_t* d_in_pk;     // [instr_cap * MK_MAX_PAGE_KEYS_PER_INSTR]
    uint64_t* d_in_rdelay;
    uint64_t* d_in_seed;
    uint8_t*  d_in_status;
    uint64_t* d_in_crit;
    uint64_t* d_in_comp;

    // edge table (flat)
    uint8_t* d_e_used;
    uint64_t* d_e_id;
    uint64_t* d_e_src;
    uint64_t* d_e_dst;
    uint32_t* d_e_chunk;
    uint64_t* d_e_tinc;
    uint64_t* d_e_seq;

    // plan table (flat); one entry per planned/committed instruction
    uint8_t* d_p_used;
    uint64_t* d_p_seq;
    uint64_t* d_p_instr;
    uint32_t* d_p_sm;
    uint64_t* d_p_wave;
    uint64_t* d_p_start;
    uint64_t* d_p_finish;
    uint64_t* d_p_release;
    uint64_t* d_p_pcnt;
    uint64_t* d_p_pk;      // [instr_cap * MK_MAX_PAGE_KEYS_PER_INSTR]

    // scalars
    uint64_t* d_event_seq;
    uint32_t* d_op_index;
    uint64_t* d_instr_seq_next;
    uint64_t* d_edge_seq_next;
    uint64_t* d_plan_seq_next;
    uint64_t* d_committed_epoch;
    uint64_t* d_counters;   // [12]
    uint64_t* d_event_hash; // [1]

    // scratch for graph traversals (reachability / invalidation)
    int32_t* d_work_stack;  // [instr_cap]
    uint8_t* d_work_mark;   // [instr_cap]

    MkOp* d_ops;
};

__device__ __forceinline__ uint64_t ref_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= REF_FNV_PRIME; return h;
}
__device__ void ref_fnv_u8(uint64_t* h, uint8_t v) { *h = ref_fnv_byte(*h, v); }
__device__ void ref_fnv_u32(uint64_t* h, uint32_t v) {
    uint64_t x=*h; const uint8_t* p=(const uint8_t*)&v; for(int i=0;i<4;++i) x=ref_fnv_byte(x,p[i]); *h=x;
}
__device__ void ref_fnv_u64(uint64_t* h, uint64_t v) {
    uint64_t x=*h; const uint8_t* p=(const uint8_t*)&v; for(int i=0;i<8;++i) x=ref_fnv_byte(x,p[i]); *h=x;
}

struct RefDev {
    uint32_t sm_count, pages_per_sm;
    uint64_t wave_quantum;
    int instr_cap, edge_cap, max_instrs, max_edges;
    uint8_t* in_used; uint64_t* in_id; uint64_t* in_seq; uint64_t* in_dur; uint64_t* in_pcnt;
    uint64_t* in_pk; uint64_t* in_rdelay; uint64_t* in_seed; uint8_t* in_status;
    uint64_t* in_crit; uint64_t* in_comp;
    uint8_t* e_used; uint64_t* e_id; uint64_t* e_src; uint64_t* e_dst; uint32_t* e_chunk;
    uint64_t* e_tinc; uint64_t* e_seq;
    uint8_t* p_used; uint64_t* p_seq; uint64_t* p_instr; uint32_t* p_sm; uint64_t* p_wave;
    uint64_t* p_start; uint64_t* p_finish; uint64_t* p_release; uint64_t* p_pcnt; uint64_t* p_pk;
    uint64_t* event_seq; uint32_t* op_index; uint64_t* instr_seq_next; uint64_t* edge_seq_next;
    uint64_t* plan_seq_next; uint64_t* committed_epoch; uint64_t* counters; uint64_t* event_hash;
};

__device__ int ref_find_instr(RefDev* d, uint64_t id) {
    for (int i=0;i<d->instr_cap;++i) if (d->in_used[i] && d->in_id[i]==id) return i;
    return -1;
}
__device__ int ref_find_edge(RefDev* d, uint64_t id) {
    for (int i=0;i<d->edge_cap;++i) if (d->e_used[i] && d->e_id[i]==id) return i;
    return -1;
}
__device__ int ref_instr_slot(RefDev* d){ for(int i=0;i<d->instr_cap;++i) if(!d->in_used[i]) return i; return -1; }
__device__ int ref_edge_slot(RefDev* d){ for(int i=0;i<d->edge_cap;++i) if(!d->e_used[i]) return i; return -1; }
__device__ int ref_plan_slot(RefDev* d){ for(int i=0;i<d->instr_cap;++i) if(!d->p_used[i]) return i; return -1; }
__device__ int ref_count_instr(RefDev* d){ int c=0; for(int i=0;i<d->instr_cap;++i) if(d->in_used[i]) ++c; return c; }
__device__ int ref_count_edge(RefDev* d){ int c=0; for(int i=0;i<d->edge_cap;++i) if(d->e_used[i]) ++c; return c; }
__device__ int ref_find_plan_of(RefDev* d, uint64_t instr_id) {
    for (int i=0;i<d->instr_cap;++i) if (d->p_used[i] && d->p_instr[i]==instr_id) return i;
    return -1;
}
__device__ bool ref_noncancelled(RefDev* d, uint64_t id) {
    int i=ref_find_instr(d,id);
    return i>=0 && d->in_status[i]!=REF_ST_CANCELLED;
}

__device__ void ref_emit(RefDev* d, uint8_t kind, uint64_t seq, uint32_t opidx,
                         uint64_t instr_or0, uint64_t edge_or0, uint32_t sm_or_max,
                         uint64_t start_or_max, uint64_t finish_or_max, uint64_t aux) {
    uint64_t h=d->event_hash[0];
    ref_fnv_u8(&h,kind); ref_fnv_u64(&h,seq); ref_fnv_u32(&h,opidx);
    ref_fnv_u64(&h,instr_or0); ref_fnv_u64(&h,edge_or0); ref_fnv_u32(&h,sm_or_max);
    ref_fnv_u64(&h,start_or_max); ref_fnv_u64(&h,finish_or_max); ref_fnv_u64(&h,aux);
    d->event_hash[0]=h;
}

// Iterative Bellman-style relaxation: crit(i)=dur(i)+max over noncancelled
// successors crit(succ). Repeat until stable (acyclic -> converges in <=V
// passes).
__device__ void ref_recompute_crit(RefDev* d) {
    for (int i=0;i<d->instr_cap;++i) {
        if (!d->in_used[i]) continue;
        if (d->in_status[i]==REF_ST_CANCELLED) d->in_crit[i]=0;
        else d->in_crit[i]=d->in_dur[i];
    }
    int n = ref_count_instr(d);
    for (int pass=0; pass<=n; ++pass) {
        bool changed=false;
        for (int i=0;i<d->instr_cap;++i) {
            if (!d->in_used[i] || d->in_status[i]==REF_ST_CANCELLED) continue;
            uint64_t best=0;
            // scan edges where src == this instr, dst noncancelled
            for (int ei=0;ei<d->edge_cap;++ei) {
                if (!d->e_used[ei]) continue;
                if (d->e_src[ei]!=d->in_id[i]) continue;
                int dj=ref_find_instr(d,d->e_dst[ei]);
                if (dj<0 || d->in_status[dj]==REF_ST_CANCELLED) continue;
                if (d->in_crit[dj]>best) best=d->in_crit[dj];
            }
            uint64_t nv = d->in_dur[i]+best;
            if (nv!=d->in_crit[i]) { d->in_crit[i]=nv; changed=true; }
        }
        if (!changed) break;
    }
}

// reachable: is `target` reachable from `start` via noncancelled edges?
__device__ bool ref_reachable(RefDev* d, uint64_t start, uint64_t target,
                              int32_t* stack, uint8_t* seen) {
    for (int i=0;i<d->instr_cap;++i) seen[i]=0;
    int top=0;
    int si=ref_find_instr(d,start);
    if (si<0) return false;
    stack[top++]=si; seen[si]=1;
    while (top>0) {
        int cur=stack[--top];
        if (d->in_id[cur]==target) return true;
        for (int ei=0;ei<d->edge_cap;++ei) {
            if (!d->e_used[ei]) continue;
            if (d->e_src[ei]!=d->in_id[cur]) continue;
            int dj=ref_find_instr(d,d->e_dst[ei]);
            if (dj<0 || d->in_status[dj]==REF_ST_CANCELLED) continue;
            if (!seen[dj]) { seen[dj]=1; stack[top++]=dj; }
        }
    }
    return false;
}

// ready check: unplanned, all noncancelled preds PLANNED or COMMITTED.
__device__ bool ref_is_ready(RefDev* d, int idx) {
    if (d->in_status[idx]!=REF_ST_UNPLANNED) return false;
    uint64_t id=d->in_id[idx];
    for (int ei=0;ei<d->edge_cap;++ei) {
        if (!d->e_used[ei]) continue;
        if (d->e_dst[ei]!=id) continue;
        int si=ref_find_instr(d,d->e_src[ei]);
        if (si<0) continue;
        if (d->in_status[si]==REF_ST_CANCELLED) continue;
        if (d->in_status[si]!=REF_ST_PLANNED && d->in_status[si]!=REF_ST_COMMITTED) return false;
    }
    return true;
}

// pop best ready: crit desc, seq asc, id asc.
__device__ int ref_pop_ready(RefDev* d) {
    int best=-1;
    for (int i=0;i<d->instr_cap;++i) {
        if (!d->in_used[i]) continue;
        if (!ref_is_ready(d,i)) continue;
        if (best<0) { best=i; continue; }
        bool b;
        if (d->in_crit[i]!=d->in_crit[best]) b=d->in_crit[i]>d->in_crit[best];
        else if (d->in_seq[i]!=d->in_seq[best]) b=d->in_seq[i]<d->in_seq[best];
        else b=d->in_id[i]<d->in_id[best];
        if (b) best=i;
    }
    return best;
}

__device__ uint64_t ref_dep_ready(RefDev* d, uint64_t instr_id) {
    uint64_t best=0;
    for (int ei=0;ei<d->edge_cap;++ei) {
        if (!d->e_used[ei]) continue;
        if (d->e_dst[ei]!=instr_id) continue;
        uint64_t pid=d->e_src[ei];
        int si=ref_find_instr(d,pid);
        if (si<0 || d->in_status[si]==REF_ST_CANCELLED) continue;
        int pp=ref_find_plan_of(d,pid);
        if (pp<0) continue;
        if (d->p_finish[pp]>best) best=d->p_finish[pp];
    }
    return best;
}

// page feasibility on sm for [cand_start, cand_end) with req keys.
__device__ bool ref_page_feasible(RefDev* d, uint32_t sm, uint64_t cand_start,
                                  uint64_t cand_end, const uint64_t* rk, uint64_t rc,
                                  uint64_t* added_out, uint64_t* ksum_out) {
    // boundaries: cand_start plus existing interval s/e in (cand_start,cand_end).
    // We scan all plans on sm. For counting distinct keys at a boundary, we use
    // a small temporary buffer on the stack (bounded by pages_per_sm * something
    // — but live distinct keys at any time is <= total page slots in flight).
    // We gather candidate boundaries first.
    // To stay simple and exact, evaluate at cand_start and at each existing
    // interval start within the window.
    uint64_t ksum=0;
    for (uint64_t i=0;i<rc;++i) ksum+=rk[i];
    *ksum_out=ksum;

    if (cand_end<=cand_start) {
        uint64_t added=0;
        for (uint64_t i=0;i<rc;++i){ bool dup=false; for(uint64_t j=0;j<i;++j) if(rk[j]==rk[i]){dup=true;break;} if(!dup) added++; }
        *added_out=added;
        return rc<=d->pages_per_sm;
    }

    // Evaluate distinct-key count at a set of boundary ticks. Boundaries are
    // cand_start and every plan start/release within (cand_start,cand_end).
    // For each boundary, count distinct keys = candidate keys + existing
    // intervals covering it.
    // Outer loop over boundary candidates: we iterate boundary source = -1
    // (meaning cand_start) then over plan slots' start and release.
    // distinct counting uses a local key buffer.
    const int MAXKEYS = 4096;
    uint64_t keybuf[MAXKEYS];

    // build a list of boundary ticks (dedup handled implicitly by recomputation)
    for (int phase=0; phase<3; ++phase) {
        // phase 0: cand_start ; phase 1: plan starts ; phase 2: plan releases
        for (int pi=-1; pi<d->instr_cap; ++pi) {
            uint64_t t;
            if (phase==0) {
                if (pi!=-1) break;
                t=cand_start;
            } else {
                if (pi<0) continue;
                if (!d->p_used[pi] || d->p_sm[pi]!=sm) continue;
                uint64_t s=d->p_start[pi], e=d->p_release[pi];
                if (e<=cand_start || s>=cand_end) continue;  // no overlap
                if (phase==1) { t=s; if (!(t>cand_start && t<cand_end)) continue; }
                else { t=e; if (!(t>cand_start && t<cand_end)) continue; }
            }
            // count distinct keys live at t
            int nk=0;
            for (uint64_t i=0;i<rc;++i) {
                bool dup=false; for (int j=0;j<nk;++j) if (keybuf[j]==rk[i]){dup=true;break;}
                if (!dup && nk<MAXKEYS) keybuf[nk++]=rk[i];
            }
            for (int qi=0; qi<d->instr_cap; ++qi) {
                if (!d->p_used[qi] || d->p_sm[qi]!=sm) continue;
                uint64_t s=d->p_start[qi], e=d->p_release[qi];
                if (!(s<=t && t<e)) continue;
                uint64_t pc=d->p_pcnt[qi];
                for (uint64_t k=0;k<pc;++k) {
                    uint64_t key=d->p_pk[qi*MK_MAX_PAGE_KEYS_PER_INSTR+k];
                    bool dup=false; for (int j=0;j<nk;++j) if (keybuf[j]==key){dup=true;break;}
                    if (!dup && nk<MAXKEYS) keybuf[nk++]=key;
                }
            }
            if ((uint64_t)nk > d->pages_per_sm) { *added_out=0; return false; }
        }
    }

    // added pages: request keys not resident at cand_start and not earlier dup.
    uint64_t added=0;
    for (uint64_t i=0;i<rc;++i) {
        uint64_t key=rk[i];
        bool earlier=false; for (uint64_t j=0;j<i;++j) if (rk[j]==key){earlier=true;break;}
        if (earlier) continue;
        bool resident=false;
        for (int qi=0; qi<d->instr_cap; ++qi) {
            if (!d->p_used[qi] || d->p_sm[qi]!=sm) continue;
            uint64_t s=d->p_start[qi], e=d->p_release[qi];
            if (!(s<=cand_start && cand_start<e)) continue;
            uint64_t pc=d->p_pcnt[qi];
            for (uint64_t k=0;k<pc;++k) {
                if (d->p_pk[qi*MK_MAX_PAGE_KEYS_PER_INSTR+k]==key){resident=true;break;}
            }
            if (resident) break;
        }
        if (!resident) added++;
    }
    *added_out=added;
    return true;
}

__device__ int ref_place(RefDev* d, int idx, uint64_t seq, uint32_t opidx) {
    uint64_t id=d->in_id[idx];
    uint64_t dur=d->in_dur[idx];
    uint64_t rdelay=d->in_rdelay[idx];
    uint64_t pcnt=d->in_pcnt[idx];
    uint64_t rk[MK_MAX_PAGE_KEYS_PER_INSTR];
    for (uint64_t k=0;k<pcnt;++k) rk[k]=d->in_pk[idx*MK_MAX_PAGE_KEYS_PER_INSTR+k];

    uint64_t drt=ref_dep_ready(d,id);

    bool have=false; uint32_t bsm=0; uint64_t bstart=0,bfinish=0,bwave=0,badded=0,bksum=0;
    for (uint32_t sm=0; sm<d->sm_count; ++sm) {
        uint64_t wave, start;
        if (drt==0) { wave=0; start=0; }
        else { wave=(drt + d->wave_quantum - 1)/d->wave_quantum; start=wave*d->wave_quantum; }
        const int MAX_SCAN=4096;
        for (int it=0; it<MAX_SCAN; ++it) {
            uint64_t finish=start+dur;
            uint64_t cend=finish+rdelay;
            uint64_t added=0,ksum=0;
            if (ref_page_feasible(d,sm,start,cend,rk,pcnt,&added,&ksum)) {
                bool b;
                if (!have) b=true;
                else if (finish!=bfinish) b=finish<bfinish;
                else if (start!=bstart) b=start<bstart;
                else if (sm!=bsm) b=sm<bsm;
                else if (added!=badded) b=added<badded;
                else b=ksum<bksum;
                if (b){ have=true; bsm=sm; bstart=start; bfinish=finish; bwave=wave; badded=added; bksum=ksum; }
                break;
            }
            wave+=1; start=wave*d->wave_quantum;
        }
    }

    if (!have) {
        d->counters[RC_PLAN_STALL]+=1;
        ref_emit(d,REF_EK_PLAN_STALL,seq,opidx,id,0,REF_U32MAX,REF_U64MAX,REF_U64MAX,0);
        return 1;
    }

    int slot=ref_plan_slot(d);
    uint64_t ps=d->plan_seq_next[0]; d->plan_seq_next[0]=ps+1;
    d->p_used[slot]=1; d->p_seq[slot]=ps; d->p_instr[slot]=id; d->p_sm[slot]=bsm;
    d->p_wave[slot]=bwave; d->p_start[slot]=bstart; d->p_finish[slot]=bfinish;
    d->p_release[slot]=bfinish+rdelay; d->p_pcnt[slot]=pcnt;
    for (uint64_t k=0;k<pcnt;++k) d->p_pk[slot*MK_MAX_PAGE_KEYS_PER_INSTR+k]=rk[k];
    d->in_status[idx]=REF_ST_PLANNED;

    d->counters[RC_PLAN_PLACE]+=1;
    ref_emit(d,REF_EK_PLAN_PLACE,seq,opidx,id,0,bsm,bstart,bfinish,ps);
    return 0;
}

// invalidate dst and planned descendants in reverse plan_seq order.
__device__ void ref_invalidate_from(RefDev* d, uint64_t dst, uint64_t seq, uint32_t opidx,
                                    int32_t* stack, uint8_t* mark) {
    for (int i=0;i<d->instr_cap;++i) mark[i]=0;
    int top=0;
    int di=ref_find_instr(d,dst);
    if (di<0) return;
    stack[top++]=di;
    // traverse: only planned nodes are marked; do not traverse through non-planned.
    while (top>0) {
        int cur=stack[--top];
        if (d->in_status[cur]!=REF_ST_PLANNED) continue;
        if (mark[cur]) continue;
        mark[cur]=1;
        for (int ei=0;ei<d->edge_cap;++ei) {
            if (!d->e_used[ei]) continue;
            if (d->e_src[ei]!=d->in_id[cur]) continue;
            int dj=ref_find_instr(d,d->e_dst[ei]);
            if (dj<0 || d->in_status[dj]==REF_ST_CANCELLED) continue;
            stack[top++]=dj;
        }
    }
    // count marked plan slots; emit in descending plan_seq via repeated max-find.
    int nmark=0;
    for (int i=0;i<d->instr_cap;++i) if (mark[i]) nmark++;
    if (nmark==0) return;
    // emit PLAN_INVALIDATE in reverse plan_seq order.
    for (;;) {
        int best=-1;
        for (int pi=0; pi<d->instr_cap; ++pi) {
            if (!d->p_used[pi]) continue;
            int ii=ref_find_instr(d,d->p_instr[pi]);
            if (ii<0 || !mark[ii]) continue;
            if (best<0 || d->p_seq[pi]>d->p_seq[best]) best=pi;
        }
        if (best<0) break;
        d->counters[RC_PLAN_INVAL]+=1;
        ref_emit(d,REF_EK_PLAN_INVALIDATE,seq,opidx,d->p_instr[best],0,
                 d->p_sm[best],d->p_start[best],d->p_finish[best],d->p_seq[best]);
        d->p_used[best]=0;
    }
    // reset status of marked instrs to UNPLANNED.
    for (int i=0;i<d->instr_cap;++i) if (mark[i]) d->in_status[i]=REF_ST_UNPLANNED;
}

__device__ bool ref_creates_cycle(RefDev* d, uint64_t src, uint64_t dst,
                                  int32_t* stack, uint8_t* seen) {
    if (!ref_noncancelled(d,src) || !ref_noncancelled(d,dst)) return false;
    // cycle iff src reachable from dst
    return ref_reachable(d,dst,src,stack,seen);
}

__global__ void ref_step_kernel(
    uint32_t sm_count, uint32_t pages_per_sm, uint64_t wave_quantum,
    int instr_cap, int edge_cap, int max_instrs, int max_edges,
    uint8_t* in_used, uint64_t* in_id, uint64_t* in_seq, uint64_t* in_dur, uint64_t* in_pcnt,
    uint64_t* in_pk, uint64_t* in_rdelay, uint64_t* in_seed, uint8_t* in_status,
    uint64_t* in_crit, uint64_t* in_comp,
    uint8_t* e_used, uint64_t* e_id, uint64_t* e_src, uint64_t* e_dst, uint32_t* e_chunk,
    uint64_t* e_tinc, uint64_t* e_seq,
    uint8_t* p_used, uint64_t* p_seq, uint64_t* p_instr, uint32_t* p_sm, uint64_t* p_wave,
    uint64_t* p_start, uint64_t* p_finish, uint64_t* p_release, uint64_t* p_pcnt, uint64_t* p_pk,
    uint64_t* event_seq, uint32_t* op_index, uint64_t* instr_seq_next, uint64_t* edge_seq_next,
    uint64_t* plan_seq_next, uint64_t* committed_epoch, uint64_t* counters, uint64_t* event_hash,
    int32_t* work_stack, uint8_t* work_mark,
    const MkOp* ops, int num_ops,
    // outputs
    uint64_t* o_counters, uint64_t* o_event_hash,
    uint64_t* o_plan_hash, uint64_t* o_instr_hash, uint64_t* o_edge_hash, uint64_t* o_pi_hash,
    uint64_t* o_epoch, uint64_t* o_live_instr, uint64_t* o_plan_count) {
    if (blockIdx.x!=0 || threadIdx.x!=0) return;

    RefDev dd;
    RefDev* d=&dd;
    d->sm_count=sm_count; d->pages_per_sm=pages_per_sm; d->wave_quantum=wave_quantum;
    d->instr_cap=instr_cap; d->edge_cap=edge_cap; d->max_instrs=max_instrs; d->max_edges=max_edges;
    d->in_used=in_used; d->in_id=in_id; d->in_seq=in_seq; d->in_dur=in_dur; d->in_pcnt=in_pcnt;
    d->in_pk=in_pk; d->in_rdelay=in_rdelay; d->in_seed=in_seed; d->in_status=in_status;
    d->in_crit=in_crit; d->in_comp=in_comp;
    d->e_used=e_used; d->e_id=e_id; d->e_src=e_src; d->e_dst=e_dst; d->e_chunk=e_chunk;
    d->e_tinc=e_tinc; d->e_seq=e_seq;
    d->p_used=p_used; d->p_seq=p_seq; d->p_instr=p_instr; d->p_sm=p_sm; d->p_wave=p_wave;
    d->p_start=p_start; d->p_finish=p_finish; d->p_release=p_release; d->p_pcnt=p_pcnt; d->p_pk=p_pk;
    d->event_seq=event_seq; d->op_index=op_index; d->instr_seq_next=instr_seq_next;
    d->edge_seq_next=edge_seq_next; d->plan_seq_next=plan_seq_next; d->committed_epoch=committed_epoch;
    d->counters=counters; d->event_hash=event_hash;

    for (int oi=0; oi<num_ops; ++oi) {
        event_seq[0]+=1;
        uint64_t seq=event_seq[0];
        uint32_t opidx=op_index[0]; op_index[0]+=1;
        MkOp op=ops[oi];

        if (op.op_type==MK_OP_ADD_INSTR) {
            uint64_t id=op.id, dur=op.a, pcnt=op.b, rdel=op.c, seed=op.d;
            if (ref_find_instr(d,id)>=0 || ref_count_instr(d)>=max_instrs ||
                dur==0 || pcnt==0 || pcnt>pages_per_sm) {
                counters[RC_INVALID]+=1;
                ref_emit(d,REF_EK_INVALID,seq,opidx,id,0,REF_U32MAX,REF_U64MAX,REF_U64MAX,0);
            } else {
                int slot=ref_instr_slot(d);
                uint64_t isq=instr_seq_next[0]; instr_seq_next[0]=isq+1;
                in_used[slot]=1; in_id[slot]=id; in_seq[slot]=isq; in_dur[slot]=dur;
                in_pcnt[slot]=pcnt; in_rdelay[slot]=rdel; in_seed[slot]=seed;
                in_status[slot]=REF_ST_UNPLANNED; in_crit[slot]=dur; in_comp[slot]=0;
                for (uint64_t k=0;k<MK_MAX_PAGE_KEYS_PER_INSTR;++k)
                    in_pk[slot*MK_MAX_PAGE_KEYS_PER_INSTR+k]=(k<pcnt)?op.page_keys[k]:0;
                ref_recompute_crit(d);
                counters[RC_INSTR_ADD]+=1;
                ref_emit(d,REF_EK_INSTR_ADD,seq,opidx,id,0,REF_U32MAX,REF_U64MAX,REF_U64MAX,isq);
            }
        } else if (op.op_type==MK_OP_ADD_EDGE) {
            uint64_t eid=op.id, src=op.src, dst=op.dst;
            uint32_t chunk=(uint32_t)op.a; uint64_t tinc=op.b;
            int si=ref_find_instr(d,src), di=ref_find_instr(d,dst);
            bool invalid=false;
            if (ref_find_edge(d,eid)>=0) invalid=true;
            else if (si<0 || di<0) invalid=true;
            else if (src==dst) invalid=true;
            else if (tinc==0) invalid=true;
            else if (ref_count_edge(d)>=max_edges) invalid=true;
            else if (in_status[di]==REF_ST_COMMITTED || in_status[di]==REF_ST_EXECUTED) invalid=true;
            else if (in_status[si]==REF_ST_CANCELLED || in_status[di]==REF_ST_CANCELLED) invalid=true;
            else if (ref_creates_cycle(d,src,dst,work_stack,work_mark)) invalid=true;
            if (invalid) {
                counters[RC_INVALID]+=1;
                ref_emit(d,REF_EK_INVALID,seq,opidx,0,eid,REF_U32MAX,REF_U64MAX,REF_U64MAX,0);
            } else {
                int slot=ref_edge_slot(d);
                uint64_t es=edge_seq_next[0]; edge_seq_next[0]=es+1;
                e_used[slot]=1; e_id[slot]=eid; e_src[slot]=src; e_dst[slot]=dst;
                e_chunk[slot]=chunk; e_tinc[slot]=tinc; e_seq[slot]=es;
                bool sp=(in_status[si]==REF_ST_PLANNED), dp=(in_status[di]==REF_ST_PLANNED);
                if (sp||dp) ref_invalidate_from(d,dst,seq,opidx,work_stack,work_mark);
                ref_recompute_crit(d);
                counters[RC_EDGE_ADD]+=1;
                ref_emit(d,REF_EK_EDGE_ADD,seq,opidx,0,eid,REF_U32MAX,REF_U64MAX,REF_U64MAX,es);
            }
        } else if (op.op_type==MK_OP_PLAN_NEXT) {
            uint64_t limit=op.a;
            if (limit!=0) {
                for (uint64_t n=0;n<limit;++n) {
                    int best=ref_pop_ready(d);
                    if (best<0) {
                        counters[RC_PLAN_EMPTY]+=1;
                        ref_emit(d,REF_EK_PLAN_EMPTY,seq,opidx,0,0,REF_U32MAX,REF_U64MAX,REF_U64MAX,0);
                        break;
                    }
                    int r=ref_place(d,best,seq,opidx);
                    if (r==1) break;
                }
            }
        } else if (op.op_type==MK_OP_COMMIT_PLAN) {
            uint64_t maxe=op.a;
            if (maxe!=0) {
                uint64_t done=0;
                // commit by plan_seq ascending among PLANNED.
                for (;;) {
                    if (done>=maxe) break;
                    int best=-1;
                    for (int pi=0; pi<d->instr_cap; ++pi) {
                        if (!p_used[pi]) continue;
                        int ii=ref_find_instr(d,p_instr[pi]);
                        if (ii<0 || in_status[ii]!=REF_ST_PLANNED) continue;
                        if (best<0 || p_seq[pi]<p_seq[best]) best=pi;
                    }
                    if (best<0) break;
                    int ii=ref_find_instr(d,p_instr[best]);
                    in_status[ii]=REF_ST_COMMITTED;
                    counters[RC_PLAN_COMMIT]+=1;
                    ref_emit(d,REF_EK_PLAN_COMMIT,seq,opidx,p_instr[best],0,
                             p_sm[best],p_start[best],p_finish[best],p_seq[best]);
                    done+=1;
                }
            }
        } else if (op.op_type==MK_OP_EXECUTE_UNTIL) {
            uint64_t tl=op.a, maxev=op.b;
            if (maxev!=0) {
                uint64_t done=0;
                for (;;) {
                    if (done>=maxev) break;
                    // pick min (finish, sm, plan_seq) among committed,finish<=tl.
                    int best=-1;
                    for (int pi=0; pi<d->instr_cap; ++pi) {
                        if (!p_used[pi]) continue;
                        int ii=ref_find_instr(d,p_instr[pi]);
                        if (ii<0 || in_status[ii]!=REF_ST_COMMITTED) continue;
                        if (p_finish[pi]>tl) continue;
                        if (best<0) { best=pi; continue; }
                        bool b;
                        if (p_finish[pi]!=p_finish[best]) b=p_finish[pi]<p_finish[best];
                        else if (p_sm[pi]!=p_sm[best]) b=p_sm[pi]<p_sm[best];
                        else b=p_seq[pi]<p_seq[best];
                        if (b) best=pi;
                    }
                    if (best<0) break;
                    int ii=ref_find_instr(d,p_instr[best]);
                    // result hash
                    uint64_t result=REF_FNV_BASIS;
                    ref_fnv_u64(&result,committed_epoch[0]);
                    ref_fnv_u64(&result,in_id[ii]);
                    ref_fnv_u64(&result,p_start[best]);
                    ref_fnv_u64(&result,p_finish[best]);
                    ref_fnv_u32(&result,p_sm[best]);
                    ref_fnv_u64(&result,in_seed[ii]);
                    for (uint64_t k=0;k<in_pcnt[ii];++k)
                        ref_fnv_u64(&result,in_pk[ii*MK_MAX_PAGE_KEYS_PER_INSTR+k]);
                    in_comp[ii]+=1;
                    in_status[ii]=REF_ST_EXECUTED;
                    counters[RC_EXEC]+=1;
                    ref_emit(d,REF_EK_EXEC_INSTR,seq,opidx,in_id[ii],0,
                             p_sm[best],p_start[best],p_finish[best],result);
                    // outgoing edges by edge_seq ascending.
                    uint64_t myid=in_id[ii];
                    {
                        uint64_t last=0; bool have_last=false;
                        for (;;) {
                            int be=-1;
                            for (int ei=0; ei<d->edge_cap; ++ei) {
                                if (!e_used[ei]) continue;
                                if (e_src[ei]!=myid) continue;
                                if (have_last && e_seq[ei]<=last) continue;
                                if (be<0 || e_seq[ei]<e_seq[be]) be=ei;
                            }
                            if (be<0) break;
                            counters[RC_EDGE_SIG]+=1;
                            uint64_t aux=((uint64_t)e_chunk[be]) ^ (e_tinc[be]*REF_FNV_PRIME);
                            ref_emit(d,REF_EK_EDGE_SIGNAL,seq,opidx,e_dst[be],e_id[be],
                                     REF_U32MAX,REF_U64MAX,REF_U64MAX,aux);
                            last=e_seq[be]; have_last=true;
                        }
                    }
                    done+=1;
                }
            }
        } else if (op.op_type==MK_OP_CANCEL_INSTR) {
            uint64_t id=op.id;
            int ii=ref_find_instr(d,id);
            if (ii<0 || in_status[ii]==REF_ST_EXECUTED || in_status[ii]==REF_ST_COMMITTED ||
                in_status[ii]==REF_ST_CANCELLED) {
                counters[RC_INVALID]+=1;
                ref_emit(d,REF_EK_INVALID,seq,opidx,id,0,REF_U32MAX,REF_U64MAX,REF_U64MAX,0);
            } else {
                if (in_status[ii]==REF_ST_PLANNED)
                    ref_invalidate_from(d,id,seq,opidx,work_stack,work_mark);
                // remove incident uncommitted edges (other endpoint not committed/executed).
                for (int ei=0; ei<d->edge_cap; ++ei) {
                    if (!e_used[ei]) continue;
                    if (e_src[ei]!=id && e_dst[ei]!=id) continue;
                    uint64_t other=(e_src[ei]==id)?e_dst[ei]:e_src[ei];
                    int oi2=ref_find_instr(d,other);
                    uint8_t ost=(oi2>=0)?in_status[oi2]:REF_ST_UNPLANNED;
                    if (ost==REF_ST_COMMITTED || ost==REF_ST_EXECUTED) continue;
                    e_used[ei]=0;
                }
                in_status[ii]=REF_ST_CANCELLED; in_crit[ii]=0;
                ref_recompute_crit(d);
                counters[RC_CANCEL]+=1;
                ref_emit(d,REF_EK_INSTR_CANCEL,seq,opidx,id,0,REF_U32MAX,REF_U64MAX,REF_U64MAX,0);
            }
        } else if (op.op_type==MK_OP_NEW_EPOCH) {
            bool bad=false;
            for (int i=0;i<d->instr_cap;++i) if (in_used[i] && in_status[i]==REF_ST_COMMITTED){bad=true;break;}
            if (bad) {
                counters[RC_INVALID]+=1;
                ref_emit(d,REF_EK_INVALID,seq,opidx,0,0,REF_U32MAX,REF_U64MAX,REF_U64MAX,0);
            } else {
                committed_epoch[0]+=1;
                for (int i=0;i<d->instr_cap;++i) {
                    if (!in_used[i]) continue;
                    if (in_status[i]!=REF_ST_CANCELLED) { in_status[i]=REF_ST_UNPLANNED; in_comp[i]=0; }
                }
                for (int pi=0; pi<d->instr_cap; ++pi) p_used[pi]=0;
                ref_recompute_crit(d);
                counters[RC_EPOCH]+=1;
                ref_emit(d,REF_EK_EPOCH_ADVANCE,seq,opidx,0,0,REF_U32MAX,REF_U64MAX,REF_U64MAX,committed_epoch[0]);
            }
        } else {
            counters[RC_INVALID]+=1;
            ref_emit(d,REF_EK_INVALID,seq,opidx,0,0,REF_U32MAX,REF_U64MAX,REF_U64MAX,0);
        }
    }

    // ---- structural hashes ----
    // plan_hash: by sm, start_tick, plan_seq.
    uint64_t ph=REF_FNV_BASIS;
    {
        uint64_t lsm=0,lstart=0,lseq=0; bool have=false;
        int total=0; for (int i=0;i<instr_cap;++i) if (p_used[i]) total++;
        for (int e=0;e<total;++e) {
            int best=-1;
            for (int i=0;i<instr_cap;++i) {
                if (!p_used[i]) continue;
                uint64_t sm=p_sm[i], st=p_start[i], sq=p_seq[i];
                if (have) {
                    if (sm<lsm) continue;
                    if (sm==lsm && st<lstart) continue;
                    if (sm==lsm && st==lstart && sq<=lseq) continue;
                }
                if (best<0) { best=i; continue; }
                uint64_t bsm=p_sm[best], bst=p_start[best], bsq=p_seq[best];
                bool b;
                if (sm!=bsm) b=sm<bsm;
                else if (st!=bst) b=st<bst;
                else b=sq<bsq;
                if (b) best=i;
            }
            if (best<0) break;
            ref_fnv_u32(&ph,(uint32_t)p_sm[best]);
            ref_fnv_u64(&ph,p_seq[best]);
            ref_fnv_u64(&ph,p_instr[best]);
            ref_fnv_u64(&ph,p_wave[best]);
            ref_fnv_u64(&ph,p_start[best]);
            ref_fnv_u64(&ph,p_finish[best]);
            ref_fnv_u64(&ph,p_release[best]);
            for (uint64_t k=0;k<p_pcnt[best];++k)
                ref_fnv_u64(&ph,p_pk[best*MK_MAX_PAGE_KEYS_PER_INSTR+k]);
            lsm=p_sm[best]; lstart=p_start[best]; lseq=p_seq[best]; have=true;
        }
    }
    o_plan_hash[0]=ph;

    // instr_hash: by instr_id ascending.
    uint64_t ih=REF_FNV_BASIS;
    {
        uint64_t lid=0; bool have=false;
        int total=0; for (int i=0;i<instr_cap;++i) if (in_used[i]) total++;
        for (int e=0;e<total;++e) {
            int best=-1;
            for (int i=0;i<instr_cap;++i) {
                if (!in_used[i]) continue;
                if (have && in_id[i]<=lid) continue;
                if (best<0 || in_id[i]<in_id[best]) best=i;
            }
            if (best<0) break;
            ref_fnv_u64(&ih,in_id[best]);
            ref_fnv_u64(&ih,in_seq[best]);
            ref_fnv_u64(&ih,in_dur[best]);
            ref_fnv_u64(&ih,in_rdelay[best]);
            ref_fnv_u64(&ih,in_seed[best]);
            ref_fnv_u8(&ih,in_status[best]);
            ref_fnv_u64(&ih,in_crit[best]);
            ref_fnv_u64(&ih,in_comp[best]);
            lid=in_id[best]; have=true;
        }
    }
    o_instr_hash[0]=ih;

    // edge_hash: by edge_id ascending.
    uint64_t eh=REF_FNV_BASIS;
    {
        uint64_t lid=0; bool have=false;
        int total=0; for (int i=0;i<edge_cap;++i) if (e_used[i]) total++;
        for (int e=0;e<total;++e) {
            int best=-1;
            for (int i=0;i<edge_cap;++i) {
                if (!e_used[i]) continue;
                if (have && e_id[i]<=lid) continue;
                if (best<0 || e_id[i]<e_id[best]) best=i;
            }
            if (best<0) break;
            ref_fnv_u64(&eh,e_id[best]);
            ref_fnv_u64(&eh,e_src[best]);
            ref_fnv_u64(&eh,e_dst[best]);
            ref_fnv_u32(&eh,e_chunk[best]);
            ref_fnv_u64(&eh,e_tinc[best]);
            ref_fnv_u64(&eh,e_seq[best]);
            lid=e_id[best]; have=true;
        }
    }
    o_edge_hash[0]=eh;

    // page_interval_hash: by sm, key, start, instr_id, key-index. The per-plan
    // key index disambiguates duplicate page keys within one plan (total order).
    uint64_t pih=REF_FNV_BASIS;
    {
        uint64_t lsm=0,lkey=0,lstart=0,lid=0,lkidx=0; bool have=false;
        int total=0; for (int i=0;i<instr_cap;++i) if (p_used[i]) total+=(int)p_pcnt[i];
        for (int e=0;e<total;++e) {
            int bpi=-1;
            uint64_t bsm=0,bkey=0,bst=0,bid=0,bkidx=0;
            for (int pi=0; pi<instr_cap; ++pi) {
                if (!p_used[pi]) continue;
                for (uint64_t k=0;k<p_pcnt[pi];++k) {
                    uint64_t sm=p_sm[pi], key=p_pk[pi*MK_MAX_PAGE_KEYS_PER_INSTR+k];
                    uint64_t st=p_start[pi], id=p_instr[pi], kidx=k;
                    if (have) {
                        if (sm<lsm) continue;
                        if (sm==lsm && key<lkey) continue;
                        if (sm==lsm && key==lkey && st<lstart) continue;
                        if (sm==lsm && key==lkey && st==lstart && id<lid) continue;
                        if (sm==lsm && key==lkey && st==lstart && id==lid && kidx<=lkidx) continue;
                    }
                    if (bpi<0) { bpi=pi; bsm=sm; bkey=key; bst=st; bid=id; bkidx=kidx; continue; }
                    bool b;
                    if (sm!=bsm) b=sm<bsm;
                    else if (key!=bkey) b=key<bkey;
                    else if (st!=bst) b=st<bst;
                    else if (id!=bid) b=id<bid;
                    else b=kidx<bkidx;
                    if (b) { bpi=pi; bsm=sm; bkey=key; bst=st; bid=id; bkidx=kidx; }
                }
            }
            if (bpi<0) break;
            ref_fnv_u32(&pih,(uint32_t)bsm);
            ref_fnv_u64(&pih,bkey);
            ref_fnv_u64(&pih,bst);
            ref_fnv_u64(&pih,p_release[bpi]);
            ref_fnv_u64(&pih,bid);
            lsm=bsm; lkey=bkey; lstart=bst; lid=bid; lkidx=bkidx; have=true;
        }
    }
    o_pi_hash[0]=pih;

    for (int i=0;i<12;++i) o_counters[i]=counters[i];
    o_event_hash[0]=event_hash[0];
    o_epoch[0]=committed_epoch[0];
    {
        uint64_t live=0; for (int i=0;i<instr_cap;++i) if (in_used[i] && in_status[i]!=REF_ST_CANCELLED) live++;
        o_live_instr[0]=live;
    }
    {
        uint64_t pc=0; for (int i=0;i<instr_cap;++i) if (p_used[i]) pc++;
        o_plan_count[0]=pc;
    }
}

// ---------------------------------------------------------------------------
// Host glue.
// ---------------------------------------------------------------------------
__global__ void ref_init_scalars(uint64_t* event_seq, uint32_t* op_index,
                                  uint64_t* instr_seq_next, uint64_t* edge_seq_next,
                                  uint64_t* plan_seq_next, uint64_t* committed_epoch,
                                  uint64_t* counters, uint64_t* event_hash) {
    if (blockIdx.x!=0 || threadIdx.x!=0) return;
    event_seq[0]=0; op_index[0]=0;
    instr_seq_next[0]=1; edge_seq_next[0]=1; plan_seq_next[0]=1; committed_epoch[0]=0;
    for (int i=0;i<12;++i) counters[i]=0;
    event_hash[0]=REF_FNV_BASIS;
}

static cudaError_t ref_reset_state(MkReferenceState* st, cudaStream_t stream) {
    cudaError_t err;
    err=cudaMemsetAsync(st->d_in_used,0,sizeof(uint8_t)*st->instr_cap,stream); if(err) return err;
    err=cudaMemsetAsync(st->d_e_used,0,sizeof(uint8_t)*st->edge_cap,stream); if(err) return err;
    err=cudaMemsetAsync(st->d_p_used,0,sizeof(uint8_t)*st->instr_cap,stream); if(err) return err;
    ref_init_scalars<<<1,1,0,stream>>>(st->d_event_seq, st->d_op_index, st->d_instr_seq_next,
        st->d_edge_seq_next, st->d_plan_seq_next, st->d_committed_epoch, st->d_counters, st->d_event_hash);
    return cudaPeekAtLastError();
}

extern "C" size_t solution_workspace_bytes(const MkProblemSpec* spec) {
    if (!mk_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(const MkProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!mk_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    MkReferenceState* st=(MkReferenceState*)malloc(sizeof(MkReferenceState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st,0,sizeof(*st));
    memcpy(&st->spec,spec,sizeof(MkProblemSpec));
    st->sm_count=(uint32_t)spec->sm_count; st->pages_per_sm=(uint32_t)spec->pages_per_sm;
    st->wave_quantum=spec->wave_quantum;
    st->instr_cap=spec->max_instrs;
    st->edge_cap=(spec->max_edges>0)?spec->max_edges:1;

    cudaError_t err=cudaSuccess;
    int IC=st->instr_cap, EC=st->edge_cap;
    int PKW=IC*MK_MAX_PAGE_KEYS_PER_INSTR;
#define ALLOC(p,b) do{ err=cudaMalloc((void**)&(p),(b)); if(err) goto fail; }while(0)
    ALLOC(st->d_in_used,sizeof(uint8_t)*IC);
    ALLOC(st->d_in_id,sizeof(uint64_t)*IC);
    ALLOC(st->d_in_seq,sizeof(uint64_t)*IC);
    ALLOC(st->d_in_dur,sizeof(uint64_t)*IC);
    ALLOC(st->d_in_pcnt,sizeof(uint64_t)*IC);
    ALLOC(st->d_in_pk,sizeof(uint64_t)*PKW);
    ALLOC(st->d_in_rdelay,sizeof(uint64_t)*IC);
    ALLOC(st->d_in_seed,sizeof(uint64_t)*IC);
    ALLOC(st->d_in_status,sizeof(uint8_t)*IC);
    ALLOC(st->d_in_crit,sizeof(uint64_t)*IC);
    ALLOC(st->d_in_comp,sizeof(uint64_t)*IC);
    ALLOC(st->d_e_used,sizeof(uint8_t)*EC);
    ALLOC(st->d_e_id,sizeof(uint64_t)*EC);
    ALLOC(st->d_e_src,sizeof(uint64_t)*EC);
    ALLOC(st->d_e_dst,sizeof(uint64_t)*EC);
    ALLOC(st->d_e_chunk,sizeof(uint32_t)*EC);
    ALLOC(st->d_e_tinc,sizeof(uint64_t)*EC);
    ALLOC(st->d_e_seq,sizeof(uint64_t)*EC);
    ALLOC(st->d_p_used,sizeof(uint8_t)*IC);
    ALLOC(st->d_p_seq,sizeof(uint64_t)*IC);
    ALLOC(st->d_p_instr,sizeof(uint64_t)*IC);
    ALLOC(st->d_p_sm,sizeof(uint32_t)*IC);
    ALLOC(st->d_p_wave,sizeof(uint64_t)*IC);
    ALLOC(st->d_p_start,sizeof(uint64_t)*IC);
    ALLOC(st->d_p_finish,sizeof(uint64_t)*IC);
    ALLOC(st->d_p_release,sizeof(uint64_t)*IC);
    ALLOC(st->d_p_pcnt,sizeof(uint64_t)*IC);
    ALLOC(st->d_p_pk,sizeof(uint64_t)*PKW);
    ALLOC(st->d_event_seq,sizeof(uint64_t));
    ALLOC(st->d_op_index,sizeof(uint32_t));
    ALLOC(st->d_instr_seq_next,sizeof(uint64_t));
    ALLOC(st->d_edge_seq_next,sizeof(uint64_t));
    ALLOC(st->d_plan_seq_next,sizeof(uint64_t));
    ALLOC(st->d_committed_epoch,sizeof(uint64_t));
    ALLOC(st->d_counters,sizeof(uint64_t)*12);
    ALLOC(st->d_event_hash,sizeof(uint64_t));
    ALLOC(st->d_work_stack,sizeof(int32_t)*IC);
    ALLOC(st->d_work_mark,sizeof(uint8_t)*IC);
    {
        size_t ob=sizeof(MkOp)*(size_t)(spec->max_ops_per_step>0?spec->max_ops_per_step:1);
        ALLOC(st->d_ops,ob);
    }
#undef ALLOC

    err=ref_reset_state(st,stream);
    if (err) goto fail;
    *state_out=st;
    return cudaSuccess;
fail:
    solution_destroy(st);
    return err;
}

extern "C" cudaError_t solution_run(void* state, const MkRunSpec* run, const void* inputs_void,
                                    void* outputs_void, void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    if (!state || !inputs_void || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 1) return cudaErrorInvalidValue;
    MkReferenceState* st=(MkReferenceState*)state;
    if (!mk_validate_run_spec(run,&st->spec)) return cudaErrorInvalidValue;
    const MkInputs* in=(const MkInputs*)inputs_void;
    MkOutputs* out=(MkOutputs*)outputs_void;
    if (run->num_ops>0 && !in->ops) return cudaErrorInvalidValue;

    // Graph-traversal scratch (work_stack / work_mark) is allocated once in
    // solution_init and persists on the state; solution_run never cudaMallocs.
    (void)workspace;

    cudaError_t err;
    if (run->num_ops>0) {
        err=cudaMemcpyAsync(st->d_ops, in->ops, sizeof(MkOp)*(size_t)run->num_ops,
                            cudaMemcpyDeviceToDevice, stream);
        if (err) return err;
    }

    ref_step_kernel<<<1,1,0,stream>>>(
        st->sm_count, st->pages_per_sm, st->wave_quantum,
        st->instr_cap, st->edge_cap, st->spec.max_instrs, st->spec.max_edges,
        st->d_in_used, st->d_in_id, st->d_in_seq, st->d_in_dur, st->d_in_pcnt,
        st->d_in_pk, st->d_in_rdelay, st->d_in_seed, st->d_in_status, st->d_in_crit, st->d_in_comp,
        st->d_e_used, st->d_e_id, st->d_e_src, st->d_e_dst, st->d_e_chunk, st->d_e_tinc, st->d_e_seq,
        st->d_p_used, st->d_p_seq, st->d_p_instr, st->d_p_sm, st->d_p_wave,
        st->d_p_start, st->d_p_finish, st->d_p_release, st->d_p_pcnt, st->d_p_pk,
        st->d_event_seq, st->d_op_index, st->d_instr_seq_next, st->d_edge_seq_next,
        st->d_plan_seq_next, st->d_committed_epoch, st->d_counters, st->d_event_hash,
        st->d_work_stack, st->d_work_mark,
        st->d_ops, run->num_ops,
        st->d_counters /*reuse o_counters*/, out->planner_event_hash,
        out->plan_hash, out->instr_hash, out->edge_hash, out->page_interval_hash,
        out->committed_epoch, out->live_instr_count, out->planned_interval_count);
    err=cudaPeekAtLastError(); if (err) return err;

    uint64_t* cdst[12] = {
        out->instr_added, out->edge_added, out->plan_invalidated, out->plan_empty,
        out->plan_stall, out->plan_placed, out->plan_committed, out->instr_executed,
        out->edge_signaled, out->instr_cancelled, out->epoch_advanced, out->invalid_count
    };
    for (int i=0;i<12;++i) {
        err=cudaMemcpyAsync(cdst[i], st->d_counters + i, sizeof(uint64_t),
                            cudaMemcpyDeviceToDevice, stream);
        if (err) return err;
    }
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return ref_reset_state((MkReferenceState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    MkReferenceState* st=(MkReferenceState*)state;
#define FREE(p) do{ if(p) cudaFree(p); }while(0)
    FREE(st->d_in_used); FREE(st->d_in_id); FREE(st->d_in_seq); FREE(st->d_in_dur);
    FREE(st->d_in_pcnt); FREE(st->d_in_pk); FREE(st->d_in_rdelay); FREE(st->d_in_seed);
    FREE(st->d_in_status); FREE(st->d_in_crit); FREE(st->d_in_comp);
    FREE(st->d_e_used); FREE(st->d_e_id); FREE(st->d_e_src); FREE(st->d_e_dst);
    FREE(st->d_e_chunk); FREE(st->d_e_tinc); FREE(st->d_e_seq);
    FREE(st->d_p_used); FREE(st->d_p_seq); FREE(st->d_p_instr); FREE(st->d_p_sm);
    FREE(st->d_p_wave); FREE(st->d_p_start); FREE(st->d_p_finish); FREE(st->d_p_release);
    FREE(st->d_p_pcnt); FREE(st->d_p_pk);
    FREE(st->d_event_seq); FREE(st->d_op_index); FREE(st->d_instr_seq_next);
    FREE(st->d_edge_seq_next); FREE(st->d_plan_seq_next); FREE(st->d_committed_epoch);
    FREE(st->d_counters); FREE(st->d_event_hash); FREE(st->d_ops);
    FREE(st->d_work_stack); FREE(st->d_work_mark);
#undef FREE
    free(st);
}
