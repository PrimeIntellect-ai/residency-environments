// PMPP_CANARY_60_e632397ecc -- held-out canary; MUST NOT appear in any submission
// file: mk_paged_interpreter_reference.cu
//
// Reference GPU implementation of MK1 (paged SM interpreter). Runs the whole
// interpreter step inside a single <<<1,1>>> kernel mutating persistent global
// device state. Data-structure strategy (independent of oracle + naive):
//   * Persistent state lives in flat device allocations created at init.
//   * The pending-event queue is an UNSORTED slot pool; canonical order is
//     materialised by linear selection-min at use time (ADVANCE / hashes).
//   * Page pool / per-SM interpreter state are flat arrays indexed (sm,page)
//     and (sm). Static program is copied to device at init.

#include "mk_paged_interpreter_common.h"

#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MK_FNV_OFFSET 1469598103934665603ULL
#define MK_FNV_PRIME  1099511628211ULL

namespace mk_ref {

// Flat device state.
struct State {
    MkProblemSpec spec;
    int SM, PAGES, NCTR, MAXP;

    // static program (device copies)
    int32_t*  program_len;     // [SM]
    int32_t*  instr_offset;    // [SM]
    MkInstr*  instrs;          // [total_instr]
    MkPageReq* reqs;           // [total_reqs]
    MkWaitRec* waits;          // [total_waits]

    // scalars: [clock, event_seq, instr_instance_seq_next, op_index,
    //           event_hash, pass_id, pass_active]
    uint64_t* scal;            // 7 u64

    uint64_t* counter;         // [NCTR]

    // per SM
    uint32_t* sm_pc;           // [SM]
    uint64_t* sm_active_inst;  // [SM]  0 == none
    uint8_t*  sm_stage;        // [SM]
    uint64_t* sm_instr_id;     // [SM]
    uint64_t* sm_compute_val;  // [SM]
    uint32_t* sm_req_count;    // [SM]
    uint32_t* sm_wait_count;   // [SM]
    uint32_t* sm_req_offset;   // [SM]
    uint32_t* sm_wait_offset;  // [SM]
    uint64_t* sm_result_seed;  // [SM]
    uint64_t* sm_load_lat;     // [SM]
    uint64_t* sm_comp_lat;     // [SM]
    uint64_t* sm_store_lat;    // [SM]
    uint32_t* sm_out_counter;  // [SM]
    uint64_t* sm_out_incr;     // [SM]
    uint32_t* sm_alloc;        // [SM*MK_MAX_PAGE_REQS]
    uint8_t*  sm_ready;        // [SM*MK_MAX_PAGE_REQS]

    // per page (sm*PAGES + page)
    uint8_t*  pg_state;
    uint64_t* pg_tile;
    uint8_t*  pg_mode;
    uint64_t* pg_owner;
    uint32_t* pg_reqidx;
    uint64_t* pg_loadseq;
    uint64_t* pg_relseq;

    // pending events (slot pool, count in scal? keep separate)
    uint64_t* pe_due;
    uint64_t* pe_cseq;
    uint8_t*  pe_kind;
    int32_t*  pe_sm;
    uint64_t* pe_inst;
    uint32_t* pe_page;
    int32_t*  pe_count;        // [1]

    int64_t*  counts;          // [MK_COUNT_N]
};

}  // namespace mk_ref

// scalars index helpers
enum { SC_CLOCK=0, SC_ESEQ=1, SC_INST_NEXT=2, SC_OPIDX=3, SC_EHASH=4, SC_PASSID=5, SC_ACTIVE=6 };

__device__ __forceinline__ uint64_t mk_ref_fb(uint64_t h, uint8_t b) { h ^= (uint64_t)b; h *= MK_FNV_PRIME; return h; }
__device__ void mk_ref_fbytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*)p; uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = mk_ref_fb(v, q[i]);
    *h = v;
}

struct MkRefCtx { mk_ref::State s; };

__device__ __forceinline__ int mk_pgi(MkRefCtx& c, int sm, int p) { return sm * c.s.PAGES + p; }

// ------------------------------------------------------- emit
__device__ int mk_ref_kind_to_count(uint8_t kind) {
    if (kind < MK_EV_PAGE_RELEASE_PINNED_HINT) return (int)kind;
    if (kind == MK_EV_PAGE_RELEASE_PINNED_HINT) return MK_C_PAGE_RELEASE;
    return (int)kind - 1;
}
__device__ void mk_ref_emit(MkRefCtx& c, uint8_t kind, int sm_or, uint64_t instr_or,
                            uint64_t inst_or, uint32_t page_or, uint32_t ctr_or, uint64_t val_or) {
    uint64_t* sc = c.s.scal;
    const uint64_t seq = sc[SC_ESEQ];
    uint64_t h = sc[SC_EHASH];
    const uint32_t smv = (sm_or < 0) ? UINT32_MAX : (uint32_t)sm_or;
    uint32_t opi = (uint32_t)sc[SC_OPIDX];
    uint64_t clk = sc[SC_CLOCK];
    mk_ref_fbytes(&h, &kind, 1);
    mk_ref_fbytes(&h, &seq, 8);
    mk_ref_fbytes(&h, &opi, 4);
    mk_ref_fbytes(&h, &clk, 8);
    mk_ref_fbytes(&h, &smv, 4);
    mk_ref_fbytes(&h, &instr_or, 8);
    mk_ref_fbytes(&h, &inst_or, 8);
    mk_ref_fbytes(&h, &page_or, 4);
    mk_ref_fbytes(&h, &ctr_or, 4);
    mk_ref_fbytes(&h, &val_or, 8);
    sc[SC_EHASH] = h;
    sc[SC_ESEQ] = seq + 1;
    c.s.counts[mk_ref_kind_to_count(kind)] += 1;
}
__device__ __forceinline__ uint64_t mk_ref_peek(MkRefCtx& c) { return c.s.scal[SC_ESEQ]; }
__device__ void mk_ref_invalid(MkRefCtx& c) {
    mk_ref_emit(c, MK_EV_INVALID, -1, UINT64_MAX, 0, UINT32_MAX, UINT32_MAX, UINT64_MAX);
}

// ------------------------------------------------------- BEGIN_PASS
__device__ void mk_ref_begin_pass(MkRefCtx& c, uint64_t new_pass) {
    uint64_t* sc = c.s.scal;
    if (sc[SC_ACTIVE] == 1) {
        for (int s = 0; s < c.s.SM; ++s) if (c.s.sm_stage[s] != MK_STAGE_DONE) { mk_ref_invalid(c); return; }
        if (*c.s.pe_count != 0) { mk_ref_invalid(c); return; }
    }
    if (*c.s.pe_count != 0) { mk_ref_invalid(c); return; }

    sc[SC_ACTIVE] = 1;
    sc[SC_PASSID] = new_pass;
    for (int i = 0; i < c.s.NCTR; ++i) c.s.counter[i] = 0;
    for (int s = 0; s < c.s.SM; ++s) {
        c.s.sm_pc[s] = 0; c.s.sm_active_inst[s] = 0; c.s.sm_stage[s] = MK_STAGE_IDLE;
        c.s.sm_instr_id[s] = 0; c.s.sm_compute_val[s] = 0;
        c.s.sm_req_count[s] = 0; c.s.sm_wait_count[s] = 0;
        for (int r = 0; r < MK_MAX_PAGE_REQS; ++r) { c.s.sm_alloc[s*MK_MAX_PAGE_REQS+r] = UINT32_MAX; c.s.sm_ready[s*MK_MAX_PAGE_REQS+r] = 0; }
    }
    for (int s = 0; s < c.s.SM; ++s) for (int p = 0; p < c.s.PAGES; ++p) {
        int i = mk_pgi(c, s, p);
        c.s.pg_state[i] = MK_PG_FREE_EMPTY; c.s.pg_tile[i] = 0; c.s.pg_mode[i] = 255;
        c.s.pg_owner[i] = 0; c.s.pg_reqidx[i] = UINT32_MAX; c.s.pg_loadseq[i] = 0; c.s.pg_relseq[i] = 0;
    }
    mk_ref_emit(c, MK_EV_PASS_BEGIN, -1, UINT64_MAX, 0, UINT32_MAX, UINT32_MAX, new_pass);
}

// ------------------------------------------------------- STEP_SM helpers
__device__ void mk_ref_load_active(MkRefCtx& c, int s) {
    const int off = c.s.instr_offset[s] + (int)c.s.sm_pc[s];
    const MkInstr& in = c.s.instrs[off];
    c.s.sm_instr_id[s] = in.instr_id;
    c.s.sm_req_count[s] = in.page_req_count;
    c.s.sm_wait_count[s] = in.wait_count;
    c.s.sm_req_offset[s] = in.req_offset;
    c.s.sm_wait_offset[s] = in.wait_offset;
    c.s.sm_result_seed[s] = in.result_seed;
    c.s.sm_load_lat[s] = in.load_latency;
    c.s.sm_comp_lat[s] = in.compute_latency;
    c.s.sm_store_lat[s] = in.store_latency;
    c.s.sm_out_counter[s] = in.out_counter;
    c.s.sm_out_incr[s] = in.out_increment;
    for (int r = 0; r < MK_MAX_PAGE_REQS; ++r) { c.s.sm_alloc[s*MK_MAX_PAGE_REQS+r] = UINT32_MAX; c.s.sm_ready[s*MK_MAX_PAGE_REQS+r] = 0; }
}

__device__ void mk_ref_pe_push(MkRefCtx& c, uint64_t due, uint8_t kind, int sm, uint64_t inst, uint32_t page) {
    int n = *c.s.pe_count;
    c.s.pe_due[n] = due; c.s.pe_cseq[n] = mk_ref_peek(c); c.s.pe_kind[n] = kind;
    c.s.pe_sm[n] = sm; c.s.pe_inst[n] = inst; c.s.pe_page[n] = page;
    *c.s.pe_count = n + 1;
}

// returns true if stalled (stop STEP_SM)
__device__ bool mk_ref_acquire(MkRefCtx& c, int s) {
    const uint32_t nreq = c.s.sm_req_count[s];
    const uint32_t roff = c.s.sm_req_offset[s];
    uint32_t chosen[MK_MAX_PAGE_REQS];
    uint8_t  is_reuse[MK_MAX_PAGE_REQS];
    for (uint32_t r = 0; r < nreq; ++r) { chosen[r] = UINT32_MAX; is_reuse[r] = 0; }

    bool stalled = false;
    for (uint32_t r = 0; r < nreq; ++r) {
        const MkPageReq& rq = c.s.reqs[roff + r];
        uint32_t pick = UINT32_MAX; uint8_t reuse = 0;
        if (rq.mode == MK_MODE_READ) {
            for (int p = 0; p < c.s.PAGES; ++p) {
                bool taken = false; for (uint32_t k = 0; k < r; ++k) if (chosen[k] == (uint32_t)p) { taken = true; break; }
                if (taken) continue;
                int idx = mk_pgi(c, s, p);
                if (c.s.pg_state[idx] == MK_PG_FREE_RESIDENT && c.s.pg_tile[idx] == rq.tile_id) { pick = (uint32_t)p; reuse = 1; break; }
            }
        }
        if (pick == UINT32_MAX) {
            for (int p = 0; p < c.s.PAGES; ++p) {
                bool taken = false; for (uint32_t k = 0; k < r; ++k) if (chosen[k] == (uint32_t)p) { taken = true; break; }
                if (taken) continue;
                int idx = mk_pgi(c, s, p);
                if (c.s.pg_state[idx] == MK_PG_FREE_EMPTY || c.s.pg_state[idx] == MK_PG_FREE_RESIDENT) { pick = (uint32_t)p; break; }
            }
        }
        if (pick == UINT32_MAX) { stalled = true; break; }
        chosen[r] = pick; is_reuse[r] = reuse;
    }

    if (stalled) {
        mk_ref_emit(c, MK_EV_PAGE_STALL, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], UINT32_MAX, UINT32_MAX, UINT64_MAX);
        return true;
    }

    bool any_loading = false;
    for (uint32_t r = 0; r < nreq; ++r) {
        const MkPageReq& rq = c.s.reqs[roff + r];
        const uint32_t pid = chosen[r];
        int idx = mk_pgi(c, s, (int)pid);
        c.s.sm_alloc[s*MK_MAX_PAGE_REQS+r] = pid;

        if (is_reuse[r]) {
            c.s.pg_state[idx] = MK_PG_HELD_READY;
            c.s.pg_owner[idx] = c.s.sm_active_inst[s];
            c.s.pg_reqidx[idx] = r;
            c.s.sm_ready[s*MK_MAX_PAGE_REQS+r] = 1;
            mk_ref_emit(c, MK_EV_PAGE_REUSE, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], pid, UINT32_MAX, rq.tile_id);
            continue;
        }
        if (c.s.pg_state[idx] == MK_PG_FREE_RESIDENT && c.s.pg_tile[idx] != rq.tile_id) {
            mk_ref_emit(c, MK_EV_PAGE_EVICT, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], pid, UINT32_MAX, c.s.pg_tile[idx]);
        } else if (c.s.pg_state[idx] == MK_PG_FREE_RESIDENT && rq.mode != MK_MODE_READ && c.s.pg_tile[idx] == rq.tile_id) {
            mk_ref_emit(c, MK_EV_PAGE_EVICT, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], pid, UINT32_MAX, c.s.pg_tile[idx]);
        }

        if (rq.mode == MK_MODE_SCRATCH) {
            c.s.pg_state[idx] = MK_PG_HELD_READY;
            c.s.pg_tile[idx] = rq.tile_id; c.s.pg_mode[idx] = rq.mode;
            c.s.pg_owner[idx] = c.s.sm_active_inst[s]; c.s.pg_reqidx[idx] = r;
            c.s.sm_ready[s*MK_MAX_PAGE_REQS+r] = 1;
            mk_ref_emit(c, MK_EV_PAGE_SCRATCH, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], pid, UINT32_MAX, rq.tile_id);
        } else {
            c.s.pg_state[idx] = MK_PG_RESERVED_LOADING;
            c.s.pg_tile[idx] = rq.tile_id; c.s.pg_mode[idx] = rq.mode;
            c.s.pg_owner[idx] = c.s.sm_active_inst[s]; c.s.pg_reqidx[idx] = r;
            c.s.sm_ready[s*MK_MAX_PAGE_REQS+r] = 0;
            mk_ref_pe_push(c, c.s.scal[SC_CLOCK] + c.s.sm_load_lat[s], MK_PEND_LOAD_DONE, s, c.s.sm_active_inst[s], pid);
            any_loading = true;
            mk_ref_emit(c, MK_EV_PAGE_LOAD_ISSUE, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], pid, UINT32_MAX, rq.tile_id);
        }
    }
    c.s.sm_stage[s] = any_loading ? MK_STAGE_LOADING : MK_STAGE_WAIT_DEPS;
    return false;
}

// returns true if should stop STEP_SM loop
__device__ bool mk_ref_step_one(MkRefCtx& c, int s) {
    const uint8_t stage = c.s.sm_stage[s];
    if (stage == MK_STAGE_DONE) {
        mk_ref_emit(c, MK_EV_SM_DONE_WAIT, s, UINT64_MAX, 0, UINT32_MAX, UINT32_MAX, UINT64_MAX);
        return true;
    }
    if (stage == MK_STAGE_IDLE) {
        if (c.s.sm_pc[s] == (uint32_t)c.s.program_len[s]) {
            c.s.sm_stage[s] = MK_STAGE_DONE;
            mk_ref_emit(c, MK_EV_SM_PROGRAM_DONE, s, UINT64_MAX, 0, UINT32_MAX, UINT32_MAX, UINT64_MAX);
            return true;
        }
        c.s.sm_active_inst[s] = c.s.scal[SC_INST_NEXT]++;
        mk_ref_load_active(c, s);
        c.s.sm_stage[s] = MK_STAGE_ACQUIRE;
        mk_ref_emit(c, MK_EV_INSTR_FETCH, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], UINT32_MAX, UINT32_MAX, UINT64_MAX);
        return false;
    }
    if (stage == MK_STAGE_ACQUIRE) return mk_ref_acquire(c, s);
    if (stage == MK_STAGE_LOADING) {
        for (uint32_t r = 0; r < c.s.sm_req_count[s]; ++r) {
            const uint32_t pid = c.s.sm_alloc[s*MK_MAX_PAGE_REQS+r];
            if (pid != UINT32_MAX && c.s.pg_state[mk_pgi(c, s, (int)pid)] == MK_PG_RESERVED_LOADING) {
                mk_ref_emit(c, MK_EV_LOAD_WAIT, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], pid, UINT32_MAX, (uint64_t)r);
                return true;
            }
        }
        c.s.sm_stage[s] = MK_STAGE_WAIT_DEPS;
        return false;
    }
    if (stage == MK_STAGE_WAIT_DEPS) {
        const uint32_t woff = c.s.sm_wait_offset[s];
        for (uint32_t w = 0; w < c.s.sm_wait_count[s]; ++w) {
            const MkWaitRec& wr = c.s.waits[woff + w];
            if (c.s.counter[wr.counter_id] < wr.target) {
                mk_ref_emit(c, MK_EV_COUNTER_WAIT, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], UINT32_MAX, wr.counter_id, wr.target);
                return true;
            }
        }
        uint64_t h = MK_FNV_OFFSET;
        uint64_t v;
        v = c.s.scal[SC_PASSID]; mk_ref_fbytes(&h, &v, 8);
        v = c.s.sm_instr_id[s]; mk_ref_fbytes(&h, &v, 8);
        v = c.s.sm_active_inst[s]; mk_ref_fbytes(&h, &v, 8);
        const uint32_t roff = c.s.sm_req_offset[s];
        for (uint32_t r = 0; r < c.s.sm_req_count[s]; ++r) {
            const MkPageReq& rq = c.s.reqs[roff + r];
            mk_ref_fbytes(&h, &rq.tile_id, 8);
            uint32_t pid = c.s.sm_alloc[s*MK_MAX_PAGE_REQS+r]; mk_ref_fbytes(&h, &pid, 4);
            mk_ref_fbytes(&h, &rq.mode, 1);
        }
        const uint32_t woff2 = c.s.sm_wait_offset[s];
        for (uint32_t w = 0; w < c.s.sm_wait_count[s]; ++w) {
            const MkWaitRec& wr = c.s.waits[woff2 + w];
            mk_ref_fbytes(&h, &wr.counter_id, 4);
            mk_ref_fbytes(&h, &wr.target, 8);
            uint64_t obs = c.s.counter[wr.counter_id]; mk_ref_fbytes(&h, &obs, 8);
        }
        v = c.s.sm_result_seed[s]; mk_ref_fbytes(&h, &v, 8);
        c.s.sm_compute_val[s] = h;

        if (c.s.sm_comp_lat[s] == 0) {
            c.s.sm_stage[s] = MK_STAGE_STORE_READY;
            mk_ref_emit(c, MK_EV_COMPUTE_DONE_INLINE, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], UINT32_MAX, UINT32_MAX, c.s.sm_compute_val[s]);
        } else {
            for (uint32_t r = 0; r < c.s.sm_req_count[s]; ++r) {
                const uint32_t pid = c.s.sm_alloc[s*MK_MAX_PAGE_REQS+r];
                if (pid != UINT32_MAX) c.s.pg_state[mk_pgi(c, s, (int)pid)] = MK_PG_HELD_COMPUTING;
            }
            mk_ref_pe_push(c, c.s.scal[SC_CLOCK] + c.s.sm_comp_lat[s], MK_PEND_COMPUTE_DONE, s, c.s.sm_active_inst[s], UINT32_MAX);
            c.s.sm_stage[s] = MK_STAGE_COMPUTING;
            mk_ref_emit(c, MK_EV_COMPUTE_ISSUE, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], UINT32_MAX, UINT32_MAX, c.s.sm_compute_val[s]);
        }
        return false;
    }
    if (stage == MK_STAGE_COMPUTING) {
        mk_ref_emit(c, MK_EV_COMPUTE_WAIT, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], UINT32_MAX, UINT32_MAX, UINT64_MAX);
        return true;
    }
    if (stage == MK_STAGE_STORE_READY) {
        for (uint32_t r = 0; r < c.s.sm_req_count[s]; ++r) {
            const uint32_t pid = c.s.sm_alloc[s*MK_MAX_PAGE_REQS+r];
            if (pid != UINT32_MAX) c.s.pg_state[mk_pgi(c, s, (int)pid)] = MK_PG_STORE_PENDING;
        }
        mk_ref_pe_push(c, c.s.scal[SC_CLOCK] + c.s.sm_store_lat[s], MK_PEND_STORE_DONE, s, c.s.sm_active_inst[s], UINT32_MAX);
        c.s.sm_stage[s] = MK_STAGE_STORING;
        mk_ref_emit(c, MK_EV_STORE_ISSUE, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], UINT32_MAX, UINT32_MAX, UINT64_MAX);
        return false;
    }
    if (stage == MK_STAGE_STORING) {
        mk_ref_emit(c, MK_EV_STORE_WAIT, s, c.s.sm_instr_id[s], c.s.sm_active_inst[s], UINT32_MAX, UINT32_MAX, UINT64_MAX);
        return true;
    }
    return true;
}

__device__ void mk_ref_step_sm(MkRefCtx& c, int sm_idx, uint32_t lim) {
    if (sm_idx < 0 || sm_idx >= c.s.SM || c.s.scal[SC_ACTIVE] == 0 || lim == 0) { mk_ref_invalid(c); return; }
    for (uint32_t t = 0; t < lim; ++t) if (mk_ref_step_one(c, sm_idx)) break;
}

// ------------------------------------------------------- ADVANCE
__device__ bool mk_ref_pe_less(MkRefCtx& c, int a, int b) {
    if (c.s.pe_due[a] != c.s.pe_due[b]) return c.s.pe_due[a] < c.s.pe_due[b];
    if (c.s.pe_cseq[a] != c.s.pe_cseq[b]) return c.s.pe_cseq[a] < c.s.pe_cseq[b];
    if (c.s.pe_sm[a] != c.s.pe_sm[b]) return c.s.pe_sm[a] < c.s.pe_sm[b];
    if (c.s.pe_inst[a] != c.s.pe_inst[b]) return c.s.pe_inst[a] < c.s.pe_inst[b];
    return c.s.pe_page[a] < c.s.pe_page[b];
}
__device__ void mk_ref_pe_erase(MkRefCtx& c, int idx) {
    int n = *c.s.pe_count;
    for (int i = idx; i < n - 1; ++i) {
        c.s.pe_due[i]=c.s.pe_due[i+1]; c.s.pe_cseq[i]=c.s.pe_cseq[i+1]; c.s.pe_kind[i]=c.s.pe_kind[i+1];
        c.s.pe_sm[i]=c.s.pe_sm[i+1]; c.s.pe_inst[i]=c.s.pe_inst[i+1]; c.s.pe_page[i]=c.s.pe_page[i+1];
    }
    *c.s.pe_count = n - 1;
}

__device__ void mk_ref_process(MkRefCtx& c, uint64_t due, uint8_t kind, int sm, uint64_t inst, uint32_t page) {
    (void)due;
    if (kind == MK_PEND_LOAD_DONE) {
        int idx = mk_pgi(c, sm, (int)page);
        if (c.s.pg_owner[idx] == inst && c.s.pg_state[idx] == MK_PG_RESERVED_LOADING) {
            c.s.pg_state[idx] = MK_PG_HELD_READY;
            c.s.pg_loadseq[idx] = mk_ref_peek(c);
            if (c.s.sm_active_inst[sm] == inst) {
                for (uint32_t r = 0; r < c.s.sm_req_count[sm]; ++r)
                    if (c.s.sm_alloc[sm*MK_MAX_PAGE_REQS+r] == page) c.s.sm_ready[sm*MK_MAX_PAGE_REQS+r] = 1;
            }
            mk_ref_emit(c, MK_EV_LOAD_DONE, sm, UINT64_MAX, inst, page, UINT32_MAX, UINT64_MAX);
        } else {
            mk_ref_emit(c, MK_EV_STALE_EVENT_DROP, sm, UINT64_MAX, inst, page, UINT32_MAX, (uint64_t)kind);
        }
        return;
    }
    if (kind == MK_PEND_COMPUTE_DONE) {
        if (c.s.sm_active_inst[sm] == inst && c.s.sm_stage[sm] == MK_STAGE_COMPUTING) {
            c.s.sm_stage[sm] = MK_STAGE_STORE_READY;
            mk_ref_emit(c, MK_EV_COMPUTE_DONE, sm, c.s.sm_instr_id[sm], inst, UINT32_MAX, UINT32_MAX, c.s.sm_compute_val[sm]);
        } else {
            mk_ref_emit(c, MK_EV_STALE_EVENT_DROP, sm, UINT64_MAX, inst, UINT32_MAX, UINT32_MAX, (uint64_t)kind);
        }
        return;
    }
    // STORE_DONE
    if (c.s.sm_active_inst[sm] == inst && c.s.sm_stage[sm] == MK_STAGE_STORING) {
        mk_ref_emit(c, MK_EV_STORE_DONE, sm, c.s.sm_instr_id[sm], inst, UINT32_MAX, UINT32_MAX, UINT64_MAX);
        const uint32_t roff = c.s.sm_req_offset[sm];
        for (uint32_t r = 0; r < c.s.sm_req_count[sm]; ++r) {
            const uint32_t pid = c.s.sm_alloc[sm*MK_MAX_PAGE_REQS+r];
            if (pid == UINT32_MAX) continue;
            const MkPageReq& rq = c.s.reqs[roff + r];
            int idx = mk_pgi(c, sm, (int)pid);
            c.s.pg_state[idx] = MK_PG_FREE_RESIDENT;
            c.s.pg_owner[idx] = 0; c.s.pg_reqidx[idx] = UINT32_MAX;
            c.s.pg_relseq[idx] = mk_ref_peek(c);
            if (rq.release_after_store == 1)
                mk_ref_emit(c, MK_EV_PAGE_RELEASE, sm, c.s.sm_instr_id[sm], inst, pid, UINT32_MAX, (uint64_t)r);
            else
                mk_ref_emit(c, MK_EV_PAGE_RELEASE_PINNED_HINT, sm, c.s.sm_instr_id[sm], inst, pid, UINT32_MAX, (uint64_t)r);
        }
        if (c.s.sm_out_counter[sm] != UINT32_MAX) {
            uint32_t cid = c.s.sm_out_counter[sm];
            c.s.counter[cid] = c.s.counter[cid] + c.s.sm_out_incr[sm];
            mk_ref_emit(c, MK_EV_COUNTER_INC, sm, c.s.sm_instr_id[sm], inst, UINT32_MAX, cid, c.s.counter[cid]);
        }
        const uint64_t done_instr = c.s.sm_instr_id[sm];
        c.s.sm_active_inst[sm] = 0;
        c.s.sm_pc[sm] += 1;
        c.s.sm_stage[sm] = MK_STAGE_IDLE;
        for (int r = 0; r < MK_MAX_PAGE_REQS; ++r) { c.s.sm_alloc[sm*MK_MAX_PAGE_REQS+r] = UINT32_MAX; c.s.sm_ready[sm*MK_MAX_PAGE_REQS+r] = 0; }
        mk_ref_emit(c, MK_EV_INSTR_COMPLETE, sm, done_instr, inst, UINT32_MAX, UINT32_MAX, UINT64_MAX);
    } else {
        mk_ref_emit(c, MK_EV_STALE_EVENT_DROP, sm, UINT64_MAX, inst, UINT32_MAX, UINT32_MAX, (uint64_t)kind);
    }
}

__device__ void mk_ref_advance(MkRefCtx& c, uint64_t delta, uint32_t max_events) {
    c.s.scal[SC_CLOCK] += delta;
    if (max_events == 0) return;
    uint32_t processed = 0;
    while (processed < max_events) {
        int best = -1;
        const uint64_t clk = c.s.scal[SC_CLOCK];
        for (int i = 0; i < *c.s.pe_count; ++i) {
            if (c.s.pe_due[i] > clk) continue;
            if (best < 0 || mk_ref_pe_less(c, i, best)) best = i;
        }
        if (best < 0) break;
        uint64_t due = c.s.pe_due[best]; uint8_t kind = c.s.pe_kind[best];
        int sm = c.s.pe_sm[best]; uint64_t inst = c.s.pe_inst[best]; uint32_t page = c.s.pe_page[best];
        mk_ref_pe_erase(c, best);
        mk_ref_process(c, due, kind, sm, inst, page);
        ++processed;
    }
}

// ------------------------------------------------------- HOST_INC / ABORT
__device__ void mk_ref_host_inc(MkRefCtx& c, uint32_t cid, uint64_t amount) {
    if (cid >= (uint32_t)c.s.NCTR) { mk_ref_invalid(c); return; }
    c.s.counter[cid] = c.s.counter[cid] + amount;
    mk_ref_emit(c, MK_EV_HOST_COUNTER_INC, -1, UINT64_MAX, 0, UINT32_MAX, cid, c.s.counter[cid]);
}

__device__ void mk_ref_abort(MkRefCtx& c) {
    if (c.s.scal[SC_ACTIVE] == 0) { mk_ref_invalid(c); return; }
    // drop pending in canonical order via repeated selection-min.
    while (*c.s.pe_count > 0) {
        int best = 0;
        for (int i = 1; i < *c.s.pe_count; ++i) if (mk_ref_pe_less(c, i, best)) best = i;
        uint8_t kind = c.s.pe_kind[best]; int sm = c.s.pe_sm[best];
        uint64_t inst = c.s.pe_inst[best]; uint32_t page = c.s.pe_page[best];
        mk_ref_pe_erase(c, best);
        mk_ref_emit(c, MK_EV_STALE_EVENT_DROP, sm, UINT64_MAX, inst, page, UINT32_MAX, (uint64_t)kind);
    }
    for (int s = 0; s < c.s.SM; ++s) {
        c.s.sm_active_inst[s] = 0; c.s.sm_stage[s] = MK_STAGE_DONE;
        for (int r = 0; r < MK_MAX_PAGE_REQS; ++r) { c.s.sm_alloc[s*MK_MAX_PAGE_REQS+r] = UINT32_MAX; c.s.sm_ready[s*MK_MAX_PAGE_REQS+r] = 0; }
    }
    for (int s = 0; s < c.s.SM; ++s) for (int p = 0; p < c.s.PAGES; ++p) {
        int idx = mk_pgi(c, s, p);
        bool nonfree = !(c.s.pg_state[idx] == MK_PG_FREE_EMPTY || c.s.pg_state[idx] == MK_PG_FREE_RESIDENT);
        if (nonfree) mk_ref_emit(c, MK_EV_PAGE_ABORT_RELEASE, s, UINT64_MAX, 0, (uint32_t)p, UINT32_MAX, (uint64_t)c.s.pg_state[idx]);
        c.s.pg_state[idx] = MK_PG_FREE_EMPTY; c.s.pg_tile[idx] = 0; c.s.pg_mode[idx] = 255;
        c.s.pg_owner[idx] = 0; c.s.pg_reqidx[idx] = UINT32_MAX; c.s.pg_loadseq[idx] = 0; c.s.pg_relseq[idx] = 0;
    }
    c.s.scal[SC_ACTIVE] = 0;
    mk_ref_emit(c, MK_EV_PASS_ABORT, -1, UINT64_MAX, 0, UINT32_MAX, UINT32_MAX, UINT64_MAX);
}

// ------------------------------------------------------- snapshot hashes
__device__ uint64_t mk_ref_page_hash(MkRefCtx& c) {
    uint64_t h = MK_FNV_OFFSET;
    for (int s = 0; s < c.s.SM; ++s) for (int p = 0; p < c.s.PAGES; ++p) {
        int idx = mk_pgi(c, s, p);
        uint32_t su = (uint32_t)s, pu = (uint32_t)p;
        mk_ref_fbytes(&h, &su, 4); mk_ref_fbytes(&h, &pu, 4);
        mk_ref_fbytes(&h, &c.s.pg_state[idx], 1);
        mk_ref_fbytes(&h, &c.s.pg_tile[idx], 8);
        mk_ref_fbytes(&h, &c.s.pg_mode[idx], 1);
        mk_ref_fbytes(&h, &c.s.pg_owner[idx], 8);
        mk_ref_fbytes(&h, &c.s.pg_reqidx[idx], 4);
        mk_ref_fbytes(&h, &c.s.pg_loadseq[idx], 8);
        mk_ref_fbytes(&h, &c.s.pg_relseq[idx], 8);
    }
    return h;
}
__device__ uint64_t mk_ref_sm_hash(MkRefCtx& c) {
    uint64_t h = MK_FNV_OFFSET;
    for (int s = 0; s < c.s.SM; ++s) {
        uint32_t su = (uint32_t)s;
        mk_ref_fbytes(&h, &su, 4);
        mk_ref_fbytes(&h, &c.s.sm_pc[s], 4);
        mk_ref_fbytes(&h, &c.s.sm_stage[s], 1);
        mk_ref_fbytes(&h, &c.s.sm_active_inst[s], 8);
        uint64_t iid = (c.s.sm_active_inst[s] == 0) ? UINT64_MAX : c.s.sm_instr_id[s];
        mk_ref_fbytes(&h, &iid, 8);
        mk_ref_fbytes(&h, &c.s.sm_compute_val[s], 8);
        uint32_t n = (c.s.sm_active_inst[s] == 0) ? 0 : c.s.sm_req_count[s];
        for (uint32_t r = 0; r < n; ++r) { uint32_t pid = c.s.sm_alloc[s*MK_MAX_PAGE_REQS+r]; mk_ref_fbytes(&h, &pid, 4); }
    }
    return h;
}
__device__ uint64_t mk_ref_counter_hash(MkRefCtx& c) {
    uint64_t h = MK_FNV_OFFSET;
    for (int i = 0; i < c.s.NCTR; ++i) { uint32_t ci = (uint32_t)i; mk_ref_fbytes(&h, &ci, 4); mk_ref_fbytes(&h, &c.s.counter[i], 8); }
    return h;
}
__device__ uint64_t mk_ref_pending_hash(MkRefCtx& c) {
    // selection over canonical order without mutating.
    uint64_t h = MK_FNV_OFFSET;
    const int n = *c.s.pe_count;
    bool have_prev = false; uint64_t pdue=0, pcseq=0; int psm=0; uint64_t pinst=0; uint32_t ppage=0;
    for (int e = 0; e < n; ++e) {
        int best = -1;
        for (int i = 0; i < n; ++i) {
            // skip already-emitted (strictly less-or-equal to prev)
            if (have_prev) {
                bool gt = false;
                if (c.s.pe_due[i] != pdue) gt = c.s.pe_due[i] > pdue;
                else if (c.s.pe_cseq[i] != pcseq) gt = c.s.pe_cseq[i] > pcseq;
                else if (c.s.pe_sm[i] != psm) gt = c.s.pe_sm[i] > psm;
                else if (c.s.pe_inst[i] != pinst) gt = c.s.pe_inst[i] > pinst;
                else gt = c.s.pe_page[i] > ppage;
                if (!gt) continue;
            }
            if (best < 0 || mk_ref_pe_less(c, i, best)) best = i;
        }
        if (best < 0) break;
        mk_ref_fbytes(&h, &c.s.pe_due[best], 8);
        mk_ref_fbytes(&h, &c.s.pe_cseq[best], 8);
        mk_ref_fbytes(&h, &c.s.pe_kind[best], 1);
        uint32_t smv = (uint32_t)c.s.pe_sm[best]; mk_ref_fbytes(&h, &smv, 4);
        mk_ref_fbytes(&h, &c.s.pe_inst[best], 8);
        mk_ref_fbytes(&h, &c.s.pe_page[best], 4);
        pdue=c.s.pe_due[best]; pcseq=c.s.pe_cseq[best]; psm=c.s.pe_sm[best]; pinst=c.s.pe_inst[best]; ppage=c.s.pe_page[best];
        have_prev = true;
    }
    return h;
}

// ------------------------------------------------------- step kernel
__global__ void mk_ref_step_kernel(
    mk_ref::State s, int op_kind, int a_sm, uint32_t a_counter, uint32_t a_trans, uint32_t a_maxev,
    uint64_t a_pass, uint64_t a_delta, uint64_t a_amount,
    int64_t* out_counts, int32_t* out_opidx, uint64_t* out_clock, uint64_t* out_eseq,
    uint64_t* out_ehash, uint64_t* out_phash, uint64_t* out_smhash, uint64_t* out_chash,
    uint64_t* out_pendhash, uint64_t* out_state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    MkRefCtx c; c.s = s;

    switch (op_kind) {
        case MK_OP_BEGIN_PASS:       mk_ref_begin_pass(c, a_pass); break;
        case MK_OP_STEP_SM:          mk_ref_step_sm(c, a_sm, a_trans); break;
        case MK_OP_ADVANCE:          mk_ref_advance(c, a_delta, a_maxev); break;
        case MK_OP_HOST_INC_COUNTER: mk_ref_host_inc(c, a_counter, a_amount); break;
        case MK_OP_ABORT_PASS:       mk_ref_abort(c); break;
        default: mk_ref_invalid(c); break;
    }
    const int32_t this_op = (int32_t)c.s.scal[SC_OPIDX];
    c.s.scal[SC_OPIDX] += 1;

    const uint64_t ph = mk_ref_page_hash(c);
    const uint64_t sh = mk_ref_sm_hash(c);
    const uint64_t ch = mk_ref_counter_hash(c);
    const uint64_t pendh = mk_ref_pending_hash(c);

    for (int i = 0; i < MK_COUNT_N; ++i) out_counts[i] = c.s.counts[i];
    *out_opidx = this_op;
    *out_clock = c.s.scal[SC_CLOCK];
    *out_eseq = c.s.scal[SC_ESEQ];
    *out_ehash = c.s.scal[SC_EHASH];
    *out_phash = ph; *out_smhash = sh; *out_chash = ch; *out_pendhash = pendh;

    uint64_t mh = MK_FNV_OFFSET;
    mk_ref_fbytes(&mh, &c.s.scal[SC_CLOCK], 8);
    mk_ref_fbytes(&mh, &c.s.scal[SC_ESEQ], 8);
    uint32_t opu = (uint32_t)c.s.scal[SC_OPIDX]; mk_ref_fbytes(&mh, &opu, 4);
    uint8_t pa = (uint8_t)c.s.scal[SC_ACTIVE]; mk_ref_fbytes(&mh, &pa, 1);
    mk_ref_fbytes(&mh, &c.s.scal[SC_PASSID], 8);
    mk_ref_fbytes(&mh, &c.s.scal[SC_INST_NEXT], 8);
    mk_ref_fbytes(&mh, &c.s.scal[SC_EHASH], 8);
    mk_ref_fbytes(&mh, &ph, 8); mk_ref_fbytes(&mh, &sh, 8); mk_ref_fbytes(&mh, &ch, 8); mk_ref_fbytes(&mh, &pendh, 8);
    for (int i = 0; i < MK_COUNT_N; ++i) { uint64_t cv = (uint64_t)c.s.counts[i]; mk_ref_fbytes(&mh, &cv, 8); }
    *out_state = mh;
}

__global__ void mk_ref_reset_kernel(mk_ref::State s) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    s.scal[SC_CLOCK]=0; s.scal[SC_ESEQ]=0; s.scal[SC_INST_NEXT]=1; s.scal[SC_OPIDX]=0;
    s.scal[SC_EHASH]=MK_FNV_OFFSET; s.scal[SC_PASSID]=0; s.scal[SC_ACTIVE]=0;
    for (int i = 0; i < s.NCTR; ++i) s.counter[i] = 0;
    for (int i = 0; i < s.SM; ++i) {
        s.sm_pc[i]=0; s.sm_active_inst[i]=0; s.sm_stage[i]=MK_STAGE_IDLE; s.sm_instr_id[i]=0;
        s.sm_compute_val[i]=0; s.sm_req_count[i]=0; s.sm_wait_count[i]=0;
        for (int r = 0; r < MK_MAX_PAGE_REQS; ++r) { s.sm_alloc[i*MK_MAX_PAGE_REQS+r]=UINT32_MAX; s.sm_ready[i*MK_MAX_PAGE_REQS+r]=0; }
    }
    for (int i = 0; i < s.SM * s.PAGES; ++i) {
        s.pg_state[i]=MK_PG_FREE_EMPTY; s.pg_tile[i]=0; s.pg_mode[i]=255; s.pg_owner[i]=0;
        s.pg_reqidx[i]=UINT32_MAX; s.pg_loadseq[i]=0; s.pg_relseq[i]=0;
    }
    *s.pe_count = 0;
    for (int i = 0; i < MK_COUNT_N; ++i) s.counts[i] = 0;
}

// ------------------------------------------------------- host ABI
extern "C" size_t solution_workspace_bytes(const MkProblemSpec* spec) {
    if (!mk_validate_problem_spec(spec)) return 0;
    return 256;
}

static cudaError_t mk_ref_do_reset(mk_ref::State* st, cudaStream_t stream) {
    mk_ref_reset_kernel<<<1,1,0,stream>>>(*st);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_init(const MkProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!mk_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    mk_ref::State* st = (mk_ref::State*)malloc(sizeof(mk_ref::State));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(mk_ref::State));
    memcpy(&st->spec, spec, sizeof(MkProblemSpec));
    st->SM = spec->sm_count; st->PAGES = spec->pages_per_sm; st->NCTR = spec->counter_count; st->MAXP = spec->max_pending_events;

    const int SM = st->SM, PAGES = st->PAGES, NCTR = st->NCTR, MAXP = st->MAXP;
    const int TI = spec->total_instr, TR = spec->total_reqs, TW = spec->total_waits;

    cudaError_t err = cudaSuccess;
    #define MK_M(p, bytes) do { err = cudaMalloc((void**)&(p), (bytes)); if (err) goto fail; } while (0)
    MK_M(st->program_len, sizeof(int32_t)*SM);
    MK_M(st->instr_offset, sizeof(int32_t)*SM);
    MK_M(st->instrs, sizeof(MkInstr)*(TI > 0 ? TI : 1));
    MK_M(st->reqs, sizeof(MkPageReq)*(TR > 0 ? TR : 1));
    MK_M(st->waits, sizeof(MkWaitRec)*(TW > 0 ? TW : 1));
    MK_M(st->scal, sizeof(uint64_t)*7);
    MK_M(st->counter, sizeof(uint64_t)*NCTR);
    MK_M(st->sm_pc, sizeof(uint32_t)*SM);
    MK_M(st->sm_active_inst, sizeof(uint64_t)*SM);
    MK_M(st->sm_stage, sizeof(uint8_t)*SM);
    MK_M(st->sm_instr_id, sizeof(uint64_t)*SM);
    MK_M(st->sm_compute_val, sizeof(uint64_t)*SM);
    MK_M(st->sm_req_count, sizeof(uint32_t)*SM);
    MK_M(st->sm_wait_count, sizeof(uint32_t)*SM);
    MK_M(st->sm_req_offset, sizeof(uint32_t)*SM);
    MK_M(st->sm_wait_offset, sizeof(uint32_t)*SM);
    MK_M(st->sm_result_seed, sizeof(uint64_t)*SM);
    MK_M(st->sm_load_lat, sizeof(uint64_t)*SM);
    MK_M(st->sm_comp_lat, sizeof(uint64_t)*SM);
    MK_M(st->sm_store_lat, sizeof(uint64_t)*SM);
    MK_M(st->sm_out_counter, sizeof(uint32_t)*SM);
    MK_M(st->sm_out_incr, sizeof(uint64_t)*SM);
    MK_M(st->sm_alloc, sizeof(uint32_t)*SM*MK_MAX_PAGE_REQS);
    MK_M(st->sm_ready, sizeof(uint8_t)*SM*MK_MAX_PAGE_REQS);
    MK_M(st->pg_state, sizeof(uint8_t)*SM*PAGES);
    MK_M(st->pg_tile, sizeof(uint64_t)*SM*PAGES);
    MK_M(st->pg_mode, sizeof(uint8_t)*SM*PAGES);
    MK_M(st->pg_owner, sizeof(uint64_t)*SM*PAGES);
    MK_M(st->pg_reqidx, sizeof(uint32_t)*SM*PAGES);
    MK_M(st->pg_loadseq, sizeof(uint64_t)*SM*PAGES);
    MK_M(st->pg_relseq, sizeof(uint64_t)*SM*PAGES);
    MK_M(st->pe_due, sizeof(uint64_t)*MAXP);
    MK_M(st->pe_cseq, sizeof(uint64_t)*MAXP);
    MK_M(st->pe_kind, sizeof(uint8_t)*MAXP);
    MK_M(st->pe_sm, sizeof(int32_t)*MAXP);
    MK_M(st->pe_inst, sizeof(uint64_t)*MAXP);
    MK_M(st->pe_page, sizeof(uint32_t)*MAXP);
    MK_M(st->pe_count, sizeof(int32_t)*1);
    MK_M(st->counts, sizeof(int64_t)*MK_COUNT_N);
    #undef MK_M

    // copy static program to device.
    if (SM > 0) {
        err = cudaMemcpy(st->program_len, spec->program_len, sizeof(int32_t)*SM, cudaMemcpyHostToDevice); if (err) goto fail;
        err = cudaMemcpy(st->instr_offset, spec->instr_offset, sizeof(int32_t)*SM, cudaMemcpyHostToDevice); if (err) goto fail;
    }
    if (TI > 0) { err = cudaMemcpy(st->instrs, spec->instrs, sizeof(MkInstr)*TI, cudaMemcpyHostToDevice); if (err) goto fail; }
    if (TR > 0) { err = cudaMemcpy(st->reqs, spec->reqs, sizeof(MkPageReq)*TR, cudaMemcpyHostToDevice); if (err) goto fail; }
    if (TW > 0) { err = cudaMemcpy(st->waits, spec->waits, sizeof(MkWaitRec)*TW, cudaMemcpyHostToDevice); if (err) goto fail; }

    err = mk_ref_do_reset(st, stream);
    if (err) goto fail;
    err = cudaStreamSynchronize(stream);
    if (err) goto fail;

    *state_out = st;
    return cudaSuccess;
fail:
    solution_destroy(st);
    return err ? err : cudaErrorMemoryAllocation;
}

extern "C" cudaError_t solution_run(void* state, const MkRunSpec* run, const void* inputs_void,
    void* outputs_void, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    (void)inputs_void; (void)workspace;
    if (!state || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 256) return cudaErrorInvalidValue;
    mk_ref::State* st = (mk_ref::State*)state;
    if (!mk_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;
    MkOutputs* out = (MkOutputs*)outputs_void;
    if (!out->counts || !out->op_index_out || !out->clock_out || !out->event_seq_out ||
        !out->event_hash || !out->page_hash || !out->sm_hash || !out->counter_hash ||
        !out->pending_hash || !out->state_checksum) return cudaErrorInvalidValue;

    mk_ref_step_kernel<<<1,1,0,stream>>>(
        *st, run->op_kind, run->a_sm, run->a_counter, run->a_transition_limit, run->a_max_events,
        run->a_pass_id, run->a_delta, run->a_amount,
        out->counts, out->op_index_out, out->clock_out, out->event_seq_out,
        out->event_hash, out->page_hash, out->sm_hash, out->counter_hash,
        out->pending_hash, out->state_checksum);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return mk_ref_do_reset((mk_ref::State*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    mk_ref::State* st = (mk_ref::State*)state;
    cudaFree(st->program_len); cudaFree(st->instr_offset); cudaFree(st->instrs); cudaFree(st->reqs); cudaFree(st->waits);
    cudaFree(st->scal); cudaFree(st->counter);
    cudaFree(st->sm_pc); cudaFree(st->sm_active_inst); cudaFree(st->sm_stage); cudaFree(st->sm_instr_id);
    cudaFree(st->sm_compute_val); cudaFree(st->sm_req_count); cudaFree(st->sm_wait_count);
    cudaFree(st->sm_req_offset); cudaFree(st->sm_wait_offset); cudaFree(st->sm_result_seed);
    cudaFree(st->sm_load_lat); cudaFree(st->sm_comp_lat); cudaFree(st->sm_store_lat);
    cudaFree(st->sm_out_counter); cudaFree(st->sm_out_incr); cudaFree(st->sm_alloc); cudaFree(st->sm_ready);
    cudaFree(st->pg_state); cudaFree(st->pg_tile); cudaFree(st->pg_mode); cudaFree(st->pg_owner);
    cudaFree(st->pg_reqidx); cudaFree(st->pg_loadseq); cudaFree(st->pg_relseq);
    cudaFree(st->pe_due); cudaFree(st->pe_cseq); cudaFree(st->pe_kind); cudaFree(st->pe_sm);
    cudaFree(st->pe_inst); cudaFree(st->pe_page); cudaFree(st->pe_count); cudaFree(st->counts);
    free(st);
}
