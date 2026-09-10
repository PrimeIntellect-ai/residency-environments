// PMPP_CANARY_67_4103a9dfef -- held-out canary; MUST NOT appear in any submission
// file: mk_pipeline_scoreboard_reference.cu
//
// Reference implementation: direct in-place mutation of persistent device state
// on a single (block 0, thread 0) lane. Persistent state is a struct-of-arrays
// resident in device global memory. All outputs are exact integers. The kernel
// mirrors the canonical MK8 semantics (issue window, RAW/WAR/WAW hazards, buffer
// slot acquire/release, in-flight LOAD/COMPUTE/STORE pending ops ordered by
// (due_clock, op_seq), counters after stores).

#include "mk_pipeline_scoreboard_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RMAX_BUF (MKPS_MAX_READS + MKPS_MAX_WRITES + MKPS_MAX_BUFFERS)

struct MkRefState {
    MkpsProblemSpec spec;

    // scalars[6] = {clock, event_seq, instr_seq_next, op_seq_next, version_seq_next, touch_seq_next}
    uint64_t* scalars;
    uint64_t* evhash;          // [1] running event hash

    // instruction slot arrays [NI]
    uint8_t*  in_used;
    uint32_t* in_id;
    uint64_t* in_seq;
    int32_t*  in_rcount;
    int32_t*  in_wcount;
    uint32_t* in_reads;        // [NI*MKPS_MAX_READS]
    uint32_t* in_writes;       // [NI*MKPS_MAX_WRITES]
    int32_t*  in_scratch;
    uint32_t* in_loadlat;
    uint32_t* in_complat;
    uint32_t* in_storelat;
    uint32_t* in_outcounter;
    uint64_t* in_seed;
    uint8_t*  in_status;
    int32_t*  in_nbuf;
    uint32_t* in_bufid;        // [NI*RMAX_BUF]
    uint8_t*  in_bufrole;      // [NI*RMAX_BUF]
    uint8_t*  in_bufreused;    // [NI*RMAX_BUF]
    uint64_t* in_rver;         // [NI*MKPS_MAX_READS]
    uint64_t* in_wver;         // [NI*MKPS_MAX_WRITES]
    uint8_t*  in_cdf;
    uint8_t*  in_lur;

    // tile arrays [T]
    uint64_t* t_curver;
    uint32_t* t_writer;
    uint64_t* t_prc;
    uint64_t* t_lss;
    uint32_t* t_resbuf;
    uint8_t*  t_dirty;

    // buffer arrays [B]
    uint8_t*  b_state;
    uint32_t* b_tile;
    uint64_t* b_ver;
    uint32_t* b_owner;
    uint64_t* b_pin;
    uint64_t* b_touch;

    // pending arrays [MP] + count [1]
    uint8_t*  p_active;
    uint8_t*  p_kind;
    uint64_t* p_due;
    uint64_t* p_seq;
    uint32_t* p_instr;
    uint32_t* p_buf;
    int32_t*  p_n;

    uint64_t* counter;         // [NC]
    int64_t*  counts;          // [MKPS_COUNT_FIELDS]
};

__device__ __forceinline__ uint64_t rf_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= 1099511628211ULL; return h;
}
__device__ void rf_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = (const uint8_t*)ptr;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = rf_byte(v, p[i]);
    *h = v;
}
#define RFNV_OFF 1469598103934665603ULL

struct RCtx {
    MkRefState st;
    int B, T, NI, MR, MW, WIN, MP, NC;
    uint64_t clock, event_seq, instr_seq_next, op_seq_next, version_seq_next, touch_seq_next;
    uint64_t evhash;
};

__device__ void rf_emit(RCtx* x, uint8_t kind, uint32_t op_index, uint64_t instr_id,
                        uint32_t tile, uint32_t buffer, uint64_t version,
                        uint32_t counter_id, uint64_t aux) {
    uint64_t h = x->evhash;
    uint64_t seq = x->event_seq;
    uint64_t clk = x->clock;
    rf_bytes(&h, &kind, sizeof(uint8_t));
    rf_bytes(&h, &seq, sizeof(uint64_t));
    rf_bytes(&h, &op_index, sizeof(uint32_t));
    rf_bytes(&h, &clk, sizeof(uint64_t));
    rf_bytes(&h, &instr_id, sizeof(uint64_t));
    rf_bytes(&h, &tile, sizeof(uint32_t));
    rf_bytes(&h, &buffer, sizeof(uint32_t));
    rf_bytes(&h, &version, sizeof(uint64_t));
    rf_bytes(&h, &counter_id, sizeof(uint32_t));
    rf_bytes(&h, &aux, sizeof(uint64_t));
    x->evhash = h;
    x->event_seq += 1;
}

__device__ int rf_find_instr(RCtx* x, uint32_t id) {
    if (id == 0) return -1;
    for (int i = 0; i < x->NI; ++i)
        if (x->st.in_used[i] && x->st.in_id[i] == id) return i;
    return -1;
}
__device__ int rf_free_instr_slot(RCtx* x) {
    for (int i = 0; i < x->NI; ++i) if (!x->st.in_used[i]) return i;
    return -1;
}
__device__ bool rf_nonterminal(RCtx* x, int si) {
    if (si < 0 || !x->st.in_used[si]) return false;
    uint8_t s = x->st.in_status[si];
    return s != MKPS_ST_DONE && s != MKPS_ST_CANCELLED;
}

// fill order[] with slot indices sorted by instr_seq (insertion sort, NI small)
__device__ int rf_order(RCtx* x, int* order) {
    int n = 0;
    for (int i = 0; i < x->NI; ++i) if (x->st.in_used[i]) order[n++] = i;
    for (int a = 1; a < n; ++a) {
        int key = order[a];
        uint64_t ks = x->st.in_seq[key];
        int b = a - 1;
        while (b >= 0 && x->st.in_seq[order[b]] > ks) { order[b + 1] = order[b]; --b; }
        order[b + 1] = key;
    }
    return n;
}

__device__ int rf_lowest_free_buf(RCtx* x, const uint8_t* chosen) {
    for (int b = 0; b < x->B; ++b)
        if (x->st.b_state[b] == MKPS_BUF_FREE && !chosen[b]) return b;
    return -1;
}
__device__ int rf_free_buf_count(RCtx* x) {
    int n = 0;
    for (int b = 0; b < x->B; ++b) if (x->st.b_state[b] == MKPS_BUF_FREE) ++n;
    return n;
}
__device__ void rf_push_pend(RCtx* x, uint8_t kind, uint64_t due, uint32_t instr_id, uint32_t buf) {
    int n = *x->st.p_n;
    if (n >= x->MP) return;
    x->st.p_active[n] = 1; x->st.p_kind[n] = kind; x->st.p_due[n] = due;
    x->st.p_seq[n] = x->op_seq_next++; x->st.p_instr[n] = instr_id; x->st.p_buf[n] = buf;
    *x->st.p_n = n + 1;
}

__device__ bool rf_read_resident_cur(RCtx* x, uint32_t tile) {
    uint32_t rb = x->st.t_resbuf[tile];
    if (rb == MKPS_NO_U32) return false;
    return x->st.b_state[rb] == MKPS_BUF_RESIDENT && x->st.b_tile[rb] == tile &&
           x->st.b_ver[rb] == x->st.t_curver[tile];
}

__device__ void rf_clear_buf(RCtx* x, int b) {
    x->st.b_state[b] = MKPS_BUF_FREE; x->st.b_tile[b] = MKPS_NO_U32;
    x->st.b_ver[b] = 0; x->st.b_owner[b] = 0; x->st.b_pin[b] = 0; x->st.b_touch[b] = 0;
}

// ---- ENQUEUE ----
__device__ void rf_enqueue(RCtx* x, uint32_t op_index, uint32_t id, uint32_t rc, uint32_t wc,
                           uint32_t sp, uint32_t llat, uint32_t clat, uint32_t slat,
                           uint32_t outc, uint64_t seed, const uint32_t* reads, const uint32_t* writes) {
    bool ok = true;
    if (id == 0) ok = false;
    else if (rf_find_instr(x, id) >= 0) ok = false;
    else if (rf_free_instr_slot(x) < 0) ok = false;
    else if ((int)rc > x->MR || (int)wc > x->MW) ok = false;
    if (ok) for (uint32_t r = 0; r < rc; ++r) if (reads[r] >= (uint32_t)x->T) { ok = false; break; }
    if (ok) for (uint32_t w = 0; w < wc; ++w) if (writes[w] >= (uint32_t)x->T) { ok = false; break; }
    if (ok) {
        for (uint32_t a = 0; a < wc && ok; ++a)
            for (uint32_t b = a + 1; b < wc; ++b)
                if (writes[a] == writes[b]) { ok = false; break; }
    }
    if (ok) {
        int distinct_r = 0;
        for (uint32_t a = 0; a < rc; ++a) {
            bool seen = false;
            for (uint32_t b = 0; b < a; ++b) if (reads[b] == reads[a]) { seen = true; break; }
            if (!seen) ++distinct_r;
        }
        int need = (int)sp + distinct_r + (int)wc;
        if (need > x->B) ok = false;
    }
    if (!ok) {
        x->st.counts[MKC_invalid_count] += 1;
        rf_emit(x, MKEV_INVALID, op_index, (uint64_t)id, MKPS_NO_U32, MKPS_NO_U32, 0, MKPS_NO_U32, 0);
        return;
    }
    int slot = rf_free_instr_slot(x);
    x->st.in_used[slot] = 1;
    x->st.in_id[slot] = id;
    x->st.in_seq[slot] = x->instr_seq_next++;
    x->st.in_rcount[slot] = (int)rc;
    x->st.in_wcount[slot] = (int)wc;
    for (uint32_t r = 0; r < rc; ++r) x->st.in_reads[slot * MKPS_MAX_READS + r] = reads[r];
    for (uint32_t w = 0; w < wc; ++w) x->st.in_writes[slot * MKPS_MAX_WRITES + w] = writes[w];
    x->st.in_scratch[slot] = (int)sp;
    x->st.in_loadlat[slot] = llat;
    x->st.in_complat[slot] = clat;
    x->st.in_storelat[slot] = slat;
    x->st.in_outcounter[slot] = outc;
    x->st.in_seed[slot] = seed;
    x->st.in_status[slot] = MKPS_ST_QUEUED;
    x->st.in_nbuf[slot] = 0;
    x->st.in_cdf[slot] = 0;
    x->st.in_lur[slot] = 0;
    for (uint32_t r = 0; r < rc; ++r) x->st.t_prc[reads[r]] += 1;
    x->st.counts[MKC_instr_enqueued] += 1;
    rf_emit(x, MKEV_INSTR_ENQUEUE, op_index, (uint64_t)id, MKPS_NO_U32, MKPS_NO_U32, 0,
            MKPS_NO_U32, x->st.in_seq[slot]);
}

// ---- ISSUE_LOADS ----
__device__ void rf_issue_loads(RCtx* x, uint32_t op_index, uint32_t limit) {
    if (limit == 0) return;
    int order[MKPS_MAX_INSTRS];
    int n = rf_order(x, order);
    int issued = 0, window_seen = 0;
    uint8_t chosen[MKPS_MAX_BUFFERS];
    for (int oi = 0; oi < n; ++oi) {
        int si = order[oi];
        if (!rf_nonterminal(x, si)) continue;
        if (window_seen >= x->WIN) break;
        window_seen += 1;
        if (issued >= (int)limit) break;
        if (x->st.in_status[si] != MKPS_ST_QUEUED) continue;

        int rc = x->st.in_rcount[si], wc = x->st.in_wcount[si];
        bool waw = false, raw = false, war = false;
        for (int oj = 0; oj < oi && !waw; ++oj) {
            int ej = order[oj];
            if (!rf_nonterminal(x, ej)) continue;
            int ewc = x->st.in_wcount[ej];
            for (int w = 0; w < wc && !waw; ++w)
                for (int ew = 0; ew < ewc; ++ew)
                    if (x->st.in_writes[ej * MKPS_MAX_WRITES + ew] ==
                        x->st.in_writes[si * MKPS_MAX_WRITES + w]) { waw = true; break; }
        }
        if (!waw) {
            for (int r = 0; r < rc && !raw; ++r) {
                uint32_t tl = x->st.in_reads[si * MKPS_MAX_READS + r];
                uint32_t writer = x->st.t_writer[tl];
                if (writer != 0) {
                    int wsi = rf_find_instr(x, writer);
                    if (wsi >= 0 && rf_nonterminal(x, wsi) &&
                        x->st.in_seq[wsi] < x->st.in_seq[si] && !rf_read_resident_cur(x, tl))
                        raw = true;
                }
            }
        }
        if (!waw && !raw) {
            for (int w = 0; w < wc && !war; ++w) {
                uint32_t tl = x->st.in_writes[si * MKPS_MAX_WRITES + w];
                for (int oj = 0; oj < oi && !war; ++oj) {
                    int ej = order[oj];
                    if (!rf_nonterminal(x, ej)) continue;
                    if (x->st.in_lur[ej]) continue;
                    int erc = x->st.in_rcount[ej];
                    for (int er = 0; er < erc; ++er)
                        if (x->st.in_reads[ej * MKPS_MAX_READS + er] == tl) { war = true; break; }
                }
            }
        }
        if (waw || raw || war) {
            x->st.counts[MKC_load_hazard_stall] += 1;
            rf_emit(x, MKEV_LOAD_HAZARD_STALL, op_index, (uint64_t)x->st.in_id[si],
                    MKPS_NO_U32, MKPS_NO_U32, 0, MKPS_NO_U32, 0);
            break;
        }
        // capacity
        int need = 0;
        for (int r = 0; r < rc; ++r) {
            uint32_t tl = x->st.in_reads[si * MKPS_MAX_READS + r];
            bool dup = false;
            for (int q = 0; q < r; ++q)
                if (x->st.in_reads[si * MKPS_MAX_READS + q] == tl) { dup = true; break; }
            if (dup) continue;
            if (!rf_read_resident_cur(x, tl)) need += 1;
        }
        need += wc + x->st.in_scratch[si];
        if (need > rf_free_buf_count(x)) {
            x->st.counts[MKC_load_hazard_stall] += 1;
            rf_emit(x, MKEV_LOAD_HAZARD_STALL, op_index, (uint64_t)x->st.in_id[si],
                    MKPS_NO_U32, MKPS_NO_U32, 0, MKPS_NO_U32, 0);
            continue;
        }
        // allocate
        for (int b = 0; b < x->B; ++b) chosen[b] = 0;
        int nb = 0;
        bool any_load = false;
        uint32_t id = x->st.in_id[si];
        for (int r = 0; r < rc; ++r) {
            uint32_t tl = x->st.in_reads[si * MKPS_MAX_READS + r];
            x->st.in_rver[si * MKPS_MAX_READS + r] = x->st.t_curver[tl];
            if (rf_read_resident_cur(x, tl)) {
                uint32_t bid = x->st.t_resbuf[tl];
                x->st.b_pin[bid] += 1;
                x->st.b_touch[bid] = x->touch_seq_next++;
                chosen[bid] = 1;
                x->st.in_bufid[si * RMAX_BUF + nb] = bid;
                x->st.in_bufrole[si * RMAX_BUF + nb] = MKPS_ROLE_READ;
                x->st.in_bufreused[si * RMAX_BUF + nb] = 1;
                nb++;
            } else {
                int bid = rf_lowest_free_buf(x, chosen);
                chosen[bid] = 1;
                x->st.b_state[bid] = MKPS_BUF_LOADING;
                x->st.b_tile[bid] = tl;
                x->st.b_ver[bid] = x->st.t_curver[tl];
                x->st.b_owner[bid] = id;
                x->st.b_pin[bid] += 1;
                x->st.b_touch[bid] = x->touch_seq_next++;
                x->st.in_bufid[si * RMAX_BUF + nb] = (uint32_t)bid;
                x->st.in_bufrole[si * RMAX_BUF + nb] = MKPS_ROLE_READ;
                x->st.in_bufreused[si * RMAX_BUF + nb] = 0;
                nb++;
                any_load = true;
            }
        }
        for (int w = 0; w < wc; ++w) {
            uint32_t tl = x->st.in_writes[si * MKPS_MAX_WRITES + w];
            int bid = rf_lowest_free_buf(x, chosen);
            chosen[bid] = 1;
            uint64_t wv = x->version_seq_next++;
            x->st.in_wver[si * MKPS_MAX_WRITES + w] = wv;
            x->st.b_state[bid] = MKPS_BUF_LOADING;
            x->st.b_tile[bid] = tl;
            x->st.b_ver[bid] = wv;
            x->st.b_owner[bid] = id;
            x->st.b_pin[bid] += 1;
            x->st.b_touch[bid] = x->touch_seq_next++;
            x->st.t_writer[tl] = id;
            x->st.in_bufid[si * RMAX_BUF + nb] = (uint32_t)bid;
            x->st.in_bufrole[si * RMAX_BUF + nb] = MKPS_ROLE_WRITE;
            x->st.in_bufreused[si * RMAX_BUF + nb] = 0;
            nb++;
            any_load = true;
        }
        int sp = x->st.in_scratch[si];
        for (int s = 0; s < sp; ++s) {
            int bid = rf_lowest_free_buf(x, chosen);
            chosen[bid] = 1;
            x->st.b_state[bid] = MKPS_BUF_LOADING;
            x->st.b_tile[bid] = MKPS_NO_U32;
            x->st.b_ver[bid] = 0;
            x->st.b_owner[bid] = id;
            x->st.b_pin[bid] += 1;
            x->st.b_touch[bid] = x->touch_seq_next++;
            x->st.in_bufid[si * RMAX_BUF + nb] = (uint32_t)bid;
            x->st.in_bufrole[si * RMAX_BUF + nb] = MKPS_ROLE_SCRATCH;
            x->st.in_bufreused[si * RMAX_BUF + nb] = 0;
            nb++;
            any_load = true;
        }
        x->st.in_nbuf[si] = nb;

        if (any_load) {
            x->st.in_status[si] = MKPS_ST_LOAD_ISSUED;
            x->st.counts[MKC_load_issue] += 1;
            for (int k = 0; k < nb; ++k) {
                if (x->st.in_bufreused[si * RMAX_BUF + k]) continue;
                uint32_t bid = x->st.in_bufid[si * RMAX_BUF + k];
                uint32_t tile = (x->st.in_bufrole[si * RMAX_BUF + k] == MKPS_ROLE_SCRATCH) ?
                                MKPS_NO_U32 : x->st.b_tile[bid];
                rf_emit(x, MKEV_LOAD_ISSUE, op_index, (uint64_t)id, tile, bid, x->st.b_ver[bid],
                        MKPS_NO_U32, 0);
            }
            for (int k = 0; k < nb; ++k) {
                if (x->st.in_bufreused[si * RMAX_BUF + k]) continue;
                rf_push_pend(x, MKPS_PEND_LOAD_DONE, x->clock + (uint64_t)x->st.in_loadlat[si],
                             id, x->st.in_bufid[si * RMAX_BUF + k]);
            }
        } else {
            x->st.in_status[si] = MKPS_ST_LOAD_READY;
            for (int k = 0; k < nb; ++k)
                x->st.b_state[x->st.in_bufid[si * RMAX_BUF + k]] = MKPS_BUF_READY_READ;
            x->st.counts[MKC_load_ready_inline] += 1;
            rf_emit(x, MKEV_LOAD_READY_INLINE, op_index, (uint64_t)id, MKPS_NO_U32,
                    MKPS_NO_U32, 0, MKPS_NO_U32, 0);
        }
        issued += 1;
    }
}

__device__ bool rf_any_loading(RCtx* x, int si) {
    int nb = x->st.in_nbuf[si];
    for (int k = 0; k < nb; ++k)
        if (x->st.b_state[x->st.in_bufid[si * RMAX_BUF + k]] == MKPS_BUF_LOADING) return true;
    return false;
}

// ---- ISSUE_COMPUTE ----
__device__ void rf_issue_compute(RCtx* x, uint32_t op_index, uint32_t limit) {
    if (limit == 0) return;
    int order[MKPS_MAX_INSTRS];
    int n = rf_order(x, order);
    int issued = 0;
    for (int oi = 0; oi < n; ++oi) {
        if (issued >= (int)limit) break;
        int si = order[oi];
        uint8_t st = x->st.in_status[si];
        if (st == MKPS_ST_LOAD_ISSUED) {
            if (rf_any_loading(x, si)) {
                x->st.counts[MKC_compute_load_wait] += 1;
                rf_emit(x, MKEV_COMPUTE_LOAD_WAIT, op_index, (uint64_t)x->st.in_id[si],
                        MKPS_NO_U32, MKPS_NO_U32, 0, MKPS_NO_U32, 0);
                break;
            }
            continue;
        }
        if (st != MKPS_ST_LOAD_READY) continue;
        int nb = x->st.in_nbuf[si];
        for (int k = 0; k < nb; ++k)
            x->st.b_state[x->st.in_bufid[si * RMAX_BUF + k]] = MKPS_BUF_COMPUTING;
        rf_push_pend(x, MKPS_PEND_COMPUTE_DONE, x->clock + (uint64_t)x->st.in_complat[si],
                     x->st.in_id[si], MKPS_NO_U32);
        x->st.in_status[si] = MKPS_ST_COMPUTE_ISSUED;
        x->st.counts[MKC_compute_issue] += 1;
        rf_emit(x, MKEV_COMPUTE_ISSUE, op_index, (uint64_t)x->st.in_id[si], MKPS_NO_U32,
                MKPS_NO_U32, 0, MKPS_NO_U32, 0);
        issued += 1;
    }
}

// ---- ISSUE_STORES ----
__device__ void rf_issue_stores(RCtx* x, uint32_t op_index, uint32_t limit) {
    if (limit == 0) return;
    int order[MKPS_MAX_INSTRS];
    int n = rf_order(x, order);
    int issued = 0;
    for (int oi = 0; oi < n; ++oi) {
        if (issued >= (int)limit) break;
        int si = order[oi];
        if (x->st.in_status[si] != MKPS_ST_COMPUTE_ISSUED) continue;
        if (!x->st.in_cdf[si]) continue;
        int nb = x->st.in_nbuf[si];
        for (int k = 0; k < nb; ++k)
            if (x->st.in_bufrole[si * RMAX_BUF + k] == MKPS_ROLE_WRITE)
                x->st.b_state[x->st.in_bufid[si * RMAX_BUF + k]] = MKPS_BUF_STORING;
        rf_push_pend(x, MKPS_PEND_STORE_DONE, x->clock + (uint64_t)x->st.in_storelat[si],
                     x->st.in_id[si], MKPS_NO_U32);
        x->st.in_status[si] = MKPS_ST_STORE_ISSUED;
        x->st.counts[MKC_store_issue] += 1;
        rf_emit(x, MKEV_STORE_ISSUE, op_index, (uint64_t)x->st.in_id[si], MKPS_NO_U32,
                MKPS_NO_U32, 0, MKPS_NO_U32, 0);
        issued += 1;
    }
}

__device__ uint64_t rf_compute_result(RCtx* x, int si) {
    uint64_t h = RFNV_OFF;
    uint32_t id = x->st.in_id[si];
    uint64_t seq = x->st.in_seq[si];
    uint64_t ps = x->st.in_seed[si];
    rf_bytes(&h, &id, sizeof(uint32_t));
    rf_bytes(&h, &seq, sizeof(uint64_t));
    rf_bytes(&h, &ps, sizeof(uint64_t));
    int rc = x->st.in_rcount[si], wc = x->st.in_wcount[si];
    for (int r = 0; r < rc; ++r) {
        uint64_t v = x->st.in_rver[si * MKPS_MAX_READS + r];
        rf_bytes(&h, &v, sizeof(uint64_t));
    }
    for (int w = 0; w < wc; ++w) {
        uint64_t v = x->st.in_wver[si * MKPS_MAX_WRITES + w];
        rf_bytes(&h, &v, sizeof(uint64_t));
    }
    int nb = x->st.in_nbuf[si];
    for (int k = 0; k < nb; ++k) {
        uint32_t b = x->st.in_bufid[si * RMAX_BUF + k];
        rf_bytes(&h, &b, sizeof(uint32_t));
    }
    return h;
}

// ---- ADVANCE op processors ----
__device__ void rf_load_done(RCtx* x, uint32_t op_index, uint8_t kind, uint32_t instr_id, uint32_t buf) {
    int si = rf_find_instr(x, instr_id);
    bool stale = false;
    if (si < 0) stale = true;
    else if (!rf_nonterminal(x, si) || x->st.b_owner[buf] != instr_id ||
             x->st.b_state[buf] != MKPS_BUF_LOADING) stale = true;
    if (stale) {
        x->st.counts[MKC_op_stale_drop] += 1;
        rf_emit(x, MKEV_OP_STALE_DROP, op_index, (uint64_t)instr_id, MKPS_NO_U32, buf, 0,
                MKPS_NO_U32, (uint64_t)kind);
        return;
    }
    uint8_t role = MKPS_ROLE_SCRATCH;
    int nb = x->st.in_nbuf[si];
    for (int k = 0; k < nb; ++k)
        if (x->st.in_bufid[si * RMAX_BUF + k] == buf) { role = x->st.in_bufrole[si * RMAX_BUF + k]; break; }
    x->st.b_state[buf] = (role == MKPS_ROLE_WRITE) ? MKPS_BUF_READY_WRITE : MKPS_BUF_READY_READ;
    x->st.b_touch[buf] = x->touch_seq_next++;
    x->st.counts[MKC_load_done] += 1;
    rf_emit(x, MKEV_LOAD_DONE, op_index, (uint64_t)instr_id, x->st.b_tile[buf], buf,
            x->st.b_ver[buf], MKPS_NO_U32, 0);
    if (x->st.in_status[si] == MKPS_ST_LOAD_ISSUED && !rf_any_loading(x, si)) {
        x->st.in_status[si] = MKPS_ST_LOAD_READY;
        x->st.counts[MKC_instr_load_ready] += 1;
        rf_emit(x, MKEV_INSTR_LOAD_READY, op_index, (uint64_t)instr_id, MKPS_NO_U32,
                MKPS_NO_U32, 0, MKPS_NO_U32, 0);
    }
}

__device__ void rf_compute_done(RCtx* x, uint32_t op_index, uint8_t kind, uint32_t instr_id) {
    int si = rf_find_instr(x, instr_id);
    if (si < 0 || x->st.in_status[si] != MKPS_ST_COMPUTE_ISSUED) {
        x->st.counts[MKC_op_stale_drop] += 1;
        rf_emit(x, MKEV_OP_STALE_DROP, op_index, (uint64_t)instr_id, MKPS_NO_U32, MKPS_NO_U32, 0,
                MKPS_NO_U32, (uint64_t)kind);
        return;
    }
    uint64_t result = rf_compute_result(x, si);
    x->st.in_cdf[si] = 1;
    x->st.counts[MKC_compute_done] += 1;
    rf_emit(x, MKEV_COMPUTE_DONE, op_index, (uint64_t)instr_id, MKPS_NO_U32, MKPS_NO_U32, 0,
            MKPS_NO_U32, result);
}

__device__ void rf_store_done(RCtx* x, uint32_t op_index, uint8_t kind, uint32_t instr_id) {
    int si = rf_find_instr(x, instr_id);
    if (si < 0 || x->st.in_status[si] != MKPS_ST_STORE_ISSUED) {
        x->st.counts[MKC_op_stale_drop] += 1;
        rf_emit(x, MKEV_OP_STALE_DROP, op_index, (uint64_t)instr_id, MKPS_NO_U32, MKPS_NO_U32, 0,
                MKPS_NO_U32, (uint64_t)kind);
        return;
    }
    int rc = x->st.in_rcount[si], wc = x->st.in_wcount[si], nb = x->st.in_nbuf[si];
    for (int w = 0; w < wc; ++w) {
        uint32_t tl = x->st.in_writes[si * MKPS_MAX_WRITES + w];
        uint64_t wv = x->st.in_wver[si * MKPS_MAX_WRITES + w];
        uint32_t wbuf = MKPS_NO_U32;
        for (int k = 0; k < nb; ++k)
            if (x->st.in_bufrole[si * RMAX_BUF + k] == MKPS_ROLE_WRITE &&
                x->st.b_tile[x->st.in_bufid[si * RMAX_BUF + k]] == tl) {
                wbuf = x->st.in_bufid[si * RMAX_BUF + k]; break;
            }
        x->st.t_curver[tl] = wv;
        x->st.t_writer[tl] = 0;
        x->st.t_lss[tl] = x->event_seq;
        x->st.t_dirty[tl] = 0;
        if (wbuf != MKPS_NO_U32) {
            x->st.b_state[wbuf] = MKPS_BUF_RESIDENT;
            x->st.b_tile[wbuf] = tl;
            x->st.b_ver[wbuf] = wv;
            x->st.b_touch[wbuf] = x->touch_seq_next++;
            x->st.t_resbuf[tl] = wbuf;
        }
        x->st.counts[MKC_tile_store] += 1;
        rf_emit(x, MKEV_TILE_STORE, op_index, (uint64_t)instr_id, tl, wbuf, wv, MKPS_NO_U32, 0);
    }
    for (int r = 0; r < rc; ++r) {
        uint32_t tl = x->st.in_reads[si * MKPS_MAX_READS + r];
        if (x->st.t_prc[tl] > 0) x->st.t_prc[tl] -= 1;
    }
    x->st.in_lur[si] = 1;
    for (int k = 0; k < nb; ++k) {
        uint8_t role = x->st.in_bufrole[si * RMAX_BUF + k];
        if (role == MKPS_ROLE_WRITE) continue;
        uint32_t bid = x->st.in_bufid[si * RMAX_BUF + k];
        if (role == MKPS_ROLE_READ) {
            uint32_t tl = x->st.b_tile[bid];
            bool zero_readers = (tl < (uint32_t)x->T) ? (x->st.t_prc[tl] == 0) : true;
            bool is_res_cur = (tl < (uint32_t)x->T) && x->st.t_resbuf[tl] == bid &&
                              x->st.b_ver[bid] == x->st.t_curver[tl] &&
                              x->st.b_state[bid] == MKPS_BUF_RESIDENT;
            if (zero_readers && !is_res_cur) {
                if (tl < (uint32_t)x->T && x->st.t_resbuf[tl] == bid) x->st.t_resbuf[tl] = MKPS_NO_U32;
                rf_clear_buf(x, bid);
                x->st.counts[MKC_read_release] += 1;
                rf_emit(x, MKEV_READ_RELEASE, op_index, (uint64_t)instr_id, tl, bid, 0, MKPS_NO_U32, 0);
            }
        } else { // scratch
            rf_clear_buf(x, bid);
            x->st.counts[MKC_read_release] += 1;
            rf_emit(x, MKEV_READ_RELEASE, op_index, (uint64_t)instr_id, MKPS_NO_U32, bid, 0, MKPS_NO_U32, 0);
        }
    }
    uint32_t outc = x->st.in_outcounter[si];
    if (outc != MKPS_NO_U32 && outc < (uint32_t)x->NC) {
        x->st.counter[outc] += 1;
        x->st.counts[MKC_counter_inc] += 1;
        rf_emit(x, MKEV_COUNTER_INC, op_index, (uint64_t)instr_id, MKPS_NO_U32, MKPS_NO_U32, 0,
                outc, x->st.counter[outc]);
    }
    x->st.in_status[si] = MKPS_ST_DONE;
    x->st.counts[MKC_instr_done] += 1;
    rf_emit(x, MKEV_INSTR_DONE, op_index, (uint64_t)instr_id, MKPS_NO_U32, MKPS_NO_U32, 0,
            MKPS_NO_U32, 0);
}

__device__ void rf_compact_pending(RCtx* x) {
    int w = 0, n = *x->st.p_n;
    for (int i = 0; i < n; ++i) {
        if (!x->st.p_active[i]) continue;
        if (w != i) {
            x->st.p_active[w] = x->st.p_active[i];
            x->st.p_kind[w] = x->st.p_kind[i];
            x->st.p_due[w] = x->st.p_due[i];
            x->st.p_seq[w] = x->st.p_seq[i];
            x->st.p_instr[w] = x->st.p_instr[i];
            x->st.p_buf[w] = x->st.p_buf[i];
        }
        ++w;
    }
    for (int i = w; i < n; ++i) {
        x->st.p_active[i] = 0; x->st.p_kind[i] = 0; x->st.p_due[i] = 0;
        x->st.p_seq[i] = 0; x->st.p_instr[i] = 0; x->st.p_buf[i] = 0;
    }
    *x->st.p_n = w;
}

__device__ void rf_advance(RCtx* x, uint32_t op_index, uint32_t delta, uint32_t max_ops) {
    x->clock += (uint64_t)delta;
    uint32_t processed = 0;
    while (processed < max_ops) {
        int best = -1;
        int n = *x->st.p_n;
        for (int i = 0; i < n; ++i) {
            if (!x->st.p_active[i]) continue;
            if (x->st.p_due[i] > x->clock) continue;
            if (best < 0 || x->st.p_due[i] < x->st.p_due[best] ||
                (x->st.p_due[i] == x->st.p_due[best] && x->st.p_seq[i] < x->st.p_seq[best]))
                best = i;
        }
        if (best < 0) break;
        uint8_t kind = x->st.p_kind[best];
        uint32_t instr = x->st.p_instr[best];
        uint32_t buf = x->st.p_buf[best];
        x->st.p_active[best] = 0;
        if (kind == MKPS_PEND_LOAD_DONE) rf_load_done(x, op_index, kind, instr, buf);
        else if (kind == MKPS_PEND_COMPUTE_DONE) rf_compute_done(x, op_index, kind, instr);
        else rf_store_done(x, op_index, kind, instr);
        ++processed;
    }
    rf_compact_pending(x);
}

// ---- CANCEL ----
__device__ void rf_cancel(RCtx* x, uint32_t op_index, uint32_t id) {
    int si = rf_find_instr(x, id);
    if (si < 0 || !rf_nonterminal(x, si)) {
        x->st.counts[MKC_invalid_count] += 1;
        rf_emit(x, MKEV_INVALID, op_index, (uint64_t)id, MKPS_NO_U32, MKPS_NO_U32, 0, MKPS_NO_U32, 0);
        return;
    }
    x->st.in_status[si] = MKPS_ST_CANCELLED;
    int n = *x->st.p_n;
    for (int i = 0; i < n; ++i)
        if (x->st.p_active[i] && x->st.p_instr[i] == id) x->st.p_active[i] = 0;
    for (int b = 0; b < x->B; ++b) {
        if (x->st.b_owner[b] == id && x->st.b_state[b] != MKPS_BUF_FREE &&
            x->st.b_state[b] != MKPS_BUF_RESIDENT) {
            uint32_t tl = x->st.b_tile[b];
            if (tl < (uint32_t)x->T && x->st.t_resbuf[tl] == (uint32_t)b) x->st.t_resbuf[tl] = MKPS_NO_U32;
            rf_clear_buf(x, b);
            x->st.counts[MKC_buffer_cancel_release] += 1;
            rf_emit(x, MKEV_BUFFER_CANCEL_RELEASE, op_index, (uint64_t)id, tl, (uint32_t)b, 0, MKPS_NO_U32, 0);
        }
    }
    if (!x->st.in_lur[si]) {
        int rc = x->st.in_rcount[si];
        for (int r = 0; r < rc; ++r) {
            uint32_t tl = x->st.in_reads[si * MKPS_MAX_READS + r];
            if (x->st.t_prc[tl] > 0) x->st.t_prc[tl] -= 1;
        }
    }
    int wc = x->st.in_wcount[si];
    for (int w = 0; w < wc; ++w) {
        uint32_t tl = x->st.in_writes[si * MKPS_MAX_WRITES + w];
        if (x->st.t_writer[tl] == id) x->st.t_writer[tl] = 0;
    }
    x->st.counts[MKC_instr_cancel] += 1;
    rf_emit(x, MKEV_INSTR_CANCEL, op_index, (uint64_t)id, MKPS_NO_U32, MKPS_NO_U32, 0, MKPS_NO_U32, 0);
    rf_compact_pending(x);
}

// ---- HOST_COUNTER ----
__device__ void rf_host_counter(RCtx* x, uint32_t op_index, uint32_t cid, uint32_t amount) {
    if (cid >= (uint32_t)x->NC) {
        x->st.counts[MKC_invalid_count] += 1;
        rf_emit(x, MKEV_INVALID, op_index, 0, MKPS_NO_U32, MKPS_NO_U32, 0, MKPS_NO_U32, 0);
        return;
    }
    x->st.counter[cid] += (uint64_t)amount;
    x->st.counts[MKC_host_counter_inc] += 1;
    rf_emit(x, MKEV_HOST_COUNTER_INC, op_index, 0, MKPS_NO_U32, MKPS_NO_U32, 0, cid, x->st.counter[cid]);
}

// ---- hashes ----
__device__ uint64_t rf_instr_hash(RCtx* x) {
    uint64_t h = RFNV_OFF;
    int order[MKPS_MAX_INSTRS];
    int n = rf_order(x, order);
    for (int oi = 0; oi < n; ++oi) {
        int si = order[oi];
        uint8_t st = x->st.in_status[si];
        if (st == MKPS_ST_DONE || st == MKPS_ST_CANCELLED) continue;
        uint32_t id = x->st.in_id[si];
        uint64_t seq = x->st.in_seq[si];
        uint32_t rc = (uint32_t)x->st.in_rcount[si], wc = (uint32_t)x->st.in_wcount[si];
        uint32_t sp = (uint32_t)x->st.in_scratch[si];
        uint8_t cdf = x->st.in_cdf[si], lur = x->st.in_lur[si];
        rf_bytes(&h, &id, sizeof(uint32_t));
        rf_bytes(&h, &seq, sizeof(uint64_t));
        rf_bytes(&h, &st, sizeof(uint8_t));
        rf_bytes(&h, &rc, sizeof(uint32_t));
        rf_bytes(&h, &wc, sizeof(uint32_t));
        rf_bytes(&h, &sp, sizeof(uint32_t));
        rf_bytes(&h, &cdf, sizeof(uint8_t));
        rf_bytes(&h, &lur, sizeof(uint8_t));
        for (uint32_t r = 0; r < rc; ++r) {
            uint32_t tl = x->st.in_reads[si * MKPS_MAX_READS + r];
            uint64_t v = x->st.in_rver[si * MKPS_MAX_READS + r];
            rf_bytes(&h, &tl, sizeof(uint32_t));
            rf_bytes(&h, &v, sizeof(uint64_t));
        }
        for (uint32_t w = 0; w < wc; ++w) {
            uint32_t tl = x->st.in_writes[si * MKPS_MAX_WRITES + w];
            uint64_t v = x->st.in_wver[si * MKPS_MAX_WRITES + w];
            rf_bytes(&h, &tl, sizeof(uint32_t));
            rf_bytes(&h, &v, sizeof(uint64_t));
        }
        uint32_t nb = (uint32_t)x->st.in_nbuf[si];
        rf_bytes(&h, &nb, sizeof(uint32_t));
        for (uint32_t k = 0; k < nb; ++k) {
            uint32_t b = x->st.in_bufid[si * RMAX_BUF + k];
            uint8_t role = x->st.in_bufrole[si * RMAX_BUF + k];
            rf_bytes(&h, &b, sizeof(uint32_t));
            rf_bytes(&h, &role, sizeof(uint8_t));
        }
    }
    return h;
}

__device__ uint64_t rf_tile_hash(RCtx* x) {
    uint64_t h = RFNV_OFF;
    for (int l = 0; l < x->T; ++l) {
        uint32_t tl = (uint32_t)l;
        uint64_t cv = x->st.t_curver[l];
        uint64_t wi = (uint64_t)x->st.t_writer[l];
        uint64_t prc = x->st.t_prc[l];
        uint64_t lss = x->st.t_lss[l];
        uint32_t rb = x->st.t_resbuf[l];
        uint8_t d = x->st.t_dirty[l];
        rf_bytes(&h, &tl, sizeof(uint32_t));
        rf_bytes(&h, &cv, sizeof(uint64_t));
        rf_bytes(&h, &wi, sizeof(uint64_t));
        rf_bytes(&h, &prc, sizeof(uint64_t));
        rf_bytes(&h, &lss, sizeof(uint64_t));
        rf_bytes(&h, &rb, sizeof(uint32_t));
        rf_bytes(&h, &d, sizeof(uint8_t));
    }
    return h;
}

__device__ uint64_t rf_buffer_hash(RCtx* x) {
    uint64_t h = RFNV_OFF;
    for (int b = 0; b < x->B; ++b) {
        uint32_t id = (uint32_t)b;
        uint8_t st = x->st.b_state[b];
        uint32_t tl = x->st.b_tile[b];
        uint64_t ver = x->st.b_ver[b];
        uint64_t own = (uint64_t)x->st.b_owner[b];
        uint64_t pin = x->st.b_pin[b];
        uint64_t lt = x->st.b_touch[b];
        rf_bytes(&h, &id, sizeof(uint32_t));
        rf_bytes(&h, &st, sizeof(uint8_t));
        rf_bytes(&h, &tl, sizeof(uint32_t));
        rf_bytes(&h, &ver, sizeof(uint64_t));
        rf_bytes(&h, &own, sizeof(uint64_t));
        rf_bytes(&h, &pin, sizeof(uint64_t));
        rf_bytes(&h, &lt, sizeof(uint64_t));
    }
    return h;
}

__device__ uint64_t rf_pending_hash(RCtx* x) {
    uint64_t h = RFNV_OFF;
    int n = *x->st.p_n;
    for (int i = 0; i < n; ++i) {
        if (!x->st.p_active[i]) continue;
        uint8_t kind = x->st.p_kind[i];
        uint64_t due = x->st.p_due[i];
        uint64_t seq = x->st.p_seq[i];
        uint64_t iid = (uint64_t)x->st.p_instr[i];
        uint32_t bid = x->st.p_buf[i];
        rf_bytes(&h, &kind, sizeof(uint8_t));
        rf_bytes(&h, &due, sizeof(uint64_t));
        rf_bytes(&h, &seq, sizeof(uint64_t));
        rf_bytes(&h, &iid, sizeof(uint64_t));
        rf_bytes(&h, &bid, sizeof(uint32_t));
    }
    return h;
}

__device__ uint64_t rf_counter_hash(RCtx* x) {
    uint64_t h = RFNV_OFF;
    for (int c = 0; c < x->NC; ++c) {
        uint32_t id = (uint32_t)c;
        uint64_t v = x->st.counter[c];
        rf_bytes(&h, &id, sizeof(uint32_t));
        rf_bytes(&h, &v, sizeof(uint64_t));
    }
    return h;
}

__device__ uint64_t rf_state_checksum(RCtx* x, uint64_t ih, uint64_t th, uint64_t bh,
                                      uint64_t ph, uint64_t cnh) {
    uint64_t h = RFNV_OFF;
    int b = x->B, t = x->T, ni = x->NI, win = x->WIN, mp = x->MP, nc = x->NC;
    rf_bytes(&h, &b, sizeof(int32_t));
    rf_bytes(&h, &t, sizeof(int32_t));
    rf_bytes(&h, &ni, sizeof(int32_t));
    rf_bytes(&h, &win, sizeof(int32_t));
    rf_bytes(&h, &mp, sizeof(int32_t));
    rf_bytes(&h, &nc, sizeof(int32_t));
    rf_bytes(&h, &x->clock, sizeof(uint64_t));
    rf_bytes(&h, &x->event_seq, sizeof(uint64_t));
    rf_bytes(&h, &x->instr_seq_next, sizeof(uint64_t));
    rf_bytes(&h, &x->op_seq_next, sizeof(uint64_t));
    rf_bytes(&h, &x->version_seq_next, sizeof(uint64_t));
    rf_bytes(&h, &ih, sizeof(uint64_t));
    rf_bytes(&h, &th, sizeof(uint64_t));
    rf_bytes(&h, &bh, sizeof(uint64_t));
    rf_bytes(&h, &ph, sizeof(uint64_t));
    rf_bytes(&h, &cnh, sizeof(uint64_t));
    for (int i = 0; i < MKPS_COUNT_FIELDS; ++i) {
        int64_t v = x->st.counts[i];
        rf_bytes(&h, &v, sizeof(int64_t));
    }
    return h;
}

__global__ void rf_reset_kernel(MkRefState st, int B, int T, int NI, int MP, int NC) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    st.scalars[0] = 0; st.scalars[1] = 0; st.scalars[2] = 1; st.scalars[3] = 1;
    st.scalars[4] = 1; st.scalars[5] = 1;
    st.evhash[0] = RFNV_OFF;
    for (int i = 0; i < NI; ++i) {
        st.in_used[i] = 0; st.in_id[i] = 0; st.in_seq[i] = 0; st.in_rcount[i] = 0;
        st.in_wcount[i] = 0; st.in_scratch[i] = 0; st.in_loadlat[i] = 0; st.in_complat[i] = 0;
        st.in_storelat[i] = 0; st.in_outcounter[i] = MKPS_NO_U32; st.in_seed[i] = 0;
        st.in_status[i] = MKPS_ST_QUEUED; st.in_nbuf[i] = 0; st.in_cdf[i] = 0; st.in_lur[i] = 0;
        for (int r = 0; r < MKPS_MAX_READS; ++r) { st.in_reads[i * MKPS_MAX_READS + r] = 0; st.in_rver[i * MKPS_MAX_READS + r] = 0; }
        for (int w = 0; w < MKPS_MAX_WRITES; ++w) { st.in_writes[i * MKPS_MAX_WRITES + w] = 0; st.in_wver[i * MKPS_MAX_WRITES + w] = 0; }
        for (int k = 0; k < RMAX_BUF; ++k) { st.in_bufid[i * RMAX_BUF + k] = 0; st.in_bufrole[i * RMAX_BUF + k] = 0; st.in_bufreused[i * RMAX_BUF + k] = 0; }
    }
    for (int l = 0; l < T; ++l) {
        st.t_curver[l] = 0; st.t_writer[l] = 0; st.t_prc[l] = 0; st.t_lss[l] = 0;
        st.t_resbuf[l] = MKPS_NO_U32; st.t_dirty[l] = 0;
    }
    for (int b = 0; b < B; ++b) {
        st.b_state[b] = MKPS_BUF_FREE; st.b_tile[b] = MKPS_NO_U32; st.b_ver[b] = 0;
        st.b_owner[b] = 0; st.b_pin[b] = 0; st.b_touch[b] = 0;
    }
    for (int i = 0; i < MP; ++i) {
        st.p_active[i] = 0; st.p_kind[i] = 0; st.p_due[i] = 0; st.p_seq[i] = 0;
        st.p_instr[i] = 0; st.p_buf[i] = 0;
    }
    *st.p_n = 0;
    for (int c = 0; c < NC; ++c) st.counter[c] = 0;
    for (int i = 0; i < MKPS_COUNT_FIELDS; ++i) st.counts[i] = 0;
}

__global__ void rf_step_kernel(
    MkRefState st, int B, int T, int NI, int MR, int MW, int WIN, int MP, int NC,
    int batch_size, const int32_t* op, const uint32_t* a0, const uint32_t* a1,
    const uint32_t* a2, const uint32_t* a3, const uint32_t* a4, const uint32_t* a5,
    const uint32_t* a6, const uint32_t* a7, const uint64_t* a8, const uint32_t* tiles,
    int64_t* out_counts, uint64_t* out_evhash, uint64_t* out_ih, uint64_t* out_th,
    uint64_t* out_bh, uint64_t* out_ph, uint64_t* out_ch, uint64_t* out_evseq,
    uint64_t* out_state) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    RCtx x;
    x.st = st;
    x.B = B; x.T = T; x.NI = NI; x.MR = MR; x.MW = MW; x.WIN = WIN; x.MP = MP; x.NC = NC;
    x.clock = st.scalars[0];
    x.event_seq = st.scalars[1];
    x.instr_seq_next = st.scalars[2];
    x.op_seq_next = st.scalars[3];
    x.version_seq_next = st.scalars[4];
    x.touch_seq_next = st.scalars[5];
    x.evhash = st.evhash[0];

    for (int i = 0; i < batch_size; ++i) {
        int o = op[i];
        const uint32_t* reads = &tiles[(size_t)i * 16];
        const uint32_t* writes = &tiles[(size_t)i * 16 + 8];
        if (o == MKPS_OP_ENQUEUE)
            rf_enqueue(&x, (uint32_t)i, a0[i], a1[i], a2[i], a3[i], a4[i], a5[i], a6[i], a7[i], a8[i], reads, writes);
        else if (o == MKPS_OP_ISSUE_LOADS) rf_issue_loads(&x, (uint32_t)i, a0[i]);
        else if (o == MKPS_OP_ISSUE_COMPUTE) rf_issue_compute(&x, (uint32_t)i, a0[i]);
        else if (o == MKPS_OP_ISSUE_STORES) rf_issue_stores(&x, (uint32_t)i, a0[i]);
        else if (o == MKPS_OP_ADVANCE) rf_advance(&x, (uint32_t)i, a0[i], a1[i]);
        else if (o == MKPS_OP_CANCEL) rf_cancel(&x, (uint32_t)i, a0[i]);
        else if (o == MKPS_OP_HOST_COUNTER) rf_host_counter(&x, (uint32_t)i, a0[i], a1[i]);
        else {
            x.st.counts[MKC_invalid_count] += 1;
            rf_emit(&x, MKEV_INVALID, (uint32_t)i, 0, MKPS_NO_U32, MKPS_NO_U32, 0, MKPS_NO_U32, 0);
        }
    }

    st.scalars[0] = x.clock;
    st.scalars[1] = x.event_seq;
    st.scalars[2] = x.instr_seq_next;
    st.scalars[3] = x.op_seq_next;
    st.scalars[4] = x.version_seq_next;
    st.scalars[5] = x.touch_seq_next;
    st.evhash[0] = x.evhash;

    uint64_t ih = rf_instr_hash(&x);
    uint64_t th = rf_tile_hash(&x);
    uint64_t bh = rf_buffer_hash(&x);
    uint64_t ph = rf_pending_hash(&x);
    uint64_t cnh = rf_counter_hash(&x);

    for (int i = 0; i < MKPS_COUNT_FIELDS; ++i) out_counts[i] = x.st.counts[i];
    out_evhash[0] = x.evhash;
    out_ih[0] = ih; out_th[0] = th; out_bh[0] = bh; out_ph[0] = ph; out_ch[0] = cnh;
    out_evseq[0] = x.event_seq;
    out_state[0] = rf_state_checksum(&x, ih, th, bh, ph, cnh);
}

static cudaError_t rf_do_reset(MkRefState* st, cudaStream_t stream) {
    rf_reset_kernel<<<1, 1, 0, stream>>>(*st, st->spec.buffer_count, st->spec.tile_count,
                                         st->spec.max_instrs, st->spec.max_pending_ops,
                                         st->spec.counter_count);
    return cudaPeekAtLastError();
}

extern "C" size_t solution_workspace_bytes(const MkpsProblemSpec* spec) {
    if (!mkps_validate_problem_spec(spec)) return 0;
    return 128;
}

#define RALLOC(field, type, count) \
    do { err = cudaMalloc((void**)&st->field, sizeof(type) * (size_t)(count)); if (err) goto fail; } while (0)

extern "C" cudaError_t solution_init(
    const MkpsProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!mkps_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    MkRefState* st = (MkRefState*)malloc(sizeof(MkRefState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(MkRefState));
    memcpy(&st->spec, spec, sizeof(MkpsProblemSpec));

    const int NI = spec->max_instrs, T = spec->tile_count, B = spec->buffer_count;
    const int MP = spec->max_pending_ops, NC = spec->counter_count;
    cudaError_t err = cudaSuccess;

    RALLOC(scalars, uint64_t, 6);
    RALLOC(evhash, uint64_t, 1);
    RALLOC(in_used, uint8_t, NI);
    RALLOC(in_id, uint32_t, NI);
    RALLOC(in_seq, uint64_t, NI);
    RALLOC(in_rcount, int32_t, NI);
    RALLOC(in_wcount, int32_t, NI);
    RALLOC(in_reads, uint32_t, (size_t)NI * MKPS_MAX_READS);
    RALLOC(in_writes, uint32_t, (size_t)NI * MKPS_MAX_WRITES);
    RALLOC(in_scratch, int32_t, NI);
    RALLOC(in_loadlat, uint32_t, NI);
    RALLOC(in_complat, uint32_t, NI);
    RALLOC(in_storelat, uint32_t, NI);
    RALLOC(in_outcounter, uint32_t, NI);
    RALLOC(in_seed, uint64_t, NI);
    RALLOC(in_status, uint8_t, NI);
    RALLOC(in_nbuf, int32_t, NI);
    RALLOC(in_bufid, uint32_t, (size_t)NI * RMAX_BUF);
    RALLOC(in_bufrole, uint8_t, (size_t)NI * RMAX_BUF);
    RALLOC(in_bufreused, uint8_t, (size_t)NI * RMAX_BUF);
    RALLOC(in_rver, uint64_t, (size_t)NI * MKPS_MAX_READS);
    RALLOC(in_wver, uint64_t, (size_t)NI * MKPS_MAX_WRITES);
    RALLOC(in_cdf, uint8_t, NI);
    RALLOC(in_lur, uint8_t, NI);
    RALLOC(t_curver, uint64_t, T);
    RALLOC(t_writer, uint32_t, T);
    RALLOC(t_prc, uint64_t, T);
    RALLOC(t_lss, uint64_t, T);
    RALLOC(t_resbuf, uint32_t, T);
    RALLOC(t_dirty, uint8_t, T);
    RALLOC(b_state, uint8_t, B);
    RALLOC(b_tile, uint32_t, B);
    RALLOC(b_ver, uint64_t, B);
    RALLOC(b_owner, uint32_t, B);
    RALLOC(b_pin, uint64_t, B);
    RALLOC(b_touch, uint64_t, B);
    RALLOC(p_active, uint8_t, MP);
    RALLOC(p_kind, uint8_t, MP);
    RALLOC(p_due, uint64_t, MP);
    RALLOC(p_seq, uint64_t, MP);
    RALLOC(p_instr, uint32_t, MP);
    RALLOC(p_buf, uint32_t, MP);
    RALLOC(p_n, int32_t, 1);
    RALLOC(counter, uint64_t, NC);
    RALLOC(counts, int64_t, MKPS_COUNT_FIELDS);

    err = rf_do_reset(st, stream);
    if (err) goto fail;
    *state_out = st;
    return cudaSuccess;
fail:
    solution_destroy(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state, const MkpsRunSpec* run, const void* inputs_void,
    void* outputs_void, void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;
    if (!state || !inputs_void || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 1) return cudaErrorInvalidValue;

    MkRefState* st = (MkRefState*)state;
    if (!mkps_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const MkpsInputs* in = (const MkpsInputs*)inputs_void;
    MkpsOutputs* out = (MkpsOutputs*)outputs_void;

    if (run->batch_size > 0 &&
        (!in->op || !in->a0 || !in->a1 || !in->a2 || !in->a3 || !in->a4 || !in->a5 ||
         !in->a6 || !in->a7 || !in->a8 || !in->tiles)) {
        return cudaErrorInvalidValue;
    }
    if (!out->counts || !out->pipeline_event_hash || !out->instr_hash || !out->tile_hash ||
        !out->buffer_hash || !out->pending_hash || !out->counter_hash || !out->event_seq_out ||
        !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    rf_step_kernel<<<1, 1, 0, stream>>>(
        *st, st->spec.buffer_count, st->spec.tile_count, st->spec.max_instrs,
        st->spec.max_reads_per_instr, st->spec.max_writes_per_instr, st->spec.issue_window,
        st->spec.max_pending_ops, st->spec.counter_count, run->batch_size,
        in->op, in->a0, in->a1, in->a2, in->a3, in->a4, in->a5, in->a6, in->a7, in->a8, in->tiles,
        out->counts, out->pipeline_event_hash, out->instr_hash, out->tile_hash, out->buffer_hash,
        out->pending_hash, out->counter_hash, out->event_seq_out, out->state_checksum);

    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return rf_do_reset((MkRefState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    MkRefState* st = (MkRefState*)state;
    cudaFree(st->scalars); cudaFree(st->evhash);
    cudaFree(st->in_used); cudaFree(st->in_id); cudaFree(st->in_seq); cudaFree(st->in_rcount);
    cudaFree(st->in_wcount); cudaFree(st->in_reads); cudaFree(st->in_writes); cudaFree(st->in_scratch);
    cudaFree(st->in_loadlat); cudaFree(st->in_complat); cudaFree(st->in_storelat);
    cudaFree(st->in_outcounter); cudaFree(st->in_seed); cudaFree(st->in_status); cudaFree(st->in_nbuf);
    cudaFree(st->in_bufid); cudaFree(st->in_bufrole); cudaFree(st->in_bufreused);
    cudaFree(st->in_rver); cudaFree(st->in_wver); cudaFree(st->in_cdf); cudaFree(st->in_lur);
    cudaFree(st->t_curver); cudaFree(st->t_writer); cudaFree(st->t_prc); cudaFree(st->t_lss);
    cudaFree(st->t_resbuf); cudaFree(st->t_dirty);
    cudaFree(st->b_state); cudaFree(st->b_tile); cudaFree(st->b_ver); cudaFree(st->b_owner);
    cudaFree(st->b_pin); cudaFree(st->b_touch);
    cudaFree(st->p_active); cudaFree(st->p_kind); cudaFree(st->p_due); cudaFree(st->p_seq);
    cudaFree(st->p_instr); cudaFree(st->p_buf); cudaFree(st->p_n);
    cudaFree(st->counter); cudaFree(st->counts);
    free(st);
}
