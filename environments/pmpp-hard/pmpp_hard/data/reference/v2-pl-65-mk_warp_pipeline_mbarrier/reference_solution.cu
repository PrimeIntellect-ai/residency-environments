// PMPP_CANARY_65_f0a69ed333 -- held-out canary; MUST NOT appear in any submission
// file: mk_warp_pipeline_mbarrier_reference.cu
//
// Reference implementation of the MK6 warp-specialized mbarrier pipeline.
//
// Data-structure strategy (independent of naive + oracle):
//   * Tile table = flat slot array, lookup by linear scan keyed by tile_id with a
//     "used" flag. Terminal tiles are kept (used stays 1) so id reuse is rejected.
//   * Input queue = ring buffer of tile_ids (head + size).
//   * Buffers and barriers are flat arrays indexed by id.
//   * Compute/store ready queues are kept as UNSORTED flat arrays; the
//     (tile_seq, buffer id) minimum is selected at use time (selection scan).
//   * Pending async events are a flat array in creation order; ADVANCE selects
//     the due minimum by (due_clock, async_seq) via linear scan, then compacts.
// Everything runs in a single <<<1,1>>> kernel mutating persistent global state.

#include "mk_warp_pipeline_mbarrier_common.h"

#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MKWP_FNV_OFFSET 1469598103934665603ULL
#define MKWP_FNV_PRIME  1099511628211ULL

// Cooperative block width for the step kernel. Only thread 0 performs the serial
// state mutation and the small ordered hashes; all threads cooperate on the
// tile-hash rank scatter (the dominant cost in the single-thread oracle).
#define MKWP_STEP_BLOCK 256

namespace mkwp_ref {

struct State {
    MkwpProblemSpec spec;

    // Scalars (device): indices below.
    //  0 clock, 1 event_seq, 2 op_index, 3 tile_seq_next, 4 phase_seq_next,
    //  5 async_seq_next, 6 pipe_hash.
    uint64_t* scal;            // 7 u64

    // Tile table (flat slots, capacity = max_tiles).
    uint8_t*  t_used;          // [cap]
    uint64_t* t_id;            // [cap]
    uint64_t* t_seq;           // [cap]
    uint64_t* t_load;          // [cap]
    uint64_t* t_compute;       // [cap]
    uint64_t* t_store;         // [cap]
    uint64_t* t_seed;          // [cap]
    int32_t*  t_status;        // [cap]
    uint32_t* t_assigned;      // [cap]
    uint64_t* t_result;        // [cap]

    // Input queue ring of tile_ids.
    uint64_t* iq_buf;          // [cap]
    int32_t*  iq_head;         // [1]
    int32_t*  iq_size;         // [1]

    // Buffers.
    int32_t*  b_state;         // [B]
    uint64_t* b_tile;          // [B]
    uint32_t* b_lbar;          // [B]
    uint32_t* b_cbar;          // [B]
    uint32_t* b_sbar;          // [B]
    uint8_t*  b_owner;         // [B]
    uint64_t* b_lastphase;     // [B]

    // Barriers.
    uint64_t* br_phase;        // [BR]
    uint32_t* br_expected;     // [BR]
    uint32_t* br_arrived;      // [BR]
    uint8_t*  br_done;         // [BR]
    uint64_t* br_waitmask;     // [BR]
    uint64_t* br_lastarrive;   // [BR]

    // Pending async events (flat, creation order).
    int32_t*  as_kind;         // [P]
    uint64_t* as_due;          // [P]
    uint64_t* as_seq;          // [P]
    uint64_t* as_tile;         // [P]
    uint32_t* as_buf;          // [P]
    uint32_t* as_bar;          // [P]
    uint64_t* as_phase;        // [P]
    int32_t*  as_count;        // [1]

    // Role ready queues (flat unsorted lists of buffer ids).
    uint32_t* cq_buf;          // [Q]
    int32_t*  cq_count;        // [1]
    uint32_t* sq_buf;          // [Q]
    int32_t*  sq_count;        // [1]

    // Output mirror.
    int64_t*  counts;          // [MKWP_COUNT_N]

    // Scratch for parallel tile-hash ordering (ascending-id rank scatter). [cap]
    uint32_t* tile_order;
};

}  // namespace mkwp_ref

// ---------------------------------------------------------------- device FNV
__device__ __forceinline__ uint64_t mkwp_ref_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= MKWP_FNV_PRIME;
    return h;
}
__device__ void mkwp_ref_fnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*)p;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = mkwp_ref_fnv_byte(v, q[i]);
    *h = v;
}
__device__ uint64_t mkwp_ref_result_hash(uint64_t tile_id, uint64_t tile_seq,
                                         uint64_t seed, uint64_t buf, uint64_t clk) {
    uint64_t h = MKWP_FNV_OFFSET;
    mkwp_ref_fnv_bytes(&h, &tile_id, 8);
    mkwp_ref_fnv_bytes(&h, &tile_seq, 8);
    mkwp_ref_fnv_bytes(&h, &seed, 8);
    mkwp_ref_fnv_bytes(&h, &buf, 8);
    mkwp_ref_fnv_bytes(&h, &clk, 8);
    return h;
}

struct MkwpRefCtx {
    mkwp_ref::State s;
    int B, BR, cap, P, Q;
    int loader_warps, compute_warps, storer_warps;
    int max_tiles;
};

// scal indices
#define SC_CLOCK 0
#define SC_ESEQ 1
#define SC_OPIDX 2
#define SC_TSEQ 3
#define SC_PSEQ 4
#define SC_ASEQ 5
#define SC_PIPE 6

__device__ __forceinline__ bool mkwp_ref_terminal(int32_t st) {
    return st == MKWP_TS_DONE || st == MKWP_TS_CANCELLED;
}

// ------------------------------------------------------------ tile table
__device__ int mkwp_ref_find(const MkwpRefCtx& c, uint64_t id) {
    for (int i = 0; i < c.cap; ++i) {
        if (c.s.t_used[i] && c.s.t_id[i] == id) return i;
    }
    return -1;
}
__device__ int mkwp_ref_alloc(const MkwpRefCtx& c) {
    for (int i = 0; i < c.cap; ++i) if (!c.s.t_used[i]) return i;
    return -1;
}
__device__ int mkwp_ref_live_count(const MkwpRefCtx& c) {
    int n = 0;
    for (int i = 0; i < c.cap; ++i) if (c.s.t_used[i]) ++n;
    return n;
}

// ------------------------------------------------------------ input queue ring
__device__ void mkwp_ref_iq_push(MkwpRefCtx& c, uint64_t id) {
    const int head = *c.s.iq_head;
    const int sz = *c.s.iq_size;
    const int pos = (head + sz) % c.cap;
    c.s.iq_buf[pos] = id;
    *c.s.iq_size = sz + 1;
}
__device__ uint64_t mkwp_ref_iq_front(const MkwpRefCtx& c) {
    return c.s.iq_buf[*c.s.iq_head];
}
__device__ void mkwp_ref_iq_pop(MkwpRefCtx& c) {
    *c.s.iq_head = (*c.s.iq_head + 1) % c.cap;
    *c.s.iq_size = *c.s.iq_size - 1;
}

// ------------------------------------------------------------ event emit
__device__ void mkwp_ref_emit(MkwpRefCtx& c, uint8_t kind, uint32_t role,
                              uint64_t tile_id, uint32_t buffer, uint32_t barrier,
                              uint64_t phase, uint64_t aux) {
    uint64_t* scal = c.s.scal;
    const uint64_t seq = scal[SC_ESEQ];
    uint64_t h = scal[SC_PIPE];
    uint32_t opi = (uint32_t)scal[SC_OPIDX];
    uint64_t clk = scal[SC_CLOCK];
    mkwp_ref_fnv_bytes(&h, &kind, 1);
    mkwp_ref_fnv_bytes(&h, &seq, 8);
    mkwp_ref_fnv_bytes(&h, &opi, 4);
    mkwp_ref_fnv_bytes(&h, &clk, 8);
    mkwp_ref_fnv_bytes(&h, &role, 4);
    mkwp_ref_fnv_bytes(&h, &tile_id, 8);
    mkwp_ref_fnv_bytes(&h, &buffer, 4);
    mkwp_ref_fnv_bytes(&h, &barrier, 4);
    mkwp_ref_fnv_bytes(&h, &phase, 8);
    mkwp_ref_fnv_bytes(&h, &aux, 8);
    scal[SC_PIPE] = h;
    scal[SC_ESEQ] = seq + 1;
}
__device__ void mkwp_ref_emit_invalid(MkwpRefCtx& c) {
    c.s.counts[MKWP_C_INVALID] += 1;
    mkwp_ref_emit(c, MKWP_EV_INVALID, UINT32_MAX, 0, UINT32_MAX, UINT32_MAX, UINT64_MAX, 0);
}

// ------------------------------------------------------------ role queues
__device__ void mkwp_ref_rq_push(MkwpRefCtx& c, uint32_t* buf, int32_t* count, uint32_t bid) {
    buf[*count] = bid;
    *count = *count + 1;
}
__device__ void mkwp_ref_rq_erase(MkwpRefCtx& c, uint32_t* buf, int32_t* count, uint32_t bid) {
    const int n = *count;
    int found = -1;
    for (int i = 0; i < n; ++i) if (buf[i] == bid) { found = i; break; }
    if (found < 0) return;
    for (int i = found; i < n - 1; ++i) buf[i] = buf[i + 1];
    *count = n - 1;
}
// Select buffer id with smallest (tile_seq, buffer id). Returns buffer id or
// UINT32_MAX. Does not remove.
__device__ uint32_t mkwp_ref_rq_min(MkwpRefCtx& c, const uint32_t* buf, int32_t count) {
    if (count <= 0) return UINT32_MAX;
    uint32_t best = UINT32_MAX;
    uint64_t bseq = 0;
    for (int i = 0; i < count; ++i) {
        const uint32_t bid = buf[i];
        const uint64_t tid = c.s.b_tile[bid];
        const int slot = mkwp_ref_find(c, tid);
        const uint64_t ts = (slot >= 0) ? c.s.t_seq[slot] : UINT64_MAX;
        bool take = false;
        if (best == UINT32_MAX) take = true;
        else if (ts < bseq) take = true;
        else if (ts == bseq && bid < best) take = true;
        if (take) { best = bid; bseq = ts; }
    }
    return best;
}

// ------------------------------------------------------------ operations
__device__ void mkwp_ref_op_enqueue(MkwpRefCtx& c, uint64_t tile_id, uint64_t lb,
                                    uint64_t ci, uint64_t sb, uint64_t seed) {
    const int ex = mkwp_ref_find(c, tile_id);
    const bool exists_nonterminal = (ex >= 0) && !mkwp_ref_terminal(c.s.t_status[ex]);
    if (exists_nonterminal || mkwp_ref_live_count(c) >= c.max_tiles ||
        lb == 0 || ci == 0 || sb == 0) {
        mkwp_ref_emit_invalid(c);
        return;
    }
    // Reuse path: a terminal tile with the same id is replaced (oracle keys the
    // tile table by id). Free its slot first so find() is unambiguous and the
    // live count stays equal to the oracle's map size.
    if (ex >= 0) c.s.t_used[ex] = 0;
    const int slot = mkwp_ref_alloc(c);
    c.s.t_used[slot] = 1;
    c.s.t_id[slot] = tile_id;
    const uint64_t tseq = c.s.scal[SC_TSEQ]++;
    c.s.t_seq[slot] = tseq;
    c.s.t_load[slot] = lb;
    c.s.t_compute[slot] = ci;
    c.s.t_store[slot] = sb;
    c.s.t_seed[slot] = seed;
    c.s.t_status[slot] = MKWP_TS_QUEUED;
    c.s.t_assigned[slot] = UINT32_MAX;
    c.s.t_result[slot] = 0;
    mkwp_ref_iq_push(c, tile_id);
    c.s.counts[MKWP_C_TILE_ENQUEUE] += 1;
    mkwp_ref_emit(c, MKWP_EV_TILE_ENQUEUE, UINT32_MAX, tile_id, UINT32_MAX,
                  UINT32_MAX, UINT64_MAX, tseq);
}

__device__ int mkwp_ref_lowest_empty(MkwpRefCtx& c) {
    for (int b = 0; b < c.B; ++b) if (c.s.b_state[b] == MKWP_BS_EMPTY) return b;
    return -1;
}

__device__ void mkwp_ref_op_loader(MkwpRefCtx& c, int loader_id, uint64_t limit) {
    if (loader_id < 0 || loader_id >= c.loader_warps || limit == 0) {
        mkwp_ref_emit_invalid(c);
        return;
    }
    for (uint64_t i = 0; i < limit; ++i) {
        const int b = mkwp_ref_lowest_empty(c);
        if (b < 0) {
            c.s.counts[MKWP_C_LOADER_NO_BUFFER] += 1;
            mkwp_ref_emit(c, MKWP_EV_LOADER_NO_BUFFER, (uint32_t)loader_id, 0,
                          UINT32_MAX, UINT32_MAX, UINT64_MAX, 0);
            return;
        }
        // pop head noncancelled tile from input queue.
        int slot = -1;
        uint64_t tid = 0;
        bool found = false;
        while (*c.s.iq_size > 0) {
            const uint64_t cand = mkwp_ref_iq_front(c);
            const int cslot = mkwp_ref_find(c, cand);
            if (cslot < 0 || c.s.t_status[cslot] != MKWP_TS_QUEUED) {
                mkwp_ref_iq_pop(c);
                continue;
            }
            mkwp_ref_iq_pop(c);
            slot = cslot; tid = cand; found = true;
            break;
        }
        if (!found) {
            c.s.counts[MKWP_C_LOADER_NO_TILE] += 1;
            mkwp_ref_emit(c, MKWP_EV_LOADER_NO_TILE, (uint32_t)loader_id, 0,
                          UINT32_MAX, UINT32_MAX, UINT64_MAX, 0);
            return;
        }
        c.s.t_assigned[slot] = (uint32_t)b;
        c.s.t_status[slot] = MKWP_TS_LOADING;
        c.s.b_tile[b] = tid;
        c.s.b_owner[b] = MKWP_ROLE_LOADER;
        const uint32_t lbar = c.s.b_lbar[b];
        c.s.br_phase[lbar] += 1;
        c.s.br_expected[lbar] = 1;
        c.s.br_arrived[lbar] = 0;
        c.s.br_done[lbar] = 0;
        c.s.br_waitmask[lbar] = 0;
        const uint64_t pseq = c.s.scal[SC_PSEQ]++;
        c.s.b_lastphase[b] = pseq;
        c.s.b_state[b] = MKWP_BS_LOAD_INFLIGHT;
        // async
        const int ai = *c.s.as_count;
        c.s.as_kind[ai] = MKWP_AK_LOAD_COMPLETE;
        c.s.as_due[ai] = c.s.scal[SC_CLOCK] + c.s.t_load[slot];
        c.s.as_seq[ai] = c.s.scal[SC_ASEQ]++;
        c.s.as_tile[ai] = tid;
        c.s.as_buf[ai] = (uint32_t)b;
        c.s.as_bar[ai] = lbar;
        c.s.as_phase[ai] = c.s.br_phase[lbar];
        *c.s.as_count = ai + 1;
        c.s.counts[MKWP_C_LOAD_ISSUE] += 1;
        mkwp_ref_emit(c, MKWP_EV_LOAD_ISSUE, (uint32_t)loader_id, tid, (uint32_t)b,
                      lbar, c.s.br_phase[lbar], pseq);
    }
}

__device__ void mkwp_ref_op_compute(MkwpRefCtx& c, int compute_id, uint64_t limit) {
    if (compute_id < 0 || compute_id >= c.compute_warps || limit == 0) {
        mkwp_ref_emit_invalid(c);
        return;
    }
    for (uint64_t i = 0; i < limit; ++i) {
        const uint32_t b = mkwp_ref_rq_min(c, c.s.cq_buf, *c.s.cq_count);
        if (b == UINT32_MAX) {
            c.s.counts[MKWP_C_COMPUTE_NO_READY] += 1;
            mkwp_ref_emit(c, MKWP_EV_COMPUTE_NO_READY, (uint32_t)compute_id, 0,
                          UINT32_MAX, UINT32_MAX, UINT64_MAX, 0);
            return;
        }
        const uint32_t lbar = c.s.b_lbar[b];
        if (c.s.br_done[lbar] != 1) {
            c.s.br_waitmask[lbar] |= (1ULL << MKWP_ROLE_COMPUTE);
            c.s.counts[MKWP_C_MBARRIER_WAIT_LOAD] += 1;
            mkwp_ref_emit(c, MKWP_EV_MBARRIER_WAIT_LOAD, (uint32_t)compute_id,
                          c.s.b_tile[b], b, lbar, c.s.br_phase[lbar], 0);
            return;
        }
        mkwp_ref_rq_erase(c, c.s.cq_buf, c.s.cq_count, b);
        const int slot = mkwp_ref_find(c, c.s.b_tile[b]);
        c.s.b_state[b] = MKWP_BS_COMPUTE_INFLIGHT;
        c.s.b_owner[b] = MKWP_ROLE_COMPUTE;
        c.s.t_status[slot] = MKWP_TS_COMPUTING;
        const uint32_t cbar = c.s.b_cbar[b];
        c.s.br_phase[cbar] += 1;
        c.s.br_expected[cbar] = 1;
        c.s.br_arrived[cbar] = 0;
        c.s.br_done[cbar] = 0;
        c.s.br_waitmask[cbar] = 0;
        const uint64_t pseq = c.s.scal[SC_PSEQ]++;
        c.s.b_lastphase[b] = pseq;
        const int ai = *c.s.as_count;
        c.s.as_kind[ai] = MKWP_AK_COMPUTE_COMPLETE;
        c.s.as_due[ai] = c.s.scal[SC_CLOCK] + c.s.t_compute[slot];
        c.s.as_seq[ai] = c.s.scal[SC_ASEQ]++;
        c.s.as_tile[ai] = c.s.b_tile[b];
        c.s.as_buf[ai] = b;
        c.s.as_bar[ai] = cbar;
        c.s.as_phase[ai] = c.s.br_phase[cbar];
        *c.s.as_count = ai + 1;
        c.s.counts[MKWP_C_COMPUTE_ISSUE] += 1;
        mkwp_ref_emit(c, MKWP_EV_COMPUTE_ISSUE, (uint32_t)compute_id, c.s.b_tile[b],
                      b, cbar, c.s.br_phase[cbar], pseq);
    }
}

__device__ void mkwp_ref_op_storer(MkwpRefCtx& c, int storer_id, uint64_t limit) {
    if (storer_id < 0 || storer_id >= c.storer_warps || limit == 0) {
        mkwp_ref_emit_invalid(c);
        return;
    }
    for (uint64_t i = 0; i < limit; ++i) {
        const uint32_t b = mkwp_ref_rq_min(c, c.s.sq_buf, *c.s.sq_count);
        if (b == UINT32_MAX) {
            c.s.counts[MKWP_C_STORER_NO_READY] += 1;
            mkwp_ref_emit(c, MKWP_EV_STORER_NO_READY, (uint32_t)storer_id, 0,
                          UINT32_MAX, UINT32_MAX, UINT64_MAX, 0);
            return;
        }
        const uint32_t cbar = c.s.b_cbar[b];
        if (c.s.br_done[cbar] != 1) {
            c.s.br_waitmask[cbar] |= (1ULL << MKWP_ROLE_STORER);
            c.s.counts[MKWP_C_MBARRIER_WAIT_COMPUTE] += 1;
            mkwp_ref_emit(c, MKWP_EV_MBARRIER_WAIT_COMPUTE, (uint32_t)storer_id,
                          c.s.b_tile[b], b, cbar, c.s.br_phase[cbar], 0);
            return;
        }
        mkwp_ref_rq_erase(c, c.s.sq_buf, c.s.sq_count, b);
        const int slot = mkwp_ref_find(c, c.s.b_tile[b]);
        c.s.b_state[b] = MKWP_BS_STORE_INFLIGHT;
        c.s.b_owner[b] = MKWP_ROLE_STORER;
        c.s.t_status[slot] = MKWP_TS_STORING;
        const uint32_t sbar = c.s.b_sbar[b];
        c.s.br_phase[sbar] += 1;
        c.s.br_expected[sbar] = 1;
        c.s.br_arrived[sbar] = 0;
        c.s.br_done[sbar] = 0;
        c.s.br_waitmask[sbar] = 0;
        const uint64_t pseq = c.s.scal[SC_PSEQ]++;
        c.s.b_lastphase[b] = pseq;
        const int ai = *c.s.as_count;
        c.s.as_kind[ai] = MKWP_AK_STORE_COMPLETE;
        c.s.as_due[ai] = c.s.scal[SC_CLOCK] + c.s.t_store[slot];
        c.s.as_seq[ai] = c.s.scal[SC_ASEQ]++;
        c.s.as_tile[ai] = c.s.b_tile[b];
        c.s.as_buf[ai] = b;
        c.s.as_bar[ai] = sbar;
        c.s.as_phase[ai] = c.s.br_phase[sbar];
        *c.s.as_count = ai + 1;
        c.s.counts[MKWP_C_STORE_ISSUE] += 1;
        mkwp_ref_emit(c, MKWP_EV_STORE_ISSUE, (uint32_t)storer_id, c.s.b_tile[b],
                      b, sbar, c.s.br_phase[sbar], pseq);
    }
}

__device__ void mkwp_ref_process_async(MkwpRefCtx& c, int ai) {
    const int32_t kind = c.s.as_kind[ai];
    const uint64_t tile_id = c.s.as_tile[ai];
    const uint32_t buf = c.s.as_buf[ai];
    const uint32_t bar = c.s.as_bar[ai];
    const uint64_t phase = c.s.as_phase[ai];
    const int slot = mkwp_ref_find(c, tile_id);
    const bool stale =
        (slot < 0) || mkwp_ref_terminal(c.s.t_status[slot]) ||
        (c.s.b_tile[buf] != tile_id) ||
        (c.s.br_phase[bar] != phase);
    if (stale) {
        c.s.counts[MKWP_C_ASYNC_STALE_DROP] += 1;
        mkwp_ref_emit(c, MKWP_EV_ASYNC_STALE_DROP, UINT32_MAX, tile_id, buf, bar,
                      phase, (uint64_t)kind);
        return;
    }
    c.s.br_arrived[bar] += 1;
    c.s.br_lastarrive[bar] = c.s.scal[SC_ESEQ];  // seq of MBARRIER_ARRIVE below
    c.s.counts[MKWP_C_MBARRIER_ARRIVE] += 1;
    mkwp_ref_emit(c, MKWP_EV_MBARRIER_ARRIVE, UINT32_MAX, tile_id, buf, bar, phase,
                  (uint64_t)c.s.br_arrived[bar]);
    if (c.s.br_arrived[bar] == c.s.br_expected[bar] && c.s.br_done[bar] == 0) {
        c.s.br_done[bar] = 1;
        c.s.counts[MKWP_C_MBARRIER_COMPLETE] += 1;
        mkwp_ref_emit(c, MKWP_EV_MBARRIER_COMPLETE, UINT32_MAX, tile_id, buf, bar,
                      phase, 0);
    }
    if (kind == MKWP_AK_LOAD_COMPLETE) {
        c.s.b_state[buf] = MKWP_BS_LOAD_READY;
        c.s.b_owner[buf] = 255;
        c.s.t_status[slot] = MKWP_TS_READY_COMPUTE;
        mkwp_ref_rq_push(c, c.s.cq_buf, c.s.cq_count, buf);
        c.s.counts[MKWP_C_LOAD_COMPLETE] += 1;
        mkwp_ref_emit(c, MKWP_EV_LOAD_COMPLETE, UINT32_MAX, tile_id, buf, bar, phase, 0);
    } else if (kind == MKWP_AK_COMPUTE_COMPLETE) {
        const uint64_t rh = mkwp_ref_result_hash(tile_id, c.s.t_seq[slot],
                                                 c.s.t_seed[slot], buf,
                                                 c.s.scal[SC_CLOCK]);
        c.s.t_result[slot] = rh;
        c.s.b_state[buf] = MKWP_BS_COMPUTE_READY;
        c.s.b_owner[buf] = 255;
        c.s.t_status[slot] = MKWP_TS_READY_STORE;
        mkwp_ref_rq_push(c, c.s.sq_buf, c.s.sq_count, buf);
        c.s.counts[MKWP_C_COMPUTE_COMPLETE] += 1;
        mkwp_ref_emit(c, MKWP_EV_COMPUTE_COMPLETE, UINT32_MAX, tile_id, buf, bar, phase, rh);
    } else {  // STORE_COMPLETE
        c.s.t_status[slot] = MKWP_TS_DONE;
        c.s.t_assigned[slot] = UINT32_MAX;
        c.s.b_state[buf] = MKWP_BS_EMPTY;
        c.s.b_tile[buf] = 0;
        c.s.b_owner[buf] = 255;
        c.s.counts[MKWP_C_STORE_COMPLETE] += 1;
        mkwp_ref_emit(c, MKWP_EV_STORE_COMPLETE, UINT32_MAX, tile_id, buf, bar, phase, 0);
        c.s.counts[MKWP_C_TILE_DONE] += 1;
        mkwp_ref_emit(c, MKWP_EV_TILE_DONE, UINT32_MAX, tile_id, buf, UINT32_MAX,
                      UINT64_MAX, 0);
    }
}

__device__ void mkwp_ref_op_advance(MkwpRefCtx& c, uint64_t delta, uint64_t max_async) {
    c.s.scal[SC_CLOCK] += delta;
    const uint64_t clk = c.s.scal[SC_CLOCK];
    uint64_t processed = 0;
    while (processed < max_async) {
        // find min (due_clock, async_seq) among due events.
        int best = -1;
        uint64_t bd = 0, bs = 0;
        const int n = *c.s.as_count;
        for (int k = 0; k < n; ++k) {
            if (c.s.as_due[k] > clk) continue;
            const uint64_t d = c.s.as_due[k];
            const uint64_t sq = c.s.as_seq[k];
            if (best < 0 || d < bd || (d == bd && sq < bs)) {
                best = k; bd = d; bs = sq;
            }
        }
        if (best < 0) break;
        mkwp_ref_process_async(c, best);
        // compact out index best.
        const int n2 = *c.s.as_count;
        for (int k = best; k < n2 - 1; ++k) {
            c.s.as_kind[k] = c.s.as_kind[k + 1];
            c.s.as_due[k] = c.s.as_due[k + 1];
            c.s.as_seq[k] = c.s.as_seq[k + 1];
            c.s.as_tile[k] = c.s.as_tile[k + 1];
            c.s.as_buf[k] = c.s.as_buf[k + 1];
            c.s.as_bar[k] = c.s.as_bar[k + 1];
            c.s.as_phase[k] = c.s.as_phase[k + 1];
        }
        *c.s.as_count = n2 - 1;
        ++processed;
    }
}

__device__ void mkwp_ref_op_cancel(MkwpRefCtx& c, uint64_t tile_id) {
    const int slot = mkwp_ref_find(c, tile_id);
    if (slot < 0 || mkwp_ref_terminal(c.s.t_status[slot])) {
        mkwp_ref_emit_invalid(c);
        return;
    }
    if (c.s.t_status[slot] == MKWP_TS_QUEUED) {
        c.s.t_status[slot] = MKWP_TS_CANCELLED;
        c.s.counts[MKWP_C_TILE_CANCEL] += 1;
        mkwp_ref_emit(c, MKWP_EV_TILE_CANCEL, UINT32_MAX, tile_id, UINT32_MAX,
                      UINT32_MAX, UINT64_MAX, 0);
        return;
    }
    const uint32_t b = c.s.t_assigned[slot];
    c.s.t_status[slot] = MKWP_TS_CANCELLED;
    c.s.t_assigned[slot] = UINT32_MAX;
    mkwp_ref_rq_erase(c, c.s.cq_buf, c.s.cq_count, b);
    mkwp_ref_rq_erase(c, c.s.sq_buf, c.s.sq_count, b);
    c.s.b_state[b] = MKWP_BS_EMPTY;
    c.s.b_tile[b] = 0;
    c.s.b_owner[b] = 255;
    const uint32_t bars[3] = { c.s.b_lbar[b], c.s.b_cbar[b], c.s.b_sbar[b] };
    for (int k = 0; k < 3; ++k) {
        const uint32_t bid = bars[k];
        c.s.br_phase[bid] += 1;
        c.s.br_arrived[bid] = 0;
        c.s.br_done[bid] = 0;
        c.s.br_waitmask[bid] = 0;
    }
    c.s.counts[MKWP_C_BUFFER_CANCEL_RELEASE] += 1;
    mkwp_ref_emit(c, MKWP_EV_BUFFER_CANCEL_RELEASE, UINT32_MAX, tile_id, b,
                  UINT32_MAX, UINT64_MAX, 0);
}

__device__ void mkwp_ref_op_reset_barrier(MkwpRefCtx& c, int32_t barrier_id) {
    if (barrier_id < 0 || barrier_id >= c.BR) {
        mkwp_ref_emit_invalid(c);
        return;
    }
    const uint32_t owning_buf = (uint32_t)barrier_id / MKWP_BARRIERS_PER_BUFFER;
    if (owning_buf < (uint32_t)c.B && c.s.b_state[owning_buf] != MKWP_BS_EMPTY) {
        mkwp_ref_emit_invalid(c);
        return;
    }
    c.s.br_phase[barrier_id] += 1;
    c.s.br_expected[barrier_id] = 0;
    c.s.br_arrived[barrier_id] = 0;
    c.s.br_done[barrier_id] = 0;
    c.s.br_waitmask[barrier_id] = 0;
    c.s.counts[MKWP_C_BARRIER_RESET] += 1;
    mkwp_ref_emit(c, MKWP_EV_BARRIER_RESET, UINT32_MAX, 0, UINT32_MAX,
                  (uint32_t)barrier_id, c.s.br_phase[barrier_id], 0);
}

// ------------------------------------------------------------ snapshot hashes
__device__ uint64_t mkwp_ref_buffer_hash(MkwpRefCtx& c) {
    uint64_t h = MKWP_FNV_OFFSET;
    for (int b = 0; b < c.B; ++b) {
        uint32_t bu = (uint32_t)b; mkwp_ref_fnv_bytes(&h, &bu, 4);
        uint8_t st = (uint8_t)c.s.b_state[b]; mkwp_ref_fnv_bytes(&h, &st, 1);
        mkwp_ref_fnv_bytes(&h, &c.s.b_tile[b], 8);
        mkwp_ref_fnv_bytes(&h, &c.s.b_lbar[b], 4);
        mkwp_ref_fnv_bytes(&h, &c.s.b_cbar[b], 4);
        mkwp_ref_fnv_bytes(&h, &c.s.b_sbar[b], 4);
        mkwp_ref_fnv_bytes(&h, &c.s.b_owner[b], 1);
        mkwp_ref_fnv_bytes(&h, &c.s.b_lastphase[b], 8);
    }
    return h;
}
__device__ uint64_t mkwp_ref_barrier_hash(MkwpRefCtx& c) {
    uint64_t h = MKWP_FNV_OFFSET;
    for (int b = 0; b < c.BR; ++b) {
        uint32_t bu = (uint32_t)b; mkwp_ref_fnv_bytes(&h, &bu, 4);
        mkwp_ref_fnv_bytes(&h, &c.s.br_phase[b], 8);
        mkwp_ref_fnv_bytes(&h, &c.s.br_expected[b], 4);
        mkwp_ref_fnv_bytes(&h, &c.s.br_arrived[b], 4);
        mkwp_ref_fnv_bytes(&h, &c.s.br_done[b], 1);
        mkwp_ref_fnv_bytes(&h, &c.s.br_waitmask[b], 8);
        mkwp_ref_fnv_bytes(&h, &c.s.br_lastarrive[b], 8);
    }
    return h;
}
// tile hash over nonterminal tiles by ascending tile id (selection scan).
__device__ uint64_t mkwp_ref_tile_hash(MkwpRefCtx& c) {
    uint64_t h = MKWP_FNV_OFFSET;
    bool have_prev = false;
    uint64_t prev = 0;
    for (;;) {
        int best = -1;
        uint64_t bid = 0;
        for (int i = 0; i < c.cap; ++i) {
            if (!c.s.t_used[i]) continue;
            if (mkwp_ref_terminal(c.s.t_status[i])) continue;
            const uint64_t id = c.s.t_id[i];
            if (have_prev && id <= prev) continue;
            if (best < 0 || id < bid) { best = i; bid = id; }
        }
        if (best < 0) break;
        mkwp_ref_fnv_bytes(&h, &c.s.t_id[best], 8);
        mkwp_ref_fnv_bytes(&h, &c.s.t_seq[best], 8);
        mkwp_ref_fnv_bytes(&h, &c.s.t_load[best], 8);
        mkwp_ref_fnv_bytes(&h, &c.s.t_compute[best], 8);
        mkwp_ref_fnv_bytes(&h, &c.s.t_store[best], 8);
        uint8_t st = (uint8_t)c.s.t_status[best]; mkwp_ref_fnv_bytes(&h, &st, 1);
        mkwp_ref_fnv_bytes(&h, &c.s.t_assigned[best], 4);
        mkwp_ref_fnv_bytes(&h, &c.s.t_result[best], 8);
        prev = bid; have_prev = true;
    }
    return h;
}
// Byte-identical to mkwp_ref_tile_hash but cooperative: the whole block builds
// the ascending-id order (rank scatter into c.s.tile_order) with fully parallel,
// latency-hidden scans, then thread 0 folds FNV over that order. Live nonterminal
// tile ids form a strict total order (unique), so ranks 0..live-1 are dense and
// the fold sequence is exactly the oracle's ascending-id selection scan.
// Must be called by ALL threads of the block. Return value valid on thread 0.
__device__ uint64_t mkwp_ref_tile_hash_block(MkwpRefCtx& c) {
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    uint32_t* order = c.s.tile_order;
    for (int i = t; i < c.cap; i += nt) order[i] = UINT32_MAX;
    __syncthreads();
    for (int i = t; i < c.cap; i += nt) {
        if (!c.s.t_used[i]) continue;
        if (mkwp_ref_terminal(c.s.t_status[i])) continue;
        const uint64_t id = c.s.t_id[i];
        int rank = 0;
        for (int j = 0; j < c.cap; ++j) {
            if (!c.s.t_used[j]) continue;
            if (mkwp_ref_terminal(c.s.t_status[j])) continue;
            if (c.s.t_id[j] < id) ++rank;
        }
        order[rank] = (uint32_t)i;
    }
    __syncthreads();
    uint64_t h = MKWP_FNV_OFFSET;
    if (t == 0) {
        for (int k = 0; k < c.cap; ++k) {
            const uint32_t slot = order[k];
            if (slot == UINT32_MAX) break;  // ranks are dense: first gap == live
            mkwp_ref_fnv_bytes(&h, &c.s.t_id[slot], 8);
            mkwp_ref_fnv_bytes(&h, &c.s.t_seq[slot], 8);
            mkwp_ref_fnv_bytes(&h, &c.s.t_load[slot], 8);
            mkwp_ref_fnv_bytes(&h, &c.s.t_compute[slot], 8);
            mkwp_ref_fnv_bytes(&h, &c.s.t_store[slot], 8);
            uint8_t st = (uint8_t)c.s.t_status[slot]; mkwp_ref_fnv_bytes(&h, &st, 1);
            mkwp_ref_fnv_bytes(&h, &c.s.t_assigned[slot], 4);
            mkwp_ref_fnv_bytes(&h, &c.s.t_result[slot], 8);
        }
    }
    return h;
}

// async hash over pending in creation (flat) order.
__device__ uint64_t mkwp_ref_async_hash(MkwpRefCtx& c) {
    uint64_t h = MKWP_FNV_OFFSET;
    const int n = *c.s.as_count;
    for (int i = 0; i < n; ++i) {
        uint8_t k = (uint8_t)c.s.as_kind[i]; mkwp_ref_fnv_bytes(&h, &k, 1);
        mkwp_ref_fnv_bytes(&h, &c.s.as_due[i], 8);
        mkwp_ref_fnv_bytes(&h, &c.s.as_seq[i], 8);
        mkwp_ref_fnv_bytes(&h, &c.s.as_tile[i], 8);
        mkwp_ref_fnv_bytes(&h, &c.s.as_buf[i], 4);
        mkwp_ref_fnv_bytes(&h, &c.s.as_bar[i], 4);
        mkwp_ref_fnv_bytes(&h, &c.s.as_phase[i], 8);
    }
    return h;
}

// ------------------------------------------------------------ kernels
__global__ void mkwp_ref_step_kernel(
    mkwp_ref::State s, int B, int BR, int cap, int P, int Q,
    int loader_warps, int compute_warps, int storer_warps, int max_tiles,
    int op_kind, int a_role_id, int a_limit, int a_barrier,
    uint64_t a_tile, uint64_t a_load, uint64_t a_compute, uint64_t a_store,
    uint64_t a_seed, uint64_t a_delta,
    int64_t* out_counts, int32_t* out_opidx, uint64_t* out_clock, uint64_t* out_eseq,
    uint64_t* out_pipe, uint64_t* out_buf, uint64_t* out_bar, uint64_t* out_tile,
    uint64_t* out_async, uint64_t* out_state) {
    if (blockIdx.x != 0) return;

    MkwpRefCtx c;
    c.s = s;
    c.B = B; c.BR = BR; c.cap = cap; c.P = P; c.Q = Q;
    c.loader_warps = loader_warps; c.compute_warps = compute_warps;
    c.storer_warps = storer_warps; c.max_tiles = max_tiles;

    // Serial state mutation: single-threaded, identical to the oracle. All other
    // threads wait; they then join the cooperative tile-hash below.
    if (threadIdx.x == 0) {
        const uint64_t limit_u = (uint64_t)(uint32_t)a_limit;
        switch (op_kind) {
            case MKWP_OP_ENQUEUE_TILE: mkwp_ref_op_enqueue(c, a_tile, a_load, a_compute, a_store, a_seed); break;
            case MKWP_OP_LOADER_STEP:  mkwp_ref_op_loader(c, a_role_id, limit_u); break;
            case MKWP_OP_COMPUTE_STEP: mkwp_ref_op_compute(c, a_role_id, limit_u); break;
            case MKWP_OP_STORER_STEP:  mkwp_ref_op_storer(c, a_role_id, limit_u); break;
            case MKWP_OP_ADVANCE:      mkwp_ref_op_advance(c, a_delta, limit_u); break;
            case MKWP_OP_CANCEL_TILE:  mkwp_ref_op_cancel(c, a_tile); break;
            case MKWP_OP_RESET_BARRIER: mkwp_ref_op_reset_barrier(c, a_barrier); break;
            default: break;
        }
    }
    __syncthreads();  // publish state mutation to the whole block

    // Cooperative tile hash (all threads); byte-identical to the serial version.
    const uint64_t tileh = mkwp_ref_tile_hash_block(c);

    if (threadIdx.x != 0) return;

    const int32_t this_op = (int32_t)c.s.scal[SC_OPIDX];
    c.s.scal[SC_OPIDX] += 1;

    const uint64_t bufh = mkwp_ref_buffer_hash(c);
    const uint64_t barh = mkwp_ref_barrier_hash(c);
    const uint64_t asyh = mkwp_ref_async_hash(c);

    for (int i = 0; i < MKWP_COUNT_N; ++i) out_counts[i] = c.s.counts[i];
    *out_opidx = this_op;
    *out_clock = c.s.scal[SC_CLOCK];
    *out_eseq = c.s.scal[SC_ESEQ];
    *out_pipe = c.s.scal[SC_PIPE];
    *out_buf = bufh;
    *out_bar = barh;
    *out_tile = tileh;
    *out_async = asyh;

    uint64_t mh = MKWP_FNV_OFFSET;
    mkwp_ref_fnv_bytes(&mh, &c.s.scal[SC_CLOCK], 8);
    mkwp_ref_fnv_bytes(&mh, &c.s.scal[SC_ESEQ], 8);
    uint32_t opu = (uint32_t)c.s.scal[SC_OPIDX];
    mkwp_ref_fnv_bytes(&mh, &opu, 4);
    mkwp_ref_fnv_bytes(&mh, &c.s.scal[SC_TSEQ], 8);
    mkwp_ref_fnv_bytes(&mh, &c.s.scal[SC_PSEQ], 8);
    mkwp_ref_fnv_bytes(&mh, &c.s.scal[SC_ASEQ], 8);
    mkwp_ref_fnv_bytes(&mh, &c.s.scal[SC_PIPE], 8);
    mkwp_ref_fnv_bytes(&mh, &bufh, 8);
    mkwp_ref_fnv_bytes(&mh, &barh, 8);
    mkwp_ref_fnv_bytes(&mh, &tileh, 8);
    mkwp_ref_fnv_bytes(&mh, &asyh, 8);
    for (int i = 0; i < MKWP_COUNT_N; ++i) {
        uint64_t cv = (uint64_t)c.s.counts[i];
        mkwp_ref_fnv_bytes(&mh, &cv, 8);
    }
    *out_state = mh;
}

__global__ void mkwp_ref_reset_kernel(mkwp_ref::State s, int B, int BR, int cap) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    s.scal[SC_CLOCK] = 0;
    s.scal[SC_ESEQ] = 0;
    s.scal[SC_OPIDX] = 0;
    s.scal[SC_TSEQ] = 1;
    s.scal[SC_PSEQ] = 1;
    s.scal[SC_ASEQ] = 1;
    s.scal[SC_PIPE] = MKWP_FNV_OFFSET;
    for (int i = 0; i < cap; ++i) s.t_used[i] = 0;
    *s.iq_head = 0;
    *s.iq_size = 0;
    for (int b = 0; b < B; ++b) {
        s.b_state[b] = MKWP_BS_EMPTY;
        s.b_tile[b] = 0;
        s.b_lbar[b] = (uint32_t)(b * MKWP_BARRIERS_PER_BUFFER + MKWP_BAR_LOAD);
        s.b_cbar[b] = (uint32_t)(b * MKWP_BARRIERS_PER_BUFFER + MKWP_BAR_COMPUTE);
        s.b_sbar[b] = (uint32_t)(b * MKWP_BARRIERS_PER_BUFFER + MKWP_BAR_STORE);
        s.b_owner[b] = 255;
        s.b_lastphase[b] = 0;
    }
    for (int b = 0; b < BR; ++b) {
        s.br_phase[b] = 0;
        s.br_expected[b] = 0;
        s.br_arrived[b] = 0;
        s.br_done[b] = 0;
        s.br_waitmask[b] = 0;
        s.br_lastarrive[b] = 0;
    }
    *s.as_count = 0;
    *s.cq_count = 0;
    *s.sq_count = 0;
    for (int i = 0; i < MKWP_COUNT_N; ++i) s.counts[i] = 0;
}

// ------------------------------------------------------------ host ABI
extern "C" size_t solution_workspace_bytes(const MkwpProblemSpec* spec) {
    if (!mkwp_validate_problem_spec(spec)) return 0;
    return 256;
}

static cudaError_t mkwp_ref_do_reset(mkwp_ref::State* st, cudaStream_t stream) {
    const MkwpProblemSpec& sp = st->spec;
    mkwp_ref_reset_kernel<<<1, 1, 0, stream>>>(*st, sp.buffer_count, sp.barrier_count, sp.max_tiles);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_init(const MkwpProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!mkwp_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;

    mkwp_ref::State* st = (mkwp_ref::State*)malloc(sizeof(mkwp_ref::State));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(mkwp_ref::State));
    memcpy(&st->spec, spec, sizeof(MkwpProblemSpec));

    const int B = spec->buffer_count;
    const int BR = spec->barrier_count;
    const int cap = spec->max_tiles;
    const int P = spec->max_pending_async;
    const int Q = spec->max_role_queue;

    cudaError_t err = cudaSuccess;
    #define MKWP_M(p, bytes) do { err = cudaMalloc((void**)&(p), (bytes)); if (err) goto fail; } while (0)
    MKWP_M(st->scal, sizeof(uint64_t) * 7);
    MKWP_M(st->t_used, sizeof(uint8_t) * cap);
    MKWP_M(st->t_id, sizeof(uint64_t) * cap);
    MKWP_M(st->t_seq, sizeof(uint64_t) * cap);
    MKWP_M(st->t_load, sizeof(uint64_t) * cap);
    MKWP_M(st->t_compute, sizeof(uint64_t) * cap);
    MKWP_M(st->t_store, sizeof(uint64_t) * cap);
    MKWP_M(st->t_seed, sizeof(uint64_t) * cap);
    MKWP_M(st->t_status, sizeof(int32_t) * cap);
    MKWP_M(st->t_assigned, sizeof(uint32_t) * cap);
    MKWP_M(st->t_result, sizeof(uint64_t) * cap);
    MKWP_M(st->iq_buf, sizeof(uint64_t) * cap);
    MKWP_M(st->iq_head, sizeof(int32_t) * 1);
    MKWP_M(st->iq_size, sizeof(int32_t) * 1);
    MKWP_M(st->b_state, sizeof(int32_t) * B);
    MKWP_M(st->b_tile, sizeof(uint64_t) * B);
    MKWP_M(st->b_lbar, sizeof(uint32_t) * B);
    MKWP_M(st->b_cbar, sizeof(uint32_t) * B);
    MKWP_M(st->b_sbar, sizeof(uint32_t) * B);
    MKWP_M(st->b_owner, sizeof(uint8_t) * B);
    MKWP_M(st->b_lastphase, sizeof(uint64_t) * B);
    MKWP_M(st->br_phase, sizeof(uint64_t) * BR);
    MKWP_M(st->br_expected, sizeof(uint32_t) * BR);
    MKWP_M(st->br_arrived, sizeof(uint32_t) * BR);
    MKWP_M(st->br_done, sizeof(uint8_t) * BR);
    MKWP_M(st->br_waitmask, sizeof(uint64_t) * BR);
    MKWP_M(st->br_lastarrive, sizeof(uint64_t) * BR);
    MKWP_M(st->as_kind, sizeof(int32_t) * P);
    MKWP_M(st->as_due, sizeof(uint64_t) * P);
    MKWP_M(st->as_seq, sizeof(uint64_t) * P);
    MKWP_M(st->as_tile, sizeof(uint64_t) * P);
    MKWP_M(st->as_buf, sizeof(uint32_t) * P);
    MKWP_M(st->as_bar, sizeof(uint32_t) * P);
    MKWP_M(st->as_phase, sizeof(uint64_t) * P);
    MKWP_M(st->as_count, sizeof(int32_t) * 1);
    MKWP_M(st->cq_buf, sizeof(uint32_t) * Q);
    MKWP_M(st->cq_count, sizeof(int32_t) * 1);
    MKWP_M(st->sq_buf, sizeof(uint32_t) * Q);
    MKWP_M(st->sq_count, sizeof(int32_t) * 1);
    MKWP_M(st->counts, sizeof(int64_t) * MKWP_COUNT_N);
    MKWP_M(st->tile_order, sizeof(uint32_t) * cap);
    #undef MKWP_M

    err = mkwp_ref_do_reset(st, stream);
    if (err) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    solution_destroy(st);
    return err ? err : cudaErrorMemoryAllocation;
}

extern "C" cudaError_t solution_run(
    void* state, const MkwpRunSpec* run, const void* inputs_void, void* outputs_void,
    void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    (void)inputs_void; (void)workspace;
    if (!state || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 256) return cudaErrorInvalidValue;

    mkwp_ref::State* st = (mkwp_ref::State*)state;
    if (!mkwp_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    MkwpOutputs* out = (MkwpOutputs*)outputs_void;
    if (!out->counts || !out->op_index_out || !out->clock_out || !out->event_seq_out ||
        !out->pipe_event_hash || !out->buffer_hash || !out->barrier_hash ||
        !out->tile_hash || !out->async_hash || !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    const MkwpProblemSpec& sp = st->spec;
    mkwp_ref_step_kernel<<<1, MKWP_STEP_BLOCK, 0, stream>>>(
        *st, sp.buffer_count, sp.barrier_count, sp.max_tiles, sp.max_pending_async,
        sp.max_role_queue, sp.loader_warps, sp.compute_warps, sp.storer_warps, sp.max_tiles,
        run->op_kind, run->a_role_id, run->a_limit, run->a_barrier,
        run->a_tile, run->a_load_bytes, run->a_compute_iters, run->a_store_bytes,
        run->a_seed, run->a_delta,
        out->counts, out->op_index_out, out->clock_out, out->event_seq_out,
        out->pipe_event_hash, out->buffer_hash, out->barrier_hash, out->tile_hash,
        out->async_hash, out->state_checksum);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return mkwp_ref_do_reset((mkwp_ref::State*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    mkwp_ref::State* st = (mkwp_ref::State*)state;
    cudaFree(st->scal);
    cudaFree(st->t_used); cudaFree(st->t_id); cudaFree(st->t_seq); cudaFree(st->t_load);
    cudaFree(st->t_compute); cudaFree(st->t_store); cudaFree(st->t_seed);
    cudaFree(st->t_status); cudaFree(st->t_assigned); cudaFree(st->t_result);
    cudaFree(st->iq_buf); cudaFree(st->iq_head); cudaFree(st->iq_size);
    cudaFree(st->b_state); cudaFree(st->b_tile); cudaFree(st->b_lbar); cudaFree(st->b_cbar);
    cudaFree(st->b_sbar); cudaFree(st->b_owner); cudaFree(st->b_lastphase);
    cudaFree(st->br_phase); cudaFree(st->br_expected); cudaFree(st->br_arrived);
    cudaFree(st->br_done); cudaFree(st->br_waitmask); cudaFree(st->br_lastarrive);
    cudaFree(st->as_kind); cudaFree(st->as_due); cudaFree(st->as_seq); cudaFree(st->as_tile);
    cudaFree(st->as_buf); cudaFree(st->as_bar); cudaFree(st->as_phase); cudaFree(st->as_count);
    cudaFree(st->cq_buf); cudaFree(st->cq_count); cudaFree(st->sq_buf); cudaFree(st->sq_count);
    cudaFree(st->counts);
    cudaFree(st->tile_order);
    free(st);
}
