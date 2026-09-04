// file: mk_multigpu_allreduce_oracle.hpp
//
// Host-side golden model for T68 (MK9). This is an INDEPENDENT third
// implementation: a plain-C++ event-driven simulation using std::vector
// containers. reference.cu and naive.cu share no algorithm code with this file.

#ifndef MK_MULTIGPU_ALLREDUCE_ORACLE_HPP_
#define MK_MULTIGPU_ALLREDUCE_ORACLE_HPP_

#include "mk_multigpu_allreduce_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <deque>
#include <sstream>
#include <string>
#include <vector>

struct MgaOracleResult {
    int64_t counters[MGA_COUNTER_COUNT] = {0};
    uint64_t remote_event_hash = 0;
    uint64_t collective_hash = 0;
    uint64_t signal_hash = 0;
    uint64_t credit_hash = 0;
    uint64_t pending_remote_hash = 0;
    uint64_t scheduler_hash = 0;
    uint64_t clock_out = 0;
    uint64_t event_seq_out = 0;
};

static inline uint64_t mga_o_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= 1099511628211ULL;
    return h;
}
static inline void mga_o_fnv(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = mga_o_fnv_byte(v, b[i]);
    *h = v;
}
// Hash of an ordered tuple of u64 values (used for the local-value seed).
static inline uint64_t mga_o_fnv_u64s(const uint64_t* vals, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) mga_o_fnv(&h, &vals[i], 8);
    return h;
}

// ---- persistent records ----

struct MgaOCell {
    int64_t local_value = 0;
    int64_t accum_value = 0;
    int64_t final_value = 0;
    uint8_t have_local = 0;
    uint8_t local_ready = 0;
    uint8_t reduced_owner_ready = 0;
    uint8_t final_ready = 0;
    uint64_t last_update_seq = 0;
};

struct MgaOColl {
    bool used = false;
    uint64_t coll_id = 0;
    uint64_t collective_seq = 0;
    uint8_t status = MGA_ST_ACTIVE;
    uint8_t phase = MGA_PH_LOCAL_REDUCE;
    int chunk_count = 0;
    int active_rank_count = 0;
    std::vector<int> active_ranks;       // sorted ascending
    std::vector<int> pos_of_rank;        // rank_count entries, -1 if not active
    bool rs_started = false;
    bool ag_started = false;
    int owners_reduced = 0;
    std::vector<MgaOCell> cells;         // rank_count * chunk_count
    std::vector<int64_t> credit;         // active_rank_count entries: edge pos->next
};

struct MgaOAction {
    uint64_t coll_id;
    int coll_slot;
    uint8_t action_kind;
    int chunk;
    int phase_index;
    int src;          // for recv actions
    int64_t value;    // for recv actions
    uint64_t phase_seq;
};

struct MgaORemote {
    uint64_t due_clock;
    uint64_t remote_seq;
    int src;
    int dst;
    uint64_t coll_id;
    int coll_slot;
    uint8_t phase_kind;   // MGA_PK_RS or MGA_PK_AG
    int phase_index;
    int chunk;
    int64_t value;
    int credit_return_src;  // reverse link source rank
    int credit_return_dst;  // reverse link dest rank
};

struct MgaORecvBuf {
    int64_t value = 0;
    int src = -1;
    uint64_t remote_seq = 0;
    uint8_t valid = 0;
};

struct MgaOracle {
    MgaProblemSpec spec{};
    int rank_count = 0, chunk_max = 0, max_colls = 0;
    int max_remote = 0, max_credit = 0, remote_latency = 0, max_sched = 0;

    uint64_t clock = 0;
    uint64_t event_seq = 0;
    uint64_t collective_seq_next = 1;
    uint64_t remote_seq_next = 1;
    uint64_t phase_seq_next = 1;

    std::vector<MgaOColl> colls;                       // max_colls
    std::vector<std::vector<MgaOAction>> sched;        // [rank] -> queue
    std::vector<MgaORemote> pending;                   // unsorted; sorted on read
    // signal_counter[rank][coll_slot][phase_kind(3)][phase_index][chunk]
    std::vector<uint64_t> signal;
    // receive buffers keyed (dst, coll_slot, chunk, phase_kind)
    std::vector<MgaORecvBuf> recvbuf;

    int64_t counters[MGA_COUNTER_COUNT];
    uint64_t run_event_hash;

    int phidx_bound = 0;   // = rank_count (max phase index span)

    int sig_index(int rank, int slot, int pk, int pidx, int chunk) const {
        // pk in 0..2
        return (((rank * max_colls + slot) * 3 + pk) * phidx_bound + pidx) * chunk_max + chunk;
    }
    int recv_index(int dst, int slot, int chunk, int pk) const {
        return ((dst * max_colls + slot) * chunk_max + chunk) * 3 + pk;
    }

    void init(const MgaProblemSpec& s) {
        spec = s;
        rank_count = s.rank_count;
        chunk_max = s.chunk_count_max;
        max_colls = s.max_collectives;
        max_remote = s.max_remote_events;
        max_credit = s.max_send_credits_per_link;
        remote_latency = s.remote_latency;
        max_sched = s.max_scheduler_queue_per_rank;
        phidx_bound = rank_count;
        reset();
    }

    void reset() {
        clock = 0; event_seq = 0;
        collective_seq_next = 1; remote_seq_next = 1; phase_seq_next = 1;
        colls.assign((size_t)max_colls, MgaOColl());
        sched.assign((size_t)rank_count, std::vector<MgaOAction>());
        pending.clear();
        signal.assign((size_t)rank_count * max_colls * 3 * phidx_bound * chunk_max, 0);
        recvbuf.assign((size_t)rank_count * max_colls * chunk_max * 3, MgaORecvBuf());
        for (int i = 0; i < MGA_COUNTER_COUNT; ++i) counters[i] = 0;
    }

    // ---- event emission ----
    void emit(uint8_t kind, uint32_t op_index, uint64_t coll_id_or_zero,
              uint32_t src_or, uint32_t dst_or, uint8_t pk_or,
              uint32_t pidx_or, uint32_t chunk_or, int64_t value_or) {
        uint64_t es = event_seq;
        uint64_t* h = &run_event_hash;
        mga_o_fnv(h, &kind, 1);
        mga_o_fnv(h, &es, 8);
        mga_o_fnv(h, &op_index, 4);
        uint64_t clk = clock; mga_o_fnv(h, &clk, 8);
        mga_o_fnv(h, &coll_id_or_zero, 8);
        mga_o_fnv(h, &src_or, 4);
        mga_o_fnv(h, &dst_or, 4);
        mga_o_fnv(h, &pk_or, 1);
        mga_o_fnv(h, &pidx_or, 4);
        mga_o_fnv(h, &chunk_or, 4);
        mga_o_fnv(h, &value_or, 8);
        event_seq = es + 1;
    }

    int find_coll(uint64_t id) const {
        for (int i = 0; i < max_colls; ++i)
            if (colls[(size_t)i].used && colls[(size_t)i].coll_id == id) return i;
        return -1;
    }

    int popcount_mask(uint32_t m) const {
        int c = 0;
        for (int i = 0; i < rank_count; ++i) if (m & (1u << i)) ++c;
        return c;
    }

    int next_rank(const MgaOColl& c, int rank) const {
        int pos = c.pos_of_rank[(size_t)rank];
        int np = (pos + 1) % c.active_rank_count;
        return c.active_ranks[(size_t)np];
    }

    void enqueue(int rank, const MgaOAction& a_in) {
        MgaOAction a = a_in;
        a.phase_seq = phase_seq_next++;
        if ((int)sched[(size_t)rank].size() >= max_sched) return;  // bounded; silent
        sched[(size_t)rank].push_back(a);
    }

    // ---- operations ----
    void op_begin(const MgaRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        uint64_t coll_id = (uint64_t)(uint32_t)run.a0;
        uint32_t mask = (uint32_t)run.a1;
        int cc = run.a2;
        uint64_t seed = (uint64_t)(uint32_t)run.a3;

        bool invalid = false;
        int existing = find_coll(coll_id);
        if (existing >= 0 && colls[(size_t)existing].status == MGA_ST_ACTIVE) invalid = true;
        int popc = popcount_mask(mask);
        if (popc < 2) invalid = true;
        if (cc < MGA_MIN_CHUNKS || cc > chunk_max) invalid = true;
        // table full: need a slot (reuse existing terminal id slot, else free).
        int slot = -1;
        if (!invalid) {
            if (existing >= 0) slot = existing;  // terminal id reuse in place
            else {
                for (int i = 0; i < max_colls; ++i)
                    if (!colls[(size_t)i].used) { slot = i; break; }
                if (slot < 0) {
                    for (int i = 0; i < max_colls; ++i)
                        if (colls[(size_t)i].status != MGA_ST_ACTIVE) { slot = i; break; }
                }
            }
            if (slot < 0) invalid = true;
        }
        if (invalid) {
            counters[18]++;
            emit(MGA_EV_INVALID, op_index, coll_id, UINT32_MAX, UINT32_MAX,
                 MGA_PK_NONE, UINT32_MAX, UINT32_MAX, INT64_MIN);
            return;
        }

        MgaOColl& c = colls[(size_t)slot];
        c = MgaOColl();
        c.used = true;
        c.coll_id = coll_id;
        c.collective_seq = collective_seq_next++;
        c.status = MGA_ST_ACTIVE;
        c.phase = MGA_PH_LOCAL_REDUCE;
        c.chunk_count = cc;
        c.active_ranks.clear();
        c.pos_of_rank.assign((size_t)rank_count, -1);
        for (int r = 0; r < rank_count; ++r)
            if (mask & (1u << r)) c.active_ranks.push_back(r);
        c.active_rank_count = (int)c.active_ranks.size();
        for (int i = 0; i < c.active_rank_count; ++i)
            c.pos_of_rank[(size_t)c.active_ranks[(size_t)i]] = i;
        c.cells.assign((size_t)rank_count * chunk_max, MgaOCell());
        c.credit.assign((size_t)c.active_rank_count, max_credit);
        c.rs_started = false;
        c.ag_started = false;
        c.owners_reduced = 0;

        for (int ai = 0; ai < c.active_rank_count; ++ai) {
            int r = c.active_ranks[(size_t)ai];
            for (int ch = 0; ch < cc; ++ch) {
                MgaOCell& cell = c.cells[(size_t)(r * chunk_max + ch)];
                uint64_t vals[4] = { seed, coll_id, (uint64_t)r, (uint64_t)ch };
                uint64_t hv = mga_o_fnv_u64s(vals, 4);
                cell.local_value = (int64_t)hv;
                cell.accum_value = cell.local_value;
                cell.final_value = 0;
                cell.have_local = 1;
                cell.local_ready = 0;
                cell.reduced_owner_ready = 0;
                cell.final_ready = 0;
                cell.last_update_seq = 0;
            }
        }
        // enqueue LOCAL_REDUCE for each active rank/chunk (rank asc, chunk asc)
        for (int ai = 0; ai < c.active_rank_count; ++ai) {
            int r = c.active_ranks[(size_t)ai];
            for (int ch = 0; ch < cc; ++ch) {
                MgaOAction a;
                a.coll_id = coll_id; a.coll_slot = slot;
                a.action_kind = MGA_ACT_LOCAL_REDUCE;
                a.chunk = ch; a.phase_index = 0; a.src = -1; a.value = 0;
                enqueue(r, a);
            }
        }
        counters[0]++;
        emit(MGA_EV_COLL_BEGIN, op_index, coll_id, UINT32_MAX, UINT32_MAX,
             MGA_PK_NONE, UINT32_MAX, UINT32_MAX, INT64_MIN);
    }

    int ring_slot(const MgaOColl& c, int rank, int p) const {
        int pos = c.pos_of_rank[(size_t)rank];
        int R = c.active_rank_count;
        return ((pos - p) % R + R) % R;
    }

    void start_reduce_scatter(MgaOColl& c, int slot, uint32_t op_index) {
        c.phase = MGA_PH_REDUCE_SCATTER;
        c.rs_started = true;
        counters[2]++;
        emit(MGA_EV_PHASE_ADVANCE_RS, op_index, c.coll_id, UINT32_MAX, UINT32_MAX,
             MGA_PK_RS, UINT32_MAX, UINT32_MAX, INT64_MIN);
        for (int ai = 0; ai < c.active_rank_count; ++ai) {
            int r = c.active_ranks[(size_t)ai];
            int s = ring_slot(c, r, 0);
            int ch = s % c.chunk_count;
            MgaOAction a;
            a.coll_id = c.coll_id; a.coll_slot = slot;
            a.action_kind = MGA_ACT_REDUCE_SCATTER_SEND;
            a.chunk = ch; a.phase_index = 0; a.src = -1; a.value = 0;
            enqueue(r, a);
        }
    }

    void start_allgather(MgaOColl& c, int slot, uint32_t op_index) {
        c.phase = MGA_PH_ALLGATHER;
        c.ag_started = true;
        counters[3]++;
        emit(MGA_EV_PHASE_ADVANCE_AG, op_index, c.coll_id, UINT32_MAX, UINT32_MAX,
             MGA_PK_AG, UINT32_MAX, UINT32_MAX, INT64_MIN);
        // each owner sends its owned chunk at allgather phase 0.
        for (int ai = 0; ai < c.active_rank_count; ++ai) {
            int r = c.active_ranks[(size_t)ai];
            for (int ch = 0; ch < c.chunk_count; ++ch) {
                MgaOCell& cell = c.cells[(size_t)(r * chunk_max + ch)];
                if (!cell.reduced_owner_ready) continue;
                // owner's final = its reduced accum.
                cell.final_value = cell.accum_value;
                cell.final_ready = 1;
                MgaOAction a;
                a.coll_id = c.coll_id; a.coll_slot = slot;
                a.action_kind = MGA_ACT_ALLGATHER_SEND;
                a.chunk = ch; a.phase_index = 0; a.src = -1; a.value = 0;
                enqueue(r, a);
            }
        }
    }

    bool all_local_ready(const MgaOColl& c) const {
        for (int ai = 0; ai < c.active_rank_count; ++ai) {
            int r = c.active_ranks[(size_t)ai];
            for (int ch = 0; ch < c.chunk_count; ++ch)
                if (!c.cells[(size_t)(r * chunk_max + ch)].local_ready) return false;
        }
        return true;
    }

    bool all_final_ready(const MgaOColl& c) const {
        for (int ai = 0; ai < c.active_rank_count; ++ai) {
            int r = c.active_ranks[(size_t)ai];
            for (int ch = 0; ch < c.chunk_count; ++ch)
                if (!c.cells[(size_t)(r * chunk_max + ch)].final_ready) return false;
        }
        return true;
    }

    void do_send(MgaOColl& c, int slot, int rank, const MgaOAction& a,
                 uint8_t phase_kind, uint32_t op_index) {
        int dst = next_rank(c, rank);
        int pos = c.pos_of_rank[(size_t)rank];
        if (c.credit[(size_t)pos] == 0) {
            counters[5]++;
            emit(MGA_EV_SEND_CREDIT_STALL, op_index, c.coll_id, (uint32_t)rank,
                 (uint32_t)dst, phase_kind, (uint32_t)a.phase_index,
                 (uint32_t)a.chunk, INT64_MIN);
            // re-append same action with new phase_seq.
            MgaOAction re = a;
            re.src = -1; re.value = 0;
            enqueue(rank, re);
            return;
        }
        c.credit[(size_t)pos] -= 1;
        int64_t val = c.cells[(size_t)(rank * chunk_max + a.chunk)].accum_value;
        if (max_remote > 0 && (int)pending.size() < max_remote) {
            MgaORemote ev;
            ev.due_clock = clock + (uint64_t)remote_latency;
            ev.remote_seq = remote_seq_next++;
            ev.src = rank; ev.dst = dst;
            ev.coll_id = c.coll_id; ev.coll_slot = slot;
            ev.phase_kind = phase_kind;
            ev.phase_index = a.phase_index;
            ev.chunk = a.chunk;
            ev.value = val;
            // reverse link to credit on arrival: edge dst->rank, sourced at dst.
            ev.credit_return_src = dst;
            ev.credit_return_dst = rank;
            pending.push_back(ev);
        } else {
            // queue full: still consumed credit and remote_seq advances only when
            // appended; to stay deterministic we DO advance remote_seq only when an
            // event is actually created. Roll back: do not advance if not appended.
            remote_seq_next--;  // undo (event not created)
        }
        counters[4]++;
        emit(MGA_EV_REMOTE_SEND, op_index, c.coll_id, (uint32_t)rank, (uint32_t)dst,
             phase_kind, (uint32_t)a.phase_index, (uint32_t)a.chunk, val);
    }

    void process_action(int rank, const MgaOAction& a, uint32_t op_index) {
        int slot = find_coll(a.coll_id);
        if (slot < 0 || colls[(size_t)slot].status != MGA_ST_ACTIVE) {
            counters[16]++;
            emit(MGA_EV_ACTION_STALE_DROP, op_index, a.coll_id, (uint32_t)rank,
                 UINT32_MAX, MGA_PK_NONE, (uint32_t)a.phase_index,
                 (uint32_t)a.chunk, INT64_MIN);
            return;
        }
        MgaOColl& c = colls[(size_t)slot];
        int R = c.active_rank_count;

        if (a.action_kind == MGA_ACT_LOCAL_REDUCE) {
            MgaOCell& cell = c.cells[(size_t)(rank * chunk_max + a.chunk)];
            if (cell.have_local) {
                cell.local_ready = 1;
                counters[1]++;
                emit(MGA_EV_LOCAL_READY, op_index, c.coll_id, (uint32_t)rank,
                     UINT32_MAX, MGA_PK_LOCAL, UINT32_MAX, (uint32_t)a.chunk, INT64_MIN);
            }
            if (!c.rs_started && all_local_ready(c)) {
                start_reduce_scatter(c, slot, op_index);
            }
        } else if (a.action_kind == MGA_ACT_REDUCE_SCATTER_SEND) {
            do_send(c, slot, rank, a, MGA_PK_RS, op_index);
        } else if (a.action_kind == MGA_ACT_ALLGATHER_SEND) {
            do_send(c, slot, rank, a, MGA_PK_AG, op_index);
        } else if (a.action_kind == MGA_ACT_REMOTE_RECV_REDUCE) {
            MgaOCell& cell = c.cells[(size_t)(rank * chunk_max + a.chunk)];
            cell.accum_value = (int64_t)((uint64_t)cell.accum_value + (uint64_t)a.value);
            cell.last_update_seq = event_seq;
            counters[10]++;
            emit(MGA_EV_REDUCE_APPLY, op_index, c.coll_id, (uint32_t)a.src,
                 (uint32_t)rank, MGA_PK_RS, (uint32_t)a.phase_index,
                 (uint32_t)a.chunk, cell.accum_value);
            int p = a.phase_index;
            if (p + 1 < R - 1) {
                int s = ring_slot(c, rank, p + 1);
                int ch = s % c.chunk_count;
                MgaOAction na;
                na.coll_id = c.coll_id; na.coll_slot = slot;
                na.action_kind = MGA_ACT_REDUCE_SCATTER_SEND;
                na.chunk = ch; na.phase_index = p + 1; na.src = -1; na.value = 0;
                enqueue(rank, na);
            } else {
                // last reduce phase: this rank/chunk becomes the owner.
                if (!cell.reduced_owner_ready) {
                    cell.reduced_owner_ready = 1;
                    c.owners_reduced++;
                    counters[11]++;
                    emit(MGA_EV_OWNER_REDUCED, op_index, c.coll_id, UINT32_MAX,
                         (uint32_t)rank, MGA_PK_RS, (uint32_t)a.phase_index,
                         (uint32_t)a.chunk, cell.accum_value);
                    if (!c.ag_started && c.owners_reduced >= R) {
                        start_allgather(c, slot, op_index);
                    }
                }
            }
        } else if (a.action_kind == MGA_ACT_REMOTE_RECV_GATHER) {
            MgaOCell& cell = c.cells[(size_t)(rank * chunk_max + a.chunk)];
            cell.final_value = a.value;
            cell.final_ready = 1;
            cell.last_update_seq = event_seq;
            counters[12]++;
            emit(MGA_EV_GATHER_APPLY, op_index, c.coll_id, (uint32_t)a.src,
                 (uint32_t)rank, MGA_PK_AG, (uint32_t)a.phase_index,
                 (uint32_t)a.chunk, cell.final_value);
            int p = a.phase_index;
            if (p + 1 < R - 1) {
                MgaOAction na;
                na.coll_id = c.coll_id; na.coll_slot = slot;
                na.action_kind = MGA_ACT_ALLGATHER_SEND;
                na.chunk = a.chunk; na.phase_index = p + 1; na.src = -1; na.value = 0;
                enqueue(rank, na);
            }
            if (c.status == MGA_ST_ACTIVE && all_final_ready(c)) {
                c.status = MGA_ST_DONE;
                c.phase = MGA_PH_DONE;
                counters[13]++;
                emit(MGA_EV_COLL_DONE, op_index, c.coll_id, UINT32_MAX, UINT32_MAX,
                     MGA_PK_NONE, UINT32_MAX, UINT32_MAX, INT64_MIN);
            }
        }
    }

    void op_rank_step(const MgaRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        int rank = run.a0;
        int action_limit = run.a1;
        if (rank < 0 || rank >= rank_count || action_limit == 0) {
            counters[18]++;
            emit(MGA_EV_INVALID, op_index, 0, (uint32_t)(rank >= 0 ? rank : -1),
                 UINT32_MAX, MGA_PK_NONE, UINT32_MAX, UINT32_MAX, INT64_MIN);
            return;
        }
        int processed = 0;
        while (processed < action_limit && !sched[(size_t)rank].empty()) {
            MgaOAction a = sched[(size_t)rank].front();
            sched[(size_t)rank].erase(sched[(size_t)rank].begin());
            process_action(rank, a, op_index);
            ++processed;
        }
    }

    void op_advance(const MgaRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        int delta = run.a0;
        int max_r = run.a1;
        if (delta < 0 || max_r < 0) {
            counters[18]++;
            emit(MGA_EV_INVALID, op_index, 0, UINT32_MAX, UINT32_MAX, MGA_PK_NONE,
                 UINT32_MAX, UINT32_MAX, INT64_MIN);
            return;
        }
        clock = clock + (uint64_t)delta;
        int processed = 0;
        while (processed < max_r) {
            // find canonical-min due event with due_clock <= clock.
            int best = -1;
            for (int i = 0; i < (int)pending.size(); ++i) {
                if (pending[(size_t)i].due_clock > clock) continue;
                if (best < 0) { best = i; continue; }
                const MgaORemote& r = pending[(size_t)i];
                const MgaORemote& b = pending[(size_t)best];
                if (r.due_clock != b.due_clock) { if (r.due_clock < b.due_clock) best = i; continue; }
                if (r.remote_seq < b.remote_seq) best = i;
            }
            if (best < 0) break;
            MgaORemote ev = pending[(size_t)best];
            pending.erase(pending.begin() + best);
            ++processed;

            int slot = find_coll(ev.coll_id);
            bool stale = (slot < 0) || colls[(size_t)slot].status != MGA_ST_ACTIVE;
            if (stale) {
                counters[7]++;
                emit(MGA_EV_REMOTE_STALE_DROP, op_index, ev.coll_id, (uint32_t)ev.src,
                     (uint32_t)ev.dst, ev.phase_kind, (uint32_t)ev.phase_index,
                     (uint32_t)ev.chunk, ev.value);
                continue;
            }
            MgaOColl& c = colls[(size_t)slot];
            // 1. write receive buffer
            int ri = recv_index(ev.dst, slot, ev.chunk, ev.phase_kind);
            recvbuf[(size_t)ri].value = ev.value;
            recvbuf[(size_t)ri].src = ev.src;
            recvbuf[(size_t)ri].remote_seq = ev.remote_seq;
            recvbuf[(size_t)ri].valid = 1;
            // 2. increment matching signal counter
            int si = sig_index(ev.dst, slot, ev.phase_kind, ev.phase_index, ev.chunk);
            signal[(size_t)si] += 1;
            counters[6]++;
            emit(MGA_EV_REMOTE_ARRIVE, op_index, ev.coll_id, (uint32_t)ev.src,
                 (uint32_t)ev.dst, ev.phase_kind, (uint32_t)ev.phase_index,
                 (uint32_t)ev.chunk, ev.value);
            // 3. return credit to reverse recorded link (edge src=credit_return_src)
            int rpos = c.pos_of_rank[(size_t)ev.credit_return_src];
            if (rpos >= 0) {
                int64_t nv = c.credit[(size_t)rpos] + 1;
                if (nv > max_credit) nv = max_credit;
                c.credit[(size_t)rpos] = nv;
            }
            counters[8]++;
            emit(MGA_EV_CREDIT_RETURN, op_index, ev.coll_id,
                 (uint32_t)ev.credit_return_src, (uint32_t)ev.credit_return_dst,
                 ev.phase_kind, (uint32_t)ev.phase_index, (uint32_t)ev.chunk, INT64_MIN);
            // 4. enqueue receiver action
            MgaOAction a;
            a.coll_id = ev.coll_id; a.coll_slot = slot;
            a.action_kind = (ev.phase_kind == MGA_PK_RS) ? MGA_ACT_REMOTE_RECV_REDUCE
                                                         : MGA_ACT_REMOTE_RECV_GATHER;
            a.chunk = ev.chunk; a.phase_index = ev.phase_index;
            a.src = ev.src; a.value = ev.value;
            enqueue(ev.dst, a);
        }
    }

    void op_poll_signal(const MgaRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        int rank = run.a0;
        uint64_t coll_id = (uint64_t)(uint32_t)run.a1;
        int pk = run.a2;
        int pidx = run.a3;
        int chunk = run.a4;
        uint64_t target = (uint64_t)(uint32_t)run.a5;

        int slot = find_coll(coll_id);
        bool invalid = false;
        if (rank < 0 || rank >= rank_count) invalid = true;
        if (slot < 0) invalid = true;
        if (pk < 0 || pk > 2) invalid = true;
        if (pidx < 0 || pidx >= phidx_bound) invalid = true;
        if (chunk < 0 || chunk >= chunk_max) invalid = true;
        if (invalid) {
            counters[18]++;
            emit(MGA_EV_INVALID, op_index, coll_id, (uint32_t)(rank >= 0 ? rank : -1),
                 UINT32_MAX, (uint8_t)((pk >= 0 && pk <= 2) ? pk : MGA_PK_NONE),
                 (uint32_t)pidx, (uint32_t)chunk, INT64_MIN);
            return;
        }
        int si = sig_index(rank, slot, pk, pidx, chunk);
        uint64_t cur = signal[(size_t)si];
        if (cur >= target) {
            counters[14]++;
            emit(MGA_EV_SIGNAL_READY, op_index, coll_id, (uint32_t)rank, UINT32_MAX,
                 (uint8_t)pk, (uint32_t)pidx, (uint32_t)chunk, (int64_t)cur);
        } else {
            counters[15]++;
            emit(MGA_EV_SIGNAL_WAIT, op_index, coll_id, (uint32_t)rank, UINT32_MAX,
                 (uint8_t)pk, (uint32_t)pidx, (uint32_t)chunk, (int64_t)cur);
        }
    }

    void op_cancel(const MgaRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        uint64_t coll_id = (uint64_t)(uint32_t)run.a0;
        int slot = find_coll(coll_id);
        if (slot < 0 || colls[(size_t)slot].status != MGA_ST_ACTIVE) {
            counters[18]++;
            emit(MGA_EV_INVALID, op_index, coll_id, UINT32_MAX, UINT32_MAX,
                 MGA_PK_NONE, UINT32_MAX, UINT32_MAX, INT64_MIN);
            return;
        }
        colls[(size_t)slot].status = MGA_ST_CANCELLED;
        counters[17]++;
        emit(MGA_EV_COLL_CANCEL, op_index, coll_id, UINT32_MAX, UINT32_MAX,
             MGA_PK_NONE, UINT32_MAX, UINT32_MAX, INT64_MIN);
    }

    void op_force_credit(const MgaRunSpec& run) {
        uint32_t op_index = (uint32_t)run.op_index;
        int src = run.a0, dst = run.a1, amount = run.a2;
        bool invalid = false;
        if (src < 0 || src >= rank_count || dst < 0 || dst >= rank_count || src == dst)
            invalid = true;
        // valid iff (src,dst) is a directed ring edge of some ACTIVE collective.
        bool any_edge = false;
        if (!invalid) {
            for (int i = 0; i < max_colls; ++i) {
                MgaOColl& c = colls[(size_t)i];
                if (!c.used || c.status != MGA_ST_ACTIVE) continue;
                int pos = c.pos_of_rank[(size_t)src];
                if (pos < 0) continue;
                if (next_rank(c, src) == dst) { any_edge = true; }
            }
            if (!any_edge) invalid = true;
        }
        if (invalid) {
            counters[18]++;
            emit(MGA_EV_INVALID, op_index, 0, (uint32_t)(src >= 0 ? src : -1),
                 (uint32_t)(dst >= 0 ? dst : -1), MGA_PK_NONE, UINT32_MAX,
                 UINT32_MAX, INT64_MIN);
            return;
        }
        for (int i = 0; i < max_colls; ++i) {
            MgaOColl& c = colls[(size_t)i];
            if (!c.used || c.status != MGA_ST_ACTIVE) continue;
            int pos = c.pos_of_rank[(size_t)src];
            if (pos < 0) continue;
            if (next_rank(c, src) != dst) continue;
            int64_t nv = c.credit[(size_t)pos] + (int64_t)amount;
            if (nv > max_credit) nv = max_credit;
            if (nv < 0) nv = 0;
            c.credit[(size_t)pos] = nv;
        }
        counters[9]++;
        emit(MGA_EV_CREDIT_FORCE, op_index, 0, (uint32_t)src, (uint32_t)dst,
             MGA_PK_NONE, UINT32_MAX, UINT32_MAX, (int64_t)amount);
    }

    // ---- hashing ----
    uint64_t hash_collectives() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<int> idxs;
        for (int i = 0; i < max_colls; ++i)
            if (colls[(size_t)i].used) idxs.push_back(i);
        std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
            return colls[(size_t)a].coll_id < colls[(size_t)b].coll_id;
        });
        for (int idx : idxs) {
            const MgaOColl& c = colls[(size_t)idx];
            for (int ai = 0; ai < c.active_rank_count; ++ai) {
                int r = c.active_ranks[(size_t)ai];
                for (int ch = 0; ch < c.chunk_count; ++ch) {
                    const MgaOCell& cell = c.cells[(size_t)(r * chunk_max + ch)];
                    uint64_t cid = c.coll_id, cseq = c.collective_seq;
                    uint8_t st = c.status, ph = c.phase;
                    uint32_t ru = (uint32_t)r, chu = (uint32_t)ch;
                    mga_o_fnv(&h, &cid, 8);
                    mga_o_fnv(&h, &cseq, 8);
                    mga_o_fnv(&h, &st, 1);
                    mga_o_fnv(&h, &ph, 1);
                    mga_o_fnv(&h, &ru, 4);
                    mga_o_fnv(&h, &chu, 4);
                    mga_o_fnv(&h, &cell.local_value, 8);
                    mga_o_fnv(&h, &cell.accum_value, 8);
                    mga_o_fnv(&h, &cell.final_value, 8);
                    mga_o_fnv(&h, &cell.reduced_owner_ready, 1);
                    mga_o_fnv(&h, &cell.final_ready, 1);
                    mga_o_fnv(&h, &cell.last_update_seq, 8);
                }
            }
        }
        return h;
    }

    uint64_t hash_signals() const {
        uint64_t h = 1469598103934665603ULL;
        for (int rank = 0; rank < rank_count; ++rank) {
            for (int slot = 0; slot < max_colls; ++slot) {
                if (!colls[(size_t)slot].used) continue;
                uint64_t cid = colls[(size_t)slot].coll_id;
                for (int pk = 0; pk < 3; ++pk) {
                    for (int pidx = 0; pidx < phidx_bound; ++pidx) {
                        for (int ch = 0; ch < chunk_max; ++ch) {
                            uint64_t v = signal[(size_t)sig_index(rank, slot, pk, pidx, ch)];
                            if (v == 0) continue;
                            uint32_t ru = (uint32_t)rank, pidu = (uint32_t)pidx, chu = (uint32_t)ch;
                            uint8_t pku = (uint8_t)pk;
                            mga_o_fnv(&h, &ru, 4);
                            mga_o_fnv(&h, &cid, 8);
                            mga_o_fnv(&h, &pku, 1);
                            mga_o_fnv(&h, &pidu, 4);
                            mga_o_fnv(&h, &chu, 4);
                            mga_o_fnv(&h, &v, 8);
                        }
                    }
                }
            }
        }
        return h;
    }

    uint64_t hash_credits() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<int> idxs;
        for (int i = 0; i < max_colls; ++i)
            if (colls[(size_t)i].used) idxs.push_back(i);
        std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
            return colls[(size_t)a].coll_id < colls[(size_t)b].coll_id;
        });
        for (int idx : idxs) {
            const MgaOColl& c = colls[(size_t)idx];
            uint64_t cid = c.coll_id;
            for (int pos = 0; pos < c.active_rank_count; ++pos) {
                int src = c.active_ranks[(size_t)pos];
                int dst = c.active_ranks[(size_t)((pos + 1) % c.active_rank_count)];
                uint32_t su = (uint32_t)src, du = (uint32_t)dst;
                uint64_t cr = (uint64_t)c.credit[(size_t)pos];
                mga_o_fnv(&h, &cid, 8);
                mga_o_fnv(&h, &su, 4);
                mga_o_fnv(&h, &du, 4);
                mga_o_fnv(&h, &cr, 8);
            }
        }
        return h;
    }

    uint64_t hash_pending() const {
        uint64_t h = 1469598103934665603ULL;
        std::vector<MgaORemote> sorted = pending;
        std::sort(sorted.begin(), sorted.end(), [](const MgaORemote& a, const MgaORemote& b) {
            if (a.due_clock != b.due_clock) return a.due_clock < b.due_clock;
            return a.remote_seq < b.remote_seq;
        });
        for (const MgaORemote& e : sorted) {
            uint32_t su = (uint32_t)e.src, du = (uint32_t)e.dst;
            uint64_t cid = e.coll_id;
            uint8_t pk = e.phase_kind;
            uint32_t pidu = (uint32_t)e.phase_index, chu = (uint32_t)e.chunk;
            mga_o_fnv(&h, &su, 4);
            mga_o_fnv(&h, &du, 4);
            mga_o_fnv(&h, &cid, 8);
            mga_o_fnv(&h, &pk, 1);
            mga_o_fnv(&h, &pidu, 4);
            mga_o_fnv(&h, &chu, 4);
            mga_o_fnv(&h, &e.value, 8);
            mga_o_fnv(&h, &e.due_clock, 8);
            mga_o_fnv(&h, &e.remote_seq, 8);
        }
        return h;
    }

    uint64_t hash_scheduler() const {
        uint64_t h = 1469598103934665603ULL;
        for (int rank = 0; rank < rank_count; ++rank) {
            const std::vector<MgaOAction>& q = sched[(size_t)rank];
            for (int pos = 0; pos < (int)q.size(); ++pos) {
                const MgaOAction& a = q[(size_t)pos];
                uint32_t ru = (uint32_t)rank, posu = (uint32_t)pos;
                uint64_t cid = a.coll_id;
                uint8_t ak = a.action_kind;
                uint32_t chu = (uint32_t)a.chunk, pidu = (uint32_t)a.phase_index;
                uint64_t ps = a.phase_seq;
                mga_o_fnv(&h, &ru, 4);
                mga_o_fnv(&h, &posu, 4);
                mga_o_fnv(&h, &cid, 8);
                mga_o_fnv(&h, &ak, 1);
                mga_o_fnv(&h, &chu, 4);
                mga_o_fnv(&h, &pidu, 4);
                mga_o_fnv(&h, &ps, 8);
            }
        }
        return h;
    }

    void step_once(const MgaRunSpec& run, MgaOracleResult* out) {
        run_event_hash = 1469598103934665603ULL;
        switch (run.op) {
            case MGA_OP_BEGIN_ALLREDUCE: op_begin(run); break;
            case MGA_OP_RANK_STEP:       op_rank_step(run); break;
            case MGA_OP_ADVANCE:         op_advance(run); break;
            case MGA_OP_POLL_SIGNAL:     op_poll_signal(run); break;
            case MGA_OP_CANCEL_COLL:     op_cancel(run); break;
            case MGA_OP_FORCE_CREDIT:    op_force_credit(run); break;
            default:
                counters[18]++;
                emit(MGA_EV_INVALID, (uint32_t)run.op_index, 0, UINT32_MAX, UINT32_MAX,
                     MGA_PK_NONE, UINT32_MAX, UINT32_MAX, INT64_MIN);
                break;
        }
        for (int i = 0; i < MGA_COUNTER_COUNT; ++i) out->counters[i] = counters[i];
        out->remote_event_hash = run_event_hash;
        out->collective_hash = hash_collectives();
        out->signal_hash = hash_signals();
        out->credit_hash = hash_credits();
        out->pending_remote_hash = hash_pending();
        out->scheduler_hash = hash_scheduler();
        out->clock_out = clock;
        out->event_seq_out = event_seq;
    }
};

static inline bool mga_check(const MgaOracleResult& exp, const MgaOracleResult& got,
                             std::string* err) {
    static const char* names[MGA_COUNTER_COUNT] = {
        "coll_begin", "local_ready", "phase_advance_rs", "phase_advance_ag",
        "remote_send", "send_credit_stall", "remote_arrive", "remote_stale_drop",
        "credit_return", "credit_force", "reduce_apply", "owner_reduced",
        "gather_apply", "coll_done", "signal_ready", "signal_wait",
        "action_stale_drop", "coll_cancel", "invalid_count"};
    for (int i = 0; i < MGA_COUNTER_COUNT; ++i) {
        if (exp.counters[i] != got.counters[i]) {
            if (err) {
                std::ostringstream o;
                o << "counter " << names[i] << " got " << got.counters[i]
                  << " expected " << exp.counters[i];
                *err = o.str();
            }
            return false;
        }
    }
#define MGA_CK(field) \
    if (exp.field != got.field) { \
        if (err) { std::ostringstream o; o << #field << " got 0x" << std::hex \
            << got.field << " expected 0x" << exp.field; *err = o.str(); } \
        return false; }
    MGA_CK(remote_event_hash);
    MGA_CK(collective_hash);
    MGA_CK(signal_hash);
    MGA_CK(credit_hash);
    MGA_CK(pending_remote_hash);
    MGA_CK(scheduler_hash);
    MGA_CK(clock_out);
    MGA_CK(event_seq_out);
#undef MGA_CK
    return true;
}

#endif  // MK_MULTIGPU_ALLREDUCE_ORACLE_HPP_
