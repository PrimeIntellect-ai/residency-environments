// file: mk_warp_pipeline_mbarrier_oracle.hpp

#ifndef MK_WARP_PIPELINE_MBARRIER_ORACLE_HPP_
#define MK_WARP_PIPELINE_MBARRIER_ORACLE_HPP_

#include "mk_warp_pipeline_mbarrier_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <deque>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ------------------------------------------------------------------ FNV-1a-64
struct MkwpFnv {
    uint64_t h = 1469598103934665603ULL;
    void byte(uint8_t b) {
        h ^= static_cast<uint64_t>(b);
        h *= 1099511628211ULL;
    }
    void bytes(const void* p, size_t n) {
        const uint8_t* q = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) byte(q[i]);
    }
    void u8(uint8_t v) { bytes(&v, sizeof(v)); }
    void u32(uint32_t v) { bytes(&v, sizeof(v)); }
    void u64(uint64_t v) { bytes(&v, sizeof(v)); }
};

// Standalone FNV-1a-64 over a tuple of u64s (used for result_hash).
static inline uint64_t mkwp_fnv_result(uint64_t tile_id, uint64_t tile_seq,
                                       uint64_t payload_seed, uint64_t buffer_id,
                                       uint64_t clock) {
    MkwpFnv f;
    f.u64(tile_id);
    f.u64(tile_seq);
    f.u64(payload_seed);
    f.u64(buffer_id);
    f.u64(clock);
    return f.h;
}

struct MkwpExpected {
    std::vector<int64_t> counts;
    int32_t op_index = 0;
    uint64_t clock = 0;
    uint64_t event_seq = 0;
    uint64_t pipe_event_hash = 0;
    uint64_t buffer_hash = 0;
    uint64_t barrier_hash = 0;
    uint64_t tile_hash = 0;
    uint64_t async_hash = 0;
    uint64_t state_checksum = 0;
};

struct MkwpHostOutputsView {
    const int64_t* counts;
    const int32_t* op_index_out;
    const uint64_t* clock_out;
    const uint64_t* event_seq_out;
    const uint64_t* pipe_event_hash;
    const uint64_t* buffer_hash;
    const uint64_t* barrier_hash;
    const uint64_t* tile_hash;
    const uint64_t* async_hash;
    const uint64_t* state_checksum;
};

// ------------------------------------------------------------------ Oracle
struct MkwpOracle {
    struct Tile {
        uint64_t tile_id = 0;
        uint64_t tile_seq = 0;
        uint64_t load_bytes = 0;
        uint64_t compute_iters = 0;
        uint64_t store_bytes = 0;
        uint64_t payload_seed = 0;
        int32_t status = MKWP_TS_QUEUED;
        uint32_t assigned_buffer = UINT32_MAX;
        uint64_t result_hash = 0;
    };
    struct Buffer {
        int32_t state = MKWP_BS_EMPTY;
        uint64_t tile_id = 0;          // 0 if none
        uint32_t load_barrier = 0;
        uint32_t compute_barrier = 0;
        uint32_t store_barrier = 0;
        uint8_t owner_role = 255;
        uint64_t last_phase_seq = 0;
    };
    struct Barrier {
        uint64_t phase = 0;
        uint32_t expected = 0;
        uint32_t arrived = 0;
        uint8_t completion_done = 0;
        uint64_t waiting_role_mask = 0;
        uint64_t last_arrive_seq = 0;
    };
    struct Async {
        int32_t async_kind = 0;
        uint64_t due_clock = 0;
        uint64_t async_seq = 0;
        uint64_t tile_id = 0;
        uint32_t buffer_id = 0;
        uint32_t barrier_id = 0;
        uint64_t phase = 0;
    };

    MkwpProblemSpec spec{};

    uint64_t clock = 0;
    uint64_t event_seq = 0;
    int32_t op_index = 0;
    uint64_t tile_seq_next = 1;
    uint64_t phase_seq_next = 1;
    uint64_t async_seq_next = 1;
    uint64_t pipe_hash = 1469598103934665603ULL;

    std::unordered_map<uint64_t, Tile> tiles;   // ALL tiles ever created
    std::deque<uint64_t> input_queue;           // tile ids in tile_seq order
    std::vector<Buffer> buffers;
    std::vector<Barrier> barriers;
    std::deque<Async> pending;                   // creation order

    // role ready queues hold buffer ids.
    std::deque<uint32_t> compute_ready;
    std::deque<uint32_t> store_ready;

    std::vector<int64_t> counts;

    void init(const MkwpProblemSpec& s) {
        spec = s;
        buffers.assign((size_t)spec.buffer_count, Buffer{});
        barriers.assign((size_t)spec.barrier_count, Barrier{});
        counts.assign(MKWP_COUNT_N, 0);
        reset();
    }

    void reset() {
        clock = 0;
        event_seq = 0;
        op_index = 0;
        tile_seq_next = 1;
        phase_seq_next = 1;
        async_seq_next = 1;
        pipe_hash = 1469598103934665603ULL;
        tiles.clear();
        input_queue.clear();
        for (int b = 0; b < spec.buffer_count; ++b) {
            Buffer buf;
            buf.state = MKWP_BS_EMPTY;
            buf.tile_id = 0;
            buf.load_barrier = (uint32_t)(b * MKWP_BARRIERS_PER_BUFFER + MKWP_BAR_LOAD);
            buf.compute_barrier = (uint32_t)(b * MKWP_BARRIERS_PER_BUFFER + MKWP_BAR_COMPUTE);
            buf.store_barrier = (uint32_t)(b * MKWP_BARRIERS_PER_BUFFER + MKWP_BAR_STORE);
            buf.owner_role = 255;
            buf.last_phase_seq = 0;
            buffers[(size_t)b] = buf;
        }
        for (auto& br : barriers) br = Barrier{};
        pending.clear();
        compute_ready.clear();
        store_ready.clear();
        for (auto& c : counts) c = 0;
    }

    Tile* find(uint64_t id) {
        auto it = tiles.find(id);
        return it == tiles.end() ? nullptr : &it->second;
    }
    static bool terminal(int32_t st) {
        return st == MKWP_TS_DONE || st == MKWP_TS_CANCELLED;
    }

    uint64_t peek_seq() const { return event_seq; }

    // -------------------------------------------------------------- event emit
    // Hash field order (contract section 4 pipe_event_hash):
    //   event_kind:u8; event_seq:u64; op_index:u32; clock:u64;
    //   role_id_or_UINT32_MAX:u32; tile_id_or_ZERO:u64; buffer_or_UINT32_MAX:u32;
    //   barrier_or_UINT32_MAX:u32; phase_or_UINT64_MAX:u64; aux_u64.
    void emit(uint8_t kind, uint32_t role, uint64_t tile_id, uint32_t buffer,
              uint32_t barrier, uint64_t phase, uint64_t aux) {
        const uint64_t seq = event_seq;
        MkwpFnv f; f.h = pipe_hash;
        f.u8(kind);
        f.u64(seq);
        f.u32((uint32_t)op_index);
        f.u64(clock);
        f.u32(role);
        f.u64(tile_id);
        f.u32(buffer);
        f.u32(barrier);
        f.u64(phase);
        f.u64(aux);
        pipe_hash = f.h;
        event_seq += 1;
    }

    void emit_invalid() {
        counts[MKWP_C_INVALID] += 1;
        emit(MKWP_EV_INVALID, UINT32_MAX, 0, UINT32_MAX, UINT32_MAX, UINT64_MAX, 0);
    }

    // -------------------------------------------------------------- queue helpers
    // Insert buffer id into a role ready queue ordered by (tile_seq, buffer_id).
    void push_role_queue(std::deque<uint32_t>& q, uint32_t buf_id) {
        const uint64_t key_seq = tiles[buffers[buf_id].tile_id].tile_seq;
        auto it = q.begin();
        for (; it != q.end(); ++it) {
            const uint32_t other = *it;
            const uint64_t oseq = tiles[buffers[other].tile_id].tile_seq;
            if (key_seq < oseq) break;
            if (key_seq == oseq && buf_id < other) break;
        }
        q.insert(it, buf_id);
    }
    void erase_role_queue(std::deque<uint32_t>& q, uint32_t buf_id) {
        for (auto it = q.begin(); it != q.end(); ++it) {
            if (*it == buf_id) { q.erase(it); return; }
        }
    }
    // Pick smallest (tile_seq, buffer id); returns index in q or -1.
    int front_role_queue(const std::deque<uint32_t>& q) {
        return q.empty() ? -1 : 0;  // queue kept sorted, so head is the min.
    }

    // -------------------------------------------------------------- operations
    void op_enqueue(uint64_t tile_id, uint64_t lb, uint64_t ci, uint64_t sb, uint64_t seed) {
        Tile* ex = find(tile_id);
        const bool exists_nonterminal = (ex && !terminal(ex->status));
        if (exists_nonterminal || (int64_t)tiles.size() >= spec.max_tiles ||
            lb == 0 || ci == 0 || sb == 0) {
            emit_invalid();
            return;
        }
        Tile t;
        t.tile_id = tile_id;
        t.tile_seq = tile_seq_next++;
        t.load_bytes = lb;
        t.compute_iters = ci;
        t.store_bytes = sb;
        t.payload_seed = seed;
        t.status = MKWP_TS_QUEUED;
        t.assigned_buffer = UINT32_MAX;
        t.result_hash = 0;
        tiles[tile_id] = t;
        input_queue.push_back(tile_id);
        counts[MKWP_C_TILE_ENQUEUE] += 1;
        emit(MKWP_EV_TILE_ENQUEUE, UINT32_MAX, tile_id, UINT32_MAX, UINT32_MAX,
             UINT64_MAX, t.tile_seq);
    }

    int lowest_empty_buffer() {
        for (int b = 0; b < spec.buffer_count; ++b) {
            if (buffers[(size_t)b].state == MKWP_BS_EMPTY) return b;
        }
        return -1;
    }
    // Pop the head noncancelled tile id from input queue; -1 if none. Cancelled /
    // terminal tiles at the head are discarded as they are encountered.
    uint64_t pop_head_noncancelled(bool* found) {
        while (!input_queue.empty()) {
            const uint64_t id = input_queue.front();
            Tile* t = find(id);
            // A tile leaves QUEUED only via load (then it is removed here) or
            // cancellation. If it is no longer QUEUED, drop it from the queue.
            if (!t || t->status != MKWP_TS_QUEUED) {
                input_queue.pop_front();
                continue;
            }
            input_queue.pop_front();
            *found = true;
            return id;
        }
        *found = false;
        return 0;
    }

    void op_loader_step(int loader_id, uint64_t limit) {
        if (loader_id < 0 || loader_id >= spec.loader_warps || limit == 0) {
            emit_invalid();
            return;
        }
        for (uint64_t i = 0; i < limit; ++i) {
            const int b = lowest_empty_buffer();
            if (b < 0) {
                counts[MKWP_C_LOADER_NO_BUFFER] += 1;
                emit(MKWP_EV_LOADER_NO_BUFFER, (uint32_t)loader_id, 0, UINT32_MAX,
                     UINT32_MAX, UINT64_MAX, 0);
                return;
            }
            bool found = false;
            const uint64_t tid = pop_head_noncancelled(&found);
            if (!found) {
                counts[MKWP_C_LOADER_NO_TILE] += 1;
                emit(MKWP_EV_LOADER_NO_TILE, (uint32_t)loader_id, 0, UINT32_MAX,
                     UINT32_MAX, UINT64_MAX, 0);
                return;
            }
            Buffer& buf = buffers[(size_t)b];
            Tile& t = tiles[tid];
            t.assigned_buffer = (uint32_t)b;
            t.status = MKWP_TS_LOADING;
            buf.tile_id = tid;
            buf.owner_role = MKWP_ROLE_LOADER;
            // init load barrier
            Barrier& lbar = barriers[buf.load_barrier];
            lbar.phase = lbar.phase + 1;
            lbar.expected = 1;
            lbar.arrived = 0;
            lbar.completion_done = 0;
            lbar.waiting_role_mask = 0;
            const uint64_t pseq = phase_seq_next++;
            buf.last_phase_seq = pseq;
            buf.state = MKWP_BS_LOAD_INFLIGHT;
            // async
            Async a;
            a.async_kind = MKWP_AK_LOAD_COMPLETE;
            a.due_clock = clock + t.load_bytes;
            a.async_seq = async_seq_next++;
            a.tile_id = tid;
            a.buffer_id = (uint32_t)b;
            a.barrier_id = buf.load_barrier;
            a.phase = lbar.phase;
            pending.push_back(a);
            counts[MKWP_C_LOAD_ISSUE] += 1;
            emit(MKWP_EV_LOAD_ISSUE, (uint32_t)loader_id, tid, (uint32_t)b,
                 buf.load_barrier, lbar.phase, pseq);
        }
    }

    void op_compute_step(int compute_id, uint64_t limit) {
        if (compute_id < 0 || compute_id >= spec.compute_warps || limit == 0) {
            emit_invalid();
            return;
        }
        for (uint64_t i = 0; i < limit; ++i) {
            const int idx = front_role_queue(compute_ready);
            if (idx < 0) {
                counts[MKWP_C_COMPUTE_NO_READY] += 1;
                emit(MKWP_EV_COMPUTE_NO_READY, (uint32_t)compute_id, 0, UINT32_MAX,
                     UINT32_MAX, UINT64_MAX, 0);
                return;
            }
            const uint32_t b = compute_ready[(size_t)idx];
            Buffer& buf = buffers[b];
            Barrier& lbar = barriers[buf.load_barrier];
            if (lbar.completion_done != 1) {
                lbar.waiting_role_mask |= (1ULL << MKWP_ROLE_COMPUTE);
                counts[MKWP_C_MBARRIER_WAIT_LOAD] += 1;
                emit(MKWP_EV_MBARRIER_WAIT_LOAD, (uint32_t)compute_id, buf.tile_id,
                     b, buf.load_barrier, lbar.phase, 0);
                return;
            }
            // proceed
            erase_role_queue(compute_ready, b);
            Tile& t = tiles[buf.tile_id];
            buf.state = MKWP_BS_COMPUTE_INFLIGHT;
            buf.owner_role = MKWP_ROLE_COMPUTE;
            t.status = MKWP_TS_COMPUTING;
            Barrier& cbar = barriers[buf.compute_barrier];
            cbar.phase = cbar.phase + 1;
            cbar.expected = 1;
            cbar.arrived = 0;
            cbar.completion_done = 0;
            cbar.waiting_role_mask = 0;
            const uint64_t pseq = phase_seq_next++;
            buf.last_phase_seq = pseq;
            Async a;
            a.async_kind = MKWP_AK_COMPUTE_COMPLETE;
            a.due_clock = clock + t.compute_iters;
            a.async_seq = async_seq_next++;
            a.tile_id = buf.tile_id;
            a.buffer_id = b;
            a.barrier_id = buf.compute_barrier;
            a.phase = cbar.phase;
            pending.push_back(a);
            counts[MKWP_C_COMPUTE_ISSUE] += 1;
            emit(MKWP_EV_COMPUTE_ISSUE, (uint32_t)compute_id, buf.tile_id, b,
                 buf.compute_barrier, cbar.phase, pseq);
        }
    }

    void op_storer_step(int storer_id, uint64_t limit) {
        if (storer_id < 0 || storer_id >= spec.storer_warps || limit == 0) {
            emit_invalid();
            return;
        }
        for (uint64_t i = 0; i < limit; ++i) {
            const int idx = front_role_queue(store_ready);
            if (idx < 0) {
                counts[MKWP_C_STORER_NO_READY] += 1;
                emit(MKWP_EV_STORER_NO_READY, (uint32_t)storer_id, 0, UINT32_MAX,
                     UINT32_MAX, UINT64_MAX, 0);
                return;
            }
            const uint32_t b = store_ready[(size_t)idx];
            Buffer& buf = buffers[b];
            Barrier& cbar = barriers[buf.compute_barrier];
            if (cbar.completion_done != 1) {
                cbar.waiting_role_mask |= (1ULL << MKWP_ROLE_STORER);
                counts[MKWP_C_MBARRIER_WAIT_COMPUTE] += 1;
                emit(MKWP_EV_MBARRIER_WAIT_COMPUTE, (uint32_t)storer_id, buf.tile_id,
                     b, buf.compute_barrier, cbar.phase, 0);
                return;
            }
            erase_role_queue(store_ready, b);
            Tile& t = tiles[buf.tile_id];
            buf.state = MKWP_BS_STORE_INFLIGHT;
            buf.owner_role = MKWP_ROLE_STORER;
            t.status = MKWP_TS_STORING;
            Barrier& sbar = barriers[buf.store_barrier];
            sbar.phase = sbar.phase + 1;
            sbar.expected = 1;
            sbar.arrived = 0;
            sbar.completion_done = 0;
            sbar.waiting_role_mask = 0;
            const uint64_t pseq = phase_seq_next++;
            buf.last_phase_seq = pseq;
            Async a;
            a.async_kind = MKWP_AK_STORE_COMPLETE;
            a.due_clock = clock + t.store_bytes;
            a.async_seq = async_seq_next++;
            a.tile_id = buf.tile_id;
            a.buffer_id = b;
            a.barrier_id = buf.store_barrier;
            a.phase = sbar.phase;
            pending.push_back(a);
            counts[MKWP_C_STORE_ISSUE] += 1;
            emit(MKWP_EV_STORE_ISSUE, (uint32_t)storer_id, buf.tile_id, b,
                 buf.store_barrier, sbar.phase, pseq);
        }
    }

    // Order pending due events by (due_clock, async_seq). We materialise indices.
    void op_advance(uint64_t delta, uint64_t max_async) {
        clock = clock + delta;  // wraps
        // process up to max_async due events.
        uint64_t processed = 0;
        while (processed < max_async) {
            // find min (due_clock, async_seq) among due events.
            int best = -1;
            uint64_t bd = 0, bs = 0;
            for (size_t k = 0; k < pending.size(); ++k) {
                const Async& a = pending[k];
                if (a.due_clock > clock) continue;  // not due
                if (best < 0 || a.due_clock < bd ||
                    (a.due_clock == bd && a.async_seq < bs)) {
                    best = (int)k; bd = a.due_clock; bs = a.async_seq;
                }
            }
            if (best < 0) break;  // no more due events
            Async a = pending[(size_t)best];
            pending.erase(pending.begin() + best);
            process_async(a);
            ++processed;
        }
    }

    void process_async(const Async& a) {
        Tile* t = find(a.tile_id);
        Buffer& buf = buffers[a.buffer_id];
        Barrier& bar = barriers[a.barrier_id];
        const bool stale =
            (t == nullptr) || terminal(t->status) ||
            (buf.tile_id != a.tile_id) ||
            (bar.phase != a.phase);
        if (stale) {
            counts[MKWP_C_ASYNC_STALE_DROP] += 1;
            emit(MKWP_EV_ASYNC_STALE_DROP, UINT32_MAX, a.tile_id, a.buffer_id,
                 a.barrier_id, a.phase, (uint64_t)a.async_kind);
            return;
        }
        // mbarrier arrive
        bar.arrived += 1;
        bar.last_arrive_seq = peek_seq();  // seq of the MBARRIER_ARRIVE event below
        counts[MKWP_C_MBARRIER_ARRIVE] += 1;
        emit(MKWP_EV_MBARRIER_ARRIVE, UINT32_MAX, a.tile_id, a.buffer_id,
             a.barrier_id, a.phase, (uint64_t)bar.arrived);
        if (bar.arrived == bar.expected && bar.completion_done == 0) {
            bar.completion_done = 1;
            counts[MKWP_C_MBARRIER_COMPLETE] += 1;
            emit(MKWP_EV_MBARRIER_COMPLETE, UINT32_MAX, a.tile_id, a.buffer_id,
                 a.barrier_id, a.phase, 0);
        }
        if (a.async_kind == MKWP_AK_LOAD_COMPLETE) {
            buf.state = MKWP_BS_LOAD_READY;
            buf.owner_role = 255;
            t->status = MKWP_TS_READY_COMPUTE;
            push_role_queue(compute_ready, a.buffer_id);
            counts[MKWP_C_LOAD_COMPLETE] += 1;
            emit(MKWP_EV_LOAD_COMPLETE, UINT32_MAX, a.tile_id, a.buffer_id,
                 a.barrier_id, a.phase, 0);
        } else if (a.async_kind == MKWP_AK_COMPUTE_COMPLETE) {
            t->result_hash = mkwp_fnv_result(t->tile_id, t->tile_seq,
                                             t->payload_seed, a.buffer_id, clock);
            buf.state = MKWP_BS_COMPUTE_READY;
            buf.owner_role = 255;
            t->status = MKWP_TS_READY_STORE;
            push_role_queue(store_ready, a.buffer_id);
            counts[MKWP_C_COMPUTE_COMPLETE] += 1;
            emit(MKWP_EV_COMPUTE_COMPLETE, UINT32_MAX, a.tile_id, a.buffer_id,
                 a.barrier_id, a.phase, t->result_hash);
        } else {  // STORE_COMPLETE
            t->status = MKWP_TS_DONE;
            t->assigned_buffer = UINT32_MAX;
            buf.state = MKWP_BS_EMPTY;
            buf.tile_id = 0;
            buf.owner_role = 255;
            counts[MKWP_C_STORE_COMPLETE] += 1;
            emit(MKWP_EV_STORE_COMPLETE, UINT32_MAX, a.tile_id, a.buffer_id,
                 a.barrier_id, a.phase, 0);
            counts[MKWP_C_TILE_DONE] += 1;
            emit(MKWP_EV_TILE_DONE, UINT32_MAX, a.tile_id, a.buffer_id,
                 UINT32_MAX, UINT64_MAX, 0);
        }
    }

    void op_cancel(uint64_t tile_id) {
        Tile* t = find(tile_id);
        if (!t || terminal(t->status)) {
            emit_invalid();
            return;
        }
        if (t->status == MKWP_TS_QUEUED) {
            t->status = MKWP_TS_CANCELLED;
            // Note: it remains in input_queue and will be skipped by the loader.
            counts[MKWP_C_TILE_CANCEL] += 1;
            emit(MKWP_EV_TILE_CANCEL, UINT32_MAX, tile_id, UINT32_MAX, UINT32_MAX,
                 UINT64_MAX, 0);
            return;
        }
        // owns a buffer
        const uint32_t b = t->assigned_buffer;
        t->status = MKWP_TS_CANCELLED;
        t->assigned_buffer = UINT32_MAX;
        // remove from any role ready queue it may sit in.
        erase_role_queue(compute_ready, b);
        erase_role_queue(store_ready, b);
        Buffer& buf = buffers[b];
        buf.state = MKWP_BS_EMPTY;
        buf.tile_id = 0;
        buf.owner_role = 255;
        // increment all three barrier phases, clear completion/arrived/waiters.
        for (int k = 0; k < MKWP_BARRIERS_PER_BUFFER; ++k) {
            uint32_t bid = (k == MKWP_BAR_LOAD) ? buf.load_barrier
                          : (k == MKWP_BAR_COMPUTE) ? buf.compute_barrier
                          : buf.store_barrier;
            Barrier& bar = barriers[bid];
            bar.phase = bar.phase + 1;
            bar.arrived = 0;
            bar.completion_done = 0;
            bar.waiting_role_mask = 0;
        }
        counts[MKWP_C_BUFFER_CANCEL_RELEASE] += 1;
        emit(MKWP_EV_BUFFER_CANCEL_RELEASE, UINT32_MAX, tile_id, b, UINT32_MAX,
             UINT64_MAX, 0);
    }

    bool buffer_references_barrier(uint32_t bid) {
        const uint32_t b = bid / MKWP_BARRIERS_PER_BUFFER;
        if (b >= (uint32_t)spec.buffer_count) return false;
        return buffers[b].state != MKWP_BS_EMPTY;
    }

    void op_reset_barrier(int32_t barrier_id) {
        if (barrier_id < 0 || barrier_id >= spec.barrier_count ||
            buffer_references_barrier((uint32_t)barrier_id)) {
            emit_invalid();
            return;
        }
        Barrier& bar = barriers[(size_t)barrier_id];
        bar.phase = bar.phase + 1;
        bar.expected = 0;
        bar.arrived = 0;
        bar.completion_done = 0;
        bar.waiting_role_mask = 0;
        counts[MKWP_C_BARRIER_RESET] += 1;
        emit(MKWP_EV_BARRIER_RESET, UINT32_MAX, 0, UINT32_MAX, (uint32_t)barrier_id,
             bar.phase, 0);
    }

    // -------------------------------------------------------------- snapshots
    uint64_t buffer_hash() const {
        MkwpFnv f;
        for (int b = 0; b < spec.buffer_count; ++b) {
            const Buffer& buf = buffers[(size_t)b];
            f.u32((uint32_t)b);
            f.u8((uint8_t)buf.state);
            f.u64(buf.tile_id);
            f.u32(buf.load_barrier);
            f.u32(buf.compute_barrier);
            f.u32(buf.store_barrier);
            f.u8(buf.owner_role);
            f.u64(buf.last_phase_seq);
        }
        return f.h;
    }
    uint64_t barrier_hash() const {
        MkwpFnv f;
        for (int b = 0; b < spec.barrier_count; ++b) {
            const Barrier& bar = barriers[(size_t)b];
            f.u32((uint32_t)b);
            f.u64(bar.phase);
            f.u32(bar.expected);
            f.u32(bar.arrived);
            f.u8(bar.completion_done);
            f.u64(bar.waiting_role_mask);
            f.u64(bar.last_arrive_seq);
        }
        return f.h;
    }
    // tile_hash over NONTERMINAL tiles by ascending tile id.
    uint64_t tile_hash() const {
        std::map<uint64_t, const Tile*> ord;
        for (const auto& kv : tiles) {
            if (!terminal(kv.second.status)) ord[kv.first] = &kv.second;
        }
        MkwpFnv f;
        for (const auto& kv : ord) {
            const Tile& t = *kv.second;
            f.u64(t.tile_id);
            f.u64(t.tile_seq);
            f.u64(t.load_bytes);
            f.u64(t.compute_iters);
            f.u64(t.store_bytes);
            f.u8((uint8_t)t.status);
            f.u32(t.assigned_buffer);
            f.u64(t.result_hash);
        }
        return f.h;
    }
    // async_hash over pending events in queue (creation) order.
    uint64_t async_hash() const {
        MkwpFnv f;
        for (const Async& a : pending) {
            f.u8((uint8_t)a.async_kind);
            f.u64(a.due_clock);
            f.u64(a.async_seq);
            f.u64(a.tile_id);
            f.u32(a.buffer_id);
            f.u32(a.barrier_id);
            f.u64(a.phase);
        }
        return f.h;
    }

    uint64_t state_checksum(uint64_t bufh, uint64_t barh, uint64_t tileh, uint64_t asyh) const {
        MkwpFnv f;
        f.u64(clock);
        f.u64(event_seq);
        f.u32((uint32_t)op_index);
        f.u64(tile_seq_next);
        f.u64(phase_seq_next);
        f.u64(async_seq_next);
        f.u64(pipe_hash);
        f.u64(bufh);
        f.u64(barh);
        f.u64(tileh);
        f.u64(asyh);
        for (int i = 0; i < MKWP_COUNT_N; ++i) f.u64((uint64_t)counts[(size_t)i]);
        return f.h;
    }

    void step_once(const MkwpRunSpec& run, MkwpExpected* exp) {
        switch (run.op_kind) {
            case MKWP_OP_ENQUEUE_TILE:
                op_enqueue(run.a_tile, run.a_load_bytes, run.a_compute_iters,
                           run.a_store_bytes, run.a_seed);
                break;
            case MKWP_OP_LOADER_STEP:
                op_loader_step(run.a_role_id, (uint64_t)(uint32_t)run.a_limit);
                break;
            case MKWP_OP_COMPUTE_STEP:
                op_compute_step(run.a_role_id, (uint64_t)(uint32_t)run.a_limit);
                break;
            case MKWP_OP_STORER_STEP:
                op_storer_step(run.a_role_id, (uint64_t)(uint32_t)run.a_limit);
                break;
            case MKWP_OP_ADVANCE:
                op_advance(run.a_delta, (uint64_t)(uint32_t)run.a_limit);
                break;
            case MKWP_OP_CANCEL_TILE:
                op_cancel(run.a_tile);
                break;
            case MKWP_OP_RESET_BARRIER:
                op_reset_barrier(run.a_barrier);
                break;
            default:
                break;
        }

        const int32_t this_op = op_index;
        op_index += 1;

        const uint64_t bufh = buffer_hash();
        const uint64_t barh = barrier_hash();
        const uint64_t tileh = tile_hash();
        const uint64_t asyh = async_hash();

        exp->counts = counts;
        exp->op_index = this_op;
        exp->clock = clock;
        exp->event_seq = event_seq;
        exp->pipe_event_hash = pipe_hash;
        exp->buffer_hash = bufh;
        exp->barrier_hash = barh;
        exp->tile_hash = tileh;
        exp->async_hash = asyh;
        exp->state_checksum = state_checksum(bufh, barh, tileh, asyh);
    }
};

static inline bool mkwp_check_outputs(
    const MkwpExpected& e, const MkwpHostOutputsView& g, std::string* err) {
    for (int i = 0; i < MKWP_COUNT_N; ++i) {
        if (g.counts[i] != e.counts[(size_t)i]) {
            if (err) {
                std::ostringstream o;
                o << "count[" << i << "] mismatch: got " << g.counts[i]
                  << " expected " << e.counts[(size_t)i];
                *err = o.str();
            }
            return false;
        }
    }
    auto chk64 = [&](const char* nm, uint64_t got, uint64_t exp) -> bool {
        if (got != exp) {
            if (err) {
                std::ostringstream o;
                o << nm << " mismatch: got 0x" << std::hex << got
                  << " expected 0x" << exp;
                *err = o.str();
            }
            return false;
        }
        return true;
    };
    if (g.op_index_out[0] != e.op_index) {
        if (err) { std::ostringstream o; o << "op_index mismatch got " << g.op_index_out[0] << " exp " << e.op_index; *err = o.str(); }
        return false;
    }
    if (!chk64("clock", g.clock_out[0], e.clock)) return false;
    if (!chk64("event_seq", g.event_seq_out[0], e.event_seq)) return false;
    if (!chk64("pipe_event_hash", g.pipe_event_hash[0], e.pipe_event_hash)) return false;
    if (!chk64("buffer_hash", g.buffer_hash[0], e.buffer_hash)) return false;
    if (!chk64("barrier_hash", g.barrier_hash[0], e.barrier_hash)) return false;
    if (!chk64("tile_hash", g.tile_hash[0], e.tile_hash)) return false;
    if (!chk64("async_hash", g.async_hash[0], e.async_hash)) return false;
    if (!chk64("state_checksum", g.state_checksum[0], e.state_checksum)) return false;
    return true;
}

/*
GRADER MODEL
  oracle.init(problem_spec); solution_init(...)
  solution_reset + oracle.reset()
  for each op:
    solution_run(...)
    oracle.step_once(...)
    mkwp_check_outputs(...)

Required harness coverage (>=6 scenarios, multi-step, adversarial):
  - basic full pipeline: enqueue -> load -> advance -> compute -> advance ->
    store -> advance (TILE_DONE), id reuse after DONE.
  - multi-buffer: more tiles than buffers -> LOADER_NO_BUFFER; round-robin slots.
  - mbarrier phase: COMPUTE before load completes -> MBARRIER_WAIT_LOAD; STORER
    before compute completes -> MBARRIER_WAIT_COMPUTE.
  - stale async after cancel: cancel a buffer-owning tile, then ADVANCE delivers
    the old async -> ASYNC_STALE_DROP (no completion of the new tile).
  - role-queue ordering by (tile_seq, buffer id) with out-of-order completions.
  - RESET_BARRIER valid (empty buffer) and invalid (busy buffer); INVALID/empty
    ops; reset + exact replay.
*/

#endif  // MK_WARP_PIPELINE_MBARRIER_ORACLE_HPP_
