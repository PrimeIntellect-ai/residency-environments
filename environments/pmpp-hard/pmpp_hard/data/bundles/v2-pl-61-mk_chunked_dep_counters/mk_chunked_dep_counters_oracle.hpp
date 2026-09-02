// file: mk_chunked_dep_counters_oracle.hpp
//
// Host reference model = canonical semantics for MK2 (Chunked Dependency-Counter
// Runtime with Released-Page Wakeups).  The device reference.cu and naive.cu
// must reproduce these exact outputs.
//
// DETERMINISTIC INTERPRETATIONS (ambiguities in designs/MK2.md resolved to the
// most specific deterministic reading):
//
//  D1. event_seq starts 0; for every emitted event we first do event_seq += 1
//      (mod 2^64) and the new value is the event's event_seq.  First event of a
//      run has event_seq == 1.
//  D2. payload_hash = FNV1a64 over, in order:
//        payload_seed:u64, producer_id:u64, edge_id:u32, chunk_id:u32,
//        store_seq:u64, epoch:u64
//      (epoch = the global epoch scalar at the time of the PRODUCE op).
//  D3. consume_value = FNV1a64 over, in order:
//        consume_seed:u64, consumer_id:u64, edge_id:u32, chunk_id:u32,
//        cell.counter:u64, cell.payload_hash:u64, wait_seq:u64.
//      It is carried in the CONSUME_CHUNK event's payload field.
//  D4. op_index is the global op index across the whole run, starting at 0 and
//      persisting across solution_run calls (like event_seq).
//  D5. edge_epoch update uses modulo (max_epoch + 1).  DEFINE_EDGE and
//      RESET_EDGE both do edge_epoch = (edge_epoch + 1) % (max_epoch + 1).
//      Initial edge_epoch == 0; the first DEFINE_EDGE makes it 1.
//  D6/D7. Per-event hash field mapping (fields not produced by an event use the
//      sentinel UINT32_MAX / UINT64_MAX; counter defaults to UINT64_MAX;
//      producer/consumer/payload default to 0):
//        EDGE_DEFINE / EDGE_RESET / RESET_STALL : edge_id set, chunk=MAX,
//            counter=MAX, prod=0, cons=0, payload=0.
//        STORE_ISSUE : edge,chunk; counter=MAX; prod=producer_id; cons=0;
//            payload=payload_hash (the assigned hash).
//        STORE_COMPLETE : edge,chunk; counter=new counter; prod=producer_id;
//            cons=0; payload=payload_hash.
//        PRODUCE_STALL_CHUNK : edge,chunk; counter=MAX; prod=producer_id; cons=0;
//            payload=0.
//        STORE_STALE_DROP : edge,chunk; counter=MAX; prod=producer_id; cons=0;
//            payload=store_seq.
//        WAITER_ARM : edge,chunk; counter=MAX; prod=0; cons=consumer_id;
//            payload=target.
//        WAITER_READY / WAITER_READY_IMMEDIATE / READY_REQUEUE :
//            edge,chunk; counter=cell.counter; prod=0; cons=consumer_id;
//            payload=target.
//        READY_STALE_DROP : edge,chunk; counter=MAX; prod=0; cons=consumer_id;
//            payload=0.
//        CONSUME_CHUNK : edge,chunk; counter=cell.counter; prod=0;
//            cons=consumer_id; payload=consume_value.
//        CHUNK_RELEASE : edge,chunk; counter=cell.counter; prod=0;
//            cons=consumer_id; payload=release_seq (= this event's event_seq).
//        WAITER_CANCEL : edge,chunk; counter=MAX; prod=0; cons=consumer_id;
//            payload=0.
//        COUNTER_FORCE : edge,chunk; counter=new counter; prod=0; cons=0;
//            payload=amount.
//        INVALID : edge=UINT32_MAX, chunk=MAX, counter=MAX, prod=0, cons=0,
//            payload=op.kind.
//  D8. CHUNK_RELEASE stores release_seq = event_seq assigned to that very
//      CHUNK_RELEASE event.
//  D9. CONSUME sets CONSUMING, computes consume_value, emits CONSUME_CHUNK; then
//      sets RELEASED, sets release_seq, clears producer_id and store_seq (keeps
//      counter and payload_hash), emits CHUNK_RELEASE; marks waiter CONSUMED.
//      cell.counter used for both events is the (unchanged) counter value.
// D10. ARM_WAIT immediate path: if cell page is READY and counter >= target,
//      waiter -> READY, ready entry enqueued (observed_epoch = edge_epoch),
//      WAITER_READY_IMMEDIATE.
// D11. A waiter is "nonterminal" iff state in {WAITING, READY}.  waiter_hash
//      hashes nonterminal waiters only.
// D12. Each ready entry stores observed_epoch = the edge's edge_epoch at the
//      moment of enqueue.
// D13. CONSUME per-entry decision order: (a) waiter absent/terminal/not-READY
//      OR observed_epoch != current edge_epoch -> READY_STALE_DROP; (b) cell not
//      READY -> READY_STALE_DROP; (c) cell.counter < waiter.target -> requeue
//      (waiter WAITING, reappend wait queue with original wait_seq) +
//      READY_REQUEUE; (d) else consume.  (a)/(b)/(c) do not count toward limit.
// D14. Lazy stale removal: on DEFINE_EDGE and RESET_EDGE every nonterminal
//      waiter of that edge is set to REMOVED (terminal, no event) and removed
//      from its wait queue.  Stale ready entries are dropped later in CONSUME
//      via the waiter-not-READY / observed_epoch checks.  REMOVED carries no
//      WAITER_CANCEL event (distinct from CANCEL_WAIT).
// D15. RESET_EDGE: invalid if edge not defined.  Else if any chunk is STORING,
//      READY, or CONSUMING -> emit a single RESET_STALL and return (no reset);
//      else if any nonterminal waiter exists for the edge -> emit a single
//      RESET_STALL and return; else increment edge_epoch, clear all chunks,
//      lazily remove waiters (none remain), emit EDGE_RESET.
// D16. PRODUCE processes chunks ascending; first busy chunk (STORING/READY/
//      CONSUMING) emits PRODUCE_STALL_CHUNK and stops; earlier chunks stay
//      scheduled.  EMPTY or RELEASED chunks are (re)issued.
// D17. ADVANCE: clock += delta; then process up to max_store_completions pending
//      stores with due_clock <= clock, scanned in queue order; non-due stores
//      are skipped and do not count.  A matching store (cell.store_seq == event
//      store_seq and cell STORING) completes (READY, set payload_hash, counter
//      += increment, STORE_COMPLETE, evaluate waiters); otherwise STORE_STALE_DROP.
//      Each processed pending store (matching or stale) counts toward the limit.
// D18. Waiter evaluation (after a store completes, and in FORCE_COUNTER): scan
//      the chunk wait queue by wait_seq.  Stale (armed_epoch != edge_epoch) or
//      cancelled/removed waiters are skipped and removed from the queue (set
//      REMOVED if stale; CANCELLED ones are just dropped from the queue).  A
//      WAITING waiter with target <= counter -> READY, ready entry (ready_seq++,
//      observed_epoch = edge_epoch), WAITER_READY.  Unmet targets stay.
// D19. ARM_WAIT invalid iff edge absent, chunk out of range, or the same
//      consumer_id already has a nonterminal waiter for the same cell.
// D20. CANCEL_WAIT: find the nonterminal waiter for (consumer,edge,chunk); if
//      none, INVALID; else set CANCELLED and emit WAITER_CANCEL.  It stays in
//      the wait queue until lazily skipped.
// D21. FORCE_COUNTER: invalid if edge absent or chunk out of range; else
//      counter += amount, COUNTER_FORCE, evaluate waiters (D18).  Does not make
//      the page READY.
// D22. Pending store queue is kept sorted by (due_clock, store_seq, edge_id,
//      chunk_id); insertion maintains the order; pending_hash hashes that order.
// D23. CONSUME(limit): limit == 0 (or negative) is a valid no-op.  Process the
//      ready queue front-to-back; stop once `limit` valid consumptions occur.
//      Dropped/requeued entries are removed from the front as encountered.
// D24. All counters and sequence values wrap modulo 2^64.
// D25. PRODUCE validity: edge defined, chunk_count > 0, first_chunk >= 0, and
//      first_chunk + chunk_count <= edge.chunk_count.  increment, payload_seed,
//      store_latency are taken as unsigned 32-bit operands (except increment in
//      arg_i64, taken as u64).
// D26. DEFINE_EDGE validity: edge_id in range, chunk_count in
//      [1, max_chunks_per_edge].  If already defined and any chunk is not EMPTY
//      or RELEASED -> INVALID.  Else (re)define: defined=1, set chunk_count,
//      increment edge_epoch, all chunks EMPTY/counter 0/cleared, lazily remove
//      this edge's waiters.
// D27. Two contract branches are faithfully implemented but provably unreachable
//      given the coupled invariants, so they act as defensive code that all
//      three implementations agree on:
//        * STORE_STALE_DROP (ADVANCE): a pending store can only exist while its
//          cell is STORING with the matching store_seq (a cell only leaves
//          STORING via that store's own completion, and DEFINE/RESET cannot
//          clear a STORING cell).  Hence the "matching" branch always holds.
//        * READY_REQUEUE (CONSUME, cell READY but counter < target): a waiter is
//          only enqueued when counter >= target, and counters are monotonically
//          non-decreasing within an epoch (stores and FORCE only add; DEFINE/
//          RESET that zero the counter also bump edge_epoch, which makes the
//          ready entry stale -> dropped, not requeued).  Hence within a live
//          epoch counter < target never holds for a ready entry.
//      They are still implemented exactly per spec for completeness.

#ifndef MK_CHUNKED_DEP_COUNTERS_ORACLE_HPP_
#define MK_CHUNKED_DEP_COUNTERS_ORACLE_HPP_

#include "mk_chunked_dep_counters_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

static inline uint64_t mk_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}
static inline void mk_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = mk_oracle_fnv_byte(v, p[i]);
    *h = v;
}
static inline void mk_fnv_u8(uint64_t* h, uint8_t v)  { mk_oracle_fnv_bytes(h, &v, 1); }
static inline void mk_fnv_u32(uint64_t* h, uint32_t v){ mk_oracle_fnv_bytes(h, &v, 4); }
static inline void mk_fnv_u64(uint64_t* h, uint64_t v){ mk_oracle_fnv_bytes(h, &v, 8); }

// payload_hash over (seed, producer, edge, chunk, store_seq, epoch).  D2.
static inline uint64_t mk_payload_hash(uint64_t seed, uint64_t producer,
                                       uint32_t edge, uint32_t chunk,
                                       uint64_t store_seq, uint64_t epoch) {
    uint64_t h = 1469598103934665603ULL;
    mk_fnv_u64(&h, seed);
    mk_fnv_u64(&h, producer);
    mk_fnv_u32(&h, edge);
    mk_fnv_u32(&h, chunk);
    mk_fnv_u64(&h, store_seq);
    mk_fnv_u64(&h, epoch);
    return h;
}

// consume_value over (seed, consumer, edge, chunk, counter, payload_hash, wait).
static inline uint64_t mk_consume_value(uint64_t seed, uint64_t consumer,
                                        uint32_t edge, uint32_t chunk,
                                        uint64_t counter, uint64_t payload_hash,
                                        uint64_t wait_seq) {
    uint64_t h = 1469598103934665603ULL;
    mk_fnv_u64(&h, seed);
    mk_fnv_u64(&h, consumer);
    mk_fnv_u32(&h, edge);
    mk_fnv_u32(&h, chunk);
    mk_fnv_u64(&h, counter);
    mk_fnv_u64(&h, payload_hash);
    mk_fnv_u64(&h, wait_seq);
    return h;
}

struct MkExpected {
    uint64_t counts[MK_COUNT_TOTAL] = {0};
    uint64_t event_hash = 0;
    uint64_t cell_hash = 0;
    uint64_t waiter_hash = 0;
    uint64_t ready_hash = 0;
    uint64_t pending_hash = 0;
    uint64_t state_scalars[6] = {0};
};

struct MkCell {
    uint64_t counter = 0;
    uint8_t  page_state = MK_PAGE_EMPTY;
    uint64_t payload_hash = 0;
    uint64_t producer_id = 0;   // 0 == none
    uint64_t store_seq = 0;     // 0 == none
    uint64_t release_seq = 0;   // 0 == none
    std::vector<int> wait_queue;  // waiter ids ordered by wait_seq
};

struct MkEdge {
    uint8_t  defined = 0;
    int32_t  chunk_count = 0;
    uint64_t edge_epoch = 0;
    std::vector<MkCell> cells;  // size = max_chunks_per_edge
};

struct MkWaiter {
    uint64_t consumer_id = 0;
    int32_t  edge_id = 0;
    int32_t  chunk_id = 0;
    uint64_t target = 0;
    uint64_t wait_seq = 0;
    uint64_t consume_seed = 0;
    uint64_t armed_epoch = 0;
    uint8_t  state = MK_WAIT_WAITING;
};

struct MkReadyEntry {
    uint64_t ready_seq = 0;
    int32_t  waiter_id = 0;
    uint64_t observed_epoch = 0;
};

struct MkPendingStore {
    uint64_t producer_id = 0;
    int32_t  edge_id = 0;
    int32_t  chunk_id = 0;
    uint64_t increment = 0;
    uint64_t payload_hash = 0;
    uint64_t store_seq = 0;
    uint64_t due_clock = 0;
};

struct MkOracleState {
    MkProblemSpec spec{};
    int edge_count = 0;
    int max_chunks = 0;
    uint64_t epoch_mod = 1;  // = max_epoch + 1

    uint64_t clock = 0;
    uint64_t event_seq = 0;
    uint64_t store_seq_next = 1;
    uint64_t wait_seq_next = 1;
    uint64_t ready_seq_next = 1;
    uint64_t epoch = 0;
    uint64_t op_index_global = 0;

    std::vector<MkEdge> edges;
    std::vector<MkWaiter> waiters;          // pool, index = waiter id
    std::vector<MkReadyEntry> ready_queue;  // ordered by ready_seq
    std::vector<MkPendingStore> pending;    // sorted by (due,seq,edge,chunk)

    uint64_t counts[MK_COUNT_TOTAL] = {0};
    uint64_t event_hash = 1469598103934665603ULL;

    void init(const MkProblemSpec& s) {
        spec = s;
        edge_count = s.edge_count;
        max_chunks = s.max_chunks_per_edge;
        epoch_mod = (uint64_t)s.max_epoch + 1;
        full_reset();
    }

    void full_reset() {
        clock = 0;
        event_seq = 0;
        store_seq_next = 1;
        wait_seq_next = 1;
        ready_seq_next = 1;
        epoch = 0;
        op_index_global = 0;

        edges.assign((size_t)edge_count, MkEdge{});
        for (auto& e : edges) {
            e.defined = 0;
            e.chunk_count = 0;
            e.edge_epoch = 0;
            e.cells.assign((size_t)max_chunks, MkCell{});
        }
        waiters.clear();
        ready_queue.clear();
        pending.clear();

        for (int i = 0; i < MK_COUNT_TOTAL; ++i) counts[i] = 0;
        event_hash = 1469598103934665603ULL;
    }

    // ---- event emission -------------------------------------------------
    void emit_event(uint8_t kind, uint32_t op_index, uint64_t producer,
                    uint64_t consumer, uint32_t edge_id, uint32_t chunk_or_max,
                    uint64_t counter_or_max, uint64_t payload) {
        event_seq += 1;  // D1
        uint64_t* h = &event_hash;
        mk_fnv_u8(h, kind);
        mk_fnv_u64(h, event_seq);
        mk_fnv_u32(h, op_index);
        mk_fnv_u64(h, clock);
        mk_fnv_u64(h, producer);
        mk_fnv_u64(h, consumer);
        mk_fnv_u32(h, edge_id);
        mk_fnv_u32(h, chunk_or_max);
        mk_fnv_u64(h, counter_or_max);
        mk_fnv_u64(h, payload);
    }

    void emit_invalid(uint32_t op_index, uint32_t opcode) {
        emit_event(MK_EV_INVALID, op_index, 0, 0, UINT32_MAX, UINT32_MAX,
                   UINT64_MAX, (uint64_t)opcode);
        counts[MK_COUNT_INVALID] += 1;
    }

    bool edge_defined(int e) const {
        return e >= 0 && e < edge_count && edges[(size_t)e].defined != 0;
    }

    // remove waiter id from its chunk wait queue (if present).
    void drop_from_wait_queue(int wid) {
        MkWaiter& w = waiters[(size_t)wid];
        MkCell& cell = edges[(size_t)w.edge_id].cells[(size_t)w.chunk_id];
        for (size_t i = 0; i < cell.wait_queue.size(); ++i) {
            if (cell.wait_queue[i] == wid) {
                cell.wait_queue.erase(cell.wait_queue.begin() + (long)i);
                break;
            }
        }
    }

    // Evaluate the wait queue of a cell after its counter changed.  D18.
    void evaluate_waiters(int e, int chunk, uint32_t op_index) {
        MkEdge& edge = edges[(size_t)e];
        MkCell& cell = edge.cells[(size_t)chunk];
        std::vector<int> kept;
        kept.reserve(cell.wait_queue.size());
        for (size_t i = 0; i < cell.wait_queue.size(); ++i) {
            int wid = cell.wait_queue[i];
            MkWaiter& w = waiters[(size_t)wid];
            // stale or cancelled/removed -> skip + remove from queue.
            if (w.state == MK_WAIT_CANCELLED || w.state == MK_WAIT_REMOVED ||
                w.state == MK_WAIT_CONSUMED) {
                continue;  // dropped
            }
            if (w.armed_epoch != edge.edge_epoch) {
                w.state = MK_WAIT_REMOVED;  // stale lazy removal, no event
                continue;
            }
            if (w.state == MK_WAIT_WAITING && w.target <= cell.counter) {
                w.state = MK_WAIT_READY;
                MkReadyEntry re;
                re.ready_seq = ready_seq_next++;
                re.waiter_id = wid;
                re.observed_epoch = edge.edge_epoch;
                ready_queue.push_back(re);
                emit_event(MK_EV_WAITER_READY, op_index, 0, w.consumer_id,
                           (uint32_t)e, (uint32_t)chunk, cell.counter, w.target);
                counts[MK_COUNT_WAITER_READY] += 1;
                kept.push_back(wid);  // stays nonterminal until consumed
            } else {
                kept.push_back(wid);
            }
        }
        cell.wait_queue = kept;
    }

    // ---- operations -----------------------------------------------------
    void op_define_edge(uint32_t op_index, const MkOp& op) {
        const int e = op.arg_a;
        const int cc = op.arg_b;
        if (e < 0 || e >= edge_count || cc <= 0 || cc > max_chunks) {
            emit_invalid(op_index, MK_OP_DEFINE_EDGE); return;
        }
        MkEdge& edge = edges[(size_t)e];
        if (edge.defined != 0) {
            for (int ci = 0; ci < edge.chunk_count; ++ci) {
                uint8_t ps = edge.cells[(size_t)ci].page_state;
                if (ps != MK_PAGE_EMPTY && ps != MK_PAGE_RELEASED) {
                    emit_invalid(op_index, MK_OP_DEFINE_EDGE); return;
                }
            }
        }
        // lazily remove this edge's nonterminal waiters (D14).
        lazy_remove_edge_waiters(e);

        edge.defined = 1;
        edge.chunk_count = cc;
        edge.edge_epoch = (edge.edge_epoch + 1) % epoch_mod;  // D5
        for (int ci = 0; ci < max_chunks; ++ci) {
            MkCell& c = edge.cells[(size_t)ci];
            c.counter = 0;
            c.page_state = MK_PAGE_EMPTY;
            c.payload_hash = 0;
            c.producer_id = 0;
            c.store_seq = 0;
            c.release_seq = 0;
            c.wait_queue.clear();
        }
        emit_event(MK_EV_EDGE_DEFINE, op_index, 0, 0, (uint32_t)e,
                   UINT32_MAX, UINT64_MAX, 0);
        counts[MK_COUNT_EDGE_DEFINED] += 1;
    }

    void lazy_remove_edge_waiters(int e) {
        for (auto& w : waiters) {
            if (w.edge_id == e &&
                (w.state == MK_WAIT_WAITING || w.state == MK_WAIT_READY)) {
                w.state = MK_WAIT_REMOVED;
            }
        }
        for (int ci = 0; ci < max_chunks; ++ci)
            edges[(size_t)e].cells[(size_t)ci].wait_queue.clear();
    }

    void op_reset_edge(uint32_t op_index, const MkOp& op) {
        const int e = op.arg_a;
        if (!edge_defined(e)) { emit_invalid(op_index, MK_OP_RESET_EDGE); return; }
        MkEdge& edge = edges[(size_t)e];

        bool busy = false;
        for (int ci = 0; ci < edge.chunk_count; ++ci) {
            uint8_t ps = edge.cells[(size_t)ci].page_state;
            if (ps == MK_PAGE_STORING || ps == MK_PAGE_READY || ps == MK_PAGE_CONSUMING) {
                busy = true; break;
            }
        }
        if (busy) {
            emit_event(MK_EV_RESET_STALL, op_index, 0, 0, (uint32_t)e,
                       UINT32_MAX, UINT64_MAX, 0);
            counts[MK_COUNT_RESET_STALL] += 1;
            return;
        }
        bool has_waiter = false;
        for (const auto& w : waiters) {
            if (w.edge_id == e &&
                (w.state == MK_WAIT_WAITING || w.state == MK_WAIT_READY)) {
                has_waiter = true; break;
            }
        }
        if (has_waiter) {
            emit_event(MK_EV_RESET_STALL, op_index, 0, 0, (uint32_t)e,
                       UINT32_MAX, UINT64_MAX, 0);
            counts[MK_COUNT_RESET_STALL] += 1;
            return;
        }
        edge.edge_epoch = (edge.edge_epoch + 1) % epoch_mod;
        for (int ci = 0; ci < max_chunks; ++ci) {
            MkCell& c = edge.cells[(size_t)ci];
            c.counter = 0;
            c.page_state = MK_PAGE_EMPTY;
            c.payload_hash = 0;
            c.producer_id = 0;
            c.store_seq = 0;
            c.release_seq = 0;
            c.wait_queue.clear();
        }
        emit_event(MK_EV_EDGE_RESET, op_index, 0, 0, (uint32_t)e,
                   UINT32_MAX, UINT64_MAX, 0);
        counts[MK_COUNT_EDGE_RESET] += 1;
    }

    void op_produce(uint32_t op_index, const MkOp& op) {
        const uint64_t producer_id = (uint64_t)(uint32_t)op.arg_a;
        const int e = op.arg_b;
        const int first_chunk = op.arg_c;
        const int chunk_count = op.arg_d;
        const uint64_t increment = (uint64_t)op.arg_i64;
        const uint64_t payload_seed = (uint64_t)(uint32_t)op.arg_e;
        const uint64_t store_latency = (uint64_t)(uint32_t)op.arg_f;

        if (!edge_defined(e) || chunk_count <= 0 || first_chunk < 0) {
            emit_invalid(op_index, MK_OP_PRODUCE); return;
        }
        MkEdge& edge = edges[(size_t)e];
        if ((int64_t)first_chunk + (int64_t)chunk_count > (int64_t)edge.chunk_count) {
            emit_invalid(op_index, MK_OP_PRODUCE); return;
        }

        for (int k = 0; k < chunk_count; ++k) {
            const int ci = first_chunk + k;
            MkCell& c = edge.cells[(size_t)ci];
            uint8_t ps = c.page_state;
            if (ps == MK_PAGE_STORING || ps == MK_PAGE_READY || ps == MK_PAGE_CONSUMING) {
                emit_event(MK_EV_PRODUCE_STALL_CHUNK, op_index, producer_id, 0,
                           (uint32_t)e, (uint32_t)ci, UINT64_MAX, 0);
                counts[MK_COUNT_PRODUCE_STALL_CHUNK] += 1;
                return;  // stop; earlier chunks remain scheduled
            }
            // EMPTY or RELEASED -> (re)issue store.
            const uint64_t store_seq = store_seq_next++;
            const uint64_t ph = mk_payload_hash(payload_seed, producer_id,
                                                (uint32_t)e, (uint32_t)ci,
                                                store_seq, epoch);
            c.page_state = MK_PAGE_STORING;
            c.producer_id = producer_id;
            c.store_seq = store_seq;
            c.payload_hash = ph;  // tentative; reaffirmed on completion
            c.release_seq = 0;

            MkPendingStore pe;
            pe.producer_id = producer_id;
            pe.edge_id = e;
            pe.chunk_id = ci;
            pe.increment = increment;
            pe.payload_hash = ph;
            pe.store_seq = store_seq;
            pe.due_clock = clock + store_latency;
            insert_pending(pe);

            emit_event(MK_EV_STORE_ISSUE, op_index, producer_id, 0,
                       (uint32_t)e, (uint32_t)ci, UINT64_MAX, ph);
            counts[MK_COUNT_STORE_ISSUED] += 1;
        }
    }

    // keep pending sorted by (due_clock, store_seq, edge_id, chunk_id).  D22.
    void insert_pending(const MkPendingStore& pe) {
        size_t pos = 0;
        while (pos < pending.size()) {
            const MkPendingStore& q = pending[pos];
            bool less = false;
            if (pe.due_clock != q.due_clock) less = pe.due_clock < q.due_clock;
            else if (pe.store_seq != q.store_seq) less = pe.store_seq < q.store_seq;
            else if (pe.edge_id != q.edge_id) less = pe.edge_id < q.edge_id;
            else less = pe.chunk_id < q.chunk_id;
            if (less) break;
            ++pos;
        }
        pending.insert(pending.begin() + (long)pos, pe);
    }

    void op_advance(uint32_t op_index, const MkOp& op) {
        const uint64_t delta = (uint64_t)(uint32_t)op.arg_a;
        const int max_completions = op.arg_b;
        clock = clock + delta;
        if (max_completions <= 0) return;

        int processed = 0;
        size_t i = 0;
        while (i < pending.size() && processed < max_completions) {
            if (pending[i].due_clock > clock) { ++i; continue; }  // non-due skipped
            MkPendingStore pe = pending[i];
            pending.erase(pending.begin() + (long)i);  // remove processed
            ++processed;

            MkCell& c = edges[(size_t)pe.edge_id].cells[(size_t)pe.chunk_id];
            if (c.store_seq == pe.store_seq && c.page_state == MK_PAGE_STORING) {
                c.page_state = MK_PAGE_READY;
                c.payload_hash = pe.payload_hash;
                c.counter = c.counter + pe.increment;  // mod 2^64
                emit_event(MK_EV_STORE_COMPLETE, op_index, pe.producer_id, 0,
                           (uint32_t)pe.edge_id, (uint32_t)pe.chunk_id,
                           c.counter, c.payload_hash);
                counts[MK_COUNT_STORE_COMPLETE] += 1;
                evaluate_waiters(pe.edge_id, pe.chunk_id, op_index);
            } else {
                emit_event(MK_EV_STORE_STALE_DROP, op_index, pe.producer_id, 0,
                           (uint32_t)pe.edge_id, (uint32_t)pe.chunk_id,
                           UINT64_MAX, pe.store_seq);
                counts[MK_COUNT_STORE_STALE_DROP] += 1;
            }
            // i unchanged because we erased element i.
        }
    }

    bool find_nonterminal_waiter(uint64_t consumer, int e, int chunk, int* out) const {
        for (size_t i = 0; i < waiters.size(); ++i) {
            const MkWaiter& w = waiters[i];
            if (w.consumer_id == consumer && w.edge_id == e && w.chunk_id == chunk &&
                (w.state == MK_WAIT_WAITING || w.state == MK_WAIT_READY)) {
                if (out) *out = (int)i;
                return true;
            }
        }
        return false;
    }

    void op_arm_wait(uint32_t op_index, const MkOp& op) {
        const uint64_t consumer_id = (uint64_t)(uint32_t)op.arg_a;
        const int e = op.arg_b;
        const int chunk = op.arg_c;
        const uint64_t target = (uint64_t)op.arg_i64;
        const uint64_t consume_seed = (uint64_t)(uint32_t)op.arg_e;

        if (!edge_defined(e) || chunk < 0 || chunk >= edges[(size_t)e].chunk_count) {
            emit_invalid(op_index, MK_OP_ARM_WAIT); return;
        }
        if (find_nonterminal_waiter(consumer_id, e, chunk, nullptr)) {
            emit_invalid(op_index, MK_OP_ARM_WAIT); return;
        }

        MkEdge& edge = edges[(size_t)e];
        MkCell& cell = edge.cells[(size_t)chunk];

        MkWaiter w;
        w.consumer_id = consumer_id;
        w.edge_id = e;
        w.chunk_id = chunk;
        w.target = target;
        w.wait_seq = wait_seq_next++;
        w.consume_seed = consume_seed;
        w.armed_epoch = edge.edge_epoch;
        const int wid = (int)waiters.size();

        if (cell.page_state == MK_PAGE_READY && cell.counter >= target) {
            w.state = MK_WAIT_READY;
            waiters.push_back(w);
            MkReadyEntry re;
            re.ready_seq = ready_seq_next++;
            re.waiter_id = wid;
            re.observed_epoch = edge.edge_epoch;
            ready_queue.push_back(re);
            emit_event(MK_EV_WAITER_READY_IMMEDIATE, op_index, 0, consumer_id,
                       (uint32_t)e, (uint32_t)chunk, cell.counter, target);
            counts[MK_COUNT_WAITER_READY_IMMEDIATE] += 1;
        } else {
            w.state = MK_WAIT_WAITING;
            waiters.push_back(w);
            cell.wait_queue.push_back(wid);  // ordered by wait_seq (increasing)
            emit_event(MK_EV_WAITER_ARM, op_index, 0, consumer_id,
                       (uint32_t)e, (uint32_t)chunk, UINT64_MAX, target);
            counts[MK_COUNT_WAITER_ARMED] += 1;
        }
    }

    void op_consume(uint32_t op_index, const MkOp& op) {
        const int limit = op.arg_a;
        if (limit <= 0) return;  // no-op (valid)

        int valid = 0;
        while (valid < limit && !ready_queue.empty()) {
            MkReadyEntry re = ready_queue.front();
            const int wid = re.waiter_id;
            MkWaiter& w = waiters[(size_t)wid];
            MkEdge& edge = edges[(size_t)w.edge_id];
            MkCell& cell = edge.cells[(size_t)w.chunk_id];

            // (a) waiter terminal / not READY / stale epoch.
            if (w.state != MK_WAIT_READY ||
                re.observed_epoch != edge.edge_epoch) {
                emit_event(MK_EV_READY_STALE_DROP, op_index, 0, w.consumer_id,
                           (uint32_t)w.edge_id, (uint32_t)w.chunk_id,
                           UINT64_MAX, 0);
                counts[MK_COUNT_READY_STALE_DROP] += 1;
                ready_queue.erase(ready_queue.begin());
                continue;
            }
            // (b) cell not READY.
            if (cell.page_state != MK_PAGE_READY) {
                emit_event(MK_EV_READY_STALE_DROP, op_index, 0, w.consumer_id,
                           (uint32_t)w.edge_id, (uint32_t)w.chunk_id,
                           UINT64_MAX, 0);
                counts[MK_COUNT_READY_STALE_DROP] += 1;
                ready_queue.erase(ready_queue.begin());
                continue;
            }
            // (c) counter below target -> requeue.
            if (cell.counter < w.target) {
                w.state = MK_WAIT_WAITING;
                cell.wait_queue.push_back(wid);
                std::sort(cell.wait_queue.begin(), cell.wait_queue.end(),
                          [this](int x, int y) {
                              return waiters[(size_t)x].wait_seq <
                                     waiters[(size_t)y].wait_seq;
                          });
                emit_event(MK_EV_READY_REQUEUE, op_index, 0, w.consumer_id,
                           (uint32_t)w.edge_id, (uint32_t)w.chunk_id,
                           cell.counter, w.target);
                counts[MK_COUNT_READY_REQUEUE] += 1;
                ready_queue.erase(ready_queue.begin());
                continue;
            }
            // (d) consume.
            cell.page_state = MK_PAGE_CONSUMING;
            const uint64_t cval = mk_consume_value(
                w.consume_seed, w.consumer_id, (uint32_t)w.edge_id,
                (uint32_t)w.chunk_id, cell.counter, cell.payload_hash, w.wait_seq);
            emit_event(MK_EV_CONSUME_CHUNK, op_index, 0, w.consumer_id,
                       (uint32_t)w.edge_id, (uint32_t)w.chunk_id,
                       cell.counter, cval);
            counts[MK_COUNT_CONSUME_CHUNK] += 1;

            cell.page_state = MK_PAGE_RELEASED;
            cell.producer_id = 0;  // clear producer/store fields (D9)
            cell.store_seq = 0;
            // release_seq == event_seq that this CHUNK_RELEASE will carry
            // (emit_event does event_seq += 1 first), so it is event_seq + 1.
            const uint64_t release_seq = event_seq + 1;
            cell.release_seq = release_seq;
            emit_event(MK_EV_CHUNK_RELEASE, op_index, 0, w.consumer_id,
                       (uint32_t)w.edge_id, (uint32_t)w.chunk_id,
                       cell.counter, release_seq);  // payload = release_seq (D8)
            counts[MK_COUNT_CHUNK_RELEASE] += 1;

            w.state = MK_WAIT_CONSUMED;
            ready_queue.erase(ready_queue.begin());
            ++valid;
        }
    }

    void op_cancel_wait(uint32_t op_index, const MkOp& op) {
        const uint64_t consumer_id = (uint64_t)(uint32_t)op.arg_a;
        const int e = op.arg_b;
        const int chunk = op.arg_c;
        int wid = -1;
        if (e < 0 || e >= edge_count ||
            !find_nonterminal_waiter(consumer_id, e, chunk, &wid)) {
            emit_invalid(op_index, MK_OP_CANCEL_WAIT); return;
        }
        MkWaiter& w = waiters[(size_t)wid];
        w.state = MK_WAIT_CANCELLED;
        emit_event(MK_EV_WAITER_CANCEL, op_index, 0, w.consumer_id,
                   (uint32_t)w.edge_id, (uint32_t)w.chunk_id, UINT64_MAX, 0);
        counts[MK_COUNT_WAITER_CANCEL] += 1;
        // stays in wait queue until lazily skipped (D20).
    }

    void op_force_counter(uint32_t op_index, const MkOp& op) {
        const int e = op.arg_a;
        const int chunk = op.arg_b;
        const uint64_t amount = (uint64_t)op.arg_i64;
        if (!edge_defined(e) || chunk < 0 || chunk >= edges[(size_t)e].chunk_count) {
            emit_invalid(op_index, MK_OP_FORCE_COUNTER); return;
        }
        MkCell& cell = edges[(size_t)e].cells[(size_t)chunk];
        cell.counter = cell.counter + amount;  // mod 2^64
        emit_event(MK_EV_COUNTER_FORCE, op_index, 0, 0, (uint32_t)e,
                   (uint32_t)chunk, cell.counter, amount);
        counts[MK_COUNT_COUNTER_FORCE] += 1;
        evaluate_waiters(e, chunk, op_index);
    }

    void apply_op(const MkOp& op) {
        const uint32_t op_index = (uint32_t)op_index_global;
        switch (op.kind) {
            case MK_OP_DEFINE_EDGE: op_define_edge(op_index, op); break;
            case MK_OP_RESET_EDGE: op_reset_edge(op_index, op); break;
            case MK_OP_PRODUCE: op_produce(op_index, op); break;
            case MK_OP_ADVANCE: op_advance(op_index, op); break;
            case MK_OP_ARM_WAIT: op_arm_wait(op_index, op); break;
            case MK_OP_CONSUME: op_consume(op_index, op); break;
            case MK_OP_CANCEL_WAIT: op_cancel_wait(op_index, op); break;
            case MK_OP_FORCE_COUNTER: op_force_counter(op_index, op); break;
            default: emit_invalid(op_index, (uint32_t)op.kind); break;
        }
        op_index_global += 1;
    }

    void run_ops(const MkOp* ops, int op_count) {
        for (int i = 0; i < op_count; ++i) apply_op(ops[i]);
    }

    // ---- structural checksums ------------------------------------------
    uint64_t compute_cell_hash() const {
        uint64_t h = 1469598103934665603ULL;
        for (int e = 0; e < edge_count; ++e) {
            const MkEdge& edge = edges[(size_t)e];
            if (edge.defined == 0) continue;
            for (int ci = 0; ci < edge.chunk_count; ++ci) {
                const MkCell& c = edge.cells[(size_t)ci];
                mk_fnv_u32(&h, (uint32_t)e);
                mk_fnv_u64(&h, edge.edge_epoch);
                mk_fnv_u32(&h, (uint32_t)ci);
                mk_fnv_u64(&h, c.counter);
                mk_fnv_u8(&h, c.page_state);
                mk_fnv_u64(&h, c.payload_hash);
                mk_fnv_u64(&h, c.producer_id);
                mk_fnv_u64(&h, c.store_seq);
                mk_fnv_u64(&h, c.release_seq);
            }
        }
        return h;
    }

    uint64_t compute_waiter_hash() const {
        std::vector<int> idx;
        for (size_t i = 0; i < waiters.size(); ++i) {
            const MkWaiter& w = waiters[i];
            if (w.state == MK_WAIT_WAITING || w.state == MK_WAIT_READY)
                idx.push_back((int)i);
        }
        std::sort(idx.begin(), idx.end(), [this](int x, int y) {
            const MkWaiter& a = waiters[(size_t)x];
            const MkWaiter& b = waiters[(size_t)y];
            if (a.edge_id != b.edge_id) return a.edge_id < b.edge_id;
            if (a.chunk_id != b.chunk_id) return a.chunk_id < b.chunk_id;
            if (a.wait_seq != b.wait_seq) return a.wait_seq < b.wait_seq;
            return a.consumer_id < b.consumer_id;
        });
        uint64_t h = 1469598103934665603ULL;
        for (int wid : idx) {
            const MkWaiter& w = waiters[(size_t)wid];
            mk_fnv_u64(&h, w.consumer_id);
            mk_fnv_u32(&h, (uint32_t)w.edge_id);
            mk_fnv_u32(&h, (uint32_t)w.chunk_id);
            mk_fnv_u64(&h, w.target);
            mk_fnv_u64(&h, w.wait_seq);
            mk_fnv_u64(&h, w.consume_seed);
            mk_fnv_u8(&h, w.state);
        }
        return h;
    }

    uint64_t compute_ready_hash() const {
        uint64_t h = 1469598103934665603ULL;
        for (const MkReadyEntry& re : ready_queue) {
            const MkWaiter& w = waiters[(size_t)re.waiter_id];
            mk_fnv_u64(&h, re.ready_seq);
            mk_fnv_u64(&h, w.consumer_id);
            mk_fnv_u32(&h, (uint32_t)w.edge_id);
            mk_fnv_u32(&h, (uint32_t)w.chunk_id);
            mk_fnv_u64(&h, re.observed_epoch);
        }
        return h;
    }

    uint64_t compute_pending_hash() const {
        uint64_t h = 1469598103934665603ULL;
        for (const MkPendingStore& pe : pending) {
            mk_fnv_u64(&h, pe.due_clock);
            mk_fnv_u64(&h, pe.store_seq);
            mk_fnv_u64(&h, pe.producer_id);
            mk_fnv_u32(&h, (uint32_t)pe.edge_id);
            mk_fnv_u32(&h, (uint32_t)pe.chunk_id);
            mk_fnv_u64(&h, pe.increment);
            mk_fnv_u64(&h, pe.payload_hash);
        }
        return h;
    }

    void snapshot(MkExpected* exp) {
        for (int i = 0; i < MK_COUNT_TOTAL; ++i) exp->counts[i] = counts[i];
        exp->event_hash = event_hash;
        exp->cell_hash = compute_cell_hash();
        exp->waiter_hash = compute_waiter_hash();
        exp->ready_hash = compute_ready_hash();
        exp->pending_hash = compute_pending_hash();
        exp->state_scalars[0] = clock;
        exp->state_scalars[1] = event_seq;
        exp->state_scalars[2] = store_seq_next;
        exp->state_scalars[3] = wait_seq_next;
        exp->state_scalars[4] = ready_seq_next;
        exp->state_scalars[5] = epoch;
    }
};

static const char* mk_count_name(int i) {
    switch (i) {
        case MK_COUNT_EDGE_DEFINED: return "edge_defined";
        case MK_COUNT_EDGE_RESET: return "edge_reset";
        case MK_COUNT_RESET_STALL: return "reset_stall";
        case MK_COUNT_STORE_ISSUED: return "store_issued";
        case MK_COUNT_STORE_COMPLETE: return "store_complete";
        case MK_COUNT_PRODUCE_STALL_CHUNK: return "produce_stall_chunk";
        case MK_COUNT_STORE_STALE_DROP: return "store_stale_drop";
        case MK_COUNT_WAITER_ARMED: return "waiter_armed";
        case MK_COUNT_WAITER_READY: return "waiter_ready";
        case MK_COUNT_WAITER_READY_IMMEDIATE: return "waiter_ready_immediate";
        case MK_COUNT_READY_STALE_DROP: return "ready_stale_drop";
        case MK_COUNT_READY_REQUEUE: return "ready_requeue";
        case MK_COUNT_CONSUME_CHUNK: return "consume_chunk";
        case MK_COUNT_CHUNK_RELEASE: return "chunk_release";
        case MK_COUNT_WAITER_CANCEL: return "waiter_cancel";
        case MK_COUNT_COUNTER_FORCE: return "counter_force";
        case MK_COUNT_INVALID: return "invalid_count";
        default: return "?";
    }
}

static const char* mk_scalar_name(int i) {
    switch (i) {
        case 0: return "clock";
        case 1: return "event_seq";
        case 2: return "store_seq_next";
        case 3: return "wait_seq_next";
        case 4: return "ready_seq_next";
        case 5: return "epoch";
        default: return "?";
    }
}

struct MkHostOutputsView {
    const uint64_t* counts;
    uint64_t event_hash;
    uint64_t cell_hash;
    uint64_t waiter_hash;
    uint64_t ready_hash;
    uint64_t pending_hash;
    const uint64_t* state_scalars;
};

static inline bool mk_check_outputs(const MkExpected& exp,
                                    const MkHostOutputsView& got,
                                    std::string* error) {
    for (int i = 0; i < MK_COUNT_TOTAL; ++i) {
        if (got.counts[i] != exp.counts[i]) {
            if (error) {
                std::ostringstream oss;
                oss << "count " << mk_count_name(i) << " mismatch: got "
                    << got.counts[i] << ", expected " << exp.counts[i];
                *error = oss.str();
            }
            return false;
        }
    }
    struct HPair { const char* n; uint64_t g; uint64_t e; };
    HPair pairs[] = {
        {"event_hash", got.event_hash, exp.event_hash},
        {"cell_hash", got.cell_hash, exp.cell_hash},
        {"waiter_hash", got.waiter_hash, exp.waiter_hash},
        {"ready_hash", got.ready_hash, exp.ready_hash},
        {"pending_hash", got.pending_hash, exp.pending_hash},
    };
    for (const HPair& p : pairs) {
        if (p.g != p.e) {
            if (error) {
                std::ostringstream oss;
                oss << p.n << " mismatch: got 0x" << std::hex << p.g
                    << ", expected 0x" << p.e;
                *error = oss.str();
            }
            return false;
        }
    }
    for (int i = 0; i < 6; ++i) {
        if (got.state_scalars[i] != exp.state_scalars[i]) {
            if (error) {
                std::ostringstream oss;
                oss << "scalar " << mk_scalar_name(i) << " mismatch: got "
                    << got.state_scalars[i] << ", expected " << exp.state_scalars[i];
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

#endif  // MK_CHUNKED_DEP_COUNTERS_ORACLE_HPP_
