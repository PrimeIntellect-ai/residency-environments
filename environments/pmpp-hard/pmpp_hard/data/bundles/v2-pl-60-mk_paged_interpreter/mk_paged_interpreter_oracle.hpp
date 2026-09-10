// file: mk_paged_interpreter_oracle.hpp
//
// CPU std:: oracle for MK1 (No-Bubbles SM Interpreter with Paged Shared Memory).
// Independent implementation: uses std::vector/std::sort and a plain struct
// model of the persistent state. This is the source of truth for grading.

#ifndef MK_PAGED_INTERPRETER_ORACLE_HPP_
#define MK_PAGED_INTERPRETER_ORACLE_HPP_

#include "mk_paged_interpreter_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

// ------------------------------------------------------------------ FNV-1a-64
struct MkFnv {
    uint64_t h = 1469598103934665603ULL;
    void byte(uint8_t b) { h ^= (uint64_t)b; h *= 1099511628211ULL; }
    void bytes(const void* p, size_t n) {
        const uint8_t* q = (const uint8_t*)p;
        for (size_t i = 0; i < n; ++i) byte(q[i]);
    }
    void u8(uint8_t v) { bytes(&v, 1); }
    void u32(uint32_t v) { bytes(&v, 4); }
    void u64(uint64_t v) { bytes(&v, 8); }
};

struct MkExpected {
    std::vector<int64_t> counts;
    int32_t op_index = 0;
    uint64_t clock = 0;
    uint64_t event_seq = 0;
    uint64_t event_hash = 0;
    uint64_t page_hash = 0;
    uint64_t sm_hash = 0;
    uint64_t counter_hash = 0;
    uint64_t pending_hash = 0;
    uint64_t state_checksum = 0;
};

struct MkHostOutputsView {
    const int64_t* counts;
    const int32_t* op_index_out;
    const uint64_t* clock_out;
    const uint64_t* event_seq_out;
    const uint64_t* event_hash;
    const uint64_t* page_hash;
    const uint64_t* sm_hash;
    const uint64_t* counter_hash;
    const uint64_t* pending_hash;
    const uint64_t* state_checksum;
};

// ------------------------------------------------------------------ Oracle
struct MkOracle {
    // ---- static program (host-resident copy) ----
    struct Instr {
        uint64_t instr_id, load_latency, compute_latency, store_latency, result_seed, out_increment;
        uint32_t out_counter, page_req_count, wait_count, req_offset, wait_offset;
    };
    struct Req { uint64_t tile_id; uint8_t mode; uint8_t release_after_store; };
    struct Wait { uint32_t counter_id; uint64_t target; };

    MkProblemSpec spec{};
    std::vector<int32_t> program_len;
    std::vector<int32_t> instr_offset;
    std::vector<Instr> instrs;
    std::vector<Req> reqs;
    std::vector<Wait> waits;

    int SM = 0, PAGES = 0, NCTR = 0;

    // ---- persistent state ----
    uint64_t clock = 0;
    uint64_t event_seq = 0;
    uint8_t  pass_active = 0;
    uint64_t pass_id = 0;
    uint64_t instr_instance_seq_next = 1;
    int32_t  op_index = 0;
    uint64_t event_hash = 1469598103934665603ULL;

    std::vector<uint64_t> counter;     // [NCTR]

    // Per SM.
    struct SmState {
        uint32_t pc = 0;
        uint64_t active_instance = 0;       // 0 == none
        uint8_t  stage = MK_STAGE_IDLE;
        uint64_t active_instr_id = 0;
        uint64_t active_compute_value = 0;
        // allocated page id per request index (UINT32_MAX absent).
        uint32_t alloc_page[MK_MAX_PAGE_REQS];
        // ready mask: 1 if request's page is HELD_READY (load done / reuse / scratch).
        uint8_t  ready_mask[MK_MAX_PAGE_REQS];
        uint32_t cur_req_count = 0;         // page_req_count of active instr
        uint32_t cur_wait_count = 0;
        uint32_t cur_req_offset = 0;
        uint32_t cur_wait_offset = 0;
        uint64_t cur_result_seed = 0;
        uint64_t cur_load_latency = 0;
        uint64_t cur_compute_latency = 0;
        uint64_t cur_store_latency = 0;
        uint32_t cur_out_counter = 0;
        uint64_t cur_out_increment = 0;
    };
    std::vector<SmState> sm;

    // Per page (flat: sm*PAGES + page).
    struct Page {
        uint8_t  state = MK_PG_FREE_EMPTY;
        uint64_t tile_id = 0;               // 0 == none
        uint8_t  mode = 255;                // 255 == none
        uint64_t owner_instance = 0;        // 0 == none
        uint32_t request_index = UINT32_MAX;
        uint64_t last_load_seq = 0;
        uint64_t last_release_seq = 0;
    };
    std::vector<Page> page;

    // Pending events.
    struct Pend {
        uint64_t due_clock;
        uint64_t create_event_seq;
        uint8_t  kind;                      // MkPendingKind
        int32_t  sm;
        uint64_t instance;
        uint32_t page_id;                   // UINT32_MAX if not page-specific
    };
    std::vector<Pend> pending;

    std::vector<int64_t> counts;

    // ----------------------------------------------------------------- init
    void init(const MkProblemSpec& s,
              const std::vector<int32_t>& plen,
              const std::vector<int32_t>& ioff,
              const std::vector<MkInstr>& fin,
              const std::vector<MkPageReq>& fre,
              const std::vector<MkWaitRec>& fwa) {
        spec = s;
        SM = s.sm_count; PAGES = s.pages_per_sm; NCTR = s.counter_count;
        program_len = plen;
        instr_offset = ioff;
        instrs.clear();
        for (const auto& i : fin) {
            Instr o;
            o.instr_id = i.instr_id; o.load_latency = i.load_latency;
            o.compute_latency = i.compute_latency; o.store_latency = i.store_latency;
            o.result_seed = i.result_seed; o.out_increment = i.out_increment;
            o.out_counter = i.out_counter; o.page_req_count = i.page_req_count;
            o.wait_count = i.wait_count; o.req_offset = i.req_offset; o.wait_offset = i.wait_offset;
            instrs.push_back(o);
        }
        reqs.clear();
        for (const auto& r : fre) reqs.push_back(Req{r.tile_id, r.mode, r.release_after_store});
        waits.clear();
        for (const auto& w : fwa) waits.push_back(Wait{w.counter_id, w.target});
        counts.assign(MK_COUNT_N, 0);
        sm.assign(SM, SmState{});
        page.assign((size_t)SM * PAGES, Page{});
        counter.assign(NCTR, 0);
        reset();
    }

    void reset() {
        clock = 0; event_seq = 0; pass_active = 0; pass_id = 0;
        instr_instance_seq_next = 1; op_index = 0;
        event_hash = 1469598103934665603ULL;
        counter.assign(NCTR, 0);
        for (int i = 0; i < SM; ++i) {
            SmState st;
            sm[i] = st;
        }
        for (auto& p : page) p = Page{};
        pending.clear();
        for (auto& c : counts) c = 0;
    }

    Page& pg(int s, int p) { return page[(size_t)s * PAGES + p]; }
    const Page& pg(int s, int p) const { return page[(size_t)s * PAGES + p]; }

    // ----------------------------------------------------------------- emit
    // Canonical event record fields:
    //   kind:u8, event_seq:u64, op_index:u32, clock:u64,
    //   sm_or_MAX:u32, instr_id_or_MAX:u64, instance_or_ZERO:u64,
    //   page_or_MAX:u32, counter_or_MAX:u32, value_or_MAX:u64.
    void emit(uint8_t kind, int32_t sm_or, uint64_t instr_id_or, uint64_t instance_or,
              uint32_t page_or, uint32_t counter_or, uint64_t value_or) {
        const uint64_t seq = event_seq;
        MkFnv f; f.h = event_hash;
        f.u8(kind);
        f.u64(seq);
        f.u32((uint32_t)op_index);
        f.u64(clock);
        f.u32(sm_or < 0 ? UINT32_MAX : (uint32_t)sm_or);
        f.u64(instr_id_or);
        f.u64(instance_or);
        f.u32(page_or);
        f.u32(counter_or);
        f.u64(value_or);
        event_hash = f.h;
        event_seq += 1;
        // count it (one accumulator per kind; PINNED_HINT folds into PAGE_RELEASE).
        if (kind == MK_EV_PAGE_RELEASE_PINNED_HINT) counts[MK_C_PAGE_RELEASE] += 1;
        else counts[kind_to_count(kind)] += 1;
    }
    static int kind_to_count(uint8_t kind) {
        // identity for kinds before PAGE_RELEASE_PINNED_HINT; after, shift by 1.
        if (kind < MK_EV_PAGE_RELEASE_PINNED_HINT) return (int)kind;     // 0..17
        if (kind == MK_EV_PAGE_RELEASE_PINNED_HINT) return MK_C_PAGE_RELEASE;
        return (int)kind - 1;  // COUNTER_INC(19)->18 ... INVALID(27)->25
    }

    uint64_t peek_seq() const { return event_seq; }

    void invalid() {
        emit(MK_EV_INVALID, -1, UINT64_MAX, 0, UINT32_MAX, UINT32_MAX, UINT64_MAX);
    }

    // ----------------------------------------------------------------- BEGIN_PASS
    bool begin_pass_valid(uint64_t /*new_pass*/) {
        if (pass_active == 1) {
            for (int s = 0; s < SM; ++s) if (sm[s].stage != MK_STAGE_DONE) return false;
            if (!pending.empty()) return false;
        }
        // If pending exists while inactive (cannot happen) also invalid; covered.
        if (!pending.empty()) return false;
        return true;
    }

    void op_begin_pass(uint64_t new_pass) {
        if (!begin_pass_valid(new_pass)) { invalid(); return; }
        pass_active = 1;
        pass_id = new_pass;
        for (auto& c : counter) c = 0;
        for (int s = 0; s < SM; ++s) {
            SmState st;             // pc=0, IDLE, no active instance
            sm[s] = st;
        }
        for (auto& p : page) p = Page{};  // FREE_EMPTY, cleared tile/owner/mode
        emit(MK_EV_PASS_BEGIN, -1, UINT64_MAX, 0, UINT32_MAX, UINT32_MAX, new_pass);
    }

    // ----------------------------------------------------------------- STEP_SM
    // Fetch the active instruction's static fields into SmState scratch.
    void load_active_instr(int s) {
        SmState& st = sm[s];
        const int32_t off = instr_offset[s] + (int32_t)st.pc;
        const Instr& in = instrs[(size_t)off];
        st.active_instr_id = in.instr_id;
        st.cur_req_count = in.page_req_count;
        st.cur_wait_count = in.wait_count;
        st.cur_req_offset = in.req_offset;
        st.cur_wait_offset = in.wait_offset;
        st.cur_result_seed = in.result_seed;
        st.cur_load_latency = in.load_latency;
        st.cur_compute_latency = in.compute_latency;
        st.cur_store_latency = in.store_latency;
        st.cur_out_counter = in.out_counter;
        st.cur_out_increment = in.out_increment;
        for (int r = 0; r < MK_MAX_PAGE_REQS; ++r) { st.alloc_page[r] = UINT32_MAX; st.ready_mask[r] = 0; }
    }

    // Returns true if it should stop (a stall/wait emitted) — caller breaks loop.
    bool step_one_transition(int s) {
        SmState& st = sm[s];

        if (st.stage == MK_STAGE_DONE) {
            emit(MK_EV_SM_DONE_WAIT, s, UINT64_MAX, 0, UINT32_MAX, UINT32_MAX, UINT64_MAX);
            return true;
        }

        if (st.stage == MK_STAGE_IDLE) {
            if (st.pc == (uint32_t)program_len[s]) {
                st.stage = MK_STAGE_DONE;
                emit(MK_EV_SM_PROGRAM_DONE, s, UINT64_MAX, 0, UINT32_MAX, UINT32_MAX, UINT64_MAX);
                return true;
            }
            // fetch
            st.active_instance = instr_instance_seq_next++;
            load_active_instr(s);
            st.stage = MK_STAGE_ACQUIRE;
            emit(MK_EV_INSTR_FETCH, s, st.active_instr_id, st.active_instance, UINT32_MAX, UINT32_MAX, UINT64_MAX);
            return false;
        }

        if (st.stage == MK_STAGE_ACQUIRE) {
            return do_acquire(s);
        }

        if (st.stage == MK_STAGE_LOADING) {
            // lowest request index still RESERVED_LOADING.
            for (uint32_t r = 0; r < st.cur_req_count; ++r) {
                const uint32_t pid = st.alloc_page[r];
                if (pid != UINT32_MAX && pg(s, (int)pid).state == MK_PG_RESERVED_LOADING) {
                    emit(MK_EV_LOAD_WAIT, s, st.active_instr_id, st.active_instance, pid, UINT32_MAX, (uint64_t)r);
                    return true;
                }
            }
            st.stage = MK_STAGE_WAIT_DEPS;
            return false;
        }

        if (st.stage == MK_STAGE_WAIT_DEPS) {
            for (uint32_t w = 0; w < st.cur_wait_count; ++w) {
                const Wait& wr = waits[(size_t)st.cur_wait_offset + w];
                if (counter[wr.counter_id] < wr.target) {
                    emit(MK_EV_COUNTER_WAIT, s, st.active_instr_id, st.active_instance,
                         UINT32_MAX, wr.counter_id, wr.target);
                    return true;
                }
            }
            // all waits satisfied -> compute value.
            MkFnv f;
            f.u64(pass_id);
            f.u64(st.active_instr_id);
            f.u64(st.active_instance);
            for (uint32_t r = 0; r < st.cur_req_count; ++r) {
                const Req& rq = reqs[(size_t)st.cur_req_offset + r];
                f.u64(rq.tile_id);
                f.u32(st.alloc_page[r]);
                f.u8(rq.mode);
            }
            for (uint32_t w = 0; w < st.cur_wait_count; ++w) {
                const Wait& wr = waits[(size_t)st.cur_wait_offset + w];
                f.u32(wr.counter_id);
                f.u64(wr.target);
                f.u64(counter[wr.counter_id]);   // observed value
            }
            f.u64(st.cur_result_seed);
            st.active_compute_value = f.h;

            if (st.cur_compute_latency == 0) {
                st.stage = MK_STAGE_STORE_READY;
                emit(MK_EV_COMPUTE_DONE_INLINE, s, st.active_instr_id, st.active_instance,
                     UINT32_MAX, UINT32_MAX, st.active_compute_value);
            } else {
                for (uint32_t r = 0; r < st.cur_req_count; ++r) {
                    const uint32_t pid = st.alloc_page[r];
                    if (pid != UINT32_MAX) pg(s, (int)pid).state = MK_PG_HELD_COMPUTING;
                }
                pending.push_back(Pend{clock + st.cur_compute_latency, peek_seq(),
                                       MK_PEND_COMPUTE_DONE, s, st.active_instance, UINT32_MAX});
                st.stage = MK_STAGE_COMPUTING;
                emit(MK_EV_COMPUTE_ISSUE, s, st.active_instr_id, st.active_instance,
                     UINT32_MAX, UINT32_MAX, st.active_compute_value);
            }
            return false;
        }

        if (st.stage == MK_STAGE_COMPUTING) {
            emit(MK_EV_COMPUTE_WAIT, s, st.active_instr_id, st.active_instance, UINT32_MAX, UINT32_MAX, UINT64_MAX);
            return true;
        }

        if (st.stage == MK_STAGE_STORE_READY) {
            for (uint32_t r = 0; r < st.cur_req_count; ++r) {
                const uint32_t pid = st.alloc_page[r];
                if (pid != UINT32_MAX) pg(s, (int)pid).state = MK_PG_STORE_PENDING;
            }
            pending.push_back(Pend{clock + st.cur_store_latency, peek_seq(),
                                   MK_PEND_STORE_DONE, s, st.active_instance, UINT32_MAX});
            st.stage = MK_STAGE_STORING;
            emit(MK_EV_STORE_ISSUE, s, st.active_instr_id, st.active_instance, UINT32_MAX, UINT32_MAX, UINT64_MAX);
            return false;
        }

        if (st.stage == MK_STAGE_STORING) {
            emit(MK_EV_STORE_WAIT, s, st.active_instr_id, st.active_instance, UINT32_MAX, UINT32_MAX, UINT64_MAX);
            return true;
        }
        return true;  // unreachable
    }

    // ACQUIRE: atomic page allocation. Returns true if stalled (stop STEP_SM).
    bool do_acquire(int s) {
        SmState& st = sm[s];
        const uint32_t nreq = st.cur_req_count;

        // Tentative assignment buffers.
        uint32_t chosen[MK_MAX_PAGE_REQS];
        uint8_t  is_reuse[MK_MAX_PAGE_REQS];
        for (uint32_t r = 0; r < nreq; ++r) { chosen[r] = UINT32_MAX; is_reuse[r] = 0; }
        // track tentatively-chosen page ids to avoid double-pick.
        // (we mark by scanning chosen[] for prior requests.)

        auto already_chosen = [&](uint32_t pid, uint32_t up_to) -> bool {
            for (uint32_t k = 0; k < up_to; ++k) if (chosen[k] == pid) return true;
            return false;
        };

        bool stalled = false;
        for (uint32_t r = 0; r < nreq; ++r) {
            const Req& rq = reqs[(size_t)st.cur_req_offset + r];
            uint32_t pick = UINT32_MAX;
            uint8_t reuse = 0;
            if (rq.mode == MK_MODE_READ) {
                // search lowest page_id FREE_RESIDENT matching tile, not chosen.
                for (int p = 0; p < PAGES; ++p) {
                    if (already_chosen((uint32_t)p, r)) continue;
                    const Page& pp = pg(s, p);
                    if (pp.state == MK_PG_FREE_RESIDENT && pp.tile_id == rq.tile_id) { pick = (uint32_t)p; reuse = 1; break; }
                }
            }
            if (pick == UINT32_MAX) {
                // lowest page_id FREE_EMPTY or FREE_RESIDENT not already chosen.
                for (int p = 0; p < PAGES; ++p) {
                    if (already_chosen((uint32_t)p, r)) continue;
                    const Page& pp = pg(s, p);
                    if (pp.state == MK_PG_FREE_EMPTY || pp.state == MK_PG_FREE_RESIDENT) { pick = (uint32_t)p; break; }
                }
            }
            if (pick == UINT32_MAX) { stalled = true; break; }
            chosen[r] = pick;
            is_reuse[r] = reuse;
        }

        if (stalled) {
            emit(MK_EV_PAGE_STALL, s, st.active_instr_id, st.active_instance, UINT32_MAX, UINT32_MAX, UINT64_MAX);
            return true;
        }

        // commit in request order.
        bool any_loading = false;
        for (uint32_t r = 0; r < nreq; ++r) {
            const Req& rq = reqs[(size_t)st.cur_req_offset + r];
            const uint32_t pid = chosen[r];
            Page& pp = pg(s, (int)pid);
            st.alloc_page[r] = pid;

            if (is_reuse[r]) {
                // reused page (was FREE_RESIDENT matching tile).
                pp.state = MK_PG_HELD_READY;
                pp.owner_instance = st.active_instance;
                pp.request_index = r;
                st.ready_mask[r] = 1;
                emit(MK_EV_PAGE_REUSE, s, st.active_instr_id, st.active_instance, pid, UINT32_MAX, rq.tile_id);
                continue;
            }
            // not reuse: if chosen page was FREE_RESIDENT with a different tile -> evict.
            if (pp.state == MK_PG_FREE_RESIDENT && pp.tile_id != rq.tile_id) {
                emit(MK_EV_PAGE_EVICT, s, st.active_instr_id, st.active_instance, pid, UINT32_MAX, pp.tile_id);
            }
            // Also a FREE_RESIDENT page chosen for a WRITE/SCRATCH whose tile happens
            // to match is still NOT a reuse (reuse only for READ); evict if resident.
            else if (pp.state == MK_PG_FREE_RESIDENT && rq.mode != MK_MODE_READ && pp.tile_id == rq.tile_id) {
                emit(MK_EV_PAGE_EVICT, s, st.active_instr_id, st.active_instance, pid, UINT32_MAX, pp.tile_id);
            }

            if (rq.mode == MK_MODE_SCRATCH) {
                pp.state = MK_PG_HELD_READY;
                pp.tile_id = rq.tile_id;
                pp.mode = rq.mode;
                pp.owner_instance = st.active_instance;
                pp.request_index = r;
                st.ready_mask[r] = 1;
                emit(MK_EV_PAGE_SCRATCH, s, st.active_instr_id, st.active_instance, pid, UINT32_MAX, rq.tile_id);
            } else {
                // READ or WRITE non-reuse -> RESERVED_LOADING + LOAD_DONE event.
                pp.state = MK_PG_RESERVED_LOADING;
                pp.tile_id = rq.tile_id;
                pp.mode = rq.mode;
                pp.owner_instance = st.active_instance;
                pp.request_index = r;
                st.ready_mask[r] = 0;
                pending.push_back(Pend{clock + st.cur_load_latency, peek_seq(),
                                       MK_PEND_LOAD_DONE, s, st.active_instance, pid});
                any_loading = true;
                emit(MK_EV_PAGE_LOAD_ISSUE, s, st.active_instr_id, st.active_instance, pid, UINT32_MAX, rq.tile_id);
            }
        }
        st.stage = any_loading ? MK_STAGE_LOADING : MK_STAGE_WAIT_DEPS;
        return false;
    }

    void op_step_sm(int sm_idx, uint32_t transition_limit) {
        if (sm_idx < 0 || sm_idx >= SM || pass_active == 0 || transition_limit == 0) { invalid(); return; }
        for (uint32_t t = 0; t < transition_limit; ++t) {
            if (step_one_transition(sm_idx)) break;
        }
    }

    // ----------------------------------------------------------------- ADVANCE
    // Canonical pending order: (due_clock, create_event_seq, sm, instance, page_id).
    static bool pend_less(const Pend& a, const Pend& b) {
        if (a.due_clock != b.due_clock) return a.due_clock < b.due_clock;
        if (a.create_event_seq != b.create_event_seq) return a.create_event_seq < b.create_event_seq;
        if (a.sm != b.sm) return a.sm < b.sm;
        if (a.instance != b.instance) return a.instance < b.instance;
        return a.page_id < b.page_id;
    }

    void op_advance(uint64_t delta, uint32_t max_events) {
        clock = clock + delta;   // wraps mod 2^64
        if (max_events == 0) return;  // only advance the clock.

        uint32_t processed = 0;
        while (processed < max_events) {
            // find the canonical-min pending event with due_clock <= clock.
            int best = -1;
            for (size_t i = 0; i < pending.size(); ++i) {
                if (pending[i].due_clock > clock) continue;
                if (best < 0 || pend_less(pending[i], pending[(size_t)best])) best = (int)i;
            }
            if (best < 0) break;
            Pend ev = pending[(size_t)best];
            pending.erase(pending.begin() + best);
            process_event(ev);
            ++processed;
        }
    }

    void process_event(const Pend& ev) {
        if (ev.kind == MK_PEND_LOAD_DONE) {
            Page& pp = pg(ev.sm, (int)ev.page_id);
            if (pp.owner_instance == ev.instance && pp.state == MK_PG_RESERVED_LOADING) {
                pp.state = MK_PG_HELD_READY;
                pp.last_load_seq = peek_seq();  // event_seq of the LOAD_DONE we are about to emit
                // mark ready mask
                SmState& st = sm[ev.sm];
                if (st.active_instance == ev.instance) {
                    for (uint32_t r = 0; r < st.cur_req_count; ++r)
                        if (st.alloc_page[r] == ev.page_id) st.ready_mask[r] = 1;
                }
                emit(MK_EV_LOAD_DONE, ev.sm, UINT64_MAX,
                     ev.instance, ev.page_id, UINT32_MAX, UINT64_MAX);
            } else {
                emit(MK_EV_STALE_EVENT_DROP, ev.sm, UINT64_MAX, ev.instance, ev.page_id, UINT32_MAX, (uint64_t)ev.kind);
            }
            return;
        }
        if (ev.kind == MK_PEND_COMPUTE_DONE) {
            SmState& st = sm[ev.sm];
            if (st.active_instance == ev.instance && st.stage == MK_STAGE_COMPUTING) {
                st.stage = MK_STAGE_STORE_READY;
                emit(MK_EV_COMPUTE_DONE, ev.sm, st.active_instr_id, ev.instance, UINT32_MAX, UINT32_MAX, st.active_compute_value);
            } else {
                emit(MK_EV_STALE_EVENT_DROP, ev.sm, UINT64_MAX, ev.instance, UINT32_MAX, UINT32_MAX, (uint64_t)ev.kind);
            }
            return;
        }
        // STORE_DONE
        {
            SmState& st = sm[ev.sm];
            if (st.active_instance == ev.instance && st.stage == MK_STAGE_STORING) {
                // 1. STORE_DONE
                emit(MK_EV_STORE_DONE, ev.sm, st.active_instr_id, ev.instance, UINT32_MAX, UINT32_MAX, UINT64_MAX);
                // 2. page releases in request order.
                for (uint32_t r = 0; r < st.cur_req_count; ++r) {
                    const uint32_t pid = st.alloc_page[r];
                    if (pid == UINT32_MAX) continue;
                    const Req& rq = reqs[(size_t)st.cur_req_offset + r];
                    Page& pp = pg(ev.sm, (int)pid);
                    pp.state = MK_PG_FREE_RESIDENT;
                    pp.owner_instance = 0;
                    pp.request_index = UINT32_MAX;
                    pp.last_release_seq = peek_seq();  // seq of the PAGE_RELEASE about to emit
                    if (rq.release_after_store == 1) {
                        emit(MK_EV_PAGE_RELEASE, ev.sm, st.active_instr_id, ev.instance, pid, UINT32_MAX, (uint64_t)r);
                    } else {
                        emit(MK_EV_PAGE_RELEASE_PINNED_HINT, ev.sm, st.active_instr_id, ev.instance, pid, UINT32_MAX, (uint64_t)r);
                    }
                }
                // 3. counter increment.
                if (st.cur_out_counter != UINT32_MAX) {
                    counter[st.cur_out_counter] = counter[st.cur_out_counter] + st.cur_out_increment;  // mod 2^64
                    emit(MK_EV_COUNTER_INC, ev.sm, st.active_instr_id, ev.instance, UINT32_MAX, st.cur_out_counter, counter[st.cur_out_counter]);
                }
                // 4. instruction completion.
                const uint64_t done_instr = st.active_instr_id;
                const uint64_t done_inst = ev.instance;
                st.active_instance = 0;
                st.pc += 1;
                st.stage = MK_STAGE_IDLE;
                for (int r = 0; r < MK_MAX_PAGE_REQS; ++r) { st.alloc_page[r] = UINT32_MAX; st.ready_mask[r] = 0; }
                emit(MK_EV_INSTR_COMPLETE, ev.sm, done_instr, done_inst, UINT32_MAX, UINT32_MAX, UINT64_MAX);
            } else {
                emit(MK_EV_STALE_EVENT_DROP, ev.sm, UINT64_MAX, ev.instance, UINT32_MAX, UINT32_MAX, (uint64_t)ev.kind);
            }
        }
    }

    // ----------------------------------------------------------------- HOST_INC_COUNTER
    void op_host_inc(uint32_t counter_id, uint64_t amount) {
        if (counter_id >= (uint32_t)NCTR) { invalid(); return; }
        counter[counter_id] = counter[counter_id] + amount;  // mod 2^64
        emit(MK_EV_HOST_COUNTER_INC, -1, UINT64_MAX, 0, UINT32_MAX, counter_id, counter[counter_id]);
    }

    // ----------------------------------------------------------------- ABORT_PASS
    void op_abort_pass() {
        if (pass_active == 0) { invalid(); return; }
        // discard all pending events in pending-event (canonical) order.
        std::vector<Pend> ordered = pending;
        std::sort(ordered.begin(), ordered.end(), pend_less);
        for (const Pend& ev : ordered) {
            emit(MK_EV_STALE_EVENT_DROP, ev.sm, UINT64_MAX, ev.instance,
                 ev.page_id == UINT32_MAX ? UINT32_MAX : ev.page_id, UINT32_MAX, (uint64_t)ev.kind);
        }
        pending.clear();
        // clear active SM instances, set every SM stage DONE.
        for (int s = 0; s < SM; ++s) {
            sm[s].active_instance = 0;
            sm[s].stage = MK_STAGE_DONE;
            for (int r = 0; r < MK_MAX_PAGE_REQS; ++r) { sm[s].alloc_page[r] = UINT32_MAX; sm[s].ready_mask[r] = 0; }
        }
        // release all pages to FREE_EMPTY, emitting PAGE_ABORT_RELEASE for every
        // non-free page in (sm,page) order.
        for (int s = 0; s < SM; ++s) {
            for (int p = 0; p < PAGES; ++p) {
                Page& pp = pg(s, p);
                const bool nonfree = !(pp.state == MK_PG_FREE_EMPTY || pp.state == MK_PG_FREE_RESIDENT);
                if (nonfree) {
                    emit(MK_EV_PAGE_ABORT_RELEASE, s, UINT64_MAX, 0, (uint32_t)p, UINT32_MAX, (uint64_t)pp.state);
                }
                pp = Page{};  // FREE_EMPTY, cleared
            }
        }
        pass_active = 0;
        emit(MK_EV_PASS_ABORT, -1, UINT64_MAX, 0, UINT32_MAX, UINT32_MAX, UINT64_MAX);
    }

    // ----------------------------------------------------------------- snapshots
    uint64_t page_snapshot_hash() const {
        MkFnv f;
        for (int s = 0; s < SM; ++s) {
            for (int p = 0; p < PAGES; ++p) {
                const Page& pp = pg(s, p);
                f.u32((uint32_t)s);
                f.u32((uint32_t)p);
                f.u8(pp.state);
                f.u64(pp.tile_id);          // 0 == none
                f.u8(pp.mode);              // 255 == none
                f.u64(pp.owner_instance);   // 0 == none
                f.u32(pp.request_index);
                f.u64(pp.last_load_seq);
                f.u64(pp.last_release_seq);
            }
        }
        return f.h;
    }

    uint64_t sm_snapshot_hash() const {
        MkFnv f;
        for (int s = 0; s < SM; ++s) {
            const SmState& st = sm[s];
            f.u32((uint32_t)s);
            f.u32(st.pc);
            f.u8(st.stage);
            f.u64(st.active_instance);  // 0 == none
            f.u64(st.active_instance == 0 ? UINT64_MAX : st.active_instr_id);
            f.u64(st.active_compute_value);
            const uint32_t n = st.active_instance == 0 ? 0 : st.cur_req_count;
            for (uint32_t r = 0; r < n; ++r) f.u32(st.alloc_page[r]);
        }
        return f.h;
    }

    uint64_t counter_snapshot_hash() const {
        MkFnv f;
        for (int c = 0; c < NCTR; ++c) {
            f.u32((uint32_t)c);
            f.u64(counter[c]);
        }
        return f.h;
    }

    uint64_t pending_snapshot_hash() const {
        std::vector<Pend> ordered = pending;
        std::sort(ordered.begin(), ordered.end(), pend_less);
        MkFnv f;
        for (const Pend& ev : ordered) {
            f.u64(ev.due_clock);
            f.u64(ev.create_event_seq);
            f.u8(ev.kind);
            f.u32((uint32_t)ev.sm);
            f.u64(ev.instance);
            f.u32(ev.page_id);
        }
        return f.h;
    }

    uint64_t master_checksum(uint64_t ph, uint64_t sh, uint64_t ch, uint64_t pendh) const {
        MkFnv f;
        f.u64(clock);
        f.u64(event_seq);
        f.u32((uint32_t)op_index);
        f.u8(pass_active);
        f.u64(pass_id);
        f.u64(instr_instance_seq_next);
        f.u64(event_hash);
        f.u64(ph);
        f.u64(sh);
        f.u64(ch);
        f.u64(pendh);
        for (int i = 0; i < MK_COUNT_N; ++i) f.u64((uint64_t)counts[i]);
        return f.h;
    }

    // ----------------------------------------------------------------- driver
    void step_once(const MkRunSpec& run, MkExpected* exp) {
        switch (run.op_kind) {
            case MK_OP_BEGIN_PASS:       op_begin_pass(run.a_pass_id); break;
            case MK_OP_STEP_SM:          op_step_sm(run.a_sm, run.a_transition_limit); break;
            case MK_OP_ADVANCE:          op_advance(run.a_delta, run.a_max_events); break;
            case MK_OP_HOST_INC_COUNTER: op_host_inc(run.a_counter, run.a_amount); break;
            case MK_OP_ABORT_PASS:       op_abort_pass(); break;
            default: invalid(); break;
        }
        const int32_t this_op = op_index;
        op_index += 1;

        const uint64_t ph = page_snapshot_hash();
        const uint64_t sh = sm_snapshot_hash();
        const uint64_t ch = counter_snapshot_hash();
        const uint64_t pendh = pending_snapshot_hash();

        exp->counts = counts;
        exp->op_index = this_op;
        exp->clock = clock;
        exp->event_seq = event_seq;
        exp->event_hash = event_hash;
        exp->page_hash = ph;
        exp->sm_hash = sh;
        exp->counter_hash = ch;
        exp->pending_hash = pendh;
        exp->state_checksum = master_checksum(ph, sh, ch, pendh);
    }
};

static inline bool mk_check_outputs(const MkExpected& e, const MkHostOutputsView& g, std::string* err) {
    for (int i = 0; i < MK_COUNT_N; ++i) {
        if (g.counts[i] != e.counts[(size_t)i]) {
            if (err) { std::ostringstream o; o << "count[" << i << "] got " << g.counts[i] << " exp " << e.counts[(size_t)i]; *err = o.str(); }
            return false;
        }
    }
    auto chk = [&](const char* nm, uint64_t got, uint64_t exp) -> bool {
        if (got != exp) {
            if (err) { std::ostringstream o; o << nm << " got 0x" << std::hex << got << " exp 0x" << exp; *err = o.str(); }
            return false;
        }
        return true;
    };
    if (g.op_index_out[0] != e.op_index) {
        if (err) { std::ostringstream o; o << "op_index got " << g.op_index_out[0] << " exp " << e.op_index; *err = o.str(); }
        return false;
    }
    if (!chk("clock", g.clock_out[0], e.clock)) return false;
    if (!chk("event_seq", g.event_seq_out[0], e.event_seq)) return false;
    if (!chk("event_hash", g.event_hash[0], e.event_hash)) return false;
    if (!chk("page_hash", g.page_hash[0], e.page_hash)) return false;
    if (!chk("sm_hash", g.sm_hash[0], e.sm_hash)) return false;
    if (!chk("counter_hash", g.counter_hash[0], e.counter_hash)) return false;
    if (!chk("pending_hash", g.pending_hash[0], e.pending_hash)) return false;
    if (!chk("state_checksum", g.state_checksum[0], e.state_checksum)) return false;
    return true;
}

#endif  // MK_PAGED_INTERPRETER_ORACLE_HPP_
