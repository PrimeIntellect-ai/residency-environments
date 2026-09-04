// file: mk_schedule_planner_oracle.hpp
//
// Host-side canonical model for MK7. This is the source of truth that BOTH
// device implementations (reference and naive) must match exactly. It is an
// INDEPENDENT third implementation (STL-backed, host code).

#ifndef MK_SCHEDULE_PLANNER_ORACLE_HPP_
#define MK_SCHEDULE_PLANNER_ORACLE_HPP_

#include "mk_schedule_planner_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Event-kind enumeration (stable byte values, hashed into the event stream).
// ---------------------------------------------------------------------------
enum MkEventKind : uint8_t {
    MK_EK_INSTR_ADD = 0,
    MK_EK_EDGE_ADD = 1,
    MK_EK_PLAN_INVALIDATE = 2,
    MK_EK_PLAN_EMPTY = 3,
    MK_EK_PLAN_STALL = 4,
    MK_EK_PLAN_PLACE = 5,
    MK_EK_PLAN_COMMIT = 6,
    MK_EK_EXEC_INSTR = 7,
    MK_EK_EDGE_SIGNAL = 8,
    MK_EK_INSTR_CANCEL = 9,
    MK_EK_EPOCH_ADVANCE = 10,
    MK_EK_INVALID = 11,
};

static const uint64_t MK_U64_MAX = 0xFFFFFFFFFFFFFFFFULL;
static const uint32_t MK_U32_MAX = 0xFFFFFFFFu;
static const uint64_t MK_FNV_BASIS = 1469598103934665603ULL;
static const uint64_t MK_FNV_PRIME = 1099511628211ULL;

// ---------------------------------------------------------------------------
// FNV-1a-64 helpers.
// ---------------------------------------------------------------------------
static inline uint64_t mk_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= MK_FNV_PRIME;
    return h;
}
static inline void mk_oracle_fnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = mk_oracle_fnv_byte(v, b[i]);
    *h = v;
}
static inline void mk_oracle_fnv_u8(uint64_t* h, uint8_t v) { mk_oracle_fnv_bytes(h, &v, 1); }
static inline void mk_oracle_fnv_u32(uint64_t* h, uint32_t v) { mk_oracle_fnv_bytes(h, &v, 4); }
static inline void mk_oracle_fnv_u64(uint64_t* h, uint64_t v) { mk_oracle_fnv_bytes(h, &v, 8); }

// ---------------------------------------------------------------------------
// Expected output snapshot.
// ---------------------------------------------------------------------------
struct MkExpected {
    uint64_t instr_added = 0;
    uint64_t edge_added = 0;
    uint64_t plan_invalidated = 0;
    uint64_t plan_empty = 0;
    uint64_t plan_stall = 0;
    uint64_t plan_placed = 0;
    uint64_t plan_committed = 0;
    uint64_t instr_executed = 0;
    uint64_t edge_signaled = 0;
    uint64_t instr_cancelled = 0;
    uint64_t epoch_advanced = 0;
    uint64_t invalid_count = 0;

    uint64_t planner_event_hash = MK_FNV_BASIS;

    uint64_t plan_hash = MK_FNV_BASIS;
    uint64_t instr_hash = MK_FNV_BASIS;
    uint64_t edge_hash = MK_FNV_BASIS;
    uint64_t page_interval_hash = MK_FNV_BASIS;

    uint64_t committed_epoch = 0;
    uint64_t live_instr_count = 0;
    uint64_t planned_interval_count = 0;
};

struct MkHostOutputsView {
    const uint64_t* instr_added;
    const uint64_t* edge_added;
    const uint64_t* plan_invalidated;
    const uint64_t* plan_empty;
    const uint64_t* plan_stall;
    const uint64_t* plan_placed;
    const uint64_t* plan_committed;
    const uint64_t* instr_executed;
    const uint64_t* edge_signaled;
    const uint64_t* instr_cancelled;
    const uint64_t* epoch_advanced;
    const uint64_t* invalid_count;
    const uint64_t* planner_event_hash;
    const uint64_t* plan_hash;
    const uint64_t* instr_hash;
    const uint64_t* edge_hash;
    const uint64_t* page_interval_hash;
    const uint64_t* committed_epoch;
    const uint64_t* live_instr_count;
    const uint64_t* planned_interval_count;
};

// ---------------------------------------------------------------------------
// Internal tables.
// ---------------------------------------------------------------------------
struct MkOracleInstr {
    uint64_t instr_id;
    uint64_t instr_seq;
    uint64_t duration;
    uint64_t page_count;
    uint64_t page_keys[MK_MAX_PAGE_KEYS_PER_INSTR];
    uint64_t release_delay;
    uint64_t payload_seed;
    uint8_t status;
    uint64_t critical_height;
    uint64_t completion_counter;
};

struct MkOracleEdge {
    uint64_t edge_id;
    uint64_t src_instr;
    uint64_t dst_instr;
    uint32_t chunk_id;
    uint64_t target_increment;
    uint64_t edge_seq;
};

// A planned interval lives on exactly one SM.
struct MkOraclePlan {
    uint64_t plan_seq;
    uint64_t instr_id;
    uint32_t sm;
    uint64_t start_wave;
    uint64_t start_tick;
    uint64_t finish_tick;
    uint64_t release_tick;
    uint64_t page_count;
    uint64_t page_keys[MK_MAX_PAGE_KEYS_PER_INSTR];
};

struct MkOracleState {
    MkProblemSpec spec{};
    uint32_t sm_count = 0;
    uint32_t pages_per_sm = 0;
    uint64_t wave_quantum = 0;

    uint64_t event_seq = 0;
    uint32_t op_index = 0;
    uint64_t instr_seq_next = 1;
    uint64_t edge_seq_next = 1;
    uint64_t plan_seq_next = 1;
    uint64_t committed_epoch = 0;

    // instr_id -> instr
    std::map<uint64_t, MkOracleInstr> instrs;
    // edge_id -> edge
    std::map<uint64_t, MkOracleEdge> edges;
    // plan intervals (one entry per planned/committed instruction)
    std::vector<MkOraclePlan> plans;

    // accumulating counters / hashes (persist across steps)
    MkExpected acc;

    void init(const MkProblemSpec& s) {
        spec = s;
        sm_count = (uint32_t)s.sm_count;
        pages_per_sm = (uint32_t)s.pages_per_sm;
        wave_quantum = s.wave_quantum;
        reset();
    }

    void reset() {
        event_seq = 0;
        op_index = 0;
        instr_seq_next = 1;
        edge_seq_next = 1;
        plan_seq_next = 1;
        committed_epoch = 0;
        instrs.clear();
        edges.clear();
        plans.clear();
        acc = MkExpected();
    }

    // ---- event hashing ----
    // planner_event_hash schema (per spec section 4):
    //   event_kind:u8; event_seq:u64; op_index:u32; instr_id_or_ZERO:u64;
    //   edge_id_or_ZERO:u64; sm_or_UINT32_MAX:u32; start_tick_or_UINT64_MAX:u64;
    //   finish_tick_or_UINT64_MAX:u64; aux_u64.
    void emit_event(uint8_t kind, uint64_t seq, uint32_t opidx,
                    uint64_t instr_id_or0, uint64_t edge_id_or0,
                    uint32_t sm_or_max, uint64_t start_or_max,
                    uint64_t finish_or_max, uint64_t aux) {
        uint64_t* h = &acc.planner_event_hash;
        mk_oracle_fnv_u8(h, kind);
        mk_oracle_fnv_u64(h, seq);
        mk_oracle_fnv_u32(h, opidx);
        mk_oracle_fnv_u64(h, instr_id_or0);
        mk_oracle_fnv_u64(h, edge_id_or0);
        mk_oracle_fnv_u32(h, sm_or_max);
        mk_oracle_fnv_u64(h, start_or_max);
        mk_oracle_fnv_u64(h, finish_or_max);
        mk_oracle_fnv_u64(h, aux);
    }

    // ---- graph helpers ----
    bool is_noncancelled(uint64_t id) const {
        auto it = instrs.find(id);
        return it != instrs.end() && it->second.status != MK_ST_CANCELLED;
    }

    // successors of src via noncancelled edges (edge alive in map), sorted by
    // successor instr id ascending, deduplicated.
    std::vector<uint64_t> noncancelled_successors(uint64_t src) const {
        std::vector<uint64_t> out;
        for (const auto& kv : edges) {
            const MkOracleEdge& e = kv.second;
            if (e.src_instr != src) continue;
            if (!is_noncancelled(e.dst_instr)) continue;
            out.push_back(e.dst_instr);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    std::vector<uint64_t> noncancelled_predecessors(uint64_t dst) const {
        std::vector<uint64_t> out;
        for (const auto& kv : edges) {
            const MkOracleEdge& e = kv.second;
            if (e.dst_instr != dst) continue;
            if (!is_noncancelled(e.src_instr)) continue;
            out.push_back(e.src_instr);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    // Detect a cycle if we (hypothetically) added edge src->dst, considering
    // only noncancelled instructions and currently-alive edges. A cycle exists
    // iff src is reachable from dst in the current noncancelled graph.
    bool creates_cycle(uint64_t src, uint64_t dst) const {
        if (!is_noncancelled(src) || !is_noncancelled(dst)) return false;
        // reachable from dst -> ... -> src ?
        std::vector<uint64_t> stack;
        std::map<uint64_t, bool> seen;
        stack.push_back(dst);
        seen[dst] = true;
        while (!stack.empty()) {
            uint64_t cur = stack.back();
            stack.pop_back();
            if (cur == src) return true;
            std::vector<uint64_t> succ = noncancelled_successors(cur);
            for (uint64_t s : succ) {
                if (!seen[s]) { seen[s] = true; stack.push_back(s); }
            }
        }
        return false;
    }

    // Recompute critical_height for all noncancelled instructions.
    // critical_height(i) = duration(i) + max over noncancelled successors j of
    // critical_height(j). Sinks (no noncancelled successors) -> duration(i).
    // Computed via memoized DFS (graph is acyclic by construction). Wraps mod
    // 2^64 (sums are unsigned).
    void recompute_critical_heights() {
        std::map<uint64_t, uint64_t> memo;
        std::map<uint64_t, int> color;  // 0=white,1=grey,2=black
        // iterative post-order to avoid recursion depth issues.
        for (const auto& kv : instrs) {
            uint64_t root = kv.first;
            if (kv.second.status == MK_ST_CANCELLED) continue;
            if (color.count(root) && color[root] == 2) continue;
            std::vector<uint64_t> stack;
            stack.push_back(root);
            while (!stack.empty()) {
                uint64_t cur = stack.back();
                if (!color.count(cur)) color[cur] = 0;
                if (color[cur] == 0) {
                    color[cur] = 1;
                    std::vector<uint64_t> succ = noncancelled_successors(cur);
                    for (uint64_t s : succ) {
                        if (!color.count(s) || color[s] == 0) stack.push_back(s);
                    }
                } else if (color[cur] == 1) {
                    color[cur] = 2;
                    uint64_t best = 0;
                    std::vector<uint64_t> succ = noncancelled_successors(cur);
                    for (uint64_t s : succ) {
                        uint64_t ch = memo.count(s) ? memo[s] : 0;
                        if (ch > best) best = ch;
                    }
                    uint64_t dur = instrs[cur].duration;
                    memo[cur] = dur + best;  // wraps mod 2^64
                    stack.pop_back();
                } else {
                    stack.pop_back();
                }
            }
        }
        for (auto& kv : instrs) {
            if (kv.second.status == MK_ST_CANCELLED) {
                kv.second.critical_height = 0;
            } else {
                kv.second.critical_height = memo.count(kv.first) ? memo[kv.first] : kv.second.duration;
            }
        }
    }

    // ---- ready set ----
    // Unplanned instructions whose noncancelled predecessors are all PLANNED or
    // COMMITTED, ordered by critical_height desc, instr_seq asc, instr_id asc.
    bool instr_is_ready(const MkOracleInstr& o) const {
        if (o.status != MK_ST_UNPLANNED) return false;
        std::vector<uint64_t> preds = noncancelled_predecessors(o.instr_id);
        for (uint64_t p : preds) {
            const MkOracleInstr& pi = instrs.at(p);
            if (pi.status != MK_ST_PLANNED && pi.status != MK_ST_COMMITTED) return false;
        }
        return true;
    }

    bool ready_less(const MkOracleInstr& a, const MkOracleInstr& b) const {
        if (a.critical_height != b.critical_height) return a.critical_height > b.critical_height;
        if (a.instr_seq != b.instr_seq) return a.instr_seq < b.instr_seq;
        return a.instr_id < b.instr_id;
    }

    // Pop the best ready instruction id; 0 sentinel meaning "none" is unsafe if
    // an instr has id 0, so return via found flag.
    bool pop_best_ready(uint64_t* out_id) const {
        const MkOracleInstr* best = nullptr;
        for (const auto& kv : instrs) {
            const MkOracleInstr& o = kv.second;
            if (!instr_is_ready(o)) continue;
            if (best == nullptr || ready_less(o, *best)) best = &o;
        }
        if (best == nullptr) return false;
        *out_id = best->instr_id;
        return true;
    }

    // ---- placement ----
    // The set of page live intervals on a given SM, derived from plans. Each
    // plan contributes, for each of its page keys, an interval
    // [start_tick, release_tick) with that key on plan.sm.
    // Page feasibility for a candidate [start, finish+release_delay) with the
    // request's page_keys on `sm`: at every interval boundary, the number of
    // DISTINCT live page keys must be <= pages_per_sm, where "live" means an
    // existing interval covering that boundary plus the candidate's own keys.
    //
    // We enumerate boundaries = all interval start/end points that fall in
    // [cand_start, cand_end) (cand_end = finish + release_delay), plus
    // cand_start itself. At each boundary tick t we count distinct keys that
    // are live at t: existing intervals [s,e) with s<=t<e, unioned with the
    // candidate's own keys (candidate is live across the whole window). If the
    // distinct count ever exceeds pages_per_sm, infeasible.
    bool page_feasible(uint32_t sm, uint64_t cand_start, uint64_t cand_end,
                       const uint64_t* req_keys, uint64_t req_count,
                       uint64_t* added_pages_out, uint64_t* keysum_out) const {
        if (cand_end <= cand_start) {
            // zero-length window cannot happen (duration>0), but be safe.
            *added_pages_out = req_count;
            uint64_t s = 0; for (uint64_t i = 0; i < req_count; ++i) s += req_keys[i];
            *keysum_out = s;
            return req_count <= pages_per_sm;
        }
        // collect existing intervals on this SM that overlap the window.
        struct Iv { uint64_t s, e, key; };
        std::vector<Iv> ivs;
        for (const auto& p : plans) {
            if (p.sm != sm) continue;
            uint64_t s = p.start_tick, e = p.release_tick;
            if (e <= cand_start || s >= cand_end) continue;  // no overlap with window
            for (uint64_t k = 0; k < p.page_count; ++k) {
                ivs.push_back(Iv{s, e, p.page_keys[k]});
            }
        }
        // candidate boundaries: cand_start, plus every existing s/e in
        // (cand_start, cand_end).
        std::vector<uint64_t> bnds;
        bnds.push_back(cand_start);
        for (const Iv& iv : ivs) {
            if (iv.s > cand_start && iv.s < cand_end) bnds.push_back(iv.s);
            if (iv.e > cand_start && iv.e < cand_end) bnds.push_back(iv.e);
        }
        std::sort(bnds.begin(), bnds.end());
        bnds.erase(std::unique(bnds.begin(), bnds.end()), bnds.end());

        for (uint64_t t : bnds) {
            // distinct live keys at t: candidate keys (always live) plus
            // existing intervals covering t.
            std::vector<uint64_t> keys;
            for (uint64_t i = 0; i < req_count; ++i) keys.push_back(req_keys[i]);
            for (const Iv& iv : ivs) {
                if (iv.s <= t && t < iv.e) keys.push_back(iv.key);
            }
            std::sort(keys.begin(), keys.end());
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            if ((uint64_t)keys.size() > pages_per_sm) return false;
        }

        // added_pages: number of request keys that are NOT already resident on
        // this SM at cand_start without conflict. "Already resident" = an
        // existing interval with the same key covers cand_start. Tie-break
        // metric only; defined deterministically.
        uint64_t added = 0, ksum = 0;
        for (uint64_t i = 0; i < req_count; ++i) {
            uint64_t key = req_keys[i];
            ksum += key;
            bool resident = false;
            for (const Iv& iv : ivs) {
                if (iv.key == key && iv.s <= cand_start && cand_start < iv.e) { resident = true; break; }
            }
            // also dedup within the request itself: a duplicate key in the
            // request only counts once toward "added".
            bool earlier_dup = false;
            for (uint64_t j = 0; j < i; ++j) if (req_keys[j] == key) { earlier_dup = true; break; }
            if (!resident && !earlier_dup) added += 1;
        }
        *added_pages_out = added;
        *keysum_out = ksum;
        return true;
    }

    // dep_ready_tick = max finish_tick of planned/committed predecessors, 0 if
    // none. (Predecessors that are not yet planned/committed cannot exist for a
    // ready instruction.)
    uint64_t dep_ready_tick(uint64_t instr_id) const {
        uint64_t best = 0;
        std::vector<uint64_t> preds = noncancelled_predecessors(instr_id);
        for (uint64_t p : preds) {
            // find that predecessor's plan interval finish_tick.
            for (const auto& pl : plans) {
                if (pl.instr_id == p) {
                    if (pl.finish_tick > best) best = pl.finish_tick;
                }
            }
        }
        return best;
    }

    // PLAN one instruction. Returns: 0=placed, 1=stall.
    int place_instr(uint64_t instr_id, uint64_t seq, uint32_t opidx) {
        MkOracleInstr& o = instrs[instr_id];
        uint64_t drt = dep_ready_tick(instr_id);

        bool have_best = false;
        uint32_t best_sm = 0;
        uint64_t best_start = 0, best_finish = 0, best_wave = 0;
        uint64_t best_added = 0, best_ksum = 0;

        for (uint32_t sm = 0; sm < sm_count; ++sm) {
            // earliest wave-quantized start >= drt
            uint64_t wave;
            uint64_t start;
            if (drt == 0) { wave = 0; start = 0; }
            else {
                wave = (drt + wave_quantum - 1) / wave_quantum;  // ceil
                start = wave * wave_quantum;
            }
            // try increasing waves until feasible (bounded scan).
            const int MAX_WAVE_SCAN = 4096;
            bool placed_here = false;
            for (int it = 0; it < MAX_WAVE_SCAN; ++it) {
                uint64_t finish = start + o.duration;          // wraps mod 2^64
                uint64_t cand_end = finish + o.release_delay;  // wraps mod 2^64
                uint64_t added = 0, ksum = 0;
                if (page_feasible(sm, start, cand_end, o.page_keys, o.page_count, &added, &ksum)) {
                    // candidate (sm, start) is feasible.
                    bool better;
                    if (!have_best) better = true;
                    else if (finish != best_finish) better = finish < best_finish;
                    else if (start != best_start) better = start < best_start;
                    else if (sm != best_sm) better = sm < best_sm;
                    else if (added != best_added) better = added < best_added;
                    else better = ksum < best_ksum;
                    if (better) {
                        have_best = true; best_sm = sm; best_start = start;
                        best_finish = finish; best_wave = wave; best_added = added; best_ksum = ksum;
                    }
                    placed_here = true;
                    break;
                }
                wave += 1;
                start = wave * wave_quantum;
            }
            (void)placed_here;
        }

        if (!have_best) {
            acc.plan_stall += 1;
            emit_event(MK_EK_PLAN_STALL, seq, opidx, instr_id, 0, MK_U32_MAX,
                       MK_U64_MAX, MK_U64_MAX, 0);
            return 1;
        }

        MkOraclePlan pl;
        pl.plan_seq = plan_seq_next++;
        pl.instr_id = instr_id;
        pl.sm = best_sm;
        pl.start_wave = best_wave;
        pl.start_tick = best_start;
        pl.finish_tick = best_finish;
        pl.release_tick = best_finish + o.release_delay;  // wraps mod 2^64
        pl.page_count = o.page_count;
        for (uint64_t k = 0; k < o.page_count; ++k) pl.page_keys[k] = o.page_keys[k];
        plans.push_back(pl);
        o.status = MK_ST_PLANNED;

        acc.plan_placed += 1;
        emit_event(MK_EK_PLAN_PLACE, seq, opidx, instr_id, 0, best_sm,
                   best_start, best_finish, pl.plan_seq);
        return 0;
    }

    // Mark dst and all its planned (uncommitted) descendants back to UNPLANNED,
    // removing their plan intervals and emitting PLAN_INVALIDATE per removed
    // interval in reverse plan-sequence order. Committed intervals are never
    // removed (they form a barrier; descendants reachable only through a
    // committed node are not invalidated unless independently planned-reachable).
    void invalidate_from(uint64_t dst, uint64_t seq, uint32_t opidx) {
        // Collect the set of instructions to invalidate: dst plus all
        // descendants reachable through edges, that are currently PLANNED
        // (not COMMITTED, not EXECUTED). Traversal does not pass THROUGH a
        // committed node (a committed node's plan is immovable), but a planned
        // node reached by any path is invalidated.
        std::map<uint64_t, bool> mark;
        std::vector<uint64_t> stack;
        stack.push_back(dst);
        while (!stack.empty()) {
            uint64_t cur = stack.back();
            stack.pop_back();
            auto it = instrs.find(cur);
            if (it == instrs.end()) continue;
            if (it->second.status != MK_ST_PLANNED) {
                // Only planned nodes are invalidated. We still may traverse
                // through dst even if dst itself is the seed; but if dst is not
                // planned there is nothing to remove from it. Do not traverse
                // through committed/executed nodes.
                if (cur != dst) continue;
                // dst not planned: still traverse its successors? The contract
                // says "mark the dst and all its planned descendants". A non-
                // planned dst contributes nothing and we do not traverse past a
                // non-planned node.
                continue;
            }
            if (mark.count(cur)) continue;
            mark[cur] = true;
            std::vector<uint64_t> succ = noncancelled_successors(cur);
            for (uint64_t s : succ) stack.push_back(s);
        }
        if (mark.empty()) return;

        // Remove plan intervals for marked instrs in reverse plan-sequence
        // order; emit PLAN_INVALIDATE each.
        std::vector<MkOraclePlan> to_remove;
        for (const auto& p : plans) {
            if (mark.count(p.instr_id)) to_remove.push_back(p);
        }
        std::sort(to_remove.begin(), to_remove.end(),
                  [](const MkOraclePlan& a, const MkOraclePlan& b) {
                      return a.plan_seq > b.plan_seq;  // descending
                  });
        for (const MkOraclePlan& p : to_remove) {
            acc.plan_invalidated += 1;
            emit_event(MK_EK_PLAN_INVALIDATE, seq, opidx, p.instr_id, 0, p.sm,
                       p.start_tick, p.finish_tick, p.plan_seq);
        }
        // physically remove and reset status
        std::vector<MkOraclePlan> kept;
        for (const auto& p : plans) {
            if (mark.count(p.instr_id)) continue;
            kept.push_back(p);
        }
        plans.swap(kept);
        for (auto& kv : mark) {
            instrs[kv.first].status = MK_ST_UNPLANNED;
        }
    }

    // ---- per-op handlers ----
    void op_add_instr(const MkOp& op, uint64_t seq, uint32_t opidx) {
        uint64_t id = op.id;
        uint64_t duration = op.a;
        uint64_t page_count = op.b;
        uint64_t release_delay = op.c;
        uint64_t payload_seed = op.d;
        if (instrs.count(id) != 0 ||
            instrs.size() >= (size_t)spec.max_instrs ||
            duration == 0 || page_count == 0 || page_count > pages_per_sm) {
            acc.invalid_count += 1;
            emit_event(MK_EK_INVALID, seq, opidx, id, 0, MK_U32_MAX,
                       MK_U64_MAX, MK_U64_MAX, 0);
            return;
        }
        MkOracleInstr o;
        o.instr_id = id;
        o.instr_seq = instr_seq_next++;
        o.duration = duration;
        o.page_count = page_count;
        for (uint64_t k = 0; k < MK_MAX_PAGE_KEYS_PER_INSTR; ++k)
            o.page_keys[k] = (k < page_count) ? op.page_keys[k] : 0;
        o.release_delay = release_delay;
        o.payload_seed = payload_seed;
        o.status = MK_ST_UNPLANNED;
        o.critical_height = duration;
        o.completion_counter = 0;
        instrs[id] = o;

        recompute_critical_heights();
        acc.instr_added += 1;
        emit_event(MK_EK_INSTR_ADD, seq, opidx, id, 0, MK_U32_MAX,
                   MK_U64_MAX, MK_U64_MAX, o.instr_seq);
    }

    void op_add_edge(const MkOp& op, uint64_t seq, uint32_t opidx) {
        uint64_t eid = op.id;
        uint64_t src = op.src;
        uint64_t dst = op.dst;
        uint32_t chunk_id = (uint32_t)op.a;
        uint64_t target_increment = op.b;

        bool invalid = false;
        if (edges.count(eid) != 0) invalid = true;
        else if (instrs.count(src) == 0 || instrs.count(dst) == 0) invalid = true;
        else if (src == dst) invalid = true;
        else if (target_increment == 0) invalid = true;
        else if (edges.size() >= (size_t)spec.max_edges) invalid = true;
        // adding an edge to a committed dst is invalid.
        else if (instrs[dst].status == MK_ST_COMMITTED || instrs[dst].status == MK_ST_EXECUTED) invalid = true;
        // src/dst cancelled -> absent semantics: a cancelled instr is treated
        // as not a valid endpoint.
        else if (instrs[src].status == MK_ST_CANCELLED || instrs[dst].status == MK_ST_CANCELLED) invalid = true;
        else if (creates_cycle(src, dst)) invalid = true;

        if (invalid) {
            acc.invalid_count += 1;
            emit_event(MK_EK_INVALID, seq, opidx, 0, eid, MK_U32_MAX,
                       MK_U64_MAX, MK_U64_MAX, 0);
            return;
        }

        MkOracleEdge e;
        e.edge_id = eid;
        e.src_instr = src;
        e.dst_instr = dst;
        e.chunk_id = chunk_id;
        e.target_increment = target_increment;
        e.edge_seq = edge_seq_next++;
        edges[eid] = e;

        // If either endpoint was planned but not committed, mark dst and its
        // planned descendants back to UNPLANNED. (src being planned does not by
        // itself invalidate, but if src was planned and dst was planned the new
        // dependency means dst's placement may now be wrong, so invalidate dst
        // subtree. The contract: "If either endpoint was planned but not
        // committed, mark the dst and all its planned descendants back to
        // UNPLANNED".)
        bool src_planned = (instrs[src].status == MK_ST_PLANNED);
        bool dst_planned = (instrs[dst].status == MK_ST_PLANNED);
        if (src_planned || dst_planned) {
            invalidate_from(dst, seq, opidx);
        }

        recompute_critical_heights();
        acc.edge_added += 1;
        emit_event(MK_EK_EDGE_ADD, seq, opidx, 0, eid, MK_U32_MAX,
                   MK_U64_MAX, MK_U64_MAX, e.edge_seq);
    }

    void op_plan_next(const MkOp& op, uint64_t seq, uint32_t opidx) {
        uint64_t limit = op.a;
        if (limit == 0) return;  // valid no-op
        for (uint64_t n = 0; n < limit; ++n) {
            uint64_t best_id = 0;
            if (!pop_best_ready(&best_id)) {
                acc.plan_empty += 1;
                emit_event(MK_EK_PLAN_EMPTY, seq, opidx, 0, 0, MK_U32_MAX,
                           MK_U64_MAX, MK_U64_MAX, 0);
                return;
            }
            int r = place_instr(best_id, seq, opidx);
            if (r == 1) return;  // stall stops the whole PLAN_NEXT
        }
    }

    void op_commit_plan(const MkOp& op, uint64_t seq, uint32_t opidx) {
        uint64_t max_entries = op.a;
        if (max_entries == 0) return;  // valid no-op
        // commit planned entries by plan_seq ascending, up to max.
        std::vector<const MkOraclePlan*> sorted;
        for (const auto& p : plans) {
            if (instrs[p.instr_id].status == MK_ST_PLANNED) sorted.push_back(&p);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const MkOraclePlan* a, const MkOraclePlan* b) {
                      return a->plan_seq < b->plan_seq;
                  });
        uint64_t done = 0;
        for (const MkOraclePlan* p : sorted) {
            if (done >= max_entries) break;
            instrs[p->instr_id].status = MK_ST_COMMITTED;
            acc.plan_committed += 1;
            emit_event(MK_EK_PLAN_COMMIT, seq, opidx, p->instr_id, 0, p->sm,
                       p->start_tick, p->finish_tick, p->plan_seq);
            done += 1;
        }
    }

    void op_execute_until(const MkOp& op, uint64_t seq, uint32_t opidx) {
        uint64_t tick_limit = op.a;
        uint64_t max_events = op.b;
        if (max_events == 0) return;  // valid no-op (no events)
        // Execute committed, nonexecuted entries with finish_tick <= tick_limit,
        // ordered by (finish_tick, sm, plan_seq), up to max_events.
        std::vector<const MkOraclePlan*> elig;
        for (const auto& p : plans) {
            const MkOracleInstr& o = instrs[p.instr_id];
            if (o.status != MK_ST_COMMITTED) continue;
            if (p.finish_tick > tick_limit) continue;
            elig.push_back(&p);
        }
        std::sort(elig.begin(), elig.end(),
                  [](const MkOraclePlan* a, const MkOraclePlan* b) {
                      if (a->finish_tick != b->finish_tick) return a->finish_tick < b->finish_tick;
                      if (a->sm != b->sm) return a->sm < b->sm;
                      return a->plan_seq < b->plan_seq;
                  });
        uint64_t done = 0;
        for (const MkOraclePlan* p : elig) {
            if (done >= max_events) break;
            MkOracleInstr& o = instrs[p->instr_id];

            // result = FNV1a64(committed_epoch, instr_id, start_tick,
            //                  finish_tick, sm, payload_seed, page_keys in
            //                  request order).
            uint64_t result = MK_FNV_BASIS;
            mk_oracle_fnv_u64(&result, committed_epoch);
            mk_oracle_fnv_u64(&result, o.instr_id);
            mk_oracle_fnv_u64(&result, p->start_tick);
            mk_oracle_fnv_u64(&result, p->finish_tick);
            mk_oracle_fnv_u32(&result, p->sm);
            mk_oracle_fnv_u64(&result, o.payload_seed);
            for (uint64_t k = 0; k < o.page_count; ++k)
                mk_oracle_fnv_u64(&result, o.page_keys[k]);

            o.completion_counter += 1;
            o.status = MK_ST_EXECUTED;
            acc.instr_executed += 1;
            emit_event(MK_EK_EXEC_INSTR, seq, opidx, o.instr_id, 0, p->sm,
                       p->start_tick, p->finish_tick, result);

            // For each outgoing edge by edge_seq, emit EDGE_SIGNAL.
            std::vector<const MkOracleEdge*> outs;
            for (const auto& kv : edges) {
                if (kv.second.src_instr == o.instr_id) outs.push_back(&kv.second);
            }
            std::sort(outs.begin(), outs.end(),
                      [](const MkOracleEdge* a, const MkOracleEdge* b) {
                          return a->edge_seq < b->edge_seq;
                      });
            for (const MkOracleEdge* e : outs) {
                acc.edge_signaled += 1;
                emit_event(MK_EK_EDGE_SIGNAL, seq, opidx, e->dst_instr, e->edge_id,
                           MK_U32_MAX, MK_U64_MAX, MK_U64_MAX,
                           ((uint64_t)e->chunk_id << 0) ^ (e->target_increment * MK_FNV_PRIME));
            }
            done += 1;
        }
    }

    void op_cancel_instr(const MkOp& op, uint64_t seq, uint32_t opidx) {
        uint64_t id = op.id;
        auto it = instrs.find(id);
        if (it == instrs.end()) { acc.invalid_count += 1;
            emit_event(MK_EK_INVALID, seq, opidx, id, 0, MK_U32_MAX, MK_U64_MAX, MK_U64_MAX, 0);
            return; }
        MkOracleInstr& o = it->second;
        if (o.status == MK_ST_EXECUTED) { acc.invalid_count += 1;
            emit_event(MK_EK_INVALID, seq, opidx, id, 0, MK_U32_MAX, MK_U64_MAX, MK_U64_MAX, 0);
            return; }
        if (o.status == MK_ST_COMMITTED) { acc.invalid_count += 1;
            emit_event(MK_EK_INVALID, seq, opidx, id, 0, MK_U32_MAX, MK_U64_MAX, MK_U64_MAX, 0);
            return; }
        if (o.status == MK_ST_CANCELLED) { acc.invalid_count += 1;
            emit_event(MK_EK_INVALID, seq, opidx, id, 0, MK_U32_MAX, MK_U64_MAX, MK_U64_MAX, 0);
            return; }

        // If planned, remove its plan interval and planned descendants in
        // reverse plan-sequence order, emitting PLAN_INVALIDATE.
        if (o.status == MK_ST_PLANNED) {
            invalidate_from(id, seq, opidx);
        }
        // (invalidate_from set o back to UNPLANNED if it was planned.)

        // Mark CANCELLED, remove incident uncommitted edges.
        // All edges incident to id are uncommitted in this model (committed
        // refers to instructions; edges have no commit state) EXCEPT that an
        // edge whose other endpoint is COMMITTED/EXECUTED cannot be removed if
        // that would alter committed structure. Per the contract, we remove
        // incident uncommitted edges; we treat an edge as committed-locked iff
        // its OTHER endpoint is COMMITTED or EXECUTED.
        std::vector<uint64_t> remove_ids;
        for (const auto& kv : edges) {
            const MkOracleEdge& e = kv.second;
            if (e.src_instr != id && e.dst_instr != id) continue;
            uint64_t other = (e.src_instr == id) ? e.dst_instr : e.src_instr;
            uint8_t ost = instrs.count(other) ? instrs[other].status : MK_ST_UNPLANNED;
            if (ost == MK_ST_COMMITTED || ost == MK_ST_EXECUTED) continue;  // locked
            remove_ids.push_back(e.edge_id);
        }
        for (uint64_t reid : remove_ids) edges.erase(reid);

        instrs[id].status = MK_ST_CANCELLED;
        instrs[id].critical_height = 0;

        recompute_critical_heights();
        acc.instr_cancelled += 1;
        emit_event(MK_EK_INSTR_CANCEL, seq, opidx, id, 0, MK_U32_MAX,
                   MK_U64_MAX, MK_U64_MAX, 0);
    }

    void op_new_epoch(const MkOp& op, uint64_t seq, uint32_t opidx) {
        (void)op;
        // Invalid if any committed instruction is not executed.
        for (const auto& kv : instrs) {
            if (kv.second.status == MK_ST_COMMITTED) {
                acc.invalid_count += 1;
                emit_event(MK_EK_INVALID, seq, opidx, 0, 0, MK_U32_MAX,
                           MK_U64_MAX, MK_U64_MAX, 0);
                return;
            }
        }
        committed_epoch += 1;
        // clear all noncancelled instructions to UNPLANNED; clear plans and
        // counters.
        for (auto& kv : instrs) {
            if (kv.second.status != MK_ST_CANCELLED) {
                kv.second.status = MK_ST_UNPLANNED;
                kv.second.completion_counter = 0;
            }
        }
        plans.clear();
        recompute_critical_heights();
        acc.epoch_advanced += 1;
        emit_event(MK_EK_EPOCH_ADVANCE, seq, opidx, 0, 0, MK_U32_MAX,
                   MK_U64_MAX, MK_U64_MAX, committed_epoch);
    }

    void apply_op(const MkOp& op) {
        event_seq += 1;  // wraps mod 2^64
        uint64_t seq = event_seq;
        uint32_t opidx = op_index;
        op_index += 1;  // wraps mod 2^32
        switch (op.op_type) {
            case MK_OP_ADD_INSTR:     op_add_instr(op, seq, opidx); break;
            case MK_OP_ADD_EDGE:      op_add_edge(op, seq, opidx); break;
            case MK_OP_PLAN_NEXT:     op_plan_next(op, seq, opidx); break;
            case MK_OP_COMMIT_PLAN:   op_commit_plan(op, seq, opidx); break;
            case MK_OP_EXECUTE_UNTIL: op_execute_until(op, seq, opidx); break;
            case MK_OP_CANCEL_INSTR:  op_cancel_instr(op, seq, opidx); break;
            case MK_OP_NEW_EPOCH:     op_new_epoch(op, seq, opidx); break;
            default:
                acc.invalid_count += 1;
                emit_event(MK_EK_INVALID, seq, opidx, 0, 0, MK_U32_MAX,
                           MK_U64_MAX, MK_U64_MAX, 0);
                break;
        }
    }

    // ---- structural hashes ----
    // plan_hash: planned/committed intervals by SM, then start tick, then plan
    // sequence:
    //   sm:u32; plan_seq:u64; instr_id:u64; start_wave:u64; start_tick:u64;
    //   finish_tick:u64; release_tick:u64; then page keys in request order.
    uint64_t compute_plan_hash() const {
        std::vector<const MkOraclePlan*> v;
        for (const auto& p : plans) v.push_back(&p);
        std::sort(v.begin(), v.end(), [](const MkOraclePlan* a, const MkOraclePlan* b) {
            if (a->sm != b->sm) return a->sm < b->sm;
            if (a->start_tick != b->start_tick) return a->start_tick < b->start_tick;
            return a->plan_seq < b->plan_seq;
        });
        uint64_t h = MK_FNV_BASIS;
        for (const MkOraclePlan* p : v) {
            mk_oracle_fnv_u32(&h, p->sm);
            mk_oracle_fnv_u64(&h, p->plan_seq);
            mk_oracle_fnv_u64(&h, p->instr_id);
            mk_oracle_fnv_u64(&h, p->start_wave);
            mk_oracle_fnv_u64(&h, p->start_tick);
            mk_oracle_fnv_u64(&h, p->finish_tick);
            mk_oracle_fnv_u64(&h, p->release_tick);
            for (uint64_t k = 0; k < p->page_count; ++k)
                mk_oracle_fnv_u64(&h, p->page_keys[k]);
        }
        return h;
    }

    // instr_hash: instructions by instr id:
    //   instr_id:u64; instr_seq:u64; duration:u64; release_delay:u64;
    //   payload_seed:u64; status:u8; critical_height:u64; completion_counter:u64.
    uint64_t compute_instr_hash() const {
        uint64_t h = MK_FNV_BASIS;
        for (const auto& kv : instrs) {  // map iterates instr_id ascending
            const MkOracleInstr& o = kv.second;
            mk_oracle_fnv_u64(&h, o.instr_id);
            mk_oracle_fnv_u64(&h, o.instr_seq);
            mk_oracle_fnv_u64(&h, o.duration);
            mk_oracle_fnv_u64(&h, o.release_delay);
            mk_oracle_fnv_u64(&h, o.payload_seed);
            mk_oracle_fnv_u8(&h, o.status);
            mk_oracle_fnv_u64(&h, o.critical_height);
            mk_oracle_fnv_u64(&h, o.completion_counter);
        }
        return h;
    }

    // edge_hash: edges by edge id:
    //   edge_id:u64; src_instr:u64; dst_instr:u64; chunk_id:u32;
    //   target_increment:u64; edge_seq:u64.
    uint64_t compute_edge_hash() const {
        uint64_t h = MK_FNV_BASIS;
        for (const auto& kv : edges) {  // map iterates edge_id ascending
            const MkOracleEdge& e = kv.second;
            mk_oracle_fnv_u64(&h, e.edge_id);
            mk_oracle_fnv_u64(&h, e.src_instr);
            mk_oracle_fnv_u64(&h, e.dst_instr);
            mk_oracle_fnv_u32(&h, e.chunk_id);
            mk_oracle_fnv_u64(&h, e.target_increment);
            mk_oracle_fnv_u64(&h, e.edge_seq);
        }
        return h;
    }

    // page_interval_hash: live page intervals by SM, page key, start tick,
    // instr id. Each plan contributes one interval per page key:
    //   sm:u32; page_key:u64; start_tick:u64; release_tick:u64; instr_id:u64.
    uint64_t compute_page_interval_hash() const {
        struct Piv { uint32_t sm; uint64_t key; uint64_t start; uint64_t release; uint64_t instr_id; uint64_t kidx; };
        std::vector<Piv> v;
        for (const auto& p : plans) {
            for (uint64_t k = 0; k < p.page_count; ++k) {
                v.push_back(Piv{p.sm, p.page_keys[k], p.start_tick, p.release_tick, p.instr_id, k});
            }
        }
        // Total order; the per-plan key index `kidx` is the final tie-break so
        // that a plan with duplicate page keys yields a stable, well-defined
        // emission order (entries are still hashed without kidx).
        std::sort(v.begin(), v.end(), [](const Piv& a, const Piv& b) {
            if (a.sm != b.sm) return a.sm < b.sm;
            if (a.key != b.key) return a.key < b.key;
            if (a.start != b.start) return a.start < b.start;
            if (a.instr_id != b.instr_id) return a.instr_id < b.instr_id;
            return a.kidx < b.kidx;
        });
        uint64_t h = MK_FNV_BASIS;
        for (const Piv& p : v) {
            mk_oracle_fnv_u32(&h, p.sm);
            mk_oracle_fnv_u64(&h, p.key);
            mk_oracle_fnv_u64(&h, p.start);
            mk_oracle_fnv_u64(&h, p.release);
            mk_oracle_fnv_u64(&h, p.instr_id);
        }
        return h;
    }

    uint64_t count_live_instrs() const {
        uint64_t c = 0;
        for (const auto& kv : instrs)
            if (kv.second.status != MK_ST_CANCELLED) c += 1;
        return c;
    }

    void step_once(const MkRunSpec& run, const MkOp* ops, MkExpected* expected) {
        for (int i = 0; i < run.num_ops; ++i) {
            apply_op(ops[i]);
        }
        *expected = acc;
        expected->plan_hash = compute_plan_hash();
        expected->instr_hash = compute_instr_hash();
        expected->edge_hash = compute_edge_hash();
        expected->page_interval_hash = compute_page_interval_hash();
        expected->committed_epoch = committed_epoch;
        expected->live_instr_count = count_live_instrs();
        expected->planned_interval_count = (uint64_t)plans.size();
    }
};

// ---------------------------------------------------------------------------
// Output comparison.
// ---------------------------------------------------------------------------
static inline bool mk_check_all_outputs(
    const MkExpected& e,
    const MkHostOutputsView& g,
    std::string* err) {
#define MK_CHECK(field) \
    if (g.field[0] != e.field) { \
        if (err) { std::ostringstream o; o << #field " mismatch: got " \
            << g.field[0] << " expected " << e.field; *err = o.str(); } \
        return false; }
#define MK_CHECK_HEX(field) \
    if (g.field[0] != e.field) { \
        if (err) { std::ostringstream o; o << #field " mismatch: got 0x" \
            << std::hex << g.field[0] << " expected 0x" << e.field; *err = o.str(); } \
        return false; }

    MK_CHECK(instr_added);
    MK_CHECK(edge_added);
    MK_CHECK(plan_invalidated);
    MK_CHECK(plan_empty);
    MK_CHECK(plan_stall);
    MK_CHECK(plan_placed);
    MK_CHECK(plan_committed);
    MK_CHECK(instr_executed);
    MK_CHECK(edge_signaled);
    MK_CHECK(instr_cancelled);
    MK_CHECK(epoch_advanced);
    MK_CHECK(invalid_count);
    MK_CHECK_HEX(planner_event_hash);
    MK_CHECK_HEX(plan_hash);
    MK_CHECK_HEX(instr_hash);
    MK_CHECK_HEX(edge_hash);
    MK_CHECK_HEX(page_interval_hash);
    MK_CHECK(committed_epoch);
    MK_CHECK(live_instr_count);
    MK_CHECK(planned_interval_count);
#undef MK_CHECK
#undef MK_CHECK_HEX
    return true;
}

/*
GRADER MODEL

  oracle.init(problem_spec)
  solution_reset + oracle.reset()
  for each step:
    solution_run(...)
    oracle.step_once(...)
    mk_check_all_outputs(...)

Required harness coverage:
  - ADD_INSTR with validity (dup id, table full, zero duration, zero/oversized
    page_count)
  - ADD_EDGE with cycle rejection + planned-endpoint invalidation cascade
  - PLAN_NEXT with wave quantization, page live-interval constraints, makespan
    tie-breaks, PLAN_EMPTY, PLAN_STALL
  - COMMIT_PLAN then ADD_EDGE-to-committed-dst rejected
  - EXECUTE_UNTIL ordering + EDGE_SIGNAL fan-out + completion counters
  - CANCEL_INSTR removing plan subtree + incident edges
  - NEW_EPOCH guarded by committed-not-executed
  - num_ops = 0 step
  - reset and replay
*/

#endif  // MK_SCHEDULE_PLANNER_ORACLE_HPP_
