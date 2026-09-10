// file: switch_moe_overflow_router_oracle.hpp
//
// Independent CPU std:: reference (the "oracle"). This file owns the exact
// authoritative semantics of the T40 contract. It shares NO algorithm code
// with reference.cu or naive.cu.

#ifndef SWITCH_MOE_OVERFLOW_ROUTER_ORACLE_HPP_
#define SWITCH_MOE_OVERFLOW_ROUTER_ORACLE_HPP_

#include "switch_moe_overflow_router_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <deque>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct SmorHostInputsView {
    const int32_t*  op_kind;
    const uint64_t* op_a;
    const uint64_t* op_b;
    const int32_t*  op_cand_off;
    const int32_t*  op_cand_count;
    const int32_t*  cand_expert;
    const int32_t*  cand_logit;
    const int32_t*  cand_ordinal;
};

struct SmorHostOutputsView {
    const uint64_t* refill_count;
    const uint64_t* accepted_primary;
    const uint64_t* accepted_secondary;
    const uint64_t* queued;
    const uint64_t* replayed_primary;
    const uint64_t* replayed_secondary;
    const uint64_t* capacity_drop;
    const uint64_t* oom_drop;
    const uint64_t* duplicate_count;
    const uint64_t* retired_live;
    const uint64_t* retired_queued;
    const uint64_t* queue_drop;
    const uint64_t* invalid_count;
    const uint64_t* route_event_hash;
    const uint64_t* credit_hash;
    const uint64_t* assignment_hash;
    const uint64_t* overflow_hash;
};

struct SmorExpected {
    uint64_t refill_count = 0;
    uint64_t accepted_primary = 0;
    uint64_t accepted_secondary = 0;
    uint64_t queued = 0;
    uint64_t replayed_primary = 0;
    uint64_t replayed_secondary = 0;
    uint64_t capacity_drop = 0;
    uint64_t oom_drop = 0;
    uint64_t duplicate_count = 0;
    uint64_t retired_live = 0;
    uint64_t retired_queued = 0;
    uint64_t queue_drop = 0;
    uint64_t invalid_count = 0;
    uint64_t route_event_hash = SMOR_FNV_OFFSET;
    uint64_t credit_hash = SMOR_FNV_OFFSET;
    uint64_t assignment_hash = SMOR_FNV_OFFSET;
    uint64_t overflow_hash = SMOR_FNV_OFFSET;
};

static inline uint64_t smor_o_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= SMOR_FNV_PRIME;
    return h;
}

static inline void smor_o_fnv(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = smor_o_fnv_byte(v, b[i]);
    *h = v;
}

static inline uint64_t smor_o_sat_add_u64(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a) return SMOR_U64_MAX;  // overflow -> saturate
    return s;
}

struct SmorOToken {
    uint64_t token_id = 0;
    int32_t  status = SMOR_STATUS_FREE;
    uint32_t primary = 0;
    uint32_t secondary = 0;
    uint64_t cost = 0;
    uint64_t arrival_seq = 0;       // QUEUED meaningful; carried over for LIVE
    uint64_t admit_seq = 0;         // LIVE only
    uint32_t assigned_expert = 0;   // LIVE only
    uint8_t  route_kind = 0;        // LIVE only
};

struct SmorOracleState {
    SmorProblemSpec spec{};
    std::vector<uint64_t> credit_cap;
    std::vector<uint64_t> initial_credit;

    uint64_t step_seq = 0;
    uint64_t event_seq = 0;

    std::vector<uint64_t> credit;     // E
    std::vector<uint64_t> live_count; // E

    // token table keyed by token_id
    std::map<uint64_t, SmorOToken> table;
    // overflow FIFO: token_ids head->tail
    std::deque<uint64_t> fifo;

    int32_t live_total = 0;  // number of LIVE tokens (== sum live_count)

    void init(const SmorProblemSpec& s,
              const std::vector<uint64_t>& cap,
              const std::vector<uint64_t>& init_cred) {
        spec = s;
        credit_cap = cap;
        initial_credit = init_cred;
        credit.assign((size_t)spec.num_experts, 0);
        live_count.assign((size_t)spec.num_experts, 0);
        reset();
    }

    void reset() {
        step_seq = 0;
        event_seq = 0;
        for (int e = 0; e < spec.num_experts; ++e) {
            credit[(size_t)e] = initial_credit[(size_t)e];
            live_count[(size_t)e] = 0;
        }
        table.clear();
        fifo.clear();
        live_total = 0;
    }

    // ---- event emission ------------------------------------------------
    // Each record (contract section 4) hashed in exact emission order.
    void emit(SmorExpected* ex,
              uint8_t event_kind,
              uint32_t op_index,
              uint64_t token_id,
              uint32_t expert_or_max,
              uint32_t primary,
              uint32_t secondary,
              uint64_t cost,
              uint64_t credit_after_or_max,
              uint64_t arrival_seq_or_max) {
        uint64_t es = event_seq;
        uint64_t* h = &ex->route_event_hash;
        smor_o_fnv(h, &event_kind, sizeof(uint8_t));
        smor_o_fnv(h, &es, sizeof(uint64_t));
        smor_o_fnv(h, &op_index, sizeof(uint32_t));
        smor_o_fnv(h, &token_id, sizeof(uint64_t));
        smor_o_fnv(h, &expert_or_max, sizeof(uint32_t));
        smor_o_fnv(h, &primary, sizeof(uint32_t));
        smor_o_fnv(h, &secondary, sizeof(uint32_t));
        smor_o_fnv(h, &cost, sizeof(uint64_t));
        smor_o_fnv(h, &credit_after_or_max, sizeof(uint64_t));
        smor_o_fnv(h, &arrival_seq_or_max, sizeof(uint64_t));
        event_seq += 1;  // wraps mod 2^64
    }

    // ---- candidate resolution -----------------------------------------
    // Returns false if invalid (no valid candidate). Otherwise fills primary,
    // secondary.
    bool resolve_candidates(const SmorHostInputsView& in,
                            int op_index,
                            uint32_t* primary,
                            uint32_t* secondary) {
        const int off = in.op_cand_off[op_index];
        const int cnt = in.op_cand_count[op_index];

        // collapse repeated expert_id: keep highest logit, tie -> smallest ordinal
        struct Cand { int32_t expert; int32_t logit; int32_t ordinal; };
        std::vector<Cand> uniq;
        for (int i = 0; i < cnt; ++i) {
            int32_t e = in.cand_expert[off + i];
            int32_t lg = in.cand_logit[off + i];
            int32_t ord = in.cand_ordinal[off + i];
            if (e < 0 || e >= spec.num_experts) continue;  // invalid expert ignored
            bool found = false;
            for (size_t j = 0; j < uniq.size(); ++j) {
                if (uniq[j].expert == e) {
                    found = true;
                    if (lg > uniq[j].logit ||
                        (lg == uniq[j].logit && ord < uniq[j].ordinal)) {
                        uniq[j].logit = lg;
                        uniq[j].ordinal = ord;
                    }
                    break;
                }
            }
            if (!found) uniq.push_back(Cand{e, lg, ord});
        }

        if (uniq.empty()) return false;

        // sort by descending logit, then ascending expert_id
        std::sort(uniq.begin(), uniq.end(), [](const Cand& a, const Cand& b) {
            if (a.logit != b.logit) return a.logit > b.logit;
            return a.expert < b.expert;
        });

        *primary = (uint32_t)uniq[0].expert;
        *secondary = (uniq.size() >= 2) ? (uint32_t)uniq[1].expert : (uint32_t)uniq[0].expert;
        return true;
    }

    // Try to choose an admission expert given primary/secondary and cost.
    // Returns -1 if neither has enough credit; else expert id; sets is_primary.
    int choose_admit(uint32_t primary, uint32_t secondary, uint64_t cost, bool* is_primary) {
        if (credit[primary] >= cost) { *is_primary = true; return (int)primary; }
        if (secondary != primary && credit[secondary] >= cost) {
            *is_primary = false; return (int)secondary;
        }
        return -1;
    }

    // ---- per-op handlers ----------------------------------------------
    void do_refill(SmorExpected* ex, int op_index, const SmorHostInputsView& in) {
        uint64_t expert = in.op_a[op_index];
        uint64_t amount = in.op_b[op_index];
        if (expert >= (uint64_t)spec.num_experts) {
            ex->invalid_count += 1;
            emit(ex, SMOR_EV_INVALID, (uint32_t)op_index, /*token*/0,
                 SMOR_U32_MAX, SMOR_U32_MAX, SMOR_U32_MAX, /*cost*/0,
                 SMOR_U64_MAX, SMOR_U64_MAX);
            return;
        }
        uint64_t nc = smor_o_sat_add_u64(credit[expert], amount);
        if (nc > credit_cap[expert]) nc = credit_cap[expert];
        credit[expert] = nc;
        ex->refill_count += 1;
        emit(ex, SMOR_EV_REFILL, (uint32_t)op_index, /*token*/0,
             (uint32_t)expert, SMOR_U32_MAX, SMOR_U32_MAX, /*cost*/0,
             /*credit_after*/nc, SMOR_U64_MAX);
    }

    void do_route(SmorExpected* ex, int op_index, const SmorHostInputsView& in) {
        uint64_t token_id = in.op_a[op_index];
        uint64_t cost = in.op_b[op_index];
        int cand_count = in.op_cand_count[op_index];

        if (cost == 0 || cand_count == 0 ||
            cand_count > spec.max_candidates_per_route) {
            ex->invalid_count += 1;
            emit(ex, SMOR_EV_INVALID, (uint32_t)op_index, token_id,
                 SMOR_U32_MAX, SMOR_U32_MAX, SMOR_U32_MAX, cost,
                 SMOR_U64_MAX, SMOR_U64_MAX);
            return;
        }

        uint32_t primary = 0, secondary = 0;
        if (!resolve_candidates(in, op_index, &primary, &secondary)) {
            ex->invalid_count += 1;
            emit(ex, SMOR_EV_INVALID, (uint32_t)op_index, token_id,
                 SMOR_U32_MAX, SMOR_U32_MAX, SMOR_U32_MAX, cost,
                 SMOR_U64_MAX, SMOR_U64_MAX);
            return;
        }

        // duplicate check first
        if (table.find(token_id) != table.end()) {
            ex->duplicate_count += 1;
            emit(ex, SMOR_EV_DUPLICATE, (uint32_t)op_index, token_id,
                 SMOR_U32_MAX, primary, secondary, cost,
                 SMOR_U64_MAX, SMOR_U64_MAX);
            return;
        }

        bool is_primary = false;
        int chosen = choose_admit(primary, secondary, cost, &is_primary);

        if (chosen >= 0) {
            // admission would succeed; storage-capacity check
            if (live_total == spec.max_live_tokens) {
                ex->oom_drop += 1;
                emit(ex, SMOR_EV_OOM_DROP, (uint32_t)op_index, token_id,
                     (uint32_t)chosen, primary, secondary, cost,
                     SMOR_U64_MAX, SMOR_U64_MAX);
                return;
            }
            // accept
            credit[chosen] -= cost;
            SmorOToken t;
            t.token_id = token_id;
            t.status = SMOR_STATUS_LIVE;
            t.primary = primary;
            t.secondary = secondary;
            t.cost = cost;
            t.assigned_expert = (uint32_t)chosen;
            t.admit_seq = event_seq;  // event_seq of ACCEPT
            t.route_kind = is_primary ? SMOR_RK_PRIMARY : SMOR_RK_SECONDARY;
            table[token_id] = t;
            live_count[chosen] += 1;
            live_total += 1;
            if (is_primary) ex->accepted_primary += 1; else ex->accepted_secondary += 1;
            emit(ex, is_primary ? SMOR_EV_ACCEPT_PRIMARY : SMOR_EV_ACCEPT_SECONDARY,
                 (uint32_t)op_index, token_id, (uint32_t)chosen, primary, secondary,
                 cost, /*credit_after*/credit[chosen], SMOR_U64_MAX);
            return;
        }

        // admission failed
        bool queue_room = ((int)fifo.size() < spec.overflow_capacity);
        bool table_room = (live_total + (int)fifo.size() < spec.max_live_tokens + spec.overflow_capacity);
        if (queue_room && table_room) {
            SmorOToken t;
            t.token_id = token_id;
            t.status = SMOR_STATUS_QUEUED;
            t.primary = primary;
            t.secondary = secondary;
            t.cost = cost;
            t.arrival_seq = event_seq;  // event_seq of QUEUE
            table[token_id] = t;
            fifo.push_back(token_id);
            ex->queued += 1;
            emit(ex, SMOR_EV_QUEUE, (uint32_t)op_index, token_id,
                 SMOR_U32_MAX, primary, secondary, cost,
                 SMOR_U64_MAX, /*arrival_seq*/t.arrival_seq);
            return;
        }
        if (!queue_room) {
            ex->capacity_drop += 1;
            emit(ex, SMOR_EV_CAPACITY_DROP, (uint32_t)op_index, token_id,
                 SMOR_U32_MAX, primary, secondary, cost,
                 SMOR_U64_MAX, SMOR_U64_MAX);
            return;
        }
        // queue room exists but token table is full
        ex->oom_drop += 1;
        emit(ex, SMOR_EV_OOM_DROP, (uint32_t)op_index, token_id,
             SMOR_U32_MAX, primary, secondary, cost,
             SMOR_U64_MAX, SMOR_U64_MAX);
    }

    void do_drain(SmorExpected* ex, int op_index, const SmorHostInputsView& in) {
        uint64_t limit = in.op_a[op_index];
        // limit == 0 is a valid no-op (no event emitted)
        while (limit > 0 && !fifo.empty()) {
            uint64_t head_id = fifo.front();
            auto it = table.find(head_id);
            SmorOToken& t = it->second;

            bool is_primary = false;
            int chosen = choose_admit(t.primary, t.secondary, t.cost, &is_primary);
            if (chosen < 0) {
                // head-of-line block: stop immediately
                break;
            }

            // admission succeeds (credit-wise); check live capacity
            if (live_total == spec.max_live_tokens) {
                // capacity would be exceeded: pop head, remove table entry,
                // emit OOM_DROP_REPLAY, decrement limit, continue.
                fifo.pop_front();
                uint64_t tid = t.token_id;
                uint32_t pr = t.primary, sc = t.secondary;
                uint64_t cst = t.cost;
                table.erase(it);
                // DETERMINISTIC INTERPRETATION: the contract's count list (sec 4)
                // has exactly one "oom_drop" count which maps 1:1 to the OOM_DROP
                // event kind. OOM_DROP_REPLAY is a distinct event kind with NO
                // dedicated count field, so it is reflected ONLY in
                // route_event_hash and does not increment any count.
                emit(ex, SMOR_EV_OOM_DROP_REPLAY, (uint32_t)op_index, tid,
                     SMOR_U32_MAX, pr, sc, cst, SMOR_U64_MAX, SMOR_U64_MAX);
                limit -= 1;
                continue;
            }

            // accept replay
            fifo.pop_front();
            credit[chosen] -= t.cost;
            t.status = SMOR_STATUS_LIVE;
            t.assigned_expert = (uint32_t)chosen;
            t.admit_seq = event_seq;  // event_seq of replay event
            t.route_kind = is_primary ? SMOR_RK_REPLAY_PRIMARY : SMOR_RK_REPLAY_SECONDARY;
            live_count[chosen] += 1;
            live_total += 1;
            if (is_primary) ex->replayed_primary += 1; else ex->replayed_secondary += 1;
            emit(ex, is_primary ? SMOR_EV_REPLAY_PRIMARY : SMOR_EV_REPLAY_SECONDARY,
                 (uint32_t)op_index, t.token_id, (uint32_t)chosen, t.primary, t.secondary,
                 t.cost, /*credit_after*/credit[chosen], SMOR_U64_MAX);
            limit -= 1;
        }
    }

    void do_retire(SmorExpected* ex, int op_index, const SmorHostInputsView& in) {
        uint64_t token_id = in.op_a[op_index];
        auto it = table.find(token_id);
        if (it == table.end()) {
            ex->invalid_count += 1;
            emit(ex, SMOR_EV_INVALID, (uint32_t)op_index, token_id,
                 SMOR_U32_MAX, SMOR_U32_MAX, SMOR_U32_MAX, /*cost*/0,
                 SMOR_U64_MAX, SMOR_U64_MAX);
            return;
        }
        SmorOToken& t = it->second;
        if (t.status == SMOR_STATUS_LIVE) {
            uint32_t assigned = t.assigned_expert;
            uint32_t pr = t.primary, sc = t.secondary;
            uint64_t cst = t.cost;
            live_count[assigned] -= 1;  // no credit refund
            live_total -= 1;
            table.erase(it);
            ex->retired_live += 1;
            emit(ex, SMOR_EV_RETIRE_LIVE, (uint32_t)op_index, token_id,
                 assigned, pr, sc, cst, SMOR_U64_MAX, SMOR_U64_MAX);
        } else {
            // queued: remove from FIFO + table
            uint32_t pr = t.primary, sc = t.secondary;
            uint64_t cst = t.cost;
            uint64_t arr = t.arrival_seq;
            for (auto fit = fifo.begin(); fit != fifo.end(); ++fit) {
                if (*fit == token_id) { fifo.erase(fit); break; }
            }
            table.erase(it);
            ex->retired_queued += 1;
            emit(ex, SMOR_EV_RETIRE_QUEUED, (uint32_t)op_index, token_id,
                 SMOR_U32_MAX, pr, sc, cst, SMOR_U64_MAX, /*arrival_seq*/arr);
        }
    }

    void do_drop_through(SmorExpected* ex, int op_index, const SmorHostInputsView& in) {
        uint64_t cutoff = in.op_a[op_index];
        // scan FIFO head->tail, drop while arrival_seq <= cutoff, stop at first larger
        while (!fifo.empty()) {
            uint64_t head_id = fifo.front();
            auto it = table.find(head_id);
            SmorOToken& t = it->second;
            if (t.arrival_seq <= cutoff) {
                uint32_t pr = t.primary, sc = t.secondary;
                uint64_t cst = t.cost;
                uint64_t arr = t.arrival_seq;
                uint64_t tid = t.token_id;
                fifo.pop_front();
                table.erase(it);
                ex->queue_drop += 1;
                emit(ex, SMOR_EV_QUEUE_DROP, (uint32_t)op_index, tid,
                     SMOR_U32_MAX, pr, sc, cst, SMOR_U64_MAX, /*arrival_seq*/arr);
            } else {
                break;
            }
        }
    }

    // ---- output checksums ---------------------------------------------
    uint64_t credit_checksum() const {
        uint64_t h = SMOR_FNV_OFFSET;
        for (int e = 0; e < spec.num_experts; ++e) {
            uint32_t eid = (uint32_t)e;
            uint64_t cr = credit[(size_t)e];
            uint64_t lc = live_count[(size_t)e];
            smor_o_fnv(&h, &eid, sizeof(uint32_t));
            smor_o_fnv(&h, &cr, sizeof(uint64_t));
            smor_o_fnv(&h, &lc, sizeof(uint64_t));
        }
        return h;
    }

    uint64_t assignment_checksum() const {
        // live assignments by expert ascending, then admit_seq ascending,
        // then token_id ascending.
        struct Rec { uint32_t expert; uint64_t admit_seq; uint64_t token_id;
                     uint32_t primary; uint32_t secondary; uint64_t cost; uint8_t route_kind; };
        std::vector<Rec> recs;
        for (const auto& kv : table) {
            const SmorOToken& t = kv.second;
            if (t.status != SMOR_STATUS_LIVE) continue;
            recs.push_back(Rec{t.assigned_expert, t.admit_seq, t.token_id,
                               t.primary, t.secondary, t.cost, t.route_kind});
        }
        std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b) {
            if (a.expert != b.expert) return a.expert < b.expert;
            if (a.admit_seq != b.admit_seq) return a.admit_seq < b.admit_seq;
            return a.token_id < b.token_id;
        });
        uint64_t h = SMOR_FNV_OFFSET;
        for (const Rec& r : recs) {
            smor_o_fnv(&h, &r.expert, sizeof(uint32_t));
            smor_o_fnv(&h, &r.token_id, sizeof(uint64_t));
            smor_o_fnv(&h, &r.admit_seq, sizeof(uint64_t));
            smor_o_fnv(&h, &r.primary, sizeof(uint32_t));
            smor_o_fnv(&h, &r.secondary, sizeof(uint32_t));
            smor_o_fnv(&h, &r.cost, sizeof(uint64_t));
            smor_o_fnv(&h, &r.route_kind, sizeof(uint8_t));
        }
        return h;
    }

    uint64_t overflow_checksum() const {
        // queued tokens head->tail
        uint64_t h = SMOR_FNV_OFFSET;
        for (uint64_t tid : fifo) {
            auto it = table.find(tid);
            const SmorOToken& t = it->second;
            uint64_t token_id = t.token_id;
            uint64_t arr = t.arrival_seq;
            uint32_t pr = t.primary, sc = t.secondary;
            uint64_t cst = t.cost;
            smor_o_fnv(&h, &token_id, sizeof(uint64_t));
            smor_o_fnv(&h, &arr, sizeof(uint64_t));
            smor_o_fnv(&h, &pr, sizeof(uint32_t));
            smor_o_fnv(&h, &sc, sizeof(uint32_t));
            smor_o_fnv(&h, &cst, sizeof(uint64_t));
        }
        return h;
    }

    void step_once(const SmorRunSpec& run,
                   const SmorHostInputsView& in,
                   SmorExpected* expected) {
        // per-step counts reset to 0; checksums of stateful tables reflect
        // post-step state, route_event_hash reflects this step's events only.
        *expected = SmorExpected();

        for (int i = 0; i < run.batch_size; ++i) {
            int kind = in.op_kind[i];
            switch (kind) {
                case SMOR_OP_REFILL: do_refill(expected, i, in); break;
                case SMOR_OP_ROUTE: do_route(expected, i, in); break;
                case SMOR_OP_DRAIN: do_drain(expected, i, in); break;
                case SMOR_OP_RETIRE: do_retire(expected, i, in); break;
                case SMOR_OP_DROP_QUEUED_THROUGH: do_drop_through(expected, i, in); break;
                default:
                    // unknown op kind -> INVALID
                    expected->invalid_count += 1;
                    emit(expected, SMOR_EV_INVALID, (uint32_t)i, /*token*/0,
                         SMOR_U32_MAX, SMOR_U32_MAX, SMOR_U32_MAX, /*cost*/0,
                         SMOR_U64_MAX, SMOR_U64_MAX);
                    break;
            }
        }

        step_seq += 1;

        expected->credit_hash = credit_checksum();
        expected->assignment_hash = assignment_checksum();
        expected->overflow_hash = overflow_checksum();
    }
};

static inline bool smor_check_u64(const char* name, uint64_t got, uint64_t exp, std::string* error) {
    if (got != exp) {
        if (error) {
            std::ostringstream oss;
            oss << name << " mismatch: got " << got << " (0x" << std::hex << got
                << std::dec << "), expected " << exp << " (0x" << std::hex << exp << std::dec << ")";
            *error = oss.str();
        }
        return false;
    }
    return true;
}

static inline bool smor_check_all_outputs(
    const SmorExpected& e,
    const SmorHostOutputsView& g,
    std::string* error) {
    if (!smor_check_u64("refill_count", g.refill_count[0], e.refill_count, error)) return false;
    if (!smor_check_u64("accepted_primary", g.accepted_primary[0], e.accepted_primary, error)) return false;
    if (!smor_check_u64("accepted_secondary", g.accepted_secondary[0], e.accepted_secondary, error)) return false;
    if (!smor_check_u64("queued", g.queued[0], e.queued, error)) return false;
    if (!smor_check_u64("replayed_primary", g.replayed_primary[0], e.replayed_primary, error)) return false;
    if (!smor_check_u64("replayed_secondary", g.replayed_secondary[0], e.replayed_secondary, error)) return false;
    if (!smor_check_u64("capacity_drop", g.capacity_drop[0], e.capacity_drop, error)) return false;
    if (!smor_check_u64("oom_drop", g.oom_drop[0], e.oom_drop, error)) return false;
    if (!smor_check_u64("duplicate_count", g.duplicate_count[0], e.duplicate_count, error)) return false;
    if (!smor_check_u64("retired_live", g.retired_live[0], e.retired_live, error)) return false;
    if (!smor_check_u64("retired_queued", g.retired_queued[0], e.retired_queued, error)) return false;
    if (!smor_check_u64("queue_drop", g.queue_drop[0], e.queue_drop, error)) return false;
    if (!smor_check_u64("invalid_count", g.invalid_count[0], e.invalid_count, error)) return false;
    if (!smor_check_u64("route_event_hash", g.route_event_hash[0], e.route_event_hash, error)) return false;
    if (!smor_check_u64("credit_hash", g.credit_hash[0], e.credit_hash, error)) return false;
    if (!smor_check_u64("assignment_hash", g.assignment_hash[0], e.assignment_hash, error)) return false;
    if (!smor_check_u64("overflow_hash", g.overflow_hash[0], e.overflow_hash, error)) return false;
    return true;
}

#endif  // SWITCH_MOE_OVERFLOW_ROUTER_ORACLE_HPP_
