// file: lock_manager_deadlock_oracle.hpp
//
// Host-side golden oracle for T55 (multi-granularity lock manager). This is the
// canonical, fully-deterministic specification the device implementations are
// graded against. It uses STL containers (std::map / std::vector); the device
// reference and naive implementations re-derive the same algorithm over
// different data structures with no shared code.

#ifndef LOCK_MANAGER_DEADLOCK_ORACLE_HPP_
#define LOCK_MANAGER_DEADLOCK_ORACLE_HPP_

#include "lock_manager_deadlock_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ------------------------------------------------------------------ FNV-1a-64
struct LmdFnv {
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

// ------------------------------------------------------------------ mode tables
// Compatibility between DIFFERENT transactions. comp[a][b] == 1 iff a holder and
// a b holder can coexist on the same resource.
static inline bool lmd_compat(int a, int b) {
    static const uint8_t M[6][6] = {
        //        NL IS IX S  SIX X
        /*NL */ {  1, 1, 1, 1, 1, 1 },
        /*IS */ {  1, 1, 1, 1, 1, 0 },
        /*IX */ {  1, 1, 1, 0, 0, 0 },
        /*S  */ {  1, 1, 0, 1, 0, 0 },
        /*SIX*/ {  1, 1, 0, 0, 0, 0 },
        /*X  */ {  1, 0, 0, 0, 0, 0 },
    };
    return M[a][b] != 0;
}

// dominates(a,b): holding mode a covers a request for mode b.
static inline bool lmd_dominates(int a, int b) {
    static const uint8_t D[6][6] = {
        //        NL IS IX S  SIX X
        /*NL */ {  1, 0, 0, 0, 0, 0 },
        /*IS */ {  1, 1, 0, 0, 0, 0 },
        /*IX */ {  1, 1, 1, 0, 0, 0 },
        /*S  */ {  1, 1, 0, 1, 0, 0 },
        /*SIX*/ {  1, 1, 1, 1, 1, 0 },
        /*X  */ {  1, 1, 1, 1, 1, 1 },
    };
    return D[a][b] != 0;
}

// least upper bound in the dominance lattice.
static inline int lmd_lub(int a, int b) {
    static const uint8_t L[6][6] = {
        //        NL  IS  IX  S   SIX X
        /*NL */ {  0,  1,  2,  3,  4,  5 },
        /*IS */ {  1,  1,  2,  3,  4,  5 },
        /*IX */ {  2,  2,  2,  4,  4,  5 },
        /*S  */ {  3,  3,  4,  3,  4,  5 },
        /*SIX*/ {  4,  4,  4,  4,  4,  5 },
        /*X  */ {  5,  5,  5,  5,  5,  5 },
    };
    return L[a][b];
}

// ------------------------------------------------------------------ resource
struct LmdRes {
    int kind;        // LMD_ROW/PARTITION/TABLE
    int t, p, r;     // -1 sentinels for unused dims
    bool operator==(const LmdRes& o) const {
        return kind == o.kind && t == o.t && p == o.p && r == o.r;
    }
};

// canonical comparison: TABLE before PARTITION before ROW; then (t,p,r) ascending.
// kind_rank: TABLE=0, PARTITION=1, ROW=2.
static inline int lmd_kind_rank_canon(int kind) {
    if (kind == LMD_TABLE) return 0;
    if (kind == LMD_PARTITION) return 1;
    return 2;  // ROW
}
struct LmdResCanonLess {
    bool operator()(const LmdRes& a, const LmdRes& b) const {
        int ra = lmd_kind_rank_canon(a.kind), rb = lmd_kind_rank_canon(b.kind);
        if (ra != rb) return ra < rb;
        if (a.t != b.t) return a.t < b.t;
        if (a.p != b.p) return a.p < b.p;
        return a.r < b.r;
    }
};
// release-cascade comparison: ROW before PARTITION before TABLE; then ascending.
static inline int lmd_kind_rank_release(int kind) {
    if (kind == LMD_ROW) return 0;
    if (kind == LMD_PARTITION) return 1;
    return 2;  // TABLE
}
struct LmdResReleaseLess {
    bool operator()(const LmdRes& a, const LmdRes& b) const {
        int ra = lmd_kind_rank_release(a.kind), rb = lmd_kind_rank_release(b.kind);
        if (ra != rb) return ra < rb;
        if (a.t != b.t) return a.t < b.t;
        if (a.p != b.p) return a.p < b.p;
        return a.r < b.r;
    }
};

static inline LmdRes lmd_make_table(int t) { return LmdRes{LMD_TABLE, t, -1, -1}; }
static inline LmdRes lmd_make_part(int t, int p) { return LmdRes{LMD_PARTITION, t, p, -1}; }
static inline LmdRes lmd_make_row(int t, int p, int r) { return LmdRes{LMD_ROW, t, p, r}; }

// ------------------------------------------------------------------ outputs view
struct LmdExpected {
    std::vector<int64_t> counts;
    int32_t op_index = 0;
    uint64_t event_seq = 0;
    uint64_t lock_event_hash = 0;
    uint64_t grant_hash = 0;
    uint64_t wait_hash = 0;
    uint64_t txn_lock_hash = 0;
    uint64_t state_checksum = 0;
};

struct LmdHostOutputsView {
    const int64_t* counts;
    const int32_t* op_index_out;
    const uint64_t* event_seq_out;
    const uint64_t* lock_event_hash;
    const uint64_t* grant_hash;
    const uint64_t* wait_hash;
    const uint64_t* txn_lock_hash;
    const uint64_t* state_checksum;
};

// ------------------------------------------------------------------ Oracle
struct LmdOracle {
    static const uint64_t U64MAX = ~0ULL;
    static const uint32_t U32MAX = 0xFFFFFFFFu;

    struct GrantKey {
        LmdRes res;
        uint64_t txn;
        bool operator<(const GrantKey& o) const {
            LmdResCanonLess lt;
            if (!(res == o.res)) { return lt(res, o.res); }
            return txn < o.txn;
        }
    };
    struct Grant {
        int mode = LMD_NL;
        uint64_t explicit_count = 0;
        uint64_t auto_count = 0;
        uint64_t grant_seq = 0;
        uint64_t last_convert_seq = 0;
    };

    struct Waiter {
        uint64_t txn;
        int requested_mode;
        int request_plan_index;
        uint64_t request_seq;
        uint8_t is_conversion;
        LmdRes original_target_res;
        int original_target_mode;
        uint64_t wait_seq;
    };

    struct TableCtr {
        uint64_t row_s_count = 0;
        uint64_t row_x_count = 0;
        uint64_t partition_s_count = 0;
        uint64_t partition_x_count = 0;
        uint64_t escalation_attempts = 0;
    };

    struct Txn {
        uint64_t txn_id = 0;
        uint64_t txn_seq = 0;
        uint32_t priority = 0;
        int status = LMD_ACTIVE;
        LmdRes blocked_resource{LMD_TABLE, -1, -1, -1};
        bool has_blocked = false;
        int blocked_mode = LMD_NL;
        uint64_t wait_seq = U64MAX;
        uint64_t locks_acquired = 0;
        uint64_t deadlock_aborts = 0;
        std::vector<TableCtr> tctr;   // per-table counters, size table_count
    };

    // A plan entry.
    struct PlanEntry { LmdRes res; int mode; bool is_target; };

    LmdProblemSpec spec{};

    uint64_t event_seq = 0;
    uint64_t txn_seq_next = 1;
    uint64_t request_seq_next = 1;
    uint64_t wait_seq_next = 1;
    uint64_t deadlock_seq_next = 1;
    int32_t op_index = 0;
    uint64_t event_hash = 1469598103934665603ULL;

    std::map<uint64_t, Txn> txns;
    std::map<GrantKey, Grant> grants;
    // wait queue per resource (canonical resource key -> entries by wait_seq)
    std::map<LmdRes, std::vector<Waiter>, LmdResCanonLess> waits;

    std::vector<int64_t> counts;

    void init(const LmdProblemSpec& s) {
        spec = s;
        counts.assign(LMD_COUNT_N, 0);
        reset();
    }
    void reset() {
        event_seq = 0;
        txn_seq_next = 1;
        request_seq_next = 1;
        wait_seq_next = 1;
        deadlock_seq_next = 1;
        op_index = 0;
        event_hash = 1469598103934665603ULL;
        txns.clear();
        grants.clear();
        waits.clear();
        for (auto& c : counts) c = 0;
    }

    Txn* find_txn(uint64_t id) {
        auto it = txns.find(id);
        return it == txns.end() ? nullptr : &it->second;
    }
    Grant* find_grant(const LmdRes& res, uint64_t txn) {
        auto it = grants.find(GrantKey{res, txn});
        return (it == grants.end()) ? nullptr : &it->second;
    }
    bool grant_exists(const Grant* g) const {
        return g && (g->explicit_count != 0 || g->auto_count != 0);
    }

    // -------------------------------------------------------------- emit
    void emit(uint8_t kind, uint64_t txn_or_zero, const LmdRes* res, int mode_or_255, uint64_t aux) {
        const uint64_t seq = event_seq;
        LmdFnv f; f.h = event_hash;
        f.u8(kind);
        f.u64(seq);
        f.u32((uint32_t)op_index);
        f.u64(txn_or_zero);
        if (res) {
            f.u8((uint8_t)res->kind);
            f.u32(res->t < 0 ? U32MAX : (uint32_t)res->t);
            f.u32(res->p < 0 ? U32MAX : (uint32_t)res->p);
            f.u32(res->r < 0 ? U32MAX : (uint32_t)res->r);
        } else {
            f.u8((uint8_t)255);
            f.u32(U32MAX); f.u32(U32MAX); f.u32(U32MAX);
        }
        f.u8(mode_or_255 < 0 ? (uint8_t)255 : (uint8_t)mode_or_255);
        f.u64(aux);
        event_hash = f.h;
        event_seq += 1;
    }

    // -------------------------------------------------------------- compatibility
    // Is requested mode compatible with all OTHER transactions' granted modes on res?
    bool compatible_with_others(const LmdRes& res, uint64_t self, int req_mode) {
        // iterate grants on this exact resource
        for (auto it = grants.lower_bound(GrantKey{res, 0});
             it != grants.end(); ++it) {
            if (!(it->first.res == res)) break;
            if (it->first.txn == self) continue;
            if (!grant_exists(&it->second)) continue;
            if (!lmd_compat(req_mode, it->second.mode)) return false;
        }
        return true;
    }

    // -------------------------------------------------------------- descendant counters
    // Per-table descendant counters are RECOMPUTED from the transaction's
    // explicitly-held row/partition grants. Each distinct explicitly-held
    // fine-grained resource contributes +1 to the s-like and/or x-like counter
    // for its granularity, classified by the resource's current effective mode:
    //   s-like : mode in {S, SIX}
    //   x-like : mode in {X, IX, SIX}
    // escalation_attempts is preserved (it is bumped explicitly).
    void recompute_counters(Txn& tx) {
        for (auto& c : tx.tctr) {
            c.row_s_count = 0; c.row_x_count = 0;
            c.partition_s_count = 0; c.partition_x_count = 0;
            // escalation_attempts preserved.
        }
        for (auto it = grants.begin(); it != grants.end(); ++it) {
            if (it->first.txn != tx.txn_id) continue;
            const Grant& g = it->second;
            if (g.explicit_count == 0) continue;  // only explicit holdings count
            const LmdRes& r = it->first.res;
            if (r.kind == LMD_TABLE) continue;
            TableCtr& c = tx.tctr[(size_t)r.t];
            bool s_like = (g.mode == LMD_S || g.mode == LMD_SIX);
            bool x_like = (g.mode == LMD_X || g.mode == LMD_IX || g.mode == LMD_SIX);
            if (r.kind == LMD_ROW) {
                if (s_like) c.row_s_count += 1;
                if (x_like) c.row_x_count += 1;
            } else {  // PARTITION
                if (s_like) c.partition_s_count += 1;
                if (x_like) c.partition_x_count += 1;
            }
        }
    }
    // total fine-grained explicit lock count under a table for tx.
    uint64_t fine_count(const Txn& tx, int t) const {
        const TableCtr& c = tx.tctr[(size_t)t];
        return c.row_s_count + c.row_x_count + c.partition_s_count + c.partition_x_count;
    }
    bool holds_x_like_descendant(const Txn& tx, int t) const {
        const TableCtr& c = tx.tctr[(size_t)t];
        return (c.row_x_count + c.partition_x_count) > 0;
    }

    // -------------------------------------------------------------- plan
    std::vector<PlanEntry> build_plan(const LmdRes& target, int mode) {
        std::vector<PlanEntry> plan;
        if (target.kind == LMD_TABLE) {
            plan.push_back(PlanEntry{lmd_make_table(target.t), mode, true});
        } else if (target.kind == LMD_PARTITION) {
            int anc = (mode == LMD_S || mode == LMD_IS) ? LMD_IS : LMD_IX;
            plan.push_back(PlanEntry{lmd_make_table(target.t), anc, false});
            plan.push_back(PlanEntry{lmd_make_part(target.t, target.p), mode, true});
        } else {  // ROW
            int tanc = (mode == LMD_S) ? LMD_IS : LMD_IX;
            int panc = (mode == LMD_S) ? LMD_IS : LMD_IX;
            plan.push_back(PlanEntry{lmd_make_table(target.t), tanc, false});
            plan.push_back(PlanEntry{lmd_make_part(target.t, target.p), panc, false});
            plan.push_back(PlanEntry{lmd_make_row(target.t, target.p, target.r), mode, true});
        }
        return plan;
    }

    // -------------------------------------------------------------- validation
    bool resource_valid(const LmdRes& res) {
        if (res.kind == LMD_TABLE) {
            return res.t >= 0 && res.t < spec.table_count && res.p < 0 && res.r < 0;
        }
        if (res.kind == LMD_PARTITION) {
            return res.t >= 0 && res.t < spec.table_count &&
                   res.p >= 0 && res.p < spec.partitions_per_table && res.r < 0;
        }
        if (res.kind == LMD_ROW) {
            return res.t >= 0 && res.t < spec.table_count &&
                   res.p >= 0 && res.p < spec.partitions_per_table &&
                   res.r >= 0 && res.r < spec.rows_per_partition;
        }
        return false;
    }
    static bool mode_is_request(int m) {
        return m == LMD_IS || m == LMD_IX || m == LMD_S || m == LMD_SIX || m == LMD_X;
    }

    void invalid() {
        counts[LMD_C_INVALID] += 1;
        emit(LMD_EV_INVALID, 0, nullptr, -1, 0);
    }

    // -------------------------------------------------------------- grant helper
    // Grant or convert mode at res for tx. is_target controls which count bumps.
    // Returns nothing; emits LOCK_GRANT or LOCK_CONVERT.
    void do_grant_or_convert(Txn& tx, const LmdRes& res, int req_mode, bool is_target) {
        GrantKey key{res, tx.txn_id};
        auto it = grants.find(key);
        if (it == grants.end() || !grant_exists(&it->second)) {
            // new grant
            Grant g;
            g.mode = req_mode;
            g.explicit_count = is_target ? 1 : 0;
            g.auto_count = is_target ? 0 : 1;
            g.grant_seq = event_seq;
            g.last_convert_seq = 0;
            grants[key] = g;
            tx.locks_acquired += 1;
            if (is_target) recompute_counters(tx);
            emit(LMD_EV_LOCK_GRANT, tx.txn_id, &res, req_mode, 0);
        } else {
            // conversion to lub
            Grant& g = it->second;
            int old = g.mode;
            int nm = lmd_lub(old, req_mode);
            (void)old;
            g.mode = nm;
            if (is_target) g.explicit_count += 1; else g.auto_count += 1;
            g.last_convert_seq = event_seq;
            if (is_target) recompute_counters(tx);
            tx.locks_acquired += 1;
            emit(LMD_EV_LOCK_CONVERT, tx.txn_id, &res, nm, 0);
        }
    }

    // -------------------------------------------------------------- escalation
    // Attempt automatic escalation for tx on table t. Called after a successful
    // fine-grained target grant when fine_count(tx,t) reaches the threshold.
    void try_escalate(Txn& tx, int t) {
        tx.tctr[(size_t)t].escalation_attempts += 1;
        int target_mode = holds_x_like_descendant(tx, t) ? LMD_X : LMD_S;
        LmdRes table_res = lmd_make_table(t);
        // compatibility check vs other holders for the table escalation mode.
        if (!compatible_with_others(table_res, tx.txn_id, target_mode)) {
            counts[LMD_C_ESCALATE_BLOCKED] += 1;
            emit(LMD_EV_ESCALATE_BLOCKED, tx.txn_id, &table_res, target_mode, 0);
            return;
        }
        // grant or convert the table lock to target_mode.
        GrantKey tkey{table_res, tx.txn_id};
        auto it = grants.find(tkey);
        if (it == grants.end() || !grant_exists(&it->second)) {
            Grant g;
            g.mode = target_mode;
            g.explicit_count = 1;
            g.auto_count = 0;
            g.grant_seq = event_seq;
            g.last_convert_seq = 0;
            grants[tkey] = g;
            tx.locks_acquired += 1;
            emit(LMD_EV_ESCALATE_GRANT, tx.txn_id, &table_res, target_mode, 0);
        } else {
            Grant& g = it->second;
            int nm = lmd_lub(g.mode, target_mode);
            g.mode = nm;
            g.explicit_count += 1;  // escalation becomes an explicit table hold
            g.last_convert_seq = event_seq;
            tx.locks_acquired += 1;
            emit(LMD_EV_ESCALATE_GRANT, tx.txn_id, &table_res, nm, 0);
        }
        // Release every non-table grant under this table in canonical
        // (TABLE,PARTITION,ROW) order. Explicit descendant locks emit
        // ESCALATE_RELEASE; pure-auto intents (whose descendants are now gone)
        // are dropped silently. Per-table descendant counters are cleared.
        std::vector<LmdRes> to_release;
        for (auto git = grants.begin(); git != grants.end(); ++git) {
            if (git->first.txn != tx.txn_id) continue;
            if (!grant_exists(&git->second)) continue;
            const LmdRes& r = git->first.res;
            if (r.kind == LMD_TABLE) continue;
            if (r.t != t) continue;
            to_release.push_back(r);
        }
        std::sort(to_release.begin(), to_release.end(), LmdResCanonLess());
        std::vector<LmdRes> affected;
        for (const LmdRes& r : to_release) {
            Grant* g = find_grant(r, tx.txn_id);
            const bool had_explicit = (g->explicit_count != 0);
            g->explicit_count = 0;
            g->auto_count = 0;
            g->mode = LMD_NL;
            if (had_explicit) {
                counts[LMD_C_ESCALATE_RELEASES] += 1;
                emit(LMD_EV_ESCALATE_RELEASE, tx.txn_id, &r, -1, 0);
            }
            affected.push_back(r);
        }
        recompute_counters(tx);
        // Wake affected resources (canonical order). Table mode changed too.
        affected.push_back(table_res);
        wake_resources(affected);
    }

    // -------------------------------------------------------------- LOCK
    void op_lock(uint64_t txn_id, const LmdRes& target, int mode) {
        Txn* tp = find_txn(txn_id);
        if (!tp || tp->status != LMD_ACTIVE || !resource_valid(target) || !mode_is_request(mode)) {
            invalid();
            return;
        }
        Txn& tx = *tp;
        (void)(request_seq_next++);  // each LOCK consumes a request_seq for its waiters
        uint64_t this_request_seq = request_seq_next - 1;
        std::vector<PlanEntry> plan = build_plan(target, mode);
        for (size_t pi = 0; pi < plan.size(); ++pi) {
            const PlanEntry& e = plan[pi];
            Grant* g = find_grant(e.res, txn_id);
            if (grant_exists(g) && lmd_dominates(g->mode, e.mode)) {
                if (e.is_target) g->explicit_count += 1;
                else g->auto_count += 1;
                counts[LMD_C_LOCK_REENTERS] += 1;
                emit(LMD_EV_LOCK_REENTER, txn_id, &e.res, e.mode, 0);
                continue;
            }
            // conversion or new grant: needs compatibility with other txns.
            int eff_req = e.mode;
            if (grant_exists(g)) eff_req = lmd_lub(g->mode, e.mode);
            if (compatible_with_others(e.res, txn_id, eff_req)) {
                do_grant_or_convert(tx, e.res, e.mode, e.is_target);
                if (e.is_target && (e.res.kind == LMD_ROW || e.res.kind == LMD_PARTITION)) {
                    if (fine_count(tx, e.res.t) >= (uint64_t)spec.escalation_threshold) {
                        try_escalate(tx, e.res.t);
                    }
                }
            } else {
                // enqueue waiter at this resource.
                enqueue_waiter(tx, e.res, e.mode, (int)pi, this_request_seq,
                               /*is_conversion=*/0, target, mode);
                counts[LMD_C_LOCK_WAITS] += 1;
                emit(LMD_EV_LOCK_WAIT, txn_id, &e.res, e.mode, 0);
                return;  // stop; earlier ancestor grants remain held.
            }
        }
    }

    void enqueue_waiter(Txn& tx, const LmdRes& res, int req_mode, int plan_index,
                        uint64_t request_seq, uint8_t is_conv,
                        const LmdRes& orig_target, int orig_mode) {
        Waiter w;
        w.txn = tx.txn_id;
        w.requested_mode = req_mode;
        w.request_plan_index = plan_index;
        w.request_seq = request_seq;
        w.is_conversion = is_conv;
        w.original_target_res = orig_target;
        w.original_target_mode = orig_mode;
        w.wait_seq = wait_seq_next++;
        waits[res].push_back(w);
        tx.status = LMD_WAITING;
        tx.blocked_resource = res;
        tx.has_blocked = true;
        tx.blocked_mode = req_mode;
        tx.wait_seq = w.wait_seq;
    }

    // -------------------------------------------------------------- UNLOCK
    void op_unlock(uint64_t txn_id, const LmdRes& res) {
        Txn* tp = find_txn(txn_id);
        if (!tp) { invalid(); return; }
        Grant* g = find_grant(res, txn_id);
        if (!grant_exists(g) || g->explicit_count == 0) { invalid(); return; }
        Txn& tx = *tp;
        g->explicit_count -= 1;
        if (g->explicit_count == 0) {
            if (g->auto_count == 0) {
                g->mode = LMD_NL;  // removed
                if (res.kind == LMD_ROW || res.kind == LMD_PARTITION) recompute_counters(tx);
                emit_release_and_wake(tx, res);
                return;
            } else {
                // only auto intent remains; downgrade to strongest needed intent.
                if (res.kind == LMD_ROW || res.kind == LMD_PARTITION) recompute_counters(tx);
                int new_mode = intent_for_descendants(tx, res);
                g->mode = new_mode;
                emit_release_and_wake(tx, res);
                return;
            }
        }
        // explicit_count still > 0: lock still held, emit release of one count.
        if (res.kind == LMD_ROW || res.kind == LMD_PARTITION) recompute_counters(tx);
        emit_release_and_wake(tx, res);
    }

    // strongest remaining intent needed by descendants of res for tx:
    // IX if any descendant X-like lock exists, otherwise IS.
    int intent_for_descendants(const Txn& tx, const LmdRes& res) {
        // We track per-table descendant counts; use the table's x-like presence.
        const TableCtr& c = tx.tctr[(size_t)res.t];
        bool x_like = (c.row_x_count + c.partition_x_count) > 0;
        return x_like ? LMD_IX : LMD_IS;
    }

    void emit_release_and_wake(Txn& tx, const LmdRes& res) {
        counts[LMD_C_LOCK_RELEASES] += 1;
        emit(LMD_EV_LOCK_RELEASE, tx.txn_id, &res, -1, 0);
        std::vector<LmdRes> affected;
        affected.push_back(res);
        wake_resources(affected);
    }

    // -------------------------------------------------------------- UNLOCK_ALL
    void op_unlock_all(uint64_t txn_id) {
        Txn* tp = find_txn(txn_id);
        if (!tp) { invalid(); return; }
        // collect all granted resources for txn in RELEASE order (ROW,PART,TABLE).
        std::vector<LmdRes> rels;
        for (auto it = grants.begin(); it != grants.end(); ++it) {
            if (it->first.txn != txn_id) continue;
            if (!grant_exists(&it->second)) continue;
            rels.push_back(it->first.res);
        }
        std::sort(rels.begin(), rels.end(), LmdResReleaseLess());
        for (const LmdRes& r : rels) {
            Grant* g = find_grant(r, txn_id);
            g->explicit_count = 0;
            g->auto_count = 0;
            g->mode = LMD_NL;
            counts[LMD_C_LOCK_RELEASE_ALL] += 1;
            emit(LMD_EV_LOCK_RELEASE_ALL, txn_id, &r, -1, 0);
        }
        // remove the transaction.
        txns.erase(txn_id);
        // wake every affected resource in canonical order.
        std::vector<LmdRes> affected = rels;
        wake_resources(affected);
    }

    // -------------------------------------------------------------- CONVERT
    void op_convert(uint64_t txn_id, const LmdRes& res, int new_mode) {
        Txn* tp = find_txn(txn_id);
        if (!tp || tp->status != LMD_ACTIVE || !resource_valid(res) || !mode_is_request(new_mode)) {
            invalid();
            return;
        }
        Grant* g = find_grant(res, txn_id);
        if (!grant_exists(g)) { invalid(); return; }
        Txn& tx = *tp;
        if (lmd_dominates(g->mode, new_mode)) {
            counts[LMD_C_CONVERT_NOOPS] += 1;
            emit(LMD_EV_CONVERT_NOOP, txn_id, &res, new_mode, 0);
            return;
        }
        int eff = lmd_lub(g->mode, new_mode);
        if (compatible_with_others(res, txn_id, eff)) {
            g->mode = eff;
            g->last_convert_seq = event_seq;
            if (res.kind == LMD_ROW || res.kind == LMD_PARTITION) recompute_counters(tx);
            tx.locks_acquired += 1;
            counts[LMD_C_LOCK_CONVERTS] += 1;
            emit(LMD_EV_LOCK_CONVERT, txn_id, &res, eff, 0);
        } else {
            uint64_t rs = request_seq_next++;
            enqueue_waiter(tx, res, new_mode, /*plan_index=*/0, rs,
                           /*is_conversion=*/1, res, new_mode);
            counts[LMD_C_CONVERT_WAITS] += 1;
            emit(LMD_EV_CONVERT_WAIT, txn_id, &res, new_mode, 0);
        }
    }

    // -------------------------------------------------------------- BEGIN
    void op_begin(uint64_t txn_id, uint32_t priority) {
        if (find_txn(txn_id) || (int)txns.size() >= spec.max_txns) { invalid(); return; }
        Txn t;
        t.txn_id = txn_id;
        t.txn_seq = txn_seq_next++;
        t.priority = priority;
        t.status = LMD_ACTIVE;
        t.has_blocked = false;
        t.wait_seq = U64MAX;
        t.tctr.assign((size_t)spec.table_count, TableCtr{});
        txns[txn_id] = t;
        counts[LMD_C_TXN_BEGUN] += 1;
        emit(LMD_EV_TXN_BEGIN, txn_id, nullptr, -1, priority);
    }

    // -------------------------------------------------------------- wake
    // For each affected resource in canonical order, process its wait queue from
    // the head per the wake rule.
    void wake_resources(std::vector<LmdRes>& affected) {
        // dedup + canonical order
        std::sort(affected.begin(), affected.end(), LmdResCanonLess());
        affected.erase(std::unique(affected.begin(), affected.end(),
                       [](const LmdRes& a, const LmdRes& b){ return a == b; }), affected.end());
        for (size_t ri = 0; ri < affected.size(); ++ri) {
            const LmdRes res_copy = affected[ri];  // stable copy (vector may grow)
            wake_one_resource(res_copy, affected);
        }
    }

    bool is_stale_waiter(const Waiter& w) {
        return find_txn(w.txn) == nullptr;
    }

    // res is taken BY VALUE: continue_plan may push_back into affected_out and
    // reallocate it, which would otherwise dangle a reference aliasing it.
    void wake_one_resource(LmdRes res, std::vector<LmdRes>& affected_out) {
        auto wit = waits.find(res);
        if (wit == waits.end()) return;
        std::vector<Waiter>& q = wit->second;
        while (!q.empty()) {
            Waiter head = q.front();
            if (is_stale_waiter(head)) { q.erase(q.begin()); continue; }
            Txn* tp = find_txn(head.txn);
            // compatibility of head request with current granted locks (others).
            int eff_req = head.requested_mode;
            Grant* hg = find_grant(res, head.txn);
            if (grant_exists(hg)) eff_req = lmd_lub(hg->mode, head.requested_mode);
            if (!compatible_with_others(res, head.txn, eff_req)) {
                // cannot grant head; stop on this resource.
                return;
            }
            // grant or convert head.
            q.erase(q.begin());
            Txn& tx = *tp;
            bool was_conv = head.is_conversion != 0;
            // perform the grant/convert at this resource.
            bool is_target = false;
            if (was_conv) {
                is_target = true;  // conversion always targets the held resource.
            } else {
                // determine from plan: the entry is target iff plan_index is last.
                std::vector<PlanEntry> plan = build_plan(head.original_target_res, head.original_target_mode);
                is_target = ((size_t)head.request_plan_index + 1 == plan.size());
            }
            grant_woken(tx, res, head.requested_mode, is_target);
            counts[LMD_C_WAKE_GRANTS] += 1;
            emit(LMD_EV_WAKE_GRANT, tx.txn_id, &res, head.requested_mode, 0);
            // clear blocked fields, set ACTIVE.
            tx.status = LMD_ACTIVE;
            tx.has_blocked = false;
            tx.blocked_mode = LMD_NL;
            tx.wait_seq = U64MAX;
            // descendant counters / escalation for woken target grant.
            if (is_target && (res.kind == LMD_ROW || res.kind == LMD_PARTITION)) {
                if (fine_count(tx, res.t) >= (uint64_t)spec.escalation_threshold) {
                    try_escalate(tx, res.t);
                }
            }
            // continue the plan if this was a non-conversion wait with remaining entries.
            if (!was_conv) {
                std::vector<PlanEntry> plan = build_plan(head.original_target_res, head.original_target_mode);
                if ((size_t)head.request_plan_index + 1 < plan.size()) {
                    continue_plan(tx, plan, head.request_plan_index + 1, head.request_seq,
                                  head.original_target_res, head.original_target_mode, affected_out);
                    // continue_plan either finished or reblocked; either way we
                    // continue scanning this resource's queue for the next head.
                }
            }
        }
        if (q.empty()) waits.erase(wit);
    }

    void grant_woken(Txn& tx, const LmdRes& res, int req_mode, bool is_target) {
        GrantKey key{res, tx.txn_id};
        auto it = grants.find(key);
        if (it == grants.end() || !grant_exists(&it->second)) {
            Grant g;
            g.mode = req_mode;
            g.explicit_count = is_target ? 1 : 0;
            g.auto_count = is_target ? 0 : 1;
            g.grant_seq = event_seq;
            g.last_convert_seq = 0;
            grants[key] = g;
            tx.locks_acquired += 1;
            if (is_target) recompute_counters(tx);
        } else {
            Grant& g = it->second;
            int old = g.mode;
            int nm = lmd_lub(old, req_mode);
            (void)old;
            g.mode = nm;
            if (is_target) g.explicit_count += 1; else g.auto_count += 1;
            g.last_convert_seq = event_seq;
            if (is_target) recompute_counters(tx);
            tx.locks_acquired += 1;
        }
    }

    // continue acquisition plan from index `from`. If it blocks again, enqueue
    // with same request_seq + new wait_seq, emit WAKE_REBLOCK, and stop.
    void continue_plan(Txn& tx, const std::vector<PlanEntry>& plan, int from,
                       uint64_t request_seq, const LmdRes& orig_target, int orig_mode,
                       std::vector<LmdRes>& affected_out) {
        for (size_t pi = (size_t)from; pi < plan.size(); ++pi) {
            const PlanEntry& e = plan[pi];
            Grant* g = find_grant(e.res, tx.txn_id);
            if (grant_exists(g) && lmd_dominates(g->mode, e.mode)) {
                if (e.is_target) g->explicit_count += 1; else g->auto_count += 1;
                counts[LMD_C_LOCK_REENTERS] += 1;
                emit(LMD_EV_LOCK_REENTER, tx.txn_id, &e.res, e.mode, 0);
                continue;
            }
            int eff_req = e.mode;
            if (grant_exists(g)) eff_req = lmd_lub(g->mode, e.mode);
            if (compatible_with_others(e.res, tx.txn_id, eff_req)) {
                do_grant_or_convert(tx, e.res, e.mode, e.is_target);
                affected_out.push_back(e.res);
                if (e.is_target && (e.res.kind == LMD_ROW || e.res.kind == LMD_PARTITION)) {
                    if (fine_count(tx, e.res.t) >= (uint64_t)spec.escalation_threshold) {
                        try_escalate(tx, e.res.t);
                    }
                }
            } else {
                // reblock at this resource with same request_seq, new wait_seq.
                Waiter w;
                w.txn = tx.txn_id;
                w.requested_mode = e.mode;
                w.request_plan_index = (int)pi;
                w.request_seq = request_seq;
                w.is_conversion = 0;
                w.original_target_res = orig_target;
                w.original_target_mode = orig_mode;
                w.wait_seq = wait_seq_next++;
                waits[e.res].push_back(w);
                tx.status = LMD_WAITING;
                tx.blocked_resource = e.res;
                tx.has_blocked = true;
                tx.blocked_mode = e.mode;
                tx.wait_seq = w.wait_seq;
                counts[LMD_C_WAKE_REBLOCKS] += 1;
                emit(LMD_EV_WAKE_REBLOCK, tx.txn_id, &e.res, e.mode, 0);
                return;
            }
        }
    }

    // -------------------------------------------------------------- deadlock
    // Build wait-for graph and find first cycle deterministically. Returns the
    // cycle as a list of txn_ids (the stack suffix), or empty if none.
    std::vector<uint64_t> find_first_cycle() {
        // adjacency: for each waiting txn, edges to blockers (sorted asc).
        // waiting txns: those with status WAITING (in some wait queue).
        std::vector<uint64_t> waiting_ids;
        for (auto& kv : txns) {
            if (kv.second.status == LMD_WAITING && kv.second.has_blocked) {
                waiting_ids.push_back(kv.first);
            }
        }
        std::sort(waiting_ids.begin(), waiting_ids.end());

        // successors(u): blockers of u on its blocked_resource whose granted
        // lock is incompatible with u's requested mode, sorted asc by txn_id.
        auto succ = [&](uint64_t u) -> std::vector<uint64_t> {
            std::vector<uint64_t> out;
            Txn* tu = find_txn(u);
            if (!tu || tu->status != LMD_WAITING || !tu->has_blocked) return out;
            const LmdRes& res = tu->blocked_resource;
            int req = tu->blocked_mode;
            for (auto it = grants.lower_bound(GrantKey{res, 0}); it != grants.end(); ++it) {
                if (!(it->first.res == res)) break;
                uint64_t b = it->first.txn;
                if (b == u) continue;
                if (!grant_exists(&it->second)) continue;
                if (!lmd_compat(req, it->second.mode)) out.push_back(b);
            }
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            return out;
        };

        // DFS from each waiting txn in ascending order, find first cycle.
        for (uint64_t start : waiting_ids) {
            std::vector<uint64_t> stack;
            std::vector<int> on_stack_marker;  // parallel; we use a helper.
            // iterative DFS preserving "successors ascending" with first cycle.
            std::vector<uint64_t> cyc = dfs_cycle(start, succ);
            if (!cyc.empty()) return cyc;
        }
        return {};
    }

    // recursive DFS that returns the first cycle found (stack suffix) or empty.
    std::vector<uint64_t> dfs_cycle(uint64_t start,
                                    const std::function<std::vector<uint64_t>(uint64_t)>& succ) {
        std::vector<uint64_t> stack;
        std::vector<uint64_t> result;
        std::vector<uint64_t> path;
        // explicit stack of (node, next-child-index)
        struct Frame { uint64_t node; std::vector<uint64_t> ch; size_t idx; };
        std::vector<Frame> fr;
        auto on_path = [&](uint64_t n) -> int {
            for (size_t i = 0; i < path.size(); ++i) if (path[i] == n) return (int)i;
            return -1;
        };
        fr.push_back(Frame{start, succ(start), 0});
        path.push_back(start);
        while (!fr.empty()) {
            Frame& top = fr.back();
            if (top.idx < top.ch.size()) {
                uint64_t nxt = top.ch[top.idx++];
                int pos = on_path(nxt);
                if (pos >= 0) {
                    // cycle = path suffix from pos..end
                    std::vector<uint64_t> cyc(path.begin() + pos, path.end());
                    return cyc;
                }
                fr.push_back(Frame{nxt, succ(nxt), 0});
                path.push_back(nxt);
            } else {
                fr.pop_back();
                path.pop_back();
            }
        }
        return {};
    }

    void abort_victim(uint64_t victim) {
        Txn* tp = find_txn(victim);
        if (!tp) return;
        // remove from wait queue if waiting.
        if (tp->status == LMD_WAITING && tp->has_blocked) {
            auto wit = waits.find(tp->blocked_resource);
            if (wit != waits.end()) {
                std::vector<Waiter>& q = wit->second;
                for (auto jt = q.begin(); jt != q.end(); ++jt) {
                    if (jt->txn == victim && jt->wait_seq == tp->wait_seq) { q.erase(jt); break; }
                }
                if (q.empty()) waits.erase(wit);
            }
        }
        // release all locks in RELEASE order.
        std::vector<LmdRes> rels;
        for (auto it = grants.begin(); it != grants.end(); ++it) {
            if (it->first.txn != victim) continue;
            if (!grant_exists(&it->second)) continue;
            rels.push_back(it->first.res);
        }
        std::sort(rels.begin(), rels.end(), LmdResReleaseLess());
        for (const LmdRes& r : rels) {
            Grant* g = find_grant(r, victim);
            g->explicit_count = 0; g->auto_count = 0; g->mode = LMD_NL;
            counts[LMD_C_VICTIM_RELEASES] += 1;
            emit(LMD_EV_VICTIM_RELEASE, victim, &r, -1, 0);
        }
        // remove transaction, emit DEADLOCK_ABORT.
        txns.erase(victim);
        counts[LMD_C_DEADLOCK_ABORTS] += 1;
        emit(LMD_EV_DEADLOCK_ABORT, victim, nullptr, -1, 0);
        (void)(deadlock_seq_next++);
        // wake affected waiters after the release cascade.
        std::vector<LmdRes> affected = rels;
        wake_resources(affected);
    }

    void op_detect(int limit) {
        if (limit == 0) return;  // valid no-op
        int repeat = std::min(limit, spec.max_deadlock_cycles_per_detect);
        for (int iter = 0; iter < repeat; ++iter) {
            std::vector<uint64_t> cyc = find_first_cycle();
            if (cyc.empty()) {
                counts[LMD_C_DEADLOCK_NONE] += 1;
                emit(LMD_EV_DEADLOCK_NONE, 0, nullptr, -1, 0);
                return;
            }
            // victim selection: lowest priority; tie largest locks_acquired;
            // tie largest txn_seq; tie largest txn_id.
            uint64_t best = 0; bool have = false;
            uint32_t b_prio = 0; uint64_t b_locks = 0, b_seq = 0, b_id = 0;
            for (uint64_t id : cyc) {
                Txn* t = find_txn(id);
                if (!t) continue;
                bool take = false;
                if (!have) take = true;
                else if (t->priority < b_prio) take = true;
                else if (t->priority == b_prio) {
                    if (t->locks_acquired > b_locks) take = true;
                    else if (t->locks_acquired == b_locks) {
                        if (t->txn_seq > b_seq) take = true;
                        else if (t->txn_seq == b_seq) {
                            if (t->txn_id > b_id) take = true;
                        }
                    }
                }
                if (take) {
                    have = true; best = id;
                    b_prio = t->priority; b_locks = t->locks_acquired;
                    b_seq = t->txn_seq; b_id = t->txn_id;
                }
            }
            if (!have) {
                counts[LMD_C_DEADLOCK_NONE] += 1;
                emit(LMD_EV_DEADLOCK_NONE, 0, nullptr, -1, 0);
                return;
            }
            abort_victim(best);
        }
    }

    // -------------------------------------------------------------- snapshots
    uint64_t compute_grant_hash() const {
        LmdFnv f;
        // grants map is keyed (res canonical, txn) so iteration order is already
        // resource-canonical then txn ascending.
        for (auto it = grants.begin(); it != grants.end(); ++it) {
            if (!(it->second.explicit_count != 0 || it->second.auto_count != 0)) continue;
            const LmdRes& r = it->first.res;
            f.u8((uint8_t)r.kind);
            f.u32((uint32_t)r.t);
            f.u32(r.p < 0 ? U32MAX : (uint32_t)r.p);
            f.u32(r.r < 0 ? U32MAX : (uint32_t)r.r);
            f.u64(it->first.txn);
            f.u8((uint8_t)it->second.mode);
            f.u64(it->second.explicit_count);
            f.u64(it->second.auto_count);
            f.u64(it->second.grant_seq);
            f.u64(it->second.last_convert_seq);
        }
        return f.h;
    }

    uint64_t compute_wait_hash() const {
        LmdFnv f;
        // waits map keyed by resource canonical; each vector ordered by queue
        // position (which equals wait_seq ascending by construction).
        for (auto it = waits.begin(); it != waits.end(); ++it) {
            const LmdRes& r = it->first;
            uint64_t pos = 0;
            for (const Waiter& w : it->second) {
                f.u8((uint8_t)r.kind);
                f.u32((uint32_t)r.t);
                f.u32(r.p < 0 ? U32MAX : (uint32_t)r.p);
                f.u32(r.r < 0 ? U32MAX : (uint32_t)r.r);
                f.u64(pos++);
                f.u64(w.txn);
                f.u8((uint8_t)w.requested_mode);
                f.u64(w.wait_seq);
                f.u64(w.request_seq);
                f.u8(w.is_conversion);
                const LmdRes& o = w.original_target_res;
                f.u8((uint8_t)o.kind);
                f.u32((uint32_t)o.t);
                f.u32(o.p < 0 ? U32MAX : (uint32_t)o.p);
                f.u32(o.r < 0 ? U32MAX : (uint32_t)o.r);
                f.u8((uint8_t)w.original_target_mode);
            }
        }
        return f.h;
    }

    uint64_t compute_txn_hash() const {
        LmdFnv f;
        // txns map keyed by txn_id ascending.
        for (auto it = txns.begin(); it != txns.end(); ++it) {
            const Txn& t = it->second;
            f.u64(t.txn_id);
            f.u64(t.txn_seq);
            f.u32(t.priority);
            f.u8((uint8_t)t.status);
            if (t.has_blocked) {
                const LmdRes& r = t.blocked_resource;
                f.u8((uint8_t)r.kind);
                f.u32((uint32_t)r.t);
                f.u32(r.p < 0 ? U32MAX : (uint32_t)r.p);
                f.u32(r.r < 0 ? U32MAX : (uint32_t)r.r);
                f.u8((uint8_t)t.blocked_mode);
            } else {
                f.u8((uint8_t)255);
                f.u32(U32MAX); f.u32(U32MAX); f.u32(U32MAX);
                f.u8((uint8_t)255);
            }
            f.u64(t.locks_acquired);
            f.u64(t.deadlock_aborts);
            for (size_t ti = 0; ti < t.tctr.size(); ++ti) {
                const TableCtr& c = t.tctr[ti];
                f.u32((uint32_t)ti);
                f.u64(c.row_s_count);
                f.u64(c.row_x_count);
                f.u64(c.partition_s_count);
                f.u64(c.partition_x_count);
                f.u64(c.escalation_attempts);
            }
        }
        return f.h;
    }

    uint64_t compute_state_checksum(uint64_t gh, uint64_t wh, uint64_t th) const {
        LmdFnv f;
        f.u64(event_seq);
        f.u64(txn_seq_next);
        f.u64(request_seq_next);
        f.u64(wait_seq_next);
        f.u64(deadlock_seq_next);
        f.u32((uint32_t)op_index);
        f.u64(event_hash);
        f.u64(gh);
        f.u64(wh);
        f.u64(th);
        for (int i = 0; i < LMD_COUNT_N; ++i) f.u64((uint64_t)counts[(size_t)i]);
        return f.h;
    }

    // -------------------------------------------------------------- step
    void step_once(const LmdRunSpec& run, LmdExpected* exp) {
        LmdRes res{run.a_res_kind, run.a_table,
                   run.a_partition, run.a_row};
        switch (run.op_kind) {
            case LMD_OP_BEGIN:
                op_begin(run.a_txn, (uint32_t)run.a_priority);
                break;
            case LMD_OP_LOCK:
                op_lock(run.a_txn, res, run.a_mode);
                break;
            case LMD_OP_UNLOCK:
                op_unlock(run.a_txn, res);
                break;
            case LMD_OP_UNLOCK_ALL:
                op_unlock_all(run.a_txn);
                break;
            case LMD_OP_CONVERT:
                op_convert(run.a_txn, res, run.a_mode);
                break;
            case LMD_OP_DETECT_DEADLOCK:
                op_detect(run.a_limit);
                break;
            default:
                break;
        }
        const int32_t this_op = op_index;
        op_index += 1;

        const uint64_t gh = compute_grant_hash();
        const uint64_t wh = compute_wait_hash();
        const uint64_t th = compute_txn_hash();

        exp->counts = counts;
        exp->op_index = this_op;
        exp->event_seq = event_seq;
        exp->lock_event_hash = event_hash;
        exp->grant_hash = gh;
        exp->wait_hash = wh;
        exp->txn_lock_hash = th;
        exp->state_checksum = compute_state_checksum(gh, wh, th);
    }
};

static inline bool lmd_check_outputs(const LmdExpected& e, const LmdHostOutputsView& g, std::string* err) {
    for (int i = 0; i < LMD_COUNT_N; ++i) {
        if (g.counts[i] != e.counts[(size_t)i]) {
            if (err) { std::ostringstream o; o << "count[" << i << "] mismatch: got " << g.counts[i] << " expected " << e.counts[(size_t)i]; *err = o.str(); }
            return false;
        }
    }
    auto chk64 = [&](const char* nm, uint64_t got, uint64_t exp) -> bool {
        if (got != exp) { if (err) { std::ostringstream o; o << nm << " mismatch: got 0x" << std::hex << got << " expected 0x" << exp; *err = o.str(); } return false; }
        return true;
    };
    if (g.op_index_out[0] != e.op_index) {
        if (err) { std::ostringstream o; o << "op_index mismatch got " << g.op_index_out[0] << " exp " << e.op_index; *err = o.str(); }
        return false;
    }
    if (!chk64("event_seq", g.event_seq_out[0], e.event_seq)) return false;
    if (!chk64("lock_event_hash", g.lock_event_hash[0], e.lock_event_hash)) return false;
    if (!chk64("grant_hash", g.grant_hash[0], e.grant_hash)) return false;
    if (!chk64("wait_hash", g.wait_hash[0], e.wait_hash)) return false;
    if (!chk64("txn_lock_hash", g.txn_lock_hash[0], e.txn_lock_hash)) return false;
    if (!chk64("state_checksum", g.state_checksum[0], e.state_checksum)) return false;
    return true;
}

#endif  // LOCK_MANAGER_DEADLOCK_ORACLE_HPP_
