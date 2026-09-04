// file: raft_log_snapshot_oracle.hpp

#ifndef RAFT_LOG_SNAPSHOT_ORACLE_HPP_
#define RAFT_LOG_SNAPSHOT_ORACLE_HPP_

#include "raft_log_snapshot_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// FNV-1a-64 helpers. All fields hashed by exact declared width, little-endian,
// so the host oracle and the two device solutions agree byte-for-byte.
// ---------------------------------------------------------------------------
static inline uint64_t raft_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}
static inline void raft_fnv_u8(uint64_t* h, uint8_t v) {
    *h = raft_fnv_byte(*h, v);
}
static inline void raft_fnv_u32(uint64_t* h, uint32_t v) {
    uint64_t hh = *h;
    for (int i = 0; i < 4; ++i) hh = raft_fnv_byte(hh, static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
    *h = hh;
}
static inline void raft_fnv_u64(uint64_t* h, uint64_t v) {
    uint64_t hh = *h;
    for (int i = 0; i < 8; ++i) hh = raft_fnv_byte(hh, static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
    *h = hh;
}
static inline void raft_fnv_i64(uint64_t* h, int64_t v) { raft_fnv_u64(h, static_cast<uint64_t>(v)); }

#define RAFT_FNV_INIT 1469598103934665603ULL
#define RAFT_U32_MAX 0xFFFFFFFFu
#define RAFT_U64_MAX 0xFFFFFFFFFFFFFFFFull

struct RaftLogEntry {
    uint64_t index;
    uint64_t term;
    uint64_t command_id;
    int64_t payload;
};

struct RaftPendingRpc {
    uint64_t rpc_id;
    uint32_t leader;
    uint32_t follower;
    uint64_t leader_term;
    uint64_t prev_index;
    uint64_t prev_term;
    uint64_t leader_commit;
    uint64_t send_seq;
    std::vector<RaftLogEntry> entries;  // copied, ascending index
};

struct RaftExpected {
    RaftCounts counts{};
    uint64_t event_hash = 0;
    uint64_t log_hash = 0;
    uint64_t leader_state_hash = 0;
    uint64_t pending_rpc_hash = 0;
    uint64_t apply_hash = 0;
};

struct RaftServer {
    uint64_t current_term = 0;
    int32_t role = RAFT_ROLE_FOLLOWER;
    uint64_t snapshot_index = 0;
    uint64_t snapshot_term = 0;
    uint64_t snapshot_state_hash = 0;
    std::vector<RaftLogEntry> log;   // index > snapshot_index, ascending
    uint64_t commit_index = 0;
    uint64_t last_applied = 0;
    uint64_t apply_accumulator = 0;
};

struct RaftOracleState {
    RaftProblemSpec spec{};

    uint64_t event_seq = 0;
    uint64_t rpc_seq_next = 1;
    bool has_leader = false;
    uint32_t leader_id = 0;

    std::vector<RaftServer> servers;
    // Leader volatile state, indexed [leader_slot? no -> single live leader]:
    // we store next/match per server for the *current* leader only.
    std::vector<uint64_t> next_index;
    std::vector<uint64_t> match_index;

    std::vector<RaftPendingRpc> pending;  // unordered insert; hashed by rpc_id asc

    RaftCounts counts{};
    uint64_t event_hash = RAFT_FNV_INIT;

    void init(const RaftProblemSpec& s) {
        spec = s;
        reset();
    }

    void reset() {
        event_seq = 0;
        rpc_seq_next = 1;
        has_leader = false;
        leader_id = 0;
        servers.assign((size_t)spec.server_count, RaftServer{});
        next_index.assign((size_t)spec.server_count, 0);
        match_index.assign((size_t)spec.server_count, 0);
        pending.clear();
        counts = RaftCounts{};
        event_hash = RAFT_FNV_INIT;
    }

    int majority() const { return spec.server_count / 2 + 1; }

    uint64_t last_log_index(int s) const {
        const RaftServer& sv = servers[(size_t)s];
        if (sv.log.empty()) return sv.snapshot_index;
        return sv.log.back().index;
    }

    // Returns pointer to log entry with given index, or nullptr.
    const RaftLogEntry* log_at(int s, uint64_t index) const {
        const RaftServer& sv = servers[(size_t)s];
        for (const RaftLogEntry& e : sv.log) {
            if (e.index == index) return &e;
        }
        return nullptr;
    }

    // -----------------------------------------------------------------------
    // Event emission: emits into raft_event_hash in emission order. Each event
    // consumes one event_seq (post-increment: the emitted event carries the
    // current event_seq, then event_seq advances).
    // -----------------------------------------------------------------------
    void emit(uint8_t kind, uint32_t op_index, uint32_t server, uint32_t peer,
              uint64_t term, uint64_t index_or_max, uint64_t count_or_zero, uint64_t aux) {
        uint64_t es = event_seq;
        raft_fnv_u8(&event_hash, kind);
        raft_fnv_u64(&event_hash, es);
        raft_fnv_u32(&event_hash, op_index);
        raft_fnv_u32(&event_hash, server);
        raft_fnv_u32(&event_hash, peer);
        raft_fnv_u64(&event_hash, term);
        raft_fnv_u64(&event_hash, index_or_max);
        raft_fnv_u64(&event_hash, count_or_zero);
        raft_fnv_u64(&event_hash, aux);
        event_seq += 1;
    }

    // -----------------------------------------------------------------------
    // Operations
    // -----------------------------------------------------------------------
    void op_become_leader(uint32_t op_index, int server, uint64_t term) {
        if (server < 0 || server >= spec.server_count ||
            term < servers[(size_t)server].current_term) {
            counts.invalid_count += 1;
            emit(RAFT_EV_INVALID, op_index, RAFT_U32_MAX, RAFT_U32_MAX, 0,
                 RAFT_U64_MAX, 0, RAFT_OP_BECOME_LEADER);
            return;
        }
        for (int s = 0; s < spec.server_count; ++s) {
            if (s != server && servers[(size_t)s].role == RAFT_ROLE_LEADER) {
                servers[(size_t)s].role = RAFT_ROLE_FOLLOWER;
            }
        }
        servers[(size_t)server].current_term = term;
        servers[(size_t)server].role = RAFT_ROLE_LEADER;
        has_leader = true;
        leader_id = (uint32_t)server;

        const uint64_t lli = last_log_index(server);
        for (int f = 0; f < spec.server_count; ++f) {
            if (f == server) continue;
            next_index[(size_t)f] = lli + 1;
            match_index[(size_t)f] = 0;
        }
        match_index[(size_t)server] = lli;
        next_index[(size_t)server] = lli + 1;

        counts.leaders_elected += 1;
        emit(RAFT_EV_BECOME_LEADER_OK, op_index, (uint32_t)server, RAFT_U32_MAX,
             term, lli, 0, 0);
    }

    void op_client_append(uint32_t op_index, uint64_t command_id, int64_t payload) {
        if (!has_leader) {
            counts.invalid_count += 1;
            emit(RAFT_EV_INVALID, op_index, RAFT_U32_MAX, RAFT_U32_MAX, 0,
                 RAFT_U64_MAX, 0, RAFT_OP_CLIENT_APPEND);
            return;
        }
        int L = (int)leader_id;
        RaftServer& sv = servers[(size_t)L];
        if ((int)sv.log.size() >= spec.max_log_entries_per_server) {
            counts.client_rejected += 1;
            emit(RAFT_EV_CLIENT_REJECT_LOG_FULL, op_index, (uint32_t)L, RAFT_U32_MAX,
                 sv.current_term, RAFT_U64_MAX, 0, 0);
            return;
        }
        const uint64_t idx = last_log_index(L) + 1;
        RaftLogEntry e{idx, sv.current_term, command_id, payload};
        sv.log.push_back(e);
        match_index[(size_t)L] = last_log_index(L);
        next_index[(size_t)L] = last_log_index(L) + 1;

        counts.client_appended += 1;
        emit(RAFT_EV_CLIENT_APPEND_OK, op_index, (uint32_t)L, RAFT_U32_MAX,
             sv.current_term, idx, 0, command_id);
    }

    void op_send_append(uint32_t op_index, int follower, int max_entries) {
        if (!has_leader || follower < 0 || follower >= spec.server_count ||
            follower == (int)leader_id || max_entries == 0) {
            counts.invalid_count += 1;
            emit(RAFT_EV_INVALID, op_index, RAFT_U32_MAX, RAFT_U32_MAX, 0,
                 RAFT_U64_MAX, 0, RAFT_OP_SEND_APPEND);
            return;
        }
        int L = (int)leader_id;
        if ((int)pending.size() >= spec.max_pending_append_rpcs) {
            counts.append_send_rejected += 1;
            emit(RAFT_EV_APPEND_SEND_REJECT, op_index, (uint32_t)L, (uint32_t)follower,
                 servers[(size_t)L].current_term, RAFT_U64_MAX, 0, 0);
            return;
        }
        RaftServer& Ls = servers[(size_t)L];
        const uint64_t ni = next_index[(size_t)follower];
        if (ni <= Ls.snapshot_index) {
            counts.append_needs_snapshot += 1;
            emit(RAFT_EV_APPEND_NEEDS_SNAPSHOT, op_index, (uint32_t)L, (uint32_t)follower,
                 Ls.current_term, Ls.snapshot_index, 0, 0);
            return;
        }
        const uint64_t prev_index = ni - 1;
        uint64_t prev_term;
        if (prev_index == Ls.snapshot_index) {
            prev_term = Ls.snapshot_term;
        } else {
            const RaftLogEntry* pe = log_at(L, prev_index);
            if (pe == nullptr) {
                counts.append_send_rejected += 1;
                emit(RAFT_EV_APPEND_SEND_REJECT, op_index, (uint32_t)L, (uint32_t)follower,
                     Ls.current_term, RAFT_U64_MAX, 0, 0);
                return;
            }
            prev_term = pe->term;
        }

        int cap = max_entries;
        if (cap > spec.max_entries_per_append) cap = spec.max_entries_per_append;

        std::vector<RaftLogEntry> copied;
        for (const RaftLogEntry& e : Ls.log) {
            if (e.index >= ni) {
                copied.push_back(e);
                if ((int)copied.size() >= cap) break;
            }
        }

        RaftPendingRpc r;
        r.rpc_id = rpc_seq_next++;
        r.leader = (uint32_t)L;
        r.follower = (uint32_t)follower;
        r.leader_term = Ls.current_term;
        r.prev_index = prev_index;
        r.prev_term = prev_term;
        r.leader_commit = Ls.commit_index;
        r.send_seq = event_seq;  // event_seq of this APPEND_SEND
        r.entries = copied;
        pending.push_back(r);

        counts.append_sent += 1;
        emit(RAFT_EV_APPEND_SEND, op_index, (uint32_t)L, (uint32_t)follower,
             Ls.current_term, prev_index, (uint64_t)copied.size(), r.rpc_id);
    }

    // first index in follower log with given term, else UINT64_MAX
    uint64_t first_index_with_term(int f, uint64_t term) const {
        const RaftServer& sv = servers[(size_t)f];
        for (const RaftLogEntry& e : sv.log) {
            if (e.term == term) return e.index;
        }
        return RAFT_U64_MAX;
    }

    // 1 + largest leader index with given term, or 0 if leader has none.
    uint64_t leader_last_plus_one_with_term(int L, uint64_t term) const {
        const RaftServer& sv = servers[(size_t)L];
        uint64_t best = 0;
        bool found = false;
        for (const RaftLogEntry& e : sv.log) {
            if (e.term == term) { best = e.index; found = true; }
        }
        if (!found) return 0;
        return best + 1;
    }

    void op_deliver_append(uint32_t op_index, uint64_t rpc_id) {
        // find pending
        int pos = -1;
        for (size_t i = 0; i < pending.size(); ++i) {
            if (pending[i].rpc_id == rpc_id) { pos = (int)i; break; }
        }
        if (pos < 0) {
            counts.invalid_count += 1;
            emit(RAFT_EV_INVALID, op_index, RAFT_U32_MAX, RAFT_U32_MAX, 0,
                 RAFT_U64_MAX, 0, RAFT_OP_DELIVER_APPEND);
            return;
        }
        // Remove exactly once.
        RaftPendingRpc R = pending[(size_t)pos];
        pending.erase(pending.begin() + pos);

        // Stale leader check.
        if (!(has_leader && leader_id == R.leader) ||
            servers[(size_t)R.leader].current_term != R.leader_term) {
            counts.append_stale += 1;
            emit(RAFT_EV_APPEND_STALE, op_index, R.leader, R.follower,
                 R.leader_term, R.rpc_id, 0, 0);
            return;
        }

        int F = (int)R.follower;
        RaftServer& Fs = servers[(size_t)F];

        if (Fs.current_term > R.leader_term) {
            counts.append_term_reject += 1;
            // next_index[F] = max(1, next_index[F]-1)
            uint64_t ni = next_index[(size_t)F];
            uint64_t newni = (ni > 1) ? (ni - 1) : 1;
            next_index[(size_t)F] = newni;
            emit(RAFT_EV_APPEND_TERM_REJECT, op_index, R.leader, R.follower,
                 R.leader_term, R.prev_index, 0, newni);
            return;
        }

        Fs.current_term = R.leader_term;

        // Consistency check.
        bool consistent = false;
        bool fail = false;
        uint64_t conflict_index = 0;
        uint64_t conflict_term = RAFT_U64_MAX;  // sentinel "no conflict term"
        bool has_conflict_term = false;

        if (R.prev_index < Fs.snapshot_index) {
            fail = true;
            conflict_index = Fs.snapshot_index + 1;
        } else if (R.prev_index == Fs.snapshot_index) {
            if (R.prev_term == Fs.snapshot_term) {
                consistent = true;
            } else {
                // boundary term mismatch at snapshot index: treat as absent prev
                // (no follower log entry exists at snapshot_index).
                fail = true;
                conflict_index = last_log_index(F) + 1;
            }
        } else {
            const RaftLogEntry* pe = log_at(F, R.prev_index);
            if (pe == nullptr) {
                fail = true;
                conflict_index = last_log_index(F) + 1;
            } else if (pe->term == R.prev_term) {
                consistent = true;
            } else {
                fail = true;
                has_conflict_term = true;
                conflict_term = pe->term;
                conflict_index = first_index_with_term(F, pe->term);
            }
        }

        if (fail) {
            uint64_t newni;
            if (has_conflict_term) {
                uint64_t lt = leader_last_plus_one_with_term((int)R.leader, conflict_term);
                if (lt != 0) {
                    newni = lt;
                } else {
                    newni = conflict_index;
                }
            } else {
                newni = conflict_index;
            }
            next_index[(size_t)F] = newni;
            counts.append_conflict += 1;
            emit(RAFT_EV_APPEND_CONFLICT, op_index, R.leader, R.follower,
                 R.leader_term, conflict_index,
                 has_conflict_term ? conflict_term : RAFT_U64_MAX, newni);
            return;
        }

        (void)consistent;

        // Consistency success. First, OOM pre-check: simulate deletes+appends
        // and verify final log size <= cap, else APPEND_FOLLOWER_OOM and do
        // nothing.
        // Determine the deletion: scan incoming in ascending index; the first
        // incoming entry whose index already exists in follower with a
        // different term triggers suffix deletion from that index.
        // After deletion the follower keeps entries with index < that index.
        // Then append all incoming entries with index not already present.

        // Compute first conflicting index (different term at same index).
        uint64_t del_from = RAFT_U64_MAX;
        for (const RaftLogEntry& ie : R.entries) {
            const RaftLogEntry* fe = log_at(F, ie.index);
            if (fe != nullptr && fe->term != ie.term) { del_from = ie.index; break; }
        }

        // Build candidate final log size.
        // Count survivors after deletion.
        size_t survivors = 0;
        uint64_t deleted_count = 0;
        uint64_t first_deleted_index = 0;
        if (del_from != RAFT_U64_MAX) {
            for (const RaftLogEntry& e : Fs.log) {
                if (e.index < del_from) survivors += 1;
                else { if (deleted_count == 0) first_deleted_index = e.index; deleted_count += 1; }
            }
        } else {
            survivors = Fs.log.size();
        }
        // Which incoming entries would be appended (index > snapshot_index and
        // not already present after deletion). After deletion, present indices
        // are those < del_from (if del happened) or all current ones.
        std::vector<const RaftLogEntry*> to_append;
        for (const RaftLogEntry& ie : R.entries) {
            if (ie.index <= Fs.snapshot_index) continue;  // covered by snapshot
            bool present;
            if (del_from != RAFT_U64_MAX) {
                present = (ie.index < del_from);  // survivors keep these
            } else {
                present = (log_at(F, ie.index) != nullptr);
            }
            if (!present) to_append.push_back(&ie);
        }

        size_t final_size = survivors + to_append.size();
        if ((int)final_size > spec.max_log_entries_per_server) {
            counts.append_follower_oom += 1;
            emit(RAFT_EV_APPEND_FOLLOWER_OOM, op_index, R.leader, R.follower,
                 R.leader_term, R.prev_index, (uint64_t)final_size, 0);
            return;
        }

        // Apply deletion.
        if (del_from != RAFT_U64_MAX && deleted_count > 0) {
            std::vector<RaftLogEntry> kept;
            for (const RaftLogEntry& e : Fs.log) {
                if (e.index < del_from) kept.push_back(e);
            }
            Fs.log.swap(kept);
            counts.follower_deleted_suffixes += 1;
            emit(RAFT_EV_FOLLOWER_DELETE_SUFFIX, op_index, R.leader, R.follower,
                 R.leader_term, first_deleted_index, deleted_count, 0);
        }

        // Append.
        for (const RaftLogEntry* ie : to_append) {
            Fs.log.push_back(*ie);
            counts.follower_appended_entries += 1;
            emit(RAFT_EV_FOLLOWER_APPEND_ENTRY, op_index, R.leader, R.follower,
                 ie->term, ie->index, 0, ie->command_id);
        }
        // Keep log sorted ascending (incoming entries are ascending and all
        // strictly greater than any survivor, so push_back preserves order;
        // sort defensively).
        std::sort(Fs.log.begin(), Fs.log.end(),
                  [](const RaftLogEntry& a, const RaftLogEntry& b) { return a.index < b.index; });

        // commit_index[F] = min(leader_commit, last_log_index(F))
        uint64_t lliF = last_log_index(F);
        uint64_t newcommit = (R.leader_commit < lliF) ? R.leader_commit : lliF;
        Fs.commit_index = newcommit;

        // match_index[F] = max(match, last index in R, or prev_index if none)
        uint64_t last_in_r = R.prev_index;
        if (!R.entries.empty()) last_in_r = R.entries.back().index;
        uint64_t newmatch = match_index[(size_t)F];
        if (last_in_r > newmatch) newmatch = last_in_r;
        match_index[(size_t)F] = newmatch;
        next_index[(size_t)F] = newmatch + 1;

        counts.append_success += 1;
        emit(RAFT_EV_APPEND_SUCCESS, op_index, R.leader, R.follower,
             R.leader_term, newmatch, (uint64_t)R.entries.size(), newcommit);
    }

    void op_advance_commit(uint32_t op_index) {
        if (!has_leader) {
            counts.invalid_count += 1;
            emit(RAFT_EV_INVALID, op_index, RAFT_U32_MAX, RAFT_U32_MAX, 0,
                 RAFT_U64_MAX, 0, RAFT_OP_ADVANCE_COMMIT);
            return;
        }
        int L = (int)leader_id;
        RaftServer& Ls = servers[(size_t)L];
        const uint64_t lli = last_log_index(L);
        uint64_t bestN = 0;
        bool found = false;
        for (uint64_t N = Ls.commit_index + 1; N <= lli; ++N) {
            // leader entry at N must exist with term == current_term[L]
            const RaftLogEntry* e = log_at(L, N);
            if (e == nullptr || e->term != Ls.current_term) continue;
            int cnt = 0;
            for (int s = 0; s < spec.server_count; ++s) {
                uint64_t mi = (s == L) ? lli : match_index[(size_t)s];
                if (mi >= N) cnt += 1;
            }
            if (cnt >= majority()) { bestN = N; found = true; }
        }
        if (found) {
            Ls.commit_index = bestN;
            counts.commit_advanced += 1;
            emit(RAFT_EV_COMMIT_ADVANCE, op_index, (uint32_t)L, RAFT_U32_MAX,
                 Ls.current_term, bestN, 0, 0);
        } else {
            counts.commit_noop += 1;
            emit(RAFT_EV_COMMIT_NOOP, op_index, (uint32_t)L, RAFT_U32_MAX,
                 Ls.current_term, Ls.commit_index, 0, 0);
        }
    }

    void op_apply(uint32_t op_index, int server, int limit) {
        if (server < 0 || server >= spec.server_count) {
            counts.invalid_count += 1;
            emit(RAFT_EV_INVALID, op_index, RAFT_U32_MAX, RAFT_U32_MAX, 0,
                 RAFT_U64_MAX, 0, RAFT_OP_APPLY);
            return;
        }
        if (limit == 0) {
            // valid no-op (does not emit, does not count invalid)
            return;
        }
        RaftServer& sv = servers[(size_t)server];
        int cap = limit;
        if (cap > spec.max_apply_per_op) cap = spec.max_apply_per_op;

        int applied = 0;
        for (int i = 0; i < cap; ++i) {
            uint64_t want = sv.last_applied + 1;
            if (want > sv.commit_index) break;
            const RaftLogEntry* e = log_at(server, want);
            if (e == nullptr) break;  // not materialized (covered or missing)
            uint64_t h = sv.apply_accumulator;
            raft_fnv_u64(&h, e->index);
            raft_fnv_u64(&h, e->term);
            raft_fnv_u64(&h, e->command_id);
            raft_fnv_i64(&h, e->payload);
            sv.apply_accumulator = h;
            sv.last_applied = e->index;
            applied += 1;
            counts.applied_entries += 1;
            emit(RAFT_EV_APPLY_ENTRY, op_index, (uint32_t)server, RAFT_U32_MAX,
                 e->term, e->index, 0, e->command_id);
        }
        if (applied == 0) {
            counts.apply_empty += 1;
            emit(RAFT_EV_APPLY_EMPTY, op_index, (uint32_t)server, RAFT_U32_MAX,
                 sv.current_term, sv.last_applied, 0, 0);
        }
    }

    void op_take_snapshot(uint32_t op_index, int server) {
        if (server < 0 || server >= spec.server_count ||
            servers[(size_t)server].last_applied <= servers[(size_t)server].snapshot_index) {
            counts.invalid_count += 1;
            emit(RAFT_EV_INVALID, op_index, RAFT_U32_MAX, RAFT_U32_MAX, 0,
                 RAFT_U64_MAX, 0, RAFT_OP_TAKE_SNAPSHOT);
            return;
        }
        RaftServer& sv = servers[(size_t)server];
        uint64_t upto = sv.last_applied;
        const RaftLogEntry* e = log_at(server, upto);
        // upto > snapshot_index and <= last_applied <= committed materialized,
        // so entry must exist.
        uint64_t snap_term = (e != nullptr) ? e->term : sv.snapshot_term;

        sv.snapshot_index = upto;
        sv.snapshot_term = snap_term;
        sv.snapshot_state_hash = sv.apply_accumulator;

        // delete log entries with index <= upto
        uint64_t deleted = 0;
        std::vector<RaftLogEntry> kept;
        for (const RaftLogEntry& le : sv.log) {
            if (le.index <= upto) deleted += 1;
            else kept.push_back(le);
        }
        sv.log.swap(kept);
        if (deleted > 0) {
            counts.snapshot_truncations += 1;
            emit(RAFT_EV_SNAPSHOT_TRUNCATE_LOCAL, op_index, (uint32_t)server, RAFT_U32_MAX,
                 snap_term, upto, deleted, 0);
        }

        counts.snapshots_taken += 1;
        emit(RAFT_EV_SNAPSHOT_TAKE, op_index, (uint32_t)server, RAFT_U32_MAX,
             snap_term, upto, 0, sv.snapshot_state_hash);

        // If this server is the leader, update its own match/next.
        if (has_leader && (int)leader_id == server) {
            uint64_t lli = last_log_index(server);
            match_index[(size_t)server] = lli;
            next_index[(size_t)server] = lli + 1;
        }
    }

    void op_install_snapshot(uint32_t op_index, int follower) {
        if (!has_leader || follower < 0 || follower >= spec.server_count ||
            follower == (int)leader_id) {
            counts.invalid_count += 1;
            emit(RAFT_EV_INVALID, op_index, RAFT_U32_MAX, RAFT_U32_MAX, 0,
                 RAFT_U64_MAX, 0, RAFT_OP_INSTALL_SNAPSHOT);
            return;
        }
        int L = (int)leader_id;
        RaftServer& Ls = servers[(size_t)L];
        RaftServer& Fs = servers[(size_t)follower];

        if (Ls.snapshot_index <= Fs.snapshot_index) {
            counts.snapshot_noop += 1;
            emit(RAFT_EV_SNAPSHOT_INSTALL_NOOP, op_index, (uint32_t)L, (uint32_t)follower,
                 Ls.snapshot_term, Ls.snapshot_index, 0, 0);
            return;
        }

        // Decide retention.
        const RaftLogEntry* be = log_at(follower, Ls.snapshot_index);
        bool retain = (be != nullptr && be->term == Ls.snapshot_term);

        uint64_t deleted = 0;
        if (retain) {
            std::vector<RaftLogEntry> kept;
            for (const RaftLogEntry& e : Fs.log) {
                if (e.index > Ls.snapshot_index) kept.push_back(e);
                else deleted += 1;
            }
            Fs.log.swap(kept);
        } else {
            deleted = Fs.log.size();
            Fs.log.clear();
        }

        Fs.snapshot_index = Ls.snapshot_index;
        Fs.snapshot_term = Ls.snapshot_term;
        Fs.snapshot_state_hash = Ls.snapshot_state_hash;

        if (Fs.commit_index < Fs.snapshot_index) Fs.commit_index = Fs.snapshot_index;
        if (Fs.last_applied < Fs.snapshot_index) Fs.last_applied = Fs.snapshot_index;
        Fs.apply_accumulator = Fs.snapshot_state_hash;

        match_index[(size_t)follower] = Ls.snapshot_index;
        next_index[(size_t)follower] = Ls.snapshot_index + 1;

        counts.snapshots_installed += 1;
        emit(RAFT_EV_SNAPSHOT_INSTALL, op_index, (uint32_t)L, (uint32_t)follower,
             Fs.snapshot_term, Fs.snapshot_index, 0, Fs.snapshot_state_hash);
        if (deleted > 0) {
            counts.snapshot_truncations += 1;
            emit(RAFT_EV_SNAPSHOT_TRUNCATE_REMOTE, op_index, (uint32_t)L, (uint32_t)follower,
                 Fs.snapshot_term, Fs.snapshot_index, deleted, 0);
        }
    }

    void apply_op(uint32_t op_index, const RaftOp& op) {
        switch (op.kind) {
            case RAFT_OP_BECOME_LEADER:
                op_become_leader(op_index, op.i_a, op.u_a);
                break;
            case RAFT_OP_CLIENT_APPEND:
                op_client_append(op_index, op.u_a, op.value);
                break;
            case RAFT_OP_SEND_APPEND:
                op_send_append(op_index, op.i_a, op.i_b);
                break;
            case RAFT_OP_DELIVER_APPEND:
                op_deliver_append(op_index, op.u_a);
                break;
            case RAFT_OP_ADVANCE_COMMIT:
                op_advance_commit(op_index);
                break;
            case RAFT_OP_APPLY:
                op_apply(op_index, op.i_a, op.i_b);
                break;
            case RAFT_OP_TAKE_SNAPSHOT:
                op_take_snapshot(op_index, op.i_a);
                break;
            case RAFT_OP_INSTALL_SNAPSHOT:
                op_install_snapshot(op_index, op.i_a);
                break;
            default:
                counts.invalid_count += 1;
                emit(RAFT_EV_INVALID, op_index, RAFT_U32_MAX, RAFT_U32_MAX, 0,
                     RAFT_U64_MAX, 0, (uint64_t)(uint32_t)op.kind);
                break;
        }
    }

    // -----------------------------------------------------------------------
    // Structural checksums (recomputed from scratch each step).
    // -----------------------------------------------------------------------
    uint64_t compute_log_hash() const {
        uint64_t h = RAFT_FNV_INIT;
        for (int s = 0; s < spec.server_count; ++s) {
            const RaftServer& sv = servers[(size_t)s];
            raft_fnv_u32(&h, (uint32_t)s);
            raft_fnv_u64(&h, sv.snapshot_index);
            raft_fnv_u64(&h, sv.snapshot_term);
            raft_fnv_u64(&h, sv.snapshot_state_hash);
            for (const RaftLogEntry& e : sv.log) {
                raft_fnv_u64(&h, e.index);
                raft_fnv_u64(&h, e.term);
                raft_fnv_u64(&h, e.command_id);
                raft_fnv_i64(&h, e.payload);
            }
        }
        return h;
    }

    uint64_t compute_leader_state_hash() const {
        uint64_t h = RAFT_FNV_INIT;
        raft_fnv_u8(&h, has_leader ? 1 : 0);
        raft_fnv_u32(&h, has_leader ? leader_id : RAFT_U32_MAX);
        for (int s = 0; s < spec.server_count; ++s) {
            raft_fnv_u32(&h, (uint32_t)s);
            raft_fnv_u64(&h, next_index[(size_t)s]);
            raft_fnv_u64(&h, match_index[(size_t)s]);
        }
        return h;
    }

    uint64_t compute_pending_rpc_hash() const {
        uint64_t h = RAFT_FNV_INIT;
        std::vector<int> order(pending.size());
        for (size_t i = 0; i < pending.size(); ++i) order[i] = (int)i;
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return pending[(size_t)a].rpc_id < pending[(size_t)b].rpc_id; });
        for (int oi : order) {
            const RaftPendingRpc& r = pending[(size_t)oi];
            raft_fnv_u64(&h, r.rpc_id);
            raft_fnv_u32(&h, r.leader);
            raft_fnv_u32(&h, r.follower);
            raft_fnv_u64(&h, r.leader_term);
            raft_fnv_u64(&h, r.prev_index);
            raft_fnv_u64(&h, r.prev_term);
            raft_fnv_u64(&h, r.leader_commit);
            raft_fnv_u64(&h, r.send_seq);
            raft_fnv_u64(&h, (uint64_t)r.entries.size());
            for (const RaftLogEntry& e : r.entries) {
                raft_fnv_u64(&h, e.index);
                raft_fnv_u64(&h, e.term);
                raft_fnv_u64(&h, e.command_id);
                raft_fnv_i64(&h, e.payload);
            }
        }
        return h;
    }

    uint64_t compute_apply_hash() const {
        uint64_t h = RAFT_FNV_INIT;
        for (int s = 0; s < spec.server_count; ++s) {
            const RaftServer& sv = servers[(size_t)s];
            raft_fnv_u32(&h, (uint32_t)s);
            raft_fnv_u64(&h, sv.commit_index);
            raft_fnv_u64(&h, sv.last_applied);
            raft_fnv_u64(&h, sv.apply_accumulator);
        }
        return h;
    }

    void step_once(const RaftRunSpec& run, const RaftOp* ops, RaftExpected* expected) {
        for (int i = 0; i < run.num_ops; ++i) {
            apply_op((uint32_t)i, ops[i]);
        }
        expected->counts = counts;
        expected->event_hash = event_hash;
        expected->log_hash = compute_log_hash();
        expected->leader_state_hash = compute_leader_state_hash();
        expected->pending_rpc_hash = compute_pending_rpc_hash();
        expected->apply_hash = compute_apply_hash();
    }
};

struct RaftHostOutputsView {
    const RaftCounts* counts;
    const uint64_t* raft_event_hash;
    const uint64_t* log_hash;
    const uint64_t* leader_state_hash;
    const uint64_t* pending_rpc_hash;
    const uint64_t* apply_hash;
};

static inline bool raft_check_counts(const RaftCounts& e, const RaftCounts& g, std::string* error) {
    const int64_t* pe = reinterpret_cast<const int64_t*>(&e);
    const int64_t* pg = reinterpret_cast<const int64_t*>(&g);
    const int n = (int)(sizeof(RaftCounts) / sizeof(int64_t));
    static const char* names[] = {
        "leaders_elected", "client_appended", "client_rejected", "append_sent",
        "append_send_rejected", "append_needs_snapshot", "append_success",
        "append_stale", "append_term_reject", "append_conflict",
        "follower_appended_entries", "follower_deleted_suffixes",
        "append_follower_oom", "commit_advanced", "commit_noop",
        "applied_entries", "apply_empty", "snapshots_taken",
        "snapshots_installed", "snapshot_noop", "snapshot_truncations",
        "invalid_count", "reserved0", "reserved1"};
    for (int i = 0; i < n; ++i) {
        if (pe[i] != pg[i]) {
            if (error) {
                std::ostringstream oss;
                oss << "count mismatch " << names[i] << ": got " << pg[i]
                    << ", expected " << pe[i];
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

static inline bool raft_check_all_outputs(
    const RaftExpected& expected,
    const RaftHostOutputsView& got,
    std::string* error) {
    if (!raft_check_counts(expected.counts, *got.counts, error)) return false;

    struct HC { const char* name; uint64_t e; uint64_t g; };
    HC hashes[] = {
        {"raft_event_hash", expected.event_hash, got.raft_event_hash[0]},
        {"log_hash", expected.log_hash, got.log_hash[0]},
        {"leader_state_hash", expected.leader_state_hash, got.leader_state_hash[0]},
        {"pending_rpc_hash", expected.pending_rpc_hash, got.pending_rpc_hash[0]},
        {"apply_hash", expected.apply_hash, got.apply_hash[0]},
    };
    for (const HC& hc : hashes) {
        if (hc.e != hc.g) {
            if (error) {
                std::ostringstream oss;
                oss << hc.name << " mismatch: got 0x" << std::hex << hc.g
                    << ", expected 0x" << hc.e;
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

/*
GRADER MODEL
  oracle.init(problem_spec)
  solution_reset + oracle.reset()
  for each step:
    solution_run(...)
    oracle.step_once(...)
    raft_check_all_outputs(...)

Required harness coverage:
  - leader changes with stale RPC delivery
  - conflict backtracking (conflict_term path and conflict_index path)
  - commit advancement gated on current-term entries
  - local TAKE_SNAPSHOT + truncation
  - INSTALL_SNAPSHOT with retained suffix and with full wipe
  - APPEND_NEEDS_SNAPSHOT
  - follower OOM rejection
  - empty-batch / no-op steps
  - reset and exact replay
*/

#endif  // RAFT_LOG_SNAPSHOT_ORACLE_HPP_
