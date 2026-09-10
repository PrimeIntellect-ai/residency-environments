// PMPP_CANARY_48_a8b4b589e3 -- held-out canary; MUST NOT appear in any submission
// file: raft_log_snapshot_reference.cu
//
// Reference implementation. Maintains persistent device state in flat arrays
// and processes each step's op batch in one single-thread kernel. Recomputes
// all structural hashes from scratch each step. Independent of both the host
// oracle and the naive replay solution.

#include "raft_log_snapshot_common.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RAFT_FNV_INIT 1469598103934665603ULL
#define RAFT_U32_MAX 0xFFFFFFFFu
#define RAFT_U64_MAX 0xFFFFFFFFFFFFFFFFull

// ---------------------------------------------------------------------------
// Device FNV-1a-64 (little-endian per declared width).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t rf_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= 1099511628211ULL;
    return h;
}
__device__ __forceinline__ void rf_u8(uint64_t* h, uint8_t v) { *h = rf_byte(*h, v); }
__device__ __forceinline__ void rf_u32(uint64_t* h, uint32_t v) {
    uint64_t hh = *h;
    for (int i = 0; i < 4; ++i) hh = rf_byte(hh, (uint8_t)((v >> (8 * i)) & 0xFF));
    *h = hh;
}
__device__ __forceinline__ void rf_u64(uint64_t* h, uint64_t v) {
    uint64_t hh = *h;
    for (int i = 0; i < 8; ++i) hh = rf_byte(hh, (uint8_t)((v >> (8 * i)) & 0xFF));
    *h = hh;
}
__device__ __forceinline__ void rf_i64(uint64_t* h, int64_t v) { rf_u64(h, (uint64_t)v); }

// ---------------------------------------------------------------------------
// Persistent device state (flat arrays). One "engine" instance.
// Layout sized by spec maxima.
// ---------------------------------------------------------------------------
struct RaftDevState {
    // scalars (single-element device arrays)
    uint64_t* event_seq;       // [1]
    uint64_t* rpc_seq_next;    // [1]
    int32_t* has_leader;       // [1]
    uint32_t* leader_id;       // [1]
    uint64_t* event_hash;      // [1]

    // per server
    uint64_t* current_term;    // [S]
    int32_t* role;             // [S]
    uint64_t* snap_index;      // [S]
    uint64_t* snap_term;       // [S]
    uint64_t* snap_hash;       // [S]
    uint64_t* commit_index;    // [S]
    uint64_t* last_applied;    // [S]
    uint64_t* apply_acc;       // [S]

    // log per server: count + entries [S * Lcap]
    int32_t* log_count;        // [S]
    uint64_t* log_index;       // [S*Lcap]
    uint64_t* log_term;        // [S*Lcap]
    uint64_t* log_cmd;         // [S*Lcap]
    int64_t* log_payload;      // [S*Lcap]

    // leader volatile per server (for current leader)
    uint64_t* next_index;      // [S]
    uint64_t* match_index;     // [S]

    // pending rpc table
    int32_t* pending_count;    // [1]
    uint64_t* p_rpc_id;        // [P]
    uint32_t* p_leader;        // [P]
    uint32_t* p_follower;      // [P]
    uint64_t* p_leader_term;   // [P]
    uint64_t* p_prev_index;    // [P]
    uint64_t* p_prev_term;     // [P]
    uint64_t* p_leader_commit; // [P]
    uint64_t* p_send_seq;      // [P]
    int32_t* p_ecount;         // [P]
    uint64_t* p_eindex;        // [P*Ecap]
    uint64_t* p_eterm;         // [P*Ecap]
    uint64_t* p_ecmd;          // [P*Ecap]
    int64_t* p_epayload;       // [P*Ecap]

    // cumulative counts
    int64_t* counts;           // [24]
};

struct RaftReferenceState {
    RaftProblemSpec spec;
    RaftDevState dev;
    RaftOp* d_ops;             // [max_ops]
};

// ---------------------------------------------------------------------------
// Device helpers operating on the flat state. All single-thread.
// ---------------------------------------------------------------------------
struct RaftCtx {
    int S, Lcap, P, Ecap, maxlog, max_epa, max_apply, majority;
    RaftDevState d;
};

__device__ uint64_t r_last_log_index(const RaftCtx& c, int s) {
    int n = c.d.log_count[s];
    if (n == 0) return c.d.snap_index[s];
    return c.d.log_index[s * c.Lcap + (n - 1)];
}

// returns slot in [0,n) or -1
__device__ int r_log_slot(const RaftCtx& c, int s, uint64_t index) {
    int n = c.d.log_count[s];
    int base = s * c.Lcap;
    for (int i = 0; i < n; ++i) {
        if (c.d.log_index[base + i] == index) return i;
    }
    return -1;
}

__device__ void r_emit(const RaftCtx& c, uint8_t kind, uint32_t op_index,
                       uint32_t server, uint32_t peer, uint64_t term,
                       uint64_t index_or_max, uint64_t count_or_zero, uint64_t aux) {
    uint64_t es = c.d.event_seq[0];
    uint64_t h = c.d.event_hash[0];
    rf_u8(&h, kind);
    rf_u64(&h, es);
    rf_u32(&h, op_index);
    rf_u32(&h, server);
    rf_u32(&h, peer);
    rf_u64(&h, term);
    rf_u64(&h, index_or_max);
    rf_u64(&h, count_or_zero);
    rf_u64(&h, aux);
    c.d.event_hash[0] = h;
    c.d.event_seq[0] = es + 1;
}

__device__ __forceinline__ void r_cnt(const RaftCtx& c, int idx) { c.d.counts[idx] += 1; }

// count index helpers mirror RaftCounts field order
enum {
    CN_leaders_elected = 0, CN_client_appended, CN_client_rejected, CN_append_sent,
    CN_append_send_rejected, CN_append_needs_snapshot, CN_append_success,
    CN_append_stale, CN_append_term_reject, CN_append_conflict,
    CN_follower_appended_entries, CN_follower_deleted_suffixes,
    CN_append_follower_oom, CN_commit_advanced, CN_commit_noop,
    CN_applied_entries, CN_apply_empty, CN_snapshots_taken,
    CN_snapshots_installed, CN_snapshot_noop, CN_snapshot_truncations,
    CN_invalid_count
};

__device__ void r_invalid(const RaftCtx& c, uint32_t op_index, uint64_t aux) {
    r_cnt(c, CN_invalid_count);
    r_emit(c, RAFT_EV_INVALID, op_index, RAFT_U32_MAX, RAFT_U32_MAX, 0, RAFT_U64_MAX, 0, aux);
}

// ---------------- ops ----------------
__device__ void r_become_leader(const RaftCtx& c, uint32_t op_index, int server, uint64_t term) {
    if (server < 0 || server >= c.S || term < c.d.current_term[server]) {
        r_invalid(c, op_index, RAFT_OP_BECOME_LEADER);
        return;
    }
    for (int s = 0; s < c.S; ++s) {
        if (s != server && c.d.role[s] == RAFT_ROLE_LEADER) c.d.role[s] = RAFT_ROLE_FOLLOWER;
    }
    c.d.current_term[server] = term;
    c.d.role[server] = RAFT_ROLE_LEADER;
    c.d.has_leader[0] = 1;
    c.d.leader_id[0] = (uint32_t)server;

    uint64_t lli = r_last_log_index(c, server);
    for (int f = 0; f < c.S; ++f) {
        if (f == server) continue;
        c.d.next_index[f] = lli + 1;
        c.d.match_index[f] = 0;
    }
    c.d.match_index[server] = lli;
    c.d.next_index[server] = lli + 1;

    r_cnt(c, CN_leaders_elected);
    r_emit(c, RAFT_EV_BECOME_LEADER_OK, op_index, (uint32_t)server, RAFT_U32_MAX, term, lli, 0, 0);
}

__device__ void r_client_append(const RaftCtx& c, uint32_t op_index, uint64_t cmd, int64_t payload) {
    if (!c.d.has_leader[0]) { r_invalid(c, op_index, RAFT_OP_CLIENT_APPEND); return; }
    int L = (int)c.d.leader_id[0];
    int n = c.d.log_count[L];
    if (n >= c.maxlog) {
        r_cnt(c, CN_client_rejected);
        r_emit(c, RAFT_EV_CLIENT_REJECT_LOG_FULL, op_index, (uint32_t)L, RAFT_U32_MAX,
               c.d.current_term[L], RAFT_U64_MAX, 0, 0);
        return;
    }
    uint64_t idx = r_last_log_index(c, L) + 1;
    int base = L * c.Lcap;
    c.d.log_index[base + n] = idx;
    c.d.log_term[base + n] = c.d.current_term[L];
    c.d.log_cmd[base + n] = cmd;
    c.d.log_payload[base + n] = payload;
    c.d.log_count[L] = n + 1;

    uint64_t lli = r_last_log_index(c, L);
    c.d.match_index[L] = lli;
    c.d.next_index[L] = lli + 1;

    r_cnt(c, CN_client_appended);
    r_emit(c, RAFT_EV_CLIENT_APPEND_OK, op_index, (uint32_t)L, RAFT_U32_MAX,
           c.d.current_term[L], idx, 0, cmd);
}

__device__ void r_send_append(const RaftCtx& c, uint32_t op_index, int follower, int max_entries) {
    if (!c.d.has_leader[0] || follower < 0 || follower >= c.S ||
        follower == (int)c.d.leader_id[0] || max_entries == 0) {
        r_invalid(c, op_index, RAFT_OP_SEND_APPEND);
        return;
    }
    int L = (int)c.d.leader_id[0];
    if (c.d.pending_count[0] >= c.P) {
        r_cnt(c, CN_append_send_rejected);
        r_emit(c, RAFT_EV_APPEND_SEND_REJECT, op_index, (uint32_t)L, (uint32_t)follower,
               c.d.current_term[L], RAFT_U64_MAX, 0, 0);
        return;
    }
    uint64_t ni = c.d.next_index[follower];
    if (ni <= c.d.snap_index[L]) {
        r_cnt(c, CN_append_needs_snapshot);
        r_emit(c, RAFT_EV_APPEND_NEEDS_SNAPSHOT, op_index, (uint32_t)L, (uint32_t)follower,
               c.d.current_term[L], c.d.snap_index[L], 0, 0);
        return;
    }
    uint64_t prev_index = ni - 1;
    uint64_t prev_term;
    if (prev_index == c.d.snap_index[L]) {
        prev_term = c.d.snap_term[L];
    } else {
        int slot = r_log_slot(c, L, prev_index);
        if (slot < 0) {
            r_cnt(c, CN_append_send_rejected);
            r_emit(c, RAFT_EV_APPEND_SEND_REJECT, op_index, (uint32_t)L, (uint32_t)follower,
                   c.d.current_term[L], RAFT_U64_MAX, 0, 0);
            return;
        }
        prev_term = c.d.log_term[L * c.Lcap + slot];
    }

    int cap = max_entries;
    if (cap > c.max_epa) cap = c.max_epa;

    int slot = c.d.pending_count[0];
    uint64_t rid = c.d.rpc_seq_next[0];
    c.d.rpc_seq_next[0] = rid + 1;

    int ebase = slot * c.Ecap;
    int copied = 0;
    int n = c.d.log_count[L];
    int lbase = L * c.Lcap;
    for (int i = 0; i < n; ++i) {
        uint64_t li = c.d.log_index[lbase + i];
        if (li >= ni) {
            c.d.p_eindex[ebase + copied] = li;
            c.d.p_eterm[ebase + copied] = c.d.log_term[lbase + i];
            c.d.p_ecmd[ebase + copied] = c.d.log_cmd[lbase + i];
            c.d.p_epayload[ebase + copied] = c.d.log_payload[lbase + i];
            copied += 1;
            if (copied >= cap) break;
        }
    }

    c.d.p_rpc_id[slot] = rid;
    c.d.p_leader[slot] = (uint32_t)L;
    c.d.p_follower[slot] = (uint32_t)follower;
    c.d.p_leader_term[slot] = c.d.current_term[L];
    c.d.p_prev_index[slot] = prev_index;
    c.d.p_prev_term[slot] = prev_term;
    c.d.p_leader_commit[slot] = c.d.commit_index[L];
    c.d.p_send_seq[slot] = c.d.event_seq[0];
    c.d.p_ecount[slot] = copied;
    c.d.pending_count[0] = slot + 1;

    r_cnt(c, CN_append_sent);
    r_emit(c, RAFT_EV_APPEND_SEND, op_index, (uint32_t)L, (uint32_t)follower,
           c.d.current_term[L], prev_index, (uint64_t)copied, rid);
}

__device__ uint64_t r_first_index_with_term(const RaftCtx& c, int f, uint64_t term) {
    int n = c.d.log_count[f];
    int base = f * c.Lcap;
    for (int i = 0; i < n; ++i) {
        if (c.d.log_term[base + i] == term) return c.d.log_index[base + i];
    }
    return RAFT_U64_MAX;
}
__device__ uint64_t r_leader_lastp1_with_term(const RaftCtx& c, int L, uint64_t term) {
    int n = c.d.log_count[L];
    int base = L * c.Lcap;
    uint64_t best = 0; bool found = false;
    for (int i = 0; i < n; ++i) {
        if (c.d.log_term[base + i] == term) { best = c.d.log_index[base + i]; found = true; }
    }
    if (!found) return 0;
    return best + 1;
}

__device__ void r_deliver_append(const RaftCtx& c, uint32_t op_index, uint64_t rpc_id) {
    int pc = c.d.pending_count[0];
    int pos = -1;
    for (int i = 0; i < pc; ++i) {
        if (c.d.p_rpc_id[i] == rpc_id) { pos = i; break; }
    }
    if (pos < 0) { r_invalid(c, op_index, RAFT_OP_DELIVER_APPEND); return; }

    // snapshot the RPC fields, then remove it (compact array)
    uint64_t R_rpc = c.d.p_rpc_id[pos];
    uint32_t R_leader = c.d.p_leader[pos];
    uint32_t R_follower = c.d.p_follower[pos];
    uint64_t R_leader_term = c.d.p_leader_term[pos];
    uint64_t R_prev_index = c.d.p_prev_index[pos];
    uint64_t R_prev_term = c.d.p_prev_term[pos];
    uint64_t R_leader_commit = c.d.p_leader_commit[pos];
    int R_ec = c.d.p_ecount[pos];
    // copy entries to a local stack buffer
    uint64_t e_idx[RAFT_MAX_ENTRIES_PER_APPEND];
    uint64_t e_trm[RAFT_MAX_ENTRIES_PER_APPEND];
    uint64_t e_cmd[RAFT_MAX_ENTRIES_PER_APPEND];
    int64_t e_pay[RAFT_MAX_ENTRIES_PER_APPEND];
    int ebase = pos * c.Ecap;
    for (int i = 0; i < R_ec; ++i) {
        e_idx[i] = c.d.p_eindex[ebase + i];
        e_trm[i] = c.d.p_eterm[ebase + i];
        e_cmd[i] = c.d.p_ecmd[ebase + i];
        e_pay[i] = c.d.p_epayload[ebase + i];
    }
    // remove by shifting tail down
    for (int i = pos; i < pc - 1; ++i) {
        c.d.p_rpc_id[i] = c.d.p_rpc_id[i + 1];
        c.d.p_leader[i] = c.d.p_leader[i + 1];
        c.d.p_follower[i] = c.d.p_follower[i + 1];
        c.d.p_leader_term[i] = c.d.p_leader_term[i + 1];
        c.d.p_prev_index[i] = c.d.p_prev_index[i + 1];
        c.d.p_prev_term[i] = c.d.p_prev_term[i + 1];
        c.d.p_leader_commit[i] = c.d.p_leader_commit[i + 1];
        c.d.p_send_seq[i] = c.d.p_send_seq[i + 1];
        int ec2 = c.d.p_ecount[i + 1];
        c.d.p_ecount[i] = ec2;
        int eb_dst = i * c.Ecap;
        int eb_src = (i + 1) * c.Ecap;
        for (int k = 0; k < ec2; ++k) {
            c.d.p_eindex[eb_dst + k] = c.d.p_eindex[eb_src + k];
            c.d.p_eterm[eb_dst + k] = c.d.p_eterm[eb_src + k];
            c.d.p_ecmd[eb_dst + k] = c.d.p_ecmd[eb_src + k];
            c.d.p_epayload[eb_dst + k] = c.d.p_epayload[eb_src + k];
        }
    }
    c.d.pending_count[0] = pc - 1;

    // stale leader
    if (!(c.d.has_leader[0] && c.d.leader_id[0] == R_leader) ||
        c.d.current_term[R_leader] != R_leader_term) {
        r_cnt(c, CN_append_stale);
        r_emit(c, RAFT_EV_APPEND_STALE, op_index, R_leader, R_follower, R_leader_term, R_rpc, 0, 0);
        return;
    }

    int F = (int)R_follower;
    if (c.d.current_term[F] > R_leader_term) {
        r_cnt(c, CN_append_term_reject);
        uint64_t ni = c.d.next_index[F];
        uint64_t newni = (ni > 1) ? (ni - 1) : 1;
        c.d.next_index[F] = newni;
        r_emit(c, RAFT_EV_APPEND_TERM_REJECT, op_index, R_leader, R_follower,
               R_leader_term, R_prev_index, 0, newni);
        return;
    }
    c.d.current_term[F] = R_leader_term;

    // consistency
    bool consistent = false, fail = false, has_ct = false;
    uint64_t conflict_index = 0, conflict_term = RAFT_U64_MAX;
    if (R_prev_index < c.d.snap_index[F]) {
        fail = true; conflict_index = c.d.snap_index[F] + 1;
    } else if (R_prev_index == c.d.snap_index[F]) {
        if (R_prev_term == c.d.snap_term[F]) consistent = true;
        else { fail = true; conflict_index = r_last_log_index(c, F) + 1; }
    } else {
        int slot = r_log_slot(c, F, R_prev_index);
        if (slot < 0) { fail = true; conflict_index = r_last_log_index(c, F) + 1; }
        else {
            uint64_t pt = c.d.log_term[F * c.Lcap + slot];
            if (pt == R_prev_term) consistent = true;
            else { fail = true; has_ct = true; conflict_term = pt;
                   conflict_index = r_first_index_with_term(c, F, pt); }
        }
    }

    if (fail) {
        uint64_t newni;
        if (has_ct) {
            uint64_t lt = r_leader_lastp1_with_term(c, (int)R_leader, conflict_term);
            newni = (lt != 0) ? lt : conflict_index;
        } else newni = conflict_index;
        c.d.next_index[F] = newni;
        r_cnt(c, CN_append_conflict);
        r_emit(c, RAFT_EV_APPEND_CONFLICT, op_index, R_leader, R_follower,
               R_leader_term, conflict_index, has_ct ? conflict_term : RAFT_U64_MAX, newni);
        return;
    }
    (void)consistent;

    // success path: compute del_from (first incoming index present with diff term)
    uint64_t del_from = RAFT_U64_MAX;
    for (int i = 0; i < R_ec; ++i) {
        int fslot = r_log_slot(c, F, e_idx[i]);
        if (fslot >= 0 && c.d.log_term[F * c.Lcap + fslot] != e_trm[i]) { del_from = e_idx[i]; break; }
    }

    int fn = c.d.log_count[F];
    int fbase = F * c.Lcap;
    int survivors = 0;
    uint64_t deleted_count = 0, first_deleted_index = 0;
    if (del_from != RAFT_U64_MAX) {
        for (int i = 0; i < fn; ++i) {
            uint64_t li = c.d.log_index[fbase + i];
            if (li < del_from) survivors += 1;
            else { if (deleted_count == 0) first_deleted_index = li; deleted_count += 1; }
        }
    } else survivors = fn;

    // which incoming to append
    bool app_flag[RAFT_MAX_ENTRIES_PER_APPEND];
    int to_append = 0;
    for (int i = 0; i < R_ec; ++i) {
        app_flag[i] = false;
        if (e_idx[i] <= c.d.snap_index[F]) continue;
        bool present;
        if (del_from != RAFT_U64_MAX) present = (e_idx[i] < del_from);
        else present = (r_log_slot(c, F, e_idx[i]) >= 0);
        if (!present) { app_flag[i] = true; to_append += 1; }
    }

    int final_size = survivors + to_append;
    if (final_size > c.maxlog) {
        r_cnt(c, CN_append_follower_oom);
        r_emit(c, RAFT_EV_APPEND_FOLLOWER_OOM, op_index, R_leader, R_follower,
               R_leader_term, R_prev_index, (uint64_t)final_size, 0);
        return;
    }

    // apply deletion: keep entries with index < del_from
    if (del_from != RAFT_U64_MAX && deleted_count > 0) {
        int w = 0;
        for (int i = 0; i < fn; ++i) {
            uint64_t li = c.d.log_index[fbase + i];
            if (li < del_from) {
                if (w != i) {
                    c.d.log_index[fbase + w] = c.d.log_index[fbase + i];
                    c.d.log_term[fbase + w] = c.d.log_term[fbase + i];
                    c.d.log_cmd[fbase + w] = c.d.log_cmd[fbase + i];
                    c.d.log_payload[fbase + w] = c.d.log_payload[fbase + i];
                }
                w += 1;
            }
        }
        c.d.log_count[F] = w;
        fn = w;
        r_cnt(c, CN_follower_deleted_suffixes);
        r_emit(c, RAFT_EV_FOLLOWER_DELETE_SUFFIX, op_index, R_leader, R_follower,
               R_leader_term, first_deleted_index, deleted_count, 0);
    }

    // append (incoming ascending; all > any survivor so order preserved)
    for (int i = 0; i < R_ec; ++i) {
        if (!app_flag[i]) continue;
        int n2 = c.d.log_count[F];
        c.d.log_index[fbase + n2] = e_idx[i];
        c.d.log_term[fbase + n2] = e_trm[i];
        c.d.log_cmd[fbase + n2] = e_cmd[i];
        c.d.log_payload[fbase + n2] = e_pay[i];
        c.d.log_count[F] = n2 + 1;
        r_cnt(c, CN_follower_appended_entries);
        r_emit(c, RAFT_EV_FOLLOWER_APPEND_ENTRY, op_index, R_leader, R_follower,
               e_trm[i], e_idx[i], 0, e_cmd[i]);
    }

    uint64_t lliF = r_last_log_index(c, F);
    uint64_t newcommit = (R_leader_commit < lliF) ? R_leader_commit : lliF;
    c.d.commit_index[F] = newcommit;

    uint64_t last_in_r = R_prev_index;
    if (R_ec > 0) last_in_r = e_idx[R_ec - 1];
    uint64_t newmatch = c.d.match_index[F];
    if (last_in_r > newmatch) newmatch = last_in_r;
    c.d.match_index[F] = newmatch;
    c.d.next_index[F] = newmatch + 1;

    r_cnt(c, CN_append_success);
    r_emit(c, RAFT_EV_APPEND_SUCCESS, op_index, R_leader, R_follower,
           R_leader_term, newmatch, (uint64_t)R_ec, newcommit);
}

__device__ void r_advance_commit(const RaftCtx& c, uint32_t op_index) {
    if (!c.d.has_leader[0]) { r_invalid(c, op_index, RAFT_OP_ADVANCE_COMMIT); return; }
    int L = (int)c.d.leader_id[0];
    uint64_t lli = r_last_log_index(c, L);
    uint64_t bestN = 0; bool found = false;
    for (uint64_t N = c.d.commit_index[L] + 1; N <= lli; ++N) {
        int slot = r_log_slot(c, L, N);
        if (slot < 0) continue;
        if (c.d.log_term[L * c.Lcap + slot] != c.d.current_term[L]) continue;
        int cnt = 0;
        for (int s = 0; s < c.S; ++s) {
            uint64_t mi = (s == L) ? lli : c.d.match_index[s];
            if (mi >= N) cnt += 1;
        }
        if (cnt >= c.majority) { bestN = N; found = true; }
    }
    if (found) {
        c.d.commit_index[L] = bestN;
        r_cnt(c, CN_commit_advanced);
        r_emit(c, RAFT_EV_COMMIT_ADVANCE, op_index, (uint32_t)L, RAFT_U32_MAX,
               c.d.current_term[L], bestN, 0, 0);
    } else {
        r_cnt(c, CN_commit_noop);
        r_emit(c, RAFT_EV_COMMIT_NOOP, op_index, (uint32_t)L, RAFT_U32_MAX,
               c.d.current_term[L], c.d.commit_index[L], 0, 0);
    }
}

__device__ void r_apply(const RaftCtx& c, uint32_t op_index, int server, int limit) {
    if (server < 0 || server >= c.S) { r_invalid(c, op_index, RAFT_OP_APPLY); return; }
    if (limit == 0) return;
    int cap = limit;
    if (cap > c.max_apply) cap = c.max_apply;
    int applied = 0;
    for (int i = 0; i < cap; ++i) {
        uint64_t want = c.d.last_applied[server] + 1;
        if (want > c.d.commit_index[server]) break;
        int slot = r_log_slot(c, server, want);
        if (slot < 0) break;
        int b = server * c.Lcap + slot;
        uint64_t h = c.d.apply_acc[server];
        rf_u64(&h, c.d.log_index[b]);
        rf_u64(&h, c.d.log_term[b]);
        rf_u64(&h, c.d.log_cmd[b]);
        rf_i64(&h, c.d.log_payload[b]);
        c.d.apply_acc[server] = h;
        c.d.last_applied[server] = c.d.log_index[b];
        applied += 1;
        r_cnt(c, CN_applied_entries);
        r_emit(c, RAFT_EV_APPLY_ENTRY, op_index, (uint32_t)server, RAFT_U32_MAX,
               c.d.log_term[b], c.d.log_index[b], 0, c.d.log_cmd[b]);
    }
    if (applied == 0) {
        r_cnt(c, CN_apply_empty);
        r_emit(c, RAFT_EV_APPLY_EMPTY, op_index, (uint32_t)server, RAFT_U32_MAX,
               c.d.current_term[server], c.d.last_applied[server], 0, 0);
    }
}

__device__ void r_take_snapshot(const RaftCtx& c, uint32_t op_index, int server) {
    if (server < 0 || server >= c.S || c.d.last_applied[server] <= c.d.snap_index[server]) {
        r_invalid(c, op_index, RAFT_OP_TAKE_SNAPSHOT);
        return;
    }
    uint64_t upto = c.d.last_applied[server];
    int slot = r_log_slot(c, server, upto);
    uint64_t snap_term = (slot >= 0) ? c.d.log_term[server * c.Lcap + slot] : c.d.snap_term[server];

    c.d.snap_index[server] = upto;
    c.d.snap_term[server] = snap_term;
    c.d.snap_hash[server] = c.d.apply_acc[server];

    int fn = c.d.log_count[server];
    int fbase = server * c.Lcap;
    uint64_t deleted = 0;
    int w = 0;
    for (int i = 0; i < fn; ++i) {
        uint64_t li = c.d.log_index[fbase + i];
        if (li <= upto) deleted += 1;
        else {
            if (w != i) {
                c.d.log_index[fbase + w] = c.d.log_index[fbase + i];
                c.d.log_term[fbase + w] = c.d.log_term[fbase + i];
                c.d.log_cmd[fbase + w] = c.d.log_cmd[fbase + i];
                c.d.log_payload[fbase + w] = c.d.log_payload[fbase + i];
            }
            w += 1;
        }
    }
    c.d.log_count[server] = w;
    if (deleted > 0) {
        r_cnt(c, CN_snapshot_truncations);
        r_emit(c, RAFT_EV_SNAPSHOT_TRUNCATE_LOCAL, op_index, (uint32_t)server, RAFT_U32_MAX,
               snap_term, upto, deleted, 0);
    }
    r_cnt(c, CN_snapshots_taken);
    r_emit(c, RAFT_EV_SNAPSHOT_TAKE, op_index, (uint32_t)server, RAFT_U32_MAX,
           snap_term, upto, 0, c.d.snap_hash[server]);

    if (c.d.has_leader[0] && (int)c.d.leader_id[0] == server) {
        uint64_t lli = r_last_log_index(c, server);
        c.d.match_index[server] = lli;
        c.d.next_index[server] = lli + 1;
    }
}

__device__ void r_install_snapshot(const RaftCtx& c, uint32_t op_index, int follower) {
    if (!c.d.has_leader[0] || follower < 0 || follower >= c.S ||
        follower == (int)c.d.leader_id[0]) {
        r_invalid(c, op_index, RAFT_OP_INSTALL_SNAPSHOT);
        return;
    }
    int L = (int)c.d.leader_id[0];
    if (c.d.snap_index[L] <= c.d.snap_index[follower]) {
        r_cnt(c, CN_snapshot_noop);
        r_emit(c, RAFT_EV_SNAPSHOT_INSTALL_NOOP, op_index, (uint32_t)L, (uint32_t)follower,
               c.d.snap_term[L], c.d.snap_index[L], 0, 0);
        return;
    }
    int bslot = r_log_slot(c, follower, c.d.snap_index[L]);
    bool retain = (bslot >= 0 && c.d.log_term[follower * c.Lcap + bslot] == c.d.snap_term[L]);

    int fn = c.d.log_count[follower];
    int fbase = follower * c.Lcap;
    uint64_t deleted = 0;
    if (retain) {
        int w = 0;
        for (int i = 0; i < fn; ++i) {
            uint64_t li = c.d.log_index[fbase + i];
            if (li > c.d.snap_index[L]) {
                if (w != i) {
                    c.d.log_index[fbase + w] = c.d.log_index[fbase + i];
                    c.d.log_term[fbase + w] = c.d.log_term[fbase + i];
                    c.d.log_cmd[fbase + w] = c.d.log_cmd[fbase + i];
                    c.d.log_payload[fbase + w] = c.d.log_payload[fbase + i];
                }
                w += 1;
            } else deleted += 1;
        }
        c.d.log_count[follower] = w;
    } else {
        deleted = (uint64_t)fn;
        c.d.log_count[follower] = 0;
    }

    c.d.snap_index[follower] = c.d.snap_index[L];
    c.d.snap_term[follower] = c.d.snap_term[L];
    c.d.snap_hash[follower] = c.d.snap_hash[L];

    if (c.d.commit_index[follower] < c.d.snap_index[follower])
        c.d.commit_index[follower] = c.d.snap_index[follower];
    if (c.d.last_applied[follower] < c.d.snap_index[follower])
        c.d.last_applied[follower] = c.d.snap_index[follower];
    c.d.apply_acc[follower] = c.d.snap_hash[follower];

    c.d.match_index[follower] = c.d.snap_index[L];
    c.d.next_index[follower] = c.d.snap_index[L] + 1;

    r_cnt(c, CN_snapshots_installed);
    r_emit(c, RAFT_EV_SNAPSHOT_INSTALL, op_index, (uint32_t)L, (uint32_t)follower,
           c.d.snap_term[follower], c.d.snap_index[follower], 0, c.d.snap_hash[follower]);
    if (deleted > 0) {
        r_cnt(c, CN_snapshot_truncations);
        r_emit(c, RAFT_EV_SNAPSHOT_TRUNCATE_REMOTE, op_index, (uint32_t)L, (uint32_t)follower,
               c.d.snap_term[follower], c.d.snap_index[follower], deleted, 0);
    }
}

// ---------------- hashes ----------------
__device__ void r_write_hashes(const RaftCtx& c,
                               uint64_t* out_log, uint64_t* out_leader,
                               uint64_t* out_pending, uint64_t* out_apply) {
    // log_hash
    uint64_t h = RAFT_FNV_INIT;
    for (int s = 0; s < c.S; ++s) {
        rf_u32(&h, (uint32_t)s);
        rf_u64(&h, c.d.snap_index[s]);
        rf_u64(&h, c.d.snap_term[s]);
        rf_u64(&h, c.d.snap_hash[s]);
        int n = c.d.log_count[s];
        int base = s * c.Lcap;
        for (int i = 0; i < n; ++i) {
            rf_u64(&h, c.d.log_index[base + i]);
            rf_u64(&h, c.d.log_term[base + i]);
            rf_u64(&h, c.d.log_cmd[base + i]);
            rf_i64(&h, c.d.log_payload[base + i]);
        }
    }
    *out_log = h;

    // leader_state_hash
    uint64_t hl = RAFT_FNV_INIT;
    rf_u8(&hl, c.d.has_leader[0] ? 1 : 0);
    rf_u32(&hl, c.d.has_leader[0] ? c.d.leader_id[0] : RAFT_U32_MAX);
    for (int s = 0; s < c.S; ++s) {
        rf_u32(&hl, (uint32_t)s);
        rf_u64(&hl, c.d.next_index[s]);
        rf_u64(&hl, c.d.match_index[s]);
    }
    *out_leader = hl;

    // pending_rpc_hash (sorted by rpc_id ascending). Selection sort over slots
    // (pending count <= P which is small).
    uint64_t hp = RAFT_FNV_INIT;
    int pc = c.d.pending_count[0];
    int idxbuf[RAFT_MAX_PENDING_APPEND_RPCS];
    for (int i = 0; i < pc; ++i) idxbuf[i] = i;
    for (int i = 0; i < pc; ++i) {
        int m = i;
        for (int j = i + 1; j < pc; ++j) {
            if (c.d.p_rpc_id[idxbuf[j]] < c.d.p_rpc_id[idxbuf[m]]) m = j;
        }
        int t = idxbuf[i]; idxbuf[i] = idxbuf[m]; idxbuf[m] = t;
    }
    for (int r = 0; r < pc; ++r) {
        int s = idxbuf[r];
        rf_u64(&hp, c.d.p_rpc_id[s]);
        rf_u32(&hp, c.d.p_leader[s]);
        rf_u32(&hp, c.d.p_follower[s]);
        rf_u64(&hp, c.d.p_leader_term[s]);
        rf_u64(&hp, c.d.p_prev_index[s]);
        rf_u64(&hp, c.d.p_prev_term[s]);
        rf_u64(&hp, c.d.p_leader_commit[s]);
        rf_u64(&hp, c.d.p_send_seq[s]);
        int ec = c.d.p_ecount[s];
        rf_u64(&hp, (uint64_t)ec);
        int eb = s * c.Ecap;
        for (int k = 0; k < ec; ++k) {
            rf_u64(&hp, c.d.p_eindex[eb + k]);
            rf_u64(&hp, c.d.p_eterm[eb + k]);
            rf_u64(&hp, c.d.p_ecmd[eb + k]);
            rf_i64(&hp, c.d.p_epayload[eb + k]);
        }
    }
    *out_pending = hp;

    // apply_hash
    uint64_t ha = RAFT_FNV_INIT;
    for (int s = 0; s < c.S; ++s) {
        rf_u32(&ha, (uint32_t)s);
        rf_u64(&ha, c.d.commit_index[s]);
        rf_u64(&ha, c.d.last_applied[s]);
        rf_u64(&ha, c.d.apply_acc[s]);
    }
    *out_apply = ha;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
__global__ void raft_ref_reset_kernel(RaftDevState d, int S, int Lcap, int P, int Ecap) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    d.event_seq[0] = 0;
    d.rpc_seq_next[0] = 1;
    d.has_leader[0] = 0;
    d.leader_id[0] = 0;
    d.event_hash[0] = RAFT_FNV_INIT;
    d.pending_count[0] = 0;
    for (int s = 0; s < S; ++s) {
        d.current_term[s] = 0;
        d.role[s] = RAFT_ROLE_FOLLOWER;
        d.snap_index[s] = 0;
        d.snap_term[s] = 0;
        d.snap_hash[s] = 0;
        d.commit_index[s] = 0;
        d.last_applied[s] = 0;
        d.apply_acc[s] = 0;
        d.log_count[s] = 0;
        d.next_index[s] = 0;
        d.match_index[s] = 0;
    }
    for (int i = 0; i < 24; ++i) d.counts[i] = 0;
    (void)Lcap; (void)P; (void)Ecap;
}

__global__ void raft_ref_step_kernel(
    RaftDevState d, int S, int Lcap, int P, int Ecap,
    int maxlog, int max_epa, int max_apply, int majority,
    int num_ops, const RaftOp* ops,
    int64_t* out_counts, uint64_t* out_event, uint64_t* out_log,
    uint64_t* out_leader, uint64_t* out_pending, uint64_t* out_apply) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    RaftCtx c;
    c.S = S; c.Lcap = Lcap; c.P = P; c.Ecap = Ecap;
    c.maxlog = maxlog; c.max_epa = max_epa; c.max_apply = max_apply;
    c.majority = majority; c.d = d;

    for (int i = 0; i < num_ops; ++i) {
        const RaftOp o = ops[i];
        switch (o.kind) {
            case RAFT_OP_BECOME_LEADER:  r_become_leader(c, (uint32_t)i, o.i_a, o.u_a); break;
            case RAFT_OP_CLIENT_APPEND:  r_client_append(c, (uint32_t)i, o.u_a, o.value); break;
            case RAFT_OP_SEND_APPEND:    r_send_append(c, (uint32_t)i, o.i_a, o.i_b); break;
            case RAFT_OP_DELIVER_APPEND: r_deliver_append(c, (uint32_t)i, o.u_a); break;
            case RAFT_OP_ADVANCE_COMMIT: r_advance_commit(c, (uint32_t)i); break;
            case RAFT_OP_APPLY:          r_apply(c, (uint32_t)i, o.i_a, o.i_b); break;
            case RAFT_OP_TAKE_SNAPSHOT:  r_take_snapshot(c, (uint32_t)i, o.i_a); break;
            case RAFT_OP_INSTALL_SNAPSHOT: r_install_snapshot(c, (uint32_t)i, o.i_a); break;
            default: r_invalid(c, (uint32_t)i, (uint64_t)(uint32_t)o.kind); break;
        }
    }

    for (int i = 0; i < 24; ++i) out_counts[i] = d.counts[i];
    out_event[0] = d.event_hash[0];
    r_write_hashes(c, out_log, out_leader, out_pending, out_apply);
}

// ---------------------------------------------------------------------------
// Host allocation / API
// ---------------------------------------------------------------------------
static cudaError_t raft_alloc_state(RaftReferenceState* st) {
    const int S = st->spec.server_count;
    const int Lcap = st->spec.max_log_entries_per_server;
    const int P = st->spec.max_pending_append_rpcs;
    const int Ecap = st->spec.max_entries_per_append;
    RaftDevState& d = st->dev;
    cudaError_t err = cudaSuccess;

#define ALLOC(field, type, n)                                                          \
    err = cudaMalloc(reinterpret_cast<void**>(&d.field), sizeof(type) * (size_t)(n));  \
    if (err != cudaSuccess) return err;

    ALLOC(event_seq, uint64_t, 1);
    ALLOC(rpc_seq_next, uint64_t, 1);
    ALLOC(has_leader, int32_t, 1);
    ALLOC(leader_id, uint32_t, 1);
    ALLOC(event_hash, uint64_t, 1);
    ALLOC(current_term, uint64_t, S);
    ALLOC(role, int32_t, S);
    ALLOC(snap_index, uint64_t, S);
    ALLOC(snap_term, uint64_t, S);
    ALLOC(snap_hash, uint64_t, S);
    ALLOC(commit_index, uint64_t, S);
    ALLOC(last_applied, uint64_t, S);
    ALLOC(apply_acc, uint64_t, S);
    ALLOC(log_count, int32_t, S);
    ALLOC(log_index, uint64_t, (size_t)S * Lcap);
    ALLOC(log_term, uint64_t, (size_t)S * Lcap);
    ALLOC(log_cmd, uint64_t, (size_t)S * Lcap);
    ALLOC(log_payload, int64_t, (size_t)S * Lcap);
    ALLOC(next_index, uint64_t, S);
    ALLOC(match_index, uint64_t, S);
    ALLOC(pending_count, int32_t, 1);
    ALLOC(p_rpc_id, uint64_t, P);
    ALLOC(p_leader, uint32_t, P);
    ALLOC(p_follower, uint32_t, P);
    ALLOC(p_leader_term, uint64_t, P);
    ALLOC(p_prev_index, uint64_t, P);
    ALLOC(p_prev_term, uint64_t, P);
    ALLOC(p_leader_commit, uint64_t, P);
    ALLOC(p_send_seq, uint64_t, P);
    ALLOC(p_ecount, int32_t, P);
    ALLOC(p_eindex, uint64_t, (size_t)P * Ecap);
    ALLOC(p_eterm, uint64_t, (size_t)P * Ecap);
    ALLOC(p_ecmd, uint64_t, (size_t)P * Ecap);
    ALLOC(p_epayload, int64_t, (size_t)P * Ecap);
    ALLOC(counts, int64_t, 24);
#undef ALLOC
    return cudaSuccess;
}

static void raft_free_state(RaftReferenceState* st) {
    RaftDevState& d = st->dev;
    void* ptrs[] = {
        d.event_seq, d.rpc_seq_next, d.has_leader, d.leader_id, d.event_hash,
        d.current_term, d.role, d.snap_index, d.snap_term, d.snap_hash,
        d.commit_index, d.last_applied, d.apply_acc, d.log_count, d.log_index,
        d.log_term, d.log_cmd, d.log_payload, d.next_index, d.match_index,
        d.pending_count, d.p_rpc_id, d.p_leader, d.p_follower, d.p_leader_term,
        d.p_prev_index, d.p_prev_term, d.p_leader_commit, d.p_send_seq,
        d.p_ecount, d.p_eindex, d.p_eterm, d.p_ecmd, d.p_epayload, d.counts};
    for (void* p : ptrs) if (p) cudaFree(p);
    if (st->d_ops) cudaFree(st->d_ops);
}

static cudaError_t raft_reset_state(RaftReferenceState* st, cudaStream_t stream) {
    raft_ref_reset_kernel<<<1, 1, 0, stream>>>(
        st->dev, st->spec.server_count, st->spec.max_log_entries_per_server,
        st->spec.max_pending_append_rpcs, st->spec.max_entries_per_append);
    return cudaPeekAtLastError();
}

extern "C" size_t solution_workspace_bytes(const RaftProblemSpec* spec) {
    if (!raft_validate_problem_spec(spec)) return 0;
    return 256;
}

extern "C" cudaError_t solution_init(const RaftProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!raft_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;

    RaftReferenceState* st = static_cast<RaftReferenceState*>(malloc(sizeof(RaftReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(RaftReferenceState));
    memcpy(&st->spec, spec, sizeof(RaftProblemSpec));

    cudaError_t err = raft_alloc_state(st);
    if (err != cudaSuccess) { raft_free_state(st); free(st); return err; }

    err = cudaMalloc(reinterpret_cast<void**>(&st->d_ops), sizeof(RaftOp) * (size_t)spec->max_ops);
    if (err != cudaSuccess) { raft_free_state(st); free(st); return err; }

    err = raft_reset_state(st, stream);
    if (err != cudaSuccess) { raft_free_state(st); free(st); return err; }

    *state_out = st;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state, const RaftRunSpec* run, const void* inputs_void,
    void* outputs_void, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    (void)workspace;
    if (!state || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 256) return cudaErrorInvalidValue;

    RaftReferenceState* st = static_cast<RaftReferenceState*>(state);
    if (!raft_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const RaftInputs* in = static_cast<const RaftInputs*>(inputs_void);
    RaftOutputs* out = static_cast<RaftOutputs*>(outputs_void);
    if (run->num_ops > 0 && (!in || !in->ops)) return cudaErrorInvalidValue;
    if (!out->counts || !out->raft_event_hash || !out->log_hash ||
        !out->leader_state_hash || !out->pending_rpc_hash || !out->apply_hash) {
        return cudaErrorInvalidValue;
    }

    if (run->num_ops > 0) {
        cudaError_t e = cudaMemcpyAsync(st->d_ops, in->ops, sizeof(RaftOp) * (size_t)run->num_ops,
                                        cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return e;
    }

    const int majority = st->spec.server_count / 2 + 1;
    raft_ref_step_kernel<<<1, 1, 0, stream>>>(
        st->dev, st->spec.server_count, st->spec.max_log_entries_per_server,
        st->spec.max_pending_append_rpcs, st->spec.max_entries_per_append,
        st->spec.max_log_entries_per_server, st->spec.max_entries_per_append,
        st->spec.max_apply_per_op, majority,
        run->num_ops, st->d_ops,
        reinterpret_cast<int64_t*>(out->counts), out->raft_event_hash, out->log_hash,
        out->leader_state_hash, out->pending_rpc_hash, out->apply_hash);

    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return raft_reset_state(static_cast<RaftReferenceState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    RaftReferenceState* st = static_cast<RaftReferenceState*>(state);
    raft_free_state(st);
    free(st);
}
