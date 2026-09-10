// file: raft_log_snapshot_common.h

#ifndef RAFT_LOG_SNAPSHOT_COMMON_H_
#define RAFT_LOG_SNAPSHOT_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define RAFT_ABI_VERSION 1

// Capacity bounds (compile-time maxima used to size persistent buffers).
#define RAFT_MIN_SERVERS 1
#define RAFT_MAX_SERVERS 16
#define RAFT_MAX_LOG_ENTRIES_PER_SERVER 256
#define RAFT_MAX_PENDING_APPEND_RPCS 64
#define RAFT_MAX_ENTRIES_PER_APPEND 64
#define RAFT_MAX_APPLY_PER_OP 256
#define RAFT_MAX_OPS 256
#define RAFT_MAX_STEPS 64

// Roles.
#define RAFT_ROLE_FOLLOWER 0
#define RAFT_ROLE_LEADER 1

// Operation kinds (op.kind).
#define RAFT_OP_BECOME_LEADER 0
#define RAFT_OP_CLIENT_APPEND 1
#define RAFT_OP_SEND_APPEND 2
#define RAFT_OP_DELIVER_APPEND 3
#define RAFT_OP_ADVANCE_COMMIT 4
#define RAFT_OP_APPLY 5
#define RAFT_OP_TAKE_SNAPSHOT 6
#define RAFT_OP_INSTALL_SNAPSHOT 7

// raft_event_hash event kinds (event_kind:u8).
#define RAFT_EV_BECOME_LEADER_OK 0
#define RAFT_EV_CLIENT_APPEND_OK 1
#define RAFT_EV_CLIENT_REJECT_LOG_FULL 2
#define RAFT_EV_APPEND_SEND 3
#define RAFT_EV_APPEND_SEND_REJECT 4
#define RAFT_EV_APPEND_NEEDS_SNAPSHOT 5
#define RAFT_EV_APPEND_SUCCESS 6
#define RAFT_EV_APPEND_STALE 7
#define RAFT_EV_APPEND_TERM_REJECT 8
#define RAFT_EV_APPEND_CONFLICT 9
#define RAFT_EV_FOLLOWER_DELETE_SUFFIX 10
#define RAFT_EV_FOLLOWER_APPEND_ENTRY 11
#define RAFT_EV_APPEND_FOLLOWER_OOM 12
#define RAFT_EV_COMMIT_ADVANCE 13
#define RAFT_EV_COMMIT_NOOP 14
#define RAFT_EV_APPLY_ENTRY 15
#define RAFT_EV_APPLY_EMPTY 16
#define RAFT_EV_SNAPSHOT_TRUNCATE_LOCAL 17
#define RAFT_EV_SNAPSHOT_TAKE 18
#define RAFT_EV_SNAPSHOT_INSTALL 19
#define RAFT_EV_SNAPSHOT_INSTALL_NOOP 20
#define RAFT_EV_SNAPSHOT_TRUNCATE_REMOTE 21
#define RAFT_EV_INVALID 22

/*
CONTRACT: raft_log_snapshot

A persistent simulated Raft cluster with deterministic leader changes,
AppendEntries RPC state, conflict backtracking, current-term commit
advancement, per-server apply streams, and snapshot installation/truncation.

Grounded in Raft (a consensus algorithm for a replicated log with log
compaction via snapshots).

Each step is a batch of ops applied in order. Ops are dispatched by
op.kind (RAFT_OP_*). After each step the harness reads cumulative counts
plus structural checksums. Every graded output is an EXACT integer
(cumulative counts + FNV-1a-64 checksums).

PERSISTENT STATE
  event_seq = 0 ; rpc_seq_next = 1.
  At most one current leader (has_leader, leader_id).
  Per server s: current_term; role; snapshot_index; snapshot_term;
    snapshot_state_hash; log (entries with index > snapshot_index, sorted
    ascending); commit_index; last_applied; apply_accumulator.
  Log entry: index; term; command_id; payload_i64.
  Leader volatile (meaningful only for current leader): per server f
    next_index[f], match_index[f]. For the leader itself
    match_index[leader] = last_log_index(leader).
  Pending AppendEntries RPC table: rpc_id; leader; follower; leader_term;
    prev_index; prev_term; copied entries; leader_commit; send_seq.

DEFINITIONS
  majority = server_count / 2 + 1.
  last_log_index(s) = snapshot_index[s] if log empty, else index of last
    log entry.

ABI: solution_init may allocate persistent device state. solution_run may
launch kernels and use the provided workspace but may not call
cudaMalloc/cudaFree. solution_run must not mutate its inputs.

The COMPLETE, NORMATIVE per-op semantics and the EXACT checksum
serialization are inlined below in the section
"=== DETERMINISM & EXACT OUTPUT SERIALIZATION (normative) ===". Reading
only this header is sufficient to reproduce every graded integer and
checksum bit-for-bit.
*/

struct alignas(8) RaftProblemSpec {
    int32_t abi_version;
    int32_t server_count;                  // [1, RAFT_MAX_SERVERS]
    int32_t max_log_entries_per_server;    // <= RAFT_MAX_LOG_ENTRIES_PER_SERVER
    int32_t max_pending_append_rpcs;       // <= RAFT_MAX_PENDING_APPEND_RPCS
    int32_t max_entries_per_append;        // <= RAFT_MAX_ENTRIES_PER_APPEND
    int32_t max_apply_per_op;              // <= RAFT_MAX_APPLY_PER_OP
    int32_t max_ops;                       // <= RAFT_MAX_OPS
    int32_t max_steps;                     // <= RAFT_MAX_STEPS
    int32_t flags;
    int32_t reserved[7];
};

// One operation within a step.
//   BECOME_LEADER(server=i_a, term=u_a)
//   CLIENT_APPEND(command_id=u_a, payload=value)
//   SEND_APPEND(follower=i_a, max_entries=i_b)
//   DELIVER_APPEND(rpc_id=u_a)
//   ADVANCE_COMMIT()
//   APPLY(server=i_a, limit=i_b)
//   TAKE_SNAPSHOT(server=i_a)
//   INSTALL_SNAPSHOT(follower=i_a)
struct alignas(8) RaftOp {
    int32_t kind;     // RAFT_OP_*
    int32_t i_a;      // server / follower
    int32_t i_b;      // max_entries / limit
    int32_t i_c;      // reserved
    int64_t value;    // CLIENT_APPEND payload_i64
    uint64_t u_a;     // term / command_id / rpc_id
    uint64_t u_b;     // reserved
};

struct alignas(8) RaftRunSpec {
    int32_t abi_version;
    int32_t num_ops;
    int32_t step_id;
    int32_t reserved[5];
};

struct alignas(8) RaftInputs {
    const RaftOp* ops;
};

// All cumulative across the run (never reset between steps, only by reset).
struct alignas(8) RaftCounts {
    int64_t leaders_elected;
    int64_t client_appended;
    int64_t client_rejected;
    int64_t append_sent;
    int64_t append_send_rejected;
    int64_t append_needs_snapshot;
    int64_t append_success;
    int64_t append_stale;
    int64_t append_term_reject;
    int64_t append_conflict;
    int64_t follower_appended_entries;
    int64_t follower_deleted_suffixes;
    int64_t append_follower_oom;
    int64_t commit_advanced;
    int64_t commit_noop;
    int64_t applied_entries;
    int64_t apply_empty;
    int64_t snapshots_taken;
    int64_t snapshots_installed;
    int64_t snapshot_noop;
    int64_t snapshot_truncations;
    int64_t invalid_count;
    int64_t reserved[2];
};

struct alignas(8) RaftOutputs {
    RaftCounts* counts;            // 1
    uint64_t* raft_event_hash;     // 1
    uint64_t* log_hash;            // 1
    uint64_t* leader_state_hash;   // 1
    uint64_t* pending_rpc_hash;    // 1
    uint64_t* apply_hash;          // 1
};

static_assert(sizeof(RaftProblemSpec) == 64, "RaftProblemSpec layout drift");
static_assert(sizeof(RaftOp) == 40, "RaftOp layout drift");
static_assert(sizeof(RaftRunSpec) == 32, "RaftRunSpec layout drift");
static_assert(sizeof(RaftInputs) == 8, "RaftInputs layout drift");
static_assert(sizeof(RaftCounts) == 24 * 8, "RaftCounts layout drift");
static_assert(sizeof(RaftOutputs) == 48, "RaftOutputs layout drift");

static inline int raft_validate_problem_spec(const RaftProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != RAFT_ABI_VERSION) return 0;
    if (spec->server_count < RAFT_MIN_SERVERS || spec->server_count > RAFT_MAX_SERVERS) return 0;
    if (spec->max_log_entries_per_server < 1 ||
        spec->max_log_entries_per_server > RAFT_MAX_LOG_ENTRIES_PER_SERVER) return 0;
    if (spec->max_pending_append_rpcs < 1 ||
        spec->max_pending_append_rpcs > RAFT_MAX_PENDING_APPEND_RPCS) return 0;
    if (spec->max_entries_per_append < 1 ||
        spec->max_entries_per_append > RAFT_MAX_ENTRIES_PER_APPEND) return 0;
    if (spec->max_apply_per_op < 1 ||
        spec->max_apply_per_op > RAFT_MAX_APPLY_PER_OP) return 0;
    if (spec->max_ops < 1 || spec->max_ops > RAFT_MAX_OPS) return 0;
    if (spec->max_steps < 1 || spec->max_steps > RAFT_MAX_STEPS) return 0;
    return 1;
}

static inline int raft_validate_run_spec(const RaftRunSpec* run, const RaftProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != RAFT_ABI_VERSION) return 0;
    if (run->num_ops < 0 || run->num_ops > spec->max_ops) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const RaftProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const RaftProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const RaftRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

// === DETERMINISM & EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section fully specifies every graded integer (the 24-slot RaftCounts
// and the five FNV-1a-64 checksums) bit-for-bit. It is derived exactly from
// the reference implementation. A solution reading only this header can
// reproduce all outputs. Nothing here defers to any held-out file.
//
// ----------------------------------------------------------------------------
// 0. FNV-1a-64 PRIMITIVES (the ONE hash used everywhere)
// ----------------------------------------------------------------------------
//   offset basis (FNV_INIT) = 1469598103934665603  (0x14650FB0739D0383)
//   prime                   = 1099511628211         (0x00000100000001B3)
//   Per-byte step:  h ^= byte;  h *= prime;   (all in uint64_t, wrapping)
//   A field is folded by hashing its bytes LITTLE-ENDIAN, exactly as many
//   bytes as its DECLARED width:
//     u8  v          -> 1 byte:  fnv_byte(v & 0xFF)
//     u32 v          -> 4 bytes: for i in 0..3: fnv_byte((v >> (8*i)) & 0xFF)
//     u64 v          -> 8 bytes: for i in 0..7: fnv_byte((v >> (8*i)) & 0xFF)
//     i64 v          -> reinterpreted as u64 (two's complement) then u64 fold
//   There are NO separators, NO length prefixes, and NO terminators inserted
//   between fields beyond what is explicitly listed below (the pending-rpc
//   entry count and the per-RPC entry list ARE explicit fields; see 6.4).
//   Sentinels used as field values: U32_MAX = 0xFFFFFFFF, U64_MAX =
//   0xFFFFFFFFFFFFFFFF.
//
// ----------------------------------------------------------------------------
// 1. SEEDING / PERSISTENCE OF EACH HASH
// ----------------------------------------------------------------------------
//   raft_event_hash  : PERSISTENT (running) across all ops and all steps.
//                      Seeded to FNV_INIT only by solution_init and by
//                      solution_reset. Each emitted event folds into it in
//                      emission order (see 5). Never recomputed from scratch.
//   log_hash, leader_state_hash, pending_rpc_hash, apply_hash : RESEEDED to
//                      FNV_INIT and RECOMPUTED FROM SCRATCH at the END of every
//                      step (after all ops in that step's batch have run), from
//                      the current persistent state. (see 6)
//   event_seq        : PERSISTENT u64, starts at 0 (reset by init/reset).
//   rpc_seq_next     : PERSISTENT u64, starts at 1 (reset by init/reset).
//   counts (RaftCounts) : PERSISTENT cumulative i64[24], all 0 at init/reset,
//                      incremented by ops, never recomputed.
//
// ----------------------------------------------------------------------------
// 2. PERSISTENT STATE (reset by init/reset to these values)
// ----------------------------------------------------------------------------
//   Globals: event_seq=0; rpc_seq_next=1; has_leader=false; leader_id=0;
//            event_hash=FNV_INIT; pending table empty; counts all 0.
//   Per server s in [0,server_count): current_term=0; role=FOLLOWER;
//            snapshot_index=0; snapshot_term=0; snapshot_state_hash=0;
//            commit_index=0; last_applied=0; apply_accumulator=0; log empty;
//            next_index[s]=0; match_index[s]=0.
//   A server's log holds entries with index > snapshot_index, stored ascending
//   by index. Entry fields: index(u64), term(u64), command_id(u64),
//   payload(i64). next_index/match_index are a SINGLE global volatile array
//   indexed by server (meaningful for the current leader); they are NOT
//   re-seeded per leader election except as op_become_leader sets them.
//   majority = server_count / 2 + 1  (integer division).
//   last_log_index(s) = (log empty) ? snapshot_index[s] : index of last entry.
//   log_at(s, idx): linear scan of server s's log; returns the entry whose
//     index == idx, else "absent".
//
// ----------------------------------------------------------------------------
// 3. STEP / DISPATCH
// ----------------------------------------------------------------------------
//   A step runs ops[0..num_ops) in order. op_index passed to each op is its
//   0-based position i within THIS step's batch. Dispatch by op.kind:
//     0 BECOME_LEADER(server=i_a, term=u_a)
//     1 CLIENT_APPEND(command_id=u_a, payload=value)
//     2 SEND_APPEND(follower=i_a, max_entries=i_b)
//     3 DELIVER_APPEND(rpc_id=u_a)
//     4 ADVANCE_COMMIT()
//     5 APPLY(server=i_a, limit=i_b)
//     6 TAKE_SNAPSHOT(server=i_a)
//     7 INSTALL_SNAPSHOT(follower=i_a)
//   Any unknown kind -> INVALID with aux = (u64)(u32)op.kind (see 4/5).
//   After all ops, the four structural hashes are recomputed (see 6) and the
//   cumulative counts + persistent event_hash are reported.
//
// ----------------------------------------------------------------------------
// 4. COUNTS: field order (index 0..23) — MUST match RaftCounts struct order
// ----------------------------------------------------------------------------
//   0 leaders_elected      1 client_appended     2 client_rejected
//   3 append_sent          4 append_send_rejected 5 append_needs_snapshot
//   6 append_success       7 append_stale        8 append_term_reject
//   9 append_conflict     10 follower_appended_entries
//  11 follower_deleted_suffixes 12 append_follower_oom
//  13 commit_advanced     14 commit_noop        15 applied_entries
//  16 apply_empty         17 snapshots_taken     18 snapshots_installed
//  19 snapshot_noop       20 snapshot_truncations 21 invalid_count
//  22 reserved0           23 reserved1   (reserved stay 0)
//   Each listed event below increments exactly the named counter by 1, EXCEPT
//   follower entry appends (increments follower_appended_entries once PER
//   appended entry) and applied entries (increments applied_entries once PER
//   applied entry).
//
// ----------------------------------------------------------------------------
// 5. EVENT EMISSION (folds into the persistent raft_event_hash)
// ----------------------------------------------------------------------------
//   emit(kind, op_index, server, peer, term, index_or_max, count_or_zero, aux)
//   folds, IN THIS EXACT FIELD ORDER AND WIDTH, into raft_event_hash:
//       u8  kind
//       u64 event_seq      (the CURRENT event_seq value, BEFORE increment)
//       u32 op_index
//       u32 server
//       u32 peer
//       u64 term
//       u64 index_or_max
//       u64 count_or_zero
//       u64 aux
//   then event_seq += 1 (post-increment: the event carries the pre-increment
//   value). Events fold in the order they are emitted within an op, and ops
//   run in batch order. Event kinds (RAFT_EV_*) are the u8 'kind' above.
//   The exact (server,peer,term,index_or_max,count_or_zero,aux) tuple for each
//   event is specified inline per-op in section 7. For ALL INVALID events:
//   kind=INVALID(22), server=U32_MAX, peer=U32_MAX, term=0,
//   index_or_max=U64_MAX, count_or_zero=0, aux = the originating op kind
//   (BECOME_LEADER..INSTALL_SNAPSHOT), or (u64)(u32)op.kind for unknown kinds.
//
// ----------------------------------------------------------------------------
// 6. STRUCTURAL CHECKSUMS (recomputed from scratch at end of step)
// ----------------------------------------------------------------------------
// Iteration over servers is ALWAYS ascending s = 0..server_count-1. Iteration
// over a server's log is ALWAYS in stored ascending-index order.
//
// 6.1 log_hash  (seed FNV_INIT):
//   for s in 0..server_count-1:
//     u32 s
//     u64 snapshot_index[s]
//     u64 snapshot_term[s]
//     u64 snapshot_state_hash[s]
//     for each log entry e of server s (ascending index):
//       u64 e.index ; u64 e.term ; u64 e.command_id ; i64 e.payload
//
// 6.2 leader_state_hash  (seed FNV_INIT):
//   u8  has_leader ? 1 : 0
//   u32 has_leader ? leader_id : U32_MAX
//   for s in 0..server_count-1:
//     u32 s ; u64 next_index[s] ; u64 match_index[s]
//
// 6.3 apply_hash  (seed FNV_INIT):
//   for s in 0..server_count-1:
//     u32 s ; u64 commit_index[s] ; u64 last_applied[s] ; u64 apply_accumulator[s]
//
// 6.4 pending_rpc_hash  (seed FNV_INIT):
//   Hash pending RPCs in ASCENDING rpc_id order (a stable sort/selection over
//   the pending table by rpc_id; rpc_ids are unique). For each RPC R:
//     u64 R.rpc_id
//     u32 R.leader
//     u32 R.follower
//     u64 R.leader_term
//     u64 R.prev_index
//     u64 R.prev_term
//     u64 R.leader_commit
//     u64 R.send_seq          (the event_seq value at the moment SEND_APPEND
//                              emitted its APPEND_SEND event for this RPC)
//     u64 R.entries_count
//     for each copied entry e (in the ascending-index order they were copied):
//       u64 e.index ; u64 e.term ; u64 e.command_id ; i64 e.payload
//
//   NOTE: there is NO master "state_checksum" combining sub-hashes. The five
//   outputs (event/log/leader_state/pending_rpc/apply) plus the 24 counts are
//   each graded independently and exactly.
//
// ----------------------------------------------------------------------------
// 7. EXACT OP SEMANTICS (term/index arithmetic, tie-breaks, edge cases)
// ----------------------------------------------------------------------------
// Notation: L = current leader_id; F = a follower; S = server_count.
//
// 7.1 BECOME_LEADER(server, term):
//   INVALID (counts.invalid_count) if server<0 || server>=S ||
//     term < current_term[server].
//   Else: for every other server in LEADER role, demote it to FOLLOWER (scan
//   s=0..S-1). Set current_term[server]=term; role[server]=LEADER;
//   has_leader=true; leader_id=server. Let lli=last_log_index(server). For each
//   f != server (scan f=0..S-1): next_index[f]=lli+1; match_index[f]=0. Then
//   match_index[server]=lli; next_index[server]=lli+1.
//   counts.leaders_elected += 1. emit BECOME_LEADER_OK(0):
//     server=server, peer=U32_MAX, term=term, index_or_max=lli,
//     count_or_zero=0, aux=0.
//
// 7.2 CLIENT_APPEND(command_id, payload):
//   INVALID if !has_leader.
//   Else if leader's log size >= max_log_entries_per_server:
//     counts.client_rejected += 1. emit CLIENT_REJECT_LOG_FULL(2):
//       server=L, peer=U32_MAX, term=current_term[L], index_or_max=U64_MAX,
//       count_or_zero=0, aux=0.
//   Else: idx=last_log_index(L)+1. Append entry {idx, current_term[L],
//     command_id, payload} to L's log (ascending; appended at end). Set
//     match_index[L]=last_log_index(L); next_index[L]=last_log_index(L)+1.
//     counts.client_appended += 1. emit CLIENT_APPEND_OK(1):
//       server=L, peer=U32_MAX, term=current_term[L], index_or_max=idx,
//       count_or_zero=0, aux=command_id.
//
// 7.3 SEND_APPEND(follower, max_entries):
//   INVALID if !has_leader || follower<0 || follower>=S ||
//     follower==L || max_entries==0.
//   Else if pending table size >= max_pending_append_rpcs:
//     counts.append_send_rejected += 1. emit APPEND_SEND_REJECT(4):
//       server=L, peer=follower, term=current_term[L], index_or_max=U64_MAX,
//       count_or_zero=0, aux=0.
//   Let ni=next_index[follower].
//   Else if ni <= snapshot_index[L]:
//     counts.append_needs_snapshot += 1. emit APPEND_NEEDS_SNAPSHOT(5):
//       server=L, peer=follower, term=current_term[L],
//       index_or_max=snapshot_index[L], count_or_zero=0, aux=0.
//   prev_index = ni - 1.
//     if prev_index == snapshot_index[L]: prev_term = snapshot_term[L].
//     else: pe = log_at(L, prev_index); if absent ->
//       counts.append_send_rejected += 1; emit APPEND_SEND_REJECT(4) with
//       server=L, peer=follower, term=current_term[L], index_or_max=U64_MAX,
//       count_or_zero=0, aux=0; return. Else prev_term = pe.term.
//   cap = min(max_entries, max_entries_per_append). Copy, in ascending index
//   order, the first up-to-cap log entries of L with index >= ni.
//   Create RPC: rpc_id = rpc_seq_next (then rpc_seq_next += 1); leader=L;
//   follower=follower; leader_term=current_term[L]; prev_index; prev_term;
//   leader_commit=commit_index[L]; send_seq = CURRENT event_seq (the value the
//   about-to-be-emitted APPEND_SEND event will carry); entries = the copied
//   list. Append RPC to pending table (insertion order; hashing later sorts by
//   rpc_id).
//   counts.append_sent += 1. emit APPEND_SEND(3):
//     server=L, peer=follower, term=current_term[L], index_or_max=prev_index,
//     count_or_zero=(copied entry count), aux=rpc_id.
//
// 7.4 DELIVER_APPEND(rpc_id):
//   Find the pending RPC with matching rpc_id (linear scan, first match). If
//   none: INVALID (aux=DELIVER_APPEND). Else snapshot its fields R and REMOVE
//   it from the pending table exactly once (compacting the table, preserving
//   the relative order of remaining RPCs).
//   Stale check: if NOT(has_leader && leader_id==R.leader) OR
//     current_term[R.leader] != R.leader_term:
//     counts.append_stale += 1. emit APPEND_STALE(7):
//       server=R.leader, peer=R.follower, term=R.leader_term,
//       index_or_max=R.rpc_id, count_or_zero=0, aux=0. return.
//   F = R.follower.
//   Term-reject: if current_term[F] > R.leader_term:
//     counts.append_term_reject += 1. ni=next_index[F];
//     newni = (ni>1) ? ni-1 : 1; next_index[F]=newni.
//     emit APPEND_TERM_REJECT(8): server=R.leader, peer=R.follower,
//       term=R.leader_term, index_or_max=R.prev_index, count_or_zero=0,
//       aux=newni. return.
//   Else set current_term[F] = R.leader_term (adopt leader term).
//   Consistency check (determines fail / consistent, plus conflict info):
//     if R.prev_index < snapshot_index[F]:
//         fail; conflict_index = snapshot_index[F] + 1; (no conflict_term)
//     else if R.prev_index == snapshot_index[F]:
//         if R.prev_term == snapshot_term[F]: consistent.
//         else: fail; conflict_index = last_log_index(F) + 1; (no conflict_term)
//     else:
//         pe = log_at(F, R.prev_index);
//         if absent: fail; conflict_index = last_log_index(F)+1; (no conflict_term)
//         else if pe.term == R.prev_term: consistent.
//         else: fail; has_conflict_term=true; conflict_term = pe.term;
//               conflict_index = first_index_with_term(F, pe.term)
//               (first index, ascending scan, whose term == pe.term).
//   On fail:
//     if has_conflict_term:
//         lt = leader_last_plus_one_with_term(R.leader, conflict_term)
//              = (1 + the LARGEST leader-log index whose term==conflict_term),
//                or 0 if the leader has no entry with that term (ascending scan
//                keeping the last match).
//         newni = (lt != 0) ? lt : conflict_index.
//     else: newni = conflict_index.
//     next_index[F] = newni. counts.append_conflict += 1.
//     emit APPEND_CONFLICT(9): server=R.leader, peer=R.follower,
//       term=R.leader_term, index_or_max=conflict_index,
//       count_or_zero = (has_conflict_term ? conflict_term : U64_MAX),
//       aux=newni. return.
//   On consistent (success path):
//     del_from: scan R.entries in ascending index; the FIRST incoming entry
//       whose index exists in F's log with a DIFFERENT term sets
//       del_from=that index (then stop). If none, del_from = U64_MAX (none).
//     Compute survivors / deletion stats:
//       if del_from != U64_MAX: scan F's log; entries with index < del_from are
//         survivors; entries with index >= del_from are the deleted suffix;
//         deleted_count = count of those; first_deleted_index = the first such
//         index encountered (ascending). Else survivors = F's whole log; no
//         deletion.
//     Determine entries to append: for each incoming entry e (ascending):
//       skip if e.index <= snapshot_index[F] (covered by snapshot); else
//       "present" is (e.index < del_from) if del_from!=U64_MAX, otherwise
//       (log_at(F,e.index) exists). Append e iff NOT present.
//     final_size = survivors + (number to append).
//     OOM PRE-CHECK: if final_size > max_log_entries_per_server:
//       counts.append_follower_oom += 1. emit APPEND_FOLLOWER_OOM(12):
//         server=R.leader, peer=R.follower, term=R.leader_term,
//         index_or_max=R.prev_index, count_or_zero=final_size, aux=0.
//       DO NOTHING ELSE (no deletion, no append, no commit/match update).
//       return.
//     Else apply, IN THIS ORDER:
//       (a) Deletion: if del_from!=U64_MAX && deleted_count>0: keep only
//           entries with index < del_from. counts.follower_deleted_suffixes
//           += 1. emit FOLLOWER_DELETE_SUFFIX(10): server=R.leader,
//           peer=R.follower, term=R.leader_term,
//           index_or_max=first_deleted_index, count_or_zero=deleted_count,
//           aux=0.
//       (b) Append each to-append entry e (in ascending incoming order; each
//           strictly greater than any survivor, appended at end keeping log
//           ascending): counts.follower_appended_entries += 1 PER entry. emit
//           FOLLOWER_APPEND_ENTRY(11): server=R.leader, peer=R.follower,
//           term=e.term, index_or_max=e.index, count_or_zero=0,
//           aux=e.command_id.
//       (c) lliF = last_log_index(F); newcommit = min(R.leader_commit, lliF);
//           commit_index[F] = newcommit.
//       (d) last_in_r = (R.entries empty ? R.prev_index : index of LAST copied
//           entry); newmatch = max(match_index[F], last_in_r);
//           match_index[F]=newmatch; next_index[F]=newmatch+1.
//       (e) counts.append_success += 1. emit APPEND_SUCCESS(6):
//           server=R.leader, peer=R.follower, term=R.leader_term,
//           index_or_max=newmatch, count_or_zero=(R.entries count), aux=newcommit.
//
// 7.5 ADVANCE_COMMIT():
//   INVALID if !has_leader.
//   lli=last_log_index(L). Scan N from commit_index[L]+1 to lli inclusive,
//   ascending. For each N: let e=log_at(L,N); skip if e absent or
//   e.term != current_term[L] (commit only current-term entries). Count
//   replicas: cnt = number of servers s (s=0..S-1) with
//   mi >= N, where mi = (s==L ? lli : match_index[s]). If cnt >= majority,
//   record bestN = N (KEEP THE LARGEST qualifying N: assignment overwrites as N
//   ascends). After the scan, if any N qualified: commit_index[L]=bestN;
//     counts.commit_advanced += 1. emit COMMIT_ADVANCE(13): server=L,
//     peer=U32_MAX, term=current_term[L], index_or_max=bestN, count_or_zero=0,
//     aux=0.
//   Else: counts.commit_noop += 1. emit COMMIT_NOOP(14): server=L,
//     peer=U32_MAX, term=current_term[L], index_or_max=commit_index[L],
//     count_or_zero=0, aux=0.
//
// 7.6 APPLY(server, limit):
//   INVALID if server<0 || server>=S.
//   If limit==0: valid NO-OP — emit nothing, count nothing (not even invalid).
//   cap = min(limit, max_apply_per_op). Loop up to cap times:
//     want = last_applied[server]+1; stop if want > commit_index[server];
//     e=log_at(server, want); stop if absent (covered/missing).
//     Fold into apply_accumulator[server] (which is PERSISTENT, running):
//       u64 e.index ; u64 e.term ; u64 e.command_id ; i64 e.payload.
//     last_applied[server]=e.index; applied++; counts.applied_entries += 1.
//     emit APPLY_ENTRY(15): server=server, peer=U32_MAX, term=e.term,
//       index_or_max=e.index, count_or_zero=0, aux=e.command_id.
//   If applied==0 (nothing applied, limit!=0): counts.apply_empty += 1.
//     emit APPLY_EMPTY(16): server=server, peer=U32_MAX,
//       term=current_term[server], index_or_max=last_applied[server],
//       count_or_zero=0, aux=0.
//
// 7.7 TAKE_SNAPSHOT(server):
//   INVALID if server<0 || server>=S || last_applied[server] <=
//     snapshot_index[server].
//   upto = last_applied[server]. e=log_at(server,upto);
//   snap_term = (e present ? e.term : snapshot_term[server]).
//   Set snapshot_index[server]=upto; snapshot_term[server]=snap_term;
//   snapshot_state_hash[server]=apply_accumulator[server].
//   Delete log entries with index <= upto (keep index > upto). deleted = number
//   removed. If deleted > 0: counts.snapshot_truncations += 1. emit
//   SNAPSHOT_TRUNCATE_LOCAL(17): server=server, peer=U32_MAX, term=snap_term,
//   index_or_max=upto, count_or_zero=deleted, aux=0.
//   counts.snapshots_taken += 1. emit SNAPSHOT_TAKE(18): server=server,
//   peer=U32_MAX, term=snap_term, index_or_max=upto, count_or_zero=0,
//   aux=snapshot_state_hash[server].
//   ORDER: the TRUNCATE_LOCAL event (if any) is emitted BEFORE the
//   SNAPSHOT_TAKE event. Then, if server == L (this server is the current
//   leader): lli=last_log_index(server); match_index[server]=lli;
//   next_index[server]=lli+1.
//
// 7.8 INSTALL_SNAPSHOT(follower):
//   INVALID if !has_leader || follower<0 || follower>=S || follower==L.
//   If snapshot_index[L] <= snapshot_index[follower]:
//     counts.snapshot_noop += 1. emit SNAPSHOT_INSTALL_NOOP(20): server=L,
//     peer=follower, term=snapshot_term[L], index_or_max=snapshot_index[L],
//     count_or_zero=0, aux=0. return.
//   Retention decision: be = log_at(follower, snapshot_index[L]);
//     retain = (be present && be.term == snapshot_term[L]).
//   If retain: keep follower entries with index > snapshot_index[L];
//     deleted = number with index <= snapshot_index[L].
//   Else: deleted = follower log size; clear follower log entirely.
//   Set snapshot_index[follower]=snapshot_index[L];
//     snapshot_term[follower]=snapshot_term[L];
//     snapshot_state_hash[follower]=snapshot_state_hash[L].
//   If commit_index[follower] < snapshot_index[follower]:
//     commit_index[follower]=snapshot_index[follower].
//   If last_applied[follower] < snapshot_index[follower]:
//     last_applied[follower]=snapshot_index[follower].
//   apply_accumulator[follower] = snapshot_state_hash[follower].
//   match_index[follower]=snapshot_index[L]; next_index[follower]=
//     snapshot_index[L]+1.
//   counts.snapshots_installed += 1. emit SNAPSHOT_INSTALL(19): server=L,
//     peer=follower, term=snapshot_term[follower],
//     index_or_max=snapshot_index[follower], count_or_zero=0,
//     aux=snapshot_state_hash[follower].
//   ORDER: SNAPSHOT_INSTALL is emitted FIRST; then if deleted > 0:
//     counts.snapshot_truncations += 1. emit SNAPSHOT_TRUNCATE_REMOTE(21):
//     server=L, peer=follower, term=snapshot_term[follower],
//     index_or_max=snapshot_index[follower], count_or_zero=deleted, aux=0.
//
// ----------------------------------------------------------------------------
// 8. OVERFLOW / CAPACITY / WRAP
// ----------------------------------------------------------------------------
//   All term/index/command/seq arithmetic is unsigned 64-bit with natural wrap
//   (no saturation); test inputs stay well within range so no wrap occurs in
//   practice. Capacity limits that change behavior: leader log full
//   (CLIENT_APPEND -> client_rejected), pending table full (SEND_APPEND ->
//   append_send_rejected), follower would exceed max_log_entries_per_server
//   (DELIVER_APPEND success path -> append_follower_oom, no mutation),
//   per-append copy capped at min(max_entries, max_entries_per_append),
//   per-apply capped at min(limit, max_apply_per_op). next_index decrements
//   floor at 1 (term-reject path). There is no modulo/ring arithmetic anywhere.
// === END DETERMINISM & EXACT OUTPUT SERIALIZATION ===

#endif  // RAFT_LOG_SNAPSHOT_COMMON_H_
