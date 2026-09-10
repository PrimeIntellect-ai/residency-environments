// file: lock_manager_deadlock_common.h

#ifndef LOCK_MANAGER_DEADLOCK_COMMON_H_
#define LOCK_MANAGER_DEADLOCK_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define LMD_ABI_VERSION 1

// ---------------------------------------------------------------- limits
#define LMD_MIN_TABLES 1
#define LMD_MAX_TABLES 16
#define LMD_MIN_PARTS 1
#define LMD_MAX_PARTS 16
#define LMD_MIN_ROWS 1
#define LMD_MAX_ROWS 64
#define LMD_MIN_TXNS 1
#define LMD_MAX_TXNS 64
#define LMD_MIN_LOCKS 1
#define LMD_MAX_LOCKS 4096
#define LMD_MIN_WAITERS 1
#define LMD_MAX_WAITERS 1024
#define LMD_MAX_STEPS 4096
#define LMD_MAX_DEADLOCK_CYCLES 64

// ---------------------------------------------------------------- lock modes
enum LmdMode {
    LMD_NL = 0,
    LMD_IS = 1,
    LMD_IX = 2,
    LMD_S = 3,
    LMD_SIX = 4,
    LMD_X = 5,
    LMD_MODE_COUNT = 6
};

// ---------------------------------------------------------------- resource kinds
// Encoded so that ROW < PARTITION < TABLE for the RELEASE cascade ordering.
enum LmdResKind {
    LMD_ROW = 0,
    LMD_PARTITION = 1,
    LMD_TABLE = 2,
    LMD_RES_KIND_COUNT = 3
};

// ---------------------------------------------------------------- operations
enum LmdOpKind {
    LMD_OP_BEGIN = 0,
    LMD_OP_LOCK = 1,
    LMD_OP_UNLOCK = 2,
    LMD_OP_UNLOCK_ALL = 3,
    LMD_OP_CONVERT = 4,
    LMD_OP_DETECT_DEADLOCK = 5,
    LMD_OP_KIND_COUNT = 6
};

// ---------------------------------------------------------------- transaction status
enum LmdTxnStatus {
    LMD_ACTIVE = 0,
    LMD_WAITING = 1
};

// ---------------------------------------------------------------- event kinds
// This ordinal is the uint8 `kind` folded into lock_event_hash; see the
// CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION section below.
enum LmdEventKind {
    LMD_EV_TXN_BEGIN = 0,
    LMD_EV_LOCK_GRANT = 1,
    LMD_EV_LOCK_CONVERT = 2,
    LMD_EV_LOCK_REENTER = 3,
    LMD_EV_LOCK_WAIT = 4,
    LMD_EV_CONVERT_WAIT = 5,
    LMD_EV_CONVERT_NOOP = 6,
    LMD_EV_ESCALATE_GRANT = 7,
    LMD_EV_ESCALATE_BLOCKED = 8,
    LMD_EV_ESCALATE_RELEASE = 9,
    LMD_EV_LOCK_RELEASE = 10,
    LMD_EV_LOCK_RELEASE_ALL = 11,
    LMD_EV_WAKE_GRANT = 12,
    LMD_EV_WAKE_REBLOCK = 13,
    LMD_EV_DEADLOCK_NONE = 14,
    LMD_EV_DEADLOCK_ABORT = 15,
    LMD_EV_VICTIM_RELEASE = 16,
    LMD_EV_INVALID = 17,
    LMD_EV_KIND_COUNT = 18
};

// ---------------------------------------------------------------- count indices
enum LmdCountIdx {
    LMD_C_TXN_BEGUN = 0,
    LMD_C_LOCK_GRANTS = 1,
    LMD_C_LOCK_CONVERTS = 2,
    LMD_C_LOCK_REENTERS = 3,
    LMD_C_LOCK_WAITS = 4,
    LMD_C_CONVERT_WAITS = 5,
    LMD_C_CONVERT_NOOPS = 6,
    LMD_C_ESCALATE_GRANTS = 7,
    LMD_C_ESCALATE_BLOCKED = 8,
    LMD_C_ESCALATE_RELEASES = 9,
    LMD_C_LOCK_RELEASES = 10,
    LMD_C_LOCK_RELEASE_ALL = 11,
    LMD_C_WAKE_GRANTS = 12,
    LMD_C_WAKE_REBLOCKS = 13,
    LMD_C_DEADLOCK_NONE = 14,
    LMD_C_DEADLOCK_ABORTS = 15,
    LMD_C_VICTIM_RELEASES = 16,
    LMD_C_INVALID = 17,
    LMD_COUNT_N = 18
};

/*
CONTRACT: lock_manager_deadlock  (T55)

A persistent, deterministic multi-granularity database lock manager simulated on
the device. Each solution_run applies exactly ONE operation (encoded in
LmdRunSpec) to persistent state and emits a fully-determined event stream.

Resources are TABLE(t), PARTITION(t,p), ROW(t,p,r) drawn from a fixed
table/partition/row hierarchy. Lock modes NL,IS,IX,S,SIX,X follow the classic
multiple-granularity locking protocol with the compatibility matrix:
  NL  : compatible with all
  IS  : compatible with IS, IX, S, SIX
  IX  : compatible with IS, IX
  S   : compatible with IS, S
  SIX : compatible with IS
  X   : compatible with none except NL
Dominance partial order: X dominates all; SIX dominates S,IX,IS; S dominates IS;
IX dominates IS; identical dominates itself.

Operations (LmdRunSpec.op_kind):
  BEGIN(txn_id, priority)
  LOCK(txn_id, resource, mode)
  UNLOCK(txn_id, resource)
  UNLOCK_ALL(txn_id)
  CONVERT(txn_id, resource, new_mode)
  DETECT_DEADLOCK(limit)
See the precise contract narrative for full per-op semantics: top-down
acquisition plans with intention ancestors, conversions via the dominance
lattice least-upper-bound, automatic escalation on fine-grained thresholds,
strict-FIFO wait queues, deterministic wait-for cycle detection with victim
selection (lowest priority; tie largest locks_acquired; tie largest txn_seq;
tie largest txn_id), and wake cascades that can grant then immediately reblock a
woken waiter on a later plan entry inside the same operation.

Outputs after EVERY op (exact integers, no float, no tolerance):
  counts[LMD_COUNT_N]   : monotonically accumulated counters.
  op_index_out          : index of this op.
  event_seq_out         : next event_seq (u64, wraps mod 2^64).
  lock_event_hash       : persistent running FNV-1a-64 over the event stream.
  grant_hash, wait_hash, txn_lock_hash : snapshot FNV hashes.
  state_checksum        : master FNV combining all of the above + counts + the
                          sequence-next counters.

Rules:
  - solution_init may allocate persistent state.
  - solution_run may NOT call cudaMalloc/cudaFree.
  - All graded outputs are exact integers.
*/

// === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section is self-contained and authoritative. It inlines every rule
// required to reproduce the six graded integers/hashes bit-for-bit. It
// supersedes any earlier prose that deferred to "section 4", "the contract
// narrative", or "the oracle". Where this section and the prose above differ,
// THIS section governs. Nothing here uses floating point; every output is an
// exact integer compared for equality.
//
// ----------------------------------------------------------------------------
// 0. FNV-1a-64 primitive (used by ALL six outputs)
// ----------------------------------------------------------------------------
//   offset basis = 0x14650FB0739D0383 = 1469598103934665603ULL
//   prime        = 0x00000100000001B3 = 1099511628211ULL
//   byte step:  h ^= (uint64_t)b;  h *= prime;
//   Multi-byte fields are hashed as their RAW LITTLE-ENDIAN bytes, in
//   increasing address order (host x86-64 / device sm are both little-endian),
//   i.e. exactly memcpy(&v,...) then feed bytes [0..width-1]. Widths:
//     u8  -> 1 byte,  u32 -> 4 bytes (LE),  u64 -> 8 bytes (LE).
//   No separators, no length prefixes, no trailing padding between fields.
//   "Emit a resource block R(kind,t,p,r)" below is shorthand for the 4-field
//   sequence: u8(kind), u32(t), u32(p), u32(r) with the -1 -> U32MAX rule
//   stated at each call site (a field whose stored value is < 0 is emitted as
//   U32MAX = 0xFFFFFFFF; a "missing" resource emits kind=255 and t=p=r=U32MAX).
//   U32MAX = 0xFFFFFFFFu, U64MAX = ~0ULL = 0xFFFFFFFFFFFFFFFF.
//
// ----------------------------------------------------------------------------
// 1. PERSISTENT SEQUENCE COUNTERS (state, survive across ops; reset values)
// ----------------------------------------------------------------------------
//   event_seq        : starts 0. Incremented by +1 after EACH emitted event.
//   txn_seq_next     : starts 1. BEGIN assigns txn_seq = txn_seq_next then ++.
//   request_seq_next : starts 1. Each LOCK consumes one (post-increment, used
//                      as the request_seq stamped on any waiters that LOCK
//                      enqueues); each CONVERT that blocks consumes one.
//   wait_seq_next    : starts 1. Each enqueued/reblocked waiter takes wait_seq
//                      = wait_seq_next then ++ (global, monotonic; this is the
//                      strict-FIFO key per resource).
//   deadlock_seq_next: starts 1. Incremented by +1 on each DEADLOCK_ABORT.
//   op_index         : starts 0. Set to the value BEFORE this op as the
//                      reported op_index_out, then incremented by +1. The value
//                      hashed into state_checksum / events is described below.
//   event_hash       : starts at the FNV offset basis. PERSISTENT running hash
//                      mutated by every emit() (see section 3). Reported as
//                      lock_event_hash. NOT reseeded per op.
//   All sequence wraps are mod 2^64 (u64) / mod 2^32 (op_index as u32).
//
// ----------------------------------------------------------------------------
// 2. PER-OP SEMANTICS (exact; determine WHICH events emit and state mutation)
// ----------------------------------------------------------------------------
// Mode ids: NL=0,IS=1,IX=2,S=3,SIX=4,X=5. Resource kinds: ROW=0,PART=1,TABLE=2.
// compat[req][held] (cross-txn coexistence), dom[hold][req] (held covers req),
// lub[a][b] (least upper bound) are the 6x6 tables:
//   compat: NL:all; IS:{NL,IS,IX,S,SIX}; IX:{NL,IS,IX}; S:{NL,IS,S};
//           SIX:{NL,IS}; X:{NL}.   (compat[a][b]==1 iff a,b coexist)
//   dom:    NL>={NL}; IS>={NL,IS}; IX>={NL,IS,IX}; S>={NL,IS,S};
//           SIX>={NL,IS,IX,S,SIX}; X>={all}.
//   lub:    {NL,IS,IX,S,SIX,X}/{IS,IS,IX,S,SIX,X}/{IX,IX,IX,SIX,SIX,X}/
//           {S,S,SIX,S,SIX,X}/{SIX,SIX,SIX,SIX,SIX,X}/{X,...,X} rows for
//           a=NL,IS,IX,S,SIX,X (matches d_lub in the reference).
//
// Resource canonical order (for grant_hash, wait_hash, wake set, escalate
// release, grant lookup tie-break): kind rank TABLE=0 < PARTITION=1 < ROW=2,
// then t asc, then p asc, then r asc.
// Resource RELEASE order (UNLOCK_ALL, victim abort cascade): kind rank ROW=0 <
// PARTITION=1 < TABLE=2, then t,p,r asc.
//
// Acquisition plan build_plan(target,mode) (top-down intention ancestors):
//   target TABLE(t):     [ (TABLE t, mode, is_target=1) ]
//   target PARTITION(t,p): anc=(mode==S||mode==IS)?IS:IX;
//       [ (TABLE t, anc, 0), (PART t,p, mode, 1) ]
//   target ROW(t,p,r):   tanc=panc=(mode==S)?IS:IX;
//       [ (TABLE t, tanc,0),(PART t,p, panc,0),(ROW t,p,r, mode,1) ]
//
// Grants: keyed by (resource, txn). Fields: mode, explicit_count, auto_count,
// grant_seq, last_convert_seq. A grant "exists" iff (explicit||auto)!=0.
// Each txn has locks_acquired (bumped +1 on every successful grant OR convert
// OR reenter-skip? NO: reenter does NOT bump locks_acquired; only grant/convert
// paths via do_grant_or_convert, escalate grant/convert, and CONVERT success
// bump locks_acquired +1). deadlock_aborts stays 0 (never mutated; the victim
// is erased, not stamped) -> always emitted as 0 in txn_lock_hash.
//
// Per-(txn,table) descendant counters {row_s,row_x,part_s,part_x,esc_attempts}
// are RECOMPUTED (not incrementally tracked) from the txn's EXPLICITLY held
// (explicit_count!=0) non-TABLE grants whenever a ROW/PARTITION target grant,
// convert, or release changes them: for each such grant classify by current
// mode: s_like=(mode in {S,SIX}), x_like=(mode in {X,IX,SIX}); ROW grant adds
// to row_s/row_x, PARTITION grant to part_s/part_x. esc_attempts is preserved
// across recompute (only try_escalate bumps it). recompute zeroes the four
// count fields first. fine_count(t)=row_s+row_x+part_s+part_x.
// holds_x_like(t) = (row_x+part_x)>0.
//
// BEGIN(txn,priority): invalid (-> INVALID event, count INVALID) if txn already
//   exists or live txn count==max_txns. Else create slot: txn_seq=txn_seq_next++
//   priority=priority, status=ACTIVE, not blocked, locks=0, aborts=0, all table
//   counters 0. count TXN_BEGUN. emit TXN_BEGIN with res=missing, mode=255,
//   aux=priority.
//
// LOCK(txn,res,mode): invalid if txn missing OR txn.status!=ACTIVE OR resource
//   out of range for its kind OR mode not in {IS,IX,S,SIX,X}. On valid: consume
//   one request_seq (this_request_seq). Walk plan entries in order; for each
//   entry (e.res,e.mode,is_target):
//     - if grant exists AND dom[grant.mode][e.mode]: REENTER. bump
//       explicit(if target) else auto by +1. count LOCK_REENTERS. emit
//       LOCK_REENTER(e.res,e.mode). continue. (no locks_acquired bump.)
//     - else eff = grant exists ? lub[grant.mode][e.mode] : e.mode. If
//       compatible_with_others(e.res, txn, eff): do_grant_or_convert(e.res,
//       e.mode, is_target) [grants new with mode=e.mode OR converts existing to
//       lub[old][e.mode]; emits LOCK_GRANT(mode=e.mode) for new, LOCK_CONVERT
//       (mode=lub) for existing; locks_acquired+=1; if target recompute]. Then
//       if is_target and e.res.kind in {ROW,PARTITION} and
//       fine_count(table)>=escalation_threshold: try_escalate(table).
//     - else (blocked): enqueue waiter at e.res (reqmode=e.mode, plan_index=pi,
//       request_seq=this_request_seq, is_conversion=0, original_target=res,
//       original_target_mode=mode, wait_seq=wait_seq_next++). Set txn WAITING,
//       blocked_resource=e.res, blocked_mode=e.mode. count LOCK_WAITS. emit
//       LOCK_WAIT(e.res,e.mode). RETURN (earlier ancestor grants stay held).
//
// UNLOCK(txn,res): invalid if txn missing OR no existing grant at (res,txn) OR
//   grant.explicit_count==0. Else explicit_count-=1. If now 0:
//     - if auto_count==0: mode=NL, grant removed; if res ROW/PART recompute.
//     - else: if res ROW/PART recompute; mode = intent_for_descendants =
//             holds_x_like(res.table) ? IX : IS (downgrade to needed intent).
//   If still >0: if res ROW/PART recompute. Always: count LOCK_RELEASES, emit
//   LOCK_RELEASE(res,mode=255). Then wake_resources([res]).
//
// UNLOCK_ALL(txn): invalid if txn missing. Else collect all existing grants of
//   txn, sort in RELEASE order; for each: explicit=auto=0, mode=NL (removed);
//   count LOCK_RELEASE_ALL; emit LOCK_RELEASE_ALL(res,mode=255). Then erase the
//   transaction. Then wake_resources(released set).
//
// CONVERT(txn,res,newmode): invalid if txn missing OR status!=ACTIVE OR
//   resource invalid OR newmode not a request mode OR no existing grant. If
//   dom[grant.mode][newmode]: count CONVERT_NOOPS, emit CONVERT_NOOP(res,
//   newmode); return. Else eff=lub[grant.mode][newmode]. If
//   compatible_with_others(res,txn,eff): grant.mode=eff; last_convert_seq=
//   event_seq(current, before emit); if ROW/PART recompute; locks_acquired+=1;
//   count LOCK_CONVERTS; emit LOCK_CONVERT(res,eff). Else: consume a
//   request_seq; enqueue waiter (reqmode=newmode, plan_index=0,
//   is_conversion=1, original_target=res, original_target_mode=newmode,
//   wait_seq++); set WAITING/blocked; count CONVERT_WAITS; emit CONVERT_WAIT
//   (res,newmode).
//
// Escalation try_escalate(txn,t): esc_attempts(t)+=1. target=holds_x_like(t)?
//   X:S. If NOT compatible_with_others(TABLE t, txn, target): count
//   ESCALATE_BLOCKED, emit ESCALATE_BLOCKED(TABLE t,target); return. Else grant
//   or convert the TABLE t lock to target as an EXPLICIT hold (new: mode=target
//   explicit=1 auto=0; existing: mode=lub[old][target] explicit+=1);
//   locks_acquired+=1; emit ESCALATE_GRANT(TABLE t, mode = target if new else
//   lub). Then release EVERY non-TABLE grant of txn under table t in CANONICAL
//   order: for each, remember had_explicit=(explicit!=0); set explicit=auto=0,
//   mode=NL (removed); if had_explicit: count ESCALATE_RELEASES, emit
//   ESCALATE_RELEASE(res,mode=255) (pure-auto intents drop silently). Collect
//   all released resources, then recompute counters, append TABLE t to the
//   affected set, and wake_resources(affected).
//
// Wake model wake_resources(set): copy initial set, SORT canonical, DEDUP
//   adjacent. Process by growing index ri=0..end; wake_one may APPEND newly
//   granted resources at the tail WITHOUT re-sort/re-dedup; stop when ri hits
//   the (grown) end. wake_one(res): loop over the resource's wait queue head
//   (lowest wait_seq among live waiters on res):
//     - if head's txn no longer exists: drop (set waiter dead) and continue.
//     - eff = head.reqmode, or lub[grant.mode][reqmode] if head already holds a
//       grant on res. If NOT compatible_with_others(res,htxn,eff): STOP this
//       resource (head blocks the queue; strict FIFO).
//     - else dequeue head; is_target = is_conversion ? true :
//       (plan_index+1 == plan.size of build_plan(original_target,otmode)).
//       grant_woken (same grant/convert logic as do_grant_or_convert but emits
//       NO grant/convert event of its own); count WAKE_GRANTS; emit WAKE_GRANT
//       (res, mode=head.reqmode). Clear blocked/ set ACTIVE. If is_target and
//       res ROW/PART and fine_count>=threshold: try_escalate. If NOT conversion
//       and plan_index+1<plan.size: continue_plan from plan_index+1 (re-walks
//       remaining plan entries; each granted entry is appended to the affected
//       set; REENTER/grant/convert events emit as in LOCK; if an entry blocks,
//       enqueue with SAME request_seq + NEW wait_seq, set WAITING, count
//       WAKE_REBLOCKS, emit WAKE_REBLOCK(e.res,e.mode), stop).
//
// DETECT_DEADLOCK(limit): limit==0 is a valid no-op (no event). Else repeat =
//   min(limit, max_deadlock_cycles_per_detect) iterations:
//     find_first_cycle(): waiting txns (status WAITING && has_blocked) in
//       ASCENDING txn_id; DFS from each; successors(u) = blocker txns holding an
//       incompatible (compat[u.blocked_mode][held]==0) grant on u's
//       blocked_resource, with txn != u, sorted ASC by txn_id and deduped;
//       child order ascending; first back-edge onto the current DFS path yields
//       the cycle = path suffix from the matched node to top.
//     If no cycle: count DEADLOCK_NONE, emit DEADLOCK_NONE(res missing,mode255),
//       and STOP the whole detect (return).
//     Else pick victim among cycle members: LOWEST priority; tie -> LARGEST
//       locks_acquired; tie -> LARGEST txn_seq; tie -> LARGEST txn_id.
//     abort_victim: if victim WAITING, remove its matching waiter (txn &&
//       wait_seq) from its blocked resource queue. Release all its grants in
//       RELEASE order: explicit=auto=0,mode=NL; count VICTIM_RELEASES; emit
//       VICTIM_RELEASE(res,mode=255). Erase txn. count DEADLOCK_ABORTS. emit
//       DEADLOCK_ABORT(res missing,mode255). deadlock_seq_next+=1. Then
//       wake_resources(released set).
//
// INVALID path: count INVALID +=1; emit INVALID with txn=0, res missing
//   (kind=255,t=p=r=U32MAX), mode=255, aux=0.
//
// compatible_with_others(res,self,req): true iff for every OTHER txn's existing
//   grant on the EXACT same resource, compat[req][grant.mode]==1.
//
// ----------------------------------------------------------------------------
// 3. EVENT STREAM HASH  (output: lock_event_hash) -- PERSISTENT running FNV
// ----------------------------------------------------------------------------
// event_hash is seeded ONCE (reset) to the FNV offset basis and then mutated by
// every emit() in the exact emission order produced by the semantics above.
// Each emit(kind, txn_or_0, res_or_none, mode_or_neg, aux) folds, IN THIS ORDER
// into the running event_hash (NO reseed):
//   u8 (kind)                              // LmdEventKind ordinal
//   u64(seq)                               // event_seq value BEFORE increment
//   u32((uint32_t)op_index)                // current op_index (pre-increment)
//   u64(txn_or_0)                          // txn id, or 0 for none
//   resource block:
//       if res present: u8(res.kind), u32(res.t<0?U32MAX:res.t),
//                       u32(res.p<0?U32MAX:res.p), u32(res.r<0?U32MAX:res.r)
//       else:           u8(255), u32(U32MAX), u32(U32MAX), u32(U32MAX)
//   u8 (mode<0 ? 255 : mode)               // requested/effective mode or 255
//   u64(aux)                               // BEGIN: priority; otherwise 0
// Then event_seq += 1. lock_event_hash output = this running hash after the op.
//
// ----------------------------------------------------------------------------
// 4. grant_hash  (snapshot; FRESH FNV seeded at offset basis each op)
// ----------------------------------------------------------------------------
// Iterate ALL grants that exist (explicit||auto != 0) in CANONICAL resource
// order, tie-broken by txn id ASCENDING. For each, fold in order:
//   u8 (res.kind)            // TABLE=2/PART=1/ROW=0
//   u32(res.t)               // table id; always >=0 (emitted raw, NOT mapped)
//   u32(res.p<0 ? U32MAX : res.p)
//   u32(res.r<0 ? U32MAX : res.r)
//   u64(txn)
//   u8 (mode)
//   u64(explicit_count)
//   u64(auto_count)
//   u64(grant_seq)           // event_seq at the moment the grant was created
//   u64(last_convert_seq)    // event_seq at last convert; 0 if never converted
//
// ----------------------------------------------------------------------------
// 5. wait_hash  (snapshot; FRESH FNV seeded at offset basis each op)
// ----------------------------------------------------------------------------
// Iterate distinct resources holding live waiters in CANONICAL resource order;
// within each resource, waiters in wait_seq ASCENDING (== queue order). A
// per-resource position counter `pos` starts at 0 and increments per waiter.
// For each waiter fold in order:
//   u8 (res.kind)
//   u32(res.t)               // always >=0, raw
//   u32(res.p<0 ? U32MAX : res.p)
//   u32(res.r<0 ? U32MAX : res.r)
//   u64(pos)                 // 0-based position within THIS resource's queue
//   u64(waiter.txn)
//   u8 (waiter.requested_mode)
//   u64(waiter.wait_seq)
//   u64(waiter.request_seq)
//   u8 (waiter.is_conversion)        // 0 or 1
//   u8 (original_target_res.kind)
//   u32(original_target_res.t)       // always >=0, raw
//   u32(original_target_res.p<0 ? U32MAX : .p)
//   u32(original_target_res.r<0 ? U32MAX : .r)
//   u8 (original_target_mode)
//
// ----------------------------------------------------------------------------
// 6. txn_lock_hash  (snapshot; FRESH FNV seeded at offset basis each op)
// ----------------------------------------------------------------------------
// Iterate ALL live transactions in txn_id ASCENDING. For each, fold in order:
//   u64(txn_id)
//   u64(txn_seq)
//   u32(priority)
//   u8 (status)              // ACTIVE=0 / WAITING=1
//   blocked block:
//       if has_blocked: u8(blocked_res.kind),
//                       u32(blocked_res.t),                 // raw, >=0
//                       u32(blocked_res.p<0?U32MAX:.p),
//                       u32(blocked_res.r<0?U32MAX:.r),
//                       u8(blocked_mode)
//       else:           u8(255), u32(U32MAX), u32(U32MAX), u32(U32MAX), u8(255)
//   u64(locks_acquired)
//   u64(deadlock_aborts)     // always 0 in practice (never stamped)
//   for ti = 0 .. table_count-1 (ASCENDING table index):
//       u32(ti)
//       u64(row_s_count)
//       u64(row_x_count)
//       u64(partition_s_count)
//       u64(partition_x_count)
//       u64(escalation_attempts)
//
// ----------------------------------------------------------------------------
// 7. state_checksum  (master; FRESH FNV seeded at offset basis each op)
// ----------------------------------------------------------------------------
// Fold the following IN THIS EXACT ORDER (gh/wh/th are the section 4/5/6 hashes
// computed for THIS op; op_index here is the value AFTER the post-increment,
// i.e. reported op_index_out + 1):
//   u64(event_seq)           // value after this op
//   u64(txn_seq_next)
//   u64(request_seq_next)
//   u64(wait_seq_next)
//   u64(deadlock_seq_next)
//   u32((uint32_t)op_index)  // POST-increment op_index (= op_index_out + 1)
//   u64(event_hash)          // == lock_event_hash output
//   u64(grant_hash)          // == section 4
//   u64(wait_hash)           // == section 5
//   u64(txn_lock_hash)       // == section 6
//   for i = 0 .. LMD_COUNT_N-1: u64((uint64_t)counts[i])  // ascending index
//
// Scalar outputs: op_index_out = op_index BEFORE increment; event_seq_out =
// event_seq AFTER this op; counts[] are the monotonically accumulated counters.
// === END CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION ===

// ---------------------------------------------------------------- structs
struct alignas(8) LmdProblemSpec {
    int32_t abi_version;
    int32_t table_count;
    int32_t partitions_per_table;
    int32_t rows_per_partition;
    int32_t max_txns;
    int32_t max_locks;
    int32_t max_waiters;
    int32_t escalation_threshold;
    int32_t max_deadlock_cycles_per_detect;
    int32_t flags;
    int32_t reserved[6];
};

struct alignas(8) LmdRunSpec {
    int32_t abi_version;
    int32_t op_kind;            // LmdOpKind
    int32_t step_id;
    int32_t a_res_kind;         // LmdResKind (LOCK/UNLOCK/CONVERT)
    int32_t a_table;            // resource table id
    int32_t a_partition;        // resource partition id (or -1)
    int32_t a_row;              // resource row id (or -1)
    int32_t a_mode;             // LmdMode (LOCK/CONVERT)
    int32_t a_priority;         // BEGIN priority
    int32_t a_limit;            // DETECT_DEADLOCK limit
    uint64_t a_txn;             // transaction id
    uint64_t reserved2;
};

struct alignas(8) LmdInputs {
    // All operands travel through LmdRunSpec; reserved for ABI symmetry.
    const void* reserved;
};

struct alignas(8) LmdOutputs {
    int64_t* counts;            // [LMD_COUNT_N]
    int32_t* op_index_out;      // [1]
    uint64_t* event_seq_out;    // [1]
    uint64_t* lock_event_hash;  // [1]
    uint64_t* grant_hash;       // [1]
    uint64_t* wait_hash;        // [1]
    uint64_t* txn_lock_hash;    // [1]
    uint64_t* state_checksum;   // [1]
};

static_assert(sizeof(LmdProblemSpec) == 64, "LmdProblemSpec layout drift");
static_assert(sizeof(LmdRunSpec) == 56, "LmdRunSpec layout drift");
static_assert(sizeof(LmdOutputs) == 64, "LmdOutputs layout drift");

static inline int lmd_validate_problem_spec(const LmdProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != LMD_ABI_VERSION) return 0;
    if (spec->table_count < LMD_MIN_TABLES || spec->table_count > LMD_MAX_TABLES) return 0;
    if (spec->partitions_per_table < LMD_MIN_PARTS || spec->partitions_per_table > LMD_MAX_PARTS) return 0;
    if (spec->rows_per_partition < LMD_MIN_ROWS || spec->rows_per_partition > LMD_MAX_ROWS) return 0;
    if (spec->max_txns < LMD_MIN_TXNS || spec->max_txns > LMD_MAX_TXNS) return 0;
    if (spec->max_locks < LMD_MIN_LOCKS || spec->max_locks > LMD_MAX_LOCKS) return 0;
    if (spec->max_waiters < LMD_MIN_WAITERS || spec->max_waiters > LMD_MAX_WAITERS) return 0;
    if (spec->escalation_threshold < 1) return 0;
    if (spec->max_deadlock_cycles_per_detect < 1 ||
        spec->max_deadlock_cycles_per_detect > LMD_MAX_DEADLOCK_CYCLES) return 0;
    return 1;
}

static inline int lmd_validate_run_spec(const LmdRunSpec* run, const LmdProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != LMD_ABI_VERSION) return 0;
    if (run->op_kind < 0 || run->op_kind >= LMD_OP_KIND_COUNT) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const LmdProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const LmdProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const LmdRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // LOCK_MANAGER_DEADLOCK_COMMON_H_
