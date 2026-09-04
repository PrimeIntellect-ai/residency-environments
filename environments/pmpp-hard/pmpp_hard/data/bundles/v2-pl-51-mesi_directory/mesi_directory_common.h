// file: mesi_directory_common.h

#ifndef MESI_DIRECTORY_COMMON_H_
#define MESI_DIRECTORY_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MESI_ABI_VERSION 1

/* Bounds. core_count is capped at 64 so a single u64 word can describe a
 * sharer / invalidation-target set. */
#define MESI_MIN_CORES 1
#define MESI_MAX_CORES 64
#define MESI_MIN_LINES 1
#define MESI_MAX_LINES 4096
#define MESI_MIN_CAP 1
#define MESI_MAX_CAP 256
#define MESI_MIN_PENDING 1
#define MESI_MAX_PENDING 4096
#define MESI_MAX_BATCH 4096
#define MESI_MAX_STEPS 64

/* Operation kinds (input op stream). */
#define MESI_OP_LOAD 0
#define MESI_OP_STORE 1
#define MESI_OP_ACK_INV 2
#define MESI_OP_EVICT 3
#define MESI_OP_FLUSH 4

/* Cache line MESI states. Invalid lines are simply absent from the cache. */
#define MESI_STATE_M 0
#define MESI_STATE_E 1
#define MESI_STATE_S 2
#define MESI_STATE_NONE 255

/* Event kinds, in the exact order of the contract enumeration. */
#define EV_LOAD_HIT 0
#define EV_LOAD_STALL_PENDING 1
#define EV_CAPACITY_EVICT 2
#define EV_DOWNGRADE_WRITEBACK 3
#define EV_DOWNGRADE_CLEAN 4
#define EV_LOAD_MISS_SHARED 5
#define EV_LOAD_MISS_EXCLUSIVE 6
#define EV_STORE_HIT_MODIFIED 7
#define EV_STORE_HIT_EXCLUSIVE 8
#define EV_STORE_STALL_PENDING 9
#define EV_INV_SEND 10
#define EV_STORE_PENDING 11
#define EV_DATA_SUPPLY_DIRTY 12
#define EV_DATA_SUPPLY_CLEAN 13
#define EV_INV_ACK 14
#define EV_STORE_COMMIT 15
#define EV_EVICT_STALL_PENDING 16
#define EV_EVICT_MISS 17
#define EV_EVICT_WRITEBACK 18
#define EV_EVICT_CLEAN 19
#define EV_EVICT_SHARED 20
#define EV_FLUSH_WRITEBACK 21
#define EV_FLUSH_CLEAN 22
#define EV_FLUSH_NOOP 23
#define EV_INVALID 24

/*
CONTRACT: mesi_directory

Persistent directory-coherence simulator (MESI) with pending store
transactions, invalidation acknowledgments, dirty writeback ordering, and
deterministic capacity eviction. All graded outputs are exact integers.

PERSISTENT STATE
  Scalars (wrap modulo 2^64):
    event_seq      starts 0 ; the seq stamped onto the NEXT emitted event.
    touch_seq_next starts 1 ; the touch stamped onto the NEXT cache touch.
    txn_seq_next   starts 1 ; the id of the NEXT created pending transaction.

  memory_value[line] : i64.

  Per-core cache: up to cache_capacity_per_core entries. Each entry is
    { line, state in {M,E,S}, value:i64, touch_seq:u64 }. Invalid lines are
    absent.

  Directory per line:
    owner_core in {none, core} (owner is in M or E).
    sharer_set : set of cores in S.
    Quiescent invariant (no pending txn): owner XOR nonempty-sharers, never
    both.

  Pending transaction per line (at most one):
    { txn_id, requester, line, new_value, target_set, supplier_value, start_seq }
    supplier_value defaults to INT64_MIN.

OPERATION STREAM (per step, processed in input order i = 0..batch_size-1)
  op[i] in {LOAD,STORE,ACK_INV,EVICT,FLUSH}; arg_core, arg_line, arg_value:i64,
  arg_txn:u64. The full per-op semantics, the capacity-eviction victim rule, and
  the EXACT output serialization are all specified normatively below (see the
  "CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION" section). The reference and
  naive kernels implement the same contract independently.

EVENT STREAM
  Every emitted event consumes one event_seq (post-increment) and is folded
  into coh_event_hash in emission order. The hashed tuple is:
    event_kind:u8, event_seq:u64, op_index:u32, core_or_UINT32_MAX:u32,
    line:u64, state_or_255:u8, value_or_INT64_MIN:i64, txn_or_UINT64_MAX:u64,
    aux_u64.

OUTPUTS (after every step)
  counts[MESI_COUNT_FIELDS] : one i64 counter per event class (see indices).
  coh_event_hash[0]   : FNV-1a-64 over the step's event tuples.
  cache_hash[0]       : FNV-1a-64 over cores ascending, entries by line ascending
                        { core:u32, line:u64, state:u8, value:i64, touch_seq:u64 }.
  directory_hash[0]   : FNV-1a-64 over lines ascending
                        { line:u64, memory_value:i64, owner_or_UINT32_MAX:u32,
                          sharer_count:u64, sharers ascending:u32 }.
  pending_hash[0]     : FNV-1a-64 over pending lines ascending
                        { line:u64, txn_id:u64, requester:u32, new_value:i64,
                          supplier_value_or_INT64_MIN:i64, start_seq:u64,
                          targets ascending:u32 }.
  event_seq_out[0]    : event_seq after the step.
  state_checksum[0]   : FNV-1a-64 over the full persistent state (scalars +
                        memory + caches + directory + pending). EXACT field
                        emission order is specified normatively below.

RULES
  - solution_init may allocate persistent state.
  - solution_run may not call cudaMalloc/cudaFree; it may use the workspace.
  - solution_reset restores the initial persistent state.
*/

// === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section is self-contained: every rule needed to reproduce the graded
// integers and checksums bit-for-bit is stated here. It is derived directly
// from the reference kernel. "FNV" below ALWAYS means the 64-bit FNV-1a defined
// in (A); all multi-byte fields are hashed in NATIVE little-endian byte order
// (the bytes of the value as stored in memory), in the declared field order.
//
// ---------------------------------------------------------------------------
// (A) FNV-1a-64 PRIMITIVE (used by EVERY hash and checksum below)
//   offset basis / init / seed = 1469598103934665603  (0x14650FB0739D0383)
//   prime                      = 1099511628211         (0x100000001B3)
//   Per byte b:  h ^= (uint64_t)b;  h *= prime;
//   A field of width W is folded by hashing its W native-endian bytes in order
//   (low address first). Widths used: u8 (1B), i32/u32 (4B), i64/u64 (8B).
//   Signed values are hashed as their two's-complement byte pattern.
//   Every hash/checksum below STARTS from a fresh init = 0x14650FB0739D0383,
//   EXCEPT coh_event_hash, which is a PERSISTENT running hash (see (C)).
//
// ---------------------------------------------------------------------------
// (B) PERSISTENT STATE SEEDING (must match for state_checksum / directory_hash)
//   On init and on solution_reset:
//     memory_value[line] = (int64_t)(1000 + line*7)   for line in [0,L)
//     all caches empty (state = MESI_STATE_NONE), owner[line] = -1 (none),
//     sharer_set[line] = {} (empty), pending[line] inactive with
//       txn=0, requester=-1, new_value=0, supplier_value=INT64_MIN,
//       start_seq=0, target_set={}.
//     all counts[] = 0.
//     event_seq = 0 ; touch_seq_next = 1 ; txn_seq_next = 1 ;
//     running coh_event_hash = 0x14650FB0739D0383 (FNV init).
//   Scalars wrap modulo 2^64. Counters are cumulative across steps and reset
//   ONLY by solution_reset (never between steps within a run).
//
// ---------------------------------------------------------------------------
// (C) EVENT EMISSION + coh_event_hash (PERSISTENT running hash)
//   Each emitted event stamps the CURRENT event_seq, then post-increments
//   event_seq by 1 (mod 2^64). Each event folds the following 9 fields, IN
//   THIS ORDER, into the persistent running coh_event_hash:
//       event_kind : u8
//       event_seq  : u64   (the value stamped, BEFORE increment)
//       op_index   : u32   (the input index i of the op producing the event)
//       core       : u32   (or UINT32_MAX sentinel)
//       line       : u64   (or UINT64_MAX sentinel)
//       state      : u8    (MESI_STATE_*; or 255 sentinel)
//       value      : i64   (or INT64_MIN sentinel)
//       txn        : u64   (or UINT64_MAX sentinel)
//       aux        : u64   (event-specific; 0 when unused)
//   coh_event_hash[0] is emitted AFTER the step as the current running hash;
//   it is cumulative-by-construction across steps. event_seq_out[0] is the
//   persistent event_seq after the step.
//   Sentinels: NO_CORE=UINT32_MAX, NO_STATE=255, NO_VAL=INT64_MIN,
//              NO_TXN=UINT64_MAX, NO_LINE=UINT64_MAX.
//
//   Per-event field meanings (core/line/state/value/txn/aux; unlisted ->
//   sentinel):
//     LOAD_HIT            core, line, state=current cached state, value=current
//                         cached value.
//     LOAD_STALL_PENDING  core, line. (value=NO_VAL.)
//     CAPACITY_EVICT      core, line=victim line, state=victim state,
//                         value=victim cache value; aux = victim touch_seq.
//     DOWNGRADE_WRITEBACK core=supplier(prev owner), line, state=S,
//                         value=supplier value.
//     DOWNGRADE_CLEAN     core=supplier(prev owner), line, state=S,
//                         value=supplier value.
//     LOAD_MISS_SHARED    core=requester, line, state=S, value=installed value.
//     LOAD_MISS_EXCLUSIVE core=requester, line, state=E, value=installed value.
//     STORE_HIT_MODIFIED  core, line, state=M, value=new value.
//     STORE_HIT_EXCLUSIVE core, line, state=M, value=new value.
//     STORE_STALL_PENDING core, line, value=attempted new value (state=NO_STATE).
//     INV_SEND            core=target, line, state=MESI_STATE_NONE(255),
//                         value=requester new value, txn=txn id;
//                         aux = requester core.
//     STORE_PENDING       core=requester, line, state=MESI_STATE_NONE(255),
//                         value=new value, txn=txn id; aux = target count.
//     DATA_SUPPLY_DIRTY   core=target(supplier), line, state=M(old),
//                         value=dirty value, txn=txn id.
//     DATA_SUPPLY_CLEAN   core=target(supplier), line, state=E(old),
//                         value=clean value, txn=txn id.
//     INV_ACK             core=target, line, txn=txn id; aux = remaining target
//                         count (after removing this core). (value=NO_VAL.)
//     STORE_COMMIT        core=requester, line, state=M, value=new value,
//                         txn=txn id (the committed txn; NO_TXN for the
//                         no-target fast-commit path in STORE).
//     EVICT_STALL_PENDING core, line. (value=NO_VAL.)
//     EVICT_MISS          core, line. (value=NO_VAL.)
//     EVICT_WRITEBACK     core, line, state=M(old), value=written-back value.
//     EVICT_CLEAN         core, line, state=E(old). (value=NO_VAL.)
//     EVICT_SHARED        core, line, state=S(old). (value=NO_VAL.)
//     FLUSH_WRITEBACK     core=owner, line, state=S(new), value=written-back
//                         value.
//     FLUSH_CLEAN         core=owner, line, state=S(new). (value=NO_VAL.)
//     FLUSH_NOOP          core=NO_CORE, line. (value=NO_VAL.)
//     INVALID             op_index; core = supplied core if in range else
//                         NO_CORE; line = supplied line if in range else
//                         NO_LINE; for an invalid ACK, txn = supplied txn (else
//                         NO_TXN); value=NO_VAL.
//
// ---------------------------------------------------------------------------
// (D) CAPACITY-EVICTION VICTIM RULE (normative, derived from the reference)
//   Triggered when a core must install a line but its cache already holds
//   cache_capacity_per_core entries (a true miss, no existing slot for the
//   line). Scan the core's CAP slots in ascending slot order, skipping empty
//   slots. The VICTIM is the occupied slot minimizing the comparison key:
//       PRIMARY key:   touch_seq        direction = MINIMUM (LRU)
//       TIE-BREAK key: line index       direction = MINIMUM (smallest line)
//   i.e. choose min touch_seq; among equal touch_seq choose min line. (The
//   first slot scanned seeds the victim; subsequent slots replace it iff
//   touch < best_touch, OR (touch == best_touch AND line < best_line).)
//   touch_seq values are unique in practice, so the tie-break is a total order.
//   The victim's eviction effects mirror an EVICT op (see (E) EVICT), but the
//   emitted event kind is CAPACITY_EVICT (state=victim state, value=victim
//   value, aux=victim touch_seq) and counts[MC_capacity_evictions] += 1:
//     - victim M: memory_value[line] = victim value; if owner[line]==core ->
//                 owner[line] = none. (no MESI_STATE-specific evict counter.)
//     - victim E: if owner[line]==core -> owner[line] = none.
//     - victim S: remove core from sharer_set[line].
//   Then the victim slot is cleared (empty). CAPACITY_EVICT increments ONLY
//   MC_capacity_evictions (never the EVICT_* counters).
//
// ---------------------------------------------------------------------------
// (E) PER-OP SEMANTICS (processed in input order i = 0..batch_size-1)
//   Common: "find slot" scans the core's CAP slots ascending; an entry matches
//   iff state != NONE and line == target. "install" updates an existing slot
//   for that line if present (preserving the slot), else fills the first empty
//   slot; in BOTH cases it sets touch_seq = touch_seq_next++ (post-increment).
//
//   LOAD(core,line):
//     1. If core or line out of range -> INVALID (counts MC_invalid_count).
//     2. Else if pending[line].active -> LOAD_STALL_PENDING
//        (counts MC_load_stall_pending); no state change.
//     3. Else if core already caches line (hit) -> set its touch_seq =
//        touch_seq_next++, LOAD_HIT (counts MC_load_hit, state/value=current).
//     4. Else (miss): if count_used(core) >= CAP -> capacity_evict(core) (D).
//        Then resolve supply:
//        a. If owner[line] exists AND owner caches line in M:
//             memory_value[line] = owner value; owner entry M->S;
//             owner[line] = none; add {owner, core} to sharer_set;
//             install core as S with supplier value; emit DOWNGRADE_WRITEBACK
//             (core=owner, value=supplier) then LOAD_MISS_SHARED (core=core).
//             counts: MC_downgrade_writeback, MC_load_miss_shared.
//        b. Else if owner[line] exists AND owner caches line in E:
//             owner entry E->S; owner[line]=none; add {owner,core} to sharers;
//             install core as S with supplier value; emit DOWNGRADE_CLEAN then
//             LOAD_MISS_SHARED. counts: MC_downgrade_clean, MC_load_miss_shared.
//        c. Else if sharer_set[line] nonempty: install core as S with
//             memory_value[line]; add core to sharers; LOAD_MISS_SHARED
//             (counts MC_load_miss_shared).
//        d. Else (no owner, no sharers): install core as E with
//             memory_value[line]; owner[line] = core; LOAD_MISS_EXCLUSIVE
//             (counts MC_load_miss_exclusive).
//
//   STORE(core,line,value):
//     1. Out-of-range -> INVALID.
//     2. pending[line].active -> STORE_STALL_PENDING (counts
//        MC_store_stall_pending; value=attempted value).
//     3. core caches line in M -> value=value, touch_seq=touch_seq_next++,
//        STORE_HIT_MODIFIED (counts MC_store_hit_modified).
//     4. core caches line in E -> state E->M, value=value,
//        touch_seq=touch_seq_next++, owner[line]=core, STORE_HIT_EXCLUSIVE
//        (counts MC_store_hit_exclusive, state=M).
//     5. Else (core in S or absent): build target_set =
//          (sharer_set[line] minus {core}) UNION ({owner[line]} if owner exists
//          and owner != core).
//        a. If target_set empty (fast commit):
//             if core does not already cache line AND count_used(core) >= CAP ->
//             capacity_evict(core) (D); install core as M with value;
//             sharer_set[line] = {}; owner[line] = core; STORE_COMMIT
//             (counts MC_store_committed, state=M, value=value, txn=NO_TXN).
//        b. Else if active_pending_count() >= max_pending_lines ->
//             STORE_STALL_PENDING (counts MC_store_stall_pending; value=value).
//        c. Else open a pending txn on line:
//             txn_id = txn_seq_next++ ; requester=core ; new_value=value ;
//             supplier_value = INT64_MIN ; start_seq = CURRENT event_seq
//             (the event_seq value BEFORE any INV_SEND of this txn is emitted) ;
//             target_set as above. counts MC_store_pending += 1.
//             Then, for each target t in ASCENDING core order: counts
//             MC_inv_sent += 1 and emit INV_SEND (core=t, value=value,
//             txn=txn_id, aux=requester core). Finally emit STORE_PENDING
//             (core=requester, value=value, txn=txn_id,
//             aux = target count = popcount(target_set)).
//
//   ACK_INV(core,line,txn):
//     1. VALID iff core,line in range AND pending[line].active AND
//        pending[line].txn_id == txn AND core is in pending[line].target_set.
//        Otherwise INVALID (counts MC_invalid_count) with txn = supplied txn.
//     2. If core caches line:
//          - in M: pending.supplier_value = entry value;
//                  memory_value[line] = entry value; emit DATA_SUPPLY_DIRTY
//                  (core=core, state=M, value, txn); counts MC_data_supply_dirty.
//          - in E: pending.supplier_value = entry value; emit DATA_SUPPLY_CLEAN
//                  (core=core, state=E, value, txn); counts MC_data_supply_clean.
//          Then clear that cache entry (empty). (If absent, no supply event.)
//     3. Remove core from sharer_set[line]; if owner[line]==core ->
//        owner[line]=none.
//     4. Remove core from pending.target_set; counts MC_inv_acked += 1; emit
//        INV_ACK (core=core, txn, aux = remaining target count AFTER removal).
//     5. If pending.target_set now empty (commit): let req=requester,
//        nv=new_value, tid=txn_id. If req does not cache line AND
//        count_used(req) >= CAP -> capacity_evict(req) (D). install req as M
//        with nv; sharer_set[line] = {}; owner[line] = req; counts
//        MC_store_committed += 1; emit STORE_COMMIT (core=req, state=M,
//        value=nv, txn=tid). Then delete the pending txn (inactive, fields
//        reset: txn=0, requester=-1, new_value=0, supplier_value=INT64_MIN,
//        start_seq=0, target_set={}).
//
//   EVICT(core,line):
//     1. Out-of-range -> INVALID.
//     2. If pending[line].active AND core in pending.target_set ->
//        EVICT_STALL_PENDING (counts MC_evict_stall_pending). (Note: a pending
//        line whose target_set does NOT contain core is NOT stalled here.)
//     3. If core does not cache line -> EVICT_MISS (counts MC_evict_miss).
//     4. Else by cached state:
//          - M: memory_value[line] = value; if owner[line]==core ->
//               owner[line]=none; clear entry; EVICT_WRITEBACK (state=M,
//               value=written-back value); counts MC_evict_writeback.
//          - E: if owner[line]==core -> owner[line]=none; clear entry;
//               EVICT_CLEAN (state=E, value=NO_VAL); counts MC_evict_clean.
//          - S: remove core from sharer_set[line]; clear entry; EVICT_SHARED
//               (state=S, value=NO_VAL); counts MC_evict_shared.
//
//   FLUSH(line):  (no core argument)
//     1. If line out of range OR pending[line].active -> INVALID (core=NO_CORE).
//     2. Else if owner[line] exists AND owner caches line in M:
//          memory_value[line] = owner value; owner entry M->S; owner[line]=none;
//          add owner to sharer_set[line]; FLUSH_WRITEBACK (core=owner, state=S,
//          value=written-back value); counts MC_flush_writeback.
//     3. Else if owner[line] exists AND owner caches line in E:
//          owner entry E->S; owner[line]=none; add owner to sharer_set[line];
//          FLUSH_CLEAN (core=owner, state=S, value=NO_VAL); counts
//          MC_flush_clean.
//     4. Else FLUSH_NOOP (core=NO_CORE, line); counts MC_flush_noop.
//
//   UNKNOWN op kind: INVALID with core=NO_CORE, line=NO_LINE (counts
//   MC_invalid_count).
//
// ---------------------------------------------------------------------------
// (F) cache_hash  (FNV init, fresh per step)
//   Iterate cores c = 0..core_count-1 (ASCENDING). Within each core, emit its
//   occupied cache entries in ASCENDING line-index order (lines are unique
//   within a core's cache). For each entry, fold these 5 fields in order:
//       core      : u32  (the core index c)
//       line      : u64
//       state     : u8
//       value     : i64
//       touch_seq : u64
//   Empty cores / empty slots contribute nothing.
//
// ---------------------------------------------------------------------------
// (G) directory_hash  (FNV init, fresh per step)
//   Iterate lines l = 0..line_count-1 (ASCENDING). For EVERY line (even with no
//   owner and no sharers), fold:
//       line          : u64
//       memory_value  : i64
//       owner         : u32  (owner core, or UINT32_MAX if none)
//       sharer_count  : u64  (popcount of sharer_set)
//   then the sharer core ids in ASCENDING order, each as u32. (sharer_count is
//   hashed even when zero; with zero sharers no sharer ids follow.)
//
// ---------------------------------------------------------------------------
// (H) pending_hash  (FNV init, fresh per step)
//   Iterate lines l = 0..line_count-1 (ASCENDING); SKIP lines whose pending txn
//   is inactive. For each active pending line, fold:
//       line            : u64
//       txn_id          : u64
//       requester       : u32
//       new_value       : i64
//       supplier_value  : i64  (INT64_MIN if none supplied yet)
//       start_seq       : u64
//   then the remaining target core ids in ASCENDING order, each as u32.
//
// ---------------------------------------------------------------------------
// (I) state_checksum  (FNV init, fresh per step) -- EXACT field emission ORDER
//   Fold, IN THIS ORDER:
//       core_count               : i32
//       line_count               : i32
//       cache_capacity_per_core  : i32
//       max_pending_lines        : i32
//       event_seq                : u64  (post-step persistent event_seq)
//       touch_seq_next           : u64
//       txn_seq_next             : u64
//       cache_hash      result   : u64  (the (F) value for this step)
//       directory_hash  result   : u64  (the (G) value for this step)
//       pending_hash    result   : u64  (the (H) value for this step)
//   then counts[i] for i = 0..MESI_COUNT_FIELDS-1 in index order, each as i64.
//   (cache_hash/directory_hash/pending_hash here are the SAME values emitted to
//   their own outputs this step; coh_event_hash and event_seq_out are NOT part
//   of state_checksum.)
// === END CONTRACT ===

/* Output count field indices. */
enum {
    MC_load_hit = 0,
    MC_load_miss_shared,
    MC_load_miss_exclusive,
    MC_load_stall_pending,
    MC_store_hit_modified,
    MC_store_hit_exclusive,
    MC_store_pending,
    MC_store_committed,
    MC_store_stall_pending,
    MC_inv_sent,
    MC_inv_acked,
    MC_data_supply_dirty,
    MC_data_supply_clean,
    MC_downgrade_writeback,
    MC_downgrade_clean,
    MC_evict_writeback,
    MC_evict_clean,
    MC_evict_shared,
    MC_evict_miss,
    MC_evict_stall_pending,
    MC_capacity_evictions,
    MC_flush_writeback,
    MC_flush_clean,
    MC_flush_noop,
    MC_invalid_count,
    MESI_COUNT_FIELDS
};

struct alignas(8) MesiProblemSpec {
    int32_t abi_version;
    int32_t core_count;
    int32_t line_count;
    int32_t cache_capacity_per_core;
    int32_t max_pending_lines;
    int32_t max_batch;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[8];
};

struct alignas(8) MesiRunSpec {
    int32_t abi_version;
    int32_t batch_size;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) MesiInputs {
    const int32_t* op;        // MESI_OP_*
    const int32_t* arg_core;  // core id (or out-of-range to force invalid)
    const int32_t* arg_line;  // line id (or out-of-range to force invalid)
    const int64_t* arg_value; // store value
    const uint64_t* arg_txn;  // ack txn id
};

struct alignas(8) MesiOutputs {
    int64_t* counts;            // MESI_COUNT_FIELDS entries
    uint64_t* coh_event_hash;   // 1
    uint64_t* cache_hash;       // 1
    uint64_t* directory_hash;   // 1
    uint64_t* pending_hash;     // 1
    uint64_t* event_seq_out;    // 1
    uint64_t* state_checksum;   // 1
};

static_assert(sizeof(MesiProblemSpec) == 64, "MesiProblemSpec layout drift");
static_assert(sizeof(MesiRunSpec) == 64, "MesiRunSpec layout drift");
static_assert(sizeof(MesiInputs) == 40, "MesiInputs layout drift");
static_assert(sizeof(MesiOutputs) == 56, "MesiOutputs layout drift");

static inline size_t mesi_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int mesi_validate_problem_spec(const MesiProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MESI_ABI_VERSION) return 0;
    if (spec->core_count < MESI_MIN_CORES || spec->core_count > MESI_MAX_CORES) return 0;
    if (spec->line_count < MESI_MIN_LINES || spec->line_count > MESI_MAX_LINES) return 0;
    if (spec->cache_capacity_per_core < MESI_MIN_CAP ||
        spec->cache_capacity_per_core > MESI_MAX_CAP) return 0;
    if (spec->max_pending_lines < MESI_MIN_PENDING ||
        spec->max_pending_lines > MESI_MAX_PENDING) return 0;
    if (spec->max_pending_lines > spec->line_count) return 0;
    if (spec->max_batch < 0 || spec->max_batch > MESI_MAX_BATCH) return 0;
    if (spec->max_steps < 1 || spec->max_steps > MESI_MAX_STEPS) return 0;
    return 1;
}

static inline int mesi_validate_run_spec(const MesiRunSpec* run, const MesiProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != MESI_ABI_VERSION) return 0;
    if (run->batch_size < 0 || run->batch_size > spec->max_batch) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MesiProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MesiProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MesiRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MESI_DIRECTORY_COMMON_H_
