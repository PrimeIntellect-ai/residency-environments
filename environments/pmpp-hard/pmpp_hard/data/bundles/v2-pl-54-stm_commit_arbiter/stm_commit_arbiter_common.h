// file: stm_commit_arbiter_common.h

#ifndef STM_COMMIT_ARBITER_COMMON_H_
#define STM_COMMIT_ARBITER_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define STM_ABI_VERSION 1

// Capacity bounds for the problem spec.
#define STM_MIN_TXNS 1
#define STM_MAX_TXNS 256
#define STM_MIN_LOCATIONS 1
#define STM_MAX_LOCATIONS 4096
#define STM_MIN_READ_SET 1
#define STM_MAX_READ_SET 256
#define STM_MIN_WRITE_SET 1
#define STM_MAX_WRITE_SET 256
#define STM_MIN_WATCH_SET 1
#define STM_MAX_WATCH_SET 256
#define STM_MIN_WAITERS 1
#define STM_MAX_WAITERS 256
#define STM_MIN_RETRY_WATCHERS 1
#define STM_MAX_RETRY_WATCHERS 256
#define STM_MAX_BATCH 4096
#define STM_MAX_STEPS 64
#define STM_MAX_WATCH_BUF 65536

// Op kinds (op_kind field of each operation row).
#define STM_OP_BEGIN         0
#define STM_OP_TX_READ       1
#define STM_OP_TX_WRITE      2
#define STM_OP_VALIDATE      3
#define STM_OP_TRY_PREPARE   4
#define STM_OP_DRAIN_COMMITS 5
#define STM_OP_RETRY         6
#define STM_OP_NON_TX_WRITE  7
#define STM_OP_ABORT         8

// Transaction status values.
#define STM_ST_ACTIVE          0
#define STM_ST_WAITING_LOCK     1
#define STM_ST_SUSPENDED_RETRY  2
#define STM_ST_PREPARED         3

// Event kinds for stm_event_hash.
#define STM_EVT_TXN_BEGIN             0
#define STM_EVT_READ_OWN_WRITE        1
#define STM_EVT_READ_SHARED           2
#define STM_EVT_WRITE_STAGE           3
#define STM_EVT_VALIDATE_OK           4
#define STM_EVT_WRITE_LOCK            5
#define STM_EVT_TXN_WAIT_LOCK         6
#define STM_EVT_WRITE_UNLOCK_PARTIAL  7
#define STM_EVT_TXN_PREPARED          8
#define STM_EVT_COMMIT_READONLY       9
#define STM_EVT_LOCATION_WRITE        10
#define STM_EVT_WRITE_UNLOCK_COMMIT   11
#define STM_EVT_TXN_WAKE_RETRY        12
#define STM_EVT_TXN_WAKE_LOCK         13
#define STM_EVT_COMMIT_DONE           14
#define STM_EVT_TXN_SUSPEND_RETRY     15
#define STM_EVT_RETRY_IMMEDIATE       16
#define STM_EVT_RETRY_WATCH_OVERFLOW  17
#define STM_EVT_NON_TX_WRITE          18
#define STM_EVT_NON_TX_STALL_LOCKED   19
#define STM_EVT_NON_TX_OOM            20
#define STM_EVT_WRITE_UNLOCK_ABORT    21
#define STM_EVT_TXN_ABORT             22
#define STM_EVT_INVALID               23

// Abort/event reason codes (reason_or_255 field).
#define STM_RS_READ_LOCK_CONFLICT     0
#define STM_RS_READ_VERSION_CONFLICT  1
#define STM_RS_VALIDATE_FAIL          2
#define STM_RS_PREPARE_OOM            3
#define STM_RS_LOCK_WAIT_OVERFLOW     4
#define STM_RS_PREPARE_VALIDATE_FAIL  5
#define STM_RS_EXPLICIT_ABORT         6
#define STM_RS_NONE                   255

// TX_READ result_kind values (read_result_hash).
#define STM_READ_RES_OWN_WRITE   0
#define STM_READ_RES_SHARED      1
#define STM_READ_RES_ABORT_LOCK  2
#define STM_READ_RES_ABORT_VER   3

// queue_kind values (queue_hash).
#define STM_QK_LOCK_WAIT   0
#define STM_QK_RETRY_WATCH 1

#define STM_FNV_OFFSET 1469598103934665603ULL
#define STM_FNV_PRIME  1099511628211ULL
#define STM_U64_MAX    0xFFFFFFFFFFFFFFFFULL
#define STM_I64_MIN    ((int64_t)0x8000000000000000LL)

/*
CONTRACT: stm_commit_arbiter

T54 -- STM Commit Arbiter with Retry Watchsets and Versioned Locations.

A persistent software-transactional-memory (STM) engine grounded in TL2: read
and write sets, commit-time write locking, a global version clock, read-set
validation, deterministic retry/wake queues, nontransactional writes, and exact
abort/finalization streams.

PERSISTENT STATE (after init/reset):
  global_version = 0; event_seq = 0; xid_next = 1; begin_seq_next = 1;
  write_seq_next = 1; wait_seq_next = 1; retry_seq_next = 1.

  Location table keyed by addr: addr; value:i64; version:u64;
  lock_owner_txn_or_ZERO; write_seq:u64. Absent locations read as value=0,
  version=0, unlocked; not materialized until first write.

  Transaction table keyed by external txn_id: txn_id; xid; begin_seq;
  attempt_no; priority:u32; start_version; status; wait_addr; wait_seq;
  prepare_seq; plus read set / write set / watch set.

  Read set keyed by addr: addr; read_version; read_value:i64; from_own_write:u8.
  Write set keyed by addr: addr; value:i64; write_set_seq.
  Lock wait queues per location: entries ordered by wait_seq, each {txn_id}.
  Retry watch queues per location: entries ordered by retry_seq, each {txn_id}.

OP STREAM:
  Each solution_run processes batch_size operations in order. op_index is a
  GLOBAL operation index since reset, starting at 0; used in hashes. event_seq
  is a global monotone counter incremented exactly once per EMITTED event
  (every stm_event_hash record consumes one event_seq, including INVALID).

HASH PRIMITIVE (normative; every graded hash uses exactly this):
  All five output hashes are FNV-1a-64 over a byte stream.
    offset_basis = STM_FNV_OFFSET = 1469598103934665603 = 0x14650FB0739D0383
                   (this PROJECT basis, NOT the canonical 0xCBF29CE484222325).
    prime        = STM_FNV_PRIME  = 1099511628211 = 0x00000100000001B3.
  Each hash accumulator h starts at offset_basis. For every byte b consumed:
      h = (h XOR b) * prime          (mod 2^64; XOR first, THEN multiply)
  A field of width W bytes is consumed as W bytes in NATIVE LITTLE-ENDIAN order
  (least-significant byte first). Field widths: u8=1, u32=4, u64=8, i64=8
  (i64 is two's-complement little-endian; e.g. STM_I64_MIN serializes as bytes
  00 00 00 00 00 00 00 80). Records within a hash are concatenated with no
  separators/padding; fields within a record are consumed in the exact order
  listed under OUTPUT HASHES. All arithmetic on counters/versions/seqs is mod
  2^64; all orderings (addr, txn_id, seq) compare as UNSIGNED 64-bit integers.

EMISSION MECHANICS (normative):
  op_index: a GLOBAL u64 op counter, 0 at reset, that is the index of the
    operation currently being processed. Every event emitted while processing
    operation N records op_index=N (PRE-increment value). After an operation
    fully finishes (valid or invalid, regardless of how many events it emitted)
    op_index advances by exactly 1. It is cast to u32 in stm_event_hash.
  event_seq: a GLOBAL u64, 0 at reset, monotone. Each EMITTED stm_event_hash
    record consumes the CURRENT event_seq into its event_seq field, then
    event_seq advances by 1. INVALID events also consume an event_seq. Several
    field values are defined as "the event_seq that event E consumes" — that is
    the event_seq value at the instant E is emitted, i.e. BEFORE its increment:
      - TXN_PREPARED.aux = txn.prepare_seq = event_seq consumed by TXN_PREPARED.
      - LOCATION_WRITE.aux = location.write_seq = event_seq consumed by that
        LOCATION_WRITE.
      - NON_TX_WRITE.aux is 0, but location.write_seq is set to the event_seq
        consumed by that NON_TX_WRITE.
  stm_event_hash and read_result_hash are PER-STEP: both reset to offset_basis
    at the start of every solution_run and accumulate only that step's records
    in emission order. location_hash/txn_hash/queue_hash are recomputed from
    scratch (starting at offset_basis) over the FULL persistent state AFTER the
    step's batch is processed.

OPERATIONS (this contract is the complete, self-contained normative authority; everything graded is specified below):

  BEGIN(txn_id, priority): invalid if txn_id exists or table full. Else
    xid=xid_next++; begin_seq=begin_seq_next++; attempt_no=0;
    start_version=global_version; empty sets; status=ACTIVE; emit TXN_BEGIN.

  TX_READ(txn_id, read_id, addr): invalid if absent or status!=ACTIVE. If addr
    in write set: return staged value, set/update read entry from_own_write=1,
    read_version=U64_MAX, emit READ_OWN_WRITE. Else inspect location: if locked
    by another txn -> abort READ_LOCK_CONFLICT; if version>start_version ->
    abort READ_VERSION_CONFLICT; else set/update read entry with location
    value/version, emit READ_SHARED. Emits one read-result record either way.

  TX_WRITE(txn_id, addr, value): invalid if absent or status!=ACTIVE. If addr
    in write set: update value, keep write_set_seq. Else if write set full ->
    invalid. Else insert write_set_seq=write_seq_next++. Emit WRITE_STAGE.

  VALIDATE(txn_id): invalid if absent or status!=ACTIVE. Validate read entries
    ascending addr, skipping from_own_write==1 and addrs in write set. Each
    remaining: location (default version 0) must not be locked by another txn
    and version==read_version. On first failure abort VALIDATE_FAIL (failing
    addr). Else emit VALIDATE_OK.

  TRY_PREPARE(txn_id): invalid if absent or status!=ACTIVE. If write set empty:
    remove txn, emit COMMIT_READONLY (global_version unchanged). Else lock
    write-set addrs ascending. Per addr: if absent and no room to materialize ->
    abort PREPARE_OOM; if unlocked or self-locked -> lock_owner=txn_id, emit
    WRITE_LOCK; if locked by other -> status=WAITING_LOCK, wait_addr=addr,
    wait_seq=wait_seq_next++, append to lock wait queue if room (else abort
    LOCK_WAIT_OVERFLOW), emit TXN_WAIT_LOCK, release locks acquired earlier in
    THIS prepare in descending addr order (WRITE_UNLOCK_PARTIAL + wakes), stop.
    After all locks held, validate read set as VALIDATE; on fail abort
    PREPARE_VALIDATE_FAIL. Else status=PREPARED, prepare_seq=event_seq of
    TXN_PREPARED, emit TXN_PREPARED.

  DRAIN_COMMITS(limit): if limit==0 valid no-op. While limit>0 and prepared set
    nonempty: choose prepared txn by descending priority, ascending prepare_seq,
    ascending txn_id. commit_version=++global_version. Apply write set ascending
    addr: materialize if needed, value, version=commit_version,
    write_seq=event_seq of LOCATION_WRITE, emit LOCATION_WRITE. Remove txn,
    release its write locks ascending addr (WRITE_UNLOCK_COMMIT). For each
    written addr ascending: wake retry watchers in queue order (clear sets,
    attempt_no++, start_version=global_version, status=ACTIVE, TXN_WAKE_RETRY;
    skip absent/active/removed). For each unlocked addr ascending: wake at most
    one lock waiter (first still WAITING_LOCK on that addr; drop stale entries
    silently first) -> ACTIVE, clear wait_addr/wait_seq, TXN_WAKE_LOCK. Emit
    COMMIT_DONE. Decrement limit.

  RETRY(txn_id, watch_count, watch_addrs[]): invalid if absent, status!=ACTIVE,
    watch_count==0, or watch_count>max_watch_set. Dedup watch_addrs ascending.
    If deduped count>max_watch_set invalid. Validate read set as VALIDATE; on
    fail: clear read/write sets, attempt_no++, start_version=global_version,
    keep ACTIVE, emit RETRY_IMMEDIATE. Else append txn to each watched
    location's retry watch queue ascending addr, single retry_seq=retry_seq_next++
    for all; if any queue lacks capacity, roll back all added entries, keep
    active, emit RETRY_WATCH_OVERFLOW. On success clear write set, KEEP read set,
    status=SUSPENDED_RETRY, emit TXN_SUSPEND_RETRY.

  NON_TX_WRITE(addr, value): if locked by any txn -> emit NON_TX_STALL_LOCKED.
    If absent and table full -> emit NON_TX_OOM. Else materialize if needed,
    version=++global_version, value, write_seq=event_seq of NON_TX_WRITE, emit
    NON_TX_WRITE. Then wake retry watchers for addr (same rule as commit).

  ABORT(txn_id): invalid if absent. Remove txn from all queues/sets, release all
    held locks ascending addr (WRITE_UNLOCK_ABORT + wake-lock events for those
    locations), then emit TXN_ABORT reason EXPLICIT_ABORT.

DETERMINISM: write locks ascending addr; prepared drain by priority desc /
  prepare_seq asc / txn_id asc; validation skips own-write & write-set addrs;
  failed-prepare lock release descending addr; commit wake order = location
  writes, commit unlocks, retry watchers (written addrs), lock waiters (unlocked
  addrs), COMMIT_DONE; retry watcher wake clears read+write sets & advances
  attempt_no; non-tx writes blocked by any lock; absent locations version 0
  until first write; all counters wrap mod 2^64, unsigned numeric ordering.

INVALID ops emit one INVALID event (event_seq consumed), increment
invalid_count, emit no read-result/other records.

OUTPUT COUNTS (per step batch): txn_begun; read_own_write; read_shared;
  write_staged; validate_ok; readonly_commits; txn_prepared; write_locks;
  wait_locks; commits_done; location_writes; non_tx_writes; non_tx_stalls;
  retry_suspended; retry_immediate; retry_watch_overflow; wake_retry; wake_lock;
  aborts; partial_unlocks; commit_unlocks; abort_unlocks; invalid_count.

OUTPUT HASHES (all FNV-1a-64, offset basis 1469598103934665603):
  stm_event_hash: all events in emission order. Fields per event (in this exact
    order, widths as shown):
    event_kind:u8; event_seq:u64; op_index:u32; txn_id_or_ZERO:u64;
    addr_or_U64MAX:u64; value_or_I64MIN:i64; version_or_U64MAX:u64;
    reason_or_255:u8; aux:u64.
    PER-EVENT FIELD POPULATION (a field not meaningful for an event takes its
    sentinel: addr->U64_MAX, value->I64_MIN, version->U64_MAX, reason->255):
      TXN_BEGIN(0):           txn; addr=MAX; value=MIN; version=start_version;
                              reason=255; aux=priority.
      READ_OWN_WRITE(1):      txn; addr; value=staged value; version=MAX;
                              reason=255; aux=read_id.
      READ_SHARED(2):         txn; addr; value=loc value; version=loc version;
                              reason=255; aux=read_id.
      WRITE_STAGE(3):         txn; addr; value=value; version=MAX; reason=255;
                              aux=write_set_seq.
      VALIDATE_OK(4):         txn; addr=MAX; value=MIN; version=MAX; reason=255;
                              aux=0.
      WRITE_LOCK(5):          txn; addr; value=MIN; version=loc version (current
                              version of the locked location; 0 if just
                              materialized); reason=255; aux=0.
      TXN_WAIT_LOCK(6):       txn; addr; value=MIN; version=MAX; reason=255;
                              aux=wait_seq.
      WRITE_UNLOCK_PARTIAL(7):txn; addr; value=MIN; version=MAX; reason=255;
                              aux=0.
      TXN_PREPARED(8):        txn; addr=MAX; value=MIN; version=MAX; reason=255;
                              aux=prepare_seq (=event_seq this event consumes).
      COMMIT_READONLY(9):     txn; addr=MAX; value=MIN; version=MAX; reason=255;
                              aux=0.
      LOCATION_WRITE(10):     txn=committing txn; addr; value=value;
                              version=commit_version; reason=255;
                              aux=write_seq (=event_seq this event consumes).
      WRITE_UNLOCK_COMMIT(11):txn; addr; value=MIN; version=MAX; reason=255;
                              aux=0.
      TXN_WAKE_RETRY(12):     txn=woken txn; addr=wake addr; value=MIN;
                              version=MAX; reason=255; aux=attempt_no AFTER the
                              ++ (post-increment value).
      TXN_WAKE_LOCK(13):      txn=woken txn; addr=wake addr; value=MIN;
                              version=MAX; reason=255; aux=0.
      COMMIT_DONE(14):        txn=committed txn; addr=MAX; value=MIN;
                              version=commit_version; reason=255; aux=0.
      TXN_SUSPEND_RETRY(15):  txn; addr=MAX; value=MIN; version=MAX; reason=255;
                              aux=retry_seq.
      RETRY_IMMEDIATE(16):    txn; addr=MAX; value=MIN; version=MAX; reason=255;
                              aux=attempt_no AFTER the ++ (post-increment value).
      RETRY_WATCH_OVERFLOW(17):txn; addr=failing (overflowing) addr; value=MIN;
                              version=MAX; reason=255; aux=0.
      NON_TX_WRITE(18):       txn=0; addr; value=value; version=new version;
                              reason=255; aux=0.
      NON_TX_STALL_LOCKED(19):txn=0; addr; value=value; version=MAX; reason=255;
                              aux=lock_owner (current owner txn_id).
      NON_TX_OOM(20):         txn=0; addr; value=value; version=MAX; reason=255;
                              aux=0.
      WRITE_UNLOCK_ABORT(21): txn; addr; value=MIN; version=MAX; reason=255;
                              aux=0.
      TXN_ABORT(22):          txn; addr=failing addr (or MAX for EXPLICIT_ABORT);
                              value=MIN; version=MAX; reason=<abort reason code>;
                              aux=0.
      INVALID(23):            txn (the op's txn_id field, may be 0); addr=MAX;
                              value=MIN; version=MAX; reason=255; aux=0.
  read_result_hash: exactly one record per TX_READ operation, in op order (fields
    in this exact order/width):
    read_id:u64; txn_id:u64; addr:u64; result_kind:u8; value_or_I64MIN:i64;
    version_or_U64MAX:u64; attempt_no:u64.
    attempt_no = the txn's attempt_no captured at the START of the TX_READ (a
    conflict-abort does not change the recorded attempt_no). result_kind:
      OWN_WRITE(0): value=staged value; version=U64_MAX.
      SHARED(1):    value=loc value;    version=loc version.
      ABORT_LOCK(2):value=I64_MIN;      version=U64_MAX (read-lock conflict).
      ABORT_VER(3): value=I64_MIN;      version=U64_MAX (read-version conflict).
    For the two abort kinds the read-result record is emitted BEFORE the
    resulting WRITE_UNLOCK_ABORT/TXN_WAKE_LOCK/TXN_ABORT events.
  location_hash: only MATERIALIZED locations, by addr ascending. Per location
    (this order/width): addr:u64; value:i64; version:u64; lock_owner_or_ZERO:u64;
    write_seq:u64.
  txn_hash: only LIVE transactions, by txn_id ascending. Per txn, the header
    fields in this exact order/width:
      txn_id:u64; xid:u64; begin_seq:u64; attempt_no:u64; priority:u32;
      start_version:u64; status:u8; wait_addr_or_U64MAX:u64;
      prepare_seq_or_U64MAX:u64
    (NOTE: wait_seq is NOT part of txn_hash); then the txn's read-set entries by
    addr ascending, then its write-set entries by addr ascending.
      Read entry:  addr:u64; read_version:u64; read_value:i64; from_own_write:u8.
      Write entry: addr:u64; value:i64; write_set_seq:u64.
  queue_hash: first ALL lock-wait queues by addr ascending (skip empty queues),
    each in queue order; then ALL retry-watch queues by addr ascending (skip
    empty), each in queue order. Per queue entry (this order/width):
      queue_kind:u8; addr:u64; position:u64; txn_id:u64; seq:u64.
    queue_kind = STM_QK_LOCK_WAIT(0) for lock-wait, STM_QK_RETRY_WATCH(1) for
    retry-watch. position = 0-based index within that addr's queue. For lock-wait
    entries seq = wait_seq and queue order = append order (wait_seq ascending).
    For retry-watch entries seq = retry_seq and queue order = (retry_seq
    ascending, then insertion order ascending) among entries sharing that addr.

stm_event_hash/read_result_hash are per-step (start at offset basis each step).
location_hash/txn_hash/queue_hash reflect full persistent state after the step.
All outputs exact integers.

Rules:
  - solution_init may allocate persistent state.
  - solution_run may not call cudaMalloc/cudaFree.
  - solution_run may launch kernels and use provided workspace.
*/

struct alignas(8) StmProblemSpec {
    int32_t abi_version;
    int32_t max_txns;
    int32_t max_locations;
    int32_t max_read_set;
    int32_t max_write_set;
    int32_t max_watch_set;
    int32_t max_waiters_per_location;
    int32_t max_retry_watchers_per_location;
    int32_t max_batch;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[5];
};

struct alignas(8) StmRunSpec {
    int32_t abi_version;
    int32_t batch_size;
    int32_t step_id;
    int32_t reserved[13];
};

// Operations in Structure-of-Arrays form. Each array has batch_size entries.
//   op_kind:   one of STM_OP_*
//   txn_id:    external transaction id (u64); 0 for NON_TX_WRITE.
//   read_id:   read identifier (TX_READ); else 0.
//   addr:      addr (TX_READ/TX_WRITE/NON_TX_WRITE); else 0 / U64_MAX-unused.
//   value:     value (TX_WRITE/NON_TX_WRITE); else 0.
//   aux:       priority (BEGIN); limit (DRAIN_COMMITS); watch_count (RETRY);
//              else 0.
//   watch_off: offset into watch_addrs[] for RETRY's watch list; else 0.
// watch_addrs is a flat global buffer; RETRY reads aux entries starting at
// watch_off.
struct alignas(8) StmInputs {
    const int32_t*  op_kind;
    const uint64_t* txn_id;
    const uint64_t* read_id;
    const uint64_t* addr;
    const int64_t*  value;
    const uint64_t* aux;
    const uint64_t* watch_off;
    const uint64_t* watch_addrs;  // flat buffer
};

struct alignas(8) StmOutputs {
    // Counts (23).
    int32_t* txn_begun;
    int32_t* read_own_write;
    int32_t* read_shared;
    int32_t* write_staged;
    int32_t* validate_ok;
    int32_t* readonly_commits;
    int32_t* txn_prepared;
    int32_t* write_locks;
    int32_t* wait_locks;
    int32_t* commits_done;
    int32_t* location_writes;
    int32_t* non_tx_writes;
    int32_t* non_tx_stalls;
    int32_t* retry_suspended;
    int32_t* retry_immediate;
    int32_t* retry_watch_overflow;
    int32_t* wake_retry;
    int32_t* wake_lock;
    int32_t* aborts;
    int32_t* partial_unlocks;
    int32_t* commit_unlocks;
    int32_t* abort_unlocks;
    int32_t* invalid_count;
    // Hashes (5).
    uint64_t* stm_event_hash;
    uint64_t* read_result_hash;
    uint64_t* location_hash;
    uint64_t* txn_hash;
    uint64_t* queue_hash;
};

static_assert(sizeof(StmProblemSpec) == 64, "StmProblemSpec layout drift");
static_assert(sizeof(StmRunSpec) == 64, "StmRunSpec layout drift");
static_assert(sizeof(StmInputs) == 64, "StmInputs layout drift");
static_assert(sizeof(StmOutputs) == 224, "StmOutputs layout drift");

static inline size_t stm_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int stm_validate_problem_spec(const StmProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != STM_ABI_VERSION) return 0;
    if (spec->max_txns < STM_MIN_TXNS || spec->max_txns > STM_MAX_TXNS) return 0;
    if (spec->max_locations < STM_MIN_LOCATIONS || spec->max_locations > STM_MAX_LOCATIONS) return 0;
    if (spec->max_read_set < STM_MIN_READ_SET || spec->max_read_set > STM_MAX_READ_SET) return 0;
    if (spec->max_write_set < STM_MIN_WRITE_SET || spec->max_write_set > STM_MAX_WRITE_SET) return 0;
    if (spec->max_watch_set < STM_MIN_WATCH_SET || spec->max_watch_set > STM_MAX_WATCH_SET) return 0;
    if (spec->max_waiters_per_location < STM_MIN_WAITERS ||
        spec->max_waiters_per_location > STM_MAX_WAITERS) return 0;
    if (spec->max_retry_watchers_per_location < STM_MIN_RETRY_WATCHERS ||
        spec->max_retry_watchers_per_location > STM_MAX_RETRY_WATCHERS) return 0;
    if (spec->max_batch < 0 || spec->max_batch > STM_MAX_BATCH) return 0;
    if (spec->max_steps < 1 || spec->max_steps > STM_MAX_STEPS) return 0;
    return 1;
}

static inline int stm_validate_run_spec(const StmRunSpec* run, const StmProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != STM_ABI_VERSION) return 0;
    if (run->batch_size < 0 || run->batch_size > spec->max_batch) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const StmProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const StmProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const StmRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // STM_COMMIT_ARBITER_COMMON_H_
