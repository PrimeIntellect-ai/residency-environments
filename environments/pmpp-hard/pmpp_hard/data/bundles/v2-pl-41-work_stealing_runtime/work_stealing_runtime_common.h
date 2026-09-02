// file: work_stealing_runtime_common.h

#ifndef WORK_STEALING_RUNTIME_COMMON_H_
#define WORK_STEALING_RUNTIME_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define WSR_ABI_VERSION 1

// Limits.
#define WSR_MIN_WORKERS 1
#define WSR_MAX_WORKERS 64
#define WSR_MIN_LOCAL_CAP 1
#define WSR_MAX_LOCAL_CAP 256
#define WSR_MIN_GLOBAL_CAP 1
#define WSR_MAX_GLOBAL_CAP 4096
#define WSR_MIN_MAX_TASKS 1
#define WSR_MAX_MAX_TASKS 8192
#define WSR_MAX_MAX_BLOCKED 8192
#define WSR_MAX_MAX_SLEEPING 8192
#define WSR_MAX_STEPS 4096

// Operation kinds, packed into WsrRunSpec.op_kind.
enum WsrOpKind {
    WSR_OP_SPAWN = 0,
    WSR_OP_RUN = 1,
    WSR_OP_YIELD = 2,
    WSR_OP_BLOCK = 3,
    WSR_OP_SLEEP = 4,
    WSR_OP_WAKE = 5,
    WSR_OP_ADVANCE = 6,
    WSR_OP_CANCEL = 7,
    WSR_OP_KIND_COUNT = 8
};

// Event kinds, in ordinal order (matches contract section 4).
enum WsrEventKind {
    WSR_EV_SPAWN_LOCAL = 0,
    WSR_EV_SPAWN_GLOBAL = 1,
    WSR_EV_SPAWN_REJECT = 2,
    WSR_EV_STEAL_TASK = 3,
    WSR_EV_SCHEDULE = 4,
    WSR_EV_RUN_SLICE = 5,
    WSR_EV_COMPLETE = 6,
    WSR_EV_YIELD = 7,
    WSR_EV_YIELD_EMPTY = 8,
    WSR_EV_BLOCK = 9,
    WSR_EV_BLOCK_EMPTY = 10,
    WSR_EV_SLEEP = 11,
    WSR_EV_SLEEP_EMPTY = 12,
    WSR_EV_WAKE_LOCAL = 13,
    WSR_EV_WAKE_GLOBAL = 14,
    WSR_EV_WAKE_STALLED = 15,
    WSR_EV_CLOCK_ADVANCE = 16,
    WSR_EV_CANCEL_EXPLICIT = 17,
    WSR_EV_CANCEL_OVERFLOW = 18,
    WSR_EV_IDLE = 19,
    WSR_EV_INVALID = 20,
    WSR_EV_KIND_COUNT = 21
};

// Cancellation reason codes (aux0 of CANCEL_OVERFLOW).
enum WsrCancelReason {
    WSR_REASON_YIELD_OVERFLOW = 1,
    WSR_REASON_BLOCK_OVERFLOW = 2,
    WSR_REASON_SLEEP_OVERFLOW = 3
};

/*
CONTRACT: work_stealing_runtime  (T41)

A deterministic, persistent multi-worker work-stealing runtime simulated on the
device. Each solution_run applies exactly ONE scheduler operation (encoded in
WsrRunSpec) to persistent state and emits a fully-determined event stream.

Persistent state (after init/reset):
  clock = 0, event_seq = 0, op_index = 0.
  For each worker w: two local deques local_high[w], local_low[w]
  (head=oldest, tail=newest), one running slot (empty or one task_id).
  Global queues global_high, global_low (FIFO).
  Task table keyed by task_id holding all NONTERMINAL tasks. Fields:
    home_worker, priority in {0,1} (1=high), remaining_work, state,
    birth_seq, last_run_seq, block_seq, sleep_seq, wait_key, wake_tick.
  Blocked index: per wait_key FIFO ordered by block_seq.
  Sleeping index: ordered by (wake_tick, sleep_seq, task_id).
  Terminal tasks (completed/cancelled/rejected) are NOT retained; ids may reuse.

Operation encoding (WsrRunSpec):
  op_kind selects the operation. Arguments:
    SPAWN  : a_task = task_id, a_worker = home_worker, a_priority = priority,
             a_work = work (remaining_work).
    RUN    : a_worker = worker, a_work = quantum.
    YIELD  : a_worker = worker.
    BLOCK  : a_worker = worker, a_key = wait_key.
    SLEEP  : a_worker = worker, a_tick = wake_tick.
    WAKE   : a_key = wait_key, a_limit = limit.
    ADVANCE: a_delta = delta.
    CANCEL : a_task = task_id.

Event sequencing:
  event_seq is a u64 counter. Every emitted event consumes the current
  event_seq as its own seq, then event_seq increments (mod 2^64). Sequence
  fields (birth_seq, block_seq, sleep_seq, last_run_seq) take the event_seq of
  the primary event recording that transition. op_index increments once per op.

Outputs after EVERY op (exact integers):
  counts[WSR_COUNT_N]   : monotonically accumulated counters (see WsrCountIdx).
  op_index_out          : index of this op.
  clock_out, event_seq_out : u64 scheduler clock and next event_seq.
  sched_event_hash      : persistent running FNV-1a-64 over the event stream.
  ready_hash, running_hash, blocked_hash, sleep_hash : snapshot FNV hashes.
  state_checksum        : master FNV combining all of the above + counts.

Rules:
  - solution_init may allocate persistent state.
  - solution_run may NOT call cudaMalloc/cudaFree.
  - All graded outputs are exact integers (no float, no tolerance).
*/

/*
=== DETERMINISM & EXACT OUTPUT SERIALIZATION (normative) ===

This section is the COMPLETE and AUTHORITATIVE specification of every graded
integer. It is self-contained: a solver reading only this header can reproduce
all hashes and counters bit-for-bit. Nothing is deferred to any other file.

----------------------------------------------------------------------------
0. FNV-1a-64 primitive
----------------------------------------------------------------------------
  FNV_OFFSET = 1469598103934665603 (0x14650FB0739D0383)
  FNV_PRIME  = 1099511628211       (0x00000100000001B3)
  fnv_byte(h, b):  h ^= (uint64_t)b ;  h *= FNV_PRIME            (wraps mod 2^64)
  To "fold value v of width N bytes" means: take the N little-endian bytes of v
  (the native x86_64/CUDA in-memory byte order, least-significant byte first)
  and apply fnv_byte to each in increasing address order (byte 0 first).
  Multi-field folds apply each field's bytes in the field order listed below;
  there is NO separator and NO length prefix between fields.

  Field widths used below: u8 = 1 byte, u32 = 4 bytes, u64 = 8 bytes.
  Sentinels: a "missing worker" folds as u32 = 0xFFFFFFFF (UINT32_MAX);
             a "missing priority" folds as u8 = 255;
             a "missing task_id"/"missing remaining" folds as u64 = 0xFFFF...FF
             (UINT64_MAX). These exact sentinels MUST be folded, not skipped.

----------------------------------------------------------------------------
1. event_seq, op_index, clock
----------------------------------------------------------------------------
  Persistent scalars (after init/reset): clock=0, event_seq=0, op_index=0,
  sched_hash=FNV_OFFSET, blocked_total=0.
  Each emitted event consumes the CURRENT event_seq as its own seq, then
  event_seq += 1 (mod 2^64). Sequence stamps written into tasks
  (birth_seq, last_run_seq, block_seq, sleep_seq) are read as the CURRENT
  event_seq AT THE MOMENT the stamp is taken, which (for every operation here)
  equals the event_seq that the very next emitted primary event will consume.
  After the whole op runs, op_index is captured as op_index_out, then
  op_index += 1. clock changes only on ADVANCE (clock += delta, wraps 2^64).

----------------------------------------------------------------------------
2. sched_event_hash  (persistent running event-stream FNV)
----------------------------------------------------------------------------
  Seeded ONCE at init/reset to FNV_OFFSET and never reseeded; it persists
  across ops. For EACH emitted event, in emission order, fold these fields in
  EXACTLY this order/width into the running sched_hash:
      kind      u8    (WsrEventKind ordinal)
      seq       u64   (the event_seq this event consumed)
      op_index  u32   (current op_index, low 32 bits)
      worker    u32   (worker, or 0xFFFFFFFF if none)
      task_id   u64   (task id, or UINT64_MAX if none)
      aux0      u64   (per-event, see table below)
      aux1      u64   (per-event, see table below)
      prio      u8    (0 or 1, or 255 if none)
      rem_after u64   (remaining_work after the transition, or UINT64_MAX)
  sched_event_hash output = the running sched_hash AFTER this op's events.

  Per-event (kind, worker, task_id, aux0, aux1, prio, rem_after) emitted by
  each operation (this enumerates EVERY event the reference emits):

   SPAWN op (args task_id,home,prio in{0,1},work):
     invalid (home out of [0,W), prio not in {0,1}, work==0, OR task_id already
       live): counts[INVALID]++ ; emit INVALID(worker=none, task=UINT64_MAX,
       aux0=0,aux1=0,prio=none,rem=UINT64_MAX).
     else if live task count >= max_tasks: counts[SPAWN_REJECT]++ ;
       emit SPAWN_REJECT(home, task_id, 0,0, prio, UINT64_MAX).
     else allocate; birth_seq = current event_seq; place ready (sec. 5):
       placed local: counts[SPAWN_LOCAL]++ ;
         emit SPAWN_LOCAL(home, task_id, 0,0, prio, work).
       placed global: counts[SPAWN_GLOBAL]++ ;
         emit SPAWN_GLOBAL(home, task_id, 0,0, prio, work).
       placement failed (both full): free slot; counts[SPAWN_REJECT]++ ;
         emit SPAWN_REJECT(home, task_id, 0,0, prio, UINT64_MAX).

   RUN op (args worker, quantum):
     invalid (worker out of [0,W) OR quantum==0): counts[INVALID]++ ; emit
       INVALID(none,UINT64_MAX,0,0,none,UINT64_MAX).
     If worker already has a running task: reuse it (no SCHEDULE, no steal).
     Else pick (sec. 6). If pick empties and steal+repick still empty:
       counts[IDLE]++ ; emit IDLE(worker, UINT64_MAX, 0,0, none, UINT64_MAX);
       return.
     On successful (re)schedule of a freshly picked task:
       set running; counts[SCHEDULED]++ ;
       emit SCHEDULE(worker, tid, 0,0, prio, rem_now).
     Then always: slice = min(quantum, rem); rem -= slice;
       last_run_seq = current event_seq; counts[RUN_SLICES]++ ;
       emit RUN_SLICE(worker, tid, aux0=slice, 0, prio, rem_after).
     If rem_after == 0: free task (terminal); running slot cleared;
       counts[COMPLETED]++ ; emit COMPLETE(worker, tid, 0,0, prio, 0).
     Steal events (sec. 7) are emitted BEFORE SCHEDULE.
     RUN emission order overall: STEAL_TASK*, SCHEDULE?, RUN_SLICE, COMPLETE?.

   YIELD op (arg worker):
     invalid worker: INVALID as above.
     no running task on worker: counts[EMPTY_OP]++ ;
       emit YIELD_EMPTY(worker, UINT64_MAX, 0,0, none, UINT64_MAX).
     else clear running, place ready (sec. 5):
       fail: free task; counts[OVERFLOW_CANCELLED]++ ;
         emit CANCEL_OVERFLOW(worker, tid, aux0=WSR_REASON_YIELD_OVERFLOW(=1),
                              0, prio, rem).
       ok: counts[YIELDED]++ ;
         emit YIELD(worker, tid, 0,0, prio, rem).

   BLOCK op (arg worker, wait_key):
     invalid worker: INVALID.
     no running task: counts[EMPTY_OP]++ ;
       emit BLOCK_EMPTY(worker, UINT64_MAX, 0,0, none, UINT64_MAX).
     else clear running. If blocked_total >= max_blocked:
       free task; counts[OVERFLOW_CANCELLED]++ ;
       emit CANCEL_OVERFLOW(worker, tid, WSR_REASON_BLOCK_OVERFLOW(=2),0,prio,rem).
     else set BLOCKED, wait_key, block_seq = current event_seq,
       append tid to the wait_key FIFO tail, blocked_total += 1;
       counts[BLOCKED]++ ;
       emit BLOCK(worker, tid, aux0=wait_key, 0, prio, rem).

   SLEEP op (arg worker, wake_tick):
     invalid worker: INVALID.
     no running task: counts[EMPTY_OP]++ ;
       emit SLEEP_EMPTY(worker, UINT64_MAX, 0,0, none, UINT64_MAX).
     else clear running. (EFFECTIVE CAP: the sleeping cap used here is
       effective_max_sleeping = max(max_sleeping, 1); i.e. a spec value of 0 is
       treated as 1, NOT as "overflow on the first sleep".) If sleeping count >=
       effective_max_sleeping:
       free task; counts[OVERFLOW_CANCELLED]++ ;
       emit CANCEL_OVERFLOW(worker, tid, WSR_REASON_SLEEP_OVERFLOW(=3),0,prio,rem).
     else set SLEEPING, wake_tick, sleep_seq = current event_seq,
       add tid to the sleeping pool; counts[SLEPT]++ ;
       emit SLEEP(worker, tid, aux0=wake_tick, 0, prio, rem).

   WAKE op (arg wait_key, limit):
     if limit <= 0: NO events, NO counter, NO state change.
     if wait_key has no active blocked FIFO: NO events.
     else repeatedly while (woken < limit AND FIFO nonempty): pop FRONT tid,
       place ready (sec. 5):
         fail: push tid back to FRONT of its FIFO, restore BLOCKED, emit
           WAKE_STALLED(home, tid, aux0=wait_key, 0, prio, rem); STOP the op
           (no further wakes; blocked_total unchanged for this tid).
         ok: blocked_total -= 1;
           placed local: counts[WOKEN_LOCAL]++ ;
             emit WAKE_LOCAL(home, tid, aux0=wait_key, 0, prio, rem).
           placed global: counts[WOKEN_GLOBAL]++ ;
             emit WAKE_GLOBAL(home, tid, aux0=wait_key, 0, prio, rem).
     (No "no-op" counter; WAKE that wakes nobody emits nothing.)

   ADVANCE op (arg delta):
     clock += delta (wraps). FIRST emit CLOCK_ADVANCE(worker=none,
       task=UINT64_MAX, aux0=delta, aux1=new_clock, prio=none, rem=UINT64_MAX).
     Then repeatedly: select the minimum sleeping task by
       (wake_tick, sleep_seq, task_id) ascending; if its wake_tick > clock STOP.
       place ready (sec. 5):
         fail: emit WAKE_STALLED(home, id, aux0=wake_tick, 0, prio, rem); STOP
           (task stays sleeping; no erase).
         ok: erase from sleeping pool;
           local: counts[WOKEN_LOCAL]++ ; emit WAKE_LOCAL(home,id,wake_tick,0,prio,rem).
           global: counts[WOKEN_GLOBAL]++ ; emit WAKE_GLOBAL(home,id,wake_tick,0,prio,rem).
     (CLOCK_ADVANCE is always emitted, even if delta==0 / nobody wakes.)

   CANCEL op (arg task_id):
     not live: counts[INVALID]++ ; emit INVALID(none,UINT64_MAX,0,0,none,UINT64_MAX).
     else remove it from whatever structure holds it (running slot / local or
       global ready deque / blocked FIFO with blocked_total -=1 / sleeping pool),
       free the slot; counts[CANCELLED]++ ;
       emit CANCEL_EXPLICIT(home, task_id, 0,0, prio, rem) where prio/rem are the
       task's values BEFORE removal.

----------------------------------------------------------------------------
3. ready_hash  (snapshot; reseeded to FNV_OFFSET each call)
----------------------------------------------------------------------------
  h = FNV_OFFSET. Traversal order:
    for worker w = 0..W-1:
       high deque (prio 1) head->tail, local position counter pos starting 0;
       then low deque (prio 0) head->tail, pos restarting at 0.
    then global high (prio 1) head->tail, pos restarting at 0;
    then global low  (prio 0) head->tail, pos restarting at 0.
  For each entry fold, in order:
       src_kind  u8   (0 = local deque, 1 = global deque)
       worker    u32  (the worker id for local entries; 0xFFFFFFFF for global)
       prio      u8   (1 or 0)
       pos       u64  (0-based index WITHIN that single deque, reset per deque)
       task_id   u64
  Deques are ring buffers; "head->tail" = logical oldest->newest order.
  Owners normally pop the tail (newest) when scheduling; that does not change
  this hash's traversal which is always head(oldest)..tail(newest).

----------------------------------------------------------------------------
4. running_hash  (snapshot; reseeded to FNV_OFFSET each call)
----------------------------------------------------------------------------
  h = FNV_OFFSET. For worker w = 0..W-1, ALWAYS fold:
       worker u32
    then if worker has a running task:
       has u8 = 1 ; task_id u64 ; rem u64 (current remaining_work)
    else:
       has u8 = 0 ; task_id u64 = UINT64_MAX ; rem u64 = UINT64_MAX
  (Empty workers are folded too, with the sentinels above.)

----------------------------------------------------------------------------
5. blocked_hash  (snapshot; reseeded to FNV_OFFSET each call)
----------------------------------------------------------------------------
  h = FNV_OFFSET. Iterate DISTINCT active wait_keys in ASCENDING key order
  (unsigned u64 compare). Within a key, iterate its FIFO head->tail (which is
  block_seq ascending, i.e. arrival order). For each blocked task fold:
       key       u64
       block_seq u64
       task_id   u64
       home      u32
       prio      u8
       rem       u64

----------------------------------------------------------------------------
6. sleep_hash  (snapshot; reseeded to FNV_OFFSET each call)
----------------------------------------------------------------------------
  h = FNV_OFFSET. Iterate sleeping tasks in ASCENDING
  (wake_tick, sleep_seq, task_id) order. For each fold:
       wake_tick u64
       sleep_seq u64
       task_id   u64
       home      u32
       prio      u8
       rem       u64

----------------------------------------------------------------------------
7. Placement (sec. 5 ref'd above) and steal/pick details
----------------------------------------------------------------------------
  place_ready(task with home, prio): if local[home][prio] has < local_cap
    entries, push to its TAIL -> ready_local (return "local"); else if
    global[prio] has < global_cap entries, push to its TAIL -> ready_global
    (return "global"); else return "fail".
  RUN pick_local_or_global(worker): local high tail, else local low tail, else
    global high head, else global low head, else "none". (Owners pop the
    NEWEST = tail; globals pop the OLDEST = head.)
  pick_local_only(worker): local high tail, else local low tail, else "none".
  Steal (only when initial pick is "none"): choose victim = the OTHER worker
    (w != thief) with the largest (local_high+local_low) ready count; ties
    broken by LOWEST worker id; skip workers with 0. If none, no steal.
    Let high_n, low_n = victim's deque sizes, total = high_n+low_n,
    take = ceil(total/2) = (total+1)/2. For i = 0..take-1: if i < high_n take
    from victim HIGH (pop head) and push to thief HIGH tail with prio=1; else
    take from victim LOW (pop head) and push to thief LOW tail with prio=0.
    For each stolen task: counts[STOLEN_TASKS]++ ; emit STEAL_TASK(thief, id,
    aux0=victim_worker_id, aux1=0, prio, rem) where rem is the task's current
    remaining_work. After stealing, re-pick with pick_local_only(thief).

----------------------------------------------------------------------------
8. state_checksum  (master; reseeded to FNV_OFFSET each op)
----------------------------------------------------------------------------
  Computed AFTER the op's op_index has been incremented. mh = FNV_OFFSET, fold
  in EXACTLY this order/width:
       clock        u64
       event_seq    u64
       op_index     u32   (the ALREADY-INCREMENTED op_index, low 32 bits)
       sched_hash   u64   (current persistent sched_event_hash)
       ready_hash   u64
       running_hash u64
       blocked_hash u64
       sleep_hash   u64
       counts[0..WSR_COUNT_N-1] each as u64, in index order
  Note op_index_out reports the PRE-increment value (this op's index), but the
  value folded into state_checksum and into each event's op_index field is the
  current op_index at the time (pre-increment for events; post-increment for the
  master checksum's op_index field).
*/

// Output counter indices.
enum WsrCountIdx {
    WSR_C_SPAWN_LOCAL = 0,
    WSR_C_SPAWN_GLOBAL = 1,
    WSR_C_SPAWN_REJECT = 2,
    WSR_C_SCHEDULED = 3,
    WSR_C_RUN_SLICES = 4,
    WSR_C_COMPLETED = 5,
    WSR_C_YIELDED = 6,
    WSR_C_BLOCKED = 7,
    WSR_C_SLEPT = 8,
    WSR_C_WOKEN_LOCAL = 9,
    WSR_C_WOKEN_GLOBAL = 10,
    WSR_C_STOLEN_TASKS = 11,
    WSR_C_CANCELLED = 12,
    WSR_C_OVERFLOW_CANCELLED = 13,
    WSR_C_IDLE = 14,
    WSR_C_EMPTY_OP = 15,
    WSR_C_INVALID = 16,
    WSR_COUNT_N = 17
};

struct alignas(8) WsrProblemSpec {
    int32_t abi_version;
    int32_t W;                 // number of workers
    int32_t local_cap_per_worker;
    int32_t global_cap;
    int32_t max_tasks;
    int32_t max_blocked;
    int32_t max_sleeping;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[7];
};

struct alignas(8) WsrRunSpec {
    int32_t abi_version;
    int32_t op_kind;           // WsrOpKind
    int32_t step_id;
    int32_t a_worker;          // worker / home_worker
    int32_t a_priority;        // priority
    int32_t a_limit;           // WAKE limit
    int32_t reserved0;
    int32_t reserved1;
    uint64_t a_task;           // task_id
    uint64_t a_work;           // work / quantum
    uint64_t a_key;            // wait_key
    uint64_t a_tick;           // wake_tick
    uint64_t a_delta;          // ADVANCE delta
    uint64_t reserved2;
};

struct alignas(8) WsrInputs {
    // All operands travel through WsrRunSpec; this struct is reserved for
    // ABI symmetry with other tasks and future batched operands.
    const void* reserved;
};

struct alignas(8) WsrOutputs {
    int64_t* counts;           // [WSR_COUNT_N]
    int32_t* op_index_out;     // [1]
    uint64_t* clock_out;       // [1]
    uint64_t* event_seq_out;   // [1]
    uint64_t* sched_event_hash;// [1]
    uint64_t* ready_hash;      // [1]
    uint64_t* running_hash;    // [1]
    uint64_t* blocked_hash;    // [1]
    uint64_t* sleep_hash;      // [1]
    uint64_t* state_checksum;  // [1]
};

static_assert(sizeof(WsrProblemSpec) == 64, "WsrProblemSpec layout drift");
static_assert(sizeof(WsrRunSpec) == 80, "WsrRunSpec layout drift");
static_assert(sizeof(WsrOutputs) == 80, "WsrOutputs layout drift");

static inline size_t wsr_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int wsr_validate_problem_spec(const WsrProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != WSR_ABI_VERSION) return 0;
    if (spec->W < WSR_MIN_WORKERS || spec->W > WSR_MAX_WORKERS) return 0;
    if (spec->local_cap_per_worker < WSR_MIN_LOCAL_CAP ||
        spec->local_cap_per_worker > WSR_MAX_LOCAL_CAP) return 0;
    if (spec->global_cap < WSR_MIN_GLOBAL_CAP ||
        spec->global_cap > WSR_MAX_GLOBAL_CAP) return 0;
    if (spec->max_tasks < WSR_MIN_MAX_TASKS ||
        spec->max_tasks > WSR_MAX_MAX_TASKS) return 0;
    if (spec->max_blocked < 0 || spec->max_blocked > WSR_MAX_MAX_BLOCKED) return 0;
    if (spec->max_sleeping < 0 || spec->max_sleeping > WSR_MAX_MAX_SLEEPING) return 0;
    if (spec->max_steps < 1 || spec->max_steps > WSR_MAX_STEPS) return 0;
    return 1;
}

static inline int wsr_validate_run_spec(const WsrRunSpec* run, const WsrProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != WSR_ABI_VERSION) return 0;
    if (run->op_kind < 0 || run->op_kind >= WSR_OP_KIND_COUNT) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const WsrProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const WsrProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const WsrRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // WORK_STEALING_RUNTIME_COMMON_H_
