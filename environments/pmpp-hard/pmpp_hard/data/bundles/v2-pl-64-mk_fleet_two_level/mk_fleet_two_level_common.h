// file: mk_fleet_two_level_common.h
//
// ABI + contract for T64: "Fleet Two-Level Scope Runtime with Chiplet-Local
// Counters". A persistent chiplet-aware megakernel runtime. Each solution_run
// applies exactly ONE operation (SUBMIT_TASK, SCHED_STEP, WORKER_START, ADVANCE,
// EVENT_FORCE, CANCEL_TASK) to device-resident persistent state and emits the
// updated cumulative counters plus canonical FNV-1a-64 hashes of the dispatch /
// counter / queue / running timeline.

#ifndef MK_FLEET_TWO_LEVEL_COMMON_H_
#define MK_FLEET_TWO_LEVEL_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MKF_ABI_VERSION 1

// Bounds for the persistent runtime (bound device allocations).
#define MKF_MIN_CHIPLETS 1
#define MKF_MAX_CHIPLETS 8
#define MKF_MIN_WPC 1
#define MKF_MAX_WPC 8
#define MKF_MAX_WORKERS 64           // chiplet_count * workers_per_chiplet <= 64
#define MKF_MIN_TASKS 1
#define MKF_MAX_TASKS 256
#define MKF_MIN_EVENTS 1
#define MKF_MAX_EVENTS 256
#define MKF_MIN_READY 1
#define MKF_MAX_READY 256
#define MKF_MIN_WQ 1
#define MKF_MAX_WQ 16
#define MKF_MIN_RUNNING 1
#define MKF_MAX_RUNNING 256
#define MKF_MAX_WAITS 2              // inline waits supported per SUBMIT_TASK

// Task scopes.
#define MKF_SCOPE_WAVE 0
#define MKF_SCOPE_CU 1
#define MKF_SCOPE_CHIPLET 2
#define MKF_SCOPE_DEVICE 3
#define MKF_SCOPE_COUNT 4

// Task statuses.
#define MKF_ST_WAITING 0
#define MKF_ST_READY 1
#define MKF_ST_DISPATCHED 2
#define MKF_ST_RUNNING 3
#define MKF_ST_LOCAL_DONE 4
#define MKF_ST_GLOBAL_DONE 5
#define MKF_ST_CANCELLED 6

// Operation codes (run.op).
#define MKF_OP_SUBMIT 0
#define MKF_OP_SCHED 1
#define MKF_OP_WORKER 2
#define MKF_OP_ADVANCE 3
#define MKF_OP_EVENT_FORCE 4
#define MKF_OP_CANCEL 5

// Event kinds (stable ordinals; hashed in emission order).
#define MKF_EV_TASK_READY 0
#define MKF_EV_TASK_WAIT 1
#define MKF_EV_SCHED_IDLE 2
#define MKF_EV_DISPATCH_SINGLE 3
#define MKF_EV_DISPATCH_CHIPLET_PART 4
#define MKF_EV_DISPATCH_DEVICE_PART 5
#define MKF_EV_DISPATCH_STALL 6
#define MKF_EV_WORKER_START_TASK 7
#define MKF_EV_WORKER_BUSY 8
#define MKF_EV_WORKER_IDLE 9
#define MKF_EV_WORKER_FINISH 10
#define MKF_EV_CANCELLED_FINISH 11
#define MKF_EV_LOCAL_L2_INC 12
#define MKF_EV_L2_FENCE 13
#define MKF_EV_GLOBAL_ATOMIC_INC 14
#define MKF_EV_EVENT_FORCE 15
#define MKF_EV_TASK_CANCEL 16
#define MKF_EV_INVALID 17

// Counter ordinals (cumulative persistent totals).
#define MKF_C_TASK_READY 0
#define MKF_C_TASK_WAIT 1
#define MKF_C_SCHED_IDLE 2
#define MKF_C_DISPATCH_SINGLE 3
#define MKF_C_DISPATCH_CHIPLET_PART 4
#define MKF_C_DISPATCH_DEVICE_PART 5
#define MKF_C_DISPATCH_STALL 6
#define MKF_C_WORKER_START_TASK 7
#define MKF_C_WORKER_BUSY 8
#define MKF_C_WORKER_IDLE 9
#define MKF_C_WORKER_FINISH 10
#define MKF_C_CANCELLED_FINISH 11
#define MKF_C_LOCAL_L2_INC 12
#define MKF_C_L2_FENCE 13
#define MKF_C_GLOBAL_ATOMIC_INC 14
#define MKF_C_EVENT_FORCE 15
#define MKF_C_TASK_CANCEL 16
#define MKF_C_INVALID 17
#define MKF_COUNTER_COUNT 18

/*
CONTRACT: mk_fleet_two_level  (T64).  This contract is fully self-contained:
every counter, sequence, iteration order, and hash byte layout needed to
reproduce the graded outputs bit-for-bit is specified below. Nothing external
needs to be consulted.

GEOMETRY / IDENTITY
  total_workers = chiplet_count * workers_per_chiplet.
  worker_id = chiplet * workers_per_chiplet + local_worker.
  worker_mask is a u64: bit i set => global worker i participates.
  A chiplet "participates" in a DEVICE task iff ANY of its workers' bits are set.

PERSISTENT SCALARS
  clock=0, event_seq=0, task_seq_next=1, dispatch_seq_next=1,
  local_count_seq_next=1, global_count_seq_next=1, ready_seq_next=1,
  last_local_seq=0, last_global_seq=0.

OPERATIONS (run.op):
  SUBMIT_TASK(task_id=a0, scope=a1, home_chiplet=a2, worker_mask=mask,
              duration=a3, payload_seed=seed, output_event=oe, output_increment=oinc,
              wait_count=wc, waits[k]=(wait_ev[k], wait_target[k]))
    Invalid (single INVALID, invalid_count++) if: task_id already used; task table
    full; home_chiplet out of range; scope out of range; output_event out of range;
    any wait event out of range; worker_mask==0.
    Else assign task_seq=task_seq_next++. Selected workers:
      WAVE/CU/CHIPLET: home-chiplet workers whose global bit is set (bits outside
        home chiplet ignored). WAVE/CU eventually pick exactly one (round robin at
        dispatch); CHIPLET runs all of them.
      DEVICE: every chiplet with any set bit participates; each participating
        chiplet-part runs ALL workers of that chiplet.
    expected_local_parts / expected_global_parts:
      WAVE/CU: local=0, global=1.
      CHIPLET: local=selected_home_worker_count, global=1.
      DEVICE: local=participating_chiplets * workers_per_chiplet,
              global=participating_chiplets.
    If all waits satisfied (global_event_counter[ev] >= target for each wait):
      status READY; enqueue to home chiplet ready queue (WAVE/CU/CHIPLET) or one
      descriptor per participating chiplet (DEVICE); emit one TASK_READY.
    Else status WAITING; emit TASK_WAIT.

  SCHED_STEP(chiplet=a0, dispatch_limit=a1)
    Invalid if chiplet out of range or dispatch_limit==0.
    Up to dispatch_limit dispatches; each iteration:
      Pop stale ready entries from the front (no event). A WAVE/CU/CHIPLET entry is
      stale if its task is missing/CANCELLED or status != READY. A DEVICE entry is
      stale if missing/CANCELLED or that chiplet's part already dispatched.
      If the queue is now empty: SCHED_IDLE, stop.
      WAVE/CU: round-robin among home-chiplet workers in the selected set with mailbox
        capacity (mailbox size < max_worker_queue). None -> DISPATCH_STALL, stop.
        Append copy, advance cursor to next local worker, status DISPATCHED,
        DISPATCH_SINGLE, pop entry.
      CHIPLET: all selected workers must have capacity else DISPATCH_STALL, stop. Init
        local_counter[chiplet][task]=0, local_target=selected count. Append one copy
        per selected worker ascending; emit DISPATCH_CHIPLET_PART each. Status
        DISPATCHED. Pop entry.
      DEVICE: all workers of the chiplet must have capacity else DISPATCH_STALL, stop.
        Init this chiplet's local_counter=0, target=workers_per_chiplet. Append one
        copy per worker ascending; emit DISPATCH_DEVICE_PART each. Mark the chiplet's
        part dispatched. Status DISPATCHED on first part. Pop entry.
      Each successfully dispatched descriptor counts ONE toward dispatch_limit.

  WORKER_START(worker=a0, limit=a1)
    Invalid if worker out of range or limit==0.
    Up to limit: if worker running -> WORKER_BUSY, stop. Drop CANCELLED-task copies
    from the front (no event). If mailbox empty -> WORKER_IDLE, stop. Start head copy:
    running due=clock+duration; emit WORKER_START_TASK. WAVE/CU set task status
    RUNNING. Counts one toward limit.

  ADVANCE(delta=a0, max_finishes=a1)
    delta==0 valid. clock += delta. Finish up to max_finishes running copies with
    due_clock<=clock, ordered by (due_clock, chiplet, worker, task_id). Each copy:
      emit WORKER_FINISH; free the worker.
      If task CANCELLED: emit CANCELLED_FINISH, no counters/increments.
      Else WAVE/CU: GLOBAL_ATOMIC_INC of output_increment to output_event,
        last_global_seq=global_count_seq_next++; ready-scan; task GLOBAL_DONE;
        compute result_hash.
      Else CHIPLET: LOCAL_L2_INC (last_local_seq=local_count_seq_next++). If local
        counter reaches target and fence not yet issued for this chiplet/task:
        L2_FENCE; GLOBAL_ATOMIC_INC; ready-scan; task GLOBAL_DONE; result_hash.
      Else DEVICE: LOCAL_L2_INC. When this chiplet's local counter reaches target:
        L2_FENCE; GLOBAL_ATOMIC_INC; ready-scan; ++global_parts_done; task LOCAL_DONE
        if not all parts done else GLOBAL_DONE + result_hash.

  EVENT_FORCE(event_id=a0, amount=a1)
    Invalid if event out of range. Else global_event_counter[ev]+=amount,
    last_global_seq=global_count_seq_next++; emit EVENT_FORCE; ready-scan.

  CANCEL_TASK(task_id=a0)
    Invalid if absent or terminal (GLOBAL_DONE/CANCELLED). Else status CANCELLED;
    mailbox/ready entries lazily stale; running copies still finish (CANCELLED_FINISH);
    emit TASK_CANCEL.

READY-SCAN: after any global counter increment, scan WAITING tasks by task_seq asc;
  any whose waits are all satisfied becomes READY and is enqueued per scope, emitting
  one TASK_READY each.

HASH PRIMITIVE (applies to EVERY hash below)
  FNV-1a-64 with a PROJECT-SPECIFIC offset basis (NOT the canonical basis):
    basis  = 1469598103934665603  (== 0x14650FB0739D0383)
    prime  = 1099511628211        (== 0x00000100000001B3)
  Each field is appended by folding its raw little-endian bytes, one byte at a
  time, in increasing memory-address order (LSB first for multi-byte integers):
    for each byte b of the field (width = its declared u8/u32/u64):
      h ^= (uint64_t)b;  h *= prime;     // u32 folds 4 bytes, u64 folds 8 bytes
  Every hash stream is SEEDED with `basis` and then folds its fields in the
  exact declared order. There is no length/terminator byte and no final mixing.
  Integer widths below are EXACT: u8 = 1 byte, u32 = 4 bytes, u64 = 8 bytes,
  i64 fields are folded as their 8-byte little-endian two's-complement image.
  VALUE NOTE: task_id, duration, payload_seed, output_increment, and EVENT_FORCE
  amount are taken as (uint64_t)(uint32_t)<int32 operand> (zero-extended 32-bit)
  before being stored/hashed. All counters/sequences wrap mod 2^64.

STORED TASK FIELD WIDTHS (for hashing): task_id:u64, task_seq:u64, scope:u8,
  status:u8, home_chiplet:u32, worker_mask:u64, duration:u64, output_event:u32,
  output_increment:u64, payload_seed:u64, expected_local_parts:u32,
  expected_global_parts:u32, ready_seq_or_zero:u64, result_hash:u64.

result_hash: 0 until the task reaches GLOBAL_DONE; at that completion it is a
  fresh FNV stream (seeded with basis) folding, IN THIS ORDER:
    task_id:u64, payload_seed:u64, duration:u64, output_event:u32,
    output_increment:u64, clock:u64        (clock = persistent clock at finish).

OUTPUTS (per run, after the op):
  Counters[18] cumulative i64, written in MKF_C_* ordinal order (index 0..17).
  clock_out, event_seq_out: the persistent clock and event_seq scalars (u64).

  fleet_event_hash: a fresh FNV stream (seeded with basis) over THIS run's events
    in emission order. Each event folds, IN THIS ORDER:
      event_kind:u8, event_seq:u64, op_index:u32, clock:u64,
      chiplet_or_UINT32_MAX:u32, worker_or_UINT32_MAX:u32,
      task_id_or_ZERO:u64, event_id_or_UINT32_MAX:u32, value_or_UINT64_MAX:u64.
    Here event_seq is the value BEFORE this event (post-increment: emit then
    event_seq++); clock is the persistent clock at emission time; absent fields
    use the sentinel shown (UINT32_MAX / UINT64_MAX; task uses 0 when absent).

  task_state_hash: a fresh FNV stream (seeded with basis). Iterate USED tasks by
    task_id ASCENDING. For each task fold, IN THIS ORDER:
      task_id:u64, task_seq:u64, scope:u8, home_chiplet:u32, worker_mask:u64,
      duration:u64, status:u8, expected_local_parts:u32,
      expected_global_parts:u32, ready_seq_or_zero:u64, result_hash:u64.

  queue_hash: a fresh FNV stream (seeded with basis). FIRST the ready queues, for
    chiplet ch = 0..chiplet_count-1 ascending:
      fold kind=0:u8, then ch:u32; then for each ready entry in FIFO order
      (position pos = 0..ready_count[ch]-1, head-relative) fold:
        pos:u64, ready.task_id:u64, ready.ready_seq:u64, ready.chiplet:u32.
    THEN the worker mailboxes, for worker w = 0..total_workers-1 ascending:
      fold kind=1:u8, then w:u32; then for each mailbox copy in FIFO order
      (pos = 0..mbox_count[w]-1, head-relative) fold:
        pos:u64, copy.task_id:u64.

  counter_hash: a fresh FNV stream (seeded with basis). FIRST the L2-local
    counters, for chiplet ch = 0..chiplet_count-1 ascending, and within each
    chiplet over its USED local entries (local_used set) by task_id ASCENDING,
    fold IN THIS ORDER:
      kind=0:u8, ch:u32, task_id:u64, local_counter:u64(from i64),
      local_target:u64(from i64), last_local_seq:u64.
    THEN the global event counters, for event e = 0..max_events-1 ascending:
      kind=1:u8, e:u32, gcounter[e]:u64(from i64), last_global_seq:u64.
    last_local_seq / last_global_seq are the persistent scalars (the seq value
    assigned by the MOST RECENT local / global increment; 0 if none yet) and are
    appended to EVERY local / global entry respectively.

  running_hash: a fresh FNV stream (seeded with basis). Iterate running copies by
    the tuple (due_clock, chiplet, worker, task_id) ASCENDING. For each fold,
    IN THIS ORDER:
      due_clock:u64, chiplet:u32, worker:u32, task_id:u64, scope:u8.

DETERMINISM: scope decides dispatch width and completion protocol. CHIPLET/DEVICE
  use L2 local counters; only the last local worker issues L2_FENCE + GLOBAL_ATOMIC_INC.
  WAVE/CU bypass local counting. Scheduler is FIFO by ready_seq (then task_id). Single
  worker choice is round-robin with capacity skipping. DEVICE completes only when all
  participating chiplets have globally incremented. Cancelled running copies finish
  without counters. All counters/sequences wrap mod 2^64.
*/

struct alignas(8) MkfProblemSpec {
    int32_t abi_version;
    int32_t chiplet_count;
    int32_t workers_per_chiplet;
    int32_t max_tasks;
    int32_t max_events;
    int32_t max_ready_per_chiplet;
    int32_t max_worker_queue;
    int32_t max_running;
    int32_t device_task_chiplet_count;  // informational bound
    int32_t reserved[7];
};

struct alignas(8) MkfRunSpec {
    int32_t abi_version;
    int32_t op;          // MKF_OP_*
    int32_t op_index;    // per-run identifier folded into event hash
    int32_t a0;
    int32_t a1;
    int32_t a2;
    int32_t a3;
    // SUBMIT-only extended operands:
    uint64_t worker_mask;
    int32_t payload_seed;
    int32_t output_event;
    int32_t output_increment;
    int32_t wait_count;          // 0..MKF_MAX_WAITS
    int32_t wait_ev[MKF_MAX_WAITS];
    int32_t wait_target[MKF_MAX_WAITS];
    int32_t reserved[2];
};

struct alignas(8) MkfInputs {
    const int32_t* reserved0;
    const int32_t* reserved1;
};

struct alignas(8) MkfOutputs {
    int64_t* counters;          // length MKF_COUNTER_COUNT
    uint64_t* fleet_event_hash;
    uint64_t* task_state_hash;
    uint64_t* queue_hash;
    uint64_t* counter_hash;
    uint64_t* running_hash;
    uint64_t* clock_out;
    uint64_t* event_seq_out;
};

static_assert(sizeof(MkfProblemSpec) == 64, "MkfProblemSpec layout drift");
static_assert(sizeof(MkfRunSpec) == 80, "MkfRunSpec layout drift");
static_assert(sizeof(MkfInputs) == 16, "MkfInputs layout drift");
static_assert(sizeof(MkfOutputs) == 64, "MkfOutputs layout drift");

static inline int mkf_validate_problem_spec(const MkfProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MKF_ABI_VERSION) return 0;
    if (spec->chiplet_count < MKF_MIN_CHIPLETS || spec->chiplet_count > MKF_MAX_CHIPLETS) return 0;
    if (spec->workers_per_chiplet < MKF_MIN_WPC || spec->workers_per_chiplet > MKF_MAX_WPC) return 0;
    if (spec->chiplet_count * spec->workers_per_chiplet > MKF_MAX_WORKERS) return 0;
    if (spec->max_tasks < MKF_MIN_TASKS || spec->max_tasks > MKF_MAX_TASKS) return 0;
    if (spec->max_events < MKF_MIN_EVENTS || spec->max_events > MKF_MAX_EVENTS) return 0;
    if (spec->max_ready_per_chiplet < MKF_MIN_READY ||
        spec->max_ready_per_chiplet > MKF_MAX_READY) return 0;
    if (spec->max_worker_queue < MKF_MIN_WQ || spec->max_worker_queue > MKF_MAX_WQ) return 0;
    if (spec->max_running < MKF_MIN_RUNNING || spec->max_running > MKF_MAX_RUNNING) return 0;
    if (spec->device_task_chiplet_count < 0 ||
        spec->device_task_chiplet_count > spec->chiplet_count) return 0;
    return 1;
}

static inline int mkf_validate_run_spec(const MkfRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != MKF_ABI_VERSION) return 0;
    if (run->op < MKF_OP_SUBMIT || run->op > MKF_OP_CANCEL) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MkfProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MkfProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MkfRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MK_FLEET_TWO_LEVEL_COMMON_H_
