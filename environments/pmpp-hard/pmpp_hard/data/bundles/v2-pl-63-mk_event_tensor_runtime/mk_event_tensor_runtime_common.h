// file: mk_event_tensor_runtime_common.h

#ifndef MK_EVENT_TENSOR_RUNTIME_COMMON_H_
#define MK_EVENT_TENSOR_RUNTIME_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MK_ABI_VERSION 1

// Capacity bounds for the problem spec.
#define MK_MIN_TENSORS 1
#define MK_MAX_TENSORS 256
#define MK_MIN_CELLS_PER_TENSOR 1
#define MK_MAX_CELLS_PER_TENSOR 4096
#define MK_MIN_TASKS 1
#define MK_MAX_TASKS 1024
#define MK_MIN_TASK_DEPS 1
#define MK_MAX_TASK_DEPS 64
#define MK_MIN_TASK_OUTPUTS 1
#define MK_MAX_TASK_OUTPUTS 64
#define MK_MIN_SCHEDULERS 1
#define MK_MAX_SCHEDULERS 64
#define MK_MIN_READY_PER_SCHEDULER 1
#define MK_MAX_READY_PER_SCHEDULER 1024
#define MK_MIN_POPPED_PER_SCHEDULER 1
#define MK_MAX_POPPED_PER_SCHEDULER 1024
#define MK_MAX_BATCH 8192
#define MK_MAX_STEPS 64
#define MK_MAX_DEP_BUF 65536
#define MK_MAX_OUT_BUF 65536

// Op kinds (op_kind field of each operation row).
#define MK_OP_DEFINE_TENSOR 0
#define MK_OP_REGISTER_TASK 1
#define MK_OP_SIGNAL_CELL   2
#define MK_OP_POP_TASKS     3
#define MK_OP_START_POPPED  4
#define MK_OP_COMPLETE_TASK 5
#define MK_OP_CANCEL_CELL   6
#define MK_OP_CANCEL_TASK   7

// Task status values.
#define MK_ST_BLOCKED   0
#define MK_ST_READY     1
#define MK_ST_POPPED    2
#define MK_ST_RUNNING   3
#define MK_ST_DONE      4
#define MK_ST_CANCELLED 5

// Event kinds for mk_event_hash (emission order).
#define MK_EVT_TENSOR_DEFINE         0
#define MK_EVT_DEFINE_STALL          1
#define MK_EVT_TASK_READY_REGISTER   2
#define MK_EVT_TASK_BLOCKED_REGISTER 3
#define MK_EVT_CELL_DECREMENT        4
#define MK_EVT_OUTPUT_DECREMENT      5
#define MK_EVT_CELL_ZERO             6
#define MK_EVT_CELL_ALREADY_ZERO     7
#define MK_EVT_TASK_READY_PUSH       8
#define MK_EVT_READY_STALE_DROP      9
#define MK_EVT_TASK_POP              10
#define MK_EVT_TASK_START            11
#define MK_EVT_POPPED_STALE_DROP     12
#define MK_EVT_TASK_COMPLETE         13
#define MK_EVT_TASK_DYNAMIC_CANCEL   14
#define MK_EVT_CELL_CANCEL           15
#define MK_EVT_TASK_CANCEL_BY_CELL   16
#define MK_EVT_TASK_CANCEL_EXPLICIT  17
#define MK_EVT_INVALID               18

#define MK_FNV_OFFSET 1469598103934665603ULL
#define MK_FNV_PRIME  1099511628211ULL
#define MK_U32_MAX    0xFFFFFFFFu
#define MK_U64_MAX    0xFFFFFFFFFFFFFFFFULL

/*
CONTRACT: mk_event_tensor_runtime  (MK4)

T63 -- Event-Tensor Pop/Push Runtime with Zero-Decrement Unblocking.

A persistent dynamic-megakernel runtime grounded in the Event Tensor Compiler.
First-class event-tensor cells hold dependency counts. Completing tasks decrement
cells; a cell reaching zero for the first time pushes its dependent tasks into
scheduler ready queues. Idle SMs pop ready tasks; started tasks run and complete,
producing output decrements that may zero further cells. The whole event/dispatch
timeline is graded via exact-integer counts and FNV-1a-64 checksums.

PERSISTENT STATE (after init/reset):
  event_seq = 0; task_seq_next = 1; cell_seq_next = 1; ready_seq_next = 1;
  pop_seq_next = 1; op_index = 0 (GLOBAL op counter since reset, used in hashes).

  Tensor table keyed by tensor_id in [0, tensor_count):
    defined:u8; rank:u8; dim0:u32; dim1:u32; generation:u64.
  A tensor's cell key is (tensor_id, i, j), j=0 for rank 1. The cell's linear
  index inside the tensor is i*max(dim1,1)+j. A cell "exists" iff its tensor is
  defined and (i,j) is in range for that tensor's dims.

  Cell state (per existing cell):
    initial_count:u64; remaining_count:u64; zero_pushed:u8; cell_seq:u64;
    last_decrement_seq:u64; cancelled:u8.

  Task table keyed by external task_id (u64):
    task_id; task_seq:u64; scheduler_hint:u32; status in {BLOCKED, READY, POPPED,
    RUNNING, DONE, CANCELLED}; dependency cells (listed order); output cells with
    decrement amounts (listed order); dynamic_mask:u64; payload_seed:u64;
    ready_seq_or_ZERO:u64; pop_seq_or_ZERO:u64.

  Reverse dependency index: for each cell, the task ids depending on it, ordered
  by task_seq.

  Ready queues per scheduler: task ids ordered by ready_seq (append order).
  Popped lists per scheduler: task ids ordered by pop_seq (append order).

OP STREAM:
  Each solution_run processes batch_size operations in order. op_index is a GLOBAL
  operation index since reset, starting at 0, used in event records. event_seq is
  a global monotone counter incremented exactly once per EMITTED event (every
  mk_event_hash record consumes one event_seq, including INVALID). All sequence
  counters wrap modulo 2^64.

OPERATIONS (full normative semantics; this contract is SELF-CONTAINED and is the
authority for grading. Counter-increment timing is exact: each *_seq value placed
into state/events is the counter's value BEFORE its post-increment, i.e.
`v = counter; counter = v + 1;`. op_index for an op's events is read ONCE at the
start of the op (= the value of the global op counter) and that same op_index is
used by every event the op emits; the op counter is incremented by 1 AFTER the op
finishes. event_seq likewise: each emitted event uses the current event_seq, then
event_seq is incremented by 1):

  DEFINE_TENSOR(tensor_id, rank, dim0, dim1, initial_count): invalid if tensor_id
    out of range, rank not 1 or 2, dim0==0, (rank==2 && dim1==0), or
    dim0*max(dim1,1) > max_cells_per_tensor. If the tensor is currently defined and
    ANY cell of it has a dependent task whose status is nonterminal (BLOCKED, READY,
    POPPED or RUNNING), emit DEFINE_STALL (no change). Otherwise define the tensor:
    increment its generation, create all cells with remaining_count=initial_count,
    zero_pushed=(initial_count==0), cell_seq=cell_seq_next++ (cells created in
    ascending linear index order), last_decrement_seq=0, cancelled=0. Redefining a
    tensor discards its old cells and its reverse-dep index entries (the old cells
    no longer exist). Emit TENSOR_DEFINE.

  REGISTER_TASK(task_id, scheduler_hint, dep_count, deps[], output_count,
    outputs[], dynamic_mask, payload_seed): invalid if the task already exists, the
    task table is full, scheduler_hint >= scheduler_count, dep_count > max_task_deps,
    output_count > max_task_outputs, or ANY dep cell or output cell is absent (its
    tensor undefined / index out of range) or is cancelled. Otherwise create the
    task with task_seq=task_seq_next++; record deps and outputs in listed order;
    insert the task into the reverse-dep index of every dependency cell (each cell's
    list stays ordered by task_seq). If ALL dependency cells have remaining_count==0
    and not cancelled (a task with zero deps trivially qualifies), set status READY,
    assign ready_seq=ready_seq_next++, append to scheduler scheduler_hint's ready
    queue, emit TASK_READY_REGISTER. Otherwise set status BLOCKED, emit
    TASK_BLOCKED_REGISTER.

  SIGNAL_CELL(tensor_id, i, j, amount): invalid if the cell is absent or cancelled,
    or amount==0. Decrement: if amount>=remaining_count set remaining_count=0, else
    subtract. Emit CELL_DECREMENT and set last_decrement_seq to that event's
    event_seq. If the cell becomes zero for the FIRST time (remaining_count just
    reached 0 and zero_pushed was 0): set zero_pushed=1, emit CELL_ZERO, then scan
    dependent tasks in task_seq order; for each task whose ALL deps are now zero and
    status==BLOCKED, set READY, assign ready_seq=ready_seq_next++, append to its
    scheduler queue, emit TASK_READY_PUSH. If the cell was ALREADY zero (zero_pushed
    was 1), emit CELL_ALREADY_ZERO and do NOT scan dependents.

  POP_TASKS(scheduler, limit): invalid if scheduler >= scheduler_count. limit==0 is
    a valid no-op. Pop up to limit VALID tasks from the front of the scheduler's
    ready queue. A queue entry is dropped (front removed, emit READY_STALE_DROP, does
    NOT count toward limit) when its task is absent, cancelled, or not status READY,
    or its ready_seq does not match the entry's ready_seq (stale). For each valid
    task: set status POPPED, assign pop_seq=pop_seq_next++, append to the scheduler's
    popped list, remove from the ready queue front, emit TASK_POP. Stop when limit
    valid tasks popped or the queue empties.

  START_POPPED(scheduler, limit): invalid if scheduler >= scheduler_count. Start up
    to limit popped tasks in popped-list order from the front. A popped entry is
    dropped (front removed, emit POPPED_STALE_DROP, does NOT count toward limit) when
    its task is absent, not status POPPED, or its pop_seq does not match. For each
    task still POPPED: set status RUNNING, remove from the popped list front, emit
    TASK_START. Stop when limit started or the list empties. limit==0 is a valid
    no-op.

  COMPLETE_TASK(task_id, observed_mask): invalid if the task is absent or status is
    not RUNNING. If (observed_mask & dynamic_mask) != dynamic_mask: cancel the task
    without outputs, set status CANCELLED, emit TASK_DYNAMIC_CANCEL, stop. Otherwise
    set status DONE, emit TASK_COMPLETE, then for each output cell in LISTED order
    apply the exact SIGNAL_CELL decrement semantics (including zero-decrement
    unblocking, already-zero handling, and last_decrement_seq), except the decrement
    event kind is OUTPUT_DECREMENT instead of CELL_DECREMENT. An output whose cell is
    absent or cancelled, or whose amount==0, is skipped silently (no event); the
    CELL_ZERO / TASK_READY_PUSH / CELL_ALREADY_ZERO events still occur for outputs
    that do decrement.

  CANCEL_CELL(tensor_id, i, j): invalid if the cell is absent. Set the cell's
    cancelled=1, emit CELL_CANCEL. Then for every task depending on that cell whose
    status is BLOCKED or READY, in task_seq order, set status CANCELLED, emit
    TASK_CANCEL_BY_CELL. Tasks already POPPED or RUNNING (or terminal) are not
    cancelled by cell cancellation.

  CANCEL_TASK(task_id): invalid if the task is absent or terminal (DONE or
    CANCELLED). Set status CANCELLED, emit TASK_CANCEL_EXPLICIT. The task remains as
    a stale entry in any ready/popped queue until lazily skipped.

DETERMINISM:
  - A cell reaching zero scans dependents immediately, in task_seq order.
  - A cell already at zero does not re-push dependents.
  - COMPLETE_TASK emits TASK_COMPLETE before any output decrements.
  - Dynamic-mask cancellation prevents all output decrements.
  - Ready/popped queues drop stale entries lazily (front-only) and the queue hash
    reflects the remaining queue after the step.
  - Cell cancellation affects BLOCKED and READY tasks, not POPPED or RUNNING.
  - Tensor redefinition is blocked (DEFINE_STALL) while any dependent nonterminal
    task exists.
  - All sequence counters wrap modulo 2^64; unsigned numeric ordering everywhere.

INVALID ops emit exactly one INVALID event (event_seq consumed), increment
invalid_count, and produce no other events.

OUTPUT COUNTS (per step batch, reset to 0 each step):
  tensors_defined; define_stall; tasks_ready_register; tasks_blocked_register;
  cell_decrements; cell_zero; cell_already_zero; tasks_ready_push; tasks_popped;
  tasks_started; tasks_completed; dynamic_cancelled; cell_cancelled;
  tasks_cancel_by_cell; tasks_cancel_explicit; ready_stale_drop; popped_stale_drop;
  invalid_count.
  (cell_decrements counts BOTH CELL_DECREMENT and OUTPUT_DECREMENT events.)

FNV-1a-64 PRIMITIVE (exact; used by ALL five hashes):
  h starts at the offset basis OFFSET = 1469598103934665603 (0x14650FB0739D0383)
  -- the PROJECT basis, NOT the canonical 0xCBF29CE484222325.
  PRIME = 1099511628211 (0x100000001B3).
  For each byte b folded:  h = (h XOR b) * PRIME, all modulo 2^64.
  A field of width W (u8=1, u32=4, u64=8 bytes) is folded by appending its W raw
  bytes in LITTLE-ENDIAN order (least-significant byte first), each byte folded by
  the rule above, in field order. There are no separators, no padding, no length
  prefix. A field declared u32/u64 is ALWAYS folded at its full width even when it
  carries a sentinel (U32MAX/U64MAX) or 0. Field order within each record is exactly
  as listed below; records are folded back-to-back into the same running h.

OUTPUT HASHES (all FNV-1a-64, offset basis 1469598103934665603):
  mk_event_hash: all events in emission order. Per-event fields:
    event_kind:u8; event_seq:u64; op_index:u32; tensor_or_U32MAX:u32;
    i_or_U32MAX:u32; j_or_U32MAX:u32; task_id_or_ZERO:u64;
    scheduler_or_U32MAX:u32; count_or_U64MAX:u64.
  cell_hash: cells by tensor_id asc, then i asc, then j asc (only existing cells):
    tensor_id:u32; generation:u64; i:u32; j:u32; initial_count:u64;
    remaining_count:u64; zero_pushed:u8; cell_seq:u64; last_decrement_seq:u64;
    cancelled:u8.
  task_hash: tasks by task_id asc:
    task_id:u64; task_seq:u64; scheduler_hint:u32; status:u8; dynamic_mask:u64;
    payload_seed:u64; ready_seq_or_ZERO:u64; pop_seq_or_ZERO:u64; then deps in
    listed order each (tensor:u32; i:u32; j:u32); then outputs in listed order each
    (tensor:u32; i:u32; j:u32; amount:u64).
  queue_hash: ready queues (scheduler asc, then queue order), then popped lists
    (scheduler asc, then list order):
    Ready: queue_kind:u8=0; scheduler:u32; position:u64; task_id:u64; ready_seq:u64.
    Popped: queue_kind:u8=1; scheduler:u32; position:u64; task_id:u64; pop_seq:u64.
  reverse_dep_hash: each existing cell (tensor asc, i asc, j asc) that has any
    dependents: tensor_id:u32; i:u32; j:u32; then dependent task ids by task_seq:
    task_id:u64.

mk_event_hash is per-step (re-initialized to the offset basis each step).
cell_hash/task_hash/queue_hash/reverse_dep_hash reflect the full persistent state
AFTER the step. All outputs are exact integers.

EXACT EVENT RECORD FIELDS, per event kind (every event folds the SAME nine fields
in this order: event_kind:u8; event_seq:u64; op_index:u32; tensor:u32; i:u32; j:u32;
task_id:u64; scheduler:u32; count:u64. event_seq is the per-event monotone value;
op_index is the op's global index; the other seven are listed below; U32=U32MAX,
U64=U64MAX):
  kind                       tensor   i      j      task_id        sched      count
  TENSOR_DEFINE(0)           t        U32    U32    0              U32        generation(after ++)
  DEFINE_STALL(1)            t        U32    U32    0              U32        U64
  TASK_READY_REGISTER(2)     U32      U32    U32    task_id        sched_hint ready_seq
  TASK_BLOCKED_REGISTER(3)   U32      U32    U32    task_id        sched_hint U64
  CELL_DECREMENT(4)          t        i      j      0              U32        remaining(after dec)
  OUTPUT_DECREMENT(5)        t        i      j      owning task_id U32        remaining(after dec)
  CELL_ZERO(6)               t        i      j      0              U32        0
  CELL_ALREADY_ZERO(7)       t        i      j      0              U32        0
  TASK_READY_PUSH(8)         U32      U32    U32    task_id        sched      ready_seq
  READY_STALE_DROP(9)        U32      U32    U32    entry task_id  scheduler  entry ready_seq
  TASK_POP(10)               U32      U32    U32    task_id        scheduler  pop_seq
  TASK_START(11)             U32      U32    U32    task_id        scheduler  pop_seq
  POPPED_STALE_DROP(12)      U32      U32    U32    entry task_id  scheduler  entry pop_seq
  TASK_COMPLETE(13)          U32      U32    U32    task_id        U32        U64
  TASK_DYNAMIC_CANCEL(14)    U32      U32    U32    task_id        U32        observed_mask
  CELL_CANCEL(15)            t        i      j      0              U32        U64
  TASK_CANCEL_BY_CELL(16)    t        i      j      task_id        U32        U64
  TASK_CANCEL_EXPLICIT(17)   U32      U32    U32    task_id        U32        U64
  INVALID(18)                U32      U32    U32    inv_task_id    U32        U64
  Notes: TASK_READY_PUSH's scheduler field is the pushed task's own scheduler_hint.
  For TASK_READY_REGISTER/TASK_READY_PUSH/TASK_POP/TASK_START the count is the value
  just assigned/held by that task (ready_seq or pop_seq). For *_STALE_DROP the task_id
  and seq are taken from the QUEUE ENTRY being dropped, not from a live task lookup.

INVALID handling: an INVALID op emits exactly ONE INVALID event (kind 18, consuming
one event_seq), increments invalid_count, and produces no other events or state
change. inv_task_id (the task_id field of that event) is the op's task_id argument
for REGISTER_TASK / COMPLETE_TASK / CANCEL_TASK and for any UNKNOWN op_kind
(out of 0..7), and is 0 for DEFINE_TENSOR / SIGNAL_CELL / POP_TASKS / START_POPPED /
CANCEL_CELL.

DECREMENT / ZERO-PUSH ORDERING (shared by SIGNAL_CELL and each COMPLETE_TASK
output, applied to an existing non-cancelled cell with amount>0):
  1. was_zero = (remaining_count == 0 before this decrement).
  2. remaining_count = (amount >= remaining_count) ? 0 : remaining_count - amount.
  3. last_decrement_seq = the event_seq that the decrement event will consume.
  4. emit the decrement event (CELL_DECREMENT for SIGNAL_CELL, OUTPUT_DECREMENT for
     COMPLETE_TASK outputs) with count = new remaining_count; increment cell_decrements.
  5. if (remaining_count == 0 && !was_zero && zero_pushed == 0): set zero_pushed = 1;
     emit CELL_ZERO (increment cell_zero); then SCAN dependents (see below).
     else if (was_zero): emit CELL_ALREADY_ZERO (increment cell_already_zero); NO scan.
     (A cell that becomes zero on this very decrement but already had zero_pushed==1
     -- only reachable via redefinition edge cases -- takes neither branch.)
  ZERO-PUSH SCAN: iterate ALL existing tasks in ascending task_seq order; for each
  task that (a) depends on this exact cell (t,i,j), (b) has status == BLOCKED, and
  (c) has ALL its dependency cells currently existing, non-cancelled, and
  remaining_count == 0: set status READY, ready_seq = ready_seq_next++ (assign then
  ++), pop_seq = 0, append (task_id, ready_seq) to scheduler_hint's ready queue tail,
  emit TASK_READY_PUSH, increment tasks_ready_push. Tasks are promoted in task_seq
  order so their ready_seq values are assigned in that order.
  COMPLETE_TASK applies its outputs in LISTED order; an output whose cell is absent,
  cancelled, or whose amount==0 is skipped silently (NO event, no count change); the
  decrement/zero/already-zero events still fire for outputs that do decrement.

ENCODING of operation rows (Structure-of-Arrays; each array has batch_size rows):
  op_kind   : MK_OP_*.
  tensor_id : tensor id (DEFINE_TENSOR/SIGNAL_CELL/CANCEL_CELL); else 0.
  ci, cj    : cell coords i, j (SIGNAL_CELL/CANCEL_CELL); else 0.
  rank      : rank (DEFINE_TENSOR); else 0.
  dim0,dim1 : dims (DEFINE_TENSOR); else 0.
  task_id   : external task id (REGISTER_TASK/COMPLETE_TASK/CANCEL_TASK); 0 else.
  sched     : scheduler_hint (REGISTER_TASK); scheduler (POP_TASKS/START_POPPED).
  amount    : initial_count (DEFINE_TENSOR); decrement amount (SIGNAL_CELL);
              limit (POP_TASKS/START_POPPED); else 0.
  mask      : dynamic_mask (REGISTER_TASK); observed_mask (COMPLETE_TASK); else 0.
  payload   : payload_seed (REGISTER_TASK); else 0.
  dep_off, dep_count   : window into dep_tensor/dep_i/dep_j for REGISTER_TASK deps.
  out_off, out_count   : window into out_tensor/out_i/out_j/out_amount for
                         REGISTER_TASK outputs.
  dep_tensor[], dep_i[], dep_j[]            : flat dependency cell coordinates.
  out_tensor[], out_i[], out_j[], out_amount[] : flat output cells + amounts.

Rules:
  - solution_init may allocate persistent state.
  - solution_run may not call cudaMalloc/cudaFree.
  - solution_run may launch kernels and use provided workspace.
*/

struct alignas(8) MkProblemSpec {
    int32_t abi_version;
    int32_t tensor_count;
    int32_t max_cells_per_tensor;
    int32_t max_tasks;
    int32_t max_task_deps;
    int32_t max_task_outputs;
    int32_t scheduler_count;
    int32_t max_ready_per_scheduler;
    int32_t max_popped_per_scheduler;
    int32_t max_batch;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[4];
};

struct alignas(8) MkRunSpec {
    int32_t abi_version;
    int32_t batch_size;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) MkInputs {
    const int32_t*  op_kind;
    const uint32_t* tensor_id;
    const uint32_t* ci;
    const uint32_t* cj;
    const uint32_t* rank;
    const uint32_t* dim0;
    const uint32_t* dim1;
    const uint64_t* task_id;
    const uint32_t* sched;
    const uint64_t* amount;
    const uint64_t* mask;
    const uint64_t* payload;
    const uint32_t* dep_off;
    const uint32_t* dep_count;
    const uint32_t* out_off;
    const uint32_t* out_count;
    const uint32_t* dep_tensor;
    const uint32_t* dep_i;
    const uint32_t* dep_j;
    const uint32_t* out_tensor;
    const uint32_t* out_i;
    const uint32_t* out_j;
    const uint64_t* out_amount;
};

struct alignas(8) MkOutputs {
    // Counts (18).
    int32_t* tensors_defined;
    int32_t* define_stall;
    int32_t* tasks_ready_register;
    int32_t* tasks_blocked_register;
    int32_t* cell_decrements;
    int32_t* cell_zero;
    int32_t* cell_already_zero;
    int32_t* tasks_ready_push;
    int32_t* tasks_popped;
    int32_t* tasks_started;
    int32_t* tasks_completed;
    int32_t* dynamic_cancelled;
    int32_t* cell_cancelled;
    int32_t* tasks_cancel_by_cell;
    int32_t* tasks_cancel_explicit;
    int32_t* ready_stale_drop;
    int32_t* popped_stale_drop;
    int32_t* invalid_count;
    // Hashes (5).
    uint64_t* mk_event_hash;
    uint64_t* cell_hash;
    uint64_t* task_hash;
    uint64_t* queue_hash;
    uint64_t* reverse_dep_hash;
};

static_assert(sizeof(MkProblemSpec) == 64, "MkProblemSpec layout drift");
static_assert(sizeof(MkRunSpec) == 64, "MkRunSpec layout drift");
static_assert(sizeof(MkInputs) == 23 * sizeof(void*), "MkInputs layout drift");
static_assert(sizeof(MkOutputs) == 23 * sizeof(void*), "MkOutputs layout drift");

static inline int mk_validate_problem_spec(const MkProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MK_ABI_VERSION) return 0;
    if (spec->tensor_count < MK_MIN_TENSORS || spec->tensor_count > MK_MAX_TENSORS) return 0;
    if (spec->max_cells_per_tensor < MK_MIN_CELLS_PER_TENSOR ||
        spec->max_cells_per_tensor > MK_MAX_CELLS_PER_TENSOR) return 0;
    if (spec->max_tasks < MK_MIN_TASKS || spec->max_tasks > MK_MAX_TASKS) return 0;
    if (spec->max_task_deps < MK_MIN_TASK_DEPS || spec->max_task_deps > MK_MAX_TASK_DEPS) return 0;
    if (spec->max_task_outputs < MK_MIN_TASK_OUTPUTS ||
        spec->max_task_outputs > MK_MAX_TASK_OUTPUTS) return 0;
    if (spec->scheduler_count < MK_MIN_SCHEDULERS ||
        spec->scheduler_count > MK_MAX_SCHEDULERS) return 0;
    if (spec->max_ready_per_scheduler < MK_MIN_READY_PER_SCHEDULER ||
        spec->max_ready_per_scheduler > MK_MAX_READY_PER_SCHEDULER) return 0;
    if (spec->max_popped_per_scheduler < MK_MIN_POPPED_PER_SCHEDULER ||
        spec->max_popped_per_scheduler > MK_MAX_POPPED_PER_SCHEDULER) return 0;
    if (spec->max_batch < 0 || spec->max_batch > MK_MAX_BATCH) return 0;
    if (spec->max_steps < 1 || spec->max_steps > MK_MAX_STEPS) return 0;
    return 1;
}

static inline int mk_validate_run_spec(const MkRunSpec* run, const MkProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != MK_ABI_VERSION) return 0;
    if (run->batch_size < 0 || run->batch_size > spec->max_batch) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MkProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MkProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MkRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MK_EVENT_TENSOR_RUNTIME_COMMON_H_
