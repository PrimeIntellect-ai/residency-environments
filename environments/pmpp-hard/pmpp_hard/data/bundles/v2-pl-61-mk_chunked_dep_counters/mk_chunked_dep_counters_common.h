// file: mk_chunked_dep_counters_common.h

#ifndef MK_CHUNKED_DEP_COUNTERS_COMMON_H_
#define MK_CHUNKED_DEP_COUNTERS_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MK_ABI_VERSION 1

// Limits (compile-time hard caps used by validation and fixed buffers).
#define MK_MIN_EDGES 1
#define MK_MAX_EDGES 32
#define MK_MIN_CHUNKS 1
#define MK_MAX_CHUNKS_PER_EDGE 32
#define MK_MAX_WAITERS 1024
#define MK_MAX_READY 1024
#define MK_MAX_STORE_EVENTS 1024
#define MK_MAX_CONSUMERS 1024
#define MK_MAX_EPOCH 255
#define MK_MAX_OPS 8192

// Operation opcodes (op.kind).
#define MK_OP_DEFINE_EDGE 0
#define MK_OP_RESET_EDGE 1
#define MK_OP_PRODUCE 2
#define MK_OP_ADVANCE 3
#define MK_OP_ARM_WAIT 4
#define MK_OP_CONSUME 5
#define MK_OP_CANCEL_WAIT 6
#define MK_OP_FORCE_COUNTER 7

// Event kinds (must match emission order positions in the contract).
#define MK_EV_EDGE_DEFINE 0
#define MK_EV_EDGE_RESET 1
#define MK_EV_RESET_STALL 2
#define MK_EV_STORE_ISSUE 3
#define MK_EV_STORE_COMPLETE 4
#define MK_EV_PRODUCE_STALL_CHUNK 5
#define MK_EV_STORE_STALE_DROP 6
#define MK_EV_WAITER_ARM 7
#define MK_EV_WAITER_READY 8
#define MK_EV_WAITER_READY_IMMEDIATE 9
#define MK_EV_READY_STALE_DROP 10
#define MK_EV_READY_REQUEUE 11
#define MK_EV_CONSUME_CHUNK 12
#define MK_EV_CHUNK_RELEASE 13
#define MK_EV_WAITER_CANCEL 14
#define MK_EV_COUNTER_FORCE 15
#define MK_EV_INVALID 16

// Page states.
#define MK_PAGE_EMPTY 0
#define MK_PAGE_STORING 1
#define MK_PAGE_READY 2
#define MK_PAGE_CONSUMING 3
#define MK_PAGE_RELEASED 4

// Waiter states.  WAITING/READY are nonterminal; CONSUMED/CANCELLED/REMOVED are
// terminal (excluded from waiter_hash).  REMOVED is an internal terminal state
// used for lazy stale removal (no event emitted), distinct from CANCELLED which
// is produced by CANCEL_WAIT (which does emit WAITER_CANCEL).
#define MK_WAIT_WAITING 0
#define MK_WAIT_READY 1
#define MK_WAIT_CONSUMED 2
#define MK_WAIT_CANCELLED 3
#define MK_WAIT_REMOVED 4

/*
CONTRACT: mk_chunked_dep_counters  (MK2 megakernel runtime)

A persistent fine-grained producer/consumer runtime where output chunks each
own an independent dependency counter, page (page_state), wait queue, and may
appear in a single global ready queue.  Consumers overlap on chunks instead of
waiting for whole tensors.  Each solution_run processes a batch of operations
against persistent state and emits exact-integer counts plus order-sensitive
FNV-1a-64 checksums over the event stream and the structural state (cells,
waiters, ready queue, pending stores).

This contract is fully self-contained: every operation's exact semantics, the
hash primitive, every graded hash's field order/width/order-of-iteration, and
all ordering / tie-break / edge-case rules are specified normatively below.

FNV-1a-64 PRIMITIVE (the ONLY hash used; every hash below uses it):
  basis = 1469598103934665603  (0x14650FB0739D0383)   <- project basis, NOT
          the canonical 0xCBF29CE484222325.
  prime = 1099511628211        (0x00000100000001B3).
  Start h = basis.  For each consumed byte b, in order:
            h = (h XOR b) * prime          (all arithmetic mod 2^64).
  A field of width W (u8=1, u32=4, u64=8, i64=8 bytes) is consumed as exactly
  its W raw little-endian bytes (low byte first).  Signed values are first
  reinterpreted as the two's-complement unsigned bit pattern of the same width.
  Fields are consumed strictly in the listed order; nothing else (no padding,
  separators, lengths, counts, or type tags) is ever hashed.

  event_hash is ONE running accumulator: initialized to basis exactly once at
  solution_init / solution_reset and PERSISTING across solution_run calls; every
  emitted event appends its 10 fields to it (see PER-EVENT FIELD MAPPING below).
  cell_hash, waiter_hash, ready_hash, pending_hash are each recomputed from
  scratch (h = basis) at the END of every solution_run over the then-current
  state.  All graded outputs are exact integers; there is no float, no tolerance.

PERSISTENT STATE
  clock          u64 (= 0)
  event_seq      u64 (= 0); incremented and assigned to every emitted event
                 BEFORE the event is hashed (first event of a run = 1).
  store_seq_next u64 (= 1)
  wait_seq_next  u64 (= 1)
  ready_seq_next u64 (= 1)
  epoch          u64 (= 0)            global epoch (used in payload hashing)
  op_index       u64 (= 0)            global op index across the whole run

  edge[edge_id]  : chunk_count, edge_epoch:u64, defined:u8
  cell(edge,chunk): counter:u64, page_state, payload_hash:u64,
                    producer_id_or_ZERO:u64, store_seq_or_ZERO:u64,
                    release_seq_or_ZERO:u64
  waiter pool    : consumer_id:u64, edge_id, chunk_id, target:u64, wait_seq:u64,
                   consume_seed:u64, armed_epoch:u64, state, in_queue:u8
  ready queue    : (waiter_id, observed_epoch) ordered by ready_seq
  pending stores : (producer_id, edge_id, chunk_id, increment, payload_hash,
                    store_seq, due_clock) ordered by (due_clock, store_seq,
                    edge_id, chunk_id)

GLOBAL SEQUENCING (applies to EVERY op, including invalid / unknown kind)
  op_index : value used for the op's events = (uint32_t)op_index_global (the
             low 32 bits of a u64 counter that starts at 0 and PERSISTS across
             runs).  It is read BEFORE the op runs and op_index_global += 1 AFTER
             the op completes (so all events of one op share one op_index, and
             even an invalid/unknown op consumes one index).
  event_seq : starts 0, persists across runs; emitting an event first does
             event_seq += 1 (mod 2^64) and that NEW value is the event's
             event_seq (first event of the whole run has event_seq == 1).
  store_seq_next / wait_seq_next / ready_seq_next : start 1; each allocation
             takes the current value then increments by 1 (mod 2^64).
  epoch (global scalar, state_scalars[5]) : starts 0 and is NEVER modified by
             any op; it is used only as the 6th input to payload_hash (so it is
             effectively always 0).  Per-edge edge_epoch is the value that moves.

OPERAND DECODING (args are int32; arg_i64 is the i64 operand)
  Identifiers/seeds/latency/delta are read as (uint32_t) then zero-extended to
  u64: producer_id/consumer_id = (u32)arg_a, payload_seed/consume_seed =
  (u32)arg_e, store_latency = (u32)arg_f, delta = (u32)arg_a.  increment /
  target / amount = (u64)arg_i64.  edge/chunk/first_chunk/chunk_count/limit/
  max_store_completions are used as signed int32.

OPERATIONS (exact semantics; "INVALID(K)" = emit INVALID event with payload=K
and counts[INVALID]++, where K is the opcode; see PER-EVENT FIELD MAPPING)

  DEFINE_EDGE(edge_id=arg_a, chunk_count=arg_b)   [opcode 0]
    - INVALID(0) if edge_id<0 || edge_id>=edge_count || chunk_count<=0 ||
      chunk_count>max_chunks_per_edge.
    - If the edge is already defined: scan chunks [0,old_chunk_count); if ANY has
      page_state not in {EMPTY, RELEASED}, INVALID(0) (no state change).
    - Lazy-remove this edge's waiters: every waiter with edge_id==e and state in
      {WAITING, READY} is set to REMOVED (terminal, NO event); clear the wait
      queue of every chunk [0,max_chunks_per_edge) of this edge.
    - defined=1; chunk_count=arg_b; edge_epoch=(edge_epoch+1)%(max_epoch+1).
    - Clear ALL max_chunks_per_edge cells: counter=0, page=EMPTY, payload_hash=0,
      producer_id=0, store_seq=0, release_seq=0, wait queue empty.
    - Emit EDGE_DEFINE; counts[EDGE_DEFINED]++.

  RESET_EDGE(edge_id=arg_a)   [opcode 1]
    - INVALID(1) if edge not defined.
    - If ANY chunk in [0,chunk_count) has page_state in {STORING, READY,
      CONSUMING}: emit ONE RESET_STALL; counts[RESET_STALL]++; return (no reset).
    - Else if ANY waiter with edge_id==e and state in {WAITING, READY} exists:
      emit ONE RESET_STALL; counts[RESET_STALL]++; return (no reset).
    - Else edge_epoch=(edge_epoch+1)%(max_epoch+1); clear ALL cells (as in
      DEFINE_EDGE); emit EDGE_RESET; counts[EDGE_RESET]++.

  PRODUCE(producer_id=arg_a, edge_id=arg_b, first_chunk=arg_c, chunk_count=arg_d,
          increment=arg_i64, payload_seed=arg_e, store_latency=arg_f)  [opcode 2]
    - INVALID(2) if !edge_defined(e) || chunk_count<=0 || first_chunk<0.
    - INVALID(2) if first_chunk+chunk_count > edge.chunk_count.
    - For k=0..chunk_count-1 (ASCENDING), ch=first_chunk+k:
        * If cell(e,ch).page in {STORING, READY, CONSUMING}: emit
          PRODUCE_STALL_CHUNK for (e,ch); counts[PRODUCE_STALL_CHUNK]++; STOP the
          whole op immediately (already-issued earlier chunks stay scheduled).
        * Else (EMPTY or RELEASED): store_seq=store_seq_next++;
          ph = payload_hash(payload_seed, producer_id, e, ch, store_seq, epoch);
          set cell page=STORING, producer_id=producer_id, store_seq=store_seq,
          payload_hash=ph, release_seq=0; insert a pending store
          (producer_id,e,ch,increment,ph,store_seq, due_clock=clock+store_latency)
          into the pending queue keeping it sorted (see PENDING ORDER); emit
          STORE_ISSUE (payload=ph); counts[STORE_ISSUED]++.

  ADVANCE(delta=arg_a, max_store_completions=arg_b)   [opcode 3]
    - clock = clock + delta (mod 2^64).  (delta==0 is valid.)
    - If max_store_completions<=0, return.
    - Scan the pending queue from the FRONT with i=0, processed=0; while
      i<pending_count && processed<max_store_completions:
        * If pending[i].due_clock > clock: i++ and continue (non-due entries are
          skipped, keep their place, and do NOT count toward the limit).
        * Else remove pending[i] (front-ward, preserving order of the rest);
          processed++ (each REMOVED store counts, matching OR stale); leave i
          unchanged (next entry shifted into slot i).  Let the store be
          (producer,e,ch,increment,ph,store_seq):
            - If cell(e,ch).store_seq==store_seq AND cell.page==STORING:
              page=READY; payload_hash=ph; counter += increment (mod 2^64); emit
              STORE_COMPLETE (counter=new counter, payload=ph);
              counts[STORE_COMPLETE]++; then EVALUATE WAITERS of (e,ch).
            - Else: emit STORE_STALE_DROP (payload=store_seq);
              counts[STORE_STALE_DROP]++.

    EVALUATE WAITERS of cell (e,ch): scan its wait queue in stored order (which
    is ascending wait_seq).  Let counter=cell.counter, ee=edge.edge_epoch.  For
    each waiter wid in order:
        - state in {CANCELLED, REMOVED, CONSUMED}: drop from the queue (skip).
        - armed_epoch != ee: set state=REMOVED (NO event); drop from the queue.
        - state==WAITING && target<=counter: set state=READY; append a ready
          entry (ready_seq=ready_seq_next++, waiter=wid, observed_epoch=ee) to
          the ready queue; emit WAITER_READY (counter=counter, payload=target);
          counts[WAITER_READY]++; KEEP wid in the wait queue.
        - otherwise (WAITING with unmet target, or already READY): KEEP wid.

  ARM_WAIT(consumer_id=arg_a, edge_id=arg_b, chunk_id=arg_c, target=arg_i64,
           consume_seed=arg_e)   [opcode 4]
    - INVALID(4) if !edge_defined(e) || chunk<0 || chunk>=edge.chunk_count.
    - INVALID(4) if a nonterminal waiter (state in {WAITING,READY}) already exists
      for the SAME (consumer_id, e, chunk).
    - Create waiter (id = current pool size): consumer_id, e, chunk, target,
      wait_seq=wait_seq_next++, consume_seed, armed_epoch=edge.edge_epoch.
    - If cell.page==READY && cell.counter>=target: state=READY; append ready entry
      (ready_seq=ready_seq_next++, observed_epoch=edge.edge_epoch); emit
      WAITER_READY_IMMEDIATE (counter=cell.counter, payload=target);
      counts[WAITER_READY_IMMEDIATE]++.
    - Else: state=WAITING; insert wid into the cell wait queue keeping ascending
      wait_seq order; emit WAITER_ARM (counter=UINT64_MAX, payload=target);
      counts[WAITER_ARMED]++.

  CONSUME(limit=arg_a)   [opcode 5]
    - If limit<=0, valid no-op (return).
    - valid=0; while valid<limit && ready queue nonempty, inspect the FRONT entry
      (waiter wid, observed_epoch); let e,chunk,consumer come from the waiter:
        (a) If waiter.state != READY OR observed_epoch != edge.edge_epoch: emit
            READY_STALE_DROP (cons=consumer); counts[READY_STALE_DROP]++; pop the
            front; continue (does NOT count toward limit).
        (b) Else if cell.page != READY: emit READY_STALE_DROP;
            counts[READY_STALE_DROP]++; pop front; continue (no count).
        (c) Else if cell.counter < waiter.target: set waiter state=WAITING;
            reinsert wid into the cell wait queue keeping ascending wait_seq
            order; emit READY_REQUEUE (counter=cell.counter, payload=target);
            counts[READY_REQUEUE]++; pop front; continue (no count).
        (d) Else CONSUME: cell.page=CONSUMING;
            cval = consume_value(consume_seed, consumer, e, chunk, cell.counter,
                                 cell.payload_hash, waiter.wait_seq);
            emit CONSUME_CHUNK (counter=cell.counter, payload=cval);
            counts[CONSUME_CHUNK]++.  Then cell.page=RELEASED; cell.producer_id=0;
            cell.store_seq=0 (counter AND payload_hash are KEPT);
            release_seq = event_seq + 1 (the event_seq the upcoming CHUNK_RELEASE
            will carry); cell.release_seq=release_seq; emit CHUNK_RELEASE
            (counter=cell.counter, payload=release_seq); counts[CHUNK_RELEASE]++.
            Set waiter state=CONSUMED; pop front; valid++.

  CANCEL_WAIT(consumer_id=arg_a, edge_id=arg_b, chunk_id=arg_c)   [opcode 6]
    - INVALID(6) if e<0 || e>=edge_count (edge need NOT be defined; chunk range
      is NOT pre-checked here).
    - Find the nonterminal waiter (state in {WAITING,READY}) for (consumer,e,
      chunk); if none, INVALID(6).
    - Else set its state=CANCELLED; emit WAITER_CANCEL; counts[WAITER_CANCEL]++.
      It stays in its wait queue until lazily skipped by EVALUATE WAITERS.

  FORCE_COUNTER(edge_id=arg_a, chunk_id=arg_b, amount=arg_i64)   [opcode 7]
    - INVALID(7) if !edge_defined(e) || chunk<0 || chunk>=edge.chunk_count.
    - cell.counter += amount (mod 2^64); emit COUNTER_FORCE (counter=new counter,
      payload=amount); counts[COUNTER_FORCE]++; then EVALUATE WAITERS of (e,chunk).
      This does NOT make the page READY.

  Any other op.kind : emit INVALID(op.kind); counts[INVALID]++.

ORDERING / TIE-BREAKS (the graded coordination traces)
  PENDING ORDER : the pending-store queue is kept sorted ascending by the tuple
    (due_clock, store_seq, edge_id, chunk_id); insertion finds the first existing
    entry strictly greater by that tuple and inserts before it (stable).  Both
    ADVANCE scanning and pending_hash use this order.
  WAIT-QUEUE ORDER : each cell's wait queue is ordered by ascending wait_seq;
    ARM_WAIT and CONSUME requeue insert at the position preserving that order.
  READY ORDER : the single global ready queue is FIFO by enqueue (i.e. ascending
    ready_seq); entries are consumed/dropped from the front; ready_hash uses this
    order.  observed_epoch on each entry is edge_epoch captured at enqueue.

GRADED OUTPUTS
  - counts[MK_COUNT_TOTAL]   exact integer counts, see MK_COUNT_* below
  - event_hash               FNV over events in emission order; per event:
      event_kind:u8, event_seq:u64, op_index:u32, clock:u64,
      producer_or_ZERO:u64, consumer_or_ZERO:u64, edge_id:u32,
      chunk_id_or_UINT32_MAX:u32, counter_or_UINT64_MAX:u64, payload_or_ZERO:u64
  - cell_hash                FNV over DEFINED edges in ascending edge_id, and for
      each such edge its chunks [0, chunk_count) in ascending chunk_id:
      edge_id:u32, edge_epoch:u64, chunk_id:u32, counter:u64, page_state:u8,
      payload_hash:u64, producer_id_or_ZERO:u64, store_seq_or_ZERO:u64,
      release_seq_or_ZERO:u64
  - waiter_hash              FNV over nonterminal waiters sorted by
      (edge_id, chunk_id, wait_seq, consumer_id):
      consumer_id:u64, edge_id:u32, chunk_id:u32, target:u64, wait_seq:u64,
      consume_seed:u64, state:u8
  - ready_hash               FNV over ready entries by ready queue order:
      ready_seq:u64, consumer_id:u64, edge_id:u32, chunk_id:u32,
      observed_epoch:u64
  - pending_hash             FNV over pending stores by queue order:
      due_clock:u64, store_seq:u64, producer_id:u64, edge_id:u32, chunk_id:u32,
      increment:u64, payload_hash:u64
  - state_scalars[6]         clock, event_seq, store_seq_next, wait_seq_next,
                             ready_seq_next, epoch  (epoch is the global scalar,
                             always 0)
  Notes: waiter "state" is hashed as the raw u8 state code (WAITING=0, READY=1).
  Terminal waiters (CONSUMED/CANCELLED/REMOVED) are EXCLUDED from waiter_hash.
  ready_hash reads consumer_id/edge_id/chunk_id from the referenced waiter.

DERIVED PAYLOAD HASHES (each is an independent FNV-1a-64 over the listed fields,
starting from basis; these results then feed cell/event hashes as u64 fields):
  payload_hash  = FNV1a64( payload_seed:u64, producer_id:u64, edge_id:u32,
                           chunk_id:u32, store_seq:u64, epoch:u64 )
                  (epoch = the global epoch scalar, i.e. 0, at PRODUCE time).
  consume_value = FNV1a64( consume_seed:u64, consumer_id:u64, edge_id:u32,
                           chunk_id:u32, cell.counter:u64, cell.payload_hash:u64,
                           wait_seq:u64 ).  Carried as CONSUME_CHUNK's payload.

PER-EVENT FIELD MAPPING (event_hash fields: kind:u8, event_seq:u64, op_index:u32,
clock:u64, producer:u64, consumer:u64, edge_id:u32, chunk_id:u32, counter:u64,
payload:u64).  Unless stated, producer=0, consumer=0, payload=0, chunk_id is the
real chunk, counter is the relevant value; sentinels are UINT32_MAX (chunk) and
UINT64_MAX (counter).  clock = the current clock at emit time.
  EDGE_DEFINE / EDGE_RESET / RESET_STALL : edge set; chunk=UINT32_MAX;
      counter=UINT64_MAX; producer=consumer=payload=0.
  STORE_ISSUE        : edge,chunk; producer=producer_id; counter=UINT64_MAX;
      payload=payload_hash (assigned hash).
  STORE_COMPLETE     : edge,chunk; producer=producer_id; counter=new counter;
      payload=cell.payload_hash.
  PRODUCE_STALL_CHUNK: edge,chunk; producer=producer_id; counter=UINT64_MAX;
      payload=0.
  STORE_STALE_DROP   : edge,chunk; producer=producer_id; counter=UINT64_MAX;
      payload=store_seq.
  WAITER_ARM         : edge,chunk; consumer=consumer_id; counter=UINT64_MAX;
      payload=target.
  WAITER_READY / WAITER_READY_IMMEDIATE / READY_REQUEUE : edge,chunk;
      consumer=consumer_id; counter=cell.counter; payload=target.
  READY_STALE_DROP   : edge,chunk; consumer=consumer_id; counter=UINT64_MAX;
      payload=0.
  CONSUME_CHUNK      : edge,chunk; consumer=consumer_id; counter=cell.counter;
      payload=consume_value.
  CHUNK_RELEASE      : edge,chunk; consumer=consumer_id; counter=cell.counter;
      payload=release_seq (= this event's own event_seq).
  WAITER_CANCEL      : edge,chunk; consumer=consumer_id; counter=UINT64_MAX;
      payload=0.
  COUNTER_FORCE      : edge,chunk; producer=consumer=0; counter=new counter;
      payload=amount.
  INVALID            : edge=UINT32_MAX, chunk=UINT32_MAX, counter=UINT64_MAX,
      producer=consumer=0, payload=opcode.

DETERMINISM / EDGE CASES
  - A store completion increments the counter and ONLY THEN evaluates waiters.
  - counter>=target is insufficient to consume unless the page is READY; a READY
    waiter whose page is no longer READY is dropped as READY_STALE_DROP.
  - Wait queues are scanned by original wait_seq; ready order is fixed at wake
    time (ready_seq).  A waiter made READY stays in its cell wait queue (it is
    removed only when consumed/cancelled/removed via later EVALUATE WAITERS).
  - Reset/cancel/define make queued ready entries stale; they are dropped (not
    requeued) when their waiter is no longer READY or observed_epoch mismatches.
  - PRODUCE stops at the first busy chunk; earlier chunks stay scheduled.
  - CONSUME counts only successful (d) consumptions against limit; (a)/(b)/(c)
    do not count.  FORCE_COUNTER can wake waiters but never makes a page READY.
  - All sequence and chunk counters wrap modulo 2^64 and are compared/sorted as
    unsigned 64-bit values.

Rules:
  - solution_init may allocate persistent state.
  - solution_run may not call cudaMalloc / cudaFree.
  - solution_run may launch kernels and use the provided workspace.
*/

struct alignas(8) MkProblemSpec {
    int32_t abi_version;
    int32_t edge_count;
    int32_t max_chunks_per_edge;
    int32_t max_waiters;
    int32_t max_ready_entries;
    int32_t max_store_events;
    int32_t max_consumers;
    int32_t max_epoch;
    int32_t max_ops;
    int32_t reserved[7];
};

struct alignas(8) MkRunSpec {
    int32_t abi_version;
    int32_t op_count;
    int32_t step_id;
    int32_t flags;
    int32_t reserved[12];
};

// One operation.  Unused arg slots are ignored by the op semantics.  The
// 64-bit operand (increment / target / amount) lives in arg_i64.
struct alignas(8) MkOp {
    int32_t kind;
    int32_t arg_a;
    int32_t arg_b;
    int32_t arg_c;
    int32_t arg_d;
    int32_t arg_e;
    int32_t arg_f;
    int32_t pad;
    int64_t arg_i64;
};

struct alignas(8) MkInputs {
    const MkOp* ops;
};

struct alignas(8) MkOutputs {
    uint64_t* counts;         // MK_COUNT_TOTAL entries
    uint64_t* event_hash;     // 1
    uint64_t* cell_hash;      // 1
    uint64_t* waiter_hash;    // 1
    uint64_t* ready_hash;     // 1
    uint64_t* pending_hash;   // 1
    uint64_t* state_scalars;  // 6
};

// counts[] indices
#define MK_COUNT_EDGE_DEFINED 0
#define MK_COUNT_EDGE_RESET 1
#define MK_COUNT_RESET_STALL 2
#define MK_COUNT_STORE_ISSUED 3
#define MK_COUNT_STORE_COMPLETE 4
#define MK_COUNT_PRODUCE_STALL_CHUNK 5
#define MK_COUNT_STORE_STALE_DROP 6
#define MK_COUNT_WAITER_ARMED 7
#define MK_COUNT_WAITER_READY 8
#define MK_COUNT_WAITER_READY_IMMEDIATE 9
#define MK_COUNT_READY_STALE_DROP 10
#define MK_COUNT_READY_REQUEUE 11
#define MK_COUNT_CONSUME_CHUNK 12
#define MK_COUNT_CHUNK_RELEASE 13
#define MK_COUNT_WAITER_CANCEL 14
#define MK_COUNT_COUNTER_FORCE 15
#define MK_COUNT_INVALID 16
#define MK_COUNT_TOTAL 17

static_assert(sizeof(MkProblemSpec) == 64, "MkProblemSpec layout drift");
static_assert(sizeof(MkRunSpec) == 64, "MkRunSpec layout drift");
static_assert(sizeof(MkOp) == 40, "MkOp layout drift");
static_assert(sizeof(MkInputs) == 8, "MkInputs layout drift");
static_assert(sizeof(MkOutputs) == 56, "MkOutputs layout drift");

static inline size_t mk_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int mk_validate_problem_spec(const MkProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MK_ABI_VERSION) return 0;
    if (spec->edge_count < MK_MIN_EDGES || spec->edge_count > MK_MAX_EDGES) return 0;
    if (spec->max_chunks_per_edge < MK_MIN_CHUNKS ||
        spec->max_chunks_per_edge > MK_MAX_CHUNKS_PER_EDGE) return 0;
    if (spec->max_waiters < 0 || spec->max_waiters > MK_MAX_WAITERS) return 0;
    if (spec->max_ready_entries < 0 || spec->max_ready_entries > MK_MAX_READY) return 0;
    if (spec->max_store_events < 0 || spec->max_store_events > MK_MAX_STORE_EVENTS) return 0;
    if (spec->max_consumers < 0 || spec->max_consumers > MK_MAX_CONSUMERS) return 0;
    if (spec->max_epoch < 1 || spec->max_epoch > MK_MAX_EPOCH) return 0;
    if (spec->max_ops < 0 || spec->max_ops > MK_MAX_OPS) return 0;
    return 1;
}

static inline int mk_validate_run_spec(const MkRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != MK_ABI_VERSION) return 0;
    if (run->op_count < 0 || run->op_count > MK_MAX_OPS) return 0;
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

#endif  // MK_CHUNKED_DEP_COUNTERS_COMMON_H_
