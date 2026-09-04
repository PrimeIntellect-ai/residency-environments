// file: mk_multigpu_allreduce_common.h
//
// MK9: Multi-GPU Persistent AllReduce with Remote Scheduler Signals (T68).
//
// A persistent multi-rank megakernel runtime simulated as logical partitions in
// a single GPU's device memory. It coordinates local reduction, ring
// reduce-scatter / allgather phases, NVSHMEM-style remote signal/wait counters,
// receive buffers, send credits, and per-rank scheduler progress -- entirely
// inside the kernel. Grounded in Hazy's cross-GPU megakernel approach, NVSHMEM's
// device-side synchronization model, and NCCL's AllReduce semantics.

#ifndef MK_MULTIGPU_ALLREDUCE_COMMON_H_
#define MK_MULTIGPU_ALLREDUCE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MGA_ABI_VERSION 1

// Bounds for the persistent runtime. These bound device allocations.
#define MGA_MIN_RANKS 2
#define MGA_MAX_RANKS 8
#define MGA_MIN_CHUNKS 1
#define MGA_MAX_CHUNKS 8
#define MGA_MIN_COLLS 1
#define MGA_MAX_COLLS 8
#define MGA_MAX_REMOTE_EVENTS 4096
#define MGA_MAX_SEND_CREDITS 16
#define MGA_MAX_REMOTE_LATENCY 64
#define MGA_MAX_SCHED_QUEUE 512

// Operation codes carried in the run spec.
#define MGA_OP_BEGIN_ALLREDUCE 0
#define MGA_OP_RANK_STEP       1
#define MGA_OP_ADVANCE         2
#define MGA_OP_POLL_SIGNAL     3
#define MGA_OP_CANCEL_COLL     4
#define MGA_OP_FORCE_CREDIT    5

// Collective status.
#define MGA_ST_ACTIVE    0
#define MGA_ST_DONE      1
#define MGA_ST_CANCELLED 2

// Collective phase.
#define MGA_PH_LOCAL_REDUCE   0
#define MGA_PH_REDUCE_SCATTER 1
#define MGA_PH_ALLGATHER      2
#define MGA_PH_DONE           3

// Signal / event phase kinds (NVSHMEM-style signal counter dimension).
#define MGA_PK_LOCAL 0
#define MGA_PK_RS    1
#define MGA_PK_AG    2
#define MGA_PK_NONE  255

// Scheduler action kinds.
#define MGA_ACT_LOCAL_REDUCE        0
#define MGA_ACT_REDUCE_SCATTER_SEND 1
#define MGA_ACT_ALLGATHER_SEND      2
#define MGA_ACT_REMOTE_RECV_REDUCE  3
#define MGA_ACT_REMOTE_RECV_GATHER  4

// Event kinds (emission-order hashing). Ordinals are stable.
#define MGA_EV_COLL_BEGIN        0
#define MGA_EV_LOCAL_READY       1
#define MGA_EV_PHASE_ADVANCE_RS  2
#define MGA_EV_PHASE_ADVANCE_AG  3
#define MGA_EV_REMOTE_SEND       4
#define MGA_EV_SEND_CREDIT_STALL 5
#define MGA_EV_REMOTE_ARRIVE     6
#define MGA_EV_REMOTE_STALE_DROP 7
#define MGA_EV_CREDIT_RETURN     8
#define MGA_EV_CREDIT_FORCE      9
#define MGA_EV_REDUCE_APPLY      10
#define MGA_EV_OWNER_REDUCED     11
#define MGA_EV_GATHER_APPLY      12
#define MGA_EV_COLL_DONE         13
#define MGA_EV_SIGNAL_READY      14
#define MGA_EV_SIGNAL_WAIT       15
#define MGA_EV_ACTION_STALE_DROP 16
#define MGA_EV_COLL_CANCEL       17
#define MGA_EV_INVALID           18

#define MGA_COUNTER_COUNT 19

/*
CONTRACT: mk_multigpu_allreduce  (T68 / MK9)

A persistent multi-rank AllReduce runtime. Each solution_run applies exactly ONE
operation and emits the updated cumulative counters plus canonical FNV-1a-64
hashes of the runtime state and the per-step event stream. The multi-GPU fabric
is simulated as logical rank partitions inside one GPU's device memory.

FIVE COUPLED SUBSYSTEMS (none may be dropped):
  1. Remote-scheduler coordination: per-rank scheduler action queues ordered by
     append-time phase_seq.
  2. Signal/wait counters: signal_counter[dst][coll][phase_kind][phase_index]
     [chunk]; polled by POLL_SIGNAL.
  3. All-reduce stages: LOCAL_REDUCE -> REDUCE_SCATTER -> ALLGATHER -> DONE over
     a ring of active ranks.
  4. Cross-GPU dependency signaling: pending remote events ordered by (due_clock,
     remote_seq); arrival writes a receive buffer, increments the signal counter,
     returns a credit, then enqueues the receiver action.
  5. Send credits: per directed active ring-link credit; a send with zero credit
     stalls and is re-appended (it does not block later actions).

PERSISTENT SCALARS
  clock=0; event_seq=0; collective_seq_next=1; remote_seq_next=1;
  phase_seq_next=1.

GEOMETRY / RING
  Active ranks of a collective are the set bits of participant_mask, sorted
  ascending; they form the ring. rank_position(r) is the index in that sorted
  list. next(rank)/prev(rank) are circular over active ranks.

OPERATIONS  (run.op selects; integer args ride in a0..a7)
  BEGIN_ALLREDUCE(coll_id=a0, participant_mask=a1, chunk_count=a2, seed=a3)
    Invalid if: coll_id already exists ACTIVE; collective table full; mask has
    fewer than two ranks; chunk_count out of [MIN,MAX]. Otherwise create an ACTIVE
    collective (collective_seq=collective_seq_next++), seed every (rank,chunk):
      local_value = signed_i64(FNV1a64(seed, coll_id, rank, chunk));
      accum_value = local_value; have_local = 1; all ready flags = 0;
      last_update_seq = 0.
    Initialise every active directed ring-link credit to max_send_credits_per_link.
    Enqueue a LOCAL_REDUCE action for every active rank/chunk in rank-asc then
    chunk-asc order. Emit COLL_BEGIN. (A DONE/CANCELLED coll_id slot may be reused.)
  RANK_STEP(rank=a0, action_limit=a1)
    Invalid if rank out of range or action_limit==0. Process up to action_limit
    actions from the rank's queue (front order). An action whose collective is
    absent or non-ACTIVE is dropped with ACTION_STALE_DROP.
      LOCAL_REDUCE: if (rank,chunk) has_local, emit LOCAL_READY and mark the cell
        local-ready. When every active rank/chunk is local-ready (and RS not yet
        started), set phase=REDUCE_SCATTER, emit PHASE_ADVANCE_RS once, and enqueue
        each active rank's first reduce-scatter send (phase 0). The ring slot sent
        by rank r at reduce phase p is (rank_position(r) - p + R) mod R; the chunk
        touched is slot mod chunk_count.
      REDUCE_SCATTER_SEND(phase p, chunk): dst=next(rank). If link credit==0 emit
        SEND_CREDIT_STALL and re-append the same action with a fresh phase_seq.
        Else credit--, enqueue a remote event due clock+remote_latency carrying
        value=accum[rank][chunk], emit REMOTE_SEND.
      REMOTE_RECV_REDUCE (created by arrival): accum[rank][chunk] += value
        (signed wrap), emit REDUCE_APPLY. If reduce phases remain (p+1 < R-1)
        enqueue this rank's next send (phase p+1). On the last reduce phase mark
        the cell reduced-owner-ready and emit OWNER_REDUCED. When R owner cells are
        reduced, set phase=ALLGATHER, emit PHASE_ADVANCE_AG once, and enqueue each
        owner's allgather phase-0 send.
      ALLGATHER_SEND(phase p, chunk): same credit rule; value=accum[rank][chunk].
      REMOTE_RECV_GATHER (created by arrival): final_value[rank][chunk]=value,
        final_ready=1, emit GATHER_APPLY. If gather phases remain (p+1 < R-1)
        enqueue this rank's next allgather send (phase p+1, same chunk). When every
        active rank has every chunk final-ready, set status=DONE, phase=DONE, emit
        COLL_DONE.
  ADVANCE(delta=a0, max_remote=a1)
    Invalid if delta<0 OR max_remote<0 (emit INVALID with coll_id=0, src=UINT32_MAX,
    dst=UINT32_MAX, phase_kind=255, phase_index=UINT32_MAX, chunk=UINT32_MAX,
    value=INT64_MIN). Otherwise: delta==0 is valid. clock += delta. Process up to
    max_remote pending remote events
    with due_clock<=clock in canonical (due_clock, remote_seq) order. For each: if
    its collective is absent/cancelled/done emit REMOTE_STALE_DROP and drop it.
    Otherwise write the receive buffer, increment the matching signal counter, emit
    REMOTE_ARRIVE; return one credit to the reverse recorded link (clamped to max)
    and emit CREDIT_RETURN; then enqueue the receiver action (REMOTE_RECV_REDUCE for
    a reduce-scatter event, REMOTE_RECV_GATHER for an allgather event).
  POLL_SIGNAL(rank=a0, coll_id=a1, phase_kind=a2, phase_index=a3, chunk=a4,
              target=a5)
    Invalid if ANY of: rank out of [0,rank_count); coll_id absent; phase_kind not
    in [0,2]; phase_index out of [0,phidx_bound); chunk out of [0,chunk_max).
    (All five bounds are checked -- phase_index and chunk are NOT exempt.) The
    INVALID event uses the POLL field mapping pinned in "EMITTED EVENT FIELD VALUES"
    below. Otherwise: if signal_counter>=target emit SIGNAL_READY else SIGNAL_WAIT.
    Enqueues no action; included in hashes.
  CANCEL_COLL(coll_id=a0)
    Invalid if absent or terminal. Set status=CANCELLED, emit COLL_CANCEL. Queued
    actions and pending remote events become stale lazily when next encountered.
  FORCE_CREDIT(src=a0, dst=a1, amount=a2)
    Invalid if (src,dst) is not a directed ring edge of any ACTIVE collective.
    Otherwise, in every ACTIVE collective where it is an edge, set the link credit
    to clamp(credit + amount, 0, max_send_credits_per_link) -- i.e. amount may be
    negative; the result is clamped to BOTH ends [0, max] (new = credit+amount;
    if new>max new=max; if new<0 new=0). Emit CREDIT_FORCE.

  EMITTED EVENT FIELD VALUES (normative; resolves the value-field and INVALID-field
  ambiguities that feed remote_event_hash). The hash folds the literal emit() field
  values (see the layout under OUTPUTS). Most events' fields follow from their per-op
  description above; the following are pinned exactly because they were ambiguous:
   - REDUCE_APPLY : src = sender rank; dst = receiving rank; phase_kind = RS;
       phase_index = p; chunk; value = the UPDATED accum[rank][chunk] AFTER the
       signed-wrap addition (the post-reduction running sum -- NOT the arriving
       a->value).
   - OWNER_REDUCED: src = UINT32_MAX; dst = owner rank; phase_kind = RS;
       phase_index = p (the last reduce phase); chunk; value = accum[rank][chunk]
       (the same post-reduction owner sum).
   - GATHER_APPLY : src = sender rank; dst = receiving rank; phase_kind = AG;
       phase_index = p; chunk; value = final_value[rank][chunk] (= the just-stored
       arriving value).
   - INVALID is emitted with OP-SPECIFIC fields (it is NOT a uniform all-sentinel
       record); for the op that raised it:
       * BEGIN invalid      : coll_id = the requested coll_id; src = UINT32_MAX;
           dst = UINT32_MAX; phase_kind = 255; phase_index = UINT32_MAX;
           chunk = UINT32_MAX; value = INT64_MIN.
       * RANK_STEP invalid  : coll_id = 0; src = (rank>=0 ? (u32)rank : UINT32_MAX);
           dst = UINT32_MAX; phase_kind = 255; phase_index = UINT32_MAX;
           chunk = UINT32_MAX; value = INT64_MIN.
       * POLL_SIGNAL invalid: coll_id = the requested coll_id;
           src = (rank>=0 ? (u32)rank : UINT32_MAX); dst = UINT32_MAX;
           phase_kind = (0<=phase_kind<=2 ? phase_kind : 255);
           phase_index = (u32)requested phase_index; chunk = (u32)requested chunk;
           value = INT64_MIN.
       * FORCE_CREDIT invalid: coll_id = 0; src = (src>=0 ? (u32)src : UINT32_MAX);
           dst = (dst>=0 ? (u32)dst : UINT32_MAX); phase_kind = 255;
           phase_index = UINT32_MAX; chunk = UINT32_MAX; value = INT64_MIN.

OUTPUTS (per run, cumulative counters + state hashes)
  Counters [19]: coll_begin, local_ready, phase_advance_rs, phase_advance_ag,
    remote_send, send_credit_stall, remote_arrive, remote_stale_drop, credit_return,
    credit_force, reduce_apply, owner_reduced, gather_apply, coll_done, signal_ready,
    signal_wait, action_stale_drop, coll_cancel, invalid_count.
  remote_event_hash: FNV-1a-64 over THIS run's events in emission order, each
    contributing event_kind:u8, event_seq:u64, op_index:u32, clock:u64,
    coll_id_or_ZERO:u64, src_or_UINT32_MAX:u32, dst_or_UINT32_MAX:u32,
    phase_kind_or_255:u8, phase_index_or_UINT32_MAX:u32, chunk_or_UINT32_MAX:u32,
    value_or_INT64_MIN:i64.
  collective_hash: used collectives by coll_id asc, then rank, then chunk.
  signal_hash: nonzero signal counters by rank, coll_id, phase_kind, phase_index,
    chunk.
  credit_hash: directed ring-link credits by coll_id, src rank, dst rank.
  pending_remote_hash: pending remote events in canonical queue order.
  scheduler_hash: per-rank action queues by rank then position.
  clock_out, event_seq_out: persistent clock and event_seq after the run.

DETERMINISM
  Active ranks sorted ascending form the ring. Reduce-scatter precedes allgather;
  allgather begins only after every owner chunk is reduced. A remote send requires
  credit before creating a pending event. A remote arrival writes buffer, increments
  signal, returns credit, then enqueues the receiver action -- in that order. Credit
  return clamps to capacity. Stalled sends are re-appended with a new phase_seq and
  do not block later actions. Cancellation is lazy. All reductions use signed
  two's-complement wrap; all counters and clock wrap modulo 2^64.
*/

// === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section inlines the EXACT byte-level serialization of every graded output,
// derived field-for-field from the reference. A solver that reproduces these
// emissions bit-for-bit will match the graded checksums. There is no external
// "section 4 / oracle / design doc" to consult; everything is here.
//
// ---- FNV-1a-64 primitive (used by EVERY hash and by the local-value seed) ----
//   offset_basis (init/seed) = 1469598103934665603  (0x14650FB0739D0383)
//   prime                    = 1099511628211        (0x00000100000001B3)
//   Per byte b (in increasing memory address order, i.e. LITTLE-ENDIAN for the
//   multi-byte integer fields below, since the device is little-endian):
//       h ^= (uint64_t)b;  h *= prime;
//   "Emit field X as width W" below means: feed the W little-endian bytes of X's
//   two's-complement / unsigned representation through the per-byte step, in
//   address order (LSB first). Signed fields (i64) are hashed by their raw 8-byte
//   two's-complement image; unsigned fields (u8/u32/u64) by their raw image.
//   Each hash is INDEPENDENT and starts from a fresh h = offset_basis.
//
// ---- Per-collective LOCAL-VALUE seed (BEGIN_ALLREDUCE) ----
//   For each seeded (rank,chunk): start hv = offset_basis, then emit, each as u64
//   (8 LE bytes), in THIS exact order:
//       seed (=(uint64_t)(uint32_t)a3), coll_id (=(uint64_t)(uint32_t)a0),
//       (uint64_t)rank, (uint64_t)chunk.
//   local_value = (int64_t)hv;  accum_value = local_value.
//
// ---- remote_event_hash (per-run event stream; ALREADY fully specified above) ----
//   Restated for completeness. h = offset_basis at the START of every run. For
//   each event, in EMISSION order (the order r_emit/emit is called during the op),
//   emit these 11 fields in EXACTLY this order and width:
//       event_kind                 : u8
//       event_seq (pre-increment)  : u64   (the value BEFORE this event bumps it)
//       op_index (run.op_index)    : u32
//       clock (current run clock)  : u64
//       coll_id_or_ZERO            : u64   (0 when the event carries no coll)
//       src_or_UINT32_MAX          : u32
//       dst_or_UINT32_MAX          : u32
//       phase_kind_or_255          : u8    (MGA_PK_NONE=255 when N/A)
//       phase_index_or_UINT32_MAX  : u32
//       chunk_or_UINT32_MAX        : u32
//       value_or_INT64_MIN         : i64   (INT64_MIN sentinel when no value)
//   Sentinels exactly as the op handlers pass them (see OPERATIONS prose). This is
//   a PER-RUN hash: it covers only events emitted by the current op, not history.
//
// ---- collective_hash (state snapshot) ----
//   ITERATION: used collectives selected by coll_id ASCENDING; within a collective,
//   active ranks in ascending (active_ranks[] sorted-asc) order (rank-major), then
//   chunk ascending over [0, chunk_count). (Cells outside [0,chunk_count) or for
//   inactive ranks are NOT emitted.)
//   PER (collective, rank, chunk) emit, in EXACTLY this order/width:
//       coll_id                    : u64
//       collective_seq             : u64
//       status                     : u8   (MGA_ST_*)
//       phase                      : u8   (MGA_PH_*)
//       rank                       : u32
//       chunk                      : u32
//       cell.local_value           : i64
//       cell.accum_value           : i64
//       cell.final_value           : i64
//       cell.reduced_owner_ready   : u8   (0/1)
//       cell.final_ready           : u8   (0/1)
//       cell.last_update_seq       : u64
//   (NOTE: have_local and local_ready are NOT part of this hash.)
//
// ---- signal_hash (state snapshot; NONZERO counters only) ----
//   ITERATION (nested loops, outermost first): rank ascending [0,rank_count);
//   then collective SLOT ascending [0,max_collectives) restricted to used slots
//   (slot order == allocation order, NOT coll_id order); then phase_kind pk in
//   {0,1,2} == {LOCAL,RS,AG}; then phase_index pidx ascending [0,phidx_bound)
//   where phidx_bound == rank_count; then chunk ascending [0,chunk_count_max).
//   Skip any entry whose counter value == 0. For each emitted (nonzero) entry,
//   emit in EXACTLY this order/width:
//       rank                       : u32
//       coll_id (of that slot)     : u64
//       phase_kind (pk)            : u8
//       phase_index (pidx)         : u32
//       chunk                      : u32
//       counter value              : u64
//
// ---- credit_hash (state snapshot) ----
//   ITERATION: used collectives by coll_id ASCENDING; within a collective, ring
//   edges by ring position pos ascending [0,active_rank_count). The edge at pos is
//   src=active_ranks[pos] -> dst=active_ranks[(pos+1) mod active_rank_count]; the
//   emitted credit is credit[pos] (the directed src->next link). For each edge,
//   emit in EXACTLY this order/width:
//       coll_id                    : u64
//       src rank                   : u32
//       dst rank                   : u32
//       credit (as u64)            : u64   (credit stored signed; cast to u64)
//
// ---- pending_remote_hash (state snapshot) ----
//   ITERATION: ALL pending remote events sorted into CANONICAL order: primary key
//   due_clock ascending, tie-break remote_seq ascending (stable, total order since
//   remote_seq is unique). For each pending event, emit in EXACTLY this order/width:
//       src                        : u32
//       dst                        : u32
//       coll_id                    : u64
//       phase_kind                 : u8   (MGA_PK_RS or MGA_PK_AG)
//       phase_index                : u32
//       chunk                      : u32
//       value                      : i64
//       due_clock                  : u64
//       remote_seq                 : u64
//
// ---- scheduler_hash (state snapshot) ----
//   ITERATION: rank ascending [0,rank_count); within a rank, queue entries in
//   FRONT-to-BACK append order, pos ascending [0,queue_len). For each action,
//   emit in EXACTLY this order/width:
//       rank                       : u32
//       pos (queue index)          : u32
//       coll_id                    : u64
//       action_kind                : u8   (MGA_ACT_*)
//       chunk                      : u32
//       phase_index                : u32
//       phase_seq                  : u64
//   (NOTE: action src/value are NOT part of this hash; phase_seq is the unique
//   append stamp assigned by enqueue.)
//
// ---- scalar outputs ----
//   clock_out      = persistent clock      AFTER the run (u64; ADVANCE adds delta).
//   event_seq_out  = persistent event_seq  AFTER the run (u64; +1 per emitted event).
//   Counters[19]: cumulative (persist across runs), int64, in the fixed index order
//   listed under OUTPUTS. Each event handler bumps exactly the counter named there.
//
// ---- additional enforced semantics not to overlook ----
//   * BEGIN seeds ALL active (rank,chunk in [0,chunk_count)); credit[pos] for active
//     positions = max_send_credits_per_link, others 0. LOCAL_REDUCE actions enqueued
//     rank-asc then chunk-asc.
//   * RS first send by rank r: ring slot = (rank_position(r) - 0) mod R; chunk = slot
//     mod chunk_count. Reduce phase p uses ring slot (pos - p + R) mod R. Reduce
//     phases run while (p+1 < R-1); on the phase where !(p+1 < R-1) the cell becomes
//     owner (OWNER_REDUCED). AG begins only once owners_reduced >= R; each owner emits
//     its owned chunk at AG phase 0. AG phases also run while (p+1 < R-1).
//   * REMOTE_RECV_REDUCE: accum += value with unsigned (two's-complement) wrap; sets
//     last_update_seq = event_seq AT apply time. REMOTE_RECV_GATHER: final_value =
//     value, final_ready=1, last_update_seq = event_seq.
//   * ADVANCE drains due events in canonical (due_clock, remote_seq) order; per event
//     order is: write recv buffer -> increment signal counter -> emit REMOTE_ARRIVE ->
//     return one credit to the reverse link credit_return_src (clamped to max) -> emit
//     CREDIT_RETURN -> enqueue receiver action (RECV_REDUCE for RS, RECV_GATHER for AG).
//     credit_return_src=dst, credit_return_dst=rank of the original send.
//   * A send with credit==0 emits SEND_CREDIT_STALL and re-appends the SAME action with
//     a FRESH phase_seq (does not block later actions); remote_seq advances ONLY when an
//     event is actually appended (queue full => no event, no remote_seq consumption).
//   * Stale/cancel: an action/event whose collective is absent or non-ACTIVE is dropped
//     (ACTION_STALE_DROP / REMOTE_STALE_DROP). CANCEL sets status=CANCELLED lazily.
//   * POLL_SIGNAL enqueues nothing; emits SIGNAL_READY if signal>=target else
//     SIGNAL_WAIT, carrying value=current signal counter (i64). Valid pk is 0..2.
//
// === END CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION ===

struct alignas(8) MgaProblemSpec {
    int32_t abi_version;
    int32_t rank_count;
    int32_t chunk_count_max;            // upper bound on per-collective chunk_count
    int32_t max_collectives;
    int32_t max_remote_events;
    int32_t max_send_credits_per_link;
    int32_t remote_latency;
    int32_t max_scheduler_queue_per_rank;
    int32_t reserved[8];
};

struct alignas(8) MgaRunSpec {
    int32_t abi_version;
    int32_t op;         // MGA_OP_*
    int32_t op_index;   // per-run identifier folded into the event hash
    int32_t a0;
    int32_t a1;
    int32_t a2;
    int32_t a3;
    int32_t a4;
    int32_t a5;
    int32_t a6;
    int32_t a7;
    int32_t reserved[5];
};

// No bulk input arrays; all operands ride in the run spec. inputs is reserved.
struct alignas(8) MgaInputs {
    const int32_t* reserved0;
    const int32_t* reserved1;
};

struct alignas(8) MgaOutputs {
    int64_t* counters;            // length 19
    uint64_t* remote_event_hash;
    uint64_t* collective_hash;
    uint64_t* signal_hash;
    uint64_t* credit_hash;
    uint64_t* pending_remote_hash;
    uint64_t* scheduler_hash;
    uint64_t* clock_out;
    uint64_t* event_seq_out;
};

static_assert(sizeof(MgaProblemSpec) == 64, "MgaProblemSpec layout drift");
static_assert(sizeof(MgaRunSpec) == 64, "MgaRunSpec layout drift");
static_assert(sizeof(MgaInputs) == 16, "MgaInputs layout drift");
static_assert(sizeof(MgaOutputs) == 72, "MgaOutputs layout drift");

static inline int mga_validate_problem_spec(const MgaProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MGA_ABI_VERSION) return 0;
    if (spec->rank_count < MGA_MIN_RANKS || spec->rank_count > MGA_MAX_RANKS) return 0;
    if (spec->chunk_count_max < MGA_MIN_CHUNKS || spec->chunk_count_max > MGA_MAX_CHUNKS) return 0;
    if (spec->max_collectives < MGA_MIN_COLLS || spec->max_collectives > MGA_MAX_COLLS) return 0;
    if (spec->max_remote_events < 1 || spec->max_remote_events > MGA_MAX_REMOTE_EVENTS) return 0;
    if (spec->max_send_credits_per_link < 1 ||
        spec->max_send_credits_per_link > MGA_MAX_SEND_CREDITS) return 0;
    if (spec->remote_latency < 0 || spec->remote_latency > MGA_MAX_REMOTE_LATENCY) return 0;
    if (spec->max_scheduler_queue_per_rank < 1 ||
        spec->max_scheduler_queue_per_rank > MGA_MAX_SCHED_QUEUE) return 0;
    return 1;
}

static inline int mga_validate_run_spec(const MgaRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != MGA_ABI_VERSION) return 0;
    if (run->op < MGA_OP_BEGIN_ALLREDUCE || run->op > MGA_OP_FORCE_CREDIT) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MgaProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MgaProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MgaRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MK_MULTIGPU_ALLREDUCE_COMMON_H_
