// file: mk_warp_pipeline_mbarrier_common.h

#ifndef MK_WARP_PIPELINE_MBARRIER_COMMON_H_
#define MK_WARP_PIPELINE_MBARRIER_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MKWP_ABI_VERSION 1

// ----------------------------------------------------------------- limits
#define MKWP_MIN_BUFFERS 1
#define MKWP_MAX_BUFFERS 16
#define MKWP_MIN_WARPS 0
#define MKWP_MAX_WARPS 32
#define MKWP_MIN_MAX_TILES 1
#define MKWP_MAX_MAX_TILES 4096
#define MKWP_MIN_MAX_PENDING 1
#define MKWP_MAX_MAX_PENDING 8192
#define MKWP_MIN_MAX_ROLE_QUEUE 1
#define MKWP_MAX_MAX_ROLE_QUEUE 4096
#define MKWP_MAX_STEPS 8192

// Each buffer owns exactly three barriers (load, compute, store).
#define MKWP_BARRIERS_PER_BUFFER 3
#define MKWP_BAR_LOAD 0
#define MKWP_BAR_COMPUTE 1
#define MKWP_BAR_STORE 2

// Role bit identities (bits in waiting_role_mask, and role ids in events).
#define MKWP_ROLE_LOADER 0
#define MKWP_ROLE_COMPUTE 1
#define MKWP_ROLE_STORER 2

// ----------------------------------------------------------------- op kinds
enum MkwpOpKind {
    MKWP_OP_ENQUEUE_TILE = 0,
    MKWP_OP_LOADER_STEP = 1,
    MKWP_OP_COMPUTE_STEP = 2,
    MKWP_OP_STORER_STEP = 3,
    MKWP_OP_ADVANCE = 4,
    MKWP_OP_CANCEL_TILE = 5,
    MKWP_OP_RESET_BARRIER = 6,
    MKWP_OP_KIND_COUNT = 7
};

// ----------------------------------------------------------------- tile status
enum MkwpTileStatus {
    MKWP_TS_QUEUED = 0,
    MKWP_TS_LOADING = 1,
    MKWP_TS_READY_COMPUTE = 2,
    MKWP_TS_COMPUTING = 3,
    MKWP_TS_READY_STORE = 4,
    MKWP_TS_STORING = 5,
    MKWP_TS_DONE = 6,
    MKWP_TS_CANCELLED = 7
};
// Terminal statuses: DONE, CANCELLED. Nonterminal: everything else.

// ----------------------------------------------------------------- buffer state
enum MkwpBufferState {
    MKWP_BS_EMPTY = 0,
    MKWP_BS_LOAD_INFLIGHT = 1,
    MKWP_BS_LOAD_READY = 2,
    MKWP_BS_COMPUTE_INFLIGHT = 3,
    MKWP_BS_COMPUTE_READY = 4,
    MKWP_BS_STORE_INFLIGHT = 5
};

// ----------------------------------------------------------------- async kinds
enum MkwpAsyncKind {
    MKWP_AK_LOAD_COMPLETE = 0,
    MKWP_AK_COMPUTE_COMPLETE = 1,
    MKWP_AK_STORE_COMPLETE = 2
};

// ----------------------------------------------------------------- event kinds
// In ordinal order. Only the NUMERIC VALUE of each event kind is observable: it
// is absorbed as the first byte (event_kind:u8) of every pipe_event_hash event
// record (see the normative serialization section below). These ordinals are
// part of the contract; do not renumber.
enum MkwpEventKind {
    MKWP_EV_TILE_ENQUEUE = 0,
    MKWP_EV_LOAD_ISSUE = 1,
    MKWP_EV_LOADER_NO_BUFFER = 2,
    MKWP_EV_LOADER_NO_TILE = 3,
    MKWP_EV_MBARRIER_WAIT_LOAD = 4,
    MKWP_EV_MBARRIER_WAIT_COMPUTE = 5,
    MKWP_EV_COMPUTE_ISSUE = 6,
    MKWP_EV_COMPUTE_NO_READY = 7,
    MKWP_EV_STORE_ISSUE = 8,
    MKWP_EV_STORER_NO_READY = 9,
    MKWP_EV_MBARRIER_ARRIVE = 10,
    MKWP_EV_MBARRIER_COMPLETE = 11,
    MKWP_EV_LOAD_COMPLETE = 12,
    MKWP_EV_COMPUTE_COMPLETE = 13,
    MKWP_EV_STORE_COMPLETE = 14,
    MKWP_EV_TILE_DONE = 15,
    MKWP_EV_ASYNC_STALE_DROP = 16,
    MKWP_EV_TILE_CANCEL = 17,
    MKWP_EV_BUFFER_CANCEL_RELEASE = 18,
    MKWP_EV_BARRIER_RESET = 19,
    MKWP_EV_INVALID = 20,
    MKWP_EV_KIND_COUNT = 21
};

// ----------------------------------------------------------------- counters
enum MkwpCountIdx {
    MKWP_C_TILE_ENQUEUE = 0,
    MKWP_C_LOAD_ISSUE = 1,
    MKWP_C_LOADER_NO_BUFFER = 2,
    MKWP_C_LOADER_NO_TILE = 3,
    MKWP_C_MBARRIER_WAIT_LOAD = 4,
    MKWP_C_MBARRIER_WAIT_COMPUTE = 5,
    MKWP_C_COMPUTE_ISSUE = 6,
    MKWP_C_COMPUTE_NO_READY = 7,
    MKWP_C_STORE_ISSUE = 8,
    MKWP_C_STORER_NO_READY = 9,
    MKWP_C_MBARRIER_ARRIVE = 10,
    MKWP_C_MBARRIER_COMPLETE = 11,
    MKWP_C_LOAD_COMPLETE = 12,
    MKWP_C_COMPUTE_COMPLETE = 13,
    MKWP_C_STORE_COMPLETE = 14,
    MKWP_C_TILE_DONE = 15,
    MKWP_C_ASYNC_STALE_DROP = 16,
    MKWP_C_TILE_CANCEL = 17,
    MKWP_C_BUFFER_CANCEL_RELEASE = 18,
    MKWP_C_BARRIER_RESET = 19,
    MKWP_C_INVALID = 20,
    MKWP_COUNT_N = 21
};

/*
CONTRACT: mk_warp_pipeline_mbarrier  (MK6)

A single-SM persistent warp-specialized pipeline simulated deterministically on
the device. Each solution_run applies exactly ONE pipeline operation (encoded in
MkwpRunSpec) to persistent state and emits a fully-determined event stream. All
graded outputs are EXACT integers (no float, no tolerance). The difficulty is
COORDINATION: loader/compute/storer warp roles, mbarrier arrive/wait phase bits,
multi-buffer slot acquire/release, and stale-async classification.

Persistent state (after init/reset):
  clock = 0; event_seq = 0; op_index = 0;
  tile_seq_next = 1; phase_seq_next = 1; async_seq_next = 1.

  Tile table keyed by tile_id holding ALL tiles ever created (terminal tiles are
  retained so id reuse is rejected; only NONTERMINAL tiles are hashed). Fields:
    tile_id, tile_seq, load_bytes, compute_iters, store_bytes, payload_seed,
    status, assigned_buffer (UINT32_MAX if none), result_hash.
  Input tile queue: tile_ids in tile_seq order (append on ENQUEUE).

  Buffer slots, ids 0..buffer_count-1. Fields:
    state, tile_id (0 if none), load_barrier id, compute_barrier id,
    store_barrier id, owner_role (255 if none), last_phase_seq.
  Each buffer b statically owns barriers:
    load_barrier  = b*3 + 0
    compute_barrier = b*3 + 1
    store_barrier = b*3 + 2

  Barrier table, ids 0..barrier_count-1 (barrier_count = buffer_count*3). Fields:
    phase (u64), expected (u32), arrived (u32), completion_done (u8),
    waiting_role_mask (u64), last_arrive_seq (u64).

  Pending async events (FIFO list in creation order). Fields:
    async_kind, due_clock, async_seq, tile_id, buffer_id, barrier_id, phase.

  Role ready queues (ids hold buffer ids), each ordered by (tile_seq, buffer_id):
    compute ready queue, store ready queue.
  (The loader's ready resource is the set of EMPTY buffers; chosen by lowest id.)

Event sequencing:
  event_seq is a u64 counter. Every emitted event consumes the current event_seq
  as its own seq, then event_seq increments (mod 2^64). op_index increments once
  per op, AFTER the op body runs. tile_seq, phase_seq, async_seq counters are
  post-increment (the value used is the pre-increment value).

Operations (MkwpRunSpec):
  ENQUEUE_TILE: a_tile, a_load_bytes, a_compute_iters, a_store_bytes, a_seed.
  LOADER_STEP : a_role_id (loader_id), a_limit.
  COMPUTE_STEP: a_role_id (compute_id), a_limit.
  STORER_STEP : a_role_id (storer_id), a_limit.
  ADVANCE     : a_delta, a_limit (max_async).
  CANCEL_TILE : a_tile.
  RESET_BARRIER: a_barrier.

Per-op semantics (these "Highlights" are the FULL normative semantics; the exact
byte-level serialization of every output is given in the dedicated normative
section that follows this comment block):
  - LOADER_STEP repeats up to limit: lowest EMPTY buffer + oldest noncancelled
    queued tile; init load barrier (phase++, expected=1, arrived=0,
    completion_done=0, last_phase_seq=phase_seq_next++); buffer LOAD_INFLIGHT;
    tile LOADING; create LOAD_COMPLETE async at clock+load_bytes; emit LOAD_ISSUE.
    If no empty buffer -> LOADER_NO_BUFFER stop. If no tile -> LOADER_NO_TILE stop.
  - COMPUTE_STEP: pick ready-compute buffer (smallest tile_seq, tie lowest buffer
    id) from compute ready queue. Its load barrier completion_done must be 1 for
    the current phase; else add compute role bit to load barrier waiting_role_mask,
    emit MBARRIER_WAIT_LOAD, stop. Else: pop from compute queue, buffer
    COMPUTE_INFLIGHT, tile COMPUTING, init compute barrier (next phase, expected
    1, arrived 0, completion 0, last_phase_seq=phase_seq_next++); create
    COMPUTE_COMPLETE async at clock+compute_iters; emit COMPUTE_ISSUE.
  - STORER_STEP: pick ready-store buffer (smallest tile_seq, tie buffer id). Its
    compute barrier must be complete (completion_done==1); else set storer bit in
    compute barrier waiting_role_mask, emit MBARRIER_WAIT_COMPUTE, stop. Else: pop
    store queue, buffer STORE_INFLIGHT, tile STORING, init store barrier (next
    phase, expected 1, arrived 0, completion 0, last_phase_seq=phase_seq_next++);
    create STORE_COMPLETE async at clock+store_bytes; emit STORE_ISSUE.
  - ADVANCE: clock += delta (wraps). Process up to max_async due events ordered by
    (due_clock, async_seq). "Due" means due_clock <= clock. For each:
      * If the tile is gone/terminal, or the buffer no longer holds this tile, or
        the barrier phase != event.phase -> emit ASYNC_STALE_DROP (and remove it).
      * Else mbarrier arrive: arrived++; last_arrive_seq = seq of the
        MBARRIER_ARRIVE event; emit MBARRIER_ARRIVE. If arrived==expected and
        completion_done==0: completion_done=1; emit MBARRIER_COMPLETE. Then by
        kind:
          LOAD_COMPLETE: buffer LOAD_READY, tile READY_COMPUTE, push compute
            ready queue, emit LOAD_COMPLETE.
          COMPUTE_COMPLETE: result_hash = FNV1a64(tile_id, tile_seq, payload_seed,
            buffer_id, clock); buffer COMPUTE_READY, tile READY_STORE, push store
            ready queue, emit COMPUTE_COMPLETE.
          STORE_COMPLETE: tile DONE, buffer EMPTY, clear assignment, owner 255,
            emit STORE_COMPLETE then TILE_DONE.
      The processed event is removed from the pending list whether stale or not.
  - CANCEL_TILE: invalid if absent or terminal. If QUEUED: mark CANCELLED, emit
    TILE_CANCEL. Else (owns a buffer): mark tile CANCELLED, set buffer EMPTY,
    clear tile_id/owner, increment ALL THREE of the buffer's barrier phases
    (load,compute,store) and clear their completion_done/arrived/waiters, emit
    BUFFER_CANCEL_RELEASE. Pending async events are NOT removed; they go stale.
  - RESET_BARRIER: invalid if barrier id out of range OR the owning buffer is
    non-EMPTY (i.e. currently references the barrier). Else phase++, expected=0,
    arrived=0, completion_done=0, clear waiters, emit BARRIER_RESET.

Validity / INVALID:
  Any op whose primary precondition fails emits exactly one INVALID event and
  increments invalid_count, UNLESS the contract specifies a dedicated non-INVALID
  outcome (LOADER_NO_BUFFER/LOADER_NO_TILE/COMPUTE_NO_READY/STORER_NO_READY/
  MBARRIER_WAIT_*). ENQUEUE_TILE invalid if id exists nonterminal, table full, or
  any of load_bytes/compute_iters/store_bytes is zero. LOADER/COMPUTE/STORER_STEP
  invalid if role_id >= role warp count or limit==0. ADVANCE with delta==0 is
  VALID. CANCEL invalid if tile absent/terminal. RESET_BARRIER invalid if id OOB
  or buffer non-EMPTY.

Outputs after EVERY op (exact integers):
  counts[MKWP_COUNT_N], op_index_out, clock_out, event_seq_out,
  pipe_event_hash (persistent running FNV-1a-64 over the emitted event stream),
  buffer_hash, barrier_hash, tile_hash, async_hash (snapshot FNV hashes),
  state_checksum (master FNV combining all of the above + counts).

Rules:
  - solution_init may allocate persistent state.
  - solution_run may NOT call cudaMalloc/cudaFree.
  - All clocks and sequence counters wrap modulo 2^64.
*/

// === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section is fully self-contained: it specifies every value graded after
// each solution_run and the EXACT byte-level recipe by which it is produced,
// derived directly from the reference. A solver reading only this header can
// reproduce every checksum bit-for-bit. There is no separate "section 4",
// oracle, or design doc; everything normative is here.
//
// ---------------------------------------------------------------------------
// 0. FNV-1a-64 PRIMITIVE  (the ONLY hash used anywhere)
// ---------------------------------------------------------------------------
//   offset basis : 1469598103934665603  (0x14650FB0739D0383)
//   prime        :       1099511628211  (0x00000100000001b3)
//   byte step    : h ^= byte;  h *= prime;   (XOR-then-multiply; arithmetic in
//                  unsigned 64-bit, wrapping mod 2^64)
//   "absorb a field of width W" = feed the field's W bytes in LITTLE-ENDIAN host
//   order (i.e. the raw in-memory bytes of an x86-64 little-endian value), least
//   significant byte first, each through the byte step in order.
//   A hash is "seeded with the offset basis" unless stated to start from a
//   running value. All multi-byte integers below are little-endian.
//   Field widths used: u8 = 1 byte, u32 = 4 bytes, u64 = 8 bytes. A signed value
//   is absorbed as the unsigned integer of the stated width with identical bits
//   (e.g. status:int32 -> cast to u8 then absorb 1 byte; op_index:int32 ->
//   reinterpret as u32, absorb 4 bytes; a count:int64 -> reinterpret as u64,
//   absorb 8 bytes). Sentinels: UINT32_MAX = 0xFFFFFFFF, UINT64_MAX =
//   0xFFFFFFFFFFFFFFFF.
//
// ---------------------------------------------------------------------------
// 1. SCALAR OUTPUTS (exact integers)
// ---------------------------------------------------------------------------
//   op_index_out : the op_index value BEFORE this op's post-increment (i.e. the
//                  0-based index of this op). op_index increments by 1 AFTER the
//                  op body runs and AFTER op_index_out is captured.
//   clock_out    : clock after the op (only ADVANCE changes clock: clock+=delta,
//                  wrapping mod 2^64).
//   event_seq_out: event_seq after the op (total events emitted since reset).
//   counts[i]    : the 21 int64 counters (indices = MkwpCountIdx). Each emitted
//                  event increments exactly the matching counter (see section 3).
//
// ---------------------------------------------------------------------------
// 2. pipe_event_hash  (PERSISTENT running event-stream hash)
// ---------------------------------------------------------------------------
//   Seed at reset = FNV offset basis (1469598103934665603). It is NEVER reseeded
//   between ops; it is a single running FNV state carried across all ops until
//   solution_reset. For EACH emitted event, in emission order, the following 10
//   fields are absorbed into the running pipe_event_hash IN THIS EXACT ORDER:
//
//     1. event_kind        u8   (MkwpEventKind ordinal)
//     2. event_seq         u64  (this event's own seq = event_seq BEFORE its
//                                post-increment)
//     3. op_index          u32  (current op_index, i.e. BEFORE the op's
//                                post-increment; same value for all events of one op)
//     4. clock             u64  (current clock at emission time)
//     5. role_id           u32  (role/warp id, or UINT32_MAX if not applicable)
//     6. tile_id           u64  (tile id, or 0 if not applicable)
//     7. buffer            u32  (buffer id, or UINT32_MAX if not applicable)
//     8. barrier           u32  (barrier id, or UINT32_MAX if not applicable)
//     9. phase             u64  (barrier phase, or UINT64_MAX if not applicable)
//    10. aux               u64  (per-event-kind auxiliary value, section 3)
//
//   After absorbing all 10 fields, event_seq increments by 1. Note clock and
//   op_index are read at emission time; for ADVANCE, clock has ALREADY been
//   advanced before any async event is processed/emitted.
//
// ---------------------------------------------------------------------------
// 3. EVENT CATALOG  (per-kind field values + aux + counter; emission order)
// ---------------------------------------------------------------------------
//   Notation: fields are (role, tile_id, buffer, barrier, phase, aux). "-" means
//   the sentinel for that width (role/buffer/barrier=UINT32_MAX, phase=UINT64_MAX,
//   tile_id=0). Each event also increments counts[<matching index>] by 1.
//
//   TILE_ENQUEUE (0)         role -, tile=tile_id, buf -, bar -, phase -,
//                            aux = tile_seq assigned to this tile.
//   LOAD_ISSUE (1)           role=loader_id, tile=tile_id, buf=b, bar=load_barrier,
//                            phase=load_barrier.phase (post-increment value),
//                            aux = last_phase_seq assigned (phase_seq pre-incr value).
//   LOADER_NO_BUFFER (2)     role=loader_id, tile 0, buf -, bar -, phase -, aux 0.
//   LOADER_NO_TILE (3)       role=loader_id, tile 0, buf -, bar -, phase -, aux 0.
//   MBARRIER_WAIT_LOAD (4)   role=compute_id, tile=buffer.tile_id, buf=b,
//                            bar=load_barrier, phase=load_barrier.phase, aux 0.
//   MBARRIER_WAIT_COMPUTE(5) role=storer_id, tile=buffer.tile_id, buf=b,
//                            bar=compute_barrier, phase=compute_barrier.phase, aux 0.
//   COMPUTE_ISSUE (6)        role=compute_id, tile=tile_id, buf=b,
//                            bar=compute_barrier, phase=compute_barrier.phase,
//                            aux = last_phase_seq (phase_seq pre-incr value).
//   COMPUTE_NO_READY (7)     role=compute_id, tile 0, buf -, bar -, phase -, aux 0.
//   STORE_ISSUE (8)          role=storer_id, tile=tile_id, buf=b,
//                            bar=store_barrier, phase=store_barrier.phase,
//                            aux = last_phase_seq (phase_seq pre-incr value).
//   STORER_NO_READY (9)      role=storer_id, tile 0, buf -, bar -, phase -, aux 0.
//   MBARRIER_ARRIVE (10)     role -, tile=tile_id, buf=buffer_id, bar=barrier_id,
//                            phase=event.phase, aux = barrier.arrived AFTER the
//                            increment (the new arrived count).
//   MBARRIER_COMPLETE (11)   role -, tile=tile_id, buf=buffer_id, bar=barrier_id,
//                            phase=event.phase, aux 0.
//   LOAD_COMPLETE (12)       role -, tile=tile_id, buf=buffer_id, bar=barrier_id,
//                            phase=event.phase, aux 0.
//   COMPUTE_COMPLETE (13)    role -, tile=tile_id, buf=buffer_id, bar=barrier_id,
//                            phase=event.phase, aux = result_hash (section 5).
//   STORE_COMPLETE (14)      role -, tile=tile_id, buf=buffer_id, bar=barrier_id,
//                            phase=event.phase, aux 0.
//   TILE_DONE (15)           role -, tile=tile_id, buf=buffer_id, bar -, phase -,
//                            aux 0.   (emitted immediately after STORE_COMPLETE)
//   ASYNC_STALE_DROP (16)    role -, tile=event.tile_id, buf=event.buffer_id,
//                            bar=event.barrier_id, phase=event.phase,
//                            aux = (u64)event.async_kind.
//   TILE_CANCEL (17)         role -, tile=tile_id, buf -, bar -, phase -, aux 0.
//   BUFFER_CANCEL_RELEASE(18) role -, tile=tile_id, buf=b (released buffer),
//                            bar -, phase -, aux 0.
//   BARRIER_RESET (19)       role -, tile 0, buf -, bar=barrier_id,
//                            phase=barrier.phase (post-increment value), aux 0.
//   INVALID (20)             role -, tile 0, buf -, bar -, phase -, aux 0.
//
//   Multi-event ops emit in this fixed order:
//     * async LOAD arrive completes: MBARRIER_ARRIVE, [MBARRIER_COMPLETE,]
//       LOAD_COMPLETE.
//     * async COMPUTE arrive completes: MBARRIER_ARRIVE, [MBARRIER_COMPLETE,]
//       COMPUTE_COMPLETE.
//     * async STORE arrive completes: MBARRIER_ARRIVE, [MBARRIER_COMPLETE,]
//       STORE_COMPLETE, TILE_DONE.
//     (MBARRIER_COMPLETE is emitted only when arrived==expected and
//      completion_done was 0.)
//   A single LOADER/COMPUTE/STORER_STEP with a_limit>1 emits one issue (or one
//   stop event) per iteration until it hits a stop condition (NO_BUFFER/NO_TILE/
//   NO_READY/WAIT_*), which terminates the loop after emitting that one event.
//   A single ADVANCE emits the full event sequence of each processed async, in
//   the order the asyncs are selected (section 4.E).
//
// ---------------------------------------------------------------------------
// 4. PER-OP STATE TRANSITIONS (exact; mirrors the semantics highlights above)
// ---------------------------------------------------------------------------
//   Counters used: tile_seq_next, phase_seq_next, async_seq_next all start at 1
//   and are POST-increment (the emitted/stored value is the pre-increment value).
//
//   A. ENQUEUE_TILE(tile_id, lb, ci, sb, seed):
//      INVALID if (a tile with tile_id exists AND is nonterminal) OR
//      (number of tiles currently in the table >= max_tiles) OR lb==0 OR ci==0
//      OR sb==0. The live-tile count for the "table full" test is the count of
//      tiles retained in the table (terminal tiles are RETAINED, so they count;
//      the reference frees a same-id terminal slot just before alloc so a reused
//      id does not double-count). Else: create tile {tile_seq=tile_seq_next++,
//      load=lb, compute=ci, store=sb, seed, status=QUEUED, assigned=UINT32_MAX,
//      result_hash=0}; append tile_id to input queue; emit TILE_ENQUEUE(aux=tile_seq).
//
//   B. LOADER_STEP(loader_id, limit):
//      INVALID if loader_id<0 || loader_id>=loader_warps || limit==0.
//      Loop i in [0,limit): pick lowest-id EMPTY buffer b; if none ->
//      LOADER_NO_BUFFER, stop. Pop input-queue head, discarding any head tile
//      that is not status==QUEUED (cancelled/already-loaded ids are dropped),
//      until a QUEUED tile is found; if none -> LOADER_NO_TILE, stop. Assign:
//      tile.assigned=b, tile.status=LOADING, buffer.tile_id=tid,
//      buffer.owner=LOADER(0). Init load barrier (id = b*3+0): phase+=1,
//      expected=1, arrived=0, completion_done=0, waiting_role_mask=0.
//      last_phase_seq=phase_seq_next++. buffer.state=LOAD_INFLIGHT. Push async
//      {kind=LOAD_COMPLETE, due=clock+tile.load, seq=async_seq_next++, tile=tid,
//      buf=b, bar=load_barrier, phase=load_barrier.phase} to the END of pending.
//      Emit LOAD_ISSUE.
//
//   C. COMPUTE_STEP(compute_id, limit):
//      INVALID if compute_id<0 || compute_id>=compute_warps || limit==0.
//      Loop: select compute-ready buffer with smallest (tile.tile_seq, buffer_id)
//      [tile_seq primary asc, buffer_id tiebreak asc]; if queue empty ->
//      COMPUTE_NO_READY, stop. Let lbar=buffer.load_barrier. If
//      lbar.completion_done!=1: set bit (1<<ROLE_COMPUTE=1<<1) in
//      lbar.waiting_role_mask, emit MBARRIER_WAIT_LOAD, stop. Else: remove b from
//      compute-ready queue; buffer.state=COMPUTE_INFLIGHT, buffer.owner=COMPUTE(1),
//      tile.status=COMPUTING. Init compute barrier (id=b*3+1): phase+=1,
//      expected=1, arrived=0, completion_done=0, waiting_role_mask=0.
//      last_phase_seq=phase_seq_next++. Push async {kind=COMPUTE_COMPLETE,
//      due=clock+tile.compute, seq=async_seq_next++, tile, buf=b,
//      bar=compute_barrier, phase=compute_barrier.phase}. Emit COMPUTE_ISSUE.
//
//   D. STORER_STEP(storer_id, limit):
//      INVALID if storer_id<0 || storer_id>=storer_warps || limit==0.
//      Loop: select store-ready buffer with smallest (tile.tile_seq, buffer_id);
//      if empty -> STORER_NO_READY, stop. Let cbar=buffer.compute_barrier. If
//      cbar.completion_done!=1: set bit (1<<ROLE_STORER=1<<2) in
//      cbar.waiting_role_mask, emit MBARRIER_WAIT_COMPUTE, stop. Else: remove b
//      from store-ready queue; buffer.state=STORE_INFLIGHT,
//      buffer.owner=STORER(2), tile.status=STORING. Init store barrier
//      (id=b*3+2): phase+=1, expected=1, arrived=0, completion_done=0,
//      waiting_role_mask=0. last_phase_seq=phase_seq_next++. Push async
//      {kind=STORE_COMPLETE, due=clock+tile.store, seq=async_seq_next++, tile,
//      buf=b, bar=store_barrier, phase=store_barrier.phase}. Emit STORE_ISSUE.
//
//   E. ADVANCE(delta, max_async):
//      clock += delta (wraps mod 2^64). Then repeat up to max_async times:
//      among all pending asyncs with due_clock <= clock ("due"), select the one
//      with smallest (due_clock, async_seq) [due_clock primary asc, async_seq
//      tiebreak asc]; if none due, stop. Remove it from pending (compacting the
//      flat list, preserving relative order of the rest), then PROCESS it
//      (section 4.F). Each processed event counts toward max_async whether stale
//      or not. (delta==0 is VALID and may still deliver already-due events.)
//
//   F. PROCESS one async {kind, tile_id, buffer_id=buf, barrier_id=bar, phase}:
//      STALE TEST: stale if find(tile_id) is absent OR that tile is terminal OR
//      buffers[buf].tile_id != tile_id OR barriers[bar].phase != phase. If stale:
//      emit ASYNC_STALE_DROP (aux=(u64)kind); done. Else mbarrier arrive:
//      barriers[bar].arrived += 1; barriers[bar].last_arrive_seq = current
//      event_seq (the seq the MBARRIER_ARRIVE event is about to consume); emit
//      MBARRIER_ARRIVE (aux=arrived). If arrived==expected && completion_done==0:
//      completion_done=1; emit MBARRIER_COMPLETE. Then by kind:
//        LOAD_COMPLETE   : buffer.state=LOAD_READY, buffer.owner=255,
//                          tile.status=READY_COMPUTE, push buf into compute-ready
//                          queue (kept ordered by (tile_seq, buffer_id)),
//                          emit LOAD_COMPLETE.
//        COMPUTE_COMPLETE: tile.result_hash = result_hash(tile_id, tile.tile_seq,
//                          tile.seed, buf, clock) [section 5]; buffer.state=
//                          COMPUTE_READY, buffer.owner=255, tile.status=
//                          READY_STORE, push buf into store-ready queue,
//                          emit COMPUTE_COMPLETE (aux=result_hash).
//        STORE_COMPLETE  : tile.status=DONE, tile.assigned=UINT32_MAX,
//                          buffer.state=EMPTY, buffer.tile_id=0, buffer.owner=255,
//                          emit STORE_COMPLETE then TILE_DONE.
//
//   G. CANCEL_TILE(tile_id):
//      INVALID if tile absent OR terminal. If status==QUEUED: status=CANCELLED,
//      emit TILE_CANCEL (the id stays in the input queue and is later skipped by
//      the loader). Else (owns buffer b=tile.assigned): tile.status=CANCELLED,
//      tile.assigned=UINT32_MAX; erase b from BOTH compute-ready and store-ready
//      queues; buffer.state=EMPTY, buffer.tile_id=0, buffer.owner=255; for each
//      of the buffer's three barriers (load=b*3+0, compute=b*3+1, store=b*3+2) in
//      that order: phase+=1, arrived=0, completion_done=0, waiting_role_mask=0
//      (expected is NOT touched here). Emit BUFFER_CANCEL_RELEASE(buf=b). Pending
//      asyncs are NOT removed; they go stale on later ADVANCE.
//
//   H. RESET_BARRIER(barrier_id):
//      INVALID if barrier_id<0 || barrier_id>=barrier_count OR the owning buffer
//      (owning_buf = barrier_id / 3) is in range AND buffers[owning_buf].state !=
//      EMPTY. Else: phase+=1, expected=0, arrived=0, completion_done=0,
//      waiting_role_mask=0; emit BARRIER_RESET (phase=post-increment value).
//
//   INVALID emission: emit_invalid increments counts[INVALID] and emits exactly
//   one INVALID event with all sentinel fields (role/buf/bar=UINT32_MAX,
//   tile=0, phase=UINT64_MAX, aux=0).
//
// ---------------------------------------------------------------------------
// 5. result_hash  (per-tile, recomputed at COMPUTE_COMPLETE)
// ---------------------------------------------------------------------------
//   A fresh FNV-1a-64 (seed=offset basis) absorbing exactly these five u64s,
//   each little-endian, in this order:
//     tile_id, tile_seq, payload_seed, buffer_id, clock
//   where clock is the CURRENT clock at the moment the COMPUTE_COMPLETE async is
//   processed. This 64-bit value is stored in tile.result_hash and is the aux of
//   the COMPUTE_COMPLETE event AND is absorbed into tile_hash (section 6.C).
//
// ---------------------------------------------------------------------------
// 6. SNAPSHOT HASHES (recomputed fresh AFTER each op; seed = offset basis)
// ---------------------------------------------------------------------------
//   Each starts from a fresh FNV state (offset basis) every op.
//
//   A. buffer_hash: iterate buffers b = 0 .. buffer_count-1 (ascending id).
//      For each absorb, in order:
//        b              u32  (the buffer index itself)
//        state          u8   (cast from int32 state)
//        tile_id        u64
//        load_barrier   u32
//        compute_barrier u32
//        store_barrier  u32
//        owner_role     u8   (255 if none)
//        last_phase_seq u64
//
//   B. barrier_hash: iterate barriers b = 0 .. barrier_count-1 (ascending id).
//      For each absorb, in order:
//        b                 u32 (the barrier index itself)
//        phase             u64
//        expected          u32
//        arrived           u32
//        completion_done   u8
//        waiting_role_mask u64
//        last_arrive_seq   u64
//
//   C. tile_hash: over NONTERMINAL tiles ONLY (status != DONE && != CANCELLED),
//      visited in ASCENDING tile_id order. For each absorb, in order:
//        tile_id        u64
//        tile_seq       u64
//        load_bytes     u64
//        compute_iters  u64
//        store_bytes    u64
//        status         u8  (cast from int32)
//        assigned_buffer u32 (UINT32_MAX if none)
//        result_hash    u64
//      Terminal tiles are skipped entirely (they never contribute to tile_hash).
//
//   D. async_hash: over pending asyncs in PENDING (creation/flat) order, i.e. the
//      order they were appended minus any removed/compacted entries. For each
//      absorb, in order:
//        async_kind u8  (cast from int32)
//        due_clock  u64
//        async_seq  u64
//        tile_id    u64
//        buffer_id  u32
//        barrier_id u32
//        phase      u64
//
// ---------------------------------------------------------------------------
// 7. state_checksum  (MASTER; recomputed fresh AFTER each op)
// ---------------------------------------------------------------------------
//   IMPORTANT: state_checksum is computed AFTER op_index has been post-incremented
//   for this op, so the op_index it absorbs is (this_op + 1) = the NEXT op index.
//   (By contrast op_index_out reports this_op, the pre-increment value.)
//   Fresh FNV (seed=offset basis) absorbing, in this exact order:
//     clock          u64
//     event_seq      u64
//     op_index       u32  (POST-increment value = this_op + 1; see note above)
//     tile_seq_next  u64
//     phase_seq_next u64
//     async_seq_next u64
//     pipe_event_hash u64 (current running value)
//     buffer_hash    u64
//     barrier_hash   u64
//     tile_hash      u64
//     async_hash     u64
//     then for i = 0 .. MKWP_COUNT_N-1: counts[i] u64 (each int64 counter
//       reinterpreted as u64), in ascending index order.
//
// ---------------------------------------------------------------------------
// 8. ORDERING / TIE-BREAK SUMMARY
// ---------------------------------------------------------------------------
//   * Loader picks the LOWEST-id EMPTY buffer; input tiles are consumed FIFO from
//     the input queue head (skipping non-QUEUED heads).
//   * Compute/store ready queues are ordered by (tile_seq asc, buffer_id asc);
//     the minimum is always taken.
//   * ADVANCE selects due asyncs by (due_clock asc, async_seq asc).
//   * tile_hash iterates ascending tile_id; buffer_hash and barrier_hash iterate
//     ascending index; async_hash iterates pending creation order.
//   * All barrier ids for buffer b: load=b*3+0, compute=b*3+1, store=b*3+2.
//
// === END CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION ===

// ----------------------------------------------------------------- ABI structs
struct alignas(8) MkwpProblemSpec {
    int32_t abi_version;
    int32_t buffer_count;
    int32_t loader_warps;
    int32_t compute_warps;
    int32_t storer_warps;
    int32_t barrier_count;      // must equal buffer_count * 3
    int32_t max_tiles;
    int32_t max_pending_async;
    int32_t max_role_queue;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[5];
};

struct alignas(8) MkwpRunSpec {
    int32_t abi_version;
    int32_t op_kind;            // MkwpOpKind
    int32_t step_id;
    int32_t a_role_id;          // loader/compute/storer id
    int32_t a_limit;            // step limit / max_async
    int32_t a_barrier;          // RESET_BARRIER barrier id
    int32_t reserved0;
    int32_t reserved1;
    uint64_t a_tile;            // tile id
    uint64_t a_load_bytes;
    uint64_t a_compute_iters;
    uint64_t a_store_bytes;
    uint64_t a_seed;            // payload_seed
    uint64_t a_delta;           // ADVANCE delta
};

struct alignas(8) MkwpInputs {
    // All operands travel through MkwpRunSpec; reserved for ABI symmetry.
    const void* reserved;
};

struct alignas(8) MkwpOutputs {
    int64_t* counts;            // [MKWP_COUNT_N]
    int32_t* op_index_out;      // [1]
    uint64_t* clock_out;        // [1]
    uint64_t* event_seq_out;    // [1]
    uint64_t* pipe_event_hash;  // [1]
    uint64_t* buffer_hash;      // [1]
    uint64_t* barrier_hash;     // [1]
    uint64_t* tile_hash;        // [1]
    uint64_t* async_hash;       // [1]
    uint64_t* state_checksum;   // [1]
};

static_assert(sizeof(MkwpProblemSpec) == 64, "MkwpProblemSpec layout drift");
static_assert(sizeof(MkwpRunSpec) == 80, "MkwpRunSpec layout drift");
static_assert(sizeof(MkwpOutputs) == 80, "MkwpOutputs layout drift");

static inline int mkwp_validate_problem_spec(const MkwpProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MKWP_ABI_VERSION) return 0;
    if (spec->buffer_count < MKWP_MIN_BUFFERS || spec->buffer_count > MKWP_MAX_BUFFERS) return 0;
    if (spec->loader_warps < MKWP_MIN_WARPS || spec->loader_warps > MKWP_MAX_WARPS) return 0;
    if (spec->compute_warps < MKWP_MIN_WARPS || spec->compute_warps > MKWP_MAX_WARPS) return 0;
    if (spec->storer_warps < MKWP_MIN_WARPS || spec->storer_warps > MKWP_MAX_WARPS) return 0;
    if (spec->barrier_count != spec->buffer_count * MKWP_BARRIERS_PER_BUFFER) return 0;
    if (spec->max_tiles < MKWP_MIN_MAX_TILES || spec->max_tiles > MKWP_MAX_MAX_TILES) return 0;
    if (spec->max_pending_async < MKWP_MIN_MAX_PENDING ||
        spec->max_pending_async > MKWP_MAX_MAX_PENDING) return 0;
    if (spec->max_role_queue < MKWP_MIN_MAX_ROLE_QUEUE ||
        spec->max_role_queue > MKWP_MAX_MAX_ROLE_QUEUE) return 0;
    if (spec->max_steps < 1 || spec->max_steps > MKWP_MAX_STEPS) return 0;
    return 1;
}

static inline int mkwp_validate_run_spec(const MkwpRunSpec* run, const MkwpProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != MKWP_ABI_VERSION) return 0;
    if (run->op_kind < 0 || run->op_kind >= MKWP_OP_KIND_COUNT) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MkwpProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MkwpProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MkwpRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MK_WARP_PIPELINE_MBARRIER_COMMON_H_
