// file: mk_pipeline_scoreboard_common.h

#ifndef MK_PIPELINE_SCOREBOARD_COMMON_H_
#define MK_PIPELINE_SCOREBOARD_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MKPS_ABI_VERSION 1

/* Bounds. */
#define MKPS_MIN_BUFFERS 1
#define MKPS_MAX_BUFFERS 64
#define MKPS_MIN_TILES 1
#define MKPS_MAX_TILES 256
#define MKPS_MIN_INSTRS 1
#define MKPS_MAX_INSTRS 256
#define MKPS_MAX_READS 8
#define MKPS_MAX_WRITES 8
#define MKPS_MIN_WINDOW 1
#define MKPS_MAX_WINDOW 64
#define MKPS_MIN_PENDING 1
#define MKPS_MAX_PENDING 256
#define MKPS_MIN_COUNTERS 1
#define MKPS_MAX_COUNTERS 64
#define MKPS_MAX_BATCH 4096
#define MKPS_MAX_STEPS 64

/* Operation kinds (input op stream). */
#define MKPS_OP_ENQUEUE       0
#define MKPS_OP_ISSUE_LOADS   1
#define MKPS_OP_ISSUE_COMPUTE 2
#define MKPS_OP_ISSUE_STORES  3
#define MKPS_OP_ADVANCE       4
#define MKPS_OP_CANCEL        5
#define MKPS_OP_HOST_COUNTER  6

/* Instruction status. */
#define MKPS_ST_QUEUED         0
#define MKPS_ST_LOAD_ISSUED    1
#define MKPS_ST_LOAD_READY     2
#define MKPS_ST_COMPUTE_ISSUED 3
#define MKPS_ST_STORE_ISSUED   4
#define MKPS_ST_DONE           5
#define MKPS_ST_CANCELLED      6

/* Buffer states. */
#define MKPS_BUF_FREE        0
#define MKPS_BUF_LOADING     1
#define MKPS_BUF_READY_READ  2
#define MKPS_BUF_READY_WRITE 3
#define MKPS_BUF_COMPUTING   4
#define MKPS_BUF_STORING     5
#define MKPS_BUF_RESIDENT    6

/* Pending op kinds. */
#define MKPS_PEND_LOAD_DONE    0
#define MKPS_PEND_COMPUTE_DONE 1
#define MKPS_PEND_STORE_DONE   2

/* Buffer role within an instruction's assigned_buffers list (used at LOAD). */
#define MKPS_ROLE_READ  0
#define MKPS_ROLE_WRITE 1
#define MKPS_ROLE_SCRATCH 2

/* Event kinds, in the exact order of the contract enumeration. */
#define MKEV_INSTR_ENQUEUE          0
#define MKEV_LOAD_ISSUE             1
#define MKEV_LOAD_READY_INLINE      2
#define MKEV_LOAD_HAZARD_STALL      3
#define MKEV_LOAD_DONE              4
#define MKEV_INSTR_LOAD_READY       5
#define MKEV_COMPUTE_ISSUE          6
#define MKEV_COMPUTE_LOAD_WAIT      7
#define MKEV_COMPUTE_DONE           8
#define MKEV_STORE_ISSUE            9
#define MKEV_TILE_STORE             10
#define MKEV_READ_RELEASE           11
#define MKEV_COUNTER_INC            12
#define MKEV_INSTR_DONE             13
#define MKEV_OP_STALE_DROP          14
#define MKEV_BUFFER_CANCEL_RELEASE  15
#define MKEV_INSTR_CANCEL           16
#define MKEV_HOST_COUNTER_INC       17
#define MKEV_INVALID                18

/* Sentinels. */
#define MKPS_NO_U32 0xFFFFFFFFu
#define MKPS_NO_U64 0xFFFFFFFFFFFFFFFFULL

/*
CONTRACT: mk_pipeline_scoreboard (MK8)

Cross-Instruction Software-Pipeline Buffer Scoreboard. A persistent megakernel
software-pipeline manager. It issues future loads early, tracks in-flight
buffers, enforces RAW/WAR/WAW tile hazards, releases buffer pages at last use,
and increments dependency counters after stores. All graded outputs are exact
integers; difficulty is COORDINATION (in-flight load tracking, double/triple
buffer slot acquire/release, hazard stalls across instructions), graded via
FNV-1a-64 checksums over the buffer/hazard timeline.

PERSISTENT STATE (all u64 scalars wrap modulo 2^64)
  clock            starts 0.
  event_seq        starts 0 ; stamped on the NEXT emitted event (post-increment).
  instr_seq_next   starts 1 ; the instr_seq of the NEXT enqueued instruction.
  op_seq_next      starts 1 ; the op_seq of the NEXT created pending op.
  version_seq_next starts 1 ; the NEXT write version reserved at load issue.

  Instruction queue (ordered by instr_seq):
    instr_id; instr_seq; reads[]; writes[]; scratch_pages; load/compute/store
    latency; out_counter (or NO); payload_seed; status; assigned_buffers[];
    read_versions[]; write_versions[]; compute_done_flag; last_use_released.
  Tile scoreboard per tile:
    current_version; writer_instr (0=none); pending_reader_count; last_store_seq;
    resident_buffer (or NO); dirty.
  Buffer table per buffer:
    state; tile (or NO); version; owner_instr (0=none); pin_count; last_touch_seq.
  Pending ops queue: kind; due_clock; op_seq; instr_id; buffer_id (or NO); active.
  Counter array.

OPERATION STREAM (per step, processed in input order i = 0..batch_size-1)
  The op_index passed to every event emitted while processing input row i is i
  (the within-batch position, NOT a global counter). See the normative CONTRACT
  section below for the full, self-contained per-op semantics and exact output
  serialization.

OUTPUTS (after every step; counts are cumulative, reset only on solution_reset)
  counts[MKPS_COUNT_FIELDS] : one i64 counter per event class.
  pipeline_event_hash[0] : running FNV-1a-64 over all event tuples (see CONTRACT).
  instr_hash[0]    : FNV-1a-64 over NONTERMINAL instructions by instr_seq ascending.
  tile_hash[0]     : FNV-1a-64 over tiles by id ascending.
  buffer_hash[0]   : FNV-1a-64 over buffers by id ascending.
  pending_hash[0]  : FNV-1a-64 over active pending ops in queue (storage) order.
  counter_hash[0]  : FNV-1a-64 over counters by id ascending.
  event_seq_out[0] : event_seq after the step.
  state_checksum[0]: FNV-1a-64 over scalars + all sub-hashes + counts.

RULES
  - solution_init may allocate persistent state.
  - solution_run may not call cudaMalloc/cudaFree; it may use the workspace.
  - solution_reset restores the initial persistent state.
*/

/* === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
 *
 * This section is the COMPLETE and SELF-CONTAINED specification. Everything a
 * solver needs to reproduce every graded checksum bit-for-bit is here; nothing
 * is deferred to any other file. It is derived exactly from the reference
 * (single-lane, block 0 / thread 0) and is identical across all implementations.
 *
 * ------------------------------------------------------------------------------
 * 0. FNV-1a-64 PRIMITIVE
 * ------------------------------------------------------------------------------
 *   offset basis = 1469598103934665603 (0x14650FB0739D0383)
 *   prime        = 1099511628211       (0x100000001B3)
 *   absorb one byte b into running hash h:  h = (h XOR b) * prime  (mod 2^64).
 *   "absorb value V of width W" means: take the W bytes of V in NATIVE memory
 *   byte order on the device, which for the sm_120 target is LITTLE-ENDIAN, and
 *   absorb them one at a time from lowest address (least significant byte) to
 *   highest (most significant byte). All multi-byte fields below are absorbed
 *   little-endian. The fixed-width C types used for each field (u8/u32/u64/i32/
 *   i64) are normative and define W.
 *   Every hash is seeded with h = offset basis (1469598103934665603) UNLESS
 *   stated otherwise. The running pipeline_event_hash is ALSO seeded with the
 *   offset basis at reset and then carried across steps (never re-seeded).
 *
 * ------------------------------------------------------------------------------
 * 1. PERSISTENT SCALARS (all u64, wrap mod 2^64). Reset/init values:
 * ------------------------------------------------------------------------------
 *   clock            = 0   advanced only by ADVANCE.
 *   event_seq        = 0   stamped on the NEXT emitted event, post-incremented.
 *   instr_seq_next   = 1   instr_seq assigned to the NEXT enqueued instruction.
 *   op_seq_next      = 1   op_seq assigned to the NEXT pushed pending op.
 *   version_seq_next = 1   NEXT write-version reserved at load issue.
 *   touch_seq_next   = 1   NEXT buffer last_touch_seq value.
 *
 * ------------------------------------------------------------------------------
 * 2. EVENT EMISSION (defines pipeline_event_hash and event_seq)
 * ------------------------------------------------------------------------------
 *   Every event, in emission order, absorbs EXACTLY these fields IN THIS ORDER
 *   into the running pipeline_event_hash (h carried across the whole run):
 *     kind        u8    (one of MKEV_* below)
 *     event_seq   u64   (the CURRENT event_seq, before increment)
 *     op_index    u32   (the within-batch input row index i)
 *     clock       u64   (current clock)
 *     instr_id    u64   (0 when not applicable)
 *     tile        u32   (MKPS_NO_U32 when not applicable)
 *     buffer      u32   (MKPS_NO_U32 when not applicable)
 *     version     u64   (0 when not applicable)
 *     counter_id  u32   (MKPS_NO_U32 when not applicable)
 *     aux         u64   (0 unless the event kind defines it; see below)
 *   After absorbing, event_seq += 1. NOTHING ELSE is absorbed.
 *   event_seq_out[0] = event_seq after the whole step's batch is processed.
 *
 * ------------------------------------------------------------------------------
 * 3. INSTRUCTION ORDER / TERMINOLOGY
 * ------------------------------------------------------------------------------
 *   Instructions live in a slot array of size max_instrs. "instr_seq order" =
 *   ascending instr_seq. Only ENQUEUE assigns instr_seq (1,2,3,...), so
 *   instr_seq order == enqueue order. The canonical scan order ("order[]") is:
 *   all USED slots sorted ascending by instr_seq. "Nonterminal" = used AND
 *   status not in {DONE, CANCELLED}. "Earlier" = strictly smaller instr_seq.
 *
 * ------------------------------------------------------------------------------
 * 4. ENQUEUE (op=MKPS_OP_ENQUEUE). Inputs: id=a0, read_count=a1, write_count=a2,
 *    scratch_pages=a3, load_lat=a4, compute_lat=a5, store_lat=a6,
 *    out_counter=a7 (MKPS_NO_U32 = none), payload_seed=a8;
 *    read tiles = tiles[i*16 + 0..read_count-1],
 *    write tiles = tiles[i*16 + 8 + 0..write_count-1].
 *   VALIDITY (evaluate in this order; ANY failure => INVALID, no other mutation):
 *     (a) id == 0                                            -> invalid
 *     (b) id already present in ANY used slot (terminal or not) -> invalid
 *     (c) no free instruction slot (all max_instrs used)     -> invalid
 *     (d) read_count > max_reads_per_instr OR
 *         write_count > max_writes_per_instr                 -> invalid
 *     (e) any read tile id >= tile_count                     -> invalid
 *     (f) any write tile id >= tile_count                    -> invalid
 *     (g) duplicate WRITE tile id (any repeat among writes)  -> invalid
 *         (duplicate READ tiles are ALLOWED.)
 *     (h) scratch_pages + (#distinct read tiles) + write_count > buffer_count
 *                                                            -> invalid
 *     out_counter is NOT validated here.
 *   On INVALID: counts[MKC_invalid_count] += 1; emit MKEV_INVALID with
 *     instr_id=id, tile=NO, buffer=NO, version=0, counter=NO, aux=0.
 *   On VALID: take the LOWEST-index free slot. Set used=1, instr_id=id,
 *     instr_seq = instr_seq_next++ , record read_count/write_count/reads/writes/
 *     scratch_pages/latencies/out_counter/payload_seed, status=QUEUED,
 *     nbuf=0, compute_done_flag=0, last_use_released=0. For EACH read occurrence
 *     (including duplicates) tiles[read].pending_reader_count += 1.
 *     counts[MKC_instr_enqueued] += 1; emit MKEV_INSTR_ENQUEUE with instr_id=id,
 *     tile=NO, buffer=NO, version=0, counter=NO, aux = the new instr_seq.
 *
 * ------------------------------------------------------------------------------
 * 5. ISSUE_LOADS (op=MKPS_OP_ISSUE_LOADS). limit=a0. limit==0 => no-op (no scan).
 * ------------------------------------------------------------------------------
 *   Scan order[] (used slots by instr_seq). Maintain window_seen=0, issued=0.
 *   "resident-current(tile)" := tile.resident_buffer != NO AND that buffer has
 *     state RESIDENT AND buffer.tile==tile AND buffer.version==tile.current_version.
 *   For each slot si in order:
 *     - if NOT nonterminal(si): skip (does not count toward window).
 *     - if window_seen >= issue_window: STOP scan.
 *     - window_seen += 1.
 *     - if issued >= limit: STOP scan.
 *     - if status != QUEUED: continue (counts toward window, not toward issue).
 *     Hazard tests vs EARLIER nonterminal instrs (those at order positions < oi),
 *     evaluated in the order WAW, then RAW, then WAR:
 *       WAW: some earlier nonterminal instr writes a tile that si also writes.
 *       RAW: for some read tile tl of si, tl.writer_instr != 0 AND that writer
 *            instr exists, is nonterminal, has instr_seq < si.instr_seq, AND
 *            tl is NOT resident-current. (If writer==0 or tl resident-current,
 *            that tile does not cause RAW.)
 *       WAR: for some write tile tl of si, some earlier nonterminal instr e with
 *            last_use_released==0 lists tl as one of its read tiles.
 *     If WAW or RAW or WAR: counts[MKC_load_hazard_stall] += 1; emit
 *       MKEV_LOAD_HAZARD_STALL (instr_id=si.id, others sentinel/0); STOP scan.
 *     Capacity: need = (#distinct read tiles of si that are NOT resident-current)
 *       + write_count + scratch_pages. avail = number of FREE buffers. If
 *       need > avail: counts[MKC_load_hazard_stall] += 1; emit
 *       MKEV_LOAD_HAZARD_STALL; CONTINUE scan (do NOT stop; does not consume a
 *       unit of `limit`, does NOT increment window_seen again).
 *     Otherwise ISSUE (allocate buffers; assigned_buffers built in this order):
 *       chosen[] starts empty; nbuf=0; any_load=false.
 *       (1) READS in listed order r=0..read_count-1:
 *           read_versions[r] = tiles[tl].current_version (recorded now).
 *           If resident-current(tl): REUSE resident buffer bid:
 *             buffer.pin_count += 1; buffer.last_touch_seq = touch_seq_next++;
 *             mark chosen[bid]; append {buf=bid, role=READ, reused=1}.
 *           Else allocate LOWEST-id FREE buffer not already chosen:
 *             state=LOADING, tile=tl, version=tiles[tl].current_version,
 *             owner_instr=si.id, pin_count += 1, last_touch_seq=touch_seq_next++;
 *             mark chosen; append {buf, role=READ, reused=0}; any_load=true.
 *       (2) WRITES in listed order w=0..write_count-1: allocate LOWEST-id FREE
 *           buffer not chosen: write_versions[w] = version_seq_next++ ;
 *           state=LOADING, tile=tl, version=write_versions[w], owner_instr=si.id,
 *           pin_count += 1, last_touch_seq=touch_seq_next++;
 *           tiles[tl].writer_instr = si.id; append {buf, role=WRITE, reused=0};
 *           any_load=true.
 *       (3) SCRATCH s=0..scratch_pages-1: allocate LOWEST-id FREE buffer not
 *           chosen: state=LOADING, tile=NO, version=0, owner_instr=si.id,
 *           pin_count += 1, last_touch_seq=touch_seq_next++;
 *           append {buf, role=SCRATCH, reused=0}; any_load=true.
 *       Set nbuf = total appended.
 *       If any_load:
 *         status = LOAD_ISSUED; counts[MKC_load_issue] += 1.
 *         For each assigned buffer k in order with reused==0: emit MKEV_LOAD_ISSUE
 *           with instr_id=si.id, tile = (role==SCRATCH ? NO : buffer.tile),
 *           buffer = buf_id[k], version = buffer.version, counter=NO, aux=0.
 *         THEN for each assigned buffer k in order with reused==0:
 *           push pending op {kind=PEND_LOAD_DONE, due=clock+load_latency,
 *           op_seq=op_seq_next++, instr_id=si.id, buffer=buf_id[k]}.
 *         (Both loops run in assigned_buffers order; all LOAD_ISSUE events are
 *          emitted before any LOAD_DONE op is pushed.)
 *       Else (any_load==false: all reads resident, no writes, no scratch):
 *         status = LOAD_READY; set every assigned (reused) buffer state to
 *         READY_READ; counts[MKC_load_ready_inline] += 1; emit
 *         MKEV_LOAD_READY_INLINE (instr_id=si.id, others sentinel/0).
 *       issued += 1.
 *   NOTE: pushing a pending op when the pending queue is already full (size ==
 *   max_pending_ops) silently drops the scheduling (no event); capacity is
 *   normally guarded so this does not occur.
 *
 * ------------------------------------------------------------------------------
 * 6. ISSUE_COMPUTE (op=MKPS_OP_ISSUE_COMPUTE). limit=a0. limit==0 => no-op.
 * ------------------------------------------------------------------------------
 *   Scan order[] (by instr_seq). issued=0.
 *   For each slot si: if issued >= limit STOP.
 *     - status==LOAD_ISSUED: if ANY assigned buffer is still LOADING ->
 *         counts[MKC_compute_load_wait] += 1; emit MKEV_COMPUTE_LOAD_WAIT
 *         (instr_id); STOP scan. Else continue (not consumed).
 *     - status==LOAD_READY: set ALL assigned buffers to COMPUTING; push pending
 *         {kind=PEND_COMPUTE_DONE, due=clock+compute_latency, op_seq=op_seq_next++,
 *         instr_id=si.id, buffer=NO}; status=COMPUTE_ISSUED;
 *         counts[MKC_compute_issue] += 1; emit MKEV_COMPUTE_ISSUE (instr_id);
 *         issued += 1.
 *     - any other status: continue (no event, not consumed).
 *
 * ------------------------------------------------------------------------------
 * 7. ISSUE_STORES (op=MKPS_OP_ISSUE_STORES). limit=a0. limit==0 => no-op.
 * ------------------------------------------------------------------------------
 *   Scan order[] (by instr_seq). issued=0. For each si: if issued>=limit STOP.
 *     - if status != COMPUTE_ISSUED: continue.
 *     - if compute_done_flag == 0: continue (compute not finished yet).
 *     - else: set assigned buffers with role==WRITE to STORING; push pending
 *         {kind=PEND_STORE_DONE, due=clock+store_latency, op_seq=op_seq_next++,
 *         instr_id=si.id, buffer=NO}; status=STORE_ISSUED;
 *         counts[MKC_store_issue] += 1; emit MKEV_STORE_ISSUE (instr_id);
 *         issued += 1.
 *
 * ------------------------------------------------------------------------------
 * 8. ADVANCE (op=MKPS_OP_ADVANCE). delta=a0, max_ops=a1. delta==0 valid.
 * ------------------------------------------------------------------------------
 *   clock += delta. Then repeat up to max_ops times:
 *     Among ACTIVE pending ops with due_clock <= clock, pick the one with the
 *     smallest (due_clock, op_seq) lexicographically (due_clock primary asc,
 *     op_seq tie-break asc). If none, STOP. Mark it inactive (removed), then
 *     dispatch by kind (see 8a/8b/8c). processed += 1.
 *   After the loop, COMPACT the pending queue: keep active ops in their existing
 *   relative order, zero the freed tail slots, and set pend_n to the kept count.
 *   (Compaction preserves storage order, which is the order pending_hash uses.)
 *
 *   8a. PEND_LOAD_DONE (instr_id, buffer):
 *     STALE if: instr not found, OR instr terminal, OR buffer.owner_instr !=
 *       instr_id, OR buffer.state != LOADING. On STALE:
 *       counts[MKC_op_stale_drop] += 1; emit MKEV_OP_STALE_DROP with
 *       instr_id, tile=NO, buffer=buffer, version=0, counter=NO,
 *       aux = (u64)PEND_LOAD_DONE (=0). Return.
 *     Else: role = the assigned role recorded for this (instr,buffer).
 *       buffer.state = (role==WRITE ? READY_WRITE : READY_READ)  [SCRATCH ->
 *       READY_READ]; buffer.last_touch_seq = touch_seq_next++;
 *       counts[MKC_load_done] += 1; emit MKEV_LOAD_DONE with instr_id,
 *       tile=buffer.tile, buffer=buffer, version=buffer.version, counter=NO, aux=0.
 *       Then if status==LOAD_ISSUED AND no assigned buffer is LOADING anymore:
 *         status=LOAD_READY; counts[MKC_instr_load_ready] += 1; emit
 *         MKEV_INSTR_LOAD_READY (instr_id).
 *
 *   8b. PEND_COMPUTE_DONE (instr_id):
 *     STALE if instr not found OR status != COMPUTE_ISSUED. On STALE:
 *       counts[MKC_op_stale_drop] += 1; emit MKEV_OP_STALE_DROP with instr_id,
 *       tile=NO, buffer=NO, version=0, counter=NO, aux=(u64)PEND_COMPUTE_DONE(=1).
 *     Else: result = COMPUTE_RESULT(si) (see 10); compute_done_flag = 1 (status
 *       stays COMPUTE_ISSUED); counts[MKC_compute_done] += 1; emit
 *       MKEV_COMPUTE_DONE with instr_id, sentinels, aux = result.
 *
 *   8c. PEND_STORE_DONE (instr_id):
 *     STALE if instr not found OR status != STORE_ISSUED. On STALE:
 *       counts[MKC_op_stale_drop] += 1; emit MKEV_OP_STALE_DROP with instr_id,
 *       tile=NO, buffer=NO, version=0, counter=NO, aux=(u64)PEND_STORE_DONE(=2).
 *     Else, in this exact order:
 *       (i) For each WRITE tile w in listed order: wv=write_versions[w];
 *           find the assigned WRITE buffer whose buffer.tile == tile (first match
 *           in assigned order; else NO). Set tiles[tl].current_version = wv,
 *           writer_instr = 0, last_store_seq = CURRENT event_seq (the seq about to
 *           be stamped on this TILE_STORE), dirty = 0. If a write buffer found:
 *           buffer.state = RESIDENT, buffer.tile = tl, buffer.version = wv,
 *           buffer.last_touch_seq = touch_seq_next++,
 *           tiles[tl].resident_buffer = that buffer. counts[MKC_tile_store] += 1;
 *           emit MKEV_TILE_STORE with instr_id, tile=tl, buffer=wbuf, version=wv,
 *           counter=NO, aux=0.
 *       (ii) For each READ tile r in listed order: if tiles[read].
 *            pending_reader_count > 0, decrement it by 1.
 *       (iii) last_use_released = 1.
 *       (iv) Release buffers: for each assigned buffer k in order, SKIP role
 *            WRITE. If role READ: let tl=buffer.tile;
 *              zero_readers = (tl<tile_count ? tiles[tl].pending_reader_count==0
 *                              : true);
 *              is_resident_cur = (tl<tile_count) AND tiles[tl].resident_buffer==
 *                                bid AND buffer.version==tiles[tl].current_version
 *                                AND buffer.state==RESIDENT.
 *              If zero_readers AND NOT is_resident_cur: if tl<tile_count and
 *              tiles[tl].resident_buffer==bid set it to NO; clear buffer to FREE
 *              defaults; counts[MKC_read_release] += 1; emit MKEV_READ_RELEASE
 *              with instr_id, tile=tl, buffer=bid, version=0, counter=NO, aux=0.
 *            If role SCRATCH: always clear buffer to FREE; counts[MKC_read_release]
 *              += 1; emit MKEV_READ_RELEASE with instr_id, tile=NO, buffer=bid,
 *              version=0, counter=NO, aux=0.
 *       (v) If out_counter != NO AND out_counter < counter_count:
 *           counter[out_counter] += 1; counts[MKC_counter_inc] += 1; emit
 *           MKEV_COUNTER_INC with instr_id, tile=NO, buffer=NO, version=0,
 *           counter_id=out_counter, aux = counter[out_counter] (new value).
 *       (vi) status = DONE; counts[MKC_instr_done] += 1; emit MKEV_INSTR_DONE
 *           (instr_id, sentinels, aux=0).
 *     "Clear buffer to FREE defaults" = state FREE, tile NO, version 0,
 *     owner_instr 0, pin_count 0, last_touch_seq 0.
 *
 * ------------------------------------------------------------------------------
 * 9. CANCEL (op=MKPS_OP_CANCEL). id=a0.
 * ------------------------------------------------------------------------------
 *   Invalid if id absent OR instr terminal (DONE/CANCELLED):
 *     counts[MKC_invalid_count] += 1; emit MKEV_INVALID (instr_id=id, sentinels).
 *   Else: status = CANCELLED. For every ACTIVE pending op with instr_id==id, set
 *   inactive (removed silently, NO event). Then for each buffer b in BUFFER-ID
 *   order: if owner_instr==id AND state != FREE AND state != RESIDENT: let
 *   tl=buffer.tile; if tl<tile_count and tiles[tl].resident_buffer==b set to NO;
 *   clear buffer to FREE defaults; counts[MKC_buffer_cancel_release] += 1; emit
 *   MKEV_BUFFER_CANCEL_RELEASE with instr_id=id, tile=tl, buffer=b, version=0,
 *   counter=NO, aux=0. Then if last_use_released==0: for each read tile (per
 *   occurrence) decrement its pending_reader_count if > 0. Then for each write
 *   tile tl with tiles[tl].writer_instr==id, set writer_instr=0.
 *   counts[MKC_instr_cancel] += 1; emit MKEV_INSTR_CANCEL (instr_id=id,
 *   sentinels). Then COMPACT the pending queue (as in ADVANCE).
 *
 * ------------------------------------------------------------------------------
 * 10. COMPUTE_RESULT(si)  (aux of MKEV_COMPUTE_DONE)
 * ------------------------------------------------------------------------------
 *   h = offset basis; absorb in order:
 *     instr_id        u32
 *     instr_seq       u64
 *     payload_seed    u64
 *     for r in 0..read_count-1:  read_versions[r]   u64
 *     for w in 0..write_count-1: write_versions[w]  u64
 *     for k in 0..nbuf-1:        assigned buf_id[k] u32
 *   result = h.
 *
 * ------------------------------------------------------------------------------
 * 11. HOST_COUNTER (op=MKPS_OP_HOST_COUNTER). counter_id=a0, amount=a1.
 * ------------------------------------------------------------------------------
 *   If counter_id >= counter_count: counts[MKC_invalid_count] += 1; emit
 *     MKEV_INVALID with instr_id=0, tile=NO, buffer=NO, version=0, counter=NO,
 *     aux=0. Else: counter[counter_id] += amount; counts[MKC_host_counter_inc]
 *     += 1; emit MKEV_HOST_COUNTER_INC with instr_id=0, tile=NO, buffer=NO,
 *     version=0, counter_id=counter_id, aux = counter[counter_id] (new value).
 *
 *   UNKNOWN op kind (not one of MKPS_OP_*): counts[MKC_invalid_count] += 1;
 *     emit MKEV_INVALID with instr_id=0 and all sentinels/0.
 *
 * ------------------------------------------------------------------------------
 * 12. SNAPSHOT HASHES (recomputed from scratch after each step's batch; each
 *     seeded with offset basis). Field order per element is EXACT.
 * ------------------------------------------------------------------------------
 *   instr_hash: iterate USED nonterminal instrs (status not DONE/CANCELLED) in
 *     instr_seq ascending order. Per instr absorb:
 *       instr_id u32; instr_seq u64; status u8; read_count u32; write_count u32;
 *       scratch_pages u32; compute_done_flag u8; last_use_released u8;
 *       then for r in 0..read_count-1: reads[r] u32, read_versions[r] u64;
 *       then for w in 0..write_count-1: writes[w] u32, write_versions[w] u64;
 *       then nbuf u32;
 *       then for k in 0..nbuf-1: buf_id[k] u32, buf_role[k] u8.
 *
 *   tile_hash: iterate tiles l=0..tile_count-1. Per tile absorb:
 *       l u32; current_version u64; (u64)writer_instr u64;
 *       pending_reader_count u64; last_store_seq u64; resident_buffer u32;
 *       dirty u8.
 *
 *   buffer_hash: iterate buffers b=0..buffer_count-1. Per buffer absorb:
 *       b u32; state u8; tile u32; version u64; (u64)owner_instr u64;
 *       pin_count u64; last_touch_seq u64.
 *
 *   pending_hash: iterate pending slots i=0..pend_n-1 in STORAGE order; skip
 *     inactive. Per active op absorb:
 *       kind u8; due_clock u64; op_seq u64; (u64)instr_id u64; buffer_id u32.
 *
 *   counter_hash: iterate counters c=0..counter_count-1. Per counter absorb:
 *       c u32; counter[c] u64.
 *
 * ------------------------------------------------------------------------------
 * 13. STATE CHECKSUM (state_checksum[0]; seeded with offset basis). Absorb in
 *     this exact order:
 * ------------------------------------------------------------------------------
 *     buffer_count i32; tile_count i32; max_instrs i32; issue_window i32;
 *     max_pending_ops i32; counter_count i32;
 *     clock u64; event_seq u64; instr_seq_next u64; op_seq_next u64;
 *     version_seq_next u64;       (touch_seq_next is NOT included)
 *     instr_hash u64; tile_hash u64; buffer_hash u64; pending_hash u64;
 *     counter_hash u64;
 *     then for i in 0..MKPS_COUNT_FIELDS-1: counts[i] i64.
 *
 * ------------------------------------------------------------------------------
 * 14. EVENT-KIND aux / SENTINEL SUMMARY
 * ------------------------------------------------------------------------------
 *   Sentinels for "not applicable": tile/buffer/counter = MKPS_NO_U32,
 *   version = 0, instr_id = 0, aux = 0.
 *   aux carries a meaningful value ONLY for:
 *     MKEV_INSTR_ENQUEUE   -> new instr_seq
 *     MKEV_COMPUTE_DONE    -> COMPUTE_RESULT (section 10)
 *     MKEV_COUNTER_INC     -> counter[out_counter] AFTER increment
 *     MKEV_HOST_COUNTER_INC-> counter[counter_id]  AFTER increment
 *     MKEV_OP_STALE_DROP   -> the pending kind (0=LOAD,1=COMPUTE,2=STORE)
 *   counter_id is set (non-NO) only for MKEV_COUNTER_INC and
 *   MKEV_HOST_COUNTER_INC. version is set (non-0) for MKEV_LOAD_ISSUE,
 *   MKEV_LOAD_DONE, MKEV_TILE_STORE. buffer is set for LOAD_ISSUE, LOAD_DONE,
 *   TILE_STORE, READ_RELEASE, BUFFER_CANCEL_RELEASE, and OP_STALE_DROP of a
 *   LOAD_DONE op. tile is set for LOAD_ISSUE (non-scratch), LOAD_DONE, TILE_STORE,
 *   READ_RELEASE (read role), BUFFER_CANCEL_RELEASE.
 */

/* Output count field indices (one per event class). */
enum {
    MKC_instr_enqueued = 0,
    MKC_load_issue,
    MKC_load_ready_inline,
    MKC_load_hazard_stall,
    MKC_load_done,
    MKC_instr_load_ready,
    MKC_compute_issue,
    MKC_compute_load_wait,
    MKC_compute_done,
    MKC_store_issue,
    MKC_tile_store,
    MKC_read_release,
    MKC_counter_inc,
    MKC_instr_done,
    MKC_op_stale_drop,
    MKC_buffer_cancel_release,
    MKC_instr_cancel,
    MKC_host_counter_inc,
    MKC_invalid_count,
    MKPS_COUNT_FIELDS
};

struct alignas(8) MkpsProblemSpec {
    int32_t abi_version;
    int32_t buffer_count;
    int32_t tile_count;
    int32_t max_instrs;
    int32_t max_reads_per_instr;
    int32_t max_writes_per_instr;
    int32_t issue_window;
    int32_t max_pending_ops;
    int32_t counter_count;
    int32_t max_batch;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[8];
};

struct alignas(8) MkpsRunSpec {
    int32_t abi_version;
    int32_t batch_size;
    int32_t step_id;
    int32_t reserved[13];
};

/* Per-op input row layout. Unused fields take 0.
 *   ENQUEUE:        instr_id=a0; read_count=a1; reads in tiles[i*16 .. +read_count];
 *                   write_count=a2; writes in tiles[i*16+8 .. +write_count];
 *                   scratch_pages=a3; load_latency=a4; compute_latency=a5;
 *                   store_latency=a6; out_counter=a7 (or NO_U32); payload_seed=a8.
 *   ISSUE_LOADS:    limit=a0.
 *   ISSUE_COMPUTE:  limit=a0.
 *   ISSUE_STORES:   limit=a0.
 *   ADVANCE:        delta=a0; max_ops=a1.
 *   CANCEL:         instr_id=a0.
 *   HOST_COUNTER:   counter_id=a0; amount=a1.
 */
struct alignas(8) MkpsInputs {
    const int32_t* op;          // MKPS_OP_*
    const uint32_t* a0;
    const uint32_t* a1;
    const uint32_t* a2;
    const uint32_t* a3;
    const uint32_t* a4;
    const uint32_t* a5;
    const uint32_t* a6;
    const uint32_t* a7;
    const uint64_t* a8;
    const uint32_t* tiles;      // [batch*16]: [0..7]=read tiles, [8..15]=write tiles
};

struct alignas(8) MkpsOutputs {
    int64_t* counts;              // MKPS_COUNT_FIELDS entries
    uint64_t* pipeline_event_hash;// 1
    uint64_t* instr_hash;         // 1
    uint64_t* tile_hash;          // 1
    uint64_t* buffer_hash;        // 1
    uint64_t* pending_hash;       // 1
    uint64_t* counter_hash;       // 1
    uint64_t* event_seq_out;      // 1
    uint64_t* state_checksum;     // 1
};

static_assert(sizeof(MkpsProblemSpec) == 80, "MkpsProblemSpec layout drift");
static_assert(sizeof(MkpsRunSpec) == 64, "MkpsRunSpec layout drift");
static_assert(sizeof(MkpsInputs) == 88, "MkpsInputs layout drift");
static_assert(sizeof(MkpsOutputs) == 72, "MkpsOutputs layout drift");

static inline int mkps_validate_problem_spec(const MkpsProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MKPS_ABI_VERSION) return 0;
    if (spec->buffer_count < MKPS_MIN_BUFFERS || spec->buffer_count > MKPS_MAX_BUFFERS) return 0;
    if (spec->tile_count < MKPS_MIN_TILES || spec->tile_count > MKPS_MAX_TILES) return 0;
    if (spec->max_instrs < MKPS_MIN_INSTRS || spec->max_instrs > MKPS_MAX_INSTRS) return 0;
    if (spec->max_reads_per_instr < 0 || spec->max_reads_per_instr > MKPS_MAX_READS) return 0;
    if (spec->max_writes_per_instr < 0 || spec->max_writes_per_instr > MKPS_MAX_WRITES) return 0;
    if (spec->issue_window < MKPS_MIN_WINDOW || spec->issue_window > MKPS_MAX_WINDOW) return 0;
    if (spec->max_pending_ops < MKPS_MIN_PENDING || spec->max_pending_ops > MKPS_MAX_PENDING) return 0;
    if (spec->counter_count < MKPS_MIN_COUNTERS || spec->counter_count > MKPS_MAX_COUNTERS) return 0;
    if (spec->max_batch < 0 || spec->max_batch > MKPS_MAX_BATCH) return 0;
    if (spec->max_steps < 1 || spec->max_steps > MKPS_MAX_STEPS) return 0;
    return 1;
}

static inline int mkps_validate_run_spec(const MkpsRunSpec* run, const MkpsProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != MKPS_ABI_VERSION) return 0;
    if (run->batch_size < 0 || run->batch_size > spec->max_batch) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const MkpsProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const MkpsProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const MkpsRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // MK_PIPELINE_SCOREBOARD_COMMON_H_
