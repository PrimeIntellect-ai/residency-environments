// file: mk_instr_decoder_hazard_common.h
//
// MK10 — Megakernel Instruction Decoder with Tile Hazard Scoreboarding.
//
// A persistent bytecode-like instruction-stream decoder that issues coarse
// load/compute/store/event instructions across SMs while enforcing
// RAW/WAR/WAW tile hazards, page ownership, dependency-counter waits, branch
// epochs, replay queues, and ordered retirement.  Every solution_run applies
// exactly ONE coarse op (encoded in MkRunSpec) to persistent device state and
// emits a fully-determined event/scoreboard timeline graded via FNV-1a-64
// checksums.  All graded outputs are EXACT integers (no float, no tolerance).

#ifndef MK_INSTR_DECODER_HAZARD_COMMON_H_
#define MK_INSTR_DECODER_HAZARD_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MK_ABI_VERSION 1

// ----- structural limits -----
#define MK_MIN_SM 1
#define MK_MAX_SM 16
#define MK_MIN_STREAM 1
#define MK_MAX_STREAM 256
#define MK_MIN_WIDTH 1
#define MK_MAX_WIDTH 64
#define MK_MIN_TILES 1
#define MK_MAX_TILES 256
#define MK_MIN_PAGES 1
#define MK_MAX_PAGES 64
#define MK_MIN_COUNTERS 1
#define MK_MAX_COUNTERS 64
#define MK_MAX_QUEUE 256
#define MK_MAX_PENDING 4096
#define MK_MAX_REPLAY 256
#define MK_MAX_RECORDS 16384

// Max read/write tiles addressed by a single ALU instruction.
#define MK_MAX_RW 2

// Opcodes.
enum MkOpcode {
    MK_LD = 0,
    MK_ALU = 1,
    MK_ST = 2,
    MK_WAIT = 3,
    MK_INC = 4,
    MK_FREE = 5,
    MK_BRANCH = 6,
    MK_NOP = 7,
    MK_OPCODE_COUNT = 8
};

// Coarse host ops (one per solution_run).
enum MkOpKind {
    MK_OP_BEGIN_KERNEL = 0,
    MK_OP_DECODE_SM = 1,
    MK_OP_ISSUE_SM = 2,
    MK_OP_ADVANCE = 3,
    MK_OP_RETIRE_SM = 4,
    MK_OP_REPLAY_SM = 5,
    MK_OP_HOST_INC_COUNTER = 6,
    MK_OP_FLUSH_SM = 7,
    MK_OP_KIND_COUNT = 8
};

// Decoded-record status.
enum MkStatus {
    MK_ST_DECODED = 0,
    MK_ST_ISSUE_READY = 1,
    MK_ST_ISSUED = 2,
    MK_ST_COMPLETED = 3,
    MK_ST_RETIRED = 4,
    MK_ST_REPLAY = 5,
    MK_ST_CANCELLED = 6
};

// Page states.
enum MkPageState {
    MK_PG_FREE = 0,
    MK_PG_LOADING = 1,
    MK_PG_RESIDENT = 2,
    MK_PG_COMPUTING = 3,
    MK_PG_STORING = 4,
    MK_PG_PINNED = 5
};

// Pending-unit kinds.
enum MkUnitKind {
    MK_U_LD_DONE = 0,
    MK_U_ALU_DONE = 1,
    MK_U_ST_DONE = 2,
    MK_U_WAIT_DONE = 3,
    MK_U_INC_DONE = 4,
    MK_U_FREE_DONE = 5,
    MK_U_BRANCH_DONE = 6
};

// Event kinds (the uint8 `kind` folded into decoder_event_hash; see CONTRACT
// section below for emit serialization).
enum MkEventKind {
    MK_EV_KERNEL_BEGIN = 0,
    MK_EV_INSTR_DECODE = 1,
    MK_EV_DECODE_PRED_SKIP = 2,
    MK_EV_DECODE_HALTED = 3,
    MK_EV_DECODE_QUEUE_FULL = 4,
    MK_EV_INSTR_ISSUE = 5,
    MK_EV_ISSUE_HAZARD = 6,
    MK_EV_ISSUE_PAGE_STALL = 7,
    MK_EV_LD_DONE = 8,
    MK_EV_ALU_DONE = 9,
    MK_EV_ST_DONE = 10,
    MK_EV_WAIT_DONE = 11,
    MK_EV_WAIT_REARM = 12,
    MK_EV_COUNTER_INC = 13,
    MK_EV_TILE_FREE = 14,
    MK_EV_FREE_REPLAY = 15,
    MK_EV_BRANCH_REDIRECT = 16,
    MK_EV_BRANCH_NOOP = 17,
    MK_EV_EPOCH_CANCEL = 18,
    MK_EV_UNIT_STALE_DROP = 19,
    MK_EV_INSTR_COMPLETE = 20,
    MK_EV_RETIRE_BLOCKED = 21,
    MK_EV_READ_RETIRE = 22,
    MK_EV_WRITE_RETIRE = 23,
    MK_EV_INSTR_RETIRE = 24,
    MK_EV_REPLAY_ENQUEUE = 25,
    MK_EV_HOST_COUNTER_INC = 26,
    MK_EV_SM_FLUSH = 27,
    MK_EV_INVALID = 28,
    MK_EV_KIND_COUNT = 29
};

// Output counter indices into MkOutputs::counts[MK_COUNT_N]; this index order
// is also the fold order of counts[] into state_checksum (see CONTRACT below).
enum MkCountIdx {
    MK_C_KERNEL_BEGIN = 0,
    MK_C_INSTR_DECODED = 1,
    MK_C_DECODE_PRED_SKIP = 2,
    MK_C_DECODE_HALTED = 3,
    MK_C_DECODE_QUEUE_FULL = 4,
    MK_C_INSTR_ISSUE = 5,
    MK_C_ISSUE_HAZARD = 6,
    MK_C_ISSUE_PAGE_STALL = 7,
    MK_C_LD_DONE = 8,
    MK_C_ALU_DONE = 9,
    MK_C_ST_DONE = 10,
    MK_C_WAIT_DONE = 11,
    MK_C_WAIT_REARM = 12,
    MK_C_COUNTER_INC = 13,
    MK_C_TILE_FREE = 14,
    MK_C_FREE_REPLAY = 15,
    MK_C_BRANCH_REDIRECT = 16,
    MK_C_BRANCH_NOOP = 17,
    MK_C_EPOCH_CANCEL = 18,
    MK_C_UNIT_STALE_DROP = 19,
    MK_C_INSTR_COMPLETE = 20,
    MK_C_RETIRE_BLOCKED = 21,
    MK_C_READ_RETIRE = 22,
    MK_C_WRITE_RETIRE = 23,
    MK_C_INSTR_RETIRE = 24,
    MK_C_REPLAY_ENQUEUE = 25,
    MK_C_HOST_COUNTER_INC = 26,
    MK_C_SM_FLUSH = 27,
    MK_C_INVALID = 28,
    MK_COUNT_N = 29
};

// Hazard codes (single dominant code; priority RAW>WAR>WAW).
enum MkHazardCode {
    MK_HZ_NONE = 0,
    MK_HZ_RAW = 1,
    MK_HZ_WAR = 2,
    MK_HZ_WAW = 3
};

// ----- one stream instruction (decode-time, immutable) -----
struct alignas(8) MkInstr {
    uint64_t instr_uid;
    uint64_t latency;
    uint64_t predicate_mask;
    uint64_t target;          // WAIT target / BRANCH counter target
    uint64_t amount;          // INC amount
    uint64_t compute_seed;    // ALU
    uint64_t store_seed;      // ST
    uint64_t page_key;        // LD page key
    int32_t  opcode;          // MkOpcode
    int32_t  dst_tile;        // LD destination tile
    int32_t  n_read;          // ALU read-tile count
    int32_t  n_write;         // ALU write-tile count
    int32_t  read_tiles[MK_MAX_RW];
    int32_t  write_tiles[MK_MAX_RW];
    int32_t  st_read_tile;    // ST read tile
    int32_t  free_tile;       // FREE tile
    int32_t  counter_id;      // WAIT/INC/BRANCH counter
    int32_t  taken_pc;        // BRANCH
    int32_t  fallthrough_pc;  // BRANCH
    int32_t  pad0;
};

struct alignas(8) MkProblemSpec {
    int32_t abi_version;
    int32_t sm_count;
    int32_t stream_len_per_sm;
    int32_t decode_width;
    int32_t issue_width;
    int32_t retire_width;
    int32_t tile_count;
    int32_t page_count_per_sm;
    int32_t counter_count;
    int32_t max_decode_queue;
    int32_t max_issue_queue;
    int32_t max_pending_units;
    int32_t max_replay;
    int32_t max_decode_records;
    int32_t flags;
    int32_t reserved[1];
};

struct alignas(8) MkRunSpec {
    int32_t abi_version;
    int32_t op_kind;          // MkOpKind
    int32_t step_id;
    int32_t a_sm;             // DECODE/ISSUE/RETIRE/REPLAY/FLUSH target sm
    int32_t a_limit;          // DECODE/ISSUE/RETIRE/REPLAY limit
    int32_t a_max_units;      // ADVANCE max_units
    int32_t a_counter_id;     // HOST_INC_COUNTER
    int32_t reserved0;
    uint64_t a_epoch_seed;    // BEGIN_KERNEL
    uint64_t a_delta;         // ADVANCE delta
    uint64_t a_amount;        // HOST_INC_COUNTER amount
    uint64_t reserved1;
};

struct alignas(8) MkInputs {
    const MkInstr* stream;    // [sm_count * stream_len_per_sm]
    int32_t stream_count;
    int32_t pad;
};

struct alignas(8) MkOutputs {
    int64_t* counts;                  // [MK_COUNT_N]
    int32_t* op_index_out;            // [1]
    uint64_t* clock_out;              // [1]
    uint64_t* event_seq_out;          // [1]
    uint64_t* decoder_event_hash;     // [1]
    uint64_t* sm_stream_hash;         // [1]
    uint64_t* decoded_hash;           // [1]
    uint64_t* tile_scoreboard_hash;   // [1]
    uint64_t* page_hash;              // [1]
    uint64_t* pending_unit_hash;      // [1]
    uint64_t* counter_hash;           // [1]
    uint64_t* state_checksum;         // [1]
};

static_assert(sizeof(MkProblemSpec) == 64, "MkProblemSpec layout drift");
static_assert(sizeof(MkRunSpec) == 64, "MkRunSpec layout drift");
static_assert(sizeof(MkOutputs) == 96, "MkOutputs layout drift");

static inline int mk_validate_problem_spec(const MkProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MK_ABI_VERSION) return 0;
    if (spec->sm_count < MK_MIN_SM || spec->sm_count > MK_MAX_SM) return 0;
    if (spec->stream_len_per_sm < MK_MIN_STREAM || spec->stream_len_per_sm > MK_MAX_STREAM) return 0;
    if (spec->decode_width < MK_MIN_WIDTH || spec->decode_width > MK_MAX_WIDTH) return 0;
    if (spec->issue_width < MK_MIN_WIDTH || spec->issue_width > MK_MAX_WIDTH) return 0;
    if (spec->retire_width < MK_MIN_WIDTH || spec->retire_width > MK_MAX_WIDTH) return 0;
    if (spec->tile_count < MK_MIN_TILES || spec->tile_count > MK_MAX_TILES) return 0;
    if (spec->page_count_per_sm < MK_MIN_PAGES || spec->page_count_per_sm > MK_MAX_PAGES) return 0;
    if (spec->counter_count < MK_MIN_COUNTERS || spec->counter_count > MK_MAX_COUNTERS) return 0;
    if (spec->max_decode_queue < 1 || spec->max_decode_queue > MK_MAX_QUEUE) return 0;
    if (spec->max_issue_queue < 1 || spec->max_issue_queue > MK_MAX_QUEUE) return 0;
    if (spec->max_pending_units < 1 || spec->max_pending_units > MK_MAX_PENDING) return 0;
    if (spec->max_replay < 1 || spec->max_replay > MK_MAX_REPLAY) return 0;
    if (spec->max_decode_records < 1 || spec->max_decode_records > MK_MAX_RECORDS) return 0;
    return 1;
}

static inline int mk_validate_run_spec(const MkRunSpec* run, const MkProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != MK_ABI_VERSION) return 0;
    if (run->op_kind < 0 || run->op_kind >= MK_OP_KIND_COUNT) return 0;
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

// ============================================================================
// === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
// ============================================================================
//
// This section is the COMPLETE and AUTHORITATIVE specification of what every
// coarse op (MkRunSpec.op_kind) computes and how every graded output is
// serialized. All graded values are EXACT integers (no float, no tolerance).
// A conforming solution must reproduce every count, scalar, and FNV-1a-64
// checksum below bit-for-bit. Nothing here is deferred to any external file.
//
// ---------------------------------------------------------------------------
// 0. FNV-1a-64 primitive (used for EVERY hash below)
// ---------------------------------------------------------------------------
//   offset basis = 1469598103934665603 (0x14650FB0739D0383)
//   prime        = 1099511628211       (0x00000100000001B3)
//   fold u8 v :  h ^= (uint64_t)v;  h *= prime;
//   fold u32 v:  fold the 4 bytes of v in LITTLE-ENDIAN (native x86/CUDA)
//                order, i.e. p=(const uint8_t*)&v; for i in 0..3: fold u8 p[i].
//   fold u64 v:  fold the 8 bytes of v in LITTLE-ENDIAN order, i in 0..7.
//   A fresh hash is INITIALIZED to the offset basis, EXCEPT the running event
//   hash (see 2), which is SEEDED from its own prior value before each event.
//   "fold X as u32/u64" below means apply the corresponding primitive. Signed
//   ints (int32_t) are folded by reinterpreting as uint32_t (two's complement).
//
// ---------------------------------------------------------------------------
// 1. Persistent scalar state (all start as in solution_reset)
// ---------------------------------------------------------------------------
//   clock=0, event_seq=0, op_index=0, event_hash=offset_basis,
//   decode_seq_next=1, issue_seq_next=1, unit_seq_next=1, retire_seq_next=1,
//   epoch_seq_next=1. Each decoded record lives in a flat pool at slot
//   (decode_seq-1); decode_seq is a 1-based global allocation counter.
//
// ---------------------------------------------------------------------------
// 2. decoder_event_hash — running event log hash (output: decoder_event_hash)
// ---------------------------------------------------------------------------
//   Every emitted event updates the running event_hash. For one event the
//   hash is RE-SEEDED to the current event_hash value, then these fields are
//   folded IN THIS EXACT ORDER:
//       u8  kind        (MkEventKind ordinal)
//       u64 event_seq   (value BEFORE this event; the pre-increment seq)
//       u32 op_index    (current op_index, i.e. value BEFORE its post-op bump)
//       u64 clock
//       u32 sm          (U32MAX = 0xFFFFFFFF when the event has no SM)
//       u32 pc          (U32MAX when no pc)
//       u64 decode_seq  (0 when none)
//       u64 instr_uid   (0 when none)
//       u32 tile        (U32MAX when none)
//       u64 aux         (per-event payload, see 8)
//   After folding, event_hash = result, then event_seq += 1. The final
//   decoder_event_hash output equals event_hash after the op completes.
//
// ---------------------------------------------------------------------------
// 3. Coarse-op SEMANTICS (one MkRunSpec per solution_run; dispatch on op_kind)
// ---------------------------------------------------------------------------
//   Any structurally-invalid op (bad sm index, limit==0, bad counter id, or
//   BEGIN_KERNEL while work is still outstanding) increments counts[MK_C_INVALID]
//   and emits MK_EV_INVALID(sm=U32MAX,pc=U32MAX,ds=0,uid=0,tile=U32MAX,aux=0),
//   and performs no other state change.
//
//   MK_OP_BEGIN_KERNEL(a_epoch_seed): valid only if NOTHING is outstanding
//     (no record with status != RETIRED/CANCELLED and no pending units), else
//     INVALID. Resets every SM: pc=0, halted=0, all queues empty, and
//     epoch = fnv_epoch(seed, sm, epoch_seq_next) then epoch_seq_next += 1,
//     where fnv_epoch(seed,sm,ctr) = fresh FNV folding u64 seed, u32 sm,
//     u64 ctr. Zeros all counters; resets every tile (version=0, writer=0,
//     oldest_reader=0, reader_count=0, resident_page=U32MAX, dirty=0, freed=0);
//     frees every page (state=FREE,key=0,tile=U32MAX,version=0,owner=0);
//     marks every existing record slot unused. counts[MK_C_KERNEL_BEGIN]+=1;
//     emit MK_EV_KERNEL_BEGIN(U32MAX,U32MAX,0,0,U32MAX, aux=seed).
//
//   MK_OP_DECODE_SM(a_sm,a_limit): decode budget = min(a_limit, decode_width).
//     Loop up to budget times; STOP (break) on: sm halted (MK_C_DECODE_HALTED +
//     EV_DECODE_HALTED); decode queue full, dq_n>=max_decode_queue
//     (MK_C_DECODE_QUEUE_FULL + EV_DECODE_QUEUE_FULL); pc>=stream_len_per_sm
//     (set halted=1, MK_C_DECODE_HALTED + EV_DECODE_HALTED); record pool full,
//     (decode_seq_next-1)>=max_decode_records (MK_C_DECODE_QUEUE_FULL +
//     EV_DECODE_QUEUE_FULL). Predicate skip (CONTINUE, advance pc): if
//     (instr.predicate_mask & sm.epoch)==0 -> MK_C_DECODE_PRED_SKIP +
//     EV_DECODE_PRED_SKIP(sm,pc,0,uid,U32MAX,0), pc+=1. Otherwise allocate a
//     record: decode_seq=decode_seq_next; decode_seq_next+=1; slot=decode_seq-1;
//     fill record (sm,pc,epoch,uid,opcode,status=DECODED,hazard=NONE,
//     result_hash=0) and per-opcode fields (see 4); push decode_seq onto BOTH
//     the SM decode queue (back) and the SM issue queue (back);
//     MK_C_INSTR_DECODED + EV_INSTR_DECODE(sm,pc,ds,uid,U32MAX,0). Next pc:
//     for MK_BRANCH, pc = instr.fallthrough_pc; otherwise pc = pc+1.
//
//   MK_OP_ISSUE_SM(a_sm,a_limit): issue budget = min(a_limit, issue_width).
//     Repeatedly walk the SM issue queue in ASCENDING decode_seq order (the
//     queue is a multiset of decode_seqs; pick the smallest unvisited each
//     pass). For each entry: if its slot is unused, remove from all queues and
//     keep scanning; if its status is not DECODED and not REPLAY, remove from
//     the issue queue and keep scanning. Else run hazard_check (see 5): if a
//     hazard is found, set record.hazard_code, MK_C_ISSUE_HAZARD +
//     EV_ISSUE_HAZARD(sm,pc,ds,uid,U32MAX, aux=hazard_code), and STOP the
//     entire issue op (head-of-line blocking). Else page_check (see 6): if it
//     fails, MK_C_ISSUE_PAGE_STALL + EV_ISSUE_PAGE_STALL(sm,pc,ds,uid,U32MAX,0)
//     and CONTINUE to the next entry (do NOT remove, do NOT count an issue).
//     Else reserve_and_issue (see 7), remove from issue queue, issued+=1. Stop
//     when issued>=budget or no progress can be made.
//
//   MK_OP_ADVANCE(a_delta,a_max_units): clock += a_delta. Then up to
//     a_max_units times, pick the READY pending unit (due_clock<=clock) with
//     the minimum (due_clock, then unit_seq); none ready -> stop. Remove it
//     from the pending list, then dispatch by unit kind (see 8). A unit whose
//     record slot is now unused, or whose record's SM epoch no longer matches
//     the unit's epoch, is a stale drop: MK_C_UNIT_STALE_DROP +
//     EV_UNIT_STALE_DROP and it does NOT complete its record. processed+=1 per
//     unit handled (including stale drops, WAIT re-arm, FREE replay).
//
//   MK_OP_RETIRE_SM(a_sm,a_limit): budget = min(a_limit, retire_width). Up to
//     budget times: head = smallest-decode_seq non-RETIRED/non-CANCELLED record
//     on this SM; none -> stop. If head's status != COMPLETED:
//     MK_C_RETIRE_BLOCKED + EV_RETIRE_BLOCKED and STOP (in-order retirement).
//     Else: for each read tile, reader_count-=1 (floored at 0); set status=
//     RETIRED; for each read tile, recompute its oldest_reader (see 5),
//     MK_C_READ_RETIRE + EV_READ_RETIRE(...,tile=t,0); for each write tile, if
//     writer==head clear writer to 0, MK_C_WRITE_RETIRE + EV_WRITE_RETIRE(
//     ...,tile=t,0); rseq=retire_seq_next; retire_seq_next+=1; MK_C_INSTR_RETIRE
//     + EV_INSTR_RETIRE(...,aux=rseq); remove head from all queues; retired+=1.
//
//   MK_OP_REPLAY_SM(a_sm,a_limit): move up to a_limit decode_seqs from the
//     FRONT of the SM replay queue (in replay-queue order) and PREPEND them, in
//     that same order, to the FRONT of the SM issue queue (so the first moved
//     entry ends up frontmost). For each moved entry: MK_C_REPLAY_ENQUEUE +
//     EV_REPLAY_ENQUEUE(sm, pc-or-U32MAX, ds, uid-or-0, U32MAX, 0).
//
//   MK_OP_HOST_INC_COUNTER(a_counter_id,a_amount): bad id -> INVALID. Else
//     counters[id]+=amount; MK_C_HOST_COUNTER_INC + EV_HOST_COUNTER_INC(
//     U32MAX,U32MAX,0,0,U32MAX, aux=counters[id]).
//
//   MK_OP_FLUSH_SM(a_sm): cancel EVERY non-RETIRED/non-CANCELLED record on the
//     SM (iterate slots HIGH->LOW): release its reserved pages (set FREE), and
//     if it was issued/completed/replay also unwind reservations (clear writer
//     it owns, decrement reader_count for its reads), set status=CANCELLED,
//     MK_C_EPOCH_CANCEL + EV_EPOCH_CANCEL(sm,pc,ds,uid,U32MAX,0). Then recompute
//     oldest_reader for the read tiles of those cancelled records. Empty all
//     three SM queues, bump sm.epoch = epoch_seq_next++ , MK_C_SM_FLUSH +
//     EV_SM_FLUSH(sm, sm.pc, 0,0,U32MAX,0).
//
// ---------------------------------------------------------------------------
// 4. Decode-time record field fill (per opcode), used by hash_decoded
// ---------------------------------------------------------------------------
//   Common: read_tiles=[], write_tiles=[], pages=[], free_tile=-1,
//   counter_id=-1, target=0, amount=0, compute_seed=0, page_key=0, dst_tile=-1,
//   taken_pc=0, fallthrough_pc=0, result_hash=0. Then by opcode:
//     MK_LD : write_tiles=[dst_tile], dst_tile=instr.dst_tile,
//             page_key=instr.page_key.
//     MK_ALU: read_tiles = instr.read_tiles[0..n_read), write_tiles =
//             instr.write_tiles[0..n_write), compute_seed=instr.compute_seed.
//     MK_ST : read_tiles=[instr.st_read_tile].
//     MK_WAIT  : counter_id=instr.counter_id, target=instr.target.
//     MK_INC   : counter_id=instr.counter_id, amount=instr.amount.
//     MK_FREE  : free_tile=instr.free_tile.
//     MK_BRANCH: counter_id=instr.counter_id, target=instr.target,
//                taken_pc=instr.taken_pc, fallthrough_pc=instr.fallthrough_pc.
//     MK_NOP   : no extra fields.
//
// ---------------------------------------------------------------------------
// 5. hazard_check & oldest_reader (decode_seq = slot+1 = "ds")
// ---------------------------------------------------------------------------
//   Priority RAW > WAR > WAW; return the FIRST (single dominant) code found:
//     RAW: any READ tile t with t.freed, OR t.writer!=0 && t.writer<ds.
//     WAR: any WRITE tile t with t.oldest_reader!=0 && t.oldest_reader<ds.
//     WAW: any WRITE tile t with t.writer!=0 && t.writer<ds.
//   oldest_reader(tile) recompute = the MINIMUM decode_seq among records on the
//   same SM that are ISSUED/COMPLETED/REPLAY (not RETIRED/CANCELLED) and read
//   that tile; 0 if none.
//
// ---------------------------------------------------------------------------
// 6. page_check (resource availability at issue)
// ---------------------------------------------------------------------------
//   MK_LD : at least 1 free page on the SM.
//   MK_ALU: every read tile must be resident (resident_page!=U32MAX) AND the
//           SM has >= n_write free pages.
//   MK_ST : if it reads a tile, that tile must be resident; else ok.
//   others: always pass.
//
// ---------------------------------------------------------------------------
// 7. reserve_and_issue (mutations at issue, then enqueue pending unit)
// ---------------------------------------------------------------------------
//   For each read tile: reader_count+=1; oldest_reader = min(existing-nonzero,
//   ds). For each write tile: writer=ds, version+=1, dirty=1. MK_LD: take the
//   LOWEST-index free page, set state=LOADING, owner=ds, key=page_key,
//   tile=dst, version=tile.version; record that page. MK_ALU: for each write
//   tile take the lowest free page, state=COMPUTING, owner=ds, key=0, tile=t,
//   version=tile.version; record pages in write order. Append a pending unit:
//   kind = LD_DONE/ALU_DONE/ST_DONE/WAIT_DONE/INC_DONE/FREE_DONE/BRANCH_DONE per
//   opcode, sm, decode_seq=ds, epoch=record.epoch, due_clock=clock+latency,
//   unit_seq=unit_seq_next (then unit_seq_next+=1). status=ISSUED.
//   MK_C_INSTR_ISSUE + EV_INSTR_ISSUE(sm,pc,ds,uid,U32MAX,0).
//
// ---------------------------------------------------------------------------
// 8. Pending-unit dispatch (op_advance) + per-event `aux` payloads
// ---------------------------------------------------------------------------
//   MK_U_LD_DONE : page->RESIDENT, tile.resident_page=page; MK_C_LD_DONE +
//                  EV_LD_DONE(tile=dst, aux=0). completes record.
//   MK_U_ALU_DONE: result_hash = alu_result_hash (see 9); each write page->
//                  RESIDENT, tile.resident_page=page; MK_C_ALU_DONE +
//                  EV_ALU_DONE(tile=U32MAX, aux=result_hash). completes.
//   MK_U_ST_DONE : MK_C_ST_DONE + EV_ST_DONE(aux=0). completes.
//   MK_U_WAIT_DONE: if counters[id] >= target: MK_C_WAIT_DONE + EV_WAIT_DONE(
//                  aux=counters[id]), completes. Else RE-ARM: append a new
//                  WAIT_DONE unit (due_clock=clock+1, fresh unit_seq),
//                  MK_C_WAIT_REARM + EV_WAIT_REARM(aux=target); does NOT
//                  complete the record.
//   MK_U_INC_DONE: counters[id]+=amount; MK_C_COUNTER_INC + EV_COUNTER_INC(
//                  aux=counters[id]). completes.
//   MK_U_FREE_DONE: let t=free_tile. "blocked" if t>=0 and (writer!=0 &&
//                  writer<ds) or (oldest_reader!=0 && oldest_reader<ds). If not
//                  blocked and t>=0: tile.freed=1, free its resident page if
//                  any (tile.resident_page=U32MAX), MK_C_TILE_FREE + EV_TILE_FREE
//                  (tile=t, aux=0), completes. Else REPLAY: status=REPLAY, push
//                  ds onto SM replay queue, MK_C_FREE_REPLAY + EV_FREE_REPLAY(
//                  tile = (t<0?U32MAX:t), aux=0); does NOT complete.
//   MK_U_BRANCH_DONE: resolved = (counters[id]>=target) ? taken_pc :
//                  fallthrough_pc. If resolved != fallthrough_pc: cancel all
//                  younger records of this SM+epoch with decode_seq>ds
//                  (iterate slots HIGH->LOW, same release/unwind as FLUSH,
//                  status=CANCELLED, MK_C_EPOCH_CANCEL + EV_EPOCH_CANCEL each;
//                  then recompute oldest_reader for their read tiles), set
//                  sm.pc=resolved, sm.epoch=epoch_seq_next++, MK_C_BRANCH_REDIRECT
//                  + EV_BRANCH_REDIRECT(pc=new sm.pc, aux=resolved). Else
//                  MK_C_BRANCH_NOOP + EV_BRANCH_NOOP(pc, aux=resolved). completes.
//   "completes record" = status=COMPLETED, MK_C_INSTR_COMPLETE +
//   EV_INSTR_COMPLETE(sm,pc,ds,uid,U32MAX,0).
//
// ---------------------------------------------------------------------------
// 9. alu_result_hash (folded into ALU record.result_hash and EV_ALU_DONE aux)
// ---------------------------------------------------------------------------
//   Fresh FNV: fold u64 compute_seed; then for each READ tile in record order:
//   u32 tile, u64 tile.version; then for each WRITE tile in record order:
//   u32 tile, u64 tile.version.
//
// ---------------------------------------------------------------------------
// 10. Snapshot hashes (recomputed fresh after every op; all use a fresh FNV)
// ---------------------------------------------------------------------------
//   sm_stream_hash: for sm=0..sm_count-1, fold u32 sm, u32 sm.pc, u64 sm.epoch,
//     u8 sm.halted; then for the DECODE queue: u32 size, then u64 each entry in
//     queue order; then the ISSUE queue (u32 size, u64 each); then the REPLAY
//     queue (u32 size, u64 each).
//   decoded_hash: OUTER loop sm=0..sm_count-1; INNER loop over records in
//     ASCENDING decode_seq (slot) order, skipping records not on this SM and
//     skipping RETIRED/CANCELLED records. Per record fold: u32 sm, u32 pc,
//     u64 epoch, u64 decode_seq, u64 instr_uid, u8 opcode, u8 status,
//     u8 hazard_code, u64 result_hash, u32 n_read then u32 each read tile,
//     u32 n_write then u32 each write tile, u32 n_pages then u32 each page.
//   tile_scoreboard_hash: for t=0..tile_count-1 fold u32 t, u64 version,
//     u64 writer, u64 oldest_reader, u64 reader_count, u32 resident_page,
//     u8 dirty, u8 freed.
//   page_hash: for sm=0..sm_count-1, for p=0..page_count_per_sm-1, fold u32 sm,
//     u32 p, u8 state, u64 page_key, u32 tile, u64 version, u64 owner.
//   pending_unit_hash: take all pending units, sort ASCENDING by (due_clock,
//     then unit_seq), then per unit fold u8 kind, u32 sm, u64 decode_seq,
//     u64 epoch, u64 due_clock, u64 unit_seq.
//   counter_hash: for c=0..counter_count-1 fold u32 c, u64 counters[c].
//
// ---------------------------------------------------------------------------
// 11. state_checksum (fresh FNV; folded AFTER the per-op op_index bump)
// ---------------------------------------------------------------------------
//   Note: op_index_out reports the op_index value BEFORE the bump; the
//   state_checksum folds op_index AFTER the bump (i.e. op_index_out + 1).
//   Fold order: u64 clock, u64 event_seq, u32 op_index(post-bump),
//   u64 decode_seq_next, u64 issue_seq_next, u64 unit_seq_next,
//   u64 retire_seq_next, u64 epoch_seq_next, u64 event_hash(=decoder_event_hash),
//   u64 sm_stream_hash, u64 decoded_hash, u64 tile_scoreboard_hash,
//   u64 page_hash, u64 pending_unit_hash, u64 counter_hash, then for
//   i=0..MK_COUNT_N-1 fold u64 (uint64_t)counts[i].
//
// ---------------------------------------------------------------------------
// 12. Scalar outputs
// ---------------------------------------------------------------------------
//   counts[MK_COUNT_N] (see MkCountIdx), op_index_out (pre-bump op_index),
//   clock_out (clock), event_seq_out (event_seq). All exact.
// ============================================================================

#endif  // MK_INSTR_DECODER_HAZARD_COMMON_H_
