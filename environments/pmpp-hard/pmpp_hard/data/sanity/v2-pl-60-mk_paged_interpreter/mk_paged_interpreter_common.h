// file: mk_paged_interpreter_common.h
//
// MK1 — No-Bubbles SM Interpreter with Paged Shared Memory (PMPP-Hard T60).
//
// A persistent, per-SM instruction interpreter modeled after Hazy Research's
// Llama megakernel: each SM fetches a preplanned instruction sequence,
// reserves shared-memory PAGES, overlaps loads, waits on integer dependency
// counters, computes a deterministic integer (FNV) reduction, stores, releases
// pages, and only then increments completion counters.
//
// The "compute" of every instruction is a trivial deterministic integer
// reduction (an FNV-1a-64 over the instruction's coordination identity). The
// DIFFICULTY is the COORDINATION STATE: the page pool (request/release +
// page-state sync), dependency counters (poll-to-target), the per-SM
// interpreter loop with its exact stage transitions, ordering & tie-breaks,
// and the pending-event timeline. Every graded output is an exact integer
// (FNV checksums / counts). No float, no tolerance.

#ifndef MK_PAGED_INTERPRETER_COMMON_H_
#define MK_PAGED_INTERPRETER_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MK_ABI_VERSION 1

// ------------------------------------------------------------------- limits
#define MK_MIN_SM 1
#define MK_MAX_SM 32
#define MK_MIN_PAGES 1
#define MK_MAX_PAGES 32
#define MK_MIN_PROGLEN 0
#define MK_MAX_PROGLEN 64
#define MK_MAX_PAGE_REQS 8
#define MK_MAX_WAITS 8
#define MK_MIN_COUNTERS 1
#define MK_MAX_COUNTERS 64
#define MK_MIN_PENDING 1
#define MK_MAX_PENDING 4096
#define MK_MAX_OPS 8192

// ------------------------------------------------------------------- enums
// Operation kinds (one per solution_run).
enum MkOpKind {
    MK_OP_BEGIN_PASS = 0,
    MK_OP_STEP_SM = 1,
    MK_OP_ADVANCE = 2,
    MK_OP_HOST_INC_COUNTER = 3,
    MK_OP_ABORT_PASS = 4,
    MK_OP_KIND_COUNT = 5
};

// Page request modes.
enum MkPageMode {
    MK_MODE_READ = 0,
    MK_MODE_WRITE = 1,
    MK_MODE_SCRATCH = 2
};

// Per-SM interpreter stages.
enum MkStage {
    MK_STAGE_IDLE = 0,
    MK_STAGE_ACQUIRE = 1,
    MK_STAGE_LOADING = 2,
    MK_STAGE_WAIT_DEPS = 3,
    MK_STAGE_COMPUTING = 4,
    MK_STAGE_STORE_READY = 5,
    MK_STAGE_STORING = 6,
    MK_STAGE_DONE = 7
};

// Page states.
enum MkPageState {
    MK_PG_FREE_EMPTY = 0,
    MK_PG_FREE_RESIDENT = 1,
    MK_PG_RESERVED_LOADING = 2,
    MK_PG_HELD_READY = 3,
    MK_PG_HELD_COMPUTING = 4,
    MK_PG_STORE_PENDING = 5
};

// Pending event kinds (the internal timed events).
enum MkPendingKind {
    MK_PEND_LOAD_DONE = 0,
    MK_PEND_COMPUTE_DONE = 1,
    MK_PEND_STORE_DONE = 2
};

// Emitted event kinds (in canonical ordinal order; see CONTRACT section below
// for the exact per-kind semantics and the per-event hash byte layout).
enum MkEventKind {
    MK_EV_PASS_BEGIN = 0,
    MK_EV_INSTR_FETCH = 1,
    MK_EV_PAGE_REUSE = 2,
    MK_EV_PAGE_SCRATCH = 3,
    MK_EV_PAGE_EVICT = 4,
    MK_EV_PAGE_LOAD_ISSUE = 5,
    MK_EV_PAGE_STALL = 6,
    MK_EV_LOAD_WAIT = 7,
    MK_EV_LOAD_DONE = 8,
    MK_EV_COUNTER_WAIT = 9,
    MK_EV_COMPUTE_ISSUE = 10,
    MK_EV_COMPUTE_DONE_INLINE = 11,
    MK_EV_COMPUTE_DONE = 12,
    MK_EV_COMPUTE_WAIT = 13,
    MK_EV_STORE_ISSUE = 14,
    MK_EV_STORE_WAIT = 15,
    MK_EV_STORE_DONE = 16,
    MK_EV_PAGE_RELEASE = 17,
    MK_EV_PAGE_RELEASE_PINNED_HINT = 18,
    MK_EV_COUNTER_INC = 19,
    MK_EV_INSTR_COMPLETE = 20,
    MK_EV_SM_PROGRAM_DONE = 21,
    MK_EV_SM_DONE_WAIT = 22,
    MK_EV_HOST_COUNTER_INC = 23,
    MK_EV_STALE_EVENT_DROP = 24,
    MK_EV_PAGE_ABORT_RELEASE = 25,
    MK_EV_PASS_ABORT = 26,
    MK_EV_INVALID = 27,
    MK_EV_KIND_COUNT = 28
};

// Output count indices (one accumulator per emitted event kind, in the same
// ordinal order as MkEventKind, EXCEPT that PAGE_RELEASE_PINNED_HINT folds into
// MK_C_PAGE_RELEASE so there is no separate slot for it; see CONTRACT section,
// "counts[] accumulation"), plus an invalid_count.
enum MkCountIdx {
    MK_C_PASS_BEGIN = 0,
    MK_C_INSTR_FETCH = 1,
    MK_C_PAGE_REUSE = 2,
    MK_C_PAGE_SCRATCH = 3,
    MK_C_PAGE_EVICT = 4,
    MK_C_PAGE_LOAD_ISSUE = 5,
    MK_C_PAGE_STALL = 6,
    MK_C_LOAD_WAIT = 7,
    MK_C_LOAD_DONE = 8,
    MK_C_COUNTER_WAIT = 9,
    MK_C_COMPUTE_ISSUE = 10,
    MK_C_COMPUTE_DONE_INLINE = 11,
    MK_C_COMPUTE_DONE = 12,
    MK_C_COMPUTE_WAIT = 13,
    MK_C_STORE_ISSUE = 14,
    MK_C_STORE_WAIT = 15,
    MK_C_STORE_DONE = 16,
    MK_C_PAGE_RELEASE = 17,
    MK_C_COUNTER_INC = 18,
    MK_C_INSTR_COMPLETE = 19,
    MK_C_SM_PROGRAM_DONE = 20,
    MK_C_SM_DONE_WAIT = 21,
    MK_C_HOST_COUNTER_INC = 22,
    MK_C_STALE_EVENT_DROP = 23,
    MK_C_PAGE_ABORT_RELEASE = 24,
    MK_C_PASS_ABORT = 25,
    MK_C_INVALID = 26,
    MK_COUNT_N = 27
};

// ---------------------------------------------------------------- flat program
// A static instruction, flattened. Page requests and waits live in side arrays
// referenced by (req_offset, wait_offset). alignas(8) for stable layout.
struct alignas(8) MkInstr {
    uint64_t instr_id;
    uint64_t load_latency;
    uint64_t compute_latency;
    uint64_t store_latency;
    uint64_t result_seed;
    uint64_t out_increment;
    uint32_t out_counter;          // UINT32_MAX if none
    uint32_t page_req_count;
    uint32_t wait_count;
    uint32_t req_offset;           // index into MkPageReq array
    uint32_t wait_offset;          // index into MkWaitRec array
    uint32_t reserved;
};

struct alignas(8) MkPageReq {
    uint64_t tile_id;
    uint8_t  mode;                 // MkPageMode
    uint8_t  release_after_store;  // 0 or 1
    uint8_t  pad0;
    uint8_t  pad1;
    uint32_t pad2;
};

struct alignas(8) MkWaitRec {
    uint32_t counter_id;
    uint32_t pad;
    uint64_t target;
};

// The problem spec. Scalar config + pointers to host-resident flat program.
// The pointers are consumed at solution_init time (copied to device). They are
// NOT touched by solution_run.
struct alignas(8) MkProblemSpec {
    int32_t abi_version;
    int32_t sm_count;
    int32_t pages_per_sm;
    int32_t counter_count;
    int32_t max_pending_events;
    int32_t max_program_len_per_sm;   // bound on per-SM instruction count
    int32_t total_instr;              // total flattened instructions
    int32_t total_reqs;               // total flattened page requests
    int32_t total_waits;              // total flattened waits
    int32_t flags;
    int32_t reserved0[2];

    // Flattened static program (host pointers, read at init only).
    const int32_t*    program_len;    // [sm_count]   instructions per SM
    const int32_t*    instr_offset;   // [sm_count]   first flat instr index per SM
    const MkInstr*    instrs;         // [total_instr]
    const MkPageReq*  reqs;           // [total_reqs]
    const MkWaitRec*  waits;          // [total_waits]
};

// One operation. alignas(8) POD.
struct alignas(8) MkRunSpec {
    int32_t abi_version;
    int32_t op_kind;               // MkOpKind
    int32_t step_id;
    int32_t a_sm;                  // STEP_SM target SM
    uint32_t a_counter;            // HOST_INC_COUNTER counter id
    uint32_t a_transition_limit;   // STEP_SM transition limit
    uint32_t a_max_events;         // ADVANCE max_events
    uint32_t reserved0;
    uint64_t a_pass_id;            // BEGIN_PASS new pass id
    uint64_t a_delta;              // ADVANCE delta
    uint64_t a_amount;             // HOST_INC_COUNTER amount
    uint64_t reserved1;
};

struct alignas(8) MkInputs {
    const void* reserved;
};

// Outputs written after EVERY op (all exact integers).
struct alignas(8) MkOutputs {
    int64_t*  counts;           // [MK_COUNT_N]
    int32_t*  op_index_out;     // [1]
    uint64_t* clock_out;        // [1]
    uint64_t* event_seq_out;    // [1]
    uint64_t* event_hash;       // [1] persistent running FNV over emitted events
    uint64_t* page_hash;        // [1] snapshot
    uint64_t* sm_hash;          // [1] snapshot
    uint64_t* counter_hash;     // [1] snapshot
    uint64_t* pending_hash;     // [1] snapshot
    uint64_t* state_checksum;   // [1] master FNV combining all
};

static_assert(sizeof(MkInstr) == 72, "MkInstr layout drift");
static_assert(sizeof(MkPageReq) == 16, "MkPageReq layout drift");
static_assert(sizeof(MkWaitRec) == 16, "MkWaitRec layout drift");
static_assert(sizeof(MkRunSpec) == 64, "MkRunSpec layout drift");
static_assert(sizeof(MkOutputs) == 80, "MkOutputs layout drift");

static inline int mk_validate_problem_spec(const MkProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MK_ABI_VERSION) return 0;
    if (spec->sm_count < MK_MIN_SM || spec->sm_count > MK_MAX_SM) return 0;
    if (spec->pages_per_sm < MK_MIN_PAGES || spec->pages_per_sm > MK_MAX_PAGES) return 0;
    if (spec->counter_count < MK_MIN_COUNTERS || spec->counter_count > MK_MAX_COUNTERS) return 0;
    if (spec->max_pending_events < MK_MIN_PENDING || spec->max_pending_events > MK_MAX_PENDING) return 0;
    if (spec->max_program_len_per_sm < MK_MIN_PROGLEN || spec->max_program_len_per_sm > MK_MAX_PROGLEN) return 0;
    if (spec->total_instr < 0 || spec->total_reqs < 0 || spec->total_waits < 0) return 0;
    if (!spec->program_len || !spec->instr_offset) return 0;
    if (spec->total_instr > 0 && !spec->instrs) return 0;
    if (spec->total_reqs > 0 && !spec->reqs) return 0;
    if (spec->total_waits > 0 && !spec->waits) return 0;
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

// === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section is fully self-contained and normative. A solver reading ONLY
// this header can reproduce EVERY graded output (counts[], op_index, clock,
// event_seq, event_hash, page_hash, sm_hash, counter_hash, pending_hash,
// state_checksum) bit-for-bit. There are no held-out rules. Everything below is
// exactly what the reference computes.
//
// ----------------------------------------------------------------------------
// 0. HASHING PRIMITIVE: FNV-1a-64
// ----------------------------------------------------------------------------
//   offset basis = 1469598103934665603  (0x14650FB0739D0383)
//   prime        = 1099511628211        (0x00000100000001B3)
//   byte(b): h ^= (uint64_t)b;  h *= prime;   // h is uint64_t, wraps mod 2^64
//   A "uN" field is absorbed as its N/8 raw bytes in LITTLE-ENDIAN order (the
//   raw memory bytes of the value on a little-endian host). I.e. u32 absorbs 4
//   bytes LSB-first, u64 absorbs 8 bytes LSB-first, u8 absorbs 1 byte.
//   Every hash that is "seeded" starts h = offset basis (1469598103934665603),
//   EXCEPT event_hash, which is a PERSISTENT running hash seeded once at reset
//   (see section 1) and then carried across all ops until the next reset.
//
// ----------------------------------------------------------------------------
// 1. PERSISTENT STATE + RESET
// ----------------------------------------------------------------------------
// State lives across solution_run calls. solution_reset (and solution_init)
// reset it to:
//   clock = 0
//   event_seq = 0
//   instr_instance_seq_next = 1          // next instruction-instance id
//   op_index = 0
//   event_hash = 1469598103934665603     // FNV offset basis (running hash seed)
//   pass_id = 0
//   pass_active = 0
//   counter[i] = 0                       for all i in [0, counter_count)
//   per-SM: pc=0, active_instance=0 (0 == none), stage=MK_STAGE_IDLE,
//           active_instr_id=0, active_compute_value=0, cur_req_count=0,
//           cur_wait_count=0, alloc_page[r]=UINT32_MAX, ready_mask[r]=0
//   per-page (sm,page): state=MK_PG_FREE_EMPTY, tile_id=0 (0==none),
//           mode=255 (255==none), owner_instance=0 (0==none),
//           request_index=UINT32_MAX, last_load_seq=0, last_release_seq=0
//   pending event list: empty
//   counts[i] = 0                        for all i in [0, MK_COUNT_N)
//
// Pages are flat-indexed as (sm * pages_per_sm + page). Per-instruction static
// fields (latencies, seeds, req/wait offsets+counts, out_counter, out_increment)
// are fetched from instrs[instr_offset[sm] + pc] into per-SM scratch at FETCH.
//
// ----------------------------------------------------------------------------
// 2. EMIT: the per-event hash byte layout (event_hash) + counts[]
// ----------------------------------------------------------------------------
// Every emitted event updates the PERSISTENT event_hash by absorbing EXACTLY
// these 10 fields, IN THIS ORDER, with these widths (LE), starting from the
// current running event_hash value:
//     emit(kind, sm_or, instr_id_or, instance_or, page_or, counter_or, value_or):
//       u8 : kind                              (MkEventKind ordinal, 0..27)
//       u64: event_seq                         (value BEFORE this event)
//       u32: op_index                          (current op_index, the 0-based
//                                               index of the op being processed;
//                                               it is incremented AFTER the op
//                                               body, see section 9)
//       u64: clock                             (current clock)
//       u32: sm_or       = (sm_or < 0) ? UINT32_MAX : (uint32_t)sm_or
//       u64: instr_id_or  (static instr_id, or UINT64_MAX when "none")
//       u64: instance_or  (instruction-instance id, or 0 when "none")
//       u32: page_or      (page id, or UINT32_MAX when "none")
//       u32: counter_or   (counter id, or UINT32_MAX when "none")
//       u64: value_or     (per-kind payload, or UINT64_MAX when "none")
// After absorbing, event_seq += 1 (the NEXT event sees the incremented seq).
//
// counts[] accumulation: each emit bumps exactly one accumulator by 1, indexed
// by kind_to_count(kind):
//   if kind <  MK_EV_PAGE_RELEASE_PINNED_HINT(18): index = kind            (0..17)
//   if kind == MK_EV_PAGE_RELEASE_PINNED_HINT(18): index = MK_C_PAGE_RELEASE(17)
//   if kind >  18:                                 index = kind - 1        (19..27 -> 18..26)
// So PINNED_HINT shares the PAGE_RELEASE accumulator; all later kinds shift down
// by one. MK_COUNT_N == 27 (28 event kinds minus the folded PINNED_HINT).
//
// "peek_seq()" used below == the CURRENT event_seq (the seq the NEXT emit will
// stamp). It is captured BEFORE that emit happens.
//
// ----------------------------------------------------------------------------
// 3. PER-KIND PAYLOAD (value_or) AND WHEN EACH EVENT IS EMITTED
// ----------------------------------------------------------------------------
// Listed with their (sm, instr_id_or, instance_or, page_or, counter_or, value_or)
// arguments to emit(). "MAX" = the all-ones sentinel of that width; "0" literal.
//
//  PASS_BEGIN(0):       (-1, MAX, 0, MAX, MAX, new_pass)            // BEGIN_PASS ok
//  INSTR_FETCH(1):      (s, instr_id, instance, MAX, MAX, MAX)      // IDLE->ACQUIRE fetch
//  PAGE_REUSE(2):       (s, instr_id, instance, page, MAX, tile_id) // READ hit FREE_RESIDENT same tile
//  PAGE_SCRATCH(3):     (s, instr_id, instance, page, MAX, tile_id) // SCRATCH mode alloc
//  PAGE_EVICT(4):       (s, instr_id, instance, page, MAX, old_tile)// reusing a resident page w/ diff tile
//  PAGE_LOAD_ISSUE(5):  (s, instr_id, instance, page, MAX, tile_id) // READ/WRITE non-reuse -> begin load
//  PAGE_STALL(6):       (s, instr_id, instance, MAX, MAX, MAX)      // ACQUIRE could not place all pages
//  LOAD_WAIT(7):        (s, instr_id, instance, page, MAX, (u64)r)  // LOADING stage, page r still loading
//  LOAD_DONE(8):        (s, MAX, instance, page, MAX, MAX)          // pending LOAD_DONE applied
//  COUNTER_WAIT(9):     (s, instr_id, instance, MAX, counter_id, target) // dep not yet met
//  COMPUTE_ISSUE(10):   (s, instr_id, instance, MAX, MAX, compute_value)  // compute_latency>0
//  COMPUTE_DONE_INLINE(11):(s, instr_id, instance, MAX, MAX, compute_value)// compute_latency==0
//  COMPUTE_DONE(12):    (s, instr_id, instance, MAX, MAX, compute_value)   // pending COMPUTE_DONE applied
//  COMPUTE_WAIT(13):    (s, instr_id, instance, MAX, MAX, MAX)      // STEP_SM hits COMPUTING stage
//  STORE_ISSUE(14):     (s, instr_id, instance, MAX, MAX, MAX)      // STORE_READY -> begin store
//  STORE_WAIT(15):      (s, instr_id, instance, MAX, MAX, MAX)      // STEP_SM hits STORING stage
//  STORE_DONE(16):      (s, instr_id, instance, MAX, MAX, MAX)      // pending STORE_DONE applied
//  PAGE_RELEASE(17):    (s, instr_id, instance, page, MAX, (u64)r)  // release_after_store==1
//  PAGE_RELEASE_PINNED_HINT(18):(s, instr_id, instance, page, MAX, (u64)r) // release_after_store!=1
//  COUNTER_INC(19):     (s, instr_id, instance, MAX, counter_id, NEW_counter_value) // device-side inc on store
//  INSTR_COMPLETE(20):  (s, instr_id, instance, MAX, MAX, MAX)      // instruction retired
//  SM_PROGRAM_DONE(21): (s, MAX, 0, MAX, MAX, MAX)                  // IDLE & pc==program_len
//  SM_DONE_WAIT(22):    (s, MAX, 0, MAX, MAX, MAX)                  // STEP_SM hits DONE stage
//  HOST_COUNTER_INC(23):(-1, MAX, 0, MAX, counter_id, NEW_counter_value) // HOST_INC_COUNTER ok
//  STALE_EVENT_DROP(24):(sm, MAX, instance, page_or_MAX, MAX, (u64)pending_kind) // dropped pending
//  PAGE_ABORT_RELEASE(25):(s, MAX, 0, page, MAX, (u64)prior_page_state) // ABORT releasing non-free page
//  PASS_ABORT(26):      (-1, MAX, 0, MAX, MAX, MAX)                 // ABORT_PASS end
//  INVALID(27):         (-1, MAX, 0, MAX, MAX, MAX)                 // any rejected op
//
// instr_id_or for LOAD_DONE/STORE-side STALE_EVENT_DROP is UINT64_MAX (the
// emitting code passes MAX there). For STALE_EVENT_DROP, page_or is the pending
// event's page_id (UINT32_MAX for compute/store-kind pending events; the actual
// page for load-kind). value_or is the pending event's kind cast to u64.
//
// ----------------------------------------------------------------------------
// 4. THE 5 OPS (op_kind dispatch). Each op runs its body, then op_index++ and
//    the snapshots/state_checksum are produced (section 9).
// ----------------------------------------------------------------------------
//
// 4a. MK_OP_BEGIN_PASS (a_pass_id = new_pass)
//   Validity: reject (emit INVALID, no other change) UNLESS:
//     - if pass_active==1: every SM stage must be MK_STAGE_DONE AND pending
//       list must be empty; otherwise INVALID.
//     - the pending list must be empty (checked again regardless of active);
//       otherwise INVALID.
//   On accept:
//     pass_active=1; pass_id=new_pass; zero all counters; reset every SM
//     (pc=0, IDLE, no active instance, alloc/ready cleared); reset every page to
//     FREE_EMPTY (tile=0, mode=255, owner=0, reqidx=MAX, loadseq=0, relseq=0);
//     then emit PASS_BEGIN(-1, MAX, 0, MAX, MAX, new_pass).
//
// 4b. MK_OP_STEP_SM (a_sm = sm_idx, a_transition_limit = lim)
//   Validity: if sm_idx<0 || sm_idx>=sm_count || pass_active==0 || lim==0:
//     emit INVALID and stop.
//   Else run up to `lim` micro-transitions on SM sm_idx; each transition is
//   step_one_transition() (section 5). If a transition returns "stop"
//   (it emitted a stall/wait/done-marker), break early.
//
// 4c. MK_OP_ADVANCE (a_delta = delta, a_max_events = max_events)
//   clock += delta (wraps mod 2^64). If max_events==0: done (clock-only).
//   Else, repeat up to max_events times: among pending events with
//   due_clock <= clock, pick the canonical-MIN (section 6 ordering), remove it,
//   and process it (section 7). Stop early when no due event remains.
//   NEVER emits INVALID; an ADVANCE with no due events simply changes clock.
//
// 4d. MK_OP_HOST_INC_COUNTER (a_counter = cid, a_amount = amount)
//   If cid >= counter_count: emit INVALID. Else counter[cid] += amount (mod
//   2^64); emit HOST_COUNTER_INC(-1, MAX, 0, MAX, cid, counter[cid] AFTER inc).
//
// 4e. MK_OP_ABORT_PASS
//   If pass_active==0: emit INVALID. Else:
//     1. Drain pending events in canonical order (section 6): for each, emit
//        STALE_EVENT_DROP(sm, MAX, instance, page_id, MAX, (u64)kind) where
//        page_id is the event's page (UINT32_MAX if it had none). Clear list.
//     2. For every SM: active_instance=0, stage=MK_STAGE_DONE, clear alloc/ready.
//     3. For every page in (sm,page) order: if it is NOT free (state is not
//        FREE_EMPTY and not FREE_RESIDENT), emit
//        PAGE_ABORT_RELEASE(s, MAX, 0, p, MAX, (u64)prior_state) BEFORE
//        resetting; then reset that page to FREE_EMPTY/cleared.
//     4. pass_active=0; emit PASS_ABORT(-1, MAX, 0, MAX, MAX, MAX).
//   Any unrecognized op_kind also emits INVALID.
//
// ----------------------------------------------------------------------------
// 5. STEP_SM MICRO-TRANSITION STATE MACHINE (one call to step_one_transition)
//    Dispatch on the SM's current stage. "stop" => caller breaks the lim-loop.
// ----------------------------------------------------------------------------
//  MK_STAGE_DONE(7):  emit SM_DONE_WAIT(s,MAX,0,MAX,MAX,MAX); return stop.
//  MK_STAGE_IDLE(0):
//     if pc == program_len[s]: stage=DONE; emit SM_PROGRAM_DONE(s,MAX,0,...);
//        return stop.
//     else (fetch next instr): active_instance = instr_instance_seq_next++;
//        load instr static fields from instrs[instr_offset[s]+pc]; clear
//        alloc/ready; stage=ACQUIRE; emit INSTR_FETCH(s,instr_id,instance,...);
//        return continue.
//  MK_STAGE_ACQUIRE(1): run ACQUIRE (section 5a). returns stop iff stalled.
//  MK_STAGE_LOADING(2):
//     scan requests r=0..cur_req_count-1; for the LOWEST r whose alloc_page[r]
//     is a page still in RESERVED_LOADING: emit
//        LOAD_WAIT(s,instr_id,instance,page,MAX,(u64)r); return stop.
//     if none loading: stage=WAIT_DEPS; return continue.
//  MK_STAGE_WAIT_DEPS(3):
//     scan waits w=0..cur_wait_count-1 in order; for the FIRST wait with
//        counter[wr.counter_id] < wr.target: emit
//        COUNTER_WAIT(s,instr_id,instance,MAX,counter_id,target); return stop.
//     if all satisfied: compute the instruction value (section 5b), store it in
//        active_compute_value, then:
//        if cur_compute_latency==0: stage=STORE_READY; emit
//           COMPUTE_DONE_INLINE(s,instr_id,instance,MAX,MAX,compute_value);
//           return continue.
//        else: set every allocated page's state to MK_PG_HELD_COMPUTING; push a
//           pending COMPUTE_DONE (section 6 fields) due at clock+cur_compute_latency;
//           stage=COMPUTING; emit COMPUTE_ISSUE(...,compute_value); return continue.
//  MK_STAGE_COMPUTING(4): emit COMPUTE_WAIT(s,instr_id,instance,...); return stop.
//  MK_STAGE_STORE_READY(5):
//     set every allocated page's state to MK_PG_STORE_PENDING; push a pending
//     STORE_DONE due at clock+cur_store_latency; stage=STORING; emit
//     STORE_ISSUE(s,instr_id,instance,...); return continue.
//  MK_STAGE_STORING(6): emit STORE_WAIT(s,instr_id,instance,...); return stop.
//
// 5a. ACQUIRE (atomic page placement for all cur_req_count requests):
//   Phase 1 (tentative pick, request order r=0..nreq-1; do NOT mutate pages):
//     - chosen[r]=UINT32_MAX, is_reuse[r]=0 initially.
//     - if reqs[r].mode==READ: scan pages p=0..PAGES-1 ascending; pick the
//       LOWEST p NOT already in chosen[0..r-1] that is FREE_RESIDENT with
//       tile_id==reqs[r].tile_id -> reuse=1.
//     - if still unpicked: scan p ascending; pick LOWEST p not already chosen
//       that is FREE_EMPTY or FREE_RESIDENT.
//     - if still unpicked: stalled=true; break.
//   If stalled: emit PAGE_STALL(s,instr_id,instance,MAX,MAX,MAX); stage stays
//     ACQUIRE; return stop. (No pages mutated.)
//   Phase 2 (commit, request order r=0..nreq-1): for pid=chosen[r], page idx:
//     alloc_page[r]=pid.
//     - if is_reuse[r]: page.state=HELD_READY, owner=active_instance,
//         request_index=r, ready_mask[r]=1; emit
//         PAGE_REUSE(s,instr_id,instance,pid,MAX,reqs[r].tile_id); continue.
//     - else (not reuse): eviction event first, if applicable:
//         if page.state==FREE_RESIDENT and page.tile != reqs[r].tile_id: emit
//            PAGE_EVICT(...,pid,...,old page.tile_id);
//         else if page.state==FREE_RESIDENT and reqs[r].mode!=READ and
//            page.tile==reqs[r].tile_id: emit PAGE_EVICT(...,old tile).
//       then place:
//         if reqs[r].mode==SCRATCH: page.state=HELD_READY, tile=reqs[r].tile_id,
//            mode=SCRATCH, owner=active_instance, request_index=r, ready_mask[r]=1;
//            emit PAGE_SCRATCH(...,pid,...,reqs[r].tile_id).
//         else (READ or WRITE non-reuse): page.state=RESERVED_LOADING,
//            tile=reqs[r].tile_id, mode=reqs[r].mode, owner=active_instance,
//            request_index=r, ready_mask[r]=0; push pending LOAD_DONE for (s,
//            active_instance, pid) due at clock+cur_load_latency;
//            emit PAGE_LOAD_ISSUE(...,pid,...,reqs[r].tile_id). any_loading=true.
//   After commit: stage = any_loading ? LOADING : WAIT_DEPS; return continue.
//
// 5b. COMPUTE VALUE (active_compute_value) — a FRESH FNV-1a-64 (seed=offset
//   basis), absorbed in EXACTLY this order:
//     u64 pass_id
//     u64 active_instr_id
//     u64 active_instance
//     for r = 0..cur_req_count-1:   u64 reqs[r].tile_id;
//                                   u32 alloc_page[r];
//                                   u8  reqs[r].mode
//     for w = 0..cur_wait_count-1:  u32 waits[w].counter_id;
//                                   u64 waits[w].target;
//                                   u64 counter[waits[w].counter_id] (observed now)
//     u64 cur_result_seed
//   The resulting h is active_compute_value (carried as the COMPUTE_* payload).
//
// ----------------------------------------------------------------------------
// 6. PENDING EVENTS: fields + canonical order + push
// ----------------------------------------------------------------------------
// A pending event carries: due_clock(u64), create_event_seq(u64, == event_seq
// at push time, i.e. peek_seq()), kind(u8: MkPendingKind), sm(int32),
// instance(u64), page_id(u32, UINT32_MAX if not page-specific).
// Canonical order pend_less(a,b) compares, in this priority, ascending:
//   1. due_clock   2. create_event_seq   3. sm   4. instance   5. page_id
// Pushes:
//   - ACQUIRE READ/WRITE non-reuse: LOAD_DONE, page_id=pid, due=clock+load_lat.
//   - WAIT_DEPS compute (lat>0): COMPUTE_DONE, page_id=MAX, due=clock+comp_lat.
//   - STORE_READY: STORE_DONE, page_id=MAX, due=clock+store_lat.
//
// ----------------------------------------------------------------------------
// 7. PROCESSING A DUE PENDING EVENT (during ADVANCE)
// ----------------------------------------------------------------------------
//  MK_PEND_LOAD_DONE(0): idx = page (sm,page_id).
//     if page.owner_instance==ev.instance AND page.state==RESERVED_LOADING:
//        page.state=HELD_READY; page.last_load_seq = peek_seq() (the seq of the
//        LOAD_DONE we are about to emit); if the SM's active_instance still
//        ==ev.instance, set ready_mask[r]=1 for every r with alloc_page[r]==page;
//        emit LOAD_DONE(sm, MAX, instance, page, MAX, MAX).
//     else: emit STALE_EVENT_DROP(sm, MAX, instance, page, MAX, (u64)kind).
//  MK_PEND_COMPUTE_DONE(1):
//     if SM.active_instance==ev.instance AND SM.stage==COMPUTING:
//        stage=STORE_READY; emit COMPUTE_DONE(sm, instr_id, instance, MAX, MAX,
//        active_compute_value).
//     else: emit STALE_EVENT_DROP(sm, MAX, instance, MAX, MAX, (u64)kind).
//  MK_PEND_STORE_DONE(2):
//     if SM.active_instance==ev.instance AND SM.stage==STORING:
//        (a) emit STORE_DONE(sm, instr_id, instance, MAX, MAX, MAX).
//        (b) page releases in request order r=0..cur_req_count-1: for each
//            allocated pid (skip UINT32_MAX): page.state=FREE_RESIDENT,
//            owner=0, request_index=MAX, last_release_seq=peek_seq() (seq of the
//            release about to emit); if reqs[r].release_after_store==1 emit
//            PAGE_RELEASE(sm,instr_id,instance,pid,MAX,(u64)r) else emit
//            PAGE_RELEASE_PINNED_HINT(...same args...).
//        (c) if cur_out_counter != UINT32_MAX: counter[cid] += cur_out_increment
//            (mod 2^64); emit COUNTER_INC(sm,instr_id,instance,MAX,cid,
//            counter[cid] AFTER inc).
//        (d) retire: active_instance=0; pc+=1; stage=IDLE; clear alloc/ready;
//            emit INSTR_COMPLETE(sm, done_instr_id, instance, MAX, MAX, MAX).
//     else: emit STALE_EVENT_DROP(sm, MAX, instance, MAX, MAX, (u64)kind).
//  Note: the emit order within STORE_DONE is exactly (a)->(b per r)->(c)->(d).
//
// ----------------------------------------------------------------------------
// 8. SNAPSHOT HASHES (recomputed fresh after every op, NOT persistent)
//    Each is a fresh FNV-1a-64 (seed=offset basis).
// ----------------------------------------------------------------------------
// page_hash: for s=0..SM-1, for p=0..PAGES-1 (row-major, s outer, p inner):
//     u32 s; u32 p; u8 page.state; u64 page.tile_id; u8 page.mode;
//     u64 page.owner_instance; u32 page.request_index; u64 page.last_load_seq;
//     u64 page.last_release_seq.
// sm_hash: for s=0..SM-1:
//     u32 s; u32 pc; u8 stage; u64 active_instance;
//     u64 (active_instance==0 ? UINT64_MAX : active_instr_id);
//     u64 active_compute_value;
//     then n = (active_instance==0 ? 0 : cur_req_count); for r=0..n-1: u32 alloc_page[r].
// counter_hash: for c=0..NCTR-1:  u32 c; u64 counter[c].
// pending_hash: take the pending list SORTED by pend_less (canonical order);
//     for each event in that order: u64 due_clock; u64 create_event_seq;
//     u8 kind; u32 (uint32_t)sm; u64 instance; u32 page_id.
//
// ----------------------------------------------------------------------------
// 9. OP EPILOGUE: op_index and state_checksum (master)
// ----------------------------------------------------------------------------
// After the op body: this_op = op_index (this is the value emitted as op_index
// out, and also the value absorbed by every event during this op via the EMIT
// layout above — events use op_index BEFORE the increment); then op_index += 1.
// Outputs op_index_out = this_op, clock_out = clock, event_seq_out = event_seq,
// event_hash = event_hash (persistent), and the four snapshot hashes.
//
// state_checksum: a fresh FNV-1a-64 (seed=offset basis), absorbing IN THIS
// ORDER (note: op_index here is the ALREADY-INCREMENTED value, this_op+1):
//     u64 clock
//     u64 event_seq
//     u32 op_index            (== this_op + 1, the post-increment value)
//     u8  pass_active
//     u64 pass_id
//     u64 instr_instance_seq_next
//     u64 event_hash          (the persistent running event hash)
//     u64 page_hash
//     u64 sm_hash
//     u64 counter_hash
//     u64 pending_hash
//     for i=0..MK_COUNT_N-1:  u64 (uint64_t)counts[i]
//
// ----------------------------------------------------------------------------
// 10. EDGE / OOM / WRAP / TIE-BREAK SUMMARY
// ----------------------------------------------------------------------------
//   - ACQUIRE OOM: if any request cannot place a page, PAGE_STALL only; no page
//     mutated; SM stays ACQUIRE (retry on next STEP_SM).
//   - Stale-event drop: a due pending event whose target SM/page no longer
//     matches (owner/instance/stage changed) is dropped as STALE_EVENT_DROP
//     (does NOT mutate state besides removal from the pending list).
//   - All counter and clock arithmetic wraps mod 2^64.
//   - Page tie-break in ACQUIRE: always LOWEST eligible page id first; READ
//     reuse (same-tile FREE_RESIDENT) preferred over fresh placement.
//   - Pending tie-break everywhere (ADVANCE select, ABORT drain, pending_hash):
//     (due_clock, create_event_seq, sm, instance, page_id) ascending.
//   - SM processing order in page/sm/counter snapshots and ABORT page release:
//     ascending sm, then ascending page/counter.
//   - "none" sentinels: instr_id=UINT64_MAX, instance=0, page=UINT32_MAX,
//     counter=UINT32_MAX, value=UINT64_MAX, tile_id=0, mode=255, owner=0.
// === END CONTRACT ===

#endif  // MK_PAGED_INTERPRETER_COMMON_H_
