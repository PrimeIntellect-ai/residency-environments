// file: incremental_gc_weakref_common.h
//
// T53 — Incremental Deterministic GC with Weak/Ephemeron/Finalizer Semantics.
//
// A persistent heap with mutator writes interleaved with incremental
// minor/full tri-color mark-sweep collection, remembered sets, weak
// references, ephemerons (key-conditional value reachability), finalizer
// resurrection, survivor promotion, and exact sweep/finalization event
// streams. Every output is an exact integer or an FNV-1a-64 checksum.
//
// This file defines the stable ABI shared by the reference implementation,
// the naive implementation, the host oracle, and the test/bench harnesses.

#ifndef INCREMENTAL_GC_WEAKREF_COMMON_H_
#define INCREMENTAL_GC_WEAKREF_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define IGCW_ABI_VERSION 1

// Capacity ceilings (the spec's tunable fields are bounded by these).
#define IGCW_MIN_OBJECTS 1
#define IGCW_MAX_OBJECTS 4096
#define IGCW_MAX_ROOTS 1024
#define IGCW_MAX_STRONG_SLOTS 16
#define IGCW_MAX_WEAK_SLOTS 16
#define IGCW_MAX_EPHEMERONS 4096
#define IGCW_MAX_MARK_QUEUE 8192
#define IGCW_MAX_FINALIZER_QUEUE 4096

// Generations.
#define IGCW_GEN_YOUNG 0
#define IGCW_GEN_OLD 1

// Tri-colors.
#define IGCW_WHITE 0
#define IGCW_GREY 1
#define IGCW_BLACK 2

// GC controller phase.
#define IGCW_PHASE_IDLE 0
#define IGCW_PHASE_MARK 1
#define IGCW_PHASE_EPHEMERON 2
#define IGCW_PHASE_FINALIZE_SCAN 3
#define IGCW_PHASE_RESURRECT_MARK 4
#define IGCW_PHASE_WEAK_CLEAR 5
#define IGCW_PHASE_SWEEP 6

// GC controller mode.
#define IGCW_MODE_NONE 0
#define IGCW_MODE_MINOR 1
#define IGCW_MODE_FULL 2

// Operation opcodes (one op per solution_run).
#define IGCW_OP_ALLOC 0
#define IGCW_OP_SET_ROOT 1
#define IGCW_OP_CLEAR_ROOT 2
#define IGCW_OP_SET_STRONG 3
#define IGCW_OP_CLEAR_STRONG 4
#define IGCW_OP_SET_WEAK 5
#define IGCW_OP_SET_EPHEMERON 6
#define IGCW_OP_DELETE_EPHEMERON 7
#define IGCW_OP_START_MINOR 8
#define IGCW_OP_START_FULL 9
#define IGCW_OP_GC_STEP 10
#define IGCW_OP_RUN_FINALIZERS 11

// Event kinds (emission order is hashed into gc_event_hash).
#define IGCW_EV_ALLOC_OK 0
#define IGCW_EV_ALLOC_BLACK 1
#define IGCW_EV_ALLOC_OOM 2
#define IGCW_EV_ROOT_SET 3
#define IGCW_EV_ROOT_CLEAR 4
#define IGCW_EV_STRONG_SET 5
#define IGCW_EV_STRONG_CLEAR 6
#define IGCW_EV_WEAK_SET 7
#define IGCW_EV_EPHEMERON_SET 8
#define IGCW_EV_EPHEMERON_DELETE 9
#define IGCW_EV_EPHEMERON_OOM 10
#define IGCW_EV_REMEMBERED_ADD 11
#define IGCW_EV_REMEMBERED_DROP 12
#define IGCW_EV_GC_START_MINOR 13
#define IGCW_EV_GC_START_FULL 14
#define IGCW_EV_BARRIER_MARK 15
#define IGCW_EV_MARK_BLACK 16
#define IGCW_EV_MARK_GREY 17
#define IGCW_EV_PHASE_EPHEMERON 18
#define IGCW_EV_EPHEMERON_MARK 19
#define IGCW_EV_PHASE_MARK 20
#define IGCW_EV_PHASE_FINALIZE_SCAN 21
#define IGCW_EV_FINALIZER_ENQUEUE 22
#define IGCW_EV_FINALIZER_QUEUE_FULL 23
#define IGCW_EV_PHASE_RESURRECT_MARK 24
#define IGCW_EV_PHASE_WEAK_CLEAR 25
#define IGCW_EV_WEAK_CLEAR_FIELD 26
#define IGCW_EV_EPHEMERON_CLEAR_SIDE 27
#define IGCW_EV_PHASE_SWEEP 28
#define IGCW_EV_SWEEP_FREE 29
#define IGCW_EV_EPHEMERON_OWNER_FREE 30
#define IGCW_EV_SWEEP_KEEP 31
#define IGCW_EV_PROMOTE_OLD 32
#define IGCW_EV_GC_END 33
#define IGCW_EV_FINALIZER_RUN 34
#define IGCW_EV_FINALIZER_RESURRECT 35
#define IGCW_EV_FINALIZER_SKIP_ABSENT 36
#define IGCW_EV_INVALID 37

#define IGCW_U32_MAX 0xFFFFFFFFu
#define IGCW_U64_MAX 0xFFFFFFFFFFFFFFFFULL

/*
CONTRACT: incremental_gc_weakref

Persistent heap + incremental tri-color mark-sweep GC. Each solution_run
executes exactly ONE operation (selected by run.opcode) against the
persistent state, then emits the full count vector and six checksums.

Persistent scalars (wrap mod 2^64):
  event_seq = 0; obj_id_next = 1; alloc_seq_next = 1; gc_cycle_id = 0.

Object table keyed by obj_id (1..max_objects), dense + present[] flag:
  size:u64; generation; age:u8; color; allocated_during_gc:u8; alloc_seq;
  finalizer_tag:u32; finalizer_root_slot (UINT32_MAX = none); finalized:u8;
  in_finalizer_queue:u8; strong[strong_slots]; weak[weak_slots].

Ephemeron table keyed by ephemeron_id (slot dense, 0..max_ephemerons-1):
  ephemeron_id; owner_obj; key_obj(0=none); value_obj(0=none); create_seq.

Remembered set: pairs (src_obj, slot), canonical order src then slot.
Free-id list: ascending object ids.
Finalizer queue: object ids in enqueue order (FIFO).

GC controller: phase, mode, gc_cycle_id, mark_queue FIFO, scan_cursor_obj,
  scan_cursor_field, sweep_cursor_obj, ephemeron_changed, in_cset[].

The COMPLETE, normative executable semantics of every operation, the exact
meaning and index of each of the 31 count-vector entries, and the exact
FNV-1a-64 serialization of all six checksums are documented inline below in
the "=== CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ==="
section. That section is the authoritative specification; the oracle and the
reference/naive .cu files implement exactly it. The test harness validates
both against the oracle, with exact replay, permuted-op-stream replay, guard
buffers, and input-immutability checks.

Rules:
  - solution_init may allocate persistent state (cudaMalloc allowed).
  - solution_run may NOT call cudaMalloc/cudaFree; it may launch kernels and
    use the provided workspace.
  - All graded outputs are exact integers / FNV-1a-64 checksums.
*/

struct alignas(8) IgcwProblemSpec {
    int32_t abi_version;
    int32_t max_objects;
    int32_t root_count;
    int32_t strong_slots_per_object;
    int32_t weak_slots_per_object;
    int32_t max_ephemerons;
    int32_t max_mark_queue;
    int32_t max_finalizer_queue;
    int32_t young_survive_threshold;
    int32_t reserved[7];
};

struct alignas(8) IgcwRunSpec {
    int32_t abi_version;
    int32_t opcode;
    int32_t op_index;     // global op index, hashed into events
    // Generic argument slots; meaning depends on opcode.
    int32_t a0;
    int32_t a1;
    int32_t a2;
    int32_t a3;
    int64_t size_arg;     // ALLOC size (u64-valued)
    int32_t reserved[6];
};

// ALLOC:           a0=size? no (size in size_arg), a0=generation,
//                  a1=finalizer_tag, a2=finalizer_root_slot
// SET_ROOT:        a0=root_id, a1=obj_id
// CLEAR_ROOT:      a0=root_id
// SET_STRONG:      a0=src_obj, a1=slot, a2=dst_obj
// CLEAR_STRONG:    a0=src_obj, a1=slot
// SET_WEAK:        a0=src_obj, a1=slot, a2=dst_obj
// SET_EPHEMERON:   a0=ephemeron_id, a1=owner_obj, a2=key_obj, a3=value_obj
// DELETE_EPHEMERON:a0=ephemeron_id
// START_MINOR / START_FULL: (no args)
// GC_STEP:         a0=mark_budget, a1=sweep_budget
// RUN_FINALIZERS:  a0=limit

struct alignas(8) IgcwInputs {
    int32_t reserved;  // all operands carried in IgcwRunSpec.
    int32_t pad;
};

// Output: full count vector + six checksums.
// The exact 0..30 meaning of every `counts` entry is in the normative
// CONTRACT section below (see "COUNTS VECTOR").
#define IGCW_NUM_COUNTS 31
struct alignas(8) IgcwOutputs {
    int32_t* counts;             // IGCW_NUM_COUNTS entries (indices defined in CONTRACT section)
    uint64_t* gc_event_hash;
    uint64_t* heap_hash;
    uint64_t* root_hash;
    uint64_t* ephemeron_hash;
    uint64_t* remembered_hash;
    uint64_t* gc_controller_hash;
    int32_t* invalid_flag;       // 1 if this op was invalid, else 0
};

static_assert(sizeof(IgcwProblemSpec) == 64, "IgcwProblemSpec layout drift");
static_assert(sizeof(IgcwRunSpec) == 64, "IgcwRunSpec layout drift");
static_assert(sizeof(IgcwOutputs) == 64, "IgcwOutputs layout drift");

static inline int igcw_validate_problem_spec(const IgcwProblemSpec* s) {
    if (!s) return 0;
    if (s->abi_version != IGCW_ABI_VERSION) return 0;
    if (s->max_objects < IGCW_MIN_OBJECTS || s->max_objects > IGCW_MAX_OBJECTS) return 0;
    if (s->root_count < 0 || s->root_count > IGCW_MAX_ROOTS) return 0;
    if (s->strong_slots_per_object < 0 || s->strong_slots_per_object > IGCW_MAX_STRONG_SLOTS) return 0;
    if (s->weak_slots_per_object < 0 || s->weak_slots_per_object > IGCW_MAX_WEAK_SLOTS) return 0;
    if (s->max_ephemerons < 0 || s->max_ephemerons > IGCW_MAX_EPHEMERONS) return 0;
    if (s->max_mark_queue < 1 || s->max_mark_queue > IGCW_MAX_MARK_QUEUE) return 0;
    if (s->max_finalizer_queue < 0 || s->max_finalizer_queue > IGCW_MAX_FINALIZER_QUEUE) return 0;
    if (s->young_survive_threshold < 1) return 0;
    return 1;
}

static inline int igcw_validate_run_spec(const IgcwRunSpec* r) {
    if (!r) return 0;
    if (r->abi_version != IGCW_ABI_VERSION) return 0;
    if (r->opcode < 0 || r->opcode > IGCW_OP_RUN_FINALIZERS) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const IgcwProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const IgcwProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const IgcwRunSpec* run,
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
// This section is the authoritative, self-contained specification. Every rule
// the grader enforces is here. The oracle and the reference implement EXACTLY
// this; nothing is deferred. Indices, widths, orders, sentinels, and
// tie-breaks below are normative. All integer widths are little-endian as
// produced by feeding the native-endian bytes of the value into FNV (the
// reference/oracle run on little-endian; hashes are defined over those bytes).
//
// --------------------------------------------------------------------------
// 0. PER-OP CONTRACT
// --------------------------------------------------------------------------
// Each solution_run executes exactly ONE op (run.opcode) against persistent
// state, then writes ALL outputs. At the START of every op:
//   - counts[0..30] are ALL zeroed (counts are PER-OP deltas, never cumulative).
//   - op_invalid is set to 0.
//   - cur_op_index = (uint32_t)run.op_index.
// After the op body runs, the six checksums are computed over the resulting
// FULL persistent state (NOT just the touched part) and invalid_flag = op_invalid.
//
// Persistent scalars (all u64, wrap mod 2^64; initial values from reset):
//   event_seq=0; obj_id_next=1; alloc_seq_next=1; gc_cycle_id=0.
// event_seq increments by 1 per emitted event (including IGCW_EV_INVALID) and
// is NEVER reset between ops (it is persistent / monotonic, wrapping mod 2^64).
//
// --------------------------------------------------------------------------
// 1. EVENT EMISSION + gc_event_hash (FNV-1a-64)
// --------------------------------------------------------------------------
// gc_event_hash is a PERSISTENT running FNV-1a-64 accumulator. IMPORTANT: it is
// seeded ONCE at reset to 0 -- i.e. it starts at 0, NOT at the FNV offset basis.
// Only the FNV multiplier 1099511628211 is applied when folding each byte
// (h = (h XOR byte) * 1099511628211); the initial accumulator value is 0.
// (Contrast the state-snapshot checksums in section 5, which DO each seed fresh
// from IGCW_FNV_INIT=1469598103934665603.)
// It is NOT re-seeded per op; it accumulates across the whole op stream. Its
// value is read out (unchanged) at the end of every op as gc_event_hash.
//
// emit(kind, obj, slot, root_id, eph, aux) folds, IN THIS EXACT FIELD ORDER:
//      u8  kind            (the IGCW_EV_* event-kind constant)
//      u64 event_seq       (value BEFORE this event's increment)
//      u32 cur_op_index    (run.op_index of the current op)
//      u64 gc_cycle_id     (current controller cycle id)
//      u64 obj             (object id, or 0 / sentinel per event)
//      u32 slot            (slot index, or IGCW_U32_MAX, or 0/1 side selector)
//      u32 root_id         (root id, or IGCW_U32_MAX)
//      u64 eph             (ephemeron id, or IGCW_U64_MAX)
//      u64 aux             (event-specific extra payload; see each event)
// then event_seq += 1.
// mark_invalid() sets op_invalid=1 and emits exactly:
//   IGCW_EV_INVALID, obj=0, slot=IGCW_U32_MAX, root_id=IGCW_U32_MAX,
//   eph=IGCW_U64_MAX, aux=0.
//
// The per-event (obj,slot,root_id,eph,aux) payloads, normatively:
//   ALLOC_OK         obj=id, slot=MAX, root=MAX, eph=MAX, aux=size
//   ALLOC_BLACK      obj=id, slot=MAX, root=MAX, eph=MAX, aux=size
//   ALLOC_OOM        obj=0,  slot=MAX, root=MAX, eph=MAX, aux=0
//   ROOT_SET         obj=newobj(0 if cleared-to-0), slot=MAX, root=rid, eph=MAX, aux=0
//   ROOT_CLEAR       obj=0,  slot=MAX, root=rid, eph=MAX, aux=0
//   STRONG_SET       obj=src,slot=slot, root=MAX, eph=MAX, aux=(u32)dst
//   STRONG_CLEAR     obj=src,slot=slot, root=MAX, eph=MAX, aux=0
//   WEAK_SET         obj=src,slot=slot, root=MAX, eph=MAX, aux=(u32)dst
//   EPHEMERON_SET    obj=owner, slot=MAX, root=MAX, eph=eid, aux=(u32)value
//   EPHEMERON_DELETE obj=0,  slot=MAX, root=MAX, eph=eid, aux=0
//   EPHEMERON_OOM    obj=0,  slot=MAX, root=MAX, eph=eid, aux=0
//   REMEMBERED_ADD   obj=src,slot=slot, root=MAX, eph=MAX, aux=0
//   REMEMBERED_DROP  obj=src,slot=slot, root=MAX, eph=MAX, aux=0
//   GC_START_MINOR   obj=0,  slot=MAX, root=MAX, eph=MAX, aux=gc_cycle_id
//   GC_START_FULL    obj=0,  slot=MAX, root=MAX, eph=MAX, aux=gc_cycle_id
//   BARRIER_MARK     obj=marked-obj; slot=slot for STRONG (else MAX);
//                    root=rid for SET_ROOT (else MAX); eph=eid for EPHEMERON
//                    (else MAX); aux=0. (Three distinct callers — see ops 1/3/6.)
//   MARK_BLACK       obj=id, slot=MAX, root=MAX, eph=MAX, aux=0
//   MARK_GREY        obj=tgt,slot=strong-slot, root=MAX, eph=MAX, aux=0
//   PHASE_EPHEMERON  obj=0,  slot=MAX, root=MAX, eph=MAX, aux=0
//   EPHEMERON_MARK   obj=val,slot=MAX, root=MAX, eph=(u64)eph_index, aux=0
//   PHASE_MARK       obj=0,  slot=MAX, root=MAX, eph=MAX, aux=0
//   PHASE_FINALIZE_SCAN obj=0, slot=MAX, root=MAX, eph=MAX, aux=0
//   FINALIZER_ENQUEUE   obj=id, slot=MAX, root=MAX, eph=MAX, aux=0
//   FINALIZER_QUEUE_FULL obj=id, slot=MAX, root=MAX, eph=MAX, aux=0
//   PHASE_RESURRECT_MARK obj=0, slot=MAX, root=MAX, eph=MAX, aux=0
//   PHASE_WEAK_CLEAR     obj=0, slot=MAX, root=MAX, eph=MAX, aux=0
//   WEAK_CLEAR_FIELD obj=holder-id, slot=weak-slot, root=MAX, eph=MAX, aux=0
//   EPHEMERON_CLEAR_SIDE obj=0, slot=side(0=key,1=value), root=MAX,
//                    eph=(u64)eph_index, aux=side(0=key,1=value)
//   PHASE_SWEEP      obj=0,  slot=MAX, root=MAX, eph=MAX, aux=0
//   SWEEP_FREE       obj=id, slot=MAX, root=MAX, eph=MAX, aux=0
//   EPHEMERON_OWNER_FREE obj=owner-id, slot=MAX, root=MAX, eph=(u64)eph_index, aux=0
//   SWEEP_KEEP       obj=id, slot=MAX, root=MAX, eph=MAX, aux=0
//   PROMOTE_OLD      obj=id, slot=MAX, root=MAX, eph=MAX, aux=0
//   GC_END           obj=0,  slot=MAX, root=MAX, eph=MAX, aux=gc_cycle_id
//   FINALIZER_RUN    obj=id, slot=MAX, root=MAX, eph=MAX, aux=0
//   FINALIZER_RESURRECT obj=id, slot=MAX, root=fin_root_slot, eph=MAX, aux=0
//   FINALIZER_SKIP_ABSENT obj=id, slot=MAX, root=MAX, eph=MAX, aux=0
//   INVALID          obj=0,  slot=MAX, root=MAX, eph=MAX, aux=0
//
// --------------------------------------------------------------------------
// 2. COUNTS VECTOR — exact meaning & index of all 31 entries (NORMATIVE)
// --------------------------------------------------------------------------
// NOTE: count indices are NOT the IGCW_EV_* enum values. Use THIS mapping.
//   counts[0]  = alloc_ok            (every successful ALLOC, incl. alloc-black case)
//   counts[1]  = alloc_black         (ALLOC during active GC: object born BLACK & in-cset)
//   counts[2]  = alloc_oom           (ALLOC failed: obj_id_next>max_objects and free list empty)
//   counts[3]  = root_sets           (SET_ROOT applied)
//   counts[4]  = root_clears         (CLEAR_ROOT applied)
//   counts[5]  = strong_sets         (SET_STRONG applied)
//   counts[6]  = strong_clears       (CLEAR_STRONG applied)
//   counts[7]  = weak_sets           (SET_WEAK applied)
//   counts[8]  = ephemeron_sets      (SET_EPHEMERON applied: create or update)
//   counts[9]  = ephemeron_deletes   (DELETE_EPHEMERON applied)
//   counts[10] = ephemeron_oom       (SET_EPHEMERON create failed: table full)
//   counts[11] = remembered_added    (new unique (src,slot) added to remembered set)
//   counts[12] = remembered_dropped  (remembered pair dropped: START_MINOR scan OR GC_END cleanup)
//   counts[13] = gc_start_minor      (START_MINOR began a cycle)
//   counts[14] = gc_start_full       (START_FULL began a cycle)
//   counts[15] = mark_black          (object popped from mark queue & colored BLACK)
//   counts[16] = mark_grey           (strong-child greyed during MARK)
//   counts[17] = ephemeron_marked    (ephemeron value greyed in EPHEMERON phase)
//   counts[18] = finalizer_enqueued  (white finalizable obj enqueued + resurrected grey)
//   counts[19] = finalizer_queue_full(finalizable obj NOT enqueued: fin queue full)
//   counts[20] = weak_fields_cleared (weak slot pointing at white-cset obj cleared)
//   counts[21] = ephemeron_sides_cleared (eph key OR value side cleared in WEAK_CLEAR; one per side)
//   counts[22] = sweep_freed         (white-cset object reclaimed during SWEEP)
//   counts[23] = sweep_kept          (black-cset object survived SWEEP; recolored white)
//   counts[24] = promoted_old        (surviving young obj reached age threshold -> OLD, MINOR only)
//   counts[25] = ephemeron_owner_freed (ephemeron removed because its owner was swept-free)
//   counts[26] = gc_end              (cycle finished: phase->IDLE)
//   counts[27] = finalizer_run       (RUN_FINALIZERS: obj finalized, no resurrection)
//   counts[28] = finalizer_resurrect (RUN_FINALIZERS: obj re-rooted into fin_root_slot)
//   counts[29] = finalizer_skip_absent (RUN_FINALIZERS popped an id no longer present)
//   counts[30] = barrier_marks       (write-barrier greying: SET_ROOT/SET_STRONG/SET_EPHEMERON)
// invalid_count is NOT in this vector; invalidity is reported via invalid_flag.
//
// --------------------------------------------------------------------------
// 3. PER-OP SEMANTICS (validity, state mutation, events, counts)
// --------------------------------------------------------------------------
// Shared predicates:
//   valid_obj(id)         : 1<=id<=max_objects && present[id-1].
//   in_cset(id)           : valid_obj(id) && in_cset[id-1].
//   gc_active()           : phase != IGCW_PHASE_IDLE.
//   strong slot pointer / weak slot pointer are row-major [obj-1][slot].
//   "dst/value/key/obj == 0" means the null reference (not an obj id).
//
// OP ALLOC (a0=generation, a1=finalizer_tag, a2=finalizer_root_slot, size_arg=size):
//   * INVALID if gen not in {YOUNG=0, OLD=1}.
//   * If tag==0: force slot=IGCW_U32_MAX (no finalizer root). Else if slot
//     != IGCW_U32_MAX and slot>=root_count: INVALID.
//   * Choose id: if free list nonempty, take its LOWEST id (free list is kept
//     ascending) and remove it. Else if obj_id_next<=max_objects, id=obj_id_next,
//     obj_id_next+=1. Else OOM: emit ALLOC_OOM, counts[2]++, return (NOT invalid).
//   * Initialize record k=id-1: present=1; size; gen; age=0;
//     alloc_seq=alloc_seq_next then alloc_seq_next+=1; fin_tag=tag;
//     fin_root_slot=slot; finalized=0; in_fin_queue=0; all strong & weak slots=0.
//   * If !gc_active: color=WHITE; alloc_during_gc=0; in_cset=0; emit ALLOC_OK; counts[0]++.
//   * Else (gc_active): in_cset=1; color=BLACK; alloc_during_gc=1;
//     emit ALLOC_BLACK (counts[1]++) THEN emit ALLOC_OK (counts[0]++) — both, in that order.
//
// OP SET_ROOT (a0=root_id, a1=obj):
//   * INVALID if root_id<0 || root_id>=root_count.
//   * INVALID if obj!=0 && !valid_obj(obj).
//   * root[root_id]=obj.
//   * Write barrier: if gc_active && obj!=0 && in_cset(obj) && color[obj]==WHITE:
//       color[obj]=GREY; push obj to mark queue; emit BARRIER_MARK(obj, slot=MAX,
//       root=root_id, eph=MAX); counts[30]++.
//   * emit ROOT_SET(obj, root=root_id); counts[3]++.
//
// OP CLEAR_ROOT (a0=root_id):
//   * INVALID if root_id out of range. Else root[root_id]=0;
//     emit ROOT_CLEAR(obj=0, root=root_id); counts[4]++. (No barrier.)
//
// OP SET_STRONG (a0=src, a1=slot, a2=dst):
//   * INVALID if !valid_obj(src); if slot<0||slot>=strong_slots; if dst!=0 && !valid_obj(dst).
//   * strong[src][slot]=dst.
//   * Remembered set: if dst!=0 && gen[src]==OLD && gen[dst]==YOUNG and (src,slot)
//     not already present: insert (kept sorted ascending by (src,slot)); emit
//     REMEMBERED_ADD(src,slot); counts[11]++.
//   * Write barrier: if gc_active && color[src]==BLACK && dst!=0 && in_cset(dst)
//     && color[dst]==WHITE: color[dst]=GREY; push dst; emit BARRIER_MARK(dst,
//     slot=slot, root=MAX, eph=MAX); counts[30]++.
//   * emit STRONG_SET(src,slot,aux=(u32)dst); counts[5]++.
//
// OP CLEAR_STRONG (a0=src, a1=slot):
//   * INVALID if !valid_obj(src) or slot out of range.
//   * strong[src][slot]=0; emit STRONG_CLEAR(src,slot); counts[6]++.
//     (No remembered-set or barrier action.)
//
// OP SET_WEAK (a0=src, a1=slot, a2=dst):
//   * INVALID if !valid_obj(src); slot<0||slot>=weak_slots; dst!=0 && !valid_obj(dst).
//   * weak[src][slot]=dst; emit WEAK_SET(src,slot,aux=(u32)dst); counts[7]++.
//     (Weak writes NEVER trigger a barrier.)
//
// OP SET_EPHEMERON (a0=ephemeron_id, a1=owner, a2=key, a3=value):
//   * eid=(u32)a0 (dense slot index).
//   * INVALID if !valid_obj(owner); if key!=0 && !valid_obj(key); if value!=0 &&
//     !valid_obj(value); if eid<0||eid>=max_ephemerons.
//   * If eph already present at eid and its owner != owner: INVALID.
//   * If NOT present: if used-count>=max_ephemerons -> emit EPHEMERON_OOM(eph=eid);
//     counts[10]++; return (NOT invalid). Else mark present; create_seq=alloc_seq_next
//     then alloc_seq_next+=1.
//   * Set owner,key,value at eid.
//   * Write barrier (ephemeron-conditional): if gc_active && key!=0 &&
//     color[key]==BLACK && value!=0 && in_cset(value) && color[value]==WHITE:
//     color[value]=GREY; push value; emit BARRIER_MARK(value, slot=MAX, root=MAX,
//     eph=eid); counts[30]++.
//   * emit EPHEMERON_SET(owner, eph=eid, aux=(u32)value); counts[8]++.
//
// OP DELETE_EPHEMERON (a0=ephemeron_id):
//   * eid=(u32)a0. INVALID if eid out of range OR not present.
//   * Clear present/owner/key/value/create_seq to 0; emit EPHEMERON_DELETE(eph=eid);
//     counts[9]++.
//
// OP START_MINOR (no args):
//   * INVALID if phase != IDLE.
//   * gc_cycle_id+=1; mode=MINOR; phase=MARK.
//   * Collection set = all present YOUNG objects: for each, in_cset=1, color=WHITE,
//     alloc_during_gc=0. All other present (OLD) objects: in_cset=0 (treated as
//     reachable/black for the cycle; their color is left UNCHANGED).
//   * Clear mark queue; scan_cursor_obj=scan_cursor_field=sweep_cursor_obj=0;
//     ephemeron_changed=0.
//   * Grey young root targets: for rid=0..root_count-1, obj=root[rid]; if obj!=0 &&
//     in_cset(obj) && color==WHITE: color=GREY, push obj. (NO event for these.)
//   * Remembered scan in canonical (sorted) order: for each (src,slot):
//       - if !valid_obj(src): drop.
//       - else tgt=strong[src][slot]; if tgt!=0 && in_cset(tgt): if color[tgt]==WHITE
//         set GREY & push (NO event); KEEP pair. else (tgt 0 or not in cset): drop.
//       - dropping emits REMEMBERED_DROP(src,slot); counts[12]++ and removes the pair.
//   * emit GC_START_MINOR(aux=gc_cycle_id); counts[13]++.
//
// OP START_FULL (no args):
//   * INVALID if phase != IDLE.
//   * gc_cycle_id+=1; mode=FULL; phase=MARK.
//   * Collection set = ALL present objects: in_cset=1, color=WHITE, alloc_during_gc=0.
//     Absent slots: in_cset=0.
//   * Clear mark queue; cursors=0; ephemeron_changed=0.
//   * Grey root targets (same loop as MINOR; NO events). NO remembered-set scan.
//   * emit GC_START_FULL(aux=gc_cycle_id); counts[14]++.
//
// OP GC_STEP (a0=mark_budget, a1=sweep_budget):
//   * INVALID if phase==IDLE. INVALID if mark_budget==0 && sweep_budget==0.
//   * Dispatch on current phase (advances ONE phase transition; but if WEAK_CLEAR
//     transitions to SWEEP in this call, SWEEP is ALSO run once with sweep_budget):
//       MARK or RESURRECT_MARK -> step_mark(mark_budget)
//       EPHEMERON              -> step_ephemeron()
//       FINALIZE_SCAN          -> step_finalize_scan()
//       WEAK_CLEAR             -> step_weak_clear(); then if phase==SWEEP, step_sweep(sweep_budget)
//       SWEEP                  -> step_sweep(sweep_budget)
//
//   PHASE MACHINE (tri-color: WHITE=0, GREY=1, BLACK=2):
//   step_mark(budget): pop up to `budget` ids FIFO from mark queue (head order):
//       skip (count toward budget) if !valid_obj(id)||!in_cset(id). Else color=BLACK,
//       emit MARK_BLACK(id) counts[15]++; for each strong slot s ascending: tgt=
//       strong[id][s]; if tgt!=0 && in_cset(tgt) && color==WHITE: color=GREY, push,
//       emit MARK_GREY(tgt, slot=s) counts[16]++. After loop, if mark queue EMPTY:
//       phase=EPHEMERON, emit PHASE_EPHEMERON. (Mark queue capacity is the
//       controller bound max_mark_queue; no overflow event is defined.)
//   step_ephemeron(): ephemeron_changed=0. For eph index i ascending (present only):
//       key=eph_key[i], val=eph_value[i]. key_reachable = key!=0 && valid_obj(key) &&
//       ( in_cset(key) ? color[key]==BLACK : true ). If key_reachable && val!=0 &&
//       in_cset(val) && color[val]==WHITE: color[val]=GREY, push val, ephemeron_changed=1,
//       emit EPHEMERON_MARK(val, eph=i) counts[17]++. After loop: if ephemeron_changed:
//       phase=MARK, emit PHASE_MARK. else phase=FINALIZE_SCAN, scan_cursor_obj=1,
//       emit PHASE_FINALIZE_SCAN.
//   step_finalize_scan(): for id=1..max_objects ascending, only in_cset(id):
//       if color==WHITE && fin_tag!=0 && finalized==0 && in_fin_queue==0:
//         if fin_queue size>=max_finalizer_queue: emit FINALIZER_QUEUE_FULL(id)
//           counts[19]++ (leave white, NOT finalized). else: finalized=1; in_fin_queue=1;
//           push id to fin_queue (FIFO tail); color=GREY; push to mark queue;
//           emit FINALIZER_ENQUEUE(id) counts[18]++.
//       After scan: scan_cursor_obj=0. If mark queue nonempty: phase=RESURRECT_MARK,
//       emit PHASE_RESURRECT_MARK. else phase=WEAK_CLEAR, emit PHASE_WEAK_CLEAR.
//   step_weak_clear(): For holder id=1..max_objects ascending (present only), for
//       weak slot s ascending: tgt=weak[id][s]; if tgt!=0 && in_cset(tgt) &&
//       color[tgt]==WHITE: weak[id][s]=0; emit WEAK_CLEAR_FIELD(id, slot=s) counts[20]++.
//       Then for eph index i ascending (present only): if key!=0 && in_cset(key) &&
//       color[key]==WHITE: eph_key[i]=0; emit EPHEMERON_CLEAR_SIDE(slot=0,eph=i,aux=0)
//       counts[21]++. Then (independently) if val!=0 && in_cset(val) && color[val]==WHITE:
//       eph_value[i]=0; emit EPHEMERON_CLEAR_SIDE(slot=1,eph=i,aux=1) counts[21]++.
//       Then phase=SWEEP, sweep_cursor_obj=1, emit PHASE_SWEEP. (Key side hashed
//       before value side for the same ephemeron.)
//   step_sweep(budget): cur=max(sweep_cursor_obj,1). For cur..max_objects while
//       processed<budget: skip ids not (present && in_cset) WITHOUT consuming budget.
//       For each in-cset present id (processed++):
//         WHITE -> free it: for eph index e ascending, if present && owner==id:
//             clear that ephemeron (present/owner/key/value/create_seq=0), emit
//             EPHEMERON_OWNER_FREE(obj=id, eph=e) counts[25]++. Remove ALL remembered
//             pairs with src==id. Fully clear the object record (present=0, in_cset=0,
//             size=0, gen=YOUNG, age=0, color=WHITE, alloc_during_gc=0, alloc_seq=0,
//             fin_tag=0, fin_root_slot=IGCW_U32_MAX, finalized=0, in_fin_queue=0, all
//             strong/weak slots=0). Insert id into free list (kept ascending). Emit
//             SWEEP_FREE(id) counts[22]++.
//         BLACK -> keep: if mode==MINOR && gen==YOUNG: age+=1 (u8 wrap); if age>=
//             young_survive_threshold: gen=OLD, emit PROMOTE_OLD(id) counts[24]++.
//             Then color=WHITE, alloc_during_gc=0, emit SWEEP_KEEP(id) counts[23]++.
//       Set sweep_cursor_obj=cur (the value at loop exit). done = no remaining id in
//       [cur..max_objects] is present && in_cset. If done:
//         Remembered cleanup pass (canonical order): for each (src,slot): drop if
//           !valid_obj(src) OR NOT(gen[src]==OLD && tgt=strong[src][slot] !=0 &&
//           valid_obj(tgt) && gen[tgt]==YOUNG); dropping emits REMEMBERED_DROP(src,slot)
//           counts[12]++. Then phase=IDLE; mode=NONE; clear all in_cset; clear mark
//           queue; sweep_cursor_obj=0; emit GC_END(aux=gc_cycle_id) counts[26]++.
//
// OP RUN_FINALIZERS (a0=limit):
//   * If limit==0: valid no-op (no events, no counts, not invalid).
//   * INVALID if limit<0.
//   * Pop up to `limit` ids FIFO from fin_queue head; for each popped id (popped++):
//       if !valid_obj(id): emit FINALIZER_SKIP_ABSENT(id) counts[29]++; continue.
//       else: in_fin_queue[id]=0; slot=fin_root_slot[id]; if slot!=IGCW_U32_MAX &&
//       slot<root_count: root[slot]=id; emit FINALIZER_RESURRECT(id, root=slot)
//       counts[28]++. else emit FINALIZER_RUN(id) counts[27]++.
//
// --------------------------------------------------------------------------
// 4. invalid_flag SEMANTICS
// --------------------------------------------------------------------------
// invalid_flag (IgcwOutputs.invalid_flag) = op_invalid, which is 1 iff this op
// called mark_invalid() (which also emits exactly one IGCW_EV_INVALID event and
// bumps the persistent event_seq), else 0. OOM cases (ALLOC_OOM, EPHEMERON_OOM)
// and the RUN_FINALIZERS limit==0 no-op are NOT invalid. mark_invalid() is the
// ONLY path that sets it; every op returns immediately after mark_invalid().
//
// --------------------------------------------------------------------------
// 5. STATE-SNAPSHOT CHECKSUMS (all FNV-1a-64, each seeded fresh per readout
//    with IGCW_FNV_INIT=1469598103934665603). Computed AFTER the op body.
// --------------------------------------------------------------------------
// heap_hash: iterate obj index i=0..max_objects-1 ASCENDING; skip if !present[i];
//   for each present object (id=i+1) fold IN ORDER:
//     u64 id; u64 size; u8 gen; u8 age; u8 color; u8 alloc_during_gc;
//     u64 alloc_seq; u32 fin_tag; u32 fin_root_slot (IGCW_U32_MAX if none);
//     u8 finalized; u8 in_fin_queue;
//     for s=0..strong_slots-1: u64 (uint32_t)strong[id][s] (0 if none);
//     for s=0..weak_slots-1:   u64 (uint32_t)weak[id][s]   (0 if none).
//   (Note strong/weak slot values are zero-extended u32->u64.)
// root_hash: for rid=0..root_count-1 ASCENDING fold: u32 rid; u64 (uint32_t)root[rid].
// ephemeron_hash: for eph index i=0..max_ephemerons-1 ASCENDING; skip if !present[i];
//   fold: u64 i; u64 (uint32_t)owner; u64 (uint32_t)key; u64 (uint32_t)value;
//   u64 create_seq.
// remembered_hash: iterate the remembered list in its canonical stored order
//   (ascending by (src,slot)); for each pair fold: u64 (uint32_t)src; u32 slot.
// gc_controller_hash: fold IN ORDER:
//     u8 phase; u8 mode; u64 gc_cycle_id; u64 scan_cursor_obj; u32 scan_cursor_field;
//     u64 sweep_cursor_obj; u8 ephemeron_changed;
//     then mark queue contents in FIFO order (head..tail): each u64 (uint32_t)id;
//     then fin_queue contents in FIFO order (head..tail): each u64 (uint32_t)id;
//     then free_ids in ascending order: each u64 (uint32_t)id.
//
// --------------------------------------------------------------------------
// 6. CANONICAL ORDERS & TIE-BREAKS (summary)
// --------------------------------------------------------------------------
//   * Object iteration (heap/sweep/finalize/weak-clear): obj id ASCENDING (1..max).
//   * Ephemeron iteration: dense slot index ASCENDING (0..max_ephemerons-1).
//   * Remembered set: stored & hashed ascending by (src,slot); src primary, slot
//     secondary; uniqueness on the pair.
//   * Free-id list: ascending; ALLOC always takes the LOWEST free id first.
//   * Finalizer queue: strict FIFO (enqueue tail at FINALIZER_ENQUEUE; dequeue head
//     at RUN_FINALIZERS).
//   * Mark queue: strict FIFO (push tail, pop head).
//   * Within MARK, strong child scan is slot ASCENDING; within WEAK_CLEAR, weak slot
//     scan is slot ASCENDING; within an ephemeron, key side precedes value side.
//   * event_seq and gc_event_hash are persistent/monotonic across the whole op
//     stream; the five state-snapshot hashes are recomputed from scratch each op.
//
// === END CONTRACT ===

#endif  // INCREMENTAL_GC_WEAKREF_COMMON_H_
