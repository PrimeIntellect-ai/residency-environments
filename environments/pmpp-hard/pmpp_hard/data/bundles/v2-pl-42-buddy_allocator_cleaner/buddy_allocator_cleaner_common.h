// file: buddy_allocator_cleaner_common.h

#ifndef BUDDY_ALLOCATOR_CLEANER_COMMON_H_
#define BUDDY_ALLOCATOR_CLEANER_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define BAC_ABI_VERSION 1

// ---------------------------------------------------------------------------
// Static capacity bounds.
// ---------------------------------------------------------------------------
#define BAC_MIN_MAX_ORDER 1
#define BAC_MAX_MAX_ORDER 20          // total_pages up to 2^20
#define BAC_MIN_SEGMENT_ORDER 0
#define BAC_MAX_NUM_CLASSES 16
#define BAC_MIN_NUM_CLASSES 1
#define BAC_MAX_OBJECTS 4096
#define BAC_MIN_OBJECTS 1
#define BAC_MAX_SEGMENTS 4096
#define BAC_MIN_SEGMENTS 1
#define BAC_MAX_OPS_PER_STEP 4096
#define BAC_MAX_STEPS 64

// Op codes for the operation stream.
#define BAC_OP_ALLOC 0
#define BAC_OP_FREE  1
#define BAC_OP_PIN   2
#define BAC_OP_UNPIN 3
#define BAC_OP_SEAL  4
#define BAC_OP_CLEAN 5

// ---------------------------------------------------------------------------
// CONTRACT: buddy_allocator_cleaner  (T42)
//
// Pin-Aware Log-Segmented Buddy Allocator Cleaner.
//
// A persistent page allocator combining buddy split/merge free lists,
// append-only segment allocation, pin-delayed frees, partial cleaner
// relocation, and ordered reclaim streams.
//
// PERSISTENT STATE (after init/reset)
//   total_pages = 2^max_order; max_order; segment_order; num_classes;
//   max_objects; max_segments.
//   segment_pages = 2^segment_order.
//   Buddy free lists for orders 0..max_order. Initially one free block:
//     (base_page = 0, order = max_order). Each free list sorted ascending base.
//   next_segment_id = 1; event_seq = 0.
//   For each class c: active_segment[c] = none.
//   Segment table and object table empty.
//
// OP STREAM
//   Each op consumes one event_seq:
//     event_seq is incremented BY ONE at the start of every op (valid OR
//     invalid). The pre-increment value is irrelevant; the op's seq is the
//     post-increment value. (event_seq wraps mod 2^64.)
//   op_index is the GLOBAL 0-based index of the op across the whole replay
//     history (it equals event_seq - 1 modulo the wrap, but is tracked as a
//     plain u32 counter that also wraps mod 2^32).
//
// Per op semantics: EXACT integer model, no float. Fully specified in the
// normative DETERMINISM section below (do not consult any other file).
//
// Each op encoded as a BacOp:
//   ALLOC: obj_id, class_id, requested_pages (a)
//   FREE:  obj_id
//   PIN:   obj_id
//   UNPIN: obj_id
//   SEAL:  class_id
//   CLEAN: a = max_segments_to_consider, b = copy_page_budget
//
// OUTPUTS (per step): cumulative counters over the whole replay so far, plus
// rolling event hashes for the whole replay, plus structural state hashes.
//   See BacOutputs. All exact.
//
// Rules:
//   - solution_init may allocate persistent state.
//   - solution_run may not call cudaMalloc/cudaFree.
//   - solution_run may launch kernels and use provided workspace.
//   - All outputs exact.
// ---------------------------------------------------------------------------
//
// === DETERMINISM & EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section is SELF-CONTAINED. Every grader-enforced counter and hash is
// fully specified here; nothing is deferred to any other file.
//
// ---- FNV-1a-64 primitive ----
//   basis = 1469598103934665603 (0x14650FB0739D0383)
//   prime = 1099511628211       (0x00000100000001B3)
//   fold one byte b:  h = (h XOR b) * prime  (mod 2^64).
//   A field is folded byte-by-byte in LITTLE-ENDIAN order (raw in-memory bytes
//   index 0..size-1). Widths used below: u8 = 1 byte; u32 = 4 bytes; u64 = 8
//   bytes. All hashes start from h = basis.
//
// ---- derived constants ----
//   total_pages   = 2^max_order
//   segment_pages = 2^segment_order
//   ceil_log2(n)  : smallest o with 2^o >= n  (ceil_log2(1)=0; ceil_log2(0)=0).
//
// ---- buddy free list ----
//   A multiset of free blocks {(order, base)}. Initially exactly one block
//   (order=max_order, base=0). next_segment_id=1, event_seq=0, op_index=0,
//   active_segment[c]=0 (none) for all classes; segment/object tables empty.
//
//   buddy_alloc(want_order) -> base (or fail):
//     Among free blocks with order >= want_order choose the SMALLEST order;
//     tie -> smallest base. If none, FAIL. Remove the chosen block. While its
//     order k > want_order: child=k-1; right_base = base + 2^child; INSERT
//     (child, right) into the free list; counter buddy_splits += 1; k=child
//     (the kept left half keeps `base`). Return base.
//
//   buddy_free(base, order):
//     o=order, b=base. While o < max_order: buddy = b XOR 2^o; if a free block
//     (o, buddy) exists, remove it, b=min(b,buddy), o+=1, counter
//     buddy_merges += 1; else stop. Then INSERT (o, b).
//
// ---- segment append allocator: append_allocate(class cls, order, block) ----
//   block = 2^order. Let active_id = active_segment[cls].
//   1. If active_id != 0: active segment's append offset is aoff; compute
//      aligned_offset = (aoff + block - 1) & ~(block - 1)  (round up to block).
//      If aligned_offset + block > segment_pages: need a new segment.
//      If active_id == 0: need a new segment (aligned_offset treated as 0).
//   2. If a new segment is needed:
//      - If there was an active segment, SEAL it implicitly (see seal below)
//        and set active_segment[cls]=0.
//      - If live segment count >= max_segments: FAIL (caller -> OOM).
//      - buddy_alloc(segment_order); if it fails: FAIL. new_base = result.
//      - Create segment: id = next_segment_id (then next_segment_id += 1),
//        class=cls, base=new_base, aoff=0, live=0, dead=0, sealed=0; set
//        active_segment[cls]=id. aligned_offset=0.
//   3. padding = aligned_offset - aoff(before). base_page = seg_base +
//      aligned_offset. Set seg_aoff = aligned_offset + block; seg_live += block;
//      seg_dead += padding; counter padding_pages_added += padding.
//      Return (segment_id, base_page).
//
// ---- seal_segment(seg, implicit?) ----
//   tail = segment_pages - seg_aoff; seg_dead += tail; seg_aoff = segment_pages;
//   seg_sealed = 1. If implicit: counter seal_implicit += 1 and emit FINALIZE
//   event kind SEAL_IMPLICIT(3). Else: counter seal_explicit += 1 and emit
//   FINALIZE kind SEAL_EXPLICIT(4). (Event field values: obj_id=U64MAX,
//   seg=seg_id, base=seg_base, pages=tail, reason=NONE(0).)
//
// ---- op semantics (each op: event_seq += 1 first; seq = post-increment;
//      opidx = current op_index, then op_index += 1; both wrap) ----
//
//   ALLOC (obj_id, class_id=cls, a=requested_pages=req):
//     INVALID (counter invalid_count += 1, NO event) iff cls>=num_classes OR
//       req==0 OR req>segment_pages OR an object with obj_id already alive.
//     Else if live object count >= max_objects: counter alloc_oom += 1; emit
//       ALLOC event kind ALLOC_OOM(1) with seg=U64MAX, base=U64MAX,
//       order=ORDER_NONE(255), req=req, block=0.
//     Else order=ceil_log2(req); block=2^order; try append_allocate(cls,order,
//       block). On FAIL: counter alloc_oom += 1; emit ALLOC_OOM(1) with
//       seg=U64MAX, base=U64MAX, order=255, req=req, block=block.
//     On success: create object {id, class=cls, seg=seg_id, base=base_page,
//       order, block, req, pin=0, pending=0, aseq=seq, mseq=seq}; counter
//       alloc_ok += 1; emit ALLOC_OK(0) with the assigned seg, base, order,
//       req, block.
//
//   FREE (obj_id):
//     If no alive object obj_id: invalid_count += 1 (no event).
//     Else if obj_pin>0: if obj_pending already 1: counter free_deferred += 1;
//       emit FINALIZE FREE_DEFERRED_DUP(1). Else set obj_pending=1; counter
//       free_deferred += 1; emit FINALIZE FREE_DEFERRED(0). (Both with
//       obj_id, seg, base, pages=block, reason=NONE(0).)
//     Else (pin==0): finalize_object(reason=FREE_IMMEDIATE(1)).
//
//   PIN (obj_id):
//     If no alive object: invalid_count += 1. Else if obj_pending==1:
//       invalid_count += 1. Else if obj_pin==U64MAX: invalid_count += 1.
//     Else obj_pin += 1; counter pin_ok += 1.
//
//   UNPIN (obj_id):
//     If no alive object: invalid_count += 1. Else if obj_pin==0:
//       invalid_count += 1. Else obj_pin -= 1; counter unpin_ok += 1; and if
//       now obj_pin==0 AND obj_pending==1: finalize_object(reason=
//       UNPIN_FINALIZE(2)).
//
//   SEAL (class_id=cls):
//     If cls>=num_classes: invalid_count += 1. Else if active_segment[cls]==0:
//       counter seal_empty += 1; emit FINALIZE SEAL_EMPTY(5) with obj_id=U64MAX,
//       seg=U64MAX, base=0, pages=0, reason=NONE. Else seal_segment(active,
//       implicit=false) and active_segment[cls]=0.
//
//   CLEAN (a=max_segments_to_consider, b=copy_page_budget):
//     If a==0 OR b==0: no-op. Else loop while a>0 and b>0:
//       victim = select_victim() (see below); if none, stop.
//       If victim seg_live==0: reclaim_segment(victim); a -= 1; continue.
//       Gather movable objects: alive objects whose obj_seg==victim.id, with
//         obj_pin==0 AND obj_pending==0. Sort ascending by (obj_base, then
//         obj_id) (stable insertion sort).
//       If none movable: counter clean_blocked_segments += 1; emit FINALIZE
//         CLEAN_BLOCKED_SEGMENT(6) with obj_id=U64MAX, seg=victim.id,
//         base=victim.base, pages=victim.live, reason=NONE; a -= 1; continue.
//       Else iterate movable objects in sorted order:
//         block = obj_block. If block > b (remaining budget): STOP the entire
//           CLEAN op (break out of all loops).
//         dst = append_allocate(obj.class, obj.order, block). If it FAILS:
//           STOP the entire CLEAN op.
//         Relocate: victim.live -= block; victim.dead += block; obj_seg=dst.seg;
//           obj_base=dst.base; obj_mseq=seq; counter relocated_objects += 1;
//           emit ALLOC event kind RELOCATE_OBJECT(2) with obj_id, class,
//           seg=dst.seg, base=dst.base, order=obj.order, req=obj.req,
//           block=block. b -= block. If victim.live becomes 0:
//           reclaim_segment(victim); a -= 1; break out of the movable loop
//           (re-select victim on the next while iteration).
//
//   Unknown op_type: invalid_count += 1.
//
//   finalize_object(obj, reason): let seg=obj.seg's segment. seg_live -=
//     obj_block; seg_dead += obj_block; counter free_finalized += 1; emit
//     FINALIZE OBJECT_FINALIZE(2) with obj_id, seg=seg_id, base=obj_base,
//     pages=obj_block, reason; then mark object dead (slot freed).
//
//   reclaim_segment(seg): if active_segment[seg.class]==seg.id set it to 0;
//     mark segment dead; buddy_free(seg.base, segment_order); counter
//     segments_reclaimed += 1; emit FINALIZE SEGMENT_RECLAIM(7) with
//     obj_id=U64MAX, seg=seg.id, base=seg.base, pages=segment_pages,
//     reason=NONE.
//
//   select_victim(): among alive segments with sealed==1 AND dead>0 choose:
//     largest dead; tie -> smallest live; tie -> smallest base; tie ->
//     smallest segment_id. None if no such segment.
//
// ---- event-kind byte constants (folded as the u8 `kind` field) ----
//   ALLOC-stream kinds:    ALLOC_OK=0, ALLOC_OOM=1, RELOCATE_OBJECT=2.
//   FINALIZE-stream kinds: FREE_DEFERRED=0, FREE_DEFERRED_DUP=1,
//     OBJECT_FINALIZE=2, SEAL_IMPLICIT=3, SEAL_EXPLICIT=4, SEAL_EMPTY=5,
//     CLEAN_BLOCKED_SEGMENT=6, SEGMENT_RECLAIM=7.
//   reason byte constants: NONE=0, FREE_IMMEDIATE=1, UNPIN_FINALIZE=2.
//   ORDER_NONE = 255 (the u8 `order` placeholder on OOM). U64MAX =
//   0xFFFFFFFFFFFFFFFF (the u64 placeholder for absent obj_id/seg/base).
//
// ---- rolling event hashes (entire replay, folded in emission order) ----
//   alloc_event_hash (ALLOC stream). Each emitted ALLOC-stream event folds, in
//   order: kind(u8), seq(u64), op_index(u32), obj_id(u64), class(u32),
//   seg(u64), base(u64), order(u8), req(u64), block(u64).
//   finalize_hash (FINALIZE stream). Each emitted FINALIZE-stream event folds,
//   in order: kind(u8), seq(u64), op_index(u32), obj_id(u64), seg(u64),
//   base(u64), pages(u64), reason(u8).
//   Both hashes are seeded h=basis at reset and persist across steps; the
//   output is the running value after this step. ALLOC and FINALIZE are
//   SEPARATE running hashes.
//
// ---- structural state hashes (current state after this step; each seeded
//      h=basis fresh every step) ----
//   buddy_hash: enumerate free blocks in (order ASC, base ASC) order. For each
//     block fold order(u8) then base(u64).
//   segment_hash: enumerate alive segments in (base ASC, then segment_id ASC).
//     For each fold: id(u64), class(u32), base(u64), aoff(u64), live(u64),
//     dead(u64), sealed(u8), is_active(u8) where is_active = 1 iff
//     active_segment[seg.class]==seg.id else 0.
//   object_hash: enumerate alive objects in obj_id ASC order. For each fold:
//     id(u64), class(u32), seg(u64), base(u64), order(u8), req(u64), block(u64),
//     pin(u64), pending(u8), aseq(u64), mseq(u64).
//
// ---- live scalars ----
//   live_object_count = number of alive objects after this step.
//   live_segment_count = number of alive segments after this step.
// ---------------------------------------------------------------------------

struct alignas(8) BacOp {
    int32_t op_type;        // BAC_OP_*
    int32_t class_id;       // ALLOC, SEAL
    uint64_t obj_id;        // ALLOC, FREE, PIN, UNPIN
    uint64_t a;             // ALLOC: requested_pages; CLEAN: max_segments
    uint64_t b;             // CLEAN: copy_page_budget
    uint64_t reserved;
};

struct alignas(8) BacProblemSpec {
    int32_t abi_version;
    int32_t max_order;
    int32_t segment_order;
    int32_t num_classes;
    int32_t max_objects;
    int32_t max_segments;
    int32_t max_ops_per_step;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[7];
};

struct alignas(8) BacRunSpec {
    int32_t abi_version;
    int32_t num_ops;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) BacInputs {
    const BacOp* ops;
};

struct alignas(8) BacOutputs {
    // Cumulative counters over the entire replay so far.
    uint64_t* alloc_ok;
    uint64_t* alloc_oom;
    uint64_t* free_finalized;
    uint64_t* free_deferred;
    uint64_t* pin_ok;
    uint64_t* unpin_ok;
    uint64_t* seal_explicit;
    uint64_t* seal_implicit;
    uint64_t* seal_empty;
    uint64_t* relocated_objects;
    uint64_t* clean_blocked_segments;
    uint64_t* segments_reclaimed;
    uint64_t* buddy_splits;
    uint64_t* buddy_merges;
    uint64_t* padding_pages_added;
    uint64_t* invalid_count;

    // Rolling event hashes (entire replay, in emission order).
    uint64_t* alloc_event_hash;
    uint64_t* finalize_hash;

    // Structural state hashes (current state after this step).
    uint64_t* buddy_hash;
    uint64_t* segment_hash;
    uint64_t* object_hash;

    // Live scalars.
    uint64_t* live_object_count;
    uint64_t* live_segment_count;
};

static_assert(sizeof(BacOp) == 40, "BacOp layout drift");
static_assert(sizeof(BacProblemSpec) == 64, "BacProblemSpec layout drift");
static_assert(sizeof(BacRunSpec) == 64, "BacRunSpec layout drift");
static_assert(sizeof(BacInputs) == 8, "BacInputs layout drift");
static_assert(sizeof(BacOutputs) == 23 * sizeof(void*), "BacOutputs layout drift");

static inline size_t bac_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int bac_validate_problem_spec(const BacProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != BAC_ABI_VERSION) return 0;
    if (spec->max_order < BAC_MIN_MAX_ORDER || spec->max_order > BAC_MAX_MAX_ORDER) return 0;
    if (spec->segment_order < BAC_MIN_SEGMENT_ORDER || spec->segment_order > spec->max_order) return 0;
    if (spec->num_classes < BAC_MIN_NUM_CLASSES || spec->num_classes > BAC_MAX_NUM_CLASSES) return 0;
    if (spec->max_objects < BAC_MIN_OBJECTS || spec->max_objects > BAC_MAX_OBJECTS) return 0;
    if (spec->max_segments < BAC_MIN_SEGMENTS || spec->max_segments > BAC_MAX_SEGMENTS) return 0;
    if (spec->max_ops_per_step < 0 || spec->max_ops_per_step > BAC_MAX_OPS_PER_STEP) return 0;
    if (spec->max_steps < 1 || spec->max_steps > BAC_MAX_STEPS) return 0;
    return 1;
}

static inline int bac_validate_run_spec(const BacRunSpec* run, const BacProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != BAC_ABI_VERSION) return 0;
    if (run->num_ops < 0 || run->num_ops > spec->max_ops_per_step) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const BacProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const BacProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const BacRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // BUDDY_ALLOCATOR_CLEANER_COMMON_H_
