// file: mk_schedule_planner_common.h

#ifndef MK_SCHEDULE_PLANNER_COMMON_H_
#define MK_SCHEDULE_PLANNER_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MK_ABI_VERSION 1

// ---------------------------------------------------------------------------
// Static capacity bounds.
// ---------------------------------------------------------------------------
#define MK_MIN_SM_COUNT 1
#define MK_MAX_SM_COUNT 64
#define MK_MIN_PAGES_PER_SM 1
#define MK_MAX_PAGES_PER_SM 64
#define MK_MIN_WAVE_QUANTUM 1
#define MK_MAX_WAVE_QUANTUM ((uint64_t)1 << 40)
#define MK_MIN_MAX_INSTRS 1
#define MK_MAX_MAX_INSTRS 2048
#define MK_MIN_MAX_EDGES 0
#define MK_MAX_MAX_EDGES 8192
#define MK_MAX_PAGE_KEYS_PER_INSTR 8     // page_count <= pages_per_sm <= this bound

#define MK_MAX_OPS_PER_STEP 4096
#define MK_MAX_STEPS 64

// Op codes for the operation stream.
#define MK_OP_ADD_INSTR    0
#define MK_OP_ADD_EDGE     1
#define MK_OP_PLAN_NEXT    2
#define MK_OP_COMMIT_PLAN  3
#define MK_OP_EXECUTE_UNTIL 4
#define MK_OP_CANCEL_INSTR 5
#define MK_OP_NEW_EPOCH    6

// Instruction status values (stable, hashed).
#define MK_ST_UNPLANNED 0
#define MK_ST_PLANNED   1
#define MK_ST_COMMITTED 2
#define MK_ST_EXECUTED  3
#define MK_ST_CANCELLED 4

// ---------------------------------------------------------------------------
// CONTRACT: mk_schedule_planner  (MK7)
//
// Wave-Quantized Static Megakernel Schedule Planner.
//
// A persistent host-planned/static schedule model that assigns instruction-DAG
// nodes to SM programs under page constraints, wave-quantized start times,
// dependency counters, and makespan tie-breaks.
//
// PERSISTENT STATE (after init/reset):
//   sm_count; pages_per_sm; wave_quantum; max_instrs; max_edges.
//   event_seq = 0; instr_seq_next = 1; edge_seq_next = 1; plan_seq_next = 1;
//   committed_epoch = 0.
//   Instruction table empty; edge table empty; per-SM plan intervals empty;
//   per-SM page live intervals empty; completion counters all zero.
//
// OP STREAM
//   event_seq is incremented BY ONE at the start of every op (valid OR invalid);
//   the op's seq is the post-increment value. op_index is the GLOBAL 0-based
//   index of the op across the whole replay history, a plain u32 counter.
//
// Each op encoded as an MkOp (see below). EXACT integer model, no float.
//
// OUTPUTS (per step): cumulative counters over the whole replay so far, plus
// rolling planner_event_hash, plus structural state hashes (plan, instr, edge,
// page-interval). See MkOutputs. All exact.
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
//   A field is folded byte-by-byte in LITTLE-ENDIAN order (raw in-memory bytes,
//   index 0..size-1). Widths below: u8=1, u32=4, u64=8 bytes. Hashes start at
//   h = basis. Constants: U32MAX=0xFFFFFFFF, U64MAX=0xFFFFFFFFFFFFFFFF.
//
// ---- per-instruction stored fields ----
//   id, seq (assignment order: instr_seq_next starts at 1, ++ per added instr),
//   dur(=a), pcnt(=b page_count), page_keys[0..pcnt), rdelay(=c), seed(=d),
//   status (UNPLANNED/PLANNED/COMMITTED/EXECUTED/CANCELLED = 0..4),
//   crit (critical height, see below), comp (completion count, starts 0).
//
// ---- per-edge stored fields ----
//   id, src, dst, chunk(=(u32)a), tinc(=b target_increment),
//   seq (edge_seq_next starts at 1, ++ per added edge).
//
// ---- per-plan stored fields (one entry per planned/committed instr) ----
//   seq (plan_seq_next starts at 1, ++ per placement), instr(=id), sm, wave,
//   start, finish, release, pcnt, page_keys[0..pcnt).
//
// ---- critical height (recomputed after every add-instr / add-edge / cancel /
//      new-epoch) ----
//   For each alive non-cancelled instr i: crit(i) starts = dur(i) (cancelled:
//   crit=0). Then iterate to fixpoint: crit(i) = dur(i) + max over noncancelled
//   successors j (edges with src==i.id, dst alive non-cancelled) of crit(j),
//   or dur(i) if no such successor. (Acyclic graph; converges.)
//
// ---- op processing (each op: event_seq += 1 first; seq = post-increment;
//      opidx = current op_index then op_index += 1; both plain counters) ----
//   Every emitted planner event folds into planner_event_hash, in emission
//   order, the 9-field record (see "event record" below).
//
//   ADD_INSTR (id,a=dur,b=pcnt,c=rdelay,d=seed,page_keys):
//     INVALID iff instr id already exists OR alive instr count>=max_instrs OR
//       dur==0 OR pcnt==0 OR pcnt>pages_per_sm. On INVALID: counter
//       invalid_count += 1; emit INVALID(11) [instr=id, edge=0].
//     Else create instr (status=UNPLANNED, crit=dur, comp=0, page_keys[k]=
//       page_keys[k] for k<pcnt else 0), recompute crit; counter instr_added
//       += 1; emit INSTR_ADD(0) with aux=isq (assigned seq).
//
//   ADD_EDGE (id,src,dst,a=chunk,b=tinc):
//     INVALID iff edge id exists OR src or dst not found OR src==dst OR tinc==0
//       OR alive edge count>=max_edges OR dst status in {COMMITTED,EXECUTED} OR
//       src or dst CANCELLED OR adding the edge would create a cycle (i.e. src
//       is reachable from dst over noncancelled edges). On INVALID:
//       invalid_count += 1; emit INVALID(11) [instr=0, edge=id].
//     Else create edge (seq=es). If src or dst is currently PLANNED, run
//       invalidate-from(dst) (see below). Recompute crit. counter edge_added
//       += 1; emit EDGE_ADD(1) with aux=es [instr=0, edge=id].
//
//   PLAN_NEXT (a=limit): if limit==0 nothing. Else repeat up to `limit` times:
//     pick the best READY instr (see ready/selection below). If none: counter
//       plan_empty += 1; emit PLAN_EMPTY(3); break. Else attempt placement
//       (ref_place). If placement STALLS (no feasible sm/wave): counter
//       plan_stall += 1; emit PLAN_STALL(4) [instr=id, sm=U32MAX,
//       start=U64MAX, finish=U64MAX, aux=0]; break. On success: counter
//       plan_placed += 1; emit PLAN_PLACE(5) [instr=id, sm=bsm, start=bstart,
//       finish=bfinish, aux=plan_seq]; instr status->PLANNED.
//
//   COMMIT_PLAN (a=max_entries): if 0 nothing. Else commit up to max_entries
//     PLANNED instrs in ascending plan_seq order: status->COMMITTED; counter
//     plan_committed += 1; emit PLAN_COMMIT(6) [instr=plan.instr, sm=plan.sm,
//     start=plan.start, finish=plan.finish, aux=plan.seq].
//
//   EXECUTE_UNTIL (a=tick_limit, b=max_events): if max_events==0 nothing. Else
//     up to max_events times, among COMMITTED plans with finish<=tick_limit
//     pick min finish; tie min sm; tie min plan_seq. If none, stop. Compute a
//     result hash (see below); instr.comp += 1; status->EXECUTED; counter
//     instr_executed += 1; emit EXEC_INSTR(7) [instr=id, sm, start, finish,
//     aux=result_hash]. Then signal outgoing edges (src==id) in ascending
//     edge_seq order: for each, counter edge_signaled += 1; emit EDGE_SIGNAL(8)
//     [instr=edge.dst, edge=edge.id, sm=U32MAX, start=U64MAX, finish=U64MAX,
//     aux = (u64)chunk XOR (tinc * FNV_PRIME)] where FNV_PRIME=1099511628211.
//     result hash: h=basis; fold committed_epoch(u64), instr.id(u64),
//       plan.start(u64), plan.finish(u64), plan.sm(u32), instr.seed(u64), then
//       instr.page_keys[0..pcnt) each as u64.
//
//   CANCEL_INSTR (id):
//     INVALID iff id not found OR status in {EXECUTED,COMMITTED,CANCELLED}:
//       invalid_count += 1; emit INVALID(11) [instr=id, edge=0].
//     Else: if status==PLANNED run invalidate-from(id). Remove every incident
//       edge whose OTHER endpoint is NOT COMMITTED/EXECUTED (an absent endpoint
//       counts as UNPLANNED, so the edge is removed). Set status=CANCELLED,
//       crit=0; recompute crit; counter instr_cancelled += 1; emit
//       INSTR_CANCEL(9) [instr=id, edge=0].
//
//   NEW_EPOCH (no fields):
//     INVALID iff any alive instr has status==COMMITTED: invalid_count += 1;
//       emit INVALID(11) [instr=0, edge=0].
//     Else committed_epoch += 1; every alive non-cancelled instr ->
//       status=UNPLANNED, comp=0; clear the whole plan table; recompute crit;
//       counter epoch_advanced += 1; emit EPOCH_ADVANCE(10) [instr=0, edge=0,
//       aux=committed_epoch].
//
//   Unknown op_type: invalid_count += 1; emit INVALID(11) [instr=0, edge=0].
//
// ---- ready test & selection (PLAN_NEXT) ----
//   An instr is READY iff status==UNPLANNED and every noncancelled predecessor
//   (edge dst==this, src alive non-cancelled) has status PLANNED or COMMITTED.
//   Among ready instrs pick: crit DESC; tie -> instr.seq ASC; tie -> id ASC.
//
// ---- placement (ref_place) ----
//   dep_ready(id) = max over noncancelled predecessors that have a plan entry
//     of that plan's finish (0 if none). For each sm = 0..sm_count-1:
//     if dep_ready==0: wave=0, start=0; else wave=ceil(dep_ready/wave_quantum),
//     start=wave*wave_quantum. Advance wave (start=wave*wave_quantum) until the
//     page-feasibility test passes (bounded scan). finish=start+dur;
//     cend=finish+rdelay. Candidate ordering across (sm,wave): pick min finish;
//     tie min start; tie min sm; tie min added-pages; tie min key-sum. If no sm
//     yields a feasible slot -> STALL. On success store plan {seq=plan_seq,
//     instr=id, sm, wave, start, finish, release=finish+rdelay, pcnt,
//     page_keys}.
//   page feasibility on sm over [start, cend) with request keys rk[0..pcnt):
//     ksum = sum of rk. If cend<=start: feasible iff pcnt<=pages_per_sm; added
//       = count of distinct keys in rk. Else: at each boundary tick (start, and
//       every existing plan start/release on this sm strictly inside
//       (start,cend) that overlaps the window) count distinct live page keys =
//       request keys plus keys of plans on this sm whose [p.start,p.release)
//       covers the tick; if that distinct count > pages_per_sm at any boundary,
//       INFEASIBLE. added = number of distinct request keys NOT already
//       resident on this sm at tick=start (resident = some plan on sm with
//       p.start<=start<p.release holds the key).
//
// ---- invalidate-from(dst) ----
//   Mark dst and all PLANNED descendants reachable through PLANNED nodes over
//   noncancelled edges (traversal stops at non-PLANNED nodes). Emit
//   PLAN_INVALIDATE(2) for each marked plan in DESCENDING plan_seq order:
//   counter plan_invalidated += 1; emit [instr=plan.instr, sm=plan.sm,
//   start=plan.start, finish=plan.finish, aux=plan.seq], removing the plan
//   entry. Then set every marked instr back to status=UNPLANNED.
//
// ---- planner event record (folded into planner_event_hash, in order) ----
//   kind(u8), seq(u64), op_index(u32), instr_or0(u64), edge_or0(u64),
//   sm_or_max(u32), start_or_max(u64), finish_or_max(u64), aux(u64).
//   Event kind bytes: INSTR_ADD=0, EDGE_ADD=1, PLAN_INVALIDATE=2, PLAN_EMPTY=3,
//   PLAN_STALL=4, PLAN_PLACE=5, PLAN_COMMIT=6, EXEC_INSTR=7, EDGE_SIGNAL=8,
//   INSTR_CANCEL=9, EPOCH_ADVANCE=10, INVALID=11. planner_event_hash is seeded
//   h=basis at reset and persists across the whole replay (output = running
//   value after this step).
//
// ---- cumulative counters (MkOutputs order) ----
//   instr_added, edge_added, plan_invalidated, plan_empty, plan_stall,
//   plan_placed, plan_committed, instr_executed, edge_signaled,
//   instr_cancelled, epoch_advanced, invalid_count. Each accumulates over the
//   whole replay as described per-op above.
//
// ---- structural state hashes (current state after this step; each seeded
//      h=basis fresh per step) ----
//   plan_hash: enumerate plan entries in (sm ASC, start ASC, plan_seq ASC).
//     For each fold: sm(u32), plan_seq(u64), instr_id(u64), wave(u64),
//     start(u64), finish(u64), release(u64), then page_keys[0..pcnt) each u64.
//   instr_hash: enumerate alive instrs in id ASC. For each fold: id(u64),
//     seq(u64), dur(u64), rdelay(u64), seed(u64), status(u8), crit(u64),
//     comp(u64).
//   edge_hash: enumerate alive edges in id ASC. For each fold: id(u64),
//     src(u64), dst(u64), chunk(u32), tinc(u64), seq(u64).
//   page_interval_hash: enumerate, over all plan entries, every (plan,
//     key-index k in [0,pcnt)) pair sorted by (sm ASC, key ASC, start ASC,
//     instr_id ASC, key-index ASC). For each fold: sm(u32), key(u64),
//     start(u64), release(u64), instr_id(u64). (The key-index only breaks
//     ties; it is NOT folded.)
//
// ---- live scalars ----
//   committed_epoch (current value). live_instr_count = alive instrs with
//   status != CANCELLED. planned_interval_count = number of plan entries.
// ---------------------------------------------------------------------------

// One operation. Field meanings depend on op_type:
//   ADD_INSTR:     id=instr_id, a=duration, b=page_count, c=release_delay,
//                  d=payload_seed, page_keys[0..page_count) used.
//   ADD_EDGE:      id=edge_id, src=src_instr, dst=dst_instr, a=chunk_id,
//                  b=target_increment.
//   PLAN_NEXT:     a=limit.
//   COMMIT_PLAN:   a=max_entries.
//   EXECUTE_UNTIL: a=tick_limit, b=max_events.
//   CANCEL_INSTR:  id=instr_id.
//   NEW_EPOCH:     (no fields).
struct alignas(8) MkOp {
    int32_t op_type;        // MK_OP_*
    int32_t reserved0;
    uint64_t id;            // instr_id / edge_id
    uint64_t src;           // ADD_EDGE
    uint64_t dst;           // ADD_EDGE
    uint64_t a;             // duration / chunk_id / limit / max_entries / tick_limit
    uint64_t b;             // page_count / target_increment / max_events
    uint64_t c;             // release_delay
    uint64_t d;             // payload_seed
    uint64_t page_keys[MK_MAX_PAGE_KEYS_PER_INSTR];
};

struct alignas(8) MkProblemSpec {
    int32_t abi_version;
    int32_t sm_count;
    int32_t pages_per_sm;
    int32_t max_instrs;
    int32_t max_edges;
    int32_t max_ops_per_step;
    int32_t max_steps;
    int32_t flags;
    uint64_t wave_quantum;
    int32_t reserved[6];
};

struct alignas(8) MkRunSpec {
    int32_t abi_version;
    int32_t num_ops;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) MkInputs {
    const MkOp* ops;
};

struct alignas(8) MkOutputs {
    // Cumulative counters over the entire replay so far.
    uint64_t* instr_added;
    uint64_t* edge_added;
    uint64_t* plan_invalidated;
    uint64_t* plan_empty;
    uint64_t* plan_stall;
    uint64_t* plan_placed;
    uint64_t* plan_committed;
    uint64_t* instr_executed;
    uint64_t* edge_signaled;
    uint64_t* instr_cancelled;
    uint64_t* epoch_advanced;
    uint64_t* invalid_count;

    // Rolling event hash (entire replay, in emission order).
    uint64_t* planner_event_hash;

    // Structural state hashes (current state after this step).
    uint64_t* plan_hash;
    uint64_t* instr_hash;
    uint64_t* edge_hash;
    uint64_t* page_interval_hash;

    // Live scalars.
    uint64_t* committed_epoch;
    uint64_t* live_instr_count;
    uint64_t* planned_interval_count;
};

static_assert(sizeof(MkOp) == 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 * MK_MAX_PAGE_KEYS_PER_INSTR,
              "MkOp layout drift");
static_assert(sizeof(MkProblemSpec) == 64, "MkProblemSpec layout drift");
static_assert(sizeof(MkRunSpec) == 64, "MkRunSpec layout drift");
static_assert(sizeof(MkInputs) == 8, "MkInputs layout drift");
static_assert(sizeof(MkOutputs) == 20 * sizeof(void*), "MkOutputs layout drift");

static inline int mk_validate_problem_spec(const MkProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != MK_ABI_VERSION) return 0;
    if (spec->sm_count < MK_MIN_SM_COUNT || spec->sm_count > MK_MAX_SM_COUNT) return 0;
    if (spec->pages_per_sm < MK_MIN_PAGES_PER_SM || spec->pages_per_sm > MK_MAX_PAGES_PER_SM) return 0;
    if (spec->pages_per_sm > MK_MAX_PAGE_KEYS_PER_INSTR) return 0;
    if (spec->wave_quantum < MK_MIN_WAVE_QUANTUM || spec->wave_quantum > MK_MAX_WAVE_QUANTUM) return 0;
    if (spec->max_instrs < MK_MIN_MAX_INSTRS || spec->max_instrs > MK_MAX_MAX_INSTRS) return 0;
    if (spec->max_edges < MK_MIN_MAX_EDGES || spec->max_edges > MK_MAX_MAX_EDGES) return 0;
    if (spec->max_ops_per_step < 0 || spec->max_ops_per_step > MK_MAX_OPS_PER_STEP) return 0;
    if (spec->max_steps < 1 || spec->max_steps > MK_MAX_STEPS) return 0;
    return 1;
}

static inline int mk_validate_run_spec(const MkRunSpec* run, const MkProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != MK_ABI_VERSION) return 0;
    if (run->num_ops < 0 || run->num_ops > spec->max_ops_per_step) return 0;
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

#endif  // MK_SCHEDULE_PLANNER_COMMON_H_
