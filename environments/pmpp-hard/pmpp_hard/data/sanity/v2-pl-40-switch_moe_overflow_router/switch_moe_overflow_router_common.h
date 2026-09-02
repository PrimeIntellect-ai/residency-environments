// file: switch_moe_overflow_router_common.h

#ifndef SWITCH_MOE_OVERFLOW_ROUTER_COMMON_H_
#define SWITCH_MOE_OVERFLOW_ROUTER_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define SMOR_ABI_VERSION 1

// ---- Limits (problem-spec bounds) --------------------------------------
#define SMOR_MIN_EXPERTS 1
#define SMOR_MAX_EXPERTS 256
#define SMOR_MIN_LIVE 1
#define SMOR_MAX_LIVE 65536
#define SMOR_MIN_OVERFLOW 0
#define SMOR_MAX_OVERFLOW 65536
#define SMOR_MIN_CANDS 1
#define SMOR_MAX_CANDS 64
// token_id is direct-indexed in the reference state; this bounds that space.
#define SMOR_MIN_TOKEN_SPACE 1
#define SMOR_MAX_TOKEN_SPACE 1048576

#define SMOR_MAX_BATCH 4096
#define SMOR_MAX_STEPS 64
// total candidates packed across one batch
#define SMOR_MAX_CAND_TOTAL (SMOR_MAX_BATCH * 8)

// ---- Operation kinds (input op_kind values) ----------------------------
#define SMOR_OP_REFILL 0
#define SMOR_OP_ROUTE 1
#define SMOR_OP_DRAIN 2
#define SMOR_OP_RETIRE 3
#define SMOR_OP_DROP_QUEUED_THROUGH 4

// ---- Event kinds (ordinal order; folded as u8 in route_event_hash) -----
#define SMOR_EV_REFILL 0
#define SMOR_EV_ACCEPT_PRIMARY 1
#define SMOR_EV_ACCEPT_SECONDARY 2
#define SMOR_EV_QUEUE 3
#define SMOR_EV_REPLAY_PRIMARY 4
#define SMOR_EV_REPLAY_SECONDARY 5
#define SMOR_EV_CAPACITY_DROP 6
#define SMOR_EV_OOM_DROP 7
#define SMOR_EV_OOM_DROP_REPLAY 8
#define SMOR_EV_DUPLICATE 9
#define SMOR_EV_RETIRE_LIVE 10
#define SMOR_EV_RETIRE_QUEUED 11
#define SMOR_EV_QUEUE_DROP 12
#define SMOR_EV_INVALID 13

// ---- Token status ------------------------------------------------------
#define SMOR_STATUS_FREE 0
#define SMOR_STATUS_LIVE 1
#define SMOR_STATUS_QUEUED 2

// ---- Route kind (route_kind for live tokens) ---------------------------
#define SMOR_RK_PRIMARY 0
#define SMOR_RK_SECONDARY 1
#define SMOR_RK_REPLAY_PRIMARY 2
#define SMOR_RK_REPLAY_SECONDARY 3

#define SMOR_U32_MAX 0xFFFFFFFFu
#define SMOR_U64_MAX 0xFFFFFFFFFFFFFFFFull
#define SMOR_FNV_OFFSET 1469598103934665603ull
#define SMOR_FNV_PRIME 1099511628211ull

/*
CONTRACT: switch_moe_overflow_router  (PMPP-Hard T40)

A persistent Switch-MoE token router with exact top-2 integer routing,
per-expert credits, FIFO overflow replay, and ordered retirement streams.

PERSISTENT STATE (after solution_init / solution_reset):
  step_seq = 0; event_seq = 0.
  per expert e in [0,E):
    credit[e]      = initial_credit[e]
    live_count[e]  = 0
    ordered live assignment list (sorted by admit_seq).
  Token table keyed by token_id (LIVE or QUEUED tokens only):
    token_id, status, primary_expert, secondary_expert, cost,
    arrival_seq, admit_seq (LIVE only), assigned_expert (LIVE only),
    route_kind (LIVE only).
  Overflow FIFO of QUEUED token references, ordered head->tail by arrival_seq.
  No dropped or retired token is retained.

PER-STEP OPERATIONS (processed in input order):

  REFILL(expert, amount):
    if expert >= E -> INVALID.
    else credit[expert] = min(credit_cap[expert],
                              sat_add_u64(credit[expert], amount));
         emit REFILL.

  ROUTE(token_id, cost, candidate_count, candidates[]):
    INVALID if cost == 0 || candidate_count == 0 ||
              candidate_count > max_candidates_per_route.
    Each candidate = (expert_id, logit_i32, candidate_ordinal).
    Drop candidates with expert_id >= E.
    Collapse repeated expert_id: keep highest logit; tie -> smallest ordinal.
    If no valid candidate remains -> INVALID.
    Sort remaining by descending logit, then ascending expert_id.
    primary = first; secondary = second if present else = primary.
    If token_id already in table -> emit DUPLICATE, no state mutation.
    Else attempt admission:
      try primary: if credit[primary] >= cost -> accept primary.
      else if secondary != primary && credit[secondary] >= cost -> accept secondary.
      else admission fails.
    If accepted but live token count == max_live_tokens:
      emit OOM_DROP (no credit consumed).
    If accepted and capacity exists:
      decrement chosen credit by cost, create LIVE token,
      admit_seq = event_seq of ACCEPT, live_count[expert]++,
      emit ACCEPT_PRIMARY / ACCEPT_SECONDARY.
    If admission failed:
      if overflow queue has room AND token table has room:
        create QUEUED token (arrival_seq = event_seq of QUEUE),
        append to overflow FIFO, emit QUEUE.
      else if overflow queue is full: emit CAPACITY_DROP.
      else (queue room exists but token table full): emit OOM_DROP.
    OOM priority: duplicate check, candidate resolution, admission, storage.

  DRAIN(limit):
    if limit == 0 -> valid no-op (no event).
    while limit > 0 and FIFO nonempty:
      look only at head entry. Try admission (REPLAY_PRIMARY/REPLAY_SECONDARY).
      if neither expert has enough credit -> stop (head-of-line block).
      if admission succeeds: pop head, LIVE, consume credit, set admit_seq,
        live_count++, emit REPLAY_PRIMARY/REPLAY_SECONDARY, limit--.
      if live capacity would exceed during replay: pop head, remove token table
        entry, emit OOM_DROP_REPLAY, limit--, continue.
    (admit_seq / arrival_seq of replay use event_seq at the emitted event.)

  RETIRE(token_id):
    if live: remove from expert live list + table, live_count[assigned]--,
      emit RETIRE_LIVE. No credit refund.
    if queued: remove from FIFO + table, emit RETIRE_QUEUED.
    if absent -> INVALID.

  DROP_QUEUED_THROUGH(cutoff_arrival_seq):
    scan FIFO head->tail; drop every queued token with
    arrival_seq <= cutoff_arrival_seq, stop at first larger arrival_seq.
    emit QUEUE_DROP per removed token, in FIFO order.

EVENT SEQ: event_seq increments once per emitted event, wraps mod 2^64.
  The replay/admission credit checks above use the per-expert credit;
  the admission "accept" priority is primary-then-secondary, the DRAIN
  capacity-exceed branch is checked at admission time (before consuming).

OUTPUTS (all exact integers): the 13 event counts plus 4 FNV-1a-64 hashes.
  The exact field-emission order, integer widths, byte order, sort order,
  and per-event aux semantics are specified normatively below.
*/

// === CONTRACT: SEMANTICS + EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section inlines every rule a solver needs to reproduce all four graded
// hashes and the assignment sort bit-for-bit. It is derived directly from the
// reference implementation; there is no separate "section 4" or oracle to
// consult. A conforming solution MUST reproduce these exact values.
//
// -------------------------------------------------------------------------
// 0. FNV-1a-64 PRIMITIVE (shared by ALL four hashes)
// -------------------------------------------------------------------------
//   - offset basis : SMOR_FNV_OFFSET = 1469598103934665603 (0x14650FB0739D0383)
//   - prime        : SMOR_FNV_PRIME  = 1099511628211       (0x00000100000001b3)
//   - Per byte b:   h = (h XOR b) * SMOR_FNV_PRIME   (mod 2^64, wrapping).
//     i.e. XOR the byte FIRST, then multiply (FNV-1a, NOT FNV-1).
//   - A field of width W bytes is "folded" by feeding its W bytes in
//     LITTLE-ENDIAN order (least-significant byte first) one at a time
//     through the per-byte step. This is exactly memcpy/byte-iteration of the
//     value's native little-endian storage (the reference runs on a
//     little-endian GPU/host; all field widths below assume little-endian).
//   - Integer widths used below:
//       u8  = 1 byte, u32 = 4 bytes, u64 = 8 bytes.
//   - Each of the four hashes is SEEDED INDEPENDENTLY at SMOR_FNV_OFFSET
//     before its first field, and emitted as a raw uint64_t (no final mix).
//   - An EMPTY hash (zero records iterated) equals SMOR_FNV_OFFSET unchanged.
//
// -------------------------------------------------------------------------
// 1. route_event_hash  (per-STEP; reflects THIS step's emitted events only)
// -------------------------------------------------------------------------
//   Seeded at SMOR_FNV_OFFSET at the start of each step's processing.
//   Every emitted event (and ONLY emitted events) folds, in this EXACT order:
//       1) event_kind     u8   (one of SMOR_EV_* above)
//       2) event_seq      u64  (value of event_seq BEFORE this event's
//                               increment; i.e. the seq assigned to THIS event)
//       3) op_index       u32  (the batch index i of the operation, 0-based)
//       4) token_id       u64  (op token_id; 0 for REFILL/unknown-op INVALID)
//       5) expert_or_max  u32  (see per-event aux table below; SMOR_U32_MAX
//                               sentinel when "n/a")
//       6) primary        u32  (resolved primary expert; SMOR_U32_MAX if n/a)
//       7) secondary      u32  (resolved secondary expert; SMOR_U32_MAX if n/a)
//       8) cost           u64  (op cost; 0 when n/a)
//       9) credit_after   u64  (chosen expert's credit AFTER mutation;
//                               SMOR_U64_MAX when n/a)
//      10) arrival        u64  (arrival_seq aux; SMOR_U64_MAX when n/a)
//   event_seq starts at 0 after init/reset, increments by exactly 1 per
//   emitted event, persists ACROSS steps, and wraps mod 2^64. Operations that
//   emit no event (DRAIN with limit==0; DRAIN head-of-line block; empty
//   DROP_QUEUED_THROUGH; per-iteration loop exits) do NOT advance event_seq.
//
//   PER-EVENT AUX FIELD VALUES (exactly what the reference folds):
//     SMOR_EV_REFILL:           token_id=0, expert_or_max=expert,
//                               primary=MAX, secondary=MAX, cost=0,
//                               credit_after=new credit (post cap-clamp),
//                               arrival=MAX.
//     SMOR_EV_ACCEPT_PRIMARY/ _SECONDARY:
//                               token_id=token, expert_or_max=chosen expert,
//                               primary,secondary=resolved, cost=cost,
//                               credit_after=credit[chosen] after subtracting
//                               cost, arrival=MAX.
//     SMOR_EV_QUEUE:            token_id=token, expert_or_max=MAX,
//                               primary,secondary=resolved, cost=cost,
//                               credit_after=MAX, arrival=arrival_seq assigned
//                               (== event_seq of THIS QUEUE event).
//     SMOR_EV_REPLAY_PRIMARY/ _SECONDARY:
//                               token_id=token, expert_or_max=chosen expert,
//                               primary,secondary=token's stored resolved
//                               experts, cost=token's stored cost,
//                               credit_after=credit[chosen] after subtract,
//                               arrival=MAX.
//     SMOR_EV_CAPACITY_DROP:    token_id=token, expert_or_max=MAX,
//                               primary,secondary=resolved, cost=cost,
//                               credit_after=MAX, arrival=MAX.
//     SMOR_EV_OOM_DROP (from ROUTE accept-but-no-live-room):
//                               token_id=token, expert_or_max=chosen expert,
//                               primary,secondary=resolved, cost=cost,
//                               credit_after=MAX, arrival=MAX.
//     SMOR_EV_OOM_DROP (from ROUTE admission-fail, queue-room-but-table-full):
//                               token_id=token, expert_or_max=MAX,
//                               primary,secondary=resolved, cost=cost,
//                               credit_after=MAX, arrival=MAX.
//     SMOR_EV_OOM_DROP_REPLAY (from DRAIN, would exceed max_live):
//                               token_id=token, expert_or_max=MAX,
//                               primary,secondary=token's stored experts,
//                               cost=token's stored cost,
//                               credit_after=MAX, arrival=MAX.
//                               NOTE: this event has NO dedicated count field;
//                               it is reflected ONLY in route_event_hash and
//                               increments NONE of the 13 counts (in
//                               particular it does NOT bump oom_drop).
//     SMOR_EV_DUPLICATE:        token_id=token, expert_or_max=MAX,
//                               primary,secondary=resolved, cost=cost,
//                               credit_after=MAX, arrival=MAX.
//     SMOR_EV_RETIRE_LIVE:      token_id=token, expert_or_max=assigned expert,
//                               primary,secondary=token's stored experts,
//                               cost=token's stored cost,
//                               credit_after=MAX, arrival=MAX.
//     SMOR_EV_RETIRE_QUEUED:    token_id=token, expert_or_max=MAX,
//                               primary,secondary=token's stored experts,
//                               cost=token's stored cost, credit_after=MAX,
//                               arrival=token's stored arrival_seq.
//     SMOR_EV_QUEUE_DROP:       token_id=token, expert_or_max=MAX,
//                               primary,secondary=token's stored experts,
//                               cost=token's stored cost, credit_after=MAX,
//                               arrival=token's stored arrival_seq.
//     SMOR_EV_INVALID:          token_id = op token_id where one exists
//                               (ROUTE/RETIRE), else 0 (REFILL bad-expert,
//                               unknown op kind); expert_or_max=MAX,
//                               primary=MAX, secondary=MAX,
//                               cost = op cost for ROUTE-INVALID else 0,
//                               credit_after=MAX, arrival=MAX.
//
// -------------------------------------------------------------------------
// 2. credit_hash  (per-STEP; reflects POST-step persistent state)
// -------------------------------------------------------------------------
//   Seeded at SMOR_FNV_OFFSET. Iterate experts e = 0 .. E-1 ASCENDING
//   (expert-major). For each expert fold, in this EXACT order:
//       1) eid         u32  (= e)
//       2) credit[e]   u64  (current credit, post-step)
//       3) live_count[e] u64 (current live token count for e, post-step)
//   Always emits exactly E records (no skipping).
//
// -------------------------------------------------------------------------
// 3. assignment_hash  (per-STEP; reflects POST-step set of LIVE tokens)
// -------------------------------------------------------------------------
//   Set = every token currently in status LIVE. Sort that set by the
//   key tuple (ASCENDING on each, in priority order):
//       1) assigned_expert  (u32)   ascending  -- primary key
//       2) admit_seq        (u64)   ascending  -- tie-break 1
//       3) token_id         (u64)   ascending  -- tie-break 2 (total order;
//                                                 token_ids are unique so this
//                                                 fully disambiguates)
//   The sort is a stable total order; the reference uses insertion sort and
//   the oracle uses std::sort with this comparator -- both yield the same
//   order because token_id is unique. Then seed SMOR_FNV_OFFSET and, for each
//   LIVE token in sorted order, fold in this EXACT order:
//       1) assigned_expert  u32
//       2) token_id         u64
//       3) admit_seq        u64
//       4) primary          u32   (resolved primary at admission)
//       5) secondary        u32   (resolved secondary at admission)
//       6) cost             u64
//       7) route_kind       u8    (SMOR_RK_*: PRIMARY/SECONDARY/
//                                  REPLAY_PRIMARY/REPLAY_SECONDARY, reflecting
//                                  how the token went LIVE)
//   QUEUED tokens are NOT included. Empty live set => SMOR_FNV_OFFSET.
//
// -------------------------------------------------------------------------
// 4. overflow_hash  (per-STEP; reflects POST-step overflow FIFO)
// -------------------------------------------------------------------------
//   Iterate the overflow FIFO in HEAD -> TAIL order (oldest arrival first; the
//   FIFO is maintained in arrival order). For each QUEUED token fold, in this
//   EXACT order:
//       1) token_id     u64
//       2) arrival_seq  u64   (== event_seq of the QUEUE event that enqueued it)
//       3) primary      u32   (resolved primary)
//       4) secondary    u32   (resolved secondary)
//       5) cost         u64
//   Empty FIFO => SMOR_FNV_OFFSET.
//
// -------------------------------------------------------------------------
// 5. OUTPUT COUNTS (13 scalars), index/meaning (each a u64, count of events):
// -------------------------------------------------------------------------
//   refill_count        = #REFILL events (valid refills only)
//   accepted_primary    = #ACCEPT_PRIMARY      replayed_primary  = #REPLAY_PRIMARY
//   accepted_secondary  = #ACCEPT_SECONDARY    replayed_secondary= #REPLAY_SECONDARY
//   queued              = #QUEUE               capacity_drop     = #CAPACITY_DROP
//   oom_drop            = #OOM_DROP  (ROUTE-path only; NOT OOM_DROP_REPLAY)
//   duplicate_count     = #DUPLICATE           retired_live      = #RETIRE_LIVE
//   retired_queued      = #RETIRE_QUEUED       queue_drop        = #QUEUE_DROP
//   invalid_count       = #INVALID
//   All counts RESET to 0 at the start of every step (per-step counts). The
//   four hashes for credit/assignment/overflow reflect post-step STATE;
//   route_event_hash reflects only THIS step's events; event_seq/step_seq are
//   the only sequence counters that persist and accumulate across steps.

struct alignas(8) SmorProblemSpec {
    int32_t abi_version;
    int32_t num_experts;            // E
    int32_t max_live_tokens;
    int32_t overflow_capacity;
    int32_t max_candidates_per_route;
    int32_t token_space;            // token_id direct-index bound [0, token_space)
    int32_t max_batch;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[7];
};

struct alignas(8) SmorRunSpec {
    int32_t abi_version;
    int32_t batch_size;             // number of operations this step
    int32_t cand_total;             // packed candidate entries this step
    int32_t step_id;
    int32_t reserved[12];
};

// Persistent per-expert configuration passed at init via the problem-config
// blob (credit_cap[E], initial_credit[E]) inside SmorInitConfig.
struct alignas(8) SmorInitConfig {
    const uint64_t* credit_cap;     // length E
    const uint64_t* initial_credit; // length E
};

struct alignas(8) SmorInputs {
    const int32_t*  op_kind;        // length batch_size
    const uint64_t* op_a;           // REFILL:expert ROUTE:token_id DRAIN:limit
                                    // RETIRE:token_id DROP:cutoff_arrival_seq
    const uint64_t* op_b;           // REFILL:amount  ROUTE:cost
    const int32_t*  op_cand_off;    // ROUTE: offset into cand_* arrays
    const int32_t*  op_cand_count;  // ROUTE: raw candidate_count
    const int32_t*  cand_expert;    // length cand_total
    const int32_t*  cand_logit;     // length cand_total
    const int32_t*  cand_ordinal;   // length cand_total
};

struct alignas(8) SmorOutputs {
    uint64_t* refill_count;
    uint64_t* accepted_primary;
    uint64_t* accepted_secondary;
    uint64_t* queued;
    uint64_t* replayed_primary;
    uint64_t* replayed_secondary;
    uint64_t* capacity_drop;
    uint64_t* oom_drop;
    uint64_t* duplicate_count;
    uint64_t* retired_live;
    uint64_t* retired_queued;
    uint64_t* queue_drop;
    uint64_t* invalid_count;
    uint64_t* route_event_hash;
    uint64_t* credit_hash;
    uint64_t* assignment_hash;
    uint64_t* overflow_hash;
};

static_assert(sizeof(SmorProblemSpec) == 64, "SmorProblemSpec layout drift");
static_assert(sizeof(SmorRunSpec) == 64, "SmorRunSpec layout drift");
static_assert(sizeof(SmorInitConfig) == 16, "SmorInitConfig layout drift");
static_assert(sizeof(SmorInputs) == 64, "SmorInputs layout drift");
static_assert(sizeof(SmorOutputs) == 136, "SmorOutputs layout drift");

static inline size_t smor_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int smor_validate_problem_spec(const SmorProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != SMOR_ABI_VERSION) return 0;
    if (spec->num_experts < SMOR_MIN_EXPERTS || spec->num_experts > SMOR_MAX_EXPERTS) return 0;
    if (spec->max_live_tokens < SMOR_MIN_LIVE || spec->max_live_tokens > SMOR_MAX_LIVE) return 0;
    if (spec->overflow_capacity < SMOR_MIN_OVERFLOW || spec->overflow_capacity > SMOR_MAX_OVERFLOW) return 0;
    if (spec->max_candidates_per_route < SMOR_MIN_CANDS || spec->max_candidates_per_route > SMOR_MAX_CANDS) return 0;
    if (spec->token_space < SMOR_MIN_TOKEN_SPACE || spec->token_space > SMOR_MAX_TOKEN_SPACE) return 0;
    if (spec->max_batch < 0 || spec->max_batch > SMOR_MAX_BATCH) return 0;
    if (spec->max_steps < 1 || spec->max_steps > SMOR_MAX_STEPS) return 0;
    return 1;
}

static inline int smor_validate_run_spec(const SmorRunSpec* run, const SmorProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != SMOR_ABI_VERSION) return 0;
    if (run->batch_size < 0 || run->batch_size > spec->max_batch) return 0;
    if (run->cand_total < 0 || run->cand_total > SMOR_MAX_CAND_TOTAL) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const SmorProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const SmorProblemSpec* spec,
    const SmorInitConfig* config,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const SmorRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // SWITCH_MOE_OVERFLOW_ROUTER_COMMON_H_
