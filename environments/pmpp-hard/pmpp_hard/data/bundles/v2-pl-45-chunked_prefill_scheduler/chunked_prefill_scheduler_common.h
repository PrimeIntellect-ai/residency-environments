// file: chunked_prefill_scheduler_common.h

#ifndef CHUNKED_PREFILL_SCHEDULER_COMMON_H_
#define CHUNKED_PREFILL_SCHEDULER_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define CPS_ABI_VERSION 1

#define CPS_MIN_TENANTS 1
#define CPS_MAX_TENANTS 64
#define CPS_MIN_EXPERTS 1
#define CPS_MAX_EXPERTS 64
#define CPS_MAX_LIVE 256
#define CPS_MAX_BATCH_SLOTS 256
#define CPS_MAX_OPS 256

/* Operation opcodes (CpsOp.opcode) */
#define CPS_OP_ARRIVE        1
#define CPS_OP_REFILL_TENANT 2
#define CPS_OP_SET_KV_CAP    3
#define CPS_OP_CANCEL        4
#define CPS_OP_STEP_ITER     5

/* phase codes */
#define CPS_PHASE_PREFILL 0
#define CPS_PHASE_DECODE  1

/* MoE route kinds */
#define CPS_ROUTE_PRIMARY   0
#define CPS_ROUTE_SECONDARY 1
#define CPS_ROUTE_DROP      2

/* batch record kinds */
#define CPS_REC_BATCH_DECODE  0
#define CPS_REC_BATCH_PREFILL 1

/* finalize event kinds */
#define CPS_EV_FINALIZE_COMPLETE 0
#define CPS_EV_FINALIZE_CANCEL   1
#define CPS_EV_FINALIZE_EVICT    2
#define CPS_EV_EVICT_KV_SHRINK   3

/* queue kinds */
#define CPS_Q_PREFILL 0
#define CPS_Q_DECODE  1

/*
CONTRACT: chunked_prefill_scheduler (T45)

Multi-Tenant Chunked-Prefill Continuous Batch Scheduler with MoE Pressure.

A persistent, deterministic, exact-integer LLM-serving scheduler. The
solution maintains scheduler state across a stream of operations (ops),
each issued by one solution_run call. After every op the solution emits a
FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
  the agent MUST use this exact basis or every checksum fails):
    offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
      the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
    prime        = 1099511628211  (0x100000001B3).
    fold: start h = offset basis; absorb each field's raw bytes little-endian at
      its full declared width (u8=1, u32=4, u64=8, i64=8 two's-complement); per
      byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator / final mix.

full snapshot of cumulative counts plus eight cumulative FNV-1a-64
checksums describing the scheduler's behavior and live state.

------------------------------------------------------------------
PERSISTENT STATE (after init/reset)

Scalars:
  iter_seq        = 0   (u64, wraps mod 2^64)
  event_seq       = 0   (u64, wraps mod 2^64)
  arrival_seq_next= 1   (u64, wraps mod 2^64)
  kv_capacity_tokens    (u64, set from spec.kv_capacity_tokens; mutable)

Per tenant (num_tenants):
  bucket_tokens[t]      (u64, init = spec.initial_bucket_tokens[t])
  bucket_cap[t]         (u64, from spec, constant)

Live request table keyed by request_id (only live requests are present):
  request_id (u64); tenant (u32); priority (u8, larger = higher);
  prompt_len (u64); prompt_done (u64); decode_len (u64); decode_done (u64);
  chunk_max (u64); kv_tokens (u64); arrival_seq (u64);
  last_scheduled_iter (u64, UINT64_MAX = never, older than any iter);
  moe_drop_count (u64); phase (u8 in {PREFILL,DECODE}).

Queues (each stores request_ids):
  prefill_queue (FIFO)
  decode_queue  (FIFO)
A live request is in exactly one queue except transiently inside STEP_ITER.

------------------------------------------------------------------
OPERATIONS  (one per solution_run; op fields in CpsOp)

ARRIVE(request_id, tenant, priority, prompt_len, decode_len, chunk_max)
  Invalid (++invalid_count, no other effect) if:
    tenant >= num_tenants, OR (prompt_len==0 AND decode_len==0), OR
    request_id already live, OR stored chunk_max==0.
  stored chunk_max = (chunk_max != 0) ? chunk_max : default_chunk_max.
  If live table full -> ARRIVE_REJECT (++arrive_reject). (checked before
    storing, after the above validity checks).
  Else create request: arrival_seq=arrival_seq_next++, last_scheduled_iter
    =UINT64_MAX, kv_tokens=0, moe_drop_count=0, prompt_done=0, decode_done=0.
    If prompt_len>0: phase=PREFILL, append to prefill_queue tail.
    Else: phase=DECODE, append to decode_queue tail.
    ++arrive_ok.

REFILL_TENANT(tenant, amount)  [amount in op.a]
  Invalid (++invalid_count) if tenant >= num_tenants.
  Else bucket_tokens[t] = min(bucket_cap[t],
        saturating_add_u64(bucket_tokens[t], amount)); ++tenant_refills.

SET_KV_CAP(new_capacity)  [new_capacity in op.a]
  kv_capacity_tokens = new_capacity.
  While total live kv_tokens > kv_capacity_tokens:
    evict one request via global eviction rule with EMPTY protected set;
    emit EVICT_KV_SHRINK (++kv_shrink_evicted). If no live requests, stop.

CANCEL(request_id)
  Invalid (++invalid_count) if request_id not live.
  Else remove from its queue, free kv_tokens, remove from table,
    emit FINALIZE_CANCEL (++cancelled).

STEP_ITER(token_budget, slot_budget)  [token_budget=op.a, slot_budget=op.b]
  ++iterations. Emit ITER_BEGIN (counter-only; no hash record).
  tokens_left = token_budget.
  slots_left  = min(slot_budget, max_batch_slots).
  expert_remaining[e] = expert_capacity[e]   (reset every STEP_ITER).
  protected_set = empty.
  If tokens_left==0 OR slots_left==0: emit ITER_END, ++iter_seq, stop.

  DECODE PASS:
    decode_scan_count = length(decode_queue) at pass start.
    Repeat at most decode_scan_count times while tokens_left>0 && slots_left>0:
      pop head id. If not live, skip (still counts against scan budget).
      If bucket_tokens[tenant]==0: ++throttle_skips, append to decode tail,
        emit THROTTLE_SKIP, continue.
      Ensure >=1 free KV token. free = kv_capacity - sum(live kv_tokens).
        If free==0: invoke eviction needed_free=1, protected_set + this req.
        Recompute free. If still 0: append to decode tail, ++kv_skips,
        emit KV_SKIP, continue.
      Schedule exactly ONE decode token:
        absolute_token_index = prompt_len + decode_done.
        record BATCH_DECODE (token_count=1) using bucket_before/kv_before/
          tokens_left_before captured BEFORE consuming.
        route this single token through MoE (see routing).
        bucket_tokens[t]-=1; tokens_left-=1; slots_left-=1; kv_tokens+=1;
        decode_done+=1; last_scheduled_iter=iter_seq; add to protected_set.
      If decode_done==decode_len: free kv, remove, emit FINALIZE_COMPLETE
        (++completed). Else append to decode tail.

  PREFILL PASS:
    prefill_scan_count = length(prefill_queue) at pass start.
    Repeat at most prefill_scan_count times while tokens_left>0 && slots_left>0:
      pop head id. If not live, skip.
      tenant_tokens = bucket_tokens[tenant].
      If tenant_tokens==0: ++throttle_skips, append to prefill tail,
        emit THROTTLE_SKIP, continue.
      remaining_prompt = prompt_len - prompt_done.
      chunk = min(chunk_max, remaining_prompt, tokens_left, tenant_tokens).
      Ensure >=chunk free KV. free=kv_capacity-sum(live kv). If free<chunk:
        invoke eviction needed_free=chunk, protected_set + this req;
        recompute free; chunk = min(chunk, free).
      If chunk==0: append to prefill tail, ++kv_skips, emit KV_SKIP, continue.
      record BATCH_PREFILL (token_count=chunk) with *_before captured BEFORE
        consuming. For j=0..chunk-1 (increasing): absolute_token_index=
        prompt_done+j; route each token through MoE in order.
      prompt_done+=chunk; kv_tokens+=chunk; tokens_left-=chunk;
        bucket_tokens[t]-=chunk; slots_left-=1; last_scheduled_iter=iter_seq;
        add to protected_set.
      If prompt_done==prompt_len && decode_len>0: phase=DECODE, append to
        decode tail (NOT eligible this iteration's decode pass).
      Else if prompt_done==prompt_len && decode_len==0: finalize complete
        (free kv, remove, FINALIZE_COMPLETE, ++completed).
      Else append back to prefill tail.

  Emit ITER_END, ++iter_seq.

MoE ROUTING (per scheduled token, in exact token execution order):
  h = FNV1a64 over (moe_seed:u64, iter_seq:u64, request_id:u64, tenant:u32,
        phase_code:u8, absolute_token_index:u64) starting at FNV offset.
  primary = h % num_experts.
  secondary = (num_experts==1) ? primary
              : (primary + 1 + ((h>>32) % (num_experts-1))) % num_experts.
  If expert_remaining[primary]>0: route PRIMARY, --expert_remaining[primary],
      assigned=primary, ++moe_primary.
  Else if secondary!=primary && expert_remaining[secondary]>0: route SECONDARY,
      --expert_remaining[secondary], assigned=secondary, ++moe_secondary.
  Else: route DROP, assigned=UINT32_MAX, ++moe_dropped, ++req.moe_drop_count.
  A DROP still consumes tenant bucket, KV, token budget, and advances progress.

EVICTION RULE (returns chosen request or none):
  Candidates = live requests NOT in protected_set AND != current candidate.
  Choose request with: smallest priority; tie largest kv_tokens; tie oldest
    last_scheduled_iter (UINT64_MAX oldest); tie smallest arrival_seq; tie
    smallest request_id.
  Remove from its queue, free kv, remove from table, emit FINALIZE_EVICT
    (++evicted). Repeat until enough free or no candidates.
  KV-shrink eviction uses same ordering but emits EVICT_KV_SHRINK
    (++kv_shrink_evicted) and has empty protected/candidate exclusions.

------------------------------------------------------------------
OUTPUTS after every op (all cumulative since last reset)

Counts (CpsCounts), all u64:
  arrive_ok arrive_reject tenant_refills iterations batch_decode_requests
  batch_prefill_requests decode_tokens prefill_tokens moe_primary
  moe_secondary moe_dropped throttle_skips kv_skips completed cancelled
  evicted kv_shrink_evicted invalid_count.

Eight cumulative FNV-1a-64 checksums (folded in emission/traversal order,
each starting from the FNV offset at reset and updated as events occur,
except snapshot hashes which are recomputed fresh each op over current
state and then folded — see below). Each is emitted as the current value.

  batch_hash: for each scheduled chunk in batch execution order, fold:
    record_kind:u8 iter_seq:u64 slot_ordinal:u32 request_id:u64 tenant:u32
    phase_code:u8 start_abs_token:u64 token_count:u64 bucket_before:u64
    kv_before:u64 tokens_left_before:u64.
    (slot_ordinal = number of chunks already scheduled in THIS STEP_ITER,
     i.e. 0-based index of this chunk within the iteration.)

  moe_hash: for each scheduled token in token execution order, fold:
    iter_seq:u64 global_token_ordinal_in_iter:u64 request_id:u64 tenant:u32
    phase_code:u8 absolute_token_index:u64 primary:u32 secondary:u32
    route_kind:u8 assigned_expert_or_UINT32_MAX:u32
    expert_remaining_after_or_UINT64_MAX:u64.
    (global_token_ordinal_in_iter = 0-based count of tokens routed so far in
     THIS STEP_ITER. expert_remaining_after = remaining capacity of the
     assigned expert AFTER decrement, or UINT64_MAX on DROP.)

  finalize_hash: for each finalize event in emission order, fold:
    event_kind:u8 event_seq:u64 op_index:u32 request_id:u64 tenant:u32
    priority:u8 prompt_done:u64 decode_done:u64 kv_tokens_freed:u64
    moe_drop_count:u64 reason:u8.
    (event_seq = value BEFORE increment, i.e. 0-based; ++event_seq after.
     op_index = 0-based index of the op producing the event.
     reason = event_kind.)

  queue_hash, request_hash, bucket_hash: SNAPSHOT hashes recomputed fresh
    over CURRENT state after the op completes, then assigned (not folded):
    queue_hash:   start FNV offset; prefill head->tail then decode head->tail:
                    queue_kind:u8 position:u64 request_id:u64.
    request_hash: start FNV offset; live requests sorted by tenant asc then
                    request_id asc: request_id:u64 tenant:u32 priority:u8
                    phase:u8 prompt_len:u64 prompt_done:u64 decode_len:u64
                    decode_done:u64 chunk_max:u64 kv_tokens:u64 arrival_seq:u64
                    last_scheduled_iter:u64 moe_drop_count:u64.
    bucket_hash:  start FNV offset; tenants asc: tenant:u32 bucket_tokens:u64
                    bucket_cap:u64.

  scalar_hash: SNAPSHOT recomputed fresh after op:
    start FNV offset; iter_seq:u64 event_seq:u64 arrival_seq_next:u64
    kv_capacity_tokens:u64 live_count:u64 total_live_kv:u64.

All outputs are exact integers. No floating point. No tolerance.

DETERMINISM: decode before prefill; pass snapshots queue length at start;
buckets refill only by REFILL_TENANT; eviction excludes current + already
scheduled; MoE experts reset per STEP_ITER; dropped tokens still advance;
counters saturate/clamp, never underflow; iter/event/arrival wrap mod 2^64.
*/

struct alignas(8) CpsProblemSpec {
    int32_t abi_version;
    int32_t num_tenants;
    int32_t kv_capacity_tokens;   // initial; nonnegative, fits in u64
    int32_t max_live_requests;
    int32_t default_chunk_max;
    int32_t max_batch_slots;
    int32_t num_experts;
    int32_t max_ops;
    uint64_t moe_seed;
    uint64_t bucket_cap[CPS_MAX_TENANTS];
    uint64_t initial_bucket_tokens[CPS_MAX_TENANTS];
    uint64_t expert_capacity[CPS_MAX_EXPERTS];
    int32_t flags;
    int32_t reserved[7];
};

/* One scheduler operation. Field meanings depend on opcode (see contract). */
struct alignas(8) CpsOp {
    int32_t abi_version;
    int32_t opcode;        // CPS_OP_*
    int32_t op_index;      // 0-based op ordinal since reset
    uint32_t tenant;       // ARRIVE / REFILL_TENANT
    uint64_t request_id;   // ARRIVE / CANCEL
    uint32_t priority;     // ARRIVE (only low 8 bits used)
    uint32_t reserved0;
    uint64_t prompt_len;   // ARRIVE
    uint64_t decode_len;   // ARRIVE
    uint64_t chunk_max;    // ARRIVE
    uint64_t a;            // REFILL amount / SET_KV_CAP new_cap / STEP token_budget
    uint64_t b;            // STEP slot_budget
    uint64_t reserved1[2];
};

struct alignas(8) CpsCounts {
    uint64_t arrive_ok;
    uint64_t arrive_reject;
    uint64_t tenant_refills;
    uint64_t iterations;
    uint64_t batch_decode_requests;
    uint64_t batch_prefill_requests;
    uint64_t decode_tokens;
    uint64_t prefill_tokens;
    uint64_t moe_primary;
    uint64_t moe_secondary;
    uint64_t moe_dropped;
    uint64_t throttle_skips;
    uint64_t kv_skips;
    uint64_t completed;
    uint64_t cancelled;
    uint64_t evicted;
    uint64_t kv_shrink_evicted;
    uint64_t invalid_count;
};

struct alignas(8) CpsOutputs {
    CpsCounts* counts;        // 1 struct
    uint64_t* batch_hash;     // [1]
    uint64_t* moe_hash;       // [1]
    uint64_t* finalize_hash;  // [1]
    uint64_t* queue_hash;     // [1]
    uint64_t* request_hash;   // [1]
    uint64_t* bucket_hash;    // [1]
    uint64_t* scalar_hash;    // [1]
};

static_assert(sizeof(CpsOp) == 88, "CpsOp layout drift");
static_assert(sizeof(CpsCounts) == 144, "CpsCounts layout drift");
static_assert(sizeof(CpsOutputs) == 64, "CpsOutputs layout drift");

static inline int cps_validate_problem_spec(const CpsProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != CPS_ABI_VERSION) return 0;
    if (spec->num_tenants < CPS_MIN_TENANTS || spec->num_tenants > CPS_MAX_TENANTS) return 0;
    if (spec->num_experts < CPS_MIN_EXPERTS || spec->num_experts > CPS_MAX_EXPERTS) return 0;
    if (spec->max_live_requests < 1 || spec->max_live_requests > CPS_MAX_LIVE) return 0;
    if (spec->max_batch_slots < 1 || spec->max_batch_slots > CPS_MAX_BATCH_SLOTS) return 0;
    if (spec->default_chunk_max < 0) return 0;
    if (spec->kv_capacity_tokens < 0) return 0;
    if (spec->max_ops < 1 || spec->max_ops > CPS_MAX_OPS) return 0;
    return 1;
}

static inline int cps_validate_op(const CpsOp* op, const CpsProblemSpec* spec) {
    if (!op || !spec) return 0;
    if (op->abi_version != CPS_ABI_VERSION) return 0;
    if (op->opcode < CPS_OP_ARRIVE || op->opcode > CPS_OP_STEP_ITER) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const CpsProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const CpsProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const CpsOp* op,
    const void* inputs,   // unused; kept for ABI symmetry (pass nullptr)
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // CHUNKED_PREFILL_SCHEDULER_COMMON_H_
