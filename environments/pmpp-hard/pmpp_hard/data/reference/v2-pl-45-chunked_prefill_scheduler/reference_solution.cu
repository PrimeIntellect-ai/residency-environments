// PMPP_CANARY_45_e6037f1933 -- held-out canary; MUST NOT appear in any submission
// file: chunked_prefill_scheduler_reference.cu
//
// Reference GPU implementation (#1) of T45. A single-thread (<<<1,1>>>)
// device kernel runs the entire scheduler step over a device-resident
// state. The request table is a slot pool; queues are arrays of slot
// indices. Algorithm is implemented independently from the naive impl.

#include "chunked_prefill_scheduler_common.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CPSR_NONE 0xFFFFFFFFu

struct CpsrState {
    CpsProblemSpec spec;
    int cap;   // max_live_requests
    int nt;    // num_tenants
    int ne;    // num_experts

    // scalars (device, length 1 each)
    uint64_t* iter_seq;
    uint64_t* event_seq;
    uint64_t* arrival_seq_next;
    uint64_t* kv_capacity;

    // per tenant
    uint64_t* bucket_tokens;   // [nt]
    uint64_t* bucket_cap;      // [nt]

    // per expert (working set, reset each step)
    uint64_t* expert_capacity; // [ne] constant
    uint64_t* expert_remaining;// [ne] scratch

    // request slot pool [cap]
    uint8_t*  r_live;
    uint64_t* r_request_id;
    uint32_t* r_tenant;
    uint8_t*  r_priority;
    uint64_t* r_prompt_len;
    uint64_t* r_prompt_done;
    uint64_t* r_decode_len;
    uint64_t* r_decode_done;
    uint64_t* r_chunk_max;
    uint64_t* r_kv_tokens;
    uint64_t* r_arrival_seq;
    uint64_t* r_last_sched;
    uint64_t* r_moe_drop;
    uint8_t*  r_phase;
    uint8_t*  r_protected;     // scratch per slot, set during STEP_ITER

    // queues store slot indices
    uint32_t* prefill_q;       // [cap]
    int32_t*  prefill_head;    // [1]
    int32_t*  prefill_len;     // [1]
    uint32_t* decode_q;        // [cap]
    int32_t*  decode_head;     // [1]
    int32_t*  decode_len;      // [1]

    // counts + folded hashes (device)
    CpsCounts* counts;
    uint64_t*  batch_hash;
    uint64_t*  moe_hash;
    uint64_t*  finalize_hash;
};

// ---- device FNV ----
__device__ __forceinline__ uint64_t cpsr_fb(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b; h *= 1099511628211ULL; return h;
}
__device__ void cpsr_fbytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* q = (const uint8_t*)p; uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = cpsr_fb(v, q[i]); *h = v;
}
__device__ __forceinline__ void cpsr_u8(uint64_t* h, uint8_t x){ cpsr_fbytes(h,&x,1); }
__device__ __forceinline__ void cpsr_u32(uint64_t* h, uint32_t x){ cpsr_fbytes(h,&x,4); }
__device__ __forceinline__ void cpsr_u64(uint64_t* h, uint64_t x){ cpsr_fbytes(h,&x,8); }

#define CPSR_FNV_OFFSET 1469598103934665603ULL

__device__ __forceinline__ uint64_t cpsr_sat_add(uint64_t a, uint64_t b){
    uint64_t s = a + b; return (s < a) ? UINT64_MAX : s;
}

// ---- queue helpers (circular buffer of size cap) ----
__device__ __forceinline__ uint32_t cpsr_q_front(const uint32_t* q, int head) {
    return q[head];
}
__device__ void cpsr_q_pop(uint32_t* q, int32_t* head, int32_t* len, int cap) {
    *head = (*head + 1) % cap;
    *len -= 1;
}
__device__ void cpsr_q_push(uint32_t* q, int32_t* head, int32_t* len, int cap, uint32_t slot) {
    int tail = (*head + *len) % cap;
    q[tail] = slot;
    *len += 1;
}
// remove first occurrence of slot from a queue (linear)
__device__ void cpsr_q_remove(uint32_t* q, int32_t* head, int32_t* len, int cap, uint32_t slot) {
    int n = *len; int h = *head;
    int found = -1;
    for (int i = 0; i < n; ++i) {
        int idx = (h + i) % cap;
        if (q[idx] == slot) { found = i; break; }
    }
    if (found < 0) return;
    // shift subsequent elements left
    for (int i = found; i < n - 1; ++i) {
        int a = (h + i) % cap;
        int b = (h + i + 1) % cap;
        q[a] = q[b];
    }
    *len = n - 1;
}

// find live slot by request_id; returns slot or CPSR_NONE
__device__ uint32_t cpsr_find(const CpsrState* st, uint64_t rid) {
    for (int i = 0; i < st->cap; ++i) {
        if (st->r_live[i] && st->r_request_id[i] == rid) return (uint32_t)i;
    }
    return CPSR_NONE;
}
__device__ uint32_t cpsr_alloc(const CpsrState* st) {
    for (int i = 0; i < st->cap; ++i) if (!st->r_live[i]) return (uint32_t)i;
    return CPSR_NONE;
}
__device__ int cpsr_live_count(const CpsrState* st) {
    int c = 0; for (int i = 0; i < st->cap; ++i) if (st->r_live[i]) ++c; return c;
}
__device__ uint64_t cpsr_total_kv(const CpsrState* st) {
    uint64_t s = 0; for (int i = 0; i < st->cap; ++i) if (st->r_live[i]) s += st->r_kv_tokens[i];
    return s;
}
__device__ uint64_t cpsr_free_kv(const CpsrState* st) {
    uint64_t used = cpsr_total_kv(st);
    uint64_t cap = *st->kv_capacity;
    return (used >= cap) ? 0 : (cap - used);
}

// remove slot from whichever queue holds it
__device__ void cpsr_remove_from_queues(CpsrState* st, uint32_t slot) {
    cpsr_q_remove(st->prefill_q, st->prefill_head, st->prefill_len, st->cap, slot);
    cpsr_q_remove(st->decode_q, st->decode_head, st->decode_len, st->cap, slot);
}

// emit finalize event into finalize_hash; reads slot fields; reason==kind
__device__ void cpsr_emit_finalize(CpsrState* st, uint8_t kind, uint32_t op_index,
                                   uint32_t slot, uint64_t kv_freed) {
    uint64_t h = *st->finalize_hash;
    cpsr_u8(&h, kind);
    cpsr_u64(&h, *st->event_seq);
    cpsr_u32(&h, op_index);
    cpsr_u64(&h, st->r_request_id[slot]);
    cpsr_u32(&h, st->r_tenant[slot]);
    cpsr_u8(&h, st->r_priority[slot]);
    cpsr_u64(&h, st->r_prompt_done[slot]);
    cpsr_u64(&h, st->r_decode_done[slot]);
    cpsr_u64(&h, kv_freed);
    cpsr_u64(&h, st->r_moe_drop[slot]);
    cpsr_u8(&h, kind);
    *st->finalize_hash = h;
    *st->event_seq = *st->event_seq + 1;
}

// pick eviction victim (slot) or CPSR_NONE. use_excl: whether to honor protected
// flags and current slot. current==CPSR_NONE means no current.
__device__ uint32_t cpsr_pick_eviction(const CpsrState* st, int use_excl, uint32_t current) {
    int found = 0;
    uint32_t best = CPSR_NONE;
    for (int i = 0; i < st->cap; ++i) {
        if (!st->r_live[i]) continue;
        if (use_excl) {
            if ((uint32_t)i == current) continue;
            if (st->r_protected[i]) continue;
        }
        if (!found) { found = 1; best = (uint32_t)i; continue; }
        // compare slot i vs best
        uint8_t pi = st->r_priority[i], pb = st->r_priority[best];
        if (pi != pb) { if (pi < pb) best = (uint32_t)i; continue; }
        uint64_t ki = st->r_kv_tokens[i], kb = st->r_kv_tokens[best];
        if (ki != kb) { if (ki > kb) best = (uint32_t)i; continue; }
        uint64_t li = st->r_last_sched[i], lb = st->r_last_sched[best];
        if (li != lb) { if (li > lb) best = (uint32_t)i; continue; }
        uint64_t ai = st->r_arrival_seq[i], ab = st->r_arrival_seq[best];
        if (ai != ab) { if (ai < ab) best = (uint32_t)i; continue; }
        uint64_t ri = st->r_request_id[i], rb = st->r_request_id[best];
        if (ri < rb) best = (uint32_t)i;
    }
    return found ? best : CPSR_NONE;
}

__device__ void cpsr_evict_for_need(CpsrState* st, uint64_t needed_free,
                                    uint32_t current, uint32_t op_index) {
    while (cpsr_free_kv(st) < needed_free) {
        uint32_t v = cpsr_pick_eviction(st, 1, current);
        if (v == CPSR_NONE) break;
        uint64_t freed = st->r_kv_tokens[v];
        cpsr_remove_from_queues(st, v);
        cpsr_emit_finalize(st, CPS_EV_FINALIZE_EVICT, op_index, v, freed);
        st->r_live[v] = 0;
        st->counts->evicted += 1;
    }
}

// route one token; updates expert_remaining, moe_hash, counts, drop count.
__device__ void cpsr_moe_route(CpsrState* st, uint32_t slot, uint8_t phase_code,
                               uint64_t abs_token, uint64_t token_ordinal) {
    uint64_t h = CPSR_FNV_OFFSET;
    cpsr_u64(&h, st->spec.moe_seed);
    cpsr_u64(&h, *st->iter_seq);
    cpsr_u64(&h, st->r_request_id[slot]);
    cpsr_u32(&h, st->r_tenant[slot]);
    cpsr_u8(&h, phase_code);
    cpsr_u64(&h, abs_token);

    uint32_t ne = (uint32_t)st->ne;
    uint32_t primary = (uint32_t)(h % ne);
    uint32_t secondary;
    if (ne == 1) secondary = primary;
    else secondary = (uint32_t)((primary + 1 + ((h >> 32) % (ne - 1))) % ne);

    uint8_t route_kind; uint32_t assigned; uint64_t remaining_after;
    if (st->expert_remaining[primary] > 0) {
        st->expert_remaining[primary] -= 1;
        route_kind = CPS_ROUTE_PRIMARY; assigned = primary;
        remaining_after = st->expert_remaining[primary];
        st->counts->moe_primary += 1;
    } else if (secondary != primary && st->expert_remaining[secondary] > 0) {
        st->expert_remaining[secondary] -= 1;
        route_kind = CPS_ROUTE_SECONDARY; assigned = secondary;
        remaining_after = st->expert_remaining[secondary];
        st->counts->moe_secondary += 1;
    } else {
        route_kind = CPS_ROUTE_DROP; assigned = UINT32_MAX;
        remaining_after = UINT64_MAX;
        st->counts->moe_dropped += 1;
        st->r_moe_drop[slot] += 1;
    }

    uint64_t mh = *st->moe_hash;
    cpsr_u64(&mh, *st->iter_seq);
    cpsr_u64(&mh, token_ordinal);
    cpsr_u64(&mh, st->r_request_id[slot]);
    cpsr_u32(&mh, st->r_tenant[slot]);
    cpsr_u8(&mh, phase_code);
    cpsr_u64(&mh, abs_token);
    cpsr_u32(&mh, primary);
    cpsr_u32(&mh, secondary);
    cpsr_u8(&mh, route_kind);
    cpsr_u32(&mh, assigned);
    cpsr_u64(&mh, remaining_after);
    *st->moe_hash = mh;
}

// ---------- snapshot hashes ----------
__device__ uint64_t cpsr_queue_hash(const CpsrState* st) {
    uint64_t h = CPSR_FNV_OFFSET;
    int n = *st->prefill_len, hd = *st->prefill_head;
    for (int i = 0; i < n; ++i) {
        uint32_t slot = st->prefill_q[(hd + i) % st->cap];
        cpsr_u8(&h, CPS_Q_PREFILL);
        cpsr_u64(&h, (uint64_t)i);
        cpsr_u64(&h, st->r_request_id[slot]);
    }
    n = *st->decode_len; hd = *st->decode_head;
    for (int i = 0; i < n; ++i) {
        uint32_t slot = st->decode_q[(hd + i) % st->cap];
        cpsr_u8(&h, CPS_Q_DECODE);
        cpsr_u64(&h, (uint64_t)i);
        cpsr_u64(&h, st->r_request_id[slot]);
    }
    return h;
}

__device__ uint64_t cpsr_request_hash(const CpsrState* st) {
    // sort by tenant asc then request_id asc via selection over slots.
    // Use a "previous best" cursor to emit in sorted order without scratch.
    uint64_t h = CPSR_FNV_OFFSET;
    int live = cpsr_live_count(st);
    // emitted tracking via (tenant, request_id) strictly increasing key
    uint64_t prev_tenant = 0; uint64_t prev_rid = 0; int have_prev = 0;
    for (int emitted = 0; emitted < live; ++emitted) {
        int found = 0; uint32_t best = CPSR_NONE;
        for (int i = 0; i < st->cap; ++i) {
            if (!st->r_live[i]) continue;
            uint64_t ti = st->r_tenant[i];
            uint64_t ri = st->r_request_id[i];
            // must be strictly greater than prev key
            if (have_prev) {
                if (ti < prev_tenant) continue;
                if (ti == prev_tenant && ri <= prev_rid) continue;
            }
            if (!found) { found = 1; best = (uint32_t)i; continue; }
            uint64_t tb = st->r_tenant[best], rb = st->r_request_id[best];
            if (ti != tb) { if (ti < tb) best = (uint32_t)i; continue; }
            if (ri < rb) best = (uint32_t)i;
        }
        if (!found) break;
        uint32_t s = best;
        cpsr_u64(&h, st->r_request_id[s]);
        cpsr_u32(&h, st->r_tenant[s]);
        cpsr_u8(&h, st->r_priority[s]);
        cpsr_u8(&h, st->r_phase[s]);
        cpsr_u64(&h, st->r_prompt_len[s]);
        cpsr_u64(&h, st->r_prompt_done[s]);
        cpsr_u64(&h, st->r_decode_len[s]);
        cpsr_u64(&h, st->r_decode_done[s]);
        cpsr_u64(&h, st->r_chunk_max[s]);
        cpsr_u64(&h, st->r_kv_tokens[s]);
        cpsr_u64(&h, st->r_arrival_seq[s]);
        cpsr_u64(&h, st->r_last_sched[s]);
        cpsr_u64(&h, st->r_moe_drop[s]);
        prev_tenant = st->r_tenant[s];
        prev_rid = st->r_request_id[s];
        have_prev = 1;
    }
    return h;
}

__device__ uint64_t cpsr_bucket_hash(const CpsrState* st) {
    uint64_t h = CPSR_FNV_OFFSET;
    for (int t = 0; t < st->nt; ++t) {
        cpsr_u32(&h, (uint32_t)t);
        cpsr_u64(&h, st->bucket_tokens[t]);
        cpsr_u64(&h, st->bucket_cap[t]);
    }
    return h;
}

__device__ uint64_t cpsr_scalar_hash(const CpsrState* st) {
    uint64_t h = CPSR_FNV_OFFSET;
    cpsr_u64(&h, *st->iter_seq);
    cpsr_u64(&h, *st->event_seq);
    cpsr_u64(&h, *st->arrival_seq_next);
    cpsr_u64(&h, *st->kv_capacity);
    cpsr_u64(&h, (uint64_t)cpsr_live_count(st));
    cpsr_u64(&h, cpsr_total_kv(st));
    return h;
}

// ---------- operations ----------
__device__ void cpsr_op_arrive(CpsrState* st, const CpsOp* op) {
    if (op->tenant >= (uint32_t)st->nt) { st->counts->invalid_count += 1; return; }
    if (op->prompt_len == 0 && op->decode_len == 0) { st->counts->invalid_count += 1; return; }
    if (cpsr_find(st, op->request_id) != CPSR_NONE) { st->counts->invalid_count += 1; return; }
    uint64_t cm = (op->chunk_max != 0) ? op->chunk_max
                                       : (uint64_t)(uint32_t)st->spec.default_chunk_max;
    if (cm == 0) { st->counts->invalid_count += 1; return; }
    if (cpsr_live_count(st) >= st->cap) { st->counts->arrive_reject += 1; return; }

    uint32_t s = cpsr_alloc(st);
    st->r_live[s] = 1;
    st->r_request_id[s] = op->request_id;
    st->r_tenant[s] = op->tenant;
    st->r_priority[s] = (uint8_t)(op->priority & 0xFF);
    st->r_prompt_len[s] = op->prompt_len;
    st->r_prompt_done[s] = 0;
    st->r_decode_len[s] = op->decode_len;
    st->r_decode_done[s] = 0;
    st->r_chunk_max[s] = cm;
    st->r_kv_tokens[s] = 0;
    st->r_arrival_seq[s] = *st->arrival_seq_next;
    *st->arrival_seq_next = *st->arrival_seq_next + 1;
    st->r_last_sched[s] = UINT64_MAX;
    st->r_moe_drop[s] = 0;
    st->r_protected[s] = 0;

    if (op->prompt_len > 0) {
        st->r_phase[s] = CPS_PHASE_PREFILL;
        cpsr_q_push(st->prefill_q, st->prefill_head, st->prefill_len, st->cap, s);
    } else {
        st->r_phase[s] = CPS_PHASE_DECODE;
        cpsr_q_push(st->decode_q, st->decode_head, st->decode_len, st->cap, s);
    }
    st->counts->arrive_ok += 1;
}

__device__ void cpsr_op_refill(CpsrState* st, const CpsOp* op) {
    if (op->tenant >= (uint32_t)st->nt) { st->counts->invalid_count += 1; return; }
    uint32_t t = op->tenant;
    uint64_t v = cpsr_sat_add(st->bucket_tokens[t], op->a);
    if (v > st->bucket_cap[t]) v = st->bucket_cap[t];
    st->bucket_tokens[t] = v;
    st->counts->tenant_refills += 1;
}

__device__ void cpsr_op_set_kv_cap(CpsrState* st, const CpsOp* op) {
    *st->kv_capacity = op->a;
    while (cpsr_total_kv(st) > *st->kv_capacity) {
        if (cpsr_live_count(st) == 0) break;
        uint32_t v = cpsr_pick_eviction(st, 0, CPSR_NONE);
        if (v == CPSR_NONE) break;
        uint64_t freed = st->r_kv_tokens[v];
        cpsr_remove_from_queues(st, v);
        cpsr_emit_finalize(st, CPS_EV_EVICT_KV_SHRINK, (uint32_t)op->op_index, v, freed);
        st->r_live[v] = 0;
        st->counts->kv_shrink_evicted += 1;
    }
}

__device__ void cpsr_op_cancel(CpsrState* st, const CpsOp* op) {
    uint32_t s = cpsr_find(st, op->request_id);
    if (s == CPSR_NONE) { st->counts->invalid_count += 1; return; }
    uint64_t freed = st->r_kv_tokens[s];
    cpsr_remove_from_queues(st, s);
    cpsr_emit_finalize(st, CPS_EV_FINALIZE_CANCEL, (uint32_t)op->op_index, s, freed);
    st->r_live[s] = 0;
    st->counts->cancelled += 1;
}

__device__ void cpsr_op_step(CpsrState* st, const CpsOp* op) {
    st->counts->iterations += 1;

    uint64_t tokens_left = op->a;
    uint64_t slots_left = op->b;
    uint64_t max_slots = (uint64_t)(uint32_t)st->spec.max_batch_slots;
    if (slots_left > max_slots) slots_left = max_slots;

    for (int e = 0; e < st->ne; ++e) st->expert_remaining[e] = st->expert_capacity[e];
    for (int i = 0; i < st->cap; ++i) st->r_protected[i] = 0;

    if (tokens_left == 0 || slots_left == 0) {
        *st->iter_seq = *st->iter_seq + 1;
        return;
    }

    uint64_t slot_ordinal = 0;
    uint64_t token_ordinal = 0;
    uint32_t op_index = (uint32_t)op->op_index;

    // ---- DECODE PASS ----
    int decode_scan = *st->decode_len;
    for (int s = 0; s < decode_scan; ++s) {
        if (tokens_left == 0 || slots_left == 0) break;
        if (*st->decode_len == 0) break;
        uint32_t slot = cpsr_q_front(st->decode_q, *st->decode_head);
        cpsr_q_pop(st->decode_q, st->decode_head, st->decode_len, st->cap);
        if (!st->r_live[slot]) continue;
        uint32_t t = st->r_tenant[slot];
        if (st->bucket_tokens[t] == 0) {
            st->counts->throttle_skips += 1;
            cpsr_q_push(st->decode_q, st->decode_head, st->decode_len, st->cap, slot);
            continue;
        }
        if (cpsr_free_kv(st) == 0) {
            // current candidate excluded via `current` arg; already-scheduled
            // requests stay protected via their persistent r_protected flag.
            cpsr_evict_for_need(st, 1, slot, op_index);
        }
        if (cpsr_free_kv(st) == 0) {
            cpsr_q_push(st->decode_q, st->decode_head, st->decode_len, st->cap, slot);
            st->counts->kv_skips += 1;
            continue;
        }
        uint64_t bucket_before = st->bucket_tokens[t];
        uint64_t kv_before = st->r_kv_tokens[slot];
        uint64_t tokens_left_before = tokens_left;
        uint64_t abs_token = st->r_prompt_len[slot] + st->r_decode_done[slot];

        uint64_t bh = *st->batch_hash;
        cpsr_u8(&bh, CPS_REC_BATCH_DECODE);
        cpsr_u64(&bh, *st->iter_seq);
        cpsr_u32(&bh, (uint32_t)slot_ordinal);
        cpsr_u64(&bh, st->r_request_id[slot]);
        cpsr_u32(&bh, st->r_tenant[slot]);
        cpsr_u8(&bh, CPS_PHASE_DECODE);
        cpsr_u64(&bh, abs_token);
        cpsr_u64(&bh, (uint64_t)1);
        cpsr_u64(&bh, bucket_before);
        cpsr_u64(&bh, kv_before);
        cpsr_u64(&bh, tokens_left_before);
        *st->batch_hash = bh;
        st->counts->batch_decode_requests += 1;
        st->counts->decode_tokens += 1;

        cpsr_moe_route(st, slot, CPS_PHASE_DECODE, abs_token, token_ordinal);
        token_ordinal += 1;

        st->bucket_tokens[t] -= 1;
        tokens_left -= 1;
        slots_left -= 1;
        st->r_kv_tokens[slot] += 1;
        st->r_decode_done[slot] += 1;
        st->r_last_sched[slot] = *st->iter_seq;
        st->r_protected[slot] = 1;
        slot_ordinal += 1;

        if (st->r_decode_done[slot] == st->r_decode_len[slot]) {
            uint64_t freed = st->r_kv_tokens[slot];
            cpsr_emit_finalize(st, CPS_EV_FINALIZE_COMPLETE, op_index, slot, freed);
            st->r_live[slot] = 0;
            st->counts->completed += 1;
        } else {
            cpsr_q_push(st->decode_q, st->decode_head, st->decode_len, st->cap, slot);
        }
    }

    // ---- PREFILL PASS ----
    int prefill_scan = *st->prefill_len;
    for (int s = 0; s < prefill_scan; ++s) {
        if (tokens_left == 0 || slots_left == 0) break;
        if (*st->prefill_len == 0) break;
        uint32_t slot = cpsr_q_front(st->prefill_q, *st->prefill_head);
        cpsr_q_pop(st->prefill_q, st->prefill_head, st->prefill_len, st->cap);
        if (!st->r_live[slot]) continue;
        uint32_t t = st->r_tenant[slot];
        uint64_t tenant_tokens = st->bucket_tokens[t];
        if (tenant_tokens == 0) {
            st->counts->throttle_skips += 1;
            cpsr_q_push(st->prefill_q, st->prefill_head, st->prefill_len, st->cap, slot);
            continue;
        }
        uint64_t remaining_prompt = st->r_prompt_len[slot] - st->r_prompt_done[slot];
        uint64_t chunk = st->r_chunk_max[slot];
        if (remaining_prompt < chunk) chunk = remaining_prompt;
        if (tokens_left < chunk) chunk = tokens_left;
        if (tenant_tokens < chunk) chunk = tenant_tokens;

        if (cpsr_free_kv(st) < chunk) {
            // current candidate excluded via `current` arg; already-scheduled
            // requests stay protected via their persistent r_protected flag.
            cpsr_evict_for_need(st, chunk, slot, op_index);
            uint64_t f = cpsr_free_kv(st);
            if (f < chunk) chunk = f;
        }
        if (chunk == 0) {
            cpsr_q_push(st->prefill_q, st->prefill_head, st->prefill_len, st->cap, slot);
            st->counts->kv_skips += 1;
            continue;
        }

        uint64_t bucket_before = st->bucket_tokens[t];
        uint64_t kv_before = st->r_kv_tokens[slot];
        uint64_t tokens_left_before = tokens_left;
        uint64_t start_abs = st->r_prompt_done[slot];

        uint64_t bh = *st->batch_hash;
        cpsr_u8(&bh, CPS_REC_BATCH_PREFILL);
        cpsr_u64(&bh, *st->iter_seq);
        cpsr_u32(&bh, (uint32_t)slot_ordinal);
        cpsr_u64(&bh, st->r_request_id[slot]);
        cpsr_u32(&bh, st->r_tenant[slot]);
        cpsr_u8(&bh, CPS_PHASE_PREFILL);
        cpsr_u64(&bh, start_abs);
        cpsr_u64(&bh, chunk);
        cpsr_u64(&bh, bucket_before);
        cpsr_u64(&bh, kv_before);
        cpsr_u64(&bh, tokens_left_before);
        *st->batch_hash = bh;
        st->counts->batch_prefill_requests += 1;
        st->counts->prefill_tokens += chunk;

        for (uint64_t j = 0; j < chunk; ++j) {
            uint64_t abs_token = st->r_prompt_done[slot] + j;
            cpsr_moe_route(st, slot, CPS_PHASE_PREFILL, abs_token, token_ordinal);
            token_ordinal += 1;
        }

        st->r_prompt_done[slot] += chunk;
        st->r_kv_tokens[slot] += chunk;
        tokens_left -= chunk;
        st->bucket_tokens[t] -= chunk;
        slots_left -= 1;
        st->r_last_sched[slot] = *st->iter_seq;
        st->r_protected[slot] = 1;
        slot_ordinal += 1;

        if (st->r_prompt_done[slot] == st->r_prompt_len[slot] && st->r_decode_len[slot] > 0) {
            st->r_phase[slot] = CPS_PHASE_DECODE;
            cpsr_q_push(st->decode_q, st->decode_head, st->decode_len, st->cap, slot);
        } else if (st->r_prompt_done[slot] == st->r_prompt_len[slot] && st->r_decode_len[slot] == 0) {
            uint64_t freed = st->r_kv_tokens[slot];
            cpsr_emit_finalize(st, CPS_EV_FINALIZE_COMPLETE, op_index, slot, freed);
            st->r_live[slot] = 0;
            st->counts->completed += 1;
        } else {
            cpsr_q_push(st->prefill_q, st->prefill_head, st->prefill_len, st->cap, slot);
        }
    }

    *st->iter_seq = *st->iter_seq + 1;
}

__global__ void cpsr_reset_kernel(CpsrState st) {
    if (blockIdx.x || threadIdx.x) return;
    *st.iter_seq = 0;
    *st.event_seq = 0;
    *st.arrival_seq_next = 1;
    *st.kv_capacity = (uint64_t)(uint32_t)st.spec.kv_capacity_tokens;
    for (int t = 0; t < st.nt; ++t) {
        st.bucket_cap[t] = st.spec.bucket_cap[t];
        st.bucket_tokens[t] = st.spec.initial_bucket_tokens[t];
    }
    for (int e = 0; e < st.ne; ++e) st.expert_capacity[e] = st.spec.expert_capacity[e];
    for (int i = 0; i < st.cap; ++i) { st.r_live[i] = 0; st.r_protected[i] = 0; }
    *st.prefill_head = 0; *st.prefill_len = 0;
    *st.decode_head = 0; *st.decode_len = 0;
    memset(st.counts, 0, sizeof(CpsCounts));
    *st.batch_hash = CPSR_FNV_OFFSET;
    *st.moe_hash = CPSR_FNV_OFFSET;
    *st.finalize_hash = CPSR_FNV_OFFSET;
}

__global__ void cpsr_step_kernel(CpsrState st, CpsOp op, CpsOutputs out) {
    if (blockIdx.x || threadIdx.x) return;
    switch (op.opcode) {
        case CPS_OP_ARRIVE: cpsr_op_arrive(&st, &op); break;
        case CPS_OP_REFILL_TENANT: cpsr_op_refill(&st, &op); break;
        case CPS_OP_SET_KV_CAP: cpsr_op_set_kv_cap(&st, &op); break;
        case CPS_OP_CANCEL: cpsr_op_cancel(&st, &op); break;
        case CPS_OP_STEP_ITER: cpsr_op_step(&st, &op); break;
        default: st.counts->invalid_count += 1; break;
    }
    *out.counts = *st.counts;
    *out.batch_hash = *st.batch_hash;
    *out.moe_hash = *st.moe_hash;
    *out.finalize_hash = *st.finalize_hash;
    *out.queue_hash = cpsr_queue_hash(&st);
    *out.request_hash = cpsr_request_hash(&st);
    *out.bucket_hash = cpsr_bucket_hash(&st);
    *out.scalar_hash = cpsr_scalar_hash(&st);
}

// ---------- host glue ----------
static void cpsr_free_all(CpsrState* st) {
    void* ptrs[] = {
        st->iter_seq, st->event_seq, st->arrival_seq_next, st->kv_capacity,
        st->bucket_tokens, st->bucket_cap, st->expert_capacity, st->expert_remaining,
        st->r_live, st->r_request_id, st->r_tenant, st->r_priority,
        st->r_prompt_len, st->r_prompt_done, st->r_decode_len, st->r_decode_done,
        st->r_chunk_max, st->r_kv_tokens, st->r_arrival_seq, st->r_last_sched,
        st->r_moe_drop, st->r_phase, st->r_protected,
        st->prefill_q, st->prefill_head, st->prefill_len,
        st->decode_q, st->decode_head, st->decode_len,
        st->counts, st->batch_hash, st->moe_hash, st->finalize_hash
    };
    for (void* p : ptrs) if (p) cudaFree(p);
}

extern "C" size_t solution_workspace_bytes(const CpsProblemSpec* spec) {
    if (!cps_validate_problem_spec(spec)) return 0;
    return 0;  // contract permits a zero workspace
}

#define CPSR_MALLOC(field, bytes) \
    do { if (cudaMalloc((void**)&st->field, (bytes)) != cudaSuccess) goto fail; } while (0)

extern "C" cudaError_t solution_init(const CpsProblemSpec* spec, void** state_out, cudaStream_t stream) {
    if (!cps_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;
    CpsrState* st = (CpsrState*)malloc(sizeof(CpsrState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(CpsrState));
    memcpy(&st->spec, spec, sizeof(CpsProblemSpec));
    st->cap = spec->max_live_requests;
    st->nt = spec->num_tenants;
    st->ne = spec->num_experts;
    int cap = st->cap;

    CPSR_MALLOC(iter_seq, sizeof(uint64_t));
    CPSR_MALLOC(event_seq, sizeof(uint64_t));
    CPSR_MALLOC(arrival_seq_next, sizeof(uint64_t));
    CPSR_MALLOC(kv_capacity, sizeof(uint64_t));
    CPSR_MALLOC(bucket_tokens, sizeof(uint64_t) * st->nt);
    CPSR_MALLOC(bucket_cap, sizeof(uint64_t) * st->nt);
    CPSR_MALLOC(expert_capacity, sizeof(uint64_t) * st->ne);
    CPSR_MALLOC(expert_remaining, sizeof(uint64_t) * st->ne);
    CPSR_MALLOC(r_live, sizeof(uint8_t) * cap);
    CPSR_MALLOC(r_request_id, sizeof(uint64_t) * cap);
    CPSR_MALLOC(r_tenant, sizeof(uint32_t) * cap);
    CPSR_MALLOC(r_priority, sizeof(uint8_t) * cap);
    CPSR_MALLOC(r_prompt_len, sizeof(uint64_t) * cap);
    CPSR_MALLOC(r_prompt_done, sizeof(uint64_t) * cap);
    CPSR_MALLOC(r_decode_len, sizeof(uint64_t) * cap);
    CPSR_MALLOC(r_decode_done, sizeof(uint64_t) * cap);
    CPSR_MALLOC(r_chunk_max, sizeof(uint64_t) * cap);
    CPSR_MALLOC(r_kv_tokens, sizeof(uint64_t) * cap);
    CPSR_MALLOC(r_arrival_seq, sizeof(uint64_t) * cap);
    CPSR_MALLOC(r_last_sched, sizeof(uint64_t) * cap);
    CPSR_MALLOC(r_moe_drop, sizeof(uint64_t) * cap);
    CPSR_MALLOC(r_phase, sizeof(uint8_t) * cap);
    CPSR_MALLOC(r_protected, sizeof(uint8_t) * cap);
    CPSR_MALLOC(prefill_q, sizeof(uint32_t) * cap);
    CPSR_MALLOC(prefill_head, sizeof(int32_t));
    CPSR_MALLOC(prefill_len, sizeof(int32_t));
    CPSR_MALLOC(decode_q, sizeof(uint32_t) * cap);
    CPSR_MALLOC(decode_head, sizeof(int32_t));
    CPSR_MALLOC(decode_len, sizeof(int32_t));
    CPSR_MALLOC(counts, sizeof(CpsCounts));
    CPSR_MALLOC(batch_hash, sizeof(uint64_t));
    CPSR_MALLOC(moe_hash, sizeof(uint64_t));
    CPSR_MALLOC(finalize_hash, sizeof(uint64_t));

    cpsr_reset_kernel<<<1,1,0,stream>>>(*st);
    if (cudaPeekAtLastError() != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;
fail:
    cpsr_free_all(st);
    free(st);
    return cudaErrorMemoryAllocation;
}

extern "C" cudaError_t solution_run(void* state, const CpsOp* op, const void* inputs,
                                    void* outputs_void, void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    (void)inputs; (void)workspace; (void)workspace_bytes;
    if (!state || !op || !outputs_void) return cudaErrorInvalidValue;
    CpsrState* st = (CpsrState*)state;
    if (!cps_validate_op(op, &st->spec)) return cudaErrorInvalidValue;
    CpsOutputs* out = (CpsOutputs*)outputs_void;
    if (!out->counts || !out->batch_hash || !out->moe_hash || !out->finalize_hash ||
        !out->queue_hash || !out->request_hash || !out->bucket_hash || !out->scalar_hash)
        return cudaErrorInvalidValue;

    cpsr_step_kernel<<<1,1,0,stream>>>(*st, *op, *out);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    CpsrState* st = (CpsrState*)state;
    cpsr_reset_kernel<<<1,1,0,stream>>>(*st);
    return cudaPeekAtLastError();
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    CpsrState* st = (CpsrState*)state;
    cpsr_free_all(st);
    free(st);
}
