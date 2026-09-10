// PMPP_CANARY_40_f5f1307807 -- held-out canary; MUST NOT appear in any submission
// file: switch_moe_overflow_router_reference.cu
//
// Reference GPU implementation. Single-thread serial kernel that processes the
// operation batch in input order. Uses direct-indexed token-table arrays and an
// array-backed FIFO. This is an INDEPENDENT implementation from naive.cu and the
// oracle: no shared algorithm code.

#include "switch_moe_overflow_router_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct SmorRefState {
    SmorProblemSpec spec;

    // persistent config
    uint64_t* credit_cap;       // E
    uint64_t* initial_credit;   // E

    // persistent state
    uint64_t* credit;           // E
    uint64_t* live_count;       // E
    uint64_t* event_seq;        // 1
    uint64_t* step_seq;         // 1
    int32_t*  live_total;       // 1

    // token table, direct-indexed by token_id in [0, token_space)
    int32_t*  tok_status;       // token_space
    uint32_t* tok_primary;      // token_space
    uint32_t* tok_secondary;    // token_space
    uint64_t* tok_cost;         // token_space
    uint64_t* tok_arrival;      // token_space
    uint64_t* tok_admit;        // token_space
    uint32_t* tok_assigned;     // token_space
    int32_t*  tok_route_kind;   // token_space

    // overflow FIFO as a dynamic array of token_ids (head..tail), capacity
    // overflow_capacity. Stored as compact list with head index.
    uint64_t* fifo;             // overflow_capacity (+1 guard slot)
    int32_t*  fifo_head;        // 1
    int32_t*  fifo_len;         // 1

    // scratch for assignment sort: collected live token ids (max_live_tokens)
    uint64_t* sort_ids;         // max_live_tokens
    // parallel-sort scratch: compact keys gathered during the live scan and the
    // rank-sorted output permutation (all sized max_live_tokens).
    uint32_t* sort_assigned;    // max_live_tokens
    uint64_t* sort_admit;       // max_live_tokens
    uint64_t* sorted_ids;       // max_live_tokens
};

// ---- device FNV ----
__device__ __forceinline__ uint64_t smr_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= SMOR_FNV_PRIME;
    return h;
}
__device__ void smr_fnv(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = smr_fnv_byte(v, b[i]);
    *h = v;
}
__device__ __forceinline__ uint64_t smr_sat_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    return (s < a) ? SMOR_U64_MAX : s;
}

// emit one event into the route_event_hash, advancing event_seq.
__device__ void smr_emit(uint64_t* hash, uint64_t* event_seq,
                         uint8_t kind, uint32_t op_index, uint64_t token_id,
                         uint32_t expert_or_max, uint32_t primary, uint32_t secondary,
                         uint64_t cost, uint64_t credit_after, uint64_t arrival) {
    uint64_t es = *event_seq;
    uint64_t h = *hash;
    smr_fnv(&h, &kind, sizeof(uint8_t));
    smr_fnv(&h, &es, sizeof(uint64_t));
    smr_fnv(&h, &op_index, sizeof(uint32_t));
    smr_fnv(&h, &token_id, sizeof(uint64_t));
    smr_fnv(&h, &expert_or_max, sizeof(uint32_t));
    smr_fnv(&h, &primary, sizeof(uint32_t));
    smr_fnv(&h, &secondary, sizeof(uint32_t));
    smr_fnv(&h, &cost, sizeof(uint64_t));
    smr_fnv(&h, &credit_after, sizeof(uint64_t));
    smr_fnv(&h, &arrival, sizeof(uint64_t));
    *hash = h;
    *event_seq = es + 1;
}

// resolve candidates: returns 1 with primary/secondary, or 0 if invalid.
__device__ int smr_resolve(int num_experts, int off, int cnt,
                           const int32_t* cand_expert, const int32_t* cand_logit,
                           const int32_t* cand_ordinal,
                           uint32_t* out_primary, uint32_t* out_secondary) {
    // collapse-by-expert into a small local array (<= SMOR_MAX_CANDS)
    int32_t ue[SMOR_MAX_CANDS];
    int32_t ul[SMOR_MAX_CANDS];
    int32_t uo[SMOR_MAX_CANDS];
    int un = 0;
    for (int i = 0; i < cnt; ++i) {
        int32_t e = cand_expert[off + i];
        int32_t lg = cand_logit[off + i];
        int32_t ord = cand_ordinal[off + i];
        if (e < 0 || e >= num_experts) continue;
        int found = -1;
        for (int j = 0; j < un; ++j) { if (ue[j] == e) { found = j; break; } }
        if (found >= 0) {
            if (lg > ul[found] || (lg == ul[found] && ord < uo[found])) {
                ul[found] = lg; uo[found] = ord;
            }
        } else {
            ue[un] = e; ul[un] = lg; uo[un] = ord; un++;
        }
    }
    if (un == 0) return 0;
    // selection sort by descending logit, then ascending expert
    for (int a = 0; a < un; ++a) {
        int best = a;
        for (int b = a + 1; b < un; ++b) {
            if (ul[b] > ul[best] || (ul[b] == ul[best] && ue[b] < ue[best])) best = b;
        }
        if (best != a) {
            int32_t te = ue[a]; ue[a] = ue[best]; ue[best] = te;
            int32_t tl = ul[a]; ul[a] = ul[best]; ul[best] = tl;
            int32_t to = uo[a]; uo[a] = uo[best]; uo[best] = to;
        }
    }
    *out_primary = (uint32_t)ue[0];
    *out_secondary = (un >= 2) ? (uint32_t)ue[1] : (uint32_t)ue[0];
    return 1;
}

__global__ void smr_step_kernel(
    int num_experts, int max_live, int overflow_cap, int max_cands, int token_space,
    int batch_size,
    const int32_t* __restrict__ op_kind,
    const uint64_t* __restrict__ op_a,
    const uint64_t* __restrict__ op_b,
    const int32_t* __restrict__ op_cand_off,
    const int32_t* __restrict__ op_cand_count,
    const int32_t* __restrict__ cand_expert,
    const int32_t* __restrict__ cand_logit,
    const int32_t* __restrict__ cand_ordinal,
    const uint64_t* __restrict__ credit_cap,
    uint64_t* __restrict__ credit,
    uint64_t* __restrict__ live_count,
    uint64_t* __restrict__ event_seq_p,
    uint64_t* __restrict__ step_seq_p,
    int32_t* __restrict__ live_total_p,
    int32_t* __restrict__ tok_status,
    uint32_t* __restrict__ tok_primary,
    uint32_t* __restrict__ tok_secondary,
    uint64_t* __restrict__ tok_cost,
    uint64_t* __restrict__ tok_arrival,
    uint64_t* __restrict__ tok_admit,
    uint32_t* __restrict__ tok_assigned,
    int32_t* __restrict__ tok_route_kind,
    uint64_t* __restrict__ fifo,
    int32_t* __restrict__ fifo_head_p,
    int32_t* __restrict__ fifo_len_p,
    uint64_t* __restrict__ sort_ids,
    uint32_t* __restrict__ sort_assigned,
    uint64_t* __restrict__ sort_admit,
    uint64_t* __restrict__ sorted_ids,
    uint64_t* __restrict__ out_counts,   // 13 counts
    uint64_t* __restrict__ out_route_hash,
    uint64_t* __restrict__ out_credit_hash,
    uint64_t* __restrict__ out_assignment_hash,
    uint64_t* __restrict__ out_overflow_hash) {
    // One block, blockDim.x threads. Thread 0 runs the inherently-serial state
    // machine + route/credit/overflow hashes; the whole block cooperates on the
    // O(token_space) live scan and the assignment-hash sort.
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    __shared__ int s_ncoll;

    if (tid == 0) {

    uint64_t event_seq = *event_seq_p;
    int live_total = *live_total_p;
    int fifo_head = *fifo_head_p;
    int fifo_len = *fifo_len_p;

    // counts
    uint64_t c_refill = 0, c_ap = 0, c_as = 0, c_q = 0, c_rp = 0, c_rs = 0;
    uint64_t c_capdrop = 0, c_oom = 0, c_dup = 0, c_rlive = 0, c_rq = 0, c_qdrop = 0, c_inv = 0;
    uint64_t route_hash = SMOR_FNV_OFFSET;

    for (int i = 0; i < batch_size; ++i) {
        int kind = op_kind[i];

        if (kind == SMOR_OP_REFILL) {
            uint64_t expert = op_a[i];
            uint64_t amount = op_b[i];
            if (expert >= (uint64_t)num_experts) {
                c_inv += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_INVALID, (uint32_t)i, 0,
                         SMOR_U32_MAX, SMOR_U32_MAX, SMOR_U32_MAX, 0, SMOR_U64_MAX, SMOR_U64_MAX);
            } else {
                uint64_t nc = smr_sat_add(credit[expert], amount);
                if (nc > credit_cap[expert]) nc = credit_cap[expert];
                credit[expert] = nc;
                c_refill += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_REFILL, (uint32_t)i, 0,
                         (uint32_t)expert, SMOR_U32_MAX, SMOR_U32_MAX, 0, nc, SMOR_U64_MAX);
            }
        } else if (kind == SMOR_OP_ROUTE) {
            uint64_t token_id = op_a[i];
            uint64_t cost = op_b[i];
            int cand_count = op_cand_count[i];
            if (cost == 0 || cand_count == 0 || cand_count > max_cands) {
                c_inv += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_INVALID, (uint32_t)i, token_id,
                         SMOR_U32_MAX, SMOR_U32_MAX, SMOR_U32_MAX, cost, SMOR_U64_MAX, SMOR_U64_MAX);
                continue;
            }
            uint32_t primary, secondary;
            int ok = smr_resolve(num_experts, op_cand_off[i], cand_count,
                                 cand_expert, cand_logit, cand_ordinal, &primary, &secondary);
            if (!ok) {
                c_inv += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_INVALID, (uint32_t)i, token_id,
                         SMOR_U32_MAX, SMOR_U32_MAX, SMOR_U32_MAX, cost, SMOR_U64_MAX, SMOR_U64_MAX);
                continue;
            }
            // duplicate
            if (tok_status[token_id] != SMOR_STATUS_FREE) {
                c_dup += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_DUPLICATE, (uint32_t)i, token_id,
                         SMOR_U32_MAX, primary, secondary, cost, SMOR_U64_MAX, SMOR_U64_MAX);
                continue;
            }
            // choose admit
            int chosen = -1; int is_primary = 0;
            if (credit[primary] >= cost) { chosen = (int)primary; is_primary = 1; }
            else if (secondary != primary && credit[secondary] >= cost) { chosen = (int)secondary; is_primary = 0; }

            if (chosen >= 0) {
                if (live_total == max_live) {
                    c_oom += 1;
                    smr_emit(&route_hash, &event_seq, SMOR_EV_OOM_DROP, (uint32_t)i, token_id,
                             (uint32_t)chosen, primary, secondary, cost, SMOR_U64_MAX, SMOR_U64_MAX);
                    continue;
                }
                credit[chosen] -= cost;
                tok_status[token_id] = SMOR_STATUS_LIVE;
                tok_primary[token_id] = primary;
                tok_secondary[token_id] = secondary;
                tok_cost[token_id] = cost;
                tok_assigned[token_id] = (uint32_t)chosen;
                tok_admit[token_id] = event_seq;
                tok_route_kind[token_id] = is_primary ? SMOR_RK_PRIMARY : SMOR_RK_SECONDARY;
                live_count[chosen] += 1;
                live_total += 1;
                if (is_primary) c_ap += 1; else c_as += 1;
                smr_emit(&route_hash, &event_seq,
                         is_primary ? SMOR_EV_ACCEPT_PRIMARY : SMOR_EV_ACCEPT_SECONDARY,
                         (uint32_t)i, token_id, (uint32_t)chosen, primary, secondary,
                         cost, credit[chosen], SMOR_U64_MAX);
                continue;
            }
            // admission failed
            int queue_room = (fifo_len < overflow_cap);
            int table_room = (live_total + fifo_len < max_live + overflow_cap);
            if (queue_room && table_room) {
                tok_status[token_id] = SMOR_STATUS_QUEUED;
                tok_primary[token_id] = primary;
                tok_secondary[token_id] = secondary;
                tok_cost[token_id] = cost;
                tok_arrival[token_id] = event_seq;
                int tail = (fifo_head + fifo_len) % (overflow_cap > 0 ? overflow_cap : 1);
                fifo[tail] = token_id;
                fifo_len += 1;
                c_q += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_QUEUE, (uint32_t)i, token_id,
                         SMOR_U32_MAX, primary, secondary, cost, SMOR_U64_MAX, tok_arrival[token_id]);
            } else if (!queue_room) {
                c_capdrop += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_CAPACITY_DROP, (uint32_t)i, token_id,
                         SMOR_U32_MAX, primary, secondary, cost, SMOR_U64_MAX, SMOR_U64_MAX);
            } else {
                c_oom += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_OOM_DROP, (uint32_t)i, token_id,
                         SMOR_U32_MAX, primary, secondary, cost, SMOR_U64_MAX, SMOR_U64_MAX);
            }
        } else if (kind == SMOR_OP_DRAIN) {
            uint64_t limit = op_a[i];
            int mod = (overflow_cap > 0 ? overflow_cap : 1);
            while (limit > 0 && fifo_len > 0) {
                uint64_t head_id = fifo[fifo_head];
                uint32_t pr = tok_primary[head_id];
                uint32_t sc = tok_secondary[head_id];
                uint64_t cst = tok_cost[head_id];
                int chosen = -1; int is_primary = 0;
                if (credit[pr] >= cst) { chosen = (int)pr; is_primary = 1; }
                else if (sc != pr && credit[sc] >= cst) { chosen = (int)sc; is_primary = 0; }
                if (chosen < 0) break;  // head-of-line block

                if (live_total == max_live) {
                    // capacity exceed: pop, remove, OOM_DROP_REPLAY
                    fifo_head = (fifo_head + 1) % mod;
                    fifo_len -= 1;
                    tok_status[head_id] = SMOR_STATUS_FREE;
                    smr_emit(&route_hash, &event_seq, SMOR_EV_OOM_DROP_REPLAY, (uint32_t)i, head_id,
                             SMOR_U32_MAX, pr, sc, cst, SMOR_U64_MAX, SMOR_U64_MAX);
                    limit -= 1;
                    continue;
                }
                // accept replay
                fifo_head = (fifo_head + 1) % mod;
                fifo_len -= 1;
                credit[chosen] -= cst;
                tok_status[head_id] = SMOR_STATUS_LIVE;
                tok_assigned[head_id] = (uint32_t)chosen;
                tok_admit[head_id] = event_seq;
                tok_route_kind[head_id] = is_primary ? SMOR_RK_REPLAY_PRIMARY : SMOR_RK_REPLAY_SECONDARY;
                live_count[chosen] += 1;
                live_total += 1;
                if (is_primary) c_rp += 1; else c_rs += 1;
                smr_emit(&route_hash, &event_seq,
                         is_primary ? SMOR_EV_REPLAY_PRIMARY : SMOR_EV_REPLAY_SECONDARY,
                         (uint32_t)i, head_id, (uint32_t)chosen, pr, sc, cst,
                         credit[chosen], SMOR_U64_MAX);
                limit -= 1;
            }
        } else if (kind == SMOR_OP_RETIRE) {
            uint64_t token_id = op_a[i];
            int st = tok_status[token_id];
            if (st == SMOR_STATUS_FREE) {
                c_inv += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_INVALID, (uint32_t)i, token_id,
                         SMOR_U32_MAX, SMOR_U32_MAX, SMOR_U32_MAX, 0, SMOR_U64_MAX, SMOR_U64_MAX);
            } else if (st == SMOR_STATUS_LIVE) {
                uint32_t assigned = tok_assigned[token_id];
                uint32_t pr = tok_primary[token_id];
                uint32_t sc = tok_secondary[token_id];
                uint64_t cst = tok_cost[token_id];
                live_count[assigned] -= 1;
                live_total -= 1;
                tok_status[token_id] = SMOR_STATUS_FREE;
                c_rlive += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_RETIRE_LIVE, (uint32_t)i, token_id,
                         assigned, pr, sc, cst, SMOR_U64_MAX, SMOR_U64_MAX);
            } else {  // QUEUED
                uint32_t pr = tok_primary[token_id];
                uint32_t sc = tok_secondary[token_id];
                uint64_t cst = tok_cost[token_id];
                uint64_t arr = tok_arrival[token_id];
                // remove from FIFO by compaction
                int mod = (overflow_cap > 0 ? overflow_cap : 1);
                int found = -1;
                for (int k = 0; k < fifo_len; ++k) {
                    int idx = (fifo_head + k) % mod;
                    if (fifo[idx] == token_id) { found = k; break; }
                }
                if (found >= 0) {
                    for (int k = found; k < fifo_len - 1; ++k) {
                        int a = (fifo_head + k) % mod;
                        int b = (fifo_head + k + 1) % mod;
                        fifo[a] = fifo[b];
                    }
                    fifo_len -= 1;
                }
                tok_status[token_id] = SMOR_STATUS_FREE;
                c_rq += 1;
                smr_emit(&route_hash, &event_seq, SMOR_EV_RETIRE_QUEUED, (uint32_t)i, token_id,
                         SMOR_U32_MAX, pr, sc, cst, SMOR_U64_MAX, arr);
            }
        } else if (kind == SMOR_OP_DROP_QUEUED_THROUGH) {
            uint64_t cutoff = op_a[i];
            int mod = (overflow_cap > 0 ? overflow_cap : 1);
            while (fifo_len > 0) {
                uint64_t head_id = fifo[fifo_head];
                uint64_t arr = tok_arrival[head_id];
                if (arr <= cutoff) {
                    uint32_t pr = tok_primary[head_id];
                    uint32_t sc = tok_secondary[head_id];
                    uint64_t cst = tok_cost[head_id];
                    fifo_head = (fifo_head + 1) % mod;
                    fifo_len -= 1;
                    tok_status[head_id] = SMOR_STATUS_FREE;
                    c_qdrop += 1;
                    smr_emit(&route_hash, &event_seq, SMOR_EV_QUEUE_DROP, (uint32_t)i, head_id,
                             SMOR_U32_MAX, pr, sc, cst, SMOR_U64_MAX, arr);
                } else break;
            }
        } else {
            c_inv += 1;
            smr_emit(&route_hash, &event_seq, SMOR_EV_INVALID, (uint32_t)i, 0,
                     SMOR_U32_MAX, SMOR_U32_MAX, SMOR_U32_MAX, 0, SMOR_U64_MAX, SMOR_U64_MAX);
        }
    }

    *event_seq_p = event_seq;
    *live_total_p = live_total;
    *fifo_head_p = fifo_head;
    *fifo_len_p = fifo_len;
    *step_seq_p = *step_seq_p + 1;

    // ---- credit_hash ----
    uint64_t ch = SMOR_FNV_OFFSET;
    for (int e = 0; e < num_experts; ++e) {
        uint32_t eid = (uint32_t)e;
        uint64_t cr = credit[e];
        uint64_t lc = live_count[e];
        smr_fnv(&ch, &eid, sizeof(uint32_t));
        smr_fnv(&ch, &cr, sizeof(uint64_t));
        smr_fnv(&ch, &lc, sizeof(uint64_t));
    }

    // ---- overflow_hash : head -> tail ----
    uint64_t oh = SMOR_FNV_OFFSET;
    int mod = (overflow_cap > 0 ? overflow_cap : 1);
    for (int k = 0; k < fifo_len; ++k) {
        int idx = (fifo_head + k) % mod;
        uint64_t tid = fifo[idx];
        uint64_t arr = tok_arrival[tid];
        uint32_t pr = tok_primary[tid];
        uint32_t sc = tok_secondary[tid];
        uint64_t cst = tok_cost[tid];
        smr_fnv(&oh, &tid, sizeof(uint64_t));
        smr_fnv(&oh, &arr, sizeof(uint64_t));
        smr_fnv(&oh, &pr, sizeof(uint32_t));
        smr_fnv(&oh, &sc, sizeof(uint32_t));
        smr_fnv(&oh, &cst, sizeof(uint64_t));
    }

    out_counts[0] = c_refill;
    out_counts[1] = c_ap;
    out_counts[2] = c_as;
    out_counts[3] = c_q;
    out_counts[4] = c_rp;
    out_counts[5] = c_rs;
    out_counts[6] = c_capdrop;
    out_counts[7] = c_oom;
    out_counts[8] = c_dup;
    out_counts[9] = c_rlive;
    out_counts[10] = c_rq;
    out_counts[11] = c_qdrop;
    out_counts[12] = c_inv;
    *out_route_hash = route_hash;
    *out_credit_hash = ch;
    *out_overflow_hash = oh;

    }  // end thread-0 serial section

    // ---- assignment_hash (block-cooperative) --------------------------------
    // POST-step LIVE-token set, sorted by (assigned_expert, admit_seq, token_id)
    // -- a strict total order (token_id is unique). Parallelizing the O(token_
    // space) live scan and using a rank-sort (each element's rank = # of
    // strictly-smaller elements) yields the identical permutation the serial
    // insertion sort produced, so the folded bytes are byte-for-byte identical.
    if (tid == 0) s_ncoll = 0;
    __syncthreads();

    // Parallel scan: gather LIVE token_ids and their sort keys compactly.
    for (int t = tid; t < token_space; t += nthreads) {
        if (tok_status[t] == SMOR_STATUS_LIVE) {
            int pos = atomicAdd(&s_ncoll, 1);
            sort_ids[pos]      = (uint64_t)t;
            sort_assigned[pos] = tok_assigned[t];
            sort_admit[pos]    = tok_admit[t];
        }
    }
    __syncthreads();
    const int ncoll = s_ncoll;

    // Rank-sort: place each collected token at its total-order rank.
    for (int i = tid; i < ncoll; i += nthreads) {
        uint64_t ti = sort_ids[i];
        uint32_t ei = sort_assigned[i];
        uint64_t ai = sort_admit[i];
        int rank = 0;
        for (int j = 0; j < ncoll; ++j) {
            uint32_t ej = sort_assigned[j];
            uint64_t aj = sort_admit[j];
            uint64_t tj = sort_ids[j];
            int less = (ej < ei) ||
                       (ej == ei && aj < ai) ||
                       (ej == ei && aj == ai && tj < ti);
            rank += less;
        }
        sorted_ids[rank] = ti;
    }
    __syncthreads();

    // Serial fold in sorted order (<= max_live records; token_table lookups).
    if (tid == 0) {
        uint64_t ah = SMOR_FNV_OFFSET;
        for (int k = 0; k < ncoll; ++k) {
            uint64_t tid2 = sorted_ids[k];
            uint32_t expert = tok_assigned[tid2];
            uint64_t admit = tok_admit[tid2];
            uint32_t pr = tok_primary[tid2];
            uint32_t sc = tok_secondary[tid2];
            uint64_t cst = tok_cost[tid2];
            uint8_t rk = (uint8_t)tok_route_kind[tid2];
            smr_fnv(&ah, &expert, sizeof(uint32_t));
            smr_fnv(&ah, &tid2, sizeof(uint64_t));
            smr_fnv(&ah, &admit, sizeof(uint64_t));
            smr_fnv(&ah, &pr, sizeof(uint32_t));
            smr_fnv(&ah, &sc, sizeof(uint32_t));
            smr_fnv(&ah, &cst, sizeof(uint64_t));
            smr_fnv(&ah, &rk, sizeof(uint8_t));
        }
        *out_assignment_hash = ah;
    }
}

// ---- reset kernel : credit[e] = initial_credit[e], everything else zero ----
__global__ void smr_reset_kernel(int num_experts, int token_space,
                                 const uint64_t* __restrict__ initial_credit,
                                 uint64_t* __restrict__ credit,
                                 uint64_t* __restrict__ live_count,
                                 int32_t* __restrict__ tok_status) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int e = idx; e < num_experts; e += stride) {
        credit[e] = initial_credit[e];
        live_count[e] = 0;
    }
    for (int t = idx; t < token_space; t += stride) {
        tok_status[t] = SMOR_STATUS_FREE;
    }
}

static cudaError_t smr_reset_state(SmorRefState* st, cudaStream_t stream) {
    cudaError_t err;
    err = cudaMemsetAsync(st->event_seq, 0, sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->step_seq, 0, sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->live_total, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->fifo_head, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->fifo_len, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) return err;

    int threads = 256;
    int blocks = (int)((size_t)(st->spec.token_space > st->spec.num_experts
                                 ? st->spec.token_space : st->spec.num_experts) + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    smr_reset_kernel<<<blocks, threads, 0, stream>>>(
        st->spec.num_experts, st->spec.token_space,
        st->initial_credit, st->credit, st->live_count, st->tok_status);
    return cudaPeekAtLastError();
}

extern "C" size_t solution_workspace_bytes(const SmorProblemSpec* spec) {
    if (!smor_validate_problem_spec(spec)) return 0;
    return 256;
}

extern "C" cudaError_t solution_init(
    const SmorProblemSpec* spec,
    const SmorInitConfig* config,
    void** state_out,
    cudaStream_t stream) {
    if (!smor_validate_problem_spec(spec) || !state_out || !config) {
        return cudaErrorInvalidValue;
    }
    if (!config->credit_cap || !config->initial_credit) return cudaErrorInvalidValue;

    SmorRefState* st = (SmorRefState*)malloc(sizeof(SmorRefState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(SmorRefState));
    memcpy(&st->spec, spec, sizeof(SmorProblemSpec));

    const size_t E = (size_t)spec->num_experts;
    const size_t TS = (size_t)spec->token_space;
    const size_t OF = (size_t)spec->overflow_capacity;
    const size_t ML = (size_t)spec->max_live_tokens;

    cudaError_t err = cudaSuccess;
    #define ALLOC(p, bytes) do { err = cudaMalloc((void**)&(p), (bytes)); if (err != cudaSuccess) goto fail; } while (0)
    ALLOC(st->credit_cap, sizeof(uint64_t) * E);
    ALLOC(st->initial_credit, sizeof(uint64_t) * E);
    ALLOC(st->credit, sizeof(uint64_t) * E);
    ALLOC(st->live_count, sizeof(uint64_t) * E);
    ALLOC(st->event_seq, sizeof(uint64_t));
    ALLOC(st->step_seq, sizeof(uint64_t));
    ALLOC(st->live_total, sizeof(int32_t));
    ALLOC(st->tok_status, sizeof(int32_t) * TS);
    ALLOC(st->tok_primary, sizeof(uint32_t) * TS);
    ALLOC(st->tok_secondary, sizeof(uint32_t) * TS);
    ALLOC(st->tok_cost, sizeof(uint64_t) * TS);
    ALLOC(st->tok_arrival, sizeof(uint64_t) * TS);
    ALLOC(st->tok_admit, sizeof(uint64_t) * TS);
    ALLOC(st->tok_assigned, sizeof(uint32_t) * TS);
    ALLOC(st->tok_route_kind, sizeof(int32_t) * TS);
    ALLOC(st->fifo, sizeof(uint64_t) * (OF + 1));
    ALLOC(st->fifo_head, sizeof(int32_t));
    ALLOC(st->fifo_len, sizeof(int32_t));
    ALLOC(st->sort_ids, sizeof(uint64_t) * (ML > 0 ? ML : 1));
    ALLOC(st->sort_assigned, sizeof(uint32_t) * (ML > 0 ? ML : 1));
    ALLOC(st->sort_admit, sizeof(uint64_t) * (ML > 0 ? ML : 1));
    ALLOC(st->sorted_ids, sizeof(uint64_t) * (ML > 0 ? ML : 1));
    #undef ALLOC

    err = cudaMemcpyAsync(st->credit_cap, config->credit_cap, sizeof(uint64_t) * E,
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpyAsync(st->initial_credit, config->initial_credit, sizeof(uint64_t) * E,
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto fail;

    err = smr_reset_state(st, stream);
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    solution_destroy(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const SmorRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;
    if (!state || !outputs_void) return cudaErrorInvalidValue;
    if (workspace_bytes < 256) return cudaErrorInvalidValue;

    SmorRefState* st = (SmorRefState*)state;
    if (!smor_validate_run_spec(run, &st->spec)) return cudaErrorInvalidValue;

    const SmorInputs* in = (const SmorInputs*)inputs_void;
    SmorOutputs* out = (SmorOutputs*)outputs_void;
    if (run->batch_size > 0 && !in) return cudaErrorInvalidValue;

    // The kernel writes 13 counts into a contiguous device buffer; use the
    // provided workspace (13 * 8 = 104 bytes <= 256), then scatter to scalars.
    uint64_t* counts = (uint64_t*)workspace;

    smr_step_kernel<<<1, 256, 0, stream>>>(
        st->spec.num_experts, st->spec.max_live_tokens, st->spec.overflow_capacity,
        st->spec.max_candidates_per_route, st->spec.token_space,
        run->batch_size,
        in ? in->op_kind : nullptr,
        in ? in->op_a : nullptr,
        in ? in->op_b : nullptr,
        in ? in->op_cand_off : nullptr,
        in ? in->op_cand_count : nullptr,
        in ? in->cand_expert : nullptr,
        in ? in->cand_logit : nullptr,
        in ? in->cand_ordinal : nullptr,
        st->credit_cap,
        st->credit, st->live_count, st->event_seq, st->step_seq, st->live_total,
        st->tok_status, st->tok_primary, st->tok_secondary, st->tok_cost,
        st->tok_arrival, st->tok_admit, st->tok_assigned, st->tok_route_kind,
        st->fifo, st->fifo_head, st->fifo_len, st->sort_ids,
        st->sort_assigned, st->sort_admit, st->sorted_ids,
        counts,
        out->route_event_hash, out->credit_hash, out->assignment_hash, out->overflow_hash);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    // scatter counts to individual output scalars
    err = cudaMemcpyAsync(out->refill_count, counts + 0, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->accepted_primary, counts + 1, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->accepted_secondary, counts + 2, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->queued, counts + 3, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->replayed_primary, counts + 4, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->replayed_secondary, counts + 5, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->capacity_drop, counts + 6, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->oom_drop, counts + 7, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->duplicate_count, counts + 8, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->retired_live, counts + 9, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->retired_queued, counts + 10, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->queue_drop, counts + 11, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(out->invalid_count, counts + 12, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return smr_reset_state((SmorRefState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    SmorRefState* st = (SmorRefState*)state;
    if (st->credit_cap) cudaFree(st->credit_cap);
    if (st->initial_credit) cudaFree(st->initial_credit);
    if (st->credit) cudaFree(st->credit);
    if (st->live_count) cudaFree(st->live_count);
    if (st->event_seq) cudaFree(st->event_seq);
    if (st->step_seq) cudaFree(st->step_seq);
    if (st->live_total) cudaFree(st->live_total);
    if (st->tok_status) cudaFree(st->tok_status);
    if (st->tok_primary) cudaFree(st->tok_primary);
    if (st->tok_secondary) cudaFree(st->tok_secondary);
    if (st->tok_cost) cudaFree(st->tok_cost);
    if (st->tok_arrival) cudaFree(st->tok_arrival);
    if (st->tok_admit) cudaFree(st->tok_admit);
    if (st->tok_assigned) cudaFree(st->tok_assigned);
    if (st->tok_route_kind) cudaFree(st->tok_route_kind);
    if (st->fifo) cudaFree(st->fifo);
    if (st->fifo_head) cudaFree(st->fifo_head);
    if (st->fifo_len) cudaFree(st->fifo_len);
    if (st->sort_ids) cudaFree(st->sort_ids);
    if (st->sort_assigned) cudaFree(st->sort_assigned);
    if (st->sort_admit) cudaFree(st->sort_admit);
    if (st->sorted_ids) cudaFree(st->sorted_ids);
    free(st);
}
