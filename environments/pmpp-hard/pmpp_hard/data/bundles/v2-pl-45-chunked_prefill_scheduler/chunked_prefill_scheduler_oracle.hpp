// file: chunked_prefill_scheduler_oracle.hpp
//
// Authoritative host reference model for T45. Independent of both GPU
// implementations. Implements every coupled invariant exactly with u64
// integer arithmetic only.

#ifndef CHUNKED_PREFILL_SCHEDULER_ORACLE_HPP_
#define CHUNKED_PREFILL_SCHEDULER_ORACLE_HPP_

#include "chunked_prefill_scheduler_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ----------------------- FNV-1a-64 -----------------------

static inline uint64_t cps_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void cps_oracle_fnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* q = static_cast<const uint8_t*>(p);
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = cps_oracle_fnv_byte(v, q[i]);
    *h = v;
}

static inline void cps_oracle_fold_u8(uint64_t* h, uint8_t x) { cps_oracle_fnv_bytes(h, &x, 1); }
static inline void cps_oracle_fold_u32(uint64_t* h, uint32_t x) { cps_oracle_fnv_bytes(h, &x, 4); }
static inline void cps_oracle_fold_u64(uint64_t* h, uint64_t x) { cps_oracle_fnv_bytes(h, &x, 8); }

static constexpr uint64_t CPS_FNV_OFFSET = 1469598103934665603ULL;

static inline uint64_t cps_sat_add_u64(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a) return UINT64_MAX;  // saturate on overflow
    return s;
}

// ----------------------- Request -----------------------

struct CpsReq {
    uint64_t request_id = 0;
    uint32_t tenant = 0;
    uint8_t priority = 0;
    uint64_t prompt_len = 0;
    uint64_t prompt_done = 0;
    uint64_t decode_len = 0;
    uint64_t decode_done = 0;
    uint64_t chunk_max = 0;
    uint64_t kv_tokens = 0;
    uint64_t arrival_seq = 0;
    uint64_t last_scheduled_iter = UINT64_MAX;
    uint64_t moe_drop_count = 0;
    uint8_t phase = CPS_PHASE_PREFILL;
};

struct CpsExpected {
    CpsCounts counts{};
    uint64_t batch_hash = 0;
    uint64_t moe_hash = 0;
    uint64_t finalize_hash = 0;
    uint64_t queue_hash = 0;
    uint64_t request_hash = 0;
    uint64_t bucket_hash = 0;
    uint64_t scalar_hash = 0;
};

struct CpsOracle {
    CpsProblemSpec spec{};

    // scalars
    uint64_t iter_seq = 0;
    uint64_t event_seq = 0;
    uint64_t arrival_seq_next = 1;
    uint64_t kv_capacity_tokens = 0;

    std::vector<uint64_t> bucket_tokens;  // [num_tenants]
    std::vector<uint64_t> bucket_cap;     // [num_tenants]
    std::vector<uint64_t> expert_capacity;// [num_experts]

    // live request table keyed by request_id
    std::map<uint64_t, CpsReq> table;

    std::deque<uint64_t> prefill_queue;
    std::deque<uint64_t> decode_queue;

    // cumulative counters + cumulative event/folded hashes
    CpsCounts counts{};
    uint64_t batch_hash = CPS_FNV_OFFSET;
    uint64_t moe_hash = CPS_FNV_OFFSET;
    uint64_t finalize_hash = CPS_FNV_OFFSET;

    void init(const CpsProblemSpec& s) {
        spec = s;
        reset();
    }

    void reset() {
        iter_seq = 0;
        event_seq = 0;
        arrival_seq_next = 1;
        kv_capacity_tokens = static_cast<uint64_t>(static_cast<uint32_t>(spec.kv_capacity_tokens));

        bucket_tokens.assign((size_t)spec.num_tenants, 0);
        bucket_cap.assign((size_t)spec.num_tenants, 0);
        for (int t = 0; t < spec.num_tenants; ++t) {
            bucket_cap[(size_t)t] = spec.bucket_cap[(size_t)t];
            bucket_tokens[(size_t)t] = spec.initial_bucket_tokens[(size_t)t];
        }
        expert_capacity.assign((size_t)spec.num_experts, 0);
        for (int e = 0; e < spec.num_experts; ++e) {
            expert_capacity[(size_t)e] = spec.expert_capacity[(size_t)e];
        }

        table.clear();
        prefill_queue.clear();
        decode_queue.clear();

        std::memset(&counts, 0, sizeof(counts));
        batch_hash = CPS_FNV_OFFSET;
        moe_hash = CPS_FNV_OFFSET;
        finalize_hash = CPS_FNV_OFFSET;
    }

    // ----- helpers -----

    uint64_t total_live_kv() const {
        uint64_t s = 0;
        for (const auto& kv : table) s += kv.second.kv_tokens;
        return s;
    }

    uint64_t free_kv() const {
        uint64_t used = total_live_kv();
        if (used >= kv_capacity_tokens) return 0;
        return kv_capacity_tokens - used;
    }

    void remove_from_queue(uint64_t rid) {
        for (auto it = prefill_queue.begin(); it != prefill_queue.end(); ++it) {
            if (*it == rid) { prefill_queue.erase(it); return; }
        }
        for (auto it = decode_queue.begin(); it != decode_queue.end(); ++it) {
            if (*it == rid) { decode_queue.erase(it); return; }
        }
    }

    // emit a finalize/evict event, folding into finalize_hash. reason==event_kind.
    void emit_finalize(uint8_t event_kind, uint32_t op_index, const CpsReq& r,
                       uint64_t kv_freed) {
        uint64_t h = finalize_hash;
        cps_oracle_fold_u8(&h, event_kind);
        cps_oracle_fold_u64(&h, event_seq);
        cps_oracle_fold_u32(&h, op_index);
        cps_oracle_fold_u64(&h, r.request_id);
        cps_oracle_fold_u32(&h, r.tenant);
        cps_oracle_fold_u8(&h, r.priority);
        cps_oracle_fold_u64(&h, r.prompt_done);
        cps_oracle_fold_u64(&h, r.decode_done);
        cps_oracle_fold_u64(&h, kv_freed);
        cps_oracle_fold_u64(&h, r.moe_drop_count);
        cps_oracle_fold_u8(&h, event_kind);  // reason
        finalize_hash = h;
        event_seq += 1;  // wraps mod 2^64
    }

    // Global eviction selection. protected_set is a set of request_ids that
    // cannot be evicted; current is the candidate being scheduled (also
    // excluded), or UINT64_MAX for none. Returns chosen rid or UINT64_MAX.
    uint64_t pick_eviction(const std::vector<uint64_t>& protected_set,
                           uint64_t current) const {
        bool found = false;
        CpsReq best{};
        uint64_t best_rid = UINT64_MAX;
        for (const auto& kv : table) {
            uint64_t rid = kv.first;
            if (rid == current) continue;
            bool prot = false;
            for (uint64_t p : protected_set) if (p == rid) { prot = true; break; }
            if (prot) continue;
            const CpsReq& r = kv.second;
            if (!found) { found = true; best = r; best_rid = rid; continue; }
            // smaller priority wins
            if (r.priority != best.priority) {
                if (r.priority < best.priority) { best = r; best_rid = rid; }
                continue;
            }
            // tie: larger kv_tokens
            if (r.kv_tokens != best.kv_tokens) {
                if (r.kv_tokens > best.kv_tokens) { best = r; best_rid = rid; }
                continue;
            }
            // tie: older last_scheduled_iter (UINT64_MAX is oldest -> "largest" wins)
            if (r.last_scheduled_iter != best.last_scheduled_iter) {
                if (r.last_scheduled_iter > best.last_scheduled_iter) { best = r; best_rid = rid; }
                continue;
            }
            // tie: smaller arrival_seq
            if (r.arrival_seq != best.arrival_seq) {
                if (r.arrival_seq < best.arrival_seq) { best = r; best_rid = rid; }
                continue;
            }
            // tie: smaller request_id
            if (rid < best_rid) { best = r; best_rid = rid; }
        }
        return found ? best_rid : UINT64_MAX;
    }

    // Evict requests (protected-set aware) until free_kv >= needed_free or no
    // candidates. Each eviction emits FINALIZE_EVICT.
    void evict_for_need(uint64_t needed_free, const std::vector<uint64_t>& protected_set,
                        uint64_t current, uint32_t op_index) {
        while (free_kv() < needed_free) {
            uint64_t victim = pick_eviction(protected_set, current);
            if (victim == UINT64_MAX) break;
            CpsReq r = table[victim];
            uint64_t freed = r.kv_tokens;
            remove_from_queue(victim);
            table.erase(victim);
            emit_finalize(CPS_EV_FINALIZE_EVICT, op_index, r, freed);
            counts.evicted += 1;
        }
    }

    // MoE route a single token. Updates expert_remaining in-place, folds moe_hash,
    // updates counts and req.moe_drop_count. token_ordinal is 0-based count in iter.
    void moe_route(CpsReq& r, uint8_t phase_code, uint64_t abs_token,
                   uint64_t token_ordinal, std::vector<uint64_t>& expert_remaining) {
        // h = FNV1a64(moe_seed, iter_seq, request_id, tenant, phase_code, abs_token)
        uint64_t h = CPS_FNV_OFFSET;
        cps_oracle_fold_u64(&h, spec.moe_seed);
        cps_oracle_fold_u64(&h, iter_seq);
        cps_oracle_fold_u64(&h, r.request_id);
        cps_oracle_fold_u32(&h, r.tenant);
        cps_oracle_fold_u8(&h, phase_code);
        cps_oracle_fold_u64(&h, abs_token);

        uint32_t ne = (uint32_t)spec.num_experts;
        uint32_t primary = (uint32_t)(h % ne);
        uint32_t secondary;
        if (ne == 1) {
            secondary = primary;
        } else {
            secondary = (uint32_t)((primary + 1 + ((h >> 32) % (ne - 1))) % ne);
        }

        uint8_t route_kind;
        uint32_t assigned;
        uint64_t remaining_after;
        if (expert_remaining[primary] > 0) {
            expert_remaining[primary] -= 1;
            route_kind = CPS_ROUTE_PRIMARY;
            assigned = primary;
            remaining_after = expert_remaining[primary];
            counts.moe_primary += 1;
        } else if (secondary != primary && expert_remaining[secondary] > 0) {
            expert_remaining[secondary] -= 1;
            route_kind = CPS_ROUTE_SECONDARY;
            assigned = secondary;
            remaining_after = expert_remaining[secondary];
            counts.moe_secondary += 1;
        } else {
            route_kind = CPS_ROUTE_DROP;
            assigned = UINT32_MAX;
            remaining_after = UINT64_MAX;
            counts.moe_dropped += 1;
            r.moe_drop_count += 1;
        }

        uint64_t mh = moe_hash;
        cps_oracle_fold_u64(&mh, iter_seq);
        cps_oracle_fold_u64(&mh, token_ordinal);
        cps_oracle_fold_u64(&mh, r.request_id);
        cps_oracle_fold_u32(&mh, r.tenant);
        cps_oracle_fold_u8(&mh, phase_code);
        cps_oracle_fold_u64(&mh, abs_token);
        cps_oracle_fold_u32(&mh, primary);
        cps_oracle_fold_u32(&mh, secondary);
        cps_oracle_fold_u8(&mh, route_kind);
        cps_oracle_fold_u32(&mh, assigned);
        cps_oracle_fold_u64(&mh, remaining_after);
        moe_hash = mh;
    }

    // ----- operations -----

    void op_arrive(const CpsOp& op) {
        // validity checks
        if (op.tenant >= (uint32_t)spec.num_tenants) { counts.invalid_count += 1; return; }
        if (op.prompt_len == 0 && op.decode_len == 0) { counts.invalid_count += 1; return; }
        if (table.find(op.request_id) != table.end()) { counts.invalid_count += 1; return; }
        uint64_t cm = (op.chunk_max != 0) ? op.chunk_max
                                          : (uint64_t)(uint32_t)spec.default_chunk_max;
        if (cm == 0) { counts.invalid_count += 1; return; }

        // full table -> reject
        if ((int)table.size() >= spec.max_live_requests) {
            counts.arrive_reject += 1;
            return;
        }

        CpsReq r{};
        r.request_id = op.request_id;
        r.tenant = op.tenant;
        r.priority = (uint8_t)(op.priority & 0xFF);
        r.prompt_len = op.prompt_len;
        r.prompt_done = 0;
        r.decode_len = op.decode_len;
        r.decode_done = 0;
        r.chunk_max = cm;
        r.kv_tokens = 0;
        r.arrival_seq = arrival_seq_next;
        arrival_seq_next += 1;  // wraps mod 2^64
        r.last_scheduled_iter = UINT64_MAX;
        r.moe_drop_count = 0;

        if (op.prompt_len > 0) {
            r.phase = CPS_PHASE_PREFILL;
            table[r.request_id] = r;
            prefill_queue.push_back(r.request_id);
        } else {
            r.phase = CPS_PHASE_DECODE;
            table[r.request_id] = r;
            decode_queue.push_back(r.request_id);
        }
        counts.arrive_ok += 1;
    }

    void op_refill(const CpsOp& op) {
        if (op.tenant >= (uint32_t)spec.num_tenants) { counts.invalid_count += 1; return; }
        uint64_t t = op.tenant;
        uint64_t v = cps_sat_add_u64(bucket_tokens[t], op.a);
        if (v > bucket_cap[t]) v = bucket_cap[t];
        bucket_tokens[t] = v;
        counts.tenant_refills += 1;
    }

    void op_set_kv_cap(const CpsOp& op) {
        kv_capacity_tokens = op.a;
        std::vector<uint64_t> empty;
        while (total_live_kv() > kv_capacity_tokens) {
            if (table.empty()) break;
            uint64_t victim = pick_eviction(empty, UINT64_MAX);
            if (victim == UINT64_MAX) break;
            CpsReq r = table[victim];
            uint64_t freed = r.kv_tokens;
            remove_from_queue(victim);
            table.erase(victim);
            emit_finalize(CPS_EV_EVICT_KV_SHRINK, (uint32_t)op.op_index, r, freed);
            counts.kv_shrink_evicted += 1;
        }
    }

    void op_cancel(const CpsOp& op) {
        auto it = table.find(op.request_id);
        if (it == table.end()) { counts.invalid_count += 1; return; }
        CpsReq r = it->second;
        uint64_t freed = r.kv_tokens;
        remove_from_queue(op.request_id);
        table.erase(it);
        emit_finalize(CPS_EV_FINALIZE_CANCEL, (uint32_t)op.op_index, r, freed);
        counts.cancelled += 1;
    }

    void op_step_iter(const CpsOp& op) {
        counts.iterations += 1;

        uint64_t tokens_left = op.a;
        uint64_t slots_left = op.b;
        uint64_t max_slots = (uint64_t)(uint32_t)spec.max_batch_slots;
        if (slots_left > max_slots) slots_left = max_slots;

        std::vector<uint64_t> expert_remaining(expert_capacity.begin(), expert_capacity.end());
        std::vector<uint64_t> protected_set;

        if (tokens_left == 0 || slots_left == 0) {
            iter_seq += 1;
            return;
        }

        uint64_t slot_ordinal = 0;        // chunks scheduled this iteration
        uint64_t token_ordinal = 0;       // tokens routed this iteration

        // -------- DECODE PASS --------
        {
            size_t decode_scan_count = decode_queue.size();
            for (size_t s = 0; s < decode_scan_count; ++s) {
                if (tokens_left == 0 || slots_left == 0) break;
                if (decode_queue.empty()) break;
                uint64_t rid = decode_queue.front();
                decode_queue.pop_front();
                auto it = table.find(rid);
                if (it == table.end()) continue;  // not live; counts against scan
                CpsReq& r = it->second;
                uint64_t t = r.tenant;
                if (bucket_tokens[t] == 0) {
                    counts.throttle_skips += 1;
                    decode_queue.push_back(rid);
                    continue;
                }
                // ensure >=1 free KV
                if (free_kv() == 0) {
                    protected_set.push_back(rid);
                    evict_for_need(1, protected_set, rid, (uint32_t)op.op_index);
                    protected_set.pop_back();
                }
                if (free_kv() == 0) {
                    decode_queue.push_back(rid);
                    counts.kv_skips += 1;
                    continue;
                }
                // capture *_before
                uint64_t bucket_before = bucket_tokens[t];
                uint64_t kv_before = r.kv_tokens;
                uint64_t tokens_left_before = tokens_left;
                uint64_t abs_token = r.prompt_len + r.decode_done;

                // batch record (decode)
                uint64_t bh = batch_hash;
                cps_oracle_fold_u8(&bh, CPS_REC_BATCH_DECODE);
                cps_oracle_fold_u64(&bh, iter_seq);
                cps_oracle_fold_u32(&bh, (uint32_t)slot_ordinal);
                cps_oracle_fold_u64(&bh, r.request_id);
                cps_oracle_fold_u32(&bh, r.tenant);
                cps_oracle_fold_u8(&bh, CPS_PHASE_DECODE);
                cps_oracle_fold_u64(&bh, abs_token);
                cps_oracle_fold_u64(&bh, (uint64_t)1);
                cps_oracle_fold_u64(&bh, bucket_before);
                cps_oracle_fold_u64(&bh, kv_before);
                cps_oracle_fold_u64(&bh, tokens_left_before);
                batch_hash = bh;
                counts.batch_decode_requests += 1;
                counts.decode_tokens += 1;

                // route exactly one token
                moe_route(r, CPS_PHASE_DECODE, abs_token, token_ordinal, expert_remaining);
                token_ordinal += 1;

                // consume
                bucket_tokens[t] -= 1;
                tokens_left -= 1;
                slots_left -= 1;
                r.kv_tokens += 1;
                r.decode_done += 1;
                r.last_scheduled_iter = iter_seq;
                protected_set.push_back(rid);
                slot_ordinal += 1;

                if (r.decode_done == r.decode_len) {
                    CpsReq fin = r;
                    uint64_t freed = fin.kv_tokens;
                    table.erase(it);
                    emit_finalize(CPS_EV_FINALIZE_COMPLETE, (uint32_t)op.op_index, fin, freed);
                    counts.completed += 1;
                } else {
                    decode_queue.push_back(rid);
                }
            }
        }

        // -------- PREFILL PASS --------
        {
            size_t prefill_scan_count = prefill_queue.size();
            for (size_t s = 0; s < prefill_scan_count; ++s) {
                if (tokens_left == 0 || slots_left == 0) break;
                if (prefill_queue.empty()) break;
                uint64_t rid = prefill_queue.front();
                prefill_queue.pop_front();
                auto it = table.find(rid);
                if (it == table.end()) continue;
                CpsReq& r = it->second;
                uint64_t t = r.tenant;
                uint64_t tenant_tokens = bucket_tokens[t];
                if (tenant_tokens == 0) {
                    counts.throttle_skips += 1;
                    prefill_queue.push_back(rid);
                    continue;
                }
                uint64_t remaining_prompt = r.prompt_len - r.prompt_done;
                uint64_t chunk = r.chunk_max;
                if (remaining_prompt < chunk) chunk = remaining_prompt;
                if (tokens_left < chunk) chunk = tokens_left;
                if (tenant_tokens < chunk) chunk = tenant_tokens;

                // ensure >= chunk free KV
                if (free_kv() < chunk) {
                    protected_set.push_back(rid);
                    evict_for_need(chunk, protected_set, rid, (uint32_t)op.op_index);
                    protected_set.pop_back();
                    uint64_t f = free_kv();
                    if (f < chunk) chunk = f;
                }
                if (chunk == 0) {
                    prefill_queue.push_back(rid);
                    counts.kv_skips += 1;
                    continue;
                }

                uint64_t bucket_before = bucket_tokens[t];
                uint64_t kv_before = r.kv_tokens;
                uint64_t tokens_left_before = tokens_left;
                uint64_t start_abs = r.prompt_done;

                // batch record (prefill)
                uint64_t bh = batch_hash;
                cps_oracle_fold_u8(&bh, CPS_REC_BATCH_PREFILL);
                cps_oracle_fold_u64(&bh, iter_seq);
                cps_oracle_fold_u32(&bh, (uint32_t)slot_ordinal);
                cps_oracle_fold_u64(&bh, r.request_id);
                cps_oracle_fold_u32(&bh, r.tenant);
                cps_oracle_fold_u8(&bh, CPS_PHASE_PREFILL);
                cps_oracle_fold_u64(&bh, start_abs);
                cps_oracle_fold_u64(&bh, chunk);
                cps_oracle_fold_u64(&bh, bucket_before);
                cps_oracle_fold_u64(&bh, kv_before);
                cps_oracle_fold_u64(&bh, tokens_left_before);
                batch_hash = bh;
                counts.batch_prefill_requests += 1;
                counts.prefill_tokens += chunk;

                // route each token in increasing j
                for (uint64_t j = 0; j < chunk; ++j) {
                    uint64_t abs_token = r.prompt_done + j;
                    moe_route(r, CPS_PHASE_PREFILL, abs_token, token_ordinal, expert_remaining);
                    token_ordinal += 1;
                }

                // consume
                r.prompt_done += chunk;
                r.kv_tokens += chunk;
                tokens_left -= chunk;
                bucket_tokens[t] -= chunk;
                slots_left -= 1;
                r.last_scheduled_iter = iter_seq;
                protected_set.push_back(rid);
                slot_ordinal += 1;

                if (r.prompt_done == r.prompt_len && r.decode_len > 0) {
                    r.phase = CPS_PHASE_DECODE;
                    decode_queue.push_back(rid);
                } else if (r.prompt_done == r.prompt_len && r.decode_len == 0) {
                    CpsReq fin = r;
                    uint64_t freed = fin.kv_tokens;
                    table.erase(it);
                    emit_finalize(CPS_EV_FINALIZE_COMPLETE, (uint32_t)op.op_index, fin, freed);
                    counts.completed += 1;
                } else {
                    prefill_queue.push_back(rid);
                }
            }
        }

        iter_seq += 1;
    }

    // ----- snapshot hashes -----

    uint64_t compute_queue_hash() const {
        uint64_t h = CPS_FNV_OFFSET;
        uint64_t pos = 0;
        for (uint64_t rid : prefill_queue) {
            cps_oracle_fold_u8(&h, CPS_Q_PREFILL);
            cps_oracle_fold_u64(&h, pos);
            cps_oracle_fold_u64(&h, rid);
            ++pos;
        }
        pos = 0;
        for (uint64_t rid : decode_queue) {
            cps_oracle_fold_u8(&h, CPS_Q_DECODE);
            cps_oracle_fold_u64(&h, pos);
            cps_oracle_fold_u64(&h, rid);
            ++pos;
        }
        return h;
    }

    uint64_t compute_request_hash() const {
        // sort by tenant asc then request_id asc
        std::vector<const CpsReq*> rs;
        rs.reserve(table.size());
        for (const auto& kv : table) rs.push_back(&kv.second);
        std::sort(rs.begin(), rs.end(), [](const CpsReq* a, const CpsReq* b) {
            if (a->tenant != b->tenant) return a->tenant < b->tenant;
            return a->request_id < b->request_id;
        });
        uint64_t h = CPS_FNV_OFFSET;
        for (const CpsReq* r : rs) {
            cps_oracle_fold_u64(&h, r->request_id);
            cps_oracle_fold_u32(&h, r->tenant);
            cps_oracle_fold_u8(&h, r->priority);
            cps_oracle_fold_u8(&h, r->phase);
            cps_oracle_fold_u64(&h, r->prompt_len);
            cps_oracle_fold_u64(&h, r->prompt_done);
            cps_oracle_fold_u64(&h, r->decode_len);
            cps_oracle_fold_u64(&h, r->decode_done);
            cps_oracle_fold_u64(&h, r->chunk_max);
            cps_oracle_fold_u64(&h, r->kv_tokens);
            cps_oracle_fold_u64(&h, r->arrival_seq);
            cps_oracle_fold_u64(&h, r->last_scheduled_iter);
            cps_oracle_fold_u64(&h, r->moe_drop_count);
        }
        return h;
    }

    uint64_t compute_bucket_hash() const {
        uint64_t h = CPS_FNV_OFFSET;
        for (int t = 0; t < spec.num_tenants; ++t) {
            cps_oracle_fold_u32(&h, (uint32_t)t);
            cps_oracle_fold_u64(&h, bucket_tokens[(size_t)t]);
            cps_oracle_fold_u64(&h, bucket_cap[(size_t)t]);
        }
        return h;
    }

    uint64_t compute_scalar_hash() const {
        uint64_t h = CPS_FNV_OFFSET;
        cps_oracle_fold_u64(&h, iter_seq);
        cps_oracle_fold_u64(&h, event_seq);
        cps_oracle_fold_u64(&h, arrival_seq_next);
        cps_oracle_fold_u64(&h, kv_capacity_tokens);
        cps_oracle_fold_u64(&h, (uint64_t)table.size());
        cps_oracle_fold_u64(&h, total_live_kv());
        return h;
    }

    void step_op(const CpsOp& op, CpsExpected* exp) {
        switch (op.opcode) {
            case CPS_OP_ARRIVE: op_arrive(op); break;
            case CPS_OP_REFILL_TENANT: op_refill(op); break;
            case CPS_OP_SET_KV_CAP: op_set_kv_cap(op); break;
            case CPS_OP_CANCEL: op_cancel(op); break;
            case CPS_OP_STEP_ITER: op_step_iter(op); break;
            default: counts.invalid_count += 1; break;
        }
        exp->counts = counts;
        exp->batch_hash = batch_hash;
        exp->moe_hash = moe_hash;
        exp->finalize_hash = finalize_hash;
        exp->queue_hash = compute_queue_hash();
        exp->request_hash = compute_request_hash();
        exp->bucket_hash = compute_bucket_hash();
        exp->scalar_hash = compute_scalar_hash();
    }
};

// ----- output comparison -----

struct CpsHostOutputs {
    CpsCounts counts{};
    uint64_t batch_hash = 0;
    uint64_t moe_hash = 0;
    uint64_t finalize_hash = 0;
    uint64_t queue_hash = 0;
    uint64_t request_hash = 0;
    uint64_t bucket_hash = 0;
    uint64_t scalar_hash = 0;
};

static inline bool cps_check_outputs(const CpsExpected& e, const CpsHostOutputs& g,
                                     std::string* err) {
#define CPS_CK_COUNT(field)                                                   \
    if (g.counts.field != e.counts.field) {                                   \
        if (err) { std::ostringstream o; o << "count " #field " got "         \
            << g.counts.field << " expected " << e.counts.field; *err=o.str(); } \
        return false; }
    CPS_CK_COUNT(arrive_ok)
    CPS_CK_COUNT(arrive_reject)
    CPS_CK_COUNT(tenant_refills)
    CPS_CK_COUNT(iterations)
    CPS_CK_COUNT(batch_decode_requests)
    CPS_CK_COUNT(batch_prefill_requests)
    CPS_CK_COUNT(decode_tokens)
    CPS_CK_COUNT(prefill_tokens)
    CPS_CK_COUNT(moe_primary)
    CPS_CK_COUNT(moe_secondary)
    CPS_CK_COUNT(moe_dropped)
    CPS_CK_COUNT(throttle_skips)
    CPS_CK_COUNT(kv_skips)
    CPS_CK_COUNT(completed)
    CPS_CK_COUNT(cancelled)
    CPS_CK_COUNT(evicted)
    CPS_CK_COUNT(kv_shrink_evicted)
    CPS_CK_COUNT(invalid_count)
#undef CPS_CK_COUNT

#define CPS_CK_HASH(field)                                                    \
    if (g.field != e.field) {                                                 \
        if (err) { std::ostringstream o; o << #field " got 0x" << std::hex    \
            << g.field << " expected 0x" << e.field; *err=o.str(); }          \
        return false; }
    CPS_CK_HASH(batch_hash)
    CPS_CK_HASH(moe_hash)
    CPS_CK_HASH(finalize_hash)
    CPS_CK_HASH(queue_hash)
    CPS_CK_HASH(request_hash)
    CPS_CK_HASH(bucket_hash)
    CPS_CK_HASH(scalar_hash)
#undef CPS_CK_HASH
    return true;
}

#endif  // CHUNKED_PREFILL_SCHEDULER_ORACLE_HPP_
