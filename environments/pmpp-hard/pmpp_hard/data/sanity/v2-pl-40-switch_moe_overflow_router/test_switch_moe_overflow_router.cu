// file: test_switch_moe_overflow_router.cu

#include "switch_moe_overflow_router_common.h"
#include "switch_moe_overflow_router_oracle.hpp"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0x51c0ffee0d15ea5eULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                   \
            throw std::runtime_error(_oss.str());                              \
        }                                                                       \
    } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x8e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    int uniform_int(int lo, int hi) {
        const uint64_t span = (uint64_t)(hi - lo + 1);
        return lo + (int)(next_u64() % span);
    }
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) {
        count = n;
        if (n == 0) { ptr = nullptr; return; }
        CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n));
    }
    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("upload size mismatch");
        if (count) CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count) CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost));
        return host;
    }
};

template <typename T>
struct GuardedDeviceBuffer {
    uint8_t* raw = nullptr;
    T* ptr = nullptr;
    size_t count = 0, data_bytes = 0, total_bytes = 0;
    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;
    ~GuardedDeviceBuffer() { if (raw) cudaFree(raw); }
    void allocate(size_t n) {
        count = n; data_bytes = sizeof(T) * count;
        total_bytes = kGuardBytes + data_bytes + kGuardBytes;
        CUDA_CHECK(cudaMalloc((void**)&raw, total_bytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, total_bytes));
        ptr = (T*)(raw + kGuardBytes);
    }
    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> before(kGuardBytes), after(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(before.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (before[i] != kGuardByte) {
                if (error) { std::ostringstream o; o << "left guard corrupted for " << name << " at " << i; *error = o.str(); }
                return false;
            }
            if (after[i] != kGuardByte) {
                if (error) { std::ostringstream o; o << "right guard corrupted for " << name << " at " << i; *error = o.str(); }
                return false;
            }
        }
        return true;
    }
};

// ---- host-side op description ----
struct OpHost {
    int32_t kind = 0;
    uint64_t a = 0;
    uint64_t b = 0;
    // ROUTE candidates
    std::vector<int32_t> cand_expert;
    std::vector<int32_t> cand_logit;
    std::vector<int32_t> cand_ordinal;
};

struct StepHost {
    SmorRunSpec run;
    std::vector<int32_t> op_kind;
    std::vector<uint64_t> op_a;
    std::vector<uint64_t> op_b;
    std::vector<int32_t> op_cand_off;
    std::vector<int32_t> op_cand_count;
    std::vector<int32_t> cand_expert;
    std::vector<int32_t> cand_logit;
    std::vector<int32_t> cand_ordinal;
};

struct StepResult {
    uint64_t v[17] = {0};
};

struct Scenario {
    std::string name;
    SmorProblemSpec spec;
    std::vector<uint64_t> credit_cap;
    std::vector<uint64_t> initial_credit;
    std::vector<StepHost> steps;
};

static SmorProblemSpec make_spec(int E, int max_live, int overflow, int max_cands,
                                 int token_space, int max_batch, int max_steps) {
    SmorProblemSpec spec = {};
    spec.abi_version = SMOR_ABI_VERSION;
    spec.num_experts = E;
    spec.max_live_tokens = max_live;
    spec.overflow_capacity = overflow;
    spec.max_candidates_per_route = max_cands;
    spec.token_space = token_space;
    spec.max_batch = max_batch;
    spec.max_steps = max_steps;
    spec.flags = 0;
    if (!smor_validate_problem_spec(&spec)) throw std::runtime_error("invalid spec");
    return spec;
}

// Pack a list of OpHost into a StepHost (flat candidate arrays).
static StepHost make_step(const SmorProblemSpec& spec, int step_id, const std::vector<OpHost>& ops) {
    StepHost s;
    s.run = {};
    s.run.abi_version = SMOR_ABI_VERSION;
    s.run.batch_size = (int32_t)ops.size();
    s.run.step_id = step_id;

    const size_t rows = std::max<size_t>(1, ops.size());
    s.op_kind.assign(rows, 0);
    s.op_a.assign(rows, 0);
    s.op_b.assign(rows, 0);
    s.op_cand_off.assign(rows, 0);
    s.op_cand_count.assign(rows, 0);

    for (size_t i = 0; i < ops.size(); ++i) {
        const OpHost& op = ops[i];
        s.op_kind[i] = op.kind;
        s.op_a[i] = op.a;
        s.op_b[i] = op.b;
        if (op.kind == SMOR_OP_ROUTE) {
            s.op_cand_off[i] = (int32_t)s.cand_expert.size();
            s.op_cand_count[i] = (int32_t)op.cand_expert.size();
            for (size_t k = 0; k < op.cand_expert.size(); ++k) {
                s.cand_expert.push_back(op.cand_expert[k]);
                s.cand_logit.push_back(op.cand_logit[k]);
                s.cand_ordinal.push_back(op.cand_ordinal[k]);
            }
        } else {
            s.op_cand_off[i] = (int32_t)s.cand_expert.size();
            s.op_cand_count[i] = 0;
        }
    }
    s.run.cand_total = (int32_t)s.cand_expert.size();

    if (s.cand_expert.empty()) {  // keep arrays nonempty for clean uploads
        s.cand_expert.assign(1, 0);
        s.cand_logit.assign(1, 0);
        s.cand_ordinal.assign(1, 0);
    }

    if (!smor_validate_run_spec(&s.run, &spec)) throw std::runtime_error("invalid run spec");
    return s;
}

// helpers to build ops
static OpHost op_refill(uint64_t expert, uint64_t amount) {
    OpHost o; o.kind = SMOR_OP_REFILL; o.a = expert; o.b = amount; return o;
}
static OpHost op_route(uint64_t token_id, uint64_t cost,
                       std::vector<int32_t> experts, std::vector<int32_t> logits,
                       std::vector<int32_t> ordinals) {
    OpHost o; o.kind = SMOR_OP_ROUTE; o.a = token_id; o.b = cost;
    o.cand_expert = std::move(experts); o.cand_logit = std::move(logits); o.cand_ordinal = std::move(ordinals);
    return o;
}
static OpHost op_drain(uint64_t limit) { OpHost o; o.kind = SMOR_OP_DRAIN; o.a = limit; return o; }
static OpHost op_retire(uint64_t token_id) { OpHost o; o.kind = SMOR_OP_RETIRE; o.a = token_id; return o; }
static OpHost op_drop_through(uint64_t cutoff) { OpHost o; o.kind = SMOR_OP_DROP_QUEUED_THROUGH; o.a = cutoff; return o; }

// ---------------------------------------------------------------------------
// Scenario builders
// ---------------------------------------------------------------------------

// 1. Top-2 routing with candidate dedup, primary/secondary fallback.
static Scenario sc_top2_dedup() {
    Scenario sc;
    sc.name = "top2_dedup_fallback";
    sc.spec = make_spec(4, 8, 4, 8, 1024, 64, 16);
    sc.credit_cap.assign(4, 1000);
    sc.initial_credit = {10, 5, 0, 3};

    // Step 0: routes exercising dedup + tie-breaks.
    std::vector<OpHost> s0;
    // token 1: experts {0,0,2} logits {5,7,7} ordinals {0,1,2}
    //   collapse expert0 -> logit 7 (ord1). sort: e0(7), e2(7) -> tie on logit,
    //   ascending expert -> primary=0, secondary=2.
    s0.push_back(op_route(1, 4, {0, 0, 2}, {5, 7, 7}, {0, 1, 2}));
    // token 2: primary e1 logit9, but credit[1]=5 enough for cost 5 -> accept primary
    s0.push_back(op_route(2, 5, {1, 3}, {9, 1}, {0, 1}));
    // token 3: primary e1 (now credit 0), secondary e3 credit 3 -> accept secondary cost3
    s0.push_back(op_route(3, 3, {1, 3}, {9, 1}, {0, 1}));
    // token 4: invalid expert ids collapse to none -> INVALID
    s0.push_back(op_route(4, 1, {9, 10}, {1, 2}, {0, 1}));
    // token 5: cost 0 -> INVALID
    s0.push_back(op_route(5, 0, {0}, {1}, {0}));
    // duplicate token 1 -> DUPLICATE
    s0.push_back(op_route(1, 1, {0}, {1}, {0}));
    sc.steps.push_back(make_step(sc.spec, 0, s0));

    // Step 1: refills then more routes.
    std::vector<OpHost> s1;
    s1.push_back(op_refill(2, 100));        // expert2 0->100
    s1.push_back(op_refill(9, 5));          // invalid expert
    s1.push_back(op_route(10, 50, {2}, {3}, {0}));  // accept primary e2
    s1.push_back(op_retire(2));             // retire live token 2
    s1.push_back(op_retire(999));           // absent -> invalid
    sc.steps.push_back(make_step(sc.spec, 1, s1));

    // Step 2: empty step
    sc.steps.push_back(make_step(sc.spec, 2, {}));
    return sc;
}

// 2. Overflow queue + strict FIFO replay (head-of-line blocking).
static Scenario sc_fifo_replay() {
    Scenario sc;
    sc.name = "fifo_replay_hol_block";
    sc.spec = make_spec(3, 16, 8, 4, 1024, 64, 16);
    sc.credit_cap.assign(3, 1000);
    sc.initial_credit = {0, 0, 0};  // all experts broke -> everything queues

    std::vector<OpHost> s0;
    // queue tokens 1..5 all to expert 0 (primary), cost 10 each. all QUEUE.
    for (int t = 1; t <= 5; ++t) s0.push_back(op_route(t, 10, {0}, {5}, {0}));
    sc.steps.push_back(make_step(sc.spec, 0, s0));

    std::vector<OpHost> s1;
    // refill expert0 with only 25 credit -> can replay 2 (cost 10 each, 25->5 left),
    // head (token1) at cost 10 ok, token2 ok, token3 needs 10 but only 5 -> HOL block.
    s1.push_back(op_refill(0, 25));
    s1.push_back(op_drain(100));  // limit large; FIFO blocks after 2 replays
    sc.steps.push_back(make_step(sc.spec, 1, s1));

    std::vector<OpHost> s2;
    // refill more, drain limit=1 -> exactly one more replay (token3)
    s2.push_back(op_refill(0, 100));
    s2.push_back(op_drain(1));
    sc.steps.push_back(make_step(sc.spec, 2, s2));

    std::vector<OpHost> s3;
    // drain 0 -> no-op; then drain remaining
    s3.push_back(op_drain(0));
    s3.push_back(op_drain(100));
    sc.steps.push_back(make_step(sc.spec, 3, s3));
    return sc;
}

// 3. Capacity drop vs OOM: overflow full, max_live small.
static Scenario sc_capacity_oom() {
    Scenario sc;
    sc.name = "capacity_drop_and_oom";
    sc.spec = make_spec(2, 2, 2, 2, 1024, 64, 16);  // max_live=2, overflow=2
    sc.credit_cap.assign(2, 1000);
    sc.initial_credit = {100, 100};

    std::vector<OpHost> s0;
    // accept 2 (fills live), then routes that fail credit? credit is plenty;
    // to force queue we drain credit: route big-cost tokens.
    s0.push_back(op_route(1, 50, {0}, {5}, {0}));   // accept primary, credit0=50
    s0.push_back(op_route(2, 50, {1}, {5}, {0}));   // accept primary, credit1=50
    // live now == max_live(2). next accepted token -> OOM_DROP (credit ok but live full)
    s0.push_back(op_route(3, 10, {0}, {5}, {0}));   // OOM_DROP
    sc.steps.push_back(make_step(sc.spec, 0, s0));

    std::vector<OpHost> s1;
    // drain all credit so further routes fail admission and queue.
    // expert0 credit 50, expert1 credit 50. route cost 60 each (> credit) -> fail -> queue.
    s1.push_back(op_route(4, 60, {0}, {5}, {0}));  // queue (room: overflow 2)
    s1.push_back(op_route(5, 60, {1}, {5}, {0}));  // queue (overflow now full)
    s1.push_back(op_route(6, 60, {0}, {5}, {0}));  // overflow full -> CAPACITY_DROP
    sc.steps.push_back(make_step(sc.spec, 1, s1));

    std::vector<OpHost> s2;
    // retire live token1, drop_queued_through to clear part of FIFO.
    s2.push_back(op_retire(1));            // RETIRE_LIVE, no credit refund
    s2.push_back(op_drop_through(1000000));  // drops all queued (arrival_seq small)
    sc.steps.push_back(make_step(sc.spec, 2, s2));
    return sc;
}

// 4. DROP_QUEUED_THROUGH partial cutoff (stop at first larger arrival_seq).
static Scenario sc_drop_through_partial() {
    Scenario sc;
    sc.name = "drop_through_partial_cutoff";
    sc.spec = make_spec(2, 16, 8, 2, 1024, 64, 16);
    sc.credit_cap.assign(2, 1000);
    sc.initial_credit = {0, 0};  // force queueing

    std::vector<OpHost> s0;
    for (int t = 1; t <= 4; ++t) s0.push_back(op_route(t, 5, {0}, {5}, {0}));  // queue 4 tokens
    sc.steps.push_back(make_step(sc.spec, 0, s0));

    std::vector<OpHost> s1;
    // arrival_seqs are 0,1,2,3 (event_seq at each QUEUE). cutoff=1 drops tokens
    // with arrival 0 and 1 (tokens 1,2), stops at token3 (arrival 2).
    s1.push_back(op_drop_through(1));
    sc.steps.push_back(make_step(sc.spec, 1, s1));

    std::vector<OpHost> s2;
    // retire a queued token (token 4), check FIFO unlink.
    s2.push_back(op_retire(4));   // RETIRE_QUEUED
    s2.push_back(op_retire(3));   // RETIRE_QUEUED -> empties FIFO
    sc.steps.push_back(make_step(sc.spec, 2, s2));
    return sc;
}

// 5. Refill saturation at credit_cap and UINT64_MAX.
static Scenario sc_refill_saturation() {
    Scenario sc;
    sc.name = "refill_saturation";
    sc.spec = make_spec(3, 8, 4, 2, 1024, 64, 16);
    sc.credit_cap = {100, SMOR_U64_MAX, 50};
    sc.initial_credit = {90, SMOR_U64_MAX - 3, 0};

    std::vector<OpHost> s0;
    s0.push_back(op_refill(0, 50));               // 90+50=140 -> cap 100
    s0.push_back(op_refill(1, 100));              // saturating add wraps -> UINT64_MAX, cap is max
    s0.push_back(op_refill(2, SMOR_U64_MAX));     // 0 + max -> max -> cap 50
    s0.push_back(op_refill(2, 0));                // no-op refill, still valid event
    sc.steps.push_back(make_step(sc.spec, 0, s0));

    std::vector<OpHost> s1;
    // route using big credit on expert1
    s1.push_back(op_route(1, 1000000, {1}, {5}, {0}));  // accept primary, huge credit
    sc.steps.push_back(make_step(sc.spec, 1, s1));
    return sc;
}

// 6. Large randomized adversarial mixed-op stress.
static Scenario sc_random_stress() {
    Scenario sc;
    sc.name = "random_mixed_stress";
    sc.spec = make_spec(8, 32, 16, 6, 4096, 96, 40);
    sc.credit_cap.assign(8, 100000);
    sc.initial_credit.assign(8, 0);
    for (int e = 0; e < 8; ++e) sc.initial_credit[e] = (uint64_t)(e * 7 + 3);

    SplitMix64 rng(g_state ^ 0x9999ULL);
    uint64_t next_token = 1;
    // track a small pool of "live-ish" token ids to retire
    std::vector<uint64_t> issued;

    for (int s = 0; s < 36; ++s) {
        std::vector<OpHost> ops;
        if (s == 7 || s == 23) {
            sc.steps.push_back(make_step(sc.spec, s, {}));  // empty steps
            continue;
        }
        int n = rng.uniform_int(4, 40);
        for (int i = 0; i < n; ++i) {
            int roll = rng.uniform_int(0, 99);
            if (roll < 45) {
                // ROUTE
                uint64_t tid;
                if (!issued.empty() && rng.uniform_int(0, 9) < 2) {
                    tid = issued[rng.uniform_int(0, (int)issued.size() - 1)];  // maybe duplicate
                } else {
                    tid = next_token++;
                }
                int cc = rng.uniform_int(1, 6);
                std::vector<int32_t> ex, lg, od;
                for (int k = 0; k < cc; ++k) {
                    int e = rng.uniform_int(0, 9);  // some >=8 -> invalid expert
                    ex.push_back(e);
                    lg.push_back(rng.uniform_int(-5, 5));
                    od.push_back(k);
                }
                uint64_t cost = (uint64_t)rng.uniform_int(0, 40);  // 0 -> invalid sometimes
                ops.push_back(op_route(tid, cost, ex, lg, od));
                issued.push_back(tid);
                if (issued.size() > 64) issued.erase(issued.begin());
            } else if (roll < 65) {
                ops.push_back(op_refill(rng.uniform_int(0, 9), (uint64_t)rng.uniform_int(0, 200)));
            } else if (roll < 80) {
                ops.push_back(op_drain(rng.uniform_int(0, 8)));
            } else if (roll < 92) {
                uint64_t tid = issued.empty() ? 0 : issued[rng.uniform_int(0, (int)issued.size() - 1)];
                ops.push_back(op_retire(tid));
            } else {
                ops.push_back(op_drop_through((uint64_t)rng.uniform_int(0, 500)));
            }
        }
        sc.steps.push_back(make_step(sc.spec, s, ops));
    }
    return sc;
}

// 7. Secondary replay + route_kind diversity (drives assignment_hash route_kind field).
static Scenario sc_route_kinds() {
    Scenario sc;
    sc.name = "route_kind_diversity";
    sc.spec = make_spec(3, 16, 8, 3, 1024, 64, 16);
    sc.credit_cap.assign(3, 1000);
    sc.initial_credit = {10, 0, 0};

    std::vector<OpHost> s0;
    // token1 primary e0 cost10 -> accept primary (credit0 0 now)
    s0.push_back(op_route(1, 10, {0, 1}, {9, 8}, {0, 1}));
    // token2 primary e0 (broke), secondary e2 credit0 -> fail -> queue
    s0.push_back(op_route(2, 5, {0, 2}, {9, 8}, {0, 1}));
    sc.steps.push_back(make_step(sc.spec, 0, s0));

    std::vector<OpHost> s1;
    // refill e2 -> drain: token2 head, primary e0 still broke, secondary e2 ok -> REPLAY_SECONDARY
    s1.push_back(op_refill(2, 100));
    s1.push_back(op_drain(10));
    // route token3: primary e1 credit0, secondary e2 ok -> ACCEPT_SECONDARY
    s1.push_back(op_route(3, 5, {1, 2}, {9, 8}, {0, 1}));
    sc.steps.push_back(make_step(sc.spec, 1, s1));

    std::vector<OpHost> s2;
    // refill e1, route token4 primary e1 -> ACCEPT_PRIMARY; queue token5 to e1 over-cost
    s2.push_back(op_refill(1, 7));
    s2.push_back(op_route(4, 5, {1}, {9}, {0}));    // accept primary e1, credit1 2 left
    s2.push_back(op_route(5, 5, {1}, {9}, {0}));    // fail (credit1=2) -> queue
    s2.push_back(op_refill(1, 100));
    s2.push_back(op_drain(10));                     // replay token5 primary -> REPLAY_PRIMARY
    sc.steps.push_back(make_step(sc.spec, 2, s2));
    return sc;
}

// ---------------------------------------------------------------------------

static bool check_inputs_unchanged(const StepHost& step,
                                   const DeviceBuffer<int32_t>& d_kind,
                                   const DeviceBuffer<uint64_t>& d_a,
                                   const DeviceBuffer<uint64_t>& d_b,
                                   const DeviceBuffer<int32_t>& d_coff,
                                   const DeviceBuffer<int32_t>& d_ccnt,
                                   const DeviceBuffer<int32_t>& d_ce,
                                   const DeviceBuffer<int32_t>& d_cl,
                                   const DeviceBuffer<int32_t>& d_co,
                                   std::string* error) {
    if (d_kind.download() != step.op_kind) { if (error) *error = "op_kind modified"; return false; }
    if (d_a.download() != step.op_a) { if (error) *error = "op_a modified"; return false; }
    if (d_b.download() != step.op_b) { if (error) *error = "op_b modified"; return false; }
    if (d_coff.download() != step.op_cand_off) { if (error) *error = "op_cand_off modified"; return false; }
    if (d_ccnt.download() != step.op_cand_count) { if (error) *error = "op_cand_count modified"; return false; }
    if (d_ce.download() != step.cand_expert) { if (error) *error = "cand_expert modified"; return false; }
    if (d_cl.download() != step.cand_logit) { if (error) *error = "cand_logit modified"; return false; }
    if (d_co.download() != step.cand_ordinal) { if (error) *error = "cand_ordinal modified"; return false; }
    return true;
}

static bool run_one_step(
    const Scenario& sc, const StepHost& step,
    SmorOracleState* oracle, void* state,
    void* workspace, size_t workspace_bytes, cudaStream_t stream,
    StepResult* result, std::string* error) {

    DeviceBuffer<int32_t> d_kind; d_kind.allocate(step.op_kind.size()); d_kind.upload(step.op_kind);
    DeviceBuffer<uint64_t> d_a; d_a.allocate(step.op_a.size()); d_a.upload(step.op_a);
    DeviceBuffer<uint64_t> d_b; d_b.allocate(step.op_b.size()); d_b.upload(step.op_b);
    DeviceBuffer<int32_t> d_coff; d_coff.allocate(step.op_cand_off.size()); d_coff.upload(step.op_cand_off);
    DeviceBuffer<int32_t> d_ccnt; d_ccnt.allocate(step.op_cand_count.size()); d_ccnt.upload(step.op_cand_count);
    DeviceBuffer<int32_t> d_ce; d_ce.allocate(step.cand_expert.size()); d_ce.upload(step.cand_expert);
    DeviceBuffer<int32_t> d_cl; d_cl.allocate(step.cand_logit.size()); d_cl.upload(step.cand_logit);
    DeviceBuffer<int32_t> d_co; d_co.allocate(step.cand_ordinal.size()); d_co.upload(step.cand_ordinal);

    GuardedDeviceBuffer<uint64_t> o[17];
    for (int k = 0; k < 17; ++k) o[k].allocate(1);

    SmorInputs inputs = {};
    inputs.op_kind = d_kind.ptr;
    inputs.op_a = d_a.ptr;
    inputs.op_b = d_b.ptr;
    inputs.op_cand_off = d_coff.ptr;
    inputs.op_cand_count = d_ccnt.ptr;
    inputs.cand_expert = d_ce.ptr;
    inputs.cand_logit = d_cl.ptr;
    inputs.cand_ordinal = d_co.ptr;

    SmorOutputs outputs = {};
    outputs.refill_count = o[0].ptr;
    outputs.accepted_primary = o[1].ptr;
    outputs.accepted_secondary = o[2].ptr;
    outputs.queued = o[3].ptr;
    outputs.replayed_primary = o[4].ptr;
    outputs.replayed_secondary = o[5].ptr;
    outputs.capacity_drop = o[6].ptr;
    outputs.oom_drop = o[7].ptr;
    outputs.duplicate_count = o[8].ptr;
    outputs.retired_live = o[9].ptr;
    outputs.retired_queued = o[10].ptr;
    outputs.queue_drop = o[11].ptr;
    outputs.invalid_count = o[12].ptr;
    outputs.route_event_hash = o[13].ptr;
    outputs.credit_hash = o[14].ptr;
    outputs.assignment_hash = o[15].ptr;
    outputs.overflow_hash = o[16].ptr;

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_inputs_unchanged(step, d_kind, d_a, d_b, d_coff, d_ccnt, d_ce, d_cl, d_co, error))
        return false;

    const char* names[17] = {
        "refill_count","accepted_primary","accepted_secondary","queued",
        "replayed_primary","replayed_secondary","capacity_drop","oom_drop",
        "duplicate_count","retired_live","retired_queued","queue_drop",
        "invalid_count","route_event_hash","credit_hash","assignment_hash","overflow_hash"};
    for (int k = 0; k < 17; ++k) if (!o[k].check_guards(names[k], error)) return false;

    uint64_t got_v[17];
    for (int k = 0; k < 17; ++k) got_v[k] = o[k].download_data()[0];

    SmorHostInputsView hin = {};
    hin.op_kind = step.op_kind.data();
    hin.op_a = step.op_a.data();
    hin.op_b = step.op_b.data();
    hin.op_cand_off = step.op_cand_off.data();
    hin.op_cand_count = step.op_cand_count.data();
    hin.cand_expert = step.cand_expert.data();
    hin.cand_logit = step.cand_logit.data();
    hin.cand_ordinal = step.cand_ordinal.data();

    SmorExpected expected;
    oracle->step_once(step.run, hin, &expected);

    SmorHostOutputsView gv = {};
    gv.refill_count = &got_v[0];
    gv.accepted_primary = &got_v[1];
    gv.accepted_secondary = &got_v[2];
    gv.queued = &got_v[3];
    gv.replayed_primary = &got_v[4];
    gv.replayed_secondary = &got_v[5];
    gv.capacity_drop = &got_v[6];
    gv.oom_drop = &got_v[7];
    gv.duplicate_count = &got_v[8];
    gv.retired_live = &got_v[9];
    gv.retired_queued = &got_v[10];
    gv.queue_drop = &got_v[11];
    gv.invalid_count = &got_v[12];
    gv.route_event_hash = &got_v[13];
    gv.credit_hash = &got_v[14];
    gv.assignment_hash = &got_v[15];
    gv.overflow_hash = &got_v[16];

    if (!smor_check_all_outputs(expected, gv, error)) return false;

    if (result) for (int k = 0; k < 17; ++k) result->v[k] = got_v[k];
    return true;
}

static bool run_scenario_once(const Scenario& sc, bool verbose,
                              std::vector<StepResult>* results,
                              int* passed, int* total, std::string* first_error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    DeviceBuffer<uint64_t> d_cap; d_cap.allocate(sc.credit_cap.size()); d_cap.upload(sc.credit_cap);
    DeviceBuffer<uint64_t> d_icr; d_icr.allocate(sc.initial_credit.size()); d_icr.upload(sc.initial_credit);
    SmorInitConfig config = {};
    config.credit_cap = d_cap.ptr;
    config.initial_credit = d_icr.ptr;

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &config, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    SmorOracleState oracle;
    oracle.init(sc.spec, sc.credit_cap, sc.initial_credit);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.steps.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult result; std::string error;
        bool ok = run_one_step(sc, sc.steps[i], &oracle, state,
                               workspace.ptr, workspace.count, stream,
                               results ? &result : nullptr, &error);
        ++(*total);
        if (ok) ++(*passed);
        else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream o; o << sc.name << " step " << i << ": " << error;
                *first_error = o.str();
            }
        }
        if (results) results->push_back(result);
        if (verbose) {
            std::printf("scenario %-28s step %02zu/%02zu ops=%d %s%s%s\n",
                        sc.name.c_str(), i, sc.steps.size(), sc.steps[i].run.batch_size,
                        ok ? "PASS" : "FAIL", ok ? "" : "  ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(const std::vector<StepResult>& a, const std::vector<StepResult>& b, std::string* error) {
    if (a.size() != b.size()) { if (error) *error = "result length mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i)
        for (int k = 0; k < 17; ++k)
            if (a[i].v[k] != b[i].v[k]) {
                if (error) { std::ostringstream o; o << "replay mismatch step " << i << " field " << k; *error = o.str(); }
                return false;
            }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        std::vector<Scenario> scenarios;
        scenarios.push_back(sc_top2_dedup());
        scenarios.push_back(sc_fifo_replay());
        scenarios.push_back(sc_capacity_oom());
        scenarios.push_back(sc_drop_through_partial());
        scenarios.push_back(sc_refill_saturation());
        scenarios.push_back(sc_random_stress());
        scenarios.push_back(sc_route_kinds());

        int passed = 0, total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base, replay;
            std::string error;
            bool ok_base = run_scenario_once(sc, true, &base, &passed, &total, &error);
            bool ok_replay = run_scenario_once(sc, false, &replay, &passed, &total, &error);

            if (ok_base && ok_replay) {
                std::string cerr;
                if (compare_results(base, replay, &cerr))
                    std::printf("scenario %-28s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-28s exact replay FAIL  %s\n", sc.name.c_str(), cerr.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-28s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
