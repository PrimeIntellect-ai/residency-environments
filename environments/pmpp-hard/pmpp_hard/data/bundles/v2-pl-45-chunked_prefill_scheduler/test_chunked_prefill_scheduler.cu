// file: test_chunked_prefill_scheduler.cu
//
// Three-way validation harness for T45. Drives the solution op-stream against
// the authoritative host oracle, checks guard buffers, op-buffer immutability,
// and exact replay determinism. Multi-step, deeply adversarial scenarios.

#include "chunked_prefill_scheduler_common.h"
#include "chunked_prefill_scheduler_oracle.hpp"

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

static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::ostringstream _oss;                                           \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "   \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                             \
        }                                                                       \
    } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t s) : state(s) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    uint64_t range_u64(uint64_t lo, uint64_t hi) {
        return lo + next_u64() % (hi - lo + 1);
    }
    int chance(int permille) {
        return (int)(next_u64() % 1000) < permille;
    }
};

template <typename T>
struct GuardedBuf {
    uint8_t* raw = nullptr;
    T* ptr = nullptr;
    size_t count = 0, bytes = 0;
    GuardedBuf() = default;
    GuardedBuf(const GuardedBuf&) = delete;
    GuardedBuf& operator=(const GuardedBuf&) = delete;
    ~GuardedBuf() { if (raw) cudaFree(raw); }
    void allocate(size_t n) {
        count = n; bytes = sizeof(T) * n;
        CUDA_CHECK(cudaMalloc((void**)&raw, kGuardBytes + bytes + kGuardBytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, kGuardBytes + bytes + kGuardBytes));
        ptr = (T*)(raw + kGuardBytes);
    }
    std::vector<T> download() const {
        std::vector<T> h(count);
        if (count) CUDA_CHECK(cudaMemcpy(h.data(), ptr, bytes, cudaMemcpyDeviceToHost));
        return h;
    }
    bool check_guards(const char* name, std::string* err) const {
        std::vector<uint8_t> l(kGuardBytes), r(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(l.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r.data(), raw + kGuardBytes + bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (l[i] != kGuardByte) { if (err) { std::ostringstream o; o<<"left guard "<<name<<" byte "<<i; *err=o.str(); } return false; }
            if (r[i] != kGuardByte) { if (err) { std::ostringstream o; o<<"right guard "<<name<<" byte "<<i; *err=o.str(); } return false; }
        }
        return true;
    }
};

template <typename T>
struct DevBuf {
    T* ptr = nullptr; size_t count = 0;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) { count = n; if (n) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T)*n)); }
    void upload(const std::vector<T>& h) { if (h.size()) CUDA_CHECK(cudaMemcpy(ptr, h.data(), sizeof(T)*h.size(), cudaMemcpyHostToDevice)); }
    std::vector<T> download() const { std::vector<T> h(count); if (count) CUDA_CHECK(cudaMemcpy(h.data(), ptr, sizeof(T)*count, cudaMemcpyDeviceToHost)); return h; }
};

struct Scenario {
    std::string name;
    CpsProblemSpec spec;
    std::vector<CpsOp> ops;
};

static CpsProblemSpec make_spec(int nt, int kvcap, int maxlive, int defchunk,
                                int maxslots, int ne, uint64_t seed,
                                const std::vector<uint64_t>& bcap,
                                const std::vector<uint64_t>& binit,
                                const std::vector<uint64_t>& ecap,
                                int max_ops) {
    CpsProblemSpec s{};
    s.abi_version = CPS_ABI_VERSION;
    s.num_tenants = nt;
    s.kv_capacity_tokens = kvcap;
    s.max_live_requests = maxlive;
    s.default_chunk_max = defchunk;
    s.max_batch_slots = maxslots;
    s.num_experts = ne;
    s.max_ops = max_ops;
    s.moe_seed = seed;
    for (int i = 0; i < nt; ++i) { s.bucket_cap[i] = bcap[i]; s.initial_bucket_tokens[i] = binit[i]; }
    for (int e = 0; e < ne; ++e) s.expert_capacity[e] = ecap[e];
    if (!cps_validate_problem_spec(&s)) throw std::runtime_error("bad spec");
    return s;
}

static CpsOp mk(int opcode, int idx) {
    CpsOp o{};
    o.abi_version = CPS_ABI_VERSION;
    o.opcode = opcode;
    o.op_index = idx;
    return o;
}

// ---------------- scenario builders ----------------

// Scenario 1: small, dense, exercises full prefill->decode pipeline, MoE drops,
// throttling, KV eviction, completions, cancels.
static Scenario s_pipeline_pressure() {
    Scenario sc; sc.name = "pipeline_pressure";
    sc.spec = make_spec(/*nt*/3, /*kvcap*/12, /*maxlive*/8, /*defchunk*/4,
                        /*maxslots*/4, /*ne*/2, /*seed*/0xABCDEF12345ULL,
                        {10,10,10}, {3,5,2}, {2,1}, 64);
    int idx = 0;
    auto& ops = sc.ops;
    // arrivals: mix of prefill-only, prefill+decode, decode-only
    { CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=100; o.tenant=0; o.priority=5; o.prompt_len=7; o.decode_len=3; o.chunk_max=3; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=101; o.tenant=1; o.priority=2; o.prompt_len=5; o.decode_len=0; o.chunk_max=0; ops.push_back(o); } // defchunk
    { CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=102; o.tenant=2; o.priority=9; o.prompt_len=0; o.decode_len=4; o.chunk_max=2; ops.push_back(o); } // decode-only
    { CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=103; o.tenant=0; o.priority=1; o.prompt_len=10; o.decode_len=2; o.chunk_max=5; ops.push_back(o); }
    // invalid arrivals
    { CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=104; o.tenant=9; o.priority=1; o.prompt_len=3; o.decode_len=1; o.chunk_max=2; ops.push_back(o); } // bad tenant
    { CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=100; o.tenant=0; o.priority=1; o.prompt_len=3; o.decode_len=1; o.chunk_max=2; ops.push_back(o); } // dup id
    { CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=105; o.tenant=0; o.priority=1; o.prompt_len=0; o.decode_len=0; o.chunk_max=2; ops.push_back(o); } // both zero
    // steps
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=8; o.b=3; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=6; o.b=4; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_REFILL_TENANT, idx++); o.tenant=2; o.a=20; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=10; o.b=4; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_SET_KV_CAP, idx++); o.a=4; ops.push_back(o); } // shrink -> evictions
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=10; o.b=4; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_CANCEL, idx++); o.request_id=103; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_CANCEL, idx++); o.request_id=999; ops.push_back(o); } // invalid
    { CpsOp o = mk(CPS_OP_REFILL_TENANT, idx++); o.tenant=0; o.a=30; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_SET_KV_CAP, idx++); o.a=64; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=20; o.b=4; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=20; o.b=4; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=20; o.b=4; ops.push_back(o); }
    return sc;
}

// Scenario 2: zero-budget steps, throttle storms (buckets at zero), no refill.
static Scenario s_throttle_starvation() {
    Scenario sc; sc.name = "throttle_starvation";
    sc.spec = make_spec(4, 100, 16, 3, 8, 3, 0x5151515151ULL,
                        {100,100,100,100}, {0,0,1,0}, {1,1,1}, 64);
    int idx = 0; auto& ops = sc.ops;
    for (uint64_t r = 0; r < 6; ++r) {
        CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=200+r; o.tenant=(uint32_t)(r%4);
        o.priority=(uint32_t)(r%7); o.prompt_len=4+r; o.decode_len=r%3; o.chunk_max=2; ops.push_back(o);
    }
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=0; o.b=4; ops.push_back(o); }      // zero token budget
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=10; o.b=0; ops.push_back(o); }     // zero slot budget
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=10; o.b=8; ops.push_back(o); }     // only tenant 2 has tokens
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=10; o.b=8; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_REFILL_TENANT, idx++); o.tenant=0; o.a=3; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_REFILL_TENANT, idx++); o.tenant=99; o.a=3; ops.push_back(o); } // invalid tenant
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=10; o.b=8; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=10; o.b=8; ops.push_back(o); }
    return sc;
}

// Scenario 3: KV-tiny capacity forcing eviction churn + priority tie-breaks.
static Scenario s_eviction_tiebreaks() {
    Scenario sc; sc.name = "eviction_tiebreaks";
    sc.spec = make_spec(2, 3, 12, 8, 6, 4, 0x9090909090ULL,
                        {1000,1000}, {1000,1000}, {1,1,1,1}, 80);
    int idx = 0; auto& ops = sc.ops;
    // many same-priority requests to force kv_tokens/last_sched/arrival tie-breaks
    for (uint64_t r = 0; r < 8; ++r) {
        CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=300+r; o.tenant=(uint32_t)(r%2);
        o.priority=3; o.prompt_len=6; o.decode_len=4; o.chunk_max=2; ops.push_back(o);
    }
    // one low-priority victim, one high-priority protected
    { CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=400; o.tenant=0; o.priority=0; o.prompt_len=6; o.decode_len=2; o.chunk_max=3; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=401; o.tenant=1; o.priority=200; o.prompt_len=6; o.decode_len=2; o.chunk_max=3; ops.push_back(o); }
    for (int k = 0; k < 12; ++k) {
        CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=5; o.b=3; ops.push_back(o);
        if (k == 4) { CpsOp r2 = mk(CPS_OP_SET_KV_CAP, idx++); r2.a=2; ops.push_back(r2); }
        if (k == 8) { CpsOp r3 = mk(CPS_OP_SET_KV_CAP, idx++); r3.a=10; ops.push_back(r3); }
    }
    return sc;
}

// Scenario 4: MoE expert pressure - many tokens through few experts to force
// primary->secondary->drop, with progress still advancing.
static Scenario s_moe_pressure() {
    Scenario sc; sc.name = "moe_pressure";
    sc.spec = make_spec(2, 200, 16, 16, 8, 3, 0xDEADBEEF99ULL,
                        {1000,1000}, {1000,1000}, {1,1,1}, 64);
    int idx = 0; auto& ops = sc.ops;
    for (uint64_t r = 0; r < 6; ++r) {
        CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=500+r; o.tenant=(uint32_t)(r%2);
        o.priority=(uint32_t)(5+r); o.prompt_len=20; o.decode_len=8; o.chunk_max=16; ops.push_back(o);
    }
    // big iterations: many tokens vs expert_capacity total of 3 per iter
    for (int k = 0; k < 10; ++k) { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=40; o.b=8; ops.push_back(o); }
    return sc;
}

// Scenario 5: single expert (secondary == primary path), single tenant.
static Scenario s_single_expert() {
    Scenario sc; sc.name = "single_expert_tenant";
    sc.spec = make_spec(1, 50, 8, 5, 4, 1, 0x1111222233ULL,
                        {1000}, {1000}, {3}, 48);
    int idx = 0; auto& ops = sc.ops;
    for (uint64_t r = 0; r < 5; ++r) {
        CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=600+r; o.tenant=0;
        o.priority=(uint32_t)(r); o.prompt_len=9; o.decode_len=5; o.chunk_max=3; ops.push_back(o);
    }
    for (int k = 0; k < 12; ++k) { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=10; o.b=4; ops.push_back(o); }
    return sc;
}

// Scenario 6: ARRIVE_REJECT when table is full + cancellation freeing slots.
static Scenario s_table_full() {
    Scenario sc; sc.name = "table_full_reject";
    sc.spec = make_spec(2, 100, 4, 4, 4, 2, 0x7777aaaabbbbULL,
                        {1000,1000}, {1000,1000}, {2,2}, 64);
    int idx = 0; auto& ops = sc.ops;
    for (uint64_t r = 0; r < 7; ++r) {  // 7 arrivals but maxlive=4 -> 3 rejects
        CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=700+r; o.tenant=(uint32_t)(r%2);
        o.priority=(uint32_t)(r%5); o.prompt_len=3; o.decode_len=2; o.chunk_max=2; ops.push_back(o);
    }
    { CpsOp o = mk(CPS_OP_CANCEL, idx++); o.request_id=700; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id=710; o.tenant=0; o.priority=4; o.prompt_len=3; o.decode_len=1; o.chunk_max=2; ops.push_back(o); } // now fits
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=20; o.b=4; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=20; o.b=4; ops.push_back(o); }
    { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a=20; o.b=4; ops.push_back(o); }
    return sc;
}

// Scenario 7b: tiny-KV eviction churn with re-pop coupling. Many requests,
// KV capacity barely > one chunk, repeatedly forcing already-scheduled
// requests to stay protected while same-iteration re-pops occur.
static Scenario s_kv_churn_repop() {
    Scenario sc; sc.name = "kv_churn_repop";
    sc.spec = make_spec(3, 4, 16, 3, 16, 3, 0x2468ACE0ULL,
                        {1000,1000,1000}, {1000,1000,1000}, {2,2,2}, 120);
    int idx = 0; auto& ops = sc.ops;
    for (uint64_t r = 0; r < 12; ++r) {
        CpsOp o = mk(CPS_OP_ARRIVE, idx++); o.request_id = 800 + r;
        o.tenant = (uint32_t)(r % 3); o.priority = (uint32_t)(r % 4);
        o.prompt_len = 5 + (r % 4); o.decode_len = 3; o.chunk_max = 3; ops.push_back(o);
    }
    for (int k = 0; k < 16; ++k) {
        CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a = 30; o.b = 16; ops.push_back(o);
        if (k == 3)  { CpsOp r2 = mk(CPS_OP_SET_KV_CAP, idx++); r2.a = 2;  ops.push_back(r2); }
        if (k == 7)  { CpsOp r3 = mk(CPS_OP_SET_KV_CAP, idx++); r3.a = 6;  ops.push_back(r3); }
        if (k == 11) { CpsOp r4 = mk(CPS_OP_SET_KV_CAP, idx++); r4.a = 1;  ops.push_back(r4); }
    }
    { CpsOp o = mk(CPS_OP_SET_KV_CAP, idx++); o.a = 64; ops.push_back(o); }
    for (int k = 0; k < 6; ++k) { CpsOp o = mk(CPS_OP_STEP_ITER, idx++); o.a = 50; o.b = 16; ops.push_back(o); }
    return sc;
}

// Scenario 7: large randomized fuzz with all op types interleaved.
static Scenario s_random_fuzz(uint64_t seed, const std::string& name) {
    Scenario sc; sc.name = name;
    sc.spec = make_spec(4, 30, 24, 6, 8, 4, seed ^ 0xF00DULL,
                        {50,50,50,50}, {8,4,12,0}, {2,3,1,2}, 200);
    SplitMix64 rng(seed);
    int idx = 0; auto& ops = sc.ops;
    uint64_t next_rid = 1000;
    int n = 90;
    for (int i = 0; i < n; ++i) {
        int roll = (int)(rng.next_u64() % 100);
        if (roll < 30) {
            CpsOp o = mk(CPS_OP_ARRIVE, idx++);
            o.request_id = next_rid++;
            // occasionally reuse an old id to trigger dup-invalid
            if (rng.chance(150)) o.request_id = 1000 + rng.range_u64(0, 20);
            o.tenant = (uint32_t)rng.range_u64(0, rng.chance(100) ? 6 : 3); // sometimes bad tenant
            o.priority = (uint32_t)rng.range_u64(0, 255);
            o.prompt_len = rng.chance(200) ? 0 : rng.range_u64(0, 18);
            o.decode_len = rng.chance(200) ? 0 : rng.range_u64(0, 10);
            o.chunk_max = rng.chance(250) ? 0 : rng.range_u64(0, 7);
            ops.push_back(o);
        } else if (roll < 45) {
            CpsOp o = mk(CPS_OP_REFILL_TENANT, idx++);
            o.tenant = (uint32_t)rng.range_u64(0, rng.chance(80) ? 6 : 3);
            o.a = rng.range_u64(0, 40);
            ops.push_back(o);
        } else if (roll < 55) {
            CpsOp o = mk(CPS_OP_SET_KV_CAP, idx++);
            o.a = rng.range_u64(0, 40);
            ops.push_back(o);
        } else if (roll < 65) {
            CpsOp o = mk(CPS_OP_CANCEL, idx++);
            o.request_id = 1000 + rng.range_u64(0, 40);
            ops.push_back(o);
        } else {
            CpsOp o = mk(CPS_OP_STEP_ITER, idx++);
            o.a = rng.chance(120) ? 0 : rng.range_u64(0, 25);
            o.b = rng.chance(120) ? 0 : rng.range_u64(0, 10);
            ops.push_back(o);
        }
    }
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(s_pipeline_pressure());
    v.push_back(s_throttle_starvation());
    v.push_back(s_eviction_tiebreaks());
    v.push_back(s_moe_pressure());
    v.push_back(s_single_expert());
    v.push_back(s_table_full());
    v.push_back(s_kv_churn_repop());
    v.push_back(s_random_fuzz(0x1234567ULL, "fuzz_A"));
    v.push_back(s_random_fuzz(0x9abcdefULL, "fuzz_B"));
    v.push_back(s_random_fuzz(0xCAFEF00DULL, "fuzz_C"));
    v.push_back(s_random_fuzz(0x0BADC0DEULL, "fuzz_D"));
    v.push_back(s_random_fuzz(0x13579BDFULL, "fuzz_E"));
    return v;
}

// ---------------- driver ----------------

struct StepSnapshot {
    CpsHostOutputs out;
};

static bool run_one_op(const CpsProblemSpec& spec, const CpsOp& op, CpsOracle* oracle,
                       void* state, void* workspace, size_t workspace_bytes,
                       cudaStream_t stream, CpsHostOutputs* snap, std::string* err) {
    // upload op into a device buffer to verify immutability afterwards
    DevBuf<CpsOp> d_op; d_op.allocate(1);
    std::vector<CpsOp> hop(1, op); d_op.upload(hop);

    GuardedBuf<CpsCounts> g_counts; g_counts.allocate(1);
    GuardedBuf<uint64_t> g_batch, g_moe, g_fin, g_queue, g_req, g_buck, g_scal;
    g_batch.allocate(1); g_moe.allocate(1); g_fin.allocate(1);
    g_queue.allocate(1); g_req.allocate(1); g_buck.allocate(1); g_scal.allocate(1);

    CpsOutputs outputs{};
    outputs.counts = g_counts.ptr;
    outputs.batch_hash = g_batch.ptr;
    outputs.moe_hash = g_moe.ptr;
    outputs.finalize_hash = g_fin.ptr;
    outputs.queue_hash = g_queue.ptr;
    outputs.request_hash = g_req.ptr;
    outputs.bucket_hash = g_buck.ptr;
    outputs.scalar_hash = g_scal.ptr;

    // Pass the (possibly device-resident copy of the) op by host pointer per ABI.
    CUDA_CHECK(solution_run(state, &op, nullptr, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // immutability: op host struct must be unchanged by the call
    if (std::memcmp(&op, &hop[0], sizeof(CpsOp)) != 0) {
        if (err) *err = "op host struct mutated by solution_run";
        return false;
    }

    if (!g_counts.check_guards("counts", err)) return false;
    if (!g_batch.check_guards("batch_hash", err)) return false;
    if (!g_moe.check_guards("moe_hash", err)) return false;
    if (!g_fin.check_guards("finalize_hash", err)) return false;
    if (!g_queue.check_guards("queue_hash", err)) return false;
    if (!g_req.check_guards("request_hash", err)) return false;
    if (!g_buck.check_guards("bucket_hash", err)) return false;
    if (!g_scal.check_guards("scalar_hash", err)) return false;

    CpsHostOutputs got{};
    got.counts = g_counts.download()[0];
    got.batch_hash = g_batch.download()[0];
    got.moe_hash = g_moe.download()[0];
    got.finalize_hash = g_fin.download()[0];
    got.queue_hash = g_queue.download()[0];
    got.request_hash = g_req.download()[0];
    got.bucket_hash = g_buck.download()[0];
    got.scalar_hash = g_scal.download()[0];

    CpsExpected expected;
    oracle->step_op(op, &expected);

    if (!cps_check_outputs(expected, got, err)) return false;

    if (snap) *snap = got;
    (void)spec;
    return true;
}

static bool run_scenario(const Scenario& sc, bool verbose, std::vector<StepSnapshot>* snaps,
                         int* passed, int* total, std::string* first_err) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DevBuf<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    CpsOracle oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (snaps) { snaps->clear(); snaps->reserve(sc.ops.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.ops.size(); ++i) {
        StepSnapshot snap; std::string err;
        bool ok = run_one_op(sc.spec, sc.ops[i], &oracle, state,
                             workspace.ptr, workspace_bytes, stream,
                             snaps ? &snap.out : nullptr, &err);
        ++(*total);
        if (ok) ++(*passed);
        else {
            all_ok = false;
            if (first_err && first_err->empty()) {
                std::ostringstream o; o << sc.name << " op " << i
                    << " (opcode " << sc.ops[i].opcode << "): " << err;
                *first_err = o.str();
            }
        }
        if (snaps) snaps->push_back(snap);
        if (verbose && !ok) {
            std::printf("  scenario %-22s op %03zu opcode=%d FAIL  %s\n",
                        sc.name.c_str(), i, sc.ops[i].opcode, err.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_snaps(const std::vector<StepSnapshot>& a, const std::vector<StepSnapshot>& b,
                          std::string* err) {
    if (a.size() != b.size()) { if (err) *err = "snap length mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        const CpsHostOutputs& x = a[i].out; const CpsHostOutputs& y = b[i].out;
        if (std::memcmp(&x.counts, &y.counts, sizeof(CpsCounts)) != 0 ||
            x.batch_hash != y.batch_hash || x.moe_hash != y.moe_hash ||
            x.finalize_hash != y.finalize_hash || x.queue_hash != y.queue_hash ||
            x.request_hash != y.request_hash || x.bucket_hash != y.bucket_hash ||
            x.scalar_hash != y.scalar_hash) {
            if (err) { std::ostringstream o; o << "replay mismatch at op " << i; *err = o.str(); }
            return false;
        }
    }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        const std::vector<Scenario> scenarios = build_scenarios();

        int passed = 0, total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepSnapshot> base, replay;
            std::string err;
            bool ok_base = run_scenario(sc, true, &base, &passed, &total, &err);
            bool ok_replay = run_scenario(sc, false, &replay, &passed, &total, &err);

            if (ok_base && ok_replay) {
                std::string cerr;
                if (compare_snaps(base, replay, &cerr))
                    std::printf("scenario %-22s ops=%zu  PASS + exact replay PASS\n",
                                sc.name.c_str(), sc.ops.size());
                else { all_ok = false; std::printf("scenario %-22s replay FAIL %s\n", sc.name.c_str(), cerr.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-22s FAIL  %s\n", sc.name.c_str(), err.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
