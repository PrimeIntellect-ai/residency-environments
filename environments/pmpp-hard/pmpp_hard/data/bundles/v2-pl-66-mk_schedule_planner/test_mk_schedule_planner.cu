// file: test_mk_schedule_planner.cu
//
// 3-way validation harness for MK7. Each scenario is run against the device
// solution (reference or naive, linked separately) and compared step-by-step
// against the host oracle. Each scenario is also replayed a second time from a
// fresh state to confirm exact determinism. Output buffers are guarded and the
// input op buffer is checked for immutability.

#include "mk_schedule_planner_common.h"
#include "mk_schedule_planner_oracle.hpp"

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

static constexpr uint64_t g_seed = 0x9a17b3cd5e0f4421ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
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
    int uniform_int(int lo, int hi) {
        return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1));
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
        count = n; data_bytes = sizeof(T) * count; total_bytes = kGuardBytes + data_bytes + kGuardBytes;
        CUDA_CHECK(cudaMalloc((void**)&raw, total_bytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, total_bytes));
        ptr = (T*)(raw + kGuardBytes);
    }
    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* err) const {
        std::vector<uint8_t> before(kGuardBytes), after(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(before.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (before[i] != kGuardByte) { if (err) { std::ostringstream o; o<<"left guard "<<name<<" byte "<<i; *err=o.str(); } return false; }
            if (after[i] != kGuardByte) { if (err) { std::ostringstream o; o<<"right guard "<<name<<" byte "<<i; *err=o.str(); } return false; }
        }
        return true;
    }
};

struct StepHost {
    MkRunSpec run;
    std::vector<MkOp> ops;
};

struct StepResult {
    MkExpected vals;
};

struct Scenario {
    std::string name;
    MkProblemSpec spec;
    std::vector<StepHost> steps;
};

static MkProblemSpec make_spec(int sm_count, int pages_per_sm, uint64_t wave_quantum,
                               int max_instrs, int max_edges, int max_ops, int max_steps) {
    MkProblemSpec s = {};
    s.abi_version = MK_ABI_VERSION;
    s.sm_count = sm_count;
    s.pages_per_sm = pages_per_sm;
    s.wave_quantum = wave_quantum;
    s.max_instrs = max_instrs;
    s.max_edges = max_edges;
    s.max_ops_per_step = max_ops;
    s.max_steps = max_steps;
    s.flags = 0;
    if (!mk_validate_problem_spec(&s)) throw std::runtime_error("invalid problem spec");
    return s;
}

// op builders
static MkOp mk_add_instr(uint64_t id, uint64_t dur, uint64_t pcnt,
                         const std::vector<uint64_t>& keys, uint64_t rdelay, uint64_t seed) {
    MkOp o = {}; o.op_type = MK_OP_ADD_INSTR; o.id = id; o.a = dur; o.b = pcnt; o.c = rdelay; o.d = seed;
    for (size_t k = 0; k < keys.size() && k < MK_MAX_PAGE_KEYS_PER_INSTR; ++k) o.page_keys[k] = keys[k];
    return o;
}
static MkOp mk_add_edge(uint64_t eid, uint64_t src, uint64_t dst, uint32_t chunk, uint64_t tinc) {
    MkOp o = {}; o.op_type = MK_OP_ADD_EDGE; o.id = eid; o.src = src; o.dst = dst; o.a = chunk; o.b = tinc; return o;
}
static MkOp mk_plan_next(uint64_t limit) { MkOp o = {}; o.op_type = MK_OP_PLAN_NEXT; o.a = limit; return o; }
static MkOp mk_commit(uint64_t maxe) { MkOp o = {}; o.op_type = MK_OP_COMMIT_PLAN; o.a = maxe; return o; }
static MkOp mk_exec(uint64_t tick_limit, uint64_t max_events) {
    MkOp o = {}; o.op_type = MK_OP_EXECUTE_UNTIL; o.a = tick_limit; o.b = max_events; return o;
}
static MkOp mk_cancel(uint64_t id) { MkOp o = {}; o.op_type = MK_OP_CANCEL_INSTR; o.id = id; return o; }
static MkOp mk_new_epoch() { MkOp o = {}; o.op_type = MK_OP_NEW_EPOCH; return o; }

static StepHost make_step(const MkProblemSpec& spec, int step_id, const std::vector<MkOp>& ops) {
    StepHost s;
    s.run = {};
    s.run.abi_version = MK_ABI_VERSION;
    s.run.num_ops = (int)ops.size();
    s.run.step_id = step_id;
    if (!mk_validate_run_spec(&s.run, &spec)) throw std::runtime_error("invalid run spec");
    s.ops = ops;
    if (s.ops.empty()) s.ops.resize(1);  // keep a row for upload
    return s;
}

// ----- scenario builders -----

// 1) Basic linear chain: add instrs, chain edges, plan all (wave quantization),
//    commit, execute, signal edges.
static Scenario sc_linear_chain() {
    Scenario sc;
    sc.name = "linear_chain_plan_exec";
    sc.spec = make_spec(2, 4, 10, 64, 128, 64, 8);
    int sid = 0;
    {
        std::vector<MkOp> ops;
        ops.push_back(mk_add_instr(1, 7, 1, {100}, 3, 0xAA));
        ops.push_back(mk_add_instr(2, 5, 2, {100, 101}, 0, 0xBB));
        ops.push_back(mk_add_instr(3, 9, 1, {102}, 4, 0xCC));
        ops.push_back(mk_add_instr(4, 4, 1, {103}, 1, 0xDD));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    // chain 1->2->3->4
    {
        std::vector<MkOp> ops;
        ops.push_back(mk_add_edge(10, 1, 2, 0, 1));
        ops.push_back(mk_add_edge(11, 2, 3, 1, 2));
        ops.push_back(mk_add_edge(12, 3, 4, 2, 3));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(10)}));   // plan all
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_commit(100)}));     // commit all
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_exec(1000, 100)})); // execute all
    sc.steps.push_back(make_step(sc.spec, sid++, {}));                   // empty
    return sc;
}

// 2) Critical-height-driven ready ordering + plan_empty/stall edge.
static Scenario sc_crit_height_order() {
    Scenario sc;
    sc.name = "crit_height_ready_order";
    sc.spec = make_spec(1, 2, 8, 64, 128, 64, 10);
    int sid = 0;
    // diamond: 1 -> {2,3} -> 4 ; durations chosen so crit heights differ.
    {
        std::vector<MkOp> ops;
        ops.push_back(mk_add_instr(1, 3, 1, {1}, 0, 1));
        ops.push_back(mk_add_instr(2, 10, 1, {2}, 0, 2));
        ops.push_back(mk_add_instr(3, 4, 1, {3}, 0, 3));
        ops.push_back(mk_add_instr(4, 6, 1, {4}, 0, 4));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    {
        std::vector<MkOp> ops;
        ops.push_back(mk_add_edge(20, 1, 2, 0, 1));
        ops.push_back(mk_add_edge(21, 1, 3, 0, 1));
        ops.push_back(mk_add_edge(22, 2, 4, 0, 1));
        ops.push_back(mk_add_edge(23, 3, 4, 0, 1));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(1)}));  // plan root
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(1)}));  // plan higher-crit child (2)
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(2)}));  // plan 3 then 4
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(5)}));  // PLAN_EMPTY (none left)
    return sc;
}

// 3) Edge-add to PLANNED endpoint invalidates dst subtree.
static Scenario sc_invalidate_cascade() {
    Scenario sc;
    sc.name = "edge_invalidate_cascade";
    sc.spec = make_spec(2, 3, 5, 64, 128, 64, 12);
    int sid = 0;
    {
        std::vector<MkOp> ops;
        for (uint64_t i = 1; i <= 5; ++i)
            ops.push_back(mk_add_instr(i, 3 + i, 1, {10 + i}, (i % 2), (uint64_t)(0x10 * i)));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    // initial chain 1->2->3
    {
        std::vector<MkOp> ops;
        ops.push_back(mk_add_edge(30, 1, 2, 0, 1));
        ops.push_back(mk_add_edge(31, 2, 3, 0, 1));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(10)}));  // plan everything
    // now add edge 4->2 : 4 unplanned, 2 planned -> dst=2 subtree invalidated (2,3)
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_add_edge(32, 4, 2, 0, 1)}));
    // replan
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(10)}));
    // cycle attempt: 3->1 would create cycle -> invalid
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_add_edge(33, 3, 1, 0, 1)}));
    return sc;
}

// 4) Commit barrier: committed dst cannot accept new edges; NEW_EPOCH guarded.
static Scenario sc_commit_barrier_epoch() {
    Scenario sc;
    sc.name = "commit_barrier_epoch";
    sc.spec = make_spec(2, 2, 6, 64, 128, 64, 14);
    int sid = 0;
    {
        std::vector<MkOp> ops;
        ops.push_back(mk_add_instr(1, 5, 1, {1}, 2, 0xF0));
        ops.push_back(mk_add_instr(2, 7, 1, {2}, 0, 0xF1));
        ops.push_back(mk_add_instr(3, 3, 1, {3}, 1, 0xF2));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_add_edge(40, 1, 2, 0, 1)}));
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(10)}));
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_commit(2)}));  // commit 2 by plan_seq
    // add edge to committed dst -> invalid (dst could be 1 or 2 if committed)
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_add_edge(41, 3, 1, 0, 1)}));
    // NEW_EPOCH now invalid (committed not executed)
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_new_epoch()}));
    // execute committed
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_exec(100000, 100)}));
    // commit remaining then execute, then NEW_EPOCH valid
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(10), mk_commit(100), mk_exec(100000, 100)}));
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_new_epoch()}));
    // re-plan in new epoch
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(10)}));
    return sc;
}

// 5) Cancel: planned-subtree removal + incident-edge removal.
static Scenario sc_cancel_paths() {
    Scenario sc;
    sc.name = "cancel_subtree_edges";
    sc.spec = make_spec(2, 4, 4, 64, 128, 64, 12);
    int sid = 0;
    {
        std::vector<MkOp> ops;
        for (uint64_t i = 1; i <= 6; ++i)
            ops.push_back(mk_add_instr(i, 2 + (i % 4), 1, {20 + i}, 0, (uint64_t)i));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    {
        std::vector<MkOp> ops;
        ops.push_back(mk_add_edge(50, 1, 2, 0, 1));
        ops.push_back(mk_add_edge(51, 2, 3, 0, 1));
        ops.push_back(mk_add_edge(52, 4, 5, 0, 1));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(10)}));  // plan all reachable
    // cancel 2 -> removes plan subtree (2,3) and edges 50,51
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_cancel(2)}));
    // invalid cancels
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_cancel(2) /*already cancelled*/, mk_cancel(999) /*absent*/}));
    // replan after cancel
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(10)}));
    // commit + execute + cancel-executed invalid
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_commit(100), mk_exec(100000, 100)}));
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_cancel(1) /*executed -> invalid*/}));
    return sc;
}

// 6) Page-pressure: many instrs sharing/conflicting page keys on few SMs force
//    wave bumping and page-interval constraints.
static Scenario sc_page_pressure() {
    Scenario sc;
    sc.name = "page_pressure_waves";
    sc.spec = make_spec(2, 2, 3, 64, 64, 64, 8);
    int sid = 0;
    {
        std::vector<MkOp> ops;
        // 8 instrs, each 2 pages, long release_delay so pages stay live and
        // collide; pages_per_sm=2 means each SM can hold at most 2 distinct keys.
        ops.push_back(mk_add_instr(1, 4, 2, {1, 2}, 5, 11));
        ops.push_back(mk_add_instr(2, 4, 2, {3, 4}, 5, 12));
        ops.push_back(mk_add_instr(3, 4, 2, {1, 2}, 5, 13));
        ops.push_back(mk_add_instr(4, 4, 2, {3, 4}, 5, 14));
        ops.push_back(mk_add_instr(5, 2, 1, {5}, 0, 15));
        ops.push_back(mk_add_instr(6, 6, 2, {5, 6}, 2, 16));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(10)}));
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_commit(100)}));
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_exec(100000, 3)}));   // partial execute
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_exec(100000, 100)})); // rest
    return sc;
}

// 7) Validity gauntlet: every invalid path.
static Scenario sc_validity_gauntlet() {
    Scenario sc;
    sc.name = "validity_gauntlet";
    sc.spec = make_spec(1, 2, 5, 4, 8, 32, 8);  // small caps: max_instrs=4
    int sid = 0;
    {
        std::vector<MkOp> ops;
        ops.push_back(mk_add_instr(1, 3, 1, {1}, 0, 1));
        ops.push_back(mk_add_instr(1, 3, 1, {1}, 0, 1));   // dup id -> invalid
        ops.push_back(mk_add_instr(2, 0, 1, {1}, 0, 1));   // zero duration -> invalid
        ops.push_back(mk_add_instr(3, 3, 0, {}, 0, 1));    // zero page_count -> invalid
        ops.push_back(mk_add_instr(4, 3, 3, {1,2,3}, 0, 1)); // page_count>pages_per_sm -> invalid
        ops.push_back(mk_add_instr(5, 3, 1, {1}, 0, 1));
        ops.push_back(mk_add_instr(6, 3, 1, {1}, 0, 1));
        ops.push_back(mk_add_instr(7, 3, 1, {1}, 0, 1));   // table full (max=4) -> invalid
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    {
        std::vector<MkOp> ops;
        ops.push_back(mk_add_edge(60, 1, 1, 0, 1));    // self -> invalid
        ops.push_back(mk_add_edge(61, 1, 99, 0, 1));   // absent dst -> invalid
        ops.push_back(mk_add_edge(62, 1, 5, 0, 0));    // tinc 0 -> invalid
        ops.push_back(mk_add_edge(63, 1, 5, 0, 1));    // valid
        ops.push_back(mk_add_edge(63, 1, 6, 0, 1));    // dup edge id -> invalid
        ops.push_back(mk_add_edge(64, 5, 1, 0, 1));    // cycle (1->5 then 5->1) -> invalid
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_commit(5) /*nothing planned -> no-op*/}));
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_exec(0, 0) /*max_events 0 -> no-op*/}));
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_plan_next(0) /*limit 0 no-op*/}));
    return sc;
}

// 8) Randomized adversarial multi-op stream.
static Scenario sc_random_stream() {
    Scenario sc;
    sc.name = "random_adversarial_stream";
    sc.spec = make_spec(4, 4, 7, 128, 256, 48, 24);
    SplitMix64 rng(g_seed ^ 0xfeed1234ULL);
    uint64_t next_instr = 1;
    uint64_t next_edge = 1;
    std::vector<uint64_t> live_ids;
    int sid = 0;
    for (int s = 0; s < 24; ++s) {
        std::vector<MkOp> ops;
        int n = rng.uniform_int(0, 30);
        for (int i = 0; i < n; ++i) {
            int r = rng.uniform_int(0, 99);
            if (r < 35) {
                uint64_t dur = (uint64_t)rng.uniform_int(1, 12);
                int pc = rng.uniform_int(1, 4);  // <= pages_per_sm
                std::vector<uint64_t> keys;
                for (int k = 0; k < pc; ++k) keys.push_back((uint64_t)rng.uniform_int(1, 8));
                ops.push_back(mk_add_instr(next_instr, dur, (uint64_t)pc, keys,
                                           (uint64_t)rng.uniform_int(0, 6),
                                           (uint64_t)rng.next_u64()));
                live_ids.push_back(next_instr);
                next_instr++;
            } else if (r < 55 && live_ids.size() >= 2) {
                uint64_t a = live_ids[rng.uniform_int(0, (int)live_ids.size()-1)];
                uint64_t b = live_ids[rng.uniform_int(0, (int)live_ids.size()-1)];
                ops.push_back(mk_add_edge(next_edge++, a, b, (uint32_t)rng.uniform_int(0, 7),
                                          (uint64_t)rng.uniform_int(1, 5)));
            } else if (r < 75) {
                ops.push_back(mk_plan_next((uint64_t)rng.uniform_int(0, 6)));
            } else if (r < 85) {
                ops.push_back(mk_commit((uint64_t)rng.uniform_int(0, 5)));
            } else if (r < 95) {
                ops.push_back(mk_exec((uint64_t)rng.uniform_int(0, 200),
                                      (uint64_t)rng.uniform_int(0, 6)));
            } else if (r < 98 && !live_ids.empty()) {
                ops.push_back(mk_cancel(live_ids[rng.uniform_int(0, (int)live_ids.size()-1)]));
            } else {
                ops.push_back(mk_new_epoch());
            }
        }
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    return sc;
}

// 9) Dense random graph with heavy cancel/epoch churn and tight page budget on
//    a single SM (maximizes coupling between wave-quantization and pages).
static Scenario sc_dense_single_sm() {
    Scenario sc;
    sc.name = "dense_single_sm_churn";
    sc.spec = make_spec(1, 2, 4, 96, 256, 40, 20);
    SplitMix64 rng(g_seed ^ 0x0bad5eedULL);
    uint64_t next_instr = 1, next_edge = 1;
    std::vector<uint64_t> ids;
    int sid = 0;
    for (int s = 0; s < 20; ++s) {
        std::vector<MkOp> ops;
        int n = rng.uniform_int(0, 25);
        for (int i = 0; i < n; ++i) {
            int r = rng.uniform_int(0, 99);
            if (r < 40) {
                int pc = rng.uniform_int(1, 2);  // <= pages_per_sm=2
                std::vector<uint64_t> keys;
                for (int k = 0; k < pc; ++k) keys.push_back((uint64_t)rng.uniform_int(1, 5));
                ops.push_back(mk_add_instr(next_instr, (uint64_t)rng.uniform_int(1, 9),
                                           (uint64_t)pc, keys,
                                           (uint64_t)rng.uniform_int(0, 8),
                                           rng.next_u64()));
                ids.push_back(next_instr++);
            } else if (r < 58 && ids.size() >= 2) {
                uint64_t a = ids[rng.uniform_int(0, (int)ids.size()-1)];
                uint64_t b = ids[rng.uniform_int(0, (int)ids.size()-1)];
                ops.push_back(mk_add_edge(next_edge++, a, b,
                                          (uint32_t)rng.uniform_int(0, 9),
                                          (uint64_t)rng.uniform_int(1, 6)));
            } else if (r < 76) {
                ops.push_back(mk_plan_next((uint64_t)rng.uniform_int(0, 8)));
            } else if (r < 86) {
                ops.push_back(mk_commit((uint64_t)rng.uniform_int(0, 4)));
            } else if (r < 94) {
                ops.push_back(mk_exec((uint64_t)rng.uniform_int(0, 150),
                                      (uint64_t)rng.uniform_int(0, 5)));
            } else if (r < 98 && !ids.empty()) {
                ops.push_back(mk_cancel(ids[rng.uniform_int(0, (int)ids.size()-1)]));
            } else {
                ops.push_back(mk_new_epoch());
            }
        }
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    return sc;
}

// --------------------------------------------------------------------------

static bool run_one_step(const MkProblemSpec& spec, const StepHost& step,
                         MkOracleState* oracle, void* state,
                         void* workspace, size_t workspace_bytes,
                         cudaStream_t stream, StepResult* result, std::string* error) {
    DeviceBuffer<MkOp> d_ops;
    d_ops.allocate(step.ops.size());
    d_ops.upload(step.ops);

    GuardedDeviceBuffer<uint64_t> g[20];
    for (int i = 0; i < 20; ++i) g[i].allocate(1);

    MkInputs inputs = {};
    inputs.ops = d_ops.ptr;

    MkOutputs out = {};
    out.instr_added = g[0].ptr; out.edge_added = g[1].ptr; out.plan_invalidated = g[2].ptr;
    out.plan_empty = g[3].ptr; out.plan_stall = g[4].ptr; out.plan_placed = g[5].ptr;
    out.plan_committed = g[6].ptr; out.instr_executed = g[7].ptr; out.edge_signaled = g[8].ptr;
    out.instr_cancelled = g[9].ptr; out.epoch_advanced = g[10].ptr; out.invalid_count = g[11].ptr;
    out.planner_event_hash = g[12].ptr;
    out.plan_hash = g[13].ptr; out.instr_hash = g[14].ptr; out.edge_hash = g[15].ptr;
    out.page_interval_hash = g[16].ptr;
    out.committed_epoch = g[17].ptr; out.live_instr_count = g[18].ptr;
    out.planned_interval_count = g[19].ptr;

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &out, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // input immutability
    {
        std::vector<MkOp> got_ops = d_ops.download();
        if (got_ops.size() != step.ops.size() ||
            (got_ops.size() &&
             std::memcmp(got_ops.data(), step.ops.data(), sizeof(MkOp) * got_ops.size()) != 0)) {
            if (error) *error = "input ops modified";
            return false;
        }
    }

    static const char* names[20] = {
        "instr_added","edge_added","plan_invalidated","plan_empty","plan_stall","plan_placed",
        "plan_committed","instr_executed","edge_signaled","instr_cancelled","epoch_advanced",
        "invalid_count","planner_event_hash","plan_hash","instr_hash","edge_hash",
        "page_interval_hash","committed_epoch","live_instr_count","planned_interval_count"
    };
    for (int i = 0; i < 20; ++i)
        if (!g[i].check_guards(names[i], error)) return false;

    uint64_t v[20];
    for (int i = 0; i < 20; ++i) v[i] = g[i].download_data()[0];

    MkExpected expected;
    oracle->step_once(step.run, step.ops.data(), &expected);

    MkHostOutputsView got = {};
    got.instr_added = &v[0]; got.edge_added = &v[1]; got.plan_invalidated = &v[2];
    got.plan_empty = &v[3]; got.plan_stall = &v[4]; got.plan_placed = &v[5];
    got.plan_committed = &v[6]; got.instr_executed = &v[7]; got.edge_signaled = &v[8];
    got.instr_cancelled = &v[9]; got.epoch_advanced = &v[10]; got.invalid_count = &v[11];
    got.planner_event_hash = &v[12];
    got.plan_hash = &v[13]; got.instr_hash = &v[14]; got.edge_hash = &v[15];
    got.page_interval_hash = &v[16];
    got.committed_epoch = &v[17]; got.live_instr_count = &v[18];
    got.planned_interval_count = &v[19];

    if (!mk_check_all_outputs(expected, got, error)) return false;

    if (result) result->vals = expected;
    return true;
}

static bool run_scenario_once(const Scenario& sc, bool verbose,
                              std::vector<StepResult>* results,
                              int* passed, int* total, std::string* first_error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    MkOracleState oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.steps.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult result;
        std::string error;
        bool ok = run_one_step(sc.spec, sc.steps[i], &oracle, state,
                               workspace.ptr, std::max<size_t>(workspace_bytes, 1), stream,
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
            std::printf("scenario %-28s step %02zu/%02zu ops=%d %s%s\n",
                        sc.name.c_str(), i, sc.steps.size(), sc.steps[i].run.num_ops,
                        ok ? "PASS" : "FAIL ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(const std::vector<StepResult>& a, const std::vector<StepResult>& b, std::string* err) {
    if (a.size() != b.size()) { if (err) *err = "result length mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        const MkExpected& x = a[i].vals; const MkExpected& y = b[i].vals;
        if (x.planner_event_hash != y.planner_event_hash || x.plan_hash != y.plan_hash ||
            x.instr_hash != y.instr_hash || x.edge_hash != y.edge_hash ||
            x.page_interval_hash != y.page_interval_hash ||
            x.committed_epoch != y.committed_epoch ||
            x.live_instr_count != y.live_instr_count ||
            x.planned_interval_count != y.planned_interval_count) {
            if (err) { std::ostringstream o; o << "replay mismatch at step " << i; *err = o.str(); }
            return false;
        }
    }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        std::vector<Scenario> scenarios;
        scenarios.push_back(sc_linear_chain());
        scenarios.push_back(sc_crit_height_order());
        scenarios.push_back(sc_invalidate_cascade());
        scenarios.push_back(sc_commit_barrier_epoch());
        scenarios.push_back(sc_cancel_paths());
        scenarios.push_back(sc_page_pressure());
        scenarios.push_back(sc_validity_gauntlet());
        scenarios.push_back(sc_random_stream());
        scenarios.push_back(sc_dense_single_sm());

        int passed = 0, total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base, replay;
            std::string error;
            bool ok_base = run_scenario_once(sc, true, &base, &passed, &total, &error);
            bool ok_replay = run_scenario_once(sc, false, &replay, &passed, &total, &error);
            if (ok_base && ok_replay) {
                std::string ce;
                if (compare_results(base, replay, &ce))
                    std::printf("scenario %-28s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-28s exact replay FAIL %s\n", sc.name.c_str(), ce.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-28s FAIL %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
