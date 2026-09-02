// file: test_work_stealing_runtime.cu

#include "work_stealing_runtime_common.h"
#include "work_stealing_runtime_oracle.hpp"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0x51d9f0c3a7b2e641ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                              \
        }                                                                       \
    } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
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
        if (n > 0) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n));
    }
};

template <typename T>
struct GuardedDeviceBuffer {
    uint8_t* raw = nullptr;
    T* ptr = nullptr;
    size_t count = 0;
    size_t data_bytes = 0;
    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;
    ~GuardedDeviceBuffer() { if (raw) cudaFree(raw); }
    void allocate(size_t n) {
        count = n;
        data_bytes = sizeof(T) * count;
        CUDA_CHECK(cudaMalloc((void**)&raw, kGuardBytes + data_bytes + kGuardBytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, kGuardBytes + data_bytes + kGuardBytes));
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }
    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> left(kGuardBytes), right(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(left.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(right.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (left[i] != kGuardByte) { if (error) { std::ostringstream o; o << "left guard corrupted for " << name << " at byte " << i; *error = o.str(); } return false; }
            if (right[i] != kGuardByte) { if (error) { std::ostringstream o; o << "right guard corrupted for " << name << " at byte " << i; *error = o.str(); } return false; }
        }
        return true;
    }
};

struct StepResult {
    std::vector<int64_t> counts;
    int32_t op_index = 0;
    uint64_t clock = 0, event_seq = 0, sched = 0, ready = 0, running = 0, blocked = 0, sleep = 0, state = 0;
};

struct Scenario {
    std::string name;
    WsrProblemSpec spec;
    std::vector<WsrRunSpec> ops;
};

static WsrProblemSpec make_spec(int W, int lcap, int gcap, int max_tasks, int max_blocked, int max_sleeping, int max_steps) {
    WsrProblemSpec s = {};
    s.abi_version = WSR_ABI_VERSION;
    s.W = W; s.local_cap_per_worker = lcap; s.global_cap = gcap;
    s.max_tasks = max_tasks; s.max_blocked = max_blocked; s.max_sleeping = max_sleeping;
    s.max_steps = max_steps; s.flags = 0;
    if (!wsr_validate_problem_spec(&s)) throw std::runtime_error("invalid WsrProblemSpec");
    return s;
}

static WsrRunSpec mk(int op_kind, int step_id) {
    WsrRunSpec r = {};
    r.abi_version = WSR_ABI_VERSION;
    r.op_kind = op_kind;
    r.step_id = step_id;
    return r;
}
static WsrRunSpec op_spawn(int step, uint64_t tid, int home, int prio, uint64_t work) {
    WsrRunSpec r = mk(WSR_OP_SPAWN, step); r.a_task = tid; r.a_worker = home; r.a_priority = prio; r.a_work = work; return r;
}
static WsrRunSpec op_run(int step, int worker, uint64_t quantum) {
    WsrRunSpec r = mk(WSR_OP_RUN, step); r.a_worker = worker; r.a_work = quantum; return r;
}
static WsrRunSpec op_yield(int step, int worker) { WsrRunSpec r = mk(WSR_OP_YIELD, step); r.a_worker = worker; return r; }
static WsrRunSpec op_block(int step, int worker, uint64_t key) { WsrRunSpec r = mk(WSR_OP_BLOCK, step); r.a_worker = worker; r.a_key = key; return r; }
static WsrRunSpec op_sleep(int step, int worker, uint64_t tick) { WsrRunSpec r = mk(WSR_OP_SLEEP, step); r.a_worker = worker; r.a_tick = tick; return r; }
static WsrRunSpec op_wake(int step, uint64_t key, int limit) { WsrRunSpec r = mk(WSR_OP_WAKE, step); r.a_key = key; r.a_limit = limit; return r; }
static WsrRunSpec op_advance(int step, uint64_t delta) { WsrRunSpec r = mk(WSR_OP_ADVANCE, step); r.a_delta = delta; return r; }
static WsrRunSpec op_cancel(int step, uint64_t tid) { WsrRunSpec r = mk(WSR_OP_CANCEL, step); r.a_task = tid; return r; }

// ------------------------------------------------------------- scenarios
// S1: LIFO local vs FIFO global, with completion + id reuse.
static Scenario sc_lifo_fifo() {
    Scenario s; s.name = "lifo_local_fifo_global";
    s.spec = make_spec(2, 4, 8, 16, 8, 8, 256);
    int t = 0;
    // local cap 4 -> first 4 to local high, next overflow to global high.
    for (uint64_t id = 1; id <= 6; ++id) s.ops.push_back(op_spawn(t++, id, 0, 1, 3));
    // run worker0: pops local tail (LIFO) first 4 ids in reverse, then global head (FIFO).
    for (int i = 0; i < 8; ++i) { s.ops.push_back(op_run(t++, 0, 1)); s.ops.push_back(op_run(t++, 0, 5)); }
    // reuse completed ids
    for (uint64_t id = 1; id <= 4; ++id) s.ops.push_back(op_spawn(t++, id, 1, 0, 2));
    for (int i = 0; i < 10; ++i) s.ops.push_back(op_run(t++, 1, 1));
    s.ops.push_back(op_run(t++, 1, 9));  // idle
    return s;
}

// S2: half-steal (ceil) with victim tie-break (largest count, lowest id).
static Scenario sc_steal() {
    Scenario s; s.name = "half_steal_tiebreak";
    s.spec = make_spec(4, 16, 32, 64, 16, 16, 256);
    int t = 0;
    // worker1 gets 5 high tasks, worker2 gets 5 low tasks (tie in count -> pick lowest id =1).
    for (uint64_t id = 10; id < 15; ++id) s.ops.push_back(op_spawn(t++, id, 1, 1, 4));
    for (uint64_t id = 20; id < 25; ++id) s.ops.push_back(op_spawn(t++, id, 2, 0, 4));
    // worker3 has nothing -> steals from worker1 (tie broken to lowest id 1). ceil(5/2)=3.
    s.ops.push_back(op_run(t++, 3, 2));
    // give worker2 an extra so it is strictly largest, then worker0 steals from worker2.
    s.ops.push_back(op_spawn(t++, 30, 2, 0, 4));
    s.ops.push_back(op_run(t++, 0, 2));
    // drain everything via repeated runs across workers.
    for (int r = 0; r < 30; ++r) s.ops.push_back(op_run(t++, r % 4, 4));
    return s;
}

// S3: sleep + ADVANCE batch wake ordering, with sleep overflow cancel.
static Scenario sc_sleep_advance() {
    Scenario s; s.name = "sleep_advance_order";
    s.spec = make_spec(2, 8, 16, 32, 16, 4, 256);
    int t = 0;
    for (uint64_t id = 1; id <= 6; ++id) s.ops.push_back(op_spawn(t++, id, 0, 1, 5));
    // schedule + sleep each with various wake_ticks (out of order).
    uint64_t ticks[6] = {50, 10, 30, 10, 70, 20};
    for (int i = 0; i < 6; ++i) {
        s.ops.push_back(op_run(t++, 0, 1));       // schedule+slice id
        s.ops.push_back(op_sleep(t++, 0, ticks[i]));  // 5th sleep overflows (max_sleeping=4)
    }
    s.ops.push_back(op_advance(t++, 15));  // wakes tick<=15: ids with 10,10 in (tick,seq,id) order
    s.ops.push_back(op_advance(t++, 20));  // clock=35 wakes 20,30
    s.ops.push_back(op_advance(t++, 100)); // clock=135 wakes rest
    for (int r = 0; r < 12; ++r) s.ops.push_back(op_run(t++, r % 2, 5));
    return s;
}

// S4: block FIFO + WAKE limit + WAKE_STALLED on global overflow.
static Scenario sc_block_wake_stall() {
    Scenario s; s.name = "block_wake_stall";
    // local cap 1, global cap 2 so wake placement overflows quickly.
    s.spec = make_spec(2, 1, 2, 32, 16, 16, 512);
    int t = 0;
    // Spawn 6 tasks home worker0 prio0 (cap1 local -> 1 local, 2 global, rest reject).
    for (uint64_t id = 1; id <= 6; ++id) s.ops.push_back(op_spawn(t++, id, 0, 0, 4));
    // schedule + block a few on key 7 and key 5.
    s.ops.push_back(op_run(t++, 0, 1)); s.ops.push_back(op_block(t++, 0, 7));
    s.ops.push_back(op_run(t++, 0, 1)); s.ops.push_back(op_block(t++, 0, 7));
    s.ops.push_back(op_run(t++, 0, 1)); s.ops.push_back(op_block(t++, 0, 5));
    // wake key7 limit 1 -> one task placed.
    s.ops.push_back(op_wake(t++, 7, 1));
    // fill queues then wake to force WAKE_STALLED.
    for (uint64_t id = 10; id <= 14; ++id) s.ops.push_back(op_spawn(t++, id, 0, 0, 4));
    s.ops.push_back(op_wake(t++, 7, 5));  // should stall (queues full)
    s.ops.push_back(op_wake(t++, 99, 3)); // nonexistent key -> no-op
    s.ops.push_back(op_wake(t++, 5, 0));  // limit 0 no-op
    // drain
    for (int r = 0; r < 20; ++r) s.ops.push_back(op_run(t++, r % 2, 4));
    s.ops.push_back(op_wake(t++, 7, 5));
    for (int r = 0; r < 10; ++r) s.ops.push_back(op_run(t++, r % 2, 4));
    return s;
}

// S5: overflow cancellation (yield/block) + explicit cancel of every state.
static Scenario sc_overflow_cancel() {
    Scenario s; s.name = "overflow_and_cancel";
    s.spec = make_spec(2, 2, 2, 8, 1, 1, 512);  // tight caps
    int t = 0;
    // Fill worker0 local(2) + global(2) = 4 ready prio1.
    for (uint64_t id = 1; id <= 4; ++id) s.ops.push_back(op_spawn(t++, id, 0, 1, 6));
    s.ops.push_back(op_spawn(t++, 5, 0, 1, 6));  // table not full (max 8) but queues full -> reject
    // run+block: first block ok (max_blocked=1), second block overflow-cancels.
    s.ops.push_back(op_run(t++, 0, 1)); s.ops.push_back(op_block(t++, 0, 1));
    s.ops.push_back(op_run(t++, 0, 1)); s.ops.push_back(op_block(t++, 0, 1));  // overflow cancel
    // sleep overflow: first sleep ok (max_sleeping=1), then another sleep overflow.
    s.ops.push_back(op_run(t++, 0, 1)); s.ops.push_back(op_sleep(t++, 0, 100));
    s.ops.push_back(op_run(t++, 0, 1)); s.ops.push_back(op_sleep(t++, 0, 100)); // overflow cancel
    // explicit cancels: ready, blocked, sleeping, running, absent.
    s.ops.push_back(op_spawn(t++, 50, 1, 0, 3));
    s.ops.push_back(op_cancel(t++, 50));     // ready
    s.ops.push_back(op_cancel(t++, 1));      // blocked (still in key1)
    s.ops.push_back(op_cancel(t++, 999));    // absent -> invalid
    s.ops.push_back(op_run(t++, 1, 1));      // schedule something on w1? none -> idle
    // empty ops
    s.ops.push_back(op_yield(t++, 1));       // empty
    s.ops.push_back(op_block(t++, 1, 2));    // empty
    s.ops.push_back(op_sleep(t++, 1, 5));    // empty
    // invalid ops
    s.ops.push_back(op_run(t++, 9, 1));      // worker oob
    s.ops.push_back(op_run(t++, 0, 0));      // quantum 0
    s.ops.push_back(op_spawn(t++, 1, 0, 1, 0)); // work 0 invalid (and id1 was cancelled)
    s.ops.push_back(op_spawn(t++, 60, 9, 0, 1)); // home oob
    s.ops.push_back(op_spawn(t++, 61, 0, 2, 1)); // prio oob
    for (int r = 0; r < 16; ++r) s.ops.push_back(op_run(t++, r % 2, 6));
    return s;
}

// S6: yield re-enqueue ordering + clock wrap + mixed big random stress.
static Scenario sc_mixed_random() {
    Scenario s; s.name = "mixed_random_stress";
    s.spec = make_spec(8, 6, 24, 256, 64, 64, 1024);
    SplitMix64 rng(g_state ^ 0xfeedface1234ULL);
    int t = 0;
    uint64_t next_id = 1;
    // initial spawns
    for (int i = 0; i < 40; ++i) {
        s.ops.push_back(op_spawn(t++, next_id++, rng.uniform_int(0, 7), rng.uniform_int(0, 1), rng.uniform_int(1, 7)));
    }
    for (int i = 0; i < 600; ++i) {
        int pick = rng.uniform_int(0, 99);
        int w = rng.uniform_int(0, 7);
        if (pick < 38) {
            s.ops.push_back(op_run(t++, w, rng.uniform_int(1, 6)));
        } else if (pick < 50) {
            s.ops.push_back(op_spawn(t++, next_id++, rng.uniform_int(0, 7), rng.uniform_int(0, 1), rng.uniform_int(1, 6)));
        } else if (pick < 60) {
            s.ops.push_back(op_yield(t++, w));
        } else if (pick < 70) {
            s.ops.push_back(op_block(t++, w, rng.uniform_int(1, 5)));
        } else if (pick < 78) {
            s.ops.push_back(op_sleep(t++, w, rng.uniform_int(1, 200)));
        } else if (pick < 86) {
            s.ops.push_back(op_wake(t++, rng.uniform_int(1, 5), rng.uniform_int(0, 4)));
        } else if (pick < 94) {
            s.ops.push_back(op_advance(t++, rng.uniform_int(1, 60)));
        } else {
            // cancel a plausibly-live id
            s.ops.push_back(op_cancel(t++, (uint64_t)rng.uniform_int(1, (int)next_id)));
        }
    }
    // wrap clock near 2^64 then advance to wake everything.
    s.ops.push_back(op_advance(t++, 0xfffffffffffffff0ULL));
    s.ops.push_back(op_advance(t++, 0x100ULL));  // wraps
    for (uint64_t k = 1; k <= 5; ++k) s.ops.push_back(op_wake(t++, k, 64));
    for (int r = 0; r < 80; ++r) s.ops.push_back(op_run(t++, r % 8, 6));
    return s;
}

// S7: priority ordering — high beats low locally and globally.
static Scenario sc_priority() {
    Scenario s; s.name = "priority_high_beats_low";
    s.spec = make_spec(1, 4, 16, 64, 16, 16, 256);
    int t = 0;
    // interleave low and high spawns to worker0.
    s.ops.push_back(op_spawn(t++, 1, 0, 0, 2));
    s.ops.push_back(op_spawn(t++, 2, 0, 1, 2));
    s.ops.push_back(op_spawn(t++, 3, 0, 0, 2));
    s.ops.push_back(op_spawn(t++, 4, 0, 1, 2));
    // run: should pop high tail first (id4), then high (id2), then low tail (id3), then low (id1).
    for (int i = 0; i < 8; ++i) s.ops.push_back(op_run(t++, 0, 2));
    // global priority: overflow local then ensure global high before global low.
    for (uint64_t id = 10; id <= 13; ++id) s.ops.push_back(op_spawn(t++, id, 0, 0, 1)); // fill local low(4)
    s.ops.push_back(op_spawn(t++, 20, 0, 0, 1)); // global low
    s.ops.push_back(op_spawn(t++, 21, 0, 1, 1)); // local high (cap free)
    for (int i = 0; i < 8; ++i) s.ops.push_back(op_run(t++, 0, 1));
    return s;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_lifo_fifo());
    v.push_back(sc_steal());
    v.push_back(sc_sleep_advance());
    v.push_back(sc_block_wake_stall());
    v.push_back(sc_overflow_cancel());
    v.push_back(sc_mixed_random());
    v.push_back(sc_priority());
    return v;
}

// ------------------------------------------------------------- runner
static bool run_one_op(
    const WsrProblemSpec& spec, const WsrRunSpec& op, WsrOracle* oracle,
    void* state, void* workspace, size_t workspace_bytes, cudaStream_t stream,
    StepResult* result, std::string* error) {

    GuardedDeviceBuffer<int64_t> d_counts;
    GuardedDeviceBuffer<int32_t> d_opidx;
    GuardedDeviceBuffer<uint64_t> d_clock, d_eseq, d_sched, d_ready, d_run, d_blocked, d_sleep, d_state;
    d_counts.allocate(WSR_COUNT_N);
    d_opidx.allocate(1);
    d_clock.allocate(1); d_eseq.allocate(1); d_sched.allocate(1); d_ready.allocate(1);
    d_run.allocate(1); d_blocked.allocate(1); d_sleep.allocate(1); d_state.allocate(1);

    WsrOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.op_index_out = d_opidx.ptr;
    outputs.clock_out = d_clock.ptr;
    outputs.event_seq_out = d_eseq.ptr;
    outputs.sched_event_hash = d_sched.ptr;
    outputs.ready_hash = d_ready.ptr;
    outputs.running_hash = d_run.ptr;
    outputs.blocked_hash = d_blocked.ptr;
    outputs.sleep_hash = d_sleep.ptr;
    outputs.state_checksum = d_state.ptr;

    WsrRunSpec op_copy = op;  // verify immutability of the op spec
    WsrInputs inputs = {};
    inputs.reserved = nullptr;

    CUDA_CHECK(solution_run(state, &op_copy, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (std::memcmp(&op_copy, &op, sizeof(WsrRunSpec)) != 0) {
        if (error) *error = "run spec mutated by solution_run";
        return false;
    }

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_opidx.check_guards("op_index", error)) return false;
    if (!d_clock.check_guards("clock", error)) return false;
    if (!d_eseq.check_guards("event_seq", error)) return false;
    if (!d_sched.check_guards("sched_hash", error)) return false;
    if (!d_ready.check_guards("ready_hash", error)) return false;
    if (!d_run.check_guards("running_hash", error)) return false;
    if (!d_blocked.check_guards("blocked_hash", error)) return false;
    if (!d_sleep.check_guards("sleep_hash", error)) return false;
    if (!d_state.check_guards("state_checksum", error)) return false;

    const std::vector<int64_t> h_counts = d_counts.download_data();
    const std::vector<int32_t> h_opidx = d_opidx.download_data();
    const std::vector<uint64_t> h_clock = d_clock.download_data();
    const std::vector<uint64_t> h_eseq = d_eseq.download_data();
    const std::vector<uint64_t> h_sched = d_sched.download_data();
    const std::vector<uint64_t> h_ready = d_ready.download_data();
    const std::vector<uint64_t> h_run = d_run.download_data();
    const std::vector<uint64_t> h_blocked = d_blocked.download_data();
    const std::vector<uint64_t> h_sleep = d_sleep.download_data();
    const std::vector<uint64_t> h_state = d_state.download_data();

    WsrExpected expected;
    oracle->step_once(op, &expected);

    WsrHostOutputsView got = {};
    got.counts = h_counts.data();
    got.op_index_out = h_opidx.data();
    got.clock_out = h_clock.data();
    got.event_seq_out = h_eseq.data();
    got.sched_event_hash = h_sched.data();
    got.ready_hash = h_ready.data();
    got.running_hash = h_run.data();
    got.blocked_hash = h_blocked.data();
    got.sleep_hash = h_sleep.data();
    got.state_checksum = h_state.data();

    if (!wsr_check_outputs(expected, got, error)) return false;

    if (result) {
        result->counts = h_counts;
        result->op_index = h_opidx[0];
        result->clock = h_clock[0];
        result->event_seq = h_eseq[0];
        result->sched = h_sched[0];
        result->ready = h_ready[0];
        result->running = h_run[0];
        result->blocked = h_blocked[0];
        result->sleep = h_sleep[0];
        result->state = h_state[0];
    }
    return true;
}

static bool run_scenario_once(
    const Scenario& sc, bool verbose, std::vector<StepResult>* results,
    int* passed, int* total, std::string* first_error) {

    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);
    if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    WsrOracle oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.ops.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.ops.size(); ++i) {
        StepResult result;
        std::string error;
        const bool ok = run_one_op(sc.spec, sc.ops[i], &oracle, state, workspace.ptr, workspace_bytes, stream,
                                   results ? &result : nullptr, &error);
        ++(*total);
        if (ok) ++(*passed);
        else { all_ok = false; if (first_error && first_error->empty()) { std::ostringstream o; o << sc.name << " op " << i << " (kind=" << sc.ops[i].op_kind << "): " << error; *first_error = o.str(); } }
        if (results) results->push_back(result);
        if (verbose && (!ok || (i % 64 == 0))) {
            std::printf("scenario %-26s op %04zu/%04zu kind=%d %s%s%s\n",
                        sc.name.c_str(), i, sc.ops.size(), sc.ops[i].op_kind,
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
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].counts != b[i].counts || a[i].op_index != b[i].op_index ||
            a[i].clock != b[i].clock || a[i].event_seq != b[i].event_seq ||
            a[i].sched != b[i].sched || a[i].ready != b[i].ready ||
            a[i].running != b[i].running || a[i].blocked != b[i].blocked ||
            a[i].sleep != b[i].sleep || a[i].state != b[i].state) {
            if (error) { std::ostringstream o; o << "replay mismatch at op " << i << " state a=0x" << std::hex << a[i].state << " b=0x" << b[i].state; *error = o.str(); }
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
            std::vector<StepResult> base_results, replay_results;
            std::string error;

            const bool ok_base = run_scenario_once(sc, true, &base_results, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay_results, &passed, &total, &error);

            if (ok_base && ok_replay) {
                std::string ce;
                if (compare_results(base_results, replay_results, &ce)) {
                    std::printf("scenario %-26s exact replay PASS (%zu ops)\n", sc.name.c_str(), sc.ops.size());
                } else { all_ok = false; std::printf("scenario %-26s exact replay FAIL  %s\n", sc.name.c_str(), ce.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-26s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
