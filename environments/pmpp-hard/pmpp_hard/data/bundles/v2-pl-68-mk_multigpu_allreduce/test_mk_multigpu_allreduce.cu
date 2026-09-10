// file: test_mk_multigpu_allreduce.cu
//
// 3-way validation harness for T68 (MK9). Drives the linked solution (reference
// or naive) and compares every per-step output against the independent host
// oracle. Exercises several deeply adversarial scenarios plus exact-replay
// determinism and guard-buffer / input-immutability checks.

#include "mk_multigpu_allreduce_common.h"
#include "mk_multigpu_allreduce_oracle.hpp"

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
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                             \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                              \
        }                                                                      \
    } while (0)

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
        if (n > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), kGuardBytes + data_bytes + kGuardBytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, kGuardBytes + data_bytes + kGuardBytes));
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> left(kGuardBytes), right(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(left.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(right.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (left[i] != kGuardByte || right[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }
        }
        return true;
    }
};

struct Scenario {
    std::string name;
    MgaProblemSpec spec;
    std::vector<MgaRunSpec> steps;
};

static MgaProblemSpec make_spec(int ranks, int chunk_max, int max_colls, int max_remote,
                                int max_credit, int remote_latency, int max_sched) {
    MgaProblemSpec s = {};
    s.abi_version = MGA_ABI_VERSION;
    s.rank_count = ranks; s.chunk_count_max = chunk_max; s.max_collectives = max_colls;
    s.max_remote_events = max_remote; s.max_send_credits_per_link = max_credit;
    s.remote_latency = remote_latency; s.max_scheduler_queue_per_rank = max_sched;
    return s;
}

static MgaRunSpec rs(int op, int op_index, int a0 = 0, int a1 = 0, int a2 = 0, int a3 = 0,
                     int a4 = 0, int a5 = 0, int a6 = 0, int a7 = 0) {
    MgaRunSpec r = {};
    r.abi_version = MGA_ABI_VERSION;
    r.op = op; r.op_index = op_index;
    r.a0 = a0; r.a1 = a1; r.a2 = a2; r.a3 = a3; r.a4 = a4; r.a5 = a5; r.a6 = a6; r.a7 = a7;
    return r;
}

static int MASK(std::initializer_list<int> ranks) {
    int m = 0; for (int r : ranks) m |= (1 << r); return m;
}

// Drive every rank a few actions and advance the clock; a compact "pump" helper.
static void pump(Scenario& sc, int& oi, int ranks, int rounds, int action_limit,
                 int delta, int max_remote) {
    for (int t = 0; t < rounds; ++t) {
        for (int r = 0; r < ranks; ++r)
            sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, r, action_limit));
        sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, delta, max_remote));
    }
}

// 1. Full small allreduce that should complete (3 ranks, 1 chunk).
static Scenario sc_basic_allreduce() {
    Scenario sc; sc.name = "basic_allreduce_3r1c";
    sc.spec = make_spec(4, 4, 4, 1024, 8, 2, 256);
    int oi = 0;
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 7, MASK({0,1,2}), 1, 123));
    pump(sc, oi, 4, 14, 4, 1, 32);
    // poll a signal counter and check completion observable.
    sc.steps.push_back(rs(MGA_OP_POLL_SIGNAL, oi++, 1, 7, MGA_PK_RS, 0, 0, 1));
    return sc;
}

// 2. Multi-chunk allreduce on 4 ranks, several chunks (chunk aliasing path).
static Scenario sc_multichunk() {
    Scenario sc; sc.name = "multichunk_4r3c";
    sc.spec = make_spec(5, 4, 4, 2048, 6, 3, 256);
    int oi = 0;
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 11, MASK({0,1,2,3}), 3, 999));
    pump(sc, oi, 5, 24, 6, 1, 48);
    sc.steps.push_back(rs(MGA_OP_POLL_SIGNAL, oi++, 2, 11, MGA_PK_AG, 1, 1, 1));
    return sc;
}

// 3. Credit pressure: 1 credit/link, 3 ranks, 3 chunks. A deliberately messy
//    interleaving (out-of-lockstep RANK_STEP / ADVANCE / FORCE_CREDIT) drives a
//    rank to issue several allgather sends on its single credit-limited link
//    before inbound credit returns -> SEND_CREDIT_STALL + requeue. The requeued
//    action does not block later actions; the collective still completes. Several
//    self-loop FORCE_CREDITs (src==dst) also exercise the invalid path. This
//    exact sequence is a frozen adversarial schedule (it provably hits the stall).
static Scenario sc_credit_stall() {
    Scenario sc; sc.name = "credit_stall_3r3c";
    sc.spec = make_spec(3, 4, 4, 4096, 1, 2, 256);  // 1 credit/link, lat=2
    int oi = 0;
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 5, MASK({0,1,2}), 3, 123));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 1, 1));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 2, 3));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 2, 1));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 2, 0, 2));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 2, 0, 2));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 0, 2));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 2, 2, 2));   // self -> invalid
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 2, 3));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 0, 2));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 2, 0, 2));
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 0, 2));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 2, 4));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 2, 0, 2));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 1, 4));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 2, 0, 2));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 1, 4));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 2, 3));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 1, 1, 2));   // self -> invalid
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 0, 0, 2));   // self -> invalid
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 1, 0, 2));
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 1, 4));
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 2, 1));
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 2, 4));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 2, 0, 2));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 2, 3));         // <- first stall here
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 1, 1, 2));   // self -> invalid
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 2, 3));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 1, 2, 2));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 2, 1));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 2, 0, 2));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 0, 2));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 0, 2, 2));
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 0, 4));
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 2, 2));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 2, 3));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 2, 0, 2));
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 1, 3));
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 1, 1));
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 1, 0, 2));
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 1, 2));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 1, 2));
    // ensure forward progress to completion regardless.
    for (int r = 0; r < 3; ++r) sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, r, (r + 1) % 3, 4));
    pump(sc, oi, 3, 24, 6, 2, 32);
    return sc;
}

// 4. Cancellation mid-flight: cancel forces ACTION_STALE_DROP + REMOTE_STALE_DROP.
static Scenario sc_cancel() {
    Scenario sc; sc.name = "cancel_midflight_4r2c";
    sc.spec = make_spec(4, 4, 4, 1024, 8, 4, 256);
    int oi = 0;
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 21, MASK({0,1,2,3}), 2, 7));
    pump(sc, oi, 4, 4, 4, 1, 16);
    // cancel while remote events are in flight and actions are queued.
    sc.steps.push_back(rs(MGA_OP_CANCEL_COLL, oi++, 21));
    pump(sc, oi, 4, 8, 6, 1, 32);
    // cancel again -> invalid (terminal).
    sc.steps.push_back(rs(MGA_OP_CANCEL_COLL, oi++, 21));
    return sc;
}

// 5. Concurrent collectives sharing ranks + reuse of a terminal coll id.
static Scenario sc_concurrent() {
    Scenario sc; sc.name = "concurrent_2coll_5r";
    sc.spec = make_spec(6, 4, 4, 4096, 4, 2, 384);
    int oi = 0;
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 100, MASK({0,1,2}), 2, 11));
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 200, MASK({2,3,4,5}), 2, 22));
    // interleave force-credit to stress per-coll credit lookup
    pump(sc, oi, 6, 6, 5, 1, 24);
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 2, 3, 2));  // edge in coll 200
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 0, 1, 2));  // edge in coll 100
    pump(sc, oi, 6, 18, 6, 1, 48);
    // poll signals on both
    sc.steps.push_back(rs(MGA_OP_POLL_SIGNAL, oi++, 1, 100, MGA_PK_AG, 0, 0, 1));
    sc.steps.push_back(rs(MGA_OP_POLL_SIGNAL, oi++, 3, 200, MGA_PK_RS, 1, 0, 1));
    return sc;
}

// 6. Adversarial invalids of every flavour + ADVANCE delta 0 + POLL on absent.
static Scenario sc_invalids() {
    Scenario sc; sc.name = "invalids_3r";
    sc.spec = make_spec(3, 4, 2, 256, 4, 1, 128);
    int oi = 0;
    // begin invalid: mask < 2 ranks
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 1, MASK({0}), 2, 1));
    // begin invalid: chunk_count out of range
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 2, MASK({0,1}), 99, 1));
    // valid begin
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 5, MASK({0,1,2}), 2, 3));
    // begin again same id while active -> invalid
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 5, MASK({0,1}), 2, 3));
    // rank_step invalid: rank out of range, action_limit 0
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 99, 4));
    sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, 0, 0));
    // poll invalid: absent coll, bad phase kind
    sc.steps.push_back(rs(MGA_OP_POLL_SIGNAL, oi++, 0, 777, MGA_PK_RS, 0, 0, 1));
    sc.steps.push_back(rs(MGA_OP_POLL_SIGNAL, oi++, 0, 5, 9, 0, 0, 1));
    // cancel absent -> invalid
    sc.steps.push_back(rs(MGA_OP_CANCEL_COLL, oi++, 4242));
    // force credit invalid: not an edge of any active coll (5,5) self / out-of-range
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 0, 2, 3));  // (0,2) not adjacent in ring {0,1,2}
    sc.steps.push_back(rs(MGA_OP_FORCE_CREDIT, oi++, 0, 0, 3));  // self
    // advance delta 0 valid
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 0, 4));
    // drive to done
    pump(sc, oi, 3, 14, 6, 1, 24);
    return sc;
}

// 7. High remote latency funnel: many sends pile up before any arrival; then a
//    big ADVANCE flushes them in canonical (due,seq) order.
static Scenario sc_latency_funnel() {
    Scenario sc; sc.name = "latency_funnel_5r2c";
    sc.spec = make_spec(5, 4, 4, 4096, 8, 16, 512);
    int oi = 0;
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 9, MASK({0,1,2,3,4}), 2, 4242));
    // many rank steps WITHOUT advancing: events accumulate at due=latency.
    for (int t = 0; t < 6; ++t)
        for (int r = 0; r < 5; ++r) sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, r, 4));
    // partial flush
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 16, 3));
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 0, 5));
    // full pump to completion
    pump(sc, oi, 5, 30, 8, 4, 64);
    return sc;
}

// 8. action_limit=1 trickle: process exactly one action per rank step, exposing
//    interleavings and stalled-send requeue ordering.
static Scenario sc_trickle() {
    Scenario sc; sc.name = "trickle_limit1_4r2c";
    sc.spec = make_spec(4, 4, 4, 2048, 2, 2, 256);
    int oi = 0;
    sc.steps.push_back(rs(MGA_OP_BEGIN_ALLREDUCE, oi++, 17, MASK({0,1,2,3}), 2, 31));
    for (int t = 0; t < 60; ++t) {
        for (int r = 0; r < 4; ++r) sc.steps.push_back(rs(MGA_OP_RANK_STEP, oi++, r, 1));
        if (t % 3 == 2) sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 1, 4));
    }
    sc.steps.push_back(rs(MGA_OP_ADVANCE, oi++, 4, 64));
    pump(sc, oi, 4, 10, 4, 1, 16);
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_basic_allreduce());
    v.push_back(sc_multichunk());
    v.push_back(sc_credit_stall());
    v.push_back(sc_cancel());
    v.push_back(sc_concurrent());
    v.push_back(sc_invalids());
    v.push_back(sc_latency_funnel());
    v.push_back(sc_trickle());
    return v;
}

static void copy_result(MgaOracleResult* r,
                        GuardedDeviceBuffer<int64_t>& cnt,
                        GuardedDeviceBuffer<uint64_t>& evh,
                        GuardedDeviceBuffer<uint64_t>& ch,
                        GuardedDeviceBuffer<uint64_t>& sh,
                        GuardedDeviceBuffer<uint64_t>& crh,
                        GuardedDeviceBuffer<uint64_t>& ph,
                        GuardedDeviceBuffer<uint64_t>& sch,
                        GuardedDeviceBuffer<uint64_t>& clk,
                        GuardedDeviceBuffer<uint64_t>& es) {
    std::vector<int64_t> c = cnt.download();
    for (int i = 0; i < MGA_COUNTER_COUNT; ++i) r->counters[i] = c[(size_t)i];
    r->remote_event_hash = evh.download()[0];
    r->collective_hash = ch.download()[0];
    r->signal_hash = sh.download()[0];
    r->credit_hash = crh.download()[0];
    r->pending_remote_hash = ph.download()[0];
    r->scheduler_hash = sch.download()[0];
    r->clock_out = clk.download()[0];
    r->event_seq_out = es.download()[0];
}

static bool run_scenario_once(const Scenario& sc, bool verbose,
                              std::vector<MgaOracleResult>* results,
                              int* passed, int* total, std::string* first_error) {
    MgaProblemSpec spec = sc.spec;

    size_t workspace_bytes = solution_workspace_bytes(&spec);
    // MANDATE: clamp only. Never fail-guard on workspace_bytes == 0.
    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MgaOracle oracle;
    oracle.init(spec);

    if (results) { results->clear(); results->reserve(sc.steps.size()); }

    bool all_ok = true;

    for (size_t i = 0; i < sc.steps.size(); ++i) {
        const MgaRunSpec& step = sc.steps[i];

        // input-immutability check: keep a pristine copy of the run spec.
        MgaRunSpec step_copy = step;

        GuardedDeviceBuffer<int64_t> d_counters;
        GuardedDeviceBuffer<uint64_t> d_evh, d_ch, d_sh, d_crh, d_ph, d_sch, d_clk, d_es;
        d_counters.allocate(MGA_COUNTER_COUNT);
        d_evh.allocate(1); d_ch.allocate(1); d_sh.allocate(1); d_crh.allocate(1);
        d_ph.allocate(1); d_sch.allocate(1); d_clk.allocate(1); d_es.allocate(1);

        MgaOutputs outputs = {};
        outputs.counters = d_counters.ptr;
        outputs.remote_event_hash = d_evh.ptr;
        outputs.collective_hash = d_ch.ptr;
        outputs.signal_hash = d_sh.ptr;
        outputs.credit_hash = d_crh.ptr;
        outputs.pending_remote_hash = d_ph.ptr;
        outputs.scheduler_hash = d_sch.ptr;
        outputs.clock_out = d_clk.ptr;
        outputs.event_seq_out = d_es.ptr;

        std::string error;
        CUDA_CHECK(solution_run(state, &step, nullptr, &outputs, workspace.ptr, workspace_bytes, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool ok = true;
        // input immutability: the run spec must not have been mutated.
        if (std::memcmp(&step_copy, &step, sizeof(MgaRunSpec)) != 0) {
            ok = false; error = "run spec mutated by solution_run";
        }
        ok = ok && d_counters.check_guards("counters", &error);
        ok = ok && d_evh.check_guards("remote_event_hash", &error);
        ok = ok && d_ch.check_guards("collective_hash", &error);
        ok = ok && d_sh.check_guards("signal_hash", &error);
        ok = ok && d_crh.check_guards("credit_hash", &error);
        ok = ok && d_ph.check_guards("pending_remote_hash", &error);
        ok = ok && d_sch.check_guards("scheduler_hash", &error);
        ok = ok && d_clk.check_guards("clock", &error);
        ok = ok && d_es.check_guards("event_seq", &error);

        MgaOracleResult got;
        copy_result(&got, d_counters, d_evh, d_ch, d_sh, d_crh, d_ph, d_sch, d_clk, d_es);

        MgaOracleResult expected;
        oracle.step_once(step, &expected);

        ok = ok && mga_check(expected, got, &error);

        ++(*total);
        if (ok) ++(*passed);
        else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream oss;
                oss << sc.name << " step " << i << " (op=" << step.op << "): " << error;
                *first_error = oss.str();
            }
        }

        if (results) results->push_back(got);

        if (verbose && !ok) {
            std::printf("scenario %-24s step %03zu/%03zu op=%d FAIL  %s\n",
                        sc.name.c_str(), i, sc.steps.size(), step.op, error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_runs(const std::vector<MgaOracleResult>& a,
                         const std::vector<MgaOracleResult>& b, std::string* err) {
    if (a.size() != b.size()) { if (err) *err = "step count mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        std::string e;
        if (!mga_check(a[i], b[i], &e)) {
            if (err) { std::ostringstream o; o << "replay mismatch at step " << i << ": " << e; *err = o.str(); }
            return false;
        }
    }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        std::vector<Scenario> scenarios = build_scenarios();
        int passed = 0, total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<MgaOracleResult> base, replay;
            std::string error;
            bool ok_base = run_scenario_once(sc, true, &base, &passed, &total, &error);
            bool ok_replay = run_scenario_once(sc, false, &replay, &passed, &total, &error);
            if (ok_base && ok_replay && !base.empty()) {
                const int64_t* c = base.back().counters;
                std::printf("  [cov %-22s] beg=%lld lr=%lld rs=%lld ag=%lld snd=%lld stall=%lld arr=%lld sdrop=%lld cret=%lld cforce=%lld rapply=%lld owr=%lld gapply=%lld done=%lld sigR=%lld sigW=%lld adrop=%lld cncl=%lld inv=%lld\n",
                    sc.name.c_str(),
                    (long long)c[0],(long long)c[1],(long long)c[2],(long long)c[3],(long long)c[4],
                    (long long)c[5],(long long)c[6],(long long)c[7],(long long)c[8],(long long)c[9],
                    (long long)c[10],(long long)c[11],(long long)c[12],(long long)c[13],(long long)c[14],
                    (long long)c[15],(long long)c[16],(long long)c[17],(long long)c[18]);
            }
            if (ok_base && ok_replay) {
                std::string ce;
                if (compare_runs(base, replay, &ce))
                    std::printf("scenario %-24s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-24s exact replay FAIL  %s\n", sc.name.c_str(), ce.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-24s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
