// file: test_mk_fleet_two_level.cu
//
// 3-way validation harness for T64 (MK5). Drives the linked solution (reference or
// naive) and compares every per-step output against the independent host oracle.
// Exercises >=6 adversarial scenarios plus exact-replay determinism.

#include "mk_fleet_two_level_common.h"
#include "mk_fleet_two_level_oracle.hpp"

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
    MkfProblemSpec spec;
    std::vector<MkfRunSpec> steps;
};

static MkfProblemSpec make_spec(int cc, int wpc, int max_tasks, int max_events,
                                int max_ready, int max_wq, int max_running, int dev_cc) {
    MkfProblemSpec s = {};
    s.abi_version = MKF_ABI_VERSION;
    s.chiplet_count = cc; s.workers_per_chiplet = wpc; s.max_tasks = max_tasks;
    s.max_events = max_events; s.max_ready_per_chiplet = max_ready;
    s.max_worker_queue = max_wq; s.max_running = max_running;
    s.device_task_chiplet_count = dev_cc;
    return s;
}

// generic op
static MkfRunSpec op(int o, int oi, int a0, int a1 = 0, int a2 = 0, int a3 = 0) {
    MkfRunSpec r = {};
    r.abi_version = MKF_ABI_VERSION;
    r.op = o; r.op_index = oi; r.a0 = a0; r.a1 = a1; r.a2 = a2; r.a3 = a3;
    return r;
}

// SUBMIT with all fields. waits: pass up to 2 pairs.
static MkfRunSpec submit(int oi, int task_id, int scope, int home, uint64_t mask,
                         int duration, int seed, int oe, int oinc,
                         int wc = 0, int we0 = 0, int wt0 = 0, int we1 = 0, int wt1 = 0) {
    MkfRunSpec r = {};
    r.abi_version = MKF_ABI_VERSION;
    r.op = MKF_OP_SUBMIT; r.op_index = oi;
    r.a0 = task_id; r.a1 = scope; r.a2 = home; r.a3 = duration;
    r.worker_mask = mask; r.payload_seed = seed; r.output_event = oe; r.output_increment = oinc;
    r.wait_count = wc; r.wait_ev[0] = we0; r.wait_target[0] = wt0;
    r.wait_ev[1] = we1; r.wait_target[1] = wt1;
    return r;
}

static uint64_t wmask(std::initializer_list<int> bits) {
    uint64_t m = 0; for (int b : bits) m |= (1ULL << b); return m;
}

// ---- scenario builders ----

// 1. WAVE/CU single-worker round-robin dispatch + finish.
static Scenario sc_single_scope() {
    Scenario sc; sc.name = "single_scope_2x4";
    sc.spec = make_spec(2, 4, 32, 8, 16, 4, 64, 2);
    int oi = 0;
    // 3 WAVE tasks on chiplet 0, workers 0..3.
    sc.steps.push_back(submit(oi++, 1, MKF_SCOPE_WAVE, 0, wmask({0,1,2,3}), 5, 111, 0, 1));
    sc.steps.push_back(submit(oi++, 2, MKF_SCOPE_CU, 0, wmask({0,1,2,3}), 7, 222, 1, 2));
    sc.steps.push_back(submit(oi++, 3, MKF_SCOPE_WAVE, 0, wmask({0,1,2,3}), 5, 333, 0, 1));
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 0, 8));     // dispatch all three (rr 0,1,2)
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 0, 2));    // start worker 0
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 1, 2));    // start worker 1
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 2, 2));    // start worker 2
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 5, 8));   // finish those due<=5
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 5, 8));   // finish rest
    return sc;
}

// 2. CHIPLET scope: two-level local counting, only last worker fences.
static Scenario sc_chiplet_two_level() {
    Scenario sc; sc.name = "chiplet_two_level_2x3";
    sc.spec = make_spec(2, 3, 32, 8, 16, 4, 64, 2);
    int oi = 0;
    // CHIPLET task home=1, selected workers 3,4,5 (chiplet 1).
    sc.steps.push_back(submit(oi++, 10, MKF_SCOPE_CHIPLET, 1, wmask({3,4,5}), 4, 7, 2, 3));
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 1, 4));     // dispatch chiplet part (3 copies)
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 3, 1));
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 4, 1));
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 5, 1));
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 4, 1));   // finish one -> LOCAL_L2_INC, no fence
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 0, 1));   // finish second
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 0, 4));   // finish third -> fence + global
    return sc;
}

// 3. DEVICE scope: per-chiplet parts dispatch/finish/fence independently.
static Scenario sc_device_partial() {
    Scenario sc; sc.name = "device_partial_3x2";
    sc.spec = make_spec(3, 2, 32, 8, 16, 4, 64, 3);
    int oi = 0;
    // DEVICE task: bits across chiplets 0 (w0),1 (w2),2 (w4) -> 3 participating chiplets.
    sc.steps.push_back(submit(oi++, 20, MKF_SCOPE_DEVICE, 0, wmask({0,2,4}), 3, 9, 3, 1));
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 0, 4));     // chiplet 0 part
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 1, 4));     // chiplet 1 part
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 2, 4));     // chiplet 2 part
    for (int w = 0; w < 6; ++w) sc.steps.push_back(op(MKF_OP_WORKER, oi++, w, 1));
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 3, 2));   // finish chiplet 0 (2 workers) -> fence part0
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 0, 2));   // finish chiplet 1 -> fence part1
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 0, 8));   // finish chiplet 2 -> fence part2 -> GLOBAL_DONE
    return sc;
}

// 4. Waits + ready-scan via WAVE completion / EVENT_FORCE.
static Scenario sc_waits_readyscan() {
    Scenario sc; sc.name = "waits_readyscan_2x2";
    sc.spec = make_spec(2, 2, 32, 8, 16, 4, 64, 2);
    int oi = 0;
    // producer: WAVE task incrementing event 4 by 2.
    sc.steps.push_back(submit(oi++, 30, MKF_SCOPE_WAVE, 0, wmask({0,1}), 2, 1, 4, 2));
    // waiter: needs event 4 >= 2.
    sc.steps.push_back(submit(oi++, 31, MKF_SCOPE_WAVE, 1, wmask({2,3}), 2, 2, 5, 1, /*wc*/1, /*we0*/4, /*wt0*/2));
    // another waiter on event 5 >= 1.
    sc.steps.push_back(submit(oi++, 32, MKF_SCOPE_CU, 0, wmask({0,1}), 2, 3, 6, 1, 1, 5, 1));
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 0, 4));     // dispatch producer 30
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 0, 1));
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 2, 1));   // 30 finishes -> event4=2 -> 31 READY
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 1, 4));     // dispatch 31
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 2, 1));
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 2, 1));   // 31 finishes -> event5=1 -> 32 READY
    sc.steps.push_back(op(MKF_OP_EVENT_FORCE, oi++, 6, 5));  // force unrelated event
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 0, 4));     // dispatch 32
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 0, 1));
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 2, 4));
    return sc;
}

// 5. Cancellation: cancel queued, dispatched, and running tasks.
static Scenario sc_cancel() {
    Scenario sc; sc.name = "cancel_mixed_2x2";
    sc.spec = make_spec(2, 2, 32, 8, 16, 4, 64, 2);
    int oi = 0;
    sc.steps.push_back(submit(oi++, 40, MKF_SCOPE_WAVE, 0, wmask({0,1}), 6, 1, 0, 1));
    sc.steps.push_back(submit(oi++, 41, MKF_SCOPE_CHIPLET, 0, wmask({0,1}), 6, 2, 1, 1));
    sc.steps.push_back(submit(oi++, 42, MKF_SCOPE_WAVE, 1, wmask({2,3}), 6, 3, 2, 1));
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 0, 4));     // dispatch 40 then 41
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 0, 1));    // start 40 on worker 0
    sc.steps.push_back(op(MKF_OP_CANCEL, oi++, 40));      // cancel running 40
    sc.steps.push_back(op(MKF_OP_CANCEL, oi++, 41));      // cancel dispatched 41
    sc.steps.push_back(op(MKF_OP_CANCEL, oi++, 42));      // cancel queued 42
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 1, 2));    // worker 1 had 41 copy -> stale drop -> idle
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 6, 8));   // 40 finishes -> CANCELLED_FINISH
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 1, 2));     // 42 was queued there -> stale -> idle
    sc.steps.push_back(op(MKF_OP_CANCEL, oi++, 40));      // invalid: already cancelled
    return sc;
}

// 6. Invalid ops of every kind.
static Scenario sc_invalid() {
    Scenario sc; sc.name = "invalid_all_2x2";
    sc.spec = make_spec(2, 2, 4, 4, 8, 2, 16, 2);
    int oi = 0;
    sc.steps.push_back(submit(oi++, 50, MKF_SCOPE_WAVE, 9, wmask({0}), 1, 0, 0, 1));      // bad chiplet
    sc.steps.push_back(submit(oi++, 51, 9, 0, wmask({0}), 1, 0, 0, 1));                   // bad scope
    sc.steps.push_back(submit(oi++, 52, MKF_SCOPE_WAVE, 0, wmask({0}), 1, 0, 99, 1));     // bad event
    sc.steps.push_back(submit(oi++, 53, MKF_SCOPE_WAVE, 0, 0ULL, 1, 0, 0, 1));            // empty mask
    sc.steps.push_back(submit(oi++, 54, MKF_SCOPE_WAVE, 0, wmask({0}), 1, 0, 0, 1, 1, 99, 1)); // bad wait event
    sc.steps.push_back(submit(oi++, 55, MKF_SCOPE_WAVE, 0, wmask({0}), 1, 0, 0, 1));      // valid
    sc.steps.push_back(submit(oi++, 55, MKF_SCOPE_WAVE, 0, wmask({0}), 1, 0, 0, 1));      // dup id
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 9, 1));    // bad chiplet
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 0, 0));    // dispatch_limit 0
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 99, 1));  // bad worker
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 0, 0));   // limit 0
    sc.steps.push_back(op(MKF_OP_EVENT_FORCE, oi++, 99, 1)); // bad event
    sc.steps.push_back(op(MKF_OP_CANCEL, oi++, 7777));   // absent
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 0, 1));  // delta 0 valid (no-op-ish)
    return sc;
}

// 7. Dispatch stall + worker busy + capacity skipping.
static Scenario sc_stall_busy() {
    Scenario sc; sc.name = "stall_busy_1x2";
    sc.spec = make_spec(1, 2, 32, 8, 16, 1, 64, 1);  // max_worker_queue=1
    int oi = 0;
    // 3 WAVE tasks, 2 workers, queue cap 1. Third dispatch stalls.
    sc.steps.push_back(submit(oi++, 60, MKF_SCOPE_WAVE, 0, wmask({0,1}), 10, 1, 0, 1));
    sc.steps.push_back(submit(oi++, 61, MKF_SCOPE_WAVE, 0, wmask({0,1}), 10, 2, 0, 1));
    sc.steps.push_back(submit(oi++, 62, MKF_SCOPE_WAVE, 0, wmask({0,1}), 10, 3, 0, 1));
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 0, 8));    // dispatch 60->w0, 61->w1, 62 stalls
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 0, 1));   // start 60
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 0, 1));   // worker 0 busy
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 1, 1));   // start 61
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 10, 1)); // finish 60 (frees w0)
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 0, 8));    // now 62 dispatches to w0
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 0, 1));
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 20, 8));
    return sc;
}

// 8. Stress: many mixed-scope tasks, partial advances, interleaved sched/worker.
static Scenario sc_stress() {
    Scenario sc; sc.name = "stress_4x4";
    sc.spec = make_spec(4, 4, 64, 16, 64, 8, 256, 4);
    int oi = 0;
    int total_workers = 16;
    for (int k = 0; k < 12; ++k) {
        int scope = k % 4;
        int home = k % 4;
        uint64_t mask;
        if (scope == MKF_SCOPE_DEVICE) {
            // span 2 chiplets
            int c0 = k % 4, c1 = (k + 1) % 4;
            mask = (1ULL << (c0 * 4)) | (1ULL << (c1 * 4 + 1));
        } else {
            // a couple of home-chiplet workers
            mask = (1ULL << (home * 4)) | (1ULL << (home * 4 + 1));
            if (scope == MKF_SCOPE_CHIPLET) mask |= (1ULL << (home * 4 + 2));
        }
        sc.steps.push_back(submit(oi++, 70 + k, scope, home, mask, 2 + (k % 3), 100 + k,
                                  k % 16, 1 + (k % 2)));
    }
    for (int ch = 0; ch < 4; ++ch) sc.steps.push_back(op(MKF_OP_SCHED, oi++, ch, 8));
    for (int w = 0; w < total_workers; ++w) sc.steps.push_back(op(MKF_OP_WORKER, oi++, w, 4));
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 2, 8));
    for (int ch = 0; ch < 4; ++ch) sc.steps.push_back(op(MKF_OP_SCHED, oi++, ch, 8));
    for (int w = 0; w < total_workers; ++w) sc.steps.push_back(op(MKF_OP_WORKER, oi++, w, 4));
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 4, 64));
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 10, 64));
    return sc;
}

// 9. DEVICE with re-dispatch staleness + cancel a device part mid-flight.
static Scenario sc_device_cancel() {
    Scenario sc; sc.name = "device_cancel_2x2";
    sc.spec = make_spec(2, 2, 32, 8, 16, 4, 64, 2);
    int oi = 0;
    // DEVICE spanning both chiplets (w0 in ch0, w2 in ch1).
    sc.steps.push_back(submit(oi++, 80, MKF_SCOPE_DEVICE, 0, wmask({0,2}), 4, 5, 0, 1));
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 0, 4));    // ch0 part
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 0, 1));
    sc.steps.push_back(op(MKF_OP_WORKER, oi++, 1, 1));
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 4, 8));  // ch0 part fences (LOCAL_DONE)
    sc.steps.push_back(op(MKF_OP_CANCEL, oi++, 80));     // cancel before ch1 part dispatched
    sc.steps.push_back(op(MKF_OP_SCHED, oi++, 1, 4));    // ch1 descriptor now stale -> idle
    sc.steps.push_back(op(MKF_OP_ADVANCE, oi++, 0, 8));
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_single_scope());
    v.push_back(sc_chiplet_two_level());
    v.push_back(sc_device_partial());
    v.push_back(sc_waits_readyscan());
    v.push_back(sc_cancel());
    v.push_back(sc_invalid());
    v.push_back(sc_stall_busy());
    v.push_back(sc_stress());
    v.push_back(sc_device_cancel());
    return v;
}

static void copy_result(MkfOracleResult* r,
                        GuardedDeviceBuffer<int64_t>& cnt,
                        GuardedDeviceBuffer<uint64_t>& evh,
                        GuardedDeviceBuffer<uint64_t>& th,
                        GuardedDeviceBuffer<uint64_t>& qh,
                        GuardedDeviceBuffer<uint64_t>& ch,
                        GuardedDeviceBuffer<uint64_t>& rh,
                        GuardedDeviceBuffer<uint64_t>& clk,
                        GuardedDeviceBuffer<uint64_t>& es) {
    std::vector<int64_t> c = cnt.download();
    for (int i = 0; i < MKF_COUNTER_COUNT; ++i) r->counters[i] = c[(size_t)i];
    r->fleet_event_hash = evh.download()[0];
    r->task_state_hash = th.download()[0];
    r->queue_hash = qh.download()[0];
    r->counter_hash = ch.download()[0];
    r->running_hash = rh.download()[0];
    r->clock_out = clk.download()[0];
    r->event_seq_out = es.download()[0];
}

static bool run_scenario_once(const Scenario& sc, bool verbose,
                              std::vector<MkfOracleResult>* results,
                              int* passed, int* total, std::string* first_error) {
    MkfProblemSpec spec = sc.spec;

    size_t workspace_bytes = solution_workspace_bytes(&spec);
    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MkfOracle oracle;
    oracle.init(spec);

    if (results) { results->clear(); results->reserve(sc.steps.size()); }

    bool all_ok = true;

    for (size_t i = 0; i < sc.steps.size(); ++i) {
        const MkfRunSpec& step = sc.steps[i];

        GuardedDeviceBuffer<int64_t> d_counters;
        GuardedDeviceBuffer<uint64_t> d_evhash, d_thash, d_qhash, d_chash, d_rhash, d_clock, d_evseq;
        d_counters.allocate(MKF_COUNTER_COUNT);
        d_evhash.allocate(1); d_thash.allocate(1); d_qhash.allocate(1);
        d_chash.allocate(1); d_rhash.allocate(1); d_clock.allocate(1); d_evseq.allocate(1);

        MkfOutputs outputs = {};
        outputs.counters = d_counters.ptr;
        outputs.fleet_event_hash = d_evhash.ptr;
        outputs.task_state_hash = d_thash.ptr;
        outputs.queue_hash = d_qhash.ptr;
        outputs.counter_hash = d_chash.ptr;
        outputs.running_hash = d_rhash.ptr;
        outputs.clock_out = d_clock.ptr;
        outputs.event_seq_out = d_evseq.ptr;

        std::string error;
        CUDA_CHECK(solution_run(state, &step, nullptr, &outputs, workspace.ptr, workspace_bytes, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool ok = true;
        ok = ok && d_counters.check_guards("counters", &error);
        ok = ok && d_evhash.check_guards("event_hash", &error);
        ok = ok && d_thash.check_guards("task_state_hash", &error);
        ok = ok && d_qhash.check_guards("queue_hash", &error);
        ok = ok && d_chash.check_guards("counter_hash", &error);
        ok = ok && d_rhash.check_guards("running_hash", &error);
        ok = ok && d_clock.check_guards("clock", &error);
        ok = ok && d_evseq.check_guards("event_seq", &error);

        MkfOracleResult got;
        copy_result(&got, d_counters, d_evhash, d_thash, d_qhash, d_chash, d_rhash, d_clock, d_evseq);

        MkfOracleResult expected;
        oracle.step_once(step, &expected);

        ok = ok && mkf_check(expected, got, &error);

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

        if (verbose) {
            std::printf("scenario %-22s step %02zu/%02zu op=%d clk=%llu %s%s%s\n",
                        sc.name.c_str(), i, sc.steps.size(), step.op,
                        (unsigned long long)got.clock_out,
                        ok ? "PASS" : "FAIL", ok ? "" : "  ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_runs(const std::vector<MkfOracleResult>& a,
                         const std::vector<MkfOracleResult>& b, std::string* err) {
    if (a.size() != b.size()) { if (err) *err = "step count mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        std::string e;
        if (!mkf_check(a[i], b[i], &e)) {
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
            std::vector<MkfOracleResult> base, replay;
            std::string error;
            bool ok_base = run_scenario_once(sc, true, &base, &passed, &total, &error);
            bool ok_replay = run_scenario_once(sc, false, &replay, &passed, &total, &error);
            if (ok_base && ok_replay && !base.empty()) {
                const int64_t* c = base.back().counters;
                std::printf("  [cov %-20s] rdy=%lld wait=%lld idle=%lld ds=%lld dcp=%lld ddp=%lld dst=%lld wst=%lld wb=%lld wi=%lld wf=%lld cf=%lld l2i=%lld l2f=%lld gai=%lld ef=%lld tc=%lld inv=%lld\n",
                    sc.name.c_str(),
                    (long long)c[0],(long long)c[1],(long long)c[2],(long long)c[3],(long long)c[4],
                    (long long)c[5],(long long)c[6],(long long)c[7],(long long)c[8],(long long)c[9],
                    (long long)c[10],(long long)c[11],(long long)c[12],(long long)c[13],(long long)c[14],
                    (long long)c[15],(long long)c[16],(long long)c[17]);
            }
            if (ok_base && ok_replay) {
                std::string ce;
                if (compare_runs(base, replay, &ce))
                    std::printf("scenario %-22s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-22s exact replay FAIL  %s\n", sc.name.c_str(), ce.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-22s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
