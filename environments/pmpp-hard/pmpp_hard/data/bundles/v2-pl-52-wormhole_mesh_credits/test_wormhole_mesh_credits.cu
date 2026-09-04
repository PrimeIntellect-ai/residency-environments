// file: test_wormhole_mesh_credits.cu
//
// 3-way validation harness for T52. Drives the linked solution (reference or
// naive) and compares every per-step output against the independent host
// oracle. Exercises >=6 adversarial scenarios plus exact-replay determinism.

#include "wormhole_mesh_credits_common.h"
#include "wormhole_mesh_credits_oracle.hpp"

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
    WmcProblemSpec spec;
    std::vector<WmcRunSpec> steps;
};

static WmcProblemSpec make_spec(int rows, int cols, int vc, int cap, int lat,
                                int max_pk, int max_injq, int max_ce) {
    WmcProblemSpec s = {};
    s.abi_version = WMC_ABI_VERSION;
    s.rows = rows; s.cols = cols; s.vc_count = vc; s.buffer_cap_per_vc = cap;
    s.credit_latency = lat; s.max_packets = max_pk;
    s.max_injection_queue_per_node = max_injq; s.max_credit_events = max_ce;
    return s;
}

static WmcRunSpec rs(int op, int op_index, int a0, int a1 = 0, int a2 = 0, int a3 = 0, int a4 = 0) {
    WmcRunSpec r = {};
    r.abi_version = WMC_ABI_VERSION;
    r.op = op; r.op_index = op_index;
    r.a0 = a0; r.a1 = a1; r.a2 = a2; r.a3 = a3; r.a4 = a4;
    return r;
}

static int NODE(const WmcProblemSpec& s, int row, int col) { return row * s.cols + col; }

// ---- scenario builders ----

// 1. Basic adaptive delivery across a 3x3 mesh with credit latency.
static Scenario sc_basic_adaptive() {
    Scenario sc; sc.name = "basic_adaptive_3x3";
    sc.spec = make_spec(3, 3, 3, 4, 2, 16, 8, 512);
    int oi = 0;
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 1, NODE(sc.spec,0,0), NODE(sc.spec,2,2), 4, 1));
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 2, NODE(sc.spec,0,1), NODE(sc.spec,2,0), 3, 1));
    for (int t = 0; t < 14; ++t) sc.steps.push_back(rs(WMC_OP_TICK, oi++, 1, 0, 0, 0, 0));
    return sc;
}

// 2. Escape-VC XY routing with a link failure forcing route stalls.
static Scenario sc_link_failure_escape() {
    Scenario sc; sc.name = "link_failure_escape_4x4";
    sc.spec = make_spec(4, 4, 2, 2, 3, 32, 8, 1024);
    int oi = 0;
    // packets prefer_adaptive=0 -> escape VC (XY).
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 10, NODE(sc.spec,0,0), NODE(sc.spec,0,3), 5, 0));
    sc.steps.push_back(rs(WMC_OP_TICK, oi++, 3, 0, 0, 0, 0));
    // break E link at node (0,1) -> XY needs E here; forces ROUTE_STALL.
    sc.steps.push_back(rs(WMC_OP_SET_LINK, oi++, NODE(sc.spec,0,1), WMC_PORT_E, 0));
    for (int t = 0; t < 6; ++t) sc.steps.push_back(rs(WMC_OP_TICK, oi++, 1, 0, 0, 0, 0));
    // restore link, let it drain.
    sc.steps.push_back(rs(WMC_OP_SET_LINK, oi++, NODE(sc.spec,0,1), WMC_PORT_E, 1));
    for (int t = 0; t < 8; ++t) sc.steps.push_back(rs(WMC_OP_TICK, oi++, 1, 0, 0, 0, 0));
    return sc;
}

// 3. Drop packets mid-flight: queued drop and in-buffer drain.
static Scenario sc_drop_packets() {
    Scenario sc; sc.name = "drop_mixed_3x4";
    sc.spec = make_spec(3, 4, 2, 3, 1, 16, 4, 512);
    int oi = 0;
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 100, NODE(sc.spec,0,0), NODE(sc.spec,2,3), 6, 1));
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 101, NODE(sc.spec,0,0), NODE(sc.spec,2,3), 5, 0));
    sc.steps.push_back(rs(WMC_OP_TICK, oi++, 2, 0, 0, 0, 0));
    // drop 100 while some flits are in flight, some still queued.
    sc.steps.push_back(rs(WMC_OP_DROP, oi++, 100, 0, 0, 0, 0));
    // drop 101 before it ever drains (still queued partly).
    sc.steps.push_back(rs(WMC_OP_TICK, oi++, 1, 0, 0, 0, 0));
    sc.steps.push_back(rs(WMC_OP_DROP, oi++, 101, 0, 0, 0, 0));
    for (int t = 0; t < 10; ++t) sc.steps.push_back(rs(WMC_OP_TICK, oi++, 1, 0, 0, 0, 0));
    // invalid: drop already-terminal.
    sc.steps.push_back(rs(WMC_OP_DROP, oi++, 100, 0, 0, 0, 0));
    return sc;
}

// 4. Credit pressure / congestion: small buffers, many packets, same dst.
static Scenario sc_congestion() {
    Scenario sc; sc.name = "congestion_4x4_cap1";
    sc.spec = make_spec(4, 4, 3, 1, 2, 64, 8, 2048);
    int oi = 0;
    int dst = NODE(sc.spec, 3, 3);
    for (int k = 0; k < 6; ++k) {
        int src = NODE(sc.spec, 0, k % 4);
        sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 200 + k, src, dst, 3, (k % 2)));
    }
    for (int t = 0; t < 24; ++t) sc.steps.push_back(rs(WMC_OP_TICK, oi++, 1, 0, 0, 0, 0));
    return sc;
}

// 5. Invalid ops, rejects, multi-tick batches, and re-used terminal ids.
static Scenario sc_invalid_and_reuse() {
    Scenario sc; sc.name = "invalid_reuse_2x3";
    sc.spec = make_spec(2, 3, 2, 2, 1, 4, 2, 256);
    int oi = 0;
    // invalid: src out of range, flit_count 0, dst out of range.
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 1, 999, 0, 2, 1));
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 2, 0, 1, 0, 1));
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 3, 0, 999, 2, 1));
    // invalid set_link (LOCAL, boundary), and valid one.
    sc.steps.push_back(rs(WMC_OP_SET_LINK, oi++, 0, WMC_PORT_LOCAL, 0));
    sc.steps.push_back(rs(WMC_OP_SET_LINK, oi++, 0, WMC_PORT_N, 0));   // boundary
    sc.steps.push_back(rs(WMC_OP_SET_LINK, oi++, 0, WMC_PORT_E, 0));   // valid
    // fill the small injection queue at node 0 (max_injq=2) to force reject.
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 10, 0, NODE(sc.spec,1,2), 2, 1));
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 11, 0, NODE(sc.spec,1,2), 2, 1));
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 12, 0, NODE(sc.spec,1,2), 2, 1)); // reject (injq full)
    // restore link then drain to terminal, then re-use id 10.
    sc.steps.push_back(rs(WMC_OP_SET_LINK, oi++, 0, WMC_PORT_E, 1));
    for (int t = 0; t < 12; ++t) sc.steps.push_back(rs(WMC_OP_TICK, oi++, 1, 0, 0, 0, 0));
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 10, 0, NODE(sc.spec,1,1), 1, 0)); // re-use terminal id
    sc.steps.push_back(rs(WMC_OP_TICK, oi++, 6, 0, 0, 0, 0));
    // invalid TICK 0.
    sc.steps.push_back(rs(WMC_OP_TICK, oi++, 0, 0, 0, 0, 0));
    return sc;
}

// 6. Large multi-tick stress on a 4x4 with mixed adaptive/escape and a failure.
static Scenario sc_stress_mixed() {
    Scenario sc; sc.name = "stress_mixed_4x4";
    sc.spec = make_spec(4, 4, 4, 2, 4, 64, 6, 4096);
    int oi = 0;
    for (int k = 0; k < 8; ++k) {
        int src = NODE(sc.spec, k % 4, (k * 3) % 4);
        int dst = NODE(sc.spec, (k + 2) % 4, (k + 1) % 4);
        if (src == dst) dst = NODE(sc.spec, 3, 3);
        sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 300 + k, src, dst, 2 + (k % 3), (k % 2)));
    }
    sc.steps.push_back(rs(WMC_OP_TICK, oi++, 5, 0, 0, 0, 0));
    sc.steps.push_back(rs(WMC_OP_SET_LINK, oi++, NODE(sc.spec,1,1), WMC_PORT_S, 0));
    sc.steps.push_back(rs(WMC_OP_SET_LINK, oi++, NODE(sc.spec,2,2), WMC_PORT_W, 0));
    sc.steps.push_back(rs(WMC_OP_TICK, oi++, 8, 0, 0, 0, 0));
    sc.steps.push_back(rs(WMC_OP_DROP, oi++, 303, 0, 0, 0, 0));
    sc.steps.push_back(rs(WMC_OP_SET_LINK, oi++, NODE(sc.spec,1,1), WMC_PORT_S, 1));
    sc.steps.push_back(rs(WMC_OP_TICK, oi++, 20, 0, 0, 0, 0));
    return sc;
}

// 7. Single-flit (SINGLE kind) packets and a credit_latency=0 fast-return case.
static Scenario sc_single_fastcredit() {
    Scenario sc; sc.name = "single_fastcredit_2x2";
    sc.spec = make_spec(2, 2, 2, 2, 0, 16, 4, 256);
    int oi = 0;
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 1, NODE(sc.spec,0,0), NODE(sc.spec,1,1), 1, 1));
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 2, NODE(sc.spec,0,0), NODE(sc.spec,1,1), 1, 0));
    sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 3, NODE(sc.spec,1,0), NODE(sc.spec,0,1), 1, 1));
    for (int t = 0; t < 10; ++t) sc.steps.push_back(rs(WMC_OP_TICK, oi++, 1, 0, 0, 0, 0));
    return sc;
}

// 8. High-latency funnel: long credit_latency so returns lag occupancy. This
// stresses the credit-vs-buffer coupling (the path that can yield CREDIT_CLAMP
// when stale returns pile up, and exercises deep credit-queue ordering).
static Scenario sc_high_latency_funnel() {
    Scenario sc; sc.name = "high_latency_funnel_4x1";
    sc.spec = make_spec(4, 1, 2, 2, 6, 32, 8, 4096);
    int oi = 0;
    int dst = NODE(sc.spec, 3, 0);
    for (int k = 0; k < 4; ++k)
        sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 400 + k, NODE(sc.spec, 0, 0), dst, 4, 0));
    for (int t = 0; t < 40; ++t) sc.steps.push_back(rs(WMC_OP_TICK, oi++, 1, 0, 0, 0, 0));
    return sc;
}

// 9. Toggle a link down then back up repeatedly while a flit waits, and inject
// after a drain, to drive bursts of credit returns at the same due_cycle
// (canonical-order ties on seq_created).
static Scenario sc_credit_burst() {
    Scenario sc; sc.name = "credit_burst_2x4";
    sc.spec = make_spec(2, 4, 2, 2, 3, 32, 8, 4096);
    int oi = 0;
    for (int k = 0; k < 4; ++k)
        sc.steps.push_back(rs(WMC_OP_INJECT, oi++, 500 + k, NODE(sc.spec, 0, k), NODE(sc.spec, 1, 0), 3, 1));
    for (int t = 0; t < 6; ++t) sc.steps.push_back(rs(WMC_OP_TICK, oi++, 1, 0, 0, 0, 0));
    sc.steps.push_back(rs(WMC_OP_TICK, oi++, 30, 0, 0, 0, 0));
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_basic_adaptive());
    v.push_back(sc_link_failure_escape());
    v.push_back(sc_drop_packets());
    v.push_back(sc_congestion());
    v.push_back(sc_invalid_and_reuse());
    v.push_back(sc_stress_mixed());
    v.push_back(sc_single_fastcredit());
    v.push_back(sc_high_latency_funnel());
    v.push_back(sc_credit_burst());
    return v;
}

static void copy_result(const WmcOutputs& dev, WmcOracleResult* r,
                        GuardedDeviceBuffer<int64_t>& cnt,
                        GuardedDeviceBuffer<uint64_t>& evh,
                        GuardedDeviceBuffer<uint64_t>& ph,
                        GuardedDeviceBuffer<uint64_t>& bh,
                        GuardedDeviceBuffer<uint64_t>& ch,
                        GuardedDeviceBuffer<uint64_t>& cqh,
                        GuardedDeviceBuffer<uint64_t>& cyc,
                        GuardedDeviceBuffer<uint64_t>& es) {
    (void)dev;
    std::vector<int64_t> c = cnt.download();
    for (int i = 0; i < WMC_COUNTER_COUNT; ++i) r->counters[i] = c[(size_t)i];
    r->fabric_event_hash = evh.download()[0];
    r->packet_hash = ph.download()[0];
    r->buffer_hash = bh.download()[0];
    r->credit_hash = ch.download()[0];
    r->credit_queue_hash = cqh.download()[0];
    r->cycle_out = cyc.download()[0];
    r->event_seq_out = es.download()[0];
}

static bool run_scenario_once(const Scenario& sc, bool verbose,
                              std::vector<WmcOracleResult>* results,
                              int* passed, int* total, std::string* first_error) {
    WmcProblemSpec spec = sc.spec;

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

    WmcOracle oracle;
    oracle.init(spec);

    if (results) { results->clear(); results->reserve(sc.steps.size()); }

    bool all_ok = true;

    for (size_t i = 0; i < sc.steps.size(); ++i) {
        const WmcRunSpec& step = sc.steps[i];

        GuardedDeviceBuffer<int64_t> d_counters;
        GuardedDeviceBuffer<uint64_t> d_evhash, d_phash, d_bhash, d_chash, d_cqhash, d_cycle, d_evseq;
        d_counters.allocate(WMC_COUNTER_COUNT);
        d_evhash.allocate(1); d_phash.allocate(1); d_bhash.allocate(1);
        d_chash.allocate(1); d_cqhash.allocate(1); d_cycle.allocate(1); d_evseq.allocate(1);

        WmcOutputs outputs = {};
        outputs.counters = d_counters.ptr;
        outputs.fabric_event_hash = d_evhash.ptr;
        outputs.packet_hash = d_phash.ptr;
        outputs.buffer_hash = d_bhash.ptr;
        outputs.credit_hash = d_chash.ptr;
        outputs.credit_queue_hash = d_cqhash.ptr;
        outputs.cycle_out = d_cycle.ptr;
        outputs.event_seq_out = d_evseq.ptr;

        std::string error;
        CUDA_CHECK(solution_run(state, &step, nullptr, &outputs, workspace.ptr, workspace_bytes, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool ok = true;
        ok = ok && d_counters.check_guards("counters", &error);
        ok = ok && d_evhash.check_guards("event_hash", &error);
        ok = ok && d_phash.check_guards("packet_hash", &error);
        ok = ok && d_bhash.check_guards("buffer_hash", &error);
        ok = ok && d_chash.check_guards("credit_hash", &error);
        ok = ok && d_cqhash.check_guards("credit_queue_hash", &error);
        ok = ok && d_cycle.check_guards("cycle", &error);
        ok = ok && d_evseq.check_guards("event_seq", &error);

        WmcOracleResult got;
        copy_result(outputs, &got, d_counters, d_evhash, d_phash, d_bhash, d_chash, d_cqhash, d_cycle, d_evseq);

        WmcOracleResult expected;
        oracle.step_once(step, &expected);

        ok = ok && wmc_check(expected, got, &error);

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
            std::printf("scenario %-26s step %02zu/%02zu op=%d cyc=%llu %s%s%s\n",
                        sc.name.c_str(), i, sc.steps.size(), step.op,
                        (unsigned long long)got.cycle_out,
                        ok ? "PASS" : "FAIL", ok ? "" : "  ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_runs(const std::vector<WmcOracleResult>& a,
                         const std::vector<WmcOracleResult>& b, std::string* err) {
    if (a.size() != b.size()) { if (err) *err = "step count mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        std::string e;
        if (!wmc_check(a[i], b[i], &e)) {
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
            std::vector<WmcOracleResult> base, replay;
            std::string error;
            bool ok_base = run_scenario_once(sc, true, &base, &passed, &total, &error);
            bool ok_replay = run_scenario_once(sc, false, &replay, &passed, &total, &error);
            if (ok_base && ok_replay && !base.empty()) {
                const int64_t* c = base.back().counters;
                std::printf("  [cov %-22s] acc=%lld rej=%lld lnk=%lld drp=%lld inj=%lld mvE=%lld mvA=%lld ej=%lld done=%lld dF=%lld cR=%lld cC=%lld iS=%lld rS=%lld oS=%lld cS=%lld eW=%lld dsy=%lld inv=%lld\n",
                    sc.name.c_str(),
                    (long long)c[0],(long long)c[1],(long long)c[2],(long long)c[3],(long long)c[4],
                    (long long)c[5],(long long)c[6],(long long)c[7],(long long)c[8],(long long)c[9],
                    (long long)c[10],(long long)c[11],(long long)c[12],(long long)c[13],(long long)c[14],
                    (long long)c[15],(long long)c[16],(long long)c[17],(long long)c[18]);
            }
            if (ok_base && ok_replay) {
                std::string ce;
                if (compare_runs(base, replay, &ce))
                    std::printf("scenario %-26s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-26s exact replay FAIL  %s\n", sc.name.c_str(), ce.c_str()); }
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
