// file: test_beps_tree_buffer.cu

#include "beps_tree_buffer_common.h"
#include "beps_tree_buffer_oracle.hpp"

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
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                   \
            throw std::runtime_error(_oss.str());                              \
        }                                                                       \
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
    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("upload size mismatch");
        if (count > 0) CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost));
        return host;
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
            if (left[i] != kGuardByte) {
                if (error) { std::ostringstream o; o << "left guard corrupted for " << name << " at byte " << i; *error = o.str(); }
                return false;
            }
            if (right[i] != kGuardByte) {
                if (error) { std::ostringstream o; o << "right guard corrupted for " << name << " at byte " << i; *error = o.str(); }
                return false;
            }
        }
        return true;
    }
};

// ---- scenario model ----
struct StepHost { std::vector<BepsOp> ops; };
struct Scenario {
    std::string name;
    BepsProblemSpec spec;
    std::vector<StepHost> steps;
};

static BepsOp put(uint64_t key, int64_t val) {
    BepsOp o = {}; o.kind = BEPS_OP_PUT; o.u_key = key; o.value = val; return o;
}
static BepsOp add(uint64_t key, int64_t delta) {
    BepsOp o = {}; o.kind = BEPS_OP_ADD; o.u_key = key; o.value = delta; return o;
}
static BepsOp del(uint64_t key) {
    BepsOp o = {}; o.kind = BEPS_OP_DELETE; o.u_key = key; return o;
}
static BepsOp pq(uint64_t read_id, uint64_t key) {
    BepsOp o = {}; o.kind = BEPS_OP_POINT_QUERY; o.u_aux = read_id; o.u_key = key; return o;
}
static BepsOp rq(uint64_t read_id, uint64_t lo, uint64_t hi, int64_t limit) {
    BepsOp o = {}; o.kind = BEPS_OP_RANGE_QUERY; o.u_aux = read_id; o.u_key = lo;
    o.u_key2 = hi; o.value = limit; return o;
}
static BepsOp flush(int64_t max_nodes, int64_t max_msgs) {
    BepsOp o = {}; o.kind = BEPS_OP_FLUSH; o.value = max_nodes; o.value2 = max_msgs; return o;
}

static BepsProblemSpec mk_spec(int max_nodes, int ibc, int lrc, int mc, int fmc,
                               int mrr, int max_ops, int max_steps) {
    BepsProblemSpec s = {};
    s.abi_version = BEPS_ABI_VERSION;
    s.max_nodes = max_nodes;
    s.internal_buffer_cap = ibc;
    s.leaf_record_cap = lrc;
    s.max_children_per_internal = mc;
    s.flush_message_cap = fmc;
    s.max_range_results = mrr;
    s.max_ops = max_ops;
    s.max_steps = max_steps;
    return s;
}

// 1: leaf root fills, splits, queries compose.
static Scenario sc_leaf_split() {
    Scenario sc; sc.name = "leaf_split";
    sc.spec = mk_spec(512, 4, 4, 4, 8, 64, 64, 8);
    StepHost s;
    for (uint64_t k = 0; k < 6; ++k) s.ops.push_back(put(k * 10, (int64_t)k + 1));
    s.ops.push_back(pq(1, 30));
    s.ops.push_back(pq(2, 999));
    s.ops.push_back(add(20, 100));
    s.ops.push_back(del(0));
    s.ops.push_back(pq(3, 0));
    s.ops.push_back(pq(4, 20));
    s.ops.push_back(rq(5, 0, 100, 50));
    sc.steps.push_back(s);
    return sc;
}

// 2: internal-root buffering + manual flush cascades to leaves.
static Scenario sc_flush_cascade() {
    Scenario sc; sc.name = "flush_cascade";
    sc.spec = mk_spec(512, 3, 3, 4, 4, 64, 128, 8);
    StepHost s;
    // build a multi-leaf tree first (force root to become internal)
    for (uint64_t k = 0; k < 8; ++k) s.ops.push_back(put(k, (int64_t)(k + 1)));
    // now buffer a batch into internal root, then flush
    for (uint64_t k = 0; k < 6; ++k) s.ops.push_back(add(k, 5));
    s.ops.push_back(flush(2, 100));   // a couple flush groups
    s.ops.push_back(pq(10, 0));
    s.ops.push_back(pq(11, 7));
    s.ops.push_back(flush(100, 100)); // drain
    s.ops.push_back(flush(100, 100)); // FLUSH_EMPTY expected
    s.ops.push_back(rq(12, 0, 7, 50));
    sc.steps.push_back(s);
    return sc;
}

// 3: write-stall via tiny buffer cap + automatic root flush on full.
static Scenario sc_write_stall() {
    Scenario sc; sc.name = "write_stall";
    sc.spec = mk_spec(512, 2, 2, 3, 1, 64, 128, 8);
    StepHost s;
    // grow to internal root
    for (uint64_t k = 0; k < 6; ++k) s.ops.push_back(put(k * 5, (int64_t)k));
    // hammer the same key region to fill the root buffer and trigger
    // auto-flush + potential WRITE_STALL (fmc=1 so each auto-flush moves 1).
    for (int r = 0; r < 12; ++r) s.ops.push_back(add(0, 1));
    s.ops.push_back(pq(1, 0));
    sc.steps.push_back(s);
    return sc;
}

// 4: deletes, tombstones, resurrection by ADD, point + range visibility.
static Scenario sc_tombstone_resurrect() {
    Scenario sc; sc.name = "tombstone_resurrect";
    sc.spec = mk_spec(512, 4, 4, 4, 8, 64, 128, 8);
    StepHost s;
    for (uint64_t k = 0; k < 5; ++k) s.ops.push_back(put(k, 100));
    s.ops.push_back(del(2));
    s.ops.push_back(pq(1, 2));        // missing
    s.ops.push_back(add(2, 7));       // resurrect to 7
    s.ops.push_back(pq(2, 2));        // found 7
    s.ops.push_back(del(2));
    s.ops.push_back(put(2, 55));      // set after delete
    s.ops.push_back(pq(3, 2));        // found 55
    s.ops.push_back(rq(4, 0, 4, 50));
    sc.steps.push_back(s);
    return sc;
}

// 5: persistent multi-step with internal splits cascading upward.
static Scenario sc_persistent_splits() {
    Scenario sc; sc.name = "persistent_splits";
    sc.spec = mk_spec(1024, 4, 2, 3, 4, 64, 64, 8);
    StepHost s1;
    for (uint64_t k = 0; k < 12; ++k) s1.ops.push_back(put(k, (int64_t)k));
    sc.steps.push_back(s1);
    StepHost s2;
    for (uint64_t k = 12; k < 24; ++k) s2.ops.push_back(put(k, (int64_t)k));
    s2.ops.push_back(flush(100, 1000));
    sc.steps.push_back(s2);
    StepHost s3;
    for (uint64_t k = 0; k < 24; k += 2) s3.ops.push_back(add(k, 1000));
    s3.ops.push_back(flush(100, 1000));
    s3.ops.push_back(rq(1, 0, 23, 100));
    s3.ops.push_back(pq(2, 5));
    s3.ops.push_back(pq(3, 23));
    sc.steps.push_back(s3);
    return sc;
}

// 6: invalid range, flush budget no-op, limit clamping.
static Scenario sc_edge_budgets() {
    Scenario sc; sc.name = "edge_budgets";
    sc.spec = mk_spec(512, 3, 3, 4, 2, 3 /*small mrr*/, 64, 8);
    StepHost s;
    for (uint64_t k = 0; k < 8; ++k) s.ops.push_back(put(k, (int64_t)k + 1));
    for (uint64_t k = 0; k < 6; ++k) s.ops.push_back(add(k, 10));
    s.ops.push_back(flush(0, 100));       // budget zero -> no-op (no FLUSH_EMPTY)
    s.ops.push_back(flush(100, 0));       // budget zero -> no-op
    s.ops.push_back(rq(1, 100, 0, 50));   // invalid lo>hi
    s.ops.push_back(rq(2, 0, 7, 50));     // limited by mrr=3
    s.ops.push_back(rq(3, 0, 7, 2));      // limited by limit=2
    s.ops.push_back(flush(100, 1000));
    s.ops.push_back(rq(4, 0, 7, 50));
    sc.steps.push_back(s);
    return sc;
}

// 7: adversarial pseudo-random mix across many steps.
static Scenario sc_random_mix() {
    Scenario sc; sc.name = "random_mix";
    sc.spec = mk_spec(4096, 4, 3, 4, 3, 64, 96, 16);
    uint64_t rng = 0xDEADBEEF12345ULL;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    const uint64_t kKeyMod = 40;
    for (int step = 0; step < 12; ++step) {
        StepHost s;
        int ops = 6 + (int)(next() % 8);
        for (int o = 0; o < ops; ++o) {
            int r = (int)(next() % 100);
            uint64_t key = next() % kKeyMod;
            if (r < 35) s.ops.push_back(put(key, (int64_t)(next() % 1000)));
            else if (r < 55) s.ops.push_back(add(key, (int64_t)(next() % 50) - 25));
            else if (r < 65) s.ops.push_back(del(key));
            else if (r < 78) s.ops.push_back(pq(next() % 1000, key));
            else if (r < 88) {
                uint64_t lo = next() % kKeyMod, hi = next() % kKeyMod;
                if (lo > hi) { uint64_t t = lo; lo = hi; hi = t; }
                s.ops.push_back(rq(next() % 1000, lo, hi, (int64_t)(next() % 20)));
            } else {
                s.ops.push_back(flush((int64_t)(next() % 4), (int64_t)(next() % 12)));
            }
        }
        sc.steps.push_back(s);
    }
    return sc;
}

// 8: large fan-out then heavy flush drain + range scan.
static Scenario sc_fanout_drain() {
    Scenario sc; sc.name = "fanout_drain";
    sc.spec = mk_spec(8192, 5, 4, 5, 6, 256, 256, 8);
    StepHost s;
    for (uint64_t k = 0; k < 60; ++k) s.ops.push_back(put(k, (int64_t)k));
    sc.steps.push_back(s);
    StepHost s2;
    for (uint64_t k = 0; k < 60; ++k) s2.ops.push_back(add(k, 1));
    for (int i = 0; i < 8; ++i) s2.ops.push_back(flush(50, 1000));
    s2.ops.push_back(rq(1, 0, 59, 200));
    sc.steps.push_back(s2);
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_leaf_split());
    v.push_back(sc_flush_cascade());
    v.push_back(sc_write_stall());
    v.push_back(sc_tombstone_resurrect());
    v.push_back(sc_persistent_splits());
    v.push_back(sc_edge_budgets());
    v.push_back(sc_random_mix());
    v.push_back(sc_fanout_drain());
    return v;
}

struct StepResult {
    BepsCounts counts;
    uint64_t meh, qh, tsh, bh, lh;
};

static bool run_scenario_once(const Scenario& sc, bool verbose,
                              std::vector<StepResult>* results, int* passed,
                              int* total, std::string* first_error) {
    BepsProblemSpec spec = sc.spec;
    if (!beps_validate_problem_spec(&spec)) {
        if (first_error) *first_error = "invalid problem spec";
        return false;
    }

    size_t workspace_bytes = solution_workspace_bytes(&spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    BepsOracleState oracle;
    oracle.init(spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) results->clear();
    bool all_ok = true;

    for (size_t si = 0; si < sc.steps.size(); ++si) {
        const StepHost& step = sc.steps[si];
        const int num_ops = (int)step.ops.size();

        DeviceBuffer<BepsOp> d_ops;
        d_ops.allocate(std::max<size_t>(num_ops, 1));
        {
            std::vector<BepsOp> hops = step.ops;
            hops.resize(d_ops.count);
            d_ops.upload(hops);
        }

        GuardedDeviceBuffer<BepsCounts> g_counts;
        GuardedDeviceBuffer<uint64_t> g_meh, g_qh, g_tsh, g_bh, g_lh;
        g_counts.allocate(1);
        g_meh.allocate(1); g_qh.allocate(1); g_tsh.allocate(1);
        g_bh.allocate(1); g_lh.allocate(1);

        BepsRunSpec run = {};
        run.abi_version = BEPS_ABI_VERSION;
        run.num_ops = num_ops;
        run.step_id = (int32_t)si;
        if (!beps_validate_run_spec(&run)) throw std::runtime_error("invalid run spec");

        BepsInputs inputs = {};
        inputs.ops = d_ops.ptr;

        BepsOutputs outputs = {};
        outputs.counts = g_counts.ptr;
        outputs.message_event_hash = g_meh.ptr;
        outputs.query_hash = g_qh.ptr;
        outputs.tree_shape_hash = g_tsh.ptr;
        outputs.buffer_hash = g_bh.ptr;
        outputs.leaf_hash = g_lh.ptr;

        std::string error;
        CUDA_CHECK(solution_run(state, &run, &inputs, &outputs, workspace.ptr,
                                std::max<size_t>(workspace_bytes, 1), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool ok = true;
        ok = ok && g_counts.check_guards("counts", &error);
        ok = ok && g_meh.check_guards("message_event_hash", &error);
        ok = ok && g_qh.check_guards("query_hash", &error);
        ok = ok && g_tsh.check_guards("tree_shape_hash", &error);
        ok = ok && g_bh.check_guards("buffer_hash", &error);
        ok = ok && g_lh.check_guards("leaf_hash", &error);

        // input immutability
        if (ok) {
            std::vector<BepsOp> after = d_ops.download();
            std::vector<BepsOp> before = step.ops; before.resize(d_ops.count);
            if (std::memcmp(after.data(), before.data(), sizeof(BepsOp) * d_ops.count) != 0) {
                ok = false; error = "ops input modified";
            }
        }

        BepsCounts h_counts = g_counts.download_data()[0];
        uint64_t h_meh = g_meh.download_data()[0];
        uint64_t h_qh = g_qh.download_data()[0];
        uint64_t h_tsh = g_tsh.download_data()[0];
        uint64_t h_bh = g_bh.download_data()[0];
        uint64_t h_lh = g_lh.download_data()[0];

        BepsHostInputsView hin = {}; hin.ops = step.ops.data();
        oracle.step(run, hin);

        BepsHostOutputsView got = {};
        got.counts = &h_counts;
        got.message_event_hash = &h_meh;
        got.query_hash = &h_qh;
        got.tree_shape_hash = &h_tsh;
        got.buffer_hash = &h_bh;
        got.leaf_hash = &h_lh;

        ok = ok && beps_check_outputs(oracle, got, &error);

        if (results) {
            StepResult r;
            r.counts = h_counts; r.meh = h_meh; r.qh = h_qh;
            r.tsh = h_tsh; r.bh = h_bh; r.lh = h_lh;
            results->push_back(r);
        }

        ++(*total);
        if (ok) ++(*passed);
        else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream oss; oss << sc.name << " step " << si << ": " << error;
                *first_error = oss.str();
            }
        }
        if (verbose) {
            std::printf("scenario %-22s step %02zu/%02zu ops=%3d %s%s%s\n",
                        sc.name.c_str(), si, sc.steps.size(), num_ops,
                        ok ? "PASS" : "FAIL", ok ? "" : "  ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(const std::vector<StepResult>& a,
                            const std::vector<StepResult>& b, std::string* error) {
    if (a.size() != b.size()) { if (error) *error = "step count mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i].counts, &b[i].counts, sizeof(BepsCounts)) != 0 ||
            a[i].meh != b[i].meh || a[i].qh != b[i].qh || a[i].tsh != b[i].tsh ||
            a[i].bh != b[i].bh || a[i].lh != b[i].lh) {
            if (error) { std::ostringstream o; o << "replay mismatch at step " << i; *error = o.str(); }
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
            std::vector<StepResult> base, replay;
            std::string error;
            const bool ok_base = run_scenario_once(sc, true, &base, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay, &passed, &total, &error);

            if (ok_base && ok_replay) {
                std::string cerr;
                if (compare_results(base, replay, &cerr))
                    std::printf("scenario %-22s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-22s exact replay FAIL  %s\n", sc.name.c_str(), cerr.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-22s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
