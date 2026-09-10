// file: test_deterministic_mergeable_quantile_sketch.cu

#include "deterministic_mergeable_quantile_sketch_common.h"
#include "deterministic_mergeable_quantile_sketch_oracle.hpp"

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

static constexpr uint64_t g_state = 0x51d3c7a90b264e8fULL;
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
    int32_t next_i32() { return (int32_t)(next_u64() >> 32); }
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
    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("DeviceBuffer upload size mismatch");
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

struct StepHost {
    DmqsRunSpec run;
    std::vector<int32_t> keys;            // INGEST
    std::vector<int32_t> merge_level_size;  // MERGE [num_levels]
    std::vector<int32_t> merge_keys;        // MERGE [num_levels * k]
};

struct StepResult {
    int64_t total_weight = 0;
    int32_t num_levels = 0;
    int32_t num_retained_items = 0;
    int32_t query_result = INT_MIN;
    uint64_t sketch_checksum = 0;
    uint64_t state_checksum = 0;
};

struct Scenario {
    std::string name;
    DmqsProblemSpec spec;
    std::vector<StepHost> steps;
};

static DmqsProblemSpec make_spec(int k, int num_levels, int max_batch, int max_steps) {
    DmqsProblemSpec spec = {};
    spec.abi_version = DMQS_ABI_VERSION;
    spec.k = k;
    spec.num_levels = num_levels;
    spec.max_batch = max_batch;
    spec.max_steps = max_steps;
    spec.flags = 0;
    if (!dmqs_validate_problem_spec(&spec)) throw std::runtime_error("invalid DmqsProblemSpec generated");
    return spec;
}

static StepHost make_ingest(const DmqsProblemSpec& spec, int step_id, const std::vector<int32_t>& keys) {
    StepHost st;
    st.run = {};
    st.run.abi_version = DMQS_ABI_VERSION;
    st.run.op = DMQS_OP_INGEST;
    st.run.batch_size = (int32_t)keys.size();
    st.run.step_id = step_id;
    if (!dmqs_validate_run_spec(&st.run, &spec)) throw std::runtime_error("invalid ingest run spec");
    st.keys = keys;
    if (st.keys.empty()) st.keys.assign(1, 0);  // never zero-size device buffer
    st.merge_level_size.assign((size_t)spec.num_levels, 0);
    st.merge_keys.assign((size_t)spec.num_levels * (size_t)spec.k, 0);
    return st;
}

static StepHost make_query(const DmqsProblemSpec& spec, int step_id, int q_num, int q_den) {
    StepHost st;
    st.run = {};
    st.run.abi_version = DMQS_ABI_VERSION;
    st.run.op = DMQS_OP_QUERY;
    st.run.batch_size = 0;
    st.run.q_num = q_num;
    st.run.q_den = q_den;
    st.run.step_id = step_id;
    if (!dmqs_validate_run_spec(&st.run, &spec)) throw std::runtime_error("invalid query run spec");
    st.keys.assign(1, 0);
    st.merge_level_size.assign((size_t)spec.num_levels, 0);
    st.merge_keys.assign((size_t)spec.num_levels * (size_t)spec.k, 0);
    return st;
}

static StepHost make_merge(
    const DmqsProblemSpec& spec,
    int step_id,
    int merge_num_levels,
    const std::vector<int32_t>& level_size,
    const std::vector<int32_t>& keys_rowmajor) {
    StepHost st;
    st.run = {};
    st.run.abi_version = DMQS_ABI_VERSION;
    st.run.op = DMQS_OP_MERGE;
    st.run.batch_size = 0;
    st.run.merge_num_levels = merge_num_levels;
    st.run.step_id = step_id;
    if (!dmqs_validate_run_spec(&st.run, &spec)) throw std::runtime_error("invalid merge run spec");
    st.keys.assign(1, 0);
    st.merge_level_size.assign((size_t)spec.num_levels, 0);
    for (int L = 0; L < merge_num_levels; ++L) st.merge_level_size[(size_t)L] = level_size[(size_t)L];
    st.merge_keys.assign((size_t)spec.num_levels * (size_t)spec.k, 0);
    for (size_t i = 0; i < keys_rowmajor.size() && i < st.merge_keys.size(); ++i)
        st.merge_keys[i] = keys_rowmajor[i];
    return st;
}

// ---------------------------------------------------------------------------
// Scenario builders
// ---------------------------------------------------------------------------

// 1. Heavy overflow cascade: small k forces many multi-level compactions.
static Scenario make_overflow_cascade_scenario() {
    Scenario sc;
    sc.name = "overflow_cascade_small_k";
    sc.spec = make_spec(4, 16, 256, 48);
    SplitMix64 rng(g_state ^ 0x1111ULL);

    sc.steps.push_back(make_query(sc.spec, 0, 1, 2));  // empty query
    for (int s = 1; s < 30; ++s) {
        if (s % 5 == 0) {
            const int qn = (s % 3 == 0) ? 1 : 2;
            sc.steps.push_back(make_query(sc.spec, s, qn, 4));
        } else {
            const int batch = 1 + (s * 7) % 40;
            std::vector<int32_t> keys;
            for (int i = 0; i < batch; ++i) keys.push_back(rng.uniform_int(-200, 200));
            sc.steps.push_back(make_ingest(sc.spec, s, keys));
        }
    }
    return sc;
}

// 2. Big single-batch overflow (batch >> k) plus boundary queries.
static Scenario make_big_batch_scenario() {
    Scenario sc;
    sc.name = "big_batch_boundaries";
    sc.spec = make_spec(8, 20, 4096, 32);
    SplitMix64 rng(g_state ^ 0x2222ULL);

    {
        std::vector<int32_t> keys;
        for (int i = 0; i < 1000; ++i) keys.push_back(rng.uniform_int(-1000000, 1000000));
        sc.steps.push_back(make_ingest(sc.spec, 0, keys));
    }
    sc.steps.push_back(make_query(sc.spec, 1, 0, 1));    // min
    sc.steps.push_back(make_query(sc.spec, 2, 1, 1));    // max
    sc.steps.push_back(make_query(sc.spec, 3, 1, 2));    // median
    {
        std::vector<int32_t> keys;
        for (int i = 0; i < 2500; ++i) keys.push_back(rng.uniform_int(-1000000, 1000000));
        sc.steps.push_back(make_ingest(sc.spec, 4, keys));
    }
    for (int s = 5; s < 28; ++s) {
        if (s % 3 == 0) {
            sc.steps.push_back(make_query(sc.spec, s, (s % 4), 4));
        } else {
            std::vector<int32_t> keys;
            const int batch = 50 + (s * 37) % 900;
            for (int i = 0; i < batch; ++i) keys.push_back(rng.uniform_int(-1000000, 1000000));
            sc.steps.push_back(make_ingest(sc.spec, s, keys));
        }
    }
    return sc;
}

// 3. Heavy duplicate keys / ties.
static Scenario make_duplicate_keys_scenario() {
    Scenario sc;
    sc.name = "duplicate_keys_ties";
    sc.spec = make_spec(8, 16, 512, 40);
    SplitMix64 rng(g_state ^ 0x3333ULL);

    for (int s = 0; s < 30; ++s) {
        if (s % 4 == 0) {
            sc.steps.push_back(make_query(sc.spec, s, (s % 5), 4));
        } else {
            std::vector<int32_t> keys;
            const int batch = 10 + (s * 11) % 60;
            // only a few distinct key values -> many ties
            for (int i = 0; i < batch; ++i) keys.push_back(rng.uniform_int(0, 5));
            sc.steps.push_back(make_ingest(sc.spec, s, keys));
        }
    }
    return sc;
}

// 4. Merge cascade: ingest into base, then merge a hand-built foreign sketch.
static Scenario make_merge_scenario() {
    Scenario sc;
    sc.name = "merge_cascade";
    sc.spec = make_spec(4, 16, 256, 40);
    SplitMix64 rng(g_state ^ 0x4444ULL);

    // seed the base sketch
    for (int s = 0; s < 6; ++s) {
        std::vector<int32_t> keys;
        const int batch = 3 + (s * 5) % 20;
        for (int i = 0; i < batch; ++i) keys.push_back(rng.uniform_int(-50, 50));
        sc.steps.push_back(make_ingest(sc.spec, s, keys));
    }
    sc.steps.push_back(make_query(sc.spec, 6, 1, 2));

    // build several foreign sketches and merge them
    int step_id = 7;
    for (int m = 0; m < 6; ++m) {
        const int k = sc.spec.k;
        const int mlev = 3 + (m % 3);
        std::vector<int32_t> level_size(sc.spec.num_levels, 0);
        std::vector<int32_t> keys(sc.spec.num_levels * k, 0);
        for (int L = 0; L < mlev; ++L) {
            // foreign level L holds between 1 and k-1 keys (valid: < k)
            const int cnt = 1 + (int)((rng.next_u64()) % (uint64_t)(k - 1));
            level_size[(size_t)L] = cnt;
            for (int j = 0; j < cnt; ++j) keys[(size_t)L * (size_t)k + (size_t)j] = rng.uniform_int(-50, 50);
        }
        sc.steps.push_back(make_merge(sc.spec, step_id++, mlev, level_size, keys));
        if (m % 2 == 1) sc.steps.push_back(make_query(sc.spec, step_id++, m % 4, 4));
    }

    for (int s = step_id; s < 38; ++s) {
        std::vector<int32_t> keys;
        const int batch = 4 + (s * 9) % 30;
        for (int i = 0; i < batch; ++i) keys.push_back(rng.uniform_int(-50, 50));
        sc.steps.push_back(make_ingest(sc.spec, s, keys));
    }
    return sc;
}

// 5. Parity stress: k=2 (every level holds at most 1 item between ops),
//    forces a long compaction chain with strict parity dependence.
static Scenario make_k2_parity_scenario() {
    Scenario sc;
    sc.name = "k2_parity_chain";
    sc.spec = make_spec(2, 24, 1024, 40);
    SplitMix64 rng(g_state ^ 0x5555ULL);

    for (int s = 0; s < 30; ++s) {
        if (s % 6 == 0) {
            sc.steps.push_back(make_query(sc.spec, s, (s % 3), 2));
        } else {
            std::vector<int32_t> keys;
            const int batch = 1 + (s * 3) % 100;
            for (int i = 0; i < batch; ++i) keys.push_back(rng.uniform_int(-30, 30));
            sc.steps.push_back(make_ingest(sc.spec, s, keys));
        }
    }
    return sc;
}

// 6. Large k, interleaved ingest/query/merge, wide value range incl extremes.
static Scenario make_large_k_mixed_scenario() {
    Scenario sc;
    sc.name = "large_k_mixed";
    sc.spec = make_spec(64, 20, 2048, 48);
    SplitMix64 rng(g_state ^ 0x6666ULL);

    for (int s = 0; s < 40; ++s) {
        const int r = s % 7;
        if (r == 3) {
            sc.steps.push_back(make_query(sc.spec, s, (s % 6), 6));
        } else if (r == 5) {
            const int k = sc.spec.k;
            const int mlev = 2 + (s % 3);
            std::vector<int32_t> level_size(sc.spec.num_levels, 0);
            std::vector<int32_t> keys(sc.spec.num_levels * k, 0);
            for (int L = 0; L < mlev; ++L) {
                const int cnt = 1 + (int)(rng.next_u64() % (uint64_t)(k - 1));
                level_size[(size_t)L] = cnt;
                for (int j = 0; j < cnt; ++j) {
                    int32_t v = rng.uniform_int(-2000000000, 2000000000);
                    keys[(size_t)L * (size_t)k + (size_t)j] = v;
                }
            }
            sc.steps.push_back(make_merge(sc.spec, s, mlev, level_size, keys));
        } else {
            std::vector<int32_t> keys;
            const int batch = 20 + (s * 53) % 600;
            for (int i = 0; i < batch; ++i) {
                int32_t v;
                if (i % 101 == 0) v = INT_MIN;
                else if (i % 103 == 0) v = INT_MAX;
                else v = rng.uniform_int(-2000000000, 2000000000);
                keys.push_back(v);
            }
            sc.steps.push_back(make_ingest(sc.spec, s, keys));
        }
    }
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(make_overflow_cascade_scenario());
    v.push_back(make_big_batch_scenario());
    v.push_back(make_duplicate_keys_scenario());
    v.push_back(make_merge_scenario());
    v.push_back(make_k2_parity_scenario());
    v.push_back(make_large_k_mixed_scenario());
    return v;
}

static bool check_input_unchanged(
    const StepHost& step,
    const DeviceBuffer<int32_t>& d_keys,
    const DeviceBuffer<int32_t>& d_msize,
    const DeviceBuffer<int32_t>& d_mkeys,
    std::string* error) {
    if (d_keys.download() != step.keys) { if (error) *error = "input keys modified"; return false; }
    if (d_msize.download() != step.merge_level_size) { if (error) *error = "input merge_level_size modified"; return false; }
    if (d_mkeys.download() != step.merge_keys) { if (error) *error = "input merge_keys modified"; return false; }
    return true;
}

static bool run_one_step(
    const DmqsProblemSpec& spec,
    const StepHost& step,
    DmqsOracleState* oracle,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    StepResult* result,
    std::string* error) {
    DeviceBuffer<int32_t> d_keys, d_msize, d_mkeys;
    d_keys.allocate(step.keys.size());
    d_msize.allocate(step.merge_level_size.size());
    d_mkeys.allocate(step.merge_keys.size());
    d_keys.upload(step.keys);
    d_msize.upload(step.merge_level_size);
    d_mkeys.upload(step.merge_keys);

    GuardedDeviceBuffer<int64_t> d_total_weight;
    GuardedDeviceBuffer<int32_t> d_num_levels;
    GuardedDeviceBuffer<int32_t> d_num_retained;
    GuardedDeviceBuffer<int32_t> d_query_result;
    GuardedDeviceBuffer<uint64_t> d_sketch_checksum;
    GuardedDeviceBuffer<uint64_t> d_state_checksum;

    d_total_weight.allocate(1);
    d_num_levels.allocate(1);
    d_num_retained.allocate(1);
    d_query_result.allocate(1);
    d_sketch_checksum.allocate(1);
    d_state_checksum.allocate(1);

    DmqsInputs inputs = {};
    inputs.keys = d_keys.ptr;
    inputs.merge_level_size = d_msize.ptr;
    inputs.merge_keys = d_mkeys.ptr;

    DmqsOutputs outputs = {};
    outputs.total_weight = d_total_weight.ptr;
    outputs.num_levels = d_num_levels.ptr;
    outputs.num_retained_items = d_num_retained.ptr;
    outputs.query_result = d_query_result.ptr;
    outputs.sketch_checksum = d_sketch_checksum.ptr;
    outputs.state_checksum = d_state_checksum.ptr;

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(step, d_keys, d_msize, d_mkeys, error)) return false;

    if (!d_total_weight.check_guards("total_weight", error)) return false;
    if (!d_num_levels.check_guards("num_levels", error)) return false;
    if (!d_num_retained.check_guards("num_retained_items", error)) return false;
    if (!d_query_result.check_guards("query_result", error)) return false;
    if (!d_sketch_checksum.check_guards("sketch_checksum", error)) return false;
    if (!d_state_checksum.check_guards("state_checksum", error)) return false;

    const std::vector<int64_t> h_tw = d_total_weight.download_data();
    const std::vector<int32_t> h_nl = d_num_levels.download_data();
    const std::vector<int32_t> h_nr = d_num_retained.download_data();
    const std::vector<int32_t> h_qr = d_query_result.download_data();
    const std::vector<uint64_t> h_sk = d_sketch_checksum.download_data();
    const std::vector<uint64_t> h_stt = d_state_checksum.download_data();

    DmqsHostInputsView host_inputs = {};
    host_inputs.keys = step.keys.data();
    host_inputs.merge_level_size = step.merge_level_size.data();
    host_inputs.merge_keys = step.merge_keys.data();

    DmqsExpected expected;
    oracle->step_once(step.run, host_inputs, &expected);

    DmqsHostOutputsView got = {};
    got.total_weight = h_tw.data();
    got.num_levels = h_nl.data();
    got.num_retained_items = h_nr.data();
    got.query_result = h_qr.data();
    got.sketch_checksum = h_sk.data();
    got.state_checksum = h_stt.data();

    if (!dmqs_check_all_outputs(spec, expected, got, error)) return false;

    if (result) {
        result->total_weight = h_tw[0];
        result->num_levels = h_nl[0];
        result->num_retained_items = h_nr[0];
        result->query_result = h_qr[0];
        result->sketch_checksum = h_sk[0];
        result->state_checksum = h_stt[0];
    }
    return true;
}

static bool run_scenario_once(
    const Scenario& sc,
    bool verbose,
    std::vector<StepResult>* results,
    int* passed_steps,
    int* total_steps,
    std::string* first_error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);
    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DmqsOracleState oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.steps.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult result;
        std::string error;
        const bool ok = run_one_step(
            sc.spec, sc.steps[i], &oracle, state,
            workspace.ptr, std::max<size_t>(workspace_bytes, 1), stream,
            results ? &result : nullptr, &error);

        ++(*total_steps);
        if (ok) ++(*passed_steps);
        else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream oss;
                oss << sc.name << " step " << i << ": " << error;
                *first_error = oss.str();
            }
        }
        if (results) results->push_back(result);

        if (verbose) {
            const char* opname =
                sc.steps[i].run.op == DMQS_OP_INGEST ? "INGEST" :
                sc.steps[i].run.op == DMQS_OP_MERGE ? "MERGE " : "QUERY ";
            std::printf("scenario %-26s step %02zu/%02zu %s batch=%-4d %s%s%s\n",
                sc.name.c_str(), i, sc.steps.size(), opname,
                sc.steps[i].run.batch_size,
                ok ? "PASS" : "FAIL", ok ? "" : "  ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(
    const std::vector<StepResult>& a,
    const std::vector<StepResult>& b,
    std::string* error) {
    if (a.size() != b.size()) { if (error) *error = "result length mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].total_weight != b[i].total_weight ||
            a[i].num_levels != b[i].num_levels ||
            a[i].num_retained_items != b[i].num_retained_items ||
            a[i].query_result != b[i].query_result ||
            a[i].sketch_checksum != b[i].sketch_checksum ||
            a[i].state_checksum != b[i].state_checksum) {
            if (error) {
                std::ostringstream oss;
                oss << "replay mismatch at step " << i << ": state a=0x" << std::hex
                    << a[i].state_checksum << " b=0x" << b[i].state_checksum;
                *error = oss.str();
            }
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
                std::string cmp;
                if (compare_results(base_results, replay_results, &cmp))
                    std::printf("scenario %-26s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-26s exact replay FAIL  %s\n", sc.name.c_str(), cmp.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-26s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
