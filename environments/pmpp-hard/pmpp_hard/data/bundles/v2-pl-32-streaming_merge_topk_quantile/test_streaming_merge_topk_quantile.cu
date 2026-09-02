// file: test_streaming_merge_topk_quantile.cu

#include "streaming_merge_topk_quantile_common.h"
#include "streaming_merge_topk_quantile_oracle.hpp"

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

static constexpr uint64_t g_state = 0x7c91b3a42d8f0e65ULL;
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

    int32_t next_i32() {
        return static_cast<int32_t>(next_u64() >> 32);
    }

    int uniform_int(int lo, int hi) {
        const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u64() % span);
    }

    bool chance_permille(int p) {
        if (p <= 0) return false;
        if (p >= 1000) return true;
        return static_cast<int>(next_u64() % 1000ULL) < p;
    }
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }

    void allocate(size_t n) {
        count = n;
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
        }
    }

    void upload(const std::vector<T>& host) {
        if (host.size() != count) {
            throw std::runtime_error("DeviceBuffer upload size mismatch");
        }
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
        }
    }

    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost));
        }
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

    ~GuardedDeviceBuffer() {
        if (raw) cudaFree(raw);
    }

    void allocate(size_t n) {
        count = n;
        data_bytes = sizeof(T) * count;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), kGuardBytes + data_bytes + kGuardBytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, kGuardBytes + data_bytes + kGuardBytes));
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }

    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        }
        return host;
    }

    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> left(kGuardBytes);
        std::vector<uint8_t> right(kGuardBytes);

        CUDA_CHECK(cudaMemcpy(left.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(right.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (left[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "left guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }

            if (right[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "right guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }
        }

        return true;
    }
};

struct StepHost {
    SmtqRunSpec run;
    std::vector<int32_t> group;
    std::vector<int32_t> key;
    std::vector<int32_t> value;
};

struct StepResult {
    std::vector<int32_t> topk_keys;
    std::vector<int32_t> topk_values;
    std::vector<int32_t> topk_count;
    std::vector<int64_t> topk_value_sum;
    uint64_t histogram_checksum = 0;
    std::vector<int32_t> quantile_key;
    int64_t total_ingested = 0;
    uint64_t state_checksum = 0;
};

struct Scenario {
    std::string name;
    SmtqProblemSpec spec;
    std::vector<StepHost> steps;
};

static SmtqProblemSpec make_spec(
    int G,
    int K,
    int num_bins,
    int key_min,
    int key_max,
    int max_batch,
    int max_steps) {
    SmtqProblemSpec spec = {};
    spec.abi_version = SMTQ_ABI_VERSION;
    spec.G = G;
    spec.K = K;
    spec.num_bins = num_bins;
    spec.key_min = key_min;
    spec.key_max = key_max;
    spec.max_batch = max_batch;
    spec.max_steps = max_steps;
    spec.flags = 0;

    if (!smtq_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid SmtqProblemSpec generated");
    }

    return spec;
}

static StepHost make_step(
    const SmtqProblemSpec& spec,
    int step_id,
    const std::vector<int32_t>& group,
    const std::vector<int32_t>& key,
    const std::vector<int32_t>& value,
    bool is_query,
    int q_num,
    int q_den) {
    if (group.size() != key.size() || group.size() != value.size()) {
        throw std::runtime_error("step vector size mismatch");
    }

    StepHost step;
    step.run = {};
    step.run.abi_version = SMTQ_ABI_VERSION;
    step.run.batch_size = static_cast<int32_t>(group.size());
    step.run.is_query = is_query ? 1 : 0;
    step.run.q_num = q_num;
    step.run.q_den = q_den;
    step.run.step_id = step_id;

    if (!smtq_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid SmtqRunSpec generated");
    }

    const size_t rows = std::max<size_t>(1, group.size());
    step.group.assign(rows, -1);
    step.key.assign(rows, 0);
    step.value.assign(rows, 0);

    for (size_t i = 0; i < group.size(); ++i) {
        step.group[i] = group[i];
        step.key[i] = key[i];
        step.value[i] = value[i];
    }

    return step;
}

static int choose_group(SplitMix64& rng, const SmtqProblemSpec& spec, int row, int dist) {
    if (dist == 1) {
        const int r = rng.uniform_int(0, 999);
        if (r < 570) return 0;
        if (r < 760) return std::min(1, spec.G - 1);
        if (r < 860) return std::min(2, spec.G - 1);
        if (r < 930) return rng.uniform_int(0, std::min(spec.G - 1, 15));
        return rng.uniform_int(0, spec.G - 1);
    }

    if (dist == 2) {
        return row % std::min(spec.G, 8);
    }

    if (dist == 3) {
        return 0;
    }

    return rng.uniform_int(0, spec.G - 1);
}

static int32_t choose_key(SplitMix64& rng, const SmtqProblemSpec& spec, int row, int group, int dist) {
    if (dist == 2) {
        const int width = std::max(1, (spec.key_max - spec.key_min + 1) / std::max(1, spec.num_bins));
        const int bin = (row / 5 + group * 7) % spec.num_bins;
        return spec.key_min + bin * width;
    }

    if (dist == 3) {
        return spec.key_min + (spec.key_max - spec.key_min) / 2;
    }

    if (dist == 1 && group == 0) {
        if ((row % 7) == 0) return spec.key_max - (row % 31);
        if ((row % 11) == 0) return spec.key_min + (row % 127);
    }

    if ((row % 97) == 0) return spec.key_min;
    if ((row % 131) == 0) return spec.key_max;

    return rng.uniform_int(spec.key_min, spec.key_max);
}

static int32_t choose_value(SplitMix64& rng, int row) {
    int32_t v = rng.next_i32();
    if (v == 0) v = row + 1;
    if ((row % 257) == 0) v = INT_MAX - (row & 1023);
    if ((row % 389) == 0) v = INT_MIN + (row & 1023);
    return v;
}

static StepHost make_random_step(
    const SmtqProblemSpec& spec,
    int step_id,
    int batch_size,
    int dist,
    bool is_query,
    int q_num,
    int q_den,
    SplitMix64& rng,
    int invalid_permille) {
    std::vector<int32_t> group;
    std::vector<int32_t> key;
    std::vector<int32_t> value;

    group.reserve(static_cast<size_t>(batch_size));
    key.reserve(static_cast<size_t>(batch_size));
    value.reserve(static_cast<size_t>(batch_size));

    for (int i = 0; i < batch_size; ++i) {
        if (rng.chance_permille(invalid_permille)) {
            group.push_back((i & 1) ? -1 : spec.G + 7);
            key.push_back(choose_key(rng, spec, i + step_id * 4099, 0, dist));
            value.push_back(choose_value(rng, i + step_id * 8191));
            continue;
        }

        const int g = choose_group(rng, spec, i + step_id * 17, dist);
        group.push_back(g);
        key.push_back(choose_key(rng, spec, i + step_id * 4099, g, dist));
        value.push_back(choose_value(rng, i + step_id * 8191));
    }

    return make_step(spec, step_id, group, key, value, is_query, q_num, q_den);
}

static StepHost permute_ignored_invalids_first(const StepHost& src, const SmtqProblemSpec& spec) {
    std::vector<int32_t> group;
    std::vector<int32_t> key;
    std::vector<int32_t> value;

    const int n = src.run.batch_size;

    // Invalid rows are ignored and do not increment insertion order, so moving
    // them around while preserving valid-event order must not affect state.
    for (int i = n - 1; i >= 0; --i) {
        const int g = src.group[(size_t)i];
        if (g < 0 || g >= spec.G) {
            group.push_back(g);
            key.push_back(src.key[(size_t)i]);
            value.push_back(src.value[(size_t)i]);
        }
    }

    for (int i = 0; i < n; ++i) {
        const int g = src.group[(size_t)i];
        if (g >= 0 && g < spec.G) {
            group.push_back(g);
            key.push_back(src.key[(size_t)i]);
            value.push_back(src.value[(size_t)i]);
        }
    }

    return make_step(
        spec,
        src.run.step_id,
        group,
        key,
        value,
        src.run.is_query != 0,
        src.run.q_num,
        src.run.q_den);
}

static Scenario permuted_replay_scenario(const Scenario& src) {
    Scenario dst;
    dst.name = src.name + "_invalid_permuted";
    dst.spec = src.spec;
    dst.steps.reserve(src.steps.size());

    for (const StepHost& step : src.steps) {
        dst.steps.push_back(permute_ignored_invalids_first(step, src.spec));
    }

    return dst;
}

static Scenario make_uniform_boundary_scenario() {
    Scenario sc;
    sc.name = "uniform_boundary_queries";
    sc.spec = make_spec(8, 4, 16, -128, 127, 32, 48);

    SplitMix64 rng(g_state ^ 0x10101010ULL);

    sc.steps.push_back(make_step(sc.spec, 0, {}, {}, {}, true, 0, 1));
    sc.steps.push_back(make_random_step(sc.spec, 1, 16, 0, false, 1, 2, rng, 250));
    sc.steps.push_back(make_random_step(sc.spec, 2, 32, 0, true, 1, 2, rng, 200));
    sc.steps.push_back(make_random_step(sc.spec, 3, 24, 0, true, 1, 1, rng, 200));
    sc.steps.push_back(make_random_step(sc.spec, 4, 0, 0, true, 0, 1, rng, 0));

    for (int s = 5; s < 24; ++s) {
        const bool is_query = (s % 3) == 0;
        const int q_num = (s % 5 == 0) ? 1 : ((s % 5 == 1) ? 3 : 1);
        const int q_den = (s % 5 == 0) ? 4 : ((s % 5 == 1) ? 4 : 2);
        const int batch = (s % 4 == 0) ? 0 : (8 + (s * 7) % 25);
        sc.steps.push_back(make_random_step(sc.spec, s, batch, 0, is_query, q_num, q_den, rng, 180));
    }

    return sc;
}

static Scenario make_zipf_hot_scenario() {
    Scenario sc;
    sc.name = "zipf_hot_large";
    sc.spec = make_spec(64, 16, 128, -10000, 10000, 128, 48);

    SplitMix64 rng(g_state ^ 0x20202020ULL);

    for (int s = 0; s < 36; ++s) {
        const bool is_query = (s % 4) == 1 || (s % 9) == 0;
        const int batch = (s == 11) ? 0 : (32 + (s * 13) % 97);
        const int q_num = (s % 7 == 0) ? 9 : ((s % 7 == 1) ? 1 : 3);
        const int q_den = (s % 7 == 0) ? 10 : ((s % 7 == 1) ? 10 : 4);
        sc.steps.push_back(make_random_step(sc.spec, s, batch, 1, is_query, q_num, q_den, rng, 120));
    }

    return sc;
}

static Scenario make_many_ties_large_g_scenario() {
    Scenario sc;
    sc.name = "many_ties_largeG";
    sc.spec = make_spec(1024, 64, 256, 0, 4095, 256, 32);

    SplitMix64 rng(g_state ^ 0x30303030ULL);

    for (int s = 0; s < 20; ++s) {
        const bool is_query = (s % 2) == 0;
        const int batch = (s == 7) ? 0 : (64 + (s * 29) % 193);
        const int q_num = (s % 4 == 0) ? 1 : ((s % 4 == 1) ? 2 : 3);
        const int q_den = 4;
        sc.steps.push_back(make_random_step(sc.spec, s, batch, 2, is_query, q_num, q_den, rng, 90));
    }

    return sc;
}

static Scenario make_same_key_k_gt_distinct_scenario() {
    Scenario sc;
    sc.name = "same_key_k_gt_distinct";
    sc.spec = make_spec(8, 64, 32, 0, 31, 64, 32);

    SplitMix64 rng(g_state ^ 0x40404040ULL);

    for (int s = 0; s < 18; ++s) {
        const bool is_query = true;
        const int batch = (s == 5 || s == 13) ? 0 : (4 + (s % 9));
        const int q_num = (s % 3 == 0) ? 0 : ((s % 3 == 1) ? 1 : 1);
        const int q_den = (s % 3 == 0) ? 1 : ((s % 3 == 1) ? 1 : 2);
        sc.steps.push_back(make_random_step(sc.spec, s, batch, 3, is_query, q_num, q_den, rng, 160));
    }

    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> scenarios;
    scenarios.push_back(make_uniform_boundary_scenario());
    scenarios.push_back(make_zipf_hot_scenario());
    scenarios.push_back(make_many_ties_large_g_scenario());
    scenarios.push_back(make_same_key_k_gt_distinct_scenario());
    return scenarios;
}

static bool check_input_unchanged(
    const StepHost& step,
    const DeviceBuffer<int32_t>& d_group,
    const DeviceBuffer<int32_t>& d_key,
    const DeviceBuffer<int32_t>& d_value,
    std::string* error) {
    if (d_group.download() != step.group) {
        if (error) *error = "input group modified";
        return false;
    }

    if (d_key.download() != step.key) {
        if (error) *error = "input key modified";
        return false;
    }

    if (d_value.download() != step.value) {
        if (error) *error = "input value modified";
        return false;
    }

    return true;
}

static bool run_one_step(
    const SmtqProblemSpec& spec,
    const StepHost& step,
    SmtqOracleState* oracle,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    StepResult* result,
    std::string* error) {
    DeviceBuffer<int32_t> d_group;
    DeviceBuffer<int32_t> d_key;
    DeviceBuffer<int32_t> d_value;

    d_group.allocate(step.group.size());
    d_key.allocate(step.key.size());
    d_value.allocate(step.value.size());

    d_group.upload(step.group);
    d_key.upload(step.key);
    d_value.upload(step.value);

    GuardedDeviceBuffer<int32_t> d_topk_keys;
    GuardedDeviceBuffer<int32_t> d_topk_values;
    GuardedDeviceBuffer<int32_t> d_topk_count;
    GuardedDeviceBuffer<int64_t> d_topk_value_sum;
    GuardedDeviceBuffer<uint64_t> d_histogram_checksum;
    GuardedDeviceBuffer<int32_t> d_quantile_key;
    GuardedDeviceBuffer<int64_t> d_total_ingested;
    GuardedDeviceBuffer<uint64_t> d_state_checksum;

    d_topk_keys.allocate((size_t)spec.G * (size_t)spec.K);
    d_topk_values.allocate((size_t)spec.G * (size_t)spec.K);
    d_topk_count.allocate((size_t)spec.G);
    d_topk_value_sum.allocate((size_t)spec.G);
    d_histogram_checksum.allocate(1);
    d_quantile_key.allocate((size_t)spec.G);
    d_total_ingested.allocate(1);
    d_state_checksum.allocate(1);

    SmtqInputs inputs = {};
    inputs.group = d_group.ptr;
    inputs.key = d_key.ptr;
    inputs.value = d_value.ptr;

    SmtqOutputs outputs = {};
    outputs.topk_keys = d_topk_keys.ptr;
    outputs.topk_values = d_topk_values.ptr;
    outputs.topk_count = d_topk_count.ptr;
    outputs.topk_value_sum = d_topk_value_sum.ptr;
    outputs.histogram_checksum = d_histogram_checksum.ptr;
    outputs.quantile_key = d_quantile_key.ptr;
    outputs.total_ingested = d_total_ingested.ptr;
    outputs.state_checksum = d_state_checksum.ptr;

    CUDA_CHECK(solution_run(
        state,
        &step.run,
        &inputs,
        &outputs,
        workspace,
        workspace_bytes,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(step, d_group, d_key, d_value, error)) return false;

    if (!d_topk_keys.check_guards("topk_keys", error)) return false;
    if (!d_topk_values.check_guards("topk_values", error)) return false;
    if (!d_topk_count.check_guards("topk_count", error)) return false;
    if (!d_topk_value_sum.check_guards("topk_value_sum", error)) return false;
    if (!d_histogram_checksum.check_guards("histogram_checksum", error)) return false;
    if (!d_quantile_key.check_guards("quantile_key", error)) return false;
    if (!d_total_ingested.check_guards("total_ingested", error)) return false;
    if (!d_state_checksum.check_guards("state_checksum", error)) return false;

    const std::vector<int32_t> h_topk_keys = d_topk_keys.download_data();
    const std::vector<int32_t> h_topk_values = d_topk_values.download_data();
    const std::vector<int32_t> h_topk_count = d_topk_count.download_data();
    const std::vector<int64_t> h_topk_value_sum = d_topk_value_sum.download_data();
    const std::vector<uint64_t> h_histogram_checksum = d_histogram_checksum.download_data();
    const std::vector<int32_t> h_quantile_key = d_quantile_key.download_data();
    const std::vector<int64_t> h_total_ingested = d_total_ingested.download_data();
    const std::vector<uint64_t> h_state_checksum = d_state_checksum.download_data();

    SmtqHostInputsView host_inputs = {};
    host_inputs.group = step.group.data();
    host_inputs.key = step.key.data();
    host_inputs.value = step.value.data();

    SmtqExpected expected;
    oracle->step_once(step.run, host_inputs, &expected);

    SmtqHostOutputsView got = {};
    got.topk_keys = h_topk_keys.data();
    got.topk_values = h_topk_values.data();
    got.topk_count = h_topk_count.data();
    got.topk_value_sum = h_topk_value_sum.data();
    got.histogram_checksum = h_histogram_checksum.data();
    got.quantile_key = h_quantile_key.data();
    got.total_ingested = h_total_ingested.data();
    got.state_checksum = h_state_checksum.data();

    if (!smtq_check_all_outputs(spec, expected, got, error)) return false;

    if (result) {
        result->topk_keys = h_topk_keys;
        result->topk_values = h_topk_values;
        result->topk_count = h_topk_count;
        result->topk_value_sum = h_topk_value_sum;
        result->histogram_checksum = h_histogram_checksum[0];
        result->quantile_key = h_quantile_key;
        result->total_ingested = h_total_ingested[0];
        result->state_checksum = h_state_checksum[0];
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
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec); if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/
    if (workspace_bytes == 0) {
        if (first_error) *first_error = "solution_workspace_bytes returned 0";
        return false;
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(workspace_bytes);

    SmtqOracleState oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) {
        results->clear();
        results->reserve(sc.steps.size());
    }

    bool all_ok = true;

    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult result;
        std::string error;

        const bool ok = run_one_step(
            sc.spec,
            sc.steps[i],
            &oracle,
            state,
            workspace.ptr,
            workspace_bytes,
            stream,
            results ? &result : nullptr,
            &error);

        ++(*total_steps);
        if (ok) {
            ++(*passed_steps);
        } else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream oss;
                oss << sc.name << " step " << i << ": " << error;
                *first_error = oss.str();
            }
        }

        if (results) results->push_back(result);

        if (verbose) {
            std::printf(
                "scenario %-30s step %02zu/%02zu batch=%d query=%d %s%s%s\n",
                sc.name.c_str(),
                i,
                sc.steps.size(),
                sc.steps[i].run.batch_size,
                sc.steps[i].run.is_query,
                ok ? "PASS" : "FAIL",
                ok ? "" : "  ",
                ok ? "" : error.c_str());
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
    if (a.size() != b.size()) {
        if (error) *error = "result length mismatch";
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].topk_keys != b[i].topk_keys ||
            a[i].topk_values != b[i].topk_values ||
            a[i].topk_count != b[i].topk_count ||
            a[i].topk_value_sum != b[i].topk_value_sum ||
            a[i].histogram_checksum != b[i].histogram_checksum ||
            a[i].quantile_key != b[i].quantile_key ||
            a[i].total_ingested != b[i].total_ingested ||
            a[i].state_checksum != b[i].state_checksum) {
            if (error) {
                std::ostringstream oss;
                oss << "replay mismatch at step " << i
                    << ": checksum a=0x" << std::hex << a[i].state_checksum
                    << ", b=0x" << b[i].state_checksum;
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

        int passed = 0;
        int total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results;
            std::vector<StepResult> replay_results;
            std::vector<StepResult> perm_results;

            std::string error;

            const bool ok_base = run_scenario_once(sc, true, &base_results, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay_results, &passed, &total, &error);

            if (ok_base && ok_replay) {
                std::string compare_error;
                if (compare_results(base_results, replay_results, &compare_error)) {
                    std::printf("scenario %-30s exact replay PASS\n", sc.name.c_str());
                } else {
                    all_ok = false;
                    std::printf("scenario %-30s exact replay FAIL  %s\n", sc.name.c_str(), compare_error.c_str());
                }
            } else {
                all_ok = false;
                std::printf("scenario %-30s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }

            Scenario permuted = permuted_replay_scenario(sc);
            std::string perm_error;
            const bool ok_perm = run_scenario_once(permuted, false, &perm_results, &passed, &total, &perm_error);

            if (ok_perm) {
                std::string compare_error;
                if (compare_results(base_results, perm_results, &compare_error)) {
                    std::printf("scenario %-30s invalid-permuted replay PASS\n", sc.name.c_str());
                } else {
                    all_ok = false;
                    std::printf("scenario %-30s invalid-permuted replay FAIL  %s\n", sc.name.c_str(), compare_error.c_str());
                }
            } else {
                all_ok = false;
                std::printf("scenario %-30s invalid-permuted replay FAIL  %s\n", sc.name.c_str(), perm_error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
