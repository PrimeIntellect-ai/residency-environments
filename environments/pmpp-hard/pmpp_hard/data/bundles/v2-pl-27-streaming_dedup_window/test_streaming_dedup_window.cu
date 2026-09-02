// file: test_streaming_dedup_window.cu

#include "streaming_dedup_window_common.h"
#include "streaming_dedup_window_oracle.hpp"

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

static constexpr uint64_t g_state = 0x3a9f12dc77e4b681ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                               \
        if (_err != cudaSuccess) {                                               \
            std::ostringstream _oss;                                             \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "     \
                 << cudaGetErrorString(_err);                                   \
            throw std::runtime_error(_oss.str());                               \
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
        if (n == 0) {
            ptr = nullptr;
            return;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
    }

    void upload(const std::vector<T>& host) {
        if (host.size() != count) {
            throw std::runtime_error("DeviceBuffer::upload size mismatch");
        }
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
        }
    }

    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count != 0) {
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
    size_t total_bytes = 0;

    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

    ~GuardedDeviceBuffer() {
        if (raw) cudaFree(raw);
    }

    void allocate(size_t n) {
        count = n;
        data_bytes = sizeof(T) * count;
        total_bytes = kGuardBytes + data_bytes + kGuardBytes;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), total_bytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, total_bytes));

        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }

    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        }
        return host;
    }

    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> before(kGuardBytes);
        std::vector<uint8_t> after(kGuardBytes);

        CUDA_CHECK(cudaMemcpy(before.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (before[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "left guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }

            if (after[i] != kGuardByte) {
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
    SdwRunSpec run;
    std::vector<int32_t> key;
    std::vector<int32_t> value;
};

struct StepResult {
    int32_t active_count = 0;
    int32_t num_new = 0;
    int32_t num_dup = 0;
    int32_t num_evicted = 0;
    uint64_t evicted_key_checksum = 0;
    int64_t live_agg_sum = 0;
    uint64_t state_checksum = 0;
};

struct Scenario {
    std::string name;
    SdwProblemSpec spec;
    std::vector<StepHost> steps;
    bool do_permuted_replay = false;
};

static SdwProblemSpec make_spec(
    int key_space,
    int capacity,
    int window_size,
    int max_batch,
    int max_steps) {
    SdwProblemSpec spec = {};
    spec.abi_version = SDW_ABI_VERSION;
    spec.key_space = key_space;
    spec.capacity = capacity;
    spec.window_size = window_size;
    spec.max_batch = max_batch;
    spec.max_steps = max_steps;
    spec.flags = 0;

    if (!sdw_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid SdwProblemSpec generated");
    }

    return spec;
}

static StepHost make_step(
    const SdwProblemSpec& spec,
    int step_id,
    const std::vector<int32_t>& keys,
    const std::vector<int32_t>& values) {
    if (keys.size() != values.size()) {
        throw std::runtime_error("step key/value size mismatch");
    }

    StepHost step;
    step.run = {};
    step.run.abi_version = SDW_ABI_VERSION;
    step.run.batch_size = static_cast<int32_t>(keys.size());
    step.run.step_id = step_id;

    if (!sdw_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid SdwRunSpec generated");
    }

    const size_t rows = std::max<size_t>(1, keys.size());
    step.key.assign(rows, 0);
    step.value.assign(rows, 0);

    for (size_t i = 0; i < keys.size(); ++i) {
        step.key[i] = keys[i];
        step.value[i] = values[i];
    }

    return step;
}

static StepHost make_random_step(
    const SdwProblemSpec& spec,
    int step_id,
    int batch_size,
    int mode,
    SplitMix64& rng) {
    std::vector<int32_t> keys((size_t)batch_size);
    std::vector<int32_t> values((size_t)batch_size);

    for (int i = 0; i < batch_size; ++i) {
        int key = 0;

        if (mode == 0) {
            key = rng.uniform_int(0, std::max(0, spec.key_space - 1));
        } else if (mode == 1) {
            key = rng.uniform_int(0, std::min(spec.key_space - 1, 7));
        } else if (mode == 2) {
            key = (step_id + i) % std::max(1, spec.key_space);
        } else if (mode == 3) {
            key = (i % 4 == 0) ? -1 : rng.uniform_int(0, std::min(spec.key_space - 1, 15));
        } else {
            key = rng.uniform_int(0, std::min(spec.key_space - 1, 3));
        }

        int32_t value = rng.next_i32();
        if (value == 0) value = step_id * 1009 + i + 1;

        keys[(size_t)i] = key;
        values[(size_t)i] = value;
    }

    return make_step(spec, step_id, keys, values);
}

static StepHost permute_step_reverse(const StepHost& src) {
    StepHost dst = src;

    const int n = src.run.batch_size;
    const size_t rows = std::max<size_t>(1, (size_t)n);

    dst.key.assign(rows, 0);
    dst.value.assign(rows, 0);

    for (int i = 0; i < n; ++i) {
        const int j = n - 1 - i;
        dst.key[(size_t)i] = src.key[(size_t)j];
        dst.value[(size_t)i] = src.value[(size_t)j];
    }

    return dst;
}

static Scenario make_permuted_scenario(const Scenario& src) {
    Scenario dst = src;
    dst.name = src.name + "_permuted";
    dst.steps.clear();
    dst.steps.reserve(src.steps.size());

    for (const StepHost& step : src.steps) {
        dst.steps.push_back(permute_step_reverse(step));
    }

    return dst;
}

static Scenario make_capacity_min_scenario() {
    Scenario sc;
    sc.name = "capacity_min_force_evict";
    sc.spec = make_spec(
        16,   // key_space
        1,    // capacity
        100,  // window: isolate LRU capacity evict
        8,    // max_batch
        48);

    SplitMix64 rng(g_state ^ 0x10101010ULL);

    sc.steps.push_back(make_step(sc.spec, 0, {1}, {10}));
    sc.steps.push_back(make_step(sc.spec, 1, {2}, {20}));
    sc.steps.push_back(make_step(sc.spec, 2, {2, 2, 2, 2}, {3, 4, 5, 6}));
    sc.steps.push_back(make_step(sc.spec, 3, {3, 4, 5}, {30, 40, 50}));
    sc.steps.push_back(make_step(sc.spec, 4, {}, {}));

    for (int s = 5; s < 32; ++s) {
        const int mode = s % 5;
        if (mode == 0) {
            sc.steps.push_back(make_step(sc.spec, s, {s % 16}, {rng.next_i32()}));
        } else if (mode == 1) {
            sc.steps.push_back(make_step(sc.spec, s, {s % 16, s % 16, s % 16}, {
                rng.next_i32(), rng.next_i32(), rng.next_i32()
            }));
        } else if (mode == 2) {
            sc.steps.push_back(make_step(sc.spec, s, {(s + 1) % 16, (s + 2) % 16}, {
                rng.next_i32(), rng.next_i32()
            }));
        } else if (mode == 3) {
            sc.steps.push_back(make_step(sc.spec, s, {-1, -2, -3, -4}, {1, 2, 3, 4}));
        } else {
            sc.steps.push_back(make_step(sc.spec, s, {}, {}));
        }
    }

    return sc;
}

static Scenario make_window_expiry_scenario() {
    Scenario sc;
    sc.name = "window_expires_all";
    sc.spec = make_spec(
        64,
        16,
        5,   // small window
        16,
        48);

    SplitMix64 rng(g_state ^ 0x20202020ULL);

    sc.steps.push_back(make_step(sc.spec, 0, {0, 1, 2, 3}, {10, 20, 30, 40}));
    sc.steps.push_back(make_step(sc.spec, 1, {0, 1, 1, 0}, {1, 2, 3, 4}));
    sc.steps.push_back(make_step(sc.spec, 2, {-1, -1, -1, -1, -1, -1, -1, -1}, {1, 1, 1, 1, 1, 1, 1, 1}));
    sc.steps.push_back(make_step(sc.spec, 3, {4, 5, 6, 7}, {50, 60, 70, 80}));

    for (int s = 4; s < 28; ++s) {
        const int mode = s % 4;
        if (mode == 0) {
            sc.steps.push_back(make_random_step(sc.spec, s, 8, 1, rng));
        } else if (mode == 1) {
            sc.steps.push_back(make_random_step(sc.spec, s, 12, 3, rng));
        } else if (mode == 2) {
            sc.steps.push_back(make_step(sc.spec, s, {}, {}));
        } else {
            sc.steps.push_back(make_random_step(sc.spec, s, 6, 2, rng));
        }
    }

    return sc;
}

static Scenario make_high_dup_pressure_scenario() {
    Scenario sc;
    sc.name = "high_dup_capacity_pressure";
    sc.spec = make_spec(
        256,
        32,
        20,
        32,
        64);

    SplitMix64 rng(g_state ^ 0x30303030ULL);

    for (int s = 0; s < 48; ++s) {
        int batch = 0;
        int mode = 0;

        if (s == 17) {
            batch = 0;
            mode = 0;
        } else if (s % 9 == 0) {
            batch = 32;
            mode = 2;  // broad unique-ish stream for capacity pressure
        } else if (s % 5 == 0) {
            batch = 32;
            mode = 3;  // invalid events also advance time
        } else {
            batch = 24;
            mode = 1;  // high duplicate rate over a small key set
        }

        sc.steps.push_back(make_random_step(sc.spec, s, batch, mode, rng));
    }

    return sc;
}

static Scenario make_permutable_same_key_scenario() {
    Scenario sc;
    sc.name = "permutable_same_key_batches";
    sc.spec = make_spec(
        32,
        4,
        1000,
        16,
        48);
    sc.do_permuted_replay = true;

    SplitMix64 rng(g_state ^ 0x40404040ULL);

    for (int s = 0; s < 24; ++s) {
        const int batch = (s % 7) + 1;
        const int key = s % 6;

        std::vector<int32_t> keys((size_t)batch, key);
        std::vector<int32_t> values((size_t)batch);

        for (int i = 0; i < batch; ++i) {
            values[(size_t)i] = rng.next_i32();
            if (values[(size_t)i] == 0) values[(size_t)i] = s * 101 + i + 1;
        }

        if (s == 11) {
            keys.assign(8, -1);
            values.assign(8, 123);
        }

        if (s == 19) {
            keys.clear();
            values.clear();
        }

        sc.steps.push_back(make_step(sc.spec, s, keys, values));
    }

    return sc;
}

static bool check_input_unchanged(
    const StepHost& step,
    const DeviceBuffer<int32_t>& d_key,
    const DeviceBuffer<int32_t>& d_value,
    std::string* error) {
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
    const SdwProblemSpec& spec,
    const StepHost& step,
    SdwOracleState* oracle,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    StepResult* result,
    std::string* error) {
    DeviceBuffer<int32_t> d_key;
    DeviceBuffer<int32_t> d_value;

    d_key.allocate(step.key.size());
    d_value.allocate(step.value.size());

    d_key.upload(step.key);
    d_value.upload(step.value);

    GuardedDeviceBuffer<int32_t> d_active_count;
    GuardedDeviceBuffer<int32_t> d_num_new;
    GuardedDeviceBuffer<int32_t> d_num_dup;
    GuardedDeviceBuffer<int32_t> d_num_evicted;
    GuardedDeviceBuffer<uint64_t> d_evicted_key_checksum;
    GuardedDeviceBuffer<int64_t> d_live_agg_sum;
    GuardedDeviceBuffer<uint64_t> d_state_checksum;

    d_active_count.allocate(1);
    d_num_new.allocate(1);
    d_num_dup.allocate(1);
    d_num_evicted.allocate(1);
    d_evicted_key_checksum.allocate(1);
    d_live_agg_sum.allocate(1);
    d_state_checksum.allocate(1);

    SdwInputs inputs = {};
    inputs.key = d_key.ptr;
    inputs.value = d_value.ptr;

    SdwOutputs outputs = {};
    outputs.active_count = d_active_count.ptr;
    outputs.num_new = d_num_new.ptr;
    outputs.num_dup = d_num_dup.ptr;
    outputs.num_evicted = d_num_evicted.ptr;
    outputs.evicted_key_checksum = d_evicted_key_checksum.ptr;
    outputs.live_agg_sum = d_live_agg_sum.ptr;
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

    if (!check_input_unchanged(step, d_key, d_value, error)) {
        return false;
    }

    if (!d_active_count.check_guards("active_count", error)) return false;
    if (!d_num_new.check_guards("num_new", error)) return false;
    if (!d_num_dup.check_guards("num_dup", error)) return false;
    if (!d_num_evicted.check_guards("num_evicted", error)) return false;
    if (!d_evicted_key_checksum.check_guards("evicted_key_checksum", error)) return false;
    if (!d_live_agg_sum.check_guards("live_agg_sum", error)) return false;
    if (!d_state_checksum.check_guards("state_checksum", error)) return false;

    const std::vector<int32_t> h_active_count = d_active_count.download_data();
    const std::vector<int32_t> h_num_new = d_num_new.download_data();
    const std::vector<int32_t> h_num_dup = d_num_dup.download_data();
    const std::vector<int32_t> h_num_evicted = d_num_evicted.download_data();
    const std::vector<uint64_t> h_evicted_hash = d_evicted_key_checksum.download_data();
    const std::vector<int64_t> h_live_sum = d_live_agg_sum.download_data();
    const std::vector<uint64_t> h_state_checksum = d_state_checksum.download_data();

    SdwHostInputsView host_inputs = {};
    host_inputs.key = step.key.data();
    host_inputs.value = step.value.data();

    SdwExpected expected;
    oracle->step_once(step.run, host_inputs, &expected);

    SdwHostOutputsView got = {};
    got.active_count = h_active_count.data();
    got.num_new = h_num_new.data();
    got.num_dup = h_num_dup.data();
    got.num_evicted = h_num_evicted.data();
    got.evicted_key_checksum = h_evicted_hash.data();
    got.live_agg_sum = h_live_sum.data();
    got.state_checksum = h_state_checksum.data();

    if (!sdw_check_all_outputs(expected, got, error)) {
        return false;
    }

    if (result) {
        result->active_count = h_active_count[0];
        result->num_new = h_num_new[0];
        result->num_dup = h_num_dup[0];
        result->num_evicted = h_num_evicted[0];
        result->evicted_key_checksum = h_evicted_hash[0];
        result->live_agg_sum = h_live_sum[0];
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

    SdwOracleState oracle;
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

        if (results) {
            results->push_back(result);
        }

        if (verbose) {
            std::printf(
                "scenario %-32s step %02zu/%02zu batch=%d %s%s%s\n",
                sc.name.c_str(),
                i,
                sc.steps.size(),
                sc.steps[i].run.batch_size,
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
        if (a[i].active_count != b[i].active_count ||
            a[i].num_new != b[i].num_new ||
            a[i].num_dup != b[i].num_dup ||
            a[i].num_evicted != b[i].num_evicted ||
            a[i].evicted_key_checksum != b[i].evicted_key_checksum ||
            a[i].live_agg_sum != b[i].live_agg_sum ||
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

        std::vector<Scenario> scenarios;
        scenarios.push_back(make_capacity_min_scenario());
        scenarios.push_back(make_window_expiry_scenario());
        scenarios.push_back(make_high_dup_pressure_scenario());
        scenarios.push_back(make_permutable_same_key_scenario());

        int passed = 0;
        int total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results;
            std::vector<StepResult> replay_results;

            std::string error;

            const bool ok_base = run_scenario_once(
                sc,
                true,
                &base_results,
                &passed,
                &total,
                &error);

            const bool ok_replay = run_scenario_once(
                sc,
                false,
                &replay_results,
                &passed,
                &total,
                &error);

            if (ok_base && ok_replay) {
                std::string compare_error;
                if (compare_results(base_results, replay_results, &compare_error)) {
                    std::printf("scenario %-32s exact replay PASS\n", sc.name.c_str());
                } else {
                    all_ok = false;
                    std::printf("scenario %-32s exact replay FAIL  %s\n", sc.name.c_str(), compare_error.c_str());
                }
            } else {
                all_ok = false;
                std::printf("scenario %-32s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }

            if (sc.do_permuted_replay) {
                Scenario permuted = make_permuted_scenario(sc);
                std::vector<StepResult> perm_results;
                std::string perm_error;

                const bool ok_perm = run_scenario_once(
                    permuted,
                    false,
                    &perm_results,
                    &passed,
                    &total,
                    &perm_error);

                if (ok_perm) {
                    std::string compare_error;
                    if (compare_results(base_results, perm_results, &compare_error)) {
                        std::printf("scenario %-32s permuted replay PASS\n", sc.name.c_str());
                    } else {
                        all_ok = false;
                        std::printf("scenario %-32s permuted replay FAIL  %s\n", sc.name.c_str(), compare_error.c_str());
                    }
                } else {
                    all_ok = false;
                    std::printf("scenario %-32s permuted replay FAIL  %s\n", sc.name.c_str(), perm_error.c_str());
                }
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
