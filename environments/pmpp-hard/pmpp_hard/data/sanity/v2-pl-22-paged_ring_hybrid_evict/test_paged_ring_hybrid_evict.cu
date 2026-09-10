// file: test_paged_ring_hybrid_evict.cu

#include "paged_ring_hybrid_evict_common.h"
#include "paged_ring_hybrid_evict_oracle.hpp"

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

static constexpr uint64_t g_state = 0xb70c9b2e44a64d13ULL;
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
        uint64_t z = (state += 0x8e3779b97f4a7c15ULL);
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
    PrheRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> append_count;
    std::vector<int32_t> token_values;
};

struct StepResult {
    std::vector<int32_t> live_count;
    std::vector<int64_t> live_sum;
    std::vector<uint64_t> live_hash;
    uint64_t page_table_checksum = 0;
    int32_t evicted_count = 0;
    int32_t free_pages = 0;
};

struct Scenario {
    std::string name;
    PrheProblemSpec spec;
    std::vector<StepHost> steps;
    bool do_permuted_replay = false;
};

static PrheProblemSpec make_spec(
    int B,
    int max_len,
    int page_size,
    int window_size,
    int max_pages,
    int max_active,
    int max_new_tokens,
    int max_steps) {
    PrheProblemSpec spec = {};
    spec.abi_version = PRHE_ABI_VERSION;
    spec.B = B;
    spec.max_len = max_len;
    spec.page_size = page_size;
    spec.window_size = window_size;
    spec.max_pages = max_pages;
    spec.max_active = max_active;
    spec.max_new_tokens = max_new_tokens;
    spec.max_steps = max_steps;
    spec.flags = 0;

    if (!prhe_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid PrheProblemSpec generated");
    }

    return spec;
}

static StepHost make_step(
    const PrheProblemSpec& spec,
    int step_id,
    const std::vector<int32_t>& seqs,
    const std::vector<int32_t>& append_counts,
    SplitMix64& rng) {
    if (seqs.size() != append_counts.size()) {
        throw std::runtime_error("step vector size mismatch");
    }

    const int active_count = static_cast<int>(seqs.size());

    StepHost step;
    step.run = {};
    step.run.abi_version = PRHE_ABI_VERSION;
    step.run.active_count = active_count;
    step.run.step_id = step_id;

    if (!prhe_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid PrheRunSpec generated");
    }

    const size_t rows = std::max<size_t>(1, (size_t)active_count);

    step.active_seq.assign(rows, 0);
    step.append_count.assign(rows, 0);
    step.token_values.assign(rows * (size_t)spec.max_new_tokens, 0);

    for (int r = 0; r < active_count; ++r) {
        step.active_seq[(size_t)r] = seqs[(size_t)r];
        step.append_count[(size_t)r] = append_counts[(size_t)r];

        for (int i = 0; i < spec.max_new_tokens; ++i) {
            int32_t v = rng.next_i32();
            if (v == 0) v = step_id * 10007 + r * 257 + i + 1;
            step.token_values[(size_t)r * (size_t)spec.max_new_tokens + (size_t)i] = v;
        }
    }

    return step;
}

static StepHost permute_step_reverse(const StepHost& src, const PrheProblemSpec& spec) {
    StepHost dst = src;

    const int A = src.run.active_count;
    const size_t rows = std::max<size_t>(1, (size_t)A);

    dst.active_seq.assign(rows, 0);
    dst.append_count.assign(rows, 0);
    dst.token_values.assign(rows * (size_t)spec.max_new_tokens, 0);

    for (int r = 0; r < A; ++r) {
        const int src_r = A - 1 - r;

        dst.active_seq[(size_t)r] = src.active_seq[(size_t)src_r];
        dst.append_count[(size_t)r] = src.append_count[(size_t)src_r];

        for (int i = 0; i < spec.max_new_tokens; ++i) {
            dst.token_values[(size_t)r * (size_t)spec.max_new_tokens + (size_t)i] =
                src.token_values[(size_t)src_r * (size_t)spec.max_new_tokens + (size_t)i];
        }
    }

    return dst;
}

static Scenario permute_scenario(const Scenario& src) {
    Scenario dst = src;
    dst.name = src.name + "_permuted";
    dst.steps.clear();
    dst.steps.reserve(src.steps.size());

    for (const StepHost& step : src.steps) {
        dst.steps.push_back(permute_step_reverse(step, src.spec));
    }

    return dst;
}

static Scenario make_budget_pressure_scenario() {
    Scenario sc;
    sc.name = "budget_pressure_evict_pages";
    sc.spec = make_spec(
        3,      // B
        96,     // max_len
        4,      // page_size
        6,      // window crosses page boundaries
        8,      // max_pages, less than total historical pages
        3,      // max_active
        4,      // max_new_tokens
        48);    // max_steps

    SplitMix64 rng(g_state ^ 0x11111111ULL);

    for (int s = 0; s < 36; ++s) {
        std::vector<int32_t> seqs;
        std::vector<int32_t> counts;

        if (s == 5 || s == 17) {
            seqs = {0, 1, 2};
            counts = {0, 0, 0};
        } else {
            seqs = {0, 1, 2};
            counts = {
                4,
                (s % 3 == 0) ? 2 : 4,
                (s % 4 == 0) ? 1 : 4
            };
        }

        sc.steps.push_back(make_step(sc.spec, s, seqs, counts, rng));
    }

    return sc;
}

static Scenario make_min_budget_scenario() {
    Scenario sc;
    sc.name = "min_budget_single_page_evict_all";
    sc.spec = make_spec(
        1,      // B
        80,
        4,
        4,      // exactly one live page
        1,      // minimum page budget
        1,
        4,
        32);

    SplitMix64 rng(g_state ^ 0x22222222ULL);

    for (int s = 0; s < 24; ++s) {
        int count = (s % 6 == 0) ? 0 : 4;
        sc.steps.push_back(make_step(sc.spec, s, {0}, {count}, rng));
    }

    return sc;
}

static Scenario make_window_boundary_scenario() {
    Scenario sc;
    sc.name = "window_page_boundary_and_invalid";
    sc.spec = make_spec(
        4,
        128,
        8,
        10,     // crosses page boundary
        10,
        4,
        4,
        48);

    SplitMix64 rng(g_state ^ 0x33333333ULL);

    for (int s = 0; s < 40; ++s) {
        if (s == 0) {
            sc.steps.push_back(make_step(sc.spec, s, {0, 1, 2, 3}, {1, 2, 3, 4}, rng));
        } else if (s == 11) {
            sc.steps.push_back(make_step(sc.spec, s, {0, 1, 2, 3}, {0, 0, 0, 0}, rng));
        } else if (s == 19) {
            sc.steps.push_back(make_step(sc.spec, s, {-1, 1, 99, 3}, {4, 4, 4, 4}, rng));
        } else {
            sc.steps.push_back(make_step(
                sc.spec,
                s,
                {0, 1, 2, 3},
                {
                    (s % 2 == 0) ? 3 : 1,
                    (s % 3 == 0) ? 4 : 2,
                    (s % 4 == 0) ? 0 : 4,
                    1
                },
                rng));
        }
    }

    return sc;
}

static Scenario make_permutable_no_new_pages_scenario() {
    Scenario sc;
    sc.name = "permutable_no_allocation_steps";
    sc.spec = make_spec(
        4,
        128,
        16,
        12,
        16,
        4,
        1,
        32);
    sc.do_permuted_replay = true;

    SplitMix64 rng(g_state ^ 0x44444444ULL);

    // First allocate one page per sequence with single-active steps, so
    // reversing active rows later does not affect physical page assignment.
    sc.steps.push_back(make_step(sc.spec, 0, {0}, {1}, rng));
    sc.steps.push_back(make_step(sc.spec, 1, {1}, {1}, rng));
    sc.steps.push_back(make_step(sc.spec, 2, {2}, {1}, rng));
    sc.steps.push_back(make_step(sc.spec, 3, {3}, {1}, rng));

    for (int s = 4; s < 16; ++s) {
        // Total length per seq remains < page_size, so no new page allocation.
        sc.steps.push_back(make_step(sc.spec, s, {0, 1, 2, 3}, {1, 1, 1, 1}, rng));
    }

    return sc;
}

static bool check_input_unchanged(
    const StepHost& step,
    const DeviceBuffer<int32_t>& d_active_seq,
    const DeviceBuffer<int32_t>& d_append_count,
    const DeviceBuffer<int32_t>& d_token_values,
    std::string* error) {
    if (d_active_seq.download() != step.active_seq) {
        if (error) *error = "input active_seq modified";
        return false;
    }

    if (d_append_count.download() != step.append_count) {
        if (error) *error = "input append_count modified";
        return false;
    }

    if (d_token_values.download() != step.token_values) {
        if (error) *error = "input token_values modified";
        return false;
    }

    return true;
}

static bool run_one_step(
    const PrheProblemSpec& spec,
    const StepHost& step,
    PrheOracleState* oracle,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    StepResult* result,
    std::string* error) {
    DeviceBuffer<int32_t> d_active_seq;
    DeviceBuffer<int32_t> d_append_count;
    DeviceBuffer<int32_t> d_token_values;

    d_active_seq.allocate(step.active_seq.size());
    d_append_count.allocate(step.append_count.size());
    d_token_values.allocate(step.token_values.size());

    d_active_seq.upload(step.active_seq);
    d_append_count.upload(step.append_count);
    d_token_values.upload(step.token_values);

    GuardedDeviceBuffer<int32_t> d_live_count;
    GuardedDeviceBuffer<int64_t> d_live_sum;
    GuardedDeviceBuffer<uint64_t> d_live_hash;
    GuardedDeviceBuffer<uint64_t> d_page_table_checksum;
    GuardedDeviceBuffer<int32_t> d_evicted_count;
    GuardedDeviceBuffer<int32_t> d_free_pages;

    d_live_count.allocate((size_t)spec.B);
    d_live_sum.allocate((size_t)spec.B);
    d_live_hash.allocate((size_t)spec.B);
    d_page_table_checksum.allocate(1);
    d_evicted_count.allocate(1);
    d_free_pages.allocate(1);

    PrheInputs inputs = {};
    inputs.active_seq = d_active_seq.ptr;
    inputs.append_count = d_append_count.ptr;
    inputs.token_values = d_token_values.ptr;

    PrheOutputs outputs = {};
    outputs.live_count = d_live_count.ptr;
    outputs.live_sum = d_live_sum.ptr;
    outputs.live_hash = d_live_hash.ptr;
    outputs.page_table_checksum = d_page_table_checksum.ptr;
    outputs.evicted_count = d_evicted_count.ptr;
    outputs.free_pages = d_free_pages.ptr;

    CUDA_CHECK(solution_run(
        state,
        &step.run,
        &inputs,
        &outputs,
        workspace,
        workspace_bytes,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(step, d_active_seq, d_append_count, d_token_values, error)) {
        return false;
    }

    if (!d_live_count.check_guards("live_count", error)) return false;
    if (!d_live_sum.check_guards("live_sum", error)) return false;
    if (!d_live_hash.check_guards("live_hash", error)) return false;
    if (!d_page_table_checksum.check_guards("page_table_checksum", error)) return false;
    if (!d_evicted_count.check_guards("evicted_count", error)) return false;
    if (!d_free_pages.check_guards("free_pages", error)) return false;

    const std::vector<int32_t> h_live_count = d_live_count.download_data();
    const std::vector<int64_t> h_live_sum = d_live_sum.download_data();
    const std::vector<uint64_t> h_live_hash = d_live_hash.download_data();
    const std::vector<uint64_t> h_checksum = d_page_table_checksum.download_data();
    const std::vector<int32_t> h_evicted = d_evicted_count.download_data();
    const std::vector<int32_t> h_free = d_free_pages.download_data();

    PrheHostInputsView host_inputs = {};
    host_inputs.active_seq = step.active_seq.data();
    host_inputs.append_count = step.append_count.data();
    host_inputs.token_values = step.token_values.data();

    PrheExpected expected;
    oracle->step_once(step.run, host_inputs, &expected);

    PrheHostOutputsView got = {};
    got.live_count = h_live_count.data();
    got.live_sum = h_live_sum.data();
    got.live_hash = h_live_hash.data();
    got.page_table_checksum = h_checksum.data();
    got.evicted_count = h_evicted.data();
    got.free_pages = h_free.data();

    if (!prhe_check_all_outputs(spec, expected, got, error)) {
        return false;
    }

    if (result) {
        result->live_count = h_live_count;
        result->live_sum = h_live_sum;
        result->live_hash = h_live_hash;
        result->page_table_checksum = h_checksum[0];
        result->evicted_count = h_evicted[0];
        result->free_pages = h_free[0];
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

    PrheOracleState oracle;
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
                "scenario %-34s step %02zu/%02zu active=%d %s%s%s\n",
                sc.name.c_str(),
                i,
                sc.steps.size(),
                sc.steps[i].run.active_count,
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
        if (a[i].live_count != b[i].live_count ||
            a[i].live_sum != b[i].live_sum ||
            a[i].live_hash != b[i].live_hash ||
            a[i].page_table_checksum != b[i].page_table_checksum ||
            a[i].evicted_count != b[i].evicted_count ||
            a[i].free_pages != b[i].free_pages) {
            if (error) {
                std::ostringstream oss;
                oss << "replay mismatch at step " << i
                    << ": checksum a=0x" << std::hex << a[i].page_table_checksum
                    << ", b=0x" << b[i].page_table_checksum;
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
        scenarios.push_back(make_budget_pressure_scenario());
        scenarios.push_back(make_min_budget_scenario());
        scenarios.push_back(make_window_boundary_scenario());
        scenarios.push_back(make_permutable_no_new_pages_scenario());

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
                    std::printf("scenario %-34s exact replay PASS\n", sc.name.c_str());
                } else {
                    all_ok = false;
                    std::printf("scenario %-34s exact replay FAIL  %s\n", sc.name.c_str(), compare_error.c_str());
                }
            } else {
                all_ok = false;
                std::printf("scenario %-34s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }

            if (sc.do_permuted_replay) {
                Scenario perm = permute_scenario(sc);
                std::vector<StepResult> perm_results;
                std::string perm_error;

                const bool ok_perm = run_scenario_once(
                    perm,
                    false,
                    &perm_results,
                    &passed,
                    &total,
                    &perm_error);

                if (ok_perm) {
                    std::string compare_error;
                    if (compare_results(base_results, perm_results, &compare_error)) {
                        std::printf("scenario %-34s permuted replay PASS\n", sc.name.c_str());
                    } else {
                        all_ok = false;
                        std::printf("scenario %-34s permuted replay FAIL  %s\n", sc.name.c_str(), compare_error.c_str());
                    }
                } else {
                    all_ok = false;
                    std::printf("scenario %-34s permuted replay FAIL  %s\n", sc.name.c_str(), perm_error.c_str());
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
