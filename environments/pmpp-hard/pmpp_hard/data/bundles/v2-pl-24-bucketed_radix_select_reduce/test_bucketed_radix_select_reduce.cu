// file: test_bucketed_radix_select_reduce.cu

#include "bucketed_radix_select_reduce_common.h"
#include "bucketed_radix_select_reduce_oracle.hpp"

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

static constexpr uint64_t g_state = 0x38f6d1ac9b247e53ULL;
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

    uint32_t next_u32() {
        return static_cast<uint32_t>(next_u64() >> 32);
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

struct HostCase {
    std::string name;
    BrsrRunSpec run;
    std::vector<uint32_t> key;
    std::vector<int32_t> value;
};

static uint32_t make_key(
    SplitMix64& rng,
    int idx,
    int N,
    int distribution_id,
    bool all_ties) {
    if (all_ties) {
        return 0x7abc1234u;
    }

    switch (distribution_id) {
        case BRSR_DIST_UNIFORM:
            return rng.next_u32();

        case BRSR_DIST_CLUSTERED: {
            const uint32_t cluster = static_cast<uint32_t>((idx / 257) & 0xff);
            const uint32_t high = ((cluster * 37u + 91u) & 0xffu) << 24;
            const uint32_t mid = static_cast<uint32_t>((idx * 13u) & 0xffffu) << 8;
            const uint32_t low = static_cast<uint32_t>(rng.uniform_int(0, 255));
            return high | mid | low;
        }

        case BRSR_DIST_MANY_TIES: {
            const uint32_t group = static_cast<uint32_t>((idx / 8) % 1024);
            const uint32_t high = ((group * 17u) & 0xffu) << 24;
            const uint32_t low = (group & 0xffffu) << 4;
            return high | low;
        }

        case BRSR_DIST_HIGH_BUCKET_HOT: {
            if ((idx % 5) != 0) {
                const uint32_t hot = 0xf0000000u | (static_cast<uint32_t>(idx % 31) << 20);
                return hot | (rng.next_u32() & 0x000fffffu);
            }
            return rng.next_u32() & 0x7fffffffu;
        }

        default:
            return rng.next_u32();
    }
}

static int32_t make_value(SplitMix64& rng, int idx) {
    int32_t v = rng.next_i32();

    if ((idx % 257) == 0) v = INT_MAX - (idx & 1023);
    if ((idx % 389) == 0) v = INT_MIN + (idx & 1023);
    if ((idx % 1021) == 0) v = 0;

    return v;
}

static HostCase make_case(
    const char* name,
    int N,
    int T,
    int distribution_id,
    uint64_t seed,
    bool all_ties) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = BRSR_ABI_VERSION;
    hc.run.N = N;
    hc.run.T = T;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!brsr_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid BrsrRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.key.resize((size_t)N);
    hc.value.resize((size_t)N);

    for (int i = 0; i < N; ++i) {
        hc.key[(size_t)i] = make_key(rng, i, N, distribution_id, all_ties);
        hc.value[(size_t)i] = make_value(rng, i);
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_case(
        "uniform_N4096_T16",
        4096,
        16,
        BRSR_DIST_UNIFORM,
        s++,
        false));

    cases.push_back(make_case(
        "clustered_N65536_T256",
        65536,
        256,
        BRSR_DIST_CLUSTERED,
        s++,
        false));

    cases.push_back(make_case(
        "many_ties_N8192_T4096",
        8192,
        4096,
        BRSR_DIST_MANY_TIES,
        s++,
        false));

    cases.push_back(make_case(
        "edge_TeqN_all_ties_N4096_T4096",
        4096,
        4096,
        BRSR_DIST_MANY_TIES,
        s++,
        true));

    cases.push_back(make_case(
        "edge_T1_uniform_N8192",
        8192,
        1,
        BRSR_DIST_UNIFORM,
        s++,
        false));

    cases.push_back(make_case(
        "large_N1M_hot_T16",
        1 << 20,
        16,
        BRSR_DIST_HIGH_BUCKET_HOT,
        s++,
        false));

    cases.push_back(make_case(
        "clustered_N131072_T256",
        131072,
        256,
        BRSR_DIST_CLUSTERED,
        s++,
        false));

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<uint32_t>& d_key,
    const DeviceBuffer<int32_t>& d_value,
    std::string* error) {
    if (d_key.download() != hc.key) {
        if (error) *error = "input key modified";
        return false;
    }

    if (d_value.download() != hc.value) {
        if (error) *error = "input value modified";
        return false;
    }

    return true;
}

static bool run_one_case(
    const HostCase& hc,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    std::string* error) {
    const int count = brsr_count_host(hc.run.N, hc.run.T);

    DeviceBuffer<uint32_t> d_key;
    DeviceBuffer<int32_t> d_value;

    d_key.allocate(hc.key.size());
    d_value.allocate(hc.value.size());

    d_key.upload(hc.key);
    d_value.upload(hc.value);

    GuardedDeviceBuffer<uint32_t> d_threshold_key;
    GuardedDeviceBuffer<int32_t> d_count;
    GuardedDeviceBuffer<int32_t> d_topT_indices;
    GuardedDeviceBuffer<int64_t> d_topT_sum;
    GuardedDeviceBuffer<int32_t> d_topT_max;
    GuardedDeviceBuffer<int32_t> d_topT_argmax;
    GuardedDeviceBuffer<int32_t> d_pass_histograms;
    GuardedDeviceBuffer<int32_t> d_chosen_bucket;
    GuardedDeviceBuffer<int32_t> d_carried_rank;
    GuardedDeviceBuffer<uint32_t> d_prefix_after_pass;

    d_threshold_key.allocate(1);
    d_count.allocate(1);
    d_topT_indices.allocate((size_t)count);
    d_topT_sum.allocate(1);
    d_topT_max.allocate(1);
    d_topT_argmax.allocate(1);
    d_pass_histograms.allocate((size_t)BRSR_NUM_PASSES * (size_t)BRSR_BUCKETS);
    d_chosen_bucket.allocate((size_t)BRSR_NUM_PASSES);
    d_carried_rank.allocate((size_t)BRSR_NUM_PASSES);
    d_prefix_after_pass.allocate((size_t)BRSR_NUM_PASSES);

    BrsrInputs inputs = {};
    inputs.key = d_key.ptr;
    inputs.value = d_value.ptr;

    BrsrOutputs outputs = {};
    outputs.threshold_key = d_threshold_key.ptr;
    outputs.count = d_count.ptr;
    outputs.topT_indices = d_topT_indices.ptr;
    outputs.topT_sum = d_topT_sum.ptr;
    outputs.topT_max = d_topT_max.ptr;
    outputs.topT_argmax = d_topT_argmax.ptr;
    outputs.pass_histograms = d_pass_histograms.ptr;
    outputs.chosen_bucket = d_chosen_bucket.ptr;
    outputs.carried_rank = d_carried_rank.ptr;
    outputs.prefix_after_pass = d_prefix_after_pass.ptr;

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(solution_run(
        state,
        &hc.run,
        &inputs,
        &outputs,
        workspace,
        workspace_bytes,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(hc, d_key, d_value, error)) {
        return false;
    }

    if (!d_threshold_key.check_guards("threshold_key", error)) return false;
    if (!d_count.check_guards("count", error)) return false;
    if (!d_topT_indices.check_guards("topT_indices", error)) return false;
    if (!d_topT_sum.check_guards("topT_sum", error)) return false;
    if (!d_topT_max.check_guards("topT_max", error)) return false;
    if (!d_topT_argmax.check_guards("topT_argmax", error)) return false;
    if (!d_pass_histograms.check_guards("pass_histograms", error)) return false;
    if (!d_chosen_bucket.check_guards("chosen_bucket", error)) return false;
    if (!d_carried_rank.check_guards("carried_rank", error)) return false;
    if (!d_prefix_after_pass.check_guards("prefix_after_pass", error)) return false;

    const std::vector<uint32_t> h_threshold_key = d_threshold_key.download_data();
    const std::vector<int32_t> h_count = d_count.download_data();
    const std::vector<int32_t> h_topT_indices = d_topT_indices.download_data();
    const std::vector<int64_t> h_topT_sum = d_topT_sum.download_data();
    const std::vector<int32_t> h_topT_max = d_topT_max.download_data();
    const std::vector<int32_t> h_topT_argmax = d_topT_argmax.download_data();
    const std::vector<int32_t> h_pass_histograms = d_pass_histograms.download_data();
    const std::vector<int32_t> h_chosen_bucket = d_chosen_bucket.download_data();
    const std::vector<int32_t> h_carried_rank = d_carried_rank.download_data();
    const std::vector<uint32_t> h_prefix_after_pass = d_prefix_after_pass.download_data();

    BrsrHostInputsView host_inputs = {};
    host_inputs.key = hc.key.data();
    host_inputs.value = hc.value.data();

    BrsrExpected expected;
    brsr_cpu_oracle(hc.run, host_inputs, &expected);

    BrsrHostOutputsView got = {};
    got.threshold_key = h_threshold_key.data();
    got.count = h_count.data();
    got.topT_indices = h_topT_indices.data();
    got.topT_sum = h_topT_sum.data();
    got.topT_max = h_topT_max.data();
    got.topT_argmax = h_topT_argmax.data();
    got.pass_histograms = h_pass_histograms.data();
    got.chosen_bucket = h_chosen_bucket.data();
    got.carried_rank = h_carried_rank.data();
    got.prefix_after_pass = h_prefix_after_pass.data();

    return brsr_check_all_outputs(expected, got, error);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int max_N = BRSR_MIN_N;
        int max_T = 1;

        for (const HostCase& hc : cases) {
            max_N = std::max(max_N, hc.run.N);
            max_T = std::max(max_T, hc.run.T);
        }

        BrsrProblemSpec spec = {};
        spec.abi_version = BRSR_ABI_VERSION;
        spec.max_N = max_N;
        spec.max_T = max_T;
        spec.flags = 0;

        size_t workspace_bytes = solution_workspace_bytes(&spec); if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/
        if (workspace_bytes == 0) {
            throw std::runtime_error("solution_workspace_bytes returned 0");
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(workspace_bytes);

        int passed = 0;
        const int total = static_cast<int>(cases.size());

        for (const HostCase& hc : cases) {
            std::string error;
            bool ok = false;

            try {
                ok = run_one_case(
                    hc,
                    state,
                    workspace.ptr,
                    workspace_bytes,
                    stream,
                    &error);
            } catch (const std::exception& ex) {
                error = ex.what();
                ok = false;
            }

            if (ok) {
                ++passed;
                std::printf(
                    "case %-36s PASS  N=%d T=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.N,
                    hc.run.T,
                    hc.run.distribution_id);
            } else {
                std::printf("case %-36s FAIL  %s\n", hc.name.c_str(), error.c_str());
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));

        std::printf("passed %d / %d\n", passed, total);
        return passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
