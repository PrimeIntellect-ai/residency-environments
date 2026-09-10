// file: test_segmented_groupby_topk.cu

#include "segmented_groupby_topk_common.h"
#include "segmented_groupby_topk_oracle.hpp"

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

static constexpr uint64_t g_state = 0x81d2c9a4703f5b6eULL;
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
    SgtRunSpec run;
    std::vector<int32_t> group_id;
    std::vector<int32_t> key;
    std::vector<int32_t> value;
};

static int choose_group(
    SplitMix64& rng,
    int row,
    int G,
    int distribution_id,
    bool force_empty_last_group,
    bool single_hot) {
    const int usable_G = force_empty_last_group && G > 1 ? G - 1 : G;

    if (usable_G <= 1) return 0;

    if (single_hot || distribution_id == SGTK_DIST_SINGLE_HOT) {
        if (rng.chance_permille(930)) return 0;
        return rng.uniform_int(1, usable_G - 1);
    }

    if (distribution_id == SGTK_DIST_ZIPF_HOT) {
        const int r = rng.uniform_int(0, 999);
        if (r < 560) return 0;
        if (r < 740) return std::min(1, usable_G - 1);
        if (r < 840) return std::min(2, usable_G - 1);
        if (r < 900) return std::min(3, usable_G - 1);

        const int hot_tail = std::min(usable_G, 16);
        if (r < 965) return rng.uniform_int(0, hot_tail - 1);

        return rng.uniform_int(0, usable_G - 1);
    }

    if (distribution_id == SGTK_DIST_MANY_TIES) {
        if (rng.chance_permille(750)) {
            return row % std::min(usable_G, 8);
        }
    }

    return rng.uniform_int(0, usable_G - 1);
}

static int32_t make_key(
    SplitMix64& rng,
    int row,
    int group,
    int distribution_id,
    bool all_key_ties) {
    if (all_key_ties) {
        return 12345;
    }

    if (distribution_id == SGTK_DIST_MANY_TIES) {
        return static_cast<int32_t>((group * 17 + (row / 8)) % 23);
    }

    if (distribution_id == SGTK_DIST_ZIPF_HOT || distribution_id == SGTK_DIST_SINGLE_HOT) {
        if (group == 0 && (row % 5) == 0) return 1000000 - (row % 97);
        if ((row % 13) == 0) return 10000 + (group % 257);
    }

    return rng.uniform_int(-1000000, 1000000);
}

static int32_t make_value(
    SplitMix64& rng,
    int row,
    int keep_permille,
    bool all_filtered,
    bool dense_keep) {
    if (all_filtered) {
        return -rng.uniform_int(1, 1000);
    }

    const bool keep = dense_keep || rng.chance_permille(keep_permille);
    if (!keep) {
        return -rng.uniform_int(0, 1000);
    }

    int32_t v = static_cast<int32_t>(rng.uniform_int(1, 100000));

    if ((row % 257) == 0) v = INT_MAX - (row & 1023);
    if ((row % 389) == 0) v = 1;
    if ((row % 1021) == 0) v = 777;

    return v;
}

static HostCase make_case(
    const char* name,
    int N,
    int G,
    int M,
    int distribution_id,
    int keep_permille,
    uint64_t seed,
    bool force_empty_last_group,
    bool all_filtered,
    bool single_hot,
    bool all_key_ties) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = SGTK_ABI_VERSION;
    hc.run.N = N;
    hc.run.G = G;
    hc.run.M = M;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!sgtk_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid SgtRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.group_id.resize((size_t)N);
    hc.key.resize((size_t)N);
    hc.value.resize((size_t)N);

    const bool dense_keep = keep_permille >= 1000;

    for (int i = 0; i < N; ++i) {
        const int g = choose_group(
            rng,
            i,
            G,
            distribution_id,
            force_empty_last_group,
            single_hot);

        hc.group_id[(size_t)i] = g;
        hc.key[(size_t)i] = make_key(rng, i, g, distribution_id, all_key_ties);
        hc.value[(size_t)i] = make_value(rng, i, keep_permille, all_filtered, dense_keep);
    }

    if (force_empty_last_group && G > 1) {
        for (int i = 0; i < N; ++i) {
            if (hc.group_id[(size_t)i] == G - 1) {
                hc.group_id[(size_t)i] = 0;
            }
        }
    }

    if (single_hot) {
        for (int i = 0; i < std::min(N, 512); ++i) {
            hc.group_id[(size_t)i] = 0;
            hc.value[(size_t)i] = 1 + (i % 31);
            hc.key[(size_t)i] = all_key_ties ? 12345 : (1000000 - (i % 17));
        }
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_case(
        "uniform_N4096_G8_M4_keep30_empty",
        4096, 8, 4, SGTK_DIST_UNIFORM, 300, s++,
        true, false, false, false));

    cases.push_back(make_case(
        "zipf_hot_N8192_G64_M16_keep50",
        8192, 64, 16, SGTK_DIST_ZIPF_HOT, 500, s++,
        false, false, false, false));

    cases.push_back(make_case(
        "ties_N16384_G1024_M64_keep70",
        16384, 1024, 64, SGTK_DIST_MANY_TIES, 700, s++,
        false, false, false, false));

    cases.push_back(make_case(
        "all_filtered_N4096_G64_M16",
        4096, 64, 16, SGTK_DIST_UNIFORM, 0, s++,
        false, true, false, false));

    cases.push_back(make_case(
        "M_gt_group_size_N4096_G1024_M64",
        4096, 1024, 64, SGTK_DIST_UNIFORM, 350, s++,
        false, false, false, false));

    cases.push_back(make_case(
        "single_hot_N65536_G1024_M64_keep90",
        65536, 1024, 64, SGTK_DIST_SINGLE_HOT, 900, s++,
        true, false, true, false));

    cases.push_back(make_case(
        "all_key_ties_N8192_G8_M16",
        8192, 8, 16, SGTK_DIST_MANY_TIES, 1000, s++,
        false, false, false, true));

    cases.push_back(make_case(
        "large_N131072_G64_M16_zipf",
        131072, 64, 16, SGTK_DIST_ZIPF_HOT, 550, s++,
        false, false, false, false));

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<int32_t>& d_group_id,
    const DeviceBuffer<int32_t>& d_key,
    const DeviceBuffer<int32_t>& d_value,
    std::string* error) {
    if (d_group_id.download() != hc.group_id) {
        if (error) *error = "input group_id modified";
        return false;
    }

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
    const int N = hc.run.N;
    const int G = hc.run.G;

    DeviceBuffer<int32_t> d_group_id;
    DeviceBuffer<int32_t> d_key;
    DeviceBuffer<int32_t> d_value;

    d_group_id.allocate(hc.group_id.size());
    d_key.allocate(hc.key.size());
    d_value.allocate(hc.value.size());

    d_group_id.upload(hc.group_id);
    d_key.upload(hc.key);
    d_value.upload(hc.value);

    GuardedDeviceBuffer<int32_t> d_group_counts;
    GuardedDeviceBuffer<int32_t> d_group_offsets;
    GuardedDeviceBuffer<int32_t> d_packed_topk_origidx;
    GuardedDeviceBuffer<int64_t> d_per_group_sum;
    GuardedDeviceBuffer<int32_t> d_per_group_max;
    GuardedDeviceBuffer<int32_t> d_per_group_argmax;
    GuardedDeviceBuffer<int32_t> d_kept_count;

    d_group_counts.allocate((size_t)G);
    d_group_offsets.allocate((size_t)G + 1);
    d_packed_topk_origidx.allocate((size_t)N);
    d_per_group_sum.allocate((size_t)G);
    d_per_group_max.allocate((size_t)G);
    d_per_group_argmax.allocate((size_t)G);
    d_kept_count.allocate((size_t)G);

    SgtInputs inputs = {};
    inputs.group_id = d_group_id.ptr;
    inputs.key = d_key.ptr;
    inputs.value = d_value.ptr;

    SgtOutputs outputs = {};
    outputs.group_counts = d_group_counts.ptr;
    outputs.group_offsets = d_group_offsets.ptr;
    outputs.packed_topk_origidx = d_packed_topk_origidx.ptr;
    outputs.per_group_sum = d_per_group_sum.ptr;
    outputs.per_group_max = d_per_group_max.ptr;
    outputs.per_group_argmax = d_per_group_argmax.ptr;
    outputs.kept_count = d_kept_count.ptr;

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

    if (!check_input_unchanged(hc, d_group_id, d_key, d_value, error)) {
        return false;
    }

    if (!d_group_counts.check_guards("group_counts", error)) return false;
    if (!d_group_offsets.check_guards("group_offsets", error)) return false;
    if (!d_packed_topk_origidx.check_guards("packed_topk_origidx", error)) return false;
    if (!d_per_group_sum.check_guards("per_group_sum", error)) return false;
    if (!d_per_group_max.check_guards("per_group_max", error)) return false;
    if (!d_per_group_argmax.check_guards("per_group_argmax", error)) return false;
    if (!d_kept_count.check_guards("kept_count", error)) return false;

    const std::vector<int32_t> h_group_counts = d_group_counts.download_data();
    const std::vector<int32_t> h_group_offsets = d_group_offsets.download_data();
    const std::vector<int32_t> h_packed_topk_origidx = d_packed_topk_origidx.download_data();
    const std::vector<int64_t> h_per_group_sum = d_per_group_sum.download_data();
    const std::vector<int32_t> h_per_group_max = d_per_group_max.download_data();
    const std::vector<int32_t> h_per_group_argmax = d_per_group_argmax.download_data();
    const std::vector<int32_t> h_kept_count = d_kept_count.download_data();

    SgtHostInputsView host_inputs = {};
    host_inputs.group_id = hc.group_id.data();
    host_inputs.key = hc.key.data();
    host_inputs.value = hc.value.data();

    SgtExpected expected;
    sgtk_cpu_oracle(hc.run, host_inputs, &expected);

    SgtHostOutputsView got = {};
    got.group_counts = h_group_counts.data();
    got.group_offsets = h_group_offsets.data();
    got.packed_topk_origidx = h_packed_topk_origidx.data();
    got.per_group_sum = h_per_group_sum.data();
    got.per_group_max = h_per_group_max.data();
    got.per_group_argmax = h_per_group_argmax.data();
    got.kept_count = h_kept_count.data();

    return sgtk_check_all_outputs(hc.run, expected, got, error);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int max_N = SGTK_MIN_N;
        int max_G = 8;

        for (const HostCase& hc : cases) {
            max_N = std::max(max_N, hc.run.N);
            max_G = std::max(max_G, hc.run.G);
        }

        SgtProblemSpec spec = {};
        spec.abi_version = SGTK_ABI_VERSION;
        spec.max_N = max_N;
        spec.max_G = max_G;
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
                    "case %-40s PASS  N=%d G=%d M=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.N,
                    hc.run.G,
                    hc.run.M,
                    hc.run.distribution_id);
            } else {
                std::printf("case %-40s FAIL  %s\n", hc.name.c_str(), error.c_str());
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
