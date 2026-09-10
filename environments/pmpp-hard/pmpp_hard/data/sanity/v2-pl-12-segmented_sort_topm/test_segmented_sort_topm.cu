// ============================================================================
// file: test_segmented_sort_topm.cu
// ============================================================================

#include "segmented_sort_topm_common.h"
#include "segmented_sort_topm_oracle.hpp"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0x91e10da5c79e7b1dULL;
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

    int uniform_int(int lo, int hi) {
        const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u64() % span);
    }

    bool chance_permille(int per_mille) {
        if (per_mille <= 0) return false;
        if (per_mille >= 1000) return true;
        return static_cast<int>(next_u64() % 1000ULL) < per_mille;
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
    SstRunSpec run;
    std::vector<int32_t> seg_offsets;
    std::vector<int32_t> item_key;
    std::vector<int32_t> item_value;
};

static void normalize_sizes_to_N(std::vector<int>& sizes, int N) {
    int64_t sum = 0;
    for (int v : sizes) sum += v;

    if (sum < N) {
        sizes[0] += static_cast<int>(N - sum);
    } else if (sum > N) {
        int64_t excess = sum - N;
        for (int i = static_cast<int>(sizes.size()) - 1; i >= 0 && excess > 0; --i) {
            int take = static_cast<int>(std::min<int64_t>(sizes[(size_t)i], excess));
            sizes[(size_t)i] -= take;
            excess -= take;
        }
    }
}

static std::vector<int> make_segment_sizes(
    int N,
    int S,
    int distribution_id,
    SplitMix64& rng,
    bool force_empty_segments,
    bool one_giant,
    bool all_size_one) {
    std::vector<int> sizes((size_t)S, 0);

    if (all_size_one || distribution_id == SST_DIST_ALL_SIZE_ONE) {
        for (int s = 0; s < S && s < N; ++s) {
            sizes[(size_t)s] = 1;
        }
        normalize_sizes_to_N(sizes, N);
        return sizes;
    }

    if (one_giant || distribution_id == SST_DIST_ONE_GIANT) {
        if (N >= S) {
            sizes[0] = N - (S - 1);
            for (int s = 1; s < S; ++s) sizes[(size_t)s] = 1;
        } else {
            for (int s = 0; s < N; ++s) sizes[(size_t)s] = 1;
        }
        normalize_sizes_to_N(sizes, N);
        return sizes;
    }

    if (distribution_id == SST_DIST_POWERLAW) {
        const int hot_count = std::min(S, 8);

        for (int i = 0; i < N; ++i) {
            int s;
            if (rng.chance_permille(850)) {
                int r = rng.uniform_int(0, 99);
                if (r < 55) s = 0;
                else if (r < 75) s = std::min(1, hot_count - 1);
                else if (r < 88) s = std::min(2, hot_count - 1);
                else s = rng.uniform_int(0, hot_count - 1);
            } else {
                s = rng.uniform_int(0, S - 1);
            }
            ++sizes[(size_t)s];
        }

        if (force_empty_segments && S > 16) {
            for (int s = S - S / 8; s < S; ++s) {
                sizes[0] += sizes[(size_t)s];
                sizes[(size_t)s] = 0;
            }
        }

        return sizes;
    }

    if (force_empty_segments) {
        const int active_segments = std::max(1, S - S / 8);
        for (int i = 0; i < N; ++i) {
            ++sizes[(size_t)rng.uniform_int(0, active_segments - 1)];
        }
        return sizes;
    }

    int base = N / S;
    int rem = N - base * S;
    for (int s = 0; s < S; ++s) {
        sizes[(size_t)s] = base + (s < rem ? 1 : 0);
    }

    if (distribution_id == SST_DIST_MANY_TIES) {
        for (int s = 0; s + 1 < S; s += 17) {
            if (sizes[(size_t)s] > 0) {
                int move = sizes[(size_t)s] / 2;
                sizes[(size_t)s] -= move;
                sizes[(size_t)s + 1] += move;
            }
        }
        normalize_sizes_to_N(sizes, N);
    }

    return sizes;
}

static int32_t make_key(
    SplitMix64& rng,
    int distribution_id,
    int seg,
    int orig,
    bool all_ties) {
    if (all_ties) return 7;

    if (distribution_id == SST_DIST_MANY_TIES) {
        return static_cast<int32_t>((seg * 13 + orig / 3) % 11);
    }

    if (distribution_id == SST_DIST_POWERLAW) {
        return static_cast<int32_t>(rng.uniform_int(-64, 64));
    }

    if (distribution_id == SST_DIST_ONE_GIANT) {
        if (orig % 5 == 0) return 1000;
        return static_cast<int32_t>(rng.uniform_int(-1000, 1000));
    }

    return static_cast<int32_t>(rng.uniform_int(-1000000, 1000000));
}

static int32_t make_value(SplitMix64& rng, int global_idx, int seg, int orig) {
    int v = rng.uniform_int(-100000, 100000);
    if (global_idx % 257 == 0) v = 2147483000 - (orig % 1024);
    if (global_idx % 389 == 0) v = -2147483000 + (seg % 1024);
    return static_cast<int32_t>(v);
}

static HostCase make_case(
    const char* name,
    int N,
    int S,
    int M,
    int distribution_id,
    uint64_t seed,
    bool force_empty_segments,
    bool one_giant,
    bool all_size_one,
    bool all_ties) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = SST_ABI_VERSION;
    hc.run.N = N;
    hc.run.S = S;
    hc.run.M = M;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!sst_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated SstRunSpec");
    }

    SplitMix64 rng(g_state ^ seed);
    std::vector<int> sizes = make_segment_sizes(
        N,
        S,
        distribution_id,
        rng,
        force_empty_segments,
        one_giant,
        all_size_one);

    hc.seg_offsets.resize((size_t)S + 1);
    hc.seg_offsets[0] = 0;

    for (int s = 0; s < S; ++s) {
        hc.seg_offsets[(size_t)s + 1] =
            hc.seg_offsets[(size_t)s] + static_cast<int32_t>(sizes[(size_t)s]);
    }

    if (hc.seg_offsets[(size_t)S] != N) {
        throw std::runtime_error("segment size generator failed to sum to N");
    }

    hc.item_key.resize((size_t)N);
    hc.item_value.resize((size_t)N);

    for (int s = 0; s < S; ++s) {
        const int begin = hc.seg_offsets[(size_t)s];
        const int end = hc.seg_offsets[(size_t)s + 1];

        for (int idx = begin; idx < end; ++idx) {
            const int orig = idx - begin;
            hc.item_key[(size_t)idx] = make_key(rng, distribution_id, s, orig, all_ties);
            hc.item_value[(size_t)idx] = make_value(rng, idx, s, orig);
        }
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_case(
        "uniform_N4096_S32_M4",
        4096, 32, 4, SST_DIST_UNIFORM, s++,
        false, false, false, false));

    cases.push_back(make_case(
        "powerlaw_N32768_S256_M16",
        32768, 256, 16, SST_DIST_POWERLAW, s++,
        false, false, false, false));

    cases.push_back(make_case(
        "all_size_one_N4096_S4096_M64",
        4096, 4096, 64, SST_DIST_ALL_SIZE_ONE, s++,
        false, false, true, false));

    cases.push_back(make_case(
        "one_giant_N131072_S4096_M64",
        131072, 4096, 64, SST_DIST_ONE_GIANT, s++,
        false, true, false, false));

    cases.push_back(make_case(
        "many_ties_N16384_S256_M16",
        16384, 256, 16, SST_DIST_MANY_TIES, s++,
        false, false, false, false));

    cases.push_back(make_case(
        "empty_segments_N4096_S256_M4",
        4096, 256, 4, SST_DIST_UNIFORM, s++,
        true, false, false, false));

    cases.push_back(make_case(
        "all_ties_N8192_S32_M64",
        8192, 32, 64, SST_DIST_MANY_TIES, s++,
        false, false, false, true));

    cases.push_back(make_case(
        "powerlaw_empty_N65536_S4096_M16",
        65536, 4096, 16, SST_DIST_POWERLAW, s++,
        true, false, false, false));

    cases.push_back(make_case(
        "uniform_large_N131072_S256_M64",
        131072, 256, 64, SST_DIST_UNIFORM, s++,
        false, false, false, false));

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<int32_t>& d_seg_offsets,
    const DeviceBuffer<int32_t>& d_item_key,
    const DeviceBuffer<int32_t>& d_item_value,
    std::string* error) {
    if (d_seg_offsets.download() != hc.seg_offsets) {
        if (error) *error = "input seg_offsets modified";
        return false;
    }

    if (d_item_key.download() != hc.item_key) {
        if (error) *error = "input item_key modified";
        return false;
    }

    if (d_item_value.download() != hc.item_value) {
        if (error) *error = "input item_value modified";
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
    const int S = hc.run.S;

    DeviceBuffer<int32_t> d_seg_offsets;
    DeviceBuffer<int32_t> d_item_key;
    DeviceBuffer<int32_t> d_item_value;

    d_seg_offsets.allocate(hc.seg_offsets.size());
    d_item_key.allocate(hc.item_key.size());
    d_item_value.allocate(hc.item_value.size());

    d_seg_offsets.upload(hc.seg_offsets);
    d_item_key.upload(hc.item_key);
    d_item_value.upload(hc.item_value);

    GuardedDeviceBuffer<int32_t> d_topm_count;
    GuardedDeviceBuffer<int32_t> d_topm_offsets;
    GuardedDeviceBuffer<int32_t> d_packed_topm_key;
    GuardedDeviceBuffer<int32_t> d_packed_topm_value;
    GuardedDeviceBuffer<int32_t> d_packed_topm_origidx;
    GuardedDeviceBuffer<int64_t> d_seg_sum;
    GuardedDeviceBuffer<int32_t> d_seg_max;
    GuardedDeviceBuffer<int32_t> d_seg_argmax;

    d_topm_count.allocate((size_t)S);
    d_topm_offsets.allocate((size_t)S + 1);
    d_packed_topm_key.allocate((size_t)N);
    d_packed_topm_value.allocate((size_t)N);
    d_packed_topm_origidx.allocate((size_t)N);
    d_seg_sum.allocate((size_t)S);
    d_seg_max.allocate((size_t)S);
    d_seg_argmax.allocate((size_t)S);

    SstInputs inputs = {};
    inputs.seg_offsets = d_seg_offsets.ptr;
    inputs.item_key = d_item_key.ptr;
    inputs.item_value = d_item_value.ptr;

    SstOutputs outputs = {};
    outputs.topm_count = d_topm_count.ptr;
    outputs.topm_offsets = d_topm_offsets.ptr;
    outputs.packed_topm_key = d_packed_topm_key.ptr;
    outputs.packed_topm_value = d_packed_topm_value.ptr;
    outputs.packed_topm_origidx = d_packed_topm_origidx.ptr;
    outputs.seg_sum = d_seg_sum.ptr;
    outputs.seg_max = d_seg_max.ptr;
    outputs.seg_argmax = d_seg_argmax.ptr;

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

    if (!check_input_unchanged(hc, d_seg_offsets, d_item_key, d_item_value, error)) {
        return false;
    }

    if (!d_topm_count.check_guards("topm_count", error)) return false;
    if (!d_topm_offsets.check_guards("topm_offsets", error)) return false;
    if (!d_packed_topm_key.check_guards("packed_topm_key", error)) return false;
    if (!d_packed_topm_value.check_guards("packed_topm_value", error)) return false;
    if (!d_packed_topm_origidx.check_guards("packed_topm_origidx", error)) return false;
    if (!d_seg_sum.check_guards("seg_sum", error)) return false;
    if (!d_seg_max.check_guards("seg_max", error)) return false;
    if (!d_seg_argmax.check_guards("seg_argmax", error)) return false;

    const std::vector<int32_t> h_topm_count = d_topm_count.download_data();
    const std::vector<int32_t> h_topm_offsets = d_topm_offsets.download_data();
    const std::vector<int32_t> h_packed_topm_key = d_packed_topm_key.download_data();
    const std::vector<int32_t> h_packed_topm_value = d_packed_topm_value.download_data();
    const std::vector<int32_t> h_packed_topm_origidx = d_packed_topm_origidx.download_data();
    const std::vector<int64_t> h_seg_sum = d_seg_sum.download_data();
    const std::vector<int32_t> h_seg_max = d_seg_max.download_data();
    const std::vector<int32_t> h_seg_argmax = d_seg_argmax.download_data();

    SstHostInputsView host_inputs = {};
    host_inputs.seg_offsets = hc.seg_offsets.data();
    host_inputs.item_key = hc.item_key.data();
    host_inputs.item_value = hc.item_value.data();

    SstExpected expected;
    sst_cpu_oracle(hc.run, host_inputs, &expected);

    SstHostOutputsView got = {};
    got.topm_count = h_topm_count.data();
    got.topm_offsets = h_topm_offsets.data();
    got.packed_topm_key = h_packed_topm_key.data();
    got.packed_topm_value = h_packed_topm_value.data();
    got.packed_topm_origidx = h_packed_topm_origidx.data();
    got.seg_sum = h_seg_sum.data();
    got.seg_max = h_seg_max.data();
    got.seg_argmax = h_seg_argmax.data();

    return sst_check_all_outputs(hc.run, expected, got, error);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int max_N = 0;
        int max_S = 32;

        for (const HostCase& hc : cases) {
            max_N = std::max(max_N, hc.run.N);
            max_S = std::max(max_S, hc.run.S);
        }

        SstProblemSpec spec = {};
        spec.abi_version = SST_ABI_VERSION;
        spec.max_N = max_N;
        spec.max_S = max_S;
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
                    "case %-42s PASS  N=%d S=%d M=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.N,
                    hc.run.S,
                    hc.run.M,
                    hc.run.distribution_id);
            } else {
                std::printf("case %-42s FAIL  %s\n", hc.name.c_str(), error.c_str());
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
