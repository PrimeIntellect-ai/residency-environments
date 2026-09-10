// ============================================================================
// file: test_route_compact_reduce.cu
// ============================================================================

#include "route_compact_reduce_common.h"
#include "route_compact_reduce_oracle.hpp"

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

static constexpr uint64_t g_state = 0x6a09e667f3bcc909ULL;
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

    double uniform01() {
        return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
    }

    bool chance(double p) {
        return uniform01() < p;
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
        if (ptr) {
            cudaFree(ptr);
        }
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
        if (raw) {
            cudaFree(raw);
        }
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
                    oss << "left guard corrupted for " << name << " at byte " << i
                        << ": got 0x" << std::hex << static_cast<int>(before[i]);
                    *error = oss.str();
                }
                return false;
            }
        }

        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (after[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "right guard corrupted for " << name << " at byte " << i
                        << ": got 0x" << std::hex << static_cast<int>(after[i]);
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
    RcrRunSpec run;
    std::vector<int16_t> x;
    std::vector<int16_t> expert;
    std::vector<int16_t> weight;
    std::vector<uint8_t> valid;
};

static int sample_zipf_1_2(SplitMix64& rng, int limit_e) {
    double total = 0.0;
    for (int i = 0; i < limit_e; ++i) {
        total += 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
    }

    const double target = rng.uniform01() * total;
    double accum = 0.0;
    for (int i = 0; i < limit_e; ++i) {
        accum += 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
        if (accum >= target) {
            return i;
        }
    }

    return limit_e - 1;
}

static int choose_expert(
    SplitMix64& rng,
    int E,
    int distribution_id,
    bool avoid_last_expert) {
    const int limit_e = avoid_last_expert && E > 1 ? E - 1 : E;
    if (limit_e <= 1) {
        return 0;
    }

    switch (distribution_id) {
        case RCR_DIST_UNIFORM:
            return rng.uniform_int(0, limit_e - 1);

        case RCR_DIST_ZIPF_1_2:
            return sample_zipf_1_2(rng, limit_e);

        case RCR_DIST_SINGLE_HOT:
            if (rng.chance(0.88)) {
                return 0;
            }
            return rng.uniform_int(1, limit_e - 1);

        case RCR_DIST_FEW_HOT: {
            const int hot_count = std::min(limit_e, limit_e >= 8 ? 4 : 2);
            if (rng.chance(0.90)) {
                return rng.uniform_int(0, hot_count - 1);
            }
            if (hot_count == limit_e) {
                return rng.uniform_int(0, limit_e - 1);
            }
            return rng.uniform_int(hot_count, limit_e - 1);
        }

        case RCR_DIST_MANY_INVALID:
            return rng.uniform_int(0, limit_e - 1);

        case RCR_DIST_MANY_DUPLICATE:
            if (rng.chance(0.70)) {
                return rng.uniform_int(0, std::min(limit_e - 1, 3));
            }
            return rng.uniform_int(0, limit_e - 1);

        default:
            return rng.uniform_int(0, limit_e - 1);
    }
}

static int16_t choose_weight(SplitMix64& rng, int distribution_id) {
    if (distribution_id == RCR_DIST_MANY_INVALID && rng.chance(0.15)) {
        return 0;
    }

    int w = rng.uniform_int(-9, 9);
    if (w == 0) {
        w = rng.chance(0.5) ? 1 : -1;
    }
    return static_cast<int16_t>(w);
}

static void force_cancel_duplicate_token(HostCase* hc) {
    const int N = hc->run.N;
    const int E = hc->run.E;
    const int K = hc->run.K;
    if (N < 8 || E < 2) {
        return;
    }

    const int t = 7;
    const int e = std::min(1, E - 1);
    hc->valid[t] = 1;

    for (int k = 0; k < K; ++k) {
        hc->expert[t * K + k] = static_cast<int16_t>(e);
        hc->weight[t * K + k] = 0;
    }

    if (K == 2) {
        hc->weight[t * K + 0] = 5;
        hc->weight[t * K + 1] = -5;
    } else {
        hc->weight[t * K + 0] = 7;
        hc->weight[t * K + 1] = -2;
        hc->weight[t * K + 2] = -5;
        hc->weight[t * K + 3] = 0;
    }
}

static HostCase make_case(
    const char* name,
    int N,
    int C,
    int E,
    int K,
    int distribution_id,
    uint64_t case_seed,
    bool force_all_invalid,
    bool force_empty_last_expert,
    bool inject_cancel_duplicate) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = RCR_ABI_VERSION;
    hc.run.N = N;
    hc.run.C = C;
    hc.run.E = E;
    hc.run.K = K;
    hc.run.seed_id = static_cast<int32_t>(case_seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((case_seed >> 32) & 0x7fffffff);

    if (!rcr_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated run spec");
    }

    SplitMix64 rng(g_state ^ case_seed);

    hc.x.resize(static_cast<size_t>(N) * static_cast<size_t>(C));
    hc.expert.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.weight.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.valid.resize(static_cast<size_t>(N));

    for (size_t i = 0; i < hc.x.size(); ++i) {
        int v = rng.uniform_int(-31, 31);
        if (v == 0 && rng.chance(0.85)) {
            v = rng.chance(0.5) ? 1 : -1;
        }
        hc.x[i] = static_cast<int16_t>(v);
    }

    for (int t = 0; t < N; ++t) {
        bool is_valid = true;

        if (force_all_invalid) {
            is_valid = false;
        } else if (distribution_id == RCR_DIST_MANY_INVALID) {
            is_valid = !rng.chance(0.55);
        } else {
            is_valid = !rng.chance(0.03);
        }

        hc.valid[t] = static_cast<uint8_t>(is_valid ? 1 : 0);

        if (!is_valid) {
            for (int k = 0; k < K; ++k) {
                hc.expert[t * K + k] = rng.chance(0.70)
                    ? static_cast<int16_t>(-1)
                    : static_cast<int16_t>(choose_expert(rng, E, distribution_id, force_empty_last_expert));
                hc.weight[t * K + k] = choose_weight(rng, distribution_id);
            }
            continue;
        }

        if (distribution_id == RCR_DIST_MANY_DUPLICATE) {
            const int e0 = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            for (int k = 0; k < K; ++k) {
                int e = e0;
                if (k == K - 1 && rng.chance(0.35)) {
                    e = choose_expert(rng, E, RCR_DIST_UNIFORM, force_empty_last_expert);
                }
                hc.expert[t * K + k] = static_cast<int16_t>(e);
                hc.weight[t * K + k] = choose_weight(rng, distribution_id);
            }
            continue;
        }

        for (int k = 0; k < K; ++k) {
            const double invalid_slot_prob =
                distribution_id == RCR_DIST_MANY_INVALID ? 0.45 : 0.04;

            if (rng.chance(invalid_slot_prob)) {
                hc.expert[t * K + k] = -1;
                hc.weight[t * K + k] = choose_weight(rng, distribution_id);
                continue;
            }

            const int e = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            hc.expert[t * K + k] = static_cast<int16_t>(e);
            hc.weight[t * K + k] = choose_weight(rng, distribution_id);
        }
    }

    if (inject_cancel_duplicate) {
        force_cancel_duplicate_token(&hc);
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_case(
        "uniform_N4096_C16_E8_K2",
        4096, 16, 8, 2,
        RCR_DIST_UNIFORM,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "zipf_N16384_C32_E32_K4",
        16384, 32, 32, 4,
        RCR_DIST_ZIPF_1_2,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "single_hot_N32768_C64_E128_K2",
        32768, 64, 128, 2,
        RCR_DIST_SINGLE_HOT,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "few_hot_N65536_C128_E32_K4",
        65536, 128, 32, 4,
        RCR_DIST_FEW_HOT,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "many_invalid_N8192_C16_E128_K4",
        8192, 16, 128, 4,
        RCR_DIST_MANY_INVALID,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "many_duplicate_N131072_C64_E8_K4",
        131072, 64, 8, 4,
        RCR_DIST_MANY_DUPLICATE,
        s++,
        false,
        false,
        true));

    cases.push_back(make_case(
        "edge_all_invalid_N4096_C32_E8_K2",
        4096, 32, 8, 2,
        RCR_DIST_UNIFORM,
        s++,
        true,
        false,
        false));

    cases.push_back(make_case(
        "edge_cancel_and_empty_expert_N8192_C128_E32_K4",
        8192, 128, 32, 4,
        RCR_DIST_MANY_DUPLICATE,
        s++,
        false,
        true,
        true));

    cases.push_back(make_case(
        "edge_empty_last_zipf_N32768_C16_E128_K2",
        32768, 16, 128, 2,
        RCR_DIST_ZIPF_1_2,
        s++,
        false,
        true,
        false));

    return cases;
}

static bool check_input_unchanged(const HostCase& hc, const DeviceBuffer<int16_t>& d_x,
                                  const DeviceBuffer<int16_t>& d_expert,
                                  const DeviceBuffer<int16_t>& d_weight,
                                  const DeviceBuffer<uint8_t>& d_valid,
                                  std::string* error) {
    const std::vector<int16_t> x_back = d_x.download();
    const std::vector<int16_t> expert_back = d_expert.download();
    const std::vector<int16_t> weight_back = d_weight.download();
    const std::vector<uint8_t> valid_back = d_valid.download();

    if (x_back != hc.x) {
        if (error) *error = "input x modified";
        return false;
    }
    if (expert_back != hc.expert) {
        if (error) *error = "input expert modified";
        return false;
    }
    if (weight_back != hc.weight) {
        if (error) *error = "input weight modified";
        return false;
    }
    if (valid_back != hc.valid) {
        if (error) *error = "input valid modified";
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
    const int C = hc.run.C;
    const int E = hc.run.E;
    const int K = hc.run.K;

    DeviceBuffer<int16_t> d_x;
    DeviceBuffer<int16_t> d_expert;
    DeviceBuffer<int16_t> d_weight;
    DeviceBuffer<uint8_t> d_valid;

    d_x.allocate(hc.x.size());
    d_expert.allocate(hc.expert.size());
    d_weight.allocate(hc.weight.size());
    d_valid.allocate(hc.valid.size());

    d_x.upload(hc.x);
    d_expert.upload(hc.expert);
    d_weight.upload(hc.weight);
    d_valid.upload(hc.valid);

    GuardedDeviceBuffer<int32_t> d_counts;
    GuardedDeviceBuffer<int32_t> d_offsets;
    GuardedDeviceBuffer<int32_t> d_packed_token;
    GuardedDeviceBuffer<int32_t> d_packed_weight;
    GuardedDeviceBuffer<int64_t> d_sum;
    GuardedDeviceBuffer<int32_t> d_argmax_abs;

    d_counts.allocate(static_cast<size_t>(E));
    d_offsets.allocate(static_cast<size_t>(E + 1));
    d_packed_token.allocate(static_cast<size_t>(N) * static_cast<size_t>(K));
    d_packed_weight.allocate(static_cast<size_t>(N) * static_cast<size_t>(K));
    d_sum.allocate(static_cast<size_t>(E) * static_cast<size_t>(C));
    d_argmax_abs.allocate(static_cast<size_t>(E) * static_cast<size_t>(C));

    RcrInputs inputs = {};
    inputs.x = d_x.ptr;
    inputs.expert = d_expert.ptr;
    inputs.weight = d_weight.ptr;
    inputs.valid = d_valid.ptr;

    RcrOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.offsets = d_offsets.ptr;
    outputs.packed_token = d_packed_token.ptr;
    outputs.packed_weight = d_packed_weight.ptr;
    outputs.sum = d_sum.ptr;
    outputs.argmax_abs = d_argmax_abs.ptr;

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

    if (!check_input_unchanged(hc, d_x, d_expert, d_weight, d_valid, error)) {
        return false;
    }

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_offsets.check_guards("offsets", error)) return false;
    if (!d_packed_token.check_guards("packed_token", error)) return false;
    if (!d_packed_weight.check_guards("packed_weight", error)) return false;
    if (!d_sum.check_guards("sum", error)) return false;
    if (!d_argmax_abs.check_guards("argmax_abs", error)) return false;

    const std::vector<int32_t> h_counts = d_counts.download_data();
    const std::vector<int32_t> h_offsets = d_offsets.download_data();
    const std::vector<int32_t> h_packed_token = d_packed_token.download_data();
    const std::vector<int32_t> h_packed_weight = d_packed_weight.download_data();
    const std::vector<int64_t> h_sum = d_sum.download_data();
    const std::vector<int32_t> h_argmax_abs = d_argmax_abs.download_data();

    RcrHostInputsView host_inputs = {};
    host_inputs.x = hc.x.data();
    host_inputs.expert = hc.expert.data();
    host_inputs.weight = hc.weight.data();
    host_inputs.valid = hc.valid.data();

    RcrExpected expected;
    rcr_cpu_oracle(hc.run, host_inputs, &expected);

    RcrHostOutputsView got = {};
    got.counts = h_counts.data();
    got.offsets = h_offsets.data();
    got.packed_token = h_packed_token.data();
    got.packed_weight = h_packed_weight.data();
    got.sum = h_sum.data();
    got.argmax_abs = h_argmax_abs.data();

    if (!rcr_check_counts_offsets(hc.run, expected, got, error)) {
        return false;
    }

    if (!rcr_check_packed_intermediates(hc.run, expected, got, error)) {
        return false;
    }

    if (!rcr_check_final_outputs(hc.run, expected, got, error)) {
        return false;
    }

    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        RcrProblemSpec spec = {};
        spec.abi_version = RCR_ABI_VERSION;
        spec.max_N = RCR_MAX_N;
        spec.max_C = RCR_MAX_C;
        spec.max_E = RCR_MAX_E;
        spec.max_K = RCR_MAX_K;
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

        const std::vector<HostCase> cases = build_test_cases();

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
                    "case %-48s PASS  N=%d C=%d E=%d K=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.N,
                    hc.run.C,
                    hc.run.E,
                    hc.run.K,
                    hc.run.distribution_id);
            } else {
                std::printf(
                    "case %-48s FAIL  %s\n",
                    hc.name.c_str(),
                    error.c_str());
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
