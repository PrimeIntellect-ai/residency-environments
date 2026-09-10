// ============================================================================
// file: test_moe_dispatch_combine.cu
// ============================================================================

#include "moe_dispatch_combine_common.h"
#include "moe_dispatch_combine_oracle.hpp"

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

static constexpr uint64_t g_state = 0x243f6a8885a308d3ULL;
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
    MdcRunSpec run;
    std::vector<int16_t> expert;
    std::vector<int16_t> gate;
    std::vector<uint8_t> valid;
    std::vector<int16_t> expert_out;
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
        case MDC_DIST_UNIFORM:
            return rng.uniform_int(0, limit_e - 1);

        case MDC_DIST_ZIPF_1_2:
            return sample_zipf_1_2(rng, limit_e);

        case MDC_DIST_SINGLE_HOT:
            if (rng.chance(0.90)) {
                return 0;
            }
            return rng.uniform_int(1, limit_e - 1);

        case MDC_DIST_FEW_HOT: {
            const int hot_count = std::min(limit_e, limit_e >= 8 ? 4 : 2);
            if (rng.chance(0.92)) {
                return rng.uniform_int(0, hot_count - 1);
            }
            if (hot_count == limit_e) {
                return rng.uniform_int(0, limit_e - 1);
            }
            return rng.uniform_int(hot_count, limit_e - 1);
        }

        case MDC_DIST_MANY_INVALID:
            return rng.uniform_int(0, limit_e - 1);

        case MDC_DIST_MANY_DUPLICATE:
            if (rng.chance(0.75)) {
                return rng.uniform_int(0, std::min(limit_e - 1, 3));
            }
            return rng.uniform_int(0, limit_e - 1);

        default:
            return rng.uniform_int(0, limit_e - 1);
    }
}

static int16_t choose_gate(SplitMix64& rng, int distribution_id) {
    switch (distribution_id) {
        case MDC_DIST_SINGLE_HOT:
        case MDC_DIST_FEW_HOT:
            return static_cast<int16_t>(rng.uniform_int(-16, 127));

        case MDC_DIST_MANY_INVALID:
            return static_cast<int16_t>(rng.uniform_int(-127, 64));

        case MDC_DIST_MANY_DUPLICATE:
            return static_cast<int16_t>(rng.uniform_int(-32, 127));

        default:
            return static_cast<int16_t>(rng.uniform_int(-127, 127));
    }
}

static HostCase make_random_case(
    const char* name,
    int N,
    int D,
    int E,
    int K,
    int cap,
    int distribution_id,
    uint64_t case_seed,
    bool force_all_invalid,
    bool force_empty_last_expert) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = MDC_ABI_VERSION;
    hc.run.N = N;
    hc.run.D = D;
    hc.run.E = E;
    hc.run.K = K;
    hc.run.cap = cap;
    hc.run.seed_id = static_cast<int32_t>(case_seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((case_seed >> 32) & 0x7fffffff);

    if (!mdc_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated run spec");
    }

    SplitMix64 rng(g_state ^ case_seed);

    hc.expert.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.gate.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.valid.resize(static_cast<size_t>(N));
    hc.expert_out.resize(static_cast<size_t>(E) * static_cast<size_t>(D));

    for (size_t i = 0; i < hc.expert_out.size(); ++i) {
        int v = rng.uniform_int(-31, 31);
        if (v == 0 && rng.chance(0.90)) {
            v = rng.chance(0.5) ? 1 : -1;
        }
        hc.expert_out[i] = static_cast<int16_t>(v);
    }

    for (int t = 0; t < N; ++t) {
        bool is_valid = true;

        if (force_all_invalid) {
            is_valid = false;
        } else if (distribution_id == MDC_DIST_MANY_INVALID) {
            is_valid = !rng.chance(0.60);
        } else {
            is_valid = !rng.chance(0.02);
        }

        hc.valid[t] = static_cast<uint8_t>(is_valid ? 1 : 0);

        if (!is_valid) {
            for (int k = 0; k < K; ++k) {
                hc.expert[t * K + k] = rng.chance(0.70)
                    ? static_cast<int16_t>(-1)
                    : static_cast<int16_t>(choose_expert(rng, E, distribution_id, force_empty_last_expert));
                hc.gate[t * K + k] = choose_gate(rng, distribution_id);
            }
            continue;
        }

        if (distribution_id == MDC_DIST_MANY_DUPLICATE) {
            const int e0 = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            for (int k = 0; k < K; ++k) {
                int e = e0;
                if (k == K - 1 && rng.chance(0.35)) {
                    e = choose_expert(rng, E, MDC_DIST_UNIFORM, force_empty_last_expert);
                }
                hc.expert[t * K + k] = static_cast<int16_t>(e);
                hc.gate[t * K + k] = choose_gate(rng, distribution_id);
            }
            continue;
        }

        for (int k = 0; k < K; ++k) {
            const double unused_prob =
                distribution_id == MDC_DIST_MANY_INVALID ? 0.50 : 0.04;

            if (rng.chance(unused_prob)) {
                hc.expert[t * K + k] = -1;
                hc.gate[t * K + k] = choose_gate(rng, distribution_id);
                continue;
            }

            const int e = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            hc.expert[t * K + k] = static_cast<int16_t>(e);
            hc.gate[t * K + k] = choose_gate(rng, distribution_id);
        }
    }

    return hc;
}

static HostCase make_empty_case(
    const char* name,
    int N,
    int D,
    int E,
    int K,
    int cap,
    int distribution_id,
    uint64_t case_seed) {
    HostCase hc = make_random_case(
        name,
        N,
        D,
        E,
        K,
        cap,
        distribution_id,
        case_seed,
        true,
        false);
    return hc;
}

static HostCase make_exact_capacity_case() {
    const int N = 4096;
    const int D = 16;
    const int E = 8;
    const int K = 2;
    const int cap = 16;

    HostCase hc = make_empty_case(
        "edge_exact_capacity_e0",
        N,
        D,
        E,
        K,
        cap,
        MDC_DIST_UNIFORM,
        0x900000000001ULL);

    for (int t = 0; t < cap; ++t) {
        hc.valid[t] = 1;
        hc.expert[t * K + 0] = 0;
        hc.gate[t * K + 0] = static_cast<int16_t>(64 - t);
        hc.expert[t * K + 1] = -1;
        hc.gate[t * K + 1] = 0;
    }

    return hc;
}

static HostCase make_tie_boundary_case() {
    const int N = 4096;
    const int D = 16;
    const int E = 8;
    const int K = 4;
    const int cap = 16;

    HostCase hc = make_empty_case(
        "edge_tie_boundary_token_id",
        N,
        D,
        E,
        K,
        cap,
        MDC_DIST_SINGLE_HOT,
        0x900000000002ULL);

    for (int t = 0; t < cap + 6; ++t) {
        hc.valid[t] = 1;
        hc.expert[t * K + 0] = 0;
        hc.gate[t * K + 0] = 42;
        for (int k = 1; k < K; ++k) {
            hc.expert[t * K + k] = -1;
            hc.gate[t * K + k] = 0;
        }
    }

    return hc;
}

static HostCase make_duplicate_capacity_case() {
    const int N = 4096;
    const int D = 32;
    const int E = 8;
    const int K = 4;
    const int cap = 16;

    HostCase hc = make_empty_case(
        "edge_duplicate_slots_capacity",
        N,
        D,
        E,
        K,
        cap,
        MDC_DIST_MANY_DUPLICATE,
        0x900000000003ULL);

    for (int t = 0; t < 5; ++t) {
        hc.valid[t] = 1;
        for (int k = 0; k < K; ++k) {
            hc.expert[t * K + k] = 0;
            hc.gate[t * K + k] = 100;
        }
    }

    return hc;
}

static HostCase make_invalid_and_empty_expert_case() {
    HostCase hc = make_random_case(
        "edge_invalid_tokens_empty_last_expert",
        8192,
        64,
        32,
        4,
        32,
        MDC_DIST_MANY_INVALID,
        0x900000000004ULL,
        false,
        true);

    for (int t = 0; t < std::min(128, hc.run.N); ++t) {
        hc.valid[t] = 0;
    }

    return hc;
}

static HostCase make_all_low_priority_routes_dropped_case() {
    const int N = 4096;
    const int D = 32;
    const int E = 8;
    const int K = 4;
    const int cap = 16;

    HostCase hc = make_empty_case(
        "edge_all_routes_for_token_dropped",
        N,
        D,
        E,
        K,
        cap,
        MDC_DIST_SINGLE_HOT,
        0x900000000005ULL);

    for (int t = 0; t < cap; ++t) {
        hc.valid[t] = 1;
        hc.expert[t * K + 0] = 0;
        hc.gate[t * K + 0] = 127;
        for (int k = 1; k < K; ++k) {
            hc.expert[t * K + k] = -1;
            hc.gate[t * K + k] = 0;
        }
    }

    const int victim = cap + 8;
    hc.valid[victim] = 1;
    for (int k = 0; k < K; ++k) {
        hc.expert[victim * K + k] = 0;
        hc.gate[victim * K + k] = -127;
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_random_case(
        "uniform_N4096_D16_E8_K2_cap64",
        4096, 16, 8, 2, 64,
        MDC_DIST_UNIFORM,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "zipf_N16384_D32_E32_K4_cap64",
        16384, 32, 32, 4, 64,
        MDC_DIST_ZIPF_1_2,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "single_hot_overflow_N32768_D64_E128_K2_cap16",
        32768, 64, 128, 2, 16,
        MDC_DIST_SINGLE_HOT,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "few_hot_overflow_N65536_D128_E32_K4_cap32",
        65536, 128, 32, 4, 32,
        MDC_DIST_FEW_HOT,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "many_invalid_N8192_D16_E128_K4_cap64",
        8192, 16, 128, 4, 64,
        MDC_DIST_MANY_INVALID,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "many_duplicate_N131072_D32_E8_K4_cap16",
        131072, 32, 8, 4, 16,
        MDC_DIST_MANY_DUPLICATE,
        s++,
        false,
        false));

    cases.push_back(make_empty_case(
        "edge_all_invalid_N4096_D32_E8_K2_cap16",
        4096, 32, 8, 2, 16,
        MDC_DIST_UNIFORM,
        s++));

    cases.push_back(make_exact_capacity_case());
    cases.push_back(make_tie_boundary_case());
    cases.push_back(make_duplicate_capacity_case());
    cases.push_back(make_invalid_and_empty_expert_case());
    cases.push_back(make_all_low_priority_routes_dropped_case());

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<int16_t>& d_expert,
    const DeviceBuffer<int16_t>& d_gate,
    const DeviceBuffer<uint8_t>& d_valid,
    const DeviceBuffer<int16_t>& d_expert_out,
    std::string* error) {
    const std::vector<int16_t> expert_back = d_expert.download();
    const std::vector<int16_t> gate_back = d_gate.download();
    const std::vector<uint8_t> valid_back = d_valid.download();
    const std::vector<int16_t> expert_out_back = d_expert_out.download();

    if (expert_back != hc.expert) {
        if (error) *error = "input expert modified";
        return false;
    }
    if (gate_back != hc.gate) {
        if (error) *error = "input gate modified";
        return false;
    }
    if (valid_back != hc.valid) {
        if (error) *error = "input valid modified";
        return false;
    }
    if (expert_out_back != hc.expert_out) {
        if (error) *error = "input expert_out modified";
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
    const int D = hc.run.D;
    const int E = hc.run.E;
    const int K = hc.run.K;
    const int cap = hc.run.cap;
    const size_t packed_capacity = static_cast<size_t>(E) * static_cast<size_t>(cap);

    DeviceBuffer<int16_t> d_expert;
    DeviceBuffer<int16_t> d_gate;
    DeviceBuffer<uint8_t> d_valid;
    DeviceBuffer<int16_t> d_expert_out;

    d_expert.allocate(hc.expert.size());
    d_gate.allocate(hc.gate.size());
    d_valid.allocate(hc.valid.size());
    d_expert_out.allocate(hc.expert_out.size());

    d_expert.upload(hc.expert);
    d_gate.upload(hc.gate);
    d_valid.upload(hc.valid);
    d_expert_out.upload(hc.expert_out);

    GuardedDeviceBuffer<int32_t> d_counts;
    GuardedDeviceBuffer<int32_t> d_offsets;
    GuardedDeviceBuffer<int32_t> d_packed_token;
    GuardedDeviceBuffer<int32_t> d_packed_slot;
    GuardedDeviceBuffer<int16_t> d_packed_gate;
    GuardedDeviceBuffer<uint8_t> d_dropped;
    GuardedDeviceBuffer<int64_t> d_y;

    d_counts.allocate(static_cast<size_t>(E));
    d_offsets.allocate(static_cast<size_t>(E + 1));
    d_packed_token.allocate(packed_capacity);
    d_packed_slot.allocate(packed_capacity);
    d_packed_gate.allocate(packed_capacity);
    d_dropped.allocate(static_cast<size_t>(N) * static_cast<size_t>(K));
    d_y.allocate(static_cast<size_t>(N) * static_cast<size_t>(D));

    MdcInputs inputs = {};
    inputs.expert = d_expert.ptr;
    inputs.gate = d_gate.ptr;
    inputs.valid = d_valid.ptr;
    inputs.expert_out = d_expert_out.ptr;

    MdcOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.offsets = d_offsets.ptr;
    outputs.packed_token = d_packed_token.ptr;
    outputs.packed_slot = d_packed_slot.ptr;
    outputs.packed_gate = d_packed_gate.ptr;
    outputs.dropped = d_dropped.ptr;
    outputs.y = d_y.ptr;

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

    if (!check_input_unchanged(hc, d_expert, d_gate, d_valid, d_expert_out, error)) {
        return false;
    }

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_offsets.check_guards("offsets", error)) return false;
    if (!d_packed_token.check_guards("packed_token", error)) return false;
    if (!d_packed_slot.check_guards("packed_slot", error)) return false;
    if (!d_packed_gate.check_guards("packed_gate", error)) return false;
    if (!d_dropped.check_guards("dropped", error)) return false;
    if (!d_y.check_guards("y", error)) return false;

    const std::vector<int32_t> h_counts = d_counts.download_data();
    const std::vector<int32_t> h_offsets = d_offsets.download_data();
    const std::vector<int32_t> h_packed_token = d_packed_token.download_data();
    const std::vector<int32_t> h_packed_slot = d_packed_slot.download_data();
    const std::vector<int16_t> h_packed_gate = d_packed_gate.download_data();
    const std::vector<uint8_t> h_dropped = d_dropped.download_data();
    const std::vector<int64_t> h_y = d_y.download_data();

    MdcHostInputsView host_inputs = {};
    host_inputs.expert = hc.expert.data();
    host_inputs.gate = hc.gate.data();
    host_inputs.valid = hc.valid.data();
    host_inputs.expert_out = hc.expert_out.data();

    MdcExpected expected;
    mdc_cpu_oracle(hc.run, host_inputs, &expected);

    if (static_cast<size_t>(expected.offsets[static_cast<size_t>(E)]) > packed_capacity) {
        if (error) *error = "oracle total packed routes exceeds E*cap";
        return false;
    }

    MdcHostOutputsView got = {};
    got.counts = h_counts.data();
    got.offsets = h_offsets.data();
    got.packed_token = h_packed_token.data();
    got.packed_slot = h_packed_slot.data();
    got.packed_gate = h_packed_gate.data();
    got.dropped = h_dropped.data();
    got.y = h_y.data();

    if (!mdc_check_counts_offsets(hc.run, expected, got, error)) {
        return false;
    }

    if (!mdc_check_packed_intermediates(hc.run, expected, got, error)) {
        return false;
    }

    if (!mdc_check_dropped(hc.run, expected, got, error)) {
        return false;
    }

    if (!mdc_check_y(hc.run, expected, got, error)) {
        return false;
    }

    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        MdcProblemSpec spec = {};
        spec.abi_version = MDC_ABI_VERSION;
        spec.max_N = MDC_MAX_N;
        spec.max_D = MDC_MAX_D;
        spec.max_E = MDC_MAX_E;
        spec.max_K = MDC_MAX_K;
        spec.max_cap = MDC_MAX_CAP;
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
                    "case %-56s PASS  N=%d D=%d E=%d K=%d cap=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.N,
                    hc.run.D,
                    hc.run.E,
                    hc.run.K,
                    hc.run.cap,
                    hc.run.distribution_id);
            } else {
                std::printf(
                    "case %-56s FAIL  %s\n",
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
