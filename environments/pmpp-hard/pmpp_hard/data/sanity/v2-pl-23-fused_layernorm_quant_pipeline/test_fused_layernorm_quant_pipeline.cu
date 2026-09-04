// file: test_fused_layernorm_quant_pipeline.cu

#include "fused_layernorm_quant_pipeline_common.h"
#include "fused_layernorm_quant_pipeline_oracle.hpp"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0x6f2d58c4a19e73b5ULL;
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
    FlqpRunSpec run;
    std::vector<float> x;
    std::vector<float> weight;
    std::vector<float> bias;
};

static float exact_frac(int num, int denom) {
    return static_cast<float>(num) / static_cast<float>(denom);
}

static float make_weight(int d, int distribution_id) {
    if (distribution_id == FLQP_DIST_ALL_EQUAL) {
        static const float vals[] = {0.5f, 1.0f, 1.5f, 2.0f};
        return vals[d & 3];
    }

    if (distribution_id == FLQP_DIST_OUTLIERS) {
        if ((d % 257) == 0) return -2.0f;
        if ((d % 149) == 0) return 3.0f;
    }

    if (distribution_id == FLQP_DIST_ZEROISH) {
        if ((d % 17) == 0) return 0.0f;
    }

    static const float vals[] = {
        -1.5f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f, 1.5f, 2.0f
    };
    return vals[(d * 13 + 5) & 7];
}

static float make_bias(int d, int distribution_id) {
    if (distribution_id == FLQP_DIST_ALL_EQUAL) {
        static const float vals[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
        return vals[d % 5];
    }

    if (distribution_id == FLQP_DIST_OUTLIERS) {
        if ((d % 509) == 0) return 8.0f;
        if ((d % 331) == 0) return -8.0f;
    }

    if (distribution_id == FLQP_DIST_ZEROISH) {
        static const float vals[] = {-0.03125f, 0.0f, 0.03125f};
        return vals[d % 3];
    }

    static const float vals[] = {
        -0.75f, -0.25f, -0.125f, 0.0f, 0.125f, 0.25f, 0.75f
    };
    return vals[(d * 7 + 3) % 7];
}

static float make_x_value(
    SplitMix64& rng,
    int row,
    int d,
    int D,
    int distribution_id) {
    (void)D;

    if (distribution_id == FLQP_DIST_ALL_EQUAL) {
        static const float row_vals[] = {-3.0f, -1.0f, 0.0f, 2.0f, 4.0f};
        return row_vals[row % 5];
    }

    if (distribution_id == FLQP_DIST_OUTLIERS) {
        if ((row % 97) == 0 && d == ((row * 13) % D)) return 512.0f;
        if ((row % 131) == 0 && d == ((row * 17 + 3) % D)) return -512.0f;

        static const float vals[] = {
            -4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f
        };
        return vals[rng.uniform_int(0, 8)];
    }

    if (distribution_id == FLQP_DIST_MANY_TIES) {
        static const float vals[] = {
            -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f
        };
        return vals[(row * 11 + d * 3) % 9];
    }

    if (distribution_id == FLQP_DIST_ZEROISH) {
        // Near-zero-variance rows: mostly a row constant plus tiny power-of-two offsets.
        const float base = exact_frac((row % 7) - 3, 8);
        if ((d % 64) == 0) return base + exact_frac(1, 1024);
        if ((d % 97) == 0) return base - exact_frac(1, 1024);
        return base;
    }

    static const float vals[] = {
        -2.0f, -1.5f, -1.0f, -0.5f, -0.25f,
         0.0f,
         0.25f, 0.5f, 1.0f, 1.5f, 2.0f
    };
    return vals[rng.uniform_int(0, 10)];
}

static HostCase make_case(
    const char* name,
    int N,
    int D,
    int distribution_id,
    float eps,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = FLQP_ABI_VERSION;
    hc.run.N = N;
    hc.run.D = D;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);
    hc.run.eps = eps;

    if (!flqp_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid FlqpRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.x.resize((size_t)N * (size_t)D);
    hc.weight.resize((size_t)D);
    hc.bias.resize((size_t)D);

    for (int d = 0; d < D; ++d) {
        hc.weight[(size_t)d] = make_weight(d, distribution_id);
        hc.bias[(size_t)d] = make_bias(d, distribution_id);
    }

    for (int row = 0; row < N; ++row) {
        for (int d = 0; d < D; ++d) {
            hc.x[(size_t)row * (size_t)D + (size_t)d] =
                make_x_value(rng, row, d, D, distribution_id);
        }
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_case(
        "uniform_N4096_D256",
        4096,
        256,
        FLQP_DIST_UNIFORM,
        1.0e-5f,
        s++));

    cases.push_back(make_case(
        "outliers_N4096_D4096",
        4096,
        4096,
        FLQP_DIST_OUTLIERS,
        1.0e-5f,
        s++));

    cases.push_back(make_case(
        "all_equal_N8192_D1024",
        8192,
        1024,
        FLQP_DIST_ALL_EQUAL,
        1.0e-5f,
        s++));

    cases.push_back(make_case(
        "many_ties_N16384_D256",
        16384,
        256,
        FLQP_DIST_MANY_TIES,
        1.0e-5f,
        s++));

    cases.push_back(make_case(
        "near_zero_var_N4096_D4096",
        4096,
        4096,
        FLQP_DIST_ZEROISH,
        1.0e-7f,
        s++));

    cases.push_back(make_case(
        "large_N131072_D256_outliers",
        131072,
        256,
        FLQP_DIST_OUTLIERS,
        1.0e-5f,
        s++));

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<float>& d_x,
    const DeviceBuffer<float>& d_weight,
    const DeviceBuffer<float>& d_bias,
    std::string* error) {
    if (d_x.download() != hc.x) {
        if (error) *error = "input x modified";
        return false;
    }

    if (d_weight.download() != hc.weight) {
        if (error) *error = "input weight modified";
        return false;
    }

    if (d_bias.download() != hc.bias) {
        if (error) *error = "input bias modified";
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

    DeviceBuffer<float> d_x;
    DeviceBuffer<float> d_weight;
    DeviceBuffer<float> d_bias;

    d_x.allocate(hc.x.size());
    d_weight.allocate(hc.weight.size());
    d_bias.allocate(hc.bias.size());

    d_x.upload(hc.x);
    d_weight.upload(hc.weight);
    d_bias.upload(hc.bias);

    GuardedDeviceBuffer<int8_t> d_q_int8;
    GuardedDeviceBuffer<float> d_scale;
    GuardedDeviceBuffer<float> d_dequant;
    GuardedDeviceBuffer<int64_t> d_code_sum;

    d_q_int8.allocate((size_t)N * (size_t)D);
    d_scale.allocate((size_t)N);
    d_dequant.allocate((size_t)N * (size_t)D);
    d_code_sum.allocate((size_t)N);

    FlqpInputs inputs = {};
    inputs.x = d_x.ptr;
    inputs.weight = d_weight.ptr;
    inputs.bias = d_bias.ptr;

    FlqpOutputs outputs = {};
    outputs.q_int8 = d_q_int8.ptr;
    outputs.scale = d_scale.ptr;
    outputs.dequant = d_dequant.ptr;
    outputs.code_sum = d_code_sum.ptr;

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

    if (!check_input_unchanged(hc, d_x, d_weight, d_bias, error)) {
        return false;
    }

    if (!d_q_int8.check_guards("q_int8", error)) return false;
    if (!d_scale.check_guards("scale", error)) return false;
    if (!d_dequant.check_guards("dequant", error)) return false;
    if (!d_code_sum.check_guards("code_sum", error)) return false;

    const std::vector<int8_t> h_q_int8 = d_q_int8.download_data();
    const std::vector<float> h_scale = d_scale.download_data();
    const std::vector<float> h_dequant = d_dequant.download_data();
    const std::vector<int64_t> h_code_sum = d_code_sum.download_data();

    FlqpHostInputsView host_inputs = {};
    host_inputs.x = hc.x.data();
    host_inputs.weight = hc.weight.data();
    host_inputs.bias = hc.bias.data();

    FlqpExpected expected;
    flqp_cpu_oracle(hc.run, host_inputs, &expected);

    FlqpHostOutputsView got = {};
    got.q_int8 = h_q_int8.data();
    got.scale = h_scale.data();
    got.dequant = h_dequant.data();
    got.code_sum = h_code_sum.data();

    return flqp_check_all_outputs(hc.run, expected, got, error);
}

int main() {
    try {
        std::fesetround(FE_TONEAREST);
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int max_N = FLQP_MIN_N;
        int max_D = 256;

        for (const HostCase& hc : cases) {
            max_N = std::max(max_N, hc.run.N);
            max_D = std::max(max_D, hc.run.D);
        }

        FlqpProblemSpec spec = {};
        spec.abi_version = FLQP_ABI_VERSION;
        spec.max_N = max_N;
        spec.max_D = max_D;
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
                    "case %-34s PASS  N=%d D=%d eps=%.1e dist=%d\n",
                    hc.name.c_str(),
                    hc.run.N,
                    hc.run.D,
                    static_cast<double>(hc.run.eps),
                    hc.run.distribution_id);
            } else {
                std::printf("case %-34s FAIL  %s\n", hc.name.c_str(), error.c_str());
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
