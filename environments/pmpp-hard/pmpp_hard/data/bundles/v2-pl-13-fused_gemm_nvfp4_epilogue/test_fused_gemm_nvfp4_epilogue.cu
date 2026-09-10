// ============================================================================
// file: test_fused_gemm_nvfp4_epilogue.cu
// ============================================================================

#include "fused_gemm_nvfp4_epilogue_common.h"
#include "fused_gemm_nvfp4_epilogue_oracle.hpp"

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

static constexpr uint64_t g_state = 0x7a6f5e2d3c1b9048ULL;
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
    FgeRunSpec run;
    std::vector<uint8_t> a_packed;
    std::vector<uint8_t> b_packed;
    std::vector<int16_t> a_scale_q;
    std::vector<int16_t> b_scale_q;
    std::vector<int32_t> bias;
};

static void set_packed_code(std::vector<uint8_t>* bytes, size_t logical_idx, uint8_t code) {
    const size_t byte_idx = logical_idx >> 1;
    const uint8_t c = code & 0x0f;

    if ((logical_idx & 1u) == 0u) {
        (*bytes)[byte_idx] = static_cast<uint8_t>(((*bytes)[byte_idx] & 0xf0u) | c);
    } else {
        (*bytes)[byte_idx] = static_cast<uint8_t>(((*bytes)[byte_idx] & 0x0fu) | (c << 4));
    }
}

static uint8_t random_code(SplitMix64& rng, int distribution_id, bool saturating) {
    if (saturating) {
        return rng.chance_permille(500) ? 7u : 15u;
    }

    switch (distribution_id) {
        case FGE_DIST_UNIFORM:
            return static_cast<uint8_t>(rng.uniform_int(0, 15));

        case FGE_DIST_PEAKED:
            if (rng.chance_permille(720)) return 0u;
            if (rng.chance_permille(650)) return 7u;
            return static_cast<uint8_t>(rng.uniform_int(0, 15));

        case FGE_DIST_MANY_ZERO:
            if (rng.chance_permille(850)) return 0u;
            return static_cast<uint8_t>(rng.uniform_int(0, 15));

        case FGE_DIST_MANY_TIES: {
            static const uint8_t codes[] = {0, 1, 1, 2, 2, 9, 9, 15};
            return codes[rng.uniform_int(0, 7)];
        }

        default:
            return static_cast<uint8_t>(rng.uniform_int(0, 15));
    }
}

static HostCase make_case(
    const char* name,
    int M,
    int N,
    int K,
    int shift,
    int activation,
    int distribution_id,
    uint64_t seed,
    bool zero_negative_scales,
    bool saturating_codes) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = FGE_ABI_VERSION;
    hc.run.M = M;
    hc.run.N = N;
    hc.run.K = K;
    hc.run.epilogue_shift = shift;
    hc.run.activation = activation;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!fge_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated FgeRunSpec");
    }

    SplitMix64 rng(g_state ^ seed);

    const size_t a_elems = (size_t)M * (size_t)K;
    const size_t b_elems = (size_t)K * (size_t)N;

    hc.a_packed.assign(fge_packed_bytes_for(a_elems), 0);
    hc.b_packed.assign(fge_packed_bytes_for(b_elems), 0);
    hc.a_scale_q.resize((size_t)M);
    hc.b_scale_q.resize((size_t)N);
    hc.bias.resize((size_t)N);

    for (size_t i = 0; i < a_elems; ++i) {
        set_packed_code(&hc.a_packed, i, random_code(rng, distribution_id, saturating_codes));
    }

    for (size_t i = 0; i < b_elems; ++i) {
        set_packed_code(&hc.b_packed, i, random_code(rng, distribution_id, saturating_codes));
    }

    for (int m = 0; m < M; ++m) {
        int v = rng.uniform_int(1, 16);
        if (zero_negative_scales) {
            if (m % 11 == 0) v = 0;
            else if (m % 5 == 0) v = -rng.uniform_int(1, 16);
        }
        hc.a_scale_q[(size_t)m] = static_cast<int16_t>(v);
    }

    for (int n = 0; n < N; ++n) {
        int v = rng.uniform_int(1, 16);
        if (zero_negative_scales) {
            if (n % 13 == 0) v = 0;
            else if (n % 7 == 0) v = -rng.uniform_int(1, 16);
        }
        hc.b_scale_q[(size_t)n] = static_cast<int16_t>(v);

        int b = rng.uniform_int(-500000, 500000);
        if (saturating_codes && (n % 17 == 0)) {
            b = rng.chance_permille(500) ? 2000000000 : -2000000000;
        }
        hc.bias[(size_t)n] = static_cast<int32_t>(b);
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_case(
        "uniform_64x64x64_none",
        64, 64, 64, 6, FGE_ACT_NONE, FGE_DIST_UNIFORM,
        s++, false, false));

    cases.push_back(make_case(
        "k_non_tile_96x128x96_relu",
        96, 128, 96, 8, FGE_ACT_RELU, FGE_DIST_UNIFORM,
        s++, false, false));

    cases.push_back(make_case(
        "many_zero_128x192x160_clamp",
        128, 192, 160, 10, FGE_ACT_CLAMP_INT8_RANGE, FGE_DIST_MANY_ZERO,
        s++, false, false));

    cases.push_back(make_case(
        "negative_scales_192x128x224_none",
        192, 128, 224, 9, FGE_ACT_NONE, FGE_DIST_MANY_TIES,
        s++, true, false));

    cases.push_back(make_case(
        "saturating_256x256x320_clamp",
        256, 256, 320, 4, FGE_ACT_CLAMP_INT8_RANGE, FGE_DIST_PEAKED,
        s++, true, true));

    cases.push_back(make_case(
        "peaked_384x256x384_relu",
        384, 256, 384, 12, FGE_ACT_RELU, FGE_DIST_PEAKED,
        s++, false, false));

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<uint8_t>& d_a_packed,
    const DeviceBuffer<uint8_t>& d_b_packed,
    const DeviceBuffer<int16_t>& d_a_scale_q,
    const DeviceBuffer<int16_t>& d_b_scale_q,
    const DeviceBuffer<int32_t>& d_bias,
    std::string* error) {
    if (d_a_packed.download() != hc.a_packed) {
        if (error) *error = "input a_packed modified";
        return false;
    }

    if (d_b_packed.download() != hc.b_packed) {
        if (error) *error = "input b_packed modified";
        return false;
    }

    if (d_a_scale_q.download() != hc.a_scale_q) {
        if (error) *error = "input a_scale_q modified";
        return false;
    }

    if (d_b_scale_q.download() != hc.b_scale_q) {
        if (error) *error = "input b_scale_q modified";
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
    const int M = hc.run.M;
    const int N = hc.run.N;

    DeviceBuffer<uint8_t> d_a_packed;
    DeviceBuffer<uint8_t> d_b_packed;
    DeviceBuffer<int16_t> d_a_scale_q;
    DeviceBuffer<int16_t> d_b_scale_q;
    DeviceBuffer<int32_t> d_bias;

    d_a_packed.allocate(hc.a_packed.size());
    d_b_packed.allocate(hc.b_packed.size());
    d_a_scale_q.allocate(hc.a_scale_q.size());
    d_b_scale_q.allocate(hc.b_scale_q.size());
    d_bias.allocate(hc.bias.size());

    d_a_packed.upload(hc.a_packed);
    d_b_packed.upload(hc.b_packed);
    d_a_scale_q.upload(hc.a_scale_q);
    d_b_scale_q.upload(hc.b_scale_q);
    d_bias.upload(hc.bias);

    GuardedDeviceBuffer<int32_t> d_c_out;
    d_c_out.allocate((size_t)M * (size_t)N);

    FgeInputs inputs = {};
    inputs.a_packed = d_a_packed.ptr;
    inputs.b_packed = d_b_packed.ptr;
    inputs.a_scale_q = d_a_scale_q.ptr;
    inputs.b_scale_q = d_b_scale_q.ptr;
    inputs.bias = d_bias.ptr;

    FgeOutputs outputs = {};
    outputs.c_out = d_c_out.ptr;

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

    if (!check_input_unchanged(
            hc,
            d_a_packed,
            d_b_packed,
            d_a_scale_q,
            d_b_scale_q,
            d_bias,
            error)) {
        return false;
    }

    if (!d_c_out.check_guards("c_out", error)) {
        return false;
    }

    const std::vector<int32_t> h_c_out = d_c_out.download_data();

    FgeHostInputsView host_inputs = {};
    host_inputs.a_packed = hc.a_packed.data();
    host_inputs.b_packed = hc.b_packed.data();
    host_inputs.a_scale_q = hc.a_scale_q.data();
    host_inputs.b_scale_q = hc.b_scale_q.data();
    host_inputs.bias = hc.bias.data();

    FgeExpected expected;
    fge_cpu_oracle(hc.run, host_inputs, &expected);

    FgeHostOutputsView got = {};
    got.c_out = h_c_out.data();

    return fge_check_all_outputs(hc.run, expected, got, error);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int max_M = 64;
        int max_N = 64;
        int max_K = 64;

        for (const HostCase& hc : cases) {
            max_M = std::max(max_M, hc.run.M);
            max_N = std::max(max_N, hc.run.N);
            max_K = std::max(max_K, hc.run.K);
        }

        FgeProblemSpec spec = {};
        spec.abi_version = FGE_ABI_VERSION;
        spec.max_M = max_M;
        spec.max_N = max_N;
        spec.max_K = max_K;
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
                    "case %-36s PASS  M=%d N=%d K=%d shift=%d act=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.M,
                    hc.run.N,
                    hc.run.K,
                    hc.run.epilogue_shift,
                    hc.run.activation,
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
