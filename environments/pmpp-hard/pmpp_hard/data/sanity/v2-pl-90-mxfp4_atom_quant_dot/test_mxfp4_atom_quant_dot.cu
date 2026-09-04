// file: test_mxfp4_atom_quant_dot.cu

#include "mxfp4_atom_quant_dot_common.h"
#include "mxfp4_atom_quant_dot_oracle.hpp"

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

static constexpr uint64_t g_state = 0x51c9a3e7d2b84f61ULL;
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
    MxqRunSpec run;
    std::vector<float> x;
    std::vector<float> v;
};

// ---------------------------------------------------------------------------
// Input generation (bit-level, adversarial).
// ---------------------------------------------------------------------------

static uint32_t f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

static float u2f(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// 2^p for p in [-149, 127], exact (subnormal below -126).
static float pow2f(int p) {
    if (p >= -126) return u2f((uint32_t)(p + 127) << 23);
    return u2f(0x800000u >> (-126 - p));
}

static const float kMidpoints[7] = {
    0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f
};

static const float kSatFracs[6] = {
    1.0f, 1.25f, 1.5f, 1.625f, 1.75f, 1.875f
};

// Fill lanes [0, n) of one 32-element scale block.
static void fill_block(
    SplitMix64& rng,
    int distribution_id,
    int r,
    int b,
    int n,
    float* lane) {
    switch (distribution_id) {
        case MXQ_DIST_SMOOTH: {
            const int t = rng.uniform_int(-12, 12);
            for (int i = 0; i < n; ++i) {
                const int q = rng.uniform_int(-1536, 1536);
                lane[i] = (float)q * pow2f(t - 8);
            }
            break;
        }
        case MXQ_DIST_TIES: {
            const int t = rng.uniform_int(-16, 16);
            const int anchor = rng.uniform_int(0, n - 1);
            for (int i = 0; i < n; ++i) {
                if (i == anchor) {
                    lane[i] = pow2f(t + 2);  // 4.0 * 2^t pins sexp = t
                } else {
                    const float m = kMidpoints[rng.uniform_int(0, 6)];
                    const float val = mxq_oracle_pow2_scale(m, t);
                    lane[i] = rng.uniform_int(0, 1) ? -val : val;
                }
            }
            break;
        }
        case MXQ_DIST_SATURATE: {
            const int t = rng.uniform_int(-10, 10);
            const int anchor = rng.uniform_int(0, n - 1);
            for (int i = 0; i < n; ++i) {
                float val;
                if (i == anchor) {
                    val = mxq_oracle_pow2_scale(1.75f, t);  // 7 * 2^(t-2): sat
                } else if (rng.uniform_int(0, 1)) {
                    val = mxq_oracle_pow2_scale(
                        kSatFracs[rng.uniform_int(0, 5)], t);
                } else {
                    val = mxq_oracle_pow2_scale(1.25f, t - rng.uniform_int(1, 6));
                }
                lane[i] = rng.uniform_int(0, 1) ? -val : val;
            }
            break;
        }
        case MXQ_DIST_ZERO_BLOCKS: {
            if (b % 3 == 0) {
                for (int i = 0; i < n; ++i) {
                    lane[i] = u2f((uint32_t)rng.uniform_int(0, 1) << 31);
                }
            } else {
                const int t = rng.uniform_int(-12, 12);
                for (int i = 0; i < n; ++i) {
                    const int q = rng.uniform_int(-1536, 1536);
                    lane[i] = (float)q * pow2f(t - 8);
                }
            }
            break;
        }
        case MXQ_DIST_DENORMAL: {
            for (int i = 0; i < n; ++i) {
                const uint32_t m = (uint32_t)rng.uniform_int(0, 255);
                const uint32_t s = (uint32_t)rng.uniform_int(0, 1) << 31;
                lane[i] = u2f(s | m);
            }
            break;
        }
        case MXQ_DIST_NEGZERO: {
            for (int i = 0; i < n; ++i) {
                if (rng.uniform_int(0, 9) < 7) {
                    lane[i] = u2f((uint32_t)rng.uniform_int(0, 1) << 31);
                } else {
                    const float val = pow2f(rng.uniform_int(-140, -128));
                    lane[i] = rng.uniform_int(0, 1) ? -val : val;
                }
            }
            break;
        }
        case MXQ_DIST_POW2_EDGE: {
            const int t = rng.uniform_int(-130, 30);
            for (int i = 0; i < n; ++i) {
                uint32_t bits = f2u(pow2f(t));
                const int variant = rng.uniform_int(0, 2);
                if (variant == 1) bits += 1;
                else if (variant == 2) bits -= 1;
                if (rng.uniform_int(0, 1)) bits |= 0x80000000u;
                lane[i] = u2f(bits);
            }
            break;
        }
        case MXQ_DIST_CLAMP_LOW: {
            for (int i = 0; i < n; ++i) {
                const uint32_t m = (uint32_t)rng.uniform_int(1, 0x3fffff);
                const uint32_t s = (uint32_t)rng.uniform_int(0, 1) << 31;
                lane[i] = u2f(s | m);
            }
            break;
        }
        case MXQ_DIST_RANDBITS:
        default: {
            for (int i = 0; i < n; ++i) {
                uint32_t u = (uint32_t)rng.next_u64();
                // Cap the biased exponent at 0xF0 so stage-4 products and
                // partial sums can never overflow (contract domain guarantee).
                while (((u >> 23) & 0xffu) > 0xf0u) u -= (0x10u << 23);
                lane[i] = u2f(u);
            }
            break;
        }
    }
    (void)r;
}

static HostCase make_case(
    const std::string& name,
    int R,
    int C,
    int distribution_id,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = MXQ_ABI_VERSION;
    hc.run.R = R;
    hc.run.C = C;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!mxq_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid MxqRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.x.resize((size_t)R * (size_t)C);
    hc.v.resize((size_t)C);

    for (int k = 0; k < C; ++k) {
        hc.v[(size_t)k] = 0.25f * (float)rng.uniform_int(-8, 8);
    }

    const int S = mxq_S(C);
    float lane[MXQ_BLOCK_K];

    for (int r = 0; r < R; ++r) {
        for (int b = 0; b < S; ++b) {
            const int k0 = 32 * b;
            const int n = (k0 + 32 <= C) ? 32 : (C - k0);
            fill_block(rng, distribution_id, r, b, n, lane);
            for (int i = 0; i < n; ++i) {
                hc.x[(size_t)r * (size_t)C + (size_t)(k0 + i)] = lane[i];
            }
        }
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    struct Shape { int R, C; };
    static const Shape shapes[6] = {
        {128, 67}, {257, 129}, {512, 255},
        {1024, 513}, {2048, 1024}, {509, 4099}
    };
    static const char* dist_names[9] = {
        "smooth", "ties", "saturate", "zeroblk", "denorm",
        "negzero", "pow2edge", "clamplow", "randbits"
    };

    std::vector<HostCase> cases;
    uint64_t s = 0x900000001b3ULL;

    for (int d = 0; d < 9; ++d) {
        for (int sh = 0; sh < 6; ++sh) {
            std::ostringstream oss;
            oss << dist_names[d] << "_R" << shapes[sh].R << "_C" << shapes[sh].C;
            cases.push_back(make_case(oss.str(), shapes[sh].R, shapes[sh].C, d, s++));
        }
    }

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<float>& d_x,
    const DeviceBuffer<float>& d_v,
    std::string* error) {
    const std::vector<float> x = d_x.download();
    const std::vector<float> v = d_v.download();

    if (x.size() != hc.x.size() ||
        std::memcmp(x.data(), hc.x.data(), x.size() * sizeof(float)) != 0) {
        if (error) *error = "input x modified";
        return false;
    }
    if (v.size() != hc.v.size() ||
        std::memcmp(v.data(), hc.v.data(), v.size() * sizeof(float)) != 0) {
        if (error) *error = "input v modified";
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
    const int R = hc.run.R;
    const int C = hc.run.C;

    DeviceBuffer<float> d_x;
    DeviceBuffer<float> d_v;

    d_x.allocate(hc.x.size());
    d_v.allocate(hc.v.size());
    d_x.upload(hc.x);
    d_v.upload(hc.v);

    GuardedDeviceBuffer<uint8_t> d_pay;
    GuardedDeviceBuffer<uint8_t> d_sf;
    GuardedDeviceBuffer<float> d_dot;
    GuardedDeviceBuffer<uint64_t> d_digest;
    GuardedDeviceBuffer<int32_t> d_sat;

    d_pay.allocate(mxq_pay_atom_bytes(R, C));
    d_sf.allocate(mxq_sf_atom_bytes(R, C));
    d_dot.allocate((size_t)R);
    d_digest.allocate((size_t)R);
    d_sat.allocate((size_t)R);

    MxqInputs inputs = {};
    inputs.x = d_x.ptr;
    inputs.v = d_v.ptr;

    MxqOutputs outputs = {};
    outputs.pay_atoms = d_pay.ptr;
    outputs.sf_atoms = d_sf.ptr;
    outputs.row_dot = d_dot.ptr;
    outputs.row_digest = d_digest.ptr;
    outputs.sat_count = d_sat.ptr;

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

    if (!check_input_unchanged(hc, d_x, d_v, error)) return false;

    if (!d_pay.check_guards("pay_atoms", error)) return false;
    if (!d_sf.check_guards("sf_atoms", error)) return false;
    if (!d_dot.check_guards("row_dot", error)) return false;
    if (!d_digest.check_guards("row_digest", error)) return false;
    if (!d_sat.check_guards("sat_count", error)) return false;

    const std::vector<uint8_t> h_pay = d_pay.download_data();
    const std::vector<uint8_t> h_sf = d_sf.download_data();
    const std::vector<float> h_dot = d_dot.download_data();
    const std::vector<uint64_t> h_digest = d_digest.download_data();
    const std::vector<int32_t> h_sat = d_sat.download_data();

    MxqHostInputsView host_inputs = {};
    host_inputs.x = hc.x.data();
    host_inputs.v = hc.v.data();

    MxqExpected expected;
    mxq_cpu_oracle(hc.run, host_inputs, &expected);

    MxqHostOutputsView got = {};
    got.pay_atoms = h_pay.data();
    got.sf_atoms = h_sf.data();
    got.row_dot = h_dot.data();
    got.row_digest = h_digest.data();
    got.sat_count = h_sat.data();

    return mxq_check_all_outputs(hc.run, expected, got, error);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int max_R = MXQ_MIN_R;
        int max_C = MXQ_MIN_C;

        for (const HostCase& hc : cases) {
            max_R = std::max(max_R, hc.run.R);
            max_C = std::max(max_C, hc.run.C);
        }

        MxqProblemSpec spec = {};
        spec.abi_version = MXQ_ABI_VERSION;
        spec.max_R = max_R;
        spec.max_C = max_C;
        spec.flags = 0;

        size_t workspace_bytes = solution_workspace_bytes(&spec);
        if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/

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
                    "case %-24s PASS  R=%d C=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.R,
                    hc.run.C,
                    hc.run.distribution_id);
            } else {
                std::printf("case %-24s FAIL  %s\n", hc.name.c_str(), error.c_str());
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
