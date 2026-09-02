// file: test_microscale_requant_chain.cu

#include "microscale_requant_chain_common.h"
#include "microscale_requant_chain_oracle.hpp"

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

static constexpr uint64_t g_state = 0x91b6e04d7a53f28cULL;
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
    MrqRunSpec run;
    std::vector<float> x;
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

// t * 2^p, exact for few-bit t (all uses below are exactly representable).
static float pow2_scale(float t, int p) {
    const uint32_t u = f2u(t);
    const int E = (int)((u >> 23) & 0xffu);
    const uint32_t Mn = u & 0x7fffffu;
    const int e = E - 127 + p;
    if (e >= -126) return u2f(((uint32_t)(e + 127) << 23) | Mn);
    return u2f((0x800000u | Mn) >> (-126 - e));
}

// Values whose |.|/2^t sit at exact LUT midpoints (all E4M3-representable)
// or their neighbours; anchor 256 pins sexp1 = t.
static const float kTies2Vals[17] = {
    16.0f, 48.0f, 80.0f, 128.0f, 208.0f, 320.0f, 448.0f,
    15.0f, 17.0f, 49.0f, 79.0f, 129.0f, 207.0f, 321.0f, 447.0f,
    210.0f, 84.0f
};

// Near-saturation magnitudes (some > 448 -> sat1 counts).
static const float kSatVals[15] = {
    440.0f, 444.0f, 446.0f, 447.0f, 448.0f, 449.0f, 456.0f, 464.0f,
    465.0f, 472.0f, 480.0f, 496.0f, 500.0f, 508.0f, 511.0f
};

// Fill lanes [0, n) of one 32-element scale block.
static void fill_block(
    SplitMix64& rng,
    int distribution_id,
    int b,
    int n,
    float* lane) {
    switch (distribution_id) {
        case MRQ_DIST_SMOOTH: {
            const int t = rng.uniform_int(-20, 20);
            for (int i = 0; i < n; ++i) {
                const int q = rng.uniform_int(-1536, 1536);
                lane[i] = (float)q * pow2f(t - 8);
            }
            break;
        }
        case MRQ_DIST_TIES1: {
            const int t = rng.uniform_int(-30, 30);
            const int anchor = rng.uniform_int(0, n - 1);
            for (int i = 0; i < n; ++i) {
                if (i == anchor) {
                    lane[i] = pow2f(t + 8);   // pins sexp1 = t
                } else if (rng.uniform_int(0, 4) == 0) {
                    // subnormal-region midpoint: odd multiple of 2^-10
                    const int m = 2 * rng.uniform_int(0, 3) + 1;
                    const float v = pow2_scale((float)m, t - 10);
                    lane[i] = rng.uniform_int(0, 1) ? -v : v;
                } else {
                    // normal-grid midpoint: (n8 + 0.5) * 2^(k-3) * 2^t
                    const int k = rng.uniform_int(-6, 8);
                    const int n8 = rng.uniform_int(8, 15);
                    const float v =
                        pow2_scale((float)(2 * n8 + 1), t + k - 4);
                    lane[i] = rng.uniform_int(0, 1) ? -v : v;
                }
            }
            break;
        }
        case MRQ_DIST_SAT: {
            const int t = rng.uniform_int(-25, 25);
            const int anchor = rng.uniform_int(0, n - 1);
            for (int i = 0; i < n; ++i) {
                float v;
                if (i == anchor) {
                    v = pow2f(t + 8);
                } else {
                    v = pow2_scale(kSatVals[rng.uniform_int(0, 14)], t);
                }
                lane[i] = rng.uniform_int(0, 1) ? -v : v;
            }
            break;
        }
        case MRQ_DIST_TIES2: {
            const int t = rng.uniform_int(-30, 30);
            const int anchor = rng.uniform_int(0, n - 1);
            for (int i = 0; i < n; ++i) {
                float v;
                if (i == anchor) {
                    v = pow2f(t + 8);   // u = 256 -> qmax msb pinned
                } else {
                    v = pow2_scale(kTies2Vals[rng.uniform_int(0, 16)], t);
                }
                lane[i] = rng.uniform_int(0, 1) ? -v : v;
            }
            break;
        }
        case MRQ_DIST_ZEROBLK: {
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
        case MRQ_DIST_DENORM: {
            const int mode = rng.uniform_int(0, 2);
            for (int i = 0; i < n; ++i) {
                uint32_t bits;
                if (mode == 0) {
                    bits = (uint32_t)rng.uniform_int(1, 255);          // tiny
                } else if (mode == 1) {
                    bits = (uint32_t)(rng.next_u64() % 0x7fffffull) + 1; // any subnormal
                } else {
                    // mix of subnormals and barely-normal values
                    bits = (rng.uniform_int(0, 1))
                        ? (uint32_t)(rng.next_u64() % 0x7fffffull) + 1
                        : (((uint32_t)rng.uniform_int(1, 24)) << 23) |
                          (uint32_t)(rng.next_u64() & 0x7fffffu);
                }
                bits |= (uint32_t)rng.uniform_int(0, 1) << 31;
                lane[i] = u2f(bits);
            }
            break;
        }
        case MRQ_DIST_NEGZ: {
            for (int i = 0; i < n; ++i) {
                if (rng.uniform_int(0, 9) < 7) {
                    lane[i] = u2f((uint32_t)rng.uniform_int(0, 1) << 31);
                } else {
                    const float v = pow2f(rng.uniform_int(-145, -120));
                    lane[i] = rng.uniform_int(0, 1) ? -v : v;
                }
            }
            break;
        }
        case MRQ_DIST_POW2: {
            const int t = rng.uniform_int(-130, 30);
            for (int i = 0; i < n; ++i) {
                uint32_t bits = f2u(pow2f(t + rng.uniform_int(-4, 4)));
                const int variant = rng.uniform_int(0, 2);
                if (variant == 1) bits += 1;
                else if (variant == 2) bits -= 1;
                if (rng.uniform_int(0, 1)) bits |= 0x80000000u;
                lane[i] = u2f(bits);
            }
            break;
        }
        case MRQ_DIST_RANDBITS:
        default: {
            for (int i = 0; i < n; ++i) {
                uint32_t u = (uint32_t)rng.next_u64();
                // finite only: biased exponent field != 0xFF
                if (((u >> 23) & 0xffu) == 0xffu) {
                    u = (u & ~(0xffu << 23)) | (0xfeu << 23);
                }
                lane[i] = u2f(u);
            }
            break;
        }
    }
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
    hc.run.abi_version = MRQ_ABI_VERSION;
    hc.run.R = R;
    hc.run.C = C;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!mrq_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid MrqRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.x.resize((size_t)R * (size_t)C);

    const int S = mrq_S(C);
    float lane[MRQ_BLOCK_K];

    for (int r = 0; r < R; ++r) {
        for (int b = 0; b < S; ++b) {
            const int k0 = 32 * b;
            const int n = (k0 + 32 <= C) ? 32 : (C - k0);
            fill_block(rng, distribution_id, b, n, lane);
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
        {1024, 513}, {2048, 1024}, {509, 4093}
    };
    static const char* dist_names[9] = {
        "smooth", "ties1", "sat", "ties2", "zeroblk",
        "denorm", "negz", "pow2", "randbits"
    };

    std::vector<HostCase> cases;
    uint64_t s = 0xF30000001b3ULL;

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
    std::string* error) {
    const std::vector<float> x = d_x.download();
    if (x.size() != hc.x.size() ||
        std::memcmp(x.data(), hc.x.data(), x.size() * sizeof(float)) != 0) {
        if (error) *error = "input x modified";
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
    const int S = mrq_S(C);
    const int Kb = mrq_Kb(C);

    DeviceBuffer<float> d_x;
    d_x.allocate(hc.x.size());
    d_x.upload(hc.x);

    GuardedDeviceBuffer<uint8_t> d_e4;
    GuardedDeviceBuffer<uint8_t> d_q4;
    GuardedDeviceBuffer<uint8_t> d_s1;
    GuardedDeviceBuffer<uint8_t> d_s2;
    GuardedDeviceBuffer<int64_t> d_err;
    GuardedDeviceBuffer<uint64_t> d_dig;
    GuardedDeviceBuffer<int32_t> d_sat;

    d_e4.allocate((size_t)R * (size_t)C);
    d_q4.allocate((size_t)R * (size_t)Kb);
    d_s1.allocate((size_t)R * (size_t)S);
    d_s2.allocate((size_t)R * (size_t)S);
    d_err.allocate((size_t)R);
    d_dig.allocate((size_t)R);
    d_sat.allocate((size_t)R);

    MrqInputs inputs = {};
    inputs.x = d_x.ptr;

    MrqOutputs outputs = {};
    outputs.e4m3_codes = d_e4.ptr;
    outputs.q4_packed = d_q4.ptr;
    outputs.sf1 = d_s1.ptr;
    outputs.sf2 = d_s2.ptr;
    outputs.row_err = d_err.ptr;
    outputs.row_digest = d_dig.ptr;
    outputs.sat1_count = d_sat.ptr;

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

    if (!check_input_unchanged(hc, d_x, error)) return false;

    if (!d_e4.check_guards("e4m3_codes", error)) return false;
    if (!d_q4.check_guards("q4_packed", error)) return false;
    if (!d_s1.check_guards("sf1", error)) return false;
    if (!d_s2.check_guards("sf2", error)) return false;
    if (!d_err.check_guards("row_err", error)) return false;
    if (!d_dig.check_guards("row_digest", error)) return false;
    if (!d_sat.check_guards("sat1_count", error)) return false;

    const std::vector<uint8_t> h_e4 = d_e4.download_data();
    const std::vector<uint8_t> h_q4 = d_q4.download_data();
    const std::vector<uint8_t> h_s1 = d_s1.download_data();
    const std::vector<uint8_t> h_s2 = d_s2.download_data();
    const std::vector<int64_t> h_err = d_err.download_data();
    const std::vector<uint64_t> h_dig = d_dig.download_data();
    const std::vector<int32_t> h_sat = d_sat.download_data();

    MrqHostInputsView host_inputs = {};
    host_inputs.x = hc.x.data();

    MrqExpected expected;
    mrq_cpu_oracle(hc.run, host_inputs, &expected);

    MrqHostOutputsView got = {};
    got.e4m3_codes = h_e4.data();
    got.q4_packed = h_q4.data();
    got.sf1 = h_s1.data();
    got.sf2 = h_s2.data();
    got.row_err = h_err.data();
    got.row_digest = h_dig.data();
    got.sat1_count = h_sat.data();

    return mrq_check_all_outputs(hc.run, expected, got, error);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int max_R = MRQ_MIN_R;
        int max_C = MRQ_MIN_C;

        for (const HostCase& hc : cases) {
            max_R = std::max(max_R, hc.run.R);
            max_C = std::max(max_C, hc.run.C);
        }

        MrqProblemSpec spec = {};
        spec.abi_version = MRQ_ABI_VERSION;
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
