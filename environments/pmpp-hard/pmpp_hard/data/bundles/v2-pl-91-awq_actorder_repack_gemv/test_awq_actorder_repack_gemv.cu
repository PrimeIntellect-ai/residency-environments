// file: test_awq_actorder_repack_gemv.cu

#include "awq_actorder_repack_gemv_common.h"
#include "awq_actorder_repack_gemv_oracle.hpp"

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

static constexpr uint64_t g_state = 0x7e91c44ab0d25f13ULL;
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
    AwqRunSpec run;
    std::vector<uint32_t> qweight;
    std::vector<uint32_t> qzeros;
    std::vector<float> scales;
    std::vector<int32_t> g_idx;
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

// Random finite fp32 with biased exponent field <= 0x9F (|v| < 2^33),
// per the contract domain guarantee. Subnormals and +-0.0 included.
static float capped_randbits(SplitMix64& rng) {
    uint32_t u = (uint32_t)rng.next_u64();
    uint32_t e = (u >> 23) & 0xffu;
    if (e > 0x9fu) {
        e %= 0xa0u;
        u = (u & ~(0xffu << 23)) | (e << 23);
    }
    return u2f(u);
}

static void gen_g_idx(
    SplitMix64& rng, int distribution_id, int K, int G, int32_t* g_idx) {
    switch (distribution_id) {
        case AWQ_DIST_REVERSED: {
            for (int k = 0; k < K; ++k) {
                g_idx[k] = (G - 1) - (int)(((long long)k * G) / K);
            }
            break;
        }
        case AWQ_DIST_SKEWED: {
            for (int k = 0; k < K; ++k) {
                const double u =
                    (double)(rng.next_u64() >> 11) * (1.0 / 9007199254740992.0);
                int g = (int)((double)G * u * u);
                if (g > G - 1) g = G - 1;
                g_idx[k] = g;
            }
            break;
        }
        case AWQ_DIST_ALLSAME: {
            for (int k = 0; k < K; ++k) g_idx[k] = G / 2;
            break;
        }
        default: {
            for (int k = 0; k < K; ++k) {
                g_idx[k] = (int)(rng.next_u64() % (uint64_t)G);
            }
            break;
        }
    }
}

static float gen_scale(SplitMix64& rng, int distribution_id) {
    const uint32_t sign = ((uint32_t)rng.uniform_int(0, 1)) << 31;
    switch (distribution_id) {
        case AWQ_DIST_ZEROMATCH: {
            if (rng.uniform_int(0, 9) < 3) return u2f(sign);  // +-0.0
            break;
        }
        case AWQ_DIST_ZEROSCALE: {
            if (rng.uniform_int(0, 1)) return u2f(sign);  // +-0.0
            break;
        }
        case AWQ_DIST_DENORM: {
            const uint32_t m = (uint32_t)rng.uniform_int(1, 0x7fffff);
            return u2f(sign | m);  // subnormal
        }
        case AWQ_DIST_POW2: {
            const int t = rng.uniform_int(-30, 30);
            return u2f(sign | f2u(pow2f(t)));
        }
        case AWQ_DIST_RANDBITS:
            return capped_randbits(rng);
        default:
            break;
    }
    const uint32_t e = (uint32_t)rng.uniform_int(121, 130);
    const uint32_t m = (uint32_t)(rng.next_u64() & 0x7fffffu);
    return u2f(sign | (e << 23) | m);
}

static float gen_x(SplitMix64& rng, int distribution_id) {
    switch (distribution_id) {
        case AWQ_DIST_ZEROMATCH: {
            if (rng.uniform_int(0, 1)) {
                return u2f(((uint32_t)rng.uniform_int(0, 1)) << 31);  // +-0.0
            }
            break;
        }
        case AWQ_DIST_DENORM: {
            const uint32_t sign = ((uint32_t)rng.uniform_int(0, 1)) << 31;
            const uint32_t e = (uint32_t)rng.uniform_int(100, 120);
            const uint32_t m = (uint32_t)(rng.next_u64() & 0x7fffffu);
            return u2f(sign | (e << 23) | m);
        }
        case AWQ_DIST_POW2: {
            if (rng.uniform_int(0, 7) == 0) {
                return u2f(((uint32_t)rng.uniform_int(0, 1)) << 31);  // +-0.0
            }
            const uint32_t sign = ((uint32_t)rng.uniform_int(0, 1)) << 31;
            return u2f(sign | f2u(pow2f(rng.uniform_int(-20, 20))));
        }
        case AWQ_DIST_RANDBITS:
            return capped_randbits(rng);
        default:
            break;
    }
    const int q = rng.uniform_int(-1536, 1536);
    const int t = rng.uniform_int(-10, 2);
    return (float)q * pow2f(t - 8);
}

static int gen_q_nibble(SplitMix64& rng, int distribution_id) {
    if (distribution_id == AWQ_DIST_POW2) {
        const int sel = rng.uniform_int(0, 3);
        if (sel == 0) return 15;
        if (sel == 1) return 0;
    }
    return rng.uniform_int(0, 15);
}

static int gen_z_nibble(SplitMix64& rng, int distribution_id) {
    if (distribution_id == AWQ_DIST_POW2) {
        return rng.uniform_int(0, 1) ? 15 : 0;
    }
    return rng.uniform_int(0, 15);
}

static HostCase make_case(
    const std::string& name,
    int K,
    int N,
    int G,
    int distribution_id,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = AWQ_ABI_VERSION;
    hc.run.K = K;
    hc.run.N = N;
    hc.run.G = G;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!awq_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid AwqRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    const int Nw = awq_Nw(N);

    hc.g_idx.resize((size_t)K);
    gen_g_idx(rng, distribution_id, K, G, hc.g_idx.data());

    std::vector<uint8_t> zlog((size_t)G * (size_t)N);
    for (int g = 0; g < G; ++g) {
        for (int n = 0; n < N; ++n) {
            zlog[(size_t)g * N + n] = (uint8_t)gen_z_nibble(rng, distribution_id);
        }
    }

    std::vector<uint8_t> qlog((size_t)K * (size_t)N);
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            uint8_t q;
            if (distribution_id == AWQ_DIST_ZEROMATCH) {
                q = zlog[(size_t)hc.g_idx[(size_t)k] * N + n];
            } else {
                q = (uint8_t)gen_q_nibble(rng, distribution_id);
            }
            qlog[(size_t)k * N + n] = q;
        }
    }

    hc.scales.resize((size_t)G * (size_t)N);
    for (size_t i = 0; i < hc.scales.size(); ++i) {
        hc.scales[i] = gen_scale(rng, distribution_id);
    }

    hc.x.resize((size_t)K);
    for (int k = 0; k < K; ++k) hc.x[(size_t)k] = gen_x(rng, distribution_id);

    // Pack (source padding nibbles are 0 per the contract guarantee).
    hc.qweight.assign((size_t)K * (size_t)Nw, 0u);
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < Nw; ++c) {
            uint32_t word = 0;
            for (int t = 0; t < 8; ++t) {
                const int n = 8 * c + t;
                const uint32_t nib = (n < N) ? qlog[(size_t)k * N + n] : 0u;
                word |= nib << (4 * awq_wlane(t));
            }
            hc.qweight[(size_t)k * Nw + c] = word;
        }
    }
    hc.qzeros.assign((size_t)G * (size_t)Nw, 0u);
    for (int g = 0; g < G; ++g) {
        for (int c = 0; c < Nw; ++c) {
            uint32_t word = 0;
            for (int t = 0; t < 8; ++t) {
                const int n = 8 * c + t;
                const uint32_t nib = (n < N) ? zlog[(size_t)g * N + n] : 0u;
                word |= nib << (4 * awq_zlane(t));
            }
            hc.qzeros[(size_t)g * Nw + c] = word;
        }
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    struct Shape { int K, N, G; };
    static const Shape shapes[6] = {
        {259, 65, 5}, {512, 127, 32}, {1023, 255, 7},
        {2048, 512, 96}, {3072, 1027, 128}, {8191, 320, 512}
    };
    static const char* dist_names[9] = {
        "balanced", "reversed", "skewed", "allsame", "zeromatch",
        "zeroscale", "denorm", "pow2", "randbits"
    };

    std::vector<HostCase> cases;
    uint64_t s = 0xB10000001b3ULL;

    for (int d = 0; d < 9; ++d) {
        for (int sh = 0; sh < 6; ++sh) {
            std::ostringstream oss;
            oss << dist_names[d] << "_K" << shapes[sh].K
                << "_N" << shapes[sh].N << "_G" << shapes[sh].G;
            cases.push_back(make_case(
                oss.str(), shapes[sh].K, shapes[sh].N, shapes[sh].G, d, s++));
        }
    }

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<uint32_t>& d_qw,
    const DeviceBuffer<uint32_t>& d_qz,
    const DeviceBuffer<float>& d_sc,
    const DeviceBuffer<int32_t>& d_gi,
    const DeviceBuffer<float>& d_x,
    std::string* error) {
    const std::vector<uint32_t> qw = d_qw.download();
    const std::vector<uint32_t> qz = d_qz.download();
    const std::vector<float> sc = d_sc.download();
    const std::vector<int32_t> gi = d_gi.download();
    const std::vector<float> x = d_x.download();

    if (qw.size() != hc.qweight.size() ||
        std::memcmp(qw.data(), hc.qweight.data(), qw.size() * 4) != 0) {
        if (error) *error = "input qweight modified";
        return false;
    }
    if (qz.size() != hc.qzeros.size() ||
        std::memcmp(qz.data(), hc.qzeros.data(), qz.size() * 4) != 0) {
        if (error) *error = "input qzeros modified";
        return false;
    }
    if (sc.size() != hc.scales.size() ||
        std::memcmp(sc.data(), hc.scales.data(), sc.size() * 4) != 0) {
        if (error) *error = "input scales modified";
        return false;
    }
    if (gi.size() != hc.g_idx.size() ||
        std::memcmp(gi.data(), hc.g_idx.data(), gi.size() * 4) != 0) {
        if (error) *error = "input g_idx modified";
        return false;
    }
    if (x.size() != hc.x.size() ||
        std::memcmp(x.data(), hc.x.data(), x.size() * 4) != 0) {
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
    const int K = hc.run.K;
    const int N = hc.run.N;

    DeviceBuffer<uint32_t> d_qw;
    DeviceBuffer<uint32_t> d_qz;
    DeviceBuffer<float> d_sc;
    DeviceBuffer<int32_t> d_gi;
    DeviceBuffer<float> d_x;

    d_qw.allocate(hc.qweight.size());
    d_qz.allocate(hc.qzeros.size());
    d_sc.allocate(hc.scales.size());
    d_gi.allocate(hc.g_idx.size());
    d_x.allocate(hc.x.size());
    d_qw.upload(hc.qweight);
    d_qz.upload(hc.qzeros);
    d_sc.upload(hc.scales);
    d_gi.upload(hc.g_idx);
    d_x.upload(hc.x);

    GuardedDeviceBuffer<uint8_t> d_rq;
    GuardedDeviceBuffer<float> d_dot;
    GuardedDeviceBuffer<uint64_t> d_digest;
    GuardedDeviceBuffer<int32_t> d_zsum;
    GuardedDeviceBuffer<int32_t> d_perm;

    d_rq.allocate(awq_rq_bytes(K, N));
    d_dot.allocate((size_t)N);
    d_digest.allocate((size_t)N);
    d_zsum.allocate((size_t)N);
    d_perm.allocate((size_t)K);

    AwqInputs inputs = {};
    inputs.qweight = d_qw.ptr;
    inputs.qzeros = d_qz.ptr;
    inputs.scales = d_sc.ptr;
    inputs.g_idx = d_gi.ptr;
    inputs.x = d_x.ptr;

    AwqOutputs outputs = {};
    outputs.rq_atoms = d_rq.ptr;
    outputs.col_dot = d_dot.ptr;
    outputs.col_digest = d_digest.ptr;
    outputs.col_zsum = d_zsum.ptr;
    outputs.perm = d_perm.ptr;

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

    if (!check_input_unchanged(hc, d_qw, d_qz, d_sc, d_gi, d_x, error)) return false;

    if (!d_rq.check_guards("rq_atoms", error)) return false;
    if (!d_dot.check_guards("col_dot", error)) return false;
    if (!d_digest.check_guards("col_digest", error)) return false;
    if (!d_zsum.check_guards("col_zsum", error)) return false;
    if (!d_perm.check_guards("perm", error)) return false;

    const std::vector<uint8_t> h_rq = d_rq.download_data();
    const std::vector<float> h_dot = d_dot.download_data();
    const std::vector<uint64_t> h_digest = d_digest.download_data();
    const std::vector<int32_t> h_zsum = d_zsum.download_data();
    const std::vector<int32_t> h_perm = d_perm.download_data();

    AwqHostInputsView host_inputs = {};
    host_inputs.qweight = hc.qweight.data();
    host_inputs.qzeros = hc.qzeros.data();
    host_inputs.scales = hc.scales.data();
    host_inputs.g_idx = hc.g_idx.data();
    host_inputs.x = hc.x.data();

    AwqExpected expected;
    awq_cpu_oracle(hc.run, host_inputs, &expected);

    AwqHostOutputsView got = {};
    got.rq_atoms = h_rq.data();
    got.col_dot = h_dot.data();
    got.col_digest = h_digest.data();
    got.col_zsum = h_zsum.data();
    got.perm = h_perm.data();

    return awq_check_all_outputs(hc.run, expected, got, error);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int max_K = AWQ_MIN_K;
        int max_N = AWQ_MIN_N;
        int max_G = AWQ_MIN_G;

        for (const HostCase& hc : cases) {
            max_K = std::max(max_K, hc.run.K);
            max_N = std::max(max_N, hc.run.N);
            max_G = std::max(max_G, hc.run.G);
        }

        AwqProblemSpec spec = {};
        spec.abi_version = AWQ_ABI_VERSION;
        spec.max_K = max_K;
        spec.max_N = max_N;
        spec.max_G = max_G;
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
                    "case %-26s PASS  K=%d N=%d G=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.K,
                    hc.run.N,
                    hc.run.G,
                    hc.run.distribution_id);
            } else {
                std::printf("case %-26s FAIL  %s\n", hc.name.c_str(), error.c_str());
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
