// file: bench_mxfp4_atom_quant_dot.cu

#include "mxfp4_atom_quant_dot_common.h"

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

static constexpr uint64_t g_state = 0x51c9a3e7d2b84f61ULL;

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
};

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

static float pow2f(int p) {
    if (p >= -126) return u2f((uint32_t)(p + 127) << 23);
    return u2f(0x800000u >> (-126 - p));
}

// Exact t * 2^p for few-bit t (see contract).
static float pow2_scale(float t, int p) {
    const uint32_t u = f2u(t);
    const int E = (int)((u >> 23) & 0xffu);
    const uint32_t M = u & 0x7fffffu;
    const int e = E - 127 + p;
    if (e >= -126) return u2f(((uint32_t)(e + 127) << 23) | M);
    return u2f((0x800000u | M) >> (-126 - e));
}

static const float kMidpoints[7] = {
    0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f
};

static const float kSatFracs[6] = {
    1.0f, 1.25f, 1.5f, 1.625f, 1.75f, 1.875f
};

static void fill_block(
    SplitMix64& rng,
    int distribution_id,
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
                    lane[i] = pow2f(t + 2);
                } else {
                    const float m = kMidpoints[rng.uniform_int(0, 6)];
                    const float val = pow2_scale(m, t);
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
                    val = pow2_scale(1.75f, t);
                } else if (rng.uniform_int(0, 1)) {
                    val = pow2_scale(kSatFracs[rng.uniform_int(0, 5)], t);
                } else {
                    val = pow2_scale(1.25f, t - rng.uniform_int(1, 6));
                }
                lane[i] = rng.uniform_int(0, 1) ? -val : val;
            }
            break;
        }
        case MXQ_DIST_RANDBITS:
        default: {
            for (int i = 0; i < n; ++i) {
                uint32_t u = (uint32_t)rng.next_u64();
                // Cap the biased exponent at 0xF0 (contract domain guarantee).
                while (((u >> 23) & 0xffu) > 0xf0u) u -= (0x10u << 23);
                lane[i] = u2f(u);
            }
            break;
        }
    }
    (void)b;
}

struct HostCase {
    std::string name;
    MxqRunSpec run;
    std::vector<float> x;
    std::vector<float> v;
};

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
            fill_block(rng, distribution_id, b, n, lane);
            for (int i = 0; i < n; ++i) {
                hc.x[(size_t)r * (size_t)C + (size_t)(k0 + i)] = lane[i];
            }
        }
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0xA00000001b3ULL;

    cases.push_back(make_case("bench_R8192_C4096_smooth", 8192, 4096, MXQ_DIST_SMOOTH, s++));
    cases.push_back(make_case("bench_R16384_C2048_randbits", 16384, 2048, MXQ_DIST_RANDBITS, s++));
    cases.push_back(make_case("bench_R4096_C6143_saturate", 4096, 6143, MXQ_DIST_SATURATE, s++));
    cases.push_back(make_case("bench_R2048_C1024_ties", 2048, 1024, MXQ_DIST_TIES, s++));

    return cases;
}

struct DeviceCase {
    HostCase host;

    DeviceBuffer<float> x;
    DeviceBuffer<float> v;

    DeviceBuffer<uint8_t> pay_atoms;
    DeviceBuffer<uint8_t> sf_atoms;
    DeviceBuffer<float> row_dot;
    DeviceBuffer<uint64_t> row_digest;
    DeviceBuffer<int32_t> sat_count;

    MxqInputs inputs;
    MxqOutputs outputs;
};

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int R = hc.run.R;
    const int C = hc.run.C;

    dc->x.allocate(hc.x.size());
    dc->v.allocate(hc.v.size());
    dc->x.upload(hc.x);
    dc->v.upload(hc.v);

    dc->pay_atoms.allocate(mxq_pay_atom_bytes(R, C));
    dc->sf_atoms.allocate(mxq_sf_atom_bytes(R, C));
    dc->row_dot.allocate((size_t)R);
    dc->row_digest.allocate((size_t)R);
    dc->sat_count.allocate((size_t)R);

    dc->inputs = {};
    dc->inputs.x = dc->x.ptr;
    dc->inputs.v = dc->v.ptr;

    dc->outputs = {};
    dc->outputs.pay_atoms = dc->pay_atoms.ptr;
    dc->outputs.sf_atoms = dc->sf_atoms.ptr;
    dc->outputs.row_dot = dc->row_dot.ptr;
    dc->outputs.row_digest = dc->row_digest.ptr;
    dc->outputs.sat_count = dc->sat_count.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        std::vector<HostCase> host_cases = build_bench_cases();

        int max_R = MXQ_MIN_R;
        int max_C = MXQ_MIN_C;

        for (const HostCase& hc : host_cases) {
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

        std::vector<DeviceCase*> cases;
        cases.reserve(host_cases.size());

        for (const HostCase& hc : host_cases) {
            cases.push_back(make_device_case(hc));
            std::printf(
                "bench_case %-28s R=%d C=%d dist=%d\n",
                hc.name.c_str(),
                hc.run.R,
                hc.run.C,
                hc.run.distribution_id);
        }

        for (int warm = 0; warm < 5; ++warm) {
            for (DeviceCase* dc : cases) {
                CUDA_CHECK(solution_run(
                    state,
                    &dc->host.run,
                    &dc->inputs,
                    &dc->outputs,
                    workspace.ptr,
                    workspace_bytes,
                    stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));

        for (int iter = 0; iter < iters; ++iter) {
            for (DeviceCase* dc : cases) {
                CUDA_CHECK(solution_run(
                    state,
                    &dc->host.run,
                    &dc->inputs,
                    &dc->outputs,
                    workspace.ptr,
                    workspace_bytes,
                    stream));
            }
        }

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        const double calls = static_cast<double>(iters) * static_cast<double>(cases.size());
        const double avg_ms = static_cast<double>(elapsed_ms) / calls;

        std::printf("avg_ms=%.6f\n", avg_ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (DeviceCase* dc : cases) {
            delete dc;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));

        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
