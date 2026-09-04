// file: bench_microscale_requant_chain.cu

#include "microscale_requant_chain_common.h"

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

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0xA30000001b3ULL;

    cases.push_back(make_case("bench_R8192_C4096_smooth", 8192, 4096, MRQ_DIST_SMOOTH, s++));
    cases.push_back(make_case("bench_R8192_C4096_randbits", 8192, 4096, MRQ_DIST_RANDBITS, s++));
    cases.push_back(make_case("bench_R4096_C4093_sat", 4096, 4093, MRQ_DIST_SAT, s++));
    cases.push_back(make_case("bench_R2048_C1024_ties2", 2048, 1024, MRQ_DIST_TIES2, s++));

    return cases;
}

struct DeviceCase {
    HostCase host;

    DeviceBuffer<float> x;

    DeviceBuffer<uint8_t> e4m3_codes;
    DeviceBuffer<uint8_t> q4_packed;
    DeviceBuffer<uint8_t> sf1;
    DeviceBuffer<uint8_t> sf2;
    DeviceBuffer<int64_t> row_err;
    DeviceBuffer<uint64_t> row_digest;
    DeviceBuffer<int32_t> sat1_count;

    MrqInputs inputs;
    MrqOutputs outputs;
};

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int R = hc.run.R;
    const int C = hc.run.C;
    const int S = mrq_S(C);
    const int Kb = mrq_Kb(C);

    dc->x.allocate(hc.x.size());
    dc->x.upload(hc.x);

    dc->e4m3_codes.allocate((size_t)R * (size_t)C);
    dc->q4_packed.allocate((size_t)R * (size_t)Kb);
    dc->sf1.allocate((size_t)R * (size_t)S);
    dc->sf2.allocate((size_t)R * (size_t)S);
    dc->row_err.allocate((size_t)R);
    dc->row_digest.allocate((size_t)R);
    dc->sat1_count.allocate((size_t)R);

    dc->inputs = {};
    dc->inputs.x = dc->x.ptr;

    dc->outputs = {};
    dc->outputs.e4m3_codes = dc->e4m3_codes.ptr;
    dc->outputs.q4_packed = dc->q4_packed.ptr;
    dc->outputs.sf1 = dc->sf1.ptr;
    dc->outputs.sf2 = dc->sf2.ptr;
    dc->outputs.row_err = dc->row_err.ptr;
    dc->outputs.row_digest = dc->row_digest.ptr;
    dc->outputs.sat1_count = dc->sat1_count.ptr;

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

        int max_R = MRQ_MIN_R;
        int max_C = MRQ_MIN_C;

        for (const HostCase& hc : host_cases) {
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
