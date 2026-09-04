// file: bench_fused_layernorm_quant_pipeline.cu

#include "fused_layernorm_quant_pipeline_common.h"

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

static constexpr uint64_t g_state = 0x6f2d58c4a19e73b5ULL;

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

struct HostCase {
    std::string name;
    FlqpRunSpec run;
    std::vector<float> x;
    std::vector<float> weight;
    std::vector<float> bias;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<float> x;
    DeviceBuffer<float> weight;
    DeviceBuffer<float> bias;

    DeviceBuffer<int8_t> q_int8;
    DeviceBuffer<float> scale;
    DeviceBuffer<float> dequant;
    DeviceBuffer<int64_t> code_sum;

    FlqpInputs inputs;
    FlqpOutputs outputs;
};

static float exact_frac(int num, int denom) {
    return static_cast<float>(num) / static_cast<float>(denom);
}

static float make_weight(int d, int distribution_id) {
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
    if (distribution_id == FLQP_DIST_OUTLIERS) {
        if ((row % 97) == 0 && d == ((row * 13) % D)) return 512.0f;
        if ((row % 131) == 0 && d == ((row * 17 + 3) % D)) return -512.0f;

        static const float vals[] = {
            -4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f
        };
        return vals[rng.uniform_int(0, 8)];
    }

    if (distribution_id == FLQP_DIST_ZEROISH) {
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

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x200000001b3ULL;

    cases.push_back(make_case(
        "bench_N65536_D256_uniform",
        65536,
        256,
        FLQP_DIST_UNIFORM,
        1.0e-5f,
        s++));

    cases.push_back(make_case(
        "bench_N16384_D1024_outliers",
        16384,
        1024,
        FLQP_DIST_OUTLIERS,
        1.0e-5f,
        s++));

    cases.push_back(make_case(
        "bench_N4096_D4096_zeroish",
        4096,
        4096,
        FLQP_DIST_ZEROISH,
        1.0e-7f,
        s++));

    cases.push_back(make_case(
        "bench_N131072_D256_outliers",
        131072,
        256,
        FLQP_DIST_OUTLIERS,
        1.0e-5f,
        s++));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int D = hc.run.D;

    dc->x.allocate(hc.x.size());
    dc->weight.allocate(hc.weight.size());
    dc->bias.allocate(hc.bias.size());

    dc->x.upload(hc.x);
    dc->weight.upload(hc.weight);
    dc->bias.upload(hc.bias);

    dc->q_int8.allocate((size_t)N * (size_t)D);
    dc->scale.allocate((size_t)N);
    dc->dequant.allocate((size_t)N * (size_t)D);
    dc->code_sum.allocate((size_t)N);

    dc->inputs = {};
    dc->inputs.x = dc->x.ptr;
    dc->inputs.weight = dc->weight.ptr;
    dc->inputs.bias = dc->bias.ptr;

    dc->outputs = {};
    dc->outputs.q_int8 = dc->q_int8.ptr;
    dc->outputs.scale = dc->scale.ptr;
    dc->outputs.dequant = dc->dequant.ptr;
    dc->outputs.code_sum = dc->code_sum.ptr;

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

        int max_N = FLQP_MIN_N;
        int max_D = 256;

        for (const HostCase& hc : host_cases) {
            max_N = std::max(max_N, hc.run.N);
            max_D = std::max(max_D, hc.run.D);
        }

        FlqpProblemSpec spec = {};
        spec.abi_version = FLQP_ABI_VERSION;
        spec.max_N = max_N;
        spec.max_D = max_D;
        spec.flags = 0;

        const size_t workspace_bytes = solution_workspace_bytes(&spec);
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

        std::vector<DeviceCase*> cases;
        cases.reserve(host_cases.size());

        for (const HostCase& hc : host_cases) {
            cases.push_back(make_device_case(hc));
            std::printf(
                "bench_case %-32s N=%d D=%d eps=%.1e dist=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.D,
                static_cast<double>(hc.run.eps),
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
