// ============================================================================
// file: bench_group_limited_topk.cu
// ============================================================================

#include "group_limited_topk_common.h"

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

static constexpr uint64_t g_state = 0x3bd39e10cb0ef593ULL;

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

    float uniform_float(float lo, float hi) {
        const float u = static_cast<float>((next_u64() >> 11) * 0x1.0p-53);
        return lo + (hi - lo) * u;
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
};

struct HostCase {
    std::string name;
    GltRunSpec run;
    std::vector<float> score;
    std::vector<float> uniform_u;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<float> score;
    DeviceBuffer<float> uniform_u;

    DeviceBuffer<int32_t> selected_group_ids;
    DeviceBuffer<int32_t> survivor_expert_ids;
    DeviceBuffer<int32_t> survivor_count;
    DeviceBuffer<float> weights;
    DeviceBuffer<float> group_scores;

    GltInputs inputs;
    GltOutputs outputs;
};

static float gen_score(
    SplitMix64& rng,
    int distribution_id,
    int row,
    int expert,
    int V,
    int G) {
    const int group_size = V / G;
    const int group = expert / group_size;
    const int off = expert - group * group_size;

    switch (distribution_id) {
        case GLT_DIST_UNIFORM:
            return rng.uniform_float(-3.0f, 3.0f);

        case GLT_DIST_PEAKED: {
            const int peak = (row * 1315423911u + 17) % V;
            if (expert == peak) return 16.0f;
            if ((expert + row * 13) % 257 == 0) {
                return rng.uniform_float(7.0f, 12.0f);
            }
            return rng.uniform_float(-4.0f, 2.0f);
        }

        case GLT_DIST_GROUP_SKEW: {
            const int hot0 = row % G;
            const int hot1 = (hot0 + 3) % G;
            const int hot2 = (hot0 + 7) % G;

            float base = rng.uniform_float(-2.0f, 2.0f);
            if (group == hot0) {
                base += 8.0f;
            } else if (group == hot1) {
                base += 5.0f;
            } else if (group == hot2) {
                base += 3.0f;
            }

            base += 0.001f * static_cast<float>(group_size - off);
            return base;
        }

        case GLT_DIST_MANY_TIES: {
            static const float vals[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            return vals[(expert + row * 7) % 5];
        }

        default:
            return rng.uniform_float(-2.0f, 2.0f);
    }
}

static HostCase make_case(
    const char* name,
    int B,
    int V,
    int G,
    int group_k,
    int n_groups,
    int final_k,
    int distribution_id,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = GLT_ABI_VERSION;
    hc.run.B = B;
    hc.run.V = V;
    hc.run.G = G;
    hc.run.group_k = group_k;
    hc.run.n_groups = n_groups;
    hc.run.final_k = final_k;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!glt_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid GltRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.score.resize((size_t)B * (size_t)V);
    hc.uniform_u.resize((size_t)B);

    for (int row = 0; row < B; ++row) {
        hc.uniform_u[(size_t)row] = rng.uniform_float(0.0f, 0.999999f);

        for (int expert = 0; expert < V; ++expert) {
            hc.score[(size_t)row * (size_t)V + (size_t)expert] =
                gen_score(rng, distribution_id, row, expert, V, G);
        }
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x200000001b3ULL;

    cases.push_back(make_case(
        "large_B4096_V256_uniform",
        4096, 256, 16, 4, 4, 64,
        GLT_DIST_UNIFORM,
        s++));

    cases.push_back(make_case(
        "largeV_B256_V4096_group_skew",
        256, 4096, 32, 4, 8, 64,
        GLT_DIST_GROUP_SKEW,
        s++));

    cases.push_back(make_case(
        "mid_B512_V1024_peaked",
        512, 1024, 16, 4, 4, 16,
        GLT_DIST_PEAKED,
        s++));

    cases.push_back(make_case(
        "ties_B1024_V1024_G32",
        1024, 1024, 32, 2, 8, 16,
        GLT_DIST_MANY_TIES,
        s++));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int B = hc.run.B;
    const int V = hc.run.V;
    const int G = hc.run.G;
    const int n_groups = hc.run.n_groups;
    const int final_k = hc.run.final_k;

    dc->score.allocate(hc.score.size());
    dc->uniform_u.allocate(hc.uniform_u.size());

    dc->score.upload(hc.score);
    dc->uniform_u.upload(hc.uniform_u);

    dc->selected_group_ids.allocate((size_t)B * (size_t)n_groups);
    dc->survivor_expert_ids.allocate((size_t)B * (size_t)final_k);
    dc->survivor_count.allocate((size_t)B);
    dc->weights.allocate((size_t)B * (size_t)final_k);
    dc->group_scores.allocate((size_t)B * (size_t)G);

    dc->inputs = {};
    dc->inputs.score = dc->score.ptr;
    dc->inputs.uniform_u = dc->uniform_u.ptr;

    dc->outputs = {};
    dc->outputs.selected_group_ids = dc->selected_group_ids.ptr;
    dc->outputs.survivor_expert_ids = dc->survivor_expert_ids.ptr;
    dc->outputs.survivor_count = dc->survivor_count.ptr;
    dc->outputs.weights = dc->weights.ptr;
    dc->outputs.group_scores = dc->group_scores.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        GltProblemSpec spec = {};
        spec.abi_version = GLT_ABI_VERSION;
        spec.max_B = GLT_MAX_B;
        spec.max_V = GLT_MAX_V;
        spec.max_G = GLT_MAX_G;
        spec.max_n_groups = GLT_MAX_N_GROUPS;
        spec.max_final_k = GLT_MAX_FINAL_K;
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

        std::vector<HostCase> host_cases = build_bench_cases();
        std::vector<DeviceCase*> cases;
        cases.reserve(host_cases.size());

        for (const HostCase& hc : host_cases) {
            cases.push_back(make_device_case(hc));
            std::printf(
                "bench_case %-32s B=%d V=%d G=%d gk=%d ng=%d fk=%d dist=%d\n",
                hc.name.c_str(),
                hc.run.B,
                hc.run.V,
                hc.run.G,
                hc.run.group_k,
                hc.run.n_groups,
                hc.run.final_k,
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
