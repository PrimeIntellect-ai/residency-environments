// ============================================================================
// file: bench_penalty_filter_sample.cu
// ============================================================================

#include "penalty_filter_sample_common.h"

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

static constexpr uint64_t g_state = 0xfbb67ae8584caa73ULL;

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
        uint64_t z = (state += 0x2ca222e1095e1da9ULL);
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
    PfsRunSpec run;
    std::vector<float> logits;
    std::vector<int32_t> history_token;
    std::vector<int32_t> history_len;
    std::vector<float> repetition_penalty;
    std::vector<float> frequency_penalty;
    std::vector<float> presence_penalty;
    std::vector<float> temperature;
    std::vector<float> min_p;
    std::vector<float> uniform_u;
};

struct DeviceCase {
    HostCase host;

    PfsProblemSpec spec;
    void* state = nullptr;
    DeviceBuffer<uint8_t> workspace;

    DeviceBuffer<float> logits;
    DeviceBuffer<int32_t> history_token;
    DeviceBuffer<int32_t> history_len;
    DeviceBuffer<float> rp;
    DeviceBuffer<float> fp;
    DeviceBuffer<float> pp;
    DeviceBuffer<float> temp;
    DeviceBuffer<float> min_p;
    DeviceBuffer<float> uniform;

    DeviceBuffer<int32_t> selected_token;
    DeviceBuffer<int32_t> survivor_count;
    DeviceBuffer<int32_t> packed_cand_token;
    DeviceBuffer<float> packed_cand_prob;

    PfsInputs inputs;
    PfsOutputs outputs;
    size_t workspace_bytes = 0;
};

static float gen_logit(SplitMix64& rng, int distribution_id, int row, int token, int V) {
    switch (distribution_id) {
        case PFS_DIST_UNIFORM:
            return rng.uniform_float(-3.0f, 3.0f);

        case PFS_DIST_PEAKED: {
            const int peak = (row * 1315423911u + 17) % V;
            if (token == peak) return 12.0f;
            if ((token + row * 13) % 257 == 0) return rng.uniform_float(6.0f, 9.0f);
            return rng.uniform_float(-5.0f, 1.5f);
        }

        case PFS_DIST_HEAVY_TAIL: {
            const int x = (token * 1103515245u + row * 12345u) & 0x7fffffff;
            const int bucket = x % 2048;
            return 9.0f - static_cast<float>(std::sqrt(static_cast<double>(bucket)) * 0.35) +
                   rng.uniform_float(-0.20f, 0.20f);
        }

        case PFS_DIST_MANY_TIES: {
            static const float vals[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            return vals[(token + row * 7) % 5];
        }

        default:
            return rng.uniform_float(-2.0f, 2.0f);
    }
}

static HostCase make_case(
    const char* name,
    int B,
    int V,
    int H,
    int distribution_id,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = PFS_ABI_VERSION;
    hc.run.B = B;
    hc.run.V = V;
    hc.run.H = H;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!pfs_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid PfsRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.logits.resize((size_t)B * (size_t)V);
    hc.history_token.resize((size_t)B * (size_t)std::max(H, 1), 0);
    hc.history_len.resize((size_t)B);
    hc.repetition_penalty.resize((size_t)B);
    hc.frequency_penalty.resize((size_t)B);
    hc.presence_penalty.resize((size_t)B);
    hc.temperature.resize((size_t)B);
    hc.min_p.resize((size_t)B);
    hc.uniform_u.resize((size_t)B);

    for (int row = 0; row < B; ++row) {
        for (int token = 0; token < V; ++token) {
            hc.logits[(size_t)row * (size_t)V + (size_t)token] =
                gen_logit(rng, distribution_id, row, token, V);
        }

        const int len = H == 0 ? 0 : rng.uniform_int(0, H);
        hc.history_len[(size_t)row] = len;

        for (int h = 0; h < H; ++h) {
            hc.history_token[(size_t)row * (size_t)std::max(H, 1) + (size_t)h] =
                rng.uniform_int(0, V - 1);
        }

        hc.repetition_penalty[(size_t)row] = rng.uniform_float(1.05f, 1.65f);
        hc.frequency_penalty[(size_t)row] = rng.uniform_float(-0.05f, 0.35f);
        hc.presence_penalty[(size_t)row] = rng.uniform_float(-0.05f, 0.55f);
        hc.temperature[(size_t)row] = rng.uniform_float(0.65f, 1.75f);

        static const float minps[] = {0.0f, 0.05f, 0.2f};
        hc.min_p[(size_t)row] = minps[row % 3];
        hc.uniform_u[(size_t)row] = rng.uniform_float(0.0f, 0.999999f);
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x200000001b3ULL;

    cases.push_back(make_case(
        "uniform_B4096_V1024_H32",
        4096, 1024, 32, PFS_DIST_UNIFORM, s++));

    cases.push_back(make_case(
        "ties_B1024_V4096_H128",
        1024, 4096, 128, PFS_DIST_MANY_TIES, s++));

    cases.push_back(make_case(
        "peaked_B256_V32768_H64",
        256, 32768, 64, PFS_DIST_PEAKED, s++));

    cases.push_back(make_case(
        "heavy_tail_B256_V32768_H256",
        256, 32768, 256, PFS_DIST_HEAVY_TAIL, s++));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc, cudaStream_t stream) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    dc->spec = {};
    dc->spec.abi_version = PFS_ABI_VERSION;
    dc->spec.max_B = hc.run.B;
    dc->spec.max_V = hc.run.V;
    dc->spec.max_H = hc.run.H;
    dc->spec.flags = 0;

    dc->workspace_bytes = solution_workspace_bytes(&dc->spec);
    if (dc->workspace_bytes == 0) {
        throw std::runtime_error("solution_workspace_bytes returned 0");
    }

    CUDA_CHECK(solution_init(&dc->spec, &dc->state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    dc->workspace.allocate(dc->workspace_bytes);

    dc->logits.allocate(hc.logits.size());
    dc->history_token.allocate(hc.history_token.size());
    dc->history_len.allocate(hc.history_len.size());
    dc->rp.allocate(hc.repetition_penalty.size());
    dc->fp.allocate(hc.frequency_penalty.size());
    dc->pp.allocate(hc.presence_penalty.size());
    dc->temp.allocate(hc.temperature.size());
    dc->min_p.allocate(hc.min_p.size());
    dc->uniform.allocate(hc.uniform_u.size());

    dc->logits.upload(hc.logits);
    dc->history_token.upload(hc.history_token);
    dc->history_len.upload(hc.history_len);
    dc->rp.upload(hc.repetition_penalty);
    dc->fp.upload(hc.frequency_penalty);
    dc->pp.upload(hc.presence_penalty);
    dc->temp.upload(hc.temperature);
    dc->min_p.upload(hc.min_p);
    dc->uniform.upload(hc.uniform_u);

    dc->selected_token.allocate((size_t)hc.run.B);
    dc->survivor_count.allocate((size_t)hc.run.B);
    dc->packed_cand_token.allocate((size_t)hc.run.B * (size_t)hc.run.V);
    dc->packed_cand_prob.allocate((size_t)hc.run.B * (size_t)hc.run.V);

    dc->inputs = {};
    dc->inputs.logits = dc->logits.ptr;
    dc->inputs.history_token = dc->history_token.ptr;
    dc->inputs.history_len = dc->history_len.ptr;
    dc->inputs.repetition_penalty = dc->rp.ptr;
    dc->inputs.frequency_penalty = dc->fp.ptr;
    dc->inputs.presence_penalty = dc->pp.ptr;
    dc->inputs.temperature = dc->temp.ptr;
    dc->inputs.min_p = dc->min_p.ptr;
    dc->inputs.uniform_u = dc->uniform.ptr;

    dc->outputs = {};
    dc->outputs.selected_token = dc->selected_token.ptr;
    dc->outputs.survivor_count = dc->survivor_count.ptr;
    dc->outputs.packed_cand_token = dc->packed_cand_token.ptr;
    dc->outputs.packed_cand_prob = dc->packed_cand_prob.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 5;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::vector<HostCase> host_cases = build_bench_cases();
        std::vector<DeviceCase*> cases;
        cases.reserve(host_cases.size());

        for (const HostCase& hc : host_cases) {
            cases.push_back(make_device_case(hc, stream));
            std::printf(
                "bench_case %-28s B=%d V=%d H=%d dist=%d\n",
                hc.name.c_str(),
                hc.run.B,
                hc.run.V,
                hc.run.H,
                hc.run.distribution_id);
        }

        for (int warm = 0; warm < 3; ++warm) {
            for (DeviceCase* dc : cases) {
                CUDA_CHECK(solution_run(
                    dc->state,
                    &dc->host.run,
                    &dc->inputs,
                    &dc->outputs,
                    dc->workspace.ptr,
                    dc->workspace_bytes,
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
                    dc->state,
                    &dc->host.run,
                    &dc->inputs,
                    &dc->outputs,
                    dc->workspace.ptr,
                    dc->workspace_bytes,
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
            solution_destroy(dc->state);
            delete dc;
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
