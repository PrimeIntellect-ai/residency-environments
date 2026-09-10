// ============================================================================
// file: bench_route_compact_reduce.cu
// ============================================================================

#include "route_compact_reduce_common.h"

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

static constexpr uint64_t g_state = 0x6a09e667f3bcc909ULL;

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

    double uniform01() {
        return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
    }

    bool chance(double p) {
        return uniform01() < p;
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
    RcrRunSpec run;
    std::vector<int16_t> x;
    std::vector<int16_t> expert;
    std::vector<int16_t> weight;
    std::vector<uint8_t> valid;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<int16_t> x;
    DeviceBuffer<int16_t> expert;
    DeviceBuffer<int16_t> weight;
    DeviceBuffer<uint8_t> valid;

    DeviceBuffer<int32_t> counts;
    DeviceBuffer<int32_t> offsets;
    DeviceBuffer<int32_t> packed_token;
    DeviceBuffer<int32_t> packed_weight;
    DeviceBuffer<int64_t> sum;
    DeviceBuffer<int32_t> argmax_abs;

    RcrInputs inputs;
    RcrOutputs outputs;
};

static int sample_zipf_1_2(SplitMix64& rng, int limit_e) {
    double total = 0.0;
    for (int i = 0; i < limit_e; ++i) {
        total += 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
    }

    const double target = rng.uniform01() * total;
    double accum = 0.0;
    for (int i = 0; i < limit_e; ++i) {
        accum += 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
        if (accum >= target) {
            return i;
        }
    }

    return limit_e - 1;
}

static int choose_expert(
    SplitMix64& rng,
    int E,
    int distribution_id,
    bool avoid_last_expert) {
    const int limit_e = avoid_last_expert && E > 1 ? E - 1 : E;
    if (limit_e <= 1) {
        return 0;
    }

    switch (distribution_id) {
        case RCR_DIST_UNIFORM:
            return rng.uniform_int(0, limit_e - 1);

        case RCR_DIST_ZIPF_1_2:
            return sample_zipf_1_2(rng, limit_e);

        case RCR_DIST_SINGLE_HOT:
            if (rng.chance(0.88)) {
                return 0;
            }
            return rng.uniform_int(1, limit_e - 1);

        case RCR_DIST_FEW_HOT: {
            const int hot_count = std::min(limit_e, limit_e >= 8 ? 4 : 2);
            if (rng.chance(0.90)) {
                return rng.uniform_int(0, hot_count - 1);
            }
            if (hot_count == limit_e) {
                return rng.uniform_int(0, limit_e - 1);
            }
            return rng.uniform_int(hot_count, limit_e - 1);
        }

        case RCR_DIST_MANY_INVALID:
            return rng.uniform_int(0, limit_e - 1);

        case RCR_DIST_MANY_DUPLICATE:
            if (rng.chance(0.70)) {
                return rng.uniform_int(0, std::min(limit_e - 1, 3));
            }
            return rng.uniform_int(0, limit_e - 1);

        default:
            return rng.uniform_int(0, limit_e - 1);
    }
}

static int16_t choose_weight(SplitMix64& rng, int distribution_id) {
    if (distribution_id == RCR_DIST_MANY_INVALID && rng.chance(0.15)) {
        return 0;
    }

    int w = rng.uniform_int(-9, 9);
    if (w == 0) {
        w = rng.chance(0.5) ? 1 : -1;
    }
    return static_cast<int16_t>(w);
}

static void force_cancel_duplicate_token(HostCase* hc) {
    const int N = hc->run.N;
    const int E = hc->run.E;
    const int K = hc->run.K;
    if (N < 8 || E < 2) {
        return;
    }

    const int t = 7;
    const int e = std::min(1, E - 1);
    hc->valid[t] = 1;

    for (int k = 0; k < K; ++k) {
        hc->expert[t * K + k] = static_cast<int16_t>(e);
        hc->weight[t * K + k] = 0;
    }

    if (K == 2) {
        hc->weight[t * K + 0] = 5;
        hc->weight[t * K + 1] = -5;
    } else {
        hc->weight[t * K + 0] = 7;
        hc->weight[t * K + 1] = -2;
        hc->weight[t * K + 2] = -5;
        hc->weight[t * K + 3] = 0;
    }
}

static HostCase make_case(
    const char* name,
    int N,
    int C,
    int E,
    int K,
    int distribution_id,
    uint64_t case_seed,
    bool force_all_invalid,
    bool force_empty_last_expert,
    bool inject_cancel_duplicate) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = RCR_ABI_VERSION;
    hc.run.N = N;
    hc.run.C = C;
    hc.run.E = E;
    hc.run.K = K;
    hc.run.seed_id = static_cast<int32_t>(case_seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((case_seed >> 32) & 0x7fffffff);

    if (!rcr_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated run spec");
    }

    SplitMix64 rng(g_state ^ case_seed);

    hc.x.resize(static_cast<size_t>(N) * static_cast<size_t>(C));
    hc.expert.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.weight.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.valid.resize(static_cast<size_t>(N));

    for (size_t i = 0; i < hc.x.size(); ++i) {
        int v = rng.uniform_int(-31, 31);
        if (v == 0 && rng.chance(0.85)) {
            v = rng.chance(0.5) ? 1 : -1;
        }
        hc.x[i] = static_cast<int16_t>(v);
    }

    for (int t = 0; t < N; ++t) {
        bool is_valid = true;

        if (force_all_invalid) {
            is_valid = false;
        } else if (distribution_id == RCR_DIST_MANY_INVALID) {
            is_valid = !rng.chance(0.55);
        } else {
            is_valid = !rng.chance(0.03);
        }

        hc.valid[t] = static_cast<uint8_t>(is_valid ? 1 : 0);

        if (!is_valid) {
            for (int k = 0; k < K; ++k) {
                hc.expert[t * K + k] = rng.chance(0.70)
                    ? static_cast<int16_t>(-1)
                    : static_cast<int16_t>(choose_expert(rng, E, distribution_id, force_empty_last_expert));
                hc.weight[t * K + k] = choose_weight(rng, distribution_id);
            }
            continue;
        }

        if (distribution_id == RCR_DIST_MANY_DUPLICATE) {
            const int e0 = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            for (int k = 0; k < K; ++k) {
                int e = e0;
                if (k == K - 1 && rng.chance(0.35)) {
                    e = choose_expert(rng, E, RCR_DIST_UNIFORM, force_empty_last_expert);
                }
                hc.expert[t * K + k] = static_cast<int16_t>(e);
                hc.weight[t * K + k] = choose_weight(rng, distribution_id);
            }
            continue;
        }

        for (int k = 0; k < K; ++k) {
            const double invalid_slot_prob =
                distribution_id == RCR_DIST_MANY_INVALID ? 0.45 : 0.04;

            if (rng.chance(invalid_slot_prob)) {
                hc.expert[t * K + k] = -1;
                hc.weight[t * K + k] = choose_weight(rng, distribution_id);
                continue;
            }

            const int e = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            hc.expert[t * K + k] = static_cast<int16_t>(e);
            hc.weight[t * K + k] = choose_weight(rng, distribution_id);
        }
    }

    if (inject_cancel_duplicate) {
        force_cancel_duplicate_token(&hc);
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x200000001b3ULL;

    cases.push_back(make_case(
        "uniform_mid",
        65536, 32, 32, 2,
        RCR_DIST_UNIFORM,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "single_hot_large",
        131072, 64, 128, 4,
        RCR_DIST_SINGLE_HOT,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "many_duplicate_wide",
        65536, 128, 32, 4,
        RCR_DIST_MANY_DUPLICATE,
        s++,
        false,
        false,
        true));

    cases.push_back(make_case(
        "many_invalid_smallE",
        32768, 16, 8, 2,
        RCR_DIST_MANY_INVALID,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "zipf_largeE",
        131072, 32, 128, 4,
        RCR_DIST_ZIPF_1_2,
        s++,
        false,
        false,
        false));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int C = hc.run.C;
    const int E = hc.run.E;
    const int K = hc.run.K;

    dc->x.allocate(hc.x.size());
    dc->expert.allocate(hc.expert.size());
    dc->weight.allocate(hc.weight.size());
    dc->valid.allocate(hc.valid.size());

    dc->x.upload(hc.x);
    dc->expert.upload(hc.expert);
    dc->weight.upload(hc.weight);
    dc->valid.upload(hc.valid);

    dc->counts.allocate(static_cast<size_t>(E));
    dc->offsets.allocate(static_cast<size_t>(E + 1));
    dc->packed_token.allocate(static_cast<size_t>(N) * static_cast<size_t>(K));
    dc->packed_weight.allocate(static_cast<size_t>(N) * static_cast<size_t>(K));
    dc->sum.allocate(static_cast<size_t>(E) * static_cast<size_t>(C));
    dc->argmax_abs.allocate(static_cast<size_t>(E) * static_cast<size_t>(C));

    dc->inputs = {};
    dc->inputs.x = dc->x.ptr;
    dc->inputs.expert = dc->expert.ptr;
    dc->inputs.weight = dc->weight.ptr;
    dc->inputs.valid = dc->valid.ptr;

    dc->outputs = {};
    dc->outputs.counts = dc->counts.ptr;
    dc->outputs.offsets = dc->offsets.ptr;
    dc->outputs.packed_token = dc->packed_token.ptr;
    dc->outputs.packed_weight = dc->packed_weight.ptr;
    dc->outputs.sum = dc->sum.ptr;
    dc->outputs.argmax_abs = dc->argmax_abs.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        RcrProblemSpec spec = {};
        spec.abi_version = RCR_ABI_VERSION;
        spec.max_N = RCR_MAX_N;
        spec.max_C = RCR_MAX_C;
        spec.max_E = RCR_MAX_E;
        spec.max_K = RCR_MAX_K;
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
                "bench_case %-24s N=%d C=%d E=%d K=%d dist=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.C,
                hc.run.E,
                hc.run.K,
                hc.run.distribution_id);
        }

        for (int warmup = 0; warmup < 5; ++warmup) {
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
