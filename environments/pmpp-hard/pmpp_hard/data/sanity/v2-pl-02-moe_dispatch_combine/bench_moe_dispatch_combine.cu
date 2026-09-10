// ============================================================================
// file: bench_moe_dispatch_combine.cu
// ============================================================================

#include "moe_dispatch_combine_common.h"

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

static constexpr uint64_t g_state = 0x243f6a8885a308d3ULL;

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
    MdcRunSpec run;
    std::vector<int16_t> expert;
    std::vector<int16_t> gate;
    std::vector<uint8_t> valid;
    std::vector<int16_t> expert_out;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<int16_t> expert;
    DeviceBuffer<int16_t> gate;
    DeviceBuffer<uint8_t> valid;
    DeviceBuffer<int16_t> expert_out;

    DeviceBuffer<int32_t> counts;
    DeviceBuffer<int32_t> offsets;
    DeviceBuffer<int32_t> packed_token;
    DeviceBuffer<int32_t> packed_slot;
    DeviceBuffer<int16_t> packed_gate;
    DeviceBuffer<uint8_t> dropped;
    DeviceBuffer<int64_t> y;

    MdcInputs inputs;
    MdcOutputs outputs;
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
        case MDC_DIST_UNIFORM:
            return rng.uniform_int(0, limit_e - 1);

        case MDC_DIST_ZIPF_1_2:
            return sample_zipf_1_2(rng, limit_e);

        case MDC_DIST_SINGLE_HOT:
            if (rng.chance(0.90)) {
                return 0;
            }
            return rng.uniform_int(1, limit_e - 1);

        case MDC_DIST_FEW_HOT: {
            const int hot_count = std::min(limit_e, limit_e >= 8 ? 4 : 2);
            if (rng.chance(0.92)) {
                return rng.uniform_int(0, hot_count - 1);
            }
            if (hot_count == limit_e) {
                return rng.uniform_int(0, limit_e - 1);
            }
            return rng.uniform_int(hot_count, limit_e - 1);
        }

        case MDC_DIST_MANY_INVALID:
            return rng.uniform_int(0, limit_e - 1);

        case MDC_DIST_MANY_DUPLICATE:
            if (rng.chance(0.75)) {
                return rng.uniform_int(0, std::min(limit_e - 1, 3));
            }
            return rng.uniform_int(0, limit_e - 1);

        default:
            return rng.uniform_int(0, limit_e - 1);
    }
}

static int16_t choose_gate(SplitMix64& rng, int distribution_id) {
    switch (distribution_id) {
        case MDC_DIST_SINGLE_HOT:
        case MDC_DIST_FEW_HOT:
            return static_cast<int16_t>(rng.uniform_int(-16, 127));

        case MDC_DIST_MANY_INVALID:
            return static_cast<int16_t>(rng.uniform_int(-127, 64));

        case MDC_DIST_MANY_DUPLICATE:
            return static_cast<int16_t>(rng.uniform_int(-32, 127));

        default:
            return static_cast<int16_t>(rng.uniform_int(-127, 127));
    }
}

static HostCase make_random_case(
    const char* name,
    int N,
    int D,
    int E,
    int K,
    int cap,
    int distribution_id,
    uint64_t case_seed,
    bool force_all_invalid,
    bool force_empty_last_expert) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = MDC_ABI_VERSION;
    hc.run.N = N;
    hc.run.D = D;
    hc.run.E = E;
    hc.run.K = K;
    hc.run.cap = cap;
    hc.run.seed_id = static_cast<int32_t>(case_seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((case_seed >> 32) & 0x7fffffff);

    if (!mdc_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated run spec");
    }

    SplitMix64 rng(g_state ^ case_seed);

    hc.expert.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.gate.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.valid.resize(static_cast<size_t>(N));
    hc.expert_out.resize(static_cast<size_t>(E) * static_cast<size_t>(D));

    for (size_t i = 0; i < hc.expert_out.size(); ++i) {
        int v = rng.uniform_int(-31, 31);
        if (v == 0 && rng.chance(0.90)) {
            v = rng.chance(0.5) ? 1 : -1;
        }
        hc.expert_out[i] = static_cast<int16_t>(v);
    }

    for (int t = 0; t < N; ++t) {
        bool is_valid = true;

        if (force_all_invalid) {
            is_valid = false;
        } else if (distribution_id == MDC_DIST_MANY_INVALID) {
            is_valid = !rng.chance(0.60);
        } else {
            is_valid = !rng.chance(0.02);
        }

        hc.valid[t] = static_cast<uint8_t>(is_valid ? 1 : 0);

        if (!is_valid) {
            for (int k = 0; k < K; ++k) {
                hc.expert[t * K + k] = rng.chance(0.70)
                    ? static_cast<int16_t>(-1)
                    : static_cast<int16_t>(choose_expert(rng, E, distribution_id, force_empty_last_expert));
                hc.gate[t * K + k] = choose_gate(rng, distribution_id);
            }
            continue;
        }

        if (distribution_id == MDC_DIST_MANY_DUPLICATE) {
            const int e0 = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            for (int k = 0; k < K; ++k) {
                int e = e0;
                if (k == K - 1 && rng.chance(0.35)) {
                    e = choose_expert(rng, E, MDC_DIST_UNIFORM, force_empty_last_expert);
                }
                hc.expert[t * K + k] = static_cast<int16_t>(e);
                hc.gate[t * K + k] = choose_gate(rng, distribution_id);
            }
            continue;
        }

        for (int k = 0; k < K; ++k) {
            const double unused_prob =
                distribution_id == MDC_DIST_MANY_INVALID ? 0.50 : 0.04;

            if (rng.chance(unused_prob)) {
                hc.expert[t * K + k] = -1;
                hc.gate[t * K + k] = choose_gate(rng, distribution_id);
                continue;
            }

            const int e = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            hc.expert[t * K + k] = static_cast<int16_t>(e);
            hc.gate[t * K + k] = choose_gate(rng, distribution_id);
        }
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x200000001b3ULL;

    cases.push_back(make_random_case(
        "uniform_mid",
        65536, 32, 32, 2, 128,
        MDC_DIST_UNIFORM,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "single_hot_overflow_large",
        131072, 64, 128, 4, 16,
        MDC_DIST_SINGLE_HOT,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "few_hot_overflow_wide",
        65536, 128, 32, 4, 32,
        MDC_DIST_FEW_HOT,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "many_duplicate_hot",
        131072, 32, 8, 4, 16,
        MDC_DIST_MANY_DUPLICATE,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "zipf_largeE",
        131072, 16, 128, 4, 64,
        MDC_DIST_ZIPF_1_2,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "many_invalid_smallE",
        32768, 16, 8, 2, 64,
        MDC_DIST_MANY_INVALID,
        s++,
        false,
        false));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int D = hc.run.D;
    const int E = hc.run.E;
    const int K = hc.run.K;
    const int cap = hc.run.cap;
    const size_t packed_capacity = static_cast<size_t>(E) * static_cast<size_t>(cap);

    dc->expert.allocate(hc.expert.size());
    dc->gate.allocate(hc.gate.size());
    dc->valid.allocate(hc.valid.size());
    dc->expert_out.allocate(hc.expert_out.size());

    dc->expert.upload(hc.expert);
    dc->gate.upload(hc.gate);
    dc->valid.upload(hc.valid);
    dc->expert_out.upload(hc.expert_out);

    dc->counts.allocate(static_cast<size_t>(E));
    dc->offsets.allocate(static_cast<size_t>(E + 1));
    dc->packed_token.allocate(packed_capacity);
    dc->packed_slot.allocate(packed_capacity);
    dc->packed_gate.allocate(packed_capacity);
    dc->dropped.allocate(static_cast<size_t>(N) * static_cast<size_t>(K));
    dc->y.allocate(static_cast<size_t>(N) * static_cast<size_t>(D));

    dc->inputs = {};
    dc->inputs.expert = dc->expert.ptr;
    dc->inputs.gate = dc->gate.ptr;
    dc->inputs.valid = dc->valid.ptr;
    dc->inputs.expert_out = dc->expert_out.ptr;

    dc->outputs = {};
    dc->outputs.counts = dc->counts.ptr;
    dc->outputs.offsets = dc->offsets.ptr;
    dc->outputs.packed_token = dc->packed_token.ptr;
    dc->outputs.packed_slot = dc->packed_slot.ptr;
    dc->outputs.packed_gate = dc->packed_gate.ptr;
    dc->outputs.dropped = dc->dropped.ptr;
    dc->outputs.y = dc->y.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        MdcProblemSpec spec = {};
        spec.abi_version = MDC_ABI_VERSION;
        spec.max_N = MDC_MAX_N;
        spec.max_D = MDC_MAX_D;
        spec.max_E = MDC_MAX_E;
        spec.max_K = MDC_MAX_K;
        spec.max_cap = MDC_MAX_CAP;
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
                "bench_case %-28s N=%d D=%d E=%d K=%d cap=%d dist=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.D,
                hc.run.E,
                hc.run.K,
                hc.run.cap,
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
