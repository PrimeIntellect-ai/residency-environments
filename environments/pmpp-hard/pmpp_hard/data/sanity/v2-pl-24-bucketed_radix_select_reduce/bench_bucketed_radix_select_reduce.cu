// file: bench_bucketed_radix_select_reduce.cu

#include "bucketed_radix_select_reduce_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0x38f6d1ac9b247e53ULL;

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

    uint32_t next_u32() {
        return static_cast<uint32_t>(next_u64() >> 32);
    }

    int32_t next_i32() {
        return static_cast<int32_t>(next_u64() >> 32);
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
    BrsrRunSpec run;
    std::vector<uint32_t> key;
    std::vector<int32_t> value;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<uint32_t> key;
    DeviceBuffer<int32_t> value;

    DeviceBuffer<uint32_t> threshold_key;
    DeviceBuffer<int32_t> count;
    DeviceBuffer<int32_t> topT_indices;
    DeviceBuffer<int64_t> topT_sum;
    DeviceBuffer<int32_t> topT_max;
    DeviceBuffer<int32_t> topT_argmax;
    DeviceBuffer<int32_t> pass_histograms;
    DeviceBuffer<int32_t> chosen_bucket;
    DeviceBuffer<int32_t> carried_rank;
    DeviceBuffer<uint32_t> prefix_after_pass;

    BrsrInputs inputs;
    BrsrOutputs outputs;
};

static uint32_t make_key(
    SplitMix64& rng,
    int idx,
    int distribution_id) {
    switch (distribution_id) {
        case BRSR_DIST_UNIFORM:
            return rng.next_u32();

        case BRSR_DIST_CLUSTERED: {
            const uint32_t cluster = static_cast<uint32_t>((idx / 257) & 0xff);
            const uint32_t high = ((cluster * 37u + 91u) & 0xffu) << 24;
            const uint32_t mid = static_cast<uint32_t>((idx * 13u) & 0xffffu) << 8;
            const uint32_t low = static_cast<uint32_t>(rng.uniform_int(0, 255));
            return high | mid | low;
        }

        case BRSR_DIST_MANY_TIES: {
            const uint32_t group = static_cast<uint32_t>((idx / 8) % 1024);
            const uint32_t high = ((group * 17u) & 0xffu) << 24;
            const uint32_t low = (group & 0xffffu) << 4;
            return high | low;
        }

        case BRSR_DIST_HIGH_BUCKET_HOT: {
            if ((idx % 5) != 0) {
                const uint32_t hot = 0xf0000000u | (static_cast<uint32_t>(idx % 31) << 20);
                return hot | (rng.next_u32() & 0x000fffffu);
            }
            return rng.next_u32() & 0x7fffffffu;
        }

        default:
            return rng.next_u32();
    }
}

static int32_t make_value(SplitMix64& rng, int idx) {
    int32_t v = rng.next_i32();
    if ((idx % 257) == 0) v = INT_MAX - (idx & 1023);
    if ((idx % 389) == 0) v = INT_MIN + (idx & 1023);
    if ((idx % 1021) == 0) v = 0;
    return v;
}

static HostCase make_case(
    const char* name,
    int N,
    int T,
    int distribution_id,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = BRSR_ABI_VERSION;
    hc.run.N = N;
    hc.run.T = T;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!brsr_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid BrsrRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.key.resize((size_t)N);
    hc.value.resize((size_t)N);

    for (int i = 0; i < N; ++i) {
        hc.key[(size_t)i] = make_key(rng, i, distribution_id);
        hc.value[(size_t)i] = make_value(rng, i);
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x200000001b3ULL;

    cases.push_back(make_case(
        "uniform_N1M_T16",
        1 << 20,
        16,
        BRSR_DIST_UNIFORM,
        s++));

    cases.push_back(make_case(
        "clustered_N262K_T256",
        1 << 18,
        256,
        BRSR_DIST_CLUSTERED,
        s++));

    cases.push_back(make_case(
        "many_ties_N65536_T4096",
        65536,
        4096,
        BRSR_DIST_MANY_TIES,
        s++));

    cases.push_back(make_case(
        "hot_N1M_T16",
        1 << 20,
        16,
        BRSR_DIST_HIGH_BUCKET_HOT,
        s++));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int count = brsr_count_host(hc.run.N, hc.run.T);

    dc->key.allocate(hc.key.size());
    dc->value.allocate(hc.value.size());

    dc->key.upload(hc.key);
    dc->value.upload(hc.value);

    dc->threshold_key.allocate(1);
    dc->count.allocate(1);
    dc->topT_indices.allocate((size_t)count);
    dc->topT_sum.allocate(1);
    dc->topT_max.allocate(1);
    dc->topT_argmax.allocate(1);
    dc->pass_histograms.allocate((size_t)BRSR_NUM_PASSES * (size_t)BRSR_BUCKETS);
    dc->chosen_bucket.allocate((size_t)BRSR_NUM_PASSES);
    dc->carried_rank.allocate((size_t)BRSR_NUM_PASSES);
    dc->prefix_after_pass.allocate((size_t)BRSR_NUM_PASSES);

    dc->inputs = {};
    dc->inputs.key = dc->key.ptr;
    dc->inputs.value = dc->value.ptr;

    dc->outputs = {};
    dc->outputs.threshold_key = dc->threshold_key.ptr;
    dc->outputs.count = dc->count.ptr;
    dc->outputs.topT_indices = dc->topT_indices.ptr;
    dc->outputs.topT_sum = dc->topT_sum.ptr;
    dc->outputs.topT_max = dc->topT_max.ptr;
    dc->outputs.topT_argmax = dc->topT_argmax.ptr;
    dc->outputs.pass_histograms = dc->pass_histograms.ptr;
    dc->outputs.chosen_bucket = dc->chosen_bucket.ptr;
    dc->outputs.carried_rank = dc->carried_rank.ptr;
    dc->outputs.prefix_after_pass = dc->prefix_after_pass.ptr;

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

        int max_N = BRSR_MIN_N;
        int max_T = 1;

        for (const HostCase& hc : host_cases) {
            max_N = std::max(max_N, hc.run.N);
            max_T = std::max(max_T, hc.run.T);
        }

        BrsrProblemSpec spec = {};
        spec.abi_version = BRSR_ABI_VERSION;
        spec.max_N = max_N;
        spec.max_T = max_T;
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
                "bench_case %-24s N=%d T=%d dist=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.T,
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
