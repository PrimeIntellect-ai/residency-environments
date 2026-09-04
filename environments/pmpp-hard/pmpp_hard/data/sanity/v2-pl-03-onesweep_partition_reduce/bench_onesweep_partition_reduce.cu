// ============================================================================
// file: bench_onesweep_partition_reduce.cu
// ============================================================================

#include "onesweep_partition_reduce_common.h"
#include "onesweep_partition_reduce_oracle.hpp"

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

static constexpr uint64_t g_state = 0x13198a2e03707344ULL;

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
    OprRunSpec run;
    std::vector<uint32_t> key;
    std::vector<int32_t> val;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<uint32_t> key;
    DeviceBuffer<int32_t> val;

    DeviceBuffer<int32_t> tile_digit_offsets;
    DeviceBuffer<int32_t> digit_counts;
    DeviceBuffer<int32_t> digit_offsets;
    DeviceBuffer<uint32_t> packed_key;
    DeviceBuffer<int32_t> packed_val;
    DeviceBuffer<int32_t> packed_src;
    DeviceBuffer<int64_t> bucket_sum;
    DeviceBuffer<int64_t> bucket_max;
    DeviceBuffer<int32_t> bucket_argmax;

    OprInputs inputs;
    OprOutputs outputs;
};

static int sample_zipf_1_2(SplitMix64& rng, int num_digits) {
    double total = 0.0;
    for (int i = 0; i < num_digits; ++i) {
        total += 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
    }

    const double target = rng.uniform01() * total;
    double accum = 0.0;
    for (int i = 0; i < num_digits; ++i) {
        accum += 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
        if (accum >= target) {
            return i;
        }
    }

    return num_digits - 1;
}

static uint32_t make_key_for_digit(int digit, int radix_bits, SplitMix64& rng) {
    const int low_bits = 32 - radix_bits;
    const uint32_t low_mask = static_cast<uint32_t>((1ULL << low_bits) - 1ULL);
    const uint32_t high = static_cast<uint32_t>(digit) << low_bits;
    const uint32_t low = static_cast<uint32_t>(rng.next_u64()) & low_mask;
    return high | low;
}

static int choose_digit(
    SplitMix64& rng,
    int i,
    int N,
    int radix_bits,
    int distribution_id,
    bool avoid_last_digit) {
    const int num_digits = opr_num_digits(radix_bits);
    const int limit_digits = avoid_last_digit && num_digits > 1 ? num_digits - 1 : num_digits;

    if (limit_digits <= 1) {
        return 0;
    }

    switch (distribution_id) {
        case OPR_DIST_UNIFORM:
            return rng.uniform_int(0, limit_digits - 1);

        case OPR_DIST_SINGLE_HOT:
            if (rng.chance(0.92)) {
                return 0;
            }
            return rng.uniform_int(1, limit_digits - 1);

        case OPR_DIST_FEW_HOT: {
            const int hot_count = std::min(limit_digits, limit_digits >= 8 ? 4 : 2);
            if (rng.chance(0.90)) {
                return rng.uniform_int(0, hot_count - 1);
            }
            if (hot_count == limit_digits) {
                return rng.uniform_int(0, limit_digits - 1);
            }
            return rng.uniform_int(hot_count, limit_digits - 1);
        }

        case OPR_DIST_ZIPF_1_2:
            return sample_zipf_1_2(rng, limit_digits);

        case OPR_DIST_NEARLY_SORTED: {
            int d = static_cast<int>((static_cast<int64_t>(i) * limit_digits) / N);
            if (d >= limit_digits) d = limit_digits - 1;
            if (rng.chance(0.08)) {
                const int jitter = rng.uniform_int(-2, 2);
                d = std::max(0, std::min(limit_digits - 1, d + jitter));
            }
            if (rng.chance(0.02)) {
                d = rng.uniform_int(0, limit_digits - 1);
            }
            return d;
        }

        case OPR_DIST_REVERSE: {
            int d = limit_digits - 1 - static_cast<int>((static_cast<int64_t>(i) * limit_digits) / N);
            if (d < 0) d = 0;
            if (rng.chance(0.08)) {
                const int jitter = rng.uniform_int(-2, 2);
                d = std::max(0, std::min(limit_digits - 1, d + jitter));
            }
            if (rng.chance(0.02)) {
                d = rng.uniform_int(0, limit_digits - 1);
            }
            return d;
        }

        default:
            return rng.uniform_int(0, limit_digits - 1);
    }
}

static HostCase make_case(
    const char* name,
    int N,
    int radix_bits,
    int distribution_id,
    uint64_t case_seed,
    bool force_one_bucket,
    int forced_bucket,
    bool force_empty_last_digit) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = OPR_ABI_VERSION;
    hc.run.N = N;
    hc.run.radix_bits = radix_bits;
    hc.run.seed_id = static_cast<int32_t>(case_seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((case_seed >> 32) & 0x7fffffff);

    if (!opr_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated run spec");
    }

    SplitMix64 rng(g_state ^ case_seed);

    hc.key.resize(static_cast<size_t>(N));
    hc.val.resize(static_cast<size_t>(N));

    for (int i = 0; i < N; ++i) {
        int d = forced_bucket;
        if (!force_one_bucket) {
            d = choose_digit(rng, i, N, radix_bits, distribution_id, force_empty_last_digit);
        }

        hc.key[static_cast<size_t>(i)] = make_key_for_digit(d, radix_bits, rng);

        int v = rng.uniform_int(-100000, 100000);
        if (v == 0 && rng.chance(0.80)) {
            v = rng.chance(0.5) ? 1 : -1;
        }
        hc.val[static_cast<size_t>(i)] = v;
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x200000001b3ULL;

    cases.push_back(make_case(
        "uniform_large_r8",
        131072,
        8,
        OPR_DIST_UNIFORM,
        s++,
        false,
        0,
        false));

    cases.push_back(make_case(
        "single_hot_large_r8",
        131072,
        8,
        OPR_DIST_SINGLE_HOT,
        s++,
        false,
        0,
        false));

    cases.push_back(make_case(
        "few_hot_mid_r8",
        65536,
        8,
        OPR_DIST_FEW_HOT,
        s++,
        false,
        0,
        false));

    cases.push_back(make_case(
        "zipf_large_r4",
        131072,
        4,
        OPR_DIST_ZIPF_1_2,
        s++,
        false,
        0,
        false));

    cases.push_back(make_case(
        "nearly_sorted_mid_r8",
        65536,
        8,
        OPR_DIST_NEARLY_SORTED,
        s++,
        false,
        0,
        false));

    cases.push_back(make_case(
        "reverse_mid_r4",
        65536,
        4,
        OPR_DIST_REVERSE,
        s++,
        false,
        0,
        false));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int num_digits = opr_num_digits(hc.run.radix_bits);
    const int num_tiles = opr_num_tiles(N);

    dc->key.allocate(hc.key.size());
    dc->val.allocate(hc.val.size());

    dc->key.upload(hc.key);
    dc->val.upload(hc.val);

    dc->tile_digit_offsets.allocate(static_cast<size_t>(num_tiles) * static_cast<size_t>(num_digits));
    dc->digit_counts.allocate(static_cast<size_t>(num_digits));
    dc->digit_offsets.allocate(static_cast<size_t>(num_digits + 1));
    dc->packed_key.allocate(static_cast<size_t>(N));
    dc->packed_val.allocate(static_cast<size_t>(N));
    dc->packed_src.allocate(static_cast<size_t>(N));
    dc->bucket_sum.allocate(static_cast<size_t>(num_digits));
    dc->bucket_max.allocate(static_cast<size_t>(num_digits));
    dc->bucket_argmax.allocate(static_cast<size_t>(num_digits));

    dc->inputs = {};
    dc->inputs.key = dc->key.ptr;
    dc->inputs.val = dc->val.ptr;

    dc->outputs = {};
    dc->outputs.tile_digit_offsets = dc->tile_digit_offsets.ptr;
    dc->outputs.digit_counts = dc->digit_counts.ptr;
    dc->outputs.digit_offsets = dc->digit_offsets.ptr;
    dc->outputs.packed_key = dc->packed_key.ptr;
    dc->outputs.packed_val = dc->packed_val.ptr;
    dc->outputs.packed_src = dc->packed_src.ptr;
    dc->outputs.bucket_sum = dc->bucket_sum.ptr;
    dc->outputs.bucket_max = dc->bucket_max.ptr;
    dc->outputs.bucket_argmax = dc->bucket_argmax.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        OprProblemSpec spec = {};
        spec.abi_version = OPR_ABI_VERSION;
        spec.max_N = OPR_MAX_N;
        spec.max_radix_bits = OPR_MAX_RADIX_BITS;
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
                "bench_case %-28s N=%d r=%d digits=%d dist=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.radix_bits,
                opr_num_digits(hc.run.radix_bits),
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
