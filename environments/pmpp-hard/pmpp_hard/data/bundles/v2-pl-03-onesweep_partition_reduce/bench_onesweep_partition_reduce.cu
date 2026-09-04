// ============================================================================
// file: bench_onesweep_partition_reduce.cu
// ============================================================================

#include "onesweep_partition_reduce_common.h"
#include "onesweep_partition_reduce_oracle.hpp"
#include "pmpp_bench_digest.cuh"

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
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
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

struct CaseSpec {
    const char* name;
    int N;
    int radix_bits;
    int distribution_id;
};

static const CaseSpec kCaseSpecs[] = {
    {"uniform_large_r8", 131072, 8, OPR_DIST_UNIFORM},
    {"single_hot_large_r8", 131072, 8, OPR_DIST_SINGLE_HOT},
    {"few_hot_mid_r8", 65536, 8, OPR_DIST_FEW_HOT},
    {"zipf_large_r4", 131072, 4, OPR_DIST_ZIPF_1_2},
    {"nearly_sorted_mid_r8", 65536, 8, OPR_DIST_NEARLY_SORTED},
    {"reverse_mid_r4", 65536, 4, OPR_DIST_REVERSE},
};

static constexpr int kNumCases = sizeof(kCaseSpecs) / sizeof(kCaseSpecs[0]);

// Every timed call gets its own input variant: variants[c][k] for timed iteration k,
// plus variants[c][K] reserved for warmup so warmup never touches a digested region.
// Shape family (N / radix_bits / distribution) is fixed per case; only data varies.
// Data seeds come from a per-case SplitMix64 stream keyed by PMPP_BENCH_SEED, so the
// same seed yields a bit-identical variant sequence (paired digest compare intact).
static std::vector<std::vector<HostCase>> build_bench_variants(int iters) {
    std::vector<std::vector<HostCase>> variants(kNumCases);
    SplitMix64 seed_mix(pmpp::bench_seed(0x200000001b3ULL));

    for (int c = 0; c < kNumCases; ++c) {
        const CaseSpec& cs = kCaseSpecs[c];
        SplitMix64 vs(seed_mix.next_u64());
        variants[c].reserve(static_cast<size_t>(iters) + 1);
        for (int k = 0; k <= iters; ++k) {
            variants[c].push_back(make_case(
                cs.name,
                cs.N,
                cs.radix_bits,
                cs.distribution_id,
                vs.next_u64(),
                false,
                0,
                false));
        }
    }

    return variants;
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

        std::vector<std::vector<HostCase>> host_variants = build_bench_variants(iters);
        std::vector<std::vector<DeviceCase*>> cases(kNumCases);

        for (int c = 0; c < kNumCases; ++c) {
            cases[c].reserve(host_variants[c].size());
            for (const HostCase& hc : host_variants[c]) {
                cases[c].push_back(make_device_case(hc));
            }
            const HostCase& hc = host_variants[c].front();
            std::printf(
                "bench_case %-28s N=%d r=%d digits=%d dist=%d variants=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.radix_bits,
                opr_num_digits(hc.run.radix_bits),
                hc.run.distribution_id,
                iters);
        }

        // Warmup runs only the dedicated variant (index iters); its outputs are never
        // folded, so warmup work cannot pre-populate any digested output region.
        for (int warmup = 0; warmup < 5; ++warmup) {
            for (int c = 0; c < kNumCases; ++c) {
                DeviceCase* dc = cases[c][static_cast<size_t>(iters)];
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
            for (int c = 0; c < kNumCases; ++c) {
                DeviceCase* dc = cases[c][static_cast<size_t>(iter)];
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

        const double calls = static_cast<double>(iters) * static_cast<double>(kNumCases);
        const double avg_ms = static_cast<double>(elapsed_ms) / calls;

        std::printf("avg_ms=%.6f\n", avg_ms);

        // Anti-hack digest: every timed call (iter, case) wrote its own output region for
        // its own input variant; fold ALL of them (all nine graded fields, fixed order).
        // A call that no-ops or replays a cached/stale output leaves a wrong region and
        // the digest cannot match the reference bench on the same PMPP_BENCH_SEED.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        for (int iter = 0; iter < iters; ++iter) {
            for (int c = 0; c < kNumCases; ++c) {
                DeviceCase* dc = cases[c][static_cast<size_t>(iter)];
                const size_t N = (size_t)dc->host.run.N;
                const size_t num_digits = (size_t)opr_num_digits(dc->host.run.radix_bits);
                const size_t num_tiles = (size_t)opr_num_tiles(dc->host.run.N);
                dg.dev(dc->tile_digit_offsets.ptr, num_tiles * num_digits * sizeof(int32_t));
                dg.dev(dc->digit_counts.ptr, num_digits * sizeof(int32_t));
                dg.dev(dc->digit_offsets.ptr, (num_digits + 1) * sizeof(int32_t));
                dg.dev(dc->packed_key.ptr, N * sizeof(uint32_t));
                dg.dev(dc->packed_val.ptr, N * sizeof(int32_t));
                dg.dev(dc->packed_src.ptr, N * sizeof(int32_t));
                dg.dev(dc->bucket_sum.ptr, num_digits * sizeof(int64_t));
                dg.dev(dc->bucket_max.ptr, num_digits * sizeof(int64_t));
                dg.dev(dc->bucket_argmax.ptr, num_digits * sizeof(int32_t));
            }
        }
        dg.print();

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (int c = 0; c < kNumCases; ++c) {
            for (DeviceCase* dc : cases[c]) {
                delete dc;
            }
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
