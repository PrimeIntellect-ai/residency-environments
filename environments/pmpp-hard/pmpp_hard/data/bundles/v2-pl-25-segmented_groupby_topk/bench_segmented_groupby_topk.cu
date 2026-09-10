// file: bench_segmented_groupby_topk.cu

#include "segmented_groupby_topk_common.h"
#include "pmpp_bench_digest.cuh"

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

static constexpr uint64_t g_state = 0x81d2c9a4703f5b6eULL;

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

    bool chance_permille(int p) {
        if (p <= 0) return false;
        if (p >= 1000) return true;
        return static_cast<int>(next_u64() % 1000ULL) < p;
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
    SgtRunSpec run;
    std::vector<int32_t> group_id;
    std::vector<int32_t> key;
    std::vector<int32_t> value;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<int32_t> group_id;
    DeviceBuffer<int32_t> key;
    DeviceBuffer<int32_t> value;

    DeviceBuffer<int32_t> group_counts;
    DeviceBuffer<int32_t> group_offsets;
    DeviceBuffer<int32_t> packed_topk_origidx;
    DeviceBuffer<int64_t> per_group_sum;
    DeviceBuffer<int32_t> per_group_max;
    DeviceBuffer<int32_t> per_group_argmax;
    DeviceBuffer<int32_t> kept_count;

    SgtInputs inputs;
    SgtOutputs outputs;
};

static int choose_group(
    SplitMix64& rng,
    int row,
    int G,
    int distribution_id,
    bool single_hot) {
    if (single_hot || distribution_id == SGTK_DIST_SINGLE_HOT) {
        if (rng.chance_permille(930)) return 0;
        return rng.uniform_int(1, G - 1);
    }

    if (distribution_id == SGTK_DIST_ZIPF_HOT) {
        const int r = rng.uniform_int(0, 999);
        if (r < 560) return 0;
        if (r < 740) return std::min(1, G - 1);
        if (r < 840) return std::min(2, G - 1);
        if (r < 900) return std::min(3, G - 1);

        const int hot_tail = std::min(G, 16);
        if (r < 965) return rng.uniform_int(0, hot_tail - 1);

        return rng.uniform_int(0, G - 1);
    }

    if (distribution_id == SGTK_DIST_MANY_TIES && rng.chance_permille(750)) {
        return row % std::min(G, 8);
    }

    return rng.uniform_int(0, G - 1);
}

static int32_t make_key(
    SplitMix64& rng,
    int row,
    int group,
    int distribution_id) {
    if (distribution_id == SGTK_DIST_MANY_TIES) {
        return static_cast<int32_t>((group * 17 + (row / 8)) % 23);
    }

    if (distribution_id == SGTK_DIST_ZIPF_HOT || distribution_id == SGTK_DIST_SINGLE_HOT) {
        if (group == 0 && (row % 5) == 0) return 1000000 - (row % 97);
        if ((row % 13) == 0) return 10000 + (group % 257);
    }

    return rng.uniform_int(-1000000, 1000000);
}

static int32_t make_value(SplitMix64& rng, int row, int keep_permille) {
    if (!rng.chance_permille(keep_permille)) {
        return -rng.uniform_int(0, 1000);
    }

    int32_t v = static_cast<int32_t>(rng.uniform_int(1, 100000));

    if ((row % 257) == 0) v = INT_MAX - (row & 1023);
    if ((row % 389) == 0) v = 1;
    if ((row % 1021) == 0) v = 777;

    return v;
}

static HostCase make_case(
    const char* name,
    int N,
    int G,
    int M,
    int distribution_id,
    int keep_permille,
    uint64_t seed,
    bool single_hot) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = SGTK_ABI_VERSION;
    hc.run.N = N;
    hc.run.G = G;
    hc.run.M = M;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!sgtk_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid SgtRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.group_id.resize((size_t)N);
    hc.key.resize((size_t)N);
    hc.value.resize((size_t)N);

    for (int i = 0; i < N; ++i) {
        const int g = choose_group(rng, i, G, distribution_id, single_hot);
        hc.group_id[(size_t)i] = g;
        hc.key[(size_t)i] = make_key(rng, i, g, distribution_id);
        hc.value[(size_t)i] = make_value(rng, i, keep_permille);
    }

    if (single_hot) {
        for (int i = 0; i < std::min(N, 2048); ++i) {
            hc.group_id[(size_t)i] = 0;
            hc.value[(size_t)i] = 1 + (i % 31);
            hc.key[(size_t)i] = 1000000 - (i % 17);
        }
    }

    return hc;
}

struct CaseSpec {
    const char* name;
    int N;
    int G;
    int M;
    int distribution_id;
    int keep_permille;
    bool single_hot;
};

static const CaseSpec kCaseSpecs[] = {
    {"uniform_N65536_G64_M16", 65536, 64, 16, SGTK_DIST_UNIFORM, 700, false},
    {"zipf_N131072_G1024_M64", 131072, 1024, 64, SGTK_DIST_ZIPF_HOT, 650, false},
    {"hot_N131072_G1024_M64", 131072, 1024, 64, SGTK_DIST_SINGLE_HOT, 900, true},
    {"ties_N65536_G8_M16", 65536, 8, 16, SGTK_DIST_MANY_TIES, 1000, false},
};

static constexpr int kNumCases = sizeof(kCaseSpecs) / sizeof(kCaseSpecs[0]);

// Every timed call gets its own input variant: variants[c][k] for timed iteration k,
// plus variants[c][K] reserved for warmup so warmup never touches a digested region.
// Shape family (N / G / M / distribution) is fixed per case; only data varies.
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
                cs.name, cs.N, cs.G, cs.M, cs.distribution_id, cs.keep_permille,
                vs.next_u64(), cs.single_hot));
        }
    }

    return variants;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int G = hc.run.G;

    dc->group_id.allocate(hc.group_id.size());
    dc->key.allocate(hc.key.size());
    dc->value.allocate(hc.value.size());

    dc->group_id.upload(hc.group_id);
    dc->key.upload(hc.key);
    dc->value.upload(hc.value);

    dc->group_counts.allocate((size_t)G);
    dc->group_offsets.allocate((size_t)G + 1);
    dc->packed_topk_origidx.allocate((size_t)N);
    dc->per_group_sum.allocate((size_t)G);
    dc->per_group_max.allocate((size_t)G);
    dc->per_group_argmax.allocate((size_t)G);
    dc->kept_count.allocate((size_t)G);

    dc->inputs = {};
    dc->inputs.group_id = dc->group_id.ptr;
    dc->inputs.key = dc->key.ptr;
    dc->inputs.value = dc->value.ptr;

    dc->outputs = {};
    dc->outputs.group_counts = dc->group_counts.ptr;
    dc->outputs.group_offsets = dc->group_offsets.ptr;
    dc->outputs.packed_topk_origidx = dc->packed_topk_origidx.ptr;
    dc->outputs.per_group_sum = dc->per_group_sum.ptr;
    dc->outputs.per_group_max = dc->per_group_max.ptr;
    dc->outputs.per_group_argmax = dc->per_group_argmax.ptr;
    dc->outputs.kept_count = dc->kept_count.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        std::vector<std::vector<HostCase>> host_variants = build_bench_variants(iters);

        int max_N = SGTK_MIN_N;
        int max_G = 8;

        for (int c = 0; c < kNumCases; ++c) {
            max_N = std::max(max_N, kCaseSpecs[c].N);
            max_G = std::max(max_G, kCaseSpecs[c].G);
        }

        SgtProblemSpec spec = {};
        spec.abi_version = SGTK_ABI_VERSION;
        spec.max_N = max_N;
        spec.max_G = max_G;
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

        std::vector<std::vector<DeviceCase*>> cases(kNumCases);

        for (int c = 0; c < kNumCases; ++c) {
            cases[c].reserve(host_variants[c].size());
            for (const HostCase& hc : host_variants[c]) {
                cases[c].push_back(make_device_case(hc));
            }
            const HostCase& hc = host_variants[c].front();
            std::printf(
                "bench_case %-26s N=%d G=%d M=%d dist=%d variants=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.G,
                hc.run.M,
                hc.run.distribution_id,
                iters);
        }

        // Warmup runs only the dedicated variant (index iters); its outputs are never
        // folded, so warmup work cannot pre-populate any digested output region.
        for (int warm = 0; warm < 5; ++warm) {
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

        // Anti-hack digest: every timed call (iter, case) wrote its own output region
        // for its own input variant; fold ALL of them so a no-op/cached timed call
        // leaves a wrong region and the digest cannot match the reference bench.
        // Per-region policy unchanged: packed_topk_origidx is graded only on
        // [0, group_offsets[G]), so fold exactly that prefix.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        for (int iter = 0; iter < iters; ++iter) {
            for (int c = 0; c < kNumCases; ++c) {
                DeviceCase* dc = cases[c][static_cast<size_t>(iter)];
                const size_t N = (size_t)dc->host.run.N;
                const size_t G = (size_t)dc->host.run.G;

                dg.dev(dc->group_counts.ptr, G * sizeof(int32_t));
                dg.dev(dc->group_offsets.ptr, (G + 1) * sizeof(int32_t));
                dg.dev(dc->per_group_sum.ptr, G * sizeof(int64_t));
                dg.dev(dc->per_group_max.ptr, G * sizeof(int32_t));
                dg.dev(dc->per_group_argmax.ptr, G * sizeof(int32_t));
                dg.dev(dc->kept_count.ptr, G * sizeof(int32_t));

                int32_t total_kept = 0;
                CUDA_CHECK(cudaMemcpy(&total_kept, dc->group_offsets.ptr + G,
                                      sizeof(int32_t), cudaMemcpyDeviceToHost));
                if (total_kept < 0) total_kept = 0;
                if ((size_t)total_kept > N) total_kept = (int32_t)N;
                dg.dev(dc->packed_topk_origidx.ptr, (size_t)total_kept * sizeof(int32_t));
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
