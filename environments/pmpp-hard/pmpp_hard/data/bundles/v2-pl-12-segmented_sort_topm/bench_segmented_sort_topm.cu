// ============================================================================
// file: bench_segmented_sort_topm.cu
// ============================================================================

#include "segmented_sort_topm_common.h"
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

static constexpr uint64_t g_state = 0x91e10da5c79e7b1dULL;

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

    bool chance_permille(int per_mille) {
        if (per_mille <= 0) return false;
        if (per_mille >= 1000) return true;
        return static_cast<int>(next_u64() % 1000ULL) < per_mille;
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
    SstRunSpec run;
    std::vector<int32_t> seg_offsets;
    std::vector<int32_t> item_key;
    std::vector<int32_t> item_value;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<int32_t> seg_offsets;
    DeviceBuffer<int32_t> item_key;
    DeviceBuffer<int32_t> item_value;

    DeviceBuffer<int32_t> topm_count;
    DeviceBuffer<int32_t> topm_offsets;
    DeviceBuffer<int32_t> packed_topm_key;
    DeviceBuffer<int32_t> packed_topm_value;
    DeviceBuffer<int32_t> packed_topm_origidx;
    DeviceBuffer<int64_t> seg_sum;
    DeviceBuffer<int32_t> seg_max;
    DeviceBuffer<int32_t> seg_argmax;

    SstInputs inputs;
    SstOutputs outputs;
};

static void normalize_sizes_to_N(std::vector<int>& sizes, int N) {
    int64_t sum = 0;
    for (int v : sizes) sum += v;

    if (sum < N) {
        sizes[0] += static_cast<int>(N - sum);
    } else if (sum > N) {
        int64_t excess = sum - N;
        for (int i = static_cast<int>(sizes.size()) - 1; i >= 0 && excess > 0; --i) {
            int take = static_cast<int>(std::min<int64_t>(sizes[(size_t)i], excess));
            sizes[(size_t)i] -= take;
            excess -= take;
        }
    }
}

static std::vector<int> make_segment_sizes(
    int N,
    int S,
    int distribution_id,
    SplitMix64& rng) {
    std::vector<int> sizes((size_t)S, 0);

    if (distribution_id == SST_DIST_ALL_SIZE_ONE) {
        for (int s = 0; s < S && s < N; ++s) sizes[(size_t)s] = 1;
        normalize_sizes_to_N(sizes, N);
        return sizes;
    }

    if (distribution_id == SST_DIST_ONE_GIANT) {
        if (N >= S) {
            sizes[0] = N - (S - 1);
            for (int s = 1; s < S; ++s) sizes[(size_t)s] = 1;
        } else {
            for (int s = 0; s < N; ++s) sizes[(size_t)s] = 1;
        }
        normalize_sizes_to_N(sizes, N);
        return sizes;
    }

    if (distribution_id == SST_DIST_POWERLAW) {
        const int hot_count = std::min(S, 8);

        for (int i = 0; i < N; ++i) {
            int s;
            if (rng.chance_permille(850)) {
                int r = rng.uniform_int(0, 99);
                if (r < 55) s = 0;
                else if (r < 75) s = std::min(1, hot_count - 1);
                else if (r < 88) s = std::min(2, hot_count - 1);
                else s = rng.uniform_int(0, hot_count - 1);
            } else {
                s = rng.uniform_int(0, S - 1);
            }
            ++sizes[(size_t)s];
        }
        return sizes;
    }

    int base = N / S;
    int rem = N - base * S;
    for (int s = 0; s < S; ++s) {
        sizes[(size_t)s] = base + (s < rem ? 1 : 0);
    }

    return sizes;
}

static int32_t make_key(
    SplitMix64& rng,
    int distribution_id,
    int seg,
    int orig) {
    if (distribution_id == SST_DIST_MANY_TIES) {
        return static_cast<int32_t>((seg * 13 + orig / 3) % 11);
    }

    if (distribution_id == SST_DIST_POWERLAW) {
        return static_cast<int32_t>(rng.uniform_int(-64, 64));
    }

    if (distribution_id == SST_DIST_ONE_GIANT) {
        if (orig % 5 == 0) return 1000;
        return static_cast<int32_t>(rng.uniform_int(-1000, 1000));
    }

    return static_cast<int32_t>(rng.uniform_int(-1000000, 1000000));
}

static int32_t make_value(SplitMix64& rng, int global_idx, int seg, int orig) {
    int v = rng.uniform_int(-100000, 100000);
    if (global_idx % 257 == 0) v = 2147483000 - (orig % 1024);
    if (global_idx % 389 == 0) v = -2147483000 + (seg % 1024);
    return static_cast<int32_t>(v);
}

static HostCase make_case(
    const char* name,
    int N,
    int S,
    int M,
    int distribution_id,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = SST_ABI_VERSION;
    hc.run.N = N;
    hc.run.S = S;
    hc.run.M = M;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!sst_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated SstRunSpec");
    }

    SplitMix64 rng(g_state ^ seed);
    std::vector<int> sizes = make_segment_sizes(N, S, distribution_id, rng);

    hc.seg_offsets.resize((size_t)S + 1);
    hc.seg_offsets[0] = 0;

    for (int s = 0; s < S; ++s) {
        hc.seg_offsets[(size_t)s + 1] =
            hc.seg_offsets[(size_t)s] + static_cast<int32_t>(sizes[(size_t)s]);
    }

    if (hc.seg_offsets[(size_t)S] != N) {
        throw std::runtime_error("segment size generator failed to sum to N");
    }

    hc.item_key.resize((size_t)N);
    hc.item_value.resize((size_t)N);

    for (int s = 0; s < S; ++s) {
        const int begin = hc.seg_offsets[(size_t)s];
        const int end = hc.seg_offsets[(size_t)s + 1];

        for (int idx = begin; idx < end; ++idx) {
            const int orig = idx - begin;
            hc.item_key[(size_t)idx] = make_key(rng, distribution_id, s, orig);
            hc.item_value[(size_t)idx] = make_value(rng, idx, s, orig);
        }
    }

    return hc;
}

struct CaseSpec {
    const char* name;
    int N;
    int S;
    int M;
    int distribution_id;
};

static const CaseSpec kCaseSpecs[] = {
    {"uniform_N32768_S256_M16", 32768, 256, 16, SST_DIST_UNIFORM},
    {"powerlaw_N65536_S256_M64", 65536, 256, 64, SST_DIST_POWERLAW},
    {"one_giant_N131072_S4096_M64", 131072, 4096, 64, SST_DIST_ONE_GIANT},
    {"many_ties_N65536_S4096_M16", 65536, 4096, 16, SST_DIST_MANY_TIES},
};

static constexpr int kNumCases = sizeof(kCaseSpecs) / sizeof(kCaseSpecs[0]);

// Every timed call gets its own input variant: variants[c][k] for timed iteration k,
// plus variants[c][K] reserved for warmup so warmup never touches a digested region.
// Shape family (N / S / M / distribution) is fixed per case; only data varies.
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
                cs.name, cs.N, cs.S, cs.M, cs.distribution_id, vs.next_u64()));
        }
    }

    return variants;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int S = hc.run.S;

    dc->seg_offsets.allocate(hc.seg_offsets.size());
    dc->item_key.allocate(hc.item_key.size());
    dc->item_value.allocate(hc.item_value.size());

    dc->seg_offsets.upload(hc.seg_offsets);
    dc->item_key.upload(hc.item_key);
    dc->item_value.upload(hc.item_value);

    dc->topm_count.allocate((size_t)S);
    dc->topm_offsets.allocate((size_t)S + 1);
    dc->packed_topm_key.allocate((size_t)N);
    dc->packed_topm_value.allocate((size_t)N);
    dc->packed_topm_origidx.allocate((size_t)N);
    dc->seg_sum.allocate((size_t)S);
    dc->seg_max.allocate((size_t)S);
    dc->seg_argmax.allocate((size_t)S);

    dc->inputs = {};
    dc->inputs.seg_offsets = dc->seg_offsets.ptr;
    dc->inputs.item_key = dc->item_key.ptr;
    dc->inputs.item_value = dc->item_value.ptr;

    dc->outputs = {};
    dc->outputs.topm_count = dc->topm_count.ptr;
    dc->outputs.topm_offsets = dc->topm_offsets.ptr;
    dc->outputs.packed_topm_key = dc->packed_topm_key.ptr;
    dc->outputs.packed_topm_value = dc->packed_topm_value.ptr;
    dc->outputs.packed_topm_origidx = dc->packed_topm_origidx.ptr;
    dc->outputs.seg_sum = dc->seg_sum.ptr;
    dc->outputs.seg_max = dc->seg_max.ptr;
    dc->outputs.seg_argmax = dc->seg_argmax.ptr;

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

        int max_N = 0;
        int max_S = 32;

        for (int c = 0; c < kNumCases; ++c) {
            max_N = std::max(max_N, kCaseSpecs[c].N);
            max_S = std::max(max_S, kCaseSpecs[c].S);
        }

        SstProblemSpec spec = {};
        spec.abi_version = SST_ABI_VERSION;
        spec.max_N = max_N;
        spec.max_S = max_S;
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
                "bench_case %-32s N=%d S=%d M=%d dist=%d variants=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.S,
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
        // Per-region policy unchanged: the packed_topm_* arrays are only graded up to
        // topm_offsets[S] (the tail is unspecified), so fold that prefix.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        for (int iter = 0; iter < iters; ++iter) {
            for (int c = 0; c < kNumCases; ++c) {
                DeviceCase* dc = cases[c][static_cast<size_t>(iter)];
                const int S = dc->host.run.S;
                int32_t total_packed = 0;
                CUDA_CHECK(cudaMemcpy(&total_packed, dc->topm_offsets.ptr + S, sizeof(int32_t),
                                      cudaMemcpyDeviceToHost));
                if (total_packed < 0) total_packed = 0;
                if (total_packed > dc->host.run.N) total_packed = dc->host.run.N;
                dg.dev(dc->topm_count.ptr, (size_t)S * sizeof(int32_t));
                dg.dev(dc->topm_offsets.ptr, (size_t)(S + 1) * sizeof(int32_t));
                if (total_packed > 0) {
                    dg.dev(dc->packed_topm_key.ptr, (size_t)total_packed * sizeof(int32_t));
                    dg.dev(dc->packed_topm_value.ptr, (size_t)total_packed * sizeof(int32_t));
                    dg.dev(dc->packed_topm_origidx.ptr, (size_t)total_packed * sizeof(int32_t));
                }
                dg.dev(dc->seg_sum.ptr, (size_t)S * sizeof(int64_t));
                dg.dev(dc->seg_max.ptr, (size_t)S * sizeof(int32_t));
                dg.dev(dc->seg_argmax.ptr, (size_t)S * sizeof(int32_t));
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
