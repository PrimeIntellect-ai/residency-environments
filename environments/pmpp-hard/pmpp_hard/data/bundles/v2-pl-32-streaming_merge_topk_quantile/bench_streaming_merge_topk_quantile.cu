// file: bench_streaming_merge_topk_quantile.cu

#include "streaming_merge_topk_quantile_common.h"
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

static constexpr uint64_t g_state = 0x7c91b3a42d8f0e65ULL;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                              \
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

    int32_t next_i32() {
        return static_cast<int32_t>(next_u64() >> 32);
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
        if (n > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
    }

    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("DeviceBuffer upload size mismatch");
        if (count > 0) CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
    }
};

struct StepHost {
    SmtqRunSpec run;
    std::vector<int32_t> group;
    std::vector<int32_t> key;
    std::vector<int32_t> value;
};

struct DeviceStep {
    StepHost host;

    DeviceBuffer<int32_t> group;
    DeviceBuffer<int32_t> key;
    DeviceBuffer<int32_t> value;

    DeviceBuffer<int32_t> topk_keys;
    DeviceBuffer<int32_t> topk_values;
    DeviceBuffer<int32_t> topk_count;
    DeviceBuffer<int64_t> topk_value_sum;
    DeviceBuffer<uint64_t> histogram_checksum;
    DeviceBuffer<int32_t> quantile_key;
    DeviceBuffer<int64_t> total_ingested;
    DeviceBuffer<uint64_t> state_checksum;

    SmtqInputs inputs;
    SmtqOutputs outputs;
};

static SmtqProblemSpec make_bench_spec() {
    SmtqProblemSpec spec = {};
    spec.abi_version = SMTQ_ABI_VERSION;
    spec.G = 64;
    spec.K = 16;
    spec.num_bins = 256;
    spec.key_min = -10000;
    spec.key_max = 10000;
    spec.max_batch = 256;
    spec.max_steps = 48;
    spec.flags = 0;

    if (!smtq_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid benchmark SmtqProblemSpec");
    }

    return spec;
}

static int choose_group(SplitMix64& rng, const SmtqProblemSpec& spec, int row, int mode) {
    if (mode == 1) {
        const int r = rng.uniform_int(0, 999);
        if (r < 570) return 0;
        if (r < 760) return std::min(1, spec.G - 1);
        if (r < 860) return std::min(2, spec.G - 1);
        if (r < 930) return rng.uniform_int(0, std::min(spec.G - 1, 15));
        return rng.uniform_int(0, spec.G - 1);
    }

    if (mode == 2) return row % std::min(spec.G, 8);

    return rng.uniform_int(0, spec.G - 1);
}

static int32_t choose_key(SplitMix64& rng, const SmtqProblemSpec& spec, int row, int group, int mode) {
    if (mode == 2) {
        const int width = std::max(1, (spec.key_max - spec.key_min + 1) / std::max(1, spec.num_bins));
        const int bin = (row / 7 + group * 11) % spec.num_bins;
        return spec.key_min + bin * width;
    }

    if (mode == 1 && group == 0) {
        if ((row % 7) == 0) return spec.key_max - (row % 31);
        if ((row % 11) == 0) return spec.key_min + (row % 127);
    }

    if ((row % 97) == 0) return spec.key_min;
    if ((row % 131) == 0) return spec.key_max;

    return rng.uniform_int(spec.key_min, spec.key_max);
}

static StepHost make_step(
    const SmtqProblemSpec& spec,
    int step_id,
    int batch_size,
    int mode,
    bool is_query,
    int q_num,
    int q_den,
    SplitMix64& rng) {
    StepHost step;
    step.run = {};
    step.run.abi_version = SMTQ_ABI_VERSION;
    step.run.batch_size = batch_size;
    step.run.is_query = is_query ? 1 : 0;
    step.run.q_num = q_num;
    step.run.q_den = q_den;
    step.run.step_id = step_id;

    if (!smtq_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid benchmark SmtqRunSpec");
    }

    const size_t rows = std::max<size_t>(1, (size_t)batch_size);
    step.group.assign(rows, -1);
    step.key.assign(rows, 0);
    step.value.assign(rows, 0);

    for (int i = 0; i < batch_size; ++i) {
        if ((i % 37) == 0 && (step_id % 3) == 0) {
            step.group[(size_t)i] = (i & 1) ? -1 : spec.G + 3;
            step.key[(size_t)i] = choose_key(rng, spec, i + step_id * 997, 0, mode);
            step.value[(size_t)i] = rng.next_i32();
            continue;
        }

        const int g = choose_group(rng, spec, i + step_id * 17, mode);
        step.group[(size_t)i] = g;
        step.key[(size_t)i] = choose_key(rng, spec, i + step_id * 4099, g, mode);

        int32_t v = rng.next_i32();
        if (v == 0) v = step_id * 8191 + i + 1;
        step.value[(size_t)i] = v;
    }

    return step;
}

// Every timed iteration replays the SAME 48-step schedule (batch sizes / query
// cadence / modes fixed for timing comparability) but with its own episode of
// step VALUES: episodes[k] for timed iteration k, plus episodes[K] reserved for
// warmup so warmup never touches a digested region. Episode value streams come
// from a SplitMix64 stream keyed by PMPP_BENCH_SEED, so the same seed yields a
// bit-identical episode sequence (paired digest compare intact).
static std::vector<std::vector<StepHost>> build_bench_episodes(
    const SmtqProblemSpec& spec,
    int iters) {
    SplitMix64 seed_mix(g_state ^ pmpp::bench_seed(0x50505050ULL));
    std::vector<std::vector<StepHost>> episodes(static_cast<size_t>(iters) + 1);

    for (int ep = 0; ep <= iters; ++ep) {
        SplitMix64 rng(seed_mix.next_u64());
        std::vector<StepHost>& steps = episodes[(size_t)ep];
        steps.reserve(48);

        for (int s = 0; s < 48; ++s) {
            const int mode = (s % 11 == 0) ? 2 : ((s % 3 == 0) ? 1 : 0);
            const int batch = (s == 13 || s == 31) ? 0 : (64 + (s * 19) % 193);
            const bool query = (s % 4 == 0) || (s % 9 == 1);
            const int q_num = (s % 5 == 0) ? 1 : ((s % 5 == 1) ? 9 : 3);
            const int q_den = (s % 5 == 0) ? 2 : ((s % 5 == 1) ? 10 : 4);

            steps.push_back(make_step(spec, s, batch, mode, query, q_num, q_den, rng));
        }
    }

    return episodes;
}

static DeviceStep* make_device_step(const StepHost& h, const SmtqProblemSpec& spec) {
    DeviceStep* ds = new DeviceStep();
    ds->host = h;

    ds->group.allocate(h.group.size());
    ds->key.allocate(h.key.size());
    ds->value.allocate(h.value.size());

    ds->group.upload(h.group);
    ds->key.upload(h.key);
    ds->value.upload(h.value);

    ds->topk_keys.allocate((size_t)spec.G * (size_t)spec.K);
    ds->topk_values.allocate((size_t)spec.G * (size_t)spec.K);
    ds->topk_count.allocate((size_t)spec.G);
    ds->topk_value_sum.allocate((size_t)spec.G);
    ds->histogram_checksum.allocate(1);
    ds->quantile_key.allocate((size_t)spec.G);
    ds->total_ingested.allocate(1);
    ds->state_checksum.allocate(1);

    ds->inputs = {};
    ds->inputs.group = ds->group.ptr;
    ds->inputs.key = ds->key.ptr;
    ds->inputs.value = ds->value.ptr;

    ds->outputs = {};
    ds->outputs.topk_keys = ds->topk_keys.ptr;
    ds->outputs.topk_values = ds->topk_values.ptr;
    ds->outputs.topk_count = ds->topk_count.ptr;
    ds->outputs.topk_value_sum = ds->topk_value_sum.ptr;
    ds->outputs.histogram_checksum = ds->histogram_checksum.ptr;
    ds->outputs.quantile_key = ds->quantile_key.ptr;
    ds->outputs.total_ingested = ds->total_ingested.ptr;
    ds->outputs.state_checksum = ds->state_checksum.ptr;

    return ds;
}

static void run_sequence(
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    const std::vector<DeviceStep*>& steps) {
    for (DeviceStep* ds : steps) {
        CUDA_CHECK(solution_run(
            state,
            &ds->host.run,
            &ds->inputs,
            &ds->outputs,
            workspace,
            workspace_bytes,
            stream));
    }
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        const SmtqProblemSpec spec = make_bench_spec();

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

        const std::vector<std::vector<StepHost>> host_episodes =
            build_bench_episodes(spec, iters);
        std::vector<std::vector<DeviceStep*>> episodes(host_episodes.size());

        for (size_t ep = 0; ep < host_episodes.size(); ++ep) {
            episodes[ep].reserve(host_episodes[ep].size());
            for (const StepHost& h : host_episodes[ep]) {
                episodes[ep].push_back(make_device_step(h, spec));
            }
        }

        std::printf(
            "bench_sequence G=%d K=%d bins=%d max_batch=%d steps=%zu episodes=%d\n",
            spec.G,
            spec.K,
            spec.num_bins,
            spec.max_batch,
            episodes.front().size(),
            iters);

        // Warmup replays only the dedicated episode (index iters); its outputs are
        // never folded, so warmup work cannot pre-populate any digested region.
        for (int warm = 0; warm < 3; ++warm) {
            CUDA_CHECK(solution_reset(state, stream));
            run_sequence(state, workspace.ptr, workspace_bytes, stream,
                         episodes[(size_t)iters]);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double total_ms = 0.0;

        for (int iter = 0; iter < iters; ++iter) {
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaEventRecord(start, stream));
            run_sequence(state, workspace.ptr, workspace_bytes, stream,
                         episodes[(size_t)iter]);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += static_cast<double>(ms);
        }

        std::printf("avg_ms=%.6f\n", total_ms / static_cast<double>(iters));

        // Anti-hack digest: every timed iteration ran its OWN episode into its own
        // per-step output regions; fold every graded field of every step of every
        // timed episode. A cached/no-op step leaves a wrong region and the digest
        // cannot match the reference bench on the same PMPP_BENCH_SEED.
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        for (int iter = 0; iter < iters; ++iter) {
            for (DeviceStep* ds : episodes[(size_t)iter]) {
                dg.dev(ds->topk_keys.ptr, (size_t)spec.G * (size_t)spec.K * sizeof(int32_t));
                dg.dev(ds->topk_values.ptr, (size_t)spec.G * (size_t)spec.K * sizeof(int32_t));
                dg.dev(ds->topk_count.ptr, (size_t)spec.G * sizeof(int32_t));
                dg.dev(ds->topk_value_sum.ptr, (size_t)spec.G * sizeof(int64_t));
                dg.dev(ds->histogram_checksum.ptr, sizeof(uint64_t));
                dg.dev(ds->quantile_key.ptr, (size_t)spec.G * sizeof(int32_t));
                dg.dev(ds->total_ingested.ptr, sizeof(int64_t));
                dg.dev(ds->state_checksum.ptr, sizeof(uint64_t));
            }
        }
        dg.print();

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (auto& eps : episodes) {
            for (DeviceStep* ds : eps) delete ds;
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
