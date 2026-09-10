// file: bench_streaming_dedup_window.cu

#include "streaming_dedup_window_common.h"
#include "pmpp_bench_digest.cuh"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0x3a9f12dc77e4b681ULL;

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

struct StepHost {
    SdwRunSpec run;
    std::vector<int32_t> key;
    std::vector<int32_t> value;
};

struct DeviceStep {
    StepHost host;

    DeviceBuffer<int32_t> key;
    DeviceBuffer<int32_t> value;

    DeviceBuffer<int32_t> active_count;
    DeviceBuffer<int32_t> num_new;
    DeviceBuffer<int32_t> num_dup;
    DeviceBuffer<int32_t> num_evicted;
    DeviceBuffer<uint64_t> evicted_key_checksum;
    DeviceBuffer<int64_t> live_agg_sum;
    DeviceBuffer<uint64_t> state_checksum;

    SdwInputs inputs;
    SdwOutputs outputs;
};

static SdwProblemSpec make_bench_spec() {
    SdwProblemSpec spec = {};
    spec.abi_version = SDW_ABI_VERSION;
    spec.key_space = 4096;
    spec.capacity = 512;
    spec.window_size = 512;
    spec.max_batch = 256;
    spec.max_steps = 48;
    spec.flags = 0;

    if (!sdw_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid benchmark SdwProblemSpec");
    }

    return spec;
}

static StepHost make_step(
    const SdwProblemSpec& spec,
    int step_id,
    int batch_size,
    int mode,
    SplitMix64& rng) {
    StepHost step;
    step.run = {};
    step.run.abi_version = SDW_ABI_VERSION;
    step.run.batch_size = batch_size;
    step.run.step_id = step_id;

    if (!sdw_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid benchmark SdwRunSpec");
    }

    const size_t rows = std::max<size_t>(1, (size_t)batch_size);

    step.key.assign(rows, 0);
    step.value.assign(rows, 0);

    for (int i = 0; i < batch_size; ++i) {
        int key = 0;

        if (mode == 0) {
            key = rng.uniform_int(0, spec.key_space - 1);
        } else if (mode == 1) {
            key = rng.uniform_int(0, 63);
        } else if (mode == 2) {
            key = (step_id * 257 + i) % spec.key_space;
        } else if (mode == 3) {
            key = (i % 5 == 0) ? -1 : rng.uniform_int(0, 127);
        } else {
            key = rng.uniform_int(0, 15);
        }

        int32_t value = rng.next_i32();
        if (value == 0) value = step_id * 4099 + i + 1;

        step.key[(size_t)i] = key;
        step.value[(size_t)i] = value;
    }

    return step;
}

// One sequence variant per timed iteration (plus one for warmup): same mode/batch
// step schedule every variant (timing comparability), but key/value DATA is derived
// from (PMPP_BENCH_SEED, variant). Every timed call thus receives fresh inputs, and
// each iteration's outputs are folded into out_fnv below, so replaying an output
// cached from warmup or an earlier iteration leaves a stale region in the digest
// → perf FAIL.
static std::vector<StepHost> build_bench_sequence(const SdwProblemSpec& spec,
                                                  uint64_t variant_seed) {
    SplitMix64 rng(variant_seed ^ 0x50505050ULL);
    std::vector<StepHost> steps;
    steps.reserve(48);

    for (int s = 0; s < 48; ++s) {
        int batch = 128;
        int mode = 1;

        if (s == 13 || s == 31) {
            batch = 0;
            mode = 0;
        } else if (s % 11 == 0) {
            batch = 256;
            mode = 2;  // capacity pressure
        } else if (s % 7 == 0) {
            batch = 256;
            mode = 3;  // invalid events advance time
        } else if (s % 5 == 0) {
            batch = 192;
            mode = 4;  // very high duplicate rate
        } else {
            batch = 160;
            mode = 1;  // duplicate-heavy
        }

        steps.push_back(make_step(spec, s, batch, mode, rng));
    }

    return steps;
}

static DeviceStep* make_device_step(const StepHost& h) {
    DeviceStep* ds = new DeviceStep();
    ds->host = h;

    ds->key.allocate(h.key.size());
    ds->value.allocate(h.value.size());

    ds->key.upload(h.key);
    ds->value.upload(h.value);

    ds->active_count.allocate(1);
    ds->num_new.allocate(1);
    ds->num_dup.allocate(1);
    ds->num_evicted.allocate(1);
    ds->evicted_key_checksum.allocate(1);
    ds->live_agg_sum.allocate(1);
    ds->state_checksum.allocate(1);

    ds->inputs = {};
    ds->inputs.key = ds->key.ptr;
    ds->inputs.value = ds->value.ptr;

    ds->outputs = {};
    ds->outputs.active_count = ds->active_count.ptr;
    ds->outputs.num_new = ds->num_new.ptr;
    ds->outputs.num_dup = ds->num_dup.ptr;
    ds->outputs.num_evicted = ds->num_evicted.ptr;
    ds->outputs.evicted_key_checksum = ds->evicted_key_checksum.ptr;
    ds->outputs.live_agg_sum = ds->live_agg_sum.ptr;
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
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        const SdwProblemSpec spec = make_bench_spec();

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

        // Variant 0 is warmup-only; variants 1..iters feed the timed iterations.
        // Per-variant seeds come from a SplitMix64 stream over PMPP_BENCH_SEED.
        SplitMix64 vseed_rng(pmpp::bench_seed(g_state) ^ 0x700000000ULL);

        std::vector<std::vector<DeviceStep*>> variants;
        variants.reserve((size_t)iters + 1);
        for (int v = 0; v <= iters; ++v) {
            std::vector<StepHost> host_steps =
                build_bench_sequence(spec, vseed_rng.next_u64());
            std::vector<DeviceStep*> steps;
            steps.reserve(host_steps.size());
            for (const StepHost& h : host_steps) {
                steps.push_back(make_device_step(h));
            }
            variants.push_back(std::move(steps));
        }

        std::printf(
            "bench_sequence key_space=%d capacity=%d window=%d max_batch=%d T=%zu\n",
            spec.key_space,
            spec.capacity,
            spec.window_size,
            spec.max_batch,
            variants[0].size());

        for (int warm = 0; warm < 3; ++warm) {
            CUDA_CHECK(solution_reset(state, stream));
            run_sequence(state, workspace.ptr, workspace_bytes, stream, variants[0]);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double total_ms = 0.0;

        // Fold each timed iteration's graded outputs (outside the timed region) so a
        // stale/replayed output on any iteration breaks the digest.
        pmpp::OutFnv dg;

        for (int iter = 0; iter < iters; ++iter) {
            const std::vector<DeviceStep*>& steps = variants[(size_t)iter + 1];

            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaEventRecord(start, stream));
            run_sequence(state, workspace.ptr, workspace_bytes, stream, steps);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += static_cast<double>(ms);

            for (DeviceStep* ds : steps) {
                dg.dev(ds->active_count.ptr, sizeof(int32_t));
                dg.dev(ds->num_new.ptr, sizeof(int32_t));
                dg.dev(ds->num_dup.ptr, sizeof(int32_t));
                dg.dev(ds->num_evicted.ptr, sizeof(int32_t));
                dg.dev(ds->evicted_key_checksum.ptr, sizeof(uint64_t));
                dg.dev(ds->live_agg_sum.ptr, sizeof(int64_t));
                dg.dev(ds->state_checksum.ptr, sizeof(uint64_t));
            }
        }

        const double avg_ms = total_ms / static_cast<double>(iters);
        std::printf("avg_ms=%.6f\n", avg_ms);
        dg.print();

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (std::vector<DeviceStep*>& steps : variants) {
            for (DeviceStep* ds : steps) {
                delete ds;
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
