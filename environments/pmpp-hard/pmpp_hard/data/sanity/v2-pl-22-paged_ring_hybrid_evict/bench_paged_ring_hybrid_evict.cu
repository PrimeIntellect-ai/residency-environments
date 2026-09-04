// file: bench_paged_ring_hybrid_evict.cu

#include "paged_ring_hybrid_evict_common.h"

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

static constexpr uint64_t g_state = 0xb70c9b2e44a64d13ULL;

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

    int32_t next_i32() {
        return static_cast<int32_t>(next_u64() >> 32);
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
    PrheRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> append_count;
    std::vector<int32_t> token_values;
};

struct DeviceStep {
    StepHost host;

    DeviceBuffer<int32_t> active_seq;
    DeviceBuffer<int32_t> append_count;
    DeviceBuffer<int32_t> token_values;

    DeviceBuffer<int32_t> live_count;
    DeviceBuffer<int64_t> live_sum;
    DeviceBuffer<uint64_t> live_hash;
    DeviceBuffer<uint64_t> page_table_checksum;
    DeviceBuffer<int32_t> evicted_count;
    DeviceBuffer<int32_t> free_pages;

    PrheInputs inputs;
    PrheOutputs outputs;
};

static PrheProblemSpec make_bench_spec() {
    PrheProblemSpec spec = {};
    spec.abi_version = PRHE_ABI_VERSION;
    spec.B = 8;
    spec.max_len = 256;
    spec.page_size = 8;
    spec.window_size = 12;
    spec.max_pages = 24;
    spec.max_active = 4;
    spec.max_new_tokens = 4;
    spec.max_steps = 48;
    spec.flags = 0;

    if (!prhe_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid benchmark PrheProblemSpec");
    }

    return spec;
}

static StepHost make_step(
    const PrheProblemSpec& spec,
    int step_id,
    const std::vector<int32_t>& seqs,
    const std::vector<int32_t>& counts,
    SplitMix64& rng) {
    if (seqs.size() != counts.size()) {
        throw std::runtime_error("step vector size mismatch");
    }

    const int active_count = static_cast<int>(seqs.size());

    StepHost step;
    step.run = {};
    step.run.abi_version = PRHE_ABI_VERSION;
    step.run.active_count = active_count;
    step.run.step_id = step_id;

    if (!prhe_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid benchmark PrheRunSpec");
    }

    const size_t rows = std::max<size_t>(1, (size_t)active_count);

    step.active_seq.assign(rows, 0);
    step.append_count.assign(rows, 0);
    step.token_values.assign(rows * (size_t)spec.max_new_tokens, 0);

    for (int r = 0; r < active_count; ++r) {
        step.active_seq[(size_t)r] = seqs[(size_t)r];
        step.append_count[(size_t)r] = counts[(size_t)r];

        for (int i = 0; i < spec.max_new_tokens; ++i) {
            int32_t v = rng.next_i32();
            if (v == 0) v = step_id * 4099 + r * 37 + i + 1;
            step.token_values[(size_t)r * (size_t)spec.max_new_tokens + (size_t)i] = v;
        }
    }

    return step;
}

static std::vector<StepHost> build_bench_steps(const PrheProblemSpec& spec) {
    SplitMix64 rng(g_state ^ 0x55555555ULL);
    std::vector<StepHost> steps;

    for (int s = 0; s < 48; ++s) {
        std::vector<int32_t> seqs;
        std::vector<int32_t> counts;

        if (s == 7 || s == 23) {
            seqs = {0, 1, 2, 3};
            counts = {0, 0, 0, 0};
        } else {
            const int base = (s * 3) % spec.B;
            seqs = {
                base,
                (base + 1) % spec.B,
                (base + 3) % spec.B,
                (base + 5) % spec.B
            };

            counts = {
                4,
                (s % 3 == 0) ? 2 : 4,
                (s % 4 == 0) ? 1 : 4,
                (s % 5 == 0) ? 0 : 3
            };
        }

        steps.push_back(make_step(spec, s, seqs, counts, rng));
    }

    return steps;
}

static DeviceStep* make_device_step(const StepHost& h, int B) {
    DeviceStep* ds = new DeviceStep();
    ds->host = h;

    ds->active_seq.allocate(h.active_seq.size());
    ds->append_count.allocate(h.append_count.size());
    ds->token_values.allocate(h.token_values.size());

    ds->active_seq.upload(h.active_seq);
    ds->append_count.upload(h.append_count);
    ds->token_values.upload(h.token_values);

    ds->live_count.allocate((size_t)B);
    ds->live_sum.allocate((size_t)B);
    ds->live_hash.allocate((size_t)B);
    ds->page_table_checksum.allocate(1);
    ds->evicted_count.allocate(1);
    ds->free_pages.allocate(1);

    ds->inputs = {};
    ds->inputs.active_seq = ds->active_seq.ptr;
    ds->inputs.append_count = ds->append_count.ptr;
    ds->inputs.token_values = ds->token_values.ptr;

    ds->outputs = {};
    ds->outputs.live_count = ds->live_count.ptr;
    ds->outputs.live_sum = ds->live_sum.ptr;
    ds->outputs.live_hash = ds->live_hash.ptr;
    ds->outputs.page_table_checksum = ds->page_table_checksum.ptr;
    ds->outputs.evicted_count = ds->evicted_count.ptr;
    ds->outputs.free_pages = ds->free_pages.ptr;

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

        const PrheProblemSpec spec = make_bench_spec();

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

        std::vector<StepHost> host_steps = build_bench_steps(spec);
        std::vector<DeviceStep*> steps;
        steps.reserve(host_steps.size());

        for (const StepHost& h : host_steps) {
            steps.push_back(make_device_step(h, spec.B));
        }

        std::printf(
            "bench_sequence B=%d max_len=%d page_size=%d window=%d pages=%d T=%zu\n",
            spec.B,
            spec.max_len,
            spec.page_size,
            spec.window_size,
            spec.max_pages,
            steps.size());

        for (int warm = 0; warm < 3; ++warm) {
            CUDA_CHECK(solution_reset(state, stream));
            run_sequence(state, workspace.ptr, workspace_bytes, stream, steps);
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
            run_sequence(state, workspace.ptr, workspace_bytes, stream, steps);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += static_cast<double>(ms);
        }

        const double avg_ms = total_ms / static_cast<double>(iters);
        std::printf("avg_ms=%.6f\n", avg_ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (DeviceStep* ds : steps) {
            delete ds;
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
