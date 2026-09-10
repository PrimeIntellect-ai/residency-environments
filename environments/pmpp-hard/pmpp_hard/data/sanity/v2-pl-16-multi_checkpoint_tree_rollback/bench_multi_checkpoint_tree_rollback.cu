// ============================================================================
// file: bench_multi_checkpoint_tree_rollback.cu
// ============================================================================

#include "multi_checkpoint_tree_rollback_common.h"

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

static constexpr uint64_t g_state = 0x5e0f8d7c3b2a1906ULL;

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
    MctrRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> op_code;
    std::vector<int32_t> token_count;
    std::vector<int32_t> accept_count;
    std::vector<int32_t> checkpoint_id;
    std::vector<int32_t> token_values;
    std::vector<int32_t> correction_value;
};

struct DeviceStep {
    StepHost host;

    DeviceBuffer<int32_t> active_seq;
    DeviceBuffer<int32_t> op_code;
    DeviceBuffer<int32_t> token_count;
    DeviceBuffer<int32_t> accept_count;
    DeviceBuffer<int32_t> checkpoint_id;
    DeviceBuffer<int32_t> token_values;
    DeviceBuffer<int32_t> correction_value;

    DeviceBuffer<int64_t> live_cache_sum;
    DeviceBuffer<uint64_t> live_cache_tail_hash;
    DeviceBuffer<int32_t> length;
    DeviceBuffer<int32_t> num_checkpoints;
    DeviceBuffer<int32_t> top_checkpoint_len;
    DeviceBuffer<uint64_t> state_checksum;

    MctrInputs inputs;
    MctrOutputs outputs;
};

static MctrProblemSpec make_spec() {
    MctrProblemSpec spec = {};
    spec.abi_version = MCTR_ABI_VERSION;
    spec.B = 32;
    spec.max_len = 256;
    spec.max_steps = 64;
    spec.max_depth = 8;
    spec.flags = 0;

    if (!mctr_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid benchmark MctrProblemSpec");
    }

    return spec;
}

static StepHost make_step(
    const MctrProblemSpec& spec,
    int step_id,
    int active_count,
    SplitMix64& rng) {
    StepHost step;
    step.run = {};
    step.run.abi_version = MCTR_ABI_VERSION;
    step.run.active_count = active_count;
    step.run.step_id = step_id;

    if (!mctr_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid benchmark MctrRunSpec");
    }

    const size_t rows = std::max<size_t>(1, (size_t)active_count);

    step.active_seq.assign(rows, 0);
    step.op_code.assign(rows, MCTR_OP_NOOP);
    step.token_count.assign(rows, 0);
    step.accept_count.assign(rows, 0);
    step.checkpoint_id.assign(rows, 0);
    step.token_values.assign(rows * (size_t)MCTR_MAX_K, 0);
    step.correction_value.assign(rows, 0);

    for (int r = 0; r < active_count; ++r) {
        const int seq = (step_id * 7 + r * 5) % spec.B;
        const int mode = (step_id * 11 + r * 3) % 6;

        step.active_seq[(size_t)r] = seq;

        if (mode == 0) {
            step.op_code[(size_t)r] = MCTR_OP_APPEND;
            step.token_count[(size_t)r] = 4;
        } else if (mode == 1) {
            step.op_code[(size_t)r] = MCTR_OP_SAVE_CHECKPOINT;
            step.checkpoint_id[(size_t)r] = 1000 + seq * 16 + (step_id % 8);
        } else if (mode == 2) {
            step.op_code[(size_t)r] = MCTR_OP_ACCEPT_PREFIX;
            step.token_count[(size_t)r] = 4;
            step.accept_count[(size_t)r] = (step_id + r) % 5;
        } else if (mode == 3) {
            step.op_code[(size_t)r] = MCTR_OP_ROLLBACK_TO;
            step.checkpoint_id[(size_t)r] = 1000 + seq * 16 + ((step_id + 5) % 8);
        } else if (mode == 4) {
            step.op_code[(size_t)r] = MCTR_OP_ROLLBACK_TO;
            step.checkpoint_id[(size_t)r] = -777;
        } else {
            step.op_code[(size_t)r] = MCTR_OP_NOOP;
        }

        for (int i = 0; i < MCTR_MAX_K; ++i) {
            int32_t v = rng.next_i32();
            if (v == 0) v = step_id * 1009 + r * 17 + i + 1;
            step.token_values[(size_t)r * (size_t)MCTR_MAX_K + (size_t)i] = v;
        }

        step.correction_value[(size_t)r] = rng.next_i32();
    }

    return step;
}

static std::vector<StepHost> build_bench_sequence(const MctrProblemSpec& spec) {
    SplitMix64 rng(g_state ^ 0x80808080ULL);
    std::vector<StepHost> steps;

    for (int s = 0; s < 48; ++s) {
        int active = 8 + (s % 9);
        if (s == 17) active = 0;
        steps.push_back(make_step(spec, s, active, rng));
    }

    return steps;
}

static DeviceStep* make_device_step(const StepHost& h, int B) {
    DeviceStep* ds = new DeviceStep();
    ds->host = h;

    ds->active_seq.allocate(h.active_seq.size());
    ds->op_code.allocate(h.op_code.size());
    ds->token_count.allocate(h.token_count.size());
    ds->accept_count.allocate(h.accept_count.size());
    ds->checkpoint_id.allocate(h.checkpoint_id.size());
    ds->token_values.allocate(h.token_values.size());
    ds->correction_value.allocate(h.correction_value.size());

    ds->active_seq.upload(h.active_seq);
    ds->op_code.upload(h.op_code);
    ds->token_count.upload(h.token_count);
    ds->accept_count.upload(h.accept_count);
    ds->checkpoint_id.upload(h.checkpoint_id);
    ds->token_values.upload(h.token_values);
    ds->correction_value.upload(h.correction_value);

    ds->live_cache_sum.allocate((size_t)B);
    ds->live_cache_tail_hash.allocate((size_t)B);
    ds->length.allocate((size_t)B);
    ds->num_checkpoints.allocate((size_t)B);
    ds->top_checkpoint_len.allocate((size_t)B);
    ds->state_checksum.allocate(1);

    ds->inputs = {};
    ds->inputs.active_seq = ds->active_seq.ptr;
    ds->inputs.op_code = ds->op_code.ptr;
    ds->inputs.token_count = ds->token_count.ptr;
    ds->inputs.accept_count = ds->accept_count.ptr;
    ds->inputs.checkpoint_id = ds->checkpoint_id.ptr;
    ds->inputs.token_values = ds->token_values.ptr;
    ds->inputs.correction_value = ds->correction_value.ptr;

    ds->outputs = {};
    ds->outputs.live_cache_sum = ds->live_cache_sum.ptr;
    ds->outputs.live_cache_tail_hash = ds->live_cache_tail_hash.ptr;
    ds->outputs.length = ds->length.ptr;
    ds->outputs.num_checkpoints = ds->num_checkpoints.ptr;
    ds->outputs.top_checkpoint_len = ds->top_checkpoint_len.ptr;
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

        const MctrProblemSpec spec = make_spec();

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

        std::vector<StepHost> host_steps = build_bench_sequence(spec);
        std::vector<DeviceStep*> steps;
        steps.reserve(host_steps.size());

        for (const StepHost& h : host_steps) {
            steps.push_back(make_device_step(h, spec.B));
        }

        std::printf(
            "bench_sequence B=%d max_len=%d max_steps=%d max_depth=%d T=%zu\n",
            spec.B,
            spec.max_len,
            spec.max_steps,
            spec.max_depth,
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
