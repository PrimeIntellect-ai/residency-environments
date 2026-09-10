// file: bench_switch_moe_overflow_router.cu

#include "switch_moe_overflow_router_common.h"

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

static constexpr uint64_t g_state = 0x51c0ffee0d15ea5eULL;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                   \
            throw std::runtime_error(_oss.str());                              \
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
    int uniform_int(int lo, int hi) { return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1)); }
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr; size_t count = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) { count = n; if (!n) { ptr = nullptr; return; } CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n)); }
    void upload(const std::vector<T>& h) { if (h.size() != count) throw std::runtime_error("size"); if (count) CUDA_CHECK(cudaMemcpy(ptr, h.data(), sizeof(T) * count, cudaMemcpyHostToDevice)); }
};

struct StepHost {
    SmorRunSpec run;
    std::vector<int32_t> op_kind;
    std::vector<uint64_t> op_a, op_b;
    std::vector<int32_t> op_cand_off, op_cand_count;
    std::vector<int32_t> cand_expert, cand_logit, cand_ordinal;
};

struct DeviceStep {
    StepHost host;
    DeviceBuffer<int32_t> kind; DeviceBuffer<uint64_t> a, b;
    DeviceBuffer<int32_t> coff, ccnt, ce, cl, co;
    DeviceBuffer<uint64_t> out[17];
    SmorInputs inputs; SmorOutputs outputs;
};

static SmorProblemSpec make_bench_spec() {
    SmorProblemSpec spec = {};
    spec.abi_version = SMOR_ABI_VERSION;
    spec.num_experts = 16;
    spec.max_live_tokens = 512;
    spec.overflow_capacity = 256;
    spec.max_candidates_per_route = 6;
    spec.token_space = 65536;
    spec.max_batch = 256;
    spec.max_steps = 48;
    spec.flags = 0;
    if (!smor_validate_problem_spec(&spec)) throw std::runtime_error("invalid bench spec");
    return spec;
}

static StepHost make_step(const SmorProblemSpec& spec, int step_id, int n, SplitMix64& rng, uint64_t* next_token) {
    StepHost s; s.run = {}; s.run.abi_version = SMOR_ABI_VERSION; s.run.step_id = step_id;
    const size_t rows = std::max<size_t>(1, (size_t)n);
    s.op_kind.assign(rows, 0); s.op_a.assign(rows, 0); s.op_b.assign(rows, 0);
    s.op_cand_off.assign(rows, 0); s.op_cand_count.assign(rows, 0);
    for (int i = 0; i < n; ++i) {
        int roll = rng.uniform_int(0, 99);
        if (roll < 55) {
            s.op_kind[i] = SMOR_OP_ROUTE;
            s.op_a[i] = (*next_token)++;
            s.op_b[i] = (uint64_t)rng.uniform_int(1, 30);
            int cc = rng.uniform_int(1, 6);
            s.op_cand_off[i] = (int32_t)s.cand_expert.size();
            s.op_cand_count[i] = cc;
            for (int k = 0; k < cc; ++k) {
                s.cand_expert.push_back(rng.uniform_int(0, 15));
                s.cand_logit.push_back(rng.uniform_int(-8, 8));
                s.cand_ordinal.push_back(k);
            }
        } else if (roll < 75) {
            s.op_kind[i] = SMOR_OP_REFILL;
            s.op_a[i] = rng.uniform_int(0, 15);
            s.op_b[i] = (uint64_t)rng.uniform_int(0, 100);
            s.op_cand_off[i] = (int32_t)s.cand_expert.size();
        } else if (roll < 90) {
            s.op_kind[i] = SMOR_OP_DRAIN;
            s.op_a[i] = rng.uniform_int(0, 16);
            s.op_cand_off[i] = (int32_t)s.cand_expert.size();
        } else {
            s.op_kind[i] = SMOR_OP_DROP_QUEUED_THROUGH;
            s.op_a[i] = (uint64_t)rng.uniform_int(0, 2000);
            s.op_cand_off[i] = (int32_t)s.cand_expert.size();
        }
    }
    s.run.batch_size = n;
    s.run.cand_total = (int32_t)s.cand_expert.size();
    if (s.cand_expert.empty()) { s.cand_expert.assign(1, 0); s.cand_logit.assign(1, 0); s.cand_ordinal.assign(1, 0); }
    if (!smor_validate_run_spec(&s.run, &spec)) throw std::runtime_error("invalid bench run");
    return s;
}

static DeviceStep* make_device_step(const StepHost& h) {
    DeviceStep* ds = new DeviceStep(); ds->host = h;
    ds->kind.allocate(h.op_kind.size()); ds->kind.upload(h.op_kind);
    ds->a.allocate(h.op_a.size()); ds->a.upload(h.op_a);
    ds->b.allocate(h.op_b.size()); ds->b.upload(h.op_b);
    ds->coff.allocate(h.op_cand_off.size()); ds->coff.upload(h.op_cand_off);
    ds->ccnt.allocate(h.op_cand_count.size()); ds->ccnt.upload(h.op_cand_count);
    ds->ce.allocate(h.cand_expert.size()); ds->ce.upload(h.cand_expert);
    ds->cl.allocate(h.cand_logit.size()); ds->cl.upload(h.cand_logit);
    ds->co.allocate(h.cand_ordinal.size()); ds->co.upload(h.cand_ordinal);
    for (int k = 0; k < 17; ++k) ds->out[k].allocate(1);
    ds->inputs = {};
    ds->inputs.op_kind = ds->kind.ptr; ds->inputs.op_a = ds->a.ptr; ds->inputs.op_b = ds->b.ptr;
    ds->inputs.op_cand_off = ds->coff.ptr; ds->inputs.op_cand_count = ds->ccnt.ptr;
    ds->inputs.cand_expert = ds->ce.ptr; ds->inputs.cand_logit = ds->cl.ptr; ds->inputs.cand_ordinal = ds->co.ptr;
    ds->outputs = {};
    uint64_t** o = (uint64_t**)&ds->outputs;
    for (int k = 0; k < 17; ++k) o[k] = ds->out[k].ptr;
    return ds;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 20; if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        const SmorProblemSpec spec = make_bench_spec();
        const size_t workspace_bytes = solution_workspace_bytes(&spec);
        if (workspace_bytes == 0) throw std::runtime_error("workspace 0");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::vector<uint64_t> cap((size_t)spec.num_experts, 1000000), icr((size_t)spec.num_experts, 50);
        DeviceBuffer<uint64_t> d_cap; d_cap.allocate(cap.size()); d_cap.upload(cap);
        DeviceBuffer<uint64_t> d_icr; d_icr.allocate(icr.size()); d_icr.upload(icr);
        SmorInitConfig config = {}; config.credit_cap = d_cap.ptr; config.initial_credit = d_icr.ptr;

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &config, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace; workspace.allocate(std::max<size_t>(workspace_bytes, 1));

        SplitMix64 rng(g_state ^ 0x70707070ULL);
        uint64_t next_token = 1;
        std::vector<DeviceStep*> steps;
        for (int s = 0; s < spec.max_steps; ++s)
            steps.push_back(make_device_step(make_step(spec, s, rng.uniform_int(64, 256), rng, &next_token)));

        std::printf("bench E=%d max_live=%d overflow=%d T=%zu\n",
                    spec.num_experts, spec.max_live_tokens, spec.overflow_capacity, steps.size());

        for (int w = 0; w < 3; ++w) {
            CUDA_CHECK(solution_reset(state, stream));
            for (DeviceStep* ds : steps)
                CUDA_CHECK(solution_run(state, &ds->host.run, &ds->inputs, &ds->outputs, workspace.ptr, workspace.count, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        double total_ms = 0.0;
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(a, stream));
            for (DeviceStep* ds : steps)
                CUDA_CHECK(solution_run(state, &ds->host.run, &ds->inputs, &ds->outputs, workspace.ptr, workspace.count, stream));
            CUDA_CHECK(cudaEventRecord(b, stream));
            CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); total_ms += ms;
        }
        std::printf("avg_ms=%.6f\n", total_ms / iters);

        CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
        for (DeviceStep* ds : steps) delete ds;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
