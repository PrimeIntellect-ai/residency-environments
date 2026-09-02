// file: bench_switch_moe_overflow_router.cu

#include "switch_moe_overflow_router_common.h"
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
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
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

// --- C3 timed-fold (patterns I1 + P1, op-stream form): every timed EPISODE
// replays the same op structure (kinds, batch sizes, candidate counts — so
// timing stays comparable) but with fresh per-episode VALUES (routing weights,
// candidate experts/logits, refill/drain/drop parameters) pregenerated from
// (PMPP_BENCH_SEED, episode, step). Warmup uses dedicated episodes. After each
// timed episode (outside the event pair — zero timing impact) a probe kernel
// folds all 17 graded u64 outputs of every step into acc[episode]; the digest
// folds every accumulator, so an episode that replays cached outputs mismatches.

struct DeviceStep {
    StepHost host;
    DeviceBuffer<int32_t> kind; DeviceBuffer<uint64_t> a, b;
    DeviceBuffer<int32_t> coff, ccnt, ce, cl, co;
    // Per-episode value variants (episode e at offset e * count).
    DeviceBuffer<uint64_t> a_var, b_var;
    DeviceBuffer<int32_t> ce_var, cl_var;
    DeviceBuffer<uint64_t> out[17];
    SmorInputs inputs; SmorOutputs outputs;
};

__device__ __forceinline__ uint64_t c3_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__global__ void c3_probe_fold(const uint64_t* const* ptrs, int n, uint64_t salt,
                              unsigned long long* acc) {
    __shared__ unsigned long long sh[256];
    unsigned long long local = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local ^= c3_mix(*ptrs[i] ^ c3_mix(salt + (uint64_t)i));
    }
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] ^= sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicXor(acc, sh[0]);
}

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

static uint64_t host_mix64(uint64_t z) {
    z ^= z >> 30; z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27; z *= 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Builds the per-episode value variants for one step: op structure (kinds,
// batch size, candidate offsets/counts, ROUTE token ids) is shared across
// episodes; only the data values vary, as a pure function of
// (PMPP_BENCH_SEED, episode, step).
static DeviceStep* make_device_step(const StepHost& h, int n_ep, uint64_t c3_root) {
    DeviceStep* ds = new DeviceStep(); ds->host = h;
    ds->kind.allocate(h.op_kind.size()); ds->kind.upload(h.op_kind);
    ds->a.allocate(h.op_a.size()); ds->a.upload(h.op_a);
    ds->b.allocate(h.op_b.size()); ds->b.upload(h.op_b);
    ds->coff.allocate(h.op_cand_off.size()); ds->coff.upload(h.op_cand_off);
    ds->ccnt.allocate(h.op_cand_count.size()); ds->ccnt.upload(h.op_cand_count);
    ds->ce.allocate(h.cand_expert.size()); ds->ce.upload(h.cand_expert);
    ds->cl.allocate(h.cand_logit.size()); ds->cl.upload(h.cand_logit);
    ds->co.allocate(h.cand_ordinal.size()); ds->co.upload(h.cand_ordinal);

    const size_t rows = h.op_kind.size();
    const size_t cn = h.cand_expert.size();
    std::vector<uint64_t> av((size_t)n_ep * rows), bv((size_t)n_ep * rows);
    std::vector<int32_t> cev((size_t)n_ep * cn, 0), clv((size_t)n_ep * cn, 0);
    for (int e = 0; e < n_ep; ++e) {
        SplitMix64 r(host_mix64(c3_root ^ (uint64_t)(e + 1) * 0x9e3779b97f4a7c15ULL ^
                                (uint64_t)h.run.step_id * 0xbf58476d1ce4e5b9ULL));
        const size_t ro = (size_t)e * rows;
        const size_t co2 = (size_t)e * cn;
        for (size_t i = 0; i < rows; ++i) {
            const int32_t kind = h.op_kind[i];
            av[ro + i] = h.op_a[i];
            bv[ro + i] = h.op_b[i];
            if (kind == SMOR_OP_ROUTE) {
                bv[ro + i] = (uint64_t)r.uniform_int(1, 30);
                const int cc = h.op_cand_count[i];
                const size_t off = (size_t)h.op_cand_off[i];
                for (int k = 0; k < cc; ++k) {
                    cev[co2 + off + k] = r.uniform_int(0, 15);
                    clv[co2 + off + k] = r.uniform_int(-8, 8);
                }
            } else if (kind == SMOR_OP_REFILL) {
                av[ro + i] = (uint64_t)r.uniform_int(0, 15);
                bv[ro + i] = (uint64_t)r.uniform_int(0, 100);
            } else if (kind == SMOR_OP_DRAIN) {
                av[ro + i] = (uint64_t)r.uniform_int(0, 16);
            } else {
                av[ro + i] = (uint64_t)r.uniform_int(0, 2000);
            }
        }
    }
    ds->a_var.allocate(av.size()); ds->a_var.upload(av);
    ds->b_var.allocate(bv.size()); ds->b_var.upload(bv);
    ds->ce_var.allocate(cev.size()); ds->ce_var.upload(cev);
    ds->cl_var.allocate(clv.size()); ds->cl_var.upload(clv);

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

        // Step data values derive from PMPP_BENCH_SEED so graded outputs are not
        // precomputable offline; the step/op family stays fixed for timing.
        SplitMix64 rng(g_state ^ 0x70707070ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL));
        SplitMix64 c3_root_rng(0xC3ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL));
        const uint64_t c3_root = c3_root_rng.next_u64();
        const int kWarmups = 3;
        const int n_ep = iters + kWarmups;  // dedicated warmup episodes at the end
        uint64_t next_token = 1;
        std::vector<DeviceStep*> steps;
        for (int s = 0; s < spec.max_steps; ++s)
            steps.push_back(make_device_step(make_step(spec, s, rng.uniform_int(64, 256), rng, &next_token), n_ep, c3_root));

        std::printf("bench E=%d max_live=%d overflow=%d T=%zu\n",
                    spec.num_experts, spec.max_live_tokens, spec.overflow_capacity, steps.size());

        // Points every step's value inputs at episode e (host-side pointer
        // flip between episodes: zero GPU work inside the timed region).
        auto set_episode = [&](int e) {
            for (DeviceStep* ds : steps) {
                const size_t rows = ds->host.op_kind.size();
                const size_t cn = ds->host.cand_expert.size();
                ds->inputs.op_a = ds->a_var.ptr + (size_t)e * rows;
                ds->inputs.op_b = ds->b_var.ptr + (size_t)e * rows;
                ds->inputs.cand_expert = ds->ce_var.ptr + (size_t)e * cn;
                ds->inputs.cand_logit = ds->cl_var.ptr + (size_t)e * cn;
            }
        };

        // Device pointer table over all steps' 17 graded outputs + per-episode
        // probe accumulators (folded into out_fnv below).
        std::vector<const uint64_t*> out_ptrs_host;
        for (DeviceStep* ds : steps)
            for (int k = 0; k < 17; ++k)
                out_ptrs_host.push_back(ds->out[k].ptr);
        DeviceBuffer<const uint64_t*> d_out_ptrs;
        d_out_ptrs.allocate(out_ptrs_host.size());
        CUDA_CHECK(cudaMemcpy(d_out_ptrs.ptr, out_ptrs_host.data(),
                              sizeof(const uint64_t*) * out_ptrs_host.size(),
                              cudaMemcpyHostToDevice));
        DeviceBuffer<unsigned long long> probe_acc;
        probe_acc.allocate((size_t)iters);
        CUDA_CHECK(cudaMemset(probe_acc.ptr, 0, sizeof(unsigned long long) * probe_acc.count));

        for (int w = 0; w < kWarmups; ++w) {
            set_episode(iters + w);
            CUDA_CHECK(solution_reset(state, stream));
            for (DeviceStep* ds : steps)
                CUDA_CHECK(solution_run(state, &ds->host.run, &ds->inputs, &ds->outputs, workspace.ptr, workspace.count, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        double total_ms = 0.0;
        for (int it = 0; it < iters; ++it) {
            set_episode(it);
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(a, stream));
            for (DeviceStep* ds : steps)
                CUDA_CHECK(solution_run(state, &ds->host.run, &ds->inputs, &ds->outputs, workspace.ptr, workspace.count, stream));
            CUDA_CHECK(cudaEventRecord(b, stream));
            // Untimed (after the stop event): fold this episode's graded
            // outputs before the next episode overwrites them.
            c3_probe_fold<<<1, 256, 0, stream>>>(
                d_out_ptrs.ptr, (int)out_ptrs_host.size(),
                host_mix64(c3_root ^ 0xF01DULL ^ (uint64_t)(it + 1)),
                probe_acc.ptr + it);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); total_ms += ms;
        }
        std::printf("avg_ms=%.6f\n", total_ms / iters);

        // Untimed graded-output digest: fresh replay of all steps, then fold every
        // step's 17 graded u64 outputs (the full SmorOutputs set) in step order.
        CUDA_CHECK(solution_reset(state, stream));
        for (DeviceStep* ds : steps)
            CUDA_CHECK(solution_run(state, &ds->host.run, &ds->inputs, &ds->outputs, workspace.ptr, workspace.count, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        // C3: the digest binds EVERY timed episode's graded outputs via the
        // per-episode probe accumulators.
        {
            std::vector<unsigned long long> acc_host(probe_acc.count);
            CUDA_CHECK(cudaMemcpy(acc_host.data(), probe_acc.ptr,
                                  sizeof(unsigned long long) * probe_acc.count,
                                  cudaMemcpyDeviceToHost));
            dg.bytes(acc_host.data(), sizeof(unsigned long long) * acc_host.size());
        }
        for (DeviceStep* ds : steps)
            for (int k = 0; k < 17; ++k)
                dg.dev(ds->out[k].ptr, sizeof(uint64_t));
        dg.print();

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
