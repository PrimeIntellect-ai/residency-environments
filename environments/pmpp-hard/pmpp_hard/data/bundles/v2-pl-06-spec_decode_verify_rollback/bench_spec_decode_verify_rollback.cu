// ============================================================================
// file: bench_spec_decode_verify_rollback.cu
// ============================================================================

#include "spec_decode_verify_rollback_common.h"
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

static constexpr uint64_t g_state = 0x082efa98ec4e6c89ULL;

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

struct StepHost {
    SdvRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> draft_value;
    std::vector<int32_t> correction_value;
    std::vector<uint32_t> p_target;
    std::vector<uint32_t> p_draft;
    std::vector<uint32_t> uniform_u32;
};

struct DeviceStep {
    StepHost host;

    DeviceBuffer<int32_t> active_seq;
    DeviceBuffer<int32_t> draft_value;
    DeviceBuffer<int32_t> correction_value;
    DeviceBuffer<uint32_t> p_target;
    DeviceBuffer<uint32_t> p_draft;
    DeviceBuffer<uint32_t> uniform_u32;

    DeviceBuffer<int32_t> accepted_count;
    DeviceBuffer<int32_t> new_length;
    DeviceBuffer<int64_t> live_cache_sum;
    DeviceBuffer<uint64_t> live_cache_tail_hash;
    DeviceBuffer<uint64_t> state_checksum;

    SdvInputs inputs;
    SdvOutputs outputs;
};

static SdvProblemSpec make_spec(int B, int max_len, int max_steps) {
    SdvProblemSpec spec = {};
    spec.abi_version = SDV_ABI_VERSION;
    spec.B = B;
    spec.max_len = max_len;
    spec.max_steps = max_steps;
    spec.flags = 0;

    if (!sdv_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid SdvProblemSpec generated");
    }

    return spec;
}

static void set_accept_bit(
    bool accept,
    uint32_t* p_target,
    uint32_t* p_draft,
    uint32_t* uniform_u32) {
    if (accept) {
        *p_target = 0xffffffffU;
        *p_draft = 1U;
        *uniform_u32 = 0xffffffffU;
    } else {
        *p_target = 0U;
        *p_draft = 0xffffffffU;
        *uniform_u32 = 1U;
    }
}

static std::vector<int> prefix_to_bits(int L, int prefix) {
    std::vector<int> bits((size_t)L, 0);
    for (int i = 0; i < L && i < prefix; ++i) {
        bits[(size_t)i] = 1;
    }
    return bits;
}

static StepHost make_step(
    const SdvProblemSpec& spec,
    int step_id,
    int draft_len,
    const std::vector<int32_t>& active,
    const std::vector<int>& prefixes,
    uint64_t seed) {
    if (active.size() != prefixes.size()) {
        throw std::runtime_error("active/prefixes size mismatch");
    }

    StepHost step;
    step.run = {};
    step.run.abi_version = SDV_ABI_VERSION;
    step.run.active_count = static_cast<int32_t>(active.size());
    step.run.draft_len = draft_len;
    step.run.step_id = step_id;

    if (!sdv_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid SdvRunSpec generated");
    }

    const size_t A = active.empty() ? 1 : active.size();
    SplitMix64 rng(seed);

    step.active_seq = active;
    if (step.active_seq.empty()) {
        step.active_seq.push_back(0);
    }

    step.draft_value.assign(A * (size_t)draft_len, 0);
    step.correction_value.assign(A, 0);
    step.p_target.assign(A * (size_t)draft_len, 0);
    step.p_draft.assign(A * (size_t)draft_len, 0);
    step.uniform_u32.assign(A * (size_t)draft_len, 0);

    for (size_t a = 0; a < active.size(); ++a) {
        step.correction_value[a] = rng.next_i32();

        const std::vector<int> bits = prefix_to_bits(draft_len, prefixes[a]);

        for (int i = 0; i < draft_len; ++i) {
            const size_t idx = a * (size_t)draft_len + (size_t)i;
            int32_t v = rng.next_i32();
            if (v == 0) {
                v = static_cast<int32_t>(i + 1 + step_id * 17);
            }

            step.draft_value[idx] = v;

            set_accept_bit(
                bits[(size_t)i] != 0,
                &step.p_target[idx],
                &step.p_draft[idx],
                &step.uniform_u32[idx]);
        }
    }

    return step;
}

// One sequence variant per timed iteration (plus one for warmup): same step/active/
// prefix STRUCTURE every variant (timing comparability), but draft/correction token
// VALUES are derived from (PMPP_BENCH_SEED, variant, step). Every timed call thus
// receives fresh inputs, and each iteration's outputs are folded into out_fnv below,
// so replaying an output cached from warmup or an earlier iteration leaves a stale
// region in the digest → perf FAIL.
static std::vector<StepHost> build_bench_sequence(const SdvProblemSpec& spec,
                                                  uint64_t variant_seed) {
    std::vector<StepHost> steps;

    const uint64_t base_seed = variant_seed;

    for (int s = 0; s < 40; ++s) {
        int L = (s % 3 == 0) ? 8 : ((s % 3 == 1) ? 4 : 2);

        std::vector<int32_t> active;
        std::vector<int> prefixes;

        if (s == 8) {
            active.clear();
            prefixes.clear();
        } else {
            for (int b = 0; b < spec.B; ++b) {
                if (b == 0 || ((b + s) % 3 != 0)) {
                    active.push_back(b);

                    if (b == 0) {
                        prefixes.push_back(L);
                    } else if ((s + b) % 5 == 0) {
                        prefixes.push_back(0);
                    } else if ((s + b) % 5 == 1) {
                        prefixes.push_back(std::max(0, L - 1));
                    } else if ((s + b) % 5 == 2) {
                        prefixes.push_back(1);
                    } else {
                        prefixes.push_back(L / 2);
                    }
                }
            }
        }

        steps.push_back(make_step(
            spec,
            s,
            L,
            active,
            prefixes,
            base_seed ^ 0x700000000ULL ^ static_cast<uint64_t>(s)));
    }

    return steps;
}

static DeviceStep* make_device_step(const StepHost& host) {
    DeviceStep* ds = new DeviceStep();
    ds->host = host;

    const int A = host.run.active_count;
    const size_t out_count = std::max<size_t>(1, (size_t)A);

    ds->active_seq.allocate(host.active_seq.size());
    ds->draft_value.allocate(host.draft_value.size());
    ds->correction_value.allocate(host.correction_value.size());
    ds->p_target.allocate(host.p_target.size());
    ds->p_draft.allocate(host.p_draft.size());
    ds->uniform_u32.allocate(host.uniform_u32.size());

    ds->active_seq.upload(host.active_seq);
    ds->draft_value.upload(host.draft_value);
    ds->correction_value.upload(host.correction_value);
    ds->p_target.upload(host.p_target);
    ds->p_draft.upload(host.p_draft);
    ds->uniform_u32.upload(host.uniform_u32);

    ds->accepted_count.allocate(out_count);
    ds->new_length.allocate(out_count);
    ds->live_cache_sum.allocate(out_count);
    ds->live_cache_tail_hash.allocate(out_count);
    ds->state_checksum.allocate(1);

    ds->inputs = {};
    ds->inputs.active_seq = ds->active_seq.ptr;
    ds->inputs.draft_value = ds->draft_value.ptr;
    ds->inputs.correction_value = ds->correction_value.ptr;
    ds->inputs.p_target = ds->p_target.ptr;
    ds->inputs.p_draft = ds->p_draft.ptr;
    ds->inputs.uniform_u32 = ds->uniform_u32.ptr;

    ds->outputs = {};
    ds->outputs.accepted_count = ds->accepted_count.ptr;
    ds->outputs.new_length = ds->new_length.ptr;
    ds->outputs.live_cache_sum = ds->live_cache_sum.ptr;
    ds->outputs.live_cache_tail_hash = ds->live_cache_tail_hash.ptr;
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

        int iters = 10;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        SdvProblemSpec spec = make_spec(
            8,      // B
            128,    // max_len
            64);    // max_steps

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
        const uint64_t base_seed = pmpp::bench_seed(g_state);
        SplitMix64 vseed_rng(base_seed ^ 0x700000000ULL);

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
            "bench_sequence B=%d max_len=%d max_steps=%d T=%zu\n",
            spec.B,
            spec.max_len,
            spec.max_steps,
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

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            total_ms += static_cast<double>(elapsed_ms);

            for (DeviceStep* ds : steps) {
                const size_t A = (size_t)ds->host.run.active_count;
                if (A > 0) {
                    dg.dev(ds->accepted_count.ptr, A * sizeof(int32_t));
                    dg.dev(ds->new_length.ptr, A * sizeof(int32_t));
                    dg.dev(ds->live_cache_sum.ptr, A * sizeof(int64_t));
                    dg.dev(ds->live_cache_tail_hash.ptr, A * sizeof(uint64_t));
                }
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
