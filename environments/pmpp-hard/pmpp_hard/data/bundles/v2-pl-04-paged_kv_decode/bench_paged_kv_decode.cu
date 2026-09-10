// ============================================================================
// file: bench_paged_kv_decode.cu
// ============================================================================

#include "paged_kv_decode_common.h"
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

static constexpr uint64_t g_state = 0xa4093822299f31d0ULL;

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

    float uniform_float(float lo, float hi) {
        const float u = static_cast<float>((next_u64() >> 11) * 0x1.0p-53);
        return lo + (hi - lo) * u;
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
    PkdRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> new_token_count;
    std::vector<float> new_k;
    std::vector<float> new_v;
    std::vector<float> new_scale;
    std::vector<float> q;
};

struct DeviceStep {
    StepHost host;

    DeviceBuffer<int32_t> active_seq;
    DeviceBuffer<int32_t> new_token_count;
    DeviceBuffer<float> new_k;
    DeviceBuffer<float> new_v;
    DeviceBuffer<float> new_scale;
    DeviceBuffer<float> q;

    DeviceBuffer<float> y;
    DeviceBuffer<int32_t> lengths;
    DeviceBuffer<uint64_t> checksum;

    PkdInputs inputs;
    PkdOutputs outputs;
};

static PkdProblemSpec make_spec(
    int B,
    int Hq,
    int Hkv,
    int D,
    int page_size,
    int max_seq_len) {
    PkdProblemSpec spec = {};
    spec.abi_version = PKD_ABI_VERSION;
    spec.B = B;
    spec.Hq = Hq;
    spec.Hkv = Hkv;
    spec.D = D;
    spec.page_size = page_size;
    spec.max_seq_len = max_seq_len;
    spec.max_pages = B * pkd_pages_per_seq(max_seq_len, page_size);
    spec.flags = 0;

    if (!pkd_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid PkdProblemSpec generated");
    }

    return spec;
}

static void fill_step_tensors(
    const PkdProblemSpec& spec,
    StepHost* step,
    uint64_t seed) {
    const int A = step->run.active_count;
    const int Hq = spec.Hq;
    const int Hkv = spec.Hkv;
    const int D = spec.D;

    const size_t a_count = A > 0 ? static_cast<size_t>(A) : 1;

    step->new_token_count.resize(a_count, 0);
    step->new_k.resize(a_count * PKD_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D, 0.0f);
    step->new_v.resize(a_count * PKD_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D, 0.0f);
    step->new_scale.resize(a_count * PKD_MAX_NEW_TOKENS * (size_t)Hkv, 1.0f);
    step->q.resize(a_count * (size_t)Hq * (size_t)D, 0.0f);

    for (int a = 0; a < A; ++a) {
        const int seq = step->active_seq[(size_t)a];
        SplitMix64 rng(seed ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(seq + 1)));

        for (int nt = 0; nt < PKD_MAX_NEW_TOKENS; ++nt) {
            for (int h = 0; h < Hkv; ++h) {
                const size_t scale_idx =
                    ((size_t)a * PKD_MAX_NEW_TOKENS + (size_t)nt) *
                    (size_t)Hkv + (size_t)h;
                step->new_scale[scale_idx] = rng.uniform_float(0.035f, 0.090f);
            }
        }

        for (int nt = 0; nt < PKD_MAX_NEW_TOKENS; ++nt) {
            for (int h = 0; h < Hkv; ++h) {
                for (int d = 0; d < D; ++d) {
                    const size_t idx =
                        ((size_t)a * PKD_MAX_NEW_TOKENS + (size_t)nt) *
                        (size_t)Hkv * (size_t)D +
                        (size_t)h * (size_t)D +
                        (size_t)d;

                    step->new_k[idx] = rng.uniform_float(-2.25f, 2.25f);
                    step->new_v[idx] = rng.uniform_float(-2.25f, 2.25f);
                }
            }
        }

        for (int h = 0; h < Hq; ++h) {
            for (int d = 0; d < D; ++d) {
                const size_t idx =
                    ((size_t)a * (size_t)Hq + (size_t)h) *
                    (size_t)D + (size_t)d;
                step->q[idx] = rng.uniform_float(-0.20f, 0.20f);
            }
        }
    }
}

static StepHost make_step(
    const PkdProblemSpec& spec,
    int step_id,
    int window_size,
    const std::vector<int32_t>& active,
    const std::vector<int32_t>& ntokens,
    uint64_t seed) {
    StepHost step;
    step.run = {};
    step.run.abi_version = PKD_ABI_VERSION;
    step.run.active_count = static_cast<int32_t>(active.size());
    step.run.step_id = step_id;
    step.run.window_size = window_size;

    if (!pkd_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid PkdRunSpec generated");
    }

    step.active_seq = active;
    if (step.active_seq.empty()) {
        step.active_seq.push_back(0);
    }

    fill_step_tensors(spec, &step, seed);

    for (size_t i = 0; i < ntokens.size(); ++i) {
        step.new_token_count[i] = ntokens[i];
    }

    return step;
}

// One sequence variant per timed iteration (plus one for warmup): same step/active/
// window STRUCTURE every variant (timing comparability), but k/v/scale/q values are
// derived from (PMPP_BENCH_SEED, variant, step). Every timed call therefore receives
// fresh inputs, and its outputs are folded into out_fnv below — replaying an output
// cached from warmup or an earlier iteration leaves a stale region in the digest.
static std::vector<StepHost> build_bench_sequence(const PkdProblemSpec& spec,
                                                  uint64_t variant_seed) {
    std::vector<StepHost> steps;

    const uint64_t base_seed = variant_seed;

    for (int s = 0; s < 32; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        if (s == 5) {
            active.clear();
            nt.clear();
        } else if (s == 10) {
            active = {0, 2, 4, 6};
            nt = {0, 0, 0, 0};
        } else {
            for (int b = 0; b < spec.B; ++b) {
                if (b == 0 || ((b + s) % 3 != 0)) {
                    active.push_back(b);
                    if (b == 0) {
                        nt.push_back(2);
                    } else {
                        nt.push_back((s + b) % 2 == 0 ? 2 : 1);
                    }
                }
            }
        }

        const int window = (s < 12) ? spec.max_seq_len : 33;

        steps.push_back(make_step(
            spec,
            s,
            window,
            active,
            nt,
            base_seed ^ 0x700000000ULL ^ static_cast<uint64_t>(s)));
    }

    return steps;
}

static DeviceStep* make_device_step(const PkdProblemSpec& spec, const StepHost& host) {
    DeviceStep* ds = new DeviceStep();
    ds->host = host;

    const int A = host.run.active_count;
    const size_t y_count =
        std::max<size_t>(1, (size_t)A * (size_t)spec.Hq * (size_t)spec.D);

    ds->active_seq.allocate(host.active_seq.size());
    ds->new_token_count.allocate(host.new_token_count.size());
    ds->new_k.allocate(host.new_k.size());
    ds->new_v.allocate(host.new_v.size());
    ds->new_scale.allocate(host.new_scale.size());
    ds->q.allocate(host.q.size());

    ds->active_seq.upload(host.active_seq);
    ds->new_token_count.upload(host.new_token_count);
    ds->new_k.upload(host.new_k);
    ds->new_v.upload(host.new_v);
    ds->new_scale.upload(host.new_scale);
    ds->q.upload(host.q);

    ds->y.allocate(y_count);
    ds->lengths.allocate((size_t)spec.B);
    ds->checksum.allocate(1);

    ds->inputs = {};
    ds->inputs.active_seq = ds->active_seq.ptr;
    ds->inputs.new_token_count = ds->new_token_count.ptr;
    ds->inputs.new_k = ds->new_k.ptr;
    ds->inputs.new_v = ds->new_v.ptr;
    ds->inputs.new_scale = ds->new_scale.ptr;
    ds->inputs.q = ds->q.ptr;

    ds->outputs = {};
    ds->outputs.y = ds->y.ptr;
    ds->outputs.lengths = ds->lengths.ptr;
    ds->outputs.state_checksum = ds->checksum.ptr;

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

        PkdProblemSpec spec = make_spec(
            8,      // B
            8,      // Hq
            2,      // Hkv
            128,    // D
            32,     // page size
            96);    // max seq len

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
                steps.push_back(make_device_step(spec, h));
            }
            variants.push_back(std::move(steps));
        }

        std::printf(
            "bench_sequence B=%d Hq=%d Hkv=%d D=%d P=%d max_seq=%d T=%zu\n",
            spec.B,
            spec.Hq,
            spec.Hkv,
            spec.D,
            spec.page_size,
            spec.max_seq_len,
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

        // NOTE: y is tolerance-graded (not byte-exact) per the contract, so it is NOT
        // folded; lengths and state_checksum are exact-graded and are. The fold runs
        // between iterations, outside the timed region.
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
                dg.dev(ds->lengths.ptr, (size_t)spec.B * sizeof(int32_t));
                dg.dev(ds->checksum.ptr, sizeof(uint64_t));
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
