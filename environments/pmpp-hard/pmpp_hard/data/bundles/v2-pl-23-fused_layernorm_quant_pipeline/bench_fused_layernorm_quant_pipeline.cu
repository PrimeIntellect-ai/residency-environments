// file: bench_fused_layernorm_quant_pipeline.cu

#include "fused_layernorm_quant_pipeline_common.h"
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

static constexpr uint64_t g_state = 0x6f2d58c4a19e73b5ULL;

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
    FlqpRunSpec run;
    std::vector<float> x;
    std::vector<float> weight;
    std::vector<float> bias;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<float> x;
    DeviceBuffer<float> weight;
    DeviceBuffer<float> bias;

    DeviceBuffer<int8_t> q_int8;
    DeviceBuffer<float> scale;
    DeviceBuffer<float> dequant;
    DeviceBuffer<int64_t> code_sum;

    FlqpInputs inputs;
    FlqpOutputs outputs;
};

__device__ __forceinline__ uint64_t c3_mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// C3 per-call output probe (recipe pattern P2): right after every timed call,
// XOR-mix the exact-graded outputs (q_int8 + code_sum) into this call's slot of
// a device accumulator array. XOR is order-independent, so the result is
// deterministic; the digest later folds all slots, binding EVERY timed call's
// output to the out_fnv. Reads hit data the solution just wrote (mostly L2), so
// the paired-cancelled overhead stays small.
__global__ void probe_fold(
    const int8_t* q,
    size_t q_words,
    const int64_t* code_sum,
    size_t n_rows,
    unsigned long long* slot) {
    __shared__ unsigned long long block_h[256];
    const uint64_t* qw = reinterpret_cast<const uint64_t*>(q);
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t h = 0;
    for (size_t w = tid; w < q_words; w += stride) {
        h ^= c3_mix64(qw[w] + 0x9e3779b97f4a7c15ULL * (uint64_t)(w + 1));
    }
    for (size_t i = tid; i < n_rows; i += stride) {
        h ^= c3_mix64((uint64_t)code_sum[i] + 0xbf58476d1ce4e5b9ULL * (uint64_t)(i + 1));
    }

    block_h[threadIdx.x] = h;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) {
            block_h[threadIdx.x] ^= block_h[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicXor(slot, block_h[0]);
    }
}

// C3 per-iteration input variation: before every timed call the bench flips the
// sign of a salt-dependent ~1/8 subset of the weight vector in place (weights
// are arbitrary finite floats in the contract, and the affine weight touches
// EVERY row's normalized output and per-row quant scale, so every timed call
// must genuinely recompute all rows). The vector is only D floats, so the
// kernel costs microseconds and is paired-cancelled anyway. The salt derives
// from (PMPP_BENCH_SEED, case, iter), so the same seed yields a bit-identical
// input sequence in the student and reference benches. Exact zeros are left
// alone so ZEROISH weights keep their crafted value.
__global__ void flip_weight_subset(float* w, int D, uint64_t salt) {
    for (int d = (int)threadIdx.x; d < D; d += (int)blockDim.x) {
        uint64_t z = ((uint64_t)d + 0x9e3779b97f4a7c15ULL) ^ salt;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        if ((z & 7ULL) == 0ULL) {
            const float v = w[d];
            if (v != 0.0f) {
                w[d] = -v;
            }
        }
    }
}

static float exact_frac(int num, int denom) {
    return static_cast<float>(num) / static_cast<float>(denom);
}

static float make_weight(int d, int distribution_id) {
    if (distribution_id == FLQP_DIST_OUTLIERS) {
        if ((d % 257) == 0) return -2.0f;
        if ((d % 149) == 0) return 3.0f;
    }

    if (distribution_id == FLQP_DIST_ZEROISH) {
        if ((d % 17) == 0) return 0.0f;
    }

    static const float vals[] = {
        -1.5f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f, 1.5f, 2.0f
    };
    return vals[(d * 13 + 5) & 7];
}

static float make_bias(int d, int distribution_id) {
    if (distribution_id == FLQP_DIST_OUTLIERS) {
        if ((d % 509) == 0) return 8.0f;
        if ((d % 331) == 0) return -8.0f;
    }

    if (distribution_id == FLQP_DIST_ZEROISH) {
        static const float vals[] = {-0.03125f, 0.0f, 0.03125f};
        return vals[d % 3];
    }

    static const float vals[] = {
        -0.75f, -0.25f, -0.125f, 0.0f, 0.125f, 0.25f, 0.75f
    };
    return vals[(d * 7 + 3) % 7];
}

static float make_x_value(
    SplitMix64& rng,
    int row,
    int d,
    int D,
    int distribution_id) {
    if (distribution_id == FLQP_DIST_OUTLIERS) {
        if ((row % 97) == 0 && d == ((row * 13) % D)) return 512.0f;
        if ((row % 131) == 0 && d == ((row * 17 + 3) % D)) return -512.0f;

        static const float vals[] = {
            -4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f
        };
        return vals[rng.uniform_int(0, 8)];
    }

    if (distribution_id == FLQP_DIST_ZEROISH) {
        const float base = exact_frac((row % 7) - 3, 8);
        if ((d % 64) == 0) return base + exact_frac(1, 1024);
        if ((d % 97) == 0) return base - exact_frac(1, 1024);
        return base;
    }

    static const float vals[] = {
        -2.0f, -1.5f, -1.0f, -0.5f, -0.25f,
         0.0f,
         0.25f, 0.5f, 1.0f, 1.5f, 2.0f
    };
    return vals[rng.uniform_int(0, 10)];
}

static HostCase make_case(
    const char* name,
    int N,
    int D,
    int distribution_id,
    float eps,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = FLQP_ABI_VERSION;
    hc.run.N = N;
    hc.run.D = D;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);
    hc.run.eps = eps;

    if (!flqp_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid FlqpRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.x.resize((size_t)N * (size_t)D);
    hc.weight.resize((size_t)D);
    hc.bias.resize((size_t)D);

    for (int d = 0; d < D; ++d) {
        hc.weight[(size_t)d] = make_weight(d, distribution_id);
        hc.bias[(size_t)d] = make_bias(d, distribution_id);
    }

    for (int row = 0; row < N; ++row) {
        for (int d = 0; d < D; ++d) {
            hc.x[(size_t)row * (size_t)D + (size_t)d] =
                make_x_value(rng, row, d, D, distribution_id);
        }
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    // Case data derives from PMPP_BENCH_SEED (whitened) so input values vary per
    // rollout; shape family (N / D / eps / distribution) stays fixed below.
    SplitMix64 seed_mix(pmpp::bench_seed(0x200000001b3ULL));
    uint64_t s = seed_mix.next_u64();

    cases.push_back(make_case(
        "bench_N65536_D256_uniform",
        65536,
        256,
        FLQP_DIST_UNIFORM,
        1.0e-5f,
        s++));

    cases.push_back(make_case(
        "bench_N16384_D1024_outliers",
        16384,
        1024,
        FLQP_DIST_OUTLIERS,
        1.0e-5f,
        s++));

    cases.push_back(make_case(
        "bench_N4096_D4096_zeroish",
        4096,
        4096,
        FLQP_DIST_ZEROISH,
        1.0e-7f,
        s++));

    cases.push_back(make_case(
        "bench_N131072_D256_outliers",
        131072,
        256,
        FLQP_DIST_OUTLIERS,
        1.0e-5f,
        s++));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int D = hc.run.D;

    dc->x.allocate(hc.x.size());
    dc->weight.allocate(hc.weight.size());
    dc->bias.allocate(hc.bias.size());

    dc->x.upload(hc.x);
    dc->weight.upload(hc.weight);
    dc->bias.upload(hc.bias);

    dc->q_int8.allocate((size_t)N * (size_t)D);
    dc->scale.allocate((size_t)N);
    dc->dequant.allocate((size_t)N * (size_t)D);
    dc->code_sum.allocate((size_t)N);

    dc->inputs = {};
    dc->inputs.x = dc->x.ptr;
    dc->inputs.weight = dc->weight.ptr;
    dc->inputs.bias = dc->bias.ptr;

    dc->outputs = {};
    dc->outputs.q_int8 = dc->q_int8.ptr;
    dc->outputs.scale = dc->scale.ptr;
    dc->outputs.dequant = dc->dequant.ptr;
    dc->outputs.code_sum = dc->code_sum.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        std::vector<HostCase> host_cases = build_bench_cases();

        int max_N = FLQP_MIN_N;
        int max_D = 256;

        for (const HostCase& hc : host_cases) {
            max_N = std::max(max_N, hc.run.N);
            max_D = std::max(max_D, hc.run.D);
        }

        FlqpProblemSpec spec = {};
        spec.abi_version = FLQP_ABI_VERSION;
        spec.max_N = max_N;
        spec.max_D = max_D;
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

        std::vector<DeviceCase*> cases;
        cases.reserve(host_cases.size());

        for (const HostCase& hc : host_cases) {
            cases.push_back(make_device_case(hc));
            std::printf(
                "bench_case %-32s N=%d D=%d eps=%.1e dist=%d variants=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.D,
                static_cast<double>(hc.run.eps),
                hc.run.distribution_id,
                iters);
        }

        // One probe slot per timed call (iter-major, case-minor), zeroed up front.
        DeviceBuffer<unsigned long long> probes;
        probes.allocate((size_t)iters * cases.size());
        CUDA_CHECK(cudaMemset(probes.ptr, 0,
                              (size_t)iters * cases.size() * sizeof(unsigned long long)));

        // Per-(case, iter) mutation salts from PMPP_BENCH_SEED: the timed input
        // sequence is deterministic per seed but differs on every timed call.
        std::vector<std::vector<uint64_t>> salts(cases.size());
        {
            SplitMix64 salt_mix(pmpp::bench_seed(0x200000001b3ULL) ^ 0xC3F01DED5EEDULL);
            for (size_t c = 0; c < cases.size(); ++c) {
                SplitMix64 cs(salt_mix.next_u64());
                salts[c].resize((size_t)iters);
                for (int k = 0; k < iters; ++k) {
                    salts[c][(size_t)k] = cs.next_u64();
                }
            }
        }

        // Warmup runs on the pristine x; every timed call below sees a freshly
        // mutated x, so warmup outputs can never satisfy a timed call's probe.
        for (int warm = 0; warm < 5; ++warm) {
            for (DeviceCase* dc : cases) {
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
            for (size_t c = 0; c < cases.size(); ++c) {
                DeviceCase* dc = cases[c];
                // Mutate the weight vector in place, run, then probe the fresh
                // outputs into this call's slot (both kernels are paired-cancelled:
                // the reference bench executes the identical sequence).
                flip_weight_subset<<<1, 256, 0, stream>>>(
                    dc->weight.ptr, dc->host.run.D, salts[c][(size_t)iter]);
                CUDA_CHECK(solution_run(
                    state,
                    &dc->host.run,
                    &dc->inputs,
                    &dc->outputs,
                    workspace.ptr,
                    workspace_bytes,
                    stream));
                const size_t N = (size_t)dc->host.run.N;
                const size_t D = (size_t)dc->host.run.D;
                probe_fold<<<256, 256, 0, stream>>>(
                    dc->q_int8.ptr,
                    N * D / 8,
                    dc->code_sum.ptr,
                    N,
                    probes.ptr + (size_t)iter * cases.size() + c);
            }
        }

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        const double calls = static_cast<double>(iters) * static_cast<double>(cases.size());
        const double avg_ms = static_cast<double>(elapsed_ms) / calls;

        std::printf("avg_ms=%.6f\n", avg_ms);

        // Anti-hack digest: the probe slots bind the exact-graded outputs (q_int8,
        // code_sum) of EVERY timed call to its own mutated input — a no-op/cached
        // call leaves a wrong slot value and the digest cannot match the reference
        // bench. Fold all slots, then anchor byte-exactness by folding the final
        // iteration's raw outputs (still resident in the shared buffers).
        // scale/dequant are tolerance-graded fp32, deliberately not folded.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        dg.dev(probes.ptr, (size_t)iters * cases.size() * sizeof(unsigned long long));
        for (DeviceCase* dc : cases) {
            const size_t N = (size_t)dc->host.run.N;
            const size_t D = (size_t)dc->host.run.D;
            dg.dev(dc->q_int8.ptr, N * D * sizeof(int8_t));
            dg.dev(dc->code_sum.ptr, N * sizeof(int64_t));
        }
        dg.print();

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (DeviceCase* dc : cases) {
            delete dc;
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
