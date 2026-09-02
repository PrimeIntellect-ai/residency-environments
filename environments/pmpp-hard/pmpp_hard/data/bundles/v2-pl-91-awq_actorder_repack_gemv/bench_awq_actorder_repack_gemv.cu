// file: bench_awq_actorder_repack_gemv.cu

#include "awq_actorder_repack_gemv_common.h"
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

static constexpr uint64_t g_state = 0x7e91c44ab0d25f13ULL;

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

static uint32_t f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

static float u2f(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static float pow2f(int p) {
    if (p >= -126) return u2f((uint32_t)(p + 127) << 23);
    return u2f(0x800000u >> (-126 - p));
}

static float capped_randbits(SplitMix64& rng) {
    uint32_t u = (uint32_t)rng.next_u64();
    uint32_t e = (u >> 23) & 0xffu;
    if (e > 0x9fu) {
        e %= 0xa0u;
        u = (u & ~(0xffu << 23)) | (e << 23);
    }
    return u2f(u);
}

static void gen_g_idx(
    SplitMix64& rng, int distribution_id, int K, int G, int32_t* g_idx) {
    switch (distribution_id) {
        case AWQ_DIST_REVERSED: {
            for (int k = 0; k < K; ++k) {
                g_idx[k] = (G - 1) - (int)(((long long)k * G) / K);
            }
            break;
        }
        case AWQ_DIST_SKEWED: {
            for (int k = 0; k < K; ++k) {
                const double u =
                    (double)(rng.next_u64() >> 11) * (1.0 / 9007199254740992.0);
                int g = (int)((double)G * u * u);
                if (g > G - 1) g = G - 1;
                g_idx[k] = g;
            }
            break;
        }
        case AWQ_DIST_ALLSAME: {
            for (int k = 0; k < K; ++k) g_idx[k] = G / 2;
            break;
        }
        default: {
            for (int k = 0; k < K; ++k) {
                g_idx[k] = (int)(rng.next_u64() % (uint64_t)G);
            }
            break;
        }
    }
}

static float gen_scale(SplitMix64& rng, int distribution_id) {
    const uint32_t sign = ((uint32_t)rng.uniform_int(0, 1)) << 31;
    if (distribution_id == AWQ_DIST_RANDBITS) return capped_randbits(rng);
    const uint32_t e = (uint32_t)rng.uniform_int(121, 130);
    const uint32_t m = (uint32_t)(rng.next_u64() & 0x7fffffu);
    return u2f(sign | (e << 23) | m);
}

static float gen_x(SplitMix64& rng, int distribution_id) {
    if (distribution_id == AWQ_DIST_RANDBITS) return capped_randbits(rng);
    const int q = rng.uniform_int(-1536, 1536);
    const int t = rng.uniform_int(-10, 2);
    return (float)q * pow2f(t - 8);
}

// C3 per-iteration input variation (I2): before every timed call, XOR a
// CONTIGUOUS rotating block of qweight words with 0xFFFFFFFF ([off, off+blk),
// off from (PMPP_BENCH_SEED, case, iter)). Each packed 4-bit weight n becomes
// 15-n, which stays in the legal [0,15] nibble domain for every distribution,
// while genuinely changing the requantized atoms / col_dot / col_zsum of the
// affected columns. A contiguous block keeps the touched memory small, so the
// kernel is a small fraction of one solution_run. Paired-cancelled: the reference
// bench runs the identical kernel with the identical off.
__global__ void awq_flip_block(uint32_t* qweight, size_t off, size_t blk) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < blk; j += stride) {
        qweight[off + j] ^= 0xFFFFFFFFu;
    }
}

struct HostCase {
    std::string name;
    AwqRunSpec run;
    std::vector<uint32_t> qweight;
    std::vector<uint32_t> qzeros;
    std::vector<float> scales;
    std::vector<int32_t> g_idx;
    std::vector<float> x;
};

static HostCase make_case(
    const std::string& name,
    int K,
    int N,
    int G,
    int distribution_id,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = AWQ_ABI_VERSION;
    hc.run.K = K;
    hc.run.N = N;
    hc.run.G = G;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!awq_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid AwqRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    const int Nw = awq_Nw(N);

    hc.g_idx.resize((size_t)K);
    gen_g_idx(rng, distribution_id, K, G, hc.g_idx.data());

    hc.scales.resize((size_t)G * (size_t)N);
    for (size_t i = 0; i < hc.scales.size(); ++i) {
        hc.scales[i] = gen_scale(rng, distribution_id);
    }

    hc.x.resize((size_t)K);
    for (int k = 0; k < K; ++k) hc.x[(size_t)k] = gen_x(rng, distribution_id);

    hc.qweight.assign((size_t)K * (size_t)Nw, 0u);
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < Nw; ++c) {
            uint32_t word = 0;
            for (int t = 0; t < 8; ++t) {
                const int n = 8 * c + t;
                const uint32_t nib =
                    (n < N) ? (uint32_t)(rng.next_u64() & 0xF) : 0u;
                word |= nib << (4 * awq_wlane(t));
            }
            hc.qweight[(size_t)k * Nw + c] = word;
        }
    }
    hc.qzeros.assign((size_t)G * (size_t)Nw, 0u);
    for (int g = 0; g < G; ++g) {
        for (int c = 0; c < Nw; ++c) {
            uint32_t word = 0;
            for (int t = 0; t < 8; ++t) {
                const int n = 8 * c + t;
                const uint32_t nib =
                    (n < N) ? (uint32_t)(rng.next_u64() & 0xF) : 0u;
                word |= nib << (4 * awq_zlane(t));
            }
            hc.qzeros[(size_t)g * Nw + c] = word;
        }
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    // Case data derives from PMPP_BENCH_SEED (whitened) so input values vary per
    // rollout; shape family (K / N / G / distribution) stays fixed below.
    SplitMix64 seed_mix(pmpp::bench_seed(0xC10000001b3ULL));
    uint64_t s = seed_mix.next_u64();

    cases.push_back(make_case("bench_K8192_N4096_G128_balanced", 8192, 4096, 128, AWQ_DIST_BALANCED, s++));
    cases.push_back(make_case("bench_K8191_N2048_G512_randbits", 8191, 2048, 512, AWQ_DIST_RANDBITS, s++));
    cases.push_back(make_case("bench_K4096_N4096_G64_skewed", 4096, 4096, 64, AWQ_DIST_SKEWED, s++));
    cases.push_back(make_case("bench_K2048_N1024_G32_reversed", 2048, 1024, 32, AWQ_DIST_REVERSED, s++));

    return cases;
}

// One output set per timed call (+1 warmup) so the digest can bind every timed
// call. Inputs are shared and qweight is mutated in place per call.
struct OutSet {
    DeviceBuffer<uint8_t> rq_atoms;
    DeviceBuffer<float> col_dot;
    DeviceBuffer<uint64_t> col_digest;
    DeviceBuffer<int32_t> col_zsum;
    DeviceBuffer<int32_t> perm;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<uint32_t> qweight;
    DeviceBuffer<uint32_t> qzeros;
    DeviceBuffer<float> scales;
    DeviceBuffer<int32_t> g_idx;
    DeviceBuffer<float> x;

    std::vector<OutSet*> outs;

    AwqInputs inputs;
    AwqOutputs outputs;

    ~DeviceCase() {
        for (OutSet* o : outs) delete o;
    }
};

static DeviceCase* make_device_case(const HostCase& hc, int iters) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int K = hc.run.K;
    const int N = hc.run.N;

    dc->qweight.allocate(hc.qweight.size());
    dc->qzeros.allocate(hc.qzeros.size());
    dc->scales.allocate(hc.scales.size());
    dc->g_idx.allocate(hc.g_idx.size());
    dc->x.allocate(hc.x.size());
    dc->qweight.upload(hc.qweight);
    dc->qzeros.upload(hc.qzeros);
    dc->scales.upload(hc.scales);
    dc->g_idx.upload(hc.g_idx);
    dc->x.upload(hc.x);

    dc->outs.resize((size_t)iters + 1);
    for (int k = 0; k <= iters; ++k) {
        OutSet* o = new OutSet();
        o->rq_atoms.allocate(awq_rq_bytes(K, N));
        o->col_dot.allocate((size_t)N);
        o->col_digest.allocate((size_t)N);
        o->col_zsum.allocate((size_t)N);
        o->perm.allocate((size_t)K);
        dc->outs[(size_t)k] = o;
    }

    dc->inputs = {};
    dc->inputs.qweight = dc->qweight.ptr;
    dc->inputs.qzeros = dc->qzeros.ptr;
    dc->inputs.scales = dc->scales.ptr;
    dc->inputs.g_idx = dc->g_idx.ptr;
    dc->inputs.x = dc->x.ptr;
    dc->outputs = {};

    return dc;
}

// Point dc->outputs at output set k.
static void bind_out(DeviceCase* dc, int k) {
    OutSet* o = dc->outs[(size_t)k];
    dc->outputs.rq_atoms = o->rq_atoms.ptr;
    dc->outputs.col_dot = o->col_dot.ptr;
    dc->outputs.col_digest = o->col_digest.ptr;
    dc->outputs.col_zsum = o->col_zsum.ptr;
    dc->outputs.perm = o->perm.ptr;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        std::vector<HostCase> host_cases = build_bench_cases();

        int max_K = AWQ_MIN_K;
        int max_N = AWQ_MIN_N;
        int max_G = AWQ_MIN_G;

        for (const HostCase& hc : host_cases) {
            max_K = std::max(max_K, hc.run.K);
            max_N = std::max(max_N, hc.run.N);
            max_G = std::max(max_G, hc.run.G);
        }

        AwqProblemSpec spec = {};
        spec.abi_version = AWQ_ABI_VERSION;
        spec.max_K = max_K;
        spec.max_N = max_N;
        spec.max_G = max_G;
        spec.flags = 0;

        size_t workspace_bytes = solution_workspace_bytes(&spec);
        if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/

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
            cases.push_back(make_device_case(hc, iters));
            std::printf(
                "bench_case %-32s K=%d N=%d G=%d dist=%d variants=%d\n",
                hc.name.c_str(),
                hc.run.K,
                hc.run.N,
                hc.run.G,
                hc.run.distribution_id,
                iters);
        }

        // Per-(case, iter) block offsets from PMPP_BENCH_SEED: deterministic per
        // seed, different on every timed call.
        std::vector<std::vector<uint64_t>> offs(cases.size());
        {
            SplitMix64 salt_mix(pmpp::bench_seed(0xC10000001b3ULL) ^ 0xC3F01DED5EEDULL);
            for (size_t c = 0; c < cases.size(); ++c) {
                SplitMix64 cm(salt_mix.next_u64());
                offs[c].resize((size_t)iters);
                for (int k = 0; k < iters; ++k) offs[c][(size_t)k] = cm.next_u64();
            }
        }

        // Warmup runs on pristine qweight into the dedicated warmup output set
        // (index iters); every timed call below mutates qweight first, so warmup
        // output can never satisfy a timed call's fold.
        for (int warm = 0; warm < 5; ++warm) {
            for (DeviceCase* dc : cases) {
                bind_out(dc, iters);
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
                const size_t total = dc->qweight.count;
                const size_t blk = std::max<size_t>(total / 32, 1);
                size_t off = offs[c][(size_t)iter] % total;
                if (off + blk > total) off = total - blk;
                // Mutate qweight in place, then run into this iteration's output set.
                awq_flip_block<<<256, 256, 0, stream>>>(dc->qweight.ptr, off, blk);
                bind_out(dc, iter);
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

        const double calls = static_cast<double>(iters) * static_cast<double>(cases.size());
        const double avg_ms = static_cast<double>(elapsed_ms) / calls;

        std::printf("avg_ms=%.6f\n", avg_ms);

        // Anti-hack digest: every timed call (iter, case) wrote every graded output
        // buffer into its OWN set for its own mutated qweight; fold ALL of them (all
        // checkers are exact, no tolerances). A no-op/cached timed call leaves a
        // stale set and the digest cannot match the reference bench.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        for (int iter = 0; iter < iters; ++iter) {
            for (DeviceCase* dc : cases) {
                const int K = dc->host.run.K;
                const int N = dc->host.run.N;
                OutSet* o = dc->outs[(size_t)iter];
                dg.dev(o->rq_atoms.ptr, awq_rq_bytes(K, N));
                dg.dev(o->col_dot.ptr, (size_t)N * sizeof(float));
                dg.dev(o->col_digest.ptr, (size_t)N * sizeof(uint64_t));
                dg.dev(o->col_zsum.ptr, (size_t)N * sizeof(int32_t));
                dg.dev(o->perm.ptr, (size_t)K * sizeof(int32_t));
            }
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
