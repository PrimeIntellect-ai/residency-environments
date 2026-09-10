// file: bench_mxfp4_atom_quant_dot.cu

#include "mxfp4_atom_quant_dot_common.h"
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

static constexpr uint64_t g_state = 0x51c9a3e7d2b84f61ULL;

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

// Exact t * 2^p for few-bit t (see contract).
static float pow2_scale(float t, int p) {
    const uint32_t u = f2u(t);
    const int E = (int)((u >> 23) & 0xffu);
    const uint32_t M = u & 0x7fffffu;
    const int e = E - 127 + p;
    if (e >= -126) return u2f(((uint32_t)(e + 127) << 23) | M);
    return u2f((0x800000u | M) >> (-126 - e));
}

static const float kMidpoints[7] = {
    0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f
};

static const float kSatFracs[6] = {
    1.0f, 1.25f, 1.5f, 1.625f, 1.75f, 1.875f
};

static void fill_block(
    SplitMix64& rng,
    int distribution_id,
    int b,
    int n,
    float* lane) {
    switch (distribution_id) {
        case MXQ_DIST_SMOOTH: {
            const int t = rng.uniform_int(-12, 12);
            for (int i = 0; i < n; ++i) {
                const int q = rng.uniform_int(-1536, 1536);
                lane[i] = (float)q * pow2f(t - 8);
            }
            break;
        }
        case MXQ_DIST_TIES: {
            const int t = rng.uniform_int(-16, 16);
            const int anchor = rng.uniform_int(0, n - 1);
            for (int i = 0; i < n; ++i) {
                if (i == anchor) {
                    lane[i] = pow2f(t + 2);
                } else {
                    const float m = kMidpoints[rng.uniform_int(0, 6)];
                    const float val = pow2_scale(m, t);
                    lane[i] = rng.uniform_int(0, 1) ? -val : val;
                }
            }
            break;
        }
        case MXQ_DIST_SATURATE: {
            const int t = rng.uniform_int(-10, 10);
            const int anchor = rng.uniform_int(0, n - 1);
            for (int i = 0; i < n; ++i) {
                float val;
                if (i == anchor) {
                    val = pow2_scale(1.75f, t);
                } else if (rng.uniform_int(0, 1)) {
                    val = pow2_scale(kSatFracs[rng.uniform_int(0, 5)], t);
                } else {
                    val = pow2_scale(1.25f, t - rng.uniform_int(1, 6));
                }
                lane[i] = rng.uniform_int(0, 1) ? -val : val;
            }
            break;
        }
        case MXQ_DIST_RANDBITS:
        default: {
            for (int i = 0; i < n; ++i) {
                uint32_t u = (uint32_t)rng.next_u64();
                // Cap the biased exponent at 0xF0 (contract domain guarantee).
                while (((u >> 23) & 0xffu) > 0xf0u) u -= (0x10u << 23);
                lane[i] = u2f(u);
            }
            break;
        }
    }
    (void)b;
}

// C3 per-iteration input variation (I2): before every timed call, flip the sign
// of a CONTIGUOUS rotating block of x in place ([off, off+blk), off from
// (PMPP_BENCH_SEED, case, iter)). A sign flip is XOR 0x80000000 — it preserves
// the biased exponent, so the mutated x stays in the contract's legal MXFP4 input
// domain for every distribution, while genuinely changing the per-32-block
// quantized payload/scale, row_dot and row_digest of the affected rows. A
// contiguous block keeps the touched memory small (vs a full-array pass), so the
// kernel is a small fraction of one solution_run. Paired-cancelled: the reference
// bench runs the identical kernel with the identical off.
__global__ void mxq_flip_block(float* x, size_t off, size_t blk) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t* xu = reinterpret_cast<uint32_t*>(x);
    for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < blk; j += stride) {
        xu[off + j] ^= 0x80000000u;
    }
}

struct HostCase {
    std::string name;
    MxqRunSpec run;
    std::vector<float> x;
    std::vector<float> v;
};

static HostCase make_case(
    const std::string& name,
    int R,
    int C,
    int distribution_id,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = MXQ_ABI_VERSION;
    hc.run.R = R;
    hc.run.C = C;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!mxq_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid MxqRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.x.resize((size_t)R * (size_t)C);
    hc.v.resize((size_t)C);

    for (int k = 0; k < C; ++k) {
        hc.v[(size_t)k] = 0.25f * (float)rng.uniform_int(-8, 8);
    }

    const int S = mxq_S(C);
    float lane[MXQ_BLOCK_K];

    for (int r = 0; r < R; ++r) {
        for (int b = 0; b < S; ++b) {
            const int k0 = 32 * b;
            const int n = (k0 + 32 <= C) ? 32 : (C - k0);
            fill_block(rng, distribution_id, b, n, lane);
            for (int i = 0; i < n; ++i) {
                hc.x[(size_t)r * (size_t)C + (size_t)(k0 + i)] = lane[i];
            }
        }
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    // Case data derives from PMPP_BENCH_SEED (whitened) so input values vary per
    // rollout; shape family (R / C / distribution) stays fixed below.
    SplitMix64 seed_mix(pmpp::bench_seed(0xA00000001b3ULL));
    uint64_t s = seed_mix.next_u64();

    cases.push_back(make_case("bench_R8192_C4096_smooth", 8192, 4096, MXQ_DIST_SMOOTH, s++));
    cases.push_back(make_case("bench_R16384_C2048_randbits", 16384, 2048, MXQ_DIST_RANDBITS, s++));
    cases.push_back(make_case("bench_R4096_C6143_saturate", 4096, 6143, MXQ_DIST_SATURATE, s++));
    cases.push_back(make_case("bench_R2048_C1024_ties", 2048, 1024, MXQ_DIST_TIES, s++));

    return cases;
}

// One output set per timed call (+1 warmup) so the digest can bind every timed
// call. x/v are shared and x is mutated in place per call (see mxq_flip_block).
struct OutSet {
    DeviceBuffer<uint8_t> pay_atoms;
    DeviceBuffer<uint8_t> sf_atoms;
    DeviceBuffer<float> row_dot;
    DeviceBuffer<uint64_t> row_digest;
    DeviceBuffer<int32_t> sat_count;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<float> x;
    DeviceBuffer<float> v;

    std::vector<OutSet*> outs;

    MxqInputs inputs;
    MxqOutputs outputs;

    ~DeviceCase() {
        for (OutSet* o : outs) delete o;
    }
};

static DeviceCase* make_device_case(const HostCase& hc, int iters) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int R = hc.run.R;
    const int C = hc.run.C;

    dc->x.allocate(hc.x.size());
    dc->v.allocate(hc.v.size());
    dc->x.upload(hc.x);
    dc->v.upload(hc.v);

    dc->outs.resize((size_t)iters + 1);
    for (int k = 0; k <= iters; ++k) {
        OutSet* o = new OutSet();
        o->pay_atoms.allocate(mxq_pay_atom_bytes(R, C));
        o->sf_atoms.allocate(mxq_sf_atom_bytes(R, C));
        o->row_dot.allocate((size_t)R);
        o->row_digest.allocate((size_t)R);
        o->sat_count.allocate((size_t)R);
        dc->outs[(size_t)k] = o;
    }

    dc->inputs = {};
    dc->inputs.x = dc->x.ptr;
    dc->inputs.v = dc->v.ptr;
    dc->outputs = {};

    return dc;
}

// Point dc->outputs at output set k.
static void bind_out(DeviceCase* dc, int k) {
    OutSet* o = dc->outs[(size_t)k];
    dc->outputs.pay_atoms = o->pay_atoms.ptr;
    dc->outputs.sf_atoms = o->sf_atoms.ptr;
    dc->outputs.row_dot = o->row_dot.ptr;
    dc->outputs.row_digest = o->row_digest.ptr;
    dc->outputs.sat_count = o->sat_count.ptr;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        std::vector<HostCase> host_cases = build_bench_cases();

        int max_R = MXQ_MIN_R;
        int max_C = MXQ_MIN_C;

        for (const HostCase& hc : host_cases) {
            max_R = std::max(max_R, hc.run.R);
            max_C = std::max(max_C, hc.run.C);
        }

        MxqProblemSpec spec = {};
        spec.abi_version = MXQ_ABI_VERSION;
        spec.max_R = max_R;
        spec.max_C = max_C;
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
                "bench_case %-28s R=%d C=%d dist=%d variants=%d\n",
                hc.name.c_str(),
                hc.run.R,
                hc.run.C,
                hc.run.distribution_id,
                iters);
        }

        // Per-(case, iter) block offsets from PMPP_BENCH_SEED: deterministic per
        // seed, different on every timed call.
        std::vector<std::vector<uint64_t>> offs(cases.size());
        {
            SplitMix64 salt_mix(pmpp::bench_seed(0xA00000001b3ULL) ^ 0xC3F01DED5EEDULL);
            for (size_t c = 0; c < cases.size(); ++c) {
                SplitMix64 cm(salt_mix.next_u64());
                offs[c].resize((size_t)iters);
                for (int k = 0; k < iters; ++k) offs[c][(size_t)k] = cm.next_u64();
            }
        }

        // Warmup runs on the pristine x into the dedicated warmup output set (index
        // iters); every timed call below mutates x first, so warmup output can
        // never satisfy a timed call's fold.
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
                const size_t total = dc->x.count;
                const size_t blk = std::max<size_t>(total / 32, 1);
                size_t off = offs[c][(size_t)iter] % total;
                if (off + blk > total) off = total - blk;
                // Mutate x in place, then run into this iteration's output set.
                mxq_flip_block<<<256, 256, 0, stream>>>(dc->x.ptr, off, blk);
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
        // buffer into its OWN set for its own mutated x; fold ALL of them (all
        // checkers are exact, no tolerances). A no-op/cached timed call leaves a
        // stale set and the digest cannot match the reference bench.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        for (int iter = 0; iter < iters; ++iter) {
            for (DeviceCase* dc : cases) {
                const int R = dc->host.run.R;
                const int C = dc->host.run.C;
                OutSet* o = dc->outs[(size_t)iter];
                dg.dev(o->pay_atoms.ptr, mxq_pay_atom_bytes(R, C));
                dg.dev(o->sf_atoms.ptr, mxq_sf_atom_bytes(R, C));
                dg.dev(o->row_dot.ptr, (size_t)R * sizeof(float));
                dg.dev(o->row_digest.ptr, (size_t)R * sizeof(uint64_t));
                dg.dev(o->sat_count.ptr, (size_t)R * sizeof(int32_t));
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
