// ============================================================================
// file: bench_fused_gemm_nvfp4_epilogue.cu
// ============================================================================

#include "fused_gemm_nvfp4_epilogue_common.h"
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

static constexpr uint64_t g_state = 0x7a6f5e2d3c1b9048ULL;

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

    bool chance_permille(int p) {
        if (p <= 0) return false;
        if (p >= 1000) return true;
        return static_cast<int>(next_u64() % 1000ULL) < p;
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

// C3 per-iteration input variation (I1): pregenerate one a_packed variant per
// timed call (a_packed is tiny and iters is small, so K device copies are cheap).
// Variant k is the base with a CONTIGUOUS block [off_k, off_k+blk) of bytes XORed
// by 0xFF — each packed 4-bit code n becomes 15-n, which stays in the legal [0,15]
// code domain for every distribution while genuinely changing the dequantized GEMM
// contribution (and thus c_out) of the affected rows. Offsets derive from
// (PMPP_BENCH_SEED, case, iter). The timed loop only swaps the input pointer, so it
// adds ZERO per-call work (no extra kernel launch). Same seed -> bit-identical
// variant sequence in the student and reference benches.
static std::vector<uint8_t> make_flipped_variant(
    const std::vector<uint8_t>& base, uint64_t off_seed) {
    std::vector<uint8_t> v = base;
    const size_t total = v.size();
    if (total == 0) return v;
    const size_t blk = std::max<size_t>(total / 32, 1);
    size_t off = (size_t)(off_seed % (uint64_t)total);
    if (off + blk > total) off = total - blk;
    for (size_t j = 0; j < blk; ++j) v[off + j] ^= 0xFFu;
    return v;
}

struct HostCase {
    std::string name;
    FgeRunSpec run;
    std::vector<uint8_t> a_packed;
    std::vector<uint8_t> b_packed;
    std::vector<int16_t> a_scale_q;
    std::vector<int16_t> b_scale_q;
    std::vector<int32_t> bias;
};

struct DeviceCase {
    HostCase host;

    // One a_packed variant per timed call (+1 warmup); the timed loop swaps the
    // input pointer between them (see C3 note above).
    std::vector<DeviceBuffer<uint8_t>*> a_packed_v;
    DeviceBuffer<uint8_t> b_packed;
    DeviceBuffer<int16_t> a_scale_q;
    DeviceBuffer<int16_t> b_scale_q;
    DeviceBuffer<int32_t> bias;

    // One c_out per timed call (+1 warmup) so the digest can bind every timed call.
    std::vector<DeviceBuffer<int32_t>*> c_out_v;

    FgeInputs inputs;
    FgeOutputs outputs;

    ~DeviceCase() {
        for (DeviceBuffer<uint8_t>* b : a_packed_v) delete b;
        for (DeviceBuffer<int32_t>* b : c_out_v) delete b;
    }
};

static void set_packed_code(std::vector<uint8_t>* bytes, size_t logical_idx, uint8_t code) {
    const size_t byte_idx = logical_idx >> 1;
    const uint8_t c = code & 0x0f;

    if ((logical_idx & 1u) == 0u) {
        (*bytes)[byte_idx] = static_cast<uint8_t>(((*bytes)[byte_idx] & 0xf0u) | c);
    } else {
        (*bytes)[byte_idx] = static_cast<uint8_t>(((*bytes)[byte_idx] & 0x0fu) | (c << 4));
    }
}

static uint8_t random_code(SplitMix64& rng, int distribution_id, bool saturating) {
    if (saturating) return rng.chance_permille(500) ? 7u : 15u;

    switch (distribution_id) {
        case FGE_DIST_UNIFORM:
            return static_cast<uint8_t>(rng.uniform_int(0, 15));

        case FGE_DIST_PEAKED:
            if (rng.chance_permille(720)) return 0u;
            if (rng.chance_permille(650)) return 7u;
            return static_cast<uint8_t>(rng.uniform_int(0, 15));

        case FGE_DIST_MANY_ZERO:
            if (rng.chance_permille(850)) return 0u;
            return static_cast<uint8_t>(rng.uniform_int(0, 15));

        case FGE_DIST_MANY_TIES: {
            static const uint8_t codes[] = {0, 1, 1, 2, 2, 9, 9, 15};
            return codes[rng.uniform_int(0, 7)];
        }

        default:
            return static_cast<uint8_t>(rng.uniform_int(0, 15));
    }
}

static HostCase make_case(
    const char* name,
    int M,
    int N,
    int K,
    int shift,
    int activation,
    int distribution_id,
    uint64_t seed,
    bool negative_scales,
    bool saturating_codes) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = FGE_ABI_VERSION;
    hc.run.M = M;
    hc.run.N = N;
    hc.run.K = K;
    hc.run.epilogue_shift = shift;
    hc.run.activation = activation;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!fge_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated FgeRunSpec");
    }

    SplitMix64 rng(g_state ^ seed);

    const size_t a_elems = (size_t)M * (size_t)K;
    const size_t b_elems = (size_t)K * (size_t)N;

    hc.a_packed.assign(fge_packed_bytes_for(a_elems), 0);
    hc.b_packed.assign(fge_packed_bytes_for(b_elems), 0);
    hc.a_scale_q.resize((size_t)M);
    hc.b_scale_q.resize((size_t)N);
    hc.bias.resize((size_t)N);

    for (size_t i = 0; i < a_elems; ++i) {
        set_packed_code(&hc.a_packed, i, random_code(rng, distribution_id, saturating_codes));
    }

    for (size_t i = 0; i < b_elems; ++i) {
        set_packed_code(&hc.b_packed, i, random_code(rng, distribution_id, saturating_codes));
    }

    for (int m = 0; m < M; ++m) {
        int v = rng.uniform_int(1, 16);
        if (negative_scales && m % 7 == 0) v = -v;
        hc.a_scale_q[(size_t)m] = static_cast<int16_t>(v);
    }

    for (int n = 0; n < N; ++n) {
        int v = rng.uniform_int(1, 16);
        if (negative_scales && n % 11 == 0) v = -v;
        hc.b_scale_q[(size_t)n] = static_cast<int16_t>(v);
        hc.bias[(size_t)n] = static_cast<int32_t>(rng.uniform_int(-500000, 500000));
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    // Case data derives from PMPP_BENCH_SEED (whitened) so input values vary per
    // rollout; shape family (M / N / K / act / distribution) stays fixed below.
    SplitMix64 seed_mix(pmpp::bench_seed(0x200000001b3ULL));
    uint64_t s = seed_mix.next_u64();

    cases.push_back(make_case(
        "uniform_384x320x192",
        384, 320, 192, 8, FGE_ACT_NONE, FGE_DIST_UNIFORM,
        s++, false, false));

    cases.push_back(make_case(
        "zero_512x512x320_relu",
        512, 512, 320, 10, FGE_ACT_RELU, FGE_DIST_MANY_ZERO,
        s++, false, false));

    cases.push_back(make_case(
        "ties_768x512x640_clamp",
        768, 512, 640, 9, FGE_ACT_CLAMP_INT8_RANGE, FGE_DIST_MANY_TIES,
        s++, true, false));

    cases.push_back(make_case(
        "sat_1024x512x960_clamp",
        1024, 512, 960, 12, FGE_ACT_CLAMP_INT8_RANGE, FGE_DIST_PEAKED,
        s++, true, true));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc, int iters,
                                    const std::vector<uint64_t>& offs) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int M = hc.run.M;
    const int N = hc.run.N;

    dc->b_packed.allocate(hc.b_packed.size());
    dc->a_scale_q.allocate(hc.a_scale_q.size());
    dc->b_scale_q.allocate(hc.b_scale_q.size());
    dc->bias.allocate(hc.bias.size());

    // Variant iters is the warmup variant (pristine base); variants 0..iters-1 are
    // the per-timed-call flipped copies keyed by offs[k].
    dc->a_packed_v.resize((size_t)iters + 1);
    for (int k = 0; k <= iters; ++k) {
        dc->a_packed_v[(size_t)k] = new DeviceBuffer<uint8_t>();
        dc->a_packed_v[(size_t)k]->allocate(hc.a_packed.size());
        if (k == iters) {
            dc->a_packed_v[(size_t)k]->upload(hc.a_packed);
        } else {
            dc->a_packed_v[(size_t)k]->upload(
                make_flipped_variant(hc.a_packed, offs[(size_t)k]));
        }
    }

    dc->c_out_v.resize((size_t)iters + 1);
    for (int k = 0; k <= iters; ++k) {
        dc->c_out_v[(size_t)k] = new DeviceBuffer<int32_t>();
        dc->c_out_v[(size_t)k]->allocate((size_t)M * (size_t)N);
    }

    dc->b_packed.upload(hc.b_packed);
    dc->a_scale_q.upload(hc.a_scale_q);
    dc->b_scale_q.upload(hc.b_scale_q);
    dc->bias.upload(hc.bias);

    dc->inputs = {};
    dc->inputs.b_packed = dc->b_packed.ptr;
    dc->inputs.a_scale_q = dc->a_scale_q.ptr;
    dc->inputs.b_scale_q = dc->b_scale_q.ptr;
    dc->inputs.bias = dc->bias.ptr;

    dc->outputs = {};

    return dc;
}

// Point dc->inputs.a_packed at variant k and dc->outputs.c_out at output set k.
static void bind_variant(DeviceCase* dc, int k) {
    dc->inputs.a_packed = dc->a_packed_v[(size_t)k]->ptr;
    dc->outputs.c_out = dc->c_out_v[(size_t)k]->ptr;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 5;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        std::vector<HostCase> host_cases = build_bench_cases();

        int max_M = 64;
        int max_N = 64;
        int max_K = 64;

        for (const HostCase& hc : host_cases) {
            max_M = std::max(max_M, hc.run.M);
            max_N = std::max(max_N, hc.run.N);
            max_K = std::max(max_K, hc.run.K);
        }

        FgeProblemSpec spec = {};
        spec.abi_version = FGE_ABI_VERSION;
        spec.max_M = max_M;
        spec.max_N = max_N;
        spec.max_K = max_K;
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

        // Per-(case, iter) block offsets from PMPP_BENCH_SEED: deterministic per
        // seed, different on every timed call. Computed before the device cases so
        // each case can pregenerate its flipped a_packed variants.
        std::vector<std::vector<uint64_t>> offs(host_cases.size());
        {
            SplitMix64 salt_mix(pmpp::bench_seed(0x200000001b3ULL) ^ 0xC3F01DED5EEDULL);
            for (size_t c = 0; c < host_cases.size(); ++c) {
                SplitMix64 cm(salt_mix.next_u64());
                offs[c].resize((size_t)iters);
                for (int k = 0; k < iters; ++k) offs[c][(size_t)k] = cm.next_u64();
            }
        }

        for (size_t ci = 0; ci < host_cases.size(); ++ci) {
            const HostCase& hc = host_cases[ci];
            cases.push_back(make_device_case(hc, iters, offs[ci]));
            std::printf(
                "bench_case %-28s M=%d N=%d K=%d shift=%d act=%d dist=%d variants=%d\n",
                hc.name.c_str(),
                hc.run.M,
                hc.run.N,
                hc.run.K,
                hc.run.epilogue_shift,
                hc.run.activation,
                hc.run.distribution_id,
                iters);
        }

        // Warmup runs on the pristine variant (index iters) into the dedicated
        // warmup output; every timed call below uses a distinct flipped variant, so
        // warmup output can never satisfy a timed call's fold.
        for (int warm = 0; warm < 3; ++warm) {
            for (DeviceCase* dc : cases) {
                bind_variant(dc, iters);
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
            for (DeviceCase* dc : cases) {
                // Swap to this iteration's input variant + output region (no extra
                // kernel launch: pure pointer swap, so zero added per-call work).
                bind_variant(dc, iter);
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

        // Anti-hack digest: every timed call (iter, case) wrote c_out (exact int32,
        // byte-exact vs the oracle) into its OWN region for its own mutated a_packed;
        // fold ALL of them. A no-op/cached timed call leaves a stale region and the
        // digest cannot match the reference bench.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        for (int iter = 0; iter < iters; ++iter) {
            for (DeviceCase* dc : cases) {
                dg.dev(dc->c_out_v[(size_t)iter]->ptr,
                       (size_t)dc->host.run.M * (size_t)dc->host.run.N * sizeof(int32_t));
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
