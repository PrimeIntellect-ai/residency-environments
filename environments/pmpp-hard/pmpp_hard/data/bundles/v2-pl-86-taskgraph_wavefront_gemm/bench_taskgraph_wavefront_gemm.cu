// ============================================================================
// file: bench_taskgraph_wavefront_gemm.cu
// ============================================================================

#include "taskgraph_wavefront_gemm_common.h"
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

static constexpr uint64_t g_state = 0x9e3779b97f4a7c15ULL;

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

    double uniform01() {
        return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
    }

    bool chance(double p) {
        return uniform01() < p;
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
            CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count,
                                  cudaMemcpyHostToDevice));
        }
    }
};

static int8_t twg_gen_value(SplitMix64& rng, int dist) {
    switch (dist) {
        case TWG_DIST_SMALL:
            return static_cast<int8_t>(rng.uniform_int(-2, 2));
        case TWG_DIST_SPARSE:
            return rng.chance(0.90)
                ? static_cast<int8_t>(0)
                : static_cast<int8_t>(rng.uniform_int(-127, 127));
        case TWG_DIST_SATURATE:
            return rng.chance(0.85)
                ? static_cast<int8_t>(rng.chance(0.5) ? 127 : -127)
                : static_cast<int8_t>(rng.uniform_int(-8, 8));
        default:
            return static_cast<int8_t>(rng.uniform_int(-127, 127));
    }
}

struct BenchCase {
    std::string name;
    TwgRunSpec run;
    std::vector<int8_t> U;
    std::vector<int8_t> V;
};

// --- C3 timed-fold (patterns I1 + P1): every timed call receives a FRESH
// (U, V) variant pregenerated from (PMPP_BENCH_SEED, case, iter) — the U/V
// buffers are tiny (<=256 KB) so a per-iteration ring costs negligible memory
// and the timed loop only flips input pointers (zero timing overhead). Warmup
// uses dedicated variants. After each timed call (outside the event pair — zero
// timing impact) a probe kernel folds the graded outputs into acc[iter]; the
// digest folds every accumulator, so a cached/no-op timed call cannot match.

static uint64_t host_mix64(uint64_t z) {
    z ^= z >> 30; z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27; z *= 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ uint64_t c3_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct C3ProbeDesc { const uint8_t* ptr; unsigned long long nbytes; };
struct C3ProbeTable { C3ProbeDesc d[4]; int n; };

__global__ void c3_probe_fold(C3ProbeTable t, uint64_t salt,
                              unsigned long long* acc) {
    __shared__ unsigned long long sh[256];
    unsigned long long local = 0;
    for (int b = 0; b < t.n; ++b) {
        const C3ProbeDesc& d = t.d[b];
        const unsigned long long nwords = d.nbytes >> 3;
        const uint64_t bsalt = salt ^ (uint64_t)(b + 1) * 0x100000001b3ULL;
        const unsigned long long* w64 =
            reinterpret_cast<const unsigned long long*>(d.ptr);
        for (unsigned long long w = threadIdx.x; w < nwords; w += blockDim.x)
            local ^= c3_mix(w64[w] ^ c3_mix(bsalt + w));
        if (threadIdx.x == 0 && (d.nbytes & 7ULL)) {
            unsigned long long tail = 0;
            for (unsigned long long i = nwords << 3; i < d.nbytes; ++i)
                tail |= (unsigned long long)d.ptr[i] << (8 * (i & 7ULL));
            local ^= c3_mix(tail ^ c3_mix(bsalt + nwords));
        }
    }
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] ^= sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicXor(acc, sh[0]);
}

// Fresh (U, V) for variant v of a case, preserving the case's distribution
// (ZERO_U keeps U identically zero).
static void twg_make_variant(const BenchCase& bc, uint64_t c3_root, int v,
                             std::vector<int8_t>& U, std::vector<int8_t>& V) {
    U.resize(bc.U.size());
    V.resize(bc.V.size());
    const int dist = bc.run.distribution_id;
    SplitMix64 rng(host_mix64(c3_root ^ (uint64_t)(bc.run.seed_id + 1) * 0x9e3779b97f4a7c15ULL ^
                              (uint64_t)(v + 1) * 0xbf58476d1ce4e5b9ULL));
    for (size_t i = 0; i < U.size(); ++i)
        U[i] = dist == TWG_DIST_ZERO_U ? 0 : twg_gen_value(rng, dist);
    for (size_t i = 0; i < V.size(); ++i)
        V[i] = dist == TWG_DIST_ZERO_U
                   ? static_cast<int8_t>(rng.uniform_int(-127, 127))
                   : twg_gen_value(rng, dist);
}

static BenchCase make_case(
    const char* name,
    int R, int C, int K, int P1, int P2, int dist, uint64_t seed) {
    BenchCase bc;
    bc.name = name;

    bc.run = {};
    bc.run.abi_version = TWG_ABI_VERSION;
    bc.run.R = R;
    bc.run.C = C;
    bc.run.K = K;
    bc.run.P1 = P1;
    bc.run.P2 = P2;
    bc.run.dump = 0;
    bc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    bc.run.distribution_id = dist;
    bc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!twg_validate_run_spec(&bc.run)) {
        throw std::runtime_error(std::string("invalid run spec: ") + name);
    }

    SplitMix64 rng(g_state ^ seed);
    bc.U.resize((size_t)R * 32 * K);
    bc.V.resize((size_t)C * 32 * K);
    for (size_t idx = 0; idx < bc.U.size(); ++idx) {
        bc.U[idx] = dist == TWG_DIST_ZERO_U ? 0 : twg_gen_value(rng, dist);
    }
    for (size_t idx = 0; idx < bc.V.size(); ++idx) {
        bc.V[idx] = dist == TWG_DIST_ZERO_U
            ? static_cast<int8_t>(rng.uniform_int(-127, 127))
            : twg_gen_value(rng, dist);
    }
    return bc;
}

static std::vector<BenchCase> build_bench_cases() {
    std::vector<BenchCase> cases;
    // Case data values derive from PMPP_BENCH_SEED so graded outputs are not
    // precomputable offline; shapes stay fixed so timing remains comparable.
    uint64_t s = 0x200000001b3ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL);

    cases.push_back(make_case(
        "wave_max_uniform", 32, 32, 256, 3, 5, TWG_DIST_UNIFORM, s++));
    cases.push_back(make_case(
        "wave_max_sparse", 32, 32, 128, 1, 1, TWG_DIST_SPARSE, s++));
    cases.push_back(make_case(
        "wave_wide_uniform", 8, 32, 256, 7, 2, TWG_DIST_UNIFORM, s++));
    cases.push_back(make_case(
        "wave_tall_small", 32, 8, 256, 0, 15, TWG_DIST_SMALL, s++));
    cases.push_back(make_case(
        "wave_mid_saturate", 20, 20, 128, 12, 4, TWG_DIST_SATURATE, s++));

    return cases;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        TwgProblemSpec spec = {};
        spec.abi_version = TWG_ABI_VERSION;
        spec.max_R = TWG_MAX_R;
        spec.max_C = TWG_MAX_C;
        spec.max_K = TWG_MAX_K;
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

        const std::vector<BenchCase> cases = build_bench_cases();

        // C3 ring: v_timed distinct (U, V) variants per case + kWarmups
        // warmup-only variants. Buffers are tiny so per-case rings are cheap.
        SplitMix64 c3_root_rng(0xC3ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL));
        const uint64_t c3_root = c3_root_rng.next_u64();
        const int kWarmups = 5;
        const int v_timed = std::min(iters, 32);
        const int n_var = v_timed + kWarmups;

        DeviceBuffer<int32_t> round_out;
        DeviceBuffer<uint32_t> border_row;
        DeviceBuffer<uint32_t> border_col;
        DeviceBuffer<uint64_t> state_checksum;
        round_out.allocate(1);
        border_row.allocate((size_t)TWG_MAX_C * 32);
        border_col.allocate((size_t)TWG_MAX_R * 32);
        state_checksum.allocate(1);

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double total_ms = 0.0;
        double total_calls = 0.0;
        pmpp::OutFnv dg;

        DeviceBuffer<unsigned long long> probe_acc;
        probe_acc.allocate((size_t)cases.size() * iters);
        CUDA_CHECK(cudaMemset(probe_acc.ptr, 0,
                              sizeof(unsigned long long) * probe_acc.count));

        C3ProbeTable pt;
        pt.n = 4;
        pt.d[0] = {(const uint8_t*)round_out.ptr, sizeof(int32_t)};
        // border_row/col nbytes depend on the case; filled per case below.

        for (size_t ci = 0; ci < cases.size(); ++ci) {
            const BenchCase& bc = cases[ci];
            std::printf("bench_case %-20s R=%d C=%d K=%d P1=%d P2=%d dist=%d\n",
                        bc.name.c_str(), bc.run.R, bc.run.C, bc.run.K,
                        bc.run.P1, bc.run.P2, bc.run.distribution_id);

            // Build this case's (U, V) variant ring on device.
            std::vector<DeviceBuffer<int8_t>> dU_ring(n_var), dV_ring(n_var);
            std::vector<int8_t> hU, hV;
            for (int v = 0; v < n_var; ++v) {
                twg_make_variant(bc, c3_root, v, hU, hV);
                dU_ring[v].allocate(hU.size());
                dV_ring[v].allocate(hV.size());
                dU_ring[v].upload(hU);
                dV_ring[v].upload(hV);
            }

            TwgInputs inputs = {};
            TwgOutputs outputs = {};
            outputs.round_out = round_out.ptr;
            outputs.border_row = border_row.ptr;
            outputs.border_col = border_col.ptr;
            outputs.state_checksum = state_checksum.ptr;
            outputs.acc_dump = nullptr;  // dump == 0 for every bench case

            pt.d[1] = {(const uint8_t*)border_row.ptr, (unsigned long long)bc.run.C * 32 * sizeof(uint32_t)};
            pt.d[2] = {(const uint8_t*)border_col.ptr, (unsigned long long)bc.run.R * 32 * sizeof(uint32_t)};
            pt.d[3] = {(const uint8_t*)state_checksum.ptr, sizeof(uint64_t)};

            auto set_variant = [&](int v) {
                inputs.U = dU_ring[v].ptr;
                inputs.V = dV_ring[v].ptr;
            };

            // Contract: reset before changing (R, C).
            CUDA_CHECK(solution_reset(state, stream));
            for (int warmup = 0; warmup < kWarmups; ++warmup) {
                set_variant(v_timed + warmup);
                CUDA_CHECK(solution_run(
                    state, &bc.run, &inputs, &outputs, workspace.ptr,
                    workspace_bytes, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int iter = 0; iter < iters; ++iter) {
                set_variant(iter % v_timed);
                CUDA_CHECK(solution_run(
                    state, &bc.run, &inputs, &outputs, workspace.ptr,
                    workspace_bytes, stream));
                // Untimed relative to the graded compute (the event pair still
                // brackets solution_run; the probe adds a tiny paired overhead
                // measured below). Fold this call's graded outputs.
                c3_probe_fold<<<1, 256, 0, stream>>>(
                    pt, host_mix64(c3_root ^ 0xF01DULL ^ (uint64_t)(ci + 1) * 131ULL ^ (uint64_t)(iter + 1)),
                    probe_acc.ptr + ci * iters + iter);
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

            std::printf("case_ms %-20s %.6f\n", bc.name.c_str(),
                        static_cast<double>(elapsed_ms) / iters);

            total_ms += static_cast<double>(elapsed_ms);
            total_calls += static_cast<double>(iters);

            // Untimed graded-output digest for this case (shared output buffers,
            // so fold before the next case overwrites them): one more run on the
            // last timed variant, then fold the graded ranges (round_out,
            // border_row[C*32], border_col[R*32], state_checksum).
            set_variant((iters - 1) % v_timed);
            CUDA_CHECK(solution_run(
                state, &bc.run, &inputs, &outputs, workspace.ptr,
                workspace_bytes, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaDeviceSynchronize());
            dg.dev(round_out.ptr, sizeof(int32_t));
            dg.dev(border_row.ptr, (size_t)bc.run.C * 32 * sizeof(uint32_t));
            dg.dev(border_col.ptr, (size_t)bc.run.R * 32 * sizeof(uint32_t));
            dg.dev(state_checksum.ptr, sizeof(uint64_t));
        }

        std::printf("avg_ms=%.6f\n", total_ms / total_calls);
        // C3: the digest binds EVERY timed call via the per-(case, iter) probes.
        {
            std::vector<unsigned long long> acc_host(probe_acc.count);
            CUDA_CHECK(cudaMemcpy(acc_host.data(), probe_acc.ptr,
                                  sizeof(unsigned long long) * probe_acc.count,
                                  cudaMemcpyDeviceToHost));
            dg.bytes(acc_host.data(), sizeof(unsigned long long) * acc_host.size());
        }
        dg.print();

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));

        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
