// ============================================================================
// file: bench_taskgraph_wavefront_gemm.cu
// ============================================================================

#include "taskgraph_wavefront_gemm_common.h"

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

static constexpr uint64_t g_state = 0x8e3779b97f4a7c15ULL;

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
    uint64_t s = 0x200000001b3ULL;

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

        DeviceBuffer<int8_t> dU;
        DeviceBuffer<int8_t> dV;
        dU.allocate((size_t)TWG_MAX_R * 32 * TWG_MAX_K);
        dV.allocate((size_t)TWG_MAX_C * 32 * TWG_MAX_K);

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

        for (const BenchCase& bc : cases) {
            std::printf("bench_case %-20s R=%d C=%d K=%d P1=%d P2=%d dist=%d\n",
                        bc.name.c_str(), bc.run.R, bc.run.C, bc.run.K,
                        bc.run.P1, bc.run.P2, bc.run.distribution_id);

            CUDA_CHECK(cudaMemcpy(dU.ptr, bc.U.data(), bc.U.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dV.ptr, bc.V.data(), bc.V.size(),
                                  cudaMemcpyHostToDevice));

            TwgInputs inputs = {};
            inputs.U = dU.ptr;
            inputs.V = dV.ptr;
            TwgOutputs outputs = {};
            outputs.round_out = round_out.ptr;
            outputs.border_row = border_row.ptr;
            outputs.border_col = border_col.ptr;
            outputs.state_checksum = state_checksum.ptr;
            outputs.acc_dump = nullptr;  // dump == 0 for every bench case

            // Contract: reset before changing (R, C).
            CUDA_CHECK(solution_reset(state, stream));
            for (int warmup = 0; warmup < 5; ++warmup) {
                CUDA_CHECK(solution_run(
                    state, &bc.run, &inputs, &outputs, workspace.ptr,
                    workspace_bytes, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int iter = 0; iter < iters; ++iter) {
                CUDA_CHECK(solution_run(
                    state, &bc.run, &inputs, &outputs, workspace.ptr,
                    workspace_bytes, stream));
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

            std::printf("case_ms %-20s %.6f\n", bc.name.c_str(),
                        static_cast<double>(elapsed_ms) / iters);

            total_ms += static_cast<double>(elapsed_ms);
            total_calls += static_cast<double>(iters);
        }

        std::printf("avg_ms=%.6f\n", total_ms / total_calls);

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
