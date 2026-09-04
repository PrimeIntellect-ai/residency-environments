// ============================================================================
// file: bench_dualtier_credit_router.cu
// ============================================================================

#include "dualtier_credit_router_common.h"

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
};

struct BenchCase {
    std::string name;
    DtrRunSpec run;
    std::vector<int8_t> x;
    std::vector<int8_t> wnode;
    std::vector<int8_t> wexp;
    std::vector<int8_t> w1;
    std::vector<int8_t> w2;
};

static BenchCase make_case(
    const char* name,
    int N, int D, int H, int E, int P, int ccap, int bq, int refill,
    int qshift, int dist, uint64_t seed) {
    BenchCase bc;
    bc.name = name;

    bc.run = {};
    bc.run.abi_version = DTR_ABI_VERSION;
    bc.run.N = N;
    bc.run.D = D;
    bc.run.H = H;
    bc.run.E = E;
    bc.run.P = P;
    bc.run.ccap = ccap;
    bc.run.bq = bq;
    bc.run.refill = refill;
    bc.run.qshift = qshift;
    bc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    bc.run.distribution_id = dist;
    bc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);
    if (!dtr_validate_run_spec(&bc.run)) {
        throw std::runtime_error(std::string("invalid run spec: ") + name);
    }

    SplitMix64 rng(g_state ^ seed);
    bc.x.resize((size_t)N * D);
    bc.wnode.resize((size_t)P * D);
    bc.wexp.resize((size_t)E * D);
    bc.w1.resize((size_t)E * H * D);
    bc.w2.resize((size_t)E * D * H);

    for (size_t i = 0; i < bc.x.size(); ++i) {
        bc.x[i] = dist == DTR_DIST_HOT_NODE || dist == DTR_DIST_BURSTY
            ? (int8_t)rng.uniform_int(0, 127)
            : (dist == DTR_DIST_TIES ? (int8_t)rng.uniform_int(-1, 1)
                                     : (int8_t)rng.uniform_int(-127, 127));
    }
    for (int p = 0; p < P; ++p) {
        for (int d = 0; d < D; ++d) {
            int8_t v;
            if (dist == DTR_DIST_HOT_NODE) {
                v = p < 2 ? (int8_t)rng.uniform_int(64, 127)
                          : (int8_t)rng.uniform_int(-32, 32);
            } else if (dist == DTR_DIST_TIES) {
                v = (int8_t)rng.uniform_int(-1, 1);
            } else {
                v = (int8_t)rng.uniform_int(-127, 127);
            }
            bc.wnode[(size_t)p * D + d] = v;
        }
    }
    for (size_t i = 0; i < bc.wexp.size(); ++i) {
        bc.wexp[i] = dist == DTR_DIST_TIES
            ? (int8_t)rng.uniform_int(-1, 1)
            : (int8_t)rng.uniform_int(-127, 127);
    }
    for (size_t i = 0; i < bc.w1.size(); ++i) {
        bc.w1[i] = (int8_t)rng.uniform_int(-127, 127);
    }
    for (size_t i = 0; i < bc.w2.size(); ++i) {
        bc.w2[i] = (int8_t)rng.uniform_int(-127, 127);
    }
    return bc;
}

static std::vector<BenchCase> build_bench_cases() {
    std::vector<BenchCase> cases;
    uint64_t s = 0x400000001b3ULL;

    cases.push_back(make_case(
        "router_max_uniform", 16384, 128, 256, 128, 8, 64, 128, 64, 6,
        DTR_DIST_UNIFORM, s++));
    cases.push_back(make_case(
        "router_max_hot", 16384, 128, 256, 128, 8, 64, 128, 32, 5,
        DTR_DIST_HOT_NODE, s++));
    cases.push_back(make_case(
        "router_mid_ties", 8192, 128, 256, 64, 8, 32, 64, 16, 6,
        DTR_DIST_TIES, s++));
    cases.push_back(make_case(
        "router_wide_bursty", 16384, 64, 128, 128, 16, 64, 128, 64, 7,
        DTR_DIST_BURSTY, s++));
    cases.push_back(make_case(
        "router_small_churn", 4096, 128, 256, 32, 4, 64, 128, 48, 4,
        DTR_DIST_UNIFORM, s++));

    return cases;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        DtrProblemSpec spec = {};
        spec.abi_version = DTR_ABI_VERSION;
        spec.max_N = DTR_MAX_N;
        spec.max_D = DTR_MAX_D;
        spec.max_H = DTR_MAX_H;
        spec.max_E = DTR_MAX_E;
        spec.max_P = DTR_MAX_P;
        spec.max_ccap = DTR_MAX_CCAP;
        spec.max_bq = DTR_MAX_BQ;
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

        DeviceBuffer<int8_t> dx, dwnode, dwexp, dw1, dw2;
        dx.allocate((size_t)DTR_MAX_N * DTR_MAX_D);
        dwnode.allocate((size_t)DTR_MAX_P * DTR_MAX_D);
        dwexp.allocate((size_t)DTR_MAX_E * DTR_MAX_D);
        dw1.allocate((size_t)DTR_MAX_E * DTR_MAX_H * DTR_MAX_D);
        dw2.allocate((size_t)DTR_MAX_E * DTR_MAX_D * DTR_MAX_H);

        const size_t cap = (size_t)DTR_MAX_E * DTR_MAX_CCAP;
        DeviceBuffer<int32_t> s2_logits, route_nodes, route_pe_be, log_len;
        DeviceBuffer<uint64_t> event_log;
        DeviceBuffer<int32_t> counts, offsets, packed_gid, packed_out;
        DeviceBuffer<uint32_t> credit_out;
        DeviceBuffer<uint64_t> state_checksum;
        s2_logits.allocate((size_t)DTR_MAX_N * DTR_MAX_E);
        route_nodes.allocate((size_t)DTR_MAX_N);
        route_pe_be.allocate((size_t)DTR_MAX_N);
        log_len.allocate(1);
        event_log.allocate((size_t)DTR_MAX_N + cap);
        counts.allocate((size_t)DTR_MAX_E);
        offsets.allocate((size_t)DTR_MAX_E + 1);
        packed_gid.allocate(cap);
        packed_out.allocate(cap * (size_t)DTR_MAX_D);
        credit_out.allocate((size_t)DTR_MAX_E);
        state_checksum.allocate(1);

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double total_ms = 0.0;
        double total_calls = 0.0;

        for (const BenchCase& bc : cases) {
            std::printf(
                "bench_case %-20s N=%d D=%d H=%d E=%d P=%d cc=%d bq=%d "
                "refill=%d dist=%d\n",
                bc.name.c_str(), bc.run.N, bc.run.D, bc.run.H, bc.run.E,
                bc.run.P, bc.run.ccap, bc.run.bq, bc.run.refill,
                bc.run.distribution_id);

            CUDA_CHECK(cudaMemcpy(dx.ptr, bc.x.data(), bc.x.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dwnode.ptr, bc.wnode.data(),
                                  bc.wnode.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dwexp.ptr, bc.wexp.data(), bc.wexp.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dw1.ptr, bc.w1.data(), bc.w1.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dw2.ptr, bc.w2.data(), bc.w2.size(),
                                  cudaMemcpyHostToDevice));

            DtrInputs inputs = {};
            inputs.x = dx.ptr;
            inputs.wnode = dwnode.ptr;
            inputs.wexp = dwexp.ptr;
            inputs.w1 = dw1.ptr;
            inputs.w2 = dw2.ptr;

            DtrOutputs outputs = {};
            outputs.s2_logits = s2_logits.ptr;
            outputs.route_nodes = route_nodes.ptr;
            outputs.route_pe_be = route_pe_be.ptr;
            outputs.log_len = log_len.ptr;
            outputs.event_log = event_log.ptr;
            outputs.counts = counts.ptr;
            outputs.offsets = offsets.ptr;
            outputs.packed_gid = packed_gid.ptr;
            outputs.packed_out = packed_out.ptr;
            outputs.credit_out = credit_out.ptr;
            outputs.state_checksum = state_checksum.ptr;

            // Contract: reset before changing the reset-scoped shape.
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
