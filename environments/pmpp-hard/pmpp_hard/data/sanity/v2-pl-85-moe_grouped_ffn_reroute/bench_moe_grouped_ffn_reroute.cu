// ============================================================================
// file: bench_moe_grouped_ffn_reroute.cu
// ============================================================================

#include "moe_grouped_ffn_reroute_common.h"

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

struct HostCase {
    std::string name;
    MgfRunSpec run;
    std::vector<int8_t> x;
    std::vector<int8_t> wr;
    std::vector<int8_t> w1;
    std::vector<int8_t> w2;
};

static int8_t gen_x_value(SplitMix64& rng, int dist) {
    switch (dist) {
        case MGF_DIST_HOT_EXPERT:
            return static_cast<int8_t>(rng.uniform_int(0, 127));
        case MGF_DIST_TIES: {
            const double u = rng.uniform01();
            return static_cast<int8_t>(u < 0.5 ? 0 : (u < 0.75 ? 1 : 2));
        }
        case MGF_DIST_SATURATE:
            return static_cast<int8_t>(rng.uniform_int(64, 127));
        default:
            return static_cast<int8_t>(rng.uniform_int(-127, 127));
    }
}

static int8_t gen_wr_value(SplitMix64& rng, int dist, int e) {
    switch (dist) {
        case MGF_DIST_HOT_EXPERT:
            if (e < 2) return static_cast<int8_t>(rng.uniform_int(64, 127));
            return static_cast<int8_t>(rng.uniform_int(-32, 32));
        case MGF_DIST_TIES:
            return static_cast<int8_t>(rng.uniform_int(-1, 1));
        case MGF_DIST_ZERO_X:
            return static_cast<int8_t>(rng.uniform_int(-8, 8));
        default:
            return static_cast<int8_t>(rng.uniform_int(-127, 127));
    }
}

static int8_t gen_w1_value(SplitMix64& rng, int dist) {
    switch (dist) {
        case MGF_DIST_TIES:
            return static_cast<int8_t>(rng.uniform_int(-3, 3));
        case MGF_DIST_SATURATE:
            return static_cast<int8_t>(rng.uniform_int(32, 127));
        default:
            return static_cast<int8_t>(rng.uniform_int(-127, 127));
    }
}

static int8_t gen_w2_value(SplitMix64& rng, int dist) {
    switch (dist) {
        case MGF_DIST_TIES:
            return static_cast<int8_t>(rng.uniform_int(-3, 3));
        default:
            return static_cast<int8_t>(rng.uniform_int(-127, 127));
    }
}

static HostCase make_random_case(
    const char* name,
    int N, int D, int H, int E, int G, int g_sel, int K, int cap, int qshift,
    int dist,
    uint64_t case_seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = MGF_ABI_VERSION;
    hc.run.N = N;
    hc.run.D = D;
    hc.run.H = H;
    hc.run.E = E;
    hc.run.G = G;
    hc.run.g_sel = g_sel;
    hc.run.K = K;
    hc.run.cap = cap;
    hc.run.qshift = qshift;
    hc.run.seed_id = static_cast<int32_t>(case_seed & 0x7fffffff);
    hc.run.distribution_id = dist;
    hc.run.case_id = static_cast<int32_t>((case_seed >> 32) & 0x7fffffff);

    if (!mgf_validate_run_spec(&hc.run)) {
        throw std::runtime_error(std::string("invalid run spec: ") + name);
    }

    SplitMix64 rng(g_state ^ case_seed);

    hc.x.resize((size_t)N * D);
    hc.wr.resize((size_t)E * D);
    hc.w1.resize((size_t)E * H * D);
    hc.w2.resize((size_t)E * D * H);

    for (int t = 0; t < N; ++t) {
        const bool zero_row =
            dist == MGF_DIST_ZERO_X && rng.chance(0.70);
        for (int d = 0; d < D; ++d) {
            hc.x[(size_t)t * D + d] =
                zero_row ? 0 : gen_x_value(rng, dist);
        }
    }
    for (int e = 0; e < E; ++e) {
        for (int d = 0; d < D; ++d) {
            hc.wr[(size_t)e * D + d] = gen_wr_value(rng, dist, e);
        }
    }
    for (size_t i = 0; i < hc.w1.size(); ++i) {
        hc.w1[i] = gen_w1_value(rng, dist);
    }
    for (size_t i = 0; i < hc.w2.size(); ++i) {
        hc.w2[i] = gen_w2_value(rng, dist);
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x200000001b3ULL;

    cases.push_back(make_random_case(
        "uniform_max", 32768, 128, 256, 128, 8, 2, 4, 256, 6,
        MGF_DIST_UNIFORM, s++));

    cases.push_back(make_random_case(
        "hot_reroute_tinycap", 32768, 128, 256, 128, 16, 4, 4, 32, 5,
        MGF_DIST_HOT_EXPERT, s++));

    cases.push_back(make_random_case(
        "ties_wide", 32768, 64, 256, 128, 16, 4, 4, 64, 6,
        MGF_DIST_TIES, s++));

    cases.push_back(make_random_case(
        "saturate_mid", 16384, 128, 128, 64, 8, 2, 4, 256, 4,
        MGF_DIST_SATURATE, s++));

    cases.push_back(make_random_case(
        "uniform_smallE_routing", 32768, 32, 64, 16, 4, 2, 2, 256, 6,
        MGF_DIST_UNIFORM, s++));

    cases.push_back(make_random_case(
        "zero_x_tie_cascade", 32768, 128, 256, 32, 4, 1, 4, 128, 7,
        MGF_DIST_ZERO_X, s++));

    return cases;
}

struct DeviceCase {
    HostCase host;

    DeviceBuffer<int8_t> x, wr, w1, w2;
    DeviceBuffer<int32_t> logits;
    DeviceBuffer<int32_t> counts;
    DeviceBuffer<int32_t> offsets;
    DeviceBuffer<int32_t> packed_token;
    DeviceBuffer<int32_t> packed_slot;
    DeviceBuffer<int32_t> packed_gate;
    DeviceBuffer<uint8_t> packed_phase;
    DeviceBuffer<int16_t> route_expert;
    DeviceBuffer<uint8_t> route_status;
    DeviceBuffer<int32_t> packed_y;
    DeviceBuffer<int64_t> y;
    DeviceBuffer<uint64_t> y_checksum;

    MgfInputs inputs;
    MgfOutputs outputs;
};

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int D = hc.run.D;
    const int E = hc.run.E;
    const int K = hc.run.K;
    const size_t packed_capacity = (size_t)E * hc.run.cap;

    dc->x.allocate(hc.x.size());
    dc->wr.allocate(hc.wr.size());
    dc->w1.allocate(hc.w1.size());
    dc->w2.allocate(hc.w2.size());
    dc->x.upload(hc.x);
    dc->wr.upload(hc.wr);
    dc->w1.upload(hc.w1);
    dc->w2.upload(hc.w2);

    dc->logits.allocate((size_t)N * E);
    dc->counts.allocate((size_t)E);
    dc->offsets.allocate((size_t)E + 1);
    dc->packed_token.allocate(packed_capacity);
    dc->packed_slot.allocate(packed_capacity);
    dc->packed_gate.allocate(packed_capacity);
    dc->packed_phase.allocate(packed_capacity);
    dc->route_expert.allocate((size_t)N * K);
    dc->route_status.allocate((size_t)N * K);
    dc->packed_y.allocate(packed_capacity * (size_t)D);
    dc->y.allocate((size_t)N * D);
    dc->y_checksum.allocate(1);

    dc->inputs = {};
    dc->inputs.x = dc->x.ptr;
    dc->inputs.wr = dc->wr.ptr;
    dc->inputs.w1 = dc->w1.ptr;
    dc->inputs.w2 = dc->w2.ptr;

    dc->outputs = {};
    dc->outputs.logits = dc->logits.ptr;
    dc->outputs.counts = dc->counts.ptr;
    dc->outputs.offsets = dc->offsets.ptr;
    dc->outputs.packed_token = dc->packed_token.ptr;
    dc->outputs.packed_slot = dc->packed_slot.ptr;
    dc->outputs.packed_gate = dc->packed_gate.ptr;
    dc->outputs.packed_phase = dc->packed_phase.ptr;
    dc->outputs.route_expert = dc->route_expert.ptr;
    dc->outputs.route_status = dc->route_status.ptr;
    dc->outputs.packed_y = dc->packed_y.ptr;
    dc->outputs.y = dc->y.ptr;
    dc->outputs.y_checksum = dc->y_checksum.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        MgfProblemSpec spec = {};
        spec.abi_version = MGF_ABI_VERSION;
        spec.max_N = MGF_MAX_N;
        spec.max_D = MGF_MAX_D;
        spec.max_H = MGF_MAX_H;
        spec.max_E = MGF_MAX_E;
        spec.max_K = MGF_MAX_K;
        spec.max_cap = MGF_MAX_CAP;
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

        std::vector<HostCase> host_cases = build_bench_cases();
        std::vector<DeviceCase*> cases;
        cases.reserve(host_cases.size());

        for (const HostCase& hc : host_cases) {
            cases.push_back(make_device_case(hc));
            std::printf(
                "bench_case %-24s N=%d D=%d H=%d E=%d G=%d g=%d K=%d cap=%d "
                "dist=%d\n",
                hc.name.c_str(), hc.run.N, hc.run.D, hc.run.H, hc.run.E,
                hc.run.G, hc.run.g_sel, hc.run.K, hc.run.cap,
                hc.run.distribution_id);
        }

        for (int warmup = 0; warmup < 5; ++warmup) {
            for (DeviceCase* dc : cases) {
                CUDA_CHECK(solution_run(
                    state, &dc->host.run, &dc->inputs, &dc->outputs,
                    workspace.ptr, workspace_bytes, stream));
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
                CUDA_CHECK(solution_run(
                    state, &dc->host.run, &dc->inputs, &dc->outputs,
                    workspace.ptr, workspace_bytes, stream));
            }
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        const double calls =
            static_cast<double>(iters) * static_cast<double>(cases.size());
        const double avg_ms = static_cast<double>(elapsed_ms) / calls;

        std::printf("avg_ms=%.6f\n", avg_ms);

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
