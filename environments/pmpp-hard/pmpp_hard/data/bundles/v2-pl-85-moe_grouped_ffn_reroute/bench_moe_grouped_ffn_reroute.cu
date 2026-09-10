// ============================================================================
// file: bench_moe_grouped_ffn_reroute.cu
// ============================================================================

#include "moe_grouped_ffn_reroute_common.h"
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

struct HostCase {
    std::string name;
    MgfRunSpec run;
    std::vector<int8_t> x;
    std::vector<int8_t> wr;
    std::vector<int8_t> w1;
    std::vector<int8_t> w2;
};

// --- C3 timed-fold: every timed call sees fresh input data and contributes its
// graded outputs to out_fnv, so a cached/no-op timed call cannot match the
// reference digest. Shapes and the zero-row sparsity pattern stay fixed; only
// data values vary per iteration (timing comparability preserved).

__device__ __forceinline__ uint64_t c3_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// In-place regeneration of x for timed iteration k, staying inside the case's
// distribution family (ZERO_X keeps its zero rows zero so routing load stays
// in-family across iterations).
__global__ void c3_mutate_x(int8_t* x, size_t n, uint64_t stream_key, int dist) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const uint64_t h = c3_mix(stream_key ^ (i + 1) * 0x9e3779b97f4a7c15ULL);
        int v;
        switch (dist) {
            case MGF_DIST_HOT_EXPERT:
                v = (int)(h % 128);
                break;
            case MGF_DIST_TIES: {
                const uint32_t u = (uint32_t)(h & 3);
                v = (u < 2) ? 0 : (u == 2 ? 1 : 2);
                break;
            }
            case MGF_DIST_SATURATE:
                v = 64 + (int)(h % 64);
                break;
            case MGF_DIST_ZERO_X: {
                if (x[i] == 0) {
                    v = 0;
                } else {
                    const int t = (int)(h % 254) - 127;
                    v = (t >= 0) ? t + 1 : t;
                }
                break;
            }
            default:
                v = (int)(h % 255) - 127;
                break;
        }
        x[i] = (int8_t)v;
    }
}

struct C3ProbeDesc {
    const uint8_t* ptr;
    unsigned long long fixed_bytes;   // used when len_ptr == nullptr
    const int32_t* len_ptr;           // device count (e.g. offsets[E])
    unsigned long long elem_stride;   // bytes per counted element
};

struct C3ProbeTable {
    C3ProbeDesc d[12];
    int n;
};

// One launch per timed call: xor-folds a position-salted mix of every graded
// output word into acc[slot]. xor accumulation is order-independent, so the
// result is deterministic; honest student and reference produce identical
// probes on the same seed.
__global__ void c3_probe_fold(C3ProbeTable t, uint64_t salt,
                              unsigned long long* acc) {
    __shared__ unsigned long long sh[256];
    unsigned long long local = 0;
    const unsigned long long gstride =
        (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long gtid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < t.n; ++b) {
        const C3ProbeDesc& d = t.d[b];
        const unsigned long long nbytes =
            d.len_ptr ? (unsigned long long)(*d.len_ptr) * d.elem_stride
                      : d.fixed_bytes;
        const unsigned long long nwords = nbytes >> 3;
        const uint64_t bsalt = salt ^ (uint64_t)(b + 1) * 0x100000001b3ULL;
        const unsigned long long* w64 =
            reinterpret_cast<const unsigned long long*>(d.ptr);
        for (unsigned long long w = gtid; w < nwords; w += gstride) {
            local ^= c3_mix(w64[w] ^ c3_mix(bsalt + w));
        }
        if (gtid == 0 && (nbytes & 7ULL)) {
            unsigned long long tail = 0;
            for (unsigned long long i = nwords << 3; i < nbytes; ++i) {
                tail |= (unsigned long long)d.ptr[i] << (8 * (i & 7ULL));
            }
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
    // Case data values derive from PMPP_BENCH_SEED so graded outputs are not
    // precomputable offline; shapes stay fixed so timing remains comparable.
    uint64_t s = 0x200000001b3ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL);

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
    C3ProbeTable probe;
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

    // C3 probe table: exactly the graded output set (mirrors the digest fold
    // below; packed_* arrays are graded only up to offsets[E] entries).
    const int32_t* d_total = dc->offsets.ptr + E;
    C3ProbeTable& pt = dc->probe;
    pt.n = 12;
    pt.d[0] = {(const uint8_t*)dc->logits.ptr,
               (unsigned long long)N * E * sizeof(int32_t), nullptr, 0};
    pt.d[1] = {(const uint8_t*)dc->counts.ptr,
               (unsigned long long)E * sizeof(int32_t), nullptr, 0};
    pt.d[2] = {(const uint8_t*)dc->offsets.ptr,
               (unsigned long long)(E + 1) * sizeof(int32_t), nullptr, 0};
    pt.d[3] = {(const uint8_t*)dc->packed_token.ptr, 0, d_total,
               sizeof(int32_t)};
    pt.d[4] = {(const uint8_t*)dc->packed_slot.ptr, 0, d_total,
               sizeof(int32_t)};
    pt.d[5] = {(const uint8_t*)dc->packed_gate.ptr, 0, d_total,
               sizeof(int32_t)};
    pt.d[6] = {(const uint8_t*)dc->packed_phase.ptr, 0, d_total,
               sizeof(uint8_t)};
    pt.d[7] = {(const uint8_t*)dc->route_expert.ptr,
               (unsigned long long)N * K * sizeof(int16_t), nullptr, 0};
    pt.d[8] = {(const uint8_t*)dc->route_status.ptr,
               (unsigned long long)N * K * sizeof(uint8_t), nullptr, 0};
    pt.d[9] = {(const uint8_t*)dc->packed_y.ptr, 0, d_total,
               (unsigned long long)D * sizeof(int32_t)};
    pt.d[10] = {(const uint8_t*)dc->y.ptr,
                (unsigned long long)N * D * sizeof(int64_t), nullptr, 0};
    pt.d[11] = {(const uint8_t*)dc->y_checksum.ptr, sizeof(uint64_t), nullptr,
                0};

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

        // C3: per-(iter, case) probe accumulators + mutation stream root.
        DeviceBuffer<unsigned long long> probe_acc;
        probe_acc.allocate((size_t)iters * cases.size());
        CUDA_CHECK(cudaMemset(probe_acc.ptr, 0,
                              sizeof(unsigned long long) * probe_acc.count));
        SplitMix64 c3_root_rng(0xC3ULL ^
                               (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL));
        const uint64_t c3_root = c3_root_rng.next_u64();

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int iter = 0; iter < iters; ++iter) {
            for (size_t ci = 0; ci < cases.size(); ++ci) {
                DeviceCase* dc = cases[ci];
                SplitMix64 krng(c3_root ^ ((uint64_t)(iter + 1) << 32) ^
                                (uint64_t)(ci + 1));
                const uint64_t skey = krng.next_u64();
                const size_t xn = dc->host.x.size();
                const int mut_blocks =
                    (int)std::min<size_t>((xn + 255) / 256, 4096);
                c3_mutate_x<<<mut_blocks, 256, 0, stream>>>(
                    dc->x.ptr, xn, skey, dc->host.run.distribution_id);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(solution_run(
                    state, &dc->host.run, &dc->inputs, &dc->outputs,
                    workspace.ptr, workspace_bytes, stream));
                c3_probe_fold<<<512, 256, 0, stream>>>(
                    dc->probe, skey ^ 0xF01DF01DF01DF01DULL,
                    probe_acc.ptr + (size_t)iter * cases.size() + ci);
                CUDA_CHECK(cudaGetLastError());
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

        // Untimed graded-output digest: one more run per case, then fold the graded
        // output ranges in fixed case order. The packed_* arrays and packed_y are
        // only graded up to offsets[E] entries (tail unspecified), so fold that prefix.
        for (DeviceCase* dc : cases) {
            CUDA_CHECK(solution_run(
                state, &dc->host.run, &dc->inputs, &dc->outputs,
                workspace.ptr, workspace_bytes, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        // C3: the digest binds EVERY timed call's graded outputs via the
        // per-(iter, case) probe accumulators; a stale/no-op timed call leaves
        // a wrong probe word and the digest cannot match the reference.
        {
            std::vector<unsigned long long> acc_host(probe_acc.count);
            CUDA_CHECK(cudaMemcpy(acc_host.data(), probe_acc.ptr,
                                  sizeof(unsigned long long) * probe_acc.count,
                                  cudaMemcpyDeviceToHost));
            dg.bytes(acc_host.data(),
                     sizeof(unsigned long long) * acc_host.size());
        }
        for (DeviceCase* dc : cases) {
            const int N = dc->host.run.N;
            const int D = dc->host.run.D;
            const int E = dc->host.run.E;
            const int K = dc->host.run.K;
            int32_t total_packed = 0;
            CUDA_CHECK(cudaMemcpy(&total_packed, dc->offsets.ptr + E, sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            dg.dev(dc->logits.ptr, (size_t)N * (size_t)E * sizeof(int32_t));
            dg.dev(dc->counts.ptr, (size_t)E * sizeof(int32_t));
            dg.dev(dc->offsets.ptr, (size_t)(E + 1) * sizeof(int32_t));
            if (total_packed > 0) {
                dg.dev(dc->packed_token.ptr, (size_t)total_packed * sizeof(int32_t));
                dg.dev(dc->packed_slot.ptr, (size_t)total_packed * sizeof(int32_t));
                dg.dev(dc->packed_gate.ptr, (size_t)total_packed * sizeof(int32_t));
                dg.dev(dc->packed_phase.ptr, (size_t)total_packed * sizeof(uint8_t));
                dg.dev(dc->packed_y.ptr, (size_t)total_packed * (size_t)D * sizeof(int32_t));
            }
            dg.dev(dc->route_expert.ptr, (size_t)N * (size_t)K * sizeof(int16_t));
            dg.dev(dc->route_status.ptr, (size_t)N * (size_t)K * sizeof(uint8_t));
            dg.dev(dc->y.ptr, (size_t)N * (size_t)D * sizeof(int64_t));
            dg.dev(dc->y_checksum.ptr, sizeof(uint64_t));
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
