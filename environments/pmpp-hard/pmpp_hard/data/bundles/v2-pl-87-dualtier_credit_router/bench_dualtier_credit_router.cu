// ============================================================================
// file: bench_dualtier_credit_router.cu
// ============================================================================

#include "dualtier_credit_router_common.h"
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

// --- C3 timed-fold (patterns I2 + P2): the engine is stateful (persistent
// per-expert credits + bounded backlog + running token ids), so it is already
// immune to compute-once-cache, but C3 makes each timed call see FRESH token
// features x (regenerated in place from PMPP_BENCH_SEED, case, iter,
// distribution-preserving; the weights stay fixed = in-family) and binds every
// timed call's graded outputs via a probe folded AFTER each call. A stale/no-op
// timed call then leaves wrong outputs for its iteration and breaks the digest.

__device__ __forceinline__ uint64_t c3_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static uint64_t c3_hmix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__global__ void c3_mutate_x(int8_t* x, long long n, uint64_t stream_key, int dist) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const uint64_t h = c3_mix(stream_key ^ (uint64_t)(i + 1) * 0x9e3779b97f4a7c15ULL);
        int v;
        if (dist == DTR_DIST_HOT_NODE || dist == DTR_DIST_BURSTY)
            v = (int)(h % 128);
        else if (dist == DTR_DIST_TIES)
            v = (int)(h % 3) - 1;
        else
            v = (int)(h % 255) - 127;
        x[i] = (int8_t)v;
    }
}

struct C3ProbeDesc {
    const uint8_t* ptr;
    unsigned long long fixed_bytes;   // used when len_ptr == nullptr
    const int32_t* len_ptr;           // device count
    unsigned long long elem_stride;   // bytes per counted element
};
struct C3ProbeTable { C3ProbeDesc d[12]; int n; };

__global__ void c3_probe_fold(C3ProbeTable t, uint64_t salt,
                              unsigned long long* acc) {
    __shared__ unsigned long long sh[256];
    unsigned long long local = 0;
    const unsigned long long gstride = (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long gtid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < t.n; ++b) {
        const C3ProbeDesc& d = t.d[b];
        const unsigned long long nbytes =
            d.len_ptr ? (unsigned long long)(*d.len_ptr) * d.elem_stride : d.fixed_bytes;
        const unsigned long long nwords = nbytes >> 3;
        const uint64_t bsalt = salt ^ (uint64_t)(b + 1) * 0x100000001b3ULL;
        const unsigned long long* w64 = reinterpret_cast<const unsigned long long*>(d.ptr);
        for (unsigned long long w = gtid; w < nwords; w += gstride)
            local ^= c3_mix(w64[w] ^ c3_mix(bsalt + w));
        if (gtid == 0 && (nbytes & 7ULL)) {
            unsigned long long tail = 0;
            for (unsigned long long i = nwords << 3; i < nbytes; ++i)
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
    // Case data values derive from PMPP_BENCH_SEED so graded outputs are not
    // precomputable offline; shapes stay fixed so timing remains comparable.
    uint64_t s = 0x400000001b3ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL);

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
        pmpp::OutFnv dg;

        SplitMix64 c3_root_rng(0xC3ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL));
        const uint64_t c3_root = c3_root_rng.next_u64();
        DeviceBuffer<unsigned long long> probe_acc;
        probe_acc.allocate((size_t)cases.size() * iters);
        CUDA_CHECK(cudaMemset(probe_acc.ptr, 0,
                              sizeof(unsigned long long) * probe_acc.count));

        for (size_t ci = 0; ci < cases.size(); ++ci) {
            const BenchCase& bc = cases[ci];
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

            const int Nb = bc.run.N, Db = bc.run.D, Eb = bc.run.E;
            const long long xn = (long long)Nb * Db;
            const int mut_blocks = (int)std::min<long long>((xn + 255) / 256, 4096);
            const int32_t* d_total = offsets.ptr + Eb;
            // Probe table: exactly the graded output set (dynamic lengths read
            // on device: event_log up to log_len, packed_* up to offsets[E]).
            C3ProbeTable pt; pt.n = 11;
            pt.d[0] = {(const uint8_t*)s2_logits.ptr, (unsigned long long)Nb * Eb * sizeof(int32_t), nullptr, 0};
            pt.d[1] = {(const uint8_t*)route_nodes.ptr, (unsigned long long)Nb * sizeof(int32_t), nullptr, 0};
            pt.d[2] = {(const uint8_t*)route_pe_be.ptr, (unsigned long long)Nb * sizeof(int32_t), nullptr, 0};
            pt.d[3] = {(const uint8_t*)log_len.ptr, sizeof(int32_t), nullptr, 0};
            pt.d[4] = {(const uint8_t*)event_log.ptr, 0, log_len.ptr, sizeof(uint64_t)};
            pt.d[5] = {(const uint8_t*)counts.ptr, (unsigned long long)Eb * sizeof(int32_t), nullptr, 0};
            pt.d[6] = {(const uint8_t*)offsets.ptr, (unsigned long long)(Eb + 1) * sizeof(int32_t), nullptr, 0};
            pt.d[7] = {(const uint8_t*)packed_gid.ptr, 0, d_total, sizeof(int32_t)};
            pt.d[8] = {(const uint8_t*)packed_out.ptr, 0, d_total, (unsigned long long)Db * sizeof(int32_t)};
            pt.d[9] = {(const uint8_t*)credit_out.ptr, (unsigned long long)Eb * sizeof(uint32_t), nullptr, 0};
            pt.d[10] = {(const uint8_t*)state_checksum.ptr, sizeof(uint64_t), nullptr, 0};

            auto mutate = [&](uint64_t tag) {
                const uint64_t skey = c3_hmix(c3_root ^ ((uint64_t)(ci + 1) << 40) ^ tag);
                c3_mutate_x<<<mut_blocks, 256, 0, stream>>>(dx.ptr, xn, skey, bc.run.distribution_id);
            };

            // Contract: reset before changing the reset-scoped shape.
            CUDA_CHECK(solution_reset(state, stream));
            for (int warmup = 0; warmup < 5; ++warmup) {
                mutate(0x40000000ULL + warmup);
                CUDA_CHECK(solution_run(
                    state, &bc.run, &inputs, &outputs, workspace.ptr,
                    workspace_bytes, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int iter = 0; iter < iters; ++iter) {
                mutate((uint64_t)(iter + 1));
                CUDA_CHECK(solution_run(
                    state, &bc.run, &inputs, &outputs, workspace.ptr,
                    workspace_bytes, stream));
                c3_probe_fold<<<256, 256, 0, stream>>>(
                    pt, c3_hmix(c3_root ^ 0xF01DULL ^ (uint64_t)(ci + 1) * 131ULL ^ (uint64_t)(iter + 1)),
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
            // so fold before the next case overwrites them): one more run on a
            // fresh (deterministic) input, then fold the graded ranges.
            // event_log is only graded up to log_len and packed_gid/packed_out
            // up to offsets[E] (tails unspecified).
            mutate((uint64_t)(iters + 1));
            CUDA_CHECK(solution_run(
                state, &bc.run, &inputs, &outputs, workspace.ptr,
                workspace_bytes, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaDeviceSynchronize());
            const int N = bc.run.N;
            const int D = bc.run.D;
            const int E = bc.run.E;
            int32_t h_log_len = 0;
            int32_t total_packed = 0;
            CUDA_CHECK(cudaMemcpy(&h_log_len, log_len.ptr, sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&total_packed, offsets.ptr + E, sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            dg.dev(s2_logits.ptr, (size_t)N * (size_t)E * sizeof(int32_t));
            dg.dev(route_nodes.ptr, (size_t)N * sizeof(int32_t));
            dg.dev(route_pe_be.ptr, (size_t)N * sizeof(int32_t));
            dg.dev(log_len.ptr, sizeof(int32_t));
            if (h_log_len > 0)
                dg.dev(event_log.ptr, (size_t)h_log_len * sizeof(uint64_t));
            dg.dev(counts.ptr, (size_t)E * sizeof(int32_t));
            dg.dev(offsets.ptr, (size_t)(E + 1) * sizeof(int32_t));
            if (total_packed > 0) {
                dg.dev(packed_gid.ptr, (size_t)total_packed * sizeof(int32_t));
                dg.dev(packed_out.ptr, (size_t)total_packed * (size_t)D * sizeof(int32_t));
            }
            dg.dev(credit_out.ptr, (size_t)E * sizeof(uint32_t));
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
