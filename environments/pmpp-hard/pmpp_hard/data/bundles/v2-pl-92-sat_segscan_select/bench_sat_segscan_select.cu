// file: bench_sat_segscan_select.cu

#include "sat_segscan_select_common.h"
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

static constexpr uint64_t g_state = 0x3d84fa17c26b905eULL;

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

__device__ __forceinline__ uint64_t sss_mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// C3 per-iteration input variation (I2): before every timed call, negate a
// CONTIGUOUS rotating block of v in place ([off, off+blk), off rotating by iter).
// v is int32 and the scan saturates to [lo,hi], so any int32 is in the legal
// domain; every generator's value pool is sign-symmetric, so negation keeps
// values in-family while genuinely changing the saturating segmented scan across
// that region (and downstream within its segments). A contiguous block keeps the
// touched memory to |blk| (vs a full-array masked pass), so the kernel stays a
// small fraction of one solution_run. Paired-cancelled: the reference bench runs
// the identical kernel with the identical (PMPP_BENCH_SEED-derived) off.
__global__ void sss_flip_block(int32_t* v, size_t off, size_t blk) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < blk; j += stride) {
        v[off + j] = -v[off + j];
    }
}

// C3 per-call output probe (P2): XOR-mix the exact-graded outputs of one timed
// call into that call's accumulator slot. y is N=33M (a full read would roughly
// double this memory-bound kernel), so instead of the whole array we bind:
//   * sat_bits (N/32 words) — flips propagate through saturation for the
//     saturating distributions;
//   * seg_last (S entries) — the per-segment scan endpoint, which reflects the
//     true segment sum for the non-saturating (smooth) distribution;
//   * sel_count;
//   * a rotating N/64 contiguous slice of y at offset (iter*N/64 mod N), so
//     across the 20 timed calls different regions of y are bound and each call's
//     probe covers a region the per-call input flip changes.
// XOR is order-independent -> deterministic; the digest folds every slot. A
// no-op/cached call leaves a stale region -> wrong slot -> digest mismatch.
// Total probe traffic is ~5% of one y-read, well under the timing budget.
__global__ void sss_probe(
    const int32_t* y, size_t ny, size_t y_off, size_t y_len,
    const uint32_t* sat_bits, size_t nw,
    const int32_t* seg_last, size_t nseg,
    const int32_t* sel_count,
    unsigned long long* slot) {
    __shared__ unsigned long long bh[256];
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t h = 0;
    for (size_t j = tid; j < y_len; j += stride) {
        const size_t i = (y_off + j) % ny;
        h ^= sss_mix64((uint64_t)(uint32_t)y[i] + 0x9e3779b97f4a7c15ULL * (uint64_t)(i + 1));
    }
    for (size_t i = tid; i < nw; i += stride)
        h ^= sss_mix64((uint64_t)sat_bits[i] + 0xD1B54A32D192ED03ULL * (uint64_t)(i + 1));
    for (size_t i = tid; i < nseg; i += stride)
        h ^= sss_mix64((uint64_t)(uint32_t)seg_last[i] + 0x9FB21C651E98DF25ULL * (uint64_t)(i + 1));
    if (tid == 0)
        h ^= sss_mix64((uint64_t)(uint32_t)(*sel_count) + 0xC2B2AE3D27D4EB4FULL);
    bh[threadIdx.x] = h;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) bh[threadIdx.x] ^= bh[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicXor(slot, bh[0]);
}

static void pick_rails(SplitMix64& rng, int dist, int32_t* lo, int32_t* hi) {
    switch (dist) {
        case SSS_DIST_SMOOTH:
            *lo = SSS_MIN_LO;
            *hi = SSS_MAX_HI;
            break;
        case SSS_DIST_TIGHT:
            *lo = -rng.uniform_int(1, 8);
            *hi = rng.uniform_int(1, 8);
            break;
        case SSS_DIST_SATRUN:
            *lo = -rng.uniform_int(100, 5000);
            *hi = rng.uniform_int(100, 5000);
            break;
        case SSS_DIST_RANDBITS:
        default:
            *lo = -(1 + (int)(rng.next_u64() % (1u << 24)));
            *hi = 1 + (int)(rng.next_u64() % (1u << 24));
            break;
    }
}

static void gen_flags(
    SplitMix64& rng, int N, int mean_seg, std::vector<uint32_t>* flags) {
    const int Nw = sss_Nw(N);
    flags->assign((size_t)Nw, 0u);
    for (int i = 0; i < N; ++i) {
        if ((int)(rng.next_u64() % (uint64_t)mean_seg) == 0) {
            (*flags)[(size_t)(i >> 5)] |= (1u << (i & 31));
        }
    }
    (*flags)[0] |= 1u;
}

static int32_t gen_value(SplitMix64& rng, int dist, int32_t lo, int32_t hi) {
    switch (dist) {
        case SSS_DIST_SMOOTH:
            return rng.uniform_int(-64, 64);
        case SSS_DIST_TIGHT:
            return rng.uniform_int(-12, 12);
        case SSS_DIST_SATRUN: {
            const int sel = rng.uniform_int(0, 6);
            switch (sel) {
                case 0: return hi;
                case 1: return -hi;
                case 2: return lo;
                case 3: return -lo;
                case 4: return hi - 1;
                case 5: return lo + 1;
                default: return rng.uniform_int(-16, 16);
            }
        }
        case SSS_DIST_RANDBITS:
        default:
            return rng.uniform_int(-SSS_MAX_ABS_V, SSS_MAX_ABS_V);
    }
}

struct HostCase {
    std::string name;
    SssRunSpec run;
    std::vector<int32_t> v;
    std::vector<uint32_t> flags;
    int S = 0;
};

static HostCase make_case(
    const std::string& name,
    int N,
    int distribution_id,
    int mean_seg,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    SplitMix64 rng(g_state ^ seed);

    int32_t lo, hi;
    pick_rails(rng, distribution_id, &lo, &hi);

    hc.run = {};
    hc.run.abi_version = SSS_ABI_VERSION;
    hc.run.N = N;
    hc.run.lo = lo;
    hc.run.hi = hi;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!sss_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid SssRunSpec generated");
    }

    gen_flags(rng, N, mean_seg, &hc.flags);

    hc.v.resize((size_t)N);
    for (int i = 0; i < N; ++i) {
        hc.v[(size_t)i] = gen_value(rng, distribution_id, lo, hi);
    }

    int S = 0;
    for (uint32_t w : hc.flags) S += __builtin_popcount(w);
    hc.S = S;

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    // Case data derives from PMPP_BENCH_SEED (whitened) so input values vary per
    // rollout; shape family (N / mean segment length / distribution) stays fixed below.
    SplitMix64 seed_mix(pmpp::bench_seed(0xE20000001b3ULL));
    uint64_t s = seed_mix.next_u64();

    cases.push_back(make_case(
        "bench_N33554432_seg512K_smooth", 33554432, SSS_DIST_SMOOTH, 524288, s++));
    cases.push_back(make_case(
        "bench_N33554432_seg128K_satrun", 33554432, SSS_DIST_SATRUN, 131072, s++));
    cases.push_back(make_case(
        "bench_N33554432_seg32K_randbits", 33554432, SSS_DIST_RANDBITS, 32768, s++));
    cases.push_back(make_case(
        "bench_N16777216_seg128K_tight", 16777216, SSS_DIST_TIGHT, 131072, s++));

    return cases;
}

struct DeviceCase {
    HostCase host;

    DeviceBuffer<int32_t> v;
    DeviceBuffer<uint32_t> flags;

    DeviceBuffer<int32_t> y;
    DeviceBuffer<uint32_t> sat_bits;
    DeviceBuffer<int32_t> sel_idx;
    DeviceBuffer<int32_t> sel_count;
    DeviceBuffer<int32_t> seg_last;

    SssInputs inputs;
    SssOutputs outputs;
};

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int Nw = sss_Nw(N);

    dc->v.allocate(hc.v.size());
    dc->flags.allocate(hc.flags.size());
    dc->v.upload(hc.v);
    dc->flags.upload(hc.flags);

    dc->y.allocate((size_t)N);
    dc->sat_bits.allocate((size_t)Nw);
    dc->sel_idx.allocate((size_t)N);
    dc->sel_count.allocate(1);
    dc->seg_last.allocate((size_t)std::max(hc.S, 1));

    dc->inputs = {};
    dc->inputs.v = dc->v.ptr;
    dc->inputs.flags = dc->flags.ptr;

    dc->outputs = {};
    dc->outputs.y = dc->y.ptr;
    dc->outputs.sat_bits = dc->sat_bits.ptr;
    dc->outputs.sel_idx = dc->sel_idx.ptr;
    dc->outputs.sel_count = dc->sel_count.ptr;
    dc->outputs.seg_last = dc->seg_last.ptr;

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

        int max_N = SSS_MIN_N;
        for (const HostCase& hc : host_cases) {
            max_N = std::max(max_N, hc.run.N);
        }

        SssProblemSpec spec = {};
        spec.abi_version = SSS_ABI_VERSION;
        spec.max_N = max_N;
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
            cases.push_back(make_device_case(hc));
            std::printf(
                "bench_case %-32s N=%d S=%d lo=%d hi=%d dist=%d variants=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.S,
                hc.run.lo,
                hc.run.hi,
                hc.run.distribution_id,
                iters);
        }

        // Per-(case, iter) block offsets from PMPP_BENCH_SEED: the timed input
        // sequence is deterministic per seed but differs on every timed call. The
        // flip block and the y probe slice share the offset so the probe always
        // covers the region the flip changed.
        std::vector<std::vector<uint64_t>> offs(cases.size());
        {
            SplitMix64 salt_mix(pmpp::bench_seed(0xE20000001b3ULL) ^ 0xC3F01DED5EEDULL);
            for (size_t c = 0; c < cases.size(); ++c) {
                SplitMix64 cs(salt_mix.next_u64());
                offs[c].resize((size_t)iters);
                for (int k = 0; k < iters; ++k) offs[c][(size_t)k] = cs.next_u64();
            }
        }
        // One probe slot per timed call (iter-major, case-minor), zeroed up front.
        DeviceBuffer<unsigned long long> probes;
        probes.allocate((size_t)iters * cases.size());
        CUDA_CHECK(cudaMemset(probes.ptr, 0,
                              (size_t)iters * cases.size() * sizeof(unsigned long long)));

        // Warmup runs on the pristine v; every timed call below mutates v first,
        // so warmup outputs can never satisfy a timed call's probe.
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
                const int N = dc->host.run.N;
                const int Nw = sss_Nw(N);
                // Contiguous block [off, off+blk) shared by the flip and the probe.
                const size_t blk = std::max<size_t>((size_t)N / 32, 1);
                size_t off = offs[c][(size_t)iter] % (size_t)N;
                if (off + blk > (size_t)N) off = (size_t)N - blk;
                // Mutate v in place, run, then probe the fresh outputs into this
                // call's slot (both kernels paired-cancelled in the reference bench).
                sss_flip_block<<<256, 256, 0, stream>>>(dc->v.ptr, off, blk);
                CUDA_CHECK(solution_run(
                    state,
                    &dc->host.run,
                    &dc->inputs,
                    &dc->outputs,
                    workspace.ptr,
                    workspace_bytes,
                    stream));
                sss_probe<<<256, 256, 0, stream>>>(
                    dc->y.ptr, (size_t)N, off, blk,
                    dc->sat_bits.ptr, (size_t)Nw,
                    dc->seg_last.ptr, (size_t)std::max(dc->host.S, 1),
                    dc->sel_count.ptr,
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

        // Anti-hack digest: the probe slots bind the exact-graded outputs (y,
        // sat_bits, seg_last, sel_count) of EVERY timed call to its own mutated
        // input — a no-op/cached call leaves a wrong slot and the digest cannot
        // match the reference bench. Fold all slots, then anchor byte-exactness by
        // folding the final iteration's raw outputs (still resident in the shared
        // buffers). sel_idx is graded only on [0, sel_count), so fold that prefix.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        dg.dev(probes.ptr, (size_t)iters * cases.size() * sizeof(unsigned long long));
        for (DeviceCase* dc : cases) {
            const int N = dc->host.run.N;
            const int Nw = sss_Nw(N);
            dg.dev(dc->y.ptr, (size_t)N * sizeof(int32_t));
            dg.dev(dc->sat_bits.ptr, (size_t)Nw * sizeof(uint32_t));
            dg.dev(dc->sel_count.ptr, sizeof(int32_t));
            dg.dev(dc->seg_last.ptr, (size_t)std::max(dc->host.S, 1) * sizeof(int32_t));

            int32_t sel_count = 0;
            CUDA_CHECK(cudaMemcpy(&sel_count, dc->sel_count.ptr,
                                  sizeof(int32_t), cudaMemcpyDeviceToHost));
            if (sel_count < 0) sel_count = 0;
            if (sel_count > N) sel_count = N;
            dg.dev(dc->sel_idx.ptr, (size_t)sel_count * sizeof(int32_t));
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
