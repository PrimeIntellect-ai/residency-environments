// ============================================================================
// file: bench_route_compact_reduce.cu
//
// C3 anti-cache design (patterns I1 + P2):
//   - Every timed solution_run call k receives fresh input data derived from
//     (PMPP_BENCH_SEED, case, k): expert routes are rotated by a per-iteration
//     amount and route weights are regenerated per slot. x stays fixed — with
//     fresh weights every sum/argmax_abs value must be recomputed from all of
//     x anyway, so there is no shortcut. All per-iteration variants are
//     pregenerated into a device ring BEFORE the timer (warmup uses its own
//     dedicated variants), so the timed loop only flips input pointers: no
//     mutation kernel and no freshly-dirtied input lines inside the timed
//     region (in-place mutation measurably slowed solution_run itself).
//   - After every timed call a fold kernel xor-mixes ALL graded output regions
//     (counts, offsets, packed prefix up to offsets[E], sum, argmax_abs) into
//     a device accumulator that is folded into out_fnv. A call that replays a
//     cached/stale output leaves wrong bytes for that iteration's inputs and
//     breaks the digest, regardless of how its timing looks.
// ============================================================================

#include "route_compact_reduce_common.h"
#include "pmpp_bench_digest.cuh"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0x6a09e667f3bcc909ULL;

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

static uint64_t host_mix64(uint64_t z) {
    z ^= z >> 30;
    z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27;
    z *= 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ uint64_t dev_mix64(uint64_t z) {
    z ^= z >> 30;
    z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27;
    z *= 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Batched across all cases so the timed loop pays only two extra kernel
// launches per full iteration (launch overhead dominates on small kernels).
#define RCR_BENCH_MAX_CASES 8

// Per-iteration input mutation: rotate expert IDs, regenerate weights.
// Reads pristine copies so iteration k is a pure function of (seed, case, k).
struct RcrMutArgs {
    int n_cases;
    long long grand_total;
    int16_t* e_dst[RCR_BENCH_MAX_CASES];
    int16_t* w_dst[RCR_BENCH_MAX_CASES];
    const int16_t* e_src[RCR_BENCH_MAX_CASES];
    const int16_t* w_src[RCR_BENCH_MAX_CASES];
    long long n_slots[RCR_BENCH_MAX_CASES];
    int E[RCR_BENCH_MAX_CASES];
    int rot[RCR_BENCH_MAX_CASES];
    unsigned long long salt[RCR_BENCH_MAX_CASES];
};

__global__ void rcr_mutate_all(RcrMutArgs a) {
    for (long long idx =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < a.grand_total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        long long r = idx;
        int ci = 0;
        while (r >= a.n_slots[ci]) {
            r -= a.n_slots[ci];
            ++ci;
        }
        const int16_t e0 = a.e_src[ci][r];
        a.e_dst[ci][r] = (e0 < 0)
            ? static_cast<int16_t>(-1)
            : static_cast<int16_t>((e0 + a.rot[ci]) % a.E[ci]);
        const int16_t w0 = a.w_src[ci][r];
        if (w0 == 0) {
            a.w_dst[ci][r] = 0;
            continue;
        }
        const uint64_t h = dev_mix64(
            a.salt[ci] ^ (static_cast<uint64_t>(r) * 0x9e3779b97f4a7c15ULL));
        const int mag = 1 + static_cast<int>(h % 9);
        a.w_dst[ci][r] =
            (h & 512) ? static_cast<int16_t>(-mag) : static_cast<int16_t>(mag);
    }
}

// Folds every graded output region of one timed call (all cases) into *acc.
// Xor of position-mixed words is order-independent → deterministic digest.
struct RcrFoldArgs {
    unsigned long long* acc;
    int n_cases;
    long long grand_total;
    const int32_t* counts[RCR_BENCH_MAX_CASES];
    const int32_t* offsets[RCR_BENCH_MAX_CASES];
    const int32_t* ptok[RCR_BENCH_MAX_CASES];
    const int32_t* pwt[RCR_BENCH_MAX_CASES];
    const long long* sum[RCR_BENCH_MAX_CASES];
    const int32_t* amax[RCR_BENCH_MAX_CASES];
    long long total[RCR_BENCH_MAX_CASES];
    int E[RCR_BENCH_MAX_CASES];
    int C[RCR_BENCH_MAX_CASES];
    long long NK[RCR_BENCH_MAX_CASES];
    unsigned long long salt[RCR_BENCH_MAX_CASES];
};

__global__ void rcr_fold_all(RcrFoldArgs a) {
    uint64_t local = 0;
    for (long long idx =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < a.grand_total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        long long r = idx;
        int ci = 0;
        while (r >= a.total[ci]) {
            r -= a.total[ci];
            ++ci;
        }
        const int E = a.E[ci];
        const long long NK = a.NK[ci];
        const long long ec = static_cast<long long>(E) * a.C[ci];
        const int n_packed = a.offsets[ci][E];
        uint64_t v;
        int region;
        if (r < E) {
            region = 0;
            v = static_cast<uint32_t>(a.counts[ci][r]);
        } else if ((r -= E) < E + 1) {
            region = 1;
            v = static_cast<uint32_t>(a.offsets[ci][r]);
        } else if ((r -= E + 1) < NK) {
            if (r >= n_packed) continue;
            region = 2;
            v = static_cast<uint32_t>(a.ptok[ci][r]);
        } else if ((r -= NK) < NK) {
            if (r >= n_packed) continue;
            region = 3;
            v = static_cast<uint32_t>(a.pwt[ci][r]);
        } else if ((r -= NK) < ec) {
            region = 4;
            v = static_cast<uint64_t>(a.sum[ci][r]);
        } else {
            r -= ec;
            region = 5;
            v = static_cast<uint32_t>(a.amax[ci][r]);
        }
        local ^= dev_mix64(
            v ^ dev_mix64(a.salt[ci] ^ (static_cast<uint64_t>(region) << 56) ^
                          static_cast<uint64_t>(r)));
    }

    __shared__ uint64_t sh[256];
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[threadIdx.x] ^= sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicXor(a.acc, static_cast<unsigned long long>(sh[0]));
    }
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
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

struct HostCase {
    std::string name;
    RcrRunSpec run;
    std::vector<int16_t> x;
    std::vector<int16_t> expert;
    std::vector<int16_t> weight;
    std::vector<uint8_t> valid;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<int16_t> x;
    DeviceBuffer<int16_t> expert;
    DeviceBuffer<int16_t> weight;
    DeviceBuffer<uint8_t> valid;

    // Pregenerated per-iteration routing-input variants (ring of n_vars
    // copies, variant v at slot offset v * N * K); expert/weight above stay
    // pristine as the generation source.
    DeviceBuffer<int16_t> expert_vars;
    DeviceBuffer<int16_t> weight_vars;

    DeviceBuffer<int32_t> counts;
    DeviceBuffer<int32_t> offsets;
    DeviceBuffer<int32_t> packed_token;
    DeviceBuffer<int32_t> packed_weight;
    DeviceBuffer<int64_t> sum;
    DeviceBuffer<int32_t> argmax_abs;

    RcrInputs inputs;
    RcrOutputs outputs;
};

static int sample_zipf_1_2(SplitMix64& rng, int limit_e) {
    double total = 0.0;
    for (int i = 0; i < limit_e; ++i) {
        total += 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
    }

    const double target = rng.uniform01() * total;
    double accum = 0.0;
    for (int i = 0; i < limit_e; ++i) {
        accum += 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
        if (accum >= target) {
            return i;
        }
    }

    return limit_e - 1;
}

static int choose_expert(
    SplitMix64& rng,
    int E,
    int distribution_id,
    bool avoid_last_expert) {
    const int limit_e = avoid_last_expert && E > 1 ? E - 1 : E;
    if (limit_e <= 1) {
        return 0;
    }

    switch (distribution_id) {
        case RCR_DIST_UNIFORM:
            return rng.uniform_int(0, limit_e - 1);

        case RCR_DIST_ZIPF_1_2:
            return sample_zipf_1_2(rng, limit_e);

        case RCR_DIST_SINGLE_HOT:
            if (rng.chance(0.88)) {
                return 0;
            }
            return rng.uniform_int(1, limit_e - 1);

        case RCR_DIST_FEW_HOT: {
            const int hot_count = std::min(limit_e, limit_e >= 8 ? 4 : 2);
            if (rng.chance(0.90)) {
                return rng.uniform_int(0, hot_count - 1);
            }
            if (hot_count == limit_e) {
                return rng.uniform_int(0, limit_e - 1);
            }
            return rng.uniform_int(hot_count, limit_e - 1);
        }

        case RCR_DIST_MANY_INVALID:
            return rng.uniform_int(0, limit_e - 1);

        case RCR_DIST_MANY_DUPLICATE:
            if (rng.chance(0.70)) {
                return rng.uniform_int(0, std::min(limit_e - 1, 3));
            }
            return rng.uniform_int(0, limit_e - 1);

        default:
            return rng.uniform_int(0, limit_e - 1);
    }
}

static int16_t choose_weight(SplitMix64& rng, int distribution_id) {
    if (distribution_id == RCR_DIST_MANY_INVALID && rng.chance(0.15)) {
        return 0;
    }

    int w = rng.uniform_int(-9, 9);
    if (w == 0) {
        w = rng.chance(0.5) ? 1 : -1;
    }
    return static_cast<int16_t>(w);
}

static void force_cancel_duplicate_token(HostCase* hc) {
    const int N = hc->run.N;
    const int E = hc->run.E;
    const int K = hc->run.K;
    if (N < 8 || E < 2) {
        return;
    }

    const int t = 7;
    const int e = std::min(1, E - 1);
    hc->valid[t] = 1;

    for (int k = 0; k < K; ++k) {
        hc->expert[t * K + k] = static_cast<int16_t>(e);
        hc->weight[t * K + k] = 0;
    }

    if (K == 2) {
        hc->weight[t * K + 0] = 5;
        hc->weight[t * K + 1] = -5;
    } else {
        hc->weight[t * K + 0] = 7;
        hc->weight[t * K + 1] = -2;
        hc->weight[t * K + 2] = -5;
        hc->weight[t * K + 3] = 0;
    }
}

static HostCase make_case(
    const char* name,
    int N,
    int C,
    int E,
    int K,
    int distribution_id,
    uint64_t case_seed,
    bool force_all_invalid,
    bool force_empty_last_expert,
    bool inject_cancel_duplicate) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = RCR_ABI_VERSION;
    hc.run.N = N;
    hc.run.C = C;
    hc.run.E = E;
    hc.run.K = K;
    hc.run.seed_id = static_cast<int32_t>(case_seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((case_seed >> 32) & 0x7fffffff);

    if (!rcr_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated run spec");
    }

    SplitMix64 rng(g_state ^ case_seed);

    hc.x.resize(static_cast<size_t>(N) * static_cast<size_t>(C));
    hc.expert.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.weight.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.valid.resize(static_cast<size_t>(N));

    for (size_t i = 0; i < hc.x.size(); ++i) {
        int v = rng.uniform_int(-31, 31);
        if (v == 0 && rng.chance(0.85)) {
            v = rng.chance(0.5) ? 1 : -1;
        }
        hc.x[i] = static_cast<int16_t>(v);
    }

    for (int t = 0; t < N; ++t) {
        bool is_valid = true;

        if (force_all_invalid) {
            is_valid = false;
        } else if (distribution_id == RCR_DIST_MANY_INVALID) {
            is_valid = !rng.chance(0.55);
        } else {
            is_valid = !rng.chance(0.03);
        }

        hc.valid[t] = static_cast<uint8_t>(is_valid ? 1 : 0);

        if (!is_valid) {
            for (int k = 0; k < K; ++k) {
                hc.expert[t * K + k] = rng.chance(0.70)
                    ? static_cast<int16_t>(-1)
                    : static_cast<int16_t>(choose_expert(rng, E, distribution_id, force_empty_last_expert));
                hc.weight[t * K + k] = choose_weight(rng, distribution_id);
            }
            continue;
        }

        if (distribution_id == RCR_DIST_MANY_DUPLICATE) {
            const int e0 = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            for (int k = 0; k < K; ++k) {
                int e = e0;
                if (k == K - 1 && rng.chance(0.35)) {
                    e = choose_expert(rng, E, RCR_DIST_UNIFORM, force_empty_last_expert);
                }
                hc.expert[t * K + k] = static_cast<int16_t>(e);
                hc.weight[t * K + k] = choose_weight(rng, distribution_id);
            }
            continue;
        }

        for (int k = 0; k < K; ++k) {
            const double invalid_slot_prob =
                distribution_id == RCR_DIST_MANY_INVALID ? 0.45 : 0.04;

            if (rng.chance(invalid_slot_prob)) {
                hc.expert[t * K + k] = -1;
                hc.weight[t * K + k] = choose_weight(rng, distribution_id);
                continue;
            }

            const int e = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            hc.expert[t * K + k] = static_cast<int16_t>(e);
            hc.weight[t * K + k] = choose_weight(rng, distribution_id);
        }
    }

    if (inject_cancel_duplicate) {
        force_cancel_duplicate_token(&hc);
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    // Case data values derive from PMPP_BENCH_SEED so graded outputs are not
    // precomputable offline; shapes stay fixed so timing remains comparable.
    uint64_t s = 0x200000001b3ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL);

    cases.push_back(make_case(
        "uniform_mid",
        65536, 32, 32, 2,
        RCR_DIST_UNIFORM,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "single_hot_large",
        131072, 64, 128, 4,
        RCR_DIST_SINGLE_HOT,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "many_duplicate_wide",
        65536, 128, 32, 4,
        RCR_DIST_MANY_DUPLICATE,
        s++,
        false,
        false,
        true));

    cases.push_back(make_case(
        "many_invalid_smallE",
        32768, 16, 8, 2,
        RCR_DIST_MANY_INVALID,
        s++,
        false,
        false,
        false));

    cases.push_back(make_case(
        "zipf_largeE",
        131072, 32, 128, 4,
        RCR_DIST_ZIPF_1_2,
        s++,
        false,
        false,
        false));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc, int n_vars) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int C = hc.run.C;
    const int E = hc.run.E;
    const int K = hc.run.K;

    dc->x.allocate(hc.x.size());
    dc->expert.allocate(hc.expert.size());
    dc->weight.allocate(hc.weight.size());
    dc->valid.allocate(hc.valid.size());

    dc->x.upload(hc.x);
    dc->expert.upload(hc.expert);
    dc->weight.upload(hc.weight);
    dc->valid.upload(hc.valid);

    dc->expert_vars.allocate(hc.expert.size() * static_cast<size_t>(n_vars));
    dc->weight_vars.allocate(hc.weight.size() * static_cast<size_t>(n_vars));

    dc->counts.allocate(static_cast<size_t>(E));
    dc->offsets.allocate(static_cast<size_t>(E + 1));
    dc->packed_token.allocate(static_cast<size_t>(N) * static_cast<size_t>(K));
    dc->packed_weight.allocate(static_cast<size_t>(N) * static_cast<size_t>(K));
    dc->sum.allocate(static_cast<size_t>(E) * static_cast<size_t>(C));
    dc->argmax_abs.allocate(static_cast<size_t>(E) * static_cast<size_t>(C));

    dc->inputs = {};
    dc->inputs.x = dc->x.ptr;
    dc->inputs.expert = dc->expert_vars.ptr;
    dc->inputs.weight = dc->weight_vars.ptr;
    dc->inputs.valid = dc->valid.ptr;

    dc->outputs = {};
    dc->outputs.counts = dc->counts.ptr;
    dc->outputs.offsets = dc->offsets.ptr;
    dc->outputs.packed_token = dc->packed_token.ptr;
    dc->outputs.packed_weight = dc->packed_weight.ptr;
    dc->outputs.sum = dc->sum.ptr;
    dc->outputs.argmax_abs = dc->argmax_abs.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        RcrProblemSpec spec = {};
        spec.abi_version = RCR_ABI_VERSION;
        spec.max_N = RCR_MAX_N;
        spec.max_C = RCR_MAX_C;
        spec.max_E = RCR_MAX_E;
        spec.max_K = RCR_MAX_K;
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

        // Variant ring: one dedicated input variant per timed iteration
        // (capped for oversized local runs) plus dedicated warmup variants, so
        // no timed call ever re-sees data that was visible before the timer.
        const int kWarmups = 5;
        const int v_timed = std::min(iters, 32);
        const int n_vars = v_timed + kWarmups;

        std::vector<HostCase> host_cases = build_bench_cases();
        std::vector<DeviceCase*> cases;
        cases.reserve(host_cases.size());

        for (const HostCase& hc : host_cases) {
            cases.push_back(make_device_case(hc, n_vars));
            std::printf(
                "bench_case %-24s N=%d C=%d E=%d K=%d dist=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.C,
                hc.run.E,
                hc.run.K,
                hc.run.distribution_id);
        }

        const uint64_t fold_seed = 0x1b3c3f01dULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL);

        if (cases.size() > RCR_BENCH_MAX_CASES) {
            throw std::runtime_error("too many bench cases for batched C3 kernels");
        }

        DeviceBuffer<unsigned long long> fold_acc;
        fold_acc.allocate(1);
        CUDA_CHECK(cudaMemsetAsync(fold_acc.ptr, 0, sizeof(unsigned long long), stream));

        // Static (pointer/shape) halves of the batched kernel args.
        RcrMutArgs mut = {};
        RcrFoldArgs fold = {};
        mut.n_cases = static_cast<int>(cases.size());
        fold.n_cases = static_cast<int>(cases.size());
        fold.acc = fold_acc.ptr;
        for (size_t ci = 0; ci < cases.size(); ++ci) {
            DeviceCase* dc = cases[ci];
            const int E = dc->host.run.E;
            const int C = dc->host.run.C;
            const long long NK =
                static_cast<long long>(dc->host.run.N) * dc->host.run.K;
            mut.e_dst[ci] = dc->expert_vars.ptr;
            mut.w_dst[ci] = dc->weight_vars.ptr;
            mut.e_src[ci] = dc->expert.ptr;
            mut.w_src[ci] = dc->weight.ptr;
            mut.n_slots[ci] = NK;
            mut.E[ci] = E;
            mut.grand_total += NK;
            fold.counts[ci] = dc->counts.ptr;
            fold.offsets[ci] = dc->offsets.ptr;
            fold.ptok[ci] = dc->packed_token.ptr;
            fold.pwt[ci] = dc->packed_weight.ptr;
            fold.sum[ci] = reinterpret_cast<const long long*>(dc->sum.ptr);
            fold.amax[ci] = dc->argmax_abs.ptr;
            fold.E[ci] = E;
            fold.C[ci] = C;
            fold.NK[ci] = NK;
            fold.total[ci] = static_cast<long long>(E) + (E + 1) + 2LL * NK +
                             2LL * static_cast<long long>(E) * C;
            fold.grand_total += fold.total[ci];
        }
        const int kThreads = 256;
        const int mut_blocks = static_cast<int>(
            std::min<long long>((mut.grand_total + kThreads - 1) / kThreads, 4096));
        const int fold_blocks = static_cast<int>(
            std::min<long long>((fold.grand_total + kThreads - 1) / kThreads, 4096));

        // Generates variant slot v of the ring from iteration tag `iter_tag`
        // (inputs of iteration k are a pure function of (PMPP_BENCH_SEED,
        // case, k)). Runs UNTIMED, before the timer starts.
        auto generate_variant = [&](int v, uint64_t iter_tag) {
            for (int ci = 0; ci < mut.n_cases; ++ci) {
                DeviceCase* dc = cases[ci];
                const size_t nk = dc->host.expert.size();
                mut.e_dst[ci] = dc->expert_vars.ptr + static_cast<size_t>(v) * nk;
                mut.w_dst[ci] = dc->weight_vars.ptr + static_cast<size_t>(v) * nk;
                const uint64_t salt = host_mix64(fold_seed ^ host_mix64(
                    static_cast<uint64_t>(ci) * 0xbf58476d1ce4e5b9ULL + iter_tag));
                mut.salt[ci] = salt;
                mut.rot[ci] = static_cast<int>(
                    (salt >> 8) % static_cast<uint64_t>(mut.E[ci]));
            }
            rcr_mutate_all<<<mut_blocks, kThreads, 0, stream>>>(mut);
        };
        // Points every case's routing inputs at variant slot v (host-side
        // pointer flip: zero GPU work inside the timed region).
        auto set_inputs = [&](int v) {
            for (size_t ci = 0; ci < cases.size(); ++ci) {
                DeviceCase* dc = cases[ci];
                const size_t nk = dc->host.expert.size();
                dc->inputs.expert =
                    dc->expert_vars.ptr + static_cast<size_t>(v) * nk;
                dc->inputs.weight =
                    dc->weight_vars.ptr + static_cast<size_t>(v) * nk;
            }
        };
        auto fold_iter = [&](uint64_t iter_tag) {
            for (int ci = 0; ci < fold.n_cases; ++ci) {
                fold.salt[ci] = host_mix64(fold_seed ^ host_mix64(
                    static_cast<uint64_t>(ci) * 0xbf58476d1ce4e5b9ULL + iter_tag));
            }
            rcr_fold_all<<<fold_blocks, kThreads, 0, stream>>>(fold);
        };
        auto run_all_cases = [&]() {
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
        };

        // Pregenerate the whole variant ring (untimed): timed variants first,
        // then the dedicated warmup variants.
        for (int v = 0; v < v_timed; ++v) {
            generate_variant(v, static_cast<uint64_t>(v));
        }
        for (int w = 0; w < kWarmups; ++w) {
            generate_variant(v_timed + w, 0x40000000ULL + w);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int warmup = 0; warmup < kWarmups; ++warmup) {
            set_inputs(v_timed + warmup);
            run_all_cases();
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));

        // Iteration k: fold outputs of k-1 (they persist until the k runs
        // overwrite them), flip inputs to variant k, run all cases. The final
        // iteration's outputs are folded right after the stop event, before
        // anything can overwrite them.
        for (int iter = 0; iter < iters; ++iter) {
            if (iter > 0) {
                fold_iter(static_cast<uint64_t>(iter - 1));
            }
            set_inputs(iter % v_timed);
            run_all_cases();
        }

        CUDA_CHECK(cudaEventRecord(stop, stream));
        fold_iter(static_cast<uint64_t>(iters - 1));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        const double calls = static_cast<double>(iters) * static_cast<double>(cases.size());
        const double avg_ms = static_cast<double>(elapsed_ms) / calls;

        std::printf("avg_ms=%.6f\n", avg_ms);

        // Untimed graded-output digest: one more run per case, then fold the graded
        // output ranges in fixed case order. packed_token/packed_weight are only
        // graded up to offsets[E] (the tail is unspecified), so fold that prefix only.
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
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        for (DeviceCase* dc : cases) {
            const int E = dc->host.run.E;
            const int C = dc->host.run.C;
            int32_t total_packed = 0;
            CUDA_CHECK(cudaMemcpy(&total_packed, dc->offsets.ptr + E, sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            dg.dev(dc->counts.ptr, (size_t)E * sizeof(int32_t));
            dg.dev(dc->offsets.ptr, (size_t)(E + 1) * sizeof(int32_t));
            if (total_packed > 0) {
                dg.dev(dc->packed_token.ptr, (size_t)total_packed * sizeof(int32_t));
                dg.dev(dc->packed_weight.ptr, (size_t)total_packed * sizeof(int32_t));
            }
            dg.dev(dc->sum.ptr, (size_t)E * (size_t)C * sizeof(int64_t));
            dg.dev(dc->argmax_abs.ptr, (size_t)E * (size_t)C * sizeof(int32_t));
        }
        // Bind every timed call's outputs: fold the per-iteration accumulator.
        unsigned long long fold_acc_host = 0;
        CUDA_CHECK(cudaMemcpy(&fold_acc_host, fold_acc.ptr,
                              sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        dg.bytes(&fold_acc_host, sizeof(fold_acc_host));
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
