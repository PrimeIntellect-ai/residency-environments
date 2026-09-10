// ============================================================================
// file: bench_moe_dispatch_combine.cu
//
// C3 anti-cache design (patterns I2 + P2):
//   - Every timed solution_run call k receives fresh input data derived from
//     (PMPP_BENCH_SEED, case, k): expert routes are rotated by a per-iteration
//     amount, and gate / expert_out values are regenerated per element. With
//     fresh gates the dispatch ranking (and drop set) must be re-derived, and
//     with fresh expert_out every y value must be recomputed — no cached or
//     incremental shortcut survives.
//   - After every timed call a fold kernel xor-mixes ALL graded output regions
//     (counts, offsets, packed prefix up to offsets[E], normalized dropped, y)
//     into a device accumulator that is folded into out_fnv. A call that
//     replays cached/stale outputs breaks the digest no matter how normal its
//     timing looks.
// ============================================================================

#include "moe_dispatch_combine_common.h"
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

static constexpr uint64_t g_state = 0x243f6a8885a308d3ULL;

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
    MdcRunSpec run;
    std::vector<int16_t> expert;
    std::vector<int16_t> gate;
    std::vector<uint8_t> valid;
    std::vector<int16_t> expert_out;
};

__device__ __forceinline__ uint64_t c3_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Per-iteration input regeneration (I2): rotates expert ids and regenerates
// gate / expert_out values from (PMPP_BENCH_SEED, case, iter), reading the
// pristine uploads so iteration k is a pure function of the seed tuple.
// valid[] and the -1 (unused-slot) pattern stay fixed: structure constant,
// data fresh, so dispatch ranking and every y value must be recomputed.
__global__ void c3_mutate_mdc(const int16_t* e_src, int16_t* e_dst,
                              const int16_t* g_src, int16_t* g_dst,
                              const int16_t* eo_src, int16_t* eo_dst,
                              long long nk, long long ed, int E, int rot,
                              uint64_t stream_key, int dist) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < nk + ed; i += stride) {
        const uint64_t h =
            c3_mix(stream_key ^ (uint64_t)(i + 1) * 0x9e3779b97f4a7c15ULL);
        if (i < nk) {
            const int16_t e0 = e_src[i];
            e_dst[i] = (e0 < 0) ? (int16_t)-1 : (int16_t)((e0 + rot) % E);
            int lo, hi;
            switch (dist) {
                case MDC_DIST_SINGLE_HOT:
                case MDC_DIST_FEW_HOT: lo = -16; hi = 127; break;
                case MDC_DIST_MANY_INVALID: lo = -127; hi = 64; break;
                case MDC_DIST_MANY_DUPLICATE: lo = -32; hi = 127; break;
                default: lo = -127; hi = 127; break;
            }
            g_dst[i] = (int16_t)(lo + (int)(h % (uint64_t)(hi - lo + 1)));
        } else {
            const long long j = i - nk;
            int v = (int)(h % 63) - 31;
            if (v == 0) v = (h & 256) ? 1 : -1;
            eo_dst[j] = (int16_t)v;
            (void)eo_src;
        }
    }
}

struct C3ProbeDesc {
    const uint8_t* ptr;
    unsigned long long fixed_bytes;   // used when len_ptr == nullptr
    const int32_t* len_ptr;           // device count (e.g. offsets[E])
    unsigned long long elem_stride;   // bytes per counted element
    int as_bool;                      // fold byte ? 1 : 0 (graded-as-bool)
    int pad;
};

struct C3ProbeTable {
    C3ProbeDesc d[8];
    int n;
};

// One launch per timed call: xor-folds a position-salted mix of every graded
// output region into acc[slot]. xor accumulation is order-independent, so the
// result is deterministic; honest student and reference produce identical
// probes on the same seed. A cached/no-op timed call leaves stale bytes for
// that iteration's inputs and breaks the digest.
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
        const uint64_t bsalt = salt ^ (uint64_t)(b + 1) * 0x100000001b3ULL;
        if (d.as_bool) {
            for (unsigned long long i = gtid; i < nbytes; i += gstride) {
                const unsigned long long v = d.ptr[i] ? 1ULL : 0ULL;
                local ^= c3_mix(v ^ c3_mix(bsalt + i));
            }
            continue;
        }
        const unsigned long long nwords = nbytes >> 3;
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

struct DeviceCase {
    HostCase host;

    DeviceBuffer<int16_t> expert;
    DeviceBuffer<int16_t> gate;
    DeviceBuffer<uint8_t> valid;
    DeviceBuffer<int16_t> expert_out;

    // Active (per-iteration mutated) inputs; the buffers above stay pristine
    // as the mutation source.
    DeviceBuffer<int16_t> expert_act;
    DeviceBuffer<int16_t> gate_act;
    DeviceBuffer<int16_t> expert_out_act;

    DeviceBuffer<int32_t> counts;
    DeviceBuffer<int32_t> offsets;
    DeviceBuffer<int32_t> packed_token;
    DeviceBuffer<int32_t> packed_slot;
    DeviceBuffer<int16_t> packed_gate;
    DeviceBuffer<uint8_t> dropped;
    DeviceBuffer<int64_t> y;

    MdcInputs inputs;
    MdcOutputs outputs;
    C3ProbeTable probe;
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
        case MDC_DIST_UNIFORM:
            return rng.uniform_int(0, limit_e - 1);

        case MDC_DIST_ZIPF_1_2:
            return sample_zipf_1_2(rng, limit_e);

        case MDC_DIST_SINGLE_HOT:
            if (rng.chance(0.90)) {
                return 0;
            }
            return rng.uniform_int(1, limit_e - 1);

        case MDC_DIST_FEW_HOT: {
            const int hot_count = std::min(limit_e, limit_e >= 8 ? 4 : 2);
            if (rng.chance(0.92)) {
                return rng.uniform_int(0, hot_count - 1);
            }
            if (hot_count == limit_e) {
                return rng.uniform_int(0, limit_e - 1);
            }
            return rng.uniform_int(hot_count, limit_e - 1);
        }

        case MDC_DIST_MANY_INVALID:
            return rng.uniform_int(0, limit_e - 1);

        case MDC_DIST_MANY_DUPLICATE:
            if (rng.chance(0.75)) {
                return rng.uniform_int(0, std::min(limit_e - 1, 3));
            }
            return rng.uniform_int(0, limit_e - 1);

        default:
            return rng.uniform_int(0, limit_e - 1);
    }
}

static int16_t choose_gate(SplitMix64& rng, int distribution_id) {
    switch (distribution_id) {
        case MDC_DIST_SINGLE_HOT:
        case MDC_DIST_FEW_HOT:
            return static_cast<int16_t>(rng.uniform_int(-16, 127));

        case MDC_DIST_MANY_INVALID:
            return static_cast<int16_t>(rng.uniform_int(-127, 64));

        case MDC_DIST_MANY_DUPLICATE:
            return static_cast<int16_t>(rng.uniform_int(-32, 127));

        default:
            return static_cast<int16_t>(rng.uniform_int(-127, 127));
    }
}

static HostCase make_random_case(
    const char* name,
    int N,
    int D,
    int E,
    int K,
    int cap,
    int distribution_id,
    uint64_t case_seed,
    bool force_all_invalid,
    bool force_empty_last_expert) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = MDC_ABI_VERSION;
    hc.run.N = N;
    hc.run.D = D;
    hc.run.E = E;
    hc.run.K = K;
    hc.run.cap = cap;
    hc.run.seed_id = static_cast<int32_t>(case_seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((case_seed >> 32) & 0x7fffffff);

    if (!mdc_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated run spec");
    }

    SplitMix64 rng(g_state ^ case_seed);

    hc.expert.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.gate.resize(static_cast<size_t>(N) * static_cast<size_t>(K));
    hc.valid.resize(static_cast<size_t>(N));
    hc.expert_out.resize(static_cast<size_t>(E) * static_cast<size_t>(D));

    for (size_t i = 0; i < hc.expert_out.size(); ++i) {
        int v = rng.uniform_int(-31, 31);
        if (v == 0 && rng.chance(0.90)) {
            v = rng.chance(0.5) ? 1 : -1;
        }
        hc.expert_out[i] = static_cast<int16_t>(v);
    }

    for (int t = 0; t < N; ++t) {
        bool is_valid = true;

        if (force_all_invalid) {
            is_valid = false;
        } else if (distribution_id == MDC_DIST_MANY_INVALID) {
            is_valid = !rng.chance(0.60);
        } else {
            is_valid = !rng.chance(0.02);
        }

        hc.valid[t] = static_cast<uint8_t>(is_valid ? 1 : 0);

        if (!is_valid) {
            for (int k = 0; k < K; ++k) {
                hc.expert[t * K + k] = rng.chance(0.70)
                    ? static_cast<int16_t>(-1)
                    : static_cast<int16_t>(choose_expert(rng, E, distribution_id, force_empty_last_expert));
                hc.gate[t * K + k] = choose_gate(rng, distribution_id);
            }
            continue;
        }

        if (distribution_id == MDC_DIST_MANY_DUPLICATE) {
            const int e0 = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            for (int k = 0; k < K; ++k) {
                int e = e0;
                if (k == K - 1 && rng.chance(0.35)) {
                    e = choose_expert(rng, E, MDC_DIST_UNIFORM, force_empty_last_expert);
                }
                hc.expert[t * K + k] = static_cast<int16_t>(e);
                hc.gate[t * K + k] = choose_gate(rng, distribution_id);
            }
            continue;
        }

        for (int k = 0; k < K; ++k) {
            const double unused_prob =
                distribution_id == MDC_DIST_MANY_INVALID ? 0.50 : 0.04;

            if (rng.chance(unused_prob)) {
                hc.expert[t * K + k] = -1;
                hc.gate[t * K + k] = choose_gate(rng, distribution_id);
                continue;
            }

            const int e = choose_expert(rng, E, distribution_id, force_empty_last_expert);
            hc.expert[t * K + k] = static_cast<int16_t>(e);
            hc.gate[t * K + k] = choose_gate(rng, distribution_id);
        }
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    // Case data values derive from PMPP_BENCH_SEED so graded outputs are not
    // precomputable offline; shapes stay fixed so timing remains comparable.
    uint64_t s = 0x200000001b3ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL);

    cases.push_back(make_random_case(
        "uniform_mid",
        65536, 32, 32, 2, 128,
        MDC_DIST_UNIFORM,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "single_hot_overflow_large",
        131072, 64, 128, 4, 16,
        MDC_DIST_SINGLE_HOT,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "few_hot_overflow_wide",
        65536, 128, 32, 4, 32,
        MDC_DIST_FEW_HOT,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "many_duplicate_hot",
        131072, 32, 8, 4, 16,
        MDC_DIST_MANY_DUPLICATE,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "zipf_largeE",
        131072, 16, 128, 4, 64,
        MDC_DIST_ZIPF_1_2,
        s++,
        false,
        false));

    cases.push_back(make_random_case(
        "many_invalid_smallE",
        32768, 16, 8, 2, 64,
        MDC_DIST_MANY_INVALID,
        s++,
        false,
        false));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    const int N = hc.run.N;
    const int D = hc.run.D;
    const int E = hc.run.E;
    const int K = hc.run.K;
    const int cap = hc.run.cap;
    const size_t packed_capacity = static_cast<size_t>(E) * static_cast<size_t>(cap);

    dc->expert.allocate(hc.expert.size());
    dc->gate.allocate(hc.gate.size());
    dc->valid.allocate(hc.valid.size());
    dc->expert_out.allocate(hc.expert_out.size());

    dc->expert.upload(hc.expert);
    dc->gate.upload(hc.gate);
    dc->valid.upload(hc.valid);
    dc->expert_out.upload(hc.expert_out);

    dc->expert_act.allocate(hc.expert.size());
    dc->gate_act.allocate(hc.gate.size());
    dc->expert_out_act.allocate(hc.expert_out.size());
    // Warmup runs before the first timed mutation: start the active buffers
    // as a copy of the pristine data.
    CUDA_CHECK(cudaMemcpy(dc->expert_act.ptr, dc->expert.ptr,
                          hc.expert.size() * sizeof(int16_t),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dc->gate_act.ptr, dc->gate.ptr,
                          hc.gate.size() * sizeof(int16_t),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dc->expert_out_act.ptr, dc->expert_out.ptr,
                          hc.expert_out.size() * sizeof(int16_t),
                          cudaMemcpyDeviceToDevice));

    dc->counts.allocate(static_cast<size_t>(E));
    dc->offsets.allocate(static_cast<size_t>(E + 1));
    dc->packed_token.allocate(packed_capacity);
    dc->packed_slot.allocate(packed_capacity);
    dc->packed_gate.allocate(packed_capacity);
    dc->dropped.allocate(static_cast<size_t>(N) * static_cast<size_t>(K));
    dc->y.allocate(static_cast<size_t>(N) * static_cast<size_t>(D));

    dc->inputs = {};
    dc->inputs.expert = dc->expert_act.ptr;
    dc->inputs.gate = dc->gate_act.ptr;
    dc->inputs.valid = dc->valid.ptr;
    dc->inputs.expert_out = dc->expert_out_act.ptr;

    dc->outputs = {};
    dc->outputs.counts = dc->counts.ptr;
    dc->outputs.offsets = dc->offsets.ptr;
    dc->outputs.packed_token = dc->packed_token.ptr;
    dc->outputs.packed_slot = dc->packed_slot.ptr;
    dc->outputs.packed_gate = dc->packed_gate.ptr;
    dc->outputs.dropped = dc->dropped.ptr;
    dc->outputs.y = dc->y.ptr;

    // C3 probe table: exactly the graded output set (mirrors the digest fold
    // below; packed_* graded up to offsets[E], dropped graded as bool).
    const int32_t* d_total = dc->offsets.ptr + E;
    C3ProbeTable& pt = dc->probe;
    pt.n = 7;
    pt.d[0] = {(const uint8_t*)dc->counts.ptr,
               (unsigned long long)E * sizeof(int32_t), nullptr, 0, 0, 0};
    pt.d[1] = {(const uint8_t*)dc->offsets.ptr,
               (unsigned long long)(E + 1) * sizeof(int32_t), nullptr, 0, 0, 0};
    pt.d[2] = {(const uint8_t*)dc->packed_token.ptr, 0, d_total,
               sizeof(int32_t), 0, 0};
    pt.d[3] = {(const uint8_t*)dc->packed_slot.ptr, 0, d_total,
               sizeof(int32_t), 0, 0};
    pt.d[4] = {(const uint8_t*)dc->packed_gate.ptr, 0, d_total,
               sizeof(int16_t), 0, 0};
    pt.d[5] = {(const uint8_t*)dc->dropped.ptr,
               (unsigned long long)N * K * sizeof(uint8_t), nullptr, 0, 1, 0};
    pt.d[6] = {(const uint8_t*)dc->y.ptr,
               (unsigned long long)N * D * sizeof(int64_t), nullptr, 0, 0, 0};

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        MdcProblemSpec spec = {};
        spec.abi_version = MDC_ABI_VERSION;
        spec.max_N = MDC_MAX_N;
        spec.max_D = MDC_MAX_D;
        spec.max_E = MDC_MAX_E;
        spec.max_K = MDC_MAX_K;
        spec.max_cap = MDC_MAX_CAP;
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

        std::vector<HostCase> host_cases = build_bench_cases();
        std::vector<DeviceCase*> cases;
        cases.reserve(host_cases.size());

        for (const HostCase& hc : host_cases) {
            cases.push_back(make_device_case(hc));
            std::printf(
                "bench_case %-28s N=%d D=%d E=%d K=%d cap=%d dist=%d\n",
                hc.name.c_str(),
                hc.run.N,
                hc.run.D,
                hc.run.E,
                hc.run.K,
                hc.run.cap,
                hc.run.distribution_id);
        }

        for (int warmup = 0; warmup < 5; ++warmup) {
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
                const int E = dc->host.run.E;
                const long long nk = (long long)dc->host.expert.size();
                const long long ed = (long long)dc->host.expert_out.size();
                const int rot = (int)((skey >> 8) % (uint64_t)E);
                const int blocks =
                    (int)std::min<long long>((nk + ed + 255) / 256, 4096);
                c3_mutate_mdc<<<blocks, 256, 0, stream>>>(
                    dc->expert.ptr, dc->expert_act.ptr, dc->gate.ptr,
                    dc->gate_act.ptr, dc->expert_out.ptr,
                    dc->expert_out_act.ptr, nk, ed, E, rot, skey,
                    dc->host.run.distribution_id);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(solution_run(
                    state,
                    &dc->host.run,
                    &dc->inputs,
                    &dc->outputs,
                    workspace.ptr,
                    workspace_bytes,
                    stream));
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

        const double calls = static_cast<double>(iters) * static_cast<double>(cases.size());
        const double avg_ms = static_cast<double>(elapsed_ms) / calls;

        std::printf("avg_ms=%.6f\n", avg_ms);

        // Untimed graded-output digest: one more run per case, then fold the graded
        // output ranges in fixed case order. packed_token/packed_slot/packed_gate are
        // only graded up to offsets[E] (the tail is unspecified), so fold that prefix.
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
        // C3: the digest binds EVERY timed call's graded outputs via the
        // per-(iter, case) probe accumulators.
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
            dg.dev(dc->counts.ptr, (size_t)E * sizeof(int32_t));
            dg.dev(dc->offsets.ptr, (size_t)(E + 1) * sizeof(int32_t));
            if (total_packed > 0) {
                dg.dev(dc->packed_token.ptr, (size_t)total_packed * sizeof(int32_t));
                dg.dev(dc->packed_slot.ptr, (size_t)total_packed * sizeof(int32_t));
                dg.dev(dc->packed_gate.ptr, (size_t)total_packed * sizeof(int16_t));
            }
            // The test grades dropped as (v ? 1 : 0), so fold normalized values,
            // not raw bytes (any nonzero byte is a legal "true").
            {
                const size_t total_routes = (size_t)N * (size_t)K;
                std::vector<uint8_t> h_dropped(total_routes);
                CUDA_CHECK(cudaMemcpy(h_dropped.data(), dc->dropped.ptr, total_routes,
                                      cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < total_routes; ++i) h_dropped[i] = h_dropped[i] ? 1 : 0;
                dg.bytes(h_dropped.data(), total_routes);
            }
            dg.dev(dc->y.ptr, (size_t)N * (size_t)D * sizeof(int64_t));
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
