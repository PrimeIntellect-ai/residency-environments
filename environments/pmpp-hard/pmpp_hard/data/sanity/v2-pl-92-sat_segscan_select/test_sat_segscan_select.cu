// file: test_sat_segscan_select.cu

#include "sat_segscan_select_common.h"
#include "sat_segscan_select_oracle.hpp"

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
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

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

    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost));
        }
        return host;
    }
};

template <typename T>
struct GuardedDeviceBuffer {
    uint8_t* raw = nullptr;
    T* ptr = nullptr;
    size_t count = 0;
    size_t data_bytes = 0;
    size_t total_bytes = 0;

    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

    ~GuardedDeviceBuffer() {
        if (raw) cudaFree(raw);
    }

    void allocate(size_t n) {
        count = n;
        data_bytes = sizeof(T) * count;
        total_bytes = kGuardBytes + data_bytes + kGuardBytes;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), total_bytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, total_bytes));

        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }

    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        }
        return host;
    }

    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> before(kGuardBytes);
        std::vector<uint8_t> after(kGuardBytes);

        CUDA_CHECK(cudaMemcpy(before.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (before[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "left guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }

            if (after[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "right guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }
        }

        return true;
    }
};

struct HostCase {
    std::string name;
    SssRunSpec run;
    std::vector<int32_t> v;
    std::vector<uint32_t> flags;
    int S = 0;
};

// ---------------------------------------------------------------------------
// Input generation (adversarial).
// ---------------------------------------------------------------------------

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
        case SSS_DIST_EXACT:
            *lo = -4096;
            *hi = 4096;
            break;
        case SSS_DIST_ONESEG:
            *lo = -100;
            *hi = 100;
            break;
        case SSS_DIST_ALLSEG:
            *lo = -rng.uniform_int(1, 1 << 20);
            *hi = rng.uniform_int(1, 1 << 20);
            break;
        case SSS_DIST_RAGGED:
            *lo = -500;
            *hi = 500;
            break;
        case SSS_DIST_ZERO:
            *lo = -1;
            *hi = 1;
            break;
        case SSS_DIST_RANDBITS:
        default:
            *lo = -(1 + (int)(rng.next_u64() % (1u << 24)));
            *hi = 1 + (int)(rng.next_u64() % (1u << 24));
            break;
    }
}

// mean_seg == 0 means "use the distribution's own flag rule".
static void gen_flags(
    SplitMix64& rng, int dist, int N, int mean_seg,
    std::vector<uint32_t>* flags) {
    const int Nw = sss_Nw(N);
    flags->assign((size_t)Nw, 0u);

    auto set_bit = [&](int i) {
        (*flags)[(size_t)(i >> 5)] |= (1u << (i & 31));
    };

    if (mean_seg > 0) {
        for (int i = 0; i < N; ++i) {
            if ((int)(rng.next_u64() % (uint64_t)mean_seg) == 0) set_bit(i);
        }
    } else {
        switch (dist) {
            case SSS_DIST_ONESEG:
                break;  // only the forced bit 0
            case SSS_DIST_ALLSEG:
                for (int w = 0; w < Nw; ++w) (*flags)[(size_t)w] = 0xffffffffu;
                // clear padding bits
                if (N & 31) {
                    (*flags)[(size_t)(Nw - 1)] =
                        0xffffffffu >> (32 - (N & 31));
                }
                break;
            case SSS_DIST_RAGGED: {
                int i = 0;
                while (i < N) {
                    set_bit(i);
                    const int shift = rng.uniform_int(0, 13);
                    const int len = 1 + (int)(rng.next_u64() %
                                              (uint64_t)(1 << shift));
                    i += len;
                }
                break;
            }
            default: {
                int dens;
                switch (dist) {
                    case SSS_DIST_SMOOTH: dens = 1024; break;
                    case SSS_DIST_TIGHT: dens = 256; break;
                    case SSS_DIST_SATRUN: dens = 512; break;
                    case SSS_DIST_EXACT: dens = 128; break;
                    case SSS_DIST_ZERO: dens = 64; break;
                    default: dens = 32; break;  // RANDBITS
                }
                for (int i = 0; i < N; ++i) {
                    if ((int)(rng.next_u64() % (uint64_t)dens) == 0) set_bit(i);
                }
                break;
            }
        }
    }

    set_bit(0);  // contract guarantee
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
        case SSS_DIST_EXACT: {
            const int sel = rng.uniform_int(0, 6);
            switch (sel) {
                case 0: return hi;
                case 1: return lo;
                case 2: return -hi;
                case 3: return -lo;
                case 4: return 1;
                case 5: return -1;
                default: return 0;
            }
        }
        case SSS_DIST_ONESEG:
            return rng.uniform_int(-150, 150);
        case SSS_DIST_ALLSEG:
            return rng.uniform_int(-SSS_MAX_ABS_V, SSS_MAX_ABS_V);
        case SSS_DIST_RAGGED:
            return rng.uniform_int(-1000, 1000);
        case SSS_DIST_ZERO: {
            const int sel = rng.uniform_int(0, 9);
            if (sel < 7) return 0;
            return (sel & 1) ? 1 : -1;
        }
        case SSS_DIST_RANDBITS:
        default:
            return rng.uniform_int(-SSS_MAX_ABS_V, SSS_MAX_ABS_V);
    }
}

static HostCase make_case(
    const std::string& name,
    int N,
    int distribution_id,
    uint64_t seed,
    int mean_seg = 0) {
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

    gen_flags(rng, distribution_id, N, mean_seg, &hc.flags);

    hc.v.resize((size_t)N);
    for (int i = 0; i < N; ++i) {
        hc.v[(size_t)i] = gen_value(rng, distribution_id, lo, hi);
    }

    int S = 0;
    for (uint32_t w : hc.flags) S += __builtin_popcount(w);
    hc.S = S;

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    static const int shapes[6] = {
        65537, 131071, 262144, 500000, 1048577, 2097151
    };
    static const char* dist_names[9] = {
        "smooth", "tight", "satrun", "exact", "oneseg",
        "allseg", "ragged", "zero", "randbits"
    };

    std::vector<HostCase> cases;
    uint64_t s = 0xD20000001b3ULL;

    for (int d = 0; d < 9; ++d) {
        for (int sh = 0; sh < 6; ++sh) {
            std::ostringstream oss;
            oss << dist_names[d] << "_N" << shapes[sh];
            cases.push_back(make_case(oss.str(), shapes[sh], d, s++));
        }
    }

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<int32_t>& d_v,
    const DeviceBuffer<uint32_t>& d_f,
    std::string* error) {
    const std::vector<int32_t> v = d_v.download();
    const std::vector<uint32_t> f = d_f.download();

    if (v.size() != hc.v.size() ||
        std::memcmp(v.data(), hc.v.data(), v.size() * 4) != 0) {
        if (error) *error = "input v modified";
        return false;
    }
    if (f.size() != hc.flags.size() ||
        std::memcmp(f.data(), hc.flags.data(), f.size() * 4) != 0) {
        if (error) *error = "input flags modified";
        return false;
    }
    return true;
}

static bool run_one_case(
    const HostCase& hc,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    std::string* error) {
    const int N = hc.run.N;
    const int Nw = sss_Nw(N);

    DeviceBuffer<int32_t> d_v;
    DeviceBuffer<uint32_t> d_f;
    d_v.allocate(hc.v.size());
    d_f.allocate(hc.flags.size());
    d_v.upload(hc.v);
    d_f.upload(hc.flags);

    GuardedDeviceBuffer<int32_t> d_y;
    GuardedDeviceBuffer<uint32_t> d_sb;
    GuardedDeviceBuffer<int32_t> d_si;
    GuardedDeviceBuffer<int32_t> d_sc;
    GuardedDeviceBuffer<int32_t> d_sl;

    d_y.allocate((size_t)N);
    d_sb.allocate((size_t)Nw);
    d_si.allocate((size_t)N);
    d_sc.allocate(1);
    d_sl.allocate((size_t)hc.S);

    SssInputs inputs = {};
    inputs.v = d_v.ptr;
    inputs.flags = d_f.ptr;

    SssOutputs outputs = {};
    outputs.y = d_y.ptr;
    outputs.sat_bits = d_sb.ptr;
    outputs.sel_idx = d_si.ptr;
    outputs.sel_count = d_sc.ptr;
    outputs.seg_last = d_sl.ptr;

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(solution_run(
        state,
        &hc.run,
        &inputs,
        &outputs,
        workspace,
        workspace_bytes,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(hc, d_v, d_f, error)) return false;

    if (!d_y.check_guards("y", error)) return false;
    if (!d_sb.check_guards("sat_bits", error)) return false;
    if (!d_si.check_guards("sel_idx", error)) return false;
    if (!d_sc.check_guards("sel_count", error)) return false;
    if (!d_sl.check_guards("seg_last", error)) return false;

    const std::vector<int32_t> h_y = d_y.download_data();
    const std::vector<uint32_t> h_sb = d_sb.download_data();
    const std::vector<int32_t> h_si = d_si.download_data();
    const std::vector<int32_t> h_sc = d_sc.download_data();
    const std::vector<int32_t> h_sl = d_sl.download_data();

    SssHostInputsView host_inputs = {};
    host_inputs.v = hc.v.data();
    host_inputs.flags = hc.flags.data();

    SssExpected expected;
    sss_cpu_oracle(hc.run, host_inputs, &expected);

    if ((int)expected.seg_last.size() != hc.S) {
        if (error) *error = "internal: S mismatch between host popcount and oracle";
        return false;
    }

    SssHostOutputsView got = {};
    got.y = h_y.data();
    got.sat_bits = h_sb.data();
    got.sel_idx = h_si.data();
    got.sel_count = h_sc.data();
    got.seg_last = h_sl.data();

    return sss_check_all_outputs(hc.run, expected, got, error);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int max_N = SSS_MIN_N;
        for (const HostCase& hc : cases) {
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

        int passed = 0;
        const int total = static_cast<int>(cases.size());

        for (const HostCase& hc : cases) {
            std::string error;
            bool ok = false;

            try {
                ok = run_one_case(
                    hc,
                    state,
                    workspace.ptr,
                    workspace_bytes,
                    stream,
                    &error);
            } catch (const std::exception& ex) {
                error = ex.what();
                ok = false;
            }

            if (ok) {
                ++passed;
                std::printf(
                    "case %-24s PASS  N=%d lo=%d hi=%d S=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.N,
                    hc.run.lo,
                    hc.run.hi,
                    hc.S,
                    hc.run.distribution_id);
            } else {
                std::printf("case %-24s FAIL  %s\n", hc.name.c_str(), error.c_str());
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));

        std::printf("passed %d / %d\n", passed, total);
        return passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
