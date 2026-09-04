// ============================================================================
// file: test_moe_grouped_ffn_reroute.cu
// ============================================================================

#include "moe_grouped_ffn_reroute_common.h"
#include "moe_grouped_ffn_reroute_oracle.hpp"

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

    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * count,
                                  cudaMemcpyDeviceToHost));
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

    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

    ~GuardedDeviceBuffer() {
        if (raw) cudaFree(raw);
    }

    void allocate(size_t n) {
        count = n;
        data_bytes = sizeof(T) * count;
        const size_t total = kGuardBytes + data_bytes + kGuardBytes;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), total));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, total));
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }

    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes,
                                  cudaMemcpyDeviceToHost));
        }
        return host;
    }

    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> guard(kGuardBytes);

        CUDA_CHECK(cudaMemcpy(guard.data(), raw, kGuardBytes,
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (guard[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "left guard corrupted for " << name
                        << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }
        }

        CUDA_CHECK(cudaMemcpy(guard.data(), raw + kGuardBytes + data_bytes,
                              kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (guard[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "right guard corrupted for " << name
                        << " at byte " << i;
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
    MgfRunSpec run;
    bool replay = false;
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

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    struct Combo {
        int N, D, H, E, G, g_sel, K, cap, qshift;
    };
    const Combo combos[8] = {
        {4096, 32, 64, 16, 4, 2, 2, 32, 6},
        {4096, 64, 128, 32, 8, 2, 4, 64, 5},
        {5000, 32, 128, 64, 16, 4, 2, 16, 4},
        {8192, 128, 64, 32, 4, 1, 4, 128, 7},
        {8192, 64, 64, 128, 16, 4, 4, 32, 8},
        {16384, 32, 256, 128, 16, 2, 4, 64, 6},
        {16384, 128, 128, 64, 8, 4, 2, 256, 5},
        {32768, 128, 256, 128, 8, 2, 4, 256, 6},
    };
    const char* dist_names[5] = {
        "uniform", "hot_expert", "ties", "zero_x", "saturate"};

    // Full sweep over the first 7 combos x 5 distributions.
    for (int d = 0; d < 5; ++d) {
        for (int c = 0; c < 7; ++c) {
            const Combo& cb = combos[c];
            char name[128];
            std::snprintf(name, sizeof(name), "%s_c%d_N%d_E%d_K%d_cap%d",
                          dist_names[d], c, cb.N, cb.E, cb.K, cb.cap);
            cases.push_back(make_random_case(
                name, cb.N, cb.D, cb.H, cb.E, cb.G, cb.g_sel, cb.K, cb.cap,
                cb.qshift, d, s++));
        }
    }

    // Max-shape combo for two distributions (bounded oracle time).
    for (int d = 0; d < 2; ++d) {
        const Combo& cb = combos[7];
        char name[128];
        std::snprintf(name, sizeof(name), "%s_cmax_N%d_E%d_cap%d",
                      dist_names[d], cb.N, cb.E, cb.cap);
        cases.push_back(make_random_case(
            name, cb.N, cb.D, cb.H, cb.E, cb.G, cb.g_sel, cb.K, cb.cap,
            cb.qshift, d, s++));
    }

    // Seed-shifted repeats of the first 3 combos.
    for (int d = 0; d < 5; ++d) {
        for (int c = 0; c < 3; ++c) {
            const Combo& cb = combos[c];
            char name[128];
            std::snprintf(name, sizeof(name), "%s_c%d_seed2", dist_names[d], c);
            cases.push_back(make_random_case(
                name, cb.N, cb.D, cb.H, cb.E, cb.G, cb.g_sel, cb.K, cb.cap,
                cb.qshift, d, s++));
        }
    }

    // Edge: g_sel * (E/G) == K -> no backups exist at all.
    cases.push_back(make_random_case(
        "edge_no_backup_S2_gsel1_K2", 4096, 32, 64, 16, 8, 1, 2, 16, 6,
        MGF_DIST_HOT_EXPERT, s++));

    // Edge: everything zero -> total tie cascade, y == 0 everywhere.
    {
        HostCase hc = make_random_case(
            "edge_all_zero_inputs", 4096, 32, 64, 16, 4, 2, 2, 16, 4,
            MGF_DIST_UNIFORM, s++);
        std::fill(hc.x.begin(), hc.x.end(), 0);
        std::fill(hc.wr.begin(), hc.wr.end(), 0);
        std::fill(hc.w1.begin(), hc.w1.end(), 0);
        std::fill(hc.w2.begin(), hc.w2.end(), 0);
        cases.push_back(std::move(hc));
    }

    // Edge: 2K candidates always available + tiny caps -> mass re-routing.
    cases.push_back(make_random_case(
        "edge_mass_reroute_tinycap", 8192, 32, 64, 16, 4, 2, 4, 16, 6,
        MGF_DIST_HOT_EXPERT, s++));

    // Determinism replay markers.
    cases[0].replay = true;
    cases[20].replay = true;
    cases[40].replay = true;

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<int8_t>& d_x,
    const DeviceBuffer<int8_t>& d_wr,
    const DeviceBuffer<int8_t>& d_w1,
    const DeviceBuffer<int8_t>& d_w2,
    std::string* error) {
    if (d_x.download() != hc.x) {
        if (error) *error = "input x modified";
        return false;
    }
    if (d_wr.download() != hc.wr) {
        if (error) *error = "input wr modified";
        return false;
    }
    if (d_w1.download() != hc.w1) {
        if (error) *error = "input w1 modified";
        return false;
    }
    if (d_w2.download() != hc.w2) {
        if (error) *error = "input w2 modified";
        return false;
    }
    return true;
}

struct CaseBuffers {
    DeviceBuffer<int8_t> x, wr, w1, w2;
    GuardedDeviceBuffer<int32_t> logits;
    GuardedDeviceBuffer<int32_t> counts;
    GuardedDeviceBuffer<int32_t> offsets;
    GuardedDeviceBuffer<int32_t> packed_token;
    GuardedDeviceBuffer<int32_t> packed_slot;
    GuardedDeviceBuffer<int32_t> packed_gate;
    GuardedDeviceBuffer<uint8_t> packed_phase;
    GuardedDeviceBuffer<int16_t> route_expert;
    GuardedDeviceBuffer<uint8_t> route_status;
    GuardedDeviceBuffer<int32_t> packed_y;
    GuardedDeviceBuffer<int64_t> y;
    GuardedDeviceBuffer<uint64_t> y_checksum;
    MgfInputs inputs = {};
    MgfOutputs outputs = {};
};

static void setup_case_buffers(const HostCase& hc, CaseBuffers* cb) {
    const int N = hc.run.N;
    const int D = hc.run.D;
    const int E = hc.run.E;
    const int K = hc.run.K;
    const size_t packed_capacity = (size_t)E * hc.run.cap;

    cb->x.allocate(hc.x.size());
    cb->wr.allocate(hc.wr.size());
    cb->w1.allocate(hc.w1.size());
    cb->w2.allocate(hc.w2.size());
    cb->x.upload(hc.x);
    cb->wr.upload(hc.wr);
    cb->w1.upload(hc.w1);
    cb->w2.upload(hc.w2);

    cb->logits.allocate((size_t)N * E);
    cb->counts.allocate((size_t)E);
    cb->offsets.allocate((size_t)E + 1);
    cb->packed_token.allocate(packed_capacity);
    cb->packed_slot.allocate(packed_capacity);
    cb->packed_gate.allocate(packed_capacity);
    cb->packed_phase.allocate(packed_capacity);
    cb->route_expert.allocate((size_t)N * K);
    cb->route_status.allocate((size_t)N * K);
    cb->packed_y.allocate(packed_capacity * (size_t)D);
    cb->y.allocate((size_t)N * D);
    cb->y_checksum.allocate(1);

    cb->inputs.x = cb->x.ptr;
    cb->inputs.wr = cb->wr.ptr;
    cb->inputs.w1 = cb->w1.ptr;
    cb->inputs.w2 = cb->w2.ptr;

    cb->outputs.logits = cb->logits.ptr;
    cb->outputs.counts = cb->counts.ptr;
    cb->outputs.offsets = cb->offsets.ptr;
    cb->outputs.packed_token = cb->packed_token.ptr;
    cb->outputs.packed_slot = cb->packed_slot.ptr;
    cb->outputs.packed_gate = cb->packed_gate.ptr;
    cb->outputs.packed_phase = cb->packed_phase.ptr;
    cb->outputs.route_expert = cb->route_expert.ptr;
    cb->outputs.route_status = cb->route_status.ptr;
    cb->outputs.packed_y = cb->packed_y.ptr;
    cb->outputs.y = cb->y.ptr;
    cb->outputs.y_checksum = cb->y_checksum.ptr;
}

static bool verify_outputs(
    const HostCase& hc,
    const MgfExpected& expected,
    const CaseBuffers& cb,
    std::string* error) {
    if (!cb.logits.check_guards("logits", error)) return false;
    if (!cb.counts.check_guards("counts", error)) return false;
    if (!cb.offsets.check_guards("offsets", error)) return false;
    if (!cb.packed_token.check_guards("packed_token", error)) return false;
    if (!cb.packed_slot.check_guards("packed_slot", error)) return false;
    if (!cb.packed_gate.check_guards("packed_gate", error)) return false;
    if (!cb.packed_phase.check_guards("packed_phase", error)) return false;
    if (!cb.route_expert.check_guards("route_expert", error)) return false;
    if (!cb.route_status.check_guards("route_status", error)) return false;
    if (!cb.packed_y.check_guards("packed_y", error)) return false;
    if (!cb.y.check_guards("y", error)) return false;
    if (!cb.y_checksum.check_guards("y_checksum", error)) return false;

    const std::vector<int32_t> h_logits = cb.logits.download_data();
    const std::vector<int32_t> h_counts = cb.counts.download_data();
    const std::vector<int32_t> h_offsets = cb.offsets.download_data();
    const std::vector<int32_t> h_ptok = cb.packed_token.download_data();
    const std::vector<int32_t> h_pslot = cb.packed_slot.download_data();
    const std::vector<int32_t> h_pgate = cb.packed_gate.download_data();
    const std::vector<uint8_t> h_pphase = cb.packed_phase.download_data();
    const std::vector<int16_t> h_rexp = cb.route_expert.download_data();
    const std::vector<uint8_t> h_rstat = cb.route_status.download_data();
    const std::vector<int32_t> h_py = cb.packed_y.download_data();
    const std::vector<int64_t> h_y = cb.y.download_data();
    const std::vector<uint64_t> h_csum = cb.y_checksum.download_data();

    MgfHostOutputsView got = {};
    got.logits = h_logits.data();
    got.counts = h_counts.data();
    got.offsets = h_offsets.data();
    got.packed_token = h_ptok.data();
    got.packed_slot = h_pslot.data();
    got.packed_gate = h_pgate.data();
    got.packed_phase = h_pphase.data();
    got.route_expert = h_rexp.data();
    got.route_status = h_rstat.data();
    got.packed_y = h_py.data();
    got.y = h_y.data();
    got.y_checksum = h_csum.data();

    return mgf_check_all_outputs(hc.run, expected, got, error);
}

static bool run_one_case(
    const HostCase& hc,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    std::string* error) {
    CaseBuffers cb;
    setup_case_buffers(hc, &cb);

    MgfHostInputsView host_inputs = {};
    host_inputs.x = hc.x.data();
    host_inputs.wr = hc.wr.data();
    host_inputs.w1 = hc.w1.data();
    host_inputs.w2 = hc.w2.data();

    MgfExpected expected;
    mgf_cpu_oracle(hc.run, host_inputs, &expected);

    if ((size_t)expected.offsets[(size_t)hc.run.E] >
        (size_t)hc.run.E * hc.run.cap) {
        if (error) *error = "oracle total kept routes exceeds E*cap";
        return false;
    }

    const int rounds = hc.replay ? 2 : 1;
    for (int rd = 0; rd < rounds; ++rd) {
        CUDA_CHECK(solution_reset(state, stream));
        CUDA_CHECK(solution_run(
            state, &hc.run, &cb.inputs, &cb.outputs, workspace,
            workspace_bytes, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (!check_input_unchanged(hc, cb.x, cb.wr, cb.w1, cb.w2, error)) {
            return false;
        }
        if (!verify_outputs(hc, expected, cb, error)) {
            if (error && rd == 1) *error += " (replay round)";
            return false;
        }
    }

    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

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

        const std::vector<HostCase> cases = build_test_cases();

        int passed = 0;
        const int total = static_cast<int>(cases.size());

        for (const HostCase& hc : cases) {
            std::string error;
            bool ok = false;
            try {
                ok = run_one_case(hc, state, workspace.ptr, workspace_bytes,
                                  stream, &error);
            } catch (const std::exception& ex) {
                error = ex.what();
                ok = false;
            }

            if (ok) {
                ++passed;
                std::printf(
                    "case %-44s PASS  N=%d D=%d H=%d E=%d G=%d g=%d K=%d "
                    "cap=%d q=%d dist=%d%s\n",
                    hc.name.c_str(), hc.run.N, hc.run.D, hc.run.H, hc.run.E,
                    hc.run.G, hc.run.g_sel, hc.run.K, hc.run.cap,
                    hc.run.qshift, hc.run.distribution_id,
                    hc.replay ? " (replayed)" : "");
            } else {
                std::printf("case %-44s FAIL  %s\n", hc.name.c_str(),
                            error.c_str());
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
