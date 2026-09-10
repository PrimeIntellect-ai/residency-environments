// ============================================================================
// file: test_mk_priority_preempt.cu
// ============================================================================

#include "mk_priority_preempt_common.h"
#include "mk_priority_preempt_oracle.hpp"

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
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    int uniform_int(int lo, int hi) {
        return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1));
    }
    double uniform01() {
        return (double)(next_u64() >> 11) * 0x1.0p-53;
    }
    bool chance(double p) { return uniform01() < p; }
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
        for (int side = 0; side < 2; ++side) {
            CUDA_CHECK(cudaMemcpy(
                guard.data(),
                side == 0 ? raw : raw + kGuardBytes + data_bytes, kGuardBytes,
                cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < kGuardBytes; ++i) {
                if (guard[i] != kGuardByte) {
                    if (error) {
                        std::ostringstream oss;
                        oss << (side == 0 ? "left" : "right")
                            << " guard corrupted for " << name << " at byte "
                            << i;
                        *error = oss.str();
                    }
                    return false;
                }
            }
        }
        return true;
    }
};

struct RoundSpec {
    MkpRunSpec run;
    int lenmax;
    bool reset_before = false;
    uint64_t seed;
};

struct Scenario {
    std::string name;
    int K, Q;
    bool replay = false;
    std::vector<RoundSpec> rounds;
};

struct RoundData {
    std::vector<int8_t> X;
    std::vector<int32_t> jarrival, jclass, jlen;
    std::vector<int8_t> jw;
};

static void gen_round_data(const RoundSpec& rs, RoundData* rd) {
    const MkpRunSpec& run = rs.run;
    SplitMix64 rng(g_state ^ rs.seed);
    rd->X.resize((size_t)run.M * run.K);
    for (size_t i = 0; i < rd->X.size(); ++i) {
        rd->X[i] = run.distribution_id == MKP_DIST_ZERO_X
            ? 0
            : (int8_t)rng.uniform_int(-127, 127);
    }
    rd->jarrival.resize((size_t)run.njobs);
    rd->jclass.resize((size_t)run.njobs);
    rd->jlen.resize((size_t)run.njobs);
    rd->jw.resize((size_t)run.njobs * run.K);
    const int Q = run.Q;
    const int R = run.R;
    const int lm = rs.lenmax;
    for (int j = 0; j < run.njobs; ++j) {
        int cls, arr, len;
        switch (run.distribution_id) {
            case MKP_DIST_STAGGER:
                if (rng.chance(0.4)) {
                    cls = 0;
                    arr = rng.uniform_int(R / 2, R - 1);
                    len = rng.uniform_int(1, lm / 2 > 0 ? lm / 2 : 1);
                } else {
                    cls = rng.uniform_int(1, Q - 1);
                    arr = rng.uniform_int(0, R / 4);
                    len = rng.uniform_int(lm / 2 > 0 ? lm / 2 : 1, lm);
                }
                break;
            case MKP_DIST_LONGTAIL:
                if (j < run.njobs / 8) {
                    cls = Q - 1;
                    arr = 0;
                    len = lm;
                } else {
                    cls = rng.uniform_int(0, Q >= 3 ? Q - 3 : 0);
                    arr = rng.uniform_int(0, R - 1);
                    len = rng.uniform_int(1, 4);
                }
                break;
            case MKP_DIST_BURST:
                cls = j % Q;
                arr = (j % 2) == 0 ? 0 : R / 2;
                len = rng.uniform_int(1, lm);
                break;
            default:  // UNIFORM, ZERO_X
                cls = rng.uniform_int(0, Q - 1);
                arr = rng.uniform_int(0, R - 1);
                len = rng.uniform_int(1, lm);
                break;
        }
        rd->jarrival[(size_t)j] = arr;
        rd->jclass[(size_t)j] = cls;
        rd->jlen[(size_t)j] = len;
    }
    for (size_t i = 0; i < rd->jw.size(); ++i) {
        rd->jw[i] = (int8_t)rng.uniform_int(-127, 127);
    }
}

static RoundSpec make_round(
    int M, int K, int Q, int R, int quantum, int njobs, int lenmax,
    int dist, uint64_t seed, bool reset_before = false) {
    RoundSpec rs;
    rs.run = {};
    rs.run.abi_version = MKP_ABI_VERSION;
    rs.run.M = M;
    rs.run.K = K;
    rs.run.Q = Q;
    rs.run.R = R;
    rs.run.quantum = quantum;
    rs.run.njobs = njobs;
    rs.run.seed_id = (int32_t)(seed & 0x7fffffff);
    rs.run.distribution_id = dist;
    rs.run.case_id = (int32_t)((seed >> 32) & 0x7fffffff);
    rs.lenmax = lenmax;
    rs.reset_before = reset_before;
    rs.seed = seed;
    if (!mkp_validate_run_spec(&rs.run)) {
        throw std::runtime_error("invalid run spec in make_round");
    }
    return rs;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> scenarios;
    uint64_t s = 0x500000001b3ULL;

    struct Combo {
        int M, K, Q, R, quantum, njobs, lenmax;
    };
    const Combo combos[7] = {
        {16384, 128, 2, 8, 2, 64, 32},
        {16384, 128, 3, 16, 4, 128, 24},
        {16640, 256, 4, 12, 1, 96, 16},
        {24576, 128, 3, 24, 8, 256, 24},
        {32768, 256, 2, 16, 2, 128, 12},
        {20480, 128, 4, 32, 4, 192, 16},
        {32768, 256, 4, 8, 8, 256, 8},
    };
    const char* dist_names[5] = {
        "uniform", "stagger", "longtail", "burst", "zero_x"};

    auto standard = [&](const char* name, const Combo& cb, int dist,
                        uint64_t seed, int nruns) {
        Scenario sc;
        sc.name = name;
        sc.K = cb.K;
        sc.Q = cb.Q;
        for (int q = 0; q < nruns; ++q) {
            // M / R / quantum / njobs drift between runs of a scenario.
            const int M = cb.M;
            const int R = q == 1 ? (cb.R > 8 ? cb.R - 4 : cb.R) : cb.R;
            const int quantum =
                q == 2 ? (cb.quantum < 8 ? cb.quantum * 2 : 4) : cb.quantum;
            sc.rounds.push_back(make_round(
                M, cb.K, cb.Q, R, quantum, cb.njobs, cb.lenmax, dist,
                seed * 1000003ULL + (uint64_t)q));
        }
        return sc;
    };

    for (int d = 0; d < 5; ++d) {
        for (int c = 0; c < 7; ++c) {
            char name[128];
            std::snprintf(name, sizeof(name), "%s_c%d_M%d_K%d_Q%d_q%d",
                          dist_names[d], c, combos[c].M, combos[c].K,
                          combos[c].Q, combos[c].quantum);
            scenarios.push_back(standard(name, combos[c], d, s++, 3));
        }
    }

    // Max shape, 2 runs.
    const Combo maxcb = {32768, 256, 4, 32, 4, 1024, 12};
    for (int d = 0; d < 2; ++d) {
        char name[128];
        std::snprintf(name, sizeof(name), "%s_max_M32768_n1024",
                      dist_names[d == 0 ? 0 : 2]);
        scenarios.push_back(standard(
            name, maxcb, d == 0 ? MKP_DIST_UNIFORM : MKP_DIST_LONGTAIL, s++,
            2));
    }

    // Seed-shifted repeats of the first 3 combos.
    for (int d = 0; d < 5; ++d) {
        for (int c = 0; c < 3; ++c) {
            char name[128];
            std::snprintf(name, sizeof(name), "%s_c%d_seed2",
                          dist_names[d], c);
            scenarios.push_back(standard(name, combos[c], d, s++, 3));
        }
    }

    // Edges.
    {
        Scenario sc;
        sc.name = "edge_starve_longtail";
        sc.K = 128;
        sc.Q = 4;
        for (int q = 0; q < 3; ++q) {
            sc.rounds.push_back(make_round(
                16384, 128, 4, 16, 2, 256, 64, MKP_DIST_LONGTAIL, s++));
        }
        scenarios.push_back(std::move(sc));
    }
    {
        Scenario sc;
        sc.name = "edge_quantum1_thrash";
        sc.K = 128;
        sc.Q = 2;
        for (int q = 0; q < 2; ++q) {
            sc.rounds.push_back(make_round(
                16384, 128, 2, 32, 1, 512, 8, MKP_DIST_UNIFORM, s++));
        }
        scenarios.push_back(std::move(sc));
    }
    {
        // All arrivals in the last round -> R-1 empty rounds first.
        Scenario sc;
        sc.name = "edge_empty_rounds";
        sc.K = 128;
        sc.Q = 2;
        RoundSpec rs = make_round(
            16384, 128, 2, 16, 4, 64, 16, MKP_DIST_UNIFORM, s++);
        sc.rounds.push_back(rs);
        scenarios.push_back(std::move(sc));
        // (arrivals forced below in gen? handled by special-casing name)
    }
    {
        Scenario sc;
        sc.name = "edge_single_class";
        sc.K = 128;
        sc.Q = 2;
        for (int q = 0; q < 2; ++q) {
            sc.rounds.push_back(make_round(
                16384, 128, 2, 8, 4, 128, 16, MKP_DIST_UNIFORM, s++));
        }
        scenarios.push_back(std::move(sc));
    }
    {
        Scenario sc;
        sc.name = "edge_mid_reset";
        sc.K = 128;
        sc.Q = 3;
        sc.rounds.push_back(make_round(
            16384, 128, 3, 16, 4, 128, 24, MKP_DIST_UNIFORM, s++));
        sc.rounds.push_back(make_round(
            16384, 128, 3, 16, 2, 128, 24, MKP_DIST_BURST, s++));
        sc.rounds.push_back(make_round(
            16384, 128, 3, 8, 8, 64, 16, MKP_DIST_UNIFORM, s++,
            /*reset_before=*/true));
        scenarios.push_back(std::move(sc));
    }
    {
        // Everything drains: queues end empty.
        Scenario sc;
        sc.name = "edge_finish_all";
        sc.K = 128;
        sc.Q = 2;
        sc.rounds.push_back(make_round(
            16384, 128, 2, 32, 8, 64, 4, MKP_DIST_BURST, s++));
        scenarios.push_back(std::move(sc));
    }
    {
        // Carryover jobs execute against different X / M each run.
        Scenario sc;
        sc.name = "edge_carry_chain";
        sc.K = 256;
        sc.Q = 3;
        sc.rounds.push_back(make_round(
            16384, 256, 3, 8, 1, 256, 32, MKP_DIST_UNIFORM, s++));
        sc.rounds.push_back(make_round(
            24576, 256, 3, 8, 2, 128, 16, MKP_DIST_STAGGER, s++));
        sc.rounds.push_back(make_round(
            32768, 256, 3, 16, 4, 64, 8, MKP_DIST_UNIFORM, s++));
        scenarios.push_back(std::move(sc));
    }
    {
        Scenario sc;
        sc.name = "edge_min";
        sc.K = 128;
        sc.Q = 2;
        sc.rounds.push_back(make_round(
            16384, 128, 2, 8, 1, 64, 8, MKP_DIST_ZERO_X, s++));
        scenarios.push_back(std::move(sc));
    }

    scenarios[0].replay = true;
    scenarios[35].replay = true;  // uniform max
    scenarios[58].replay = true;  // edge_carry_chain

    return scenarios;
}

// Force the edge_empty_rounds / edge_single_class special job patterns.
static void apply_special_pattern(
    const Scenario& sc, const RoundSpec& rs, RoundData* rd) {
    if (sc.name == "edge_empty_rounds") {
        for (int j = 0; j < rs.run.njobs; ++j) {
            rd->jarrival[(size_t)j] = rs.run.R - 1;
        }
    } else if (sc.name == "edge_single_class") {
        for (int j = 0; j < rs.run.njobs; ++j) {
            rd->jclass[(size_t)j] = rs.run.Q - 1;
        }
    }
}

static bool run_scenario(
    const Scenario& sc,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    std::string* error) {
    const size_t trace_cap = (size_t)MKP_MAX_R * MKP_MAX_JOBS_TOTAL;

    DeviceBuffer<int8_t> dX, djw;
    DeviceBuffer<int32_t> darr, dcls, dlen;
    dX.allocate((size_t)MKP_MAX_M * sc.K);
    djw.allocate((size_t)MKP_MAX_NJOBS * sc.K);
    darr.allocate(MKP_MAX_NJOBS);
    dcls.allocate(MKP_MAX_NJOBS);
    dlen.allocate(MKP_MAX_NJOBS);

    GuardedDeviceBuffer<int32_t> trace_len;
    GuardedDeviceBuffer<uint64_t> trace, exec_digest, y, queue_dump,
        state_checksum;
    GuardedDeviceBuffer<int32_t> queue_len;
    trace_len.allocate(1);
    trace.allocate(trace_cap);
    exec_digest.allocate(trace_cap);
    y.allocate(MKP_MAX_M);
    queue_len.allocate((size_t)sc.Q);
    queue_dump.allocate(MKP_MAX_JOBS_TOTAL);
    state_checksum.allocate(1);

    // Oracle pass 0 computes and stores expectations; pass 1 (replay)
    // regenerates identical inputs and reuses them.
    std::vector<MkpExpected> expected(sc.rounds.size());
    const int passes = sc.replay ? 2 : 1;

    for (int pass = 0; pass < passes; ++pass) {
        CUDA_CHECK(solution_reset(state, stream));
        MkpOracle oracle;
        oracle.reset(sc.K, sc.Q);

        for (size_t ri = 0; ri < sc.rounds.size(); ++ri) {
            const RoundSpec& rs = sc.rounds[ri];
            if (ri > 0 && rs.reset_before) {
                CUDA_CHECK(solution_reset(state, stream));
            }

            RoundData rd;
            gen_round_data(rs, &rd);
            apply_special_pattern(sc, rs, &rd);

            if (pass == 0) {
                if (ri > 0 && rs.reset_before) oracle.reset(sc.K, sc.Q);
                oracle.run_round(rs.run, rd.X.data(), rd.jarrival.data(),
                                 rd.jclass.data(), rd.jlen.data(),
                                 rd.jw.data(), &expected[ri]);
            }

            CUDA_CHECK(cudaMemcpy(dX.ptr, rd.X.data(), rd.X.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(darr.ptr, rd.jarrival.data(),
                                  rd.jarrival.size() * 4,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dcls.ptr, rd.jclass.data(),
                                  rd.jclass.size() * 4,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dlen.ptr, rd.jlen.data(),
                                  rd.jlen.size() * 4,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(djw.ptr, rd.jw.data(), rd.jw.size(),
                                  cudaMemcpyHostToDevice));

            MkpInputs inputs = {};
            inputs.X = dX.ptr;
            inputs.jarrival = darr.ptr;
            inputs.jclass = dcls.ptr;
            inputs.jlen = dlen.ptr;
            inputs.jw = djw.ptr;

            MkpOutputs outputs = {};
            outputs.trace_len = trace_len.ptr;
            outputs.trace = trace.ptr;
            outputs.exec_digest = exec_digest.ptr;
            outputs.y = y.ptr;
            outputs.queue_len = queue_len.ptr;
            outputs.queue_dump = queue_dump.ptr;
            outputs.state_checksum = state_checksum.ptr;

            CUDA_CHECK(solution_run(state, &rs.run, &inputs, &outputs,
                                    workspace, workspace_bytes, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            std::ostringstream where;
            where << "pass " << pass << " round " << ri << ": ";

            std::string sub;
            if (!trace_len.check_guards("trace_len", &sub) ||
                !trace.check_guards("trace", &sub) ||
                !exec_digest.check_guards("exec_digest", &sub) ||
                !y.check_guards("y", &sub) ||
                !queue_len.check_guards("queue_len", &sub) ||
                !queue_dump.check_guards("queue_dump", &sub) ||
                !state_checksum.check_guards("state_checksum", &sub)) {
                if (error) *error = where.str() + sub;
                return false;
            }

            // Input immutability (X + weights).
            {
                std::vector<int8_t> h(rd.X.size());
                CUDA_CHECK(cudaMemcpy(h.data(), dX.ptr, h.size(),
                                      cudaMemcpyDeviceToHost));
                if (h != rd.X) {
                    if (error) *error = where.str() + "input X modified";
                    return false;
                }
                h.resize(rd.jw.size());
                CUDA_CHECK(cudaMemcpy(h.data(), djw.ptr, h.size(),
                                      cudaMemcpyDeviceToHost));
                if (h != rd.jw) {
                    if (error) *error = where.str() + "input jw modified";
                    return false;
                }
            }

            const std::vector<int32_t> h_tl = trace_len.download_data();
            const std::vector<uint64_t> h_tr = trace.download_data();
            const std::vector<uint64_t> h_ed = exec_digest.download_data();
            const std::vector<uint64_t> h_y = y.download_data();
            const std::vector<int32_t> h_ql = queue_len.download_data();
            const std::vector<uint64_t> h_qd = queue_dump.download_data();
            const std::vector<uint64_t> h_cs = state_checksum.download_data();

            MkpHostOutputsView got = {};
            got.trace_len = h_tl.data();
            got.trace = h_tr.data();
            got.exec_digest = h_ed.data();
            got.y = h_y.data();
            got.queue_len = h_ql.data();
            got.queue_dump = h_qd.data();
            got.state_checksum = h_cs.data();

            std::string check_error;
            if (!mkp_check_outputs(rs.run, expected[ri], got, &check_error)) {
                if (error) *error = where.str() + check_error;
                return false;
            }
        }
    }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        MkpProblemSpec spec = {};
        spec.abi_version = MKP_ABI_VERSION;
        spec.max_M = MKP_MAX_M;
        spec.max_K = 256;
        spec.max_Q = 4;
        spec.max_R = MKP_MAX_R;
        spec.max_njobs = MKP_MAX_NJOBS;
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

        const std::vector<Scenario> scenarios = build_scenarios();

        int passed = 0;
        const int total = static_cast<int>(scenarios.size());

        for (const Scenario& sc : scenarios) {
            std::string error;
            bool ok = false;
            try {
                ok = run_scenario(sc, state, workspace.ptr, workspace_bytes,
                                  stream, &error);
            } catch (const std::exception& ex) {
                error = ex.what();
                ok = false;
            }
            if (ok) {
                ++passed;
                std::printf("case %-36s PASS  K=%d Q=%d runs=%d%s\n",
                            sc.name.c_str(), sc.K, sc.Q,
                            (int)sc.rounds.size(),
                            sc.replay ? " (replayed)" : "");
            } else {
                std::printf("case %-36s FAIL  %s\n", sc.name.c_str(),
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
