// ============================================================================
// file: test_dualtier_credit_router.cu
// ============================================================================

#include "dualtier_credit_router_common.h"
#include "dualtier_credit_router_oracle.hpp"

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
                    oss << "left guard corrupted for " << name << " at byte "
                        << i;
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
                    oss << "right guard corrupted for " << name << " at byte "
                        << i;
                    *error = oss.str();
                }
                return false;
            }
        }
        return true;
    }
};

struct Shape {
    int D, H, E, P, ccap, bq;
};

struct RoundData {
    DtrRunSpec run;
    bool reset_before = false;
    std::vector<int8_t> x;
    std::vector<int8_t> wnode;
    std::vector<int8_t> wexp;
    std::vector<int8_t> w1;
    std::vector<int8_t> w2;
};

struct Scenario {
    std::string name;
    Shape shape;
    bool replay = false;
    std::vector<RoundData> rounds;
};

static RoundData make_round(
    const Shape& sh, int N, int refill, int qshift, int dist,
    uint64_t seed, int round_idx, bool reset_before = false) {
    RoundData rd;
    rd.reset_before = reset_before;

    rd.run = {};
    rd.run.abi_version = DTR_ABI_VERSION;
    rd.run.N = N;
    rd.run.D = sh.D;
    rd.run.H = sh.H;
    rd.run.E = sh.E;
    rd.run.P = sh.P;
    rd.run.ccap = sh.ccap;
    rd.run.bq = sh.bq;
    rd.run.refill = refill;
    rd.run.qshift = qshift;
    rd.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    rd.run.distribution_id = dist;
    rd.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);
    if (!dtr_validate_run_spec(&rd.run)) {
        throw std::runtime_error("invalid run spec in make_round");
    }

    SplitMix64 rng(g_state ^ seed);
    rd.x.resize((size_t)N * sh.D);
    rd.wnode.resize((size_t)sh.P * sh.D);
    rd.wexp.resize((size_t)sh.E * sh.D);
    rd.w1.resize((size_t)sh.E * sh.H * sh.D);
    rd.w2.resize((size_t)sh.E * sh.D * sh.H);

    const int hot = (int)((seed + (uint64_t)round_idx) % (uint64_t)sh.P);

    for (int t = 0; t < N; ++t) {
        const bool zero_row = dist == DTR_DIST_ZERO_X && rng.chance(0.70);
        for (int d = 0; d < sh.D; ++d) {
            int8_t v;
            switch (dist) {
                case DTR_DIST_HOT_NODE:
                case DTR_DIST_BURSTY:
                    v = (int8_t)rng.uniform_int(0, 127);
                    break;
                case DTR_DIST_TIES:
                    v = (int8_t)rng.uniform_int(-1, 1);
                    break;
                default:
                    v = (int8_t)rng.uniform_int(-127, 127);
                    break;
            }
            rd.x[(size_t)t * sh.D + d] = zero_row ? 0 : v;
        }
    }
    for (int p = 0; p < sh.P; ++p) {
        for (int d = 0; d < sh.D; ++d) {
            int8_t v;
            switch (dist) {
                case DTR_DIST_HOT_NODE:
                    v = p < 2 ? (int8_t)rng.uniform_int(64, 127)
                              : (int8_t)rng.uniform_int(-32, 32);
                    break;
                case DTR_DIST_BURSTY:
                    v = p == hot ? (int8_t)rng.uniform_int(64, 127)
                                 : (int8_t)rng.uniform_int(-16, 16);
                    break;
                case DTR_DIST_TIES:
                    v = (int8_t)rng.uniform_int(-1, 1);
                    break;
                case DTR_DIST_ZERO_X:
                    v = (int8_t)rng.uniform_int(-8, 8);
                    break;
                default:
                    v = (int8_t)rng.uniform_int(-127, 127);
                    break;
            }
            rd.wnode[(size_t)p * sh.D + d] = v;
        }
    }
    for (size_t i = 0; i < rd.wexp.size(); ++i) {
        rd.wexp[i] = dist == DTR_DIST_TIES
            ? (int8_t)rng.uniform_int(-1, 1)
            : (dist == DTR_DIST_ZERO_X ? (int8_t)rng.uniform_int(-8, 8)
                                       : (int8_t)rng.uniform_int(-127, 127));
    }
    for (size_t i = 0; i < rd.w1.size(); ++i) {
        rd.w1[i] = dist == DTR_DIST_TIES ? (int8_t)rng.uniform_int(-3, 3)
                                         : (int8_t)rng.uniform_int(-127, 127);
    }
    for (size_t i = 0; i < rd.w2.size(); ++i) {
        rd.w2[i] = dist == DTR_DIST_TIES ? (int8_t)rng.uniform_int(-3, 3)
                                         : (int8_t)rng.uniform_int(-127, 127);
    }
    return rd;
}

static Scenario make_standard_scenario(
    const std::string& name, const Shape& sh, int Nbase, int dist,
    uint64_t seed, int rounds) {
    Scenario sc;
    sc.name = name;
    sc.shape = sh;
    const int qs[4] = {6, 4, 8, 5};
    for (int q = 0; q < rounds; ++q) {
        const int N = q == 1 ? (Nbase / 2 >= DTR_MIN_N ? Nbase / 2 : Nbase)
                             : Nbase;
        const int refill =
            q == 0 ? sh.ccap : (q == 1 ? sh.ccap / 2 : (sh.ccap >= 4 ? 1 : 0));
        sc.rounds.push_back(make_round(
            sh, N, refill, qs[q & 3], dist, seed * 1000003ULL + (uint64_t)q,
            q));
    }
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> scenarios;
    uint64_t s = 0x300000001b3ULL;

    struct Combo {
        int N;
        Shape sh;
    };
    const Combo combos[7] = {
        {2048, {64, 128, 32, 4, 16, 32}},
        {3072, {64, 128, 32, 8, 8, 64}},
        {2048, {64, 256, 64, 16, 8, 32}},
        {4096, {128, 128, 64, 8, 16, 64}},
        {3000, {64, 128, 128, 16, 8, 32}},
        {6144, {128, 256, 64, 4, 32, 128}},
        {4096, {64, 128, 128, 16, 32, 64}},
    };
    const char* dist_names[5] = {
        "uniform", "hot_node", "ties", "zero_x", "bursty"};

    // Full sweep: 5 distributions x 7 shape combos, 3 rounds each.
    for (int d = 0; d < 5; ++d) {
        for (int c = 0; c < 7; ++c) {
            char name[128];
            std::snprintf(name, sizeof(name), "%s_c%d_N%d_E%d_P%d_cc%d_bq%d",
                          dist_names[d], c, combos[c].N, combos[c].sh.E,
                          combos[c].sh.P, combos[c].sh.ccap, combos[c].sh.bq);
            scenarios.push_back(make_standard_scenario(
                name, combos[c].sh, combos[c].N, d, s++, 3));
        }
    }

    // Max shape (bounded oracle time), 2 rounds each.
    const Shape maxsh = {128, 256, 128, 8, 64, 128};
    for (int d = 0; d < 2; ++d) {
        char name[128];
        std::snprintf(name, sizeof(name), "%s_max_N16384_E128",
                      dist_names[d == 0 ? DTR_DIST_UNIFORM
                                        : DTR_DIST_HOT_NODE]);
        scenarios.push_back(make_standard_scenario(
            name, maxsh, 16384,
            d == 0 ? DTR_DIST_UNIFORM : DTR_DIST_HOT_NODE, s++, 2));
    }

    // Seed-shifted repeats of the first 3 combos.
    for (int d = 0; d < 5; ++d) {
        for (int c = 0; c < 3; ++c) {
            char name[128];
            std::snprintf(name, sizeof(name), "%s_c%d_seed2",
                          dist_names[d], c);
            scenarios.push_back(make_standard_scenario(
                name, combos[c].sh, combos[c].N, d, s++, 3));
        }
    }

    // Edge: refill goes to zero -> starvation, mass QUEUED then DROPPED.
    {
        Scenario sc;
        sc.name = "edge_refill_zero";
        sc.shape = {64, 128, 32, 4, 16, 32};
        sc.rounds.push_back(make_round(
            sc.shape, 2048, 16, 6, DTR_DIST_HOT_NODE, s++, 0));
        sc.rounds.push_back(make_round(
            sc.shape, 2048, 0, 5, DTR_DIST_HOT_NODE, s++, 1));
        sc.rounds.push_back(make_round(
            sc.shape, 2048, 0, 7, DTR_DIST_HOT_NODE, s++, 2));
        scenarios.push_back(std::move(sc));
    }

    // Edge: tiny credits + tiny backlog -> constant saturation and drops.
    {
        Scenario sc;
        sc.name = "edge_mass_drop";
        sc.shape = {64, 128, 32, 4, 8, 32};
        for (int q = 0; q < 3; ++q) {
            sc.rounds.push_back(make_round(
                sc.shape, 4096, 8, 6, DTR_DIST_HOT_NODE, s++, q));
        }
        scenarios.push_back(std::move(sc));
    }

    // Edge: zero-x storm -> everything ties onto expert 0.
    {
        Scenario sc;
        sc.name = "edge_zero_x_storm";
        sc.shape = {64, 128, 32, 4, 8, 32};
        for (int q = 0; q < 3; ++q) {
            sc.rounds.push_back(make_round(
                sc.shape, 2048, q == 0 ? 8 : 4, 4, DTR_DIST_ZERO_X, s++, q));
        }
        scenarios.push_back(std::move(sc));
    }

    // Edge: coarse-valued tie storm.
    {
        Scenario sc;
        sc.name = "edge_ties_storm";
        sc.shape = {64, 128, 64, 8, 16, 64};
        for (int q = 0; q < 3; ++q) {
            sc.rounds.push_back(make_round(
                sc.shape, 2048, 16, 8, DTR_DIST_TIES, s++, q));
        }
        scenarios.push_back(std::move(sc));
    }

    // Edge: reset mid-scenario (gids, credits, backlog must restart).
    {
        Scenario sc;
        sc.name = "edge_mid_reset";
        sc.shape = {64, 128, 32, 4, 16, 32};
        sc.rounds.push_back(make_round(
            sc.shape, 2048, 16, 6, DTR_DIST_HOT_NODE, s++, 0));
        sc.rounds.push_back(make_round(
            sc.shape, 2048, 4, 5, DTR_DIST_HOT_NODE, s++, 1));
        sc.rounds.push_back(make_round(
            sc.shape, 2048, 8, 7, DTR_DIST_UNIFORM, s++, 2,
            /*reset_before=*/true));
        sc.rounds.push_back(make_round(
            sc.shape, 1024, 2, 4, DTR_DIST_HOT_NODE, s++, 3));
        scenarios.push_back(std::move(sc));
    }

    // Edge: qshift changes every round; backlog tokens enqueued under one
    // qshift are delivered under another (delivering run's qshift wins).
    {
        Scenario sc;
        sc.name = "edge_qshift_sweep";
        sc.shape = {128, 128, 64, 8, 16, 64};
        const int qs[3] = {4, 8, 6};
        for (int q = 0; q < 3; ++q) {
            sc.rounds.push_back(make_round(
                sc.shape, 4096, q == 0 ? 16 : 6, qs[q], DTR_DIST_HOT_NODE,
                s++, q));
        }
        scenarios.push_back(std::move(sc));
    }

    // Edge: bursty churn -- the hot node rotates, so backlogs built one
    // round drain in later rounds while new ones fill.
    {
        Scenario sc;
        sc.name = "edge_bursty_churn";
        sc.shape = {64, 128, 64, 8, 16, 64};
        for (int q = 0; q < 4; ++q) {
            sc.rounds.push_back(make_round(
                sc.shape, 4096, q == 0 ? 16 : 8, 6, DTR_DIST_BURSTY, s++, q));
        }
        scenarios.push_back(std::move(sc));
    }

    // Edge: minimum N, generous credits, single round (near-empty state).
    {
        Scenario sc;
        sc.name = "edge_min_n";
        sc.shape = {64, 128, 32, 4, 64, 128};
        sc.rounds.push_back(make_round(
            sc.shape, 1024, 64, 6, DTR_DIST_UNIFORM, s++, 0));
        scenarios.push_back(std::move(sc));
    }

    // Determinism replay markers.
    scenarios[0].replay = true;
    scenarios[35].replay = true;  // uniform max shape
    scenarios[58].replay = true;  // edge_bursty_churn

    return scenarios;
}

static bool run_scenario(
    const Scenario& sc,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    std::string* error) {
    const Shape& sh = sc.shape;
    int maxN = 0;
    for (const RoundData& rd : sc.rounds) {
        maxN = std::max(maxN, rd.run.N);
    }
    const size_t cap = (size_t)sh.E * sh.ccap;

    DeviceBuffer<int8_t> dx, dwnode, dwexp, dw1, dw2;
    dx.allocate((size_t)maxN * sh.D);
    dwnode.allocate((size_t)sh.P * sh.D);
    dwexp.allocate((size_t)sh.E * sh.D);
    dw1.allocate((size_t)sh.E * sh.H * sh.D);
    dw2.allocate((size_t)sh.E * sh.D * sh.H);

    GuardedDeviceBuffer<int32_t> s2_logits;
    GuardedDeviceBuffer<int32_t> route_nodes;
    GuardedDeviceBuffer<int32_t> route_pe_be;
    GuardedDeviceBuffer<int32_t> log_len;
    GuardedDeviceBuffer<uint64_t> event_log;
    GuardedDeviceBuffer<int32_t> counts;
    GuardedDeviceBuffer<int32_t> offsets;
    GuardedDeviceBuffer<int32_t> packed_gid;
    GuardedDeviceBuffer<int32_t> packed_out;
    GuardedDeviceBuffer<uint32_t> credit_out;
    GuardedDeviceBuffer<uint64_t> state_checksum;
    s2_logits.allocate((size_t)maxN * sh.E);
    route_nodes.allocate((size_t)maxN);
    route_pe_be.allocate((size_t)maxN);
    log_len.allocate(1);
    event_log.allocate((size_t)maxN + cap);
    counts.allocate((size_t)sh.E);
    offsets.allocate((size_t)sh.E + 1);
    packed_gid.allocate(cap);
    packed_out.allocate(cap * (size_t)sh.D);
    credit_out.allocate((size_t)sh.E);
    state_checksum.allocate(1);

    // Oracle pass: expected outputs of every round.
    std::vector<DtrExpected> expected(sc.rounds.size());
    {
        DtrOracle oracle;
        oracle.reset(sc.rounds[0].run);
        for (size_t ri = 0; ri < sc.rounds.size(); ++ri) {
            const RoundData& rd = sc.rounds[ri];
            if (ri > 0 && rd.reset_before) oracle.reset(rd.run);
            oracle.run_round(rd.run, rd.x.data(), rd.wnode.data(),
                             rd.wexp.data(), rd.w1.data(), rd.w2.data(),
                             &expected[ri]);
        }
    }

    const int passes = sc.replay ? 2 : 1;
    for (int pass = 0; pass < passes; ++pass) {
        CUDA_CHECK(solution_reset(state, stream));

        for (size_t ri = 0; ri < sc.rounds.size(); ++ri) {
            const RoundData& rd = sc.rounds[ri];
            if (ri > 0 && rd.reset_before) {
                CUDA_CHECK(solution_reset(state, stream));
            }

            CUDA_CHECK(cudaMemcpy(dx.ptr, rd.x.data(), rd.x.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dwnode.ptr, rd.wnode.data(),
                                  rd.wnode.size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dwexp.ptr, rd.wexp.data(), rd.wexp.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dw1.ptr, rd.w1.data(), rd.w1.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dw2.ptr, rd.w2.data(), rd.w2.size(),
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

            CUDA_CHECK(solution_run(
                state, &rd.run, &inputs, &outputs, workspace,
                workspace_bytes, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            std::ostringstream where;
            where << "pass " << pass << " round " << ri << ": ";

            std::string sub;
            if (!s2_logits.check_guards("s2_logits", &sub) ||
                !route_nodes.check_guards("route_nodes", &sub) ||
                !route_pe_be.check_guards("route_pe_be", &sub) ||
                !log_len.check_guards("log_len", &sub) ||
                !event_log.check_guards("event_log", &sub) ||
                !counts.check_guards("counts", &sub) ||
                !offsets.check_guards("offsets", &sub) ||
                !packed_gid.check_guards("packed_gid", &sub) ||
                !packed_out.check_guards("packed_out", &sub) ||
                !credit_out.check_guards("credit_out", &sub) ||
                !state_checksum.check_guards("state_checksum", &sub)) {
                if (error) *error = where.str() + sub;
                return false;
            }

            // Input immutability.
            {
                std::vector<int8_t> h(rd.x.size());
                CUDA_CHECK(cudaMemcpy(h.data(), dx.ptr, h.size(),
                                      cudaMemcpyDeviceToHost));
                if (h != rd.x) {
                    if (error) *error = where.str() + "input x modified";
                    return false;
                }
                h.resize(rd.wexp.size());
                CUDA_CHECK(cudaMemcpy(h.data(), dwexp.ptr, h.size(),
                                      cudaMemcpyDeviceToHost));
                if (h != rd.wexp) {
                    if (error) *error = where.str() + "input wexp modified";
                    return false;
                }
                h.resize(rd.w1.size());
                CUDA_CHECK(cudaMemcpy(h.data(), dw1.ptr, h.size(),
                                      cudaMemcpyDeviceToHost));
                if (h != rd.w1) {
                    if (error) *error = where.str() + "input w1 modified";
                    return false;
                }
            }

            const std::vector<int32_t> h_s2 = s2_logits.download_data();
            const std::vector<int32_t> h_rn = route_nodes.download_data();
            const std::vector<int32_t> h_rp = route_pe_be.download_data();
            const std::vector<int32_t> h_ll = log_len.download_data();
            const std::vector<uint64_t> h_log = event_log.download_data();
            const std::vector<int32_t> h_cnt = counts.download_data();
            const std::vector<int32_t> h_off = offsets.download_data();
            const std::vector<int32_t> h_pg = packed_gid.download_data();
            const std::vector<int32_t> h_po = packed_out.download_data();
            const std::vector<uint32_t> h_cr = credit_out.download_data();
            const std::vector<uint64_t> h_cs = state_checksum.download_data();

            DtrHostOutputsView got = {};
            got.s2_logits = h_s2.data();
            got.route_nodes = h_rn.data();
            got.route_pe_be = h_rp.data();
            got.log_len = h_ll.data();
            got.event_log = h_log.data();
            got.counts = h_cnt.data();
            got.offsets = h_off.data();
            got.packed_gid = h_pg.data();
            got.packed_out = h_po.data();
            got.credit_out = h_cr.data();
            got.state_checksum = h_cs.data();

            std::string check_error;
            if (!dtr_check_outputs(rd.run, expected[ri], got, &check_error)) {
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
                std::printf(
                    "case %-36s PASS  E=%d P=%d cc=%d bq=%d rounds=%d%s\n",
                    sc.name.c_str(), sc.shape.E, sc.shape.P, sc.shape.ccap,
                    sc.shape.bq, static_cast<int>(sc.rounds.size()),
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
