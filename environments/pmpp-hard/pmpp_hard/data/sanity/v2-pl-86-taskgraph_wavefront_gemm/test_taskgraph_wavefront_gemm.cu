// ============================================================================
// file: test_taskgraph_wavefront_gemm.cu
// ============================================================================

#include "taskgraph_wavefront_gemm_common.h"
#include "taskgraph_wavefront_gemm_oracle.hpp"

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

    void refill_data_with_guard() {
        CUDA_CHECK(cudaMemset(ptr, kGuardByte, data_bytes));
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

struct RoundData {
    TwgRunSpec run;
    bool reset_before = false;  // reset state before this round (ri > 0 only)
    std::vector<int8_t> U;      // [R*32, K]
    std::vector<int8_t> V;      // [C*32, K]
};

struct Scenario {
    std::string name;
    int R = 0;
    int C = 0;
    bool replay = false;
    std::vector<RoundData> rounds;
};

static int8_t twg_gen_value(SplitMix64& rng, int dist) {
    switch (dist) {
        case TWG_DIST_SMALL:
            return static_cast<int8_t>(rng.uniform_int(-2, 2));
        case TWG_DIST_SPARSE:
            return rng.chance(0.90)
                ? static_cast<int8_t>(0)
                : static_cast<int8_t>(rng.uniform_int(-127, 127));
        case TWG_DIST_SATURATE:
            return rng.chance(0.85)
                ? static_cast<int8_t>(rng.chance(0.5) ? 127 : -127)
                : static_cast<int8_t>(rng.uniform_int(-8, 8));
        default:
            return static_cast<int8_t>(rng.uniform_int(-127, 127));
    }
}

static RoundData make_round(
    int R, int C, int K, int P1, int P2, int dump,
    int dist, uint64_t seed, bool reset_before = false) {
    RoundData rd;
    rd.reset_before = reset_before;

    rd.run = {};
    rd.run.abi_version = TWG_ABI_VERSION;
    rd.run.R = R;
    rd.run.C = C;
    rd.run.K = K;
    rd.run.P1 = P1;
    rd.run.P2 = P2;
    rd.run.dump = dump;
    rd.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    rd.run.distribution_id = dist;
    rd.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!twg_validate_run_spec(&rd.run)) {
        throw std::runtime_error("invalid run spec in make_round");
    }

    SplitMix64 rng(g_state ^ seed);
    rd.U.resize((size_t)R * 32 * K);
    rd.V.resize((size_t)C * 32 * K);
    for (size_t idx = 0; idx < rd.U.size(); ++idx) {
        rd.U[idx] = dist == TWG_DIST_ZERO_U ? 0 : twg_gen_value(rng, dist);
    }
    for (size_t idx = 0; idx < rd.V.size(); ++idx) {
        rd.V[idx] = dist == TWG_DIST_ZERO_U
            ? static_cast<int8_t>(rng.uniform_int(-127, 127))
            : twg_gen_value(rng, dist);
    }
    return rd;
}

static int twg_alt_K(int K) {
    if (K == 64) return 128;
    if (K == 128) return 256;
    return 64;
}

static Scenario make_standard_scenario(
    const std::string& name, int R, int C, int K, int dist, uint64_t seed,
    int rounds) {
    Scenario sc;
    sc.name = name;
    sc.R = R;
    sc.C = C;
    const int p1_seq[4] = {3, 0, 7, 12};
    const int p2_seq[4] = {5, 15, 2, 1};
    for (int q = 0; q < rounds; ++q) {
        const int Kq = (q % 3 == 1) ? twg_alt_K(K) : K;
        const int dump = q == 1 ? 1 : 0;
        sc.rounds.push_back(make_round(
            R, C, Kq, p1_seq[q & 3], p2_seq[q & 3], dump, dist,
            seed * 1000003ULL + (uint64_t)q));
    }
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> scenarios;
    uint64_t s = 0x100000001b3ULL;

    struct Combo {
        int R, C, K;
    };
    const Combo combos[7] = {
        {4, 4, 64},   {8, 6, 128},  {6, 8, 64},  {16, 16, 128},
        {32, 8, 64},  {8, 32, 256}, {17, 13, 128},
    };
    const char* dist_names[5] = {
        "uniform", "small", "sparse", "saturate", "zero_u"};

    // Full sweep: 5 distributions x 7 shape combos, 3 rounds each
    // (round 1 changes K and dumps the full state).
    for (int d = 0; d < 5; ++d) {
        for (int c = 0; c < 7; ++c) {
            const Combo& cb = combos[c];
            char name[128];
            std::snprintf(name, sizeof(name), "%s_c%d_R%d_C%d_K%d",
                          dist_names[d], c, cb.R, cb.C, cb.K);
            scenarios.push_back(make_standard_scenario(
                name, cb.R, cb.C, cb.K, d, s++, 3));
        }
    }

    // Max shape (32 x 32 tiles, K = 256) for two distributions
    // (bounded oracle time), 2 rounds each.
    for (int d = 0; d < 2; ++d) {
        const int dd = d == 0 ? TWG_DIST_UNIFORM : TWG_DIST_SPARSE;
        char name[128];
        std::snprintf(name, sizeof(name), "%s_max_R32_C32_K256",
                      dist_names[dd]);
        scenarios.push_back(make_standard_scenario(
            name, 32, 32, 256, dd, s++, 2));
    }

    // Seed-shifted repeats of the first 3 combos.
    for (int d = 0; d < 5; ++d) {
        for (int c = 0; c < 3; ++c) {
            const Combo& cb = combos[c];
            char name[128];
            std::snprintf(name, sizeof(name), "%s_c%d_seed2",
                          dist_names[d], c);
            scenarios.push_back(make_standard_scenario(
                name, cb.R, cb.C, cb.K, d, s++, 3));
        }
    }

    // Edge: single tile row -> the DAG is a pure left-to-right chain.
    scenarios.push_back(make_standard_scenario(
        "edge_chain_row_R1_C32", 1, 32, 256, TWG_DIST_UNIFORM, s++, 2));

    // Edge: single tile column -> pure top-to-bottom chain.
    scenarios.push_back(make_standard_scenario(
        "edge_chain_col_R32_C1", 32, 1, 256, TWG_DIST_UNIFORM, s++, 2));

    // Edge: single tile, 6 rounds (deep round-counter / salt coupling).
    {
        Scenario sc;
        sc.name = "edge_single_tile_6rounds";
        sc.R = 1;
        sc.C = 1;
        for (int q = 0; q < 6; ++q) {
            sc.rounds.push_back(make_round(
                1, 1, 64, q, 15 - q, q == 5 ? 1 : 0, TWG_DIST_SATURATE, s));
            s++;
        }
        scenarios.push_back(std::move(sc));
    }

    // Edge: K flips every round (64 -> 256 -> 128 -> 64).
    {
        Scenario sc;
        sc.name = "edge_k_flip_R12_C12";
        sc.R = 12;
        sc.C = 12;
        const int ks[4] = {64, 256, 128, 64};
        for (int q = 0; q < 4; ++q) {
            sc.rounds.push_back(make_round(
                12, 12, ks[q], 2 * q, 9 - q, q == 3 ? 1 : 0,
                TWG_DIST_UNIFORM, s));
            s++;
        }
        scenarios.push_back(std::move(sc));
    }

    // Edge: P1 = P2 = 0 -> coupling flows only through the salt.
    {
        Scenario sc;
        sc.name = "edge_p_zero_R16_C16";
        sc.R = 16;
        sc.C = 16;
        for (int q = 0; q < 2; ++q) {
            sc.rounds.push_back(make_round(
                16, 16, 128, 0, 0, q == 1 ? 1 : 0, TWG_DIST_SPARSE, s));
            s++;
        }
        scenarios.push_back(std::move(sc));
    }

    // Edge: U all zero + max P coefficients -> state driven purely by the
    // salt XOR of zero bytes and the linear neighbor terms.
    {
        Scenario sc;
        sc.name = "edge_zero_u_pmax_R16_C16";
        sc.R = 16;
        sc.C = 16;
        for (int q = 0; q < 3; ++q) {
            sc.rounds.push_back(make_round(
                16, 16, 128, 15, 15, q == 2 ? 1 : 0, TWG_DIST_ZERO_U, s));
            s++;
        }
        scenarios.push_back(std::move(sc));
    }

    // Edge: reset in the middle of a scenario (round counter must restart).
    {
        Scenario sc;
        sc.name = "edge_mid_reset_R8_C8";
        sc.R = 8;
        sc.C = 8;
        sc.rounds.push_back(make_round(
            8, 8, 128, 3, 5, 0, TWG_DIST_UNIFORM, s++));
        sc.rounds.push_back(make_round(
            8, 8, 128, 1, 2, 1, TWG_DIST_UNIFORM, s++));
        sc.rounds.push_back(make_round(
            8, 8, 128, 4, 4, 0, TWG_DIST_UNIFORM, s++, /*reset_before=*/true));
        sc.rounds.push_back(make_round(
            8, 8, 64, 6, 0, 1, TWG_DIST_UNIFORM, s++));
        scenarios.push_back(std::move(sc));
    }

    // Edge: dump every round.
    {
        Scenario sc;
        sc.name = "edge_dump_every_R8_C8";
        sc.R = 8;
        sc.C = 8;
        for (int q = 0; q < 3; ++q) {
            sc.rounds.push_back(make_round(
                8, 8, 64, 5, 5, 1, TWG_DIST_SMALL, s));
            s++;
        }
        scenarios.push_back(std::move(sc));
    }

    // Determinism replay markers (full scenario re-executed byte-exactly).
    scenarios[0].replay = true;
    scenarios[35].replay = true;   // uniform max shape
    scenarios[52].replay = true;   // edge_chain_row (pure chain DAG)

    return scenarios;
}

static bool run_scenario(
    const Scenario& sc,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    std::string* error) {
    const int R = sc.R;
    const int C = sc.C;

    DeviceBuffer<int8_t> dU;
    DeviceBuffer<int8_t> dV;
    dU.allocate((size_t)R * 32 * TWG_MAX_K);
    dV.allocate((size_t)C * 32 * TWG_MAX_K);

    GuardedDeviceBuffer<int32_t> round_out;
    GuardedDeviceBuffer<uint32_t> border_row;
    GuardedDeviceBuffer<uint32_t> border_col;
    GuardedDeviceBuffer<uint64_t> state_checksum;
    GuardedDeviceBuffer<uint32_t> acc_dump;
    round_out.allocate(1);
    border_row.allocate((size_t)C * 32);
    border_col.allocate((size_t)R * 32);
    state_checksum.allocate(1);
    acc_dump.allocate((size_t)R * 32 * (size_t)C * 32);

    // Oracle pass: compute the expected outputs of every round once.
    std::vector<TwgExpected> expected(sc.rounds.size());
    {
        TwgOracle oracle;
        oracle.reset(R, C);
        for (size_t ri = 0; ri < sc.rounds.size(); ++ri) {
            if (ri > 0 && sc.rounds[ri].reset_before) oracle.reset(R, C);
            oracle.run_round(sc.rounds[ri].run, sc.rounds[ri].U.data(),
                             sc.rounds[ri].V.data(), &expected[ri]);
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

            CUDA_CHECK(cudaMemcpy(dU.ptr, rd.U.data(), rd.U.size(),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dV.ptr, rd.V.data(), rd.V.size(),
                                  cudaMemcpyHostToDevice));
            round_out.refill_data_with_guard();
            border_row.refill_data_with_guard();
            border_col.refill_data_with_guard();
            state_checksum.refill_data_with_guard();
            acc_dump.refill_data_with_guard();

            TwgInputs inputs = {};
            inputs.U = dU.ptr;
            inputs.V = dV.ptr;
            TwgOutputs outputs = {};
            outputs.round_out = round_out.ptr;
            outputs.border_row = border_row.ptr;
            outputs.border_col = border_col.ptr;
            outputs.state_checksum = state_checksum.ptr;
            outputs.acc_dump = acc_dump.ptr;

            CUDA_CHECK(solution_run(
                state, &rd.run, &inputs, &outputs, workspace,
                workspace_bytes, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            std::ostringstream where;
            where << "pass " << pass << " round " << ri << ": ";

            std::string sub;
            if (!round_out.check_guards("round_out", &sub) ||
                !border_row.check_guards("border_row", &sub) ||
                !border_col.check_guards("border_col", &sub) ||
                !state_checksum.check_guards("state_checksum", &sub) ||
                !acc_dump.check_guards("acc_dump", &sub)) {
                if (error) *error = where.str() + sub;
                return false;
            }

            // Input immutability.
            {
                std::vector<int8_t> hU(rd.U.size());
                std::vector<int8_t> hV(rd.V.size());
                CUDA_CHECK(cudaMemcpy(hU.data(), dU.ptr, hU.size(),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hV.data(), dV.ptr, hV.size(),
                                      cudaMemcpyDeviceToHost));
                if (hU != rd.U) {
                    if (error) *error = where.str() + "input U modified";
                    return false;
                }
                if (hV != rd.V) {
                    if (error) *error = where.str() + "input V modified";
                    return false;
                }
            }

            const std::vector<int32_t> h_round = round_out.download_data();
            const std::vector<uint32_t> h_brow = border_row.download_data();
            const std::vector<uint32_t> h_bcol = border_col.download_data();
            const std::vector<uint64_t> h_csum =
                state_checksum.download_data();
            const std::vector<uint32_t> h_dump = acc_dump.download_data();

            if (rd.run.dump == 0) {
                // acc_dump must be untouched on non-dump rounds.
                const uint32_t guard_word = 0xA5A5A5A5u;
                for (size_t idx = 0; idx < h_dump.size(); ++idx) {
                    if (h_dump[idx] != guard_word) {
                        if (error) {
                            std::ostringstream oss;
                            oss << where.str()
                                << "acc_dump written on a dump==0 round at "
                                << "flat index " << idx;
                            *error = oss.str();
                        }
                        return false;
                    }
                }
            }

            TwgHostOutputsView got = {};
            got.round_out = h_round.data();
            got.border_row = h_brow.data();
            got.border_col = h_bcol.data();
            got.state_checksum = h_csum.data();
            got.acc_dump = rd.run.dump == 1 ? h_dump.data() : nullptr;

            std::string check_error;
            if (!twg_check_outputs(rd.run, expected[ri], got, &check_error)) {
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

        TwgProblemSpec spec = {};
        spec.abi_version = TWG_ABI_VERSION;
        spec.max_R = TWG_MAX_R;
        spec.max_C = TWG_MAX_C;
        spec.max_K = TWG_MAX_K;
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
                    "case %-36s PASS  R=%d C=%d rounds=%d%s\n",
                    sc.name.c_str(), sc.R, sc.C,
                    static_cast<int>(sc.rounds.size()),
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
