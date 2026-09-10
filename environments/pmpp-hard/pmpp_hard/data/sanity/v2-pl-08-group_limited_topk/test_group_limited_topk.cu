// ============================================================================
// file: test_group_limited_topk.cu
// ============================================================================

#include "group_limited_topk_common.h"
#include "group_limited_topk_oracle.hpp"

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

static constexpr uint64_t g_state = 0x3bd39e10cb0ef593ULL;
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

    float uniform_float(float lo, float hi) {
        const float u = static_cast<float>((next_u64() >> 11) * 0x1.0p-53);
        return lo + (hi - lo) * u;
    }

    bool chance(float p) {
        return uniform_float(0.0f, 1.0f) < p;
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
        if (raw) {
            cudaFree(raw);
        }
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
        }

        for (size_t i = 0; i < kGuardBytes; ++i) {
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
    GltRunSpec run;
    std::vector<float> score;
    std::vector<float> uniform_u;
};

static float gen_score(
    SplitMix64& rng,
    int distribution_id,
    int row,
    int expert,
    int V,
    int G) {
    const int group_size = V / G;
    const int group = expert / group_size;
    const int off = expert - group * group_size;

    switch (distribution_id) {
        case GLT_DIST_UNIFORM:
            return rng.uniform_float(-3.0f, 3.0f);

        case GLT_DIST_PEAKED: {
            const int peak = (row * 1315423911u + 17) % V;
            if (expert == peak) {
                return 16.0f;
            }
            if ((expert + row * 13) % 257 == 0) {
                return rng.uniform_float(7.0f, 12.0f);
            }
            return rng.uniform_float(-4.0f, 2.0f);
        }

        case GLT_DIST_GROUP_SKEW: {
            const int hot0 = row % G;
            const int hot1 = (hot0 + 3) % G;
            const int hot2 = (hot0 + 7) % G;

            float base = rng.uniform_float(-2.0f, 2.0f);
            if (group == hot0) {
                base += 8.0f;
            } else if (group == hot1) {
                base += 5.0f;
            } else if (group == hot2) {
                base += 3.0f;
            }

            base += 0.001f * static_cast<float>(group_size - off);
            return base;
        }

        case GLT_DIST_MANY_TIES: {
            static const float vals[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            return vals[(expert + row * 7) % 5];
        }

        default:
            return rng.uniform_float(-2.0f, 2.0f);
    }
}

static void force_group_boundary_ties(HostCase* hc) {
    const int B = hc->run.B;
    const int V = hc->run.V;
    const int G = hc->run.G;
    const int group_size = V / G;

    if (G < 8 || group_size < 8) {
        return;
    }

    for (int row = 0; row < B; ++row) {
        const int tie_groups = std::min(G, 8);

        for (int g = 0; g < G; ++g) {
            for (int off = 0; off < group_size; ++off) {
                float v = -10.0f - static_cast<float>(g) * 0.01f - static_cast<float>(off) * 0.0001f;

                if (g < tie_groups) {
                    if (off < hc->run.group_k) {
                        v = 5.0f;
                    } else {
                        v = 0.0f;
                    }
                }

                hc->score[(size_t)row * (size_t)V + (size_t)g * (size_t)group_size + (size_t)off] = v;
            }
        }
    }
}

static void force_final_boundary_ties(HostCase* hc) {
    const int B = hc->run.B;
    const int V = hc->run.V;
    const int G = hc->run.G;
    const int group_size = V / G;

    for (int row = 0; row < B; ++row) {
        for (int expert = 0; expert < V; ++expert) {
            const int g = expert / group_size;
            const int off = expert - g * group_size;

            float v = -5.0f;

            if (g < hc->run.n_groups) {
                v = 3.0f;
                if (off >= group_size / 2) {
                    v = 2.0f;
                }
            } else {
                v = -3.0f;
            }

            hc->score[(size_t)row * (size_t)V + (size_t)expert] = v;
        }
    }
}

static HostCase make_case(
    const char* name,
    int B,
    int V,
    int G,
    int group_k,
    int n_groups,
    int final_k,
    int distribution_id,
    uint64_t seed,
    bool force_group_ties,
    bool force_final_ties) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = GLT_ABI_VERSION;
    hc.run.B = B;
    hc.run.V = V;
    hc.run.G = G;
    hc.run.group_k = group_k;
    hc.run.n_groups = n_groups;
    hc.run.final_k = final_k;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!glt_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid GltRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.score.resize((size_t)B * (size_t)V);
    hc.uniform_u.resize((size_t)B);

    for (int row = 0; row < B; ++row) {
        hc.uniform_u[(size_t)row] = rng.uniform_float(0.0f, 0.999999f);

        for (int expert = 0; expert < V; ++expert) {
            hc.score[(size_t)row * (size_t)V + (size_t)expert] =
                gen_score(rng, distribution_id, row, expert, V, G);
        }
    }

    if (force_group_ties) {
        force_group_boundary_ties(&hc);
    }

    if (force_final_ties) {
        force_final_boundary_ties(&hc);
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_case(
        "uniform_B128_V256_G8_gk2_ng2_fk8",
        128, 256, 8, 2, 2, 8,
        GLT_DIST_UNIFORM,
        s++,
        false,
        false));

    cases.push_back(make_case(
        "peaked_B256_V1024_G16_gk4_ng4_fk16",
        256, 1024, 16, 4, 4, 16,
        GLT_DIST_PEAKED,
        s++,
        false,
        false));

    cases.push_back(make_case(
        "group_skew_B256_V4096_G32_gk4_ng8_fk64",
        256, 4096, 32, 4, 8, 64,
        GLT_DIST_GROUP_SKEW,
        s++,
        false,
        false));

    cases.push_back(make_case(
        "ties_B512_V1024_G32_gk2_ng8_fk16",
        512, 1024, 32, 2, 8, 16,
        GLT_DIST_MANY_TIES,
        s++,
        false,
        false));

    cases.push_back(make_case(
        "largeB_uniform_B4096_V256_G16_gk4_ng4_fk64",
        4096, 256, 16, 4, 4, 64,
        GLT_DIST_UNIFORM,
        s++,
        false,
        false));

    cases.push_back(make_case(
        "edge_group_boundary_ties_G8_ng8",
        128, 256, 8, 4, 8, 64,
        GLT_DIST_MANY_TIES,
        s++,
        true,
        false));

    cases.push_back(make_case(
        "edge_final_boundary_ties",
        128, 1024, 16, 2, 4, 64,
        GLT_DIST_MANY_TIES,
        s++,
        false,
        true));

    cases.push_back(make_case(
        "edge_ngroups_equals_G",
        256, 256, 8, 2, 8, 16,
        GLT_DIST_GROUP_SKEW,
        s++,
        false,
        false));

    cases.push_back(make_case(
        "edge_final_k_gt_survivors",
        128, 256, 32, 4, 2, 64,
        GLT_DIST_UNIFORM,
        s++,
        false,
        false));

    cases.push_back(make_case(
        "edge_min_group_size_gk4",
        128, 256, 32, 4, 2, 8,
        GLT_DIST_MANY_TIES,
        s++,
        false,
        false));

    cases.push_back(make_case(
        "combo_B256_V4096_G8_gk2_ng4_fk16_peaked",
        256, 4096, 8, 2, 4, 16,
        GLT_DIST_PEAKED,
        s++,
        false,
        false));

    cases.push_back(make_case(
        "combo_B512_V1024_G8_gk4_ng2_fk8_group_skew",
        512, 1024, 8, 4, 2, 8,
        GLT_DIST_GROUP_SKEW,
        s++,
        false,
        false));

    // Note: with the legal shape family V in {256,1024,4096}, G in {8,16,32},
    // the minimum group size is 8, while group_k is only {2,4}. Therefore a
    // literal legal group_k > group_size edge is impossible; edge_min_group_size_gk4
    // exercises the closest legal boundary.

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<float>& d_score,
    const DeviceBuffer<float>& d_uniform_u,
    std::string* error) {
    if (d_score.download() != hc.score) {
        if (error) *error = "input score modified";
        return false;
    }

    if (d_uniform_u.download() != hc.uniform_u) {
        if (error) *error = "input uniform_u modified";
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
    const int B = hc.run.B;
    const int V = hc.run.V;
    const int G = hc.run.G;
    const int n_groups = hc.run.n_groups;
    const int final_k = hc.run.final_k;

    DeviceBuffer<float> d_score;
    DeviceBuffer<float> d_uniform_u;

    d_score.allocate(hc.score.size());
    d_uniform_u.allocate(hc.uniform_u.size());

    d_score.upload(hc.score);
    d_uniform_u.upload(hc.uniform_u);

    GuardedDeviceBuffer<int32_t> d_selected_group_ids;
    GuardedDeviceBuffer<int32_t> d_survivor_expert_ids;
    GuardedDeviceBuffer<int32_t> d_survivor_count;
    GuardedDeviceBuffer<float> d_weights;
    GuardedDeviceBuffer<float> d_group_scores;

    d_selected_group_ids.allocate((size_t)B * (size_t)n_groups);
    d_survivor_expert_ids.allocate((size_t)B * (size_t)final_k);
    d_survivor_count.allocate((size_t)B);
    d_weights.allocate((size_t)B * (size_t)final_k);
    d_group_scores.allocate((size_t)B * (size_t)G);

    GltInputs inputs = {};
    inputs.score = d_score.ptr;
    inputs.uniform_u = d_uniform_u.ptr;

    GltOutputs outputs = {};
    outputs.selected_group_ids = d_selected_group_ids.ptr;
    outputs.survivor_expert_ids = d_survivor_expert_ids.ptr;
    outputs.survivor_count = d_survivor_count.ptr;
    outputs.weights = d_weights.ptr;
    outputs.group_scores = d_group_scores.ptr;

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

    if (!check_input_unchanged(hc, d_score, d_uniform_u, error)) {
        return false;
    }

    if (!d_selected_group_ids.check_guards("selected_group_ids", error)) return false;
    if (!d_survivor_expert_ids.check_guards("survivor_expert_ids", error)) return false;
    if (!d_survivor_count.check_guards("survivor_count", error)) return false;
    if (!d_weights.check_guards("weights", error)) return false;
    if (!d_group_scores.check_guards("group_scores", error)) return false;

    const std::vector<int32_t> h_selected_group_ids = d_selected_group_ids.download_data();
    const std::vector<int32_t> h_survivor_expert_ids = d_survivor_expert_ids.download_data();
    const std::vector<int32_t> h_survivor_count = d_survivor_count.download_data();
    const std::vector<float> h_weights = d_weights.download_data();
    const std::vector<float> h_group_scores = d_group_scores.download_data();

    GltHostInputsView host_inputs = {};
    host_inputs.score = hc.score.data();
    host_inputs.uniform_u = hc.uniform_u.data();

    GltExpected expected;
    glt_cpu_oracle(hc.run, host_inputs, &expected);

    GltHostOutputsView got = {};
    got.selected_group_ids = h_selected_group_ids.data();
    got.survivor_expert_ids = h_survivor_expert_ids.data();
    got.survivor_count = h_survivor_count.data();
    got.weights = h_weights.data();
    got.group_scores = h_group_scores.data();

    if (!glt_check_group_scores(hc.run, expected, got, error)) {
        return false;
    }

    if (!glt_check_selected_groups(hc.run, expected, got, error)) {
        return false;
    }

    if (!glt_check_survivors_and_weights(hc.run, expected, got, error)) {
        return false;
    }

    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        GltProblemSpec spec = {};
        spec.abi_version = GLT_ABI_VERSION;
        spec.max_B = GLT_MAX_B;
        spec.max_V = GLT_MAX_V;
        spec.max_G = GLT_MAX_G;
        spec.max_n_groups = GLT_MAX_N_GROUPS;
        spec.max_final_k = GLT_MAX_FINAL_K;
        spec.flags = 0;

        size_t workspace_bytes = solution_workspace_bytes(&spec); if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/
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

        const std::vector<HostCase> cases = build_test_cases();

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
                    "case %-52s PASS  B=%d V=%d G=%d gk=%d ng=%d fk=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.B,
                    hc.run.V,
                    hc.run.G,
                    hc.run.group_k,
                    hc.run.n_groups,
                    hc.run.final_k,
                    hc.run.distribution_id);
            } else {
                std::printf(
                    "case %-52s FAIL  %s\n",
                    hc.name.c_str(),
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
