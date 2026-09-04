// ============================================================================
// file: test_penalty_filter_sample.cu
// ============================================================================

#include "penalty_filter_sample_common.h"
#include "penalty_filter_sample_oracle.hpp"

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

static constexpr uint64_t g_state = 0xfbb67ae8584caa73ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;
static constexpr float kMinPBoundaryEps = 1.0e-4f;

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
        uint64_t z = (state += 0x2ca222e1095e1da9ULL);
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
    PfsRunSpec run;
    std::vector<float> logits;
    std::vector<int32_t> history_token;
    std::vector<int32_t> history_len;
    std::vector<float> repetition_penalty;
    std::vector<float> frequency_penalty;
    std::vector<float> presence_penalty;
    std::vector<float> temperature;
    std::vector<float> min_p;
    std::vector<float> uniform_u;
};

struct BoundaryRow {
    std::vector<PfsScoreId> sorted;
    int min_count = 1;
    int max_count = 1;
};

static float gen_logit(
    SplitMix64& rng,
    int distribution_id,
    int row,
    int token,
    int V) {
    switch (distribution_id) {
        case PFS_DIST_UNIFORM:
            return rng.uniform_float(-3.0f, 3.0f);

        case PFS_DIST_PEAKED: {
            const int peak = (row * 1315423911u + 17) % V;
            if (token == peak) return 12.0f;
            if ((token + row * 13) % 257 == 0) return rng.uniform_float(6.0f, 9.0f);
            return rng.uniform_float(-5.0f, 1.5f);
        }

        case PFS_DIST_HEAVY_TAIL: {
            const int x = (token * 1103515245u + row * 12345u) & 0x7fffffff;
            const int bucket = x % 2048;
            return 9.0f - static_cast<float>(std::sqrt(static_cast<double>(bucket)) * 0.35) +
                   rng.uniform_float(-0.20f, 0.20f);
        }

        case PFS_DIST_MANY_TIES: {
            static const float vals[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            return vals[(token + row * 7) % 5];
        }

        case PFS_DIST_ALL_EQUAL:
            return static_cast<float>((row % 3) - 1);

        default:
            return rng.uniform_float(-2.0f, 2.0f);
    }
}

static HostCase make_case(
    const char* name,
    int B,
    int V,
    int H,
    int distribution_id,
    uint64_t seed,
    bool all_penalized_hot,
    bool all_equal_params) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = PFS_ABI_VERSION;
    hc.run.B = B;
    hc.run.V = V;
    hc.run.H = H;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!pfs_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid PfsRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.logits.resize((size_t)B * (size_t)V);
    hc.history_token.resize((size_t)B * (size_t)std::max(H, 1), 0);
    hc.history_len.resize((size_t)B);
    hc.repetition_penalty.resize((size_t)B);
    hc.frequency_penalty.resize((size_t)B);
    hc.presence_penalty.resize((size_t)B);
    hc.temperature.resize((size_t)B);
    hc.min_p.resize((size_t)B);
    hc.uniform_u.resize((size_t)B);

    for (int row = 0; row < B; ++row) {
        for (int token = 0; token < V; ++token) {
            hc.logits[(size_t)row * (size_t)V + (size_t)token] =
                gen_logit(rng, distribution_id, row, token, V);
        }

        int len = H == 0 ? 0 : rng.uniform_int(0, H);
        if (all_penalized_hot && H > 0) {
            len = H;
        }
        hc.history_len[(size_t)row] = len;

        for (int h = 0; h < H; ++h) {
            int tok;
            if (all_penalized_hot) {
                tok = h % std::min(V, 64);
            } else if (distribution_id == PFS_DIST_PEAKED) {
                tok = (row * 1315423911u + 17 + h * 13) % V;
            } else {
                tok = rng.uniform_int(0, V - 1);
            }
            hc.history_token[(size_t)row * (size_t)std::max(H, 1) + (size_t)h] = tok;
        }

        if (all_equal_params) {
            hc.repetition_penalty[(size_t)row] = 1.20f;
            hc.frequency_penalty[(size_t)row] = 0.0f;
            hc.presence_penalty[(size_t)row] = 0.0f;
            hc.temperature[(size_t)row] = 1.0f;
        } else {
            hc.repetition_penalty[(size_t)row] = rng.uniform_float(1.05f, 1.65f);
            hc.frequency_penalty[(size_t)row] = rng.uniform_float(-0.05f, 0.35f);
            hc.presence_penalty[(size_t)row] = rng.uniform_float(-0.05f, 0.55f);
            hc.temperature[(size_t)row] = rng.uniform_float(0.65f, 1.75f);
        }

        static const float minps[] = {0.0f, 0.05f, 0.2f};
        hc.min_p[(size_t)row] = minps[row % 3];
        hc.uniform_u[(size_t)row] = rng.uniform_float(0.0f, 0.999999f);
    }

    if (B >= 4) {
        hc.min_p[0] = 0.0f;
        hc.min_p[1] = 0.05f;
        hc.min_p[2] = 0.2f;
        hc.uniform_u[3] = 0.999999f;
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_case("uniform_empty_history_B256_V1024_H0",
        256, 1024, 0, PFS_DIST_UNIFORM, s++, false, false));

    cases.push_back(make_case("peaked_B256_V32768_H64",
        256, 32768, 64, PFS_DIST_PEAKED, s++, false, false));

    cases.push_back(make_case("heavy_tail_B512_V4096_H128",
        512, 4096, 128, PFS_DIST_HEAVY_TAIL, s++, false, false));

    cases.push_back(make_case("many_ties_B1024_V1024_H256",
        1024, 1024, 256, PFS_DIST_MANY_TIES, s++, false, false));

    cases.push_back(make_case("all_equal_B256_V1024_H256",
        256, 1024, 256, PFS_DIST_ALL_EQUAL, s++, false, true));

    cases.push_back(make_case("largeB_uniform_B4096_V1024_H32",
        4096, 1024, 32, PFS_DIST_UNIFORM, s++, false, false));

    cases.push_back(make_case("all_penalized_hot_B256_V4096_H256",
        256, 4096, 256, PFS_DIST_PEAKED, s++, true, false));

    return cases;
}

static BoundaryRow compute_boundary_row(const HostCase& hc, int row) {
    const int V = hc.run.V;
    const int H = hc.run.H;
    BoundaryRow br;
    br.sorted.resize((size_t)V);

    std::vector<int32_t> counts((size_t)V, 0);

    int hist_len = hc.history_len[(size_t)row];
    if (hist_len < 0) hist_len = 0;
    if (hist_len > H) hist_len = H;

    for (int h = 0; h < hist_len; ++h) {
        const int tok = hc.history_token[(size_t)row * (size_t)std::max(H, 1) + (size_t)h];
        if (tok >= 0 && tok < V) ++counts[(size_t)tok];
    }

    const float* row_logits = hc.logits.data() + (size_t)row * (size_t)V;

    for (int tok = 0; tok < V; ++tok) {
        br.sorted[(size_t)tok] = PfsScoreId{
            pfs_oracle_adjust_score(
                row_logits[tok],
                counts[(size_t)tok],
                hc.repetition_penalty[(size_t)row],
                hc.frequency_penalty[(size_t)row],
                hc.presence_penalty[(size_t)row],
                hc.temperature[(size_t)row]),
            tok
        };
    }

    std::sort(br.sorted.begin(), br.sorted.end(), pfs_oracle_better);

    const float threshold = pfs_oracle_sanitize_min_p(hc.min_p[(size_t)row]);

    if (threshold <= 0.0f) {
        br.min_count = V;
        br.max_count = V;
        return br;
    }

    const float max_score = br.sorted[0].score;
    int definite = 0;
    int possible = 0;

    for (int i = 0; i < V; ++i) {
        const float rel = pfs_oracle_expf(br.sorted[(size_t)i].score - max_score);
        if (rel >= threshold + kMinPBoundaryEps) ++definite;
        if (rel >= threshold - kMinPBoundaryEps) ++possible;
    }

    if (definite < 1) definite = 1;
    if (possible < definite) possible = definite;

    br.min_count = definite;
    br.max_count = possible;
    return br;
}

static int expected_selected_for_count(const HostCase& hc, const BoundaryRow& br, int row, int count) {
    const float max_score = br.sorted[0].score;

    float denom = 0.0f;
    for (int i = 0; i < count; ++i) {
        denom = pfs_oracle_fadd(denom, pfs_oracle_expf(br.sorted[(size_t)i].score - max_score));
    }
    if (denom <= 0.0f) denom = 1.0f;

    const float u = pfs_oracle_sanitize_u(hc.uniform_u[(size_t)row]);
    float cdf = 0.0f;
    int selected = br.sorted[(size_t)count - 1].id;

    for (int i = 0; i < count; ++i) {
        const float w = pfs_oracle_expf(br.sorted[(size_t)i].score - max_score) / denom;
        cdf = pfs_oracle_fadd(cdf, w);
        if (cdf > u) {
            selected = br.sorted[(size_t)i].id;
            break;
        }
    }

    return selected;
}

static bool tolerant_check_outputs(
    const HostCase& hc,
    const PfsHostOutputsView& got,
    std::string* error) {
    const int B = hc.run.B;
    const int V = hc.run.V;

    for (int row = 0; row < B; ++row) {
        BoundaryRow br = compute_boundary_row(hc, row);

        const int got_count = got.survivor_count[row];
        if (got_count < br.min_count || got_count > br.max_count) {
            if (error) {
                std::ostringstream oss;
                oss << "survivor_count mismatch at row=" << row
                    << ": got " << got_count
                    << ", accepted range [" << br.min_count << ", " << br.max_count << "]";
                *error = oss.str();
            }
            return false;
        }

        if (got_count < 1 || got_count > V) {
            if (error) *error = "survivor_count out of legal range";
            return false;
        }

        const int exp_selected = expected_selected_for_count(hc, br, row, got_count);
        if (got.selected_token[row] != exp_selected) {
            if (error) {
                std::ostringstream oss;
                oss << "selected_token mismatch at row=" << row
                    << ": got " << got.selected_token[row]
                    << ", expected " << exp_selected
                    << " for accepted count " << got_count;
                *error = oss.str();
            }
            return false;
        }

        const int32_t* row_tokens = got.packed_cand_token + (size_t)row * (size_t)V;
        const float* row_probs = got.packed_cand_prob + (size_t)row * (size_t)V;

        const float max_score = br.sorted[0].score;
        float denom = 0.0f;
        for (int i = 0; i < got_count; ++i) {
            denom = pfs_oracle_fadd(denom, pfs_oracle_expf(br.sorted[(size_t)i].score - max_score));
        }
        if (denom <= 0.0f) denom = 1.0f;

        float sum = 0.0f;
        for (int i = 0; i < got_count; ++i) {
            if (row_tokens[i] != br.sorted[(size_t)i].id) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_cand_token mismatch at row=" << row
                        << ", i=" << i
                        << ": got " << row_tokens[i]
                        << ", expected " << br.sorted[(size_t)i].id;
                    *error = oss.str();
                }
                return false;
            }

            const float exp_w = pfs_oracle_expf(br.sorted[(size_t)i].score - max_score) / denom;
            const float diff = fabsf(row_probs[i] - exp_w);
            const float tol = PFS_PROB_ATOL + PFS_PROB_RTOL * fabsf(exp_w);

            if (!(diff <= tol)) {
                if (error) {
                    std::ostringstream oss;
                    oss << "packed_cand_prob mismatch at row=" << row
                        << ", i=" << i
                        << ": got " << row_probs[i]
                        << ", expected " << exp_w
                        << ", diff=" << diff
                        << ", tol=" << tol;
                    *error = oss.str();
                }
                return false;
            }

            sum = pfs_oracle_fadd(sum, row_probs[i]);
        }

        const float sum_diff = fabsf(sum - 1.0f);
        if (!(sum_diff <= 5.0e-4f)) {
            if (error) {
                std::ostringstream oss;
                oss << "probability sum mismatch at row=" << row
                    << ": sum=" << sum;
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<float>& d_logits,
    const DeviceBuffer<int32_t>& d_history_token,
    const DeviceBuffer<int32_t>& d_history_len,
    const DeviceBuffer<float>& d_rp,
    const DeviceBuffer<float>& d_fp,
    const DeviceBuffer<float>& d_pp,
    const DeviceBuffer<float>& d_temp,
    const DeviceBuffer<float>& d_min_p,
    const DeviceBuffer<float>& d_uniform,
    std::string* error) {
    if (d_logits.download() != hc.logits) {
        if (error) *error = "input logits modified";
        return false;
    }
    if (d_history_token.download() != hc.history_token) {
        if (error) *error = "input history_token modified";
        return false;
    }
    if (d_history_len.download() != hc.history_len) {
        if (error) *error = "input history_len modified";
        return false;
    }
    if (d_rp.download() != hc.repetition_penalty) {
        if (error) *error = "input repetition_penalty modified";
        return false;
    }
    if (d_fp.download() != hc.frequency_penalty) {
        if (error) *error = "input frequency_penalty modified";
        return false;
    }
    if (d_pp.download() != hc.presence_penalty) {
        if (error) *error = "input presence_penalty modified";
        return false;
    }
    if (d_temp.download() != hc.temperature) {
        if (error) *error = "input temperature modified";
        return false;
    }
    if (d_min_p.download() != hc.min_p) {
        if (error) *error = "input min_p modified";
        return false;
    }
    if (d_uniform.download() != hc.uniform_u) {
        if (error) *error = "input uniform_u modified";
        return false;
    }
    return true;
}

static bool run_one_case(const HostCase& hc, std::string* error) {
    PfsProblemSpec spec = {};
    spec.abi_version = PFS_ABI_VERSION;
    spec.max_B = hc.run.B;
    spec.max_V = hc.run.V;
    spec.max_H = hc.run.H;
    spec.flags = 0;

    size_t workspace_bytes = solution_workspace_bytes(&spec); if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/
    if (workspace_bytes == 0) {
        if (error) *error = "solution_workspace_bytes returned 0";
        return false;
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(workspace_bytes);

    DeviceBuffer<float> d_logits;
    DeviceBuffer<int32_t> d_history_token;
    DeviceBuffer<int32_t> d_history_len;
    DeviceBuffer<float> d_rp;
    DeviceBuffer<float> d_fp;
    DeviceBuffer<float> d_pp;
    DeviceBuffer<float> d_temp;
    DeviceBuffer<float> d_min_p;
    DeviceBuffer<float> d_uniform;

    d_logits.allocate(hc.logits.size());
    d_history_token.allocate(hc.history_token.size());
    d_history_len.allocate(hc.history_len.size());
    d_rp.allocate(hc.repetition_penalty.size());
    d_fp.allocate(hc.frequency_penalty.size());
    d_pp.allocate(hc.presence_penalty.size());
    d_temp.allocate(hc.temperature.size());
    d_min_p.allocate(hc.min_p.size());
    d_uniform.allocate(hc.uniform_u.size());

    d_logits.upload(hc.logits);
    d_history_token.upload(hc.history_token);
    d_history_len.upload(hc.history_len);
    d_rp.upload(hc.repetition_penalty);
    d_fp.upload(hc.frequency_penalty);
    d_pp.upload(hc.presence_penalty);
    d_temp.upload(hc.temperature);
    d_min_p.upload(hc.min_p);
    d_uniform.upload(hc.uniform_u);

    GuardedDeviceBuffer<int32_t> d_selected_token;
    GuardedDeviceBuffer<int32_t> d_survivor_count;
    GuardedDeviceBuffer<int32_t> d_packed_cand_token;
    GuardedDeviceBuffer<float> d_packed_cand_prob;

    d_selected_token.allocate((size_t)hc.run.B);
    d_survivor_count.allocate((size_t)hc.run.B);
    d_packed_cand_token.allocate((size_t)hc.run.B * (size_t)hc.run.V);
    d_packed_cand_prob.allocate((size_t)hc.run.B * (size_t)hc.run.V);

    PfsInputs inputs = {};
    inputs.logits = d_logits.ptr;
    inputs.history_token = d_history_token.ptr;
    inputs.history_len = d_history_len.ptr;
    inputs.repetition_penalty = d_rp.ptr;
    inputs.frequency_penalty = d_fp.ptr;
    inputs.presence_penalty = d_pp.ptr;
    inputs.temperature = d_temp.ptr;
    inputs.min_p = d_min_p.ptr;
    inputs.uniform_u = d_uniform.ptr;

    PfsOutputs outputs = {};
    outputs.selected_token = d_selected_token.ptr;
    outputs.survivor_count = d_survivor_count.ptr;
    outputs.packed_cand_token = d_packed_cand_token.ptr;
    outputs.packed_cand_prob = d_packed_cand_prob.ptr;

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(solution_run(
        state,
        &hc.run,
        &inputs,
        &outputs,
        workspace.ptr,
        workspace_bytes,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(
            hc,
            d_logits,
            d_history_token,
            d_history_len,
            d_rp,
            d_fp,
            d_pp,
            d_temp,
            d_min_p,
            d_uniform,
            error)) {
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return false;
    }

    if (!d_selected_token.check_guards("selected_token", error) ||
        !d_survivor_count.check_guards("survivor_count", error) ||
        !d_packed_cand_token.check_guards("packed_cand_token", error) ||
        !d_packed_cand_prob.check_guards("packed_cand_prob", error)) {
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return false;
    }

    const std::vector<int32_t> h_selected_token = d_selected_token.download_data();
    const std::vector<int32_t> h_survivor_count = d_survivor_count.download_data();
    const std::vector<int32_t> h_packed_cand_token = d_packed_cand_token.download_data();
    const std::vector<float> h_packed_cand_prob = d_packed_cand_prob.download_data();

    PfsHostOutputsView got = {};
    got.selected_token = h_selected_token.data();
    got.survivor_count = h_survivor_count.data();
    got.packed_cand_token = h_packed_cand_token.data();
    got.packed_cand_prob = h_packed_cand_prob.data();

    const bool ok = tolerant_check_outputs(hc, got, error);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));

    return ok;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int passed = 0;
        const int total = static_cast<int>(cases.size());

        for (const HostCase& hc : cases) {
            std::string error;
            bool ok = false;

            try {
                ok = run_one_case(hc, &error);
            } catch (const std::exception& ex) {
                error = ex.what();
                ok = false;
            }

            if (ok) {
                ++passed;
                std::printf(
                    "case %-42s PASS  B=%d V=%d H=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.B,
                    hc.run.V,
                    hc.run.H,
                    hc.run.distribution_id);
            } else {
                std::printf("case %-42s FAIL  %s\n", hc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
