// file: test_fused_csr_spmm_topk.cu

#include "fused_csr_spmm_topk_common.h"
#include "fused_csr_spmm_topk_oracle.hpp"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0xe3a91c52bb740f19ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                              \
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

    int32_t next_i32() {
        return static_cast<int32_t>(next_u64() >> 32);
    }

    int uniform_int(int lo, int hi) {
        const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u64() % span);
    }

    bool chance_permille(int p) {
        if (p <= 0) return false;
        if (p >= 1000) return true;
        return static_cast<int>(next_u64() % 1000ULL) < p;
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
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
        }
    }

    void upload(const std::vector<T>& host) {
        if (host.size() != count) {
            throw std::runtime_error("DeviceBuffer upload size mismatch");
        }
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
        }
    }

    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count > 0) {
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

    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

    ~GuardedDeviceBuffer() {
        if (raw) cudaFree(raw);
    }

    void allocate(size_t n) {
        count = n;
        data_bytes = sizeof(T) * count;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), kGuardBytes + data_bytes + kGuardBytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, kGuardBytes + data_bytes + kGuardBytes));
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }

    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        }
        return host;
    }

    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> left(kGuardBytes);
        std::vector<uint8_t> right(kGuardBytes);

        CUDA_CHECK(cudaMemcpy(left.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(right.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (left[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "left guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }

            if (right[i] != kGuardByte) {
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
    FcstRunSpec run;
    std::vector<int32_t> row_offsets;
    std::vector<int32_t> col_indices;
    std::vector<int32_t> vals;
    std::vector<int32_t> dense_b;
};

static int choose_degree(
    SplitMix64& rng,
    int row,
    int rows,
    int K,
    int avg,
    int distribution_id,
    bool force_empty_rows,
    bool force_nnz_gt_k) {
    if (force_empty_rows && (row % 97) == 0) return 0;

    if (force_nnz_gt_k && row == 0) return K + 32;

    if (distribution_id == FCST_DIST_POWERLAW || distribution_id == FCST_DIST_ROW_HOT) {
        if (row == 0) return std::min(K * 2, std::max(avg * 32, avg + 1));
        if ((row % 257) == 0) return std::min(K, std::max(avg * 16, avg + 1));
        if ((row % 17) == 0) return std::max(1, avg * 2);
        return std::max(0, avg / 2 + rng.uniform_int(0, std::max(1, avg)));
    }

    if (distribution_id == FCST_DIST_GRID) {
        int deg = 4;
        if (row == 0 || row == rows - 1) deg = 2;
        return std::min(K, deg);
    }

    return std::max(0, avg + rng.uniform_int(-std::max(1, avg / 2), std::max(1, avg / 2)));
}

static int32_t choose_sparse_val(SplitMix64& rng, int edge, int distribution_id, bool all_ties) {
    if (all_ties) return 1;

    if (distribution_id == FCST_DIST_MANY_TIES) {
        static const int32_t vals[] = {-2, -1, 0, 1, 2};
        return vals[edge % 5];
    }

    if ((edge % 257) == 0) return 17;
    if ((edge % 389) == 0) return -19;

    return rng.uniform_int(-8, 8);
}

static int32_t choose_dense_val(SplitMix64& rng, int idx, int distribution_id, bool all_ties) {
    if (all_ties) return 0;

    if (distribution_id == FCST_DIST_MANY_TIES) {
        static const int32_t vals[] = {-3, -1, 0, 1, 3};
        return vals[(idx / 7) % 5];
    }

    if ((idx % 313) == 0) return 11;
    if ((idx % 509) == 0) return -13;

    return rng.uniform_int(-7, 7);
}

static HostCase make_case(
    const char* name,
    int rows,
    int K,
    int N,
    int avg_nnz,
    int M,
    int distribution_id,
    uint64_t seed,
    bool force_empty_rows,
    bool force_nnz_gt_k,
    bool all_ties,
    bool include_invalid_cols) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = FCST_ABI_VERSION;
    hc.run.rows = rows;
    hc.run.K = K;
    hc.run.N = N;
    hc.run.M = M;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    SplitMix64 rng(g_state ^ seed);

    hc.row_offsets.assign((size_t)rows + 1, 0);

    for (int r = 0; r < rows; ++r) {
        hc.row_offsets[(size_t)r] = static_cast<int32_t>(hc.col_indices.size());

        int deg = choose_degree(
            rng,
            r,
            rows,
            K,
            avg_nnz,
            distribution_id,
            force_empty_rows,
            force_nnz_gt_k);

        if (deg < 0) deg = 0;

        for (int j = 0; j < deg; ++j) {
            int col = 0;

            if (include_invalid_cols && ((j + r) % 113) == 0) {
                col = (j & 1) ? -1 : (K + 7);
            } else if (distribution_id == FCST_DIST_GRID) {
                col = (r + j * 17 + 3) % K;
            } else if (force_nnz_gt_k && r == 0) {
                col = j % K;
            } else {
                col = rng.uniform_int(0, K - 1);
            }

            hc.col_indices.push_back(col);
            hc.vals.push_back(choose_sparse_val(rng, static_cast<int>(hc.vals.size()), distribution_id, all_ties));
        }
    }

    hc.row_offsets[(size_t)rows] = static_cast<int32_t>(hc.col_indices.size());
    hc.run.nnz = static_cast<int32_t>(hc.col_indices.size());

    if (!fcst_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid FcstRunSpec generated");
    }

    hc.dense_b.resize((size_t)K * (size_t)N);
    for (size_t i = 0; i < hc.dense_b.size(); ++i) {
        hc.dense_b[i] = choose_dense_val(rng, static_cast<int>(i), distribution_id, all_ties);
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_case(
        "uniform_rows1024_N256_avg4_M4",
        1024, 512, 256, 4, 4, FCST_DIST_UNIFORM, s++,
        true, false, false, false));

    cases.push_back(make_case(
        "powerlaw_rows2048_N1024_avg16_M16",
        2048, 1024, 1024, 16, 16, FCST_DIST_POWERLAW, s++,
        false, false, false, false));

    cases.push_back(make_case(
        "large_rows65536_N256_avg4_M16",
        65536, 8192, 256, 4, 16, FCST_DIST_ROW_HOT, s++,
        true, false, false, true));

    cases.push_back(make_case(
        "many_ties_rows1024_N4096_avg4_M64",
        1024, 512, 4096, 4, 64, FCST_DIST_MANY_TIES, s++,
        false, false, true, false));

    cases.push_back(make_case(
        "grid_rows4096_N256_avg16_M16",
        4096, 1024, 256, 16, 16, FCST_DIST_GRID, s++,
        true, false, false, true));

    cases.push_back(make_case(
        "edge_M_gt_N_rows1024_N32_M64",
        1024, 256, 32, 16, 64, FCST_DIST_UNIFORM, s++,
        true, false, false, false));

    cases.push_back(make_case(
        "edge_nnz_gt_K_rows1024_N256_M64",
        1024, 64, 256, 64, 64, FCST_DIST_POWERLAW, s++,
        false, true, false, false));

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<int32_t>& d_row_offsets,
    const DeviceBuffer<int32_t>& d_col_indices,
    const DeviceBuffer<int32_t>& d_vals,
    const DeviceBuffer<int32_t>& d_dense_b,
    std::string* error) {
    if (d_row_offsets.download() != hc.row_offsets) {
        if (error) *error = "input row_offsets modified";
        return false;
    }

    if (d_col_indices.download() != hc.col_indices) {
        if (error) *error = "input col_indices modified";
        return false;
    }

    if (d_vals.download() != hc.vals) {
        if (error) *error = "input vals modified";
        return false;
    }

    if (d_dense_b.download() != hc.dense_b) {
        if (error) *error = "input dense_b modified";
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
    DeviceBuffer<int32_t> d_row_offsets;
    DeviceBuffer<int32_t> d_col_indices;
    DeviceBuffer<int32_t> d_vals;
    DeviceBuffer<int32_t> d_dense_b;

    d_row_offsets.allocate(hc.row_offsets.size());
    d_col_indices.allocate(hc.col_indices.size());
    d_vals.allocate(hc.vals.size());
    d_dense_b.allocate(hc.dense_b.size());

    d_row_offsets.upload(hc.row_offsets);
    d_col_indices.upload(hc.col_indices);
    d_vals.upload(hc.vals);
    d_dense_b.upload(hc.dense_b);

    GuardedDeviceBuffer<int32_t> d_topm_cols;
    GuardedDeviceBuffer<int64_t> d_topm_vals;
    GuardedDeviceBuffer<int32_t> d_topm_count;
    GuardedDeviceBuffer<int64_t> d_row_sum;
    GuardedDeviceBuffer<int64_t> d_row_max;
    GuardedDeviceBuffer<int32_t> d_row_argmax;
    GuardedDeviceBuffer<int32_t> d_row_nnz;
    GuardedDeviceBuffer<int32_t> d_row_nnz_prefix;

    d_topm_cols.allocate((size_t)hc.run.rows * (size_t)hc.run.M);
    d_topm_vals.allocate((size_t)hc.run.rows * (size_t)hc.run.M);
    d_topm_count.allocate((size_t)hc.run.rows);
    d_row_sum.allocate((size_t)hc.run.rows);
    d_row_max.allocate((size_t)hc.run.rows);
    d_row_argmax.allocate((size_t)hc.run.rows);
    d_row_nnz.allocate((size_t)hc.run.rows);
    d_row_nnz_prefix.allocate((size_t)hc.run.rows + 1);

    FcstInputs inputs = {};
    inputs.row_offsets = d_row_offsets.ptr;
    inputs.col_indices = d_col_indices.ptr;
    inputs.vals = d_vals.ptr;
    inputs.dense_b = d_dense_b.ptr;

    FcstOutputs outputs = {};
    outputs.topm_cols = d_topm_cols.ptr;
    outputs.topm_vals = d_topm_vals.ptr;
    outputs.topm_count = d_topm_count.ptr;
    outputs.row_sum = d_row_sum.ptr;
    outputs.row_max = d_row_max.ptr;
    outputs.row_argmax = d_row_argmax.ptr;
    outputs.row_nnz = d_row_nnz.ptr;
    outputs.row_nnz_prefix = d_row_nnz_prefix.ptr;

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

    if (!check_input_unchanged(hc, d_row_offsets, d_col_indices, d_vals, d_dense_b, error)) {
        return false;
    }

    if (!d_topm_cols.check_guards("topm_cols", error)) return false;
    if (!d_topm_vals.check_guards("topm_vals", error)) return false;
    if (!d_topm_count.check_guards("topm_count", error)) return false;
    if (!d_row_sum.check_guards("row_sum", error)) return false;
    if (!d_row_max.check_guards("row_max", error)) return false;
    if (!d_row_argmax.check_guards("row_argmax", error)) return false;
    if (!d_row_nnz.check_guards("row_nnz", error)) return false;
    if (!d_row_nnz_prefix.check_guards("row_nnz_prefix", error)) return false;

    const std::vector<int32_t> h_topm_cols = d_topm_cols.download_data();
    const std::vector<int64_t> h_topm_vals = d_topm_vals.download_data();
    const std::vector<int32_t> h_topm_count = d_topm_count.download_data();
    const std::vector<int64_t> h_row_sum = d_row_sum.download_data();
    const std::vector<int64_t> h_row_max = d_row_max.download_data();
    const std::vector<int32_t> h_row_argmax = d_row_argmax.download_data();
    const std::vector<int32_t> h_row_nnz = d_row_nnz.download_data();
    const std::vector<int32_t> h_row_nnz_prefix = d_row_nnz_prefix.download_data();

    FcstHostInputsView host_inputs = {};
    host_inputs.row_offsets = hc.row_offsets.data();
    host_inputs.col_indices = hc.col_indices.data();
    host_inputs.vals = hc.vals.data();
    host_inputs.dense_b = hc.dense_b.data();

    FcstExpected expected;
    fcst_cpu_oracle(hc.run, host_inputs, &expected);

    FcstHostOutputsView got = {};
    got.topm_cols = h_topm_cols.data();
    got.topm_vals = h_topm_vals.data();
    got.topm_count = h_topm_count.data();
    got.row_sum = h_row_sum.data();
    got.row_max = h_row_max.data();
    got.row_argmax = h_row_argmax.data();
    got.row_nnz = h_row_nnz.data();
    got.row_nnz_prefix = h_row_nnz_prefix.data();

    return fcst_check_all_outputs(hc.run, expected, got, error);
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::vector<HostCase> cases = build_test_cases();

        int max_rows = FCST_MIN_ROWS;
        int max_K = FCST_MIN_K;
        int max_N = FCST_MIN_N;
        int max_nnz = 0;
        int max_M = 1;

        for (const HostCase& hc : cases) {
            max_rows = std::max(max_rows, hc.run.rows);
            max_K = std::max(max_K, hc.run.K);
            max_N = std::max(max_N, hc.run.N);
            max_nnz = std::max(max_nnz, hc.run.nnz);
            max_M = std::max(max_M, hc.run.M);
        }

        FcstProblemSpec spec = {};
        spec.abi_version = FCST_ABI_VERSION;
        spec.max_rows = max_rows;
        spec.max_K = max_K;
        spec.max_N = max_N;
        spec.max_nnz = max_nnz;
        spec.max_M = max_M;
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
                    "case %-40s PASS  rows=%d K=%d N=%d nnz=%d M=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.rows,
                    hc.run.K,
                    hc.run.N,
                    hc.run.nnz,
                    hc.run.M,
                    hc.run.distribution_id);
            } else {
                std::printf("case %-40s FAIL  %s\n", hc.name.c_str(), error.c_str());
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
