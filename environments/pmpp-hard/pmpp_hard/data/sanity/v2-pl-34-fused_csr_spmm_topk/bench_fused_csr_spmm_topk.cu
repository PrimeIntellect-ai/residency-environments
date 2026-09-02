// file: bench_fused_csr_spmm_topk.cu

#include "fused_csr_spmm_topk_common.h"

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
        uint64_t z = (state += 0x8e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    int uniform_int(int lo, int hi) {
        return lo + static_cast<int>(next_u64() % static_cast<uint64_t>(hi - lo + 1));
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
        if (n > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
    }

    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("DeviceBuffer upload size mismatch");
        if (count > 0) CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
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

struct DeviceCase {
    HostCase host;

    DeviceBuffer<int32_t> row_offsets;
    DeviceBuffer<int32_t> col_indices;
    DeviceBuffer<int32_t> vals;
    DeviceBuffer<int32_t> dense_b;

    DeviceBuffer<int32_t> topm_cols;
    DeviceBuffer<int64_t> topm_vals;
    DeviceBuffer<int32_t> topm_count;
    DeviceBuffer<int64_t> row_sum;
    DeviceBuffer<int64_t> row_max;
    DeviceBuffer<int32_t> row_argmax;
    DeviceBuffer<int32_t> row_nnz;
    DeviceBuffer<int32_t> row_nnz_prefix;

    FcstInputs inputs;
    FcstOutputs outputs;
};

static int choose_degree(SplitMix64& rng, int row, int K, int avg, int dist) {
    if (dist == FCST_DIST_POWERLAW || dist == FCST_DIST_ROW_HOT) {
        if (row == 0) return std::min(K * 2, avg * 32);
        if ((row % 257) == 0) return std::min(K, avg * 16);
        if ((row % 17) == 0) return std::max(1, avg * 2);
        return std::max(1, avg / 2 + rng.uniform_int(0, std::max(1, avg)));
    }

    if (dist == FCST_DIST_MANY_TIES) return avg;

    return std::max(0, avg + rng.uniform_int(-std::max(1, avg / 2), std::max(1, avg / 2)));
}

static int32_t sparse_val(SplitMix64& rng, int edge, int dist) {
    if (dist == FCST_DIST_MANY_TIES) {
        static const int32_t vals[] = {-2, -1, 0, 1, 2};
        return vals[edge % 5];
    }

    if ((edge % 257) == 0) return 17;
    if ((edge % 389) == 0) return -19;

    return rng.uniform_int(-8, 8);
}

static int32_t dense_val(SplitMix64& rng, int idx, int dist) {
    if (dist == FCST_DIST_MANY_TIES) {
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
    int dist,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = FCST_ABI_VERSION;
    hc.run.rows = rows;
    hc.run.K = K;
    hc.run.N = N;
    hc.run.M = M;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = dist;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    SplitMix64 rng(g_state ^ seed);

    hc.row_offsets.assign((size_t)rows + 1, 0);

    for (int r = 0; r < rows; ++r) {
        hc.row_offsets[(size_t)r] = static_cast<int32_t>(hc.col_indices.size());

        int deg = choose_degree(rng, r, K, avg_nnz, dist);
        if ((r % 997) == 0) deg = 0;

        for (int j = 0; j < deg; ++j) {
            int col = 0;
            if (dist == FCST_DIST_ROW_HOT && r == 0) col = j % K;
            else col = rng.uniform_int(0, K - 1);

            hc.col_indices.push_back(col);
            hc.vals.push_back(sparse_val(rng, static_cast<int>(hc.vals.size()), dist));
        }
    }

    hc.row_offsets[(size_t)rows] = static_cast<int32_t>(hc.col_indices.size());
    hc.run.nnz = static_cast<int32_t>(hc.col_indices.size());

    if (!fcst_validate_run_spec(&hc.run)) throw std::runtime_error("invalid FcstRunSpec");

    hc.dense_b.resize((size_t)K * (size_t)N);
    for (size_t i = 0; i < hc.dense_b.size(); ++i) {
        hc.dense_b[i] = dense_val(rng, static_cast<int>(i), dist);
    }

    return hc;
}

static std::vector<HostCase> build_bench_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x200000001b3ULL;

    cases.push_back(make_case(
        "rows8192_N256_avg16_M16",
        8192, 2048, 256, 16, 16, FCST_DIST_POWERLAW, s++));

    cases.push_back(make_case(
        "rows2048_N1024_avg16_M16",
        2048, 1024, 1024, 16, 16, FCST_DIST_POWERLAW, s++));

    cases.push_back(make_case(
        "rows1024_N4096_avg4_M4",
        1024, 1024, 4096, 4, 4, FCST_DIST_UNIFORM, s++));

    cases.push_back(make_case(
        "rows2048_N256_avg64_M64",
        2048, 2048, 256, 64, 64, FCST_DIST_ROW_HOT, s++));

    return cases;
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    dc->row_offsets.allocate(hc.row_offsets.size());
    dc->col_indices.allocate(hc.col_indices.size());
    dc->vals.allocate(hc.vals.size());
    dc->dense_b.allocate(hc.dense_b.size());

    dc->row_offsets.upload(hc.row_offsets);
    dc->col_indices.upload(hc.col_indices);
    dc->vals.upload(hc.vals);
    dc->dense_b.upload(hc.dense_b);

    dc->topm_cols.allocate((size_t)hc.run.rows * (size_t)hc.run.M);
    dc->topm_vals.allocate((size_t)hc.run.rows * (size_t)hc.run.M);
    dc->topm_count.allocate((size_t)hc.run.rows);
    dc->row_sum.allocate((size_t)hc.run.rows);
    dc->row_max.allocate((size_t)hc.run.rows);
    dc->row_argmax.allocate((size_t)hc.run.rows);
    dc->row_nnz.allocate((size_t)hc.run.rows);
    dc->row_nnz_prefix.allocate((size_t)hc.run.rows + 1);

    dc->inputs = {};
    dc->inputs.row_offsets = dc->row_offsets.ptr;
    dc->inputs.col_indices = dc->col_indices.ptr;
    dc->inputs.vals = dc->vals.ptr;
    dc->inputs.dense_b = dc->dense_b.ptr;

    dc->outputs = {};
    dc->outputs.topm_cols = dc->topm_cols.ptr;
    dc->outputs.topm_vals = dc->topm_vals.ptr;
    dc->outputs.topm_count = dc->topm_count.ptr;
    dc->outputs.row_sum = dc->row_sum.ptr;
    dc->outputs.row_max = dc->row_max.ptr;
    dc->outputs.row_argmax = dc->row_argmax.ptr;
    dc->outputs.row_nnz = dc->row_nnz.ptr;
    dc->outputs.row_nnz_prefix = dc->row_nnz_prefix.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        std::vector<HostCase> host_cases = build_bench_cases();

        int max_rows = FCST_MIN_ROWS;
        int max_K = FCST_MIN_K;
        int max_N = FCST_MIN_N;
        int max_nnz = 0;
        int max_M = 1;

        for (const HostCase& hc : host_cases) {
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

        const size_t workspace_bytes = solution_workspace_bytes(&spec);
        if (workspace_bytes == 0) throw std::runtime_error("solution_workspace_bytes returned 0");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(workspace_bytes);

        std::vector<DeviceCase*> cases;
        cases.reserve(host_cases.size());

        for (const HostCase& hc : host_cases) {
            cases.push_back(make_device_case(hc));
            std::printf(
                "bench_case %-28s rows=%d K=%d N=%d nnz=%d M=%d dist=%d\n",
                hc.name.c_str(),
                hc.run.rows,
                hc.run.K,
                hc.run.N,
                hc.run.nnz,
                hc.run.M,
                hc.run.distribution_id);
        }

        for (int warm = 0; warm < 5; ++warm) {
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

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, stream));

        for (int iter = 0; iter < iters; ++iter) {
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

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        const double calls = static_cast<double>(iters) * static_cast<double>(cases.size());
        std::printf("avg_ms=%.6f\n", static_cast<double>(elapsed_ms) / calls);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (DeviceCase* dc : cases) delete dc;

        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));

        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
