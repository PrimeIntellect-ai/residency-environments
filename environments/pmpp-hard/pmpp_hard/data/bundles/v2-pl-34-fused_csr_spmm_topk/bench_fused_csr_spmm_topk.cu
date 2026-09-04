// file: bench_fused_csr_spmm_topk.cu

#include "fused_csr_spmm_topk_common.h"
#include "pmpp_bench_digest.cuh"

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
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
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

// --- C3 timed-fold (patterns I2 + P2): every timed call sees fresh input
// values and contributes its graded outputs to out_fnv, so a cached/no-op
// timed call cannot match the reference digest. The CSR structure
// (row_offsets/col_indices) stays fixed; vals and dense_b are regenerated
// in-place per iteration from (PMPP_BENCH_SEED, case, iter), preserving the
// generator's sentinel positions and value ranges (in-family data).

__device__ __forceinline__ uint64_t c3_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__global__ void c3_mutate_fcst(int32_t* vals, long long nnz, int32_t* dense,
                               long long dn, uint64_t stream_key, int dist) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < nnz + dn; i += stride) {
        const uint64_t h =
            c3_mix(stream_key ^ (uint64_t)(i + 1) * 0x9e3779b97f4a7c15ULL);
        if (i < nnz) {
            int32_t v;
            if (dist == FCST_DIST_MANY_TIES) {
                const int32_t tab[5] = {-2, -1, 0, 1, 2};
                v = tab[(i + (long long)(stream_key % 5)) % 5];
            } else if ((i % 257) == 0) {
                v = 17;
            } else if ((i % 389) == 0) {
                v = -19;
            } else {
                v = (int32_t)(h % 17) - 8;
            }
            vals[i] = v;
        } else {
            const long long j = i - nnz;
            int32_t v;
            if (dist == FCST_DIST_MANY_TIES) {
                const int32_t tab[5] = {-3, -1, 0, 1, 3};
                v = tab[((j / 7) + (long long)(stream_key % 5)) % 5];
            } else if ((j % 313) == 0) {
                v = 11;
            } else if ((j % 509) == 0) {
                v = -13;
            } else {
                v = (int32_t)(h % 15) - 7;
            }
            dense[j] = v;
        }
    }
}

struct C3ProbeDesc {
    const uint8_t* ptr;
    unsigned long long nbytes;
};

struct C3ProbeTable {
    C3ProbeDesc d[8];
    int n;
};

// One launch per timed call: xor-folds a position-salted mix of every graded
// output word into acc[slot]. xor accumulation is order-independent, so the
// result is deterministic; a stale/no-op timed call breaks the digest.
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
        const unsigned long long nwords = d.nbytes >> 3;
        const uint64_t bsalt = salt ^ (uint64_t)(b + 1) * 0x100000001b3ULL;
        const unsigned long long* w64 =
            reinterpret_cast<const unsigned long long*>(d.ptr);
        for (unsigned long long w = gtid; w < nwords; w += gstride) {
            local ^= c3_mix(w64[w] ^ c3_mix(bsalt + w));
        }
        if (gtid == 0 && (d.nbytes & 7ULL)) {
            unsigned long long tail = 0;
            for (unsigned long long i = nwords << 3; i < d.nbytes; ++i) {
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
    C3ProbeTable probe;
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
    // Case data values derive from PMPP_BENCH_SEED so graded outputs are not
    // precomputable offline; shapes stay fixed so timing remains comparable.
    uint64_t s = 0x200000001b3ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL);

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

    // C3 probe table: exactly the graded output set (all graded full-range;
    // mirrors the digest fold below).
    const unsigned long long rows = (unsigned long long)hc.run.rows;
    const unsigned long long M = (unsigned long long)hc.run.M;
    C3ProbeTable& pt = dc->probe;
    pt.n = 8;
    pt.d[0] = {(const uint8_t*)dc->topm_cols.ptr, rows * M * sizeof(int32_t)};
    pt.d[1] = {(const uint8_t*)dc->topm_vals.ptr, rows * M * sizeof(int64_t)};
    pt.d[2] = {(const uint8_t*)dc->topm_count.ptr, rows * sizeof(int32_t)};
    pt.d[3] = {(const uint8_t*)dc->row_sum.ptr, rows * sizeof(int64_t)};
    pt.d[4] = {(const uint8_t*)dc->row_max.ptr, rows * sizeof(int64_t)};
    pt.d[5] = {(const uint8_t*)dc->row_argmax.ptr, rows * sizeof(int32_t)};
    pt.d[6] = {(const uint8_t*)dc->row_nnz.ptr, rows * sizeof(int32_t)};
    pt.d[7] = {(const uint8_t*)dc->row_nnz_prefix.ptr,
               (rows + 1) * sizeof(int32_t)};

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
                const long long nnz = (long long)dc->host.vals.size();
                const long long dn = (long long)dc->host.dense_b.size();
                const int blocks =
                    (int)std::min<long long>((nnz + dn + 255) / 256, 4096);
                c3_mutate_fcst<<<blocks, 256, 0, stream>>>(
                    dc->vals.ptr, nnz, dc->dense_b.ptr, dn, skey,
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
        std::printf("avg_ms=%.6f\n", static_cast<double>(elapsed_ms) / calls);

        // Untimed graded-output digest: one more run per case, then fold every
        // graded output array (all graded over their full ranges) in case order.
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
            const size_t rows = (size_t)dc->host.run.rows;
            const size_t M = (size_t)dc->host.run.M;
            dg.dev(dc->topm_cols.ptr, rows * M * sizeof(int32_t));
            dg.dev(dc->topm_vals.ptr, rows * M * sizeof(int64_t));
            dg.dev(dc->topm_count.ptr, rows * sizeof(int32_t));
            dg.dev(dc->row_sum.ptr, rows * sizeof(int64_t));
            dg.dev(dc->row_max.ptr, rows * sizeof(int64_t));
            dg.dev(dc->row_argmax.ptr, rows * sizeof(int32_t));
            dg.dev(dc->row_nnz.ptr, rows * sizeof(int32_t));
            dg.dev(dc->row_nnz_prefix.ptr, (rows + 1) * sizeof(int32_t));
        }
        dg.print();

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
