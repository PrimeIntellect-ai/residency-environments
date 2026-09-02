// ============================================================================
// file: bench_mk_priority_preempt.cu
// ============================================================================

#include "mk_priority_preempt_common.h"

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
        return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1));
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
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * host.size(),
                              cudaMemcpyHostToDevice));
    }
};

struct BenchCase {
    std::string name;
    MkpRunSpec run;
    std::vector<int8_t> X;
    std::vector<int32_t> jarrival, jclass, jlen;
    std::vector<int8_t> jw;
};

static BenchCase make_case(
    const char* name, int M, int K, int Q, int R, int quantum, int njobs,
    int arr_max, int len_lo, int len_hi, int dist, uint64_t seed) {
    BenchCase bc;
    bc.name = name;
    bc.run = {};
    bc.run.abi_version = MKP_ABI_VERSION;
    bc.run.M = M;
    bc.run.K = K;
    bc.run.Q = Q;
    bc.run.R = R;
    bc.run.quantum = quantum;
    bc.run.njobs = njobs;
    bc.run.seed_id = (int32_t)(seed & 0x7fffffff);
    bc.run.distribution_id = dist;
    bc.run.case_id = (int32_t)((seed >> 32) & 0x7fffffff);
    if (!mkp_validate_run_spec(&bc.run)) {
        throw std::runtime_error(std::string("invalid run spec: ") + name);
    }
    SplitMix64 rng(g_state ^ seed);
    bc.X.resize((size_t)M * K);
    for (size_t i = 0; i < bc.X.size(); ++i) {
        bc.X[i] = (int8_t)rng.uniform_int(-127, 127);
    }
    bc.jarrival.resize((size_t)njobs);
    bc.jclass.resize((size_t)njobs);
    bc.jlen.resize((size_t)njobs);
    bc.jw.resize((size_t)njobs * K);
    for (int j = 0; j < njobs; ++j) {
        bc.jarrival[(size_t)j] = rng.uniform_int(0, arr_max);
        bc.jclass[(size_t)j] = dist == MKP_DIST_LONGTAIL && j < njobs / 8
            ? Q - 1
            : rng.uniform_int(0, Q - 1);
        bc.jlen[(size_t)j] = dist == MKP_DIST_LONGTAIL && j < njobs / 8
            ? len_hi * 2 < MKP_MAX_LEN ? len_hi * 2 : MKP_MAX_LEN
            : rng.uniform_int(len_lo, len_hi);
    }
    for (size_t i = 0; i < bc.jw.size(); ++i) {
        bc.jw[i] = (int8_t)rng.uniform_int(-127, 127);
    }
    return bc;
}

static std::vector<BenchCase> build_bench_cases() {
    std::vector<BenchCase> cases;
    uint64_t s = 0x600000001b3ULL;
    cases.push_back(make_case(
        "mk_max_drain", 32768, 256, 3, 32, 4, 1024, 7, 16, 24,
        MKP_DIST_UNIFORM, s++));
    cases.push_back(make_case(
        "mk_longtail", 32768, 256, 4, 32, 8, 1024, 15, 4, 16,
        MKP_DIST_LONGTAIL, s++));
    cases.push_back(make_case(
        "mk_spread", 24576, 256, 4, 32, 2, 1024, 15, 16, 32,
        MKP_DIST_UNIFORM, s++));
    cases.push_back(make_case(
        "mk_burst", 16384, 256, 2, 16, 8, 1024, 7, 8, 16,
        MKP_DIST_UNIFORM, s++));
    cases.push_back(make_case(
        "mk_thrash_q1", 32768, 256, 2, 32, 1, 1024, 0, 4, 12,
        MKP_DIST_UNIFORM, s++));
    return cases;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

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

        const std::vector<BenchCase> cases = build_bench_cases();

        DeviceBuffer<int8_t> dX, djw;
        DeviceBuffer<int32_t> darr, dcls, dlen;
        dX.allocate((size_t)MKP_MAX_M * 256);
        djw.allocate((size_t)MKP_MAX_NJOBS * 256);
        darr.allocate(MKP_MAX_NJOBS);
        dcls.allocate(MKP_MAX_NJOBS);
        dlen.allocate(MKP_MAX_NJOBS);

        const size_t trace_cap = (size_t)MKP_MAX_R * MKP_MAX_JOBS_TOTAL;
        DeviceBuffer<int32_t> trace_len, queue_len;
        DeviceBuffer<uint64_t> trace, exec_digest, y, queue_dump, state_csum;
        trace_len.allocate(1);
        trace.allocate(trace_cap);
        exec_digest.allocate(trace_cap);
        y.allocate(MKP_MAX_M);
        queue_len.allocate(4);
        queue_dump.allocate(MKP_MAX_JOBS_TOTAL);
        state_csum.allocate(1);

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double total_ms = 0.0;
        double total_calls = 0.0;

        for (const BenchCase& bc : cases) {
            std::printf(
                "bench_case %-16s M=%d K=%d Q=%d R=%d q=%d njobs=%d dist=%d\n",
                bc.name.c_str(), bc.run.M, bc.run.K, bc.run.Q, bc.run.R,
                bc.run.quantum, bc.run.njobs, bc.run.distribution_id);

            dX.upload(bc.X);
            darr.upload(bc.jarrival);
            dcls.upload(bc.jclass);
            dlen.upload(bc.jlen);
            djw.upload(bc.jw);

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
            outputs.state_checksum = state_csum.ptr;

            // Each iteration is reset + run: the job-count cap per reset
            // epoch (MKP_MAX_JOBS_TOTAL) must hold for any iteration count.
            for (int warmup = 0; warmup < 5; ++warmup) {
                CUDA_CHECK(solution_reset(state, stream));
                CUDA_CHECK(solution_run(state, &bc.run, &inputs, &outputs,
                                        workspace.ptr, workspace_bytes,
                                        stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int iter = 0; iter < iters; ++iter) {
                CUDA_CHECK(solution_reset(state, stream));
                CUDA_CHECK(solution_run(state, &bc.run, &inputs, &outputs,
                                        workspace.ptr, workspace_bytes,
                                        stream));
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            std::printf("case_ms %-16s %.6f\n", bc.name.c_str(),
                        (double)elapsed_ms / iters);
            total_ms += (double)elapsed_ms;
            total_calls += (double)iters;
        }

        std::printf("avg_ms=%.6f\n", total_ms / total_calls);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
