// file: bench_cuckoo_tombstone_table.cu

#include "cuckoo_tombstone_table_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                   \
            throw std::runtime_error(_oss.str());                              \
        }                                                                       \
    } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t s) : state(s) {}
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
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) {
        count = n;
        if (n == 0) { ptr = nullptr; return; }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
    }
    void upload(const std::vector<T>& h) {
        if (h.size() != count) throw std::runtime_error("upload size mismatch");
        if (count) CUDA_CHECK(cudaMemcpy(ptr, h.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
    }
};

struct StepHost {
    CktRunSpec run;
    std::vector<int32_t> op_type;
    std::vector<int64_t> a0, a1;
};

struct DeviceStep {
    StepHost host;
    DeviceBuffer<int32_t> op_type;
    DeviceBuffer<int64_t> a0, a1;
    DeviceBuffer<int32_t> counts;
    DeviceBuffer<uint64_t> ev_h, rd_h, slot_h, stash_h, page_h;
    CktInputs inputs;
    CktOutputs outputs;
};

static CktProblemSpec make_bench_spec() {
    CktProblemSpec spec = {};
    spec.abi_version = CKT_ABI_VERSION;
    spec.slot_count = 512;
    spec.page_size = 16;
    spec.neighborhood = 8;
    spec.max_displacements_per_home = 16;
    spec.stash_capacity = 64;
    spec.max_tombstones = 64;
    spec.max_ops = 256;
    spec.max_steps = 48;
    spec.flags = 0;
    spec.seed0 = 0x1234567890abcdefULL;
    spec.seed1 = 0x0fedcba987654321ULL;
    if (!ckt_validate_problem_spec(&spec))
        throw std::runtime_error("invalid bench spec");
    return spec;
}

static std::vector<StepHost> build_sequence(const CktProblemSpec& spec) {
    SplitMix64 rng(0x9090909090909090ULL);
    std::vector<StepHost> steps;
    int read_id = 1;
    for (int s = 0; s < 48; ++s) {
        StepHost st;
        st.run = {};
        st.run.abi_version = CKT_ABI_VERSION;
        st.run.step_id = s;
        int n = (s == 13 || s == 31) ? 0 : 200;
        st.run.num_ops = n;
        const size_t rows = std::max<size_t>(1, (size_t)n);
        st.op_type.assign(rows, 0);
        st.a0.assign(rows, 0);
        st.a1.assign(rows, 0);
        for (int i = 0; i < n; ++i) {
            int op = rng.uniform_int(0, 9);
            int k = rng.uniform_int(0, 700);
            int pg = rng.uniform_int(0, 32);
            if (op <= 3) { st.op_type[i] = CKT_OP_PUT; st.a0[i] = k; st.a1[i] = (int64_t)rng.next_u64(); }
            else if (op <= 5) { st.op_type[i] = CKT_OP_GET; st.a0[i] = read_id++; st.a1[i] = k; }
            else if (op <= 7) { st.op_type[i] = CKT_OP_DELETE; st.a0[i] = k; }
            else if (op == 8) { st.op_type[i] = CKT_OP_PIN_PAGE; st.a0[i] = pg; }
            else { st.op_type[i] = CKT_OP_UNPIN_PAGE; st.a0[i] = pg; }
        }
        steps.push_back(st);
    }
    return steps;
}

static DeviceStep* make_device_step(const StepHost& h) {
    DeviceStep* ds = new DeviceStep();
    ds->host = h;
    ds->op_type.allocate(h.op_type.size());
    ds->a0.allocate(h.a0.size());
    ds->a1.allocate(h.a1.size());
    ds->op_type.upload(h.op_type);
    ds->a0.upload(h.a0);
    ds->a1.upload(h.a1);
    ds->counts.allocate(CKT_NUM_COUNTS);
    ds->ev_h.allocate(1); ds->rd_h.allocate(1); ds->slot_h.allocate(1);
    ds->stash_h.allocate(1); ds->page_h.allocate(1);
    ds->inputs = {};
    ds->inputs.op_type = ds->op_type.ptr;
    ds->inputs.a0 = ds->a0.ptr;
    ds->inputs.a1 = ds->a1.ptr;
    ds->outputs = {};
    ds->outputs.counts = ds->counts.ptr;
    ds->outputs.op_event_hash = ds->ev_h.ptr;
    ds->outputs.read_hash = ds->rd_h.ptr;
    ds->outputs.slot_state_hash = ds->slot_h.ptr;
    ds->outputs.stash_hash = ds->stash_h.ptr;
    ds->outputs.page_hash = ds->page_h.ptr;
    return ds;
}

static void run_sequence(void* state, void* workspace, size_t wbytes,
                         cudaStream_t stream, const std::vector<DeviceStep*>& steps) {
    for (DeviceStep* ds : steps) {
        CUDA_CHECK(solution_run(state, &ds->host.run, &ds->inputs, &ds->outputs,
                                workspace, wbytes, stream));
    }
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 20;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        const CktProblemSpec spec = make_bench_spec();
        size_t wbytes = solution_workspace_bytes(&spec);
        wbytes = std::max<size_t>(wbytes, 1);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(wbytes);

        std::vector<StepHost> host_steps = build_sequence(spec);
        std::vector<DeviceStep*> steps;
        for (const StepHost& h : host_steps) steps.push_back(make_device_step(h));

        std::printf("bench slot=%d page=%d nbhd=%d T=%zu\n",
                    spec.slot_count, spec.page_size, spec.neighborhood, steps.size());

        for (int w = 0; w < 3; ++w) {
            CUDA_CHECK(solution_reset(state, stream));
            run_sequence(state, workspace.ptr, wbytes, stream, steps);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start = nullptr, stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        double total_ms = 0.0;
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(start, stream));
            run_sequence(state, workspace.ptr, wbytes, stream, steps);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += (double)ms;
        }
        std::printf("avg_ms=%.6f\n", total_ms / (double)iters);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        for (DeviceStep* ds : steps) delete ds;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
