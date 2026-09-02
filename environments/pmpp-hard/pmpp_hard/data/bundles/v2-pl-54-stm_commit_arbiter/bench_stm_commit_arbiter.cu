// file: bench_stm_commit_arbiter.cu

#include "stm_commit_arbiter_common.h"
#include "pmpp_bench_digest.cuh"

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

static constexpr uint64_t g_state = 0x9b71c4d3a02f5e68ULL;

#define CUDA_CHECK(expr)                                                        \
    do { cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) { std::ostringstream _oss;                    \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                   \
            throw std::runtime_error(_oss.str()); } } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    int64_t next_i64() { return (int64_t)next_u64(); }
    int uniform_int(int lo, int hi) { return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1)); }
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr; size_t count = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) { count = n; if (!n) { ptr = nullptr; return; }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n)); }
    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("upload size mismatch");
        if (count) CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice)); }
};

struct StepHost {
    StmRunSpec run;
    std::vector<int32_t> op_kind;
    std::vector<uint64_t> txn_id, read_id, addr;
    std::vector<int64_t> value;
    std::vector<uint64_t> aux, watch_off, watch_addrs;
};

struct DeviceStep {
    StepHost host;
    DeviceBuffer<int32_t> op_kind; DeviceBuffer<uint64_t> txn_id, read_id, addr, aux, woff, watch;
    DeviceBuffer<int64_t> value;
    DeviceBuffer<int32_t> counts[23];
    DeviceBuffer<uint64_t> evt, read, loc, txn, queue;
    StmInputs inputs; StmOutputs outputs;
};

static StmProblemSpec make_bench_spec() {
    StmProblemSpec spec = {};
    spec.abi_version = STM_ABI_VERSION;
    spec.max_txns = 16; spec.max_locations = 256; spec.max_read_set = 32;
    spec.max_write_set = 16; spec.max_watch_set = 16;
    spec.max_waiters_per_location = 8; spec.max_retry_watchers_per_location = 8;
    spec.max_batch = 64; spec.max_steps = 24; spec.flags = 0;
    if (!stm_validate_problem_spec(&spec)) throw std::runtime_error("invalid bench spec");
    return spec;
}

// One op-stream variant per timed iteration (plus one for warmup): same batch sizes
// and step count every variant (timing comparable), but the picks/addrs/values/txn
// ids derive from (PMPP_BENCH_SEED, variant). Every timed iteration thus runs a fresh
// op stream, and each iteration's outputs are folded into out_fnv, so replaying an
// output cached from warmup or an earlier iteration leaves a stale region in the
// digest → perf FAIL.
static std::vector<StepHost> build_seq(const StmProblemSpec& spec, uint64_t variant_seed) {
    SplitMix64 rng(variant_seed ^ 0x12345678ULL);
    std::vector<StepHost> steps;
    int next_txn = 1;
    for (int s = 0; s < 24; ++s) {
        StepHost st; st.run = {}; st.run.abi_version = STM_ABI_VERSION; st.run.step_id = s;
        std::vector<int32_t> ok; std::vector<uint64_t> tid, rid, ad, ax, wo, wa;
        std::vector<int64_t> vv;
        int batch = (s == 11) ? 0 : 64;
        for (int i = 0; i < batch; ++i) {
            int pick = rng.uniform_int(0, 9);
            uint64_t t = (uint64_t)(next_txn - rng.uniform_int(0, 3));
            if ((int64_t)t < 1) t = 1;
            uint64_t addr = (uint64_t)rng.uniform_int(0, 63);
            int64_t val = rng.next_i64();
            int32_t kind; uint64_t txn = t, readid = 0, A = 0, aux = 0, woff = (uint64_t)wa.size();
            int64_t V = 0;
            if (pick == 0) { kind = STM_OP_BEGIN; txn = (uint64_t)next_txn++; aux = (uint64_t)rng.uniform_int(0, 9); }
            else if (pick == 1) { kind = STM_OP_TX_READ; readid = (uint64_t)i; A = addr; }
            else if (pick == 2) { kind = STM_OP_TX_WRITE; A = addr; V = val; }
            else if (pick == 3) { kind = STM_OP_VALIDATE; }
            else if (pick == 4) { kind = STM_OP_TRY_PREPARE; }
            else if (pick == 5) { kind = STM_OP_DRAIN_COMMITS; txn = 0; aux = (uint64_t)rng.uniform_int(0, 4); }
            else if (pick == 6) { kind = STM_OP_RETRY; aux = (uint64_t)rng.uniform_int(1, 3);
                for (uint64_t k = 0; k < aux; ++k) wa.push_back((uint64_t)rng.uniform_int(0, 63)); }
            else if (pick == 7) { kind = STM_OP_NON_TX_WRITE; txn = 0; A = addr; V = val; }
            else if (pick == 8) { kind = STM_OP_ABORT; }
            else { kind = STM_OP_TX_WRITE; A = addr; V = val; }
            ok.push_back(kind); tid.push_back(txn); rid.push_back(readid); ad.push_back(A);
            vv.push_back(V); ax.push_back(aux); wo.push_back(woff);
        }
        st.run.batch_size = (int32_t)ok.size();
        const size_t rows = std::max<size_t>(1, ok.size());
        ok.resize(rows, 0); tid.resize(rows, 0); rid.resize(rows, 0); ad.resize(rows, 0);
        vv.resize(rows, 0); ax.resize(rows, 0); wo.resize(rows, 0);
        if (wa.empty()) wa.push_back(0);
        st.op_kind = ok; st.txn_id = tid; st.read_id = rid; st.addr = ad; st.value = vv;
        st.aux = ax; st.watch_off = wo; st.watch_addrs = wa;
        steps.push_back(st);
    }
    return steps;
}

static DeviceStep* make_device_step(const StepHost& h) {
    DeviceStep* ds = new DeviceStep();
    ds->host = h;
    ds->op_kind.allocate(h.op_kind.size()); ds->txn_id.allocate(h.txn_id.size());
    ds->read_id.allocate(h.read_id.size()); ds->addr.allocate(h.addr.size());
    ds->value.allocate(h.value.size()); ds->aux.allocate(h.aux.size());
    ds->woff.allocate(h.watch_off.size()); ds->watch.allocate(h.watch_addrs.size());
    ds->op_kind.upload(h.op_kind); ds->txn_id.upload(h.txn_id); ds->read_id.upload(h.read_id);
    ds->addr.upload(h.addr); ds->value.upload(h.value); ds->aux.upload(h.aux);
    ds->woff.upload(h.watch_off); ds->watch.upload(h.watch_addrs);
    for (int i = 0; i < 23; ++i) ds->counts[i].allocate(1);
    ds->evt.allocate(1); ds->read.allocate(1); ds->loc.allocate(1); ds->txn.allocate(1); ds->queue.allocate(1);
    ds->inputs = {};
    ds->inputs.op_kind = ds->op_kind.ptr; ds->inputs.txn_id = ds->txn_id.ptr;
    ds->inputs.read_id = ds->read_id.ptr; ds->inputs.addr = ds->addr.ptr;
    ds->inputs.value = ds->value.ptr; ds->inputs.aux = ds->aux.ptr;
    ds->inputs.watch_off = ds->woff.ptr; ds->inputs.watch_addrs = ds->watch.ptr;
    ds->outputs = {};
    int32_t** oc = (int32_t**)&ds->outputs;
    for (int i = 0; i < 23; ++i) oc[i] = ds->counts[i].ptr;
    ds->outputs.stm_event_hash = ds->evt.ptr; ds->outputs.read_result_hash = ds->read.ptr;
    ds->outputs.location_hash = ds->loc.ptr; ds->outputs.txn_hash = ds->txn.ptr;
    ds->outputs.queue_hash = ds->queue.ptr;
    return ds;
}

static void run_sequence(void* state, void* ws, size_t wsb, cudaStream_t stream,
                         const std::vector<DeviceStep*>& steps) {
    for (DeviceStep* ds : steps)
        CUDA_CHECK(solution_run(state, &ds->host.run, &ds->inputs, &ds->outputs, ws, wsb, stream));
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 20;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));
        const StmProblemSpec spec = make_bench_spec();
        const size_t wsb = solution_workspace_bytes(&spec);
        if (wsb == 0) throw std::runtime_error("workspace_bytes 0");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace; workspace.allocate(wsb);

        // Variant 0 is warmup-only; variants 1..iters feed the timed iterations.
        // Per-variant seeds come from a SplitMix64 stream over PMPP_BENCH_SEED.
        SplitMix64 vseed_rng(pmpp::bench_seed(g_state) ^ 0x700000000ULL);
        std::vector<std::vector<DeviceStep*>> variants;
        variants.reserve((size_t)iters + 1);
        for (int v = 0; v <= iters; ++v) {
            std::vector<StepHost> host_steps = build_seq(spec, vseed_rng.next_u64());
            std::vector<DeviceStep*> steps;
            for (const StepHost& h : host_steps) steps.push_back(make_device_step(h));
            variants.push_back(std::move(steps));
        }

        std::printf("bench stm_commit_arbiter max_txns=%d max_locations=%d T=%zu\n",
                    spec.max_txns, spec.max_locations, variants[0].size());

        for (int w = 0; w < 3; ++w) {
            CUDA_CHECK(solution_reset(state, stream));
            run_sequence(state, workspace.ptr, wsb, stream, variants[0]);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        cudaEvent_t a = nullptr, b = nullptr;
        CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
        double total_ms = 0.0;

        // Fold each timed iteration's graded outputs (outside the timed region) so a
        // stale/replayed output on any iteration breaks the digest.
        pmpp::OutFnv dg;

        for (int it = 0; it < iters; ++it) {
            const std::vector<DeviceStep*>& steps = variants[(size_t)it + 1];
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(a, stream));
            run_sequence(state, workspace.ptr, wsb, stream, steps);
            CUDA_CHECK(cudaEventRecord(b, stream));
            CUDA_CHECK(cudaEventSynchronize(b));
            float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            total_ms += ms;

            for (DeviceStep* ds : steps) {
                for (int i = 0; i < 23; ++i) dg.dev(ds->counts[i].ptr, sizeof(int32_t));
                dg.dev(ds->evt.ptr, sizeof(uint64_t));
                dg.dev(ds->read.ptr, sizeof(uint64_t));
                dg.dev(ds->loc.ptr, sizeof(uint64_t));
                dg.dev(ds->txn.ptr, sizeof(uint64_t));
                dg.dev(ds->queue.ptr, sizeof(uint64_t));
            }
        }
        std::printf("avg_ms=%.6f\n", total_ms / iters);
        dg.print();
        CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
        for (std::vector<DeviceStep*>& steps : variants)
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
