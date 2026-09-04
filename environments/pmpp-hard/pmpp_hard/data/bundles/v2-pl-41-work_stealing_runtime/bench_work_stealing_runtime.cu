// file: bench_work_stealing_runtime.cu

#include "work_stealing_runtime_common.h"
#include "pmpp_bench_digest.cuh"

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
    int uniform_int(int lo, int hi) { return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1)); }
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr; size_t count = 0;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) { count = n; if (n) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n)); }
};

static WsrRunSpec mk(int kind, int step) { WsrRunSpec r = {}; r.abi_version = WSR_ABI_VERSION; r.op_kind = kind; r.step_id = step; return r; }

// --- C3 timed-fold (patterns I1 + P1, op-stream form): every timed EPISODE
// replays the same op structure (kind sequence, step ids, SPAWN task-id
// sequence — timing comparability) with fresh per-episode op VALUES derived
// from (PMPP_BENCH_SEED, episode, op). Warmup uses dedicated episodes. After
// each timed episode (outside the event pair — zero timing impact) a probe
// kernel folds the graded end-of-episode outputs into acc[episode]; the digest
// folds every accumulator, so a cached-replay episode cannot match.

static uint64_t host_mix64(uint64_t z) {
    z ^= z >> 30; z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27; z *= 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ uint64_t c3_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct C3ProbeDesc { const uint8_t* ptr; unsigned long long nbytes; };
struct C3ProbeTable { C3ProbeDesc d[10]; int n; };

__global__ void c3_probe_fold(C3ProbeTable t, uint64_t salt,
                              unsigned long long* acc) {
    __shared__ unsigned long long sh[256];
    unsigned long long local = 0;
    for (int b = 0; b < t.n; ++b) {
        const C3ProbeDesc& d = t.d[b];
        const unsigned long long nwords = d.nbytes >> 3;
        const uint64_t bsalt = salt ^ (uint64_t)(b + 1) * 0x100000001b3ULL;
        const unsigned long long* w64 =
            reinterpret_cast<const unsigned long long*>(d.ptr);
        for (unsigned long long w = threadIdx.x; w < nwords; w += blockDim.x) {
            local ^= c3_mix(w64[w] ^ c3_mix(bsalt + w));
        }
        if (threadIdx.x == 0 && (d.nbytes & 7ULL)) {
            unsigned long long tail = 0;
            for (unsigned long long i = nwords << 3; i < d.nbytes; ++i)
                tail |= (unsigned long long)d.ptr[i] << (8 * (i & 7ULL));
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

// Fresh op values for episode e over the shared structure.
static std::vector<WsrRunSpec> make_episode_ops(
    const std::vector<WsrRunSpec>& base, uint64_t c3_root, int e) {
    std::vector<WsrRunSpec> ops = base;
    SplitMix64 r(host_mix64(c3_root ^ (uint64_t)(e + 1) * 0x9e3779b97f4a7c15ULL));
    for (size_t i = 0; i < ops.size(); ++i) {
        WsrRunSpec& op = ops[i];
        switch (op.op_kind) {
            case WSR_OP_SPAWN:
                op.a_worker = r.uniform_int(0, 7);
                op.a_priority = r.uniform_int(0, 1);
                op.a_work = r.uniform_int(1, i < 64 ? 8 : 6);
                break;
            case WSR_OP_RUN:
                op.a_worker = r.uniform_int(0, 7);
                op.a_work = r.uniform_int(1, 6);
                break;
            case WSR_OP_YIELD:
                op.a_worker = r.uniform_int(0, 7);
                break;
            case WSR_OP_BLOCK:
                op.a_worker = r.uniform_int(0, 7);
                op.a_key = r.uniform_int(1, 6);
                break;
            case WSR_OP_SLEEP:
                op.a_worker = r.uniform_int(0, 7);
                op.a_tick = r.uniform_int(1, 200);
                break;
            case WSR_OP_WAKE:
                op.a_key = r.uniform_int(1, 6);
                op.a_limit = r.uniform_int(0, 4);
                break;
            case WSR_OP_ADVANCE:
                op.a_delta = r.uniform_int(1, 50);
                break;
            default:
                break;
        }
    }
    return ops;
}

static std::vector<WsrRunSpec> build_ops(const WsrProblemSpec& spec) {
    std::vector<WsrRunSpec> ops;
    // Op data values derive from PMPP_BENCH_SEED so graded outputs are not
    // precomputable offline; the op-count/mix family stays fixed for timing.
    SplitMix64 rng(0xabcdef0123456789ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL));
    uint64_t nid = 1;
    int t = 0;
    for (int i = 0; i < 64; ++i) {
        WsrRunSpec r = mk(WSR_OP_SPAWN, t++); r.a_task = nid++; r.a_worker = rng.uniform_int(0, spec.W - 1);
        r.a_priority = rng.uniform_int(0, 1); r.a_work = rng.uniform_int(1, 8); ops.push_back(r);
    }
    for (int i = 0; i + 64 < spec.max_steps; ++i) {
        int p = rng.uniform_int(0, 99); int w = rng.uniform_int(0, spec.W - 1);
        WsrRunSpec r;
        if (p < 45) { r = mk(WSR_OP_RUN, t++); r.a_worker = w; r.a_work = rng.uniform_int(1, 6); }
        else if (p < 58) { r = mk(WSR_OP_SPAWN, t++); r.a_task = nid++; r.a_worker = w; r.a_priority = rng.uniform_int(0, 1); r.a_work = rng.uniform_int(1, 6); }
        else if (p < 68) { r = mk(WSR_OP_YIELD, t++); r.a_worker = w; }
        else if (p < 78) { r = mk(WSR_OP_BLOCK, t++); r.a_worker = w; r.a_key = rng.uniform_int(1, 6); }
        else if (p < 86) { r = mk(WSR_OP_SLEEP, t++); r.a_worker = w; r.a_tick = rng.uniform_int(1, 200); }
        else if (p < 92) { r = mk(WSR_OP_WAKE, t++); r.a_key = rng.uniform_int(1, 6); r.a_limit = rng.uniform_int(0, 4); }
        else { r = mk(WSR_OP_ADVANCE, t++); r.a_delta = rng.uniform_int(1, 50); }
        ops.push_back(r);
    }
    return ops;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        int iters = 10;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        WsrProblemSpec spec = {};
        spec.abi_version = WSR_ABI_VERSION; spec.W = 8; spec.local_cap_per_worker = 8;
        spec.global_cap = 32; spec.max_tasks = 256; spec.max_blocked = 64; spec.max_sleeping = 64;
        spec.max_steps = 1024;
        if (!wsr_validate_problem_spec(&spec)) throw std::runtime_error("bad spec");

        const size_t wb = solution_workspace_bytes(&spec);
        if (wb == 0) throw std::runtime_error("workspace 0");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace; workspace.allocate(wb);

        DeviceBuffer<int64_t> d_counts; d_counts.allocate(WSR_COUNT_N);
        DeviceBuffer<int32_t> d_opidx; d_opidx.allocate(1);
        DeviceBuffer<uint64_t> d_clock, d_eseq, d_sched, d_ready, d_run, d_blocked, d_sleep, d_state;
        d_clock.allocate(1); d_eseq.allocate(1); d_sched.allocate(1); d_ready.allocate(1);
        d_run.allocate(1); d_blocked.allocate(1); d_sleep.allocate(1); d_state.allocate(1);

        WsrOutputs out = {};
        out.counts = d_counts.ptr; out.op_index_out = d_opidx.ptr; out.clock_out = d_clock.ptr;
        out.event_seq_out = d_eseq.ptr; out.sched_event_hash = d_sched.ptr; out.ready_hash = d_ready.ptr;
        out.running_hash = d_run.ptr; out.blocked_hash = d_blocked.ptr; out.sleep_hash = d_sleep.ptr;
        out.state_checksum = d_state.ptr;

        WsrInputs in = {}; in.reserved = nullptr;
        const std::vector<WsrRunSpec> base_ops = build_ops(spec);

        // C3: per-episode op-value variants (structure shared with base_ops);
        // the last kWarmups episodes are warmup-only.
        SplitMix64 c3_root_rng(0xC3ULL ^ (pmpp::bench_seed(0) * 0x9e3779b97f4a7c15ULL));
        const uint64_t c3_root = c3_root_rng.next_u64();
        const int kWarmups = 2;
        const int n_ep = iters + kWarmups;
        std::vector<std::vector<WsrRunSpec>> ep_ops;
        ep_ops.reserve(n_ep);
        for (int e = 0; e < n_ep; ++e) ep_ops.push_back(make_episode_ops(base_ops, c3_root, e));

        C3ProbeTable pt;
        pt.n = 10;
        pt.d[0] = {(const uint8_t*)d_counts.ptr, (unsigned long long)WSR_COUNT_N * sizeof(int64_t)};
        pt.d[1] = {(const uint8_t*)d_opidx.ptr, sizeof(int32_t)};
        pt.d[2] = {(const uint8_t*)d_clock.ptr, sizeof(uint64_t)};
        pt.d[3] = {(const uint8_t*)d_eseq.ptr, sizeof(uint64_t)};
        pt.d[4] = {(const uint8_t*)d_sched.ptr, sizeof(uint64_t)};
        pt.d[5] = {(const uint8_t*)d_ready.ptr, sizeof(uint64_t)};
        pt.d[6] = {(const uint8_t*)d_run.ptr, sizeof(uint64_t)};
        pt.d[7] = {(const uint8_t*)d_blocked.ptr, sizeof(uint64_t)};
        pt.d[8] = {(const uint8_t*)d_sleep.ptr, sizeof(uint64_t)};
        pt.d[9] = {(const uint8_t*)d_state.ptr, sizeof(uint64_t)};
        DeviceBuffer<unsigned long long> probe_acc;
        probe_acc.allocate((size_t)iters);
        CUDA_CHECK(cudaMemset(probe_acc.ptr, 0, sizeof(unsigned long long) * probe_acc.count));

        std::printf("bench W=%d ops=%zu max_tasks=%d\n", spec.W, base_ops.size(), spec.max_tasks);

        for (int warm = 0; warm < kWarmups; ++warm) {
            CUDA_CHECK(solution_reset(state, stream));
            for (const WsrRunSpec& op : ep_ops[(size_t)iters + warm]) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, wb, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        double total_ms = 0;
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaEventRecord(start, stream));
            for (const WsrRunSpec& op : ep_ops[(size_t)it]) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, wb, stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            // Untimed (after the stop event): fold this episode's graded
            // end-of-episode outputs before the next episode overwrites them.
            c3_probe_fold<<<1, 256, 0, stream>>>(
                pt, host_mix64(c3_root ^ 0xF01DULL ^ (uint64_t)(it + 1)),
                probe_acc.ptr + it);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }
        std::printf("avg_ms=%.6f\n", total_ms / iters);

        // Untimed graded-output digest: fresh replay of the full op sequence, then fold
        // the final values of every graded output buffer (see WsrOutputs contract).
        CUDA_CHECK(solution_reset(state, stream));
        for (const WsrRunSpec& op : ep_ops[(size_t)iters - 1]) CUDA_CHECK(solution_run(state, &op, &in, &out, workspace.ptr, wb, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        // C3: the digest binds EVERY timed episode via the probe accumulators.
        {
            std::vector<unsigned long long> acc_host(probe_acc.count);
            CUDA_CHECK(cudaMemcpy(acc_host.data(), probe_acc.ptr,
                                  sizeof(unsigned long long) * probe_acc.count,
                                  cudaMemcpyDeviceToHost));
            dg.bytes(acc_host.data(), sizeof(unsigned long long) * acc_host.size());
        }
        dg.dev(d_counts.ptr, (size_t)WSR_COUNT_N * sizeof(int64_t));
        dg.dev(d_opidx.ptr, sizeof(int32_t));
        dg.dev(d_clock.ptr, sizeof(uint64_t));
        dg.dev(d_eseq.ptr, sizeof(uint64_t));
        dg.dev(d_sched.ptr, sizeof(uint64_t));
        dg.dev(d_ready.ptr, sizeof(uint64_t));
        dg.dev(d_run.ptr, sizeof(uint64_t));
        dg.dev(d_blocked.ptr, sizeof(uint64_t));
        dg.dev(d_sleep.ptr, sizeof(uint64_t));
        dg.dev(d_state.ptr, sizeof(uint64_t));
        dg.print();

        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
