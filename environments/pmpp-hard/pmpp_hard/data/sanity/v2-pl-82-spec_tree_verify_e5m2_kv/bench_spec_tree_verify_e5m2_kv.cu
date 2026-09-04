// ============================================================================
// file: bench_spec_tree_verify_e5m2_kv.cu
//
// Benchmark: replays a fixed 110-step speculation sequence
// (B=24, Hq=16, Hkv=4, D=128, P=16, max_nodes=32, max_seq_len=768):
// every step verifies a 4-chain x 8 draft forest per sequence with a planned
// acceptance pattern (avg ~3.5 accepted + bonus per step, L -> ~495).
// Prints the average latency of one full sequence replay.
// ============================================================================

#include "spec_tree_verify_e5m2_kv_common.h"
#include "spec_tree_verify_e5m2_kv_oracle.hpp"

#include <cuda_runtime.h>

#include <stdint.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
};

struct InputPool {
    std::vector<float> h_draft_k;
    std::vector<float> h_draft_v;
    std::vector<float> h_bonus_k;
    std::vector<float> h_bonus_v;
    DeviceBuffer<float> draft_k;
    DeviceBuffer<float> draft_v;
    DeviceBuffer<float> q;
    DeviceBuffer<float> bonus_k;
    DeviceBuffer<float> bonus_v;
};

int main(int argc, char** argv) {
    int iters = 5;
    if (argc > 1) {
        iters = std::atoi(argv[1]);
        if (iters < 1) iters = 1;
    }

    try {
        CUDA_CHECK(cudaSetDevice(0));

        StvProblemSpec spec = {};
        spec.abi_version = STV_ABI_VERSION;
        spec.B = 24;
        spec.Hq = 16;
        spec.Hkv = 4;
        spec.D = 128;
        spec.page_size = 16;
        spec.max_nodes = 32;
        spec.max_seq_len = 768;
        spec.max_pages = 1280;
        spec.flags = 0;
        if (!stv_validate_problem_spec(&spec)) {
            throw std::runtime_error("bad bench spec");
        }

        const int B = spec.B;
        const int N = spec.max_nodes;
        const int Hq = spec.Hq;
        const int Hkv = spec.Hkv;
        const int D = spec.D;
        const int kSteps = 110;
        const int kPools = 4;
        const int kChains = 4;
        const int kChainLen = 8;

        // Tree topology is fixed: 4 chains of 8 (heap of chains).
        std::vector<int32_t> h_parent((size_t)B * N);
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < kChains; ++c) {
                for (int k = 0; k < kChainLen; ++k) {
                    const int i = c * kChainLen + k;
                    h_parent[(size_t)b * N + i] = (k == 0) ? -1 : i - 1;
                }
            }
        }

        // Input pools.
        std::vector<InputPool> pools(kPools);
        for (int v = 0; v < kPools; ++v) {
            SplitMix64 rng(0xA5117E00ULL + v);
            InputPool& pl = pools[v];

            pl.h_draft_k.resize((size_t)B * N * Hkv * D);
            for (float& x : pl.h_draft_k) x = rng.uniform_float(-2.5f, 2.5f);
            pl.h_draft_v.resize((size_t)B * N * Hkv * D);
            for (float& x : pl.h_draft_v) x = rng.uniform_float(-2.5f, 2.5f);
            std::vector<float> hq_((size_t)B * N * Hq * D);
            for (float& x : hq_) x = rng.uniform_float(-1.0f, 1.0f);
            pl.h_bonus_k.resize((size_t)B * Hkv * D);
            for (float& x : pl.h_bonus_k) x = rng.uniform_float(-2.5f, 2.5f);
            pl.h_bonus_v.resize((size_t)B * Hkv * D);
            for (float& x : pl.h_bonus_v) x = rng.uniform_float(-2.5f, 2.5f);

            pl.draft_k.allocate(pl.h_draft_k.size());
            pl.draft_k.upload(pl.h_draft_k);
            pl.draft_v.allocate(pl.h_draft_v.size());
            pl.draft_v.upload(pl.h_draft_v);
            pl.q.allocate(hq_.size());
            pl.q.upload(hq_);
            pl.bonus_k.allocate(pl.h_bonus_k.size());
            pl.bonus_k.upload(pl.h_bonus_k);
            pl.bonus_v.allocate(pl.h_bonus_v.size());
            pl.bonus_v.upload(pl.h_bonus_v);
        }

        // Byte-exact correct signatures per (pool, row, node).
        StvOracleState helper;
        helper.init(spec);
        std::vector<std::vector<uint64_t>> correct_sig(kPools);
        for (int v = 0; v < kPools; ++v) {
            correct_sig[v].resize((size_t)B * N);
            for (int b = 0; b < B; ++b) {
                for (int i = 0; i < N; ++i) {
                    const StvOracleToken tok = helper.quantize_token(
                        pools[v].h_draft_k.data() + (((size_t)b * N + i) * Hkv) * D,
                        pools[v].h_draft_v.data() + (((size_t)b * N + i) * Hkv) * D);
                    correct_sig[v][(size_t)b * N + i] =
                        StvOracleState::token_fold(
                            kStvOracleFnvBasis, tok, Hkv, D);
                }
            }
        }

        // Per-step target_sig buffers implementing the acceptance plan.
        const int pattern[8] = {4, 2, 6, 0, 3, 5, 1, 7};
        std::vector<DeviceBuffer<uint64_t>*> d_sigs(kSteps);
        {
            long total_committed = 0;
            for (int s = 0; s < kSteps; ++s) {
                const int v = s % kPools;
                std::vector<uint64_t> sig((size_t)B * N);
                for (int b = 0; b < B; ++b) {
                    const int chain = (s + b) % kChains;
                    const int dep = pattern[(s + b) % 8];
                    for (int i = 0; i < N; ++i) {
                        const int c = i / kChainLen;
                        const int k = i % kChainLen;
                        const bool acc = (c == chain) && (k < dep);
                        sig[(size_t)b * N + i] =
                            correct_sig[v][(size_t)b * N + i] ^
                            (acc ? 0ULL : 0x5A5A5A5A5A5A5A5AULL);
                    }
                    if (b == 0) total_committed += dep + 1;
                }
                d_sigs[s] = new DeviceBuffer<uint64_t>();
                d_sigs[s]->allocate(sig.size());
                d_sigs[s]->upload(sig);
            }
            if (total_committed > spec.max_seq_len) {
                throw std::runtime_error("bench schedule overflows max_seq_len");
            }
        }

        std::vector<int32_t> h_active(B);
        for (int b = 0; b < B; ++b) h_active[b] = b;
        std::vector<int32_t> h_ncount(B, N);

        DeviceBuffer<int32_t> d_active;
        d_active.allocate(B);
        d_active.upload(h_active);
        DeviceBuffer<int32_t> d_ncount;
        d_ncount.allocate(B);
        d_ncount.upload(h_ncount);
        DeviceBuffer<int32_t> d_parent;
        d_parent.allocate(h_parent.size());
        d_parent.upload(h_parent);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t workspace_bytes = solution_workspace_bytes(&spec);
        if (workspace_bytes == 0) workspace_bytes = 1;
        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(workspace_bytes);

        DeviceBuffer<float> d_y;
        DeviceBuffer<float> d_lse;
        DeviceBuffer<int32_t> d_tail;
        DeviceBuffer<int32_t> d_alen;
        DeviceBuffer<int32_t> d_seq_len;
        DeviceBuffer<uint64_t> d_kv_hash;
        DeviceBuffer<uint64_t> d_pcs;
        DeviceBuffer<int32_t> d_free;
        DeviceBuffer<int32_t> d_allocs;
        DeviceBuffer<int32_t> d_frees;

        d_y.allocate((size_t)B * N * Hq * D);
        d_lse.allocate((size_t)B * N * Hq);
        d_tail.allocate(B);
        d_alen.allocate(B);
        d_seq_len.allocate(B);
        d_kv_hash.allocate(B);
        d_pcs.allocate(1);
        d_free.allocate(1);
        d_allocs.allocate(1);
        d_frees.allocate(1);

        StvOutputs outputs = {};
        outputs.y = d_y.ptr;
        outputs.lse = d_lse.ptr;
        outputs.accepted_tail = d_tail.ptr;
        outputs.accepted_len = d_alen.ptr;
        outputs.seq_len = d_seq_len.ptr;
        outputs.kv_hash = d_kv_hash.ptr;
        outputs.page_state_checksum = d_pcs.ptr;
        outputs.free_pages = d_free.ptr;
        outputs.total_allocs = d_allocs.ptr;
        outputs.total_frees = d_frees.ptr;

        std::printf(
            "bench_sequence B=%d Hq=%d Hkv=%d D=%d P=%d N=%d msl=%d pages=%d T=%d\n",
            spec.B, spec.Hq, spec.Hkv, spec.D, spec.page_size, spec.max_nodes,
            spec.max_seq_len, spec.max_pages, kSteps);

        auto run_sequence = [&]() {
            for (int s = 0; s < kSteps; ++s) {
                const int v = s % kPools;
                StvRunSpec run = {};
                run.abi_version = STV_ABI_VERSION;
                run.active_count = B;
                run.step_id = s;

                StvInputs inputs = {};
                inputs.active_seq = d_active.ptr;
                inputs.node_count = d_ncount.ptr;
                inputs.parent = d_parent.ptr;
                inputs.draft_k = pools[v].draft_k.ptr;
                inputs.draft_v = pools[v].draft_v.ptr;
                inputs.q = pools[v].q.ptr;
                inputs.target_sig = d_sigs[s]->ptr;
                inputs.bonus_k = pools[v].bonus_k.ptr;
                inputs.bonus_v = pools[v].bonus_v.ptr;

                CUDA_CHECK(solution_run(
                    state, &run, &inputs, &outputs,
                    workspace.ptr, workspace_bytes, stream));
            }
        };

        for (int warm = 0; warm < 2; ++warm) {
            CUDA_CHECK(solution_reset(state, stream));
            run_sequence();
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double total_ms = 0.0;

        for (int iter = 0; iter < iters; ++iter) {
            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaEventRecord(start, stream));
            run_sequence();
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            total_ms += static_cast<double>(elapsed_ms);
        }

        const double avg_ms = total_ms / static_cast<double>(iters);
        std::printf("avg_ms=%.6f\n", avg_ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (int s = 0; s < kSteps; ++s) delete d_sigs[s];

        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
