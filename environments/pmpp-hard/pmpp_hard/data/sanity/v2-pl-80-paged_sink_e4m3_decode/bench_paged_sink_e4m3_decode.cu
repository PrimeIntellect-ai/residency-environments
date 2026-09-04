// ============================================================================
// file: bench_paged_sink_e4m3_decode.cu
//
// Steady-state timed loop over a mixed chunked-prefill + decode schedule:
//   - 64 prefill steps (all sequences, 8 tokens each) grow every sequence to
//     L = 512 (past n_sink + window = 512, so the window slides and pages
//     churn),
//   - 48 decode steps (most sequences active, 1-2 tokens, some attention-only
//     rows, one empty step).
// Each timed iteration replays the whole schedule after a reset.
// K/V/Q payloads are cycled from a small pool of pre-staged variants so the
// bench fits in memory; correctness is not checked here (see test binary).
// ============================================================================

#include "paged_sink_e4m3_decode_common.h"

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

static constexpr uint64_t g_state = 0x9b5ad4b1c2e07d63ULL;

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

static PseProblemSpec make_spec() {
    PseProblemSpec spec = {};
    spec.abi_version = PSE_ABI_VERSION;
    spec.B = 48;
    spec.Hq = 16;
    spec.Hkv = 4;
    spec.D = 128;
    spec.page_size = 16;
    spec.n_sink = 32;
    spec.window = 480;
    spec.max_seq_len = 4096;
    spec.max_pages = 2048;
    spec.flags = 0;

    if (!pse_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid PseProblemSpec generated");
    }

    return spec;
}

struct StepMeta {
    PseRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> new_token_count;
    int pool_variant;
};

struct InputPool {
    DeviceBuffer<float> new_k;
    DeviceBuffer<float> new_v;
    DeviceBuffer<float> q;
};

static constexpr int kPoolVariants = 4;

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 10;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        const PseProblemSpec spec = make_spec();
        const int B = spec.B;
        const int Hq = spec.Hq;
        const int Hkv = spec.Hkv;
        const int D = spec.D;

        size_t workspace_bytes = solution_workspace_bytes(&spec);
        if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(workspace_bytes);

        // Pre-staged K/V/Q payload variants shared across steps.
        std::vector<InputPool> pools(kPoolVariants);
        const size_t kv_elems = (size_t)B * PSE_MAX_NEW_TOKENS * Hkv * D;
        const size_t q_elems = (size_t)B * Hq * D;
        for (int v = 0; v < kPoolVariants; ++v) {
            std::vector<float> h_k(kv_elems);
            std::vector<float> h_v(kv_elems);
            std::vector<float> h_q(q_elems);
            SplitMix64 rng(g_state ^ (0xABCD0000ULL + (uint64_t)v));
            for (size_t i = 0; i < kv_elems; ++i) {
                h_k[i] = rng.uniform_float(-2.50f, 2.50f);
                h_v[i] = rng.uniform_float(-2.50f, 2.50f);
            }
            for (size_t i = 0; i < q_elems; ++i) {
                h_q[i] = rng.uniform_float(-0.50f, 0.50f);
            }
            pools[v].new_k.allocate(kv_elems);
            pools[v].new_v.allocate(kv_elems);
            pools[v].q.allocate(q_elems);
            pools[v].new_k.upload(h_k);
            pools[v].new_v.upload(h_v);
            pools[v].q.upload(h_q);
        }

        // Build the schedule: 64 prefill steps, then 48 decode steps.
        std::vector<StepMeta> steps;
        for (int s = 0; s < 112; ++s) {
            StepMeta m;
            m.run = {};
            m.run.abi_version = PSE_ABI_VERSION;
            m.run.step_id = s;
            m.pool_variant = s % kPoolVariants;

            if (s < 64) {
                for (int b = 0; b < B; ++b) {
                    m.active_seq.push_back(b);
                    m.new_token_count.push_back(8);
                }
            } else if (s == 90) {
                // empty step
            } else {
                for (int b = 0; b < B; ++b) {
                    if ((b + s) % 7 == 0) continue;
                    m.active_seq.push_back(b);
                    if ((b + s) % 11 == 0) {
                        m.new_token_count.push_back(0);  // attention-only row
                    } else {
                        m.new_token_count.push_back(1 + ((b + s) % 2));
                    }
                }
            }

            m.run.active_count = static_cast<int32_t>(m.active_seq.size());
            if (m.active_seq.empty()) {
                m.active_seq.push_back(0);
                m.new_token_count.push_back(0);
            }
            steps.push_back(std::move(m));
        }

        // Stage per-step metadata.
        std::vector<DeviceBuffer<int32_t>*> d_active(steps.size());
        std::vector<DeviceBuffer<int32_t>*> d_counts(steps.size());
        for (size_t i = 0; i < steps.size(); ++i) {
            d_active[i] = new DeviceBuffer<int32_t>();
            d_counts[i] = new DeviceBuffer<int32_t>();
            d_active[i]->allocate(steps[i].active_seq.size());
            d_counts[i]->allocate(steps[i].new_token_count.size());
            d_active[i]->upload(steps[i].active_seq);
            d_counts[i]->upload(steps[i].new_token_count);
        }

        // Shared outputs.
        DeviceBuffer<float> d_y;
        DeviceBuffer<float> d_lse;
        DeviceBuffer<int32_t> d_seq_len;
        DeviceBuffer<uint64_t> d_kv_hash;
        DeviceBuffer<uint64_t> d_pcs;
        DeviceBuffer<int32_t> d_free_pages;
        DeviceBuffer<int32_t> d_total_allocs;
        DeviceBuffer<int32_t> d_total_frees;

        d_y.allocate((size_t)B * Hq * D);
        d_lse.allocate((size_t)B * Hq);
        d_seq_len.allocate((size_t)B);
        d_kv_hash.allocate((size_t)B);
        d_pcs.allocate(1);
        d_free_pages.allocate(1);
        d_total_allocs.allocate(1);
        d_total_frees.allocate(1);

        PseOutputs outputs = {};
        outputs.y = d_y.ptr;
        outputs.lse = d_lse.ptr;
        outputs.seq_len = d_seq_len.ptr;
        outputs.kv_hash = d_kv_hash.ptr;
        outputs.page_state_checksum = d_pcs.ptr;
        outputs.free_pages = d_free_pages.ptr;
        outputs.total_allocs = d_total_allocs.ptr;
        outputs.total_frees = d_total_frees.ptr;

        std::printf(
            "bench_sequence B=%d Hq=%d Hkv=%d D=%d P=%d S=%d W=%d pages=%d T=%zu\n",
            spec.B,
            spec.Hq,
            spec.Hkv,
            spec.D,
            spec.page_size,
            spec.n_sink,
            spec.window,
            spec.max_pages,
            steps.size());

        auto run_sequence = [&]() {
            for (size_t i = 0; i < steps.size(); ++i) {
                PseInputs inputs = {};
                inputs.active_seq = d_active[i]->ptr;
                inputs.new_token_count = d_counts[i]->ptr;
                inputs.new_k = pools[steps[i].pool_variant].new_k.ptr;
                inputs.new_v = pools[steps[i].pool_variant].new_v.ptr;
                inputs.q = pools[steps[i].pool_variant].q.ptr;

                CUDA_CHECK(solution_run(
                    state,
                    &steps[i].run,
                    &inputs,
                    &outputs,
                    workspace.ptr,
                    workspace_bytes,
                    stream));
            }
        };

        for (int warm = 0; warm < 3; ++warm) {
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

        for (size_t i = 0; i < steps.size(); ++i) {
            delete d_active[i];
            delete d_counts[i];
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));

        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
