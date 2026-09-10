// ============================================================================
// file: bench_mla_latent_absorb_decode.cu
//
// Benchmark: replays a fixed 162-step mixed prefill+decode sequence
// (B=32, Hq=16, d_c=256, d_r=64, d_h=128, d_v=128, max_seq_len=2048):
//   - 100 chunked-prefill steps (all rows, 8 tokens each, L -> 800),
//   - 30 mixed steps (even rows 8 tokens, odd rows 1 token),
//   - 32 decode steps (all rows, 1 token).
// Prints the average latency of one full sequence replay.
// ============================================================================

#include "mla_latent_absorb_decode_common.h"

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

struct BenchStep {
    MlaRunSpec run;
    int pool_variant;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> new_token_count;
};

struct InputPool {
    DeviceBuffer<float> new_c;
    DeviceBuffer<float> new_r;
    DeviceBuffer<float> q;
    DeviceBuffer<float> q_rope;
};

int main(int argc, char** argv) {
    int iters = 5;
    if (argc > 1) {
        iters = std::atoi(argv[1]);
        if (iters < 1) iters = 1;
    }

    try {
        CUDA_CHECK(cudaSetDevice(0));

        MlaProblemSpec spec = {};
        spec.abi_version = MLA_ABI_VERSION;
        spec.B = 32;
        spec.Hq = 16;
        spec.d_c = 256;
        spec.d_r = 64;
        spec.d_h = 128;
        spec.d_v = 128;
        spec.max_seq_len = 2048;
        spec.flags = 0;
        if (!mla_validate_problem_spec(&spec)) {
            throw std::runtime_error("bad bench spec");
        }

        const int B = spec.B;
        const int Hq = spec.Hq;
        const int half_r = spec.d_r / 2;

        // Weights.
        SplitMix64 wrng(0xB81A0BE7C4ULL);
        std::vector<float> wuk((size_t)Hq * spec.d_h * spec.d_c);
        for (float& v : wuk) v = wrng.uniform_float(-0.25f, 0.25f);
        std::vector<float> wuv((size_t)Hq * spec.d_v * spec.d_c);
        for (float& v : wuv) v = wrng.uniform_float(-0.25f, 0.25f);
        std::vector<float> rc((size_t)spec.max_seq_len * half_r);
        std::vector<float> rs((size_t)spec.max_seq_len * half_r);
        for (int t = 0; t < spec.max_seq_len; ++t) {
            for (int j = 0; j < half_r; ++j) {
                const double freq =
                    std::pow(10000.0, -2.0 * j / static_cast<double>(spec.d_r));
                const double ang = t * freq;
                rc[(size_t)t * half_r + j] = static_cast<float>(std::cos(ang));
                rs[(size_t)t * half_r + j] = static_cast<float>(std::sin(ang));
            }
        }

        DeviceBuffer<float> d_wuk;
        DeviceBuffer<float> d_wuv;
        DeviceBuffer<float> d_cos;
        DeviceBuffer<float> d_sin;
        d_wuk.allocate(wuk.size());
        d_wuv.allocate(wuv.size());
        d_cos.allocate(rc.size());
        d_sin.allocate(rs.size());
        d_wuk.upload(wuk);
        d_wuv.upload(wuv);
        d_cos.upload(rc);
        d_sin.upload(rs);

        // Input pools (4 variants, cycled across steps).
        const int kPools = 4;
        std::vector<InputPool> pools(kPools);
        for (int v = 0; v < kPools; ++v) {
            SplitMix64 rng(0x9F1E0000ULL + v);

            std::vector<float> hc((size_t)B * MLA_MAX_NEW_TOKENS * spec.d_c);
            for (float& x : hc) x = rng.uniform_float(-3.0f, 3.0f);
            std::vector<float> hr((size_t)B * MLA_MAX_NEW_TOKENS * spec.d_r);
            for (float& x : hr) x = rng.uniform_float(-3.0f, 3.0f);
            std::vector<float> hq((size_t)B * MLA_MAX_NEW_TOKENS * Hq * spec.d_h);
            for (float& x : hq) x = rng.uniform_float(-1.0f, 1.0f);
            std::vector<float> hqr((size_t)B * MLA_MAX_NEW_TOKENS * Hq * spec.d_r);
            for (float& x : hqr) x = rng.uniform_float(-1.0f, 1.0f);

            pools[v].new_c.allocate(hc.size());
            pools[v].new_c.upload(hc);
            pools[v].new_r.allocate(hr.size());
            pools[v].new_r.upload(hr);
            pools[v].q.allocate(hq.size());
            pools[v].q.upload(hq);
            pools[v].q_rope.allocate(hqr.size());
            pools[v].q_rope.upload(hqr);
        }

        // Step schedule.
        std::vector<BenchStep> steps;
        int step_id = 0;
        auto push_step = [&](const std::vector<int32_t>& active,
                             const std::vector<int32_t>& counts) {
            BenchStep st;
            st.run = {};
            st.run.abi_version = MLA_ABI_VERSION;
            st.run.active_count = static_cast<int32_t>(active.size());
            st.run.step_id = step_id;
            st.pool_variant = step_id % kPools;
            st.active_seq = active;
            st.new_token_count = counts;
            steps.push_back(std::move(st));
            ++step_id;
        };

        {
            std::vector<int32_t> all(B);
            for (int b = 0; b < B; ++b) all[b] = b;

            std::vector<int32_t> eight(B, 8);
            for (int s = 0; s < 100; ++s) push_step(all, eight);

            std::vector<int32_t> mixed(B);
            for (int b = 0; b < B; ++b) mixed[b] = (b % 2 == 0) ? 8 : 1;
            for (int s = 0; s < 30; ++s) push_step(all, mixed);

            std::vector<int32_t> one(B, 1);
            for (int s = 0; s < 32; ++s) push_step(all, one);
        }

        std::vector<DeviceBuffer<int32_t>*> d_active(steps.size());
        std::vector<DeviceBuffer<int32_t>*> d_counts(steps.size());
        for (size_t i = 0; i < steps.size(); ++i) {
            d_active[i] = new DeviceBuffer<int32_t>();
            d_active[i]->allocate(steps[i].active_seq.size());
            d_active[i]->upload(steps[i].active_seq);
            d_counts[i] = new DeviceBuffer<int32_t>();
            d_counts[i]->allocate(steps[i].new_token_count.size());
            d_counts[i]->upload(steps[i].new_token_count);
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        MlaInitInputs init_inputs = {};
        init_inputs.W_uk = d_wuk.ptr;
        init_inputs.W_uv = d_wuv.ptr;
        init_inputs.rope_cos = d_cos.ptr;
        init_inputs.rope_sin = d_sin.ptr;

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &init_inputs, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        size_t workspace_bytes = solution_workspace_bytes(&spec);
        if (workspace_bytes == 0) workspace_bytes = 1;
        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(workspace_bytes);

        DeviceBuffer<float> d_y;
        DeviceBuffer<float> d_lse;
        DeviceBuffer<int32_t> d_seq_len;
        DeviceBuffer<uint64_t> d_cache_hash;
        DeviceBuffer<uint64_t> d_meta;
        DeviceBuffer<int32_t> d_sat;
        DeviceBuffer<int32_t> d_total;

        d_y.allocate((size_t)B * MLA_MAX_NEW_TOKENS * Hq * spec.d_v);
        d_lse.allocate((size_t)B * MLA_MAX_NEW_TOKENS * Hq);
        d_seq_len.allocate((size_t)B);
        d_cache_hash.allocate((size_t)B);
        d_meta.allocate(1);
        d_sat.allocate(1);
        d_total.allocate(1);

        MlaOutputs outputs = {};
        outputs.y = d_y.ptr;
        outputs.lse = d_lse.ptr;
        outputs.seq_len = d_seq_len.ptr;
        outputs.cache_hash = d_cache_hash.ptr;
        outputs.meta_checksum = d_meta.ptr;
        outputs.sat_count = d_sat.ptr;
        outputs.total_tokens = d_total.ptr;

        std::printf(
            "bench_sequence B=%d Hq=%d d_c=%d d_r=%d d_h=%d d_v=%d msl=%d T=%zu\n",
            spec.B, spec.Hq, spec.d_c, spec.d_r, spec.d_h, spec.d_v,
            spec.max_seq_len, steps.size());

        auto run_sequence = [&]() {
            for (size_t i = 0; i < steps.size(); ++i) {
                MlaInputs inputs = {};
                inputs.active_seq = d_active[i]->ptr;
                inputs.new_token_count = d_counts[i]->ptr;
                inputs.new_c = pools[steps[i].pool_variant].new_c.ptr;
                inputs.new_r = pools[steps[i].pool_variant].new_r.ptr;
                inputs.q = pools[steps[i].pool_variant].q.ptr;
                inputs.q_rope = pools[steps[i].pool_variant].q_rope.ptr;

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

        for (size_t i = 0; i < steps.size(); ++i) {
            delete d_active[i];
            delete d_counts[i];
        }

        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));

        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
