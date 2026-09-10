// ============================================================================
// file: bench_paged_cow_prefill_decode.cu
//
// Benchmark: replays a fixed 96-step continuous-batching sequence
// (B=16, Hq=16, Hkv=4, D=128, P=16, C=64, max_seq_len=2048):
//   - steps 0-5: rows 0-11 chunked prefill (64 tokens/step, L -> 384),
//   - steps 6-95: rows 0-11 decode; rows 12-13 run rotating prefill bursts
//     with releases (6x64 then release); rows 14-15 cycle release ->
//     fork-from-a-decode-row (copy-on-write tail) -> decode.
// Prints the average latency of one full sequence replay.
// ============================================================================

#include "paged_cow_prefill_decode_common.h"

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
    CpdRunSpec run;
    int pool_variant;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> op_kind;
    std::vector<int32_t> fork_src;
    std::vector<int32_t> new_token_count;
};

struct InputPool {
    DeviceBuffer<float> new_k;
    DeviceBuffer<float> new_v;
    DeviceBuffer<float> q;
};

int main(int argc, char** argv) {
    int iters = 5;
    if (argc > 1) {
        iters = std::atoi(argv[1]);
        if (iters < 1) iters = 1;
    }

    try {
        CUDA_CHECK(cudaSetDevice(0));

        CpdProblemSpec spec = {};
        spec.abi_version = CPD_ABI_VERSION;
        spec.B = 16;
        spec.Hq = 16;
        spec.Hkv = 4;
        spec.D = 128;
        spec.page_size = 16;
        spec.max_chunk = 64;
        spec.max_seq_len = 2048;
        spec.max_pages = 1024;
        spec.flags = 0;
        if (!cpd_validate_problem_spec(&spec)) {
            throw std::runtime_error("bad bench spec");
        }

        const int B = spec.B;
        const int C = spec.max_chunk;
        const int Hq = spec.Hq;
        const int Hkv = spec.Hkv;
        const int D = spec.D;
        const int kSteps = 96;
        const int kPools = 4;

        std::vector<InputPool> pools(kPools);
        for (int v = 0; v < kPools; ++v) {
            SplitMix64 rng(0xC0DEC0DEULL + v);
            std::vector<float> hk((size_t)B * C * Hkv * D);
            for (float& x : hk) x = rng.uniform_float(-2.0f, 2.0f);
            std::vector<float> hv((size_t)B * C * Hkv * D);
            for (float& x : hv) x = rng.uniform_float(-2.0f, 2.0f);
            std::vector<float> hq_((size_t)B * C * Hq * D);
            for (float& x : hq_) x = rng.uniform_float(-1.0f, 1.0f);

            pools[v].new_k.allocate(hk.size());
            pools[v].new_k.upload(hk);
            pools[v].new_v.allocate(hv.size());
            pools[v].new_v.upload(hv);
            pools[v].q.allocate(hq_.size());
            pools[v].q.upload(hq_);
        }

        // Build the schedule with host-side length tracking.
        std::vector<BenchStep> steps;
        std::vector<int> len(B, 0);
        int step_id = 0;

        auto push_step = [&](const std::vector<int32_t>& seqs,
                             const std::vector<int32_t>& ops,
                             const std::vector<int32_t>& srcs,
                             const std::vector<int32_t>& cnts) {
            BenchStep st;
            st.run = {};
            st.run.abi_version = CPD_ABI_VERSION;
            st.run.active_count = static_cast<int32_t>(seqs.size());
            st.run.step_id = step_id;
            st.pool_variant = step_id % kPools;
            st.active_seq = seqs;
            st.op_kind = ops;
            st.fork_src = srcs;
            st.new_token_count = cnts;
            steps.push_back(std::move(st));

            for (size_t i = 0; i < seqs.size(); ++i) {
                if (ops[i] == CPD_OP_RELEASE) len[seqs[i]] = 0;
            }
            for (size_t i = 0; i < seqs.size(); ++i) {
                if (ops[i] == CPD_OP_FORK_APPEND) len[seqs[i]] = len[srcs[i]];
            }
            for (size_t i = 0; i < seqs.size(); ++i) {
                if (ops[i] != CPD_OP_RELEASE) len[seqs[i]] += cnts[i];
            }
            for (int b = 0; b < B; ++b) {
                if (len[b] > spec.max_seq_len) {
                    throw std::runtime_error("bench schedule overflows cap");
                }
            }
            ++step_id;
        };

        for (int s = 0; s < kSteps; ++s) {
            std::vector<int32_t> seqs, ops, srcs, cnts;

            if (s < 6) {
                for (int b = 0; b < 12; ++b) {
                    seqs.push_back(b);
                    ops.push_back(CPD_OP_APPEND);
                    srcs.push_back(0);
                    cnts.push_back(64);
                }
            } else {
                for (int b = 0; b < 12; ++b) {
                    seqs.push_back(b);
                    ops.push_back(CPD_OP_APPEND);
                    srcs.push_back(0);
                    cnts.push_back(1);
                }
                // Rows 12-13: rotating prefill bursts (6x64 then release).
                for (int r = 12; r <= 13; ++r) {
                    const int phase = (s - 6 + (r - 12) * 3) % 7;
                    if (phase == 6) {
                        if (len[r] > 0) {
                            seqs.push_back(r);
                            ops.push_back(CPD_OP_RELEASE);
                            srcs.push_back(0);
                            cnts.push_back(0);
                        }
                    } else {
                        seqs.push_back(r);
                        ops.push_back(CPD_OP_APPEND);
                        srcs.push_back(0);
                        cnts.push_back(64);
                    }
                }
                // Rows 14-15: release -> fork-from-decode-row -> decode.
                for (int r = 14; r <= 15; ++r) {
                    const int phase = (s - 6 + (r - 14) * 4) % 8;
                    if (phase == 0) {
                        if (len[r] > 0) {
                            seqs.push_back(r);
                            ops.push_back(CPD_OP_RELEASE);
                            srcs.push_back(0);
                            cnts.push_back(0);
                        }
                    } else if (phase == 1) {
                        const int src = (s + r) % 12;  // decode row, live
                        seqs.push_back(r);
                        ops.push_back(CPD_OP_FORK_APPEND);
                        srcs.push_back(src);
                        cnts.push_back(1);
                    } else if (len[r] > 0) {
                        seqs.push_back(r);
                        ops.push_back(CPD_OP_APPEND);
                        srcs.push_back(0);
                        cnts.push_back(1);
                    }
                }
            }
            push_step(seqs, ops, srcs, cnts);
        }

        std::vector<DeviceBuffer<int32_t>*> d_seqs(steps.size());
        std::vector<DeviceBuffer<int32_t>*> d_ops(steps.size());
        std::vector<DeviceBuffer<int32_t>*> d_srcs(steps.size());
        std::vector<DeviceBuffer<int32_t>*> d_cnts(steps.size());
        for (size_t i = 0; i < steps.size(); ++i) {
            d_seqs[i] = new DeviceBuffer<int32_t>();
            d_seqs[i]->allocate(steps[i].active_seq.size());
            d_seqs[i]->upload(steps[i].active_seq);
            d_ops[i] = new DeviceBuffer<int32_t>();
            d_ops[i]->allocate(steps[i].op_kind.size());
            d_ops[i]->upload(steps[i].op_kind);
            d_srcs[i] = new DeviceBuffer<int32_t>();
            d_srcs[i]->allocate(steps[i].fork_src.size());
            d_srcs[i]->upload(steps[i].fork_src);
            d_cnts[i] = new DeviceBuffer<int32_t>();
            d_cnts[i]->allocate(steps[i].new_token_count.size());
            d_cnts[i]->upload(steps[i].new_token_count);
        }

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
        DeviceBuffer<int32_t> d_seq_len;
        DeviceBuffer<uint64_t> d_kv_hash;
        DeviceBuffer<uint64_t> d_pcs;
        DeviceBuffer<int32_t> d_free;
        DeviceBuffer<int32_t> d_allocs;
        DeviceBuffer<int32_t> d_frees;
        DeviceBuffer<int32_t> d_forks;
        DeviceBuffer<int32_t> d_rels;

        d_y.allocate((size_t)B * C * Hq * D);
        d_lse.allocate((size_t)B * C * Hq);
        d_seq_len.allocate(B);
        d_kv_hash.allocate(B);
        d_pcs.allocate(1);
        d_free.allocate(1);
        d_allocs.allocate(1);
        d_frees.allocate(1);
        d_forks.allocate(1);
        d_rels.allocate(1);

        CpdOutputs outputs = {};
        outputs.y = d_y.ptr;
        outputs.lse = d_lse.ptr;
        outputs.seq_len = d_seq_len.ptr;
        outputs.kv_hash = d_kv_hash.ptr;
        outputs.page_state_checksum = d_pcs.ptr;
        outputs.free_pages = d_free.ptr;
        outputs.total_allocs = d_allocs.ptr;
        outputs.total_frees = d_frees.ptr;
        outputs.total_forks = d_forks.ptr;
        outputs.total_releases = d_rels.ptr;

        std::printf(
            "bench_sequence B=%d Hq=%d Hkv=%d D=%d P=%d C=%d msl=%d pages=%d T=%d\n",
            spec.B, spec.Hq, spec.Hkv, spec.D, spec.page_size, spec.max_chunk,
            spec.max_seq_len, spec.max_pages, kSteps);

        auto run_sequence = [&]() {
            for (size_t i = 0; i < steps.size(); ++i) {
                CpdInputs inputs = {};
                inputs.active_seq = d_seqs[i]->ptr;
                inputs.op_kind = d_ops[i]->ptr;
                inputs.fork_src = d_srcs[i]->ptr;
                inputs.new_token_count = d_cnts[i]->ptr;
                inputs.new_k = pools[steps[i].pool_variant].new_k.ptr;
                inputs.new_v = pools[steps[i].pool_variant].new_v.ptr;
                inputs.q = pools[steps[i].pool_variant].q.ptr;

                CUDA_CHECK(solution_run(
                    state, &steps[i].run, &inputs, &outputs,
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

        for (size_t i = 0; i < steps.size(); ++i) {
            delete d_seqs[i];
            delete d_ops[i];
            delete d_srcs[i];
            delete d_cnts[i];
        }

        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
