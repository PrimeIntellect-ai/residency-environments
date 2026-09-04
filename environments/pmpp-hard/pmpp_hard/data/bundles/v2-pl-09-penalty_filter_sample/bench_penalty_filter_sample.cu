// ============================================================================
// file: bench_penalty_filter_sample.cu
// ============================================================================

#include "penalty_filter_sample_common.h"
#include "pmpp_bench_digest.cuh"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0xfbb67ae8584caa73ULL;

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
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    int uniform_int(int lo, int hi) {
        const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u64() % span);
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
        if (host.size() != count) {
            throw std::runtime_error("DeviceBuffer::upload size mismatch");
        }
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
        }
    }
};

struct HostCase {
    std::string name;
    PfsRunSpec run;
    std::vector<float> logits;
    std::vector<int32_t> history_token;
    std::vector<int32_t> history_len;
    std::vector<float> repetition_penalty;
    std::vector<float> frequency_penalty;
    std::vector<float> presence_penalty;
    std::vector<float> temperature;
    std::vector<float> min_p;
    std::vector<float> uniform_u;
};

struct DeviceCase {
    HostCase host;

    DeviceBuffer<float> logits;
    DeviceBuffer<int32_t> history_token;
    DeviceBuffer<int32_t> history_len;
    DeviceBuffer<float> rp;
    DeviceBuffer<float> fp;
    DeviceBuffer<float> pp;
    DeviceBuffer<float> temp;
    DeviceBuffer<float> min_p;
    DeviceBuffer<float> uniform;

    DeviceBuffer<int32_t> selected_token;
    DeviceBuffer<int32_t> survivor_count;
    DeviceBuffer<int32_t> packed_cand_token;
    DeviceBuffer<float> packed_cand_prob;

    PfsInputs inputs;
    PfsOutputs outputs;
};

// One solution state/workspace per case SHAPE, shared by all input variants of that
// case (mirrors the original bench, which reused one state across all timed calls).
struct CaseCtx {
    PfsProblemSpec spec = {};
    void* state = nullptr;
    DeviceBuffer<uint8_t> workspace;
    size_t workspace_bytes = 0;
};

static float gen_logit(SplitMix64& rng, int distribution_id, int row, int token, int V) {
    switch (distribution_id) {
        case PFS_DIST_UNIFORM:
            return rng.uniform_float(-3.0f, 3.0f);

        case PFS_DIST_PEAKED: {
            const int peak = (row * 1315423911u + 17) % V;
            if (token == peak) return 12.0f;
            if ((token + row * 13) % 257 == 0) return rng.uniform_float(6.0f, 9.0f);
            return rng.uniform_float(-5.0f, 1.5f);
        }

        case PFS_DIST_HEAVY_TAIL: {
            const int x = (token * 1103515245u + row * 12345u) & 0x7fffffff;
            const int bucket = x % 2048;
            return 9.0f - static_cast<float>(std::sqrt(static_cast<double>(bucket)) * 0.35) +
                   rng.uniform_float(-0.20f, 0.20f);
        }

        case PFS_DIST_MANY_TIES: {
            static const float vals[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            return vals[(token + row * 7) % 5];
        }

        default:
            return rng.uniform_float(-2.0f, 2.0f);
    }
}

static HostCase make_case(
    const char* name,
    int B,
    int V,
    int H,
    int distribution_id,
    uint64_t seed) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = PFS_ABI_VERSION;
    hc.run.B = B;
    hc.run.V = V;
    hc.run.H = H;
    hc.run.seed_id = static_cast<int32_t>(seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((seed >> 32) & 0x7fffffff);

    if (!pfs_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid PfsRunSpec generated");
    }

    SplitMix64 rng(g_state ^ seed);

    hc.logits.resize((size_t)B * (size_t)V);
    hc.history_token.resize((size_t)B * (size_t)std::max(H, 1), 0);
    hc.history_len.resize((size_t)B);
    hc.repetition_penalty.resize((size_t)B);
    hc.frequency_penalty.resize((size_t)B);
    hc.presence_penalty.resize((size_t)B);
    hc.temperature.resize((size_t)B);
    hc.min_p.resize((size_t)B);
    hc.uniform_u.resize((size_t)B);

    for (int row = 0; row < B; ++row) {
        for (int token = 0; token < V; ++token) {
            hc.logits[(size_t)row * (size_t)V + (size_t)token] =
                gen_logit(rng, distribution_id, row, token, V);
        }

        const int len = H == 0 ? 0 : rng.uniform_int(0, H);
        hc.history_len[(size_t)row] = len;

        for (int h = 0; h < H; ++h) {
            hc.history_token[(size_t)row * (size_t)std::max(H, 1) + (size_t)h] =
                rng.uniform_int(0, V - 1);
        }

        hc.repetition_penalty[(size_t)row] = rng.uniform_float(1.05f, 1.65f);
        hc.frequency_penalty[(size_t)row] = rng.uniform_float(-0.05f, 0.35f);
        hc.presence_penalty[(size_t)row] = rng.uniform_float(-0.05f, 0.55f);
        hc.temperature[(size_t)row] = rng.uniform_float(0.65f, 1.75f);

        static const float minps[] = {0.0f, 0.05f, 0.2f};
        hc.min_p[(size_t)row] = minps[row % 3];
        hc.uniform_u[(size_t)row] = rng.uniform_float(0.0f, 0.999999f);
    }

    return hc;
}

struct CaseSpec {
    const char* name;
    int B;
    int V;
    int H;
    int distribution_id;
};

static const CaseSpec kCaseSpecs[] = {
    {"uniform_B4096_V1024_H32", 4096, 1024, 32, PFS_DIST_UNIFORM},
    {"ties_B1024_V4096_H128", 1024, 4096, 128, PFS_DIST_MANY_TIES},
    {"peaked_B256_V32768_H64", 256, 32768, 64, PFS_DIST_PEAKED},
    {"heavy_tail_B256_V32768_H256", 256, 32768, 256, PFS_DIST_HEAVY_TAIL},
};

static constexpr int kNumCases = sizeof(kCaseSpecs) / sizeof(kCaseSpecs[0]);

// Every timed call gets its own input variant: variants[c][k] for timed iteration k,
// plus variants[c][K] reserved for warmup so warmup never touches a digested region.
// Shape family (B / V / H / distribution) is fixed per case; only data varies.
// Data seeds come from a per-case SplitMix64 stream keyed by PMPP_BENCH_SEED, so the
// same seed yields a bit-identical variant sequence (paired digest compare intact).
static std::vector<std::vector<HostCase>> build_bench_variants(int iters) {
    std::vector<std::vector<HostCase>> variants(kNumCases);
    SplitMix64 seed_mix(pmpp::bench_seed(0x200000001b3ULL));

    for (int c = 0; c < kNumCases; ++c) {
        const CaseSpec& cs = kCaseSpecs[c];
        SplitMix64 vs(seed_mix.next_u64());
        variants[c].reserve(static_cast<size_t>(iters) + 1);
        for (int k = 0; k <= iters; ++k) {
            variants[c].push_back(make_case(
                cs.name, cs.B, cs.V, cs.H, cs.distribution_id, vs.next_u64()));
        }
    }

    return variants;
}

static void init_case_ctx(CaseCtx* ctx, const CaseSpec& cs, cudaStream_t stream) {
    ctx->spec = {};
    ctx->spec.abi_version = PFS_ABI_VERSION;
    ctx->spec.max_B = cs.B;
    ctx->spec.max_V = cs.V;
    ctx->spec.max_H = cs.H;
    ctx->spec.flags = 0;

    ctx->workspace_bytes = solution_workspace_bytes(&ctx->spec);
    if (ctx->workspace_bytes == 0) {
        throw std::runtime_error("solution_workspace_bytes returned 0");
    }

    CUDA_CHECK(solution_init(&ctx->spec, &ctx->state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ctx->workspace.allocate(ctx->workspace_bytes);
}

static DeviceCase* make_device_case(const HostCase& hc) {
    DeviceCase* dc = new DeviceCase();
    dc->host = hc;

    dc->logits.allocate(hc.logits.size());
    dc->history_token.allocate(hc.history_token.size());
    dc->history_len.allocate(hc.history_len.size());
    dc->rp.allocate(hc.repetition_penalty.size());
    dc->fp.allocate(hc.frequency_penalty.size());
    dc->pp.allocate(hc.presence_penalty.size());
    dc->temp.allocate(hc.temperature.size());
    dc->min_p.allocate(hc.min_p.size());
    dc->uniform.allocate(hc.uniform_u.size());

    dc->logits.upload(hc.logits);
    dc->history_token.upload(hc.history_token);
    dc->history_len.upload(hc.history_len);
    dc->rp.upload(hc.repetition_penalty);
    dc->fp.upload(hc.frequency_penalty);
    dc->pp.upload(hc.presence_penalty);
    dc->temp.upload(hc.temperature);
    dc->min_p.upload(hc.min_p);
    dc->uniform.upload(hc.uniform_u);

    dc->selected_token.allocate((size_t)hc.run.B);
    dc->survivor_count.allocate((size_t)hc.run.B);
    dc->packed_cand_token.allocate((size_t)hc.run.B * (size_t)hc.run.V);
    dc->packed_cand_prob.allocate((size_t)hc.run.B * (size_t)hc.run.V);

    dc->inputs = {};
    dc->inputs.logits = dc->logits.ptr;
    dc->inputs.history_token = dc->history_token.ptr;
    dc->inputs.history_len = dc->history_len.ptr;
    dc->inputs.repetition_penalty = dc->rp.ptr;
    dc->inputs.frequency_penalty = dc->fp.ptr;
    dc->inputs.presence_penalty = dc->pp.ptr;
    dc->inputs.temperature = dc->temp.ptr;
    dc->inputs.min_p = dc->min_p.ptr;
    dc->inputs.uniform_u = dc->uniform.ptr;

    dc->outputs = {};
    dc->outputs.selected_token = dc->selected_token.ptr;
    dc->outputs.survivor_count = dc->survivor_count.ptr;
    dc->outputs.packed_cand_token = dc->packed_cand_token.ptr;
    dc->outputs.packed_cand_prob = dc->packed_cand_prob.ptr;

    return dc;
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 5;
        if (argc >= 2) {
            iters = std::max(1, std::atoi(argv[1]));
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::vector<std::vector<HostCase>> host_variants = build_bench_variants(iters);
        std::vector<CaseCtx> ctx(kNumCases);
        std::vector<std::vector<DeviceCase*>> cases(kNumCases);

        for (int c = 0; c < kNumCases; ++c) {
            init_case_ctx(&ctx[c], kCaseSpecs[c], stream);
            cases[c].reserve(host_variants[c].size());
            for (const HostCase& hc : host_variants[c]) {
                cases[c].push_back(make_device_case(hc));
            }
            const HostCase& hc = host_variants[c].front();
            std::printf(
                "bench_case %-28s B=%d V=%d H=%d dist=%d variants=%d\n",
                hc.name.c_str(),
                hc.run.B,
                hc.run.V,
                hc.run.H,
                hc.run.distribution_id,
                iters);
        }

        // Warmup runs only the dedicated variant (index iters); its outputs are never
        // folded, so warmup work cannot pre-populate any digested output region.
        for (int warm = 0; warm < 3; ++warm) {
            for (int c = 0; c < kNumCases; ++c) {
                DeviceCase* dc = cases[c][static_cast<size_t>(iters)];
                CUDA_CHECK(solution_run(
                    ctx[c].state,
                    &dc->host.run,
                    &dc->inputs,
                    &dc->outputs,
                    ctx[c].workspace.ptr,
                    ctx[c].workspace_bytes,
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
            for (int c = 0; c < kNumCases; ++c) {
                DeviceCase* dc = cases[c][static_cast<size_t>(iter)];
                CUDA_CHECK(solution_run(
                    ctx[c].state,
                    &dc->host.run,
                    &dc->inputs,
                    &dc->outputs,
                    ctx[c].workspace.ptr,
                    ctx[c].workspace_bytes,
                    stream));
            }
        }

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        const double calls = static_cast<double>(iters) * static_cast<double>(kNumCases);
        const double avg_ms = static_cast<double>(elapsed_ms) / calls;

        std::printf("avg_ms=%.6f\n", avg_ms);

        // Anti-hack digest: every timed call (iter, case) wrote its own output region
        // for its own input variant; fold ALL of them so a no-op/cached timed call
        // leaves a wrong region and the digest cannot match the reference bench.
        // Per-region policy unchanged: packed_cand_prob is tolerance-graded fp32 and
        // packed_cand_token slots past survivor_count are unspecified padding, so fold
        // selected_token + survivor_count fully and packed_cand_token only over each
        // row's graded prefix.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        pmpp::OutFnv dg;
        for (int iter = 0; iter < iters; ++iter) {
            for (int ci = 0; ci < kNumCases; ++ci) {
                DeviceCase* dc = cases[ci][static_cast<size_t>(iter)];
                const size_t B = (size_t)dc->host.run.B;
                const size_t V = (size_t)dc->host.run.V;

                dg.dev(dc->selected_token.ptr, B * sizeof(int32_t));
                dg.dev(dc->survivor_count.ptr, B * sizeof(int32_t));

                std::vector<int32_t> h_count(B);
                std::vector<int32_t> h_tokens(B * V);
                CUDA_CHECK(cudaMemcpy(h_count.data(), dc->survivor_count.ptr,
                                      B * sizeof(int32_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_tokens.data(), dc->packed_cand_token.ptr,
                                      B * V * sizeof(int32_t), cudaMemcpyDeviceToHost));
                for (size_t row = 0; row < B; ++row) {
                    int32_t c = h_count[row];
                    if (c < 0) c = 0;
                    if ((size_t)c > V) c = (int32_t)V;
                    dg.bytes(h_tokens.data() + row * V, (size_t)c * sizeof(int32_t));
                }
            }
        }
        dg.print();

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (int c = 0; c < kNumCases; ++c) {
            solution_destroy(ctx[c].state);
            for (DeviceCase* dc : cases[c]) {
                delete dc;
            }
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
