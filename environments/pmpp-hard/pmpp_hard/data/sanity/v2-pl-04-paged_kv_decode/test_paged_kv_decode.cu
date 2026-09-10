// ============================================================================
// file: test_paged_kv_decode.cu
// ============================================================================

#include "paged_kv_decode_common.h"
#include "paged_kv_decode_oracle.hpp"

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

static constexpr uint64_t g_state = 0xa4093822299f31d0ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

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
        const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u64() % span);
    }

    float uniform_float(float lo, float hi) {
        const float u = static_cast<float>((next_u64() >> 11) * 0x1.0p-53);
        return lo + (hi - lo) * u;
    }

    bool chance(float p) {
        return uniform_float(0.0f, 1.0f) < p;
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

    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost));
        }
        return host;
    }
};

template <typename T>
struct GuardedDeviceBuffer {
    uint8_t* raw = nullptr;
    T* ptr = nullptr;
    size_t count = 0;
    size_t data_bytes = 0;
    size_t total_bytes = 0;

    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

    ~GuardedDeviceBuffer() {
        if (raw) {
            cudaFree(raw);
        }
    }

    void allocate(size_t n) {
        count = n;
        data_bytes = sizeof(T) * count;
        total_bytes = kGuardBytes + data_bytes + kGuardBytes;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), total_bytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, total_bytes));
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }

    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        }
        return host;
    }

    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> before(kGuardBytes);
        std::vector<uint8_t> after(kGuardBytes);

        CUDA_CHECK(cudaMemcpy(before.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (before[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "left guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }
        }

        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (after[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "right guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }
        }

        return true;
    }
};

struct StepHost {
    PkdRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> new_token_count;
    std::vector<float> new_k;
    std::vector<float> new_v;
    std::vector<float> new_scale;
    std::vector<float> q;
};

struct Scenario {
    std::string name;
    PkdProblemSpec spec;
    std::vector<StepHost> steps;
};

struct StepResult {
    std::vector<int32_t> active_seq;
    std::vector<float> y;
    std::vector<int32_t> lengths;
    uint64_t checksum = 0;
};

static PkdProblemSpec make_spec(
    int B,
    int Hq,
    int Hkv,
    int D,
    int page_size,
    int max_seq_len) {
    PkdProblemSpec spec = {};
    spec.abi_version = PKD_ABI_VERSION;
    spec.B = B;
    spec.Hq = Hq;
    spec.Hkv = Hkv;
    spec.D = D;
    spec.page_size = page_size;
    spec.max_seq_len = max_seq_len;
    spec.max_pages = B * pkd_pages_per_seq(max_seq_len, page_size);
    spec.flags = 0;

    if (!pkd_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid PkdProblemSpec generated");
    }

    return spec;
}

static void fill_step_tensors(
    const PkdProblemSpec& spec,
    StepHost* step,
    uint64_t seed) {
    const int A = step->run.active_count;
    const int Hq = spec.Hq;
    const int Hkv = spec.Hkv;
    const int D = spec.D;

    const size_t a_count = A > 0 ? static_cast<size_t>(A) : 1;

    step->new_token_count.resize(a_count, 0);
    step->new_k.resize(a_count * PKD_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D, 0.0f);
    step->new_v.resize(a_count * PKD_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D, 0.0f);
    step->new_scale.resize(a_count * PKD_MAX_NEW_TOKENS * (size_t)Hkv, 1.0f);
    step->q.resize(a_count * (size_t)Hq * (size_t)D, 0.0f);

    for (int a = 0; a < A; ++a) {
        const int seq = step->active_seq[(size_t)a];
        SplitMix64 rng(seed ^ (0x8e3779b97f4a7c15ULL * (uint64_t)(seq + 1)));

        for (int nt = 0; nt < PKD_MAX_NEW_TOKENS; ++nt) {
            for (int h = 0; h < Hkv; ++h) {
                const size_t scale_idx =
                    ((size_t)a * (size_t)PKD_MAX_NEW_TOKENS + (size_t)nt) *
                    (size_t)Hkv + (size_t)h;
                step->new_scale[scale_idx] = rng.uniform_float(0.035f, 0.090f);
            }
        }

        for (int nt = 0; nt < PKD_MAX_NEW_TOKENS; ++nt) {
            for (int h = 0; h < Hkv; ++h) {
                for (int d = 0; d < D; ++d) {
                    const size_t idx =
                        ((size_t)a * (size_t)PKD_MAX_NEW_TOKENS + (size_t)nt) *
                        (size_t)Hkv * (size_t)D +
                        (size_t)h * (size_t)D +
                        (size_t)d;

                    step->new_k[idx] = rng.uniform_float(-2.25f, 2.25f);
                    step->new_v[idx] = rng.uniform_float(-2.25f, 2.25f);
                }
            }
        }

        for (int h = 0; h < Hq; ++h) {
            for (int d = 0; d < D; ++d) {
                const size_t idx =
                    ((size_t)a * (size_t)Hq + (size_t)h) *
                    (size_t)D + (size_t)d;
                step->q[idx] = rng.uniform_float(-0.20f, 0.20f);
            }
        }
    }
}

static StepHost make_step(
    const PkdProblemSpec& spec,
    int step_id,
    int window_size,
    const std::vector<int32_t>& active,
    const std::vector<int32_t>& ntokens,
    uint64_t seed) {
    if (active.size() != ntokens.size()) {
        throw std::runtime_error("active/ntokens size mismatch");
    }

    StepHost step;
    step.run = {};
    step.run.abi_version = PKD_ABI_VERSION;
    step.run.active_count = static_cast<int32_t>(active.size());
    step.run.step_id = step_id;
    step.run.window_size = window_size;

    if (!pkd_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid PkdRunSpec generated");
    }

    step.active_seq = active;
    if (step.active_seq.empty()) {
        step.active_seq.push_back(0);
    }

    fill_step_tensors(spec, &step, seed);

    for (size_t i = 0; i < ntokens.size(); ++i) {
        step.new_token_count[i] = ntokens[i];
    }

    return step;
}

static Scenario make_pressure_scenario() {
    Scenario sc;
    sc.name = "pressure_maxlen_D64_P16";
    sc.spec = make_spec(8, 4, 2, 64, 16, 64);

    for (int s = 0; s < 32; ++s) {
        int window = (s < 10) ? 64 : 17;

        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        if (s == 5) {
            active.clear();
            nt.clear();
        } else if (s == 9) {
            active = {0, 1, 2, 3};
            nt = {0, 0, 0, 0};
        } else {
            active.push_back(0);
            nt.push_back(2);

            active.push_back((s % 7) + 1);
            nt.push_back((s % 3 == 0) ? 2 : 1);

            if (s % 2 == 0) {
                active.push_back(((s + 2) % 7) + 1);
                nt.push_back(1);
            }

            if (s % 5 == 0) {
                active.push_back(((s + 4) % 7) + 1);
                nt.push_back(0);
            }
        }

        sc.steps.push_back(make_step(
            sc.spec,
            s,
            window,
            active,
            nt,
            g_state ^ 0x100000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_gqa_d128_scenario() {
    Scenario sc;
    sc.name = "gqa_D128_P32_skew";
    sc.spec = make_spec(6, 8, 2, 128, 32, 96);

    for (int s = 0; s < 20; ++s) {
        int window = (s % 4 == 0) ? 24 : 96;

        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        if (s == 3) {
            active.clear();
            nt.clear();
        } else if (s == 6) {
            active = {5, 3, 1};
            nt = {0, 0, 0};
        } else {
            for (int b = 0; b < sc.spec.B; ++b) {
                if ((b + s) % 2 == 0 || b == 1) {
                    active.push_back(b);
                    nt.push_back((b == 1 || s % 3 == 0) ? 2 : 1);
                }
            }
        }

        sc.steps.push_back(make_step(
            sc.spec,
            s,
            window,
            active,
            nt,
            g_state ^ 0x200000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_sliding_boundary_scenario() {
    Scenario sc;
    sc.name = "sliding_window_page_boundary";
    sc.spec = make_spec(4, 4, 1, 64, 16, 40);

    for (int s = 0; s < 24; ++s) {
        const int window = (s < 8) ? 40 : ((s % 2 == 0) ? 16 : 17);

        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        if (s == 12) {
            active = {0, 1, 2, 3};
            nt = {0, 0, 0, 0};
        } else {
            active = {0, 1};
            nt = {2, (s % 2 == 0) ? 1 : 2};

            if (s % 3 == 0) {
                active.push_back(2);
                nt.push_back(1);
            }

            if (s % 5 == 1) {
                active.push_back(3);
                nt.push_back(2);
            }
        }

        sc.steps.push_back(make_step(
            sc.spec,
            s,
            window,
            active,
            nt,
            g_state ^ 0x300000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static StepHost permute_step_rows(const PkdProblemSpec& spec, const StepHost& src) {
    StepHost dst = src;
    const int A = src.run.active_count;
    const int Hq = spec.Hq;
    const int Hkv = spec.Hkv;
    const int D = spec.D;

    if (A == 0) {
        return dst;
    }

    dst.active_seq.resize((size_t)A);
    dst.new_token_count.assign((size_t)A, 0);
    dst.new_k.assign((size_t)A * PKD_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D, 0.0f);
    dst.new_v.assign((size_t)A * PKD_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D, 0.0f);
    dst.new_scale.assign((size_t)A * PKD_MAX_NEW_TOKENS * (size_t)Hkv, 1.0f);
    dst.q.assign((size_t)A * (size_t)Hq * (size_t)D, 0.0f);

    for (int new_a = 0; new_a < A; ++new_a) {
        const int old_a = A - 1 - new_a;
        dst.active_seq[(size_t)new_a] = src.active_seq[(size_t)old_a];
        dst.new_token_count[(size_t)new_a] = src.new_token_count[(size_t)old_a];

        for (int nt = 0; nt < PKD_MAX_NEW_TOKENS; ++nt) {
            for (int h = 0; h < Hkv; ++h) {
                const size_t dst_sidx =
                    ((size_t)new_a * PKD_MAX_NEW_TOKENS + (size_t)nt) *
                    (size_t)Hkv + (size_t)h;
                const size_t src_sidx =
                    ((size_t)old_a * PKD_MAX_NEW_TOKENS + (size_t)nt) *
                    (size_t)Hkv + (size_t)h;
                dst.new_scale[dst_sidx] = src.new_scale[src_sidx];

                for (int d = 0; d < D; ++d) {
                    const size_t dst_idx =
                        ((size_t)new_a * PKD_MAX_NEW_TOKENS + (size_t)nt) *
                        (size_t)Hkv * (size_t)D +
                        (size_t)h * (size_t)D + (size_t)d;
                    const size_t src_idx =
                        ((size_t)old_a * PKD_MAX_NEW_TOKENS + (size_t)nt) *
                        (size_t)Hkv * (size_t)D +
                        (size_t)h * (size_t)D + (size_t)d;

                    dst.new_k[dst_idx] = src.new_k[src_idx];
                    dst.new_v[dst_idx] = src.new_v[src_idx];
                }
            }
        }

        for (int h = 0; h < Hq; ++h) {
            for (int d = 0; d < D; ++d) {
                const size_t dst_idx =
                    ((size_t)new_a * (size_t)Hq + (size_t)h) *
                    (size_t)D + (size_t)d;
                const size_t src_idx =
                    ((size_t)old_a * (size_t)Hq + (size_t)h) *
                    (size_t)D + (size_t)d;

                dst.q[dst_idx] = src.q[src_idx];
            }
        }
    }

    return dst;
}

static Scenario make_permuted_scenario(const Scenario& src) {
    Scenario dst;
    dst.name = src.name + "_reversed_active_order";
    dst.spec = src.spec;
    dst.steps.reserve(src.steps.size());

    for (const StepHost& step : src.steps) {
        dst.steps.push_back(permute_step_rows(src.spec, step));
    }

    return dst;
}

static bool compare_input_unchanged(
    const StepHost& step,
    const DeviceBuffer<int32_t>& d_active_seq,
    const DeviceBuffer<int32_t>& d_new_token_count,
    const DeviceBuffer<float>& d_new_k,
    const DeviceBuffer<float>& d_new_v,
    const DeviceBuffer<float>& d_new_scale,
    const DeviceBuffer<float>& d_q,
    std::string* error) {
    if (d_active_seq.download() != step.active_seq) {
        if (error) *error = "input active_seq modified";
        return false;
    }

    if (d_new_token_count.download() != step.new_token_count) {
        if (error) *error = "input new_token_count modified";
        return false;
    }

    if (d_new_k.download() != step.new_k) {
        if (error) *error = "input new_k modified";
        return false;
    }

    if (d_new_v.download() != step.new_v) {
        if (error) *error = "input new_v modified";
        return false;
    }

    if (d_new_scale.download() != step.new_scale) {
        if (error) *error = "input new_scale modified";
        return false;
    }

    if (d_q.download() != step.q) {
        if (error) *error = "input q modified";
        return false;
    }

    return true;
}

static bool run_step(
    const PkdProblemSpec& spec,
    const StepHost& step,
    PkdOracleState* oracle,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    StepResult* result,
    std::string* error) {
    const int A = step.run.active_count;
    const int Hq = spec.Hq;
    const int D = spec.D;
    const size_t y_count = std::max<size_t>(1, (size_t)A * (size_t)Hq * (size_t)D);

    DeviceBuffer<int32_t> d_active_seq;
    DeviceBuffer<int32_t> d_new_token_count;
    DeviceBuffer<float> d_new_k;
    DeviceBuffer<float> d_new_v;
    DeviceBuffer<float> d_new_scale;
    DeviceBuffer<float> d_q;

    d_active_seq.allocate(step.active_seq.size());
    d_new_token_count.allocate(step.new_token_count.size());
    d_new_k.allocate(step.new_k.size());
    d_new_v.allocate(step.new_v.size());
    d_new_scale.allocate(step.new_scale.size());
    d_q.allocate(step.q.size());

    d_active_seq.upload(step.active_seq);
    d_new_token_count.upload(step.new_token_count);
    d_new_k.upload(step.new_k);
    d_new_v.upload(step.new_v);
    d_new_scale.upload(step.new_scale);
    d_q.upload(step.q);

    GuardedDeviceBuffer<float> d_y;
    GuardedDeviceBuffer<int32_t> d_lengths;
    GuardedDeviceBuffer<uint64_t> d_checksum;

    d_y.allocate(y_count);
    d_lengths.allocate((size_t)spec.B);
    d_checksum.allocate(1);

    PkdInputs inputs = {};
    inputs.active_seq = d_active_seq.ptr;
    inputs.new_token_count = d_new_token_count.ptr;
    inputs.new_k = d_new_k.ptr;
    inputs.new_v = d_new_v.ptr;
    inputs.new_scale = d_new_scale.ptr;
    inputs.q = d_q.ptr;

    PkdOutputs outputs = {};
    outputs.y = d_y.ptr;
    outputs.lengths = d_lengths.ptr;
    outputs.state_checksum = d_checksum.ptr;

    CUDA_CHECK(solution_run(
        state,
        &step.run,
        &inputs,
        &outputs,
        workspace,
        workspace_bytes,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!compare_input_unchanged(
            step,
            d_active_seq,
            d_new_token_count,
            d_new_k,
            d_new_v,
            d_new_scale,
            d_q,
            error)) {
        return false;
    }

    if (!d_y.check_guards("y", error)) return false;
    if (!d_lengths.check_guards("lengths", error)) return false;
    if (!d_checksum.check_guards("state_checksum", error)) return false;

    const std::vector<float> h_y = d_y.download_data();
    const std::vector<int32_t> h_lengths = d_lengths.download_data();
    const std::vector<uint64_t> h_checksum = d_checksum.download_data();

    PkdHostInputsView host_inputs = {};
    host_inputs.active_seq = step.active_seq.data();
    host_inputs.new_token_count = step.new_token_count.data();
    host_inputs.new_k = step.new_k.data();
    host_inputs.new_v = step.new_v.data();
    host_inputs.new_scale = step.new_scale.data();
    host_inputs.q = step.q.data();

    PkdExpected expected;
    oracle->step(step.run, host_inputs, &expected);

    PkdHostOutputsView got = {};
    got.y = h_y.data();
    got.lengths = h_lengths.data();
    got.state_checksum = h_checksum.data();

    if (!pkd_check_all_outputs(step.run, spec, expected, got, error)) {
        return false;
    }

    if (result) {
        result->active_seq.assign(step.active_seq.begin(), step.active_seq.begin() + A);
        result->y.assign(h_y.begin(), h_y.begin() + ((size_t)A * (size_t)Hq * (size_t)D));
        result->lengths = h_lengths;
        result->checksum = h_checksum[0];
    }

    return true;
}

static bool run_scenario_once(
    const Scenario& sc,
    bool verbose,
    std::vector<StepResult>* results,
    int* passed_steps,
    int* total_steps,
    std::string* error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec); if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/
    if (workspace_bytes == 0) {
        if (error) *error = "solution_workspace_bytes returned 0";
        return false;
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(workspace_bytes);

    PkdOracleState oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    bool all_ok = true;

    if (results) {
        results->clear();
        results->reserve(sc.steps.size());
    }

    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult step_result;
        std::string step_error;

        const bool ok = run_step(
            sc.spec,
            sc.steps[i],
            &oracle,
            state,
            workspace.ptr,
            workspace_bytes,
            stream,
            results ? &step_result : nullptr,
            &step_error);

        ++(*total_steps);
        if (ok) {
            ++(*passed_steps);
        } else {
            all_ok = false;
            if (error && error->empty()) {
                std::ostringstream oss;
                oss << sc.name << " step " << i << ": " << step_error;
                *error = oss.str();
            }
        }

        if (results) {
            results->push_back(step_result);
        }

        if (verbose) {
            std::printf(
                "scenario %-34s step %02zu/%02zu active=%d window=%d %s%s%s\n",
                sc.name.c_str(),
                i,
                sc.steps.size(),
                sc.steps[i].run.active_count,
                sc.steps[i].run.window_size,
                ok ? "PASS" : "FAIL",
                ok ? "" : "  ",
                ok ? "" : step_error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));

    return all_ok;
}

static int find_active_row(const std::vector<int32_t>& active, int seq) {
    for (size_t i = 0; i < active.size(); ++i) {
        if (active[i] == seq) return static_cast<int>(i);
    }
    return -1;
}

static bool compare_order_invariant_outputs(
    const Scenario& base,
    const std::vector<StepResult>& a,
    const std::vector<StepResult>& b,
    std::string* error) {
    const int Hq = base.spec.Hq;
    const int D = base.spec.D;

    if (a.size() != b.size()) {
        if (error) *error = "result step count mismatch";
        return false;
    }

    for (size_t step = 0; step < a.size(); ++step) {
        if (a[step].lengths != b[step].lengths) {
            if (error) {
                std::ostringstream oss;
                oss << "permuted replay length mismatch at step " << step;
                *error = oss.str();
            }
            return false;
        }

        if (a[step].active_seq.size() != b[step].active_seq.size()) {
            if (error) {
                std::ostringstream oss;
                oss << "permuted replay active count mismatch at step " << step;
                *error = oss.str();
            }
            return false;
        }

        for (size_t row_a = 0; row_a < a[step].active_seq.size(); ++row_a) {
            const int seq = a[step].active_seq[row_a];
            const int row_b = find_active_row(b[step].active_seq, seq);
            if (row_b < 0) {
                if (error) {
                    std::ostringstream oss;
                    oss << "permuted replay missing seq " << seq
                        << " at step " << step;
                    *error = oss.str();
                }
                return false;
            }

            for (int h = 0; h < Hq; ++h) {
                for (int d = 0; d < D; ++d) {
                    const size_t idx_a =
                        ((size_t)row_a * (size_t)Hq + (size_t)h) *
                        (size_t)D + (size_t)d;
                    const size_t idx_b =
                        ((size_t)row_b * (size_t)Hq + (size_t)h) *
                        (size_t)D + (size_t)d;

                    const float av = a[step].y[idx_a];
                    const float bv = b[step].y[idx_b];
                    const float diff = std::fabs(av - bv);
                    const float tol = PKD_Y_ATOL + PKD_Y_RTOL * std::fabs(av);

                    if (!(diff <= tol)) {
                        if (error) {
                            std::ostringstream oss;
                            oss << "permuted replay y mismatch at step " << step
                                << ", seq=" << seq
                                << ", h=" << h
                                << ", d=" << d
                                << ": base=" << av
                                << ", permuted=" << bv
                                << ", diff=" << diff
                                << ", tol=" << tol;
                            *error = oss.str();
                        }
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        std::vector<Scenario> scenarios;
        scenarios.push_back(make_pressure_scenario());
        scenarios.push_back(make_gqa_d128_scenario());
        scenarios.push_back(make_sliding_boundary_scenario());

        int passed = 0;
        int total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results;
            std::vector<StepResult> repeat_results;
            std::vector<StepResult> permuted_results;

            std::string error;

            const bool ok_base = run_scenario_once(
                sc,
                true,
                &base_results,
                &passed,
                &total,
                &error);

            Scenario exact_replay = sc;
            exact_replay.name = sc.name + "_exact_replay";

            const bool ok_repeat = run_scenario_once(
                exact_replay,
                true,
                &repeat_results,
                &passed,
                &total,
                &error);

            if (ok_base && ok_repeat) {
                std::string cmp_error;
                if (!compare_order_invariant_outputs(sc, base_results, repeat_results, &cmp_error)) {
                    all_ok = false;
                    std::printf("scenario %-34s deterministic replay FAIL  %s\n",
                                sc.name.c_str(),
                                cmp_error.c_str());
                } else {
                    std::printf("scenario %-34s deterministic replay PASS\n", sc.name.c_str());
                }
            }

            Scenario permuted = make_permuted_scenario(sc);
            const bool ok_permuted = run_scenario_once(
                permuted,
                true,
                &permuted_results,
                &passed,
                &total,
                &error);

            if (ok_base && ok_permuted) {
                std::string cmp_error;
                if (!compare_order_invariant_outputs(sc, base_results, permuted_results, &cmp_error)) {
                    all_ok = false;
                    std::printf("scenario %-34s permuted-order y/length replay FAIL  %s\n",
                                sc.name.c_str(),
                                cmp_error.c_str());
                } else {
                    std::printf("scenario %-34s permuted-order y/length replay PASS\n",
                                sc.name.c_str());
                }
            }

            if (!ok_base || !ok_repeat || !ok_permuted) {
                all_ok = false;
                std::printf("scenario %-34s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
