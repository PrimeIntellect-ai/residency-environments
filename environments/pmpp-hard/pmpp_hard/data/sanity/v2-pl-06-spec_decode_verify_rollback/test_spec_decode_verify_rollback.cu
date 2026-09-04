// ============================================================================
// file: test_spec_decode_verify_rollback.cu
// ============================================================================

#include "spec_decode_verify_rollback_common.h"
#include "spec_decode_verify_rollback_oracle.hpp"

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

static constexpr uint64_t g_state = 0x082efa98ec4e6c89ULL;
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

    int32_t next_i32() {
        return static_cast<int32_t>(next_u64() >> 32);
    }

    int uniform_int(int lo, int hi) {
        const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u64() % span);
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
    SdvRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> draft_value;
    std::vector<int32_t> correction_value;
    std::vector<uint32_t> p_target;
    std::vector<uint32_t> p_draft;
    std::vector<uint32_t> uniform_u32;
};

struct Scenario {
    std::string name;
    SdvProblemSpec spec;
    std::vector<StepHost> steps;
};

struct StepResult {
    std::vector<int32_t> active_seq;
    std::vector<int32_t> accepted_count;
    std::vector<int32_t> new_length;
    std::vector<int64_t> live_cache_sum;
    std::vector<uint64_t> live_cache_tail_hash;
    uint64_t state_checksum = 0;
};

static SdvProblemSpec make_spec(int B, int max_len, int max_steps) {
    SdvProblemSpec spec = {};
    spec.abi_version = SDV_ABI_VERSION;
    spec.B = B;
    spec.max_len = max_len;
    spec.max_steps = max_steps;
    spec.flags = 0;

    if (!sdv_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid SdvProblemSpec generated");
    }

    return spec;
}

static void set_accept_bit(
    bool accept,
    uint32_t* p_target,
    uint32_t* p_draft,
    uint32_t* uniform_u32) {
    if (accept) {
        *p_target = 0xffffffffU;
        *p_draft = 1U;
        *uniform_u32 = 0xffffffffU;
    } else {
        *p_target = 0U;
        *p_draft = 0xffffffffU;
        *uniform_u32 = 1U;
    }
}

static std::vector<int> prefix_to_bits(int L, int prefix) {
    std::vector<int> bits((size_t)L, 0);
    for (int i = 0; i < L && i < prefix; ++i) {
        bits[(size_t)i] = 1;
    }
    return bits;
}

static std::vector<int> alternating_bits(int L) {
    std::vector<int> bits((size_t)L, 0);
    for (int i = 0; i < L; ++i) {
        bits[(size_t)i] = (i % 2 == 0) ? 1 : 0;
    }
    return bits;
}

static StepHost make_step_from_bits(
    const SdvProblemSpec& spec,
    int step_id,
    int draft_len,
    const std::vector<int32_t>& active,
    const std::vector<std::vector<int>>& accept_bits,
    uint64_t seed) {
    if (active.size() != accept_bits.size()) {
        throw std::runtime_error("active/accept_bits size mismatch");
    }

    StepHost step;
    step.run = {};
    step.run.abi_version = SDV_ABI_VERSION;
    step.run.active_count = static_cast<int32_t>(active.size());
    step.run.draft_len = draft_len;
    step.run.step_id = step_id;

    if (!sdv_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid SdvRunSpec generated");
    }

    const size_t A = active.empty() ? 1 : active.size();
    SplitMix64 rng(seed);

    step.active_seq = active;
    if (step.active_seq.empty()) {
        step.active_seq.push_back(0);
    }

    step.draft_value.assign(A * (size_t)draft_len, 0);
    step.correction_value.assign(A, 0);
    step.p_target.assign(A * (size_t)draft_len, 0);
    step.p_draft.assign(A * (size_t)draft_len, 0);
    step.uniform_u32.assign(A * (size_t)draft_len, 0);

    for (size_t a = 0; a < active.size(); ++a) {
        step.correction_value[a] = rng.next_i32();

        for (int i = 0; i < draft_len; ++i) {
            const size_t idx = a * (size_t)draft_len + (size_t)i;
            int32_t v = rng.next_i32();
            if (v == 0) {
                v = static_cast<int32_t>(i + 1 + step_id * 17);
            }
            step.draft_value[idx] = v;

            const bool accept = (i < static_cast<int>(accept_bits[a].size()) && accept_bits[a][(size_t)i] != 0);
            set_accept_bit(
                accept,
                &step.p_target[idx],
                &step.p_draft[idx],
                &step.uniform_u32[idx]);
        }
    }

    return step;
}

static StepHost make_step_prefixes(
    const SdvProblemSpec& spec,
    int step_id,
    int draft_len,
    const std::vector<int32_t>& active,
    const std::vector<int>& prefixes,
    uint64_t seed) {
    if (active.size() != prefixes.size()) {
        throw std::runtime_error("active/prefixes size mismatch");
    }

    std::vector<std::vector<int>> bits;
    bits.reserve(prefixes.size());

    for (int p : prefixes) {
        bits.push_back(prefix_to_bits(draft_len, p));
    }

    return make_step_from_bits(spec, step_id, draft_len, active, bits, seed);
}

static StepHost make_step_pattern(
    const SdvProblemSpec& spec,
    int step_id,
    int draft_len,
    const std::vector<int32_t>& active,
    int pattern,
    uint64_t seed) {
    std::vector<std::vector<int>> bits;
    bits.reserve(active.size());

    for (size_t a = 0; a < active.size(); ++a) {
        int mode = (pattern + static_cast<int>(a)) % 5;

        if (mode == 0) {
            bits.push_back(prefix_to_bits(draft_len, draft_len));          // all accept
        } else if (mode == 1) {
            bits.push_back(prefix_to_bits(draft_len, 0));                  // all reject / a=0
        } else if (mode == 2) {
            bits.push_back(alternating_bits(draft_len));                   // accepted prefix 1
        } else if (mode == 3) {
            bits.push_back(prefix_to_bits(draft_len, std::max(0, draft_len - 1)));
        } else {
            bits.push_back(prefix_to_bits(draft_len, draft_len / 2));
        }
    }

    return make_step_from_bits(spec, step_id, draft_len, active, bits, seed);
}

static Scenario make_mixed_patterns_scenario() {
    Scenario sc;
    sc.name = "mixed_patterns_cap48";
    sc.spec = make_spec(6, 48, 64);

    for (int s = 0; s < 32; ++s) {
        int L = (s % 3 == 0) ? 8 : ((s % 3 == 1) ? 4 : 2);

        std::vector<int32_t> active;
        if (s == 5) {
            active.clear();                    // all-inactive step
        } else {
            active.push_back(0);               // pressure sequence
            active.push_back((s % 5) + 1);

            if (s % 2 == 0) {
                active.push_back(((s + 2) % 5) + 1);
            }

            if (s % 7 == 0) {
                active.push_back(((s + 4) % 5) + 1);
            }
        }

        sc.steps.push_back(make_step_pattern(
            sc.spec,
            s,
            L,
            active,
            s,
            g_state ^ 0x100000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_rollback_zero_edges_scenario() {
    Scenario sc;
    sc.name = "rollback_zero_edges";
    sc.spec = make_spec(4, 32, 64);

    sc.steps.push_back(make_step_prefixes(
        sc.spec,
        0,
        8,
        {0, 1, 2},
        {0, 0, 0},
        g_state ^ 0x200000000ULL));

    sc.steps.push_back(make_step_prefixes(
        sc.spec,
        1,
        8,
        {0, 1, 2},
        {8, 8, 8},
        g_state ^ 0x200000001ULL));

    sc.steps.push_back(make_step_from_bits(
        sc.spec,
        2,
        8,
        {0, 1, 3},
        {alternating_bits(8), alternating_bits(8), prefix_to_bits(8, 0)},
        g_state ^ 0x200000002ULL));

    for (int s = 3; s < 24; ++s) {
        const int L = (s % 2 == 0) ? 4 : 2;
        std::vector<int32_t> active;
        if (s == 9) {
            active.clear();
        } else {
            active = {static_cast<int32_t>(s % 4), static_cast<int32_t>((s + 1) % 4)};
        }

        sc.steps.push_back(make_step_pattern(
            sc.spec,
            s,
            L,
            active,
            s + 2,
            g_state ^ 0x200000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_near_max_len_scenario() {
    Scenario sc;
    sc.name = "near_max_len_cap_rule";
    sc.spec = make_spec(5, 24, 64);

    for (int s = 0; s < 28; ++s) {
        int L = 8;

        std::vector<int32_t> active;
        std::vector<int> prefixes;

        if (s == 6) {
            active = {0, 1, 2, 3};
            prefixes = {0, 0, 0, 0};
        } else if (s == 13) {
            active.clear();
            prefixes.clear();
        } else {
            active.push_back(0);
            prefixes.push_back(8);

            if (s % 2 == 0) {
                active.push_back(1);
                prefixes.push_back(7);
            }

            if (s % 3 == 0) {
                active.push_back(2);
                prefixes.push_back(0);
            }

            if (s % 4 == 0) {
                active.push_back(3);
                prefixes.push_back(4);
            }

            if (s % 5 == 0) {
                active.push_back(4);
                prefixes.push_back(8);
            }
        }

        sc.steps.push_back(make_step_prefixes(
            sc.spec,
            s,
            L,
            active,
            prefixes,
            g_state ^ 0x300000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_sparse_order_scenario() {
    Scenario sc;
    sc.name = "sparse_order_exact";
    sc.spec = make_spec(8, 64, 64);

    for (int s = 0; s < 32; ++s) {
        int L = (s % 4 == 0) ? 8 : ((s % 4 == 1) ? 4 : 2);

        std::vector<int32_t> active;
        for (int b = 0; b < sc.spec.B; ++b) {
            if ((b + s) % 3 != 0 || b == 0) {
                active.push_back(b);
            }
        }

        if (s == 12) {
            active.clear();
        }

        sc.steps.push_back(make_step_pattern(
            sc.spec,
            s,
            L,
            active,
            s + 4,
            g_state ^ 0x400000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static StepHost permute_step_rows(const StepHost& src) {
    StepHost dst = src;
    const int A = src.run.active_count;
    const int L = src.run.draft_len;

    if (A == 0) {
        return dst;
    }

    dst.active_seq.resize((size_t)A);
    dst.draft_value.assign((size_t)A * (size_t)L, 0);
    dst.correction_value.assign((size_t)A, 0);
    dst.p_target.assign((size_t)A * (size_t)L, 0);
    dst.p_draft.assign((size_t)A * (size_t)L, 0);
    dst.uniform_u32.assign((size_t)A * (size_t)L, 0);

    for (int new_a = 0; new_a < A; ++new_a) {
        const int old_a = A - 1 - new_a;

        dst.active_seq[(size_t)new_a] = src.active_seq[(size_t)old_a];
        dst.correction_value[(size_t)new_a] = src.correction_value[(size_t)old_a];

        for (int i = 0; i < L; ++i) {
            const size_t dst_idx = (size_t)new_a * (size_t)L + (size_t)i;
            const size_t src_idx = (size_t)old_a * (size_t)L + (size_t)i;

            dst.draft_value[dst_idx] = src.draft_value[src_idx];
            dst.p_target[dst_idx] = src.p_target[src_idx];
            dst.p_draft[dst_idx] = src.p_draft[src_idx];
            dst.uniform_u32[dst_idx] = src.uniform_u32[src_idx];
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
        dst.steps.push_back(permute_step_rows(step));
    }

    return dst;
}

static bool compare_input_unchanged(
    const StepHost& step,
    const DeviceBuffer<int32_t>& d_active_seq,
    const DeviceBuffer<int32_t>& d_draft_value,
    const DeviceBuffer<int32_t>& d_correction_value,
    const DeviceBuffer<uint32_t>& d_p_target,
    const DeviceBuffer<uint32_t>& d_p_draft,
    const DeviceBuffer<uint32_t>& d_uniform_u32,
    std::string* error) {
    if (d_active_seq.download() != step.active_seq) {
        if (error) *error = "input active_seq modified";
        return false;
    }

    if (d_draft_value.download() != step.draft_value) {
        if (error) *error = "input draft_value modified";
        return false;
    }

    if (d_correction_value.download() != step.correction_value) {
        if (error) *error = "input correction_value modified";
        return false;
    }

    if (d_p_target.download() != step.p_target) {
        if (error) *error = "input p_target modified";
        return false;
    }

    if (d_p_draft.download() != step.p_draft) {
        if (error) *error = "input p_draft modified";
        return false;
    }

    if (d_uniform_u32.download() != step.uniform_u32) {
        if (error) *error = "input uniform_u32 modified";
        return false;
    }

    return true;
}

static bool run_step(
    const SdvProblemSpec& spec,
    const StepHost& step,
    SdvOracleState* oracle,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    StepResult* result,
    std::string* error) {
    const int A = step.run.active_count;
    const size_t out_count = std::max<size_t>(1, (size_t)A);

    DeviceBuffer<int32_t> d_active_seq;
    DeviceBuffer<int32_t> d_draft_value;
    DeviceBuffer<int32_t> d_correction_value;
    DeviceBuffer<uint32_t> d_p_target;
    DeviceBuffer<uint32_t> d_p_draft;
    DeviceBuffer<uint32_t> d_uniform_u32;

    d_active_seq.allocate(step.active_seq.size());
    d_draft_value.allocate(step.draft_value.size());
    d_correction_value.allocate(step.correction_value.size());
    d_p_target.allocate(step.p_target.size());
    d_p_draft.allocate(step.p_draft.size());
    d_uniform_u32.allocate(step.uniform_u32.size());

    d_active_seq.upload(step.active_seq);
    d_draft_value.upload(step.draft_value);
    d_correction_value.upload(step.correction_value);
    d_p_target.upload(step.p_target);
    d_p_draft.upload(step.p_draft);
    d_uniform_u32.upload(step.uniform_u32);

    GuardedDeviceBuffer<int32_t> d_accepted_count;
    GuardedDeviceBuffer<int32_t> d_new_length;
    GuardedDeviceBuffer<int64_t> d_live_cache_sum;
    GuardedDeviceBuffer<uint64_t> d_live_cache_tail_hash;
    GuardedDeviceBuffer<uint64_t> d_state_checksum;

    d_accepted_count.allocate(out_count);
    d_new_length.allocate(out_count);
    d_live_cache_sum.allocate(out_count);
    d_live_cache_tail_hash.allocate(out_count);
    d_state_checksum.allocate(1);

    SdvInputs inputs = {};
    inputs.active_seq = d_active_seq.ptr;
    inputs.draft_value = d_draft_value.ptr;
    inputs.correction_value = d_correction_value.ptr;
    inputs.p_target = d_p_target.ptr;
    inputs.p_draft = d_p_draft.ptr;
    inputs.uniform_u32 = d_uniform_u32.ptr;

    SdvOutputs outputs = {};
    outputs.accepted_count = d_accepted_count.ptr;
    outputs.new_length = d_new_length.ptr;
    outputs.live_cache_sum = d_live_cache_sum.ptr;
    outputs.live_cache_tail_hash = d_live_cache_tail_hash.ptr;
    outputs.state_checksum = d_state_checksum.ptr;

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
            d_draft_value,
            d_correction_value,
            d_p_target,
            d_p_draft,
            d_uniform_u32,
            error)) {
        return false;
    }

    if (!d_accepted_count.check_guards("accepted_count", error)) return false;
    if (!d_new_length.check_guards("new_length", error)) return false;
    if (!d_live_cache_sum.check_guards("live_cache_sum", error)) return false;
    if (!d_live_cache_tail_hash.check_guards("live_cache_tail_hash", error)) return false;
    if (!d_state_checksum.check_guards("state_checksum", error)) return false;

    const std::vector<int32_t> h_accepted_count = d_accepted_count.download_data();
    const std::vector<int32_t> h_new_length = d_new_length.download_data();
    const std::vector<int64_t> h_live_cache_sum = d_live_cache_sum.download_data();
    const std::vector<uint64_t> h_live_cache_tail_hash = d_live_cache_tail_hash.download_data();
    const std::vector<uint64_t> h_state_checksum = d_state_checksum.download_data();

    SdvHostInputsView host_inputs = {};
    host_inputs.active_seq = step.active_seq.data();
    host_inputs.draft_value = step.draft_value.data();
    host_inputs.correction_value = step.correction_value.data();
    host_inputs.p_target = step.p_target.data();
    host_inputs.p_draft = step.p_draft.data();
    host_inputs.uniform_u32 = step.uniform_u32.data();

    SdvExpected expected;
    oracle->step(step.run, host_inputs, &expected);

    SdvHostOutputsView got = {};
    got.accepted_count = h_accepted_count.data();
    got.new_length = h_new_length.data();
    got.live_cache_sum = h_live_cache_sum.data();
    got.live_cache_tail_hash = h_live_cache_tail_hash.data();
    got.state_checksum = h_state_checksum.data();

    if (!sdv_check_all_outputs(step.run, expected, got, error)) {
        return false;
    }

    if (result) {
        result->active_seq.assign(step.active_seq.begin(), step.active_seq.begin() + A);
        result->accepted_count.assign(h_accepted_count.begin(), h_accepted_count.begin() + A);
        result->new_length.assign(h_new_length.begin(), h_new_length.begin() + A);
        result->live_cache_sum.assign(h_live_cache_sum.begin(), h_live_cache_sum.begin() + A);
        result->live_cache_tail_hash.assign(h_live_cache_tail_hash.begin(), h_live_cache_tail_hash.begin() + A);
        result->state_checksum = h_state_checksum[0];
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

    SdvOracleState oracle;
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
                "scenario %-34s step %02zu/%02zu active=%d L=%d %s%s%s\n",
                sc.name.c_str(),
                i,
                sc.steps.size(),
                sc.steps[i].run.active_count,
                sc.steps[i].run.draft_len,
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

static bool compare_replay_outputs(
    const Scenario& base,
    const std::vector<StepResult>& a,
    const std::vector<StepResult>& b,
    bool require_checksum_equal,
    std::string* error) {
    (void)base;

    if (a.size() != b.size()) {
        if (error) *error = "result step count mismatch";
        return false;
    }

    for (size_t step = 0; step < a.size(); ++step) {
        if (require_checksum_equal && a[step].state_checksum != b[step].state_checksum) {
            if (error) {
                std::ostringstream oss;
                oss << "replay checksum mismatch at step " << step
                    << ": base=0x" << std::hex << a[step].state_checksum
                    << ", replay=0x" << b[step].state_checksum;
                *error = oss.str();
            }
            return false;
        }

        if (a[step].active_seq.size() != b[step].active_seq.size()) {
            if (error) {
                std::ostringstream oss;
                oss << "replay active count mismatch at step " << step;
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
                    oss << "replay missing seq " << seq << " at step " << step;
                    *error = oss.str();
                }
                return false;
            }

            if (a[step].accepted_count[row_a] != b[step].accepted_count[(size_t)row_b] ||
                a[step].new_length[row_a] != b[step].new_length[(size_t)row_b] ||
                a[step].live_cache_sum[row_a] != b[step].live_cache_sum[(size_t)row_b] ||
                a[step].live_cache_tail_hash[row_a] != b[step].live_cache_tail_hash[(size_t)row_b]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "replay row mismatch at step " << step
                        << ", seq=" << seq;
                    *error = oss.str();
                }
                return false;
            }
        }
    }

    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        std::vector<Scenario> scenarios;
        scenarios.push_back(make_mixed_patterns_scenario());
        scenarios.push_back(make_rollback_zero_edges_scenario());
        scenarios.push_back(make_near_max_len_scenario());
        scenarios.push_back(make_sparse_order_scenario());

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
                if (!compare_replay_outputs(sc, base_results, repeat_results, true, &cmp_error)) {
                    all_ok = false;
                    std::printf(
                        "scenario %-34s deterministic replay FAIL  %s\n",
                        sc.name.c_str(),
                        cmp_error.c_str());
                } else {
                    std::printf(
                        "scenario %-34s deterministic replay PASS\n",
                        sc.name.c_str());
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
                if (!compare_replay_outputs(sc, base_results, permuted_results, true, &cmp_error)) {
                    all_ok = false;
                    std::printf(
                        "scenario %-34s permuted-order replay FAIL  %s\n",
                        sc.name.c_str(),
                        cmp_error.c_str());
                } else {
                    std::printf(
                        "scenario %-34s permuted-order replay PASS\n",
                        sc.name.c_str());
                }
            }

            if (!ok_base || !ok_repeat || !ok_permuted) {
                all_ok = false;
                std::printf(
                    "scenario %-34s FAIL  %s\n",
                    sc.name.c_str(),
                    error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
