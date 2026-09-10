// ============================================================================
// file: test_multi_checkpoint_tree_rollback.cu
// ============================================================================

#include "multi_checkpoint_tree_rollback_common.h"
#include "multi_checkpoint_tree_rollback_oracle.hpp"

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

static constexpr uint64_t g_state = 0x5e0f8d7c3b2a1906ULL;
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
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
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
        if (raw) cudaFree(raw);
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
    MctrRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> op_code;
    std::vector<int32_t> token_count;
    std::vector<int32_t> accept_count;
    std::vector<int32_t> checkpoint_id;
    std::vector<int32_t> token_values;
    std::vector<int32_t> correction_value;
};

struct Scenario {
    std::string name;
    MctrProblemSpec spec;
    std::vector<StepHost> steps;
};

struct StepResult {
    std::vector<int64_t> live_cache_sum;
    std::vector<uint64_t> live_cache_tail_hash;
    std::vector<int32_t> length;
    std::vector<int32_t> num_checkpoints;
    std::vector<int32_t> top_checkpoint_len;
    uint64_t state_checksum = 0;
};

static MctrProblemSpec make_spec(int B, int max_len, int max_steps, int max_depth) {
    MctrProblemSpec spec = {};
    spec.abi_version = MCTR_ABI_VERSION;
    spec.B = B;
    spec.max_len = max_len;
    spec.max_steps = max_steps;
    spec.max_depth = max_depth;
    spec.flags = 0;

    if (!mctr_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid MctrProblemSpec generated");
    }

    return spec;
}

static StepHost make_step(
    const MctrProblemSpec& spec,
    int step_id,
    const std::vector<int32_t>& seqs,
    const std::vector<int32_t>& ops,
    const std::vector<int32_t>& ks,
    const std::vector<int32_t>& accepts,
    const std::vector<int32_t>& ckpt_ids,
    SplitMix64& rng) {
    const int A = static_cast<int>(seqs.size());

    if ((int)ops.size() != A ||
        (int)ks.size() != A ||
        (int)accepts.size() != A ||
        (int)ckpt_ids.size() != A) {
        throw std::runtime_error("step vector size mismatch");
    }

    StepHost step;
    step.run = {};
    step.run.abi_version = MCTR_ABI_VERSION;
    step.run.active_count = A;
    step.run.step_id = step_id;

    if (!mctr_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid MctrRunSpec generated");
    }

    const size_t rows = std::max<size_t>(1, (size_t)A);

    step.active_seq.assign(rows, 0);
    step.op_code.assign(rows, MCTR_OP_NOOP);
    step.token_count.assign(rows, 0);
    step.accept_count.assign(rows, 0);
    step.checkpoint_id.assign(rows, 0);
    step.token_values.assign(rows * (size_t)MCTR_MAX_K, 0);
    step.correction_value.assign(rows, 0);

    for (int r = 0; r < A; ++r) {
        step.active_seq[(size_t)r] = seqs[(size_t)r];
        step.op_code[(size_t)r] = ops[(size_t)r];
        step.token_count[(size_t)r] = ks[(size_t)r];
        step.accept_count[(size_t)r] = accepts[(size_t)r];
        step.checkpoint_id[(size_t)r] = ckpt_ids[(size_t)r];
        step.correction_value[(size_t)r] = rng.next_i32();

        for (int i = 0; i < MCTR_MAX_K; ++i) {
            int32_t v = rng.next_i32();
            if (v == 0) v = step_id * 101 + r * 17 + i + 1;
            step.token_values[(size_t)r * (size_t)MCTR_MAX_K + (size_t)i] = v;
        }
    }

    return step;
}

static StepHost permute_step_reverse(const StepHost& src) {
    StepHost dst = src;
    const int A = src.run.active_count;
    const size_t rows = std::max<size_t>(1, (size_t)A);

    dst.active_seq.assign(rows, 0);
    dst.op_code.assign(rows, MCTR_OP_NOOP);
    dst.token_count.assign(rows, 0);
    dst.accept_count.assign(rows, 0);
    dst.checkpoint_id.assign(rows, 0);
    dst.token_values.assign(rows * (size_t)MCTR_MAX_K, 0);
    dst.correction_value.assign(rows, 0);

    for (int r = 0; r < A; ++r) {
        const int src_r = A - 1 - r;

        dst.active_seq[(size_t)r] = src.active_seq[(size_t)src_r];
        dst.op_code[(size_t)r] = src.op_code[(size_t)src_r];
        dst.token_count[(size_t)r] = src.token_count[(size_t)src_r];
        dst.accept_count[(size_t)r] = src.accept_count[(size_t)src_r];
        dst.checkpoint_id[(size_t)r] = src.checkpoint_id[(size_t)src_r];
        dst.correction_value[(size_t)r] = src.correction_value[(size_t)src_r];

        for (int i = 0; i < MCTR_MAX_K; ++i) {
            dst.token_values[(size_t)r * (size_t)MCTR_MAX_K + (size_t)i] =
                src.token_values[(size_t)src_r * (size_t)MCTR_MAX_K + (size_t)i];
        }
    }

    return dst;
}

static Scenario make_permuted_scenario(const Scenario& src) {
    Scenario dst;
    dst.name = src.name + "_permuted_active_order";
    dst.spec = src.spec;
    dst.steps.reserve(src.steps.size());

    for (const StepHost& step : src.steps) {
        dst.steps.push_back(permute_step_reverse(step));
    }

    return dst;
}

static Scenario make_nested_checkpoint_scenario() {
    Scenario sc;
    sc.name = "nested_checkpoint_reappend";
    sc.spec = make_spec(8, 96, 64, 8);

    SplitMix64 rng(g_state ^ 0x10101010ULL);

    sc.steps.push_back(make_step(sc.spec, 0, {0, 1, 2, 3}, {MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT}, {0, 0, 0, 0}, {0, 0, 0, 0}, {100, 100, 100, 100}, rng));
    sc.steps.push_back(make_step(sc.spec, 1, {0, 1, 2, 3}, {MCTR_OP_APPEND, MCTR_OP_APPEND, MCTR_OP_APPEND, MCTR_OP_APPEND}, {4, 2, 1, 3}, {0, 0, 0, 0}, {0, 0, 0, 0}, rng));
    sc.steps.push_back(make_step(sc.spec, 2, {0, 1, 2, 3}, {MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT}, {0, 0, 0, 0}, {0, 0, 0, 0}, {200, 200, 200, 200}, rng));
    sc.steps.push_back(make_step(sc.spec, 3, {0, 1, 2, 3}, {MCTR_OP_ACCEPT_PREFIX, MCTR_OP_ACCEPT_PREFIX, MCTR_OP_ACCEPT_PREFIX, MCTR_OP_ACCEPT_PREFIX}, {4, 4, 2, 1}, {0, 4, 1, 0}, {0, 0, 0, 0}, rng));
    sc.steps.push_back(make_step(sc.spec, 4, {0, 1, 2, 3}, {MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT}, {0, 0, 0, 0}, {0, 0, 0, 0}, {300, 300, 300, 300}, rng));
    sc.steps.push_back(make_step(sc.spec, 5, {0, 1, 2, 3}, {MCTR_OP_APPEND, MCTR_OP_APPEND, MCTR_OP_APPEND, MCTR_OP_APPEND}, {2, 2, 4, 4}, {0, 0, 0, 0}, {0, 0, 0, 0}, rng));
    sc.steps.push_back(make_step(sc.spec, 6, {0, 1, 2, 3}, {MCTR_OP_ROLLBACK_TO, MCTR_OP_ROLLBACK_TO, MCTR_OP_ROLLBACK_TO, MCTR_OP_ROLLBACK_TO}, {0, 0, 0, 0}, {0, 0, 0, 0}, {200, 999, 300, 100}, rng));
    sc.steps.push_back(make_step(sc.spec, 7, {0, 1, 2, 3}, {MCTR_OP_APPEND, MCTR_OP_APPEND, MCTR_OP_ACCEPT_PREFIX, MCTR_OP_SAVE_CHECKPOINT}, {4, 4, 4, 0}, {0, 0, 2, 0}, {0, 0, 0, 400}, rng));

    for (int s = 8; s < 32; ++s) {
        std::vector<int32_t> seqs;
        std::vector<int32_t> ops;
        std::vector<int32_t> ks;
        std::vector<int32_t> accepts;
        std::vector<int32_t> ckpts;

        for (int r = 0; r < 4; ++r) {
            const int seq = (s + r) % sc.spec.B;
            seqs.push_back(seq);

            const int mode = (s + r) % 5;
            if (mode == 0) {
                ops.push_back(MCTR_OP_APPEND);
                ks.push_back((s % 4) + 1);
                accepts.push_back(0);
                ckpts.push_back(0);
            } else if (mode == 1) {
                ops.push_back(MCTR_OP_SAVE_CHECKPOINT);
                ks.push_back(0);
                accepts.push_back(0);
                ckpts.push_back(500 + s + r);
            } else if (mode == 2) {
                ops.push_back(MCTR_OP_ACCEPT_PREFIX);
                ks.push_back(4);
                accepts.push_back((s + r) % 5);
                ckpts.push_back(0);
            } else if (mode == 3) {
                ops.push_back(MCTR_OP_ROLLBACK_TO);
                ks.push_back(0);
                accepts.push_back(0);
                ckpts.push_back(((s + r) % 2 == 0) ? 100 : 500 + (s - 1));
            } else {
                ops.push_back(MCTR_OP_NOOP);
                ks.push_back(0);
                accepts.push_back(0);
                ckpts.push_back(0);
            }
        }

        sc.steps.push_back(make_step(sc.spec, s, seqs, ops, ks, accepts, ckpts, rng));
    }

    return sc;
}

static Scenario make_depth_and_cap_scenario() {
    Scenario sc;
    sc.name = "depth_stack_and_max_len";
    sc.spec = make_spec(4, 24, 64, 8);

    SplitMix64 rng(g_state ^ 0x20202020ULL);

    for (int s = 0; s < 8; ++s) {
        sc.steps.push_back(make_step(
            sc.spec,
            s,
            {0, 1, 2, 3},
            {MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {1000 + s, 1000 + s, 1000 + s, 1000 + s},
            rng));
    }

    sc.steps.push_back(make_step(sc.spec, 8, {0, 1, 2, 3}, {MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT, MCTR_OP_SAVE_CHECKPOINT}, {0, 0, 0, 0}, {0, 0, 0, 0}, {9999, 9999, 9999, 9999}, rng)); // stack-full no-op
    sc.steps.push_back(make_step(sc.spec, 9, {0, 1, 2, 3}, {MCTR_OP_APPEND, MCTR_OP_APPEND, MCTR_OP_APPEND, MCTR_OP_APPEND}, {4, 4, 4, 4}, {0, 0, 0, 0}, {0, 0, 0, 0}, rng));
    sc.steps.push_back(make_step(sc.spec, 10, {0, 1, 2, 3}, {MCTR_OP_ACCEPT_PREFIX, MCTR_OP_ACCEPT_PREFIX, MCTR_OP_ACCEPT_PREFIX, MCTR_OP_ACCEPT_PREFIX}, {4, 4, 4, 4}, {4, 0, 2, 1}, {0, 0, 0, 0}, rng));

    for (int s = 11; s < 30; ++s) {
        std::vector<int32_t> ops;
        std::vector<int32_t> ks;
        std::vector<int32_t> accepts;
        std::vector<int32_t> ckpts;

        for (int r = 0; r < 4; ++r) {
            const int mode = (s + r) % 4;
            if (mode == 0) {
                ops.push_back(MCTR_OP_APPEND);
                ks.push_back(4);
                accepts.push_back(0);
                ckpts.push_back(0);
            } else if (mode == 1) {
                ops.push_back(MCTR_OP_ACCEPT_PREFIX);
                ks.push_back(4);
                accepts.push_back((s + r) % 5);
                ckpts.push_back(0);
            } else if (mode == 2) {
                ops.push_back(MCTR_OP_ROLLBACK_TO);
                ks.push_back(0);
                accepts.push_back(0);
                ckpts.push_back(1000 + ((s + r) % 8));
            } else {
                ops.push_back(MCTR_OP_ROLLBACK_TO);
                ks.push_back(0);
                accepts.push_back(0);
                ckpts.push_back(-12345);
            }
        }

        sc.steps.push_back(make_step(sc.spec, s, {0, 1, 2, 3}, ops, ks, accepts, ckpts, rng));
    }

    return sc;
}

static Scenario make_sparse_active_scenario() {
    Scenario sc;
    sc.name = "sparse_active_mixed_ops";
    sc.spec = make_spec(16, 128, 64, 8);

    SplitMix64 rng(g_state ^ 0x30303030ULL);

    for (int s = 0; s < 48; ++s) {
        std::vector<int32_t> seqs;
        std::vector<int32_t> ops;
        std::vector<int32_t> ks;
        std::vector<int32_t> accepts;
        std::vector<int32_t> ckpts;

        int active = 1 + (s % 5);
        if (s == 17) active = 0;

        for (int r = 0; r < active; ++r) {
            const int seq = (s * 3 + r * 5) % sc.spec.B;
            seqs.push_back(seq);

            const int mode = (s * 7 + r) % 6;
            if (mode == 0) {
                ops.push_back(MCTR_OP_APPEND);
                ks.push_back(1 + ((s + r) % 4));
                accepts.push_back(0);
                ckpts.push_back(0);
            } else if (mode == 1) {
                ops.push_back(MCTR_OP_SAVE_CHECKPOINT);
                ks.push_back(0);
                accepts.push_back(0);
                ckpts.push_back(2000 + seq * 10 + (s % 7));
            } else if (mode == 2) {
                ops.push_back(MCTR_OP_ACCEPT_PREFIX);
                ks.push_back(4);
                accepts.push_back((s + r) % 5);
                ckpts.push_back(0);
            } else if (mode == 3) {
                ops.push_back(MCTR_OP_ROLLBACK_TO);
                ks.push_back(0);
                accepts.push_back(0);
                ckpts.push_back(2000 + seq * 10 + ((s + 3) % 7));
            } else if (mode == 4) {
                ops.push_back(MCTR_OP_ROLLBACK_TO);
                ks.push_back(0);
                accepts.push_back(0);
                ckpts.push_back(-999);
            } else {
                ops.push_back(MCTR_OP_NOOP);
                ks.push_back(0);
                accepts.push_back(0);
                ckpts.push_back(0);
            }
        }

        sc.steps.push_back(make_step(sc.spec, s, seqs, ops, ks, accepts, ckpts, rng));
    }

    return sc;
}

static bool check_input_unchanged(
    const StepHost& step,
    const DeviceBuffer<int32_t>& d_active_seq,
    const DeviceBuffer<int32_t>& d_op_code,
    const DeviceBuffer<int32_t>& d_token_count,
    const DeviceBuffer<int32_t>& d_accept_count,
    const DeviceBuffer<int32_t>& d_checkpoint_id,
    const DeviceBuffer<int32_t>& d_token_values,
    const DeviceBuffer<int32_t>& d_correction_value,
    std::string* error) {
    if (d_active_seq.download() != step.active_seq) {
        if (error) *error = "input active_seq modified";
        return false;
    }

    if (d_op_code.download() != step.op_code) {
        if (error) *error = "input op_code modified";
        return false;
    }

    if (d_token_count.download() != step.token_count) {
        if (error) *error = "input token_count modified";
        return false;
    }

    if (d_accept_count.download() != step.accept_count) {
        if (error) *error = "input accept_count modified";
        return false;
    }

    if (d_checkpoint_id.download() != step.checkpoint_id) {
        if (error) *error = "input checkpoint_id modified";
        return false;
    }

    if (d_token_values.download() != step.token_values) {
        if (error) *error = "input token_values modified";
        return false;
    }

    if (d_correction_value.download() != step.correction_value) {
        if (error) *error = "input correction_value modified";
        return false;
    }

    return true;
}

static bool run_one_step(
    const MctrProblemSpec& spec,
    const StepHost& step,
    MctrOracleState* oracle,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    StepResult* result,
    std::string* error) {
    DeviceBuffer<int32_t> d_active_seq;
    DeviceBuffer<int32_t> d_op_code;
    DeviceBuffer<int32_t> d_token_count;
    DeviceBuffer<int32_t> d_accept_count;
    DeviceBuffer<int32_t> d_checkpoint_id;
    DeviceBuffer<int32_t> d_token_values;
    DeviceBuffer<int32_t> d_correction_value;

    d_active_seq.allocate(step.active_seq.size());
    d_op_code.allocate(step.op_code.size());
    d_token_count.allocate(step.token_count.size());
    d_accept_count.allocate(step.accept_count.size());
    d_checkpoint_id.allocate(step.checkpoint_id.size());
    d_token_values.allocate(step.token_values.size());
    d_correction_value.allocate(step.correction_value.size());

    d_active_seq.upload(step.active_seq);
    d_op_code.upload(step.op_code);
    d_token_count.upload(step.token_count);
    d_accept_count.upload(step.accept_count);
    d_checkpoint_id.upload(step.checkpoint_id);
    d_token_values.upload(step.token_values);
    d_correction_value.upload(step.correction_value);

    GuardedDeviceBuffer<int64_t> d_live_cache_sum;
    GuardedDeviceBuffer<uint64_t> d_live_cache_tail_hash;
    GuardedDeviceBuffer<int32_t> d_length;
    GuardedDeviceBuffer<int32_t> d_num_checkpoints;
    GuardedDeviceBuffer<int32_t> d_top_checkpoint_len;
    GuardedDeviceBuffer<uint64_t> d_state_checksum;

    d_live_cache_sum.allocate((size_t)spec.B);
    d_live_cache_tail_hash.allocate((size_t)spec.B);
    d_length.allocate((size_t)spec.B);
    d_num_checkpoints.allocate((size_t)spec.B);
    d_top_checkpoint_len.allocate((size_t)spec.B);
    d_state_checksum.allocate(1);

    MctrInputs inputs = {};
    inputs.active_seq = d_active_seq.ptr;
    inputs.op_code = d_op_code.ptr;
    inputs.token_count = d_token_count.ptr;
    inputs.accept_count = d_accept_count.ptr;
    inputs.checkpoint_id = d_checkpoint_id.ptr;
    inputs.token_values = d_token_values.ptr;
    inputs.correction_value = d_correction_value.ptr;

    MctrOutputs outputs = {};
    outputs.live_cache_sum = d_live_cache_sum.ptr;
    outputs.live_cache_tail_hash = d_live_cache_tail_hash.ptr;
    outputs.length = d_length.ptr;
    outputs.num_checkpoints = d_num_checkpoints.ptr;
    outputs.top_checkpoint_len = d_top_checkpoint_len.ptr;
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

    if (!check_input_unchanged(
            step,
            d_active_seq,
            d_op_code,
            d_token_count,
            d_accept_count,
            d_checkpoint_id,
            d_token_values,
            d_correction_value,
            error)) {
        return false;
    }

    if (!d_live_cache_sum.check_guards("live_cache_sum", error)) return false;
    if (!d_live_cache_tail_hash.check_guards("live_cache_tail_hash", error)) return false;
    if (!d_length.check_guards("length", error)) return false;
    if (!d_num_checkpoints.check_guards("num_checkpoints", error)) return false;
    if (!d_top_checkpoint_len.check_guards("top_checkpoint_len", error)) return false;
    if (!d_state_checksum.check_guards("state_checksum", error)) return false;

    const std::vector<int64_t> h_live_cache_sum = d_live_cache_sum.download_data();
    const std::vector<uint64_t> h_live_cache_tail_hash = d_live_cache_tail_hash.download_data();
    const std::vector<int32_t> h_length = d_length.download_data();
    const std::vector<int32_t> h_num_checkpoints = d_num_checkpoints.download_data();
    const std::vector<int32_t> h_top_checkpoint_len = d_top_checkpoint_len.download_data();
    const std::vector<uint64_t> h_state_checksum = d_state_checksum.download_data();

    MctrHostInputsView host_inputs = {};
    host_inputs.active_seq = step.active_seq.data();
    host_inputs.op_code = step.op_code.data();
    host_inputs.token_count = step.token_count.data();
    host_inputs.accept_count = step.accept_count.data();
    host_inputs.checkpoint_id = step.checkpoint_id.data();
    host_inputs.token_values = step.token_values.data();
    host_inputs.correction_value = step.correction_value.data();

    MctrExpected expected;
    oracle->step(step.run, host_inputs, &expected);

    MctrHostOutputsView got = {};
    got.live_cache_sum = h_live_cache_sum.data();
    got.live_cache_tail_hash = h_live_cache_tail_hash.data();
    got.length = h_length.data();
    got.num_checkpoints = h_num_checkpoints.data();
    got.top_checkpoint_len = h_top_checkpoint_len.data();
    got.state_checksum = h_state_checksum.data();

    if (!mctr_check_all_outputs(spec, expected, got, error)) {
        return false;
    }

    if (result) {
        result->live_cache_sum = h_live_cache_sum;
        result->live_cache_tail_hash = h_live_cache_tail_hash;
        result->length = h_length;
        result->num_checkpoints = h_num_checkpoints;
        result->top_checkpoint_len = h_top_checkpoint_len;
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
    std::string* first_error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec); if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/
    if (workspace_bytes == 0) {
        if (first_error) *first_error = "solution_workspace_bytes returned 0";
        return false;
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(workspace_bytes);

    MctrOracleState oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) {
        results->clear();
        results->reserve(sc.steps.size());
    }

    bool all_ok = true;

    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult result;
        std::string error;

        const bool ok = run_one_step(
            sc.spec,
            sc.steps[i],
            &oracle,
            state,
            workspace.ptr,
            workspace_bytes,
            stream,
            results ? &result : nullptr,
            &error);

        ++(*total_steps);
        if (ok) {
            ++(*passed_steps);
        } else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream oss;
                oss << sc.name << " step " << i << ": " << error;
                *first_error = oss.str();
            }
        }

        if (results) {
            results->push_back(result);
        }

        if (verbose) {
            std::printf(
                "scenario %-32s step %02zu/%02zu active=%d %s%s%s\n",
                sc.name.c_str(),
                i,
                sc.steps.size(),
                sc.steps[i].run.active_count,
                ok ? "PASS" : "FAIL",
                ok ? "" : "  ",
                ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));

    return all_ok;
}

static bool compare_results(
    const std::vector<StepResult>& a,
    const std::vector<StepResult>& b,
    std::string* error) {
    if (a.size() != b.size()) {
        if (error) *error = "step count mismatch";
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].live_cache_sum != b[i].live_cache_sum ||
            a[i].live_cache_tail_hash != b[i].live_cache_tail_hash ||
            a[i].length != b[i].length ||
            a[i].num_checkpoints != b[i].num_checkpoints ||
            a[i].top_checkpoint_len != b[i].top_checkpoint_len ||
            a[i].state_checksum != b[i].state_checksum) {
            if (error) {
                std::ostringstream oss;
                oss << "permuted replay mismatch at step " << i
                    << ": checksum base=0x" << std::hex << a[i].state_checksum
                    << ", permuted=0x" << b[i].state_checksum;
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        std::vector<Scenario> scenarios;
        scenarios.push_back(make_nested_checkpoint_scenario());
        scenarios.push_back(make_depth_and_cap_scenario());
        scenarios.push_back(make_sparse_active_scenario());

        int passed = 0;
        int total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results;
            std::vector<StepResult> permuted_results;

            std::string error;

            const bool ok_base = run_scenario_once(
                sc,
                true,
                &base_results,
                &passed,
                &total,
                &error);

            Scenario permuted = make_permuted_scenario(sc);

            const bool ok_permuted = run_scenario_once(
                permuted,
                true,
                &permuted_results,
                &passed,
                &total,
                &error);

            if (ok_base && ok_permuted) {
                std::string compare_error;
                if (compare_results(base_results, permuted_results, &compare_error)) {
                    std::printf(
                        "scenario %-32s permuted-order replay PASS\n",
                        sc.name.c_str());
                } else {
                    all_ok = false;
                    std::printf(
                        "scenario %-32s permuted-order replay FAIL  %s\n",
                        sc.name.c_str(),
                        compare_error.c_str());
                }
            } else {
                all_ok = false;
                std::printf(
                    "scenario %-32s FAIL  %s\n",
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
