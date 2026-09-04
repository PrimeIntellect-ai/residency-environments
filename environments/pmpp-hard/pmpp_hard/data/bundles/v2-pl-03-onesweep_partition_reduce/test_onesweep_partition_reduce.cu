// ============================================================================
// file: test_onesweep_partition_reduce.cu
// ============================================================================

#include "onesweep_partition_reduce_common.h"
#include "onesweep_partition_reduce_oracle.hpp"

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

static constexpr uint64_t g_state = 0x13198a2e03707344ULL;
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

    int uniform_int(int lo, int hi) {
        const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u64() % span);
    }

    double uniform01() {
        return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
    }

    bool chance(double p) {
        return uniform01() < p;
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
                    oss << "left guard corrupted for " << name << " at byte " << i
                        << ": got 0x" << std::hex << static_cast<int>(before[i]);
                    *error = oss.str();
                }
                return false;
            }
        }

        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (after[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "right guard corrupted for " << name << " at byte " << i
                        << ": got 0x" << std::hex << static_cast<int>(after[i]);
                    *error = oss.str();
                }
                return false;
            }
        }

        return true;
    }
};

struct HostCase {
    std::string name;
    OprRunSpec run;
    std::vector<uint32_t> key;
    std::vector<int32_t> val;
};

static int sample_zipf_1_2(SplitMix64& rng, int num_digits) {
    double total = 0.0;
    for (int i = 0; i < num_digits; ++i) {
        total += 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
    }

    const double target = rng.uniform01() * total;
    double accum = 0.0;
    for (int i = 0; i < num_digits; ++i) {
        accum += 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
        if (accum >= target) {
            return i;
        }
    }

    return num_digits - 1;
}

static uint32_t make_key_for_digit(int digit, int radix_bits, SplitMix64& rng) {
    const int low_bits = 32 - radix_bits;
    const uint32_t low_mask = static_cast<uint32_t>((1ULL << low_bits) - 1ULL);
    const uint32_t high = static_cast<uint32_t>(digit) << low_bits;
    const uint32_t low = static_cast<uint32_t>(rng.next_u64()) & low_mask;
    return high | low;
}

static int choose_digit(
    SplitMix64& rng,
    int i,
    int N,
    int radix_bits,
    int distribution_id,
    bool avoid_last_digit) {
    const int num_digits = opr_num_digits(radix_bits);
    const int limit_digits = avoid_last_digit && num_digits > 1 ? num_digits - 1 : num_digits;

    if (limit_digits <= 1) {
        return 0;
    }

    switch (distribution_id) {
        case OPR_DIST_UNIFORM:
            return rng.uniform_int(0, limit_digits - 1);

        case OPR_DIST_SINGLE_HOT:
            if (rng.chance(0.92)) {
                return 0;
            }
            return rng.uniform_int(1, limit_digits - 1);

        case OPR_DIST_FEW_HOT: {
            const int hot_count = std::min(limit_digits, limit_digits >= 8 ? 4 : 2);
            if (rng.chance(0.90)) {
                return rng.uniform_int(0, hot_count - 1);
            }
            if (hot_count == limit_digits) {
                return rng.uniform_int(0, limit_digits - 1);
            }
            return rng.uniform_int(hot_count, limit_digits - 1);
        }

        case OPR_DIST_ZIPF_1_2:
            return sample_zipf_1_2(rng, limit_digits);

        case OPR_DIST_NEARLY_SORTED: {
            int d = static_cast<int>((static_cast<int64_t>(i) * limit_digits) / N);
            if (d >= limit_digits) d = limit_digits - 1;
            if (rng.chance(0.08)) {
                const int jitter = rng.uniform_int(-2, 2);
                d = std::max(0, std::min(limit_digits - 1, d + jitter));
            }
            if (rng.chance(0.02)) {
                d = rng.uniform_int(0, limit_digits - 1);
            }
            return d;
        }

        case OPR_DIST_REVERSE: {
            int d = limit_digits - 1 - static_cast<int>((static_cast<int64_t>(i) * limit_digits) / N);
            if (d < 0) d = 0;
            if (rng.chance(0.08)) {
                const int jitter = rng.uniform_int(-2, 2);
                d = std::max(0, std::min(limit_digits - 1, d + jitter));
            }
            if (rng.chance(0.02)) {
                d = rng.uniform_int(0, limit_digits - 1);
            }
            return d;
        }

        default:
            return rng.uniform_int(0, limit_digits - 1);
    }
}

static HostCase make_case(
    const char* name,
    int N,
    int radix_bits,
    int distribution_id,
    uint64_t case_seed,
    bool force_one_bucket,
    int forced_bucket,
    bool force_empty_last_digit,
    bool force_tie_max) {
    HostCase hc;
    hc.name = name;

    hc.run = {};
    hc.run.abi_version = OPR_ABI_VERSION;
    hc.run.N = N;
    hc.run.radix_bits = radix_bits;
    hc.run.seed_id = static_cast<int32_t>(case_seed & 0x7fffffff);
    hc.run.distribution_id = distribution_id;
    hc.run.case_id = static_cast<int32_t>((case_seed >> 32) & 0x7fffffff);

    if (!opr_validate_run_spec(&hc.run)) {
        throw std::runtime_error("invalid generated run spec");
    }

    SplitMix64 rng(g_state ^ case_seed);

    hc.key.resize(static_cast<size_t>(N));
    hc.val.resize(static_cast<size_t>(N));

    for (int i = 0; i < N; ++i) {
        int d = forced_bucket;
        if (!force_one_bucket) {
            d = choose_digit(rng, i, N, radix_bits, distribution_id, force_empty_last_digit);
        }

        hc.key[static_cast<size_t>(i)] = make_key_for_digit(d, radix_bits, rng);

        int v = rng.uniform_int(-100000, 100000);
        if (v == 0 && rng.chance(0.80)) {
            v = rng.chance(0.5) ? 1 : -1;
        }
        hc.val[static_cast<size_t>(i)] = v;
    }

    if (force_tie_max && N >= 16) {
        const int num_digits = opr_num_digits(radix_bits);
        const int tie_bucket = std::min(3, num_digits - 1);

        const int src0 = 5;
        const int src1 = 11;

        hc.key[static_cast<size_t>(src0)] = make_key_for_digit(tie_bucket, radix_bits, rng);
        hc.key[static_cast<size_t>(src1)] = make_key_for_digit(tie_bucket, radix_bits, rng);

        hc.val[static_cast<size_t>(src0)] = 2147483000;
        hc.val[static_cast<size_t>(src1)] = 2147483000;

        for (int i = 0; i < N; ++i) {
            if (i == src0 || i == src1) continue;
            if (opr_oracle_digit(hc.key[static_cast<size_t>(i)], radix_bits) == tie_bucket &&
                hc.val[static_cast<size_t>(i)] >= 2147483000) {
                hc.val[static_cast<size_t>(i)] = 17;
            }
        }
    }

    return hc;
}

static std::vector<HostCase> build_test_cases() {
    std::vector<HostCase> cases;
    uint64_t s = 0x100000001b3ULL;

    cases.push_back(make_case(
        "uniform_N4096_r4",
        4096,
        4,
        OPR_DIST_UNIFORM,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "uniform_N65536_r8",
        65536,
        8,
        OPR_DIST_UNIFORM,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "single_hot_N32768_r4",
        32768,
        4,
        OPR_DIST_SINGLE_HOT,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "single_hot_N131072_r8",
        131072,
        8,
        OPR_DIST_SINGLE_HOT,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "few_hot_N65536_r8",
        65536,
        8,
        OPR_DIST_FEW_HOT,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "zipf_N8192_r4",
        8192,
        4,
        OPR_DIST_ZIPF_1_2,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "zipf_N131072_r8",
        131072,
        8,
        OPR_DIST_ZIPF_1_2,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "nearly_sorted_N16384_r8",
        16384,
        8,
        OPR_DIST_NEARLY_SORTED,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "reverse_N32768_r4",
        32768,
        4,
        OPR_DIST_REVERSE,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "edge_empty_last_bucket_N4096_r8",
        4096,
        8,
        OPR_DIST_UNIFORM,
        s++,
        false,
        0,
        true,
        false));

    cases.push_back(make_case(
        "edge_all_one_bucket_N8192_r4",
        8192,
        4,
        OPR_DIST_SINGLE_HOT,
        s++,
        true,
        0,
        false,
        false));

    cases.push_back(make_case(
        "edge_N_just_under_tile_multiple_r8",
        OPR_TILE_ITEMS * 17 - 1,
        8,
        OPR_DIST_FEW_HOT,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "edge_N_just_over_tile_multiple_r4",
        OPR_TILE_ITEMS * 17 + 1,
        4,
        OPR_DIST_UNIFORM,
        s++,
        false,
        0,
        false,
        false));

    cases.push_back(make_case(
        "edge_tie_bucket_max_r8",
        4096,
        8,
        OPR_DIST_UNIFORM,
        s++,
        false,
        0,
        false,
        true));

    return cases;
}

static bool check_input_unchanged(
    const HostCase& hc,
    const DeviceBuffer<uint32_t>& d_key,
    const DeviceBuffer<int32_t>& d_val,
    std::string* error) {
    const std::vector<uint32_t> key_back = d_key.download();
    const std::vector<int32_t> val_back = d_val.download();

    if (key_back != hc.key) {
        if (error) *error = "input key modified";
        return false;
    }

    if (val_back != hc.val) {
        if (error) *error = "input val modified";
        return false;
    }

    return true;
}

static bool run_one_case(
    const HostCase& hc,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    std::string* error) {
    const int N = hc.run.N;
    const int radix_bits = hc.run.radix_bits;
    const int num_digits = opr_num_digits(radix_bits);
    const int num_tiles = opr_num_tiles(N);

    DeviceBuffer<uint32_t> d_key;
    DeviceBuffer<int32_t> d_val;

    d_key.allocate(hc.key.size());
    d_val.allocate(hc.val.size());

    d_key.upload(hc.key);
    d_val.upload(hc.val);

    GuardedDeviceBuffer<int32_t> d_tile_digit_offsets;
    GuardedDeviceBuffer<int32_t> d_digit_counts;
    GuardedDeviceBuffer<int32_t> d_digit_offsets;
    GuardedDeviceBuffer<uint32_t> d_packed_key;
    GuardedDeviceBuffer<int32_t> d_packed_val;
    GuardedDeviceBuffer<int32_t> d_packed_src;
    GuardedDeviceBuffer<int64_t> d_bucket_sum;
    GuardedDeviceBuffer<int64_t> d_bucket_max;
    GuardedDeviceBuffer<int32_t> d_bucket_argmax;

    d_tile_digit_offsets.allocate(static_cast<size_t>(num_tiles) * static_cast<size_t>(num_digits));
    d_digit_counts.allocate(static_cast<size_t>(num_digits));
    d_digit_offsets.allocate(static_cast<size_t>(num_digits + 1));
    d_packed_key.allocate(static_cast<size_t>(N));
    d_packed_val.allocate(static_cast<size_t>(N));
    d_packed_src.allocate(static_cast<size_t>(N));
    d_bucket_sum.allocate(static_cast<size_t>(num_digits));
    d_bucket_max.allocate(static_cast<size_t>(num_digits));
    d_bucket_argmax.allocate(static_cast<size_t>(num_digits));

    OprInputs inputs = {};
    inputs.key = d_key.ptr;
    inputs.val = d_val.ptr;

    OprOutputs outputs = {};
    outputs.tile_digit_offsets = d_tile_digit_offsets.ptr;
    outputs.digit_counts = d_digit_counts.ptr;
    outputs.digit_offsets = d_digit_offsets.ptr;
    outputs.packed_key = d_packed_key.ptr;
    outputs.packed_val = d_packed_val.ptr;
    outputs.packed_src = d_packed_src.ptr;
    outputs.bucket_sum = d_bucket_sum.ptr;
    outputs.bucket_max = d_bucket_max.ptr;
    outputs.bucket_argmax = d_bucket_argmax.ptr;

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(solution_run(
        state,
        &hc.run,
        &inputs,
        &outputs,
        workspace,
        workspace_bytes,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(hc, d_key, d_val, error)) {
        return false;
    }

    if (!d_tile_digit_offsets.check_guards("tile_digit_offsets", error)) return false;
    if (!d_digit_counts.check_guards("digit_counts", error)) return false;
    if (!d_digit_offsets.check_guards("digit_offsets", error)) return false;
    if (!d_packed_key.check_guards("packed_key", error)) return false;
    if (!d_packed_val.check_guards("packed_val", error)) return false;
    if (!d_packed_src.check_guards("packed_src", error)) return false;
    if (!d_bucket_sum.check_guards("bucket_sum", error)) return false;
    if (!d_bucket_max.check_guards("bucket_max", error)) return false;
    if (!d_bucket_argmax.check_guards("bucket_argmax", error)) return false;

    const std::vector<int32_t> h_tile_digit_offsets = d_tile_digit_offsets.download_data();
    const std::vector<int32_t> h_digit_counts = d_digit_counts.download_data();
    const std::vector<int32_t> h_digit_offsets = d_digit_offsets.download_data();
    const std::vector<uint32_t> h_packed_key = d_packed_key.download_data();
    const std::vector<int32_t> h_packed_val = d_packed_val.download_data();
    const std::vector<int32_t> h_packed_src = d_packed_src.download_data();
    const std::vector<int64_t> h_bucket_sum = d_bucket_sum.download_data();
    const std::vector<int64_t> h_bucket_max = d_bucket_max.download_data();
    const std::vector<int32_t> h_bucket_argmax = d_bucket_argmax.download_data();

    OprHostInputsView host_inputs = {};
    host_inputs.key = hc.key.data();
    host_inputs.val = hc.val.data();

    OprExpected expected;
    opr_cpu_oracle(hc.run, host_inputs, &expected);

    OprHostOutputsView got = {};
    got.tile_digit_offsets = h_tile_digit_offsets.data();
    got.digit_counts = h_digit_counts.data();
    got.digit_offsets = h_digit_offsets.data();
    got.packed_key = h_packed_key.data();
    got.packed_val = h_packed_val.data();
    got.packed_src = h_packed_src.data();
    got.bucket_sum = h_bucket_sum.data();
    got.bucket_max = h_bucket_max.data();
    got.bucket_argmax = h_bucket_argmax.data();

    if (!opr_check_metadata(hc.run, expected, got, error)) {
        return false;
    }

    if (!opr_check_packed(hc.run, expected, got, error)) {
        return false;
    }

    if (!opr_check_reductions(hc.run, expected, got, error)) {
        return false;
    }

    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        OprProblemSpec spec = {};
        spec.abi_version = OPR_ABI_VERSION;
        spec.max_N = OPR_MAX_N;
        spec.max_radix_bits = OPR_MAX_RADIX_BITS;
        spec.flags = 0;

        size_t workspace_bytes = solution_workspace_bytes(&spec); if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/
        if (workspace_bytes == 0) {
            throw std::runtime_error("solution_workspace_bytes returned 0");
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(workspace_bytes);

        const std::vector<HostCase> cases = build_test_cases();

        int passed = 0;
        const int total = static_cast<int>(cases.size());

        for (const HostCase& hc : cases) {
            std::string error;
            bool ok = false;

            try {
                ok = run_one_case(
                    hc,
                    state,
                    workspace.ptr,
                    workspace_bytes,
                    stream,
                    &error);
            } catch (const std::exception& ex) {
                error = ex.what();
                ok = false;
            }

            if (ok) {
                ++passed;
                std::printf(
                    "case %-48s PASS  N=%d r=%d digits=%d dist=%d\n",
                    hc.name.c_str(),
                    hc.run.N,
                    hc.run.radix_bits,
                    opr_num_digits(hc.run.radix_bits),
                    hc.run.distribution_id);
            } else {
                std::printf(
                    "case %-48s FAIL  %s\n",
                    hc.name.c_str(),
                    error.c_str());
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));

        std::printf("passed %d / %d\n", passed, total);
        return passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
