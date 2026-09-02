// ============================================================================
// file: test_mla_latent_absorb_decode.cu
// ============================================================================

#include "mla_latent_absorb_decode_common.h"
#include "mla_latent_absorb_decode_oracle.hpp"

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

static constexpr uint64_t g_state = 0x51c3a9e7b8d2f045ULL;
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

    void free_now() {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
            count = 0;
        }
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
    MlaRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> new_token_count;
    std::vector<float> new_c;
    std::vector<float> new_r;
    std::vector<float> q;
    std::vector<float> q_rope;
};

struct Weights {
    std::vector<float> W_uk;
    std::vector<float> W_uv;
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
};

struct Scenario {
    std::string name;
    MlaProblemSpec spec;
    Weights weights;
    std::vector<StepHost> steps;
};

struct StepResult {
    std::vector<int32_t> active_seq;
    std::vector<int32_t> new_token_count;
    std::vector<float> y;
    std::vector<float> lse;
    std::vector<int32_t> seq_len;
    std::vector<uint64_t> cache_hash;
    uint64_t meta_checksum = 0;
    int32_t sat_count = 0;
    int32_t total_tokens = 0;
};

static MlaProblemSpec make_spec(
    int B, int Hq, int d_c, int d_r, int d_h, int d_v, int max_seq_len) {
    MlaProblemSpec spec = {};
    spec.abi_version = MLA_ABI_VERSION;
    spec.B = B;
    spec.Hq = Hq;
    spec.d_c = d_c;
    spec.d_r = d_r;
    spec.d_h = d_h;
    spec.d_v = d_v;
    spec.max_seq_len = max_seq_len;
    spec.flags = 0;

    if (!mla_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid MlaProblemSpec generated");
    }
    return spec;
}

static Weights make_weights(const MlaProblemSpec& spec, uint64_t seed) {
    Weights w;
    SplitMix64 rng(seed);

    w.W_uk.resize((size_t)spec.Hq * spec.d_h * spec.d_c);
    for (float& v : w.W_uk) v = rng.uniform_float(-0.25f, 0.25f);

    w.W_uv.resize((size_t)spec.Hq * spec.d_v * spec.d_c);
    for (float& v : w.W_uv) v = rng.uniform_float(-0.25f, 0.25f);

    const int half_r = spec.d_r / 2;
    w.rope_cos.resize((size_t)spec.max_seq_len * half_r);
    w.rope_sin.resize((size_t)spec.max_seq_len * half_r);
    for (int t = 0; t < spec.max_seq_len; ++t) {
        for (int j = 0; j < half_r; ++j) {
            const double freq =
                std::pow(10000.0, -2.0 * j / static_cast<double>(spec.d_r));
            const double ang = t * freq;
            w.rope_cos[(size_t)t * half_r + j] = static_cast<float>(std::cos(ang));
            w.rope_sin[(size_t)t * half_r + j] = static_cast<float>(std::sin(ang));
        }
    }
    return w;
}

static void fill_step_tensors(
    const MlaProblemSpec& spec,
    StepHost* step,
    uint64_t seed) {
    const int A = step->run.active_count;
    const int Hq = spec.Hq;
    const int d_c = spec.d_c;
    const int d_r = spec.d_r;
    const int d_h = spec.d_h;

    const size_t a_count = A > 0 ? static_cast<size_t>(A) : 1;

    step->new_token_count.resize(a_count, 1);
    step->new_c.resize(a_count * MLA_MAX_NEW_TOKENS * (size_t)d_c, 0.0f);
    step->new_r.resize(a_count * MLA_MAX_NEW_TOKENS * (size_t)d_r, 0.0f);
    step->q.resize(a_count * MLA_MAX_NEW_TOKENS * (size_t)Hq * (size_t)d_h, 0.0f);
    step->q_rope.resize(a_count * MLA_MAX_NEW_TOKENS * (size_t)Hq * (size_t)d_r, 0.0f);

    for (int a = 0; a < A; ++a) {
        const int seq = step->active_seq[(size_t)a];
        SplitMix64 rng(seed ^ (0x8e3779b97f4a7c15ULL * (uint64_t)(seq + 1)));

        for (int nt = 0; nt < MLA_MAX_NEW_TOKENS; ++nt) {
            const bool pad = nt >= step->new_token_count[(size_t)a];
            for (int d = 0; d < d_c; ++d) {
                const size_t idx =
                    ((size_t)a * MLA_MAX_NEW_TOKENS + nt) * (size_t)d_c + d;
                step->new_c[idx] = pad ? rng.uniform_float(-3.0e5f, 3.0e5f)
                                       : rng.uniform_float(-3.0f, 3.0f);
            }
            for (int d = 0; d < d_r; ++d) {
                const size_t idx =
                    ((size_t)a * MLA_MAX_NEW_TOKENS + nt) * (size_t)d_r + d;
                step->new_r[idx] = pad ? rng.uniform_float(-3.0e5f, 3.0e5f)
                                       : rng.uniform_float(-3.0f, 3.0f);
            }
            for (int h = 0; h < Hq; ++h) {
                for (int d = 0; d < d_h; ++d) {
                    const size_t idx =
                        (((size_t)a * MLA_MAX_NEW_TOKENS + nt) * Hq + h) *
                            (size_t)d_h + d;
                    step->q[idx] = pad ? rng.uniform_float(-3.0e5f, 3.0e5f)
                                       : rng.uniform_float(-1.0f, 1.0f);
                }
                for (int d = 0; d < d_r; ++d) {
                    const size_t idx =
                        (((size_t)a * MLA_MAX_NEW_TOKENS + nt) * Hq + h) *
                            (size_t)d_r + d;
                    step->q_rope[idx] = pad ? rng.uniform_float(-3.0e5f, 3.0e5f)
                                            : rng.uniform_float(-1.0f, 1.0f);
                }
            }
        }
    }
}

static StepHost make_step(
    const MlaProblemSpec& spec,
    int step_id,
    const std::vector<int32_t>& active,
    const std::vector<int32_t>& ntokens,
    uint64_t seed) {
    if (active.size() != ntokens.size()) {
        throw std::runtime_error("active/ntokens size mismatch");
    }
    for (size_t i = 0; i < active.size(); ++i) {
        if (active[i] < 0 || active[i] >= spec.B) {
            throw std::runtime_error("active id out of range");
        }
        if (ntokens[i] < 1 || ntokens[i] > MLA_MAX_NEW_TOKENS) {
            throw std::runtime_error("new_token_count out of range");
        }
        for (size_t j = i + 1; j < active.size(); ++j) {
            if (active[i] == active[j]) {
                throw std::runtime_error("duplicate active id in step");
            }
        }
    }

    StepHost step;
    step.run = {};
    step.run.abi_version = MLA_ABI_VERSION;
    step.run.active_count = static_cast<int32_t>(active.size());
    step.run.step_id = step_id;

    if (!mla_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid MlaRunSpec generated");
    }

    step.active_seq = active;
    if (step.active_seq.empty()) {
        step.active_seq.push_back(0);
    }

    // Overwrite the (dummy-safe) counts BEFORE fill so pad detection works.
    step.new_token_count.assign(std::max<size_t>(1, ntokens.size()), 1);
    for (size_t i = 0; i < ntokens.size(); ++i) {
        step.new_token_count[i] = ntokens[i];
    }
    {
        // fill_step_tensors reads new_token_count for pad poisoning.
        std::vector<int32_t> saved = step.new_token_count;
        fill_step_tensors(spec, &step, seed);
        step.new_token_count = saved;
    }

    return step;
}

// ---------------------------------------------------------------------------
// Adversarial quantization vectors. Element 0 anchors the group amax so the
// scale exponent is known; the rest probe exact integer ties (round half to
// even), the 127.5 -> 128 saturation tie, zeros, and tiny magnitudes.
// ---------------------------------------------------------------------------
static void craft_quant_group(int variant, float* x) {
    for (int d = 0; d < MLA_QUANT_GROUP; ++d) x[d] = 0.0f;

    switch (variant % 5) {
        case 0:
            // amax 1.9921875 = 127.5/64: kfloor=0, se=-6, z = x*64.
            x[0] = 1.9921875f;      // z=127.5: tie -> 128 -> clamp 127, SAT
            x[1] = -1.9921875f;     // z=-127.5 -> -128 -> clamp -127, SAT
            x[2] = 1.984375f;       // z=127 exact
            x[3] = 0.0390625f;      // z=2.5: tie -> 2 (even)
            x[4] = 0.0546875f;      // z=3.5: tie -> 4 (even)
            x[5] = -0.0390625f;     // z=-2.5 -> -2
            x[6] = 0.0078125f;      // z=0.5: tie -> 0
            x[7] = 0.0234375f;      // z=1.5: tie -> 2
            x[8] = 1.0f;            // z=64 exact
            break;
        case 1:
            // amax exactly a power of two: 2.0 -> kfloor=1, se=-5, z = x*32.
            x[0] = 2.0f;            // z=64 exact
            x[1] = -2.0f;
            x[2] = 1.9999999f;      // z just under 64
            x[3] = 0.5f;            // z=16 exact
            x[4] = 0.046875f;       // z=1.5: tie -> 2
            x[5] = 0.015625f;       // z=0.5: tie -> 0
            break;
        case 2:
            // tiny anchor: 2^-40 -> kfloor=-40, se=-46, z = x*2^46.
            x[0] = 9.094947017729282e-13f;   // 2^-40, z=64
            x[1] = 4.547473508864641e-13f;   // 2^-41, z=32
            x[2] = -6.821210263296962e-13f;  // -1.5*2^-41, z=-48
            break;
        case 3:
            // near-max magnitudes: amax 4.0 (input bound), se=-4, z = x*16.
            x[0] = 4.0f;            // z=64 exact
            x[1] = 3.96875f;        // z=63.5: tie -> 64
            x[2] = -3.90625f;       // z=-62.5: tie -> -62
            x[3] = 0.15625f;        // z=2.5: tie -> 2
            break;
        default:
            // all zeros: amax == 0 path (scale 0, all bytes 0).
            break;
    }
}

static void inject_adversarial_quant(
    const MlaProblemSpec& spec,
    StepHost* step,
    int step_idx) {
    const int A = step->run.active_count;
    const int gc = spec.d_c / MLA_QUANT_GROUP;
    const int gr = spec.d_r / MLA_QUANT_GROUP;

    for (int a = 0; a < A; ++a) {
        if ((a + step_idx) % 2 != 0) continue;
        const int cnt = step->new_token_count[(size_t)a];
        const int nt = step_idx % cnt;

        float* cvec = step->new_c.data() +
            ((size_t)a * MLA_MAX_NEW_TOKENS + nt) * spec.d_c;
        float* rvec = step->new_r.data() +
            ((size_t)a * MLA_MAX_NEW_TOKENS + nt) * spec.d_r;

        for (int g = 0; g < gc; ++g) {
            craft_quant_group(step_idx + a + g, cvec + g * MLA_QUANT_GROUP);
        }
        for (int g = 0; g < gr; ++g) {
            craft_quant_group(step_idx + a + g + 1, rvec + g * MLA_QUANT_GROUP);
        }
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

static Scenario make_boundary_min_scenario() {
    Scenario sc;
    sc.name = "boundary_min_B2_H2";
    sc.spec = make_spec(2, 2, 128, 32, 64, 64, 64);
    sc.weights = make_weights(sc.spec, g_state ^ 0x1001);

    for (int s = 0; s < 14; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;
        if (s == 5) {
            // empty step
        } else if (s % 3 == 2) {
            active.push_back(s % 2);
            nt.push_back(1 + (s % 4));
        } else {
            active.push_back(0);
            nt.push_back(s < 6 ? 4 : 1);
            active.push_back(1);
            nt.push_back(s < 6 ? 3 : 2);
        }
        sc.steps.push_back(make_step(
            sc.spec, s, active, nt, g_state ^ 0x100000ULL ^ (uint64_t)s));
    }
    return sc;
}

static Scenario make_quant_adversarial_scenario() {
    Scenario sc;
    sc.name = "quant_adversarial_ties_saturation";
    sc.spec = make_spec(2, 4, 128, 32, 64, 64, 128);
    sc.weights = make_weights(sc.spec, g_state ^ 0x2002);

    for (int s = 0; s < 16; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;
        active.push_back(0);
        nt.push_back(1 + (s % 8));
        if (s % 2 == 0) {
            active.push_back(1);
            nt.push_back(1 + ((s / 2) % 4));
        }
        StepHost step = make_step(
            sc.spec, s, active, nt, g_state ^ 0x200000ULL ^ (uint64_t)s);
        inject_adversarial_quant(sc.spec, &step, s);
        sc.steps.push_back(step);
    }
    return sc;
}

static Scenario make_heads_max_scenario() {
    Scenario sc;
    sc.name = "heads32_asym_dv";
    sc.spec = make_spec(1, 32, 128, 64, 64, 128, 96);
    sc.weights = make_weights(sc.spec, g_state ^ 0x3003);

    for (int s = 0; s < 12; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;
        if (s != 7) {
            active.push_back(0);
            nt.push_back(s < 8 ? 8 : 1 + (s % 3));
        }
        sc.steps.push_back(make_step(
            sc.spec, s, active, nt, g_state ^ 0x300000ULL ^ (uint64_t)s));
    }
    return sc;
}

static Scenario make_rope_deep_scenario() {
    Scenario sc;
    sc.name = "rope_deep_prefill";
    sc.spec = make_spec(2, 8, 128, 64, 128, 64, 640);
    sc.weights = make_weights(sc.spec, g_state ^ 0x4004);

    for (int s = 0; s < 46; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;
        if (s < 40) {
            active.push_back(0);
            nt.push_back(8);
            active.push_back(1);
            nt.push_back(8);
        } else {
            active.push_back(0);
            nt.push_back(1);
            if (s % 2 == 0) {
                active.push_back(1);
                nt.push_back(2);
            }
        }
        sc.steps.push_back(make_step(
            sc.spec, s, active, nt, g_state ^ 0x400000ULL ^ (uint64_t)s));
    }
    return sc;
}

static Scenario make_singleton_scenario() {
    Scenario sc;
    sc.name = "singleton_dc192_empty_steps";
    sc.spec = make_spec(1, 4, 192, 32, 64, 64, 128);
    sc.weights = make_weights(sc.spec, g_state ^ 0x5005);

    for (int s = 0; s < 15; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;
        if (s % 4 == 3) {
            // empty step
        } else {
            active.push_back(0);
            nt.push_back(1 + ((s * 3) % 8));
        }
        sc.steps.push_back(make_step(
            sc.spec, s, active, nt, g_state ^ 0x500000ULL ^ (uint64_t)s));
    }
    return sc;
}

static Scenario make_asym_all_scenario() {
    Scenario sc;
    sc.name = "asym_dc256_dv64";
    sc.spec = make_spec(3, 8, 256, 32, 128, 64, 160);
    sc.weights = make_weights(sc.spec, g_state ^ 0x6006);

    for (int s = 0; s < 14; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;
        for (int b = 0; b < 3; ++b) {
            if ((s + b) % 3 == 0 && s % 5 != 4) {
                active.push_back(b);
                nt.push_back(1 + ((s + 2 * b) % 8));
            }
        }
        sc.steps.push_back(make_step(
            sc.spec, s, active, nt, g_state ^ 0x600000ULL ^ (uint64_t)s));
    }
    return sc;
}

static Scenario make_ragged_scenario() {
    Scenario sc;
    sc.name = "ragged_rates_B8";
    sc.spec = make_spec(8, 4, 128, 32, 64, 128, 192);
    sc.weights = make_weights(sc.spec, g_state ^ 0x7007);

    for (int s = 0; s < 20; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;
        for (int b = 0; b < 8; ++b) {
            const bool fast = b < 3;
            const bool go = fast ? true : ((s + b) % 4 == 0);
            if (go && !(s == 10 && b == 0)) {
                active.push_back(b);
                nt.push_back(fast ? 1 + ((s + b) % 8) : 1 + (b % 3));
            }
        }
        sc.steps.push_back(make_step(
            sc.spec, s, active, nt, g_state ^ 0x700000ULL ^ (uint64_t)s));
    }
    return sc;
}

static Scenario make_bench_like_scenario() {
    Scenario sc;
    sc.name = "bench_like_small";
    sc.spec = make_spec(8, 16, 256, 64, 128, 128, 320);
    sc.weights = make_weights(sc.spec, g_state ^ 0x8008);

    for (int s = 0; s < 28; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;
        if (s < 16) {
            for (int b = 0; b < 8; ++b) {
                active.push_back(b);
                nt.push_back(8);
            }
        } else {
            for (int b = 0; b < 8; ++b) {
                if (s < 22 && b % 2 == 0) {
                    active.push_back(b);
                    nt.push_back(4);
                } else {
                    active.push_back(b);
                    nt.push_back(1);
                }
            }
        }
        sc.steps.push_back(make_step(
            sc.spec, s, active, nt, g_state ^ 0x800000ULL ^ (uint64_t)s));
    }
    return sc;
}

// ---------------------------------------------------------------------------

static StepHost permute_step_rows(const MlaProblemSpec& spec, const StepHost& src) {
    StepHost dst = src;
    const int A = src.run.active_count;
    const int Hq = spec.Hq;
    const int d_c = spec.d_c;
    const int d_r = spec.d_r;
    const int d_h = spec.d_h;

    if (A == 0) {
        return dst;
    }

    dst.active_seq.resize((size_t)A);
    dst.new_token_count.assign((size_t)A, 1);
    dst.new_c.assign((size_t)A * MLA_MAX_NEW_TOKENS * (size_t)d_c, 0.0f);
    dst.new_r.assign((size_t)A * MLA_MAX_NEW_TOKENS * (size_t)d_r, 0.0f);
    dst.q.assign((size_t)A * MLA_MAX_NEW_TOKENS * (size_t)Hq * (size_t)d_h, 0.0f);
    dst.q_rope.assign((size_t)A * MLA_MAX_NEW_TOKENS * (size_t)Hq * (size_t)d_r, 0.0f);

    for (int new_a = 0; new_a < A; ++new_a) {
        const int old_a = A - 1 - new_a;
        dst.active_seq[(size_t)new_a] = src.active_seq[(size_t)old_a];
        dst.new_token_count[(size_t)new_a] = src.new_token_count[(size_t)old_a];

        const size_t c_row = (size_t)MLA_MAX_NEW_TOKENS * d_c;
        std::memcpy(
            dst.new_c.data() + (size_t)new_a * c_row,
            src.new_c.data() + (size_t)old_a * c_row,
            c_row * sizeof(float));

        const size_t r_row = (size_t)MLA_MAX_NEW_TOKENS * d_r;
        std::memcpy(
            dst.new_r.data() + (size_t)new_a * r_row,
            src.new_r.data() + (size_t)old_a * r_row,
            r_row * sizeof(float));

        const size_t q_row = (size_t)MLA_MAX_NEW_TOKENS * Hq * d_h;
        std::memcpy(
            dst.q.data() + (size_t)new_a * q_row,
            src.q.data() + (size_t)old_a * q_row,
            q_row * sizeof(float));

        const size_t qr_row = (size_t)MLA_MAX_NEW_TOKENS * Hq * d_r;
        std::memcpy(
            dst.q_rope.data() + (size_t)new_a * qr_row,
            src.q_rope.data() + (size_t)old_a * qr_row,
            qr_row * sizeof(float));
    }

    return dst;
}

static Scenario make_permuted_scenario(const Scenario& src) {
    Scenario dst;
    dst.name = src.name + "_reversed_active_order";
    dst.spec = src.spec;
    dst.weights = src.weights;
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
    const DeviceBuffer<float>& d_new_c,
    const DeviceBuffer<float>& d_new_r,
    const DeviceBuffer<float>& d_q,
    const DeviceBuffer<float>& d_q_rope,
    std::string* error) {
    if (d_active_seq.download() != step.active_seq) {
        if (error) *error = "input active_seq modified";
        return false;
    }
    if (d_new_token_count.download() != step.new_token_count) {
        if (error) *error = "input new_token_count modified";
        return false;
    }

    const std::vector<float> got_c = d_new_c.download();
    const std::vector<float> got_r = d_new_r.download();
    const std::vector<float> got_q = d_q.download();
    const std::vector<float> got_qr = d_q_rope.download();

    if (std::memcmp(got_c.data(), step.new_c.data(), got_c.size() * sizeof(float)) != 0) {
        if (error) *error = "input new_c modified";
        return false;
    }
    if (std::memcmp(got_r.data(), step.new_r.data(), got_r.size() * sizeof(float)) != 0) {
        if (error) *error = "input new_r modified";
        return false;
    }
    if (std::memcmp(got_q.data(), step.q.data(), got_q.size() * sizeof(float)) != 0) {
        if (error) *error = "input q modified";
        return false;
    }
    if (std::memcmp(got_qr.data(), step.q_rope.data(), got_qr.size() * sizeof(float)) != 0) {
        if (error) *error = "input q_rope modified";
        return false;
    }

    return true;
}

static bool run_step(
    const MlaProblemSpec& spec,
    const StepHost& step,
    MlaOracleState* oracle,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    StepResult* result,
    std::string* error) {
    const int A = step.run.active_count;
    const int Hq = spec.Hq;
    const int d_v = spec.d_v;
    const size_t y_count =
        std::max<size_t>(1, (size_t)A * MLA_MAX_NEW_TOKENS * Hq * d_v);
    const size_t lse_count =
        std::max<size_t>(1, (size_t)A * MLA_MAX_NEW_TOKENS * Hq);

    DeviceBuffer<int32_t> d_active_seq;
    DeviceBuffer<int32_t> d_new_token_count;
    DeviceBuffer<float> d_new_c;
    DeviceBuffer<float> d_new_r;
    DeviceBuffer<float> d_q;
    DeviceBuffer<float> d_q_rope;

    d_active_seq.allocate(step.active_seq.size());
    d_new_token_count.allocate(step.new_token_count.size());
    d_new_c.allocate(step.new_c.size());
    d_new_r.allocate(step.new_r.size());
    d_q.allocate(step.q.size());
    d_q_rope.allocate(step.q_rope.size());

    d_active_seq.upload(step.active_seq);
    d_new_token_count.upload(step.new_token_count);
    d_new_c.upload(step.new_c);
    d_new_r.upload(step.new_r);
    d_q.upload(step.q);
    d_q_rope.upload(step.q_rope);

    GuardedDeviceBuffer<float> d_y;
    GuardedDeviceBuffer<float> d_lse;
    GuardedDeviceBuffer<int32_t> d_seq_len;
    GuardedDeviceBuffer<uint64_t> d_cache_hash;
    GuardedDeviceBuffer<uint64_t> d_meta;
    GuardedDeviceBuffer<int32_t> d_sat;
    GuardedDeviceBuffer<int32_t> d_total;

    d_y.allocate(y_count);
    d_lse.allocate(lse_count);
    d_seq_len.allocate((size_t)spec.B);
    d_cache_hash.allocate((size_t)spec.B);
    d_meta.allocate(1);
    d_sat.allocate(1);
    d_total.allocate(1);

    MlaInputs inputs = {};
    inputs.active_seq = d_active_seq.ptr;
    inputs.new_token_count = d_new_token_count.ptr;
    inputs.new_c = d_new_c.ptr;
    inputs.new_r = d_new_r.ptr;
    inputs.q = d_q.ptr;
    inputs.q_rope = d_q_rope.ptr;

    MlaOutputs outputs = {};
    outputs.y = d_y.ptr;
    outputs.lse = d_lse.ptr;
    outputs.seq_len = d_seq_len.ptr;
    outputs.cache_hash = d_cache_hash.ptr;
    outputs.meta_checksum = d_meta.ptr;
    outputs.sat_count = d_sat.ptr;
    outputs.total_tokens = d_total.ptr;

    CUDA_CHECK(solution_run(
        state,
        &step.run,
        &inputs,
        &outputs,
        workspace,
        workspace_bytes,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!compare_input_unchanged(step, d_active_seq, d_new_token_count,
                                 d_new_c, d_new_r, d_q, d_q_rope, error)) {
        return false;
    }

    if (!d_y.check_guards("y", error)) return false;
    if (!d_lse.check_guards("lse", error)) return false;
    if (!d_seq_len.check_guards("seq_len", error)) return false;
    if (!d_cache_hash.check_guards("cache_hash", error)) return false;
    if (!d_meta.check_guards("meta_checksum", error)) return false;
    if (!d_sat.check_guards("sat_count", error)) return false;
    if (!d_total.check_guards("total_tokens", error)) return false;

    const std::vector<float> h_y = d_y.download_data();
    const std::vector<float> h_lse = d_lse.download_data();
    const std::vector<int32_t> h_seq_len = d_seq_len.download_data();
    const std::vector<uint64_t> h_cache_hash = d_cache_hash.download_data();
    const std::vector<uint64_t> h_meta = d_meta.download_data();
    const std::vector<int32_t> h_sat = d_sat.download_data();
    const std::vector<int32_t> h_total = d_total.download_data();

    MlaHostInputsView host_inputs = {};
    host_inputs.active_seq = step.active_seq.data();
    host_inputs.new_token_count = step.new_token_count.data();
    host_inputs.new_c = step.new_c.data();
    host_inputs.new_r = step.new_r.data();
    host_inputs.q = step.q.data();
    host_inputs.q_rope = step.q_rope.data();

    MlaExpected expected;
    oracle->step(step.run, host_inputs, &expected);

    MlaHostOutputsView got = {};
    got.y = h_y.data();
    got.lse = h_lse.data();
    got.seq_len = h_seq_len.data();
    got.cache_hash = h_cache_hash.data();
    got.meta_checksum = h_meta.data();
    got.sat_count = h_sat.data();
    got.total_tokens = h_total.data();

    if (!mla_check_all_outputs(step.run, spec, expected, got, error)) {
        return false;
    }

    if (result) {
        result->active_seq.assign(step.active_seq.begin(), step.active_seq.begin() + A);
        result->new_token_count.assign(
            step.new_token_count.begin(), step.new_token_count.begin() + A);
        result->y.assign(
            h_y.begin(),
            h_y.begin() + ((size_t)A * MLA_MAX_NEW_TOKENS * Hq * d_v));
        result->lse.assign(
            h_lse.begin(),
            h_lse.begin() + ((size_t)A * MLA_MAX_NEW_TOKENS * Hq));
        result->seq_len = h_seq_len;
        result->cache_hash = h_cache_hash;
        result->meta_checksum = h_meta[0];
        result->sat_count = h_sat[0];
        result->total_tokens = h_total[0];
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
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);
    if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Upload weights; freed after init so implementations must copy.
    DeviceBuffer<float> d_wuk;
    DeviceBuffer<float> d_wuv;
    DeviceBuffer<float> d_cos;
    DeviceBuffer<float> d_sin;
    d_wuk.allocate(sc.weights.W_uk.size());
    d_wuv.allocate(sc.weights.W_uv.size());
    d_cos.allocate(sc.weights.rope_cos.size());
    d_sin.allocate(sc.weights.rope_sin.size());
    d_wuk.upload(sc.weights.W_uk);
    d_wuv.upload(sc.weights.W_uv);
    d_cos.upload(sc.weights.rope_cos);
    d_sin.upload(sc.weights.rope_sin);

    MlaInitInputs init_inputs = {};
    init_inputs.W_uk = d_wuk.ptr;
    init_inputs.W_uv = d_wuv.ptr;
    init_inputs.rope_cos = d_cos.ptr;
    init_inputs.rope_sin = d_sin.ptr;

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &init_inputs, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Poison + free the weight buffers: catches solutions that keep the
    // borrowed pointers instead of copying.
    CUDA_CHECK(cudaMemset(d_wuk.ptr, 0xCD, sizeof(float) * d_wuk.count));
    CUDA_CHECK(cudaMemset(d_wuv.ptr, 0xCD, sizeof(float) * d_wuv.count));
    CUDA_CHECK(cudaMemset(d_cos.ptr, 0xCD, sizeof(float) * d_cos.count));
    CUDA_CHECK(cudaMemset(d_sin.ptr, 0xCD, sizeof(float) * d_sin.count));
    d_wuk.free_now();
    d_wuv.free_now();
    d_cos.free_now();
    d_sin.free_now();

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(workspace_bytes);

    MlaOracleState oracle;
    oracle.init(sc.spec);
    oracle.set_weights(sc.weights.W_uk, sc.weights.W_uv,
                       sc.weights.rope_cos, sc.weights.rope_sin);

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

        if (verbose && (!ok || i + 1 == sc.steps.size())) {
            std::printf(
                "scenario %-44s step %02zu/%02zu active=%d %s%s%s\n",
                sc.name.c_str(),
                i,
                sc.steps.size(),
                sc.steps[i].run.active_count,
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
    const int d_v = base.spec.d_v;

    if (a.size() != b.size()) {
        if (error) *error = "result step count mismatch";
        return false;
    }

    for (size_t step = 0; step < a.size(); ++step) {
        if (a[step].seq_len != b[step].seq_len) {
            if (error) {
                std::ostringstream oss;
                oss << "replay seq_len mismatch at step " << step;
                *error = oss.str();
            }
            return false;
        }

        if (a[step].cache_hash != b[step].cache_hash) {
            if (error) {
                std::ostringstream oss;
                oss << "replay cache_hash mismatch at step " << step;
                *error = oss.str();
            }
            return false;
        }

        if (a[step].meta_checksum != b[step].meta_checksum ||
            a[step].sat_count != b[step].sat_count ||
            a[step].total_tokens != b[step].total_tokens) {
            if (error) {
                std::ostringstream oss;
                oss << "replay counter/meta mismatch at step " << step;
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

            const size_t slots = MLA_MAX_NEW_TOKENS;
            for (size_t nt = 0; nt < slots; ++nt) {
                for (int h = 0; h < Hq; ++h) {
                    const size_t li_a = (row_a * slots + nt) * Hq + h;
                    const size_t li_b = ((size_t)row_b * slots + nt) * Hq + h;
                    {
                        const float av = a[step].lse[li_a];
                        const float bv = b[step].lse[li_b];
                        const float diff = std::fabs(av - bv);
                        const float tol = MLA_LSE_ATOL + MLA_LSE_RTOL * std::fabs(av);
                        if (!(diff <= tol)) {
                            if (error) {
                                std::ostringstream oss;
                                oss << "replay lse mismatch at step " << step
                                    << ", seq=" << seq << ", nt=" << nt
                                    << ", h=" << h << ": base=" << av
                                    << ", replay=" << bv;
                                *error = oss.str();
                            }
                            return false;
                        }
                    }
                    for (int d = 0; d < d_v; ++d) {
                        const float av = a[step].y[li_a * d_v + d];
                        const float bv = b[step].y[li_b * d_v + d];
                        const float diff = std::fabs(av - bv);
                        const float tol = MLA_Y_ATOL + MLA_Y_RTOL * std::fabs(av);
                        if (!(diff <= tol)) {
                            if (error) {
                                std::ostringstream oss;
                                oss << "replay y mismatch at step " << step
                                    << ", seq=" << seq << ", nt=" << nt
                                    << ", h=" << h << ", d=" << d
                                    << ": base=" << av << ", replay=" << bv
                                    << ", diff=" << diff << ", tol=" << tol;
                                *error = oss.str();
                            }
                            return false;
                        }
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
        scenarios.push_back(make_boundary_min_scenario());
        scenarios.push_back(make_quant_adversarial_scenario());
        scenarios.push_back(make_heads_max_scenario());
        scenarios.push_back(make_rope_deep_scenario());
        scenarios.push_back(make_singleton_scenario());
        scenarios.push_back(make_asym_all_scenario());
        scenarios.push_back(make_ragged_scenario());
        scenarios.push_back(make_bench_like_scenario());

        int passed = 0;
        int total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results;
            std::vector<StepResult> repeat_results;
            std::vector<StepResult> permuted_results;

            std::string error;

            const bool ok_base = run_scenario_once(
                sc, true, &base_results, &passed, &total, &error);

            Scenario exact_replay = sc;
            exact_replay.name = sc.name + "_exact_replay";

            const bool ok_repeat = run_scenario_once(
                exact_replay, true, &repeat_results, &passed, &total, &error);

            if (ok_base && ok_repeat) {
                std::string cmp_error;
                bool bitexact = true;
                for (size_t s = 0; s < base_results.size() && bitexact; ++s) {
                    if (base_results[s].y != repeat_results[s].y ||
                        base_results[s].lse != repeat_results[s].lse ||
                        base_results[s].cache_hash != repeat_results[s].cache_hash ||
                        base_results[s].meta_checksum != repeat_results[s].meta_checksum) {
                        bitexact = false;
                        std::ostringstream oss;
                        oss << "bit-exact replay mismatch at step " << s;
                        cmp_error = oss.str();
                    }
                }
                if (!bitexact) {
                    all_ok = false;
                    std::printf(
                        "scenario %-44s deterministic replay FAIL  %s\n",
                        sc.name.c_str(), cmp_error.c_str());
                } else {
                    std::printf(
                        "scenario %-44s deterministic replay PASS\n",
                        sc.name.c_str());
                }
            }

            Scenario permuted = make_permuted_scenario(sc);
            const bool ok_permuted = run_scenario_once(
                permuted, true, &permuted_results, &passed, &total, &error);

            if (ok_base && ok_permuted) {
                std::string cmp_error;
                if (!compare_order_invariant_outputs(
                        sc, base_results, permuted_results, &cmp_error)) {
                    all_ok = false;
                    std::printf(
                        "scenario %-44s permuted-order replay FAIL  %s\n",
                        sc.name.c_str(), cmp_error.c_str());
                } else {
                    std::printf(
                        "scenario %-44s permuted-order replay PASS\n",
                        sc.name.c_str());
                }
            }

            if (!ok_base || !ok_repeat || !ok_permuted) {
                all_ok = false;
                std::printf(
                    "scenario %-44s FAIL  %s\n",
                    sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
