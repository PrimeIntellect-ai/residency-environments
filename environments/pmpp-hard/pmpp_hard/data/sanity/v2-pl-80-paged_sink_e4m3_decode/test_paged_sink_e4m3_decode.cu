// ============================================================================
// file: test_paged_sink_e4m3_decode.cu
// ============================================================================

#include "paged_sink_e4m3_decode_common.h"
#include "paged_sink_e4m3_decode_oracle.hpp"

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

static constexpr uint64_t g_state = 0x9b5ad4b1c2e07d63ULL;
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
    PseRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> new_token_count;
    std::vector<float> new_k;
    std::vector<float> new_v;
    std::vector<float> q;
};

struct Scenario {
    std::string name;
    PseProblemSpec spec;
    std::vector<StepHost> steps;
};

struct StepResult {
    std::vector<int32_t> active_seq;
    std::vector<float> y;
    std::vector<float> lse;
    std::vector<int32_t> seq_len;
    std::vector<uint64_t> kv_hash;
    uint64_t page_state_checksum = 0;
    int32_t free_pages = 0;
    int32_t total_allocs = 0;
    int32_t total_frees = 0;
};

static int needed_pages(int B, int S, int W, int P) {
    return B * (pse_ceil_div_int(S, P) + pse_ceil_div_int(W, P) + 3);
}

static PseProblemSpec make_spec(
    int B,
    int Hq,
    int Hkv,
    int D,
    int page_size,
    int n_sink,
    int window,
    int max_seq_len,
    int max_pages) {
    PseProblemSpec spec = {};
    spec.abi_version = PSE_ABI_VERSION;
    spec.B = B;
    spec.Hq = Hq;
    spec.Hkv = Hkv;
    spec.D = D;
    spec.page_size = page_size;
    spec.n_sink = n_sink;
    spec.window = window;
    spec.max_seq_len = max_seq_len;
    spec.max_pages = max_pages;
    spec.flags = 0;

    if (!pse_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid PseProblemSpec generated");
    }
    if (max_pages < needed_pages(B, n_sink, window, page_size)) {
        throw std::runtime_error("scenario violates max_pages invariant");
    }

    return spec;
}

static void fill_step_tensors(
    const PseProblemSpec& spec,
    StepHost* step,
    uint64_t seed) {
    const int A = step->run.active_count;
    const int Hq = spec.Hq;
    const int Hkv = spec.Hkv;
    const int D = spec.D;

    const size_t a_count = A > 0 ? static_cast<size_t>(A) : 1;

    step->new_token_count.resize(a_count, 0);
    step->new_k.resize(a_count * PSE_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D, 0.0f);
    step->new_v.resize(a_count * PSE_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D, 0.0f);
    step->q.resize(a_count * (size_t)Hq * (size_t)D, 0.0f);

    for (int a = 0; a < A; ++a) {
        const int seq = step->active_seq[(size_t)a];
        SplitMix64 rng(seed ^ (0x8e3779b97f4a7c15ULL * (uint64_t)(seq + 1)));

        for (int nt = 0; nt < PSE_MAX_NEW_TOKENS; ++nt) {
            for (int h = 0; h < Hkv; ++h) {
                for (int d = 0; d < D; ++d) {
                    const size_t idx =
                        ((size_t)a * PSE_MAX_NEW_TOKENS + (size_t)nt) *
                        (size_t)Hkv * (size_t)D +
                        (size_t)h * (size_t)D +
                        (size_t)d;

                    step->new_k[idx] = rng.uniform_float(-2.50f, 2.50f);
                    step->new_v[idx] = rng.uniform_float(-2.50f, 2.50f);
                }
            }
        }

        for (int h = 0; h < Hq; ++h) {
            for (int d = 0; d < D; ++d) {
                const size_t idx =
                    ((size_t)a * (size_t)Hq + (size_t)h) *
                    (size_t)D + (size_t)d;
                step->q[idx] = rng.uniform_float(-0.50f, 0.50f);
            }
        }
    }
}

static StepHost make_step(
    const PseProblemSpec& spec,
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
        for (size_t j = i + 1; j < active.size(); ++j) {
            if (active[i] == active[j]) {
                throw std::runtime_error("duplicate active id in step");
            }
        }
    }

    StepHost step;
    step.run = {};
    step.run.abi_version = PSE_ABI_VERSION;
    step.run.active_count = static_cast<int32_t>(active.size());
    step.run.step_id = step_id;

    if (!pse_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid PseRunSpec generated");
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

// ---------------------------------------------------------------------------
// Adversarial quantization vectors. Each vector's element 0 anchors amax so
// the scale exponent is known; the rest probe exact ties, saturation, signed
// zeros, subnormal-region magnitudes, and exact powers of two.
// ---------------------------------------------------------------------------
static void craft_quant_vector(int variant, float* x, int D, bool bounded) {
    for (int d = 0; d < D; ++d) x[d] = 0.0f;

    // For K vectors (bounded=true) huge anchors are excluded and the case-0
    // anchor is scaled by 2^-7: the scale exponent shifts by exactly -7, so
    // every z value (and therefore every stored byte) is identical, but the
    // dequantized magnitudes stay small enough that attention scores remain
    // O(1) and any correct fp32 softmax stays well inside tolerance.
    const int n_variants = bounded ? 4 : 5;
    const float c0 = bounded ? 0.0078125f : 1.0f;  // 2^-7 or 1

    switch (variant % n_variants) {
        case 0:
            // amax anchor 460*c0: scale_exp = 0 (or -7), z = x / c0.
            x[0] = 460.0f * c0;          // saturates to 448
            x[1] = 448.0f * c0;          // exact max finite
            x[2] = 449.0f * c0;          // saturates
            x[3] = -456.0f * c0;         // saturates negative
            x[4] = 0.0f;                 // +0
            x[5] = -0.0f;                // -0 -> 0x80
            x[6] = 0.75f * c0;           // exact
            x[7] = 0.0009765625f * c0;   // z=2^-10: tie 0 vs 2^-9, even -> 0
            x[8] = -0.0009765625f * c0;  // -> 0x80
            x[9] = 0.0029296875f * c0;   // z=1.5*2^-9: tie 0x01/0x02 -> 0x02
            x[10] = 12.5f * c0;          // z=12.5: tie 12/13 -> 12 (even M)
            x[11] = 208.0f * c0;         // exact grid value
            x[12] = -368.0f * c0;        // z=-368: tie 352/384 -> -384 (even M)
            x[13] = 0.001953125f * c0;   // z=2^-9 exact min subnormal
            break;
        case 1:
            // amax anchor 2.0 -> kfloor=1, scale_exp=-7, z = x*128.
            x[0] = 2.0f;
            x[1] = 3.125f / 128.0f;    // z=3.125: tie 3.0/3.25 -> 3.0
            x[2] = -3.375f / 128.0f;   // z=-3.375: tie 3.25/3.5 -> -3.5
            x[3] = 0.21875f / 128.0f;  // z exact subnormal-ish normal
            x[4] = 1.75f / 128.0f;     // z=1.75 exact
            x[5] = -2.0f;              // z=-256 exact
            x[6] = 0.0f;
            x[7] = 1.0f / 65536.0f;    // z tiny -> rounds near zero
            break;
        case 2:
            // amax exactly a power of two: 4.0 -> kfloor=2, scale_exp=-6.
            x[0] = 4.0f;               // z=256 exact
            x[1] = -4.0f;
            x[2] = 3.9999999f;         // z just under 256
            x[3] = 2.0000002f;         // z just over 128
            x[4] = 0.5f;               // z=32 exact
            x[5] = 0.4921875f;         // z=31.5: tie 30/32? step 2: 31.5->32
            break;
        case 3:
            // tiny anchor: 2^-20 -> kfloor=-20, scale_exp=-28.
            x[0] = 0.00000095367431640625f;   // 2^-20, z=256
            x[1] = 0.0000004768371582031250f; // 2^-21, z=128
            x[2] = -0.00000059604644775390625f; // -2^-20.678..., z=-160
            break;
        default:
            // huge anchor: 2^15 -> kfloor=15, scale_exp=7.
            x[0] = 32768.0f;   // z=256
            x[1] = 32767.0f;   // z=255.99... -> 256
            x[2] = -17000.0f;  // z=-132.8 -> nearest
            x[3] = 448.0f;     // z=3.5 exact
            break;
    }
}

static void inject_adversarial_quant(
    const PseProblemSpec& spec,
    StepHost* step,
    int step_idx) {
    const int A = step->run.active_count;
    const int Hkv = spec.Hkv;
    const int D = spec.D;

    for (int a = 0; a < A; ++a) {
        if ((a + step_idx) % 2 != 0) continue;
        const int cnt = step->new_token_count[(size_t)a];
        if (cnt <= 0) continue;
        const int nt = step_idx % cnt;
        const int h = (a + step_idx) % Hkv;
        const size_t base =
            ((size_t)a * PSE_MAX_NEW_TOKENS + (size_t)nt) *
            (size_t)Hkv * (size_t)D +
            (size_t)h * (size_t)D;

        craft_quant_vector(step_idx + a, step->new_k.data() + base, D, true);
        craft_quant_vector(step_idx + a + 1, step->new_v.data() + base, D, false);
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

static Scenario make_boundary_small_scenario() {
    Scenario sc;
    sc.name = "boundary_small_S4_W16_P8";
    sc.spec = make_spec(6, 4, 2, 64, 8, 4, 16, 512, 40);

    for (int s = 0; s < 44; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        if (s == 0) {
            active = {0, 1, 2, 3, 4, 5};
            nt = {8, 8, 8, 8, 8, 8};
        } else if (s == 1) {
            active = {0, 1, 2, 3, 4, 5};
            nt = {1, 2, 0, 8, 3, 5};
        } else if (s == 2 || s == 17) {
            // empty step
        } else if (s == 9) {
            active = {5, 3, 1};
            nt = {0, 0, 0};
        } else {
            active.push_back(0);
            nt.push_back(2);
            active.push_back(1);
            nt.push_back(1);
            if (s % 2 == 0) {
                active.push_back(2);
                nt.push_back(8);
            }
            if (s % 2 == 1) {
                active.push_back(3);
                nt.push_back(0);
            }
            if (s % 3 == 0) {
                active.push_back(4);
                nt.push_back((s % 5 == 0) ? 5 : 1);
            }
            if (s % 7 == 1) {
                active.push_back(5);
                nt.push_back(2);
            }
        }

        sc.steps.push_back(make_step(
            sc.spec,
            s,
            active,
            nt,
            g_state ^ 0x100000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_quant_adversarial_scenario() {
    Scenario sc;
    sc.name = "quant_adversarial_ties_saturation";
    sc.spec = make_spec(4, 2, 2, 64, 8, 2, 24, 256, 32);

    for (int s = 0; s < 40; ++s) {
        std::vector<int32_t> active = {0, 1, 2, 3};
        std::vector<int32_t> nt = {
            1 + (s % 2),
            1 + ((s + 1) % 2),
            (s % 5 == 0) ? 0 : 1,
            2};

        StepHost step = make_step(
            sc.spec,
            s,
            active,
            nt,
            g_state ^ 0x200000000ULL ^ static_cast<uint64_t>(s));

        inject_adversarial_quant(sc.spec, &step, s);
        sc.steps.push_back(step);
    }

    return sc;
}

static Scenario make_group1_nosink_scenario() {
    Scenario sc;
    sc.name = "group1_nosink_S0_W64_P16";
    sc.spec = make_spec(8, 8, 8, 128, 16, 0, 64, 1024, 64);

    for (int s = 0; s < 36; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        if (s == 7) {
            // empty
        } else {
            for (int b = 0; b < sc.spec.B; ++b) {
                if ((b + s) % 3 == 2) continue;
                active.push_back(b);
                if (b % 2 == 0) {
                    nt.push_back((s % 3 == 0) ? 8 : 2);
                } else {
                    nt.push_back(s % 4 == 1 ? 0 : 1);
                }
            }
        }

        sc.steps.push_back(make_step(
            sc.spec,
            s,
            active,
            nt,
            g_state ^ 0x300000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_group32_smallwin_scenario() {
    Scenario sc;
    sc.name = "group32_smallwin_S16_W16_P32";
    sc.spec = make_spec(4, 32, 1, 64, 32, 16, 16, 512, 24);

    for (int s = 0; s < 36; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        if (s == 13) {
            // empty
        } else {
            active.push_back(0);
            nt.push_back((s % 4 == 3) ? 8 : 2);
            if (s % 2 == 0) {
                active.push_back(1);
                nt.push_back(4);
            }
            if (s % 3 == 0) {
                active.push_back(2);
                nt.push_back(0);
            }
            if (s % 5 == 2) {
                active.push_back(3);
                nt.push_back(7);
            }
        }

        sc.steps.push_back(make_step(
            sc.spec,
            s,
            active,
            nt,
            g_state ^ 0x400000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_churn_tight_scenario() {
    // max_pages == exact invariant bound: stresses reuse of freed pages.
    Scenario sc;
    sc.name = "churn_tight_pool_S8_W40_P8";
    sc.spec = make_spec(12, 8, 2, 64, 8, 8, 40, 2048, 108);

    for (int s = 0; s < 80; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        if (s == 25 || s == 61) {
            // empty
        } else {
            for (int b = 0; b < sc.spec.B; ++b) {
                if ((b * 5 + s) % 4 == 3) continue;
                active.push_back(b);
                if ((b + s) % 6 == 0) {
                    nt.push_back(8);
                } else if ((b + s) % 6 == 3) {
                    nt.push_back(0);
                } else {
                    nt.push_back(1 + ((b + s) % 3));
                }
            }
        }

        sc.steps.push_back(make_step(
            sc.spec,
            s,
            active,
            nt,
            g_state ^ 0x500000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_full_attn_scenario() {
    // W >= any reachable L: nothing ever dies; live set is [0, L).
    Scenario sc;
    sc.name = "full_attn_S64_W1024_P16";
    sc.spec = make_spec(8, 16, 4, 128, 16, 64, 1024, 256, 576);

    for (int s = 0; s < 24; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        if (s == 5) {
            // empty
        } else {
            for (int b = 0; b < sc.spec.B; ++b) {
                if ((b + s) % 2 == 1 && b != 0) continue;
                active.push_back(b);
                nt.push_back((s % 3 == 0) ? 8 : (b % 3));
            }
        }

        sc.steps.push_back(make_step(
            sc.spec,
            s,
            active,
            nt,
            g_state ^ 0x600000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_sink_heavy_scenario() {
    Scenario sc;
    sc.name = "sink_heavy_S64_W16_P8";
    sc.spec = make_spec(16, 4, 4, 64, 8, 64, 16, 512, 224);

    for (int s = 0; s < 30; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        for (int b = 0; b < sc.spec.B; ++b) {
            if ((b + 2 * s) % 3 == 1) continue;
            active.push_back(b);
            if (b % 4 == 0) {
                nt.push_back(8);  // crosses S=64 around step 8
            } else if (b % 4 == 1) {
                nt.push_back(s % 2);
            } else {
                nt.push_back((s + b) % 4);
            }
        }

        sc.steps.push_back(make_step(
            sc.spec,
            s,
            active,
            nt,
            g_state ^ 0x700000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

static Scenario make_bench_like_scenario() {
    Scenario sc;
    sc.name = "bench_like_mid_S32_W480_P16";
    sc.spec = make_spec(24, 16, 4, 128, 16, 32, 480, 4096, 1024);

    for (int s = 0; s < 20; ++s) {
        std::vector<int32_t> active;
        std::vector<int32_t> nt;

        if (s < 10) {
            for (int b = 0; b < sc.spec.B; ++b) {
                active.push_back(b);
                nt.push_back(8);
            }
        } else {
            for (int b = 0; b < sc.spec.B; ++b) {
                if ((b + s) % 3 == 0) continue;
                active.push_back(b);
                nt.push_back(1 + ((b + s) % 2));
            }
        }

        sc.steps.push_back(make_step(
            sc.spec,
            s,
            active,
            nt,
            g_state ^ 0x800000000ULL ^ static_cast<uint64_t>(s)));
    }

    return sc;
}

// ---------------------------------------------------------------------------

static StepHost permute_step_rows(const PseProblemSpec& spec, const StepHost& src) {
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
    dst.new_k.assign((size_t)A * PSE_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D, 0.0f);
    dst.new_v.assign((size_t)A * PSE_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D, 0.0f);
    dst.q.assign((size_t)A * (size_t)Hq * (size_t)D, 0.0f);

    for (int new_a = 0; new_a < A; ++new_a) {
        const int old_a = A - 1 - new_a;
        dst.active_seq[(size_t)new_a] = src.active_seq[(size_t)old_a];
        dst.new_token_count[(size_t)new_a] = src.new_token_count[(size_t)old_a];

        const size_t kv_row = (size_t)PSE_MAX_NEW_TOKENS * (size_t)Hkv * (size_t)D;
        std::memcpy(
            dst.new_k.data() + (size_t)new_a * kv_row,
            src.new_k.data() + (size_t)old_a * kv_row,
            kv_row * sizeof(float));
        std::memcpy(
            dst.new_v.data() + (size_t)new_a * kv_row,
            src.new_v.data() + (size_t)old_a * kv_row,
            kv_row * sizeof(float));

        const size_t q_row = (size_t)Hq * (size_t)D;
        std::memcpy(
            dst.q.data() + (size_t)new_a * q_row,
            src.q.data() + (size_t)old_a * q_row,
            q_row * sizeof(float));
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

    const std::vector<float> got_k = d_new_k.download();
    const std::vector<float> got_v = d_new_v.download();
    const std::vector<float> got_q = d_q.download();

    if (std::memcmp(got_k.data(), step.new_k.data(), got_k.size() * sizeof(float)) != 0) {
        if (error) *error = "input new_k modified";
        return false;
    }
    if (std::memcmp(got_v.data(), step.new_v.data(), got_v.size() * sizeof(float)) != 0) {
        if (error) *error = "input new_v modified";
        return false;
    }
    if (std::memcmp(got_q.data(), step.q.data(), got_q.size() * sizeof(float)) != 0) {
        if (error) *error = "input q modified";
        return false;
    }

    return true;
}

static bool run_step(
    const PseProblemSpec& spec,
    const StepHost& step,
    PseOracleState* oracle,
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
    const size_t lse_count = std::max<size_t>(1, (size_t)A * (size_t)Hq);

    DeviceBuffer<int32_t> d_active_seq;
    DeviceBuffer<int32_t> d_new_token_count;
    DeviceBuffer<float> d_new_k;
    DeviceBuffer<float> d_new_v;
    DeviceBuffer<float> d_q;

    d_active_seq.allocate(step.active_seq.size());
    d_new_token_count.allocate(step.new_token_count.size());
    d_new_k.allocate(step.new_k.size());
    d_new_v.allocate(step.new_v.size());
    d_q.allocate(step.q.size());

    d_active_seq.upload(step.active_seq);
    d_new_token_count.upload(step.new_token_count);
    d_new_k.upload(step.new_k);
    d_new_v.upload(step.new_v);
    d_q.upload(step.q);

    GuardedDeviceBuffer<float> d_y;
    GuardedDeviceBuffer<float> d_lse;
    GuardedDeviceBuffer<int32_t> d_seq_len;
    GuardedDeviceBuffer<uint64_t> d_kv_hash;
    GuardedDeviceBuffer<uint64_t> d_pcs;
    GuardedDeviceBuffer<int32_t> d_free_pages;
    GuardedDeviceBuffer<int32_t> d_total_allocs;
    GuardedDeviceBuffer<int32_t> d_total_frees;

    d_y.allocate(y_count);
    d_lse.allocate(lse_count);
    d_seq_len.allocate((size_t)spec.B);
    d_kv_hash.allocate((size_t)spec.B);
    d_pcs.allocate(1);
    d_free_pages.allocate(1);
    d_total_allocs.allocate(1);
    d_total_frees.allocate(1);

    PseInputs inputs = {};
    inputs.active_seq = d_active_seq.ptr;
    inputs.new_token_count = d_new_token_count.ptr;
    inputs.new_k = d_new_k.ptr;
    inputs.new_v = d_new_v.ptr;
    inputs.q = d_q.ptr;

    PseOutputs outputs = {};
    outputs.y = d_y.ptr;
    outputs.lse = d_lse.ptr;
    outputs.seq_len = d_seq_len.ptr;
    outputs.kv_hash = d_kv_hash.ptr;
    outputs.page_state_checksum = d_pcs.ptr;
    outputs.free_pages = d_free_pages.ptr;
    outputs.total_allocs = d_total_allocs.ptr;
    outputs.total_frees = d_total_frees.ptr;

    CUDA_CHECK(solution_run(
        state,
        &step.run,
        &inputs,
        &outputs,
        workspace,
        workspace_bytes,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!compare_input_unchanged(step, d_active_seq, d_new_token_count, d_new_k, d_new_v, d_q, error)) {
        return false;
    }

    if (!d_y.check_guards("y", error)) return false;
    if (!d_lse.check_guards("lse", error)) return false;
    if (!d_seq_len.check_guards("seq_len", error)) return false;
    if (!d_kv_hash.check_guards("kv_hash", error)) return false;
    if (!d_pcs.check_guards("page_state_checksum", error)) return false;
    if (!d_free_pages.check_guards("free_pages", error)) return false;
    if (!d_total_allocs.check_guards("total_allocs", error)) return false;
    if (!d_total_frees.check_guards("total_frees", error)) return false;

    const std::vector<float> h_y = d_y.download_data();
    const std::vector<float> h_lse = d_lse.download_data();
    const std::vector<int32_t> h_seq_len = d_seq_len.download_data();
    const std::vector<uint64_t> h_kv_hash = d_kv_hash.download_data();
    const std::vector<uint64_t> h_pcs = d_pcs.download_data();
    const std::vector<int32_t> h_free_pages = d_free_pages.download_data();
    const std::vector<int32_t> h_total_allocs = d_total_allocs.download_data();
    const std::vector<int32_t> h_total_frees = d_total_frees.download_data();

    PseHostInputsView host_inputs = {};
    host_inputs.active_seq = step.active_seq.data();
    host_inputs.new_token_count = step.new_token_count.data();
    host_inputs.new_k = step.new_k.data();
    host_inputs.new_v = step.new_v.data();
    host_inputs.q = step.q.data();

    PseExpected expected;
    oracle->step(step.run, host_inputs, &expected);

    PseHostOutputsView got = {};
    got.y = h_y.data();
    got.lse = h_lse.data();
    got.seq_len = h_seq_len.data();
    got.kv_hash = h_kv_hash.data();
    got.page_state_checksum = h_pcs.data();
    got.free_pages = h_free_pages.data();
    got.total_allocs = h_total_allocs.data();
    got.total_frees = h_total_frees.data();

    if (!pse_check_all_outputs(step.run, spec, expected, got, error)) {
        return false;
    }

    if (result) {
        result->active_seq.assign(step.active_seq.begin(), step.active_seq.begin() + A);
        result->y.assign(h_y.begin(), h_y.begin() + ((size_t)A * (size_t)Hq * (size_t)D));
        result->lse.assign(h_lse.begin(), h_lse.begin() + ((size_t)A * (size_t)Hq));
        result->seq_len = h_seq_len;
        result->kv_hash = h_kv_hash;
        result->page_state_checksum = h_pcs[0];
        result->free_pages = h_free_pages[0];
        result->total_allocs = h_total_allocs[0];
        result->total_frees = h_total_frees[0];
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

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(workspace_bytes);

    PseOracleState oracle;
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

        if (verbose && (!ok || i + 1 == sc.steps.size())) {
            std::printf(
                "scenario %-40s step %02zu/%02zu active=%d %s%s%s\n",
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
    bool require_page_checksum_equal,
    std::string* error) {
    const int Hq = base.spec.Hq;
    const int D = base.spec.D;

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

        if (a[step].kv_hash != b[step].kv_hash) {
            if (error) {
                std::ostringstream oss;
                oss << "replay kv_hash mismatch at step " << step;
                *error = oss.str();
            }
            return false;
        }

        if (a[step].free_pages != b[step].free_pages ||
            a[step].total_allocs != b[step].total_allocs ||
            a[step].total_frees != b[step].total_frees) {
            if (error) {
                std::ostringstream oss;
                oss << "replay page counter mismatch at step " << step;
                *error = oss.str();
            }
            return false;
        }

        if (require_page_checksum_equal &&
            a[step].page_state_checksum != b[step].page_state_checksum) {
            if (error) {
                std::ostringstream oss;
                oss << "replay page_state_checksum mismatch at step " << step
                    << ": base=0x" << std::hex << a[step].page_state_checksum
                    << ", replay=0x" << b[step].page_state_checksum;
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

            for (int h = 0; h < Hq; ++h) {
                {
                    const float av = a[step].lse[row_a * (size_t)Hq + (size_t)h];
                    const float bv = b[step].lse[(size_t)row_b * (size_t)Hq + (size_t)h];
                    const float diff = std::fabs(av - bv);
                    const float tol = PSE_LSE_ATOL + PSE_LSE_RTOL * std::fabs(av);
                    if (!(diff <= tol)) {
                        if (error) {
                            std::ostringstream oss;
                            oss << "replay lse mismatch at step " << step
                                << ", seq=" << seq << ", h=" << h
                                << ": base=" << av << ", replay=" << bv;
                            *error = oss.str();
                        }
                        return false;
                    }
                }

                for (int d = 0; d < D; ++d) {
                    const size_t idx_a =
                        (row_a * (size_t)Hq + (size_t)h) * (size_t)D + (size_t)d;
                    const size_t idx_b =
                        ((size_t)row_b * (size_t)Hq + (size_t)h) * (size_t)D + (size_t)d;

                    const float av = a[step].y[idx_a];
                    const float bv = b[step].y[idx_b];
                    const float diff = std::fabs(av - bv);
                    const float tol = PSE_Y_ATOL + PSE_Y_RTOL * std::fabs(av);

                    if (!(diff <= tol)) {
                        if (error) {
                            std::ostringstream oss;
                            oss << "replay y mismatch at step " << step
                                << ", seq=" << seq
                                << ", h=" << h
                                << ", d=" << d
                                << ": base=" << av
                                << ", replay=" << bv
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
        scenarios.push_back(make_boundary_small_scenario());
        scenarios.push_back(make_quant_adversarial_scenario());
        scenarios.push_back(make_group1_nosink_scenario());
        scenarios.push_back(make_group32_smallwin_scenario());
        scenarios.push_back(make_churn_tight_scenario());
        scenarios.push_back(make_full_attn_scenario());
        scenarios.push_back(make_sink_heavy_scenario());
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
                if (!compare_order_invariant_outputs(sc, base_results, repeat_results, true, &cmp_error)) {
                    all_ok = false;
                    std::printf(
                        "scenario %-40s deterministic replay FAIL  %s\n",
                        sc.name.c_str(),
                        cmp_error.c_str());
                } else {
                    std::printf(
                        "scenario %-40s deterministic replay PASS\n",
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
                if (!compare_order_invariant_outputs(sc, base_results, permuted_results, false, &cmp_error)) {
                    all_ok = false;
                    std::printf(
                        "scenario %-40s permuted-order replay FAIL  %s\n",
                        sc.name.c_str(),
                        cmp_error.c_str());
                } else {
                    std::printf(
                        "scenario %-40s permuted-order replay PASS\n",
                        sc.name.c_str());
                }
            }

            if (!ok_base || !ok_repeat || !ok_permuted) {
                all_ok = false;
                std::printf(
                    "scenario %-40s FAIL  %s\n",
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
