// ============================================================================
// file: test_spec_tree_verify_e5m2_kv.cu
// ============================================================================

#include "spec_tree_verify_e5m2_kv_common.h"
#include "spec_tree_verify_e5m2_kv_oracle.hpp"

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

static constexpr uint64_t g_state = 0x7de91b40c85fa263ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;
static constexpr uint64_t kRejectXor = 0x5A5A5A5A5A5A5A5AULL;

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
    StvRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> node_count;
    std::vector<int32_t> parent;
    std::vector<float> draft_k;
    std::vector<float> draft_v;
    std::vector<float> q;
    std::vector<uint64_t> target_sig;
    std::vector<float> bonus_k;
    std::vector<float> bonus_v;
};

struct Scenario {
    std::string name;
    StvProblemSpec spec;
    std::vector<StepHost> steps;
};

struct StepResult {
    std::vector<int32_t> active_seq;
    std::vector<int32_t> node_count;
    std::vector<float> y;
    std::vector<float> lse;
    std::vector<int32_t> accepted_tail;
    std::vector<int32_t> accepted_len;
    std::vector<int32_t> seq_len;
    std::vector<uint64_t> kv_hash;
    uint64_t page_state_checksum = 0;
    int32_t free_pages = 0;
    int32_t total_allocs = 0;
    int32_t total_frees = 0;
};

static StvProblemSpec make_spec(
    int B, int Hq, int Hkv, int D, int P, int N, int msl, int max_pages) {
    StvProblemSpec spec = {};
    spec.abi_version = STV_ABI_VERSION;
    spec.B = B;
    spec.Hq = Hq;
    spec.Hkv = Hkv;
    spec.D = D;
    spec.page_size = P;
    spec.max_nodes = N;
    spec.max_seq_len = msl;
    spec.max_pages = max_pages;
    spec.flags = 0;

    if (!stv_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid StvProblemSpec generated");
    }
    if (max_pages <
        B * (stv_ceil_div_int(msl, P) + stv_ceil_div_int(N, P))) {
        throw std::runtime_error("scenario violates max_pages invariant");
    }
    return spec;
}

// ---------------------------------------------------------------------------
// Adversarial E5M2 quantization vectors. Element 0 anchors amax; the rest
// probe saturation above 57344 after scaling, 2-bit-mantissa RNE ties,
// signed zeros, and subnormal-region magnitudes.
// ---------------------------------------------------------------------------
static void craft_e5m2_vector(int variant, float* x, int D) {
    for (int d = 0; d < D; ++d) x[d] = 0.0f;

    switch (variant % 5) {
        case 0:
            // amax 30000: kfloor=14, se=-1, z = 2x.
            x[0] = 30000.0f;      // z=60000 > 57344 -> saturates to 0x7B
            x[1] = -30000.0f;     // -> 0xFB
            x[2] = 28672.0f;      // z=57344 exact max finite
            x[3] = 28671.0f;      // z=57342 -> rounds within top bin
            x[4] = 0.0f;          // +0
            x[5] = -0.0f;         // -0 -> 0x80
            x[6] = 5.0f;          // z=10: tie 10 between 8-step? z=10 exact E5M2
            x[7] = 2.5f;          // z=5: tie between 4 and 6 -> 4 (even M)
            x[8] = 3.5f;          // z=7: tie between 6 and 8 -> 8 (even M)
            x[9] = -2.5f;         // -> -4
            break;
        case 1:
            // amax exactly 2^12=4096: kfloor=12, se=-3, z = x*8.
            x[0] = 4096.0f;       // z=32768 exact
            x[1] = -4096.0f;
            x[2] = 640.0f;        // z=5120 exact
            x[3] = 560.0f;        // z=4480: tie 4096/4864? step=1024: 4480 lies
                                  // between 4096 and 5120 midpoint 4608 -> 4096
            x[4] = 576.0f;        // z=4608: exact midpoint -> 4096 (even M=0)
            x[5] = 608.0f;        // z=4864: midpoint 4608..5120? no: exact repr?
                                  // 4864 = (1+3/4)*2^11? = 1.1875*2^12 -> between
                                  // 4608 mid handled by encoder
            break;
        case 2:
            // tiny anchor: 2^-40: kfloor=-40, se=-55, z = x*2^55.
            x[0] = 9.094947017729282e-13f;    // 2^-40, z=2^15=32768
            x[1] = 4.547473508864641e-13f;    // z=16384
            x[2] = -6.821210263296962e-13f;   // z=-24576 exact
            break;
        case 3:
            // subnormal probes: anchor 2^15-ish with tiny elements.
            x[0] = 32768.0f;      // kfloor=15, se=0, z=x
            x[1] = 0.0000152587890625f;   // z=2^-16 = smallest subnormal
            x[2] = 0.00002288818359375f;  // z=1.5*2^-16: tie -> 2*2^-16 (even)
            x[3] = 0.00000762939453125f;  // z=0.5*2^-16: tie -> 0
            x[4] = -0.0000457763671875f;  // z=-3*2^-16 exact subnormal
            x[5] = 0.0001f;               // z rounds near 1.625*2^-13
            break;
        default:
            // amax 1.75*2^k exact top-of-bin values.
            x[0] = 448.0f;        // kfloor=8, se=-7, z=x*128: 57344 exact max
            x[1] = 447.0f;        // z=57216 -> rounds to 57344 (nearest)
            x[2] = -448.0f;       // -57344 exact
            x[3] = 0.875f;        // z=112 exact
            x[4] = 0.9375f;       // z=120: tie 112/128 -> 128 (even M=0)
            break;
    }
}

// ---------------------------------------------------------------------------
// Step construction. Acceptance plans are byte-exact: sigs are computed with
// the oracle quantizer; rejected nodes get sig ^ kRejectXor.
// ---------------------------------------------------------------------------

struct TreePlan {
    std::vector<int32_t> parent;   // [n]
    std::vector<uint8_t> accept;   // [n] whether target_sig matches
};

static StepHost make_step(
    const StvProblemSpec& spec,
    StvOracleState* sig_helper,
    int step_id,
    const std::vector<int32_t>& active,
    const std::vector<TreePlan>& plans,
    uint64_t seed,
    bool adversarial_quant) {
    if (active.size() != plans.size()) {
        throw std::runtime_error("active/plans size mismatch");
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

    const int A = static_cast<int>(active.size());
    const int N = spec.max_nodes;
    const int Hq = spec.Hq;
    const int Hkv = spec.Hkv;
    const int D = spec.D;

    StepHost step;
    step.run = {};
    step.run.abi_version = STV_ABI_VERSION;
    step.run.active_count = A;
    step.run.step_id = step_id;
    if (!stv_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid StvRunSpec generated");
    }

    const size_t a_count = A > 0 ? static_cast<size_t>(A) : 1;
    step.active_seq = active;
    if (step.active_seq.empty()) step.active_seq.push_back(0);
    step.node_count.assign(a_count, 1);
    step.parent.assign(a_count * N, -1);
    step.draft_k.assign(a_count * N * Hkv * (size_t)D, 0.0f);
    step.draft_v.assign(a_count * N * Hkv * (size_t)D, 0.0f);
    step.q.assign(a_count * N * Hq * (size_t)D, 0.0f);
    step.target_sig.assign(a_count * N, 0);
    step.bonus_k.assign(a_count * Hkv * (size_t)D, 0.0f);
    step.bonus_v.assign(a_count * Hkv * (size_t)D, 0.0f);

    for (int a = 0; a < A; ++a) {
        const TreePlan& plan = plans[a];
        const int n = static_cast<int>(plan.parent.size());
        if (n < 1 || n > N) throw std::runtime_error("bad plan node count");
        step.node_count[a] = n;

        const int seq = active[a];
        SplitMix64 rng(seed ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(seq + 1)));

        for (int i = 0; i < N; ++i) {
            const bool pad = i >= n;
            step.parent[(size_t)a * N + i] = pad ? -1 : plan.parent[i];

            for (int hh = 0; hh < Hkv; ++hh) {
                float* kv = step.draft_k.data() +
                    (((size_t)a * N + i) * Hkv + hh) * D;
                float* vv = step.draft_v.data() +
                    (((size_t)a * N + i) * Hkv + hh) * D;
                for (int d = 0; d < D; ++d) {
                    kv[d] = pad ? rng.uniform_float(-2.0e4f, 2.0e4f)
                                : rng.uniform_float(-2.5f, 2.5f);
                    vv[d] = pad ? rng.uniform_float(-2.0e4f, 2.0e4f)
                                : rng.uniform_float(-2.5f, 2.5f);
                }
                if (!pad && adversarial_quant && ((i + step_id) % 3 == 0)) {
                    craft_e5m2_vector(step_id + i + hh, kv, D);
                    craft_e5m2_vector(step_id + i + hh + 1, vv, D);
                    // Scale K crafts by an exact power of two: every scaled
                    // z (and therefore every byte, sig, and hash) is
                    // IDENTICAL, but dequantized K magnitudes stay O(1) so
                    // attention scores remain well-conditioned in fp32.
                    for (int d = 0; d < D; ++d) kv[d] *= 0.0001220703125f;
                }
            }
            for (int hq = 0; hq < Hq; ++hq) {
                float* qq = step.q.data() +
                    (((size_t)a * N + i) * Hq + hq) * D;
                for (int d = 0; d < D; ++d) {
                    qq[d] = pad ? rng.uniform_float(-3.0e3f, 3.0e3f)
                                : rng.uniform_float(-1.0f, 1.0f);
                }
            }
        }

        for (int hh = 0; hh < Hkv; ++hh) {
            float* bk = step.bonus_k.data() + ((size_t)a * Hkv + hh) * D;
            float* bv = step.bonus_v.data() + ((size_t)a * Hkv + hh) * D;
            for (int d = 0; d < D; ++d) {
                bk[d] = rng.uniform_float(-2.5f, 2.5f);
                bv[d] = rng.uniform_float(-2.5f, 2.5f);
            }
        }

        // Byte-exact target signatures from the oracle quantizer.
        for (int i = 0; i < n; ++i) {
            const StvOracleToken tok = sig_helper->quantize_token(
                step.draft_k.data() + (((size_t)a * N + i) * Hkv) * D,
                step.draft_v.data() + (((size_t)a * N + i) * Hkv) * D);
            uint64_t sig = StvOracleState::token_fold(
                kStvOracleFnvBasis, tok, Hkv, D);
            if (!plan.accept[i]) sig ^= kRejectXor;
            step.target_sig[(size_t)a * N + i] = sig;
        }
        for (int i = n; i < N; ++i) {
            SplitMix64 r2(seed ^ (uint64_t)(i * 977 + seq));
            step.target_sig[(size_t)a * N + i] = r2.next_u64();
        }
    }

    return step;
}

// Tree plan helpers.
static TreePlan chain_plan(int n, int accept_upto /* nodes accepted from root */) {
    TreePlan p;
    p.parent.resize(n);
    p.accept.resize(n);
    for (int i = 0; i < n; ++i) {
        p.parent[i] = i == 0 ? -1 : i - 1;
        p.accept[i] = i < accept_upto ? 1 : 0;
    }
    return p;
}

// Forest of `roots` chains each of length `len` (n = roots*len). Chain c is
// accepted to depth dep[c].
static TreePlan forest_plan(int roots, int len, const std::vector<int>& dep) {
    TreePlan p;
    const int n = roots * len;
    p.parent.resize(n);
    p.accept.resize(n);
    for (int c = 0; c < roots; ++c) {
        for (int k = 0; k < len; ++k) {
            const int i = c * len + k;
            p.parent[i] = k == 0 ? -1 : i - 1;
            p.accept[i] = k < dep[c] ? 1 : 0;
        }
    }
    return p;
}

// Binary tree with `n` nodes (heap order), accept mask given explicitly.
static TreePlan binary_plan(int n, const std::vector<uint8_t>& accept) {
    TreePlan p;
    p.parent.resize(n);
    p.accept = accept;
    for (int i = 0; i < n; ++i) {
        p.parent[i] = i == 0 ? -1 : (i - 1) / 2;
    }
    return p;
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

static Scenario make_chain_min_scenario() {
    Scenario sc;
    sc.name = "chain_min_B2";
    sc.spec = make_spec(2, 2, 1, 64, 8, 16, 96, 32);
    StvOracleState helper;
    helper.init(sc.spec);

    for (int s = 0; s < 14; ++s) {
        std::vector<int32_t> active;
        std::vector<TreePlan> plans;
        if (s == 6) {
            // empty step
        } else {
            active.push_back(0);
            plans.push_back(chain_plan(4 + (s % 4), s % 5));
            if (s % 2 == 0) {
                active.push_back(1);
                plans.push_back(chain_plan(1 + (s % 3), (s / 2) % 3));
            }
        }
        sc.steps.push_back(make_step(
            sc.spec, &helper, s, active, plans,
            g_state ^ 0x110000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_quant_adversarial_scenario() {
    Scenario sc;
    sc.name = "quant_e5m2_ties_saturation";
    sc.spec = make_spec(2, 4, 2, 64, 8, 16, 128, 40);
    StvOracleState helper;
    helper.init(sc.spec);

    for (int s = 0; s < 16; ++s) {
        std::vector<int32_t> active;
        std::vector<TreePlan> plans;
        active.push_back(0);
        plans.push_back(chain_plan(3 + (s % 6), (s + 1) % 5));
        if (s % 2 == 1) {
            active.push_back(1);
            plans.push_back(chain_plan(2 + (s % 4), s % 4));
        }
        sc.steps.push_back(make_step(
            sc.spec, &helper, s, active, plans,
            g_state ^ 0x220000ULL ^ (uint64_t)s, true));
    }
    return sc;
}

static Scenario make_forest_ties_scenario() {
    Scenario sc;
    sc.name = "forest_star_depth_ties";
    sc.spec = make_spec(2, 8, 2, 64, 16, 32, 192, 32);
    StvOracleState helper;
    helper.init(sc.spec);

    for (int s = 0; s < 14; ++s) {
        std::vector<int32_t> active;
        std::vector<TreePlan> plans;

        active.push_back(0);
        {
            // 8 chains of 4; two chains accepted to the SAME depth ->
            // smallest node index must win.
            std::vector<int> dep(8, 0);
            dep[(s + 1) % 8] = 2 + (s % 3);
            dep[(s + 5) % 8] = 2 + (s % 3);
            plans.push_back(forest_plan(8, 4, dep));
        }
        if (s % 3 != 2) {
            active.push_back(1);
            // Binary tree with sig-ok nodes under a rejected parent
            // (must NOT be accepted).
            std::vector<uint8_t> acc(15, 0);
            acc[0] = (s % 4 != 0) ? 1 : 0;
            acc[1] = 1;             // accepted iff node 0 accepted
            acc[2] = 0;
            acc[3] = 1;             // depth 3 via 0-1-3
            acc[5] = 1;             // sig ok but parent 2 rejected
            acc[6] = 1;             // sig ok but parent 2 rejected
            plans.push_back(binary_plan(15, acc));
        }
        sc.steps.push_back(make_step(
            sc.spec, &helper, s, active, plans,
            g_state ^ 0x330000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_deep_chain_scenario() {
    Scenario sc;
    sc.name = "deep_chain64_midbreak";
    sc.spec = make_spec(1, 4, 1, 128, 8, 64, 256, 48);
    StvOracleState helper;
    helper.init(sc.spec);

    const int breaks[12] = {9, 0, 64, 17, 3, 33, 1, 25, 50, 5, 12, 2};
    for (int s = 0; s < 12; ++s) {
        std::vector<int32_t> active;
        std::vector<TreePlan> plans;
        active.push_back(0);
        const int n = (s % 3 == 0) ? 64 : 16 + (s % 5) * 8;
        const int upto = std::min(breaks[s], n);
        plans.push_back(chain_plan(n, upto));
        // Cap acceptance so seq_len stays within max_seq_len.
        sc.steps.push_back(make_step(
            sc.spec, &helper, s, active, plans,
            g_state ^ 0x440000ULL ^ (uint64_t)s, s % 4 == 1));
        // seq growth: upto+1 per step; worst 65+... ensure cap below.
    }
    // Re-plan pass: enforce total <= max_seq_len by trimming acceptance.
    // (Total worst-case: sum(min(breaks,n)+1) = manageable; verified below.)
    {
        int total = 0;
        for (int s = 0; s < 12; ++s) {
            const int n = (s % 3 == 0) ? 64 : 16 + (s % 5) * 8;
            total += std::min(breaks[s], n) + 1;
        }
        if (total > sc.spec.max_seq_len) {
            throw std::runtime_error("deep_chain scenario overflows cap");
        }
    }
    return sc;
}

static Scenario make_churn_tight_scenario() {
    Scenario sc;
    // Pool exactly at the guaranteed bound.
    sc.spec = make_spec(4, 4, 2, 64, 8, 32, 128, 4 * (16 + 4));
    sc.name = "churn_tight_pool";
    StvOracleState helper;
    helper.init(sc.spec);

    for (int s = 0; s < 16; ++s) {
        std::vector<int32_t> active;
        std::vector<TreePlan> plans;
        for (int b = 0; b < 4; ++b) {
            if ((s + b) % 2 == 0) {
                active.push_back(b);
                std::vector<int> dep(4, 0);
                dep[(s + b) % 4] = (s + b) % 4;
                plans.push_back(forest_plan(4, 8, dep));
            }
        }
        sc.steps.push_back(make_step(
            sc.spec, &helper, s, active, plans,
            g_state ^ 0x550000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_gqa_wide_scenario() {
    Scenario sc;
    sc.name = "gqa_hq32_hkv8";
    sc.spec = make_spec(2, 32, 8, 64, 32, 16, 160, 16);
    StvOracleState helper;
    helper.init(sc.spec);

    for (int s = 0; s < 12; ++s) {
        std::vector<int32_t> active;
        std::vector<TreePlan> plans;
        active.push_back(s % 2);
        plans.push_back(chain_plan(4 + (s % 8), s % 6));
        sc.steps.push_back(make_step(
            sc.spec, &helper, s, active, plans,
            g_state ^ 0x660000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_reject_all_scenario() {
    Scenario sc;
    sc.name = "reject_all_bonus_only";
    sc.spec = make_spec(3, 4, 2, 64, 16, 16, 96, 3 * (6 + 1));
    StvOracleState helper;
    helper.init(sc.spec);

    for (int s = 0; s < 14; ++s) {
        std::vector<int32_t> active;
        std::vector<TreePlan> plans;
        if (s % 5 == 4) {
            // empty step
        } else {
            for (int b = 0; b < 3; ++b) {
                if ((s + b) % 2 == 0) {
                    active.push_back(b);
                    if (s % 3 == 0) {
                        plans.push_back(chain_plan(1, 0));   // single reject
                    } else if (s % 3 == 1) {
                        plans.push_back(chain_plan(8, 0));   // all reject
                    } else {
                        plans.push_back(chain_plan(1, 1));   // single accept
                    }
                }
            }
        }
        sc.steps.push_back(make_step(
            sc.spec, &helper, s, active, plans,
            g_state ^ 0x770000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_bench_like_scenario() {
    Scenario sc;
    sc.name = "bench_like_small";
    sc.spec = make_spec(4, 16, 4, 128, 16, 32, 320, 4 * (20 + 2));
    StvOracleState helper;
    helper.init(sc.spec);

    const int pattern[8] = {4, 2, 6, 0, 3, 5, 1, 7};
    for (int s = 0; s < 20; ++s) {
        std::vector<int32_t> active;
        std::vector<TreePlan> plans;
        for (int b = 0; b < 4; ++b) {
            active.push_back(b);
            std::vector<int> dep(4, 0);
            dep[(s + b) % 4] = pattern[(s + b) % 8];
            plans.push_back(forest_plan(4, 8, dep));
        }
        sc.steps.push_back(make_step(
            sc.spec, &helper, s, active, plans,
            g_state ^ 0x880000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

// ---------------------------------------------------------------------------

static StepHost permute_step_rows(const StvProblemSpec& spec, const StepHost& src) {
    StepHost dst = src;
    const int A = src.run.active_count;
    if (A == 0) return dst;

    const int N = spec.max_nodes;
    const int Hq = spec.Hq;
    const int Hkv = spec.Hkv;
    const int D = spec.D;

    for (int new_a = 0; new_a < A; ++new_a) {
        const int old_a = A - 1 - new_a;
        dst.active_seq[new_a] = src.active_seq[old_a];
        dst.node_count[new_a] = src.node_count[old_a];

        std::memcpy(dst.parent.data() + (size_t)new_a * N,
                    src.parent.data() + (size_t)old_a * N,
                    sizeof(int32_t) * N);
        std::memcpy(dst.target_sig.data() + (size_t)new_a * N,
                    src.target_sig.data() + (size_t)old_a * N,
                    sizeof(uint64_t) * N);

        const size_t kv_row = (size_t)N * Hkv * D;
        std::memcpy(dst.draft_k.data() + (size_t)new_a * kv_row,
                    src.draft_k.data() + (size_t)old_a * kv_row,
                    sizeof(float) * kv_row);
        std::memcpy(dst.draft_v.data() + (size_t)new_a * kv_row,
                    src.draft_v.data() + (size_t)old_a * kv_row,
                    sizeof(float) * kv_row);

        const size_t q_row = (size_t)N * Hq * D;
        std::memcpy(dst.q.data() + (size_t)new_a * q_row,
                    src.q.data() + (size_t)old_a * q_row,
                    sizeof(float) * q_row);

        const size_t b_row = (size_t)Hkv * D;
        std::memcpy(dst.bonus_k.data() + (size_t)new_a * b_row,
                    src.bonus_k.data() + (size_t)old_a * b_row,
                    sizeof(float) * b_row);
        std::memcpy(dst.bonus_v.data() + (size_t)new_a * b_row,
                    src.bonus_v.data() + (size_t)old_a * b_row,
                    sizeof(float) * b_row);
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

static bool run_step(
    const StvProblemSpec& spec,
    const StepHost& step,
    StvOracleState* oracle,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    StepResult* result,
    std::string* error) {
    const int A = step.run.active_count;
    const int N = spec.max_nodes;
    const int Hq = spec.Hq;
    const int D = spec.D;
    const size_t y_count = std::max<size_t>(1, (size_t)A * N * Hq * D);
    const size_t lse_count = std::max<size_t>(1, (size_t)A * N * Hq);
    const size_t a_count = std::max<size_t>(1, (size_t)A);

    DeviceBuffer<int32_t> d_active;
    DeviceBuffer<int32_t> d_ncount;
    DeviceBuffer<int32_t> d_parent;
    DeviceBuffer<float> d_dk;
    DeviceBuffer<float> d_dv;
    DeviceBuffer<float> d_q;
    DeviceBuffer<uint64_t> d_sig;
    DeviceBuffer<float> d_bk;
    DeviceBuffer<float> d_bv;

    d_active.allocate(step.active_seq.size());
    d_ncount.allocate(step.node_count.size());
    d_parent.allocate(step.parent.size());
    d_dk.allocate(step.draft_k.size());
    d_dv.allocate(step.draft_v.size());
    d_q.allocate(step.q.size());
    d_sig.allocate(step.target_sig.size());
    d_bk.allocate(step.bonus_k.size());
    d_bv.allocate(step.bonus_v.size());

    d_active.upload(step.active_seq);
    d_ncount.upload(step.node_count);
    d_parent.upload(step.parent);
    d_dk.upload(step.draft_k);
    d_dv.upload(step.draft_v);
    d_q.upload(step.q);
    d_sig.upload(step.target_sig);
    d_bk.upload(step.bonus_k);
    d_bv.upload(step.bonus_v);

    GuardedDeviceBuffer<float> d_y;
    GuardedDeviceBuffer<float> d_lse;
    GuardedDeviceBuffer<int32_t> d_tail;
    GuardedDeviceBuffer<int32_t> d_alen;
    GuardedDeviceBuffer<int32_t> d_seq_len;
    GuardedDeviceBuffer<uint64_t> d_kv_hash;
    GuardedDeviceBuffer<uint64_t> d_pcs;
    GuardedDeviceBuffer<int32_t> d_free;
    GuardedDeviceBuffer<int32_t> d_allocs;
    GuardedDeviceBuffer<int32_t> d_frees;

    d_y.allocate(y_count);
    d_lse.allocate(lse_count);
    d_tail.allocate(a_count);
    d_alen.allocate(a_count);
    d_seq_len.allocate((size_t)spec.B);
    d_kv_hash.allocate((size_t)spec.B);
    d_pcs.allocate(1);
    d_free.allocate(1);
    d_allocs.allocate(1);
    d_frees.allocate(1);

    StvInputs inputs = {};
    inputs.active_seq = d_active.ptr;
    inputs.node_count = d_ncount.ptr;
    inputs.parent = d_parent.ptr;
    inputs.draft_k = d_dk.ptr;
    inputs.draft_v = d_dv.ptr;
    inputs.q = d_q.ptr;
    inputs.target_sig = d_sig.ptr;
    inputs.bonus_k = d_bk.ptr;
    inputs.bonus_v = d_bv.ptr;

    StvOutputs outputs = {};
    outputs.y = d_y.ptr;
    outputs.lse = d_lse.ptr;
    outputs.accepted_tail = d_tail.ptr;
    outputs.accepted_len = d_alen.ptr;
    outputs.seq_len = d_seq_len.ptr;
    outputs.kv_hash = d_kv_hash.ptr;
    outputs.page_state_checksum = d_pcs.ptr;
    outputs.free_pages = d_free.ptr;
    outputs.total_allocs = d_allocs.ptr;
    outputs.total_frees = d_frees.ptr;

    CUDA_CHECK(solution_run(
        state, &step.run, &inputs, &outputs,
        workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Inputs unchanged?
    if (d_active.download() != step.active_seq) {
        if (error) *error = "input active_seq modified";
        return false;
    }
    if (d_ncount.download() != step.node_count) {
        if (error) *error = "input node_count modified";
        return false;
    }
    {
        const std::vector<int32_t> gp = d_parent.download();
        const std::vector<uint64_t> gs = d_sig.download();
        const std::vector<float> gk = d_dk.download();
        const std::vector<float> gv = d_dv.download();
        const std::vector<float> gq = d_q.download();
        const std::vector<float> gbk = d_bk.download();
        const std::vector<float> gbv = d_bv.download();
        if (gp != step.parent) { if (error) *error = "input parent modified"; return false; }
        if (gs != step.target_sig) { if (error) *error = "input target_sig modified"; return false; }
        if (std::memcmp(gk.data(), step.draft_k.data(), gk.size() * 4) != 0) {
            if (error) *error = "input draft_k modified"; return false;
        }
        if (std::memcmp(gv.data(), step.draft_v.data(), gv.size() * 4) != 0) {
            if (error) *error = "input draft_v modified"; return false;
        }
        if (std::memcmp(gq.data(), step.q.data(), gq.size() * 4) != 0) {
            if (error) *error = "input q modified"; return false;
        }
        if (std::memcmp(gbk.data(), step.bonus_k.data(), gbk.size() * 4) != 0) {
            if (error) *error = "input bonus_k modified"; return false;
        }
        if (std::memcmp(gbv.data(), step.bonus_v.data(), gbv.size() * 4) != 0) {
            if (error) *error = "input bonus_v modified"; return false;
        }
    }

    if (!d_y.check_guards("y", error)) return false;
    if (!d_lse.check_guards("lse", error)) return false;
    if (!d_tail.check_guards("accepted_tail", error)) return false;
    if (!d_alen.check_guards("accepted_len", error)) return false;
    if (!d_seq_len.check_guards("seq_len", error)) return false;
    if (!d_kv_hash.check_guards("kv_hash", error)) return false;
    if (!d_pcs.check_guards("page_state_checksum", error)) return false;
    if (!d_free.check_guards("free_pages", error)) return false;
    if (!d_allocs.check_guards("total_allocs", error)) return false;
    if (!d_frees.check_guards("total_frees", error)) return false;

    const std::vector<float> h_y = d_y.download_data();
    const std::vector<float> h_lse = d_lse.download_data();
    const std::vector<int32_t> h_tail = d_tail.download_data();
    const std::vector<int32_t> h_alen = d_alen.download_data();
    const std::vector<int32_t> h_seq_len = d_seq_len.download_data();
    const std::vector<uint64_t> h_kv_hash = d_kv_hash.download_data();
    const std::vector<uint64_t> h_pcs = d_pcs.download_data();
    const std::vector<int32_t> h_free = d_free.download_data();
    const std::vector<int32_t> h_allocs = d_allocs.download_data();
    const std::vector<int32_t> h_frees = d_frees.download_data();

    StvHostInputsView host_inputs = {};
    host_inputs.active_seq = step.active_seq.data();
    host_inputs.node_count = step.node_count.data();
    host_inputs.parent = step.parent.data();
    host_inputs.draft_k = step.draft_k.data();
    host_inputs.draft_v = step.draft_v.data();
    host_inputs.q = step.q.data();
    host_inputs.target_sig = step.target_sig.data();
    host_inputs.bonus_k = step.bonus_k.data();
    host_inputs.bonus_v = step.bonus_v.data();

    StvExpected expected;
    oracle->step(step.run, host_inputs, &expected);

    StvHostOutputsView got = {};
    got.y = h_y.data();
    got.lse = h_lse.data();
    got.accepted_tail = h_tail.data();
    got.accepted_len = h_alen.data();
    got.seq_len = h_seq_len.data();
    got.kv_hash = h_kv_hash.data();
    got.page_state_checksum = h_pcs.data();
    got.free_pages = h_free.data();
    got.total_allocs = h_allocs.data();
    got.total_frees = h_frees.data();

    if (!stv_check_all_outputs(step.run, spec, expected, got,
                               step.node_count.data(), error)) {
        return false;
    }

    if (result) {
        result->active_seq.assign(step.active_seq.begin(), step.active_seq.begin() + A);
        result->node_count.assign(step.node_count.begin(), step.node_count.begin() + A);
        result->y.assign(h_y.begin(), h_y.begin() + ((size_t)A * N * Hq * D));
        result->lse.assign(h_lse.begin(), h_lse.begin() + ((size_t)A * N * Hq));
        result->accepted_tail.assign(h_tail.begin(), h_tail.begin() + A);
        result->accepted_len.assign(h_alen.begin(), h_alen.begin() + A);
        result->seq_len = h_seq_len;
        result->kv_hash = h_kv_hash;
        result->page_state_checksum = h_pcs[0];
        result->free_pages = h_free[0];
        result->total_allocs = h_allocs[0];
        result->total_frees = h_frees[0];
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

    StvOracleState oracle;
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
            sc.spec, sc.steps[i], &oracle, state,
            workspace.ptr, workspace_bytes, stream,
            results ? &step_result : nullptr, &step_error);

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

        if (results) results->push_back(step_result);

        if (verbose && (!ok || i + 1 == sc.steps.size())) {
            std::printf(
                "scenario %-44s step %02zu/%02zu active=%d %s%s%s\n",
                sc.name.c_str(), i, sc.steps.size(),
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
    const int N = base.spec.max_nodes;
    const int Hq = base.spec.Hq;
    const int D = base.spec.D;

    if (a.size() != b.size()) {
        if (error) *error = "result step count mismatch";
        return false;
    }

    for (size_t step = 0; step < a.size(); ++step) {
        if (a[step].seq_len != b[step].seq_len ||
            a[step].kv_hash != b[step].kv_hash ||
            a[step].free_pages != b[step].free_pages ||
            a[step].total_allocs != b[step].total_allocs ||
            a[step].total_frees != b[step].total_frees) {
            if (error) {
                std::ostringstream oss;
                oss << "order-invariant global mismatch at step " << step;
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
                    oss << "missing seq " << seq << " at step " << step;
                    *error = oss.str();
                }
                return false;
            }

            if (a[step].accepted_tail[row_a] != b[step].accepted_tail[row_b] ||
                a[step].accepted_len[row_a] != b[step].accepted_len[row_b]) {
                if (error) {
                    std::ostringstream oss;
                    oss << "accepted mismatch at step " << step << " seq " << seq;
                    *error = oss.str();
                }
                return false;
            }

            for (int i = 0; i < N; ++i) {
                for (int hq = 0; hq < Hq; ++hq) {
                    const size_t li_a = ((size_t)row_a * N + i) * Hq + hq;
                    const size_t li_b = ((size_t)row_b * N + i) * Hq + hq;
                    {
                        const float av = a[step].lse[li_a];
                        const float bv = b[step].lse[li_b];
                        const float diff = std::fabs(av - bv);
                        const float tol = STV_LSE_ATOL + STV_LSE_RTOL * std::fabs(av);
                        if (!(diff <= tol)) {
                            if (error) {
                                std::ostringstream oss;
                                oss << "permuted lse mismatch step " << step
                                    << " seq " << seq << " node " << i
                                    << " hq " << hq;
                                *error = oss.str();
                            }
                            return false;
                        }
                    }
                    for (int d = 0; d < D; ++d) {
                        const float av = a[step].y[li_a * D + d];
                        const float bv = b[step].y[li_b * D + d];
                        const float diff = std::fabs(av - bv);
                        const float tol = STV_Y_ATOL + STV_Y_RTOL * std::fabs(av);
                        if (!(diff <= tol)) {
                            if (error) {
                                std::ostringstream oss;
                                oss << "permuted y mismatch step " << step
                                    << " seq " << seq << " node " << i
                                    << " hq " << hq << " d " << d;
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
        scenarios.push_back(make_chain_min_scenario());
        scenarios.push_back(make_quant_adversarial_scenario());
        scenarios.push_back(make_forest_ties_scenario());
        scenarios.push_back(make_deep_chain_scenario());
        scenarios.push_back(make_churn_tight_scenario());
        scenarios.push_back(make_gqa_wide_scenario());
        scenarios.push_back(make_reject_all_scenario());
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
                bool bitexact = true;
                std::string cmp_error;
                for (size_t s = 0; s < base_results.size() && bitexact; ++s) {
                    if (base_results[s].y != repeat_results[s].y ||
                        base_results[s].lse != repeat_results[s].lse ||
                        base_results[s].kv_hash != repeat_results[s].kv_hash ||
                        base_results[s].page_state_checksum !=
                            repeat_results[s].page_state_checksum) {
                        bitexact = false;
                        std::ostringstream oss;
                        oss << "bit-exact replay mismatch at step " << s;
                        cmp_error = oss.str();
                    }
                }
                if (!bitexact) {
                    all_ok = false;
                    std::printf("scenario %-44s deterministic replay FAIL  %s\n",
                                sc.name.c_str(), cmp_error.c_str());
                } else {
                    std::printf("scenario %-44s deterministic replay PASS\n",
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
                    std::printf("scenario %-44s permuted-order replay FAIL  %s\n",
                                sc.name.c_str(), cmp_error.c_str());
                } else {
                    std::printf("scenario %-44s permuted-order replay PASS\n",
                                sc.name.c_str());
                }
            }

            if (!ok_base || !ok_repeat || !ok_permuted) {
                all_ok = false;
                std::printf("scenario %-44s FAIL  %s\n",
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
