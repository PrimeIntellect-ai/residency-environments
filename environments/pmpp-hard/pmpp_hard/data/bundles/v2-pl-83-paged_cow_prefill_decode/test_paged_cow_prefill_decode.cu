// ============================================================================
// file: test_paged_cow_prefill_decode.cu
// ============================================================================

#include "paged_cow_prefill_decode_common.h"
#include "paged_cow_prefill_decode_oracle.hpp"

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

static constexpr uint64_t g_state = 0x3ac57e9d10b48f62ULL;
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
    CpdRunSpec run;
    std::vector<int32_t> active_seq;
    std::vector<int32_t> op_kind;
    std::vector<int32_t> fork_src;
    std::vector<int32_t> new_token_count;
    std::vector<float> new_k;
    std::vector<float> new_v;
    std::vector<float> q;
};

struct Scenario {
    std::string name;
    CpdProblemSpec spec;
    std::vector<StepHost> steps;
};

struct StepResult {
    std::vector<int32_t> active_seq;
    std::vector<int32_t> op_kind;
    std::vector<int32_t> new_token_count;
    std::vector<float> y;
    std::vector<float> lse;
    std::vector<int32_t> seq_len;
    std::vector<uint64_t> kv_hash;
    uint64_t page_state_checksum = 0;
    int32_t free_pages = 0;
    int32_t total_allocs = 0;
    int32_t total_frees = 0;
    int32_t total_forks = 0;
    int32_t total_releases = 0;
};

static CpdProblemSpec make_spec(
    int B, int Hq, int Hkv, int D, int P, int C, int msl, int max_pages) {
    CpdProblemSpec spec = {};
    spec.abi_version = CPD_ABI_VERSION;
    spec.B = B;
    spec.Hq = Hq;
    spec.Hkv = Hkv;
    spec.D = D;
    spec.page_size = P;
    spec.max_chunk = C;
    spec.max_seq_len = msl;
    spec.max_pages = max_pages;
    spec.flags = 0;

    if (!cpd_validate_problem_spec(&spec)) {
        throw std::runtime_error("invalid CpdProblemSpec generated");
    }
    return spec;
}

// Adversarial fp16 values: RNE ties at normal halves, subnormal ties,
// boundary values, signed zeros. |x| <= 1024, nonzero |x| >= 2^-90.
static void craft_f16_vector(int variant, float* x, int D) {
    for (int d = 0; d < D; ++d) x[d] = 0.0f;

    switch (variant % 4) {
        case 0:
            x[0] = 1.00048828125f;      // 1 + 2^-11: tie -> 1.0 (even)
            x[1] = 1.00146484375f;      // 1 + 3*2^-11: tie -> 1 + 2^-9? no:
                                        // between 1+2^-10 and 1+2^-9 -> even
            x[2] = -1.00048828125f;
            x[3] = 2049.0f;             // tie between 2048 and 2050 -> 2048
            x[4] = 2051.0f;             // tie between 2050 and 2052 -> 2052
            x[5] = 1024.0f;             // exact (input bound)
            x[6] = -1024.0f;
            x[7] = 0.0f;
            x[8] = -0.0f;               // 0x8000
            break;
        case 1:
            // Subnormal half territory.
            x[0] = 2.9802322387695312e-08f;   // 2^-25: tie -> 0 (even)
            x[1] = 8.940696716308594e-08f;    // 1.5*2^-24: tie -> 2 units
            x[2] = 5.960464477539063e-08f;    // 2^-24: exactly 1 unit
            x[3] = -1.4901161193847656e-07f;  // -2.5 units: tie -> -2 units
            x[4] = 6.103515625e-05f;          // 2^-14: smallest normal exact
            x[5] = 6.097555160522461e-05f;    // just below 2^-14: rounds up?
            x[6] = 1.0e-9f;                   // < 2^-25 -> 0
            break;
        case 2:
            x[0] = 0.10009765625f;      // near 0.1: rounding exercised
            x[1] = 0.3333333f;
            x[2] = -0.66666667f;
            x[3] = 3.140625f;           // pi-ish, exact half?
            x[4] = 1.0f / 3.0f + 1.0e-4f;
            x[5] = 511.75f;             // exact half at step 0.25? step near
                                        // 512 is 0.5: 511.75 tie -> 512? no:
                                        // 511.75 between 511.5/512 -> 512 even
            break;
        default:
            x[0] = 1.0f;
            x[1] = -2.0f;
            x[2] = 0.5f;
            x[3] = 0.0009765625f;       // 2^-10 exact
            x[4] = -0.0001220703125f;   // 2^-13 exact
            break;
    }
}

struct RowPlan {
    int seq;
    int op;        // CPD_OP_*
    int src;       // fork source (ignored otherwise)
    int ntok;
};

static StepHost make_step(
    const CpdProblemSpec& spec,
    int step_id,
    const std::vector<RowPlan>& rows,
    uint64_t seed,
    bool adversarial) {
    const int A = static_cast<int>(rows.size());
    const int C = spec.max_chunk;
    const int Hq = spec.Hq;
    const int Hkv = spec.Hkv;
    const int D = spec.D;

    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = i + 1; j < rows.size(); ++j) {
            if (rows[i].seq == rows[j].seq) {
                throw std::runtime_error("duplicate active id in step");
            }
        }
    }

    StepHost step;
    step.run = {};
    step.run.abi_version = CPD_ABI_VERSION;
    step.run.active_count = A;
    step.run.step_id = step_id;
    if (!cpd_validate_run_spec(&step.run, &spec)) {
        throw std::runtime_error("invalid CpdRunSpec generated");
    }

    const size_t a_count = A > 0 ? static_cast<size_t>(A) : 1;
    step.active_seq.assign(a_count, 0);
    step.op_kind.assign(a_count, CPD_OP_APPEND);
    step.fork_src.assign(a_count, 0);
    step.new_token_count.assign(a_count, 0);
    step.new_k.assign(a_count * C * Hkv * (size_t)D, 0.0f);
    step.new_v.assign(a_count * C * Hkv * (size_t)D, 0.0f);
    step.q.assign(a_count * C * Hq * (size_t)D, 0.0f);

    for (int a = 0; a < A; ++a) {
        step.active_seq[a] = rows[a].seq;
        step.op_kind[a] = rows[a].op;
        step.fork_src[a] = rows[a].src;
        step.new_token_count[a] = rows[a].ntok;

        SplitMix64 rng(seed ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(rows[a].seq + 1)));

        for (int nt = 0; nt < C; ++nt) {
            const bool pad =
                rows[a].op == CPD_OP_RELEASE || nt >= rows[a].ntok;
            for (int hh = 0; hh < Hkv; ++hh) {
                float* kk = step.new_k.data() +
                    (((size_t)a * C + nt) * Hkv + hh) * D;
                float* vv = step.new_v.data() +
                    (((size_t)a * C + nt) * Hkv + hh) * D;
                for (int d = 0; d < D; ++d) {
                    kk[d] = pad ? rng.uniform_float(-9.0e2f, 9.0e2f)
                                : rng.uniform_float(-2.0f, 2.0f);
                    vv[d] = pad ? rng.uniform_float(-9.0e2f, 9.0e2f)
                                : rng.uniform_float(-2.0f, 2.0f);
                }
                if (!pad && adversarial && ((nt + step_id + hh) % 3 == 0)) {
                    craft_f16_vector(step_id + nt + hh, kk, D);
                    craft_f16_vector(step_id + nt + hh + 1, vv, D);
                }
            }
            for (int hq = 0; hq < Hq; ++hq) {
                float* qq = step.q.data() +
                    (((size_t)a * C + nt) * Hq + hq) * D;
                for (int d = 0; d < D; ++d) {
                    qq[d] = pad ? rng.uniform_float(-9.0e2f, 9.0e2f)
                                : rng.uniform_float(-1.0f, 1.0f);
                }
            }
        }
    }

    return step;
}

// ---------------------------------------------------------------------------
// Scenario builders track live lengths on the host so every schedule
// respects the contract invariants.
// ---------------------------------------------------------------------------

struct SeqTrack {
    std::vector<int> len;
    explicit SeqTrack(int B) : len(B, 0) {}

    void apply(const std::vector<RowPlan>& rows) {
        for (const RowPlan& r : rows) {
            if (r.op == CPD_OP_RELEASE) {
                len[r.seq] = 0;
            }
        }
        for (const RowPlan& r : rows) {
            if (r.op == CPD_OP_FORK_APPEND) {
                len[r.seq] = len[r.src];
            }
        }
        for (const RowPlan& r : rows) {
            if (r.op != CPD_OP_RELEASE) {
                len[r.seq] += r.ntok;
            }
        }
    }
};

static Scenario make_basic_mixed_scenario() {
    Scenario sc;
    sc.name = "basic_mixed_min";
    sc.spec = make_spec(2, 2, 1, 64, 8, 16, 128, 40);
    SeqTrack trk(2);

    for (int s = 0; s < 14; ++s) {
        std::vector<RowPlan> rows;
        if (s == 7) {
            // empty step
        } else if (s < 4) {
            rows.push_back({0, CPD_OP_APPEND, 0, 16});
            rows.push_back({1, CPD_OP_APPEND, 0, 5 + s});
        } else {
            rows.push_back({0, CPD_OP_APPEND, 0, 1});
            if (s % 2 == 0) rows.push_back({1, CPD_OP_APPEND, 0, 2});
        }
        trk.apply(rows);
        if (trk.len[0] > sc.spec.max_seq_len || trk.len[1] > sc.spec.max_seq_len) {
            throw std::runtime_error("basic_mixed overflow");
        }
        sc.steps.push_back(make_step(
            sc.spec, s, rows, g_state ^ 0xA10000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_f16_adversarial_scenario() {
    Scenario sc;
    sc.name = "f16_rne_ties_subnormals";
    sc.spec = make_spec(2, 4, 2, 64, 8, 16, 128, 48);
    SeqTrack trk(2);

    for (int s = 0; s < 14; ++s) {
        std::vector<RowPlan> rows;
        rows.push_back({0, CPD_OP_APPEND, 0, 1 + (s % 8)});
        if (s % 2 == 1) rows.push_back({1, CPD_OP_APPEND, 0, 1 + (s % 4)});
        trk.apply(rows);
        sc.steps.push_back(make_step(
            sc.spec, s, rows, g_state ^ 0xA20000ULL ^ (uint64_t)s, true));
    }
    return sc;
}

static Scenario make_fork_chain_scenario() {
    Scenario sc;
    sc.name = "fork_chain_cow";
    sc.spec = make_spec(4, 4, 2, 64, 8, 16, 192, 140);
    SeqTrack trk(4);

    for (int s = 0; s < 18; ++s) {
        std::vector<RowPlan> rows;
        if (s < 3) {
            rows.push_back({0, CPD_OP_APPEND, 0, 15});  // len 45: partial tails
        } else if (s == 3) {
            rows.push_back({1, CPD_OP_FORK_APPEND, 0, 3});  // fork@45 (45%8=5)
            rows.push_back({0, CPD_OP_APPEND, 0, 1});
        } else if (s == 5) {
            rows.push_back({2, CPD_OP_FORK_APPEND, 1, 0});  // fork-only, no append
            rows.push_back({0, CPD_OP_APPEND, 0, 2});
            rows.push_back({1, CPD_OP_APPEND, 0, 2});
        } else if (s == 8) {
            rows.push_back({3, CPD_OP_FORK_APPEND, 2, 8});  // chained fork
        } else {
            for (int b = 0; b < 4; ++b) {
                if (trk.len[b] > 0 && (s + b) % 2 == 0) {
                    rows.push_back({b, CPD_OP_APPEND, 0, 1 + (b % 3)});
                }
            }
        }
        trk.apply(rows);
        for (int b = 0; b < 4; ++b) {
            if (trk.len[b] > sc.spec.max_seq_len) {
                throw std::runtime_error("fork_chain overflow");
            }
        }
        sc.steps.push_back(make_step(
            sc.spec, s, rows, g_state ^ 0xA30000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_release_recycle_scenario() {
    Scenario sc;
    sc.name = "release_recycle_ids";
    sc.spec = make_spec(4, 4, 1, 64, 8, 16, 96, 60);
    SeqTrack trk(4);

    for (int s = 0; s < 16; ++s) {
        std::vector<RowPlan> rows;
        const int phase = s % 4;
        if (phase == 0) {
            for (int b = 0; b < 3; ++b) {
                if (trk.len[b] == 0) rows.push_back({b, CPD_OP_APPEND, 0, 9 + b});
                else rows.push_back({b, CPD_OP_APPEND, 0, 2});
            }
        } else if (phase == 1) {
            // release one live row, decode others.
            int released = -1;
            for (int b = 0; b < 3; ++b) {
                if (released < 0 && trk.len[b] > 0 && (s / 4) % 3 == b) {
                    rows.push_back({b, CPD_OP_RELEASE, 0, 0});
                    released = b;
                } else if (trk.len[b] > 0) {
                    rows.push_back({b, CPD_OP_APPEND, 0, 1});
                }
            }
        } else if (phase == 2) {
            // fork into id 3 from the lowest live seq (release 3 first if live).
            int src = -1;
            for (int b = 0; b < 3; ++b) {
                if (trk.len[b] > 0) { src = b; break; }
            }
            if (trk.len[3] > 0) {
                rows.push_back({3, CPD_OP_RELEASE, 0, 0});
                // fork next step instead (target must be empty pre-step;
                // release happens phase 2 < fork phase 3, same step is legal
                // ONLY for different rows -- here same row, so defer).
            } else if (src >= 0) {
                rows.push_back({3, CPD_OP_FORK_APPEND, src, 2});
            }
            for (int b = 0; b < 3; ++b) {
                if (trk.len[b] > 0 && b != src) {
                    rows.push_back({b, CPD_OP_APPEND, 0, 1});
                }
            }
        } else {
            for (int b = 0; b < 4; ++b) {
                if (trk.len[b] > 0) rows.push_back({b, CPD_OP_APPEND, 0, 1});
            }
        }
        trk.apply(rows);
        for (int b = 0; b < 4; ++b) {
            if (trk.len[b] > sc.spec.max_seq_len) {
                throw std::runtime_error("release_recycle overflow");
            }
        }
        sc.steps.push_back(make_step(
            sc.spec, s, rows, g_state ^ 0xA40000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_cow_boundary_scenario() {
    Scenario sc;
    sc.name = "cow_page_boundaries";
    sc.spec = make_spec(3, 8, 2, 64, 16, 32, 160, 60);
    SeqTrack trk(3);

    for (int s = 0; s < 14; ++s) {
        std::vector<RowPlan> rows;
        if (s == 0) {
            rows.push_back({0, CPD_OP_APPEND, 0, 32});  // len 32 = 2 full pages
        } else if (s == 1) {
            rows.push_back({1, CPD_OP_FORK_APPEND, 0, 1});  // fork at len%P==0:
                                                            // NO tail copy
        } else if (s == 2) {
            rows.push_back({0, CPD_OP_APPEND, 0, 15});      // len 47 = P-1 rem
        } else if (s == 3) {
            rows.push_back({2, CPD_OP_FORK_APPEND, 0, 1});  // fork at rem 15,
                                                            // copy then fill
        } else if (s == 6) {
            rows.push_back({2, CPD_OP_RELEASE, 0, 0});
            rows.push_back({0, CPD_OP_APPEND, 0, 1});
        } else if (s == 7) {
            rows.push_back({2, CPD_OP_FORK_APPEND, 1, 16});
        } else {
            for (int b = 0; b < 3; ++b) {
                if (trk.len[b] > 0 && (s + b) % 2 == 0) {
                    rows.push_back({b, CPD_OP_APPEND, 0, 1 + (b == 0 ? 2 : 0)});
                }
            }
        }
        trk.apply(rows);
        sc.steps.push_back(make_step(
            sc.spec, s, rows, g_state ^ 0xA50000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_gqa_chunk_max_scenario() {
    Scenario sc;
    sc.name = "gqa_hq32_chunk64";
    sc.spec = make_spec(2, 32, 4, 128, 16, 64, 320, 60);
    SeqTrack trk(2);

    for (int s = 0; s < 8; ++s) {
        std::vector<RowPlan> rows;
        if (s < 3) {
            rows.push_back({0, CPD_OP_APPEND, 0, 64});
            rows.push_back({1, CPD_OP_APPEND, 0, 32});
        } else {
            rows.push_back({0, CPD_OP_APPEND, 0, 1});
            rows.push_back({1, CPD_OP_APPEND, 0, s % 3 == 0 ? 16 : 1});
        }
        trk.apply(rows);
        if (trk.len[0] > 320 || trk.len[1] > 320) {
            throw std::runtime_error("gqa_chunk overflow");
        }
        sc.steps.push_back(make_step(
            sc.spec, s, rows, g_state ^ 0xA60000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_decode_heavy_scenario() {
    Scenario sc;
    sc.name = "decode_heavy_empty_steps";
    sc.spec = make_spec(6, 4, 2, 64, 32, 16, 128, 40);
    SeqTrack trk(6);

    for (int s = 0; s < 16; ++s) {
        std::vector<RowPlan> rows;
        if (s % 6 == 5) {
            // empty step
        } else if (s == 0) {
            for (int b = 0; b < 6; ++b) rows.push_back({b, CPD_OP_APPEND, 0, 4 + b});
        } else if (s == 9) {
            rows.push_back({5, CPD_OP_RELEASE, 0, 0});
            for (int b = 0; b < 5; ++b) rows.push_back({b, CPD_OP_APPEND, 0, 1});
        } else if (s == 11) {
            rows.push_back({5, CPD_OP_FORK_APPEND, 2, 1});
            for (int b = 0; b < 3; ++b) rows.push_back({b, CPD_OP_APPEND, 0, 1});
        } else {
            for (int b = 0; b < 6; ++b) {
                if (trk.len[b] > 0 && (s + b) % 3 != 0) {
                    rows.push_back({b, CPD_OP_APPEND, 0, 1});
                }
            }
        }
        trk.apply(rows);
        sc.steps.push_back(make_step(
            sc.spec, s, rows, g_state ^ 0xA70000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

static Scenario make_bench_like_scenario() {
    Scenario sc;
    sc.name = "bench_like_small";
    sc.spec = make_spec(6, 8, 4, 128, 16, 32, 256, 120);
    SeqTrack trk(6);

    for (int s = 0; s < 14; ++s) {
        std::vector<RowPlan> rows;
        if (s < 3) {
            for (int b = 0; b < 4; ++b) rows.push_back({b, CPD_OP_APPEND, 0, 32});
        } else if (s == 5) {
            rows.push_back({4, CPD_OP_FORK_APPEND, 1, 4});
            for (int b = 0; b < 4; ++b) rows.push_back({b, CPD_OP_APPEND, 0, 1});
        } else if (s == 9) {
            rows.push_back({4, CPD_OP_RELEASE, 0, 0});
            rows.push_back({5, CPD_OP_FORK_APPEND, 2, 8});
            for (int b = 0; b < 4; ++b) rows.push_back({b, CPD_OP_APPEND, 0, 1});
        } else if (s == 12) {
            rows.push_back({0, CPD_OP_APPEND, 0, 32});
            for (int b = 1; b < 4; ++b) rows.push_back({b, CPD_OP_APPEND, 0, 1});
            if (trk.len[5] > 0) rows.push_back({5, CPD_OP_APPEND, 0, 1});
        } else {
            for (int b = 0; b < 6; ++b) {
                if (trk.len[b] > 0) rows.push_back({b, CPD_OP_APPEND, 0, 1});
            }
        }
        trk.apply(rows);
        for (int b = 0; b < 6; ++b) {
            if (trk.len[b] > sc.spec.max_seq_len) {
                throw std::runtime_error("bench_like overflow");
            }
        }
        sc.steps.push_back(make_step(
            sc.spec, s, rows, g_state ^ 0xA80000ULL ^ (uint64_t)s, false));
    }
    return sc;
}

// ---------------------------------------------------------------------------

static StepHost permute_step_rows(const CpdProblemSpec& spec, const StepHost& src) {
    StepHost dst = src;
    const int A = src.run.active_count;
    if (A == 0) return dst;

    const int C = spec.max_chunk;
    const int Hq = spec.Hq;
    const int Hkv = spec.Hkv;
    const int D = spec.D;

    for (int new_a = 0; new_a < A; ++new_a) {
        const int old_a = A - 1 - new_a;
        dst.active_seq[new_a] = src.active_seq[old_a];
        dst.op_kind[new_a] = src.op_kind[old_a];
        dst.fork_src[new_a] = src.fork_src[old_a];
        dst.new_token_count[new_a] = src.new_token_count[old_a];

        const size_t kv_row = (size_t)C * Hkv * D;
        std::memcpy(dst.new_k.data() + (size_t)new_a * kv_row,
                    src.new_k.data() + (size_t)old_a * kv_row,
                    sizeof(float) * kv_row);
        std::memcpy(dst.new_v.data() + (size_t)new_a * kv_row,
                    src.new_v.data() + (size_t)old_a * kv_row,
                    sizeof(float) * kv_row);

        const size_t q_row = (size_t)C * Hq * D;
        std::memcpy(dst.q.data() + (size_t)new_a * q_row,
                    src.q.data() + (size_t)old_a * q_row,
                    sizeof(float) * q_row);
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
    const CpdProblemSpec& spec,
    const StepHost& step,
    CpdOracleState* oracle,
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    StepResult* result,
    std::string* error) {
    const int A = step.run.active_count;
    const int C = spec.max_chunk;
    const int Hq = spec.Hq;
    const int D = spec.D;
    const size_t y_count = std::max<size_t>(1, (size_t)A * C * Hq * D);
    const size_t lse_count = std::max<size_t>(1, (size_t)A * C * Hq);

    DeviceBuffer<int32_t> d_active;
    DeviceBuffer<int32_t> d_op;
    DeviceBuffer<int32_t> d_src;
    DeviceBuffer<int32_t> d_cnt;
    DeviceBuffer<float> d_nk;
    DeviceBuffer<float> d_nv;
    DeviceBuffer<float> d_q;

    d_active.allocate(step.active_seq.size());
    d_op.allocate(step.op_kind.size());
    d_src.allocate(step.fork_src.size());
    d_cnt.allocate(step.new_token_count.size());
    d_nk.allocate(step.new_k.size());
    d_nv.allocate(step.new_v.size());
    d_q.allocate(step.q.size());

    d_active.upload(step.active_seq);
    d_op.upload(step.op_kind);
    d_src.upload(step.fork_src);
    d_cnt.upload(step.new_token_count);
    d_nk.upload(step.new_k);
    d_nv.upload(step.new_v);
    d_q.upload(step.q);

    GuardedDeviceBuffer<float> d_y;
    GuardedDeviceBuffer<float> d_lse;
    GuardedDeviceBuffer<int32_t> d_seq_len;
    GuardedDeviceBuffer<uint64_t> d_kv_hash;
    GuardedDeviceBuffer<uint64_t> d_pcs;
    GuardedDeviceBuffer<int32_t> d_free;
    GuardedDeviceBuffer<int32_t> d_allocs;
    GuardedDeviceBuffer<int32_t> d_frees;
    GuardedDeviceBuffer<int32_t> d_forks;
    GuardedDeviceBuffer<int32_t> d_rels;

    d_y.allocate(y_count);
    d_lse.allocate(lse_count);
    d_seq_len.allocate((size_t)spec.B);
    d_kv_hash.allocate((size_t)spec.B);
    d_pcs.allocate(1);
    d_free.allocate(1);
    d_allocs.allocate(1);
    d_frees.allocate(1);
    d_forks.allocate(1);
    d_rels.allocate(1);

    CpdInputs inputs = {};
    inputs.active_seq = d_active.ptr;
    inputs.op_kind = d_op.ptr;
    inputs.fork_src = d_src.ptr;
    inputs.new_token_count = d_cnt.ptr;
    inputs.new_k = d_nk.ptr;
    inputs.new_v = d_nv.ptr;
    inputs.q = d_q.ptr;

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

    CUDA_CHECK(solution_run(
        state, &step.run, &inputs, &outputs,
        workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (d_active.download() != step.active_seq) {
        if (error) *error = "input active_seq modified";
        return false;
    }
    if (d_op.download() != step.op_kind) {
        if (error) *error = "input op_kind modified";
        return false;
    }
    if (d_src.download() != step.fork_src) {
        if (error) *error = "input fork_src modified";
        return false;
    }
    if (d_cnt.download() != step.new_token_count) {
        if (error) *error = "input new_token_count modified";
        return false;
    }
    {
        const std::vector<float> gk = d_nk.download();
        const std::vector<float> gv = d_nv.download();
        const std::vector<float> gq = d_q.download();
        if (std::memcmp(gk.data(), step.new_k.data(), gk.size() * 4) != 0) {
            if (error) *error = "input new_k modified"; return false;
        }
        if (std::memcmp(gv.data(), step.new_v.data(), gv.size() * 4) != 0) {
            if (error) *error = "input new_v modified"; return false;
        }
        if (std::memcmp(gq.data(), step.q.data(), gq.size() * 4) != 0) {
            if (error) *error = "input q modified"; return false;
        }
    }

    if (!d_y.check_guards("y", error)) return false;
    if (!d_lse.check_guards("lse", error)) return false;
    if (!d_seq_len.check_guards("seq_len", error)) return false;
    if (!d_kv_hash.check_guards("kv_hash", error)) return false;
    if (!d_pcs.check_guards("page_state_checksum", error)) return false;
    if (!d_free.check_guards("free_pages", error)) return false;
    if (!d_allocs.check_guards("total_allocs", error)) return false;
    if (!d_frees.check_guards("total_frees", error)) return false;
    if (!d_forks.check_guards("total_forks", error)) return false;
    if (!d_rels.check_guards("total_releases", error)) return false;

    const std::vector<float> h_y = d_y.download_data();
    const std::vector<float> h_lse = d_lse.download_data();
    const std::vector<int32_t> h_seq_len = d_seq_len.download_data();
    const std::vector<uint64_t> h_kv_hash = d_kv_hash.download_data();
    const std::vector<uint64_t> h_pcs = d_pcs.download_data();
    const std::vector<int32_t> h_free = d_free.download_data();
    const std::vector<int32_t> h_allocs = d_allocs.download_data();
    const std::vector<int32_t> h_frees = d_frees.download_data();
    const std::vector<int32_t> h_forks = d_forks.download_data();
    const std::vector<int32_t> h_rels = d_rels.download_data();

    CpdHostInputsView host_inputs = {};
    host_inputs.active_seq = step.active_seq.data();
    host_inputs.op_kind = step.op_kind.data();
    host_inputs.fork_src = step.fork_src.data();
    host_inputs.new_token_count = step.new_token_count.data();
    host_inputs.new_k = step.new_k.data();
    host_inputs.new_v = step.new_v.data();
    host_inputs.q = step.q.data();

    CpdExpected expected;
    oracle->step(step.run, host_inputs, &expected);

    CpdHostOutputsView got = {};
    got.y = h_y.data();
    got.lse = h_lse.data();
    got.seq_len = h_seq_len.data();
    got.kv_hash = h_kv_hash.data();
    got.page_state_checksum = h_pcs.data();
    got.free_pages = h_free.data();
    got.total_allocs = h_allocs.data();
    got.total_frees = h_frees.data();
    got.total_forks = h_forks.data();
    got.total_releases = h_rels.data();

    if (!cpd_check_all_outputs(step.run, spec, expected, got,
                               step.op_kind.data(),
                               step.new_token_count.data(), error)) {
        return false;
    }

    if (result) {
        result->active_seq.assign(step.active_seq.begin(), step.active_seq.begin() + A);
        result->op_kind.assign(step.op_kind.begin(), step.op_kind.begin() + A);
        result->new_token_count.assign(
            step.new_token_count.begin(), step.new_token_count.begin() + A);
        result->y.assign(h_y.begin(), h_y.begin() + ((size_t)A * C * Hq * D));
        result->lse.assign(h_lse.begin(), h_lse.begin() + ((size_t)A * C * Hq));
        result->seq_len = h_seq_len;
        result->kv_hash = h_kv_hash;
        result->page_state_checksum = h_pcs[0];
        result->free_pages = h_free[0];
        result->total_allocs = h_allocs[0];
        result->total_frees = h_frees[0];
        result->total_forks = h_forks[0];
        result->total_releases = h_rels[0];
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

    CpdOracleState oracle;
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
    const int C = base.spec.max_chunk;
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
            a[step].total_frees != b[step].total_frees ||
            a[step].total_forks != b[step].total_forks ||
            a[step].total_releases != b[step].total_releases) {
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

            for (int nt = 0; nt < C; ++nt) {
                for (int hq = 0; hq < Hq; ++hq) {
                    const size_t li_a = ((size_t)row_a * C + nt) * Hq + hq;
                    const size_t li_b = ((size_t)row_b * C + nt) * Hq + hq;
                    {
                        const float av = a[step].lse[li_a];
                        const float bv = b[step].lse[li_b];
                        const float diff = std::fabs(av - bv);
                        const float tol = CPD_LSE_ATOL + CPD_LSE_RTOL * std::fabs(av);
                        if (!(diff <= tol)) {
                            if (error) {
                                std::ostringstream oss;
                                oss << "permuted lse mismatch step " << step
                                    << " seq " << seq << " nt " << nt
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
                        const float tol = CPD_Y_ATOL + CPD_Y_RTOL * std::fabs(av);
                        if (!(diff <= tol)) {
                            if (error) {
                                std::ostringstream oss;
                                oss << "permuted y mismatch step " << step
                                    << " seq " << seq << " nt " << nt
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
        scenarios.push_back(make_basic_mixed_scenario());
        scenarios.push_back(make_f16_adversarial_scenario());
        scenarios.push_back(make_fork_chain_scenario());
        scenarios.push_back(make_release_recycle_scenario());
        scenarios.push_back(make_cow_boundary_scenario());
        scenarios.push_back(make_gqa_chunk_max_scenario());
        scenarios.push_back(make_decode_heavy_scenario());
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
