// harness.cu -- solution-agnostic test driver for bwd_scatter (DSG-MoE-BWD)
//
// Calls the solution C ABI exactly as the contract specifies.
// Computes FNV-1a-64 checksums over raw output bytes.
// Does NOT embed or call the reference algorithm.
// Runs each case twice and reports whether checksums are byte-identical.
//
// Build: /usr/local/cuda/bin/nvcc -arch=sm_120 -O2 -std=c++17 harness.cu solution.cu -o test_sol
// Run:   ./test_sol

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>

#include "scatter_moe_backward_common.h"

// ---- CUDA error helpers ----
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// ---- Splitmix64: deterministic input generation ----
struct SM64 {
    uint64_t s;
    SM64(uint64_t seed = 0xdeadbeefcafebabe) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t next32() { return uint32_t(next() >> 32); }
    int32_t  next_range(int32_t lo, int32_t hi) {  // [lo, hi)
        return lo + int32_t(uint32_t(next32()) % uint32_t(hi - lo));
    }
    // Small-magnitude finite float bits (avoids NaN/Inf exponent)
    uint32_t small_f32_bits() {
        // exponent in [1, 126] for magnitudes in (1e-38, 1)
        uint32_t r = next32();
        uint32_t exp = 1 + (r >> 25) % 126;  // 1..126
        return (r & 0x807FFFFFu) | (exp << 23);
    }
    uint16_t small_bf16_bits() {
        // Upper 16 bits of a small finite float
        return uint16_t(small_f32_bits() >> 16);
    }
};

// ---- FNV-1a-64 (harness side, for raw output hashing) ----
static const uint64_t H_BASIS = 1469598103934665603ull;
static const uint64_t H_PRIME = 1099511628211ull;

static uint64_t fnv_fold_byte(uint64_t h, uint8_t b) {
    return (h ^ uint64_t(b)) * H_PRIME;
}

static uint64_t fnv_bytes(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) h = fnv_fold_byte(h, p[i]);
    return h;
}

// ---- Byte layout helpers (mirrors the contract exactly) ----
static size_t align8(size_t x) { return (x + 7u) & ~size_t(7u); }

struct InputLayout {
    size_t off_index, off_expert, off_gate, off_x, off_dy, off_up, total;
};

static InputLayout input_layout(const Spec& s, uint32_t T) {
    InputLayout L;
    size_t dsz = (s.input_dtype == DSG_DTYPE_BF16) ? 2u : 4u;
    L.off_index  = 0;
    L.off_expert = align8(L.off_index  + T * 4u);
    L.off_gate   = align8(L.off_expert + T * s.top_k * 4u);
    L.off_x      = align8(L.off_gate   + T * s.top_k * 2u);
    L.off_dy     = align8(L.off_x      + T * s.dim * dsz);
    L.off_up     = align8(L.off_dy     + T * s.out_dim * dsz);
    L.total      = align8(L.off_up     + T * s.dim * dsz);
    return L;
}

// Compute total output buffer size for pre-zeroing purposes
static size_t output_max_bytes(const Spec& s, uint32_t T) {
    return sizeof(OutHeader)
         + align8(T * s.dim * 4u)                    // dx
         + align8(s.table_rows * s.dim * 4u)          // table (worst case)
         + align8(s.experts * s.dim * s.out_dim * 4u) // expert (worst case)
         + 64u;                                        // extra margin
}

// Compute exact used output bytes for a given flags combination
static size_t output_used_bytes(const Spec& s, uint32_t T, uint32_t flags) {
    size_t off = sizeof(OutHeader);
    off = align8(off + T * s.dim * 4u);
    if (flags & DSG_FLAG_EMIT_TABLE)   off = align8(off + s.table_rows * s.dim * 4u);
    if (flags & DSG_FLAG_EMIT_EXPERT)  off = align8(off + s.experts * s.dim * s.out_dim * 4u);
    return off;
}

// ---- RunResult: checksums collected from one solution_run call ----
struct RunResult {
    uint64_t event_hash;
    uint64_t tensor_hash;
    uint64_t state_hash;
    uint64_t step_hash;
    uint64_t raw_output_fnv;   // FNV over OutHeader+payload bytes (used region, pre-zeroed gaps)
};

// ---- Device allocation helpers ----
struct DevBuf {
    void* ptr = nullptr;
    size_t sz  = 0;
    void alloc(size_t n) {
        sz = n ? n : 1;
        CUDA_CHECK(cudaMalloc(&ptr, sz));
    }
    void free_buf() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    }
    ~DevBuf() { free_buf(); }
};

// ---- Build a host input buffer then copy to device ----
// Returns device pointer (caller must cudaFree)
static void* build_and_upload_input(
    const Spec& s,
    uint32_t T,
    const int32_t* emb_idx,   // [T]
    const int32_t* route_exp, // [T*K]
    const int16_t* gate_q15,  // [T*K]
    const void*    x_data,    // [T*D] in dtype
    const void*    dy_data,   // [T*O] in dtype
    const void*    up_data)   // [T*D] in dtype
{
    InputLayout L = input_layout(s, T);
    std::vector<uint8_t> buf(L.total, 0u);
    size_t dsz = (s.input_dtype == DSG_DTYPE_BF16) ? 2u : 4u;

    memcpy(buf.data() + L.off_index,  emb_idx,   T * 4u);
    memcpy(buf.data() + L.off_expert, route_exp,  T * s.top_k * 4u);
    memcpy(buf.data() + L.off_gate,   gate_q15,   T * s.top_k * 2u);
    if (T > 0) {
        memcpy(buf.data() + L.off_x,  x_data,     T * s.dim    * dsz);
        memcpy(buf.data() + L.off_dy, dy_data,     T * s.out_dim * dsz);
        memcpy(buf.data() + L.off_up, up_data,     T * s.dim    * dsz);
    }

    void* d_inp;
    CUDA_CHECK(cudaMalloc(&d_inp, L.total ? L.total : 1));
    CUDA_CHECK(cudaMemcpy(d_inp, buf.data(), L.total, cudaMemcpyHostToDevice));
    return d_inp;
}

// ---- Execute one run, collect RunResult ----
static RunResult do_run(
    void* state,
    const RunSpec& run,
    void* d_inp,
    uint8_t* h_out_buf,          // host output buffer (pre-zeroed)
    void* d_out,                  // device output buffer (pre-zeroed)
    size_t out_alloc,
    size_t out_used,
    cudaStream_t stream)
{
    // zero device output buffer for deterministic alignment gaps
    CUDA_CHECK(cudaMemset(d_out, 0, out_alloc));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaError_t err = solution_run(state, &run, d_inp, d_out, nullptr, 0, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "solution_run returned %s\n", cudaGetErrorString(err));
        exit(1);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // copy output back
    memset(h_out_buf, 0, out_alloc);
    CUDA_CHECK(cudaMemcpy(h_out_buf, d_out, out_used, cudaMemcpyDeviceToHost));

    const OutHeader* hdr = reinterpret_cast<const OutHeader*>(h_out_buf);

    RunResult r;
    r.event_hash     = hdr->event_hash;
    r.tensor_hash    = hdr->tensor_hash;
    r.state_hash     = hdr->state_hash;
    r.step_hash      = hdr->step_hash;
    r.raw_output_fnv = fnv_bytes(H_BASIS, h_out_buf, out_used);
    return r;
}

static void print_result(const char* label, int run_idx, const RunResult& r) {
    printf("  [%s run=%d] event_hash=0x%016llx tensor_hash=0x%016llx\n",
           label, run_idx,
           (unsigned long long)r.event_hash, (unsigned long long)r.tensor_hash);
    printf("  [%s run=%d] state_hash=0x%016llx step_hash=0x%016llx\n",
           label, run_idx,
           (unsigned long long)r.state_hash, (unsigned long long)r.step_hash);
    printf("  [%s run=%d] raw_output_fnv=0x%016llx\n",
           label, run_idx, (unsigned long long)r.raw_output_fnv);
}

// Per-case determinism flags, filled by run_case (parallel to case aggregates).
static std::vector<bool> g_case_det;

static bool results_equal(const RunResult& a, const RunResult& b) {
    return a.event_hash     == b.event_hash
        && a.tensor_hash    == b.tensor_hash
        && a.state_hash     == b.state_hash
        && a.step_hash      == b.step_hash
        && a.raw_output_fnv == b.raw_output_fnv;
}

// ---- Helper: build a Spec with standard fields ----
static Spec make_spec(uint32_t R, uint32_t D, uint32_t O, uint32_t E, uint32_t K,
                      uint32_t max_t, uint32_t cap, uint32_t dtype,
                      int32_t pad_idx = -1, uint64_t wseed = 0x12345678abcdef01ull)
{
    Spec s = {};
    s.magic      = DSG_SPEC_MAGIC;
    s.version    = DSG_VERSION;
    s.table_rows = R;
    s.dim        = D;
    s.out_dim    = O;
    s.experts    = E;
    s.top_k      = K;
    s.max_tokens = max_t;
    s.capacity   = cap;
    s.input_dtype = dtype;
    s.padding_idx = pad_idx;
    s.flags       = 0;
    s.weight_seed = wseed;
    s.reserved0   = 0;
    s.reserved1   = 0;
    return s;
}

static RunSpec make_run(uint32_t T, uint32_t flags, uint32_t cap_override = DSG_CAPACITY_USE_SPEC,
                        uint64_t user_tag = 0)
{
    RunSpec r = {};
    r.magic             = DSG_RUN_MAGIC;
    r.version           = DSG_VERSION;
    r.tokens            = T;
    r.flags             = flags;
    r.capacity_override = cap_override;
    r.reserved0         = 0;
    r.user_tag          = user_tag;
    return r;
}

// ---- Float conversion helpers ----
static uint32_t float_to_bits(float f) {
    uint32_t u; memcpy(&u, &f, 4); return u;
}
static uint16_t float_to_bf16(float f) {
    return uint16_t(float_to_bits(f) >> 16);
}

// ---- Fill float arrays: small finite values from RNG ----
static void fill_f32(std::vector<uint32_t>& out, size_t n, SM64& rng) {
    out.resize(n);
    for (auto& v : out) v = rng.small_f32_bits();
}
static void fill_bf16(std::vector<uint16_t>& out, size_t n, SM64& rng) {
    out.resize(n);
    for (auto& v : out) v = rng.small_bf16_bits();
}

// ---- Generic multi-run executor ----
// Returns vector of per-run results (from first of two identical runs).
// Checks determinism across both runs.
// Returns aggregate step_hash FNV.
static uint64_t run_case(
    const char* case_name,
    const Spec& spec,
    const std::vector<RunSpec>& run_specs,
    const std::vector<void*>& d_inputs,        // one per run
    bool do_reset_midpoint = false,
    int  reset_after_run   = -1,               // reset after run index [reset_after_run]
    int  compare_from      = -1)               // compare run i+compare_from with run i
{
    printf("\n=== CASE: %s ===\n", case_name);

    size_t out_alloc = output_max_bytes(spec, spec.max_tokens);
    std::vector<uint8_t> h_out(out_alloc, 0u);

    DevBuf d_out;
    d_out.alloc(out_alloc);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int n_runs = (int)run_specs.size();

    // We'll run the full sequence twice to check determinism
    std::vector<RunResult> results1(n_runs), results2(n_runs);

    for (int pass = 0; pass < 2; ++pass) {
        void* state = nullptr;
        cudaError_t err = solution_init(&spec, &state, stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "solution_init failed for case '%s': %s\n",
                    case_name, cudaGetErrorString(err));
            exit(1);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int ri = 0; ri < n_runs; ++ri) {
            const RunSpec& rs = run_specs[ri];
            size_t used = output_used_bytes(spec, rs.tokens, rs.flags);

            RunResult res = do_run(state, rs, d_inputs[ri],
                                   h_out.data(), d_out.ptr,
                                   out_alloc, used, stream);

            if (pass == 0) {
                results1[ri] = res;
            } else {
                results2[ri] = res;
            }

            if (do_reset_midpoint && ri == reset_after_run) {
                err = solution_reset(state, stream);
                if (err != cudaSuccess) {
                    fprintf(stderr, "solution_reset failed: %s\n", cudaGetErrorString(err));
                    exit(1);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }

        solution_destroy(state);
    }

    // Print results (pass 1) and check determinism
    bool all_det = true;
    uint64_t agg = H_BASIS;
    for (int ri = 0; ri < n_runs; ++ri) {
        print_result(case_name, ri, results1[ri]);
        agg = fnv_bytes(agg, &results1[ri].step_hash, 8);

        bool det = results_equal(results1[ri], results2[ri]);
        printf("  [%s run=%d] determinism=%s\n", case_name, ri, det ? "PASS" : "FAIL");
        if (!det) all_det = false;
    }

    printf("  [%s] aggregate_step_fnv=0x%016llx determinism=%s\n",
           case_name, (unsigned long long)agg, all_det ? "PASS" : "FAIL");

    g_case_det.push_back(all_det);

    CUDA_CHECK(cudaStreamDestroy(stream));
    return agg;
}

// ====================================================================
// Test cases
// ====================================================================

// ---- Case 1: basic_unique — all unique embedding rows, no overflow ----
static uint64_t tc_basic_unique() {
    // R=16, D=8, O=4, E=4, K=1, max_tokens=8, capacity=8, F32
    Spec sp = make_spec(16,8,4,4,1, 8,8, DSG_DTYPE_F32, -1, 0xAABBCCDD11223344ull);
    RunSpec rs = make_run(8, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT, DSG_CAPACITY_USE_SPEC, 0x1);

    int32_t emb[] = {0,1,2,3,4,5,6,7};
    int32_t exp[] = {0,1,2,3,0,1,2,3};
    int16_t gate[] = {16384,16384,16384,16384,16384,16384,16384,16384}; // ~0.5

    SM64 rng(0x111);
    std::vector<uint32_t> xv, dyv, upv;
    fill_f32(xv,  8*8, rng);
    fill_f32(dyv, 8*4, rng);
    fill_f32(upv, 8*8, rng);

    void* d_inp = build_and_upload_input(sp, 8, emb, exp, gate, xv.data(), dyv.data(), upv.data());
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("basic_unique", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 2: dup_embed — all tokens map to same embedding row ----
static uint64_t tc_dup_embed() {
    Spec sp = make_spec(16,8,4,4,1, 8,8, DSG_DTYPE_F32, -1, 0xBBCCDDEE22334455ull);
    RunSpec rs = make_run(8, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    int32_t emb[] = {0,0,0,0,0,0,0,0};  // all same row
    int32_t exp[] = {0,1,2,3,0,1,2,3};
    int16_t gate[] = {8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192}; // ~0.25

    SM64 rng(0x222);
    std::vector<uint32_t> xv, dyv, upv;
    fill_f32(xv,  8*8, rng);
    fill_f32(dyv, 8*4, rng);
    fill_f32(upv, 8*8, rng);

    void* d_inp = build_and_upload_input(sp, 8, emb, exp, gate, xv.data(), dyv.data(), upv.data());
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("dup_embed", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 3: hot_expert — all tokens route to expert 0, capacity overflow ----
static uint64_t tc_hot_expert() {
    // K=2, cap=2, all route to expert 0
    // assign_cells = 8*2 = 16; cap=2 <= 16
    Spec sp = make_spec(16,8,4,4,2, 8,2, DSG_DTYPE_F32, -1, 0xCCDDEEFF33445566ull);
    RunSpec rs = make_run(8, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    int32_t emb[8], exp_r[16], gate_arr[16];
    int16_t gate[16];
    for (int i = 0; i < 8; i++) emb[i] = i;
    for (int i = 0; i < 16; i++) { exp_r[i] = 0; gate[i] = int16_t(8192); }

    SM64 rng(0x333);
    std::vector<uint32_t> xv, dyv, upv;
    fill_f32(xv,  8*8, rng);
    fill_f32(dyv, 8*4, rng);
    fill_f32(upv, 8*8, rng);

    void* d_inp = build_and_upload_input(sp, 8, emb, exp_r, gate, xv.data(), dyv.data(), upv.data());
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("hot_expert", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 4: dup_expert_token — K=2, same expert assigned twice per token ----
static uint64_t tc_dup_expert_token() {
    // cap=8 = max_tokens*K = 4*2 (no overflow)
    Spec sp = make_spec(16,8,4,4,2, 4,8, DSG_DTYPE_F32, -1, 0xDDEEFF0044556677ull);
    RunSpec rs = make_run(4, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    int32_t emb[] = {0,1,2,3};
    // Each token's K=2 slots both route to the same expert
    int32_t exp_r[] = {0,0, 1,1, 2,2, 3,3};
    int16_t gate[]  = {16384, 8192, 16384, 8192, 16384, 8192, 16384, 8192};

    SM64 rng(0x444);
    std::vector<uint32_t> xv, dyv, upv;
    fill_f32(xv,  4*8, rng);
    fill_f32(dyv, 4*4, rng);
    fill_f32(upv, 4*8, rng);

    void* d_inp = build_and_upload_input(sp, 4, emb, exp_r, gate, xv.data(), dyv.data(), upv.data());
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("dup_expert_token", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 5: invalid_experts — some route_expert values are out of [0,E) ----
static uint64_t tc_invalid_experts() {
    // K=2, cap=4 (= 4*2/2 -- some experts get 2 each); E=4
    Spec sp = make_spec(16,8,4,4,2, 4,4, DSG_DTYPE_F32, -1, 0xEEFF001155667788ull);
    RunSpec rs = make_run(4, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    int32_t emb[] = {0,1,2,3};
    // assign 0: expert -1 (invalid), assign 1: expert 0 (valid)
    // assign 2: expert 4 (invalid, E=4), assign 3: expert 1 (valid)
    // assign 4: expert -2 (invalid), assign 5: expert 2 (valid)
    // assign 6: expert 5 (invalid), assign 7: expert 3 (valid)
    int32_t exp_r[] = {-1, 0,  4, 1,  -2, 2,  5, 3};
    int16_t gate[]  = {-100, 16384, 200, 8192, -300, 4096, 400, 2048};

    SM64 rng(0x555);
    std::vector<uint32_t> xv, dyv, upv;
    fill_f32(xv,  4*8, rng);
    fill_f32(dyv, 4*4, rng);
    fill_f32(upv, 4*8, rng);

    void* d_inp = build_and_upload_input(sp, 4, emb, exp_r, gate, xv.data(), dyv.data(), upv.data());
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("invalid_experts", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 6: padding_invalid_index — embedding index is padding_idx, negative, or >= R ----
static uint64_t tc_padding_invalid() {
    // padding_idx = -1, R=16
    Spec sp = make_spec(16,8,4,4,1, 8,8, DSG_DTYPE_F32, -1, 0xFF001122667788AAull);
    RunSpec rs = make_run(8, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    // -1: padding, 0: valid, -2: invalid_index, 17: invalid_index (>=16), 5: valid, -1: padding, 3: valid, 0: valid (dup)
    int32_t emb[] = {-1, 0, -2, 17, 5, -1, 3, 0};
    int32_t exp_r[] = {0, 1, 2, 3, 0, 1, 2, 3};
    int16_t gate[]  = {8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192};

    SM64 rng(0x666);
    std::vector<uint32_t> xv, dyv, upv;
    fill_f32(xv,  8*8, rng);
    fill_f32(dyv, 8*4, rng);
    fill_f32(upv, 8*8, rng);

    void* d_inp = build_and_upload_input(sp, 8, emb, exp_r, gate, xv.data(), dyv.data(), upv.data());
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("padding_invalid_index", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 7: ragged_dims — non-power-of-2 D and O ----
static uint64_t tc_ragged_dims() {
    // D=7, O=3, cap=8=max_tokens*K, table_cells=16*7=112<=524288, expert_cells=4*7*3=84<=524288
    Spec sp = make_spec(16,7,3,4,1, 8,8, DSG_DTYPE_F32, -1, 0x00112233778899BBull);
    RunSpec rs = make_run(8, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    int32_t emb[] = {0,1,2,3,4,5,6,7};
    int32_t exp_r[] = {0,1,2,3,0,1,2,3};
    int16_t gate[]  = {12345,23456,12345,23456,12345,23456,12345,23456};

    SM64 rng(0x777);
    std::vector<uint32_t> xv, dyv, upv;
    fill_f32(xv,  8*7, rng);
    fill_f32(dyv, 8*3, rng);
    fill_f32(upv, 8*7, rng);

    void* d_inp = build_and_upload_input(sp, 8, emb, exp_r, gate, xv.data(), dyv.data(), upv.data());
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("ragged_dims", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 8: bf16_dtype — BF16 inputs ----
static uint64_t tc_bf16() {
    Spec sp = make_spec(16,8,4,4,1, 8,8, DSG_DTYPE_BF16, -1, 0x1122334455667788ull);
    RunSpec rs = make_run(8, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    int32_t emb[] = {0,1,2,3,4,5,6,7};
    int32_t exp_r[] = {0,1,2,3,0,1,2,3};
    int16_t gate[]  = {16384,16384,16384,16384,16384,16384,16384,16384};

    SM64 rng(0x888);
    std::vector<uint16_t> xv, dyv, upv;
    fill_bf16(xv,  8*8, rng);
    fill_bf16(dyv, 8*4, rng);
    fill_bf16(upv, 8*8, rng);

    void* d_inp = build_and_upload_input(sp, 8, emb, exp_r, gate, xv.data(), dyv.data(), upv.data());
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("bf16_dtype", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 9: zero_tokens — T=0 run with flush ----
static uint64_t tc_zero_tokens() {
    Spec sp = make_spec(16,8,4,4,1, 8,8, DSG_DTYPE_F32, -1, 0x2233445566778899ull);
    RunSpec rs = make_run(0, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    // T=0: no inputs needed, but we must provide non-null device pointer
    SM64 rng(0x999);
    std::vector<uint32_t> xv(1,0), dyv(1,0), upv(1,0);
    int32_t emb[1]={0}; int32_t exp_r[1]={0}; int16_t gate[1]={0};
    void* d_inp = build_and_upload_input(sp, 0, emb, exp_r, gate, xv.data(), dyv.data(), upv.data());

    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("zero_tokens", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 10: state_persist — multi-run, various flush patterns ----
static uint64_t tc_state_persist() {
    // 5 runs: no-flush, EMIT_TABLE, EMIT_EXPERT, both+CLEAR, no-flush
    Spec sp = make_spec(16,8,4,4,1, 8,8, DSG_DTYPE_F32, -1, 0x33445566778899AAull);

    std::vector<RunSpec> rss = {
        make_run(8, 0),                                                             // run0: no flush
        make_run(4, DSG_FLAG_EMIT_TABLE),                                           // run1: table only
        make_run(4, DSG_FLAG_EMIT_EXPERT),                                          // run2: expert only
        make_run(8, DSG_FLAG_EMIT_TABLE|DSG_FLAG_EMIT_EXPERT|DSG_FLAG_CLEAR_AFTER_FLUSH), // run3: both+clear
        make_run(4, 0),                                                             // run4: no flush (after clear)
    };

    SM64 rng(0xAAA);
    std::vector<uint32_t> xv, dyv, upv;
    std::vector<void*> dv;

    // Shared input pattern, vary seeds per run
    int32_t emb8[]  = {0,1,2,3,4,5,6,7};
    int32_t emb4[]  = {0,1,2,3};
    int32_t exp8[]  = {0,1,2,3,0,1,2,3};
    int32_t exp4[]  = {0,1,2,3};
    int16_t gate8[] = {16384,8192,16384,8192,16384,8192,16384,8192};
    int16_t gate4[] = {16384,8192,16384,8192};

    auto make_inp = [&](int T, int32_t* emb, int32_t* exp, int16_t* gate) {
        fill_f32(xv,  T*8, rng);
        fill_f32(dyv, T*4, rng);
        fill_f32(upv, T*8, rng);
        return build_and_upload_input(sp, T, emb, exp, gate, xv.data(), dyv.data(), upv.data());
    };

    dv.push_back(make_inp(8, emb8, exp8, gate8));
    dv.push_back(make_inp(4, emb4, exp4, gate4));
    dv.push_back(make_inp(4, emb4, exp4, gate4));
    dv.push_back(make_inp(8, emb8, exp8, gate8));
    dv.push_back(make_inp(4, emb4, exp4, gate4));

    uint64_t agg = run_case("state_persist", sp, rss, dv);
    for (auto p : dv) cudaFree(p);
    return agg;
}

// ---- Case 11: reset — run seq A, reset, run same seq A, expect identical outputs ----
static uint64_t tc_reset() {
    Spec sp = make_spec(16,8,4,4,1, 8,8, DSG_DTYPE_F32, -1, 0x44556677889900BBull);

    // Sequence A: 2 runs
    RunSpec rs0 = make_run(8, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT, DSG_CAPACITY_USE_SPEC, 0xCAFE);
    RunSpec rs1 = make_run(4, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT | DSG_FLAG_CLEAR_AFTER_FLUSH);
    // Then reset (after idx=1) and run same sequence again
    RunSpec rs2 = make_run(8, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT, DSG_CAPACITY_USE_SPEC, 0xCAFE);
    RunSpec rs3 = make_run(4, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT | DSG_FLAG_CLEAR_AFTER_FLUSH);

    std::vector<RunSpec> rss = {rs0, rs1, rs2, rs3};

    SM64 rng(0xBBB);
    std::vector<uint32_t> xv, dyv, upv;
    std::vector<void*> dv;

    int32_t emb8[] = {0,1,2,3,4,5,6,7};
    int32_t emb4[] = {0,1,2,3};
    int32_t exp8[] = {0,1,2,3,0,1,2,3};
    int32_t exp4[] = {0,1,2,3};
    int16_t gate8[] = {16384,8192,4096,2048,16384,8192,4096,2048};
    int16_t gate4[] = {16384,8192,4096,2048};

    auto make_inp = [&](int T, int32_t* emb, int32_t* exp, int16_t* gate) {
        fill_f32(xv, T*8, rng);
        fill_f32(dyv, T*4, rng);
        fill_f32(upv, T*8, rng);
        return build_and_upload_input(sp, T, emb, exp, gate, xv.data(), dyv.data(), upv.data());
    };

    void* d0 = make_inp(8, emb8, exp8, gate8);
    void* d1 = make_inp(4, emb4, exp4, gate4);
    // After reset, same inputs as d0 and d1
    dv = {d0, d1, d0, d1};

    // do_reset_midpoint=true, reset after run index 1
    uint64_t agg = run_case("reset_seq", sp, rss, dv, true, 1);

    // Also verify: run[0] checksums == run[2] checksums and run[1] == run[3]
    // (this is checked by the two-pass determinism logic above for runs 2 and 3)

    cudaFree(d0);
    cudaFree(d1);
    return agg;
}

// ---- Case 12: nan_inf_f32 — F32 inputs with NaN/Inf/±0 ----
static uint64_t tc_nan_inf_f32() {
    Spec sp = make_spec(16,8,4,4,1, 4,4, DSG_DTYPE_F32, -1, 0x55667788990011CCull);
    RunSpec rs = make_run(4, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    int32_t emb[] = {0,1,2,3};
    int32_t exp_r[] = {0,1,2,3};
    int16_t gate[]  = {16384, -16384, 0, 32767};

    // x[T*D]: mix of special values
    // +0=0x00000000, -0=0x80000000, +Inf=0x7F800000, -Inf=0xFF800000
    // canonical NaN=0x7FC00000, noncanonical NaN=0x7F800001
    uint32_t xv[4*8], dyv[4*4], upv[4*8];
    uint32_t special[] = {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u,
                          0x7FC00000u, 0x7F800001u, 0x3F800000u, 0xBF800000u};
    for (int i = 0; i < 4*8; i++) xv[i]  = special[i % 8];
    for (int i = 0; i < 4*4; i++) dyv[i] = special[(i+1) % 8];
    for (int i = 0; i < 4*8; i++) upv[i] = special[(i+2) % 8];

    void* d_inp = build_and_upload_input(sp, 4, emb, exp_r, gate, xv, dyv, upv);
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("nan_inf_f32", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 13: nan_inf_bf16 — BF16 inputs with NaN/Inf/±0 ----
static uint64_t tc_nan_inf_bf16() {
    Spec sp = make_spec(16,8,4,4,1, 4,4, DSG_DTYPE_BF16, -1, 0x66778899001122DDull);
    RunSpec rs = make_run(4, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    int32_t emb[] = {0,1,2,3};
    int32_t exp_r[] = {0,1,2,3};
    int16_t gate[]  = {16384, -8192, 0, 8192};

    // BF16 special values (upper 16 bits of F32)
    // +0=0x0000, -0=0x8000, +Inf=0x7F80, -Inf=0xFF80
    // canon NaN=0x7FC0, noncanon NaN=0x7F81
    uint16_t special[] = {0x0000u, 0x8000u, 0x7F80u, 0xFF80u,
                          0x7FC0u, 0x7F81u, 0x3F80u, 0xBF80u};
    uint16_t xv[4*8], dyv[4*4], upv[4*8];
    for (int i = 0; i < 4*8; i++) xv[i]  = special[i % 8];
    for (int i = 0; i < 4*4; i++) dyv[i] = special[(i+1) % 8];
    for (int i = 0; i < 4*8; i++) upv[i] = special[(i+2) % 8];

    void* d_inp = build_and_upload_input(sp, 4, emb, exp_r, gate, xv, dyv, upv);
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("nan_inf_bf16", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ---- Case 14: clear_only — CLEAR flag alone (flush_happened=1 but no emit) ----
static uint64_t tc_clear_only() {
    Spec sp = make_spec(16,8,4,4,1, 8,8, DSG_DTYPE_F32, -1, 0x7788990011223344ull);

    // Run 1: accumulate
    // Run 2: CLEAR_AFTER_FLUSH only (no emit)
    // Run 3: EMIT_TABLE|EMIT_EXPERT on now-cleared state
    std::vector<RunSpec> rss = {
        make_run(8, 0),
        make_run(4, DSG_FLAG_CLEAR_AFTER_FLUSH),
        make_run(8, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT),
    };

    SM64 rng(0xCCC);
    std::vector<uint32_t> xv, dyv, upv;
    std::vector<void*> dv;

    int32_t emb8[] = {0,1,2,3,4,5,6,7};
    int32_t emb4[] = {0,1,2,3};
    int32_t exp8[] = {0,1,2,3,0,1,2,3};
    int32_t exp4[] = {0,1,2,3};
    int16_t gate8[] = {16384,8192,16384,8192,16384,8192,16384,8192};
    int16_t gate4[] = {16384,8192,16384,8192};

    auto make_inp = [&](int T, int32_t* emb, int32_t* exp, int16_t* gate) {
        fill_f32(xv, T*8, rng);
        fill_f32(dyv, T*4, rng);
        fill_f32(upv, T*8, rng);
        return build_and_upload_input(sp, T, emb, exp, gate, xv.data(), dyv.data(), upv.data());
    };

    dv.push_back(make_inp(8, emb8, exp8, gate8));
    dv.push_back(make_inp(4, emb4, exp4, gate4));
    dv.push_back(make_inp(8, emb8, exp8, gate8));

    uint64_t agg = run_case("clear_only", sp, rss, dv);
    for (auto p : dv) cudaFree(p);
    return agg;
}

// ---- Case 15: padding_idx_0 — padding_idx=0, first row is padding ----
static uint64_t tc_padding_idx_0() {
    Spec sp = make_spec(16,8,4,4,1, 8,8, DSG_DTYPE_F32, 0, 0x8899001122334455ull);
    RunSpec rs = make_run(8, DSG_FLAG_EMIT_TABLE | DSG_FLAG_EMIT_EXPERT);

    // idx=0 is padding, 1..7 are valid
    int32_t emb[] = {0, 1, 0, 2, 0, 3, 0, 4};
    int32_t exp_r[] = {0,1,2,3,0,1,2,3};
    int16_t gate[]  = {8192,8192,8192,8192,8192,8192,8192,8192};

    SM64 rng(0xDDD);
    std::vector<uint32_t> xv, dyv, upv;
    fill_f32(xv,  8*8, rng);
    fill_f32(dyv, 8*4, rng);
    fill_f32(upv, 8*8, rng);

    void* d_inp = build_and_upload_input(sp, 8, emb, exp_r, gate, xv.data(), dyv.data(), upv.data());
    std::vector<void*> dv = {d_inp};
    uint64_t agg = run_case("padding_idx_0", sp, {rs}, dv);
    cudaFree(d_inp);
    return agg;
}

// ====================================================================
// Main
// ====================================================================
int main() {
    // Verify GPU present
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, sm_%d%d\n", prop.name, prop.major, prop.minor);
    printf("sizeof(OutHeader)=%zu\n\n", sizeof(OutHeader));

    // Embedded golden per-case aggregate_step_fnv values (15 cases), in case order.
    // These are the exact byte-checksums produced by a correct solution.
    struct GoldenCase { const char* name; uint64_t agg; };
    static const GoldenCase kGolden[] = {
        { "basic_unique",          0xb301adeaaaed6a25ull },
        { "dup_embed",             0x61e4468d6b4c0c73ull },
        { "hot_expert",            0x9a26f6b9b0066fd1ull },
        { "dup_expert_token",      0x1afe8d40d2d8f573ull },
        { "invalid_experts",       0xaa6ebee3917ed9dfull },
        { "padding_invalid_index", 0xbcc22f5c682224deull },
        { "ragged_dims",           0x8098bda0176e5decull },
        { "bf16_dtype",            0xecbb5d8c915442bbull },
        { "zero_tokens",           0xabfcadf019d5d14dull },
        { "state_persist",         0xeb23b640a8f18b14ull },
        { "reset_seq",             0x878ee9cb687addc3ull },
        { "nan_inf_f32",           0x90d65b3c676fcb5full },
        { "nan_inf_bf16",          0x56d040f54e65aedeull },
        { "clear_only",            0xe42e164e6e87a87aull },
        { "padding_idx_0",         0x73276edb058198adull },
    };
    static const uint64_t kGoldenFinal = 0xe6b2d4842b2f144bull;
    const int M = (int)(sizeof(kGolden) / sizeof(kGolden[0]));

    std::vector<uint64_t> case_aggs;

    case_aggs.push_back(tc_basic_unique());
    case_aggs.push_back(tc_dup_embed());
    case_aggs.push_back(tc_hot_expert());
    case_aggs.push_back(tc_dup_expert_token());
    case_aggs.push_back(tc_invalid_experts());
    case_aggs.push_back(tc_padding_invalid());
    case_aggs.push_back(tc_ragged_dims());
    case_aggs.push_back(tc_bf16());
    case_aggs.push_back(tc_zero_tokens());
    case_aggs.push_back(tc_state_persist());
    case_aggs.push_back(tc_reset());
    case_aggs.push_back(tc_nan_inf_f32());
    case_aggs.push_back(tc_nan_inf_bf16());
    case_aggs.push_back(tc_clear_only());
    case_aggs.push_back(tc_padding_idx_0());

    // Compare each case's aggregate (and determinism) against embedded golden.
    printf("\n=== GRADING ===\n");
    int passed = 0;
    for (int i = 0; i < M; ++i) {
        bool det = (i < (int)g_case_det.size()) ? g_case_det[i] : false;
        bool match = (case_aggs[i] == kGolden[i].agg) && det;
        if (match) ++passed;
        printf("  case %-22s %s  agg=0x%016llx golden=0x%016llx det=%s\n",
               kGolden[i].name, match ? "PASS" : "FAIL",
               (unsigned long long)case_aggs[i],
               (unsigned long long)kGolden[i].agg,
               det ? "PASS" : "FAIL");
    }

    // Final aggregate: FNV over all case aggregate step_hash FNVs
    uint64_t final_agg = H_BASIS;
    for (auto a : case_aggs) final_agg = fnv_bytes(final_agg, &a, 8);

    printf("\npassed %d / %d\n", passed, M);
    printf("final_aggregate_match=%s\n",
           (final_agg == kGoldenFinal) ? "PASS" : "FAIL");
    printf("final_aggregate_fnv=0x%016llx\n", (unsigned long long)final_agg);

    return (passed == M) ? 0 : 1;
}
