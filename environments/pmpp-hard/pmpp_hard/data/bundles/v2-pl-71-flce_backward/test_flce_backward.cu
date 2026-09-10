// harness.cu — solution-agnostic test harness for FLCB-StateTrain v1.
// Implements deterministic input generation for §6.2 visible scenarios.
// Computes FNV-1a-64 checksums per §2.16 and prints them labeled per case/step.
// Does NOT embed or call the reference algorithm — the same harness grades
// independent student solutions.

#include "flce_backward_common.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

// ── FNV-1a-64 (§2.16: basis = FLCB_FNV_BASIS, prime = FLCB_FNV_PRIME) ───────
static uint64_t fnv_init() { return FLCB_FNV_BASIS; }
static uint64_t fnv_feed(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; ++i) { h ^= (uint64_t)p[i]; h *= FLCB_FNV_PRIME; }
    return h;
}

// ── Error helpers ──────────────────────────────────────────────────────────────
#define CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)
#define WARN(x, tag) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) \
        fprintf(stderr, "[%s] solution_run returned: %s\n", tag, cudaGetErrorString(_e)); \
} while(0)

// ── xorshift64 PRNG for deterministic input generation ────────────────────────
struct RNG {
    uint64_t s;
    explicit RNG(uint64_t seed) : s(seed | 1ull) {}
    uint64_t next() { s^=s<<13; s^=s>>7; s^=s<<17; return s; }
    // BF16 bit pattern with exponent in [0x3C,0x42]: magnitude in [0.0625,8.0]
    uint16_t small_bf16() {
        uint32_t exp  = 0x3Cu + (uint32_t)(next() % 7u);
        uint32_t mant = (uint32_t)(next() & 0x7Fu);
        uint32_t sign = (uint32_t)((next() >> 32) & 1u);
        return (uint16_t)((sign << 15) | (exp << 7) | mant);
    }
};

static uint32_t float_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint32_t cdiv32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// ── Compute all five hashes from device output pointers + already-fetched report ─
struct Hashes { uint64_t tensor, event, counter, report, combined; };

static Hashes compute_hashes(
    uint16_t* d_dx, float* d_loss, float* d_dW, float* d_dbias,
    FLCBEvent* d_evs, FLCBCounters* d_ctr, FLCBRunReport* d_rep,
    uint32_t H, uint32_t V)
{
    FLCBRunReport rep{};
    CHECK(cudaMemcpy(&rep, d_rep, sizeof(rep), cudaMemcpyDeviceToHost));

    // tensor_hash: dx[output_rows,H] || loss[output_rows] || dW[flushes,V,H] || db[flushes,V]
    uint64_t th = fnv_init();
    if (rep.output_rows > 0) {
        std::vector<uint16_t> dx((uint64_t)rep.output_rows * H);
        CHECK(cudaMemcpy(dx.data(), d_dx, dx.size() * 2, cudaMemcpyDeviceToHost));
        th = fnv_feed(th, dx.data(), dx.size() * 2);

        std::vector<float> loss(rep.output_rows);
        CHECK(cudaMemcpy(loss.data(), d_loss, loss.size() * 4, cudaMemcpyDeviceToHost));
        th = fnv_feed(th, loss.data(), loss.size() * 4);
    }
    if (rep.flushes_written > 0) {
        std::vector<float> dW((uint64_t)rep.flushes_written * V * H);
        CHECK(cudaMemcpy(dW.data(), d_dW, dW.size() * 4, cudaMemcpyDeviceToHost));
        th = fnv_feed(th, dW.data(), dW.size() * 4);

        std::vector<float> db((uint64_t)rep.flushes_written * V);
        CHECK(cudaMemcpy(db.data(), d_dbias, db.size() * 4, cudaMemcpyDeviceToHost));
        th = fnv_feed(th, db.data(), db.size() * 4);
    }

    // event_hash: events[events_written] raw bytes
    uint64_t eh = fnv_init();
    if (rep.events_written > 0) {
        std::vector<FLCBEvent> evs(rep.events_written);
        CHECK(cudaMemcpy(evs.data(), d_evs, evs.size() * sizeof(FLCBEvent), cudaMemcpyDeviceToHost));
        eh = fnv_feed(eh, evs.data(), evs.size() * sizeof(FLCBEvent));
    }

    // counter_hash: counters[1]
    FLCBCounters ctr{};
    CHECK(cudaMemcpy(&ctr, d_ctr, sizeof(ctr), cudaMemcpyDeviceToHost));
    uint64_t ch = fnv_feed(fnv_init(), &ctr, sizeof(ctr));

    // report_hash: report[1]
    uint64_t rh = fnv_feed(fnv_init(), &rep, sizeof(rep));

    // combined_hash: four u64 values in little-endian byte order
    uint64_t combined = fnv_init();
    combined = fnv_feed(combined, &th, 8);
    combined = fnv_feed(combined, &eh, 8);
    combined = fnv_feed(combined, &ch, 8);
    combined = fnv_feed(combined, &rh, 8);

    return {th, eh, ch, rh, combined};
}

static void print_hashes(const char* case_name, int step, const Hashes& h) {
    printf("[%s step%d] tensor_hash   = %016llx\n", case_name, step, (unsigned long long)h.tensor);
    printf("[%s step%d] event_hash    = %016llx\n", case_name, step, (unsigned long long)h.event);
    printf("[%s step%d] counter_hash  = %016llx\n", case_name, step, (unsigned long long)h.counter);
    printf("[%s step%d] report_hash   = %016llx\n", case_name, step, (unsigned long long)h.report);
    printf("[%s step%d] combined_hash = %016llx\n", case_name, step, (unsigned long long)h.combined);
}

// ── Embedded golden checksums (self-contained; see GOLDEN.txt) ────────────────
// Generated by reference_solution.cu on RTX 5080 (sm_120), CUDA 13.0, with
// --fmad=false --prec-div=true --prec-sqrt=true --ftz=false. One entry per
// printed (case,step). A unit "passes" only if all five hashes match.
struct Golden {
    const char* name; int step;
    uint64_t tensor, event, counter, report, combined;
};
static const Golden GOLDEN[] = {
    {"tiny_sanity",    0, 0x07f5ff167ef74958ull, 0x1c9d6e363e64019cull, 0x6fb025a34a7cb1f6ull, 0x94b36bf2007723bbull, 0x0f0ad772dc68e2cdull},
    {"ragged",         0, 0xf6931ba1e4449d61ull, 0x4a9c9c5b06a67e55ull, 0x4bd9f2a865d6318full, 0x7f53b56bacd74c43ull, 0xf87b7b514166577bull},
    {"mean_denom",     0, 0x79f66ad6e8feefeaull, 0x880a9b2dc21dbcf8ull, 0x66a84c4d05d7b117ull, 0x58cea3cf2217df46ull, 0x2b1252b8912aaeb7ull},
    {"label_smooth",   0, 0xc1e7eb909db3f7aaull, 0xa4419243146afb60ull, 0xc3ee39c991ed4840ull, 0xc6e6c077262fbe52ull, 0x1ac86287e94ef2c3ull},
    {"flush_state",    0, 0x6a300ee8e4d21e66ull, 0x2ab468755650da78ull, 0xab42948bf2cab03full, 0x18e9d80ba39622b2ull, 0xd8ba346811697c58ull},
    {"flush_state",    1, 0x5673cf37505c5862ull, 0x90fbe64f157d7678ull, 0x39261df2551b3db2ull, 0x91c8a141dbfba1dcull, 0xb48554f8efa878b1ull},
    {"invalid_target", 0, 0x82f5cd4ccb60099aull, 0x24a8d5647ef14ed3ull, 0x7a758f2f1d00cb1dull, 0x8e6b103b2f4cf2b9ull, 0xa33daba10ec63d60ull},
    {"invalid_op",     0, 0x969db3311fb01f16ull, 0x628d3b67f5bfa8f1ull, 0x8eceb18d455d6590ull, 0x68c4633853b4de13ull, 0xb3c71aff671bc10cull},
};
static const int GOLDEN_COUNT = (int)(sizeof(GOLDEN) / sizeof(GOLDEN[0]));

static int g_pass = 0;
static int g_total = 0;

// Compare one (case,step)'s computed hashes against the embedded golden entry.
// Prints the hashes (same labels as the reference oracle) plus a PASS/FAIL line,
// and updates the global pass/total accounting consumed by main().
static void check_case(const char* case_name, int step, const Hashes& h) {
    print_hashes(case_name, step, h);
    g_total += 1;
    const Golden* g = nullptr;
    for (int i = 0; i < GOLDEN_COUNT; ++i)
        if (step == GOLDEN[i].step && strcmp(case_name, GOLDEN[i].name) == 0) { g = &GOLDEN[i]; break; }
    bool ok = g != nullptr
        && h.tensor   == g->tensor
        && h.event    == g->event
        && h.counter  == g->counter
        && h.report   == g->report
        && h.combined == g->combined;
    if (ok) g_pass += 1;
    printf("[%s step%d] %s\n", case_name, step, ok ? "PASS" : "FAIL");
}

// ── Estimate event capacity needed — mirrors reference.cu preflight exactly ───
// Fixed knob: for MICRO row_count=0, cdiv(0,row_chunk)=0 → 2 events (BEGIN+DONE).
static uint32_t estimate_events(const FLCBSpec& s, const FLCBRunSpec& r,
                                const std::vector<FLCBOp>& ops)
{
    uint32_t n = 2u; // RUN_BEGIN + RUN_END
    for (const auto& op : ops) {
        if (op.opcode == FLCB_OP_MICRO) {
            uint64_t end = (uint64_t)op.row_offset + op.row_count;
            bool ok = (end <= r.input_rows
                    && op.label_smoothing_q16 <= 65536u
                    && (op.reduction == FLCB_RED_SUM || op.reduction == FLCB_RED_MEAN_VALID));
            if (!ok) { n += 1u; continue; }
            n += 2u + cdiv32(op.row_count, s.row_chunk);
        } else if (op.opcode == FLCB_OP_FLUSH) {
            n += 1u; // valid or invalid, one event either way
        } else if (op.opcode == FLCB_OP_BUMP) {
            n += 1u;
        } else {
            n += 1u; // INVALID_OP for unknown opcode
        }
    }
    return n;
}

// ── Per-run device buffer set ─────────────────────────────────────────────────
struct RunBufs {
    uint16_t*      d_ops_storage; // raw byte storage for FLCBOp array
    uint16_t*      d_x;
    int32_t*       d_target;
    uint16_t*      d_w;
    uint16_t*      d_bias;
    uint16_t*      d_dx;
    float*         d_loss;
    float*         d_dW;
    float*         d_dbias;
    FLCBEvent*     d_evs;
    FLCBCounters*  d_ctr;
    FLCBRunReport* d_rep;
    void*          d_ws;

    uint32_t H, V, n_flushes;

    void alloc(const FLCBSpec& spec,
               uint32_t max_ops_n, uint32_t input_rows_n,
               uint32_t out_rows, uint32_t flushes, uint32_t events,
               size_t ws_bytes)
    {
        H = spec.H; V = spec.V; n_flushes = flushes;
        CHECK(cudaMalloc(&d_ops_storage, (uint64_t)(max_ops_n + 1) * sizeof(FLCBOp)));
        CHECK(cudaMalloc(&d_x,           (uint64_t)input_rows_n * H * sizeof(uint16_t)));
        CHECK(cudaMalloc(&d_target,      (uint64_t)input_rows_n * sizeof(int32_t)));
        CHECK(cudaMalloc(&d_w,           (uint64_t)V * H * sizeof(uint16_t)));
        d_bias = nullptr;
        if (spec.has_bias)
            CHECK(cudaMalloc(&d_bias, (uint64_t)V * sizeof(uint16_t)));
        CHECK(cudaMalloc(&d_dx,  (uint64_t)(out_rows + 1) * H * sizeof(uint16_t)));
        CHECK(cudaMalloc(&d_loss,(uint64_t)(out_rows + 1) * sizeof(float)));
        if (flushes > 0) {
            CHECK(cudaMalloc(&d_dW,    (uint64_t)flushes * V * H * sizeof(float)));
            CHECK(cudaMalloc(&d_dbias, (uint64_t)flushes * V * sizeof(float)));
        } else {
            d_dW = nullptr; d_dbias = nullptr;
        }
        CHECK(cudaMalloc(&d_evs, (uint64_t)(events + 1) * sizeof(FLCBEvent)));
        CHECK(cudaMalloc(&d_ctr, sizeof(FLCBCounters)));
        CHECK(cudaMalloc(&d_rep, sizeof(FLCBRunReport)));
        if (ws_bytes > 0)
            CHECK(cudaMalloc(&d_ws, ws_bytes));
        else
            d_ws = nullptr;
    }

    void upload_ops(const std::vector<FLCBOp>& v) {
        CHECK(cudaMemcpy(d_ops_storage, v.data(), v.size() * sizeof(FLCBOp), cudaMemcpyHostToDevice));
    }
    void upload_x(const std::vector<uint16_t>& v) {
        CHECK(cudaMemcpy(d_x, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
    void upload_target(const std::vector<int32_t>& v) {
        CHECK(cudaMemcpy(d_target, v.data(), v.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    void upload_w(const std::vector<uint16_t>& v) {
        CHECK(cudaMemcpy(d_w, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
    void upload_bias(const std::vector<uint16_t>& v) {
        if (d_bias) CHECK(cudaMemcpy(d_bias, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }

    FLCBInputs make_inputs(uint32_t op_count) const {
        FLCBInputs in{};
        in.ops       = op_count ? (const FLCBOp*)d_ops_storage : nullptr;
        in.x_bf16    = d_x;
        in.target    = d_target;
        in.w_bf16    = d_w;
        in.bias_bf16 = d_bias;
        return in;
    }
    FLCBOutputs make_outputs() const {
        FLCBOutputs out{};
        out.dx_bf16         = d_dx;
        out.loss_f32        = d_loss;
        out.flush_dW_f32    = d_dW;
        out.flush_dbias_f32 = d_dbias;
        out.events          = d_evs;
        out.counters        = d_ctr;
        out.report          = d_rep;
        return out;
    }
    Hashes hashes() const {
        return compute_hashes(d_dx, d_loss, d_dW, d_dbias, d_evs, d_ctr, d_rep, H, V);
    }
    void free_all() {
        cudaFree(d_ops_storage); cudaFree(d_x); cudaFree(d_target);
        cudaFree(d_w); if (d_bias) cudaFree(d_bias);
        cudaFree(d_dx); cudaFree(d_loss);
        if (d_dW) cudaFree(d_dW); if (d_dbias) cudaFree(d_dbias);
        cudaFree(d_evs); cudaFree(d_ctr); cudaFree(d_rep);
        if (d_ws) cudaFree(d_ws);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// CASE 1: tiny_sanity
// H=16, V=7, row_chunk=3, vocab_tile=5, has_bias=1, ignore_index=-100
// Fixed seed: 1001
// One run: MICRO(0,6,SUM,1.0f,eps=0), MICRO(2,2,SUM,overlap), FLUSH(SNAPSHOT)
// ─────────────────────────────────────────────────────────────────────────────
static void case_tiny_sanity(cudaStream_t stream) {
    const char* name = "tiny_sanity";
    printf("\n=== %s ===\n", name);

    FLCBSpec spec{};
    spec.magic = FLCB_MAGIC; spec.version = FLCB_VERSION;
    spec.H = 16; spec.V = 7; spec.max_run_rows = 8; spec.max_ops = 8;
    spec.row_chunk = 3; spec.vocab_tile = 5;
    spec.ignore_index = -100; spec.has_bias = 1; spec.max_flushes_per_run = 4;

    const uint32_t input_rows = 8u;
    RNG rng(1001u);
    std::vector<uint16_t> h_x(input_rows * spec.H); for (auto& v : h_x) v = rng.small_bf16();
    std::vector<uint16_t> h_w(spec.V * spec.H);     for (auto& v : h_w) v = rng.small_bf16();
    std::vector<uint16_t> h_b(spec.V);              for (auto& v : h_b) v = rng.small_bf16();
    // one ignored row (-100), one last-class (V-1=6), duplicate labels, overlap for op1
    std::vector<int32_t> h_tgt = {0, 1, -100, 3, 6, 0, 2, 3};

    std::vector<FLCBOp> ops(3);
    ops[0] = {FLCB_OP_MICRO, 0, 6, FLCB_RED_SUM, 0, float_bits(1.0f), 0, 0, 0ull, 0x10ull};
    ops[1] = {FLCB_OP_MICRO, 2, 2, FLCB_RED_SUM, 0, float_bits(1.0f), 0, 0, 0ull, 0x11ull};
    ops[2] = {FLCB_OP_FLUSH, 0, 0, 0, 0, 0, FLCB_FLUSH_SNAPSHOT, 0, 0ull, 0x12ull};

    FLCBRunSpec rs{};
    rs.run_id = 0xABCD0001ull; rs.input_rows = input_rows; rs.op_count = 3;
    rs.output_row_capacity = 6 + 2; rs.flush_capacity = 1;
    rs.event_capacity = estimate_events(spec, rs, ops) + 8u;

    size_t ws = solution_workspace_bytes(&spec);
    RunBufs b{}; b.alloc(spec, 3, input_rows, rs.output_row_capacity, 1, rs.event_capacity, ws);
    b.upload_x(h_x); b.upload_target(h_tgt); b.upload_w(h_w); b.upload_bias(h_b); b.upload_ops(ops);

    void* state = nullptr;
    CHECK(solution_init(&spec, &state, stream));

    FLCBInputs in = b.make_inputs(rs.op_count);
    FLCBOutputs out = b.make_outputs();
    WARN(solution_run(state, &rs, &in, &out, b.d_ws, ws, stream), name);
    CHECK(cudaStreamSynchronize(stream));

    check_case(name, 0, b.hashes());
    solution_destroy(state);
    b.free_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 2: ragged
// H=31, V=127, row_chunk=7, vocab_tile=17, has_bias=0, ignore_index=-100
// Fixed seed: 1002
// Row counts 0,1,6,7,8,23 — exercises chunk/event counts and 0-row op
// ─────────────────────────────────────────────────────────────────────────────
static void case_ragged(cudaStream_t stream) {
    const char* name = "ragged";
    printf("\n=== %s ===\n", name);

    FLCBSpec spec{};
    spec.magic = FLCB_MAGIC; spec.version = FLCB_VERSION;
    spec.H = 31; spec.V = 127; spec.max_run_rows = 32; spec.max_ops = 16;
    spec.row_chunk = 7; spec.vocab_tile = 17;
    spec.ignore_index = -100; spec.has_bias = 0; spec.max_flushes_per_run = 4;

    const uint32_t input_rows = 23u;
    RNG rng(1002u);
    std::vector<uint16_t> h_x(input_rows * spec.H); for (auto& v : h_x) v = rng.small_bf16();
    std::vector<uint16_t> h_w(spec.V * spec.H);     for (auto& v : h_w) v = rng.small_bf16();
    std::vector<int32_t>  h_tgt(input_rows);
    for (auto& t : h_tgt) {
        uint64_t r = rng.next() % 10u;
        t = (r < 1) ? -100 : (int32_t)(rng.next() % spec.V);
    }

    // Six micro ops: offsets/counts as per §6.2 prose, fitting in 23 input rows
    struct MD { uint32_t off, cnt; };
    std::vector<MD> descs = {{0,0},{0,1},{0,6},{6,7},{0,8},{0,23}};

    uint32_t total_out = 0;
    for (auto& d : descs) total_out += d.cnt;

    std::vector<FLCBOp> ops;
    for (size_t i = 0; i < descs.size(); ++i) {
        FLCBOp op{};
        op.opcode = FLCB_OP_MICRO; op.row_offset = descs[i].off; op.row_count = descs[i].cnt;
        op.reduction = FLCB_RED_SUM; op.loss_scale_bits = float_bits(1.0f);
        op.tag = (uint64_t)(0x20 + i);
        ops.push_back(op);
    }
    { FLCBOp flush{}; flush.opcode = FLCB_OP_FLUSH; flush.aux = FLCB_FLUSH_SNAPSHOT; flush.tag = 0x29ull; ops.push_back(flush); }

    FLCBRunSpec rs{};
    rs.run_id = 0xABCD0002ull; rs.input_rows = input_rows; rs.op_count = (uint32_t)ops.size();
    rs.output_row_capacity = total_out; rs.flush_capacity = 1;
    rs.event_capacity = estimate_events(spec, rs, ops) + 8u;

    size_t ws = solution_workspace_bytes(&spec);
    RunBufs b{}; b.alloc(spec, (uint32_t)ops.size(), input_rows, total_out, 1, rs.event_capacity, ws);
    b.upload_x(h_x); b.upload_target(h_tgt); b.upload_w(h_w); b.upload_ops(ops);

    void* state = nullptr;
    CHECK(solution_init(&spec, &state, stream));

    FLCBInputs in = b.make_inputs(rs.op_count);
    FLCBOutputs out = b.make_outputs();
    WARN(solution_run(state, &rs, &in, &out, b.d_ws, ws, stream), name);
    CHECK(cudaStreamSynchronize(stream));

    check_case(name, 0, b.hashes());
    solution_destroy(state);
    b.free_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 3: mean_denom
// H=32, V=31, row_chunk=4, vocab_tile=31, has_bias=0, ignore_index=-100
// Fixed seed: 1003
// Op0: 6 rows, 4 ignored → valid_count=2, grad_scale=scale/2
// Op1: 2 rows, all ignored → valid_count=0, grad_scale=0
// ─────────────────────────────────────────────────────────────────────────────
static void case_mean_denom(cudaStream_t stream) {
    const char* name = "mean_denom";
    printf("\n=== %s ===\n", name);

    FLCBSpec spec{};
    spec.magic = FLCB_MAGIC; spec.version = FLCB_VERSION;
    spec.H = 32; spec.V = 31; spec.max_run_rows = 16; spec.max_ops = 8;
    spec.row_chunk = 4; spec.vocab_tile = 31;
    spec.ignore_index = -100; spec.has_bias = 0; spec.max_flushes_per_run = 4;

    const uint32_t input_rows = 8u;
    RNG rng(1003u);
    std::vector<uint16_t> h_x(input_rows * spec.H); for (auto& v : h_x) v = rng.small_bf16();
    std::vector<uint16_t> h_w(spec.V * spec.H);     for (auto& v : h_w) v = rng.small_bf16();
    // 4 ignored + 2 valid in rows 0-5; 2 ignored in rows 6-7
    std::vector<int32_t> h_tgt = {-100, 5, -100, 20, -100, -100, -100, -100};

    std::vector<FLCBOp> ops(2);
    ops[0] = {FLCB_OP_MICRO, 0, 6, FLCB_RED_MEAN_VALID, 0, float_bits(2.0f), 0, 0, 0ull, 0x30ull};
    ops[1] = {FLCB_OP_MICRO, 6, 2, FLCB_RED_MEAN_VALID, 0, float_bits(2.0f), 0, 0, 0ull, 0x31ull};

    FLCBRunSpec rs{};
    rs.run_id = 0xABCD0003ull; rs.input_rows = input_rows; rs.op_count = 2;
    rs.output_row_capacity = 8; rs.flush_capacity = 0;
    rs.event_capacity = estimate_events(spec, rs, ops) + 8u;

    size_t ws = solution_workspace_bytes(&spec);
    RunBufs b{}; b.alloc(spec, 2, input_rows, 8, 0, rs.event_capacity, ws);
    b.upload_x(h_x); b.upload_target(h_tgt); b.upload_w(h_w); b.upload_ops(ops);

    void* state = nullptr;
    CHECK(solution_init(&spec, &state, stream));

    FLCBInputs in = b.make_inputs(rs.op_count);
    FLCBOutputs out = b.make_outputs();
    WARN(solution_run(state, &rs, &in, &out, b.d_ws, ws, stream), name);
    CHECK(cudaStreamSynchronize(stream));

    check_case(name, 0, b.hashes());
    solution_destroy(state);
    b.free_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 4: label_smooth
// H=16, V=64, row_chunk=16, vocab_tile=97, has_bias=1, ignore_index=-100
// Fixed seed: 1004; label_smoothing_q16=6554 (≈0.1); repeated target class
// ─────────────────────────────────────────────────────────────────────────────
static void case_label_smooth(cudaStream_t stream) {
    const char* name = "label_smooth";
    printf("\n=== %s ===\n", name);

    FLCBSpec spec{};
    spec.magic = FLCB_MAGIC; spec.version = FLCB_VERSION;
    spec.H = 16; spec.V = 64; spec.max_run_rows = 16; spec.max_ops = 8;
    spec.row_chunk = 16; spec.vocab_tile = 97;
    spec.ignore_index = -100; spec.has_bias = 1; spec.max_flushes_per_run = 4;

    const uint32_t input_rows = 4u;
    RNG rng(1004u);
    std::vector<uint16_t> h_x(input_rows * spec.H); for (auto& v : h_x) v = rng.small_bf16();
    std::vector<uint16_t> h_w(spec.V * spec.H);     for (auto& v : h_w) v = rng.small_bf16();
    std::vector<uint16_t> h_b(spec.V);              for (auto& v : h_b) v = rng.small_bf16();
    std::vector<int32_t> h_tgt = {5, 5, -100, 5};   // repeated target, one ignored

    std::vector<FLCBOp> ops(1);
    ops[0] = {FLCB_OP_MICRO, 0, 4, FLCB_RED_SUM, 6554, float_bits(1.0f), 0, 0, 0ull, 0x40ull};

    FLCBRunSpec rs{};
    rs.run_id = 0xABCD0004ull; rs.input_rows = input_rows; rs.op_count = 1;
    rs.output_row_capacity = 4; rs.flush_capacity = 0;
    rs.event_capacity = estimate_events(spec, rs, ops) + 8u;

    size_t ws = solution_workspace_bytes(&spec);
    RunBufs b{}; b.alloc(spec, 1, input_rows, 4, 0, rs.event_capacity, ws);
    b.upload_x(h_x); b.upload_target(h_tgt); b.upload_w(h_w); b.upload_bias(h_b); b.upload_ops(ops);

    void* state = nullptr;
    CHECK(solution_init(&spec, &state, stream));

    FLCBInputs in = b.make_inputs(rs.op_count);
    FLCBOutputs out = b.make_outputs();
    WARN(solution_run(state, &rs, &in, &out, b.d_ws, ws, stream), name);
    CHECK(cudaStreamSynchronize(stream));

    check_case(name, 0, b.hashes());
    solution_destroy(state);
    b.free_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 5: flush_state — two solution_run calls, state persists
// H=32, V=31, row_chunk=3, vocab_tile=7, has_bias=1, ignore_index=-100
// Fixed seed: 1005
// Run 0: MICRO(0,3), FLUSH(SNAPSHOT)   → accumulates; snapshot = acc after 3 rows
// Run 1: MICRO(3,3), FLUSH(EMIT_AND_ZERO), MICRO(0,3), FLUSH(SNAPSHOT)
//        → flush 0 = acc from run0+run1 first 3 rows
//        → after zero: flush 1 = acc from only run1 last 3 rows
// ─────────────────────────────────────────────────────────────────────────────
static void case_flush_state(cudaStream_t stream) {
    const char* name = "flush_state";
    printf("\n=== %s ===\n", name);

    FLCBSpec spec{};
    spec.magic = FLCB_MAGIC; spec.version = FLCB_VERSION;
    spec.H = 32; spec.V = 31; spec.max_run_rows = 8; spec.max_ops = 8;
    spec.row_chunk = 3; spec.vocab_tile = 7;
    spec.ignore_index = -100; spec.has_bias = 1; spec.max_flushes_per_run = 4;

    const uint32_t input_rows = 6u;
    RNG rng(1005u);
    std::vector<uint16_t> h_x(input_rows * spec.H); for (auto& v : h_x) v = rng.small_bf16();
    std::vector<uint16_t> h_w(spec.V * spec.H);     for (auto& v : h_w) v = rng.small_bf16();
    std::vector<uint16_t> h_b(spec.V);              for (auto& v : h_b) v = rng.small_bf16();
    std::vector<int32_t>  h_tgt = {0, 5, -100, 10, 2, 15};

    // Upload shared tensors once
    uint16_t *d_x, *d_w, *d_b; int32_t *d_tgt;
    CHECK(cudaMalloc(&d_x,   (uint64_t)input_rows * spec.H * sizeof(uint16_t)));
    CHECK(cudaMalloc(&d_w,   (uint64_t)spec.V * spec.H * sizeof(uint16_t)));
    CHECK(cudaMalloc(&d_b,   (uint64_t)spec.V * sizeof(uint16_t)));
    CHECK(cudaMalloc(&d_tgt, (uint64_t)input_rows * sizeof(int32_t)));
    CHECK(cudaMemcpy(d_x,   h_x.data(),   h_x.size()  * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w,   h_w.data(),   h_w.size()  * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b,   h_b.data(),   h_b.size()  * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tgt, h_tgt.data(), h_tgt.size()* 4, cudaMemcpyHostToDevice));

    size_t ws = solution_workspace_bytes(&spec);
    void* d_ws; CHECK(cudaMalloc(&d_ws, ws));

    void* state = nullptr;
    CHECK(solution_init(&spec, &state, stream));

    auto run_one = [&](int step, const std::vector<FLCBOp>& ops,
                       uint32_t out_rows, uint32_t flushes, uint32_t input_row_count) {
        FLCBOp* d_ops;
        CHECK(cudaMalloc(&d_ops, ops.size() * sizeof(FLCBOp)));
        CHECK(cudaMemcpy(d_ops, ops.data(), ops.size() * sizeof(FLCBOp), cudaMemcpyHostToDevice));

        uint32_t n_evs = 64u;
        uint16_t* d_dx; float* d_loss; float* d_dW; float* d_dbias;
        FLCBEvent* d_evs; FLCBCounters* d_ctr; FLCBRunReport* d_rep;
        CHECK(cudaMalloc(&d_dx,    (uint64_t)(out_rows+1) * spec.H * 2));
        CHECK(cudaMalloc(&d_loss,  (uint64_t)(out_rows+1) * 4));
        CHECK(cudaMalloc(&d_dW,    (uint64_t)flushes * spec.V * spec.H * 4));
        CHECK(cudaMalloc(&d_dbias, (uint64_t)flushes * spec.V * 4));
        CHECK(cudaMalloc(&d_evs,   (uint64_t)n_evs * sizeof(FLCBEvent)));
        CHECK(cudaMalloc(&d_ctr,   sizeof(FLCBCounters)));
        CHECK(cudaMalloc(&d_rep,   sizeof(FLCBRunReport)));

        FLCBRunSpec rs{};
        rs.run_id = (uint64_t)(0xF000005ull + step);
        rs.input_rows = input_row_count;
        rs.op_count = (uint32_t)ops.size();
        rs.output_row_capacity = out_rows;
        rs.flush_capacity = flushes;
        rs.event_capacity = n_evs;

        FLCBInputs in{}; in.ops = d_ops; in.x_bf16 = d_x; in.target = d_tgt; in.w_bf16 = d_w; in.bias_bf16 = d_b;
        FLCBOutputs out_bufs{}; out_bufs.dx_bf16 = d_dx; out_bufs.loss_f32 = d_loss;
        out_bufs.flush_dW_f32 = d_dW; out_bufs.flush_dbias_f32 = d_dbias;
        out_bufs.events = d_evs; out_bufs.counters = d_ctr; out_bufs.report = d_rep;

        WARN(solution_run(state, &rs, &in, &out_bufs, d_ws, ws, stream), name);
        CHECK(cudaStreamSynchronize(stream));

        Hashes h = compute_hashes(d_dx, d_loss, d_dW, d_dbias, d_evs, d_ctr, d_rep, spec.H, spec.V);
        check_case(name, step, h);

        cudaFree(d_ops); cudaFree(d_dx); cudaFree(d_loss);
        cudaFree(d_dW); cudaFree(d_dbias); cudaFree(d_evs); cudaFree(d_ctr); cudaFree(d_rep);
    };

    // Run 0: MICRO(0,3), FLUSH(SNAPSHOT)
    run_one(0, {
        {FLCB_OP_MICRO, 0, 3, FLCB_RED_SUM, 0, float_bits(1.0f), 0, 0, 0ull, 0x50ull},
        {FLCB_OP_FLUSH, 0, 0, 0, 0, 0, FLCB_FLUSH_SNAPSHOT, 0, 0ull, 0x51ull}
    }, 3, 1, input_rows);

    // Run 1: MICRO(3,3), FLUSH(EMIT_AND_ZERO), MICRO(0,3), FLUSH(SNAPSHOT)
    run_one(1, {
        {FLCB_OP_MICRO, 3, 3, FLCB_RED_SUM, 0, float_bits(1.0f), 0, 0, 0ull, 0x52ull},
        {FLCB_OP_FLUSH, 0, 0, 0, 0, 0, FLCB_FLUSH_EMIT_AND_ZERO, 0, 0ull, 0x53ull},
        {FLCB_OP_MICRO, 0, 3, FLCB_RED_SUM, 0, float_bits(1.0f), 0, 0, 0ull, 0x54ull},
        {FLCB_OP_FLUSH, 0, 0, 0, 0, 0, FLCB_FLUSH_SNAPSHOT, 0, 0ull, 0x55ull}
    }, 6, 2, input_rows);

    solution_destroy(state);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_tgt); cudaFree(d_ws);
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 6: invalid_target_vs_ignore
// H=16, V=16, row_chunk=4, vocab_tile=8, has_bias=0, ignore_index=0
// Fixed seed: 1006
// target 0 → ignored (valid class but == ignore_index)
// target -7 → invalid; target 21 → invalid (> V-1)
// Also includes BUMP op to exercise counter tracking
// ─────────────────────────────────────────────────────────────────────────────
static void case_invalid_target(cudaStream_t stream) {
    const char* name = "invalid_target";
    printf("\n=== %s ===\n", name);

    FLCBSpec spec{};
    spec.magic = FLCB_MAGIC; spec.version = FLCB_VERSION;
    spec.H = 16; spec.V = 16; spec.max_run_rows = 8; spec.max_ops = 8;
    spec.row_chunk = 4; spec.vocab_tile = 8;
    spec.ignore_index = 0; spec.has_bias = 0; spec.max_flushes_per_run = 4;

    const uint32_t input_rows = 6u;
    RNG rng(1006u);
    std::vector<uint16_t> h_x(input_rows * spec.H); for (auto& v : h_x) v = rng.small_bf16();
    std::vector<uint16_t> h_w(spec.V * spec.H);     for (auto& v : h_w) v = rng.small_bf16();
    // 0=ignored, 5=valid, -7=invalid, 21=invalid, 3=valid, 0=ignored
    std::vector<int32_t> h_tgt = {0, 5, -7, 21, 3, 0};

    std::vector<FLCBOp> ops(3);
    ops[0] = {FLCB_OP_MICRO, 0, 6, FLCB_RED_SUM, 0, float_bits(1.0f), 0, 0, 0ull, 0x60ull};
    ops[1] = {FLCB_OP_BUMP,  0, 0, 0, 0, 0, FLCB_BUMP_ROWS_TOTAL, 0, 1000ull, 0x61ull};
    ops[2] = {FLCB_OP_MICRO, 0, 4, FLCB_RED_SUM, 0, float_bits(0.5f), 0, 0, 0ull, 0x62ull};

    FLCBRunSpec rs{};
    rs.run_id = 0xABCD0006ull; rs.input_rows = input_rows; rs.op_count = 3;
    rs.output_row_capacity = 6 + 4; rs.flush_capacity = 0;
    rs.event_capacity = estimate_events(spec, rs, ops) + 8u;

    size_t ws = solution_workspace_bytes(&spec);
    RunBufs b{}; b.alloc(spec, 3, input_rows, rs.output_row_capacity, 0, rs.event_capacity, ws);
    b.upload_x(h_x); b.upload_target(h_tgt); b.upload_w(h_w); b.upload_ops(ops);

    void* state = nullptr;
    CHECK(solution_init(&spec, &state, stream));

    FLCBInputs in = b.make_inputs(rs.op_count);
    FLCBOutputs out = b.make_outputs();
    WARN(solution_run(state, &rs, &in, &out, b.d_ws, ws, stream), name);
    CHECK(cudaStreamSynchronize(stream));

    check_case(name, 0, b.hashes());
    solution_destroy(state);
    b.free_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 7: invalid_op_continuation
// H=16, V=31, row_chunk=4, vocab_tile=8, has_bias=0, ignore_index=-100
// Fixed seed: 1007
// bad opcode (99) between two valid micro ops;
// bad label_smoothing_q16 (65537) after a flush
// solver must emit INVALID_OP events and continue without corrupting state
// ─────────────────────────────────────────────────────────────────────────────
static void case_invalid_op(cudaStream_t stream) {
    const char* name = "invalid_op";
    printf("\n=== %s ===\n", name);

    FLCBSpec spec{};
    spec.magic = FLCB_MAGIC; spec.version = FLCB_VERSION;
    spec.H = 16; spec.V = 31; spec.max_run_rows = 8; spec.max_ops = 8;
    spec.row_chunk = 4; spec.vocab_tile = 8;
    spec.ignore_index = -100; spec.has_bias = 0; spec.max_flushes_per_run = 4;

    const uint32_t input_rows = 4u;
    RNG rng(1007u);
    std::vector<uint16_t> h_x(input_rows * spec.H); for (auto& v : h_x) v = rng.small_bf16();
    std::vector<uint16_t> h_w(spec.V * spec.H);     for (auto& v : h_w) v = rng.small_bf16();
    std::vector<int32_t> h_tgt = {0, 5, -100, 10};

    std::vector<FLCBOp> ops(5);
    ops[0] = {FLCB_OP_MICRO, 0, 4, FLCB_RED_SUM, 0,     float_bits(1.0f), 0, 0, 0ull, 0x70ull};
    ops[1] = {99,            0, 0, 0, 0,     0,            0, 0, 0ull, 0x71ull}; // bad opcode
    ops[2] = {FLCB_OP_MICRO, 0, 2, FLCB_RED_SUM, 0,     float_bits(1.0f), 0, 0, 0ull, 0x72ull};
    ops[3] = {FLCB_OP_FLUSH, 0, 0, 0, 0,     0,            FLCB_FLUSH_SNAPSHOT, 0, 0ull, 0x73ull};
    ops[4] = {FLCB_OP_MICRO, 0, 2, FLCB_RED_SUM, 65537, float_bits(1.0f), 0, 0, 0ull, 0x74ull}; // bad eps

    FLCBRunSpec rs{};
    rs.run_id = 0xABCD0007ull; rs.input_rows = input_rows; rs.op_count = 5;
    rs.output_row_capacity = 4 + 2; rs.flush_capacity = 1;
    rs.event_capacity = estimate_events(spec, rs, ops) + 8u;

    size_t ws = solution_workspace_bytes(&spec);
    RunBufs b{}; b.alloc(spec, 5, input_rows, rs.output_row_capacity, 1, rs.event_capacity, ws);
    b.upload_x(h_x); b.upload_target(h_tgt); b.upload_w(h_w); b.upload_ops(ops);

    void* state = nullptr;
    CHECK(solution_init(&spec, &state, stream));

    FLCBInputs in = b.make_inputs(rs.op_count);
    FLCBOutputs out = b.make_outputs();
    WARN(solution_run(state, &rs, &in, &out, b.d_ws, ws, stream), name);
    CHECK(cudaStreamSynchronize(stream));

    check_case(name, 0, b.hashes());
    solution_destroy(state);
    b.free_all();
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    // Confirm struct sizes match contract before running any cases
    static_assert(sizeof(FLCBSpec)      == 80,  "FLCBSpec size mismatch");
    static_assert(sizeof(FLCBRunSpec)   == 48,  "FLCBRunSpec size mismatch");
    static_assert(sizeof(FLCBOp)        == 48,  "FLCBOp size mismatch");
    static_assert(sizeof(FLCBEvent)     == 104, "FLCBEvent size mismatch");
    static_assert(sizeof(FLCBCounters)  == 184, "FLCBCounters size mismatch");
    static_assert(sizeof(FLCBRunReport) == 64,  "FLCBRunReport size mismatch");

    printf("sizeof(FLCBSpec)      = %zu\n", sizeof(FLCBSpec));
    printf("sizeof(FLCBRunSpec)   = %zu\n", sizeof(FLCBRunSpec));
    printf("sizeof(FLCBOp)        = %zu\n", sizeof(FLCBOp));
    printf("sizeof(FLCBEvent)     = %zu\n", sizeof(FLCBEvent));
    printf("sizeof(FLCBCounters)  = %zu\n", sizeof(FLCBCounters));
    printf("sizeof(FLCBRunReport) = %zu\n", sizeof(FLCBRunReport));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    case_tiny_sanity(stream);
    case_ragged(stream);
    case_mean_denom(stream);
    case_label_smooth(stream);
    case_flush_state(stream);
    case_invalid_target(stream);
    case_invalid_op(stream);

    CHECK(cudaStreamDestroy(stream));

    // Final grade line — the grader greps `passed [0-9]+ / [0-9]+`.
    printf("\npassed %d / %d\n", g_pass, g_total);
    printf("Done.\n");
    return 0;
}
