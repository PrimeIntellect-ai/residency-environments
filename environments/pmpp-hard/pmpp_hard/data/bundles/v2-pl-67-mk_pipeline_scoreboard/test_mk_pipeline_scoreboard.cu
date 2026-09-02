// file: test_mk_pipeline_scoreboard.cu

#include "mk_pipeline_scoreboard_common.h"
#include "mk_pipeline_scoreboard_oracle.hpp"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0x6700670067006767ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                              \
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
        const uint64_t span = (uint64_t)(hi - lo + 1);
        return lo + (int)(next_u64() % span);
    }
    bool chance_permille(int p) {
        if (p <= 0) return false;
        if (p >= 1000) return true;
        return (int)(next_u64() % 1000ULL) < p;
    }
};

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) {
        count = n;
        if (n > 0) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n));
    }
    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("DeviceBuffer upload size mismatch");
        if (count > 0) CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost));
        return host;
    }
};

template <typename T>
struct GuardedDeviceBuffer {
    uint8_t* raw = nullptr;
    T* ptr = nullptr;
    size_t count = 0;
    size_t data_bytes = 0;
    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;
    ~GuardedDeviceBuffer() { if (raw) cudaFree(raw); }
    void allocate(size_t n) {
        count = n;
        data_bytes = sizeof(T) * count;
        CUDA_CHECK(cudaMalloc((void**)&raw, kGuardBytes + data_bytes + kGuardBytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, kGuardBytes + data_bytes + kGuardBytes));
        ptr = (T*)(raw + kGuardBytes);
    }
    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> left(kGuardBytes), right(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(left.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(right.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (left[i] != kGuardByte) { if (error) { std::ostringstream o; o << "left guard corrupted for " << name << " at byte " << i; *error = o.str(); } return false; }
            if (right[i] != kGuardByte) { if (error) { std::ostringstream o; o << "right guard corrupted for " << name << " at byte " << i; *error = o.str(); } return false; }
        }
        return true;
    }
};

struct StepHost {
    MkpsRunSpec run;
    std::vector<int32_t> op;
    std::vector<uint32_t> a0, a1, a2, a3, a4, a5, a6, a7;
    std::vector<uint64_t> a8;
    std::vector<uint32_t> tiles; // [rows*16]
};

struct StepResult {
    std::vector<int64_t> counts;
    uint64_t pipeline_event_hash = 0, instr_hash = 0, tile_hash = 0, buffer_hash = 0;
    uint64_t pending_hash = 0, counter_hash = 0, event_seq_out = 0, state_checksum = 0;
};

struct Scenario {
    std::string name;
    MkpsProblemSpec spec;
    std::vector<StepHost> steps;
};

static MkpsProblemSpec make_spec(int buffers, int tiles, int instrs, int mr, int mw,
                                 int win, int mp, int nc, int max_batch, int max_steps) {
    MkpsProblemSpec spec = {};
    spec.abi_version = MKPS_ABI_VERSION;
    spec.buffer_count = buffers;
    spec.tile_count = tiles;
    spec.max_instrs = instrs;
    spec.max_reads_per_instr = mr;
    spec.max_writes_per_instr = mw;
    spec.issue_window = win;
    spec.max_pending_ops = mp;
    spec.counter_count = nc;
    spec.max_batch = max_batch;
    spec.max_steps = max_steps;
    spec.flags = 0;
    if (!mkps_validate_problem_spec(&spec)) throw std::runtime_error("invalid MkpsProblemSpec generated");
    return spec;
}

// Op-builder.
struct OpBuilder {
    std::vector<int32_t> op;
    std::vector<uint32_t> a0, a1, a2, a3, a4, a5, a6, a7;
    std::vector<uint64_t> a8;
    std::vector<uint32_t> tiles; // grows by 16 per row
    void push(int o, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, uint32_t v4,
              uint32_t v5, uint32_t v6, uint32_t v7, uint64_t v8,
              const std::vector<uint32_t>& reads, const std::vector<uint32_t>& writes) {
        op.push_back(o);
        a0.push_back(v0); a1.push_back(v1); a2.push_back(v2); a3.push_back(v3);
        a4.push_back(v4); a5.push_back(v5); a6.push_back(v6); a7.push_back(v7);
        a8.push_back(v8);
        uint32_t t[16] = {0};
        for (size_t i = 0; i < reads.size() && i < 8; ++i) t[i] = reads[i];
        for (size_t i = 0; i < writes.size() && i < 8; ++i) t[8 + i] = writes[i];
        for (int i = 0; i < 16; ++i) tiles.push_back(t[i]);
    }
    void enqueue(uint32_t id, const std::vector<uint32_t>& reads, const std::vector<uint32_t>& writes,
                 uint32_t scratch, uint32_t llat, uint32_t clat, uint32_t slat,
                 uint32_t out_counter, uint64_t seed) {
        push(MKPS_OP_ENQUEUE, id, (uint32_t)reads.size(), (uint32_t)writes.size(), scratch,
             llat, clat, slat, out_counter, seed, reads, writes);
    }
    void issue_loads(uint32_t limit) { push(MKPS_OP_ISSUE_LOADS, limit, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}); }
    void issue_compute(uint32_t limit) { push(MKPS_OP_ISSUE_COMPUTE, limit, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}); }
    void issue_stores(uint32_t limit) { push(MKPS_OP_ISSUE_STORES, limit, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}); }
    void advance(uint32_t delta, uint32_t max_ops) { push(MKPS_OP_ADVANCE, delta, max_ops, 0, 0, 0, 0, 0, 0, 0, {}, {}); }
    void cancel(uint32_t id) { push(MKPS_OP_CANCEL, id, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}); }
    void host_counter(uint32_t cid, uint32_t amount) { push(MKPS_OP_HOST_COUNTER, cid, amount, 0, 0, 0, 0, 0, 0, 0, {}, {}); }
    void raw(int o, uint32_t v0, uint32_t v1) { push(o, v0, v1, 0, 0, 0, 0, 0, 0, 0, {}, {}); }
};

static StepHost make_step(const MkpsProblemSpec& spec, int step_id, const OpBuilder& b) {
    StepHost step;
    step.run = {};
    step.run.abi_version = MKPS_ABI_VERSION;
    step.run.batch_size = (int32_t)b.op.size();
    step.run.step_id = step_id;
    if (!mkps_validate_run_spec(&step.run, &spec)) throw std::runtime_error("invalid MkpsRunSpec generated");
    const size_t rows = std::max<size_t>(1, b.op.size());
    step.op.assign(rows, -1);
    step.a0.assign(rows, 0); step.a1.assign(rows, 0); step.a2.assign(rows, 0); step.a3.assign(rows, 0);
    step.a4.assign(rows, 0); step.a5.assign(rows, 0); step.a6.assign(rows, 0); step.a7.assign(rows, 0);
    step.a8.assign(rows, 0);
    step.tiles.assign(rows * 16, 0);
    for (size_t i = 0; i < b.op.size(); ++i) {
        step.op[i] = b.op[i];
        step.a0[i] = b.a0[i]; step.a1[i] = b.a1[i]; step.a2[i] = b.a2[i]; step.a3[i] = b.a3[i];
        step.a4[i] = b.a4[i]; step.a5[i] = b.a5[i]; step.a6[i] = b.a6[i]; step.a7[i] = b.a7[i];
        step.a8[i] = b.a8[i];
        for (int k = 0; k < 16; ++k) step.tiles[i * 16 + k] = b.tiles[i * 16 + k];
    }
    return step;
}

// ---- Scenario 1: software pipeline overlap (loads issued while earlier stores) ----
static Scenario make_pipeline_overlap_scenario() {
    Scenario sc;
    sc.name = "pipeline_overlap";
    sc.spec = make_spec(8, 8, 16, 4, 4, 8, 32, 4, 64, 32);
    const uint32_t NO = MKPS_NO_U32;
    // Step 0: enqueue I1 (read t0, write t1), I2 (read t1, write t2), I3 (read t2, write t3)
    {
        OpBuilder b;
        b.enqueue(1, {0}, {1}, 0, 5, 3, 4, 0, 1111);
        b.enqueue(2, {1}, {2}, 0, 5, 3, 4, 1, 2222);
        b.enqueue(3, {2}, {3}, 0, 5, 3, 4, 2, 3333);
        b.issue_loads(4);   // I1 issues (read t0, write t1); I2 RAW on t1 (writer I1) -> stall, scan stops
        sc.steps.push_back(make_step(sc.spec, 0, b));
    }
    // Step 1: advance to land I1 loads; compute, store I1; then I2 can issue.
    {
        OpBuilder b;
        b.advance(5, 16);    // I1 LOAD_DONE x2 -> LOAD_READY
        b.issue_compute(4);  // I1 compute issue
        b.advance(3, 16);    // compute done
        b.issue_stores(4);   // I1 store issue
        b.advance(4, 16);    // store done -> tile t1 stored, counter inc, I1 done
        b.issue_loads(4);    // now I2 issuable (t1 resident current), I3 RAW stall on t2 (writer I2)
        sc.steps.push_back(make_step(sc.spec, 1, b));
    }
    // Step 2: drain I2 then I3.
    {
        OpBuilder b;
        b.advance(5, 16);
        b.issue_compute(4);
        b.advance(3, 16);
        b.issue_stores(4);
        b.advance(4, 16);
        b.issue_loads(4);   // I3 now issuable
        b.advance(5, 16);
        b.issue_compute(4);
        b.advance(3, 16);
        b.issue_stores(4);
        b.advance(4, 16);
        sc.steps.push_back(make_step(sc.spec, 2, b));
    }
    // Step 3: host counter increments + invalid ops.
    {
        OpBuilder b;
        b.host_counter(0, 7);
        b.host_counter(3, 100);
        b.host_counter(9, 5);     // invalid counter
        b.cancel(99);             // invalid (absent)
        b.cancel(1);              // invalid (terminal/DONE)
        sc.steps.push_back(make_step(sc.spec, 3, b));
    }
    (void)NO;
    return sc;
}

// ---- Scenario 2: WAW + WAR + page capacity out-of-order ----
static Scenario make_hazard_capacity_scenario() {
    Scenario sc;
    sc.name = "hazard_capacity";
    sc.spec = make_spec(4, 6, 12, 4, 4, 8, 16, 4, 64, 24);
    // Step 0: I1 writes t0; I2 writes t0 (WAW); I3 reads t5 writes t4 (independent, but capacity).
    {
        OpBuilder b;
        b.enqueue(1, {}, {0}, 0, 4, 2, 3, 0, 10);     // write t0
        b.enqueue(2, {}, {0}, 0, 4, 2, 3, 1, 20);     // WAW on t0 vs I1
        b.enqueue(3, {5}, {4}, 2, 4, 2, 3, 2, 30);    // read t5 + write t4 + 2 scratch = 4 buffers
        b.issue_loads(8);  // I1 issues (1 buf). I2 WAW -> stall, scan stops. I3 never reached.
        sc.steps.push_back(make_step(sc.spec, 0, b));
    }
    // Step 1: complete I1; then I2 issuable; capacity test with I3.
    {
        OpBuilder b;
        b.advance(4, 16); b.issue_compute(8); b.advance(2, 16); b.issue_stores(8); b.advance(3, 16);
        // I1 done. Now I2 (write t0) issuable. I3 needs 4 buffers (read t5 nonresident + write + 2 scratch)
        b.issue_loads(8);  // I2 issues (1 buf used). I3 needs 4 but only 3 free -> capacity stall, continue
        sc.steps.push_back(make_step(sc.spec, 1, b));
    }
    // Step 2: drain I2, then I3 fits.
    {
        OpBuilder b;
        b.advance(4, 16); b.issue_compute(8); b.advance(2, 16); b.issue_stores(8); b.advance(3, 16);
        b.issue_loads(8);  // I3 now fits (4 free)
        b.advance(4, 16); b.issue_compute(8); b.advance(2, 16); b.issue_stores(8); b.advance(3, 16);
        sc.steps.push_back(make_step(sc.spec, 2, b));
    }
    // Step 3: WAR. I4 reads t7? out of range tiles? tile_count=6. Use t3. I4 reads t3 (unreleased),
    // I5 writes t3 (WAR vs I4 read).
    {
        OpBuilder b;
        b.enqueue(4, {3}, {1}, 0, 4, 2, 3, 0, 40);
        b.enqueue(5, {0}, {3}, 0, 4, 2, 3, 1, 50);  // writes t3 while I4 reads t3 -> WAR
        b.issue_loads(8);   // I4 issues; I5 WAR stall on t3
        sc.steps.push_back(make_step(sc.spec, 3, b));
    }
    return sc;
}

// ---- Scenario 3: cancel + stale ops + resident reuse ----
static Scenario make_cancel_stale_scenario() {
    Scenario sc;
    sc.name = "cancel_stale";
    sc.spec = make_spec(6, 6, 12, 4, 4, 8, 16, 4, 64, 24);
    // Step 0: enqueue I1 (write t0), issue loads, then cancel I1 mid-flight (still LOADING).
    {
        OpBuilder b;
        b.enqueue(1, {1, 2}, {0}, 1, 6, 2, 3, 0, 100); // read t1,t2 + write t0 + scratch = 4 bufs
        b.issue_loads(4);   // I1 LOAD_ISSUED, pending LOAD_DONE x4
        b.cancel(1);        // cancel mid-flight: release nonresident bufs, dec readers, clear writer
        sc.steps.push_back(make_step(sc.spec, 0, b));
    }
    // Step 1: advance -> the cancelled instr's ops were removed; no stale drop expected (removed at cancel).
    {
        OpBuilder b;
        b.advance(10, 16);  // nothing to process
        sc.steps.push_back(make_step(sc.spec, 1, b));
    }
    // Step 2: enqueue I2 reads t1 writes t3, fully pipeline, store -> t3 resident.
    //         enqueue I3 reads t3 -> reuse resident buffer.
    {
        OpBuilder b;
        b.enqueue(2, {1}, {3}, 0, 4, 2, 3, 0, 200);
        b.issue_loads(4); b.advance(4, 16); b.issue_compute(4); b.advance(2, 16);
        b.issue_stores(4); b.advance(3, 16); // I2 done; t3 resident
        b.enqueue(3, {3}, {4}, 0, 4, 2, 3, 1, 300);
        b.issue_loads(4);   // I3 reads t3 -> reuse resident buffer; write t4 new buffer
        sc.steps.push_back(make_step(sc.spec, 2, b));
    }
    // Step 3: stale op via cancel after compute issued.
    {
        OpBuilder b;
        b.advance(4, 16); b.issue_compute(4); // I3 compute issued, COMPUTE_DONE pending
        b.cancel(3);        // cancel removes the COMPUTE_DONE op
        b.advance(5, 16);   // nothing stale (removed)
        sc.steps.push_back(make_step(sc.spec, 3, b));
    }
    return sc;
}

// ---- Scenario 4: triple-buffer deep pipeline, many counters, limits ----
static Scenario make_triple_buffer_scenario() {
    Scenario sc;
    sc.name = "triple_buffer";
    sc.spec = make_spec(12, 12, 24, 4, 4, 16, 64, 8, 96, 32);
    // independent chains so loads issue many at once; vary latencies for op_seq ordering.
    {
        OpBuilder b;
        for (uint32_t k = 0; k < 6; ++k) {
            uint32_t rt = k;        // distinct read tiles 0..5
            uint32_t wt = 6 + k;    // distinct write tiles 6..11
            b.enqueue(10 + k, {rt}, {wt}, 1, 2 + (k % 3), 1 + (k % 2), 2 + (k % 2),
                      k % 8, 7000 + k);
        }
        b.issue_loads(3);   // limit=3: only first 3 issue this op
        b.issue_loads(3);   // next 3
        sc.steps.push_back(make_step(sc.spec, 0, b));
    }
    {
        OpBuilder b;
        b.advance(2, 100);
        b.issue_compute(6);
        b.advance(2, 100);
        b.issue_stores(6);
        b.advance(3, 100);  // all stores land in (due_clock, op_seq) order
        sc.steps.push_back(make_step(sc.spec, 1, b));
    }
    {
        OpBuilder b;
        b.advance(5, 100);  // drain everything
        sc.steps.push_back(make_step(sc.spec, 2, b));
    }
    return sc;
}

// ---- randomized scenario generator ----
static StepHost make_random_step(const MkpsProblemSpec& spec, int step_id, int batch,
                                 SplitMix64& rng, uint32_t* next_id) {
    OpBuilder b;
    for (int i = 0; i < batch; ++i) {
        int roll = rng.uniform_int(0, 99);
        if (roll < 30) {
            // enqueue
            int rc = rng.uniform_int(0, spec.max_reads_per_instr);
            int wc = rng.uniform_int(0, spec.max_writes_per_instr);
            std::vector<uint32_t> reads, writes;
            for (int r = 0; r < rc; ++r) reads.push_back((uint32_t)rng.uniform_int(0, spec.tile_count + 1)); // occasionally OOR
            // ensure distinct writes mostly
            for (int w = 0; w < wc; ++w) {
                uint32_t t = (uint32_t)rng.uniform_int(0, spec.tile_count - 1);
                bool dup = false; for (uint32_t x : writes) if (x == t) dup = true;
                if (!dup) writes.push_back(t);
            }
            int scratch = rng.uniform_int(0, 2);
            uint32_t id = rng.chance_permille(120) ? 0 : (*next_id)++; // sometimes id=0 (invalid)
            uint32_t outc = rng.chance_permille(700) ? (uint32_t)rng.uniform_int(0, spec.counter_count - 1) : MKPS_NO_U32;
            b.enqueue(id, reads, writes, (uint32_t)scratch,
                      (uint32_t)rng.uniform_int(1, 6), (uint32_t)rng.uniform_int(1, 5),
                      (uint32_t)rng.uniform_int(1, 5), outc, rng.next_u64());
        } else if (roll < 45) {
            b.issue_loads((uint32_t)rng.uniform_int(0, 4));
        } else if (roll < 58) {
            b.issue_compute((uint32_t)rng.uniform_int(0, 4));
        } else if (roll < 70) {
            b.issue_stores((uint32_t)rng.uniform_int(0, 4));
        } else if (roll < 90) {
            b.advance((uint32_t)rng.uniform_int(0, 7), (uint32_t)rng.uniform_int(0, 8));
        } else if (roll < 96) {
            b.cancel((uint32_t)rng.uniform_int(1, (int)(*next_id > 1 ? *next_id : 2)));
        } else {
            b.host_counter((uint32_t)rng.uniform_int(0, spec.counter_count + 1),
                           (uint32_t)rng.uniform_int(1, 50));
        }
    }
    return make_step(spec, step_id, b);
}

static Scenario make_random_scenario(const std::string& name, MkpsProblemSpec spec,
                                     uint64_t seed, int nsteps) {
    Scenario sc;
    sc.name = name;
    sc.spec = spec;
    SplitMix64 rng(seed);
    uint32_t next_id = 1;
    for (int s = 0; s < nsteps; ++s) {
        int batch = (s % 8 == 0) ? 0 : rng.uniform_int(1, std::min(spec.max_batch, 20));
        sc.steps.push_back(make_random_step(spec, s, batch, rng, &next_id));
    }
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(make_pipeline_overlap_scenario());
    v.push_back(make_hazard_capacity_scenario());
    v.push_back(make_cancel_stale_scenario());
    v.push_back(make_triple_buffer_scenario());
    v.push_back(make_random_scenario("rand_small", make_spec(4, 4, 8, 3, 3, 4, 16, 3, 16, 24),
                                     g_state ^ 0x111ULL, 24));
    v.push_back(make_random_scenario("rand_mid", make_spec(8, 12, 16, 4, 4, 8, 32, 6, 24, 28),
                                     g_state ^ 0x222ULL, 28));
    v.push_back(make_random_scenario("rand_wide", make_spec(16, 32, 32, 4, 4, 16, 64, 8, 24, 24),
                                     g_state ^ 0x333ULL, 24));
    v.push_back(make_random_scenario("rand_tight", make_spec(2, 3, 6, 2, 2, 3, 8, 2, 12, 20),
                                     g_state ^ 0x444ULL, 20));
    return v;
}

static bool check_input_unchanged(const StepHost& step,
                                  const DeviceBuffer<int32_t>& d_op,
                                  const DeviceBuffer<uint32_t>& d_a0, const DeviceBuffer<uint32_t>& d_a1,
                                  const DeviceBuffer<uint32_t>& d_a2, const DeviceBuffer<uint32_t>& d_a3,
                                  const DeviceBuffer<uint32_t>& d_a4, const DeviceBuffer<uint32_t>& d_a5,
                                  const DeviceBuffer<uint32_t>& d_a6, const DeviceBuffer<uint32_t>& d_a7,
                                  const DeviceBuffer<uint64_t>& d_a8, const DeviceBuffer<uint32_t>& d_tiles,
                                  std::string* error) {
    if (d_op.download() != step.op) { if (error) *error = "input op modified"; return false; }
    if (d_a0.download() != step.a0) { if (error) *error = "input a0 modified"; return false; }
    if (d_a1.download() != step.a1) { if (error) *error = "input a1 modified"; return false; }
    if (d_a2.download() != step.a2) { if (error) *error = "input a2 modified"; return false; }
    if (d_a3.download() != step.a3) { if (error) *error = "input a3 modified"; return false; }
    if (d_a4.download() != step.a4) { if (error) *error = "input a4 modified"; return false; }
    if (d_a5.download() != step.a5) { if (error) *error = "input a5 modified"; return false; }
    if (d_a6.download() != step.a6) { if (error) *error = "input a6 modified"; return false; }
    if (d_a7.download() != step.a7) { if (error) *error = "input a7 modified"; return false; }
    if (d_a8.download() != step.a8) { if (error) *error = "input a8 modified"; return false; }
    if (d_tiles.download() != step.tiles) { if (error) *error = "input tiles modified"; return false; }
    return true;
}

static bool run_one_step(const MkpsProblemSpec& spec, const StepHost& step,
                        MkpsOracleState* oracle, void* state, void* workspace,
                        size_t workspace_bytes, cudaStream_t stream,
                        StepResult* result, std::string* error) {
    DeviceBuffer<int32_t> d_op;
    DeviceBuffer<uint32_t> d_a0, d_a1, d_a2, d_a3, d_a4, d_a5, d_a6, d_a7, d_tiles;
    DeviceBuffer<uint64_t> d_a8;
    d_op.allocate(step.op.size());
    d_a0.allocate(step.a0.size()); d_a1.allocate(step.a1.size()); d_a2.allocate(step.a2.size());
    d_a3.allocate(step.a3.size()); d_a4.allocate(step.a4.size()); d_a5.allocate(step.a5.size());
    d_a6.allocate(step.a6.size()); d_a7.allocate(step.a7.size()); d_a8.allocate(step.a8.size());
    d_tiles.allocate(step.tiles.size());
    d_op.upload(step.op);
    d_a0.upload(step.a0); d_a1.upload(step.a1); d_a2.upload(step.a2); d_a3.upload(step.a3);
    d_a4.upload(step.a4); d_a5.upload(step.a5); d_a6.upload(step.a6); d_a7.upload(step.a7);
    d_a8.upload(step.a8); d_tiles.upload(step.tiles);

    GuardedDeviceBuffer<int64_t> d_counts;
    GuardedDeviceBuffer<uint64_t> d_ev, d_ih, d_th, d_bh, d_ph, d_ch, d_evseq, d_state;
    d_counts.allocate(MKPS_COUNT_FIELDS);
    d_ev.allocate(1); d_ih.allocate(1); d_th.allocate(1); d_bh.allocate(1);
    d_ph.allocate(1); d_ch.allocate(1); d_evseq.allocate(1); d_state.allocate(1);

    MkpsInputs inputs = {};
    inputs.op = d_op.ptr;
    inputs.a0 = d_a0.ptr; inputs.a1 = d_a1.ptr; inputs.a2 = d_a2.ptr; inputs.a3 = d_a3.ptr;
    inputs.a4 = d_a4.ptr; inputs.a5 = d_a5.ptr; inputs.a6 = d_a6.ptr; inputs.a7 = d_a7.ptr;
    inputs.a8 = d_a8.ptr; inputs.tiles = d_tiles.ptr;

    MkpsOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.pipeline_event_hash = d_ev.ptr;
    outputs.instr_hash = d_ih.ptr;
    outputs.tile_hash = d_th.ptr;
    outputs.buffer_hash = d_bh.ptr;
    outputs.pending_hash = d_ph.ptr;
    outputs.counter_hash = d_ch.ptr;
    outputs.event_seq_out = d_evseq.ptr;
    outputs.state_checksum = d_state.ptr;

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(step, d_op, d_a0, d_a1, d_a2, d_a3, d_a4, d_a5, d_a6, d_a7, d_a8, d_tiles, error))
        return false;

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_ev.check_guards("pipeline_event_hash", error)) return false;
    if (!d_ih.check_guards("instr_hash", error)) return false;
    if (!d_th.check_guards("tile_hash", error)) return false;
    if (!d_bh.check_guards("buffer_hash", error)) return false;
    if (!d_ph.check_guards("pending_hash", error)) return false;
    if (!d_ch.check_guards("counter_hash", error)) return false;
    if (!d_evseq.check_guards("event_seq_out", error)) return false;
    if (!d_state.check_guards("state_checksum", error)) return false;

    const std::vector<int64_t> h_counts = d_counts.download_data();
    const std::vector<uint64_t> h_ev = d_ev.download_data();
    const std::vector<uint64_t> h_ih = d_ih.download_data();
    const std::vector<uint64_t> h_th = d_th.download_data();
    const std::vector<uint64_t> h_bh = d_bh.download_data();
    const std::vector<uint64_t> h_ph = d_ph.download_data();
    const std::vector<uint64_t> h_ch = d_ch.download_data();
    const std::vector<uint64_t> h_evseq = d_evseq.download_data();
    const std::vector<uint64_t> h_state = d_state.download_data();

    MkpsHostInputsView host_inputs = {};
    host_inputs.op = step.op.data();
    host_inputs.a0 = step.a0.data(); host_inputs.a1 = step.a1.data(); host_inputs.a2 = step.a2.data();
    host_inputs.a3 = step.a3.data(); host_inputs.a4 = step.a4.data(); host_inputs.a5 = step.a5.data();
    host_inputs.a6 = step.a6.data(); host_inputs.a7 = step.a7.data(); host_inputs.a8 = step.a8.data();
    host_inputs.tiles = step.tiles.data();

    MkpsExpected expected;
    oracle->step_once(step.run, host_inputs, &expected);

    MkpsHostOutputsView got = {};
    got.counts = h_counts.data();
    got.pipeline_event_hash = h_ev.data();
    got.instr_hash = h_ih.data();
    got.tile_hash = h_th.data();
    got.buffer_hash = h_bh.data();
    got.pending_hash = h_ph.data();
    got.counter_hash = h_ch.data();
    got.event_seq_out = h_evseq.data();
    got.state_checksum = h_state.data();

    if (!mkps_check_all_outputs(spec, expected, got, error)) return false;

    if (result) {
        result->counts = h_counts;
        result->pipeline_event_hash = h_ev[0];
        result->instr_hash = h_ih[0];
        result->tile_hash = h_th[0];
        result->buffer_hash = h_bh[0];
        result->pending_hash = h_ph[0];
        result->counter_hash = h_ch[0];
        result->event_seq_out = h_evseq[0];
        result->state_checksum = h_state[0];
    }
    return true;
}

static bool run_scenario_once(const Scenario& sc, bool verbose, std::vector<StepResult>* results,
                             int* passed_steps, int* total_steps, std::string* first_error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    MkpsOracleState oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.steps.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult result;
        std::string error;
        const bool ok = run_one_step(sc.spec, sc.steps[i], &oracle, state, workspace.ptr,
                                     std::max<size_t>(workspace_bytes, 1), stream,
                                     results ? &result : nullptr, &error);
        ++(*total_steps);
        if (ok) ++(*passed_steps);
        else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream oss;
                oss << sc.name << " step " << i << ": " << error;
                *first_error = oss.str();
            }
        }
        if (results) results->push_back(result);
        if (verbose) {
            std::printf("scenario %-22s step %02zu/%02zu batch=%d %s%s%s\n",
                        sc.name.c_str(), i, sc.steps.size(), sc.steps[i].run.batch_size,
                        ok ? "PASS" : "FAIL", ok ? "" : "  ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(const std::vector<StepResult>& a, const std::vector<StepResult>& b, std::string* error) {
    if (a.size() != b.size()) { if (error) *error = "result length mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].counts != b[i].counts || a[i].pipeline_event_hash != b[i].pipeline_event_hash ||
            a[i].instr_hash != b[i].instr_hash || a[i].tile_hash != b[i].tile_hash ||
            a[i].buffer_hash != b[i].buffer_hash || a[i].pending_hash != b[i].pending_hash ||
            a[i].counter_hash != b[i].counter_hash || a[i].event_seq_out != b[i].event_seq_out ||
            a[i].state_checksum != b[i].state_checksum) {
            if (error) {
                std::ostringstream oss;
                oss << "replay mismatch at step " << i << ": state a=0x" << std::hex
                    << a[i].state_checksum << ", b=0x" << b[i].state_checksum;
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
        const std::vector<Scenario> scenarios = build_scenarios();

        int passed = 0, total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results, replay_results;
            std::string error;

            const bool ok_base = run_scenario_once(sc, true, &base_results, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay_results, &passed, &total, &error);

            if (ok_base && ok_replay) {
                std::string ce;
                if (compare_results(base_results, replay_results, &ce))
                    std::printf("scenario %-22s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-22s exact replay FAIL  %s\n", sc.name.c_str(), ce.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-22s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
