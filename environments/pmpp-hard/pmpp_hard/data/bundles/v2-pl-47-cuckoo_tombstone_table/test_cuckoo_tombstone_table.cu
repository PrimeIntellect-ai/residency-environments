// file: test_cuckoo_tombstone_table.cu

#include "cuckoo_tombstone_table_common.h"
#include "cuckoo_tombstone_table_oracle.hpp"

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

static constexpr uint64_t g_state = 0x51d3c0ffee123456ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
                 << cudaGetErrorString(_err);                                   \
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
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t n) {
        count = n;
        if (n == 0) { ptr = nullptr; return; }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
    }
    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("upload size mismatch");
        if (count) CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count) CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost));
        return host;
    }
};

template <typename T>
struct GuardedDeviceBuffer {
    uint8_t* raw = nullptr;
    T* ptr = nullptr;
    size_t count = 0, data_bytes = 0, total_bytes = 0;
    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;
    ~GuardedDeviceBuffer() { if (raw) cudaFree(raw); }
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
        if (count) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> before(kGuardBytes), after(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(before.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (before[i] != kGuardByte) {
                if (error) { std::ostringstream o; o << "left guard corrupt for " << name << " at " << i; *error = o.str(); }
                return false;
            }
            if (after[i] != kGuardByte) {
                if (error) { std::ostringstream o; o << "right guard corrupt for " << name << " at " << i; *error = o.str(); }
                return false;
            }
        }
        return true;
    }
};

struct StepHost {
    CktRunSpec run;
    std::vector<int32_t> op_type;
    std::vector<int64_t> a0;
    std::vector<int64_t> a1;
};

struct StepResult {
    int32_t counts[CKT_NUM_COUNTS] = {0};
    uint64_t op_event_hash = 0, read_hash = 0, slot_state_hash = 0;
    uint64_t stash_hash = 0, page_hash = 0;
};

struct Scenario {
    std::string name;
    CktProblemSpec spec;
    std::vector<StepHost> steps;
};

static CktProblemSpec make_spec(int slot_count, int page_size, int neighborhood,
                                int max_disp, int stash_capacity, int max_tombstones,
                                int max_ops, int max_steps, uint64_t seed0, uint64_t seed1) {
    CktProblemSpec spec = {};
    spec.abi_version = CKT_ABI_VERSION;
    spec.slot_count = slot_count;
    spec.page_size = page_size;
    spec.neighborhood = neighborhood;
    spec.max_displacements_per_home = max_disp;
    spec.stash_capacity = stash_capacity;
    spec.max_tombstones = max_tombstones;
    spec.max_ops = max_ops;
    spec.max_steps = max_steps;
    spec.flags = 0;
    spec.seed0 = seed0;
    spec.seed1 = seed1;
    if (!ckt_validate_problem_spec(&spec))
        throw std::runtime_error("invalid CktProblemSpec generated");
    return spec;
}

static StepHost make_step(const CktProblemSpec& spec, int step_id,
                          const std::vector<int32_t>& op_type,
                          const std::vector<int64_t>& a0,
                          const std::vector<int64_t>& a1) {
    if (op_type.size() != a0.size() || op_type.size() != a1.size())
        throw std::runtime_error("step arg size mismatch");
    StepHost s;
    s.run = {};
    s.run.abi_version = CKT_ABI_VERSION;
    s.run.num_ops = (int32_t)op_type.size();
    s.run.step_id = step_id;
    if (!ckt_validate_run_spec(&s.run, &spec))
        throw std::runtime_error("invalid CktRunSpec generated");
    const size_t rows = std::max<size_t>(1, op_type.size());
    s.op_type.assign(rows, 0);
    s.a0.assign(rows, 0);
    s.a1.assign(rows, 0);
    for (size_t i = 0; i < op_type.size(); ++i) {
        s.op_type[i] = op_type[i];
        s.a0[i] = a0[i];
        s.a1[i] = a1[i];
    }
    return s;
}

// Builder helpers for readable scenario ops.
struct OpList {
    std::vector<int32_t> ot;
    std::vector<int64_t> a0;
    std::vector<int64_t> a1;
    void get(int64_t read_id, int64_t key) { ot.push_back(CKT_OP_GET); a0.push_back(read_id); a1.push_back(key); }
    void put(int64_t key, int64_t value) { ot.push_back(CKT_OP_PUT); a0.push_back(key); a1.push_back(value); }
    void del(int64_t key) { ot.push_back(CKT_OP_DELETE); a0.push_back(key); a1.push_back(0); }
    void pin(int64_t page) { ot.push_back(CKT_OP_PIN_PAGE); a0.push_back(page); a1.push_back(0); }
    void unpin(int64_t page) { ot.push_back(CKT_OP_UNPIN_PAGE); a0.push_back(page); a1.push_back(0); }
    void sweep(int64_t limit) { ot.push_back(CKT_OP_SWEEP_TOMBSTONES); a0.push_back(limit); a1.push_back(0); }
    void replay(int64_t limit) { ot.push_back(CKT_OP_REPLAY_STASH); a0.push_back(limit); a1.push_back(0); }
};

// ---------------------------------------------------------------------------
// Scenario 1: small dense table, heavy displacement chains + collisions.
// ---------------------------------------------------------------------------
static Scenario make_displacement_scenario() {
    Scenario sc;
    sc.name = "displacement_chains";
    sc.spec = make_spec(/*slot*/16, /*page*/4, /*nbhd*/4, /*disp*/8,
                        /*stash*/8, /*tomb*/6, /*max_ops*/64, /*steps*/32,
                        0x1111111111111111ULL, 0x2222222222222222ULL);
    SplitMix64 rng(g_state ^ 0xA1A1ULL);
    int read_id = 1;
    for (int s = 0; s < 24; ++s) {
        OpList ops;
        if (s == 0) {
            for (int k = 0; k < 14; ++k) ops.put(k, (int64_t)k * 100 + 1);
        } else if (s % 5 == 0) {
            // mass deletes to create tombstones
            for (int k = 0; k < 8; ++k) ops.del(k);
        } else if (s % 5 == 1) {
            // re-put deleted keys -> resurrection
            for (int k = 0; k < 8; ++k) ops.put(k, (int64_t)k * 7 + s);
        } else if (s % 5 == 2) {
            for (int k = 0; k < 16; ++k) ops.get(read_id++, k);
        } else {
            int n = rng.uniform_int(2, 10);
            for (int i = 0; i < n; ++i) {
                int k = rng.uniform_int(0, 24);
                int op = rng.uniform_int(0, 3);
                if (op == 0) ops.put(k, (int64_t)rng.next_u64());
                else if (op == 1) ops.del(k);
                else ops.get(read_id++, k);
            }
        }
        if (s == 19) { sc.steps.push_back(make_step(sc.spec, s, {}, {}, {})); continue; }
        sc.steps.push_back(make_step(sc.spec, s, ops.ot, ops.a0, ops.a1));
    }
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 2: pinning blocks placement/relocation, forces stash + OOM,
// then unpin + replay.
// ---------------------------------------------------------------------------
static Scenario make_pin_stash_replay_scenario() {
    Scenario sc;
    sc.name = "pin_stash_replay";
    sc.spec = make_spec(/*slot*/8, /*page*/2, /*nbhd*/3, /*disp*/4,
                        /*stash*/4, /*tomb*/4, /*max_ops*/48, /*steps*/24,
                        0x3333333333333333ULL, 0x4444444444444444ULL);
    {
        OpList ops;  // step 0: fill some, then pin all pages
        ops.put(10, 1000);
        ops.put(11, 1100);
        ops.put(12, 1200);
        sc.steps.push_back(make_step(sc.spec, 0, ops.ot, ops.a0, ops.a1));
    }
    {
        OpList ops;  // pin every page (4 pages) then attempt inserts -> stash/oom
        ops.pin(0); ops.pin(1); ops.pin(2); ops.pin(3);
        ops.put(20, 2000); ops.put(21, 2100); ops.put(22, 2200);
        ops.put(23, 2300); ops.put(24, 2400);  // beyond stash cap -> OOM
        sc.steps.push_back(make_step(sc.spec, 1, ops.ot, ops.a0, ops.a1));
    }
    {
        OpList ops;  // updates/gets still work on stash + table while pinned
        ops.put(20, 9999);   // update stash entry
        ops.put(10, 8888);   // update live (pin doesn't block updates)
        ops.del(11);         // delete live -> tombstone even while pinned
        ops.get(1, 20); ops.get(2, 10); ops.get(3, 11); ops.get(4, 999);
        sc.steps.push_back(make_step(sc.spec, 2, ops.ot, ops.a0, ops.a1));
    }
    {
        OpList ops;  // unpin pages, replay stash
        ops.unpin(0); ops.unpin(1); ops.unpin(2); ops.unpin(3);
        ops.replay(10);
        ops.get(5, 20); ops.get(6, 21);
        sc.steps.push_back(make_step(sc.spec, 3, ops.ot, ops.a0, ops.a1));
    }
    {
        OpList ops;  // invalid pin/unpin
        ops.unpin(0);     // already 0 -> invalid
        ops.pin(99);      // out of range -> invalid
        ops.unpin(99);    // out of range -> invalid
        ops.sweep(100);
        sc.steps.push_back(make_step(sc.spec, 4, ops.ot, ops.a0, ops.a1));
    }
    SplitMix64 rng(g_state ^ 0xB2B2ULL);
    for (int s = 5; s < 20; ++s) {
        OpList ops;
        int n = rng.uniform_int(1, 8);
        for (int i = 0; i < n; ++i) {
            int op = rng.uniform_int(0, 6);
            int k = rng.uniform_int(0, 30);
            int pg = rng.uniform_int(0, 4);
            if (op == CKT_OP_GET) ops.get(s * 100 + i, k);
            else if (op == CKT_OP_PUT) ops.put(k, (int64_t)rng.next_u64());
            else if (op == CKT_OP_DELETE) ops.del(k);
            else if (op == CKT_OP_PIN_PAGE) ops.pin(pg);
            else if (op == CKT_OP_UNPIN_PAGE) ops.unpin(pg);
            else if (op == CKT_OP_SWEEP_TOMBSTONES) ops.sweep(rng.uniform_int(0, 3));
            else ops.replay(rng.uniform_int(0, 3));
        }
        sc.steps.push_back(make_step(sc.spec, s, ops.ot, ops.a0, ops.a1));
    }
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 3: tombstone auto-sweep pressure with some pages pinned.
// ---------------------------------------------------------------------------
static Scenario make_tombstone_sweep_scenario() {
    Scenario sc;
    sc.name = "tombstone_autosweep";
    sc.spec = make_spec(/*slot*/24, /*page*/4, /*nbhd*/5, /*disp*/6,
                        /*stash*/6, /*tomb*/3, /*max_ops*/64, /*steps*/24,
                        0x55aa55aa55aa55aaULL, 0x9696969696969696ULL);
    {
        OpList ops;
        for (int k = 0; k < 20; ++k) ops.put(k, (int64_t)k + 1);
        sc.steps.push_back(make_step(sc.spec, 0, ops.ot, ops.a0, ops.a1));
    }
    {
        OpList ops;  // pin a couple pages so some tombstones can't be swept
        ops.pin(0); ops.pin(2);
        for (int k = 0; k < 16; ++k) ops.del(k);  // each delete may trigger auto-sweep
        sc.steps.push_back(make_step(sc.spec, 1, ops.ot, ops.a0, ops.a1));
    }
    {
        OpList ops;
        ops.unpin(0); ops.unpin(2);
        ops.sweep(100);  // sweep remaining
        for (int k = 0; k < 20; ++k) ops.put(k, (int64_t)k * 3 + 7);  // resurrect/insert
        sc.steps.push_back(make_step(sc.spec, 2, ops.ot, ops.a0, ops.a1));
    }
    SplitMix64 rng(g_state ^ 0xC3C3ULL);
    for (int s = 3; s < 20; ++s) {
        OpList ops;
        int n = rng.uniform_int(2, 10);
        for (int i = 0; i < n; ++i) {
            int op = rng.uniform_int(0, 6);
            int k = rng.uniform_int(0, 30);
            int pg = rng.uniform_int(0, 6);
            if (op == CKT_OP_GET) ops.get(s * 50 + i, k);
            else if (op == CKT_OP_PUT) ops.put(k, (int64_t)rng.next_u64());
            else if (op == CKT_OP_DELETE) ops.del(k);
            else if (op == CKT_OP_PIN_PAGE) ops.pin(pg);
            else if (op == CKT_OP_UNPIN_PAGE) ops.unpin(pg);
            else if (op == CKT_OP_SWEEP_TOMBSTONES) ops.sweep(rng.uniform_int(0, 4));
            else ops.replay(rng.uniform_int(0, 4));
        }
        sc.steps.push_back(make_step(sc.spec, s, ops.ot, ops.a0, ops.a1));
    }
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 4: neighborhood == 1 (pure cuckoo, no hopscotch displacement room).
// ---------------------------------------------------------------------------
static Scenario make_nbhd1_scenario() {
    Scenario sc;
    sc.name = "neighborhood_one";
    sc.spec = make_spec(/*slot*/13, /*page*/3, /*nbhd*/1, /*disp*/3,
                        /*stash*/5, /*tomb*/4, /*max_ops*/48, /*steps*/20,
                        0x0123456789abcdefULL, 0xfedcba9876543210ULL);
    SplitMix64 rng(g_state ^ 0xD4D4ULL);
    int read_id = 1;
    for (int s = 0; s < 18; ++s) {
        OpList ops;
        int n = rng.uniform_int(1, 8);
        for (int i = 0; i < n; ++i) {
            int op = rng.uniform_int(0, 6);
            int k = rng.uniform_int(0, 20);
            int pg = rng.uniform_int(0, 5);
            if (op == CKT_OP_GET) ops.get(read_id++, k);
            else if (op == CKT_OP_PUT) ops.put(k, (int64_t)rng.next_u64());
            else if (op == CKT_OP_DELETE) ops.del(k);
            else if (op == CKT_OP_PIN_PAGE) ops.pin(pg);
            else if (op == CKT_OP_UNPIN_PAGE) ops.unpin(pg);
            else if (op == CKT_OP_SWEEP_TOMBSTONES) ops.sweep(rng.uniform_int(0, 3));
            else ops.replay(rng.uniform_int(0, 3));
        }
        sc.steps.push_back(make_step(sc.spec, s, ops.ot, ops.a0, ops.a1));
    }
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 5: page_size == slot_count (single page) + large neighborhood.
// ---------------------------------------------------------------------------
static Scenario make_single_page_scenario() {
    Scenario sc;
    sc.name = "single_page_big_nbhd";
    sc.spec = make_spec(/*slot*/12, /*page*/12, /*nbhd*/12, /*disp*/12,
                        /*stash*/4, /*tomb*/5, /*max_ops*/48, /*steps*/20,
                        0xdeadbeefcafef00dULL, 0xa5a5a5a5a5a5a5a5ULL);
    SplitMix64 rng(g_state ^ 0xE5E5ULL);
    int read_id = 1;
    for (int s = 0; s < 18; ++s) {
        OpList ops;
        if (s == 9) { sc.steps.push_back(make_step(sc.spec, s, {}, {}, {})); continue; }
        int n = rng.uniform_int(1, 10);
        for (int i = 0; i < n; ++i) {
            int op = rng.uniform_int(0, 6);
            int k = rng.uniform_int(0, 18);
            if (op == CKT_OP_GET) ops.get(read_id++, k);
            else if (op == CKT_OP_PUT) ops.put(k, (int64_t)rng.next_u64());
            else if (op == CKT_OP_DELETE) ops.del(k);
            else if (op == CKT_OP_PIN_PAGE) ops.pin(0);
            else if (op == CKT_OP_UNPIN_PAGE) ops.unpin(0);
            else if (op == CKT_OP_SWEEP_TOMBSTONES) ops.sweep(rng.uniform_int(0, 4));
            else ops.replay(rng.uniform_int(0, 4));
        }
        sc.steps.push_back(make_step(sc.spec, s, ops.ot, ops.a0, ops.a1));
    }
    return sc;
}

// ---------------------------------------------------------------------------
// Scenario 6: large stress, mixed ops, many steps.
// ---------------------------------------------------------------------------
static Scenario make_large_stress_scenario() {
    Scenario sc;
    sc.name = "large_stress_mixed";
    sc.spec = make_spec(/*slot*/64, /*page*/8, /*nbhd*/6, /*disp*/10,
                        /*stash*/16, /*tomb*/10, /*max_ops*/128, /*steps*/40,
                        0x9e3779b97f4a7c15ULL, 0xc2b2ae3d27d4eb4fULL);
    SplitMix64 rng(g_state ^ 0xF6F6ULL);
    int read_id = 1;
    for (int s = 0; s < 36; ++s) {
        OpList ops;
        if (s == 17) { sc.steps.push_back(make_step(sc.spec, s, {}, {}, {})); continue; }
        int n = rng.uniform_int(4, 40);
        for (int i = 0; i < n; ++i) {
            int op = rng.uniform_int(0, 9);  // bias toward put/get/del
            int k = rng.uniform_int(0, 80);
            int pg = rng.uniform_int(0, 8);
            if (op <= 2) ops.put(k, (int64_t)rng.next_u64());
            else if (op <= 4) ops.get(read_id++, k);
            else if (op <= 6) ops.del(k);
            else if (op == 7) ops.pin(pg);
            else if (op == 8) ops.unpin(pg);
            else ops.replay(rng.uniform_int(0, 5));
        }
        // occasional sweeps
        if (s % 6 == 0) ops.sweep(rng.uniform_int(0, 5));
        sc.steps.push_back(make_step(sc.spec, s, ops.ot, ops.a0, ops.a1));
    }
    return sc;
}

static bool check_input_unchanged(const StepHost& step,
                                  const DeviceBuffer<int32_t>& d_ot,
                                  const DeviceBuffer<int64_t>& d_a0,
                                  const DeviceBuffer<int64_t>& d_a1,
                                  std::string* error) {
    if (d_ot.download() != step.op_type) { if (error) *error = "input op_type modified"; return false; }
    if (d_a0.download() != step.a0) { if (error) *error = "input a0 modified"; return false; }
    if (d_a1.download() != step.a1) { if (error) *error = "input a1 modified"; return false; }
    return true;
}

static bool run_one_step(const CktProblemSpec& spec, const StepHost& step,
                         CktOracleState* oracle, void* state, void* workspace,
                         size_t workspace_bytes, cudaStream_t stream,
                         StepResult* result, std::string* error) {
    DeviceBuffer<int32_t> d_ot;
    DeviceBuffer<int64_t> d_a0, d_a1;
    d_ot.allocate(step.op_type.size());
    d_a0.allocate(step.a0.size());
    d_a1.allocate(step.a1.size());
    d_ot.upload(step.op_type);
    d_a0.upload(step.a0);
    d_a1.upload(step.a1);

    GuardedDeviceBuffer<int32_t> d_counts;
    GuardedDeviceBuffer<uint64_t> d_ev_h, d_rd_h, d_slot_h, d_stash_h, d_page_h;
    d_counts.allocate(CKT_NUM_COUNTS);
    d_ev_h.allocate(1);
    d_rd_h.allocate(1);
    d_slot_h.allocate(1);
    d_stash_h.allocate(1);
    d_page_h.allocate(1);

    CktInputs inputs = {};
    inputs.op_type = d_ot.ptr;
    inputs.a0 = d_a0.ptr;
    inputs.a1 = d_a1.ptr;

    CktOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.op_event_hash = d_ev_h.ptr;
    outputs.read_hash = d_rd_h.ptr;
    outputs.slot_state_hash = d_slot_h.ptr;
    outputs.stash_hash = d_stash_h.ptr;
    outputs.page_hash = d_page_h.ptr;

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(step, d_ot, d_a0, d_a1, error)) return false;

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_ev_h.check_guards("op_event_hash", error)) return false;
    if (!d_rd_h.check_guards("read_hash", error)) return false;
    if (!d_slot_h.check_guards("slot_state_hash", error)) return false;
    if (!d_stash_h.check_guards("stash_hash", error)) return false;
    if (!d_page_h.check_guards("page_hash", error)) return false;

    std::vector<int32_t> h_counts = d_counts.download_data();
    std::vector<uint64_t> h_ev = d_ev_h.download_data();
    std::vector<uint64_t> h_rd = d_rd_h.download_data();
    std::vector<uint64_t> h_slot = d_slot_h.download_data();
    std::vector<uint64_t> h_stash = d_stash_h.download_data();
    std::vector<uint64_t> h_page = d_page_h.download_data();

    CktHostInputsView hin = {};
    hin.op_type = step.op_type.data();
    hin.a0 = step.a0.data();
    hin.a1 = step.a1.data();

    CktExpected expected;
    oracle->step_once(step.run, hin, &expected);

    CktHostOutputsView got = {};
    got.counts = h_counts.data();
    got.op_event_hash = h_ev.data();
    got.read_hash = h_rd.data();
    got.slot_state_hash = h_slot.data();
    got.stash_hash = h_stash.data();
    got.page_hash = h_page.data();

    if (!ckt_check_all_outputs(expected, got, error)) return false;

    if (result) {
        for (int i = 0; i < CKT_NUM_COUNTS; ++i) result->counts[i] = h_counts[i];
        result->op_event_hash = h_ev[0];
        result->read_hash = h_rd[0];
        result->slot_state_hash = h_slot[0];
        result->stash_hash = h_stash[0];
        result->page_hash = h_page[0];
    }
    return true;
}

static bool run_scenario_once(const Scenario& sc, bool verbose,
                              std::vector<StepResult>* results, int* passed_steps,
                              int* total_steps, std::string* first_error) {
    // CRITICAL: clamp workspace to >= 1; never fail-guard on zero.
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    CktOracleState oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.steps.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult result;
        std::string error;
        const bool ok = run_one_step(sc.spec, sc.steps[i], &oracle, state,
                                     workspace.ptr, std::max<size_t>(workspace_bytes, 1),
                                     stream, results ? &result : nullptr, &error);
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
            std::printf("scenario %-28s step %02zu/%02zu ops=%d %s%s%s\n",
                        sc.name.c_str(), i, sc.steps.size(),
                        sc.steps[i].run.num_ops, ok ? "PASS" : "FAIL",
                        ok ? "" : "  ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(const std::vector<StepResult>& a,
                            const std::vector<StepResult>& b, std::string* error) {
    if (a.size() != b.size()) { if (error) *error = "result length mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        bool counts_eq = true;
        for (int j = 0; j < CKT_NUM_COUNTS; ++j)
            if (a[i].counts[j] != b[i].counts[j]) counts_eq = false;
        if (!counts_eq ||
            a[i].op_event_hash != b[i].op_event_hash ||
            a[i].read_hash != b[i].read_hash ||
            a[i].slot_state_hash != b[i].slot_state_hash ||
            a[i].stash_hash != b[i].stash_hash ||
            a[i].page_hash != b[i].page_hash) {
            if (error) {
                std::ostringstream oss;
                oss << "replay mismatch at step " << i << ": slot a=0x" << std::hex
                    << a[i].slot_state_hash << ", b=0x" << b[i].slot_state_hash;
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
        scenarios.push_back(make_displacement_scenario());
        scenarios.push_back(make_pin_stash_replay_scenario());
        scenarios.push_back(make_tombstone_sweep_scenario());
        scenarios.push_back(make_nbhd1_scenario());
        scenarios.push_back(make_single_page_scenario());
        scenarios.push_back(make_large_stress_scenario());

        int passed = 0, total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results, replay_results;
            std::string error;

            const bool ok_base = run_scenario_once(sc, true, &base_results, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay_results, &passed, &total, &error);

            if (ok_base && ok_replay) {
                std::string cerr;
                if (compare_results(base_results, replay_results, &cerr))
                    std::printf("scenario %-28s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-28s exact replay FAIL  %s\n", sc.name.c_str(), cerr.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-28s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
