// file: test_stm_commit_arbiter.cu

#include "stm_commit_arbiter_common.h"
#include "stm_commit_arbiter_oracle.hpp"

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

static constexpr uint64_t g_state = 0x9b71c4d3a02f5e68ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                              \
            std::ostringstream _oss;                                            \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "    \
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
    int64_t next_i64() { return (int64_t)next_u64(); }
    int uniform_int(int lo, int hi) {
        const uint64_t span = (uint64_t)(hi - lo + 1);
        return lo + (int)(next_u64() % span);
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
        if (host.size() != count) throw std::runtime_error("DeviceBuffer::upload size mismatch");
        if (count != 0)
            CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count != 0)
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost));
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
        count = n; data_bytes = sizeof(T) * count;
        total_bytes = kGuardBytes + data_bytes + kGuardBytes;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), total_bytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, total_bytes));
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }
    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count != 0)
            CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> before(kGuardBytes), after(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(before.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (before[i] != kGuardByte) { if (error) { std::ostringstream o; o << "left guard corrupted for " << name << " at byte " << i; *error = o.str(); } return false; }
            if (after[i] != kGuardByte) { if (error) { std::ostringstream o; o << "right guard corrupted for " << name << " at byte " << i; *error = o.str(); } return false; }
        }
        return true;
    }
};

// Host op descriptor.
struct Op {
    int32_t kind; uint64_t txn; uint64_t read_id; uint64_t addr; int64_t value;
    uint64_t aux; std::vector<uint64_t> watch;
};
static Op OBEGIN(uint64_t t, uint64_t pri)            { Op o{STM_OP_BEGIN, t, 0, 0, 0, pri, {}}; return o; }
static Op OREAD(uint64_t t, uint64_t rid, uint64_t a) { Op o{STM_OP_TX_READ, t, rid, a, 0, 0, {}}; return o; }
static Op OWRITE(uint64_t t, uint64_t a, int64_t v)   { Op o{STM_OP_TX_WRITE, t, 0, a, v, 0, {}}; return o; }
static Op OVALIDATE(uint64_t t)                       { Op o{STM_OP_VALIDATE, t, 0, 0, 0, 0, {}}; return o; }
static Op OPREPARE(uint64_t t)                        { Op o{STM_OP_TRY_PREPARE, t, 0, 0, 0, 0, {}}; return o; }
static Op ODRAIN(uint64_t limit)                      { Op o{STM_OP_DRAIN_COMMITS, 0, 0, 0, 0, limit, {}}; return o; }
static Op ORETRY(uint64_t t, std::vector<uint64_t> w) { Op o{STM_OP_RETRY, t, 0, 0, 0, (uint64_t)w.size(), w}; return o; }
static Op ONONTX(uint64_t a, int64_t v)               { Op o{STM_OP_NON_TX_WRITE, 0, 0, a, v, 0, {}}; return o; }
static Op OABORT(uint64_t t)                          { Op o{STM_OP_ABORT, t, 0, 0, 0, 0, {}}; return o; }

struct StepHost {
    StmRunSpec run;
    std::vector<int32_t> op_kind;
    std::vector<uint64_t> txn_id, read_id, addr;
    std::vector<int64_t> value;
    std::vector<uint64_t> aux, watch_off, watch_addrs;
};

struct StepResult {
    int32_t counts[23] = {0};
    uint64_t stm_event_hash = 0, read_result_hash = 0, location_hash = 0, txn_hash = 0, queue_hash = 0;
};

struct Scenario {
    std::string name;
    StmProblemSpec spec;
    std::vector<StepHost> steps;
};

static StmProblemSpec make_spec(int mt, int ml, int mr, int mw, int mwatch,
                                int mwait, int mretry, int mbatch, int msteps) {
    StmProblemSpec spec = {};
    spec.abi_version = STM_ABI_VERSION;
    spec.max_txns = mt; spec.max_locations = ml; spec.max_read_set = mr;
    spec.max_write_set = mw; spec.max_watch_set = mwatch;
    spec.max_waiters_per_location = mwait; spec.max_retry_watchers_per_location = mretry;
    spec.max_batch = mbatch; spec.max_steps = msteps; spec.flags = 0;
    if (!stm_validate_problem_spec(&spec)) throw std::runtime_error("invalid StmProblemSpec");
    return spec;
}

static StepHost make_step(const StmProblemSpec& spec, int step_id, const std::vector<Op>& ops) {
    StepHost s;
    s.run = {};
    s.run.abi_version = STM_ABI_VERSION;
    s.run.batch_size = (int32_t)ops.size();
    s.run.step_id = step_id;
    if (!stm_validate_run_spec(&s.run, &spec)) throw std::runtime_error("invalid StmRunSpec");
    const size_t rows = std::max<size_t>(1, ops.size());
    s.op_kind.assign(rows, 0); s.txn_id.assign(rows, 0); s.read_id.assign(rows, 0);
    s.addr.assign(rows, 0); s.value.assign(rows, 0); s.aux.assign(rows, 0);
    s.watch_off.assign(rows, 0);
    s.watch_addrs.clear();
    for (size_t i = 0; i < ops.size(); ++i) {
        s.op_kind[i] = ops[i].kind;
        s.txn_id[i] = ops[i].txn;
        s.read_id[i] = ops[i].read_id;
        s.addr[i] = ops[i].addr;
        s.value[i] = ops[i].value;
        s.aux[i] = ops[i].aux;
        s.watch_off[i] = (uint64_t)s.watch_addrs.size();
        for (uint64_t w : ops[i].watch) s.watch_addrs.push_back(w);
    }
    if (s.watch_addrs.empty()) s.watch_addrs.push_back(0);  // never zero-size buffer
    return s;
}

// ---- Scenarios ----

// 1: basic begin/read/write/validate/prepare/drain commit path + read-own-write.
static Scenario sc_basic() {
    Scenario sc; sc.name = "basic_commit_path";
    sc.spec = make_spec(16, 64, 32, 32, 16, 8, 8, 64, 16);
    sc.steps.push_back(make_step(sc.spec, 0, {
        OBEGIN(1, 5),
        OWRITE(1, 10, 100), OWRITE(1, 20, 200),
        OREAD(1, 1, 10),               // read-own-write -> 100
        OREAD(1, 2, 30),               // shared, absent -> 0/v0
        OVALIDATE(1),
        OPREPARE(1),                   // locks 10,20; prepared
        ODRAIN(1),                     // commit
    }));
    sc.steps.push_back(make_step(sc.spec, 1, {
        OBEGIN(2, 1),
        OREAD(2, 10, 10),              // shared -> 100 v1
        OREAD(2, 11, 20),              // shared -> 200 v1
        OREAD(2, 12, 30),              // absent -> 0 v0
        OVALIDATE(2),
        OPREPARE(2),                   // read-only commit
    }));
    return sc;
}

// 2: lock conflict -> wait -> partial unlock; second txn drains and wakes waiter.
static Scenario sc_lock_wait() {
    Scenario sc; sc.name = "lock_wait_and_wake";
    sc.spec = make_spec(16, 64, 32, 32, 16, 8, 8, 64, 16);
    sc.steps.push_back(make_step(sc.spec, 0, {
        OBEGIN(1, 3), OBEGIN(2, 3),
        OWRITE(1, 5, 50), OWRITE(1, 7, 70),
        OWRITE(2, 7, 77), OWRITE(2, 9, 99),
        OPREPARE(1),     // locks 5,7 -> prepared
        OPREPARE(2),     // locks 9? ascending: 7 first (locked by 1) -> wait on 7,
                         // release 9 partial; T2 WAITING_LOCK
    }));
    sc.steps.push_back(make_step(sc.spec, 1, {
        ODRAIN(1),       // commit T1: writes 5,7; unlock; wake waiter T2 on 7
    }));
    sc.steps.push_back(make_step(sc.spec, 2, {
        OPREPARE(2),     // now T2 ACTIVE again; re-prepare: locks 7 then 9
        ODRAIN(1),       // commit T2
    }));
    return sc;
}

// 3: read-lock-conflict abort + read-version-conflict abort.
static Scenario sc_read_aborts() {
    Scenario sc; sc.name = "read_conflict_aborts";
    sc.spec = make_spec(16, 64, 32, 32, 16, 8, 8, 64, 16);
    sc.steps.push_back(make_step(sc.spec, 0, {
        OBEGIN(1, 1), OWRITE(1, 3, 33), OPREPARE(1),  // T1 holds lock on 3
        OBEGIN(2, 1),
        OREAD(2, 1, 3),    // addr 3 locked by T1 -> READ_LOCK_CONFLICT abort T2
    }));
    sc.steps.push_back(make_step(sc.spec, 1, {
        ODRAIN(1),         // commit T1 -> addr3 version=1
        OBEGIN(3, 1),      // start_version = global_version (1 now)
        OREAD(3, 2, 3),    // version 1 <= start 1, OK shared
    }));
    sc.steps.push_back(make_step(sc.spec, 2, {
        OBEGIN(4, 1),      // start_version = 1
        ONONTX(3, 999),    // non-tx write addr3 -> version=2 (> start 1)
        OREAD(4, 3, 3),    // version 2 > start_version 1 -> READ_VERSION_CONFLICT abort
    }));
    return sc;
}

// 4: retry suspend + wake on commit; retry-immediate on validation failure.
static Scenario sc_retry() {
    Scenario sc; sc.name = "retry_watch_and_wake";
    sc.spec = make_spec(16, 64, 32, 32, 16, 8, 8, 64, 16);
    // seed addr 8 = 80 (committed), addr 12 = 120
    sc.steps.push_back(make_step(sc.spec, 0, {
        OBEGIN(1, 1), OWRITE(1, 8, 80), OWRITE(1, 12, 120), OPREPARE(1), ODRAIN(1),
    }));
    // T2 reads 8, then RETRY watching {8,12}: validation OK -> suspend.
    sc.steps.push_back(make_step(sc.spec, 1, {
        OBEGIN(2, 1),
        OREAD(2, 1, 8),    // shared 80 v1
        ORETRY(2, {8, 12, 8}),  // dedup -> {8,12}; suspend
    }));
    // T3 writes addr 8, commits -> wakes T2 (retry watcher).
    sc.steps.push_back(make_step(sc.spec, 2, {
        OBEGIN(3, 1), OWRITE(3, 8, 800), OPREPARE(3), ODRAIN(1),
    }));
    // T2 now ACTIVE again with cleared sets, attempt_no incremented.
    // RETRY-immediate: T4 reads addr 8 (now v2), then a non-tx write bumps it,
    // RETRY sees version mismatch -> immediate.
    sc.steps.push_back(make_step(sc.spec, 3, {
        OBEGIN(4, 1),
        OREAD(4, 1, 8),     // shared v2
        ONONTX(8, 8888),    // version bumps to 3
        ORETRY(4, {8}),     // validation fails (v3 != read v2) -> RETRY_IMMEDIATE
        OABORT(2), OABORT(4),
    }));
    return sc;
}

// 5: prepared arbitration order (priority desc, prepare_seq asc, txn_id asc) +
//    multi-commit drain + retry watch overflow.
static Scenario sc_arbitration() {
    Scenario sc; sc.name = "arbitration_and_overflow";
    // small retry watcher cap to force overflow.
    sc.spec = make_spec(16, 64, 32, 32, 16, 8, 2, 64, 16);
    // Prepare three txns on disjoint addrs with differing priority.
    sc.steps.push_back(make_step(sc.spec, 0, {
        OBEGIN(1, 5), OWRITE(1, 1, 1), OPREPARE(1),   // prio5, pseq small
        OBEGIN(2, 9), OWRITE(2, 2, 2), OPREPARE(2),   // prio9 (highest)
        OBEGIN(3, 5), OWRITE(3, 3, 3), OPREPARE(3),   // prio5, larger pseq
    }));
    // Drain all 3: order should be T2 (prio9), then T1 (prio5,smaller pseq), then T3.
    sc.steps.push_back(make_step(sc.spec, 1, {
        ODRAIN(10),
    }));
    // retry watch overflow: cap=2. Three txns watch same addr 50.
    sc.steps.push_back(make_step(sc.spec, 2, {
        OBEGIN(10, 1), OREAD(10, 1, 50), ORETRY(10, {50}),
        OBEGIN(11, 1), OREAD(11, 1, 50), ORETRY(11, {50}),
        OBEGIN(12, 1), OREAD(12, 1, 50), ORETRY(12, {50}),  // overflow on 50 (cap 2)
    }));
    return sc;
}

// 6: PREPARE_OOM, LOCK_WAIT_OVERFLOW, NON_TX_STALL, NON_TX_OOM, explicit abort,
//    invalids.
static Scenario sc_limits() {
    Scenario sc; sc.name = "limits_and_invalids";
    // max_locations 3 to force OOM; max_waiters 1 to force wait overflow.
    sc.spec = make_spec(16, 3, 32, 32, 16, 1, 4, 64, 16);
    // Fill location table with 3 addrs.
    sc.steps.push_back(make_step(sc.spec, 0, {
        ONONTX(1, 10), ONONTX(2, 20), ONONTX(3, 30),  // 3 locations materialized
    }));
    // PREPARE_OOM: txn writes a NEW addr 4 -> no room.
    sc.steps.push_back(make_step(sc.spec, 1, {
        OBEGIN(1, 1), OWRITE(1, 4, 40), OPREPARE(1),  // PREPARE_OOM abort
    }));
    // NON_TX_STALL: lock addr1 via prepared txn, then non-tx write addr1.
    sc.steps.push_back(make_step(sc.spec, 2, {
        OBEGIN(2, 1), OWRITE(2, 1, 111), OPREPARE(2),  // locks addr1
        ONONTX(1, 999),   // stall: locked
        ONONTX(4, 40),    // OOM: absent + table full (3 used)
    }));
    // LOCK_WAIT_OVERFLOW: addr1 still locked by T2. Two more txns wait; cap=1.
    sc.steps.push_back(make_step(sc.spec, 3, {
        OBEGIN(3, 1), OWRITE(3, 1, 222), OPREPARE(3),  // wait on addr1 (queue size1)
        OBEGIN(4, 1), OWRITE(4, 1, 333), OPREPARE(4),  // wait overflow -> abort
    }));
    // explicit abort releases T2 lock and wakes T3 waiter.
    sc.steps.push_back(make_step(sc.spec, 4, {
        OABORT(2),     // releases addr1 lock, wakes T3
        OABORT(999),   // invalid (absent)
        OREAD(2, 1, 1),// invalid (T2 gone)
        ODRAIN(0),     // valid no-op
    }));
    return sc;
}

// 7: adversarial random mix (deterministic seed) exercising all paths.
static Scenario sc_random() {
    Scenario sc; sc.name = "random_mixed";
    sc.spec = make_spec(8, 16, 12, 8, 8, 4, 4, 64, 40);
    SplitMix64 rng(g_state ^ 0xfa11beefULL);
    int next_txn = 1;
    sc.steps.push_back(make_step(sc.spec, 0, {}));  // empty batch
    for (int s = 1; s < 36; ++s) {
        if (s % 9 == 0) { sc.steps.push_back(make_step(sc.spec, s, {})); continue; }
        std::vector<Op> ops;
        int nops = rng.uniform_int(2, 9);
        for (int i = 0; i < nops; ++i) {
            int pick = rng.uniform_int(0, 11);
            uint64_t t = (uint64_t)(next_txn - rng.uniform_int(0, 3));
            if ((int64_t)t < 1) t = 1;
            uint64_t addr = (uint64_t)rng.uniform_int(0, 7);
            int64_t val = rng.next_i64();
            if (pick == 0) { ops.push_back(OBEGIN((uint64_t)next_txn, (uint64_t)rng.uniform_int(0, 9))); next_txn++; }
            else if (pick == 1) ops.push_back(OREAD(t, (uint64_t)i, addr));
            else if (pick == 2) ops.push_back(OWRITE(t, addr, val));
            else if (pick == 3) ops.push_back(OVALIDATE(t));
            else if (pick == 4) ops.push_back(OPREPARE(t));
            else if (pick == 5) ops.push_back(ODRAIN((uint64_t)rng.uniform_int(0, 3)));
            else if (pick == 6) {
                std::vector<uint64_t> w;
                int nw = rng.uniform_int(1, 3);
                for (int k = 0; k < nw; ++k) w.push_back((uint64_t)rng.uniform_int(0, 7));
                ops.push_back(ORETRY(t, w));
            }
            else if (pick == 7) ops.push_back(ONONTX(addr, val));
            else if (pick == 8) ops.push_back(OABORT(t));
            else if (pick == 9) ops.push_back(OWRITE(t, addr, val));
            else if (pick == 10) ops.push_back(OREAD(t, (uint64_t)i, addr));
            else ops.push_back(OPREPARE(t));
        }
        sc.steps.push_back(make_step(sc.spec, s, ops));
    }
    return sc;
}

static bool check_input_unchanged(const StepHost& step,
    const DeviceBuffer<int32_t>& d_op, const DeviceBuffer<uint64_t>& d_txn,
    const DeviceBuffer<uint64_t>& d_rid, const DeviceBuffer<uint64_t>& d_addr,
    const DeviceBuffer<int64_t>& d_val, const DeviceBuffer<uint64_t>& d_aux,
    const DeviceBuffer<uint64_t>& d_woff, const DeviceBuffer<uint64_t>& d_watch,
    std::string* error) {
    if (d_op.download() != step.op_kind) { if (error) *error = "input op_kind modified"; return false; }
    if (d_txn.download() != step.txn_id) { if (error) *error = "input txn_id modified"; return false; }
    if (d_rid.download() != step.read_id) { if (error) *error = "input read_id modified"; return false; }
    if (d_addr.download() != step.addr) { if (error) *error = "input addr modified"; return false; }
    if (d_val.download() != step.value) { if (error) *error = "input value modified"; return false; }
    if (d_aux.download() != step.aux) { if (error) *error = "input aux modified"; return false; }
    if (d_woff.download() != step.watch_off) { if (error) *error = "input watch_off modified"; return false; }
    if (d_watch.download() != step.watch_addrs) { if (error) *error = "input watch_addrs modified"; return false; }
    return true;
}

static const char* kCountNames[23] = {
    "txn_begun","read_own_write","read_shared","write_staged","validate_ok",
    "readonly_commits","txn_prepared","write_locks","wait_locks","commits_done",
    "location_writes","non_tx_writes","non_tx_stalls","retry_suspended",
    "retry_immediate","retry_watch_overflow","wake_retry","wake_lock","aborts",
    "partial_unlocks","commit_unlocks","abort_unlocks","invalid_count"};

static bool run_one_step(const StmProblemSpec& spec, const StepHost& step,
    StmOracleState* oracle, void* state, void* workspace, size_t workspace_bytes,
    cudaStream_t stream, StepResult* result, std::string* error) {

    DeviceBuffer<int32_t> d_op; DeviceBuffer<uint64_t> d_txn, d_rid, d_addr, d_aux, d_woff, d_watch;
    DeviceBuffer<int64_t> d_val;
    d_op.allocate(step.op_kind.size()); d_txn.allocate(step.txn_id.size());
    d_rid.allocate(step.read_id.size()); d_addr.allocate(step.addr.size());
    d_val.allocate(step.value.size()); d_aux.allocate(step.aux.size());
    d_woff.allocate(step.watch_off.size()); d_watch.allocate(step.watch_addrs.size());
    d_op.upload(step.op_kind); d_txn.upload(step.txn_id); d_rid.upload(step.read_id);
    d_addr.upload(step.addr); d_val.upload(step.value); d_aux.upload(step.aux);
    d_woff.upload(step.watch_off); d_watch.upload(step.watch_addrs);

    GuardedDeviceBuffer<int32_t> g_counts[23];
    for (int i = 0; i < 23; ++i) g_counts[i].allocate(1);
    GuardedDeviceBuffer<uint64_t> g_evt, g_read, g_loc, g_txn, g_queue;
    g_evt.allocate(1); g_read.allocate(1); g_loc.allocate(1); g_txn.allocate(1); g_queue.allocate(1);

    StmInputs inputs = {};
    inputs.op_kind = d_op.ptr; inputs.txn_id = d_txn.ptr; inputs.read_id = d_rid.ptr;
    inputs.addr = d_addr.ptr; inputs.value = d_val.ptr; inputs.aux = d_aux.ptr;
    inputs.watch_off = d_woff.ptr; inputs.watch_addrs = d_watch.ptr;

    StmOutputs outputs = {};
    int32_t** ocount = (int32_t**)&outputs;  // first 23 fields are int32_t*
    for (int i = 0; i < 23; ++i) ocount[i] = g_counts[i].ptr;
    outputs.stm_event_hash = g_evt.ptr; outputs.read_result_hash = g_read.ptr;
    outputs.location_hash = g_loc.ptr; outputs.txn_hash = g_txn.ptr; outputs.queue_hash = g_queue.ptr;

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(step, d_op, d_txn, d_rid, d_addr, d_val, d_aux, d_woff, d_watch, error))
        return false;

    for (int i = 0; i < 23; ++i) if (!g_counts[i].check_guards(kCountNames[i], error)) return false;
    if (!g_evt.check_guards("stm_event_hash", error)) return false;
    if (!g_read.check_guards("read_result_hash", error)) return false;
    if (!g_loc.check_guards("location_hash", error)) return false;
    if (!g_txn.check_guards("txn_hash", error)) return false;
    if (!g_queue.check_guards("queue_hash", error)) return false;

    std::vector<int32_t> hc[23];
    for (int i = 0; i < 23; ++i) hc[i] = g_counts[i].download_data();
    std::vector<uint64_t> h_evt = g_evt.download_data(), h_read = g_read.download_data(),
        h_loc = g_loc.download_data(), h_txn = g_txn.download_data(), h_queue = g_queue.download_data();

    StmHostInputsView hi = {};
    hi.op_kind = step.op_kind.data(); hi.txn_id = step.txn_id.data();
    hi.read_id = step.read_id.data(); hi.addr = step.addr.data();
    hi.value = step.value.data(); hi.aux = step.aux.data();
    hi.watch_off = step.watch_off.data(); hi.watch_addrs = step.watch_addrs.data();

    StmExpected expected;
    oracle->step_once(step.run, hi, &expected);

    StmHostOutputsView got = {};
    const int32_t** gcount = (const int32_t**)&got;
    for (int i = 0; i < 23; ++i) gcount[i] = hc[i].data();
    got.stm_event_hash = h_evt.data(); got.read_result_hash = h_read.data();
    got.location_hash = h_loc.data(); got.txn_hash = h_txn.data(); got.queue_hash = h_queue.data();

    if (!stm_check_all_outputs(expected, got, error)) return false;

    if (result) {
        for (int i = 0; i < 23; ++i) result->counts[i] = hc[i][0];
        result->stm_event_hash = h_evt[0]; result->read_result_hash = h_read[0];
        result->location_hash = h_loc[0]; result->txn_hash = h_txn[0]; result->queue_hash = h_queue[0];
    }
    return true;
}

static bool run_scenario_once(const Scenario& sc, bool verbose, std::vector<StepResult>* results,
    int* passed_steps, int* total_steps, std::string* first_error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);
    workspace_bytes = std::max<size_t>(workspace_bytes, 1);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(workspace_bytes);

    StmOracleState oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.steps.size()); }
    bool all_ok = true;
    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult result; std::string error;
        const bool ok = run_one_step(sc.spec, sc.steps[i], &oracle, state,
            workspace.ptr, workspace_bytes, stream, results ? &result : nullptr, &error);
        ++(*total_steps);
        if (ok) ++(*passed_steps);
        else { all_ok = false; if (first_error && first_error->empty()) {
            std::ostringstream o; o << sc.name << " step " << i << ": " << error; *first_error = o.str(); } }
        if (results) results->push_back(result);
        if (verbose)
            std::printf("scenario %-28s step %02zu/%02zu batch=%d %s%s%s\n",
                sc.name.c_str(), i, sc.steps.size(), sc.steps[i].run.batch_size,
                ok ? "PASS" : "FAIL", ok ? "" : "  ", ok ? "" : error.c_str());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(const std::vector<StepResult>& a, const std::vector<StepResult>& b, std::string* error) {
    if (a.size() != b.size()) { if (error) *error = "result length mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        bool eq = true;
        for (int k = 0; k < 23; ++k) if (a[i].counts[k] != b[i].counts[k]) eq = false;
        if (a[i].stm_event_hash != b[i].stm_event_hash || a[i].read_result_hash != b[i].read_result_hash ||
            a[i].location_hash != b[i].location_hash || a[i].txn_hash != b[i].txn_hash ||
            a[i].queue_hash != b[i].queue_hash) eq = false;
        if (!eq) { if (error) { std::ostringstream o; o << "replay mismatch at step " << i; *error = o.str(); } return false; }
    }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        std::vector<Scenario> scenarios;
        scenarios.push_back(sc_basic());
        scenarios.push_back(sc_lock_wait());
        scenarios.push_back(sc_read_aborts());
        scenarios.push_back(sc_retry());
        scenarios.push_back(sc_arbitration());
        scenarios.push_back(sc_limits());
        scenarios.push_back(sc_random());

        int passed = 0, total = 0;
        bool all_ok = true;
        long agg[23] = {0};
        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results, replay_results;
            std::string error;
            const bool ok_base = run_scenario_once(sc, true, &base_results, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay_results, &passed, &total, &error);
            for (const StepResult& r : base_results)
                for (int k = 0; k < 23; ++k) agg[k] += r.counts[k];
            if (ok_base && ok_replay) {
                std::string cmp_err;
                if (compare_results(base_results, replay_results, &cmp_err))
                    std::printf("scenario %-28s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-28s exact replay FAIL  %s\n", sc.name.c_str(), cmp_err.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-28s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }
        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
