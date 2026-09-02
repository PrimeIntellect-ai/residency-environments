// file: test_lock_manager_deadlock.cu

#include "lock_manager_deadlock_common.h"
#include "lock_manager_deadlock_oracle.hpp"

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

static constexpr uint64_t g_state = 0x9e3a17c5b240d8f1ULL;
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
        uint64_t z = (state += 0x8e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
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
        if (n > 0) CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n));
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
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
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

struct StepResult {
    std::vector<int64_t> counts;
    int32_t op_index = 0;
    uint64_t event_seq = 0, ev = 0, grant = 0, wait = 0, txn = 0, state = 0;
};

struct Scenario {
    std::string name;
    LmdProblemSpec spec;
    std::vector<LmdRunSpec> ops;
};

static LmdProblemSpec make_spec(int tables, int parts, int rows, int max_txns,
                                int max_locks, int max_waiters, int esc_thresh,
                                int max_dl_cycles) {
    LmdProblemSpec s = {};
    s.abi_version = LMD_ABI_VERSION;
    s.table_count = tables;
    s.partitions_per_table = parts;
    s.rows_per_partition = rows;
    s.max_txns = max_txns;
    s.max_locks = max_locks;
    s.max_waiters = max_waiters;
    s.escalation_threshold = esc_thresh;
    s.max_deadlock_cycles_per_detect = max_dl_cycles;
    s.flags = 0;
    if (!lmd_validate_problem_spec(&s)) throw std::runtime_error("invalid LmdProblemSpec");
    return s;
}

static LmdRunSpec mk(int op_kind, int step_id) {
    LmdRunSpec r = {};
    r.abi_version = LMD_ABI_VERSION;
    r.op_kind = op_kind;
    r.step_id = step_id;
    r.a_partition = -1;
    r.a_row = -1;
    return r;
}
static LmdRunSpec op_begin(int step, uint64_t txn, int prio) {
    LmdRunSpec r = mk(LMD_OP_BEGIN, step); r.a_txn = txn; r.a_priority = prio; return r;
}
static LmdRunSpec op_lock_table(int step, uint64_t txn, int t, int mode) {
    LmdRunSpec r = mk(LMD_OP_LOCK, step); r.a_txn = txn; r.a_res_kind = LMD_TABLE; r.a_table = t; r.a_partition = -1; r.a_row = -1; r.a_mode = mode; return r;
}
static LmdRunSpec op_lock_part(int step, uint64_t txn, int t, int p, int mode) {
    LmdRunSpec r = mk(LMD_OP_LOCK, step); r.a_txn = txn; r.a_res_kind = LMD_PARTITION; r.a_table = t; r.a_partition = p; r.a_row = -1; r.a_mode = mode; return r;
}
static LmdRunSpec op_lock_row(int step, uint64_t txn, int t, int p, int rw, int mode) {
    LmdRunSpec r = mk(LMD_OP_LOCK, step); r.a_txn = txn; r.a_res_kind = LMD_ROW; r.a_table = t; r.a_partition = p; r.a_row = rw; r.a_mode = mode; return r;
}
static LmdRunSpec op_unlock_table(int step, uint64_t txn, int t) {
    LmdRunSpec r = mk(LMD_OP_UNLOCK, step); r.a_txn = txn; r.a_res_kind = LMD_TABLE; r.a_table = t; r.a_partition = -1; r.a_row = -1; return r;
}
static LmdRunSpec op_unlock_part(int step, uint64_t txn, int t, int p) {
    LmdRunSpec r = mk(LMD_OP_UNLOCK, step); r.a_txn = txn; r.a_res_kind = LMD_PARTITION; r.a_table = t; r.a_partition = p; r.a_row = -1; return r;
}
static LmdRunSpec op_unlock_row(int step, uint64_t txn, int t, int p, int rw) {
    LmdRunSpec r = mk(LMD_OP_UNLOCK, step); r.a_txn = txn; r.a_res_kind = LMD_ROW; r.a_table = t; r.a_partition = p; r.a_row = rw; return r;
}
static LmdRunSpec op_unlock_all(int step, uint64_t txn) {
    LmdRunSpec r = mk(LMD_OP_UNLOCK_ALL, step); r.a_txn = txn; return r;
}
static LmdRunSpec op_convert_table(int step, uint64_t txn, int t, int mode) {
    LmdRunSpec r = mk(LMD_OP_CONVERT, step); r.a_txn = txn; r.a_res_kind = LMD_TABLE; r.a_table = t; r.a_partition = -1; r.a_row = -1; r.a_mode = mode; return r;
}
static LmdRunSpec op_convert_row(int step, uint64_t txn, int t, int p, int rw, int mode) {
    LmdRunSpec r = mk(LMD_OP_CONVERT, step); r.a_txn = txn; r.a_res_kind = LMD_ROW; r.a_table = t; r.a_partition = p; r.a_row = rw; r.a_mode = mode; return r;
}
static LmdRunSpec op_detect(int step, int limit) {
    LmdRunSpec r = mk(LMD_OP_DETECT_DEADLOCK, step); r.a_limit = limit; return r;
}

// ----------------------------------------------------------------- scenarios
// S1: compatibility matrix + intention plan + reenter + dominance.
static Scenario sc_compat_plan() {
    Scenario s; s.name = "compat_plan_reenter";
    s.spec = make_spec(2, 3, 4, 16, 256, 64, 1000000, 8);  // huge threshold -> no escalation
    int t = 0;
    s.ops.push_back(op_begin(t++, 1, 5));
    s.ops.push_back(op_begin(t++, 2, 5));
    // T1 locks ROW(0,0,0) S -> plan TABLE IS, PART IS, ROW S.
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 0, LMD_S));
    // T2 locks ROW(0,0,1) X -> TABLE IX, PART IX, ROW X (compatible: IS vs IX ok).
    s.ops.push_back(op_lock_row(t++, 2, 0, 0, 1, LMD_X));
    // T1 re-lock ROW(0,0,0) S again -> reenter (dominates).
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 0, LMD_S));
    // T1 lock ROW(0,0,0) IS -> dominated by S -> reenter.
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 0, LMD_IS));
    // T2 tries ROW(0,0,0) X -> incompatible with T1 S on that row -> WAIT.
    s.ops.push_back(op_lock_row(t++, 2, 0, 0, 0, LMD_X));
    // detect: T2 waits for T1, T1 not waiting -> no cycle.
    s.ops.push_back(op_detect(t++, 4));
    // T1 unlock its S three times (explicit_count built to 3) then release.
    s.ops.push_back(op_unlock_row(t++, 1, 0, 0, 0));
    s.ops.push_back(op_unlock_row(t++, 1, 0, 0, 0));
    s.ops.push_back(op_unlock_row(t++, 1, 0, 0, 0));  // now T2 wakes on ROW(0,0,0) maybe
    s.ops.push_back(op_detect(t++, 2));
    s.ops.push_back(op_unlock_all(t++, 1));
    s.ops.push_back(op_unlock_all(t++, 2));
    return s;
}

// S2: deadlock cycle + victim selection + abort cascade + wake.
static Scenario sc_deadlock() {
    Scenario s; s.name = "deadlock_victim_cascade";
    s.spec = make_spec(2, 2, 4, 16, 256, 64, 1000000, 8);
    int t = 0;
    s.ops.push_back(op_begin(t++, 1, 5));
    s.ops.push_back(op_begin(t++, 2, 3));  // lower priority -> victim candidate
    s.ops.push_back(op_begin(t++, 3, 5));
    // T1 X row(0,0,0), T2 X row(0,0,1).
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 0, LMD_X));
    s.ops.push_back(op_lock_row(t++, 2, 0, 0, 1, LMD_X));
    // T1 wants row(0,0,1) X -> blocked by T2.
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 1, LMD_X));
    // T2 wants row(0,0,0) X -> blocked by T1 -> cycle T1<->T2.
    s.ops.push_back(op_lock_row(t++, 2, 0, 0, 0, LMD_X));
    s.ops.push_back(op_detect(t++, 1));  // abort lower priority (T2)
    s.ops.push_back(op_detect(t++, 4));  // now no cycle
    s.ops.push_back(op_unlock_all(t++, 1));
    s.ops.push_back(op_unlock_all(t++, 3));
    return s;
}

// S3: automatic escalation (grant + release cascade) and escalate-blocked.
static Scenario sc_escalation() {
    Scenario s; s.name = "escalation_threshold";
    s.spec = make_spec(1, 2, 8, 16, 256, 64, 3, 8);  // threshold=3
    int t = 0;
    s.ops.push_back(op_begin(t++, 1, 5));
    s.ops.push_back(op_begin(t++, 2, 5));
    // T1 takes 3 X rows under table 0 -> reaches threshold -> escalate to X table.
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 0, LMD_X));
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 1, LMD_X));
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 2, LMD_X));  // 3rd -> escalate X
    // Now T1 holds X on TABLE 0. T2 tries S row -> blocked at TABLE (IS vs X incompatible).
    s.ops.push_back(op_lock_row(t++, 2, 0, 1, 0, LMD_S));
    s.ops.push_back(op_detect(t++, 2));  // T2 waits T1, no cycle
    // T1 unlock_all -> releases table X -> T2 wakes and continues plan.
    s.ops.push_back(op_unlock_all(t++, 1));
    s.ops.push_back(op_detect(t++, 2));
    s.ops.push_back(op_unlock_all(t++, 2));
    return s;
}

// S4: conversion waiting (upgrader keeps old mode) feeding a deadlock cycle.
static Scenario sc_convert_wait() {
    Scenario s; s.name = "convert_wait_cycle";
    s.spec = make_spec(1, 1, 4, 16, 256, 64, 1000000, 8);
    int t = 0;
    s.ops.push_back(op_begin(t++, 1, 5));
    s.ops.push_back(op_begin(t++, 2, 5));
    // both take S on row(0,0,0).
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 0, LMD_S));
    s.ops.push_back(op_lock_row(t++, 2, 0, 0, 0, LMD_S));
    // T1 convert row to X -> blocked by T2's S -> CONVERT_WAIT, keeps S.
    s.ops.push_back(op_convert_row(t++, 1, 0, 0, 0, LMD_X));
    // T2 convert row to X -> blocked by T1's S -> CONVERT_WAIT. Now cycle.
    s.ops.push_back(op_convert_row(t++, 2, 0, 0, 0, LMD_X));
    s.ops.push_back(op_detect(t++, 1));  // break cycle: victim by tie largest txn_id = T2
    s.ops.push_back(op_detect(t++, 2));
    // T1 convert row to IS -> dominated by its X -> noop.
    s.ops.push_back(op_convert_row(t++, 1, 0, 0, 0, LMD_IS));
    s.ops.push_back(op_unlock_all(t++, 1));
    return s;
}

// S5: invalid ops, convert-noop, unlock errors, begin dup/full.
static Scenario sc_invalid() {
    Scenario s; s.name = "invalid_and_noop";
    s.spec = make_spec(1, 1, 2, 2, 256, 64, 1000000, 8);  // max_txns=2
    int t = 0;
    s.ops.push_back(op_begin(t++, 1, 5));
    s.ops.push_back(op_begin(t++, 1, 5));   // duplicate -> invalid
    s.ops.push_back(op_begin(t++, 2, 5));
    s.ops.push_back(op_begin(t++, 3, 5));   // table full -> invalid
    s.ops.push_back(op_lock_row(t++, 9, 0, 0, 0, LMD_S));  // absent txn -> invalid
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 9, LMD_S));  // row oob -> invalid
    s.ops.push_back(op_lock_table(t++, 1, 5, LMD_S));      // table oob -> invalid
    s.ops.push_back(op_lock_table(t++, 1, 0, LMD_NL));     // mode NL invalid
    s.ops.push_back(op_lock_row(t++, 1, 0, 0, 0, LMD_S));  // ok
    s.ops.push_back(op_unlock_row(t++, 1, 0, 0, 1));       // not held -> invalid
    s.ops.push_back(op_unlock_table(t++, 2, 0));           // not held -> invalid
    s.ops.push_back(op_convert_row(t++, 1, 0, 0, 0, LMD_IS)); // dominated -> noop
    s.ops.push_back(op_convert_row(t++, 1, 0, 0, 0, LMD_X));  // upgrade ok (sole holder)
    s.ops.push_back(op_convert_table(t++, 2, 0, LMD_S));   // no grant -> invalid
    s.ops.push_back(op_detect(t++, 0));   // limit 0 no-op
    s.ops.push_back(op_detect(t++, 3));   // no cycle
    s.ops.push_back(op_unlock_all(t++, 1));
    s.ops.push_back(op_unlock_all(t++, 9));  // absent -> invalid
    return s;
}

// S6: three-way deadlock cycle + waiter requeue + intent ancestors as blockers.
static Scenario sc_three_cycle() {
    Scenario s; s.name = "three_way_cycle";
    s.spec = make_spec(1, 3, 4, 16, 256, 64, 1000000, 8);
    int t = 0;
    for (uint64_t id = 1; id <= 3; ++id) s.ops.push_back(op_begin(t++, id, 5));
    // T1 X part0, T2 X part1, T3 X part2 (each holds a partition X).
    s.ops.push_back(op_lock_part(t++, 1, 0, 0, LMD_X));
    s.ops.push_back(op_lock_part(t++, 2, 0, 1, LMD_X));
    s.ops.push_back(op_lock_part(t++, 3, 0, 2, LMD_X));
    // T1 wants part1 X -> blocked by T2; T2 wants part2 X -> blocked by T3;
    // T3 wants part0 X -> blocked by T1. 3-cycle.
    s.ops.push_back(op_lock_part(t++, 1, 0, 1, LMD_X));
    s.ops.push_back(op_lock_part(t++, 2, 0, 2, LMD_X));
    s.ops.push_back(op_lock_part(t++, 3, 0, 0, LMD_X));
    s.ops.push_back(op_detect(t++, 1));  // abort one victim
    s.ops.push_back(op_detect(t++, 3));  // resolve rest
    s.ops.push_back(op_unlock_all(t++, 1));
    s.ops.push_back(op_unlock_all(t++, 2));
    s.ops.push_back(op_unlock_all(t++, 3));
    return s;
}

// S7: large randomized stress over all operations.
static Scenario sc_random_stress() {
    Scenario s; s.name = "random_stress";
    s.spec = make_spec(3, 3, 6, 24, 1024, 256, 4, 16);
    SplitMix64 rng(g_state ^ 0xc0ffee1234ULL);
    int t = 0;
    for (uint64_t id = 1; id <= 12; ++id) s.ops.push_back(op_begin(t++, id, rng.uniform_int(0, 4)));
    auto rnd_mode = [&]() {
        int m[5] = {LMD_IS, LMD_IX, LMD_S, LMD_SIX, LMD_X};
        return m[rng.uniform_int(0, 4)];
    };
    for (int i = 0; i < 200; ++i) {
        int pick = rng.uniform_int(0, 99);
        uint64_t txn = (uint64_t)rng.uniform_int(1, 12);
        int tb = rng.uniform_int(0, 2);
        int pp = rng.uniform_int(0, 2);
        int rr = rng.uniform_int(0, 5);
        if (pick < 38) {
            int g = rng.uniform_int(0, 2);
            if (g == 0) s.ops.push_back(op_lock_row(t++, txn, tb, pp, rr, rnd_mode()));
            else if (g == 1) s.ops.push_back(op_lock_part(t++, txn, tb, pp, rnd_mode()));
            else s.ops.push_back(op_lock_table(t++, txn, tb, rnd_mode()));
        } else if (pick < 52) {
            int g = rng.uniform_int(0, 1);
            if (g == 0) s.ops.push_back(op_convert_row(t++, txn, tb, pp, rr, rnd_mode()));
            else s.ops.push_back(op_convert_table(t++, txn, tb, rnd_mode()));
        } else if (pick < 66) {
            int g = rng.uniform_int(0, 2);
            if (g == 0) s.ops.push_back(op_unlock_row(t++, txn, tb, pp, rr));
            else if (g == 1) s.ops.push_back(op_unlock_part(t++, txn, tb, pp));
            else s.ops.push_back(op_unlock_table(t++, txn, tb));
        } else if (pick < 74) {
            s.ops.push_back(op_unlock_all(t++, txn));
        } else if (pick < 86) {
            s.ops.push_back(op_detect(t++, rng.uniform_int(0, 6)));
        } else {
            s.ops.push_back(op_begin(t++, txn, rng.uniform_int(0, 4)));
        }
    }
    // drain with detects + unlock_all everything.
    for (int i = 0; i < 6; ++i) s.ops.push_back(op_detect(t++, 16));
    for (uint64_t id = 1; id <= 12; ++id) s.ops.push_back(op_unlock_all(t++, id));
    for (int i = 0; i < 4; ++i) s.ops.push_back(op_detect(t++, 16));
    return s;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_compat_plan());
    v.push_back(sc_deadlock());
    v.push_back(sc_escalation());
    v.push_back(sc_convert_wait());
    v.push_back(sc_invalid());
    v.push_back(sc_three_cycle());
    v.push_back(sc_random_stress());
    return v;
}

// ----------------------------------------------------------------- runner
static bool run_one_op(
    const LmdProblemSpec& spec, const LmdRunSpec& op, LmdOracle* oracle,
    void* state, void* workspace, size_t workspace_bytes, cudaStream_t stream,
    StepResult* result, std::string* error) {

    GuardedDeviceBuffer<int64_t> d_counts;
    GuardedDeviceBuffer<int32_t> d_opidx;
    GuardedDeviceBuffer<uint64_t> d_eseq, d_ev, d_grant, d_wait, d_txn, d_state;
    d_counts.allocate(LMD_COUNT_N);
    d_opidx.allocate(1);
    d_eseq.allocate(1); d_ev.allocate(1); d_grant.allocate(1);
    d_wait.allocate(1); d_txn.allocate(1); d_state.allocate(1);

    LmdOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.op_index_out = d_opidx.ptr;
    outputs.event_seq_out = d_eseq.ptr;
    outputs.lock_event_hash = d_ev.ptr;
    outputs.grant_hash = d_grant.ptr;
    outputs.wait_hash = d_wait.ptr;
    outputs.txn_lock_hash = d_txn.ptr;
    outputs.state_checksum = d_state.ptr;

    LmdRunSpec op_copy = op;  // verify immutability of the op spec
    LmdInputs inputs = {};
    inputs.reserved = nullptr;

    CUDA_CHECK(solution_run(state, &op_copy, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (std::memcmp(&op_copy, &op, sizeof(LmdRunSpec)) != 0) {
        if (error) *error = "run spec mutated by solution_run";
        return false;
    }

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_opidx.check_guards("op_index", error)) return false;
    if (!d_eseq.check_guards("event_seq", error)) return false;
    if (!d_ev.check_guards("lock_event_hash", error)) return false;
    if (!d_grant.check_guards("grant_hash", error)) return false;
    if (!d_wait.check_guards("wait_hash", error)) return false;
    if (!d_txn.check_guards("txn_lock_hash", error)) return false;
    if (!d_state.check_guards("state_checksum", error)) return false;

    const std::vector<int64_t> h_counts = d_counts.download_data();
    const std::vector<int32_t> h_opidx = d_opidx.download_data();
    const std::vector<uint64_t> h_eseq = d_eseq.download_data();
    const std::vector<uint64_t> h_ev = d_ev.download_data();
    const std::vector<uint64_t> h_grant = d_grant.download_data();
    const std::vector<uint64_t> h_wait = d_wait.download_data();
    const std::vector<uint64_t> h_txn = d_txn.download_data();
    const std::vector<uint64_t> h_state = d_state.download_data();

    LmdExpected expected;
    oracle->step_once(op, &expected);

    LmdHostOutputsView got = {};
    got.counts = h_counts.data();
    got.op_index_out = h_opidx.data();
    got.event_seq_out = h_eseq.data();
    got.lock_event_hash = h_ev.data();
    got.grant_hash = h_grant.data();
    got.wait_hash = h_wait.data();
    got.txn_lock_hash = h_txn.data();
    got.state_checksum = h_state.data();

    if (!lmd_check_outputs(expected, got, error)) return false;

    if (result) {
        result->counts = h_counts;
        result->op_index = h_opidx[0];
        result->event_seq = h_eseq[0];
        result->ev = h_ev[0];
        result->grant = h_grant[0];
        result->wait = h_wait[0];
        result->txn = h_txn[0];
        result->state = h_state[0];
    }
    return true;
}

static bool run_scenario_once(
    const Scenario& sc, bool verbose, std::vector<StepResult>* results,
    int* passed, int* total, std::string* first_error) {

    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    LmdOracle oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.ops.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.ops.size(); ++i) {
        StepResult result;
        std::string error;
        const bool ok = run_one_op(sc.spec, sc.ops[i], &oracle, state, workspace.ptr, workspace_bytes, stream,
                                   results ? &result : nullptr, &error);
        ++(*total);
        if (ok) ++(*passed);
        else { all_ok = false; if (first_error && first_error->empty()) { std::ostringstream o; o << sc.name << " op " << i << " (kind=" << sc.ops[i].op_kind << "): " << error; *first_error = o.str(); } }
        if (results) results->push_back(result);
        if (verbose && (!ok)) {
            const LmdRunSpec& o2 = sc.ops[i];
            std::printf("scenario %-26s op %04zu/%04zu kind=%d txn=%llu rk=%d t=%d p=%d r=%d mode=%d prio=%d lim=%d %s%s%s\n",
                        sc.name.c_str(), i, sc.ops.size(), o2.op_kind,
                        (unsigned long long)o2.a_txn, o2.a_res_kind, o2.a_table, o2.a_partition, o2.a_row,
                        o2.a_mode, o2.a_priority, o2.a_limit,
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
        if (a[i].counts != b[i].counts || a[i].op_index != b[i].op_index ||
            a[i].event_seq != b[i].event_seq || a[i].ev != b[i].ev ||
            a[i].grant != b[i].grant || a[i].wait != b[i].wait ||
            a[i].txn != b[i].txn || a[i].state != b[i].state) {
            if (error) { std::ostringstream o; o << "replay mismatch at op " << i << " state a=0x" << std::hex << a[i].state << " b=0x" << b[i].state; *error = o.str(); }
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
                if (compare_results(base_results, replay_results, &ce)) {
                    std::printf("scenario %-26s exact replay PASS (%zu ops)\n", sc.name.c_str(), sc.ops.size());
                } else { all_ok = false; std::printf("scenario %-26s exact replay FAIL  %s\n", sc.name.c_str(), ce.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-26s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
