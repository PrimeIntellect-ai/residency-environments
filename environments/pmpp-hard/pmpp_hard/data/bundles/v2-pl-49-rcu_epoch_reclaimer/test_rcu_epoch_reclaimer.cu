// file: test_rcu_epoch_reclaimer.cu

#include "rcu_epoch_reclaimer_common.h"
#include "rcu_epoch_reclaimer_oracle.hpp"

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

static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::ostringstream _oss;                                           \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "   \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                             \
        }                                                                       \
    } while (0)

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
        if (n > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
    }
    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("upload size mismatch");
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
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), kGuardBytes + data_bytes + kGuardBytes));
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
            if (left[i] != kGuardByte) {
                if (error) { std::ostringstream o; o << "left guard corrupted for " << name << " at byte " << i; *error = o.str(); }
                return false;
            }
            if (right[i] != kGuardByte) {
                if (error) { std::ostringstream o; o << "right guard corrupted for " << name << " at byte " << i; *error = o.str(); }
                return false;
            }
        }
        return true;
    }
};

// ---- scenario model ----

struct StepHost {
    std::vector<RcuOp> ops;
    int step_id = 0;
};

struct Scenario {
    std::string name;
    RcuProblemSpec spec{};
    std::vector<StepHost> steps;
};

struct StepResult {
    uint64_t counts[RCU_COUNT_TOTAL] = {0};
    uint64_t event_hash = 0, read_hash = 0, root_hash = 0;
    uint64_t object_hash = 0, thread_hash = 0, free_hash = 0;
    uint64_t scalars[6] = {0};
    bool operator==(const StepResult& o) const {
        for (int i = 0; i < RCU_COUNT_TOTAL; ++i) if (counts[i] != o.counts[i]) return false;
        if (event_hash != o.event_hash || read_hash != o.read_hash || root_hash != o.root_hash) return false;
        if (object_hash != o.object_hash || thread_hash != o.thread_hash || free_hash != o.free_hash) return false;
        for (int i = 0; i < 6; ++i) if (scalars[i] != o.scalars[i]) return false;
        return true;
    }
};

// ---- op builders ----
static RcuOp mk(int kind, int a = 0, int b = 0, int cc = 0, int64_t i64 = 0) {
    RcuOp op; op.kind = kind; op.arg_a = a; op.arg_b = b; op.arg_c = cc; op.arg_i64 = i64; return op;
}
static RcuOp OP_ALLOC(int req, int64_t val)         { return mk(RCU_OP_ALLOC, req, 0, 0, val); }
static RcuOp OP_SET_ROOT(int root, int obj)         { return mk(RCU_OP_SET_ROOT, root, obj); }
static RcuOp OP_LINK(int src, int slot, int dst)    { return mk(RCU_OP_LINK, src, slot, dst); }
static RcuOp OP_RLOCK(int t)                        { return mk(RCU_OP_READ_LOCK, t); }
static RcuOp OP_RUNLOCK(int t)                      { return mk(RCU_OP_READ_UNLOCK, t); }
static RcuOp OP_QUIESCE(int t)                      { return mk(RCU_OP_QUIESCE, t); }
static RcuOp OP_ADV()                               { return mk(RCU_OP_ADVANCE_EPOCH); }
static RcuOp OP_RCHAIN(int t, int root, int hops)   { return mk(RCU_OP_READ_CHAIN, t, root, hops); }
static RcuOp OP_RETIRE(int obj, int tag)            { return mk(RCU_OP_RETIRE, obj, tag); }
static RcuOp OP_RECLAIM(int limit)                  { return mk(RCU_OP_RECLAIM, limit); }
static RcuOp OP_DROP(int limit)                     { return mk(RCU_OP_FORCE_DROP_FREE_IDS, limit); }

static RcuProblemSpec make_spec(int threads, int roots, int max_obj, int edges,
                                int max_retired, int max_free, int max_ops) {
    RcuProblemSpec s{};
    s.abi_version = RCU_ABI_VERSION;
    s.thread_count = threads;
    s.root_count = roots;
    s.max_objects = max_obj;
    s.max_edges_per_object = edges;
    s.max_retired = max_retired;
    s.max_free_ids = max_free;
    s.max_ops = max_ops;
    return s;
}

// Scenario 1: basic lifecycle - alloc, link, root, retire, reclaim, reuse.
static Scenario sc_basic() {
    Scenario sc; sc.name = "basic_lifecycle";
    sc.spec = make_spec(4, 4, 64, 4, 64, 64, 256);
    StepHost s0;
    // alloc 5 objects (ids 1..5)
    for (int i = 0; i < 5; ++i) s0.ops.push_back(OP_ALLOC(100 + i, 1000 + i));
    s0.ops.push_back(OP_SET_ROOT(0, 1));
    s0.ops.push_back(OP_LINK(1, 0, 2));
    s0.ops.push_back(OP_LINK(2, 0, 3));
    s0.ops.push_back(OP_LINK(3, 0, 4));
    sc.steps.push_back(s0);

    StepHost s1;
    // reader t0 snapshots root, reads chain
    s1.ops.push_back(OP_RLOCK(0));
    s1.ops.push_back(OP_RCHAIN(0, 0, 10));
    // unlink 4 from 3, then retire 4 (now no incoming)
    s1.ops.push_back(OP_LINK(3, 0, 0));
    s1.ops.push_back(OP_RETIRE(4, 77));
    // reader still active: retire_seq(4)=1, t0 outer_read_seq=1 -> not < 1, eligible
    s1.ops.push_back(OP_RECLAIM(8));
    s1.ops.push_back(OP_RUNLOCK(0));
    sc.steps.push_back(s1);

    StepHost s2;
    // reuse: alloc should reuse id 4 (from free list)
    s2.ops.push_back(OP_ALLOC(200, 9999));
    s2.ops.push_back(OP_ADV());
    s2.ops.push_back(OP_QUIESCE(1));
    sc.steps.push_back(s2);
    return sc;
}

// Scenario 2: grace-period gating - a reader that started BEFORE a retire blocks it.
static Scenario sc_grace() {
    Scenario sc; sc.name = "grace_period_gating";
    sc.spec = make_spec(4, 2, 32, 3, 32, 32, 256);
    StepHost s0;
    for (int i = 0; i < 4; ++i) s0.ops.push_back(OP_ALLOC(i, 10 * i));  // ids 1..4
    s0.ops.push_back(OP_SET_ROOT(0, 1));
    s0.ops.push_back(OP_LINK(1, 0, 2));
    sc.steps.push_back(s0);

    StepHost s1;
    // t0 locks (outer_read_seq=1) BEFORE any retire
    s1.ops.push_back(OP_RLOCK(0));
    s1.ops.push_back(OP_RCHAIN(0, 0, 5));
    // now retire 3 and 4 (no incoming). retire_seq 3=1,4=2.
    s1.ops.push_back(OP_RETIRE(3, 1));
    s1.ops.push_back(OP_RETIRE(4, 2));
    // t0 active with outer_read_seq=1; eligibility: no thread with nest>0 && outer<R.
    // For R=1: 1<1 false -> eligible. For R=2: 1<2 true -> NOT eligible.
    // So reclaim should reclaim obj3 (R=1) but skip obj4 (R=2).
    s1.ops.push_back(OP_RECLAIM(8));
    sc.steps.push_back(s1);

    StepHost s2;
    // t0 unlocks -> now no active blocking reader. reclaim obj4.
    s2.ops.push_back(OP_RUNLOCK(0));
    s2.ops.push_back(OP_RECLAIM(8));
    sc.steps.push_back(s2);
    return sc;
}

// Scenario 3: nested locks + stale snapshot reads through retired objects.
static Scenario sc_nested_snapshot() {
    Scenario sc; sc.name = "nested_snapshot_reads";
    sc.spec = make_spec(3, 3, 32, 2, 32, 32, 256);
    StepHost s0;
    for (int i = 0; i < 5; ++i) s0.ops.push_back(OP_ALLOC(i, 100 + i));  // 1..5
    s0.ops.push_back(OP_SET_ROOT(0, 1));
    s0.ops.push_back(OP_LINK(1, 0, 2));
    s0.ops.push_back(OP_LINK(2, 0, 3));
    sc.steps.push_back(s0);

    StepHost s1;
    s1.ops.push_back(OP_RLOCK(0));               // outer snapshot root0->1
    s1.ops.push_back(OP_RLOCK(0));               // nested, no refresh
    // publisher changes root to 5 AFTER snapshot
    s1.ops.push_back(OP_SET_ROOT(0, 5));
    // reader still sees old snapshot (1->2->3)
    s1.ops.push_back(OP_RCHAIN(0, 0, 10));
    // unlink chain so 3 has no incoming, retire 3 (readable through snapshot)
    s1.ops.push_back(OP_LINK(2, 0, 0));
    s1.ops.push_back(OP_RETIRE(3, 33));
    // retired-but-unreclaimed: reader can still read it via stale snapshot
    s1.ops.push_back(OP_RCHAIN(0, 0, 10));
    s1.ops.push_back(OP_RUNLOCK(0));             // nested -> NESTED event
    s1.ops.push_back(OP_RUNLOCK(0));             // outer -> QUIESCENT
    sc.steps.push_back(s1);

    StepHost s2;
    // now reclaim 3 (no active reader). Then a new reader sees absent boundary.
    s2.ops.push_back(OP_RECLAIM(4));
    s2.ops.push_back(OP_RLOCK(1));
    s2.ops.push_back(OP_RCHAIN(1, 0, 10));       // root0->5 now, single node
    s2.ops.push_back(OP_RUNLOCK(1));
    sc.steps.push_back(s2);
    return sc;
}

// Scenario 4: address reuse / OOM / free-id dropping.
static Scenario sc_reuse_oom() {
    Scenario sc; sc.name = "address_reuse_oom";
    // tiny object table to force OOM and reuse. max_free small to force drops.
    sc.spec = make_spec(2, 2, 3, 2, 8, 2, 256);
    StepHost s0;
    // alloc 3 objects (fills table): ids 1,2,3
    s0.ops.push_back(OP_ALLOC(0, 1));
    s0.ops.push_back(OP_ALLOC(1, 2));
    s0.ops.push_back(OP_ALLOC(2, 3));
    // table full, free empty -> OOM
    s0.ops.push_back(OP_ALLOC(3, 4));
    sc.steps.push_back(s0);

    StepHost s1;
    // retire all 3 (no edges, no roots) and reclaim. free list cap=2 -> one dropped.
    s1.ops.push_back(OP_RETIRE(1, 10));
    s1.ops.push_back(OP_RETIRE(2, 20));
    s1.ops.push_back(OP_RETIRE(3, 30));
    s1.ops.push_back(OP_RECLAIM(8));  // reclaim 1,2,3. free gets reclaim_seq for 1,2; 3 dropped.
    sc.steps.push_back(s1);

    StepHost s2;
    // now alloc: reuse from free list (lowest reclaim_seq, obj_id) -> id 1.
    s2.ops.push_back(OP_ALLOC(4, 111));   // reuse 1
    s2.ops.push_back(OP_ALLOC(5, 222));   // reuse 2
    s2.ops.push_back(OP_ALLOC(6, 333));   // free empty now; fresh id = obj_id_next (4)
    s2.ops.push_back(OP_ALLOC(7, 444));   // table full -> OOM
    sc.steps.push_back(s2);
    return sc;
}

// Scenario 5: retire validity (incoming edges/roots block retire), reject-full.
static Scenario sc_retire_rules() {
    Scenario sc; sc.name = "retire_rules_rejectfull";
    sc.spec = make_spec(2, 2, 16, 3, 2, 16, 256);  // max_retired=2 to force reject
    StepHost s0;
    for (int i = 0; i < 5; ++i) s0.ops.push_back(OP_ALLOC(i, i));  // 1..5
    s0.ops.push_back(OP_SET_ROOT(0, 1));
    s0.ops.push_back(OP_LINK(1, 0, 2));
    s0.ops.push_back(OP_LINK(2, 0, 3));
    sc.steps.push_back(s0);

    StepHost s1;
    // retire 1: has root incoming -> INVALID
    s1.ops.push_back(OP_RETIRE(1, 1));
    // retire 2: has incoming edge from 1 -> INVALID
    s1.ops.push_back(OP_RETIRE(2, 2));
    // retire 3: incoming from 2 -> INVALID
    s1.ops.push_back(OP_RETIRE(3, 3));
    // retire 4: no incoming -> OK (retired=1)
    s1.ops.push_back(OP_RETIRE(4, 4));
    // retire 5: no incoming -> OK (retired=2)
    s1.ops.push_back(OP_RETIRE(5, 5));
    // retire 4 again: already retired -> INVALID
    s1.ops.push_back(OP_RETIRE(4, 6));
    sc.steps.push_back(s1);

    StepHost s2;
    // unlink to free 3, retire 3: retired queue would exceed max_retired(2) -> REJECT_FULL
    s2.ops.push_back(OP_LINK(2, 0, 0));
    s2.ops.push_back(OP_RETIRE(3, 7));   // reject full
    // reclaim 1 -> frees a retired slot
    s2.ops.push_back(OP_RECLAIM(1));
    // now retire 3 succeeds
    s2.ops.push_back(OP_RETIRE(3, 8));
    sc.steps.push_back(s2);
    return sc;
}

// Scenario 6: invalid ops, force-drop, multi-thread epochs, big read fan.
static Scenario sc_adversarial() {
    Scenario sc; sc.name = "adversarial_mixed";
    sc.spec = make_spec(6, 4, 48, 4, 48, 4, 512);
    StepHost s0;
    for (int i = 0; i < 10; ++i) s0.ops.push_back(OP_ALLOC(i, i * 7 - 3));  // 1..10
    // build chains across roots
    s0.ops.push_back(OP_SET_ROOT(0, 1));
    s0.ops.push_back(OP_SET_ROOT(1, 5));
    s0.ops.push_back(OP_LINK(1, 0, 2));
    s0.ops.push_back(OP_LINK(2, 0, 3));
    s0.ops.push_back(OP_LINK(3, 0, 4));
    s0.ops.push_back(OP_LINK(5, 0, 6));
    s0.ops.push_back(OP_LINK(6, 0, 7));
    // invalid ops
    s0.ops.push_back(OP_SET_ROOT(99, 1));     // bad root id
    s0.ops.push_back(OP_SET_ROOT(0, 999));    // nonexistent obj
    s0.ops.push_back(OP_LINK(999, 0, 1));     // bad src
    s0.ops.push_back(OP_LINK(1, 99, 2));      // bad slot
    s0.ops.push_back(OP_RLOCK(99));           // bad thread
    s0.ops.push_back(OP_RUNLOCK(0));          // not locked
    s0.ops.push_back(OP_QUIESCE(99));         // bad thread
    s0.ops.push_back(OP_RCHAIN(0, 0, 5));     // not locked -> invalid
    s0.ops.push_back(mk(42));                 // unknown opcode
    sc.steps.push_back(s0);

    StepHost s1;
    // multiple readers at staggered epochs
    s1.ops.push_back(OP_RLOCK(0));            // read_seq 1
    s1.ops.push_back(OP_ADV());              // epoch 1
    s1.ops.push_back(OP_RLOCK(1));            // read_seq 2, epoch 1
    s1.ops.push_back(OP_RCHAIN(0, 0, 8));
    s1.ops.push_back(OP_RCHAIN(1, 1, 8));
    s1.ops.push_back(OP_QUIESCE(2));         // quiesce idle thread
    // retire tail nodes
    s1.ops.push_back(OP_LINK(3, 0, 0));
    s1.ops.push_back(OP_RETIRE(4, 44));      // retire_seq 1
    s1.ops.push_back(OP_LINK(6, 0, 0));
    s1.ops.push_back(OP_RETIRE(7, 77));      // retire_seq 2
    // reclaim: t0 outer=1,t1 outer=2 active. R=1: 1<1 F,2<1 F -> eligible.
    // R=2: 1<2 T -> not eligible. So obj4 reclaimed, obj7 skipped.
    s1.ops.push_back(OP_RECLAIM(8));
    s1.ops.push_back(OP_RUNLOCK(0));
    s1.ops.push_back(OP_RUNLOCK(1));
    sc.steps.push_back(s1);

    StepHost s2;
    // reclaim obj7 now, then alloc reuse + force-drop free ids (cap=4)
    s2.ops.push_back(OP_RECLAIM(8));
    for (int i = 0; i < 6; ++i) s2.ops.push_back(OP_ALLOC(300 + i, i));  // some reuse + fresh
    s2.ops.push_back(OP_DROP(2));            // drop 2 free ids
    s2.ops.push_back(OP_DROP(100));          // drop the rest
    s2.ops.push_back(OP_RECLAIM(0));         // no-op
    sc.steps.push_back(s2);
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_basic());
    v.push_back(sc_grace());
    v.push_back(sc_nested_snapshot());
    v.push_back(sc_reuse_oom());
    v.push_back(sc_retire_rules());
    v.push_back(sc_adversarial());
    return v;
}

static bool run_scenario_once(const Scenario& sc, bool verbose,
                              std::vector<StepResult>* results,
                              int* passed_steps, int* total_steps,
                              std::string* first_error) {
    RcuProblemSpec spec = sc.spec;
    if (!rcu_validate_problem_spec(&spec)) {
        if (first_error) *first_error = "invalid problem spec";
        return false;
    }

    // CRITICAL: clamp only, never fail-guard on workspace_bytes == 0.
    size_t workspace_bytes = solution_workspace_bytes(&spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    RcuOracleState oracle;
    oracle.init(spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.full_reset();

    if (results) { results->clear(); results->reserve(sc.steps.size()); }
    bool all_ok = true;

    for (size_t i = 0; i < sc.steps.size(); ++i) {
        const StepHost& step = sc.steps[i];

        // upload ops (at least 1 row).
        std::vector<RcuOp> ops_host = step.ops;
        const size_t op_count = ops_host.size();
        std::vector<RcuOp> ops_upload = ops_host;
        if (ops_upload.empty()) ops_upload.push_back(mk(-1));  // pad row, op_count stays driven below

        DeviceBuffer<RcuOp> d_ops;
        d_ops.allocate(ops_upload.size());
        d_ops.upload(ops_upload);

        GuardedDeviceBuffer<uint64_t> d_counts;
        GuardedDeviceBuffer<uint64_t> d_event_hash, d_read_hash, d_root_hash;
        GuardedDeviceBuffer<uint64_t> d_object_hash, d_thread_hash, d_free_hash;
        GuardedDeviceBuffer<uint64_t> d_scalars;
        d_counts.allocate(RCU_COUNT_TOTAL);
        d_event_hash.allocate(1); d_read_hash.allocate(1); d_root_hash.allocate(1);
        d_object_hash.allocate(1); d_thread_hash.allocate(1); d_free_hash.allocate(1);
        d_scalars.allocate(6);

        RcuRunSpec run{};
        run.abi_version = RCU_ABI_VERSION;
        run.op_count = static_cast<int32_t>(op_count);
        run.step_id = step.step_id;
        run.flags = 0;

        RcuInputs inputs{};
        inputs.ops = d_ops.ptr;

        RcuOutputs outputs{};
        outputs.counts = d_counts.ptr;
        outputs.event_hash = d_event_hash.ptr;
        outputs.read_hash = d_read_hash.ptr;
        outputs.root_hash = d_root_hash.ptr;
        outputs.object_hash = d_object_hash.ptr;
        outputs.thread_hash = d_thread_hash.ptr;
        outputs.free_hash = d_free_hash.ptr;
        outputs.state_scalars = d_scalars.ptr;

        std::string error;

        CUDA_CHECK(solution_run(state, &run, &inputs, &outputs,
                                workspace.ptr, std::max<size_t>(workspace_bytes, 1), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool ok = true;

        // input-immutability: ops buffer must be unchanged (byte-wise).
        if (ok) {
            std::vector<RcuOp> after = d_ops.download();
            if (after.size() != ops_upload.size() ||
                (after.size() > 0 &&
                 std::memcmp(after.data(), ops_upload.data(), sizeof(RcuOp) * after.size()) != 0)) {
                error = "input ops buffer modified";
                ok = false;
            }
        }

        ok = ok && d_counts.check_guards("counts", &error);
        ok = ok && d_event_hash.check_guards("event_hash", &error);
        ok = ok && d_read_hash.check_guards("read_hash", &error);
        ok = ok && d_root_hash.check_guards("root_hash", &error);
        ok = ok && d_object_hash.check_guards("object_hash", &error);
        ok = ok && d_thread_hash.check_guards("thread_hash", &error);
        ok = ok && d_free_hash.check_guards("free_hash", &error);
        ok = ok && d_scalars.check_guards("state_scalars", &error);

        const std::vector<uint64_t> h_counts = d_counts.download_data();
        const std::vector<uint64_t> h_event = d_event_hash.download_data();
        const std::vector<uint64_t> h_read = d_read_hash.download_data();
        const std::vector<uint64_t> h_root = d_root_hash.download_data();
        const std::vector<uint64_t> h_obj = d_object_hash.download_data();
        const std::vector<uint64_t> h_thr = d_thread_hash.download_data();
        const std::vector<uint64_t> h_free = d_free_hash.download_data();
        const std::vector<uint64_t> h_sc = d_scalars.download_data();

        // oracle: replay only the real ops (not the pad).
        oracle.run_ops(ops_host.data(), static_cast<int>(op_count));
        RcuExpected expected;
        oracle.snapshot(&expected);

        RcuHostOutputsView got{};
        got.counts = h_counts.data();
        got.event_hash = h_event[0];
        got.read_hash = h_read[0];
        got.root_hash = h_root[0];
        got.object_hash = h_obj[0];
        got.thread_hash = h_thr[0];
        got.free_hash = h_free[0];
        got.state_scalars = h_sc.data();

        ok = ok && rcu_check_outputs(expected, got, &error);

        StepResult result;
        for (int k = 0; k < RCU_COUNT_TOTAL; ++k) result.counts[k] = h_counts[k];
        result.event_hash = h_event[0]; result.read_hash = h_read[0]; result.root_hash = h_root[0];
        result.object_hash = h_obj[0]; result.thread_hash = h_thr[0]; result.free_hash = h_free[0];
        for (int k = 0; k < 6; ++k) result.scalars[k] = h_sc[k];

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
            std::printf("scenario %-26s step %02zu/%02zu ops=%2zu %s%s%s\n",
                        sc.name.c_str(), i, sc.steps.size(), op_count,
                        ok ? "PASS" : "FAIL", ok ? "" : "  ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(const std::vector<StepResult>& a,
                            const std::vector<StepResult>& b, std::string* error) {
    if (a.size() != b.size()) { if (error) *error = "step count mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!(a[i] == b[i])) {
            if (error) { std::ostringstream o; o << "replay mismatch at step " << i; *error = o.str(); }
            return false;
        }
    }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        std::vector<Scenario> scenarios = build_scenarios();
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
                    std::printf("scenario %-26s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-26s exact replay FAIL  %s\n", sc.name.c_str(), ce.c_str()); }
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
