// file: test_mk_event_tensor_runtime.cu

#include "mk_event_tensor_runtime_common.h"
#include "mk_event_tensor_runtime_oracle.hpp"

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

static constexpr uint64_t g_state = 0x7a3c91e5b6d20f48ULL;
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

// ---- Host op descriptor ----
struct CellRef { uint32_t tensor, i, j; };
struct OutRef { uint32_t tensor, i, j; uint64_t amount; };

struct Op {
    int32_t kind = 0;
    uint32_t tensor = 0, ci = 0, cj = 0, rank = 0, dim0 = 0, dim1 = 0;
    uint64_t task = 0;
    uint32_t sched = 0;
    uint64_t amount = 0, mask = 0, payload = 0;
    std::vector<CellRef> deps;
    std::vector<OutRef> outs;
};

static Op ODEFINE(uint32_t t, uint32_t rank, uint32_t d0, uint32_t d1, uint64_t init) {
    Op o; o.kind = MK_OP_DEFINE_TENSOR; o.tensor = t; o.rank = rank; o.dim0 = d0; o.dim1 = d1; o.amount = init; return o;
}
static Op OREGISTER(uint64_t task, uint32_t sched, std::vector<CellRef> deps,
                    std::vector<OutRef> outs, uint64_t mask, uint64_t payload) {
    Op o; o.kind = MK_OP_REGISTER_TASK; o.task = task; o.sched = sched; o.deps = deps;
    o.outs = outs; o.mask = mask; o.payload = payload; return o;
}
static Op OSIGNAL(uint32_t t, uint32_t i, uint32_t j, uint64_t amt) {
    Op o; o.kind = MK_OP_SIGNAL_CELL; o.tensor = t; o.ci = i; o.cj = j; o.amount = amt; return o;
}
static Op OPOP(uint32_t sched, uint64_t limit) {
    Op o; o.kind = MK_OP_POP_TASKS; o.sched = sched; o.amount = limit; return o;
}
static Op OSTART(uint32_t sched, uint64_t limit) {
    Op o; o.kind = MK_OP_START_POPPED; o.sched = sched; o.amount = limit; return o;
}
static Op OCOMPLETE(uint64_t task, uint64_t observed) {
    Op o; o.kind = MK_OP_COMPLETE_TASK; o.task = task; o.mask = observed; return o;
}
static Op OCANCEL_CELL(uint32_t t, uint32_t i, uint32_t j) {
    Op o; o.kind = MK_OP_CANCEL_CELL; o.tensor = t; o.ci = i; o.cj = j; return o;
}
static Op OCANCEL_TASK(uint64_t task) {
    Op o; o.kind = MK_OP_CANCEL_TASK; o.task = task; return o;
}

struct StepHost {
    MkRunSpec run;
    std::vector<int32_t> op_kind;
    std::vector<uint32_t> tensor_id, ci, cj, rank, dim0, dim1, sched;
    std::vector<uint64_t> task_id, amount, mask, payload;
    std::vector<uint32_t> dep_off, dep_count, out_off, out_count;
    std::vector<uint32_t> dep_tensor, dep_i, dep_j, out_tensor, out_i, out_j;
    std::vector<uint64_t> out_amount;
};

struct StepResult {
    int32_t counts[18] = {0};
    uint64_t mk_event_hash = 0, cell_hash = 0, task_hash = 0, queue_hash = 0, reverse_dep_hash = 0;
};

struct Scenario {
    std::string name;
    MkProblemSpec spec;
    std::vector<StepHost> steps;
};

static MkProblemSpec make_spec(int tc, int mc, int mt, int md, int mo, int sc,
                               int mr, int mp, int mbatch, int msteps) {
    MkProblemSpec spec = {};
    spec.abi_version = MK_ABI_VERSION;
    spec.tensor_count = tc; spec.max_cells_per_tensor = mc; spec.max_tasks = mt;
    spec.max_task_deps = md; spec.max_task_outputs = mo; spec.scheduler_count = sc;
    spec.max_ready_per_scheduler = mr; spec.max_popped_per_scheduler = mp;
    spec.max_batch = mbatch; spec.max_steps = msteps; spec.flags = 0;
    if (!mk_validate_problem_spec(&spec)) throw std::runtime_error("invalid MkProblemSpec");
    return spec;
}

static StepHost make_step(const MkProblemSpec& spec, int step_id, const std::vector<Op>& ops) {
    StepHost s;
    s.run = {};
    s.run.abi_version = MK_ABI_VERSION;
    s.run.batch_size = (int32_t)ops.size();
    s.run.step_id = step_id;
    if (!mk_validate_run_spec(&s.run, &spec)) throw std::runtime_error("invalid MkRunSpec");
    const size_t rows = std::max<size_t>(1, ops.size());
    s.op_kind.assign(rows, 0); s.tensor_id.assign(rows, 0); s.ci.assign(rows, 0);
    s.cj.assign(rows, 0); s.rank.assign(rows, 0); s.dim0.assign(rows, 0);
    s.dim1.assign(rows, 0); s.sched.assign(rows, 0); s.task_id.assign(rows, 0);
    s.amount.assign(rows, 0); s.mask.assign(rows, 0); s.payload.assign(rows, 0);
    s.dep_off.assign(rows, 0); s.dep_count.assign(rows, 0);
    s.out_off.assign(rows, 0); s.out_count.assign(rows, 0);
    s.dep_tensor.clear(); s.dep_i.clear(); s.dep_j.clear();
    s.out_tensor.clear(); s.out_i.clear(); s.out_j.clear(); s.out_amount.clear();

    for (size_t r = 0; r < ops.size(); ++r) {
        const Op& o = ops[r];
        s.op_kind[r] = o.kind;
        s.tensor_id[r] = o.tensor; s.ci[r] = o.ci; s.cj[r] = o.cj;
        s.rank[r] = o.rank; s.dim0[r] = o.dim0; s.dim1[r] = o.dim1;
        s.sched[r] = o.sched; s.task_id[r] = o.task;
        s.amount[r] = o.amount; s.mask[r] = o.mask; s.payload[r] = o.payload;
        s.dep_off[r] = (uint32_t)s.dep_tensor.size();
        s.dep_count[r] = (uint32_t)o.deps.size();
        for (const CellRef& d : o.deps) {
            s.dep_tensor.push_back(d.tensor); s.dep_i.push_back(d.i); s.dep_j.push_back(d.j);
        }
        s.out_off[r] = (uint32_t)s.out_tensor.size();
        s.out_count[r] = (uint32_t)o.outs.size();
        for (const OutRef& ou : o.outs) {
            s.out_tensor.push_back(ou.tensor); s.out_i.push_back(ou.i);
            s.out_j.push_back(ou.j); s.out_amount.push_back(ou.amount);
        }
    }
    // never zero-size flat buffers (CUDA malloc of 0).
    if (s.dep_tensor.empty()) { s.dep_tensor.push_back(0); s.dep_i.push_back(0); s.dep_j.push_back(0); }
    if (s.out_tensor.empty()) { s.out_tensor.push_back(0); s.out_i.push_back(0); s.out_j.push_back(0); s.out_amount.push_back(0); }
    return s;
}

// ---- Scenarios ----

// 1: basic define / register-ready / pop / start / complete with zero-decrement
//    unblocking of a blocked successor.
static Scenario sc_basic() {
    Scenario sc; sc.name = "basic_pop_push_complete";
    sc.spec = make_spec(4, 64, 32, 8, 8, 4, 64, 64, 64, 64);
    // tensor 0: rank1 dim0=4 init=0 (already zero -> producers ready immediately).
    // tensor 1: rank1 dim0=4 init=1 (blocks consumers until decremented to 0).
    sc.steps.push_back(make_step(sc.spec, 0, {
        ODEFINE(0, 1, 4, 0, 0),
        ODEFINE(1, 1, 4, 0, 1),
        // producer task A: no deps (ready); output decrements cell (1,0) by 1.
        OREGISTER(100, 0, {}, {{1, 0, 0, 1}}, 0, 0xAA),
        // consumer task B: depends on cell (1,0) which is init=1 -> BLOCKED.
        OREGISTER(200, 0, {{1, 0, 0}}, {}, 0, 0xBB),
    }));
    sc.steps.push_back(make_step(sc.spec, 1, {
        OPOP(0, 4),     // pop producer A (ready)
        OSTART(0, 4),   // start A
        OCOMPLETE(100, 0),  // A completes -> output decrements (1,0) to zero ->
                            // CELL_ZERO -> push B ready
    }));
    sc.steps.push_back(make_step(sc.spec, 2, {
        OPOP(0, 4),     // pop B (now ready)
        OSTART(0, 4),
        OCOMPLETE(200, 0),
    }));
    return sc;
}

// 2: multi-dep AND join + repeated zero decrements must not double-push.
static Scenario sc_join() {
    Scenario sc; sc.name = "and_join_no_double_push";
    sc.spec = make_spec(4, 64, 32, 8, 8, 4, 64, 64, 64, 64);
    sc.steps.push_back(make_step(sc.spec, 0, {
        ODEFINE(0, 1, 3, 0, 2),  // cell (0,0),(0,1),(0,2) each init=2
        // join task depends on (0,0) and (0,1): needs both zero.
        OREGISTER(1, 1, {{0, 0, 0}, {0, 1, 0}}, {}, 0, 1),
    }));
    sc.steps.push_back(make_step(sc.spec, 1, {
        OSIGNAL(0, 0, 0, 2),  // (0,0) -> 0, CELL_ZERO, but (0,1) still 2 -> no push
        OSIGNAL(0, 0, 0, 5),  // already zero -> CELL_ALREADY_ZERO, no scan
        OSIGNAL(0, 1, 0, 1),  // (0,1) -> 1, not zero
        OSIGNAL(0, 1, 0, 9),  // (0,1) -> 0 first time -> push join ready
        OSIGNAL(0, 1, 0, 1),  // already zero -> ALREADY_ZERO, no double-push
    }));
    return sc;
}

// 3: dynamic-mask cancellation at completion suppresses output decrements.
static Scenario sc_dynamic() {
    Scenario sc; sc.name = "dynamic_cancel_at_complete";
    sc.spec = make_spec(4, 64, 32, 8, 8, 4, 64, 64, 64, 64);
    sc.steps.push_back(make_step(sc.spec, 0, {
        ODEFINE(0, 1, 2, 0, 5),
        // task with dynamic_mask=0b101; output decrements (0,0) by 5.
        OREGISTER(7, 0, {}, {{0, 0, 0, 5}}, 0x5, 0),
        OPOP(0, 1), OSTART(0, 1),
    }));
    sc.steps.push_back(make_step(sc.spec, 1, {
        // observed=0b001 lacks bit 2 -> (observed & mask)=1 != mask=5 -> CANCEL,
        // no OUTPUT_DECREMENT, cell (0,0) stays 5.
        OCOMPLETE(7, 0x1),
    }));
    // Now a second task that DOES satisfy its mask.
    sc.steps.push_back(make_step(sc.spec, 2, {
        OREGISTER(8, 0, {}, {{0, 0, 0, 5}}, 0x5, 0),
        OPOP(0, 1), OSTART(0, 1),
        OCOMPLETE(8, 0xF),  // observed superset of mask -> output decrements -> zero
    }));
    return sc;
}

// 4: lazy stale-drop of ready and popped queues after explicit task cancel.
static Scenario sc_stale() {
    Scenario sc; sc.name = "lazy_stale_drop";
    sc.spec = make_spec(4, 64, 32, 8, 8, 4, 64, 64, 64, 64);
    sc.steps.push_back(make_step(sc.spec, 0, {
        ODEFINE(0, 1, 4, 0, 0),  // all cells already zero -> tasks ready
        OREGISTER(1, 0, {}, {}, 0, 0),
        OREGISTER(2, 0, {}, {}, 0, 0),
        OREGISTER(3, 0, {}, {}, 0, 0),
        // cancel task 2 while it sits READY in the queue (stale).
        OCANCEL_TASK(2),
    }));
    sc.steps.push_back(make_step(sc.spec, 1, {
        // pop 3: first entry task1 valid -> pop; entry task2 stale -> READY_STALE_DROP
        // (not counted); entry task3 valid -> pop. So 2 popped, 1 stale drop.
        OPOP(0, 3),
    }));
    sc.steps.push_back(make_step(sc.spec, 2, {
        // cancel task 1 while POPPED (cell cancel rule excludes popped; explicit
        // cancel still works). Then START: task1 popped-stale-drop, task3 starts.
        OCANCEL_TASK(1),
        OSTART(0, 5),
    }));
    return sc;
}

// 5: cell cancellation cancels BLOCKED/READY dependents but not POPPED/RUNNING;
//    register-task invalid on cancelled dep cell; DEFINE_STALL on redefine.
static Scenario sc_cancel_cell() {
    Scenario sc; sc.name = "cell_cancel_and_define_stall";
    sc.spec = make_spec(4, 64, 32, 8, 8, 4, 64, 64, 64, 64);
    sc.steps.push_back(make_step(sc.spec, 0, {
        ODEFINE(0, 1, 4, 0, 1),  // cells init=1
        OREGISTER(10, 0, {{0, 0, 0}}, {}, 0, 0),  // BLOCKED on (0,0)
        OREGISTER(11, 0, {{0, 0, 0}}, {}, 0, 0),  // BLOCKED on (0,0)
        OREGISTER(12, 0, {{0, 1, 0}}, {}, 0, 0),  // BLOCKED on (0,1)
    }));
    sc.steps.push_back(make_step(sc.spec, 1, {
        // redefine tensor 0 should DEFINE_STALL: nonterminal tasks depend on it.
        ODEFINE(0, 1, 4, 0, 3),
    }));
    sc.steps.push_back(make_step(sc.spec, 2, {
        // make (0,1) zero -> task12 ready; now cancel cell (0,0) -> cancels 10,11
        // (blocked); task12 (ready, depends on (0,1)) is unaffected.
        OSIGNAL(0, 1, 0, 1),         // (0,1)->0, push task12 ready
        OCANCEL_CELL(0, 0, 0),       // cancel (0,0): cancels 10,11
        // register a task depending on cancelled (0,0) -> INVALID.
        OREGISTER(13, 0, {{0, 0, 0}}, {}, 0, 0),
    }));
    sc.steps.push_back(make_step(sc.spec, 3, {
        // pop+start task12 (running), then cancel cell (0,1): task12 is RUNNING,
        // must NOT be cancelled by cell cancel.
        OPOP(0, 4), OSTART(0, 4),
        OCANCEL_CELL(0, 1, 0),
    }));
    return sc;
}

// 6: rank-2 tensors, multi-scheduler routing, multi-output fan-out + invalids.
static Scenario sc_rank2_multi() {
    Scenario sc; sc.name = "rank2_multi_scheduler";
    sc.spec = make_spec(4, 64, 32, 8, 8, 4, 64, 64, 64, 64);
    sc.steps.push_back(make_step(sc.spec, 0, {
        ODEFINE(0, 2, 3, 4, 1),   // 3x4 rank-2 cells init=1
        ODEFINE(1, 2, 2, 2, 0),   // 2x2 already zero
        // invalids: bad rank, dim0=0, too many cells, out-of-range tensor.
        ODEFINE(2, 3, 2, 2, 0),   // INVALID rank
        ODEFINE(2, 2, 0, 2, 0),   // INVALID dim0
        ODEFINE(2, 2, 9999, 9999, 0),  // INVALID too many cells
        ODEFINE(9, 1, 2, 0, 0),   // INVALID tensor out of range
    }));
    sc.steps.push_back(make_step(sc.spec, 1, {
        // producer fans out to two cells of tensor 0 in different schedulers.
        OREGISTER(1, 1, {}, {{0, 0, 0, 1}, {0, 1, 2, 1}}, 0, 0),  // ready, sched1
        OREGISTER(2, 2, {{0, 0, 0}}, {}, 0, 0),   // blocked on (0,0,0), sched2
        OREGISTER(3, 3, {{0, 1, 2}}, {}, 0, 0),   // blocked on (0,1,2), sched3
    }));
    sc.steps.push_back(make_step(sc.spec, 2, {
        OPOP(1, 1), OSTART(1, 1),
        OCOMPLETE(1, 0),  // decrements (0,0,0) and (0,1,2) to zero -> push 2 & 3
    }));
    sc.steps.push_back(make_step(sc.spec, 3, {
        OPOP(2, 4), OPOP(3, 4),       // pop task2 from sched2, task3 from sched3
        OSTART(2, 4), OSTART(3, 4),
        OCOMPLETE(2, 0), OCOMPLETE(3, 0),
        OCOMPLETE(2, 0),  // INVALID: task2 not RUNNING (already DONE)
    }));
    return sc;
}

// 7: adversarial random mix (deterministic seed) over all op kinds.
static Scenario sc_random() {
    Scenario sc; sc.name = "random_mixed";
    sc.spec = make_spec(3, 16, 12, 4, 4, 3, 32, 32, 32, 48);
    SplitMix64 rng(g_state ^ 0xc0ffeeULL);
    // define a couple tensors first.
    sc.steps.push_back(make_step(sc.spec, 0, {
        ODEFINE(0, 1, 4, 0, 2),
        ODEFINE(1, 2, 2, 3, 1),
        ODEFINE(2, 1, 3, 0, 0),
    }));
    int next_task = 1;
    for (int s = 1; s < 44; ++s) {
        if (s % 11 == 0) { sc.steps.push_back(make_step(sc.spec, s, {})); continue; }
        std::vector<Op> ops;
        int nops = rng.uniform_int(2, 7);
        for (int k = 0; k < nops; ++k) {
            int pick = rng.uniform_int(0, 9);
            uint32_t t = (uint32_t)rng.uniform_int(0, 2);
            uint32_t i = (uint32_t)rng.uniform_int(0, 3);
            uint32_t j = (uint32_t)rng.uniform_int(0, 2);
            uint32_t sch = (uint32_t)rng.uniform_int(0, 2);
            uint64_t amt = (uint64_t)rng.uniform_int(1, 4);
            uint64_t tid = (uint64_t)(next_task - rng.uniform_int(0, 3));
            if ((int64_t)tid < 1) tid = 1;
            if (pick == 0) {
                std::vector<CellRef> deps;
                std::vector<OutRef> outs;
                int nd = rng.uniform_int(0, 2);
                for (int d = 0; d < nd; ++d)
                    deps.push_back({(uint32_t)rng.uniform_int(0, 2), (uint32_t)rng.uniform_int(0, 3), (uint32_t)rng.uniform_int(0, 2)});
                int no = rng.uniform_int(0, 2);
                for (int d = 0; d < no; ++d)
                    outs.push_back({(uint32_t)rng.uniform_int(0, 2), (uint32_t)rng.uniform_int(0, 3), (uint32_t)rng.uniform_int(0, 2), (uint64_t)rng.uniform_int(1, 3)});
                ops.push_back(OREGISTER((uint64_t)next_task, sch, deps, outs, (uint64_t)rng.uniform_int(0, 7), (uint64_t)next_task * 13));
                next_task++;
            } else if (pick == 1) ops.push_back(OSIGNAL(t, i, j, amt));
            else if (pick == 2) ops.push_back(OPOP(sch, (uint64_t)rng.uniform_int(0, 3)));
            else if (pick == 3) ops.push_back(OSTART(sch, (uint64_t)rng.uniform_int(0, 3)));
            else if (pick == 4) ops.push_back(OCOMPLETE(tid, (uint64_t)rng.uniform_int(0, 7)));
            else if (pick == 5) ops.push_back(OCANCEL_CELL(t, i, j));
            else if (pick == 6) ops.push_back(OCANCEL_TASK(tid));
            else if (pick == 7) ops.push_back(OSIGNAL(t, i, j, amt));
            else if (pick == 8) ops.push_back(OPOP(sch, (uint64_t)rng.uniform_int(0, 2)));
            else ops.push_back(OCOMPLETE(tid, (uint64_t)rng.uniform_int(0, 7)));
        }
        sc.steps.push_back(make_step(sc.spec, s, ops));
    }
    return sc;
}

static const char* kCountNames[18] = {
    "tensors_defined","define_stall","tasks_ready_register","tasks_blocked_register",
    "cell_decrements","cell_zero","cell_already_zero","tasks_ready_push",
    "tasks_popped","tasks_started","tasks_completed","dynamic_cancelled",
    "cell_cancelled","tasks_cancel_by_cell","tasks_cancel_explicit",
    "ready_stale_drop","popped_stale_drop","invalid_count"};

static bool check_input_unchanged(const StepHost& step,
    const DeviceBuffer<int32_t>& d_op,
    const DeviceBuffer<uint32_t>& d_tensor, const DeviceBuffer<uint32_t>& d_ci,
    const DeviceBuffer<uint32_t>& d_cj, const DeviceBuffer<uint32_t>& d_rank,
    const DeviceBuffer<uint32_t>& d_dim0, const DeviceBuffer<uint32_t>& d_dim1,
    const DeviceBuffer<uint64_t>& d_task, const DeviceBuffer<uint32_t>& d_sched,
    const DeviceBuffer<uint64_t>& d_amount, const DeviceBuffer<uint64_t>& d_mask,
    const DeviceBuffer<uint64_t>& d_payload, const DeviceBuffer<uint32_t>& d_doff,
    const DeviceBuffer<uint32_t>& d_dcnt, const DeviceBuffer<uint32_t>& d_ooff,
    const DeviceBuffer<uint32_t>& d_ocnt, const DeviceBuffer<uint32_t>& d_dt,
    const DeviceBuffer<uint32_t>& d_di, const DeviceBuffer<uint32_t>& d_dj,
    const DeviceBuffer<uint32_t>& d_ot, const DeviceBuffer<uint32_t>& d_oi,
    const DeviceBuffer<uint32_t>& d_oj, const DeviceBuffer<uint64_t>& d_oa,
    std::string* error) {
#define MK_IMMUT(buf, vec, label) \
    if (buf.download() != vec) { if (error) *error = "input " label " modified"; return false; }
    MK_IMMUT(d_op, step.op_kind, "op_kind");
    MK_IMMUT(d_tensor, step.tensor_id, "tensor_id");
    MK_IMMUT(d_ci, step.ci, "ci"); MK_IMMUT(d_cj, step.cj, "cj");
    MK_IMMUT(d_rank, step.rank, "rank"); MK_IMMUT(d_dim0, step.dim0, "dim0");
    MK_IMMUT(d_dim1, step.dim1, "dim1"); MK_IMMUT(d_task, step.task_id, "task_id");
    MK_IMMUT(d_sched, step.sched, "sched"); MK_IMMUT(d_amount, step.amount, "amount");
    MK_IMMUT(d_mask, step.mask, "mask"); MK_IMMUT(d_payload, step.payload, "payload");
    MK_IMMUT(d_doff, step.dep_off, "dep_off"); MK_IMMUT(d_dcnt, step.dep_count, "dep_count");
    MK_IMMUT(d_ooff, step.out_off, "out_off"); MK_IMMUT(d_ocnt, step.out_count, "out_count");
    MK_IMMUT(d_dt, step.dep_tensor, "dep_tensor"); MK_IMMUT(d_di, step.dep_i, "dep_i");
    MK_IMMUT(d_dj, step.dep_j, "dep_j"); MK_IMMUT(d_ot, step.out_tensor, "out_tensor");
    MK_IMMUT(d_oi, step.out_i, "out_i"); MK_IMMUT(d_oj, step.out_j, "out_j");
    MK_IMMUT(d_oa, step.out_amount, "out_amount");
#undef MK_IMMUT
    return true;
}

static bool run_one_step(const MkProblemSpec& spec, const StepHost& step,
    MkOracleState* oracle, void* state, void* workspace, size_t workspace_bytes,
    cudaStream_t stream, StepResult* result, std::string* error) {

    DeviceBuffer<int32_t> d_op;
    DeviceBuffer<uint32_t> d_tensor, d_ci, d_cj, d_rank, d_dim0, d_dim1, d_sched;
    DeviceBuffer<uint64_t> d_task, d_amount, d_mask, d_payload;
    DeviceBuffer<uint32_t> d_doff, d_dcnt, d_ooff, d_ocnt;
    DeviceBuffer<uint32_t> d_dt, d_di, d_dj, d_ot, d_oi, d_oj;
    DeviceBuffer<uint64_t> d_oa;

    d_op.allocate(step.op_kind.size()); d_op.upload(step.op_kind);
    d_tensor.allocate(step.tensor_id.size()); d_tensor.upload(step.tensor_id);
    d_ci.allocate(step.ci.size()); d_ci.upload(step.ci);
    d_cj.allocate(step.cj.size()); d_cj.upload(step.cj);
    d_rank.allocate(step.rank.size()); d_rank.upload(step.rank);
    d_dim0.allocate(step.dim0.size()); d_dim0.upload(step.dim0);
    d_dim1.allocate(step.dim1.size()); d_dim1.upload(step.dim1);
    d_sched.allocate(step.sched.size()); d_sched.upload(step.sched);
    d_task.allocate(step.task_id.size()); d_task.upload(step.task_id);
    d_amount.allocate(step.amount.size()); d_amount.upload(step.amount);
    d_mask.allocate(step.mask.size()); d_mask.upload(step.mask);
    d_payload.allocate(step.payload.size()); d_payload.upload(step.payload);
    d_doff.allocate(step.dep_off.size()); d_doff.upload(step.dep_off);
    d_dcnt.allocate(step.dep_count.size()); d_dcnt.upload(step.dep_count);
    d_ooff.allocate(step.out_off.size()); d_ooff.upload(step.out_off);
    d_ocnt.allocate(step.out_count.size()); d_ocnt.upload(step.out_count);
    d_dt.allocate(step.dep_tensor.size()); d_dt.upload(step.dep_tensor);
    d_di.allocate(step.dep_i.size()); d_di.upload(step.dep_i);
    d_dj.allocate(step.dep_j.size()); d_dj.upload(step.dep_j);
    d_ot.allocate(step.out_tensor.size()); d_ot.upload(step.out_tensor);
    d_oi.allocate(step.out_i.size()); d_oi.upload(step.out_i);
    d_oj.allocate(step.out_j.size()); d_oj.upload(step.out_j);
    d_oa.allocate(step.out_amount.size()); d_oa.upload(step.out_amount);

    GuardedDeviceBuffer<int32_t> g_counts[18];
    for (int i = 0; i < 18; ++i) g_counts[i].allocate(1);
    GuardedDeviceBuffer<uint64_t> g_evt, g_cell, g_task, g_queue, g_rdep;
    g_evt.allocate(1); g_cell.allocate(1); g_task.allocate(1); g_queue.allocate(1); g_rdep.allocate(1);

    MkInputs inputs = {};
    inputs.op_kind = d_op.ptr; inputs.tensor_id = d_tensor.ptr; inputs.ci = d_ci.ptr;
    inputs.cj = d_cj.ptr; inputs.rank = d_rank.ptr; inputs.dim0 = d_dim0.ptr;
    inputs.dim1 = d_dim1.ptr; inputs.task_id = d_task.ptr; inputs.sched = d_sched.ptr;
    inputs.amount = d_amount.ptr; inputs.mask = d_mask.ptr; inputs.payload = d_payload.ptr;
    inputs.dep_off = d_doff.ptr; inputs.dep_count = d_dcnt.ptr;
    inputs.out_off = d_ooff.ptr; inputs.out_count = d_ocnt.ptr;
    inputs.dep_tensor = d_dt.ptr; inputs.dep_i = d_di.ptr; inputs.dep_j = d_dj.ptr;
    inputs.out_tensor = d_ot.ptr; inputs.out_i = d_oi.ptr; inputs.out_j = d_oj.ptr;
    inputs.out_amount = d_oa.ptr;

    MkOutputs outputs = {};
    int32_t** ocount = (int32_t**)&outputs;  // first 18 fields are int32_t*
    for (int i = 0; i < 18; ++i) ocount[i] = g_counts[i].ptr;
    outputs.mk_event_hash = g_evt.ptr; outputs.cell_hash = g_cell.ptr;
    outputs.task_hash = g_task.ptr; outputs.queue_hash = g_queue.ptr;
    outputs.reverse_dep_hash = g_rdep.ptr;

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(step, d_op, d_tensor, d_ci, d_cj, d_rank, d_dim0,
            d_dim1, d_task, d_sched, d_amount, d_mask, d_payload, d_doff, d_dcnt,
            d_ooff, d_ocnt, d_dt, d_di, d_dj, d_ot, d_oi, d_oj, d_oa, error))
        return false;

    for (int i = 0; i < 18; ++i) if (!g_counts[i].check_guards(kCountNames[i], error)) return false;
    if (!g_evt.check_guards("mk_event_hash", error)) return false;
    if (!g_cell.check_guards("cell_hash", error)) return false;
    if (!g_task.check_guards("task_hash", error)) return false;
    if (!g_queue.check_guards("queue_hash", error)) return false;
    if (!g_rdep.check_guards("reverse_dep_hash", error)) return false;

    std::vector<int32_t> hc[18];
    for (int i = 0; i < 18; ++i) hc[i] = g_counts[i].download_data();
    std::vector<uint64_t> h_evt = g_evt.download_data(), h_cell = g_cell.download_data(),
        h_task = g_task.download_data(), h_queue = g_queue.download_data(),
        h_rdep = g_rdep.download_data();

    MkHostInputsView hi = {};
    hi.op_kind = step.op_kind.data(); hi.tensor_id = step.tensor_id.data();
    hi.ci = step.ci.data(); hi.cj = step.cj.data(); hi.rank = step.rank.data();
    hi.dim0 = step.dim0.data(); hi.dim1 = step.dim1.data(); hi.task_id = step.task_id.data();
    hi.sched = step.sched.data(); hi.amount = step.amount.data(); hi.mask = step.mask.data();
    hi.payload = step.payload.data(); hi.dep_off = step.dep_off.data();
    hi.dep_count = step.dep_count.data(); hi.out_off = step.out_off.data();
    hi.out_count = step.out_count.data(); hi.dep_tensor = step.dep_tensor.data();
    hi.dep_i = step.dep_i.data(); hi.dep_j = step.dep_j.data();
    hi.out_tensor = step.out_tensor.data(); hi.out_i = step.out_i.data();
    hi.out_j = step.out_j.data(); hi.out_amount = step.out_amount.data();

    MkExpected expected;
    oracle->step_once(step.run, hi, &expected);

    MkHostOutputsView got = {};
    const int32_t** gcount = (const int32_t**)&got;
    for (int i = 0; i < 18; ++i) gcount[i] = hc[i].data();
    got.mk_event_hash = h_evt.data(); got.cell_hash = h_cell.data();
    got.task_hash = h_task.data(); got.queue_hash = h_queue.data();
    got.reverse_dep_hash = h_rdep.data();

    if (!mk_check_all_outputs(expected, got, error)) return false;

    if (result) {
        for (int i = 0; i < 18; ++i) result->counts[i] = hc[i][0];
        result->mk_event_hash = h_evt[0]; result->cell_hash = h_cell[0];
        result->task_hash = h_task[0]; result->queue_hash = h_queue[0];
        result->reverse_dep_hash = h_rdep[0];
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

    MkOracleState oracle;
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
            std::printf("scenario %-32s step %02zu/%02zu batch=%d %s%s%s\n",
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
        for (int k = 0; k < 18; ++k) if (a[i].counts[k] != b[i].counts[k]) eq = false;
        if (a[i].mk_event_hash != b[i].mk_event_hash || a[i].cell_hash != b[i].cell_hash ||
            a[i].task_hash != b[i].task_hash || a[i].queue_hash != b[i].queue_hash ||
            a[i].reverse_dep_hash != b[i].reverse_dep_hash) eq = false;
        if (!eq) { if (error) { std::ostringstream o; o << "replay mismatch at step " << i; *error = o.str(); } return false; }
    }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        std::vector<Scenario> scenarios;
        scenarios.push_back(sc_basic());
        scenarios.push_back(sc_join());
        scenarios.push_back(sc_dynamic());
        scenarios.push_back(sc_stale());
        scenarios.push_back(sc_cancel_cell());
        scenarios.push_back(sc_rank2_multi());
        scenarios.push_back(sc_random());

        int passed = 0, total = 0;
        bool all_ok = true;
        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results, replay_results;
            std::string error;
            const bool ok_base = run_scenario_once(sc, true, &base_results, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay_results, &passed, &total, &error);
            if (ok_base && ok_replay) {
                std::string cmp_err;
                if (compare_results(base_results, replay_results, &cmp_err))
                    std::printf("scenario %-32s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-32s exact replay FAIL  %s\n", sc.name.c_str(), cmp_err.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-32s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }
        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
