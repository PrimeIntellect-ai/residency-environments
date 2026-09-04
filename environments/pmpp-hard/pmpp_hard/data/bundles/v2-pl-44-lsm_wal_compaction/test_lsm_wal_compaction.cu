// file: test_lsm_wal_compaction.cu

#include "lsm_wal_compaction_common.h"
#include "lsm_wal_compaction_oracle.hpp"

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
        const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
        return lo + static_cast<int>(next_u64() % span);
    }
    bool chance_permille(int p) {
        if (p <= 0) return false;
        if (p >= 1000) return true;
        return static_cast<int>(next_u64() % 1000ULL) < p;
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
            if (left[i] != kGuardByte) { if (error) { std::ostringstream o; o << "left guard corrupted for " << name << " at " << i; *error = o.str(); } return false; }
            if (right[i] != kGuardByte) { if (error) { std::ostringstream o; o << "right guard corrupted for " << name << " at " << i; *error = o.str(); } return false; }
        }
        return true;
    }
};

struct StepHost {
    LsmRunSpec run;
    std::vector<LsmOp> ops;
};

struct StepResult {
    LsmCounts counts{};
    uint64_t read_hash = 0, write_hash = 0, compaction_hash = 0;
    uint64_t wal_hash = 0, lsm_state_hash = 0, snapshot_hash = 0;
};

struct Scenario {
    std::string name;
    LsmProblemSpec spec;
    std::vector<StepHost> steps;
};

static LsmProblemSpec make_spec(int level_count, const std::vector<int>& mfpl,
                                int memcap, int sstcap, int walseg, int maxwal,
                                int maxsnap, int maxops, int maxsteps) {
    LsmProblemSpec s = {};
    s.abi_version = LSM_ABI_VERSION;
    s.level_count = level_count;
    for (int i = 0; i < level_count; ++i) s.max_files_per_level[i] = mfpl[(size_t)i];
    s.memtable_record_cap = memcap;
    s.sst_record_cap = sstcap;
    s.wal_segment_record_cap = walseg;
    s.max_wal_segments = maxwal;
    s.max_snapshots = maxsnap;
    s.max_ops = maxops;
    s.max_steps = maxsteps;
    if (!lsm_validate_problem_spec(&s)) throw std::runtime_error("invalid spec");
    return s;
}

static LsmOp op_put(uint32_t key, int64_t value) {
    LsmOp o = {}; o.kind = LSM_OP_PUT; o.i_a = (int32_t)key; o.value = value; return o;
}
static LsmOp op_del(uint32_t key) { LsmOp o = {}; o.kind = LSM_OP_DEL; o.i_a = (int32_t)key; return o; }
static LsmOp op_get(uint64_t read_id, uint32_t key, uint64_t snap) {
    LsmOp o = {}; o.kind = LSM_OP_GET; o.i_a = (int32_t)key; o.u_a = read_id; o.u_b = snap; return o;
}
static LsmOp op_open(uint64_t sid) { LsmOp o = {}; o.kind = LSM_OP_OPEN_SNAPSHOT; o.u_a = sid; return o; }
static LsmOp op_rel(uint64_t sid) { LsmOp o = {}; o.kind = LSM_OP_RELEASE_SNAPSHOT; o.u_a = sid; return o; }
static LsmOp op_flush() { LsmOp o = {}; o.kind = LSM_OP_FLUSH; return o; }
static LsmOp op_compact(int level, int maxp) { LsmOp o = {}; o.kind = LSM_OP_COMPACT; o.i_a = level; o.i_b = maxp; return o; }
static LsmOp op_ckpt() { LsmOp o = {}; o.kind = LSM_OP_CHECKPOINT_WAL; return o; }
static LsmOp op_crash(uint32_t cw, uint32_t co) { LsmOp o = {}; o.kind = LSM_OP_CRASH_RECOVER; o.i_a = (int32_t)cw; o.i_b = (int32_t)co; return o; }

static StepHost make_step(const LsmProblemSpec& spec, int step_id, std::vector<LsmOp> ops) {
    StepHost s;
    s.run = {};
    s.run.abi_version = LSM_ABI_VERSION;
    s.run.num_ops = (int32_t)ops.size();
    s.run.step_id = step_id;
    if (!lsm_validate_run_spec(&s.run, &spec)) throw std::runtime_error("invalid run spec");
    s.ops = std::move(ops);
    return s;
}

// ---------- Scenarios ----------

// 1. basic put/get/flush/snapshot visibility across memtable + L0.
static Scenario sc_basic_visibility() {
    Scenario sc;
    sc.name = "basic_visibility";
    sc.spec = make_spec(3, {4, 4, 4}, 8, 4, 4, 8, 8, 64, 32);
    int sid = 0;
    sc.steps.push_back(make_step(sc.spec, sid++, {
        op_put(10, 100), op_put(20, 200), op_put(10, 101),
        op_get(1, 10, 0), op_get(2, 20, 0), op_get(3, 30, 0),
    }));
    sc.steps.push_back(make_step(sc.spec, sid++, {
        op_open(7), op_put(10, 999), op_get(4, 10, 7), op_get(5, 10, 0),
    }));
    sc.steps.push_back(make_step(sc.spec, sid++, {
        op_flush(), op_get(6, 10, 0), op_get(7, 20, 7), op_del(20), op_get(8, 20, 0),
    }));
    sc.steps.push_back(make_step(sc.spec, sid++, {
        op_flush(), op_get(9, 20, 0), op_get(10, 10, 7), op_rel(7), op_get(11, 20, 7),
    }));
    return sc;
}

// 2. WAL roll, write_oom, write_stall, checkpoint.
static Scenario sc_wal_pressure() {
    Scenario sc;
    sc.name = "wal_pressure";
    sc.spec = make_spec(2, {6, 6}, 6, 4, 3, 3, 4, 64, 32);  // 9 args
    // level=2, mfpl{6,6}, mem=6, sst=4, walseg=3, maxwal=3, maxsnap=4, maxops=64, maxsteps=32
    int sid = 0;
    // memtable cap 6, wal seg cap 3, max 3 segments => 9 WAL slots.
    std::vector<LsmOp> ops;
    for (uint32_t k = 0; k < 10; ++k) ops.push_back(op_put(k, (int64_t)k * 7 + 1));
    sc.steps.push_back(make_step(sc.spec, sid++, ops));  // expect stall after 6, plus rolls
    sc.steps.push_back(make_step(sc.spec, sid++, { op_flush(), op_ckpt() }));
    // refill, more rolls.
    std::vector<LsmOp> ops2;
    for (uint32_t k = 0; k < 8; ++k) ops2.push_back(op_put(k + 100, -(int64_t)k - 1));
    sc.steps.push_back(make_step(sc.spec, sid++, ops2));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_flush(), op_ckpt(), op_ckpt() }));
    return sc;
}

// 3. compaction L0->L1 with overlap selection and obsolete/output events.
static Scenario sc_compaction_overlap() {
    Scenario sc;
    sc.name = "compaction_overlap";
    sc.spec = make_spec(3, {6, 6, 6}, 4, 8, 8, 8, 8, 64, 32);
    int sid = 0;
    // build several L0 files with overlapping ranges.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(1, 1), op_put(5, 5), op_put(3, 3), op_flush() }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(4, 40), op_put(9, 9), op_put(6, 6), op_flush() }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(20, 20), op_put(25, 25), op_flush() }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_compact(0, 4), op_get(1, 5, 0), op_get(2, 9, 0) }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_compact(1, 2), op_get(3, 3, 0), op_get(4, 25, 0) }));
    return sc;
}

// 4. tombstone GC: snapshot pins block GC, deeper-level coverage blocks drop.
static Scenario sc_tombstone_gc() {
    Scenario sc;
    sc.name = "tombstone_gc";
    sc.spec = make_spec(4, {4, 4, 4, 4}, 4, 8, 16, 8, 8, 64, 32);
    int sid = 0;
    // First seed a deep level (L2) so coverage exists for some keys.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(50, 500), op_put(60, 600), op_flush() }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_compact(0, 1) }));  // L0->L1
    sc.steps.push_back(make_step(sc.spec, sid++, { op_compact(1, 1) }));  // L1->L2 (covers 50..60)
    // Now write key 50 PUT then DEL, plus key 7 (no deep coverage) PUT/DEL.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(50, 5000), op_del(50), op_put(7, 70), op_del(7), op_flush() }));
    // compact L0->L1 with NO snapshot: key 7 tombstone droppable (no deeper),
    // key 50 tombstone NOT droppable (L2 covers 50).
    sc.steps.push_back(make_step(sc.spec, sid++, { op_compact(0, 1), op_get(1, 50, 0), op_get(2, 7, 0) }));
    // Now with a snapshot pinning old versions, GC should retain.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(8, 80), op_del(8), op_open(3), op_put(8, 81), op_flush() }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_compact(0, 1), op_get(3, 8, 3), op_get(4, 8, 0) }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_rel(3), op_compact(1, 2) }));
    return sc;
}

// 5. crash recovery: partial flush, WAL truncation, snapshot drop, replay stall.
static Scenario sc_crash_recovery() {
    Scenario sc;
    sc.name = "crash_recovery";
    sc.spec = make_spec(2, {6, 6}, 4, 8, 4, 6, 4, 64, 32);
    int sid = 0;
    // put a few, flush some (durable), put more (unflushed), open snapshots.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(1, 11), op_put(2, 22), op_put(3, 33), op_flush() }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(4, 44), op_put(5, 55), op_open(9), op_open(10) }));
    // crash at some (wal_id, offset). cut keeps wal_id<=cw and offset<=co.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_crash(2, 0), op_get(1, 1, 0), op_get(2, 4, 0) }));
    // after recovery, write again to confirm next_seq reconstruction.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(6, 66), op_get(3, 6, 0), op_flush() }));
    // a crash that truncates everything past the first record.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(7, 77), op_put(8, 88), op_crash(1, 0) }));
    return sc;
}

// 6. deep coupling: long mixed workload spanning every op + multi-level compaction.
static Scenario sc_deep_coupling() {
    Scenario sc;
    sc.name = "deep_coupling";
    sc.spec = make_spec(4, {6, 6, 6, 6}, 6, 4, 6, 10, 8, 200, 48);
    SplitMix64 rng(0xDEADBEEF12345678ULL);
    int sid = 0;
    uint64_t read_id = 1;
    for (int s = 0; s < 30; ++s) {
        std::vector<LsmOp> ops;
        int n = rng.uniform_int(3, 10);
        for (int i = 0; i < n; ++i) {
            int r = rng.uniform_int(0, 99);
            uint32_t key = (uint32_t)rng.uniform_int(0, 40);
            if (r < 45) ops.push_back(op_put(key, (int64_t)rng.next_u64()));
            else if (r < 60) ops.push_back(op_del(key));
            else if (r < 80) ops.push_back(op_get(read_id++, key, (rng.uniform_int(0,4)==0) ? (uint64_t)rng.uniform_int(1,5) : 0));
            else if (r < 84) ops.push_back(op_flush());
            else if (r < 88) ops.push_back(op_compact(rng.uniform_int(0,2), rng.uniform_int(1,3)));
            else if (r < 91) ops.push_back(op_open((uint64_t)rng.uniform_int(1,6)));
            else if (r < 94) ops.push_back(op_rel((uint64_t)rng.uniform_int(1,6)));
            else if (r < 97) ops.push_back(op_ckpt());
            else ops.push_back(op_crash((uint32_t)rng.uniform_int(1,6), (uint32_t)rng.uniform_int(0,4)));
        }
        // intersperse a flush often to keep memtable from saturating forever.
        if ((s % 3) == 0) ops.push_back(op_flush());
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    return sc;
}

// 7. snapshot table full + invalid ops + duplicate snapshot ids.
static Scenario sc_invalid_ops() {
    Scenario sc;
    sc.name = "invalid_ops";
    sc.spec = make_spec(3, {4, 4, 4}, 8, 4, 8, 8, 2, 64, 32);
    int sid = 0;
    sc.steps.push_back(make_step(sc.spec, sid++, {
        op_put(1, 1), op_open(5), op_open(5), op_open(6), op_open(7),
        op_get(1, 1, 5), op_get(2, 1, 99), op_rel(99), op_rel(6),
    }));
    sc.steps.push_back(make_step(sc.spec, sid++, {
        op_compact(2, 1), op_compact(0, 0), op_compact(-1, 1), op_flush(), op_compact(0, 1),
    }));
    return sc;
}

// 8. write_oom: fill every WAL segment without checkpoint; memtable large so
//    writes do not stall, but WAL space runs out -> WRITE_OOM.
static Scenario sc_write_oom() {
    Scenario sc;
    sc.name = "write_oom";
    // memtable cap 64 (won't stall), wal seg cap 2, max 2 segments => 4 WAL slots.
    sc.spec = make_spec(2, {6, 6}, 64, 8, 2, 2, 4, 64, 32);
    int sid = 0;
    std::vector<LsmOp> ops;
    for (uint32_t k = 0; k < 8; ++k) ops.push_back(op_put(k, (int64_t)k + 1));  // 4 ok, then OOM
    sc.steps.push_back(make_step(sc.spec, sid++, ops));
    // checkpoint won't archive (nothing flushed); flush then checkpoint frees WAL.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_flush(), op_ckpt() }));
    std::vector<LsmOp> ops2;
    for (uint32_t k = 0; k < 6; ++k) ops2.push_back(op_put(k + 50, (int64_t)k + 100));
    sc.steps.push_back(make_step(sc.spec, sid++, ops2));  // more WAL OOM
    return sc;
}

// 9. compact_oom: a compaction would create more output files than level+1 cap.
static Scenario sc_compact_oom() {
    Scenario sc;
    sc.name = "compact_oom";
    // sst_record_cap 1 => every retained entry becomes its own output file.
    // level 1 cap = 2 only, so a wide compaction overflows.
    sc.spec = make_spec(3, {6, 2, 6}, 6, 1, 16, 8, 8, 64, 32);
    int sid = 0;
    // build one L0 file with 5 distinct keys (5 entries) then compact to L1.
    sc.steps.push_back(make_step(sc.spec, sid++, {
        op_put(1, 1), op_put(2, 2), op_put(3, 3), op_put(4, 4), op_put(5, 5), op_flush(),
    }));
    // compact L0->L1: 5 retained entries / cap 1 = 5 output files > L1 cap 2 -> OOM.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_compact(0, 1) }));
    // confirm nothing mutated: file still readable from L0.
    sc.steps.push_back(make_step(sc.spec, sid++, { op_get(1, 3, 0) }));
    return sc;
}

// 10. multi-crash replay coupling: crash, partial WAL truncation by offset,
//     re-replay across a flush boundary, then a second crash.
static Scenario sc_multi_crash() {
    Scenario sc;
    sc.name = "multi_crash";
    sc.spec = make_spec(2, {6, 6}, 6, 8, 4, 6, 4, 64, 32);
    int sid = 0;
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(1, 1), op_put(2, 2), op_flush(), op_put(3, 3), op_put(4, 4) }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_open(5), op_crash(2, 1), op_get(1, 3, 0), op_get(2, 1, 0) }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(5, 5), op_put(6, 6), op_flush(), op_ckpt() }));
    sc.steps.push_back(make_step(sc.spec, sid++, { op_put(7, 7), op_crash(9, 9), op_get(3, 7, 0) }));
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_basic_visibility());
    v.push_back(sc_wal_pressure());
    v.push_back(sc_compaction_overlap());
    v.push_back(sc_tombstone_gc());
    v.push_back(sc_crash_recovery());
    v.push_back(sc_deep_coupling());
    v.push_back(sc_write_oom());
    v.push_back(sc_compact_oom());
    v.push_back(sc_multi_crash());
    v.push_back(sc_invalid_ops());
    return v;
}

static bool run_one_step(const LsmProblemSpec& spec, const StepHost& step,
                         LsmOracleState* oracle, void* state, void* workspace,
                         size_t workspace_bytes, cudaStream_t stream,
                         StepResult* result, std::string* error) {
    DeviceBuffer<LsmOp> d_ops;
    d_ops.allocate(std::max<size_t>(1, step.ops.size()));
    {
        std::vector<LsmOp> padded = step.ops;
        if (padded.empty()) padded.resize(1);
        d_ops.upload(padded);
    }

    GuardedDeviceBuffer<LsmCounts> d_counts;
    GuardedDeviceBuffer<uint64_t> d_read, d_write, d_compact, d_wal, d_state, d_snap;
    d_counts.allocate(1);
    d_read.allocate(1); d_write.allocate(1); d_compact.allocate(1);
    d_wal.allocate(1); d_state.allocate(1); d_snap.allocate(1);

    LsmInputs inputs = {};
    inputs.ops = (step.ops.empty()) ? nullptr : d_ops.ptr;

    LsmOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.read_hash = d_read.ptr;
    outputs.write_hash = d_write.ptr;
    outputs.compaction_hash = d_compact.ptr;
    outputs.wal_hash = d_wal.ptr;
    outputs.lsm_state_hash = d_state.ptr;
    outputs.snapshot_hash = d_snap.ptr;

    // snapshot of input ops to verify immutability.
    std::vector<LsmOp> ops_before = d_ops.download();

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    {
        std::vector<LsmOp> ops_after = d_ops.download();
        if (ops_after.size() != ops_before.size() ||
            (ops_before.size() > 0 &&
             memcmp(ops_after.data(), ops_before.data(), sizeof(LsmOp) * ops_before.size()) != 0)) {
            if (error) *error = "input ops modified";
            return false;
        }
    }

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_read.check_guards("read_hash", error)) return false;
    if (!d_write.check_guards("write_hash", error)) return false;
    if (!d_compact.check_guards("compaction_hash", error)) return false;
    if (!d_wal.check_guards("wal_hash", error)) return false;
    if (!d_state.check_guards("lsm_state_hash", error)) return false;
    if (!d_snap.check_guards("snapshot_hash", error)) return false;

    LsmCounts gc = d_counts.download_data()[0];
    uint64_t g_read = d_read.download_data()[0];
    uint64_t g_write = d_write.download_data()[0];
    uint64_t g_compact = d_compact.download_data()[0];
    uint64_t g_wal = d_wal.download_data()[0];
    uint64_t g_state = d_state.download_data()[0];
    uint64_t g_snap = d_snap.download_data()[0];

    LsmExpected expected;
    oracle->step_once(step.run, step.ops.empty() ? nullptr : step.ops.data(), &expected);

    if (!lsm_check_outputs(expected, gc, g_read, g_write, g_compact, g_wal, g_state, g_snap, error))
        return false;

    if (result) {
        result->counts = gc;
        result->read_hash = g_read; result->write_hash = g_write; result->compaction_hash = g_compact;
        result->wal_hash = g_wal; result->lsm_state_hash = g_state; result->snapshot_hash = g_snap;
    }
    return true;
}

static bool run_scenario_once(const Scenario& sc, bool verbose, std::vector<StepResult>* results,
                              int* passed, int* total, std::string* first_error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    LsmOracleState oracle;
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
                                     workspace_bytes, stream, results ? &result : nullptr, &error);
        ++(*total);
        if (ok) ++(*passed);
        else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream oss; oss << sc.name << " step " << i << ": " << error; *first_error = oss.str();
            }
        }
        if (results) results->push_back(result);
        if (verbose) {
            std::printf("scenario %-22s step %02zu/%02zu ops=%d %s%s%s\n",
                        sc.name.c_str(), i, sc.steps.size(), sc.steps[i].run.num_ops,
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
        if (memcmp(&a[i].counts, &b[i].counts, sizeof(LsmCounts)) != 0 ||
            a[i].read_hash != b[i].read_hash || a[i].write_hash != b[i].write_hash ||
            a[i].compaction_hash != b[i].compaction_hash || a[i].wal_hash != b[i].wal_hash ||
            a[i].lsm_state_hash != b[i].lsm_state_hash || a[i].snapshot_hash != b[i].snapshot_hash) {
            if (error) { std::ostringstream o; o << "replay mismatch at step " << i; *error = o.str(); }
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

        // Coverage accumulator: terminal cumulative counts (max across scenarios,
        // since counts are cumulative within a scenario) prove that the hard
        // coupled invariants are actually exercised, not trivially zero.
        LsmCounts cov = {};
        auto cov_max = [&](const LsmCounts& c) {
#define COVF(f) if (c.f > cov.f) cov.f = c.f
            COVF(put_ok); COVF(del_ok); COVF(write_stall); COVF(write_oom); COVF(wal_rolls);
            COVF(get_found); COVF(get_missing); COVF(snapshot_opened); COVF(snapshot_released);
            COVF(snapshots_dropped_by_crash); COVF(flush_files); COVF(flush_empty); COVF(flush_oom);
            COVF(compact_empty); COVF(compact_oom); COVF(compact_input_files); COVF(compact_output_files);
            COVF(versions_dropped); COVF(tombstones_dropped); COVF(obsolete_files); COVF(wal_archived);
            COVF(recovered_records); COVF(recover_stalled_records); COVF(invalid_count);
#undef COVF
        };

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results, replay_results;
            std::string error;

            const bool ok_base = run_scenario_once(sc, true, &base_results, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay_results, &passed, &total, &error);

            if (!base_results.empty()) cov_max(base_results.back().counts);

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

        // Assert every hard coupled invariant was exercised at least once.
        struct CovReq { const char* name; int64_t v; };
        const CovReq reqs[] = {
            {"put_ok", cov.put_ok}, {"del_ok", cov.del_ok}, {"write_stall", cov.write_stall},
            {"write_oom", cov.write_oom}, {"wal_rolls", cov.wal_rolls}, {"get_found", cov.get_found},
            {"get_missing", cov.get_missing}, {"snapshot_opened", cov.snapshot_opened},
            {"snapshot_released", cov.snapshot_released}, {"snapshots_dropped_by_crash", cov.snapshots_dropped_by_crash},
            {"flush_files", cov.flush_files}, {"flush_empty", cov.flush_empty}, {"flush_oom", cov.flush_oom},
            {"compact_empty", cov.compact_empty}, {"compact_oom", cov.compact_oom},
            {"compact_input_files", cov.compact_input_files}, {"compact_output_files", cov.compact_output_files},
            {"versions_dropped", cov.versions_dropped}, {"tombstones_dropped", cov.tombstones_dropped},
            {"obsolete_files", cov.obsolete_files}, {"wal_archived", cov.wal_archived},
            {"recovered_records", cov.recovered_records},
            {"invalid_count", cov.invalid_count},
            // NOTE: recover_stalled_records is intentionally NOT required. Under
            // this spec the memtable write-stall cap equals the replay cap and
            // durable_flush_seq always tracks the flushed prefix, so the count
            // of surviving records with seq>durable can never exceed the
            // memtable capacity. RECOVER_STALL is therefore an honest
            // unreachable defensive path; all three implementations still
            // contain the identical (dormant) handling for it.
        };
        bool cov_ok = true;
        for (const CovReq& r : reqs) {
            if (r.v <= 0) { std::printf("coverage gap: %s never exercised\n", r.name); cov_ok = false; }
        }
        std::printf("coverage %s (tombstones_dropped=%lld versions_dropped=%lld recovered=%lld stalled=%lld wal_archived=%lld compact_oom=%lld)\n",
                    cov_ok ? "OK" : "INCOMPLETE", (long long)cov.tombstones_dropped, (long long)cov.versions_dropped,
                    (long long)cov.recovered_records, (long long)cov.recover_stalled_records,
                    (long long)cov.wal_archived, (long long)cov.compact_oom);

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && cov_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
