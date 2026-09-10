// file: test_mesi_directory.cu

#include "mesi_directory_common.h"
#include "mesi_directory_oracle.hpp"

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

static constexpr uint64_t g_state = 0x51e51e51e51e5151ULL;
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
    MesiRunSpec run;
    std::vector<int32_t> op;
    std::vector<int32_t> core;
    std::vector<int32_t> line;
    std::vector<int64_t> value;
    std::vector<uint64_t> txn;
};

struct StepResult {
    std::vector<int64_t> counts;
    uint64_t coh_event_hash = 0;
    uint64_t cache_hash = 0;
    uint64_t directory_hash = 0;
    uint64_t pending_hash = 0;
    uint64_t event_seq_out = 0;
    uint64_t state_checksum = 0;
};

struct Scenario {
    std::string name;
    MesiProblemSpec spec;
    std::vector<StepHost> steps;
};

static MesiProblemSpec make_spec(int cores, int lines, int cap, int maxp, int max_batch, int max_steps) {
    MesiProblemSpec spec = {};
    spec.abi_version = MESI_ABI_VERSION;
    spec.core_count = cores;
    spec.line_count = lines;
    spec.cache_capacity_per_core = cap;
    spec.max_pending_lines = maxp;
    spec.max_batch = max_batch;
    spec.max_steps = max_steps;
    spec.flags = 0;
    if (!mesi_validate_problem_spec(&spec)) throw std::runtime_error("invalid MesiProblemSpec generated");
    return spec;
}

// A simple op-builder used by hand-crafted scenarios.
struct OpBuilder {
    std::vector<int32_t> op, core, line;
    std::vector<int64_t> value;
    std::vector<uint64_t> txn;
    void add(int o, int c, int l, int64_t v, uint64_t t) {
        op.push_back(o); core.push_back(c); line.push_back(l); value.push_back(v); txn.push_back(t);
    }
    void load(int c, int l) { add(MESI_OP_LOAD, c, l, 0, 0); }
    void store(int c, int l, int64_t v) { add(MESI_OP_STORE, c, l, v, 0); }
    void ack(int c, int l, uint64_t t) { add(MESI_OP_ACK_INV, c, l, 0, t); }
    void evict(int c, int l) { add(MESI_OP_EVICT, c, l, 0, 0); }
    void flush(int l) { add(MESI_OP_FLUSH, -1, l, 0, 0); }
};

static StepHost make_step(const MesiProblemSpec& spec, int step_id, const OpBuilder& b) {
    StepHost step;
    step.run = {};
    step.run.abi_version = MESI_ABI_VERSION;
    step.run.batch_size = (int32_t)b.op.size();
    step.run.step_id = step_id;
    if (!mesi_validate_run_spec(&step.run, &spec)) throw std::runtime_error("invalid MesiRunSpec generated");
    const size_t rows = std::max<size_t>(1, b.op.size());
    step.op.assign(rows, -1);
    step.core.assign(rows, -1);
    step.line.assign(rows, -1);
    step.value.assign(rows, 0);
    step.txn.assign(rows, 0);
    for (size_t i = 0; i < b.op.size(); ++i) {
        step.op[i] = b.op[i]; step.core[i] = b.core[i]; step.line[i] = b.line[i];
        step.value[i] = b.value[i]; step.txn[i] = b.txn[i];
    }
    return step;
}

// ---- hand-crafted adversarial scenario ----
static Scenario make_pending_writeback_scenario() {
    Scenario sc;
    sc.name = "pending_writeback_order";
    sc.spec = make_spec(4, 6, 4, 4, 32, 32);
    // txn ids are assigned in order of pending-creation; we craft acks to match.

    // Step 0: cores 0,1,2 all share line 0; core 3 stores -> pending with 3 targets.
    {
        OpBuilder b;
        b.load(0, 0); // E for core0
        b.load(1, 0); // downgrade clean -> shared {0,1}
        b.load(2, 0); // shared add -> {0,1,2}
        b.store(3, 0, 12345); // targets {0,1,2} -> txn 1
        sc.steps.push_back(make_step(sc.spec, 0, b)); // is pending now
    }
    // Step 1: loads/stores/flush on line 0 must all stall; acks proceed; line 1 free.
    {
        OpBuilder b;
        b.load(0, 0);   // stall pending
        b.store(1, 0, 999); // stall pending
        b.flush(0);     // invalid (pending)
        b.ack(0, 0, 1); // INV_ACK (core0 was clean shared) -> data supply clean
        b.ack(1, 0, 1); // INV_ACK
        sc.steps.push_back(make_step(sc.spec, 1, b));
    }
    // Step 2: last ack commits store to core 3 in M.
    {
        OpBuilder b;
        b.ack(2, 0, 1); // last ack -> STORE_COMMIT to core 3
        b.load(0, 0);   // now load hits memory? core3 owns M -> downgrade writeback
        sc.steps.push_back(make_step(sc.spec, 2, b));
    }
    // Step 3: dirty owner store-with-invalidate; ack from M target updates memory.
    {
        OpBuilder b;
        b.store(0, 1, 77);  // core0 stores line1 (no sharers) -> commit M
        b.load(1, 1);       // downgrade writeback (memory<-77), share {0,1}
        b.store(2, 1, 88);  // targets {0,1} -> pending txn 2
        sc.steps.push_back(make_step(sc.spec, 3, b));
    }
    // Step 4: ack the txn 2 targets; core0 is shared, core1 shared.
    {
        OpBuilder b;
        b.evict(0, 1);  // core0 in target set -> evict stall pending
        b.ack(0, 1, 2);
        b.ack(1, 1, 2); // commit to core2 M
        b.load(3, 1);   // downgrade writeback from core2 (value 88)
        sc.steps.push_back(make_step(sc.spec, 4, b));
    }
    // Step 5: capacity eviction interplay. cap=4. Fill core0 with 4 lines then load 5th.
    {
        OpBuilder b;
        b.load(0, 2);
        b.load(0, 3);
        b.load(0, 4);
        b.load(0, 5);  // core0 cache full (lines 1? evicted earlier) ensure 4 entries
        b.load(0, 0);  // forces capacity evict of smallest touch
        sc.steps.push_back(make_step(sc.spec, 5, b));
    }
    // Step 6: invalid ops mixed.
    {
        OpBuilder b;
        b.ack(0, 0, 999);     // no pending -> invalid
        b.load(-1, 0);        // bad core
        b.store(0, 100, 5);   // bad line
        b.flush(100);         // bad line
        b.evict(2, 3);        // maybe miss
        sc.steps.push_back(make_step(sc.spec, 6, b));
    }
    return sc;
}

// ---- randomized scenario generator ----
static StepHost make_random_step(const MesiProblemSpec& spec, int step_id, int batch,
                                 SplitMix64& rng, std::vector<uint64_t>* live_txn_for_line) {
    OpBuilder b;
    for (int i = 0; i < batch; ++i) {
        int roll = rng.uniform_int(0, 99);
        bool invalid = rng.chance_permille(90);
        int core = invalid && (i & 1) ? -1 : rng.uniform_int(0, spec.core_count - 1);
        int line = invalid && !(i & 1) ? spec.line_count + 3 : rng.uniform_int(0, spec.line_count - 1);
        if (roll < 35) {
            b.load(core, line);
        } else if (roll < 60) {
            b.store(core, line, (int64_t)rng.next_u64());
        } else if (roll < 75) {
            // ack: pick a txn id; usually wrong (exercises invalid path) but
            // occasionally guess small ids to sometimes be valid.
            uint64_t t = rng.chance_permille(500) ? (uint64_t)rng.uniform_int(1, 8) : rng.next_u64();
            b.ack(core, line, t);
        } else if (roll < 90) {
            b.evict(core, line);
        } else {
            b.flush(line);
        }
    }
    (void)live_txn_for_line;
    return make_step(spec, step_id, b);
}

static Scenario make_random_scenario(const std::string& name, MesiProblemSpec spec,
                                     uint64_t seed, int nsteps) {
    Scenario sc;
    sc.name = name;
    sc.spec = spec;
    SplitMix64 rng(seed);
    std::vector<uint64_t> live;
    for (int s = 0; s < nsteps; ++s) {
        int batch = (s % 7 == 0) ? 0 : rng.uniform_int(1, std::min(spec.max_batch, 24));
        sc.steps.push_back(make_random_step(spec, s, batch, rng, &live));
    }
    return sc;
}

// A scenario that drives many valid acks by tracking the oracle-independent
// protocol via deterministic sequences: stores that create pending, then ack
// every core in range with the predicted txn id (txn ids increase by 1 per
// committed pending). We approximate by acking ids 1..N which the oracle
// validates; mismatches simply become invalid (still checked).
static Scenario make_pending_storm_scenario() {
    Scenario sc;
    sc.name = "pending_storm";
    sc.spec = make_spec(6, 8, 3, 8, 40, 40);
    SplitMix64 rng(g_state ^ 0xABCDEF01ULL);
    uint64_t next_txn = 1;
    for (int s = 0; s < 30; ++s) {
        OpBuilder b;
        int batch = rng.uniform_int(2, 16);
        for (int i = 0; i < batch; ++i) {
            int roll = rng.uniform_int(0, 99);
            int line = rng.uniform_int(0, sc.spec.line_count - 1);
            int core = rng.uniform_int(0, sc.spec.core_count - 1);
            if (roll < 30) b.load(core, line);
            else if (roll < 55) b.store(core, line, (int64_t)rng.next_u64());
            else if (roll < 85) {
                // try acking a plausible recent txn id across cores
                uint64_t t = (next_txn > 1) ? (uint64_t)rng.uniform_int(1, (int)next_txn) : 1;
                b.ack(core, line, t);
            } else b.evict(core, line);
        }
        // Heuristic: assume up to a couple pending txns may have been created
        next_txn += 2;
        sc.steps.push_back(make_step(sc.spec, s, b));
    }
    return sc;
}

// Forces: ack from an M target (DATA_SUPPLY_DIRTY, memory updated with old dirty
// value before requester's new value becomes M-only), ack from an E target
// (DATA_SUPPLY_CLEAN), and a capacity eviction at store-commit time.
static Scenario make_dirty_supply_capacity_scenario() {
    Scenario sc;
    sc.name = "dirty_supply_capacity";
    sc.spec = make_spec(3, 5, 2, 2, 24, 16);  // cap=2 forces eviction pressure

    // Step 0: core0 gets line0 exclusive then modified (dirty owner).
    {
        OpBuilder b;
        b.load(0, 0);        // E
        b.store(0, 0, 5000); // M (dirty, owner core0)
        sc.steps.push_back(make_step(sc.spec, 0, b));
    }
    // Step 1: core1 stores line0. core0 is owner-M => target {0}; pending txn1.
    {
        OpBuilder b;
        b.store(1, 0, 6000); // targets {0}; txn1
        sc.steps.push_back(make_step(sc.spec, 1, b));
    }
    // Step 2: ack from core0 (M) supplies dirty value 5000 -> memory[0]=5000, then
    // commit to core1 in M with 6000 (memory NOT updated to 6000).
    {
        OpBuilder b;
        b.ack(0, 0, 1);  // DATA_SUPPLY_DIRTY, memory<-5000, commit core1 M=6000
        sc.steps.push_back(make_step(sc.spec, 2, b));
    }
    // Step 3: clean-supply path. core2 gets line1 E. core0 stores line1 -> target {2}.
    {
        OpBuilder b;
        b.load(2, 1);        // E for core2
        b.store(0, 1, 7000); // targets {2}; txn2
        sc.steps.push_back(make_step(sc.spec, 3, b));
    }
    // Step 4: ack from core2 (E) supplies clean; commit to core0. core0 cap=2,
    // already holds line0(M,6000)? no - core0 had line0 M from step? core0 lost
    // line0 (it acked away). core0 currently holds nothing on line0. Fill core0.
    {
        OpBuilder b;
        b.ack(2, 1, 2);   // DATA_SUPPLY_CLEAN, commit core0 M line1=7000
        b.load(0, 2);     // core0 now holds {line1, line2} (cap=2 full)
        b.load(0, 3);     // capacity evict (line1 dirty -> writeback memory)
        sc.steps.push_back(make_step(sc.spec, 4, b));
    }
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(make_pending_writeback_scenario());
    v.push_back(make_dirty_supply_capacity_scenario());
    v.push_back(make_pending_storm_scenario());
    v.push_back(make_random_scenario("rand_small", make_spec(2, 3, 2, 3, 16, 24),
                                     g_state ^ 0x111ULL, 24));
    v.push_back(make_random_scenario("rand_mid", make_spec(8, 16, 4, 16, 24, 30),
                                     g_state ^ 0x222ULL, 30));
    v.push_back(make_random_scenario("rand_wide", make_spec(16, 64, 8, 64, 24, 24),
                                     g_state ^ 0x333ULL, 24));
    v.push_back(make_random_scenario("rand_maxcores", make_spec(64, 32, 4, 32, 20, 20),
                                     g_state ^ 0x444ULL, 20));
    return v;
}

static bool check_input_unchanged(const StepHost& step, const DeviceBuffer<int32_t>& d_op,
                                  const DeviceBuffer<int32_t>& d_core, const DeviceBuffer<int32_t>& d_line,
                                  const DeviceBuffer<int64_t>& d_value, const DeviceBuffer<uint64_t>& d_txn,
                                  std::string* error) {
    if (d_op.download() != step.op) { if (error) *error = "input op modified"; return false; }
    if (d_core.download() != step.core) { if (error) *error = "input core modified"; return false; }
    if (d_line.download() != step.line) { if (error) *error = "input line modified"; return false; }
    if (d_value.download() != step.value) { if (error) *error = "input value modified"; return false; }
    if (d_txn.download() != step.txn) { if (error) *error = "input txn modified"; return false; }
    return true;
}

static bool run_one_step(const MesiProblemSpec& spec, const StepHost& step,
                        MesiOracleState* oracle, void* state, void* workspace,
                        size_t workspace_bytes, cudaStream_t stream,
                        StepResult* result, std::string* error) {
    DeviceBuffer<int32_t> d_op, d_core, d_line;
    DeviceBuffer<int64_t> d_value;
    DeviceBuffer<uint64_t> d_txn;
    d_op.allocate(step.op.size());
    d_core.allocate(step.core.size());
    d_line.allocate(step.line.size());
    d_value.allocate(step.value.size());
    d_txn.allocate(step.txn.size());
    d_op.upload(step.op);
    d_core.upload(step.core);
    d_line.upload(step.line);
    d_value.upload(step.value);
    d_txn.upload(step.txn);

    GuardedDeviceBuffer<int64_t> d_counts;
    GuardedDeviceBuffer<uint64_t> d_coh, d_cache, d_dir, d_pend, d_evseq, d_state;
    d_counts.allocate(MESI_COUNT_FIELDS);
    d_coh.allocate(1); d_cache.allocate(1); d_dir.allocate(1);
    d_pend.allocate(1); d_evseq.allocate(1); d_state.allocate(1);

    MesiInputs inputs = {};
    inputs.op = d_op.ptr; inputs.arg_core = d_core.ptr; inputs.arg_line = d_line.ptr;
    inputs.arg_value = d_value.ptr; inputs.arg_txn = d_txn.ptr;

    MesiOutputs outputs = {};
    outputs.counts = d_counts.ptr;
    outputs.coh_event_hash = d_coh.ptr;
    outputs.cache_hash = d_cache.ptr;
    outputs.directory_hash = d_dir.ptr;
    outputs.pending_hash = d_pend.ptr;
    outputs.event_seq_out = d_evseq.ptr;
    outputs.state_checksum = d_state.ptr;

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &outputs, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!check_input_unchanged(step, d_op, d_core, d_line, d_value, d_txn, error)) return false;

    if (!d_counts.check_guards("counts", error)) return false;
    if (!d_coh.check_guards("coh_event_hash", error)) return false;
    if (!d_cache.check_guards("cache_hash", error)) return false;
    if (!d_dir.check_guards("directory_hash", error)) return false;
    if (!d_pend.check_guards("pending_hash", error)) return false;
    if (!d_evseq.check_guards("event_seq_out", error)) return false;
    if (!d_state.check_guards("state_checksum", error)) return false;

    const std::vector<int64_t> h_counts = d_counts.download_data();
    const std::vector<uint64_t> h_coh = d_coh.download_data();
    const std::vector<uint64_t> h_cache = d_cache.download_data();
    const std::vector<uint64_t> h_dir = d_dir.download_data();
    const std::vector<uint64_t> h_pend = d_pend.download_data();
    const std::vector<uint64_t> h_evseq = d_evseq.download_data();
    const std::vector<uint64_t> h_state = d_state.download_data();

    MesiHostInputsView host_inputs = {};
    host_inputs.op = step.op.data();
    host_inputs.arg_core = step.core.data();
    host_inputs.arg_line = step.line.data();
    host_inputs.arg_value = step.value.data();
    host_inputs.arg_txn = step.txn.data();

    MesiExpected expected;
    oracle->step_once(step.run, host_inputs, &expected);

    MesiHostOutputsView got = {};
    got.counts = h_counts.data();
    got.coh_event_hash = h_coh.data();
    got.cache_hash = h_cache.data();
    got.directory_hash = h_dir.data();
    got.pending_hash = h_pend.data();
    got.event_seq_out = h_evseq.data();
    got.state_checksum = h_state.data();

    if (!mesi_check_all_outputs(spec, expected, got, error)) return false;

    if (result) {
        result->counts = h_counts;
        result->coh_event_hash = h_coh[0];
        result->cache_hash = h_cache[0];
        result->directory_hash = h_dir[0];
        result->pending_hash = h_pend[0];
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

    MesiOracleState oracle;
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
            std::printf("scenario %-26s step %02zu/%02zu batch=%d %s%s%s\n",
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
        if (a[i].counts != b[i].counts || a[i].coh_event_hash != b[i].coh_event_hash ||
            a[i].cache_hash != b[i].cache_hash || a[i].directory_hash != b[i].directory_hash ||
            a[i].pending_hash != b[i].pending_hash || a[i].event_seq_out != b[i].event_seq_out ||
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
                    std::printf("scenario %-26s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-26s exact replay FAIL  %s\n", sc.name.c_str(), ce.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-26s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
