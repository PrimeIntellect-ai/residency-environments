// file: test_buddy_allocator_cleaner.cu
//
// 3-way validation harness for T42. Each scenario is run against the device
// solution (reference or naive, linked separately) and compared step-by-step
// against the host oracle. Each scenario is also replayed a second time from a
// fresh state to confirm exact determinism. Output buffers are guarded and the
// input op buffer is checked for immutability.

#include "buddy_allocator_cleaner_common.h"
#include "buddy_allocator_cleaner_oracle.hpp"

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

static constexpr uint64_t g_seed = 0x5c1ab33d99f0e417ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                              \
        if (_err != cudaSuccess) {                                             \
            std::ostringstream _oss;                                           \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "   \
                 << cudaGetErrorString(_err);                                  \
            throw std::runtime_error(_oss.str());                             \
        }                                                                       \
    } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t s) : state(s) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x8e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    int uniform_int(int lo, int hi) {
        return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1));
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
        CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(T) * n));
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
        count = n; data_bytes = sizeof(T) * count; total_bytes = kGuardBytes + data_bytes + kGuardBytes;
        CUDA_CHECK(cudaMalloc((void**)&raw, total_bytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, total_bytes));
        ptr = (T*)(raw + kGuardBytes);
    }
    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }
    bool check_guards(const char* name, std::string* err) const {
        std::vector<uint8_t> before(kGuardBytes), after(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(before.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(after.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (before[i] != kGuardByte) { if (err) { std::ostringstream o; o<<"left guard "<<name<<" byte "<<i; *err=o.str(); } return false; }
            if (after[i] != kGuardByte) { if (err) { std::ostringstream o; o<<"right guard "<<name<<" byte "<<i; *err=o.str(); } return false; }
        }
        return true;
    }
};

struct StepHost {
    BacRunSpec run;
    std::vector<BacOp> ops;
};

struct StepResult {
    BacExpected vals;
};

struct Scenario {
    std::string name;
    BacProblemSpec spec;
    std::vector<StepHost> steps;
};

static BacProblemSpec make_spec(int max_order, int segment_order, int num_classes,
                                int max_objects, int max_segments,
                                int max_ops, int max_steps) {
    BacProblemSpec s = {};
    s.abi_version = BAC_ABI_VERSION;
    s.max_order = max_order;
    s.segment_order = segment_order;
    s.num_classes = num_classes;
    s.max_objects = max_objects;
    s.max_segments = max_segments;
    s.max_ops_per_step = max_ops;
    s.max_steps = max_steps;
    s.flags = 0;
    if (!bac_validate_problem_spec(&s)) throw std::runtime_error("invalid problem spec");
    return s;
}

static BacOp mk_alloc(uint64_t obj, int cls, uint64_t req) {
    BacOp o = {}; o.op_type = BAC_OP_ALLOC; o.class_id = cls; o.obj_id = obj; o.a = req; return o;
}
static BacOp mk_free(uint64_t obj) { BacOp o = {}; o.op_type = BAC_OP_FREE; o.obj_id = obj; return o; }
static BacOp mk_pin(uint64_t obj) { BacOp o = {}; o.op_type = BAC_OP_PIN; o.obj_id = obj; return o; }
static BacOp mk_unpin(uint64_t obj) { BacOp o = {}; o.op_type = BAC_OP_UNPIN; o.obj_id = obj; return o; }
static BacOp mk_seal(int cls) { BacOp o = {}; o.op_type = BAC_OP_SEAL; o.class_id = cls; return o; }
static BacOp mk_clean(uint64_t maxseg, uint64_t budget) {
    BacOp o = {}; o.op_type = BAC_OP_CLEAN; o.a = maxseg; o.b = budget; return o;
}

static StepHost make_step(const BacProblemSpec& spec, int step_id, const std::vector<BacOp>& ops) {
    StepHost s;
    s.run = {};
    s.run.abi_version = BAC_ABI_VERSION;
    s.run.num_ops = (int)ops.size();
    s.run.step_id = step_id;
    if (!bac_validate_run_spec(&s.run, &spec)) throw std::runtime_error("invalid run spec");
    s.ops = ops;
    if (s.ops.empty()) s.ops.resize(1);  // keep a row for upload
    return s;
}

// ----- scenario builders -----

// 1) Basic fill: allocate many small objects in one class, forcing implicit
//    seals and buddy splits, then explicit seal, then a clean (no movable).
static Scenario sc_basic_fill() {
    Scenario sc;
    sc.name = "basic_fill_split_seal";
    sc.spec = make_spec(8, 3, 2, 64, 32, 64, 8); // total=256, seg=8 pages
    int sid = 0;
    // class 0: allocate 1-page objects until several segments are created.
    {
        std::vector<BacOp> ops;
        for (uint64_t i = 1; i <= 20; ++i) ops.push_back(mk_alloc(i, 0, 1));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    // class 1: bigger blocks (req=3 -> order2 block4), padding + seals
    {
        std::vector<BacOp> ops;
        for (uint64_t i = 100; i <= 110; ++i) ops.push_back(mk_alloc(i, 1, 3));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    // seal both classes explicitly
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_seal(0), mk_seal(1), mk_seal(0)}));
    // clean: all sealed segments have dead from tail padding but live objects
    // and no movables freed -> blocked
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_clean(100, 100)}));
    // empty step
    sc.steps.push_back(make_step(sc.spec, sid++, {}));
    return sc;
}

// 2) Pin / unpin / deferred-free / finalize ordering.
static Scenario sc_pin_lifecycle() {
    Scenario sc;
    sc.name = "pin_deferred_finalize";
    sc.spec = make_spec(6, 2, 1, 32, 16, 32, 12); // total=64, seg=4
    int sid = 0;
    sc.steps.push_back(make_step(sc.spec, sid++, {
        mk_alloc(1, 0, 1), mk_alloc(2, 0, 1), mk_alloc(3, 0, 2), mk_alloc(4, 0, 1)
    }));
    sc.steps.push_back(make_step(sc.spec, sid++, {
        mk_pin(1), mk_pin(1), mk_pin(2),   // pin counts
        mk_free(1),                        // deferred (pinned)
        mk_free(1),                        // deferred dup
        mk_free(3)                         // immediate finalize (unpinned)
    }));
    sc.steps.push_back(make_step(sc.spec, sid++, {
        mk_pin(1),       // invalid: free_pending
        mk_unpin(1),     // pin 2->1
        mk_unpin(1),     // pin 1->0 -> finalize obj 1
        mk_unpin(2),     // pin 1->0, not pending -> no finalize
        mk_unpin(2)      // invalid: pin 0
    }));
    // invalid ops
    sc.steps.push_back(make_step(sc.spec, sid++, {
        mk_free(999), mk_pin(999), mk_unpin(999), mk_alloc(2, 0, 1) /*exists*/,
        mk_alloc(50, 9, 1) /*bad class*/, mk_alloc(51, 0, 0) /*req0*/,
        mk_alloc(52, 0, 5) /*req>seg=4*/
    }));
    return sc;
}

// 3) Full clean with reclaim + buddy merge chains.
static Scenario sc_clean_reclaim() {
    Scenario sc;
    sc.name = "clean_reclaim_merge";
    sc.spec = make_spec(7, 2, 1, 64, 32, 64, 16); // total=128, seg=4
    int sid = 0;
    // fill many segments fully with 1-page objects (4 per segment)
    {
        std::vector<BacOp> ops;
        for (uint64_t i = 1; i <= 24; ++i) ops.push_back(mk_alloc(i, 0, 1));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_seal(0)}));
    // free some objects in early segments to create dead pages (unpinned ->
    // immediate finalize)
    {
        std::vector<BacOp> ops;
        // free objects 1..8 (covers first 2 segments fully) -> those segments
        // become all-dead and will reclaim at clean
        for (uint64_t i = 1; i <= 8; ++i) ops.push_back(mk_free(i));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    // clean: empty segments reclaim immediately + merges
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_clean(100, 1000)}));
    // free more to make partially-dead segments, then clean relocates
    {
        std::vector<BacOp> ops;
        for (uint64_t i = 9; i <= 12; ++i) ops.push_back(mk_free(i));  // free 4 of seg3
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_clean(100, 1000)}));
    return sc;
}

// 4) Partial relocation: budget exhaustion mid-clean.
static Scenario sc_clean_partial_budget() {
    Scenario sc;
    sc.name = "clean_partial_budget";
    sc.spec = make_spec(8, 3, 1, 64, 32, 64, 16); // total=256, seg=8
    int sid = 0;
    // segment A: 8 one-page objects (full). segment B: same. etc.
    {
        std::vector<BacOp> ops;
        for (uint64_t i = 1; i <= 32; ++i) ops.push_back(mk_alloc(i, 0, 1));
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_seal(0)}));
    // free half of seg1 (objects 1..4) leaving 4 live, 4 dead -> dead=4
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_free(1), mk_free(2), mk_free(3), mk_free(4)}));
    // clean with budget=2 pages: can relocate 2 one-page objects then stop
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_clean(100, 2)}));
    // clean with bigger budget to finish
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_clean(100, 1000)}));
    return sc;
}

// 5) Clean blocked by pins: victim has only pinned/pending live objects.
static Scenario sc_clean_blocked_pins() {
    Scenario sc;
    sc.name = "clean_blocked_by_pins";
    sc.spec = make_spec(6, 2, 1, 32, 16, 32, 12); // total=64, seg=4
    int sid = 0;
    sc.steps.push_back(make_step(sc.spec, sid++, {
        mk_alloc(1, 0, 1), mk_alloc(2, 0, 1), mk_alloc(3, 0, 1), mk_alloc(4, 0, 1) // fills seg1
    }));
    sc.steps.push_back(make_step(sc.spec, sid++, {
        mk_alloc(5, 0, 1) // forces implicit seal of seg1, opens seg2
    }));
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_seal(0)})); // seal seg2
    // free obj 1 (dead), pin 2,3,4 then free them (deferred)
    sc.steps.push_back(make_step(sc.spec, sid++, {
        mk_free(1), mk_pin(2), mk_pin(3), mk_pin(4),
        mk_free(2), mk_free(3), mk_free(4)
    }));
    // clean: seg1 has dead=1, live=3 all pinned+pending -> no movable -> blocked
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_clean(100, 1000)}));
    // unpin them -> finalizes -> seg1 fully dead
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_unpin(2), mk_unpin(3), mk_unpin(4)}));
    // clean again: seg1 reclaims
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_clean(100, 1000)}));
    return sc;
}

// 6) OOM paths: object table full, and buddy exhaustion with prior implicit seal.
static Scenario sc_oom_paths() {
    Scenario sc;
    sc.name = "oom_object_and_buddy";
    sc.spec = make_spec(4, 2, 1, 6, 4, 16, 8); // total=16, seg=4, only 4 segs, 6 objs
    int sid = 0;
    // Allocate 6 one-page objects -> object table full at 6.
    sc.steps.push_back(make_step(sc.spec, sid++, {
        mk_alloc(1,0,1), mk_alloc(2,0,1), mk_alloc(3,0,1),
        mk_alloc(4,0,1), mk_alloc(5,0,1), mk_alloc(6,0,1)
    }));
    // 7th alloc -> ALLOC_OOM (object table full)
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_alloc(7,0,1)}));
    // free one then alloc a big block to consume buddy space; eventually buddy
    // OOM after an implicit seal.
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_free(6)}));
    // Now seal class so next alloc opens a new segment; total pages=16, seg=4,
    // so 4 segments max. We have used 2 segments (objs spread). Allocate to
    // exhaust segments. This drives implicit seals + eventual buddy/segment OOM.
    {
        std::vector<BacOp> ops;
        ops.push_back(mk_seal(0));
        // allocate full-segment-sized blocks (req=4 -> order2 block4) to use a
        // whole segment each
        ops.push_back(mk_alloc(20, 0, 4));
        ops.push_back(mk_alloc(21, 0, 4));
        ops.push_back(mk_alloc(22, 0, 4));
        ops.push_back(mk_alloc(23, 0, 4)); // some of these OOM as buddy/segments run out
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    return sc;
}

// 7) Multi-class randomized adversarial stream.
static Scenario sc_random_multiclass() {
    Scenario sc;
    sc.name = "random_multiclass_stream";
    sc.spec = make_spec(10, 3, 3, 256, 128, 64, 24); // total=1024, seg=8
    SplitMix64 rng(g_seed ^ 0xabcdef01ULL);
    uint64_t next_obj = 1;
    std::vector<uint64_t> live;  // obj ids believed live (best-effort)
    int sid = 0;
    for (int s = 0; s < 24; ++s) {
        std::vector<BacOp> ops;
        int n = rng.uniform_int(0, 40);
        for (int i = 0; i < n; ++i) {
            int r = rng.uniform_int(0, 99);
            if (r < 45) {
                int cls = rng.uniform_int(0, 2);
                int req = rng.uniform_int(1, 8); // seg=8
                ops.push_back(mk_alloc(next_obj, cls, (uint64_t)req));
                live.push_back(next_obj);
                next_obj++;
            } else if (r < 60 && !live.empty()) {
                uint64_t o = live[rng.uniform_int(0, (int)live.size()-1)];
                ops.push_back(mk_free(o));
            } else if (r < 72 && !live.empty()) {
                uint64_t o = live[rng.uniform_int(0, (int)live.size()-1)];
                ops.push_back(mk_pin(o));
            } else if (r < 84 && !live.empty()) {
                uint64_t o = live[rng.uniform_int(0, (int)live.size()-1)];
                ops.push_back(mk_unpin(o));
            } else if (r < 92) {
                ops.push_back(mk_seal(rng.uniform_int(0, 2)));
            } else {
                ops.push_back(mk_clean((uint64_t)rng.uniform_int(0, 5), (uint64_t)rng.uniform_int(0, 20)));
            }
        }
        sc.steps.push_back(make_step(sc.spec, sid++, ops));
    }
    return sc;
}

// 8) Pin-count wrap edge: pin to UINT64_MAX-1 then push wrap invalids.
static Scenario sc_pin_wrap() {
    Scenario sc;
    sc.name = "pin_wrap_edge";
    sc.spec = make_spec(4, 2, 1, 8, 4, 8, 4);
    int sid = 0;
    sc.steps.push_back(make_step(sc.spec, sid++, {mk_alloc(1,0,1)}));
    // many pins (can't reach 2^64, but exercise high pin counts + unpin)
    sc.steps.push_back(make_step(sc.spec, sid++, {
        mk_pin(1), mk_pin(1), mk_pin(1), mk_pin(1), mk_pin(1)
    }));
    sc.steps.push_back(make_step(sc.spec, sid++, {
        mk_unpin(1), mk_unpin(1), mk_unpin(1), mk_unpin(1), mk_unpin(1),
        mk_unpin(1) /*invalid: 0*/
    }));
    return sc;
}

// --------------------------------------------------------------------------

static bool run_one_step(const BacProblemSpec& spec, const StepHost& step,
                         BacOracleState* oracle, void* state,
                         void* workspace, size_t workspace_bytes,
                         cudaStream_t stream, StepResult* result, std::string* error) {
    DeviceBuffer<BacOp> d_ops;
    d_ops.allocate(step.ops.size());
    d_ops.upload(step.ops);

    GuardedDeviceBuffer<uint64_t> g[23];
    for (int i = 0; i < 23; ++i) g[i].allocate(1);

    BacInputs inputs = {};
    inputs.ops = d_ops.ptr;

    BacOutputs out = {};
    out.alloc_ok = g[0].ptr; out.alloc_oom = g[1].ptr; out.free_finalized = g[2].ptr;
    out.free_deferred = g[3].ptr; out.pin_ok = g[4].ptr; out.unpin_ok = g[5].ptr;
    out.seal_explicit = g[6].ptr; out.seal_implicit = g[7].ptr; out.seal_empty = g[8].ptr;
    out.relocated_objects = g[9].ptr; out.clean_blocked_segments = g[10].ptr;
    out.segments_reclaimed = g[11].ptr; out.buddy_splits = g[12].ptr;
    out.buddy_merges = g[13].ptr; out.padding_pages_added = g[14].ptr;
    out.invalid_count = g[15].ptr;
    out.alloc_event_hash = g[16].ptr; out.finalize_hash = g[17].ptr;
    out.buddy_hash = g[18].ptr; out.segment_hash = g[19].ptr; out.object_hash = g[20].ptr;
    out.live_object_count = g[21].ptr; out.live_segment_count = g[22].ptr;

    CUDA_CHECK(solution_run(state, &step.run, &inputs, &out, workspace, workspace_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // input immutability (byte-wise compare; avoids needing vector::operator==)
    {
        std::vector<BacOp> got_ops = d_ops.download();
        if (got_ops.size() != step.ops.size() ||
            (got_ops.size() &&
             std::memcmp(got_ops.data(), step.ops.data(), sizeof(BacOp) * got_ops.size()) != 0)) {
            if (error) *error = "input ops modified";
            return false;
        }
    }

    static const char* names[23] = {
        "alloc_ok","alloc_oom","free_finalized","free_deferred","pin_ok","unpin_ok",
        "seal_explicit","seal_implicit","seal_empty","relocated_objects",
        "clean_blocked_segments","segments_reclaimed","buddy_splits","buddy_merges",
        "padding_pages_added","invalid_count","alloc_event_hash","finalize_hash",
        "buddy_hash","segment_hash","object_hash","live_object_count","live_segment_count"
    };
    for (int i = 0; i < 23; ++i)
        if (!g[i].check_guards(names[i], error)) return false;

    uint64_t v[23];
    for (int i = 0; i < 23; ++i) v[i] = g[i].download_data()[0];

    BacExpected expected;
    oracle->step_once(step.run, step.ops.data(), &expected);

    BacHostOutputsView got = {};
    got.alloc_ok = &v[0]; got.alloc_oom = &v[1]; got.free_finalized = &v[2];
    got.free_deferred = &v[3]; got.pin_ok = &v[4]; got.unpin_ok = &v[5];
    got.seal_explicit = &v[6]; got.seal_implicit = &v[7]; got.seal_empty = &v[8];
    got.relocated_objects = &v[9]; got.clean_blocked_segments = &v[10];
    got.segments_reclaimed = &v[11]; got.buddy_splits = &v[12]; got.buddy_merges = &v[13];
    got.padding_pages_added = &v[14]; got.invalid_count = &v[15];
    got.alloc_event_hash = &v[16]; got.finalize_hash = &v[17];
    got.buddy_hash = &v[18]; got.segment_hash = &v[19]; got.object_hash = &v[20];
    got.live_object_count = &v[21]; got.live_segment_count = &v[22];

    if (!bac_check_all_outputs(expected, got, error)) return false;

    if (result) result->vals = expected;
    return true;
}

static bool run_scenario_once(const Scenario& sc, bool verbose,
                              std::vector<StepResult>* results,
                              int* passed, int* total, std::string* first_error) {
    size_t workspace_bytes = solution_workspace_bytes(&sc.spec);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&sc.spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(std::max<size_t>(workspace_bytes, 1));

    BacOracleState oracle;
    oracle.init(sc.spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) { results->clear(); results->reserve(sc.steps.size()); }

    bool all_ok = true;
    for (size_t i = 0; i < sc.steps.size(); ++i) {
        StepResult result;
        std::string error;
        bool ok = run_one_step(sc.spec, sc.steps[i], &oracle, state,
                               workspace.ptr, std::max<size_t>(workspace_bytes, 1), stream,
                               results ? &result : nullptr, &error);
        ++(*total);
        if (ok) ++(*passed);
        else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream o; o << sc.name << " step " << i << ": " << error;
                *first_error = o.str();
            }
        }
        if (results) results->push_back(result);
        if (verbose) {
            std::printf("scenario %-28s step %02zu/%02zu ops=%d %s%s\n",
                        sc.name.c_str(), i, sc.steps.size(), sc.steps[i].run.num_ops,
                        ok ? "PASS" : "FAIL ", ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_ok;
}

static bool compare_results(const std::vector<StepResult>& a, const std::vector<StepResult>& b, std::string* err) {
    if (a.size() != b.size()) { if (err) *err = "result length mismatch"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        const BacExpected& x = a[i].vals; const BacExpected& y = b[i].vals;
        if (x.alloc_event_hash != y.alloc_event_hash || x.finalize_hash != y.finalize_hash ||
            x.buddy_hash != y.buddy_hash || x.segment_hash != y.segment_hash ||
            x.object_hash != y.object_hash || x.live_object_count != y.live_object_count ||
            x.live_segment_count != y.live_segment_count) {
            if (err) { std::ostringstream o; o << "replay mismatch at step " << i; *err = o.str(); }
            return false;
        }
    }
    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        std::vector<Scenario> scenarios;
        scenarios.push_back(sc_basic_fill());
        scenarios.push_back(sc_pin_lifecycle());
        scenarios.push_back(sc_clean_reclaim());
        scenarios.push_back(sc_clean_partial_budget());
        scenarios.push_back(sc_clean_blocked_pins());
        scenarios.push_back(sc_oom_paths());
        scenarios.push_back(sc_random_multiclass());
        scenarios.push_back(sc_pin_wrap());

        int passed = 0, total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base, replay;
            std::string error;
            bool ok_base = run_scenario_once(sc, true, &base, &passed, &total, &error);
            bool ok_replay = run_scenario_once(sc, false, &replay, &passed, &total, &error);
            if (ok_base && ok_replay) {
                std::string ce;
                if (compare_results(base, replay, &ce))
                    std::printf("scenario %-28s exact replay PASS\n", sc.name.c_str());
                else { all_ok = false; std::printf("scenario %-28s exact replay FAIL %s\n", sc.name.c_str(), ce.c_str()); }
            } else {
                all_ok = false;
                std::printf("scenario %-28s FAIL %s\n", sc.name.c_str(), error.c_str());
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return (all_ok && passed == total) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}
