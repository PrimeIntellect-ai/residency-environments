// file: test_mk_chunked_dep_counters.cu

#include "mk_chunked_dep_counters_common.h"
#include "mk_chunked_dep_counters_oracle.hpp"

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
    std::vector<MkOp> ops;
    int step_id = 0;
};

struct Scenario {
    std::string name;
    MkProblemSpec spec{};
    std::vector<StepHost> steps;
};

struct StepResult {
    uint64_t counts[MK_COUNT_TOTAL] = {0};
    uint64_t event_hash = 0, cell_hash = 0, waiter_hash = 0;
    uint64_t ready_hash = 0, pending_hash = 0;
    uint64_t scalars[6] = {0};
    bool operator==(const StepResult& o) const {
        for (int i = 0; i < MK_COUNT_TOTAL; ++i) if (counts[i] != o.counts[i]) return false;
        if (event_hash != o.event_hash || cell_hash != o.cell_hash || waiter_hash != o.waiter_hash) return false;
        if (ready_hash != o.ready_hash || pending_hash != o.pending_hash) return false;
        for (int i = 0; i < 6; ++i) if (scalars[i] != o.scalars[i]) return false;
        return true;
    }
};

// ---- op builders ----
static MkOp mk(int kind, int a = 0, int b = 0, int cc = 0, int d = 0, int e = 0,
               int f = 0, int64_t i64 = 0) {
    MkOp op;
    op.kind = kind; op.arg_a = a; op.arg_b = b; op.arg_c = cc; op.arg_d = d;
    op.arg_e = e; op.arg_f = f; op.pad = 0; op.arg_i64 = i64;
    return op;
}
static MkOp OP_DEFINE(int edge, int chunk_count)        { return mk(MK_OP_DEFINE_EDGE, edge, chunk_count); }
static MkOp OP_RESET(int edge)                          { return mk(MK_OP_RESET_EDGE, edge); }
static MkOp OP_PRODUCE(int prod, int edge, int first, int n, int64_t inc,
                       int seed, int latency)            { return mk(MK_OP_PRODUCE, prod, edge, first, n, seed, latency, inc); }
static MkOp OP_ADVANCE(int delta, int maxc)             { return mk(MK_OP_ADVANCE, delta, maxc); }
static MkOp OP_ARM(int cons, int edge, int chunk, int64_t target, int seed) {
    return mk(MK_OP_ARM_WAIT, cons, edge, chunk, 0, seed, 0, target);
}
static MkOp OP_CONSUME(int limit)                       { return mk(MK_OP_CONSUME, limit); }
static MkOp OP_CANCEL(int cons, int edge, int chunk)    { return mk(MK_OP_CANCEL_WAIT, cons, edge, chunk); }
static MkOp OP_FORCE(int edge, int chunk, int64_t amt)  { return mk(MK_OP_FORCE_COUNTER, edge, chunk, 0, 0, 0, 0, amt); }

static MkProblemSpec make_spec(int edges, int max_chunks, int max_waiters,
                               int max_ready, int max_store, int max_consumers,
                               int max_epoch, int max_ops) {
    MkProblemSpec s{};
    s.abi_version = MK_ABI_VERSION;
    s.edge_count = edges;
    s.max_chunks_per_edge = max_chunks;
    s.max_waiters = max_waiters;
    s.max_ready_entries = max_ready;
    s.max_store_events = max_store;
    s.max_consumers = max_consumers;
    s.max_epoch = max_epoch;
    s.max_ops = max_ops;
    return s;
}

// Scenario 1: basic chunked pipeline - define, produce, advance, arm, consume.
static Scenario sc_basic() {
    Scenario sc; sc.name = "basic_pipeline";
    sc.spec = make_spec(4, 8, 64, 64, 64, 64, 16, 256);
    StepHost s0;
    s0.ops.push_back(OP_DEFINE(0, 4));
    // produce chunks 0..3 with increment 5, latency 10.
    s0.ops.push_back(OP_PRODUCE(7, 0, 0, 4, 5, 1234, 10));
    sc.steps.push_back(s0);

    StepHost s1;
    // arm a waiter on chunk 0 target 5 before completion.
    s1.ops.push_back(OP_ARM(100, 0, 0, 5, 999));
    // advance to complete all 4 stores (clock 0->10).
    s1.ops.push_back(OP_ADVANCE(10, 8));
    // consume: chunk 0 should be ready (counter 5 >= target 5).
    s1.ops.push_back(OP_CONSUME(4));
    sc.steps.push_back(s1);

    StepHost s2;
    // re-produce chunk 0 (RELEASED -> STORING allowed), arm with higher target.
    s2.ops.push_back(OP_PRODUCE(8, 0, 0, 1, 7, 55, 3));
    s2.ops.push_back(OP_ARM(101, 0, 0, 12, 7));
    s2.ops.push_back(OP_ADVANCE(3, 4));   // counter goes 5+7=12 -> waiter ready
    s2.ops.push_back(OP_CONSUME(2));
    sc.steps.push_back(s2);
    return sc;
}

// Scenario 2: partial production - first busy chunk stalls, earlier stay.
static Scenario sc_partial_produce() {
    Scenario sc; sc.name = "partial_production";
    sc.spec = make_spec(2, 6, 32, 32, 64, 32, 16, 256);
    StepHost s0;
    s0.ops.push_back(OP_DEFINE(0, 5));
    // produce chunks 0..2 (latency 100, stay STORING).
    s0.ops.push_back(OP_PRODUCE(1, 0, 0, 3, 2, 10, 100));
    // produce chunks 1..3: chunk 1 is STORING -> PRODUCE_STALL_CHUNK at chunk 1,
    // chunk 1 issue stops immediately (no earlier chunks in THIS op).
    s0.ops.push_back(OP_PRODUCE(2, 0, 1, 3, 2, 11, 5));
    sc.steps.push_back(s0);

    StepHost s1;
    // produce chunks 3..4 (fresh EMPTY) ok then complete some.
    s1.ops.push_back(OP_PRODUCE(3, 0, 3, 2, 4, 12, 1));
    s1.ops.push_back(OP_ADVANCE(2, 2));   // only the 2 short-latency (due<=2) complete
    sc.steps.push_back(s1);

    StepHost s2;
    s2.ops.push_back(OP_ADVANCE(200, 100)); // complete remaining long-latency stores
    sc.steps.push_back(s2);
    return sc;
}

// Scenario 3: stale ready entries via reset and cancellation.
static Scenario sc_stale_ready() {
    Scenario sc; sc.name = "stale_ready_drops";
    sc.spec = make_spec(2, 4, 32, 32, 32, 32, 16, 256);
    StepHost s0;
    s0.ops.push_back(OP_DEFINE(0, 3));
    s0.ops.push_back(OP_PRODUCE(1, 0, 0, 3, 9, 7, 2));
    s0.ops.push_back(OP_ARM(50, 0, 0, 9, 1));   // armed before completion
    s0.ops.push_back(OP_ARM(51, 0, 1, 9, 2));
    s0.ops.push_back(OP_ADVANCE(2, 3));         // all complete, both waiters ready
    sc.steps.push_back(s0);

    StepHost s1;
    // cancel waiter 50 (it is READY in ready queue). Then consume:
    // entry for 50 is CANCELLED -> READY_STALE_DROP; entry for 51 consumes.
    s1.ops.push_back(OP_CANCEL(50, 0, 0));
    s1.ops.push_back(OP_CONSUME(4));
    sc.steps.push_back(s1);

    StepHost s2;
    // arm a waiter on chunk 2 (still READY), then redefine the edge: makes the
    // ready entry stale via epoch change; consume drops it.
    s2.ops.push_back(OP_PRODUCE(2, 0, 2, 1, 9, 8, 1));
    s2.ops.push_back(OP_ADVANCE(1, 1));
    s2.ops.push_back(OP_ARM(52, 0, 2, 9, 3));   // immediate ready (READY + counter>=target)
    // chunks now: 0 RELEASED, 1 CONSUMING? no consumed->RELEASED, 2 READY.
    // To redefine, all chunks must be EMPTY or RELEASED. chunk 2 is READY -> invalid define.
    // Instead reset is also stalled. So consume chunk2 first to release it.
    s2.ops.push_back(OP_CONSUME(1));            // consume chunk2 -> RELEASED
    // now all chunks RELEASED, redefine edge0 (epoch bumps, any leftover ready stale)
    s2.ops.push_back(OP_DEFINE(0, 3));
    s2.ops.push_back(OP_CONSUME(4));            // ready queue empty now -> no-op
    sc.steps.push_back(s2);
    return sc;
}

// Scenario 4: FORCE_COUNTER wakes a waiter (target met) while the page is NOT
// READY, so the consume attempt must drop the ready entry (READY_STALE_DROP)
// rather than consume it; then a genuine produce/complete + FORCE lets a later
// waiter consume.  Exercises the "counter reaching target is insufficient unless
// the page is READY" determinism rule.
static Scenario sc_force_requeue() {
    Scenario sc; sc.name = "force_counter_notready";
    sc.spec = make_spec(2, 4, 32, 32, 32, 32, 16, 256);
    StepHost s0;
    s0.ops.push_back(OP_DEFINE(0, 2));
    // arm waiter on chunk 0 target 10, page EMPTY.
    s0.ops.push_back(OP_ARM(70, 0, 0, 10, 1));
    // FORCE_COUNTER chunk 0 by 10 -> waiter target met, WAITER_READY emitted,
    // BUT page is EMPTY (force does not make READY).
    s0.ops.push_back(OP_FORCE(0, 0, 10));
    // consume: waiter READY in queue, but cell page != READY -> READY_STALE_DROP.
    s0.ops.push_back(OP_CONSUME(2));
    sc.steps.push_back(s0);

    StepHost s1;
    // now genuinely produce + complete chunk 0; counter becomes 10+3=13.
    s1.ops.push_back(OP_PRODUCE(1, 0, 0, 1, 3, 21, 1));
    s1.ops.push_back(OP_ARM(71, 0, 0, 20, 2));   // target 20, not yet met
    s1.ops.push_back(OP_ADVANCE(1, 1));          // counter 13, target 20 unmet -> stays
    s1.ops.push_back(OP_FORCE(0, 0, 9));         // counter 22 -> waiter ready
    // page is READY (from store complete) and counter 22>=20 -> consume succeeds.
    s1.ops.push_back(OP_CONSUME(2));
    sc.steps.push_back(s1);
    return sc;
}

// Scenario 5: reset stalls, store stale drops, re-produce over RELEASED.
static Scenario sc_reset_and_stale_store() {
    Scenario sc; sc.name = "reset_stall_stale_store";
    sc.spec = make_spec(3, 4, 32, 32, 32, 32, 8, 256);
    StepHost s0;
    s0.ops.push_back(OP_DEFINE(1, 3));
    s0.ops.push_back(OP_PRODUCE(5, 1, 0, 2, 4, 33, 50)); // long latency, stay STORING
    // reset edge while chunk STORING -> RESET_STALL.
    s0.ops.push_back(OP_RESET(1));
    sc.steps.push_back(s0);

    StepHost s1;
    // redefine edge 1 while chunk0/1 STORING -> invalid (not EMPTY/RELEASED).
    s1.ops.push_back(OP_DEFINE(1, 3));
    // advance to complete both stores.
    s1.ops.push_back(OP_ADVANCE(50, 8));
    // Now chunks READY. reset still stalls (READY pages).
    s1.ops.push_back(OP_RESET(1));
    sc.steps.push_back(s1);

    StepHost s2;
    // arm + consume both ready chunks to release them.
    s2.ops.push_back(OP_ARM(80, 1, 0, 4, 1));
    s2.ops.push_back(OP_ARM(81, 1, 1, 4, 2));
    s2.ops.push_back(OP_CONSUME(8));
    // now chunks RELEASED. reset succeeds (epoch bump, clears).
    s2.ops.push_back(OP_RESET(1));
    sc.steps.push_back(s2);

    StepHost s3;
    // re-produce chunk0, but DON'T complete; then redefine (chunk STORING) ->
    // invalid; advance partially then a stale store: re-produce same chunk to
    // bump store_seq so the OLD pending store becomes stale on completion.
    s3.ops.push_back(OP_PRODUCE(9, 1, 0, 1, 2, 5, 100));   // store_seq A, due=100
    // produce chunk0 again is busy (STORING) -> stall. Instead advance clock and
    // re-issue after release. Here trigger STORE_STALE_DROP: reset cannot (busy).
    // Force the cell to a new store by completing then re-producing then make
    // the first one stale: complete it, re-produce (new seq), then advance the
    // earlier-due none... Simpler: produce another edge cell and let a store go
    // stale through redefine. Define edge 2, produce, redefine after release.
    s3.ops.push_back(OP_ADVANCE(100, 8));   // completes store A (counter 2)
    sc.steps.push_back(s3);
    return sc;
}

// Scenario 6: adversarial mixed - invalids, multi-edge, requeue, epoch wrap,
// immediate-ready, duplicate-arm rejection, pending ordering across latencies.
static Scenario sc_adversarial() {
    Scenario sc; sc.name = "adversarial_mixed";
    sc.spec = make_spec(4, 6, 64, 64, 64, 64, 2, 512);  // small max_epoch=2 to wrap
    StepHost s0;
    // invalid ops first.
    s0.ops.push_back(OP_DEFINE(99, 2));        // edge OOB
    s0.ops.push_back(OP_DEFINE(0, 0));         // chunk_count 0
    s0.ops.push_back(OP_DEFINE(0, 99));        // chunk_count too big
    s0.ops.push_back(OP_RESET(0));             // not defined
    s0.ops.push_back(OP_PRODUCE(1, 0, 0, 1, 1, 1, 1)); // edge not defined
    s0.ops.push_back(OP_ARM(1, 0, 0, 1, 1));   // edge not defined
    s0.ops.push_back(OP_CANCEL(1, 0, 0));      // no waiter
    s0.ops.push_back(OP_FORCE(0, 0, 1));       // edge not defined
    s0.ops.push_back(mk(42));                  // unknown opcode
    // now define edges and wrap epoch (max_epoch=2 -> mod 3).
    s0.ops.push_back(OP_DEFINE(0, 4));         // epoch 1
    s0.ops.push_back(OP_DEFINE(0, 4));         // epoch 2 (all EMPTY ok)
    s0.ops.push_back(OP_DEFINE(0, 4));         // epoch 3%3=0
    s0.ops.push_back(OP_DEFINE(1, 3));         // edge1 epoch 1
    sc.steps.push_back(s0);

    StepHost s1;
    // produce on two edges with mixed latencies; check pending order.
    s1.ops.push_back(OP_PRODUCE(10, 0, 0, 3, 6, 100, 7));  // due 7
    s1.ops.push_back(OP_PRODUCE(11, 1, 0, 2, 8, 200, 3));  // due 3
    s1.ops.push_back(OP_PRODUCE(12, 0, 3, 1, 6, 101, 3));  // due 3 (tie -> store_seq)
    // arm waiters (some immediate-ready impossible since not READY yet).
    s1.ops.push_back(OP_ARM(300, 0, 0, 6, 9));
    s1.ops.push_back(OP_ARM(300, 0, 0, 6, 9));   // duplicate nonterminal -> invalid
    s1.ops.push_back(OP_ARM(301, 1, 0, 8, 9));
    s1.ops.push_back(OP_ADVANCE(3, 1));          // only 1 due store completes (lowest key)
    sc.steps.push_back(s1);

    StepHost s2;
    s2.ops.push_back(OP_ADVANCE(10, 100));       // complete the rest, wake waiters
    // immediate-ready arm on an already-READY chunk.
    s2.ops.push_back(OP_ARM(302, 0, 3, 6, 9));   // chunk3 ready, counter 6>=6 -> immediate
    s2.ops.push_back(OP_CONSUME(2));             // consume 2 valid
    s2.ops.push_back(OP_CONSUME(0));             // no-op
    s2.ops.push_back(OP_CONSUME(10));            // drain rest
    sc.steps.push_back(s2);

    StepHost s3;
    // requeue path: arm on a fresh chunk, force counter below target won't help.
    s3.ops.push_back(OP_PRODUCE(13, 1, 2, 1, 5, 77, 1));
    s3.ops.push_back(OP_ADVANCE(1, 1));          // chunk2 edge1 ready, counter 5
    s3.ops.push_back(OP_ARM(310, 1, 2, 5, 4));   // immediate ready
    // now reset edge1? has nonterminal waiter -> RESET_STALL.
    s3.ops.push_back(OP_RESET(1));
    s3.ops.push_back(OP_CONSUME(4));             // consume chunk2 -> released
    s3.ops.push_back(OP_RESET(1));               // edge1: chunk2 RELEASED, others EMPTY -> ok
    sc.steps.push_back(s3);
    return sc;
}

// Scenario 7: residual structural state - leaves queued waiters (unmet targets),
// residual ready entries, and multiple pending stores at distinct due_clocks so
// waiter_hash / ready_hash / pending_hash are all exercised non-trivially,
// including the (edge,chunk,wait_seq,consumer) waiter ordering and the
// (due,store_seq,edge,chunk) pending ordering.
static Scenario sc_residual_state() {
    Scenario sc; sc.name = "residual_structures";
    sc.spec = make_spec(3, 6, 64, 64, 64, 64, 16, 256);
    StepHost s0;
    s0.ops.push_back(OP_DEFINE(0, 4));
    s0.ops.push_back(OP_DEFINE(2, 4));
    // pending stores at staggered due_clocks across two edges (ordering check).
    s0.ops.push_back(OP_PRODUCE(10, 0, 0, 2, 3, 100, 40)); // due 40 (two chunks)
    s0.ops.push_back(OP_PRODUCE(11, 2, 0, 2, 3, 200, 15)); // due 15
    s0.ops.push_back(OP_PRODUCE(12, 0, 2, 1, 3, 101, 15)); // due 15 (tie -> store_seq, edge)
    sc.steps.push_back(s0);

    StepHost s1;
    // multiple waiters on the same cell with different targets + consumers; some
    // met (ready), some unmet (stay queued) -> exercises waiter ordering.
    s1.ops.push_back(OP_ARM(500, 0, 0, 3, 1));   // met after completion
    s1.ops.push_back(OP_ARM(501, 0, 0, 99, 2));  // unmet (stays queued)
    s1.ops.push_back(OP_ARM(502, 0, 0, 3, 3));   // met, same cell
    s1.ops.push_back(OP_ARM(400, 2, 0, 3, 4));   // different edge/chunk
    // complete only the due<=20 stores (edge2 ch0, edge0 ch2), leaving edge0 ch0/ch1 pending.
    s1.ops.push_back(OP_ADVANCE(20, 100));
    sc.steps.push_back(s1);

    StepHost s2;
    // complete edge0 ch0/ch1 -> wakes waiters 500/502 (ready), 501 stays queued.
    s2.ops.push_back(OP_ADVANCE(30, 1));   // only 1 completion (ch0), ch1 stays pending
    // Now: ready queue has 400 (from earlier) + 500,502; waiter 501 queued unmet;
    // pending still has edge0 ch1 (due 40). Leave them as residual (no consume).
    sc.steps.push_back(s2);
    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> v;
    v.push_back(sc_basic());
    v.push_back(sc_partial_produce());
    v.push_back(sc_stale_ready());
    v.push_back(sc_force_requeue());
    v.push_back(sc_reset_and_stale_store());
    v.push_back(sc_adversarial());
    v.push_back(sc_residual_state());
    return v;
}

static bool run_scenario_once(const Scenario& sc, bool verbose,
                              std::vector<StepResult>* results,
                              int* passed_steps, int* total_steps,
                              std::string* first_error) {
    MkProblemSpec spec = sc.spec;
    if (!mk_validate_problem_spec(&spec)) {
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

    MkOracleState oracle;
    oracle.init(spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.full_reset();

    if (results) { results->clear(); results->reserve(sc.steps.size()); }
    bool all_ok = true;

    for (size_t i = 0; i < sc.steps.size(); ++i) {
        const StepHost& step = sc.steps[i];

        std::vector<MkOp> ops_host = step.ops;
        const size_t op_count = ops_host.size();
        std::vector<MkOp> ops_upload = ops_host;
        if (ops_upload.empty()) ops_upload.push_back(mk(-1));

        DeviceBuffer<MkOp> d_ops;
        d_ops.allocate(ops_upload.size());
        d_ops.upload(ops_upload);

        GuardedDeviceBuffer<uint64_t> d_counts;
        GuardedDeviceBuffer<uint64_t> d_event_hash, d_cell_hash, d_waiter_hash;
        GuardedDeviceBuffer<uint64_t> d_ready_hash, d_pending_hash;
        GuardedDeviceBuffer<uint64_t> d_scalars;
        d_counts.allocate(MK_COUNT_TOTAL);
        d_event_hash.allocate(1); d_cell_hash.allocate(1); d_waiter_hash.allocate(1);
        d_ready_hash.allocate(1); d_pending_hash.allocate(1);
        d_scalars.allocate(6);

        MkRunSpec run{};
        run.abi_version = MK_ABI_VERSION;
        run.op_count = static_cast<int32_t>(op_count);
        run.step_id = step.step_id;
        run.flags = 0;

        MkInputs inputs{};
        inputs.ops = d_ops.ptr;

        MkOutputs outputs{};
        outputs.counts = d_counts.ptr;
        outputs.event_hash = d_event_hash.ptr;
        outputs.cell_hash = d_cell_hash.ptr;
        outputs.waiter_hash = d_waiter_hash.ptr;
        outputs.ready_hash = d_ready_hash.ptr;
        outputs.pending_hash = d_pending_hash.ptr;
        outputs.state_scalars = d_scalars.ptr;

        std::string error;

        CUDA_CHECK(solution_run(state, &run, &inputs, &outputs,
                                workspace.ptr, std::max<size_t>(workspace_bytes, 1), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool ok = true;

        // input-immutability: ops buffer must be unchanged (byte-wise).
        if (ok) {
            std::vector<MkOp> after = d_ops.download();
            if (after.size() != ops_upload.size() ||
                (after.size() > 0 &&
                 std::memcmp(after.data(), ops_upload.data(), sizeof(MkOp) * after.size()) != 0)) {
                error = "input ops buffer modified";
                ok = false;
            }
        }

        ok = ok && d_counts.check_guards("counts", &error);
        ok = ok && d_event_hash.check_guards("event_hash", &error);
        ok = ok && d_cell_hash.check_guards("cell_hash", &error);
        ok = ok && d_waiter_hash.check_guards("waiter_hash", &error);
        ok = ok && d_ready_hash.check_guards("ready_hash", &error);
        ok = ok && d_pending_hash.check_guards("pending_hash", &error);
        ok = ok && d_scalars.check_guards("state_scalars", &error);

        const std::vector<uint64_t> h_counts = d_counts.download_data();
        const std::vector<uint64_t> h_event = d_event_hash.download_data();
        const std::vector<uint64_t> h_cell = d_cell_hash.download_data();
        const std::vector<uint64_t> h_wait = d_waiter_hash.download_data();
        const std::vector<uint64_t> h_ready = d_ready_hash.download_data();
        const std::vector<uint64_t> h_pend = d_pending_hash.download_data();
        const std::vector<uint64_t> h_sc = d_scalars.download_data();

        oracle.run_ops(ops_host.data(), static_cast<int>(op_count));
        MkExpected expected;
        oracle.snapshot(&expected);

        MkHostOutputsView got{};
        got.counts = h_counts.data();
        got.event_hash = h_event[0];
        got.cell_hash = h_cell[0];
        got.waiter_hash = h_wait[0];
        got.ready_hash = h_ready[0];
        got.pending_hash = h_pend[0];
        got.state_scalars = h_sc.data();

        ok = ok && mk_check_outputs(expected, got, &error);

        StepResult result;
        for (int k = 0; k < MK_COUNT_TOTAL; ++k) result.counts[k] = h_counts[k];
        result.event_hash = h_event[0]; result.cell_hash = h_cell[0]; result.waiter_hash = h_wait[0];
        result.ready_hash = h_ready[0]; result.pending_hash = h_pend[0];
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
